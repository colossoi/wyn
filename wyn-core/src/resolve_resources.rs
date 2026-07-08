//! Resolve `#[view(resource, usage)]` params against top-level `resource`
//! declarations.
//!
//! Runs after name resolution and before type checking. For each view it
//! derives the backing resource's `(set, binding)` (current frame, and — for a
//! `history` resource — the previous frame) and rewrites the `View` attribute
//! into the concrete `StorageImage` / `Texture` binding attribute. A `previous`
//! view also records a `FeedbackPair` on its entry, which flows to the pipeline
//! descriptor so the runtime double-buffers it. After this pass no `View`
//! attributes or `resource` declarations affect later stages — the program
//! looks exactly as if the bindings had been written inline.

use crate::ast::{self, Declaration, Pattern, PatternKind};
use crate::error::Result;
use crate::interface::{Attribute, FeedbackPair, ResourceDecl, ResourceUsage, StorageAccess};
use crate::types::{Type, TypeName};
use crate::{bail_type_at, BindingRef, LookupMap, LookupSet};

/// Default descriptor set for auto-assigned resource bindings. Set 0 is
/// compiler-reserved; user resources live on set 1+.
const DEFAULT_RESOURCE_SET: u32 = 1;

/// A resource's derived bindings — one distinct descriptor slot per *view
/// kind*, since a storage-write view and a sampled view are different
/// descriptor types and must not share a `(set, binding)`. All slots name
/// views of the single backing texture allocation (`current_storage`).
struct ResolvedResource {
    decl: ResourceDecl,
    /// Write/read storage-image view of the current frame. Present iff the
    /// resource declares a `storage_write`/`storage_read` usage. This is the
    /// allocation key the sampled views are `backing`ed by.
    current_storage: Option<BindingRef>,
    /// Sampled view of the current frame. Present iff `sampled` is declared.
    current_sampled: Option<BindingRef>,
    /// Sampled view of the previous frame, present iff `decl.history >= 1`
    /// and `sampled` is declared.
    previous_sampled: Option<BindingRef>,
}

pub fn run(program: &mut ast::Program) -> Result<()> {
    let decls: Vec<ResourceDecl> = program
        .declarations
        .iter()
        .filter_map(|d| match d {
            Declaration::Resource(r) => Some(r.clone()),
            _ => None,
        })
        .collect();
    if decls.is_empty() && !any_view(program) {
        return Ok(());
    }

    let table = derive_bindings(&decls, program)?;

    for decl in &mut program.declarations {
        let Declaration::Entry(entry) = decl else {
            continue;
        };
        let mut feedback: Vec<FeedbackPair> = Vec::new();
        for param in &mut entry.params {
            rewrite_view_param(param, &table, &mut feedback)?;
        }
        entry.feedback.extend(feedback);
    }
    Ok(())
}

/// Assign each resource its current (and, for history resources, previous)
/// binding: honor pins, then auto-assign the rest to free slots on the default
/// set, avoiding slots already taken by explicit param attributes or pins.
fn derive_bindings(
    decls: &[ResourceDecl],
    program: &ast::Program,
) -> Result<LookupMap<String, ResolvedResource>> {
    let mut used: LookupSet<(u32, u32)> = collect_explicit_slots(program);
    for r in decls {
        if let Some(b) = r.layout {
            used.insert((b.set, b.binding));
        }
    }

    // Next free binding on a given set.
    let auto_next = |used: &mut LookupSet<(u32, u32)>, set: u32| -> BindingRef {
        let mut b = 0u32;
        while used.contains(&(set, b)) {
            b += 1;
        }
        used.insert((set, b));
        BindingRef::new(set, b)
    };

    let has = |r: &ResourceDecl, u: ResourceUsage| r.usages.contains(&u);

    let mut table: LookupMap<String, ResolvedResource> = LookupMap::new();
    let mut pinned: LookupMap<(u32, u32), String> = LookupMap::new();
    for r in decls {
        if table.contains_key(&r.name) {
            bail_type_at!(r.span, "duplicate resource '{}'", r.name);
        }
        let wants_storage = has(r, ResourceUsage::StorageWrite) || has(r, ResourceUsage::StorageRead);
        let wants_sampled = has(r, ResourceUsage::Sampled);

        // One distinct slot per view kind, assigned storage → sampled →
        // previous so the pin (if any) lands on the storage allocation and
        // the views stay grouped. A `layout =` pin applies to the primary
        // slot (storage if present, else sampled).
        let pin_set = r.layout.map(|b| b.set).unwrap_or(DEFAULT_RESOURCE_SET);
        let next = |used: &mut LookupSet<(u32, u32)>| auto_next(used, pin_set);

        let current_storage =
            if wants_storage { Some(r.layout.unwrap_or_else(|| next(&mut used))) } else { None };
        let current_sampled = if wants_sampled {
            // If sampled is the primary (no storage) view, a pin lands here.
            Some(match (current_storage, r.layout) {
                (None, Some(pin)) => pin,
                _ => next(&mut used),
            })
        } else {
            None
        };
        let previous_sampled = (r.history >= 1 && wants_sampled).then(|| next(&mut used));

        // Two distinct resources must not pin the same primary slot.
        if let Some(pin) = r.layout {
            if let Some(prev) = pinned.insert((pin.set, pin.binding), r.name.clone()) {
                bail_type_at!(
                    r.span,
                    "resources '{}' and '{}' both pin (set={}, binding={})",
                    r.name,
                    prev,
                    pin.set,
                    pin.binding
                );
            }
        }
        table.insert(
            r.name.clone(),
            ResolvedResource {
                decl: r.clone(),
                current_storage,
                current_sampled,
                previous_sampled,
            },
        );
    }
    Ok(table)
}

/// Rewrite a single param's `View` attribute, if any, into a concrete binding
/// attribute. Pushes a `FeedbackPair` when the view is `previous`.
fn rewrite_view_param(
    param: &mut Pattern,
    table: &LookupMap<String, ResolvedResource>,
    feedback: &mut Vec<FeedbackPair>,
) -> Result<()> {
    if !param_attrs(param).iter().any(|a| matches!(a, Attribute::View { .. })) {
        return Ok(());
    }
    let span = param.h.span;
    let handle = param.pattern_type().and_then(type_name_of);
    let attrs = param_attrs_mut(param)
        .ok_or_else(|| crate::err_type_at!(span, "view attribute on a param without an attribute list"))?;
    for attr in attrs.iter_mut() {
        let Attribute::View {
            resource,
            usage,
            previous,
        } = attr
        else {
            continue;
        };
        let res = table
            .get(resource)
            .ok_or_else(|| crate::err_type_at!(span, "unknown resource '{}' in view", resource))?;
        if !res.decl.usages.contains(usage) {
            bail_type_at!(span, "resource '{}' does not declare usage {:?}", resource, usage);
        }
        // Lower each view to its own descriptor slot: storage and sampled
        // views are distinct descriptor types of the same backing texture,
        // never the same `(set, binding)`. Sampled views carry `backing` —
        // the storage allocation they view — so the runtime aliases one
        // allocation across both; a `previous` view also records the
        // ping-pong feedback pair.
        *attr = match usage {
            ResourceUsage::StorageWrite | ResourceUsage::StorageRead => {
                if handle != Some(TypeName::StorageTexture) {
                    bail_type_at!(
                        span,
                        "view usage {:?} of '{}' requires a `storage_image` param",
                        usage,
                        resource
                    );
                }
                let binding = res.current_storage.expect("storage usage implies a storage slot");
                let access = if matches!(usage, ResourceUsage::StorageWrite) {
                    StorageAccess::WriteOnly
                } else {
                    StorageAccess::ReadOnly
                };
                Attribute::StorageImage {
                    set: binding.set,
                    binding: binding.binding,
                    format: res.decl.format,
                    access,
                    size: res.decl.size,
                }
            }
            ResourceUsage::Sampled => {
                if handle != Some(TypeName::Texture2D) {
                    bail_type_at!(
                        span,
                        "view usage Sampled of '{}' requires a `texture2d` param",
                        resource
                    );
                }
                let binding = if *previous {
                    let prev = res.previous_sampled.ok_or_else(|| {
                        crate::err_type_at!(
                            span,
                            "view of '{}' uses `previous`, but the resource has no `history`",
                            resource
                        )
                    })?;
                    if let Some(write) = res.current_storage {
                        feedback.push(FeedbackPair { read: prev, write });
                    }
                    prev
                } else {
                    res.current_sampled.expect("sampled usage implies a sampled slot")
                };
                Attribute::Texture {
                    set: binding.set,
                    binding: binding.binding,
                    backing: res.current_storage,
                    // A resource with no storage allocation is a render target;
                    // carry its name as the frame-graph identity so a fragment
                    // `#[target(name)]` write and this sampled read coalesce.
                    resource: res.current_storage.is_none().then(|| resource.clone()),
                }
            }
        };
    }
    Ok(())
}

/// Every `(set, binding)` already claimed by an explicit binding attribute on
/// any entry param — so auto-assigned resources don't collide with them.
fn collect_explicit_slots(program: &ast::Program) -> LookupSet<(u32, u32)> {
    let mut used = LookupSet::new();
    for decl in &program.declarations {
        let Declaration::Entry(entry) = decl else {
            continue;
        };
        for param in &entry.params {
            for attr in param_attrs(param) {
                if let Some((s, b)) = explicit_slot(attr) {
                    used.insert((s, b));
                }
            }
        }
    }
    used
}

fn explicit_slot(attr: &Attribute) -> Option<(u32, u32)> {
    match attr {
        Attribute::Storage { set, binding, .. }
        | Attribute::Uniform { set, binding }
        | Attribute::Texture { set, binding, .. }
        | Attribute::Sampler { set, binding }
        | Attribute::StorageImage { set, binding, .. } => Some((*set, *binding)),
        _ => None,
    }
}

fn any_view(program: &ast::Program) -> bool {
    program.declarations.iter().any(|d| match d {
        Declaration::Entry(e) => {
            e.params.iter().any(|p| param_attrs(p).iter().any(|a| matches!(a, Attribute::View { .. })))
        }
        _ => false,
    })
}

/// The attribute list on an entry param (`#[attr] name: ty`), peeling the
/// `Attributed` layer.
fn param_attrs(p: &Pattern) -> &[Attribute] {
    match &p.kind {
        PatternKind::Attributed(attrs, _) => attrs,
        PatternKind::Typed(inner, _) => param_attrs(inner),
        _ => &[],
    }
}

fn param_attrs_mut(p: &mut Pattern) -> Option<&mut Vec<Attribute>> {
    match &mut p.kind {
        PatternKind::Attributed(attrs, _) => Some(attrs),
        PatternKind::Typed(inner, _) => param_attrs_mut(inner),
        _ => None,
    }
}

fn type_name_of(ty: &Type) -> Option<TypeName> {
    match ty {
        Type::Constructed(name, _) => Some(name.clone()),
        _ => None,
    }
}
