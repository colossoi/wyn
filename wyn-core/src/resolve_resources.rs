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

/// A resource's derived bindings.
struct ResolvedResource {
    decl: ResourceDecl,
    current: BindingRef,
    /// The previous-frame binding, present iff `decl.history >= 1`.
    previous: Option<BindingRef>,
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
        if let Some(b) = r.previous_layout {
            used.insert((b.set, b.binding));
        }
    }

    let auto_next = |used: &mut LookupSet<(u32, u32)>| -> BindingRef {
        let mut b = 0u32;
        while used.contains(&(DEFAULT_RESOURCE_SET, b)) {
            b += 1;
        }
        used.insert((DEFAULT_RESOURCE_SET, b));
        BindingRef::new(DEFAULT_RESOURCE_SET, b)
    };

    let mut table: LookupMap<String, ResolvedResource> = LookupMap::new();
    let mut pinned: LookupMap<(u32, u32), String> = LookupMap::new();
    for r in decls {
        if table.contains_key(&r.name) {
            bail_type_at!(r.span, "duplicate resource '{}'", r.name);
        }
        let current = r.layout.unwrap_or_else(|| auto_next(&mut used));
        let previous = if r.history >= 1 {
            Some(r.previous_layout.unwrap_or_else(|| auto_next(&mut used)))
        } else {
            None
        };
        // Distinct resources must not pin the same slot.
        for b in [Some(current), previous].into_iter().flatten() {
            if r.layout == Some(b) || r.previous_layout == Some(b) {
                if let Some(prev) = pinned.insert((b.set, b.binding), r.name.clone()) {
                    bail_type_at!(
                        r.span,
                        "resources '{}' and '{}' both pin (set={}, binding={})",
                        r.name,
                        prev,
                        b.set,
                        b.binding
                    );
                }
            }
        }
        table.insert(
            r.name.clone(),
            ResolvedResource {
                decl: r.clone(),
                current,
                previous,
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
        // Pick the binding: previous frame (history resource) or current.
        let binding = if *previous {
            let prev = res.previous.ok_or_else(|| {
                crate::err_type_at!(
                    span,
                    "view of '{}' uses `previous`, but the resource has no `history`",
                    resource
                )
            })?;
            feedback.push(FeedbackPair {
                read: prev,
                write: res.current,
            });
            prev
        } else {
            res.current
        };
        // Validate usage against the param's handle type, then desugar.
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
                Attribute::Texture {
                    set: binding.set,
                    binding: binding.binding,
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
        | Attribute::Texture { set, binding }
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
