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

use crate::ast::{self, Declaration, Pattern};
use crate::error::Result;
use crate::interface::{
    Attribute, FeedbackPair, ResolvedAttribute, ResourceDecl, ResourceUsage, StorageAccess,
};
use crate::types::{Type, TypeName};
use crate::{bail_type_at, BindingRef, LookupMap, LookupSet};

pub type ResourcesResolvedFamily = ast::AstFamily<
    ast::SourceTree,
    ast::DefinitionSyntax,
    ast::ResolvedEntry,
    ResolvedAttribute,
    ast::ExternSyntax,
    ast::ResourcesResolvedFrontend<ast::NestedDeclaration>,
>;

/// AST after resource declarations and source-only `#[view]` attributes have
/// been consumed into concrete entry metadata.
#[derive(Debug, Clone, Copy)]
pub enum ResourcesResolvedTag {}
pub type ResourcesResolved =
    ast::Program<ResourcesResolvedTag, ResourcesResolvedFamily, crate::module_manager::ModuleManager>;

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

pub fn resolve_resources(mut program: crate::name_resolution::NamesResolved) -> Result<ResourcesResolved> {
    let decls: Vec<ResourceDecl> = program
        .declarations
        .iter()
        .filter_map(|d| match d {
            Declaration::Frontend(ast::ModulesElaboratedFrontend::Resource(r)) => Some(r.clone()),
            _ => None,
        })
        .collect();
    let table = derive_bindings(&decls, &program)?;
    let mut entry_feedback = std::collections::VecDeque::new();
    for declaration in &mut program.declarations {
        if let Declaration::Entry(entry) = declaration {
            let mut feedback = Vec::new();
            for param in &mut entry.params {
                rewrite_view_param(param, &table, &mut feedback)?;
            }
            entry_feedback.push_back(feedback);
        }
    }
    materialize(program, entry_feedback)
}

/// Assign each resource its current (and, for history resources, previous)
/// binding: honor pins, then auto-assign the rest to free slots on the default
/// set, avoiding slots already taken by explicit param attributes or pins.
fn derive_bindings(
    decls: &[ResourceDecl],
    program: &crate::name_resolution::NamesResolved,
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

/// Apply the established entry-parameter rewrite. The later materialization
/// step changes only the AST's attribute type.
fn rewrite_view_param(
    param: &mut Pattern,
    table: &LookupMap<String, ResolvedResource>,
    feedback: &mut Vec<FeedbackPair>,
) -> Result<()> {
    if !param.attributes().iter().any(|attribute| matches!(attribute, Attribute::View(_))) {
        return Ok(());
    }
    let span = param.h.span;
    let handle = param.pattern_type().and_then(type_name_of);
    let attributes = param
        .attributes_mut()
        .ok_or_else(|| crate::err_type_at!(span, "view attribute on a param without an attribute list"))?;
    for attribute in attributes {
        let Attribute::View(view) = attribute else {
            continue;
        };
        let crate::interface::ViewAttribute {
            resource,
            usage,
            previous,
        } = view.clone();
        let resolved = table
            .get(&resource)
            .ok_or_else(|| crate::err_type_at!(span, "unknown resource '{}' in view", resource))?;
        if !resolved.decl.usages.contains(&usage) {
            bail_type_at!(span, "resource '{}' does not declare usage {:?}", resource, usage);
        }
        *attribute = match usage {
            ResourceUsage::StorageWrite | ResourceUsage::StorageRead => {
                if handle != Some(TypeName::StorageTexture) {
                    bail_type_at!(
                        span,
                        "view usage {:?} of '{}' requires a `storage_image` param",
                        usage,
                        resource
                    );
                }
                let binding = resolved.current_storage.expect("storage usage implies a storage slot");
                Attribute::StorageImage {
                    set: binding.set,
                    binding: binding.binding,
                    format: resolved.decl.format,
                    access: if matches!(usage, ResourceUsage::StorageWrite) {
                        StorageAccess::WriteOnly
                    } else {
                        StorageAccess::ReadOnly
                    },
                    size: resolved.decl.size,
                    resource: Some(resource),
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
                let binding = if previous {
                    let previous_binding = resolved.previous_sampled.ok_or_else(|| {
                        crate::err_type_at!(
                            span,
                            "view of '{}' uses `previous`, but the resource has no `history`",
                            resource
                        )
                    })?;
                    if let Some(write) = resolved.current_storage {
                        feedback.push(FeedbackPair {
                            read: previous_binding,
                            write,
                        });
                    }
                    previous_binding
                } else {
                    resolved.current_sampled.expect("sampled usage implies a sampled slot")
                };
                Attribute::Texture {
                    set: binding.set,
                    binding: binding.binding,
                    backing: resolved.current_storage,
                    resource: (!previous).then_some(resource),
                }
            }
        };
    }
    Ok(())
}

fn materialize(
    program: crate::name_resolution::NamesResolved,
    mut entry_feedback: std::collections::VecDeque<Vec<FeedbackPair>>,
) -> Result<ResourcesResolved> {
    let ast::Program {
        declarations,
        node_ids,
        global_context,
        state: _,
    } = program;
    let mut resolved = Vec::with_capacity(declarations.len());
    for declaration in declarations {
        let declaration = match declaration {
            Declaration::Decl(definition) => Some(Declaration::Decl(definition)),
            Declaration::Entry(entry) => Some(Declaration::Entry(materialize_entry(
                entry,
                entry_feedback.pop_front().expect("resource analysis records every entry"),
            )?)),
            Declaration::Extern(ext) => Some(Declaration::Extern(ext)),
            Declaration::Frontend(frontend) => materialize_frontend(frontend).map(Declaration::Frontend),
        };
        if let Some(declaration) = declaration {
            resolved.push(declaration);
        }
    }
    debug_assert!(entry_feedback.is_empty());
    Ok(ast::Program {
        declarations: resolved,
        node_ids,
        global_context,
        state: std::marker::PhantomData,
    })
}

fn materialize_attribute(attribute: Attribute, span: ast::Span) -> Result<ResolvedAttribute> {
    let Attribute::View(_) = attribute else {
        return Ok(attribute.map_view(|_| unreachable!()));
    };
    bail_type_at!(span, "view attributes are only valid on entry parameters")
}

fn materialize_pattern(pattern: Pattern) -> Result<Pattern<ast::Header, ResolvedAttribute>> {
    let span = pattern.h.span;
    pattern.try_map_attributes(&mut |attribute| materialize_attribute(attribute, span))
}

fn materialize_entry(
    entry: ast::EntryDecl,
    feedback: Vec<FeedbackPair>,
) -> Result<ast::EntryDecl<ast::ResolvedEntry, ast::SourceTree, ResolvedAttribute>> {
    let ast::EntryDecl {
        data,
        name,
        name_span,
        size_params,
        type_params,
        params,
        body,
    } = entry;
    let syntax = data.try_map_attributes(|attribute| materialize_attribute(attribute, name_span))?;
    Ok(ast::EntryDecl {
        data: ast::ResolvedEntry { syntax, feedback },
        name,
        name_span,
        size_params,
        type_params,
        params: params.into_iter().map(materialize_pattern).collect::<Result<_>>()?,
        body,
    })
}

fn materialize_frontend(
    declaration: ast::ModulesElaboratedFrontend<ast::NestedDeclaration>,
) -> Option<ast::ResourcesResolvedFrontend<ast::NestedDeclaration>> {
    match declaration {
        ast::ModulesElaboratedFrontend::Sig(sig) => Some(ast::ResourcesResolvedFrontend::Sig(sig)),
        ast::ModulesElaboratedFrontend::TypeBind(bind) => {
            Some(ast::ResourcesResolvedFrontend::TypeBind(bind))
        }
        ast::ModulesElaboratedFrontend::Open(open) => Some(ast::ResourcesResolvedFrontend::Open(open)),
        ast::ModulesElaboratedFrontend::Resource(_) => None,
    }
}

/// Every `(set, binding)` already claimed by an explicit binding attribute on
/// any entry param — so auto-assigned resources don't collide with them.
fn collect_explicit_slots(program: &crate::name_resolution::NamesResolved) -> LookupSet<(u32, u32)> {
    let mut used = LookupSet::new();
    for decl in &program.declarations {
        let Declaration::Entry(entry) = decl else {
            continue;
        };
        for param in &entry.params {
            for attr in param.attributes() {
                if let Some((s, b)) = attr.binding_slot() {
                    used.insert((s, b));
                }
            }
        }
    }
    used
}

fn type_name_of(ty: &Type) -> Option<TypeName> {
    match ty {
        Type::Constructed(name, _) => Some(name.clone()),
        _ => None,
    }
}
