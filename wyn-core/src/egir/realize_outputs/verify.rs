//! Post-realization invariant check.
//!
//! After output realization and residency planning, the following must hold
//! for every `SemanticEntry` and materialization entry:
//!
//!   * Every declared output has at least one explicit route, and every route
//!     names at least one realized writer.
//!
//!   * No runtime-sized Composite array is reachable from any entry
//!     output or from any output-side-effect operand.
//!
//! This catches the failure mode that would otherwise crash the SPIR-V
//! backend with "Composite variant unsized arrays not supported" at
//! `spirv/mod.rs:383`. By the time we hit codegen, every storage write
//! goes through a `StorageView`; every retargetable SOAC has
//! `destination: OutputView` or `InputBuffer`.
//!
//! The verifier walks each entry's e-graph from the side-effects' Pure
//! operand nodes and from each `SkeletonTerminator::Return(Some(_))`.
//! For every reached Pure node, if its type is a runtime-sized
//! `Array` with `ArrayVariantComposite`, emit a diagnostic naming the
//! entry and offending NodeId.
//!
//! In debug builds, the residency planner calls `check` after all rewrites.

use crate::LookupSet;
use polytype::Type;

use crate::ast::TypeName;

use super::super::from_tlc::ConvertError;
use super::super::program::{Entry, Program};
use super::super::types::{EGraph, ENode, EgirPhase, NodeId, SkeletonTerminator};

/// Verify the post-realization invariant for every entry. Returns
/// `ConvertError::Internal` on the first violation, naming the entry
/// and offending NodeId.
pub fn check<P: EgirPhase>(inner: &Program<P>) -> Result<(), ConvertError> {
    for (endpoint, entry) in inner.entries_with_endpoints() {
        if matches!(endpoint, super::super::program::CompilerFlowEndpoint::Entry(_)) {
            check_routes(entry)?;
        }
        check_entry(&entry.name, &entry.graph)?;
    }
    Ok(())
}

fn check_routes<P: EgirPhase>(entry: &Entry<P>) -> Result<(), ConvertError> {
    for route in &entry.output_routes {
        if route.slot.0 >= entry.outputs.len() {
            return Err(ConvertError::Internal(format!(
                "realize_outputs verifier: entry `{}` has a route to output slot {} but declares only {} outputs",
                entry.name,
                route.slot.0,
                entry.outputs.len()
            )));
        }
        if route.writers.is_empty() {
            return Err(ConvertError::Internal(format!(
                "realize_outputs verifier: entry `{}` output slot {} has a source value but no realized writer",
                entry.name, route.slot.0
            )));
        }
    }
    for slot in 0..entry.outputs.len() {
        if !entry.output_routes.iter().any(|route| route.slot.0 == slot) {
            return Err(ConvertError::Internal(format!(
                "realize_outputs verifier: entry `{}` output slot {} has no explicit route",
                entry.name, slot
            )));
        }
    }
    Ok(())
}

fn check_entry<P: EgirPhase>(entry_name: &str, graph: &EGraph<P>) -> Result<(), ConvertError> {
    // Roots: the operand of every Return(Some(_)) terminator, plus
    // every Pure NodeId referenced by a side-effect store's operands.
    // We don't walk SOAC `EgirSoac` operands here: those are
    // legitimate consumers of arrays at the SOAC's input position,
    // not output operands. The runtime-sized check applies to values
    // that flow into a store or off the entry's return.
    let mut roots: Vec<NodeId> = Vec::new();
    for (_, block) in &graph.skeleton.blocks {
        if let SkeletonTerminator::Return(Some(r)) = block.term {
            roots.push(r);
        }
        for se in &block.side_effects {
            // Stores' operand_nodes carry the value being written.
            // Skip EgirSoacs — their array operands are inputs,
            // not output writes.
            use super::super::types::SideEffectKind;
            match &se.kind {
                SideEffectKind::Soac(_) => continue,
                _ => {
                    roots.extend(se.operand_nodes.iter().copied());
                }
            }
        }
    }

    // Walk Pure operand edges from each root, checking each node's type.
    let mut seen: LookupSet<NodeId> = LookupSet::new();
    let mut work: Vec<NodeId> = roots;
    while let Some(nid) = work.pop() {
        if !seen.insert(nid) {
            continue;
        }
        let ENode::Pure { operands, .. } = &graph.nodes[nid] else {
            continue;
        };
        if let Some(ty) = node_type(graph, nid) {
            if is_runtime_sized_composite_array(ty) {
                return Err(ConvertError::Internal(format!(
                    "realize_outputs verifier: entry `{}` leaks a \
                     runtime-sized Composite array at NodeId {:?} \
                     (type {:?}) reachable from an entry output or \
                     output-side-effect operand. This would crash \
                     the SPIR-V backend at codegen; investigate the \
                     producer of this NodeId.",
                    entry_name, nid, ty
                )));
            }
        }
        work.extend(operands.iter().copied());
    }
    Ok(())
}

/// True iff `ty` is `Array(elem, ArrayVariantComposite, size)` where
/// `size` is a free variable or `SizePlaceholder` (runtime-sized).
/// `Array` with `ArrayVariantView` is fine — view-arrays are
/// storage-buffer-backed, not in-register composites.
fn is_runtime_sized_composite_array(ty: &Type<TypeName>) -> bool {
    let Type::Constructed(TypeName::Array, args) = ty else {
        return false;
    };
    if args.len() < 3 {
        return false;
    }
    let variant = &args[1];
    let size = &args[2];
    let is_composite = matches!(variant, Type::Constructed(TypeName::ArrayVariantComposite, _));
    let is_runtime = matches!(
        size,
        Type::Variable(_) | Type::Constructed(TypeName::SizePlaceholder, _)
    );
    is_composite && is_runtime
}

/// Look up the Pure result type for `nid`. ENode::Pure carries its
/// declared type; we just project the field.
fn node_type<P: EgirPhase>(graph: &EGraph<P>, nid: NodeId) -> Option<&Type<TypeName>> {
    match &graph.nodes[nid] {
        ENode::Pure { .. } => graph.types.get(&nid),
        _ => None,
    }
}
