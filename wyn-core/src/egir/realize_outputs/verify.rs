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
//! entry and offending ValueId.
//!
//! In debug builds, the residency planner calls `check` after all rewrites.

use crate::LookupSet;
use polytype::Type;

use crate::ast::TypeName;
use crate::types::TypeExt;

use super::super::allocation::{entries_with_endpoints, CompilerFlowEndpoint, ResourcesAllocated};
use super::super::from_tlc::ConvertError;
use super::super::program::SemanticEntry;
use super::super::types::{EGraph, Family, SkeletonTerminator, ValueId, ValueKind};

/// Verify the post-realization invariant for every entry. Returns
/// `ConvertError::Internal` on the first violation, naming the entry
/// and offending ValueId.
pub fn check(inner: &ResourcesAllocated) -> Result<(), ConvertError> {
    for (endpoint, entry) in entries_with_endpoints(inner) {
        if matches!(endpoint, CompilerFlowEndpoint::Entry(_)) {
            check_routes(entry)?;
        }
        check_entry(&entry.name, &entry.graph)?;
    }
    Ok(())
}

fn check_routes(entry: &SemanticEntry) -> Result<(), ConvertError> {
    for (slot, output) in entry.outputs.iter().enumerate() {
        if output.routes.is_empty() {
            return Err(ConvertError::Internal(format!(
                "realize_outputs verifier: entry `{}` output slot {} has no explicit route",
                entry.name, slot
            )));
        }
        for route in &output.routes {
            if route.writers.is_empty() {
                return Err(ConvertError::Internal(format!(
                    "realize_outputs verifier: entry `{}` output slot {} has a source value but no realized writer",
                    entry.name, slot
                )));
            }
        }
    }
    Ok(())
}

fn check_entry<P: Family>(entry_name: &str, graph: &EGraph<P>) -> Result<(), ConvertError> {
    // Roots: the operand of every Return(Some(_)) terminator, plus
    // every Pure ValueId referenced by a side-effect store's operands.
    // We don't walk SOAC `EgirSoac` operands here: those are
    // legitimate consumers of arrays at the SOAC's input position,
    // not output operands. The runtime-sized check applies to values
    // that flow into a store or off the entry's return.
    let mut roots: Vec<ValueId> = Vec::new();
    for (_, block) in &graph.skeleton.blocks {
        if let SkeletonTerminator::Return(Some(r)) = &block.term {
            roots.extend(r.values());
        }
        for se in &block.side_effects {
            // Stores' operands carry the value being written.
            // Skip EgirSoacs — their array operands are inputs,
            // not output writes.
            use super::super::types::SideEffectKind;
            match &se.kind {
                SideEffectKind::Soac(_) => continue,
                _ => {
                    roots.extend(se.operand_values());
                }
            }
        }
    }

    // Walk Pure operand edges from each root, checking each node's type.
    let mut seen: LookupSet<ValueId> = LookupSet::new();
    let mut work: Vec<ValueId> = roots;
    while let Some(nid) = work.pop() {
        if !seen.insert(nid) {
            continue;
        }
        let ValueKind::Pure { operands, .. } = &graph.nodes[nid].kind else {
            continue;
        };
        if let Some(ty) = node_type(graph, nid) {
            if ty.contains_runtime_sized_composite_array() {
                return Err(ConvertError::Internal(format!(
                    "realize_outputs verifier: entry `{}` leaks a \
                     runtime-sized Composite array at ValueId {:?} \
                     (type {:?}) reachable from an entry output or \
                     output-side-effect operand. This would crash \
                     the SPIR-V backend at codegen; investigate the \
                     producer of this ValueId.",
                    entry_name, nid, ty
                )));
            }
        }
        work.extend(operands.iter().copied());
    }
    Ok(())
}

/// Look up the Pure result type for `nid`. ValueKind::Pure carries its
/// declared type; we just project the field.
fn node_type<P: Family>(graph: &EGraph<P>, nid: ValueId) -> Option<&Type<TypeName>> {
    match &graph.nodes[nid].kind {
        ValueKind::Pure { .. } => Some(&graph.nodes[nid].ty),
        _ => None,
    }
}
