//! Post-realization invariant check.
//!
//! After `realize_outputs::run`, the following must hold for every
//! `EgirEntry`:
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
//! In `debug_assertions` builds (the default for `cargo test`,
//! `cargo build`, anything but `--release`), `realize_outputs::run`
//! calls `check` automatically. Release builds skip the check —
//! production compilation is the same as today.

use crate::LookupSet;
use polytype::Type;

use crate::ast::TypeName;

use super::super::from_tlc::ConvertError;
use super::super::program::EgirInner;
use super::super::types::{ENode, NodeId, SkeletonTerminator};

/// Verify the post-realization invariant for every entry. Returns
/// `ConvertError::Internal` on the first violation, naming the entry
/// and offending NodeId.
pub fn check(inner: &EgirInner) -> Result<(), ConvertError> {
    for entry in &inner.entry_points {
        check_entry(&entry.name, &entry.graph)?;
    }
    Ok(())
}

fn check_entry(entry_name: &str, graph: &super::super::types::EGraph) -> Result<(), ConvertError> {
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

    // BFS over Pure operand edges from each root, checking each
    // node's type.
    let mut seen: LookupSet<NodeId> = LookupSet::new();
    let mut work: Vec<NodeId> = roots;
    while let Some(nid) = work.pop() {
        if !seen.insert(nid) {
            continue;
        }
        let node = &graph.nodes[nid];
        match node {
            ENode::Pure { operands, .. } => {
                // Type check.
                let ty = node_type(graph, nid);
                if let Some(ty) = ty {
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
                for op in operands {
                    work.push(*op);
                }
            }
            _ => {}
        }
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
fn node_type<'a>(graph: &'a super::super::types::EGraph, nid: NodeId) -> Option<&'a Type<TypeName>> {
    match &graph.nodes[nid] {
        ENode::Pure { .. } => graph.types.get(&nid),
        _ => None,
    }
}
