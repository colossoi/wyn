//! Semantic SegOp fusion (EGIR milestone 5).
//!
//! These passes are driven by `egir::semantic_opt` and rest on two primitives:
//! provenance-based `SegSpace` equality (`space`), and
//! `egir::semantic_graph::SemanticGraph`, the query layer over the semantic
//! dependency DAG. That oracle owns the invariant fusion rests on: never move
//! an operation across resource or effect ordering.
//!
//! Horizontal fusion combines independent siblings. Vertical fusion composes
//! callable regions for a pure, single-consumer `SegMap` producer and its
//! same-space consumer. Multi-consumer producers deliberately survive for the
//! allocation pass to materialize once. Envelope fusion composes maps into
//! filters, scatters, and histograms; filter fusion redirects scalar consumers
//! through the compacted route; indexed scalarization removes producers whose
//! complete demand is a set of scalar element reads.

use polytype::Type;

use crate::ast::TypeName;
use crate::egir::types::NodeId;

pub(crate) mod envelope;
pub(crate) mod filter;
pub(crate) mod horizontal;
pub(crate) mod indexed;
pub(crate) mod space;
pub(crate) mod vertical;

/// Canonicalize a semantic operation's parallel array inputs by `NodeId` and
/// return an old-index to new-index map. Fusion frequently concatenates input
/// vectors from independently built operations; retaining duplicate nodes
/// would duplicate region parameters and obscure equal-domain provenance.
pub(super) fn deduplicate_array_inputs(
    nodes: Vec<NodeId>,
    array_types: Vec<Type<TypeName>>,
    elem_types: Vec<Type<TypeName>>,
) -> (Vec<NodeId>, Vec<Type<TypeName>>, Vec<Type<TypeName>>, Vec<usize>) {
    debug_assert_eq!(nodes.len(), array_types.len());
    debug_assert_eq!(nodes.len(), elem_types.len());
    let mut unique_nodes = Vec::new();
    let mut unique_array_types = Vec::new();
    let mut unique_elem_types = Vec::new();
    let mut remap = Vec::with_capacity(nodes.len());
    for ((node, array_ty), elem_ty) in nodes.into_iter().zip(array_types).zip(elem_types) {
        if let Some(index) = unique_nodes.iter().position(|existing| *existing == node) {
            debug_assert_eq!(unique_array_types[index], array_ty);
            debug_assert_eq!(unique_elem_types[index], elem_ty);
            remap.push(index);
        } else {
            remap.push(unique_nodes.len());
            unique_nodes.push(node);
            unique_array_types.push(array_ty);
            unique_elem_types.push(elem_ty);
        }
    }
    (unique_nodes, unique_array_types, unique_elem_types, remap)
}
