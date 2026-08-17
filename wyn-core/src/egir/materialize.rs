//! EGIR materialize pass.
//!
//! Rewrites every pure `Index(arr, i)` whose index isn't a compile-time
//! constant into `DynamicExtract(Materialize(arr), i)`. The SPIR-V backend
//! needs this because `OpCompositeExtract` requires literal indices —
//! anything else has to spill the composite to a memory-backed handle and
//! `OpAccessChain` into it.
//!
//! `Materialize` depends only on the array operand, so `elaborate`'s
//! scoped, loop-aware placement emits it at the deepest dominator of the
//! array—outside the loop whenever the array is loop-invariant.
//!
//! Two `Index` nodes with the same array share a single `Materialize` node
//! via hash-consing, so we don't need a separate dedup step either.
//!
//! Storage-view arrays and index spines rooted in them are exempt: they are
//! memory-backed already, so elaboration retains their coordinates as one
//! address chain into the backing buffer. Spilling an intermediate row to a
//! Function-local composite would both lose that chain and copy data that the
//! source never selected. Only genuinely in-register composites need the
//! rewrite.

use smallvec::smallvec;

use polytype::Type;

use crate::ast::TypeName;
use crate::ssa::types::ConstantValue;
use crate::types::TypeExt;

/// Physical EGIR with dynamic composite extraction made explicit.
#[derive(Debug, Clone, Copy)]
pub enum MaterializedTag {}
pub type Materialized = super::program::Program<
    MaterializedTag,
    super::ir::ProgramFamily<
        super::types::Physical,
        crate::interface::StorageBindingDecl,
        super::ir::RealizedOutputRoute,
        super::program::CoreProgramData,
    >,
    super::program::PlannedGlobal,
>;

use super::types::{
    EGraph, EffectOp, EffectToken, Family, PureOp, ResultBinding, ResultDestination, SideEffectKind,
    ValueId, ValueKind,
};

/// Make dynamic composite extraction explicit in every body.
pub fn materialize_dynamic_extracts(program: super::partial_inline::PartiallyInlined) -> Materialized {
    program
        .try_map_graphs_with_state(|_, mut graph, _, context| {
            run_one_body(&mut graph, &mut context.effect_ids);
            Ok::<_, std::convert::Infallible>(graph)
        })
        .unwrap()
        .retag()
}

/// Rewrite all dynamic Index nodes in the e-graph to Materialize +
/// DynamicExtract.
fn run_one_body<P: Family>(graph: &mut EGraph<P>, effect_ids: &mut crate::IdSource<EffectToken>) {
    normalize_place_backed_stores(graph, effect_ids);
    let place_backed = graph
        .nodes
        .iter()
        .filter_map(|(value, node)| matches!(node.kind(), ValueKind::PlaceView { .. }).then_some(value))
        .collect::<Vec<_>>();
    for value in place_backed {
        super::graph_ops::normalize_place_backed_value_consumers(graph, value);
    }

    // Snapshot first; we'll mutate node entries and add new Materialize nodes.
    let targets: Vec<(ValueId, ValueId, ValueId)> = graph
        .nodes
        .iter()
        .filter_map(|(nid, node)| match &node.kind {
            ValueKind::Pure {
                op: PureOp::Index,
                operands,
            } if node.alias().is_none() && operands.len() == 2 => {
                let arr = graph.canonical_value(operands[0]);
                let idx = graph.canonical_value(operands[1]);
                if is_const_int(graph, idx) || index_spine_reaches_addressable_source(graph, nid) {
                    None
                } else {
                    Some((nid, arr, idx))
                }
            }
            _ => None,
        })
        .collect();

    for (index_nid, arr_nid, idx_nid) in targets {
        let arr_ty = graph.nodes[arr_nid].ty.clone();

        // Materialize is hash-consed: two Index(arr, _) share the same
        // Materialize(arr) handle automatically.
        let mat_nid = graph.intern_pure(PureOp::Materialize, smallvec![arr_nid], arr_ty, None);

        // Replace the original Index node in place with DynamicExtract(mat, idx).
        // The ValueId stays the same so all consumers continue to resolve through it.
        // The node's stored type is unchanged (still elem_ty).
        graph.replace_pure_node(index_nid, PureOp::DynamicExtract, smallvec![mat_nid, idx_nid]);
    }
}

fn normalize_place_backed_stores<P: Family>(
    graph: &mut EGraph<P>,
    effect_ids: &mut crate::IdSource<EffectToken>,
) {
    loop {
        let candidate = graph.skeleton.blocks.values().find_map(|contents| {
            contents.side_effects.iter().find_map(|effect| {
                let SideEffectKind::Effect(EffectOp::Store { .. }) = effect.kind() else {
                    return None;
                };
                let [source] = effect.operands() else {
                    return None;
                };
                let source = graph.canonical_value(source.value()?);
                if let Some(origin) =
                    graph.nodes[source].result_origins().iter().find(|origin| !origin.places().is_empty())
                {
                    return Some((source, origin.clone()));
                }
                matches!(graph.operand_ref(source), super::types::OperandRef::View(_)).then(|| {
                    (
                        source,
                        ResultBinding::destination(
                            graph.nodes[source].ty.clone(),
                            ResultDestination::ReturnValue(source),
                        ),
                    )
                })
            })
        });
        let Some((source, result)) = candidate else {
            break;
        };
        super::graph_ops::rewrite_result_store_consumers(graph, source, &result, effect_ids)
            .expect("place-backed store must have an addressable physical copy");
    }
}

/// Is `nid`'s array type a storage view? `lower_index` reads a view with a
/// native dynamic `OpAccessChain`, so it must not be spilled to a composite.
fn is_view<P: Family>(graph: &EGraph<P>, nid: ValueId) -> bool {
    let nid = graph.canonical_value(nid);
    graph.nodes.get(nid).is_some_and(|node| {
        matches!(node.kind(), ValueKind::PlaceView { .. })
            || matches!(
                node.ty.array_variant(),
                Some(Type::Constructed(TypeName::ArrayVariantView, _))
            )
    })
}

/// True when `nid` is an `Index` whose base chain reaches a storage view or a
/// callable parameter without crossing a non-index value operation. Both are
/// already addressable at the physical boundary; spilling an intermediate row
/// would replace direct indexing with a whole-aggregate copy.
fn index_spine_reaches_addressable_source<P: Family>(graph: &EGraph<P>, mut nid: ValueId) -> bool {
    loop {
        nid = graph.canonical_value(nid);
        let Some(ValueKind::Pure {
            op: PureOp::Index,
            operands,
        }) = graph.nodes.get(nid).map(|node| &node.kind)
        else {
            return false;
        };
        let Some(&base) = operands.first() else {
            return false;
        };
        let base = graph.canonical_value(base);
        if is_view(graph, base) || matches!(graph.nodes[base].kind(), ValueKind::FuncParam { .. }) {
            return true;
        }
        nid = base;
    }
}

/// Is this ValueId a compile-time integer constant? Includes both the inline
/// `ValueKind::Constant(ConstantValue::I32|U32)` form and `ValueKind::Pure(PureOp::Int|Uint)`.
fn is_const_int<P: Family>(graph: &EGraph<P>, nid: ValueId) -> bool {
    match &graph.nodes[nid].kind {
        ValueKind::Constant(ConstantValue::I32(_) | ConstantValue::U32(_)) => true,
        ValueKind::Pure {
            op: PureOp::Int(_) | PureOp::Uint(_),
            ..
        } => true,
        _ => false,
    }
}
