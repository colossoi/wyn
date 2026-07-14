//! Same-space horizontal fusion: merge sibling SegOps over an equal iteration
//! space into one multi-output SegOp so a single kernel produces several
//! outputs. This is the structural inverse of `split_multidomain_seg_maps` and
//! the rewrite that retires the ad-hoc multi-reduce demotion guard.
//!
//! One fusion per call: the driver rebuilds the dependency DAG and re-invokes,
//! so the legality oracle is never stale.

use polytype::Type;
use smallvec::SmallVec;

use super::space::seg_space_fusable;
use crate::ast::TypeName;
use crate::egir::graph_ops;
use crate::egir::program::SemanticProgram;
use crate::egir::semantic_graph::SemanticGraph;
use crate::egir::types::{
    reify_seg_kind_operators, EGraph, EgirSoac, NodeId, PureOp, SegBody, SegOpKind, SegPlacement,
    SegResourceAccess, SideEffectKind, SoacDestination,
};
use crate::ssa::framework::BlockId;
use crate::tlc::ScremaAccumulator;

/// Find one legal sibling pair anywhere in the program and fuse it. Returns
/// whether a fusion happened.
pub fn fuse_sibling_seg_ops(inner: &mut SemanticProgram, oracle: &SemanticGraph) -> bool {
    for idx in 0..inner.entry_points.len() {
        let scope = inner.entry_points[idx].name.clone();
        if fuse_in_graph(&mut inner.entry_points[idx].graph, &scope, oracle) {
            return true;
        }
    }
    for idx in 0..inner.functions.len() {
        let scope = inner.functions[idx].name.clone();
        if fuse_in_graph(&mut inner.functions[idx].graph, &scope, oracle) {
            return true;
        }
    }
    false
}

fn fuse_in_graph(graph: &mut EGraph, scope: &str, oracle: &SemanticGraph) -> bool {
    let block_ids: Vec<BlockId> = graph.skeleton.blocks.iter().map(|(id, _)| id).collect();
    for block_id in block_ids {
        let segs: Vec<usize> = (0..graph.skeleton.blocks[block_id].side_effects.len())
            .filter(|&i| is_fusable_seg(&graph.skeleton.blocks[block_id].side_effects[i].kind))
            .collect();
        for a in 0..segs.len() {
            for b in (a + 1)..segs.len() {
                let (i, j) = (segs[a], segs[b]);
                if sibling_fusable(graph, block_id, i, j, scope, oracle) {
                    fuse_pair(graph, block_id, i, j);
                    return true;
                }
            }
        }
    }
    false
}

fn is_fusable_seg(kind: &SideEffectKind) -> bool {
    matches!(kind, SideEffectKind::Soac(EgirSoac::Seg { .. }))
}

/// Legality: equal space, compatible placement, no resource/effect conflict, and
/// not a producer/consumer chain (that is vertical fusion's job).
fn sibling_fusable(
    graph: &EGraph,
    block_id: BlockId,
    i: usize,
    j: usize,
    _scope: &str,
    oracle: &SemanticGraph,
) -> bool {
    let block = &graph.skeleton.blocks[block_id];
    let effect_i = &block.side_effects[i];
    let effect_j = &block.side_effects[j];
    let (
        SideEffectKind::Soac(EgirSoac::Seg {
            space: sp_i,
            placement: pl_i,
            kind: kind_i,
            input_array_types: inputs_i,
            ..
        }),
        Some(_),
    ) = (&effect_i.kind, effect_i.result)
    else {
        return false;
    };
    let (
        SideEffectKind::Soac(EgirSoac::Seg {
            space: sp_j,
            placement: pl_j,
            kind: kind_j,
            input_array_types: inputs_j,
            ..
        }),
        Some(_),
    ) = (&effect_j.kind, effect_j.result)
    else {
        return false;
    };
    // A shared input value or a shared symbolic size is also a proof of equal
    // dynamic extent, even when the two length expressions were not hash-consed.
    // Independently-authored `[]T` parameters receive distinct skolems during
    // type checking, while `[n]` parameters intentionally share one.
    let shared_input = effect_i
        .operand_nodes
        .first()
        .zip(effect_j.operand_nodes.first())
        .is_some_and(|(left, right)| left == right);
    let shared_size = inputs_i
        .first()
        .and_then(crate::types::array_size)
        .zip(inputs_j.first().and_then(crate::types::array_size))
        .is_some_and(|(left, right)| left == right);
    let symbolic_domain_matches = shared_input || shared_size;
    if pl_i != pl_j || (!seg_space_fusable(sp_i, sp_j) && !symbolic_domain_matches) {
        return false;
    }
    let operators = kind_i.operators().iter().chain(kind_j.operators());
    let (mut has_reduce, mut has_scan) = (false, false);
    for operator in operators {
        has_reduce |= operator.kind == ScremaAccumulator::Reduce;
        has_scan |= operator.kind == ScremaAccumulator::Scan;
    }
    if has_reduce && has_scan {
        // The target scheduler has separate reduction and scan phase models.
        // Keep mixed siblings independent until a joint physical recipe exists.
        return false;
    }
    let Some(op_i) = block.side_effects[i].semantic_id else {
        return false;
    };
    let Some(op_j) = block.side_effects[j].semantic_id else {
        return false;
    };
    // A value edge either way makes them a producer/consumer chain (fused at the
    // TLC level), never fusable siblings.
    if oracle.reachable_between(&op_i, &op_j) || oracle.reachable_between(&op_j, &op_i) {
        return false;
    }
    // Fusing performs P's effects then Q's, in order, so a P–Q resource conflict
    // (e.g. both writing fields of one aggregate output) is *not* blocking.
    // Moving Q up past an *intervening* op is, though, if that op aliases P or Q:
    // require every op strictly between them to be conflict-free. A non-Seg
    // effectful op (not summarized in the DAG) is treated conservatively as a
    // possible aliaser.
    ((i + 1)..j).all(|k| {
        let effect = &block.side_effects[k];
        match (&effect.kind, effect.result) {
            (SideEffectKind::Soac(EgirSoac::Seg { .. }), Some(_)) => {
                let Some(op_k) = effect.semantic_id else {
                    return false;
                };
                !oracle.conflicts(&op_k, &op_i) && !oracle.conflicts(&op_k, &op_j)
            }
            _ => effect.effects.is_none(),
        }
    })
}

/// Merge the Seg at `j` into the Seg at `i`, remove `j`, and rewire results.
/// Fields concatenate lane-wise; the operand vector is rebuilt in the canonical
/// `[inputs, init_accs, output_views]` order with `map_input_indices` rebased by
/// the input offset. The fused op produces a tuple of both operands' outputs, so
/// consumers of each original result are re-pointed at the corresponding fields.
fn fuse_pair(graph: &mut EGraph, block_id: BlockId, i: usize, j: usize) {
    let p = extract_seg(graph, block_id, i);
    let q = extract_seg(graph, block_id, j);

    let base = p.input_array_types.len();
    let mut input_array_types = p.input_array_types.clone();
    input_array_types.extend(q.input_array_types.iter().cloned());
    let mut input_elem_types = p.input_elem_types.clone();
    input_elem_types.extend(q.input_elem_types.iter().cloned());

    let mut map_bodies = p.map_bodies.clone();
    map_bodies.extend(q.map_bodies.iter().cloned());
    // Q's map funcs now read inputs shifted right by P's input count.
    let mut map_input_indices = p.map_input_indices.clone();
    map_input_indices
        .extend(q.map_input_indices.iter().map(|lane| lane.iter().map(|&k| k + base).collect()));

    let mut map_output_elem_types = p.map_output_elem_types.clone();
    map_output_elem_types.extend(q.map_output_elem_types.iter().cloned());
    let mut map_destinations = p.map_destinations.clone();
    map_destinations.extend(q.map_destinations.iter().cloned());
    let mut acc_destinations = p.acc_destinations.clone();
    acc_destinations.extend(q.acc_destinations.iter().cloned());

    let mut operators = p.kind.operators().to_vec();
    operators.extend(q.kind.operators().iter().cloned().map(|mut operator| {
        for input in &mut operator.input_indices {
            *input += base;
        }
        operator
    }));
    let kind = reify_seg_kind_operators(operators);

    let mut result_types = p.result_types.clone();
    result_types.extend(q.result_types.iter().cloned());

    let mut output_slots = p.output_slots.clone();
    output_slots.extend(q.output_slots.iter().copied());
    output_slots.sort_unstable();
    output_slots.dedup();

    let resources = SegResourceAccess::merge(&p.resources, &q.resources);

    // Rebuild `[inputs, init_accs, output_views]`. init_accs are the operators'
    // neutrals; output views are the trailing operands of each original op.
    let mut operands: SmallVec<[NodeId; 4]> = SmallVec::new();
    operands.extend(p.inputs.iter().copied());
    operands.extend(q.inputs.iter().copied());
    operands.extend(p.init_accs.iter().copied());
    operands.extend(q.init_accs.iter().copied());
    operands.extend(p.output_views.iter().copied());
    operands.extend(q.output_views.iter().copied());

    // Fused tuple result: fields are P's outputs then Q's.
    let tuple_ty = Type::Constructed(TypeName::Tuple(result_types.len()), result_types.clone());
    let fused_result = graph.alloc_side_effect_result(tuple_ty);

    // Re-point consumers: `Project(P.result, f)` → `Project(fused, f)`,
    // `Project(Q.result, f)` → `Project(fused, base_fields + f)`.
    reproject(graph, p.result, fused_result, 0, &p.result_types);
    reproject(
        graph,
        q.result,
        fused_result,
        p.result_types.len() as u32,
        &q.result_types,
    );

    let fused = EgirSoac::Seg {
        space: p.space.clone(),
        placement: p.placement,
        kind,
        map_bodies,
        input_array_types,
        input_elem_types,
        map_output_elem_types,
        map_input_indices,
        map_destinations,
        acc_destinations,
        result_types,
        output_slots,
        resources,
    };

    let block = &mut graph.skeleton.blocks[block_id];
    // Splice the effect chain: the fused op spans from P's input token to Q's
    // output token (P precedes Q in the block), so any downstream effect that
    // read Q's output stays connected once Q is removed.
    let fused_effects = match (block.side_effects[i].effects, block.side_effects[j].effects) {
        (Some((p_in, _)), Some((_, q_out))) => Some((p_in, q_out)),
        (Some(effects), None) | (None, Some(effects)) => Some(effects),
        (None, None) => None,
    };
    block.side_effects[i].kind = SideEffectKind::Soac(fused);
    block.side_effects[i].operand_nodes = operands;
    block.side_effects[i].result = Some(fused_result);
    block.side_effects[i].effects = fused_effects;
    block.side_effects.remove(j);
}

/// Everything `fuse_pair` needs from one Seg side-effect, cloned so the two can
/// be read independently while the block is rebuilt.
struct SegParts {
    space: crate::egir::types::SegSpace,
    placement: SegPlacement,
    kind: SegOpKind,
    map_bodies: Vec<SegBody>,
    input_array_types: Vec<Type<TypeName>>,
    input_elem_types: Vec<Type<TypeName>>,
    map_output_elem_types: Vec<Type<TypeName>>,
    map_input_indices: Vec<Vec<usize>>,
    map_destinations: Vec<SoacDestination>,
    acc_destinations: Vec<SoacDestination>,
    result_types: Vec<Type<TypeName>>,
    output_slots: Vec<usize>,
    resources: Vec<SegResourceAccess>,
    result: NodeId,
    inputs: Vec<NodeId>,
    init_accs: Vec<NodeId>,
    output_views: Vec<NodeId>,
}

fn extract_seg(graph: &EGraph, block_id: BlockId, idx: usize) -> SegParts {
    let effect = &graph.skeleton.blocks[block_id].side_effects[idx];
    let SideEffectKind::Soac(EgirSoac::Seg {
        space,
        placement,
        kind,
        map_bodies,
        input_array_types,
        input_elem_types,
        map_output_elem_types,
        map_input_indices,
        map_destinations,
        acc_destinations,
        result_types,
        output_slots,
        resources,
        ..
    }) = &effect.kind
    else {
        unreachable!("extract_seg on non-Seg");
    };
    let n_inputs = input_array_types.len();
    let n_accs = kind.operators().len();
    let inputs = effect.operand_nodes[..n_inputs].to_vec();
    let init_accs = effect.operand_nodes[n_inputs..n_inputs + n_accs].to_vec();
    let output_views = effect.operand_nodes[n_inputs + n_accs..].to_vec();
    SegParts {
        space: space.clone(),
        placement: *placement,
        kind: kind.clone(),
        map_bodies: map_bodies.clone(),
        input_array_types: input_array_types.clone(),
        input_elem_types: input_elem_types.clone(),
        map_output_elem_types: map_output_elem_types.clone(),
        map_input_indices: map_input_indices.clone(),
        map_destinations: map_destinations.clone(),
        acc_destinations: acc_destinations.clone(),
        result_types: result_types.clone(),
        output_slots: output_slots.clone(),
        resources: resources.clone(),
        result: effect.result.expect("fusable Seg has a result"),
        inputs,
        init_accs,
        output_views,
    }
}

/// Rewrite every `Project(old_result, f)` to `Project(new_result, f + offset)`,
/// and rebuild any whole-result use with the old tuple type.
fn reproject(
    graph: &mut EGraph,
    old_result: NodeId,
    new_result: NodeId,
    offset: u32,
    field_types: &[Type<TypeName>],
) {
    let projects: Vec<(NodeId, u32)> = graph
        .nodes
        .iter()
        .filter_map(|(nid, node)| match node {
            crate::egir::types::ENode::Pure {
                op: PureOp::Project { index },
                operands,
            } if operands.first() == Some(&old_result) => Some((nid, *index)),
            _ => None,
        })
        .collect();
    for (project_nid, index) in projects {
        graph.update_pure_node(project_nid, |op, operands| {
            *op = PureOp::Project {
                index: index + offset,
            };
            operands[0] = new_result;
        });
    }
    // Preserve the old tuple type for whole-result users. Pointing those users
    // at the larger fused tuple would silently change their argument type.
    let fields: SmallVec<[NodeId; 4]> = field_types
        .iter()
        .enumerate()
        .map(|(index, ty)| {
            graph.intern_pure(
                PureOp::Project {
                    index: offset + index as u32,
                },
                smallvec::smallvec![new_result],
                ty.clone(),
            )
        })
        .collect();
    let old_ty = graph.types[&old_result].clone();
    let rebuilt = graph.intern_pure(PureOp::Tuple(field_types.len()), fields, old_ty);
    graph_ops::replace_all_references(graph, old_result, rebuilt);
}
