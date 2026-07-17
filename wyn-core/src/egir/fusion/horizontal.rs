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
use crate::egir::ir::splice_effect_tokens;
use crate::egir::program::{OutputSlotId, SemanticProgram};
use crate::egir::semantic_graph::SemanticGraph;
use crate::egir::soac::screma;
use crate::egir::types::{
    EGraph, NodeId, PureOp, SegResourceAccess, SideEffectKind, Soac, SoacEffect, SoacInputType,
};
use crate::flow::BlockId;

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
    matches!(kind, SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) if matches!(op.semantic_state(), screma::SemanticState::Segmented { .. }))
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
    let (SideEffectKind::Soac(SoacEffect(op_i_id, Soac::Screma(op_i))), Some(_)) =
        (&effect_i.kind, effect_i.result)
    else {
        return false;
    };
    let screma::SemanticState::Segmented {
        space: sp_i,
        placement: pl_i,
        ..
    } = op_i.semantic_state()
    else {
        return false;
    };
    let (SideEffectKind::Soac(SoacEffect(op_j_id, Soac::Screma(op_j))), Some(_)) =
        (&effect_j.kind, effect_j.result)
    else {
        return false;
    };
    let screma::SemanticState::Segmented {
        space: sp_j,
        placement: pl_j,
        ..
    } = op_j.semantic_state()
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
    let shared_size = op_i
        .lanes()
        .inputs
        .first()
        .and_then(|input| crate::types::array_size(&input.array))
        .zip(op_j.lanes().inputs.first().and_then(|input| crate::types::array_size(&input.array)))
        .is_some_and(|(left, right)| left == right);
    let symbolic_domain_matches = shared_input || shared_size;
    if pl_i != pl_j || (!seg_space_fusable(sp_i, sp_j) && !symbolic_domain_matches) {
        return false;
    }
    let (mut has_reduce, mut has_scan) = (false, false);
    for op in [op_i, op_j] {
        for index in 0..op.operators().len() {
            has_scan |= op.is_scan(index);
            has_reduce |= !op.is_scan(index);
        }
    }
    if has_reduce && has_scan {
        // The target scheduler has separate reduction and scan phase models.
        // Keep mixed siblings independent until a joint physical recipe exists.
        return false;
    }
    let (op_i, op_j) = (*op_i_id, *op_j_id);
    // A value edge either way makes them a producer/consumer chain (handled by
    // vertical EGIR fusion), never fusable siblings.
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
            (SideEffectKind::Soac(SoacEffect(op_k, Soac::Screma(_))), Some(_)) => {
                !oracle.conflicts(&op_k, &op_i) && !oracle.conflicts(&op_k, &op_j)
            }
            _ => effect.effects.is_none(),
        }
    })
}

/// Merge the Seg at `j` into the Seg at `i`, remove `j`, and rewire results.
/// Fields concatenate lane-wise; the operand vector is rebuilt in the canonical
/// `[inputs, output_views]` order with `map_input_indices` rebased by
/// the input offset. The fused op produces a tuple of both operands' outputs, so
/// consumers of each original result are re-pointed at the corresponding fields.
fn fuse_pair(graph: &mut EGraph, block_id: BlockId, i: usize, j: usize) {
    let p = extract_seg(graph, block_id, i);
    let q = extract_seg(graph, block_id, j);

    let base = p.lanes.inputs.len();
    let mut raw_inputs = p.inputs.clone();
    raw_inputs.extend(q.inputs.iter().copied());
    let mut raw_array_types: Vec<Type<TypeName>> =
        p.lanes.inputs.iter().map(|input| input.array.clone()).collect();
    raw_array_types.extend(q.lanes.inputs.iter().map(|input| input.array.clone()));
    let mut raw_elem_types: Vec<Type<TypeName>> =
        p.lanes.inputs.iter().map(SoacInputType::element).collect();
    raw_elem_types.extend(q.lanes.inputs.iter().map(SoacInputType::element));
    let (inputs, input_array_types, _input_elem_types, input_remap) =
        super::deduplicate_array_inputs(raw_inputs, raw_array_types, raw_elem_types);
    let input_types = input_array_types.into_iter().map(|array| SoacInputType { array }).collect();

    let mut maps: Vec<screma::Map> = p
        .lanes
        .maps
        .iter()
        .cloned()
        .map(|mut map| {
            for input in &mut map.input_indices {
                *input = screma::InputId(input_remap[input.index()]);
            }
            map
        })
        .collect();
    maps.extend(q.lanes.maps.iter().cloned().map(|mut map| {
        for input in &mut map.input_indices {
            *input = screma::InputId(input_remap[base + input.index()]);
        }
        map
    }));

    let mut operators = p.operators.clone();
    for operator in &mut operators {
        for input in &mut operator.input_indices {
            *input = screma::InputId(input_remap[input.index()]);
        }
    }
    operators.extend(q.operators.iter().cloned().map(|mut operator| {
        for input in &mut operator.input_indices {
            *input = screma::InputId(input_remap[base + input.index()]);
        }
        operator
    }));
    let has_scan = p.has_scan || q.has_scan;

    let p_result_types = p.result_types();
    let q_result_types = q.result_types();
    let mut result_types = p_result_types.clone();
    result_types.extend(q_result_types.iter().cloned());

    let mut output_slots = p.output_slots.clone();
    output_slots.extend(q.output_slots.iter().copied());
    output_slots.sort_unstable();
    output_slots.dedup();

    let resources = SegResourceAccess::merge(&p.resources, &q.resources);

    // Rebuild `[inputs, output_views]`. Operator neutrals stay in `kind`;
    // output views are the trailing operands of each original op.
    let mut operands: SmallVec<[NodeId; 4]> = SmallVec::new();
    operands.extend(inputs);
    operands.extend(p.output_views.iter().copied());
    operands.extend(q.output_views.iter().copied());

    // Fused tuple result: fields are P's outputs then Q's.
    let tuple_ty = Type::Constructed(TypeName::Tuple(result_types.len()), result_types.clone());
    let fused_result = graph.alloc_side_effect_result(tuple_ty);

    // Re-point consumers: `Project(P.result, f)` → `Project(fused, f)`,
    // `Project(Q.result, f)` → `Project(fused, base_fields + f)`.
    reproject(graph, p.result, fused_result, 0, &p_result_types);
    reproject(
        graph,
        q.result,
        fused_result,
        p_result_types.len() as u32,
        &q_result_types,
    );

    let state = screma::SemanticState::Segmented {
        space: p.space.clone(),
        placement: p.placement,
        output_slots,
        resources,
    };
    let lanes = screma::Lanes {
        inputs: input_types,
        maps,
    };
    let fused = Soac::Screma(match screma::NonEmpty::from_vec(operators) {
        None => screma::Op::Map { lanes, state },
        Some(operators) if has_scan => screma::Op::Scan {
            lanes,
            operators,
            state,
        },
        Some(operators) => screma::Op::Reduce {
            lanes,
            operators,
            state,
        },
    });

    let block = &mut graph.skeleton.blocks[block_id];
    // Splice the effect chain: the fused op spans from P's input token to Q's
    // output token (P precedes Q in the block), so any downstream effect that
    // read Q's output stays connected once Q is removed.
    let fused_effects = splice_effect_tokens(block.side_effects[i].effects, block.side_effects[j].effects);
    block.side_effects[i].kind = SideEffectKind::Soac(SoacEffect(p.id, fused));
    block.side_effects[i].operand_nodes = operands;
    block.side_effects[i].result = Some(fused_result);
    block.side_effects[i].effects = fused_effects;
    block.side_effects.remove(j);
}

/// Everything `fuse_pair` needs from one Seg side-effect, cloned so the two can
/// be read independently while the block is rebuilt.
struct SegParts {
    id: crate::egir::program::SemanticOpId,
    space: crate::egir::types::SegSpace,
    placement: screma::Placement,
    lanes: screma::Lanes,
    operators: Vec<screma::Operator>,
    has_scan: bool,
    output_slots: Vec<OutputSlotId>,
    resources: Vec<SegResourceAccess>,
    result: NodeId,
    inputs: Vec<NodeId>,
    output_views: Vec<NodeId>,
}

impl SegParts {
    fn result_types(&self) -> Vec<Type<TypeName>> {
        self.lanes
            .maps
            .iter()
            .map(|map| map.result_type.clone())
            .chain(self.operators.iter().map(|operator| operator.result_type.clone()))
            .collect()
    }
}

fn extract_seg(graph: &EGraph, block_id: BlockId, idx: usize) -> SegParts {
    let effect = &graph.skeleton.blocks[block_id].side_effects[idx];
    let SideEffectKind::Soac(SoacEffect(id, Soac::Screma(op))) = &effect.kind else {
        unreachable!("extract_seg on non-Seg");
    };
    let screma::SemanticState::Segmented {
        space,
        placement,
        output_slots,
        resources,
    } = op.semantic_state()
    else {
        unreachable!("extract_seg on non-Seg");
    };
    let n_inputs = op.lanes().inputs.len();
    let inputs = effect.operand_nodes[..n_inputs].to_vec();
    let output_views = effect.operand_nodes[n_inputs..].to_vec();
    SegParts {
        id: *id,
        space: space.clone(),
        placement: *placement,
        lanes: op.lanes().clone(),
        operators: op.operators().into_iter().cloned().collect(),
        has_scan: (0..op.operators().len()).any(|index| op.is_scan(index)),
        output_slots: output_slots.clone(),
        resources: resources.clone(),
        result: effect.result.expect("fusable Seg has a result"),
        inputs,
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
                None,
            )
        })
        .collect();
    let old_ty = graph.types[&old_result].clone();
    let rebuilt = graph.intern_pure(PureOp::Tuple(field_types.len()), fields, old_ty, None);
    graph_ops::replace_all_references(graph, old_result, rebuilt);
}
