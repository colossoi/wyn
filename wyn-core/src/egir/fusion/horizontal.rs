//! Same-space horizontal fusion for canonical Scremas.
//!
//! Sibling Scremas are fused by composing their complete pre/post lambdas.  The
//! wrapper lambdas perform the canonical scan/reduction/map partitioning and
//! result reordering; no legacy lane graph is reconstructed.

use polytype::Type;
use smallvec::{smallvec, SmallVec};

use super::graph_and_span;
use super::screma as fusion_screma;
use super::space::seg_space_fusable;
use crate::ast::{Span, TypeName};
use crate::egir::graph_ops;
use crate::egir::ir::{splice_effect_tokens, Body, BodySite};
use crate::egir::program::{CoreProgramData, OutputSlotId, RegionInterner, SemanticFunc};
use crate::egir::reify::Segmented;
use crate::egir::semantic_graph::SemanticGraph;
use crate::egir::soac::screma;
use crate::egir::types::{
    EGraph, NodeId, PureOp, SegResourceAccess, Semantic, SideEffectKind, Soac, SoacEffect,
};
use crate::flow::BlockId;
use crate::LookupMap;

#[derive(Clone, Copy)]
pub(crate) struct Candidate {
    site: BodySite,
    block: BlockId,
    left: usize,
    right: usize,
}

pub(super) fn analyze(inner: &Segmented, oracle: &SemanticGraph) -> Option<Candidate> {
    for (index, entry) in inner.entry_points.iter().enumerate() {
        if let Some((block, left, right)) = find_in_graph(&entry.graph, oracle) {
            return Some(Candidate {
                site: BodySite::Entry(index),
                block,
                left,
                right,
            });
        }
    }
    for function in &inner.functions {
        if let Some((block, left, right)) = find_in_graph(&function.graph, oracle) {
            return Some(Candidate {
                site: BodySite::Function(function.region),
                block,
                left,
                right,
            });
        }
    }
    None
}

fn find_in_graph(graph: &EGraph, oracle: &SemanticGraph) -> Option<(BlockId, usize, usize)> {
    for (block_id, block) in &graph.skeleton.blocks {
        let scremas = (0..block.side_effects.len())
            .filter(|&index| is_segmented_screma(&block.side_effects[index].kind))
            .collect::<Vec<_>>();
        for left in 0..scremas.len() {
            for right in (left + 1)..scremas.len() {
                let pair = (scremas[left], scremas[right]);
                if sibling_fusable(graph, block_id, pair.0, pair.1, oracle) {
                    return Some((block_id, pair.0, pair.1));
                }
            }
        }
    }
    None
}

fn is_segmented_screma(kind: &SideEffectKind) -> bool {
    matches!(
        kind,
        SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op)))
            if matches!(op.semantic_state(), screma::SemanticState::Segmented { .. })
    )
}

fn sibling_fusable(
    graph: &EGraph,
    block_id: BlockId,
    left: usize,
    right: usize,
    oracle: &SemanticGraph,
) -> bool {
    let block = &graph.skeleton.blocks[block_id];
    let left_effect = &block.side_effects[left];
    let right_effect = &block.side_effects[right];
    let (SideEffectKind::Soac(SoacEffect(left_id, Soac::Screma(left_op))), Some(_)) =
        (&left_effect.kind, left_effect.result)
    else {
        return false;
    };
    let (SideEffectKind::Soac(SoacEffect(right_id, Soac::Screma(right_op))), Some(_)) =
        (&right_effect.kind, right_effect.result)
    else {
        return false;
    };
    let screma::SemanticState::Segmented {
        space: left_space,
        placement: _,
        ..
    } = left_op.semantic_state()
    else {
        return false;
    };
    let screma::SemanticState::Segmented {
        space: right_space,
        placement: _,
        ..
    } = right_op.semantic_state()
    else {
        return false;
    };

    let left_inputs = &left_effect.operand_nodes[..left_op.inputs.len()];
    let right_inputs = &right_effect.operand_nodes[..right_op.inputs.len()];
    let shared_input = left_inputs.iter().any(|node| right_inputs.contains(node));
    let shared_size = left_op.inputs.iter().any(|left| {
        crate::types::array_size(&left.array).is_some_and(|left_size| {
            right_op
                .inputs
                .iter()
                .filter_map(|right| crate::types::array_size(&right.array))
                .any(|right_size| right_size == left_size)
        })
    });
    if !seg_space_fusable(left_space, right_space) && !shared_input && !shared_size {
        return false;
    }

    let has_scan = !left_op.form.scans.is_empty() || !right_op.form.scans.is_empty();
    let has_reduce = !left_op.form.reductions.is_empty() || !right_op.form.reductions.is_empty();
    if has_scan
        && has_reduce
        && (oracle.value_consumer_count(left_id) != 0 || oracle.value_consumer_count(right_id) != 0)
    {
        return false;
    }

    if oracle.reachable_between(left_id, right_id) || oracle.reachable_between(right_id, left_id) {
        return false;
    }

    ((left + 1)..right).all(|index| {
        let effect = &block.side_effects[index];
        match (&effect.kind, effect.result) {
            (SideEffectKind::Soac(SoacEffect(id, Soac::Screma(_))), Some(_)) => {
                !oracle.conflicts(id, left_id) && !oracle.conflicts(id, right_id)
            }
            _ => effect.effects.is_none(),
        }
    })
}

pub(super) fn apply(inner: Segmented, candidate: Candidate) -> Segmented {
    let (graph, span, scope) = graph_and_span(&inner, candidate.site);
    let outer_types =
        graph.nodes.iter().map(|(node, data)| (node, data.ty.clone())).collect::<LookupMap<_, _>>();
    let left = extract_screma(graph, candidate.block, candidate.left);
    let right = extract_screma(graph, candidate.block, candidate.right);
    let mut interner = inner.data.region_interner.clone();
    let plan = build_plan(&inner, &mut interner, &scope, span, &outer_types, left, right);
    let synthesized = plan.synthesized.clone();

    let rebuilt = inner.rewrite_body(candidate.site, |body| {
        let rewrite = |graph: &mut EGraph| {
            apply_plan(graph, candidate.block, candidate.left, candidate.right, &plan);
        };
        match body {
            Body::Entry(mut entry) => {
                rewrite(&mut entry.graph);
                Body::Entry(entry)
            }
            Body::Function(mut function) => {
                rewrite(&mut function.graph);
                Body::Function(function)
            }
            Body::Constant(_) => unreachable!("horizontal fusion never targets constants"),
        }
    });
    rebuilt.extend_functions(synthesized).map_data(|data| CoreProgramData {
        region_interner: interner,
        ..data
    })
}

#[derive(Clone)]
pub(super) struct ScremaParts {
    pub(super) id: crate::egir::program::SemanticOpId,
    pub(super) op: screma::Op<Semantic>,
    pub(super) space: crate::egir::types::SegSpace,
    pub(super) placement: screma::Placement,
    pub(super) output_slots: Vec<OutputSlotId>,
    pub(super) resources: Vec<SegResourceAccess>,
    pub(super) result: NodeId,
    pub(super) result_types: Vec<Type<TypeName>>,
    pub(super) input_nodes: Vec<NodeId>,
    pub(super) output_nodes: Vec<Option<NodeId>>,
}

pub(super) fn extract_screma(graph: &EGraph, block: BlockId, index: usize) -> ScremaParts {
    let effect = &graph.skeleton.blocks[block].side_effects[index];
    let SideEffectKind::Soac(SoacEffect(id, Soac::Screma(op))) = &effect.kind else {
        unreachable!("horizontal fusion selected a non-Screma");
    };
    let screma::SemanticState::Segmented {
        space,
        placement,
        output_slots,
        resources,
    } = op.semantic_state()
    else {
        unreachable!("horizontal fusion selected a serial Screma");
    };
    let result = effect.result.expect("fusable Screma has no result");
    let Type::Constructed(TypeName::Tuple(arity), result_types) = graph.nodes[result].ty.clone() else {
        unreachable!("Screma result is not a tuple");
    };
    assert_eq!(arity, op.result_count());
    assert_eq!(result_types.len(), op.result_count());

    let input_count = op.inputs.len();
    let input_nodes = effect.operand_nodes[..input_count].to_vec();
    let mut output_operands = effect.operand_nodes[input_count..].iter().copied();
    let output_nodes = (0..op.result_count())
        .map(|field| {
            op.destination(field)
                .filter(|destination| destination.is_output_view())
                .map(|_| output_operands.next().expect("missing Screma output-view operand"))
        })
        .collect::<Vec<_>>();
    assert!(output_operands.next().is_none());

    ScremaParts {
        id: *id,
        op: op.clone(),
        space: space.clone(),
        placement: *placement,
        output_slots: output_slots.clone(),
        resources: resources.clone(),
        result,
        result_types,
        input_nodes,
        output_nodes,
    }
}

#[derive(Clone)]
struct FusionPlan {
    id: crate::egir::program::SemanticOpId,
    op: screma::Op<Semantic>,
    operands: SmallVec<[NodeId; 4]>,
    result_types: Vec<Type<TypeName>>,
    left_result: NodeId,
    right_result: NodeId,
    left_mapping: Vec<usize>,
    right_mapping: Vec<usize>,
    left_result_types: Vec<Type<TypeName>>,
    right_result_types: Vec<Type<TypeName>>,
    synthesized: Vec<SemanticFunc>,
}

#[allow(clippy::too_many_arguments)]
fn build_plan(
    inner: &Segmented,
    interner: &mut RegionInterner,
    scope: &str,
    span: Span,
    outer_types: &LookupMap<NodeId, Type<TypeName>>,
    left: ScremaParts,
    right: ScremaParts,
) -> FusionPlan {
    let mut context = fusion_screma::Context {
        program: inner,
        interner,
        scope,
        span,
        outer_types,
    };
    let normalized = fusion_screma::fuse_horizontal(
        &mut context,
        fusion_screma::Source {
            input_nodes: &left.input_nodes,
            inputs: &left.op.inputs,
            form: &left.op.form,
        },
        fusion_screma::Source {
            input_nodes: &right.input_nodes,
            inputs: &right.op.inputs,
            form: &right.op.form,
        },
    );

    let mut left_mapping = vec![usize::MAX; left.result_types.len()];
    let mut right_mapping = vec![usize::MAX; right.result_types.len()];
    let mut result_state = Vec::with_capacity(normalized.outputs.len());
    let mut result_types = Vec::with_capacity(normalized.outputs.len());
    let mut output_nodes = Vec::with_capacity(normalized.outputs.len());
    for (fused_field, origin) in normalized.outputs.iter().copied().enumerate() {
        let (source_field, source_state, source_types, source_outputs, mapping) = match origin {
            fusion_screma::OutputOrigin::Producer(field) => (
                field,
                &left.op.result_state,
                &left.result_types,
                &left.output_nodes,
                &mut left_mapping,
            ),
            fusion_screma::OutputOrigin::Consumer(field) => (
                field,
                &right.op.result_state,
                &right.result_types,
                &right.output_nodes,
                &mut right_mapping,
            ),
        };
        mapping[source_field] = fused_field;
        result_state.push(source_state[source_field].clone());
        result_types.push(source_types[source_field].clone());
        output_nodes.push(source_outputs[source_field]);
    }
    debug_assert!(left_mapping.iter().all(|field| *field != usize::MAX));
    debug_assert!(right_mapping.iter().all(|field| *field != usize::MAX));

    let mut output_slots = left.output_slots.clone();
    output_slots.extend(right.output_slots.iter().copied());
    output_slots.sort_unstable();
    output_slots.dedup();
    let resources = SegResourceAccess::merge(&left.resources, &right.resources);
    let op = screma::Op {
        inputs: normalized.inputs,
        form: normalized.form,
        result_state,
        state: screma::SemanticState::Segmented {
            space: left.space,
            placement: if left.placement == screma::Placement::Kernel
                || right.placement == screma::Placement::Kernel
            {
                screma::Placement::Kernel
            } else {
                screma::Placement::LaneLocal
            },
            output_slots,
            resources,
        },
    };
    debug_assert!(
        op.validate().is_ok(),
        "invalid horizontally fused Screma: {:?}",
        op.validate()
    );

    let mut operands = SmallVec::new();
    operands.extend(normalized.input_nodes);
    operands.extend(output_nodes.into_iter().flatten());

    FusionPlan {
        id: left.id,
        op,
        operands,
        result_types,
        left_result: left.result,
        right_result: right.result,
        left_mapping,
        right_mapping,
        left_result_types: left.result_types,
        right_result_types: right.result_types,
        synthesized: normalized.synthesized,
    }
}
fn apply_plan(graph: &mut EGraph, block: BlockId, left: usize, right: usize, plan: &FusionPlan) {
    let tuple = Type::Constructed(
        TypeName::Tuple(plan.result_types.len()),
        plan.result_types.clone(),
    );
    let fused_result = graph.alloc_side_effect_result(tuple);
    reproject_fields(
        graph,
        plan.left_result,
        fused_result,
        &plan.left_mapping,
        &plan.left_result_types,
    );
    reproject_fields(
        graph,
        plan.right_result,
        fused_result,
        &plan.right_mapping,
        &plan.right_result_types,
    );

    let block = &mut graph.skeleton.blocks[block];
    let effects = splice_effect_tokens(
        block.side_effects[left].effects,
        block.side_effects[right].effects,
    );
    block.side_effects[left].kind =
        SideEffectKind::Soac(SoacEffect(plan.id, Soac::Screma(plan.op.clone())));
    block.side_effects[left].operand_nodes = plan.operands.clone();
    block.side_effects[left].result = Some(fused_result);
    block.side_effects[left].effects = effects;
    block.side_effects.remove(right);
}

pub(super) fn reproject_fields(
    graph: &mut EGraph,
    old_result: NodeId,
    new_result: NodeId,
    mapping: &[usize],
    field_types: &[Type<TypeName>],
) {
    let projects = graph
        .nodes
        .iter()
        .filter_map(|(node, data)| match &data.kind {
            crate::egir::types::ENode::Pure {
                op: PureOp::Project { index },
                operands,
            } if operands.first() == Some(&old_result) => Some((node, *index as usize)),
            _ => None,
        })
        .collect::<Vec<_>>();
    for (project, field) in projects {
        graph.update_pure_node(project, |op, operands| {
            *op = PureOp::Project {
                index: mapping[field] as u32,
            };
            operands[0] = new_result;
        });
    }

    let fields = field_types
        .iter()
        .enumerate()
        .map(|(field, ty)| {
            graph.intern_pure(
                PureOp::Project {
                    index: mapping[field] as u32,
                },
                smallvec![new_result],
                ty.clone(),
                None,
            )
        })
        .collect::<SmallVec<[NodeId; 4]>>();
    let old_type = graph.nodes[old_result].ty.clone();
    let rebuilt = graph.intern_pure(PureOp::Tuple(field_types.len()), fields, old_type, None);
    graph_ops::replace_all_references(graph, old_result, rebuilt);
}
