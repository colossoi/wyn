//! Same-space horizontal fusion for canonical Scremas.
//!
//! Sibling Scremas are fused by composing their complete pre/post lambdas.  The
//! wrapper lambdas perform the canonical scan/reduction/map partitioning and
//! result reordering.

use crate::egir;
use polytype::Type;
use smallvec::SmallVec;

use super::graph_and_span;
use super::screma as fusion_screma;
use super::space::seg_space_fusable;
use super::support;
use crate::ast::{Span, TypeName};
use crate::egir::graph_ops;
use crate::egir::ir::{splice_effect_tokens, BodySite};
use crate::egir::program::{CoreProgramData, Func, OutputSlotId, ProgramIdentities};
use crate::egir::reify::Segmented;
use crate::egir::semantic_graph::SemanticGraph;
use crate::egir::soac::screma;
use crate::egir::types::{
    EGraph, ResultBinding, SegResourceAccess, Semantic, SideEffectKind, Soac, SoacEffect, ValueId,
};
use crate::flow::BlockId;
use crate::LookupMap;

#[derive(Clone, Copy)]
pub(super) struct Candidate {
    site: BodySite,
    block: BlockId,
    left: usize,
    right: usize,
}

pub(super) fn analyze(inner: &Segmented, oracle: &SemanticGraph) -> Option<Candidate> {
    super::bodies(inner).find_map(|(site, graph, _)| {
        find_in_graph(graph, oracle).map(|(block, left, right)| Candidate {
            site,
            block,
            left,
            right,
        })
    })
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
        (&left_effect.kind, left_effect.result.as_ref())
    else {
        return false;
    };
    let (SideEffectKind::Soac(SoacEffect(right_id, Soac::Screma(right_op))), Some(_)) =
        (&right_effect.kind, right_effect.result.as_ref())
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

    let left_inputs = &left_effect.operands[..left_op.inputs.len()];
    let right_inputs = &right_effect.operands[..right_op.inputs.len()];
    let shared_input = left_inputs.iter().any(|node| right_inputs.contains(node));
    // Horizontal fusion does not eliminate an intermediate array. Its reliable
    // benefit is sharing one traversal of an actual input, which is also the
    // candidate rule used by Futhark's SOAC fusion graph. Equal extents alone
    // merely couple independent kernels and may increase resource pressure.
    if !shared_input || !seg_space_fusable(left_space, right_space) {
        return false;
    }

    if oracle.reachable_between(left_id, right_id) || oracle.reachable_between(right_id, left_id) {
        return false;
    }

    ((left + 1)..right).all(|index| {
        let effect = &block.side_effects[index];
        match (&effect.kind, effect.result.as_ref()) {
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
    let mut identities = inner.data.identities.clone();
    let plan = build_plan(&inner, &mut identities, &scope, span, &outer_types, left, right);
    let synthesized = plan.synthesized.clone();

    let rebuilt = inner.rewrite_body(candidate.site, |body| {
        let rewrite =
            |graph: &mut EGraph| apply_plan(graph, candidate.block, candidate.left, candidate.right, &plan);
        support::rewrite_body_graph_with_entry(body, rewrite, |entry, replacements| {
            support::replace_route_values(entry, &replacements);
        })
    });
    rebuilt.extend_functions(synthesized).map_data(|data| CoreProgramData {
        identities: identities,
        ..data
    })
}

#[derive(Clone)]
pub(super) struct ScremaParts {
    pub(super) id: egir::program::SemanticOpId,
    pub(super) op: screma::Op<Semantic>,
    pub(super) space: egir::types::SegSpace,
    pub(super) placement: screma::Placement,
    pub(super) output_slots: Vec<OutputSlotId>,
    pub(super) resources: Vec<SegResourceAccess>,
    pub(super) results: Vec<ResultBinding<Type<TypeName>>>,
    pub(super) result_types: Vec<Type<TypeName>>,
    pub(super) input_nodes: Vec<ValueId>,
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
    let operands = screma::ScremaOperands::decode(op, &effect.operands, effect.result.as_ref())
        .expect("fusable Screma has invalid operands or results");
    let results = operands.result_fields();
    let result_types = results.iter().map(|result| result.ty().clone()).collect();

    let input_count = op.inputs.len();
    let input_nodes = effect.operands[..input_count]
        .iter()
        .map(|operand| operand.value().expect("Screma inputs are values or views"))
        .collect();
    ScremaParts {
        id: *id,
        op: op.clone(),
        space: space.clone(),
        placement: *placement,
        output_slots: output_slots.clone(),
        resources: resources.clone(),
        results,
        result_types,
        input_nodes,
    }
}

#[derive(Clone)]
struct FusionPlan {
    id: egir::program::SemanticOpId,
    op: screma::Op<Semantic>,
    operands: SmallVec<[ValueId; 4]>,
    result_types: Vec<Type<TypeName>>,
    left_results: Vec<ResultBinding<Type<TypeName>>>,
    right_results: Vec<ResultBinding<Type<TypeName>>>,
    left_mapping: Vec<usize>,
    right_mapping: Vec<usize>,
    synthesized: Vec<Func<Semantic>>,
}

#[allow(clippy::too_many_arguments)]
fn build_plan(
    inner: &Segmented,
    identities: &mut ProgramIdentities,
    scope: &str,
    span: Span,
    outer_types: &LookupMap<ValueId, Type<TypeName>>,
    left: ScremaParts,
    right: ScremaParts,
) -> FusionPlan {
    let mut context = fusion_screma::Context {
        program: inner,
        identities,
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
    for (fused_field, origin) in normalized.outputs.iter().copied().enumerate() {
        let (source_field, source_state, source_types, mapping) = match origin {
            fusion_screma::OutputOrigin::Producer(field) => (
                field,
                &left.op.result_state,
                &left.result_types,
                &mut left_mapping,
            ),
            fusion_screma::OutputOrigin::Consumer(field) => (
                field,
                &right.op.result_state,
                &right.result_types,
                &mut right_mapping,
            ),
        };
        mapping[source_field] = fused_field;
        result_state.push(source_state[source_field].clone());
        result_types.push(source_types[source_field].clone());
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

    FusionPlan {
        id: left.id,
        op,
        operands,
        result_types,
        left_results: left.results,
        right_results: right.results,
        left_mapping,
        right_mapping,
        synthesized: normalized.synthesized,
    }
}
fn apply_plan(
    graph: &mut EGraph,
    block: BlockId,
    left: usize,
    right: usize,
    plan: &FusionPlan,
) -> Vec<(ValueId, ValueId)> {
    let tuple = Type::Constructed(
        TypeName::Tuple(plan.result_types.len()),
        plan.result_types.clone(),
    );
    let result = graph_ops::alloc_by_value_effect_result(graph, tuple);
    let fused_results = result.top_level_fields();
    let mut replacements = rebind_fields(graph, &plan.left_results, &fused_results, &plan.left_mapping);
    replacements.extend(rebind_fields(
        graph,
        &plan.right_results,
        &fused_results,
        &plan.right_mapping,
    ));

    let operands = plan.operands.iter().map(|operand| graph.operand_ref(*operand)).collect();
    let block = &mut graph.skeleton.blocks[block];
    let effects = splice_effect_tokens(
        block.side_effects[left].effects,
        block.side_effects[right].effects,
    );
    block.side_effects[left].kind =
        SideEffectKind::Soac(SoacEffect(plan.id, Soac::Screma(plan.op.clone())));
    block.side_effects[left].operands = operands;
    block.side_effects[left].result = Some(result);
    block.side_effects[left].effects = effects;
    block.side_effects.remove(right);
    replacements
}

pub(super) fn rebind_fields(
    graph: &mut EGraph,
    old_results: &[ResultBinding<Type<TypeName>>],
    new_results: &[ResultBinding<Type<TypeName>>],
    mapping: &[usize],
) -> Vec<(ValueId, ValueId)> {
    assert_eq!(old_results.len(), mapping.len());
    let mut replacements = Vec::new();
    for (old, field) in old_results.iter().zip(mapping) {
        replacements.extend(
            graph_ops::rebind_result_value_references(graph, old, &new_results[*field])
                .expect("fused Screma result fields must have the same by-value shape"),
        );
    }
    replacements
}
