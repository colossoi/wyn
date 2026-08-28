//! Fuse a non-escaping Filter into scalar Screma reductions.
//!
//! The Filter's map and predicate become a canonical Screma pre-lambda.  For
//! each reduction component the pre-lambda yields the consumer value when the
//! predicate holds and that reduction's neutral value otherwise.  The original
//! associative reduction operators remain unchanged.  `length` is represented
//! by one additional sum reduction over `1`/`0` values.

use crate::ast;
use crate::egir;
use crate::types;
use std::collections::HashSet;

use polytype::Type;
use smallvec::smallvec;

use super::{capture_types, graph_and_span, support, FusionEffect};
use crate::ast::TypeName;
use crate::builtins::catalog;
use crate::egir::graph_ops;
use crate::egir::ir::{splice_effect_tokens, RealizedOutputRoute};
use crate::egir::program::{Func, OutputWriter, ProgramIdentities, SemanticProgramData};
use crate::egir::reify::Segmented;
use crate::egir::semantic_graph::SemanticGraph;
use crate::egir::soac::{filter, lambda as lambda_ops, screma};
use crate::egir::types::{
    EGraph, PureOp, ResultBinding, SegResourceAccess, SegSpace, Semantic, SideEffect, SideEffectKind,
    SkeletonTerminator, Soac, SoacEffect, ValueId, ValueKind,
};
use crate::flow::{BlockId, ControlHeader};
use crate::op::BinaryOperator;
use crate::LookupMap;

#[derive(Clone)]
pub(super) struct Candidate {
    filter: FusionEffect,
    consumer: Option<FusionEffect>,
    lengths: Vec<ValueId>,
}

#[derive(Clone)]
struct FilterParts {
    space: SegSpace,
    body: filter::Body,
    input_nodes: Vec<ValueId>,
}

pub(super) fn analyze(inner: &Segmented, oracle: &SemanticGraph) -> Option<Candidate> {
    super::bodies(inner).find_map(|(_, graph, entry)| {
        let routes = entry.map(|entry| entry.routes().cloned().collect::<Vec<_>>());
        find_in_graph(graph, oracle, routes.as_deref())
    })
}

fn find_in_graph(
    graph: &EGraph,
    oracle: &SemanticGraph,
    routes: Option<&[RealizedOutputRoute]>,
) -> Option<Candidate> {
    let live = graph_ops::reachable_execution_values_with_roots(
        graph,
        routes
            .into_iter()
            .flat_map(|routes| routes.iter())
            .flat_map(RealizedOutputRoute::referenced_values),
    )
    .into_iter()
    .collect::<HashSet<_>>();
    for (block_id, block) in &graph.skeleton.blocks {
        for (filter_index, effect) in block.side_effects.iter().enumerate() {
            let SideEffectKind::Soac(SoacEffect(filter_id, Soac::Filter(_))) = &effect.kind else {
                continue;
            };
            let Some(result) = effect.value_result() else {
                continue;
            };
            let lengths = graph
                .nodes
                .iter()
                .filter_map(|(node, _)| {
                    (live.contains(&node) && is_length_of(graph, node, result)).then_some(node)
                })
                .collect::<Vec<_>>();
            if lengths.iter().skip(1).any(|length| graph.nodes[*length].ty != graph.nodes[lengths[0]].ty) {
                continue;
            }

            let consumers = block
                .side_effects
                .iter()
                .enumerate()
                .skip(filter_index + 1)
                .filter_map(|(index, consumer)| is_reduction_of_filter(consumer, result).then_some(index))
                .collect::<Vec<_>>();
            if consumers.len() > 1 {
                continue;
            }
            let consumer = consumers.first().copied();
            if consumer.is_none() && lengths.is_empty() {
                continue;
            }
            let stops = lengths.iter().copied().collect::<HashSet<_>>();
            if routes.is_some_and(|routes| {
                routes.iter().any(|route| {
                    support::pure_depends_on_avoiding(graph, route.source.value, result, &stops)
                })
            }) {
                continue;
            }
            if let Some(consumer_index) = consumer {
                if !((filter_index + 1)..consumer_index).all(|index| {
                    let intervening = &block.side_effects[index];
                    match &intervening.kind {
                        SideEffectKind::Soac(SoacEffect(id, _)) => !oracle.conflicts(filter_id, id),
                        _ => intervening.effects.is_none(),
                    }
                }) {
                    continue;
                }
            }
            if filter_result_escapes(graph, block_id, filter_index, consumer, result, &lengths) {
                continue;
            }
            return Some(Candidate {
                filter: FusionEffect(*filter_id),
                consumer: consumer.and_then(|index| FusionEffect::from_effect(&block.side_effects[index])),
                lengths,
            });
        }
    }
    None
}

fn is_length_of(graph: &EGraph, node: ValueId, filter_result: ValueId) -> bool {
    let ValueKind::Pure { op, operands } = &graph.nodes[node].kind else {
        return false;
    };
    if operands.as_slice() != [filter_result] {
        return false;
    }
    match op {
        PureOp::Intrinsic { id, .. } => *id == catalog().known().length,
        _ => false,
    }
}

fn is_reduction_of_filter(effect: &SideEffect, filter_result: ValueId) -> bool {
    let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &effect.kind else {
        return false;
    };
    if !op.is_reduce()
        || op.form.pre.result_types.len() != op.form.reduction_input_count()
        || !op.form.post.result_types.is_empty()
    {
        return false;
    }
    let input_count = op.inputs.len();
    input_count != 0
        && effect.operands.len() >= input_count
        && effect.operands[..input_count].iter().all(|input| input.value() == Some(filter_result))
}

fn filter_result_escapes(
    graph: &EGraph,
    filter_block: BlockId,
    filter_index: usize,
    consumer: Option<usize>,
    result: ValueId,
    lengths: &[ValueId],
) -> bool {
    let stops = lengths.iter().copied().collect::<HashSet<_>>();
    for (block_id, block) in &graph.skeleton.blocks {
        for (effect_index, effect) in block.side_effects.iter().enumerate() {
            if block_id == filter_block && effect_index == filter_index {
                continue;
            }
            if block_id == filter_block && Some(effect_index) == consumer {
                if !is_reduction_of_filter(effect, result) {
                    return true;
                }
                continue;
            }
            if graph_ops::effect_value_inputs(graph, effect)
                .into_iter()
                .any(|root| support::pure_depends_on_avoiding(graph, root, result, &stops))
            {
                return true;
            }
        }
        if block
            .term
            .referenced_nodes()
            .into_iter()
            .any(|root| support::pure_depends_on_avoiding(graph, root, result, &stops))
        {
            return true;
        }
    }
    false
}

fn filter_parts(effect: &SideEffect) -> super::FusionResult<FilterParts> {
    let SideEffectKind::Soac(SoacEffect(_, Soac::Filter(op))) = &effect.kind else {
        return Err(super::FusionError::InvalidCandidate(
            "selected Filter changed kind after candidate analysis".to_owned(),
        ));
    };
    let input_count = op.body.inputs.len();
    let Some(input_operands) = effect.operands.get(..input_count) else {
        return Err(super::FusionError::InvalidCandidate(
            "Filter has fewer operands than input types".to_owned(),
        ));
    };
    let input_nodes = input_operands.iter().map(|operand| operand.value()).collect::<Option<Vec<_>>>();
    let Some(input_nodes) = input_nodes else {
        return Err(super::FusionError::InvalidCandidate(
            "Filter input uses the place channel".to_owned(),
        ));
    };
    Ok(FilterParts {
        space: op.state.space.clone(),
        body: op.body.clone(),
        input_nodes,
    })
}

pub(super) fn apply(inner: Segmented, candidate: Candidate) -> super::FusionResult<Segmented> {
    let filter_location = candidate.filter.resolve(&inner)?;
    let consumer_location = if let Some(consumer) = candidate.consumer {
        let (_, consumer_location) = super::resolve_pair(&inner, candidate.filter, consumer)?;
        Some(consumer_location)
    } else {
        None
    };
    let (filter_effect, consumer_effect, outer_types, span, scope) = {
        let (graph, span, scope) = graph_and_span(&inner, filter_location.body)?;
        let block = &graph.skeleton.blocks[filter_location.block];
        (
            block.side_effects[filter_location.index].clone(),
            consumer_location.map(|consumer| block.side_effects[consumer.index].clone()),
            graph.nodes.iter().map(|(node, data)| (node, data.ty.clone())).collect::<LookupMap<_, _>>(),
            span,
            scope,
        )
    };
    let filter = filter_parts(&filter_effect)?;
    let mut identities = inner.data.identities.clone();
    let count_ty = candidate.lengths.first().map(|length| outer_types[length].clone());
    let consumer_form = if let Some(effect) = consumer_effect.as_ref() {
        let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &effect.kind else {
            return Err(super::FusionError::InvalidCandidate(
                "Filter consumer is no longer a Screma".to_owned(),
            ));
        };
        Some(&op.form)
    } else {
        None
    };
    let (pre, pre_function) = build_masked_pre(
        &inner,
        &mut identities,
        &scope,
        span,
        &filter,
        consumer_form,
        count_ty.as_ref(),
        &outer_types,
    )?;
    let (count_reduction, count_function) = if let Some(ty) = count_ty.as_ref() {
        let (reduction, function) = build_count_reduction(&mut identities, &scope, span, ty.clone())?;
        (Some(reduction), Some(function))
    } else {
        (None, None)
    };

    let (rebuilt, _) = if let Some(consumer_location) = consumer_location {
        let Some(consumer_effect) = consumer_effect else {
            return Err(super::FusionError::InvalidCandidate(
                "Filter consumer disappeared after candidate resolution".to_owned(),
            ));
        };
        rewrite_with_consumer(
            inner,
            &candidate,
            filter_location,
            consumer_location,
            filter_effect,
            consumer_effect,
            filter,
            pre,
            count_reduction,
            count_ty,
        )?
    } else {
        let Some(count_reduction) = count_reduction else {
            return Err(super::FusionError::InvalidCandidate(
                "length-only Filter has no count reduction".to_owned(),
            ));
        };
        let Some(count_ty) = count_ty else {
            return Err(super::FusionError::InvalidCandidate(
                "length-only Filter has no count type".to_owned(),
            ));
        };
        rewrite_count_only(
            inner,
            &candidate,
            filter_location,
            filter_effect,
            filter,
            pre,
            count_reduction,
            count_ty,
        )?
    };
    let synthesized = std::iter::once(pre_function).chain(count_function).collect::<Vec<_>>();
    Ok(
        rebuilt.extend_functions(synthesized).map_data(|data| SemanticProgramData {
            identities: identities,
            ..data
        }),
    )
}

fn build_count_reduction(
    identities: &mut ProgramIdentities,
    scope: &str,
    span: ast::Span,
    count_ty: Type<TypeName>,
) -> super::FusionResult<(screma::Reduce, Func<Semantic>)> {
    let mut graph = EGraph::new();
    let params = lambda_ops::named_parameters(&[count_ty.clone(), count_ty.clone()], "count");
    let arguments = lambda_ops::function_parameters(&mut graph, &params);
    let [left, right] = arguments.as_slice() else {
        return Err(super::FusionError::InvalidCandidate(
            "filter count lambda did not create two parameters".to_owned(),
        ));
    };
    let Some(left) = left.value() else {
        return Err(super::FusionError::InvalidCandidate(
            "left filter count parameter is not available by value".to_owned(),
        ));
    };
    let Some(right) = right.value() else {
        return Err(super::FusionError::InvalidCandidate(
            "right filter count parameter is not available by value".to_owned(),
        ));
    };
    let sum = graph.intern_pure(
        PureOp::BinOp(BinaryOperator::Add),
        smallvec![left, right],
        count_ty.clone(),
        None,
    );
    let entry = graph.skeleton.entry;
    let parameter_types = vec![count_ty.clone(), count_ty.clone()];
    let (operator, function) = lambda_ops::finish_region_lambda(
        identities,
        scope,
        "filter_count_combine",
        span,
        graph,
        entry,
        params,
        vec![],
        parameter_types,
        vec![count_ty],
        vec![sum],
        false,
    );
    let Some(function) = function else {
        return Err(super::FusionError::InvalidCandidate(
            "filter count operator unexpectedly became an identity".to_owned(),
        ));
    };
    Ok((
        screma::Reduce {
            operator,
            neutral: Vec::new(),
            commutative: true,
        },
        function,
    ))
}
#[allow(clippy::too_many_arguments)]
fn build_masked_pre(
    inner: &Segmented,
    identities: &mut ProgramIdentities,
    scope: &str,
    span: ast::Span,
    filter: &FilterParts,
    consumer: Option<&screma::ScremaForm>,
    count_ty: Option<&Type<TypeName>>,
    outer_types: &LookupMap<ValueId, Type<TypeName>>,
) -> super::FusionResult<(screma::Lambda, Func<Semantic>)> {
    let mut captures = filter.body.map.captures().to_vec();
    captures.extend_from_slice(filter.body.predicate.captures());
    if let Some(consumer) = consumer {
        captures.extend_from_slice(consumer.pre.captures());
        captures.extend(
            consumer
                .reductions
                .iter()
                .flat_map(|reduction| reduction.neutral.iter().copied())
                .map(egir::types::OperandRef::Value),
        );
    }
    let Some(capture_types) = capture_types(outer_types, captures.iter()) else {
        return Err(super::FusionError::InvalidCandidate(
            "Filter lambda capture is not available by value".to_owned(),
        ));
    };
    let input_types =
        filter.body.inputs.iter().map(egir::types::SoacInputType::element).collect::<Vec<_>>();
    let mut result_types = consumer.map(|consumer| consumer.pre.result_types.clone()).unwrap_or_default();
    if let Some(count_ty) = count_ty {
        result_types.push(count_ty.clone());
    }
    let mut params = lambda_ops::named_parameters(&input_types, "input");
    params.append(lambda_ops::named_parameters(&capture_types, "capture"));
    let mut graph = EGraph::new();
    let args = lambda_ops::function_parameters(&mut graph, &params);
    let mut cursor = input_types.len();
    let mapped_capture_count = filter.body.map.capture_count();
    let Some(input_args) = args.get(..input_types.len()) else {
        return Err(super::FusionError::InvalidCandidate(
            "Filter map has fewer generated arguments than inputs".to_owned(),
        ));
    };
    let Some(mapped_captures) = args.get(cursor..cursor + mapped_capture_count) else {
        return Err(super::FusionError::InvalidCandidate(
            "Filter map has fewer generated arguments than captures".to_owned(),
        ));
    };
    let Some(mapped) =
        support::invoke_lambda(&mut graph, inner, &filter.body.map, input_args, mapped_captures)
    else {
        return Err(super::FusionError::InvalidCandidate(
            "Filter map lambda region is absent".to_owned(),
        ));
    };
    let mapped = lambda_ops::result_argument_values(&mut graph, &mapped);
    cursor += mapped_capture_count;
    let predicate_capture_count = filter.body.predicate.capture_count();
    let mapped_arguments = mapped.iter().map(|value| graph.operand_ref(*value)).collect::<Vec<_>>();
    let Some(predicate_captures) = args.get(cursor..cursor + predicate_capture_count) else {
        return Err(super::FusionError::InvalidCandidate(
            "Filter predicate has fewer generated arguments than captures".to_owned(),
        ));
    };
    let Some(predicate) = support::invoke_lambda(
        &mut graph,
        inner,
        &filter.body.predicate,
        &mapped_arguments,
        predicate_captures,
    ) else {
        return Err(super::FusionError::InvalidCandidate(
            "Filter predicate lambda region is absent".to_owned(),
        ));
    };
    let predicate = lambda_ops::result_argument_values(&mut graph, &predicate);
    cursor += predicate_capture_count;
    let Some(&predicate) = predicate.first() else {
        return Err(super::FusionError::InvalidCandidate(
            "Filter predicate has no by-value result".to_owned(),
        ));
    };

    let mut selected = Vec::new();
    let mut fallback = Vec::new();
    if let Some(consumer) = consumer {
        let consumer_capture_count = consumer.pre.capture_count();
        let Some(&mapped) = mapped.first() else {
            return Err(super::FusionError::InvalidCandidate(
                "Filter map has no by-value result".to_owned(),
            ));
        };
        let consumer_args = vec![graph.operand_ref(mapped); consumer.pre.parameter_types.len()];
        let Some(consumer_captures) = args.get(cursor..cursor + consumer_capture_count) else {
            return Err(super::FusionError::InvalidCandidate(
                "Filter consumer has fewer generated arguments than captures".to_owned(),
            ));
        };
        let Some(results) = support::invoke_lambda(
            &mut graph,
            inner,
            &consumer.pre,
            &consumer_args,
            consumer_captures,
        ) else {
            return Err(super::FusionError::InvalidCandidate(
                "Filter consumer lambda region is absent".to_owned(),
            ));
        };
        selected.extend(lambda_ops::result_argument_values(&mut graph, &results));
        cursor += consumer_capture_count;
        let neutral_count = consumer.reduction_input_count();
        let Some(neutral_arguments) = args.get(cursor..cursor + neutral_count) else {
            return Err(super::FusionError::InvalidCandidate(
                "Filter consumer has fewer generated arguments than reduction neutrals".to_owned(),
            ));
        };
        let neutral_values =
            neutral_arguments.iter().map(|argument| argument.value()).collect::<Option<Vec<_>>>();
        let Some(neutral_values) = neutral_values else {
            return Err(super::FusionError::InvalidCandidate(
                "Filter reduction neutral is not available by value".to_owned(),
            ));
        };
        fallback.extend(neutral_values);
        cursor += neutral_count;
    }
    if let Some(count_ty) = count_ty {
        selected.push(integer_literal(&mut graph, "1", count_ty));
        fallback.push(integer_literal(&mut graph, "0", count_ty));
    }
    debug_assert_eq!(cursor, args.len());
    let (return_block, results) =
        conditional_results(&mut graph, predicate, fallback, selected, &result_types);
    let (pre, function) = lambda_ops::finish_region_lambda(
        identities,
        scope,
        "filter_pre",
        span,
        graph,
        return_block,
        params,
        captures,
        input_types,
        result_types,
        results,
        false,
    );
    let Some(function) = function else {
        return Err(super::FusionError::InvalidCandidate(
            "masked Filter pre-lambda unexpectedly became an identity".to_owned(),
        ));
    };
    Ok((pre, function))
}

fn conditional_results(
    graph: &mut EGraph,
    predicate: ValueId,
    fallback: Vec<ValueId>,
    selected: Vec<ValueId>,
    types: &[Type<TypeName>],
) -> (BlockId, Vec<ValueId>) {
    debug_assert_eq!(fallback.len(), types.len());
    debug_assert_eq!(selected.len(), types.len());
    let entry = graph.skeleton.entry;
    let then_block = graph.skeleton.create_block();
    let else_block = graph.skeleton.create_block();
    let merge = graph.skeleton.create_block();
    let results = types.iter().cloned().map(|ty| graph.add_block_param(merge, ty)).collect::<Vec<_>>();
    graph.skeleton.blocks[entry].term = SkeletonTerminator::CondBranch {
        cond: predicate,
        then_target: then_block,
        then_args: vec![],
        else_target: else_block,
        else_args: vec![],
    };
    graph.skeleton.blocks[entry].control_header = Some(ControlHeader::Selection { merge });
    graph.skeleton.blocks[then_block].term = SkeletonTerminator::Branch {
        target: merge,
        args: graph.admit_flow_values(selected),
    };
    graph.skeleton.blocks[else_block].term = SkeletonTerminator::Branch {
        target: merge,
        args: graph.admit_flow_values(fallback),
    };
    (merge, results)
}

struct EntryMetadataPatch {
    replacement: Option<ValueId>,
    old_writer: Option<ValueId>,
    replacement_writer: Option<ValueId>,
}

#[allow(clippy::too_many_arguments)]
fn rewrite_with_consumer(
    inner: Segmented,
    candidate: &Candidate,
    filter_location: super::ResolvedFusionEffect,
    consumer_location: super::ResolvedFusionEffect,
    filter_effect: SideEffect,
    consumer_effect: SideEffect,
    filter: FilterParts,
    pre: screma::Lambda,
    count: Option<screma::Reduce>,
    count_ty: Option<Type<TypeName>>,
) -> super::FusionResult<(Segmented, Vec<Func<Semantic>>)> {
    let synthesized = Vec::new();
    let rebuilt = inner.try_rewrite_body(filter_location.body, |body| {
        support::try_rewrite_body_graph_with_entry(
            body,
            |graph| {
                let SideEffectKind::Soac(SoacEffect(consumer_id, Soac::Screma(old_op))) =
                    &consumer_effect.kind
                else {
                    return Err(super::FusionError::InvalidCandidate(
                        "Filter consumer changed kind after candidate analysis".to_owned(),
                    ));
                };
                let Some(old_result) = consumer_effect.result().cloned() else {
                    return Err(super::FusionError::InvalidCandidate(
                        "Filter reduction has no result binding".to_owned(),
                    ));
                };
                let old_result_types = old_op
                    .form
                    .reductions
                    .iter()
                    .flat_map(|reduction| reduction.operator.result_types.iter().cloned())
                    .collect::<Vec<_>>();
                let count_neutral = count_ty.as_ref().map(|ty| integer_literal(graph, "0", ty));
                let count_project = if let (Some(mut count), Some(count_ty), Some(neutral)) =
                    (count.clone(), count_ty.as_ref(), count_neutral)
                {
                    count.neutral = vec![neutral];
                    let field = old_result_types.len() as u32;
                    let (new_result, count_value) =
                        extend_result(graph, &old_result, &old_result_types, count_ty.clone());
                    debug_assert_eq!(field as usize, new_result.field_count() - 1);
                    Some((count, new_result, count_value))
                } else {
                    None
                };

                let mut op = old_op.clone();
                op.inputs = filter.body.inputs.clone();
                op.form.pre = pre.clone();
                if let Some((count, _, _)) = &count_project {
                    op.form.reductions.push(count.clone());
                    op.result_state.push(screma::ResultState {
                        ownership: types::SoacOwnership::Fresh,
                    });
                }
                let screma::SemanticState::Segmented { space, resources, .. } = op.semantic_state_mut()
                else {
                    return Err(super::FusionError::InvalidCandidate(
                        "Filter consumer is no longer segmented".to_owned(),
                    ));
                };
                *space = filter.space.clone();
                let _ = resources;
                debug_assert!(
                    op.validate().is_ok(),
                    "invalid filtered Screma: {:?}",
                    op.validate()
                );

                let fused_effects = splice_effect_tokens(filter_effect.effects, consumer_effect.effects);
                let consumer_operands =
                    filter.input_nodes.iter().map(|value| graph.operand_ref(*value)).collect();
                let replacement_result =
                    count_project.as_ref().map(|(_, new_result, _)| new_result.clone());
                {
                    let consumer = &mut graph.skeleton.blocks[consumer_location.block].side_effects
                        [consumer_location.index];
                    consumer.kind = SideEffectKind::Soac(SoacEffect(*consumer_id, Soac::Screma(op)));
                    consumer.operands = consumer_operands;
                    consumer.effects = fused_effects;
                    if let Some(result) = replacement_result {
                        consumer.result = Some(result);
                    }
                }
                let consumer_snapshot = graph.skeleton.blocks[consumer_location.block].side_effects
                    [consumer_location.index]
                    .clone();
                let reads = egir::semantic_graph::read_resources(graph, &consumer_snapshot);
                if let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &mut graph.skeleton.blocks
                    [consumer_location.block]
                    .side_effects[consumer_location.index]
                    .kind
                {
                    let screma::SemanticState::Segmented { resources, .. } = op.semantic_state_mut() else {
                        return Err(super::FusionError::InvalidCandidate(
                            "fused Filter consumer is no longer segmented".to_owned(),
                        ));
                    };
                    *resources = SegResourceAccess::merge(resources, &reads);
                } else {
                    return Err(super::FusionError::InvalidCandidate(
                        "fused Filter consumer changed kind during rewrite".to_owned(),
                    ));
                }
                if let Some((_, _, project)) = &count_project {
                    replace_lengths(graph, &candidate.lengths, *project);
                }
                graph.skeleton.blocks[filter_location.block].side_effects.remove(filter_location.index);
                Ok(EntryMetadataPatch {
                    replacement: count_project.as_ref().map(|(_, _, value)| *value),
                    old_writer: filter_effect.value_result(),
                    replacement_writer: old_result.values().first().copied(),
                })
            },
            |entry, metadata| {
                finish_entry_metadata(entry, &candidate.lengths, metadata);
                Ok(())
            },
        )
    })?;
    Ok((rebuilt, synthesized))
}

fn rewrite_count_only(
    inner: Segmented,
    candidate: &Candidate,
    filter_location: super::ResolvedFusionEffect,
    filter_effect: SideEffect,
    filter: FilterParts,
    pre: screma::Lambda,
    mut count: screma::Reduce,
    count_ty: Type<TypeName>,
) -> super::FusionResult<(Segmented, Vec<Func<Semantic>>)> {
    let rebuilt = inner.try_rewrite_body(filter_location.body, |body| {
        support::try_rewrite_body_graph_with_entry(
            body,
            |graph| {
                let neutral = integer_literal(graph, "0", &count_ty);
                count.neutral = vec![neutral];
                let tuple_ty = Type::Constructed(TypeName::Tuple(1), vec![count_ty.clone()]);
                let count_value = graph.alloc_side_effect_result(count_ty.clone());
                let result_binding = ResultBinding::product(tuple_ty, [graph.value_result(count_value)]);
                replace_lengths(graph, &candidate.lengths, count_value);
                let operands = filter.input_nodes.iter().map(|value| graph.operand_ref(*value)).collect();
                {
                    let effect = &mut graph.skeleton.blocks[filter_location.block].side_effects
                        [filter_location.index];
                    let SideEffectKind::Soac(SoacEffect(id, _)) = effect.kind else {
                        return Err(super::FusionError::InvalidCandidate(
                            "selected Filter changed kind during count rewrite".to_owned(),
                        ));
                    };
                    effect.kind = SideEffectKind::Soac(SoacEffect(
                        id,
                        Soac::Screma(screma::Op {
                            inputs: filter.body.inputs.clone(),
                            form: screma::ScremaForm {
                                pre: pre.clone(),
                                scans: vec![],
                                reductions: vec![count.clone()],
                                post: screma::Lambda::identity(vec![]),
                            },
                            result_state: vec![screma::ResultState {
                                ownership: types::SoacOwnership::Fresh,
                            }],
                            state: screma::SemanticState::Segmented {
                                space: filter.space.clone(),
                                output_slots: vec![],
                                resources: vec![],
                            },
                        }),
                    ));
                    effect.operands = operands;
                    effect.result = Some(result_binding);
                }
                let effect_snapshot = graph.skeleton.blocks[filter_location.block].side_effects
                    [filter_location.index]
                    .clone();
                let reads = egir::semantic_graph::read_resources(graph, &effect_snapshot);
                if let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) =
                    &mut graph.skeleton.blocks[filter_location.block].side_effects[filter_location.index]
                        .kind
                {
                    let screma::SemanticState::Segmented { resources, .. } = op.semantic_state_mut() else {
                        return Err(super::FusionError::InvalidCandidate(
                            "count-only Filter rewrite is no longer segmented".to_owned(),
                        ));
                    };
                    *resources = reads;
                } else {
                    return Err(super::FusionError::InvalidCandidate(
                        "count-only Filter changed kind during rewrite".to_owned(),
                    ));
                }
                Ok(EntryMetadataPatch {
                    replacement: Some(count_value),
                    old_writer: filter_effect.value_result(),
                    replacement_writer: Some(count_value),
                })
            },
            |entry, metadata| {
                finish_entry_metadata(entry, &candidate.lengths, metadata);
                Ok(())
            },
        )
    })?;
    Ok((rebuilt, Vec::new()))
}

fn finish_entry_metadata(
    entry: &mut egir::program::Entry<Semantic>,
    old_values: &[ValueId],
    patch: EntryMetadataPatch,
) {
    if let Some(replacement) = patch.replacement {
        for route in entry.routes_mut() {
            if old_values.contains(&route.source.value) {
                route.source.value = replacement;
            }
            for writer in &mut route.writers {
                if matches!(writer, OutputWriter::Value(value) if old_values.contains(value)) {
                    *writer = OutputWriter::Value(replacement);
                }
            }
        }
    }
    if let (Some(old_writer), Some(replacement_writer)) = (patch.old_writer, patch.replacement_writer) {
        for route in entry.routes_mut() {
            for writer in &mut route.writers {
                if *writer == OutputWriter::Value(old_writer) {
                    *writer = OutputWriter::Value(replacement_writer);
                }
            }
        }
    }
}

fn replace_lengths(graph: &mut EGraph, lengths: &[ValueId], replacement: ValueId) {
    for &length in lengths {
        graph.replace_value_references(length, replacement);
    }
}

fn extend_result(
    graph: &mut EGraph,
    old_result: &ResultBinding<Type<TypeName>>,
    old_fields: &[Type<TypeName>],
    extra: Type<TypeName>,
) -> (ResultBinding<Type<TypeName>>, ValueId) {
    let mut fields = old_fields.to_vec();
    fields.push(extra.clone());
    let count = graph.alloc_side_effect_result(extra);
    let mut bindings = old_result.top_level_fields();
    debug_assert_eq!(bindings.len(), old_fields.len());
    bindings.push(graph.value_result(count));
    (
        ResultBinding::product(Type::Constructed(TypeName::Tuple(fields.len()), fields), bindings),
        count,
    )
}

fn integer_literal(graph: &mut EGraph, value: &str, ty: &Type<TypeName>) -> ValueId {
    let op = match ty {
        Type::Constructed(TypeName::UInt(_), _) => PureOp::Uint(value.to_string()),
        _ => PureOp::Int(value.to_string()),
    };
    graph.intern_pure(op, smallvec![], ty.clone(), None)
}
