//! Same-space producer/consumer fusion for canonical Scremas.
//!
//! A pure map producer can be folded into any downstream Screma by composing
//! its pre-lambda with the consumer's pre-lambda while preserving the consumer's
//! single collective barrier.

use crate::builtins;
use crate::op;
use crate::types;
use polytype::Type;
use smallvec::{smallvec, SmallVec};

use super::horizontal;
use super::screma as fusion_screma;
use super::support;
use super::{graph_and_span, FusionEffect};
use crate::ast::TypeName;
use crate::egir::graph_ops;
use crate::egir::ir::splice_effect_tokens;
use crate::egir::program::SemanticProgramData;
use crate::egir::reify::Segmented;
use crate::egir::semantic_graph::SemanticGraph;
use crate::egir::soac::screma;
use crate::egir::types::{
    EGraph, PureOp, ResourceAccess, ResultBinding, SegResourceAccess, SideEffectKind, Soac, SoacEffect,
    SoacInputType, ValueId, ValueKind,
};
use crate::flow::BlockId;
use crate::types::TypeExt;
use crate::LookupMap;

/// Ordered array-to-array transforms on a routed producer result. Slice is the
/// only such operation in Wyn's current pure IR; scalar `Index` transforms are
/// handled by the point-demand pass because they do not remain SOAC inputs.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct InputTransform {
    slices: Vec<SliceTransform>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SliceTransform {
    start: ValueId,
    extent: SliceExtent,
    size: Type<TypeName>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum SliceExtent {
    End {
        end: ValueId,
        op: PureOp,
    },
    Length(ValueId),
}

impl InputTransform {
    fn route(
        graph: &EGraph,
        operand: ValueId,
        producer_results: &[ResultBinding<Type<TypeName>>],
    ) -> Option<(usize, Self)> {
        let mut current = operand;
        let mut slices = Vec::new();
        loop {
            if let Some(field) =
                producer_results.iter().position(|result| graph.value_has_result_origin(current, result))
            {
                slices.reverse();
                return Some((field, Self { slices }));
            }
            current = graph.canonical_value(current);
            let ValueKind::Pure { op, operands } = &graph.nodes.get(current)?.kind else {
                return None;
            };
            let (base, start, extent) = match op {
                PureOp::StorageView(op::PureViewSource::Inherited) => {
                    let [start, len, base] = operands.as_slice() else {
                        return None;
                    };
                    (*base, *start, SliceExtent::Length(*len))
                }
                PureOp::Intrinsic { id, .. } if *id == builtins::catalog().known().slice => {
                    let [base, start, end] = operands.as_slice() else {
                        return None;
                    };
                    (
                        *base,
                        *start,
                        SliceExtent::End {
                            end: *end,
                            op: op.clone(),
                        },
                    )
                }
                _ => return None,
            };
            slices.push(SliceTransform {
                start,
                extent,
                size: graph.nodes[current].ty.array_size()?.clone(),
            });
            current = base;
        }
    }

    fn is_identity(&self) -> bool {
        self.slices.is_empty()
    }

    fn apply(
        &self,
        graph: &mut EGraph,
        mut node: ValueId,
        input: &SoacInputType,
    ) -> Option<(ValueId, SoacInputType)> {
        let mut array = input.array.clone();
        for slice in &self.slices {
            array = array_with_outer_size(&array, &slice.size)?;
            if array.array_variant().is_some_and(types::is_array_variant_view) {
                let len = match &slice.extent {
                    SliceExtent::Length(len) => *len,
                    SliceExtent::End { end, .. } => graph_ops::intern_binop(
                        graph,
                        op::BinaryOperator::Subtract,
                        *end,
                        slice.start,
                        graph.nodes[*end].ty().clone(),
                        None,
                    ),
                };
                node = graph_ops::intern_inherited_view(graph, node, slice.start, len, array.clone(), None);
            } else {
                let (end, op) = match &slice.extent {
                    SliceExtent::End { end, op } => (*end, op.clone()),
                    SliceExtent::Length(len) => {
                        let end = graph_ops::intern_binop(
                            graph,
                            op::BinaryOperator::Add,
                            slice.start,
                            *len,
                            graph.nodes[*len].ty().clone(),
                            None,
                        );
                        (
                            end,
                            PureOp::Intrinsic {
                                id: builtins::catalog().known().slice,
                                overload_idx: 0,
                            },
                        )
                    }
                };
                node = graph.intern_pure(op, smallvec![node, slice.start, end], array.clone(), None);
            }
        }
        let mut input = input.clone();
        input.array = array;
        Some((node, input))
    }
}

fn array_with_outer_size(array: &Type<TypeName>, size: &Type<TypeName>) -> Option<Type<TypeName>> {
    match array {
        Type::Constructed(TypeName::Array, arguments) if arguments.len() >= 4 => {
            let mut arguments = arguments.clone();
            arguments[2] = size.clone();
            Some(Type::Constructed(TypeName::Array, arguments))
        }
        Type::Constructed(TypeName::Tuple(arity), fields) => Some(Type::Constructed(
            TypeName::Tuple(*arity),
            fields.iter().map(|field| array_with_outer_size(field, size)).collect::<Option<Vec<_>>>()?,
        )),
        _ => None,
    }
}

#[derive(Clone)]
pub(super) struct Candidate {
    producer: FusionEffect,
    consumer: FusionEffect,
    routes: Vec<fusion_screma::InputRoute>,
    transform: InputTransform,
    retained_producer_outputs: Vec<usize>,
}

pub(super) fn analyze(inner: &Segmented, oracle: &SemanticGraph) -> Option<Candidate> {
    super::bodies(inner).find_map(|(_, graph, entry)| {
        let external_roots = entry
            .into_iter()
            .flat_map(|entry| entry.routes())
            .flat_map(|route| route.referenced_values())
            .collect::<Vec<_>>();
        find_in_graph(inner, graph, oracle, &external_roots)
    })
}

fn find_in_graph(
    inner: &Segmented,
    graph: &EGraph,
    oracle: &SemanticGraph,
    external_roots: &[ValueId],
) -> Option<Candidate> {
    for (block_id, block) in &graph.skeleton.blocks {
        for producer_index in 0..block.side_effects.len().saturating_sub(1) {
            let producer = &block.side_effects[producer_index];
            let SideEffectKind::Soac(SoacEffect(producer_id, Soac::Screma(producer_op))) = &producer.kind
            else {
                continue;
            };
            let screma::SemanticState::Segmented { resources, .. } = producer_op.semantic_state() else {
                continue;
            };
            let Some(producer_result) = producer.result.as_ref() else {
                continue;
            };
            let producer_results = producer_result.top_level_fields();
            if producer_results.len() != producer_op.result_count() {
                continue;
            }
            let value_consumers =
                oracle.value_consumers(producer_id).collect::<std::collections::HashSet<_>>();
            let value_consumer_kinds = block
                .side_effects
                .iter()
                .filter_map(|effect| match &effect.kind {
                    SideEffectKind::Soac(SoacEffect(id, Soac::Screma(op)))
                        if value_consumers.contains(id) =>
                    {
                        Some(op.is_map())
                    }
                    SideEffectKind::Soac(SoacEffect(id, _)) if value_consumers.contains(id) => Some(false),
                    _ => None,
                })
                .collect::<Vec<_>>();
            let all_value_consumers_are_maps = value_consumers.len() > 1
                && value_consumer_kinds.len() == value_consumers.len()
                && value_consumer_kinds.into_iter().all(|is_map| is_map);

            for consumer_index in (producer_index + 1)..block.side_effects.len() {
                let consumer = &block.side_effects[consumer_index];
                let SideEffectKind::Soac(SoacEffect(consumer_id, Soac::Screma(consumer_op))) =
                    &consumer.kind
                else {
                    continue;
                };
                let screma::SemanticState::Segmented {
                    resources: consumer_resources,
                    ..
                } = consumer_op.semantic_state()
                else {
                    continue;
                };
                if consumer.result.is_none() {
                    continue;
                }

                if !((producer_index + 1)..consumer_index).all(|index| {
                    let effect = &block.side_effects[index];
                    match (&effect.kind, effect.result.as_ref()) {
                        (SideEffectKind::Soac(SoacEffect(intervening, Soac::Screma(_))), Some(_)) => {
                            !oracle.conflicts(producer_id, intervening)
                        }
                        _ => effect.effects.is_none(),
                    }
                }) {
                    continue;
                }

                let consumer_input_count = consumer_op.inputs.len();
                let Some(consumer_inputs) = consumer.operands.get(..consumer_input_count) else {
                    continue;
                };
                let routed = consumer_inputs
                    .iter()
                    .enumerate()
                    .filter_map(|(input, operand)| {
                        let operand = operand.value()?;
                        InputTransform::route(graph, operand, &producer_results)
                            .map(|(field, transform)| (input, field, transform))
                    })
                    .collect::<Vec<_>>();
                if routed.is_empty() {
                    continue;
                }
                let routed_inputs =
                    routed.iter().map(|(input, _, _)| *input).collect::<std::collections::HashSet<_>>();
                let has_unrouteable_input = consumer_inputs.iter().enumerate().any(|(input, operand)| {
                    operand.value().is_some_and(|operand| {
                        !routed_inputs.contains(&input)
                            && depends_on_any_result(graph, operand, &producer_results)
                    })
                });
                if has_unrouteable_input {
                    continue;
                }
                let transform = routed[0].2.clone();
                if routed.iter().any(|(_, _, candidate_transform)| *candidate_transform != transform)
                    || (!transform.is_identity() && !producer_op.is_map())
                {
                    continue;
                }
                let routes = routed
                    .iter()
                    .map(|(consumer_input, producer_field, _)| {
                        let screma::ResultId::Post(producer_post_output) =
                            producer_op.form.result_id(*producer_field)?
                        else {
                            return None;
                        };
                        Some(fusion_screma::InputRoute {
                            consumer_input: *consumer_input,
                            producer_post_output,
                        })
                    })
                    .collect::<Option<Vec<_>>>();
                let Some(routes) = routes else {
                    continue;
                };
                let routed_resources = routes
                    .iter()
                    .filter_map(|route| consumer_op.inputs[route.consumer_input].array.array_buffer())
                    .filter_map(|buffer| match buffer {
                        Type::Constructed(TypeName::Buffer(binding), _) => Some(*binding),
                        _ => None,
                    })
                    .collect::<std::collections::HashSet<_>>();
                let unrouteable_resource_conflict = resources.iter().any(|producer_resource| {
                    consumer_resources.iter().any(|consumer_resource| {
                        producer_resource.resource == consumer_resource.resource
                            && (producer_resource.access != ResourceAccess::Read
                                || consumer_resource.access != ResourceAccess::Read)
                            && !routed_resources.contains(&producer_resource.resource)
                    })
                });
                let direct_effect_chain = matches!(
                    (producer.effects, consumer.effects),
                    (Some((_, producer_out)), Some((consumer_in, _))) if producer_out == consumer_in
                );
                if unrouteable_resource_conflict
                    || (oracle.conflicts(producer_id, consumer_id)
                        && !direct_effect_chain
                        && routed_resources.is_empty())
                {
                    continue;
                }
                let routed_roots = routed
                    .iter()
                    .filter_map(|(input, _, _)| {
                        consumer_inputs.get(*input).and_then(|operand| operand.value())
                    })
                    .collect::<std::collections::HashSet<_>>();
                let semantic_roots = consumer_op
                    .capture_nodes()
                    .into_iter()
                    .chain(consumer_op.form.scans.iter().flat_map(|scan| scan.neutral.iter().copied()))
                    .chain(
                        consumer_op
                            .form
                            .reductions
                            .iter()
                            .flat_map(|reduction| reduction.neutral.iter().copied()),
                    );
                if semantic_roots.into_iter().any(|root| {
                    depends_on_any_result(graph, root, &producer_results) && !routed_roots.contains(&root)
                }) {
                    continue;
                }
                if !fusion_screma::can_fuse_vertical(inner, &producer_op.form, &consumer_op.form, &routes) {
                    continue;
                }
                let consumer_count = oracle.value_consumer_count(producer_id);
                if consumer_count > 1 && !all_value_consumers_are_maps {
                    continue;
                }
                let retained_producer_outputs = retained_producer_outputs(
                    graph,
                    block_id,
                    producer_index,
                    consumer_index,
                    &producer_results,
                    external_roots,
                );
                if !transform.is_identity() && !retained_producer_outputs.is_empty() {
                    continue;
                }
                return Some(Candidate {
                    producer: FusionEffect(*producer_id),
                    consumer: FusionEffect(*consumer_id),
                    routes,
                    transform,
                    retained_producer_outputs,
                });
            }
        }
    }
    None
}

fn retained_producer_outputs(
    graph: &EGraph,
    producer_block: BlockId,
    producer_index: usize,
    consumer_index: usize,
    producer_results: &[ResultBinding<Type<TypeName>>],
    external_roots: &[ValueId],
) -> Vec<usize> {
    (0..producer_results.len())
        .filter(|field| {
            let results = producer_results[*field].values();
            for (block_id, block) in &graph.skeleton.blocks {
                for (index, effect) in block.side_effects.iter().enumerate() {
                    if block_id == producer_block && (index == producer_index || index == consumer_index) {
                        continue;
                    }
                    if graph_ops::effect_value_inputs(graph, effect).into_iter().any(|root| {
                        results.iter().any(|result| graph_ops::pure_depends_on(graph, root, *result))
                    }) {
                        return true;
                    }
                }
                if block.term.referenced_nodes().into_iter().any(|root| {
                    results.iter().any(|result| graph_ops::pure_depends_on(graph, root, *result))
                }) {
                    return true;
                }
            }
            external_roots
                .iter()
                .any(|root| results.iter().any(|result| graph_ops::pure_depends_on(graph, *root, *result)))
        })
        .collect()
}

fn depends_on_any_result(graph: &EGraph, root: ValueId, results: &[ResultBinding<Type<TypeName>>]) -> bool {
    results
        .iter()
        .any(|result| result.values().iter().any(|result| graph_ops::pure_depends_on(graph, root, *result)))
}
pub(super) fn apply(mut inner: Segmented, candidate: Candidate) -> super::FusionResult<Segmented> {
    let (producer_location, _) = super::resolve_pair(&inner, candidate.producer, candidate.consumer)?;
    let transform = candidate.transform.clone();
    let mut transformed_source = None;
    inner = inner.try_rewrite_body(producer_location.body, |body| {
        support::try_rewrite_body_graph(body, |graph| {
            let producer =
                horizontal::extract_screma(graph, producer_location.block, producer_location.index)?;
            let transformed = producer
                .inputs
                .iter()
                .map(|input| {
                    transform
                        .apply(graph, input.node, &input.ty)
                        .map(|(node, ty)| super::FusionInput { node, ty })
                })
                .collect::<Option<Vec<_>>>();
            let Some(transformed) = transformed else {
                return Err(super::FusionError::InvalidCandidate(
                    "producer input transform failed after candidate analysis".to_owned(),
                ));
            };
            transformed_source = Some(transformed);
            Ok(())
        })
    })?;
    let Some(producer_inputs) = transformed_source else {
        return Err(super::FusionError::InvalidCandidate(
            "vertical fusion did not visit its selected body".to_owned(),
        ));
    };

    let (producer_location, consumer_location) =
        super::resolve_pair(&inner, candidate.producer, candidate.consumer)?;
    let (graph, span, scope) = graph_and_span(&inner, producer_location.body)?;
    let outer_types =
        graph.nodes.iter().map(|(node, data)| (node, data.ty.clone())).collect::<LookupMap<_, _>>();
    let producer = horizontal::extract_screma(graph, producer_location.block, producer_location.index)?;
    let consumer = horizontal::extract_screma(graph, consumer_location.block, consumer_location.index)?;
    let mut identities = inner.data.identities.clone();
    let mut context = fusion_screma::Context {
        program: &inner,
        identities: &mut identities,
        scope: &scope,
        span,
        outer_types: &outer_types,
    };
    let normalized = fusion_screma::fuse_vertical(
        &mut context,
        fusion_screma::Source {
            inputs: &producer_inputs,
            form: &producer.op.form,
        },
        fusion_screma::Source {
            inputs: &consumer.inputs,
            form: &consumer.op.form,
        },
        &candidate.routes,
        &candidate.retained_producer_outputs,
    );
    let Some(normalized) = normalized else {
        return Err(super::FusionError::InvalidCandidate(
            "vertical Screma normalization failed after candidate analysis".to_owned(),
        ));
    };

    let mut producer_mapping = vec![usize::MAX; producer.result_types.len()];
    let mut consumer_mapping = vec![usize::MAX; consumer.result_types.len()];
    let mut result_state = Vec::with_capacity(normalized.outputs.len());
    let mut result_types = Vec::with_capacity(normalized.outputs.len());
    for (fused_field, origin) in normalized.outputs.iter().copied().enumerate() {
        let (source_field, source_state, source_types, mapping) = match origin {
            fusion_screma::OutputOrigin::Producer(field) => (
                field,
                &producer.op.result_state,
                &producer.result_types,
                &mut producer_mapping,
            ),
            fusion_screma::OutputOrigin::Consumer(field) => (
                field,
                &consumer.op.result_state,
                &consumer.result_types,
                &mut consumer_mapping,
            ),
        };
        let Some(mapped_field) = mapping.get_mut(source_field) else {
            return Err(super::FusionError::InvalidCandidate(
                "vertical fusion output is outside its source result".to_owned(),
            ));
        };
        let Some(state) = source_state.get(source_field) else {
            return Err(super::FusionError::InvalidCandidate(
                "vertical fusion output has no result state".to_owned(),
            ));
        };
        let Some(ty) = source_types.get(source_field) else {
            return Err(super::FusionError::InvalidCandidate(
                "vertical fusion output has no result type".to_owned(),
            ));
        };
        *mapped_field = fused_field;
        result_state.push(*state);
        result_types.push(ty.clone());
    }

    debug_assert!(consumer_mapping.iter().all(|field| *field != usize::MAX));

    let mut output_slots = producer.output_slots.clone();
    output_slots.extend(consumer.output_slots.iter().copied());
    output_slots.sort_unstable();
    output_slots.dedup();
    let resources = SegResourceAccess::merge(&producer.resources, &consumer.resources);
    let fused_op = screma::Op {
        inputs: normalized.inputs.iter().map(|input| input.ty.clone()).collect(),
        form: normalized.form,
        result_state,
        state: screma::SemanticState::Segmented {
            space: if candidate.transform.is_identity() { producer.space } else { consumer.space },
            output_slots,
            resources,
        },
    };
    debug_assert!(
        fused_op.validate().is_ok(),
        "invalid vertically fused Screma: {:?}",
        fused_op.validate()
    );

    let mut operand_values = SmallVec::<[ValueId; 4]>::new();
    operand_values.extend(normalized.inputs.iter().map(|input| input.node));
    let synthesized = normalized.synthesized;
    let producer_results = producer.results;
    let consumer_results = consumer.results;
    let consumer_id = consumer.id;
    let site = producer_location.body;
    let rebuilt = inner.try_rewrite_body(site, |body| {
        support::try_rewrite_body_graph_with_entry(
            body,
            |graph| {
                let tuple_type =
                    Type::Constructed(TypeName::Tuple(result_types.len()), result_types.clone());
                let result = graph_ops::alloc_by_value_effect_result(graph, tuple_type);
                let fused_results = result.top_level_fields();
                let mut replacements = Vec::new();
                for (old, field) in producer_results.iter().zip(&producer_mapping) {
                    if *field != usize::MAX {
                        let Some(fused_result) = fused_results.get(*field) else {
                            return Err(super::FusionError::InvalidCandidate(
                                "retained producer result maps outside the fused result".to_owned(),
                            ));
                        };
                        replacements.extend(
                            graph_ops::rebind_result_value_references(graph, old, fused_result)
                                .map_err(super::FusionError::InvalidCandidate)?,
                        );
                    }
                }
                replacements.extend(horizontal::rebind_fields(
                    graph,
                    &consumer_results,
                    &fused_results,
                    &consumer_mapping,
                )?);

                let operands = operand_values
                    .iter()
                    .map(|operand| graph.operand_ref(*operand))
                    .collect::<SmallVec<_>>();
                let block = &mut graph.skeleton.blocks[producer_location.block];
                let effects = splice_effect_tokens(
                    block.side_effects[producer_location.index].effects,
                    block.side_effects[consumer_location.index].effects,
                );
                let consumer_effect = &mut block.side_effects[consumer_location.index];
                consumer_effect.kind =
                    SideEffectKind::Soac(SoacEffect(consumer_id, Soac::Screma(fused_op.clone())));
                consumer_effect.operands = operands;
                consumer_effect.result = Some(result);
                consumer_effect.effects = effects;
                block.side_effects.remove(producer_location.index);
                Ok(replacements)
            },
            |entry, replacements| {
                support::replace_route_values(entry, &replacements);
                Ok(())
            },
        )
    })?;
    Ok(
        rebuilt.extend_functions(synthesized).map_data(|data| SemanticProgramData {
            identities: identities,
            ..data
        }),
    )
}
