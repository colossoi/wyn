//! Futhark-style Screma fusion algebra.
//!
//! Graph discovery and EGIR effect rewiring live in the horizontal/vertical
//! modules. This module is the representation-level core: it constructs the
//! equivalent of Futhark's `SuperScrema`, normalises the legal barrier ordering,
//! deduplicates forwarded inputs, and synthesises whole pre/post lambdas.

use polytype::Type;

use super::{capture_types, deduplicate_array_inputs};
use crate::ast::{Span, TypeName};
use crate::egir::program::{RegionInterner, SemanticFunc};
use crate::egir::reify::Segmented;
use crate::egir::soac::{lambda as lambda_ops, screma};
use crate::egir::types::{EGraph, ENode, NodeId, PureOp, SoacInputType};
use crate::egir::{graph_ops, inlining};
use crate::LookupMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OutputOrigin {
    Producer(usize),
    Consumer(usize),
}

pub(crate) struct Normalized {
    pub input_nodes: Vec<NodeId>,
    pub inputs: Vec<SoacInputType>,
    pub form: screma::ScremaForm,
    /// Canonical fused field order, expressed in the source operations' field
    /// spaces. Independent siblings retain left-to-right post-result order.
    pub outputs: Vec<OutputOrigin>,
    pub synthesized: Vec<SemanticFunc>,
}

pub(crate) struct Source<'a> {
    pub input_nodes: &'a [NodeId],
    pub inputs: &'a [SoacInputType],
    pub form: &'a screma::ScremaForm,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct InputRoute {
    pub consumer_input: usize,
    /// Producer post-lambda result index. Reduction results cannot be array routes.
    pub producer_post_output: usize,
}

fn routed_producer_post_output(routes: &[InputRoute], consumer_input: usize) -> Option<usize> {
    routes
        .iter()
        .find(|route| route.consumer_input == consumer_input)
        .map(|route| route.producer_post_output)
}

fn result_post_output(form: &screma::ScremaForm, output: usize) -> Option<usize> {
    let post = output.checked_sub(form.reduction_result_count())?;
    (post < form.post.result_types.len()).then_some(post)
}

pub(crate) struct Context<'a> {
    pub program: &'a Segmented,
    pub interner: &'a mut RegionInterner,
    pub scope: &'a str,
    pub span: Span,
    pub outer_types: &'a LookupMap<NodeId, Type<TypeName>>,
}

/// Horizontal normalisation using the SuperScrema barrier algebra. Independent
/// collective groups move to the first barrier; the wrapper forms the
/// associative product and preserves left-to-right results within each partition.
pub(crate) fn fuse_horizontal(
    context: &mut Context<'_>,
    producer: Source<'_>,
    consumer: Source<'_>,
) -> Normalized {
    let producer_input_count = producer.inputs.len();
    let mut raw_nodes = producer.input_nodes.to_vec();
    raw_nodes.extend_from_slice(consumer.input_nodes);
    let mut raw_array_types = producer.inputs.iter().map(|input| input.array.clone()).collect::<Vec<_>>();
    raw_array_types.extend(consumer.inputs.iter().map(|input| input.array.clone()));
    let mut raw_element_types = producer.inputs.iter().map(SoacInputType::element).collect::<Vec<_>>();
    raw_element_types.extend(consumer.inputs.iter().map(SoacInputType::element));
    let (input_nodes, input_array_types, input_element_types, remap) =
        deduplicate_array_inputs(raw_nodes, raw_array_types, raw_element_types);

    let producer_parameters = remap[..producer_input_count].to_vec();
    let consumer_parameters = remap[producer_input_count..].to_vec();
    let (pre, pre_function) = parallel_pre(
        context,
        &input_element_types,
        producer.form,
        consumer.form,
        producer_parameters,
        consumer_parameters,
    );
    let (post, post_function) = parallel_post(context, producer.form, consumer.form);

    let producer_reductions = producer.form.reduction_result_count();
    let consumer_reductions = consumer.form.reduction_result_count();
    let outputs = (0..producer_reductions)
        .map(OutputOrigin::Producer)
        .chain((0..consumer_reductions).map(OutputOrigin::Consumer))
        .chain((producer_reductions..producer.form.result_count()).map(OutputOrigin::Producer))
        .chain((consumer_reductions..consumer.form.result_count()).map(OutputOrigin::Consumer))
        .collect();

    Normalized {
        input_nodes,
        inputs: input_array_types.into_iter().map(|array| SoacInputType { array }).collect(),
        form: screma::ScremaForm {
            pre,
            scans: producer.form.scans.iter().chain(&consumer.form.scans).cloned().collect(),
            reductions: producer.form.reductions.iter().chain(&consumer.form.reductions).cloned().collect(),
            post,
        },
        outputs,
        synthesized: pre_function.into_iter().chain(post_function).collect(),
    }
}

/// A producer/consumer pair in the transient three-stage form used by
/// Futhark's fusion algorithm: producer pre/barrier, the data-dependent middle
/// work and consumer barrier, then final post work. Sources stay intact until
/// normalization so legality and construction do not accrete in graph clients.
struct SuperScrema<'a> {
    producer: Source<'a>,
    consumer: Source<'a>,
    routes: &'a [InputRoute],
    retained_producer_outputs: &'a [usize],
}

pub(crate) fn can_fuse_vertical(
    program: &Segmented,
    producer: &screma::ScremaForm,
    consumer: &screma::ScremaForm,
    routes: &[InputRoute],
) -> bool {
    if routes.iter().any(|route| route.producer_post_output >= producer.post.result_types.len()) {
        return false;
    }
    if producer.scans.is_empty() && producer.reductions.is_empty() && producer.post.is_identity() {
        return true;
    }
    if producer.scans.is_empty() && producer.reductions.is_empty() {
        return false;
    }

    let collective_results = consumer.operator_input_count();
    if collective_results == 0 {
        return true;
    }
    let collective = 0..collective_results;
    if !lambda_results_projectable(program, &consumer.pre, collective.clone()) {
        return false;
    }
    let scan_parameters = (0..producer.scan_input_count()).collect::<Vec<_>>();
    routes.iter().all(|route| {
        let producer_output = route.producer_post_output;
        let depends_on_route = lambda_results_depend_on_parameters(
            program,
            &consumer.pre,
            collective.clone(),
            &[route.consumer_input],
        )
        .unwrap_or(true);
        !depends_on_route
            || (lambda_results_projectable(program, &producer.post, producer_output..producer_output + 1)
                && !lambda_results_depend_on_parameters(
                    program,
                    &producer.post,
                    producer_output..producer_output + 1,
                    &scan_parameters,
                )
                .unwrap_or(true))
    })
}

pub(crate) fn fuse_vertical(
    context: &mut Context<'_>,
    producer: Source<'_>,
    consumer: Source<'_>,
    routes: &[InputRoute],
    retained_producer_outputs: &[usize],
) -> Option<Normalized> {
    SuperScrema {
        producer,
        consumer,
        routes,
        retained_producer_outputs,
    }
    .normalize(context)
}

pub(crate) struct LambdaSource<'a> {
    pub input_nodes: &'a [NodeId],
    pub inputs: &'a [SoacInputType],
    pub lambda: &'a screma::Lambda,
}

pub(crate) struct NormalizedLambda {
    pub input_nodes: Vec<NodeId>,
    pub inputs: Vec<SoacInputType>,
    pub lambda: screma::Lambda,
    pub synthesized: Vec<SemanticFunc>,
}

/// Compose a pure map producer into an arbitrary element lambda. This is the
/// common Futhark `fuseMaps` operation used by non-Screma envelopes.
pub(crate) fn fuse_map_into_lambda(
    context: &mut Context<'_>,
    producer: Source<'_>,
    consumer: LambdaSource<'_>,
    routes: &[InputRoute],
) -> Option<NormalizedLambda> {
    if !producer.form.scans.is_empty()
        || !producer.form.reductions.is_empty()
        || !producer.form.post.is_identity()
        || routes.iter().any(|route| route.producer_post_output >= producer.form.post.result_types.len())
    {
        return None;
    }

    let remaining_slots = (0..consumer.inputs.len())
        .filter(|slot| routed_producer_post_output(routes, *slot).is_none())
        .collect::<Vec<_>>();
    let mut raw_nodes = remaining_slots.iter().map(|&slot| consumer.input_nodes[slot]).collect::<Vec<_>>();
    raw_nodes.extend_from_slice(producer.input_nodes);
    let mut raw_array_types =
        remaining_slots.iter().map(|&slot| consumer.inputs[slot].array.clone()).collect::<Vec<_>>();
    raw_array_types.extend(producer.inputs.iter().map(|input| input.array.clone()));
    let mut raw_element_types =
        remaining_slots.iter().map(|&slot| consumer.inputs[slot].element()).collect::<Vec<_>>();
    raw_element_types.extend(producer.inputs.iter().map(SoacInputType::element));
    let (input_nodes, input_array_types, input_element_types, remap) =
        deduplicate_array_inputs(raw_nodes, raw_array_types, raw_element_types);

    let producer_base = remaining_slots.len();
    let producer_parameters = remap[producer_base..].to_vec();
    let consumer_parameters = (0..consumer.inputs.len())
        .map(|slot| {
            remaining_slots.iter().position(|candidate| *candidate == slot).map(|position| remap[position])
        })
        .collect::<Vec<_>>();
    let outputs =
        (0..consumer.lambda.result_types.len()).map(VerticalValueRef::Consumer).collect::<Vec<_>>();
    let (lambda, function) = vertical_lambda(
        context,
        "map_envelope",
        &input_element_types,
        &producer.form.pre,
        producer_parameters,
        0,
        consumer.lambda,
        &consumer_parameters,
        routes,
        &outputs,
    );

    Some(NormalizedLambda {
        input_nodes,
        inputs: input_array_types.into_iter().map(|array| SoacInputType { array }).collect(),
        lambda,
        synthesized: function.into_iter().collect(),
    })
}
impl SuperScrema<'_> {
    fn normalize(self, context: &mut Context<'_>) -> Option<Normalized> {
        if self.producer.form.scans.is_empty() && self.producer.form.post.is_identity() {
            return Some(fuse_scanless_producer(
                context,
                self.producer,
                self.consumer,
                self.routes,
                self.retained_producer_outputs,
            ));
        }
        if can_fuse_vertical(
            context.program,
            self.producer.form,
            self.consumer.form,
            self.routes,
        ) {
            return fuse_across_middle_barrier(
                context,
                self.producer,
                self.consumer,
                self.routes,
                self.retained_producer_outputs,
            );
        }
        None
    }
}
/// Normalize a SuperScrema whose producer has no scans. Producer map work and
/// consumer pre-work compose before the first barrier; producer reductions and
/// consumer collectives become sibling operators at that barrier.
fn fuse_scanless_producer(
    context: &mut Context<'_>,
    producer: Source<'_>,
    consumer: Source<'_>,
    routes: &[InputRoute],
    retained_producer_outputs: &[usize],
) -> Normalized {
    let remaining_slots = (0..consumer.inputs.len())
        .filter(|slot| routed_producer_post_output(routes, *slot).is_none())
        .collect::<Vec<_>>();
    let mut raw_nodes = remaining_slots.iter().map(|&slot| consumer.input_nodes[slot]).collect::<Vec<_>>();
    raw_nodes.extend_from_slice(producer.input_nodes);
    let mut raw_array_types =
        remaining_slots.iter().map(|&slot| consumer.inputs[slot].array.clone()).collect::<Vec<_>>();
    raw_array_types.extend(producer.inputs.iter().map(|input| input.array.clone()));
    let mut raw_element_types =
        remaining_slots.iter().map(|&slot| consumer.inputs[slot].element()).collect::<Vec<_>>();
    raw_element_types.extend(producer.inputs.iter().map(SoacInputType::element));
    let (input_nodes, input_array_types, input_element_types, remap) =
        deduplicate_array_inputs(raw_nodes, raw_array_types, raw_element_types);

    let producer_base = remaining_slots.len();
    let producer_parameters = remap[producer_base..].to_vec();
    let consumer_parameters = (0..consumer.inputs.len())
        .map(|slot| {
            remaining_slots.iter().position(|candidate| *candidate == slot).map(|position| remap[position])
        })
        .collect::<Vec<_>>();
    let producer_reductions = producer.form.reduction_result_count();
    let consumer_scan_inputs = consumer.form.scan_input_count();
    let consumer_operator_inputs = consumer.form.operator_input_count();
    let consumer_reductions = consumer.form.reduction_result_count();
    let retained_producer_post_outputs = retained_producer_outputs
        .iter()
        .filter_map(|&output| result_post_output(producer.form, output).map(|post| (output, post)))
        .collect::<Vec<_>>();
    let pre_outputs = (0..consumer_scan_inputs)
        .map(VerticalValueRef::Consumer)
        .chain((0..producer_reductions).map(VerticalValueRef::Producer))
        .chain((consumer_scan_inputs..consumer_operator_inputs).map(VerticalValueRef::Consumer))
        .chain(
            (consumer_operator_inputs..consumer.form.pre.result_types.len())
                .map(VerticalValueRef::Consumer),
        )
        .chain(
            retained_producer_post_outputs
                .iter()
                .map(|(_, post)| VerticalValueRef::Producer(producer_reductions + post)),
        )
        .collect::<Vec<_>>();
    let (pre, pre_function) = vertical_lambda(
        context,
        "vertical_pre",
        &input_element_types,
        &producer.form.pre,
        producer_parameters,
        producer_reductions,
        &consumer.form.pre,
        &consumer_parameters,
        routes,
        &pre_outputs,
    );

    let retained_producer_types = retained_producer_post_outputs
        .iter()
        .map(|(_, post)| producer.form.post.result_types[*post].clone())
        .collect::<Vec<_>>();
    let producer_outputs = retained_producer_types.len();
    let consumer_post_parameters = consumer.form.post.parameter_types.len();
    let post_parameter_types = consumer
        .form
        .post
        .parameter_types
        .iter()
        .cloned()
        .chain(retained_producer_types.iter().cloned())
        .collect::<Vec<_>>();
    let (post, post_function) = if consumer.form.scans.is_empty() {
        (screma::Lambda::identity(post_parameter_types), None)
    } else {
        let forwarded = screma::Lambda::identity(retained_producer_types);
        let outputs = (0..consumer.form.post.result_types.len())
            .map(|result| ValueRef { call: 0, result })
            .chain((0..producer_outputs).map(|result| ValueRef { call: 1, result }))
            .collect();
        parallel_lambdas(
            context,
            "vertical_forward_post",
            post_parameter_types,
            vec![
                LambdaCall {
                    lambda: &consumer.form.post,
                    parameters: (0..consumer_post_parameters).collect(),
                },
                LambdaCall {
                    lambda: &forwarded,
                    parameters: (consumer_post_parameters..consumer_post_parameters + producer_outputs)
                        .collect(),
                },
            ],
            outputs,
        )
    };
    let outputs = (0..producer_reductions)
        .map(OutputOrigin::Producer)
        .chain((0..consumer_reductions).map(OutputOrigin::Consumer))
        .chain((consumer_reductions..consumer.form.result_count()).map(OutputOrigin::Consumer))
        .chain(retained_producer_post_outputs.iter().map(|(output, _)| OutputOrigin::Producer(*output)))
        .collect::<Vec<_>>();
    debug_assert_eq!(
        producer_reductions + consumer_reductions + post.result_types.len(),
        outputs.len()
    );

    Normalized {
        input_nodes,
        inputs: input_array_types.into_iter().map(|array| SoacInputType { array }).collect(),
        form: screma::ScremaForm {
            pre,
            scans: consumer.form.scans.clone(),
            reductions: producer.form.reductions.iter().chain(&consumer.form.reductions).cloned().collect(),
            post,
        },
        outputs,
        synthesized: pre_function.into_iter().chain(post_function).collect(),
    }
}
/// Move the consumer's independent collective inputs to the first barrier.
/// The consumer pre-lambda is partitioned by result dependency: its scan and
/// reduction inputs run before the producer barrier, while its mapped suffix
/// remains between the combined barrier and the consumer post-lambda.
fn fuse_across_middle_barrier(
    context: &mut Context<'_>,
    producer: Source<'_>,
    consumer: Source<'_>,
    routes: &[InputRoute],
    retained_producer_outputs: &[usize],
) -> Option<Normalized> {
    let remaining_slots = (0..consumer.inputs.len())
        .filter(|slot| routed_producer_post_output(routes, *slot).is_none())
        .collect::<Vec<_>>();
    let producer_input_count = producer.inputs.len();
    let mut raw_nodes = producer.input_nodes.to_vec();
    raw_nodes.extend(remaining_slots.iter().map(|&slot| consumer.input_nodes[slot]));
    let mut raw_array_types = producer.inputs.iter().map(|input| input.array.clone()).collect::<Vec<_>>();
    raw_array_types.extend(remaining_slots.iter().map(|&slot| consumer.inputs[slot].array.clone()));
    let mut raw_element_types = producer.inputs.iter().map(SoacInputType::element).collect::<Vec<_>>();
    raw_element_types.extend(remaining_slots.iter().map(|&slot| consumer.inputs[slot].element()));
    let (input_nodes, input_array_types, input_element_types, remap) =
        deduplicate_array_inputs(raw_nodes, raw_array_types, raw_element_types);

    let producer_parameters = remap[..producer_input_count].to_vec();
    let forwarded_parameters = remap[producer_input_count..].to_vec();
    let producer_scan_inputs = producer.form.scan_input_count();
    let producer_reduction_inputs = producer.form.reduction_input_count();
    let producer_operator_inputs = producer.form.operator_input_count();
    let producer_reduction_results = producer.form.reduction_result_count();
    let consumer_reduction_results = consumer.form.reduction_result_count();
    let retained_producer_post_outputs = retained_producer_outputs
        .iter()
        .filter_map(|&output| result_post_output(producer.form, output).map(|post| (output, post)))
        .collect::<Vec<_>>();
    let producer_mapped_types = producer.form.mapped_types()?.to_vec();
    let consumer_scan_inputs = consumer.form.scan_input_count();
    let consumer_collective_inputs = consumer.form.operator_input_count();
    let dependent_routes = routes
        .iter()
        .copied()
        .filter(|route| {
            lambda_results_depend_on_parameters(
                context.program,
                &consumer.form.pre,
                0..consumer_collective_inputs,
                &[route.consumer_input],
            )
            .unwrap_or(true)
        })
        .collect::<Vec<_>>();

    let pre_captures = producer
        .form
        .pre
        .seg_body()
        .into_iter()
        .flat_map(|body| body.captures.iter().copied())
        .chain(
            (!dependent_routes.is_empty())
                .then_some(&producer.form.post)
                .into_iter()
                .flat_map(|lambda| lambda.seg_body())
                .flat_map(|body| body.captures.iter().copied()),
        )
        .chain(
            (consumer_collective_inputs > 0)
                .then_some(&consumer.form.pre)
                .into_iter()
                .flat_map(|lambda| lambda.seg_body())
                .flat_map(|body| body.captures.iter().copied()),
        )
        .collect::<Vec<_>>();
    let mut pre_params = lambda_ops::named_parameters(&input_element_types, "element");
    pre_params.extend(lambda_ops::named_parameters(
        &capture_types(context.outer_types, pre_captures.iter()),
        "capture",
    ));
    let mut pre_graph = EGraph::new();
    let pre_arguments = lambda_ops::function_parameters(&mut pre_graph, &pre_params);
    let mut pre_capture_cursor = input_element_types.len();

    let mut producer_pre_arguments =
        producer_parameters.iter().map(|&index| pre_arguments[index]).collect::<Vec<_>>();
    append_wrapper_captures(
        &mut producer_pre_arguments,
        &pre_arguments,
        &mut pre_capture_cursor,
        &producer.form.pre,
    );
    let producer_pre_results = invoke_lambda(
        &mut pre_graph,
        context.program,
        &producer.form.pre,
        producer_pre_arguments,
    );
    let producer_mapped = producer_pre_results[producer_operator_inputs..].to_vec();

    let mut produced_before_barrier = LookupMap::new();
    if !dependent_routes.is_empty() {
        let mut producer_post_arguments = vec![None; producer_scan_inputs];
        producer_post_arguments.extend(producer_mapped.iter().copied().map(Some));
        append_optional_wrapper_captures(
            &mut producer_post_arguments,
            &pre_arguments,
            &mut pre_capture_cursor,
            &producer.form.post,
        );
        let mut outputs =
            dependent_routes.iter().map(|route| route.producer_post_output).collect::<Vec<_>>();
        outputs.sort_unstable();
        outputs.dedup();
        let values = emit_projected_lambda_result_indices(
            &mut pre_graph,
            context.program,
            &producer.form.post,
            &producer_post_arguments,
            &outputs,
        )?;
        produced_before_barrier.extend(outputs.into_iter().zip(values));
    }

    let mut consumer_pre_arguments = (0..consumer.inputs.len())
        .map(|slot| {
            if let Some(output) = routed_producer_post_output(routes, slot) {
                produced_before_barrier.get(&output).copied()
            } else {
                let position = remaining_slots.iter().position(|candidate| *candidate == slot)?;
                Some(pre_arguments[forwarded_parameters[position]])
            }
        })
        .collect::<Vec<_>>();
    append_optional_wrapper_captures(
        &mut consumer_pre_arguments,
        &pre_arguments,
        &mut pre_capture_cursor,
        &consumer.form.pre,
    );
    let consumer_collective = emit_projected_lambda_results(
        &mut pre_graph,
        context.program,
        &consumer.form.pre,
        &consumer_pre_arguments,
        0..consumer_collective_inputs,
    )?;
    debug_assert_eq!(pre_capture_cursor, pre_arguments.len());

    let pre_results = producer_pre_results[..producer_scan_inputs]
        .iter()
        .copied()
        .chain(consumer_collective[..consumer_scan_inputs].iter().copied())
        .chain(producer_pre_results[producer_scan_inputs..producer_operator_inputs].iter().copied())
        .chain(consumer_collective[consumer_scan_inputs..].iter().copied())
        .chain(producer_mapped.iter().copied())
        .chain(forwarded_parameters.iter().map(|&index| pre_arguments[index]))
        .collect::<Vec<_>>();
    let pre_result_types = producer.form.pre.result_types[..producer_scan_inputs]
        .iter()
        .cloned()
        .chain(consumer.form.pre.result_types[..consumer_scan_inputs].iter().cloned())
        .chain(
            producer.form.pre.result_types
                [producer_scan_inputs..producer_scan_inputs + producer_reduction_inputs]
                .iter()
                .cloned(),
        )
        .chain(
            consumer.form.pre.result_types[consumer_scan_inputs..consumer_collective_inputs]
                .iter()
                .cloned(),
        )
        .chain(producer_mapped_types.iter().cloned())
        .chain(remaining_slots.iter().map(|&slot| consumer.inputs[slot].element()))
        .collect::<Vec<_>>();
    let (pre, pre_function) = finish_lambda(
        context,
        "vertical_middle_pre",
        pre_graph,
        pre_params,
        pre_captures,
        input_element_types,
        pre_result_types,
        pre_results,
    );

    let producer_scan_types = producer
        .form
        .scans
        .iter()
        .flat_map(|scan| scan.operator.result_types.iter().cloned())
        .collect::<Vec<_>>();
    let consumer_scan_types = consumer
        .form
        .scans
        .iter()
        .flat_map(|scan| scan.operator.result_types.iter().cloned())
        .collect::<Vec<_>>();
    let forwarded_types =
        remaining_slots.iter().map(|&slot| consumer.inputs[slot].element()).collect::<Vec<_>>();
    let post_parameter_types = producer_scan_types
        .iter()
        .cloned()
        .chain(consumer_scan_types.iter().cloned())
        .chain(producer_mapped_types.iter().cloned())
        .chain(forwarded_types.iter().cloned())
        .collect::<Vec<_>>();
    let post_captures = producer
        .form
        .post
        .seg_body()
        .into_iter()
        .flat_map(|body| body.captures.iter().copied())
        .chain(consumer.form.pre.seg_body().into_iter().flat_map(|body| body.captures.iter().copied()))
        .chain(consumer.form.post.seg_body().into_iter().flat_map(|body| body.captures.iter().copied()))
        .collect::<Vec<_>>();
    let mut post_params = lambda_ops::named_parameters(&post_parameter_types, "value");
    post_params.extend(lambda_ops::named_parameters(
        &capture_types(context.outer_types, post_captures.iter()),
        "capture",
    ));
    let mut post_graph = EGraph::new();
    let post_arguments = lambda_ops::function_parameters(&mut post_graph, &post_params);
    let mut post_capture_cursor = post_parameter_types.len();
    let producer_scan_end = producer_scan_types.len();
    let consumer_scan_end = producer_scan_end + consumer_scan_types.len();
    let producer_mapped_end = consumer_scan_end + producer_mapped_types.len();

    let mut producer_post_arguments = post_arguments[..producer_scan_end].to_vec();
    producer_post_arguments.extend_from_slice(&post_arguments[consumer_scan_end..producer_mapped_end]);
    append_wrapper_captures(
        &mut producer_post_arguments,
        &post_arguments,
        &mut post_capture_cursor,
        &producer.form.post,
    );
    let producer_post_results = invoke_lambda(
        &mut post_graph,
        context.program,
        &producer.form.post,
        producer_post_arguments,
    );
    let mut consumer_pre_arguments = (0..consumer.inputs.len())
        .map(|slot| {
            if let Some(output) = routed_producer_post_output(routes, slot) {
                producer_post_results[output]
            } else {
                let position = remaining_slots
                    .iter()
                    .position(|candidate| *candidate == slot)
                    .expect("remaining consumer input");
                post_arguments[producer_mapped_end + position]
            }
        })
        .collect::<Vec<_>>();
    append_wrapper_captures(
        &mut consumer_pre_arguments,
        &post_arguments,
        &mut post_capture_cursor,
        &consumer.form.pre,
    );
    let consumer_pre_results = invoke_lambda(
        &mut post_graph,
        context.program,
        &consumer.form.pre,
        consumer_pre_arguments,
    );
    let mut consumer_post_arguments = post_arguments[producer_scan_end..consumer_scan_end].to_vec();
    consumer_post_arguments.extend_from_slice(&consumer_pre_results[consumer_collective_inputs..]);
    append_wrapper_captures(
        &mut consumer_post_arguments,
        &post_arguments,
        &mut post_capture_cursor,
        &consumer.form.post,
    );
    let mut post_results = invoke_lambda(
        &mut post_graph,
        context.program,
        &consumer.form.post,
        consumer_post_arguments,
    );
    post_results
        .extend(retained_producer_post_outputs.iter().map(|(_, post)| producer_post_results[*post]));
    debug_assert_eq!(post_capture_cursor, post_arguments.len());
    let post_result_types = consumer
        .form
        .post
        .result_types
        .iter()
        .cloned()
        .chain(
            retained_producer_post_outputs
                .iter()
                .map(|(_, post)| producer.form.post.result_types[*post].clone()),
        )
        .collect();
    let (post, post_function) = finish_lambda(
        context,
        "vertical_middle_post",
        post_graph,
        post_params,
        post_captures,
        post_parameter_types,
        post_result_types,
        post_results,
    );

    Some(Normalized {
        input_nodes,
        inputs: input_array_types.into_iter().map(|array| SoacInputType { array }).collect(),
        form: screma::ScremaForm {
            pre,
            scans: producer.form.scans.iter().chain(&consumer.form.scans).cloned().collect(),
            reductions: producer.form.reductions.iter().chain(&consumer.form.reductions).cloned().collect(),
            post,
        },
        outputs: (0..producer_reduction_results)
            .map(OutputOrigin::Producer)
            .chain((0..consumer_reduction_results).map(OutputOrigin::Consumer))
            .chain((consumer_reduction_results..consumer.form.result_count()).map(OutputOrigin::Consumer))
            .chain(retained_producer_post_outputs.iter().map(|(output, _)| OutputOrigin::Producer(*output)))
            .collect(),
        synthesized: pre_function.into_iter().chain(post_function).collect(),
    })
}

fn lambda_results_projectable(
    program: &Segmented,
    lambda: &screma::Lambda,
    results: std::ops::Range<usize>,
) -> bool {
    if results.end > lambda.result_types.len() {
        return false;
    }
    lambda.is_identity() || results.is_empty() || lambda_result_roots(program, lambda).is_some()
}

fn lambda_results_depend_on_parameters(
    program: &Segmented,
    lambda: &screma::Lambda,
    results: std::ops::Range<usize>,
    parameters: &[usize],
) -> Option<bool> {
    if results.end > lambda.result_types.len() {
        return None;
    }
    if results.is_empty() || parameters.is_empty() {
        return Some(false);
    }
    if lambda.is_identity() {
        return Some(results.into_iter().any(|result| parameters.contains(&result)));
    }

    let body = lambda.seg_body()?;
    let function = program.region(body.region)?;
    let roots = lambda_result_roots(program, lambda)?;
    let closure = graph_ops::value_producer_closure(&function.graph, results.map(|result| roots[result]));
    Some(closure.nodes.into_iter().any(|node| {
        matches!(
            function.graph.nodes.get(node).map(|node| &node.kind),
            Some(ENode::FuncParam { index }) if parameters.contains(index)
        )
    }))
}

fn lambda_result_roots(program: &Segmented, lambda: &screma::Lambda) -> Option<Vec<NodeId>> {
    let body = lambda.seg_body()?;
    let function = program.region(body.region)?;
    let root = inlining::inlineable_return_root(function)?;
    match lambda.result_types.as_slice() {
        [_] => Some(vec![root]),
        results => match &function.graph.nodes.get(root)?.kind {
            ENode::Pure {
                op: PureOp::Tuple(arity),
                operands,
            } if *arity == results.len() && operands.len() == results.len() => Some(operands.to_vec()),
            _ => None,
        },
    }
}

fn emit_projected_lambda_results(
    graph: &mut EGraph,
    program: &Segmented,
    lambda: &screma::Lambda,
    arguments: &[Option<NodeId>],
    results: std::ops::Range<usize>,
) -> Option<Vec<NodeId>> {
    let indices = results.collect::<Vec<_>>();
    emit_projected_lambda_result_indices(graph, program, lambda, arguments, &indices)
}

fn emit_projected_lambda_result_indices(
    graph: &mut EGraph,
    program: &Segmented,
    lambda: &screma::Lambda,
    arguments: &[Option<NodeId>],
    results: &[usize],
) -> Option<Vec<NodeId>> {
    if results.iter().any(|result| *result >= lambda.result_types.len()) {
        return None;
    }
    if lambda.is_identity() {
        return results.iter().map(|result| arguments.get(*result).copied().flatten()).collect();
    }
    if results.is_empty() {
        return Some(Vec::new());
    }

    let body = lambda.seg_body()?;
    let function = program.region(body.region)?;
    if arguments.len() != function.params.len() {
        return None;
    }
    let roots = lambda_result_roots(program, lambda)?;
    let mut memo = LookupMap::new();
    for (node, definition) in &function.graph.nodes {
        if let ENode::FuncParam { index } = &definition.kind {
            if let Some(Some(argument)) = arguments.get(*index) {
                memo.insert(node, *argument);
            }
        }
    }
    results
        .iter()
        .map(|result| {
            graph_ops::clone_value_subgraph(
                &function.graph,
                graph,
                roots[*result],
                &mut memo,
                graph_ops::ConstantCopy::Intern,
                true,
            )
            .ok()
        })
        .collect()
}
fn append_wrapper_captures(
    arguments: &mut Vec<NodeId>,
    wrapper_arguments: &[NodeId],
    cursor: &mut usize,
    lambda: &screma::Lambda,
) {
    let capture_count = lambda.capture_count();
    arguments.extend_from_slice(&wrapper_arguments[*cursor..*cursor + capture_count]);
    *cursor += capture_count;
}

fn append_optional_wrapper_captures(
    arguments: &mut Vec<Option<NodeId>>,
    wrapper_arguments: &[NodeId],
    cursor: &mut usize,
    lambda: &screma::Lambda,
) {
    let capture_count = lambda.capture_count();
    arguments.extend(wrapper_arguments[*cursor..*cursor + capture_count].iter().copied().map(Some));
    *cursor += capture_count;
}

fn invoke_lambda(
    graph: &mut EGraph,
    program: &Segmented,
    lambda: &screma::Lambda,
    arguments: Vec<NodeId>,
) -> Vec<NodeId> {
    let callee = lambda
        .seg_body()
        .map(|body| program.region(body.region).expect("Screma lambda region").name.as_str());
    lambda_ops::emit_call(graph, lambda, callee, arguments)
}
#[derive(Clone, Copy)]
struct ValueRef {
    call: usize,
    result: usize,
}

struct LambdaCall<'a> {
    lambda: &'a screma::Lambda,
    parameters: Vec<usize>,
}

fn parallel_pre(
    context: &mut Context<'_>,
    parameter_types: &[Type<TypeName>],
    producer: &screma::ScremaForm,
    consumer: &screma::ScremaForm,
    producer_parameters: Vec<usize>,
    consumer_parameters: Vec<usize>,
) -> (screma::Lambda, Option<SemanticFunc>) {
    let producer_scans = producer.scan_input_count();
    let consumer_scans = consumer.scan_input_count();
    let producer_reductions = producer.reduction_input_count();
    let consumer_reductions = consumer.reduction_input_count();
    let producer_mapped = producer.pre.result_types.len() - producer_scans - producer_reductions;
    let consumer_mapped = consumer.pre.result_types.len() - consumer_scans - consumer_reductions;

    let outputs = (0..producer_scans)
        .map(|result| ValueRef { call: 0, result })
        .chain((0..consumer_scans).map(|result| ValueRef { call: 1, result }))
        .chain((0..producer_reductions).map(|offset| ValueRef {
            call: 0,
            result: producer_scans + offset,
        }))
        .chain((0..consumer_reductions).map(|offset| ValueRef {
            call: 1,
            result: consumer_scans + offset,
        }))
        .chain((0..producer_mapped).map(|offset| ValueRef {
            call: 0,
            result: producer_scans + producer_reductions + offset,
        }))
        .chain((0..consumer_mapped).map(|offset| ValueRef {
            call: 1,
            result: consumer_scans + consumer_reductions + offset,
        }))
        .collect();

    parallel_lambdas(
        context,
        "horizontal_pre",
        parameter_types.to_vec(),
        vec![
            LambdaCall {
                lambda: &producer.pre,
                parameters: producer_parameters,
            },
            LambdaCall {
                lambda: &consumer.pre,
                parameters: consumer_parameters,
            },
        ],
        outputs,
    )
}

fn parallel_post(
    context: &mut Context<'_>,
    producer: &screma::ScremaForm,
    consumer: &screma::ScremaForm,
) -> (screma::Lambda, Option<SemanticFunc>) {
    let producer_scans = producer.scan_input_count();
    let consumer_scans = consumer.scan_input_count();
    let producer_mapped = producer.mapped_types().expect("validated producer Screma").len();
    let consumer_mapped = consumer.mapped_types().expect("validated consumer Screma").len();
    let scan_count = producer_scans + consumer_scans;

    let parameter_types = producer
        .scans
        .iter()
        .flat_map(|scan| scan.operator.result_types.iter().cloned())
        .chain(consumer.scans.iter().flat_map(|scan| scan.operator.result_types.iter().cloned()))
        .chain(producer.mapped_types().expect("validated producer Screma").iter().cloned())
        .chain(consumer.mapped_types().expect("validated consumer Screma").iter().cloned())
        .collect::<Vec<_>>();
    let producer_parameters = (0..producer_scans).chain(scan_count..scan_count + producer_mapped).collect();
    let consumer_parameters = (producer_scans..scan_count)
        .chain(scan_count + producer_mapped..scan_count + producer_mapped + consumer_mapped)
        .collect();
    let outputs = (0..producer.post.result_types.len())
        .map(|result| ValueRef { call: 0, result })
        .chain((0..consumer.post.result_types.len()).map(|result| ValueRef { call: 1, result }))
        .collect();

    parallel_lambdas(
        context,
        "horizontal_post",
        parameter_types,
        vec![
            LambdaCall {
                lambda: &producer.post,
                parameters: producer_parameters,
            },
            LambdaCall {
                lambda: &consumer.post,
                parameters: consumer_parameters,
            },
        ],
        outputs,
    )
}

fn parallel_lambdas(
    context: &mut Context<'_>,
    label: &str,
    parameter_types: Vec<Type<TypeName>>,
    calls: Vec<LambdaCall<'_>>,
    outputs: Vec<ValueRef>,
) -> (screma::Lambda, Option<SemanticFunc>) {
    let result_types = outputs
        .iter()
        .map(|output| calls[output.call].lambda.result_types[output.result].clone())
        .collect::<Vec<_>>();
    let is_identity = calls.iter().all(|call| call.lambda.is_identity())
        && result_types == parameter_types
        && outputs.len() == parameter_types.len()
        && outputs
            .iter()
            .enumerate()
            .all(|(index, output)| calls[output.call].parameters.get(output.result) == Some(&index));
    if is_identity {
        return (screma::Lambda::identity(parameter_types), None);
    }

    let captures = calls
        .iter()
        .flat_map(|call| call.lambda.seg_body().into_iter().flat_map(|body| body.captures.iter().copied()))
        .collect::<Vec<_>>();
    let capture_types = capture_types(context.outer_types, captures.iter());
    let mut params = lambda_ops::named_parameters(&parameter_types, "element");
    params.extend(lambda_ops::named_parameters(&capture_types, "capture"));
    let mut graph = EGraph::new();
    let arguments = lambda_ops::function_parameters(&mut graph, &params);
    let mut capture_cursor = parameter_types.len();
    let mut call_results = Vec::with_capacity(calls.len());
    for call in &calls {
        let mut call_arguments = call.parameters.iter().map(|&index| arguments[index]).collect::<Vec<_>>();
        if let Some(body) = call.lambda.seg_body() {
            let capture_end = capture_cursor + body.captures.len();
            call_arguments.extend_from_slice(&arguments[capture_cursor..capture_end]);
            capture_cursor = capture_end;
            call_results.push(invoke_lambda(
                &mut graph,
                context.program,
                call.lambda,
                call_arguments,
            ));
        } else {
            call_results.push(call_arguments);
        }
    }
    debug_assert_eq!(capture_cursor, arguments.len());
    let selected =
        outputs.iter().map(|output| call_results[output.call][output.result]).collect::<Vec<_>>();
    finish_lambda(
        context,
        label,
        graph,
        params,
        captures,
        parameter_types,
        result_types,
        selected,
    )
}

#[derive(Clone, Copy)]
enum VerticalValueRef {
    Producer(usize),
    Consumer(usize),
}

#[allow(clippy::too_many_arguments)]
fn vertical_lambda(
    context: &mut Context<'_>,
    label: &str,
    parameter_types: &[Type<TypeName>],
    producer: &screma::Lambda,
    producer_parameters: Vec<usize>,
    producer_route_offset: usize,
    consumer: &screma::Lambda,
    consumer_parameters: &[Option<usize>],
    routes: &[InputRoute],
    outputs: &[VerticalValueRef],
) -> (screma::Lambda, Option<SemanticFunc>) {
    let producer_body = producer.seg_body();
    let consumer_body = consumer.seg_body();
    let captures = producer_body
        .into_iter()
        .flat_map(|body| body.captures.iter().copied())
        .chain(consumer_body.into_iter().flat_map(|body| body.captures.iter().copied()))
        .collect::<Vec<_>>();
    let capture_types = capture_types(context.outer_types, captures.iter());
    let mut params = lambda_ops::named_parameters(parameter_types, "element");
    params.extend(lambda_ops::named_parameters(&capture_types, "capture"));
    let mut graph = EGraph::new();
    let arguments = lambda_ops::function_parameters(&mut graph, &params);
    let mut capture_cursor = parameter_types.len();

    let mut producer_arguments =
        producer_parameters.iter().map(|&index| arguments[index]).collect::<Vec<_>>();
    let produced = if let Some(body) = producer_body {
        let capture_end = capture_cursor + body.captures.len();
        producer_arguments.extend_from_slice(&arguments[capture_cursor..capture_end]);
        capture_cursor = capture_end;
        invoke_lambda(&mut graph, context.program, producer, producer_arguments)
    } else {
        producer_arguments
    };

    let mut consumer_arguments = consumer_parameters
        .iter()
        .enumerate()
        .map(|(slot, parameter)| {
            routed_producer_post_output(routes, slot)
                .map(|output| produced[producer_route_offset + output])
                .or_else(|| parameter.map(|index| arguments[index]))
                .expect("consumer input is routed or retained")
        })
        .collect::<Vec<_>>();
    let consumed = if let Some(body) = consumer_body {
        let capture_end = capture_cursor + body.captures.len();
        consumer_arguments.extend_from_slice(&arguments[capture_cursor..capture_end]);
        capture_cursor = capture_end;
        invoke_lambda(&mut graph, context.program, consumer, consumer_arguments)
    } else {
        consumer_arguments
    };
    debug_assert_eq!(capture_cursor, arguments.len());

    let result_types = outputs
        .iter()
        .map(|output| match output {
            VerticalValueRef::Producer(index) => producer.result_types[*index].clone(),
            VerticalValueRef::Consumer(index) => consumer.result_types[*index].clone(),
        })
        .collect();
    let results = outputs
        .iter()
        .map(|output| match output {
            VerticalValueRef::Producer(index) => produced[*index],
            VerticalValueRef::Consumer(index) => consumed[*index],
        })
        .collect();
    let (lambda, function) = finish_lambda(
        context,
        label,
        graph,
        params,
        captures,
        parameter_types.to_vec(),
        result_types,
        results,
    );
    (lambda, function)
}
#[allow(clippy::too_many_arguments)]
fn finish_lambda(
    context: &mut Context<'_>,
    label: &str,
    graph: EGraph,
    params: Vec<(Type<TypeName>, String)>,
    captures: Vec<NodeId>,
    parameter_types: Vec<Type<TypeName>>,
    result_types: Vec<Type<TypeName>>,
    results: Vec<NodeId>,
) -> (screma::Lambda, Option<SemanticFunc>) {
    let return_block = graph.skeleton.entry;
    lambda_ops::finish_region_lambda(
        context.interner,
        context.scope,
        label,
        context.span,
        graph,
        return_block,
        params,
        captures,
        parameter_types,
        result_types,
        results,
        true,
    )
}
