//! Screma barrier algebra over owned symbolic lambda recipes.
use super::recipe::{self as screma, Builder, Lambda, NodeId, Recipes, TypeId};
use super::snapshot::deduplicate_inputs as deduplicate_array_inputs;
use super::snapshot::ArrayInput;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum OutputOrigin {
    Producer(usize),
    Consumer(usize),
}

pub(super) struct Normalized {
    pub inputs: Vec<ArrayInput>,
    pub form: screma::ScremaForm,
    /// Canonical fused field order, expressed in the source operations' field
    /// spaces. Independent siblings retain left-to-right post-result order.
    pub outputs: Vec<OutputOrigin>,
}

pub(super) struct Source<'a> {
    pub inputs: &'a [ArrayInput],
    pub form: &'a screma::ScremaForm,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct InputRoute {
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

pub(super) struct Context<'a> {
    pub recipes: &'a mut Recipes,
}

/// Horizontal normalisation using the SuperScrema barrier algebra. Independent
/// collective groups move to the first barrier; the wrapper forms the
/// associative product and preserves left-to-right results within each partition.
pub(super) fn fuse_horizontal(
    context: &mut Context<'_>,
    producer: Source<'_>,
    consumer: Source<'_>,
) -> Option<Normalized> {
    let producer_input_count = producer.inputs.len();
    let mut raw_inputs = producer.inputs.to_vec();
    raw_inputs.extend_from_slice(consumer.inputs);
    let (inputs, remap) = deduplicate_array_inputs(raw_inputs);
    let input_element_types = inputs.iter().map(ArrayInput::element).collect::<Vec<_>>();

    let producer_parameters = remap[..producer_input_count].to_vec();
    let consumer_parameters = remap[producer_input_count..].to_vec();
    let pre = parallel_pre(
        context,
        &input_element_types,
        producer.form,
        consumer.form,
        producer_parameters,
        consumer_parameters,
    )?;
    let post = parallel_post(context, producer.form, consumer.form)?;

    let producer_reductions = producer.form.reduction_result_count();
    let consumer_reductions = consumer.form.reduction_result_count();
    let outputs = (0..producer_reductions)
        .map(OutputOrigin::Producer)
        .chain((0..consumer_reductions).map(OutputOrigin::Consumer))
        .chain((producer_reductions..producer.form.result_count()).map(OutputOrigin::Producer))
        .chain((consumer_reductions..consumer.form.result_count()).map(OutputOrigin::Consumer))
        .collect();

    Some(Normalized {
        inputs,
        form: screma::ScremaForm {
            pre,
            scans: producer.form.scans.iter().chain(&consumer.form.scans).cloned().collect(),
            reductions: producer.form.reductions.iter().chain(&consumer.form.reductions).cloned().collect(),
            post,
        },
        outputs,
    })
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

pub(super) fn can_fuse_vertical(
    producer: &screma::ScremaForm,
    consumer: &screma::ScremaForm,
    routes: &[InputRoute],
) -> bool {
    if routes.iter().any(|route| route.producer_post_output >= producer.post.result_types.len()) {
        return false;
    }
    if producer.scans.is_empty() {
        return producer.post.is_identity();
    }

    let collective_results = consumer.operator_input_count();
    if collective_results == 0 {
        return true;
    }
    let collective = 0..collective_results;
    if !consumer.pre.projectable(collective.clone()) {
        return false;
    }
    let scan_parameters = (0..producer.scan_input_count()).collect::<Vec<_>>();
    routes.iter().all(|route| {
        let producer_output = route.producer_post_output;
        let depends_on_route = consumer.pre.depends_on(collective.clone(), &[route.consumer_input]);
        !depends_on_route
            || (producer.post.projectable(producer_output..producer_output + 1)
                && !producer.post.depends_on(producer_output..producer_output + 1, &scan_parameters))
    })
}

pub(super) fn fuse_vertical(
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

pub(super) struct LambdaSource<'a> {
    pub inputs: &'a [ArrayInput],
    pub lambda: &'a screma::Lambda,
}

pub(super) struct NormalizedLambda {
    pub inputs: Vec<ArrayInput>,
    pub lambda: screma::Lambda,
}

/// Compose a pure map producer into an arbitrary element lambda. This is the
/// common Futhark `fuseMaps` operation used by non-Screma envelopes.
pub(super) fn fuse_map_into_lambda(
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
    let mut raw_inputs =
        remaining_slots.iter().map(|&slot| consumer.inputs[slot].clone()).collect::<Vec<_>>();
    raw_inputs.extend_from_slice(producer.inputs);
    let (inputs, remap) = deduplicate_array_inputs(raw_inputs);
    let input_element_types = inputs.iter().map(ArrayInput::element).collect::<Vec<_>>();

    let producer_base = remaining_slots.len();
    let producer_parameters = remap[producer_base..].to_vec();
    let consumer_parameters = (0..consumer.inputs.len())
        .map(|slot| {
            remaining_slots.iter().position(|candidate| *candidate == slot).map(|position| remap[position])
        })
        .collect::<Vec<_>>();
    let outputs =
        (0..consumer.lambda.result_types.len()).map(VerticalValueRef::Consumer).collect::<Vec<_>>();
    let lambda = vertical_lambda(
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
    )?;

    Some(NormalizedLambda { inputs, lambda })
}
impl SuperScrema<'_> {
    fn normalize(self, context: &mut Context<'_>) -> Option<Normalized> {
        if self.producer.form.scans.is_empty() && self.producer.form.post.is_identity() {
            return fuse_scanless_producer(
                context,
                self.producer,
                self.consumer,
                self.routes,
                self.retained_producer_outputs,
            );
        }
        if can_fuse_vertical(self.producer.form, self.consumer.form, self.routes) {
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
) -> Option<Normalized> {
    let remaining_slots = (0..consumer.inputs.len())
        .filter(|slot| routed_producer_post_output(routes, *slot).is_none())
        .collect::<Vec<_>>();
    let mut raw_inputs =
        remaining_slots.iter().map(|&slot| consumer.inputs[slot].clone()).collect::<Vec<_>>();
    raw_inputs.extend_from_slice(producer.inputs);
    let (inputs, remap) = deduplicate_array_inputs(raw_inputs);
    let input_element_types = inputs.iter().map(ArrayInput::element).collect::<Vec<_>>();

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
    let pre = vertical_lambda(
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
    )?;

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
    let post = if consumer.form.scans.is_empty() {
        screma::Lambda::identity(post_parameter_types)
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
        )?
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

    Some(Normalized {
        inputs,
        form: screma::ScremaForm {
            pre,
            scans: consumer.form.scans.clone(),
            reductions: producer.form.reductions.iter().chain(&consumer.form.reductions).cloned().collect(),
            post,
        },
        outputs,
    })
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
    let mut raw_inputs = producer.inputs.to_vec();
    raw_inputs.extend(remaining_slots.iter().map(|&slot| consumer.inputs[slot].clone()));
    let (inputs, remap) = deduplicate_array_inputs(raw_inputs);
    let input_element_types = inputs.iter().map(ArrayInput::element).collect::<Vec<_>>();

    let producer_parameters = remap[..producer_input_count].to_vec();
    let forwarded_parameters = remap[producer_input_count..].to_vec();
    let producer_scan_inputs = producer.form.scan_input_count();
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
            consumer.form.pre.depends_on(0..consumer_collective_inputs, &[route.consumer_input])
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
    let mut pre_graph = Builder::new(input_element_types.clone(), pre_captures.clone());
    let pre_arguments = pre_graph.arguments.clone();
    let mut pre_capture_cursor = input_element_types.len();

    let mut producer_pre_arguments =
        producer_parameters.iter().map(|&index| pre_arguments[index]).collect::<Vec<_>>();
    append_wrapper_captures(
        &mut producer_pre_arguments,
        &pre_arguments,
        &mut pre_capture_cursor,
        &producer.form.pre,
    )?;
    let producer_pre_results = invoke_lambda(
        &mut pre_graph,
        context,
        &producer.form.pre,
        producer_pre_arguments,
    )?;
    let producer_mapped = producer_pre_results[producer_operator_inputs..].to_vec();

    let mut produced_before_barrier = crate::LookupMap::new();
    if !dependent_routes.is_empty() {
        let mut producer_post_arguments = vec![None; producer_scan_inputs];
        producer_post_arguments.extend(producer_mapped.iter().copied().map(Some));
        append_optional_wrapper_captures(
            &mut producer_post_arguments,
            &pre_arguments,
            &mut pre_capture_cursor,
            &producer.form.post,
        )?;
        let mut outputs =
            dependent_routes.iter().map(|route| route.producer_post_output).collect::<Vec<_>>();
        outputs.sort_unstable();
        outputs.dedup();
        let values = pre_graph.project(
            context.recipes,
            &producer.form.post,
            &producer_post_arguments,
            &outputs,
        )?;
        produced_before_barrier.extend(outputs.into_iter().zip(values));
    }

    let consumer_pre_arguments = (0..consumer.inputs.len())
        .map(|slot| {
            if let Some(output) = routed_producer_post_output(routes, slot) {
                produced_before_barrier.get(&output).copied()
            } else {
                let position = remaining_slots.iter().position(|candidate| *candidate == slot)?;
                pre_arguments.get(forwarded_parameters[position]).copied()
            }
        })
        .collect::<Vec<_>>();
    let mut consumer_pre_arguments = consumer_pre_arguments;
    append_optional_wrapper_captures(
        &mut consumer_pre_arguments,
        &pre_arguments,
        &mut pre_capture_cursor,
        &consumer.form.pre,
    )?;
    let consumer_collective = pre_graph.project(
        context.recipes,
        &consumer.form.pre,
        &consumer_pre_arguments,
        &(0..consumer_collective_inputs).collect::<Vec<_>>(),
    )?;
    debug_assert_eq!(pre_capture_cursor, pre_arguments.len());

    let forwarded_results = forwarded_parameters
        .iter()
        .map(|index| pre_arguments.get(*index).copied())
        .collect::<Option<Vec<_>>>()?;
    let pre_results = producer_pre_results[..producer_scan_inputs]
        .iter()
        .copied()
        .chain(consumer_collective[..consumer_scan_inputs].iter().copied())
        .chain(producer_pre_results[producer_scan_inputs..producer_operator_inputs].iter().copied())
        .chain(consumer_collective[consumer_scan_inputs..].iter().copied())
        .chain(producer_mapped.iter().copied())
        .chain(forwarded_results)
        .collect::<Vec<_>>();
    let pre = pre_graph.finish_named(context.recipes, pre_results, "vertical_middle_pre");

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
    let mut post_graph = Builder::new(post_parameter_types.clone(), post_captures.clone());
    let post_arguments = post_graph.arguments.clone();
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
    )?;
    let producer_post_results = invoke_lambda(
        &mut post_graph,
        context,
        &producer.form.post,
        producer_post_arguments,
    )?;
    let consumer_pre_arguments = (0..consumer.inputs.len())
        .map(|slot| {
            if let Some(output) = routed_producer_post_output(routes, slot) {
                Some(*producer_post_results.get(output)?)
            } else {
                let position = remaining_slots.iter().position(|candidate| *candidate == slot)?;
                post_arguments.get(producer_mapped_end + position).copied()
            }
        })
        .collect::<Option<Vec<_>>>()?;
    let mut consumer_pre_arguments = consumer_pre_arguments;
    append_wrapper_captures(
        &mut consumer_pre_arguments,
        &post_arguments,
        &mut post_capture_cursor,
        &consumer.form.pre,
    )?;
    let consumer_pre_results = invoke_lambda(
        &mut post_graph,
        context,
        &consumer.form.pre,
        consumer_pre_arguments,
    )?;
    let mut consumer_post_arguments = post_arguments[producer_scan_end..consumer_scan_end].to_vec();
    consumer_post_arguments
        .extend(consumer_pre_results[consumer_collective_inputs..].iter().map(|value| *value));
    append_wrapper_captures(
        &mut consumer_post_arguments,
        &post_arguments,
        &mut post_capture_cursor,
        &consumer.form.post,
    )?;
    let mut post_results = invoke_lambda(
        &mut post_graph,
        context,
        &consumer.form.post,
        consumer_post_arguments,
    )?;
    post_results
        .extend(retained_producer_post_outputs.iter().map(|(_, post)| producer_post_results[*post]));
    debug_assert_eq!(post_capture_cursor, post_arguments.len());
    let post = post_graph.finish_named(context.recipes, post_results, "vertical_middle_post");

    Some(Normalized {
        inputs,
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
    })
}

fn parallel_pre(
    context: &mut Context<'_>,
    parameter_types: &[TypeId],
    producer: &screma::ScremaForm,
    consumer: &screma::ScremaForm,
    producer_parameters: Vec<usize>,
    consumer_parameters: Vec<usize>,
) -> Option<screma::Lambda> {
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
) -> Option<screma::Lambda> {
    let producer_scans = producer.scan_input_count();
    let consumer_scans = consumer.scan_input_count();
    let producer_mapped_types = producer.mapped_types()?;
    let consumer_mapped_types = consumer.mapped_types()?;
    let producer_mapped = producer_mapped_types.len();
    let consumer_mapped = consumer_mapped_types.len();
    let scan_count = producer_scans + consumer_scans;

    let parameter_types = producer
        .scans
        .iter()
        .flat_map(|scan| scan.operator.result_types.iter().cloned())
        .chain(consumer.scans.iter().flat_map(|scan| scan.operator.result_types.iter().cloned()))
        .chain(producer_mapped_types.iter().cloned())
        .chain(consumer_mapped_types.iter().cloned())
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

fn append_wrapper_captures(
    arguments: &mut Vec<NodeId>,
    wrapper: &[NodeId],
    cursor: &mut usize,
    lambda: &Lambda,
) -> Option<()> {
    arguments.extend_from_slice(wrapper.get(*cursor..*cursor + lambda.capture_count())?);
    *cursor += lambda.capture_count();
    Some(())
}
fn append_optional_wrapper_captures(
    arguments: &mut Vec<Option<NodeId>>,
    wrapper: &[NodeId],
    cursor: &mut usize,
    lambda: &Lambda,
) -> Option<()> {
    arguments.extend(wrapper.get(*cursor..*cursor + lambda.capture_count())?.iter().copied().map(Some));
    *cursor += lambda.capture_count();
    Some(())
}
fn invoke_lambda(
    graph: &mut Builder,
    context: &Context<'_>,
    lambda: &Lambda,
    arguments: Vec<NodeId>,
) -> Option<Vec<NodeId>> {
    graph.invoke(context.recipes, lambda, arguments)
}
#[derive(Clone, Copy)]
struct ValueRef {
    call: usize,
    result: usize,
}
struct LambdaCall<'a> {
    lambda: &'a Lambda,
    parameters: Vec<usize>,
}
fn parallel_lambdas(
    context: &mut Context<'_>,
    label: &str,
    parameter_types: Vec<TypeId>,
    calls: Vec<LambdaCall<'_>>,
    outputs: Vec<ValueRef>,
) -> Option<Lambda> {
    let captures = calls.iter().flat_map(|call| call.lambda.captures().iter().copied()).collect();
    let mut graph = Builder::new(parameter_types.clone(), captures);
    let arguments = graph.arguments.clone();
    let mut cursor = parameter_types.len();
    let mut results = Vec::new();
    for call in calls {
        let mut args = call.parameters.iter().map(|slot| arguments[*slot]).collect();
        append_wrapper_captures(&mut args, &arguments, &mut cursor, call.lambda)?;
        results.push(graph.invoke(context.recipes, call.lambda, args)?);
    }
    let results = outputs.iter().map(|output| results[output.call][output.result]).collect();
    Some(graph.finish_named(context.recipes, results, label))
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
    parameter_types: &[TypeId],
    producer: &Lambda,
    producer_parameters: Vec<usize>,
    producer_route_offset: usize,
    consumer: &Lambda,
    consumer_parameters: &[Option<usize>],
    routes: &[InputRoute],
    outputs: &[VerticalValueRef],
) -> Option<Lambda> {
    let captures = producer.captures().iter().chain(consumer.captures()).copied().collect();
    let mut graph = Builder::new(parameter_types.to_vec(), captures);
    let arguments = graph.arguments.clone();
    let mut cursor = parameter_types.len();
    let mut producer_args = producer_parameters.iter().map(|slot| arguments[*slot]).collect();
    append_wrapper_captures(&mut producer_args, &arguments, &mut cursor, producer)?;
    let produced = graph.invoke(context.recipes, producer, producer_args)?;
    let mut consumer_args = consumer_parameters
        .iter()
        .enumerate()
        .map(|(slot, parameter)| {
            routed_producer_post_output(routes, slot)
                .map(|output| produced[producer_route_offset + output])
                .or_else(|| parameter.map(|index| arguments[index]))
        })
        .collect::<Option<Vec<_>>>()?;
    append_wrapper_captures(&mut consumer_args, &arguments, &mut cursor, consumer)?;
    let consumed = graph.invoke(context.recipes, consumer, consumer_args)?;
    let results = outputs
        .iter()
        .map(|output| match output {
            VerticalValueRef::Producer(index) => produced[*index],
            VerticalValueRef::Consumer(index) => consumed[*index],
        })
        .collect();
    Some(graph.finish_named(context.recipes, results, label))
}
