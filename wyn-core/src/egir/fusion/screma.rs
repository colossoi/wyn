//! Futhark-style Screma fusion algebra.
//!
//! Graph discovery and EGIR effect rewiring live in the horizontal/vertical
//! modules. This module is the representation-level core: it constructs the
//! equivalent of Futhark's `SuperScrema`, normalises the legal barrier ordering,
//! deduplicates forwarded inputs, and synthesises whole pre/post lambdas.

use polytype::Type;
use smallvec::smallvec;

use super::{capture_types, deduplicate_array_inputs};
use crate::ast::{Span, TypeName};
use crate::egir::program::{fresh_region_name, RegionInterner, SemanticFunc};
use crate::egir::reify::Segmented;
use crate::egir::soac::screma;
use crate::egir::types::{EGraph, NodeId, PureOp, SegBody, SkeletonTerminator, SoacInputType};
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
    consumer_inputs_from_producer: &'a [usize],
    producer_output: usize,
}

pub(crate) fn can_fuse_vertical(producer: &screma::ScremaForm, _consumer: &screma::ScremaForm) -> bool {
    // No lambda split is needed when the first collective group is empty:
    // all producer element work can move directly into the consumer pre.
    // Other shapes will be enabled here only with the dependency-based lambda
    // partition corresponding to Futhark's `splitLambdaByPar`.
    producer.scans.is_empty() && producer.reductions.is_empty() && producer.post.is_identity()
}

pub(crate) fn fuse_vertical(
    context: &mut Context<'_>,
    producer: Source<'_>,
    consumer: Source<'_>,
    consumer_inputs_from_producer: &[usize],
    producer_output: usize,
) -> Option<Normalized> {
    SuperScrema {
        producer,
        consumer,
        consumer_inputs_from_producer,
        producer_output,
    }
    .normalize(context)
}

impl SuperScrema<'_> {
    fn normalize(self, context: &mut Context<'_>) -> Option<Normalized> {
        can_fuse_vertical(self.producer.form, self.consumer.form).then(|| {
            fuse_before_first_barrier(
                context,
                self.producer,
                self.consumer,
                self.consumer_inputs_from_producer,
                self.producer_output,
            )
        })
    }
}

/// Normalize a SuperScrema whose producer has no collective barrier. Producer
/// element work composes into the consumer pre-lambda; only the consumer
/// barrier and post-lambda remain.
fn fuse_before_first_barrier(
    context: &mut Context<'_>,
    producer: Source<'_>,
    consumer: Source<'_>,
    consumer_inputs_from_producer: &[usize],
    producer_output: usize,
) -> Normalized {
    let remaining_slots = (0..consumer.inputs.len())
        .filter(|slot| !consumer_inputs_from_producer.contains(slot))
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
    let (pre, function) = vertical_pre(
        context,
        &input_element_types,
        &producer.form.pre,
        producer_parameters,
        producer_output,
        &consumer.form.pre,
        &consumer_parameters,
    );

    Normalized {
        input_nodes,
        inputs: input_array_types.into_iter().map(|array| SoacInputType { array }).collect(),
        form: screma::ScremaForm {
            pre,
            scans: consumer.form.scans.clone(),
            reductions: consumer.form.reductions.clone(),
            post: consumer.form.post.clone(),
        },
        outputs: (0..consumer.form.result_count()).map(OutputOrigin::Consumer).collect(),
        synthesized: vec![function],
    }
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
    let mut params = named_parameters(&parameter_types, "element");
    params.extend(named_parameters(&capture_types, "capture"));
    let mut graph = EGraph::new();
    let arguments = function_parameters(&mut graph, &params);
    let mut capture_cursor = parameter_types.len();
    let mut call_results = Vec::with_capacity(calls.len());
    for call in &calls {
        let mut call_arguments = call.parameters.iter().map(|&index| arguments[index]).collect::<Vec<_>>();
        if let Some(body) = call.lambda.seg_body() {
            let capture_end = capture_cursor + body.captures.len();
            call_arguments.extend_from_slice(&arguments[capture_cursor..capture_end]);
            capture_cursor = capture_end;
            call_results.push(call_lambda(
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

#[allow(clippy::too_many_arguments)]
fn vertical_pre(
    context: &mut Context<'_>,
    parameter_types: &[Type<TypeName>],
    producer: &screma::Lambda,
    producer_parameters: Vec<usize>,
    producer_output: usize,
    consumer: &screma::Lambda,
    consumer_parameters: &[Option<usize>],
) -> (screma::Lambda, SemanticFunc) {
    let producer_body = producer.seg_body();
    let consumer_body = consumer.seg_body();
    let captures = producer_body
        .into_iter()
        .flat_map(|body| body.captures.iter().copied())
        .chain(consumer_body.into_iter().flat_map(|body| body.captures.iter().copied()))
        .collect::<Vec<_>>();
    let capture_types = capture_types(context.outer_types, captures.iter());
    let mut params = named_parameters(parameter_types, "element");
    params.extend(named_parameters(&capture_types, "capture"));
    let mut graph = EGraph::new();
    let arguments = function_parameters(&mut graph, &params);
    let mut capture_cursor = parameter_types.len();

    let mut producer_arguments =
        producer_parameters.iter().map(|&index| arguments[index]).collect::<Vec<_>>();
    let produced = if let Some(body) = producer_body {
        let capture_end = capture_cursor + body.captures.len();
        producer_arguments.extend_from_slice(&arguments[capture_cursor..capture_end]);
        capture_cursor = capture_end;
        call_lambda(&mut graph, context.program, producer, producer_arguments)
    } else {
        producer_arguments
    }[producer_output];

    let mut consumer_arguments = consumer_parameters
        .iter()
        .map(|parameter| parameter.map_or(produced, |index| arguments[index]))
        .collect::<Vec<_>>();
    let results = if let Some(body) = consumer_body {
        let capture_end = capture_cursor + body.captures.len();
        consumer_arguments.extend_from_slice(&arguments[capture_cursor..capture_end]);
        capture_cursor = capture_end;
        call_lambda(&mut graph, context.program, consumer, consumer_arguments)
    } else {
        consumer_arguments
    };
    debug_assert_eq!(capture_cursor, arguments.len());

    let (lambda, function) = finish_lambda(
        context,
        "vertical_pre",
        graph,
        params,
        captures,
        parameter_types.to_vec(),
        consumer.result_types.clone(),
        results,
    );
    (lambda, function.expect("vertical composition is never identity"))
}

fn named_parameters(types: &[Type<TypeName>], prefix: &str) -> Vec<(Type<TypeName>, String)> {
    types.iter().enumerate().map(|(index, ty)| (ty.clone(), format!("{prefix}_{index}"))).collect()
}

fn function_parameters(graph: &mut EGraph, params: &[(Type<TypeName>, String)]) -> Vec<NodeId> {
    params.iter().enumerate().map(|(index, (ty, _))| graph.add_func_param(index, ty.clone())).collect()
}

fn call_lambda(
    graph: &mut EGraph,
    program: &Segmented,
    lambda: &screma::Lambda,
    arguments: Vec<NodeId>,
) -> Vec<NodeId> {
    let body = lambda.seg_body().expect("identity lambda calls are handled by the caller");
    let name = program.region(body.region).expect("Screma lambda region").name.clone();
    let result = graph.intern_pure(
        PureOp::Call(name),
        arguments.into_iter().collect(),
        lambda_return_type(&lambda.result_types),
        None,
    );
    unpack_result(graph, result, &lambda.result_types)
}

#[allow(clippy::too_many_arguments)]
fn finish_lambda(
    context: &mut Context<'_>,
    label: &str,
    mut graph: EGraph,
    params: Vec<(Type<TypeName>, String)>,
    captures: Vec<NodeId>,
    parameter_types: Vec<Type<TypeName>>,
    result_types: Vec<Type<TypeName>>,
    results: Vec<NodeId>,
) -> (screma::Lambda, Option<SemanticFunc>) {
    let return_type = lambda_return_type(&result_types);
    let result = match results.as_slice() {
        [result] => *result,
        results => graph.intern_pure(
            PureOp::Tuple(results.len()),
            results.iter().copied().collect(),
            return_type.clone(),
            None,
        ),
    };
    graph.skeleton.blocks[graph.skeleton.entry].term = SkeletonTerminator::Return(Some(result));
    let name = fresh_region_name(context.interner, &format!("{}_{}", context.scope, label));
    let region = context.interner.intern(&name);
    let function = SemanticFunc::new(region, name, context.span, None, params, return_type, graph);
    (
        screma::Lambda::region(SegBody { region, captures }, parameter_types, result_types),
        Some(function),
    )
}

fn lambda_return_type(results: &[Type<TypeName>]) -> Type<TypeName> {
    match results {
        [result] => result.clone(),
        results => Type::Constructed(TypeName::Tuple(results.len()), results.to_vec()),
    }
}

fn unpack_result(graph: &mut EGraph, result: NodeId, types: &[Type<TypeName>]) -> Vec<NodeId> {
    match types {
        [_] => vec![result],
        types => types
            .iter()
            .enumerate()
            .map(|(index, ty)| {
                graph.intern_pure(
                    PureOp::Project { index: index as u32 },
                    smallvec![result],
                    ty.clone(),
                    None,
                )
            })
            .collect(),
    }
}
