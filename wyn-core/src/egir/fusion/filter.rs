//! Fuse filters into scalar consumers.
//!
//! A filter whose array does not escape can be represented as masked reduction
//! steps over the original domain. Reductions consume a value only when the
//! predicate succeeds; `length` becomes one shared count reduction. This
//! removes both compaction work and its scratch/output requirements before
//! allocation and scheduling see them.

use std::collections::HashSet;

use polytype::Type;
use smallvec::{smallvec, SmallVec};

use super::envelope::splice_effects;
use super::vertical::{capture_types, fresh_region_name, graph_and_span, graph_mut, FusionSite};
use crate::ast::TypeName;
use crate::builtins::catalog;
use crate::egir::graph_ops;
use crate::egir::program::{OutputWriter, SemanticFunc, SemanticProgram, SemanticResourceRef};
use crate::egir::semantic_graph::SemanticGraph;
use crate::egir::soac::{filter, screma};
use crate::egir::types::{
    EGraph, ENode, NodeId, PureOp, SegBody, SegResourceAccess, SegSpace, SideEffect, SideEffectKind,
    SkeletonTerminator, Soac, SoacDestination, SoacInputType,
};
use crate::flow::{BlockId, ControlHeader};
use crate::LookupMap;

#[derive(Clone)]
struct Candidate {
    site: FusionSite,
    block: BlockId,
    filter: usize,
    consumer: Option<usize>,
    lengths: Vec<NodeId>,
}

#[derive(Clone)]
struct FilterParts {
    space: SegSpace,
    map: Option<SegBody>,
    pred: SegBody,
    input: NodeId,
    input_array_type: Type<TypeName>,
    input_elem_type: Type<TypeName>,
    scratch: Option<SemanticResourceRef>,
}

pub fn fuse_filter_consumers(inner: &mut SemanticProgram, oracle: &SemanticGraph) -> bool {
    let Some(candidate) = find_candidate(inner, oracle) else {
        return false;
    };
    apply(inner, candidate);
    true
}

fn find_candidate(inner: &SemanticProgram, oracle: &SemanticGraph) -> Option<Candidate> {
    for (index, entry) in inner.entry_points.iter().enumerate() {
        if let Some(candidate) = find_in_graph(
            &entry.graph,
            FusionSite::Entry(index),
            oracle,
            Some(&entry.output_routes),
        ) {
            return Some(candidate);
        }
    }
    for (index, function) in inner.functions.iter().enumerate() {
        if let Some(candidate) = find_in_graph(&function.graph, FusionSite::Function(index), oracle, None) {
            return Some(candidate);
        }
    }
    None
}

fn find_in_graph(
    graph: &EGraph,
    site: FusionSite,
    oracle: &SemanticGraph,
    routes: Option<&[crate::egir::program::OutputRoute]>,
) -> Option<Candidate> {
    let live = live_nodes(graph);
    for (block_id, block) in &graph.skeleton.blocks {
        for (filter_index, effect) in block.side_effects.iter().enumerate() {
            let SideEffectKind::Soac(filter_id, Soac::Filter(_)) = &effect.kind else {
                continue;
            };
            let Some(result) = effect.result else {
                continue;
            };
            let lengths: Vec<NodeId> = graph
                .nodes
                .iter()
                .filter_map(|(node, _)| {
                    (live.contains(&node) && is_length_of(graph, node, result)).then_some(node)
                })
                .collect();
            if lengths.iter().any(|length| graph.types[length] != graph.types[&lengths[0]]) {
                continue;
            }

            let consumers: Vec<usize> = block
                .side_effects
                .iter()
                .enumerate()
                .skip(filter_index + 1)
                .filter_map(|(index, consumer)| is_reduction_of_filter(consumer, result).then_some(index))
                .collect();
            if consumers.len() > 1 {
                continue;
            }
            let consumer = consumers.first().copied();
            if consumer.is_none() && lengths.is_empty() {
                continue;
            }
            let stops: HashSet<_> = lengths.iter().copied().collect();
            if routes.is_some_and(|routes| {
                routes.iter().any(|route| reaches_without(graph, route.source.value, result, &stops))
            }) {
                continue;
            }
            if let Some(consumer_index) = consumer {
                if !intervening_ops_are_safe(block, filter_index, consumer_index, filter_id, oracle) {
                    continue;
                }
            }
            if filter_result_escapes(graph, block_id, filter_index, consumer, result, &lengths) {
                continue;
            }
            return Some(Candidate {
                site,
                block: block_id,
                filter: filter_index,
                consumer,
                lengths,
            });
        }
    }
    None
}

fn live_nodes(graph: &EGraph) -> HashSet<NodeId> {
    let roots = graph.skeleton.blocks.iter().flat_map(|(_, block)| {
        block
            .side_effects
            .iter()
            .flat_map(SideEffect::referenced_nodes)
            .chain(block.term.referenced_nodes())
    });
    wyn_graph::reachable_set(roots, wyn_graph::WalkOrder::DepthFirst, |node, out| {
        out.extend(graph.nodes[node].children());
    })
}

fn is_length_of(graph: &EGraph, node: NodeId, filter_result: NodeId) -> bool {
    let ENode::Pure { op, operands } = &graph.nodes[node] else {
        return false;
    };
    if operands.as_slice() != [filter_result] {
        return false;
    }
    match op {
        PureOp::Intrinsic { id, .. } => *id == catalog().known().length,
        PureOp::UnaryOp(name) => *name == crate::builtins::by_id(catalog().known().length).dispatch_name(),
        _ => false,
    }
}

fn is_reduction_of_filter(effect: &SideEffect, filter_result: NodeId) -> bool {
    let SideEffectKind::Soac(_, Soac::Screma(op)) = &effect.kind else {
        return false;
    };
    let screma::Op::Reduce { lanes, operators, .. } = op else {
        return false;
    };
    let n_inputs = lanes.inputs.len();
    n_inputs != 0
        && lanes.maps.is_empty()
        && effect.operand_nodes[..n_inputs].iter().all(|&input| input == filter_result)
        && operators.iter().all(|operator| {
            operator.input_indices.len() == 1 && operator.input_indices[0].index() < n_inputs
        })
}

fn intervening_ops_are_safe(
    block: &crate::egir::types::SkeletonBlock,
    filter: usize,
    consumer: usize,
    filter_id: &crate::egir::program::SemanticOpId,
    oracle: &SemanticGraph,
) -> bool {
    ((filter + 1)..consumer).all(|index| {
        let effect = &block.side_effects[index];
        match &effect.kind {
            SideEffectKind::Soac(intervening, _) => !oracle.conflicts(filter_id, intervening),
            _ => effect.effects.is_none(),
        }
    })
}

fn filter_result_escapes(
    graph: &EGraph,
    filter_block: BlockId,
    filter_index: usize,
    consumer: Option<usize>,
    result: NodeId,
    lengths: &[NodeId],
) -> bool {
    let stops: HashSet<NodeId> = lengths.iter().copied().collect();
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
            if effect.referenced_nodes().any(|root| reaches_without(graph, root, result, &stops)) {
                return true;
            }
        }
        if block
            .term
            .referenced_nodes()
            .into_iter()
            .any(|root| reaches_without(graph, root, result, &stops))
        {
            return true;
        }
    }
    false
}

fn reaches_without(graph: &EGraph, root: NodeId, target: NodeId, stops: &HashSet<NodeId>) -> bool {
    if stops.contains(&root) {
        return false;
    }
    wyn_graph::reaches_ordered(root, target, wyn_graph::WalkOrder::DepthFirst, |node, out| {
        if !stops.contains(&node) {
            out.extend(graph.nodes[node].children());
        }
    })
}

fn filter_parts(effect: &SideEffect) -> FilterParts {
    let SideEffectKind::Soac(_, Soac::Filter(op)) = &effect.kind else {
        unreachable!();
    };
    let (map, input_type) = match &op.body.input {
        filter::Input::Plain(input) => (None, input),
        filter::Input::Mapped { input, body, .. } => (Some(body.clone()), input),
    };
    FilterParts {
        space: op.state.space.clone(),
        map,
        pred: op.body.predicate.clone(),
        input: effect.operand_nodes[0],
        input_array_type: input_type.array.clone(),
        input_elem_type: input_type.element.clone(),
        scratch: match &op.state.storage {
            filter::Output::Local { .. } => None,
            filter::Output::Runtime { scratch, .. } => Some(*scratch),
        },
    }
}

fn apply(inner: &mut SemanticProgram, candidate: Candidate) {
    let (filter_effect, consumer_effect, outer_types, span, scope) = {
        let (graph, span, scope) = graph_and_span(inner, candidate.site);
        let block = &graph.skeleton.blocks[candidate.block];
        (
            block.side_effects[candidate.filter].clone(),
            candidate.consumer.map(|consumer| block.side_effects[consumer].clone()),
            graph.types.clone(),
            span,
            scope,
        )
    };
    let filter = filter_parts(&filter_effect);

    let mut operators: Vec<screma::Operator> = consumer_effect
        .as_ref()
        .map(|effect| {
            let SideEffectKind::Soac(_, Soac::Screma(op)) = &effect.kind else {
                unreachable!();
            };
            op.operators().into_iter().cloned().collect()
        })
        .unwrap_or_default();
    for (index, operator) in operators.iter_mut().enumerate() {
        let (step, function) = masked_step(inner, &scope, span, index, &filter, operator, &outer_types);
        inner.define_region(function);
        operator.step = step;
        operator.input_indices = vec![screma::InputId(0)];
    }

    let count_ty = candidate.lengths.first().map(|length| outer_types[length].clone());
    let count = count_ty.as_ref().map(|ty| {
        let (mut operator, step_function, combine_function) =
            count_operator(inner, &scope, span, &filter, ty.clone(), &outer_types);
        let step_region = inner.define_region(step_function);
        operator.step.region = step_region;
        let combine_region = inner.define_region(combine_function);
        operator.combine.region = combine_region;
        operator
    });

    if let Some(consumer) = candidate.consumer {
        apply_with_consumer(
            inner,
            &candidate,
            filter_effect,
            consumer_effect.expect("filter reduction consumer disappeared"),
            filter,
            consumer,
            operators,
            count,
            count_ty,
        );
    } else {
        apply_count_only(
            inner,
            &candidate,
            filter_effect,
            filter,
            count.expect("length-only filter has no count operator"),
            count_ty.expect("length-only filter has no count type"),
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_with_consumer(
    inner: &mut SemanticProgram,
    candidate: &Candidate,
    filter_effect: SideEffect,
    consumer_effect: SideEffect,
    filter: FilterParts,
    consumer_index: usize,
    mut operators: Vec<screma::Operator>,
    count: Option<screma::Operator>,
    count_ty: Option<Type<TypeName>>,
) {
    let SideEffectKind::Soac(_, Soac::Screma(op)) = &consumer_effect.kind else {
        unreachable!();
    };
    let old_input_count = op.lanes().inputs.len();
    let old_result = consumer_effect.result.expect("filter reduction has no result");
    let mut result_types = op.result_types();
    let screma::SemanticState::Segmented {
        resources: old_resources,
        ..
    } = op.semantic_state()
    else {
        unreachable!();
    };
    let old_resources = old_resources.clone();

    let graph = graph_mut(inner, candidate.site);
    let count_neutral = count_ty.as_ref().map(|ty| integer_literal(graph, "0", ty));
    let count_project = if let (Some(mut count), Some(count_ty), Some(neutral)) =
        (count, count_ty.as_ref(), count_neutral)
    {
        count.neutral = neutral;
        let field = result_types.len() as u32;
        let new_result = extend_result(graph, old_result, &result_types, count_ty.clone());
        let project = graph.intern_pure(
            PureOp::Project { index: field },
            smallvec![new_result],
            count_ty.clone(),
            None,
        );
        result_types.push(count_ty.clone());
        operators.push(count);
        Some((new_result, project))
    } else {
        None
    };

    let output_views = consumer_effect.operand_nodes[old_input_count..].iter().copied();
    let mut operands = SmallVec::<[NodeId; 4]>::new();
    operands.push(filter.input);
    operands.extend(output_views);

    let fused_effects = splice_effects(filter_effect.effects, consumer_effect.effects);
    {
        let consumer = &mut graph.skeleton.blocks[candidate.block].side_effects[consumer_index];
        consumer.operand_nodes = operands;
        consumer.effects = fused_effects;
        if let Some((new_result, _)) = count_project {
            consumer.result = Some(new_result);
        }
        if let SideEffectKind::Soac(_, Soac::Screma(op)) = &mut consumer.kind {
            let mut state = op.semantic_state().clone();
            let screma::SemanticState::Segmented { space, .. } = &mut state else {
                unreachable!();
            };
            *space = filter.space.clone();
            for (operator, result_type) in operators.iter_mut().zip(&result_types) {
                operator.result_type = result_type.clone();
            }
            *op = screma::Op::Reduce {
                lanes: screma::Lanes {
                    inputs: vec![SoacInputType {
                        array: filter.input_array_type.clone(),
                        element: filter.input_elem_type.clone(),
                    }],
                    maps: Vec::new(),
                },
                operators: screma::NonEmpty::from_vec(operators)
                    .expect("filter reduction must retain at least one operator"),
                state,
            };
        }
    }
    let reads = {
        let consumer = &graph.skeleton.blocks[candidate.block].side_effects[consumer_index];
        crate::egir::semantic_graph::read_resources(graph, consumer)
    };
    let retained: Vec<_> = old_resources
        .iter()
        .copied()
        .filter(|resource| Some(resource.resource) != filter.scratch)
        .collect();
    if let SideEffectKind::Soac(_, Soac::Screma(op)) =
        &mut graph.skeleton.blocks[candidate.block].side_effects[consumer_index].kind
    {
        let screma::SemanticState::Segmented { resources, .. } = op.semantic_state_mut() else {
            unreachable!();
        };
        *resources = SegResourceAccess::merge(&retained, &reads);
    }
    if let Some((_, project)) = count_project {
        replace_lengths(graph, &candidate.lengths, project);
    }
    graph.skeleton.blocks[candidate.block].side_effects.remove(candidate.filter);
    finish_entry_metadata(
        inner,
        candidate.site,
        &candidate.lengths,
        count_project.map(|(_, value)| value),
        filter_effect.result,
        Some(count_project.map(|(result, _)| result).unwrap_or(old_result)),
        filter.scratch,
    );
}

fn apply_count_only(
    inner: &mut SemanticProgram,
    candidate: &Candidate,
    filter_effect: SideEffect,
    filter: FilterParts,
    mut count: screma::Operator,
    count_ty: Type<TypeName>,
) {
    let graph = graph_mut(inner, candidate.site);
    let neutral = integer_literal(graph, "0", &count_ty);
    count.neutral = neutral;
    let tuple_ty = Type::Constructed(TypeName::Tuple(1), vec![count_ty.clone()]);
    let result = graph.alloc_side_effect_result(tuple_ty);
    let project = graph.intern_pure(
        PureOp::Project { index: 0 },
        smallvec![result],
        count_ty.clone(),
        None,
    );
    replace_lengths(graph, &candidate.lengths, project);
    let effect = &mut graph.skeleton.blocks[candidate.block].side_effects[candidate.filter];
    let SideEffectKind::Soac(id, _) = effect.kind else {
        unreachable!("count-only rewrite requires a filter SOAC");
    };
    effect.kind = SideEffectKind::Soac(
        id,
        Soac::Screma(screma::Op::Reduce {
            lanes: screma::Lanes {
                inputs: vec![SoacInputType {
                    array: filter.input_array_type,
                    element: filter.input_elem_type,
                }],
                maps: Vec::new(),
            },
            operators: screma::NonEmpty {
                first: count,
                rest: Vec::new(),
            },
            state: screma::SemanticState::Segmented {
                space: filter.space,
                placement: screma::Placement::LaneLocal,
                output_slots: Vec::new(),
                resources: Vec::new(),
            },
        }),
    );
    effect.operand_nodes = smallvec![filter.input];
    effect.result = Some(result);
    let reads = {
        let effect = &graph.skeleton.blocks[candidate.block].side_effects[candidate.filter];
        crate::egir::semantic_graph::read_resources(graph, effect)
    };
    if let SideEffectKind::Soac(_, Soac::Screma(op)) =
        &mut graph.skeleton.blocks[candidate.block].side_effects[candidate.filter].kind
    {
        let screma::SemanticState::Segmented { resources, .. } = op.semantic_state_mut() else {
            unreachable!();
        };
        *resources = reads;
    }
    finish_entry_metadata(
        inner,
        candidate.site,
        &candidate.lengths,
        Some(project),
        filter_effect.result,
        Some(result),
        filter.scratch,
    );
}

fn finish_entry_metadata(
    inner: &mut SemanticProgram,
    site: FusionSite,
    old_values: &[NodeId],
    replacement: Option<NodeId>,
    old_writer: Option<NodeId>,
    replacement_writer: Option<NodeId>,
    scratch: Option<SemanticResourceRef>,
) {
    let FusionSite::Entry(index) = site else {
        return;
    };
    let entry = &mut inner.entry_points[index];
    if let Some(replacement) = replacement {
        for route in &mut entry.output_routes {
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
    if let (Some(old_writer), Some(replacement_writer)) = (old_writer, replacement_writer) {
        for route in &mut entry.output_routes {
            for writer in &mut route.writers {
                if *writer == OutputWriter::Value(old_writer) {
                    *writer = OutputWriter::Value(replacement_writer);
                }
            }
        }
    }
    if let Some(scratch) = scratch {
        entry.resource_declarations.retain(|declaration| declaration.resource != scratch);
    }
}

fn replace_lengths(graph: &mut EGraph, lengths: &[NodeId], replacement: NodeId) {
    for &length in lengths {
        graph_ops::replace_all_references(graph, length, replacement);
    }
}

fn extend_result(
    graph: &mut EGraph,
    old_result: NodeId,
    old_fields: &[Type<TypeName>],
    extra: Type<TypeName>,
) -> NodeId {
    let mut fields = old_fields.to_vec();
    fields.push(extra);
    let new_result =
        graph.alloc_side_effect_result(Type::Constructed(TypeName::Tuple(fields.len()), fields));
    let projects: Vec<(NodeId, u32)> = graph
        .nodes
        .iter()
        .filter_map(|(node, definition)| match definition {
            ENode::Pure {
                op: PureOp::Project { index },
                operands,
            } if operands.first() == Some(&old_result) => Some((node, *index)),
            _ => None,
        })
        .collect();
    for (project, index) in projects {
        graph.update_pure_node(project, |_, operands| operands[0] = new_result);
        debug_assert!((index as usize) < old_fields.len());
    }
    let rebuilt_fields: SmallVec<[NodeId; 4]> = old_fields
        .iter()
        .enumerate()
        .map(|(index, ty)| {
            graph.intern_pure(
                PureOp::Project { index: index as u32 },
                smallvec![new_result],
                ty.clone(),
                None,
            )
        })
        .collect();
    let old_ty = graph.types[&old_result].clone();
    let rebuilt = graph.intern_pure(PureOp::Tuple(old_fields.len()), rebuilt_fields, old_ty, None);
    graph_ops::replace_all_references(graph, old_result, rebuilt);
    new_result
}

fn integer_literal(graph: &mut EGraph, value: &str, ty: &Type<TypeName>) -> NodeId {
    let op = match ty {
        Type::Constructed(TypeName::UInt(_), _) => PureOp::Uint(value.to_string()),
        _ => PureOp::Int(value.to_string()),
    };
    graph.intern_pure(op, smallvec![], ty.clone(), None)
}

#[allow(clippy::too_many_arguments)]
fn masked_step(
    inner: &mut SemanticProgram,
    scope: &str,
    span: crate::ast::Span,
    index: usize,
    filter: &FilterParts,
    operator: &screma::Operator,
    outer_types: &LookupMap<NodeId, Type<TypeName>>,
) -> (SegBody, SemanticFunc) {
    let consumer = inner.region(operator.step.region).expect("operator step region").clone();
    let accumulator_ty = consumer.params[0].0.clone();
    let captures: Vec<NodeId> = filter
        .map
        .iter()
        .flat_map(|body| body.captures.iter())
        .chain(&filter.pred.captures)
        .chain(&operator.step.captures)
        .copied()
        .collect();
    let capture_types = capture_types(outer_types, captures.iter());
    let mut params = vec![
        (accumulator_ty.clone(), "accumulator".to_string()),
        (filter.input_elem_type.clone(), "element".to_string()),
    ];
    params.extend(
        capture_types.iter().enumerate().map(|(capture, ty)| (ty.clone(), format!("capture_{capture}"))),
    );
    let mut graph = EGraph::new();
    let args: Vec<_> =
        params.iter().enumerate().map(|(param, (ty, _))| graph.add_func_param(param, ty.clone())).collect();
    let mut cursor = 2;
    let value = if let Some(map) = &filter.map {
        let region = inner.region(map.region).expect("map region");
        let mut call_args = smallvec![args[1]];
        call_args.extend(args[cursor..cursor + map.captures.len()].iter().copied());
        cursor += map.captures.len();
        graph.intern_pure(
            PureOp::Call(region.name.clone()),
            call_args,
            region.return_ty.clone(),
            None,
        )
    } else {
        args[1]
    };
    let pred_region = inner.region(filter.pred.region).expect("filter predicate region");
    let mut pred_args = smallvec![value];
    pred_args.extend(args[cursor..cursor + filter.pred.captures.len()].iter().copied());
    cursor += filter.pred.captures.len();
    let pred = graph.intern_pure(
        PureOp::Call(pred_region.name.clone()),
        pred_args,
        pred_region.return_ty.clone(),
        None,
    );
    let mut reduce_args = smallvec![args[0], value];
    reduce_args.extend(args[cursor..].iter().copied());
    let reduced = graph.intern_pure(
        PureOp::Call(consumer.name.clone()),
        reduce_args,
        consumer.return_ty.clone(),
        None,
    );
    let control_headers = conditional_return(&mut graph, pred, args[0], reduced, accumulator_ty);
    let name = fresh_region_name(inner, &format!("{scope}_filter_reduce_{index}"));
    let region = inner.region_interner.intern(&name);
    let function = SemanticFunc::new(
        name,
        span,
        None,
        params,
        consumer.return_ty,
        graph,
        control_headers,
    );
    (SegBody { region, captures }, function)
}

fn count_operator(
    inner: &mut SemanticProgram,
    scope: &str,
    span: crate::ast::Span,
    filter: &FilterParts,
    count_ty: Type<TypeName>,
    outer_types: &LookupMap<NodeId, Type<TypeName>>,
) -> (screma::Operator, SemanticFunc, SemanticFunc) {
    let captures: Vec<NodeId> = filter
        .map
        .iter()
        .flat_map(|body| body.captures.iter())
        .chain(&filter.pred.captures)
        .copied()
        .collect();
    let capture_types = capture_types(outer_types, captures.iter());
    let mut params = vec![
        (count_ty.clone(), "count".to_string()),
        (filter.input_elem_type.clone(), "element".to_string()),
    ];
    params.extend(
        capture_types.iter().enumerate().map(|(capture, ty)| (ty.clone(), format!("capture_{capture}"))),
    );
    let mut graph = EGraph::new();
    let args: Vec<_> =
        params.iter().enumerate().map(|(param, (ty, _))| graph.add_func_param(param, ty.clone())).collect();
    let mut cursor = 2;
    let value = if let Some(map) = &filter.map {
        let region = inner.region(map.region).expect("map region");
        let mut call_args = smallvec![args[1]];
        call_args.extend(args[cursor..cursor + map.captures.len()].iter().copied());
        cursor += map.captures.len();
        graph.intern_pure(
            PureOp::Call(region.name.clone()),
            call_args,
            region.return_ty.clone(),
            None,
        )
    } else {
        args[1]
    };
    let pred_region = inner.region(filter.pred.region).expect("filter predicate region");
    let mut pred_args = smallvec![value];
    pred_args.extend(args[cursor..].iter().copied());
    let pred = graph.intern_pure(
        PureOp::Call(pred_region.name.clone()),
        pred_args,
        pred_region.return_ty.clone(),
        None,
    );
    let one = integer_literal(&mut graph, "1", &count_ty);
    let incremented = graph.intern_pure(
        PureOp::BinOp("+".into()),
        smallvec![args[0], one],
        count_ty.clone(),
        None,
    );
    let control_headers = conditional_return(&mut graph, pred, args[0], incremented, count_ty.clone());
    let step_name = fresh_region_name(inner, &format!("{scope}_filter_count_step"));
    let step_region = inner.region_interner.intern(&step_name);
    let step = SemanticFunc::new(
        step_name,
        span,
        None,
        params,
        count_ty.clone(),
        graph,
        control_headers,
    );

    let mut combine_graph = EGraph::new();
    let left = combine_graph.add_func_param(0, count_ty.clone());
    let right = combine_graph.add_func_param(1, count_ty.clone());
    let sum = combine_graph.intern_pure(
        PureOp::BinOp("+".into()),
        smallvec![left, right],
        count_ty.clone(),
        None,
    );
    combine_graph.skeleton.blocks[combine_graph.skeleton.entry].term =
        SkeletonTerminator::Return(Some(sum));
    let combine_name = fresh_region_name(inner, &format!("{scope}_filter_count_combine"));
    let combine_region = inner.region_interner.intern(&combine_name);
    let combine = SemanticFunc::new(
        combine_name,
        span,
        None,
        vec![
            (count_ty.clone(), "left".to_string()),
            (count_ty.clone(), "right".to_string()),
        ],
        count_ty.clone(),
        combine_graph,
        LookupMap::new(),
    );
    (
        screma::Operator {
            step: SegBody {
                region: step_region,
                captures,
            },
            combine: SegBody {
                region: combine_region,
                captures: vec![],
            },
            input_indices: vec![screma::InputId(0)],
            neutral: NodeId::default(),
            shape: vec![],
            commutative: true,
            destination: SoacDestination::Fresh,
            result_type: count_ty,
        },
        step,
        combine,
    )
}

fn conditional_return(
    graph: &mut EGraph,
    predicate: NodeId,
    fallback: NodeId,
    selected: NodeId,
    ty: Type<TypeName>,
) -> LookupMap<BlockId, ControlHeader> {
    let entry = graph.skeleton.entry;
    let then_block = graph.skeleton.create_block();
    let merge = graph.skeleton.create_block();
    let result = graph.add_block_param(merge, 0, ty);
    graph.skeleton.blocks[merge].params.push(result);
    graph.skeleton.blocks[entry].term = SkeletonTerminator::CondBranch {
        cond: predicate,
        then_target: then_block,
        then_args: vec![],
        else_target: merge,
        else_args: vec![fallback],
    };
    graph.skeleton.blocks[then_block].term = SkeletonTerminator::Branch {
        target: merge,
        args: vec![selected],
    };
    graph.skeleton.blocks[merge].term = SkeletonTerminator::Return(Some(result));
    let mut headers = LookupMap::new();
    headers.insert(entry, ControlHeader::Selection { merge });
    headers
}
