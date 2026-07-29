//! Fuse a non-escaping Filter into scalar Screma reductions.
//!
//! The Filter's map and predicate become a canonical Screma pre-lambda.  For
//! each reduction component the pre-lambda yields the consumer value when the
//! predicate holds and that reduction's neutral value otherwise.  The original
//! associative reduction operators remain unchanged.  `length` is represented
//! by one additional sum reduction over `1`/`0` values.

use std::collections::HashSet;

use polytype::Type;
use smallvec::{smallvec, SmallVec};

use super::{capture_types, graph_and_span, support};
use crate::ast::TypeName;
use crate::builtins::catalog;
use crate::egir::graph_ops;
use crate::egir::ir::{splice_effect_tokens, BodySite, RealizedOutputRoute};
use crate::egir::program::{
    CoreProgramData, OutputWriter, RegionInterner, SemanticFunc, SemanticResourceRef,
};
use crate::egir::reify::Segmented;
use crate::egir::semantic_graph::SemanticGraph;
use crate::egir::soac::{filter, lambda as lambda_ops, screma};
use crate::egir::types::{
    EGraph, ENode, NodeId, PureOp, SegResourceAccess, SegSpace, SideEffect, SideEffectKind,
    SkeletonTerminator, Soac, SoacEffect,
};
use crate::flow::{BlockId, ControlHeader};
use crate::LookupMap;

#[derive(Clone)]
pub(crate) struct Candidate {
    site: BodySite,
    block: BlockId,
    filter: usize,
    consumer: Option<usize>,
    lengths: Vec<NodeId>,
}

#[derive(Clone)]
struct FilterParts {
    space: SegSpace,
    body: filter::Body,
    input_nodes: Vec<NodeId>,
    scratch: Option<SemanticResourceRef>,
}

pub(super) fn analyze(inner: &Segmented, oracle: &SemanticGraph) -> Option<Candidate> {
    for (index, entry) in inner.entry_points.iter().enumerate() {
        let routes = entry.routes().cloned().collect::<Vec<_>>();
        if let Some(candidate) = find_in_graph(&entry.graph, BodySite::Entry(index), oracle, Some(&routes))
        {
            return Some(candidate);
        }
    }
    for function in &inner.functions {
        if let Some(candidate) =
            find_in_graph(&function.graph, BodySite::Function(function.region), oracle, None)
        {
            return Some(candidate);
        }
    }
    None
}

fn find_in_graph(
    graph: &EGraph,
    site: BodySite,
    oracle: &SemanticGraph,
    routes: Option<&[RealizedOutputRoute]>,
) -> Option<Candidate> {
    let live = graph_ops::reachable_execution_values(graph).into_iter().collect::<HashSet<_>>();
    for (block_id, block) in &graph.skeleton.blocks {
        for (filter_index, effect) in block.side_effects.iter().enumerate() {
            let SideEffectKind::Soac(SoacEffect(filter_id, Soac::Filter(_))) = &effect.kind else {
                continue;
            };
            let Some(result) = effect.result else {
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

fn is_length_of(graph: &EGraph, node: NodeId, filter_result: NodeId) -> bool {
    let ENode::Pure { op, operands } = &graph.nodes[node].kind else {
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
        && effect.operand_nodes.len() >= input_count
        && effect.operand_nodes[..input_count].iter().all(|&input| input == filter_result)
}

fn filter_result_escapes(
    graph: &EGraph,
    filter_block: BlockId,
    filter_index: usize,
    consumer: Option<usize>,
    result: NodeId,
    lengths: &[NodeId],
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
            if effect
                .referenced_nodes()
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

fn filter_parts(effect: &SideEffect) -> FilterParts {
    let SideEffectKind::Soac(SoacEffect(_, Soac::Filter(op))) = &effect.kind else {
        unreachable!();
    };
    let input_count = op.body.inputs.len();
    FilterParts {
        space: op.state.space.clone(),
        body: op.body.clone(),
        input_nodes: effect.operand_nodes[..input_count].to_vec(),
        scratch: match &op.state.storage {
            filter::Output::Local { .. } => None,
            filter::Output::Runtime { scratch, .. } => Some(*scratch),
        },
    }
}

pub(super) fn apply(inner: Segmented, candidate: Candidate) -> Segmented {
    let (filter_effect, consumer_effect, outer_types, span, scope) = {
        let (graph, span, scope) = graph_and_span(&inner, candidate.site);
        let block = &graph.skeleton.blocks[candidate.block];
        (
            block.side_effects[candidate.filter].clone(),
            candidate.consumer.map(|consumer| block.side_effects[consumer].clone()),
            graph.nodes.iter().map(|(node, data)| (node, data.ty.clone())).collect::<LookupMap<_, _>>(),
            span,
            scope,
        )
    };
    let filter = filter_parts(&filter_effect);
    let mut interner = inner.data.region_interner.clone();
    let count_ty = candidate.lengths.first().map(|length| outer_types[length].clone());
    let consumer_form = consumer_effect.as_ref().map(|effect| {
        let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &effect.kind else {
            unreachable!();
        };
        &op.form
    });
    let (pre, pre_function) = build_masked_pre(
        &inner,
        &mut interner,
        &scope,
        span,
        &filter,
        consumer_form,
        count_ty.as_ref(),
        &outer_types,
    );
    let (count_reduction, count_function) = count_ty
        .as_ref()
        .map(|ty| build_count_reduction(&mut interner, &scope, span, ty.clone()))
        .map_or((None, None), |(reduction, function)| {
            (Some(reduction), Some(function))
        });

    let (rebuilt, _) = if let Some(consumer_index) = candidate.consumer {
        rewrite_with_consumer(
            inner,
            &candidate,
            filter_effect,
            consumer_effect.expect("Filter reduction consumer disappeared"),
            filter,
            consumer_index,
            pre,
            count_reduction,
            count_ty,
        )
    } else {
        rewrite_count_only(
            inner,
            &candidate,
            filter_effect,
            filter,
            pre,
            count_reduction.expect("length-only Filter has no count reduction"),
            count_ty.expect("length-only Filter has no count type"),
        )
    };
    let synthesized = std::iter::once(pre_function).chain(count_function).collect::<Vec<_>>();
    rebuilt.extend_functions(synthesized).map_data(|data| CoreProgramData {
        region_interner: interner,
        ..data
    })
}

fn build_count_reduction(
    interner: &mut RegionInterner,
    scope: &str,
    span: crate::ast::Span,
    count_ty: Type<TypeName>,
) -> (screma::Reduce, SemanticFunc) {
    let mut graph = EGraph::new();
    let left = graph.add_func_param(0, count_ty.clone());
    let right = graph.add_func_param(1, count_ty.clone());
    let sum = graph.intern_pure(
        PureOp::BinOp("+".into()),
        smallvec![left, right],
        count_ty.clone(),
        None,
    );
    let entry = graph.skeleton.entry;
    let parameter_types = vec![count_ty.clone(), count_ty.clone()];
    let (operator, function) = lambda_ops::finish_region_lambda(
        interner,
        scope,
        "filter_count_combine",
        span,
        graph,
        entry,
        vec![
            (count_ty.clone(), "left".to_string()),
            (count_ty.clone(), "right".to_string()),
        ],
        vec![],
        parameter_types,
        vec![count_ty],
        vec![sum],
        false,
    );
    (
        screma::Reduce {
            operator,
            neutral: Vec::new(),
            commutative: true,
        },
        function.expect("filter count operator cannot be identity"),
    )
}
#[allow(clippy::too_many_arguments)]
fn build_masked_pre(
    inner: &Segmented,
    interner: &mut RegionInterner,
    scope: &str,
    span: crate::ast::Span,
    filter: &FilterParts,
    consumer: Option<&screma::ScremaForm>,
    count_ty: Option<&Type<TypeName>>,
    outer_types: &LookupMap<NodeId, Type<TypeName>>,
) -> (screma::Lambda, SemanticFunc) {
    let mut captures = filter.body.map.captures().to_vec();
    captures.extend_from_slice(filter.body.predicate.captures());
    if let Some(consumer) = consumer {
        captures.extend_from_slice(consumer.pre.captures());
        captures.extend(consumer.reductions.iter().flat_map(|reduction| reduction.neutral.iter().copied()));
    }
    let capture_types = capture_types(outer_types, captures.iter());
    let input_types =
        filter.body.inputs.iter().map(crate::egir::types::SoacInputType::element).collect::<Vec<_>>();
    let mut result_types = consumer.map(|consumer| consumer.pre.result_types.clone()).unwrap_or_default();
    if let Some(count_ty) = count_ty {
        result_types.push(count_ty.clone());
    }
    let mut params = input_types
        .iter()
        .enumerate()
        .map(|(index, ty)| (ty.clone(), format!("input_{index}")))
        .collect::<Vec<_>>();
    params.extend(
        capture_types.iter().enumerate().map(|(index, ty)| (ty.clone(), format!("capture_{index}"))),
    );
    let mut graph = EGraph::new();
    let args = params
        .iter()
        .enumerate()
        .map(|(index, (ty, _))| graph.add_func_param(index, ty.clone()))
        .collect::<Vec<_>>();
    let mut cursor = input_types.len();
    let mapped_capture_count = filter.body.map.capture_count();
    let mapped = support::invoke_lambda(
        &mut graph,
        inner,
        &filter.body.map,
        &args[..input_types.len()],
        &args[cursor..cursor + mapped_capture_count],
    );
    cursor += mapped_capture_count;
    let predicate_capture_count = filter.body.predicate.capture_count();
    let predicate = support::invoke_lambda(
        &mut graph,
        inner,
        &filter.body.predicate,
        &mapped,
        &args[cursor..cursor + predicate_capture_count],
    );
    cursor += predicate_capture_count;
    debug_assert_eq!(predicate.len(), 1);

    let mut selected = Vec::new();
    let mut fallback = Vec::new();
    if let Some(consumer) = consumer {
        let consumer_capture_count = consumer.pre.capture_count();
        let consumer_args = vec![mapped[0]; consumer.pre.parameter_types.len()];
        selected.extend(support::invoke_lambda(
            &mut graph,
            inner,
            &consumer.pre,
            &consumer_args,
            &args[cursor..cursor + consumer_capture_count],
        ));
        cursor += consumer_capture_count;
        let neutral_count = consumer.reduction_input_count();
        fallback.extend_from_slice(&args[cursor..cursor + neutral_count]);
        cursor += neutral_count;
    }
    if let Some(count_ty) = count_ty {
        selected.push(integer_literal(&mut graph, "1", count_ty));
        fallback.push(integer_literal(&mut graph, "0", count_ty));
    }
    debug_assert_eq!(cursor, args.len());
    let (return_block, results) =
        conditional_results(&mut graph, predicate[0], fallback, selected, &result_types);
    let (pre, function) = lambda_ops::finish_region_lambda(
        interner,
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
    (pre, function.expect("filter pre-lambda cannot be identity"))
}

fn conditional_results(
    graph: &mut EGraph,
    predicate: NodeId,
    fallback: Vec<NodeId>,
    selected: Vec<NodeId>,
    types: &[Type<TypeName>],
) -> (BlockId, Vec<NodeId>) {
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
        args: selected,
    };
    graph.skeleton.blocks[else_block].term = SkeletonTerminator::Branch {
        target: merge,
        args: fallback,
    };
    (merge, results)
}

struct EntryMetadataPatch {
    replacement: Option<NodeId>,
    old_writer: Option<NodeId>,
    replacement_writer: Option<NodeId>,
    scratch: Option<SemanticResourceRef>,
}

#[allow(clippy::too_many_arguments)]
fn rewrite_with_consumer(
    inner: Segmented,
    candidate: &Candidate,
    filter_effect: SideEffect,
    consumer_effect: SideEffect,
    filter: FilterParts,
    consumer_index: usize,
    pre: screma::Lambda,
    count: Option<screma::Reduce>,
    count_ty: Option<Type<TypeName>>,
) -> (Segmented, Vec<SemanticFunc>) {
    let synthesized = Vec::new();
    let rebuilt = inner.rewrite_body(candidate.site, |body| {
        let rewrite = |graph: &mut EGraph| {
            let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(old_op))) = &consumer_effect.kind else {
                unreachable!();
            };
            let old_result = consumer_effect.result.expect("Filter reduction has no result");
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
                let new_result = extend_result(graph, old_result, &old_result_types, count_ty.clone());
                let project = graph.intern_pure(
                    PureOp::Project { index: field },
                    smallvec![new_result],
                    count_ty.clone(),
                    None,
                );
                Some((count, new_result, project))
            } else {
                None
            };

            let mut op = old_op.clone();
            op.inputs = filter.body.inputs.clone();
            op.form.pre = pre.clone();
            if let Some((count, _, _)) = &count_project {
                op.form.reductions.push(count.clone());
                op.result_state.push(screma::ResultState {
                    destination: crate::egir::types::SoacDestination::fresh(),
                });
            }
            let screma::SemanticState::Segmented { space, resources, .. } = op.semantic_state_mut() else {
                unreachable!();
            };
            *space = filter.space.clone();
            *resources = resources
                .iter()
                .copied()
                .filter(|resource| Some(resource.resource) != filter.scratch)
                .collect();
            debug_assert!(
                op.validate().is_ok(),
                "invalid filtered Screma: {:?}",
                op.validate()
            );

            let fused_effects = splice_effect_tokens(filter_effect.effects, consumer_effect.effects);
            let consumer_id = consumer_effect.kind.soac_id().copied().expect("consumer SOAC id");
            {
                let consumer = &mut graph.skeleton.blocks[candidate.block].side_effects[consumer_index];
                consumer.kind = SideEffectKind::Soac(SoacEffect(consumer_id, Soac::Screma(op)));
                consumer.operand_nodes = SmallVec::from_vec(filter.input_nodes.clone());
                consumer.effects = fused_effects;
                if let Some((_, new_result, _)) = &count_project {
                    consumer.result = Some(*new_result);
                }
            }
            let consumer_snapshot =
                graph.skeleton.blocks[candidate.block].side_effects[consumer_index].clone();
            let reads = crate::egir::semantic_graph::read_resources(graph, &consumer_snapshot);
            if let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) =
                &mut graph.skeleton.blocks[candidate.block].side_effects[consumer_index].kind
            {
                let screma::SemanticState::Segmented { resources, .. } = op.semantic_state_mut() else {
                    unreachable!();
                };
                *resources = SegResourceAccess::merge(resources, &reads);
            }
            if let Some((_, _, project)) = &count_project {
                replace_lengths(graph, &candidate.lengths, *project);
            }
            graph.skeleton.blocks[candidate.block].side_effects.remove(candidate.filter);
            EntryMetadataPatch {
                replacement: count_project.as_ref().map(|(_, _, value)| *value),
                old_writer: filter_effect.result,
                replacement_writer: Some(
                    count_project.as_ref().map(|(_, result, _)| *result).unwrap_or(old_result),
                ),
                scratch: filter.scratch,
            }
        };
        support::rewrite_body_graph_with_entry(body, rewrite, |entry, metadata| {
            finish_entry_metadata(entry, &candidate.lengths, metadata);
        })
    });
    (rebuilt, synthesized)
}

fn rewrite_count_only(
    inner: Segmented,
    candidate: &Candidate,
    filter_effect: SideEffect,
    filter: FilterParts,
    pre: screma::Lambda,
    mut count: screma::Reduce,
    count_ty: Type<TypeName>,
) -> (Segmented, Vec<SemanticFunc>) {
    let rebuilt = inner.rewrite_body(candidate.site, |body| {
        let rewrite = |graph: &mut EGraph| {
            let neutral = integer_literal(graph, "0", &count_ty);
            count.neutral = vec![neutral];
            let tuple_ty = Type::Constructed(TypeName::Tuple(1), vec![count_ty.clone()]);
            let result = graph.alloc_side_effect_result(tuple_ty);
            let project = graph.intern_pure(
                PureOp::Project { index: 0 },
                smallvec![result],
                count_ty.clone(),
                None,
            );
            replace_lengths(graph, &candidate.lengths, project);
            {
                let effect = &mut graph.skeleton.blocks[candidate.block].side_effects[candidate.filter];
                let SideEffectKind::Soac(SoacEffect(id, _)) = effect.kind else {
                    unreachable!();
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
                            destination: crate::egir::types::SoacDestination::fresh(),
                        }],
                        state: screma::SemanticState::Segmented {
                            space: filter.space.clone(),
                            placement: screma::Placement::LaneLocal,
                            output_slots: vec![],
                            resources: vec![],
                        },
                    }),
                ));
                effect.operand_nodes = SmallVec::from_vec(filter.input_nodes.clone());
                effect.result = Some(result);
            }
            let effect_snapshot =
                graph.skeleton.blocks[candidate.block].side_effects[candidate.filter].clone();
            let reads = crate::egir::semantic_graph::read_resources(graph, &effect_snapshot);
            if let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) =
                &mut graph.skeleton.blocks[candidate.block].side_effects[candidate.filter].kind
            {
                let screma::SemanticState::Segmented { resources, .. } = op.semantic_state_mut() else {
                    unreachable!();
                };
                *resources = reads;
            }
            EntryMetadataPatch {
                replacement: Some(project),
                old_writer: filter_effect.result,
                replacement_writer: Some(result),
                scratch: filter.scratch,
            }
        };
        support::rewrite_body_graph_with_entry(body, rewrite, |entry, metadata| {
            finish_entry_metadata(entry, &candidate.lengths, metadata);
        })
    });
    (rebuilt, Vec::new())
}

fn finish_entry_metadata(
    entry: &mut crate::egir::program::SemanticEntry,
    old_values: &[NodeId],
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
    if let Some(scratch) = patch.scratch {
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
    let projects = graph
        .nodes
        .iter()
        .filter_map(|(node, definition)| match &definition.kind {
            ENode::Pure {
                op: PureOp::Project { index },
                operands,
            } if operands.first() == Some(&old_result) => Some((node, *index)),
            _ => None,
        })
        .collect::<Vec<_>>();
    for (project, index) in projects {
        graph.update_pure_node(project, |_, operands| operands[0] = new_result);
        debug_assert!((index as usize) < old_fields.len());
    }
    let rebuilt_fields = old_fields
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
        .collect::<SmallVec<[NodeId; 4]>>();
    let old_ty = graph.nodes[old_result].ty.clone();
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
