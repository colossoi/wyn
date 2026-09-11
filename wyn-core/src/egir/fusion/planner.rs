//! Deterministic fusion policy over the owned snapshot and composed recipes.
#[cfg(test)]
#[path = "planner_tests.rs"]
mod tests;

use super::{
    algebra, recipe,
    snapshot::{Kind, Operation, Slice, Snapshot, Use},
    FusionResult,
};
use crate::egir::semantic_opt::{SemanticOptimizationRelation, SemanticOptimizationTrace};
use crate::egir::types::{ResourceAccess, SegExtent, SegResourceAccess};
use crate::SortedSet;
use wyn_fusion::{GroupId, OrderingReason, PortId, Proposal};

pub(super) struct Planned {
    pub plan: wyn_fusion::Plan<super::snapshot::ScopeKey, crate::BindingRef, Operation>,
    pub recipes: recipe::Recipes,
    pub trace: SemanticOptimizationTrace,
}

struct Composition {
    operation: Operation,
    results: Vec<Vec<PortId>>,
    absorbed: Vec<PortId>,
}

pub(super) fn plan(snapshot: Snapshot, limit: Option<usize>) -> FusionResult<Planned> {
    plan_with_priority(snapshot, limit, &[0, 1, 2, 3, 4, 5])
}

pub(super) fn plan_with_priority(
    mut snapshot: Snapshot,
    limit: Option<usize>,
    priority: &[usize],
) -> FusionResult<Planned> {
    let mut recipes = std::mem::take(&mut snapshot.recipes);
    let mut trace = SemanticOptimizationTrace::default();
    loop {
        if limit.is_some_and(|limit| snapshot.graph.actions().len() >= limit) {
            break;
        }
        let groups = snapshot.graph.order()?;
        let mut accepted = false;
        'families: for &family in priority {
            for (position, left) in groups.iter().copied().enumerate() {
                let candidates = if family == 0 {
                    vec![None]
                } else if family == 4 {
                    groups[position + 1..].iter().copied().map(Some).chain(std::iter::once(None)).collect()
                } else {
                    groups[position + 1..].iter().copied().map(Some).collect::<Vec<_>>()
                };
                for right in candidates {
                    if right.is_some_and(|right| {
                        snapshot.graph.group(left).unwrap().scope()
                            != snapshot.graph.group(right).unwrap().scope()
                    }) {
                        continue;
                    }
                    let sources = std::iter::once(left).chain(right).collect::<Vec<_>>();
                    let mut staged_recipes = recipes.clone();
                    let composition = match (family, right) {
                        (0, None) => indexed(&snapshot, left),
                        (1, Some(right)) => vertical(&snapshot, &mut staged_recipes, left, right),
                        (2 | 3, Some(right)) => {
                            envelope(&snapshot, &mut staged_recipes, left, right, family == 2)
                        }
                        (4, right) => filter(&snapshot, &mut staged_recipes, left, right),
                        (5, Some(right)) => horizontal(&snapshot, &mut staged_recipes, left, right),
                        _ => None,
                    };
                    let Some(composition) = composition else {
                        continue;
                    };
                    let boundary = snapshot.graph.boundary(&sources, &composition.absorbed)?;
                    let proposal = Proposal {
                        sources: sources.clone(),
                        absorbed_values: composition.absorbed,
                        results: composition.results,
                        accounted_constraints: boundary.constraints,
                        payload: composition.operation,
                    };
                    let after = if matches!(proposal.payload.kind, Kind::Indexed { .. }) {
                        vec![]
                    } else {
                        proposal.payload.semantic_id.into_iter().collect()
                    };
                    let mut before = sources
                        .iter()
                        .filter_map(|id| operation(&snapshot, *id).semantic_id)
                        .collect::<Vec<_>>();
                    let result = if right.is_some() {
                        snapshot.graph.contract(proposal)
                    } else {
                        snapshot.graph.rewrite(proposal)
                    };
                    let Ok(_) = result else {
                        continue;
                    };
                    before.sort_unstable();
                    before.dedup();
                    trace.relations.push(SemanticOptimizationRelation { before, after });
                    recipes = staged_recipes;
                    accepted = true;
                    break 'families;
                }
            }
        }
        if !accepted {
            break;
        }
    }
    Ok(Planned {
        plan: snapshot.graph.finalize()?,
        recipes,
        trace,
    })
}

fn operation(snapshot: &Snapshot, id: GroupId) -> Operation {
    let group = snapshot.graph.group(id).unwrap();
    let mut operation = group.payload().clone();
    operation.outputs = group.outputs().to_vec();
    for input in &mut operation.inputs {
        input.node = snapshot.graph.canonical(input.node).unwrap();
    }
    operation
}

fn route(snapshot: &Snapshot, mut port: PortId, producer: &Operation) -> Option<(usize, Vec<Slice>)> {
    let mut slices = Vec::new();
    loop {
        port = snapshot.graph.canonical(port).ok()?;
        if let Some(slot) = producer.outputs.iter().position(|output| *output == port) {
            slices.reverse();
            return Some((slot, slices));
        }
        match &snapshot.values.get(&port)?.usage {
            Use::Slice { base, slice } => {
                slices.push(slice.clone());
                port = *base;
            }
            _ => return None,
        }
    }
}

fn routed(
    snapshot: &Snapshot,
    producer: &Operation,
    consumer: &Operation,
) -> Option<(Vec<algebra::InputRoute>, Vec<Slice>)> {
    let Kind::Screma(form) = &producer.kind else {
        return None;
    };
    let mut routes = Vec::new();
    let mut transform = None;
    for (slot, input) in consumer.inputs.iter().enumerate() {
        if let Some((field, slices)) = route(snapshot, input.node, producer) {
            let post = field.checked_sub(form.layout().reduction_result_count())?;
            if post >= form.post.result_types.len() || transform.as_ref().is_some_and(|old| *old != slices)
            {
                return None;
            }
            transform = Some(slices);
            routes.push(algebra::InputRoute {
                consumer_input: slot,
                producer_post_output: post,
            });
        } else if !snapshot
            .graph
            .producers(input.node)
            .ok()?
            .is_disjoint(&producer_group(snapshot, producer))
        {
            return None;
        }
    }
    (!routes.is_empty()).then_some((routes, transform.unwrap_or_default()))
}

fn producer_group(snapshot: &Snapshot, operation: &Operation) -> SortedSet<GroupId> {
    operation
        .outputs
        .iter()
        .flat_map(|port| snapshot.graph.producers(*port).unwrap_or_default())
        .collect()
}

fn resource_accounting(
    snapshot: &Snapshot,
    left: GroupId,
    right: GroupId,
    routed_resources: &[crate::BindingRef],
    allow_effect: bool,
) -> bool {
    let Ok(boundary) = snapshot.graph.boundary(&[left, right], &[]) else {
        return false;
    };
    let producer = operation(snapshot, left);
    let consumer = operation(snapshot, right);
    let direct = matches!((producer.effect_tokens, consumer.effect_tokens), (Some((_, output)), Some((input, _))) if output == input);
    boundary.constraints.iter().all(
        |id| match snapshot.graph.constraint(*id).map(|edge| edge.reason) {
            Some(OrderingReason::Resource(id)) => routed_resources.contains(&id),
            Some(OrderingReason::Effect) => allow_effect && direct,
            _ => false,
        },
    )
}

fn merge_state(producer: &Operation, consumer: &Operation, result: &mut Operation) {
    result.owned_resources =
        producer.owned_resources.iter().chain(&consumer.owned_resources).copied().collect();
    result.resources = SegResourceAccess::merge(&producer.resources, &consumer.resources);
    result.output_slots = producer.output_slots.iter().chain(&consumer.output_slots).copied().collect();
    result.output_slots.sort_unstable();
    result.output_slots.dedup();
    result.effect_tokens =
        crate::egir::ir::splice_effect_tokens(producer.effect_tokens, consumer.effect_tokens);
}

fn normalized(
    producer: &Operation,
    consumer: &Operation,
    normalized: algebra::Normalized,
    mut result: Operation,
) -> Composition {
    result.inputs = normalized.inputs;
    result.kind = Kind::Screma(normalized.form);
    let mut origins = Vec::new();
    result.result_types.clear();
    result.result_state.clear();
    for origin in normalized.outputs {
        let (source, slot) = match origin {
            algebra::OutputOrigin::Producer(slot) => (producer, slot),
            algebra::OutputOrigin::Consumer(slot) => (consumer, slot),
        };
        origins.push(source.outputs[slot]);
        result.result_types.push(source.result_types[slot]);
        let mut state = source.result_state[slot];
        // Input reuse is valid only while the sole storage input survives.
        if source.inputs.len() != 1
            || result.inputs.len() != 1
            || source.inputs[0].node != result.inputs[0].node
            || source.inputs[0].slices != result.inputs[0].slices
        {
            state.ownership = crate::types::SoacOwnership::Fresh;
        }
        result.result_state.push(state);
    }
    merge_state(producer, consumer, &mut result);
    Composition {
        operation: result,
        results: origins.into_iter().map(|port| vec![port]).collect(),
        absorbed: vec![],
    }
}

fn horizontal(
    snapshot: &Snapshot,
    recipes: &mut recipe::Recipes,
    left: GroupId,
    right: GroupId,
) -> Option<Composition> {
    let producer = operation(snapshot, left);
    let consumer = operation(snapshot, right);
    let (Kind::Screma(a), Kind::Screma(b)) = (&producer.kind, &consumer.kind) else {
        return None;
    };
    // Output-resource summaries describe the deferred route writes. Parallel
    // composition retains both result bindings and their original routes;
    // write-only ownership can therefore be internalized without reordering a
    // lambda memory access. Reads and other resource conflicts remain barriers.
    let output_writes = producer
        .resources
        .iter()
        .filter(|access| {
            access.access == ResourceAccess::Write
                && producer.owned_resources.contains(&access.resource)
                && consumer.owned_resources.contains(&access.resource)
                && consumer
                    .resources
                    .iter()
                    .any(|other| other.resource == access.resource && other.access == ResourceAccess::Write)
        })
        .map(|access| access.resource)
        .collect::<Vec<_>>();
    if !producer.inputs.iter().any(|input| {
        consumer.inputs.iter().any(|other| input.node == other.node && input.slices == other.slices)
    }) || !super::space::seg_space_fusable(producer.space.as_ref()?, consumer.space.as_ref()?)
        || !snapshot.graph.boundary(&[left, right], &[]).ok()?.internal.is_empty()
        || !resource_accounting(snapshot, left, right, &output_writes, true)
    {
        return None;
    }
    let normalized = algebra::fuse_horizontal(
        &mut algebra::Context { recipes },
        algebra::Source {
            inputs: &producer.inputs,
            form: a,
        },
        algebra::Source {
            inputs: &consumer.inputs,
            form: b,
        },
    )?;
    Some(normalized_result(&producer, &consumer, normalized, true))
}

fn normalized_result(
    producer: &Operation,
    consumer: &Operation,
    form: algebra::Normalized,
    horizontal: bool,
) -> Composition {
    normalized(
        producer,
        consumer,
        form,
        if horizontal { producer.clone() } else { consumer.clone() },
    )
}

fn vertical(
    snapshot: &Snapshot,
    recipes: &mut recipe::Recipes,
    left: GroupId,
    right: GroupId,
) -> Option<Composition> {
    let mut producer = operation(snapshot, left);
    let consumer = operation(snapshot, right);
    let (Kind::Screma(a), Kind::Screma(b)) = (&producer.kind, &consumer.kind) else {
        return None;
    };
    let (routes, slices) = routed(snapshot, &producer, &consumer)?;
    let resources = routes
        .iter()
        .filter_map(|route| consumer.inputs.get(route.consumer_input).and_then(|input| input.resource))
        .collect::<Vec<_>>();
    if !resource_accounting(snapshot, left, right, &resources, true) {
        return None;
    }
    let producer_groups = producer_group(snapshot, &producer);
    // Captures and neutrals cannot silently become per-element arguments.
    let scalar_inputs = b.captures().map(|capture| capture.value).chain(
        b.scans
            .iter()
            .flat_map(|scan| &scan.neutral)
            .chain(b.reductions.iter().flat_map(|reduce| &reduce.neutral))
            .filter_map(|neutral| match neutral {
                recipe::Neutral::Value(port) => Some(*port),
                _ => None,
            }),
    );
    if scalar_inputs
        .into_iter()
        .any(|port| !snapshot.graph.producers(port).unwrap_or_default().is_disjoint(&producer_groups))
    {
        return None;
    }
    let routed_outputs = routes
        .iter()
        .map(|route| producer.outputs[a.layout().reduction_result_count() + route.producer_post_output])
        .collect::<Vec<_>>();
    let consumers = snapshot.graph.consumers(&routed_outputs).ok()?;
    if consumers.len() > 1
        && !consumers
            .iter()
            .all(|id| matches!(&operation(snapshot, *id).kind, Kind::Screma(form) if form.is_map()))
    {
        return None;
    }
    let boundary = snapshot.graph.boundary(&[left, right], &[]).ok()?;
    let retained = producer
        .outputs
        .iter()
        .enumerate()
        .filter_map(|(slot, port)| boundary.outputs.contains(port).then_some(slot))
        .collect::<Vec<_>>();
    if !slices.is_empty() && (!a.is_map() || !retained.is_empty()) {
        return None;
    }
    if !algebra::can_fuse_vertical(a, b, &routes) {
        return None;
    }
    let a = a.clone();
    for input in &mut producer.inputs {
        input.slices.extend(slices.iter().cloned());
    }
    let normalized = algebra::fuse_vertical(
        &mut algebra::Context { recipes },
        algebra::Source {
            inputs: &producer.inputs,
            form: &a,
        },
        algebra::Source {
            inputs: &consumer.inputs,
            form: b,
        },
        &routes,
        &retained,
    )?;
    let mut result = normalized_result(&producer, &consumer, normalized, false);
    if slices.is_empty() {
        result.operation.space = producer.space;
    }
    Some(result)
}

fn envelope(
    snapshot: &Snapshot,
    recipes: &mut recipe::Recipes,
    left: GroupId,
    right: GroupId,
    histogram: bool,
) -> Option<Composition> {
    let producer = operation(snapshot, left);
    let consumer = operation(snapshot, right);
    let Kind::Screma(form) = &producer.kind else {
        return None;
    };
    let lambda = match (&consumer.kind, histogram) {
        (Kind::Hist { bucket }, true) => bucket,
        (Kind::Filter { map, .. }, false) => map,
        _ => return None,
    };
    if !form.is_map()
        || !producer.output_slots.is_empty()
        || producer.resources.iter().any(|resource| resource.access != ResourceAccess::Read)
    {
        return None;
    }
    let (routes, slices) = routed(snapshot, &producer, &consumer)?;
    if !slices.is_empty()
        || routes.iter().map(|route| route.producer_post_output).collect::<SortedSet<_>>().len()
            != form.post.result_types.len()
        || !resource_accounting(snapshot, left, right, &[], false)
        || snapshot
            .graph
            .boundary(&[left, right], &[])
            .ok()?
            .outputs
            .iter()
            .any(|port| producer.outputs.contains(port))
    {
        return None;
    }
    let normalized = algebra::fuse_map_into_lambda(
        &mut algebra::Context { recipes },
        algebra::Source {
            inputs: &producer.inputs,
            form,
        },
        algebra::LambdaSource {
            inputs: &consumer.inputs,
            lambda,
        },
        &routes,
    )?;
    let mut result = consumer.clone();
    result.inputs = normalized.inputs;
    result.kind = match &consumer.kind {
        Kind::Hist { .. } => Kind::Hist {
            bucket: normalized.lambda,
        },
        Kind::Filter { predicate, .. } => Kind::Filter {
            map: normalized.lambda,
            predicate: predicate.clone(),
        },
        _ => return None,
    };
    if !histogram || routes.iter().any(|route| route.consumer_input == 0) {
        result.space = producer.space.clone();
    }
    merge_state(&producer, &consumer, &mut result);
    Some(Composition {
        operation: result,
        results: consumer.outputs.into_iter().map(|port| vec![port]).collect(),
        absorbed: vec![],
    })
}

fn demands(
    snapshot: &Snapshot,
    operation: &Operation,
    indexed: bool,
) -> Vec<(PortId, PortId, usize, Vec<usize>)> {
    let mut uses = snapshot.values.keys().copied().collect::<Vec<_>>();
    uses.sort_unstable();
    uses.into_iter()
        .filter_map(|port| {
            let (base, index) = match snapshot.values[&port].usage {
                Use::Index { base, index } if indexed => (base, index),
                Use::Length { base } if !indexed => (base, base),
                _ => return None,
            };
            let base = snapshot.graph.canonical(base).ok()?;
            let (base, path) = match snapshot.values.get(&base).map(|fact| &fact.usage) {
                Some(Use::Project { base, path }) if indexed => {
                    (snapshot.graph.canonical(*base).ok()?, path.clone())
                }
                _ => (base, vec![]),
            };
            let slot = operation.outputs.iter().position(|output| *output == base)?;
            Some((port, index, slot, path))
        })
        .collect()
}

fn indexed(snapshot: &Snapshot, group: GroupId) -> Option<Composition> {
    let producer = operation(snapshot, group);
    let Kind::Screma(form) = &producer.kind else {
        return None;
    };
    if !form.is_map()
        || !form.post.is_identity()
        || form.pre.result_types.is_empty()
        || producer.resources.iter().any(|resource| {
            resource.access != ResourceAccess::Read
                && !producer.owned_resources.contains(&resource.resource)
        })
    {
        return None;
    }
    let demands = demands(snapshot, &producer, true);
    if demands.is_empty() {
        return None;
    }
    let profitable = match producer.space.as_ref()?.dims() {
        [SegExtent::Fixed(size)] => demands.len() as u64 <= u64::from(*size),
        _ => demands.len() <= 2,
    };
    if !profitable
        || demands
            .iter()
            .any(|(_, index, _, _)| snapshot.graph.producers(*index).unwrap_or_default().contains(&group))
    {
        return None;
    }
    let absorbed = demands.iter().map(|(port, _, _, _)| *port).collect::<Vec<_>>();
    let boundary = snapshot.graph.boundary(&[group], &absorbed).ok()?;
    if boundary.outputs.iter().any(|port| !absorbed.contains(port)) {
        return None;
    }
    let mut result = producer.clone();
    result.kind = Kind::Indexed {
        lambda: form.pre.clone(),
        demands: demands.iter().map(|(_, index, slot, path)| (*index, *slot, path.clone())).collect(),
    };
    result.result_types = absorbed.iter().map(|port| snapshot.values[port].ty).collect();
    result.output_slots.clear();
    result.result_state.clear();
    Some(Composition {
        operation: result,
        results: absorbed.iter().map(|port| vec![*port]).collect(),
        absorbed,
    })
}

fn filter(
    snapshot: &Snapshot,
    recipes: &mut recipe::Recipes,
    left: GroupId,
    right: Option<GroupId>,
) -> Option<Composition> {
    let producer = operation(snapshot, left);
    let Kind::Filter { map, predicate } = &producer.kind else {
        return None;
    };
    let consumer = right.map(|id| operation(snapshot, id));
    let form = match &consumer {
        Some(op) => {
            let Kind::Screma(form) = &op.kind else {
                return None;
            };
            if !form.scans.is_empty()
                || form.reductions.is_empty()
                || !form.post.result_types.is_empty()
                || form.pre.result_types.len() != form.layout().reduction_input_count()
                || op.inputs.is_empty()
                || op.inputs.iter().any(|input| producer.outputs.as_slice() != [input.node])
            {
                return None;
            }
            Some(form)
        }
        None => None,
    };
    if let Some(right) = right {
        let resources =
            consumer.as_ref()?.inputs.iter().filter_map(|input| input.resource).collect::<Vec<_>>();
        if !resource_accounting(snapshot, left, right, &resources, true) {
            return None;
        }
    }
    let lengths =
        demands(snapshot, &producer, false).iter().map(|(port, _, _, _)| *port).collect::<Vec<_>>();
    if consumer.is_none() && lengths.is_empty() {
        return None;
    }
    let count_ty = lengths.first().map(|port| snapshot.values[port].ty);
    if lengths.iter().any(|port| Some(snapshot.values[port].ty) != count_ty) {
        return None;
    }
    let sources = std::iter::once(left).chain(right).collect::<Vec<_>>();
    if snapshot
        .graph
        .boundary(&sources, &lengths)
        .ok()?
        .outputs
        .iter()
        .any(|port| producer.outputs.contains(port))
    {
        return None;
    }
    let mut captures = map.captures().iter().chain(predicate.captures()).copied().collect::<Vec<_>>();
    if let Some(form) = form {
        captures.extend_from_slice(form.pre.captures());
        for neutral in form.reductions.iter().flat_map(|reduce| &reduce.neutral) {
            if let recipe::Neutral::Value(value) = neutral {
                captures.push(recipe::Capture {
                    value: *value,
                    ty: snapshot.values.get(value)?.ty,
                });
            }
        }
    }
    let parameter_types = producer.inputs.iter().map(|input| input.element()).collect::<Vec<_>>();
    let mut builder = recipe::Builder::new(parameter_types.clone(), captures);
    let args = builder.arguments.clone();
    let mut cursor = parameter_types.len();
    let mapped_args =
        args[..cursor].iter().chain(&args[cursor..cursor + map.capture_count()]).copied().collect();
    cursor += map.capture_count();
    let mapped = builder.invoke(recipes, map, mapped_args)?;
    let predicate_args =
        mapped.iter().chain(&args[cursor..cursor + predicate.capture_count()]).copied().collect();
    cursor += predicate.capture_count();
    let condition = *builder.invoke(recipes, predicate, predicate_args)?.first()?;
    let mut results = Vec::new();
    let mut reductions = form.map(|form| form.reductions.clone()).unwrap_or_default();
    if let Some(form) = form {
        let consumer_args = std::iter::repeat_n(*mapped.first()?, form.pre.parameter_types.len())
            .chain(args[cursor..cursor + form.pre.capture_count()].iter().copied())
            .collect();
        cursor += form.pre.capture_count();
        let values = builder.invoke(recipes, &form.pre, consumer_args)?;
        for ((value, ty), neutral) in values
            .into_iter()
            .zip(&form.pre.result_types)
            .zip(form.reductions.iter().flat_map(|reduce| &reduce.neutral))
        {
            let fallback = match neutral {
                recipe::Neutral::Value(_) => {
                    let value = args[cursor];
                    cursor += 1;
                    value
                }
                recipe::Neutral::Zero(ty) => builder.integer(0, *ty),
            };
            results.push(builder.select(condition, value, fallback, *ty));
        }
    }
    if let Some(ty) = count_ty {
        let yes = builder.integer(1, ty);
        let no = builder.integer(0, ty);
        results.push(builder.select(condition, yes, no, ty));
        let mut count = recipe::Builder::new(vec![ty, ty], vec![]);
        let sum = count.add(count.arguments[0], count.arguments[1], ty);
        reductions.push(recipe::Reduce {
            operator: count.finish_named(recipes, vec![sum], "filter_count_combine"),
            neutral: vec![recipe::Neutral::Zero(ty)],
            commutative: true,
        });
    }
    let pre = builder.finish_named(recipes, results, "filter_pre");
    let mut result = consumer.clone().unwrap_or_else(|| producer.clone());
    result.kind = Kind::Screma(recipe::ScremaForm {
        pre,
        scans: vec![],
        reductions,
        post: recipe::Lambda::identity(vec![]),
    });
    result.inputs = producer.inputs.clone();
    result.space = producer.space.clone();
    let mut origins = consumer.as_ref().map(|op| op.outputs.clone()).unwrap_or_default();
    result.result_types = consumer.as_ref().map(|op| op.result_types.clone()).unwrap_or_default();
    result.result_state = consumer.as_ref().map(|op| op.result_state.clone()).unwrap_or_default();
    // Equal length observers route to the same count result, represented by a
    // single symbolic result; aliases are added by the boundary certificate.
    if let Some(port) = lengths.first() {
        origins.push(*port);
        result.result_types.push(count_ty?);
        result.result_state.push(crate::egir::soac::screma::ResultState {
            ownership: crate::types::SoacOwnership::Fresh,
        });
    }
    let mut results = origins.into_iter().map(|port| vec![port]).collect::<Vec<_>>();
    if !lengths.is_empty() {
        *results.last_mut()? = lengths.clone();
    }
    if let Some(consumer) = &consumer {
        merge_state(&producer, consumer, &mut result);
    }
    Some(Composition {
        operation: result,
        results,
        absorbed: lengths,
    })
}
