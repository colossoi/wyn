//! Emit only facts consumed by fusion. The complete program stays in the sidecar.

use super::data::{Array, AssociatedData, ExprKind, OperationKind, SoacBody};
use super::{snapshot, SCHEMA};
use std::collections::BTreeSet;

pub(super) fn program(data: &AssociatedData) -> String {
    let _timing = super::timing::span("emit fusion facts");
    let snapshot = super::timing::time("analyze dependencies", || snapshot::analyze(data));
    let scopes: BTreeSet<_> = snapshot
        .live
        .iter()
        .filter_map(|&op| {
            matches!(
                data.operations[op].kind,
                OperationKind::Screma { .. } | OperationKind::Filter { .. }
            )
            .then_some(data.operations[op].region)
        })
        .collect();
    let included: BTreeSet<_> =
        snapshot.live.iter().copied().filter(|&op| scopes.contains(&data.operations[op].region)).collect();
    let mut output = SCHEMA.to_owned();
    let mut emit = |fact: String| {
        output.push_str(&fact);
        output.push('\n');
    };
    for &id in &included {
        let operation = &data.operations[id];
        let op = id.egglog();
        emit(format!("(Operation {} {op})", operation.region.egglog()));
        if super::fusion::demands(data, id).is_some() {
            emit(format!("(IndexedDemands {op})"));
        }
        if snapshot.discardable.contains(&id) {
            emit(format!("(Safe {op})"));
        }
        if snapshot.movable.contains(&id) {
            emit(format!("(Movable {op})"));
        }
        match &operation.kind {
            OperationKind::Filter { map, body, .. } => {
                emit(format!("(Filter {op})"));
                if snapshot.movable.contains(&id)
                    && super::fusion::masked(&mut data.clone(), id, id).is_some()
                {
                    emit(format!("(Maskable {op} {op})"));
                }
                if snapshot::safe_body(map, &snapshot.safe_regions)
                    && snapshot::safe_body(body, &snapshot.safe_regions)
                {
                    emit(format!("(ElementConsumer {op})"));
                }
            }
            OperationKind::Scatter { body, .. } => {
                if snapshot::safe_body(body, &snapshot.safe_regions) {
                    emit(format!("(ElementConsumer {op})"));
                }
            }
            OperationKind::ReduceByIndex { map, body, .. } => {
                if snapshot::safe_body(map, &snapshot.safe_regions)
                    && snapshot::safe_body(body, &snapshot.safe_regions)
                {
                    emit(format!("(ElementConsumer {op})"));
                }
            }
            OperationKind::BucketScatter { body, shape, .. } => {
                if data.bucket_shapes[*shape].domain_rank == 1
                    && snapshot::safe_body(body, &snapshot.safe_regions)
                {
                    emit(format!("(ElementConsumer {op})"));
                }
            }
            _ => {}
        }
        if let OperationKind::Screma {
            form,
            inputs,
            ownership,
        } = &operation.kind
        {
            emit(format!(
                "(Screma {op} {} {} {} {})",
                inputs.len(),
                ownership.len(),
                form.scans.len(),
                form.reductions.len()
            ));
            if matches!(form.post, SoacBody::Identity(_)) {
                emit(format!("(IdentityPost {op})"));
            }
            if snapshot.discardable.contains(&id) {
                emit(format!("(Safe {op})"));
            }
            if snapshot.movable.contains(&id) {
                emit(format!("(Movable {op})"));
            }
            for (slot, input) in inputs.iter().enumerate() {
                let Array::Value(value) = input else {
                    continue;
                };
                let ExprKind::Project { tuple, index } = data.expressions[*value].kind else {
                    continue;
                };
                let ExprKind::OperationResult(producer) = data.expressions[tuple].kind else {
                    continue;
                };
                emit(format!("(InputFrom {op} {slot} {} {index})", producer.egglog()));
            }
        }
    }
    for &(consumer, producer) in &snapshot.dependencies {
        if included.contains(&consumer) && included.contains(&producer) {
            emit(format!("(DependsOn {} {})", consumer.egglog(), producer.egglog()));
        }
    }
    for &a in &included {
        for &b in &included {
            if a == b || data.operations[a].region != data.operations[b].region {
                continue;
            }
            if super::fusion::same_domain(data, a, b) {
                emit(format!("(SameDomain {} {})", a.egglog(), b.egglog()));
            }
            let routes = super::fusion::routes(data, a, b);
            if !routes.is_empty() {
                emit(format!("(Stream {} {})", a.egglog(), b.egglog()));
                if super::fusion::memory_compatible(data, a, b) {
                    emit(format!("(MemoryCompatible {} {})", a.egglog(), b.egglog()));
                }
                if matches!(data.operations[a].kind, OperationKind::Filter { .. })
                    && super::fusion::masked(&mut data.clone(), a, b).is_some()
                {
                    emit(format!("(Maskable {} {})", a.egglog(), b.egglog()));
                }
                if let OperationKind::Screma { form, .. } = &data.operations[a].kind {
                    let reductions: usize = form.reductions.iter().map(|r| r.neutral.len()).sum();
                    let sliced = super::fusion::input_slices(data, a, b).is_some_and(|s| !s.is_empty());
                    let retained = snapshot.observed.contains(&a)
                        || snapshot.uses.iter().any(|&(p, c, _)| p == a && c != b);
                    if (!sliced || (form.scans.is_empty() && reductions == 0 && !retained))
                        && routes.iter().all(|&r| r >= reductions)
                        && (form.scans.is_empty()
                            || matches!(data.operations[b].kind, OperationKind::Screma { .. })
                                && super::fusion::scremas(&mut data.clone(), a, b, false, true).is_some())
                    {
                        emit(format!("(BarrierCompatible {} {})", a.egglog(), b.egglog()));
                    }
                }
            }
        }
    }
    for &(before, after) in &snapshot.effects {
        if included.contains(&before) && included.contains(&after) {
            emit(format!("(EffectBefore {} {})", before.egglog(), after.egglog()));
        }
    }
    for &(producer, consumer, role) in &snapshot.uses {
        if included.contains(&producer) {
            emit(format!(
                "(Use {} (OperationUse {} {}))",
                producer.egglog(),
                consumer.egglog(),
                role.egglog()
            ));
        }
    }
    for &producer in &snapshot.observed {
        if included.contains(&producer) {
            emit(format!("(Use {} (ReturnUse))", producer.egglog()));
        }
    }
    output
}
