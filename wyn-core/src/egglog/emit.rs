//! Emit only facts consumed by fusion. The complete program stays in the sidecar.

use super::data::{Array, AssociatedData, ExprKind, OperationKind, SoacBody};
use super::{snapshot, SCHEMA};
use std::collections::BTreeSet;

pub(super) fn program(data: &AssociatedData) -> String {
    let snapshot = snapshot::analyze(data);
    let scopes: BTreeSet<_> = snapshot
        .live
        .iter()
        .filter_map(|&op| {
            matches!(data.operations[op].kind, OperationKind::Screma { .. })
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
