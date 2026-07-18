//! Deterministic attachment of allocation-created producer entries.

#![cfg_attr(not(test), deny(clippy::expect_used, clippy::unwrap_used))]

use super::*;

pub(super) fn attach_materializations(
    inner: &AllocatedProgram,
    schedule: &mut schedule::KernelPlan,
    resources: &planning::ResourceIndex<'_>,
    flows: &planning::ResourceFlowIndex,
    candidates: &planning::CandidateIndex,
    policy: ParallelPolicy,
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> error::Result<()> {
    let mut ready = std::collections::BTreeSet::new();
    for (_, flow) in flows.flows() {
        for consumer in &flow.consumers {
            if schedule.contains_flow_source(*consumer) {
                ready.insert((flow.producer, *consumer));
            }
        }
    }

    while let Some((producer_id, consumer_id)) = ready.pop_first() {
        if schedule.contains_flow_source(producer_id) {
            continue;
        }
        let consumer = schedule.kernel_for_flow_source(consumer_id).ok_or_else(|| {
            error::ParallelizeError::Invalid(format!(
                "scheduled flow consumer {consumer_id:?} has no kernel handle"
            ))
        })?;
        let crate::egir::program::CompilerFlowEndpoint::Materialization(id) = producer_id else {
            return Err(error::ParallelizeError::Invalid(
                "typed entry/prepass producer was omitted while seeding the kernel plan".into(),
            ));
        };
        let requirement = inner
            .materializations
            .get(id.0 as usize)
            .filter(|requirement| requirement.id == id)
            .ok_or_else(|| {
                error::ParallelizeError::Invalid(format!(
                    "materialization flow references missing requirement {id:?}"
                ))
            })?;
        let materialization = schedule.add_materialization_before(consumer, requirement)?;
        let body = crate::egir::program::PlannedEntry::project(&requirement.entry)?;
        if segmented_recipe_effect(&body).is_some_and(|(_, _, effect)| {
            matches!(
                &effect.kind,
                SideEffectKind::Soac(SoacEffect(
                    _,
                    Soac::Screma(screma::Op::Reduce { .. } | screma::Op::Scan { .. })
                ))
            )
        }) {
            plan_segmented_kernel_body(
                body,
                materialization,
                schedule,
                resources,
                candidates,
                policy,
                effect_ids,
            )?;
        }
        for upstream in flows.incoming(producer_id) {
            ready.insert((*upstream, producer_id));
        }
    }
    Ok(())
}
