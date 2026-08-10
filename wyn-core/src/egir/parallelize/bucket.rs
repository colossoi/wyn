//! Three-stage lowering for capacity-bounded bucket insertion.

use super::*;
use crate::egir::program::LogicalSize;
use crate::egir::soac::hist;
use std::collections::HashSet;

#[derive(Clone)]
pub(super) struct BucketCandidate {
    site: SideEffectSite,
    space: SegSpace,
    bucket_count: u32,
    destination: ResourceId,
    input_resources: Vec<ResourceId>,
    counts: ResourceId,
    overflow: ResourceId,
}

pub(super) fn analyze_bucket_candidate(
    entry: &crate::egir::program::PlannedEntry,
    site: SideEffectSite,
) -> error::Result<Option<BucketCandidate>> {
    let effect = entry.graph.skeleton.effect(site);
    let SideEffectKind::Soac(SoacEffect(_, Soac::Hist(op))) = &effect.kind else {
        return Err(error::ParallelizeError::Invalid(
            "bucket_scatter candidate no longer contains a histogram".into(),
        ));
    };
    let hist::State::Segmented(space) = &op.state else {
        return Ok(None);
    };
    let hist::UpdatePolicy::BucketInsert { counts, overflow, .. } = op.body.update_policy else {
        return Ok(None);
    };
    let logical =
        entry.resource_declarations.iter().find(|declaration| declaration.resource == counts).ok_or_else(
            || error::ParallelizeError::Invalid("bucket_scatter count resource has no declaration".into()),
        )?;
    let LogicalSize::FixedBytes(bytes) = logical.size else {
        return Ok(None);
    };
    let bucket_count = u32::try_from(bytes / 4).map_err(|_| {
        error::ParallelizeError::Invalid("bucket_scatter count resource is too large".into())
    })?;
    if bytes % 4 != 0 || bucket_count == 0 {
        return Err(error::ParallelizeError::Invalid(
            "bucket_scatter count resource has an invalid size".into(),
        ));
    }
    let resource_for = |node| {
        let resources = entry.resources_referenced_by_nodes(&entry.graph, [node]);
        let mut resources = resources.into_iter();
        let resource = resources.next();
        if resources.next().is_some() {
            return Err(error::ParallelizeError::Invalid(
                "bucket_scatter operand has multiple resource identities".into(),
            ));
        }
        Ok(resource)
    };
    let destination = resource_for(effect.operand_nodes[0])?.ok_or_else(|| {
        error::ParallelizeError::Invalid("bucket_scatter destination has no resource identity".into())
    })?;
    // Envelope fusion can replace key/value arrays with virtual sources such
    // as `iota`. Those operands are computed inside the insertion kernel and
    // intentionally have no storage resource to schedule.
    let input_resources = effect.operand_nodes[1..1 + op.body.inputs.len()]
        .iter()
        .map(|node| resource_for(*node))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect();
    Ok(Some(BucketCandidate {
        site,
        space: space.clone(),
        bucket_count,
        destination,
        input_resources,
        counts: counts.0,
        overflow: overflow.0,
    }))
}

impl KernelPlanBuilder<'_, '_> {
    pub(super) fn lower_parallel_bucket(
        &mut self,
        body: crate::egir::program::PlannedEntry,
        kernel: schedule::KernelId,
        candidate: BucketCandidate,
        output_projection: Option<Vec<usize>>,
    ) -> error::Result<()> {
        let insert_domain = schedule::domain_from_space(&candidate.space)
            .unwrap_or(schedule::KernelDomain::Fixed { x: 1, y: 1, z: 1 });
        let resources = body.resource_declarations.clone();
        let init_spec = ProjectionSpec::unit(
            format!("{}_bucket_init", body.name),
            body.execution_model.clone(),
            resources.clone(),
        );
        let insert_spec = ProjectionSpec::unit(
            format!("{}_bucket_insert", body.name),
            body.execution_model.clone(),
            resources.clone(),
        );
        let finish_spec = ProjectionSpec::preserving_interface(&body, resources);
        let init_body = project_single_effect_body(&body, candidate.site, init_spec)?;
        let init = BuiltPhase::new(
            init_body,
            vec![
                schedule::ScheduledResource {
                    resource: candidate.counts,
                    access: crate::ResourceAccess::Write,
                },
                schedule::ScheduledResource {
                    resource: candidate.overflow,
                    access: crate::ResourceAccess::Write,
                },
            ],
        )
        .bucket(
            schedule::KernelDispatch::explicit(schedule::KernelDomain::Elements(
                crate::pipeline_descriptor::DispatchLen::Fixed {
                    count: candidate.bucket_count,
                },
            )),
            hist::ParallelStage::Init,
        );
        let insert_body = project_single_effect_body(&body, candidate.site, insert_spec)?;
        let mut insert_resources = vec![
            schedule::ScheduledResource {
                resource: candidate.destination,
                access: crate::ResourceAccess::Write,
            },
            schedule::ScheduledResource {
                resource: candidate.counts,
                access: crate::ResourceAccess::ReadWrite,
            },
            schedule::ScheduledResource {
                resource: candidate.overflow,
                access: crate::ResourceAccess::Write,
            },
        ];
        insert_resources.extend(candidate.input_resources.iter().map(|resource| {
            schedule::ScheduledResource {
                resource: *resource,
                access: crate::ResourceAccess::Read,
            }
        }));
        let insert_resources = merge_scheduled_resources(&[], &insert_resources);
        let insert = BuiltPhase::new(insert_body, insert_resources).bucket(
            schedule::KernelDispatch::inferred(insert_domain),
            hist::ParallelStage::Insert,
        );
        let finish_body = project_kernel_body(&body, finish_spec)?;
        let published_outputs = finish_body
            .outputs
            .iter()
            .filter_map(|output| output.resource.map(|resource| resource.0))
            .collect::<HashSet<_>>();
        let output_resources = published_outputs
            .into_iter()
            .filter(|resource| *resource != candidate.destination && *resource != candidate.counts)
            .map(|resource| schedule::ScheduledResource {
                resource,
                access: crate::ResourceAccess::Write,
            })
            .collect::<Vec<_>>();
        let finish_resources = merge_scheduled_resources(
            &[schedule::ScheduledResource {
                resource: candidate.overflow,
                access: crate::ResourceAccess::Read,
            }],
            &output_resources,
        );
        let finish = BuiltPhase::new(finish_body, finish_resources)
            .bucket(
                schedule::KernelDispatch::explicit(schedule::KernelDomain::Fixed { x: 1, y: 1, z: 1 }),
                hist::ParallelStage::Finish,
            )
            .with_output_projection(output_projection);
        self.schedule.replace_chain(kernel, vec![init, insert], finish, Vec::new())?;
        Ok(())
    }
}
