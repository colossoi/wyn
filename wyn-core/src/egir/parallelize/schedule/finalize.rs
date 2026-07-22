//! Validation, descriptor publication, binding allocation, and physical construction.

use std::collections::{HashMap, HashSet};

use super::{execution_workgroup, KernelDispatch, KernelDomain, KernelPlan, PhaseGroup};
use crate::egir::from_tlc::ConvertError;
use crate::egir::program::{
    AllocatedProgram, EntryPublication, PhysicalProgram, PhysicalResourceTable, SemanticEntry,
};
use crate::egir::publish::PipelineDescriptorPublish;
use crate::pipeline_descriptor::{
    ComputeStage, DispatchLen, DispatchSize, Pipeline, PipelineDescriptor, StageBindingUses,
};
use crate::{BindingRef, IdSource, LoweringProfile, SchedulePolicy};

impl KernelPlan {
    pub(in crate::egir::parallelize) fn finalize(
        self,
        inner: AllocatedProgram,
        binding_ids: &mut IdSource<u32>,
        profile: LoweringProfile,
    ) -> Result<(PhysicalProgram, super::KernelPlanSummary), ConvertError> {
        let mut inner = inner;
        let mut descriptor = std::mem::take(&mut inner.pipeline);
        #[cfg(debug_assertions)]
        {
            let verification = self.validate();
            debug_assert!(
                verification.is_ok(),
                "internally constructed kernel plan failed verification: {}",
                verification.as_ref().err().map(String::as_str).unwrap_or("unknown verification failure")
            );
        }
        self.check_explicit_dispatch_coverage(&inner.entry_points)
            .map_err(ConvertError::InvalidDispatch)?;
        self.install_phase_shells(&mut descriptor)?;
        let mut reserved_bindings = descriptor
            .pipelines
            .iter()
            .flat_map(|pipeline| match pipeline {
                Pipeline::Compute(compute) => compute.bindings.iter(),
                Pipeline::Graphics(graphics) => graphics.bindings.iter(),
            })
            .filter_map(binding_ref)
            .collect::<HashSet<_>>();
        reserved_bindings.extend(
            inner
                .entry_points
                .iter()
                .flat_map(|entry| &entry.inputs)
                .filter_map(|input| input.descriptor_binding()),
        );
        let physical_resources =
            PhysicalResourceTable::allocate_avoiding(&inner.resources, binding_ids, reserved_bindings);
        let publications = self.publications(&physical_resources)?;
        let publication_refs = publications.iter().collect::<Vec<_>>();
        descriptor.publish_implicit_bindings(&publication_refs)?;
        descriptor.publish_graphics_io(&publication_refs);
        self.publish(&mut descriptor, &physical_resources)?;
        descriptor.publish_stage_binding_uses(&publication_refs);
        descriptor.relabel_input_storage_names(&inner.input_names);
        descriptor.rebuild_frame_graph();

        let summary = super::KernelPlanSummary::from(&self);
        let physical = PhysicalProgram::from_plan(
            inner,
            &self,
            &physical_resources,
            profile.schedule == SchedulePolicy::Serial,
            descriptor,
        )?;
        crate::egir::verify_physical::check(&physical, &physical_resources)?;
        Ok((physical, summary))
    }

    /// Entry ABI records in deterministic descriptor-publication order. The
    /// kernel plan, rather than physical graphs, is the sole authority for
    /// backend-visible entry metadata.
    fn publications(&self, resources: &PhysicalResourceTable) -> Result<Vec<EntryPublication>, String> {
        let mut names = HashSet::new();
        let mut publications = Vec::new();
        for source in &self.source_entries {
            if names.insert(source.publication.name.as_str()) {
                publications.push(source.publication.publication(resources)?);
            }
        }
        for phase in self.phases() {
            let entry = &phase.entry;
            if names.insert(entry.name.as_str()) {
                publications.push(entry.publication(resources)?);
            }
        }
        Ok(publications)
    }

    fn install_phase_shells(&self, descriptor: &mut PipelineDescriptor) -> Result<(), String> {
        let mut rebuilt = descriptor
            .pipelines
            .iter()
            .enumerate()
            .filter_map(|(order, pipeline)| {
                matches!(pipeline, Pipeline::Graphics(_)).then(|| (order, pipeline.clone()))
            })
            .collect::<Vec<_>>();
        for scheduled in &self.pipelines {
            let mut compute = scheduled.template.clone();
            let phase_ids = self.phase_ids_in(PhaseGroup::Pipeline(scheduled.id));
            compute.stages = phase_ids
                .iter()
                .map(|id| self.phase(*id))
                .map(|phase| ComputeStage {
                    entry_point: phase.entry_point().to_owned(),
                    owner: self.phase_owner(phase),
                    workgroup_size: phase.workgroup_size(),
                    dispatch_size: DispatchSize::Fixed {
                        x: 1,
                        y: 1,
                        z: 1,
                        explicit: false,
                    },
                    uses: StageBindingUses::default(),
                })
                .collect();
            rebuilt.push((scheduled.order, Pipeline::Compute(compute)));
        }
        rebuilt.sort_by_key(|(order, _)| *order);
        descriptor.pipelines = rebuilt.into_iter().map(|(_, pipeline)| pipeline).collect();
        Ok(())
    }

    fn publish(
        &self,
        descriptor: &mut PipelineDescriptor,
        physical_resources: &PhysicalResourceTable,
    ) -> Result<(), String> {
        for scheduled in &self.pipelines {
            let phase_ids = self.phase_ids_in(PhaseGroup::Pipeline(scheduled.id));
            let Some(Pipeline::Compute(compute)) = descriptor.pipelines.iter_mut().find(|pipeline| {
                matches!(pipeline, Pipeline::Compute(candidate) if phase_ids.iter().any(|phase| {
                    candidate.stages.iter().any(|stage| stage.entry_point == self.phase(*phase).entry_point())
                }))
            }) else {
                return Err("scheduled compute pipeline was not installed".into());
            };
            let binding_index = compute
                .bindings
                .iter()
                .enumerate()
                .filter_map(|(index, binding)| binding_ref(binding).map(|binding| (binding, index)))
                .collect::<HashMap<_, _>>();
            let mut stages = Vec::with_capacity(phase_ids.len());
            for id in &phase_ids {
                let phase = self.phase(*id);
                let mut reads = Vec::new();
                let mut writes = Vec::new();
                for resource in phase.resources() {
                    let binding = physical_resources.binding(resource.resource);
                    let index = *binding_index.get(&binding).ok_or_else(|| {
                        format!(
                            "kernel `{}` references unpublished storage {binding}",
                            phase.entry_point()
                        )
                    })?;
                    if resource.access.reads() && !reads.contains(&index) {
                        reads.push(index);
                    }
                    if resource.access.writes() && !writes.contains(&index) {
                        writes.push(index);
                    }
                }
                let dispatch_size = match &phase.dispatch.domain {
                    KernelDomain::Fixed { x, y, z } => DispatchSize::Fixed {
                        x: *x,
                        y: *y,
                        z: *z,
                        explicit: phase.dispatch.explicit,
                    },
                    KernelDomain::Elements(len) => DispatchSize::DerivedFrom {
                        len: len.clone(),
                        workgroup_size: phase.workgroup_size().0,
                    },
                    KernelDomain::ResourceElements { resource, elem_bytes } => {
                        let binding = physical_resources.binding(*resource);
                        DispatchSize::DerivedFrom {
                            len: DispatchLen::InputBinding {
                                set: binding.set,
                                binding: binding.binding,
                                elem_bytes: *elem_bytes,
                            },
                            workgroup_size: phase.workgroup_size().0,
                        }
                    }
                };
                stages.push(ComputeStage {
                    entry_point: phase.entry_point().to_owned(),
                    owner: self.phase_owner(phase),
                    workgroup_size: phase.workgroup_size(),
                    dispatch_size,
                    uses: StageBindingUses { reads, writes },
                });
            }
            compute.stages = stages;
        }
        Ok(())
    }

    fn phase_owner(&self, phase: &super::KernelPhase) -> String {
        phase
            .source_entry
            .and_then(|source| self.source_entries.get(source.index()))
            .map(|source| source.publication.name.clone())
            .unwrap_or_else(|| phase.entry_point().to_owned())
    }

    fn check_explicit_dispatch_coverage(&self, entries: &[SemanticEntry]) -> Result<(), String> {
        let by_name = entries.iter().map(|entry| (entry.name.as_str(), entry)).collect::<HashMap<_, _>>();
        for phase in
            self.phases.iter().filter(|phase| matches!(phase.placement.group, PhaseGroup::Pipeline(_)))
        {
            let KernelDispatch {
                domain: KernelDomain::Fixed { x, y, z },
                explicit: true,
            } = &phase.dispatch
            else {
                continue;
            };
            let Some(entry) = by_name.get(phase.entry_point()) else {
                continue;
            };
            let Some(count) = phase.required_elements else {
                continue;
            };
            let (wx, wy, wz) = execution_workgroup(&entry.execution_model);
            let total = x
                .saturating_mul(*y)
                .saturating_mul(*z)
                .saturating_mul(wx)
                .saturating_mul(wy)
                .saturating_mul(wz);
            if total < count {
                return Err(format!(
                    "#[dispatch({x}, {y}, {z})] launches {total} threads but entry `{}` requires {count}; {} elements would be dropped",
                    phase.entry_point(),
                    count - total,
                ));
            }
        }
        Ok(())
    }
}

fn binding_ref(binding: &crate::pipeline_descriptor::Binding) -> Option<BindingRef> {
    binding.slot().map(|(set, binding)| BindingRef::new(set, binding))
}
