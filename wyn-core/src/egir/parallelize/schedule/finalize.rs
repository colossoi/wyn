//! Validation, descriptor publication, binding allocation, and physical construction.

use crate::egir;
use crate::pipeline_descriptor;
use std::collections::{HashMap, HashSet};

use super::{execution_workgroup, KernelDispatch, KernelDomain, KernelPlan};
use crate::egir::from_tlc::ConvertError;
use crate::egir::program::KernelProgram;
use crate::egir::program::{
    host_resource_names, physicalize_program, EntryPublication, PhysicalResourceTable,
};
use crate::egir::publish::{PipelineDescriptorPublish, StageEntryAssociations};
use crate::pipeline_descriptor::{
    ComputeStage, DispatchLen, DispatchSize, Pipeline, PipelineDescriptor, StageBindingUses,
};
use crate::{BindingRef, SchedulePolicy};

impl KernelPlan {
    pub(in crate::egir::parallelize) fn finalize(
        self,
        mut program: KernelProgram<()>,
    ) -> Result<egir::parallelize::Planned, ConvertError> {
        let profile = program.data.profile;
        self.check_explicit_dispatch_coverage().map_err(ConvertError::InvalidDispatch)?;
        let physical_resources = self.publish_physical_layout(&mut program)?;
        let physical_kernels = super::PhysicalKernelGraph::from(&self);
        let entries = self.into_physical_entries();
        let physical = physicalize_program(
            program,
            entries,
            &physical_resources,
            profile.schedule == SchedulePolicy::Serial,
            physical_kernels,
            profile,
        )?;
        egir::verify_physical::check(&physical, &physical_resources)?;
        Ok(physical)
    }

    /// Publish physical binding/layout facts derived from the validated
    /// kernel graph without consuming its topology or bodies.
    fn publish_physical_layout(
        &self,
        program: &mut KernelProgram<()>,
    ) -> Result<PhysicalResourceTable, ConvertError> {
        self.install_phase_shells(&mut program.data.core.pipeline)?;
        let mut reserved_bindings = program
            .data
            .core
            .pipeline
            .pipelines
            .iter()
            .flat_map(|pipeline| match pipeline {
                Pipeline::Compute(compute) => compute.bindings.iter(),
                Pipeline::Graphics(graphics) => graphics.bindings.iter(),
            })
            .filter_map(binding_ref)
            .collect::<HashSet<_>>();
        reserved_bindings.extend(&program.data.reserved_bindings);
        let physical_resources = PhysicalResourceTable::allocate_avoiding(
            &program.data.core.resources,
            &mut program.global_context.binding_ids,
            reserved_bindings,
        );
        let publications = self.entry_publications(&physical_resources)?;
        let publication_refs = publications.iter().collect::<Vec<_>>();
        let stage_entries = self.stage_entry_associations(&program.data.core.pipeline)?;
        program.data.core.pipeline.publish_implicit_bindings(&publication_refs, &stage_entries)?;
        program.data.core.pipeline.publish_graphics_io(&publication_refs, &stage_entries);
        self.publish(&mut program.data.core.pipeline, &physical_resources)?;
        program.data.core.pipeline.publish_stage_binding_uses(&publication_refs, &stage_entries);
        let input_names = host_resource_names(&program.data.core.resources);
        program.data.core.pipeline.relabel_input_storage_names(&input_names);
        program.data.core.pipeline.rebuild_frame_graph();
        // `install_phase_shells` replaces authored stages with their physical
        // phase family. Preserve the matching structural identities for SSA
        // elaboration; the authored associations no longer line up by index.
        program.data.core.stage_entries = stage_entries;
        Ok(physical_resources)
    }

    /// Entry ABI records in deterministic descriptor-publication order. The
    /// kernel plan, rather than physical graphs, is the sole authority for
    /// backend-visible entry metadata.
    fn entry_publications(
        &self,
        resources: &PhysicalResourceTable,
    ) -> Result<Vec<EntryPublication>, String> {
        let mut names = HashSet::new();
        let mut publications = Vec::new();
        for source in self.source_entries.values() {
            if names.insert(source.name.as_str()) {
                publications.push(source.publication(resources)?);
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

    fn stage_entry_associations(
        &self,
        descriptor: &PipelineDescriptor,
    ) -> Result<StageEntryAssociations, String> {
        if descriptor.pipelines.len() != self.pipelines.len() {
            return Err("publication group count differs from descriptor".into());
        }
        Ok(self
            .pipelines
            .iter()
            .map(|scheduled| match &scheduled.template {
                Pipeline::Compute(_) => {
                    self.phase_ids_in(scheduled.id).iter().map(|id| self.phase(*id).entry.id).collect()
                }
                Pipeline::Graphics(_) => scheduled.graphics_entries.clone(),
            })
            .collect())
    }

    fn install_phase_shells(&self, descriptor: &mut PipelineDescriptor) -> Result<(), String> {
        descriptor.pipelines = self
            .pipelines
            .iter()
            .map(|scheduled| {
                let mut pipeline = scheduled.template.clone();
                if let Pipeline::Compute(compute) = &mut pipeline {
                    compute.stages = self
                        .phase_ids_in(scheduled.id)
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
                }
                pipeline
            })
            .collect();
        Ok(())
    }

    fn publish(
        &self,
        descriptor: &mut PipelineDescriptor,
        physical_resources: &PhysicalResourceTable,
    ) -> Result<(), String> {
        for (index, scheduled) in self.pipelines.iter().enumerate() {
            if !matches!(scheduled.template, Pipeline::Compute(_)) {
                continue;
            }
            let phase_ids = self.phase_ids_in(scheduled.id);
            let Some(Pipeline::Compute(compute)) = descriptor.pipelines.get_mut(index) else {
                return Err(
                    "scheduled compute pipeline was not installed at its structural descriptor position"
                        .into(),
                );
            };
            let binding_index = compute
                .bindings
                .iter()
                .enumerate()
                .filter_map(|(index, binding)| binding_ref(binding).map(|binding| (binding, index)))
                .collect::<HashMap<_, _>>();
            let mut stages = Vec::with_capacity(phase_ids.len());
            for id in phase_ids {
                let phase = self.phase(*id);
                let mut reads = Vec::new();
                let mut writes = Vec::new();
                for resource in phase.resources() {
                    let binding = physical_resources.binding(resource.resource);
                    let index = *binding_index.get(&binding).ok_or_else(|| {
                        format!(
                            "kernel `{}` references unpublished resource {:?} ({:?}) at {binding}; published bindings: {:?}",
                            phase.entry_point(),
                            resource.resource,
                            physical_resources.logical_name(resource.resource),
                            binding_index.keys().collect::<Vec<_>>()
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

    fn phase_owner(&self, phase: &super::PreparedKernel) -> String {
        phase
            .source_entry
            .and_then(|source| self.source_entries.get(&source))
            .map(|source| source.name.clone())
            .unwrap_or_else(|| phase.entry_point().to_owned())
    }

    fn check_explicit_dispatch_coverage(&self) -> Result<(), String> {
        for phase in self
            .pipelines
            .iter()
            .filter(|pipeline| matches!(pipeline.template, Pipeline::Compute(_)))
            .flat_map(|pipeline| self.phase_ids_in(pipeline.id))
            .map(|id| self.phase(*id))
        {
            let KernelDispatch {
                domain: KernelDomain::Fixed { x, y, z },
                explicit: true,
            } = &phase.dispatch
            else {
                continue;
            };
            let Some(count) = phase.required_elements else {
                continue;
            };
            let (wx, wy, wz) = execution_workgroup(&phase.entry.execution_model);
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

fn binding_ref(binding: &pipeline_descriptor::Binding) -> Option<BindingRef> {
    binding.slot().map(|(set, binding)| BindingRef::new(set, binding))
}
