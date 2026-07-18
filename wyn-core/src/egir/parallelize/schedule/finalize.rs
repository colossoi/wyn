//! Validation, publication, binding allocation, and physical construction.

use super::KernelPlan;
use crate::egir::from_tlc::ConvertError;
use crate::egir::program::{AllocatedProgram, PhysicalProgram, PhysicalResourceTable};
use crate::egir::publish::PipelineDescriptorPublish;
use crate::pipeline_descriptor::PipelineDescriptor;
use crate::{IdSource, LoweringProfile, SchedulePolicy};

impl KernelPlan {
    pub(in crate::egir::parallelize) fn finalize(
        self,
        inner: AllocatedProgram,
        binding_ids: &mut IdSource<u32>,
        profile: LoweringProfile,
        mut descriptor: PipelineDescriptor,
    ) -> Result<(PhysicalProgram, super::PublishedKernelPlan), ConvertError> {
        self.check_explicit_dispatch_coverage(&inner.entry_points)
            .map_err(ConvertError::InvalidDispatch)?;
        let validated = self
            .into_validated(&inner.entry_points, &inner.resources, &descriptor)
            .map_err(ConvertError::Internal)?;

        validated.install_phase_shells(&mut descriptor).map_err(ConvertError::Internal)?;
        let physical_resources = PhysicalResourceTable::allocate(&inner.resources, binding_ids);
        let publications = validated.publications(&physical_resources).map_err(ConvertError::Internal)?;
        let publication_refs = publications.iter().collect::<Vec<_>>();
        descriptor.publish_implicit_bindings(&publication_refs).map_err(ConvertError::DescriptorLayout)?;
        descriptor.publish_graphics_io(&publication_refs);
        validated.publish(&mut descriptor, &physical_resources).map_err(ConvertError::Internal)?;
        descriptor.relabel_input_storage_names(&inner.input_names);
        descriptor.rebuild_frame_graph();

        let published_plan = validated.published_plan();
        let physical = PhysicalProgram::from_validated(
            inner,
            validated,
            physical_resources,
            profile.schedule == SchedulePolicy::SingleStage,
            descriptor,
        )
        .map_err(ConvertError::Internal)?;
        crate::egir::verify_physical::check(&physical).map_err(ConvertError::Internal)?;
        Ok((physical, published_plan))
    }
}
