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
    ) -> Result<(PhysicalProgram, super::KernelPlanSummary), ConvertError> {
        #[cfg(debug_assertions)]
        {
            let verification = self.validate_for_finalization(&inner.resources, &descriptor);
            debug_assert!(
                verification.is_ok(),
                "internally constructed kernel plan failed verification: {}",
                verification.as_ref().err().map(String::as_str).unwrap_or("unknown verification failure")
            );
        }
        self.check_explicit_dispatch_coverage(&inner.entry_points)
            .map_err(ConvertError::InvalidDispatch)?;
        self.install_phase_shells(&mut descriptor)?;
        let physical_resources = PhysicalResourceTable::allocate(&inner.resources, binding_ids);
        let publications = self.publications(&physical_resources)?;
        let publication_refs = publications.iter().collect::<Vec<_>>();
        descriptor.publish_implicit_bindings(&publication_refs)?;
        descriptor.publish_graphics_io(&publication_refs);
        self.publish(&mut descriptor, &physical_resources)?;
        descriptor.relabel_input_storage_names(&inner.input_names);
        descriptor.rebuild_frame_graph();

        let summary = self.summary();
        let physical = PhysicalProgram::from_plan(
            inner,
            &self,
            &physical_resources,
            profile.schedule == SchedulePolicy::SingleStage,
            descriptor,
        )?;
        crate::egir::verify_physical::check(&physical, &physical_resources)?;
        Ok((physical, summary))
    }
}
