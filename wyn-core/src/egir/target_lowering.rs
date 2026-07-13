//! Target-aware destruction of semantic EGIR.
//!
//! No pass before this module may replace SegRed/SegScan with physical phase
//! entries.  This module owns scheduling, scratch allocation, descriptor
//! publication, and the transition to low-level SOAC expansion.

use super::from_tlc::ConvertError;
use super::parallelize;
use super::program::{
    physicalize_resource_references, PhysicalProgram, PhysicalResourceTable, SemanticProgram,
};
use super::publish::PipelineDescriptorPublish;
use crate::{IdSource, LoweringProfile, SchedulePolicy};

pub fn schedule(
    mut inner: SemanticProgram,
    binding_ids: &mut IdSource<u32>,
    profile: LoweringProfile,
) -> Result<PhysicalProgram, ConvertError> {
    super::program::verify_allocated_resources(&inner).map_err(ConvertError::Internal)?;
    let unpublished_descriptor = inner.pipeline.clone();

    if profile.schedule == SchedulePolicy::Parallel {
        parallelize::lower(&mut inner);
    } else {
        parallelize::restore_all_serial(&mut inner);
        let mut schedule = parallelize::schedule::KernelPlan::seed(
            &inner.pipeline,
            &inner.entry_points,
            &inner.prepasses,
            &inner.resources,
        );
        parallelize::attach_compiler_prepasses(&inner, &mut schedule);
        schedule.coalesce_resource_flows(&inner.resources);
        inner.kernel_plan = schedule;
    }
    parallelize::finalize_scheduled_states(&mut inner);
    inner
        .kernel_plan
        .check_explicit_dispatch_coverage(&inner.entry_points)
        .map_err(ConvertError::InvalidDispatch)?;

    let validated = inner
        .kernel_plan
        .clone()
        .into_validated(&inner.entry_points, &inner.resources, &unpublished_descriptor)
        .map_err(ConvertError::Internal)?;

    let mut descriptor = unpublished_descriptor;
    validated.install_phase_shells(&mut descriptor).map_err(ConvertError::Internal)?;
    let physical_resources = PhysicalResourceTable::allocate(&inner.resources, binding_ids);
    let publications = validated.publications(&physical_resources).map_err(ConvertError::Internal)?;
    let publication_refs = publications.iter().collect::<Vec<_>>();
    descriptor.publish_implicit_bindings(&publication_refs).map_err(ConvertError::DescriptorLayout)?;
    descriptor.publish_graphics_io(&publication_refs);
    validated.publish(&mut descriptor, &physical_resources).map_err(ConvertError::Internal)?;
    descriptor.relabel_input_storage_names(&inner.input_names);
    descriptor.rebuild_frame_graph();
    inner.pipeline = descriptor;
    physicalize_resource_references(&mut inner, &physical_resources).map_err(ConvertError::Internal)?;
    let physical = PhysicalProgram::from_validated(inner, validated, physical_resources)
        .map_err(ConvertError::Internal)?;
    super::verify_physical::check(&physical).map_err(ConvertError::Internal)?;
    Ok(physical)
}
