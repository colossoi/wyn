//! Target-aware destruction of semantic EGIR.
//!
//! No pass before this module may replace SegRed/SegScan with physical phase
//! entries.  This module owns scheduling, scratch allocation, descriptor
//! publication, and the transition to low-level SOAC expansion.

use super::from_tlc::ConvertError;
use super::parallelize;
use super::program::{AllocatedProgram, PhysicalProgram, PhysicalResourceTable};
use super::publish::PipelineDescriptorPublish;
use crate::{IdSource, LoweringProfile, SchedulePolicy};

pub fn schedule(
    inner: AllocatedProgram,
    binding_ids: &mut IdSource<u32>,
    profile: LoweringProfile,
) -> Result<PhysicalProgram, ConvertError> {
    let unpublished_descriptor = inner.pipeline.clone();

    let kernel_plan = if profile.schedule == SchedulePolicy::Parallel {
        parallelize::lower(&inner).map_err(ConvertError::Internal)?
    } else {
        parallelize::lower_sequential(&inner)
    };
    kernel_plan
        .check_explicit_dispatch_coverage(&inner.entry_points)
        .map_err(ConvertError::InvalidDispatch)?;

    let validated = kernel_plan
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
    let physical = PhysicalProgram::from_validated(
        inner,
        validated,
        physical_resources,
        profile.schedule == SchedulePolicy::SingleStage,
        descriptor,
    )
    .map_err(ConvertError::Internal)?;
    super::verify_physical::check(&physical).map_err(ConvertError::Internal)?;
    Ok(physical)
}
