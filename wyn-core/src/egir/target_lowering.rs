//! Target-aware destruction of semantic EGIR.
//!
//! No pass before this module may replace SegRed/SegScan with physical phase
//! entries.  This module owns scheduling, scratch allocation, descriptor
//! publication, and the transition to low-level SOAC expansion.

use super::from_tlc::ConvertError;
use super::parallelize;
use super::program::{
    physicalize_resource_references, refresh_logical_resources, PhysicalProgram, PhysicalResourceTable,
    SemanticProgram,
};
use super::publish::PipelineDescriptorPublish;
use crate::{IdSource, LoweringProfile, SchedulePolicy};

pub fn schedule(
    mut inner: SemanticProgram,
    binding_ids: &mut IdSource<u32>,
    profile: LoweringProfile,
) -> Result<PhysicalProgram, ConvertError> {
    physicalize_resource_references(&mut inner).map_err(ConvertError::Internal)?;
    // Source ABI publication is private to terminal lowering. It seeds stable
    // names/output declarations but is not observable unless the complete
    // schedule validates and the returned program is committed.
    let unpublished_descriptor = inner.pipeline.clone();
    let mut source_descriptor = unpublished_descriptor.clone();
    source_descriptor
        .publish_implicit_bindings(&inner.entry_points)
        .map_err(ConvertError::DescriptorLayout)?;
    source_descriptor.publish_graphics_io(&inner.entry_points);
    source_descriptor.relabel_input_storage_names(&inner.input_names);
    inner.pipeline = source_descriptor;

    if profile.schedule == SchedulePolicy::Parallel {
        parallelize::lower(&mut inner);
    } else {
        parallelize::restore_all_serial(&mut inner);
        let mut schedule =
            parallelize::schedule::KernelPlan::seed(&inner.pipeline, &inner.entry_points, &inner.resources);
        parallelize::attach_materialization_prepasses(&inner, &mut schedule);
        schedule.coalesce_compiler_dependencies(&inner.entry_points);
        inner.kernel_plan = schedule;
    }
    parallelize::finalize_scheduled_states(&mut inner);
    // Re-mirror after lowering (split clones / phase entries added new host and
    // intermediate storage); the parallel SegRed/SegScan ops are gone, so no
    // further scratch is drawn here.
    let _ = binding_ids;
    refresh_logical_resources(&mut inner);

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
    descriptor.publish_implicit_bindings(&inner.entry_points).map_err(ConvertError::DescriptorLayout)?;
    descriptor.publish_graphics_io(&inner.entry_points);
    let physical_resources = PhysicalResourceTable::from_resources(&inner.resources);
    validated.publish(&mut descriptor, &physical_resources).map_err(ConvertError::Internal)?;
    descriptor.relabel_input_storage_names(&inner.input_names);
    descriptor.rebuild_frame_graph();
    inner.pipeline = descriptor;
    Ok(PhysicalProgram::from_validated(inner, validated))
}
