//! Target-aware destruction of semantic EGIR.
//!
//! No pass before this module may replace SegRed/SegScan with physical phase
//! entries.  This module owns scheduling, scratch allocation, descriptor
//! publication, and the transition to low-level SOAC expansion.

use super::from_tlc::ConvertError;
use super::parallelize;
use super::program::{plan_logical_resources, EgirInner};
use super::publish::PipelineDescriptorPublish;
use crate::{IdSource, LoweringProfile, SchedulePolicy};

pub fn schedule(
    inner: &mut EgirInner,
    binding_ids: &mut IdSource<u32>,
    profile: LoweringProfile,
) -> Result<(), ConvertError> {
    // Source ABI publication is private to terminal lowering. It seeds stable
    // names/output declarations but is not observable unless the complete
    // schedule validates and the returned program is committed.
    let unpublished_descriptor = inner.pipeline.clone();
    let mut source_descriptor = unpublished_descriptor.clone();
    source_descriptor.publish_implicit_bindings(&inner.entry_points);
    source_descriptor.publish_graphics_io(&inner.entry_points);
    source_descriptor.relabel_input_storage_names(&inner.input_names);
    inner.pipeline = source_descriptor;

    if profile.schedule == SchedulePolicy::Parallel {
        parallelize::lower(inner, binding_ids);
    } else {
        parallelize::restore_all_serial(inner);
        inner.kernel_schedule =
            parallelize::schedule::KernelSchedule::seed(&inner.pipeline, &inner.entry_points);
    }
    plan_logical_resources(inner);

    let mut descriptor = unpublished_descriptor;
    inner.kernel_schedule.install_phase_shells(&mut descriptor).map_err(ConvertError::Internal)?;
    descriptor.publish_implicit_bindings(&inner.entry_points);
    descriptor.publish_graphics_io(&inner.entry_points);
    inner.kernel_schedule.publish(&mut descriptor).map_err(ConvertError::Internal)?;
    descriptor.relabel_input_storage_names(&inner.input_names);
    inner.pipeline = descriptor;
    Ok(())
}
