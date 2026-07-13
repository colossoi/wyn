//! Invariants of the validated-plan-to-physical-EGIR boundary.

use std::collections::HashSet;

use super::program::PhysicalProgram;
use super::types::{ENode, EgirSoac, FilterState, SegOpKind, SideEffectKind};

pub fn check(program: &PhysicalProgram) -> Result<(), String> {
    let entries = program.entry_points.iter().map(|entry| entry.name.as_str()).collect::<HashSet<_>>();
    for phase in program.plan.phases() {
        if !entries.contains(phase.entry_point.as_str()) {
            return Err(format!(
                "validated kernel `{}` has no physical entry",
                phase.entry_point
            ));
        }
        for required in &phase.resources {
            if program.physical_resources.binding(required.resource).is_none() {
                return Err(format!(
                    "validated kernel `{}` has no binding for resource {:?}",
                    phase.entry_point, required.resource
                ));
            }
        }
    }

    for entry in &program.entry_points {
        for (_, node) in &entry.graph.nodes {
            if matches!(
                node,
                ENode::Pure {
                    op: super::types::PureOp::StorageView(crate::op::PureViewSource::Resource(_)),
                    ..
                }
            ) {
                return Err(format!(
                    "physical entry `{}` still contains a semantic resource view",
                    entry.name
                ));
            }
        }
        for (_, block) in &entry.graph.skeleton.blocks {
            for effect in &block.side_effects {
                match &effect.kind {
                    SideEffectKind::Soac(EgirSoac::Filter {
                        state: FilterState::Semantic { .. },
                        ..
                    }) => {
                        return Err(format!(
                            "physical entry `{}` contains an unscheduled filter",
                            entry.name
                        ));
                    }
                    SideEffectKind::Soac(EgirSoac::Seg {
                        kind: SegOpKind::SegRed { .. } | SegOpKind::SegScan { .. },
                        ..
                    }) => {
                        return Err(format!(
                            "physical entry `{}` contains an unresolved reduction/scan",
                            entry.name
                        ));
                    }
                    _ => {}
                }
            }
        }
    }
    Ok(())
}
