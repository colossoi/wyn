//! Invariants of the validated-plan-to-physical-EGIR boundary.

use std::collections::HashSet;

use polytype::Type;

use crate::ast::TypeName;

use super::program::{GraphResourceRef, PhysicalProgram};
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
        for input in &entry.inputs {
            verify_physical_type(&input.ty, &entry.name)?;
        }
        for output in &entry.outputs {
            verify_physical_type(&output.ty, &entry.name)?;
        }
        for declaration in &entry.storage_bindings {
            verify_physical_type(&declaration.elem_ty, &entry.name)?;
        }
        for (ty, _) in &entry.params {
            verify_physical_type(ty, &entry.name)?;
        }
        verify_physical_type(&entry.return_ty, &entry.name)?;
        verify_physical_graph(&entry.graph, &entry.name, true)?;
    }
    for function in &program.functions {
        for (ty, _) in &function.params {
            verify_physical_type(ty, &function.name)?;
        }
        verify_physical_type(&function.return_ty, &function.name)?;
        verify_physical_graph(&function.graph, &function.name, false)?;
    }
    for region in program.regions.values() {
        for (ty, _) in &region.params {
            verify_physical_type(ty, &region.name)?;
        }
        verify_physical_type(&region.return_ty, &region.name)?;
        verify_physical_graph(&region.graph, &region.name, false)?;
    }
    Ok(())
}

fn verify_physical_type(ty: &Type<TypeName>, owner: &str) -> Result<(), String> {
    let Type::Constructed(name, arguments) = ty else {
        return Ok(());
    };
    match name {
        TypeName::Resource(resource) => {
            return Err(format!(
                "physical body `{owner}` still contains semantic type resource {resource:?}"
            ));
        }
        TypeName::Sum(variants) => {
            for (_, fields) in variants {
                for field in fields {
                    verify_physical_type(field, owner)?;
                }
            }
        }
        _ => {}
    }
    for argument in arguments {
        verify_physical_type(argument, owner)?;
    }
    Ok(())
}

fn verify_physical_graph(
    graph: &super::types::EGraph,
    owner: &str,
    require_scheduled: bool,
) -> Result<(), String> {
    for (_, node) in &graph.nodes {
        if let ENode::Pure { op, .. } = node {
            match op {
                super::types::PureOp::StorageView(crate::op::PureViewSource::Resource(resource))
                | super::types::PureOp::ResourceLen(resource) => {
                    return Err(format!(
                        "physical body `{owner}` still contains semantic resource {resource:?}"
                    ));
                }
                _ => {}
            }
        }
    }
    for ty in graph.types.values() {
        verify_physical_type(ty, owner)?;
    }
    for (_, block) in &graph.skeleton.blocks {
        for effect in &block.side_effects {
            match &effect.kind {
                SideEffectKind::Inst(crate::ssa::types::InstKind::Op { tag, .. }) => match tag {
                    crate::op::OpTag::StorageView(crate::op::PureViewSource::Resource(resource))
                    | crate::op::OpTag::ResourceLen(resource) => {
                        return Err(format!(
                            "physical body `{owner}` still contains semantic resource {resource:?}"
                        ));
                    }
                    _ => {}
                },
                SideEffectKind::Soac(soac) => {
                    if require_scheduled {
                        match soac {
                            EgirSoac::Filter {
                                state: FilterState::Semantic { .. },
                                ..
                            } => {
                                return Err(format!(
                                    "physical entry `{owner}` contains an unscheduled filter"
                                ));
                            }
                            EgirSoac::Seg {
                                kind: SegOpKind::SegRed { .. } | SegOpKind::SegScan { .. },
                                ..
                            } => {
                                return Err(format!(
                                    "physical entry `{owner}` contains an unresolved reduction/scan"
                                ));
                            }
                            _ => {}
                        }
                    }
                    let mut soac = soac.clone();
                    let mut error = None;
                    soac.visit_resource_refs_mut(|reference| {
                        if let GraphResourceRef::Resource(resource) = *reference {
                            error.get_or_insert_with(|| {
                                format!(
                                    "physical body `{owner}` still contains semantic resource {resource:?}"
                                )
                            });
                        }
                    });
                    soac.visit_types_mut(|ty| {
                        if error.is_none() {
                            error = verify_physical_type(ty, owner).err();
                        }
                    });
                    if let Some(error) = error {
                        return Err(error);
                    }
                }
                _ => {}
            }
        }
    }
    Ok(())
}
