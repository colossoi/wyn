use polytype::Type;

use crate::ast::TypeName;

use super::program::PhysicalProgram;
use super::types::{EgirSoac, FilterState, SegOpKind, SegPlacement, SideEffectKind};

pub fn check(program: &PhysicalProgram) -> Result<(), String> {
    for resource in &program.resources {
        if program.physical_resources.binding(resource.id).is_none() {
            return Err(format!("resource {:?} has no physical binding", resource.id));
        }
    }
    for entry in &program.entry_points {
        if !program.plan.contains_entry(&entry.name) {
            return Err(format!("physical entry `{}` is absent from the plan", entry.name));
        }
        for input in &entry.inputs {
            physical_type(&input.ty, &entry.name)?;
        }
        for output in &entry.outputs {
            physical_type(&output.ty, &entry.name)?;
        }
        for declaration in &entry.storage_bindings {
            physical_type(&declaration.elem_ty, &entry.name)?;
        }
        graph(&entry.graph, &entry.name, true)?;
    }
    for function in &program.functions {
        graph(&function.graph, &function.name, false)?;
    }
    for region in program.regions.values() {
        graph(&region.graph, &region.name, false)?;
    }
    Ok(())
}

fn physical_type(ty: &Type<TypeName>, owner: &str) -> Result<(), String> {
    let Type::Constructed(name, arguments) = ty else {
        return Ok(());
    };
    if matches!(name, TypeName::Resource(_)) {
        return Err(format!(
            "physical body `{owner}` retains a semantic resource type"
        ));
    }
    if let TypeName::Sum(variants) = name {
        for field in variants.iter().flat_map(|(_, fields)| fields) {
            physical_type(field, owner)?;
        }
    }
    for argument in arguments {
        physical_type(argument, owner)?;
    }
    Ok(())
}

fn graph(
    graph: &super::types::EGraph<super::program::PhysicalResourceRef>,
    owner: &str,
    entry: bool,
) -> Result<(), String> {
    for ty in graph.types.values() {
        physical_type(ty, owner)?;
    }
    for node in graph.nodes.values() {
        if matches!(
            node,
            super::types::ENode::Pure {
                op: super::types::PureOp::ResourceLen(_),
                ..
            }
        ) {
            return Err(format!("physical body `{owner}` retains ResourceLen"));
        }
    }
    for effect in graph.skeleton.blocks.values().flat_map(|block| &block.side_effects) {
        let SideEffectKind::Soac(soac) = &effect.kind else {
            continue;
        };
        match soac {
            EgirSoac::Filter {
                state: FilterState::Semantic { .. },
                ..
            } => return Err(format!("physical body `{owner}` retains an unscheduled filter")),
            EgirSoac::Seg {
                placement: SegPlacement::Kernel,
                kind: SegOpKind::SegRed { .. } | SegOpKind::SegScan { .. },
                ..
            } if entry => {
                return Err(format!(
                    "physical entry `{owner}` retains a pending segmented kernel"
                ));
            }
            _ => {}
        }
        let mut soac = soac.clone();
        let mut error = None;
        soac.visit_types_mut(|ty| {
            if error.is_none() {
                error = physical_type(ty, owner).err();
            }
        });
        if let Some(error) = error {
            return Err(error);
        }
    }
    Ok(())
}
