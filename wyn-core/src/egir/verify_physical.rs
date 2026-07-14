use polytype::Type;

use crate::ast::TypeName;

use super::program::{visit_type_names_mut, PhysicalEGraph, PhysicalProgram};
use super::types::SideEffectKind;

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
        graph(&entry.graph, &entry.name)?;
    }
    for function in &program.functions {
        graph(&function.graph, &function.name)?;
    }
    for region in program.regions.values() {
        graph(&region.graph, &region.name)?;
    }
    Ok(())
}

fn physical_type(ty: &Type<TypeName>, owner: &str) -> Result<(), String> {
    let mut ty = ty.clone();
    let mut semantic = false;
    visit_type_names_mut(&mut ty, |name| semantic |= matches!(name, TypeName::Resource(_)));
    if semantic {
        return Err(format!(
            "physical body `{owner}` retains a semantic resource type"
        ));
    }
    Ok(())
}

fn graph(graph: &PhysicalEGraph, owner: &str) -> Result<(), String> {
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
        let mut soac = soac.clone();
        let mut error = None;
        soac.for_each_type_mut(|ty| {
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
