#![deny(clippy::expect_used, clippy::panic, clippy::unreachable, clippy::unwrap_used)]

use polytype::Type;

use crate::ast::TypeName;
use crate::{BindingRef, LookupMap};

use super::ir::PlaceOp;
use super::physical_call_abi::CallableBoundary;
use super::program::{visit_type_names_mut, Func, PhysicalResourceTable};
use super::types::{
    EGraph, OperandType, ParameterId, Parameters, Physical, PlaceDestination, ResultDestination,
    SideEffectKind, SkeletonTerminator, SoacEffect, ValueKind,
};

pub fn check(
    program: &super::parallelize::Planned,
    _physical_resources: &PhysicalResourceTable,
) -> Result<(), String> {
    let boundaries = super::physical_call_abi::callable_boundaries(&program.functions, &program.externs);
    for entry in &program.entry_points {
        for ty in entry
            .inputs
            .iter()
            .map(|input| &input.ty)
            .chain(entry.outputs.iter().map(|output| &output.ty))
            .chain(entry.resource_declarations.iter().map(|declaration| &declaration.elem_ty))
        {
            physical_type(ty, &entry.name)?;
        }
        graph(&entry.graph, &entry.name, &boundaries)?;
    }
    for function in &program.functions {
        function_boundary(function)?;
        graph(&function.graph, &function.name, &boundaries)?;
    }
    for constant in &program.constants {
        physical_type(&constant.return_ty, &constant.name)?;
        graph(&constant.graph, &constant.name, &boundaries)?;
    }
    Ok(())
}

fn physical_type(ty: &Type<TypeName>, owner: &str) -> Result<(), String> {
    let mut ty = ty.clone();
    let mut semantic = false;
    visit_type_names_mut(&mut ty, |name| semantic |= matches!(name, TypeName::Resource(_)));
    (!semantic)
        .then_some(())
        .ok_or_else(|| format!("physical body `{owner}` retains a semantic resource type"))
}

fn parameter_bindings(
    graph: &EGraph<Physical>,
    parameters: &Parameters<BindingRef, Type<TypeName>>,
    owner: &str,
) -> Result<(), String> {
    let mut counts = LookupMap::<ParameterId, usize>::new();
    for (_, node) in graph.values() {
        let ValueKind::FuncParam { parameter } = node.kind() else {
            continue;
        };
        let declaration = parameters.get(*parameter).ok_or_else(|| {
            format!("physical body `{owner}` retains obsolete parameter node {parameter:?}")
        })?;
        if !matches!(
            declaration.representation(),
            OperandType::Value(_) | OperandType::View(_)
        ) || declaration.ty() != node.ty()
        {
            return Err(format!(
                "physical body `{owner}` value binding disagrees with parameter {parameter:?}"
            ));
        }
        *counts.entry(*parameter).or_default() += 1;
    }
    for (_, place) in graph.places() {
        let PlaceOp::Parameter { parameter } = place.op() else {
            continue;
        };
        let declaration = parameters.get(*parameter).ok_or_else(|| {
            format!("physical body `{owner}` retains obsolete place parameter {parameter:?}")
        })?;
        if !matches!(declaration.representation(), OperandType::Place(expected) if expected == place.ty()) {
            return Err(format!(
                "physical body `{owner}` place binding disagrees with parameter {parameter:?}"
            ));
        }
        *counts.entry(*parameter).or_default() += 1;
    }
    for parameter in parameters.ids() {
        let count = counts.get(&parameter).copied().unwrap_or(0);
        if count != 1 {
            return Err(format!(
                "physical body `{owner}` parameter {parameter:?} has {count} graph bindings"
            ));
        }
    }
    Ok(())
}

fn function_boundary(function: &Func<Physical>) -> Result<(), String> {
    let owner = function.name.as_str();
    for parameter in function.params().iter() {
        physical_type(parameter.ty(), owner)?;
    }
    physical_type(function.result().ty(), owner)?;
    parameter_bindings(&function.graph, function.params(), owner)?;

    let destinations = function.result().destination_parameters();
    let suffix_start = function.params().len().checked_sub(destinations.len()).ok_or_else(|| {
        format!("physical function `{owner}` has more result destinations than parameters")
    })?;
    let suffix = function.params().ids().skip(suffix_start).collect::<Vec<_>>();
    if suffix != destinations {
        return Err(format!(
            "physical function `{owner}` has misplaced destination parameters"
        ));
    }

    for (block, contents) in &function.graph.skeleton.blocks {
        let SkeletonTerminator::Return(binding) = &contents.term else {
            continue;
        };
        let Some(binding) = binding else {
            if function.result().destination_count() == 0 {
                continue;
            }
            return Err(format!(
                "physical function `{owner}` has an empty return in {block:?}"
            ));
        };
        let expected = function.result().destination_leaves_with_paths();
        let actual = binding.destination_leaves_with_paths();
        if expected.len() != actual.len() {
            return Err(format!(
                "physical function `{owner}` return in {block:?} has stale arity"
            ));
        }
        for ((expected_path, expected), (actual_path, actual)) in expected.into_iter().zip(actual) {
            let (expected_ty, expected_destination) = expected.parts();
            let (actual_ty, actual_destination) = actual.parts();
            let valid = expected_path == actual_path
                && expected_ty == actual_ty
                && match (expected_destination, actual_destination) {
                    (ResultDestination::ReturnValue(_), ResultDestination::ReturnValue(value)) => {
                        function.graph.values().get(*value).is_some_and(|node| node.ty() == expected_ty)
                    }
                    (
                        ResultDestination::Place(PlaceDestination::Fixed(parameter)),
                        ResultDestination::Place(PlaceDestination::Fixed(place)),
                    ) => matches!(
                        function.graph.places().get(*place).map(|place| place.op()),
                        Some(PlaceOp::Parameter { parameter: actual }) if actual == parameter
                    ),
                    _ => false,
                };
            if !valid {
                return Err(format!(
                    "physical function `{owner}` return in {block:?} disagrees with its ABI"
                ));
            }
        }
    }
    Ok(())
}

fn graph(
    graph: &EGraph<Physical>,
    owner: &str,
    boundaries: &LookupMap<crate::FunctionId, CallableBoundary>,
) -> Result<(), String> {
    graph
        .skeleton
        .verify_branch_arities()
        .map_err(|error| format!("physical body `{owner}` has invalid control flow: {error}"))?;
    check_graph_flow(graph, owner)?;
    for node in graph.values().values() {
        physical_type(node.ty(), owner)?;
        if matches!(
            node.kind(),
            ValueKind::Pure {
                op: super::types::PureOp::ResourceLen(_),
                ..
            }
        ) {
            return Err(format!("physical body `{owner}` retains ResourceLen"));
        }
    }
    for effect in graph.skeleton.blocks.values().flat_map(|block| &block.side_effects) {
        let SideEffectKind::Soac(SoacEffect(_, soac)) = effect.kind() else {
            continue;
        };
        let mut soac = soac.clone();
        let mut type_error = None;
        soac.for_each_type_mut(|ty| {
            type_error = type_error.take().or_else(|| physical_type(ty, owner).err())
        });
        if let Some(error) = type_error {
            return Err(error);
        }
        let inputs = soac.input_types_mut();
        if inputs.len() != effect.operands().len() {
            return Err(format!(
                "physical body `{owner}` has {} SOAC inputs for {} operands",
                inputs.len(),
                effect.operands().len()
            ));
        }
        for (index, (input, operand)) in inputs.iter().zip(effect.operands()).enumerate() {
            if operand.value().is_some_and(|value| graph.value(value).ty() != &input.array) {
                return Err(format!(
                    "physical body `{owner}` has stale SOAC input metadata at {index}"
                ));
            }
        }
    }
    for (site, _) in graph.side_effect_index().calls() {
        let call = graph.call(site);
        let boundary = boundaries.get(&call.callee()).ok_or_else(|| {
            format!(
                "physical body `{owner}` calls {:?} without a boundary",
                call.callee()
            )
        })?;
        graph
            .verify_call_boundary(site, &boundary.0, &boundary.1, boundary.2)
            .map_err(|error| format!("physical body `{owner}`: {error}"))?;
    }
    Ok(())
}

pub(crate) fn check_graph_flow(graph: &EGraph<Physical>, owner: &str) -> Result<(), String> {
    for (block, contents) in &graph.skeleton.blocks {
        for (slot, parameter) in contents.params.iter().enumerate() {
            let ty = graph.value(parameter.value()).ty();
            if super::physical_flow::type_contains_materialized_flow(ty) {
                return Err(format!(
                    "physical body `{owner}` has materialized block parameter {slot} in {block:?}: {ty:?}"
                ));
            }
        }
    }
    Ok(())
}
