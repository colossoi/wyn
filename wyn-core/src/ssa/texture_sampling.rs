//! Publish filtering requirements from the final shader's texture uses.

use crate::ast::TypeName;
use crate::builtins;
use crate::host::{Binding, Pipeline, TextureSampleType};
use crate::op::OpTag;
use crate::ssa::ir::{Terminator, ValueDef, ValueRef};
use crate::ssa::types::{stage, FuncBody, InstKind};
use crate::{BindingRef, FunctionId, LookupMap, LookupSet};
use polytype::Type;

pub(super) fn publish_texture_sampling(program: &mut stage::Reachable) {
    let mut sampled = program
        .functions
        .iter()
        .map(|function| (function.id, LookupSet::new()))
        .collect::<LookupMap<_, _>>();
    loop {
        let mut changed = false;
        for function in &program.functions {
            let parameters = if function.linkage_name.is_some() {
                texture_parameters(&function.body)
            } else {
                sampled_parameters(&function.body, &sampled)
            };
            changed |= sampled[&function.id] != parameters;
            sampled.insert(function.id, parameters);
        }
        if !changed {
            break;
        }
    }

    let entries = program
        .entry_points
        .iter()
        .map(|entry| {
            let bindings = sampled_parameters(&entry.body, &sampled)
                .into_iter()
                .flat_map(|parameter| &entry.parameter_inputs[parameter])
                .filter_map(|&input| entry.inputs[input].texture_binding())
                .collect::<LookupSet<_>>();
            (entry.name.as_str(), bindings)
        })
        .collect::<LookupMap<_, _>>();
    for pipeline in &mut program.global_context.pipeline.pipelines {
        let (bindings, stages) = match pipeline {
            Pipeline::Compute(compute) => (
                &mut compute.bindings,
                compute.stages.iter().map(|stage| stage.entry_point.as_str()).collect::<Vec<_>>(),
            ),
            Pipeline::Graphics(graphics) => (
                &mut graphics.bindings,
                graphics.stages.iter().map(|stage| stage.entry_point.as_str()).collect::<Vec<_>>(),
            ),
        };
        for binding in bindings {
            let Binding::Texture {
                set,
                binding,
                sample_type: TextureSampleType::Float { filterable },
                ..
            } = binding
            else {
                continue;
            };
            let slot = BindingRef {
                set: *set,
                binding: *binding,
            };
            // A shared layout must support sampling in any of its stages.
            *filterable = stages
                .iter()
                .any(|stage| entries.get(stage).is_none_or(|bindings| bindings.contains(&slot)));
        }
    }
}

fn contains_texture(ty: &Type<TypeName>) -> bool {
    match ty {
        Type::Constructed(TypeName::Texture2D, _) => true,
        Type::Constructed(_, arguments) => arguments.iter().any(contains_texture),
        Type::Variable(_) => false,
    }
}

fn texture_parameters(body: &FuncBody) -> LookupSet<usize> {
    body.inner
        .params
        .iter()
        .enumerate()
        .filter_map(|(index, &value)| contains_texture(&body.inner.values[value].ty).then_some(index))
        .collect()
}

/// Follow opaque texture values back to parameters, including helper calls and
/// control-flow joins. Numeric dependencies (such as UVs loaded from a second
/// texture) do not impose filtering requirements on their source textures.
fn sampled_parameters(
    body: &FuncBody,
    functions: &LookupMap<FunctionId, LookupSet<usize>>,
) -> LookupSet<usize> {
    let mut pending = Vec::new();
    for (_, inst) in &body.inner.insts {
        let InstKind::Op { tag, operands } = &inst.data else {
            continue;
        };
        match tag {
            OpTag::Intrinsic { id, .. } if *id == builtins::catalog().known().texture_sample => {
                pending.extend(operands.first().copied().and_then(ValueRef::as_ssa));
            }
            OpTag::Call(function) => {
                for (index, operand) in operands.iter().enumerate() {
                    if functions.get(function).is_none_or(|parameters| parameters.contains(&index)) {
                        pending.extend(operand.as_ssa());
                    }
                }
            }
            _ => {}
        }
    }

    let mut parameters = LookupSet::new();
    let mut visited = LookupSet::new();
    while let Some(value) = pending.pop() {
        if !visited.insert(value) || !contains_texture(&body.inner.values[value].ty) {
            continue;
        }
        match body.inner.values[value].def {
            ValueDef::FunctionParam { index } => {
                parameters.insert(index);
            }
            ValueDef::Inst { inst } => {
                let instruction = &body.inner.insts[inst].data;
                if matches!(instruction, InstKind::Load { .. }) {
                    // Addressable opaque values may alias any texture parameter.
                    parameters.extend(texture_parameters(body));
                } else {
                    pending.extend(instruction.ssa_uses());
                }
            }
            ValueDef::Param { block, index } => {
                for (_, predecessor) in &body.inner.blocks {
                    match &predecessor.term {
                        Terminator::Branch { target, args } if *target == block => {
                            pending.extend(args[index].as_ssa());
                        }
                        Terminator::CondBranch {
                            then_target,
                            then_args,
                            else_target,
                            else_args,
                            ..
                        } => {
                            if *then_target == block {
                                pending.extend(then_args[index].as_ssa());
                            }
                            if *else_target == block {
                                pending.extend(else_args[index].as_ssa());
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    parameters
}
