//! Adapt late SSA constants to the same evaluator used by TLC. Inlining can
//! expose literal operands after source-level partial evaluation has finished.

use super::ir::Substitutions;
use super::types::{ConstantValue, FuncBody, InstKind, ValueId, ValueRef, WynFunction};
use crate::builtins::{by_id, catalog, select, Purity};
use crate::constant_eval::{self, Constant};
use crate::op::OpTag;
use crate::scalar_eval::{self, Scalar};
use crate::types::{bool_type, Type, TypeExt, TypeName};
use crate::{BindingRef, FunctionId};
use std::collections::HashMap;
use wyn_graph::topo_sort_by_dependencies;

#[cfg(test)]
#[path = "constant_folding_tests.rs"]
mod tests;

pub(super) fn fold(body: &mut FuncBody) {
    let function = &mut body.inner;
    // Block parameters break loop cycles. Include floating definitions too:
    // physical block order need not be dependency order after inlining.
    let Ok(order) = topo_sort_by_dependencies(function.insts.keys(), |id, out| {
        out.extend(
            function.insts[id].data.ssa_uses().into_iter().filter_map(|v| function.inst_of_value(v)),
        );
    }) else {
        return;
    };
    let mut constants = HashMap::new();
    let mut replacements = Substitutions::default();
    for id in order {
        function.insts[id].data.substitute_values(&mut |value| replacements.resolve(value));
        let node = &function.insts[id];
        let (Some(result), InstKind::Op { tag, operands }) = (node.result, &node.data) else {
            continue;
        };
        let ty = &function.values[result].ty;
        if matches!(tag, OpTag::Intrinsic { id, overload_idx: 0 } if *id == catalog().known().select)
            && select::supported_type(ty)
        {
            if let Some(choice) = select::Selection::from_operands(operands) {
                let chosen = if choice.yes == choice.no {
                    Some(choice.yes)
                } else if let Some(Constant::Bool(condition)) = operand(choice.condition, &constants) {
                    Some(if condition { choice.yes } else { choice.no })
                } else {
                    None
                };
                if let Some(chosen) = chosen.filter(|_| {
                    operand_type(function, choice.yes) == *ty
                        && operand_type(function, choice.no) == *ty
                        && operand_type(function, choice.condition) == bool_type()
                }) {
                    // Operand producers remain in the instruction graph; DCE
                    // separately decides whether their evaluation is discardable.
                    replacements.insert(result, chosen);
                    function.insts.remove(id);
                    continue;
                }
            }
        }
        let Some(mut value) = evaluate(function, tag, operands, ty, &constants) else {
            continue;
        };
        match &value {
            Constant::Vector(lanes) => {
                let Some(element) = ty.elem_type() else { continue };
                let Some(operands) =
                    lanes.iter().map(|v| inline_scalar(*v, element)).collect::<Option<Vec<_>>>()
                else {
                    continue;
                };
                let Some(normalized) = operands
                    .iter()
                    .map(|v| operand(*v, &constants)?.as_scalar())
                    .collect::<Option<Vec<_>>>()
                else {
                    continue;
                };
                function.insts[id].data = InstKind::Op {
                    tag: OpTag::Vector(lanes.len()),
                    operands,
                };
                value = Constant::Vector(normalized);
            }
            scalar => {
                let Some(replacement) = scalar.as_scalar().and_then(|v| inline_scalar(v, ty)) else {
                    continue;
                };
                // Cache the typed bit pattern, including unsigned literals
                // whose spelling uses a signed representation.
                let Some(normalized) = operand(replacement, &constants) else {
                    continue;
                };
                value = normalized;
                replacements.insert(result, replacement);
                function.insts.remove(id);
            }
        }
        constants.insert(result, value);
    }
    replacements.finish(function);
    for block in function.blocks.values_mut() {
        block.insts.retain(|id| function.insts.contains_key(*id));
    }
    // Folding includes eager selects introduced by if-conversion. Their results
    // can expose constant conditions in otherwise unconvertible selections.
    super::ir::fold_constant_selections(function);
}

fn operand(value: ValueRef, constants: &HashMap<ValueId, Constant>) -> Option<Constant> {
    Some(match value {
        ValueRef::Ssa(id) => return constants.get(&id).cloned(),
        ValueRef::Const(ConstantValue::I32(v)) => Constant::Int(i64::from(v)),
        ValueRef::Const(ConstantValue::U32(v)) => Constant::Int(i64::from(v)),
        ValueRef::Const(ConstantValue::F32(bits)) => Constant::Float(f32::from_bits(bits) as f64),
        ValueRef::Const(ConstantValue::Bool(v)) => Constant::Bool(v),
    })
}

fn operand_type(function: &WynFunction, value: ValueRef) -> Type {
    let name = match value {
        ValueRef::Ssa(id) => return function.values[id].ty.clone(),
        ValueRef::Const(ConstantValue::I32(_)) => TypeName::Int(32),
        ValueRef::Const(ConstantValue::U32(_)) => TypeName::UInt(32),
        ValueRef::Const(ConstantValue::F32(_)) => TypeName::Float(32),
        ValueRef::Const(ConstantValue::Bool(_)) => TypeName::Bool,
    };
    Type::Constructed(name, vec![])
}

fn evaluate(
    function: &WynFunction,
    tag: &OpTag<BindingRef, FunctionId>,
    operands: &[ValueRef],
    ty: &Type,
    constants: &HashMap<ValueId, Constant>,
) -> Option<Constant> {
    match tag {
        OpTag::Int(text) | OpTag::Uint(text) => Some(Constant::Int(text.parse().ok()?)),
        OpTag::Float(text) if matches!(ty, Type::Constructed(TypeName::Float(32), _)) => {
            Some(Constant::Float(text.parse::<f32>().ok()? as f64))
        }
        OpTag::Bool(value) => Some(Constant::Bool(*value)),
        OpTag::Vector(_) => operands
            .iter()
            .map(|v| operand(*v, constants)?.as_scalar())
            .collect::<Option<Vec<_>>>()
            .map(Constant::Vector),
        OpTag::Project { index } => {
            let [value] = operands else { return None };
            let Constant::Vector(lanes) = operand(*value, constants)? else {
                return None;
            };
            lanes.get(*index as usize).copied().map(Constant::from_scalar)
        }
        OpTag::BinOp(op) => {
            let [a, b] = operands else { return None };
            constant_eval::binary(
                *op,
                &operand(*a, constants)?,
                &operand(*b, constants)?,
                &operand_type(function, *a),
            )
        }
        OpTag::UnaryOp(op) => {
            let [value] = operands else { return None };
            scalar_eval::unary(
                *op,
                operand(*value, constants)?.as_scalar()?,
                &operand_type(function, *value),
            )
            .map(Constant::from_scalar)
        }
        OpTag::Intrinsic { id, overload_idx } => {
            let builtin = by_id(*id);
            if builtin.raw.purity != Purity::Pure {
                return None;
            }
            let lowering = &builtin.overloads().get(*overload_idx)?.lowering;
            let args = operands
                .iter()
                .map(|v| Some((operand(*v, constants)?, operand_type(function, *v))))
                .collect::<Option<Vec<_>>>()?;
            constant_eval::builtin(lowering, &args, ty)
        }
        _ => None,
    }
}

fn inline_scalar(value: Scalar, ty: &Type) -> Option<ValueRef> {
    Some(ValueRef::Const(match (value, ty) {
        (Scalar::Int(v), Type::Constructed(TypeName::Int(32), _)) => ConstantValue::I32(v as i32),
        (Scalar::Int(v), Type::Constructed(TypeName::UInt(32), _)) => ConstantValue::U32(v as u32),
        (Scalar::Float(v), Type::Constructed(TypeName::Float(32), _)) if (v as f32).is_finite() => {
            ConstantValue::from_f32(v as f32)
        }
        (Scalar::Bool(v), Type::Constructed(TypeName::Bool, _)) => ConstantValue::Bool(v),
        _ => return None,
    }))
}
