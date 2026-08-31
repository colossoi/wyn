//! Optional promotion of constant materializations to module-level SSA data.

use crate::ast::TypeName;
use crate::op::{AddressableConstantId, OpTag};
use crate::ssa::framework::ValueDef;
use crate::ssa::types::{
    AddressableConstant, AddressableConstantKind, AddressableConstantValue, ConstantValue, FuncBody,
    InstKind, Program, ValueRef,
};
use polytype::Type;

/// Replace `Materialize` over a constant-expression tree with an operand-free
/// `AddressableConstant` reference. The original tree then becomes eligible
/// for ordinary SSA dead-code elimination.
pub fn promote_addressable_constants<Tag, GlobalContext>(program: &mut Program<Tag, GlobalContext>) {
    let table = &mut program.addressable_constants;
    for function in &mut program.functions {
        promote_body(&mut function.body, table);
    }
    for entry in &mut program.entry_points {
        promote_body(&mut entry.body, table);
    }
    for constant in &mut program.constants {
        promote_body(&mut constant.body, table);
    }
}

fn promote_body(body: &mut FuncBody, table: &mut Vec<AddressableConstant>) {
    let candidates = body
        .inner
        .insts
        .iter()
        .filter_map(|(instruction, node)| match &node.data {
            InstKind::Op {
                tag: OpTag::Materialize,
                operands,
            } => Some((instruction, *operands.first()?)),
            _ => None,
        })
        .collect::<Vec<_>>();

    for (instruction, operand) in candidates {
        let Some(value) = constant_tree(body, operand) else {
            continue;
        };
        let id = table
            .iter()
            .find(|constant| constant.value == value)
            .map(|constant| constant.id)
            .unwrap_or_else(|| {
                let id = AddressableConstantId(table.len() as u32);
                table.push(AddressableConstant { id, value });
                id
            });
        if let Some(node) = body.inner.insts.get_mut(instruction) {
            node.data = InstKind::Op {
                tag: OpTag::AddressableConstant(id),
                operands: Vec::new(),
            };
        }
    }
}

fn constant_tree(body: &FuncBody, value: ValueRef) -> Option<AddressableConstantValue> {
    match value {
        ValueRef::Const(value) => Some(AddressableConstantValue {
            ty: scalar_constant_type(value),
            kind: AddressableConstantKind::Scalar(value),
        }),
        ValueRef::Ssa(value) => {
            let ty = body.get_value_type(value).clone();
            let info = body.inner.values.get(value)?;
            let ValueDef::Inst { inst } = info.def else {
                return None;
            };
            let node = body.inner.insts.get(inst)?;
            let InstKind::Op { tag, operands } = &node.data else {
                return None;
            };
            let kind = match tag {
                OpTag::Int(value) => AddressableConstantKind::Signed(value.clone()),
                OpTag::Uint(value) => AddressableConstantKind::Unsigned(value.clone()),
                OpTag::Float(value) => AddressableConstantKind::Float(value.clone()),
                OpTag::Bool(value) => AddressableConstantKind::Bool(*value),
                OpTag::Tuple(_) | OpTag::Vector(_) | OpTag::ArrayLit(_) | OpTag::Matrix { .. } => {
                    AddressableConstantKind::Composite(
                        operands
                            .iter()
                            .map(|operand| constant_tree(body, *operand))
                            .collect::<Option<Vec<_>>>()?,
                    )
                }
                _ => return None,
            };
            Some(AddressableConstantValue { ty, kind })
        }
    }
}

fn scalar_constant_type(value: ConstantValue) -> Type<TypeName> {
    let name = match value {
        ConstantValue::I32(_) => TypeName::Int(32),
        ConstantValue::U32(_) => TypeName::UInt(32),
        ConstantValue::F32(_) => TypeName::Float(32),
        ConstantValue::Bool(_) => TypeName::Bool,
    };
    Type::Constructed(name, Vec::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ssa::builder::FuncBuilder;
    use crate::ssa::eliminate_dead_pure_instructions;

    #[test]
    fn promotion_precedes_dce_and_detaches_the_constructor_tree() {
        let f32_ty = Type::Constructed(TypeName::Float(32), vec![]);
        let vec2_ty = Type::Constructed(
            TypeName::Vec,
            vec![f32_ty, Type::Constructed(TypeName::Size(2), vec![])],
        );
        let mut builder = FuncBuilder::new(Vec::new(), vec2_ty.clone());
        let vector = builder
            .push_inst(
                InstKind::Op {
                    tag: OpTag::Vector(2),
                    operands: vec![
                        ValueRef::Const(ConstantValue::from_f32(1.0)),
                        ValueRef::Const(ConstantValue::from_f32(2.0)),
                    ],
                },
                vec2_ty.clone(),
            )
            .unwrap();
        let materialized = builder
            .push_inst(
                InstKind::Op {
                    tag: OpTag::Materialize,
                    operands: vec![vector.into()],
                },
                vec2_ty,
            )
            .unwrap();
        builder.terminate(crate::ssa::types::Terminator::Return(Some(materialized.into()))).unwrap();
        let mut body = builder.finish().unwrap();
        let mut table = Vec::new();

        promote_body(&mut body, &mut table);
        eliminate_dead_pure_instructions(&mut body);

        assert_eq!(table.len(), 1);
        assert_eq!(
            body.num_insts(),
            1,
            "the detached vector constructor should be dead"
        );
        assert!(body.inner.insts.values().any(|node| matches!(
            node.data,
            InstKind::Op {
                tag: OpTag::AddressableConstant(_),
                ref operands,
            } if operands.is_empty()
        )));
    }
}
