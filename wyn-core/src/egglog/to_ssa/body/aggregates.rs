//! Identities for aggregates created after scalar selection. No evaluation moves:
//! callers have already emitted every argument, including effectful arguments.
use super::{Type, TypeExt, TypeName, Typed};
use crate::op::OpTag;
use crate::ssa::types::{ConstantValue, InstKind, ValueRef, WynFunction};
use crate::{BindingRef, FunctionId};

fn definition(function: &WynFunction, value: ValueRef) -> Option<&InstKind> {
    let ValueRef::Ssa(value) = value else { return None };
    Some(&function.insts[function.inst_of_value(value)?].data)
}

fn arity(ty: &Type) -> Option<usize> {
    match ty {
        Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields) => Some(fields.len()),
        _ => ty.vec_size(),
    }
}

pub(super) fn forward(
    function: &WynFunction,
    tag: &OpTag<BindingRef, FunctionId>,
    args: &[Typed],
    ty: &Type,
) -> Option<Typed> {
    let value = match tag {
        OpTag::Project { index } if args.len() == 1 => {
            let InstKind::Op {
                tag: OpTag::Tuple(n) | OpTag::Vector(n),
                operands,
            } = definition(function, args[0].value)?
            else {
                return None;
            };
            if arity(&args[0].ty) != Some(*n) || operands.len() != *n {
                return None;
            }
            *operands.get(*index as usize)?
        }
        OpTag::Tuple(n) | OpTag::Vector(n) if !args.is_empty() && args.len() == *n => {
            if arity(ty) != Some(*n) {
                return None;
            }
            let mut original = None;
            for (i, arg) in args.iter().enumerate() {
                let InstKind::Op {
                    tag: OpTag::Project { index },
                    operands,
                } = definition(function, arg.value)?
                else {
                    return None;
                };
                let [source] = operands.as_slice() else {
                    return None;
                };
                if *index as usize != i || original.is_some_and(|value| value != *source) {
                    return None;
                }
                original = Some(*source);
            }
            original?
        }
        _ => return None,
    };
    let value_ty = match value {
        ValueRef::Ssa(value) => function.value_type(value).clone(),
        ValueRef::Const(constant) => Type::Constructed(
            match constant {
                ConstantValue::I32(_) => TypeName::Int(32),
                ConstantValue::U32(_) => TypeName::UInt(32),
                ConstantValue::F32(_) => TypeName::Float(32),
                ConstantValue::Bool(_) => TypeName::Bool,
            },
            vec![],
        ),
    };
    (value_ty == *ty).then(|| Typed {
        value,
        ty: ty.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ssa::builder::FuncBuilder;
    use crate::types::{i32, tuple, vec};

    fn typed(function: &WynFunction, value: crate::ssa::types::ValueId) -> Typed {
        Typed {
            value: value.into(),
            ty: function.value_type(value).clone(),
        }
    }

    #[test]
    fn forwards_generated_projections_and_complete_reconstructions() {
        for aggregate in [tuple(vec![i32(), i32()]), vec(2, i32())] {
            let mut b = FuncBuilder::new(vec![(aggregate.clone(), "input".into())], i32());
            let source = b.get_param(0);
            let mut fields = Vec::new();
            for index in 0..2 {
                let value = b
                    .push_inst(
                        InstKind::Op {
                            tag: OpTag::Project { index },
                            operands: vec![source.into()],
                        },
                        i32(),
                    )
                    .unwrap();
                fields.push(typed(b.func(), value));
            }
            let tag = if aggregate.is_vec() { OpTag::Vector(2) } else { OpTag::Tuple(2) };
            assert_eq!(
                forward(b.func(), &tag, &fields, &aggregate).unwrap().value,
                source.into()
            );
            fields.swap(0, 1);
            assert!(forward(b.func(), &tag, &fields, &aggregate).is_none());
            let construct = b
                .push_inst(
                    InstKind::Op {
                        tag,
                        operands: fields.iter().map(|field| field.value).collect(),
                    },
                    aggregate,
                )
                .unwrap();
            let args = [typed(b.func(), construct)];
            assert_eq!(
                forward(b.func(), &OpTag::Project { index: 0 }, &args, &i32()).unwrap().value,
                fields[0].value
            );
            assert!(forward(
                b.func(),
                &OpTag::Project { index: 0 },
                &args,
                &crate::types::bool_type()
            )
            .is_none());
        }
    }

    #[test]
    fn preserves_field_subsets_and_mixed_sources() {
        let wide = tuple(vec![i32(), i32(), i32()]);
        let narrow = tuple(vec![i32(), i32()]);
        let mut b = FuncBuilder::new(vec![(wide.clone(), "a".into()), (wide, "b".into())], i32());
        let a = b.get_param(0);
        let other = b.get_param(1);
        let mut fields = Vec::new();
        for (index, source) in [(0, a), (1, a), (1, other)] {
            let value = b
                .push_inst(
                    InstKind::Op {
                        tag: OpTag::Project { index },
                        operands: vec![source.into()],
                    },
                    i32(),
                )
                .unwrap();
            fields.push(typed(b.func(), value));
        }
        assert!(forward(b.func(), &OpTag::Tuple(2), &fields[..2], &narrow).is_none());
        assert!(forward(
            b.func(),
            &OpTag::Tuple(2),
            &[fields[0].clone(), fields[2].clone()],
            &narrow
        )
        .is_none());
    }
}
