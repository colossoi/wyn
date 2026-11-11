use super::{builder_error, error, Body, OptimizeError, Typed};
use crate::builtins::{catalog, select};
use crate::op::{BinaryOperator, OpTag, PureViewSource};
use crate::ssa::ir::ValueDef;
use crate::ssa::types::ConstantValue;
use crate::ssa::types::ValueRef;
use crate::ssa::types::{InstKind, PlaceId};
use crate::types::{self, bool_type, strip_existentials, Type, TypeExt, TypeName};

impl Body<'_, '_, '_> {
    pub(in crate::egglog::to_ssa) fn field(
        &mut self,
        value: Typed,
        index: usize,
    ) -> Result<Typed, OptimizeError> {
        let Some(ty) = (match &value.ty {
            Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields) => {
                fields.get(index).cloned()
            }
            Type::Constructed(TypeName::Vec, fields) => fields.first().cloned(),
            _ if index == 1 && value.ty.array_variant().is_some_and(types::is_array_variant_bounded) => {
                Some(types::i32())
            }
            _ => None,
        }) else {
            return Err(error(format!("invalid projection {index} of {:?}", value.ty)));
        };
        // Keep physical resource identity when projecting an emitted aggregate.
        // A shared descriptor's storage is not encoded in its source-level type.
        if let ValueRef::Ssa(id) = value.value {
            let function = self.builder.func();
            if let ValueDef::Inst { inst } = function.values[id].def {
                if let InstKind::Op {
                    tag: OpTag::Tuple,
                    operands,
                } = &function.insts[inst].data
                {
                    if let Some(value) = operands.get(index) {
                        return Ok(Typed {
                            value: value.clone(),
                            ty,
                        });
                    }
                }
            }
        }
        self.op(OpTag::Project { index: index as u32 }, vec![value], ty)
    }
    pub(in crate::egglog::to_ssa) fn index_place(
        &mut self,
        array: Typed,
        index: Typed,
    ) -> Result<(PlaceId, Type), OptimizeError> {
        let Some(ty) = array.ty.elem_type().cloned() else {
            return Err(error("indexed value has no element type"));
        };
        let place = self.builder.new_place(ty.clone());
        if let Some(&array) = self.local_arrays.get(&array.value) {
            self.builder
                .push_void_inst(InstKind::PlaceIndex {
                    place: array,
                    index: index.value,
                    result: place,
                })
                .map_err(builder_error)?;
            return Ok((place, ty));
        }
        self.builder
            .push_void_inst(InstKind::ViewIndex {
                view: array.value,
                index: index.value,
                result: place,
            })
            .map_err(builder_error)?;
        Ok((place, ty))
    }
    pub(in crate::egglog::to_ssa) fn cast(
        &mut self,
        value: Typed,
        ty: &Type,
    ) -> Result<Typed, OptimizeError> {
        let ty = strip_existentials(ty);
        if value.ty == *ty {
            return Ok(value);
        }
        if matches!(ty, Type::Constructed(TypeName::Int(_) | TypeName::UInt(_), _)) {
            if let Some(number) = self.integer_literal(&value) {
                return self.literal(number, ty);
            }
        }
        if let (
            Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), targets),
            Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), sources),
        ) = (ty, &value.ty)
        {
            if targets.len() != sources.len() {
                return Err(error("physical product conversion arity mismatch"));
            }
            let fields = targets
                .iter()
                .enumerate()
                .map(|(i, ty)| {
                    let field = self.field(value.clone(), i)?;
                    self.cast(field, ty)
                })
                .collect::<Result<Vec<_>, _>>()?;
            return self.op(OpTag::Tuple, fields, ty.clone());
        }
        if *ty == bool_type() {
            let zero = self.literal(0, &value.ty)?;
            return self.binary(BinaryOperator::NotEqual, value, zero);
        }
        if value.ty == bool_type() {
            let one = self.cast(Self::number(1), ty)?;
            let zero = self.cast(Self::number(0), ty)?;
            return self.select(value, one, zero);
        }
        let (Type::Constructed(target, _), Type::Constructed(source, _)) = (ty, &value.ty) else {
            return Err(error("unsupported conversion"));
        };
        let Some(id) = catalog().conversion(target, source) else {
            return Err(error(format!(
                "unsupported physical conversion from {:?} to {ty:?}",
                value.ty
            )));
        };
        self.op(OpTag::Intrinsic { id, overload_idx: 0 }, vec![value], ty.clone())
    }
    pub(in crate::egglog::to_ssa) fn binary(
        &mut self,
        op: BinaryOperator,
        a: Typed,
        b: Typed,
    ) -> Result<Typed, OptimizeError> {
        let b = self.cast(b, &a.ty)?;
        let ty = if matches!(
            op,
            BinaryOperator::Equal
                | BinaryOperator::NotEqual
                | BinaryOperator::Less
                | BinaryOperator::LessEqual
                | BinaryOperator::Greater
                | BinaryOperator::GreaterEqual
                | BinaryOperator::LogicalAnd
                | BinaryOperator::LogicalOr
        ) {
            bool_type()
        } else {
            a.ty.clone()
        };
        self.op(OpTag::BinOp(op), vec![a, b], ty)
    }
    pub(in crate::egglog::to_ssa) fn tuple(&mut self, values: Vec<Typed>) -> Result<Typed, OptimizeError> {
        let ty = types::tuple(values.iter().map(|value| value.ty.clone()).collect());
        self.op(OpTag::Tuple, values, ty)
    }
    pub fn number(n: u32) -> Typed {
        Typed {
            value: ValueRef::Const(ConstantValue::U32(n)),
            ty: Type::Constructed(TypeName::UInt(32), vec![]),
        }
    }
    pub fn local_id(&mut self) -> Result<Typed, OptimizeError> {
        if let Some(subgroup) = &self.subgroup {
            return Ok(subgroup.lane.clone());
        }
        self.op(
            OpTag::Intrinsic {
                id: catalog().known().local_id,
                overload_idx: 0,
            },
            vec![],
            Type::Constructed(TypeName::UInt(32), vec![]),
        )
    }
    pub(in crate::egglog::to_ssa) fn select(
        &mut self,
        c: Typed,
        a: Typed,
        b: Typed,
    ) -> Result<Typed, OptimizeError> {
        if a.ty != b.ty || !select::supported_type(&a.ty) {
            return Err(error("select requires matching scalar or vector types"));
        }
        let ty = a.ty.clone();
        self.op(
            OpTag::Intrinsic {
                id: catalog().known().select,
                overload_idx: 0,
            },
            vec![b, a, c],
            ty,
        )
    }
    pub(in crate::egglog::to_ssa) fn literal(
        &mut self,
        bits: i64,
        ty: &Type,
    ) -> Result<Typed, OptimizeError> {
        let tag = match strip_existentials(ty) {
            Type::Constructed(TypeName::UInt(_), _) => OpTag::Uint((bits as u64).to_string()),
            Type::Constructed(TypeName::Int(_), _) => OpTag::Int(bits.to_string()),
            Type::Constructed(TypeName::Bool, _) => OpTag::Bool(bits != 0),
            Type::Constructed(TypeName::Float(64), _) => {
                OpTag::Float(f64::from_bits(bits as u64).to_string())
            }
            Type::Constructed(TypeName::Float(_), _) => {
                let bits = u32::try_from(bits).map_err(|_| error("invalid folded float bits"))?;
                if matches!(ty, Type::Constructed(TypeName::Float(32), _)) {
                    return Ok(Typed {
                        value: ValueRef::Const(ConstantValue::F32(bits)),
                        ty: ty.clone(),
                    });
                }
                OpTag::Float(f32::from_bits(bits).to_string())
            }
            _ => return Err(error("unsupported scalar literal type")),
        };
        self.op(tag, vec![], ty.clone())
    }
}

impl Body<'_, '_, '_> {
    /// Allocate mutable local storage; reads used as values are explicit snapshots.
    pub(super) fn local_array(&mut self, ty: Type) -> Result<Typed, OptimizeError> {
        let place = self.builder.new_place(ty.clone());
        // Fixed private blocks outlive the control-flow arm that produces a view.
        let entry = self.builder.entry();
        self.builder.func_mut().append_void_inst(entry, InstKind::Alloca { result: place });
        let value = self.builder.func_mut().append_inst(entry, InstKind::Load { place }, ty.clone());
        let array = Typed {
            value: value.into(),
            ty,
        };
        self.local_arrays.insert(array.value, place);
        Ok(array)
    }
}

impl Body<'_, '_, '_> {
    pub fn index(&mut self, array: Typed, index: Typed) -> Result<Typed, OptimizeError> {
        self.storage_access(&array.ty, crate::ResourceAccess::Read);
        let (place, ty) = self.index_place(array, index)?;
        let value = self.builder.push_inst(InstKind::Load { place }, ty.clone()).map_err(builder_error)?;
        Ok(Typed {
            value: value.into(),
            ty,
        })
    }

    pub fn store(&mut self, array: Typed, index: Typed, value: Typed) -> Result<(), OptimizeError> {
        self.storage_access(&array.ty, crate::ResourceAccess::Write);
        let (place, ty) = self.index_place(array, index)?;
        let value = self.cast(value, &ty)?;
        self.builder
            .push_void_inst(InstKind::Store {
                place,
                value: value.value,
            })
            .map_err(builder_error)?;
        Ok(())
    }

    pub fn workgroup_array(&mut self, count: u32, element: &Type) -> Result<Typed, OptimizeError> {
        let id = self.next_workgroup_array;
        self.next_workgroup_array += 1;
        self.op(
            OpTag::StorageView(PureViewSource::Workgroup {
                id,
                count: count.max(1),
            }),
            vec![Self::number(0), Self::number(count)],
            types::view_array_with_size(
                &crate::ssa::layout::storage_value_type(element),
                Type::Constructed(TypeName::Size(count as usize), vec![]),
                types::no_buffer(),
            ),
        )
    }

    pub fn workgroup_barrier(&mut self) -> Result<(), OptimizeError> {
        self.builder.push_void_inst(InstKind::ControlBarrier).map(|_| ()).map_err(builder_error)
    }
}
