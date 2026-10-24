use super::{builder_error, error, Body, OptimizeError, Typed};
use crate::builtins::{catalog, select};
use crate::op::{BinaryOperator, OpTag, PureViewSource};
use crate::ssa::ir::ValueDef;
use crate::ssa::types::ConstantValue;
use crate::ssa::types::ValueRef;
use crate::ssa::types::{InstKind, PlaceId};
use crate::types::{
    self, bool_type, is_array_variant_view, is_array_variant_virtual, strip_existentials, Type, TypeExt,
    TypeName,
};
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
    pub(in crate::egglog::to_ssa) fn index(
        &mut self,
        array: Typed,
        index: Typed,
    ) -> Result<Typed, OptimizeError> {
        if let Type::Constructed(TypeName::Tuple(_), fields) = &array.ty {
            let mut values = vec![];
            for i in 0..fields.len() {
                let part = self.field(array.clone(), i)?;
                values.push(self.index(part, index.clone())?);
            }
            return self.tuple(values);
        }
        if array.ty.array_variant().is_some_and(is_array_variant_view)
            && array.ty.array_rank().is_some_and(|rank| rank > 1)
        {
            let Type::Constructed(TypeName::Array, mut fields) = array.ty.clone() else {
                return Err(error("rectangular view has no array type"));
            };
            let rank = fields.len() - 3;
            let mut stride = Self::number(1);
            let mut dimensions = Vec::new();
            for axis in 1..rank {
                let length = self.op(
                    OpTag::Project {
                        index: (axis + 1) as u32,
                    },
                    vec![array.clone()],
                    Type::Constructed(TypeName::UInt(32), vec![]),
                )?;
                stride = self.binary(BinaryOperator::Multiply, stride, length.clone())?;
                dimensions.push(length);
            }
            let index = self.cast(index, &Type::Constructed(TypeName::UInt(32), vec![]))?;
            let offset = self.binary(BinaryOperator::Multiply, index, stride)?;
            let mut operands = vec![offset];
            operands.extend(dimensions);
            operands.push(array);
            fields.remove(2);
            return self.op(
                OpTag::StorageView(PureViewSource::Inherited),
                operands,
                Type::Constructed(TypeName::Array, fields),
            );
        }
        if array.ty.array_variant().is_some_and(is_array_variant_view)
            || self.local_arrays.contains_key(&array.value)
        {
            let (place, ty) = self.index_place(array, index)?;
            return self.load_place(place, &ty);
        }
        let Some(ty) = array.ty.elem_type().cloned() else {
            return Err(error(format!("index of non-array {:?}", array.ty)));
        };
        let index = if array.ty.array_variant().is_some_and(is_array_variant_virtual) {
            self.cast(index, &ty)?
        } else {
            index
        };
        self.op(OpTag::Index, vec![array, index], ty)
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
        // Source array qualifiers describe shape and ownership, not a physical
        // conversion. Boundary layouts explicitly request any required copy.
        if value.ty.is_array() && ty.is_array() {
            return Ok(value);
        }
        if let (
            Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), a),
            Type::Constructed(name @ (TypeName::Tuple(_) | TypeName::Record(_)), b),
        ) = (&value.ty, ty)
        {
            if a.len() == b.len() {
                let mut fields = vec![];
                for (i, target) in b.iter().enumerate() {
                    let field = self.field(value.clone(), i)?;
                    fields.push(self.cast(field, target)?);
                }
                let ty = Type::Constructed(name.clone(), fields.iter().map(|f| f.ty.clone()).collect());
                return self.op(OpTag::Tuple, fields, ty);
            }
        }
        if *ty == bool_type() {
            return self.binary(BinaryOperator::NotEqual, value, Self::number(0));
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
        text: &str,
        ty: &Type,
    ) -> Result<Typed, OptimizeError> {
        let tag = match strip_existentials(ty) {
            Type::Constructed(TypeName::UInt(_), _) => OpTag::Uint(text.into()),
            Type::Constructed(TypeName::Int(_), _) => OpTag::Int(text.into()),
            Type::Constructed(TypeName::Bool, _) => OpTag::Bool(text == "true"),
            Type::Constructed(TypeName::Float(_), _) => {
                let bits: u32 = text.parse().map_err(|_| error("invalid folded float bits"))?;
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
    pub(in crate::egglog::to_ssa) fn length(&mut self, array: Typed) -> Result<Typed, OptimizeError> {
        if matches!(array.ty, Type::Constructed(TypeName::Tuple(_), _)) {
            let first = self.field(array, 0)?;
            return self.length(first);
        }
        if array.ty.array_variant().is_some_and(is_array_variant_view) {
            let length = self.op(
                OpTag::Project { index: 1 },
                vec![array],
                Type::Constructed(TypeName::UInt(32), vec![]),
            )?;
            return self.cast(length, &types::i32());
        }
        self.op(
            OpTag::Intrinsic {
                id: catalog().known().length,
                overload_idx: 0,
            },
            vec![array],
            types::i32(),
        )
    }
}

impl Body<'_, '_, '_> {
    fn child_place(&mut self, place: PlaceId, index: Typed, ty: &Type) -> Result<PlaceId, OptimizeError> {
        let result = self.builder.new_place(ty.clone());
        self.builder
            .push_void_inst(InstKind::PlaceIndex {
                place,
                index: index.value,
                result,
            })
            .map_err(builder_error)?;
        Ok(result)
    }

    // Transfer aggregates through element pointers so buffer layout conversion
    // does not expand fixed arrays into one instruction per element.
    fn load_place(&mut self, place: PlaceId, ty: &Type) -> Result<Typed, OptimizeError> {
        if let Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields) = ty {
            let mut values = Vec::new();
            for (i, field) in fields.iter().enumerate() {
                let child = self.child_place(place, Self::number(i as u32), field)?;
                values.push(self.load_place(child, field)?);
            }
            return self.op(OpTag::Tuple, values, ty.clone());
        }
        if ty.array_variant().is_some_and(types::is_array_variant_composite) {
            let Some(Type::Constructed(TypeName::Size(n), _)) = ty.array_size() else {
                return Err(error("aggregate load requires a concrete array dimension"));
            };
            let Some(element) = ty.elem_type() else {
                return Err(error("aggregate load requires an array element type"));
            };
            let output = self.builder.new_place(ty.clone());
            self.builder.push_void_inst(InstKind::Alloca { result: output }).map_err(builder_error)?;
            let count = self.literal(&n.to_string(), &types::i32())?;
            self.counted(
                Self::number(0),
                count,
                Self::number(1),
                vec![],
                |body, index, _| {
                    let source = body.child_place(place, index.clone(), element)?;
                    let item = body.load_place(source, element)?;
                    body.store_place(output, ty, &[index], item)?;
                    Ok(vec![])
                },
            )?;
            let value = self
                .builder
                .push_inst(InstKind::Load { place: output }, ty.clone())
                .map_err(builder_error)?;
            return Ok(Typed {
                value: value.into(),
                ty: ty.clone(),
            });
        }
        let value = self.builder.push_inst(InstKind::Load { place }, ty.clone()).map_err(builder_error)?;
        Ok(Typed {
            value: value.into(),
            ty: ty.clone(),
        })
    }

    pub(in crate::egglog::to_ssa) fn store(
        &mut self,
        array: Typed,
        index: Typed,
        value: Typed,
    ) -> Result<(), OptimizeError> {
        self.store_at(array, &[index], value)
    }

    pub fn store_at(&mut self, array: Typed, indices: &[Typed], value: Typed) -> Result<(), OptimizeError> {
        let Some((first, rest)) = indices.split_first() else {
            let length = self.length(value.clone())?;
            return self.copy_array(array, value, length, &[]);
        };
        if array.ty.array_variant().is_some_and(is_array_variant_view)
            && array.ty.array_rank().is_some_and(|rank| rank > 1)
        {
            let row = self.index(array, first.clone())?;
            return self.store_at(row, rest, value);
        }
        let (place, ty) = self.index_place(array, first.clone())?;
        self.store_place(place, &ty, rest, value)
    }

    fn store_place(
        &mut self,
        place: PlaceId,
        ty: &Type,
        indices: &[Typed],
        value: Typed,
    ) -> Result<(), OptimizeError> {
        if let Some((index, rest)) = indices.split_first() {
            if let Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields) = ty {
                for (i, field) in fields.iter().enumerate() {
                    let next = self.builder.new_place(field.clone());
                    self.builder
                        .push_void_inst(InstKind::PlaceIndex {
                            place,
                            index: Self::number(i as u32).value,
                            result: next,
                        })
                        .map_err(builder_error)?;
                    let value = self.field(value.clone(), i)?;
                    self.store_place(next, field, indices, value)?;
                }
                return Ok(());
            }
            let Some(element) = ty.elem_type() else {
                return Err(error("store index exceeds array rank"));
            };
            let next = self.builder.new_place(element.clone());
            self.builder
                .push_void_inst(InstKind::PlaceIndex {
                    place,
                    index: index.value,
                    result: next,
                })
                .map_err(builder_error)?;
            return self.store_place(next, element, rest, value);
        }
        // A view carries a descriptor; an owned destination needs its elements.
        let copy_array = ty.array_variant().is_some_and(types::is_array_variant_composite);
        if value.ty.is_array()
            && (copy_array || matches!(ty, Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), _)))
        {
            let n = self.length(value.clone())?;
            let zero = self.literal("0", &types::i32())?;
            let one = self.literal("1", &types::i32())?;
            self.counted(zero, n, one, vec![], |body, index, _| {
                let item = body.index(value.clone(), index.clone())?;
                body.store_place(place, ty, &[index], item)?;
                Ok(vec![])
            })?;
            return Ok(());
        }
        if let Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields) = ty {
            for (i, field) in fields.iter().enumerate() {
                let child = self.child_place(place, Self::number(i as u32), field)?;
                let item = self.field(value.clone(), i)?;
                self.store_place(child, field, &[], item)?;
            }
            return Ok(());
        }
        let value = self.cast(value, ty)?;
        self.builder
            .push_void_inst(InstKind::Store {
                place,
                value: value.value,
            })
            .map_err(builder_error)?;
        Ok(())
    }

    pub(in crate::egglog::to_ssa) fn view_type(element: &Type, region: Type) -> Type {
        types::view_array_with_size(
            element,
            Type::Constructed(TypeName::SizePlaceholder, vec![]),
            region,
        )
    }
}

impl Body<'_, '_, '_> {
    pub fn copy_array(
        &mut self,
        destination: Typed,
        source: Typed,
        length: Typed,
        fields: &[usize],
    ) -> Result<(), OptimizeError> {
        let zero = self.literal("0", &length.ty)?;
        let one = self.literal("1", &length.ty)?;
        self.copy_array_strided(destination, source, length, fields, zero, one)
    }

    pub fn copy_array_strided(
        &mut self,
        destination: Typed,
        source: Typed,
        length: Typed,
        fields: &[usize],
        start: Typed,
        step: Typed,
    ) -> Result<(), OptimizeError> {
        self.counted(start, length, step, vec![], |body, index, _| {
            let mut value = body.index(source.clone(), index.clone())?;
            if !fields.is_empty() && source.ty.array_rank().is_some_and(|rank| rank > 1) {
                let target = body.index(destination.clone(), index)?;
                let length = body.length(value.clone())?;
                body.copy_array(target, value, length, fields)?;
            } else {
                for &field in fields {
                    value = body.field(value, field)?;
                }
                body.store(destination.clone(), index, value)?;
            }
            Ok(vec![])
        })?;
        Ok(())
    }
}
