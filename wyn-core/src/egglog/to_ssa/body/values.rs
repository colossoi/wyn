use super::{builder_error, error, Body, OptimizeError, Typed};
use crate::builtins::{catalog, select};
use crate::op::{BinaryOperator, OpTag};
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
            || self.local_arrays.contains_key(&array.value)
        {
            let (place, ty) = self.index_place(array, index)?;
            let value =
                self.builder.push_inst(InstKind::Load { place }, ty.clone()).map_err(builder_error)?;
            return Ok(Typed {
                value: value.into(),
                ty,
            });
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
                return self.op(OpTag::Tuple(fields.len()), fields, ty);
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
        self.op(OpTag::Tuple(values.len()), values, ty)
    }
    fn number(n: u32) -> Typed {
        Typed {
            value: ValueRef::Const(ConstantValue::U32(n)),
            ty: Type::Constructed(TypeName::UInt(32), vec![]),
        }
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
