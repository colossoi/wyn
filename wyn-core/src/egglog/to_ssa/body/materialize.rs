//! Expand a selected boundary layout into representation conversions and copies.
use super::{builder_error, error, Body, OptimizeError, Typed};
use crate::op::OpTag;
use crate::ssa::types::InstKind;
use crate::types::{self, TypeExt};
use egglog_engine::Value;

impl Body<'_, '_, '_> {
    pub(super) fn materialize(&mut self, value: Typed, layout: Value) -> Result<Typed, OptimizeError> {
        self.materialize_component(value, layout, &[])
    }

    fn materialize_component(
        &mut self,
        mut value: Typed,
        layout: Value,
        mut path: &[usize],
    ) -> Result<Typed, OptimizeError> {
        if !value.ty.is_array() {
            for &field in path {
                value = self.field(value, field)?;
            }
            path = &[];
        }
        let ty = self.compiler.facts.layout_type(layout)?;
        if path.is_empty() && value.ty == ty {
            return Ok(value);
        }
        if let Some(fields) = self.compiler.facts.enode("ArrayLayout", layout) {
            let n = self.compiler.facts.integer(fields[0]);
            let element = self.compiler.facts.layout_type(fields[1])?;
            let place = self.builder.new_place(ty.clone());
            self.builder
                .push_void_inst(InstKind::Alloca {
                    elem_ty: ty.clone(),
                    result: place,
                })
                .map_err(builder_error)?;
            let zero = self.literal("0", &types::i32())?;
            let length = self.literal(&n.to_string(), &types::i32())?;
            let one = self.literal("1", &types::i32())?;
            self.counted(zero, length, one, vec![], |body, index, _| {
                let item = body.index(value.clone(), index.clone())?;
                let item = body.materialize_component(item, fields[1], path)?;
                let slot = body.builder.new_place(element.clone());
                body.builder
                    .push_void_inst(InstKind::PlaceIndex {
                        place,
                        index: index.value,
                        result: slot,
                    })
                    .map_err(builder_error)?;
                body.builder
                    .push_void_inst(InstKind::Store {
                        place: slot,
                        value: item.value,
                    })
                    .map_err(builder_error)?;
                Ok(vec![])
            })?;
            let result =
                self.builder.push_inst(InstKind::Load { place }, ty.clone()).map_err(builder_error)?;
            return Ok(Typed {
                value: result.into(),
                ty,
            });
        }
        if let Some(fields) = self.compiler.facts.enode("TupleLayout", layout) {
            let layouts = self.compiler.facts.vector(fields[1])?;
            let mut values = Vec::new();
            for (i, layout) in layouts.into_iter().enumerate() {
                let field = if value.ty.is_array() {
                    let mut field = path.to_vec();
                    field.push(i);
                    self.materialize_component(value.clone(), layout, &field)?
                } else {
                    let field = self.field(value.clone(), i)?;
                    self.materialize(field, layout)?
                };
                values.push(field);
            }
            return self.op(OpTag::Tuple(values.len()), values, ty);
        }
        if !path.is_empty() {
            return Err(error("component view requires an array layout"));
        }
        self.cast(value, &ty)
    }
}
