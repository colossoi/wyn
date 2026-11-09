//! Flatten carried products once; execution planning chooses their residency.
use super::Import;
use crate::egglog::OptimizeError;
use crate::ssa::layout::type_byte_size;
use crate::types::{self, strip_existentials, Type, TypeExt, TypeName};
use egglog_engine::{Value, Write};

pub(super) fn supported(ty: &Type) -> bool {
    match ty {
        Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields) => {
            !fields.is_empty() && fields.iter().all(supported)
        }
        ty if ty.is_array() => ty.elem_type().is_some_and(|element| {
            type_byte_size(element).is_some() || (element.is_array() && supported(element))
        }),
        ty => crate::egglog::host::lower::scalar_type(ty).is_some(),
    }
}

impl Import<'_, '_, '_, '_> {
    pub(super) fn loop_carries(
        &mut self,
        owner: Value,
        ty: &Type,
        values: [Value; 4],
        slot: &mut i64,
    ) -> Result<Value, OptimizeError> {
        let [initial, state, next, output] = values;
        let descriptor =
            ty.elem_type().filter(|element| element.is_array() && type_byte_size(element).is_none());
        let parts = descriptor.map(|element| {
            vec![
                types::view_array_with_size(
                    &types::i32(),
                    Type::Constructed(TypeName::SizePlaceholder, vec![]),
                    types::no_buffer(),
                ),
                element.clone(),
            ]
        });
        let fields = match strip_existentials(ty) {
            Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields) => Some(fields.as_slice()),
            _ => parts.as_deref(),
        };
        if let Some(fields) = fields {
            let mut domain = self.sink.add("Fixed", 1i64)?;
            self.sink.add("SourceLoopAggregate", (owner, state))?;
            self.sink.add("SourceLoopAggregate", (owner, output))?;
            for (index, field) in fields.iter().enumerate() {
                let constructor =
                    if descriptor.is_some() { "SourceDescriptorPart" } else { "SourceProjected" };
                let mut children = values;
                for (child, parent) in children.iter_mut().zip(values) {
                    *child = self.sink.add(constructor, (parent, index as i64))?;
                    self.value_type(*child, field)?;
                    self.flags(*child, false, true, true, true, 0)?;
                    if descriptor.is_none() {
                        let projection = self.sink.add("ProjectionInfo", (parent, index as i64))?;
                        self.sink.set("SourceProjection", *child, projection)?;
                    }
                    self.use_value(*child, parent, false)?;
                    if parent == state || parent == output {
                        self.sink.add("SourceLoopField", (owner, parent, index as i64, *child))?;
                    }
                }
                let extent = self.loop_carries(owner, field, children, slot)?;
                domain = self.sink.add("Sum", (domain, extent))?;
            }
            return Ok(domain);
        }
        self.sink.add(
            "SourceLoopCarry",
            (owner, *slot, initial, state, next, output, ty.is_array()),
        )?;
        *slot += 1;
        let mut domain = self.sink.add("Fixed", 1i64)?;
        if ty.is_array() {
            domain = self.sink.add("Length", initial)?;
            if !matches!(ty.array_size(), Some(Type::Constructed(TypeName::Size(_), _))) {
                let mut lengths = values;
                for (length, value) in lengths.iter_mut().zip(values) {
                    *length = self.sink.add("SourceShape", value)?;
                    self.value_type(*length, &types::i32())?;
                    self.flags(*length, false, true, true, true, 0)?;
                    self.use_value(*length, value, false)?;
                }
                let metadata = self.loop_carries(owner, &types::i32(), lengths, slot)?;
                domain = self.sink.add("Sum", (domain, metadata))?;
            }
        }
        Ok(domain)
    }
}
