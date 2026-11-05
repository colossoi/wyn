//! Flatten carried products once; execution planning chooses their residency.
use super::Import;
use crate::egglog::OptimizeError;
use crate::types::{strip_existentials, Type, TypeExt, TypeName};
use egglog_engine::{Value, Write};

pub(super) fn supported(ty: &Type) -> bool {
    match ty {
        Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields) => {
            !fields.is_empty() && fields.iter().all(supported)
        }
        // A swap exchanges storage, not shape descriptors. Existential and
        // bounded arrays can change live length, so they retain local lowering.
        // Named/type-variable extents are invariant under the loop's type.
        ty if ty.is_array() => {
            !matches!(
                crate::types::get_array_variant(ty),
                Some(TypeName::ArrayVariantBounded)
            ) && ty.array_dims().is_some_and(|dims| {
                dims.iter().all(|dim| !matches!(dim, Type::Constructed(TypeName::SizePlaceholder, _)))
            }) && ty
                .elem_type()
                .is_some_and(|element| crate::ssa::layout::type_byte_size(element).is_some())
        }
        ty => crate::egglog::host::lower::scalar_type(ty).is_some(),
    }
}

impl Import<'_, '_, '_, '_> {
    pub(super) fn loop_carries(
        &mut self,
        owner: Value,
        ty: &Type,
        initial: Value,
        state: Value,
        next: Value,
        output: Value,
        slot: &mut i64,
    ) -> Result<Value, OptimizeError> {
        if let Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields) = strip_existentials(ty)
        {
            let mut domain = self.sink.add("Fixed", 1i64)?;
            self.sink.add("SourceLoopAggregate", (owner, state))?;
            self.sink.add("SourceLoopAggregate", (owner, output))?;
            for (index, field) in fields.iter().enumerate() {
                let mut values = Vec::new();
                for parent in [initial, state, next, output] {
                    let child = self.sink.add("SourceProjected", (parent, index as i64))?;
                    self.value_type(child, field)?;
                    self.flags(child, false, true, true, true, 0)?;
                    let projection = self.sink.add("ProjectionInfo", (parent, index as i64))?;
                    self.sink.set("SourceProjection", child, projection)?;
                    self.use_summary(child, parent, false);
                    if parent == state || parent == output {
                        self.sink.add("SourceLoopField", (owner, parent, index as i64, child))?;
                    }
                    values.push(child);
                }
                let extent =
                    self.loop_carries(owner, field, values[0], values[1], values[2], values[3], slot)?;
                domain = self.sink.add("Sum", (domain, extent))?;
            }
            Ok(domain)
        } else {
            self.sink.add(
                "SourceLoopCarry",
                (owner, *slot, initial, state, next, output, ty.is_array()),
            )?;
            *slot += 1;
            if ty.is_array() {
                Ok(self.sink.add("Length", initial)?)
            } else {
                Ok(self.sink.add("Fixed", 1i64)?)
            }
        }
    }
}
