//! Physical nested arrays are ordinary SSA aggregates of offset and data views.
use super::{error, Body, OptimizeError, Typed};
use crate::op::{BinaryOperator, OpTag};
use crate::types::{self, Type, TypeName};
use egglog_engine::Value;

impl Typed {
    pub(in crate::egglog::to_ssa) fn is_segmented(&self) -> bool {
        matches!(&self.ty, Type::Constructed(TypeName::Record(names), _) if names.0 == ["$offsets", "$elements"])
    }
}

impl Body<'_, '_, '_> {
    pub(in crate::egglog::to_ssa) fn segmented_array(
        &mut self,
        offsets: Typed,
        elements: Typed,
    ) -> Result<Typed, OptimizeError> {
        let ty = types::record(vec![
            ("$offsets".into(), offsets.ty.clone()),
            ("$elements".into(), elements.ty.clone()),
        ]);
        self.op(OpTag::Tuple, vec![offsets, elements], ty)
    }

    pub(in crate::egglog::to_ssa) fn segmented_storage(
        &mut self,
        scope: Value,
        shape: Value,
        elements: Typed,
    ) -> Result<Typed, OptimizeError> {
        if self.compiler.facts.enode("OneAxis", shape).is_some() {
            return Ok(elements);
        }
        let Some(fields) = self.compiler.facts.enode("Segments", shape) else {
            return Err(error("segmented storage contains an invalid parent shape"));
        };
        let offsets = self.resource(scope, fields[1], 1)?;
        let array = self.segmented_array(offsets, elements)?;
        self.segmented_storage(scope, fields[0], array)
    }

    pub(in crate::egglog::to_ssa) fn segment(
        &mut self,
        array: Typed,
        index: Typed,
    ) -> Result<Typed, OptimizeError> {
        let offsets = self.field(array.clone(), 0)?;
        let elements = self.field(array, 1)?;
        let next = self.binary(BinaryOperator::Add, index.clone(), Self::number(1))?;
        let start = self.index(offsets.clone(), index)?;
        let end = self.index(offsets, next)?;
        self.slice(
            elements,
            start,
            end,
            Type::Constructed(TypeName::SizePlaceholder, vec![]),
        )
    }
}
