use super::{builder_error, error, Body, OptimizeError, Typed};
use crate::op::BinaryOperator;
use crate::ssa::types::{AtomicOp, InstKind, PlaceId};
use crate::types::{self, Type};
use crate::ResourceAccess;
use egglog_engine::Value;

use crate::egglog::execution::AtomicUpdate;

impl Body<'_, '_, '_> {
    fn atomic(
        &mut self,
        place: PlaceId,
        op: AtomicOp,
        values: Vec<Typed>,
        element: &Type,
    ) -> Result<Typed, OptimizeError> {
        let ty = if op == AtomicOp::CompareExchange {
            types::tuple(vec![element.clone(), types::bool_type()])
        } else {
            element.clone()
        };
        let values = values.into_iter().map(|value| value.value).collect();
        let value = self
            .builder
            .push_inst(InstKind::Atomic { place, op, values }, ty.clone())
            .map_err(builder_error)?;
        Ok(Typed {
            value: value.into(),
            ty,
        })
    }

    pub(in crate::egglog::to_ssa) fn atomic_accumulate(
        &mut self,
        name: Value,
        indices: &[Typed],
        value: Typed,
        update: AtomicUpdate,
    ) -> Result<(), OptimizeError> {
        let array = self.named(name)?;
        let [index] = indices else {
            return Err(error("atomic update requires a scalar cell"));
        };
        self.storage_access(&array.ty, ResourceAccess::ReadWrite);
        let (place, element) = self.index_place(array, index.clone())?;
        let value = self.cast(value, &element)?;
        if let AtomicUpdate::Native(op) = update {
            self.atomic(place, op, vec![value], &element)?;
            return Ok(());
        }
        let AtomicUpdate::CompareExchange(reducer) = update else {
            return Err(error("atomic reducer missing"));
        };
        let previous = self.atomic(place, AtomicOp::Load, vec![], &element)?;
        let pending = self.literal(0, &types::bool_type())?;
        self.while_loop(
            vec![previous, pending.clone()],
            |body, state| body.binary(BinaryOperator::Equal, state[1].clone(), pending),
            |body, state| {
                let combined = body.lambda(reducer, vec![state[0].clone(), value.clone()])?;
                let [next] = combined.as_slice() else {
                    return Err(error("atomic reducer must return one scalar"));
                };
                let next = body.cast(next.clone(), &element)?;
                let observed = body.atomic(
                    place,
                    AtomicOp::CompareExchange,
                    vec![state[0].clone(), next],
                    &element,
                )?;
                body.unpack(observed)
            },
        )?;
        Ok(())
    }
}
