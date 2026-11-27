use super::{error, Body, OptimizeError, Typed, Value};
use crate::host::Expr;
use crate::op::BinaryOperator;
use crate::ssa::types::AtomicOp;
use crate::types;
use crate::ResourceAccess;

impl Body<'_, '_, '_> {
    pub(super) fn reservation_cursor(&mut self) -> Result<Typed, OptimizeError> {
        let zero = Self::number(0);
        let one = Self::number(1);
        if self.active_dispatch.is_none() && self.generated.is_some() {
            // Fresh device allocations are zero initialized by the host.
            let (_, cursor) = self.device_buffer(&zero.ty, Expr::Integer(4), one)?;
            return Ok(cursor);
        }
        let cursor = self.local_array_with_length(types::sized_array(1, zero.ty.clone()), one)?;
        self.store(cursor.clone(), zero.clone(), zero)?;
        Ok(cursor)
    }

    pub(super) fn reserve_span(&mut self, inputs: &[Value]) -> Result<Typed, OptimizeError> {
        let [cursor, capacity, source, count] = inputs else {
            return Err(error(
                "row reservation expects a cursor, capacity, lane count, and row size",
            ));
        };
        let cursor = self.named(*cursor)?;
        let zero = Self::number(0);
        let unsigned = &zero.ty;
        let source_value = self.named(*source)?;
        let query = self.query();
        let row_bound = if let Some(index) = query.constructor("IndexValue", *count)? {
            query
                .lookup("IndexUpper", index)?
                .and_then(|bound| u32::try_from(query.0.value_to_base::<i64>(bound)).ok())
        } else {
            None
        };
        let count = self.named(*count)?;
        let count = self.cast(count, unsigned)?;
        let capacity = self.named(*capacity)?;
        let capacity = self.cast(capacity, unsigned)?;
        let lanes = self.array_length(source_value)?;
        let lane_bound = self
            .integer_literal(&lanes)
            .or_else(|| self.compiler.execution.storage.static_count.get(source).copied())
            .and_then(|n| u32::try_from(n).ok());
        let possible_count =
            row_bound.or_else(|| self.integer_literal(&capacity).and_then(|n| u32::try_from(n).ok()));
        let sum_fits =
            lane_bound.zip(possible_count).is_some_and(|(n, width)| n.checked_mul(width).is_some());
        let lanes = self.cast(lanes, unsigned)?;
        let fits = self.binary(BinaryOperator::LessEqual, count.clone(), capacity.clone())?;
        let positive = self.binary(BinaryOperator::Greater, count.clone(), zero.clone())?;
        let request = self.binary(BinaryOperator::LogicalAnd, fits, positive)?;
        let failed = self.literal(-1, &types::i32())?;
        self.branch(
            request,
            |body| {
                let last_start = body.binary(BinaryOperator::Subtract, capacity.clone(), count.clone())?;
                let start = if types::array_view_buffer(&cursor.ty).is_some() {
                    body.storage_access(&cursor.ty, ResourceAccess::ReadWrite);
                    let (place, element) = body.index_place(cursor, zero.clone())?;
                    // Checked reservations end at or before capacity. All fetch-adds combined
                    // can add at most the remaining u32 headroom, even when the arena is full.
                    if sum_fits {
                        body.atomic(place, AtomicOp::Add, vec![count.clone()], &element)?
                    } else {
                        let headroom =
                            body.binary(BinaryOperator::Subtract, Self::number(u32::MAX), capacity)?;
                        let share = body.binary(BinaryOperator::Divide, headroom, lanes)?;
                        let fast = body.binary(BinaryOperator::LessEqual, count.clone(), share)?;
                        body.branch(
                            fast,
                            |body| body.atomic(place, AtomicOp::Add, vec![count.clone()], &element),
                            |body| {
                                let observed = body.atomic(place, AtomicOp::Load, vec![], &element)?;
                                let pending = body.literal(0, &types::bool_type())?;
                                let result = body.while_loop(
                                    vec![observed, pending.clone()],
                                    |body, state| {
                                        let fits = body.binary(
                                            BinaryOperator::LessEqual,
                                            state[0].clone(),
                                            last_start.clone(),
                                        )?;
                                        let retry =
                                            body.binary(BinaryOperator::Equal, state[1].clone(), pending)?;
                                        body.binary(BinaryOperator::LogicalAnd, fits, retry)
                                    },
                                    |body, state| {
                                        let next = body.binary(
                                            BinaryOperator::Add,
                                            state[0].clone(),
                                            count.clone(),
                                        )?;
                                        let claim = body.atomic(
                                            place,
                                            AtomicOp::CompareExchange,
                                            vec![state[0].clone(), next],
                                            &element,
                                        )?;
                                        body.unpack(claim)
                                    },
                                )?;
                                Ok(result[0].clone())
                            },
                        )?
                    }
                } else {
                    let old = body.index(cursor.clone(), zero.clone())?;
                    let fits = body.binary(BinaryOperator::LessEqual, old.clone(), last_start.clone())?;
                    body.when(fits, |body| {
                        let next = body.binary(BinaryOperator::Add, old.clone(), count.clone())?;
                        body.store(cursor, zero.clone(), next)
                    })?;
                    old
                };
                let fits = body.binary(BinaryOperator::LessEqual, start.clone(), last_start)?;
                let start = body.cast(start, &types::i32())?;
                body.select(fits, start, failed.clone())
            },
            |body| {
                let empty = body.binary(BinaryOperator::Equal, count.clone(), zero.clone())?;
                let start = body.literal(0, &types::i32())?;
                body.select(empty, start, failed.clone())
            },
        )
    }
}
