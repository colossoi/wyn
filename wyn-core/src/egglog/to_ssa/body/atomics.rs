use super::{builder_error, error, Body, OptimizeError, Typed};
use crate::egglog::graph::CollectiveInput;
use crate::egglog::payload::{read, Payload};
use crate::op::BinaryOperator;
use crate::ssa::types::{AtomicOp, InstKind, PlaceId};
use crate::types::{self, Type};
use crate::ResourceAccess;
use egglog_engine::Value;

pub(in crate::egglog::to_ssa) enum AtomicUpdate {
    Native(AtomicOp),
    CompareExchange(Value),
}

impl Body<'_, '_, '_> {
    pub(in crate::egglog::to_ssa) fn atomic_policy(
        &self,
        arrays: &[Value],
        policy: Value,
        combine: Option<Value>,
    ) -> Result<Option<AtomicUpdate>, OptimizeError> {
        let [name] = arrays else {
            return Ok(None);
        };
        if !self.query().any("AtomicAccumulatorArray", |row| row[2] == *name)? {
            return Ok(None);
        }
        let array = self.named(*name)?;
        if types::array_view_buffer(&array.ty).is_none() {
            return Ok(None);
        }
        let Some(combine) = combine else {
            return Ok(self
                .query()
                .enode("IdenticalWrites", policy)?
                .map(|_| AtomicUpdate::Native(AtomicOp::Exchange)));
        };
        if !self
            .query()
            .lookup("RetrySafe", combine)?
            .is_some_and(|safe| self.query().0.value_to_base::<bool>(safe))
        {
            return Ok(None);
        }
        let Some(opcode) = self.query().lookup("NativeBinaryReducer", combine)? else {
            return Ok(Some(AtomicUpdate::CompareExchange(combine)));
        };
        let opcode = read(self.query().0, opcode);
        let op = match opcode {
            Payload::Binary(BinaryOperator::Add) => AtomicOp::Add,
            Payload::Binary(BinaryOperator::BitwiseAnd) => AtomicOp::And,
            Payload::Binary(BinaryOperator::BitwiseOr) => AtomicOp::Or,
            Payload::Binary(BinaryOperator::BitwiseXor) => AtomicOp::Xor,
            _ => return Err(error("unknown native integer reducer")),
        };
        Ok(Some(AtomicUpdate::Native(op)))
    }

    pub(in crate::egglog::to_ssa) fn atomic_contributions(
        &self,
        operation: Value,
    ) -> Result<bool, OptimizeError> {
        let query = self.query();
        for clause in 0..query.0.result_count(operation) {
            let kind = query.0.result_kind(operation, clause)?;
            if query.enode("Contributed", kind)?.is_none() {
                return Ok(false);
            }
            for field in 0..query.0.collective_input_count(operation, clause, CollectiveInput::Contribution)
            {
                let name =
                    query.0.collective_input(operation, clause, CollectiveInput::Contribution, field)?;
                let token = query.fields("AccNameRef", name)?[0];
                let capability =
                    query.fields("TokenBindingAt", query.required("TokenCapability", token)?)?;
                let owner = capability[0];
                let group = query.0.value_to_base::<i64>(capability[1]);
                let arrays = query.0.collective_inputs(owner, group, CollectiveInput::Backing);
                let policy = query.0.accumulator_policy(owner, group)?;
                let combine = query.0.combiner_at(owner, group);
                if self.atomic_policy(&arrays, policy, combine)?.is_none() {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

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
