use super::{builder_error, error, Body, OptimizeError, Typed};
use crate::flow::ControlHeader;
use crate::op::{BinaryOperator, OpTag};
use crate::ssa::types::{ConstantValue, InstKind, Terminator, ValueRef};
use crate::types;

impl Body<'_, '_, '_> {
    pub(in crate::egglog::to_ssa) fn branch(
        &mut self,
        condition: Typed,
        yes_value: impl FnOnce(&mut Self) -> Result<Typed, OptimizeError>,
        no_value: impl FnOnce(&mut Self) -> Result<Typed, OptimizeError>,
    ) -> Result<Typed, OptimizeError> {
        let start = self.current()?;
        let yes = self.builder.create_block();
        let no = self.builder.create_block();
        let end = self.builder.create_block();
        self.builder.set_control_header(start, ControlHeader::Selection { merge: end });
        self.terminate(Terminator::CondBranch {
            cond: condition.value,
            then_target: yes,
            then_args: vec![],
            else_target: no,
            else_args: vec![],
        })
        .map_err(builder_error)?;
        let outer_values = self.values.checkpoint();
        self.builder.switch_to_block_unchecked(yes);
        let a = yes_value(self)?;
        let yes_end = self.current()?;
        self.values.restore(outer_values);
        self.builder.switch_to_block_unchecked(no);
        let b = no_value(self)?;
        let no_end = self.current()?;
        if a.ty != b.ty {
            return Err(error(format!(
                "branch arms do not match their selected representation: {:?} vs {:?}",
                a.ty, b.ty
            )));
        }
        self.builder.switch_to_block_unchecked(yes_end);
        self.terminate(Terminator::Branch {
            target: end,
            args: vec![a.value],
        })
        .map_err(builder_error)?;
        self.builder.switch_to_block_unchecked(no_end);
        self.terminate(Terminator::Branch {
            target: end,
            args: vec![b.value],
        })
        .map_err(builder_error)?;
        let result = self.builder.add_block_param(end, a.ty.clone());
        self.values.restore(outer_values);
        self.builder.switch_to_block_unchecked(end);
        Ok(Typed {
            value: result.into(),
            ty: a.ty,
        })
    }
}

impl Body<'_, '_, '_> {
    // Source loops can contain arbitrarily large bodies, even with small trip counts.
    pub(in crate::egglog::to_ssa) fn counted(
        &mut self,
        start: Typed,
        bound: Typed,
        step: Typed,
        initial: Vec<Typed>,
        emit: impl FnOnce(&mut Self, Typed, Vec<Typed>) -> Result<Vec<Typed>, OptimizeError>,
    ) -> Result<Vec<Typed>, OptimizeError> {
        let bound = self.cast(bound, &start.ty)?;
        let mut state = vec![start];
        state.extend(initial);
        let mut results = self.while_loop(
            state,
            |body, state| body.binary(BinaryOperator::Less, state[0].clone(), bound),
            |body, mut state| {
                let index = state.remove(0);
                let mut next = emit(body, index.clone(), state)?;
                next.insert(0, body.binary(BinaryOperator::Add, index, step)?);
                Ok(next)
            },
        )?;
        results.remove(0);
        Ok(results)
    }

    /// Every structured loop has typed carried values and one exit. Keep the
    /// condition in its own block so nested selections cannot replace the header.
    pub(in crate::egglog::to_ssa) fn while_loop(
        &mut self,
        initial: Vec<Typed>,
        test: impl FnOnce(&mut Self, &[Typed]) -> Result<Typed, OptimizeError>,
        emit: impl FnOnce(&mut Self, Vec<Typed>) -> Result<Vec<Typed>, OptimizeError>,
    ) -> Result<Vec<Typed>, OptimizeError> {
        let types: Vec<_> = initial.iter().map(|v| v.ty.clone()).collect();
        let (header, parameters) = self.builder.create_block_with_params(types.clone());
        let condition_block = self.builder.create_block();
        let active = self.builder.create_block();
        let continuing = self.builder.create_block();
        let (done, results) = self.builder.create_block_with_params(types.clone());
        self.terminate(Terminator::Branch {
            target: header,
            args: initial.into_iter().map(|v| v.value).collect(),
        })
        .map_err(builder_error)?;
        self.builder.switch_to_block_unchecked(header);
        self.builder.set_control_header(
            header,
            ControlHeader::Loop {
                merge: done,
                continue_block: continuing,
            },
        );
        self.terminate(Terminator::Branch {
            target: condition_block,
            args: vec![],
        })
        .map_err(builder_error)?;
        let state: Vec<_> = parameters
            .into_iter()
            .zip(&types)
            .map(|(value, ty)| Typed {
                value: value.into(),
                ty: ty.clone(),
            })
            .collect();
        let values = self.values.checkpoint();
        self.builder.switch_to_block_unchecked(condition_block);
        let condition = test(self, &state)?;
        self.terminate(Terminator::CondBranch {
            cond: condition.value,
            then_target: active,
            then_args: vec![],
            else_target: done,
            else_args: state.iter().map(|v| v.value).collect(),
        })
        .map_err(builder_error)?;
        self.builder.switch_to_block_unchecked(active);
        let next = emit(self, state)?;
        if next.len() != types.len() {
            return Err(error("loop accumulator arity mismatch"));
        }
        let next = next
            .into_iter()
            .zip(&types)
            .map(|(value, ty)| self.cast(value, ty).map(|v| v.value))
            .collect::<Result<Vec<_>, _>>()?;
        self.terminate(Terminator::Branch {
            target: continuing,
            args: vec![],
        })
        .map_err(builder_error)?;
        self.builder.switch_to_block_unchecked(continuing);
        self.terminate(Terminator::Branch {
            target: header,
            args: next,
        })
        .map_err(builder_error)?;
        self.builder.switch_to_block_unchecked(done);
        self.values.restore(values);
        Ok(results
            .into_iter()
            .zip(types)
            .map(|(value, ty)| Typed {
                value: value.into(),
                ty,
            })
            .collect())
    }

    pub(in crate::egglog::to_ssa) fn integer_literal(&self, value: &Typed) -> Option<i64> {
        match value.value {
            ValueRef::Const(ConstantValue::I32(n)) => Some(i64::from(n)),
            ValueRef::Const(ConstantValue::U32(n)) => Some(i64::from(n)),
            ValueRef::Ssa(id) => {
                let function = self.builder.func();
                let inst = function.inst_of_value(id)?;
                match &function.insts[inst].data {
                    InstKind::Op {
                        tag: OpTag::Int(n) | OpTag::Uint(n),
                        ..
                    } => n.parse().ok(),
                    InstKind::Op {
                        tag: OpTag::BinOp(operator),
                        operands,
                    } if operands.len() == 2 => {
                        let a = self.integer_literal(&Typed {
                            value: operands[0],
                            ty: value.ty.clone(),
                        })?;
                        let b = self.integer_literal(&Typed {
                            value: operands[1],
                            ty: value.ty.clone(),
                        })?;
                        match operator {
                            BinaryOperator::Add => a.checked_add(b),
                            BinaryOperator::Subtract => a.checked_sub(b),
                            BinaryOperator::Multiply => a.checked_mul(b),
                            BinaryOperator::Divide => a.checked_div(b),
                            BinaryOperator::Remainder => a.checked_rem(b),
                            _ => None,
                        }
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }
}

impl Body<'_, '_, '_> {
    pub(in crate::egglog::to_ssa) fn require_condition(
        &mut self,
        condition: Typed,
    ) -> Result<(), OptimizeError> {
        let start = self.current()?;
        let yes = self.builder.create_block();
        let no = self.builder.create_block();
        self.builder.set_control_header(start, ControlHeader::Selection { merge: yes });
        self.terminate(Terminator::CondBranch {
            cond: condition.value,
            then_target: yes,
            then_args: vec![],
            else_target: no,
            else_args: vec![],
        })
        .map_err(builder_error)?;
        self.builder.switch_to_block_unchecked(no);
        self.terminate(Terminator::Unreachable).map_err(builder_error)?;
        self.builder.switch_to_block_unchecked(yes);
        Ok(())
    }

    pub(in crate::egglog::to_ssa) fn when(
        &mut self,
        condition: Typed,
        emit: impl FnOnce(&mut Self) -> Result<(), OptimizeError>,
    ) -> Result<(), OptimizeError> {
        self.branch(
            condition,
            |body| {
                emit(body)?;
                body.tuple(Vec::new())
            },
            |body| body.tuple(Vec::new()),
        )?;
        Ok(())
    }
}

impl Body<'_, '_, '_> {
    /// Retry an atomic update until its `(observed, exchanged)` result succeeds.
    pub(in crate::egglog::to_ssa) fn retry(
        &mut self,
        initial: Typed,
        emit: impl FnOnce(&mut Self, Typed) -> Result<Typed, OptimizeError>,
    ) -> Result<Typed, OptimizeError> {
        let mut results = self.while_loop(
            vec![initial],
            |body, state| {
                let exchanged = body.field(state[0].clone(), 1)?;
                let no = body.op(OpTag::Bool(false), vec![], types::bool_type())?;
                body.binary(BinaryOperator::Equal, exchanged, no)
            },
            |body, mut state| {
                let Some(state) = state.pop() else {
                    return Err(error("atomic retry state is missing"));
                };
                Ok(vec![emit(body, state)?])
            },
        )?;
        let Some(result) = results.pop() else {
            return Err(error("atomic retry result is missing"));
        };
        Ok(result)
    }
}
