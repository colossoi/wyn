use super::{builder_error, error, Body, OptimizeError, Typed, Value};
use crate::flow::ControlHeader;
use crate::op::{BinaryOperator, OpTag};
use crate::ssa::types::{ConstantValue, InstKind, Terminator, ValueRef};
use crate::types;

impl Body<'_, '_, '_> {
    pub(in crate::egglog::to_ssa) fn branch(
        &mut self,
        scope: Value,
        condition: Typed,
        yes_value: impl FnOnce(&mut Self) -> Result<Typed, OptimizeError>,
        no_value: impl FnOnce(&mut Self) -> Result<Typed, OptimizeError>,
        source_scopes: Option<(Value, Value)>,
    ) -> Result<Typed, OptimizeError> {
        self.branch_with_type(
            Some(scope),
            condition,
            yes_value,
            |body, _| no_value(body),
            source_scopes,
        )
    }

    /// Let a synthetic fallback use the representation produced by the true arm.
    pub(in crate::egglog::to_ssa) fn branch_with_type(
        &mut self,
        scope: Option<Value>,
        condition: Typed,
        yes_value: impl FnOnce(&mut Self) -> Result<Typed, OptimizeError>,
        no_value: impl FnOnce(&mut Self, &types::Type) -> Result<Typed, OptimizeError>,
        source_scopes: Option<(Value, Value)>,
    ) -> Result<Typed, OptimizeError> {
        let start = self.current()?;
        if let Some(scope) = scope {
            self.scopes.insert(scope, start);
        }
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
        let outer_scopes = self.scopes.checkpoint();
        self.builder.switch_to_block_unchecked(yes);
        if let Some((scope, _)) = source_scopes {
            self.scopes.insert(scope, yes);
        }
        let a = yes_value(self)?;
        let yes_end = self.current()?;
        self.values.restore(outer_values);
        self.scopes.restore(outer_scopes);
        self.builder.switch_to_block_unchecked(no);
        if let Some((_, scope)) = source_scopes {
            self.scopes.insert(scope, no);
        }
        let b = no_value(self, &a.ty)?;
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
        self.scopes.restore(outer_scopes);
        self.builder.switch_to_block_unchecked(end);
        if let Some(scope) = scope {
            self.scopes.insert(scope, end);
        }
        Ok(Typed {
            value: result.into(),
            ty: a.ty,
        })
    }
}

impl Body<'_, '_, '_> {
    pub(super) fn loop_(&mut self, scope: Value, source: Value) -> Result<Typed, OptimizeError> {
        let Some((header_scope, iteration_scope)) = self.compiler.facts.loops(source) else {
            return Err(error("loop has no regions"));
        };
        let (Some(state), Some(next), Some(form)) = (
            self.compiler.facts.loop_state(header_scope),
            self.compiler.facts.result(iteration_scope),
            self.compiler.facts.loop_form(source),
        ) else {
            return Err(error("loop has incomplete control facts"));
        };
        let layout = self.compiler.facts.value_layout(state)?;
        let initial = self.compiler.facts.loop_initial(header_scope)?;
        let initial = self.value(scope, initial)?;
        let target = self.boundary_target(source)?;
        let initial = self.materialize_boundary(initial, layout, target.as_ref())?;
        if let Some(fields) = self.compiler.facts.enode("WhileCondition", form) {
            let mut results = self.while_loop(
                vec![initial],
                |body, state_values| {
                    body.scopes.insert(header_scope, body.current()?);
                    let carried = body.rebind_boundary(state_values[0].clone(), target.as_ref())?;
                    body.values.insert(state, carried);
                    body.value(header_scope, fields[0])
                },
                |body, _| {
                    body.scopes.insert(iteration_scope, body.current()?);
                    let next = body.value(iteration_scope, next)?;
                    Ok(vec![body.materialize_boundary(next, layout, target.as_ref())?])
                },
            )?;
            self.scopes.insert(scope, self.current()?);
            let Some(result) = results.pop() else {
                return Err(error("loop result is missing"));
            };
            return self.rebind_boundary(result, target.as_ref());
        }
        let (bound, array, index_ty) = if let Some(fields) = self.compiler.facts.enode("ForCount", form) {
            let Some(ty) = self.compiler.facts.ty(fields[1]).cloned() else {
                return Err(error("counted loop has no index type"));
            };
            (self.value(scope, fields[0])?, None, ty)
        } else if let Some(fields) = self.compiler.facts.enode("ForEach", form) {
            let array = self.value(scope, fields[0])?;
            (self.length(array.clone())?, Some(array), types::i32())
        } else {
            return Err(error("unknown loop form"));
        };
        let Some(parameter) = self.compiler.facts.iteration(iteration_scope) else {
            return Err(error("counted loop has no iteration parameter"));
        };
        let zero = self.literal(0, &index_ty)?;
        let one = self.literal(1, &index_ty)?;
        let results = self.counted_loop(zero, bound, one, vec![initial], |body, index, mut carried| {
            let Some(value) = carried.pop() else {
                return Err(error("loop state parameter is missing"));
            };
            let block = body.current()?;
            body.scopes.insert(header_scope, block);
            body.scopes.insert(iteration_scope, block);
            let value = body.rebind_boundary(value, target.as_ref())?;
            body.values.insert(state, value);
            let value = if let Some(array) = array { body.index(array, index)? } else { index };
            let Some(ty) = body.compiler.facts.source_type(parameter).cloned() else {
                return Err(error("iteration parameter has no type"));
            };
            let value = body.cast(value, &ty)?;
            body.values.insert(parameter, value);
            let next = body.value(iteration_scope, next)?;
            Ok(vec![body.materialize_boundary(next, layout, target.as_ref())?])
        })?;
        self.scopes.insert(scope, self.current()?);
        let Some(result) = results.into_iter().next() else {
            return Err(error("loop result is missing"));
        };
        self.rebind_boundary(result, target.as_ref())
    }
}

impl Body<'_, '_, '_> {
    /// Expand small local traversals selected by policy; retain a loop otherwise.
    pub(in crate::egglog::to_ssa) fn counted(
        &mut self,
        start: Typed,
        bound: Typed,
        step: Typed,
        initial: Vec<Typed>,
        mut emit: impl FnMut(&mut Self, Typed, Vec<Typed>) -> Result<Vec<Typed>, OptimizeError>,
    ) -> Result<Vec<Typed>, OptimizeError> {
        if let (Some(0), Some(n), Some(1)) = (
            self.integer_literal(&start),
            self.integer_literal(&bound),
            self.integer_literal(&step),
        ) {
            if self.compiler.facts.contains("ExpandLocalTripCount", (n,)) {
                let types: Vec<_> = initial.iter().map(|value| value.ty.clone()).collect();
                let mut state = initial;
                for i in 0..n {
                    let index = self.literal((i) as i64, &start.ty)?;
                    let values = self.values.checkpoint();
                    let scopes = self.scopes.checkpoint();
                    let scalar_values = std::mem::take(&mut self.scalar_values);
                    state = emit(self, index, state)?;
                    if state.len() != types.len() {
                        return Err(error("loop accumulator arity mismatch"));
                    }
                    state = state
                        .into_iter()
                        .zip(&types)
                        .map(|(value, ty)| self.cast(value, ty))
                        .collect::<Result<_, _>>()?;
                    self.values.restore(values);
                    self.scopes.restore(scopes);
                    self.scalar_values = scalar_values;
                }
                return Ok(state);
            }
        }
        self.counted_loop(start, bound, step, initial, emit)
    }

    // Source loops can contain arbitrarily large bodies, even with small trip counts.
    fn counted_loop(
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
    fn while_loop(
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
        let scopes = self.scopes.checkpoint();
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
        self.scopes.restore(scopes);
        Ok(results
            .into_iter()
            .zip(types)
            .map(|(value, ty)| Typed {
                value: value.into(),
                ty,
            })
            .collect())
    }

    fn integer_literal(&self, value: &Typed) -> Option<i64> {
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
                    _ => None,
                }
            }
            _ => None,
        }
    }
}

impl Body<'_, '_, '_> {
    pub(in crate::egglog::to_ssa) fn when(
        &mut self,
        condition: Typed,
        emit: impl FnOnce(&mut Self) -> Result<(), OptimizeError>,
    ) -> Result<(), OptimizeError> {
        self.branch_with_type(
            None,
            condition,
            |body| {
                emit(body)?;
                body.tuple(Vec::new())
            },
            |body, _| body.tuple(Vec::new()),
            None,
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
