use super::{builder_error, error, Body, OptimizeError, Term, TermKind, Typed, Value};
use crate::flow::ControlHeader;
use crate::op::BinaryOperator;
use crate::ssa::types::InstKind;
use crate::ssa::types::Terminator;
use crate::tlc::LoopKind;
use crate::types::{self, Type, TypeExt, TypeName};

impl Body<'_, '_, '_> {
    pub(in crate::egglog::to_ssa) fn branch(
        &mut self,
        scope: Value,
        condition: Typed,
        yes_value: impl FnOnce(&mut Self) -> Result<Typed, OptimizeError>,
        no_value: impl FnOnce(&mut Self) -> Result<Typed, OptimizeError>,
        source_scopes: Option<(Value, Value)>,
    ) -> Result<Typed, OptimizeError> {
        let start = self.current()?;
        self.scopes.insert(scope, start);
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
        let b = no_value(self)?;
        let no_end = self.current()?;
        let ty = merge_type(&a.ty, &b.ty);
        self.builder.switch_to_block_unchecked(yes_end);
        let a = self.stored(a, &ty)?;
        self.terminate(Terminator::Branch {
            target: end,
            args: vec![a.value],
        })
        .map_err(builder_error)?;
        self.builder.switch_to_block_unchecked(no_end);
        let b = self.stored(b, &ty)?;
        self.terminate(Terminator::Branch {
            target: end,
            args: vec![b.value],
        })
        .map_err(builder_error)?;
        let result = self.builder.add_block_param(end, a.ty.clone());
        self.values.restore(outer_values);
        self.scopes.restore(outer_scopes);
        self.builder.switch_to_block_unchecked(end);
        self.scopes.insert(scope, end);
        Ok(Typed {
            value: result.into(),
            ty: a.ty,
        })
    }
}

impl<'source> Body<'_, '_, 'source> {
    pub(super) fn loop_(
        &mut self,
        scope: Value,
        source: Value,
        term: &'source Term,
    ) -> Result<Typed, OptimizeError> {
        let TermKind::Loop {
            init,
            init_bindings,
            kind,
            body,
            ..
        } = &term.kind
        else {
            return Err(error("loop source is not a loop"));
        };
        let initial = self.source(scope, init)?;
        // Fixed local accumulator arrays have value semantics, including a
        // branch that keeps the preceding iteration unchanged.
        let ty = local_state_type(&initial.ty);
        let initial = self.stored(initial, &ty)?;
        let Some((header_scope, iteration_scope)) = self.compiler.facts.loops(source) else {
            return Err(error("missing loop scopes"));
        };
        let Some(state) = self.compiler.facts.loop_state(header_scope) else {
            return Err(error("missing loop accumulator"));
        };
        let (bound, array, index_ty) = match kind {
            LoopKind::ForRange { bound, var_ty, .. } => {
                (Some(self.source(scope, bound)?), None, var_ty.clone())
            }
            LoopKind::For { iter, var_ty: _, .. } => {
                let array = self.source(scope, iter)?;
                (
                    Some(self.length(array.clone())?),
                    Some(array),
                    Type::Constructed(TypeName::UInt(32), vec![]),
                )
            }
            LoopKind::While { .. } => (None, None, Type::Constructed(TypeName::UInt(32), vec![])),
        };
        let zero = self.literal("0", &index_ty)?;
        self.scopes.insert(scope, self.current()?);
        let outer_values = self.values.checkpoint();
        let outer_scopes = self.scopes.checkpoint();
        let (header, params) =
            self.builder.create_block_with_params(vec![initial.ty.clone(), index_ty.clone()]);
        let iteration = self.builder.create_block();
        let continuing = self.builder.create_block();
        let (end, result) = self.builder.create_block_with_params(vec![initial.ty.clone()]);
        self.terminate(Terminator::Branch {
            target: header,
            args: vec![initial.value, zero.value],
        })
        .map_err(builder_error)?;
        self.builder.switch_to_block_unchecked(header);
        self.builder.set_control_header(
            header,
            ControlHeader::Loop {
                merge: end,
                continue_block: continuing,
            },
        );
        self.scopes.insert(header_scope, header);
        self.values.insert(
            state,
            Typed {
                value: params[0].into(),
                ty: initial.ty.clone(),
            },
        );
        let index = Typed {
            value: params[1].into(),
            ty: index_ty.clone(),
        };
        let condition_block = self.builder.create_block();
        self.terminate(Terminator::Branch {
            target: condition_block,
            args: vec![],
        })
        .map_err(builder_error)?;
        self.builder.switch_to_block_unchecked(condition_block);
        for (_, _, term) in init_bindings {
            self.source(header_scope, term)?;
        }
        let condition = match kind {
            LoopKind::While { cond } => self.source(header_scope, cond)?,
            _ => {
                let Some(bound) = bound else {
                    return Err(error("counted loop has no bound"));
                };
                self.binary(BinaryOperator::Less, index.clone(), bound)?
            }
        };
        self.terminate(Terminator::CondBranch {
            cond: condition.value,
            then_target: iteration,
            then_args: vec![],
            else_target: end,
            else_args: vec![params[0].into()],
        })
        .map_err(builder_error)?;
        self.builder.switch_to_block_unchecked(iteration);
        self.scopes.insert(iteration_scope, iteration);
        if let Some(parameter) = self.compiler.facts.iteration(iteration_scope) {
            let value =
                if let Some(array) = array { self.index(array, index.clone())? } else { index.clone() };
            self.values.insert(parameter, value);
        }
        let next = self.source(iteration_scope, body)?;
        let next = self.cast(next, &initial.ty)?;
        self.terminate(Terminator::Branch {
            target: continuing,
            args: vec![],
        })
        .map_err(builder_error)?;
        self.builder.switch_to_block_unchecked(continuing);
        let one = self.literal("1", &index_ty)?;
        let index = self.binary(BinaryOperator::Add, index, one)?;
        self.terminate(Terminator::Branch {
            target: header,
            args: vec![next.value, index.value],
        })
        .map_err(builder_error)?;
        self.values.restore(outer_values);
        self.scopes.restore(outer_scopes);
        self.builder.switch_to_block_unchecked(end);
        self.scopes.insert(scope, end);
        Ok(Typed {
            value: result[0].into(),
            ty: initial.ty,
        })
    }
}

impl Body<'_, '_, '_> {
    pub(in crate::egglog::to_ssa) fn copy_array(
        &mut self,
        destination: Typed,
        source: Typed,
        length: Typed,
    ) -> Result<(), OptimizeError> {
        let index_ty = length.ty.clone();
        let zero = self.literal("0", &index_ty)?;
        let (header, indices) = self.builder.create_block_with_params(vec![index_ty.clone()]);
        let body = self.builder.create_block();
        let continuing = self.builder.create_block();
        let done = self.builder.create_block();
        self.terminate(Terminator::Branch {
            target: header,
            args: vec![zero.value],
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
        let index = Typed {
            value: indices[0].into(),
            ty: index_ty.clone(),
        };
        let condition = self.binary(BinaryOperator::Less, index.clone(), length)?;
        self.terminate(Terminator::CondBranch {
            cond: condition.value,
            then_target: body,
            then_args: vec![],
            else_target: done,
            else_args: vec![],
        })
        .map_err(builder_error)?;
        self.builder.switch_to_block_unchecked(body);
        let value = self.index(source, index.clone())?;
        let (place, ty) = self.index_place(destination, index.clone())?;
        let value = self.cast(value, &ty)?;
        self.builder
            .push_void_inst(InstKind::Store {
                place,
                value: value.value,
            })
            .map_err(builder_error)?;
        self.terminate(Terminator::Branch {
            target: continuing,
            args: vec![],
        })
        .map_err(builder_error)?;
        self.builder.switch_to_block_unchecked(continuing);
        let one = self.literal("1", &index_ty)?;
        let next = self.binary(BinaryOperator::Add, index, one)?;
        self.terminate(Terminator::Branch {
            target: header,
            args: vec![next.value],
        })
        .map_err(builder_error)?;
        self.builder.switch_to_block_unchecked(done);
        Ok(())
    }
}

impl Body<'_, '_, '_> {
    /// Emit a structured counted loop, carrying only its SSA accumulator values.
    pub(in crate::egglog::to_ssa) fn counted(
        &mut self,
        start: Typed,
        bound: Typed,
        step: Typed,
        initial: Vec<Typed>,
        emit: impl FnOnce(&mut Self, Typed, Vec<Typed>) -> Result<Vec<Typed>, OptimizeError>,
    ) -> Result<Vec<Typed>, OptimizeError> {
        let bound = self.cast(bound, &start.ty)?;
        let mut types = vec![start.ty.clone()];
        types.extend(initial.iter().map(|v| v.ty.clone()));
        let (header, parameters) = self.builder.create_block_with_params(types.clone());
        let body = self.builder.create_block();
        let continuing = self.builder.create_block();
        let (done, results) = self.builder.create_block_with_params(types[1..].to_vec());
        self.terminate(Terminator::Branch {
            target: header,
            args: std::iter::once(start.value).chain(initial.iter().map(|v| v.value)).collect(),
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
        let index = Typed {
            value: parameters[0].into(),
            ty: start.ty,
        };
        let state: Vec<_> = parameters[1..]
            .iter()
            .zip(&types[1..])
            .map(|(&v, ty)| Typed {
                value: v.into(),
                ty: ty.clone(),
            })
            .collect();
        let condition = self.binary(BinaryOperator::Less, index.clone(), bound)?;
        self.terminate(Terminator::CondBranch {
            cond: condition.value,
            then_target: body,
            then_args: vec![],
            else_target: done,
            else_args: state.iter().map(|v| v.value).collect(),
        })
        .map_err(builder_error)?;
        self.builder.switch_to_block_unchecked(body);
        let outer_values = self.values.checkpoint();
        let outer_scopes = self.scopes.checkpoint();
        let next = emit(self, index.clone(), state)?;
        if next.len() != initial.len() {
            return Err(error("loop accumulator arity mismatch"));
        }
        let mut args = Vec::new();
        for (value, ty) in next.into_iter().zip(&types[1..]) {
            args.push(self.cast(value, ty)?.value);
        }
        self.terminate(Terminator::Branch {
            target: continuing,
            args: vec![],
        })
        .map_err(builder_error)?;
        self.builder.switch_to_block_unchecked(continuing);
        let next_index = self.binary(BinaryOperator::Add, index, step)?;
        args.insert(0, next_index.value);
        self.terminate(Terminator::Branch { target: header, args }).map_err(builder_error)?;
        self.builder.switch_to_block_unchecked(done);
        self.values.restore(outer_values);
        self.scopes.restore(outer_scopes);
        Ok(results
            .into_iter()
            .zip(&types[1..])
            .map(|(v, ty)| Typed {
                value: v.into(),
                ty: ty.clone(),
            })
            .collect())
    }
}

impl Body<'_, '_, '_> {
    pub(in crate::egglog::to_ssa) fn when(
        &mut self,
        condition: Typed,
        emit: impl FnOnce(&mut Self) -> Result<(), OptimizeError>,
    ) -> Result<(), OptimizeError> {
        let start = self.current()?;
        let yes = self.builder.create_block();
        let done = self.builder.create_block();
        self.builder.set_control_header(start, ControlHeader::Selection { merge: done });
        self.terminate(Terminator::CondBranch {
            cond: condition.value,
            then_target: yes,
            then_args: vec![],
            else_target: done,
            else_args: vec![],
        })
        .map_err(builder_error)?;
        let values = self.values.checkpoint();
        let scopes = self.scopes.checkpoint();
        self.builder.switch_to_block_unchecked(yes);
        emit(self)?;
        self.terminate(Terminator::Branch {
            target: done,
            args: vec![],
        })
        .map_err(builder_error)?;
        self.builder.switch_to_block_unchecked(done);
        self.values.restore(values);
        self.scopes.restore(scopes);
        Ok(())
    }
}

impl Body<'_, '_, '_> {
    /// Retry an atomic update until its `(observed, exchanged)` result succeeds.
    pub(in crate::egglog::to_ssa) fn retry(
        &mut self,
        initial: Typed,
        emit: impl FnOnce(&mut Self, Typed) -> Result<Typed, OptimizeError>,
    ) -> Result<(), OptimizeError> {
        let (header, parameters) = self.builder.create_block_with_params(vec![initial.ty.clone()]);
        let active = self.builder.create_block();
        let continuing = self.builder.create_block();
        let done = self.builder.create_block();
        self.terminate(Terminator::Branch {
            target: header,
            args: vec![initial.value],
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
        let state = Typed {
            value: parameters[0].into(),
            ty: initial.ty,
        };
        let exchanged = self.field(state.clone(), 1)?;
        self.terminate(Terminator::CondBranch {
            cond: exchanged.value,
            then_target: done,
            then_args: vec![],
            else_target: active,
            else_args: vec![],
        })
        .map_err(builder_error)?;
        self.builder.switch_to_block_unchecked(active);
        let next = emit(self, state)?;
        self.terminate(Terminator::Branch {
            target: continuing,
            args: vec![],
        })
        .map_err(builder_error)?;
        self.builder.switch_to_block_unchecked(continuing);
        self.terminate(Terminator::Branch {
            target: header,
            args: vec![next.value],
        })
        .map_err(builder_error)?;
        self.builder.switch_to_block_unchecked(done);
        Ok(())
    }
}

pub(super) fn local_state_type(ty: &Type) -> Type {
    if let (Some(Type::Constructed(TypeName::Size(n), _)), Some(element)) =
        (ty.array_size(), ty.elem_type())
    {
        return types::sized_array((*n).max(1), local_state_type(element));
    }
    if ty.is_array() {
        return ty.clone();
    }
    match ty {
        Type::Constructed(name, fields) => {
            Type::Constructed(name.clone(), fields.iter().map(local_state_type).collect())
        }
        _ => ty.clone(),
    }
}

fn merge_type(a: &Type, b: &Type) -> Type {
    if a == b {
        return a.clone();
    }
    if a.is_array() && b.is_array() {
        if !a.array_variant().is_some_and(types::is_array_variant_view) {
            return a.clone();
        }
        if !b.array_variant().is_some_and(types::is_array_variant_view) {
            return b.clone();
        }
        return local_state_type(a);
    }
    if let (Type::Constructed(name, fields), Type::Constructed(_, other)) = (a, b) {
        if matches!(name, TypeName::Tuple(_) | TypeName::Record(_)) && fields.len() == other.len() {
            return Type::Constructed(
                name.clone(),
                fields.iter().zip(other).map(|(a, b)| merge_type(a, b)).collect(),
            );
        }
    }
    a.clone()
}
