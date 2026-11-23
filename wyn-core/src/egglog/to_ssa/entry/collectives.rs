use super::{
    domain_names, error, host, names_in, statement_names, Body, EntryEmission, OptimizeError, Typed,
};
use crate::builtins::catalog;
use crate::egglog::execution::Collective;
use crate::egglog::graph::CollectiveInput;
use crate::egglog::to_ssa::kernels::CollectivePhase;
use crate::egglog::to_ssa::runtime_binding;
use crate::host::{DispatchSize, Expr};
use crate::op::{BinaryOperator, OpTag};
use crate::ssa::layout::storage_elem_stride;
use crate::{BindingRef, LookupSet};
use egglog_engine::Value;

impl EntryEmission<'_, '_, '_> {
    pub(super) fn emit_collective(
        &mut self,
        statement: Value,
        plan: &Collective,
        after: &LookupSet<Value>,
        terminal: bool,
        epilogue: Option<Value>,
        preparation: &[Value],
    ) -> Result<(), OptimizeError> {
        let query = self.body.query();
        let dispatch = *self.body.compiler.execution.dispatch(statement)?;
        let domain = query.required("OperationDomain", statement)?;
        let mut carry_inputs = LookupSet::default();
        for clause in 0..query.0.result_count(statement) {
            if let Some(reducer) = query.0.reducer_at(statement, clause) {
                carry_inputs.extend(names_in(query, "RegionCapture", reducer)?);
                for slot in 0..query.0.collective_input_count(statement, clause, CollectiveInput::Neutral) {
                    if let Some(name) = query.value_name(query.0.collective_input(
                        statement,
                        clause,
                        CollectiveInput::Neutral,
                        slot,
                    )?)? {
                        carry_inputs.insert(name);
                    }
                }
            }
        }
        let pattern = query.0.outputs(statement);
        let mut output_names = LookupSet::default();
        for (slot, &binding) in pattern.iter().enumerate() {
            if let Some((name, _)) = runtime_binding(query, binding)? {
                output_names.insert(name);
                // The final carry dispatch can use arrays already published by the chunk dispatch.
                if terminal && !plan.scan && after.contains(&name) {
                    let role = query.0.output_role(statement, slot as i64)?;
                    if query.enode("CollectedOutput", role)?.is_some() {
                        carry_inputs.insert(name);
                    }
                }
            }
        }
        let mut inside = after.clone();
        inside.extend(output_names.iter().copied());
        inside.extend(carry_inputs.iter().copied());
        inside.extend(domain_names(query, domain)?);
        if let Some(epilogue) = epilogue {
            inside.extend(statement_names(query, preparation)?);
            inside.extend(names_in(query, "OperationUse", epilogue)?);
        }
        let count = host::guarded_index(
            query,
            &self.body.host_values,
            plan.extent,
            self.body.guard.as_ref(),
        )?;
        let count = Expr::Max(Box::new(count), Box::new(Expr::Integer(0)));
        let groups = Expr::Max(
            Box::new(Expr::Integer(1)),
            Box::new(Expr::Min(
                Box::new(Expr::Integer(i64::from(plan.group_limit))),
                Box::new(Expr::Ceiling(
                    Box::new(count.clone()),
                    Box::new(Expr::Integer(i64::from(dispatch.width.get()))),
                )),
            )),
        );
        let carry_width = plan.carry_width;
        let (group, group_count, range) = self.chunk_range(plan.extent)?;
        let mut partials = Vec::new();
        let mut carries = Vec::new();
        for ty in plan.types.clone() {
            let Some(stride) = storage_elem_stride(&ty) else {
                return Err(error("collective state stride missing"));
            };
            let bytes = Expr::Multiply(Box::new(groups.clone()), Box::new(Expr::Integer(stride.into())));
            let (partial, _) = self.body.device_buffer(&ty, bytes.clone(), group_count.clone())?;
            partials.push((partial, ty.clone()));
            if plan.scan {
                let (carry, _) = self.body.device_buffer(&ty, bytes, group_count.clone())?;
                carries.push((carry, ty));
            }
        }
        let values = self.body.parallel_phase(
            statement,
            &pattern,
            Some(CollectivePhase::Chunks {
                range,
                group,
                partials: &partials,
            }),
        )?;
        self.body.bind(&pattern, values)?;
        self.pending = true;
        if terminal {
            carry_inputs.extend(after.iter().filter(|name| !output_names.contains(name)).copied());
        }
        self.restore_names = Some(carry_inputs);
        self.dispatch_override = Some((
            dispatch.local_size,
            DispatchSize::Computed {
                elements: count.clone(),
                groups: [groups.clone(), Expr::Integer(1), Expr::Integer(1)],
            },
        ));
        self.flush(&inside, vec![], true)?;

        self.body.active_dispatch = Some(statement);
        let group_count = self.buffer_elements(partials[0].0)?;
        self.body.parallel_phase(
            statement,
            &pattern,
            Some(CollectivePhase::Carries {
                partials: &partials,
                carries: &carries,
                groups: group_count,
                width: carry_width,
            }),
        )?;
        self.pending = true;
        self.dispatch_override = Some((
            (carry_width, 1, 1),
            DispatchSize::Fixed {
                x: 1,
                y: 1,
                z: 1,
                explicit: true,
            },
        ));
        if !plan.scan && terminal {
            return Ok(());
        }
        self.flush(&inside, vec![], true)?;

        if plan.scan {
            self.body.active_dispatch = Some(statement);
            let (group, _, range) = self.chunk_range(plan.extent)?;
            self.body.parallel_phase(
                statement,
                &pattern,
                Some(CollectivePhase::Apply {
                    range: range.clone(),
                    group,
                    carries: &carries,
                }),
            )?;
            if let Some(epilogue) = epilogue {
                self.body.statements(preparation)?;
                self.scan_epilogue(epilogue, range, true, plan.extent)?;
            }
            self.pending = true;
            self.dispatch_override = Some((
                dispatch.local_size,
                DispatchSize::Computed {
                    elements: count,
                    groups: [groups, Expr::Integer(1), Expr::Integer(1)],
                },
            ));
            let mut completed = after.clone();
            completed.extend(output_names);
            if !terminal {
                self.flush(&completed, vec![], true)?;
            }
        }
        Ok(())
    }

    pub(super) fn scan_epilogue(
        &mut self,
        statement: Value,
        range: (Typed, Typed),
        cooperative: bool,
        capacity: Value,
    ) -> Result<(), OptimizeError> {
        let pattern = self.body.query().0.outputs(statement);
        let values = self.body.parallel_phase(
            statement,
            &pattern,
            Some(CollectivePhase::Epilogue {
                range,
                cooperative,
                capacity,
            }),
        )?;
        self.body.bind(&pattern, values)
    }

    fn buffer_elements(&mut self, binding: BindingRef) -> Result<Typed, OptimizeError> {
        self.body.op(
            OpTag::Intrinsic {
                id: catalog().known().storage_len,
                overload_idx: 0,
            },
            vec![Body::number(binding.set), Body::number(binding.binding)],
            Body::number(0).ty,
        )
    }

    fn chunk_range(&mut self, count: Value) -> Result<(Typed, Typed, (Typed, Typed)), OptimizeError> {
        let count = self.body.index_expression(count)?;
        let zero = self.body.literal(0, &count.ty)?;
        let positive = self.body.binary(BinaryOperator::Greater, count.clone(), zero.clone())?;
        let count = self.body.select(positive, count, zero)?;
        let count = self.body.cast(count, &Body::number(0).ty)?;
        let (group, groups) = self.body.workgroups(None)?;
        let quotient = self.body.binary(BinaryOperator::Divide, count.clone(), groups.clone())?;
        let remainder = self.body.binary(BinaryOperator::Remainder, count, groups.clone())?;
        let base = self.body.binary(BinaryOperator::Multiply, group.clone(), quotient.clone())?;
        let before = self.body.binary(BinaryOperator::Less, group.clone(), remainder.clone())?;
        let offset = self.body.select(before.clone(), group.clone(), remainder)?;
        let start = self.body.binary(BinaryOperator::Add, base, offset)?;
        let extra = self.body.cast(before, &Body::number(0).ty)?;
        let length = self.body.binary(BinaryOperator::Add, quotient, extra)?;
        let end = self.body.binary(BinaryOperator::Add, start.clone(), length)?;
        Ok((group, groups, (start, end)))
    }
}
