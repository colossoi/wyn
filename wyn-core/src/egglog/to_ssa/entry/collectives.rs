use super::*;
use crate::egglog::to_ssa::kernels::CollectivePhase;

impl EntryEmission<'_, '_, '_> {
    pub(super) fn chunked_collective(
        &mut self,
        statement: Value,
        after: &LookupSet<Value>,
        terminal: bool,
    ) -> Result<bool, OptimizeError> {
        let query = self.body.query();
        let statement_fields = query.fields("Let", statement)?;
        let arguments = query.fields("Parallel", statement_fields[3])?;
        let domain = query.fields("IterationSpace", arguments[1])?;
        let axes = query.vector(domain[1])?;
        let dispatch = *self.body.compiler.execution.dispatch(statement)?;
        if axes.len() != 1 || dispatch.groups.is_some() {
            return Ok(false);
        }
        let axis = query.fields("DenseAxis", axes[0])?;
        if query
            .lookup("StaticIndex", (self.body.function, axis[1]))?
            .is_some_and(|n| query.0.value_to_base::<i64>(n) <= i64::from(dispatch.width.get()))
        {
            return Ok(false);
        }
        let block = query.fields("ParallelBlock", arguments[2])?;
        let mut types = Vec::new();
        let mut reducer_names = LookupSet::default();
        let mut prefixes = false;
        for clause in query.vector(block[1])? {
            let Some(fold) = query.enode("Fold", clause)? else {
                if query.enode("Collect", clause)?.is_some() {
                    continue;
                }
                return Ok(false);
            };
            prefixes |= query.enode("Total", fold[0])?.is_none();
            let reducer = query.fields("Reduction", fold[1])?;
            let effects = query.required("EffectsLambda", (self.body.function, reducer[2]))?;
            if !query.contains("ReadOnlyEffects", (effects,))? {
                return Ok(false);
            }
            let used = names_in(query, "NamesReducer", fold[1])?;
            let bound = names_in(query, "BindersReducer", fold[1])?;
            reducer_names.extend(used.difference(&bound).copied());
            let lambda = query.fields("FunctionBody", reducer[2])?;
            let params = query.vector(lambda[0])?;
            for (i, (&parameter, neutral)) in params.iter().zip(query.vector(reducer[1])?).enumerate() {
                let parameter = query.fields("Parameter", parameter)?;
                let Some((_, ty)) = super::super::runtime_binding(query, parameter[0])? else {
                    return Ok(false);
                };
                let ty = super::super::reducer_state_type(
                    query,
                    self.body.function,
                    reducer[2],
                    i,
                    neutral,
                    ty,
                )?;
                let ty = storage_value_type(&ty);
                if storage_elem_stride(&ty).is_none() {
                    return Ok(false);
                }
                types.push(ty);
            }
        }
        if types.is_empty() {
            return Ok(false);
        }
        let pattern = query.vector(statement_fields[1])?;
        let mut output_names = LookupSet::default();
        for &binding in &pattern {
            if let Some((name, _)) = super::super::runtime_binding(query, binding)? {
                output_names.insert(name);
            }
        }
        let mut inside = after.clone();
        inside.extend(output_names.iter().copied());
        inside.extend(reducer_names.iter().copied());
        inside.extend(names_in(query, "NamesDomain", arguments[1])?);
        let count = host::guarded_index(query, &self.body.host_values, axis[1], self.body.guard.as_ref())?;
        let count = Expr::Max(Box::new(count), Box::new(Expr::Integer(0)));
        let groups = Expr::Max(
            Box::new(Expr::Integer(1)),
            Box::new(Expr::Min(
                Box::new(Expr::Integer(256)),
                Box::new(Expr::Ceiling(
                    Box::new(count.clone()),
                    Box::new(Expr::Integer(i64::from(dispatch.width.get()))),
                )),
            )),
        );
        let carry_width = query
            .lookup("StaticIndex", (self.body.function, axis[1]))?
            .map(|count| {
                let count = query.0.value_to_base::<i64>(count).max(0) as u64;
                let groups = count.div_ceil(u64::from(dispatch.width.get())).clamp(1, 256) as u32;
                groups.next_power_of_two().max(32).min(dispatch.width.get())
            })
            .unwrap_or(dispatch.width.get());
        let (group, group_count, range) = self.chunk_range(axis[1])?;
        let mut partials = Vec::new();
        let mut carries = Vec::new();
        for ty in types {
            let Some(stride) = storage_elem_stride(&ty) else {
                return Err(error("collective state stride missing"));
            };
            let bytes = Expr::Multiply(Box::new(groups.clone()), Box::new(Expr::Integer(stride.into())));
            let (partial, _) = self.body.device_buffer(&ty, bytes.clone(), group_count.clone())?;
            partials.push((partial, ty.clone()));
            if prefixes {
                let (carry, _) = self.body.device_buffer(&ty, bytes, group_count.clone())?;
                carries.push((carry, ty));
            }
        }
        let values = self.body.parallel_phase(
            &arguments,
            &pattern,
            Some(CollectivePhase::Chunks {
                range,
                group,
                partials: &partials,
            }),
        )?;
        self.body.bind(&pattern, values)?;
        self.pending = true;
        self.restore_names = Some(reducer_names);
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
            &arguments,
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
        if !prefixes && terminal {
            return Ok(true);
        }
        self.flush(&inside, vec![], true)?;

        if prefixes {
            self.body.active_dispatch = Some(statement);
            let (group, _, range) = self.chunk_range(axis[1])?;
            self.body.parallel_phase(
                &arguments,
                &pattern,
                Some(CollectivePhase::Apply {
                    range,
                    group,
                    carries: &carries,
                }),
            )?;
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
        Ok(true)
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
