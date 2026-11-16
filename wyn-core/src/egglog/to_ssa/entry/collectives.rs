use super::*;
use crate::egglog::cost::CollectiveWork;
use crate::egglog::to_ssa::kernels::CollectivePhase;
use egglog_engine::sort::Z;

impl EntryEmission<'_, '_, '_> {
    pub(super) fn chunked_collective(
        &mut self,
        statement: Value,
        after: &LookupSet<Value>,
        terminal: bool,
        epilogue: Option<Value>,
        preparation: &[Value],
    ) -> Result<bool, OptimizeError> {
        let query = self.body.query();
        let domain = query.required("OperationDomain", statement)?;
        let axes = super::super::domain_axes(query, domain)?;
        let dispatch = *self.body.compiler.execution.dispatch(statement)?;
        if axes.len() != 1 || dispatch.groups.is_some() {
            return Ok(false);
        }
        let axis = axes[0];
        if query
            .lookup("StaticIndex", axis.extent)?
            .is_some_and(|n| query.0.value_to_base::<i64>(n) <= i64::from(dispatch.width.get()))
        {
            return Ok(false);
        }
        let mut types = Vec::new();
        let mut reducer_names = LookupSet::default();
        let mut work = CollectiveWork {
            map_ops: region_work(query, query.required("ControlRegion", (statement, 0_i64))?)?,
            ..Default::default()
        };
        for clause in 0..crate::egglog::analysis::count(query, "ResultCount", statement)? {
            let kind = query.required("ResultKindOf", (statement, clause))?;
            let Some(fold) = query.enode("Folded", kind)? else {
                if query.enode("Collected", kind)?.is_some() {
                    continue;
                }
                return Ok(false);
            };
            if query.lookup("ResultPredicate", (statement, clause))?.is_some() {
                return Ok(false);
            }
            work.scan |= query.enode("Total", fold[0])?.is_none();
            let reducer = query.required("ResultReducer", (statement, clause))?;
            work.combine_ops = work.combine_ops.saturating_add(region_work(query, reducer)?);
            if query.contains("RegionTrap", reducer)? {
                return Ok(false);
            }
            let mut writes = false;
            query.for_each("RegionWrite", |row| {
                writes |= row[0] == reducer;
                Ok(())
            })?;
            if writes {
                return Ok(false);
            }
            reducer_names.extend(names_in(query, "RegionCapture", reducer)?);
            let params = query.sequence("RegionParameter", reducer)?;
            let count = crate::egglog::analysis::count(query, "ResultNeutralCount", (statement, clause))?;
            let neutral = query.indexed(
                "ResultNeutral",
                &[statement, query.0.base_to_value(clause)],
                count,
            )?;
            for (i, (&parameter, neutral)) in params.iter().zip(neutral).enumerate() {
                if let Some(name) = query.value_name(neutral)? {
                    reducer_names.insert(name);
                }
                let Some((_, ty)) = super::super::runtime_binding(query, parameter)? else {
                    return Ok(false);
                };
                let ty =
                    super::super::reducer_state_type(query, self.body.function, reducer, i, neutral, ty)?;
                let ty = storage_value_type(&ty);
                let Some(bytes) = storage_elem_stride(&ty) else {
                    return Ok(false);
                };
                work.state_bytes = work.state_bytes.saturating_add(u64::from(bytes));
                types.push(ty);
            }
        }
        if types.is_empty() {
            return Ok(false);
        }
        // Estimate contribution traffic at the reducer-state stride. Unknown
        // extents use the residency cap, clamped by the actual host extent.
        work.input_bytes = work.state_bytes;
        work.output_bytes = if work.scan { work.state_bytes } else { 0 };
        let selected_groups = if let Some(count) = query.lookup("StaticIndex", axis.extent)? {
            work.elements = query.0.value_to_base::<i64>(count).max(0) as u64;
            Some(self.body.compiler.program.costs.select(work, dispatch.width.get(), 256).map_err(error)?)
        } else {
            None
        };
        let pattern = query.sequence("Output", statement)?;
        let mut output_names = LookupSet::default();
        for &binding in &pattern {
            if let Some((name, _)) = super::super::runtime_binding(query, binding)? {
                output_names.insert(name);
            }
        }
        let mut inside = after.clone();
        inside.extend(output_names.iter().copied());
        inside.extend(reducer_names.iter().copied());
        inside.extend(domain_names(query, domain)?);
        let count = host::guarded_index(
            query,
            &self.body.host_values,
            axis.extent,
            self.body.guard.as_ref(),
        )?;
        let count = Expr::Max(Box::new(count), Box::new(Expr::Integer(0)));
        let groups = Expr::Max(
            Box::new(Expr::Integer(1)),
            Box::new(Expr::Min(
                Box::new(Expr::Integer(i64::from(selected_groups.unwrap_or(
                    self.body.compiler.program.costs.resident_workgroups.min(256),
                )))),
                Box::new(Expr::Ceiling(
                    Box::new(count.clone()),
                    Box::new(Expr::Integer(i64::from(dispatch.width.get()))),
                )),
            )),
        );
        let carry_width = selected_groups
            .map(|groups| groups.next_power_of_two().max(32).min(dispatch.width.get()))
            .unwrap_or(dispatch.width.get());
        let (group, group_count, range) = self.chunk_range(axis.extent)?;
        let mut partials = Vec::new();
        let mut carries = Vec::new();
        for ty in types {
            let Some(stride) = storage_elem_stride(&ty) else {
                return Err(error("collective state stride missing"));
            };
            let bytes = Expr::Multiply(Box::new(groups.clone()), Box::new(Expr::Integer(stride.into())));
            let (partial, _) = self.body.device_buffer(&ty, bytes.clone(), group_count.clone())?;
            partials.push((partial, ty.clone()));
            if work.scan {
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
            reducer_names.extend(after.iter().filter(|name| !output_names.contains(name)).copied());
        }
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
        if !work.scan && terminal {
            return Ok(true);
        }
        self.flush(&inside, vec![], true)?;

        if work.scan {
            self.body.active_dispatch = Some(statement);
            let (group, _, range) = self.chunk_range(axis.extent)?;
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
                self.scan_epilogue(epilogue, range, true, axis.extent)?;
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
        Ok(true)
    }

    pub(super) fn scan_epilogue(
        &mut self,
        statement: Value,
        range: (Typed, Typed),
        cooperative: bool,
        capacity: Value,
    ) -> Result<(), OptimizeError> {
        let pattern = self.body.query().sequence("Output", statement)?;
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

fn region_work(query: Query<'_>, region: Value) -> Result<u64, OptimizeError> {
    Ok(query
        .lookup("RegionWork", region)?
        .and_then(|work| u64::try_from(&*query.0.value_to_base::<Z>(work)).ok())
        .unwrap_or(65))
}
