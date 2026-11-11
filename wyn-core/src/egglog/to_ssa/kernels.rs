//! Parallel results share one domain traversal and one reducer calling convention.
use super::{error, Body, OptimizeError, Typed};
use crate::builtins::catalog;
use crate::egglog::to_ssa::runtime_binding;
use crate::op::{BinaryOperator, OpTag};
use crate::types;
use egglog_engine::{RawValues, Value};

enum FoldMode {
    Total,
    Scan {
        exclusive: bool,
    },
}

struct Fold {
    mode: FoldMode,
    reducer: Value,
    values: Value,
    neutral: Vec<Typed>,
    prefixes: Vec<Value>,
    totals: Vec<Value>,
}

impl Body<'_, '_, '_> {
    pub(super) fn parallel(
        &mut self,
        arguments: &[Value],
        _destinations: &[Value],
        pattern: &[Value],
    ) -> Result<Vec<Typed>, OptimizeError> {
        let dispatch = if self.query().enode("DeviceKernel", arguments[0])?.is_some() {
            let Some(statement) = self.active_dispatch else {
                return Err(error("device parallel region requires a dispatch entry"));
            };
            Some(*self.compiler.execution.dispatch(statement)?)
        } else {
            None
        };
        let domain = self.query().fields("IterationSpace", arguments[1])?;
        let axes = self.query().vector(domain[1])?;
        let body = self.query().fields("ParallelBlock", arguments[2])?;
        let mut bindings = pattern.iter().copied();
        let mut collects = Vec::new();
        let mut folds = Vec::new();
        let mut output_names = |count: usize| -> Result<Vec<Value>, OptimizeError> {
            (0..count)
                .map(|_| {
                    let Some(binding) = bindings.next() else {
                        return Err(error("parallel result arity mismatch"));
                    };
                    Ok(self.query().application(binding, &["BindValue", "BindAccumulator"])?.1[0])
                })
                .collect()
        };
        let mut fold_data = Vec::new();
        for result in self.query().vector(body[1])? {
            let (kind, fields) = self.query().application(result, &["Collect", "Fold", "Contribute"])?;
            match kind {
                "Collect" => {
                    let values = self.query().vector(fields[0])?;
                    for (name, value) in output_names(values.len())?.into_iter().zip(values) {
                        collects.push((name, value));
                    }
                }
                "Fold" => {
                    let (mode, _) = self.query().application(
                        fields[0],
                        &[
                            "Total",
                            "InclusiveScan",
                            "ExclusiveScan",
                            "InclusiveScanAndTotal",
                            "ExclusiveScanAndTotal",
                        ],
                    )?;
                    let reducer = self.query().fields("Reduction", fields[1])?;
                    let atoms = self.query().vector(reducer[1])?;
                    let prefixes = if mode == "Total" { Vec::new() } else { output_names(atoms.len())? };
                    let totals = if mode == "Total" || mode.ends_with("AndTotal") {
                        output_names(atoms.len())?
                    } else {
                        Vec::new()
                    };
                    let mode = if mode == "Total" {
                        FoldMode::Total
                    } else {
                        FoldMode::Scan {
                            exclusive: mode.starts_with("Exclusive"),
                        }
                    };
                    fold_data.push((mode, reducer[2], atoms, fields[2], prefixes, totals));
                }
                "Contribute" => {
                    let tokens = self.query().vector(fields[0])?;
                    output_names(tokens.len())?;
                }
                _ => unreachable!(),
            }
        }
        if bindings.next().is_some() {
            return Err(error("parallel result arity mismatch"));
        }
        for (mode, reducer, atoms, values, prefixes, totals) in fold_data {
            let neutral = atoms.into_iter().map(|atom| self.atom(atom)).collect::<Result<Vec<_>, _>>()?;
            folds.push(Fold {
                mode,
                reducer,
                values,
                neutral,
                prefixes,
                totals,
            });
        }
        if !folds.is_empty() && axes.is_empty() {
            return Err(error("fold has no innermost axis"));
        }
        if let Some(dispatch) = dispatch {
            let enabled = self
                .query()
                .lookup("SubgroupsEnabled", RawValues(Vec::new()))?
                .is_some_and(|value| self.query().0.value_to_base::<bool>(value));
            let native = !folds.is_empty()
                && folds.iter().try_fold(true, |native, fold| {
                    let addition = self
                        .query()
                        .lookup("NativeAddition", (fold.reducer,))?
                        .is_some_and(|value| self.query().0.value_to_base::<bool>(value));
                    Ok::<_, OptimizeError>(
                        native
                            && addition
                            && fold.neutral.iter().all(|value| self.integer_literal(value) == Some(0)),
                    )
                })?;
            if enabled && native {
                self.prepare_subgroups(dispatch.width.get())?;
            }
        }
        let outer = if folds.is_empty() { &axes[..] } else { &axes[..axes.len() - 1] };
        let partition = if let Some(dispatch) = dispatch {
            let (group, groups) = self.workgroups(dispatch.grid)?;
            if folds.is_empty() {
                let base = self.binary(
                    BinaryOperator::Multiply,
                    group,
                    Self::number(dispatch.width.get()),
                )?;
                let lane = self.local_id()?;
                let start = self.binary(BinaryOperator::Add, base, lane)?;
                let step = self.binary(
                    BinaryOperator::Multiply,
                    groups,
                    Self::number(dispatch.width.get()),
                )?;
                Some((start, step))
            } else {
                Some((group, groups))
            }
        } else {
            None
        };
        let mut initial = Vec::new();
        for fold in &folds {
            for (&name, value) in fold.totals.iter().zip(&fold.neutral) {
                let info = self.query().required("BindingInfo", (self.function, name))?;
                if self.query().enode("ScalarInfo", info)?.is_some() {
                    initial.push(value.clone());
                }
            }
        }
        let scalar_results = self.domain_axis(
            outer,
            0,
            domain[0],
            Vec::new(),
            Self::number(0),
            initial,
            partition,
            &mut |body_lower, coordinates, _| {
                if folds.is_empty() {
                    body_lower.parallel_lane(body[0], arguments[1], &collects, &folds, vec![])?;
                    return Ok(vec![]);
                }
                let outer_domain = body_lower.query().required("OuterDomain", (arguments[1],))?;
                let total_coordinates = body_lower.domain_coordinates(outer_domain)?;
                let ordinal = body_lower.named(domain[0])?;
                let state = if let Some(dispatch) = dispatch {
                    body_lower.cooperative_segment(
                        arguments[1],
                        axes[axes.len() - 1],
                        domain[0],
                        ordinal,
                        body[0],
                        &collects,
                        &folds,
                        dispatch.width.get(),
                    )?
                } else {
                    let initial = folds.iter().flat_map(|fold| fold.neutral.clone()).collect::<Vec<_>>();
                    body_lower.domain_axis(
                        &axes[axes.len() - 1..],
                        0,
                        domain[0],
                        coordinates.to_vec(),
                        ordinal,
                        initial,
                        None,
                        &mut |body_lower, _, state| {
                            body_lower.parallel_lane(body[0], arguments[1], &collects, &folds, state)
                        },
                    )?
                };
                let mut state = state.into_iter();
                let mut scalars = Vec::new();
                for fold in &folds {
                    let total = state.by_ref().take(fold.neutral.len()).collect::<Vec<_>>();
                    for (&name, value) in fold.totals.iter().zip(total) {
                        let info =
                            body_lower.query().required("BindingInfo", (body_lower.function, name))?;
                        if body_lower.query().enode("ArrayInfo", info)?.is_some() {
                            if dispatch.is_some() {
                                let lane = body_lower.local_id()?;
                                let leader =
                                    body_lower.binary(BinaryOperator::Equal, lane, Self::number(0))?;
                                body_lower.when(leader, |body| {
                                    body.write_array(name, &total_coordinates, value)
                                })?;
                            } else {
                                body_lower.write_array(name, &total_coordinates, value)?;
                            }
                        } else {
                            scalars.push(value);
                        }
                    }
                }
                Ok(scalars)
            },
        )?;
        let count = pattern.iter().try_fold(0, |count, &binding| {
            Ok::<_, OptimizeError>(count + usize::from(runtime_binding(self.query(), binding)?.is_some()))
        })?;
        if scalar_results.len() != count {
            return Err(error("parallel scalar result arity mismatch"));
        }
        Ok(scalar_results)
    }

    fn domain_coordinates(&mut self, domain: Value) -> Result<Vec<Typed>, OptimizeError> {
        let indices = self.query().required("DomainIndices", (domain,))?;
        self.query().vector(indices)?.into_iter().map(|index| self.index_expression(index)).collect()
    }

    fn parallel_lane(
        &mut self,
        statements: Value,
        domain: Value,
        collects: &[(Value, Value)],
        folds: &[Fold],
        state: Vec<Typed>,
    ) -> Result<Vec<Typed>, OptimizeError> {
        let coordinates = self.domain_coordinates(domain)?;
        self.statements(statements)?;
        for &(output, result) in collects {
            let yielded = self.query().fields("Yield", result)?;
            let value = self.query().fields("ValueOperand", yielded[1])?;
            self.write_atom(output, &coordinates, value[0])?;
        }
        let mut state = state.into_iter();
        let mut next = Vec::new();
        for fold in folds {
            let previous = state.by_ref().take(fold.neutral.len()).collect::<Vec<_>>();
            let contribution = self.results(fold.values)?;
            let mut args = previous.clone();
            args.extend(contribution);
            let combined = self.lambda(fold.reducer, args)?;
            let prefix = if matches!(fold.mode, FoldMode::Scan { exclusive: true, .. }) {
                &previous
            } else {
                &combined
            };
            for (&output, value) in fold.prefixes.iter().zip(prefix) {
                self.write_array(output, &coordinates, value.clone())?;
            }
            next.extend(combined);
        }
        Ok(next)
    }

    fn domain_axis(
        &mut self,
        axes: &[Value],
        depth: usize,
        flat: Value,
        coordinates: Vec<Typed>,
        ordinal: Typed,
        initial: Vec<Typed>,
        partition: Option<(Typed, Typed)>,
        emit: &mut impl FnMut(&mut Self, &[Typed], Vec<Typed>) -> Result<Vec<Typed>, OptimizeError>,
    ) -> Result<Vec<Typed>, OptimizeError> {
        if depth == 0 && !axes.is_empty() {
            if let Some((start, step)) = &partition {
                let dense = axes
                    .iter()
                    .map(|&axis| self.query().enode("DenseAxis", axis))
                    .collect::<Result<Option<Vec<_>>, _>>()?;
                if let Some(dense) = dense {
                    let shape = dense
                        .iter()
                        .map(|axis| self.index_expression(axis[1]))
                        .collect::<Result<Vec<_>, _>>()?;
                    let count = self.shape_count(&shape)?;
                    let base = self.binary(BinaryOperator::Multiply, ordinal, count.clone())?;
                    return self.counted(
                        start.clone(),
                        count,
                        step.clone(),
                        initial,
                        |body, index, state| {
                            let indices = body.coordinates(index.clone(), &shape)?;
                            for (axis, index) in dense.iter().zip(&indices) {
                                body.values.insert(axis[0], index.clone());
                            }
                            let ordinal = body.binary(BinaryOperator::Add, base, index)?;
                            body.values.insert(flat, ordinal);
                            let mut coordinates = coordinates;
                            coordinates.extend(indices);
                            emit(body, &coordinates, state)
                        },
                    );
                }
            }
        }
        let Some(&axis) = axes.get(depth) else {
            if let Some((start, step)) = partition {
                return self.counted(start, Self::number(1), step, initial, |body, _, state| {
                    body.values.insert(flat, ordinal);
                    emit(body, &coordinates, state)
                });
            }
            self.values.insert(flat, ordinal);
            return emit(self, &coordinates, initial);
        };
        let (name, base, count) = self.axis_bounds(axis, ordinal)?;
        let (start, step) = partition.unwrap_or_else(|| (Self::number(0), Self::number(1)));
        self.counted(start, count, step, initial, |body, index, state| {
            body.values.insert(name, index.clone());
            let ordinal = body.binary(BinaryOperator::Add, base, index.clone())?;
            let mut coordinates = coordinates;
            coordinates.push(index);
            body.domain_axis(axes, depth + 1, flat, coordinates, ordinal, state, None, emit)
        })
    }

    fn axis_bounds(&mut self, axis: Value, ordinal: Typed) -> Result<(Value, Typed, Typed), OptimizeError> {
        let (kind, fields) = self.query().application(axis, &["DenseAxis", "RaggedAxis"])?;
        let (base, count) = if kind == "DenseAxis" {
            let count = self.index_expression(fields[1])?;
            let base = self.binary(BinaryOperator::Multiply, ordinal, count.clone())?;
            (base, count)
        } else {
            let parent = self.index_expression(fields[2])?;
            let next = self.binary(BinaryOperator::Add, parent.clone(), Self::number(1))?;
            let start = self.read_array(fields[1], &[parent])?;
            let end = self.read_array(fields[1], &[next])?;
            let count = self.binary(BinaryOperator::Subtract, end, start.clone())?;
            (start, count)
        };
        Ok((fields[0], base, count))
    }

    pub(super) fn workgroups(
        &mut self,
        grid: Option<(u32, u32, u32)>,
    ) -> Result<(Typed, Typed), OptimizeError> {
        let uint = Self::number(0).ty;
        let coordinates = self.op(
            OpTag::Intrinsic {
                id: catalog().known().workgroup_id,
                overload_idx: 0,
            },
            vec![],
            types::vec(3, uint.clone()),
        )?;
        let mut group = self.field(coordinates.clone(), 0)?;
        let groups = if let Some((x, y, z)) = grid {
            for (axis, stride) in [(1, u64::from(x)), (2, u64::from(x) * u64::from(y))] {
                let coordinate = self.field(coordinates.clone(), axis)?;
                let stride = self.literal(
                    i64::try_from(stride).map_err(|_| error("dispatch stride exceeds i64"))?,
                    &uint,
                )?;
                let delta = self.binary(BinaryOperator::Multiply, coordinate, stride)?;
                group = self.binary(BinaryOperator::Add, group, delta)?;
            }
            let groups = u64::from(x)
                .checked_mul(u64::from(y))
                .and_then(|n| n.checked_mul(u64::from(z)))
                .and_then(|n| u32::try_from(n).ok());
            let Some(groups) = groups else {
                return Err(error("dispatch has too many workgroups"));
            };
            Self::number(groups)
        } else {
            self.op(
                OpTag::Intrinsic {
                    id: catalog().known().num_workgroups,
                    overload_idx: 0,
                },
                vec![],
                uint,
            )?
        };
        Ok((group, groups))
    }

    fn combine_folds(
        &mut self,
        folds: &[Fold],
        left: &[Typed],
        right: &[Typed],
    ) -> Result<Vec<Typed>, OptimizeError> {
        let mut offset = 0;
        let mut result = Vec::new();
        for fold in folds {
            let end = offset + fold.neutral.len();
            let mut args = left[offset..end].to_vec();
            args.extend_from_slice(&right[offset..end]);
            let values = self.lambda(fold.reducer, args)?;
            if values.len() != fold.neutral.len() {
                return Err(error("reducer changes its accumulator arity"));
            }
            result.extend(values);
            offset = end;
        }
        Ok(result)
    }

    fn cooperative_segment(
        &mut self,
        domain: Value,
        axis: Value,
        flat: Value,
        ordinal: Typed,
        statements: Value,
        collects: &[(Value, Value)],
        folds: &[Fold],
        width: u32,
    ) -> Result<Vec<Typed>, OptimizeError> {
        let (name, base, count) = self.axis_bounds(axis, ordinal)?;
        let lane = self.local_id()?;
        let initial = folds.iter().flat_map(|fold| fold.neutral.clone()).collect::<Vec<_>>();
        self.counted(
            Self::number(0),
            count.clone(),
            Self::number(width),
            initial.clone(),
            |body, tile, carried| {
                let index = body.binary(BinaryOperator::Add, tile, lane.clone())?;
                let active = body.binary(BinaryOperator::Less, index.clone(), count)?;
                body.values.insert(name, index.clone());
                let ordinal = body.binary(BinaryOperator::Add, base, index)?;
                body.values.insert(flat, ordinal);
                let coordinates = body.domain_coordinates(domain)?;
                let values = body.branch(
                    active.clone(),
                    |body| {
                        body.statements(statements)?;
                        for &(output, result) in collects {
                            let yielded = body.query().fields("Yield", result)?;
                            let value = body.query().fields("ValueOperand", yielded[1])?;
                            body.write_atom(output, &coordinates, value[0])?;
                        }
                        let mut values = Vec::new();
                        for fold in folds {
                            values.extend(body.results(fold.values)?);
                        }
                        body.tuple(values)
                    },
                    |body| body.tuple(initial.clone()),
                )?;
                let values = (0..initial.len())
                    .map(|i| body.field(values.clone(), i))
                    .collect::<Result<Vec<_>, _>>()?;
                let (exclusive, totals) = body.workgroup_fold(folds, &values, lane.clone(), width)?;
                let inclusive = body.combine_folds(folds, &exclusive, &values)?;
                let exclusive = body.combine_folds(folds, &carried, &exclusive)?;
                let inclusive = body.combine_folds(folds, &carried, &inclusive)?;
                body.when(active, |body| {
                    let mut offset = 0;
                    for fold in folds {
                        let values = if matches!(fold.mode, FoldMode::Scan { exclusive: true }) {
                            &exclusive
                        } else {
                            &inclusive
                        };
                        for (&destination, value) in fold.prefixes.iter().zip(&values[offset..]) {
                            body.write_array(destination, &coordinates, value.clone())?;
                        }
                        offset += fold.neutral.len();
                    }
                    Ok(())
                })?;
                body.combine_folds(folds, &carried, &totals)
            },
        )
    }

    fn workgroup_fold(
        &mut self,
        _folds: &[Fold],
        values: &[Typed],
        _lane: Typed,
        width: u32,
    ) -> Result<(Vec<Typed>, Vec<Typed>), OptimizeError> {
        let (_, exclusive, totals) = self.subgroup_scan(values.to_vec(), width)?;
        Ok((exclusive, totals))
    }
}
