//! Parallel results share one domain traversal and one reducer calling convention.
use super::body::{Collection, CollectionShape};
use super::{domain_axes, error, Axis, Body, OptimizeError, Typed};
use crate::builtins::catalog;
use crate::egglog::execution::Dispatch;
use crate::egglog::to_ssa::host;
use crate::egglog::to_ssa::runtime_binding;
use crate::host::Expr;
use crate::op::{BinaryOperator, OpTag, PureViewSource};
use crate::ssa::layout::{storage_elem_stride, storage_value_type};
use crate::ssa::types::{BlockId, InstKind};
use crate::types::{self, Type, TypeExt, TypeName};
use crate::BindingRef;
use crate::LookupSet;
use egglog_engine::Value;

enum FoldMode {
    Total,
    Scan {
        exclusive: bool,
    },
}

struct Fold {
    mode: FoldMode,
    reducer: Value,
    values: Vec<Value>,
    neutral: Vec<Typed>,
    prefixes: Vec<Value>,
    totals: Vec<Value>,
    compact: Option<usize>,
}

// Physical SSA operands only; callback bodies remain in the common IR.
pub(super) enum CollectivePhase<'a> {
    Epilogue {
        range: (Typed, Typed),
        cooperative: bool,
        capacity: Value,
    },
    Chunks {
        range: (Typed, Typed),
        group: Typed,
        partials: &'a [(BindingRef, Type)],
    },
    Carries {
        partials: &'a [(BindingRef, Type)],
        carries: &'a [(BindingRef, Type)],
        groups: Typed,
        width: u32,
    },
    Apply {
        range: (Typed, Typed),
        group: Typed,
        carries: &'a [(BindingRef, Type)],
    },
}

fn collection_shape(ty: &Type, dimensions: &[Option<usize>]) -> Result<Type, OptimizeError> {
    let Some((size, rest)) = dimensions.split_first() else {
        return Ok(ty.clone());
    };
    if let Type::Constructed(name @ (TypeName::Tuple(_) | TypeName::Record(_)), fields) = ty {
        return Ok(Type::Constructed(
            name.clone(),
            fields.iter().map(|field| collection_shape(field, dimensions)).collect::<Result<_, _>>()?,
        ));
    }
    let Some(element) = ty.elem_type() else {
        return Err(error("collection shape exceeds array rank"));
    };
    let element = collection_shape(element, rest)?;
    Ok(if let Some(size) = size {
        types::sized_array(*size, element)
    } else {
        types::make_array1(
            element,
            types::array_variant_composite(),
            ty.array_size().cloned().ok_or_else(|| error("collection dimension missing"))?,
            types::no_buffer(),
        )
    })
}

impl Body<'_, '_, '_> {
    pub(super) fn parallel(
        &mut self,
        operation: Value,
        pattern: &[Value],
    ) -> Result<Vec<Typed>, OptimizeError> {
        self.parallel_phase(operation, pattern, None)
    }

    pub(super) fn parallel_phase(
        &mut self,
        operation: Value,
        pattern: &[Value],
        phase: Option<CollectivePhase<'_>>,
    ) -> Result<Vec<Typed>, OptimizeError> {
        if let Some(plan) = self.query().lookup("ColumnScan", operation)? {
            return self.scan_columns(operation, pattern, plan);
        }
        let execution = self.query().required("OperationExecution", operation)?;
        let dispatch = if self.query().enode("DeviceKernel", execution)?.is_some() {
            let Some(statement) = self.active_dispatch else {
                return Err(error("device parallel region requires a dispatch entry"));
            };
            Some(*self.compiler.execution.dispatch(statement)?)
        } else {
            None
        };
        let workgroup_map = dispatch.is_some() && self.query().flag("WorkgroupMap", operation)?;
        let cooperative_width = if let Some(dispatch) = dispatch {
            Some(dispatch.width.get())
        } else if self.query().contains("WorkgroupCollective", operation)? {
            let parent =
                self.active_dispatch.ok_or_else(|| error("workgroup reduction has no dispatch"))?;
            Some(self.compiler.execution.dispatch(parent)?.width.get())
        } else {
            None
        };
        let scan_size = self
            .query()
            .lookup("WorkgroupScan", operation)?
            .map(|size| self.query().0.value_to_base::<i64>(size) as u32);
        let domain = self.query().required("OperationDomain", operation)?;
        let ordinal_name = self.query().required("DomainOrdinal", domain)?;
        let axes = domain_axes(self.query(), domain)?;
        let region = self.query().required("ControlRegion", (operation, 0_i64))?;
        let mut collects = Vec::new();
        let mut compacts = Vec::new();
        let mut contributes = false;
        let mut folds = Vec::new();
        let mut fold_data = Vec::new();
        for clause in 0..crate::egglog::analysis::count(self.query(), "ResultCount", operation)? {
            let kind = self.query().required("ResultKindOf", (operation, clause))?;
            let count =
                crate::egglog::analysis::count(self.query(), "ResultValueCount", (operation, clause))?;
            let values = self.query().indexed(
                "ResultValue",
                &[operation, self.query().0.base_to_value(clause)],
                count,
            )?;
            let output_names = |role: &str| -> Result<Vec<Value>, OptimizeError> {
                let query = self.query();
                let mut names = Vec::new();
                for slot in 0..crate::egglog::analysis::count(query, "OutputCount", operation)? {
                    let selected = query.required("OutputClause", (operation, slot))?;
                    if query.0.value_to_base::<i64>(selected) == clause
                        && query.enode(role, query.required("OutputRole", (operation, slot))?)?.is_some()
                    {
                        names.push(query.name(query.required("Output", (operation, slot))?)?);
                    }
                }
                Ok(names)
            };
            if self.query().enode("Collected", kind)?.is_some() {
                collects.extend(output_names("CollectedOutput")?.into_iter().zip(values));
            } else if self.query().enode("Compacted", kind)?.is_some() {
                let predicate = self.query().required("ResultPredicate", (operation, clause))?;
                compacts.push((
                    predicate,
                    output_names("CompactedOutput")?.into_iter().zip(values).collect::<Vec<_>>(),
                ));
            } else if let Some(fold) = self.query().enode("Folded", kind)? {
                let (kind, _) = self.query().application(
                    fold[0],
                    &[
                        "Total",
                        "InclusiveScan",
                        "ExclusiveScan",
                        "InclusiveScanAndTotal",
                        "ExclusiveScanAndTotal",
                    ],
                )?;
                let mode = if kind == "Total" {
                    FoldMode::Total
                } else {
                    FoldMode::Scan {
                        exclusive: kind.starts_with("Exclusive"),
                    }
                };
                let reducer = self.query().required("ResultReducer", (operation, clause))?;
                let count = crate::egglog::analysis::count(
                    self.query(),
                    "ResultNeutralCount",
                    (operation, clause),
                )?;
                let neutral = self.query().indexed(
                    "ResultNeutral",
                    &[operation, self.query().0.base_to_value(clause)],
                    count,
                )?;
                let mut prefixes = output_names("PrefixOutput")?;
                let packed = output_names("CompactedOutput")?;
                let compact = if packed.is_empty() {
                    None
                } else {
                    prefixes.extend(packed);
                    let index = compacts.len();
                    compacts.push((
                        self.query().required("ResultPredicate", (operation, clause))?,
                        vec![],
                    ));
                    Some(index)
                };
                fold_data.push((
                    compact,
                    mode,
                    reducer,
                    neutral,
                    values,
                    prefixes,
                    output_names("TotalOutput")?,
                ));
            } else if self.query().enode("Contributed", kind)?.is_some() {
                contributes = true;
            } else {
                return Err(error("unknown collective result role"));
            }
        }
        for (compact, mode, reducer, atoms, values, prefixes, totals) in fold_data {
            let parameters = self.query().sequence("RegionParameter", reducer)?;
            let mut neutral = Vec::new();
            for (i, atom) in atoms.into_iter().enumerate() {
                let Some(parameter) = parameters.get(i) else {
                    return Err(error("reducer has no state parameter"));
                };
                let Some((_, mut ty)) = runtime_binding(self.query(), *parameter)? else {
                    return Err(error("reducer state has no runtime value"));
                };
                ty = super::reducer_state_type(self.query(), reducer, i, atom, ty)?;
                let Some(value) = self.operand(atom)? else {
                    return Err(error("collective neutral is a capability"));
                };
                if ty.is_array()
                    && !matches!(ty.array_size(), Some(Type::Constructed(TypeName::Size(_), _)))
                {
                    let count = match value.ty.array_size() {
                        Some(Type::Constructed(TypeName::Size(size), _)) => Some(*size),
                        _ => None,
                    };
                    if !self.query().contains("ReducerPreservesOuterLength", (reducer, i as i64))? {
                        return Err(error(
                            "array-valued collective has no proven fixed reducer state shape",
                        ));
                    }
                    let Some(count) = count else {
                        return Err(error("array-valued collective has no proven fixed neutral shape"));
                    };
                    let Some(element) = ty.elem_type() else {
                        return Err(error("array reducer element type missing"));
                    };
                    ty = types::sized_array(count, element.clone());
                }
                neutral.push(self.canonical_value(value, &ty)?);
            }
            folds.push(Fold {
                compact,
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
        if let Some(CollectivePhase::Carries {
            partials,
            carries,
            groups,
            width,
        }) = &phase
        {
            let Some(_) = dispatch else {
                return Err(error("collective carry dispatch missing"));
            };
            self.collective_carries(&folds, partials, carries, groups.clone(), *width)?;
            return Ok(vec![]);
        }
        if let Some(CollectivePhase::Apply {
            range,
            group,
            carries,
        }) = &phase
        {
            let Some(dispatch) = dispatch else {
                return Err(error("collective application dispatch missing"));
            };
            self.apply_carries(
                &folds,
                carries,
                group.clone(),
                range.clone(),
                dispatch.width.get(),
            )?;
            return Ok(vec![]);
        }
        let mut collected = collects.iter().map(|(name, _)| *name).collect::<LookupSet<_>>();
        collected.extend(compacts.iter().flat_map(|(_, outputs)| outputs.iter().map(|(name, _)| *name)));
        for fold in &folds {
            collected.extend(fold.prefixes.iter().copied());
            if axes.len() > 1 {
                collected.extend(fold.totals.iter().copied());
            }
        }
        for &binding in pattern {
            if let Some((name, ty)) = runtime_binding(self.query(), binding)? {
                if !collected.contains(&name) {
                    continue;
                }
                let rank = if folds.iter().any(|fold| fold.totals.contains(&name)) {
                    axes.len() - 1
                } else {
                    axes.len()
                };
                let dimensions = axes[..rank]
                    .iter()
                    .map(|&axis| {
                        self.query()
                            .lookup("StaticIndex", axis.extent)?
                            .map(|size| {
                                usize::try_from(self.query().0.value_to_base::<i64>(size).max(0))
                                    .map_err(|_| error("collection dimension exceeds usize"))
                            })
                            .transpose()
                    })
                    .collect::<Result<Vec<_>, OptimizeError>>()?;
                let ty = collection_shape(&self.collection_output_type(name, ty), &dimensions)?;
                let Some(axis) = axes.first() else {
                    return Err(error("collection requires an axis"));
                };
                if let Some(size) = scan_size {
                    let element =
                        ty.elem_type().ok_or_else(|| error("workgroup scan has no scalar element"))?;
                    let storage = self.workgroup_array(size, element)?;
                    self.collections.insert(name, Collection::Allocated(storage));
                    continue;
                }
                if let Some(storage) = self.reuse_storage(name, &ty)? {
                    self.collections.insert(name, Collection::Allocated(storage));
                    continue;
                }
                let shape = if dispatch.is_some() {
                    let ranked = rank > 1
                        && (dimensions[1..].contains(&None)
                            || self.query().contains("RankedLoopCollection", name)?);
                    let rank = if ranked { rank } else { 1 };
                    let mut shape = Vec::new();
                    for &axis in &axes[..rank] {
                        let count = self.index_expression(axis.extent)?;
                        let length = host::guarded_index(
                            self.query(),
                            &self.host_values,
                            match &phase {
                                Some(CollectivePhase::Epilogue { capacity, .. }) => *capacity,
                                _ => axis.extent,
                            },
                            self.guard.as_ref(),
                        )?;
                        shape.push((count, Expr::Max(Box::new(length), Box::new(Expr::Integer(0)))));
                    }
                    if let Some(row) = self.query().lookup("LoopInitialRow", name)? {
                        let row = self.named(row)?;
                        let length = self.host_array_length(&row)?;
                        let count = self.array_length(row)?;
                        shape.push((count, length));
                    }
                    CollectionShape::Device(shape)
                } else {
                    CollectionShape::Local(self.index_expression(axis.extent)?)
                };
                let block = self.current()?;
                self.collections.insert(name, Collection::Pending { ty, shape, block });
            }
        }
        let atomic_updates = contributes
            && compacts.is_empty()
            && folds.is_empty()
            && self.atomic_contributions(operation)?;
        if contributes && !atomic_updates && matches!(phase, Some(CollectivePhase::Epilogue { .. })) {
            return Err(error("scan overwrite epilogue requires atomic storage"));
        }
        let algorithm = self.query().required("SelectedKernelAlgorithm", (operation, atomic_updates))?;
        let compact = self.query().enode("CompactionKernel", algorithm)?.is_some();
        if compact || self.query().enode("OrderedKernel", algorithm)?.is_some() {
            let mut initial = vec![Self::number(0); compacts.len()];
            initial.extend(folds.iter().flat_map(|fold| fold.neutral.clone()));
            let statements = region;
            let run = |body: &mut Self| {
                let outer = if folds.is_empty() { &axes[..0] } else { &axes[..axes.len() - 1] };
                let inner = if folds.is_empty() { &axes[..] } else { &axes[axes.len() - 1..] };
                let state = body.domain_axis(
                    outer,
                    0,
                    ordinal_name,
                    vec![],
                    Self::number(0),
                    initial.clone(),
                    None,
                    &mut |body, coordinates, state| {
                        let mut initial = state[..compacts.len()].to_vec();
                        initial.extend(folds.iter().flat_map(|fold| fold.neutral.clone()));
                        let ordinal = body.named(ordinal_name)?;
                        let state = body.domain_axis(
                            inner,
                            0,
                            ordinal_name,
                            coordinates.to_vec(),
                            ordinal,
                            initial,
                            None,
                            &mut |body, _, state| {
                                let folded = body.parallel_lane(
                                    statements,
                                    domain,
                                    &collects,
                                    &folds,
                                    state[compacts.len()..].to_vec(),
                                    None,
                                )?;
                                let mut next = Vec::new();
                                for (compact, ((predicate, outputs), count)) in
                                    compacts.iter().zip(&state).enumerate()
                                {
                                    let keep = body.atom(*predicate)?;
                                    let count = body.branch(
                                        keep,
                                        |body| {
                                            for &(name, result) in outputs {
                                                let Some(value) = body.operand(result)? else {
                                                    return Err(error(
                                                        "compaction value has no SSA representation",
                                                    ));
                                                };
                                                body.write_array(name, &[count.clone()], value)?;
                                            }
                                            let mut offset = 0;
                                            for fold in &folds {
                                                if fold.compact == Some(compact) {
                                                    let prefix = if matches!(
                                                        fold.mode,
                                                        FoldMode::Scan { exclusive: true }
                                                    ) {
                                                        &state[compacts.len()..]
                                                    } else {
                                                        &folded
                                                    };
                                                    for (i, &name) in fold.prefixes.iter().enumerate() {
                                                        body.write_array(
                                                            name,
                                                            &[count.clone()],
                                                            prefix[offset + i].clone(),
                                                        )?;
                                                    }
                                                }
                                                offset += fold.neutral.len();
                                            }
                                            body.binary(BinaryOperator::Add, count.clone(), Self::number(1))
                                        },
                                        |_body| Ok(count.clone()),
                                    )?;
                                    next.push(count);
                                }
                                next.extend(folded);
                                Ok(next)
                            },
                        )?;
                        let mut offset = compacts.len();
                        for fold in &folds {
                            for (i, &name) in fold.totals.iter().enumerate() {
                                if collected.contains(&name) {
                                    body.write_array(name, coordinates, state[offset + i].clone())?;
                                }
                            }
                            offset += fold.neutral.len();
                        }
                        Ok(state)
                    },
                )?;
                body.pack_results(state)
            };
            let counts = if let Some(dispatch) = dispatch {
                let (group, _) = self.workgroups(dispatch.grid)?;
                let first_group = self.binary(BinaryOperator::Equal, group, Self::number(0))?;
                if compact {
                    self.branch(
                        first_group,
                        |body| {
                            body.compact_group(
                                domain,
                                statements,
                                &collects,
                                &compacts,
                                dispatch.width.get(),
                            )
                        },
                        |body| body.pack_results(initial.clone()),
                    )?
                } else {
                    let lane = self.local_id()?;
                    let first_lane = self.binary(BinaryOperator::Equal, lane, Self::number(0))?;
                    let leader = self.binary(BinaryOperator::LogicalAnd, first_group, first_lane)?;
                    self.branch(leader, run, |body| body.pack_results(initial.clone()))?
                }
            } else {
                run(self)?
            };
            let counts = self.unpack_results(counts, initial.len())?;
            for (i, (_, outputs)) in compacts.iter().enumerate() {
                let count = counts[i].clone();
                for &(name, _) in outputs {
                    let array = self.named(name)?;
                    let view = self.slice_array(array, Self::number(0), count.clone(), Self::number(1))?;
                    self.values.insert(name, view);
                }
            }
            let mut offset = compacts.len();
            for fold in &folds {
                if let Some(compact) = fold.compact {
                    for &name in &fold.prefixes {
                        let array = self.named(name)?;
                        let view = self.slice_array(
                            array,
                            Self::number(0),
                            counts[compact].clone(),
                            Self::number(1),
                        )?;
                        self.values.insert(name, view);
                    }
                }
                for (i, &name) in fold.totals.iter().enumerate() {
                    if !collected.contains(&name) {
                        let value = counts[offset + i].clone();
                        self.values.insert(name, value);
                    }
                }
                offset += fold.neutral.len();
            }
            let mut values = Vec::new();
            for &binding in pattern {
                if let Some((name, _)) = runtime_binding(self.query(), binding)? {
                    values.push(self.named(name)?);
                }
            }
            return Ok(values);
        }
        if let Some(dispatch) = dispatch {
            if self.query().contains("UseNativeSubgroups", operation)? {
                self.prepare_subgroups(dispatch.width.get())?;
            }
        }
        let outer = if folds.is_empty() { &axes[..] } else { &axes[..axes.len() - 1] };
        let partition = if phase.is_some() {
            None
        } else if let Some(dispatch) = dispatch {
            if folds.is_empty() && !workgroup_map {
                Some(self.invocation_partition(dispatch)?)
            } else {
                Some(self.workgroups(dispatch.grid)?)
            }
        } else {
            None
        };
        let mut initial = Vec::new();
        for fold in &folds {
            for (&name, value) in fold.totals.iter().zip(&fold.neutral) {
                if !collected.contains(&name) {
                    initial.push(value.clone());
                }
            }
        }
        // The same lane owns each prefix and its pointwise consumer, so no
        // cross-workgroup visibility or additional barrier is required.
        let scalar_results = if let Some(CollectivePhase::Epilogue {
            range, cooperative, ..
        }) = &phase
        {
            let Some(dispatch) = dispatch else {
                return Err(error("scan epilogue dispatch missing"));
            };
            let axis = axes[0];
            let lane = if *cooperative { self.local_id()? } else { Self::number(0) };
            let step = if *cooperative { dispatch.width.get() } else { 1 };
            let start = self.binary(BinaryOperator::Add, range.0.clone(), lane)?;
            self.counted(
                start,
                range.1.clone(),
                Self::number(step),
                vec![],
                |body_lower, index, _| {
                    body_lower.values.insert(axis.name, index.clone());
                    body_lower.values.insert(ordinal_name, index);
                    body_lower.parallel_lane(region, domain, &collects, &folds, vec![], None)
                },
            )?
        } else {
            self.domain_axis(
                outer,
                0,
                ordinal_name,
                Vec::new(),
                Self::number(0),
                initial,
                partition,
                &mut |body_lower, coordinates, _| {
                    if folds.is_empty() {
                        let leader = if workgroup_map {
                            let lane = body_lower.local_id()?;
                            Some(body_lower.binary(BinaryOperator::Equal, lane, Self::number(0))?)
                        } else {
                            None
                        };
                        body_lower.parallel_lane(region, domain, &collects, &folds, vec![], leader)?;
                        return Ok(vec![]);
                    }
                    let total_coordinates = coordinates.to_vec();
                    let ordinal = body_lower.named(ordinal_name)?;
                    let state = if let Some(width) = cooperative_width {
                        body_lower.cooperative_segment(
                            domain,
                            axes[axes.len() - 1],
                            ordinal_name,
                            ordinal,
                            region,
                            &collects,
                            &folds,
                            width,
                            if let Some(CollectivePhase::Chunks { range, .. }) = &phase {
                                Some(range.clone())
                            } else {
                                None
                            },
                        )?
                    } else {
                        let initial =
                            folds.iter().flat_map(|fold| fold.neutral.clone()).collect::<Vec<_>>();
                        body_lower.domain_axis(
                            &axes[axes.len() - 1..],
                            0,
                            ordinal_name,
                            coordinates.to_vec(),
                            ordinal,
                            initial,
                            None,
                            &mut |body_lower, _, state| {
                                body_lower.parallel_lane(region, domain, &collects, &folds, state, None)
                            },
                        )?
                    };
                    if let Some(CollectivePhase::Chunks { group, partials, .. }) = &phase {
                        let lane = body_lower.local_id()?;
                        let leader = body_lower.binary(BinaryOperator::Equal, lane, Self::number(0))?;
                        body_lower.when(leader, |body| {
                            let (_, groups) = body.workgroups(None)?;
                            for ((binding, element), value) in partials.iter().zip(&state) {
                                let storage = super::entry::storage_view(
                                    body,
                                    *binding,
                                    element.clone(),
                                    groups.clone(),
                                )?;
                                body.store(storage, group.clone(), value.clone())?;
                            }
                            Ok(())
                        })?;
                    }
                    let mut state = state.into_iter();
                    let mut scalars = Vec::new();
                    for fold in &folds {
                        let total = state.by_ref().take(fold.neutral.len()).collect::<Vec<_>>();
                        for (&name, value) in fold.totals.iter().zip(total) {
                            if collected.contains(&name) {
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
            )?
        };
        if scan_size.is_some() {
            self.workgroup_barrier()?;
        }
        let mut scalars = scalar_results.into_iter();
        let mut values = Vec::new();
        for &binding in pattern {
            if let Some((name, _)) = runtime_binding(self.query(), binding)? {
                if collected.contains(&name) {
                    let mut array = self.named(name)?;
                    if let Some(CollectivePhase::Epilogue {
                        range,
                        cooperative: false,
                        ..
                    }) = &phase
                    {
                        array =
                            self.slice_array(array, Self::number(0), range.1.clone(), Self::number(1))?;
                    }
                    values.push(self.snapshot(array)?);
                } else {
                    let Some(value) = scalars.next() else {
                        return Err(error("parallel scalar result arity mismatch"));
                    };
                    values.push(value);
                }
            }
        }
        if scalars.next().is_some() {
            return Err(error("parallel scalar result arity mismatch"));
        }
        Ok(values)
    }

    fn scan_columns(
        &mut self,
        operation: Value,
        pattern: &[Value],
        plan: Value,
    ) -> Result<Vec<Typed>, OptimizeError> {
        let query = self.query();
        let fields = query.fields("ScanColumns", plan)?;
        let source = self.named(fields[0])?;
        let map = fields[2];
        let columns = query.required("OperationDomain", map)?;
        let axis = query.required("DomainAxis", (columns, 0_i64))?;
        let ordinal = query.required("DomainOrdinal", columns)?;
        let lane = query.required("ControlRegion", (map, 0_i64))?;
        let result = query.required("ResultValue", (map, 0_i64, 0_i64))?;
        let domain = query.required("OperationDomain", operation)?;
        let rows = query.required("DomainExtent", (domain, 0_i64))?;
        let neutral =
            self.named(query.name(query.required("ResultNeutral", (operation, 0_i64, 0_i64))?)?)?;
        let [binding] = pattern else {
            return Err(error("column scan requires one result"));
        };
        let Some((name, ty)) = runtime_binding(query, *binding)? else {
            return Err(error("column scan result cannot be a capability"));
        };
        let steps = query.operations(lane)?;
        let row_count = self.array_length(source.clone())?;
        let width = self.array_length(neutral.clone())?;
        let shape = [(rows, row_count.clone()), (fields[1], width.clone())]
            .into_iter()
            .map(|(extent, size)| {
                host::guarded_index(self.query(), &self.host_values, extent, self.guard.as_ref())
                    .map(|count| (size, count))
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.values.insert(fields[3], width.clone());
        let output =
            self.collection_storage(&ty, &CollectionShape::Device(shape), None, self.current()?)?;
        self.collections.insert(name, Collection::Allocated(output.clone()));
        let dispatch = *self.compiler.execution.dispatch(operation)?;
        let (start, step) = self.invocation_partition(dispatch)?;
        self.counted(start, width, step, vec![], |body, column, _| {
            body.values.insert(axis, column.clone());
            body.values.insert(ordinal, column.clone());
            let initial = body.index(neutral, column.clone())?;
            body.counted(
                Self::number(0),
                row_count,
                Self::number(1),
                vec![initial],
                |body, row, state| {
                    let item = body.index(source.clone(), row.clone())?;
                    let inputs = [state[0].clone(), body.index(item, column.clone())?];
                    for &step in &steps {
                        let query = body.query();
                        if let Some(slot) = query.lookup("ColumnScanLoad", (operation, step))? {
                            let slot = query.0.value_to_base::<i64>(slot) as usize;
                            let name = query.name(query.required("Output", (step, 0_i64))?)?;
                            body.values.insert(name, inputs[slot].clone());
                        } else {
                            body.statement(step)?;
                        }
                    }
                    let Some(next) = body.operand(result)? else {
                        return Err(error("column reducer returns a capability"));
                    };
                    body.write_array(name, &[row, column.clone()], next.clone())?;
                    Ok(vec![next])
                },
            )?;
            Ok(vec![])
        })?;
        Ok(vec![output])
    }

    pub(super) fn collection_storage(
        &mut self,
        ty: &Type,
        shape: &CollectionShape,
        capacity: Option<i64>,
        block: BlockId,
    ) -> Result<Typed, OptimizeError> {
        if let Type::Constructed(name @ (TypeName::Tuple(_) | TypeName::Record(_)), fields) = ty {
            let values = fields
                .iter()
                .map(|field| self.collection_storage(field, shape, capacity, block))
                .collect::<Result<Vec<_>, _>>()?;
            let ty = Type::Constructed(
                name.clone(),
                values.iter().map(|value| value.ty.clone()).collect(),
            );
            let value = self.builder.func_mut().append_inst(
                block,
                InstKind::Op {
                    tag: OpTag::Tuple,
                    operands: values.into_iter().map(|value| value.value).collect(),
                },
                ty.clone(),
            );
            return Ok(Typed {
                value: value.into(),
                ty,
            });
        }
        let shape = match shape {
            CollectionShape::Local(count) => {
                let Some(element) = ty.elem_type().cloned() else {
                    return Err(error("collection binding must be an array or product of arrays"));
                };
                let capacity = self.integer_literal(count).or(capacity).or_else(|| match ty.array_size() {
                    Some(Type::Constructed(TypeName::Size(size), _)) => i64::try_from(*size).ok(),
                    _ => None,
                });
                let Some(capacity) = capacity else {
                    return Err(error("local collection requires a capacity bound"));
                };
                let capacity = usize::try_from(capacity.max(0))
                    .map_err(|_| error("collection capacity exceeds usize"))?;
                return self.local_array_with_length(types::sized_array(capacity, element), count.clone());
            }
            CollectionShape::Device(shape) => shape,
        };
        let Some((first, rest)) = shape.split_first() else {
            return Err(error("device collection requires a dimension"));
        };
        let mut element = ty;
        for _ in shape {
            let Some(child) = element.elem_type() else {
                return Err(error("collection shape exceeds array rank"));
            };
            element = child;
        }
        let element = storage_value_type(element);
        let Some(stride) = storage_elem_stride(&element) else {
            return Err(error("collection element requires representation lowering"));
        };
        let mut count = first.0.clone();
        let mut length = first.1.clone();
        if rest.is_empty() {
            length = capacity.map(Expr::Integer).unwrap_or(length);
        }
        for (dimension, extent) in rest {
            let value = self.builder.func_mut().append_inst(
                block,
                InstKind::Op {
                    tag: OpTag::BinOp(BinaryOperator::Multiply),
                    operands: vec![count.value, dimension.value],
                },
                types::i32(),
            );
            count = Typed {
                value: value.into(),
                ty: types::i32(),
            };
            length = Expr::Multiply(Box::new(length), Box::new(extent.clone()));
        }
        let bytes = Expr::Multiply(Box::new(length), Box::new(Expr::Integer(stride.into())));
        let (binding, mut value) = self.device_buffer_at(&element, bytes, count, block)?;
        if !rest.is_empty() {
            let mut fields = vec![element, types::array_variant_view()];
            fields.extend(shape.iter().map(|_| Type::Constructed(TypeName::SizePlaceholder, vec![])));
            fields.push(types::buffer_tag(binding));
            let ty = Type::Constructed(TypeName::Array, fields);
            let mut operands = vec![Self::number(0).value];
            operands.extend(shape.iter().map(|(dimension, _)| dimension.value));
            let descriptor = self.builder.func_mut().append_inst(
                block,
                InstKind::Op {
                    tag: OpTag::StorageView(PureViewSource::Storage(binding)),
                    operands,
                },
                ty.clone(),
            );
            value = Typed {
                value: descriptor.into(),
                ty,
            };
            self.register_array(
                &value,
                Self::number(0),
                shape.iter().map(|(dimension, _)| dimension.clone()).collect(),
            )?;
        }
        self.set_shape(&value, shape.iter().map(|(_, extent)| extent.clone()).collect())?;
        Ok(value)
    }

    fn collective_carries(
        &mut self,
        folds: &[Fold],
        partials: &[(BindingRef, Type)],
        carries: &[(BindingRef, Type)],
        groups: Typed,
        width: u32,
    ) -> Result<(), OptimizeError> {
        let lane = self.local_id()?;
        let neutral = folds.iter().flat_map(|fold| fold.neutral.clone()).collect::<Vec<_>>();
        let totals = self.counted(
            Self::number(0),
            groups.clone(),
            Self::number(width),
            neutral.clone(),
            |body, tile, state| {
                let index = body.binary(BinaryOperator::Add, tile, lane.clone())?;
                let active = body.binary(BinaryOperator::Less, index.clone(), groups.clone())?;
                let values = body.branch(
                    active.clone(),
                    |body| {
                        let mut values = Vec::new();
                        for ((binding, element), initial) in partials.iter().zip(&neutral) {
                            let buffer = super::entry::storage_view(
                                body,
                                *binding,
                                element.clone(),
                                groups.clone(),
                            )?;
                            let value = body.index(buffer, index.clone())?;
                            values.push(body.canonical_value(value, &initial.ty)?);
                        }
                        body.pack_results(values)
                    },
                    |body| body.pack_results(neutral.clone()),
                )?;
                let values = body.unpack_results(values, neutral.len())?;
                let (exclusive, totals) = body.workgroup_fold(folds, &values, lane.clone(), width)?;
                let exclusive = body.combine_folds(folds, &state, &exclusive)?;
                body.when(active, |body| {
                    for ((binding, element), value) in carries.iter().zip(&exclusive) {
                        let buffer =
                            super::entry::storage_view(body, *binding, element.clone(), groups.clone())?;
                        body.store(buffer, index.clone(), value.clone())?;
                    }
                    Ok(())
                })?;
                body.combine_folds(folds, &state, &totals)
            },
        )?;
        let mut totals = totals.into_iter();
        for fold in folds {
            let values = totals.by_ref().take(fold.neutral.len()).collect::<Vec<_>>();
            for (&name, value) in fold.totals.iter().zip(values) {
                self.values.insert(name, value);
            }
        }
        Ok(())
    }

    fn apply_carries(
        &mut self,
        folds: &[Fold],
        carries: &[(BindingRef, Type)],
        group: Typed,
        range: (Typed, Typed),
        width: u32,
    ) -> Result<(), OptimizeError> {
        let (_, groups) = self.workgroups(None)?;
        let mut state = Vec::new();
        for (binding, element) in carries {
            let buffer = super::entry::storage_view(self, *binding, element.clone(), groups.clone())?;
            state.push(self.index(buffer, group.clone())?);
        }
        let lane = self.local_id()?;
        let start = self.binary(BinaryOperator::Add, range.0, lane)?;
        self.counted(start, range.1, Self::number(width), vec![], |body, index, _| {
            let mut offset = 0;
            for fold in folds {
                let end = offset + fold.neutral.len();
                if !fold.prefixes.is_empty() {
                    let mut args = state[offset..end].to_vec();
                    for &name in &fold.prefixes {
                        let array = body.named(name)?;
                        args.push(body.index(array, index.clone())?);
                    }
                    let values = body.lambda(fold.reducer, args)?;
                    for (&name, value) in fold.prefixes.iter().zip(values) {
                        body.write_array(name, &[index.clone()], value)?;
                    }
                }
                offset = end;
            }
            Ok(vec![])
        })?;
        Ok(())
    }

    fn domain_coordinates(&mut self, domain: Value) -> Result<Vec<Typed>, OptimizeError> {
        domain_axes(self.query(), domain)?.into_iter().map(|axis| self.named(axis.name)).collect()
    }

    fn parallel_lane(
        &mut self,
        statements: Value,
        domain: Value,
        collects: &[(Value, Value)],
        folds: &[Fold],
        state: Vec<Typed>,
        leader: Option<Typed>,
    ) -> Result<Vec<Typed>, OptimizeError> {
        let coordinates = self.domain_coordinates(domain)?;
        let operations = self.query().operations(statements)?;
        let suffix = if let (Some(_), Some(active)) = (&leader, self.active_dispatch) {
            self.query().lookup("WorkgroupMapSuffix", active)?
        } else {
            None
        };
        let split = if let Some(suffix) = suffix {
            operations
                .iter()
                .position(|&operation| operation == suffix)
                .ok_or_else(|| error("workgroup scan consumer is outside the map"))?
        } else {
            operations.len()
        };
        let (prefix, tail) = operations.split_at(split);
        self.statements(prefix)?;
        let publish = |body: &mut Self| {
            body.statements(tail)?;
            for &(output, result) in collects {
                let Some(value) = body.operand(result)? else {
                    return Err(error("collected result is a capability"));
                };
                body.write_array(output, &coordinates, value)?;
            }
            Ok(())
        };
        if let Some(leader) = leader {
            self.when(leader, publish)?;
        } else {
            publish(self)?;
        }
        if suffix.is_some() {
            self.workgroup_barrier()?;
        }
        let mut state = state.into_iter();
        let mut next = Vec::new();
        for fold in folds {
            let previous = state.by_ref().take(fold.neutral.len()).collect::<Vec<_>>();
            let contribution = self.yield_values(&fold.values)?;
            let mut args = previous.clone();
            args.extend(contribution);
            let combined = self.lambda(fold.reducer, args)?;
            let prefix = if matches!(fold.mode, FoldMode::Scan { exclusive: true, .. }) {
                &previous
            } else {
                &combined
            };
            if fold.compact.is_none() {
                for (&output, value) in fold.prefixes.iter().zip(prefix) {
                    self.write_array(output, &coordinates, value.clone())?;
                }
            }
            next.extend(combined);
        }
        Ok(next)
    }

    fn domain_axis(
        &mut self,
        axes: &[Axis],
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
                let shape = axes
                    .iter()
                    .map(|axis| self.index_expression(axis.extent))
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
                        for (axis, index) in axes.iter().zip(&indices) {
                            body.values.insert(axis.name, index.clone());
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

    fn axis_bounds(&mut self, axis: Axis, ordinal: Typed) -> Result<(Value, Typed, Typed), OptimizeError> {
        let count = self.index_expression(axis.extent)?;
        let base = self.binary(BinaryOperator::Multiply, ordinal, count.clone())?;
        Ok((axis.name, base, count))
    }

    fn invocation_partition(&mut self, dispatch: Dispatch) -> Result<(Typed, Typed), OptimizeError> {
        let (group, groups) = self.workgroups(dispatch.grid)?;
        let lanes = Self::number(dispatch.width.get());
        let base = self.binary(BinaryOperator::Multiply, group, lanes.clone())?;
        let lane = self.local_id()?;
        let start = self.binary(BinaryOperator::Add, base, lane)?;
        let step = self.binary(BinaryOperator::Multiply, groups, lanes)?;
        Ok((start, step))
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
            for (value, neutral) in values.into_iter().zip(&fold.neutral) {
                result.push(self.canonical_value(value, &neutral.ty)?);
            }
            offset = end;
        }
        Ok(result)
    }

    fn cooperative_segment(
        &mut self,
        domain: Value,
        axis: Axis,
        flat: Value,
        ordinal: Typed,
        statements: Value,
        collects: &[(Value, Value)],
        folds: &[Fold],
        width: u32,
        range: Option<(Typed, Typed)>,
    ) -> Result<Vec<Typed>, OptimizeError> {
        let (name, base, count) = self.axis_bounds(axis, ordinal)?;
        let (start, count) = range.unwrap_or_else(|| (Self::number(0), count));
        let lane = self.local_id()?;
        let initial = folds.iter().flat_map(|fold| fold.neutral.clone()).collect::<Vec<_>>();
        self.counted(
            start,
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
                        body.statements(&body.query().operations(statements)?)?;
                        for &(output, result) in collects {
                            let Some(value) = body.operand(result)? else {
                                return Err(error("collected result is a capability"));
                            };
                            body.write_array(output, &coordinates, value)?;
                        }
                        let mut values = Vec::new();
                        for fold in folds {
                            let contributions = body.yield_values(&fold.values)?;
                            for (value, neutral) in contributions.into_iter().zip(&fold.neutral) {
                                values.push(body.canonical_value(value, &neutral.ty)?);
                            }
                        }
                        body.pack_results(values)
                    },
                    |body| body.pack_results(initial.clone()),
                )?;
                let values = body.unpack_results(values, initial.len())?;
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
        folds: &[Fold],
        values: &[Typed],
        lane: Typed,
        width: u32,
    ) -> Result<(Vec<Typed>, Vec<Typed>), OptimizeError> {
        if self.subgroup.is_some() {
            let (_, exclusive, totals) = self.subgroup_scan(values.to_vec(), width)?;
            return Ok((exclusive, totals));
        }
        let neutral = folds.iter().flat_map(|fold| fold.neutral.clone()).collect::<Vec<_>>();
        self.workgroup_scan(values, &neutral, lane, width, |body, left, right| {
            body.combine_folds(folds, left, right)
        })
    }

    fn workgroup_scan(
        &mut self,
        values: &[Typed],
        neutral: &[Typed],
        lane: Typed,
        width: u32,
        combine: impl Fn(&mut Self, &[Typed], &[Typed]) -> Result<Vec<Typed>, OptimizeError>,
    ) -> Result<(Vec<Typed>, Vec<Typed>), OptimizeError> {
        let buffers = values
            .iter()
            .map(|value| self.workgroup_array(width, &value.ty))
            .collect::<Result<Vec<_>, _>>()?;
        let mut initial = vec![Self::number(1)];
        initial.extend_from_slice(values);
        let inclusive = self.while_loop(
            initial,
            |body, state| body.binary(BinaryOperator::Less, state[0].clone(), Self::number(width)),
            |body, state| {
                for (buffer, value) in buffers.iter().zip(&state[1..]) {
                    body.store(buffer.clone(), lane.clone(), value.clone())?;
                }
                body.workgroup_barrier()?;
                let active = body.binary(BinaryOperator::GreaterEqual, lane.clone(), state[0].clone())?;
                let next = body.branch(
                    active,
                    |body| {
                        let previous =
                            body.binary(BinaryOperator::Subtract, lane.clone(), state[0].clone())?;
                        let left = buffers
                            .iter()
                            .map(|buffer| body.index(buffer.clone(), previous.clone()))
                            .collect::<Result<Vec<_>, _>>()?;
                        let combined = combine(body, &left, &state[1..])?;
                        body.pack_results(combined)
                    },
                    |body| body.pack_results(state[1..].to_vec()),
                )?;
                // No lane overwrites scratch until all lanes have read this step.
                body.workgroup_barrier()?;
                let mut result =
                    vec![body.binary(BinaryOperator::Multiply, state[0].clone(), Self::number(2))?];
                result.extend(body.unpack_results(next, values.len())?);
                Ok(result)
            },
        )?;
        for (buffer, value) in buffers.iter().zip(&inclusive[1..]) {
            self.store(buffer.clone(), lane.clone(), value.clone())?;
        }
        self.workgroup_barrier()?;
        let active = self.binary(BinaryOperator::Greater, lane.clone(), Self::number(0))?;
        let exclusive = self.branch(
            active,
            |body| {
                let previous = body.binary(BinaryOperator::Subtract, lane, Self::number(1))?;
                let values = buffers
                    .iter()
                    .map(|buffer| body.index(buffer.clone(), previous.clone()))
                    .collect::<Result<Vec<_>, _>>()?;
                body.pack_results(values)
            },
            |body| body.pack_results(neutral.to_vec()),
        )?;
        let exclusive = self.unpack_results(exclusive, values.len())?;
        let totals = buffers
            .iter()
            .map(|buffer| self.index(buffer.clone(), Self::number(width - 1)))
            .collect::<Result<Vec<_>, _>>()?;
        self.workgroup_barrier()?;
        Ok((exclusive, totals))
    }

    fn compact_group(
        &mut self,
        domain: Value,
        statements: Value,
        collects: &[(Value, Value)],
        compacts: &[(Value, Vec<(Value, Value)>)],
        width: u32,
    ) -> Result<Typed, OptimizeError> {
        let axes = domain_axes(self.query(), domain)?;
        let ordinal_name = self.query().required("DomainOrdinal", domain)?;
        let axis = axes[0];
        let count = self.index_expression(axis.extent)?;
        let lane = self.local_id()?;
        let neutral = vec![Self::number(0); compacts.len()];
        let totals = self.counted(
            Self::number(0),
            count.clone(),
            Self::number(width),
            neutral.clone(),
            |body, tile, carried| {
                let index = body.binary(BinaryOperator::Add, tile, lane.clone())?;
                body.values.insert(ordinal_name, index.clone());
                body.values.insert(axis.name, index.clone());
                let active = body.binary(BinaryOperator::Less, index, count)?;
                let flags = compacts
                    .iter()
                    .map(|_| {
                        let flag = body.local_place(Self::number(0).ty)?;
                        body.store_place(flag, Self::number(0))?;
                        Ok(flag)
                    })
                    .collect::<Result<Vec<_>, OptimizeError>>()?;
                let mut payloads = Vec::new();
                body.when(active, |body| {
                    body.parallel_lane(statements, domain, collects, &[], vec![], None)?;
                    for ((predicate, outputs), flag) in compacts.iter().zip(&flags) {
                        let keep = body.atom(*predicate)?;
                        let keep = body.cast(keep, &Self::number(0).ty)?;
                        body.store_place(*flag, keep)?;
                        let mut fields = Vec::new();
                        for &(name, result) in outputs {
                            let Some(value) = body.operand(result)? else {
                                return Err(error("compaction payload has no value"));
                            };
                            let temporary = body.local_place(value.ty.clone())?;
                            body.store_place(temporary, value)?;
                            fields.push((name, temporary));
                        }
                        payloads.push(fields);
                    }
                    Ok(())
                })?;
                let flags =
                    flags.into_iter().map(|flag| body.load_place(flag)).collect::<Result<Vec<_>, _>>()?;
                let (prefixes, totals) =
                    body.workgroup_scan(&flags, &neutral, lane.clone(), width, |body, left, right| {
                        left.iter()
                            .zip(right)
                            .map(|(a, b)| body.binary(BinaryOperator::Add, a.clone(), b.clone()))
                            .collect()
                    })?;
                let mut next = Vec::new();
                for ((((flag, prefix), total), base), fields) in
                    flags.into_iter().zip(prefixes).zip(totals).zip(carried).zip(payloads)
                {
                    let keep = body.binary(BinaryOperator::NotEqual, flag, Self::number(0))?;
                    let destination = body.binary(BinaryOperator::Add, base.clone(), prefix)?;
                    body.when(keep, |body| {
                        for (name, temporary) in fields {
                            let value = body.load_place(temporary)?;
                            body.write_array(name, &[destination.clone()], value)?;
                        }
                        Ok(())
                    })?;
                    next.push(body.binary(BinaryOperator::Add, base, total)?);
                }
                Ok(next)
            },
        )?;
        self.pack_results(totals)
    }
}
