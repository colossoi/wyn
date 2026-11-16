//! Freeze dispatch recipes and storage choices after the final graph rewrites.
use super::allocation::{ArrayOutput, Output, Storage};
use super::analysis::count;
use super::bindings::Bindings;
use super::cost::{CollectiveWork, CostModel};
use super::dataflow::ir::{self, Code};
use super::host;
use super::payload::{read, Payload};
use super::query::Query;
use super::{output_error as error, representation, OptimizeError, Optimized, Program};
use crate::builtins::select;
use crate::host::ScalarExpr;
use crate::op::BinaryOperator;
use crate::ssa::layout::{storage_elem_stride, storage_value_type};
use crate::ssa::types::AtomicOp;
use crate::types::{self, Type, TypeExt, TypeName};
use crate::{LookupMap, LookupSet, StableMap};
use egglog_engine::sort::Z;
use egglog_engine::{EGraph, RawValues, Value};
use std::num::NonZeroU32;

#[derive(Default)]
pub(super) struct Execution {
    pub entries: LookupMap<i64, Vec<Value>>,
    dispatches: LookupMap<Value, Dispatch>,
    pub collectives: LookupMap<Value, Collective>,
    pub regions: LookupMap<Value, Vec<Step>>,
    pub local_operations: LookupSet<Value>,
    pub host_shapes: LookupSet<Value>,
    pub storage: Storage,
    pub atomics: LookupMap<(Value, i64), AtomicUpdate>,
    pub algorithms: LookupMap<Value, Algorithm>,
    pub subgroups: LookupSet<Value>,
    pub launches: LookupMap<Value, Launch>,
    pub reducer_states: LookupMap<(Value, i64, usize), Type>,
    pub speculated: LookupSet<Value>,
}

#[derive(Clone, Copy)]
pub(super) struct Dispatch {
    pub groups: Option<[Value; 3]>,
    pub local_size: (u32, u32, u32),
    pub width: NonZeroU32,
    pub grid: Option<(u32, u32, u32)>,
}

impl Execution {
    pub fn prepare(program: &Program<'_, Optimized>) -> Result<Self, OptimizeError> {
        let graph = &program.graph;
        let query = Query(graph);
        let mut execution = Self::select(graph)?;
        for (&operation, dispatch) in &execution.dispatches {
            if let Some(collective) = collective(query, operation, *dispatch, program.costs)? {
                execution.collectives.insert(operation, collective);
            }
        }
        execution.plan_regions(query)?;
        execution.plan_states(query)?;
        execution.storage = Storage::select(program, &execution)?;
        execution.plan_kernels(query)?;
        Ok(execution)
    }

    pub fn select(graph: &EGraph) -> Result<Self, OptimizeError> {
        let query = Query(graph);
        let mut execution = Self::default();
        query.for_function("EntryOrder", |keys, owner| {
            let root = query.required("FunctionRoot", keys[0])?;
            let mut stages = Vec::new();
            execution.region(query, root, &mut stages)?;
            execution.entries.insert(graph.value_to_base::<i64>(owner), stages);
            Ok(())
        })?;
        Ok(execution)
    }

    fn region(
        &mut self,
        query: Query<'_>,
        region: Value,
        stages: &mut Vec<Value>,
    ) -> Result<(), OptimizeError> {
        for operation in query.operations(region)? {
            let selected = query.lookup("OperationExecution", operation)?;
            let grid =
                selected.map(|selected| query.enode("DeviceKernel", selected)).transpose()?.flatten();
            if let Some(grid) = grid {
                let width = if let Some(selected_width) = query.lookup("OperationWidth", operation)? {
                    u32::try_from(query.0.value_to_base::<i64>(selected_width))
                        .ok()
                        .and_then(NonZeroU32::new)
                } else {
                    let Some(selected) = selected else {
                        return Err(error("dispatch execution missing"));
                    };
                    NonZeroU32::new(width(query, operation, selected)?)
                };
                let Some(width) = width else {
                    return Err(error("invalid workgroup size"));
                };
                let (groups, local_size, grid) = if query.enode("AutomaticGrid", grid[0])?.is_some() {
                    let domain = query.required("OperationDomain", operation)?;
                    if count(query, "DomainRank", domain)? == 0 {
                        return Err(error("automatic dispatch requires an iteration axis"));
                    }
                    (None, (width.get(), 1, 1), None)
                } else {
                    let fields = query.fields("Grid3D", grid[0])?;
                    let groups = triple(query, fields[0])?;
                    let local = triple(query, fields[1])?;
                    let [x, y, z] = local.map(|index| constant(query, index));
                    let (Some(x), Some(y), Some(z)) = (x?, y?, z?) else {
                        return Err(error("local dimensions must be constant"));
                    };
                    if x.checked_mul(y).and_then(|n| n.checked_mul(z)) != Some(width.get()) {
                        return Err(error("invalid workgroup size"));
                    }
                    let [gx, gy, gz] = groups.map(|index| constant(query, index));
                    let grid = match (gx?, gy?, gz?) {
                        (Some(x), Some(y), Some(z)) => Some((x, y, z)),
                        _ => None,
                    };
                    if grid.is_some_and(|(x, y, z)| [x, y, z].into_iter().any(|n| n == 0 || n > 65_535)) {
                        return Err(error("dispatch grid axis must be in 1..=65535"));
                    }
                    (Some(groups), (x, y, z), grid)
                };
                self.dispatches.insert(
                    operation,
                    Dispatch {
                        groups,
                        local_size,
                        width,
                        grid,
                    },
                );
                stages.push(operation);
                continue;
            }
            // Nested execution is explicit; callbacks and reducer regions stay local.
            let mut children = Vec::new();
            query.for_function("ControlRegion", |keys, child| {
                if keys[0] == operation {
                    children.push((query.0.value_to_base::<i64>(keys[1]), child));
                }
                Ok(())
            })?;
            children.sort_by_key(|&(slot, _)| slot);
            for (_, child) in children {
                self.region(query, child, stages)?;
            }
        }
        Ok(())
    }

    pub fn dispatch(&self, stage: Value) -> Result<&Dispatch, OptimizeError> {
        let Some(dispatch) = self.dispatches.get(&stage) else {
            return Err(error("operation is outside the emitted dispatch interfaces"));
        };
        Ok(dispatch)
    }
}

/// Select a portable workgroup width from the final reducer-state layout.
pub(super) fn width(query: Query<'_>, operation: Value, selected: Value) -> Result<u32, OptimizeError> {
    let Some(grid) = query.enode("DeviceKernel", selected)? else {
        return Ok(1);
    };
    if let Some(grid) = query.enode("Grid3D", grid[0])? {
        let [x, y, z] = triple(query, grid[1])?.map(|index| constant(query, index));
        let (Some(x), Some(y), Some(z)) = (x?, y?, z?) else {
            return Err(error("local dimensions must be constant"));
        };
        let Some(width) = x.checked_mul(y).and_then(|n| n.checked_mul(z)).filter(|&n| n > 0) else {
            return Err(error("invalid workgroup size"));
        };
        return Ok(width);
    }
    let domain = query.required("OperationDomain", operation)?;
    let rank = count(query, "DomainRank", domain)?;
    if rank == 0 {
        return Err(error("automatic dispatch requires an iteration axis"));
    }
    let mut indexed_updates = false;
    let mut state_bytes = 0u64;
    for clause in 0..count(query, "ResultCount", operation)? {
        let kind = query.required("ResultKindOf", (operation, clause))?;
        indexed_updates |= query.enode("Contributed", kind)?.is_some();
        if query.enode("Folded", kind)?.is_some() {
            let reducer = query.required("ResultReducer", (operation, clause))?;
            let params = query.sequence("RegionParameter", reducer)?;
            let count = count(query, "ResultNeutralCount", (operation, clause))?;
            let neutral = query.indexed(
                "ResultNeutral",
                &[operation, query.0.base_to_value(clause)],
                count,
            )?;
            for (i, (parameter, neutral)) in params.into_iter().zip(neutral).enumerate() {
                let Some((_, ty)) = representation::runtime_binding(query, parameter)? else {
                    return Err(error("collective state has no runtime type"));
                };
                let ty = representation::reducer_state_type(query, reducer, i, neutral, ty)?;
                let bytes = storage_elem_stride(&storage_value_type(&ty)).unwrap_or(256);
                state_bytes = state_bytes.saturating_add(u64::from(bytes));
            }
        }
    }
    // Target 8 KiB of shared state, retaining at least one subgroup.
    let mut width = if indexed_updates {
        64
    } else if state_bytes > 0 {
        [256, 128, 64, 32]
            .into_iter()
            .find(|width| state_bytes.saturating_mul(u64::from(*width)) <= 8192)
            .unwrap_or(32)
    } else {
        128
    };
    if rank > 1 && state_bytes > 0 {
        let extent = query.required("DomainExtent", (domain, rank - 1))?;
        if let Some(size) = query.lookup("StaticIndex", extent)? {
            if let Ok(size) = u32::try_from(query.0.value_to_base::<i64>(size).max(0)) {
                if let Some(size) = size.checked_next_power_of_two() {
                    width = width.min(size.max(32));
                }
            }
        }
    }
    Ok(width)
}

fn triple(query: Query<'_>, vector: Value) -> Result<[Value; 3], OptimizeError> {
    query.vector(vector)?.try_into().map_err(|_| error("dispatch requires three dimensions"))
}

fn constant(query: Query<'_>, index: Value) -> Result<Option<u32>, OptimizeError> {
    let value = if let Some(fields) = query.enode("IndexConstant", index)? {
        Some(fields[0])
    } else {
        query.lookup("StaticIndex", index)?
    };
    let Some(value) = value else {
        return Ok(None);
    };
    let value = u32::try_from(query.0.value_to_base::<i64>(value))
        .map_err(|_| error("negative or oversized grid dimension"))?;
    Ok(Some(value))
}

#[cfg(test)]
#[path = "execution_tests.rs"]
mod tests;

/// Scratch layout and launch policy for a multi-dispatch collective.
#[derive(Clone)]
pub(super) struct Collective {
    pub extent: Value,
    pub types: Vec<Type>,
    pub group_limit: u32,
    pub carry_width: u32,
    pub scan: bool,
}

fn collective(
    query: Query<'_>,
    operation: Value,
    dispatch: Dispatch,
    costs: CostModel,
) -> Result<Option<Collective>, OptimizeError> {
    let domain = query.required("OperationDomain", operation)?;
    let axes = representation::domain_axes(query, domain)?;

    if axes.len() != 1 || dispatch.groups.is_some() {
        return Ok(None);
    }
    let axis = axes[0];
    if query
        .lookup("StaticIndex", axis.extent)?
        .is_some_and(|n| query.0.value_to_base::<i64>(n) <= i64::from(dispatch.width.get()))
    {
        return Ok(None);
    }
    let mut types = Vec::new();
    let mut work = CollectiveWork {
        map_ops: region_work(query, query.required("ControlRegion", (operation, 0_i64))?)?,
        ..Default::default()
    };
    for clause in 0..count(query, "ResultCount", operation)? {
        let kind = query.required("ResultKindOf", (operation, clause))?;
        let Some(fold) = query.enode("Folded", kind)? else {
            if query.enode("Collected", kind)?.is_some() {
                continue;
            }
            return Ok(None);
        };
        if query.lookup("ResultPredicate", (operation, clause))?.is_some() {
            return Ok(None);
        }
        work.scan |= query.enode("Total", fold[0])?.is_none();
        let reducer = query.required("ResultReducer", (operation, clause))?;
        work.combine_ops = work.combine_ops.saturating_add(region_work(query, reducer)?);
        if query.contains("RegionTrap", reducer)? {
            return Ok(None);
        }
        let mut writes = false;
        query.for_each("RegionWrite", |row| {
            writes |= row[0] == reducer;
            Ok(())
        })?;
        if writes {
            return Ok(None);
        }
        let params = query.sequence("RegionParameter", reducer)?;
        let count = count(query, "ResultNeutralCount", (operation, clause))?;
        let neutral = query.indexed(
            "ResultNeutral",
            &[operation, query.0.base_to_value(clause)],
            count,
        )?;
        for (i, (&parameter, neutral)) in params.iter().zip(neutral).enumerate() {
            let Some((_, ty)) = representation::runtime_binding(query, parameter)? else {
                return Ok(None);
            };
            let ty = representation::reducer_state_type(query, reducer, i, neutral, ty)?;
            let ty = storage_value_type(&ty);
            let Some(bytes) = storage_elem_stride(&ty) else {
                return Ok(None);
            };
            work.state_bytes = work.state_bytes.saturating_add(u64::from(bytes));
            types.push(ty);
        }
    }
    if types.is_empty() {
        return Ok(None);
    }
    // Estimate contribution traffic at the reducer-state stride. Unknown
    // extents use the residency cap, clamped by the actual host extent.
    work.input_bytes = work.state_bytes;
    work.output_bytes = if work.scan { work.state_bytes } else { 0 };
    let selected_groups = if let Some(count) = query.lookup("StaticIndex", axis.extent)? {
        work.elements = query.0.value_to_base::<i64>(count).max(0) as u64;
        Some(costs.select(work, dispatch.width.get(), 256).map_err(error)?)
    } else {
        None
    };

    Ok(Some(Collective {
        extent: axis.extent,
        types,
        group_limit: selected_groups.unwrap_or(costs.resident_workgroups.min(256)),
        carry_width: selected_groups
            .map(|groups| groups.next_power_of_two().max(32).min(dispatch.width.get()))
            .unwrap_or(dispatch.width.get()),
        scan: work.scan,
    }))
}

fn region_work(query: Query<'_>, region: Value) -> Result<u64, OptimizeError> {
    Ok(query
        .lookup("RegionWork", region)?
        .and_then(|work| u64::try_from(&*query.0.value_to_base::<Z>(work)).ok())
        .unwrap_or(65))
}

#[derive(Clone)]
pub(super) struct Step {
    pub operation: Value,
    pub boundary: bool,
    pub live: LookupSet<Value>,
    pub cpu_captures: Vec<Value>,
    pub kind: StepKind,
}

#[derive(Clone)]
pub(super) enum StepKind {
    Shader,
    Branch,
    Loop,
    ScalarLoop,
    Accumulator,
    Dispatch {
        terminal: bool,
        epilogue: Option<Epilogue>,
    },
}

#[derive(Clone)]
pub(super) struct Epilogue {
    pub operation: Value,
    pub preparation: Vec<Value>,
    pub cooperative: bool,
    pub capacity: Value,
}

impl Execution {
    fn plan_regions(&mut self, query: Query<'_>) -> Result<(), OptimizeError> {
        let generated =
            query.0.value_to_base::<bool>(query.required("GeneratedDispatches", RawValues(Vec::new()))?);
        let mut roots = Vec::new();
        query.for_function("EntryOrder", |keys, _| {
            roots.push(keys[0]);
            Ok(())
        })?;
        for function in roots {
            let root = query.required("FunctionRoot", function)?;
            let mut host_values = Bindings::default();
            for (i, parameter) in query.sequence("RegionParameter", root)?.into_iter().enumerate() {
                if let Some((name, ty)) = representation::runtime_binding(query, parameter)? {
                    host_values.insert(name, ScalarExpr::Local(format!("parameter_{i}")));
                    if ty.is_array() {
                        self.host_shapes.insert(name);
                    }
                }
            }
            let invocation = query
                .lookup("InvocationEntry", function)?
                .is_some_and(|value| query.0.value_to_base::<bool>(value));
            self.plan_region(query, function, root, true, generated && !invocation, host_values)?;
        }
        Ok(())
    }

    fn plan_region(
        &mut self,
        query: Query<'_>,
        function: Value,
        region: Value,
        terminal: bool,
        generated: bool,
        mut host_values: Bindings<Value, ScalarExpr>,
    ) -> Result<(), OptimizeError> {
        let operations = query.operations(region)?;
        let mut steps = Vec::new();
        let mut ordered = false;
        let mut host_ordered = Vec::new();
        let mut i = 0;
        while i < operations.len() {
            let operation = operations[i];
            let code = ir::code(query.0, query.required("OperationPayload", operation)?);
            let pattern = query.sequence("Output", operation)?;
            let mut boundary = query.contains("RequiresSnapshot", operation)?;
            let kind = if self.dispatches.contains_key(&operation) {
                let dispatch = self.dispatch(operation)?;
                boundary |=
                    ordered || !self.launch_evaluates(query, operation, &host_values, &host_ordered)?;
                let epilogue_index = if generated && dispatch.groups.is_none() {
                    operations
                        .iter()
                        .enumerate()
                        .skip(i + 1)
                        .map(|(index, &next)| {
                            query
                                .contains("ScanEpilogue", (operation, next))
                                .map(|yes| yes.then_some(index))
                        })
                        .collect::<Result<Vec<_>, _>>()?
                        .into_iter()
                        .flatten()
                        .next()
                } else {
                    None
                };
                let epilogue = if let Some(end) = epilogue_index {
                    let mut packed = false;
                    query.for_function("ResultPredicate", |keys, _| {
                        packed |= keys[0] == operation;
                        Ok(())
                    })?;
                    let domain = query.required("OperationDomain", operation)?;
                    Some(Epilogue {
                        operation: operations[end],
                        preparation: operations[i + 1..end].to_vec(),
                        cooperative: !packed,
                        capacity: query.required("DomainExtent", (domain, 0_i64))?,
                    })
                } else {
                    None
                };
                let end = epilogue_index.unwrap_or(i);
                let last = terminal
                    && operations[end + 1..].iter().try_fold(true, |safe, &next| {
                        Ok::<_, OptimizeError>(
                            safe && !self.dispatches.contains_key(&next)
                                && query
                                    .lookup("PureOperation", next)?
                                    .is_some_and(|value| query.0.value_to_base::<bool>(value)),
                        )
                    })?;
                i = end;
                ordered = false;
                host_ordered.clear();
                StepKind::Dispatch {
                    terminal: last,
                    epilogue,
                }
            } else if matches!(code, Code::Branch | Code::CountedLoop | Code::WhileLoop) {
                let children = control_regions(query, operation)?;
                let interface = if matches!(code, Code::Branch) {
                    pattern.clone()
                } else {
                    query.sequence(
                        "RegionParameter",
                        query.required("ControlRegion", (operation, 0_i64))?,
                    )?
                };
                let buffer_result = generated
                    && interface.iter().try_fold(false, |any, &binding| {
                        Ok::<_, OptimizeError>(
                            any || representation::runtime_binding(query, binding)?
                                .is_some_and(|(_, ty)| contains_array_descriptor(&ty)),
                        )
                    })?;
                let host_effect = children.iter().try_fold(false, |any, &child| {
                    Ok::<_, OptimizeError>(any || self.has_dispatch(query, child)?)
                })?;
                if !matches!(code, Code::Branch)
                    && generated
                    && !pattern.is_empty()
                    && pattern.iter().try_fold(true, |all, &binding| {
                        Ok::<_, OptimizeError>(
                            all && representation::runtime_binding(query, binding)?
                                .is_some_and(|(_, ty)| host::scalar_type(&ty).is_some()),
                        )
                    })?
                    && host::cpu_loop(query, function, &host_values, operation)?.is_some()
                {
                    StepKind::ScalarLoop
                } else if buffer_result || host_effect {
                    for child in children {
                        let values = host_values
                            .iter()
                            .map(|(&name, value)| (name, value.clone()))
                            .collect::<StableMap<_, _>>()
                            .into();
                        self.plan_region(query, function, child, false, generated, values)?;
                    }
                    ordered = false;
                    host_ordered.clear();
                    if matches!(code, Code::Branch) {
                        StepKind::Branch
                    } else {
                        StepKind::Loop
                    }
                } else {
                    StepKind::Shader
                }
            } else if matches!(code, Code::WithAcc) {
                let child = query.required("ControlRegion", (operation, 0_i64))?;
                let values = host_values
                    .iter()
                    .map(|(&name, value)| (name, value.clone()))
                    .collect::<StableMap<_, _>>()
                    .into();
                self.plan_region(
                    query,
                    function,
                    child,
                    terminal && i + 1 == operations.len(),
                    generated,
                    values,
                )?;
                StepKind::Accumulator
            } else {
                StepKind::Shader
            };
            if boundary {
                ordered = false;
                host_ordered.clear();
            }
            if matches!(kind, StepKind::Shader | StepKind::ScalarLoop) {
                if matches!(kind, StepKind::ScalarLoop) {
                    let Some(value) = host::cpu_loop(query, function, &host_values, operation)? else {
                        return Err(error("host loop plan has no expression"));
                    };
                    for (slot, &binding) in pattern.iter().enumerate() {
                        if let Some(name) = query.value_name(binding)? {
                            host_values.insert(
                                name,
                                ScalarExpr::Field {
                                    tuple: Box::new(value.clone()),
                                    index: slot,
                                },
                            );
                        }
                    }
                } else {
                    host::remember(query, function, &mut host_values, operation)?;
                    if matches!(code, Code::Length { axis: 0 }) {
                        let source = query.name(query.required("Input", (operation, 0_i64))?)?;
                        if self.host_shapes.contains(&source) {
                            let name = query.name(query.required("Output", (operation, 0_i64))?)?;
                            let length = query
                                .lookup("LogicalOuterCount", source)?
                                .map(|count| host::scalar_index(query, &host_values, count))
                                .transpose()?
                                .flatten()
                                .unwrap_or_else(|| ScalarExpr::Local(format!("length_{source:?}")));
                            host_values.insert(name, length);
                        }
                    }
                }
                let mut memory = false;
                for table in ["OperationRead", "OperationWrite"] {
                    query.for_each(table, |row| {
                        memory |= row[0] == operation;
                        Ok(())
                    })?;
                }
                if !query.contains("IndexOperation", operation)? {
                    if memory {
                        ordered = true;
                    } else if query.contains("OperationTrap", operation)? {
                        let value = if let [binding] = pattern.as_slice() {
                            query
                                .value_name(*binding)?
                                .and_then(|name| host_values.get(&name))
                                .filter(|value| host::cpu_available(value))
                        } else {
                            None
                        };
                        if let Some(value) = value {
                            host_ordered.push(value.clone());
                        } else {
                            ordered = true;
                        }
                    }
                }
            }
            if matches!(kind, StepKind::Branch) {
                for (slot, &binding) in pattern.iter().enumerate() {
                    let mut known = true;
                    for child in control_regions(query, operation)? {
                        let result = query.name(query.required("RegionYield", (child, slot as i64))?)?;
                        known &= self.host_shapes.contains(&result);
                    }
                    if known {
                        self.host_shapes.insert(query.name(binding)?);
                    }
                }
            }
            for &binding in &pattern {
                if let Some((name, ty)) = representation::runtime_binding(query, binding)? {
                    if ty.is_array() {
                        let known = if let Some(count) = query.lookup("LogicalOuterCount", name)? {
                            host::scalar_index(query, &host_values, count)?.is_some()
                        } else {
                            false
                        };
                        let inherited = query.any("OuterShapeSource", |row| {
                            row[0] == name && self.host_shapes.contains(&row[1])
                        })?;
                        if known || inherited || host_values.contains_key(&name) {
                            self.host_shapes.insert(name);
                        }
                    }
                }
            }
            let mut live = LookupSet::default();
            query.for_each("NameLiveAfter", |row| {
                if row[0] == operations[i] {
                    if let Some(name) = query.value_name(row[1])? {
                        live.insert(name);
                    }
                }
                Ok(())
            })?;
            let mut cpu_captures = Vec::new();
            if let StepKind::Dispatch { epilogue, .. } = &kind {
                let mut lanes = vec![query.required("ControlRegion", (operation, 0_i64))?];
                if let Some(epilogue) = epilogue {
                    lanes.push(query.required("ControlRegion", (epilogue.operation, 0_i64))?);
                }
                query.for_each("RegionCapture", |row| {
                    if lanes.contains(&row[0]) {
                        if let Some(name) = query.value_name(row[1])? {
                            if let Some(value) = host_values.get(&name) {
                                if host::cpu_available(value) && matches!(value, ScalarExpr::Apply { op, .. } if !op.starts_with("to-")) {
                                    cpu_captures.push(name);
                                }
                            }
                        }
                    }
                    Ok(())
                })?;
            }
            steps.push(Step {
                operation,
                boundary,
                live,
                cpu_captures,
                kind,
            });
            i += 1;
        }
        let mut in_dispatch = false;
        for step in &steps {
            if step.boundary {
                in_dispatch = false;
            }
            match &step.kind {
                StepKind::Dispatch { terminal, epilogue } => {
                    self.mark_local(query, step.operation)?;
                    if let Some(epilogue) = epilogue {
                        for &operation in &epilogue.preparation {
                            self.mark_local(query, operation)?;
                        }
                    }
                    in_dispatch = *terminal;
                }
                StepKind::Shader if in_dispatch => self.mark_local(query, step.operation)?,
                StepKind::Branch | StepKind::Loop | StepKind::ScalarLoop => in_dispatch = false,
                _ => {}
            }
        }
        self.regions.insert(region, steps);
        Ok(())
    }

    fn mark_local(&mut self, query: Query<'_>, operation: Value) -> Result<(), OptimizeError> {
        self.local_operations.insert(operation);
        for region in control_regions(query, operation)? {
            for child in query.operations(region)? {
                self.mark_local(query, child)?;
            }
        }
        Ok(())
    }

    fn has_dispatch(&self, query: Query<'_>, region: Value) -> Result<bool, OptimizeError> {
        for operation in query.operations(region)? {
            if self.dispatches.contains_key(&operation) {
                return Ok(true);
            }
            for child in control_regions(query, operation)? {
                if self.has_dispatch(query, child)? {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    fn launch_evaluates(
        &self,
        query: Query<'_>,
        operation: Value,
        values: &Bindings<Value, ScalarExpr>,
        ordered: &[ScalarExpr],
    ) -> Result<bool, OptimizeError> {
        if ordered.is_empty() {
            return Ok(true);
        }
        let dispatch = self.dispatch(operation)?;
        if dispatch.grid.is_some() || dispatch.groups.is_some() {
            return Ok(false);
        }
        for clause in 0..count(query, "ResultCount", operation)? {
            if query.enode("Collected", query.required("ResultKindOf", (operation, clause))?)?.is_none() {
                return Ok(false);
            }
        }
        let domain = query.required("OperationDomain", operation)?;
        let mut sizes = Vec::new();
        for slot in 0..count(query, "DomainRank", domain)? {
            if let Some(size) =
                host::scalar_index(query, values, query.required("DomainExtent", (domain, slot))?)?
            {
                sizes.push(size);
            }
        }
        Ok(ordered.iter().all(|value| sizes.iter().any(|size| host::evaluates(size, value))))
    }
}

fn control_regions(query: Query<'_>, operation: Value) -> Result<Vec<Value>, OptimizeError> {
    let mut children = Vec::new();
    query.for_function("ControlRegion", |keys, child| {
        if keys[0] == operation {
            children.push((query.0.value_to_base::<i64>(keys[1]), child));
        }
        Ok(())
    })?;
    children.sort_by_key(|&(slot, _)| slot);
    Ok(children.into_iter().map(|(_, child)| child).collect())
}

fn contains_array_descriptor(ty: &Type) -> bool {
    (ty.is_array()
        && !(ty.array_variant().is_some_and(types::is_array_variant_composite)
            && matches!(ty.array_size(), Some(Type::Constructed(TypeName::Size(_), _)))))
        || matches!(ty, Type::Constructed(_, fields) if fields.iter().any(contains_array_descriptor))
}

#[derive(Clone)]
pub(super) enum AtomicUpdate {
    Native(AtomicOp),
    CompareExchange(Value),
}

#[derive(Clone, Copy)]
pub(super) enum Algorithm {
    Parallel,
    Ordered,
    Compaction,
}

pub(super) enum Grouping {
    Single,
    PerElement,
    PerWorkgroup,
}

pub(super) struct Launch {
    pub extents: Vec<Value>,
    pub grouping: Grouping,
    pub minimum: i64,
}

impl Execution {
    fn plan_states(&mut self, query: Query<'_>) -> Result<(), OptimizeError> {
        query.for_each("ParallelOperation", |row| {
            let operation = row[0];
            if !query.contains("LiveOperation", operation)? {
                return Ok(());
            }
            for clause in 0..count(query, "ResultCount", operation)? {
                let Some(reducer) = query.lookup("ResultReducer", (operation, clause))? else {
                    continue;
                };
                let parameters = query.sequence("RegionParameter", reducer)?;
                for slot in 0..count(query, "ResultNeutralCount", (operation, clause))? {
                    let neutral = query.required("ResultNeutral", (operation, clause, slot))?;
                    let Some(&parameter) = parameters.get(slot as usize) else {
                        return Err(error("reducer state parameter missing"));
                    };
                    let Some((_, ty)) = representation::runtime_binding(query, parameter)? else {
                        return Err(error("reducer state has no runtime type"));
                    };
                    let ty =
                        representation::reducer_state_type(query, reducer, slot as usize, neutral, ty)?;
                    if ty.is_array()
                        && !matches!(ty.array_size(), Some(Type::Constructed(TypeName::Size(_), _)))
                    {
                        return Err(error(
                            "array-valued collective has no proven fixed reducer state shape",
                        ));
                    }
                    self.reducer_states.insert((operation, clause, slot as usize), ty);
                }
            }
            Ok(())
        })?;
        Ok(())
    }

    fn plan_kernels(&mut self, query: Query<'_>) -> Result<(), OptimizeError> {
        query.for_each("LiveOperation", |row| {
            let operation = row[0];
            if !query.contains("BranchOperation", operation)? {
                return Ok(());
            }
            let types = query
                .sequence("Output", operation)?
                .into_iter()
                .map(|binding| representation::runtime_binding(query, binding))
                .collect::<Result<Vec<_>, _>>()?;
            let mut speculate = types.iter().flatten().all(|(_, ty)| select::supported_type(ty));
            for region in control_regions(query, operation)? {
                for table in ["RegionRead", "RegionWrite", "RegionTrap"] {
                    query.for_each(table, |row| {
                        speculate &= row[0] != region;
                        Ok(())
                    })?;
                }
                speculate &= query.lookup("RegionWork", region)?.is_some_and(|work| {
                    u64::try_from(&*query.0.value_to_base::<Z>(work)).is_ok_and(|work| work <= 4)
                });
            }
            if speculate {
                self.speculated.insert(operation);
            }
            Ok(())
        })?;
        query.for_function("AccumulatorArrayCount", |keys, count| {
            let owner = keys[0];
            let group = query.0.value_to_base::<i64>(keys[1]);
            if query.0.value_to_base::<i64>(count) != 1 {
                return Ok(());
            }
            let name = query.required("AccumulatorArray", (owner, group, 0_i64))?;
            if !query.contains("FreshArray", name)?
                || !self.storage.collections.get(&name).is_some_and(|plan| plan.allocation.device())
            {
                return Ok(());
            }
            let ty = representation::value_type(query, query.required("ValueType", name)?)?;
            if !matches!(
                ty.elem_type(),
                Some(Type::Constructed(TypeName::Int(32) | TypeName::UInt(32), _))
            ) {
                return Ok(());
            }
            let update = if let Some(combine) = query.lookup("AccumulatorCombiner", (owner, group))? {
                let mut writes = false;
                query.for_each("RegionWrite", |row| {
                    writes |= row[0] == combine;
                    Ok(())
                })?;
                if writes || query.contains("RegionTrap", combine)? {
                    return Ok(());
                }
                if let Some(opcode) = query.lookup("NativeBinaryReducer", combine)? {
                    AtomicUpdate::Native(match read(query.0, opcode) {
                        Payload::Binary(BinaryOperator::Add) => AtomicOp::Add,
                        Payload::Binary(BinaryOperator::BitwiseAnd) => AtomicOp::And,
                        Payload::Binary(BinaryOperator::BitwiseOr) => AtomicOp::Or,
                        Payload::Binary(BinaryOperator::BitwiseXor) => AtomicOp::Xor,
                        _ => return Err(error("unknown native integer reducer")),
                    })
                } else {
                    AtomicUpdate::CompareExchange(combine)
                }
            } else if query
                .enode(
                    "IdenticalWrites",
                    query.required("AccumulatorPolicy", (owner, group))?,
                )?
                .is_some()
            {
                AtomicUpdate::Native(AtomicOp::Exchange)
            } else {
                return Ok(());
            };
            self.atomics.insert((owner, group), update);
            Ok(())
        })?;
        let enabled = query
            .lookup("SubgroupsEnabled", RawValues(Vec::new()))?
            .is_some_and(|value| query.0.value_to_base::<bool>(value));
        let mut operations = Vec::new();
        query.for_each("ParallelOperation", |row| {
            if query.contains("LiveOperation", row[0])? {
                operations.push(row[0]);
            }
            Ok(())
        })?;
        for operation in operations {
            let domain = query.required("OperationDomain", operation)?;
            let mut extents = (0..count(query, "DomainRank", domain)?)
                .map(|slot| query.required("DomainExtent", (domain, slot)))
                .collect::<Result<Vec<_>, _>>()?;
            let mut folded = false;
            let mut compact = false;
            let mut contributed = false;
            let mut native = enabled;
            let atomic = self.atomic_contributions(query, operation)?;
            for clause in 0..count(query, "ResultCount", operation)? {
                let kind = query.required("ResultKindOf", (operation, clause))?;
                contributed |= query.enode("Contributed", kind)?.is_some();
                compact |= query.enode("Compacted", kind)?.is_some()
                    || (query.enode("Contributed", kind)?.is_some() && !atomic);
                if query.enode("Folded", kind)?.is_some() {
                    folded = true;
                    let reducer = query.required("ResultReducer", (operation, clause))?;
                    native &= query.contains("NativeAddition", reducer)?;
                    for slot in 0..count(query, "ResultNeutralCount", (operation, clause))? {
                        let neutral = query.required("ResultNeutral", (operation, clause, slot))?;
                        native &= query
                            .lookup("ConstantValue", neutral)?
                            .is_some_and(|value| query.0.value_to_base::<i64>(value) == 0);
                    }
                }
            }
            let rank = extents.len();
            let mut packed = false;
            query.for_function("ResultPredicate", |keys, _| {
                packed |= keys[0] == operation;
                Ok(())
            })?;
            let algorithm = if compact || packed {
                if !folded && !contributed && rank == 1 && self.dispatches.contains_key(&operation) {
                    Algorithm::Compaction
                } else {
                    Algorithm::Ordered
                }
            } else {
                Algorithm::Parallel
            };
            self.algorithms.insert(operation, algorithm);
            if !self.dispatches.contains_key(&operation) {
                continue;
            }
            if folded {
                extents.pop();
            }
            if folded && native {
                self.subgroups.insert(operation);
            }
            let mut terminal = false;
            for (&function, results) in &self.storage.results {
                let root = query.required("FunctionRoot", function)?;
                let parent = query.required("OperationRegion", operation)?;
                terminal |= query.contains("FlowScopeAncestor", (root, parent))?
                    && self.regions.values().flatten().any(|step| {
                        step.operation == operation
                            && matches!(step.kind, StepKind::Dispatch { terminal: true, .. })
                    })
                    && results.layouts.iter().all(|layout| {
                        !matches!(
                            layout,
                            Output::Array {
                                mode: ArrayOutput::Count | ArrayOutput::Pack,
                                ..
                            }
                        )
                    });
            }
            self.launches.insert(
                operation,
                Launch {
                    extents,
                    grouping: if compact {
                        Grouping::Single
                    } else if folded {
                        Grouping::PerElement
                    } else {
                        Grouping::PerWorkgroup
                    },
                    minimum: i64::from(compact || !terminal),
                },
            );
        }
        Ok(())
    }

    fn atomic_contributions(&self, query: Query<'_>, operation: Value) -> Result<bool, OptimizeError> {
        for clause in 0..count(query, "ResultCount", operation)? {
            if query
                .enode(
                    "Contributed",
                    query.required("ResultKindOf", (operation, clause))?,
                )?
                .is_none()
            {
                return Ok(false);
            }
            for field in 0..count(query, "ResultValueCount", (operation, clause))? {
                let name = query.required("ResultValue", (operation, clause, field))?;
                let token = query.fields("AccNameRef", name)?[0];
                let capability =
                    query.fields("TokenBindingAt", query.required("TokenCapability", token)?)?;
                let group = query.0.value_to_base::<i64>(capability[1]);
                if !self.atomics.contains_key(&(capability[0], group)) {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }
}
