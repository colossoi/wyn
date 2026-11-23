//! Freeze dispatch recipes and storage choices after the final graph rewrites.
use super::analysis::count;
use super::bindings::Bindings;
use super::cost::{CollectiveWork, CostModel};
use super::dataflow::ir::Code;
use super::host;
use super::query::Query;
use super::{output_error as error, representation, InterfacesPrepared, OptimizeError, Program};
use crate::egglog::graph::Graph;
use crate::host::ScalarExpr;
use crate::ssa::layout::{storage_elem_stride, storage_value_type};
use crate::tlc::DefMeta;
use crate::types::{self, Type, TypeExt, TypeName};
use crate::{LookupMap, LookupSet, StableMap};
use egglog_engine::sort::Z;
use egglog_engine::{RawValues, Value, Write};
use std::num::NonZeroU32;

#[derive(Default)]
pub(super) struct Execution {
    dispatches: LookupMap<Value, Dispatch>,
    pub collectives: LookupMap<Value, Collective>,
    pub regions: LookupMap<Value, Vec<Step>>,
    pub host_shapes: LookupSet<Value>,
    pub reducer_states: LookupMap<(Value, i64, usize), Type>,
}

#[derive(Clone, Copy)]
pub(super) struct Dispatch {
    pub groups: Option<[Value; 3]>,
    pub local_size: (u32, u32, u32),
    pub width: NonZeroU32,
    pub grid: Option<(u32, u32, u32)>,
}

impl Execution {
    pub fn prepare(program: &mut Program<'_, InterfacesPrepared>) -> Result<Self, OptimizeError> {
        let mut execution = Self::select(&mut program.graph)?;
        program.graph.update(|mut sink| {
            for &operation in execution.dispatches.keys() {
                sink.add("EmittedDispatch", operation)?;
            }
            Ok(())
        })?;
        program.graph.parse_and_run_program(
            None,
            "(run-schedule (saturate execution-features graph-work) (saturate execution-atomics))",
        )?;
        let mut workgroup_bytes = LookupMap::<_, i64>::default();
        Query(&program.graph).for_each("WorkgroupStorageRequest", |row| {
            let bytes = i64::try_from(&*program.graph.value_to_base::<Z>(row[3])).unwrap_or(i64::MAX);
            let total = workgroup_bytes.entry(row[0]).or_default();
            *total = total.saturating_add(bytes);
            Ok(())
        })?;
        if !workgroup_bytes.is_empty() {
            program.graph.update(|mut sink| {
                for (operation, bytes) in workgroup_bytes {
                    sink.set("WorkgroupStorageBytes", operation, bytes)?;
                }
                Ok(())
            })?;
        }
        program.graph.parse_and_run_program(
            None,
            "(run-schedule (saturate execution-policy) (saturate execution-select))",
        )?;
        let query = Query(&program.graph);
        query.for_function("WorkgroupMapWidth", |keys, width| {
            let dispatch = execution
                .dispatches
                .get_mut(&keys[0])
                .ok_or_else(|| error("workgroup scan has no dispatch"))?;
            dispatch.width = u32::try_from(query.0.value_to_base::<i64>(width))
                .ok()
                .and_then(NonZeroU32::new)
                .ok_or_else(|| error("invalid workgroup map width"))?;
            dispatch.local_size = (dispatch.width.get(), 1, 1);
            Ok(())
        })?;
        execution.plan_states(query)?;
        for (&operation, dispatch) in &execution.dispatches {
            if let Some(collective) = collective(
                query,
                operation,
                *dispatch,
                program.costs,
                &execution.reducer_states,
            )? {
                execution.collectives.insert(operation, collective);
            }
        }
        execution.plan_regions(program)?;
        Ok(execution)
    }

    pub fn select(graph: &mut Graph) -> Result<Self, OptimizeError> {
        graph.parse_and_run_program(None, "(run-schedule (saturate reducer-state))")?;
        let query = Query(graph);
        let mut execution = Self::default();
        query.for_function("EntryOrder", |keys, _| {
            execution.region(query, query.0.function_root(keys[0])?)
        })?;
        Ok(execution)
    }

    fn region(&mut self, query: Query<'_>, region: Value) -> Result<(), OptimizeError> {
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
                continue;
            }
            // Nested execution is explicit; callbacks and reducer regions stay local.
            for child in control_regions(query, operation)? {
                self.region(query, child)?;
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
    query.for_function("ResultKindOf", |keys, kind| {
        indexed_updates |= keys[0] == operation && query.enode("Contributed", kind)?.is_some();
        Ok(())
    })?;
    query.for_each("ReducerState", |row| {
        if row[0] == operation {
            let Some((_, ty)) = representation::runtime_binding(query, row[5])? else {
                return Err(error("collective state has no runtime type"));
            };
            let slot = usize::try_from(query.0.value_to_base::<i64>(row[2]))
                .map_err(|_| error("negative reducer state slot"))?;
            let ty = representation::reducer_state_type(query, row[3], slot, row[4], ty)?;
            state_bytes = state_bytes.saturating_add(u64::from(
                storage_elem_stride(&storage_value_type(&ty)).unwrap_or(256),
            ));
        }
        Ok(())
    })?;
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
    if rank > 1
        && state_bytes > 0
        && query.0.value_to_base::<bool>(query.required("SubgroupsEnabled", RawValues(Vec::new()))?)
    {
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
    states: &LookupMap<(Value, i64, usize), Type>,
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
    if !query.contains("CanChunkCollective", operation)? {
        return Ok(None);
    }
    let mut work = CollectiveWork {
        map_ops: region_work(query, query.0.control_region(operation, 0_i64)?)?,
        scan: query.flag("ChunkScan", operation)?,
        ..Default::default()
    };
    for (_, reducer) in query.0.reducers(operation) {
        work.combine_ops = work.combine_ops.saturating_add(region_work(query, reducer)?);
    }
    let mut states = states.iter().filter(|((owner, _, _), _)| *owner == operation).collect::<Vec<_>>();
    states.sort_by_key(|((_, clause, slot), _)| (*clause, *slot));
    let mut types = Vec::new();
    for (_, ty) in states {
        let ty = storage_value_type(ty);
        let Some(bytes) = storage_elem_stride(&ty) else {
            return Ok(None);
        };
        work.state_bytes = work.state_bytes.saturating_add(u64::from(bytes));
        types.push(ty);
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

#[derive(Clone, Copy)]
struct RegionPlanning<'a> {
    query: Query<'a>,
    live_after: &'a LookupMap<Value, LookupSet<Value>>,
}

impl Execution {
    fn plan_regions(&mut self, program: &Program<'_, InterfacesPrepared>) -> Result<(), OptimizeError> {
        let query = Query(&program.graph);
        let generated =
            query.0.value_to_base::<bool>(query.required("GeneratedDispatches", RawValues(Vec::new()))?);
        let mut live_after = LookupMap::<_, LookupSet<_>>::default();
        query.for_each("NameLiveAfter", |row| {
            if let Some(name) = query.value_name(row[1])? {
                live_after.entry(row[0]).or_default().insert(name);
            }
            Ok(())
        })?;
        let planning = RegionPlanning {
            query,
            live_after: &live_after,
        };
        let mut roots = Vec::new();
        query.for_function("EntryOrder", |keys, order| {
            roots.push((keys[0], query.0.value_to_base::<i64>(order)));
            Ok(())
        })?;
        let declarations = program
            .source
            .defs
            .iter()
            .filter_map(|definition| match &definition.meta {
                DefMeta::EntryPoint(entry) => Some((definition.name, &entry.declaration.params)),
                _ => None,
            })
            .collect::<LookupMap<_, _>>();
        for (function, order) in roots {
            let root = query.0.function_root(function)?;
            let Some(&declarations) = declarations.get(program.symbols.resolve(order)) else {
                return Err(error("entry source metadata missing"));
            };
            let mut host_values = Bindings::default();
            for (i, parameter) in query.0.parameters(root).into_iter().enumerate() {
                if let Some((name, ty)) = representation::runtime_binding(query, parameter)? {
                    if !declarations.get(i).is_some_and(|param| param.device_produced) {
                        host_values.insert(name, ScalarExpr::Local(format!("parameter_{i}")));
                    }
                    if ty.is_array() {
                        self.host_shapes.insert(name);
                    }
                }
            }
            let invocation = query
                .lookup("InvocationEntry", function)?
                .is_some_and(|value| query.0.value_to_base::<bool>(value));
            self.plan_region(
                planning,
                function,
                root,
                true,
                generated && !invocation,
                host_values,
            )?;
        }
        Ok(())
    }

    fn plan_region(
        &mut self,
        planning: RegionPlanning<'_>,
        function: Value,
        region: Value,
        terminal: bool,
        generated: bool,
        mut host_values: Bindings<Value, ScalarExpr>,
    ) -> Result<(), OptimizeError> {
        let query = planning.query;
        let operations = query.0.region_order(region)?.to_vec();
        let mut steps = Vec::new();
        let mut ordered = false;
        let mut host_ordered = Vec::new();
        let mut i = 0;
        while i < operations.len() {
            let operation = operations[i];
            let code = query.0.code(operation)?;
            let pattern = query.0.outputs(operation);
            let mut boundary = query.contains("RequiresSnapshot", operation)?;
            let mut scalar_loop = None;
            let kind = if self.dispatches.contains_key(&operation) {
                let dispatch = self.dispatch(operation)?;
                boundary |=
                    ordered || !self.launch_evaluates(query, operation, &host_values, &host_ordered)?;
                let epilogue_index = if generated
                    && dispatch.groups.is_none()
                    && query.lookup("ColumnScan", operation)?.is_none()
                {
                    operations
                        .iter()
                        .enumerate()
                        .skip(i + 1)
                        .map(|(index, &next)| {
                            query.contains("ScanEpilogue", (operation, next)).and_then(|yes| {
                                if !yes {
                                    return Ok(None);
                                }
                                let ready = self.collectives.get(&operation).is_some_and(|plan| plan.scan)
                                    || !query
                                        .contains("EpilogueRequiresCompletedTotal", (operation, next))?;
                                Ok((ready && !query.flag("WorkgroupMap", next)?).then_some(index))
                            })
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
                    query.0.parameters(query.0.control_region(operation, 0_i64)?)
                };
                let buffer_result = generated
                    && interface.iter().try_fold(false, |any, &binding| {
                        Ok::<_, OptimizeError>(
                            any || representation::runtime_binding(query, binding)?
                                .is_some_and(|(_, ty)| contains_array_descriptor(&ty)),
                        )
                    })?;
                let host_effect = children.iter().try_fold(false, |any, &child| {
                    Ok::<_, OptimizeError>(any || query.contains("DispatchInRegion", child)?)
                })?;
                let fixed_state = buffer_result
                    && !matches!(code, Code::Branch)
                    && query
                        .lookup("LoopStateBytes", (operation, query.0.output_count(operation)))?
                        .is_some();
                if !matches!(code, Code::Branch)
                    && generated
                    && !pattern.is_empty()
                    && pattern.iter().try_fold(true, |all, &binding| {
                        Ok::<_, OptimizeError>(
                            all && representation::runtime_binding(query, binding)?
                                .is_some_and(|(_, ty)| host::scalar_type(&ty).is_some()),
                        )
                    })?
                {
                    scalar_loop = host::cpu_loop(query, function, &host_values, operation)?;
                }
                if scalar_loop.is_some() {
                    StepKind::ScalarLoop
                } else if host_effect || (buffer_result && !fixed_state) {
                    for child in children {
                        let values = host_values
                            .iter()
                            .map(|(&name, value)| (name, value.clone()))
                            .collect::<StableMap<_, _>>()
                            .into();
                        self.plan_region(planning, function, child, false, generated, values)?;
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
                let child = query.0.control_region(operation, 0_i64)?;
                boundary |= query.contains("AtomicInitialization", operation)?;
                let values = host_values
                    .iter()
                    .map(|(&name, value)| (name, value.clone()))
                    .collect::<StableMap<_, _>>()
                    .into();
                self.plan_region(
                    planning,
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
                if let Some(value) = scalar_loop {
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
                        let source = query.name(query.0.input(operation, 0_i64)?)?;
                        if self.host_shapes.contains(&source) {
                            let name = query.name(query.0.output(operation, 0_i64)?)?;
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
                        let result = query.name(query.0.yield_value(child, slot as i64)?)?;
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
            let live = planning.live_after.get(&operations[i]).cloned().unwrap_or_default();
            let mut cpu_captures = Vec::new();
            if let StepKind::Dispatch { epilogue, .. } = &kind {
                let mut lanes = vec![query.0.control_region(operation, 0_i64)?];
                if let Some(epilogue) = epilogue {
                    lanes.push(query.0.control_region(epilogue.operation, 0_i64)?);
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
        self.regions.insert(region, steps);
        Ok(())
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

pub(super) fn control_regions(query: Query<'_>, operation: Value) -> Result<Vec<Value>, OptimizeError> {
    Ok(query.0.controls(operation).map(|(_, child)| child).collect())
}

fn contains_array_descriptor(ty: &Type) -> bool {
    (ty.is_array()
        && !(ty.array_variant().is_some_and(types::is_array_variant_composite)
            && matches!(ty.array_size(), Some(Type::Constructed(TypeName::Size(_), _)))))
        || matches!(ty, Type::Constructed(_, fields) if fields.iter().any(contains_array_descriptor))
}

impl Execution {
    fn plan_states(&mut self, query: Query<'_>) -> Result<(), OptimizeError> {
        query.for_each("ReducerState", |row| {
            let operation = row[0];
            if query.lookup("ColumnScan", operation)?.is_some() {
                return Ok(());
            }
            let clause = query.0.value_to_base::<i64>(row[1]);
            let slot = usize::try_from(query.0.value_to_base::<i64>(row[2]))
                .map_err(|_| error("negative reducer state slot"))?;
            let Some((_, ty)) = representation::runtime_binding(query, row[5])? else {
                return Err(error("reducer state has no runtime type"));
            };
            let ty = representation::reducer_state_type(query, row[3], slot, row[4], ty)?;
            if ty.is_array() && !matches!(ty.array_size(), Some(Type::Constructed(TypeName::Size(_), _))) {
                return Err(error(
                    "array-valued collective has no proven fixed reducer state shape",
                ));
            }
            self.reducer_states.insert((operation, clause, slot), ty);
            Ok(())
        })?;
        Ok(())
    }
}
