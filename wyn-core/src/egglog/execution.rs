//! Freeze dispatch recipes and storage choices after the final graph rewrites.
use super::analysis::Liveness;
use super::bindings::Bindings;
use super::cost::{CollectiveWork, CostModel};
use super::dataflow::ir::Code;
use super::host;
use super::query::Query;
use super::{output_error as error, representation, InterfacesPrepared, OptimizeError, Program};
use crate::egglog::graph::CollectiveInput;
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
    inputs: LookupMap<Value, Vec<Value>>,
    captures: LookupMap<Value, Vec<Value>>,
    launches: LookupMap<Value, Launch>,
    columns: LookupMap<Value, ColumnScan>,
    column_loads: LookupMap<(Value, Value), usize>,
    algorithms: LookupMap<(Value, bool), KernelAlgorithm>,
    device: LookupSet<Value>,
    workgroups: LookupMap<Value, Workgroup>,
    subgroups: LookupSet<Value>,
    pub reducer_states: LookupMap<(Value, i64, usize), Type>,
    reuse: LookupMap<Value, Vec<Value>>,
    snapshots: LookupSet<Value>,
    pub atomic_arrays: LookupSet<Value>,
    pub retry_safe: LookupSet<Value>,
    pub native_reducers: LookupMap<Value, super::payload::Payload>,
    pub observable: LookupSet<Value>,
    tokens: LookupMap<Value, (Value, i64)>,
    pub speculation: LookupSet<Value>,
    pub lane_dependent_functions: LookupSet<Value>,
    pub storage: StorageShapes,
    pub return_only_collections: LookupSet<Value>,
    pub host_input_lengths: LookupSet<Value>,
    pub output_copy_widths: LookupMap<Value, u32>,
}

#[derive(Default)]
pub(super) struct StorageShapes {
    pub outer_capacity: LookupMap<Value, i64>,
    pub static_count: LookupMap<Value, i64>,
    pub ranked: LookupSet<Value>,
    pub initial_rows: LookupMap<Value, Value>,
    pub fixed_loop: LookupSet<(Value, i64)>,
    pub loop_preserves_length: LookupSet<(Value, i64)>,
    pub reducer_preserves_length: LookupSet<(Value, i64)>,
    components: LookupMap<(Value, Value), Option<Value>>,
}

impl StorageShapes {
    fn read(query: Query<'_>) -> Result<Self, OptimizeError> {
        let mut shapes = Self::default();
        query.for_function("OuterCapacity", |keys, value| {
            shapes.outer_capacity.insert(keys[0], query.0.value_to_base::<i64>(value));
            Ok(())
        })?;
        query.for_function("LogicalOuterCount", |keys, term| {
            if let Some(value) = query.lookup("StaticIndex", term)? {
                shapes.static_count.insert(keys[0], query.0.value_to_base::<i64>(value));
            }
            Ok(())
        })?;
        shapes.ranked = relation_keys(query, "RankedLoopCollection", 0)?;
        query.for_function("LoopInitialRow", |keys, initial| {
            shapes.initial_rows.insert(keys[0], initial);
            Ok(())
        })?;
        for (table, set) in [
            ("FixedLoopState", &mut shapes.fixed_loop),
            ("LoopPreservesOuterLength", &mut shapes.loop_preserves_length),
            (
                "ReducerPreservesOuterLength",
                &mut shapes.reducer_preserves_length,
            ),
        ] {
            query.for_each(table, |row| {
                set.insert((row[0], query.0.value_to_base::<i64>(row[1])));
                Ok(())
            })?;
        }
        query.for_each("ProductComponent", |row| {
            shapes
                .components
                .entry((row[0], row[1]))
                .and_modify(|value| {
                    if *value != Some(row[2]) {
                        *value = None;
                    }
                })
                .or_insert(Some(row[2]));
            Ok(())
        })?;
        Ok(shapes)
    }

    pub fn component(&self, initial: Value, path: Value) -> Result<Option<Value>, OptimizeError> {
        match self.components.get(&(initial, path)) {
            Some(None) => Err(error("ambiguous ProductComponent")),
            Some(Some(value)) => Ok(Some(*value)),
            None => Ok(None),
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct Dispatch {
    pub groups: Option<[Value; 3]>,
    pub local_size: (u32, u32, u32),
    pub width: NonZeroU32,
    pub grid: Option<(u32, u32, u32)>,
}

#[derive(Clone, Copy)]
pub(super) enum LaunchGrouping {
    Single,
    PerElement,
    PerWorkgroup,
}

pub(super) struct Launch {
    pub extents: Vec<Value>,
    grouping: [LaunchGrouping; 2],
    minimum: [[i64; 2]; 2],
}

impl Launch {
    fn read(
        query: Query<'_>,
        operation: Value,
        columns: Option<ColumnScan>,
    ) -> Result<Self, OptimizeError> {
        let extents = if let Some(columns) = columns {
            vec![columns.width]
        } else {
            let rank = super::analysis::count(query, "LaunchRank", operation)?;
            if rank < 0 {
                return Err(error("negative launch rank"));
            }
            let domain = query.0.operation_domain(operation)?;
            (0..rank).map(|slot| query.0.domain_extent(domain, slot)).collect::<Result<_, _>>()?
        };
        let mut grouping = [LaunchGrouping::Single; 2];
        let mut minimum = [[0; 2]; 2];
        for atomic in [false, true] {
            let value = query.required("SelectedLaunchGrouping", (operation, atomic))?;
            let (kind, _) =
                query.application(value, &["SingleGroup", "ElementGroups", "WorkgroupGroups"])?;
            grouping[usize::from(atomic)] = match kind {
                _ if columns.is_some() => LaunchGrouping::PerWorkgroup,
                "SingleGroup" => LaunchGrouping::Single,
                "ElementGroups" => LaunchGrouping::PerElement,
                "WorkgroupGroups" => LaunchGrouping::PerWorkgroup,
                _ => unreachable!(),
            };
            for empty in [false, true] {
                let value = query.required("LaunchMinimum", (operation, atomic, empty))?;
                minimum[usize::from(atomic)][usize::from(empty)] = query.0.value_to_base::<i64>(value);
            }
        }
        Ok(Self {
            extents,
            grouping,
            minimum,
        })
    }

    pub fn grouping(&self, atomic: bool) -> LaunchGrouping {
        self.grouping[usize::from(atomic)]
    }

    pub fn minimum(&self, atomic: bool, empty: bool) -> i64 {
        self.minimum[usize::from(atomic)][usize::from(empty)]
    }
}

#[derive(Clone, Copy)]
pub(super) struct ColumnScan {
    pub source: Value,
    pub width: Value,
    pub map: Value,
    pub count: Value,
}

#[derive(Clone, Copy)]
pub(super) enum KernelAlgorithm {
    Parallel,
    Compact,
    Ordered,
}

#[derive(Clone, Copy, Default)]
pub(super) struct Workgroup {
    pub per_map_element: bool,
    pub collective: bool,
    pub scan_size: Option<u32>,
    pub map_suffix: Option<Value>,
    pub map_loop: bool,
    pub uniform_loop: bool,
    pub accumulator: bool,
}

impl Execution {
    pub fn prepare(
        program: &mut Program<'_, InterfacesPrepared>,
        live: &Liveness,
        speculate: LookupSet<Value>,
    ) -> Result<Self, OptimizeError> {
        let _timing = super::timing::span("physical execution planning");
        let selection = super::timing::span("Rust dispatch selection");
        let mut execution = Self::select(&mut program.graph)?;
        let loop_scans = execution.workgroup_loop_scans(Query(&program.graph))?;
        drop(selection);
        let features = super::timing::span("egglog dispatch policies");
        program.graph.update(|mut sink| {
            for (outer, loop_op, body, scan, size) in loop_scans {
                sink.add("WorkgroupLoopScan", (outer, loop_op, body, scan, size))?;
                sink.set("WorkgroupLoopSafe", outer, true)?;
            }
            for &operation in execution.dispatches.keys() {
                sink.add("EmittedDispatch", operation)?;
            }
            Ok(())
        })?;
        program.graph.parse_and_run_program(
            None,
            "(run-schedule (saturate execution-features) (saturate execution-atomics))",
        )?;
        super::planning::work::solve(&mut program.graph)?;
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
        drop(features);
        let _freeze = super::timing::span("Rust execution plan readout");
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
        execution.freeze_kernels(query)?;
        execution.storage = StorageShapes::read(query)?;
        execution.lane_dependent_functions = true_keys(query, "LaneDependentFunction")?;
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
        query.for_each("CanReuse", |row| {
            execution.reuse.entry(row[0]).or_default().push(row[1]);
            Ok(())
        })?;
        execution.snapshots = relation_keys(query, "CanSnapshotCopy", 0)?;
        execution.atomic_arrays = relation_keys(query, "AtomicAccumulatorArray", 2)?;
        execution.retry_safe = true_keys(query, "RetrySafe")?;
        query.for_function("NativeBinaryReducer", |keys, payload| {
            execution.native_reducers.insert(keys[0], super::payload::read(query.0, payload));
            Ok(())
        })?;
        query.for_function("TokenCapability", |keys, value| {
            let binding = query.fields("TokenBindingAt", value)?;
            execution.tokens.insert(keys[0], (binding[0], query.0.value_to_base::<i64>(binding[1])));
            Ok(())
        })?;
        for relation in ["OperationRead", "OperationWrite", "OperationTrap"] {
            query.for_each(relation, |row| {
                execution.observable.insert(row[0]);
                Ok(())
            })?;
        }
        execution.speculation = speculate;
        execution.return_only_collections = true_keys(query, "ReturnOnlyCollection")?;
        execution.host_input_lengths = relation_keys(query, "HostInputLength", 0)?;
        query.for_function("ParallelPublicationWidth", |keys, width| {
            let width = u32::try_from(query.0.value_to_base::<i64>(width))
                .map_err(|_| error("invalid output copy workgroup size"))?;
            execution.output_copy_widths.insert(keys[0], width);
            Ok(())
        })?;
        for (table, names) in [
            ("OperationUse", &mut execution.inputs),
            ("RegionCapture", &mut execution.captures),
        ] {
            query.for_each(table, |row| {
                if let Some(name) = query.value_name(row[1])? {
                    names.entry(row[0]).or_default().push(name);
                }
                Ok(())
            })?;
        }
        execution.plan_regions(program, live)?;
        Ok(execution)
    }

    // Follow actual bodies to find the while loops and scans that share a workgroup.
    fn workgroup_loop_scans(
        &self,
        query: Query<'_>,
    ) -> Result<Vec<(Value, Value, Value, Value, i64)>, OptimizeError> {
        let graph = query.0;
        let mut candidates = Vec::new();
        for (&outer, dispatch) in &self.dispatches {
            if dispatch.groups.is_some()
                || graph.result_count(outer) != 1
                || query.enode("Collected", graph.result_kind(outer, 0)?)?.is_none()
                || graph.domain_rank(graph.operation_domain(outer)?)? != 1
            {
                continue;
            }
            let lane = graph.control_region(outer, 0)?;
            for &loop_op in graph.region_order(lane)? {
                if !matches!(graph.code(loop_op)?, Code::WhileLoop) {
                    continue;
                }
                let body = graph.control_region(loop_op, 0)?;
                for &scan in graph.region_order(body)? {
                    if !matches!(graph.code(scan)?, Code::Parallel)
                        || query
                            .enode("InvocationLocal", query.required("OperationExecution", scan)?)?
                            .is_none()
                    {
                        continue;
                    }
                    let domain = graph.operation_domain(scan)?;
                    if graph.domain_rank(domain)? != 1 {
                        continue;
                    }
                    let Some(size) = query.lookup("StaticIndex", graph.domain_extent(domain, 0)?)? else {
                        continue;
                    };
                    let size = graph.value_to_base::<i64>(size);
                    if size <= 0 {
                        continue;
                    }
                    for slot in 0..graph.output_count(scan) {
                        if query.enode("PrefixOutput", graph.output_role(scan, slot)?)?.is_some() {
                            candidates.push((outer, loop_op, body, scan, size));
                            break;
                        }
                    }
                }
            }
        }
        Ok(candidates)
    }

    fn freeze_kernels(&mut self, query: Query<'_>) -> Result<(), OptimizeError> {
        query.for_function("OperationExecution", |keys, value| {
            if query.enode("DeviceKernel", value)?.is_some() {
                self.device.insert(keys[0]);
            }
            Ok(())
        })?;
        query.for_function("ColumnScan", |keys, value| {
            let fields = query.fields("ScanColumns", value)?;
            self.columns.insert(
                keys[0],
                ColumnScan {
                    source: fields[0],
                    width: fields[1],
                    map: fields[2],
                    count: fields[3],
                },
            );
            Ok(())
        })?;
        query.for_function("ColumnScanLoad", |keys, value| {
            let slot = usize::try_from(query.0.value_to_base::<i64>(value))
                .map_err(|_| error("invalid column scan input"))?;
            self.column_loads.insert((keys[0], keys[1]), slot);
            Ok(())
        })?;
        for (&operation, dispatch) in &self.dispatches {
            if dispatch.groups.is_none() {
                self.launches.insert(
                    operation,
                    Launch::read(query, operation, self.columns.get(&operation).copied())?,
                );
            }
        }
        query.for_function("SelectedKernelAlgorithm", |keys, value| {
            let (kind, _) =
                query.application(value, &["ParallelKernel", "CompactionKernel", "OrderedKernel"])?;
            let algorithm = match kind {
                "ParallelKernel" => KernelAlgorithm::Parallel,
                "CompactionKernel" => KernelAlgorithm::Compact,
                "OrderedKernel" => KernelAlgorithm::Ordered,
                _ => unreachable!(),
            };
            self.algorithms.insert((keys[0], query.0.value_to_base::<bool>(keys[1])), algorithm);
            Ok(())
        })?;
        self.subgroups = relation_keys(query, "UseNativeSubgroups", 0)?;
        query.for_function("WorkgroupMap", |keys, flag| {
            if query.0.value_to_base::<bool>(flag) {
                self.workgroups.entry(keys[0]).or_default().per_map_element = true;
            }
            Ok(())
        })?;
        for (table, flag) in [
            (
                "WorkgroupCollective",
                (|plan: &mut Workgroup| &mut plan.collective) as fn(&mut Workgroup) -> &mut bool,
            ),
            ("WorkgroupLoopMap", |plan| &mut plan.map_loop),
            ("WorkgroupLoop", |plan| &mut plan.uniform_loop),
            ("WorkgroupAccumulator", |plan| &mut plan.accumulator),
        ] {
            query.for_each(table, |row| {
                *flag(self.workgroups.entry(row[0]).or_default()) = true;
                Ok(())
            })?;
        }
        query.for_function("WorkgroupScan", |keys, width| {
            let width = u32::try_from(query.0.value_to_base::<i64>(width))
                .map_err(|_| error("invalid workgroup scan size"))?;
            self.workgroups.entry(keys[0]).or_default().scan_size = Some(width);
            Ok(())
        })?;
        query.for_function("WorkgroupMapSuffix", |keys, suffix| {
            self.workgroups.entry(keys[0]).or_default().map_suffix = Some(suffix);
            Ok(())
        })
    }

    pub fn launch(&self, operation: Value) -> Result<&Launch, OptimizeError> {
        self.launches.get(&operation).ok_or_else(|| error("automatic launch plan missing"))
    }

    pub fn column_scan(&self, operation: Value) -> Option<ColumnScan> {
        self.columns.get(&operation).copied()
    }

    pub fn column_load(&self, scan: Value, step: Value) -> Option<usize> {
        self.column_loads.get(&(scan, step)).copied()
    }

    pub fn algorithm(&self, operation: Value, atomic: bool) -> Result<KernelAlgorithm, OptimizeError> {
        self.algorithms.get(&(operation, atomic)).copied().ok_or_else(|| error("kernel algorithm missing"))
    }

    pub fn is_device(&self, operation: Value) -> bool {
        self.device.contains(&operation)
    }

    pub fn workgroup(&self, operation: Value) -> Workgroup {
        self.workgroups.get(&operation).copied().unwrap_or_default()
    }

    pub fn use_subgroups(&self, operation: Value) -> bool {
        self.subgroups.contains(&operation)
    }

    pub fn inputs(&self, operation: Value) -> &[Value] {
        self.inputs.get(&operation).map_or(&[], Vec::as_slice)
    }

    pub fn captures(&self, region: Value) -> &[Value] {
        self.captures.get(&region).map_or(&[], Vec::as_slice)
    }

    pub fn reuse_candidates(&self, value: Value) -> &[Value] {
        self.reuse.get(&value).map_or(&[], Vec::as_slice)
    }

    pub fn can_snapshot(&self, value: Value) -> bool {
        self.snapshots.contains(&value)
    }

    pub fn token_binding(&self, token: Value) -> Result<(Value, i64), OptimizeError> {
        self.tokens.get(&token).copied().ok_or_else(|| error("accumulator capability has no backing"))
    }

    pub fn select(graph: &mut Graph) -> Result<Self, OptimizeError> {
        graph.parse_and_run_program(None, "(run-schedule (saturate reducer-state))")?;
        let query = Query(graph);
        let mut execution = Self::default();
        for (function, _) in graph.entries() {
            execution.region(query, graph.function_root(function)?)?;
        }
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
                    let domain = query.0.operation_domain(operation)?;
                    if query.0.domain_rank(domain)? == 0 {
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
    let domain = query.0.operation_domain(operation)?;
    let rank = query.0.domain_rank(domain)?;
    if rank == 0 {
        return Err(error("automatic dispatch requires an iteration axis"));
    }
    let mut indexed_updates = false;
    let mut state_bytes = 0u64;
    for kind in query.0.parallel_results(operation)?.kinds.into_iter().flatten() {
        indexed_updates |= query.enode("Contributed", kind)?.is_some();
    }
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
        let extent = query.0.domain_extent(domain, rank - 1)?;
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

fn relation_keys(
    query: Query<'_>,
    relation: &str,
    column: usize,
) -> Result<LookupSet<Value>, OptimizeError> {
    let mut values = LookupSet::new();
    query.for_each(relation, |row| {
        values.insert(row[column]);
        Ok(())
    })?;
    Ok(values)
}

fn true_keys(query: Query<'_>, function: &str) -> Result<LookupSet<Value>, OptimizeError> {
    let mut values = LookupSet::new();
    query.for_function(function, |keys, flag| {
        if query.0.value_to_base::<bool>(flag) {
            values.insert(keys[0]);
        }
        Ok(())
    })?;
    Ok(values)
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
    let domain = query.0.operation_domain(operation)?;
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
    live: &'a Liveness,
    outer_sources: &'a LookupMap<Value, Vec<Value>>,
}

impl Execution {
    fn plan_regions(
        &mut self,
        program: &Program<'_, InterfacesPrepared>,
        live: &Liveness,
    ) -> Result<(), OptimizeError> {
        let query = Query(&program.graph);
        let generated =
            query.0.value_to_base::<bool>(query.required("GeneratedDispatches", RawValues(Vec::new()))?);
        let mut outer_sources = LookupMap::<_, Vec<_>>::default();
        query.for_each("OuterShapeSource", |row| {
            outer_sources.entry(row[0]).or_default().push(row[1]);
            Ok(())
        })?;
        let planning = RegionPlanning {
            query,
            live,
            outer_sources: &outer_sources,
        };
        let roots = program.graph.entries();
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
                    let packed = query.0.has_collective_inputs(operation, CollectiveInput::Predicate);
                    let domain = query.0.operation_domain(operation)?;
                    Some(Epilogue {
                        operation: operations[end],
                        preparation: operations[i + 1..end].to_vec(),
                        cooperative: !packed,
                        capacity: query.0.domain_extent(domain, 0_i64)?,
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
                        let inherited = planning.outer_sources.get(&name).is_some_and(|sources| {
                            sources.iter().any(|source| self.host_shapes.contains(source))
                        });
                        if known || inherited || host_values.contains_key(&name) {
                            self.host_shapes.insert(name);
                        }
                    }
                }
            }
            let live = if boundary
                || matches!(
                    kind,
                    StepKind::Dispatch { .. } | StepKind::Branch | StepKind::Loop | StepKind::Accumulator
                ) {
                planning.live.after(region, i)
            } else {
                LookupSet::default()
            };
            let mut cpu_captures = Vec::new();
            if let StepKind::Dispatch { epilogue, .. } = &kind {
                let mut lanes = vec![query.0.control_region(operation, 0_i64)?];
                if let Some(epilogue) = epilogue {
                    lanes.push(query.0.control_region(epilogue.operation, 0_i64)?);
                }
                for lane in lanes {
                    for &name in self.captures(lane) {
                        if let Some(value) = host_values.get(&name) {
                            if host::cpu_available(value)
                                && matches!(value, ScalarExpr::Apply { op, .. } if !op.starts_with("to-"))
                            {
                                cpu_captures.push(name);
                            }
                        }
                    }
                }
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
        for clause in 0..query.0.result_count(operation) {
            if query.enode("Collected", query.0.result_kind(operation, clause)?)?.is_none() {
                return Ok(false);
            }
        }
        let domain = query.0.operation_domain(operation)?;
        let mut sizes = Vec::new();
        for slot in 0..query.0.domain_rank(domain)? {
            if let Some(size) = host::scalar_index(query, values, query.0.domain_extent(domain, slot)?)? {
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
