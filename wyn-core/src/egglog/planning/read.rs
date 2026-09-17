//! Read a completed relational plan directly into compiler arenas.
//! Native egglog values identify resources until their arena IDs are assigned.
use crate::egglog::blocks::{
    BufferData, DispatchData, Function, FunctionKind, GridData, Instruction, Storage, Value,
};
use crate::egglog::data::{
    BufferId, DispatchId, EntryId, ExprId, ExprKind, Ir, OperationId, OutputId, TypeId,
};
use crate::egglog::timing::span;
use crate::egglog::visit::Operand;
use crate::egglog::{OptimizeError, Program, Scheduled};
use crate::types::TypeExt;
use egglog_engine::sort::S;
use egglog_engine::{EGraph, Read, Value as EggValue};
use std::collections::{BTreeMap, BTreeSet, HashMap};

#[derive(Clone, Copy, Debug)]
pub(in crate::egglog) enum Recipe {
    Elements,
    Totals,
    Prefixes,
    Compact,
    Serial,
}

#[derive(Default)]
pub(in crate::egglog) struct Readout {
    pub buffers: HashMap<EggValue, BufferId>,
    pub launches: HashMap<EggValue, DispatchId>,
    pub local_slots: BTreeMap<(OperationId, String, u32), BufferId>,
    pub recipes: BTreeMap<OperationId, Recipe>,
    pub slots: BTreeMap<(OperationId, String, u32), BufferId>,
    pub stages: BTreeMap<(OperationId, String), DispatchId>,
}

pub(in crate::egglog) fn read(
    graph: &EGraph,
    data: &mut Program<Scheduled>,
) -> Result<Readout, OptimizeError> {
    let _timing = span("read completed plan");
    let mut result = Readout::default();
    let operations = keys(graph, "OperationId")?;
    let expressions = keys(graph, "ExprId")?;
    let mut status = Ok(());
    graph.function_entries_while("Plan", |entry| {
        status = graph.read(|state| {
            for (name, recipe) in [
                ("Elements", Recipe::Elements),
                ("Totals", Recipe::Totals),
                ("Prefixes", Recipe::Prefixes),
                ("Compact", Recipe::Compact),
                ("Serial", Recipe::Serial),
            ] {
                let mut selected = false;
                state.enodes_for_eclass(name, entry.output, |_| selected = true)?;
                if selected {
                    result.recipes.insert(OperationId::from(operations[&entry.inputs[0]]), recipe);
                    return Ok(());
                }
            }
            Err(invalid("unknown scheduling recipe"))
        });
        status.is_ok()
    })?;
    status?;

    // Reserve identities before translating extents: a capacity can read the
    // live count stored in another planned resource.
    let mut buffers = HashMap::new();
    rows(graph, "PlannedBuffer", |a| {
        buffers.entry(a[0]).or_insert_with(|| data.state.buffers.alloc_id());
        Ok(())
    })?;
    rows(graph, "ExternalBuffer", |a| {
        let e = ExprId::from(number(graph, a[1])?);
        let ty = &data.ir.types[data.ir.expressions[e].ty].ty;
        if ty.is_array() {
            let Some(element) = ty.elem_type() else {
                return Err(invalid("array resource has no element type"));
            };
            buffers.entry(a[0]).or_insert_with(|| {
                data.state.buffers.alloc(BufferData {
                    name: format!("input{}", e.as_u32()),
                    length: Value::op("length", [Value::Source(e)]),
                    element: element.clone(),
                    storage: Storage::View(e),
                })
            });
        }
        Ok(())
    })?;

    // Decode directly to the final grid/capacity representation. Only compound
    // extents need a child index; memoization visits each such edge once.
    let mut extents = HashMap::new();
    for name in ["Fixed", "Length", "Scalar", "Stored"] {
        let mut status: Result<(), OptimizeError> = Ok(());
        graph.constructor_enodes_while(name, |e| {
            status = (|| {
                let arg = e.children[0];
                let value = match name {
                    "Fixed" => Value::Int(number(graph, arg)?),
                    "Length" => Value::op("length", [Value::Source(ExprId::from(expressions[&arg]))]),
                    "Scalar" => Value::Source(ExprId::from(expressions[&arg])),
                    "Stored" => {
                        let Some(&buffer) = buffers.get(&arg) else {
                            // An unused live-length expression need not have storage.
                            return Ok(());
                        };
                        Value::op("index", [Value::Buffer(buffer), Value::Int(0)])
                    }
                    _ => unreachable!("unknown extent constructor {name}"),
                };
                extents.insert(e.eclass, value);
                Ok(())
            })();
            status.is_ok()
        })?;
        status?;
    }
    let mut chunks = HashMap::new();
    graph.constructor_enodes("ChunkCount", |e| {
        chunks.insert(e.eclass, (e.children[0], e.children[1]));
    })?;
    rows(graph, "PlannedBuffer", |a| {
        let id = buffers[&a[0]];
        data.state.buffers.insert(
            id,
            BufferData {
                name: format!("resource{}", id.as_u32()),
                length: extent(graph, a[2], &mut extents, &chunks)?,
                element: data.types[TypeId::from(number(graph, a[1])?)].ty.clone(),
                storage: Storage::Discarded,
            },
        );
        Ok(())
    })?;
    rows(graph, "Allocation", |a| {
        if let Some(&id) = buffers.get(&a[0]) {
            data.state.buffers[id].storage = Storage::Device;
        }
        Ok(())
    })?;
    let mut status = Ok(());
    graph.function_entries_while("StorageFor", |entry| {
        status = graph.read(|state| -> Result<(), OptimizeError> {
            if let Some(&id) = buffers.get(&entry.inputs[0]) {
                state.enodes_for_eclass("Reuse", entry.output, |node| {
                    data.state.buffers[id].storage =
                        Storage::View(ExprId::from(expressions[&node.children[0]]));
                })?;
            }
            Ok(())
        });
        status.is_ok()
    })?;
    status?;
    rows(graph, "BufferSlot", |a| {
        if let Some(&id) = buffers.get(&a[3]) {
            result.slots.insert(
                (
                    OperationId::from(operations[&a[0]]),
                    graph.value_to_base::<S>(a[1]).to_string(),
                    number(graph, a[2])?,
                ),
                id,
            );
        }
        Ok(())
    })?;
    rows(graph, "OutputBacking", |a| {
        data.state.outputs[OutputId::from(number(graph, a[0])?)].buffer = buffers.get(&a[1]).copied();
        Ok(())
    })?;
    rows(graph, "CopyOutput", |a| {
        data.state.outputs[OutputId::from(number(graph, a[0])?)].copy = true;
        Ok(())
    })?;

    let mut stages = HashMap::new();
    rows(graph, "PlannedStage", |a| {
        let key = (
            OperationId::from(number(graph, a[1])?),
            graph.value_to_base::<S>(a[2]).to_string(),
        );
        let owner = EntryId::from(number(graph, a[3])?);
        if result.stages.contains_key(&key) {
            return Err(invalid("shared host dispatch requires entry specialization"));
        }
        let n = extent(graph, a[4], &mut extents, &chunks)?;
        let width = Value::Int(number(graph, a[5])?);
        let groups = Value::op(
            "min",
            [
                Value::Int(65_535),
                Value::op("max", [Value::Int(1), Value::op("ceil_div", [n, width])]),
            ],
        );
        let grid = data.state.grids.alloc(GridData {
            groups: [groups, Value::Int(1), Value::Int(1)],
        });
        let captures = if key.1 == "scalar" {
            scalar_captures(&data.ir, key.0)
        } else {
            let mut captures = BTreeSet::new();
            data.operations[key.0].kind.for_each_operand(&mut |operand| {
                if let Operand::Value(_, e) = operand {
                    captures.insert(e);
                }
            });
            captures.into_iter().collect()
        };
        // The root and its interface are complete before recipe bodies are built.
        let kernel = Function {
            name: key.1.clone(),
            kind: FunctionKind::Kernel([number(graph, a[5])?, 1, 1]),
            results: 0,
            blocks: vec![],
        }
        .insert(
            (0..captures.len()).map(|i| format!("c{i}")).collect(),
            &mut data.state.blocks,
            &mut data.state.bodies,
        );
        let body = data.state.blocks[kernel].body;
        data.state.bodies[body].instructions.extend(
            captures
                .iter()
                .enumerate()
                .map(|(i, &e)| Instruction::BindExpression(e, Value::Local(format!("c{i}")))),
        );
        let dispatch = data.state.dispatches.alloc(DispatchData {
            owner,
            kernel,
            grid,
            captures,
            reads: BTreeSet::new(),
            writes: BTreeSet::new(),
            dependencies: BTreeSet::new(),
        });
        stages.insert(a[0], dispatch);
        result.stages.insert(key, dispatch);
        Ok(())
    })?;
    rows(graph, "Access", |a| {
        if let Some(&id) = buffers.get(&a[1]) {
            let Some(&dispatch) = stages.get(&a[0]) else {
                return Err(invalid("access without a planned stage"));
            };
            match graph.value_to_base::<S>(a[2]).as_str() {
                "read" => {
                    data.state.dispatches[dispatch].reads.insert(id);
                }
                "write" => {
                    data.state.dispatches[dispatch].writes.insert(id);
                }
                _ => return Err(invalid("unknown access")),
            }
        }
        Ok(())
    })?;
    rows(graph, "DispatchDependency", |a| {
        let Some(&before) = stages.get(&a[0]) else {
            return Err(invalid("dependency without a planned predecessor"));
        };
        let Some(&after) = stages.get(&a[1]) else {
            return Err(invalid("dependency without a planned successor"));
        };
        data.state.dispatches[after].dependencies.insert(before);
        Ok(())
    })?;
    let mut lengths = BTreeMap::new();
    rows(graph, "AbiLocalLength", |a| {
        lengths.insert(
            (
                OperationId::from(operations[&a[0]]),
                graph.value_to_base::<S>(a[1]).to_string(),
                number(graph, a[2])?,
            ),
            graph.value_to_base::<i64>(a[3]) as u64,
        );
        Ok(())
    })?;
    let types = keys(graph, "TypeId")?;
    rows(graph, "LocalBuffer", |a| {
        let key = (
            OperationId::from(operations[&a[0]]),
            graph.value_to_base::<S>(a[1]).to_string(),
            number(graph, a[2])?,
        );
        let Some(&capacity) = lengths.get(&key) else {
            return Err(invalid("TODO: dynamic invocation-local allocation"));
        };
        let id = data.state.buffers.alloc(BufferData {
            name: key.1.clone(),
            length: extent(graph, a[4], &mut extents, &chunks)?,
            element: data.types[TypeId::from(types[&a[3]])].ty.clone(),
            storage: Storage::Function,
        });
        data.state.abi.local_lengths.insert(id, capacity);
        result.local_slots.insert(key, id);
        Ok(())
    })?;
    result.buffers = buffers;
    result.launches = stages;
    Ok(result)
}

// Capture external leaves, not whole expressions: branches and partial
// expressions must still execute inside the scalar invocation that owns them.
fn scalar_captures(data: &Ir, op: OperationId) -> Vec<ExprId> {
    let mut operations = vec![op];
    let mut regions = BTreeSet::new();
    let mut pending_regions = data.operations[op].kind.structured_regions();
    let mut pending = vec![];
    while let Some(r) = pending_regions.pop() {
        if !regions.insert(r) {
            continue;
        }
        let region = &data.regions[r];
        pending.extend(&region.results);
        for &child in &region.members {
            operations.push(child);
            pending_regions.extend(data.operations[child].kind.structured_regions());
        }
    }
    for op in operations {
        data.operations[op].kind.for_each_operand(&mut |operand| {
            if let Operand::Value(_, e) = operand {
                pending.push(e);
            }
        });
    }
    let mut seen = BTreeSet::new();
    let mut captures = BTreeSet::new();
    while let Some(e) = pending.pop() {
        if !seen.insert(e) {
            continue;
        }
        match &data.expressions[e].kind {
            ExprKind::Parameter(p) if !regions.contains(&data.parameters[*p].region) => {
                captures.insert(e);
            }
            ExprKind::OperationResult(op) if !regions.contains(&data.operations[*op].region) => {
                captures.insert(e);
            }
            kind => pending.extend(kind.children()),
        }
    }
    captures.into_iter().collect()
}

fn extent(
    graph: &EGraph,
    id: EggValue,
    values: &mut HashMap<EggValue, Value>,
    chunks: &HashMap<EggValue, (EggValue, EggValue)>,
) -> Result<Value, OptimizeError> {
    if let Some(value) = values.get(&id) {
        return Ok(value.clone());
    }
    let Some(&(n, width)) = chunks.get(&id) else {
        return Err(invalid("extent has no value or stored length"));
    };
    let n = extent(graph, n, values, chunks)?;
    let value = Value::op("ceil_div", [n, Value::Int(number(graph, width)?)]);
    values.insert(id, value.clone());
    Ok(value)
}

fn keys(graph: &EGraph, name: &str) -> Result<HashMap<EggValue, u32>, OptimizeError> {
    let mut keys = HashMap::new();
    let mut status = Ok(());
    graph.constructor_enodes_while(name, |e| {
        status = number(graph, e.children[0]).map(|id| {
            keys.insert(e.eclass, id);
        });
        status.is_ok()
    })?;
    status?;
    Ok(keys)
}

pub(in crate::egglog) fn number(graph: &EGraph, value: EggValue) -> Result<u32, OptimizeError> {
    u32::try_from(graph.value_to_base::<i64>(value)).map_err(|_| invalid("integer range"))
}

pub(in crate::egglog) fn rows(
    graph: &EGraph,
    name: &str,
    mut f: impl FnMut(&[EggValue]) -> Result<(), OptimizeError>,
) -> Result<(), OptimizeError> {
    let mut status = Ok(());
    graph.constructor_enodes_while(name, |e| {
        status = f(e.children);
        status.is_ok()
    })?;
    status
}

fn invalid(what: &str) -> OptimizeError {
    OptimizeError::Output(format!("relational plan readout: {what}"))
}
