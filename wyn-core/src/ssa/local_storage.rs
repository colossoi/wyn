//! Function-local bufferization. Logical array versions remain immutable until
//! edge-aware liveness proves that a web can share one physical allocation.
use super::dataflow::Dataflow;
use super::ir::{Substitutions, ValueDef};
use super::stage::Reachable;
use super::storage::is_local_value;
use super::types::{
    BlockId, ConstantValue, FuncBody, InstId, InstKind, PlaceId, PlaceInfo, PlaceOrigin, Terminator,
    ValueId, ValueRef,
};
use super::{eliminate_dead_pure_instructions, ownership, ValueUses};
use crate::builtins::catalog;
use crate::error::{CompilerError, Result};
use crate::op::OpTag;
use crate::timing;
use crate::types::{is_array_variant_composite, Type, TypeExt, TypeName};
use crate::{BindingRef, FunctionId, LookupMap, LookupSet};
use egglog_engine::{Read, Write};
use wyn_graph::DominatorTree;

#[path = "workgroup_storage.rs"]
mod workgroup;

pub(super) fn apply(program: &mut Reachable) -> Result<()> {
    for (name, body) in program
        .functions
        .iter_mut()
        .map(|f| (f.name.as_str(), &mut f.body))
        .chain(program.entry_points.iter_mut().map(|e| (e.name.as_str(), &mut e.body)))
    {
        bufferize(name, body)?;
    }
    workgroup::apply(program)?;
    for body in program
        .functions
        .iter_mut()
        .map(|f| &mut f.body)
        .chain(program.entry_points.iter_mut().map(|e| &mut e.body))
    {
        super::array_copy::lower(body);
        super::eliminate_dead_pure_instructions(body);
    }
    timing::time("ssa storage / ownership", || ownership::apply(program));
    Ok(())
}

fn fields(ty: &Type) -> Option<&[Type]> {
    match ty {
        Type::Constructed(TypeName::Tuple(_), xs) => Some(xs),
        _ => None,
    }
}
fn local_array(ty: &Type) -> bool {
    ty.is_array()
        && is_local_value(ty)
        && ty.array_variant().is_some_and(is_array_variant_composite)
        && matches!(ty.array_size(), Some(Type::Constructed(TypeName::Size(_), _)))
}
fn contains_local_array(ty: &Type) -> bool {
    local_array(ty) || fields(ty).is_some_and(|xs| xs.iter().any(contains_local_array))
}
fn op(
    body: &mut FuncBody,
    block: BlockId,
    tag: OpTag<BindingRef, FunctionId>,
    operands: Vec<ValueRef>,
    ty: Type,
) -> ValueRef {
    body.inner.append_inst(block, InstKind::Op { tag, operands }, ty).into()
}
fn project(body: &mut FuncBody, block: BlockId, value: ValueRef, index: usize, ty: Type) -> ValueRef {
    if let Some(inst) = value.as_ssa().and_then(|v| body.inner.inst_of_value(v)) {
        if let InstKind::Op {
            tag: OpTag::Tuple,
            operands,
        } = &body.inner.insts[inst].data
        {
            return operands[index];
        }
    }
    op(
        body,
        block,
        OpTag::Project { index: index as u32 },
        vec![value],
        ty,
    )
}
fn flatten_arg(body: &mut FuncBody, block: BlockId, value: ValueRef, ty: &Type, out: &mut Vec<ValueRef>) {
    if let Some(xs) = fields(ty).filter(|_| contains_local_array(ty)) {
        for (i, ty) in xs.iter().enumerate() {
            let part = project(body, block, value, i, ty.clone());
            flatten_arg(body, block, part, ty, out);
        }
    } else {
        out.push(value);
    }
}
fn flatten_param(body: &mut FuncBody, block: BlockId, ty: &Type) -> ValueRef {
    if let Some(xs) = fields(ty).filter(|_| contains_local_array(ty)) {
        let values = xs.iter().map(|ty| flatten_param(body, block, ty)).collect();
        op(body, block, OpTag::Tuple, values, ty.clone())
    } else {
        body.inner.add_block_param(block, ty.clone()).into()
    }
}
/// Expose array components without changing function signatures. Reconstruction
/// instructions are retained only if an aggregate actually escapes.
fn split_parameters(body: &mut FuncBody) {
    let blocks: Vec<_> = body.inner.blocks.keys().collect();
    let mut shapes = LookupMap::default();
    let mut substitutions = Substitutions::default();
    for &block in &blocks {
        let old = body.inner.blocks[block].params.clone();
        if !old.iter().any(|v| {
            fields(body.get_value_type(*v)).is_some() && contains_local_array(body.get_value_type(*v))
        }) {
            continue;
        }
        let types: Vec<_> = old.iter().map(|v| body.get_value_type(*v).clone()).collect();
        shapes.insert(block, types.clone());
        body.inner.blocks[block].params.clear();
        let tail = std::mem::take(&mut body.inner.blocks[block].insts);
        for (old, ty) in old.into_iter().zip(types) {
            let value = flatten_param(body, block, &ty);
            substitutions.insert(old, value);
        }
        body.inner.blocks[block].insts.extend(tail);
    }
    // Rewrite uses before extracting edge components; old params may be inputs
    // to a backedge or to another block's reconstructed tuple.
    substitutions.finish(&mut body.inner);
    for block in blocks {
        let mut term = body.inner.blocks[block].term.clone();
        for (target, args) in term.edges_mut() {
            if let Some(types) = shapes.get(&target) {
                let mut flat = Vec::new();
                for (&arg, ty) in args.iter().zip(types) {
                    flatten_arg(body, block, arg, ty, &mut flat);
                }
                *args = flat;
            }
        }
        body.inner.blocks[block].term = term;
    }
    // Projection chains may cross block storage order. Resolve to a fixed point.
    loop {
        let mut substitutions = Substitutions::default();
        let mut removed = LookupSet::default();
        for (id, node) in &body.inner.insts {
            if let (
                Some(result),
                InstKind::Op {
                    tag: OpTag::Project { index },
                    operands,
                },
            ) = (node.result, &node.data)
            {
                if let Some(base) = operands[0].as_ssa().and_then(|v| body.inner.inst_of_value(v)) {
                    if let InstKind::Op {
                        tag: OpTag::Tuple,
                        operands,
                    } = &body.inner.insts[base].data
                    {
                        substitutions.insert(result, operands[*index as usize]);
                        removed.insert(id);
                    }
                }
            }
        }
        if removed.is_empty() {
            break;
        }
        for &id in &removed {
            body.inner.insts.remove(id);
        }
        for block in body.inner.blocks.values_mut() {
            block.insts.retain(|id| !removed.contains(id));
        }
        substitutions.finish(&mut body.inner);
    }
    eliminate_dead_pure_instructions(body);
}

fn update(data: &InstKind) -> Option<&[ValueRef]> {
    let known = catalog().known();
    match data {
        InstKind::Op {
            tag: OpTag::Intrinsic { id, .. },
            operands,
        } if *id == known.array_with || *id == known.array_with_in_place => Some(operands),
        _ => None,
    }
}

fn bufferize(name: &str, body: &mut FuncBody) -> Result<()> {
    if !body.inner.values.values().any(|value| contains_local_array(&value.ty)) {
        return Ok(());
    }
    let label = timing::enabled().then(|| format!("ssa storage body / {name}"));
    let _body = label.as_deref().and_then(timing::span);
    timing::time("ssa storage / split parameters", || split_parameters(body));
    timing::time("ssa storage / remove loads", || remove_unused_loads(body));
    let analysis = timing::time("ssa storage / analysis", || analyze(body))?;
    let (places, reused) = timing::time("ssa storage / allocate", || allocate(body, &analysis))?;
    timing::time("ssa storage / verify", || verify(body, &analysis, &places))?;
    timing::time("ssa storage / rewrite", || {
        rewrite(body, &analysis.roots, &places, &reused)
    })
}

fn remove_unused_loads(body: &mut FuncBody) {
    let uses = ValueUses::analyze(&body.inner);
    let unused: LookupSet<_> = body
        .inner
        .insts
        .iter()
        .filter_map(|(id, n)| {
            if let (Some(v), InstKind::Load { place }) = (n.result, &n.data) {
                if uses.count(v) == 0
                    && local_array(&body.places[*place].elem_ty)
                    && body
                        .inner
                        .insts
                        .values()
                        .any(|n| matches!(n.data, InstKind::Alloca { result, .. } if result == *place))
                {
                    return Some(id);
                }
            }
            None
        })
        .collect();
    for &id in &unused {
        if let Some(n) = body.inner.insts.remove(id) {
            if let Some(v) = n.result {
                body.inner.values.remove(v);
            }
        }
    }
    for b in body.inner.blocks.values_mut() {
        b.insts.retain(|id| !unused.contains(id));
    }
}

struct Analysis {
    roots: LookupMap<ValueId, ValueId>,
    origins: LookupMap<ValueId, Vec<ValueId>>,
    candidates: Vec<ValueId>,
    existing: LookupMap<ValueId, PlaceId>,
}

const ANALYSIS_SCHEDULE: &str = "(run-schedule (saturate storage-webs) (saturate flow-live) (saturate storage-select) (saturate storage-existing) (saturate storage-allocation) (saturate storage-initialization) (saturate flow-initialization))";

fn storage_places(body: &FuncBody, arrays: &LookupMap<ValueId, usize>) -> LookupSet<PlaceId> {
    let mut allocations = LookupSet::default();
    let mut loads = Vec::new();
    for node in body.inner.insts.values() {
        match node.data {
            InstKind::Alloca { result } => {
                allocations.insert(result);
            }
            InstKind::Load { place } if node.result.is_some_and(|v| arrays.contains_key(&v)) => {
                loads.push(place);
            }
            _ => {}
        }
    }
    // Only direct allocation loads can seed Existing; retain every derived alias of those places.
    loads.retain(|place| allocations.contains(place));
    let roots = super::storage::MemoryRoots::analyze(body);
    body.places
        .keys()
        .filter(|p| {
            roots.places.get(p).is_some_and(|xs| {
                xs.iter().any(
                    |r| matches!(r, super::storage::MemoryRoot::Allocation(base) if loads.contains(base)),
                )
            })
        })
        .collect()
}

fn storage_events(
    body: &FuncBody,
    arrays: &LookupMap<ValueId, usize>,
    places: &LookupSet<PlaceId>,
) -> LookupMap<BlockId, Vec<Vec<InstId>>> {
    body.inner
        .blocks
        .iter()
        .map(|(block, data)| {
            let relevant = data.insts.iter().copied().filter(|&inst| {
                let node = &body.inner.insts[inst];
                node.result.is_some_and(|v| arrays.contains_key(&v))
                    || matches!(node.data, InstKind::Alloca { result } if places.contains(&result))
                    || node.data.place_uses().iter().any(|p| places.contains(p))
                    || node
                        .data
                        .value_uses()
                        .iter()
                        .any(|v| v.as_ssa().is_some_and(|v| arrays.contains_key(&v)))
            });
            let mut events: Vec<Vec<InstId>> = Vec::new();
            let mut previous_boundary = true;
            for inst in relevant {
                let node = &body.inner.insts[inst];
                let boundary = node.result.is_some_and(|v| arrays.contains_key(&v))
                    || matches!(node.data, InstKind::Alloca { .. });
                // Accesses between definitions/allocations share one liveness and reachability state.
                match events.last_mut() {
                    Some(event) if !boundary && !previous_boundary => event.push(inst),
                    _ => events.push(vec![inst]),
                }
                previous_boundary = boundary;
            }
            (block, events)
        })
        .collect()
}

fn analyze(body: &FuncBody) -> Result<Analysis> {
    let inputs = timing::span("ssa storage analysis / collect inputs");
    let values: Vec<_> =
        body.inner.values.iter().filter_map(|(v, info)| local_array(&info.ty).then_some(v)).collect();
    let indices: LookupMap<_, _> = values.iter().enumerate().map(|(i, v)| (*v, i)).collect();
    // Contract steps with no storage facts, preserving event order and every CFG edge.
    let relevant_places = storage_places(body, &indices);
    let events = storage_events(body, &indices, &relevant_places);
    let places: Vec<_> = body.places.keys().collect();
    let place_ids: LookupMap<_, _> = places.iter().enumerate().map(|(i, p)| (*p, i as i64)).collect();
    if timing::enabled() {
        eprintln!(
            "ssa storage facts: {} instructions, {} events, {} arrays, {} blocks",
            body.num_insts(),
            events.values().map(Vec::len).sum::<usize>(),
            values.len(),
            body.num_blocks()
        );
    }
    drop(inputs);
    let schema = timing::span("ssa storage analysis / schema");
    let map_error = |e: egglog_engine::Error| invalid(&e.to_string());
    let mut flow = Dataflow::new(body, &events).map_err(map_error)?;
    let Dataflow {
        graph,
        blocks: points,
        instructions,
        edges: control_edges,
        ..
    } = &mut flow;

    graph.parse_and_run_program(None, include_str!("local_storage.egg")).map_err(map_error)?;
    drop(schema);
    let import = timing::span("ssa storage analysis / import facts");
    let mut origin_values = Vec::new();
    graph.update(|mut sink| {
        sink.add("StorageEntry", (points[&body.inner.entry].0, values.len() as i64))?;
        for (i, &v) in values.iter().enumerate() {
            let id = i as i64;
            sink.add("Array", (id,))?;
            match body.inner.values[v].def {
                ValueDef::Param { block, .. } => { sink.add("FlowParameter", (points[&block].0, id))?; }
                ValueDef::FunctionParam { .. } => {
                    sink.add("Escape", (id,))?;
                }
                ValueDef::Inst { inst } => {
                    let data = &body.inner.insts[inst].data;
                    let forwarded = match data {
                        InstKind::Op { tag: OpTag::Materialize, operands } => Some(operands.as_slice()),
                        _ => update(data),
                    };
                    if let Some(xs) = forwarded {
                        if let Some(source) = xs[0].as_ssa().and_then(|v| indices.get(&v)) {
                            sink.add("Link", (id, *source as i64))?;
                        } else { sink.add("Escape", (id,))?; }
                        if update(data).is_some() { sink.add("Updated", (id,))?; }
                    } else {
                        origin_values.push(v);
                        sink.add("Origin", (id,))?;
                    }
                }
            }
        }
        for (b, block) in &body.inner.blocks {
            for (index, (target, args)) in block.term.edges().into_iter().enumerate() {
                let edge = control_edges[&(b, index)];
                for (&param, arg) in body.inner.blocks[target].params.iter().zip(args) {
                    if let Some(&param) = indices.get(&param) {
                        if let Some(&arg) = arg.as_ssa().and_then(|v| indices.get(&v)) {
                            sink.add("FlowArgument", (edge, param as i64, arg as i64))?;
                            sink.add("Link", (param as i64, arg as i64))?;
                        } else { sink.add("Escape", (param as i64,))?; }
                    }
                }
            }
            if let Terminator::Return(Some(ValueRef::Ssa(v))) = block.term {
                if let Some(&v) = indices.get(&v) { sink.add("Escape", (v as i64,))?; }
            }
            for event in &events[&b] {
                for &inst in event {
                    let (before, after) = instructions[&inst];
                    let node = &body.inner.insts[inst];
                    if let Some(&v) = node.result.and_then(|v| indices.get(&v)) {
                        sink.set("FlowKilled", (before, v as i64), true)?;
                        sink.add("StorageDefinition", (before, v as i64))?;
                        if !matches!(node.data, InstKind::Op { tag: OpTag::Materialize, .. }) {
                            sink.add("Write", (after, v as i64))?;
                        }
                        if let InstKind::Load { place } = node.data {
                            sink.add("OriginLoad", (v as i64, place_ids[&place], after))?;
                        }
                    }
                    for (i, value) in node.data.value_uses().into_iter().enumerate() {
                        let Some(&v) = value.as_ssa().and_then(|v| indices.get(&v)) else { continue; };
                        sink.add("FlowUse", (before, v as i64))?;
                        if i == 0 && matches!(&node.data, InstKind::Op { tag: OpTag::Index | OpTag::DynamicExtract, .. }) {
                            sink.add("Indexed", (v as i64,))?;
                        }
                        let modeled = matches!(&node.data, InstKind::Op { tag: OpTag::Tuple, .. })
                            || matches!(&node.data,
                            InstKind::Op { tag: OpTag::Index | OpTag::DynamicExtract | OpTag::Materialize, .. } if i == 0)
                            || update(&node.data).is_some_and(|_| i == 0)
                            || matches!(&node.data, InstKind::Op { tag: OpTag::Intrinsic { id, .. }, .. }
                                if *id == catalog().known().length || *id == catalog().known().storage_len);
                        if !modeled { sink.add("Escape", (v as i64,))?; }
                    }
                    for place in node.data.place_uses().into_iter().filter(|p| relevant_places.contains(p)) {
                        sink.add("PlaceUse", (before, place_ids[&place]))?;
                    }
                    match node.data {
                        InstKind::Alloca { result, .. } => {
                            sink.add("Allocation", (place_ids[&result],))?;
                            sink.set("AllocationBoundary", (before, place_ids[&result]), true)?;
                        }
                        InstKind::PlaceIndex { place, result, .. } => { sink.add("PlaceIndex", (place_ids[&place], place_ids[&result]))?; }
                        _ => {}
                    }
                }
            }
        }
        Ok::<_, egglog_engine::Error>(())
    }).map_err(map_error)?;
    drop(import);
    timing::time("ssa storage analysis / rules", || {
        graph.parse_and_run_program(None, ANALYSIS_SCHEDULE)
    })
    .map_err(map_error)?;
    if std::env::var_os("WYN_EGGLOG_PROFILE").is_some() {
        for output in graph.parse_and_run_program(None, "(print-stats)").map_err(map_error)? {
            eprintln!("{output}");
        }
    }
    let _readout = timing::span("ssa storage analysis / read results");
    let mut roots = LookupMap::default();
    let mut candidates = Vec::new();
    let mut existing = LookupMap::default();
    for (id, &v) in values.iter().enumerate() {
        let Some(root) = graph.read(|r| r.lookup("Web", (id as i64,))).map_err(map_error)? else {
            return Err(invalid("array has no planned version web"));
        };
        roots.insert(v, values[graph.value_to_base::<i64>(root) as usize]);

        if graph.read(|r| r.contains("Reuse", (id as i64,))).map_err(map_error)? {
            if graph.read(|r| r.contains("StorageUninitialized", id as i64)).map_err(map_error)? {
                return Err(invalid("promoted array is read before initialization"));
            }
            candidates.push(v);
        }
        if let Some(p) = graph.read(|r| r.lookup("Existing", (id as i64,))).map_err(map_error)? {
            existing.insert(v, places[graph.value_to_base::<i64>(p) as usize]);
        }
    }
    let mut origins = LookupMap::<_, Vec<_>>::default();
    for v in origin_values {
        origins.entry(roots[&v]).or_default().push(v);
    }
    Ok(Analysis {
        roots,
        origins,
        candidates,
        existing,
    })
}

fn allocate(
    body: &mut FuncBody,
    analysis: &Analysis,
) -> Result<(LookupMap<ValueId, PlaceId>, LookupSet<ValueId>)> {
    let Analysis {
        origins,
        candidates,
        existing,
        ..
    } = analysis;
    let mut places = LookupMap::default();
    let mut reused = LookupSet::default();
    for &root in candidates {
        let Some(sources) = origins.get(&root).filter(|xs| !xs.is_empty()) else {
            continue;
        };
        let existing = existing.get(&root).copied();
        let place = existing.unwrap_or_else(|| {
            let place = body.places.insert(PlaceInfo {
                elem_ty: body.get_value_type(root).clone(),
                origin: PlaceOrigin::Instruction,
            });
            let entry = body.inner.entry;
            let tail = std::mem::take(&mut body.inner.blocks[entry].insts);
            body.inner.append_void_inst(entry, InstKind::Alloca { result: place });
            body.inner.blocks[entry].insts.extend(tail);
            place
        });
        if existing.is_some() {
            reused.insert(sources[0]);
        }
        places.insert(root, place);
    }
    Ok((places, reused))
}

fn rewrite(
    body: &mut FuncBody,
    roots: &LookupMap<ValueId, ValueId>,
    places: &LookupMap<ValueId, PlaceId>,
    reused: &LookupSet<ValueId>,
) -> Result<()> {
    if places.is_empty() {
        return Ok(());
    }
    let blocks: Vec<_> = body.inner.blocks.keys().collect();
    let promoted: LookupMap<_, _> =
        roots.keys().filter_map(|v| places.get(&roots[v]).map(|p| (*v, *p))).collect();
    let mut removed = LookupSet::default();
    let mut substitutions = Substitutions::default();
    for &block in &blocks {
        let old = std::mem::take(&mut body.inner.blocks[block].insts);
        for inst in old {
            let node = body.inner.insts[inst].clone();
            if let Some(xs) = update(&node.data) {
                if let Some(&place) = xs[0].as_ssa().and_then(|v| promoted.get(&v)) {
                    let Some(elem_ty) = body.places[place].elem_ty.elem_type().cloned() else {
                        return Err(invalid("array place has no element type"));
                    };
                    let element = body.places.insert(PlaceInfo {
                        elem_ty,
                        origin: PlaceOrigin::Instruction,
                    });
                    body.inner.append_void_inst(
                        block,
                        InstKind::PlaceIndex {
                            place,
                            index: xs[1],
                            result: element,
                        },
                    );
                    body.inner.append_void_inst(
                        block,
                        InstKind::Store {
                            place: element,
                            value: xs[2],
                        },
                    );
                    removed.insert(inst);
                    continue;
                }
            }
            if let InstKind::Op {
                tag: OpTag::Index | OpTag::DynamicExtract,
                operands,
            } = &node.data
            {
                if let Some(&place) = operands[0].as_ssa().and_then(|v| promoted.get(&v)) {
                    let Some(elem_ty) = body.places[place].elem_ty.elem_type().cloned() else {
                        return Err(invalid("array place has no element type"));
                    };
                    let element = body.places.insert(PlaceInfo {
                        elem_ty,
                        origin: PlaceOrigin::Instruction,
                    });
                    body.inner.append_void_inst(
                        block,
                        InstKind::PlaceIndex {
                            place,
                            index: operands[1],
                            result: element,
                        },
                    );
                    body.inner.insts[inst].data = InstKind::Load { place: element };
                    body.inner.blocks[block].insts.push(inst);
                    continue;
                }
            }
            if let Some(v) = node.result.filter(|v| promoted.contains_key(v)) {
                if matches!(
                    node.data,
                    InstKind::Op {
                        tag: OpTag::Materialize,
                        ..
                    }
                ) || reused.contains(&v)
                {
                    removed.insert(inst);
                    continue;
                }
                body.inner.blocks[block].insts.push(inst);
                body.inner.append_void_inst(
                    block,
                    InstKind::Store {
                        place: promoted[&v],
                        value: v.into(),
                    },
                );
                continue;
            }
            if let InstKind::Op {
                tag: OpTag::Intrinsic { id, .. },
                operands,
            } = &node.data
            {
                if (*id == catalog().known().length || *id == catalog().known().storage_len)
                    && operands[0].as_ssa().is_some_and(|v| promoted.contains_key(&v))
                {
                    // Fixed-array metadata never needs a snapshot of contents.
                    let Some(v) = operands[0].as_ssa() else {
                        return Err(invalid("array metadata has no source"));
                    };
                    if let Some(Type::Constructed(TypeName::Size(n), _)) =
                        body.get_value_type(v).array_size()
                    {
                        let Some(result) = node.result else {
                            return Err(invalid("array metadata has no result"));
                        };
                        substitutions.insert(result, ValueRef::Const(ConstantValue::I32(*n as i32)));
                        removed.insert(inst);
                        continue;
                    }
                }
            }
            if let InstKind::Op {
                tag: OpTag::Tuple,
                operands,
            } = &node.data
            {
                let mut snapshots = operands.clone();
                for operand in &mut snapshots {
                    if let Some(&place) = operand.as_ssa().and_then(|v| promoted.get(&v)) {
                        let ty = body.places[place].elem_ty.clone();
                        *operand = body.inner.append_inst(block, InstKind::Load { place }, ty).into();
                    }
                }
                body.inner.insts[inst].data = InstKind::Op {
                    tag: OpTag::Tuple,
                    operands: snapshots,
                };
            }
            body.inner.blocks[block].insts.push(inst);
        }
    }
    let masks: LookupMap<_, Vec<_>> = body
        .inner
        .blocks
        .iter()
        .map(|(b, data)| (b, data.params.iter().map(|v| !promoted.contains_key(v)).collect()))
        .collect();
    for (b, data) in &mut body.inner.blocks {
        let mut index = 0;
        data.params.retain(|_| {
            let keep = masks[&b][index];
            index += 1;
            keep
        });
        for (index, &v) in data.params.iter().enumerate() {
            body.inner.values[v].def = ValueDef::Param { block: b, index };
        }
        for (target, args) in data.term.edges_mut() {
            let mut index = 0;
            args.retain(|_| {
                let keep = masks[&target][index];
                index += 1;
                keep
            });
        }
    }
    for id in removed {
        body.inner.insts.remove(id);
    }
    substitutions.finish(&mut body.inner);
    eliminate_dead_pure_instructions(body);
    // Retired versions have no remaining uses. Drop their metadata after DCE
    // has discarded now-unneeded aggregate reconstructions.
    let used = ValueUses::analyze(&body.inner);
    let params: LookupSet<_> = body.inner.blocks.values().flat_map(|b| b.params.iter().copied()).collect();
    let instructions = &body.inner.insts;
    body.inner.values.retain(|v, info| match info.def {
        ValueDef::Param { .. } => params.contains(&v) || used.count(v) != 0,
        ValueDef::Inst { inst } => instructions.contains_key(inst),
        ValueDef::FunctionParam { .. } => true,
    });
    Ok(())
}

fn invalid(message: &str) -> CompilerError {
    CompilerError::Internal(format!("local storage: {message}"))
}

#[cfg(test)]
#[path = "local_storage_tests.rs"]
mod tests;

fn verify(body: &FuncBody, analysis: &Analysis, places: &LookupMap<ValueId, PlaceId>) -> Result<()> {
    if places.is_empty() {
        return Ok(());
    }
    let dominators = DominatorTree::build(body.inner.entry, |b, out| {
        out.extend(body.inner.blocks[b].term.successors());
    });
    let blocks = dominators.preorder();
    let mut allocations = LookupMap::default();
    for &block in blocks {
        for (position, &inst) in body.inner.blocks[block].insts.iter().enumerate() {
            if let InstKind::Alloca { result, .. } = body.inner.insts[inst].data {
                allocations.insert(result, (block, position));
            }
        }
    }
    let origins: LookupMap<_, _> = analysis
        .origins
        .iter()
        .filter_map(|(root, values)| places.get(root).map(|p| (values, *p)))
        .flat_map(|(values, p)| values.iter().map(move |v| (*v, p)))
        .collect();
    let promoted: LookupMap<_, _> =
        analysis.roots.keys().filter_map(|v| places.get(&analysis.roots[v]).map(|p| (*v, *p))).collect();
    for &block in blocks {
        for (position, &inst) in body.inner.blocks[block].insts.iter().enumerate() {
            let node = &body.inner.insts[inst];
            let origin = node.result.and_then(|v| origins.get(&v)).copied();
            let reads: Vec<_> = node
                .data
                .value_uses()
                .into_iter()
                .filter_map(|v| v.as_ssa().and_then(|v| promoted.get(&v).copied()))
                .collect();
            for place in reads.iter().copied().chain(origin) {
                let Some(&(allocation, index)) = allocations.get(&place) else {
                    return Err(invalid("promoted storage has no allocation"));
                };
                if !dominators.dominates(allocation, block) || (allocation == block && index >= position) {
                    return Err(invalid("storage allocation does not dominate its access"));
                }
            }
        }
    }
    Ok(())
}
