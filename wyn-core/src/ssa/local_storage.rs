//! Function-local bufferization. Logical array versions remain immutable until
//! edge-aware liveness proves that a web can share one physical allocation.
use super::ir::{Substitutions, ValueDef};
use super::stage::Reachable;
use super::storage::is_local_value;
use super::types::{
    BlockId, ConstantValue, FuncBody, InstKind, PlaceId, PlaceInfo, PlaceOrigin, Terminator, ValueId,
    ValueRef,
};
use super::{eliminate_dead_pure_instructions, ownership, ValueUses};
use crate::builtins::catalog;
use crate::error::{CompilerError, Result};
use crate::op::OpTag;
use crate::types::{is_array_variant_composite, Type, TypeExt, TypeName};
use crate::{BindingRef, FunctionId, LookupMap, LookupSet};
use egglog_engine::{EGraph, Read, Write};

#[path = "local_storage_validation.rs"]
mod validation;

pub(super) fn apply(program: &mut Reachable) -> Result<()> {
    for body in program
        .functions
        .iter_mut()
        .map(|f| &mut f.body)
        .chain(program.entry_points.iter_mut().map(|e| &mut e.body))
        .chain(program.constants.iter_mut().map(|c| &mut c.body))
    {
        bufferize(body)?;
    }
    ownership::apply(program);
    Ok(())
}

fn fields(ty: &Type) -> Option<&[Type]> {
    match ty {
        Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), xs) => Some(xs),
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
            tag: OpTag::Tuple(_),
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
        op(body, block, OpTag::Tuple(xs.len()), values, ty.clone())
    } else {
        body.inner.add_block_param(block, ty.clone()).into()
    }
}
fn edges(term: &Terminator) -> Vec<(BlockId, Vec<ValueRef>)> {
    match term {
        Terminator::Branch { target, args } => vec![(*target, args.clone())],
        Terminator::CondBranch {
            then_target,
            then_args,
            else_target,
            else_args,
            ..
        } => vec![
            (*then_target, then_args.clone()),
            (*else_target, else_args.clone()),
        ],
        _ => vec![],
    }
}
fn map_edges(term: &mut Terminator, mut f: impl FnMut(BlockId, &mut Vec<ValueRef>)) {
    match term {
        Terminator::Branch { target, args } => f(*target, args),
        Terminator::CondBranch {
            then_target,
            then_args,
            else_target,
            else_args,
            ..
        } => {
            f(*then_target, then_args);
            f(*else_target, else_args);
        }
        _ => {}
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
        map_edges(&mut term, |target, args| {
            if let Some(types) = shapes.get(&target) {
                let mut flat = Vec::new();
                for (&arg, ty) in args.iter().zip(types) {
                    flatten_arg(body, block, arg, ty, &mut flat);
                }
                *args = flat;
            }
        });
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
                        tag: OpTag::Tuple(_),
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

struct Webs {
    indices: LookupMap<ValueId, usize>,
    roots: LookupMap<ValueId, ValueId>,
}
impl Webs {
    fn find(&self, value: ValueId) -> ValueId {
        self.roots[&value]
    }
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

fn bufferize(body: &mut FuncBody) -> Result<()> {
    if !body.inner.insts.values().any(|n| {
        update(&n.data)
            .is_some_and(|xs| xs[0].as_ssa().is_some_and(|v| local_array(body.get_value_type(v))))
    }) {
        return Ok(());
    }
    split_parameters(body);
    remove_unused_loads(body);
    let analysis = analyze(body)?;
    let (places, reused) = allocate(body, &analysis)?;
    validation::verify(body, &analysis, &places)?;
    rewrite(body, &analysis.webs, &places, &reused)
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
    webs: Webs,
    origins: LookupMap<ValueId, Vec<ValueId>>,
    candidates: Vec<ValueId>,
    existing: LookupMap<ValueId, PlaceId>,
}

fn analyze(body: &FuncBody) -> Result<Analysis> {
    let values: Vec<_> =
        body.inner.values.iter().filter_map(|(v, info)| local_array(&info.ty).then_some(v)).collect();
    let indices: LookupMap<_, _> = values.iter().enumerate().map(|(i, v)| (*v, i)).collect();
    let places: Vec<_> = body.places.keys().collect();
    let place_ids: LookupMap<_, _> = places.iter().enumerate().map(|(i, p)| (*p, i as i64)).collect();
    let mut next = 0i64;
    let points: LookupMap<_, _> = body
        .inner
        .blocks
        .iter()
        .map(|(b, data)| {
            let start = next;
            next += data.insts.len() as i64 + 1;
            (b, (start, next - 1))
        })
        .collect();
    let mut graph = EGraph::default();
    let map_error = |e: egglog_engine::Error| invalid(&e.to_string());
    graph.parse_and_run_program(None, include_str!("local_storage.egg")).map_err(map_error)?;
    let mut origin_values = Vec::new();
    graph.update(|mut sink| {
        for (i, &v) in values.iter().enumerate() {
            let id = i as i64;
            sink.add("Array", (id,))?;
            match body.inner.values[v].def {
                ValueDef::Param { block, .. } => { sink.add("Parameter", (points[&block].0, id))?; }
                ValueDef::FunctionParam { .. } => {
                    sink.add("NonParameter", (id,))?;
                    sink.add("Escape", (id,))?;
                }
                ValueDef::Inst { inst } => {
                    sink.add("NonParameter", (id,))?;
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
        let mut edge = 0i64;
        for (b, block) in &body.inner.blocks {
            let (entry, exit) = points[&b];
            for (target, args) in edges(&block.term) {
                sink.add("Edge", (edge, exit, points[&target].0))?;
                sink.add("Flow", (exit, points[&target].0))?;
                for (&param, arg) in body.inner.blocks[target].params.iter().zip(args) {
                    if let Some(&param) = indices.get(&param) {
                        if let Some(&arg) = arg.as_ssa().and_then(|v| indices.get(&v)) {
                            sink.add("Argument", (edge, param as i64, arg as i64))?;
                            sink.add("Link", (param as i64, arg as i64))?;
                        } else { sink.add("Escape", (param as i64,))?; }
                    }
                }
                edge += 1;
            }
            if let Terminator::Return(Some(ValueRef::Ssa(v))) = block.term {
                if let Some(&v) = indices.get(&v) { sink.add("Escape", (v as i64,))?; }
            }
            for (offset, &inst) in block.insts.iter().enumerate() {
                let before = entry + offset as i64;
                let after = before + 1;
                let node = &body.inner.insts[inst];
                sink.add("Flow", (before, after))?;
                if let Some(&v) = node.result.and_then(|v| indices.get(&v)) {
                    sink.add("Define", (before, after, v as i64))?;
                    if !matches!(node.data, InstKind::Op { tag: OpTag::Materialize, .. }) {
                        sink.add("Write", (after, v as i64))?;
                    }
                    if let InstKind::Load { place } = node.data {
                        sink.add("OriginLoad", (v as i64, place_ids[&place], after))?;
                    }
                } else { sink.add("Step", (before, after))?; }
                for (i, value) in node.data.value_uses().into_iter().enumerate() {
                    let Some(&v) = value.as_ssa().and_then(|v| indices.get(&v)) else { continue; };
                    sink.add("Use", (before, v as i64))?;
                    let modeled = matches!(&node.data,
                        InstKind::Op { tag: OpTag::Index | OpTag::DynamicExtract | OpTag::Materialize, .. } if i == 0)
                        || update(&node.data).is_some_and(|_| i == 0)
                        || matches!(&node.data, InstKind::Op { tag: OpTag::Intrinsic { id, .. }, .. }
                            if *id == catalog().known().length || *id == catalog().known().storage_len);
                    if !modeled { sink.add("Escape", (v as i64,))?; }
                }
                for place in node.data.place_uses() { sink.add("PlaceUse", (before, place_ids[&place]))?; }
                match node.data {
                    InstKind::Alloca { result, .. } => { sink.add("Allocation", (place_ids[&result],))?; }
                    InstKind::PlaceIndex { place, result, .. } => { sink.add("PlaceIndex", (place_ids[&place], place_ids[&result]))?; }
                    _ => {}
                }
            }
        }
        Ok::<_, egglog_engine::Error>(())
    }).map_err(map_error)?;
    graph.parse_and_run_program(None, "(run-schedule (saturate storage-webs) (saturate storage-liveness) (saturate storage-select) (saturate storage-allocation))").map_err(map_error)?;
    let mut roots = LookupMap::default();
    let mut candidates = Vec::new();
    let mut existing = LookupMap::default();
    for (id, &v) in values.iter().enumerate() {
        let Some(root) = graph.read(|r| r.lookup("Web", (id as i64,))).map_err(map_error)? else {
            return Err(invalid("array has no planned version web"));
        };
        roots.insert(v, values[graph.value_to_base::<i64>(root) as usize]);
        if graph.read(|r| r.contains("Reuse", (id as i64,))).map_err(map_error)? {
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
        webs: Webs { indices, roots },
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
        let ty = body.get_value_type(root).clone();
        let existing = existing.get(&root).copied();
        let place = existing.unwrap_or_else(|| {
            let place = body.places.insert(PlaceInfo {
                elem_ty: ty.clone(),
                origin: PlaceOrigin::Instruction,
            });
            let entry = body.inner.entry;
            let tail = std::mem::take(&mut body.inner.blocks[entry].insts);
            body.inner.append_void_inst(
                entry,
                InstKind::Alloca {
                    elem_ty: ty,
                    result: place,
                },
            );
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
    webs: &Webs,
    places: &LookupMap<ValueId, PlaceId>,
    reused: &LookupSet<ValueId>,
) -> Result<()> {
    if places.is_empty() {
        return Ok(());
    }
    let blocks: Vec<_> = body.inner.blocks.keys().collect();
    let promoted: LookupMap<_, _> =
        webs.indices.keys().filter_map(|v| places.get(&webs.find(*v)).map(|p| (*v, *p))).collect();
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
        map_edges(&mut data.term, |target, args| {
            let mut index = 0;
            args.retain(|_| {
                let keep = masks[&target][index];
                index += 1;
                keep
            });
        });
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
