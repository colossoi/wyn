//! Source-oriented names for published shaders and GPU resources.
use super::abi::{error, Abi};
use super::blocks::{BlockData, FunctionKind, Storage};
use super::data::{BlockId, BufferId, EntryId, Ir, OutputData};
use super::{OptimizeError, Program, Scheduled};
use crate::host::ResultKind;
use crate::interface::EntryKind;
use crate::types::{strip_existentials, Type, TypeName};
use std::collections::{BTreeMap, BTreeSet};
use wyn_base::IdArena;

fn component(source: &str) -> String {
    let mut name = String::new();
    for c in source.chars() {
        if c.is_ascii_alphanumeric() {
            name.push(c);
        } else if !name.is_empty() && !name.ends_with('_') {
            name.push('_');
        }
    }
    let mut name = name.trim_end_matches('_').to_owned();
    if name.is_empty() {
        name.push_str("entry");
    }
    if name.starts_with(|c: char| c.is_ascii_digit()) {
        name.insert_str(0, "entry_");
    }
    name
}

fn unique(base: String, used: &mut BTreeSet<String>) -> String {
    let mut name = base.clone();
    let mut suffix = 2usize;
    while !used.insert(name.clone()) {
        name = format!("{base}_{suffix}");
        suffix += 1;
    }
    name
}

fn owner(ir: &Ir, entry: EntryId) -> Result<String, OptimizeError> {
    let declaration = &ir.entries[entry].declaration;
    if let Some(group) = &declaration.graphics_group {
        let Some(symbol) = ir.symbols.values().find(|s| s.source == group.root) else {
            return Err(error("graphics output has no source owner"));
        };
        Ok(component(&symbol.name))
    } else {
        Ok(component(&declaration.name))
    }
}

pub(super) fn result_field(ir: &Ir, output: &OutputData) -> Result<(String, ResultKind), OptimizeError> {
    let entry = &ir.entries[output.entry];
    let region = &ir.regions[ir.definitions[entry.definition].body];
    let ty = region.results.first().map(|e| strip_existentials(&ir.types[ir.expressions[*e].ty].ty));
    Ok(match ty {
        Some(Type::Constructed(TypeName::Record(names), _)) => {
            let Some(name) = names.0.get(output.index) else {
                return Err(error("source result has no record field"));
            };
            (name.clone(), ResultKind::RecordField)
        }
        Some(Type::Constructed(TypeName::Tuple(_), _)) => {
            (format!("result_{}", output.index), ResultKind::TupleField)
        }
        _ => (entry.declaration.name.clone(), ResultKind::Value),
    })
}

pub(super) fn entry_points(
    abi: &Abi,
    ir: &Ir,
    blocks: &IdArena<BlockId, BlockData>,
) -> Result<BTreeMap<BlockId, String>, OptimizeError> {
    let mut names = BTreeMap::new();
    let mut used = BTreeSet::new();
    for (&entry, roots) in &abi.entry_roots {
        let source = owner(ir, entry)?;
        for &root in roots {
            let Some(function) = &blocks[root].interface else {
                return Err(error("shader root has no function interface"));
            };
            let phase = match function.kind {
                FunctionKind::Kernel(_) => component(match function.name.as_str() {
                    "elements" | "scalar" => "compute",
                    "chunks" => "partials",
                    other => other,
                }),
                FunctionKind::Entry(_) => match ir.entries[entry].declaration.entry_kind {
                    EntryKind::Vertex => "vertex",
                    EntryKind::Fragment => "fragment",
                    EntryKind::Compute if roots.len() == 1 => "compute",
                    EntryKind::Compute => "finish",
                    EntryKind::Root => return Err(error("unextracted shader root")),
                }
                .into(),
                FunctionKind::Host | FunctionKind::Device => {
                    return Err(error("shader entry has no stage kind"))
                }
            };
            names.insert(root, unique(format!("{source}_{phase}"), &mut used));
        }
    }
    Ok(names)
}

pub(super) fn buffers(
    data: &Program<Scheduled>,
    pinned: &BTreeSet<BufferId>,
    used: &mut BTreeSet<String>,
) -> Result<BTreeMap<BufferId, String>, OptimizeError> {
    let mut names = BTreeMap::new();
    let device_buffer =
        |id: BufferId| data.state.buffers[id].storage == Storage::Device && !pinned.contains(&id);
    for output in data.state.outputs.values() {
        let Some(id) = output.buffer else { continue };
        if !device_buffer(id) || names.contains_key(&id) {
            continue;
        }
        let source = owner(&data.ir, output.entry)?;
        let (field, kind) = result_field(&data.ir, output)?;
        let field = if kind == ResultKind::Value { "output".into() } else { component(&field) };
        names.insert(id, unique(format!("{source}_{field}"), used));
    }
    for dispatch in data.state.dispatches.values() {
        let source = owner(&data.ir, dispatch.owner)?;
        for &id in dispatch.writes.iter().chain(&dispatch.reads) {
            if device_buffer(id) && !names.contains_key(&id) {
                names.insert(id, unique(format!("{source}_scratch"), used));
            }
        }
    }
    for (&id, buffer) in &data.state.buffers {
        if buffer.storage == Storage::Device && !pinned.contains(&id) && !names.contains_key(&id) {
            names.insert(id, unique("scratch".into(), used));
        }
    }
    Ok(names)
}

pub(super) fn function(ir: &Ir, name: &str, specialization: u32) -> String {
    let source =
        if ir.symbols.values().any(|s| s.name == name) { component(name) } else { "lambda".into() };
    format!("{source}_call_{specialization}")
}
