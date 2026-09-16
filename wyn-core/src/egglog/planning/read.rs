//! Mechanical readout: allocate arena identities for derived resources and
//! index recipe slots. All residency, aliasing and ordering choices are facts.

use super::*;
use crate::egglog::blocks::{BufferData, Storage, Value};
use crate::types::TypeExt;
use egglog_engine::TermDag;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Resource {
    Output(OutputId),
    Source(ExprId),
    Result(OperationId, u32),
    Temporary(OperationId, String, u32),
}

#[derive(Default)]
pub(crate) struct Stage {
    pub owner: Option<EntryId>,
    pub groups: Option<Value>,
    pub reads: BTreeSet<BufferId>,
    pub writes: BTreeSet<BufferId>,
}

#[derive(Default)]
pub(crate) struct Readout {
    pub identities: String,
    pub slots: BTreeMap<(OperationId, String, u32), BufferId>,
    pub stages: BTreeMap<(OperationId, String), Stage>,
}

pub(in crate::egglog) fn read(graph: &EGraph, data: &mut AssociatedData) -> Result<Readout, OptimizeError> {
    let _timing = timing::span("read resources and domains");
    let mut result = Readout::default();
    let mut produced = BTreeSet::new();
    rows(graph, "Produces", 2, |dag, a| {
        produced.insert(resource(dag, a[1])?);
        Ok(())
    })?;
    let mut types = BTreeMap::new();
    rows(graph, "ElementType", 2, |dag, a| {
        let value = resource(dag, a[0])?;
        if matches!(value, Resource::Output(_)) {
            produced.insert(value.clone());
        }
        types.insert(value, super::super::extract::key::<TypeId>(dag, a[1], "TypeId")?);
        Ok(())
    })?;
    let mut allocations = BTreeSet::new();
    rows(graph, "Allocation", 2, |dag, a| {
        allocations.insert(resource(dag, a[0])?);
        Ok(())
    })?;
    let mut buffers = BTreeMap::new();
    for value in &produced {
        if types.contains_key(value) {
            buffers.insert(value.clone(), data.buffers.alloc_id());
        }
    }
    rows(graph, "Backing", 2, |dag, a| {
        let value = resource(dag, a[1])?;
        if let Resource::Source(e) = value {
            let ty = &data.types[data.expressions[e].ty].ty;
            if ty.is_array() && !buffers.contains_key(&value) {
                let id = data.buffers.alloc(BufferData {
                    name: format!("input{}", e.as_u32()),
                    length: Value::op("length", [Value::Source(e)]),
                    element: ty.elem_type().unwrap().clone(),
                    storage: Storage::External(e),
                });
                buffers.insert(value, id);
            }
        }
        Ok(())
    })?;
    rows(graph, "Capacity", 2, |dag, a| {
        let value = resource(dag, a[0])?;
        if let Some(&id) = buffers.get(&value) {
            if let Some(&ty) = types.get(&value) {
                data.buffers.insert(
                    id,
                    BufferData {
                        name: format!("resource{}", id.as_u32()),
                        length: extent(dag, a[1], &buffers)?,
                        element: data.types[ty].ty.clone(),
                        storage: if allocations.contains(&value) {
                            Storage::Device
                        } else {
                            Storage::Discarded
                        },
                    },
                );
            }
        }
        Ok(())
    })?;
    rows(graph, "BufferSlot", 4, |dag, a| {
        if let Some(&id) = buffers.get(&resource(dag, a[3])?) {
            result.slots.insert(
                (operation(dag, a[0])?, string(dag, a[1])?, number(dag, a[2])?),
                id,
            );
        }
        Ok(())
    })?;
    rows(graph, "OutputBacking", 2, |dag, a| {
        data.outputs[OutputId::from(number(dag, a[0])?)].buffer =
            buffers.get(&resource(dag, a[1])?).copied();
        Ok(())
    })?;
    rows(graph, "PhaseOwner", 2, |dag, a| {
        let entry = result.stages.entry(stage(dag, a[0])?).or_default();
        let owner = EntryId::from(number(dag, a[1])?);
        if entry.owner.is_some_and(|other| other != owner) {
            return Err(invalid("shared host dispatch requires entry specialization"));
        }
        entry.owner = Some(owner);
        Ok(())
    })?;
    rows(graph, "PhaseDomain", 3, |dag, a| {
        let n = extent(dag, a[1], &buffers)?;
        let width = Value::Int(number(dag, a[2])?);
        result.stages.entry(stage(dag, a[0])?).or_default().groups = Some(Value::op(
            "min",
            [
                Value::Int(65_535),
                Value::op("max", [Value::Int(1), Value::op("ceil_div", [n, width])]),
            ],
        ));
        Ok(())
    })?;
    rows(graph, "Access", 3, |dag, a| {
        if let Some(&id) = buffers.get(&resource(dag, a[1])?) {
            let entry = result.stages.entry(stage(dag, a[0])?).or_default();
            match string(dag, a[2])?.as_str() {
                "read" => {
                    entry.reads.insert(id);
                }
                "write" => {
                    entry.writes.insert(id);
                }
                _ => return Err(invalid("unknown access")),
            }
        }
        Ok(())
    })?;
    for (value, id) in buffers {
        if data.buffers[id].storage == Storage::Discarded {
            continue;
        }
        let value = match value {
            Resource::Output(id) => format!("(Output {})", id.as_u32()),
            Resource::Source(e) => format!("(Source (ExprId {}))", e.as_u32()),
            Resource::Result(op, slot) => format!("(Result {} {slot})", op.egglog()),
            Resource::Temporary(op, name, slot) => format!("(Temporary {} {name:?} {slot})", op.egglog()),
        };
        writeln!(result.identities, "(EmittedResource {value} {})", id.as_u32()).unwrap();
    }
    Ok(result)
}

fn extent(d: &TermDag, id: usize, buffers: &BTreeMap<Resource, BufferId>) -> Result<Value, OptimizeError> {
    let Term::App(name, a) = d.get(id) else {
        return Err(invalid("extent"));
    };
    Ok(match (name.as_str(), a.as_slice()) {
        ("Fixed", [n]) => Value::Int(number(d, *n)?),
        ("Length", [e]) => Value::op(
            "length",
            [Value::Source(super::super::extract::key(d, *e, "ExprId")?)],
        ),
        ("Scalar", [e]) => Value::Source(super::super::extract::key(d, *e, "ExprId")?),
        ("ChunkCount", [n, width]) => Value::op(
            "max",
            [
                Value::Int(1),
                Value::op(
                    "ceil_div",
                    [extent(d, *n, buffers)?, Value::Int(number(d, *width)?)],
                ),
            ],
        ),
        ("Stored", [v]) => Value::op(
            "index",
            [
                Value::Buffer(*buffers.get(&resource(d, *v)?).ok_or_else(|| invalid("stored length"))?),
                Value::Int(0),
            ],
        ),
        _ => return Err(invalid("extent")),
    })
}

fn resource(d: &TermDag, id: usize) -> Result<Resource, OptimizeError> {
    let Term::App(name, a) = d.get(id) else {
        return Err(invalid("resource"));
    };
    Ok(match (name.as_str(), a.as_slice()) {
        ("Output", [id]) => Resource::Output(OutputId::from(number(d, *id)?)),
        ("Source", [e]) => Resource::Source(super::super::extract::key(d, *e, "ExprId")?),
        ("Result", [o, i]) => Resource::Result(operation(d, *o)?, number(d, *i)?),
        ("Temporary", [o, name, i]) => {
            Resource::Temporary(operation(d, *o)?, string(d, *name)?, number(d, *i)?)
        }
        _ => return Err(invalid("resource")),
    })
}
fn stage(d: &TermDag, id: usize) -> Result<(OperationId, String), OptimizeError> {
    let a = super::super::extract::app(d, id, "Stage", 2)?;
    Ok((operation(d, a[0])?, string(d, a[1])?))
}
fn operation(d: &TermDag, id: usize) -> Result<OperationId, OptimizeError> {
    super::super::extract::key(d, id, "OperationId")
}
fn number(d: &TermDag, id: usize) -> Result<u32, OptimizeError> {
    let Term::Lit(Literal::Int(n)) = d.get(id) else {
        return Err(invalid("integer"));
    };
    u32::try_from(*n).map_err(|_| invalid("integer range"))
}
fn string(d: &TermDag, id: usize) -> Result<String, OptimizeError> {
    let Term::Lit(Literal::String(s)) = d.get(id) else {
        return Err(invalid("string"));
    };
    Ok(s.to_string())
}
fn rows(
    graph: &EGraph,
    name: &str,
    arity: usize,
    mut f: impl FnMut(&TermDag, &[usize]) -> Result<(), OptimizeError>,
) -> Result<(), OptimizeError> {
    let (rows, _, dag) = graph.function_to_dag(name, usize::MAX, false)?;
    for row in rows {
        f(&dag, super::super::extract::app(&dag, row, name, arity)?)?;
    }
    Ok(())
}
fn invalid(what: &str) -> OptimizeError {
    OptimizeError::Output(format!("relational plan readout: {what}"))
}
