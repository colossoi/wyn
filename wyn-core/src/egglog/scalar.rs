//! Scalar rewriting uses the same statements, scopes, and effects as memory.
use super::dataflow::{edit, ir::Code};
use super::query::Query;
use super::{memory, output_error, OptimizeError, ScalarOptimization};
use crate::egglog::graph::Graph;
use egglog_engine::sort::S;

mod branch;
mod fold;
mod leaf;

pub(super) fn load(graph: &mut Graph) -> Result<(), OptimizeError> {
    fold::register(graph);
    super::timing::evaluate(
        graph,
        Some("scalar rewrites".into()),
        include_str!("scalar/simplify.egg"),
    )?;
    let mut types = String::new();
    for (primitive, name, integer) in [
        ("I8", "i8", true),
        ("I16", "i16", true),
        ("I32", "i32", true),
        ("I64", "i64", true),
        ("U8", "u8", true),
        ("U16", "u16", true),
        ("U32", "u32", true),
        ("U64", "u64", true),
        ("F16", "f16", false),
        ("F32", "f32", false),
        ("F64", "f64", false),
        ("Bool", "bool", false),
    ] {
        types.push_str(&format!("(set (ScalarTypeName ({primitive})) \"{name}\")\n"));
        if integer {
            types.push_str(&format!("(IntegerScalar ({primitive}))\n"));
        }
    }
    super::timing::evaluate(graph, Some("scalar types".into()), &types)?;
    Ok(())
}

pub(super) fn run(graph: &mut Graph, policy: ScalarOptimization) -> Result<(), OptimizeError> {
    let _timing = super::timing::span("egglog scalar optimization");
    if policy == ScalarOptimization::Full {
        memory::inline(graph, true)?;
    }
    memory::simplify(graph, true)
}

pub(super) fn apply(graph: &mut Graph) -> Result<bool, OptimizeError> {
    if branch::apply(graph)? {
        return Ok(true);
    }
    let query = Query(graph);
    let mut constants = Vec::new();
    query.for_each("FoldedConstant", |row| {
        let output = query.0.output(row[0], 0_i64)?;
        let primitive = query.required("ValuePrimitive", output)?;
        let ty = query.0.value_to_base::<S>(query.required("ScalarTypeName", primitive)?);
        let Some(ty) = fold::scalar_type(ty.as_str()) else {
            return Err(output_error("folded constant has an unsupported scalar type"));
        };
        constants.push((
            row[0],
            row[1],
            Code::Constant {
                ty,
                bits: query.0.value_to_base::<i64>(row[1]),
            },
        ));
        Ok(())
    })?;
    let mut scoped = false;
    let mut structural = Vec::new();
    for (operation, bits, code) in constants {
        if leaf::fold(graph, operation, bits, code.clone())? {
            scoped = true;
        } else {
            structural.push((operation, code));
        }
    }
    if scoped {
        edit::preserve_revisions(graph)?;
    }
    if structural.is_empty() {
        return Ok(false);
    }
    let _timing = super::timing::span("egglog structural constant replacement");
    for (operation, code) in &structural {
        edit::set_code(graph, *operation, code.clone())?;
        edit::inputs(graph, *operation, &[])?;
    }
    Ok(true)
}

pub(super) use leaf::{forward_lengths, materialize_scratch, remove_use, ScalarInputs};

#[cfg(test)]
#[path = "scalar_tests.rs"]
mod tests;
