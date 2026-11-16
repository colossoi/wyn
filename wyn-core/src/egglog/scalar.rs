//! Scalar rewriting uses the same statements, scopes, and effects as memory.
use super::dataflow::{edit, ir::Code};
use super::query::Query;
use super::{memory, output_error, OptimizeError, ScalarOptimization};
use crate::types;
use egglog_engine::sort::S;
use egglog_engine::EGraph;
use egglog_engine::Write;

mod fold;

pub(super) fn load(graph: &mut EGraph) -> Result<(), OptimizeError> {
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

pub(super) fn run(graph: &mut EGraph, policy: ScalarOptimization) -> Result<(), OptimizeError> {
    let _timing = super::timing::span("egglog scalar optimization");
    if policy == ScalarOptimization::Full {
        graph.update(|mut sink| {
            sink.remove("InlineAll", egglog_engine::RawValues(Vec::new()))?;
            sink.set("InlineAll", egglog_engine::RawValues(Vec::new()), true)
        })?;
        memory::inline(graph)?;
    }
    memory::simplify(graph, true)
}

pub(super) fn apply(graph: &mut EGraph) -> Result<bool, OptimizeError> {
    let query = Query(graph);
    if let Some(candidate) = query.first("BranchCandidate")? {
        let operation = candidate[0];
        let region = candidate[1];
        let outputs = query.sequence("Output", operation)?;
        let yields = query.sequence("RegionYield", region)?;
        if outputs.len() != yields.len() {
            return Err(output_error("constant branch has inconsistent results"));
        }
        edit::replace(graph, &outputs.into_iter().zip(yields).collect::<Vec<_>>())?;
        let operations = Query(graph).operations(region)?;
        edit::splice(graph, operation, &operations)?;
        return Ok(true);
    }
    let mut constants = Vec::new();
    query.for_each("FoldedConstant", |row| {
        let output = query.required("Output", (row[0], 0_i64))?;
        let primitive = query.required("ValuePrimitive", output)?;
        let ty = query.0.value_to_base::<S>(query.required("ScalarTypeName", primitive)?);
        let Some(ty) = fold::scalar_type(ty.as_str()) else {
            return Err(output_error("folded constant has an unsupported scalar type"));
        };
        constants.push((
            row[0],
            Code::Constant {
                ty,
                bits: query.0.value_to_base::<i64>(row[1]),
            },
        ));
        Ok(())
    })?;
    for (operation, code) in &constants {
        edit::set_code(graph, *operation, code.clone())?;
        edit::inputs(graph, *operation, &[])?;
    }
    if !constants.is_empty() {
        return Ok(true);
    }
    if let Some(candidate) = Query(graph).first("ForwardLength")? {
        let operation = candidate[0];
        let index = candidate[1];
        let code = if let Some(bits) = Query(graph).lookup("StaticIndex", index)? {
            Code::Constant {
                ty: types::i32(),
                bits: graph.value_to_base::<i64>(bits),
            }
        } else {
            Code::IndexValue
        };
        let indexed = matches!(code, Code::IndexValue);
        edit::set_code(graph, operation, code)?;
        edit::inputs(graph, operation, &[])?;
        if indexed {
            graph.update(|mut sink| sink.set("OperationIndex", (operation, 0_i64), index))?;
        }
        return Ok(true);
    }
    Ok(false)
}

#[cfg(test)]
#[path = "scalar_tests.rs"]
mod tests;
