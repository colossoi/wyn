//! Scalar rewriting uses the same statements, scopes, and effects as memory.
use super::{memory, OptimizeError, ScalarOptimization};
use egglog_engine::EGraph;

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
    if policy == ScalarOptimization::Full {
        super::timing::evaluate(graph, None, "(InlineScalarFunctions)")?;
        memory::inline(graph)?;
    }
    memory::simplify(graph, true)
}

#[cfg(test)]
#[path = "scalar_tests.rs"]
mod tests;
