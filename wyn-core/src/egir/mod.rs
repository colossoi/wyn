//! Acyclic e-graph (aegraph) mid-end optimizer for Wyn.
//!
//! Pipeline: `FuncBody → canonicalize → EGraph → elaborate → FuncBody`
//!
//! The EGraph is a sea-of-nodes-with-CFG representation where:
//! - Pure operators float in a hash-consed acyclic graph (GVN for free)
//! - Side-effectful operators remain anchored in a CFG skeleton
//! - Scoped elaboration converts back to sequential SSA (DCE for free)
//! - Rewrite rules are applied eagerly during construction (Phase 2+)

mod canonicalize;
mod domtree;
mod elaborate;
mod extract;
mod fold;
mod loop_analysis;
mod materialize;
pub mod pipeline;
mod rewrite;
mod scoped_map;
mod skel_opt;
mod soac_expand;
pub mod types;

pub mod from_tlc;

#[cfg(test)]
mod tests;

use crate::ssa::types::{FuncBody, Program};

/// Run the aegraph optimization on a full SSA program.
///
/// Returns a new program where each function/entry point body has been
/// optimized through the aegraph pipeline.
pub fn optimize_program(program: &Program) -> Program {
    let mut functions = Vec::new();
    for func in &program.functions {
        let body = optimize_func(&func.body);
        functions.push(crate::ssa::types::Function {
            name: func.name.clone(),
            span: func.span,
            linkage_name: func.linkage_name.clone(),
            body,
        });
    }

    let mut entry_points = Vec::new();
    for ep in &program.entry_points {
        let body = optimize_func(&ep.body);
        entry_points.push(crate::ssa::types::EntryPoint {
            name: ep.name.clone(),
            execution_model: ep.execution_model.clone(),
            inputs: ep.inputs.clone(),
            outputs: ep.outputs.clone(),
            span: ep.span,
            body,
        });
    }

    Program {
        functions,
        entry_points,
        constants: program.constants.clone(),
        uniforms: program.uniforms.clone(),
        storage: program.storage.clone(),
    }
}

/// Round-trip a function body through the aegraph pipeline (canonicalize →
/// expand_soacs → optimize_skeleton → elaborate). Used by EGIR's own tests
/// to validate the canonicalize-from-SSA path; production goes through
/// `from_tlc::convert_program`.
///
/// Wraps the single body in a one-function `ProgramEgir<Raw>`, runs the full
/// typestate chain, and extracts the single elaborated body back out.
pub(crate) fn optimize_func(body: &FuncBody) -> FuncBody {
    let (graph, _initial_domtree, orig_block_map) = canonicalize::canonicalize(body);

    // Remap body.control_headers (orig block ids) to skeleton block ids.
    let control_headers: std::collections::HashMap<_, _> = body
        .control_headers
        .iter()
        .filter_map(|(orig, hdr)| {
            let skel = orig_block_map.get(orig)?;
            Some((*skel, hdr.remap(&|b| orig_block_map[&b])))
        })
        .collect();

    let params: Vec<_> = body.params.iter().map(|(_, ty, name)| (ty.clone(), name.clone())).collect();

    let func = pipeline::FuncEgir::new(
        "<optimize_func>".to_string(),
        crate::ast::Span::new(0, 0, 0, 0),
        None,
        params,
        body.return_ty.clone(),
        graph,
        control_headers,
    );
    let ssa =
        pipeline::ProgramEgir::single_function(func).expand_soacs().optimize_skeleton().elaborate().ssa;
    ssa.functions.into_iter().next().expect("optimize_func produced no body").body
}
