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

/// Optimize a single function body through the aegraph pipeline.
pub(crate) fn optimize_func(body: &FuncBody) -> FuncBody {
    // Phase 1: canonicalize SSA → sea-of-nodes (with hash-consing = GVN).
    let (mut graph, initial_domtree, orig_block_map) = canonicalize::canonicalize(body);

    // Remap body.control_headers (orig block ids) to skeleton block ids so
    // soac_expand can insert new Loop headers keyed on skeleton blocks.
    let mut control_headers: std::collections::HashMap<_, _> = body
        .control_headers
        .iter()
        .filter_map(|(orig, hdr)| {
            let skel = orig_block_map.get(orig)?;
            Some((*skel, hdr.remap(&|b| orig_block_map[&b])))
        })
        .collect();

    // Phase 1b: expand SOAC side-effects into loops (mutates skeleton + headers).
    soac_expand::expand_soacs(&mut graph, &mut control_headers);

    // Phase 1c: skeleton rewrites (branch folding + redundant phi elim).
    let aliases = skel_opt::optimize_skeleton(&mut graph);

    // Skeleton rewrites can change CFG edges; rebuild the skeleton domtree.
    let skel_domtree = domtree::DomTree::build(&domtree::SkeletonCfgView {
        skeleton: &graph.skeleton,
    });
    drop(initial_domtree);

    let params: Vec<_> = body.params.iter().map(|(_, ty, name)| (ty.clone(), name.clone())).collect();

    // elaborate takes an orig→skel block map and control_headers keyed on orig.
    // We already remapped headers into skel-space, so use an identity map.
    let identity_map: std::collections::HashMap<_, _> =
        graph.skeleton.blocks.keys().map(|b| (b, b)).collect();

    // Phase 2: elaborate sea-of-nodes → FuncBody (with scoped dedup = DCE).
    elaborate::elaborate(
        &graph,
        &skel_domtree,
        &params,
        body.return_ty.clone(),
        &control_headers,
        &identity_map,
        &aliases,
    )
}
