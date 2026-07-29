//! Semantic SegOp fusion (EGIR milestone 5).
//!
//! These passes are driven by `egir::semantic_opt` and rest on two primitives:
//! provenance-based `SegSpace` equality (`space`), and
//! `egir::semantic_graph::SemanticGraph`, the query layer over the semantic
//! dependency DAG. That oracle owns the invariant fusion rests on: never move
//! an operation across resource or effect ordering.
//!
//! Horizontal fusion combines independent siblings. Vertical fusion composes
//! canonical Scremas while retaining producer outputs needed by other consumers
//! or output routes. Histogram fusion composes pure maps into the general
//! bucket lambda shared by ordered scatter and reducing histogram updates.

use polytype::Type;

use crate::ast::{Span, TypeName};

use crate::egir::ir::BodySite;
use crate::egir::reify::Segmented;
use crate::egir::semantic_graph::SemanticGraph;
use crate::egir::types::{EGraph, NodeId};

use crate::LookupMap;

mod envelope;
mod filter;
mod histogram;
mod horizontal;
mod indexed;
mod screma;
mod space;
mod support;
mod vertical;

enum Rewrite {
    Indexed(indexed::Candidate),
    Vertical(vertical::Candidate),
    Histogram(histogram::Candidate),
    Envelope(envelope::Candidate),
    Filter(filter::Candidate),
    Horizontal(horizontal::Candidate),
}

/// Select one semantics-preserving rewrite. Each call applies exactly one
/// candidate so the dependency oracle is rebuilt before another fusion.
fn analyze(program: &Segmented, oracle: &SemanticGraph) -> Option<Rewrite> {
    indexed::analyze(program)
        .map(Rewrite::Indexed)
        .or_else(|| vertical::analyze(program, oracle).map(Rewrite::Vertical))
        .or_else(|| histogram::analyze(program, oracle).map(Rewrite::Histogram))
        .or_else(|| envelope::analyze(program, oracle).map(Rewrite::Envelope))
        .or_else(|| filter::analyze(program, oracle).map(Rewrite::Filter))
        .or_else(|| horizontal::analyze(program, oracle).map(Rewrite::Horizontal))
}

/// Consume a rewrite selected by [`analyze`].
fn apply(program: Segmented, rewrite: Rewrite) -> Segmented {
    match rewrite {
        Rewrite::Indexed(candidate) => indexed::apply(program, candidate),
        Rewrite::Vertical(candidate) => vertical::apply(program, candidate),
        Rewrite::Histogram(candidate) => histogram::apply(program, candidate),
        Rewrite::Envelope(candidate) => envelope::apply(program, candidate),
        Rewrite::Filter(candidate) => filter::apply(program, candidate),
        Rewrite::Horizontal(candidate) => horizontal::apply(program, candidate),
    }
}

/// Apply at most one fusion rewrite while keeping candidate types private to
/// this module. The caller rebuilds the dependency oracle after a change.
pub(super) fn rewrite_once(program: Segmented, oracle: &SemanticGraph) -> (Segmented, bool) {
    match analyze(&program, oracle) {
        Some(rewrite) => (apply(program, rewrite), true),
        None => (program, false),
    }
}

pub(super) fn graph_and_span(program: &Segmented, site: BodySite) -> (&EGraph, Span, String) {
    let graph = program.body_graph(site).expect("semantic fusion body");
    let (span, scope) = match site {
        BodySite::Entry(index) => {
            let entry = &program.entry_points[index];
            (entry.span, entry.name.clone())
        }
        BodySite::Function(region) => {
            let function = program.region(region).expect("fusion region");
            (function.span, function.name.clone())
        }
        BodySite::Constant(_) => unreachable!("semantic fusion never targets constants"),
    };
    (graph, span, scope)
}

pub(super) fn capture_types<'a>(
    types: &LookupMap<NodeId, Type<TypeName>>,
    captures: impl Iterator<Item = &'a NodeId>,
) -> Vec<Type<TypeName>> {
    captures
        .map(|capture| types.get(capture).expect("capture node is absent from its owning graph").clone())
        .collect()
}

/// Canonicalize a semantic operation's parallel array inputs by `NodeId` and
/// return an old-index to new-index map. Fusion frequently concatenates input
/// vectors from independently built operations; retaining duplicate nodes
/// would duplicate region parameters and obscure equal-domain provenance.
pub(super) fn deduplicate_array_inputs(
    nodes: Vec<NodeId>,
    array_types: Vec<Type<TypeName>>,
    elem_types: Vec<Type<TypeName>>,
) -> (Vec<NodeId>, Vec<Type<TypeName>>, Vec<Type<TypeName>>, Vec<usize>) {
    debug_assert_eq!(nodes.len(), array_types.len());
    debug_assert_eq!(nodes.len(), elem_types.len());
    let mut unique_nodes = Vec::new();
    let mut unique_array_types = Vec::new();
    let mut unique_elem_types = Vec::new();
    let mut remap = Vec::with_capacity(nodes.len());
    for ((node, array_ty), elem_ty) in nodes.into_iter().zip(array_types).zip(elem_types) {
        if let Some(index) = unique_nodes.iter().position(|existing| *existing == node) {
            debug_assert_eq!(unique_array_types[index], array_ty);
            debug_assert_eq!(unique_elem_types[index], elem_ty);
            remap.push(index);
        } else {
            remap.push(unique_nodes.len());
            unique_nodes.push(node);
            unique_array_types.push(array_ty);
            unique_elem_types.push(elem_ty);
        }
    }
    (unique_nodes, unique_array_types, unique_elem_types, remap)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::egir::types::{SideEffectKind, Soac, SoacEffect};

    fn reified(source: &str) -> crate::egir::reify::Segmented {
        let program = crate::compile_thru_tlc(source).expect("compile through TLC");
        let program = crate::tlc::infer_input_slice_bounds(program);
        let program = crate::to_egraph(program).expect("convert to raw EGIR");
        let program = crate::egir::realize_outputs(program).expect("realize EGIR outputs");
        crate::egir::reify_soacs(program)
    }

    fn force_horizontal_then_vertical(source: &str) -> crate::egir::ResourcesAllocated {
        let program = reified(source);
        let dependencies = crate::egir::semantic_graph::dependencies(&program);
        let oracle = SemanticGraph::new(&dependencies);
        let horizontal = horizontal::analyze(&program, &oracle)
            .expect("the sibling collective and array producer should fuse horizontally");
        let program = horizontal::apply(program, horizontal);

        let producer = program
            .entry_points
            .iter()
            .flat_map(|entry| entry.graph.skeleton.blocks.iter().flat_map(|(_, block)| &block.side_effects))
            .find_map(|effect| {
                let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &effect.kind else {
                    return None;
                };
                (!op.form.reductions.is_empty() && !op.form.post.result_types.is_empty()).then_some(op)
            })
            .expect("horizontal fusion should construct a reduction-bearing producer");
        assert_eq!(producer.form.reduction_result_count(), 1);
        assert_eq!(producer.form.post.result_types.len(), 1);

        let dependencies = crate::egir::semantic_graph::dependencies(&program);
        let oracle = SemanticGraph::new(&dependencies);
        let vertical = vertical::analyze(&program, &oracle).unwrap_or_else(|| {
            panic!(
                "the reduction-bearing producer should fuse into its map consumer:\n{}\ndependencies: {dependencies:#?}",
                crate::egir::semantic_graph::summary(&program)
            )
        });
        let program = vertical::apply(program, vertical);
        let optimized = crate::egir::optimize_semantics(program);
        crate::egir::plan_logical_resources(optimized)
            .expect("allocate the vertically normalized composite Screma")
    }

    fn assert_composite_and_lower(allocated: crate::egir::ResourcesAllocated, scans: usize) {
        let scremas = allocated
            .entry_points
            .iter()
            .flat_map(|entry| entry.graph.skeleton.blocks.iter().flat_map(|(_, block)| &block.side_effects))
            .filter_map(|effect| {
                let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &effect.kind else {
                    return None;
                };
                Some(op)
            })
            .collect::<Vec<_>>();
        assert_eq!(scremas.len(), 1);
        assert_eq!(scremas[0].form.scan_input_count(), scans);
        assert_eq!(scremas[0].form.reduction_result_count(), 1);
        assert_eq!(scremas[0].form.post.result_types.len(), 1);
        assert!(scremas[0].validate().is_ok());

        let planned = crate::egir::plan(allocated, crate::LoweringProfile::PORTABLE)
            .expect("plan the vertically normalized composite Screma");
        crate::lower_egir_to_ssa(planned).expect("lower the vertically normalized composite Screma");
    }

    #[test]
    fn vertical_normalization_accepts_a_reduction_bearing_producer() {
        let allocated = force_horizontal_then_vertical(
            r#"
#[compute]
entry redomap_then_map<[n]>(xs: [n]i32) (i32, [n]i32) =
  let mapped = map(|x: i32| x + 1, xs) in
  let total = reduce(|a: i32, b: i32| a + b, 0, xs) in
  let consumed = map(|x: i32| x * 2, mapped) in
  (total, consumed)
"#,
        );
        assert_composite_and_lower(allocated, 0);
    }

    #[test]
    fn vertical_normalization_accepts_a_scan_and_reduction_producer() {
        let allocated = force_horizontal_then_vertical(
            r#"
#[compute]
entry scan_redomap_then_map<[n]>(xs: [n]i32) (i32, [n]i32) =
  let prefixes = scan(|a: i32, b: i32| a + b, 0, xs) in
  let total = reduce(|a: i32, b: i32| a + b, 0, xs) in
  let consumed = map(|x: i32| x * 2, prefixes) in
  (total, consumed)
"#,
        );
        assert_composite_and_lower(allocated, 1);
    }
}
