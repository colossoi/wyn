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
use crate::egir::program::Entry;
use crate::egir::reify::Segmented;
use crate::egir::semantic_graph::SemanticGraph;
use crate::egir::types::{EGraph, Semantic, ValueId};

use crate::LookupMap;

mod envelope;
mod filter;
mod histogram;
mod horizontal;
mod indexed;
mod map_anchor;
mod screma;
mod space;
mod support;
mod vertical;

enum Rewrite {
    Indexed(indexed::Candidate),
    Vertical(vertical::Candidate),
    Histogram(map_anchor::Candidate),
    Envelope(map_anchor::Candidate),
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

/// Iterate fusion candidates in entry-first priority order. Constant bodies
/// are excluded because semantic fusion never targets them.
pub(super) fn bodies(
    program: &Segmented,
) -> impl Iterator<Item = (BodySite, &EGraph, Option<&Entry<Semantic>>)> {
    program
        .entry_points
        .iter()
        .enumerate()
        .map(|(index, entry)| (BodySite::Entry(index), &entry.graph, Some(entry)))
        .chain(
            program
                .functions
                .iter()
                .map(|function| (BodySite::Function(function.region), &function.graph, None)),
        )
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
    types: &LookupMap<ValueId, Type<TypeName>>,
    captures: impl Iterator<Item = &'a super::ir::OperandRef>,
) -> Vec<Type<TypeName>> {
    captures
        .map(|capture| {
            let value = capture
                .value()
                .expect("fusion cannot internalize an address-only capture before destination selection");
            types.get(&value).expect("capture node is absent from its owning graph").clone()
        })
        .collect()
}

/// Canonicalize a semantic operation's parallel array inputs by `ValueId` and
/// return an old-index to new-index map. Fusion frequently concatenates input
/// vectors from independently built operations; retaining duplicate nodes
/// would duplicate region parameters and obscure equal-domain provenance.
pub(super) fn deduplicate_array_inputs(
    nodes: Vec<ValueId>,
    array_types: Vec<Type<TypeName>>,
    elem_types: Vec<Type<TypeName>>,
) -> (Vec<ValueId>, Vec<Type<TypeName>>, Vec<Type<TypeName>>, Vec<usize>) {
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
#[path = "mod_tests.rs"]
mod tests;
