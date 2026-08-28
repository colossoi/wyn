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
use thiserror::Error;

use crate::ast::{Span, TypeName};

use crate::egir::ir::BodySite;
use crate::egir::program::{Entry, SemanticOpId};
use crate::egir::reify::Segmented;
use crate::egir::semantic_graph::SemanticGraph;
use crate::egir::types::{
    EGraph, Semantic, SideEffect, SideEffectKind, SoacEffect, SoacInputType, ValueId,
};
use crate::flow::BlockId;

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FusionEffect(SemanticOpId);

#[derive(Clone, Debug)]
struct FusionInput {
    node: ValueId,
    ty: SoacInputType,
}

impl FusionInput {
    fn element(&self) -> Type<TypeName> {
        self.ty.element()
    }

    fn join(nodes: &[ValueId], types: &[SoacInputType]) -> Option<Vec<Self>> {
        if nodes.len() != types.len() {
            return None;
        }
        Some(nodes.iter().copied().zip(types.iter().cloned()).map(|(node, ty)| Self { node, ty }).collect())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ResolvedFusionEffect {
    body: BodySite,
    block: BlockId,
    index: usize,
}

#[derive(Debug, Error)]
pub(crate) enum FusionError {
    #[error("semantic fusion operation {0:?} is absent from the program")]
    MissingEffect(SemanticOpId),

    #[error("semantic fusion operation {0:?} occurs more than once in the program")]
    DuplicateEffect(SemanticOpId),

    #[error("semantic fusion operations {left:?} and {right:?} are no longer in the same block")]
    SeparatedEffects {
        left: SemanticOpId,
        right: SemanticOpId,
    },

    #[error("semantic fusion candidate is no longer valid: {0}")]
    InvalidCandidate(String),
}

fn resolve_pair(
    program: &Segmented,
    left: FusionEffect,
    right: FusionEffect,
) -> FusionResult<(ResolvedFusionEffect, ResolvedFusionEffect)> {
    let left_location = left.resolve(program)?;
    let right_location = right.resolve(program)?;
    if left_location.body != right_location.body || left_location.block != right_location.block {
        return Err(FusionError::SeparatedEffects {
            left: left.0,
            right: right.0,
        });
    }
    Ok((left_location, right_location))
}

type FusionResult<T> = std::result::Result<T, FusionError>;

impl FusionEffect {
    fn from_effect(effect: &SideEffect) -> Option<Self> {
        let SideEffectKind::Soac(SoacEffect(id, _)) = effect.kind() else {
            return None;
        };
        Some(Self(*id))
    }

    fn resolve(self, program: &Segmented) -> FusionResult<ResolvedFusionEffect> {
        let mut found = None;
        for (body, graph, _) in bodies(program) {
            for (block, contents) in &graph.skeleton.blocks {
                for (index, effect) in contents.side_effects.iter().enumerate() {
                    if Self::from_effect(effect) != Some(self) {
                        continue;
                    }
                    if found.is_some() {
                        return Err(FusionError::DuplicateEffect(self.0));
                    }
                    found = Some(ResolvedFusionEffect { body, block, index });
                }
            }
        }
        let Some(found) = found else {
            return Err(FusionError::MissingEffect(self.0));
        };
        Ok(found)
    }
}

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
fn apply(program: Segmented, rewrite: Rewrite) -> FusionResult<Segmented> {
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
pub(super) fn rewrite_once(program: Segmented, oracle: &SemanticGraph) -> FusionResult<(Segmented, bool)> {
    match analyze(&program, oracle) {
        Some(rewrite) => Ok((apply(program, rewrite)?, true)),
        None => Ok((program, false)),
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

pub(super) fn graph_and_span(program: &Segmented, site: BodySite) -> FusionResult<(&EGraph, Span, String)> {
    let Some(graph) = program.body_graph(site) else {
        return Err(FusionError::InvalidCandidate(format!(
            "semantic fusion body {site:?} is absent"
        )));
    };
    let (span, scope) = match site {
        BodySite::Entry(index) => {
            let Some(entry) = program.entry_points.get(index) else {
                return Err(FusionError::InvalidCandidate(format!(
                    "semantic fusion entry {index} is absent"
                )));
            };
            (entry.span, entry.name.clone())
        }
        BodySite::Function(region) => {
            let Some(function) = program.region(region) else {
                return Err(FusionError::InvalidCandidate(format!(
                    "semantic fusion region {region:?} is absent"
                )));
            };
            (function.span, function.name.clone())
        }
        BodySite::Constant(_) => {
            return Err(FusionError::InvalidCandidate(
                "semantic fusion cannot target a constant body".to_owned(),
            ));
        }
    };
    Ok((graph, span, scope))
}

pub(super) fn capture_types<'a>(
    types: &LookupMap<ValueId, Type<TypeName>>,
    captures: impl Iterator<Item = &'a super::ir::OperandRef>,
) -> Option<Vec<Type<TypeName>>> {
    captures
        .map(|capture| {
            let value = capture.value()?;
            types.get(&value).cloned()
        })
        .collect()
}

/// Canonicalize a semantic operation's parallel array inputs by `ValueId` and
/// return an old-index to new-index map. Fusion frequently concatenates input
/// vectors from independently built operations; retaining duplicate nodes
/// would duplicate region parameters and obscure equal-domain provenance.
fn deduplicate_array_inputs(inputs: Vec<FusionInput>) -> (Vec<FusionInput>, Vec<usize>) {
    let mut unique = Vec::<FusionInput>::new();
    let mut remap = Vec::with_capacity(inputs.len());
    for input in inputs {
        if let Some(index) = unique.iter().position(|existing| existing.node == input.node) {
            debug_assert_eq!(unique[index].ty.array, input.ty.array);
            debug_assert_eq!(unique[index].ty.dimensions, input.ty.dimensions);
            debug_assert_eq!(unique[index].ty.layout, input.ty.layout);
            remap.push(index);
        } else {
            remap.push(unique.len());
            unique.push(input);
        }
    }
    (unique, remap)
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
