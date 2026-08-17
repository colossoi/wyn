//! Shared legality and graph-rewrite mechanics for semantic fusion passes.

use std::collections::HashSet;

use polytype::Type;

use crate::ast::TypeName;
use crate::egir::graph_ops;
use crate::egir::ir::{Body, SideEffectSite};
use crate::egir::program::{Entry, OutputWriter, RealizedOutputRoute, SemanticResourceDecl};
use crate::egir::reify::Segmented;
use crate::egir::soac::{lambda as lambda_ops, screma};
use crate::egir::types::{EGraph, OperandRef, ResultBinding, Semantic, ValueId, WynLanguage};
use crate::flow::BlockId;

type FusionBody = Body<Semantic, SemanticResourceDecl, RealizedOutputRoute, WynLanguage>;

pub(super) fn invoke_lambda(
    graph: &mut EGraph,
    program: &Segmented,
    lambda: &screma::Lambda,
    arguments: &[OperandRef],
    captures: &[OperandRef],
) -> Vec<ResultBinding<Type<TypeName>>> {
    debug_assert_eq!(captures.len(), lambda.capture_count());
    let mut operands = Vec::with_capacity(arguments.len() + captures.len());
    operands.extend_from_slice(arguments);
    operands.extend_from_slice(captures);
    let callee = if let Some(body) = lambda.seg_body() {
        debug_assert!(program.contains_region(body.region));
        Some(program.region(body.region).expect("fusion lambda region"))
    } else {
        None
    };
    let block = graph.skeleton.entry;
    lambda_ops::emit_call(graph, block, lambda, callee, operands)
}
pub(super) fn result_used_only_by_effect_pair(
    graph: &EGraph,
    block: BlockId,
    producer: usize,
    consumer: usize,
    result: ValueId,
) -> bool {
    result_used_only_by_effects(
        graph,
        result,
        &[
            SideEffectSite {
                block,
                index: producer,
            },
            SideEffectSite {
                block,
                index: consumer,
            },
        ],
    )
}
/// Whether every executable use of `result` belongs to an allowed effect.
pub(super) fn result_used_only_by_effects(
    graph: &EGraph,
    result: ValueId,
    allowed: &[SideEffectSite],
) -> bool {
    for (block_id, block) in &graph.skeleton.blocks {
        for (index, effect) in block.side_effects.iter().enumerate() {
            if allowed.contains(&SideEffectSite {
                block: block_id,
                index,
            }) {
                continue;
            }
            if graph_ops::effect_value_inputs(graph, effect)
                .into_iter()
                .any(|root| graph_ops::pure_depends_on(graph, root, result))
            {
                return false;
            }
        }
        if block
            .term
            .referenced_nodes()
            .into_iter()
            .any(|root| graph_ops::pure_depends_on(graph, root, result))
        {
            return false;
        }
    }
    true
}

/// Does `root` reach `target` without crossing any node in `cut`?
pub(super) fn pure_depends_on_avoiding(
    graph: &EGraph,
    root: ValueId,
    target: ValueId,
    cut: &HashSet<ValueId>,
) -> bool {
    let mut pending = vec![root];
    let mut visited = HashSet::new();
    while let Some(node) = pending.pop() {
        if cut.contains(&node) || !visited.insert(node) {
            continue;
        }
        if node == target {
            return true;
        }
        if let Some(definition) = graph.nodes.get(node) {
            pending.extend(definition.kind.children());
        }
    }
    false
}

pub(super) fn rewrite_body_graph(body: FusionBody, rewrite: impl FnOnce(&mut EGraph)) -> FusionBody {
    rewrite_body_graph_with_entry(
        body,
        |graph| {
            rewrite(graph);
        },
        |_, ()| {},
    )
}

pub(super) fn rewrite_body_graph_with_entry<T>(
    body: FusionBody,
    rewrite: impl FnOnce(&mut EGraph) -> T,
    finish_entry: impl FnOnce(&mut Entry<Semantic>, T),
) -> FusionBody {
    match body {
        Body::Entry(mut entry) => {
            let result = rewrite(&mut entry.graph);
            finish_entry(&mut entry, result);
            Body::Entry(entry)
        }
        Body::Function(mut function) => {
            rewrite(&mut function.graph);
            Body::Function(function)
        }
        Body::Constant(_) => unreachable!("semantic fusion never targets constants"),
    }
}

pub(super) fn replace_route_sources(entry: &mut Entry<Semantic>, replacements: &[(ValueId, ValueId)]) {
    for route in entry.routes_mut() {
        if let Some((_, replacement)) = replacements.iter().find(|(old, _)| route.source.value == *old) {
            route.source.value = *replacement;
        }
    }
}

pub(super) fn replace_route_values(entry: &mut Entry<Semantic>, replacements: &[(ValueId, ValueId)]) {
    for route in entry.routes_mut() {
        route.replace_values(replacements);
    }
}

pub(super) fn remove_value_writer(entry: &mut Entry<Semantic>, result: ValueId) {
    for route in entry.routes_mut() {
        route.writers.retain(|writer| *writer != OutputWriter::Value(result));
    }
}
