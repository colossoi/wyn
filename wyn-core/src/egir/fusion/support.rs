//! Shared legality and graph-rewrite mechanics for semantic fusion passes.

use std::collections::HashSet;

use polytype::Type;

use crate::ast::TypeName;
use crate::egir::graph_ops;
use crate::egir::ir::{Body, SideEffectSite};
use crate::egir::program::{Entry, NoStorageDeclaration, OutputWriter, RealizedOutputRoute};
use crate::egir::reify::Segmented;
use crate::egir::soac::{lambda as lambda_ops, screma};
use crate::egir::types::{EGraph, OperandRef, ResultBinding, Semantic, ValueId, WynLanguage};
use crate::flow::BlockId;

type FusionBody = Body<Semantic, NoStorageDeclaration, RealizedOutputRoute, WynLanguage>;

pub(super) fn invoke_lambda(
    graph: &mut EGraph,
    program: &Segmented,
    lambda: &screma::Lambda,
    arguments: &[OperandRef],
    captures: &[OperandRef],
) -> Option<Vec<ResultBinding<Type<TypeName>>>> {
    debug_assert_eq!(captures.len(), lambda.capture_count());
    let mut operands = Vec::with_capacity(arguments.len() + captures.len());
    operands.extend_from_slice(arguments);
    operands.extend_from_slice(captures);
    let callee = if let Some(body) = lambda.seg_body() { Some(program.region(body.region)?) } else { None };
    let block = graph.skeleton.entry;
    Some(lambda_ops::emit_call(graph, block, lambda, callee, operands))
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
    let target = graph.canonical_value(target);
    let mut pending = vec![root];
    let mut visited = HashSet::new();
    while let Some(node) = pending.pop() {
        if cut.contains(&node) {
            continue;
        }
        let node = graph.canonical_value(node);
        if cut.contains(&node) || !visited.insert(node) {
            continue;
        }
        if node == target {
            return true;
        }
        if graph.nodes.get(node).is_some() {
            pending.extend(graph.value_dependencies(node));
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::TypeName;
    use crate::ssa::types::ConstantValue;
    use polytype::Type;

    #[test]
    fn dependency_walk_follows_canonical_aliases() {
        let mut graph = EGraph::new();
        let ty = Type::Constructed(TypeName::Int(32), vec![]);
        let target = graph.alloc_side_effect_result(ty.clone());
        let alias = graph.intern_constant(ConstantValue::I32(0), ty);
        graph.nodes[alias].alias = Some(target);

        assert!(pure_depends_on_avoiding(&graph, alias, target, &HashSet::new()));
        assert!(!pure_depends_on_avoiding(
            &graph,
            alias,
            target,
            &HashSet::from([alias]),
        ));
        assert!(!pure_depends_on_avoiding(
            &graph,
            alias,
            target,
            &HashSet::from([target]),
        ));
    }
}

pub(super) fn try_rewrite_body_graph(
    body: FusionBody,
    rewrite: impl FnOnce(&mut EGraph) -> super::FusionResult<()>,
) -> super::FusionResult<FusionBody> {
    try_rewrite_body_graph_with_entry(body, rewrite, |_, ()| Ok(()))
}

pub(super) fn try_rewrite_body_graph_with_entry<T>(
    body: FusionBody,
    rewrite: impl FnOnce(&mut EGraph) -> super::FusionResult<T>,
    finish_entry: impl FnOnce(&mut Entry<Semantic>, T) -> super::FusionResult<()>,
) -> super::FusionResult<FusionBody> {
    match body {
        Body::Entry(mut entry) => {
            let result = rewrite(&mut entry.graph)?;
            finish_entry(&mut entry, result)?;
            Ok(Body::Entry(entry))
        }
        Body::Function(mut function) => {
            rewrite(&mut function.graph)?;
            Ok(Body::Function(function))
        }
        Body::Constant(_) => Err(super::FusionError::InvalidCandidate(
            "semantic fusion cannot rewrite a constant body".to_owned(),
        )),
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
