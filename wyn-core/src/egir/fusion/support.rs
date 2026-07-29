//! Shared legality and graph-rewrite mechanics for semantic fusion passes.

use std::collections::HashSet;

use polytype::Type;

use crate::ast::TypeName;
use crate::egir::graph_ops;
use crate::egir::ir::{Body, SideEffectSite};
use crate::egir::program::{OutputWriter, RealizedOutputRoute, SemanticEntry, SemanticResourceDecl};
use crate::egir::reify::Segmented;
use crate::egir::soac::{lambda as lambda_ops, screma};
use crate::egir::types::{EGraph, ENode, NodeId, PureOp, Semantic, WynLanguage};
use crate::flow::BlockId;

type FusionBody = Body<Semantic, SemanticResourceDecl, RealizedOutputRoute, WynLanguage>;

pub(super) fn invoke_lambda(
    graph: &mut EGraph,
    program: &Segmented,
    lambda: &screma::Lambda,
    arguments: &[NodeId],
    captures: &[NodeId],
) -> Vec<NodeId> {
    debug_assert_eq!(captures.len(), lambda.capture_count());
    let mut operands = Vec::with_capacity(arguments.len() + captures.len());
    operands.extend_from_slice(arguments);
    operands.extend_from_slice(captures);
    let callee = lambda
        .seg_body()
        .map(|body| program.region(body.region).expect("SOAC lambda region is absent").name.as_str());
    lambda_ops::emit_call(graph, lambda, callee, operands)
}
pub(super) fn result_used_only_by_effect_pair(
    graph: &EGraph,
    block: BlockId,
    producer: usize,
    consumer: usize,
    result: NodeId,
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
    result: NodeId,
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
            if effect.referenced_nodes().any(|root| graph_ops::pure_depends_on(graph, root, result)) {
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
    root: NodeId,
    target: NodeId,
    cut: &HashSet<NodeId>,
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
    finish_entry: impl FnOnce(&mut SemanticEntry, T),
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

pub(super) fn replace_route_sources(entry: &mut SemanticEntry, replacements: &[(NodeId, NodeId)]) {
    for route in entry.routes_mut() {
        if let Some((_, replacement)) = replacements.iter().find(|(old, _)| route.source.value == *old) {
            route.source.value = *replacement;
        }
    }
}

pub(super) fn remove_value_writer(entry: &mut SemanticEntry, result: NodeId) {
    for route in entry.routes_mut() {
        route.writers.retain(|writer| *writer != OutputWriter::Value(result));
    }
}

/// Retarget direct projections of `old_result` according to a partial field map.
pub(super) fn retarget_projects(
    graph: &mut EGraph,
    old_result: NodeId,
    new_result: NodeId,
    mapping: &[Option<usize>],
) {
    let projects = graph
        .nodes
        .iter()
        .filter_map(|(node, definition)| match &definition.kind {
            ENode::Pure {
                op: PureOp::Project { index },
                operands,
            } if operands.first() == Some(&old_result) => Some((node, *index as usize)),
            _ => None,
        })
        .collect::<Vec<_>>();
    for (project, field) in projects {
        let Some(Some(new_field)) = mapping.get(field) else {
            continue;
        };
        graph.update_pure_node(project, |op, operands| {
            *op = PureOp::Project {
                index: *new_field as u32,
            };
            operands[0] = new_result;
        });
    }
}

pub(super) fn rebuild_result(
    graph: &mut EGraph,
    old_result: NodeId,
    new_result: NodeId,
    mapping: &[usize],
    field_types: &[Type<TypeName>],
) {
    let fields = field_types
        .iter()
        .enumerate()
        .map(|(field, ty)| {
            graph.intern_pure(
                PureOp::Project {
                    index: mapping[field] as u32,
                },
                smallvec::smallvec![new_result],
                ty.clone(),
                None,
            )
        })
        .collect();
    let old_type = graph.nodes[old_result].ty.clone();
    let rebuilt = graph.intern_pure(PureOp::Tuple(field_types.len()), fields, old_type, None);
    graph_ops::replace_all_references(graph, old_result, rebuilt);
}
