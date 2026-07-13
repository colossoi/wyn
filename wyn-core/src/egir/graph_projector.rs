//! Fresh graph construction for physical-entry recipes.
//!
//! A projection never clones an `EGraph` wholesale. It copies the CFG, the
//! selected side effects, and the transitive value/effect producer closure,
//! allocating fresh block, node, and side-effect-result identities throughout.

use std::collections::{HashMap, HashSet};

use crate::ssa::types::ControlHeader;
use crate::LookupMap;

use super::program::{OutputRoute, OutputWriter};
use super::types::{
    EGraph, ENode, EffectToken, NodeId, SideEffect, SideEffectIndex, SideEffectSite, SkeletonTerminator,
};
use crate::ssa::framework::BlockId;

pub struct GraphProjection {
    pub graph: EGraph,
    pub control_headers: LookupMap<BlockId, ControlHeader>,
    nodes: HashMap<NodeId, NodeId>,
    blocks: HashMap<BlockId, BlockId>,
    effects: HashSet<EffectToken>,
}

impl GraphProjection {
    pub fn node(&self, source: NodeId) -> Option<NodeId> {
        self.nodes.get(&source).copied()
    }

    pub fn effect(&self, source: EffectToken) -> Option<EffectToken> {
        self.effects.contains(&source).then_some(source)
    }

    pub fn block(&self, source: BlockId) -> Option<BlockId> {
        self.blocks.get(&source).copied()
    }

    pub fn remap_aliases(&self, aliases: &LookupMap<NodeId, NodeId>) -> LookupMap<NodeId, NodeId> {
        aliases.iter().filter_map(|(from, to)| Some((self.node(*from)?, self.node(*to)?))).collect()
    }

    pub fn remap_output_routes(&self, routes: Vec<OutputRoute>) -> Result<Vec<OutputRoute>, String> {
        remap_output_routes(
            routes,
            |node| self.node(node),
            |block| self.block(block),
            |effect| self.effect(effect),
            false,
            "graph projection",
        )
    }
}

pub(crate) fn remap_output_routes(
    routes: Vec<OutputRoute>,
    mut map_node: impl FnMut(NodeId) -> Option<NodeId>,
    mut map_block: impl FnMut(BlockId) -> Option<BlockId>,
    mut map_effect: impl FnMut(EffectToken) -> Option<EffectToken>,
    require_writers: bool,
    context: &str,
) -> Result<Vec<OutputRoute>, String> {
    routes
        .into_iter()
        .map(|mut route| {
            route.source.block = map_block(route.source.block)
                .ok_or_else(|| format!("{context} omitted output-route block"))?;
            route.source.value = map_node(route.source.value)
                .ok_or_else(|| format!("{context} omitted output-route value"))?;
            route.writers = route
                .writers
                .into_iter()
                .map(|writer| {
                    let mapped = match writer {
                        OutputWriter::Value(node) => map_node(node).map(OutputWriter::Value),
                        OutputWriter::Effect(effect) => map_effect(effect).map(OutputWriter::Effect),
                    };
                    if require_writers && mapped.is_none() {
                        Err(format!("{context} omitted output-route writer"))
                    } else {
                        Ok(mapped)
                    }
                })
                .collect::<Result<Vec<_>, String>>()?
                .into_iter()
                .flatten()
                .collect();
            Ok(route)
        })
        .collect()
}

pub struct GraphProjector<'a> {
    source: &'a EGraph,
    control_headers: &'a LookupMap<BlockId, ControlHeader>,
}

impl<'a> GraphProjector<'a> {
    pub fn new(source: &'a EGraph, control_headers: &'a LookupMap<BlockId, ControlHeader>) -> Self {
        Self {
            source,
            control_headers,
        }
    }

    pub fn all(&self) -> Result<GraphProjection, String> {
        self.all_with_values(Vec::new())
    }

    pub fn all_with_values(&self, extra_values: Vec<NodeId>) -> Result<GraphProjection, String> {
        let selected = self
            .source
            .skeleton
            .blocks
            .iter()
            .flat_map(|(block, body)| {
                (0..body.side_effects.len()).map(move |index| SideEffectSite { block, index })
            })
            .collect();
        self.selected_with_values(selected, extra_values)
    }

    pub fn selected(&self, roots: HashSet<SideEffectSite>) -> Result<GraphProjection, String> {
        self.selected_with_values(roots, Vec::new())
    }

    pub fn selected_with_values(
        &self,
        roots: HashSet<SideEffectSite>,
        extra_values: Vec<NodeId>,
    ) -> Result<GraphProjection, String> {
        let producers = self.source.side_effect_index();
        let mut selected = roots;
        let mut value_roots = self.terminator_values();
        value_roots.extend(extra_values);
        for site in selected.clone() {
            let effect = self.effect_at(site)?;
            value_roots.extend(effect.referenced_nodes());
        }
        let projected_values = self.close_producers(&mut selected, &mut value_roots, &producers)?;

        let mut graph = EGraph::new();
        let mut blocks = HashMap::new();
        blocks.insert(self.source.skeleton.entry, graph.skeleton.entry);
        for (source_block, _) in &self.source.skeleton.blocks {
            if source_block != self.source.skeleton.entry {
                blocks.insert(source_block, graph.skeleton.create_block());
            }
        }

        let mut nodes = HashMap::new();
        let mut block_params: HashMap<BlockId, Vec<(usize, NodeId)>> = HashMap::new();
        // Parameters and selected effect results must exist before value
        // closure cloning because branches and cross-block uses may point at
        // them before their defining block is visited.
        for (source_id, node) in &self.source.nodes {
            match node {
                ENode::FuncParam { index } => {
                    let target = graph.add_func_param(*index, self.source.types[&source_id].clone());
                    nodes.insert(source_id, target);
                }
                ENode::BlockParam { block, index } => {
                    let target_block = blocks[block];
                    let target =
                        graph.add_block_param(target_block, *index, self.source.types[&source_id].clone());
                    nodes.insert(source_id, target);
                    block_params.entry(target_block).or_default().push((*index, target));
                }
                _ => {}
            }
        }
        for (block, params) in &mut block_params {
            params.sort_by_key(|(index, _)| *index);
            graph.skeleton.blocks[*block].params = params.iter().map(|(_, node)| *node).collect();
        }
        for site in &selected {
            if let Some(result) = self.effect_at(*site)?.result {
                let target = graph.alloc_side_effect_result(self.source.types[&result].clone());
                nodes.insert(result, target);
            }
        }

        for value in projected_values {
            super::graph_ops::clone_value_subgraph(
                self.source,
                &mut graph,
                value,
                &mut nodes,
                super::graph_ops::ConstantCopy::PreserveIdentity,
                true,
            )?;
        }

        let mut effects = HashSet::new();
        for (source_block, body) in &self.source.skeleton.blocks {
            let target_block = blocks[&source_block];
            for (index, effect) in body.side_effects.iter().enumerate() {
                if !selected.contains(&SideEffectSite {
                    block: source_block,
                    index,
                }) {
                    continue;
                }
                let mut projected = effect.clone();
                projected.visit_referenced_nodes_mut(|node| {
                    *node = nodes[node];
                });
                projected.result = effect.result.map(|result| nodes[&result]);
                if let Some((input, output)) = projected.effects {
                    effects.insert(input);
                    effects.insert(output);
                }
                graph.skeleton.blocks[target_block].side_effects.push(projected);
            }
        }

        for (source_block, body) in &self.source.skeleton.blocks {
            let target_block = blocks[&source_block];
            graph.skeleton.blocks[target_block].term = remap_terminator(&body.term, &nodes, &blocks)?;
        }
        let control_headers =
            super::program::remap_control_headers(self.control_headers, |block| blocks[&block]);
        graph.verify_hash_cons()?;
        Ok(GraphProjection {
            graph,
            control_headers,
            nodes,
            blocks,
            effects,
        })
    }

    fn effect_at(&self, site: SideEffectSite) -> Result<&SideEffect, String> {
        self.source
            .skeleton
            .blocks
            .get(site.block)
            .and_then(|block| block.side_effects.get(site.index))
            .ok_or_else(|| format!("invalid graph-projection effect site {site:?}"))
    }

    fn terminator_values(&self) -> Vec<NodeId> {
        self.source.skeleton.blocks.iter().flat_map(|(_, block)| block.term.referenced_nodes()).collect()
    }

    fn close_producers(
        &self,
        selected: &mut HashSet<SideEffectSite>,
        values: &mut Vec<NodeId>,
        producers: &SideEffectIndex,
    ) -> Result<HashSet<NodeId>, String> {
        let mut seen = HashSet::new();
        while let Some(value) = values.pop() {
            if !seen.insert(value) {
                continue;
            }
            let node = self
                .source
                .nodes
                .get(value)
                .ok_or_else(|| format!("graph projection references missing node {value:?}"))?;
            match node {
                ENode::Pure { operands, .. } => values.extend(operands.iter().copied()),
                ENode::Union { left, right } => values.extend([*left, *right]),
                ENode::SideEffectResult => {
                    let site = producers
                        .site(value)
                        .ok_or_else(|| format!("side-effect result {value:?} has no producer"))?;
                    if selected.insert(site) {
                        values.extend(self.effect_at(site)?.referenced_nodes());
                    }
                }
                ENode::FuncParam { .. } | ENode::BlockParam { .. } | ENode::Constant(_) => {}
            }
        }
        Ok(seen)
    }
}

fn remap_terminator(
    term: &SkeletonTerminator,
    nodes: &HashMap<NodeId, NodeId>,
    blocks: &HashMap<BlockId, BlockId>,
) -> Result<SkeletonTerminator, String> {
    let node = |source: NodeId| {
        nodes
            .get(&source)
            .copied()
            .ok_or_else(|| format!("graph projection omitted terminator value {source:?}"))
    };
    term.try_map(node, |target| Ok(blocks[&target]))
}

#[cfg(test)]
#[path = "graph_projector_tests.rs"]
mod graph_projector_tests;
