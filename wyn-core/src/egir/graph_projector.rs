//! Fresh graph construction for physical-entry recipes.
//!
//! A projection never clones an `EGraph` wholesale. It copies the CFG, the
//! selected side effects, and the transitive value/effect producer closure,
//! allocating fresh block, node, and side-effect-result identities throughout.

use std::collections::{HashMap, HashSet};

use crate::flow::{BlockId, ControlHeader};
use crate::LookupMap;

use super::program::{OutputRoute, OutputWriter};
use super::types::{
    EGraph, ENode, EffectToken, NodeId, Semantic, SideEffect, SideEffectIndex, SideEffectSite,
    SkeletonTerminator,
};
pub struct GraphProjection {
    pub graph: EGraph<Semantic>,
    pub control_headers: LookupMap<BlockId, ControlHeader>,
    nodes: HashMap<NodeId, NodeId>,
    blocks: HashMap<BlockId, BlockId>,
    effects: HashSet<EffectToken>,
    source_effects: HashSet<SideEffectSite>,
}

/// A projected producer recipe together with the projected identity of the
/// value requested by the caller.
pub struct ProjectedValueRecipe {
    pub projection: GraphProjection,
    pub value: NodeId,
    pub result_block: BlockId,
    pub source: ValueRecipeSource,
}

/// How a projected value recipe is removed from its source entry after its
/// result has been replaced by a handoff load. This describes a prefix
/// boundary, not the control construct inside it: selections, loops, and
/// nested structured regions all use the same continuation form.
pub enum ValueRecipeSource {
    EntryBlock,
    StructuredPrefix {
        continuation: BlockId,
    },
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

    pub fn source_effects(&self) -> &HashSet<SideEffectSite> {
        &self.source_effects
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
    source: &'a EGraph<Semantic>,
    control_headers: &'a LookupMap<BlockId, ControlHeader>,
}

#[derive(Clone, Copy)]
enum ProjectionMode {
    Complete,
    EntryRecipe {
        effect_limit: Option<usize>,
    },
    DetachedRecipe {
        block: BlockId,
    },
    StructuredPrefix {
        continuation: BlockId,
        effect_limit: usize,
    },
}

struct ProjectionSelection {
    blocks: HashSet<BlockId>,
    effects: HashSet<SideEffectSite>,
    values: HashSet<NodeId>,
}

struct ProjectionShell {
    graph: EGraph<Semantic>,
    blocks: HashMap<BlockId, BlockId>,
    nodes: HashMap<NodeId, NodeId>,
}

impl<'a> GraphProjector<'a> {
    pub fn new(
        source: &'a EGraph<Semantic>,
        control_headers: &'a LookupMap<BlockId, ControlHeader>,
    ) -> Self {
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
        self.project(selected, extra_values, ProjectionMode::Complete)
    }

    pub fn selected(&self, roots: HashSet<SideEffectSite>) -> Result<GraphProjection, String> {
        self.selected_with_values(roots, Vec::new())
    }

    /// Project selected operation effects as a standalone entry recipe. An
    /// operation in a continuation block may be detached when its complete
    /// producer closure is independent of that block's parameters and of
    /// effects in other blocks.
    pub fn selected_operation_recipe(
        &self,
        roots: HashSet<SideEffectSite>,
    ) -> Result<GraphProjection, String> {
        let mut blocks = roots.iter().map(|site| site.block);
        let Some(block) = blocks.next() else {
            return Err("operation recipe effects must belong to one block".into());
        };
        if blocks.any(|other| other != block) {
            return Err("operation recipe effects must belong to one block".into());
        }
        let mode = if block == self.source.skeleton.entry {
            ProjectionMode::EntryRecipe { effect_limit: None }
        } else {
            ProjectionMode::DetachedRecipe { block }
        };
        self.project(roots, Vec::new(), mode)
    }

    /// Project the prefix recipe for a value captured by a parallel operation.
    /// A consumer in the entry block produces the existing straight-line
    /// recipe. A later consumer block projects the complete structured CFG
    /// prefix ending at that block, without encoding whether the prefix
    /// contains a selection, loop, or nested combination of both.
    pub fn captured_value_recipe(
        &self,
        value: NodeId,
        consumer: SideEffectSite,
    ) -> Result<ProjectedValueRecipe, String> {
        let (mode, source) = if consumer.block == self.source.skeleton.entry {
            (
                ProjectionMode::EntryRecipe {
                    effect_limit: Some(consumer.index),
                },
                ValueRecipeSource::EntryBlock,
            )
        } else {
            let continuation = self
                .source
                .skeleton
                .blocks
                .get(consumer.block)
                .ok_or_else(|| "captured value consumer block is absent".to_string())?;
            if continuation.params.as_slice() != [value] {
                return Err("structured prefix must produce one captured boundary value".into());
            }
            (
                ProjectionMode::StructuredPrefix {
                    continuation: consumer.block,
                    effect_limit: consumer.index,
                },
                ValueRecipeSource::StructuredPrefix {
                    continuation: consumer.block,
                },
            )
        };
        let projection = self.project(HashSet::new(), vec![value], mode)?;
        let projected = projection
            .node(value)
            .ok_or_else(|| "captured value projection omitted its root".to_string())?;
        let result_block = projection
            .block(consumer.block)
            .ok_or_else(|| "captured value projection omitted its result block".to_string())?;
        Ok(ProjectedValueRecipe {
            projection,
            value: projected,
            result_block,
            source,
        })
    }

    pub fn selected_with_values(
        &self,
        roots: HashSet<SideEffectSite>,
        extra_values: Vec<NodeId>,
    ) -> Result<GraphProjection, String> {
        self.project(roots, extra_values, ProjectionMode::Complete)
    }

    fn project(
        &self,
        selected: HashSet<SideEffectSite>,
        extra_values: Vec<NodeId>,
        mode: ProjectionMode,
    ) -> Result<GraphProjection, String> {
        let selection = self.select_projection(selected, extra_values, mode)?;
        let mut shell = self.projection_shell(mode, &selection)?;
        for value in &selection.values {
            super::graph_ops::clone_value_subgraph(
                self.source,
                &mut shell.graph,
                *value,
                &mut shell.nodes,
                super::graph_ops::ConstantCopy::PreserveIdentity,
                true,
            )?;
        }
        let effects = self.clone_effects(&selection, &mut shell)?;
        self.project_terminators(mode, &selection.blocks, &mut shell)?;
        let control_headers = if matches!(
            mode,
            ProjectionMode::EntryRecipe { .. } | ProjectionMode::DetachedRecipe { .. }
        ) {
            LookupMap::new()
        } else {
            self.project_control_headers(&shell.blocks)?
        };
        shell.graph.verify_hash_cons()?;
        super::graph_ops::verify_branch_arities(&shell.graph)?;
        Ok(GraphProjection {
            graph: shell.graph,
            control_headers,
            nodes: shell.nodes,
            blocks: shell.blocks,
            effects,
            source_effects: selection.effects,
        })
    }

    fn select_projection(
        &self,
        mut selected: HashSet<SideEffectSite>,
        extra_values: Vec<NodeId>,
        mode: ProjectionMode,
    ) -> Result<ProjectionSelection, String> {
        let blocks = self.projected_blocks(mode)?;
        let allowed_effects = self.allowed_effects(mode, &blocks);
        if matches!(mode, ProjectionMode::StructuredPrefix { .. }) {
            selected.extend(allowed_effects.iter().copied());
        }
        let mut roots = self.projected_terminator_values(mode, &blocks);
        roots.extend(extra_values);
        for site in selected.clone() {
            roots.extend(self.effect_at(site)?.referenced_nodes());
        }
        let values = self.close_producers(&mut selected, &mut roots, &self.source.side_effect_index())?;
        if selected.iter().any(|site| !allowed_effects.contains(site)) {
            return Err("value recipe depends on an effect outside its prefix boundary".into());
        }
        if values.iter().any(|node| match self.source.nodes[*node] {
            ENode::BlockParam { block, .. } => {
                !blocks.contains(&block) || matches!(mode, ProjectionMode::DetachedRecipe { .. })
            }
            _ => false,
        }) {
            return Err("value recipe depends on a block parameter outside its prefix boundary".into());
        }
        Ok(ProjectionSelection {
            blocks,
            effects: selected,
            values,
        })
    }

    fn projection_shell(
        &self,
        mode: ProjectionMode,
        selection: &ProjectionSelection,
    ) -> Result<ProjectionShell, String> {
        let mut graph = EGraph::new();
        let source_entry = match mode {
            ProjectionMode::DetachedRecipe { block } => block,
            _ => self.source.skeleton.entry,
        };
        let mut blocks = HashMap::from([(source_entry, graph.skeleton.entry)]);
        for (source_block, _) in &self.source.skeleton.blocks {
            if source_block != source_entry && selection.blocks.contains(&source_block) {
                blocks.insert(source_block, graph.skeleton.create_block());
            }
        }

        let mut nodes = HashMap::new();
        for (source_id, node) in &self.source.nodes {
            if let ENode::FuncParam { index } = node {
                let target = graph.add_func_param(*index, self.source.types[&source_id].clone());
                nodes.insert(source_id, target);
            }
        }
        if !matches!(mode, ProjectionMode::DetachedRecipe { .. }) {
            self.clone_live_block_params(&selection.blocks, &mut graph, &blocks, &mut nodes);
        }
        for site in &selection.effects {
            if let Some(result) = self.effect_at(*site)?.result {
                let target = graph.alloc_side_effect_result(self.source.types[&result].clone());
                nodes.insert(result, target);
            }
        }
        Ok(ProjectionShell { graph, blocks, nodes })
    }

    fn clone_live_block_params(
        &self,
        projected_blocks: &HashSet<BlockId>,
        graph: &mut EGraph<Semantic>,
        blocks: &HashMap<BlockId, BlockId>,
        nodes: &mut HashMap<NodeId, NodeId>,
    ) {
        // The skeleton parameter list is authoritative. CFG cleanup leaves
        // eliminated BlockParam definitions in the sea, so iterating every
        // historical BlockParam node would resurrect removed parameters.
        for (source_block, source_body) in &self.source.skeleton.blocks {
            if !projected_blocks.contains(&source_block) {
                continue;
            }
            let target_block = blocks[&source_block];
            for (index, source_param) in source_body.params.iter().copied().enumerate() {
                let target =
                    graph.add_block_param(target_block, index, self.source.types[&source_param].clone());
                nodes.insert(source_param, target);
                graph.skeleton.blocks[target_block].params.push(target);
            }
        }
    }

    fn clone_effects(
        &self,
        selection: &ProjectionSelection,
        shell: &mut ProjectionShell,
    ) -> Result<HashSet<EffectToken>, String> {
        let mut effects = HashSet::new();
        for (source_block, body) in &self.source.skeleton.blocks {
            let Some(&target_block) = shell.blocks.get(&source_block) else {
                continue;
            };
            for (index, effect) in body.side_effects.iter().enumerate() {
                if !selection.effects.contains(&SideEffectSite {
                    block: source_block,
                    index,
                }) {
                    continue;
                }
                let mut projected = effect.clone();
                for node in projected.referenced_node_slots() {
                    *node = shell.nodes[node];
                }
                projected.result = effect.result.map(|result| shell.nodes[&result]);
                if let Some((input, output)) = projected.effects {
                    effects.extend([input, output]);
                }
                shell.graph.skeleton.blocks[target_block].side_effects.push(projected);
            }
        }
        Ok(effects)
    }

    fn project_terminators(
        &self,
        mode: ProjectionMode,
        projected_blocks: &HashSet<BlockId>,
        shell: &mut ProjectionShell,
    ) -> Result<(), String> {
        for source_block in projected_blocks {
            let target_block = shell.blocks[source_block];
            let is_recipe_exit = matches!(
                mode,
                ProjectionMode::EntryRecipe { .. }
                    if *source_block == self.source.skeleton.entry
            ) || matches!(
                mode,
                ProjectionMode::DetachedRecipe { block }
                    if *source_block == block
            ) || matches!(
                mode,
                ProjectionMode::StructuredPrefix { continuation, .. }
                    if *source_block == continuation
            );
            shell.graph.skeleton.blocks[target_block].term = if is_recipe_exit {
                SkeletonTerminator::Return(None)
            } else {
                remap_terminator(
                    &self.source.skeleton.blocks[*source_block].term,
                    &shell.nodes,
                    &shell.blocks,
                )?
            };
        }
        Ok(())
    }

    fn projected_blocks(&self, mode: ProjectionMode) -> Result<HashSet<BlockId>, String> {
        match mode {
            ProjectionMode::Complete => Ok(self.source.skeleton.blocks.keys().collect()),
            ProjectionMode::EntryRecipe { .. } => Ok(HashSet::from([self.source.skeleton.entry])),
            ProjectionMode::DetachedRecipe { block } => Ok(HashSet::from([block])),
            ProjectionMode::StructuredPrefix { continuation, .. } => {
                self.structured_prefix_blocks(continuation)
            }
        }
    }

    fn structured_prefix_blocks(&self, continuation: BlockId) -> Result<HashSet<BlockId>, String> {
        if !self.source.skeleton.blocks.contains_key(continuation) {
            return Err("structured prefix continuation is absent".into());
        }
        let mut blocks = HashSet::new();
        let mut pending = vec![self.source.skeleton.entry];
        while let Some(block) = pending.pop() {
            if !blocks.insert(block) || block == continuation {
                continue;
            }
            pending.extend(terminator_targets(&self.source.skeleton.blocks[block].term));
        }
        if !blocks.contains(&continuation) {
            return Err("structured prefix continuation is unreachable".into());
        }

        let mut reaches_continuation = HashSet::from([continuation]);
        loop {
            let mut changed = false;
            for block in blocks.iter().copied().collect::<Vec<_>>() {
                if reaches_continuation.contains(&block) {
                    continue;
                }
                if terminator_targets(&self.source.skeleton.blocks[block].term)
                    .iter()
                    .any(|target| reaches_continuation.contains(target))
                {
                    reaches_continuation.insert(block);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        if reaches_continuation != blocks {
            return Err("structured prefix contains a path that does not reach its continuation".into());
        }
        Ok(blocks)
    }

    fn allowed_effects(&self, mode: ProjectionMode, blocks: &HashSet<BlockId>) -> HashSet<SideEffectSite> {
        let limit = match mode {
            ProjectionMode::EntryRecipe { effect_limit } => effect_limit,
            ProjectionMode::StructuredPrefix { effect_limit, .. } => Some(effect_limit),
            ProjectionMode::Complete | ProjectionMode::DetachedRecipe { .. } => None,
        };
        let boundary_block = match mode {
            ProjectionMode::StructuredPrefix { continuation, .. } => Some(continuation),
            ProjectionMode::EntryRecipe { .. } => Some(self.source.skeleton.entry),
            ProjectionMode::DetachedRecipe { block } => Some(block),
            ProjectionMode::Complete => None,
        };
        blocks
            .iter()
            .flat_map(|block| {
                let count = if Some(*block) == boundary_block {
                    limit.unwrap_or(self.source.skeleton.blocks[*block].side_effects.len())
                } else {
                    self.source.skeleton.blocks[*block].side_effects.len()
                };
                (0..count).map(move |index| SideEffectSite { block: *block, index })
            })
            .collect()
    }

    fn projected_terminator_values(&self, mode: ProjectionMode, blocks: &HashSet<BlockId>) -> Vec<NodeId> {
        blocks
            .iter()
            .filter(|block| {
                !matches!(
                    mode,
                    ProjectionMode::EntryRecipe { .. } | ProjectionMode::DetachedRecipe { .. }
                ) && !matches!(
                    mode,
                    ProjectionMode::StructuredPrefix { continuation, .. }
                        if **block == continuation
                )
            })
            .flat_map(|block| self.source.skeleton.blocks[*block].term.referenced_nodes())
            .collect()
    }

    fn project_control_headers(
        &self,
        blocks: &HashMap<BlockId, BlockId>,
    ) -> Result<LookupMap<BlockId, ControlHeader>, String> {
        let mut projected = LookupMap::new();
        for (header, control) in self.control_headers {
            let Some(&target_header) = blocks.get(header) else {
                continue;
            };
            let targets = control_header_targets(control);
            if targets.iter().any(|target| !blocks.contains_key(target)) {
                return Err("projected structured prefix cuts through a control region".into());
            }
            projected.insert(target_header, control.remap(&|block| blocks[&block]));
        }
        Ok(projected)
    }

    fn effect_at(&self, site: SideEffectSite) -> Result<&SideEffect<Semantic>, String> {
        self.source
            .skeleton
            .blocks
            .get(site.block)
            .and_then(|block| block.side_effects.get(site.index))
            .ok_or_else(|| format!("invalid graph-projection effect site {site:?}"))
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

fn terminator_targets(term: &SkeletonTerminator) -> Vec<BlockId> {
    match term {
        SkeletonTerminator::Branch { target, .. } => vec![*target],
        SkeletonTerminator::CondBranch {
            then_target,
            else_target,
            ..
        } => vec![*then_target, *else_target],
        SkeletonTerminator::Return(_) | SkeletonTerminator::Unreachable => Vec::new(),
    }
}

fn control_header_targets(control: &ControlHeader) -> Vec<BlockId> {
    match control {
        ControlHeader::Loop {
            merge,
            continue_block,
        } => vec![*merge, *continue_block],
        ControlHeader::Selection { merge } => vec![*merge],
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
