//! Fresh graph construction for physical-entry recipes.
//!
//! A projection never clones an `EGraph` wholesale. It copies the CFG, the
//! selected side effects, and the transitive value/effect producer closure,
//! allocating fresh block, node, and side-effect-result identities throughout.

use std::collections::{HashMap, HashSet};

use smallvec::SmallVec;

use crate::ssa::types::ControlHeader;
use crate::LookupMap;

use super::types::{EGraph, ENode, EffectToken, NodeId, SideEffect, SkeletonTerminator};
use crate::ssa::framework::BlockId;

/// Stable location of a source side effect selected by a kernel recipe.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EffectSite {
    pub block: BlockId,
    pub index: usize,
}

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
                (0..body.side_effects.len()).map(move |index| EffectSite { block, index })
            })
            .collect();
        self.selected_with_values(selected, extra_values)
    }

    pub fn selected(&self, roots: HashSet<EffectSite>) -> Result<GraphProjection, String> {
        self.selected_with_values(roots, Vec::new())
    }

    pub fn selected_with_values(
        &self,
        roots: HashSet<EffectSite>,
        extra_values: Vec<NodeId>,
    ) -> Result<GraphProjection, String> {
        let producers = self.effect_producers();
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

        let mut copier = ValueCopier {
            source: self.source,
            target: &mut graph,
            nodes: &mut nodes,
        };
        for value in projected_values {
            copier.copy(value)?;
        }

        let mut effects = HashSet::new();
        for (source_block, body) in &self.source.skeleton.blocks {
            let target_block = blocks[&source_block];
            for (index, effect) in body.side_effects.iter().enumerate() {
                if !selected.contains(&EffectSite {
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
        let control_headers = self
            .control_headers
            .iter()
            .map(|(block, header)| {
                let target = blocks[block];
                (target, header.remap(&|source| blocks[&source]))
            })
            .collect();
        graph.verify_hash_cons()?;
        Ok(GraphProjection {
            graph,
            control_headers,
            nodes,
            blocks,
            effects,
        })
    }

    fn effect_at(&self, site: EffectSite) -> Result<&SideEffect, String> {
        self.source
            .skeleton
            .blocks
            .get(site.block)
            .and_then(|block| block.side_effects.get(site.index))
            .ok_or_else(|| format!("invalid graph-projection effect site {site:?}"))
    }

    fn effect_producers(&self) -> HashMap<NodeId, EffectSite> {
        self.source
            .skeleton
            .blocks
            .iter()
            .flat_map(|(block, body)| {
                body.side_effects.iter().enumerate().filter_map(move |(index, effect)| {
                    effect.result.map(|result| (result, EffectSite { block, index }))
                })
            })
            .collect()
    }

    fn terminator_values(&self) -> Vec<NodeId> {
        self.source
            .skeleton
            .blocks
            .iter()
            .flat_map(|(_, block)| match &block.term {
                SkeletonTerminator::Return(value) => value.iter().copied().collect(),
                SkeletonTerminator::Branch { args, .. } => args.clone(),
                SkeletonTerminator::CondBranch {
                    cond,
                    then_args,
                    else_args,
                    ..
                } => std::iter::once(*cond)
                    .chain(then_args.iter().copied())
                    .chain(else_args.iter().copied())
                    .collect(),
                SkeletonTerminator::Unreachable => Vec::new(),
            })
            .collect()
    }

    fn close_producers(
        &self,
        selected: &mut HashSet<EffectSite>,
        values: &mut Vec<NodeId>,
        producers: &HashMap<NodeId, EffectSite>,
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
                        .get(&value)
                        .copied()
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

struct ValueCopier<'a, 'b> {
    source: &'a EGraph,
    target: &'b mut EGraph,
    nodes: &'b mut HashMap<NodeId, NodeId>,
}

impl ValueCopier<'_, '_> {
    fn copy(&mut self, source_id: NodeId) -> Result<NodeId, String> {
        if let Some(target) = self.nodes.get(&source_id) {
            return Ok(*target);
        }
        let ty = self
            .source
            .types
            .get(&source_id)
            .cloned()
            .ok_or_else(|| format!("graph projection node {source_id:?} has no type"))?;
        let target = match self
            .source
            .nodes
            .get(source_id)
            .ok_or_else(|| format!("graph projection references missing node {source_id:?}"))?
            .clone()
        {
            ENode::Pure { op, operands } => {
                let operands = operands.into_iter().map(|operand| self.copy(operand)).collect::<Result<
                    SmallVec<[NodeId; 4]>,
                    _,
                >>(
                )?;
                self.target.intern_pure_with_span(
                    op,
                    operands,
                    ty,
                    self.source.node_spans.get(&source_id).copied(),
                )
            }
            ENode::Union { left, right } => {
                let left = self.copy(left)?;
                let right = self.copy(right)?;
                self.target.add_union(left, right)
            }
            ENode::Constant(value) => {
                // `EGraph::const_cache` is intentionally value-only. A graph
                // projection must preserve source node types even when equal
                // literal payloads occur at several integer widths, so copy
                // constants by identity instead of consulting that cache.
                let target = self.target.nodes.insert(ENode::Constant(value));
                self.target.types.insert(target, ty);
                target
            }
            ENode::SideEffectResult => {
                return self.nodes.get(&source_id).copied().ok_or_else(|| {
                    format!("graph projection omitted producer of side-effect result {source_id:?}")
                })
            }
            ENode::FuncParam { .. } | ENode::BlockParam { .. } => {
                return self.nodes.get(&source_id).copied().ok_or_else(|| {
                    format!("graph projection failed to preallocate parameter {source_id:?}")
                })
            }
        };
        self.nodes.insert(source_id, target);
        Ok(target)
    }
}

fn remap_terminator(
    term: &SkeletonTerminator,
    nodes: &HashMap<NodeId, NodeId>,
    blocks: &HashMap<BlockId, BlockId>,
) -> Result<SkeletonTerminator, String> {
    let node = |source: &NodeId| {
        nodes
            .get(source)
            .copied()
            .ok_or_else(|| format!("graph projection omitted terminator value {source:?}"))
    };
    Ok(match term {
        SkeletonTerminator::Return(value) => {
            SkeletonTerminator::Return(value.as_ref().map(node).transpose()?)
        }
        SkeletonTerminator::Branch { target, args } => SkeletonTerminator::Branch {
            target: blocks[target],
            args: args.iter().map(node).collect::<Result<_, _>>()?,
        },
        SkeletonTerminator::CondBranch {
            cond,
            then_target,
            then_args,
            else_target,
            else_args,
        } => SkeletonTerminator::CondBranch {
            cond: node(cond)?,
            then_target: blocks[then_target],
            then_args: then_args.iter().map(node).collect::<Result<_, _>>()?,
            else_target: blocks[else_target],
            else_args: else_args.iter().map(node).collect::<Result<_, _>>()?,
        },
        SkeletonTerminator::Unreachable => SkeletonTerminator::Unreachable,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::TypeName;
    use crate::egir::program::SemanticOpId;
    use crate::egir::types::{EffectToken, SideEffectKind};
    use crate::ssa::types::{ConstantValue, InstKind, ValueRef};
    use polytype::Type;
    use smallvec::smallvec;

    fn u32_ty() -> Type<TypeName> {
        Type::Constructed(TypeName::UInt(32), vec![])
    }

    #[test]
    fn selected_projection_remaps_cfg_aliases_and_value_producers() {
        let mut graph = EGraph::new();
        let entry = graph.skeleton.entry;
        let body = graph.skeleton.create_block();
        let exit = graph.skeleton.create_block();
        let cond = graph.intern_constant(
            ConstantValue::Bool(true),
            Type::Constructed(TypeName::Bool, vec![]),
        );
        let place = graph.intern_constant(ConstantValue::U32(0), u32_ty());
        let produced = graph.alloc_side_effect_result(u32_ty());
        graph.skeleton.blocks[entry].side_effects.push(SideEffect {
            semantic_id: Some(SemanticOpId(0)),
            kind: SideEffectKind::Inst(InstKind::Load {
                place: Default::default(),
            }),
            operand_nodes: smallvec![place],
            result: Some(produced),
            effects: Some((EffectToken(0), EffectToken(1))),
            span: None,
        });
        let unrelated = graph.alloc_side_effect_result(u32_ty());
        graph.skeleton.blocks[entry].side_effects.push(SideEffect {
            semantic_id: Some(SemanticOpId(1)),
            kind: SideEffectKind::Inst(InstKind::Load {
                place: Default::default(),
            }),
            operand_nodes: smallvec![place],
            result: Some(unrelated),
            effects: Some((EffectToken(1), EffectToken(2))),
            span: None,
        });
        let body_param = graph.add_block_param(body, 0, u32_ty());
        graph.skeleton.blocks[body].params.push(body_param);
        graph.skeleton.blocks[body].side_effects.push(SideEffect {
            semantic_id: Some(SemanticOpId(2)),
            kind: SideEffectKind::Inst(InstKind::Store {
                place: Default::default(),
                value: ValueRef::Ssa(Default::default()),
            }),
            operand_nodes: smallvec![place, body_param],
            result: None,
            effects: Some((EffectToken(2), EffectToken(3))),
            span: None,
        });
        graph.skeleton.blocks[entry].term = SkeletonTerminator::CondBranch {
            cond,
            then_target: body,
            then_args: vec![produced],
            else_target: exit,
            else_args: vec![],
        };
        graph.skeleton.blocks[body].term = SkeletonTerminator::Branch {
            target: exit,
            args: vec![],
        };
        graph.skeleton.blocks[exit].term = SkeletonTerminator::Return(None);
        let headers = LookupMap::from([(entry, ControlHeader::Selection { merge: exit })]);
        let aliases = LookupMap::from([(produced, place), (unrelated, place)]);

        let projected = GraphProjector::new(&graph, &headers)
            .selected(HashSet::from([EffectSite {
                block: body,
                index: 0,
            }]))
            .expect("projection");
        assert_eq!(
            projected
                .graph
                .skeleton
                .blocks
                .iter()
                .map(|(_, block)| block.side_effects.len())
                .sum::<usize>(),
            2,
            "selected store and its load producer survive; unrelated load does not"
        );
        assert!(projected.node(produced).is_some());
        assert!(projected.node(unrelated).is_none());
        assert_eq!(projected.remap_aliases(&aliases).len(), 1);
        assert!(matches!(
            projected.control_headers.get(&projected.block(entry).unwrap()),
            Some(ControlHeader::Selection { merge }) if *merge == projected.block(exit).unwrap()
        ));
    }

    #[test]
    fn complete_projection_remaps_loop_headers_and_parameters() {
        let mut graph = EGraph::new();
        let entry = graph.skeleton.entry;
        let header = graph.skeleton.create_block();
        let exit = graph.skeleton.create_block();
        let zero = graph.intern_constant(ConstantValue::U32(0), u32_ty());
        graph.skeleton.blocks[entry].term = SkeletonTerminator::Branch {
            target: header,
            args: vec![zero],
        };
        let index = graph.add_block_param(header, 0, u32_ty());
        graph.skeleton.blocks[header].params.push(index);
        graph.skeleton.blocks[header].term = SkeletonTerminator::Branch {
            target: exit,
            args: vec![],
        };
        graph.skeleton.blocks[exit].term = SkeletonTerminator::Return(None);
        let headers = LookupMap::from([(
            header,
            ControlHeader::Loop {
                merge: exit,
                continue_block: header,
            },
        )]);
        let projected = GraphProjector::new(&graph, &headers).all().expect("complete projection");
        assert_eq!(projected.graph.skeleton.blocks.len(), 3);
        assert_eq!(
            projected.graph.skeleton.blocks[projected.block(header).unwrap()].params.len(),
            1
        );
        assert!(projected.node(zero).is_some());
        assert!(matches!(
            projected.control_headers.get(&projected.block(header).unwrap()),
            Some(ControlHeader::Loop { merge, continue_block })
                if *merge == projected.block(exit).unwrap()
                    && *continue_block == projected.block(header).unwrap()
        ));
    }
}
