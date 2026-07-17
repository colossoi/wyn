//! EGIR phase markers and phase-specific IR behavior.
//!
//! The phase-agnostic graph substrate lives in [`super::ir`]. This module
//! keeps the concrete phase types and their `EgirPhase` implementations so the
//! low-level IR does not need to know which phases the compiler defines.

use polytype::Type;
use slotmap::SlotMap;

use crate::ast::TypeName;
use crate::flow::BlockId;
use crate::op::OpTag;
use crate::ssa::types::ConstantValue;
use crate::LookupMap;

#[cfg(test)]
use smallvec::SmallVec;

use super::soac::{filter, hist, screma};

pub use super::ir::{
    EffectOp, EffectToken, EgirPhase, GraphResource, Language, NodeId, RegionId, SegBody, SegLevel,
    SegResourceAccessKind, SideEffectIndex, SideEffectSite, SkeletonTerminator, Soac, SoacDestination,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct WynLanguage;

impl Language for WynLanguage {
    type Const = ConstantValue;
    type Ty = Type<TypeName>;
}

// Keep the existing semantic defaults at the compatibility boundary. The
// definitions in `ir` themselves have no knowledge of concrete EGIR phases.
pub type PureOp<R = super::program::SemanticResourceRef> = OpTag<R>;
pub type PureViewSource<R = super::program::SemanticResourceRef> = super::ir::PureViewSource<R>;
pub type NodeKey<R = super::program::SemanticResourceRef, Lang = WynLanguage> = super::ir::NodeKey<R, Lang>;
pub type ENode<R = super::program::SemanticResourceRef, Lang = WynLanguage> = super::ir::ENode<R, Lang>;
pub type SegExtent<R = super::program::SemanticResourceRef> = super::ir::SegExtent<R>;
pub type SegSpace<R = super::program::SemanticResourceRef> = super::ir::SegSpace<R>;
pub type SegResourceAccess<R = super::program::SemanticResourceRef> = super::ir::SegResourceAccess<R>;
pub type SideEffect<P = Semantic, Lang = WynLanguage> = super::ir::SideEffect<P, Lang>;
pub type SideEffectKind<P = Semantic, Lang = WynLanguage> = super::ir::SideEffectKind<P, Lang>;
pub type SkeletonBlock<P = Semantic, Lang = WynLanguage> = super::ir::SkeletonBlock<P, Lang>;
pub type Skeleton<P = Semantic, Lang = WynLanguage> = super::ir::Skeleton<P, Lang>;
pub type SoacInputType<Ty = Type<TypeName>> = super::ir::SoacInputType<Ty>;
pub type EGraph<P = Semantic, Lang = WynLanguage> = super::ir::EGraph<P, Lang>;

#[derive(Clone, Copy, Debug, Default)]
pub struct Raw<R = super::program::SemanticResourceRef>(std::marker::PhantomData<fn() -> R>);

#[derive(Clone, Copy, Debug, Default)]
pub struct Semantic<R = super::program::SemanticResourceRef>(std::marker::PhantomData<fn() -> R>);

#[derive(Clone, Copy, Debug, Default)]
pub struct Scheduled<R = super::program::SemanticResourceRef>(std::marker::PhantomData<fn() -> R>);

#[derive(Clone, Copy, Debug, Default)]
pub struct Physical;

impl<R: GraphResource> EgirPhase for Raw<R> {
    type Resource = R;
    type ResourceDecl = super::program::SemanticResourceDecl;
    type SoacId = ();
    type MapState = screma::RawState;
    type ReduceState = screma::RawState;
    type ScanState = screma::RawState;
    type CompositeState = screma::RawState;
    type FilterState = filter::RawState<R>;
    type HistState = hist::RawState;
}

impl<R: GraphResource> EgirPhase for Semantic<R> {
    type Resource = R;
    type ResourceDecl = super::program::SemanticResourceDecl;
    type SoacId = super::program::SemanticOpId;
    type MapState = screma::SemanticState<R>;
    type ReduceState = screma::SemanticState<R>;
    type ScanState = screma::SemanticState<R>;
    type CompositeState = screma::SemanticState<R>;
    type FilterState = filter::SemanticState<R>;
    type HistState = hist::SemanticState<R>;
}

impl<R: GraphResource> EgirPhase for Scheduled<R> {
    type Resource = R;
    type ResourceDecl = super::program::SemanticResourceDecl;
    type SoacId = super::program::SemanticOpId;
    type MapState = screma::ScheduledState<R>;
    type ReduceState = screma::ScheduledState<R>;
    type ScanState = screma::ScheduledState<R>;
    type CompositeState = screma::ScheduledState<R>;
    type FilterState = filter::ScheduledState<R>;
    type HistState = hist::ScheduledState<R>;
}

impl EgirPhase for Physical {
    type Resource = super::program::PhysicalResourceRef;
    type ResourceDecl = crate::interface::StorageBindingDecl;
    type SoacId = super::program::SemanticOpId;
    type MapState = screma::PhysicalMapState;
    type ReduceState = screma::PhysicalSerialState;
    type ScanState = screma::PhysicalSerialState;
    type CompositeState = screma::PhysicalSerialState;
    type FilterState = filter::PhysicalState;
    type HistState = hist::PhysicalState;
}

impl<P: EgirPhase> super::ir::EGraph<P, WynLanguage> {
    pub(crate) fn try_map_phase<Q, E>(
        self,
        mut map_soac: impl FnMut(BlockId, usize, P::SoacId, Soac<P>) -> Result<(Q::SoacId, Soac<Q>), E>,
    ) -> Result<(EGraph<Q>, LookupMap<BlockId, BlockId>), E>
    where
        Q: EgirPhase<Resource = P::Resource>,
    {
        let super::ir::EGraphParts {
            mut nodes,
            types,
            skeleton,
            node_spans,
        } = self.into_parts();
        let source_entry = skeleton.entry;
        let source_blocks = skeleton.blocks.into_iter().collect::<Vec<_>>();
        let mut blocks = SlotMap::with_key();
        let mut block_map = LookupMap::new();
        for (source, _) in &source_blocks {
            block_map.insert(
                *source,
                blocks.insert(super::ir::SkeletonBlock::<Q, WynLanguage>::new()),
            );
        }

        for node in nodes.values_mut() {
            if let super::ir::ENode::BlockParam { block, .. } = node {
                *block = block_map[block];
            }
        }

        for (source, block) in source_blocks {
            let side_effects = block
                .side_effects
                .into_iter()
                .enumerate()
                .map(|(index, effect)| {
                    let kind = match effect.kind {
                        super::ir::SideEffectKind::Effect(effect) => {
                            super::ir::SideEffectKind::Effect(effect)
                        }
                        super::ir::SideEffectKind::Soac(id, soac) => {
                            let (id, soac) = map_soac(source, index, id, soac)?;
                            super::ir::SideEffectKind::Soac(id, soac)
                        }
                    };
                    Ok(super::ir::SideEffect {
                        kind,
                        operand_nodes: effect.operand_nodes,
                        result: effect.result,
                        effects: effect.effects,
                        span: effect.span,
                    })
                })
                .collect::<Result<_, E>>()?;
            let term =
                block.term.try_map(|node| Ok::<_, E>(node), |target| Ok::<_, E>(block_map[&target]))?;
            blocks[block_map[&source]] = super::ir::SkeletonBlock {
                params: block.params,
                side_effects,
                term,
            };
        }

        Ok((
            super::ir::EGraph::<Q, WynLanguage>::from_parts(super::ir::EGraphParts {
                nodes,
                types,
                skeleton: super::ir::Skeleton {
                    entry: block_map[&source_entry],
                    blocks,
                },
                node_spans,
            }),
            block_map,
        ))
    }

    /// Rebuild a graph when both its compiler phase and resource identity
    /// change. Graph structure is mapped here; the caller owns the direct
    /// business-logic conversion for each SOAC.
    pub(crate) fn try_map_resources_and_phase<Q, E>(
        self,
        mut map_resource: impl FnMut(P::Resource) -> Result<Q::Resource, E>,
        mut map_soac: impl FnMut(
            P::SoacId,
            Soac<P>,
            &LookupMap<NodeId, NodeId>,
        ) -> Result<(Q::SoacId, Soac<Q>), E>,
    ) -> Result<(EGraph<Q>, LookupMap<NodeId, NodeId>, LookupMap<BlockId, BlockId>), E>
    where
        Q: EgirPhase,
    {
        let super::ir::EGraphParts {
            nodes,
            types,
            skeleton,
            node_spans,
        } = self.into_parts();
        let source_entry = skeleton.entry;
        let source_blocks = skeleton.blocks.into_iter().collect::<Vec<_>>();
        let mut blocks = SlotMap::with_key();
        let mut block_map = LookupMap::new();
        for (source, _) in &source_blocks {
            block_map.insert(
                *source,
                blocks.insert(super::ir::SkeletonBlock::<Q, WynLanguage>::new()),
            );
        }

        let source_nodes = nodes.into_iter().collect::<Vec<_>>();
        let mut nodes = SlotMap::with_key();
        let mut node_map = LookupMap::new();
        for (source, _) in &source_nodes {
            node_map.insert(
                *source,
                nodes.insert(super::ir::ENode::<Q::Resource, WynLanguage>::Constant(
                    ConstantValue::Bool(false),
                )),
            );
        }
        for (source, node) in source_nodes {
            let node = match node {
                super::ir::ENode::Pure { op, operands } => super::ir::ENode::Pure {
                    op: op.try_map_resource(&mut map_resource)?,
                    operands: operands.into_iter().map(|node| node_map[&node]).collect(),
                },
                super::ir::ENode::Union { left, right } => super::ir::ENode::Union {
                    left: node_map[&left],
                    right: node_map[&right],
                },
                super::ir::ENode::FuncParam { index } => super::ir::ENode::FuncParam { index },
                super::ir::ENode::BlockParam { block, index } => super::ir::ENode::BlockParam {
                    block: block_map[&block],
                    index,
                },
                super::ir::ENode::Constant(value) => super::ir::ENode::Constant(value),
                super::ir::ENode::SideEffectResult => super::ir::ENode::SideEffectResult,
            };
            nodes[node_map[&source]] = node;
        }

        for (source, block) in source_blocks {
            let side_effects = block
                .side_effects
                .into_iter()
                .map(|effect| {
                    let kind = match effect.kind {
                        super::ir::SideEffectKind::Effect(effect) => {
                            super::ir::SideEffectKind::Effect(effect.try_map_resource(&mut map_resource)?)
                        }
                        super::ir::SideEffectKind::Soac(id, soac) => {
                            let (id, soac) = map_soac(id, soac, &node_map)?;
                            super::ir::SideEffectKind::Soac(id, soac)
                        }
                    };
                    Ok(super::ir::SideEffect::<Q, WynLanguage> {
                        kind,
                        operand_nodes: effect
                            .operand_nodes
                            .into_iter()
                            .map(|node| node_map[&node])
                            .collect(),
                        result: effect.result.map(|node| node_map[&node]),
                        effects: effect.effects,
                        span: effect.span,
                    })
                })
                .collect::<Result<Vec<_>, E>>()?;
            let term = block.term.try_map(
                |node| Ok::<_, E>(node_map[&node]),
                |target| Ok::<_, E>(block_map[&target]),
            )?;
            blocks[block_map[&source]] = super::ir::SkeletonBlock {
                params: block.params.into_iter().map(|node| node_map[&node]).collect(),
                side_effects,
                term,
            };
        }

        let graph = super::ir::EGraph::<Q, WynLanguage>::from_parts(super::ir::EGraphParts {
            nodes,
            types: types.into_iter().map(|(node, ty)| (node_map[&node], ty)).collect(),
            skeleton: super::ir::Skeleton {
                entry: block_map[&source_entry],
                blocks,
            },
            node_spans: node_spans.into_iter().map(|(node, span)| (node_map[&node], span)).collect(),
        });
        Ok((graph, node_map, block_map))
    }
}

impl From<crate::tlc::SoacDestination> for SoacDestination {
    fn from(destination: crate::tlc::SoacDestination) -> Self {
        match destination {
            crate::tlc::SoacDestination::Fresh => Self::Fresh,
            crate::tlc::SoacDestination::UniqueInput => Self::UniqueInput,
        }
    }
}

impl<P: EgirPhase> super::ir::Soac<P> {
    fn for_each_body_type_mut(&mut self, visit: &mut impl FnMut(&mut Type<TypeName>)) {
        match self {
            Self::Screma(op) => op.for_each_type_mut(visit),
            Self::Filter(op) => op.body.for_each_type_mut(visit),
            Self::Hist(op) => op.body.for_each_type_mut(visit),
        }
    }
}

impl<R: GraphResource> super::ir::Soac<Raw<R>> {
    pub(crate) fn for_each_type_mut(&mut self, mut visit: impl FnMut(&mut Type<TypeName>)) {
        self.for_each_body_type_mut(&mut visit);
        if let Self::Filter(op) = self {
            op.state.for_each_type_mut(&mut visit);
        }
    }
}

impl super::ir::Soac<Physical> {
    pub(crate) fn for_each_type_mut(&mut self, mut visit: impl FnMut(&mut Type<TypeName>)) {
        self.for_each_body_type_mut(&mut visit);
        if let Self::Filter(op) = self {
            op.state.for_each_type_mut(&mut visit);
        }
    }
}

impl<R: GraphResource> super::ir::Soac<Semantic<R>> {
    pub fn capture_nodes(&self) -> impl Iterator<Item = NodeId> {
        let nodes = match self {
            Self::Screma(op) => op.capture_nodes(),
            Self::Filter(op) => op.capture_nodes(),
            Self::Hist(op) => op.capture_nodes(),
        };
        nodes.into_iter()
    }

    /// Concrete iteration space seen by scheduling, independent of SOAC
    /// family. Serial Scremas and histograms have no dispatched space.
    pub(crate) fn scheduling_space(&self) -> Option<&SegSpace<R>> {
        match self {
            Self::Screma(op) => match op.semantic_state() {
                screma::SemanticState::Serial => None,
                screma::SemanticState::Segmented { space, .. } => Some(space),
            },
            Self::Filter(op) => Some(&op.state.space),
            Self::Hist(op) => match &op.state {
                hist::SemanticState::Serial => None,
                hist::SemanticState::Segmented(space) => Some(space),
            },
        }
    }

    fn referenced_nodes(&self) -> Vec<NodeId> {
        match self {
            Self::Screma(op) => op.referenced_nodes(),
            Self::Filter(op) => op.referenced_nodes(),
            Self::Hist(op) => op.referenced_nodes(),
        }
    }

    fn referenced_node_slots(&mut self) -> Vec<&mut NodeId> {
        match self {
            Self::Screma(op) => op.referenced_node_slots(),
            Self::Filter(op) => op.referenced_node_slots(),
            Self::Hist(op) => op.referenced_node_slots(),
        }
    }
}

impl<R: GraphResource> super::ir::SideEffect<Semantic<R>, WynLanguage> {
    /// Every graph value used by the effect, including SOAC captures,
    /// operator metadata, and semantic iteration-space extents.
    pub fn referenced_nodes(&self) -> impl Iterator<Item = NodeId> + '_ {
        let metadata = match &self.kind {
            SideEffectKind::Soac(_, soac) => soac.referenced_nodes(),
            SideEffectKind::Effect(_) => Vec::new(),
        };
        self.operand_nodes.iter().copied().chain(metadata)
    }

    pub fn referenced_node_slots(&mut self) -> Vec<&mut NodeId> {
        let Self {
            kind, operand_nodes, ..
        } = self;
        let mut slots = operand_nodes.iter_mut().collect::<Vec<_>>();
        if let SideEffectKind::Soac(_, soac) = kind {
            slots.extend(soac.referenced_node_slots());
        }
        slots
    }
}

#[cfg(test)]
#[path = "types_tests.rs"]
mod types_tests;
