//! EGIR phase markers and phase-specific IR behavior.
//!
//! The phase-agnostic graph substrate lives in [`super::ir`]. This module
//! keeps the concrete family types and their `Family` implementations so the
//! low-level IR does not need to know which phases the compiler defines.

use polytype::Type;
use slotmap::SlotMap;

use crate::ast::TypeName;
use crate::flow::{BlockId, ControlHeader};
use crate::op::OpTag;
use crate::ssa::types::ConstantValue;
use crate::LookupMap;

#[cfg(test)]
use smallvec::SmallVec;

use super::soac::{filter, hist, screma};

pub use super::ir::{
    EffectOp, EffectToken, Family, GraphResource, Language, NodeId, RegionId, SegBody, SideEffectIndex,
    SideEffectSite, SkeletonTerminator, SoacDestination, SoacOwnership, SoacPlacement,
};
pub use crate::ResourceAccess;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct WynLanguage;

impl Language for WynLanguage {
    type Const = ConstantValue;
    type Ty = Type<TypeName>;
}

pub trait WynSoacPhase: Family<Soac = SoacEffect<Self>> + Sized {
    type SoacId: Clone + std::fmt::Debug;
    type ScremaResults: screma::PhaseResults;
    type ScremaState: Clone + std::fmt::Debug;
    type FilterState: Clone + std::fmt::Debug;
    type HistState: Clone + std::fmt::Debug;
}

/// A compiler SOAC together with its stable semantic identity.
#[derive(Clone, Debug)]
pub struct SoacEffect<P: WynSoacPhase>(pub P::SoacId, pub Soac<P>);

#[derive(Clone, Debug)]
pub enum Soac<P: WynSoacPhase> {
    Screma(screma::Op<P>),
    Filter(filter::Op<P>),
    Hist(hist::Op<P>),
}

impl<P: WynSoacPhase> Soac<P> {
    pub(crate) fn seg_bodies(&self) -> Vec<&SegBody> {
        match self {
            Self::Screma(op) => {
                let mut bodies = Vec::new();
                if let Some(body) = op.form.pre.seg_body() {
                    bodies.push(body);
                }
                bodies.extend(op.form.scans.iter().filter_map(|scan| scan.operator.seg_body()));
                bodies.extend(
                    op.form.reductions.iter().filter_map(|reduction| reduction.operator.seg_body()),
                );
                if let Some(body) = op.form.post.seg_body() {
                    bodies.push(body);
                }
                bodies
            }
            Self::Filter(op) => {
                op.body.map.seg_body().into_iter().chain(op.body.predicate.seg_body()).collect()
            }
            Self::Hist(op) => {
                let mut bodies = op.form.bucket.seg_body().into_iter().collect::<Vec<_>>();
                bodies.extend(
                    op.form.operations.iter().filter_map(|operation| match &operation.update {
                        hist::Update::OrderedOverwrite { .. } => None,
                        hist::Update::Reduce { operator, .. } => operator.seg_body(),
                    }),
                );
                bodies
            }
        }
    }

    pub(crate) fn seg_body_mut(&mut self, index: usize) -> Option<&mut SegBody> {
        match self {
            Self::Screma(op) => {
                let mut remaining = index;
                if let Some(body) = op.form.pre.seg_body_mut() {
                    if remaining == 0 {
                        return Some(body);
                    }
                    remaining -= 1;
                }
                for scan in &mut op.form.scans {
                    if let Some(body) = scan.operator.seg_body_mut() {
                        if remaining == 0 {
                            return Some(body);
                        }
                        remaining -= 1;
                    }
                }
                for reduction in &mut op.form.reductions {
                    if let Some(body) = reduction.operator.seg_body_mut() {
                        if remaining == 0 {
                            return Some(body);
                        }
                        remaining -= 1;
                    }
                }
                op.form.post.seg_body_mut().filter(|_| remaining == 0)
            }
            Self::Filter(op) => {
                let mut remaining = index;
                if let Some(body) = op.body.map.seg_body_mut() {
                    if remaining == 0 {
                        return Some(body);
                    }
                    remaining -= 1;
                }
                op.body.predicate.seg_body_mut().filter(|_| remaining == 0)
            }
            Self::Hist(op) => {
                let mut remaining = index;
                if let Some(body) = op.form.bucket.seg_body_mut() {
                    if remaining == 0 {
                        return Some(body);
                    }
                    remaining -= 1;
                }
                for operation in &mut op.form.operations {
                    if let hist::Update::Reduce { operator, .. } = &mut operation.update {
                        if let Some(body) = operator.seg_body_mut() {
                            if remaining == 0 {
                                return Some(body);
                            }
                            remaining -= 1;
                        }
                    }
                }
                None
            }
        }
    }
}

impl<P: WynSoacPhase> super::ir::SideEffectKind<P, WynLanguage> {
    pub fn soac_id(&self) -> Option<&P::SoacId> {
        match self {
            Self::Effect(_) => None,
            Self::Soac(SoacEffect(id, _)) => Some(id),
        }
    }
}

// Concrete aliases default to semantic EGIR while the definitions in `ir`
// remain independent of compiler-specific families.
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

/// If `ty` is a structure-of-arrays tuple, return its array component types.
pub(crate) fn as_soa_tuple(ty: &Type<TypeName>) -> Option<&[Type<TypeName>]> {
    let Type::Constructed(TypeName::Tuple(_), components) = ty else {
        return None;
    };
    if components.is_empty() {
        return None;
    }
    components
        .iter()
        .all(|component| {
            matches!(component, Type::Constructed(TypeName::Array, args) if args.len() == 4)
                || as_soa_tuple(component).is_some()
        })
        .then_some(components)
}

/// Derive the logical element represented by an array or SoA tuple type.
pub(crate) fn soac_element_type(array: &Type<TypeName>) -> Type<TypeName> {
    if as_soa_tuple(array).is_some() {
        let Type::Constructed(TypeName::Tuple(arity), components) = array else {
            unreachable!()
        };
        return Type::Constructed(
            TypeName::Tuple(*arity),
            components.iter().map(soac_element_type).collect(),
        );
    }
    crate::types::array_elem(array)
        .cloned()
        .unwrap_or_else(|| panic!("expected an array or SoA tuple, got {array:?}"))
}

impl super::ir::SoacInputType<Type<TypeName>> {
    pub(crate) fn element(&self) -> Type<TypeName> {
        soac_element_type(&self.array)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Raw<R = super::program::SemanticResourceRef>(std::marker::PhantomData<fn() -> R>);

#[derive(Clone, Copy, Debug, Default)]
pub struct Semantic<R = super::program::SemanticResourceRef>(std::marker::PhantomData<fn() -> R>);

#[derive(Clone, Copy, Debug, Default)]
pub struct Scheduled<R = super::program::SemanticResourceRef>(std::marker::PhantomData<fn() -> R>);

#[derive(Clone, Copy, Debug, Default)]
pub struct Physical;

impl<R: GraphResource> Family for Raw<R> {
    type Resource = R;
    type Soac = SoacEffect<Self>;
}

impl<R: GraphResource> WynSoacPhase for Raw<R> {
    type SoacId = ();
    type ScremaResults = Vec<screma::ResultState>;
    type ScremaState = screma::RawState;
    type FilterState = filter::RawState<R>;
    type HistState = hist::RawState;
}

impl<R: GraphResource> Family for Semantic<R> {
    type Resource = R;
    type Soac = SoacEffect<Self>;
}

impl<R: GraphResource> WynSoacPhase for Semantic<R> {
    type SoacId = super::program::SemanticOpId;
    type ScremaResults = Vec<screma::ResultState>;
    type ScremaState = screma::SemanticState<R>;
    type FilterState = filter::SemanticState<R>;
    type HistState = hist::SemanticState<R>;
}

impl<R: GraphResource> Family for Scheduled<R> {
    type Resource = R;
    type Soac = SoacEffect<Self>;
}

impl<R: GraphResource> WynSoacPhase for Scheduled<R> {
    type SoacId = super::program::SemanticOpId;
    type ScremaResults = Vec<screma::ResultState>;
    type ScremaState = screma::ScheduledState<R>;
    type FilterState = filter::ScheduledState<R>;
    type HistState = hist::ScheduledState<R>;
}

impl Family for Physical {
    type Resource = super::program::PhysicalResourceRef;
    type Soac = SoacEffect<Self>;
}

impl WynSoacPhase for Physical {
    type SoacId = super::program::SemanticOpId;
    type ScremaResults = Vec<screma::ResultState>;
    type ScremaState = screma::PhysicalState;
    type FilterState = filter::PhysicalState;
    type HistState = hist::PhysicalState;
}

fn remap_control_header(header: ControlHeader, blocks: &LookupMap<BlockId, BlockId>) -> ControlHeader {
    match header {
        ControlHeader::Loop {
            merge,
            continue_block,
        } => ControlHeader::Loop {
            merge: blocks[&merge],
            continue_block: blocks[&continue_block],
        },
        ControlHeader::Selection { merge } => ControlHeader::Selection {
            merge: blocks[&merge],
        },
    }
}

impl<P: Family> super::ir::EGraph<P, WynLanguage> {
    pub(crate) fn try_map_phase<Q, E>(
        self,
        mut map_soac: impl FnMut(BlockId, usize, P::SoacId, Soac<P>) -> Result<(Q::SoacId, Soac<Q>), E>,
    ) -> Result<(EGraph<Q>, LookupMap<BlockId, BlockId>), E>
    where
        P: WynSoacPhase,
        Q: WynSoacPhase<Resource = P::Resource>,
    {
        let super::ir::EGraphParts { mut nodes, skeleton } = self.into_parts();
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
            if let super::ir::ENode::BlockParam { block, .. } = &mut node.kind {
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
                        super::ir::SideEffectKind::Soac(SoacEffect(id, soac)) => {
                            let (id, soac) = map_soac(source, index, id, soac)?;
                            super::ir::SideEffectKind::Soac(SoacEffect(id, soac))
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
                control_header: block.control_header.map(|header| remap_control_header(header, &block_map)),
            };
        }

        Ok((
            super::ir::EGraph::<Q, WynLanguage>::from_parts(super::ir::EGraphParts {
                nodes,
                skeleton: super::ir::Skeleton {
                    entry: block_map[&source_entry],
                    blocks,
                },
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
        P: WynSoacPhase,
        Q: WynSoacPhase,
    {
        let super::ir::EGraphParts { nodes, skeleton } = self.into_parts();
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
        for (source, node) in &source_nodes {
            node_map.insert(
                *source,
                nodes.insert(super::ir::Node {
                    kind: super::ir::ENode::<Q::Resource, WynLanguage>::Constant(ConstantValue::Bool(
                        false,
                    )),
                    ty: node.ty.clone(),
                    span: node.span,
                    alias: None,
                }),
            );
        }
        for (source, node) in source_nodes {
            let kind = match node.kind {
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
            nodes[node_map[&source]] = super::ir::Node {
                kind,
                ty: node.ty,
                span: node.span,
                alias: node.alias.map(|alias| node_map[&alias]),
            };
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
                        super::ir::SideEffectKind::Soac(SoacEffect(id, soac)) => {
                            let (id, soac) = map_soac(id, soac, &node_map)?;
                            super::ir::SideEffectKind::Soac(SoacEffect(id, soac))
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
                control_header: block.control_header.map(|header| remap_control_header(header, &block_map)),
            };
        }

        let graph = super::ir::EGraph::<Q, WynLanguage>::from_parts(super::ir::EGraphParts {
            nodes,
            skeleton: super::ir::Skeleton {
                entry: block_map[&source_entry],
                blocks,
            },
        });
        Ok((graph, node_map, block_map))
    }
}

impl<P, ResourceDecl, Route> super::ir::Entry<P, ResourceDecl, Route, WynLanguage>
where
    P: WynSoacPhase,
    Route: super::ir::RemapBlockIds,
{
    /// Change an entry's SOAC phase while preserving its interface and
    /// remapping every block-bearing piece of entry metadata with the graph.
    pub(crate) fn try_map_phase<Q, E>(
        self,
        map_soac: impl FnMut(BlockId, usize, P::SoacId, Soac<P>) -> Result<(Q::SoacId, Soac<Q>), E>,
    ) -> Result<super::ir::Entry<Q, ResourceDecl, Route, WynLanguage>, E>
    where
        Q: WynSoacPhase<Resource = P::Resource>,
    {
        let super::ir::Entry {
            id,
            name,
            span,
            execution_model,
            inputs,
            parameter_inputs,
            mut outputs,
            resource_declarations,
            params,
            return_ty,
            graph,
        } = self;
        let (graph, blocks) = graph.try_map_phase(map_soac)?;
        for route in outputs.iter_mut().flat_map(|output| &mut output.routes) {
            route.remap_block_ids(&blocks);
        }
        Ok(super::ir::Entry {
            id,
            name,
            span,
            execution_model,
            inputs,
            parameter_inputs,
            outputs,
            resource_declarations,
            params,
            return_ty,
            graph,
        })
    }
}

impl From<SoacOwnership> for SoacDestination {
    fn from(ownership: SoacOwnership) -> Self {
        match ownership {
            SoacOwnership::Fresh => Self::fresh(),
            SoacOwnership::UniqueInput => Self::unique_input(),
        }
    }
}

impl<P: WynSoacPhase> Soac<P> {
    fn for_each_body_type_mut(&mut self, visit: &mut impl FnMut(&mut Type<TypeName>)) {
        match self {
            Self::Screma(op) => op.for_each_type_mut(visit),
            Self::Filter(op) => op.body.for_each_type_mut(visit),
            Self::Hist(op) => op.for_each_type_mut(visit),
        }
    }
}

impl<R: GraphResource> Soac<Raw<R>> {
    pub(crate) fn for_each_type_mut(&mut self, mut visit: impl FnMut(&mut Type<TypeName>)) {
        self.for_each_body_type_mut(&mut visit);
        if let Self::Filter(op) = self {
            op.state.for_each_type_mut(&mut visit);
        }
    }
}

impl Soac<Physical> {
    pub(crate) fn for_each_type_mut(&mut self, mut visit: impl FnMut(&mut Type<TypeName>)) {
        self.for_each_body_type_mut(&mut visit);
        if let Self::Filter(op) = self {
            op.state.for_each_type_mut(&mut visit);
        }
    }
}

impl<R: GraphResource> Soac<Semantic<R>> {
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
            Self::Hist(op) => op.referenced_nodes_with_state(),
        }
    }

    fn referenced_node_slots(&mut self) -> Vec<&mut NodeId> {
        match self {
            Self::Screma(op) => op.referenced_node_slots(),
            Self::Filter(op) => op.referenced_node_slots(),
            Self::Hist(op) => op.referenced_node_slots_with_state(),
        }
    }
}

impl<R: GraphResource> super::ir::SideEffect<Semantic<R>, WynLanguage> {
    /// Every graph value used by the effect, including SOAC captures,
    /// operator metadata, and semantic iteration-space extents.
    pub fn referenced_nodes(&self) -> impl Iterator<Item = NodeId> + '_ {
        let metadata = match &self.kind {
            SideEffectKind::Soac(SoacEffect(_, soac)) => soac.referenced_nodes(),
            SideEffectKind::Effect(_) => Vec::new(),
        };
        self.operand_nodes.iter().copied().chain(metadata)
    }

    pub fn referenced_node_slots(&mut self) -> Vec<&mut NodeId> {
        let Self {
            kind, operand_nodes, ..
        } = self;
        let mut slots = operand_nodes.iter_mut().collect::<Vec<_>>();
        if let SideEffectKind::Soac(SoacEffect(_, soac)) = kind {
            slots.extend(soac.referenced_node_slots());
        }
        slots
    }

    /// Select a segmented body using the same stable ordering as
    /// [`Soac::seg_bodies`].
    pub(crate) fn seg_body_mut(&mut self, index: usize) -> Option<&mut SegBody> {
        let SideEffectKind::Soac(SoacEffect(_, soac)) = &mut self.kind else {
            return None;
        };
        soac.seg_body_mut(index)
    }
}

#[cfg(test)]
#[path = "types_tests.rs"]
mod types_tests;
