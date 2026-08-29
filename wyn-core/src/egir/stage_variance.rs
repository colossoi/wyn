//! Stage-dependence analysis for EGIR values.
//!
//! Uniformity, loop dependence, and provenance are independent facts. A
//! read-only storage value, for example, may be stage-uniform or
//! invocation-varying depending on its index; a loop induction value may be
//! stage-uniform while still changing between iterations.
//!
//! Repeated-region parameters are analyzed per [`SegBody`] use because the
//! same region may receive captures with different dependence at different
//! call sites. This module only computes facts. Relocation legality and
//! profitability remain scheduling decisions.

use smallvec::SmallVec;

use crate::builtins::catalog;
use crate::flow::{BlockId, Terminator};
use crate::interface::{EntryInputKind, IoDecoration};
use crate::{FunctionId, LookupMap, LookupSet};

use super::ir::{
    CallEffects, CallSiteId, EffectOp, Family, FlowValueId, OperandRef, ParameterId, Parameters,
    SideEffectKind,
};
use super::loop_analysis::LoopAnalysis;
use super::program::Entry;
use super::reify::Segmented;
use super::types::{EGraph, PureOp, PureViewSource, SegBody, ValueId, ValueKind};

#[cfg(test)]
#[path = "stage_variance_tests.rs"]
mod stage_variance_tests;

/// The finest invocation scope across which a value is identical.
///
/// Variants are ordered from most uniform to least uniform. `Constant` means
/// independent of runtime stage inputs, not merely constant within one loop
/// iteration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Uniformity {
    Constant,
    StageUniform,
    WorkgroupUniform,
    InvocationVarying,
}

impl Uniformity {
    fn join(self, other: Self) -> Self {
        self.max(other)
    }
}

/// Runtime source categories retained independently from uniformity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum DependenceSource {
    Uniform,
    PushConstant,
    StageInput,
    Storage,
    Texture,
    Sampler,
    StorageImage,
    DispatchBuiltin,
    InvocationBuiltin,
    RepeatedRegionInput,
    WorkgroupMemory,
    Output,
    SideEffect,
    Unknown,
}

/// Dependence facts for one value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StageDependence {
    uniformity: Uniformity,
    loop_dependencies: LookupSet<BlockId>,
    sources: LookupSet<DependenceSource>,
}

impl StageDependence {
    pub(crate) fn constant() -> Self {
        Self {
            uniformity: Uniformity::Constant,
            loop_dependencies: LookupSet::new(),
            sources: LookupSet::new(),
        }
    }

    pub(crate) fn from_source(uniformity: Uniformity, source: DependenceSource) -> Self {
        Self {
            uniformity,
            loop_dependencies: LookupSet::new(),
            sources: [source].into_iter().collect(),
        }
    }

    pub(crate) fn uniformity(&self) -> Uniformity {
        self.uniformity
    }

    /// Whether the value is identical across every invocation in the stage.
    pub(crate) fn is_stage_invariant(&self) -> bool {
        self.uniformity <= Uniformity::StageUniform
    }

    pub(crate) fn is_compile_time_constant(&self) -> bool {
        self.uniformity == Uniformity::Constant
            && self.loop_dependencies.is_empty()
            && self.sources.is_empty()
    }

    pub(crate) fn is_loop_invariant(&self, header: BlockId) -> bool {
        !self.loop_dependencies.contains(&header)
    }

    pub(crate) fn loop_dependencies(&self) -> &LookupSet<BlockId> {
        &self.loop_dependencies
    }

    pub(crate) fn sources(&self) -> &LookupSet<DependenceSource> {
        &self.sources
    }

    pub(crate) fn depends_on(&self, source: DependenceSource) -> bool {
        self.sources.contains(&source)
    }

    fn with_loop_dependencies(mut self, dependencies: &LookupSet<BlockId>) -> Self {
        self.loop_dependencies.extend(dependencies.iter().copied());
        self
    }

    fn join(mut self, other: &Self) -> Self {
        self.uniformity = self.uniformity.join(other.uniformity);
        self.loop_dependencies.extend(other.loop_dependencies.iter().copied());
        self.sources.extend(other.sources.iter().copied());
        self
    }
}

/// Dependence of every argument at one pure user-call node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CallArgumentDependences {
    pub(crate) callee: FunctionId,
    pub(crate) arguments: SmallVec<[(ValueId, StageDependence); 4]>,
}

impl CallArgumentDependences {
    /// Whether the call combines stage-invariant and invocation-varying data.
    pub(crate) fn has_mixed_stage_variance(&self) -> bool {
        self.arguments.iter().any(|(_, facts)| facts.is_stage_invariant())
            && self.arguments.iter().any(|(_, facts)| !facts.is_stage_invariant())
    }
}

/// Context-specific stage-dependence facts for one EGIR graph.
#[derive(Clone, Debug)]
pub(crate) struct StageDependenceAnalysis {
    values: LookupMap<ValueId, StageDependence>,
}

#[derive(Clone, Copy)]
struct IncomingValue {
    source: BlockId,
    condition: Option<ValueId>,
    value: ValueId,
}

pub(crate) fn bind_parameter_dependences<R, Ty>(
    parameters: &Parameters<R, Ty>,
    dependences: &[StageDependence],
) -> LookupMap<ParameterId, StageDependence> {
    parameters.ids().zip(dependences.iter().cloned()).collect()
}

impl StageDependenceAnalysis {
    /// Analyze a graph using one dependence seed per function parameter.
    ///
    /// Missing seeds are conservatively invocation-varying. This keeps
    /// projected graphs and temporarily unused parameter placeholders safe to
    /// inspect.
    pub(crate) fn for_graph<P: Family>(
        graph: &EGraph<P>,
        parameter_dependences: &LookupMap<ParameterId, StageDependence>,
    ) -> Result<Self, String> {
        let (incoming_blocks, incoming_values) = collect_incoming(graph)?;
        let block_loop_dependencies = block_loop_dependencies(graph);
        let effect_blocks = graph
            .skeleton
            .blocks
            .iter()
            .flat_map(|(block, body)| {
                body.side_effects.iter().flat_map(move |effect| {
                    effect
                        .result()
                        .into_iter()
                        .flat_map(|result| result.values())
                        .map(move |result| (result, block))
                })
            })
            .collect::<LookupMap<_, _>>();
        let call_blocks = graph
            .skeleton
            .blocks
            .iter()
            .flat_map(|(block, body)| {
                body.side_effects.iter().filter_map(move |effect| match effect.kind() {
                    SideEffectKind::Effect(EffectOp::Call { site }) => Some((*site, block)),
                    _ => None,
                })
            })
            .collect::<LookupMap<_, _>>();
        let mut values = graph
            .nodes
            .iter()
            .map(|(node, definition)| {
                let dependence = match &definition.kind {
                    ValueKind::Constant(_) => StageDependence::constant(),
                    ValueKind::FuncParam { parameter } => {
                        parameter_dependences.get(parameter).cloned().unwrap_or_else(unknown_dependence)
                    }
                    ValueKind::BlockParam { .. }
                    | ValueKind::Pure { .. }
                    | ValueKind::Union { .. }
                    | ValueKind::CallResult { .. }
                    | ValueKind::PlaceLength { .. }
                    | ValueKind::PlaceView { .. } => StageDependence::constant(),
                    ValueKind::SideEffectResult => {
                        side_effect_dependence(node, &effect_blocks, &block_loop_dependencies)
                    }
                };
                (node, dependence)
            })
            .collect::<LookupMap<_, _>>();
        let mut block_controls = graph
            .skeleton
            .blocks
            .keys()
            .map(|block| (block, StageDependence::constant()))
            .collect::<LookupMap<_, _>>();

        // `Constant` plus empty dependence sets is the optimistic lattice
        // bottom. Facts only accumulate, so CFG cycles converge.
        loop {
            let mut changed = false;

            for block in graph.skeleton.blocks.keys() {
                if block == graph.skeleton.entry {
                    continue;
                }
                let next = match incoming_blocks.get(&block) {
                    Some(edges) if !edges.is_empty() => {
                        edges.iter().fold(StageDependence::constant(), |dependence, (source, condition)| {
                            dependence.join(&edge_dependence(&block_controls, &values, *source, *condition))
                        })
                    }
                    _ => unknown_dependence(),
                };
                changed |= accumulate(&mut block_controls, block, next);
            }

            for (node, definition) in &graph.nodes {
                let next = match &definition.kind {
                    ValueKind::Constant(_) => StageDependence::constant(),
                    ValueKind::FuncParam { parameter } => {
                        parameter_dependences.get(parameter).cloned().unwrap_or_else(unknown_dependence)
                    }
                    ValueKind::SideEffectResult => {
                        side_effect_dependence(node, &effect_blocks, &block_loop_dependencies)
                    }
                    ValueKind::BlockParam { block, .. } => {
                        let incoming = match incoming_values.get(&node) {
                            Some(incoming) if !incoming.is_empty() => {
                                incoming.iter().fold(StageDependence::constant(), |dependence, incoming| {
                                    dependence.join(&value_dependence(&values, incoming.value)).join(
                                        &edge_dependence(
                                            &block_controls,
                                            &values,
                                            incoming.source,
                                            incoming.condition,
                                        ),
                                    )
                                })
                            }
                            _ => unknown_dependence(),
                        };
                        incoming.with_loop_dependencies(
                            block_loop_dependencies.get(block).unwrap_or(&LookupSet::new()),
                        )
                    }
                    ValueKind::Pure { op, operands } => pure_dependence(op, operands, &values),
                    ValueKind::CallResult { call, .. } => call_result_dependence(
                        graph,
                        *call,
                        &values,
                        &call_blocks,
                        &block_loop_dependencies,
                    ),
                    ValueKind::PlaceLength { place } | ValueKind::PlaceView { place } => graph
                        .place_value_dependencies(*place)
                        .into_iter()
                        .fold(StageDependence::constant(), |dependence, value| {
                            dependence.join(&value_dependence(&values, value))
                        }),
                    ValueKind::Union { left, right } => {
                        value_dependence(&values, *left).join(&value_dependence(&values, *right))
                    }
                };
                changed |= accumulate(&mut values, node, next);
            }

            if !changed {
                break;
            }
        }

        Ok(Self { values })
    }

    /// Analyze an entry using its declared interface as parameter seeds.
    pub(crate) fn for_entry<P: Family, ResourceDecl, Route>(
        entry: &Entry<P, ResourceDecl, Route>,
    ) -> Result<Self, String> {
        Self::for_entry_graph(entry, &entry.graph)
    }

    /// Analyze a graph projected from `entry`. Function-parameter indices are
    /// preserved by graph projection, so the source entry remains the
    /// authority for their stage dependence.
    pub(crate) fn for_entry_graph<EntryP: Family, P: Family, ResourceDecl, Route>(
        entry: &Entry<EntryP, ResourceDecl, Route>,
        graph: &EGraph<P>,
    ) -> Result<Self, String> {
        let dependences = bind_parameter_dependences(&entry.params, &entry_parameter_dependences(entry));
        Self::for_graph(graph, &dependences)
    }

    /// Analyze one use of a repeated region.
    ///
    /// The region ABI places lane/element parameters first and captured values
    /// last. Leading parameters depend on the repeated-region invocation;
    /// trailing parameters inherit the corresponding enclosing capture facts.
    pub(crate) fn for_seg_body(
        program: &Segmented,
        enclosing: &Self,
        body: &SegBody,
    ) -> Result<Self, String> {
        let region = program
            .region(body.region)
            .ok_or_else(|| format!("stage-dependence analysis cannot resolve region {}", body.region))?;
        let parameter_dependences = seg_body_parameter_dependences(region.params.len(), enclosing, body)?;
        let parameter_dependences = bind_parameter_dependences(&region.params, &parameter_dependences);
        Self::for_graph(&region.graph, &parameter_dependences)
    }

    pub(crate) fn seg_body_parameter_dependences(
        parameter_count: usize,
        enclosing: &Self,
        body: &SegBody,
    ) -> Result<Vec<StageDependence>, String> {
        seg_body_parameter_dependences(parameter_count, enclosing, body)
    }

    pub(crate) fn dependence(&self, node: ValueId) -> StageDependence {
        self.values.get(&node).cloned().unwrap_or_else(unknown_dependence)
    }

    pub(crate) fn call_arguments<P: Family>(
        &self,
        graph: &EGraph<P>,
        node: ValueId,
    ) -> Option<CallArgumentDependences> {
        let ValueKind::CallResult { call, .. } = &graph.nodes.get(node)?.kind else {
            return None;
        };
        let call = graph.call(*call);
        Some(CallArgumentDependences {
            callee: call.callee(),
            arguments: call
                .arguments()
                .filter_map(|argument| argument.value())
                .map(|argument| (argument, self.dependence(argument)))
                .collect(),
        })
    }
}

fn seg_body_parameter_dependences(
    parameter_count: usize,
    enclosing: &StageDependenceAnalysis,
    body: &SegBody,
) -> Result<Vec<StageDependence>, String> {
    let leading = parameter_count.checked_sub(body.captures.len()).ok_or_else(|| {
        format!(
            "repeated region has {parameter_count} parameters but {} captures",
            body.captures.len()
        )
    })?;
    let mut parameter_dependences = vec![
        StageDependence::from_source(
            Uniformity::InvocationVarying,
            DependenceSource::RepeatedRegionInput,
        );
        leading
    ];
    parameter_dependences.extend(body.captures.iter().map(|capture| {
        capture.value().map_or_else(unknown_dependence, |capture| enclosing.dependence(capture))
    }));
    Ok(parameter_dependences)
}

pub(crate) fn entry_parameter_input_kind<P: Family, ResourceDecl, Route>(
    entry: &Entry<P, ResourceDecl, Route>,
    index: usize,
) -> Option<&EntryInputKind> {
    let slot = *entry.parameter_inputs.get(index)?.first()?;
    entry.inputs.get(slot.0).map(|input| &input.kind)
}

pub(crate) fn entry_parameter_dependences<P: Family, ResourceDecl, Route>(
    entry: &Entry<P, ResourceDecl, Route>,
) -> Vec<StageDependence> {
    (0..entry.params.len())
        .map(|index| {
            entry_parameter_input_kind(entry, index).map_or_else(unknown_dependence, entry_input_dependence)
        })
        .collect()
}

fn entry_input_dependence(kind: &EntryInputKind) -> StageDependence {
    let (uniformity, source) = match kind {
        EntryInputKind::Uniform { .. } => (Uniformity::StageUniform, DependenceSource::Uniform),
        EntryInputKind::PushConstant { .. } => (Uniformity::StageUniform, DependenceSource::PushConstant),
        EntryInputKind::Value {
            decoration: Some(IoDecoration::BuiltIn(spirv::BuiltIn::NumWorkgroups)),
        } => (Uniformity::StageUniform, DependenceSource::DispatchBuiltin),
        EntryInputKind::Value { .. } => (Uniformity::InvocationVarying, DependenceSource::StageInput),
        EntryInputKind::Storage { .. } => (Uniformity::StageUniform, DependenceSource::Storage),
        EntryInputKind::Texture { .. } => (Uniformity::StageUniform, DependenceSource::Texture),
        EntryInputKind::Sampler { .. } => (Uniformity::StageUniform, DependenceSource::Sampler),
        EntryInputKind::StorageImage { .. } => (Uniformity::StageUniform, DependenceSource::StorageImage),
    };
    StageDependence::from_source(uniformity, source)
}

fn pure_dependence<R>(
    op: &PureOp<R>,
    operands: &[ValueId],
    values: &LookupMap<ValueId, StageDependence>,
) -> StageDependence {
    let known = catalog().known();
    let intrinsic = match op {
        PureOp::Intrinsic { id, .. }
            if *id == known.thread_id
                || *id == known.thread_id_y
                || *id == known.thread_id_z
                || *id == known.local_id =>
        {
            StageDependence::from_source(Uniformity::InvocationVarying, DependenceSource::InvocationBuiltin)
        }
        PureOp::Intrinsic { id, .. } if *id == known.num_workgroups => {
            StageDependence::from_source(Uniformity::StageUniform, DependenceSource::DispatchBuiltin)
        }
        PureOp::StorageView(PureViewSource::Storage(_)) | PureOp::ResourceLen(_) => {
            StageDependence::from_source(Uniformity::StageUniform, DependenceSource::Storage)
        }
        PureOp::StorageView(PureViewSource::Workgroup { .. }) => {
            StageDependence::from_source(Uniformity::WorkgroupUniform, DependenceSource::WorkgroupMemory)
        }
        PureOp::StorageImageLoad(_) | PureOp::StorageImageStore(_) => {
            StageDependence::from_source(Uniformity::StageUniform, DependenceSource::StorageImage)
        }
        _ => StageDependence::constant(),
    };
    operands.iter().fold(intrinsic, |dependence, operand| {
        dependence.join(&value_dependence(values, *operand))
    })
}

fn unknown_dependence() -> StageDependence {
    StageDependence::from_source(Uniformity::InvocationVarying, DependenceSource::Unknown)
}

fn side_effect_dependence(
    node: ValueId,
    effect_blocks: &LookupMap<ValueId, BlockId>,
    block_loop_dependencies: &LookupMap<BlockId, LookupSet<BlockId>>,
) -> StageDependence {
    let dependence =
        StageDependence::from_source(Uniformity::InvocationVarying, DependenceSource::SideEffect);
    let Some(block) = effect_blocks.get(&node) else {
        return dependence;
    };
    match block_loop_dependencies.get(block) {
        Some(loops) => dependence.with_loop_dependencies(loops),
        None => dependence,
    }
}

fn call_result_dependence<P: Family>(
    graph: &EGraph<P>,
    call: CallSiteId,
    values: &LookupMap<ValueId, StageDependence>,
    call_blocks: &LookupMap<CallSiteId, BlockId>,
    block_loop_dependencies: &LookupMap<BlockId, LookupSet<BlockId>>,
) -> StageDependence {
    let site = graph.call(call);
    let arguments = site.arguments().fold(StageDependence::constant(), |dependence, argument| {
        dependence.join(&operand_dependence(graph, argument, values))
    });
    if matches!(site.effects(), CallEffects::Pure) {
        return arguments;
    }
    let effects = StageDependence::from_source(Uniformity::InvocationVarying, DependenceSource::SideEffect);
    let effects = call_blocks
        .get(&call)
        .and_then(|block| block_loop_dependencies.get(block))
        .map_or(effects.clone(), |loops| effects.with_loop_dependencies(loops));
    arguments.join(&effects)
}

fn operand_dependence<P: Family>(
    graph: &EGraph<P>,
    operand: OperandRef,
    values: &LookupMap<ValueId, StageDependence>,
) -> StageDependence {
    match operand {
        OperandRef::Value(value) => value_dependence(values, value),
        OperandRef::View(view) => value_dependence(values, view.value()),
        OperandRef::Place(place) => graph
            .place_value_dependencies(place)
            .into_iter()
            .fold(StageDependence::constant(), |dependence, value| {
                dependence.join(&value_dependence(values, value))
            }),
    }
}

fn value_dependence(values: &LookupMap<ValueId, StageDependence>, node: ValueId) -> StageDependence {
    values.get(&node).cloned().unwrap_or_else(unknown_dependence)
}

fn edge_dependence(
    block_controls: &LookupMap<BlockId, StageDependence>,
    values: &LookupMap<ValueId, StageDependence>,
    source: BlockId,
    condition: Option<ValueId>,
) -> StageDependence {
    let control = block_controls.get(&source).cloned().unwrap_or_else(unknown_dependence);
    condition.map_or(control.clone(), |condition| {
        control.join(&value_dependence(values, condition))
    })
}

fn accumulate<K: Eq + std::hash::Hash + Copy>(
    facts: &mut LookupMap<K, StageDependence>,
    key: K,
    next: StageDependence,
) -> bool {
    let current = facts.get(&key).cloned().unwrap_or_else(StageDependence::constant);
    let accumulated = current.clone().join(&next);
    if current != accumulated {
        facts.insert(key, accumulated);
        true
    } else {
        false
    }
}

fn block_loop_dependencies<P: Family>(graph: &EGraph<P>) -> LookupMap<BlockId, LookupSet<BlockId>> {
    let loops = LoopAnalysis::build(&graph.skeleton);
    graph
        .skeleton
        .blocks
        .keys()
        .map(|block| {
            let dependencies = graph
                .skeleton
                .blocks
                .keys()
                .filter(|header| loops.is_header(*header) && loops.is_in_loop(block, *header))
                .collect();
            (block, dependencies)
        })
        .collect()
}

type IncomingBlocks = LookupMap<BlockId, Vec<(BlockId, Option<ValueId>)>>;
type IncomingValues = LookupMap<ValueId, Vec<IncomingValue>>;

fn collect_incoming<P: Family>(graph: &EGraph<P>) -> Result<(IncomingBlocks, IncomingValues), String> {
    let mut incoming_blocks = LookupMap::new();
    let mut incoming_values = LookupMap::new();
    let mut seen_edges = LookupSet::new();

    for (source, block) in &graph.skeleton.blocks {
        match &block.term {
            Terminator::Branch { target, args } => record_edge(
                graph,
                &mut incoming_blocks,
                &mut incoming_values,
                &mut seen_edges,
                source,
                *target,
                None,
                args,
            )?,
            Terminator::CondBranch {
                cond,
                then_target,
                then_args,
                else_target,
                else_args,
            } => {
                record_edge(
                    graph,
                    &mut incoming_blocks,
                    &mut incoming_values,
                    &mut seen_edges,
                    source,
                    *then_target,
                    Some(*cond),
                    then_args,
                )?;
                record_edge(
                    graph,
                    &mut incoming_blocks,
                    &mut incoming_values,
                    &mut seen_edges,
                    source,
                    *else_target,
                    Some(*cond),
                    else_args,
                )?;
            }
            Terminator::Return(_) | Terminator::Unreachable => {}
        }
    }

    Ok((incoming_blocks, incoming_values))
}

#[allow(clippy::too_many_arguments)]
fn record_edge<P: Family>(
    graph: &EGraph<P>,
    incoming_blocks: &mut IncomingBlocks,
    incoming_values: &mut IncomingValues,
    seen_edges: &mut LookupSet<(BlockId, BlockId, Option<ValueId>)>,
    source: BlockId,
    target: BlockId,
    condition: Option<ValueId>,
    args: &[FlowValueId],
) -> Result<(), String> {
    let target_block =
        graph.skeleton.blocks.get(target).ok_or_else(|| {
            format!("stage-dependence analysis found an unknown branch target {target:?}")
        })?;
    if target_block.params.len() != args.len() {
        return Err(format!(
            "stage-dependence analysis found branch {source:?} -> {target:?} with {} arguments for {} parameters",
            args.len(),
            target_block.params.len()
        ));
    }
    if seen_edges.insert((source, target, condition)) {
        incoming_blocks.entry(target).or_insert_with(Vec::new).push((source, condition));
    }
    for (&parameter, &value) in target_block.params.iter().zip(args) {
        incoming_values.entry(parameter.value()).or_insert_with(Vec::new).push(IncomingValue {
            source,
            condition,
            value: value.value(),
        });
    }
    Ok(())
}
