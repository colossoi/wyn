//! Construction state for result-aware Physical EGIR control-flow merges.
//!
//! A logical result tree can use one aggregate value at a CFG join. Its
//! physical form instead binds each scalar or view leaf to a block parameter
//! and each materialized-array leaf to a stable place. The types in this
//! module keep the logical tree and that physical binding together so callers
//! cannot lose their correspondence while constructing predecessor edges.

use crate::ast::TypeName;
use crate::flow::BlockId;
use crate::types::TypeExt;
use crate::LookupMap;
use polytype::Type;
use slotmap::SlotMap;
use wyn_base::IdSource;

use super::graph_ops::{
    bind_physical_result_value, detached_alloca, emit_result_to_place, pack_result_references,
};
use super::ir::Language;
use super::program::{PhysicalResourceTable, SemanticResourceRef};
use super::types::{
    EGraph, EffectOp, EffectToken, FlowValueId, FunctionResult, GraphPhaseRemap, Physical,
    PlaceDestination, PlaceId, PureOp, ResultBinding, ResultDestination, Scheduled, SideEffect,
    SideEffectKind, SkeletonBlock, SkeletonTerminator, ValueId, ValueKind, WynLanguage,
};

/// One logical result tree together with its representation at a physical
/// merge block.
///
/// `logical` supplies the stable result-tree shape used to decompose every
/// predecessor result. `physical` binds the same leaves to either block
/// parameter values or fixed places.
#[derive(Clone)]
pub(crate) struct PhysicalFlowBinding {
    logical: FunctionResult<Type<TypeName>>,
    physical: ResultBinding<Type<TypeName>>,
}

/// All structured values carried into one physical merge block.
///
/// Fields are private so the only constructors can establish the invariant
/// that `physical` has exactly the same tree as `logical`, with no
/// materialized-array return-value leaf.
pub(crate) struct PhysicalMerge {
    block: BlockId,
    bindings: Box<[PhysicalFlowBinding]>,
}

/// Validated scalar arguments and place transfers for one physical edge.
struct PhysicalEdgePlan {
    arguments: Vec<FlowValueId>,
    transfers: Vec<(ResultBinding<Type<TypeName>>, PlaceId)>,
}

impl PhysicalFlowBinding {
    pub(crate) fn result(&self) -> &ResultBinding<Type<TypeName>> {
        &self.physical
    }

    fn field(&self, index: usize) -> Option<Self> {
        if !self.logical.is_product() {
            return None;
        }
        let logical = self.logical.top_level_fields().get(index)?.clone();
        let physical = self.physical.top_level_fields().get(index)?.clone();
        Some(Self { logical, physical })
    }
}

impl PhysicalMerge {
    /// Construct the final representation for logical results entering
    /// `block`. Materialized leaves receive function-local storage; other
    /// leaves receive block parameters.
    pub(crate) fn new(
        graph: &mut EGraph<Physical>,
        block: BlockId,
        logical: impl IntoIterator<Item = FunctionResult<Type<TypeName>>>,
        effect_ids: &mut IdSource<EffectToken>,
    ) -> Result<Self, String> {
        Self::with_reused_places(
            graph,
            block,
            logical.into_iter().map(|logical| (logical, None)),
            effect_ids,
        )
    }

    /// Construct a merge while selectively reusing already-established
    /// materialized destinations. This is used by Scheduled-to-Physical
    /// construction when every incoming logical edge names the same earlier
    /// physical binding.
    pub(crate) fn with_reused_places(
        graph: &mut EGraph<Physical>,
        block: BlockId,
        logical: impl IntoIterator<
            Item = (
                FunctionResult<Type<TypeName>>,
                Option<ResultBinding<Type<TypeName>>>,
            ),
        >,
        effect_ids: &mut IdSource<EffectToken>,
    ) -> Result<Self, String> {
        let bindings = logical
            .into_iter()
            .map(|(logical, reuse)| build_binding(graph, block, logical, reuse.as_ref(), effect_ids))
            .collect::<Result<Box<[_]>, _>>()?;
        Ok(Self { block, bindings })
    }

    /// Construct a merge whose materialized leaves reuse places from existing
    /// physical bindings. Scalar and view leaves still receive parameters in
    /// the new block.
    pub(crate) fn reusing_places<'a>(
        graph: &mut EGraph<Physical>,
        block: BlockId,
        sources: impl IntoIterator<Item = &'a PhysicalFlowBinding>,
        effect_ids: &mut IdSource<EffectToken>,
    ) -> Result<Self, String> {
        Self::with_reused_places(
            graph,
            block,
            sources.into_iter().map(|source| (source.logical.clone(), Some(source.physical.clone()))),
            effect_ids,
        )
    }

    pub(crate) fn block(&self) -> BlockId {
        self.block
    }

    pub(crate) fn bindings(&self) -> &[PhysicalFlowBinding] {
        &self.bindings
    }

    fn plan_results(
        &self,
        graph: &EGraph<Physical>,
        produced: &[ResultBinding<Type<TypeName>>],
    ) -> Result<PhysicalEdgePlan, String> {
        if produced.len() != self.bindings.len() {
            return Err(format!(
                "physical merge {:?} received {} results for {} bindings",
                self.block,
                produced.len(),
                self.bindings.len()
            ));
        }

        let mut arguments = Vec::new();
        let mut transfers = Vec::new();
        for (binding, source) in self.bindings.iter().zip(produced) {
            if binding.logical.ty() != source.ty() {
                return Err(format!(
                    "physical merge {:?} expected result type {:?}, received {:?}",
                    self.block,
                    binding.logical.ty(),
                    source.ty()
                ));
            }
            let targets = binding.physical.destination_leaves();
            let sources = source.destination_leaves();
            if targets.len() != sources.len() {
                return Err(format!(
                    "physical merge {:?} result trees have different leaf counts",
                    self.block
                ));
            }
            for (target, source) in targets.iter().zip(sources) {
                let (target_ty, destination) = target
                    .single_destination()
                    .ok_or_else(|| "physical merge target leaf has no destination".to_owned())?;
                let (source_ty, source_destination) = source
                    .single_destination()
                    .ok_or_else(|| "physical merge source leaf has no destination".to_owned())?;
                if target_ty != source_ty {
                    return Err(format!(
                        "physical merge {:?} leaf expected type {target_ty:?}, received {source_ty:?}",
                        self.block
                    ));
                }
                match destination {
                    ResultDestination::ReturnValue(_) => {
                        let ResultDestination::ReturnValue(value) = source_destination else {
                            return Err(format!(
                                "physical merge {:?} scalar or view leaf was produced through a place",
                                self.block
                            ));
                        };
                        arguments.push(graph.admit_flow_value(*value));
                    }
                    ResultDestination::Place(PlaceDestination::Fixed(place)) => {
                        let needs_transfer = match source_destination {
                            ResultDestination::Place(PlaceDestination::Fixed(source)) => source != place,
                            ResultDestination::Place(PlaceDestination::Bounded { storage, .. }) => {
                                storage != place
                            }
                            ResultDestination::ReturnValue(value) => !matches!(
                                graph.nodes[graph.canonical_value(*value)].kind(),
                                ValueKind::PlaceView { place: source } if source == place
                            ),
                        };
                        if needs_transfer {
                            transfers.push((source, *place));
                        }
                    }
                    ResultDestination::Place(PlaceDestination::Bounded { .. }) => {
                        return Err(format!(
                            "physical merge {:?} cannot use a bounded carried destination",
                            self.block
                        ));
                    }
                }
            }
        }
        Ok(PhysicalEdgePlan { arguments, transfers })
    }

    /// Bind already-structured predecessor results to this merge.
    pub(crate) fn connect_results(
        &self,
        graph: &mut EGraph<Physical>,
        predecessor: BlockId,
        produced: &[ResultBinding<Type<TypeName>>],
        effect_ids: &mut IdSource<EffectToken>,
    ) -> Result<(BlockId, Vec<FlowValueId>), String> {
        self.plan_results(graph, produced)?.connect(graph, predecessor, effect_ids)
    }
}

impl PhysicalEdgePlan {
    fn needs_transfer(&self) -> bool {
        !self.transfers.is_empty()
    }

    fn connect(
        self,
        graph: &mut EGraph<Physical>,
        predecessor: BlockId,
        effect_ids: &mut IdSource<EffectToken>,
    ) -> Result<(BlockId, Vec<FlowValueId>), String> {
        let mut tail = predecessor;
        for (source, place) in self.transfers {
            tail = emit_result_to_place(graph, tail, &source, place, effect_ids, None)?;
        }
        Ok((tail, self.arguments))
    }
}

fn build_binding(
    graph: &mut EGraph<Physical>,
    block: BlockId,
    logical: FunctionResult<Type<TypeName>>,
    reuse: Option<&ResultBinding<Type<TypeName>>>,
    effect_ids: &mut IdSource<EffectToken>,
) -> Result<PhysicalFlowBinding, String> {
    if reuse.is_some_and(|binding| binding.ty() != logical.ty()) {
        return Err("physical merge cannot reuse places from a different result type".into());
    }
    let reused = reuse.map(ResultBinding::destination_leaves).unwrap_or_default();
    let logical_leaves = logical.destination_leaves();
    if !reused.is_empty() && reused.len() != logical_leaves.len() {
        return Err("physical merge cannot reuse a result tree with different leaf count".into());
    }

    let mut destinations = Vec::with_capacity(logical_leaves.len());
    for (index, leaf) in logical_leaves.iter().enumerate() {
        if type_contains_materialized_flow(leaf.ty()) {
            let place = if let Some(reused) = reused.get(index) {
                let (_, destination) = reused
                    .single_destination()
                    .ok_or_else(|| "reused physical merge leaf has no destination".to_owned())?;
                match destination {
                    ResultDestination::Place(PlaceDestination::Fixed(place)) => *place,
                    ResultDestination::Place(PlaceDestination::Bounded { .. })
                    | ResultDestination::ReturnValue(_) => {
                        return Err("materialized physical flow leaf has no fixed place".into());
                    }
                }
            } else {
                let (place, allocation) = detached_alloca(graph, leaf.ty().clone(), effect_ids, None);
                graph.skeleton.blocks[graph.skeleton.entry].side_effects.insert(0, allocation);
                place
            };
            destinations.push(ResultDestination::Place(PlaceDestination::Fixed(place)));
        } else {
            let parameter = graph.add_block_param(block, leaf.ty().clone());
            destinations.push(ResultDestination::ReturnValue(parameter));
        }
    }

    let mut destination_index = 0usize;
    let physical = logical.map_destinations(|_, _| {
        let destination = destinations[destination_index].clone();
        destination_index += 1;
        destination
    });
    Ok(PhysicalFlowBinding { logical, physical })
}

pub(crate) fn type_contains_materialized_flow(ty: &Type<TypeName>) -> bool {
    if ty.array_variant().is_some() {
        return WynLanguage::is_materialized_aggregate(ty);
    }
    WynLanguage::product_fields(ty).is_some_and(|fields| fields.iter().any(type_contains_materialized_flow))
}

type ScheduledBlock = SkeletonBlock<Scheduled, WynLanguage>;

/// Construct Physical EGIR directly from Scheduled EGIR. Logical block
/// parameters are consumed as merge declarations: the target graph receives
/// only scalar/view parameters and fixed places, and every target edge is
/// emitted against that final representation.
pub(crate) fn construct_physical_graph(
    source: EGraph<Scheduled>,
    bindings: &PhysicalResourceTable,
    effect_ids: &mut IdSource<EffectToken>,
) -> Result<
    (
        EGraph<Physical>,
        LookupMap<ValueId, ValueId>,
        LookupMap<BlockId, BlockId>,
    ),
    String,
> {
    let mut remap = GraphPhaseRemap::<Scheduled, Physical>::new(source);
    let incoming = collect_incoming_arguments(&remap.source.skeleton.blocks)?;
    let mut merges = LookupMap::new();
    let mut physical_parameters = LookupMap::<ValueId, PhysicalFlowBinding>::new();
    {
        let source = &remap.source;
        let target = &mut remap.target;
        let node_map = &mut remap.nodes;
        for (source_block, block) in &source.skeleton.blocks {
            let target_block = remap.blocks[&source_block];
            let declarations = block
                .params
                .iter()
                .enumerate()
                .map(|(slot, parameter)| {
                    let source_parameter = parameter.value();
                    let ty = source.nodes[source_parameter].ty().clone();
                    if type_contains_materialized_flow(&ty)
                        && incoming
                            .get(&source_block)
                            .and_then(|slots| slots.get(slot))
                            .is_none_or(Vec::is_empty)
                    {
                        return Err(format!(
                            "logical materialized block parameter {slot} in {source_block:?} has no incoming edge"
                        ));
                    }
                    let reuse = common_incoming_parameter(&incoming, source_block, slot)
                        .and_then(|source| physical_parameters.get(&source))
                        .map(|binding| binding.physical.clone());
                    Ok((super::types::by_value_function_result::<WynLanguage>(ty), reuse))
                })
                .collect::<Result<Vec<_>, String>>()?;
            let merge = PhysicalMerge::with_reused_places(target, target_block, declarations, effect_ids)?;
            for (parameter, binding) in block.params.iter().zip(merge.bindings()) {
                let reference = pack_result_references(target, binding.result())?;
                node_map.insert(parameter.value(), reference);
                physical_parameters.insert(parameter.value(), binding.clone());
            }
            merges.insert(source_block, merge);
        }

        // Projection uses of a logical merge are mapped directly to the matching
        // physical field. They never become Physical projection nodes that need a
        // later folding or rebinding pass.
        for (source_value, definition) in &source.nodes {
            if physical_parameters.contains_key(&source_value) {
                continue;
            }
            let projected = match definition.kind() {
                ValueKind::Pure {
                    op: PureOp::Project { index },
                    operands,
                } if operands.len() == 1 => {
                    physical_parameters.get(&operands[0]).and_then(|binding| binding.field(*index as usize))
                }
                _ => None,
            };
            if let Some(binding) = projected {
                let reference = pack_result_references(target, binding.result())?;
                node_map.insert(source_value, reference);
                physical_parameters.insert(source_value, binding);
            }
        }
    }
    let mut map_resource = |reference: SemanticResourceRef| Ok::<_, String>(bindings.binding(reference.0));
    remap.map_graph_data(&mut map_resource)?;
    let mut map_soac =
        |id, soac, nodes: &LookupMap<ValueId, ValueId>, places: &LookupMap<PlaceId, PlaceId>| {
            super::program::physicalize_soac(soac, nodes, places, bindings).map(|soac| (id, soac))
        };
    let source_blocks = std::mem::take(&mut remap.source.skeleton.blocks);

    let mut block_tails = LookupMap::new();
    for (source_block, block) in &source_blocks {
        let mut tail = remap.block(source_block);
        for effect in &block.side_effects {
            if let Some(next) = emit_place_backed_store(&mut remap, tail, effect, effect_ids)? {
                tail = next;
                continue;
            }
            let effect = remap.map_effect(effect.clone(), &mut map_resource, &mut map_soac)?;
            remap.target.skeleton.blocks[tail].side_effects.push(effect);
        }
        block_tails.insert(source_block, tail);
    }

    for (source_block, block) in &source_blocks {
        let tail = block_tails[&source_block];
        let term = match &block.term {
            SkeletonTerminator::Branch { target, args } => {
                let produced = map_edge_results(&mut remap, args);
                let (tail, arguments) =
                    merges[target].connect_results(&mut remap.target, tail, &produced, effect_ids)?;
                remap.target.skeleton.blocks[tail].term = SkeletonTerminator::Branch {
                    target: merges[target].block(),
                    args: arguments,
                };
                None
            }
            SkeletonTerminator::CondBranch {
                cond,
                then_target,
                then_args,
                else_target,
                else_args,
            } => {
                let (then_target, then_args) =
                    construct_conditional_edge(&mut remap, &merges[then_target], then_args, effect_ids)?;
                let (else_target, else_args) =
                    construct_conditional_edge(&mut remap, &merges[else_target], else_args, effect_ids)?;
                Some(SkeletonTerminator::CondBranch {
                    cond: remap.node(*cond),
                    then_target,
                    then_args,
                    else_target,
                    else_args,
                })
            }
            SkeletonTerminator::Return(result) => Some(SkeletonTerminator::Return(
                result
                    .as_ref()
                    .map(|result| {
                        remap
                            .map_result::<String>(result.clone())
                            .map(|result| physicalize_result_binding(&remap.target, result))
                    })
                    .transpose()?,
            )),
            SkeletonTerminator::Unreachable => Some(SkeletonTerminator::Unreachable),
        };
        if let Some(term) = term {
            remap.target.skeleton.blocks[tail].term = term;
        }
        remap.target.skeleton.blocks[tail].control_header = block
            .control_header
            .clone()
            .map(|header| super::types::remap_control_header(header, &remap.blocks));
    }

    remap.target.canonicalize_boundary_operands();
    Ok(remap.finish())
}

fn collect_incoming_arguments(
    blocks: &SlotMap<BlockId, ScheduledBlock>,
) -> Result<LookupMap<BlockId, Vec<Vec<ValueId>>>, String> {
    let mut incoming = blocks
        .iter()
        .map(|(block, contents)| (block, vec![Vec::new(); contents.params.len()]))
        .collect::<LookupMap<_, _>>();
    for (source, block) in blocks {
        match &block.term {
            SkeletonTerminator::Branch { target, args } => {
                record_incoming(&mut incoming, source, *target, args)?;
            }
            SkeletonTerminator::CondBranch {
                then_target,
                then_args,
                else_target,
                else_args,
                ..
            } => {
                record_incoming(&mut incoming, source, *then_target, then_args)?;
                record_incoming(&mut incoming, source, *else_target, else_args)?;
            }
            SkeletonTerminator::Return(_) | SkeletonTerminator::Unreachable => {}
        }
    }
    Ok(incoming)
}

fn record_incoming(
    incoming: &mut LookupMap<BlockId, Vec<Vec<ValueId>>>,
    source: BlockId,
    target: BlockId,
    arguments: &[FlowValueId],
) -> Result<(), String> {
    let slots = incoming
        .get_mut(&target)
        .ok_or_else(|| format!("physical construction found unknown branch target {target:?}"))?;
    if slots.len() != arguments.len() {
        return Err(format!(
            "physical construction found branch {source:?} -> {target:?} with {} arguments for {} parameters",
            arguments.len(),
            slots.len()
        ));
    }
    for (slot, argument) in slots.iter_mut().zip(arguments) {
        slot.push(argument.value());
    }
    Ok(())
}

fn common_incoming_parameter(
    incoming: &LookupMap<BlockId, Vec<Vec<ValueId>>>,
    block: BlockId,
    slot: usize,
) -> Option<ValueId> {
    let values = incoming.get(&block)?.get(slot)?;
    let first = *values.first()?;
    values.iter().all(|value| *value == first).then_some(first)
}

fn physicalize_result_binding(
    graph: &EGraph<Physical>,
    result: ResultBinding<Type<TypeName>>,
) -> ResultBinding<Type<TypeName>> {
    result.map_destinations(|_, destination| match destination {
        ResultDestination::ReturnValue(value) => {
            let value = graph.canonical_value(*value);
            match graph.nodes[value].kind() {
                ValueKind::PlaceView { place } => ResultDestination::Place(PlaceDestination::Fixed(*place)),
                _ => ResultDestination::ReturnValue(value),
            }
        }
        ResultDestination::Place(destination) => ResultDestination::Place(destination.clone()),
    })
}

fn emit_place_backed_store(
    remap: &mut GraphPhaseRemap<Scheduled, Physical>,
    block: BlockId,
    effect: &SideEffect<Scheduled>,
    effect_ids: &mut IdSource<EffectToken>,
) -> Result<Option<BlockId>, String> {
    let SideEffectKind::Effect(EffectOp::Store { place }) = effect.kind() else {
        return Ok(None);
    };
    let [operand] = effect.operands() else {
        return Ok(None);
    };
    let Some(source) = operand.value() else {
        return Ok(None);
    };
    let mapped = remap.nodes[&source];
    let binding =
        bind_physical_result_value(&mut remap.target, remap.source.nodes[source].ty().clone(), mapped);
    if binding.places().is_empty() {
        return Ok(None);
    }
    let destination = remap.places[place];
    emit_result_to_place(
        &mut remap.target,
        block,
        &binding,
        destination,
        effect_ids,
        effect.span(),
    )
    .map(Some)
}

fn map_edge_results(
    remap: &mut GraphPhaseRemap<Scheduled, Physical>,
    arguments: &[FlowValueId],
) -> Vec<ResultBinding<Type<TypeName>>> {
    arguments
        .iter()
        .map(|argument| {
            let source = argument.value();
            bind_physical_result_value(
                &mut remap.target,
                remap.source.nodes[source].ty().clone(),
                remap.nodes[&source],
            )
        })
        .collect()
}

fn construct_conditional_edge(
    remap: &mut GraphPhaseRemap<Scheduled, Physical>,
    merge: &PhysicalMerge,
    arguments: &[FlowValueId],
    effect_ids: &mut IdSource<EffectToken>,
) -> Result<(BlockId, Vec<FlowValueId>), String> {
    let produced = map_edge_results(remap, arguments);
    let plan = merge.plan_results(&remap.target, &produced)?;
    if !plan.needs_transfer() {
        return Ok((merge.block(), plan.arguments));
    }
    let edge_block = remap.target.skeleton.create_block();
    let (tail, arguments) = plan.connect(&mut remap.target, edge_block, effect_ids)?;
    remap.target.skeleton.blocks[tail].term = SkeletonTerminator::Branch {
        target: merge.block(),
        args: arguments,
    };
    Ok((edge_block, Vec::new()))
}
