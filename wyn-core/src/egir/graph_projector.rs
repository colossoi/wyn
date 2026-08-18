//! Fresh graph construction for physical-entry recipes.
//!
//! A projection never clones an `EGraph` wholesale. It copies the CFG, the
//! selected side effects, and the transitive value/effect producer closure,
//! allocating fresh block, node, and side-effect-result identities throughout.

use crate::ast;
use std::collections::{HashMap, HashSet};

use crate::flow::{BlockId, ControlHeader};
use crate::{BindingRef, StableMap};

use super::graph_ops::ValueUseIndex;
use super::ir::RealizedOutputRoute;
use super::ir::{CallSiteId, FlowValueId, OperandRef, PlaceId, PlaceOp, ResultBinding};
use super::program::OutputWriter;
use super::types::{
    EGraph, EffectToken, GraphResource, Semantic, SideEffect, SideEffectIndex, SideEffectSite,
    SkeletonTerminator, ValueId, ValueKind,
};
pub struct GraphProjection<R: GraphResource = BindingRef> {
    pub graph: EGraph<Semantic<R>>,
    nodes: HashMap<ValueId, ValueId>,
    places: HashMap<PlaceId, PlaceId>,
    blocks: HashMap<BlockId, BlockId>,
    effects: HashSet<EffectToken>,
    source_effects: HashSet<SideEffectSite>,
    source_values: HashSet<ValueId>,
    effect_sites: HashMap<SideEffectSite, SideEffectSite>,
    detached_output_block: Option<BlockId>,
}

/// A projected producer recipe together with the projected identities of the
/// values requested by the caller.
pub struct ProjectedValueRecipe<R: GraphResource = BindingRef> {
    pub projection: GraphProjection<R>,
    pub values: Vec<ValueId>,
    pub result_block: BlockId,
    pub source: ValueRecipeSource,
    live_outs: Vec<ValueId>,
}

/// How a projected value recipe is removed from its source entry after its
/// result has been replaced by a handoff load. This describes a prefix
/// boundary, not the control construct inside it: selections, loops, and
/// nested structured regions all use the same continuation form.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueRecipeSource {
    EntryBlock,
    StructuredPrefix {
        continuation: BlockId,
    },
}

impl<R: GraphResource> ProjectedValueRecipe<R> {
    /// Additional projected values observed by graph structure retained after
    /// this recipe is detached. The requested values themselves are not
    /// included.
    pub fn live_outs(&self) -> impl Iterator<Item = ValueId> + '_ {
        self.live_outs.iter().copied()
    }
}

impl<R: GraphResource> GraphProjection<R> {
    pub fn node(&self, source: ValueId) -> Option<ValueId> {
        self.nodes.get(&source).copied()
    }

    pub fn place(&self, source: PlaceId) -> Option<PlaceId> {
        self.places.get(&source).copied()
    }

    pub fn result<Ty: Clone>(&self, source: &ResultBinding<Ty>) -> Result<ResultBinding<Ty>, String> {
        source.clone().try_map(
            &mut |ty| Ok(ty),
            &mut |value| {
                self.node(value).ok_or_else(|| format!("graph projection omitted result value {value:?}"))
            },
            &mut |place| {
                self.place(place).ok_or_else(|| format!("graph projection omitted result place {place:?}"))
            },
        )
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

    /// Source values retained by this projection. The projector owns this
    /// reachability decision; consumers should not rediscover it by walking
    /// the completed graph.
    pub fn source_nodes(&self) -> impl Iterator<Item = ValueId> + '_ {
        self.source_values.iter().copied()
    }

    pub fn effect_site(&self, source: SideEffectSite) -> Option<SideEffectSite> {
        self.effect_sites.get(&source).copied()
    }

    pub fn remap_output_routes(
        &self,
        routes: Vec<RealizedOutputRoute>,
    ) -> Result<Vec<RealizedOutputRoute>, String> {
        remap_output_routes(
            routes,
            |node| self.node(node),
            |block| self.block(block).or(self.detached_output_block),
            |effect| self.effect(effect),
            false,
            "graph projection",
        )
    }
}

pub(crate) fn remap_output_routes(
    routes: Vec<RealizedOutputRoute>,
    mut map_node: impl FnMut(ValueId) -> Option<ValueId>,
    mut map_block: impl FnMut(BlockId) -> Option<BlockId>,
    mut map_effect: impl FnMut(EffectToken) -> Option<EffectToken>,
    require_writers: bool,
    context: &str,
) -> Result<Vec<RealizedOutputRoute>, String> {
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

pub struct GraphProjector<'a, R: GraphResource = BindingRef> {
    source: &'a EGraph<Semantic<R>>,
    uses: ValueUseIndex,
    effects: SideEffectIndex,
}

#[derive(Clone, Copy)]
enum ProjectionMode {
    Complete,
    /// Preserve structured control flow while projecting only the value lanes
    /// needed to compute caller-selected pure results.
    ValueFlow,
    /// Preserve structured control flow and the effect producers of selected
    /// value lanes while omitting unrelated result lanes.
    Component,
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
    values: HashSet<ValueId>,
}

struct ProjectionShell<R: GraphResource> {
    graph: EGraph<Semantic<R>>,
    blocks: HashMap<BlockId, BlockId>,
    nodes: HashMap<ValueId, ValueId>,
    places: HashMap<PlaceId, PlaceId>,
    calls: HashMap<CallSiteId, CallSiteId>,
}

impl<'a, R: GraphResource> GraphProjector<'a, R> {
    pub fn new(source: &'a EGraph<Semantic<R>>) -> Self {
        Self {
            source,
            uses: ValueUseIndex::build(source),
            effects: SideEffectIndex::build(source),
        }
    }

    pub(crate) fn use_index(&self) -> &ValueUseIndex {
        &self.uses
    }

    pub fn all(&self) -> Result<GraphProjection<R>, String> {
        self.all_with_values(Vec::new())
    }

    pub fn all_with_values(&self, extra_values: Vec<ValueId>) -> Result<GraphProjection<R>, String> {
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

    /// Project selected pure values through the source CFG. Structured control
    /// flow is retained, but unrelated block-parameter lanes and function
    /// parameters are omitted. This is the control-flow counterpart to cloning
    /// a straight-line value DAG.
    pub(super) fn value_flow(&self, values: Vec<ValueId>) -> Result<GraphProjection<R>, String> {
        if values.is_empty() {
            return Err("value-flow projection requires at least one result".into());
        }
        if self.source.has_ordered_effects() {
            return Err("value-flow projection requires an effect-free graph".into());
        }
        self.project(HashSet::new(), values, ProjectionMode::ValueFlow)
    }

    pub fn selected(&self, roots: HashSet<SideEffectSite>) -> Result<GraphProjection<R>, String> {
        self.selected_with_values(roots, Vec::new())
    }

    /// Project selected operation effects as a standalone entry recipe. An
    /// operation in a continuation block may be detached when its complete
    /// producer closure is independent of that block's parameters and of
    /// effects in other blocks.
    pub fn selected_operation_recipe(
        &self,
        roots: HashSet<SideEffectSite>,
    ) -> Result<GraphProjection<R>, String> {
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
        value: ValueId,
        consumer: SideEffectSite,
    ) -> Result<ProjectedValueRecipe<R>, String> {
        self.captured_value_recipe_with_retained_values(value, consumer, Vec::new())
    }

    /// Project a captured value recipe while treating caller-owned values as
    /// additional retained observers. Entry output routes live outside the
    /// `EGraph`, so their source values are supplied through this boundary.
    pub fn captured_value_recipe_with_retained_values(
        &self,
        value: ValueId,
        consumer: SideEffectSite,
        retained_values: impl IntoIterator<Item = ValueId>,
    ) -> Result<ProjectedValueRecipe<R>, String> {
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
            if continuation.params.iter().map(|parameter| parameter.value()).ne([value]) {
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
        let live_outs = self.recipe_live_outs(&[value], &projection, source, retained_values);
        Ok(ProjectedValueRecipe {
            projection,
            values: vec![projected],
            result_block,
            source,
            live_outs,
        })
    }

    /// Project a value computed in the entry block as a standalone recipe.
    ///
    /// Unlike [`Self::captured_value_recipe`], this has no effect consumer
    /// boundary: direct vertex/fragment expressions can feed terminators and
    /// output effects throughout the entry. Producer closure still selects
    /// only effects required by `value`.
    pub fn entry_value_recipe(&self, value: ValueId) -> Result<ProjectedValueRecipe<R>, String> {
        self.entry_value_recipe_with_retained_values(value, Vec::new())
    }

    /// Project an entry value recipe with retained observers represented
    /// outside the graph, such as entry output routes.
    pub fn entry_value_recipe_with_retained_values(
        &self,
        value: ValueId,
        retained_values: impl IntoIterator<Item = ValueId>,
    ) -> Result<ProjectedValueRecipe<R>, String> {
        self.entry_values_recipe_with_retained_values([value], retained_values)
    }

    /// Project several entry-block values as one standalone recipe. Their
    /// order is preserved in [`ProjectedValueRecipe::values`].
    pub fn entry_values_recipe(
        &self,
        values: impl IntoIterator<Item = ValueId>,
    ) -> Result<ProjectedValueRecipe<R>, String> {
        self.entry_values_recipe_with_retained_values(values, Vec::new())
    }

    /// Project several entry-block values while treating caller-owned values
    /// as retained observers outside the graph.
    pub fn entry_values_recipe_with_retained_values(
        &self,
        values: impl IntoIterator<Item = ValueId>,
        retained_values: impl IntoIterator<Item = ValueId>,
    ) -> Result<ProjectedValueRecipe<R>, String> {
        let mut requested = values.into_iter().collect::<Vec<_>>();
        let mut seen = HashSet::new();
        requested.retain(|value| seen.insert(*value));
        if requested.is_empty() {
            return Err("entry value recipe requires at least one value".into());
        }
        let projection = self.project(
            HashSet::new(),
            requested.clone(),
            ProjectionMode::EntryRecipe { effect_limit: None },
        )?;
        let projected = requested
            .iter()
            .map(|value| {
                projection
                    .node(*value)
                    .ok_or_else(|| "entry value projection omitted a requested root".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let result_block = projection
            .block(self.source.skeleton.entry)
            .ok_or_else(|| "entry value projection omitted its result block".to_string())?;
        let source = ValueRecipeSource::EntryBlock;
        let live_outs = self.recipe_live_outs(&requested, &projection, source, retained_values);
        Ok(ProjectedValueRecipe {
            projection,
            values: projected,
            result_block,
            source,
            live_outs,
        })
    }

    fn recipe_live_outs(
        &self,
        roots: &[ValueId],
        projection: &GraphProjection<R>,
        source: ValueRecipeSource,
        retained_values: impl IntoIterator<Item = ValueId>,
    ) -> Vec<ValueId> {
        let retained_values = retained_values.into_iter().collect::<Vec<_>>();
        let retained_terminators = self.retained_recipe_terminators(source);
        let producer_effects = projection.source_effects();
        let mut candidates = producer_effects
            .iter()
            .filter_map(|site| self.source.effect_result_binding(self.source.skeleton.effect(*site)))
            .flat_map(|result| result.values())
            .collect::<Vec<_>>();
        candidates.extend(self.source.skeleton.blocks.iter().flat_map(|(_, block)| {
            block
                .params
                .iter()
                .map(|value| value.value())
                .filter(|value| !roots.contains(value) && projection.node(*value).is_some())
        }));
        candidates.sort_unstable();
        candidates.dedup();
        candidates.retain(|candidate| {
            if roots.contains(candidate) || projection.node(*candidate).is_none() {
                return false;
            }
            let observers = self.uses.pure_observers(*candidate);
            observers.effect_sites().any(|site| !producer_effects.contains(&site))
                || observers.terminator_blocks().any(|block| retained_terminators.contains(&block))
                || retained_values.iter().any(|value| self.uses.pure_reaches(*candidate, *value))
        });
        candidates
    }

    fn retained_recipe_terminators(&self, source: ValueRecipeSource) -> HashSet<BlockId> {
        match source {
            ValueRecipeSource::EntryBlock => self.source.skeleton.blocks.keys().collect(),
            ValueRecipeSource::StructuredPrefix { continuation } => {
                let mut retained = HashSet::new();
                let mut pending = vec![continuation];
                while let Some(block) = pending.pop() {
                    if !retained.insert(block) {
                        continue;
                    }
                    pending.extend(self.source.skeleton.blocks[block].term.successors());
                }
                retained
            }
        }
    }

    pub fn selected_with_values(
        &self,
        roots: HashSet<SideEffectSite>,
        extra_values: Vec<ValueId>,
    ) -> Result<GraphProjection<R>, String> {
        self.project(roots, extra_values, ProjectionMode::Complete)
    }

    /// Project one independently scheduled component. A component whose
    /// selected effects and complete value-producer closure are local to one
    /// block can use that block as a fresh entry, dropping unrelated control
    /// prefixes. Components that consume a block parameter or otherwise need
    /// the surrounding CFG conservatively retain the complete graph.
    pub fn selected_component_with_values(
        &self,
        roots: HashSet<SideEffectSite>,
        extra_values: Vec<ValueId>,
    ) -> Result<GraphProjection<R>, String> {
        if let Some(block) = self.component_local_block(&roots, &extra_values)? {
            return self.project(roots, extra_values, ProjectionMode::DetachedRecipe { block });
        }
        self.project(roots, extra_values, ProjectionMode::Component)
    }

    fn component_local_block(
        &self,
        roots: &HashSet<SideEffectSite>,
        extra_values: &[ValueId],
    ) -> Result<Option<BlockId>, String> {
        let mut selected = roots.clone();
        let mut values = extra_values.to_vec();
        for site in roots {
            values.extend(super::graph_ops::effect_value_inputs(
                self.source,
                self.effect_at(*site)?,
            ));
        }
        let values = self.close_producers(&mut selected, &mut values, &self.effects)?;
        let mut blocks = selected.iter().map(|site| site.block);
        let block = blocks.next().unwrap_or(self.source.skeleton.entry);
        Ok((blocks.all(|other| other == block)
            && values
                .iter()
                .all(|node| !matches!(&self.source.nodes[*node].kind, ValueKind::BlockParam { .. })))
        .then_some(block))
    }

    fn project(
        &self,
        selected: HashSet<SideEffectSite>,
        extra_values: Vec<ValueId>,
        mode: ProjectionMode,
    ) -> Result<GraphProjection<R>, String> {
        let selection = self.select_projection(selected, extra_values, mode)?;
        let mut shell = self.projection_shell(mode, &selection)?;
        for value in &selection.values {
            self.prepare_value(*value, &mut shell)?;
        }
        let (effects, effect_sites) = self.clone_effects(&selection, &mut shell)?;
        self.project_terminators(mode, &selection.blocks, &mut shell)?;
        if matches!(
            mode,
            ProjectionMode::EntryRecipe { .. } | ProjectionMode::DetachedRecipe { .. }
        ) {
            for (_, block) in shell.graph.skeleton.blocks.iter_mut() {
                block.control_header = None;
            }
        } else {
            self.project_control_headers(&mut shell)?;
        }
        self.project_aliases(&mut shell);
        shell.graph.verify_hash_cons()?;
        shell.graph.skeleton.verify_branch_arities()?;
        let detached_output_block =
            matches!(mode, ProjectionMode::DetachedRecipe { .. }).then_some(shell.graph.skeleton.entry);
        Ok(GraphProjection {
            graph: shell.graph,
            nodes: shell.nodes,
            places: shell.places,
            blocks: shell.blocks,
            effects,
            source_effects: selection.effects,
            source_values: selection.values,
            effect_sites,
            detached_output_block,
        })
    }

    fn prepare_value(&self, source: ValueId, shell: &mut ProjectionShell<R>) -> Result<ValueId, String> {
        if let Some(&target) = shell.nodes.get(&source) {
            return Ok(target);
        }
        let node = self
            .source
            .nodes
            .get(source)
            .ok_or_else(|| format!("graph projection references missing value {source:?}"))?;
        if let Some(alias) = node.alias {
            let target = self.prepare_value(alias, shell)?;
            shell.nodes.insert(source, target);
            return Ok(target);
        }
        if let Some(field) = super::graph_ops::projected_tuple_field(self.source, source) {
            let target = self.prepare_value(field, shell)?;
            shell.nodes.insert(source, target);
            return Ok(target);
        }
        match node.kind() {
            ValueKind::Pure { operands, .. } => {
                for operand in operands {
                    self.prepare_value(*operand, shell)?;
                }
            }
            ValueKind::Union { left, right } => {
                self.prepare_value(*left, shell)?;
                self.prepare_value(*right, shell)?;
            }
            ValueKind::CallResult { call, .. } => {
                self.prepare_call(*call, shell)?;
                return shell
                    .nodes
                    .get(&source)
                    .copied()
                    .ok_or_else(|| "projected call omitted one of its result bindings".to_string());
            }
            ValueKind::PlaceLength { place } => {
                let target_place = self.prepare_place(*place, shell)?;
                let target = shell.graph.add_place_length(target_place, node.ty().clone(), node.span());
                shell.nodes.insert(source, target);
                return Ok(target);
            }
            ValueKind::PlaceView { place } => {
                let target_place = self.prepare_place(*place, shell)?;
                let target =
                    shell.graph.add_place_view(target_place, node.ty().clone(), node.span()).value();
                shell.nodes.insert(source, target);
                return Ok(target);
            }
            ValueKind::FuncParam { .. } | ValueKind::BlockParam { .. } | ValueKind::SideEffectResult => {
                return Err(format!("projection shell omitted boundary value {source:?}"));
            }
            ValueKind::Constant(_) => {}
        }
        super::graph_ops::clone_value_subgraph(
            self.source,
            &mut shell.graph,
            source,
            &mut shell.nodes,
            super::graph_ops::ConstantCopy::PreserveIdentity,
            true,
            super::graph_ops::PureCopy::Preserve,
        )
    }

    fn prepare_place(&self, source: PlaceId, shell: &mut ProjectionShell<R>) -> Result<PlaceId, String> {
        if let Some(&target) = shell.places.get(&source) {
            return Ok(target);
        }
        let place = self
            .source
            .places()
            .get(source)
            .cloned()
            .ok_or_else(|| format!("graph projection references missing place {source:?}"))?;
        match place.op() {
            PlaceOp::Parameter { .. } => {
                return Err(format!("projection shell omitted parameter place {source:?}"));
            }
            PlaceOp::AllocaResult | PlaceOp::OutputSlot { .. } => {}
            PlaceOp::View { view } => {
                self.prepare_value(view.value(), shell)?;
            }
            PlaceOp::Index { base, index } => {
                self.prepare_place(*base, shell)?;
                self.prepare_value(*index, shell)?;
            }
            PlaceOp::Slice { base, start, length } => {
                self.prepare_place(*base, shell)?;
                self.prepare_value(*start, shell)?;
                self.prepare_value(*length, shell)?;
            }
            PlaceOp::ViewIndex { view, index } => {
                self.prepare_value(view.value(), shell)?;
                self.prepare_value(*index, shell)?;
            }
        }
        let mapped = place.try_map(
            |resource| Ok::<_, String>(resource),
            |value| {
                shell
                    .nodes
                    .get(&value)
                    .copied()
                    .ok_or_else(|| format!("projection omitted place value dependency {value:?}"))
            },
            |place| {
                shell
                    .places
                    .get(&place)
                    .copied()
                    .ok_or_else(|| format!("projection omitted parent place {place:?}"))
            },
        )?;
        let target = shell.graph.places.insert(mapped);
        shell.places.insert(source, target);
        Ok(target)
    }

    fn prepare_call(
        &self,
        source: CallSiteId,
        shell: &mut ProjectionShell<R>,
    ) -> Result<CallSiteId, String> {
        if let Some(&target) = shell.calls.get(&source) {
            return Ok(target);
        }
        let call = self.source.call(source).clone();
        for argument in call.arguments() {
            match *argument {
                OperandRef::Value(value) => {
                    self.prepare_value(value, shell)?;
                }
                OperandRef::View(view) => {
                    self.prepare_value(view.value(), shell)?;
                }
                OperandRef::Place(place) => {
                    self.prepare_place(place, shell)?;
                }
            }
        }
        let mut result_places = Vec::new();
        call.result().for_each_place(|place| result_places.push(place));
        for place in result_places {
            self.prepare_place(place, shell)?;
        }
        let arguments = call
            .argument_bindings()
            .iter()
            .map(|(&parameter, &argument)| {
                argument
                    .try_map(
                        |value| Ok::<_, String>(shell.nodes[&value]),
                        |view| view.try_remap(|value| Ok::<_, String>(shell.nodes[&value])),
                        |place| Ok::<_, String>(shell.places[&place]),
                    )
                    .map(|argument| (parameter, argument))
            })
            .collect::<Result<StableMap<_, _>, _>>()?;
        let places = &shell.places;
        let (target, _, values) = shell.graph.add_projected_call(
            &call,
            arguments,
            |source_value| {
                let node = &self.source.nodes[source_value];
                let ValueKind::CallResult {
                    call: result_call,
                    slot,
                } = node.kind()
                else {
                    panic!("call result binding contains a non-call value");
                };
                assert_eq!(
                    *result_call, source,
                    "call result binding references another call site"
                );
                (*slot, node.ty().clone(), node.span())
            },
            |place| places[&place],
        );
        shell.calls.insert(source, target);
        shell.nodes.extend(values);
        Ok(target)
    }

    fn select_projection(
        &self,
        mut selected: HashSet<SideEffectSite>,
        extra_values: Vec<ValueId>,
        mode: ProjectionMode,
    ) -> Result<ProjectionSelection, String> {
        let blocks = self.projected_blocks(mode)?;
        if matches!(mode, ProjectionMode::ValueFlow | ProjectionMode::Component) {
            let control_values =
                blocks.iter().filter_map(|block| match &self.source.skeleton.blocks[*block].term {
                    SkeletonTerminator::CondBranch { cond, .. } => Some(*cond),
                    _ => None,
                });
            let effect_inputs = selected.iter().flat_map(|site| {
                super::graph_ops::effect_value_inputs(self.source, self.source.skeleton.effect(*site))
            });
            let closure = super::graph_ops::value_producer_closure(
                self.source,
                extra_values.into_iter().chain(control_values).chain(effect_inputs),
            );
            if matches!(mode, ProjectionMode::ValueFlow) && !closure.effects.is_empty() {
                return Err("value-flow projection depends on an effect".into());
            }
            let call_effects = closure
                .nodes
                .iter()
                .filter_map(|value| match self.source.nodes[*value].kind() {
                    ValueKind::CallResult { call, .. } => Some(
                        self.effects
                            .call_site(*call)
                            .ok_or_else(|| format!("call {call:?} has no explicit skeleton site")),
                    ),
                    _ => None,
                })
                .collect::<Result<HashSet<_>, _>>()?;
            if matches!(mode, ProjectionMode::ValueFlow)
                && call_effects.iter().any(|site| !self.is_pure_call_site(*site))
            {
                return Err("value-flow projection depends on an effectful call".into());
            }
            let mut effects = closure.effects;
            effects.extend(selected);
            effects.extend(call_effects);
            return Ok(ProjectionSelection {
                blocks,
                effects,
                values: closure.nodes,
            });
        }
        let allowed_effects = self.allowed_effects(mode, &blocks);
        if let ProjectionMode::StructuredPrefix { continuation, .. } = mode {
            selected.extend(allowed_effects.iter().filter(|site| site.block != continuation).copied());
        }
        let mut roots = self.projected_terminator_values(mode, &blocks);
        roots.extend(extra_values);
        for site in selected.clone() {
            roots.extend(super::graph_ops::effect_value_inputs(
                self.source,
                self.effect_at(site)?,
            ));
        }
        let values = self.close_producers(&mut selected, &mut roots, &self.effects)?;
        if selected.iter().any(|site| !allowed_effects.contains(site)) {
            return Err("value recipe depends on an effect outside its prefix boundary".into());
        }
        if values.iter().any(|node| match &self.source.nodes[*node].kind {
            ValueKind::BlockParam { block, .. } => {
                !blocks.contains(block) || matches!(mode, ProjectionMode::DetachedRecipe { .. })
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
    ) -> Result<ProjectionShell<R>, String> {
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
            if let ValueKind::FuncParam { parameter } = &node.kind {
                if matches!(mode, ProjectionMode::ValueFlow | ProjectionMode::Component)
                    && !selection.values.contains(&source_id)
                {
                    continue;
                }
                let abi = super::ir::callable_parameter::<R, super::types::WynLanguage>(
                    String::new(),
                    node.ty.clone(),
                );
                let target = graph
                    .add_parameter(*parameter, abi.representation())
                    .value()
                    .expect("value and view parameters occupy the value arena");
                nodes.insert(source_id, target);
            }
        }
        let mut places = HashMap::new();
        for (source_id, place) in self.source.places() {
            if let PlaceOp::Parameter { parameter } = place.op() {
                let target = graph.add_place_parameter(*parameter, place.ty().clone());
                places.insert(source_id, target);
            }
        }
        if !matches!(mode, ProjectionMode::DetachedRecipe { .. }) {
            self.clone_live_block_params(
                &selection.blocks,
                matches!(mode, ProjectionMode::ValueFlow | ProjectionMode::Component)
                    .then_some(&selection.values),
                &mut graph,
                &blocks,
                &mut nodes,
            );
        }
        let mut shell = ProjectionShell {
            graph,
            blocks,
            nodes,
            places,
            calls: HashMap::new(),
        };
        for site in &selection.effects {
            if let Some(result) = self.effect_at(*site)?.result.as_ref() {
                for source in result.values() {
                    if shell.nodes.contains_key(&source) {
                        continue;
                    }
                    let definition = &self.source.nodes[source];
                    if definition.alias().is_none()
                        && matches!(definition.kind(), ValueKind::SideEffectResult)
                    {
                        let target = shell.graph.alloc_side_effect_result(definition.ty().clone());
                        shell.nodes.insert(source, target);
                    } else {
                        self.prepare_value(source, &mut shell)?;
                    }
                }
            }
        }
        Ok(shell)
    }

    fn clone_live_block_params(
        &self,
        projected_blocks: &HashSet<BlockId>,
        retained_values: Option<&HashSet<ValueId>>,
        graph: &mut EGraph<Semantic<R>>,
        blocks: &HashMap<BlockId, BlockId>,
        nodes: &mut HashMap<ValueId, ValueId>,
    ) {
        // The skeleton parameter list is authoritative. CFG cleanup leaves
        // eliminated BlockParam definitions in the sea; iterating the sea
        // would therefore resurrect parameters absent from the live skeleton.
        for (source_block, source_body) in &self.source.skeleton.blocks {
            if !projected_blocks.contains(&source_block) {
                continue;
            }
            let target_block = blocks[&source_block];
            for source_param in source_body.params.iter().copied() {
                let source_value = source_param.value();
                if retained_values.is_some_and(|values| !values.contains(&source_value)) {
                    continue;
                }
                let target =
                    graph.add_block_param(target_block, self.source.nodes[source_value].ty.clone());
                nodes.insert(source_value, target);
            }
        }
    }

    fn clone_effects(
        &self,
        selection: &ProjectionSelection,
        shell: &mut ProjectionShell<R>,
    ) -> Result<(HashSet<EffectToken>, HashMap<SideEffectSite, SideEffectSite>), String> {
        let mut effects = HashSet::new();
        let mut effect_sites = HashMap::new();
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
                let source_site = SideEffectSite {
                    block: source_block,
                    index,
                };
                let target_site = SideEffectSite {
                    block: target_block,
                    index: shell.graph.skeleton.blocks[target_block].side_effects.len(),
                };
                effect_sites.insert(source_site, target_site);
                for value in super::graph_ops::effect_value_inputs(self.source, effect) {
                    self.prepare_value(value, shell)?;
                }
                for operand in effect.operands() {
                    if let OperandRef::Place(place) = *operand {
                        self.prepare_place(place, shell)?;
                    }
                }
                if let Some(result) = effect.result() {
                    for place in result.places() {
                        self.prepare_place(place, shell)?;
                    }
                }
                if let super::ir::SideEffectKind::Effect(operation) = effect.kind() {
                    match operation {
                        super::ir::EffectOp::Call { site } => {
                            self.prepare_call(*site, shell)?;
                        }
                        super::ir::EffectOp::Alloca { result }
                        | super::ir::EffectOp::Load { place: result, .. }
                        | super::ir::EffectOp::Store { place: result }
                        | super::ir::EffectOp::Atomic { place: result, .. } => {
                            self.prepare_place(*result, shell)?;
                        }
                        super::ir::EffectOp::Op { .. } | super::ir::EffectOp::ControlBarrier => {}
                    }
                }
                if let super::ir::SideEffectKind::Soac(super::types::SoacEffect(_, soac)) = effect.kind() {
                    for body in soac.seg_bodies() {
                        for capture in body.captures() {
                            if let OperandRef::Place(place) = *capture {
                                self.prepare_place(place, shell)?;
                            }
                        }
                    }
                }
                let mut projected = effect.clone();
                projected.remap_referenced_values(|value| shell.nodes[&value]);
                for operand in projected.operands_mut() {
                    if let OperandRef::Place(place) = operand {
                        *place = shell.places[place];
                    }
                }
                if let Some(result) = &mut projected.result {
                    result.for_each_value_mut(|value| *value = shell.nodes[value]);
                    result.for_each_place_mut(|place| *place = shell.places[place]);
                }
                match projected.kind_mut() {
                    super::ir::SideEffectKind::Effect(operation) => {
                        *operation = operation.clone().try_map(
                            |resource| Ok::<_, String>(resource),
                            |call| Ok::<_, String>(shell.calls[&call]),
                            |place| Ok::<_, String>(shell.places[&place]),
                        )?;
                    }
                    super::ir::SideEffectKind::Soac(super::types::SoacEffect(_, soac)) => {
                        let mut index = 0;
                        while let Some(body) = soac.seg_body_mut(index) {
                            for capture in body.captures_mut() {
                                if let OperandRef::Place(place) = capture {
                                    *place = shell.places[place];
                                }
                            }
                            index += 1;
                        }
                    }
                }
                if let Some((input, output)) = projected.effects {
                    effects.extend([input, output]);
                }
                shell.graph.skeleton.blocks[target_block].side_effects.push(projected);
            }
        }
        Ok((effects, effect_sites))
    }

    fn project_terminators(
        &self,
        mode: ProjectionMode,
        projected_blocks: &HashSet<BlockId>,
        shell: &mut ProjectionShell<R>,
    ) -> Result<(), String> {
        for source_block in projected_blocks {
            let target_block = shell.blocks[source_block];
            if matches!(mode, ProjectionMode::ValueFlow | ProjectionMode::Component) {
                shell.graph.skeleton.blocks[target_block].term =
                    self.project_value_flow_terminator(*source_block, shell)?;
                continue;
            }
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
                    &shell.places,
                    &shell.blocks,
                )?
            };
        }
        Ok(())
    }

    fn project_value_flow_terminator(
        &self,
        source_block: BlockId,
        shell: &ProjectionShell<R>,
    ) -> Result<SkeletonTerminator, String> {
        let map_node = |source: ValueId| {
            shell
                .nodes
                .get(&source)
                .copied()
                .ok_or_else(|| format!("value-flow projection omitted control value {source:?}"))
        };
        let map_args = |target: BlockId, args: &[FlowValueId]| {
            self.source.skeleton.blocks[target]
                .params
                .iter()
                .zip(args)
                .filter(|(parameter, _)| shell.nodes.contains_key(&parameter.value()))
                .map(|(_, argument)| argument.try_remap(&map_node))
                .collect::<Result<Vec<_>, _>>()
        };
        Ok(match &self.source.skeleton.blocks[source_block].term {
            SkeletonTerminator::Branch { target, args } => SkeletonTerminator::Branch {
                target: shell.blocks[target],
                args: map_args(*target, args)?,
            },
            SkeletonTerminator::CondBranch {
                cond,
                then_target,
                then_args,
                else_target,
                else_args,
            } => SkeletonTerminator::CondBranch {
                cond: map_node(*cond)?,
                then_target: shell.blocks[then_target],
                then_args: map_args(*then_target, then_args)?,
                else_target: shell.blocks[else_target],
                else_args: map_args(*else_target, else_args)?,
            },
            SkeletonTerminator::Return(_) => SkeletonTerminator::Return(None),
            SkeletonTerminator::Unreachable => SkeletonTerminator::Unreachable,
        })
    }

    fn projected_blocks(&self, mode: ProjectionMode) -> Result<HashSet<BlockId>, String> {
        match mode {
            ProjectionMode::Complete | ProjectionMode::ValueFlow | ProjectionMode::Component => {
                Ok(self.source.skeleton.blocks.keys().collect())
            }
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
            ProjectionMode::Complete
            | ProjectionMode::ValueFlow
            | ProjectionMode::Component
            | ProjectionMode::DetachedRecipe { .. } => None,
        };
        let boundary_block = match mode {
            ProjectionMode::StructuredPrefix { continuation, .. } => Some(continuation),
            ProjectionMode::EntryRecipe { .. } => Some(self.source.skeleton.entry),
            ProjectionMode::DetachedRecipe { block } => Some(block),
            ProjectionMode::Complete | ProjectionMode::ValueFlow | ProjectionMode::Component => None,
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

    fn projected_terminator_values(&self, mode: ProjectionMode, blocks: &HashSet<BlockId>) -> Vec<ValueId> {
        if matches!(mode, ProjectionMode::ValueFlow | ProjectionMode::Component) {
            return Vec::new();
        }
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

    fn project_control_headers(&self, shell: &mut ProjectionShell<R>) -> Result<(), String> {
        for (header, block) in &self.source.skeleton.blocks {
            let Some(control) = &block.control_header else {
                continue;
            };
            let Some(&target_header) = shell.blocks.get(&header) else {
                continue;
            };
            let targets = control_header_targets(control);
            if targets.iter().any(|target| !shell.blocks.contains_key(target)) {
                return Err("projected structured prefix cuts through a control region".into());
            }
            shell.graph.skeleton.blocks[target_header].control_header =
                Some(control.remap(&|block| shell.blocks[&block]));
        }
        Ok(())
    }

    fn project_aliases(&self, shell: &mut ProjectionShell<R>) {
        for (&source, &target) in &shell.nodes {
            let Some(alias) = self.source.nodes[source].alias else {
                continue;
            };
            let Some(alias) = shell.nodes.get(&alias).copied() else {
                continue;
            };
            if target != alias {
                shell.graph.nodes[target].alias = Some(alias);
            }
        }
    }

    fn effect_at(&self, site: SideEffectSite) -> Result<&SideEffect<Semantic<R>>, String> {
        self.source
            .skeleton
            .blocks
            .get(site.block)
            .and_then(|block| block.side_effects.get(site.index))
            .ok_or_else(|| format!("invalid graph-projection effect site {site:?}"))
    }

    fn is_pure_call_site(&self, site: SideEffectSite) -> bool {
        self.source
            .skeleton
            .get_effect(site)
            .is_some_and(|effect| !self.source.effect_requires_ordering(effect))
    }

    fn close_producers(
        &self,
        selected: &mut HashSet<SideEffectSite>,
        values: &mut Vec<ValueId>,
        producers: &SideEffectIndex,
    ) -> Result<HashSet<ValueId>, String> {
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
            if let Some(alias) = node.alias() {
                values.push(alias);
                continue;
            }
            if let Some(field) = super::graph_ops::projected_tuple_field(self.source, value) {
                values.push(field);
                continue;
            }
            match &node.kind {
                ValueKind::Pure { operands, .. } => values.extend(operands.iter().copied()),
                ValueKind::Union { left, right } => values.extend([*left, *right]),
                ValueKind::SideEffectResult => {
                    let site = producers
                        .site(value)
                        .ok_or_else(|| format!("side-effect result {value:?} has no producer"))?;
                    if selected.insert(site) {
                        values.extend(super::graph_ops::effect_value_inputs(
                            self.source,
                            self.effect_at(site)?,
                        ));
                    }
                }
                ValueKind::CallResult { call, .. } => {
                    values.extend(self.source.call_value_dependencies(*call));
                    let site = producers
                        .call_site(*call)
                        .ok_or_else(|| format!("call {call:?} has no explicit skeleton site"))?;
                    if selected.insert(site) {
                        values.extend(super::graph_ops::effect_value_inputs(
                            self.source,
                            self.effect_at(site)?,
                        ));
                    }
                }
                ValueKind::PlaceLength { place } | ValueKind::PlaceView { place } => {
                    values.extend(self.source.place_value_dependencies(*place));
                }
                ValueKind::FuncParam { .. } | ValueKind::BlockParam { .. } | ValueKind::Constant(_) => {}
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
    nodes: &HashMap<ValueId, ValueId>,
    places: &HashMap<PlaceId, PlaceId>,
    blocks: &HashMap<BlockId, BlockId>,
) -> Result<SkeletonTerminator, String> {
    let node = |source: ValueId| {
        nodes
            .get(&source)
            .copied()
            .ok_or_else(|| format!("graph projection omitted terminator value {source:?}"))
    };
    term.clone().try_map_parts(
        &mut |condition| node(condition),
        &mut |argument: FlowValueId| argument.try_remap(&node),
        &mut |result: super::ir::ResultBinding<polytype::Type<ast::TypeName>>| {
            result.try_map(
                &mut |ty| Ok::<_, String>(ty),
                &mut |value| node(value),
                &mut |place| {
                    places
                        .get(&place)
                        .copied()
                        .ok_or_else(|| format!("graph projection omitted return place {place:?}"))
                },
            )
        },
        &mut |target| Ok(blocks[&target]),
    )
}

#[cfg(test)]
#[path = "graph_projector_tests.rs"]
mod graph_projector_tests;
