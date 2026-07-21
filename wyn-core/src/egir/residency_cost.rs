//! Safety and cost summaries for scalar values captured by scheduled SOACs.
//!
//! These weights are deliberately scheduling-specific. They compare repeated
//! lane-local evaluation with a singleton launch plus a scalar handoff; they
//! are not the e-graph extraction weights.

use std::collections::{HashMap, HashSet};

use crate::builtins::{catalog, Purity};
use crate::flow::{BlockId, ControlHeader};
use crate::interface::StorageAccess;
use crate::op::OpTag;
use crate::ssa::types::ConstantValue;
use crate::LookupMap;

use super::graph_ops;
use super::program::{SemanticEntry, SemanticProgram};
use super::types::{
    EGraph, ENode, EffectOp, NodeId, PureViewSource, SideEffect, SideEffectKind, SkeletonTerminator, Soac,
    SoacEffect,
};

pub(crate) const STORAGE_LOAD_COST: u64 = 4;
const SCALAR_OP_COST: u64 = 1;
const COMPLEX_INTRINSIC_COST: u64 = 4;
const CALL_OVERHEAD: u64 = 1;
const BRANCH_COST: u64 = 1;
const LOOP_SETUP_COST: u64 = 1;
const UNKNOWN_LOOP_COST: u64 = 4096;
pub(crate) const SINGLETON_LAUNCH_COST: u64 = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PreludeMaterializationPolicy {
    /// The structured prefix represents input-dependent serial work whose
    /// evaluation count must not be multiplied by parallel consumers.
    Required,
    /// Pure recomputation is legal, so residency is an optimization choice.
    CostBased,
}

#[derive(Debug)]
pub(crate) struct PreludeAnalysis {
    cost: u64,
    policy: PreludeMaterializationPolicy,
}

impl PreludeAnalysis {
    pub(crate) fn should_materialize(&self, invocations: u64) -> bool {
        self.policy == PreludeMaterializationPolicy::Required
            || materialization_is_profitable(self.cost, invocations)
    }
}

/// Summarize one projected parallel-prefix recipe. The same structured path
/// analysis prices straight-line values, selections, and loops. Residency
/// policy separately preserves single evaluation for structured storage
/// prefixes; all other recipes remain cost-based.
pub(crate) fn analyze_prelude(
    program: &SemanticProgram,
    entry: &SemanticEntry,
    recipe: &super::graph_projector::ProjectedValueRecipe,
) -> Option<PreludeAnalysis> {
    let graph = &recipe.projection.graph;
    let reachable = graph_ops::execution_value_producer_closure(graph, [recipe.value]).nodes;
    for node in reachable {
        if let ENode::FuncParam { index } = graph.nodes[node] {
            if !entry_input_is_invariant(entry, index) {
                return None;
            }
        }
    }

    let mut summaries = HashMap::new();
    let mut visiting = HashSet::new();
    let extra_roots = HashMap::from([(recipe.result_block, vec![recipe.value])]);
    let block_costs = graph_block_costs(program, graph, &extra_roots, &mut summaries, &mut visiting)?;
    let cost = StructuredCost::new(graph, &recipe.projection.control_headers, &block_costs).path_cost(
        graph.skeleton.entry,
        None,
        &mut HashSet::new(),
    )?;
    Some(PreludeAnalysis {
        cost,
        policy: prelude_materialization_policy(recipe),
    })
}

fn prelude_materialization_policy(
    recipe: &super::graph_projector::ProjectedValueRecipe,
) -> PreludeMaterializationPolicy {
    let structured = matches!(
        &recipe.source,
        super::graph_projector::ValueRecipeSource::StructuredPrefix { .. }
    );
    let reads_storage =
        recipe.projection.graph.skeleton.blocks.iter().flat_map(|(_, block)| &block.side_effects).any(
            |effect| {
                matches!(effect.kind, SideEffectKind::Effect(EffectOp::Load))
                    && !graph_ops::read_storage_resources(
                        &recipe.projection.graph,
                        effect.referenced_nodes(),
                    )
                    .is_empty()
            },
        );
    if structured && reads_storage {
        PreludeMaterializationPolicy::Required
    } else {
        PreludeMaterializationPolicy::CostBased
    }
}

/// Compare repeated evaluation with a one-thread producer and one handoff load
/// per launched consumer invocation. Materialization must win by 25%.
pub(crate) fn materialization_is_profitable(producer_cost: u64, invocations: u64) -> bool {
    let recompute = producer_cost.saturating_mul(invocations);
    let handoff = SINGLETON_LAUNCH_COST
        .saturating_add(producer_cost)
        .saturating_add(STORAGE_LOAD_COST.saturating_mul(invocations));
    recompute.saturating_mul(4) >= handoff.saturating_mul(5)
}

fn entry_input_is_invariant(entry: &SemanticEntry, index: usize) -> bool {
    let Some(input) = entry.inputs.get(index) else {
        return false;
    };
    if input.storage_image_binding().is_some()
        || input.texture_binding().is_some()
        || input.sampler_binding().is_some()
        || input.decoration().is_some()
    {
        return false;
    }
    if input.uniform_binding().is_some() || input.push_constant().is_some() {
        return true;
    }
    input.storage_binding().is_some()
        && !matches!(
            input.storage_access(),
            Some(StorageAccess::WriteOnly | StorageAccess::ReadWrite)
        )
}

fn effect_cost(
    program: &SemanticProgram,
    effect: &SideEffect,
    summaries: &mut HashMap<String, u64>,
    visiting: &mut HashSet<String>,
) -> Option<u64> {
    match &effect.kind {
        SideEffectKind::Soac(SoacEffect(_, Soac::Screma(_)))
        | SideEffectKind::Soac(SoacEffect(_, Soac::Filter(_)))
        | SideEffectKind::Soac(SoacEffect(_, Soac::Hist(_))) => None,
        SideEffectKind::Effect(EffectOp::Load) => Some(STORAGE_LOAD_COST),
        SideEffectKind::Effect(EffectOp::Op { tag }) => operation_cost(program, tag, summaries, visiting),
        SideEffectKind::Effect(EffectOp::Alloca { .. } | EffectOp::Store | EffectOp::ControlBarrier) => {
            None
        }
    }
}

fn operation_cost(
    program: &SemanticProgram,
    op: &OpTag<super::program::SemanticResourceRef>,
    summaries: &mut HashMap<String, u64>,
    visiting: &mut HashSet<String>,
) -> Option<u64> {
    match op {
        OpTag::Int(_)
        | OpTag::Uint(_)
        | OpTag::Float(_)
        | OpTag::Bool(_)
        | OpTag::Unit
        | OpTag::Global(_)
        | OpTag::Project { .. }
        | OpTag::ResourceLen(_)
        | OpTag::StorageViewLen
        | OpTag::ViewIndex => Some(0),
        OpTag::BinOp(_)
        | OpTag::UnaryOp(_)
        | OpTag::Tuple(_)
        | OpTag::Vector(_)
        | OpTag::Matrix { .. }
        | OpTag::ArrayLit(_)
        | OpTag::ArrayRange { .. }
        | OpTag::Materialize
        | OpTag::DynamicExtract => Some(SCALAR_OP_COST),
        // An EGIR array index becomes a value load once a storage-backed
        // array is physicalized. Charging the load cost is conservative for
        // fixed local arrays and accurate for captured entry storage.
        OpTag::Index => Some(STORAGE_LOAD_COST),
        OpTag::StorageView(PureViewSource::Storage(_) | PureViewSource::Inherited) => Some(0),
        OpTag::StorageView(PureViewSource::Workgroup { .. })
        | OpTag::PlaceIndex
        | OpTag::OutputSlot { .. }
        | OpTag::Extern(_)
        | OpTag::StorageImageLoad(_)
        | OpTag::StorageImageStore(_) => None,
        OpTag::Intrinsic { id, .. } => {
            let known = catalog().known();
            if [
                known.thread_id,
                known.local_id,
                known.num_workgroups,
                known.storage_store,
                known.texture_load,
                known.texture_sample,
                known.image_with,
                known.image_load,
            ]
            .contains(id)
                || crate::builtins::by_id(*id).raw.purity != Purity::Pure
            {
                None
            } else {
                Some(COMPLEX_INTRINSIC_COST)
            }
        }
        OpTag::Call(callee) => function_cost(program, callee, summaries, visiting)
            .map(|cost| CALL_OVERHEAD.saturating_add(cost)),
    }
}

fn function_cost(
    program: &SemanticProgram,
    callee: &str,
    summaries: &mut HashMap<String, u64>,
    visiting: &mut HashSet<String>,
) -> Option<u64> {
    if let Some(cost) = summaries.get(callee) {
        return Some(*cost);
    }
    if !visiting.insert(callee.to_string()) {
        return None;
    }
    let function = program
        .functions
        .iter()
        .find(|function| function.name == callee && function.linkage_name.is_none())?;

    let block_costs = graph_block_costs(program, &function.graph, &HashMap::new(), summaries, visiting)?;
    let cost = StructuredCost::new(&function.graph, &function.control_headers, &block_costs).path_cost(
        function.graph.skeleton.entry,
        None,
        &mut HashSet::new(),
    )?;
    visiting.remove(callee);
    summaries.insert(callee.to_string(), cost);
    Some(cost)
}

fn graph_block_costs(
    program: &SemanticProgram,
    graph: &EGraph,
    extra_roots: &HashMap<BlockId, Vec<NodeId>>,
    summaries: &mut HashMap<String, u64>,
    visiting: &mut HashSet<String>,
) -> Option<HashMap<BlockId, u64>> {
    graph
        .skeleton
        .blocks
        .iter()
        .map(|(block_id, block)| {
            let roots = block
                .side_effects
                .iter()
                .flat_map(|effect| effect.referenced_nodes())
                .chain(block.term.referenced_nodes())
                .chain(extra_roots.get(&block_id).into_iter().flatten().copied());
            let mut local = local_value_cost(program, graph, roots, summaries, visiting)?;
            for effect in &block.side_effects {
                local = local.saturating_add(effect_cost(program, effect, summaries, visiting)?);
            }
            Some((block_id, local))
        })
        .collect()
}

fn local_value_cost(
    program: &SemanticProgram,
    graph: &EGraph,
    roots: impl IntoIterator<Item = NodeId>,
    summaries: &mut HashMap<String, u64>,
    visiting: &mut HashSet<String>,
) -> Option<u64> {
    let mut pending = roots.into_iter().collect::<Vec<_>>();
    let mut seen = HashSet::new();
    let mut cost = 0u64;
    while let Some(node) = pending.pop() {
        if !seen.insert(node) {
            continue;
        }
        match &graph.nodes[node] {
            ENode::Pure { op, operands } => {
                cost = cost.saturating_add(operation_cost(program, op, summaries, visiting)?);
                pending.extend(operands.iter().copied());
            }
            ENode::Union { left, right } => pending.extend([*left, *right]),
            ENode::FuncParam { .. }
            | ENode::BlockParam { .. }
            | ENode::Constant(_)
            | ENode::SideEffectResult => {}
        }
    }
    Some(cost)
}

struct StructuredCost<'a> {
    graph: &'a EGraph,
    headers: &'a LookupMap<BlockId, ControlHeader>,
    block_costs: &'a HashMap<BlockId, u64>,
}

impl<'a> StructuredCost<'a> {
    fn new(
        graph: &'a EGraph,
        headers: &'a LookupMap<BlockId, ControlHeader>,
        block_costs: &'a HashMap<BlockId, u64>,
    ) -> Self {
        Self {
            graph,
            headers,
            block_costs,
        }
    }

    fn path_cost(
        &self,
        start: BlockId,
        stop: Option<BlockId>,
        active: &mut HashSet<BlockId>,
    ) -> Option<u64> {
        if Some(start) == stop {
            return Some(0);
        }
        if !active.insert(start) {
            return Some(UNKNOWN_LOOP_COST);
        }
        let local = self.block_costs.get(&start).copied().unwrap_or(0);
        let result = match self.headers.get(&start) {
            Some(ControlHeader::Selection { merge }) => {
                self.selection_cost(start, *merge, stop, local, active)?
            }
            Some(ControlHeader::Loop {
                merge,
                continue_block,
            }) => self.loop_cost(start, *merge, *continue_block, stop, local, active)?,
            None => self.linear_cost(start, stop, local, active)?,
        };
        active.remove(&start);
        Some(result)
    }

    fn selection_cost(
        &self,
        header: BlockId,
        merge: BlockId,
        stop: Option<BlockId>,
        local: u64,
        active: &mut HashSet<BlockId>,
    ) -> Option<u64> {
        let SkeletonTerminator::CondBranch {
            then_target,
            else_target,
            ..
        } = self.graph.skeleton.blocks[header].term
        else {
            return None;
        };
        let left = self.path_cost(then_target, Some(merge), &mut active.clone())?;
        let right = self.path_cost(else_target, Some(merge), &mut active.clone())?;
        let tail = self.path_cost(merge, stop, active)?;
        Some(local.saturating_add(BRANCH_COST).saturating_add(left.max(right)).saturating_add(tail))
    }

    fn loop_cost(
        &self,
        header: BlockId,
        merge: BlockId,
        continue_block: BlockId,
        stop: Option<BlockId>,
        local: u64,
        active: &mut HashSet<BlockId>,
    ) -> Option<u64> {
        let body = self.loop_body(header, merge)?;
        let loop_cost =
            if let Some(trips) = fixed_loop_trip_count(self.graph, header, continue_block, merge) {
                let body_cost = self.path_cost(body, Some(header), &mut active.clone())?;
                LOOP_SETUP_COST.saturating_add(
                    local.saturating_add(BRANCH_COST).saturating_add(body_cost).saturating_mul(trips),
                )
            } else {
                UNKNOWN_LOOP_COST
            };
        let tail = self.path_cost(merge, stop, active)?;
        Some(loop_cost.saturating_add(tail))
    }

    fn loop_body(&self, header: BlockId, merge: BlockId) -> Option<BlockId> {
        let SkeletonTerminator::CondBranch {
            then_target,
            else_target,
            ..
        } = self.graph.skeleton.blocks[header].term
        else {
            return None;
        };
        if then_target == merge {
            Some(else_target)
        } else if else_target == merge {
            Some(then_target)
        } else {
            None
        }
    }

    fn linear_cost(
        &self,
        block: BlockId,
        stop: Option<BlockId>,
        local: u64,
        active: &mut HashSet<BlockId>,
    ) -> Option<u64> {
        match &self.graph.skeleton.blocks[block].term {
            SkeletonTerminator::Return(_) | SkeletonTerminator::Unreachable => Some(local),
            SkeletonTerminator::Branch { target, .. } => {
                Some(local.saturating_add(self.path_cost(*target, stop, active)?))
            }
            SkeletonTerminator::CondBranch { .. } => Some(local.saturating_add(UNKNOWN_LOOP_COST)),
        }
    }
}

pub(crate) fn fixed_loop_trip_count(
    graph: &EGraph,
    header: BlockId,
    continue_block: BlockId,
    merge: BlockId,
) -> Option<u64> {
    let SkeletonTerminator::CondBranch {
        cond,
        then_target,
        else_target,
        ..
    } = graph.skeleton.blocks[header].term
    else {
        return None;
    };
    if else_target != merge || then_target == merge {
        return None;
    }
    let ENode::Pure {
        op: OpTag::BinOp(operator),
        operands,
    } = &graph.nodes[cond]
    else {
        return None;
    };
    if operator != "<" || operands.len() != 2 {
        return None;
    }
    let index = operands[0];
    let bound = integer_literal(graph, operands[1])?;
    let parameter =
        graph.skeleton.blocks[header].params.iter().position(|parameter| *parameter == index)?;
    let start = graph
        .skeleton
        .blocks
        .iter()
        .filter(|(block, _)| *block != continue_block)
        .find_map(|(_, block)| branch_argument(&block.term, header, parameter))
        .and_then(|value| integer_literal(graph, value))?;
    let next = branch_argument(&graph.skeleton.blocks[continue_block].term, header, parameter)?;
    let ENode::Pure {
        op: OpTag::BinOp(operator),
        operands,
    } = &graph.nodes[next]
    else {
        return None;
    };
    if operator != "+" || operands.len() != 2 {
        return None;
    }
    let step = if operands[0] == index {
        integer_literal(graph, operands[1])?
    } else if operands[1] == index {
        integer_literal(graph, operands[0])?
    } else {
        return None;
    };
    if step <= 0 || bound <= start {
        return Some(0);
    }
    let distance = u64::try_from(bound - start).ok()?;
    let step = u64::try_from(step).ok()?;
    Some(distance.div_ceil(step))
}

fn branch_argument(term: &SkeletonTerminator, target: BlockId, index: usize) -> Option<NodeId> {
    match term {
        SkeletonTerminator::Branch {
            target: branch_target,
            args,
        } if *branch_target == target => args.get(index).copied(),
        SkeletonTerminator::CondBranch {
            then_target,
            then_args,
            else_target,
            else_args,
            ..
        } if *then_target == target => then_args.get(index).copied(),
        SkeletonTerminator::CondBranch {
            else_target,
            else_args,
            ..
        } if *else_target == target => else_args.get(index).copied(),
        _ => None,
    }
}

fn integer_literal(graph: &EGraph, node: NodeId) -> Option<i64> {
    match &graph.nodes[node] {
        ENode::Constant(ConstantValue::I32(value)) => Some(i64::from(*value)),
        ENode::Constant(ConstantValue::U32(value)) => Some(i64::from(*value)),
        ENode::Pure {
            op: OpTag::Int(value),
            operands,
        }
        | ENode::Pure {
            op: OpTag::Uint(value),
            operands,
        } if operands.is_empty() => value.parse().ok(),
        _ => None,
    }
}

#[cfg(test)]
#[path = "residency_cost_tests.rs"]
mod residency_cost_tests;
