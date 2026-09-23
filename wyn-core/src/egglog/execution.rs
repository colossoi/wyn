//! Summarize immutable source evaluation before importing scheduling facts.
//! Egglog selects execution placement from availability and duplication cost.
use super::data::{
    is_slice, length_source, BlockId, ExprId, ExprKind, OperationId, OperationKind, RegionId,
};
use super::dependencies::Dependencies;
use super::scalar::total_node;
use super::timing::time;
use super::visit::Operand;
use super::{Program, Scheduled};
use crate::builtins::{
    by_id,
    lowering::{BuiltinLowering, PrimOp},
};
use crate::interface::{EntryInputKind, StorageAccess};
use crate::{LookupMap, LookupSet};
use egglog_engine::{Error, FullState, Write};
use std::collections::{btree_map::Entry, BTreeMap, BTreeSet, VecDeque};

mod work;

#[derive(Clone, Debug, Default)]
pub(super) struct Execution {
    pub host_values: BTreeSet<ExprId>,
    pub host_operations: BTreeSet<OperationId>,
    pub rematerialized: BTreeSet<OperationId>,
    pub view_lengths: BTreeMap<ExprId, super::blocks::Value>,
    pub expansions: BTreeMap<OperationId, (BlockId, Vec<ExprId>)>,
    pub groups: BTreeMap<OperationId, Vec<OperationId>>,
    pub leaders: BTreeMap<OperationId, OperationId>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Node {
    Expr(ExprId),
    Operation(OperationId),
    Region(RegionId),
}

pub(super) fn order_facts(
    schedules: &BTreeMap<RegionId, Vec<OperationId>>,
    sink: &mut FullState<'_, '_>,
) -> Result<(), Error> {
    for (&region, operations) in schedules {
        let region = sink.add("RegionId", i64::from(region.as_u32()))?;
        let mut previous = None;
        for (i, op) in operations.iter().enumerate() {
            let op = sink.add("OperationId", i64::from(op.as_u32()))?;
            sink.add("SourcePosition", (op, region, i as i64))?;
            if let Some(before) = previous {
                sink.add("Consecutive", (before, op))?;
            }
            previous = Some(op);
        }
    }
    Ok(())
}

// Bound straight-line recomputation while allowing small uniform setup records
// to travel into their consumers instead of requiring a dispatch and buffer.
// A bounded DAG walk estimates shared work, including helper calls and arguments.
// This is a heuristic code-growth cap, not a hardware latency break-even point:
// it does not estimate consumer extent, call-site folding or hardware caches.
const WORK_BUDGET: u8 = 64;
const OVER_BUDGET: u8 = WORK_BUDGET + 1;

#[derive(Clone, Copy, Default, PartialEq, Eq)]
struct Summary {
    device: bool,
    mutable: bool,
    duplication_blocked: bool,
}

struct Evaluation {
    children: Vec<Node>,
    backings: Vec<Node>,
    reads: Vec<Node>,
    local: Summary,
    count_children: bool,
}

fn duplicable_node(data: &Program<Scheduled>, expression: ExprId) -> bool {
    if total_node(data, expression) {
        return true;
    }
    let ExprKind::PureApp { function, .. } = &data.expressions[expression].kind else {
        return false;
    };
    // Rematerialization sinks an evaluation into its consumers and preserves
    // the helper's branches. Division/remainder may be repeated with the same
    // operands and guards, even though they must not be speculated onto new
    // paths. Keep total_node (used for hoisting) deliberately more restrictive.
    // Context-dependent builtins, unknown calls and mutable reads still need
    // their existing independent proofs; this exception is arithmetic only.
    match &data.expressions[*function].kind {
        ExprKind::BinOp(op) => matches!(op.as_str(), "/" | "%"),
        ExprKind::Builtin(id) => {
            let builtin = &data.builtins[*id];
            matches!(
                by_id(builtin.builtin).overloads()[builtin.overload_idx].lowering,
                BuiltinLowering::PrimOp(
                    PrimOp::FDiv
                        | PrimOp::FRem
                        | PrimOp::FMod
                        | PrimOp::SDiv
                        | PrimOp::UDiv
                        | PrimOp::SRem
                        | PrimOp::SMod
                        | PrimOp::UMod
                )
            )
        }
        _ => false,
    }
}

pub(super) fn facts(
    data: &mut Program<Scheduled>,
    summary: &Dependencies,
    sink: &mut FullState<'_, '_>,
) -> Result<(), Error> {
    let evaluations = time("egglog scheduling / execution / graph", || {
        evaluations(data, summary)
    });
    let summaries = time("egglog scheduling / execution / summaries", || {
        summarize(&evaluations)
    });
    // Cost cannot authorize duplication or change host availability. Estimate
    // only operations for which the execution rules can use that decision.
    let candidates = summary.live.iter().copied().filter(|&op| {
        let properties = &summaries[&Node::Operation(op)];
        properties.device && !properties.duplication_blocked
    });
    let scalar_work = time("egglog scheduling / execution / work", || {
        work::estimate(data, &summary.live, candidates)
    });
    for &node in evaluations.keys() {
        let properties = &summaries[&node];
        match node {
            Node::Expr(e) if !properties.device => {
                data.state.execution.host_values.insert(e);
            }
            Node::Operation(op) if summary.live.contains(&op) => {
                let cheap = scalar_work.get(&op).is_some_and(|&work| work < OVER_BUDGET);
                let op = sink.add("OperationId", i64::from(op.as_u32()))?;
                sink.add(
                    "ExecutionSummary",
                    (op, properties.device, !properties.duplication_blocked, cheap),
                )?;
            }
            _ => {}
        }
    }
    Ok(())
}

fn evaluations(data: &Program<Scheduled>, summary: &Dependencies) -> BTreeMap<Node, Evaluation> {
    let definitions: BTreeMap<_, _> = data.definitions.values().map(|d| (d.symbol, d.body)).collect();
    let mut pending: Vec<_> = summary.live.iter().copied().map(Node::Operation).collect();
    pending.extend(data.entries.values().map(|e| Node::Region(data.definitions[e.definition].body)));
    let mut evaluations = BTreeMap::new();
    while let Some(node) = pending.pop() {
        let Entry::Vacant(entry) = evaluations.entry(node) else {
            continue;
        };
        let mut children = vec![];
        let mut backings = vec![];
        let mut reads = vec![];
        let mut local = Summary::default();
        let mut count_children = true;
        match node {
            Node::Expr(e) => {
                let expression = &data.expressions[e];
                let value_children = expression.kind.children();
                children.extend(value_children.iter().copied().map(Node::Expr));
                match &expression.kind {
                    ExprKind::Parameter(p) => {
                        if let Some(inputs) = data.state.abi.inputs.get(p) {
                            if !inputs.iter().all(|i| matches!(i.kind, EntryInputKind::PushConstant { .. }))
                            {
                                local.device = true;
                            }
                            if inputs.iter().any(|i| i.storage_access() == Some(StorageAccess::ReadWrite)) {
                                local.mutable = true;
                            }
                        }
                    }
                    ExprKind::OperationResult(op) => {
                        children.push(Node::Operation(*op));
                        if matches!(
                            data.operations[*op].kind,
                            OperationKind::Index { .. }
                                | OperationKind::Call { .. }
                                | OperationKind::If { .. }
                                | OperationKind::Loop { .. }
                                | OperationKind::EvalGlobal(_)
                        ) {
                            let kind = &data.operations[*op].kind;
                            kind.for_each_operand(&mut |operand| {
                                if let Operand::Value(_, e) = operand {
                                    backings.push(Node::Expr(e));
                                }
                            });
                            let mut regions = kind.structured_regions();
                            regions.extend(kind.called_region(data, &definitions));
                            for region in regions {
                                backings
                                    .extend(data.regions[region].results.iter().copied().map(Node::Expr));
                            }
                        }
                        count_children = !data.operations[*op].kind.has_stored_result();
                        local.mutable = match &data.operations[*op].kind {
                            OperationKind::Screma { reuse_inputs, .. } => {
                                reuse_inputs.iter().any(Option::is_some)
                            }
                            OperationKind::Filter { reuse_input, .. } => reuse_input.is_some(),
                            OperationKind::Scatter { initialize, .. } => !initialize,
                            OperationKind::BucketScatter { .. } | OperationKind::ReduceByIndex { .. } => {
                                true
                            }
                            _ => false,
                        };
                    }
                    ExprKind::Global(symbol) | ExprKind::Closure { code: symbol, .. } => {
                        match definitions.get(symbol) {
                            Some(&r) => children.push(Node::Region(r)),
                            None => {
                                local.device = true;
                                local.duplication_blocked = true;
                            }
                        }
                    }
                    ExprKind::Lambda(r) => children.push(Node::Region(*r)),
                    ExprKind::Extern(_) => {
                        local.device = true;
                        local.duplication_blocked = true;
                    }
                    ExprKind::PureApp { .. } => {
                        local.duplication_blocked = !duplicable_node(data, e);
                    }
                    _ => {}
                }
                let slice = matches!(expression.kind, ExprKind::PureApp { function, .. } if is_slice(data, function));
                if slice
                    || matches!(
                        expression.kind,
                        ExprKind::Coerce(_)
                            | ExprKind::Project { .. }
                            | ExprKind::Array(_)
                            | ExprKind::Tuple(_)
                            | ExprKind::Closure { .. }
                    )
                {
                    backings.extend(value_children.into_iter().map(Node::Expr));
                }
            }
            Node::Operation(op) => {
                let operation = &data.operations[op];
                operation.kind.for_each_operand(&mut |operand| {
                    children.push(match operand {
                        Operand::Value(_, e) => Node::Expr(e),
                        Operand::Region(r) => Node::Region(r),
                    })
                });
                match &operation.kind {
                    OperationKind::Index { array, .. } => {
                        local.device = true;
                        reads.push(Node::Expr(*array));
                    }
                    OperationKind::Call { function, args } => {
                        if length_source(data, &operation.kind).is_some() {
                            // A view's length observes metadata, not its elements or
                            // the work that produced them. Availability still follows
                            // the view, including a compacted array's GPU count.
                            count_children = false;
                        } else {
                            reads.extend(args.iter().copied().map(Node::Expr));
                            if matches!(data.expressions[*function].kind, ExprKind::Builtin(_)) {
                                // Builtin applications with motion proofs use PureApp.
                                local.device = true;
                                local.duplication_blocked = true;
                            }
                        }
                    }
                    OperationKind::EvalGlobal(symbol) => match definitions.get(symbol) {
                        Some(&r) => children.push(Node::Region(r)),
                        None => {
                            local.device = true;
                            local.duplication_blocked = true;
                        }
                    },
                    OperationKind::If { .. } => {}
                    OperationKind::Loop { .. } => {
                        local.duplication_blocked = true;
                    }
                    _ => {
                        local.device = true;
                        local.duplication_blocked = true;
                    }
                }
            }
            Node::Region(r) => {
                children.extend(data.regions[r].results.iter().copied().map(Node::Expr));
                children.extend(
                    data.regions[r].members.intersection(&summary.live).copied().map(Node::Operation),
                );
            }
        }
        pending.extend(children.iter().chain(&backings).copied());
        entry.insert(Evaluation {
            children,
            backings,
            reads,
            local,
            count_children,
        });
    }
    evaluations
}

fn summarize(evaluations: &BTreeMap<Node, Evaluation>) -> LookupMap<Node, Summary> {
    let mut users = LookupMap::<Node, Vec<Node>>::new();
    for (&node, evaluation) in evaluations {
        for &child in evaluation.children.iter().chain(&evaluation.backings).chain(&evaluation.reads) {
            users.entry(child).or_default().push(node);
        }
    }
    let mut summaries: LookupMap<_, _> = evaluations.iter().map(|(&n, e)| (n, e.local)).collect();
    // Propagation only adds properties. Local facts seed their readers; nodes
    // with no path from such a fact already have their final all-false summary.
    let mut queued = LookupSet::new();
    let mut pending: VecDeque<_> = evaluations
        .iter()
        .filter(|(_, e)| e.local != Summary::default())
        .flat_map(|(n, _)| users.get(n).into_iter().flatten().copied())
        .filter(|&n| queued.insert(n))
        .collect();
    // The finite lattice also handles recursive calls. Revisit only users of
    // changed availability and safety summaries. Cost cannot grant permission
    // to duplicate an evaluation.
    while let Some(node) = pending.pop_front() {
        queued.remove(&node);
        let evaluation = &evaluations[&node];
        let mut next = evaluation.local;
        next.mutable |= evaluation.backings.iter().any(|n| summaries[n].mutable);
        if evaluation.reads.iter().any(|n| summaries[n].mutable) {
            next.duplication_blocked = true;
        }
        for child in &evaluation.children {
            next.device |= summaries[child].device;
            if evaluation.count_children {
                next.duplication_blocked |= summaries[child].duplication_blocked;
            }
        }
        if summaries[&node] != next {
            summaries.insert(node, next);
            for &user in users.get(&node).into_iter().flatten() {
                if queued.insert(user) {
                    pending.push_back(user);
                }
            }
        }
    }
    summaries
}

#[cfg(test)]
#[path = "execution_tests.rs"]
mod tests;
