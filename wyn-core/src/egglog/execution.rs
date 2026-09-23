//! Summarize immutable source evaluation before importing scheduling facts.
//! Egglog selects execution placement from availability and duplication cost.
use super::data::{
    is_slice, length_source, BlockId, ExprId, ExprKind, OperationId, OperationKind, RegionId,
};
use super::dependencies::Dependencies;
use super::scalar::total_node;
use super::visit::Operand;
use super::{Program, Scheduled};
use crate::interface::{EntryInputKind, StorageAccess};
use egglog_engine::{Error, FullState, Write};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

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

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
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

const OVER_BUDGET: u8 = 9;

#[derive(Clone, Copy, Default, PartialEq, Eq)]
struct Summary {
    work: u8,
    device: bool,
    mutable: bool,
}

struct Evaluation {
    children: Vec<Node>,
    backings: Vec<Node>,
    reads: Vec<Node>,
    local: Summary,
    count_children: bool,
}

pub(super) fn facts(
    data: &Program<Scheduled>,
    summary: &Dependencies,
    sink: &mut FullState<'_, '_>,
) -> Result<(), Error> {
    let evaluations = evaluations(data, summary);
    let summaries = summarize(&evaluations);
    for (&node, properties) in &summaries {
        match node {
            Node::Expr(e) if !properties.device => {
                sink.add("HostValue", i64::from(e.as_u32()))?;
            }
            Node::Operation(op) if summary.live.contains(&op) => {
                let op = sink.add("OperationId", i64::from(op.as_u32()))?;
                sink.add(
                    "ExecutionSummary",
                    (op, properties.device, properties.work < OVER_BUDGET),
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
        if evaluations.contains_key(&node) {
            continue;
        }
        let mut children = vec![];
        let mut backings = vec![];
        let mut reads = vec![];
        let mut local = Summary::default();
        let mut count_children = true;
        match node {
            Node::Expr(e) => {
                let expression = &data.expressions[e];
                children.extend(expression.kind.children().into_iter().map(Node::Expr));
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
                            let called = match kind {
                                OperationKind::Call { function, .. } => {
                                    match data.expressions[*function].kind {
                                        ExprKind::Lambda(r) => Some(r),
                                        ExprKind::Global(s) | ExprKind::Closure { code: s, .. } => {
                                            definitions.get(&s).copied()
                                        }
                                        _ => None,
                                    }
                                }
                                OperationKind::EvalGlobal(s) => definitions.get(s).copied(),
                                _ => None,
                            };
                            regions.extend(called);
                            for region in regions {
                                backings
                                    .extend(data.regions[region].results.iter().copied().map(Node::Expr));
                            }
                        }
                        // Loop and collective results are stored once. Their
                        // producer work is not duplicated by a later scalar use.
                        count_children = !matches!(
                            data.operations[*op].kind,
                            OperationKind::Loop { .. }
                                | OperationKind::Screma { .. }
                                | OperationKind::Filter { .. }
                                | OperationKind::Scatter { .. }
                                | OperationKind::BucketScatter { .. }
                                | OperationKind::ReduceByIndex { .. }
                        );
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
                                local.work = OVER_BUDGET;
                            }
                        }
                    }
                    ExprKind::Lambda(r) => children.push(Node::Region(*r)),
                    ExprKind::Extern(_) => {
                        local.device = true;
                        local.work = OVER_BUDGET;
                    }
                    ExprKind::PureApp { .. } => {
                        local.work = if total_node(data, e) { 1 } else { OVER_BUDGET };
                    }
                    ExprKind::If { .. } => local.work = 1,
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
                    backings.extend(expression.kind.children().into_iter().map(Node::Expr));
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
                        local.work = 1;
                        local.device = true;
                        reads.push(Node::Expr(*array));
                    }
                    OperationKind::Call { function, args } => {
                        local.work = 1;
                        if length_source(data, &operation.kind).is_some() {
                            // A view's length observes metadata, not its elements or
                            // the work that produced them. Availability still follows
                            // the view, including a compacted array's GPU count.
                            count_children = false;
                        } else {
                            reads.extend(args.iter().copied().map(Node::Expr));
                            if matches!(data.expressions[*function].kind, ExprKind::Builtin(_)) {
                                // Builtin applications with motion proofs use PureApp.
                                local.work = OVER_BUDGET;
                                local.device = true;
                            }
                        }
                    }
                    OperationKind::EvalGlobal(symbol) => {
                        local.work = 1;
                        match definitions.get(symbol) {
                            Some(&r) => children.push(Node::Region(r)),
                            None => {
                                local.device = true;
                                local.work = OVER_BUDGET;
                            }
                        }
                    }
                    OperationKind::If { .. } => local.work = 1,
                    OperationKind::Loop { .. } => local.work = OVER_BUDGET,
                    _ => {
                        local.device = true;
                        local.work = OVER_BUDGET;
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
        evaluations.insert(
            node,
            Evaluation {
                children,
                backings,
                reads,
                local,
                count_children,
            },
        );
    }
    evaluations
}

fn summarize(evaluations: &BTreeMap<Node, Evaluation>) -> BTreeMap<Node, Summary> {
    let mut users = BTreeMap::<Node, Vec<Node>>::new();
    for (&node, evaluation) in evaluations {
        for &child in evaluation.children.iter().chain(&evaluation.backings).chain(&evaluation.reads) {
            users.entry(child).or_default().push(node);
        }
    }
    let mut summaries: BTreeMap<_, _> = evaluations.keys().map(|&n| (n, Summary::default())).collect();
    let mut pending: VecDeque<_> = evaluations.keys().copied().collect();
    let mut queued: BTreeSet<_> = evaluations.keys().copied().collect();
    // The finite lattice also handles recursive calls. Revisit only users of
    // changed summaries; work saturates one above the duplication budget.
    while let Some(node) = pending.pop_front() {
        queued.remove(&node);
        let evaluation = &evaluations[&node];
        let mut next = evaluation.local;
        next.mutable |= evaluation.backings.iter().any(|n| summaries[n].mutable);
        if evaluation.reads.iter().any(|n| summaries[n].mutable) {
            next.work = OVER_BUDGET;
        }
        for child in &evaluation.children {
            next.device |= summaries[child].device;
            if evaluation.count_children {
                next.work = (next.work + summaries[child].work).min(OVER_BUDGET);
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
