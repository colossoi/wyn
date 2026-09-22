//! Import local source facts into the existing scheduling program.
//! Availability, transitive cost, and execution placement are derived by its rules.
use super::data::{is_slice, BlockId, ExprId, ExprKind, OperationId, OperationKind, RegionId};
use super::dependencies::Dependencies;
use super::scalar::total_node;
use super::visit::Operand;
use super::{Program, Scheduled};
use crate::interface::{EntryInputKind, StorageAccess};
use egglog_engine::{Error, FullState, RawValues, Value, Write};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Default)]
pub(super) struct Execution {
    pub host_values: BTreeSet<ExprId>,
    pub host_operations: BTreeSet<OperationId>,
    pub rematerialized: BTreeSet<OperationId>,
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

fn key(sink: &mut FullState<'_, '_>, node: Node) -> Result<Value, Error> {
    let (tag, id) = match node {
        Node::Expr(e) => ("EvalExpr", i64::from(e.as_u32())),
        Node::Operation(op) => ("EvalOperation", i64::from(op.as_u32())),
        Node::Region(r) => ("EvalRegion", i64::from(r.as_u32())),
    };
    sink.add(tag, id)
}

pub(super) fn facts(
    data: &Program<Scheduled>,
    summary: &Dependencies,
    sink: &mut FullState<'_, '_>,
) -> Result<(), Error> {
    let mut pending: Vec<_> = summary.live.iter().copied().map(Node::Operation).collect();
    pending.extend(data.entries.values().map(|e| Node::Region(data.definitions[e.definition].body)));
    let mut seen = BTreeSet::new();
    while let Some(node) = pending.pop() {
        if !seen.insert(node) {
            continue;
        }
        let value = key(sink, node)?;
        sink.add("Evaluation", value)?;
        let mut children = vec![];
        let mut class = "leaf";
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
                                sink.add("DeviceValue", value)?;
                            }
                            if inputs.iter().any(|i| i.storage_access() == Some(StorageAccess::ReadWrite)) {
                                sink.add("MutableValue", i64::from(e.as_u32()))?;
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
                            let operation = sink.add("OperationId", i64::from(op.as_u32()))?;
                            sink.add("ForwardingResult", (i64::from(e.as_u32()), operation))?;
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
                        let mutable = match &data.operations[*op].kind {
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
                        if mutable {
                            sink.add("MutableValue", i64::from(e.as_u32()))?;
                        }
                    }
                    ExprKind::Global(symbol) | ExprKind::Closure { code: symbol, .. } => {
                        match data.definitions.values().find(|d| d.symbol == *symbol) {
                            Some(d) => children.push(Node::Region(d.body)),
                            None => {
                                sink.add("DeviceValue", value)?;
                                class = "opaque";
                            }
                        }
                    }
                    ExprKind::Lambda(r) => children.push(Node::Region(*r)),
                    ExprKind::Extern(_) => {
                        sink.add("DeviceValue", value)?;
                        class = "opaque";
                    }
                    ExprKind::PureApp { .. } => {
                        class = if total_node(data, e) { "simple" } else { "opaque" };
                    }
                    ExprKind::If { .. } => class = "simple",
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
                    for child in expression.kind.children() {
                        sink.add("BackingChild", (i64::from(e.as_u32()), i64::from(child.as_u32())))?;
                    }
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
                        class = "simple";
                        sink.add("DeviceValue", value)?;
                        sink.add("ReadsValue", (value, i64::from(array.as_u32())))?;
                    }
                    OperationKind::Call { function, args } => {
                        class = "simple";
                        for &arg in args {
                            sink.add("ReadsValue", (value, i64::from(arg.as_u32())))?;
                        }
                        if matches!(data.expressions[*function].kind, ExprKind::Builtin(_)) {
                            // Builtin applications with motion proofs use PureApp.
                            class = "opaque";
                            sink.add("DeviceValue", value)?;
                        }
                    }
                    OperationKind::EvalGlobal(symbol) => {
                        class = "simple";
                        match data.definitions.values().find(|d| d.symbol == *symbol) {
                            Some(d) => children.push(Node::Region(d.body)),
                            None => {
                                sink.add("DeviceValue", value)?;
                                class = "opaque";
                            }
                        }
                    }
                    OperationKind::If { .. } => class = "simple",
                    OperationKind::Loop { .. } => class = "loop",
                    _ => {
                        sink.add("DeviceValue", value)?;
                        class = "collective";
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
        sink.add("EvaluationClass", (value, class))?;
        let mut list = sink.add("NoEvaluations", RawValues(vec![]))?;
        for &child in children.iter().rev() {
            let child = key(sink, child)?;
            sink.add("EvaluationDependency", (value, child))?;
            if count_children {
                list = sink.add("Evaluations", (child, list))?;
            }
        }
        sink.add("EvaluationWork", (value, list))?;
        pending.extend(children);
    }
    Ok(())
}
