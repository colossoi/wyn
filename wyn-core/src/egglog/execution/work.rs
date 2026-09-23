//! Coarse scalar code-work, not GPU time or a model of placement scopes.
//! Shared nodes count once per walk; arms and callees use separate walks, so
//! branch sharing is conservatively ignored. Calls consume budget before
//! recursion, bounding even cyclic helpers. Loops and unknown calls cost too much.
use super::{OVER_BUDGET, WORK_BUDGET};
use crate::egglog::data::{
    is_slice, length_source, value_source, Array, ExprId, ExprKind, Ir, OperationId, OperationKind,
    RegionId, SymbolId,
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Node {
    Expr(ExprId),
    Operation(OperationId),
    Region(RegionId),
    Length(ExprId),
}

pub(super) fn estimate(
    data: &Ir,
    live: &BTreeSet<OperationId>,
    candidates: impl IntoIterator<Item = OperationId>,
) -> BTreeMap<OperationId, u8> {
    let mut costs = Costs::new(data, live);
    candidates.into_iter().map(|op| (op, costs.count([Node::Operation(op)], WORK_BUDGET))).collect()
}

struct Costs<'a> {
    data: &'a Ir,
    live: &'a BTreeSet<OperationId>,
    definitions: BTreeMap<SymbolId, RegionId>,
    callees: BTreeMap<RegionId, u8>,
}
impl<'a> Costs<'a> {
    fn new(data: &'a Ir, live: &'a BTreeSet<OperationId>) -> Self {
        Self {
            data,
            live,
            definitions: data.definitions.values().map(|d| (d.symbol, d.body)).collect(),
            callees: BTreeMap::new(),
        }
    }

    fn count(&mut self, roots: impl IntoIterator<Item = Node>, budget: u8) -> u8 {
        let mut pending: Vec<_> = roots.into_iter().collect();
        let mut seen = BTreeSet::new();
        let mut cost = 0u8;
        while let Some(node) = pending.pop() {
            if !seen.insert(node) {
                continue;
            }
            let mut branches = vec![];
            match node {
                Node::Expr(e) => match &self.data.expressions[e].kind {
                    ExprKind::OperationResult(op) => {
                        if !self.data.operations[*op].kind.has_stored_result() {
                            pending.push(Node::Operation(*op));
                        }
                    }
                    ExprKind::If {
                        condition,
                        then_value,
                        else_value,
                    } => {
                        cost += 1;
                        pending.push(Node::Expr(*condition));
                        branches.extend([Node::Expr(*then_value), Node::Expr(*else_value)]);
                    }
                    kind => {
                        cost += u8::from(matches!(kind, ExprKind::PureApp { .. }));
                        pending.extend(kind.children().into_iter().map(Node::Expr));
                    }
                },
                Node::Operation(op) => {
                    cost += 1;
                    if cost > budget {
                        return OVER_BUDGET;
                    }
                    let kind = &self.data.operations[op].kind;
                    if let Some(array) = length_source(self.data, kind) {
                        pending.push(Node::Length(array));
                        continue;
                    }
                    let (mut values, mut regions) = (vec![], vec![]);
                    kind.operands(&mut values, &mut regions);
                    pending.extend(values.into_iter().map(Node::Expr));
                    match kind {
                        OperationKind::Index { .. } => {}
                        OperationKind::If { .. } => branches.extend(regions.into_iter().map(Node::Region)),
                        OperationKind::Call { .. } | OperationKind::EvalGlobal(_) => {
                            let Some(region) = kind.called_region(self.data, &self.definitions) else {
                                return OVER_BUDGET;
                            };
                            let work = match self.callees.get(&region) {
                                Some(&work) => work,
                                None => self.count([Node::Region(region)], budget - cost),
                            };
                            // A cutoff depends on the caller's remaining budget.
                            // Cache complete counts; charge the body at every call.
                            if work < OVER_BUDGET {
                                self.callees.insert(region, work);
                            }
                            cost = cost.saturating_add(work);
                        }
                        _ => return OVER_BUDGET,
                    }
                }
                Node::Region(r) => {
                    let region = &self.data.regions[r];
                    pending.extend(region.results.iter().copied().map(Node::Expr));
                    pending.extend(region.members.intersection(self.live).copied().map(Node::Operation));
                }
                Node::Length(e) => match &self.data.expressions[value_source(self.data, e)].kind {
                    ExprKind::PureApp { function, args }
                        if is_slice(self.data, *function) && args.len() == 3 =>
                    {
                        // The query's unit covers the subtraction. Only the bounds
                        // are evaluated, not the sliced array's elements.
                        pending.extend(args[1..].iter().copied().map(Node::Expr));
                    }
                    ExprKind::Array(array) => pending.extend(array_length(array)),
                    ExprKind::Tuple(fields) => pending.extend(fields.first().copied().map(Node::Length)),
                    ExprKind::Parameter(_) => {}
                    ExprKind::OperationResult(op) if self.data.operations[*op].kind.has_stored_result() => {
                    }
                    // Unknown metadata paths get the full value cost, not a free pass.
                    _ => pending.push(Node::Expr(e)),
                },
            }
            for branch in branches {
                if cost > budget {
                    return OVER_BUDGET;
                }
                cost = cost.saturating_add(self.count([branch], budget - cost));
            }
            if cost > budget {
                return OVER_BUDGET;
            }
        }
        cost
    }
}

fn array_length(mut array: &Array) -> Option<Node> {
    loop {
        match array {
            Array::Value(e) => return Some(Node::Length(*e)),
            Array::Zip(arrays) => array = arrays.first()?,
            Array::Literal(_) => return None,
            Array::Range { len, .. } => return Some(Node::Expr(*len)),
        }
    }
}

#[cfg(test)]
#[path = "work_tests.rs"]
mod tests;
