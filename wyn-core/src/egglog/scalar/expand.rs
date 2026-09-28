//! Expand scalar helpers into the interned expression DAG before EqSat.
use crate::egglog::data::{ExprId, ExprKind, Ir, OperationKind, RegionId};
use crate::egglog::rewrite::Rewriter;
use std::collections::{BTreeMap, BTreeSet};

#[cfg(test)]
#[path = "inline_tests.rs"]
mod tests;

pub(in crate::egglog) fn small(data: &Ir, region: RegionId) -> bool {
    if !data.regions[region].members.is_empty() {
        return false;
    }
    let mut pending = data.regions[region].results.clone();
    let mut seen = BTreeSet::new();
    while let Some(expression) = pending.pop() {
        if seen.insert(expression) {
            pending.extend(data.expressions[expression].kind.children());
        }
        if seen.len() > 128 {
            return false;
        }
    }
    true
}

pub(in crate::egglog) fn run(data: &mut Ir) {
    let symbols: BTreeMap<_, _> = data.definitions.values().map(|d| (d.symbol, d.body)).collect();
    let target = |data: &Ir, function: ExprId| match &data.expressions[function].kind {
        ExprKind::Global(symbol) | ExprKind::Closure { code: symbol, .. } => symbols.get(symbol).copied(),
        ExprKind::Lambda(region) => Some(*region),
        _ => None,
    };
    let Ok(order) =
        wyn_graph::topo_sort_by_dependencies(data.regions.iter().map(|(&id, _)| id), |region, out| {
            for &op in &data.regions[region].members {
                match &data.operations[op].kind {
                    OperationKind::Call { function, .. } => out.extend(target(data, *function)),
                    OperationKind::EvalGlobal(symbol) => out.extend(symbols.get(symbol).copied()),
                    _ => {}
                }
            }
        })
    else {
        return;
    };
    let mut rewrite = Rewriter::new(data);
    let mut candidates = BTreeSet::<RegionId>::new();
    let mut replacements = BTreeMap::new();
    let parameters: BTreeMap<_, _> = data
        .expressions
        .iter()
        .filter_map(
            |(&id, e)| {
                if let ExprKind::Parameter(p) = e.kind {
                    Some((p, id))
                } else {
                    None
                }
            },
        )
        .collect();
    let results: BTreeMap<_, _> = data
        .expressions
        .iter()
        .filter_map(
            |(&id, e)| {
                if let ExprKind::OperationResult(op) = e.kind {
                    Some((op, id))
                } else {
                    None
                }
            },
        )
        .collect();
    for region in order {
        let operations = data.regions[region].members.clone();
        for op in operations {
            let (callee, args) = match &data.operations[op].kind {
                OperationKind::Call { function, args } => {
                    let Some(callee) = target(data, *function) else {
                        continue;
                    };
                    let mut args = args.clone();
                    if let ExprKind::Closure { captures, .. } = &data.expressions[*function].kind {
                        args.extend(captures);
                    }
                    (callee, args)
                }
                OperationKind::EvalGlobal(symbol) => {
                    let Some(&callee) = symbols.get(symbol) else {
                        continue;
                    };
                    (callee, vec![])
                }
                _ => continue,
            };
            if !candidates.contains(&callee) || args.len() != data.regions[callee].parameters.len() {
                continue;
            }
            let mut substitutions = replacements.clone();
            for (parameter, argument) in data.regions[callee].parameters.clone().into_iter().zip(args) {
                let argument = rewrite.value(data, argument, &mut replacements.clone());
                if let Some(&expression) = parameters.get(&parameter) {
                    substitutions.insert(expression, argument);
                }
            }
            let returned = data.regions[callee]
                .results
                .clone()
                .into_iter()
                .map(|e| rewrite.value(data, e, &mut substitutions))
                .collect::<Vec<_>>();
            if let Some(&result) = results.get(&op) {
                let expression = if returned.len() == 1 {
                    returned[0]
                } else {
                    rewrite.intern(data, data.operations[op].ty, ExprKind::Tuple(returned))
                };
                replacements.insert(result, expression);
            }
            data.regions[region].members.remove(&op);
        }
        let returned = data.regions[region]
            .results
            .clone()
            .into_iter()
            .map(|e| rewrite.value(data, e, &mut replacements.clone()))
            .collect::<Vec<_>>();
        data.regions[region].results = returned.clone();
        if small(data, region) {
            candidates.insert(region);
        }
    }
    // Calls can depend on calls whose arena IDs appear later. Normalize the
    // replacement DAG before rewriting roots, rather than leaving references
    // to executions that have been removed from region membership.
    fn resolve(
        data: &mut Ir,
        rewrite: &mut Rewriter,
        id: ExprId,
        replacements: &BTreeMap<ExprId, ExprId>,
        memo: &mut BTreeMap<ExprId, ExprId>,
    ) -> ExprId {
        if let Some(&value) = memo.get(&id) {
            return value;
        }
        let value = if let Some(&next) = replacements.get(&id) {
            resolve(data, rewrite, next, replacements, memo)
        } else {
            let mut expression = data.expressions[id].clone();
            expression.kind.for_each_child_mut(&mut |child| {
                *child = resolve(data, rewrite, *child, replacements, memo);
            });
            rewrite.intern(data, expression.ty, expression.kind)
        };
        memo.insert(id, value);
        value
    }
    let mut normalized = BTreeMap::new();
    for &expression in replacements.keys() {
        resolve(data, &mut rewrite, expression, &replacements, &mut normalized);
    }
    rewrite.all(data, &normalized);
}
