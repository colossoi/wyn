//! Specialize callback interfaces without changing other calls to the same body.
use super::captures::Context;
use super::OptimizeError;
use crate::egglog::data::{
    ExprId, ExprKind, Ir, OperationId, OperationKind, ParameterData, PlacementSite, RegionData, RegionId,
    SoacBody,
};
use crate::egglog::scalar::error;
use std::collections::{BTreeMap, BTreeSet};
use wyn_base::persistent_sets::EMPTY;

pub(super) fn apply(
    data: &mut Ir,
    op: OperationId,
    region: RegionId,
    values: &[ExprId],
    context: &mut Context<'_>,
) -> Result<(), OptimizeError> {
    // Parameterizing the same computations produces the same callable interface,
    // regardless of each invocation's actual capture values.
    let key = (region, values.to_vec());
    let replacement = match context.specializations.get(&key) {
        Some(&r) => r,
        None => {
            let r = clone_region(data, region, values, context);
            context.specializations.insert(key, r);
            r
        }
    };
    let mut kind = data.operations[op].kind.clone();
    let mut changed = false;
    kind.for_each_callback_mut(&mut |body| {
        let SoacBody::Apply {
            region: target,
            parameters,
            captures,
            ..
        } = body
        else {
            return;
        };
        if *target != region {
            return;
        }
        let pairs: BTreeMap<_, _> = data.regions[region].parameters[parameters.len()..]
            .iter()
            .copied()
            .zip(captures.iter().copied())
            .collect();
        let mut substitutions = BTreeMap::new();
        let needed = context.uses.dag.include(data, values);
        for id in context.uses.dag.sets.iter(needed).map(ExprId::from) {
            if let ExprKind::Parameter(p) = data.expressions[id].kind {
                if let Some(&v) = pairs.get(&p) {
                    substitutions.insert(id, v);
                }
            }
        }
        for &value in values {
            let argument = context.rewrite.value(data, value, &mut substitutions);
            captures.push(argument);
            context.placements.insert(PlacementSite::Operation(op), argument);
        }
        *target = replacement;
        changed = true;
    });
    if !changed {
        return Err(error("missing callback selected for specialization"));
    }
    data.operations[op].kind = kind;
    Ok(())
}

fn clone_region(data: &mut Ir, root: RegionId, values: &[ExprId], context: &mut Context<'_>) -> RegionId {
    // Follow current live references, not historical lexical arena records.
    let mut regions = vec![];
    let mut pending = vec![root];
    let mut seen = BTreeSet::new();
    let mut needed = EMPTY;
    while let Some(r) = pending.pop() {
        if !seen.insert(r) {
            continue;
        }
        if !context.lexical.contains(context.original(root), context.original(r)) {
            continue;
        }
        regions.push(r);
        let mut roots = data.regions[r].results.clone();
        for &op in data.regions[r].members.intersection(&context.live) {
            data.operations[op].kind.operands(&mut roots, &mut pending);
        }
        let set = context.uses.dag.include(data, &roots);
        let mut lambdas = EMPTY;
        for e in roots {
            lambdas = context.uses.dag.sets.union(lambdas, context.uses.dag.lambdas[&e]);
        }
        pending.extend(context.uses.dag.sets.iter(lambdas).map(RegionId::from));
        needed = context.uses.dag.sets.union(needed, set);
    }
    let region_map: BTreeMap<_, _> = regions.iter().map(|&r| (r, data.regions.alloc_id())).collect();
    let mut parameters = BTreeMap::new();
    let mut operations = BTreeMap::new();
    for &r in &regions {
        let record = data.regions[r].clone();
        context.originals.insert(region_map[&r], context.original(r));
        let target = region_map[&r];
        let new_params = record
            .parameters
            .iter()
            .map(|&p| {
                let mut record = data.parameters[p].clone();
                record.region = target;
                let new = data.parameters.alloc(record);
                parameters.insert(p, new);
                new
            })
            .collect();
        let new_ops = record
            .members
            .intersection(&context.live)
            .map(|&op| {
                let new = data.operations.alloc_id();
                operations.insert(op, new);
                new
            })
            .collect();
        data.regions.insert(
            target,
            RegionData {
                parent: record.parent.map(|p| region_map.get(&p).copied().unwrap_or(p)),
                parameters: new_params,
                members: new_ops,
                ..record
            },
        );
    }
    let mut substitutions = BTreeMap::new();
    let old: Vec<_> = context
        .uses
        .dag
        .sets
        .iter(needed)
        .map(ExprId::from)
        .map(|id| (id, data.expressions[id].clone()))
        .collect();
    for (id, e) in old {
        let kind = match e.kind {
            ExprKind::Parameter(p) => parameters.get(&p).copied().map(ExprKind::Parameter),
            ExprKind::OperationResult(op) => operations.get(&op).copied().map(ExprKind::OperationResult),
            ExprKind::Lambda(r) => region_map.get(&r).copied().map(ExprKind::Lambda),
            _ => None,
        };
        if let Some(kind) = kind {
            substitutions.insert(id, context.rewrite.intern(data, e.ty, kind));
        }
    }
    let target = region_map[&root];
    let symbol = data.definitions[data.regions[root].definition].symbol;
    for &value in values {
        let ty = data.expressions[value].ty;
        let p = data.parameters.alloc(ParameterData {
            region: target,
            symbol,
            ty,
        });
        data.regions[target].parameters.push(p);
        substitutions.insert(value, context.rewrite.intern(data, ty, ExprKind::Parameter(p)));
    }
    for (&old, &new) in &operations {
        let mut record = data.operations[old].clone();
        record.region = region_map[&record.region];
        context.rewrite.operation(data, &mut record.kind, &mut substitutions);
        match &mut record.kind {
            OperationKind::If {
                then_region,
                else_region,
                ..
            } => {
                *then_region = region_map[then_region];
                *else_region = region_map[else_region];
            }
            OperationKind::Loop { header, body, .. } => {
                *header = region_map[header];
                *body = region_map[body];
            }
            _ => {}
        }
        record.kind.for_each_callback_mut(&mut |b| {
            if let SoacBody::Apply { region, .. } = b {
                if let Some(&new) = region_map.get(region) {
                    *region = new;
                }
            }
        });
        data.operations.insert(new, record);
        context.live.insert(new);
    }
    for &old in &regions {
        let results = data.regions[old].results.clone();
        data.regions[region_map[&old]].results =
            results.into_iter().map(|e| context.rewrite.value(data, e, &mut substitutions)).collect();
    }
    for (&old, &new) in &operations {
        let values =
            context.placements.sites.get(&PlacementSite::Operation(old)).cloned().unwrap_or_default();
        for e in values {
            let value = context.rewrite.value(data, e, &mut substitutions);
            if !matches!(data.expressions[value].kind, ExprKind::Parameter(_)) {
                context.placements.insert(PlacementSite::Operation(new), value);
            }
        }
    }
    target
}
