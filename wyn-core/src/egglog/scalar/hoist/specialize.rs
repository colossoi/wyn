//! Specialize callback interfaces without changing other calls to the same body.
use super::*;

pub(super) fn apply(
    data: &mut AssociatedData,
    op: OperationId,
    region: RegionId,
    values: &[ExprId],
) -> Result<(), OptimizeError> {
    let replacement = clone_region(data, region, values);
    let mut kind = data.operations[op].kind.clone();
    let mut changed = false;
    bodies(&mut kind, &mut |body| {
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
        for (&id, e) in &data.expressions {
            if let ExprKind::Parameter(p) = e.kind {
                if let Some(&v) = pairs.get(&p) {
                    substitutions.insert(id, v);
                }
            }
        }
        for &value in values {
            let argument = rewrite::value(data, value, &mut substitutions);
            captures.push(argument);
            place(data, PlacementSite::Operation(op), argument);
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

fn clone_region(data: &mut AssociatedData, root: RegionId, values: &[ExprId]) -> RegionId {
    let regions: Vec<_> = data
        .regions
        .ids()
        .filter(|&id| {
            let mut parent = Some(id);
            while let Some(p) = parent {
                if p == root {
                    return true;
                }
                parent = data.regions[p].parent;
            }
            false
        })
        .collect();
    let region_map: BTreeMap<_, _> = regions.iter().map(|&r| (r, data.regions.alloc_id())).collect();
    let mut parameters = BTreeMap::new();
    let mut operations = BTreeMap::new();
    for &r in &regions {
        let record = data.regions[r].clone();
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
            .iter()
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
    let old: Vec<_> = data.expressions.iter().map(|(&id, e)| (id, e.clone())).collect();
    for (id, e) in old {
        let kind = match e.kind {
            ExprKind::Parameter(p) => parameters.get(&p).copied().map(ExprKind::Parameter),
            ExprKind::OperationResult(op) => operations.get(&op).copied().map(ExprKind::OperationResult),
            ExprKind::Lambda(r) => region_map.get(&r).copied().map(ExprKind::Lambda),
            _ => None,
        };
        if let Some(kind) = kind {
            substitutions.insert(id, expr(data, e.ty, kind));
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
        substitutions.insert(value, expr(data, ty, ExprKind::Parameter(p)));
    }
    for (&old, &new) in &operations {
        let mut record = data.operations[old].clone();
        record.region = region_map[&record.region];
        rewrite::operation(data, &mut record.kind, &mut substitutions);
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
        bodies(&mut record.kind, &mut |b| {
            if let SoacBody::Apply { region, .. } = b {
                if let Some(&new) = region_map.get(region) {
                    *region = new;
                }
            }
        });
        data.operations.insert(new, record);
    }
    for &old in &regions {
        let results = data.regions[old].results.clone();
        data.regions[region_map[&old]].results =
            results.into_iter().map(|e| rewrite::value(data, e, &mut substitutions)).collect();
    }
    let placements: Vec<_> = data.placements.values().cloned().collect();
    for placement in placements {
        if let PlacementSite::Operation(op) = placement.before {
            if let Some(&new) = operations.get(&op) {
                let value = rewrite::value(data, placement.expression, &mut substitutions);
                if !matches!(data.expressions[value].kind, ExprKind::Parameter(_)) {
                    place(data, PlacementSite::Operation(new), value);
                }
            }
        }
    }
    target
}

fn bodies(kind: &mut OperationKind, f: &mut impl FnMut(&mut SoacBody)) {
    fn body(b: &mut SoacBody, f: &mut impl FnMut(&mut SoacBody)) {
        match b {
            SoacBody::Compose { first, then } => {
                body(first, f);
                body(then, f);
            }
            SoacBody::Parallel { left, right } => {
                body(left, f);
                body(right, f);
            }
            _ => f(b),
        }
    }
    match kind {
        OperationKind::Screma { form, .. } => {
            body(&mut form.pre, f);
            body(&mut form.post, f);
            for scan in &mut form.scans {
                body(&mut scan.operator, f);
            }
            for reduce in &mut form.reductions {
                body(&mut reduce.operator, f);
            }
        }
        OperationKind::Filter { map, body: b, .. } | OperationKind::ReduceByIndex { map, body: b, .. } => {
            body(map, f);
            body(b, f);
        }
        OperationKind::Scatter { body: b, .. } | OperationKind::BucketScatter { body: b, .. } => body(b, f),
        _ => {}
    }
}
