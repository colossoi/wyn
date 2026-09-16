//! Propagate newly proved facts through reverse edges; never rescan a fixed point.
use super::*;
use std::collections::VecDeque;

pub(super) fn safety(
    data: &AssociatedData,
    members: &BTreeSet<OperationId>,
) -> (BTreeSet<RegionId>, BTreeSet<OperationId>, BTreeSet<OperationId>) {
    let mut safe = BTreeSet::new();
    let mut region_remaining: BTreeMap<_, _> =
        data.regions.iter().map(|(&r, d)| (r, d.members.len())).collect();
    for (&r, &n) in &region_remaining {
        if n == 0 {
            safe.insert(r);
        }
    }
    let mut remaining = BTreeMap::new();
    let mut on_region = BTreeMap::<RegionId, Vec<OperationId>>::new();
    let mut on_operation = BTreeMap::<OperationId, Vec<OperationId>>::new();
    let mut can_move = BTreeSet::new();
    let mut ready = VecDeque::new();
    for &op in members {
        let kind = &data.operations[op].kind;
        let mut regions = BTreeSet::new();
        let mut producer = None;
        let movable = match kind {
            k if super::super::data::length_source(data, k).is_some() => true,
            OperationKind::Index { array, .. } => {
                if !types::is_copy(&data.types[data.expressions[*array].ty].ty) {
                    producer = fresh_source(data, *array);
                    if producer.is_none() {
                        continue;
                    }
                }
                true
            }
            OperationKind::Screma { ownership, .. } => {
                ownership.iter().all(|o| *o == types::SoacOwnership::Fresh)
            }
            OperationKind::Filter { ownership, .. } => *ownership == types::SoacOwnership::Fresh,
            _ => continue,
        };
        for body in kind.callbacks() {
            if let SoacBody::Apply { region, .. } = body {
                regions.insert(*region);
            }
        }
        if movable {
            can_move.insert(op);
        }
        regions.retain(|r| !safe.contains(r));
        let count = regions.len() + usize::from(producer.is_some());
        remaining.insert(op, count);
        if count == 0 {
            ready.push_back(op);
        }
        for r in regions {
            on_region.entry(r).or_default().push(op);
        }
        if let Some(p) = producer {
            on_operation.entry(p).or_default().push(op);
        }
    }
    fn release(
        users: Option<&Vec<OperationId>>,
        remaining: &mut BTreeMap<OperationId, usize>,
        ready: &mut VecDeque<OperationId>,
    ) {
        for &op in users.into_iter().flatten() {
            if let Some(count) = remaining.get_mut(&op) {
                *count -= 1;
                if *count == 0 {
                    ready.push_back(op);
                }
            }
        }
    }
    let mut movable = BTreeSet::new();
    let mut discardable = BTreeSet::new();
    while let Some(op) = ready.pop_front() {
        discardable.insert(op);
        if !can_move.contains(&op) || !movable.insert(op) {
            continue;
        }
        release(on_operation.get(&op), &mut remaining, &mut ready);
        let region = data.operations[op].region;
        if let Some(count) = region_remaining.get_mut(&region) {
            *count -= 1;
            if *count == 0 {
                safe.insert(region);
                release(on_region.get(&region), &mut remaining, &mut ready);
            }
        }
    }
    (safe, movable, discardable)
}

fn fresh_source(data: &AssociatedData, mut e: ExprId) -> Option<OperationId> {
    loop {
        match data.expressions[e].kind {
            ExprKind::Project { tuple, .. } | ExprKind::Coerce(tuple) => e = tuple,
            ExprKind::OperationResult(op)
                if matches!(
                    data.operations[op].kind,
                    OperationKind::Screma { .. } | OperationKind::Filter { .. }
                ) =>
            {
                return Some(op)
            }
            _ => return None,
        }
    }
}

pub(super) fn external(
    data: &AssociatedData,
    members: &BTreeSet<OperationId>,
    operands: &BTreeMap<OperationId, References>,
    results: &BTreeMap<RegionId, References>,
    sets: &mut Sets,
) -> BTreeMap<RegionId, Set> {
    let tree = super::super::regions::Regions::new(data);
    let mut external: BTreeMap<_, _> = data.regions.ids().map(|r| (r, EMPTY)).collect();
    let mut users = BTreeMap::<RegionId, BTreeSet<RegionId>>::new();
    let mut pending = VecDeque::new();
    for (&r, region) in &data.regions {
        let mut refs = results[&r];
        for op in &region.members {
            refs.extend(&operands[op], sets);
        }
        for child in sets.iter(refs.regions).map(RegionId::from) {
            users.entry(child).or_default().insert(r);
        }
        let values: Vec<_> = sets.iter(refs.operations).map(OperationId::from).collect();
        for op in values {
            let owner = data.operations[op].region;
            if members.contains(&op) && owner != r && tree.contains(owner, r) {
                let next = sets.insert(external[&r], op.as_u32());
                if next != external[&r] {
                    external.insert(r, next);
                    pending.push_back((r, op));
                }
            }
        }
    }
    while let Some((child, op)) = pending.pop_front() {
        for &r in users.get(&child).into_iter().flatten() {
            let owner = data.operations[op].region;
            if owner != r && tree.contains(owner, r) {
                let next = sets.insert(external[&r], op.as_u32());
                if next != external[&r] {
                    external.insert(r, next);
                    pending.push_back((r, op));
                }
            }
        }
    }
    external
}

pub(super) fn live(
    data: &AssociatedData,
    members: &BTreeSet<OperationId>,
    discardable: &BTreeSet<OperationId>,
    operands: &BTreeMap<OperationId, References>,
    results: &BTreeMap<RegionId, References>,
    sets: &Sets,
) -> (BTreeSet<RegionId>, BTreeSet<OperationId>) {
    let mut regions: VecDeque<_> = data.definitions.values().map(|d| d.body).collect();
    let mut operations = VecDeque::new();
    let mut active = BTreeSet::new();
    let mut live = BTreeSet::new();
    while !regions.is_empty() || !operations.is_empty() {
        if let Some(r) = regions.pop_front() {
            if !active.insert(r) {
                continue;
            }
            operations.extend(data.regions[r].members.difference(discardable));
            operations.extend(sets.iter(results[&r].operations).map(OperationId::from));
            regions.extend(sets.iter(results[&r].regions).map(RegionId::from));
        } else if let Some(op) = operations.pop_front() {
            if !members.contains(&op) || !live.insert(op) {
                continue;
            }
            operations.extend(sets.iter(operands[&op].operations).map(OperationId::from));
            regions.extend(sets.iter(operands[&op].regions).map(RegionId::from));
        }
    }
    (active, live)
}
