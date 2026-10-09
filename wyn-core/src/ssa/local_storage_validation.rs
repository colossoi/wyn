//! Check the storage plan independently of the reuse/liveness decision.
use super::{invalid, Analysis};
use crate::error::Result;
use crate::ssa::types::{FuncBody, InstKind, PlaceId, ValueId};
use crate::{LookupMap, LookupSet};
use wyn_graph::DominatorTree;

pub(super) fn verify(
    body: &FuncBody,
    analysis: &Analysis,
    places: &LookupMap<ValueId, PlaceId>,
) -> Result<()> {
    if places.is_empty() {
        return Ok(());
    }
    let dominators = DominatorTree::build(body.inner.entry, |b, out| {
        out.extend(body.inner.blocks[b].term.successors());
    });
    let blocks = dominators.preorder();
    let mut predecessors = LookupMap::<_, Vec<_>>::default();
    let mut allocations = LookupMap::default();
    for &block in blocks {
        for next in body.inner.blocks[block].term.successors() {
            predecessors.entry(next).or_default().push(block);
        }
        for (position, &inst) in body.inner.blocks[block].insts.iter().enumerate() {
            if let InstKind::Alloca { result, .. } = body.inner.insts[inst].data {
                allocations.insert(result, (block, position));
            }
        }
    }
    let origins: LookupMap<_, _> = analysis
        .origins
        .iter()
        .filter_map(|(root, values)| places.get(root).map(|p| (values, *p)))
        .flat_map(|(values, p)| values.iter().map(move |v| (*v, p)))
        .collect();
    let promoted: LookupMap<_, _> = analysis
        .webs
        .indices
        .keys()
        .filter_map(|v| places.get(&analysis.webs.find(*v)).map(|p| (*v, *p)))
        .collect();
    // Definite initialization is an intersection over incoming paths. Starting
    // non-entry blocks at top handles backedges without assuming a first trip.
    let all: LookupSet<_> = places.values().copied().collect();
    let mut outputs: LookupMap<_, _> = blocks.iter().map(|b| (*b, all.clone())).collect();
    let incoming = |block, outputs: &LookupMap<_, LookupSet<_>>| {
        let mut state = if block == body.inner.entry { LookupSet::default() } else { all.clone() };
        if let Some(preds) = predecessors.get(&block) {
            for pred in preds {
                state.retain(|p| outputs[pred].contains(p));
            }
        }
        state
    };
    loop {
        let mut changed = false;
        for &block in blocks {
            let mut state = incoming(block, &outputs);
            for &inst in &body.inner.blocks[block].insts {
                if let Some(place) = body.inner.insts[inst].result.and_then(|v| origins.get(&v)) {
                    state.insert(*place);
                }
            }
            if outputs[&block] != state {
                outputs.insert(block, state);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    for &block in blocks {
        let mut initialized = incoming(block, &outputs);
        for (position, &inst) in body.inner.blocks[block].insts.iter().enumerate() {
            let node = &body.inner.insts[inst];
            let origin = node.result.and_then(|v| origins.get(&v)).copied();
            let reads: Vec<_> = node
                .data
                .value_uses()
                .into_iter()
                .filter_map(|v| v.as_ssa().and_then(|v| promoted.get(&v).copied()))
                .collect();
            for place in reads.iter().copied().chain(origin) {
                let Some(&(allocation, index)) = allocations.get(&place) else {
                    return Err(invalid("promoted storage has no allocation"));
                };
                if !dominators.dominates(allocation, block) || (allocation == block && index >= position) {
                    return Err(invalid("storage allocation does not dominate its access"));
                }
            }
            if reads.iter().any(|p| !initialized.contains(p)) {
                return Err(invalid("promoted array is read before initialization"));
            }
            if let Some(place) = origin {
                initialized.insert(place);
            }
        }
    }
    Ok(())
}
