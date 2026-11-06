//! Check the storage plan independently of the reuse/liveness decision.
use super::{invalid, Analysis};
use crate::error::Result;
use crate::ssa::types::{FuncBody, InstKind, PlaceId, ValueId};
use crate::LookupMap;
use egglog_engine::{Read, Write};
use wyn_graph::DominatorTree;

pub(super) fn verify(
    body: &FuncBody,
    analysis: &mut Analysis,
    places: &LookupMap<ValueId, PlaceId>,
) -> Result<()> {
    if places.is_empty() {
        return Ok(());
    }
    let dominators = DominatorTree::build(body.inner.entry, |b, out| {
        out.extend(body.inner.blocks[b].term.successors());
    });
    let blocks = dominators.preorder();
    let mut allocations = LookupMap::default();
    for &block in blocks {
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
    let map_error = |e: egglog_engine::Error| invalid(&e.to_string());
    let flow = &mut analysis.flow;
    // Place resources are disjoint from the array-version resources already in this graph.
    let ids: LookupMap<_, _> =
        places.values().enumerate().map(|(i, &p)| (p, (i + analysis.webs.indices.len()) as i64)).collect();
    flow.graph
        .update(|mut sink| {
            for &id in ids.values() {
                sink.add("FlowUninitialized", (flow.blocks[&body.inner.entry].0, id))?;
            }
            for (&inst, &(before, _)) in &flow.instructions {
                if let Some(place) = body.inner.insts[inst].result.and_then(|v| origins.get(&v)) {
                    sink.set("FlowKilled", (before, ids[place]), true)?;
                }
            }
            Ok::<_, egglog_engine::Error>(())
        })
        .map_err(map_error)?;
    flow.graph
        .parse_and_run_program(None, "(run-schedule (saturate flow-initialization))")
        .map_err(map_error)?;
    for &block in blocks {
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
            for place in reads {
                if flow
                    .graph
                    .read(|r| r.contains("FlowUninitialized", (flow.instructions[&inst].0, ids[&place])))
                    .map_err(map_error)?
                {
                    return Err(invalid("promoted array is read before initialization"));
                }
            }
        }
    }
    Ok(())
}
