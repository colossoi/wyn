//! Find scan epilogue candidates through declared traversal inputs.
use crate::egglog::dataflow::ir::Code;
use crate::egglog::graph::Graph;
use crate::egglog::{query::Query, OptimizeError};
use crate::LookupMap;
use egglog_engine::{Value, Write};

pub(super) fn seed(graph: &mut Graph) -> Result<(), OptimizeError> {
    let query = Query(graph);
    let mut pairs = Vec::new();
    for (&region, operations) in &graph.region_orders {
        for (position, &first) in operations.iter().enumerate() {
            if !matches!(graph.code(first)?, Code::Parallel) {
                continue;
            }
            let mut scan = false;
            let mut outputs = Vec::new();
            for (slot, output) in graph.outputs(first).into_iter().enumerate() {
                let slot = slot as i64;
                let role = graph.output_role(first, slot)?;
                let prefix = query.enode("PrefixOutput", role)?.is_some();
                let packed = query.enode("CompactedOutput", role)?.is_some()
                    && query
                        .enode(
                            "Folded",
                            graph.result_kind(first, graph.output_clause(first, slot)?)?,
                        )?
                        .is_some();
                scan |= prefix || packed;
                if prefix || packed || query.enode("CollectedOutput", role)?.is_some() {
                    outputs.push(output);
                }
            }
            if !scan {
                continue;
            }
            let domain = graph.operation_domain(first)?;
            if graph.domain_rank(domain)? != 1 {
                continue;
            }
            let extent = graph.domain_extent(domain, 0)?;
            let mut candidates = consumers(query, region, outputs)?;
            if candidates.is_empty() {
                continue;
            }
            for &next in &operations[position + 1..] {
                if let Some(loads) = candidates.remove(&next) {
                    let domain = graph.operation_domain(next)?;
                    pairs.push((first, next, extent == graph.domain_extent(domain, 0)?, loads));
                }
                if !query.contains("DiscardableOperation", next)?
                    || !query.contains("RepeatableOperation", next)?
                    || !query.flag("PureOperation", next)?
                {
                    break;
                }
            }
        }
    }
    graph.update(|mut sink| {
        for (first, next, same_extent, loads) in pairs {
            sink.add("EpilogueCandidate", (first, next))?;
            for load in loads {
                sink.add("EpilogueInputLoad", (first, next, load))?;
            }
            if same_extent {
                sink.add("EpilogueDomainPrefix", (first, next, 1_i64))?;
            }
        }
        Ok(())
    })?;
    Ok(())
}

fn consumers(
    query: Query<'_>,
    parent: Value,
    outputs: Vec<Value>,
) -> Result<LookupMap<Value, Vec<Value>>, OptimizeError> {
    let graph = query.0;
    let structure = graph.structure().read();
    let mut candidates = LookupMap::<_, Vec<_>>::new();
    for output in outputs {
        for &(load, slot) in structure.inputs.uses.get(&output).into_iter().flatten() {
            if slot != 0
                || !query.contains("LiveOperation", load)?
                || graph.code(load)? != Code::TraversalInput
            {
                continue;
            }
            let Some(mut region) = structure.operations.get(&load).and_then(|position| position.region)
            else {
                continue;
            };
            while region != parent {
                let Some(owner) = query.0.region_owner(region) else {
                    break;
                };
                let Some(enclosing) = structure.operations.get(&owner).and_then(|position| position.region)
                else {
                    break;
                };
                if enclosing == parent {
                    if matches!(graph.code(owner)?, Code::Parallel)
                        && structure.controls.get(&(owner, 0)).is_some_and(|edge| edge.region == region)
                    {
                        let domain = graph.operation_domain(owner)?;
                        if graph.domain_rank(domain)? == 1 && graph.input_count(load) == 2 {
                            let axis = graph.domain_axis(domain, 0)?;
                            if query.constructor("ValueNameRef", axis)? == graph.input_at(load, 1) {
                                candidates.entry(owner).or_default().push(load);
                            }
                        }
                    }
                    break;
                }
                region = enclosing;
            }
        }
    }
    Ok(candidates)
}
