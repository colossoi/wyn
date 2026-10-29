//! Rank bounded local iterations before loop storage identities are installed.
use super::super::{cost::CostModel, query::Query, OptimizeError};
use crate::{LookupMap, LookupSet};
use egglog_engine::sort::{SetContainer, S};
use egglog_engine::{EGraph, Value, Write};

struct Work<'a> {
    query: Query<'a>,
    operations: LookupMap<Value, Vec<(Value, String)>>,
    regions: LookupMap<Value, Option<u64>>,
    active: LookupSet<Value>,
}

impl Work<'_> {
    fn integer(&self, value: Value) -> u64 {
        self.query.0.value_to_base::<i64>(value).max(0) as u64
    }

    fn extent(&self, extent: Value) -> Result<Option<u64>, OptimizeError> {
        let q = self.query;
        if let Some(fields) = q.enode("Fixed", extent)? {
            return Ok(Some(self.integer(fields[0])));
        }
        if let Some(fields) = q.enode("Length", extent)? {
            let ty = q.required("SourceType", (fields[0],))?;
            let size = q.required("SourceArraySize", (ty,))?;
            return Ok(q.enode("FixedSize", size)?.map(|fields| self.integer(fields[0])));
        }
        if let Some(fields) = q.enode("Scalar", extent)? {
            return Ok(q.lookup("SourceConstantInteger", (fields[0],))?.map(|n| self.integer(n)));
        }
        for operator in ["Product", "Sum", "Difference"] {
            if let Some(fields) = q.enode(operator, extent)? {
                return Ok(match (self.extent(fields[0])?, self.extent(fields[1])?) {
                    (Some(a), Some(b)) => Some(match operator {
                        "Product" => a.saturating_mul(b),
                        "Sum" => a.saturating_add(b),
                        _ => a.saturating_sub(b),
                    }),
                    _ => None,
                });
            }
        }
        Ok(None)
    }

    fn region(&mut self, region: Value) -> Result<Option<u64>, OptimizeError> {
        if let Some(&work) = self.regions.get(&region) {
            return Ok(work);
        }
        if !self.active.insert(region) {
            return Ok(None);
        }
        let work = self.region_work(region)?;
        self.active.remove(&region);
        self.regions.insert(region, work);
        Ok(work)
    }

    fn region_work(&mut self, region: Value) -> Result<Option<u64>, OptimizeError> {
        let q = self.query;
        if !q.flag("SourceRegionReadOnly", (region,))? {
            return Ok(None);
        }
        let mut work = self.integer(q.required("SourceRegionWork", (region,))?);
        let mut counted_regions = LookupSet::default();
        for (op, kind) in self.operations.get(&region).cloned().unwrap_or_default() {
            match kind.as_str() {
                "map" | "scan" | "reduce" => {
                    let Some(extent) = q.lookup("SourceDomainExtent", (op,))? else {
                        return Ok(None);
                    };
                    let Some(elements) = self.extent(extent)? else {
                        return Ok(None);
                    };
                    let body = q.required("SourceOperatorBody", (op,))?;
                    let Some(inner) = self.region(body)? else {
                        return Ok(None);
                    };
                    // Retain the source operation's overhead and charge every
                    // callback iteration, including nested array operations.
                    work = work.saturating_add(elements.saturating_mul(inner.saturating_add(1)));
                    counted_regions.insert(body);
                }
                "index" => {}
                "control" => {
                    let source = q.required("SourceOperationValue", (op,))?;
                    if let Some(regions) = q.record("SourceLoop", "LoopInfo", (source,))? {
                        let form = q.required("SourceLoopForm", (source,))?;
                        let Some(form) = q.enode("ForCount", form)? else {
                            return Ok(None);
                        };
                        let Some(count) = q.lookup("SourceConstantInteger", (form[0],))? else {
                            return Ok(None);
                        };
                        let count = self.integer(count);
                        // Header and body execute per trip, not once as lexical children.
                        for child in regions {
                            if count > 0 {
                                let Some(inner) = self.region(child)? else {
                                    return Ok(None);
                                };
                                work = work.saturating_add(count.saturating_mul(inner.saturating_add(1)));
                            }
                            counted_regions.insert(child);
                        }
                    }
                }
                // Unknown calls, indexed updates, and unbounded nested loops retain
                // dispatch placement until their iteration cost is modeled.
                _ => return Ok(None),
            }
        }
        let children = q.required("WorkChildren", (region,))?;
        let children =
            q.0.value_to_container::<SetContainer>(children)
                .ok_or_else(|| OptimizeError::Output("invalid work regions".into()))?;
        for &child in &children.data {
            if !counted_regions.contains(&child) {
                let Some(inner) = self.region(child)? else {
                    return Ok(None);
                };
                work = work.saturating_add(inner);
            }
        }
        Ok(Some(work))
    }
}

pub(super) fn select(graph: &mut EGraph, costs: CostModel) -> Result<(), OptimizeError> {
    let query = Query(graph);
    let mut work = Work {
        query,
        operations: LookupMap::default(),
        regions: LookupMap::default(),
        active: LookupSet::default(),
    };
    query.for_function("SourceOperation", |keys, value| {
        let fields = query.enode("OperationInfo", value)?.expect("source operation info");
        work.operations
            .entry(fields[0])
            .or_default()
            .push((keys[0], graph.value_to_base::<S>(fields[1]).to_string()));
        Ok(())
    })?;
    let mut state_bytes = LookupMap::<_, Option<u64>>::default();
    query.for_each("SourceLoopCarry", |row| {
        let ty = query.required("SourceType", (row[3],))?;
        let fixed = if graph.value_to_base::<bool>(row[6]) {
            let size = query.required("SourceArraySize", (ty,))?;
            query.enode("FixedSize", size)?.is_some()
        } else {
            true
        };
        let size = query.lookup("SourceByteSize", (ty,))?.filter(|_| fixed).map(|n| work.integer(n));
        let bytes = state_bytes.entry(row[0]).or_insert(Some(0));
        *bytes = bytes.zip(size).map(|(a, b)| a.saturating_add(b));
        Ok(())
    })?;
    let mut choices = Vec::new();
    query.for_each("SourceDispatchLoop", |row| {
        let source = row[0];
        let op = query.required("SourceValueOperation", (source,))?;
        let regions = query.record("SourceLoop", "LoopInfo", (source,))?.expect("source loop info");
        let form = query.required("SourceLoopForm", (source,))?;
        let conditional = query.enode("WhileCondition", form)?.is_some();
        let local = match state_bytes.get(&source).copied().flatten() {
            Some(bytes) => match (work.region(regions[0])?, work.region(regions[1])?) {
                (Some(header), Some(body)) => {
                    costs.prefer_local_loop(header.saturating_add(body), bytes, conditional)
                }
                _ => false,
            },
            None => false,
        };
        choices.push((op, !local));
        Ok(())
    })?;
    graph.update(|mut sink| {
        for (op, dispatch) in choices {
            sink.set("LoopDispatchProfitable", (op,), dispatch)?;
        }
        Ok::<_, egglog_engine::Error>(())
    })?;
    Ok(())
}
