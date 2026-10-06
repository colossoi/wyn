//! Borrowed access to selected schedules, resources, and fused operations.
use super::{error, Facts, OptimizeError};
use crate::egglog::query::Query;
use crate::ssa::types::AtomicOp;
use crate::{types::Type, SymbolId};
use egglog_engine::{sort::S, Value};
use wyn_graph::topo_sort_by_dependencies;

impl<'a, 'source> Facts<'a, 'source> {
    pub fn stages(&self, owner: SymbolId) -> Result<Vec<Value>, OptimizeError> {
        let Some(owner) = self.program.identities.symbols.get(&owner) else {
            return Err(error("entry identity missing"));
        };
        let mut stages = Vec::new();
        self.program.graph.constructor_enodes("PhaseOwner", |row| {
            if self.integer(row.children[1]) == owner {
                stages.push(row.children[0]);
            }
        })?;
        stages.sort();
        topo_sort_by_dependencies(stages, |key, out| out.extend(self.dependencies(key)))
            .map_err(|_| error("dispatch plan contains a cycle"))
    }

    pub fn phase_operation(&self, stage: Value) -> Result<Value, OptimizeError> {
        let fields = self.enode("Stage", stage).ok_or_else(|| error("stage identity missing"))?;
        Ok(fields[0])
    }

    pub fn phase_name(&self, stage: Value) -> Result<String, OptimizeError> {
        let fields = self.enode("Stage", stage).ok_or_else(|| error("stage identity missing"))?;
        Ok(self.program.graph.value_to_base::<S>(fields[1]).to_string())
    }

    pub fn phase_extent(&self, stage: Value) -> Result<Value, OptimizeError> {
        let row = Query(&self.program.graph)
            .row("PhaseDomain", |r| r[0] == stage)?
            .ok_or_else(|| error("stage domain missing"))?;
        Ok(row[1])
    }

    pub fn phase_width(&self, stage: Value) -> Result<u32, OptimizeError> {
        let row = Query(&self.program.graph)
            .row("PhaseDomain", |r| r[0] == stage)?
            .ok_or_else(|| error("stage domain missing"))?;
        self.positive(row[2], "stage width")
    }

    pub fn dispatch_grid(&self, key: Value) -> Result<Option<(u32, u32, u32)>, OptimizeError> {
        let query = Query(&self.program.graph);
        let grid = query.required("PhaseGrid", (key,))?;
        if query.enode("AutomaticGrid", grid)?.is_some() {
            return Ok(None);
        }
        let grid = self.grid(grid)?;
        if [grid.0, grid.1, grid.2].into_iter().any(|axis| axis > 65_535) {
            return Err(error("dispatch grid axis must be in 1..=65535"));
        }
        Ok(Some(grid))
    }

    pub fn outputs(&self, owner: SymbolId) -> Result<Vec<i64>, OptimizeError> {
        let Some(owner) = self.program.identities.symbols.get(&owner) else {
            return Err(error("output owner missing"));
        };
        let mut outputs = Vec::new();
        self.program.graph.constructor_enodes("SourceOutput", |row| {
            let id = self.integer(row.children[0]);
            if self.integer(row.children[1]) == owner && self.lookup("SsaOutputBacking", (id,)).is_some() {
                outputs.push(id);
            }
        })?;
        outputs.sort();
        Ok(outputs)
    }

    pub fn output(&self, id: i64) -> Result<(Value, &'a Type, Value), OptimizeError> {
        let query = Query(&self.program.graph);
        let Some(row) = query.row("SourceOutput", |r| self.integer(r[0]) == id)? else {
            return Err(error("output declaration missing"));
        };
        let Some(ty) = self.ty(row[3]) else {
            return Err(error("output type missing"));
        };
        Ok((row[2], ty, query.required("SsaOutputBacking", (id,))?))
    }
    pub fn group(&self, value: Value) -> Option<Value> {
        self.lookup("SsaGroup", (value,))
    }
    pub fn source(&self, value: Value) -> Option<Value> {
        self.lookup("SourceOperationValue", (value,))
    }
    pub fn value_ref(&self, value: Value) -> Option<Value> {
        self.lookup("SsaValueRef", (value,))
    }
    pub fn value_read(&self, source: Value) -> Option<Value> {
        self.lookup("ValueRead", (source,))
    }
    pub fn access(&self, source: Value) -> Option<Value> {
        self.lookup("SelectedAccess", (source,))
    }
    pub fn view_extent(&self, source: Value) -> Option<Value> {
        self.lookup("LogicalExtent", (source,))
    }
    pub fn external(&self, value: Value) -> Option<Value> {
        Some(self.enode("Source", value)?[0])
    }
    pub fn live_length(&self, value: Value) -> Option<Value> {
        self.lookup("LiveLength", (value,))
    }
    pub fn capacity(&self, value: Value) -> Option<Value> {
        self.lookup("CapacityExtent", (value,))
    }
    pub fn reuse_source(&self, value: Value) -> Option<Value> {
        self.lookup("SameBacking", (value,))
    }
    pub fn domain(&self, value: Value) -> Option<Value> {
        self.lookup("SsaDomain", (value,))
    }
    pub fn backing(&self, value: Value) -> Option<Value> {
        self.lookup("Backing", (value,))
    }
    pub fn filter(&self, plan: Value) -> Result<Value, OptimizeError> {
        let Some(operation) = self.lookup("SsaFilter", (plan,)) else {
            return Err(error("filter plan has no filter operation"));
        };
        Ok(operation)
    }
    pub fn scalar_group(&self, value: Value) -> Vec<Value> {
        self.set("SsaScalarGroup", (value,))
    }
    pub fn dependencies(&self, value: Value) -> Vec<Value> {
        self.set("SsaDependencies", (value,))
    }
    pub fn member(&self, plan: Value, op: Value) -> bool {
        self.contains("PlanMember", (plan, op))
    }
    pub fn bucket_axis(&self, op: Value, axis: i64) -> Option<Value> {
        self.lookup("SourceBucketAxis", (op, axis))
    }
    pub fn slot(&self, op: Value, role: &str, index: i64) -> Option<Value> {
        self.lookup("SsaSlot", (op, role, index))
    }
    pub fn atomic(&self, op: Value) -> Option<AtomicOp> {
        let recipe = self.lookup("Plan", (op,))?;
        let value = self.enode("Atomic", recipe)?[0];
        for (name, update) in [
            ("AtomicAdd", AtomicOp::Add),
            ("AtomicAnd", AtomicOp::And),
            ("AtomicOr", AtomicOp::Or),
            ("AtomicXor", AtomicOp::Xor),
            ("AtomicCas", AtomicOp::CompareExchange),
        ] {
            if self.enode(name, value).is_some() {
                return Some(update);
            }
        }
        None
    }
    pub fn extent(&self, value: Value) -> Option<(&'static str, Vec<Value>)> {
        for name in ["Fixed", "Length", "Scalar", "Product", "Difference", "Stored"] {
            if let Some(fields) = self.enode(name, value) {
                return Some((name, fields));
            }
        }
        None
    }
    pub fn results(&self, plan: Value) -> Result<Vec<(String, i64, Value)>, OptimizeError> {
        let graph = &self.program.graph;
        let mut results = Vec::new();
        graph.constructor_enodes("PlanResult", |row| {
            if row.children[0] == plan {
                results.push((
                    graph.value_to_base::<S>(row.children[1]).to_string(),
                    self.integer(row.children[2]),
                    row.children[3],
                ));
            }
        })?;
        results.sort_by(|a, b| (&a.0, a.1).cmp(&(&b.0, b.1)));
        Ok(results)
    }
    pub fn counts(&self, plan: Value) -> Result<Vec<(i64, Value)>, OptimizeError> {
        let mut indices = Vec::new();
        self.program.graph.constructor_enodes("PlanCountSlot", |row| {
            if row.children[0] == plan {
                indices.push(self.integer(row.children[2]));
            }
        })?;
        indices.sort();
        indices
            .into_iter()
            .map(|index| {
                let Some(value) = self.lookup("SsaCountAt", (plan, index)) else {
                    return Err(error("count slot has no selected operation"));
                };
                Ok((index, value))
            })
            .collect()
    }
}
