//! Borrowed access to selected schedules, resources, and fused operations.
use super::{error, Facts, OptimizeError};
use crate::egglog::query::Query;
use crate::ssa::types::AtomicOp;
use crate::{types::Type, SymbolId};
use egglog_engine::{sort::S, Value};

pub enum OutputAction {
    Publish,
    Pack {
        predicate: Value,
        input: Value,
    },
    Updates,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum FoldOutput {
    Prefix(i64),
    Total(i64),
    PackedLength,
}

pub struct AccumulatorComponent {
    pub output: FoldOutput,
    pub source: Value,
}

impl<'a, 'source> Facts<'a, 'source> {
    pub fn phase_operation(&self, stage: Value) -> Result<Value, OptimizeError> {
        let fields = self.enode("Stage", stage).ok_or_else(|| error("stage identity missing"))?;
        Ok(fields[0])
    }

    pub fn phase_name(&self, stage: Value) -> Result<String, OptimizeError> {
        let fields = self.enode("Stage", stage).ok_or_else(|| error("stage identity missing"))?;
        Ok(self.program.graph.value_to_base::<S>(fields[1]).to_string())
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
        let mut outputs: Vec<_> = self
            .set("SourceOutputsForEntry", (owner,))
            .into_iter()
            .map(|id| self.integer(id))
            .filter(|id| self.lookup("OutputBacking", (*id,)).is_some())
            .collect();
        outputs.sort();
        Ok(outputs)
    }

    pub fn output(&self, id: i64) -> Result<(Value, &'a Type, Value), OptimizeError> {
        let query = Query(&self.program.graph);
        let Some(row) = query.record("SourceOutput", "OutputInfo", (id,))? else {
            return Err(error("output declaration missing"));
        };
        let Some(ty) = self.ty(row[2]) else {
            return Err(error("output type missing"));
        };
        Ok((row[1], ty, query.required("OutputBacking", (id,))?))
    }
    pub fn group(&self, value: Value) -> Option<Value> {
        self.lookup("SelectedPlan", (value,))
    }
    pub fn source(&self, value: Value) -> Option<Value> {
        self.lookup("SourceOperationValue", (value,))
    }
    pub fn value_ref(&self, value: Value) -> Option<Value> {
        self.lookup("ValueRef", (value,))
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
    pub fn storage_shape(&self, value: Value) -> Result<Option<Value>, OptimizeError> {
        let query = Query(&self.program.graph);
        let Some(fields) = query.record("StorageLayout", "FlatStorage", (value,))? else {
            return Ok(None);
        };
        Ok(Some(query.required("ShapeDimensions", (fields[1],))?))
    }
    pub fn capacity(&self, value: Value) -> Option<Value> {
        self.lookup("CapacityExtent", (value,))
    }
    pub fn reuse_source(&self, value: Value) -> Option<Value> {
        self.lookup("SameBacking", (value,))
    }
    pub fn domain(&self, value: Value) -> Option<Value> {
        self.lookup("InputDomain", (value,))
    }
    pub fn backing(&self, value: Value) -> Option<Value> {
        self.lookup("Backing", (value,))
    }
    pub fn action(&self, plan: Value) -> Result<OutputAction, OptimizeError> {
        let action = Query(&self.program.graph).required("PlanAction", (plan,))?;
        if self.enode("Publish", action).is_some() {
            return Ok(OutputAction::Publish);
        }
        if let Some(f) = self.enode("Pack", action) {
            return Ok(OutputAction::Pack {
                predicate: f[0],
                input: f[1],
            });
        }
        if self.lookup("PlanUpdates", (plan,)).is_some() {
            return Ok(OutputAction::Updates);
        }
        Err(error("invalid parallel output action"))
    }
    pub fn initialized_arrays(&self, owner: Value) -> Result<Vec<(Value, Value)>, OptimizeError> {
        let query = Query(&self.program.graph);
        let mut arrays = Vec::new();
        query.for_function("StorageDeclaration", |keys, declaration| {
            if let Some(result) = query.enode("Result", keys[0])? {
                if result[0] == owner {
                    if let Some(fields) = query.enode("AllocateResult", declaration)? {
                        if let Some(initial) = query.enode("CopyFrom", fields[1])? {
                            arrays.push((keys[0], initial[0]));
                        }
                    }
                }
            }
            Ok(())
        })?;
        Ok(arrays)
    }
    pub fn scalar_group(&self, value: Value) -> Vec<Value> {
        self.set("DispatchGroupMembers", (value,))
    }
    pub fn dependencies(&self, value: Value) -> Vec<Value> {
        self.set("DispatchPredecessors", (value,))
    }
    pub fn slot(&self, op: Value, role: &str, index: i64) -> Option<Value> {
        self.lookup("BufferSlot", (op, role, index))
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
        for name in [
            "Fixed",
            "Length",
            "Scalar",
            "Sum",
            "Product",
            "Difference",
            "Stored",
        ] {
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
    pub fn accumulators(&self, plan: Value) -> Result<Vec<AccumulatorComponent>, OptimizeError> {
        let owner = Query(&self.program.graph).required("Owner", (plan,))?;
        let mut components = Vec::new();
        Query(&self.program.graph).for_each("Accumulator", |row| {
            if row[0] == owner {
                let output = if let Some(fields) = self.enode("PrefixOutput", row[3]) {
                    FoldOutput::Prefix(self.integer(fields[0]))
                } else if let Some(fields) = self.enode("TotalOutput", row[3]) {
                    FoldOutput::Total(self.integer(fields[0]))
                } else if self.enode("PackedLength", row[3]).is_some() {
                    FoldOutput::PackedLength
                } else {
                    return Err(error("invalid fold output"));
                };
                components.push((
                    self.integer(row[1]),
                    AccumulatorComponent {
                        output,
                        source: row[4],
                    },
                ));
            }
            Ok(())
        })?;
        components.sort_by_key(|(slot, _)| *slot);
        components
            .into_iter()
            .enumerate()
            .map(|(expected, (slot, component))| {
                if slot != expected as i64 {
                    return Err(error("noncontiguous accumulator slots"));
                }
                if matches!(component.output, FoldOutput::Prefix(i) | FoldOutput::Total(i) if i < 0) {
                    return Err(error("invalid accumulator output slot"));
                }
                Ok(component)
            })
            .collect()
    }
}
