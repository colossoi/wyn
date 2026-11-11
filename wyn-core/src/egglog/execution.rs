//! Read dispatch interfaces from the scheduled statement tree for final emission.
use super::{output_error as error, query::Query, OptimizeError};
use crate::LookupMap;
use egglog_engine::{EGraph, Value};
use std::num::NonZeroU32;

#[derive(Default)]
pub(super) struct Execution {
    pub entries: LookupMap<i64, Vec<Value>>,
    dispatches: LookupMap<Value, Dispatch>,
}

#[derive(Clone, Copy)]
pub(super) struct Dispatch {
    pub groups: [Value; 3],
    pub local_size: (u32, u32, u32),
    pub width: NonZeroU32,
    pub grid: Option<(u32, u32, u32)>,
}

impl Execution {
    pub fn select(graph: &EGraph) -> Result<Self, OptimizeError> {
        let query = Query(graph);
        let mut execution = Self::default();
        query.for_each("CurrentProgram", |program| {
            let Some(fields) = query.enode("MemoryProgram", program[0])? else {
                return Err(error("invalid memory program"));
            };
            let functions = query.vector(fields[0])?;
            for entry in query.vector(fields[2])? {
                let Some(entry) = query.enode("EntryPoint", entry)? else {
                    return Err(error("invalid memory entry"));
                };
                let mut matched = false;
                for &function in &functions {
                    let Some(function) = query.enode("DefineFunction", function)? else {
                        return Err(error("invalid memory function"));
                    };
                    if function[0] == entry[0] {
                        let owner = graph.value_to_base::<i64>(entry[1]);
                        let mut stages = Vec::new();
                        execution.body(query, function[3], &mut stages)?;
                        execution.entries.insert(owner, stages);
                        matched = true;
                    }
                }
                if !matched {
                    return Err(error("entry function is missing"));
                }
            }
            Ok(())
        })?;
        Ok(execution)
    }

    fn body(
        &mut self,
        query: Query<'_>,
        body: Value,
        stages: &mut Vec<Value>,
    ) -> Result<(), OptimizeError> {
        let Some(body) = query.enode("Block", body)? else {
            return Err(error("invalid memory body"));
        };
        for statement in query.vector(body[0])? {
            let Some(fields) = query.enode("Let", statement)? else {
                return Err(error("invalid memory statement"));
            };
            let expression = fields[3];
            let grid = if let Some(launch) = query.enode("Launch", expression)? {
                Some(launch[0])
            } else if let Some(parallel) = query.enode("Parallel", expression)? {
                query.enode("DeviceKernel", parallel[0])?.map(|kernel| kernel[0])
            } else {
                None
            };
            if let Some(grid) = grid {
                let Some(grid) = query.enode("Grid3D", grid)? else {
                    return Err(error("unresolved dispatch grid"));
                };
                let sizes = query.vector(grid[1])?;
                let [x, y, z] = sizes.as_slice() else {
                    return Err(error("dispatch needs three local dimensions"));
                };
                let [x, y, z] = [*x, *y, *z].map(|n| constant(query, n));
                let (Some(x), Some(y), Some(z)) = (x?, y?, z?) else {
                    return Err(error("local dimensions must be constant"));
                };
                let local_size = (x, y, z);
                let width = x.checked_mul(y).and_then(|n| n.checked_mul(z)).and_then(NonZeroU32::new);
                let Some(width) = width else {
                    return Err(error("invalid workgroup size"));
                };
                let groups = query.vector(grid[0])?;
                let [x, y, z] = groups.as_slice() else {
                    return Err(error("dispatch needs three group dimensions"));
                };
                let groups = [*x, *y, *z];
                let [x, y, z] = groups.map(|n| constant(query, n));
                let grid = match (x?, y?, z?) {
                    (Some(x), Some(y), Some(z)) => Some((x, y, z)),
                    _ => None,
                };
                self.dispatches.insert(
                    statement,
                    Dispatch {
                        groups,
                        local_size,
                        width,
                        grid,
                    },
                );
                stages.push(statement);
            } else if let Some(branch) = query.enode("If", expression)? {
                self.body(query, branch[2], stages)?;
                self.body(query, branch[3], stages)?;
            } else if let Some(loop_) = query.enode("Loop", expression)? {
                self.body(query, loop_[3], stages)?;
            } else if let Some(accumulator) = query.enode("WithAcc", expression)? {
                let Some(lambda) = query.enode("FunctionBody", accumulator[1])? else {
                    return Err(error("invalid accumulator body"));
                };
                self.body(query, lambda[2], stages)?;
            }
        }
        Ok(())
    }

    pub fn dispatch(&self, stage: Value) -> Result<&Dispatch, OptimizeError> {
        let Some(dispatch) = self.dispatches.get(&stage) else {
            return Err(error("statement is outside the emitted dispatch interfaces"));
        };
        Ok(dispatch)
    }
}

fn constant(query: Query<'_>, index: Value) -> Result<Option<u32>, OptimizeError> {
    let Some(fields) = query.enode("IndexConstant", index)? else {
        return Ok(None);
    };
    let value = u32::try_from(query.0.value_to_base::<i64>(fields[0]))
        .map_err(|_| error("negative or oversized grid dimension"))?;
    Ok(Some(value))
}

#[cfg(test)]
#[path = "execution_tests.rs"]
mod tests;
