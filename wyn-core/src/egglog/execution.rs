//! Read dispatch interfaces from the scheduled statement tree for final emission.
use super::{output_error as error, query::Query, OptimizeError};
use crate::ssa::layout::{storage_elem_stride, storage_value_type};
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
    pub groups: Option<[Value; 3]>,
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
                let entry = query.fields("EntryPoint", entry)?;
                let mut matched = false;
                for &definition in &functions {
                    let Some(function) = query.enode("DefineFunction", definition)? else {
                        return Err(error("invalid memory function"));
                    };
                    if function[0] == entry[0] {
                        let owner = graph.value_to_base::<i64>(entry[1]);
                        let mut stages = Vec::new();
                        execution.body(query, definition, function[3], &mut stages)?;
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
        function: Value,
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
            let grid = if let Some(parallel) = query.enode("Parallel", expression)? {
                query.enode("DeviceKernel", parallel[0])?.map(|kernel| kernel[0])
            } else {
                None
            };
            if let Some(grid) = grid {
                if query.enode("AutomaticGrid", grid)?.is_some() {
                    let parallel = query.fields("Parallel", expression)?;
                    let domain = query.fields("IterationSpace", parallel[1])?;
                    let axes = query.vector(domain[1])?;
                    if axes.is_empty() {
                        return Err(error("automatic dispatch requires an iteration axis"));
                    }
                    let body = query.fields("ParallelBlock", parallel[2])?;
                    let mut indexed_updates = false;
                    let mut state_bytes = 0u64;
                    for clause in query.vector(body[1])? {
                        indexed_updates |= query.enode("Contribute", clause)?.is_some();
                        if let Some(fold) = query.enode("Fold", clause)? {
                            let reducer = query.fields("Reduction", fold[1])?;
                            let lambda = query.fields("FunctionBody", reducer[2])?;
                            for (i, (parameter, neutral)) in query
                                .vector(lambda[0])?
                                .into_iter()
                                .zip(query.vector(reducer[1])?)
                                .enumerate()
                            {
                                let parameter = query.fields("Parameter", parameter)?;
                                let Some((_, ty)) = super::to_ssa::runtime_binding(query, parameter[0])?
                                else {
                                    return Err(error("collective state has no runtime type"));
                                };
                                let ty = super::to_ssa::reducer_state_type(
                                    query, function, reducer[2], i, neutral, ty,
                                )?;
                                let bytes = storage_elem_stride(&storage_value_type(&ty)).unwrap_or(256);
                                state_bytes = state_bytes.saturating_add(u64::from(bytes));
                            }
                        }
                    }
                    // Target 8 KiB of shared state, retaining at least one subgroup.
                    let mut width = if indexed_updates {
                        64
                    } else if state_bytes > 0 {
                        [256, 128, 64, 32]
                            .into_iter()
                            .find(|width| state_bytes.saturating_mul(u64::from(*width)) <= 8192)
                            .unwrap_or(32)
                    } else {
                        128
                    };
                    if axes.len() > 1 && state_bytes > 0 {
                        let axis = query.fields("DenseAxis", axes[axes.len() - 1])?;
                        if let Some(size) = query.lookup("StaticIndex", (function, axis[1]))? {
                            if let Ok(size) = u32::try_from(query.0.value_to_base::<i64>(size).max(0)) {
                                if let Some(size) = size.checked_next_power_of_two() {
                                    width = width.min(size.max(32));
                                }
                            }
                        }
                    }
                    let Some(width) = NonZeroU32::new(width) else {
                        return Err(error("invalid workgroup size"));
                    };
                    self.dispatches.insert(
                        statement,
                        Dispatch {
                            groups: None,
                            local_size: (width.get(), 1, 1),
                            width,
                            grid: None,
                        },
                    );
                    stages.push(statement);
                    continue;
                }
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
                        groups: Some(groups),
                        local_size,
                        width,
                        grid,
                    },
                );
                stages.push(statement);
            } else if let Some(branch) = query.enode("If", expression)? {
                self.body(query, function, branch[2], stages)?;
                self.body(query, function, branch[3], stages)?;
            } else if let Some(loop_) = query.enode("Loop", expression)? {
                self.body(query, function, loop_[3], stages)?;
            } else if let Some(accumulator) = query.enode("WithAcc", expression)? {
                let Some(lambda) = query.enode("FunctionBody", accumulator[1])? else {
                    return Err(error("invalid accumulator body"));
                };
                self.body(query, function, lambda[2], stages)?;
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
