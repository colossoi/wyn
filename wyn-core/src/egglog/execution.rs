//! Read selected dispatch interfaces from flat operation and region cells.
use super::query::Query;
use super::{output_error as error, to_ssa, OptimizeError};
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
        query.for_function("EntryOrder", |keys, owner| {
            let root = query.required("FunctionRoot", keys[0])?;
            let mut stages = Vec::new();
            execution.region(query, root, &mut stages)?;
            execution.entries.insert(graph.value_to_base::<i64>(owner), stages);
            Ok(())
        })?;
        Ok(execution)
    }

    fn region(
        &mut self,
        query: Query<'_>,
        region: Value,
        stages: &mut Vec<Value>,
    ) -> Result<(), OptimizeError> {
        for operation in query.operations(region)? {
            let selected = query.lookup("OperationExecution", operation)?;
            let grid =
                selected.map(|selected| query.enode("DeviceKernel", selected)).transpose()?.flatten();
            if let Some(grid) = grid {
                let width = u32::try_from(
                    query.0.value_to_base::<i64>(query.required("OperationWidth", operation)?),
                )
                .ok()
                .and_then(NonZeroU32::new);
                let Some(width) = width else {
                    return Err(error("invalid workgroup size"));
                };
                let (groups, local_size, grid) = if query.enode("AutomaticGrid", grid[0])?.is_some() {
                    let domain = query.required("OperationDomain", operation)?;
                    if super::analysis::count(query, "DomainRank", domain)? == 0 {
                        return Err(error("automatic dispatch requires an iteration axis"));
                    }
                    (None, (width.get(), 1, 1), None)
                } else {
                    let fields = query.fields("Grid3D", grid[0])?;
                    let groups = triple(query, fields[0])?;
                    let local = triple(query, fields[1])?;
                    let [x, y, z] = local.map(|index| constant(query, index));
                    let (Some(x), Some(y), Some(z)) = (x?, y?, z?) else {
                        return Err(error("local dimensions must be constant"));
                    };
                    if x.checked_mul(y).and_then(|n| n.checked_mul(z)) != Some(width.get()) {
                        return Err(error("invalid workgroup size"));
                    }
                    let [gx, gy, gz] = groups.map(|index| constant(query, index));
                    let grid = match (gx?, gy?, gz?) {
                        (Some(x), Some(y), Some(z)) => Some((x, y, z)),
                        _ => None,
                    };
                    if grid.is_some_and(|(x, y, z)| [x, y, z].into_iter().any(|n| n == 0 || n > 65_535)) {
                        return Err(error("dispatch grid axis must be in 1..=65535"));
                    }
                    (Some(groups), (x, y, z), grid)
                };
                self.dispatches.insert(
                    operation,
                    Dispatch {
                        groups,
                        local_size,
                        width,
                        grid,
                    },
                );
                stages.push(operation);
                continue;
            }
            // Nested execution is explicit; callbacks and reducer regions stay local.
            let mut children = Vec::new();
            query.for_function("ControlRegion", |keys, child| {
                if keys[0] == operation {
                    children.push((query.0.value_to_base::<i64>(keys[1]), child));
                }
                Ok(())
            })?;
            children.sort_by_key(|&(slot, _)| slot);
            for (_, child) in children {
                self.region(query, child, stages)?;
            }
        }
        Ok(())
    }

    pub fn dispatch(&self, stage: Value) -> Result<&Dispatch, OptimizeError> {
        let Some(dispatch) = self.dispatches.get(&stage) else {
            return Err(error("operation is outside the emitted dispatch interfaces"));
        };
        Ok(dispatch)
    }
}

/// Apply the portable shared-state policy before storing the choice in egglog.
pub(super) fn width(query: Query<'_>, operation: Value, selected: Value) -> Result<u32, OptimizeError> {
    let Some(grid) = query.enode("DeviceKernel", selected)? else {
        return Ok(1);
    };
    if let Some(grid) = query.enode("Grid3D", grid[0])? {
        let [x, y, z] = triple(query, grid[1])?.map(|index| constant(query, index));
        let (Some(x), Some(y), Some(z)) = (x?, y?, z?) else {
            return Err(error("local dimensions must be constant"));
        };
        let Some(width) = x.checked_mul(y).and_then(|n| n.checked_mul(z)).filter(|&n| n > 0) else {
            return Err(error("invalid workgroup size"));
        };
        return Ok(width);
    }
    let domain = query.required("OperationDomain", operation)?;
    let rank = super::analysis::count(query, "DomainRank", domain)?;
    if rank == 0 {
        return Err(error("automatic dispatch requires an iteration axis"));
    }
    let mut indexed_updates = false;
    let mut state_bytes = 0u64;
    for clause in 0..super::analysis::count(query, "ResultCount", operation)? {
        let kind = query.required("ResultKindOf", (operation, clause))?;
        indexed_updates |= query.enode("Contributed", kind)?.is_some();
        if query.enode("Folded", kind)?.is_some() {
            let reducer = query.required("ResultReducer", (operation, clause))?;
            let params = query.sequence("RegionParameter", reducer)?;
            let count = super::analysis::count(query, "ResultNeutralCount", (operation, clause))?;
            let neutral = query.indexed(
                "ResultNeutral",
                &[operation, query.0.base_to_value(clause)],
                count,
            )?;
            for (i, (parameter, neutral)) in params.into_iter().zip(neutral).enumerate() {
                let Some((_, ty)) = to_ssa::runtime_binding(query, parameter)? else {
                    return Err(error("collective state has no runtime type"));
                };
                let ty = to_ssa::reducer_state_type(query, operation, reducer, i, neutral, ty)?;
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
    if rank > 1 && state_bytes > 0 {
        let extent = query.required("DomainExtent", (domain, rank - 1))?;
        if let Some(size) = query.lookup("StaticIndex", extent)? {
            if let Ok(size) = u32::try_from(query.0.value_to_base::<i64>(size).max(0)) {
                if let Some(size) = size.checked_next_power_of_two() {
                    width = width.min(size.max(32));
                }
            }
        }
    }
    Ok(width)
}

fn triple(query: Query<'_>, vector: Value) -> Result<[Value; 3], OptimizeError> {
    query.vector(vector)?.try_into().map_err(|_| error("dispatch requires three dimensions"))
}

fn constant(query: Query<'_>, index: Value) -> Result<Option<u32>, OptimizeError> {
    let value = if let Some(fields) = query.enode("IndexConstant", index)? {
        Some(fields[0])
    } else {
        query.lookup("StaticIndex", index)?
    };
    let Some(value) = value else {
        return Ok(None);
    };
    let value = u32::try_from(query.0.value_to_base::<i64>(value))
        .map_err(|_| error("negative or oversized grid dimension"))?;
    Ok(Some(value))
}

#[cfg(test)]
#[path = "execution_tests.rs"]
mod tests;
