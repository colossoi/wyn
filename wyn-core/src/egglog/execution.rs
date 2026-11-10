//! Complete execution and local allocation decisions, frozen before scalar lowering.
use super::{output_error as error, query::Query, OptimizeError};
use crate::LookupMap;
use egglog_engine::{sort::SetContainer, EGraph, RawValues, Value};
use std::num::NonZeroU32;
use wyn_graph::topo_sort_by_dependencies;

#[derive(Default)]
pub(super) struct Execution {
    pub entries: LookupMap<i64, Vec<Value>>,
    dispatches: LookupMap<Value, Dispatch>,
    locals: LookupMap<Value, Local>,
}

pub(super) struct Dispatch {
    pub extent: Value,
    pub width: NonZeroU32,
    pub grid: Option<(u32, u32, u32)>,
}

pub(super) struct Local {
    pub workgroup: bool,
    pub buffers: Vec<LocalBuffer>,
    pub exports: Vec<LocalExport>,
}

pub(super) enum LocalBuffer {
    Allocate {
        resource: Value,
        capacity: NonZeroU32,
        layout: Value,
    },
    /// A runtime-sized result must be written into its caller's destination.
    Output(Value),
}

impl LocalBuffer {
    fn resource(&self) -> Value {
        match self {
            Self::Allocate { resource, .. } | Self::Output(resource) => *resource,
        }
    }
}

pub(super) struct LocalExport {
    pub source: Value,
    pub resource: Value,
    pub array: bool,
}

impl Execution {
    pub fn select(graph: &EGraph) -> Result<Self, OptimizeError> {
        let query = Query(graph);
        let integer = |value| graph.value_to_base::<i64>(value);
        let positive = |value, label| {
            let n = u32::try_from(integer(value)).map_err(|_| error(format!("{label} must fit u32")))?;
            let Some(n) = NonZeroU32::new(n) else {
                return Err(error(format!("{label} must be positive")));
            };
            Ok(n)
        };
        let mut domains = LookupMap::default();
        query.for_each("PhaseDomain", |row| {
            if domains.insert(row[0], (row[1], row[2])).is_some() {
                return Err(error("ambiguous PhaseDomain"));
            }
            Ok(())
        })?;
        let mut execution = Self::default();
        query.for_each("Phase", |row| {
            let Some(&(extent, width)) = domains.get(&row[0]) else {
                return Err(error("stage domain missing"));
            };
            let width = positive(width, "stage width")?;
            let grid = query.required("PhaseGrid", (row[0],))?;
            let grid = if query.enode("AutomaticGrid", grid)?.is_some() {
                None
            } else {
                let Some(fields) = query.enode("FixedGrid", grid)? else {
                    return Err(error("invalid dispatch grid"));
                };
                let [x, y, z] = [0, 1, 2].map(|i| positive(fields[i], "grid dimension"));
                let (x, y, z) = (x?.get(), y?.get(), z?.get());
                if [x, y, z].into_iter().any(|axis| axis > 65_535) {
                    return Err(error("dispatch grid axis must be in 1..=65535"));
                }
                Some((x, y, z))
            };
            execution.dispatches.insert(row[0], Dispatch { extent, width, grid });
            Ok(())
        })?;
        query.for_function("LocalExecution", |keys, scope| {
            if query.enode("InvocationScope", keys[1])?.is_none() {
                return Ok(());
            }
            if query.enode("InvocationScope", scope)?.is_none() {
                return Err(error("invocation-local execution cannot require a workgroup"));
            }
            let Some(group) = query.constructor("WorkgroupScope", RawValues(Vec::new()))? else {
                return Err(error("workgroup execution scope missing"));
            };
            let selected = query.required("LocalExecution", (keys[0], group))?;
            execution.locals.insert(
                keys[0],
                Local {
                    workgroup: selected == group,
                    buffers: Vec::new(),
                    exports: Vec::new(),
                },
            );
            Ok(())
        })?;
        query.for_each("LocalBuffer", |row| {
            let Some(local) = execution.locals.get_mut(&row[0]) else {
                return Err(error("local buffer has no execution plan"));
            };
            let resource = query.required("BufferSlot", (row[0], row[1], row[2]))?;
            if query.lookup("StaticCapacityBound", (row[4],))?.is_none()
                && query.lookup("BufferSlot", (row[0], "output", row[2]))? == Some(resource)
            {
                local.buffers.push(LocalBuffer::Output(resource));
            } else {
                let capacity = query.required("PhysicalLocalCapacity", (row[4],))?;
                let capacity = positive(capacity, "local capacity")?;
                let layout = query.required("LocalElementLayout", (resource,))?;
                local.buffers.push(LocalBuffer::Allocate {
                    resource,
                    capacity,
                    layout,
                });
            }
            Ok(())
        })?;
        query.for_function("LocalExports", |keys, exports| {
            let Some(local) = execution.locals.get_mut(&keys[0]) else {
                return Ok(());
            };
            let Some(exports) = graph.value_to_container::<SetContainer>(exports) else {
                return Err(error("invalid local export set"));
            };
            for &source in &exports.data {
                let access = query.required("ValueRead", (source,))?;
                let (resource, array) = if let Some(fields) = query.enode("ReadView", access)? {
                    (fields[0], true)
                } else {
                    let Some(fields) = query.enode("ReadElement", access)? else {
                        return Err(error("invalid local export read"));
                    };
                    (fields[0], false)
                };
                if local.buffers.iter().any(|buffer| buffer.resource() == resource)
                    || query.lookup("Backing", (resource,))?.is_some()
                {
                    local.exports.push(LocalExport {
                        source,
                        resource,
                        array,
                    });
                }
            }
            Ok(())
        })?;
        let mut dependencies = LookupMap::<_, Vec<_>>::default();
        query.for_each("DispatchDependency", |row| {
            dependencies.entry(row[1]).or_default().push(row[0]);
            Ok(())
        })?;
        query.for_each("PhaseOwner", |row| {
            execution.dispatch(row[0])?;
            execution.entries.entry(integer(row[1])).or_default().push(row[0]);
            Ok(())
        })?;
        for stages in execution.entries.values_mut() {
            stages.sort();
            *stages = topo_sort_by_dependencies(std::mem::take(stages), |key, out| {
                out.extend(dependencies.get(&key).into_iter().flatten().copied());
            })
            .map_err(|_| error("dispatch plan contains a cycle"))?;
        }
        Ok(execution)
    }

    pub fn dispatch(&self, stage: Value) -> Result<&Dispatch, OptimizeError> {
        let Some(dispatch) = self.dispatches.get(&stage) else {
            return Err(error("dispatch is outside the selected execution plan"));
        };
        Ok(dispatch)
    }

    pub fn local(&self, owner: Value) -> Result<&Local, OptimizeError> {
        let Some(local) = self.locals.get(&owner) else {
            return Err(error("local collective is outside the selected execution plan"));
        };
        Ok(local)
    }
}

#[cfg(test)]
#[path = "execution_tests.rs"]
mod tests;
