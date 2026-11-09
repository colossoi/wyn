//! Lower local recipes at their selected execution scope using fused kernel algorithms.
use super::{error, Body, OptimizeError, Typed, Value};
use crate::egglog::to_ssa::kernels;
use crate::types;
use egglog_engine::sort::S;

impl Body<'_, '_, '_> {
    pub(super) fn collective(
        &mut self,
        scope: Value,
        source: Value,
        owner: Value,
    ) -> Result<Typed, OptimizeError> {
        let Some(execution) =
            self.compiler.facts.lookup("LocalExecution", (owner, self.execution_scope()?))
        else {
            return Err(error("local collective has not been selected"));
        };
        let width = if self.compiler.facts.enode("WorkgroupScope", execution).is_some() {
            let Some(width) = self.workgroup_width else {
                return Err(error("workgroup recipe requires a workgroup calling context"));
            };
            Some(width)
        } else if self.compiler.facts.enode("InvocationScope", execution).is_some() {
            None
        } else {
            return Err(error("invalid local execution scope"));
        };
        if width.is_some() {
            self.workgroup_barrier()?;
        }
        let allocations = self.compiler.local_buffers.get(&owner).cloned().unwrap_or_default();
        for [role, index, _element, extent] in allocations {
            let role = self.compiler.program.graph.value_to_base::<S>(role).to_string();
            let index = self.compiler.facts.integer(index);
            let Some(resource) = self.compiler.facts.slot(owner, &role, index) else {
                return Err(error("local slot missing"));
            };
            let Some(n) = self.compiler.facts.lookup("PhysicalLocalCapacity", (extent,)) else {
                return Err(error(format!("local slot {role} has no planned static capacity")));
            };
            let n = usize::try_from(self.compiler.facts.integer(n))
                .map_err(|_| error("invalid local capacity"))?;
            let Some(layout) = self.compiler.facts.lookup("LocalElementLayout", (resource,)) else {
                return Err(error(format!("local slot {role} has no planned element layout")));
            };
            if width.is_some() {
                let count = u32::try_from(n).map_err(|_| error("invalid workgroup capacity"))?;
                let element = self.compiler.facts.layout_type(layout)?;
                let array = self.workgroup_array(count, &element)?;
                self.local_resources.insert(resource, array);
                continue;
            }
            let ty = types::sized_array(n, self.compiler.facts.layout_type(layout)?);
            let value = self.local_array(ty)?;
            self.local_resources.insert(resource, value);
        }
        kernels::emit(
            self,
            scope,
            owner,
            if width.is_some() { "local" } else { "ordered" },
            width.unwrap_or(1),
        )?;
        if width.is_some() {
            self.workgroup_barrier()?;
        }
        for result in self.compiler.facts.set("LocalExports", (owner,)) {
            let Some(access) = self.compiler.facts.value_read(result) else {
                return Err(error("local export has no read recipe"));
            };
            let (resource, array) = if let Some(fields) = self.compiler.facts.enode("ReadView", access) {
                (fields[0], true)
            } else if let Some(fields) = self.compiler.facts.enode("ReadElement", access) {
                (fields[0], false)
            } else {
                return Err(error("invalid local export read"));
            };
            if !self.local_resources.contains_key(&resource)
                && self.compiler.facts.backing(resource).is_none()
            {
                continue;
            }
            let value = self.resource(scope, resource, 1)?;
            let value = if array {
                self.local_array_result(value, result)?
            } else {
                self.index(value, Self::number(0))?
            };
            self.values.insert(result, value);
        }
        if let Some(value) = self.values.get(&source) {
            return Ok(value.clone());
        }
        Err(error(format!("local recipe did not produce {source:?}")))
    }

    /// Snapshot the local buffer after its recipe has finished writing.
    fn local_array_result(&mut self, array: Typed, source: Value) -> Result<Typed, OptimizeError> {
        if self.local_arrays.contains_key(&array.value) {
            return self.snapshot_array(array);
        }
        if self.workgroup_width.is_some() && self.has_shared_storage(source) {
            return Ok(array);
        }
        let layout = self.compiler.facts.value_layout(source)?;
        self.materialize(array, layout)
    }
}
