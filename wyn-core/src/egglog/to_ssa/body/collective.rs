//! Lower local recipes at their selected execution scope using fused kernel algorithms.
use super::{error, Body, OptimizeError, Typed, Value};
use crate::egglog::{execution::LocalBuffer, to_ssa::kernels};
use crate::types;

impl Body<'_, '_, '_> {
    pub(super) fn collective(
        &mut self,
        scope: Value,
        source: Value,
        owner: Value,
    ) -> Result<Typed, OptimizeError> {
        let program = self.compiler.program;
        let recipe = program.stage.execution.local(owner)?;
        let width = if !self.invocation_execution && recipe.workgroup {
            let Some(width) = self.workgroup_width else {
                return Err(error("workgroup recipe requires a workgroup calling context"));
            };
            Some(width)
        } else {
            None
        };
        if width.is_some() {
            self.workgroup_barrier()?;
        }
        for buffer in &recipe.buffers {
            let LocalBuffer::Allocate {
                resource,
                capacity,
                layout,
            } = *buffer
            else {
                return Err(error("runtime-sized local result requires an output destination"));
            };
            let n = capacity.get();
            if width.is_some() {
                let element = self.compiler.facts.layout_type(layout)?;
                let array = self.workgroup_array(n, &element)?;
                self.local_resources.insert(resource, array);
                continue;
            }
            let ty = types::sized_array(n as usize, self.compiler.facts.layout_type(layout)?);
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
        for export in &recipe.exports {
            let result = export.source;
            let resource = export.resource;
            let value = self.resource(scope, resource, 1)?;
            let value = if export.array {
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
