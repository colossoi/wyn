//! Lower local recipes at their selected execution scope using fused kernel algorithms.
use super::{builder_error, error, Body, OptimizeError, Typed, Value};
use crate::egglog::to_ssa::kernels;
use crate::op::BinaryOperator;
use crate::ssa::types::InstKind;
use crate::types;
use egglog_engine::sort::S;

impl Body<'_, '_, '_> {
    pub(super) fn collective(
        &mut self,
        scope: Value,
        source: Value,
        owner: Value,
        plan: Value,
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
        for prerequisite in self.compiler.facts.set("LocalPrerequisites", (owner,)) {
            self.value(scope, prerequisite)?;
        }
        if width.is_some() {
            self.workgroup_barrier()?;
        }
        let Some(domain) = self.compiler.facts.domain(owner) else {
            return Err(error("local domain missing"));
        };
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
            let place = self.builder.new_place(ty.clone());
            self.builder
                .push_void_inst(InstKind::Alloca {
                    elem_ty: ty.clone(),
                    result: place,
                })
                .map_err(builder_error)?;
            let value =
                self.builder.push_inst(InstKind::Load { place }, ty.clone()).map_err(builder_error)?;
            let value = Typed {
                value: value.into(),
                ty,
            };
            self.local_arrays.insert(value.value, place);
            self.local_resources.insert(resource, value);
        }
        if let Some(destination) = self.compiler.facts.destination(owner) {
            if self.compiler.facts.contains("LocalInitialize", (owner, destination)) {
                let initial = self.value(scope, destination)?;
                let Some(output) = self.slot(scope, owner, "output", 0, 2)? else {
                    return Err(error("local destination missing"));
                };
                let n = self.length(initial.clone())?;
                if width.is_some() {
                    let lane = self.local_id()?;
                    let first = self.binary(BinaryOperator::Equal, lane, Self::number(0))?;
                    self.when(first, |body| body.copy_array(output, initial, n, &[]))?;
                    self.workgroup_barrier()?;
                } else {
                    self.copy_array(output, initial, n, &[])?;
                }
            }
        }
        kernels::emit(
            self,
            scope,
            owner,
            if width.is_some() { "local" } else { "ordered" },
            domain,
            width.unwrap_or(1),
        )?;
        if width.is_some() {
            self.workgroup_barrier()?;
        }
        let mut output_index = 0;
        for (role, index, result) in self.compiler.facts.results(plan)? {
            if role == "scan" {
                continue;
            }
            let slot = if role == "total" { "total" } else { "output" };
            // Plan result indices include totals and counts; output buffers
            // number only the array results, matching the array writes.
            let index = if role == "array" {
                let index = output_index;
                output_index += 1;
                index
            } else {
                index
            };
            let Some(array) = self.slot(scope, owner, slot, index, 1)? else {
                continue;
            };
            let value = if role == "total" {
                let zero = self.literal("0", &types::i32())?;
                self.index(array, zero)?
            } else {
                self.local_array_result(array, result)?
            };
            self.values.insert(result, value);
        }
        if let Some(value) = self.values.get(&source) {
            return Ok(value.clone());
        }
        if let Some(access) = self.compiler.facts.value_read(source) {
            if let Some(fields) = self.compiler.facts.enode("ReadElement", access) {
                let array = self.resource(scope, fields[0], 1)?;
                let zero = self.literal("0", &types::i32())?;
                return self.index(array, zero);
            }
            if let Some(fields) = self.compiler.facts.enode("ReadView", access) {
                let array = self.resource(scope, fields[0], 1)?;
                return self.local_array_result(array, source);
            }
        }
        Err(error(format!("local recipe did not produce {source:?}")))
    }

    /// Snapshot the local buffer after its recipe has finished writing.
    fn local_array_result(&mut self, array: Typed, source: Value) -> Result<Typed, OptimizeError> {
        let Some(&place) = self.local_arrays.get(&array.value) else {
            if self.workgroup_width.is_some() && self.has_shared_storage(source) {
                return Ok(array);
            }
            let layout = self.compiler.facts.value_layout(source)?;
            return self.materialize(array, layout);
        };
        let value =
            self.builder.push_inst(InstKind::Load { place }, array.ty.clone()).map_err(builder_error)?;
        Ok(Typed {
            value: value.into(),
            ty: array.ty,
        })
    }
}
