//! Expand invocation-local recipes using the same fused kernel algorithms.
use super::{builder_error, error, Body, OptimizeError, Typed, Value};
use crate::egglog::to_ssa::kernels;
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
        if !self.compiler.facts.contains("LocalCollective", (owner,)) {
            return Err(error("local collective has not been selected"));
        }
        let Some(domain) = self.compiler.facts.domain(owner) else {
            return Err(error("local domain missing"));
        };
        let mut allocations = Vec::new();
        self.compiler.program.graph.constructor_enodes("LocalBuffer", |row| {
            if row.children[0] == owner {
                allocations.push(row.children.to_vec());
            }
        })?;
        for fields in allocations {
            let role = self.compiler.program.graph.value_to_base::<S>(fields[1]).to_string();
            let index = self.compiler.facts.integer(fields[2]);
            let Some(resource) = self.compiler.facts.slot(owner, &role, index) else {
                return Err(error("local slot missing"));
            };
            let Some(n) = self.compiler.facts.lookup("PhysicalLocalCapacity", (fields[4],)) else {
                return Err(error(format!("local slot {role} has no planned static capacity")));
            };
            let n = usize::try_from(self.compiler.facts.integer(n))
                .map_err(|_| error("invalid local capacity"))?;
            let Some(element) = self.compiler.facts.ty(fields[3]) else {
                return Err(error("local element missing"));
            };
            let ty = types::sized_array(n, self.compiler.facts.physical_type(element, false)?);
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
                self.copy_array(output, initial, n)?;
            }
        }
        kernels::emit(self, scope, owner, "ordered", domain, 1)?;
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
                let Some(&place) = self.local_arrays.get(&array.value) else {
                    return Err(error("local result is not addressable"));
                };
                let value = self
                    .builder
                    .push_inst(InstKind::Load { place }, array.ty.clone())
                    .map_err(builder_error)?;
                Typed {
                    value: value.into(),
                    ty: array.ty,
                }
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
                return self.resource(scope, fields[0], 1);
            }
        }
        Err(error(format!("local recipe did not produce {source:?}")))
    }
}
