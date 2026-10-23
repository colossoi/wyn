//! Expand a selected boundary layout into representation conversions and copies.
use super::{builder_error, error, Body, OptimizeError, Typed};
use crate::op::{OpTag, PureViewSource};
use crate::ssa::types::InstKind;
use crate::types::{self, TypeExt};
use egglog_engine::{RawValues, Value};

/// Stable shared destinations for arrays crossing a control boundary with cross-lane uses.
enum BoundaryStorage {
    Value,
    Array(Typed),
    Tuple(Vec<BoundaryStorage>),
}

pub(super) struct BoundaryTarget {
    plan: Value,
    storage: BoundaryStorage,
    staged: bool,
}

impl Body<'_, '_, '_> {
    pub(in crate::egglog::to_ssa) fn materialize(
        &mut self,
        value: Typed,
        layout: Value,
    ) -> Result<Typed, OptimizeError> {
        self.materialize_component(value, layout, &[], None)
    }

    fn materialize_component(
        &mut self,
        mut value: Typed,
        layout: Value,
        mut path: &[usize],
        target: Option<&BoundaryStorage>,
    ) -> Result<Typed, OptimizeError> {
        if !value.ty.is_array() {
            for &field in path {
                value = self.field(value, field)?;
            }
            path = &[];
        }
        if let Some(BoundaryStorage::Array(output)) = target {
            let Some(width) = self.workgroup_width else {
                return Err(error("shared transfer requires a workgroup context"));
            };
            let start = self.local_id()?;
            let n = self.length(output.clone())?;
            self.copy_array_strided(output.clone(), value, n, path, start, Self::number(width))?;
            return Ok(output.clone());
        }
        if let Some(fields) = self.compiler.facts.enode("ViewLayout", layout) {
            let storage =
                self.compiler.facts.backing(fields[0]).ok_or_else(|| error("view storage missing"))?;
            let Some((binding, element, _)) = self.compiler.bindings.buffer(storage)? else {
                return Err(error("view storage allocation missing"));
            };
            let element = self.compiler.facts.physical_type(element, true)?;
            let length = self.length(value.clone())?;
            if path.is_empty() && types::array_view_buffer(&value.ty) == Some(binding) {
                return Ok(value);
            }
            let output = self.op(
                OpTag::StorageView(PureViewSource::Storage(binding)),
                vec![Self::number(0), length.clone()],
                Self::view_type(&element, types::buffer_tag(binding)),
            )?;
            self.copy_array(output.clone(), value, length, path)?;
            return Ok(output);
        }
        let ty = self.compiler.facts.layout_type(layout)?;
        if target.is_none() && path.is_empty() && value.ty == ty {
            return Ok(value);
        }
        if let Some(fields) = self.compiler.facts.enode("ArrayLayout", layout) {
            let n = self.compiler.facts.integer(fields[0]);
            let element = self.compiler.facts.layout_type(fields[1])?;
            let place = self.builder.new_place(ty.clone());
            self.builder.push_void_inst(InstKind::Alloca { result: place }).map_err(builder_error)?;
            let zero = self.literal("0", &types::i32())?;
            let length = self.literal(&n.to_string(), &types::i32())?;
            let one = self.literal("1", &types::i32())?;
            self.counted(zero, length, one, vec![], |body, index, _| {
                let item = body.index(value.clone(), index.clone())?;
                let item = body.materialize_component(item, fields[1], path, None)?;
                let slot = body.builder.new_place(element.clone());
                body.builder
                    .push_void_inst(InstKind::PlaceIndex {
                        place,
                        index: index.value,
                        result: slot,
                    })
                    .map_err(builder_error)?;
                body.builder
                    .push_void_inst(InstKind::Store {
                        place: slot,
                        value: item.value,
                    })
                    .map_err(builder_error)?;
                Ok(vec![])
            })?;
            let result =
                self.builder.push_inst(InstKind::Load { place }, ty.clone()).map_err(builder_error)?;
            return Ok(Typed {
                value: result.into(),
                ty,
            });
        }
        if let Some(fields) = self.compiler.facts.enode("TupleLayout", layout) {
            let layouts = self.compiler.facts.vector(fields[1])?;
            let mut values = Vec::new();
            for (i, layout) in layouts.into_iter().enumerate() {
                let target = match target {
                    Some(BoundaryStorage::Tuple(fields)) => fields.get(i),
                    None | Some(BoundaryStorage::Value) => None,
                    Some(BoundaryStorage::Array(_)) => return Err(error("array target for tuple layout")),
                };
                let field = if value.ty.is_array() {
                    let mut field = path.to_vec();
                    field.push(i);
                    self.materialize_component(value.clone(), layout, &field, target)?
                } else {
                    let field = self.field(value.clone(), i)?;
                    self.materialize_component(field, layout, &[], target)?
                };
                values.push(field);
            }
            let types::Type::Constructed(name, _) = ty else {
                return Err(error("tuple layout has no aggregate type"));
            };
            let ty = types::Type::Constructed(name, values.iter().map(|v| v.ty.clone()).collect());
            return self.op(OpTag::Tuple, values, ty);
        }
        if !path.is_empty() {
            return Err(error("component view requires an array layout"));
        }
        self.cast(value, &ty)
    }
}

impl Body<'_, '_, '_> {
    pub(super) fn has_shared_storage(&self, source: Value) -> bool {
        let Some(scope) = self.compiler.facts.constructor("WorkgroupScope", RawValues(Vec::new())) else {
            return false;
        };
        self.compiler.facts.contains("ValueStorage", (source, scope))
    }

    pub(super) fn boundary_target(
        &mut self,
        source: Value,
    ) -> Result<Option<BoundaryTarget>, OptimizeError> {
        if self.invocation_execution {
            return Ok(None);
        }
        let Some(path) = self.compiler.facts.constructor("StorageRoot", RawValues(Vec::new())) else {
            return Err(error("boundary storage path missing"));
        };
        let Some(plan) = self.compiler.facts.lookup("BoundaryFieldPlan", (source, path)) else {
            return Err(error("boundary representation was not selected"));
        };
        let Some(shared) = self.compiler.facts.lookup("BoundaryHasShared", (plan,)) else {
            return Err(error("boundary storage summary missing"));
        };
        if !self.compiler.program.graph.value_to_base::<bool>(shared) {
            return Ok(None);
        }
        let Some(staged) = self.compiler.facts.lookup("BoundaryTransfer", (source,)) else {
            return Err(error("boundary transfer was not selected"));
        };
        let staged = self.compiler.program.graph.value_to_base::<bool>(staged);
        let storage = self.allocate_boundary(plan)?;
        Ok(Some(BoundaryTarget {
            plan,
            storage,
            staged,
        }))
    }

    fn allocate_boundary(&mut self, plan: Value) -> Result<BoundaryStorage, OptimizeError> {
        if self.compiler.facts.enode("BoundaryValue", plan).is_some() {
            return Ok(BoundaryStorage::Value);
        }
        if let Some(fields) = self.compiler.facts.enode("BoundaryArray", plan) {
            let Some(layout) = self.compiler.facts.enode("ArrayLayout", fields[0]) else {
                return Err(error("shared array plan requires an array layout"));
            };
            let count = self.compiler.facts.unsigned(layout[0], "shared boundary capacity")?;
            let element = self.compiler.facts.layout_type(layout[1])?;
            return self.workgroup_array(count, &element).map(BoundaryStorage::Array);
        }
        if let Some(fields) = self.compiler.facts.enode("BoundaryTuple", plan) {
            let mut targets = Vec::new();
            for field in self.compiler.facts.vector(fields[1])? {
                targets.push(self.allocate_boundary(field)?);
            }
            return Ok(BoundaryStorage::Tuple(targets));
        }
        Err(error("unknown boundary representation"))
    }

    pub(super) fn materialize_boundary(
        &mut self,
        value: Typed,
        layout: Value,
        target: Option<&BoundaryTarget>,
    ) -> Result<Typed, OptimizeError> {
        let Some(target) = target else {
            return self.materialize(value, layout);
        };
        // Synchronize whole transfers, not individual tuple fields. When the
        // plan requires staging, capture all sources before any destination write.
        self.workgroup_barrier()?;
        let value = if target.staged {
            let staging = self.allocate_boundary(target.plan)?;
            let value = self.materialize_component(value, layout, &[], Some(&staging))?;
            self.workgroup_barrier()?;
            value
        } else {
            value
        };
        let value = self.materialize_component(value, layout, &[], Some(&target.storage))?;
        self.workgroup_barrier()?;
        Ok(value)
    }

    pub(super) fn rebind_boundary(
        &mut self,
        value: Typed,
        target: Option<&BoundaryTarget>,
    ) -> Result<Typed, OptimizeError> {
        let Some(target) = target else {
            return Ok(value);
        };
        self.rebind_storage(value, &target.storage)
    }

    fn rebind_storage(&mut self, value: Typed, target: &BoundaryStorage) -> Result<Typed, OptimizeError> {
        match target {
            BoundaryStorage::Value => Ok(value),
            BoundaryStorage::Array(array) => Ok(array.clone()),
            BoundaryStorage::Tuple(targets) => {
                let mut fields = Vec::new();
                for (i, target) in targets.iter().enumerate() {
                    let field = self.field(value.clone(), i)?;
                    fields.push(self.rebind_storage(field, target)?);
                }
                let types::Type::Constructed(name, _) = value.ty else {
                    return Err(error("boundary result is not an aggregate"));
                };
                let ty = types::Type::Constructed(name, fields.iter().map(|v| v.ty.clone()).collect());
                self.op(OpTag::Tuple, fields, ty)
            }
        }
    }
}
