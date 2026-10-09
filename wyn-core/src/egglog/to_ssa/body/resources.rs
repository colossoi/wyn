//! Final storage views and launch extents decoded from the selected plan.
use super::{builder_error, error, Body, OptimizeError, Typed, Value};
use crate::builtins::catalog;
use crate::op::{BinaryOperator, OpTag, PureViewSource};
use crate::ssa::types::InstKind;
use crate::types::{self, buffer_tag, Type, TypeExt, TypeName};

impl Body<'_, '_, '_> {
    pub fn workgroup_array(&mut self, id: u32, count: u32, element: &Type) -> Result<Typed, OptimizeError> {
        self.op(
            OpTag::StorageView(PureViewSource::Workgroup { id, count }),
            vec![Self::number(0), Self::number(count)],
            Self::view_type(element, types::no_buffer()),
        )
    }

    pub fn workgroup_barrier(&mut self) -> Result<(), OptimizeError> {
        self.builder.push_void_inst(InstKind::ControlBarrier).map(|_| ()).map_err(builder_error)
    }

    pub(super) fn boundary(&mut self, scope: Value, source: Value) -> Result<Typed, OptimizeError> {
        if let Some(access) = self.compiler.facts.access(source) {
            if let Some(fields) = self.compiler.facts.enode("RunCollective", access) {
                return self.collective(scope, source, fields[0], fields[1]);
            }
            let Some(fields) = self.compiler.facts.enode("StoredRead", access) else {
                return Err(error("unknown boundary access recipe"));
            };
            let access = fields[0];
            if let Some(fields) = self.compiler.facts.enode("ReadView", access) {
                return self.resource(scope, fields[0], 1);
            }
            if let Some(fields) = self.compiler.facts.enode("ReadElement", access) {
                let view = self.resource(scope, fields[0], 1)?;
                let zero = self.literal("0", &types::i32())?;
                let value = self.index(view, zero)?;
                let Some(ty) = self.compiler.facts.source_type(source).cloned() else {
                    return Err(error("stored value has no semantic type"));
                };
                return self.cast(value, &ty);
            }
            return Err(error("unknown storage read recipe"));
        }
        self.local(scope, source)
    }

    pub(super) fn local(&mut self, scope: Value, source: Value) -> Result<Typed, OptimizeError> {
        self.planned_boundary(scope, source)?;
        if let Some(access) = self.compiler.facts.access(source) {
            if let Some(fields) = self.compiler.facts.enode("RunCollective", access) {
                return self.collective(scope, source, fields[0], fields[1]);
            }
        }
        if self.compiler.facts.loops(source).is_some() {
            return self.loop_(scope, source);
        }
        if let Some((yes, no)) = self.compiler.facts.branches(source) {
            let Some(condition) = self.compiler.facts.lookup("SsaBranchCondition", (source,)) else {
                return Err(error("branch has no condition"));
            };
            let (Some(a), Some(b)) = (self.compiler.facts.result(yes), self.compiler.facts.result(no))
            else {
                return Err(error("branch has incomplete result facts"));
            };
            let layout = self.compiler.facts.value_layout(source)?;
            let condition = self.value(scope, condition)?;
            let selected = &self.compiler.program.stage.selected;
            let (Some(&a_root), Some(&b_root)) = (
                selected.roots.get(&(self.context, a)),
                selected.roots.get(&(self.context, b)),
            ) else {
                return Err(error("branch has no selected result roots"));
            };
            let shared = self.scalar_common(scope, [&[a_root], &[b_root]]);
            for term in shared {
                self.scalar_roots(scope, &[term], &mut crate::LookupSet::default())?;
            }
            return self.branch(
                scope,
                condition,
                |body| {
                    let value = body.value(yes, a)?;
                    body.materialize(value, layout)
                },
                |body| {
                    let value = body.value(no, b)?;
                    body.materialize(value, layout)
                },
                Some((yes, no)),
            );
        }
        if let Some(array) = self.compiler.facts.lookup("SourceLength", (source,)) {
            let value = self.source_length(scope, array)?;
            let Some(ty) = self.compiler.facts.source_type(source).cloned() else {
                return Err(error("length expression has no type"));
            };
            return self.cast(value, &ty);
        }
        if let Some(actual) = self.compiler.facts.alias(source) {
            return self.value(scope, actual);
        }
        if let Some((parent, index)) = self.compiler.facts.projection(source) {
            let parent = self.value(scope, parent)?;
            return self.field(parent, index);
        }
        if let Some((array, start, end)) = self.compiler.facts.slice(source) {
            let array = self.value(scope, array)?;
            let start = self.value(scope, start)?;
            let end = self.value(scope, end)?;
            let Some(size) = self
                .compiler
                .facts
                .source_type(source)
                .and_then(|ty| types::strip_existentials(ty).array_size())
            else {
                return Err(error("slice has no result size"));
            };
            return self.slice(array, start, end, size.clone());
        }
        Err(error(format!(
            "value {source:?} has no selected boundary lowering in {:?}",
            self.context
        )))
    }

    fn slice(
        &mut self,
        array: Typed,
        start: Typed,
        end: Typed,
        size: Type,
    ) -> Result<Typed, OptimizeError> {
        let Type::Constructed(TypeName::Array, mut fields) = array.ty.clone() else {
            return Err(error("slice operand has no array representation"));
        };
        let Some(dimension) = fields.get_mut(2) else {
            return Err(error("slice operand has no outer dimension"));
        };
        *dimension = size;
        self.op(
            OpTag::Intrinsic {
                id: catalog().known().slice,
                overload_idx: 0,
            },
            vec![array, start, end],
            Type::Constructed(TypeName::Array, fields),
        )
    }

    fn source_length(&mut self, scope: Value, source: Value) -> Result<Typed, OptimizeError> {
        let Some(extent) = self.compiler.facts.view_extent(source) else {
            return Err(error(format!(
                "array {source:?} has no planned view extent, projection {:?}, type {:?}",
                self.compiler.facts.projection(source),
                self.compiler.facts.source_type(source)
            )));
        };
        // A parameter's Length(self) queries its bound descriptor. Other lengths
        // follow the plan, including compacted counts and slice bounds.
        if let Some(("Length", fields)) = self.compiler.facts.extent(extent) {
            if fields[0] == source {
                let value = self.value(scope, source)?;
                return self.length(value);
            }
        }
        self.extent(scope, extent)
    }

    /// Resolve a component array without materializing it separately from its parent.
    pub(in crate::egglog::to_ssa) fn source_array(
        &mut self,
        scope: Value,
        source: Value,
    ) -> Result<(Typed, Vec<usize>), OptimizeError> {
        if let Some(actual) = self.compiler.facts.alias(source) {
            return self.source_array(scope, actual);
        }
        if let Some((parent, field)) = self.compiler.facts.projection(source) {
            if self
                .compiler
                .facts
                .source_type(parent)
                .is_some_and(|ty| types::as_soa_tuple(types::strip_existentials(ty)).is_some())
            {
                let (array, mut fields) = self.source_array(scope, parent)?;
                if fields.is_empty() && types::as_soa_tuple(&array.ty).is_some() {
                    return Ok((self.field(array, field)?, fields));
                }
                fields.push(field);
                return Ok((array, fields));
            }
        }
        Ok((self.value(scope, source)?, vec![]))
    }

    pub(in crate::egglog::to_ssa) fn resource(
        &mut self,
        scope: Value,
        resource: Value,
        access: i64,
    ) -> Result<Typed, OptimizeError> {
        if let Some(value) = self.local_resources.get(&resource) {
            return Ok(value.clone());
        }
        if let Some(owner) = self.compiler.facts.lookup("LocalStorageOwner", (resource,)) {
            let (Some(source), Some(plan)) = (
                self.compiler.facts.source(owner),
                self.compiler.facts.group(owner),
            ) else {
                return Err(error("local resource owner missing"));
            };
            self.collective(scope, source, owner, plan)?;
            let Some(value) = self.local_resources.get(&resource) else {
                return Err(error("local recipe did not allocate its resource"));
            };
            return Ok(value.clone());
        }
        let Some(backing) = self.compiler.facts.backing(resource) else {
            return Err(error(format!(
                "selected resource {resource:?} has no backing, source {:?}",
                self.compiler.facts.enode("Result", resource)
            )));
        };
        let reused = self.compiler.facts.reuse_source(resource);
        let extent = if access == 2 {
            self.compiler.facts.capacity(resource)
        } else {
            self.compiler.facts.live_length(resource)
        };
        if let Some(source) = reused {
            let Some(extent) = extent else {
                return Err(error("reused resource has no planned extent"));
            };
            let view = self.value(scope, source)?;
            if !view.ty.array_variant().is_some_and(types::is_array_variant_view) {
                return Err(error("reused resource source is not a storage view"));
            }
            let length = self.extent(scope, extent)?;
            let zero = self.literal("0", &types::i32())?;
            return self.slice(
                view,
                zero,
                length,
                Type::Constructed(TypeName::SizePlaceholder, vec![]),
            );
        }
        if self.compiler.bindings.buffer(backing)?.is_none() {
            if let Some(source) = self.compiler.facts.external(backing) {
                return self.value(scope, source);
            }
        }
        let Some((binding, element, _)) = self.compiler.bindings.buffer(backing)? else {
            return Err(error("selected resource has no allocation"));
        };
        let Some(extent) = extent else {
            return Err(error("selected resource has no planned extent"));
        };
        let length = self.extent(scope, extent)?;
        let zero = self.literal("0", &types::i32())?;
        let element = self.compiler.facts.physical_type(element, true)?;
        let ty = Body::view_type(&element, buffer_tag(binding));
        self.op(
            OpTag::StorageView(PureViewSource::Storage(binding)),
            vec![zero, length],
            ty,
        )
    }

    pub(in crate::egglog::to_ssa) fn slot(
        &mut self,
        scope: Value,
        operation: Value,
        role: &str,
        index: i64,
        access: i64,
    ) -> Result<Option<Typed>, OptimizeError> {
        let Some(resource) = self.compiler.facts.slot(operation, role, index) else {
            return Err(error(format!("missing {role} slot {index}")));
        };
        if !self.local_resources.contains_key(&resource) && self.compiler.facts.backing(resource).is_none()
        {
            return Ok(None);
        }
        self.resource(scope, resource, access).map(Some)
    }

    pub(in crate::egglog::to_ssa) fn extent(
        &mut self,
        scope: Value,
        extent: Value,
    ) -> Result<Typed, OptimizeError> {
        let Some((name, children)) = self.compiler.facts.extent(extent) else {
            return Err(error("unknown extent"));
        };
        match name {
            "Fixed" => {
                let n = self.compiler.program.graph.value_to_base::<i64>(children[0]);
                self.literal(&n.to_string(), &types::i32())
            }
            "Length" | "Scalar" => {
                let source = children[0];
                if name == "Length" {
                    self.source_length(scope, source)
                } else {
                    self.value(scope, source)
                }
            }
            "Stored" => {
                let view = self.resource(scope, children[0], 1)?;
                let zero = self.literal("0", &types::i32())?;
                self.index(view, zero)
            }
            "Product" | "Difference" => {
                let a = self.extent(scope, children[0])?;
                let b = self.extent(scope, children[1])?;
                let op =
                    if name == "Product" { BinaryOperator::Multiply } else { BinaryOperator::Subtract };
                self.binary(op, a, b)
            }
            _ => Err(error("invalid extent constructor")),
        }
    }
}
