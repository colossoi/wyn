//! Final storage views and launch extents decoded from the selected plan.
use super::{error, Body, OptimizeError, Typed, Value};
use crate::builtins::catalog;
use crate::egglog::to_ssa::interface::view_type;
use crate::op::{BinaryOperator, OpTag, PureViewSource};
use crate::types::{self, buffer_tag, Type, TypeExt, TypeName};

impl Body<'_, '_, '_> {
    pub(super) fn boundary(&mut self, scope: Value, source: Value) -> Result<Typed, OptimizeError> {
        if let Some(access) = self.compiler.plan.access(source) {
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
        if let Some(access) = self.compiler.plan.access(source) {
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
            let shared = self.compiler.placements.shared(
                self.compiler.program,
                scope,
                [&[a_root], &[b_root]],
                |source| self.values.contains_key(&source),
            )?;
            self.materializations(scope, &shared)?;
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
        let Some(extent) = self.compiler.plan.view_extent(source) else {
            return Err(error(format!(
                "array {source:?} has no planned view extent, projection {:?}, type {:?}",
                self.compiler.facts.projection(source),
                self.compiler.facts.source_type(source)
            )));
        };
        // A parameter's Length(self) queries its bound descriptor. Other lengths
        // follow the plan, including compacted counts and slice bounds.
        if let Some(("Length", fields)) = self.compiler.plan.extent(extent) {
            if self.compiler.plan.expr(fields[0]) == Some(source) {
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
            if self.compiler.facts.source_type(parent).is_some_and(|ty| types::as_soa_tuple(ty).is_some()) {
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
            let (Some(source), Some(plan)) =
                (self.compiler.plan.source(owner), self.compiler.plan.group(owner))
            else {
                return Err(error("local resource owner missing"));
            };
            self.collective(scope, source, owner, plan)?;
            let Some(value) = self.local_resources.get(&resource) else {
                return Err(error("local recipe did not allocate its resource"));
            };
            return Ok(value.clone());
        }
        let Some(backing) = self.compiler.plan.backing(resource) else {
            return Err(error(format!(
                "selected resource {resource:?} has no backing, source {:?}",
                self.compiler.facts.enode("Result", resource)
            )));
        };
        let reused = self.compiler.plan.reuse_source(resource);
        let extent = if access == 2 {
            self.compiler.plan.capacity(resource)
        } else {
            self.compiler.plan.live_length(resource)
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
        if self.compiler.plan.buffer(backing)?.is_none() {
            if let Some(source) = self.compiler.plan.external(backing) {
                return self.value(scope, source);
            }
        }
        let Some((binding, element, _)) = self.compiler.plan.buffer(backing)? else {
            return Err(error("selected resource has no allocation"));
        };
        let Some(extent) = extent else {
            return Err(error("selected resource has no planned extent"));
        };
        let length = self.extent(scope, extent)?;
        let zero = self.literal("0", &types::i32())?;
        let element = self.compiler.facts.physical_type(element, true)?;
        let ty = view_type(&element, buffer_tag(binding));
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
        let Some(resource) = self.compiler.plan.slot(operation, role, index) else {
            return Err(error(format!("missing {role} slot {index}")));
        };
        if !self.local_resources.contains_key(&resource) && self.compiler.plan.backing(resource).is_none() {
            return Ok(None);
        }
        self.resource(scope, resource, access).map(Some)
    }

    pub(in crate::egglog::to_ssa) fn extent(
        &mut self,
        scope: Value,
        extent: Value,
    ) -> Result<Typed, OptimizeError> {
        let Some((name, children)) = self.compiler.plan.extent(extent) else {
            return Err(error("unknown extent"));
        };
        match name {
            "Fixed" => {
                let n = self.compiler.program.graph.value_to_base::<i64>(children[0]);
                self.literal(&n.to_string(), &types::i32())
            }
            "Length" | "Scalar" => {
                let Some(source) = self.compiler.plan.expr(children[0]) else {
                    return Err(error("extent source is missing"));
                };
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
            "ChunkCount" => {
                let value = self.extent(scope, children[0])?;
                let width = self.compiler.program.graph.value_to_base::<i64>(children[1]);
                let extra = self.literal(&(width - 1).to_string(), &types::i32())?;
                let sum = self.binary(BinaryOperator::Add, value, extra)?;
                let width = self.literal(&width.to_string(), &types::i32())?;
                self.binary(BinaryOperator::Divide, sum, width)
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
