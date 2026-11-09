//! Final storage views and launch extents decoded from the selected plan.
use super::{builder_error, error, Body, OptimizeError, Typed, Value};
use crate::builtins::catalog;
use crate::egglog::query::Query;
use crate::op::{BinaryOperator, OpTag, PureViewSource};
use crate::ssa::types::InstKind;
use crate::types::{self, buffer_tag, Type, TypeExt, TypeName};

#[derive(Clone)]
pub(in crate::egglog::to_ssa) enum OutputTarget {
    Elements(Typed, Vec<Typed>),
    Total(Typed, Vec<Typed>),
}

impl OutputTarget {
    pub fn store(
        &self,
        body: &mut Body<'_, '_, '_>,
        coordinates: &[Typed],
        value: Typed,
    ) -> Result<(), OptimizeError> {
        let (array, indices) = self.destination(coordinates);
        body.store_at(array, &indices, value)
    }

    pub fn destination(&self, coordinates: &[Typed]) -> (Typed, Vec<Typed>) {
        let (array, prefix) = match self {
            Self::Elements(array, prefix) | Self::Total(array, prefix) => (array, prefix),
        };
        let mut indices = prefix.clone();
        if matches!(self, Self::Elements(..)) {
            indices.extend_from_slice(coordinates);
        }
        (array.clone(), indices)
    }
}

impl Body<'_, '_, '_> {
    pub fn workgroup_array(&mut self, count: u32, element: &Type) -> Result<Typed, OptimizeError> {
        let id = self.next_workgroup_array;
        self.next_workgroup_array += 1;
        self.op(
            OpTag::StorageView(PureViewSource::Workgroup {
                id,
                count: count.max(1),
            }),
            vec![Self::number(0), Self::number(count)],
            Self::view_type(
                &crate::ssa::layout::storage_value_type(element),
                types::no_buffer(),
            ),
        )
    }

    pub fn workgroup_barrier(&mut self) -> Result<(), OptimizeError> {
        self.builder.push_void_inst(InstKind::ControlBarrier).map(|_| ()).map_err(builder_error)
    }

    pub(super) fn boundary(&mut self, scope: Value, source: Value) -> Result<Typed, OptimizeError> {
        if let Some(access) = self.compiler.facts.access(source) {
            if let Some(fields) = self.compiler.facts.enode("RunCollective", access) {
                return self.collective(scope, source, fields[0]);
            }
            let Some(fields) = self.compiler.facts.enode("StoredRead", access) else {
                return Err(error("unknown boundary access recipe"));
            };
            let access = fields[0];
            if let Some(fields) = self.compiler.facts.enode("ReadView", access) {
                let view = self.resource(scope, fields[0], 1)?;
                // Scalar transport buffers have no semantic array descriptor.
                let Some(fields) = self.compiler.facts.record("ArrayView", "ViewInfo", (source,)) else {
                    return Ok(view);
                };
                let start = self.extent(scope, fields[1])?;
                let length = self.extent(scope, fields[2])?;
                let end = self.binary(BinaryOperator::Add, start.clone(), length)?;
                return self.slice(
                    view,
                    start,
                    end,
                    Type::Constructed(TypeName::SizePlaceholder, vec![]),
                );
            }
            if let Some(fields) = self.compiler.facts.enode("ReadElement", access) {
                let view = self.resource(scope, fields[0], 1)?;
                let zero = self.literal(0, &types::i32())?;
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

    pub(in crate::egglog::to_ssa) fn local(
        &mut self,
        scope: Value,
        source: Value,
    ) -> Result<Typed, OptimizeError> {
        if self.compiler.facts.contains("HostLoopAggregate", (source,)) {
            let ty = self
                .compiler
                .facts
                .source_type(source)
                .ok_or_else(|| error("loop product type missing"))?;
            let Type::Constructed(_, fields) = ty else {
                return Err(error("loop product is not an aggregate"));
            };
            let segmented = ty.is_array();
            let count = if segmented { 2 } else { fields.len() };
            let mut values = Vec::with_capacity(count);
            for index in 0..count {
                let child = self
                    .compiler
                    .facts
                    .source_field(source, index)?
                    .ok_or_else(|| error("loop product field missing"))?;
                values.push(self.value(scope, child)?);
            }
            return if segmented {
                let elements = values.remove(1);
                self.segmented_array(values.remove(0), elements)
            } else {
                self.tuple(values)
            };
        }
        if let Some(fields) = self.compiler.facts.enode("SourceDescriptorPart", source) {
            let array = self.value(scope, fields[0])?;
            let index = self.compiler.facts.unsigned(fields[1], "descriptor field")?;
            return self.field(array, index as usize);
        }
        if let Some(formal) = self.compiler.facts.enode("SourceFormal", source) {
            if let Some(actual) = self.compiler.facts.lookup("SourceCapture", (formal[0], source)) {
                return self.value(scope, actual);
            }
        }
        if let Some(fields) = self.compiler.facts.enode("SourceShape", source) {
            return self.source_length(scope, fields[0]);
        }
        if let Some(fields) = self.compiler.facts.enode("SourceOffset", source) {
            let view = self.value(scope, fields[0])?;
            let offset = self.op(
                OpTag::Project { index: 0 },
                vec![view],
                Type::Constructed(TypeName::UInt(32), vec![]),
            )?;
            return self.cast(offset, &types::i32());
        }
        self.planned_boundary(scope, source)?;
        if let Some(access) = self.compiler.facts.access(source) {
            if let Some(fields) = self.compiler.facts.enode("RunCollective", access) {
                return self.collective(scope, source, fields[0]);
            }
        }
        if self.compiler.facts.loops(source).is_some() {
            return self.loop_(scope, source);
        }
        if let Some(actual) = self.compiler.facts.alias(source) {
            return self.value(scope, actual);
        }
        if let Some(array) = self.compiler.facts.lookup("SourceLength", (source,)) {
            let value = self.source_length(scope, array)?;
            let Some(ty) = self.compiler.facts.source_type(source).cloned() else {
                return Err(error("length expression has no type"));
            };
            return self.cast(value, &ty);
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

    pub(in crate::egglog::to_ssa) fn slice(
        &mut self,
        array: Typed,
        start: Typed,
        end: Typed,
        size: Type,
    ) -> Result<Typed, OptimizeError> {
        if array.is_segmented() {
            let offsets = self.field(array.clone(), 0)?;
            let elements = self.field(array, 1)?;
            let end = self.binary(BinaryOperator::Add, end, Self::number(1))?;
            let offsets = self.slice(offsets, start, end, size)?;
            return self.segmented_array(offsets, elements);
        }
        let Type::Constructed(TypeName::Array, mut fields) = array.ty.clone() else {
            return Err(error("slice operand has no array representation"));
        };
        let Some(dimension) = fields.get_mut(2) else {
            return Err(error("slice operand has no outer dimension"));
        };
        *dimension = size;
        if array.ty.array_variant().is_some_and(types::is_array_variant_view) && fields.len() > 4 {
            let length = self.binary(BinaryOperator::Subtract, end, start.clone())?;
            let mut operands = vec![start, length];
            for axis in 1..fields.len() - 3 {
                let length = self.op(
                    OpTag::Project {
                        index: (axis + 1) as u32,
                    },
                    vec![array.clone()],
                    Type::Constructed(TypeName::UInt(32), vec![]),
                )?;
                let offset = self.cast(operands[0].clone(), &length.ty)?;
                operands[0] = self.binary(BinaryOperator::Multiply, offset, length.clone())?;
                operands.push(length);
            }
            operands.push(array);
            return self.op(
                OpTag::StorageView(PureViewSource::Inherited),
                operands,
                Type::Constructed(TypeName::Array, fields),
            );
        }
        self.op(
            OpTag::Intrinsic {
                id: catalog().known().slice,
                overload_idx: 0,
            },
            vec![array, start, end],
            Type::Constructed(TypeName::Array, fields),
        )
    }

    pub(in crate::egglog::to_ssa) fn source_length(
        &mut self,
        scope: Value,
        source: Value,
    ) -> Result<Typed, OptimizeError> {
        if let Some(value) = self.values.get(&source) {
            return self.length(value.clone());
        }
        let Some(extent) = self.compiler.facts.view_extent(source) else {
            let value = self.value(scope, source)?;
            return self.length(value);
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
            let Some(source) = self.compiler.facts.source(owner) else {
                return Err(error("local resource owner missing"));
            };
            self.collective(scope, source, owner)?;
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
            let zero = self.literal(0, &types::i32())?;
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
        let Some((binding, _element, _)) = self.compiler.bindings.buffer(backing)? else {
            return Err(error("selected resource has no allocation"));
        };
        let Some(extent) = extent else {
            return Err(error("selected resource has no planned extent"));
        };
        let element = self.compiler.bindings.buffer_element(backing)?;
        let domain = self.compiler.facts.storage_domain(backing)?;
        let segmented = domain.filter(|&shape| self.compiler.facts.enode("Segments", shape).is_some());
        let dimensions =
            match domain.and_then(|shape| self.compiler.facts.lookup("ShapeDimensions", (shape,))) {
                Some(shape) => self.compiler.facts.vector(shape)?,
                None => vec![if let Some(shape) = segmented {
                    Query(&self.compiler.program.graph).required("ShapeExtent", (shape,))?
                } else {
                    extent
                }],
            };
        let offset = self
            .compiler
            .facts
            .lookup("MetadataIndex", (resource,))
            .map(|index| self.compiler.facts.integer(index))
            .unwrap_or(0);
        let mut operands = vec![Self::number(
            u32::try_from(offset).map_err(|_| error("metadata offset overflow"))?,
        )];
        for extent in &dimensions {
            operands.push(self.extent(scope, *extent)?);
        }
        let mut fields = vec![element, types::array_variant_view()];
        fields.extend(vec![
            Type::Constructed(TypeName::SizePlaceholder, vec![]);
            dimensions.len()
        ]);
        fields.push(buffer_tag(binding));
        let array = self.op(
            OpTag::StorageView(PureViewSource::Storage(binding)),
            operands,
            Type::Constructed(TypeName::Array, fields),
        )?;
        if let Some(shape) = segmented {
            self.segmented_storage(scope, shape, array)
        } else {
            Ok(array)
        }
    }

    /// Bind a loop result to an enclosing output for the duration of emission.
    pub fn with_output<T>(
        &mut self,
        owner: Value,
        port: Value,
        target: OutputTarget,
        coordinates: &[Typed],
        emit: impl FnOnce(&mut Self) -> Result<T, OptimizeError>,
    ) -> Result<T, OptimizeError> {
        let (array, coordinates) = target.destination(coordinates);
        let (role, index, target) = if let Some(fields) = self.compiler.facts.enode("TotalPort", port) {
            ("total", fields[0], OutputTarget::Total(array, coordinates))
        } else if let Some(fields) = self.compiler.facts.enode("ArrayPort", port) {
            ("output", fields[0], OutputTarget::Elements(array, coordinates))
        } else {
            return Err(error("invalid parallel result port"));
        };
        let Some(resource) = self.compiler.facts.slot(owner, role, self.compiler.facts.integer(index))
        else {
            return Err(error("parallel result port has no buffer slot"));
        };
        let checkpoint = self.outputs.checkpoint();
        self.outputs.insert(resource, target);
        let result = emit(self);
        self.outputs.restore(checkpoint);
        result
    }

    pub fn output_target(
        &mut self,
        scope: Value,
        owner: Value,
        role: &str,
        index: i64,
    ) -> Result<Option<OutputTarget>, OptimizeError> {
        if let Some(resource) = self.compiler.facts.slot(owner, role, index) {
            if let Some(target) = self.outputs.get(&resource) {
                return Ok(Some(target.clone()));
            }
        }
        Ok(self.slot(scope, owner, role, index, 2)?.map(|array| OutputTarget::Elements(array, Vec::new())))
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
        self.optional_resource(scope, resource, access)
    }

    pub(in crate::egglog::to_ssa) fn optional_resource(
        &mut self,
        scope: Value,
        resource: Value,
        access: i64,
    ) -> Result<Option<Typed>, OptimizeError> {
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
        let Some(&term) = self.compiler.program.stage.selected.extents.get(&extent) else {
            return Err(error(format!("extent {extent:?} has no selected expression")));
        };
        self.scalar_body(scope, term)
    }
}
