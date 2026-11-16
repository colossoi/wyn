//! Storage choices for logical collections, before any SSA values exist.
use super::analysis::count;
use super::dataflow::ir::{self, Code};
use super::execution::{Execution, StepKind};
use super::query::Query;
use super::representation::{runtime_binding, same_value_structure, value_type};
use super::{output_error as error, OptimizeError, Optimized, Program};
use crate::binding_layout::{
    extract_io_decoration, extract_sampler_binding, extract_storage_binding, extract_storage_image_binding,
    extract_storage_image_resource, extract_texture_backing, extract_texture_binding,
    extract_texture_resource, extract_uniform_binding,
};
use crate::host::BufferLen;
use crate::interface::lowering::entry_output_arity;
use crate::interface::publish::{EntryInput, InputHost};
use crate::interface::results::output_type;
use crate::interface::{
    BindingExposure, EntryInputKind, EntryKind, EntryParamBindingKind, PushConstantSlot, StorageAccess,
};
use crate::ssa::layout::{std430_type_layout, storage_elem_stride, storage_value_type, type_byte_size};
use crate::tlc::{extract_lambda_params_ref, DefMeta};
use crate::types::{self, canonical_storage_buffer_ty, strip_existentials, Type, TypeExt, TypeName};
use crate::{LookupMap, LookupSet};
use egglog_engine::{RawValues, Value};

#[derive(Clone)]
pub(super) enum Allocation {
    Product(TypeName, Vec<Allocation>),
    Local {
        element: Type,
        capacity: Capacity,
    },
    Device {
        element: Type,
        stride: u32,
        rank: usize,
    },
}

impl Allocation {
    pub fn device(&self) -> bool {
        match self {
            Self::Device { .. } => true,
            Self::Local { .. } => false,
            Self::Product(_, fields) => fields.iter().any(Self::device),
        }
    }
}

#[derive(Clone, Copy)]
pub(super) enum Capacity {
    Fixed(usize),
    /// A reducer's specialization supplies a constant extent when emitted.
    Extent,
}

#[derive(Clone)]
pub(super) struct Collection {
    pub allocation: Allocation,
    /// Logical extent and allocation extent for each physical axis.
    pub dimensions: Vec<(Value, Value)>,
    pub capacity: Option<i64>,
}

#[derive(Default)]
pub(super) struct Storage {
    pub collections: LookupMap<Value, Collection>,
    pub parameters: LookupMap<Value, Parameters>,
    pub scratch: LookupMap<Value, Collection>,
    pub coalesce: LookupSet<Value>,
    pub snapshots: LookupSet<Value>,
    pub reuse: LookupMap<Value, Value>,
    pub transfers: LookupMap<Value, Transfer>,
    pub results: LookupMap<Value, EntryResults>,
    pub fixed_states: LookupMap<(Value, i64), Type>,
    host_controls: LookupSet<Value>,
}

impl Storage {
    pub fn select(program: &Program<'_, Optimized>, execution: &Execution) -> Result<Self, OptimizeError> {
        let query = Query(&program.graph);
        let mut storage = Self::default();
        query.for_function("EntryOrder", |keys, order| {
            let function = keys[0];
            let symbol = *program.symbols.resolve(query.0.value_to_base::<i64>(order));
            let Some(source) = program.source.defs.iter().find(|definition| definition.name == symbol)
            else {
                return Err(error("entry source metadata missing"));
            };
            let DefMeta::EntryPoint(metadata) = &source.meta else {
                return Err(error("entry declaration missing"));
            };
            let declaration = &metadata.declaration;
            let (_, parameters) = extract_lambda_params_ref(&source.body);
            let params = query.sequence("RegionParameter", query.required("FunctionRoot", function)?)?;
            let mut pc_offset = 0u32;
            let mut parameter_inputs: Vec<Vec<usize>> = Vec::new();
            let mut templates = Vec::new();
            for (index, (_, ty)) in parameters.iter().enumerate() {
                let Some(declared) = declaration.params.get(index) else {
                    return Err(error("entry parameter declaration missing"));
                };
                let binding = metadata.data.param_bindings.get(index).and_then(Option::as_ref);
                let first = templates.len();
                if let Some(EntryParamBindingKind::TupleOfViews(layout)) =
                    binding.map(|binding| &binding.kind)
                {
                    let Type::Constructed(TypeName::Tuple(_), fields) = strip_existentials(ty) else {
                        return Err(error("split parameter is not a tuple"));
                    };
                    if fields.len() != layout.len() {
                        return Err(error("split parameter arity mismatch"));
                    }
                    for (i, (ty, field)) in fields.iter().zip(layout).enumerate() {
                        let input = EntryInput {
                            name: format!("{}_{}", declared.name, i),
                            ty: canonical_storage_buffer_ty(ty),
                            host: InputHost::default(),
                            kind: EntryInputKind::Storage {
                                exposure: BindingExposure::Host(field.binding),
                                access: StorageAccess::ReadWrite,
                            },
                        };
                        templates.push((input, ty.clone()));
                    }
                } else {
                    let mut host = InputHost::default();
                    let kind = if let Some(binding) = binding
                        .map(|binding| binding.first_buffer().0)
                        .or_else(|| extract_storage_binding(declared))
                    {
                        host.length = type_byte_size(&storage_value_type(ty))
                            .map(|bytes| BufferLen::Fixed { bytes: bytes.into() });
                        EntryInputKind::Storage {
                            exposure: BindingExposure::Host(binding),
                            access: StorageAccess::ReadWrite,
                        }
                    } else if let Some(binding) = extract_uniform_binding(declared) {
                        EntryInputKind::Uniform { binding }
                    } else if let Some(binding) = extract_texture_binding(declared) {
                        host.backing = extract_texture_backing(declared);
                        host.resource = extract_texture_resource(declared);
                        EntryInputKind::Texture { binding }
                    } else if let Some(binding) = extract_sampler_binding(declared) {
                        EntryInputKind::Sampler { binding }
                    } else if let Some((binding, format, access, size)) =
                        extract_storage_image_binding(declared)
                    {
                        host.image_size = size;
                        host.resource = extract_storage_image_resource(declared);
                        EntryInputKind::StorageImage {
                            binding,
                            format,
                            access,
                        }
                    } else if declaration.entry_kind != EntryKind::Compute
                        || extract_io_decoration(declared).is_some()
                    {
                        EntryInputKind::Value {
                            decoration: extract_io_decoration(declared),
                        }
                    } else {
                        let Some((size, align)) = std430_type_layout(&storage_value_type(ty)) else {
                            return Err(error("push constant has no static size"));
                        };
                        pc_offset = pc_offset.div_ceil(align) * align;
                        let slot = PushConstantSlot {
                            offset: pc_offset,
                            size,
                        };
                        pc_offset += size;
                        EntryInputKind::PushConstant { slot }
                    };
                    templates.push((
                        EntryInput {
                            name: declared.name.clone(),
                            ty: canonical_storage_buffer_ty(ty),
                            kind,
                            host,
                        },
                        ty.clone(),
                    ));
                }
                parameter_inputs.push((first..templates.len()).collect());
            }

            for (&parameter, components) in params.iter().zip(&parameter_inputs) {
                let mut fields = components
                    .iter()
                    .map(|&index| {
                        let (input, ty) = &templates[index];
                        parameter_transfer(ty, input.storage_binding().is_some())
                    })
                    .collect::<Vec<_>>();
                let plan = if fields.len() == 1 { fields.remove(0) } else { Transfer::Product(fields) };
                storage.transfers.insert(query.name(parameter)?, plan);
            }
            storage.parameters.insert(
                function,
                Parameters {
                    templates,
                    components: parameter_inputs,
                },
            );
            Ok(())
        })?;
        for step in execution.regions.values().flatten() {
            if matches!(step.kind, StepKind::Branch | StepKind::Loop) {
                storage.host_controls.insert(step.operation);
            }
        }
        let mut operations = Vec::new();
        query.for_each("LiveOperation", |row| {
            operations.push(row[0]);
            Ok(())
        })?;
        let mut ordered = operations
            .into_iter()
            .map(|operation| {
                let output = query
                    .lookup("Output", (operation, 0_i64))?
                    .map(|name| query.value_name(name))
                    .transpose()?
                    .flatten();
                let ordinal = output.map(|name| ordinal(query, name)).transpose()?.flatten();
                Ok::<_, OptimizeError>((
                    !query.contains("ParallelOperation", operation)?,
                    ordinal,
                    operation,
                ))
            })
            .collect::<Result<Vec<_>, _>>()?;
        ordered.sort_by_key(|&(ordinary, ordinal, _)| (ordinary, ordinal));
        for (_, _, operation) in ordered {
            if !query.contains("ParallelOperation", operation)? {
                let code = ir::code(query.0, query.required("OperationPayload", operation)?);
                if matches!(code, Code::ArrayLiteral) {
                    let inputs = query.sequence("Input", operation)?;
                    let mut layouts = Vec::new();
                    for input in inputs {
                        if let Some((name, ty)) = runtime_binding(query, input)? {
                            if ty.is_array() {
                                let ty = refined_type(query, name, &mut LookupSet::default())?;
                                let mut roots = LookupSet::default();
                                if !fixed_state(&ty)
                                    && !ty.array_variant().is_some_and(types::is_array_variant_virtual)
                                {
                                    query.for_each("ResourceRoot", |row| {
                                        if row[0] == input {
                                            roots.insert(row[1]);
                                        }
                                        Ok(())
                                    })?;
                                }
                                layouts.push((ty, roots));
                            }
                        }
                    }
                    if layouts.first().is_some_and(|first| layouts.iter().any(|layout| layout != first)) {
                        storage.coalesce.insert(operation);
                    }
                }
                if matches!(code, Code::BucketScatter { .. }) {
                    let name = query.name(query.required("Input", (operation, 0_i64))?)?;
                    let source = refined_type(query, name, &mut LookupSet::default())?;
                    let Some(size) = source.array_size() else {
                        return Err(error("bucket destination has no extent"));
                    };
                    let ty = types::make_array1(
                        Type::Constructed(TypeName::UInt(32), vec![]),
                        types::array_variant_composite(),
                        size.clone(),
                        types::no_buffer(),
                    );
                    let capacity = query
                        .lookup("OuterCapacity", name)?
                        .map(|value| query.0.value_to_base::<i64>(value));
                    storage.scratch.insert(
                        operation,
                        Collection {
                            allocation: allocation(
                                &ty,
                                !local_allocation(query, execution, operation)?,
                                1,
                                capacity,
                            )?,
                            dimensions: vec![],
                            capacity,
                        },
                    );
                }
                if !matches!(code, Code::Copy | Code::Scratch | Code::Uninitialized) {
                    continue;
                }
                let pattern = query.sequence("Output", operation)?;
                let [binding] = pattern.as_slice() else {
                    return Err(error("array allocation must have one output"));
                };
                let Some((name, _)) = runtime_binding(query, *binding)? else {
                    return Err(error("array allocation output missing"));
                };
                let mut ty = refined_type(query, name, &mut LookupSet::default())?;
                let mut local = local_allocation(query, execution, operation)?;
                if matches!(code, Code::Copy) {
                    let input = query.name(query.required("Input", (operation, 0_i64))?)?;
                    local |= matches!(storage.transfer(query, input)?, Transfer::Array { copy: true });
                }
                let mut capacity = query
                    .lookup("OuterCapacity", name)?
                    .map(|capacity| query.0.value_to_base::<i64>(capacity));
                if let Some(extent) = query.lookup("LogicalOuterCount", name)? {
                    if let Some(size) = static_index(query, extent)? {
                        ty = collection_shape(&ty, &[Some(size)])?;
                    }
                }
                if capacity.is_none() && !matches!(code, Code::Uninitialized) {
                    let input = query.name(query.required("Input", (operation, 0_i64))?)?;
                    capacity = query
                        .lookup("OuterCapacity", input)?
                        .map(|capacity| query.0.value_to_base::<i64>(capacity));
                }
                let allocation = allocation(&ty, !local, 1, capacity)?;
                storage.collections.insert(
                    name,
                    Collection {
                        allocation,
                        dimensions: vec![],
                        capacity,
                    },
                );
                continue;
            }
            let domain = query.required("OperationDomain", operation)?;
            let axes = (0..count(query, "DomainRank", domain)?)
                .map(|slot| query.required("DomainExtent", (domain, slot)))
                .collect::<Result<Vec<_>, _>>()?;
            let device = execution.dispatch(operation).is_ok();
            for slot in 0..count(query, "OutputCount", operation)? {
                let role = query.required("OutputRole", (operation, slot))?;
                let total = query.enode("TotalOutput", role)?.is_some();
                if query.enode("UpdatedOutput", role)?.is_some() || (total && axes.len() <= 1) {
                    continue;
                }
                let Some((name, _)) = runtime_binding(query, query.required("Output", (operation, slot))?)?
                else {
                    continue;
                };
                let rank = axes.len() - usize::from(total);
                if rank == 0 {
                    return Err(error("collection requires an iteration axis"));
                }
                let dimensions = axes[..rank]
                    .iter()
                    .map(|&extent| static_index(query, extent))
                    .collect::<Result<Vec<_>, _>>()?;
                let ty = refined_type(query, name, &mut LookupSet::default())?;
                let capacity = query
                    .lookup("OuterCapacity", name)?
                    .map(|capacity| query.0.value_to_base::<i64>(capacity));
                let physical_rank =
                    if device && rank > 1 && dimensions[1..].contains(&None) { rank } else { 1 };
                let allocation = allocation(&ty, device, physical_rank, capacity)?;
                storage.collections.insert(
                    name,
                    Collection {
                        allocation,
                        dimensions: axes[..physical_rank].iter().map(|&extent| (extent, extent)).collect(),
                        capacity,
                    },
                );
            }
        }
        for step in execution.regions.values().flatten() {
            if let StepKind::Dispatch {
                epilogue: Some(epilogue),
                ..
            } = &step.kind
            {
                for binding in query.sequence("Output", epilogue.operation)? {
                    if let Some(collection) = storage.collections.get_mut(&query.name(binding)?) {
                        if let Some(extent) = collection.dimensions.first_mut() {
                            extent.1 = epilogue.capacity;
                        }
                    }
                }
            }
        }
        // A permission becomes one choice; competing outputs cannot claim the same input.
        let mut candidates = Vec::new();
        query.for_each("CanReuse", |row| {
            candidates.push((row[0], row[1]));
            Ok(())
        })?;
        let mut candidates = candidates
            .into_iter()
            .map(|(output, input)| {
                Ok::<_, OptimizeError>((ordinal(query, output)?, ordinal(query, input)?, output, input))
            })
            .collect::<Result<Vec<_>, _>>()?;
        candidates.sort_by_key(|&(output, input, _, _)| (output, input));
        let mut claimed = LookupSet::default();
        for (_, _, output, input) in candidates {
            if storage.reuse.contains_key(&output) || claimed.contains(&input) {
                continue;
            }
            if !matches!(storage.transfer(query, input)?, Transfer::Array { copy: false }) {
                continue;
            }
            let output_ty = value_type(query, query.required("ValueType", output)?)?;
            let input_ty = value_type(query, query.required("ValueType", input)?)?;
            if output_ty.elem_type().map(storage_value_type) != input_ty.elem_type().map(storage_value_type)
            {
                continue;
            }
            storage.reuse.insert(output, input);
            claimed.insert(input);
        }
        query.for_each("CanSnapshotCopy", |row| {
            let Some((operation, _)) = definition(query, row[0])? else {
                return Err(error("snapshot copy has no definition"));
            };
            let input = query.name(query.required("Input", (operation, 0_i64))?)?;
            if matches!(storage.transfer(query, input)?, Transfer::Array { copy: true })
                && refined_type(query, input, &mut LookupSet::default())?
                    .array_variant()
                    .is_some_and(types::is_array_variant_composite)
            {
                storage.snapshots.insert(row[0]);
            }
            Ok(())
        })?;
        let mut names = Vec::new();
        query.for_function("ValueType", |keys, _| {
            names.push(keys[0]);
            Ok(())
        })?;
        for name in names {
            storage.transfers.insert(name, storage.transfer(query, name)?);
        }
        query.for_each("LoopValueCarry", |row| {
            let operation = row[0];
            let slot = query.0.value_to_base::<i64>(row[1]);
            let mut components = 0;
            let mut preserved = true;
            query.for_each("LoopArrayComponent", |component| {
                if component[0] == operation && component[1] == row[1] {
                    components += 1;
                    preserved &= query.contains(
                        "LoopPreservesComponentLength",
                        (component[0], component[1], component[2]),
                    )?;
                }
                Ok(())
            })?;
            if query.contains("LoopPreservesOuterLength", (operation, slot))?
                || (components > 0 && preserved)
            {
                storage.fixed_states.insert(
                    (operation, slot),
                    refined_type(query, row[3], &mut LookupSet::default())?,
                );
            }
            Ok(())
        })?;
        query.for_function("EntryOrder", |keys, order| {
            let symbol = *program.symbols.resolve(query.0.value_to_base::<i64>(order));
            let Some(source) = program.source.defs.iter().find(|definition| definition.name == symbol)
            else {
                return Err(error("entry source metadata missing"));
            };
            let DefMeta::EntryPoint(metadata) = &source.meta else {
                return Err(error("entry declaration missing"));
            };
            let declaration = &metadata.declaration;
            let (body, _) = extract_lambda_params_ref(&source.body);
            let arity = entry_output_arity(declaration, &body.ty);
            let whole = declaration.buffer_demand
                || (arity == 1
                    && (declaration.source_entry.is_some()
                        || declaration.outputs.first().is_some_and(|output| output.ty.is_array())));
            let lowered = match strip_existentials(&body.ty) {
                ty if whole => std::slice::from_ref(ty),
                Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields) => fields.as_slice(),
                ty => std::slice::from_ref(ty),
            };
            let region = query.required("FunctionRoot", keys[0])?;
            let yields = query.sequence("RegionYield", region)?;
            let unpack = !whole
                && matches!(
                    strip_existentials(&body.ty),
                    Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), _)
                )
                && yields.len() == 1;
            let mut types = Vec::new();
            let mut layouts = Vec::new();
            for (slot, ty) in lowered.iter().enumerate() {
                let declared = declaration.outputs.get(slot).map_or(ty, |output| &output.ty);
                let Some(ty) = output_type(declared, ty) else {
                    return Err(error(
                        "output representation differs from its declared array nesting",
                    ));
                };
                let name = if unpack {
                    product_field(query, query.value_name(yields[0])?, slot)?
                } else {
                    yields.get(slot).map(|&name| query.value_name(name)).transpose()?.flatten()
                };
                let input_field = if unpack && name.is_none() {
                    let root = query.name(yields[0])?;
                    definition(query, root)?.is_none()
                } else {
                    false
                };
                layouts.push(storage.output(query, name, &ty, input_field, &execution.host_shapes)?);
                types.push(ty);
            }
            storage.results.insert(
                keys[0],
                EntryResults {
                    arity,
                    whole,
                    unpack,
                    types,
                    layouts,
                },
            );
            Ok(())
        })?;
        Ok(storage)
    }
}

fn ordinal(query: Query<'_>, name: Value) -> Result<Option<i64>, OptimizeError> {
    let Some(name) = query.constructor("ValueNameRef", name)? else {
        return Err(error("allocation name has no reference"));
    };
    Ok(query.lookup("NameOrdinal", name)?.map(|ordinal| query.0.value_to_base::<i64>(ordinal)))
}

pub(super) fn allocation(
    ty: &Type,
    device: bool,
    rank: usize,
    capacity: Option<i64>,
) -> Result<Allocation, OptimizeError> {
    if let Type::Constructed(name @ (TypeName::Tuple(_) | TypeName::Record(_)), fields) = ty {
        return Ok(Allocation::Product(
            name.clone(),
            fields
                .iter()
                .map(|field| allocation(field, device, rank, capacity))
                .collect::<Result<_, _>>()?,
        ));
    }
    let Some(element) = ty.elem_type() else {
        return Err(error("collection must be an array or product of arrays"));
    };
    if !device {
        let fixed = match ty.array_size() {
            Some(Type::Constructed(TypeName::Size(size), _)) => Some(*size),
            _ => None,
        };
        let capacity = capacity
            .map(|size| {
                usize::try_from(size.max(0)).map_err(|_| error("collection capacity exceeds usize"))
            })
            .transpose()?
            .or(fixed);
        let capacity = capacity.map_or(Capacity::Extent, Capacity::Fixed);
        return Ok(Allocation::Local {
            element: element.clone(),
            capacity,
        });
    }
    let mut element = ty;
    for _ in 0..rank {
        let Some(child) = element.elem_type() else {
            return Err(error("collection shape exceeds array rank"));
        };
        element = child;
    }
    let element = storage_value_type(element);
    let Some(stride) = storage_elem_stride(&element) else {
        return Err(error("collection element requires a physical representation"));
    };
    Ok(Allocation::Device {
        element,
        stride,
        rank,
    })
}

fn static_index(query: Query<'_>, extent: Value) -> Result<Option<usize>, OptimizeError> {
    query
        .lookup("StaticIndex", extent)?
        .map(|size| {
            usize::try_from(query.0.value_to_base::<i64>(size).max(0))
                .map_err(|_| error("collection dimension exceeds usize"))
        })
        .transpose()
}

fn collection_shape(ty: &Type, dimensions: &[Option<usize>]) -> Result<Type, OptimizeError> {
    let Some((size, rest)) = dimensions.split_first() else {
        return Ok(ty.clone());
    };
    if let Type::Constructed(name @ (TypeName::Tuple(_) | TypeName::Record(_)), fields) = ty {
        return Ok(Type::Constructed(
            name.clone(),
            fields.iter().map(|field| collection_shape(field, dimensions)).collect::<Result<_, _>>()?,
        ));
    }
    let Some(element) = ty.elem_type() else {
        return Err(error("collection shape exceeds array rank"));
    };
    let element = collection_shape(element, rest)?;
    let size = size
        .map(|size| Type::Constructed(TypeName::Size(size), vec![]))
        .or_else(|| ty.array_size().cloned());
    let Some(size) = size else {
        return Err(error("collection dimension missing"));
    };
    Ok(types::make_array1(
        element,
        types::array_variant_composite(),
        size,
        types::no_buffer(),
    ))
}

/// How a logical value survives a dispatch boundary. SSA emission only supplies operands.
#[derive(Clone)]
pub(super) enum Transfer {
    Product(Vec<Transfer>),
    Scalar,
    Virtual,
    Array {
        /// Materialize inline storage once; later boundaries reuse its buffer.
        copy: bool,
    },
}

impl Storage {
    pub fn transfer(&self, query: Query<'_>, name: Value) -> Result<Transfer, OptimizeError> {
        let ty = value_type(query, query.required("ValueType", name)?)?;
        self.transfer_type(query, Some(name), &ty, &mut LookupSet::default())
    }

    fn scope_transfer(
        &self,
        query: Query<'_>,
        ty: &Type,
        sources: &mut std::iter::Peekable<std::vec::IntoIter<Value>>,
        visiting: &mut LookupSet<Value>,
    ) -> Result<Transfer, OptimizeError> {
        if let Some(&source) = sources.peek() {
            let actual = value_type(query, query.required("ValueType", source)?)?;
            if same_value_structure(&actual, ty) {
                sources.next();
                return self.transfer_type(query, Some(source), &actual, visiting);
            }
        }
        if let Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields) = ty {
            return Ok(Transfer::Product(
                fields
                    .iter()
                    .map(|ty| self.scope_transfer(query, ty, sources, visiting))
                    .collect::<Result<_, _>>()?,
            ));
        }
        Err(error(
            "accumulator result transport does not match its arrays and yields",
        ))
    }

    fn transfer_type(
        &self,
        query: Query<'_>,
        name: Option<Value>,
        ty: &Type,
        visiting: &mut LookupSet<Value>,
    ) -> Result<Transfer, OptimizeError> {
        if let Some(plan) = name.and_then(|name| self.transfers.get(&name)) {
            return Ok(plan.clone());
        }
        if let Some(name) = name {
            if let Some((operation, slot)) = definition(query, name)? {
                if matches!(
                    ir::code(query.0, query.required("OperationPayload", operation)?),
                    Code::WithAcc
                ) {
                    let mut sources = Vec::new();
                    for group in 0..count(query, "AccumulatorGroupCount", operation)? {
                        for field in 0..count(query, "AccumulatorArrayCount", (operation, group))? {
                            sources.push(query.required("AccumulatorArray", (operation, group, field))?);
                        }
                    }
                    for value in query.sequence(
                        "RegionYield",
                        query.required("ControlRegion", (operation, 0_i64))?,
                    )? {
                        if let Some(name) = query.value_name(value)? {
                            sources.push(name);
                        }
                    }
                    let mut sources = sources.into_iter().peekable();
                    for output in 0..=slot {
                        let name = query.name(query.required("Output", (operation, output))?)?;
                        let ty = value_type(query, query.required("ValueType", name)?)?;
                        let plan = self.scope_transfer(query, &ty, &mut sources, visiting)?;
                        if output == slot {
                            return Ok(plan);
                        }
                    }
                }
                if matches!(
                    ir::code(query.0, query.required("OperationPayload", operation)?),
                    Code::BucketScatter { .. }
                ) {
                    let input = query.name(query.required("Input", (operation, 0_i64))?)?;
                    let Some(counts) = self.scratch.get(&operation) else {
                        return Err(error("bucket counts have no allocation"));
                    };
                    return Ok(Transfer::Product(vec![
                        self.transfer(query, input)?,
                        Transfer::Array {
                            copy: !counts.allocation.device(),
                        },
                        Transfer::Scalar,
                    ]));
                }
            }
            if let Some(row) = query.row("LoopValueCarry", |row| row[2] == name)? {
                let initial = refined_type(query, row[3], &mut LookupSet::default())?;
                return Ok(parameter_transfer(&initial, !fixed_state(&initial)));
            }
        }
        if let Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields) = ty {
            let mut children = Vec::new();
            for (slot, field) in fields.iter().enumerate() {
                let child = product_field(query, name, slot)?;
                children.push(self.transfer_type(query, child, field, visiting)?);
            }
            return Ok(Transfer::Product(children));
        }
        if !ty.is_array() {
            return Ok(Transfer::Scalar);
        }
        let Some(name) = name else {
            return Ok(Transfer::Array {
                copy: fixed_state(ty),
            });
        };
        if !visiting.insert(name) {
            return Ok(Transfer::Array { copy: false });
        }
        let result = if let Some(&input) = self.reuse.get(&name) {
            self.transfer_type(query, Some(input), ty, visiting)?
        } else if self.snapshots.contains(&name) {
            let Some((operation, _)) = definition(query, name)? else {
                return Err(error("snapshot has no copy operation"));
            };
            let input = query.name(query.required("Input", (operation, 0_i64))?)?;
            self.transfer_type(query, Some(input), ty, visiting)?
        } else if let Some(plan) = self.collections.get(&name) {
            Transfer::Array {
                copy: !plan.allocation.device(),
            }
        } else if let Some((operation, _)) = definition(query, name)? {
            let code = ir::code(query.0, query.required("OperationPayload", operation)?);
            match code {
                Code::Range => Transfer::Virtual,
                Code::ArrayLiteral => Transfer::Array { copy: true },
                Code::Use | Code::View | Code::Update { .. } => {
                    let input = query.name(query.required("Input", (operation, 0_i64))?)?;
                    self.transfer_type(query, Some(input), ty, visiting)?
                }
                Code::Parallel => Transfer::Array { copy: true },
                Code::Branch | Code::CountedLoop | Code::WhileLoop => Transfer::Array {
                    copy: !self.host_controls.contains(&operation),
                },
                Code::Index => {
                    let input = query.name(query.required("Input", (operation, 0_i64))?)?;
                    let copy = if let Some(source) = self.collections.get(&input) {
                        match &source.allocation {
                            Allocation::Local { .. } => true,
                            Allocation::Device { element, rank, .. } => *rank == 1 && element.is_array(),
                            Allocation::Product(_, _) => false,
                        }
                    } else {
                        if let Some((operation, _)) = definition(query, input)? {
                            matches!(
                                ir::code(query.0, query.required("OperationPayload", operation)?),
                                Code::ArrayLiteral
                            )
                        } else {
                            false
                        }
                    };
                    Transfer::Array { copy }
                }
                Code::Project { index } => {
                    let input = query.name(query.required("Input", (operation, 0_i64))?)?;
                    let input_ty = value_type(query, query.required("ValueType", input)?)?;
                    let source = self.transfer_type(query, Some(input), &input_ty, visiting)?;
                    let Transfer::Product(fields) = source else {
                        return Err(error("projection transport is not a product"));
                    };
                    fields.get(index).cloned().ok_or_else(|| error("projection transport field missing"))?
                }
                _ => Transfer::Array { copy: false },
            }
        } else {
            Transfer::Array { copy: false }
        };
        visiting.remove(&name);
        Ok(result)
    }
}

fn definition(query: Query<'_>, name: Value) -> Result<Option<(Value, i64)>, OptimizeError> {
    let Some(reference) = query.constructor("ValueNameRef", name)? else {
        return Ok(None);
    };
    let Some(row) = query.row("Defines", |row| row[0] == reference)? else {
        return Ok(None);
    };
    Ok(query
        .enode("OperationSite", row[1])?
        .map(|fields| (fields[0], query.0.value_to_base::<i64>(fields[1]))))
}

#[derive(Clone)]
pub(super) enum Output {
    Value,
    Array {
        rank: usize,
        leaf: Type,
        mode: ArrayOutput,
        capacity: Option<i64>,
    },
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum ArrayOutput {
    Direct,
    Count,
    Pack,
}

impl Storage {
    fn output(
        &self,
        query: Query<'_>,
        name: Option<Value>,
        ty: &Type,
        input_field: bool,
        host_shapes: &LookupSet<Value>,
    ) -> Result<Output, OptimizeError> {
        if !ty.is_array() {
            return Ok(Output::Value);
        }
        let mut leaf = ty;
        let mut rank = 0;
        let mut rectangular = true;
        while leaf.is_array() {
            let Some(dimensions) = leaf.array_dims() else {
                return Err(error("output dimensions missing"));
            };
            for dimension in dimensions {
                rectangular &= rank == 0
                    || (matches!(dimension, Type::Constructed(TypeName::Size(_), _))
                        && !leaf.array_variant().is_some_and(types::is_array_variant_bounded));
                rank += 1;
            }
            let Some(element) = leaf.elem_type() else {
                return Err(error("output element missing"));
            };
            leaf = element;
        }
        let mut exact = name.is_some_and(|name| host_shapes.contains(&name));
        let mut zero_offset = false;
        let capacity = name
            .map(|name| query.lookup("OuterCapacity", name))
            .transpose()?
            .flatten()
            .map(|value| query.0.value_to_base::<i64>(value));
        if input_field {
            exact = true;
            rectangular = true;
        }
        if let Some(name) = name {
            if let Some(collection) = self.collections.get(&name) {
                zero_offset = true;
                if let Allocation::Device {
                    rank: physical_rank, ..
                } = &collection.allocation
                {
                    rectangular |= *physical_rank == rank;
                    // An epilogue can reserve the producer's capacity but return fewer elements.
                    exact |= query.lookup("LogicalOuterCount", name)?.is_some_and(|count| {
                        collection.dimensions.first().is_some_and(|&(_, capacity)| count == capacity)
                    });
                }
            }
            if definition(query, name)?.is_none() {
                exact = true;
                rectangular = true;
            }
        }
        if let Some(name) = name {
            let physical = value_type(query, query.required("ValueType", name)?)?;
            rectangular &= physical.is_array();
        }
        let mode = if !rectangular {
            ArrayOutput::Pack
        } else if exact {
            ArrayOutput::Direct
        } else if rank == 1 && zero_offset {
            ArrayOutput::Count
        } else {
            ArrayOutput::Pack
        };
        Ok(Output::Array {
            rank,
            leaf: leaf.clone(),
            mode,
            capacity,
        })
    }
}

fn refined_type(
    query: Query<'_>,
    name: Value,
    visiting: &mut LookupSet<Value>,
) -> Result<Type, OptimizeError> {
    let mut ty = value_type(query, query.required("ValueType", name)?)?;
    if !visiting.insert(name) {
        return Ok(ty);
    }
    if let Some(row) = query.row("LoopValueCarry", |row| row[2] == name)? {
        ty = refined_type(query, row[3], visiting)?;
    }
    if let Some((operation, slot)) = definition(query, name)? {
        match ir::code(query.0, query.required("OperationPayload", operation)?) {
            Code::Index => {
                let input = query.name(query.required("Input", (operation, 0_i64))?)?;
                let mut source = refined_type(query, input, visiting)?;
                for _ in 1..count(query, "InputCount", operation)? {
                    source = indexed_type(&source)?;
                }
                ty = source;
            }
            Code::ArrayLiteral if count(query, "InputCount", operation)? > 0 => {
                let input = query.name(query.required("Input", (operation, 0_i64))?)?;
                ty = types::sized_array(
                    count(query, "InputCount", operation)? as usize,
                    refined_type(query, input, visiting)?,
                );
            }
            Code::Use | Code::Copy | Code::Scratch | Code::Update { .. } => {
                let input = query.name(query.required("Input", (operation, 0_i64))?)?;
                ty = refined_type(query, input, visiting)?;
            }
            Code::CountedLoop | Code::WhileLoop => {
                let input = query.name(query.required("EdgeArgument", (operation, 0_i64, slot))?)?;
                ty = refined_type(query, input, visiting)?;
            }
            Code::Project { index } => {
                let input = query.name(query.required("Input", (operation, 0_i64))?)?;
                if let Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields) =
                    refined_type(query, input, visiting)?
                {
                    if let Some(field) = fields.get(index) {
                        ty = field.clone();
                    }
                }
            }
            _ => {}
        }
    }
    if let Some(rank) = query.lookup("CollectionRank", name)? {
        let rank = query.0.value_to_base::<i64>(rank);
        let dimensions = (0..rank)
            .map(|axis| {
                if axis == 0 {
                    Ok(None)
                } else {
                    static_index(query, query.required("CollectionExtent", (name, axis))?)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        ty = collection_shape(&ty, &dimensions)?;
        if let Some(source) = query.lookup("CollectionElement", name)? {
            ty = collection_type(&ty, &refined_type(query, source, visiting)?, rank as usize)?;
        }
    }
    if ty.array_variant().is_some_and(types::is_array_variant_virtual) {
        visiting.remove(&name);
        return Ok(ty);
    }
    if let Type::Constructed(kind @ (TypeName::Tuple(_) | TypeName::Record(_)), fields) = &ty {
        let mut refined = Vec::new();
        for (slot, field) in fields.iter().enumerate() {
            refined.push(if let Some(value) = product_field(query, Some(name), slot)? {
                refined_type(query, value, visiting)?
            } else {
                field.clone()
            });
        }
        ty = Type::Constructed(kind.clone(), refined);
    } else if ty.is_array() {
        if let Some((operation, _)) = definition(query, name)? {
            if matches!(
                ir::code(query.0, query.required("OperationPayload", operation)?),
                Code::Range
            ) {
                let Some(element) = ty.elem_type() else {
                    return Err(error("range element missing"));
                };
                visiting.remove(&name);
                return Ok(types::make_array1(
                    element.clone(),
                    Type::Constructed(TypeName::ArrayVariantVirtual, vec![]),
                    Type::Constructed(TypeName::SizePlaceholder, vec![]),
                    types::no_buffer(),
                ));
            }
        }
        let exact = if let Some(index) = query.lookup("LogicalOuterCount", name)? {
            static_index(query, index)?
        } else {
            None
        };
        if let Some(size) = exact {
            ty = collection_shape(&ty, &[Some(size)])?;
        } else if !matches!(ty.array_size(), Some(Type::Constructed(TypeName::Size(_), _))) {
            if let Some(bound) = query.lookup("OuterCapacity", name)? {
                let size = usize::try_from(query.0.value_to_base::<i64>(bound).max(0))
                    .map_err(|_| error("array capacity exceeds usize"))?;
                let Some(element) = ty.elem_type() else {
                    return Err(error("array element missing"));
                };
                ty = types::make_array1(
                    element.clone(),
                    types::array_variant_bounded(),
                    Type::Constructed(TypeName::Size(size), vec![]),
                    types::no_buffer(),
                );
            }
        }
    }
    visiting.remove(&name);
    Ok(ty)
}

fn indexed_type(ty: &Type) -> Result<Type, OptimizeError> {
    if let Type::Constructed(name @ (TypeName::Tuple(_) | TypeName::Record(_)), fields) = ty {
        return Ok(Type::Constructed(
            name.clone(),
            fields.iter().map(indexed_type).collect::<Result<_, _>>()?,
        ));
    }
    let Some(element) = ty.elem_type() else {
        return Err(error("index exceeds physical array rank"));
    };
    Ok(element.clone())
}

fn collection_type(declared: &Type, element: &Type, axes: usize) -> Result<Type, OptimizeError> {
    if axes == 0 {
        if let (Some(Type::Constructed(TypeName::Size(size), _)), Some(declared_item), Some(actual_item)) =
            (declared.array_size(), declared.elem_type(), element.elem_type())
        {
            return Ok(types::sized_array(
                *size,
                collection_type(declared_item, actual_item, 0)?,
            ));
        }
        return Ok(element.clone());
    }
    if let (Type::Constructed(TypeName::Tuple(_), fields), Type::Constructed(TypeName::Tuple(_), values)) =
        (declared, element)
    {
        if fields.len() != values.len() {
            return Err(error("collection product arity mismatch"));
        }
        return Ok(types::tuple(
            fields
                .iter()
                .zip(values)
                .map(|(field, value)| collection_type(field, value, axes))
                .collect::<Result<_, _>>()?,
        ));
    }
    let Some(child) = declared.elem_type() else {
        return Err(error("collection rank exceeds declared rank"));
    };
    let Some(size) = declared.array_size() else {
        return Err(error("collection extent missing"));
    };
    Ok(types::make_array1(
        collection_type(child, element, axes - 1)?,
        types::array_variant_composite(),
        size.clone(),
        types::no_buffer(),
    ))
}

fn local_allocation(
    query: Query<'_>,
    execution: &Execution,
    operation: Value,
) -> Result<bool, OptimizeError> {
    let region = query.required("OperationRegion", operation)?;
    Ok(execution.local_operations.contains(&operation) || !execution.regions.contains_key(&region))
}

fn fixed_state(ty: &Type) -> bool {
    match ty {
        Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields) => {
            fields.iter().all(fixed_state)
        }
        _ if ty.array_variant().is_some_and(types::is_array_variant_virtual) => true,
        _ if ty.is_array() => {
            matches!(ty.array_size(), Some(Type::Constructed(TypeName::Size(_), _)))
                && ty.array_variant().is_some_and(types::is_array_variant_composite)
                && ty.elem_type().is_some_and(fixed_state)
        }
        _ => true,
    }
}

fn product_field(
    query: Query<'_>,
    name: Option<Value>,
    slot: usize,
) -> Result<Option<Value>, OptimizeError> {
    let Some(name) = name else {
        return Ok(None);
    };
    let Some(root) = query.constructor("RootPath", RawValues(vec![]))? else {
        return Ok(None);
    };
    let Some(path) = query.constructor("FieldPath", (root, slot as i64))? else {
        return Ok(None);
    };
    Ok(query.row("ProductComponent", |row| row[0] == name && row[1] == path)?.map(|row| row[2]))
}

#[derive(Clone)]
pub(super) struct EntryResults {
    pub arity: usize,
    pub whole: bool,
    pub unpack: bool,
    pub types: Vec<Type>,
    pub layouts: Vec<Output>,
}

#[derive(Clone)]
pub(super) struct Parameters {
    pub templates: Vec<(EntryInput, Type)>,
    pub components: Vec<Vec<usize>>,
}

fn parameter_transfer(ty: &Type, storage: bool) -> Transfer {
    match ty {
        Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields) => {
            Transfer::Product(fields.iter().map(|ty| parameter_transfer(ty, storage)).collect())
        }
        _ if ty.array_variant().is_some_and(types::is_array_variant_virtual) => Transfer::Virtual,
        _ if ty.is_array() => Transfer::Array { copy: !storage },
        _ => Transfer::Scalar,
    }
}
