//! Lower public declarations and bind their values to the final memory signature.
use super::{builder_error, error, host, Body, Compiler, OptimizeError, Typed};
use crate::binding_layout::{
    extract_io_decoration, extract_sampler_binding, extract_storage_binding, extract_storage_image_binding,
    extract_storage_image_resource, extract_texture_backing, extract_texture_binding,
    extract_texture_resource, extract_uniform_binding,
};
use crate::builtins::catalog;
use crate::egglog::from_tlc::ragged_parts;
use crate::egglog::query::Query;
use crate::flow::ExecutionModel;
use crate::host::{
    BufferLen, BufferSource, ComputePipeline, ComputeStage, DispatchLoop, DispatchLoopControl,
    DispatchSize, Expr, FrameGraph, LoopCarry, ModuleInterface, Pipeline, ScalarExpr, ScalarSource,
    ScalarType,
};
use crate::host::{
    DrawBufferRef, GraphicsInvocation, GraphicsPipeline, GraphicsStage, ResultKind, ResultShape,
    ShaderStage, SourceResultBinding,
};
use crate::interface::lowering::build_entry_outputs;
use crate::interface::publish::{
    reconcile_storage_binding_access, EntryInput, EntryOutput, EntryPoint, InputHost,
    ModuleInterfacePublish, StorageBindingDecl, StorageHost,
};
use crate::interface::results::result_layout_with_storage;
use crate::interface::DrawBufferOperand;
use crate::interface::{
    BindingExposure, EntryInputKind, EntryKind, EntryParamBindingKind, PushConstantSlot, StorageAccess,
    StorageLayout, StorageRole,
};
use crate::op::{BinaryOperator, OpTag, PureViewSource};
use crate::ssa::context::BackendGlobal;
use crate::ssa::layout::{block_layout, storage_elem_stride, storage_value_type, type_byte_size};
use crate::ssa::stage::Elaborated;
use crate::ssa::types::InstKind;
use crate::tlc::{extract_lambda_params_ref, DefMeta};
use crate::types::{
    self, buffer_tag, canonical_storage_buffer_ty, strip_existentials, Type, TypeExt, TypeName,
};
use crate::LookupSet;
use crate::{BindingRef, CodegenTarget, LookupMap, ResourceAccess};
use egglog_engine::sort::SetContainer;
use egglog_engine::Value;
use wyn_base::IdSource;

pub(super) fn lower(
    compiler: &mut Compiler<'_, '_>,
    roots: Vec<Value>,
    target: CodegenTarget,
) -> Result<Elaborated, OptimizeError> {
    let mut entries = Vec::new();
    let mut source_names = LookupMap::default();
    let mut pipeline = ModuleInterface::default();
    let mut associations = Vec::new();
    let mut ids = IdSource::new();
    let mut origins = LookupMap::default();
    let mut graphics = LookupMap::default();
    let mut invocations = Vec::new();
    for root in roots {
        let query = super::Query(&compiler.program.graph);
        let entry = query.fields("EntryPoint", root)?;
        let symbol = *compiler.program.symbols.resolve(query.0.value_to_base::<i64>(entry[1]));
        let Some(source) = compiler.program.source.defs.iter().find(|definition| definition.name == symbol)
        else {
            return Err(error("entry source metadata missing"));
        };
        let DefMeta::EntryPoint(metadata) = &source.meta else {
            return Err(error("entry declaration missing"));
        };
        let declaration = &metadata.declaration;
        let (source_body, parameters) = extract_lambda_params_ref(&source.body);
        let Some(&function) = compiler.definitions.get(&entry[0]) else {
            return Err(error("entry memory function missing"));
        };
        let fields = query.fields("DefineFunction", function)?;
        let block = query.fields("Block", fields[3])?;
        let params = query
            .vector(fields[1])?
            .into_iter()
            .map(|param| Ok(query.fields("Parameter", param)?[0]))
            .collect::<Result<Vec<_>, OptimizeError>>()?;
        let mut pc_offset = 0;
        let mut parameter_inputs = Vec::new();
        let mut templates = Vec::new();
        for (index, (_, ty)) in parameters.iter().enumerate() {
            let Some(declared) = declaration.params.get(index) else {
                return Err(error("entry parameter declaration missing"));
            };
            let binding = metadata.data.param_bindings.get(index).and_then(Option::as_ref);
            let first = templates.len();
            if let Some(EntryParamBindingKind::TupleOfViews(layout)) = binding.map(|binding| &binding.kind)
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
                    let Some(size) = type_byte_size(&storage_value_type(ty)) else {
                        return Err(error("push constant has no static size"));
                    };
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

        let next = templates
            .iter()
            .filter_map(|(input, _)| input.descriptor_binding())
            .filter(|binding| binding.set == 0)
            .map(|binding| binding.binding + 1)
            .max()
            .unwrap_or(0);
        let lane_dependent = query
            .lookup("LaneDependentFunction", (function,))?
            .is_some_and(|value| query.0.value_to_base::<bool>(value));
        let invocation = if lane_dependent && declaration.entry_kind == EntryKind::Compute {
            let size = declaration.compute_dispatch.map_or(
                DispatchSize::Fixed {
                    x: 1,
                    y: 1,
                    z: 1,
                    explicit: false,
                },
                |grid| DispatchSize::Fixed {
                    x: grid.x,
                    y: grid.y,
                    z: grid.z,
                    explicit: true,
                },
            );
            ((128, 1, 1), size)
        } else {
            (
                (1, 1, 1),
                DispatchSize::Fixed {
                    x: 1,
                    y: 1,
                    z: 1,
                    explicit: true,
                },
            )
        };
        let mut body = Body::new(compiler, function, vec![], types::unit());
        body.generated = Some(GeneratedBuffers {
            owner: declaration.name.clone(),
            next,
            inputs: vec![],
            declarations: vec![],
            memories: LookupMap::default(),
            scalar_slots: LookupMap::default(),
        });
        let mut emission = EntryEmission {
            body,
            templates,
            params,
            parameter_inputs,
            inputs: vec![],
            parameters: LookupSet::default(),
            saved: LookupMap::default(),
            name: declaration.name.clone(),
            kind: declaration.entry_kind,
            span: source_body.span,
            pipeline: pipeline.pipelines.len(),
            invocation,
            pending: false,
            stages: vec![],
            entries: vec![],
            loops: vec![],
        };
        emission.restart()?;
        let live = live_names(emission.body.query(), &[], block[1])?;
        emission.statements(block[0], &live)?;
        let output_arity = crate::interface::lowering::entry_output_arity(declaration, &source_body.ty);
        let capacities = if output_arity == 0 {
            vec![]
        } else {
            output_capacities(&emission.body, &source_body.ty, block[1])?
        };
        let Some(generated) = emission.body.generated.as_ref() else {
            return Err(error("entry buffers missing"));
        };
        let mut bindings = IdSource::new();
        while bindings.peek_id() < generated.next {
            bindings.next_id();
        }
        let publish_direct = declaration.entry_kind == EntryKind::Compute
            && declaration.outputs.iter().all(|output| output.attribute.is_none());
        let physical = if output_arity == 0 {
            source_body.ty.clone()
        } else {
            output_type(&emission.body, &source_body.ty, block[1], publish_direct)?
        };
        let destinations = if publish_direct {
            output_destinations(&emission.body, &source_body.ty, block[1])?
        } else {
            vec![]
        };
        let outputs = build_entry_outputs(
            declaration,
            &physical,
            &capacities,
            &destinations,
            declaration.entry_kind == EntryKind::Compute,
            &mut bindings,
        )
        .map_err(|failure| error(failure.to_string()))?;
        if let Some(generated) = emission.body.generated.as_mut() {
            generated.next = bindings.peek_id();
        }
        if declaration.entry_kind == EntryKind::Compute && !declaration.buffer_demand {
            let types = match strip_existentials(&source_body.ty) {
                Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields) => fields.as_slice(),
                ty => std::slice::from_ref(ty),
            };
            let mut results = emission.body.query().vector(block[1])?.into_iter();
            for (index, (ty, output)) in types.iter().zip(&outputs).enumerate() {
                let Some(binding) = output.storage_binding() else {
                    return Err(error("source result has no output buffer"));
                };
                let (logical, shape) = if ty.is_array() {
                    let (name, offsets, payload) = output_array_parts(&emission.body, ty, &mut results)?;
                    let (_, _, layout) = emission.body.array_descriptor(name)?;
                    let dimensions = emission
                        .body
                        .query()
                        .vector(layout[1])?
                        .into_iter()
                        .map(|dim| {
                            let dim = emission.body.query().fields("Dim", dim)?;
                            host::index(emission.body.query(), &emission.body.host_values, dim[1])
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    if offsets.is_empty() {
                        let mut element = ty;
                        while element.is_array() {
                            let Some(next) = element.elem_type() else {
                                return Err(error("result array element missing"));
                            };
                            element = next;
                        }
                        (
                            types::view_array_with_size(
                                element,
                                Type::Constructed(TypeName::SizePlaceholder, vec![]),
                                types::no_buffer(),
                            ),
                            ResultShape::Rectangular(dimensions),
                        )
                    } else {
                        let Some(elements) = dimensions.first().cloned() else {
                            return Err(error("ragged payload has no extent"));
                        };
                        let offsets = offsets
                            .into_iter()
                            .map(|name| publish_offset_array(&mut emission.body, name))
                            .collect::<Result<Vec<_>, _>>()?;
                        (payload, ResultShape::Segmented { offsets, elements })
                    }
                } else {
                    skip_result_values(ty, &mut results)?;
                    (ty.clone(), ResultShape::Rectangular(vec![]))
                };
                let (name, kind) = match strip_existentials(&source_body.ty) {
                    Type::Constructed(TypeName::Record(names), _) => {
                        (names.0[index].clone(), ResultKind::RecordField)
                    }
                    Type::Constructed(TypeName::Tuple(_), _) => {
                        (format!("result_{index}"), ResultKind::TupleField)
                    }
                    _ => (declaration.name.clone(), ResultKind::Value),
                };
                let default = [crate::interface::SourceResult { index, name, kind }];
                let results = if let Some(source) = &declaration.source_entry {
                    let Some(results) = source.outputs.get(index) else {
                        return Err(error("source output route missing"));
                    };
                    results.as_slice()
                } else {
                    &default
                };
                let physical_layout = storage_value_type(&logical);
                for result in results {
                    pipeline.source_results.push(SourceResultBinding {
                        entry: declaration
                            .source_entry
                            .as_ref()
                            .map_or(&declaration.name, |source| &source.name)
                            .clone(),
                        name: result.name.clone(),
                        kind: result.kind.clone(),
                        layout: result_layout_with_storage(&logical, &physical_layout),
                        shape: shape.clone(),
                        result: result.index,
                        pipeline_index: emission.pipeline,
                        set: binding.set,
                        binding: binding.binding,
                    });
                }
            }
        }
        let mut results = emission.body.query().vector(block[1])?.into_iter();
        emission.pending |= write_results(&mut emission.body, &source_body.ty, &mut results, &outputs)?;
        if !emission.pending && !emission.entries.is_empty() {
            let Some(last) = emission.entries.last_mut() else {
                return Err(error("entry publication has no shader"));
            };
            last.outputs.extend(outputs);
        } else {
            emission.flush(&LookupSet::default(), outputs, true)?;
        }
        let mut entry_ids = Vec::new();
        for mut entry in emission.entries {
            entry.id = ids.next_id();
            let owner = if let Some(group) = &declaration.graphics_group {
                let Some(name) = emission.body.compiler.program.source.symbols.get(group.root) else {
                    return Err(error("graphics owner missing"));
                };
                name.clone()
            } else {
                declaration.name.clone()
            };
            source_names.insert(entry.id, owner);
            origins.insert(entry.id, symbol);
            entry_ids.push(entry.id);
            entries.push(entry);
        }
        pipeline.dispatch_loops.extend(emission.loops);
        if declaration.entry_kind == EntryKind::Compute {
            pipeline.pipelines.push(Pipeline::Compute(ComputePipeline {
                stages: emission.stages,
                bindings: vec![],
                default_total_threads: std::num::NonZeroU32::new(1),
            }));
            associations.push(entry_ids);
        } else {
            let Some(group) = &declaration.graphics_group else {
                return Err(error("graphics entry has no operation identity"));
            };
            let index = *graphics.entry((group.root, group.operation)).or_insert_with(|| {
                let index = pipeline.pipelines.len();
                pipeline.pipelines.push(Pipeline::Graphics(GraphicsPipeline {
                    source_operation: Some(group.operation),
                    invocation: GraphicsInvocation::default(),
                    stages: vec![],
                    bindings: vec![],
                    vertex_inputs: vec![],
                    fragment_outputs: vec![],
                }));
                associations.push(vec![]);
                invocations.push((index, &group.invocation));
                index
            });
            let Pipeline::Graphics(graphics) = &mut pipeline.pipelines[index] else {
                return Err(error("graphics pipeline identity mismatch"));
            };
            for id in entry_ids {
                let Some(entry) = entries.iter().find(|entry| entry.id == id) else {
                    return Err(error("graphics shader missing"));
                };
                graphics.stages.push(GraphicsStage {
                    entry_point: entry.name.clone(),
                    owner: source_names[&id].clone(),
                    stage: if declaration.entry_kind == EntryKind::Vertex {
                        ShaderStage::Vertex
                    } else {
                        ShaderStage::Fragment
                    },
                    uses: Default::default(),
                });
                associations[index].push(id);
            }
        }
    }

    for (index, invocation) in invocations {
        let draw = invocation.draw.try_map_buffers(|operand| {
            let (owner, slot) = match operand {
                DrawBufferOperand::Input(input) => return Ok(input.clone()),
                DrawBufferOperand::Result { entry, slot } => (*entry, *slot),
            };
            let Some(entry) = entries
                .iter()
                .find(|entry| origins.get(&entry.id) == Some(&owner) && !entry.outputs.is_empty())
            else {
                return Err(error("draw producer has no published output"));
            };
            let Some(output) = entry.outputs.get(slot) else {
                return Err(error("draw output slot missing"));
            };
            let Some(binding) = output.storage_binding() else {
                return Err(error("draw output is not storage"));
            };
            let name = output.host.target.clone().unwrap_or_else(|| {
                if entry.outputs.len() == 1 {
                    format!("{}_output", entry.name)
                } else {
                    format!("{}_output_{slot}", entry.name)
                }
            });
            Ok::<_, OptimizeError>(DrawBufferRef {
                set: binding.set,
                binding: binding.binding,
                resource: Some(name.clone()),
                name,
            })
        })?;
        let Pipeline::Graphics(graphics) = &mut pipeline.pipelines[index] else {
            return Err(error("graphics invocation has no pipeline"));
        };
        graphics.invocation = GraphicsInvocation {
            topology: invocation.topology,
            raster_state: invocation.raster_state.clone(),
            fragment_state: invocation.fragment_state.clone(),
            draw,
        };
    }
    for association in &associations {
        let mut accesses = LookupMap::default();
        for entry in entries.iter().filter(|entry| association.contains(&entry.id)) {
            for (&binding, &access) in &entry.stage_descriptor_storage_accesses {
                accesses
                    .entry(binding)
                    .and_modify(|previous: &mut ResourceAccess| *previous = previous.merge(access))
                    .or_insert(access);
            }
        }
        for entry in entries.iter_mut().filter(|entry| association.contains(&entry.id)) {
            entry.pipeline_storage_accesses = accesses.clone();
        }
    }
    let publications: Vec<_> = entries.iter().collect();
    pipeline
        .publish_implicit_bindings(&publications, &associations, |entry| {
            source_names[&entry.id].as_str()
        })
        .map_err(|failure| error(failure.to_string()))?;
    pipeline.publish_graphics_io(&publications, &associations);
    for pipeline in &mut pipeline.pipelines {
        if let Pipeline::Compute(pipeline) = pipeline {
            for stage in &mut pipeline.stages {
                let Some(entry) = entries.iter().find(|entry| entry.name == stage.entry_point) else {
                    return Err(error("published stage has no shader"));
                };
                for (i, binding) in pipeline.bindings.iter().enumerate() {
                    if let Some((set, slot)) = binding.slot() {
                        if let Some(&access) =
                            entry.stage_descriptor_storage_accesses.get(&BindingRef::new(set, slot))
                        {
                            stage.uses.record(
                                i,
                                match access {
                                    ResourceAccess::Read => crate::host::Access::ReadOnly,
                                    ResourceAccess::Write => crate::host::Access::WriteOnly,
                                    ResourceAccess::ReadWrite => crate::host::Access::ReadWrite,
                                },
                            );
                        }
                    } else {
                        stage.uses.record(i, crate::host::Access::ReadOnly);
                    }
                }
            }
            reconcile_storage_binding_access(
                &mut pipeline.bindings,
                pipeline.stages.iter().map(|stage| &stage.uses),
            );
        } else if let Pipeline::Graphics(pipeline) = pipeline {
            for stage in &mut pipeline.stages {
                let Some(entry) = entries.iter().find(|entry| entry.name == stage.entry_point) else {
                    return Err(error("graphics stage has no shader"));
                };
                for (i, binding) in pipeline.bindings.iter().enumerate() {
                    if let Some((set, slot)) = binding.slot() {
                        if let Some(&access) =
                            entry.stage_descriptor_storage_accesses.get(&BindingRef::new(set, slot))
                        {
                            stage.uses.record(
                                i,
                                match access {
                                    ResourceAccess::Read => crate::host::Access::ReadOnly,
                                    ResourceAccess::Write => crate::host::Access::WriteOnly,
                                    ResourceAccess::ReadWrite => crate::host::Access::ReadWrite,
                                },
                            );
                        } else if !matches!(binding, crate::host::Binding::StorageBuffer { .. }) {
                            stage.uses.record(i, crate::host::Access::ReadOnly);
                        }
                    }
                }
            }
            reconcile_storage_binding_access(
                &mut pipeline.bindings,
                pipeline.stages.iter().map(|stage| &stage.uses),
            );
        }
    }
    pipeline.frame_graph = FrameGraph::from_selected_pipelines(&pipeline.pipelines, &[]).map_err(error)?;
    Ok(Elaborated::from_parts(
        std::mem::take(&mut compiler.functions),
        entries.into_iter().map(EntryPoint::into_shader).collect(),
        BackendGlobal { pipeline, target },
    ))
}

struct EntryEmission<'a, 'p, 'source> {
    body: Body<'a, 'p, 'source>,
    templates: Vec<(EntryInput, Type)>,
    params: Vec<Value>,
    parameter_inputs: Vec<Vec<usize>>,
    inputs: Vec<EntryInput>,
    parameters: LookupSet<Value>,
    saved: LookupMap<Value, SavedValue>,
    name: String,
    kind: EntryKind,
    span: crate::ast::Span,
    pipeline: usize,
    invocation: ((u32, u32, u32), DispatchSize),
    pending: bool,
    stages: Vec<ComputeStage>,
    entries: Vec<EntryPoint>,
    loops: Vec<DispatchLoop>,
}

impl EntryEmission<'_, '_, '_> {
    fn restart(&mut self) -> Result<(), OptimizeError> {
        self.body.reset_shader();
        self.inputs.clear();
        let mut params = self.params.iter().copied();
        for (input, ty) in &self.templates {
            bind_input(&mut self.body, input.clone(), ty, &mut params, &mut self.inputs)?;
        }
        if params.next().is_some() {
            return Err(error("unbound entry parameters"));
        }
        self.parameters = self.body.values.keys().copied().collect();
        let Some(generated) = self.body.generated.as_ref() else {
            return Err(error("entry buffers missing"));
        };
        for input in &generated.inputs {
            self.body.builder.func_mut().add_function_param(input.ty.clone());
        }
        restore(&mut self.body, &self.saved)
    }

    fn flush(
        &mut self,
        live: &LookupSet<Value>,
        outputs: Vec<EntryOutput>,
        force: bool,
    ) -> Result<(), OptimizeError> {
        if !self.pending && !force {
            return Ok(());
        }
        if outputs.is_empty() {
            save(&mut self.body, live, &self.parameters, &mut self.saved)?;
        }
        let (local_size, mut dispatch_size) = if let Some(statement) = self.body.active_dispatch {
            let dispatch = *self.body.compiler.execution.dispatch(statement)?;
            let size = if let Some((x, y, z)) = dispatch.grid {
                DispatchSize::Fixed {
                    x,
                    y,
                    z,
                    explicit: true,
                }
            } else {
                let [x, y, z] = dispatch.groups.map(|index| {
                    host::guarded_index(
                        self.body.query(),
                        &self.body.host_values,
                        index,
                        self.body.guard.as_ref(),
                    )
                });
                let groups = [x?, y?, z?];
                let elements = Expr::Multiply(
                    Box::new(Expr::Multiply(
                        Box::new(groups[0].clone()),
                        Box::new(groups[1].clone()),
                    )),
                    Box::new(groups[2].clone()),
                );
                DispatchSize::Computed { elements, groups }
            };
            (dispatch.local_size, size)
        } else {
            self.invocation.clone()
        };
        if let Some(condition) = &self.body.guard {
            let enabled = Expr::Scalar(ScalarExpr::If {
                condition: Box::new(condition.clone()),
                yes: Box::new(ScalarExpr::I32(1)),
                no: Box::new(ScalarExpr::I32(0)),
            });
            let (elements, mut groups) = match dispatch_size {
                DispatchSize::Fixed { x, y, z, .. } => (
                    Expr::Integer(i64::from(x)),
                    [
                        Expr::Integer(i64::from(x)),
                        Expr::Integer(i64::from(y)),
                        Expr::Integer(i64::from(z)),
                    ],
                ),
                DispatchSize::Computed { elements, groups } => (elements, groups),
            };
            groups[0] = Expr::Multiply(Box::new(groups[0].clone()), Box::new(enabled));
            dispatch_size = DispatchSize::Computed { elements, groups };
        }
        let Some(generated) = self.body.generated.as_ref() else {
            return Err(error("entry buffers missing"));
        };
        let mut inputs = self.inputs.clone();
        inputs.extend(generated.inputs.iter().cloned());
        let storage_bindings = generated.declarations.clone();
        let mut accesses = std::mem::take(&mut self.body.storage_accesses);
        for input in &inputs {
            if matches!(
                input.kind,
                EntryInputKind::Texture { .. }
                    | EntryInputKind::Sampler { .. }
                    | EntryInputKind::Uniform { .. }
            ) {
                if let Some(binding) = input.descriptor_binding() {
                    accesses.entry(binding).or_insert(ResourceAccess::Read);
                }
            }
        }
        let function_body = self.body.finish_shader()?;
        let phase = match self.kind {
            EntryKind::Vertex => "vertex",
            EntryKind::Fragment => "fragment",
            _ => "compute",
        };
        let name = if self.entries.is_empty() {
            format!("{}_{phase}", self.name)
        } else {
            format!("{}_{phase}_{}", self.name, self.entries.len())
        };
        let execution_model = match self.kind {
            EntryKind::Compute => ExecutionModel::Compute { local_size },
            EntryKind::Vertex => ExecutionModel::Vertex,
            EntryKind::Fragment => ExecutionModel::Fragment,
            EntryKind::Root => return Err(error("unextracted graphics entry")),
        };
        if self.kind == EntryKind::Compute {
            self.stages.push(ComputeStage {
                dependencies: self
                    .stages
                    .last()
                    .map(|_| (self.pipeline, self.stages.len() - 1))
                    .into_iter()
                    .collect(),
                entry_point: name.clone(),
                owner: self.name.clone(),
                workgroup_size: local_size,
                dispatch_size,
                uses: Default::default(),
            });
        }
        self.entries.push(EntryPoint {
            id: crate::EntryId(0),
            name,
            body: function_body,
            execution_model,
            inputs,
            parameter_inputs: self.parameter_inputs.clone(),
            outputs,
            storage_bindings,
            stage_descriptor_storage_accesses: accesses.clone(),
            pipeline_storage_accesses: accesses,
            span: self.span,
        });
        self.pending = false;
        self.restart()
    }

    fn statements(&mut self, statements: Value, after: &LookupSet<Value>) -> Result<(), OptimizeError> {
        let statements = self.body.query().vector(statements)?;
        for (i, &statement) in statements.iter().enumerate() {
            let mut live = statement_names(self.body.query(), &statements[i + 1..])?;
            live.extend(after.iter().copied());
            let fields = self.body.query().fields("Let", statement)?;
            let expression = fields[3];
            if self.body.compiler.execution.dispatch(statement).is_ok() {
                let mut before = statement_names(self.body.query(), &[statement])?;
                before.extend(live.iter().copied());
                self.flush(&before, vec![], false)?;
                self.body.active_dispatch = Some(statement);
                self.body.statement(statement)?;
                self.pending = true;
                self.flush(&live, vec![], true)?;
            } else if let Some(allocation) = self.body.query().enode("Alloc", expression)? {
                if self.body.query().enode("DeviceSpace", allocation[0])?.is_some() {
                    let mut needed = names_in(self.body.query(), "NamesIndexExp", allocation[1])?;
                    if needed.iter().any(|name| !self.body.host_values.contains_key(name)) {
                        needed.extend(live.iter().copied());
                        self.flush(&needed, vec![], false)?;
                    }
                }
                self.body.statement(statement)?;
                // Device allocation belongs to the host program; declaring its
                // shader view does not itself require a dispatch.
                self.pending |= self.body.query().enode("DeviceSpace", allocation[0])?.is_none();
            } else if let Some(branch) = self.body.query().enode("If", expression)? {
                if self.has_host_effect(branch[2])? || self.has_host_effect(branch[3])? {
                    self.branch(fields[1], &branch, &live)?;
                } else {
                    self.body.statement(statement)?;
                    self.pending = true;
                }
            } else if let Some(loop_) = self.body.query().enode("Loop", expression)? {
                if self.has_host_effect(loop_[3])? {
                    self.loop_body(fields[1], &loop_, &live)?;
                } else {
                    self.body.statement(statement)?;
                    self.pending = true;
                }
            } else if let Some(accumulator) = self.body.query().enode("WithAcc", expression)? {
                let lambda = self.body.query().fields("FunctionBody", accumulator[1])?;
                let block = self.body.query().fields("Block", lambda[2])?;
                let mut needed = live_names(self.body.query(), &[], block[1])?;
                needed.extend(live.iter().copied());
                self.statements(block[0], &needed)?;
                let values = self.body.results(block[1])?;
                let pattern = self.body.query().vector(fields[1])?;
                self.body.bind(&pattern, values)?;
                self.pending = true;
            } else {
                self.body.statement(statement)?;
                let uninitialized = match self.body.query().enode("GenerateArray", expression)? {
                    Some(fields) => self.body.query().enode("Uninitialized", fields[0])?.is_some(),
                    None => false,
                };
                self.pending |= !uninitialized;
            }
        }
        Ok(())
    }

    fn has_host_effect(&self, block: Value) -> Result<bool, OptimizeError> {
        let block = self.body.query().fields("Block", block)?;
        for statement in self.body.query().vector(block[0])? {
            if self.body.compiler.execution.dispatch(statement).is_ok() {
                return Ok(true);
            }
            let fields = self.body.query().fields("Let", statement)?;
            if let Some(allocation) = self.body.query().enode("Alloc", fields[3])? {
                if self.body.query().enode("DeviceSpace", allocation[0])?.is_some() {
                    return Ok(true);
                }
            }
            if let Some(branch) = self.body.query().enode("If", fields[3])? {
                if self.has_host_effect(branch[2])? || self.has_host_effect(branch[3])? {
                    return Ok(true);
                }
            } else if let Some(loop_) = self.body.query().enode("Loop", fields[3])? {
                if self.has_host_effect(loop_[3])? {
                    return Ok(true);
                }
            } else if let Some(accumulator) = self.body.query().enode("WithAcc", fields[3])? {
                let lambda = self.body.query().fields("FunctionBody", accumulator[1])?;
                if self.has_host_effect(lambda[2])? {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    fn scalar_slot(&mut self, name: Value, ty: &Type) -> Result<BindingRef, OptimizeError> {
        let Some(bytes) = storage_elem_stride(&storage_value_type(ty)) else {
            return Err(error("dispatch witness has no storage size"));
        };
        let (slot, _) = self.body.device_buffer(ty, Expr::Integer(bytes.into()), Body::number(1))?;
        let Some(generated) = self.body.generated.as_mut() else {
            return Err(error("entry buffers missing"));
        };
        generated.scalar_slots.insert(name, (slot, ty.clone()));
        Ok(slot)
    }

    fn read_slot(&mut self, name: Value, slot: BindingRef, ty: &Type) {
        self.saved.insert(name, SavedValue::Buffer(slot, ty.clone()));
        if let Some(ty) = host::scalar_type(ty) {
            self.body.host_values.insert(
                name,
                ScalarExpr::Read {
                    source: scalar_source(slot),
                    offset: 0,
                    ty,
                },
            );
        }
    }

    fn buffer_source(&self, output: Value) -> Result<BufferSource, OptimizeError> {
        let mut cases = Vec::new();
        for (_, tag, storage) in self.body.storage_roots(output)? {
            let Some(binding) = types::array_view_buffer(&storage.ty) else {
                return Err(error("host loop carries invocation-local memory"));
            };
            cases.push((tag, scalar_source(binding)));
        }
        cases.sort_by_key(|(tag, _)| *tag);
        cases.dedup_by_key(|(tag, _)| *tag);
        if cases.is_empty() {
            return Err(error("loop memory result has no reachable storage"));
        }
        if let [(_, source)] = cases.as_slice() {
            return Ok(BufferSource::Binding(source.clone()));
        }
        let Some(generated) = self.body.generated.as_ref() else {
            return Err(error("entry buffers missing"));
        };
        let Some((slot, _)) = generated.scalar_slots.get(&output) else {
            return Err(error("loop memory tag has no saved value"));
        };
        Ok(BufferSource::Select {
            tag: ScalarExpr::Read {
                source: scalar_source(*slot),
                offset: 0,
                ty: ScalarType::I32,
            },
            cases,
        })
    }

    fn loop_body(
        &mut self,
        pattern: Value,
        loop_: &[Value],
        after: &LookupSet<Value>,
    ) -> Result<(), OptimizeError> {
        let pattern = self.body.query().vector(pattern)?;
        let merges = self.body.query().vector(loop_[0])?;
        if merges.len() != pattern.len() {
            return Err(error("loop merge/result arity mismatch"));
        }
        let block = self.body.query().fields("Block", loop_[3])?;
        let mut needed = live_names(self.body.query(), &self.body.query().vector(block[0])?, block[1])?;
        needed.extend(after.iter().copied());
        needed.extend(names_in(self.body.query(), "NamesMerges", loop_[0])?);
        needed.extend(names_in(self.body.query(), "NamesLoopForm", loop_[1])?);
        self.flush(&needed, vec![], false)?;
        let incoming = self.saved.clone();
        let incoming_host = self
            .body
            .host_values
            .iter()
            .map(|(&name, value)| (name, value.clone()))
            .collect::<LookupMap<_, _>>();
        let mut parameters = Vec::new();
        let mut current = Vec::new();
        let mut incoming_blocks = Vec::new();
        for (merge, &output) in merges.iter().zip(&pattern) {
            let merge = self.body.query().fields("Carry", *merge)?;
            let parameter = self.body.query().fields("Parameter", merge[0])?[0];
            parameters.push(parameter);
            if let Some((name, ty)) = super::runtime_binding(self.body.query(), parameter)? {
                let Some(value) = self.body.operand(merge[1])? else {
                    return Err(error("loop runtime argument missing"));
                };
                self.body.values.insert(name, value.clone());
                if let Some(expression) =
                    host::operand(self.body.query(), &self.body.host_values, merge[1])?
                {
                    self.body.host_values.insert(name, expression);
                }
                let output_name = super::runtime_binding(self.body.query(), output)?
                    .ok_or_else(|| error("loop result changes its binding kind"))?
                    .0;
                needed.insert(name);
                if self.body.query().enode("BindBlock", parameter)?.is_some() {
                    let element = self.body.block_element(name)?;
                    // This ABI slot is rebound before use; it owns no array payload.
                    let (slot, storage) =
                        self.body.device_buffer(&element, Expr::Integer(0), Body::number(0))?;
                    incoming_blocks.push(Some(self.body.query().fields("BlockOperand", merge[1])?[0]));
                    let handle = self.body.register_memory(name, storage)?;
                    self.body.values.insert(name, handle);
                    let Some(generated) = self.body.generated.as_mut() else {
                        return Err(error("entry buffers missing"));
                    };
                    generated.memories.insert(name, (slot, storage_value_type(&element)));
                    current.push((name, output_name, slot, None));
                } else {
                    incoming_blocks.push(None);
                    let slot = self.scalar_slot(name, &ty)?;
                    current.push((name, output_name, slot, Some(ty)));
                }
            }
        }
        let control = if let Some(form) = self.body.query().enode("Counted", loop_[1])? {
            let Some(count) = host::scalar_index(self.body.query(), &self.body.host_values, form[2])?
            else {
                return Err(error("loop count has no host value"));
            };
            let slot = self.scalar_slot(form[0], &types::i32())?;
            let zero = self.body.literal(0, &types::i32())?;
            self.body.values.insert(form[0], zero);
            needed.insert(form[0]);
            DispatchLoopControl::Counted {
                count,
                index: scalar_source(slot),
            }
        } else {
            let condition = self.body.query().fields("While", loop_[1])?[0];
            let Some((_, _, slot, _)) = current.iter().find(|(name, _, _, _)| *name == condition) else {
                return Err(error("loop predicate is not carried"));
            };
            DispatchLoopControl::While {
                condition: ScalarExpr::Read {
                    source: scalar_source(*slot),
                    offset: 0,
                    ty: ScalarType::Bool,
                },
            }
        };
        self.flush(&needed, vec![], true)?;
        let setup = self.stages.len() - 1;
        let incoming_buffers = incoming_blocks
            .into_iter()
            .zip(&current)
            .map(|(block, (_, _, slot, _))| {
                if let Some(block) = block {
                    self.buffer_source(block)
                } else {
                    Ok(BufferSource::Binding(scalar_source(*slot)))
                }
            })
            .collect::<Result<Vec<_>, OptimizeError>>()?;
        for &(name, _, slot, ref ty) in &current {
            if let Some(ty) = ty {
                self.read_slot(name, slot, ty);
            }
        }
        if let Some(form) = self.body.query().enode("Counted", loop_[1])? {
            let DispatchLoopControl::Counted {
                index: ScalarSource::Binding { set, binding },
                ..
            } = &control
            else {
                return Err(error("counted loop index slot missing"));
            };
            self.read_slot(form[0], BindingRef::new(*set, *binding), &types::i32());
        }
        self.restart()?;
        if let Some(form) = self.body.query().enode("Counted", loop_[1])? {
            let ordinal = self.body.named(form[0])?;
            let start = self.body.index_expression(form[1])?;
            let step = self.body.index_expression(form[3])?;
            let delta = self.body.binary(BinaryOperator::Multiply, ordinal, step)?;
            let index = self.body.binary(BinaryOperator::Add, start, delta)?;
            self.body.values.insert(form[0], index);
            let Some(ordinal) = self.body.host_values.get(&form[0]).cloned() else {
                return Err(error("loop ordinal has no host value"));
            };
            let Some(start) = host::scalar_index(self.body.query(), &self.body.host_values, form[1])?
            else {
                return Err(error("loop start has no host value"));
            };
            let Some(step) = host::scalar_index(self.body.query(), &self.body.host_values, form[3])? else {
                return Err(error("loop step has no host value"));
            };
            let delta = ScalarExpr::Apply {
                op: "mul".into(),
                ty: ScalarType::I32,
                args: vec![ordinal, step],
            };
            self.body.host_values.insert(
                form[0],
                ScalarExpr::Apply {
                    op: "add".into(),
                    ty: ScalarType::I32,
                    args: vec![start, delta],
                },
            );
        }
        let mut live = live_names(self.body.query(), &[], block[1])?;
        live.extend(after.iter().copied());
        for &(name, _, _, _) in &current {
            live.insert(name);
        }
        self.statements(block[0], &live)?;
        let values = self.body.results(block[1])?;
        self.body.bind(&pattern, values)?;
        let mut returned = after.clone();
        for &(_, name, _, _) in &current {
            returned.insert(name);
        }
        for &binding in &pattern {
            if let Some((name, ty)) = super::runtime_binding(self.body.query(), binding)? {
                self.scalar_slot(name, &ty)?;
            }
        }
        self.flush(&returned, vec![], true)?;
        let mut carried = Vec::new();
        for ((_, output, current_slot, ty), initial) in current.iter().zip(incoming_buffers) {
            let next = if ty.is_some() {
                let Some(generated) = self.body.generated.as_ref() else {
                    return Err(error("entry buffers missing"));
                };
                let Some((next_slot, _)) = generated.scalar_slots.get(output) else {
                    return Err(error("loop result slot missing"));
                };
                BufferSource::Binding(scalar_source(*next_slot))
            } else {
                self.buffer_source(*output)?
            };
            carried.push(LoopCarry {
                binding: scalar_source(*current_slot),
                initial,
                next,
            });
        }
        let completion = self.stages.len();
        self.saved = incoming;
        self.body.host_values = incoming_host.into();
        for (parameter, output, slot, ty) in current {
            if let Some(ty) = ty {
                self.read_slot(output, slot, &ty);
                let Some(generated) = self.body.generated.as_mut() else {
                    return Err(error("entry buffers missing"));
                };
                generated.scalar_slots.insert(output, (slot, ty));
            } else {
                let Some(&tag) = self.body.compiler.block_ids.get(&parameter) else {
                    return Err(error("loop memory has no block identity"));
                };
                self.body.compiler.block_ids.insert(output, tag);
                self.saved.insert(output, SavedValue::Constant(i64::from(tag), types::i32()));
            }
        }
        self.restart()?;
        self.flush(after, vec![], true)?;
        self.loops.push(DispatchLoop {
            pipeline: self.pipeline,
            setup,
            body: (setup + 1..completion).collect(),
            test: vec![],
            completion,
            control,
            carried,
        });
        Ok(())
    }

    fn branch(
        &mut self,
        pattern: Value,
        branch: &[Value],
        after: &LookupSet<Value>,
    ) -> Result<(), OptimizeError> {
        let yes = self.body.query().fields("Block", branch[2])?;
        let no = self.body.query().fields("Block", branch[3])?;
        let mut before = live_names(self.body.query(), &self.body.query().vector(yes[0])?, yes[1])?;
        before.extend(live_names(
            self.body.query(),
            &self.body.query().vector(no[0])?,
            no[1],
        )?);
        if let Some(var) = self.body.query().enode("Var", branch[0])? {
            before.insert(var[0]);
        }
        before.extend(after.iter().copied());
        self.flush(&before, vec![], false)?;
        let Some(condition) = host::atom(self.body.query(), &self.body.host_values, branch[0])? else {
            return Err(error("branch condition has no host value"));
        };
        let pattern = self.body.query().vector(pattern)?;
        let mut joined = LookupMap::default();
        for &binding in &pattern {
            if let Some((name, ty)) = super::runtime_binding(self.body.query(), binding)? {
                let Some(bytes) = storage_elem_stride(&storage_value_type(&ty)) else {
                    return Err(error("branch witness has no storage size"));
                };
                let (slot, _) =
                    self.body.device_buffer(&ty, Expr::Integer(bytes.into()), Body::number(1))?;
                let Some(generated) = self.body.generated.as_mut() else {
                    return Err(error("entry buffers missing"));
                };
                generated.scalar_slots.insert(name, (slot, ty.clone()));
                joined.insert(name, SavedValue::Buffer(slot, ty));
            }
        }
        let incoming = self.saved.clone();
        let incoming_host = self
            .body
            .host_values
            .iter()
            .map(|(&name, value)| (name, value.clone()))
            .collect::<LookupMap<_, _>>();
        let enclosing = self.body.guard.clone();
        for (block, take_yes) in [(yes, true), (no, false)] {
            self.saved = incoming.clone();
            self.body.host_values = incoming_host.clone().into();
            self.restart()?;
            let condition = if take_yes {
                condition.clone()
            } else {
                ScalarExpr::If {
                    condition: Box::new(condition.clone()),
                    yes: Box::new(ScalarExpr::Bool(false)),
                    no: Box::new(ScalarExpr::Bool(true)),
                }
            };
            self.body.guard = Some(if let Some(outer) = &enclosing {
                ScalarExpr::If {
                    condition: Box::new(outer.clone()),
                    yes: Box::new(condition),
                    no: Box::new(ScalarExpr::Bool(false)),
                }
            } else {
                condition
            });
            let mut live = live_names(self.body.query(), &[], block[1])?;
            live.extend(after.iter().copied());
            self.statements(block[0], &live)?;
            let values = self.body.results(block[1])?;
            self.body.bind(&pattern, values)?;
            let mut live = after.clone();
            live.extend(joined.keys().copied());
            self.flush(&live, vec![], true)?;
        }
        self.body.guard = enclosing;
        self.saved = incoming;
        self.saved.extend(joined);
        self.body.host_values = incoming_host.into();
        for (&name, value) in &self.saved {
            if let SavedValue::Buffer(binding, ty) = value {
                if let Some(ty) = host::scalar_type(ty) {
                    self.body.host_values.insert(
                        name,
                        ScalarExpr::Read {
                            source: ScalarSource::Binding {
                                set: binding.set,
                                binding: binding.binding,
                            },
                            offset: 0,
                            ty,
                        },
                    );
                }
            }
        }
        self.restart()
    }
}

fn scalar_source(binding: BindingRef) -> ScalarSource {
    ScalarSource::Binding {
        set: binding.set,
        binding: binding.binding,
    }
}

fn statement_names(query: Query<'_>, statements: &[Value]) -> Result<LookupSet<Value>, OptimizeError> {
    let mut names = LookupSet::default();
    for &statement in statements {
        names.extend(names_in(query, "NamesStm", statement)?);
    }
    Ok(names)
}

fn names_in(query: Query<'_>, table: &str, value: Value) -> Result<LookupSet<Value>, OptimizeError> {
    let set = query.required(table, (value,))?;
    let Some(set) = query.0.value_to_container::<SetContainer>(set) else {
        return Err(error("invalid free-name set"));
    };
    set.data
        .iter()
        .map(|&name| Ok(query.application(name, &["ValueNameRef", "BlockNameRef", "AccNameRef"])?.1[0]))
        .collect()
}

pub(super) struct GeneratedBuffers {
    owner: String,
    next: u32,
    inputs: Vec<EntryInput>,
    declarations: Vec<StorageBindingDecl>,
    pub memories: LookupMap<Value, (BindingRef, Type)>,
    scalar_slots: LookupMap<Value, (BindingRef, Type)>,
}

impl Body<'_, '_, '_> {
    pub(in crate::egglog::to_ssa) fn device_buffer(
        &mut self,
        element: &Type,
        bytes: Expr,
        count: Typed,
    ) -> Result<(BindingRef, Typed), OptimizeError> {
        let element = storage_value_type(element);
        let Some(stride) = storage_elem_stride(&element) else {
            return Err(error("device element has no byte stride"));
        };
        let Some(generated) = self.generated.as_mut() else {
            return Err(error("device allocation is outside an entry"));
        };
        let binding = BindingRef::new(0, generated.next);
        generated.next =
            generated.next.checked_add(1).ok_or_else(|| error("descriptor slots exhausted"))?;
        let name = format!("{}_memory_{}", generated.owner, binding.binding);
        let length = BufferLen::Computed {
            bytes: Expr::Max(Box::new(bytes), Box::new(Expr::Integer(stride.into()))),
        };
        generated.declarations.push(StorageBindingDecl {
            binding,
            role: StorageRole::Intermediate,
            elem_ty: element.clone(),
            host: StorageHost {
                logical_resource: name.clone(),
                length: length.clone(),
            },
        });
        let input = EntryInput {
            name,
            ty: types::view_array_with_size(
                &element,
                Type::Constructed(TypeName::SizePlaceholder, vec![]),
                buffer_tag(binding),
            ),
            kind: EntryInputKind::Storage {
                exposure: BindingExposure::Host(binding),
                access: StorageAccess::ReadWrite,
            },
            host: InputHost {
                length: Some(length),
                ..InputHost::default()
            },
        };
        self.builder.func_mut().add_function_param(input.ty.clone());
        generated.inputs.push(input);
        Ok((binding, storage_view(self, binding, element, count)?))
    }
}

// These slots are final cross-dispatch ABI transport, not a second memory layout.
#[derive(Clone)]
enum SavedValue {
    Constant(i64, Type),
    Buffer(BindingRef, Type),
}

fn live_names(
    query: Query<'_>,
    statements: &[Value],
    results: Value,
) -> Result<LookupSet<Value>, OptimizeError> {
    let mut names = statement_names(query, statements)?;
    names.extend(names_in(query, "NamesResults", results)?);
    Ok(names)
}

fn restore(body: &mut Body<'_, '_, '_>, saved: &LookupMap<Value, SavedValue>) -> Result<(), OptimizeError> {
    let Some(generated) = body.generated.as_ref() else {
        return Err(error("dispatch buffers missing"));
    };
    let memories = generated
        .memories
        .iter()
        .map(|(&name, (binding, ty))| (name, *binding, ty.clone()))
        .collect::<Vec<_>>();
    for (name, binding, element) in memories {
        let count = body.op(
            OpTag::Intrinsic {
                id: catalog().known().storage_len,
                overload_idx: 0,
            },
            vec![Body::number(binding.set), Body::number(binding.binding)],
            Body::number(0).ty,
        )?;
        let storage = storage_view(body, binding, element, count)?;
        let handle = body.register_memory(name, storage)?;
        body.values.insert(name, handle);
    }
    for (&name, saved) in saved {
        let value = match saved {
            SavedValue::Constant(n, ty) => body.literal(*n, ty)?,
            SavedValue::Buffer(binding, ty) => {
                let storage = storage_view(body, *binding, storage_value_type(ty), Body::number(1))?;
                let value = body.index(storage, Body::number(0))?;
                body.cast(value, ty)?
            }
        };
        body.values.insert(name, value);
    }
    Ok(())
}

fn save(
    body: &mut Body<'_, '_, '_>,
    live: &LookupSet<Value>,
    parameters: &LookupSet<Value>,
    saved: &mut LookupMap<Value, SavedValue>,
) -> Result<(), OptimizeError> {
    // A view keeps its layout witnesses and backing handle live as well.
    let mut live = live.clone();
    let mut pending = live.iter().copied().collect::<Vec<_>>();
    while let Some(name) = pending.pop() {
        if let Some(info) = body.query().lookup("BindingInfo", (body.function, name))? {
            for dependency in names_in(body.query(), "NamesValueInfo", info)? {
                if live.insert(dependency) {
                    pending.push(dependency);
                }
            }
        }
    }
    saved.retain(|name, _| live.contains(name));
    let values = body
        .values
        .iter()
        .filter(|(name, _)| live.contains(name) && !parameters.contains(name))
        .map(|(&name, value)| (name, value.clone()))
        .collect::<Vec<_>>();
    let leader = if let Some(statement) = body.active_dispatch {
        let dispatch = *body.compiler.execution.dispatch(statement)?;
        let (group, _) = body.workgroups(dispatch.grid)?;
        let first = body.binary(BinaryOperator::Equal, group, Body::number(0))?;
        let lane = body.local_id()?;
        let first_lane = body.binary(BinaryOperator::Equal, lane, Body::number(0))?;
        Some(body.binary(BinaryOperator::LogicalAnd, first, first_lane)?)
    } else {
        None
    };
    let mut roots = LookupMap::default();
    for &name in &live {
        let Some(handle) = body.values.get(&name).cloned() else {
            continue;
        };
        for (root, tag, storage) in body.storage_roots(name)? {
            if types::array_view_buffer(&storage.ty).is_none() {
                roots.entry(root).or_insert_with(|| (tag, storage, Vec::new())).2.push(handle.clone());
            }
        }
    }
    for (name, (tag, storage, handles)) in roots {
        let Some(element) = storage.ty.elem_type().cloned() else {
            return Err(error("private memory has no element type"));
        };
        let Some(bytes) = type_byte_size(&storage_value_type(&storage.ty)) else {
            return Err(error("escaping private memory has no static capacity"));
        };
        let Some(stride) = storage_elem_stride(&storage_value_type(&element)) else {
            return Err(error("escaping memory element has no stride"));
        };
        let (binding, destination) = body.device_buffer(
            &element,
            Expr::Integer(bytes.into()),
            Body::number(bytes / stride),
        )?;
        let handle = body.literal(i64::from(tag), &types::i32())?;
        let mut selected = body.literal(0, &types::bool_type())?;
        for live_handle in handles {
            let matches = body.binary(BinaryOperator::Equal, live_handle, handle.clone())?;
            selected = body.binary(BinaryOperator::LogicalOr, selected, matches)?;
        }
        if let Some(leader) = &leader {
            selected = body.binary(BinaryOperator::LogicalAnd, selected, leader.clone())?;
        }
        body.when(selected, |body| {
            body.transfer_block(handle, Body::number(bytes), &element, destination.clone())
        })?;
        body.register_memory(name, destination)?;
        let Some(generated) = body.generated.as_mut() else {
            return Err(error("dispatch buffers missing"));
        };
        generated.memories.insert(name, (binding, storage_value_type(&element)));
    }
    for (name, value) in values {
        let Some(generated) = body.generated.as_ref() else {
            return Err(error("dispatch buffers missing"));
        };
        let slot = generated.scalar_slots.get(&name).cloned();
        if slot.is_none() {
            if let Some(n) = body.integer_literal(&value) {
                saved.insert(name, SavedValue::Constant(n, value.ty));
                continue;
            }
        }
        if value.ty == types::unit() {
            continue;
        }
        let (binding, storage) = if let Some((binding, ty)) = slot {
            (
                binding,
                storage_view(body, binding, storage_value_type(&ty), Body::number(1))?,
            )
        } else {
            let Some(bytes) = storage_elem_stride(&storage_value_type(&value.ty)) else {
                return Err(error("cross-dispatch value has no storage size"));
            };
            body.device_buffer(&value.ty, Expr::Integer(bytes.into()), Body::number(1))?
        };
        if let Some(leader) = &leader {
            body.when(leader.clone(), |body| {
                body.store(storage, Body::number(0), value.clone())
            })?;
        } else {
            body.store(storage, Body::number(0), value.clone())?;
        }
        if !body.host_values.contains_key(&name) {
            if let Some(ty) = host::scalar_type(&value.ty) {
                body.host_values.insert(
                    name,
                    ScalarExpr::Read {
                        source: ScalarSource::Binding {
                            set: binding.set,
                            binding: binding.binding,
                        },
                        offset: 0,
                        ty,
                    },
                );
            }
        }
        let Some(generated) = body.generated.as_mut() else {
            return Err(error("dispatch buffers missing"));
        };
        generated.scalar_slots.insert(name, (binding, value.ty.clone()));
        saved.insert(
            name,
            match body.integer_literal(&value) {
                Some(n) => SavedValue::Constant(n, value.ty),
                None => SavedValue::Buffer(binding, value.ty),
            },
        );
    }
    Ok(())
}

fn output_array(
    body: &Body<'_, '_, '_>,
    results: &mut impl Iterator<Item = Value>,
) -> Result<Value, OptimizeError> {
    for result in results {
        let result = body.query().fields("Yield", result)?;
        if let Some(operand) = body.query().enode("ValueOperand", result[1])? {
            if let Some(var) = body.query().enode("Var", operand[0])? {
                let info = body.query().required("BindingInfo", (body.function, var[0]))?;
                if body.query().enode("ArrayInfo", info)?.is_some() {
                    return Ok(var[0]);
                }
            }
        }
    }
    Err(error("entry array result missing"))
}

fn output_array_parts(
    body: &Body<'_, '_, '_>,
    ty: &Type,
    results: &mut impl Iterator<Item = Value>,
) -> Result<(Value, Vec<Value>, Type), OptimizeError> {
    let mut payload = ty.clone();
    let mut offsets = Vec::new();
    while let Some((_, data)) = ragged_parts(&payload) {
        offsets.push(output_array(body, results)?);
        payload = data;
    }
    Ok((output_array(body, results)?, offsets, payload))
}

fn publish_offset_array(body: &mut Body<'_, '_, '_>, source: Value) -> Result<ScalarSource, OptimizeError> {
    let (_, _, layout) = body.array_descriptor(source)?;
    let dimensions = body.query().vector(layout[1])?;
    let [dimension] = dimensions.as_slice() else {
        return Err(error("segment offsets must have rank one"));
    };
    let dimension = body.query().fields("Dim", *dimension)?;
    let length = host::index(body.query(), &body.host_values, dimension[1])?;
    let count = body.index_expression(dimension[1])?;
    let bytes = Expr::Multiply(Box::new(length), Box::new(Expr::Integer(4)));
    let (binding, destination) = body.device_buffer(&types::i32(), bytes, count.clone())?;
    body.counted(
        Body::number(0),
        count,
        Body::number(1),
        vec![],
        |body, index, _| {
            let value = body.read_array(source, &[index.clone()])?;
            body.store(destination, index, value)?;
            Ok(vec![])
        },
    )?;
    Ok(scalar_source(binding))
}

fn output_type(
    body: &Body<'_, '_, '_>,
    ty: &Type,
    results: Value,
    publish_direct: bool,
) -> Result<Type, OptimizeError> {
    let mut results = body.query().vector(results)?.into_iter();
    let mut field = |ty: &Type| -> Result<Type, OptimizeError> {
        if ty.is_array() {
            let (name, _, _) = output_array_parts(body, ty, &mut results)?;
            if publish_direct {
                if let Some((_, storage)) = direct_output(body, name)? {
                    return Ok(storage);
                }
            }
            let (element, _, _) = body.array_descriptor(name)?;
            Ok(types::view_array_with_size(
                &storage_value_type(&element),
                Type::Constructed(TypeName::SizePlaceholder, vec![]),
                types::no_buffer(),
            ))
        } else {
            skip_result_values(ty, &mut results)?;
            Ok(storage_value_type(ty))
        }
    };
    match strip_existentials(ty) {
        Type::Constructed(name @ (TypeName::Tuple(_) | TypeName::Record(_)), fields) => {
            Ok(Type::Constructed(
                name.clone(),
                fields.iter().map(&mut field).collect::<Result<_, _>>()?,
            ))
        }
        ty => field(ty),
    }
}

fn direct_output(
    body: &Body<'_, '_, '_>,
    source: Value,
) -> Result<Option<(BindingRef, Type)>, OptimizeError> {
    if !body.query().contains("PublicArray", (body.function, source))? {
        return Ok(None);
    }
    let (_, block, _) = body.array_descriptor(source)?;
    let Some(tag) = body.integer_literal(&body.named(block)?) else {
        return Ok(None);
    };
    let Some(storage) = body.memories.get(&i32::try_from(tag).map_err(|_| error("invalid memory tag"))?)
    else {
        return Err(error("published memory block is missing"));
    };
    Ok(types::array_view_buffer(&storage.ty).map(|binding| (binding, storage.ty.clone())))
}

fn output_destinations(
    body: &Body<'_, '_, '_>,
    ty: &Type,
    results: Value,
) -> Result<Vec<Option<BindingRef>>, OptimizeError> {
    let fields = match strip_existentials(ty) {
        Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields) => fields.as_slice(),
        ty => std::slice::from_ref(ty),
    };
    let mut results = body.query().vector(results)?.into_iter();
    fields
        .iter()
        .map(|ty| {
            if ty.is_array() {
                let (source, _, _) = output_array_parts(body, ty, &mut results)?;
                Ok(direct_output(body, source)?.map(|(binding, _)| binding))
            } else {
                skip_result_values(ty, &mut results)?;
                Ok(None)
            }
        })
        .collect()
}

fn output_capacities(
    body: &Body<'_, '_, '_>,
    ty: &Type,
    results: Value,
) -> Result<Vec<BufferLen>, OptimizeError> {
    let fields = match strip_existentials(ty) {
        Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields) => fields.as_slice(),
        ty => std::slice::from_ref(ty),
    };
    let mut results = body.query().vector(results)?.into_iter();
    fields
        .iter()
        .map(|ty| {
            if ty.is_array() {
                let (name, _, _) = output_array_parts(body, ty, &mut results)?;
                let (element, _, layout) = body.array_descriptor(name)?;
                let Some(stride) = storage_elem_stride(&storage_value_type(&element)) else {
                    return Err(error("output array element stride missing"));
                };
                let mut bytes = Expr::Integer(stride.into());
                for dim in body.query().vector(layout[1])? {
                    let dim = body.query().fields("Dim", dim)?;
                    bytes = Expr::Multiply(
                        Box::new(bytes),
                        Box::new(
                            host::index(body.query(), &body.host_values, dim[1])
                                .map_err(|failure| error(format!("output {name:?}: {failure}")))?,
                        ),
                    );
                }
                Ok(BufferLen::Computed { bytes })
            } else {
                skip_result_values(ty, &mut results)?;
                let Some(bytes) = type_byte_size(&storage_value_type(ty)) else {
                    return Err(error("output value has no static size"));
                };
                Ok(BufferLen::Fixed { bytes: bytes.into() })
            }
        })
        .collect()
}

fn skip_result_values(ty: &Type, results: &mut impl Iterator<Item = Value>) -> Result<(), OptimizeError> {
    if let Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields) = strip_existentials(ty) {
        for field in fields {
            skip_result_values(field, results)?;
        }
    } else if results.next().is_none() {
        return Err(error("entry result arity mismatch"));
    }
    Ok(())
}

fn storage_view(
    body: &mut Body<'_, '_, '_>,
    binding: BindingRef,
    element: Type,
    count: Typed,
) -> Result<Typed, OptimizeError> {
    let zero = Body::number(0);
    let ty = types::view_array_with_size(
        &element,
        Type::Constructed(TypeName::SizePlaceholder, vec![]),
        buffer_tag(binding),
    );
    body.op(
        OpTag::StorageView(PureViewSource::Storage(binding)),
        vec![zero, count],
        ty,
    )
}

fn bind_input(
    body: &mut Body<'_, '_, '_>,
    mut input: EntryInput,
    ty: &Type,
    params: &mut impl Iterator<Item = Value>,
    inputs: &mut Vec<EntryInput>,
) -> Result<(), OptimizeError> {
    let scalar_storage = input.storage_binding().is_some() && !ty.is_array();
    if scalar_storage {
        input.ty = types::sized_array(1, input.ty.clone());
    }
    input.ty = storage_value_type(&input.ty);
    let param_ty = if let Some(binding) = input.storage_binding() {
        let Some(element) = input.ty.elem_type() else {
            return Err(error("storage input has no element"));
        };
        types::view_array_with_size(
            element,
            Type::Constructed(TypeName::SizePlaceholder, vec![]),
            buffer_tag(binding),
        )
    } else {
        input.ty.clone()
    };
    let parameter = body.builder.func_mut().add_function_param(param_ty.clone());
    let mut value = Typed {
        value: parameter.into(),
        ty: param_ty,
    };
    if let Some(binding) = input.storage_binding() {
        let length = if let Some(Type::Constructed(TypeName::Size(n), _)) = input.ty.array_size() {
            Body::number(u32::try_from(*n).map_err(|_| error("input size exceeds u32"))?)
        } else {
            body.op(
                OpTag::Intrinsic {
                    id: catalog().known().storage_len,
                    overload_idx: 0,
                },
                vec![Body::number(binding.set), Body::number(binding.binding)],
                Type::Constructed(TypeName::UInt(32), vec![]),
            )?
        };
        let Some(element) = value.ty.elem_type().cloned() else {
            return Err(error("input view element missing"));
        };
        value = storage_view(body, binding, element, length.clone())?;
        if scalar_storage {
            value = body.index(value, Body::number(0))?;
        }
        if ty.is_array() {
            bind_array(
                body,
                ty,
                value,
                length,
                Some(ScalarExpr::BufferLength {
                    source: ScalarSource::Binding {
                        set: binding.set,
                        binding: binding.binding,
                    },
                    stride: storage_elem_stride(
                        input.ty.elem_type().ok_or_else(|| error("input element missing"))?,
                    )
                    .ok_or_else(|| error("input element stride missing"))?,
                }),
                params,
            )?;
            inputs.push(input);
            return Ok(());
        }
    }
    if ty.is_array() {
        let Some(Type::Constructed(TypeName::Size(n), _)) = ty.array_size() else {
            return Err(error("value array input has no static extent"));
        };
        let length = Body::number(u32::try_from(*n).map_err(|_| error("value array extent exceeds u32"))?);
        bind_array(body, ty, value, length, None, params)?;
    } else {
        let host = input_host_value(&input, ty, 0)?;
        bind_value(body, ty, value, host, params)?;
    }
    inputs.push(input);
    Ok(())
}

fn bind_value(
    body: &mut Body<'_, '_, '_>,
    ty: &Type,
    value: Typed,
    host: Option<ScalarExpr>,
    params: &mut impl Iterator<Item = Value>,
) -> Result<(), OptimizeError> {
    if let Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields) = strip_existentials(ty) {
        for (i, ty) in fields.iter().enumerate() {
            let field = body.field(value.clone(), i)?;
            let host = host.as_ref().map(|value| match value {
                ScalarExpr::Tuple(fields) => fields[i].clone(),
                value => ScalarExpr::Field {
                    tuple: Box::new(value.clone()),
                    index: i,
                },
            });
            bind_value(body, ty, field, host, params)?;
        }
        return Ok(());
    }
    let Some(binding) = params.next() else {
        return Err(error("entry scalar binding missing"));
    };
    let Some((name, ty)) = super::runtime_binding(body.query(), binding)? else {
        return Err(error("entry scalar parameter is not scalar"));
    };
    let value = body.cast(value, &ty)?;
    body.values.insert(name, value);
    if let Some(host) = host {
        body.host_values.insert(name, host);
    }
    Ok(())
}

fn assign_index(body: &mut Body<'_, '_, '_>, index: Value, value: Typed) -> Result<(), OptimizeError> {
    if let Some(fields) = body.query().enode("IndexValue", index)? {
        let value = body.cast(value, &types::i32())?;
        body.values.insert(fields[0], value);
        return Ok(());
    }
    if let Some(fields) = body.query().enode("IndexConstant", index)? {
        let expected = body.query().0.value_to_base::<i64>(fields[0]);
        if body.integer_literal(&value) == Some(expected) {
            return Ok(());
        }
        return Err(error("entry dimension disagrees with its static shape"));
    }
    Err(error("entry layout witness is not a variable"))
}

fn bind_array(
    body: &mut Body<'_, '_, '_>,
    ty: &Type,
    value: Typed,
    outer_length: Typed,
    outer_host: Option<ScalarExpr>,
    params: &mut impl Iterator<Item = Value>,
) -> Result<(), OptimizeError> {
    let mut block_capacity = None;
    let info = loop {
        let Some(binding) = params.next() else {
            return Err(error("entry array parameter missing"));
        };
        if let Some(fields) = body.query().enode("BindBlock", binding)? {
            block_capacity = Some(fields[1]);
        }
        if let Some(fields) = body.query().enode("BindValue", binding)? {
            if let Some(info) = body.query().enode("ArrayInfo", fields[1])? {
                break info;
            }
        }
    };
    let storage = body.query().fields("ArrayIn", info[1])?;
    let layout = body.query().fields("Layout", storage[1])?;
    let mut shape = Vec::new();
    let mut host_shape = Vec::new();
    let mut element = ty;
    while element.is_array() {
        let rank = element.array_rank().unwrap_or(1);
        for i in 0..rank {
            let size = element.array_dim(i);
            host_shape.push(match size {
                Some(Type::Constructed(TypeName::Size(n), _)) => Some(ScalarExpr::I32(
                    i32::try_from(*n).map_err(|_| error("input dimension exceeds i32"))?,
                )),
                _ if shape.is_empty() => outer_host.clone(),
                _ => None,
            });
            shape.push(match size {
                Some(Type::Constructed(TypeName::Size(n), _)) => {
                    Body::number(u32::try_from(*n).map_err(|_| error("input dimension exceeds u32"))?)
                }
                _ if shape.is_empty() => outer_length.clone(),
                _ => return Err(error("nested input dimension has no ABI witness")),
            });
        }
        let Some(next) = element.elem_type() else {
            return Err(error("array element missing"));
        };
        element = next;
    }
    let dims = body.query().vector(layout[1])?;
    if dims.len() != shape.len() {
        return Err(error("input layout rank mismatch"));
    }
    let mut stride = Body::number(1);
    let mut host_stride = Some(ScalarExpr::I32(1));
    for ((dimension, length), host) in dims.into_iter().zip(shape).zip(host_shape).rev() {
        let dim = body.query().fields("Dim", dimension)?;
        assign_index(body, dim[0], stride.clone())?;
        assign_index(body, dim[1], length.clone())?;
        if let Some(host) = &host_stride {
            let fields = body.query().fields("IndexValue", dim[0])?;
            body.host_values.insert(fields[0], host.clone());
        }
        if let Some(host) = &host {
            if let Some(fields) = body.query().enode("IndexValue", dim[1])? {
                body.host_values.insert(fields[0], host.clone());
            }
        }
        host_stride = host_stride.zip(host).map(|(stride, length)| ScalarExpr::Apply {
            op: "mul".into(),
            ty: ScalarType::I32,
            args: vec![stride, length],
        });
        stride = body.binary(crate::op::BinaryOperator::Multiply, stride, length)?;
    }
    assign_index(body, layout[0], Body::number(0))?;
    let offset = body.query().fields("IndexValue", layout[0])?;
    body.host_values.insert(offset[0], ScalarExpr::I32(0));
    let Some(capacity) = block_capacity else {
        return Err(error("entry block capacity missing"));
    };
    let Some(bytes) = type_byte_size(&storage_value_type(element)) else {
        return Err(error("input element size missing"));
    };
    if let Some(elements) = host_stride {
        let name = body.query().fields("IndexValue", capacity)?[0];
        body.host_values.insert(
            name,
            ScalarExpr::Apply {
                op: "mul".into(),
                ty: ScalarType::I32,
                args: vec![
                    elements,
                    ScalarExpr::I32(i32::try_from(bytes).map_err(|_| error("element stride exceeds i32"))?),
                ],
            },
        );
    }
    let bytes = body.binary(crate::op::BinaryOperator::Multiply, stride, Body::number(bytes))?;
    assign_index(body, capacity, bytes)?;
    let handle = body.register_memory(storage[0], value)?;
    body.values.insert(storage[0], handle);
    Ok(())
}

fn write_results(
    body: &mut Body<'_, '_, '_>,
    ty: &Type,
    results: &mut impl Iterator<Item = Value>,
    outputs: &[EntryOutput],
) -> Result<bool, OptimizeError> {
    let mut writes = false;
    let fields = match strip_existentials(ty) {
        Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields) => fields.as_slice(),
        ty => std::slice::from_ref(ty),
    };
    for (index, (ty, output)) in fields.iter().zip(outputs).enumerate() {
        if ty.is_array() {
            let (source, _, _) = output_array_parts(body, ty, results)?;
            let Some(binding) = output.storage_binding() else {
                return Err(error("array result has no storage output"));
            };
            if direct_output(body, source)?.is_some_and(|(source, _)| source == binding) {
                continue;
            }
            writes = true;
            let shape = body.array_shape(source)?;
            let count = body.shape_count(&shape)?;
            let (element, _, _) = body.array_descriptor(source)?;
            let destination = storage_view(body, binding, storage_value_type(&element), count.clone())?;
            body.counted(
                Body::number(0),
                count,
                Body::number(1),
                vec![],
                |body, ordinal, _| {
                    let indices = body.coordinates(ordinal.clone(), &shape)?;
                    let value = body.read_array(source, &indices)?;
                    body.store(destination, ordinal, value)?;
                    Ok(vec![])
                },
            )?;
        } else {
            writes = true;
            let value = result_value(body, ty, results)?;
            if let Some(binding) = output.storage_binding() {
                let destination = storage_view(body, binding, output.ty.clone(), Body::number(1))?;
                body.store(destination, Body::number(0), value)?;
            } else {
                let value = body.cast(value, &output.ty)?;
                let place = body.builder.new_place(value.ty.clone());
                body.builder
                    .push_void_inst(InstKind::OutputSlot { index, result: place })
                    .map_err(builder_error)?;
                body.builder
                    .push_void_inst(InstKind::Store {
                        place,
                        value: value.value,
                    })
                    .map_err(builder_error)?;
            }
        }
    }
    Ok(writes)
}

fn result_value(
    body: &mut Body<'_, '_, '_>,
    ty: &Type,
    results: &mut impl Iterator<Item = Value>,
) -> Result<Typed, OptimizeError> {
    if let Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), types) = strip_existentials(ty) {
        let values =
            types.iter().map(|ty| result_value(body, ty, results)).collect::<Result<Vec<_>, _>>()?;
        return body.op(OpTag::Tuple, values, ty.clone());
    }
    let Some(result) = results.next() else {
        return Err(error("entry scalar result missing"));
    };
    let result = body.query().fields("Yield", result)?;
    let Some(value) = body.operand(result[1])? else {
        return Err(error("scalar result is not a runtime value"));
    };
    Ok(value)
}

fn input_host_value(
    input: &EntryInput,
    ty: &Type,
    offset: u32,
) -> Result<Option<ScalarExpr>, OptimizeError> {
    let physical = storage_value_type(ty);
    if let Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields) = ty {
        let rules =
            if input.uniform_binding().is_some() { StorageLayout::Std140 } else { StorageLayout::Std430 };
        let Some(layout) = block_layout(&physical, rules) else {
            return Err(error("host parameter has no aggregate layout"));
        };
        return fields
            .iter()
            .enumerate()
            .map(|(i, ty)| input_host_value(input, ty, offset + layout.member_offsets[i]))
            .collect::<Result<Option<Vec<_>>, _>>()
            .map(|fields| fields.map(ScalarExpr::Tuple));
    }
    let ty = match ty {
        Type::Constructed(TypeName::Bool, _) => ScalarType::Bool,
        Type::Constructed(TypeName::Int(32), _) => ScalarType::I32,
        Type::Constructed(TypeName::UInt(32), _) => ScalarType::U32,
        Type::Constructed(TypeName::Float(32), _) => ScalarType::F32,
        _ => return Ok(None),
    };
    let source = if let Some(slot) = input.push_constant() {
        ScalarSource::PushConstant {
            name: input.name.clone(),
            offset: slot.offset,
        }
    } else if let Some(binding) = input.storage_binding().or_else(|| input.uniform_binding()) {
        ScalarSource::Binding {
            set: binding.set,
            binding: binding.binding,
        }
    } else {
        return Ok(None);
    };
    Ok(Some(ScalarExpr::Parameter { source, offset, ty }))
}
