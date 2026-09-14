//! Target-independent logical resource allocation.
//!
//! This pass resolves legal in-place destinations, materializes values that
//! must survive a scheduling boundary, and assigns logical sizes. Physical
//! descriptor selection remains the responsibility of target planning.

mod cost;
mod host_length;
mod residency;

pub use residency::resolve_residency;

use crate::egir::soac::SegmentedMetadata;

use crate::ast::TypeName;
use crate::interface;
use crate::interface::{EntryInputKind, EntryOutputKind};
use crate::pipeline_descriptor::BufferLen;
use crate::types::TypeExt;
use crate::BindingRef;
use polytype::Type;
use std::collections::{HashMap, HashSet};

use super::from_tlc::ConvertError;
use super::ir::{PlaceId, RemapBlockIds};
use super::program::{
    AllocatedEntry, CompilerResource, CompilerResourceKind, Entry, HostResource, HostSizePolicy,
    LogicalResourceArena, LogicalSize, Program, ResidencyProgramData, ResourceId, ResourceProgramData,
    RewriteGlobal, SemanticResourceDecl, SemanticResourceRef, StageOrigin,
};
use super::semantic_opt::Optimized;
use super::soac::{filter, hist, screma};
use super::types::{Semantic, SideEffectKind, Soac, SoacEffect, ValueId};

#[derive(Debug)]
pub enum ResidencyDraftTag {}
pub type ResidencyDraft = super::program::Program<
    ResidencyDraftTag,
    super::ir::ProgramFamily<
        Semantic<SemanticResourceRef>,
        super::program::SemanticResourceDecl,
        super::ir::RealizedOutputRoute,
        ResidencyProgramData,
    >,
    RewriteGlobal,
>;

/// Replace pre-allocation descriptor bindings with target-independent logical
/// resource identities. This is intentionally the first pass allowed to own
/// or create resources.
pub fn allocate_semantic_resources(mut program: Optimized) -> Result<ResidencyDraft, ConvertError> {
    host_length::retain_output_lengths(&mut program)?;
    let mut builder = ResourceAllocationBuilder::default();
    reserve_host_resources(&program, &mut builder)?;
    lower_host_size_policies(&program, &mut builder)?;
    let context = builder.finalize()?;
    let program = remap_program_resources(program, context)?;
    realize_dynamic_publication(program)
}

impl LogicalResourceArena {
    fn resource_for_binding(&self, binding: BindingRef) -> Result<SemanticResourceRef, ConvertError> {
        self.host_resource(binding).map(SemanticResourceRef).ok_or_else(|| {
            ConvertError::GraphError(format!(
                "resource binding set={} binding={} is not declared by an entry interface",
                binding.set, binding.binding
            ))
        })
    }
    fn logical_size(&self, length: Option<&BufferLen>) -> Result<HostSizePolicy, ConvertError> {
        Ok(match length {
            Some(BufferLen::HostProvided { inputs, elem_bytes }) => {
                HostSizePolicy::Known(LogicalSize::HostProvided {
                    inputs: inputs.clone(),
                    elem_bytes: *elem_bytes,
                })
            }
            Some(BufferLen::Fixed { bytes }) => HostSizePolicy::Known(LogicalSize::FixedBytes(*bytes)),
            Some(BufferLen::LikeInput {
                set,
                binding,
                elem_bytes,
                src_elem_bytes,
            }) => HostSizePolicy::Known(LogicalSize::LikeResource {
                resource: self.resource_for_binding(BindingRef::new(*set, *binding))?.0,
                elem_bytes: *elem_bytes,
                src_elem_bytes: *src_elem_bytes,
            }),
            Some(BufferLen::SameAsDispatch { elem_bytes }) => {
                HostSizePolicy::Known(LogicalSize::SameAsDispatch {
                    elem_bytes: *elem_bytes,
                })
            }
            None => HostSizePolicy::RuntimeProvided,
        })
    }
}

/// Resource identities are allocated once. Pending policies cannot escape finalization.
#[derive(Default)]
struct ResourceAllocationBuilder {
    resources: LogicalResourceArena,
    pending: std::collections::HashSet<ResourceId>,
}

impl ResourceAllocationBuilder {
    fn reserve_host(&mut self, binding: BindingRef, elem_ty: Type<TypeName>) -> Result<ResourceId, String> {
        if let Some(id) = self.resources.host_resource(binding) {
            let current = self.resources[id].elem_ty();
            if current != &elem_ty {
                return Err(format!(
                    "host resource set={} binding={} has conflicting element types: {:?} and {:?}",
                    binding.set, binding.binding, current, elem_ty
                ));
            }
            return Ok(id);
        }
        let id = self.resources.allocate_host(
            HostResource { binding, name: None },
            elem_ty,
            HostSizePolicy::RuntimeProvided,
        );
        self.pending.insert(id);
        Ok(id)
    }

    fn set_host_size(&mut self, binding: BindingRef, size: HostSizePolicy) -> Result<(), String> {
        let id = self.resources.host_resource(binding).ok_or_else(|| {
            format!(
                "host resource set={} binding={} must be reserved before its size is set",
                binding.set, binding.binding
            )
        })?;
        if let HostSizePolicy::Known(proposed) = size {
            self.resources.get_mut(id).refine_size(&proposed).map_err(|current| {
                format!(
                    "host resource set={} binding={} has conflicting size policies: {:?} and {:?}",
                    binding.set, binding.binding, current, proposed
                )
            })?;
        }
        self.pending.remove(&id);
        Ok(())
    }

    fn finalize(self) -> Result<LogicalResourceArena, ConvertError> {
        if let Some(resource) = self.resources.iter().find(|resource| self.pending.contains(&resource.id()))
        {
            let binding = resource.host_binding().expect("reserved host resource");
            return Err(ConvertError::GraphError(format!(
                "host resource set={} binding={} was reserved but its size policy was not processed",
                binding.set, binding.binding
            )));
        }
        Ok(self.resources)
    }
}

fn remap_program_resources(
    program: Optimized,
    context: LogicalResourceArena,
) -> Result<ResidencyDraft, ConvertError> {
    let Program {
        functions,
        externs,
        entry_points,
        constants,
        data,
        global_context,
        state: _,
    } = program;

    let functions = functions
        .into_iter()
        .map(|function| remap_function_resources(function, &context))
        .collect::<Result<Vec<_>, _>>()?;
    let constants = constants
        .into_iter()
        .map(|constant| remap_constant_resources(constant, &context))
        .collect::<Result<Vec<_>, _>>()?;
    let entry_points = entry_points
        .into_iter()
        .map(|entry| remap_entry_resources(entry, &context))
        .collect::<Result<Vec<_>, _>>()?;

    let mut stages = super::program::StagedProgramBuilder::new();
    for entry in entry_points {
        stages
            .add_stage(StageOrigin::Authored, entry)
            .map_err(|error| ConvertError::Internal(error.to_string()))?;
    }

    Ok(Program::from_parts(
        functions,
        externs,
        Vec::new(),
        constants,
        ResidencyProgramData {
            core: ResourceProgramData {
                pipeline: data.pipeline,
                stage_entries: data.stage_entries,
                resources: context,
                identities: data.identities,
            },
            stages,
            resident_flows: HashMap::new(),
        },
        global_context,
    ))
}

struct InterfaceResource<'a> {
    binding: BindingRef,
    role: Option<interface::StorageRole>,
    elem_ty: &'a Type<TypeName>,
    length: Option<&'a BufferLen>,
}

fn interface_resources(entry: &Entry<Semantic>) -> impl Iterator<Item = InterfaceResource<'_>> {
    let inputs = entry.inputs.iter().filter_map(|input| {
        let binding = input.resource?;
        let (role, length) = match &input.kind {
            EntryInputKind::Storage { length, .. } => {
                (Some(interface::StorageRole::Input), length.as_ref())
            }
            _ => (None, None),
        };
        Some(InterfaceResource {
            binding,
            role,
            elem_ty: input.ty.elem_type().unwrap_or(&input.ty),
            length,
        })
    });
    let outputs = entry.outputs.iter().filter_map(|output| {
        let binding = output.resource?;
        let length = match &output.kind {
            EntryOutputKind::Storage { length, .. } => length.as_ref(),
            _ => None,
        };
        Some(InterfaceResource {
            binding,
            role: Some(interface::StorageRole::Output),
            elem_ty: output.ty.elem_type().unwrap_or(&output.ty),
            length,
        })
    });
    inputs.chain(outputs)
}

fn reserve_host_resources(
    program: &Optimized,
    builder: &mut ResourceAllocationBuilder,
) -> Result<(), ConvertError> {
    for entry in &program.entry_points {
        for resource in interface_resources(entry) {
            builder
                .reserve_host(resource.binding, resource.elem_ty.clone())
                .map_err(ConvertError::GraphError)?;
        }
    }
    Ok(())
}

fn lower_host_size_policies(
    program: &Optimized,
    builder: &mut ResourceAllocationBuilder,
) -> Result<(), ConvertError> {
    for entry in &program.entry_points {
        for resource in interface_resources(entry) {
            let size = builder.resources.logical_size(resource.length)?;
            builder.set_host_size(resource.binding, size).map_err(ConvertError::GraphError)?;
        }
    }
    Ok(())
}

fn entry_resource_declarations(
    entry: &Entry<Semantic>,
    context: &LogicalResourceArena,
) -> Result<Vec<SemanticResourceDecl>, ConvertError> {
    let mut declarations = crate::StableMap::<SemanticResourceRef, interface::StorageRole>::new();
    for item in interface_resources(entry) {
        let Some(role) = item.role else { continue };
        let resource = context.resource_for_binding(item.binding)?;
        declarations.entry(resource).and_modify(|current| *current = current.merge(role)).or_insert(role);
    }
    Ok(declarations.into_iter().map(|(resource, role)| SemanticResourceDecl { resource, role }).collect())
}

fn allocate_type_resources(
    ty: &mut Type<TypeName>,
    context: &LogicalResourceArena,
) -> Result<(), ConvertError> {
    let mut error = None;
    super::program::visit_type_names_mut(ty, |name| {
        if let TypeName::Buffer(binding) = *name {
            match context.resource_for_binding(binding) {
                Ok(resource) => *name = TypeName::Resource(resource.0),
                Err(binding_error) => error = Some(binding_error),
            }
        }
    });
    error.map_or(Ok(()), Err)
}

fn remap_soac_resources(
    soac: Soac<Semantic>,
    nodes: &crate::LookupMap<ValueId, ValueId>,
    places: &crate::LookupMap<PlaceId, PlaceId>,
    context: &LogicalResourceArena,
) -> Result<Soac<Semantic<SemanticResourceRef>>, ConvertError> {
    let mut remap =
        super::soac::remap::Remap::new(nodes, places, |binding| context.resource_for_binding(binding));
    Ok(match soac {
        Soac::Screma(screma::Op {
            inputs,
            form,
            result_state,
            state,
        }) => Soac::Screma(screma::Op {
            inputs,
            form: remap.screma_form(form),
            result_state,
            state: match state {
                screma::SemanticState::Serial => screma::SemanticState::Serial,
                screma::SemanticState::Segmented(segment) => {
                    screma::SemanticState::Segmented(remap.segment(segment)?)
                }
            },
        }),
        Soac::Filter(filter::Op { body, state }) => {
            let segment = remap.segment(state.segment)?;
            Soac::Filter(filter::Op {
                body: remap.filter_body(body),
                state: filter::SemanticState {
                    segment,
                    output: remap.filter_output(state.output)?,
                },
            })
        }
        Soac::Hist(hist::Op { inputs, form, state }) => Soac::Hist(hist::Op {
            inputs,
            form: remap.hist_form(form),
            state: match state {
                hist::SemanticState::Serial => hist::SemanticState::Serial,
                hist::SemanticState::Segmented(space) => {
                    hist::SemanticState::Segmented(remap.space(space)?)
                }
            },
        }),
    })
}

fn remap_graph_resources(
    graph: super::types::EGraph<Semantic>,
    context: &LogicalResourceArena,
) -> Result<
    (
        super::types::EGraph<Semantic<SemanticResourceRef>>,
        crate::LookupMap<crate::flow::BlockId, crate::flow::BlockId>,
    ),
    ConvertError,
> {
    let (mut graph, _, blocks) = graph.try_map_resources_and_phase(
        |binding| context.resource_for_binding(binding),
        |owner, soac, nodes, places| {
            Ok::<_, ConvertError>((owner, remap_soac_resources(soac, nodes, places, context)?))
        },
    )?;
    let mut type_error = None;
    super::program::rewrite_graph_types(&mut graph, |ty| {
        if let Err(error) = allocate_type_resources(ty, context) {
            type_error = Some(error);
        }
    });
    if let Some(error) = type_error {
        return Err(error);
    }
    Ok((graph, blocks))
}

fn realize_dynamic_publication(mut program: ResidencyDraft) -> Result<ResidencyDraft, ConvertError> {
    let resources = &mut program.data.core.resources;
    for function in &mut program.functions {
        let _ = realize_graph_dynamic_publication(&mut function.graph, resources)
            .map_err(ConvertError::GraphError)?;
    }
    for constant in &mut program.constants {
        let _ = realize_graph_dynamic_publication(&mut constant.graph, resources)
            .map_err(ConvertError::GraphError)?;
    }
    for entry in program.data.stages.stage_bodies_mut() {
        let filter_data = realize_graph_dynamic_publication(&mut entry.graph, resources)
            .map_err(ConvertError::GraphError)?;
        realize_filter_output_capacities(entry, resources, &filter_data)
            .map_err(ConvertError::GraphError)?;
    }
    Ok(program)
}

fn filter_capacity_size(
    owner: super::program::SemanticOpId,
    space: &super::types::SegSpace<SemanticResourceRef>,
    elem_ty: &Type<TypeName>,
) -> Result<LogicalSize, String> {
    LogicalSize::for_space(space, elem_ty).ok_or_else(|| {
        format!("runtime Filter {owner:?} has no legal storage layout for element type {elem_ty:?}")
    })
}

fn require_filter_resource(
    resources: &mut LogicalResourceArena,
    owner: super::program::SemanticOpId,
    resource: ResourceId,
    role: &str,
    elem_ty: &Type<TypeName>,
    size: &LogicalSize,
) -> Result<(), String> {
    let actual = resources.get_mut(resource);
    if actual.elem_ty() != elem_ty {
        return Err(format!(
            "runtime Filter {owner:?} {role} resource {resource:?} has element type {:?}, expected {elem_ty:?}",
            actual.elem_ty()
        ));
    }
    if let Err(actual_size) = actual.refine_size(size) {
        return Err(format!(
            "runtime Filter {owner:?} {role} resource {resource:?} has logical size {actual_size:?}, expected {size:?}"
        ));
    }
    Ok(())
}

fn bind_filter_storage(
    resources: &mut LogicalResourceArena,
    owner: super::program::SemanticOpId,
    elem_ty: Type<TypeName>,
    size: LogicalSize,
    backing: Option<ResourceId>,
    length: Option<ResourceId>,
) -> Result<filter::RuntimeStorage<ResourceId>, String> {
    let data = backing.unwrap_or_else(|| {
        resources.allocate_compiler(
            CompilerResource::new(CompilerResourceKind::FilterData, Some(owner), 0),
            elem_ty.clone(),
            size.clone(),
        )
    });
    require_filter_resource(resources, owner, data, "data", &elem_ty, &size)?;

    let length_ty = Type::Constructed(TypeName::UInt(32), Vec::new());
    let length_size = LogicalSize::FixedBytes(4);
    let length = length.unwrap_or_else(|| {
        resources.allocate_compiler(
            CompilerResource::new(CompilerResourceKind::FilterLenCell, Some(owner), 1),
            length_ty.clone(),
            length_size.clone(),
        )
    });
    require_filter_resource(resources, owner, length, "length", &length_ty, &length_size)?;
    Ok(filter::RuntimeStorage { data, length })
}

fn realize_graph_dynamic_publication(
    graph: &mut super::types::EGraph<Semantic<SemanticResourceRef>>,
    resources: &mut LogicalResourceArena,
) -> Result<Vec<ResourceId>, String> {
    let mut filter_data = Vec::new();
    let mut result_types = Vec::new();
    for (_, block) in &mut graph.skeleton.blocks {
        for effect in &mut block.side_effects {
            let SideEffectKind::Soac(SoacEffect(
                owner,
                Soac::Filter(filter::Op {
                    body,
                    state:
                        filter::SemanticState {
                            segment:
                                SegmentedMetadata {
                                    space,
                                    output_slots,
                                    resources: accesses,
                                },
                            output: filter::Output::Runtime(runtime),
                            ..
                        },
                    ..
                }),
            )) = &mut effect.kind
            else {
                continue;
            };
            if !output_slots.is_empty() && matches!(runtime.backing, filter::RuntimeBacking::Deferred) {
                runtime.backing = accesses
                    .iter()
                    .find(|access| access.access != crate::ResourceAccess::Read)
                    .map(|access| filter::RuntimeBacking::Bound(access.resource))
                    .unwrap_or(filter::RuntimeBacking::Deferred);
            }
            let filter::RuntimeBacking::Bound(backing) = runtime.backing else {
                continue;
            };
            let elem_ty = body.output_element_type();
            if !output_slots.is_empty() {
                let length = match runtime.length {
                    filter::RuntimeLength::Implicit => None,
                    filter::RuntimeLength::Stored(length) => Some(length.0),
                };
                let size = filter_capacity_size(*owner, space, &elem_ty)?;
                let storage =
                    bind_filter_storage(resources, *owner, elem_ty.clone(), size, Some(backing.0), length)?;
                filter_data.push(storage.data);
                runtime.backing = filter::RuntimeBacking::Bound(SemanticResourceRef(storage.data));
                runtime.length = filter::RuntimeLength::Stored(SemanticResourceRef(storage.length));
                *accesses = super::types::SegResourceAccess::merge(
                    accesses,
                    &[super::types::SegResourceAccess {
                        resource: SemanticResourceRef(storage.length),
                        access: crate::ResourceAccess::Write,
                    }],
                );
            }
            // Bound results need storage-backed types even without host publication.
            if let Some(result) = effect.result.as_ref().and_then(|result| result.single_value()) {
                result_types.push((
                    result,
                    crate::types::view_array_of(
                        &elem_ty,
                        Type::Constructed(TypeName::Resource(backing.0), Vec::new()),
                    ),
                ));
            }
        }
    }
    for (result, ty) in result_types {
        graph.retype_node(result, ty);
    }
    Ok(filter_data)
}

fn filter_capacity_buffer_len(
    resources: &LogicalResourceArena,
    size: &LogicalSize,
) -> Result<BufferLen, String> {
    Ok(match size {
        LogicalSize::HostProvided { inputs, elem_bytes } => BufferLen::HostProvided {
            inputs: inputs.clone(),
            elem_bytes: *elem_bytes,
        },
        LogicalSize::FixedBytes(bytes) => BufferLen::Fixed { bytes: *bytes },
        LogicalSize::LikeResource {
            resource,
            elem_bytes,
            src_elem_bytes,
        } => {
            let binding = resources[*resource].host_binding().ok_or_else(|| {
                format!("host Filter output capacity depends on non-host resource {resource:?}")
            })?;
            BufferLen::LikeInput {
                set: binding.set,
                binding: binding.binding,
                elem_bytes: *elem_bytes,
                src_elem_bytes: *src_elem_bytes,
            }
        }
        LogicalSize::SameAsDispatch { elem_bytes } => BufferLen::SameAsDispatch {
            elem_bytes: *elem_bytes,
        },
    })
}

fn realize_filter_output_capacities(
    entry: &mut AllocatedEntry,
    resources: &LogicalResourceArena,
    filter_data: &[ResourceId],
) -> Result<(), String> {
    for output in &mut entry.outputs {
        let Some(SemanticResourceRef(resource)) = output.resource else {
            continue;
        };
        if !filter_data.contains(&resource) {
            continue;
        }
        let capacity = filter_capacity_buffer_len(
            resources,
            resources[resource].size().expect("runtime Filter data size was refined while binding storage"),
        )?;
        *output.storage_length_mut().ok_or_else(|| {
            format!(
                "entry `{}` publishes runtime Filter data {resource:?} through a non-storage output",
                entry.name
            )
        })? = Some(capacity);
    }
    Ok(())
}

fn remap_function_resources(
    function: super::program::Func<Semantic>,
    context: &LogicalResourceArena,
) -> Result<super::program::AllocatedFunc, ConvertError> {
    let super::program::Func {
        region,
        name,
        span,
        linkage_name,
        params,
        mut result,
        effects,
        graph,
        abi: _,
    } = function;
    let (graph, _) = remap_graph_resources(graph, context)?;
    let params = params.try_map(
        &mut |binding| context.resource_for_binding(binding),
        &mut |mut ty| {
            allocate_type_resources(&mut ty, context)?;
            Ok(ty)
        },
    )?;
    let mut result_error = None;
    result.for_each_type_mut(|ty| {
        if let Err(error) = allocate_type_resources(ty, context) {
            result_error = Some(error);
        }
    });
    if let Some(error) = result_error {
        return Err(error);
    }
    Ok(super::program::Func::new(
        region,
        name,
        span,
        linkage_name,
        params,
        result,
        effects,
        graph,
    ))
}

fn remap_constant_resources(
    constant: super::program::ConstantDef<Semantic>,
    context: &LogicalResourceArena,
) -> Result<super::program::AllocatedConstantDef, ConvertError> {
    let super::program::ConstantDef {
        id,
        name,
        span,
        mut return_ty,
        graph,
    } = constant;
    let (graph, _) = remap_graph_resources(graph, context)?;
    allocate_type_resources(&mut return_ty, context)?;
    Ok(super::program::ConstantDef {
        id,
        name,
        span,
        return_ty,
        graph,
    })
}

fn remap_entry_resources(
    entry: Entry<Semantic>,
    context: &LogicalResourceArena,
) -> Result<AllocatedEntry, ConvertError> {
    let declarations = entry_resource_declarations(&entry, context)?;
    let Entry {
        id,
        name,
        span,
        execution_model,
        inputs,
        parameter_inputs,
        outputs,
        internal_results,
        resource_declarations: _,
        params,
        mut result,
        graph,
    } = entry;
    let (graph, blocks) = remap_graph_resources(graph, context)?;
    let inputs = inputs
        .into_iter()
        .map(|mut input| {
            allocate_type_resources(&mut input.ty, context)?;
            Ok(super::ir::EntryInput {
                inner: input.inner,
                resource: input
                    .resource
                    .map(|binding| context.resource_for_binding(binding))
                    .transpose()?,
            })
        })
        .collect::<Result<Vec<_>, ConvertError>>()?;
    let outputs = outputs
        .into_iter()
        .map(|mut output| {
            allocate_type_resources(&mut output.ty, context)?;
            for route in &mut output.routes {
                route.remap_block_ids(&blocks);
            }
            Ok(super::ir::EntryOutput {
                inner: output.inner,
                resource: output
                    .resource
                    .map(|binding| context.resource_for_binding(binding))
                    .transpose()?,
                routes: output.routes,
            })
        })
        .collect::<Result<Vec<_>, ConvertError>>()?;
    let internal_results = internal_results
        .into_iter()
        .map(|mut result| {
            result.route.remap_block_ids(&blocks);
            Ok(super::ir::InternalResultRoute {
                resource: context.resource_for_binding(result.resource)?,
                route: result.route,
            })
        })
        .collect::<Result<Vec<_>, ConvertError>>()?;
    let params = params.try_map(
        &mut |binding| context.resource_for_binding(binding),
        &mut |mut ty| {
            allocate_type_resources(&mut ty, context)?;
            Ok(ty)
        },
    )?;
    let mut result_error = None;
    result.for_each_type_mut(|ty| {
        if let Err(error) = allocate_type_resources(ty, context) {
            result_error = Some(error);
        }
    });
    if let Some(error) = result_error {
        return Err(error);
    }
    Ok(Entry {
        id,
        name,
        span,
        execution_model,
        inputs,
        parameter_inputs,
        outputs,
        internal_results,
        resource_declarations: declarations,
        params,
        result,
        graph,
    })
}

pub fn finalize_staged_ir(
    program: ResidencyDraft,
    profile: crate::LoweringProfile,
) -> Result<super::parallelize::KernelRecipesPlanned, ConvertError> {
    use super::program::{KernelProgramData, PlannedPublication, StageOrigin};

    let Program {
        functions,
        externs,
        entry_points: _,
        constants,
        mut data,
        global_context,
        state: _,
    } = program;
    for resource in &data.core.resources {
        if let Some(LogicalSize::LikeResource { resource: source, .. }) = resource.size() {
            if !data.core.resources.contains(*source) {
                return Err(ConvertError::Internal(format!(
                    "resource {:?} has a logical size that references missing source {source:?}",
                    resource.id()
                )));
            }
        }
    }
    for (_, _, entry) in data.stages.stages() {
        for declaration in &entry.resource_declarations {
            if !data.core.resources.contains(declaration.resource.0) {
                return Err(ConvertError::Internal(format!(
                    "entry `{}` declares missing resource {:?}",
                    entry.name, declaration.resource.0
                )));
            }
        }
        for input in &entry.inputs {
            if let Some(resource) = input.resource {
                if !data.core.resources.contains(resource.0) {
                    return Err(ConvertError::Internal(format!(
                        "entry `{}` input references missing resource {:?}",
                        entry.name, resource.0
                    )));
                }
            }
        }
        for output in &entry.outputs {
            if let Some(resource) = output.resource {
                if !data.core.resources.contains(resource.0) {
                    return Err(ConvertError::Internal(format!(
                        "entry `{}` output references missing resource {:?}",
                        entry.name, resource.0
                    )));
                }
            }
        }
        for result in &entry.internal_results {
            if !data.core.resources.contains(result.resource.0) {
                return Err(ConvertError::Internal(format!(
                    "entry `{}` internal result references missing resource {:?}",
                    entry.name, result.resource.0
                )));
            }
        }
        for (slot, output) in entry.outputs.iter().enumerate() {
            if output.routes.is_empty() {
                return Err(ConvertError::Internal(format!(
                    "entry `{}` output slot {slot} has no explicit route",
                    entry.name
                )));
            }
            if output.routes.iter().any(|route| route.writers.is_empty()) {
                return Err(ConvertError::Internal(format!(
                    "entry `{}` output slot {slot} has a source value but no producer",
                    entry.name
                )));
            }
        }
    }
    let mut consumers = Vec::new();
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    for (stage, _, entry) in data.stages.stages() {
        for declaration in entry.resource_declarations.iter().filter(|declaration| declaration.role.reads())
        {
            if let Some(flow) = data.resident_flows.get(&declaration.resource.0).copied() {
                if data.stages.flow(flow).is_some_and(|resident| {
                    resident.producer() != stage && !resident.consumers().contains(&stage)
                }) {
                    consumers.push((flow, stage));
                }
            }
        }
        inputs.extend(entry.inputs.iter().filter_map(|input| {
            let resource = input.resource?.0;
            data.core.resources[resource].host_binding()?;
            Some((
                stage,
                input.ty.clone(),
                super::program::ResidentStorage {
                    data: resource,
                    length: None,
                },
            ))
        }));
        outputs.extend(entry.outputs.iter().filter_map(|output| {
            let resource = output.resource?.0;
            data.core.resources[resource].host_binding()?;
            Some((
                stage,
                output.ty.clone(),
                super::program::ResidentStorage {
                    data: resource,
                    length: None,
                },
            ))
        }));
    }
    for (flow, stage) in consumers {
        data.stages.add_consumer(flow, stage).map_err(|error| ConvertError::Internal(error.to_string()))?;
    }
    for (stage, ty, storage) in inputs {
        data.stages
            .add_external_input(ty, storage, [stage])
            .map_err(|error| ConvertError::Internal(error.to_string()))?;
    }
    for (stage, ty, storage) in outputs {
        let flow = data
            .stages
            .add_flow(stage, ty, storage)
            .map_err(|error| ConvertError::Internal(error.to_string()))?;
        data.stages.publish(flow).map_err(|error| ConvertError::Internal(error.to_string()))?;
    }
    let mut reserved_bindings = data
        .stages
        .stages()
        .flat_map(|(_, _, entry)| entry.inputs.iter().filter_map(|input| input.descriptor_binding()))
        .collect::<HashSet<_>>();
    for pipeline in &data.core.pipeline.pipelines {
        let crate::pipeline_descriptor::Pipeline::Graphics(graphics) = pipeline else {
            continue;
        };
        for buffer in [
            graphics.invocation.draw.indices(),
            graphics.invocation.draw.indirect_commands(),
        ]
        .into_iter()
        .flatten()
        {
            reserved_bindings.insert(BindingRef::new(buffer.set, buffer.binding));
        }
    }
    let stages = data.stages.finish().map_err(|error| ConvertError::Internal(error.to_string()))?;
    let program = Program::from_parts(
        functions,
        externs,
        Vec::new(),
        constants,
        KernelProgramData {
            core: data.core,
            profile,
            reserved_bindings,
            topology: stages,
        },
        global_context,
    );
    super::parallelize::validate_finalized_stages(&program)?;
    let (program, stages) = program.split_topology();
    let stages = stages.try_map_stage_bodies(|_, origin, mut entry| {
        let dispatch = super::parallelize::stage_dispatch(&program.data.core, origin, &entry)
            .map_err(|error| ConvertError::Internal(error.to_string()))?;
        let publication =
            matches!(origin, StageOrigin::Authored).then(|| PlannedPublication::from_semantic(&entry));
        entry.bind_mapped_output_destinations().map_err(ConvertError::Internal)?;
        super::parallelize::planning::construct_stage(&program, entry, origin, publication, dispatch)
            .map_err(ConvertError::from)
    })?;
    Ok(program.with_topology(stages))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_ty() -> Type<TypeName> {
        Type::Constructed(TypeName::Unit, Vec::new())
    }

    fn u32_ty() -> Type<TypeName> {
        Type::Constructed(TypeName::UInt(32), Vec::new())
    }

    #[test]
    fn resource_draft_cannot_finalize_before_policy_lowering() {
        let binding = BindingRef::new(2, 4);
        let mut builder = ResourceAllocationBuilder::default();
        builder.reserve_host(binding, unit_ty()).unwrap();

        let Err(error) = builder.finalize() else {
            panic!("unfinished resource draft must fail")
        };
        assert!(
            error.to_string().contains("size policy was not processed"),
            "{error}"
        );
    }

    #[test]
    fn processed_unspecified_policy_survives_finalization() {
        let binding = BindingRef::new(2, 4);
        let mut builder = ResourceAllocationBuilder::default();
        let resource = builder.reserve_host(binding, unit_ty()).unwrap();
        builder.set_host_size(binding, HostSizePolicy::RuntimeProvided).unwrap();

        let context = builder.finalize().expect("processed draft must finalize");
        assert_eq!(context[resource].size(), None);
    }

    #[test]
    fn resource_draft_rejects_policy_for_unreserved_binding() {
        let binding = BindingRef::new(2, 4);
        let mut builder = ResourceAllocationBuilder::default();

        let error = builder
            .set_host_size(binding, HostSizePolicy::Known(LogicalSize::FixedBytes(16)))
            .expect_err("unreserved binding must fail");
        assert!(error.contains("must be reserved"), "{error}");
    }

    #[test]
    fn input_and_output_roles_merge_to_explicit_read_write() {
        assert_eq!(
            interface::StorageRole::Input.merge(interface::StorageRole::Output),
            interface::StorageRole::InputOutput
        );
    }

    #[test]
    fn filter_storage_rejects_existing_capacity_mismatch_inline() {
        let owner = super::super::program::SemanticOpId::for_test(3);
        let mut resources = LogicalResourceArena::default();
        let backing = resources.allocate_host(
            HostResource {
                binding: BindingRef::new(0, 1),
                name: None,
            },
            u32_ty(),
            HostSizePolicy::Known(LogicalSize::FixedBytes(8)),
        );

        let error = bind_filter_storage(
            &mut resources,
            owner,
            u32_ty(),
            LogicalSize::FixedBytes(16),
            Some(backing),
            None,
        )
        .expect_err("an incompatible published capacity must fail while it is bound");

        assert!(error.contains("data resource"), "{error}");
        assert!(error.contains("logical size"), "{error}");
    }

    #[test]
    fn filter_storage_refines_unspecified_capacity_inline() {
        let owner = super::super::program::SemanticOpId::for_test(5);
        let mut resources = LogicalResourceArena::default();
        let backing = resources.allocate_host(
            HostResource {
                binding: BindingRef::new(0, 1),
                name: None,
            },
            u32_ty(),
            HostSizePolicy::RuntimeProvided,
        );

        bind_filter_storage(
            &mut resources,
            owner,
            u32_ty(),
            LogicalSize::FixedBytes(16),
            Some(backing),
            None,
        )
        .expect("an unresolved capacity should be completed while it is bound");

        assert_eq!(resources[backing].size(), Some(&LogicalSize::FixedBytes(16)));
    }

    #[test]
    fn filter_storage_rejects_existing_length_mismatch_inline() {
        let owner = super::super::program::SemanticOpId::for_test(4);
        let mut resources = LogicalResourceArena::default();
        let backing = resources.allocate_host(
            HostResource {
                binding: BindingRef::new(0, 1),
                name: None,
            },
            u32_ty(),
            HostSizePolicy::Known(LogicalSize::FixedBytes(16)),
        );
        let length = resources.allocate_host(
            HostResource {
                binding: BindingRef::new(0, 2),
                name: None,
            },
            u32_ty(),
            HostSizePolicy::Known(LogicalSize::FixedBytes(8)),
        );

        let error = bind_filter_storage(
            &mut resources,
            owner,
            u32_ty(),
            LogicalSize::FixedBytes(16),
            Some(backing),
            Some(length),
        )
        .expect_err("an incompatible stored length must fail while it is bound");

        assert!(error.contains("length resource"), "{error}");
        assert!(error.contains("logical size"), "{error}");
    }
}
