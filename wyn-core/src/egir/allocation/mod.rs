//! Target-independent logical resource allocation.
//!
//! This pass resolves legal in-place destinations, materializes values that
//! must survive a scheduling boundary, and assigns logical sizes. Physical
//! descriptor selection remains the responsibility of target planning.

mod cost;
mod residency;

use crate::ast::TypeName;
use crate::interface;
use crate::interface::{EntryInputKind, EntryOutputKind};
use crate::pipeline_descriptor;
use crate::pipeline_descriptor::BufferLen;
use crate::ssa;
use crate::types::TypeExt;
use crate::BindingRef;
use polytype::Type;
use std::collections::HashMap;

use super::from_tlc::ConvertError;
use super::ir::{PlaceId, RemapBlockIds};
use super::program::{
    AllocatedEntry, AllocatedProgramData, CompilerResource, CompilerResourceKind, Entry,
    LogicalResourceArena, LogicalSize, Program, ResidencyProgramData, ResourceId, ResourceOrigin,
    ResourceProgramData, RewriteGlobal, SemanticResourceDecl, SemanticResourceRef, StageOrigin,
};
use super::semantic_opt::Optimized;
use super::soac::{filter, hist, screma};
use super::types::{Semantic, SideEffectKind, Soac, SoacEffect, ValueId};

/// EGIR after logical resources and materialization entries have been planned.
#[derive(Debug, Clone, Copy)]
pub enum ResourcesAllocatedTag {}
pub type ResourcesAllocated = super::program::Program<
    ResourcesAllocatedTag,
    super::ir::ProgramFamily<
        Semantic<SemanticResourceRef>,
        super::program::SemanticResourceDecl,
        super::ir::RealizedOutputRoute,
        AllocatedProgramData,
    >,
    RewriteGlobal,
>;

pub(crate) enum ResidencyDraftTag {}
pub(crate) type ResidencyDraft = super::program::Program<
    ResidencyDraftTag,
    super::ir::ProgramFamily<
        Semantic<SemanticResourceRef>,
        super::program::SemanticResourceDecl,
        super::ir::RealizedOutputRoute,
        ResidencyProgramData,
    >,
    RewriteGlobal,
>;

impl ResourcesAllocated {
    /// Human-readable semantic IR including segmented spaces, captures,
    /// output routing, and logical resource accesses.
    pub fn semantic_ir(&self) -> String {
        let mut output = String::new();
        for (_, stage) in self.data.stages.stages() {
            let entry = stage.body();
            super::semantic_graph::write_graph_summary(
                &mut output,
                &format!("entry {}", entry.name),
                &entry.graph,
            );
        }
        for function in &self.functions {
            super::semantic_graph::write_graph_summary(
                &mut output,
                &format!("function {}", function.name),
                &function.graph,
            );
        }
        output
    }

    /// Target-independent logical resources known before recipe selection.
    pub fn logical_resources(&self) -> &[super::program::LogicalResource] {
        &self.data.core.resources
    }
}

/// Establish target-independent residency and logical resources.
pub fn plan_logical_resources(program: Optimized) -> Result<ResourcesAllocated, ConvertError> {
    let program = allocate_semantic_resources(program)?;
    let program = residency::resolve_residency(program)?;
    let program = resolve_scratch_sizes(program);
    let program = finalize_staged_ir(program)?;
    if cfg!(debug_assertions) {
        verify_allocated_resources(&program).expect("invalid allocated semantic resources");
    }
    Ok(program)
}

/// Replace pre-allocation descriptor bindings with target-independent logical
/// resource identities. This is intentionally the first pass allowed to own
/// or create resources.
fn allocate_semantic_resources(program: Optimized) -> Result<ResidencyDraft, ConvertError> {
    let mut builder = ResourceAllocationBuilder::default();
    reserve_host_resources(&program, &mut builder)?;
    lower_host_size_policies(&program, &mut builder)?;
    let context = builder.finalize()?;
    let program = remap_program_resources(program, context)?;
    Ok(realize_dynamic_publication(program))
}

struct ResourceAllocationContext {
    resources: LogicalResourceArena,
}

impl ResourceAllocationContext {
    fn resource_for_binding(&self, binding: BindingRef) -> Result<SemanticResourceRef, ConvertError> {
        self.resources.host_resource(binding).map(SemanticResourceRef).ok_or_else(|| {
            ConvertError::GraphError(format!(
                "resource binding set={} binding={} is not declared by an entry interface",
                binding.set, binding.binding
            ))
        })
    }
}

struct DraftLogicalResource {
    binding: BindingRef,
    elem_ty: Type<TypeName>,
    /// `None` means policy lowering has not visited this resource. Once
    /// visited, `Some(Unspecified)` records a deliberate external/deferred
    /// sizing decision rather than an unfinished reservation.
    size: Option<LogicalSize>,
}

#[derive(Default)]
struct ResourceAllocationBuilder {
    resources: Vec<DraftLogicalResource>,
    host: HashMap<BindingRef, ResourceId>,
}

impl ResourceAllocationBuilder {
    fn reserve_host(&mut self, binding: BindingRef, elem_ty: Type<TypeName>) -> Result<ResourceId, String> {
        if let Some(id) = self.host.get(&binding).copied() {
            let resource = &self.resources[id.index()];
            if resource.elem_ty != elem_ty {
                return Err(format!(
                    "host resource set={} binding={} has conflicting element types: {:?} and {:?}",
                    binding.set, binding.binding, resource.elem_ty, elem_ty
                ));
            }
            return Ok(id);
        }
        let id = ResourceId::for_allocation(self.resources.len());
        self.host.insert(binding, id);
        self.resources.push(DraftLogicalResource {
            binding,
            elem_ty,
            size: None,
        });
        Ok(id)
    }

    fn resource_for_binding(&self, binding: BindingRef) -> Result<SemanticResourceRef, ConvertError> {
        self.host.get(&binding).copied().map(SemanticResourceRef).ok_or_else(|| {
            ConvertError::GraphError(format!(
                "resource binding set={} binding={} is not declared by an entry interface",
                binding.set, binding.binding
            ))
        })
    }

    fn logical_size(&self, length: Option<&BufferLen>) -> Result<LogicalSize, ConvertError> {
        Ok(match length {
            Some(BufferLen::Fixed { bytes }) => LogicalSize::FixedBytes(*bytes),
            Some(BufferLen::LikeInput {
                set,
                binding,
                elem_bytes,
                src_elem_bytes,
            }) => LogicalSize::LikeResource {
                resource: self.resource_for_binding(BindingRef::new(*set, *binding))?.0,
                elem_bytes: *elem_bytes,
                src_elem_bytes: *src_elem_bytes,
            },
            Some(BufferLen::SameAsDispatch { elem_bytes }) => LogicalSize::SameAsDispatch {
                elem_bytes: *elem_bytes,
            },
            None => LogicalSize::Unspecified,
        })
    }

    fn set_host_size(&mut self, binding: BindingRef, size: LogicalSize) -> Result<(), String> {
        let id = self.host.get(&binding).copied().ok_or_else(|| {
            format!(
                "host resource set={} binding={} must be reserved before its size is set",
                binding.set, binding.binding
            )
        })?;
        let resource = &mut self.resources[id.index()];
        match (resource.size.as_ref(), &size) {
            (None, _) => {
                resource.size = Some(size);
                Ok(())
            }
            (Some(LogicalSize::Unspecified), LogicalSize::Unspecified)
            | (Some(_), LogicalSize::Unspecified) => Ok(()),
            (Some(LogicalSize::Unspecified), _) => {
                resource.size = Some(size);
                Ok(())
            }
            (Some(current), proposed) if current == proposed => Ok(()),
            (Some(current), proposed) => Err(format!(
                "host resource set={} binding={} has conflicting size policies: {:?} and {:?}",
                binding.set, binding.binding, current, proposed
            )),
        }
    }

    fn finalize(self) -> Result<ResourceAllocationContext, ConvertError> {
        if let Some(resource) = self.resources.iter().find(|resource| resource.size.is_none()) {
            return Err(ConvertError::GraphError(format!(
                "host resource set={} binding={} was reserved but its size policy was not processed",
                resource.binding.set, resource.binding.binding
            )));
        }
        let mut resources = LogicalResourceArena::default();
        for draft in self.resources {
            let expected = self.host[&draft.binding];
            let allocated = resources.allocate(
                ResourceOrigin::host(draft.binding),
                draft.elem_ty,
                draft.size.expect("all draft sizes were checked above"),
            );
            if allocated != expected {
                return Err(ConvertError::GraphError(format!(
                    "logical resource reservation for set={} binding={} changed identity during finalization",
                    draft.binding.set, draft.binding.binding
                )));
            }
        }
        Ok(ResourceAllocationContext { resources })
    }
}

fn remap_program_resources(
    program: Optimized,
    context: ResourceAllocationContext,
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
    let mut stage_ids = HashMap::new();
    for entry in &entry_points {
        let stage = stages
            .add_stage(StageOrigin::Authored, entry.id)
            .map_err(|error| ConvertError::Internal(error.to_string()))?;
        stage_ids.insert(entry.id, stage);
    }

    Ok(Program::from_parts(
        functions,
        externs,
        entry_points,
        constants,
        ResidencyProgramData {
            core: ResourceProgramData {
                pipeline: data.pipeline,
                stage_entries: data.stage_entries,
                resources: context.resources,
                identities: data.identities,
            },
            stages,
            stage_ids,
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
            let size = builder.logical_size(resource.length)?;
            builder.set_host_size(resource.binding, size).map_err(ConvertError::GraphError)?;
        }
    }
    Ok(())
}

fn entry_resource_declarations(
    entry: &Entry<Semantic>,
    context: &ResourceAllocationContext,
) -> Result<Vec<SemanticResourceDecl>, ConvertError> {
    let mut positions: HashMap<SemanticResourceRef, usize> = HashMap::new();
    let mut declarations: Vec<SemanticResourceDecl> = Vec::new();
    for item in interface_resources(entry) {
        let Some(role) = item.role else { continue };
        let resource = context.resource_for_binding(item.binding)?;
        if let Some(position) = positions.get(&resource).copied() {
            declarations[position].role = declarations[position].role.merge(role);
            continue;
        }
        positions.insert(resource, declarations.len());
        declarations.push(SemanticResourceDecl { resource, role });
    }
    Ok(declarations)
}

fn allocate_type_resources(
    ty: &mut Type<TypeName>,
    context: &ResourceAllocationContext,
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
    context: &ResourceAllocationContext,
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
                screma::SemanticState::Segmented {
                    space,
                    output_slots,
                    resources: accesses,
                } => {
                    let segment = remap.segment(screma::Segmented {
                        space,
                        output_slots,
                        resources: accesses,
                    })?;
                    screma::SemanticState::Segmented {
                        space: segment.space,
                        output_slots: segment.output_slots,
                        resources: segment.resources,
                    }
                }
            },
        }),
        Soac::Filter(filter::Op { body, state }) => {
            let segment = remap.segment(screma::Segmented {
                space: state.space,
                output_slots: state.output_slots,
                resources: state.resources,
            })?;
            Soac::Filter(filter::Op {
                body: remap.filter_body(body),
                state: filter::SemanticState {
                    space: segment.space,
                    output: remap.filter_output(state.output)?,
                    output_slots: segment.output_slots,
                    resources: segment.resources,
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
    context: &ResourceAllocationContext,
) -> Result<
    (
        super::types::EGraph<Semantic<SemanticResourceRef>>,
        crate::LookupMap<ValueId, ValueId>,
        crate::LookupMap<crate::flow::BlockId, crate::flow::BlockId>,
    ),
    ConvertError,
> {
    let (mut graph, nodes, blocks) = graph.try_map_resources_and_phase(
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
    Ok((graph, nodes, blocks))
}

fn realize_dynamic_publication(mut program: ResidencyDraft) -> ResidencyDraft {
    let resources = &mut program.data.core.resources;
    for function in &mut program.functions {
        realize_graph_dynamic_publication(&mut function.graph, resources);
    }
    for constant in &mut program.constants {
        realize_graph_dynamic_publication(&mut constant.graph, resources);
    }
    for entry in &mut program.entry_points {
        realize_graph_dynamic_publication(&mut entry.graph, resources);
    }
    program
}

fn allocate_filter_storage(
    resources: &mut LogicalResourceArena,
    owner: super::program::SemanticOpId,
    elem_ty: Type<TypeName>,
    size: LogicalSize,
    backing: Option<ResourceId>,
    length: Option<ResourceId>,
) -> filter::RuntimeStorage<ResourceId> {
    let data = backing.unwrap_or_else(|| {
        resources.allocate(
            ResourceOrigin::Compiler(CompilerResource::new(
                CompilerResourceKind::FilterScratch,
                Some(owner),
                0,
            )),
            elem_ty,
            size,
        )
    });
    let length = length.unwrap_or_else(|| {
        resources.allocate(
            ResourceOrigin::Compiler(CompilerResource::new(
                CompilerResourceKind::FilterLenCell,
                Some(owner),
                1,
            )),
            Type::Constructed(TypeName::UInt(32), Vec::new()),
            LogicalSize::FixedBytes(4),
        )
    });
    filter::RuntimeStorage { data, length }
}

fn realize_graph_dynamic_publication(
    graph: &mut super::types::EGraph<Semantic<SemanticResourceRef>>,
    resources: &mut LogicalResourceArena,
) {
    for (_, block) in &mut graph.skeleton.blocks {
        for effect in &mut block.side_effects {
            let SideEffectKind::Soac(SoacEffect(
                owner,
                Soac::Filter(filter::Op {
                    body,
                    state:
                        filter::SemanticState {
                            output: filter::Output::Runtime(runtime),
                            output_slots,
                            resources: accesses,
                            ..
                        },
                    ..
                }),
            )) = &mut effect.kind
            else {
                continue;
            };
            if output_slots.is_empty() {
                continue;
            }

            if matches!(runtime.backing, filter::RuntimeBacking::Deferred) {
                runtime.backing = accesses
                    .iter()
                    .find(|access| access.access != crate::ResourceAccess::Read)
                    .map(|access| filter::RuntimeBacking::Bound(access.resource))
                    .unwrap_or(filter::RuntimeBacking::Deferred);
            }
            let filter::RuntimeBacking::Bound(backing) = runtime.backing else {
                continue;
            };
            let length = match runtime.length {
                filter::RuntimeLength::Implicit => None,
                filter::RuntimeLength::Stored(length) => Some(length.0),
            };
            let storage = allocate_filter_storage(
                resources,
                *owner,
                body.output_element_type(),
                resources[backing.0].size.clone(),
                Some(backing.0),
                length,
            );
            runtime.backing = filter::RuntimeBacking::Bound(SemanticResourceRef(storage.data));
            runtime.length = filter::RuntimeLength::Stored(SemanticResourceRef(storage.length));
            if !accesses.iter().any(|access| access.resource.0 == storage.length) {
                accesses.push(super::types::SegResourceAccess {
                    resource: SemanticResourceRef(storage.length),
                    access: crate::ResourceAccess::Write,
                });
                accesses.sort_by_key(|access| access.resource);
            }
        }
    }
    realize_filter_result_types(graph);
}

fn realize_filter_result_types(graph: &mut super::types::EGraph<Semantic<SemanticResourceRef>>) {
    let results = graph
        .skeleton
        .blocks
        .values()
        .flat_map(|block| &block.side_effects)
        .filter_map(|effect| {
            let SideEffectKind::Soac(SoacEffect(
                _,
                Soac::Filter(filter::Op {
                    body,
                    state:
                        filter::SemanticState {
                            output:
                                filter::Output::Runtime(filter::RuntimeOutput {
                                    backing: filter::RuntimeBacking::Bound(backing),
                                    ..
                                }),
                            ..
                        },
                }),
            )) = &effect.kind
            else {
                return None;
            };
            Some((
                effect.result.as_ref()?.single_value()?,
                crate::types::view_array_of(
                    &body.output_element_type(),
                    Type::Constructed(TypeName::Resource(backing.0), Vec::new()),
                ),
            ))
        })
        .collect::<Vec<_>>();
    for (result, ty) in results {
        graph.retype_node(result, ty);
    }
}

fn remap_function_resources(
    function: super::program::Func<Semantic>,
    context: &ResourceAllocationContext,
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
    } = function;
    let (graph, _, _) = remap_graph_resources(graph, context)?;
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
    Ok(super::program::Func {
        region,
        name,
        span,
        linkage_name,
        params,
        result,
        effects,
        graph,
    })
}

fn remap_constant_resources(
    constant: super::program::ConstantDef<Semantic>,
    context: &ResourceAllocationContext,
) -> Result<super::program::AllocatedConstantDef, ConvertError> {
    let super::program::ConstantDef {
        id,
        name,
        span,
        mut return_ty,
        graph,
    } = constant;
    let (graph, _, _) = remap_graph_resources(graph, context)?;
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
    context: &ResourceAllocationContext,
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
    let (graph, _, blocks) = remap_graph_resources(graph, context)?;
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

pub(crate) fn verify_allocated_resources(program: &ResourcesAllocated) -> Result<(), String> {
    let check_size = |size: &LogicalSize| match size {
        LogicalSize::LikeResource { resource, .. } if !program.data.core.resources.contains(*resource) => {
            Err(format!("resource size references missing source {resource:?}"))
        }
        _ => Ok(()),
    };
    for resource in &program.data.core.resources {
        check_size(&resource.size)?;
    }
    for (_, stage) in program.data.stages.stages() {
        let entry = stage.body();
        for declaration in &entry.resource_declarations {
            if !program.data.core.resources.contains(declaration.resource.0) {
                return Err(format!(
                    "entry references missing resource {:?}",
                    declaration.resource.0
                ));
            }
        }
        for (slot, output) in entry.outputs.iter().enumerate() {
            if output.routes.is_empty() {
                return Err(format!(
                    "entry `{}` output slot {slot} has no explicit route",
                    entry.name
                ));
            }
            if output.routes.iter().any(|route| route.writers.is_empty()) {
                return Err(format!(
                    "entry `{}` output slot {slot} has a source value but no producer",
                    entry.name
                ));
            }
        }
    }
    Ok(())
}

fn resolve_scratch_sizes(program: ResidencyDraft) -> ResidencyDraft {
    let mut resolved = Vec::new();
    for entry in &program.entry_points {
        for (_, block) in &entry.graph.skeleton.blocks {
            for effect in &block.side_effects {
                let SideEffectKind::Soac(SoacEffect(
                    _,
                    Soac::Filter(filter::Op {
                        body,
                        state:
                            filter::SemanticState {
                                space,
                                output:
                                    filter::Output::Runtime(filter::RuntimeOutput {
                                        backing: filter::RuntimeBacking::Bound(scratch),
                                        ..
                                    }),
                                ..
                            },
                    }),
                )) = &effect.kind
                else {
                    continue;
                };
                let elem_bytes = ssa::layout::storage_elem_stride(&body.output_element_type()).unwrap_or(1);
                let size = match space.dims() {
                    [super::types::SegExtent::Fixed(count)] => {
                        LogicalSize::FixedBytes(*count as u64 * elem_bytes as u64)
                    }
                    [super::types::SegExtent::ResourceLength {
                        resource,
                        elem_bytes: src_elem_bytes,
                        ..
                    }] => LogicalSize::LikeResource {
                        resource: resource.0,
                        elem_bytes,
                        src_elem_bytes: *src_elem_bytes,
                    },
                    _ => LogicalSize::SameAsDispatch { elem_bytes },
                };
                let output_len = match &size {
                    LogicalSize::FixedBytes(bytes) => {
                        Some(pipeline_descriptor::BufferLen::Fixed { bytes: *bytes })
                    }
                    LogicalSize::LikeResource {
                        resource,
                        elem_bytes,
                        src_elem_bytes,
                    } => program.data.core.resources[*resource].host_binding().map(|binding| {
                        pipeline_descriptor::BufferLen::LikeInput {
                            set: binding.set,
                            binding: binding.binding,
                            elem_bytes: *elem_bytes,
                            src_elem_bytes: *src_elem_bytes,
                        }
                    }),
                    LogicalSize::SameAsDispatch { elem_bytes } => {
                        Some(pipeline_descriptor::BufferLen::SameAsDispatch {
                            elem_bytes: *elem_bytes,
                        })
                    }
                    LogicalSize::Unspecified => None,
                };
                resolved.push((scratch.0, size, output_len));
            }
        }
    }

    let Program {
        functions,
        externs,
        mut entry_points,
        constants,
        mut data,
        global_context,
        state: _,
    } = program;
    for (resource, size, output_len) in resolved {
        data.core.resources[resource].size = size.clone();
        for entry in &mut entry_points {
            for output in &mut entry.outputs {
                if output.resource == Some(SemanticResourceRef(resource)) {
                    *output.storage_length_mut().expect("filter output resource must be storage") =
                        output_len.clone();
                }
            }
        }
    }
    Program::from_parts(functions, externs, entry_points, constants, data, global_context)
}

fn finalize_staged_ir(program: ResidencyDraft) -> Result<ResourcesAllocated, ConvertError> {
    let Program {
        functions,
        externs,
        entry_points,
        constants,
        mut data,
        global_context,
        state: _,
    } = program;
    for entry in &entry_points {
        let stage = data.stage_ids[&entry.id];
        for declaration in entry.resource_declarations.iter().filter(|declaration| declaration.role.reads())
        {
            let Some(flow) = data.resident_flows.get(&declaration.resource.0).copied() else {
                continue;
            };
            let needs_edge = data.stages.flow(flow).is_some_and(|resident| {
                resident.producer() != stage && !resident.consumers().contains(&stage)
            });
            if needs_edge {
                data.stages
                    .add_consumer(flow, stage)
                    .map_err(|error| ConvertError::Internal(error.to_string()))?;
            }
        }
    }
    for entry in &entry_points {
        let stage = data.stage_ids[&entry.id];
        for input in &entry.inputs {
            let Some(SemanticResourceRef(resource)) = input.resource else {
                continue;
            };
            if !matches!(&data.core.resources[resource].origin, ResourceOrigin::Host(_)) {
                continue;
            }
            data.stages
                .add_external_input(
                    input.ty.clone(),
                    super::program::ResidentStorage {
                        data: resource,
                        length: None,
                    },
                    [stage],
                )
                .map_err(|error| ConvertError::Internal(error.to_string()))?;
        }
        for output in &entry.outputs {
            let Some(SemanticResourceRef(resource)) = output.resource else {
                continue;
            };
            if !matches!(&data.core.resources[resource].origin, ResourceOrigin::Host(_)) {
                continue;
            }
            let flow = data
                .stages
                .add_flow(
                    stage,
                    output.ty.clone(),
                    super::program::ResidentStorage {
                        data: resource,
                        length: None,
                    },
                )
                .map_err(|error| ConvertError::Internal(error.to_string()))?;
            data.stages.publish(flow).map_err(|error| ConvertError::Internal(error.to_string()))?;
        }
    }

    let mut entries = entry_points.into_iter().map(|entry| (entry.id, entry)).collect::<HashMap<_, _>>();
    let stages = data
        .stages
        .finish()
        .map_err(|error| ConvertError::Internal(error.to_string()))?
        .map_stage_bodies(|_, entry| {
            entries.remove(&entry).expect("staged lowering retained every executable entry exactly once")
        });
    assert!(
        entries.is_empty(),
        "every executable entry must be owned by one staged body"
    );
    Ok(Program::from_parts(
        functions,
        externs,
        Vec::new(),
        constants,
        AllocatedProgramData {
            core: data.core,
            stages,
        },
        global_context,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_ty() -> Type<TypeName> {
        Type::Constructed(TypeName::Unit, Vec::new())
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
        builder.set_host_size(binding, LogicalSize::Unspecified).unwrap();

        let context = builder.finalize().expect("processed draft must finalize");
        assert_eq!(context.resources[resource].size, LogicalSize::Unspecified);
    }

    #[test]
    fn resource_draft_rejects_policy_for_unreserved_binding() {
        let binding = BindingRef::new(2, 4);
        let mut builder = ResourceAllocationBuilder::default();

        let error = builder
            .set_host_size(binding, LogicalSize::FixedBytes(16))
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
}
