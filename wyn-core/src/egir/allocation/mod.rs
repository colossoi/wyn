//! Target-independent logical resource allocation.
//!
//! This pass resolves legal in-place destinations, materializes values that
//! must survive a scheduling boundary, assigns logical sizes, and removes
//! compiler-only storage from the host ABI. Physical descriptor selection
//! remains the responsibility of target planning.

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
use crate::IdArena;
use polytype::Type;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

use super::from_tlc::ConvertError;
use super::ir::{PlaceId, RemapBlockIds};
use super::program::{
    AllocatedEntry, AllocatedProgramData, CompilerResource, CompilerResourceKind, Entry,
    LogicalResourceArena, LogicalSize, MaterializationId, Program, ResourceId, ResourceOrigin,
    ResourceProgramData, RewriteGlobal, SemanticResourceDecl, SemanticResourceRef,
};
use super::semantic_opt::Optimized;
use super::soac::{filter, hist, screma};
use super::types::{Semantic, SideEffectKind, Soac, SoacEffect, ValueId};
use crate::EntryId;

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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CompilerFlowEndpoint {
    Entry(EntryId),
    Materialization(MaterializationId),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompilerResourceFlow {
    pub producer: CompilerFlowEndpoint,
    pub consumers: Vec<CompilerFlowEndpoint>,
}

impl ResourcesAllocated {
    /// Human-readable semantic IR including segmented spaces, captures,
    /// output routing, and logical resource accesses.
    pub fn semantic_ir(&self) -> String {
        super::semantic_graph::summary(self)
    }

    /// Target-independent logical resources known before recipe selection.
    pub fn logical_resources(&self) -> &[super::program::LogicalResource] {
        &self.data.core.resources
    }
}

/// Establish target-independent residency and logical resources.
pub fn plan_logical_resources(program: Optimized) -> Result<ResourcesAllocated, ConvertError> {
    let program = allocate_semantic_resources(program)?;
    let program = classify_existing_compiler_resources(program);
    let program = residency::resolve_residency(program)?;
    let program = resolve_scratch_sizes(program);
    let program = strip_compiler_abi(program);
    if cfg!(debug_assertions) {
        verify_allocated_resources(&program).expect("invalid allocated semantic resources");
    }
    Ok(program)
}

/// Replace pre-allocation descriptor bindings with target-independent logical
/// resource identities. This is intentionally the first pass allowed to own
/// or create resources.
fn allocate_semantic_resources(program: Optimized) -> Result<ResourcesAllocated, ConvertError> {
    let Program {
        functions,
        externs,
        entry_points,
        constants,
        data,
        global_context,
        state: _,
    } = program;
    let resources = RefCell::new(LogicalResourceArena::default());

    // Reserve every authored interface resource first so cross-entry
    // `LikeInput` sizes can only refer to declared bindings.
    for entry in &entry_points {
        reserve_entry_resources(entry, &resources);
    }
    for entry in &entry_points {
        refine_entry_resources(entry, &resources)?;
    }

    let functions = functions
        .into_iter()
        .map(|function| allocate_function(function, &resources))
        .collect::<Result<Vec<_>, _>>()?;
    let constants = constants
        .into_iter()
        .map(|constant| allocate_constant(constant, &resources))
        .collect::<Result<Vec<_>, _>>()?;
    let entry_points = entry_points
        .into_iter()
        .map(|entry| allocate_entry(entry, &resources))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(Program::from_parts(
        functions,
        externs,
        entry_points,
        constants,
        AllocatedProgramData {
            core: ResourceProgramData {
                pipeline: data.pipeline,
                stage_entries: data.stage_entries,
                resources: resources.into_inner(),
                identities: data.identities,
            },
            materializations: IdArena::new(),
        },
        global_context,
    ))
}

fn resource_for_binding(
    resources: &RefCell<LogicalResourceArena>,
    binding: BindingRef,
) -> Result<SemanticResourceRef, ConvertError> {
    resources.borrow().host_resource(binding).map(SemanticResourceRef).ok_or_else(|| {
        ConvertError::GraphError(format!(
            "resource binding set={} binding={} is not declared by an entry interface",
            binding.set, binding.binding
        ))
    })
}

fn logical_size(
    resources: &RefCell<LogicalResourceArena>,
    length: Option<&BufferLen>,
) -> Result<LogicalSize, ConvertError> {
    Ok(match length {
        Some(BufferLen::Fixed { bytes }) => LogicalSize::FixedBytes(*bytes),
        Some(BufferLen::LikeInput {
            set,
            binding,
            elem_bytes,
            src_elem_bytes,
        }) => LogicalSize::LikeResource {
            resource: resource_for_binding(resources, BindingRef::new(*set, *binding))?.0,
            elem_bytes: *elem_bytes,
            src_elem_bytes: *src_elem_bytes,
        },
        Some(BufferLen::SameAsDispatch { elem_bytes }) => LogicalSize::SameAsDispatch {
            elem_bytes: *elem_bytes,
        },
        None => LogicalSize::Unspecified,
    })
}

struct InterfaceResource {
    binding: BindingRef,
    role: Option<interface::StorageRole>,
    elem_ty: Type<TypeName>,
    length: Option<BufferLen>,
}

fn interface_resources(entry: &Entry<Semantic>) -> Vec<InterfaceResource> {
    let inputs = entry.inputs.iter().filter_map(|input| {
        let binding = input.resource?;
        let (role, length) = match &input.kind {
            EntryInputKind::Storage { length, .. } => (Some(interface::StorageRole::Input), length.clone()),
            _ => (None, None),
        };
        Some(InterfaceResource {
            binding,
            role,
            elem_ty: input.ty.elem_type().cloned().unwrap_or_else(|| input.ty.clone()),
            length,
        })
    });
    let outputs = entry.outputs.iter().filter_map(|output| {
        let binding = output.resource?;
        let length = match &output.kind {
            EntryOutputKind::Storage { length, .. } => length.clone(),
            _ => None,
        };
        Some(InterfaceResource {
            binding,
            role: Some(interface::StorageRole::Output),
            elem_ty: output.ty.elem_type().cloned().unwrap_or_else(|| output.ty.clone()),
            length,
        })
    });
    inputs.chain(outputs).collect()
}

fn reserve_entry_resources(entry: &Entry<Semantic>, resources: &RefCell<LogicalResourceArena>) {
    for resource in interface_resources(entry) {
        resources.borrow_mut().declare_host(resource.binding, resource.elem_ty, LogicalSize::Unspecified);
    }
}

fn refine_entry_resources(
    entry: &Entry<Semantic>,
    resources: &RefCell<LogicalResourceArena>,
) -> Result<(), ConvertError> {
    for resource in interface_resources(entry) {
        let size = logical_size(resources, resource.length.as_ref())?;
        resources.borrow_mut().declare_host(resource.binding, resource.elem_ty, size);
    }
    Ok(())
}

fn entry_resource_declarations(
    entry: &Entry<Semantic>,
    resources: &RefCell<LogicalResourceArena>,
) -> Result<Vec<SemanticResourceDecl>, ConvertError> {
    let mut seen = HashSet::new();
    let mut declarations = Vec::new();
    for item in interface_resources(entry) {
        let Some(role) = item.role else { continue };
        let resource = resource_for_binding(resources, item.binding)?;
        if seen.insert(resource) {
            declarations.push(SemanticResourceDecl {
                resource,
                role,
                elem_ty: item.elem_ty,
                size: logical_size(resources, item.length.as_ref())?,
            });
        }
    }
    Ok(declarations)
}

fn allocate_type_resources(
    ty: &mut Type<TypeName>,
    resources: &RefCell<LogicalResourceArena>,
) -> Result<(), ConvertError> {
    let mut error = None;
    super::program::visit_type_names_mut(ty, |name| {
        if let TypeName::Buffer(binding) = *name {
            match resource_for_binding(resources, binding) {
                Ok(resource) => *name = TypeName::Resource(resource.0),
                Err(binding_error) => error = Some(binding_error),
            }
        }
    });
    error.map_or(Ok(()), Err)
}

fn allocate_filter_publication(
    owner: super::program::SemanticOpId,
    output: filter::Output<SemanticResourceRef>,
    output_slots: &[super::program::OutputSlotId],
    accesses: &mut Vec<super::types::SegResourceAccess<SemanticResourceRef>>,
    resources: &RefCell<LogicalResourceArena>,
) -> filter::Output<SemanticResourceRef> {
    let filter::Output::Runtime(mut runtime) = output else {
        return output;
    };
    if !output_slots.is_empty() {
        if matches!(runtime.backing, filter::RuntimeBacking::Deferred) {
            runtime.backing = accesses
                .iter()
                .find(|access| access.access != crate::ResourceAccess::Read)
                .map(|access| filter::RuntimeBacking::Bound(access.resource))
                .unwrap_or(filter::RuntimeBacking::Deferred);
        }
        let id = resources.borrow_mut().allocate(
            ResourceOrigin::Compiler(CompilerResource::new(
                CompilerResourceKind::FilterLenCell,
                Some(owner),
                0,
            )),
            Type::Constructed(TypeName::UInt(32), Vec::new()),
            LogicalSize::FixedBytes(4),
        );
        let length = SemanticResourceRef(id);
        runtime.length = filter::RuntimeLength::Stored(length);
        accesses.push(super::types::SegResourceAccess {
            resource: length,
            access: crate::ResourceAccess::Write,
        });
        accesses.sort_by_key(|access| access.resource);
        accesses.dedup_by_key(|access| access.resource);
    }
    filter::Output::Runtime(runtime)
}

fn allocate_soac(
    owner: super::program::SemanticOpId,
    soac: Soac<Semantic>,
    nodes: &crate::LookupMap<ValueId, ValueId>,
    places: &crate::LookupMap<PlaceId, PlaceId>,
    resources: &RefCell<LogicalResourceArena>,
) -> Result<Soac<Semantic<SemanticResourceRef>>, ConvertError> {
    let mut remap =
        super::soac::remap::Remap::new(nodes, places, |binding| resource_for_binding(resources, binding));
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
            let mut accesses = segment.resources;
            let output = allocate_filter_publication(
                owner,
                remap.filter_output(state.output)?,
                &segment.output_slots,
                &mut accesses,
                resources,
            );
            Soac::Filter(filter::Op {
                body: remap.filter_body(body),
                state: filter::SemanticState {
                    space: segment.space,
                    output,
                    output_slots: segment.output_slots,
                    resources: accesses,
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

fn allocate_graph(
    graph: super::types::EGraph<Semantic>,
    resources: &RefCell<LogicalResourceArena>,
) -> Result<
    (
        super::types::EGraph<Semantic<SemanticResourceRef>>,
        crate::LookupMap<ValueId, ValueId>,
        crate::LookupMap<crate::flow::BlockId, crate::flow::BlockId>,
    ),
    ConvertError,
> {
    let (mut graph, nodes, blocks) = graph.try_map_resources_and_phase(
        |binding| resource_for_binding(resources, binding),
        |owner, soac, nodes, places| {
            Ok::<_, ConvertError>((owner, allocate_soac(owner, soac, nodes, places, resources)?))
        },
    )?;
    let mut type_error = None;
    super::program::rewrite_graph_types(&mut graph, |ty| {
        if let Err(error) = allocate_type_resources(ty, resources) {
            type_error = Some(error);
        }
    });
    if let Some(error) = type_error {
        return Err(error);
    }
    realize_filter_result_types(&mut graph);
    Ok((graph, nodes, blocks))
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

fn allocate_function(
    function: super::program::Func<Semantic>,
    resources: &RefCell<LogicalResourceArena>,
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
    let (graph, _, _) = allocate_graph(graph, resources)?;
    let params = params
        .into_iter()
        .map(|param| {
            param.try_map(
                &mut |binding| resource_for_binding(resources, binding),
                &mut |mut ty| {
                    allocate_type_resources(&mut ty, resources)?;
                    Ok(ty)
                },
            )
        })
        .collect::<Result<Vec<_>, ConvertError>>()?;
    let mut result_error = None;
    result.for_each_type_mut(|ty| {
        if let Err(error) = allocate_type_resources(ty, resources) {
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

fn allocate_constant(
    constant: super::program::ConstantDef<Semantic>,
    resources: &RefCell<LogicalResourceArena>,
) -> Result<super::program::AllocatedConstantDef, ConvertError> {
    let super::program::ConstantDef {
        id,
        name,
        span,
        mut return_ty,
        graph,
    } = constant;
    let (graph, _, _) = allocate_graph(graph, resources)?;
    allocate_type_resources(&mut return_ty, resources)?;
    Ok(super::program::ConstantDef {
        id,
        name,
        span,
        return_ty,
        graph,
    })
}

fn allocate_entry(
    entry: Entry<Semantic>,
    resources: &RefCell<LogicalResourceArena>,
) -> Result<AllocatedEntry, ConvertError> {
    let declarations = entry_resource_declarations(&entry, resources)?;
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
    let (graph, _, blocks) = allocate_graph(graph, resources)?;
    let inputs = inputs
        .into_iter()
        .map(|mut input| {
            allocate_type_resources(&mut input.ty, resources)?;
            Ok(super::ir::EntryInput {
                inner: input.inner,
                resource: input
                    .resource
                    .map(|binding| resource_for_binding(resources, binding))
                    .transpose()?,
            })
        })
        .collect::<Result<Vec<_>, ConvertError>>()?;
    let outputs = outputs
        .into_iter()
        .map(|mut output| {
            allocate_type_resources(&mut output.ty, resources)?;
            for route in &mut output.routes {
                route.remap_block_ids(&blocks);
            }
            Ok(super::ir::EntryOutput {
                inner: output.inner,
                resource: output
                    .resource
                    .map(|binding| resource_for_binding(resources, binding))
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
                resource: resource_for_binding(resources, result.resource)?,
                route: result.route,
            })
        })
        .collect::<Result<Vec<_>, ConvertError>>()?;
    let params = params
        .into_iter()
        .map(|param| {
            param.try_map(
                &mut |binding| resource_for_binding(resources, binding),
                &mut |mut ty| {
                    allocate_type_resources(&mut ty, resources)?;
                    Ok(ty)
                },
            )
        })
        .collect::<Result<Vec<_>, ConvertError>>()?;
    let mut result_error = None;
    result.for_each_type_mut(|ty| {
        if let Err(error) = allocate_type_resources(ty, resources) {
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

pub(crate) fn entries_with_endpoints(
    program: &ResourcesAllocated,
) -> impl Iterator<Item = (CompilerFlowEndpoint, &AllocatedEntry)> {
    program.entry_points.iter().map(|entry| (CompilerFlowEndpoint::Entry(entry.id), entry)).chain(
        program.data.materializations.ids().map(|id| {
            (
                CompilerFlowEndpoint::Materialization(id),
                program.data.materializations[id].entry(),
            )
        }),
    )
}

/// Derived resource-flow edges consumed by target scheduling. They are not
/// stored on resources because entry rewrites are their source of truth.
pub(crate) fn resource_flows(program: &ResourcesAllocated) -> Vec<(ResourceId, CompilerResourceFlow)> {
    let mut producers: HashMap<ResourceId, Vec<CompilerFlowEndpoint>> = HashMap::new();
    let mut consumers: HashMap<ResourceId, Vec<CompilerFlowEndpoint>> = HashMap::new();
    for (endpoint, entry) in entries_with_endpoints(program) {
        for declaration in &entry.resource_declarations {
            let resource = declaration.resource.0;
            match &declaration.role {
                interface::StorageRole::Output => {
                    producers.entry(resource).or_default().push(endpoint);
                }
                interface::StorageRole::Input => {
                    consumers.entry(resource).or_default().push(endpoint);
                }
                interface::StorageRole::Intermediate => {}
            }
        }
    }

    let mut flows = Vec::new();
    for resource in &program.data.core.resources {
        let ResourceOrigin::Compiler(compiler) = &resource.origin else {
            continue;
        };
        if !matches!(
            compiler.kind,
            CompilerResourceKind::GatherHandoff
                | CompilerResourceKind::ScalarHandoff
                | CompilerResourceKind::MultiConsumerArray
                | CompilerResourceKind::FilterScratch
                | CompilerResourceKind::FilterLenCell
        ) {
            continue;
        }
        let mut resource_producers = producers.remove(&resource.id()).unwrap_or_default();
        resource_producers.sort_unstable();
        resource_producers.dedup();
        let [producer] = resource_producers.as_slice() else {
            continue;
        };
        let mut resource_consumers = consumers.remove(&resource.id()).unwrap_or_default();
        resource_consumers.retain(|consumer| consumer != producer);
        resource_consumers.sort_unstable();
        resource_consumers.dedup();
        flows.push((
            resource.id(),
            CompilerResourceFlow {
                producer: *producer,
                consumers: resource_consumers,
            },
        ));
    }
    flows
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
    for (_, entry) in entries_with_endpoints(program) {
        for declaration in &entry.resource_declarations {
            if !program.data.core.resources.contains(declaration.resource.0) {
                return Err(format!(
                    "entry references missing resource {:?}",
                    declaration.resource.0
                ));
            }
            check_size(&declaration.size)?;
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

fn classify_existing_compiler_resources(program: ResourcesAllocated) -> ResourcesAllocated {
    let mut classifications = HashMap::new();
    for entry in &program.entry_points {
        for declaration in &entry.resource_declarations {
            if declaration.role == interface::StorageRole::Intermediate {
                classifications
                    .entry(declaration.resource.0)
                    .or_insert_with(|| CompilerResource::new(CompilerResourceKind::Staging, None, 0));
            }
        }
    }
    program.map_data(|mut data| {
        for (resource, compiler) in classifications {
            data.core.resources.reclassify_as_compiler(resource, compiler);
        }
        data
    })
}

fn resolve_scratch_sizes(program: ResourcesAllocated) -> ResourcesAllocated {
    let mut resolved = Vec::new();
    for (_, entry) in entries_with_endpoints(&program) {
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
        let entries = entry_points.iter_mut().chain(
            (&mut data.materializations).into_iter().map(|(_, requirement)| requirement.entry_mut()),
        );
        for entry in entries {
            if let Some(declaration) = entry
                .resource_declarations
                .iter_mut()
                .find(|declaration| declaration.resource.0 == resource)
            {
                declaration.size = size.clone();
            }
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

fn strip_compiler_abi(program: ResourcesAllocated) -> ResourcesAllocated {
    let compiler_resources = program
        .data
        .core
        .resources
        .iter()
        .filter_map(|resource| {
            matches!(&resource.origin, ResourceOrigin::Compiler(_)).then_some(resource.id())
        })
        .collect::<HashSet<_>>();
    let strip = |entry: &mut AllocatedEntry| {
        for input in &mut entry.inputs {
            if input.resource.is_some_and(|resource| compiler_resources.contains(&resource.0)) {
                input.make_storage_internal();
            }
        }
    };

    let Program {
        functions,
        externs,
        mut entry_points,
        constants,
        mut data,
        global_context,
        state: _,
    } = program;
    for entry in &mut entry_points {
        strip(entry);
    }
    for (_, requirement) in &mut data.materializations {
        strip(requirement.entry_mut());
    }
    Program::from_parts(functions, externs, entry_points, constants, data, global_context)
}
