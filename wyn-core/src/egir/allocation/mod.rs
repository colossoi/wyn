//! Target-independent logical resource allocation.
//!
//! This pass resolves legal in-place destinations, materializes values that
//! must survive a scheduling boundary, assigns logical sizes, and removes
//! compiler-only storage from the host ABI. Physical descriptor selection
//! remains the responsibility of target planning.

mod cost;
mod destinations;
mod residency;

use std::collections::{HashMap, HashSet};

use super::from_tlc::ConvertError;
use super::ir::Stage;
use super::program::{
    AllocatedProgramData, CompilerResource, CompilerResourceKind, LogicalSize, MaterializationId, Program,
    ResourceId, ResourceOrigin, RewriteGlobal, SemanticEntry, SemanticEntryId, SemanticResourceRef,
};
use super::semantic_opt::Optimized;
use super::soac::filter;
use super::types::{Semantic, SideEffectKind, Soac, SoacEffect, WynLanguage};

/// EGIR after logical resources and materialization entries have been planned.
#[derive(Debug, Clone, Copy, Default)]
pub struct ResourcesAllocated;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CompilerFlowEndpoint {
    Entry(SemanticEntryId),
    Materialization(MaterializationId),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompilerResourceFlow {
    pub producer: CompilerFlowEndpoint,
    pub consumers: Vec<CompilerFlowEndpoint>,
}

impl Stage for ResourcesAllocated {
    type Family = Semantic;
    type ResourceDecl = super::program::SemanticResourceDecl;
    type OutputRoute = super::ir::RealizedOutputRoute;
    type ProgramData = AllocatedProgramData;
    type GlobalContext = RewriteGlobal;
}

impl Program<ResourcesAllocated, WynLanguage> {
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
pub fn plan_logical_resources(
    program: Program<Optimized, WynLanguage>,
) -> Result<Program<ResourcesAllocated, WynLanguage>, ConvertError> {
    let Program {
        functions,
        externs,
        entry_points,
        constants,
        data,
        global_context,
    } = program;
    let program = Program::from_parts(
        functions,
        externs,
        entry_points,
        constants,
        AllocatedProgramData {
            core: data,
            materializations: crate::IdArena::new(),
        },
        global_context,
    );

    let program = classify_existing_compiler_resources(program);
    let program = destinations::run(program);
    let program = residency::run(program)?;
    let program = resolve_scratch_sizes(program);
    let program = strip_compiler_abi(program);
    if cfg!(debug_assertions) {
        verify_allocated_resources(&program).expect("invalid allocated semantic resources");
    }
    Ok(program)
}

pub(crate) fn entries_with_endpoints(
    program: &Program<ResourcesAllocated, WynLanguage>,
) -> impl Iterator<Item = (CompilerFlowEndpoint, &SemanticEntry)> {
    program
        .entry_points
        .iter()
        .enumerate()
        .map(|(index, entry)| {
            (
                CompilerFlowEndpoint::Entry(SemanticEntryId::from_index(index)),
                entry,
            )
        })
        .chain(program.data.materializations.ids().map(|id| {
            (
                CompilerFlowEndpoint::Materialization(id),
                program.data.materializations[id].entry(),
            )
        }))
}

/// Derived resource-flow edges consumed by target scheduling. They are not
/// stored on resources because entry rewrites are their source of truth.
pub(crate) fn resource_flows(
    program: &Program<ResourcesAllocated, WynLanguage>,
) -> Vec<(ResourceId, CompilerResourceFlow)> {
    let mut producers: HashMap<ResourceId, Vec<CompilerFlowEndpoint>> = HashMap::new();
    let mut consumers: HashMap<ResourceId, Vec<CompilerFlowEndpoint>> = HashMap::new();
    for (endpoint, entry) in entries_with_endpoints(program) {
        for declaration in &entry.resource_declarations {
            let resource = declaration.resource.0;
            match &declaration.role {
                crate::interface::StorageRole::Output => {
                    producers.entry(resource).or_default().push(endpoint);
                }
                crate::interface::StorageRole::Input => {
                    consumers.entry(resource).or_default().push(endpoint);
                }
                crate::interface::StorageRole::Intermediate => {}
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

pub(crate) fn verify_allocated_resources(
    program: &Program<ResourcesAllocated, WynLanguage>,
) -> Result<(), String> {
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
    }
    Ok(())
}

fn classify_existing_compiler_resources(
    program: Program<ResourcesAllocated, WynLanguage>,
) -> Program<ResourcesAllocated, WynLanguage> {
    let mut classifications = HashMap::new();
    for entry in &program.entry_points {
        for declaration in &entry.resource_declarations {
            if declaration.role == crate::interface::StorageRole::Intermediate {
                classifications
                    .entry(declaration.resource.0)
                    .or_insert_with(|| CompilerResource::new(CompilerResourceKind::Staging, None, 0));
            }
        }
    }
    let source_outputs = program
        .entry_points
        .iter()
        .flat_map(|entry| {
            entry.outputs.iter().filter_map(|output| output.resource.map(|resource| resource.0))
        })
        .collect::<HashSet<_>>();
    classifications.extend(
        filter_resource_kinds(&program)
            .into_iter()
            .filter(|(resource, _)| !source_outputs.contains(resource)),
    );

    program.map_data(|mut data| {
        for (resource, compiler) in classifications {
            data.core.resources.reclassify_as_compiler(resource, compiler);
        }
        data
    })
}

fn filter_resource_kinds(
    program: &Program<ResourcesAllocated, WynLanguage>,
) -> HashMap<ResourceId, CompilerResource> {
    let mut kinds = HashMap::new();
    for (_, entry) in entries_with_endpoints(program) {
        for (_, block) in &entry.graph.skeleton.blocks {
            for effect in &block.side_effects {
                let SideEffectKind::Soac(SoacEffect(
                    owner,
                    Soac::Filter(filter::Op {
                        state:
                            filter::SemanticState {
                                storage: filter::Output::Runtime { scratch, length },
                                ..
                            },
                        ..
                    }),
                )) = &effect.kind
                else {
                    continue;
                };
                let owner = Some(*owner);
                kinds.insert(
                    scratch.0,
                    CompilerResource::new(CompilerResourceKind::FilterScratch, owner, 0),
                );
                if let filter::RuntimeLength::Stored(length) = length {
                    kinds.insert(
                        length.0,
                        CompilerResource::new(CompilerResourceKind::FilterLenCell, owner, 1),
                    );
                }
            }
        }
    }
    kinds
}

fn resolve_scratch_sizes(
    program: Program<ResourcesAllocated, WynLanguage>,
) -> Program<ResourcesAllocated, WynLanguage> {
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
                                storage: filter::Output::Runtime { scratch, .. },
                            },
                    }),
                )) = &effect.kind
                else {
                    continue;
                };
                let elem_bytes =
                    crate::ssa::layout::storage_elem_stride(&body.output_element_type()).unwrap_or(1);
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
                        Some(crate::pipeline_descriptor::BufferLen::Fixed { bytes: *bytes })
                    }
                    LogicalSize::LikeResource {
                        resource,
                        elem_bytes,
                        src_elem_bytes,
                    } => program.data.core.resources[*resource].host_binding().map(|binding| {
                        crate::pipeline_descriptor::BufferLen::LikeInput {
                            set: binding.set,
                            binding: binding.binding,
                            elem_bytes: *elem_bytes,
                            src_elem_bytes: *src_elem_bytes,
                        }
                    }),
                    LogicalSize::SameAsDispatch { elem_bytes } => {
                        Some(crate::pipeline_descriptor::BufferLen::SameAsDispatch {
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

fn strip_compiler_abi(
    program: Program<ResourcesAllocated, WynLanguage>,
) -> Program<ResourcesAllocated, WynLanguage> {
    let compiler_resources = program
        .data
        .core
        .resources
        .iter()
        .filter_map(|resource| {
            matches!(&resource.origin, ResourceOrigin::Compiler(_)).then_some(resource.id())
        })
        .collect::<HashSet<_>>();
    let strip = |entry: &mut SemanticEntry| {
        for input in &mut entry.inputs {
            if input.resource.is_some_and(|resource| compiler_resources.contains(&resource.0)) {
                input.make_storage_internal();
            }
        }
        entry.outputs.retain(|output| {
            !output.resource.is_some_and(|resource| compiler_resources.contains(&resource.0))
        });
    };

    let Program {
        functions,
        externs,
        mut entry_points,
        constants,
        mut data,
        global_context,
    } = program;
    for entry in &mut entry_points {
        strip(entry);
    }
    for (_, requirement) in &mut data.materializations {
        strip(requirement.entry_mut());
    }
    Program::from_parts(functions, externs, entry_points, constants, data, global_context)
}
