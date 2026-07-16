use std::collections::HashMap;

use polytype::Type;

use crate::ast::TypeName;

use super::super::program::{
    CompilerResource, CompilerResourceKind, LogicalResource, LogicalSize, PhysicalResourceRef, ResourceId,
    SemanticProgram, SemanticResourceRef,
};
use super::super::types::{
    EgirPhase, GraphResource, NodeId, SegBody, SegExtent, SegSpace, Semantic, SideEffectKind, Soac,
    SoacDestination, SoacInputType,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkBuffers<R = SemanticResourceRef> {
    pub flags: R,
    pub offsets: R,
    pub block_sums: R,
    pub block_offsets: R,
}

#[derive(Clone, Debug)]
pub enum Output<R = SemanticResourceRef> {
    Local {
        capacity: Type<TypeName>,
        destination: SoacDestination,
    },
    Runtime {
        scratch: R,
        length: RuntimeLength<R>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeLength<R = SemanticResourceRef> {
    ViewOnly,
    /// Logical length stored in a scalar resource. Public filter outputs and
    /// compiler-internal runtime-array handoffs use the same representation;
    /// publication decides whether the resource belongs to the host ABI.
    Stored(R),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Plan<R = SemanticResourceRef> {
    Serial,
    Flags(WorkBuffers<R>),
    Scan(WorkBuffers<R>),
    Scatter(WorkBuffers<R>),
}

#[derive(Clone, Debug)]
pub enum Input {
    Plain(SoacInputType),
    Mapped {
        input: SoacInputType,
        body: SegBody,
        output_element_type: Type<TypeName>,
    },
}

#[derive(Clone, Debug)]
pub struct Body {
    pub input: Input,
    pub predicate: SegBody,
}

impl Body {
    pub(crate) fn for_each_type_mut(&mut self, visit: &mut impl FnMut(&mut Type<TypeName>)) {
        let input = match &mut self.input {
            Input::Plain(input) => input,
            Input::Mapped {
                input,
                output_element_type,
                ..
            } => {
                visit(output_element_type);
                input
            }
        };
        visit(&mut input.array);
        visit(&mut input.element);
    }

    pub fn output_element_type(&self) -> &Type<TypeName> {
        match &self.input {
            Input::Plain(input) => &input.element,
            Input::Mapped {
                output_element_type, ..
            } => output_element_type,
        }
    }

    pub(crate) fn capture_nodes(&self) -> Vec<NodeId> {
        let mut nodes = match &self.input {
            Input::Plain(_) => Vec::new(),
            Input::Mapped { body, .. } => body.captures.clone(),
        };
        nodes.extend(self.predicate.captures.iter().copied());
        nodes
    }

    fn referenced_node_slots(&mut self) -> Vec<&mut NodeId> {
        let Self { input, predicate } = self;
        let mut nodes = match input {
            Input::Plain(_) => Vec::new(),
            Input::Mapped { body, .. } => body.captures.iter_mut().collect(),
        };
        nodes.extend(predicate.captures.iter_mut());
        nodes
    }
}

#[derive(Clone, Debug)]
pub enum RawStorage<R> {
    Local {
        capacity: Type<TypeName>,
        destination: SoacDestination,
    },
    Runtime {
        scratch: R,
        length: RuntimeLength<R>,
    },
}

#[derive(Clone, Debug)]
pub struct RawState<R> {
    pub storage: RawStorage<R>,
}

impl<R> RawState<R> {
    pub(crate) fn for_each_type_mut(&mut self, visit: &mut impl FnMut(&mut Type<TypeName>)) {
        if let RawStorage::Local { capacity, .. } = &mut self.storage {
            visit(capacity);
        }
    }
}

#[derive(Clone, Debug)]
pub struct SemanticState<R> {
    pub space: SegSpace<R>,
    pub storage: Output<R>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuntimeStorage<R> {
    pub scratch: R,
    pub length: RuntimeLength<R>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParallelStage {
    Flags,
    Scan,
    Scatter,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParallelPlan<R> {
    pub stage: ParallelStage,
    pub buffers: WorkBuffers<R>,
}

#[derive(Clone, Debug)]
pub enum ScheduledState<R> {
    Serial {
        space: SegSpace<R>,
        storage: Output<R>,
    },
    Parallel {
        space: SegSpace<R>,
        storage: RuntimeStorage<R>,
        plan: ParallelPlan<R>,
    },
}

impl<R> ScheduledState<R> {
    pub(crate) fn for_each_type_mut(&mut self, visit: &mut impl FnMut(&mut Type<TypeName>)) {
        if let Self::Serial {
            storage: Output::Local { capacity, .. },
            ..
        } = self
        {
            visit(capacity);
        }
    }
}

pub type PhysicalState = ScheduledState<PhysicalResourceRef>;

#[derive(Clone, Debug)]
pub struct Op<P: EgirPhase> {
    pub body: Body,
    pub state: P::FilterState,
}

impl<R: GraphResource> Op<Semantic<R>> {
    pub(crate) fn capture_nodes(&self) -> Vec<NodeId> {
        self.body.capture_nodes()
    }

    pub(crate) fn referenced_nodes(&self) -> Vec<NodeId> {
        let mut nodes = self.body.capture_nodes();
        nodes.extend(self.state.space.referenced_nodes());
        nodes
    }

    pub(crate) fn referenced_node_slots(&mut self) -> Vec<&mut NodeId> {
        let Self { body, state } = self;
        let mut nodes = body.referenced_node_slots();
        nodes.extend(state.space.referenced_node_slots());
        nodes
    }
}

/// Resolve compaction capacity from the post-fusion semantic domain.
pub(crate) fn resolve_scratch_sizes(inner: &mut SemanticProgram) {
    let mut resolved = Vec::new();
    for (_, entry) in inner.entries_with_endpoints() {
        for (_, block) in &entry.graph.skeleton.blocks {
            for effect in &block.side_effects {
                let SideEffectKind::Soac(
                    _,
                    Soac::Filter(Op {
                        body,
                        state:
                            SemanticState {
                                space,
                                storage: Output::Runtime { scratch, .. },
                            },
                    }),
                ) = &effect.kind
                else {
                    continue;
                };
                let elem_bytes =
                    crate::ssa::layout::storage_elem_stride(body.output_element_type()).unwrap_or(1);
                let size = match space.dims.as_slice() {
                    [SegExtent::Fixed(count)] => LogicalSize::FixedBytes(*count as u64 * elem_bytes as u64),
                    [SegExtent::ResourceLength {
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
                    } => inner
                        .resources
                        .get(resource.0 as usize)
                        .and_then(LogicalResource::host_binding)
                        .map(|binding| crate::pipeline_descriptor::BufferLen::LikeInput {
                            set: binding.set,
                            binding: binding.binding,
                            elem_bytes: *elem_bytes,
                            src_elem_bytes: *src_elem_bytes,
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
    for (resource, size, output_len) in resolved {
        if let Some(logical) = inner.resources.get_mut(resource.0 as usize) {
            logical.size = size.clone();
        }
        for entry in inner
            .entry_points
            .iter_mut()
            .chain(inner.materializations.iter_mut().map(|requirement| &mut requirement.entry))
        {
            if let Some(declaration) = entry
                .resource_declarations
                .iter_mut()
                .find(|declaration| declaration.resource.0 == resource)
            {
                declaration.size = size.clone();
            }
            for (slot, output_resource) in entry.resource_abi.outputs.iter().enumerate() {
                if *output_resource == Some(resource) {
                    entry.outputs[slot].length = output_len.clone();
                }
            }
        }
    }
}

pub(crate) fn allocate_work_resources(inner: &mut SemanticProgram) {
    let mut pending = Vec::new();
    for (_, entry) in inner.entries_with_endpoints() {
        for (_, block) in &entry.graph.skeleton.blocks {
            for effect in &block.side_effects {
                let SideEffectKind::Soac(
                    owner,
                    Soac::Filter(Op {
                        state:
                            SemanticState {
                                space,
                                storage: Output::Runtime { .. },
                            },
                        ..
                    }),
                ) = &effect.kind
                else {
                    continue;
                };
                let element_count_size = match space.dims.first() {
                    Some(SegExtent::Fixed(count)) if space.dims.len() == 1 => {
                        LogicalSize::FixedBytes(*count as u64 * 4)
                    }
                    Some(SegExtent::ResourceLength {
                        resource, elem_bytes, ..
                    }) if space.dims.len() == 1 => LogicalSize::LikeResource {
                        resource: resource.0,
                        elem_bytes: 4,
                        src_elem_bytes: *elem_bytes,
                    },
                    _ => LogicalSize::SameAsDispatch { elem_bytes: 4 },
                };
                let worker_count_size = LogicalSize::FixedBytes(
                    (super::super::parallelize::FILTER_SCAN_GROUPS
                        * super::super::parallelize::REDUCE_PHASE1_WIDTH) as u64
                        * 4,
                );
                let owner = Some(*owner);
                for (slot, (kind, size)) in [
                    (CompilerResourceKind::FilterFlags, element_count_size.clone()),
                    (CompilerResourceKind::FilterOffsets, element_count_size.clone()),
                    (
                        CompilerResourceKind::FilterScanBlockSums,
                        worker_count_size.clone(),
                    ),
                    (
                        CompilerResourceKind::FilterScanBlockOffsets,
                        worker_count_size.clone(),
                    ),
                ]
                .into_iter()
                .enumerate()
                {
                    pending.push((CompilerResource::new(kind, owner, slot), size));
                }
            }
        }
    }
    for (compiler, size) in pending {
        inner.alloc_compiler_resource(compiler, Type::Constructed(TypeName::UInt(32), vec![]), size);
    }
}

/// Runtime filter identities that predate logical allocation.
pub(crate) fn resource_kinds(inner: &SemanticProgram) -> HashMap<ResourceId, CompilerResource> {
    let mut kinds = HashMap::new();
    for (_, entry) in inner.entries_with_endpoints() {
        for (_, block) in &entry.graph.skeleton.blocks {
            for effect in &block.side_effects {
                let SideEffectKind::Soac(
                    owner,
                    Soac::Filter(Op {
                        state:
                            SemanticState {
                                storage: Output::Runtime { scratch, length },
                                ..
                            },
                        ..
                    }),
                ) = &effect.kind
                else {
                    continue;
                };
                let owner = Some(*owner);
                kinds.insert(
                    scratch.0,
                    CompilerResource::new(CompilerResourceKind::FilterScratch, owner, 0),
                );
                if let RuntimeLength::Stored(length) = length {
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
