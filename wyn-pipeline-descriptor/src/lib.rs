//! Pipeline descriptor for compiled Wyn programs.
//!
//! The compiler emits a JSON pipeline descriptor alongside the SPIR-V module
//! describing how to execute the program: which entry points to invoke, in
//! what order, and what GPU resources (buffers, uniforms, push constants) each
//! stage uses.
//!
//! A generic host runtime (e.g. `viz`) reads this descriptor and sets up the
//! Vulkan/WebGPU pipeline accordingly. All algorithm knowledge lives in the
//! compiler.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

/// Top-level pipeline descriptor. One per compiled program.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PipelineDescriptor {
    /// Individual pipelines in this program (one per top-level entry or multi-dispatch SOAC).
    pub pipelines: Vec<Pipeline>,
    /// Descriptor-derived pass/resource DAG. The compiler rebuilds this after
    /// binding publication so host runtimes can drive scheduling and allocation
    /// from data dependencies instead of hand-authored pass lists.
    #[serde(default, skip_serializing_if = "FrameGraph::is_empty")]
    pub frame_graph: FrameGraph,
}

impl PipelineDescriptor {
    /// Rebuild the frame graph from the currently published pipelines.
    pub fn rebuild_frame_graph(&mut self) {
        self.frame_graph = FrameGraph::from_pipelines(&self.pipelines);
    }
}

/// A descriptor-level frame graph: passes, logical resources, and the
/// dependencies induced by same-frame reads/writes.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FrameGraph {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub passes: Vec<FramePass>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub resources: Vec<FrameResource>,
    /// Ping-pong/history pairs resolved to logical frame-graph resources.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub feedback: Vec<FrameFeedback>,
    /// Reserved for future indirect draw command buffers. Keeping this in the
    /// schema lets runtimes consume the descriptor as the scheduling contract.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub indirect_draws: Vec<IndirectDrawDependency>,
}

impl FrameGraph {
    pub fn is_empty(&self) -> bool {
        self.passes.is_empty()
            && self.resources.is_empty()
            && self.feedback.is_empty()
            && self.indirect_draws.is_empty()
    }

    pub fn from_pipelines(pipelines: &[Pipeline]) -> Self {
        let mut builder = FrameGraphBuilder::default();

        for (pipeline_index, pipeline) in pipelines.iter().enumerate() {
            for (binding_index, binding) in pipeline_bindings(pipeline).iter().enumerate() {
                builder.ensure_binding(pipeline_index, binding_index, binding);
            }
        }

        for (pipeline_index, pipeline) in pipelines.iter().enumerate() {
            match pipeline {
                Pipeline::Compute(compute) => {
                    builder.publish_feedback(pipeline_index, &compute.bindings, &compute.feedback);
                }
                Pipeline::Graphics(graphics) => {
                    builder.publish_feedback(pipeline_index, &graphics.bindings, &graphics.feedback);
                }
            }
        }

        let mut last_writer = vec![None; builder.graph.resources.len()];
        let mut last_readers = vec![BTreeSet::new(); builder.graph.resources.len()];
        for (pipeline_index, pipeline) in pipelines.iter().enumerate() {
            match pipeline {
                Pipeline::Compute(compute) => {
                    let feedback_reads = feedback_read_slots(&compute.feedback);
                    for (stage_index, stage) in compute.stages.iter().enumerate() {
                        let (reads, writes) =
                            builder.compute_stage_accesses(pipeline_index, compute, stage, &feedback_reads);
                        builder.push_pass(
                            FramePassKind::Compute,
                            stage.entry_point.clone(),
                            pipeline_index,
                            stage_index,
                            reads,
                            writes,
                            &mut last_writer,
                            &mut last_readers,
                        );
                    }
                }
                Pipeline::Graphics(graphics) => {
                    let feedback_reads = feedback_read_slots(&graphics.feedback);
                    // The descriptor's graphics binding table is flat — it does
                    // not record which stage (vertex vs fragment) uses each
                    // binding — so every stage is attributed the whole table's
                    // declared accesses. This is a conservative over-
                    // approximation: intra-pipeline `depends_on` edges between a
                    // pipeline's own stages may be spurious. It is sound (no real
                    // dependency is missed) and currently has no consumer; a
                    // precise split would need per-binding stage visibility.
                    // Computed once and shared, since the accesses are identical.
                    let (reads, writes) = builder.binding_table_accesses(
                        pipeline_index,
                        &graphics.bindings,
                        None,
                        None,
                        &feedback_reads,
                    );
                    for (stage_index, stage) in graphics.stages.iter().enumerate() {
                        builder.push_pass(
                            FramePassKind::from_shader_stage(&stage.stage),
                            stage.entry_point.clone(),
                            pipeline_index,
                            stage_index,
                            reads.clone(),
                            writes.clone(),
                            &mut last_writer,
                            &mut last_readers,
                        );
                    }
                }
            }
        }

        builder.graph
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FramePass {
    pub name: String,
    pub kind: FramePassKind,
    pub pipeline_index: usize,
    pub stage_index: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reads: Vec<FrameAccess>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub writes: Vec<FrameAccess>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub depends_on: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FramePassKind {
    Compute,
    Vertex,
    Fragment,
}

impl FramePassKind {
    fn from_shader_stage(stage: &ShaderStage) -> Self {
        match stage {
            ShaderStage::Vertex => FramePassKind::Vertex,
            ShaderStage::Fragment => FramePassKind::Fragment,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameAccess {
    pub resource: usize,
    #[serde(default, skip_serializing_if = "FrameAccessRole::is_current")]
    pub role: FrameAccessRole,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FrameAccessRole {
    #[default]
    Current,
    Previous,
}

impl FrameAccessRole {
    fn is_current(&self) -> bool {
        matches!(self, FrameAccessRole::Current)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameResource {
    pub name: String,
    pub kind: FrameResourceKind,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub bindings: Vec<FrameBindingRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extent: Option<FrameResourceExtent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_pass: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_pass: Option<usize>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub history: Vec<FrameHistoryRole>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FrameResourceKind {
    StorageBuffer,
    Uniform,
    PushConstant,
    Texture,
    Sampler,
    StorageTexture,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameBindingRef {
    pub pipeline_index: usize,
    pub binding_index: usize,
    pub name: String,
    pub kind: FrameResourceKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub set: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub binding: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FrameResourceExtent {
    StorageTexture {
        size: StorageTextureSize,
    },
    StorageBuffer {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        length: Option<BufferLen>,
    },
    Uniform {
        bytes: u32,
    },
    PushConstant {
        bytes: u32,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FrameHistoryRole {
    pub feedback: usize,
    pub role: FrameHistoryRoleKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FrameHistoryRoleKind {
    ReadPrevious,
    WriteCurrent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameFeedback {
    pub pipeline_index: usize,
    pub pair: FeedbackPair,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub read_resource: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub write_resource: Option<usize>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct IndirectDrawDependency {
    pub draw_pass: usize,
    pub buffer_resource: usize,
}

/// A single pipeline within the program.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Pipeline {
    /// One or more compute dispatches sharing a binding table. A
    /// single-dispatch SOAC (Map, Scatter, simple compute) is the
    /// `stages.len() == 1` case; multi-dispatch SOACs (Reduce, Scan,
    /// Filter, ordered-prefix scheduling) populate multiple stages
    /// run in order by the host runtime.
    Compute(ComputePipeline),
    /// Graphics pipeline (Vertex → Fragment).
    Graphics(GraphicsPipeline),
}

/// Compute pipeline: a binding table plus N≥1 dispatch stages run in
/// order, sharing the same bindings. The `stages.len() == 1` case
/// covers single-dispatch SOACs; multi-stage covers Reduce/Scan/
/// Filter phase chains and the ordered-prefix scheduler's lifted
/// stages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputePipeline {
    /// All bindings used across all stages.
    pub bindings: Vec<Binding>,
    /// Stages to execute in order. Length ≥ 1.
    pub stages: Vec<ComputeStage>,
    /// Host-runtime default for the total work size, sourced from
    /// `#[size_hint(N)]` on an input parameter. When the application
    /// doesn't supply an explicit dispatch count, a thin host can
    /// dispatch `ceil(default_total_threads / workgroup_size.0)`
    /// workgroups without inspecting buffer length. The compiled
    /// shader does not assume the actual length equals this hint —
    /// it remains dynamic.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_total_threads: Option<std::num::NonZeroU32>,
    /// Ping-pong feedback pairs: each entry's `read` slot samples the previous
    /// frame of its `write` slot (a `history` resource's `previous` view). The
    /// runtime double-buffers and swaps each frame — the declarative form of a
    /// `--feedback ENTRY:READ=WRITE` flag.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub feedback: Vec<FeedbackPair>,
}

/// A ping-pong feedback pair within a pipeline: the read slot samples the
/// previous frame of the write slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeedbackPair {
    pub read_set: u32,
    pub read_binding: u32,
    pub write_set: u32,
    pub write_binding: u32,
}

/// The `(set, binding)` of a `StorageTexture` allocation that a sampled
/// `Texture` binding is a view of. See `Binding::Texture::backing`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackingRef {
    pub set: u32,
    pub binding: u32,
}

/// A single dispatch stage within a `ComputePipeline`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeStage {
    pub entry_point: String,
    pub workgroup_size: (u32, u32, u32),
    pub dispatch_size: DispatchSize,
    /// Indices into the parent pipeline's `bindings` that this stage
    /// reads. Empty when the host derives access from `Binding.access`
    /// (the single-stage case has historically left this empty).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reads: Vec<usize>,
    /// Indices into the parent pipeline's `bindings` that this stage
    /// writes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub writes: Vec<usize>,
}

/// Graphics pipeline (vertex + fragment stages).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphicsPipeline {
    pub stages: Vec<GraphicsStage>,
    pub bindings: Vec<Binding>,
    pub vertex_inputs: Vec<VertexAttribute>,
    pub fragment_outputs: Vec<FragmentOutput>,
    /// Ping-pong feedback pairs (see `ComputePipeline::feedback`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub feedback: Vec<FeedbackPair>,
}

/// A stage in a graphics pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphicsStage {
    pub entry_point: String,
    pub stage: ShaderStage,
}

/// Shader stage type.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShaderStage {
    Vertex,
    Fragment,
}

/// How to determine the compute dispatch grid size.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DispatchSize {
    /// Fixed dispatch grid (absolute workgroup counts).
    Fixed {
        x: u32,
        y: u32,
        z: u32,
        /// `true` when this grid was deliberately chosen (a source
        /// `#[dispatch(...)]` or a compiler-pinned phase) rather than the
        /// default `1x1x1` placeholder that domain inference may upgrade.
        /// Lets the scheduler tell a user-pinned `#[dispatch(1,1,1)]` apart
        /// from the unspecified default instead of guessing from the value.
        #[serde(default)]
        explicit: bool,
    },
    /// Dispatch `ceil(len / workgroup_size)` workgroups, where `len` is the
    /// number of iterations resolved from `DispatchLen`. Replaces the old
    /// `DerivedFromInputLength`, which only said "derive from *an* input
    /// buffer" and silently guessed the wrong one for range-driven maps.
    DerivedFrom {
        len: DispatchLen,
        workgroup_size: u32,
    },
}

/// The source of truth for a `DerivedFrom` dispatch's iteration count.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DispatchLen {
    /// One iteration per element of the buffer at (`set`, `binding`) — e.g.
    /// `map(f, arr)` over a storage-buffer input. The host reads the buffer's
    /// element count.
    InputBinding {
        set: u32,
        binding: u32,
        /// Bytes per element of that buffer, so the host recovers the element
        /// count from its byte size.
        elem_bytes: u32,
    },
    /// A compile-time-known iteration count — e.g. `map(f, iota(6144))`.
    /// (Struct variant, not `Fixed(u32)`, so it serializes under the internal
    /// `kind` tag.)
    Fixed {
        count: u32,
    },
    /// A runtime count read from a scalar push-constant — e.g. `map(f,
    /// iota(n))` where `n` is an entry parameter. The host reads the u32 at
    /// `offset` in the push-constant block.
    PushConstant {
        offset: u32,
    },
    /// One iteration per texel of the storage texture at (`set`,
    /// `binding`) — used for compute entries whose primary output is a
    /// storage image update. The host reads the allocated
    /// `wgpu::Texture`'s `width × height` (the storage texture's
    /// resolution is set by the descriptor's `StorageTextureSize`
    /// policy at allocation time). 2D dispatch: the host divides by
    /// the workgroup_size's x/y dims to produce workgroup counts.
    StorageImage {
        set: u32,
        binding: u32,
    },
}

/// One member of a uniform block: where the host writes the value.
/// Mirrors the `PushConstant { offset, size }` contract, per member.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniformMember {
    pub name: String,
    pub offset: u32,
    pub size: u32,
}

/// A GPU resource binding used by the pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Binding {
    /// Storage buffer (descriptor set binding).
    StorageBuffer {
        set: u32,
        binding: u32,
        access: Access,
        usage: BufferUsage,
        name: String,
        /// Sizing policy for compiler-managed buffers whose length isn't a
        /// host-supplied input (e.g. a gather intermediate). `None` for
        /// host inputs (sized from the supplied data) and ordinary outputs.
        #[serde(default)]
        length: Option<BufferLen>,
    },
    /// Uniform buffer (descriptor set binding).
    Uniform {
        set: u32,
        binding: u32,
        name: String,
        /// std140 byte size of the block. `0` in descriptors that
        /// predate block-layout publication (hosts fall back to their
        /// known-name tables).
        #[serde(default)]
        size: u32,
        /// Flattened block members in declaration order — the record
        /// fields of a record-typed uniform. Empty when unpublished.
        #[serde(default)]
        members: Vec<UniformMember>,
    },
    /// Push constant range.
    PushConstant {
        offset: u32,
        size: u32,
        name: String,
    },
    /// Sampled texture (descriptor set binding). Bound from a
    /// `#[texture(set, binding)]` entry-point param of type `texture2d`.
    ///
    /// `backing`, when present, names the `StorageTexture` binding whose
    /// allocation this is a sampled *view* of — a `resource`'s `sampled`
    /// view aliasing its `storage_write` allocation. The runtime binds this
    /// slot to that allocation's sampled view (current frame); a previous
    /// view of the same allocation is additionally listed in `feedback`.
    /// `None` is a host-provided / external texture.
    Texture {
        set: u32,
        binding: u32,
        name: String,
        sample_type: TextureSampleType,
        view_dimension: TextureViewDimension,
        multisampled: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        backing: Option<BackingRef>,
    },
    /// Sampler (descriptor set binding). Bound from a
    /// `#[sampler(set, binding)]` entry-point param of type `sampler`.
    Sampler {
        set: u32,
        binding: u32,
        name: String,
        binding_type: SamplerBindingType,
    },
    /// Storage image (descriptor set binding). Bound from a
    /// `#[storage_image(set, binding, format, access)]` entry-point
    /// param of type `storage_image`. The same `(set, binding)` slot
    /// may also be declared as a `Texture` in another pipeline — viz
    /// allocates one wgpu texture and binds it through two views.
    ///
    /// `size` is the resolution policy the host uses to allocate the
    /// backing `wgpu::Texture`. Defaults to `SameAsWindow` so a
    /// compute shader writing per-pixel naturally tracks the swapchain
    /// size; producers that want a fixed grid (e.g. the Mountains
    /// shader's BUFFER_SIZE-capped erosion textures) opt in to
    /// `Fixed`.
    StorageTexture {
        set: u32,
        binding: u32,
        name: String,
        format: StorageImageFormat,
        access: Access,
        #[serde(default)]
        size: StorageTextureSize,
    },
}

/// Resolution policy for a storage texture's backing `wgpu::Texture`.
/// Resolved by the host at allocation time (and on window resize for
/// `SameAsWindow`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StorageTextureSize {
    /// Track the swapchain surface size. The default — a fragment
    /// shader sampling this texture covers each output pixel exactly
    /// once.
    #[default]
    SameAsWindow,
    /// Fixed `(width, height)` in pixels. Used when the producer's
    /// dispatch is sized to a constant grid (e.g. the Mountains
    /// shader's `BUFFER_SIZE` cap that decouples compute resolution
    /// from window resolution).
    Fixed {
        width: u32,
        height: u32,
    },
}

impl Binding {
    /// Descriptor-set binding number for storage / uniform / texture /
    /// sampler bindings. Panics on `PushConstant`, which has no binding
    /// number — push constants live in their own range and are addressed
    /// by offset.
    pub fn wgpu_binding(&self) -> u32 {
        match self {
            Binding::StorageBuffer { binding, .. } => *binding,
            Binding::Uniform { binding, .. } => *binding,
            Binding::Texture { binding, .. } => *binding,
            Binding::Sampler { binding, .. } => *binding,
            Binding::StorageTexture { binding, .. } => *binding,
            Binding::PushConstant { .. } => panic!("PushConstant has no binding number"),
        }
    }

    /// True iff this is a storage buffer marked as a host-supplied input.
    pub fn is_input(&self) -> bool {
        matches!(
            self,
            Binding::StorageBuffer {
                usage: BufferUsage::Input,
                ..
            }
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ResourceKey {
    Descriptor {
        kind: FrameResourceKind,
        set: u32,
        binding: u32,
    },
    PushConstant {
        pipeline_index: usize,
        offset: u32,
        size: u32,
    },
}

#[derive(Default)]
struct FrameGraphBuilder {
    graph: FrameGraph,
    resources: BTreeMap<ResourceKey, usize>,
}

impl FrameGraphBuilder {
    fn ensure_binding(&mut self, pipeline_index: usize, binding_index: usize, binding: &Binding) -> usize {
        let key = resource_key(pipeline_index, binding);
        let index = if let Some(&index) = self.resources.get(&key) {
            index
        } else {
            let index = self.graph.resources.len();
            self.resources.insert(key, index);
            self.graph.resources.push(FrameResource {
                name: binding_name(binding).to_string(),
                kind: resource_kind_from_key(key),
                bindings: Vec::new(),
                extent: binding_extent(binding),
                first_pass: None,
                last_pass: None,
                history: Vec::new(),
            });
            index
        };

        let resource = &mut self.graph.resources[index];
        // Bindings merging onto one slot must be the same resource kind — the
        // sole cross-kind case is a sampled texture view aliasing its
        // storage-texture backing (keyed to the backing's StorageTexture slot).
        debug_assert!(
            binding_kind(binding) == resource.kind
                || (matches!(binding, Binding::Texture { backing: Some(_), .. })
                    && resource.kind == FrameResourceKind::StorageTexture),
            "binding kind {:?} disagrees with merged resource kind {:?} at one slot",
            binding_kind(binding),
            resource.kind
        );
        merge_extent(&mut resource.extent, binding_extent(binding));
        if matches!(binding, Binding::StorageTexture { .. }) {
            resource.name = binding_name(binding).to_string();
        }

        let binding_ref = FrameBindingRef {
            pipeline_index,
            binding_index,
            name: binding_name(binding).to_string(),
            kind: binding_kind(binding),
            set: binding_slot(binding).map(|(set, _)| set),
            binding: binding_slot(binding).map(|(_, binding)| binding),
        };
        if !resource.bindings.iter().any(|existing| {
            existing.pipeline_index == pipeline_index && existing.binding_index == binding_index
        }) {
            resource.bindings.push(binding_ref);
        }
        index
    }

    fn publish_feedback(&mut self, pipeline_index: usize, bindings: &[Binding], pairs: &[FeedbackPair]) {
        for pair in pairs {
            let read_resource =
                self.resource_for_slot(pipeline_index, bindings, pair.read_set, pair.read_binding);
            let write_resource =
                self.resource_for_slot(pipeline_index, bindings, pair.write_set, pair.write_binding);
            let feedback_index = self.graph.feedback.len();
            if let Some(resource) = read_resource {
                self.push_history_role(
                    resource,
                    FrameHistoryRole {
                        feedback: feedback_index,
                        role: FrameHistoryRoleKind::ReadPrevious,
                    },
                );
            }
            if let Some(resource) = write_resource {
                self.push_history_role(
                    resource,
                    FrameHistoryRole {
                        feedback: feedback_index,
                        role: FrameHistoryRoleKind::WriteCurrent,
                    },
                );
            }
            self.graph.feedback.push(FrameFeedback {
                pipeline_index,
                pair: *pair,
                read_resource,
                write_resource,
            });
        }
    }

    fn push_history_role(&mut self, resource: usize, role: FrameHistoryRole) {
        let roles = &mut self.graph.resources[resource].history;
        if !roles.iter().any(|existing| existing.feedback == role.feedback && existing.role == role.role) {
            roles.push(role);
        }
    }

    fn resource_for_slot(
        &mut self,
        pipeline_index: usize,
        bindings: &[Binding],
        set: u32,
        binding: u32,
    ) -> Option<usize> {
        bindings
            .iter()
            .enumerate()
            .find(|(_, candidate)| binding_slot(candidate) == Some((set, binding)))
            .map(|(binding_index, candidate)| self.ensure_binding(pipeline_index, binding_index, candidate))
    }

    fn compute_stage_accesses(
        &mut self,
        pipeline_index: usize,
        compute: &ComputePipeline,
        stage: &ComputeStage,
        feedback_reads: &BTreeSet<(u32, u32)>,
    ) -> (Vec<FrameAccess>, Vec<FrameAccess>) {
        let explicit_reads = (!stage.reads.is_empty()).then_some(stage.reads.as_slice());
        let explicit_writes = (!stage.writes.is_empty()).then_some(stage.writes.as_slice());
        self.binding_table_accesses(
            pipeline_index,
            &compute.bindings,
            explicit_reads,
            explicit_writes,
            feedback_reads,
        )
    }

    fn binding_table_accesses(
        &mut self,
        pipeline_index: usize,
        bindings: &[Binding],
        explicit_reads: Option<&[usize]>,
        explicit_writes: Option<&[usize]>,
        feedback_reads: &BTreeSet<(u32, u32)>,
    ) -> (Vec<FrameAccess>, Vec<FrameAccess>) {
        let explicit = explicit_reads.is_some() || explicit_writes.is_some();
        let mut reads = Vec::new();
        let mut writes = Vec::new();

        if explicit {
            for index in explicit_reads.into_iter().flatten().copied() {
                if let Some(binding) = bindings.get(index) {
                    self.push_read(&mut reads, pipeline_index, index, binding, feedback_reads);
                }
            }
            for index in explicit_writes.into_iter().flatten().copied() {
                if let Some(binding) = bindings.get(index) {
                    self.push_write(&mut writes, pipeline_index, index, binding);
                }
            }
        }

        for (index, binding) in bindings.iter().enumerate() {
            // In explicit mode the stage's read/write lists are the *complete*
            // access spec for storage buffers — a storage buffer absent from
            // them is not touched by this stage, so it must not auto-derive.
            // Other binding kinds (textures, uniforms, samplers) are never in
            // the lists and always auto-derive from their declared access. If a
            // stage ever needs explicit read/write control over a non-buffer
            // binding, the lists (populated upstream in EGIR) must be extended
            // to name it and this carve-out generalized to "skip if listed".
            if explicit && matches!(binding, Binding::StorageBuffer { .. }) {
                continue;
            }
            let (read, write) = binding_declared_access(binding);
            if read {
                self.push_read(&mut reads, pipeline_index, index, binding, feedback_reads);
            }
            if write {
                self.push_write(&mut writes, pipeline_index, index, binding);
            }
        }

        (reads, writes)
    }

    fn push_read(
        &mut self,
        reads: &mut Vec<FrameAccess>,
        pipeline_index: usize,
        binding_index: usize,
        binding: &Binding,
        feedback_reads: &BTreeSet<(u32, u32)>,
    ) {
        let resource = self.ensure_binding(pipeline_index, binding_index, binding);
        let role = if binding_slot(binding).is_some_and(|slot| feedback_reads.contains(&slot)) {
            FrameAccessRole::Previous
        } else {
            FrameAccessRole::Current
        };
        push_unique_access(reads, FrameAccess { resource, role });
    }

    fn push_write(
        &mut self,
        writes: &mut Vec<FrameAccess>,
        pipeline_index: usize,
        binding_index: usize,
        binding: &Binding,
    ) {
        let resource = self.ensure_binding(pipeline_index, binding_index, binding);
        push_unique_access(
            writes,
            FrameAccess {
                resource,
                role: FrameAccessRole::Current,
            },
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn push_pass(
        &mut self,
        kind: FramePassKind,
        name: String,
        pipeline_index: usize,
        stage_index: usize,
        reads: Vec<FrameAccess>,
        writes: Vec<FrameAccess>,
        last_writer: &mut Vec<Option<usize>>,
        last_readers: &mut Vec<BTreeSet<usize>>,
    ) {
        let needed = self.graph.resources.len();
        if last_writer.len() < needed {
            last_writer.resize(needed, None);
            last_readers.resize_with(needed, BTreeSet::new);
        }

        let mut depends_on = BTreeSet::new();
        for access in &reads {
            if access.role == FrameAccessRole::Current {
                if let Some(writer) = last_writer[access.resource] {
                    depends_on.insert(writer);
                }
            }
        }
        for access in &writes {
            if let Some(writer) = last_writer[access.resource] {
                depends_on.insert(writer);
            }
            depends_on.extend(last_readers[access.resource].iter().copied());
        }

        let pass_index = self.graph.passes.len();
        for access in reads.iter().chain(writes.iter()) {
            let resource = &mut self.graph.resources[access.resource];
            resource.first_pass.get_or_insert(pass_index);
            resource.last_pass = Some(pass_index);
        }

        for access in &writes {
            last_writer[access.resource] = Some(pass_index);
            last_readers[access.resource].clear();
        }
        for access in &reads {
            if access.role == FrameAccessRole::Current
                && !writes.iter().any(|write| write.resource == access.resource)
            {
                last_readers[access.resource].insert(pass_index);
            }
        }

        self.graph.passes.push(FramePass {
            name,
            kind,
            pipeline_index,
            stage_index,
            reads,
            writes,
            depends_on: depends_on.into_iter().collect(),
        });
    }
}

fn pipeline_bindings(pipeline: &Pipeline) -> &[Binding] {
    match pipeline {
        Pipeline::Compute(compute) => &compute.bindings,
        Pipeline::Graphics(graphics) => &graphics.bindings,
    }
}

fn resource_key(pipeline_index: usize, binding: &Binding) -> ResourceKey {
    match binding {
        Binding::StorageBuffer { set, binding, .. } => ResourceKey::Descriptor {
            kind: FrameResourceKind::StorageBuffer,
            set: *set,
            binding: *binding,
        },
        Binding::Uniform { set, binding, .. } => ResourceKey::Descriptor {
            kind: FrameResourceKind::Uniform,
            set: *set,
            binding: *binding,
        },
        Binding::PushConstant { offset, size, .. } => ResourceKey::PushConstant {
            pipeline_index,
            offset: *offset,
            size: *size,
        },
        Binding::Texture {
            set,
            binding,
            backing,
            ..
        } => {
            let (kind, set, binding) = if let Some(backing) = backing {
                (FrameResourceKind::StorageTexture, backing.set, backing.binding)
            } else {
                (FrameResourceKind::Texture, *set, *binding)
            };
            ResourceKey::Descriptor { kind, set, binding }
        }
        Binding::Sampler { set, binding, .. } => ResourceKey::Descriptor {
            kind: FrameResourceKind::Sampler,
            set: *set,
            binding: *binding,
        },
        Binding::StorageTexture { set, binding, .. } => ResourceKey::Descriptor {
            kind: FrameResourceKind::StorageTexture,
            set: *set,
            binding: *binding,
        },
    }
}

fn resource_kind_from_key(key: ResourceKey) -> FrameResourceKind {
    match key {
        ResourceKey::Descriptor { kind, .. } => kind,
        ResourceKey::PushConstant { .. } => FrameResourceKind::PushConstant,
    }
}

fn binding_kind(binding: &Binding) -> FrameResourceKind {
    match binding {
        Binding::StorageBuffer { .. } => FrameResourceKind::StorageBuffer,
        Binding::Uniform { .. } => FrameResourceKind::Uniform,
        Binding::PushConstant { .. } => FrameResourceKind::PushConstant,
        Binding::Texture { .. } => FrameResourceKind::Texture,
        Binding::Sampler { .. } => FrameResourceKind::Sampler,
        Binding::StorageTexture { .. } => FrameResourceKind::StorageTexture,
    }
}

fn binding_name(binding: &Binding) -> &str {
    match binding {
        Binding::StorageBuffer { name, .. }
        | Binding::Uniform { name, .. }
        | Binding::PushConstant { name, .. }
        | Binding::Texture { name, .. }
        | Binding::Sampler { name, .. }
        | Binding::StorageTexture { name, .. } => name,
    }
}

fn binding_slot(binding: &Binding) -> Option<(u32, u32)> {
    match binding {
        Binding::StorageBuffer { set, binding, .. }
        | Binding::Uniform { set, binding, .. }
        | Binding::Texture { set, binding, .. }
        | Binding::Sampler { set, binding, .. }
        | Binding::StorageTexture { set, binding, .. } => Some((*set, *binding)),
        Binding::PushConstant { .. } => None,
    }
}

fn binding_extent(binding: &Binding) -> Option<FrameResourceExtent> {
    match binding {
        Binding::StorageBuffer { length, .. } => Some(FrameResourceExtent::StorageBuffer {
            length: length.clone(),
        }),
        Binding::Uniform { size, .. } => Some(FrameResourceExtent::Uniform { bytes: *size }),
        Binding::PushConstant { size, .. } => Some(FrameResourceExtent::PushConstant { bytes: *size }),
        Binding::StorageTexture { size, .. } => Some(FrameResourceExtent::StorageTexture { size: *size }),
        Binding::Texture { .. } | Binding::Sampler { .. } => None,
    }
}

fn merge_extent(target: &mut Option<FrameResourceExtent>, candidate: Option<FrameResourceExtent>) {
    match candidate {
        Some(FrameResourceExtent::StorageTexture { size }) => {
            // A sampled texture view aliases its storage-texture backing onto
            // one slot; the storage-texture extent wins. Distinct resources
            // cannot share a slot (program-global allocation + the type-checker
            // reject it), so a differing existing storage-texture size would
            // signal a broken invariant.
            debug_assert!(
                !matches!(&target, Some(FrameResourceExtent::StorageTexture { size: existing }) if *existing != size),
                "two storage-texture aliases at one slot disagree on size"
            );
            *target = Some(FrameResourceExtent::StorageTexture { size });
        }
        Some(extent) => match target {
            None => *target = Some(extent),
            // Same slot ⇒ same resource ⇒ same extent; a mismatch means two
            // distinct resources collided (which the allocator/type-checker
            // forbid).
            Some(existing) => debug_assert_eq!(
                *existing, extent,
                "two bindings merging at one slot disagree on extent"
            ),
        },
        None => {}
    }
}

fn binding_declared_access(binding: &Binding) -> (bool, bool) {
    match binding {
        Binding::StorageBuffer { access, .. } | Binding::StorageTexture { access, .. } => match access {
            Access::ReadOnly => (true, false),
            Access::WriteOnly => (false, true),
            Access::ReadWrite => (true, true),
        },
        Binding::Uniform { .. }
        | Binding::PushConstant { .. }
        | Binding::Texture { .. }
        | Binding::Sampler { .. } => (true, false),
    }
}

fn feedback_read_slots(feedback: &[FeedbackPair]) -> BTreeSet<(u32, u32)> {
    feedback.iter().map(|pair| (pair.read_set, pair.read_binding)).collect()
}

fn push_unique_access(accesses: &mut Vec<FrameAccess>, access: FrameAccess) {
    if !accesses.contains(&access) {
        accesses.push(access);
    }
}

/// Compile-time sizing policy for a compiler-managed storage buffer whose
/// length isn't a host-supplied input. The host runtime resolves this to a
/// byte size when allocating the buffer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BufferLen {
    /// Minimum bytes the host must allocate. For an *output* binding this
    /// is the bytes the shader will write (so it's also the maximum
    /// useful size). For an *input* binding it's the bytes the shader
    /// will read (e.g. inferred from a `param[0..K]` slice); the host is
    /// free to over-allocate.
    Fixed {
        bytes: u64,
    },
    /// Same element *count* as the buffer at (`set`, `binding`), whose
    /// elements are `src_elem_bytes` each; this buffer's elements are
    /// `elem_bytes` each. Byte size = `src_bytes / src_elem_bytes *
    /// elem_bytes`. A `map` output keeps its input's element count but its
    /// element size may differ (e.g. `[]vec4f32` → `[]i32`).
    LikeInput {
        set: u32,
        binding: u32,
        elem_bytes: u32,
        src_elem_bytes: u32,
    },
    /// One `elem_bytes`-sized element per dispatched thread. A parallel
    /// `map`/`scan` writes exactly one output element per thread, so its
    /// output length equals the resolved dispatch thread count — which the
    /// host computes anyway (it covers buffer inputs, static and dynamic
    /// ranges uniformly). Byte size = `dispatch_threads * elem_bytes`. The
    /// thread count isn't a `src_bytes` lookup, so this resolves via
    /// `dispatch_elem_bytes`, not `resolve_bytes`.
    SameAsDispatch {
        elem_bytes: u32,
    },
}

impl BufferLen {
    /// Resolve to a byte size given a lookup of already-allocated buffers'
    /// byte sizes by (set, binding). Returns `None` if a referenced source
    /// buffer hasn't been sized yet, or for `SameAsDispatch` (which needs the
    /// resolved dispatch thread count — see `dispatch_elem_bytes`).
    pub fn resolve_bytes(&self, src_bytes: impl Fn(u32, u32) -> Option<u64>) -> Option<u64> {
        match self {
            BufferLen::Fixed { bytes } => Some(*bytes),
            BufferLen::LikeInput {
                set,
                binding,
                elem_bytes,
                src_elem_bytes,
            } => {
                let bytes = src_bytes(*set, *binding)?;
                Some(bytes / *src_elem_bytes as u64 * *elem_bytes as u64)
            }
            BufferLen::SameAsDispatch { .. } => None,
        }
    }

    /// Element byte size if this buffer is sized by the dispatch thread count
    /// (`SameAsDispatch`); the host multiplies it by the resolved threads.
    pub fn dispatch_elem_bytes(&self) -> Option<u32> {
        match self {
            BufferLen::SameAsDispatch { elem_bytes } => Some(*elem_bytes),
            _ => None,
        }
    }
}

/// Access mode for a storage buffer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Access {
    ReadOnly,
    WriteOnly,
    ReadWrite,
}

/// How a buffer is used in the pipeline.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BufferUsage {
    /// Read-only input from the host.
    Input,
    /// Written by the pipeline, read back by the host.
    Output,
    /// Internal to the pipeline (written by one stage, read by another).
    Intermediate,
}

/// Sampled type of a texture binding. Mirrors the wgpu
/// `TextureSampleType` subset Wyn produces. v1 always emits
/// `Float { filterable: true }` (the only `texture2d` sampled type).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextureSampleType {
    Float {
        filterable: bool,
    },
    Sint,
    Uint,
    Depth,
}

/// View dimension of a texture binding. v1 always emits `D2`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextureViewDimension {
    D1,
    D2,
    D2Array,
    Cube,
    CubeArray,
    D3,
}

/// Sampler binding mode. v1 always emits `Filtering`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SamplerBindingType {
    Filtering,
    NonFiltering,
    Comparison,
}

/// Pixel format for a storage-image binding. Bound at shader-compile
/// time via the `#[storage_image(..., format=FMT, ...)]` attribute;
/// the host allocates the wgpu texture with the matching format.
/// The whitelist starts narrow — formats are added as shaders demand
/// them. Names match the lowercase wgpu/WGSL spelling for round-trip
/// clarity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StorageImageFormat {
    Rgba8Unorm,
    Rgba16Float,
    Rgba32Float,
    R32Float,
}

/// Scalar/vector format of a vertex-buffer attribute. Mirrors the
/// wgpu `VertexFormat` subset Wyn can currently produce — 32-bit
/// float / signed-int / unsigned-int scalars and 2-4 wide vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VertexFormat {
    Float32,
    Float32x2,
    Float32x3,
    Float32x4,
    Sint32,
    Sint32x2,
    Sint32x3,
    Sint32x4,
    Uint32,
    Uint32x2,
    Uint32x3,
    Uint32x4,
}

impl VertexFormat {
    /// Byte size of one attribute element: 4 bytes per 32-bit component.
    pub fn byte_size(self) -> u32 {
        use VertexFormat::*;
        match self {
            Float32 | Sint32 | Uint32 => 4,
            Float32x2 | Sint32x2 | Uint32x2 => 8,
            Float32x3 | Sint32x3 | Uint32x3 => 12,
            Float32x4 | Sint32x4 | Uint32x4 => 16,
        }
    }
}

/// Vertex input attribute. One attribute == one vertex buffer: the
/// host uploads a tightly-packed buffer per attribute (offset 0,
/// stride = `format.byte_size()`), mirroring viz's one-`.bin`-per-
/// binding `--storage-dir` convention. Interleaved buffers (explicit
/// offset/stride) are a later extension.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VertexAttribute {
    pub slot: u32,
    pub name: String,
    pub format: VertexFormat,
}

/// Fragment output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FragmentOutput {
    pub location: u32,
    pub name: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertex_format_byte_size() {
        assert_eq!(VertexFormat::Float32.byte_size(), 4);
        assert_eq!(VertexFormat::Sint32.byte_size(), 4);
        assert_eq!(VertexFormat::Uint32.byte_size(), 4);
        assert_eq!(VertexFormat::Float32x2.byte_size(), 8);
        assert_eq!(VertexFormat::Float32x3.byte_size(), 12);
        assert_eq!(VertexFormat::Float32x4.byte_size(), 16);
        assert_eq!(VertexFormat::Uint32x4.byte_size(), 16);
    }

    #[test]
    fn buffer_len_resolve_like_input() {
        // A `[]vec4f32` (16 B/elem) input of 64 elements → 1024 bytes; a gather
        // of `[]i32` (4 B/elem) keeping its element count is 64 * 4 = 256 bytes.
        let len = BufferLen::LikeInput {
            set: 0,
            binding: 0,
            elem_bytes: 4,
            src_elem_bytes: 16,
        };
        assert_eq!(
            len.resolve_bytes(|s, b| (s == 0 && b == 0).then_some(1024)),
            Some(256)
        );
        // Source not yet allocated → unresolved.
        assert_eq!(len.resolve_bytes(|_, _| None), None);
        // Fixed is independent of any source.
        assert_eq!(
            BufferLen::Fixed { bytes: 40 }.resolve_bytes(|_, _| None),
            Some(40)
        );
    }

    #[test]
    fn buffer_len_same_as_dispatch() {
        // A dispatch-sized output isn't a `src_bytes` lookup — it resolves via
        // `dispatch_elem_bytes` and the host scales by the dispatch thread count.
        let len = BufferLen::SameAsDispatch { elem_bytes: 4 };
        assert_eq!(len.resolve_bytes(|_, _| Some(1024)), None);
        assert_eq!(len.dispatch_elem_bytes(), Some(4));
        assert_eq!(BufferLen::Fixed { bytes: 40 }.dispatch_elem_bytes(), None);
        let json = serde_json::to_string(&len).unwrap();
        assert!(json.contains("\"same_as_dispatch\""), "got: {json}");
        assert_eq!(serde_json::from_str::<BufferLen>(&json).unwrap(), len);
    }

    #[test]
    fn dispatch_len_serde_round_trip() {
        // `DerivedFrom` wraps an internally-tagged `DispatchLen`, so each source
        // variant must round-trip with its `kind` tag.
        for len in [
            DispatchLen::InputBinding {
                set: 0,
                binding: 1,
                elem_bytes: 4,
            },
            DispatchLen::Fixed { count: 6144 },
            DispatchLen::PushConstant { offset: 8 },
        ] {
            let size = DispatchSize::DerivedFrom {
                len: len.clone(),
                workgroup_size: 64,
            };
            let json = serde_json::to_string(&size).unwrap();
            assert!(json.contains("\"derived_from\""), "got: {json}");
            assert_eq!(serde_json::from_str::<DispatchSize>(&json).unwrap(), size);
        }
    }

    #[test]
    fn buffer_len_serde_round_trip() {
        let len = BufferLen::LikeInput {
            set: 0,
            binding: 2,
            elem_bytes: 4,
            src_elem_bytes: 16,
        };
        let json = serde_json::to_string(&len).unwrap();
        assert!(json.contains("\"like_input\""), "got: {json}");
        assert_eq!(serde_json::from_str::<BufferLen>(&json).unwrap(), len);
    }

    #[test]
    fn uniform_binding_members_serde_round_trip() {
        let binding = Binding::Uniform {
            set: 1,
            binding: 0,
            name: "c".to_string(),
            size: 32,
            members: vec![
                UniformMember {
                    name: "radius".to_string(),
                    offset: 0,
                    size: 4,
                },
                UniformMember {
                    name: "tint".to_string(),
                    offset: 8,
                    size: 8,
                },
            ],
        };
        let json = serde_json::to_string(&binding).unwrap();
        assert!(json.contains("\"members\""), "got: {json}");
        let back: Binding = serde_json::from_str(&json).unwrap();
        let Binding::Uniform { size, members, .. } = back else {
            panic!("round trip changed the variant");
        };
        assert_eq!(size, 32);
        assert_eq!(members.len(), 2);
        assert_eq!(
            (members[1].name.as_str(), members[1].offset, members[1].size),
            ("tint", 8, 8)
        );

        // Descriptors that predate size/members publication still parse:
        // the fields default to 0 / empty.
        let old = r#"{"type":"uniform","set":1,"binding":0,"name":"iTime"}"#;
        let Binding::Uniform { size, members, .. } = serde_json::from_str::<Binding>(old).unwrap() else {
            panic!("old-shape uniform must still parse");
        };
        assert_eq!(size, 0);
        assert!(members.is_empty());
    }

    #[test]
    fn frame_graph_aliases_storage_texture_views_and_orders_consumers() {
        let mut descriptor = PipelineDescriptor {
            pipelines: vec![
                Pipeline::Compute(ComputePipeline {
                    bindings: vec![Binding::StorageTexture {
                        set: 1,
                        binding: 0,
                        name: "out_color".to_string(),
                        format: StorageImageFormat::Rgba32Float,
                        access: Access::WriteOnly,
                        size: StorageTextureSize::Fixed {
                            width: 64,
                            height: 32,
                        },
                    }],
                    stages: vec![ComputeStage {
                        entry_point: "paint".to_string(),
                        workgroup_size: (8, 8, 1),
                        dispatch_size: DispatchSize::DerivedFrom {
                            len: DispatchLen::StorageImage { set: 1, binding: 0 },
                            workgroup_size: 8,
                        },
                        reads: vec![],
                        writes: vec![],
                    }],
                    default_total_threads: None,
                    feedback: vec![],
                }),
                Pipeline::Graphics(GraphicsPipeline {
                    stages: vec![GraphicsStage {
                        entry_point: "shade".to_string(),
                        stage: ShaderStage::Fragment,
                    }],
                    bindings: vec![Binding::Texture {
                        set: 2,
                        binding: 0,
                        name: "color_tex".to_string(),
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                        backing: Some(BackingRef { set: 1, binding: 0 }),
                    }],
                    vertex_inputs: vec![],
                    fragment_outputs: vec![],
                    feedback: vec![],
                }),
            ],
            frame_graph: FrameGraph::default(),
        };

        descriptor.rebuild_frame_graph();
        let graph = &descriptor.frame_graph;
        assert_eq!(graph.resources.len(), 1);
        assert_eq!(graph.resources[0].kind, FrameResourceKind::StorageTexture);
        assert_eq!(graph.resources[0].first_pass, Some(0));
        assert_eq!(graph.resources[0].last_pass, Some(1));
        assert!(matches!(
            graph.resources[0].extent.as_ref(),
            Some(FrameResourceExtent::StorageTexture { size })
                if *size == (StorageTextureSize::Fixed {
                    width: 64,
                    height: 32
                })
        ));
        assert_eq!(graph.passes.len(), 2);
        assert_eq!(graph.passes[1].depends_on, vec![0]);
    }

    #[test]
    fn frame_graph_marks_history_reads_without_self_dependency() {
        let mut descriptor = PipelineDescriptor {
            pipelines: vec![Pipeline::Compute(ComputePipeline {
                bindings: vec![
                    Binding::StorageTexture {
                        set: 1,
                        binding: 0,
                        name: "out_acc".to_string(),
                        format: StorageImageFormat::Rgba16Float,
                        access: Access::WriteOnly,
                        size: StorageTextureSize::SameAsWindow,
                    },
                    Binding::Texture {
                        set: 1,
                        binding: 1,
                        name: "prev_acc".to_string(),
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                        backing: Some(BackingRef { set: 1, binding: 0 }),
                    },
                ],
                stages: vec![ComputeStage {
                    entry_point: "step".to_string(),
                    workgroup_size: (8, 8, 1),
                    dispatch_size: DispatchSize::DerivedFrom {
                        len: DispatchLen::StorageImage { set: 1, binding: 0 },
                        workgroup_size: 8,
                    },
                    reads: vec![],
                    writes: vec![],
                }],
                default_total_threads: None,
                feedback: vec![FeedbackPair {
                    read_set: 1,
                    read_binding: 1,
                    write_set: 1,
                    write_binding: 0,
                }],
            })],
            frame_graph: FrameGraph::default(),
        };

        descriptor.rebuild_frame_graph();
        let graph = &descriptor.frame_graph;
        assert_eq!(graph.resources.len(), 1);
        assert_eq!(graph.feedback.len(), 1);
        assert_eq!(graph.feedback[0].read_resource, Some(0));
        assert_eq!(graph.feedback[0].write_resource, Some(0));
        assert!(graph.resources[0]
            .history
            .iter()
            .any(|role| role.role == FrameHistoryRoleKind::ReadPrevious));
        assert!(graph.resources[0]
            .history
            .iter()
            .any(|role| role.role == FrameHistoryRoleKind::WriteCurrent));
        assert_eq!(graph.passes[0].depends_on, Vec::<usize>::new());
        assert!(graph.passes[0].reads.iter().any(|access| access.role == FrameAccessRole::Previous));
    }

    #[test]
    fn vertex_attribute_serde_round_trip() {
        let attr = VertexAttribute {
            slot: 1,
            name: "color".to_string(),
            format: VertexFormat::Float32x3,
        };
        let json = serde_json::to_string(&attr).unwrap();
        // Format serializes snake_case.
        assert!(json.contains("\"float32x3\""), "got: {json}");
        let back: VertexAttribute = serde_json::from_str(&json).unwrap();
        assert_eq!(back.slot, 1);
        assert_eq!(back.name, "color");
        assert_eq!(back.format, VertexFormat::Float32x3);
    }
}
