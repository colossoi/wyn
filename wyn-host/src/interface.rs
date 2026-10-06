//! Published shader interfaces, resource identities, and execution domains.

use crate::{ResultKind, ResultLayout, ScalarExpr, ScalarSource, ScalarTask};
use std::collections::{BTreeMap, VecDeque};

/// Published shader declarations and physical resource interfaces for one module.
#[derive(Debug, Clone, Default)]
pub struct ModuleInterface {
    /// Scalar evaluations ordered at their consuming or replaced dispatch sites.
    pub scalar_tasks: Vec<ScalarTask>,
    pub dispatch_loops: Vec<DispatchLoop>,
    /// Individual pipelines in this program (one per top-level entry or multi-dispatch SOAC).
    pub pipelines: Vec<Pipeline>,
    /// Storage bindings that implement authored entry results. This preserves
    /// source-level result identity independently of generated binding names.
    pub source_results: Vec<SourceResultBinding>,
    /// Selected pass dependencies and explicit resource accesses, translated to
    /// published pipeline and binding indices for host execution.
    pub frame_graph: FrameGraph,
}

/// The descriptor binding that stores one top-level result of an authored
/// entry. `result` is the zero-based source result slot (a non-tuple return is
/// slot 0); `pipeline_index` locates the binding table containing `(set,
/// binding)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceResultBinding {
    pub entry: String,
    /// Authored record field, or entry name for a single unnamed result.
    pub name: String,
    pub kind: ResultKind,
    pub layout: ResultLayout,
    pub result: usize,
    pub pipeline_index: usize,
    pub set: u32,
    pub binding: u32,
}

#[derive(Debug, Clone)]
pub struct DispatchLoop {
    /// Logical carry length; empty initialization performs no device work.
    pub initial_length: ScalarExpr,
    /// Compute pipeline containing this loop and its carried bindings.
    pub pipeline: usize,
    /// Initialization dispatch, executed once even for zero iterations.
    pub setup: usize,
    /// Ordered dispatch indices executed on every iteration.
    pub body: Vec<usize>,
    /// Stage exposing the carried result and performing any required output copies.
    pub completion: usize,
    pub count: ScalarExpr,
    pub index: ScalarSource,
    pub current: ScalarSource,
    pub next: ScalarSource,
}

/// A descriptor-level frame graph: passes, logical resources, and the
/// selected dependencies and explicit resource accesses.
#[derive(Debug, Clone, Default)]
pub struct FrameGraph {
    pub passes: Vec<FramePass>,
    pub resources: Vec<FrameResource>,
    /// Draw passes and the command buffers that supply their indirect parameters.
    pub indirect_draws: Vec<IndirectDrawDependency>,
}

impl FrameGraph {
    pub fn is_empty(&self) -> bool {
        self.passes.is_empty() && self.resources.is_empty() && self.indirect_draws.is_empty()
    }

    /// An execution order satisfying every `depends_on`, or the passes that lie
    /// on or behind a cycle.
    ///
    /// A cycle means no order runs each pass after the ones it depends on — a
    /// producer and a consumer that each need the other's result within one
    /// frame. Declaration order is one solution whenever the graph is acyclic,
    /// but not the only one; passes with no path between them may overlap.
    pub fn topological_order(&self) -> Result<Vec<usize>, Vec<usize>> {
        self.order_units(&(0..self.passes.len()).collect::<Vec<_>>())
    }

    /// Read out selected loop bodies contiguously, keeping all dependencies.
    /// A cycle between a loop and an outer pass makes that grouping invalid.
    pub(crate) fn execution_order(&self, loops: &[DispatchLoop]) -> Result<Vec<usize>, String> {
        let mut order =
            self.topological_order().map_err(|cycle| format!("cyclic pass dependencies: {cycle:?}"))?;
        if loops.is_empty() {
            return Ok(order);
        }
        let indices: BTreeMap<_, _> = self
            .passes
            .iter()
            .enumerate()
            .map(|(index, pass)| ((pass.pipeline_index, pass.stage_index), index))
            .collect();
        let mut units: Vec<_> = (0..self.passes.len()).collect();
        let mut owners = vec![None; self.passes.len()];
        for (region, repeated) in loops.iter().enumerate() {
            let Some(&setup) = indices.get(&(repeated.pipeline, repeated.setup)) else {
                return Err("loop setup has no published pass".into());
            };
            for stage in std::iter::once(repeated.setup)
                .chain(repeated.body.iter().copied())
                .chain(std::iter::once(repeated.completion))
            {
                let Some(&index) = indices.get(&(repeated.pipeline, stage)) else {
                    return Err("loop stage has no published pass".into());
                };
                if let Some(previous) = owners[index].replace(region) {
                    return Err(format!(
                        "dispatch loop {region} reuses pass {index} already in loop {previous}"
                    ));
                }
                units[index] = setup;
            }
        }
        let grouped = self
            .order_units(&units)
            .map_err(|cycle| format!("cyclic dependencies across dispatch loop boundaries: {cycle:?}"))?;
        let mut position = vec![0; self.passes.len()];
        for (rank, unit) in grouped.into_iter().enumerate() {
            position[unit] = rank;
        }
        // Stable sorting retains the dependency order within each loop.
        order.sort_by_key(|&index| position[units[index]]);
        Ok(order)
    }

    fn order_units(&self, units: &[usize]) -> Result<Vec<usize>, Vec<usize>> {
        let mut remaining = vec![0; self.passes.len()];
        let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); self.passes.len()];
        for (index, pass) in self.passes.iter().enumerate() {
            for &dependency in &pass.depends_on {
                if index == dependency || units[index] != units[dependency] {
                    remaining[units[index]] += 1;
                    dependents[units[dependency]].push(units[index]);
                }
            }
        }

        let representatives: Vec<_> =
            (0..self.passes.len()).filter(|&index| units[index] == index).collect();
        let mut ready: VecDeque<usize> =
            representatives.iter().copied().filter(|&index| remaining[index] == 0).collect();
        let mut order = Vec::with_capacity(self.passes.len());
        while let Some(index) = ready.pop_front() {
            order.push(index);
            for &next in &dependents[index] {
                remaining[next] -= 1;
                if remaining[next] == 0 {
                    ready.push_back(next);
                }
            }
        }

        if order.len() == representatives.len() {
            Ok(order)
        } else {
            Err(representatives.into_iter().filter(|&index| remaining[index] > 0).collect())
        }
    }

    /// Translate published resources and explicit planning edges without hazard inference.
    pub fn from_selected_pipelines(
        pipelines: &[Pipeline],
        edges: &[((usize, usize), (usize, usize))],
    ) -> Result<Self, String> {
        let mut graph = Self::build(pipelines)?;
        let passes: BTreeMap<_, _> = graph
            .passes
            .iter()
            .enumerate()
            .map(|(i, pass)| ((pass.pipeline_index, pass.stage_index), i))
            .collect();
        for &(before, after) in edges {
            let (Some(&before), Some(&after)) = (passes.get(&before), passes.get(&after)) else {
                return Err("selected dependency has no published pass".into());
            };
            if before != after {
                graph.passes[after].depends_on.push(before);
            }
        }
        for pass in &mut graph.passes {
            pass.depends_on.sort_unstable();
            pass.depends_on.dedup();
        }
        Ok(graph)
    }

    /// Refresh resource indices after an ABI conversion, retaining every selected edge.
    pub fn refresh_resources(&self, pipelines: &[Pipeline]) -> Result<Self, String> {
        let mut edges = Vec::new();
        for after in &self.passes {
            for &before in &after.depends_on {
                let Some(before) = self.passes.get(before) else {
                    return Err("invalid selected pass dependency".into());
                };
                edges.push((
                    (before.pipeline_index, before.stage_index),
                    (after.pipeline_index, after.stage_index),
                ));
            }
        }
        Self::from_selected_pipelines(pipelines, &edges)
    }

    fn build(pipelines: &[Pipeline]) -> Result<Self, String> {
        let mut builder = FrameGraphBuilder::default();

        for (pipeline_index, pipeline) in pipelines.iter().enumerate() {
            for (binding_index, binding) in pipeline_bindings(pipeline).iter().enumerate() {
                builder.ensure_binding(pipeline_index, binding_index, binding);
            }
        }

        for (pipeline_index, pipeline) in pipelines.iter().enumerate() {
            match pipeline {
                Pipeline::Compute(compute) => {
                    for (stage_index, stage) in compute.stages.iter().enumerate() {
                        let accesses =
                            builder.stage_accesses(pipeline_index, &compute.bindings, &stage.uses)?;
                        builder.push_pass(
                            FramePassKind::Compute,
                            stage.entry_point.clone(),
                            pipeline_index,
                            stage_index,
                            accesses.reads,
                            accesses.writes,
                        );
                    }
                }
                Pipeline::Graphics(graphics) => {
                    let Some(stage) = graphics.stages.first() else {
                        continue;
                    };
                    // A draw is one scheduling operation. Its attachment writes
                    // become available only after all shader inputs are ready.
                    let mut writes: Vec<FrameAccess> = graphics
                        .fragment_outputs
                        .iter()
                        .map(|output| FrameAccess {
                            resource: builder.ensure_named(FrameResourceKind::Texture, &output.name),
                        })
                        .collect();
                    let mut reads = Vec::new();
                    for stage in &graphics.stages {
                        let accesses =
                            builder.stage_accesses(pipeline_index, &graphics.bindings, &stage.uses)?;
                        for access in accesses.reads {
                            push_unique_access(&mut reads, access);
                        }
                        for access in accesses.writes {
                            push_unique_access(&mut writes, access);
                        }
                    }
                    for attribute in &graphics.vertex_inputs {
                        let resource =
                            builder.ensure_named(FrameResourceKind::StorageBuffer, &attribute.name);
                        push_unique_access(&mut reads, FrameAccess { resource });
                    }
                    let indirect_resource = graphics.invocation.draw.indirect_commands().map(|buffer| {
                        builder.ensure_named(FrameResourceKind::StorageBuffer, buffer.frame_name())
                    });
                    let index_resource = graphics.invocation.draw.indices().map(|buffer| {
                        builder.ensure_named(FrameResourceKind::StorageBuffer, buffer.frame_name())
                    });
                    for resource in [indirect_resource, index_resource].into_iter().flatten() {
                        push_unique_access(&mut reads, FrameAccess { resource });
                    }
                    builder.push_pass(
                        FramePassKind::Draw,
                        stage.entry_point.clone(),
                        pipeline_index,
                        0,
                        reads,
                        writes,
                    );
                    if let Some(buffer_resource) = indirect_resource {
                        builder.graph.indirect_draws.push(IndirectDrawDependency {
                            draw_pass: builder.graph.passes.len() - 1,
                            buffer_resource,
                        });
                    }
                }
            }
        }

        Ok(builder.graph)
    }
}

#[derive(Debug, Clone)]
pub struct FramePass {
    pub name: String,
    pub kind: FramePassKind,
    pub pipeline_index: usize,
    /// Compute-stage index; zero for a draw containing all graphics stages.
    pub stage_index: usize,
    pub reads: Vec<FrameAccess>,
    pub writes: Vec<FrameAccess>,
    pub depends_on: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FramePassKind {
    Compute,
    Draw,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameAccess {
    pub resource: usize,
}

#[derive(Debug, Clone)]
pub struct FrameResource {
    pub name: String,
    pub kind: FrameResourceKind,
    pub bindings: Vec<FrameBindingRef>,
    pub extent: Option<FrameResourceExtent>,
    pub first_pass: Option<usize>,
    pub last_pass: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum FrameResourceKind {
    StorageBuffer,
    Uniform,
    PushConstant,
    Texture,
    Sampler,
    StorageTexture,
}

#[derive(Debug, Clone)]
pub struct FrameBindingRef {
    pub pipeline_index: usize,
    pub binding_index: usize,
    pub name: String,
    pub kind: FrameResourceKind,
    pub set: Option<u32>,
    pub binding: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrameResourceExtent {
    StorageTexture {
        size: StorageTextureSize,
    },
    StorageBuffer {
        length: Option<BufferLen>,
    },
    Uniform {
        bytes: u32,
    },
    PushConstant {
        bytes: u32,
    },
}

#[derive(Debug, Clone, Copy)]
pub struct IndirectDrawDependency {
    pub draw_pass: usize,
    pub buffer_resource: usize,
}

/// A single pipeline within the program.
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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
    pub default_total_threads: Option<std::num::NonZeroU32>,
}

/// The `(set, binding)` of a `StorageTexture` allocation that a sampled
/// `Texture` binding is a view of. See `Binding::Texture::backing`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackingRef {
    pub set: u32,
    pub binding: u32,
}

/// Per-stage uses of a pipeline's binding table. Binding declarations describe
/// slots and their attached resources; this records how one shader entry point
/// accesses those slots. Indices address the parent pipeline's `bindings`.
#[derive(Debug, Clone, Default)]
pub struct StageBindingUses {
    pub reads: Vec<usize>,
    pub writes: Vec<usize>,
}

impl StageBindingUses {
    pub fn record(&mut self, binding: usize, access: Access) {
        if matches!(access, Access::ReadOnly | Access::ReadWrite) && !self.reads.contains(&binding) {
            self.reads.push(binding);
        }
        if matches!(access, Access::WriteOnly | Access::ReadWrite) && !self.writes.contains(&binding) {
            self.writes.push(binding);
        }
    }

    pub fn access(&self, binding: usize) -> Option<Access> {
        match (self.reads.contains(&binding), self.writes.contains(&binding)) {
            (true, true) => Some(Access::ReadWrite),
            (true, false) => Some(Access::ReadOnly),
            (false, true) => Some(Access::WriteOnly),
            (false, false) => None,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.reads.is_empty() && self.writes.is_empty()
    }
}

/// A single dispatch stage within a `ComputePipeline`.
#[derive(Debug, Clone)]
pub struct ComputeStage {
    /// Required predecessor stages, identified by (pipeline index, stage index).
    /// Includes control dependencies that cannot be recovered from buffer access.
    pub dependencies: Vec<(usize, usize)>,
    pub entry_point: String,
    /// Authored entry whose execution this stage implements. Generated stages
    /// retain their source owner instead of asking runtimes to infer it from
    /// the backend entry-point name.
    pub owner: String,
    pub workgroup_size: (u32, u32, u32),
    pub dispatch_size: DispatchSize,
    pub uses: StageBindingUses,
}

impl std::ops::Deref for ComputeStage {
    type Target = StageBindingUses;

    fn deref(&self) -> &Self::Target {
        &self.uses
    }
}

impl std::ops::DerefMut for ComputeStage {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.uses
    }
}

/// Graphics pipeline (vertex + fragment stages).
#[derive(Debug, Clone)]
pub struct GraphicsPipeline {
    /// Authored graphics-operation ordinal within the stages' source owner.
    /// Independent of pipeline ordering and generated shader entry-point names.
    pub source_operation: Option<u32>,
    pub stages: Vec<GraphicsStage>,
    /// Primitive assembly and draw request selected by the unified invocation.
    pub invocation: GraphicsInvocation,
    pub bindings: Vec<Binding>,
    pub vertex_inputs: Vec<VertexAttribute>,
    pub fragment_outputs: Vec<FragmentOutput>,
}

/// The source-level rasterization request associated with one graphics pipeline.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphicsInvocation {
    pub topology: PrimitiveTopology,
    pub draw: DrawCall,
    pub raster_state: RasterState,
    pub fragment_state: FragmentState,
}

impl Default for GraphicsInvocation {
    fn default() -> Self {
        Self {
            topology: PrimitiveTopology::TriangleList,
            draw: DrawCall::Direct {
                vertex_count: 3,
                instance_count: 1,
                first_vertex: 0,
                first_instance: 0,
            },
            raster_state: RasterState::default(),
            fragment_state: FragmentState::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RasterState {
    pub viewport: Viewport,
    pub scissor: Scissor,
    pub front_face: FrontFace,
    pub cull: CullMode,
    pub fill: FillMode,
}

impl Default for RasterState {
    fn default() -> Self {
        Self {
            viewport: Viewport::Target,
            scissor: Scissor::Target,
            front_face: FrontFace::CounterClockwise,
            cull: CullMode::None,
            fill: FillMode::Fill,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Viewport {
    Target,
    Custom {
        origin: [f32; 2],
        extent: [f32; 2],
        depth: [f32; 2],
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scissor {
    Target,
    Custom {
        origin: [i32; 2],
        extent: [u32; 2],
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrontFace {
    Clockwise,
    CounterClockwise,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CullMode {
    None,
    Front,
    Back,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FillMode {
    Fill,
    Line,
    Point,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FragmentState {
    pub depth_test: DepthTest,
    pub depth_write: bool,
    pub blend: BlendMode,
    pub color_write: bool,
}

impl Default for FragmentState {
    fn default() -> Self {
        Self {
            depth_test: DepthTest::Disabled,
            depth_write: false,
            blend: BlendMode::Replace,
            color_write: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepthTest {
    Disabled,
    Never,
    Less,
    LessEqual,
    Equal,
    GreaterEqual,
    Greater,
    Always,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMode {
    Replace,
    SourceOver,
    Add,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimitiveTopology {
    TriangleList,
    TriangleStrip,
    LineList,
    LineStrip,
    PointList,
}

/// A descriptor-visible reference to an array buffer consumed by draw execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DrawBufferRef {
    pub set: u32,
    pub binding: u32,
    pub name: String,
    pub resource: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexFormat {
    Uint16,
    Uint32,
}

/// The number of index elements or indirect commands consumed by a draw.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrawCount {
    Fixed(u32),
    /// Use the logical element count of the referenced array, not its allocation capacity.
    BufferLength,
}

impl Default for DrawCount {
    fn default() -> Self {
        Self::Fixed(1)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DrawCall {
    Direct {
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    },
    Indexed {
        indices: DrawBufferRef,
        index_format: IndexFormat,
        index_count: DrawCount,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    },
    Indirect {
        commands: DrawBufferRef,
        offset: u64,
        draw_count: DrawCount,
    },
    IndexedIndirect {
        indices: DrawBufferRef,
        index_format: IndexFormat,
        commands: DrawBufferRef,
        offset: u64,
        draw_count: DrawCount,
    },
}

impl DrawBufferRef {
    pub fn frame_name(&self) -> &str {
        self.resource.as_deref().unwrap_or(&self.name)
    }
}

impl DrawCall {
    pub fn indirect_commands(&self) -> Option<&DrawBufferRef> {
        match self {
            Self::Indirect { commands, .. } | Self::IndexedIndirect { commands, .. } => Some(commands),
            Self::Direct { .. } | Self::Indexed { .. } => None,
        }
    }

    pub fn indices(&self) -> Option<&DrawBufferRef> {
        match self {
            Self::Indexed { indices, .. } | Self::IndexedIndirect { indices, .. } => Some(indices),
            Self::Direct { .. } | Self::Indirect { .. } => None,
        }
    }
}

/// A stage in a graphics pipeline.
#[derive(Debug, Clone)]
pub struct GraphicsStage {
    pub entry_point: String,
    /// Authored entry whose execution this stage implements.
    pub owner: String,
    pub stage: ShaderStage,
    pub uses: StageBindingUses,
}

impl std::ops::Deref for GraphicsStage {
    type Target = StageBindingUses;

    fn deref(&self) -> &Self::Target {
        &self.uses
    }
}

impl std::ops::DerefMut for GraphicsStage {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.uses
    }
}

/// Shader stage type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShaderStage {
    Vertex,
    Fragment,
}

/// How to determine the compute dispatch grid size.
#[derive(Debug, Clone, PartialEq, Eq)]
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
        explicit: bool,
    },
    /// Dispatch `ceil(len / workgroup_size)` workgroups, where `len` is the
    /// launch bound resolved from the explicit `DispatchLen` source.
    DerivedFrom {
        len: DispatchLen,
        workgroup_size: u32,
    },
}

/// The source of a `DerivedFrom` dispatch's launch bound. Grid-stride kernels
/// enforce their logical iteration count independently of this bound.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DispatchLen {
    /// Launch from the capacity of the buffer at (`set`, `binding`) — e.g.
    /// the input to `map(f, arr)`, or an output covering a uniform-sized domain.
    /// The host uses the buffer's byte size without reading its contents.
    InputBinding {
        set: u32,
        binding: u32,
        /// Bytes per element of that buffer, so the host recovers the element
        /// count from its byte size.
        elem_bytes: u32,
    },
    /// A compile-time-known iteration count — e.g. `map(f, iota(6144))`.
    Fixed {
        count: u32,
    },
    /// A runtime count read from a scalar push-constant — e.g. `map(f,
    /// iota(n))` where `n` is an entry parameter. The host reads the u32 at
    /// `offset` in the push-constant block.
    PushConstant {
        offset: u32,
    },
    /// A runtime u32 read from a packed, read-only storage parameter block.
    /// WGSL/WebGPU uses this in place of `PushConstant`.
    StorageBuffer {
        set: u32,
        binding: u32,
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

/// One named member of a host-populated interface block: where the host writes
/// the value. Uniform buffers always publish these members. A storage buffer
/// publishes them when the WGSL backend uses it as the WebGPU replacement for
/// an entry point's packed push-constant block.
#[derive(Debug, Clone)]
pub struct UniformMember {
    pub name: String,
    pub offset: u32,
    pub size: u32,
}

/// A GPU resource binding used by the pipeline.
#[derive(Debug, Clone)]
pub enum Binding {
    /// Storage buffer (descriptor set binding).
    StorageBuffer {
        set: u32,
        binding: u32,
        access: Access,
        usage: BufferUsage,
        /// Entry-local binding name used for diagnostics and host handles.
        name: String,
        /// Logical frame-graph resource viewed through this binding. Producer
        /// and consumer bindings may have different local names and slots.
        resource: Option<String>,
        /// Sizing policy for compiler-managed buffers and fixed-size WGSL
        /// parameter blocks. `None` for variable host inputs (sized from the
        /// supplied data) and ordinary unsized outputs.
        length: Option<BufferLen>,
        /// Named fields when this storage binding represents a packed
        /// host-populated parameter block. Ordinary array buffers leave this
        /// empty.
        members: Vec<UniformMember>,
    },
    /// Uniform buffer (descriptor set binding).
    Uniform {
        set: u32,
        binding: u32,
        name: String,
        /// Published std140 block size in bytes.
        size: u32,
        /// Flattened block members in declaration order — the record
        /// fields of a record-typed uniform. Empty when unpublished.
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
    /// slot to that allocation's sampled view. `None` is a host-provided /
    /// external texture.
    Texture {
        set: u32,
        binding: u32,
        name: String,
        sample_type: TextureSampleType,
        view_dimension: TextureViewDimension,
        multisampled: bool,
        backing: Option<BackingRef>,
        /// The logical render-target resource this samples, from
        /// `#[view(name, sampled)]` on a resource with no storage backing. It
        /// is the frame-graph identity — a fragment `#[target(name)]` write and
        /// this read share the resource `name`, so a producer→consumer edge
        /// forms. `None` for a host-provided texture or a storage-backed view.
        resource: Option<String>,
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
    /// param of type `storage_image`. A sampled view of the same allocation
    /// is represented as a separate `Texture` descriptor slot whose
    /// `backing` points at this storage texture binding.
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
        size: StorageTextureSize,
        /// The logical `resource` this storage view accesses, from
        /// `#[view(name, storage_read|storage_write)]`. Frame-graph identity:
        /// storage views, sampled views, and a fragment `#[target(name)]` of the
        /// same resource collapse to one texture-kind resource keyed by `name`.
        resource: Option<String>,
    },
}

/// Resolution policy for a storage texture's backing `wgpu::Texture`.
/// Resolved by the host at allocation time (and on window resize for
/// `SameAsWindow`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
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
    /// Descriptor-set slot for resources represented by `(set, binding)`.
    /// Push constants are addressed by byte range and therefore have no slot.
    pub fn slot(&self) -> Option<(u32, u32)> {
        match self {
            Binding::StorageBuffer { set, binding, .. }
            | Binding::Uniform { set, binding, .. }
            | Binding::Texture { set, binding, .. }
            | Binding::Sampler { set, binding, .. }
            | Binding::StorageTexture { set, binding, .. } => Some((*set, *binding)),
            Binding::PushConstant { .. } => None,
        }
    }

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

/// Identity of a frame-graph resource. A resource is a *logical* buffer/texture,
/// distinct from any one pipeline's `(set, binding)` contract for it: the same
/// buffer bound by several pipelines (at possibly different slots) is one
/// resource. Identity is the logical name for the kinds that are shared across
/// pipelines (storage buffers, sampled textures); the physical slot is retained
/// only where it *is* the identity — a sampled view aliasing a compute
/// storage-texture keys to that backing slot, and push constants are inherently
/// per-pipeline.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum ResourceKey {
    Named {
        kind: FrameResourceKind,
        name: String,
    },
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

/// Explicit frame resource accesses of one selected stage.
#[derive(Default)]
struct StageAccesses {
    reads: Vec<FrameAccess>,
    writes: Vec<FrameAccess>,
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
            let kind = resource_kind_from_key(&key);
            self.resources.insert(key, index);
            self.graph.resources.push(FrameResource {
                name: resource_name(binding).to_string(),
                kind,
                bindings: Vec::new(),
                extent: binding_extent(binding),
                first_pass: None,
                last_pass: None,
            });
            index
        };

        let resource = &mut self.graph.resources[index];
        // Bindings merging onto one resource are normally the same kind. Two
        // cross-kind cases exist: a sampled texture view aliasing its
        // storage-texture backing (keyed to that StorageTexture slot), and the
        // views of one named `resource` — storage and sampled reads plus a
        // fragment `#[target]` write — which collapse to a Texture-kind resource.
        let named_resource = matches!(
            binding,
            Binding::Texture {
                resource: Some(_),
                ..
            } | Binding::StorageTexture {
                resource: Some(_),
                ..
            }
        );
        debug_assert!(
            binding_kind(binding) == resource.kind
                || (matches!(binding, Binding::Texture { backing: Some(_), .. })
                    && resource.kind == FrameResourceKind::StorageTexture)
                || (named_resource && resource.kind == FrameResourceKind::Texture),
            "binding kind {:?} disagrees with merged resource kind {:?} at one slot",
            binding_kind(binding),
            resource.kind
        );
        merge_extent(&mut resource.extent, binding_extent(binding));
        // A storage-texture write names the resource (a sampled reader that
        // merged first may have carried a stale placeholder name); the logical
        // resource name wins over a local param name.
        if matches!(binding, Binding::StorageTexture { .. }) {
            resource.name = resource_name(binding).to_string();
        }

        let binding_ref = FrameBindingRef {
            pipeline_index,
            binding_index,
            name: binding_name(binding).to_string(),
            kind: binding_kind(binding),
            set: binding.slot().map(|(set, _)| set),
            binding: binding.slot().map(|(_, binding)| binding),
        };
        if !resource.bindings.iter().any(|existing| {
            existing.pipeline_index == pipeline_index && existing.binding_index == binding_index
        }) {
            resource.bindings.push(binding_ref);
        }
        index
    }

    /// Ensure a resource that has no descriptor binding — a fragment render
    /// target, identified by logical name. If a binding of the same name and
    /// kind already created the resource (a downstream pass sampling it), this
    /// returns that same resource, so the render write and the sampled read
    /// share one identity and a producer→consumer edge forms.
    fn ensure_named(&mut self, kind: FrameResourceKind, name: &str) -> usize {
        let key = ResourceKey::Named {
            kind,
            name: name.to_string(),
        };
        if let Some(&index) = self.resources.get(&key) {
            return index;
        }
        let index = self.graph.resources.len();
        self.resources.insert(key, index);
        self.graph.resources.push(FrameResource {
            name: name.to_string(),
            kind,
            bindings: Vec::new(),
            extent: None,
            first_pass: None,
            last_pass: None,
        });
        index
    }

    fn stage_accesses(
        &mut self,
        pipeline_index: usize,
        bindings: &[Binding],
        uses: &StageBindingUses,
    ) -> Result<StageAccesses, String> {
        let mut accesses = StageAccesses::default();
        for (indices, output) in [
            (&uses.reads, &mut accesses.reads),
            (&uses.writes, &mut accesses.writes),
        ] {
            for &index in indices {
                let Some(binding) = bindings.get(index) else {
                    return Err("selected access has no descriptor binding".into());
                };
                let resource = self.ensure_binding(pipeline_index, index, binding);
                push_unique_access(output, FrameAccess { resource });
            }
        }
        Ok(accesses)
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
    ) {
        let pass_index = self.graph.passes.len();
        for access in reads.iter().chain(&writes) {
            let resource = &mut self.graph.resources[access.resource];
            resource.first_pass.get_or_insert(pass_index);
            resource.last_pass = Some(pass_index);
        }
        self.graph.passes.push(FramePass {
            name,
            kind,
            pipeline_index,
            stage_index,
            reads,
            writes,
            depends_on: Vec::new(),
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
        // A storage buffer is one logical resource across every pipeline that
        // binds it, regardless of each pipeline's `(set, binding)` slot.
        Binding::StorageBuffer { name, resource, .. } => ResourceKey::Named {
            kind: FrameResourceKind::StorageBuffer,
            name: resource.clone().unwrap_or_else(|| name.clone()),
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
            name,
            backing,
            resource,
            ..
        } => {
            // A view of a named `resource` keys to that name, sharing identity
            // with the resource's storage views and with the fragment
            // `#[target(name)]` that writes it — a sampled view records the
            // storage allocation it aliases in `backing`, but the name is the
            // identity. A view carrying only a `backing` keys to that storage
            // slot. A plain sampled texture is one logical resource by name
            // across its readers.
            match (resource, backing) {
                (Some(resource), _) => ResourceKey::Named {
                    kind: FrameResourceKind::Texture,
                    name: resource.clone(),
                },
                (None, Some(backing)) => ResourceKey::Descriptor {
                    kind: FrameResourceKind::StorageTexture,
                    set: backing.set,
                    binding: backing.binding,
                },
                (None, None) => ResourceKey::Named {
                    kind: FrameResourceKind::Texture,
                    name: name.clone(),
                },
            }
        }
        Binding::Sampler { set, binding, .. } => ResourceKey::Descriptor {
            kind: FrameResourceKind::Sampler,
            set: *set,
            binding: *binding,
        },
        // A storage view of a named `resource` shares identity with that
        // resource's other views and its `#[target]` write — one texture-kind
        // resource keyed by name. A compute-only storage texture keys by slot.
        Binding::StorageTexture {
            set,
            binding,
            resource,
            ..
        } => match resource {
            Some(resource) => ResourceKey::Named {
                kind: FrameResourceKind::Texture,
                name: resource.clone(),
            },
            None => ResourceKey::Descriptor {
                kind: FrameResourceKind::StorageTexture,
                set: *set,
                binding: *binding,
            },
        },
    }
}

fn resource_kind_from_key(key: &ResourceKey) -> FrameResourceKind {
    match key {
        ResourceKey::Named { kind, .. } | ResourceKey::Descriptor { kind, .. } => *kind,
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

/// The binding's logical resource name — its frame-graph identity. A texture
/// viewing a render-target `resource` reports that resource's name (shared with
/// the fragment `#[target(name)]` write and any other reader), not its local
/// param name. Everything else reports its own name.
fn resource_name(binding: &Binding) -> &str {
    match binding {
        Binding::StorageBuffer {
            resource: Some(resource),
            ..
        }
        | Binding::Texture {
            resource: Some(resource),
            ..
        }
        | Binding::StorageTexture {
            resource: Some(resource),
            ..
        } => resource,
        _ => binding_name(binding),
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
        Some(FrameResourceExtent::StorageBuffer { length }) => match target {
            None => *target = Some(FrameResourceExtent::StorageBuffer { length }),
            Some(FrameResourceExtent::StorageBuffer { length: existing }) => match (&*existing, &length) {
                (None, Some(_)) => *existing = length,
                (Some(left), Some(right)) => {
                    debug_assert_eq!(left, right, "two storage-buffer aliases disagree on length")
                }
                _ => {}
            },
            Some(existing) => debug_assert!(
                matches!(existing, FrameResourceExtent::StorageBuffer { .. }),
                "two bindings merging at one slot disagree on extent kind"
            ),
        },
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

fn push_unique_access(accesses: &mut Vec<FrameAccess>, access: FrameAccess) {
    if !accesses.contains(&access) {
        accesses.push(access);
    }
}

/// Allocation policy for a storage buffer, including its final host expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BufferLen {
    /// Compute an allocation capacity from the published host expression.
    Computed {
        bytes: crate::Expr,
    },
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
    /// One element per point in the producer's logical dispatch domain,
    /// before workgroup rounding. Byte size = `logical_count * elem_bytes`.
    /// A fixed physical workgroup grid alone cannot resolve this policy.
    /// The host needs the domain metadata, not the padded invocation count.
    /// This resolves via `dispatch_elem_bytes`, not `resolve_bytes`.
    SameAsDispatch {
        elem_bytes: u32,
    },
}

impl BufferLen {
    pub fn reads_mut(&mut self, visit: &mut impl FnMut(&mut ScalarSource, &mut u32)) {
        if let Self::Computed { bytes } = self {
            bytes.reads_mut(visit);
        }
    }
    /// Resolve to a byte size given a lookup of already-allocated buffers'
    /// byte sizes by (set, binding). Returns `None` if a referenced source
    /// buffer hasn't been sized yet, or when a separate context is needed:
    /// `SameAsDispatch` uses `dispatch_elem_bytes`; `Computed` needs scalar evaluation.
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
            BufferLen::Computed { .. } | BufferLen::SameAsDispatch { .. } => None,
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Access {
    ReadOnly,
    WriteOnly,
    ReadWrite,
}

/// How a buffer is used in the pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
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
/// `Float`; filtering is required only when the shader samples the texture
/// through a sampler, rather than loading texels at integer coordinates.
#[derive(Debug, Clone, PartialEq)]
pub enum TextureSampleType {
    Float {
        filterable: bool,
    },
    Sint,
    Uint,
    Depth,
}

/// View dimension of a texture binding. v1 always emits `D2`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TextureViewDimension {
    D1,
    D2,
    D2Array,
    Cube,
    CubeArray,
    D3,
}

/// Sampler binding mode. v1 always emits `Filtering`.
#[derive(Debug, Clone, PartialEq, Eq)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageImageFormat {
    Rgba8Unorm,
    Rgba16Float,
    Rgba32Float,
    R32Float,
}

/// Scalar/vector format of a vertex-buffer attribute. Mirrors the
/// wgpu `VertexFormat` subset Wyn can currently produce — 32-bit
/// float / signed-int / unsigned-int scalars and 2-4 wide vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
#[derive(Debug, Clone)]
pub struct VertexAttribute {
    pub slot: u32,
    pub name: String,
    pub format: VertexFormat,
}

/// Fragment output.
#[derive(Debug, Clone)]
pub struct FragmentOutput {
    pub location: u32,
    pub name: String,
}

#[cfg(test)]
#[path = "interface_tests.rs"]
mod interface_tests;
