use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

use crate::{
    Binding, BufferLen, BufferUsage, DepthTest, DispatchSize, DrawBufferRef, DrawCall, DrawCount,
    FrameResource, FrameResourceKind, ModuleInterface, Pipeline, ScalarExpr, ScalarSource,
    StorageTextureSize,
};

#[derive(Debug, Error)]
pub enum HostError {
    #[error("invalid host program: {0}")]
    Invalid(String),
    #[error("Rust host generation: {0}")]
    Rust(#[from] syn::Error),
    #[error("host output formatting: {0}")]
    Format(#[from] std::fmt::Error),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShaderFormat {
    Spirv,
    Wgsl,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ResourceId(pub usize);

/// Checked i64 capacity arithmetic, with explicit wrapping i32/u32 source operations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Expr {
    /// Source arithmetic retains its typed host operations through code generation.
    Scalar(ScalarExpr),
    /// Capacity from a physical buffer, without reading or narrowing its contents.
    BufferLength {
        source: ScalarSource,
        stride: u32,
    },
    Integer(i64),
    Input(String),
    BufferSize(ResourceId),
    ReadScalar {
        resource: ResourceId,
        offset: u32,
        signed: bool,
    },
    TextureDimension {
        source: ScalarSource,
        axis: usize,
    },
    Subtract(Box<Expr>, Box<Expr>),
    Multiply(Box<Expr>, Box<Expr>),
    Floor(Box<Expr>, Box<Expr>),
    Ceiling(Box<Expr>, Box<Expr>),
    Min(Box<Expr>, Box<Expr>),
    Max(Box<Expr>, Box<Expr>),
}

/// Resource references and host arguments collected together while walking an
/// expression. Zero CPU bytes means that only the resource itself is needed.
#[derive(Default)]
struct Dependencies {
    resources: BTreeSet<ResourceId>,
    host_inputs: BTreeMap<ResourceId, u64>,
    scalar_inputs: BTreeSet<String>,
}

impl Dependencies {
    fn resource(&mut self, id: ResourceId, host_bytes: u64) {
        self.resources.insert(id);
        if host_bytes > 0 {
            self.host_inputs
                .entry(id)
                .and_modify(|bytes| *bytes = (*bytes).max(host_bytes))
                .or_insert(host_bytes);
        }
    }

    fn extend(&mut self, other: Self) {
        self.resources.extend(other.resources);
        self.scalar_inputs.extend(other.scalar_inputs);
        for (id, bytes) in other.host_inputs {
            self.resource(id, bytes);
        }
    }

    fn scalar(
        &mut self,
        value: &ScalarExpr,
        resolve: impl Fn(&ScalarSource) -> Result<ResourceId, HostError>,
    ) -> Result<(), HostError> {
        let mut sources = Vec::new();
        value.visit(&mut |value| match value {
            ScalarExpr::Parameter { source, offset, .. } => sources.push((source, u64::from(*offset) + 4)),
            ScalarExpr::Read { source, .. } | ScalarExpr::BufferLength { source, .. } => {
                sources.push((source, 0))
            }
            _ => {}
        });
        for (source, bytes) in sources {
            self.resource(resolve(source)?, bytes);
        }
        Ok(())
    }
}

impl Expr {
    fn dependencies(&self, result: &mut Dependencies) -> Result<(), HostError> {
        match self {
            Self::Scalar(value) => result.scalar(value, ScalarSource::resource)?,
            Self::BufferLength { source, .. } | Self::TextureDimension { source, .. } => {
                result.resource(source.resource()?, 0)
            }
            Self::BufferSize(id) | Self::ReadScalar { resource: id, .. } => result.resource(*id, 0),
            Self::Input(name) => {
                result.scalar_inputs.insert(name.clone());
            }
            Self::Subtract(a, b)
            | Self::Multiply(a, b)
            | Self::Floor(a, b)
            | Self::Ceiling(a, b)
            | Self::Min(a, b)
            | Self::Max(a, b) => {
                a.dependencies(result)?;
                b.dependencies(result)?;
            }
            Self::Integer(_) => {}
        }
        Ok(())
    }

    pub fn multiply(self, rhs: Self) -> Self {
        Self::Multiply(Box::new(self), Box::new(rhs))
    }

    pub fn floor(self, divisor: u32) -> Result<Self, HostError> {
        if divisor == 0 {
            return Err(HostError::Invalid("zero element stride".into()));
        }
        Ok(Self::Floor(
            Box::new(self),
            Box::new(Self::Integer(divisor.into())),
        ))
    }

    pub fn ceiling(self, divisor: u32) -> Result<Self, HostError> {
        if divisor == 0 {
            return Err(HostError::Invalid("zero workgroup dimension".into()));
        }
        Ok(Self::Ceiling(
            Box::new(self),
            Box::new(Self::Integer(divisor.into())),
        ))
    }

    pub fn reads_mut(&mut self, visit: &mut impl FnMut(&mut ScalarSource, &mut u32)) {
        match self {
            Self::Scalar(value) => value.reads_mut(visit),
            Self::BufferLength { source, .. } | Self::TextureDimension { source, .. } => {
                visit(source, &mut 0)
            }
            Self::Subtract(left, right)
            | Self::Multiply(left, right)
            | Self::Floor(left, right)
            | Self::Ceiling(left, right)
            | Self::Min(left, right)
            | Self::Max(left, right) => {
                left.reads_mut(visit);
                right.reads_mut(visit);
            }
            _ => {}
        }
    }
}

#[derive(Clone, Debug)]
pub enum Allocation {
    Buffer {
        resource: ResourceId,
        bytes: Expr,
    },
    Texture {
        resource: ResourceId,
        width: Expr,
        height: Expr,
    },
}

impl Allocation {
    pub fn resource(&self) -> ResourceId {
        match self {
            Self::Buffer { resource, .. } | Self::Texture { resource, .. } => *resource,
        }
    }

    fn expressions(&self) -> Vec<&Expr> {
        match self {
            Self::Buffer { bytes, .. } => vec![bytes],
            Self::Texture { width, height, .. } => vec![width, height],
        }
    }
}

#[derive(Clone, Debug)]
pub enum Operation {
    Loop {
        setup: Vec<Operation>,
        pipeline: usize,
        region: usize,
        body: Vec<Operation>,
    },
    Scalar {
        pipeline: usize,
        task: usize,
    },
    Dispatch {
        pipeline: usize,
        stage: usize,
        groups: [Expr; 3],
    },
    Draw {
        pipeline: usize,
    },
}

#[derive(Clone, Debug)]
pub struct Entry {
    pub name: String,
    pub inputs: BTreeSet<ResourceId>,
    /// CPU parameter dependencies and their minimum byte spans. Includes push
    /// constants; buffer-backed parameters additionally retain a GPU binding.
    /// Derived from host expressions, independently of either output language.
    pub host_inputs: BTreeMap<ResourceId, u64>,
    pub scalar_inputs: BTreeSet<String>,
    pub allocations: Vec<Allocation>,
    pub operations: Vec<Operation>,
    pub results: Vec<ResourceId>,
}

/// Shader declarations and the host functions that orchestrate them.
#[derive(Clone, Debug)]
pub struct Program {
    pub interface: ModuleInterface,
    pub entries: Vec<Entry>,
    pub(crate) depth_targets: BTreeMap<usize, ResourceId>,
}

impl Program {
    /// Build host functions from published shader interfaces and resource dependencies.
    pub fn new(mut interface: ModuleInterface) -> Result<Self, HostError> {
        let expected_passes: usize = interface
            .pipelines
            .iter()
            .map(|pipeline| match pipeline {
                Pipeline::Compute(pipeline) => pipeline.stages.len(),
                Pipeline::Graphics(pipeline) => usize::from(!pipeline.stages.is_empty()),
            })
            .sum();
        if interface.frame_graph.passes.len() != expected_passes {
            return Err(HostError::Invalid(
                "shader interface has no complete selected frame graph".into(),
            ));
        }
        let mut depth_targets = BTreeMap::new();
        for (p, pipeline) in interface.pipelines.iter().enumerate() {
            if let Pipeline::Graphics(g) = pipeline {
                for attribute in &g.vertex_inputs {
                    if !interface.frame_graph.resources.iter().any(|r| r.name == attribute.name) {
                        interface.frame_graph.resources.push(FrameResource {
                            name: attribute.name.clone(),
                            kind: FrameResourceKind::StorageBuffer,
                            bindings: vec![],
                            extent: None,
                            first_pass: None,
                            last_pass: None,
                        });
                    }
                }
                if g.invocation.fragment_state.depth_test != DepthTest::Disabled {
                    depth_targets.insert(p, ResourceId(interface.frame_graph.resources.len()));
                    interface.frame_graph.resources.push(FrameResource {
                        name: format!("depth-target-{p}"),
                        kind: FrameResourceKind::Texture,
                        bindings: vec![],
                        extent: None,
                        first_pass: None,
                        last_pass: None,
                    });
                }
            }
        }
        let mut program = Self {
            interface,
            entries: vec![],
            depth_targets,
        };
        let order = program
            .interface
            .frame_graph
            .execution_order(&program.interface.dispatch_loops)
            .map_err(HostError::Invalid)?;
        let mut entries = BTreeMap::<String, Entry>::new();
        let mut stage_operations = BTreeMap::new();
        let mut entry_stages = BTreeMap::<String, Vec<(usize, usize)>>::new();
        for index in order {
            let pass = &program.interface.frame_graph.passes[index];
            let pipeline = pass.pipeline_index;
            let (owner, operation) = match &program.interface.pipelines[pipeline] {
                Pipeline::Compute(compute) => {
                    let stage = &compute.stages[pass.stage_index];
                    (
                        stage.owner.clone(),
                        Operation::Dispatch {
                            pipeline,
                            stage: pass.stage_index,
                            groups: program.groups(pipeline, &stage.dispatch_size)?,
                        },
                    )
                }
                Pipeline::Graphics(g) => {
                    let Some(stage) = g.stages.first() else {
                        return Err(HostError::Invalid("graphics pipeline without stages".into()));
                    };
                    (stage.owner.clone(), Operation::Draw { pipeline })
                }
            };
            let entry = entries.entry(owner.clone()).or_insert_with(|| Entry {
                name: owner,
                inputs: BTreeSet::new(),
                host_inputs: BTreeMap::new(),
                scalar_inputs: BTreeSet::new(),
                allocations: vec![],
                operations: vec![],
                results: vec![],
            });
            let start = entry.operations.len();
            let key = (pipeline, pass.stage_index);
            entry_stages.entry(entry.name.clone()).or_default().push(key);
            let mut replaced = false;
            if let Operation::Dispatch { stage, .. } = &operation {
                let Pipeline::Compute(compute) = &program.interface.pipelines[pipeline] else {
                    return Err(HostError::Invalid(
                        "compute operation in graphics pipeline".into(),
                    ));
                };
                for (task, scalar) in program.interface.scalar_tasks.iter().enumerate() {
                    if scalar.stage == compute.stages[*stage].entry_point {
                        entry.operations.push(Operation::Scalar { pipeline, task });
                        replaced |= scalar.replaces_dispatch;
                    }
                }
            }
            if !replaced {
                entry.operations.push(operation);
            }
            stage_operations.insert(key, entry.operations[start..].to_vec());
        }
        for result in &program.interface.source_results {
            entry_stages.entry(result.entry.clone()).or_default();
            entries.entry(result.entry.clone()).or_insert_with(|| Entry {
                name: result.entry.clone(),
                inputs: BTreeSet::new(),
                host_inputs: BTreeMap::new(),
                scalar_inputs: BTreeSet::new(),
                allocations: vec![],
                operations: vec![],
                results: vec![],
            });
        }
        for (_, mut entry) in entries {
            let stages = &entry_stages[&entry.name];
            let mut loops = BTreeMap::new();
            let mut members = BTreeSet::new();
            for (region, repeated) in program.interface.dispatch_loops.iter().enumerate() {
                let setup = (repeated.pipeline, repeated.setup);
                let Some(begin) = stages.iter().position(|key| *key == setup) else {
                    continue;
                };
                let completion = (repeated.pipeline, repeated.completion);
                let end = stages
                    .iter()
                    .position(|key| *key == completion)
                    .ok_or_else(|| HostError::Invalid("loop completion is outside its entry".into()))?;
                if end <= begin {
                    return Err(HostError::Invalid("loop completion precedes its setup".into()));
                }
                loops.insert(setup, region);
                let mut previous = begin;
                for &stage in &repeated.body {
                    let key = (repeated.pipeline, stage);
                    let position = stages
                        .iter()
                        .position(|candidate| *candidate == key)
                        .ok_or_else(|| HostError::Invalid("loop body stage is outside its entry".into()))?;
                    if position <= previous || position >= end {
                        return Err(HostError::Invalid("loop body is unordered".into()));
                    }
                    members.insert(key);
                    previous = position;
                }
                members.insert(completion);
            }
            let mut operations = Vec::new();
            for key in stages {
                if members.contains(key) {
                    continue;
                }
                if let Some(&region) = loops.get(key) {
                    let repeated = &program.interface.dispatch_loops[region];
                    let body = repeated
                        .body
                        .iter()
                        .flat_map(|stage| stage_operations[&(repeated.pipeline, *stage)].iter().cloned())
                        .collect();
                    operations.push(Operation::Loop {
                        setup: stage_operations[key].clone(),
                        pipeline: repeated.pipeline,
                        region,
                        body,
                    });
                    let Pipeline::Compute(compute) = &program.interface.pipelines[repeated.pipeline] else {
                        return Err(HostError::Invalid(
                            "loop completion must be a compute stage".into(),
                        ));
                    };
                    if !compute.stages[repeated.completion].uses.writes.is_empty() {
                        operations.extend(
                            stage_operations[&(repeated.pipeline, repeated.completion)].iter().cloned(),
                        );
                    }
                } else {
                    operations.extend(stage_operations[key].iter().cloned());
                }
            }
            entry.operations = operations;
            program.prepare_entry(&mut entry)?;
            program.entries.push(entry);
        }
        Ok(program)
    }

    pub fn depth_target(&self, pipeline: usize) -> Result<ResourceId, HostError> {
        let Some(&id) = self.depth_targets.get(&pipeline) else {
            return Err(HostError::Invalid(format!(
                "pipeline {pipeline} has no depth target"
            )));
        };
        Ok(id)
    }

    pub fn bindings(&self, pipeline: usize) -> &[Binding] {
        match &self.interface.pipelines[pipeline] {
            Pipeline::Compute(p) => &p.bindings,
            Pipeline::Graphics(p) => &p.bindings,
        }
    }

    /// Match the shader declaration rather than allocation-wide access. SPIR-V
    /// qualifies compute storage per entry point; WGSL uses the pipeline union.
    pub(crate) fn shader_binding(
        &self,
        pipeline: usize,
        stage: Option<usize>,
        index: usize,
        format: ShaderFormat,
    ) -> Result<Cow<'_, Binding>, HostError> {
        let binding = &self.bindings(pipeline)[index];
        if let (
            ShaderFormat::Spirv,
            Some(stage),
            Pipeline::Compute(compute),
            Binding::StorageBuffer { access, .. },
        ) = (format, stage, &self.interface.pipelines[pipeline], binding)
        {
            let Some(stage_access) = compute.stages[stage].uses.access(index) else {
                return Err(HostError::Invalid(format!(
                    "storage binding {index} has no access in {}",
                    compute.stages[stage].entry_point
                )));
            };
            if *access != stage_access {
                let mut binding = binding.clone();
                if let Binding::StorageBuffer { access, .. } = &mut binding {
                    *access = stage_access;
                }
                return Ok(Cow::Owned(binding));
            }
        }
        Ok(Cow::Borrowed(binding))
    }

    pub fn binding_resource(&self, pipeline: usize, binding: usize) -> Result<ResourceId, HostError> {
        let found = self.interface.frame_graph.resources.iter().enumerate().find_map(|(id, resource)| {
            resource
                .bindings
                .iter()
                .any(|b| b.pipeline_index == pipeline && b.binding_index == binding)
                .then_some(ResourceId(id))
        });
        let Some(id) = found else {
            return Err(HostError::Invalid(format!(
                "unmapped binding {pipeline}:{binding}"
            )));
        };
        Ok(id)
    }

    pub fn slot_resource(&self, pipeline: usize, set: u32, binding: u32) -> Result<ResourceId, HostError> {
        let Some(index) = self.bindings(pipeline).iter().position(|b| b.slot() == Some((set, binding)))
        else {
            return Err(HostError::Invalid(format!(
                "missing binding {set}:{binding} in pipeline {pipeline}"
            )));
        };
        self.binding_resource(pipeline, index)
    }

    pub fn draw_resource(
        &self,
        pipeline: usize,
        reference: &DrawBufferRef,
    ) -> Result<ResourceId, HostError> {
        self.slot_resource(pipeline, reference.set, reference.binding).or_else(|_| {
            let name = reference.frame_name();
            let Some(id) = self.interface.frame_graph.resources.iter().position(|r| r.name == name) else {
                return Err(HostError::Invalid(format!("missing draw resource {name}")));
            };
            Ok(ResourceId(id))
        })
    }

    pub fn resource_binding(&self, id: ResourceId) -> Option<&Binding> {
        self.interface.frame_graph.resources[id.0]
            .bindings
            .iter()
            .find_map(|r| self.bindings(r.pipeline_index).get(r.binding_index))
    }

    fn groups(&self, pipeline: usize, size: &DispatchSize) -> Result<[Expr; 3], HostError> {
        match size {
            DispatchSize::Fixed { x, y, z, .. } => Ok([
                Expr::Integer((*x).into()),
                Expr::Integer((*y).into()),
                Expr::Integer((*z).into()),
            ]),
            DispatchSize::Computed { groups, .. } => {
                let mut groups = groups.clone();
                for group in &mut groups {
                    self.resolve_size_sources(pipeline, group)?;
                }
                Ok(groups)
            }
        }
    }

    fn prepare_entry(&self, entry: &mut Entry) -> Result<(), HostError> {
        let mut dependencies = Dependencies::default();
        let mut pipelines = BTreeSet::new();
        let mut operations: Vec<_> = entry.operations.iter().collect();
        while let Some(op) = operations.pop() {
            let pipeline = match op {
                Operation::Dispatch { pipeline, groups, .. } => {
                    for e in groups {
                        e.dependencies(&mut dependencies)?;
                    }
                    *pipeline
                }
                Operation::Scalar { pipeline, task } => {
                    dependencies.scalar(&self.interface.scalar_tasks[*task].value, |source| {
                        self.scalar_resource(*pipeline, source)
                    })?;
                    *pipeline
                }
                Operation::Loop {
                    pipeline,
                    region,
                    setup,
                    body,
                } => {
                    let repeated = &self.interface.dispatch_loops[*region];
                    for value in [&repeated.initial_length, &repeated.count] {
                        dependencies.scalar(value, |source| self.scalar_resource(*pipeline, source))?;
                    }
                    operations.extend(setup);
                    operations.extend(body);
                    *pipeline
                }
                Operation::Draw { pipeline } => *pipeline,
            };
            if !pipelines.insert(pipeline) {
                continue;
            }
            for index in 0..self.bindings(pipeline).len() {
                dependencies.resources.insert(self.binding_resource(pipeline, index)?);
            }
            if let Pipeline::Graphics(g) = &self.interface.pipelines[pipeline] {
                for a in &g.vertex_inputs {
                    let Some(id) =
                        self.interface.frame_graph.resources.iter().position(|r| r.name == a.name)
                    else {
                        return Err(HostError::Invalid(format!("missing vertex input {}", a.name)));
                    };
                    dependencies.resources.insert(ResourceId(id));
                }
                if g.invocation.fragment_state.depth_test != DepthTest::Disabled {
                    dependencies.resources.insert(self.depth_target(pipeline)?);
                }
                for target in &g.fragment_outputs {
                    let Some(id) = self.interface.frame_graph.resources.iter().position(|r| {
                        r.name == target.name
                            && matches!(
                                r.kind,
                                FrameResourceKind::Texture | FrameResourceKind::StorageTexture
                            )
                    }) else {
                        return Err(HostError::Invalid(format!(
                            "missing color target {}",
                            target.name
                        )));
                    };
                    dependencies.resources.insert(ResourceId(id));
                }
                if let Some(reference) = g.invocation.draw.indices() {
                    dependencies.resources.insert(self.draw_resource(pipeline, reference)?);
                }
                if let Some(reference) = g.invocation.draw.indirect_commands() {
                    dependencies.resources.insert(self.draw_resource(pipeline, reference)?);
                }
                match &g.invocation.draw {
                    DrawCall::Indexed {
                        index_count: DrawCount::BufferLength,
                        indices,
                        ..
                    } => {
                        dependencies.scalar_inputs.insert(format!(
                            "count-resource-{}",
                            self.draw_resource(pipeline, indices)?.0
                        ));
                    }
                    DrawCall::Indirect {
                        draw_count: DrawCount::BufferLength,
                        commands,
                        ..
                    }
                    | DrawCall::IndexedIndirect {
                        draw_count: DrawCount::BufferLength,
                        commands,
                        ..
                    } => {
                        dependencies.scalar_inputs.insert(format!(
                            "count-resource-{}",
                            self.draw_resource(pipeline, commands)?.0
                        ));
                    }
                    _ => {}
                }
            }
        }
        let mut outputs =
            self.interface.source_results.iter().filter(|r| r.entry == entry.name).collect::<Vec<_>>();
        outputs.sort_by_key(|r| r.result);
        for output in outputs {
            pipelines.insert(output.pipeline_index);
            entry.results.push(self.slot_resource(output.pipeline_index, output.set, output.binding)?);
        }
        for &pipeline in &pipelines {
            if let Pipeline::Graphics(g) = &self.interface.pipelines[pipeline] {
                for output in &g.fragment_outputs {
                    if let Some(id) = self.interface.frame_graph.resources.iter().position(|r| {
                        r.name == output.name
                            && matches!(
                                r.kind,
                                FrameResourceKind::Texture | FrameResourceKind::StorageTexture
                            )
                    }) {
                        let id = ResourceId(id);
                        if !entry.results.contains(&id) {
                            entry.results.push(id);
                        }
                    }
                }
            }
        }
        dependencies.resources.extend(entry.results.iter().copied());
        let mut pending = BTreeMap::new();
        for id in dependencies.resources.clone() {
            if let Some(allocation) = self.allocation(id, &pipelines)? {
                let mut required = Dependencies::default();
                for expr in allocation.expressions() {
                    expr.dependencies(&mut required)?;
                }
                let refs = required.resources.clone();
                dependencies.extend(required);
                pending.insert(id, (allocation, refs));
            }
        }
        entry.inputs =
            dependencies.resources.difference(&pending.keys().copied().collect()).copied().collect();
        for &id in dependencies.host_inputs.keys() {
            if !entry.inputs.contains(&id)
                || !matches!(
                    self.interface.frame_graph.resources[id.0].kind,
                    FrameResourceKind::StorageBuffer
                        | FrameResourceKind::Uniform
                        | FrameResourceKind::PushConstant
                )
            {
                return Err(HostError::Invalid(
                    "CPU parameter is not an entry buffer input".into(),
                ));
            }
        }
        entry.host_inputs = dependencies.host_inputs;
        entry.scalar_inputs = dependencies.scalar_inputs;
        let mut available = entry.inputs.clone();
        while !pending.is_empty() {
            let next = pending.iter().find_map(|(&id, (_, refs))| refs.is_subset(&available).then_some(id));
            let Some(id) = next else {
                return Err(HostError::Invalid("cyclic resource allocation sizes".into()));
            };
            let Some((allocation, _)) = pending.remove(&id) else {
                return Err(HostError::Invalid("missing planned allocation".into()));
            };
            available.insert(id);
            entry.allocations.push(allocation);
        }
        Ok(())
    }

    fn resolve_size_sources(&self, pipeline: usize, expr: &mut Expr) -> Result<(), HostError> {
        let mut failure = None;
        expr.reads_mut(&mut |source, _| match self.scalar_resource(pipeline, source) {
            Ok(id) => *source = ScalarSource::Resource(id),
            Err(error) => failure = Some(error),
        });
        failure.map_or(Ok(()), Err)
    }

    fn allocation(
        &self,
        id: ResourceId,
        pipelines: &BTreeSet<usize>,
    ) -> Result<Option<Allocation>, HostError> {
        let resource = &self.interface.frame_graph.resources[id.0];
        let refs =
            resource.bindings.iter().filter(|b| pipelines.contains(&b.pipeline_index)).collect::<Vec<_>>();
        if refs.iter().any(|r| self.bindings(r.pipeline_index)[r.binding_index].is_input()) {
            return Ok(None);
        }
        for r in refs {
            match &self.bindings(r.pipeline_index)[r.binding_index] {
                Binding::StorageBuffer {
                    usage: BufferUsage::Output | BufferUsage::Intermediate,
                    length: Some(length),
                    ..
                } => {
                    let bytes = match length {
                        BufferLen::Fixed { bytes } => {
                            Expr::Integer(i64::try_from(*bytes).map_err(|_| {
                                HostError::Invalid("buffer size exceeds supported literal range".into())
                            })?)
                        }
                        BufferLen::LikeInput {
                            set,
                            binding,
                            elem_bytes,
                            src_elem_bytes,
                        } => Expr::BufferSize(self.slot_resource(r.pipeline_index, *set, *binding)?)
                            .floor(*src_elem_bytes)?
                            .multiply(Expr::Integer((*elem_bytes).into())),
                        BufferLen::SameAsDispatch { elem_bytes } => {
                            let Pipeline::Compute(p) = &self.interface.pipelines[r.pipeline_index] else {
                                return Ok(None);
                            };
                            let domain = p.stages.iter().find_map(|s| match &s.dispatch_size {
                                DispatchSize::Computed { elements, .. } => Some(elements),
                                DispatchSize::Fixed { .. } => None,
                            });
                            let Some(domain) = domain else {
                                return Ok(None);
                            };
                            let mut domain = domain.clone();
                            self.resolve_size_sources(r.pipeline_index, &mut domain)?;
                            domain.multiply(Expr::Integer((*elem_bytes).into()))
                        }
                        BufferLen::Computed { bytes } => {
                            let mut bytes = bytes.clone();
                            self.resolve_size_sources(r.pipeline_index, &mut bytes)?;
                            bytes
                        }
                    };
                    return Ok(Some(Allocation::Buffer { resource: id, bytes }));
                }
                Binding::StorageTexture { size, .. } => {
                    let (width, height) = match size {
                        StorageTextureSize::Fixed { width, height } => {
                            (Expr::Integer((*width).into()), Expr::Integer((*height).into()))
                        }
                        StorageTextureSize::SameAsWindow => (
                            Expr::Input("target-width".into()),
                            Expr::Input("target-height".into()),
                        ),
                    };
                    return Ok(Some(Allocation::Texture {
                        resource: id,
                        width,
                        height,
                    }));
                }
                _ => {}
            }
        }
        Ok(None)
    }
}
