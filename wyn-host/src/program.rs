use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

use crate::{
    arithmetic, Binding, BufferLen, BufferSource, BufferUsage, DepthTest, DispatchLoopControl,
    DispatchSize, DrawBufferRef, DrawCall, DrawCount, FrameResource, FrameResourceKind, ModuleInterface,
    Pipeline, ResultShape, ScalarExpr, ScalarSource, SourceResultBinding, StorageTextureSize,
};

mod readbacks;

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
    Add(Box<Expr>, Box<Expr>),
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
    reads: BTreeSet<ResourceId>,
    writes: BTreeSet<ResourceId>,
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
        self.reads.extend(other.reads);
        self.writes.extend(other.writes);
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
            ScalarExpr::Parameter { source, offset, .. } => {
                sources.push((source, u64::from(*offset) + 4, false))
            }
            ScalarExpr::Read { source, .. } => sources.push((source, 0, true)),
            ScalarExpr::BufferLength { source, .. } => sources.push((source, 0, false)),
            _ => {}
        });
        for (source, bytes, read) in sources {
            let id = resolve(source)?;
            if read {
                self.reads.insert(id);
            }
            self.resource(id, bytes);
        }
        Ok(())
    }
}

impl Expr {
    /// Evaluate capacity arithmetic that does not depend on host inputs or resources.
    pub fn constant(&self) -> Result<Option<i64>, HostError> {
        let (left, right, operation): (_, _, fn(i64, i64) -> Result<i64, HostError>) = match self {
            Self::Integer(value) => return Ok(Some(*value)),
            Self::Scalar(ScalarExpr::I32(value)) => return Ok(Some(i64::from(*value))),
            Self::Scalar(ScalarExpr::U32(value)) => return Ok(Some(i64::from(*value))),
            Self::Add(a, b) => (a, b, arithmetic::add),
            Self::Subtract(a, b) => (a, b, arithmetic::subtract),
            Self::Multiply(a, b) => (a, b, arithmetic::multiply),
            Self::Floor(a, b) => (a, b, arithmetic::floor),
            Self::Ceiling(a, b) => (a, b, arithmetic::ceiling),
            Self::Min(a, b) => (a, b, |a, b| Ok(a.min(b))),
            Self::Max(a, b) => (a, b, |a, b| Ok(a.max(b))),
            _ => return Ok(None),
        };
        match (left.constant()?, right.constant()?) {
            (Some(left), Some(right)) => operation(left, right).map(Some),
            _ => Ok(None),
        }
    }

    fn dependencies(&self, result: &mut Dependencies) -> Result<(), HostError> {
        match self {
            Self::Scalar(value) => result.scalar(value, ScalarSource::resource)?,
            Self::BufferLength { source, .. } | Self::TextureDimension { source, .. } => {
                result.resource(source.resource()?, 0)
            }
            Self::BufferSize(id) => result.resource(*id, 0),
            Self::ReadScalar { resource: id, .. } => {
                result.resource(*id, 0);
                result.reads.insert(*id);
            }
            Self::Input(name) => {
                result.scalar_inputs.insert(name.clone());
            }
            Self::Add(a, b)
            | Self::Subtract(a, b)
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
            Self::Add(left, right)
            | Self::Subtract(left, right)
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
    Bind {
        name: String,
        value: ScalarExpr,
    },
    Allocate(Allocation),
    Loop {
        pipeline: usize,
        region: usize,
        test: Vec<Operation>,
        body: Vec<Operation>,
    },
    Scalar {
        pipeline: usize,
        task: usize,
        value: ScalarExpr,
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

impl Operation {
    pub(crate) fn pipeline(&self) -> Option<usize> {
        match self {
            Self::Allocate(_) | Self::Bind { .. } => None,
            Self::Loop { pipeline, .. }
            | Self::Scalar { pipeline, .. }
            | Self::Dispatch { pipeline, .. }
            | Self::Draw { pipeline } => Some(*pipeline),
        }
    }
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
    pub operations: Vec<Operation>,
    pub results: Vec<ResourceId>,
    pub source_results: Vec<SourceResultBinding>,
}

impl Entry {
    pub(crate) fn scalar_bindings(&self) -> impl Iterator<Item = &str> {
        walk_operations(&self.operations).filter_map(|operation| match operation {
            Operation::Bind { name, .. } => Some(name.as_str()),
            _ => None,
        })
    }
    pub fn allocations(&self) -> impl Iterator<Item = &Allocation> {
        walk_operations(&self.operations).filter_map(|op| match op {
            Operation::Allocate(allocation) => Some(allocation),
            _ => None,
        })
    }
}

fn walk_operations(operations: &[Operation]) -> impl Iterator<Item = &Operation> {
    let mut pending: Vec<_> = operations.iter().rev().collect();
    std::iter::from_fn(move || {
        let operation = pending.pop()?;
        if let Operation::Loop { test, body, .. } = operation {
            pending.extend(body.iter().rev());
            pending.extend(test.iter().rev());
        }
        Some(operation)
    })
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
        for i in 0..program.interface.source_results.len() {
            let pipeline = program.interface.source_results[i].pipeline_index;
            let mut shape = std::mem::take(&mut program.interface.source_results[i].shape);
            for dimension in shape.expressions_mut() {
                program.resolve_size_sources(pipeline, dimension)?;
            }
            if let ResultShape::Segmented { offsets, .. } = &mut shape {
                if offsets.is_empty() {
                    return Err(HostError::Invalid("segmented result has no offsets".into()));
                }
                for source in offsets {
                    let id = program.scalar_resource(pipeline, source)?;
                    if program.interface.frame_graph.resources[id.0].kind
                        != FrameResourceKind::StorageBuffer
                    {
                        return Err(HostError::Invalid(
                            "segment offsets must be storage buffers".into(),
                        ));
                    }
                    *source = ScalarSource::Resource(id);
                }
            }
            program.interface.source_results[i].shape = shape;
        }
        for pipeline in 0..program.interface.pipelines.len() {
            let Pipeline::Graphics(graphics) = &program.interface.pipelines[pipeline] else {
                continue;
            };
            let draw = graphics.invocation.draw.try_map_buffers(|buffer| {
                let mut buffer = buffer.clone();
                if let Some((producer, count)) = &mut buffer.elements {
                    program.resolve_size_sources(*producer, count)?;
                }
                Ok::<_, HostError>(buffer)
            })?;
            let Pipeline::Graphics(graphics) = &mut program.interface.pipelines[pipeline] else {
                unreachable!()
            };
            graphics.invocation.draw = draw;
        }
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
                operations: vec![],
                results: vec![],
                source_results: vec![],
            });
            let start = entry.operations.len();
            let key = (pipeline, pass.stage_index);
            entry_stages.entry(entry.name.clone()).or_default().push(key);
            let mut replaced = matches!(&operation, Operation::Dispatch { groups, .. }
                if groups.iter().any(|group| matches!(group, Expr::Integer(0))));
            if let Operation::Dispatch { stage, .. } = &operation {
                let Pipeline::Compute(compute) = &program.interface.pipelines[pipeline] else {
                    return Err(HostError::Invalid(
                        "compute operation in graphics pipeline".into(),
                    ));
                };
                for (task, scalar) in program.interface.scalar_tasks.iter().enumerate() {
                    if scalar.stage == compute.stages[*stage].entry_point {
                        let mut value = Expr::Scalar(scalar.value.clone());
                        program.resolve_size_sources(pipeline, &mut value)?;
                        let Expr::Scalar(value) = value else {
                            unreachable!()
                        };
                        entry.operations.push(Operation::Scalar {
                            pipeline,
                            task,
                            value,
                        });
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
                operations: vec![],
                results: vec![],
                source_results: vec![],
            });
        }
        for (_, mut entry) in entries {
            entry.source_results = program
                .interface
                .source_results
                .iter()
                .filter(|result| result.entry == entry.name)
                .cloned()
                .collect();
            let stages = &entry_stages[&entry.name];
            let positions: BTreeMap<_, _> = stages.iter().enumerate().map(|(i, key)| (*key, i)).collect();
            entry.operations = program.structured_operations(
                stages,
                &stage_operations,
                &positions,
                &mut BTreeSet::new(),
            )?;
            program.prepare_entry(&mut entry)?;
            program.cache_readbacks(&mut entry)?;
            program.entries.push(entry);
        }
        Ok(program)
    }

    fn structured_operations(
        &self,
        stages: &[(usize, usize)],
        operations: &BTreeMap<(usize, usize), Vec<Operation>>,
        positions: &BTreeMap<(usize, usize), usize>,
        emitted: &mut BTreeSet<(usize, usize)>,
    ) -> Result<Vec<Operation>, HostError> {
        let mut result = Vec::new();
        for &key in stages {
            if !emitted.insert(key) {
                continue;
            }
            let repeated = self
                .interface
                .dispatch_loops
                .iter()
                .enumerate()
                .find(|(_, repeated)| key == (repeated.pipeline, repeated.setup));
            let Some((region, repeated)) = repeated else {
                result.extend(operations[&key].iter().cloned());
                continue;
            };
            let completion = (repeated.pipeline, repeated.completion);
            let begin = positions[&key];
            let end = *positions
                .get(&completion)
                .ok_or_else(|| HostError::Invalid("loop completion is outside its entry".into()))?;
            if begin >= end {
                return Err(HostError::Invalid("loop completion precedes its setup".into()));
            }
            let ordered = |members: &[usize]| -> Result<Vec<(usize, usize)>, HostError> {
                let mut keys = members.iter().map(|&stage| (repeated.pipeline, stage)).collect::<Vec<_>>();
                for key in &keys {
                    if !positions.get(key).is_some_and(|&position| begin < position && position < end) {
                        return Err(HostError::Invalid("loop member is outside its boundaries".into()));
                    }
                }
                keys.sort_by_key(|key| positions[key]);
                Ok(keys)
            };
            let test =
                self.structured_operations(&ordered(&repeated.test)?, operations, positions, emitted)?;
            let body =
                self.structured_operations(&ordered(&repeated.body)?, operations, positions, emitted)?;
            result.extend(operations[&key].iter().cloned());
            result.push(Operation::Loop {
                pipeline: repeated.pipeline,
                region,
                test,
                body,
            });
            let Pipeline::Compute(compute) = &self.interface.pipelines[repeated.pipeline] else {
                return Err(HostError::Invalid(
                    "loop completion must be a compute stage".into(),
                ));
            };
            if !compute.stages[repeated.completion].uses.writes.is_empty() {
                result.extend(operations[&completion].iter().cloned());
            }
            emitted.insert(completion);
        }
        Ok(result)
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

    pub(crate) fn result_resource(&self, entry: &Entry, id: ResourceId) -> bool {
        entry.results.contains(&id)
            || self
                .interface
                .source_results
                .iter()
                .filter(|r| r.entry == entry.name)
                .any(|r| r.shape.offsets().iter().any(|s| s.resource().is_ok_and(|r| r == id)))
    }

    pub(crate) fn draw_count(
        &self,
        pipeline: usize,
        count: &DrawCount,
        buffer: &DrawBufferRef,
    ) -> Result<Expr, HostError> {
        Ok(match count {
            DrawCount::Fixed(n) => Expr::Integer(i64::from(*n)),
            DrawCount::BufferLength => {
                if let Some((_, count)) = &buffer.elements {
                    count.clone()
                } else {
                    Expr::Input(format!(
                        "count-resource-{}",
                        self.draw_resource(pipeline, buffer)?.0
                    ))
                }
            }
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
        for op in walk_operations(&entry.operations) {
            dependencies.extend(self.operation_dependencies(op)?);
            let Some(pipeline) = op.pipeline() else {
                return Err(HostError::Invalid("entry was already scheduled".into()));
            };
            if !pipelines.insert(pipeline) {
                continue;
            }
            for index in 0..self.bindings(pipeline).len() {
                dependencies.resources.insert(self.binding_resource(pipeline, index)?);
            }
            if let Pipeline::Graphics(g) = &self.interface.pipelines[pipeline] {
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
                    let id = ResourceId(id);
                    dependencies.resources.insert(id);
                    entry.results.push(id);
                }
            }
        }
        let attachments = std::mem::take(&mut entry.results);
        let mut outputs =
            self.interface.source_results.iter().filter(|r| r.entry == entry.name).collect::<Vec<_>>();
        outputs.sort_by_key(|r| r.result);
        for output in outputs {
            for dimension in output.shape.expressions() {
                dimension.dependencies(&mut dependencies)?;
            }
            for source in output.shape.offsets() {
                dependencies.resources.insert(source.resource()?);
            }
            pipelines.insert(output.pipeline_index);
            entry.results.push(self.slot_resource(output.pipeline_index, output.set, output.binding)?);
        }
        for attachment in attachments {
            if !entry.results.contains(&attachment) {
                entry.results.push(attachment);
            }
        }
        dependencies.resources.extend(entry.results.iter().copied());
        let mut pending = BTreeMap::new();
        let mut unresolved = dependencies.resources.iter().copied().collect::<Vec<_>>();
        let mut seen = BTreeSet::new();
        while let Some(id) = unresolved.pop() {
            if !seen.insert(id) {
                continue;
            }
            if let Some(allocation) = self.allocation(id, &pipelines)? {
                let mut required = Dependencies::default();
                for expr in allocation.expressions() {
                    expr.dependencies(&mut required)?;
                }
                let refs = required.resources.clone();
                let reads = required.reads.clone();
                unresolved.extend(refs.iter().copied());
                dependencies.extend(required);
                pending.insert(id, (allocation, refs, reads));
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
        let mut initialized = available.clone();
        // Hoist only allocations whose sizes depend on immutable entry inputs.
        let invariant = pending
            .iter()
            .filter_map(|(&id, (_, refs, reads))| {
                (reads.is_empty() && refs.is_subset(&entry.inputs)).then_some(id)
            })
            .collect::<Vec<_>>();
        let mut body = std::mem::take(&mut entry.operations);
        for id in invariant {
            Self::allocate_resource(
                id,
                &mut entry.operations,
                &mut pending,
                &mut available,
                &initialized,
            )?;
        }
        self.allocate_operations(&mut body, &mut pending, &mut available, &mut initialized)?;
        entry.operations.extend(body);
        for &id in &entry.results {
            Self::allocate_resource(
                id,
                &mut entry.operations,
                &mut pending,
                &mut available,
                &initialized,
            )?;
        }
        Ok(())
    }

    fn allocate_resource(
        id: ResourceId,
        operations: &mut Vec<Operation>,
        pending: &mut BTreeMap<ResourceId, (Allocation, BTreeSet<ResourceId>, BTreeSet<ResourceId>)>,
        available: &mut BTreeSet<ResourceId>,
        initialized: &BTreeSet<ResourceId>,
    ) -> Result<(), HostError> {
        if available.contains(&id) {
            return Ok(());
        }
        let Some((allocation, refs, reads)) = pending.remove(&id) else {
            return Err(HostError::Invalid(format!(
                "cyclic or escaping allocation dependency for resource {id:?}"
            )));
        };
        for dependency in refs {
            Self::allocate_resource(dependency, operations, pending, available, initialized)?;
        }
        if !reads.is_subset(initialized) {
            return Err(HostError::Invalid(
                "allocation reads an uninitialized size".into(),
            ));
        }
        available.insert(id);
        operations.push(Operation::Allocate(allocation));
        Ok(())
    }

    fn allocate_operations(
        &self,
        operations: &mut Vec<Operation>,
        pending: &mut BTreeMap<ResourceId, (Allocation, BTreeSet<ResourceId>, BTreeSet<ResourceId>)>,
        available: &mut BTreeSet<ResourceId>,
        initialized: &mut BTreeSet<ResourceId>,
    ) -> Result<(), HostError> {
        for mut operation in std::mem::take(operations) {
            let required = self.operation_dependencies(&operation)?;
            for id in required.resources {
                Self::allocate_resource(id, operations, pending, available, initialized)?;
            }
            if let Operation::Loop {
                pipeline,
                region,
                test,
                body,
            } = &mut operation
            {
                // Local allocations are recreated on each iteration; only the
                // explicit carried bindings can leave the repeated region.
                let mut local = available.clone();
                let mut written = initialized.clone();
                let repeated = &self.interface.dispatch_loops[*region];
                for carry in &repeated.carried {
                    let initial = self.buffer_dependencies(*pipeline, &carry.initial)?;
                    if !initial.reads.is_subset(initialized) {
                        return Err(HostError::Invalid(
                            "loop carry has no initialized incoming storage".into(),
                        ));
                    }
                    written.insert(self.scalar_resource(*pipeline, &carry.binding)?);
                }
                if let DispatchLoopControl::Counted { index, .. } = &repeated.control {
                    written.insert(self.scalar_resource(*pipeline, index)?);
                }
                self.allocate_operations(test, pending, &mut local, &mut written)?;
                if !required.reads.is_subset(&written) {
                    return Err(HostError::Invalid(
                        "loop control reads uninitialized storage".into(),
                    ));
                }
                self.allocate_operations(body, pending, &mut local, &mut written)?;
                for carry in &repeated.carried {
                    let selected = self.buffer_dependencies(*pipeline, &carry.next)?;
                    if !selected.reads.is_subset(&written) {
                        return Err(HostError::Invalid(
                            "loop carry selects uninitialized storage".into(),
                        ));
                    }
                }
            } else if !required.reads.is_subset(initialized) {
                return Err(HostError::Invalid("operation reads uninitialized storage".into()));
            }
            initialized.extend(required.writes);
            operations.push(operation);
        }
        Ok(())
    }

    fn buffer_dependencies(
        &self,
        pipeline: usize,
        source: &BufferSource,
    ) -> Result<Dependencies, HostError> {
        let mut dependencies = Dependencies::default();
        // Rebinding a buffer handle does not read its payload (which may be empty or uninitialized).
        for binding in source.bindings() {
            dependencies.resources.insert(self.scalar_resource(pipeline, binding)?);
        }
        if let BufferSource::Select { tag, cases } = source {
            let keys: BTreeSet<_> = cases.iter().map(|(key, _)| key).collect();
            if cases.is_empty() || keys.len() != cases.len() {
                return Err(HostError::Invalid(
                    "loop buffer selection requires distinct cases".into(),
                ));
            }
            dependencies.scalar(tag, |source| self.scalar_resource(pipeline, source))?;
        }
        Ok(dependencies)
    }

    fn operation_dependencies(&self, operation: &Operation) -> Result<Dependencies, HostError> {
        let mut dependencies = Dependencies::default();
        match operation {
            Operation::Bind { value, .. } => dependencies.scalar(value, ScalarSource::resource)?,
            Operation::Allocate(_) => {
                return Err(HostError::Invalid("entry was already scheduled".into()));
            }
            Operation::Loop {
                pipeline,
                region,
                test,
                body,
            } => {
                let repeated = &self.interface.dispatch_loops[*region];
                dependencies.scalar(repeated.control.expression(), |source| {
                    self.scalar_resource(*pipeline, source)
                })?;
                if let DispatchLoopControl::Counted { index, .. } = &repeated.control {
                    dependencies.resources.insert(self.scalar_resource(*pipeline, index)?);
                }
                for carry in &repeated.carried {
                    let current = self.scalar_resource(*pipeline, &carry.binding)?;
                    dependencies.resources.insert(current);
                    dependencies.writes.insert(current);
                    dependencies.extend(self.buffer_dependencies(*pipeline, &carry.initial)?);
                }
                let mut produced = BTreeSet::new();
                for operation in test.iter().chain(body) {
                    produced.extend(self.operation_dependencies(operation)?.writes);
                }
                let mut selections = Dependencies::default();
                for carry in &repeated.carried {
                    selections.extend(self.buffer_dependencies(*pipeline, &carry.next)?);
                }
                selections.resources.retain(|resource| !produced.contains(resource));
                selections.reads.retain(|resource| !produced.contains(resource));
                dependencies.extend(selections);
            }
            Operation::Scalar {
                pipeline,
                task,
                value,
            } => {
                let task = &self.interface.scalar_tasks[*task];
                let id = self.scalar_resource(*pipeline, &task.destination)?;
                dependencies.resources.insert(id);
                dependencies.writes.insert(id);
                dependencies.scalar(value, |source| self.scalar_resource(*pipeline, source))?;
            }
            Operation::Dispatch { pipeline, .. } | Operation::Draw { pipeline } => {
                let stage = if let Operation::Dispatch { stage, groups, .. } = operation {
                    for group in groups {
                        group.dependencies(&mut dependencies)?;
                    }
                    *stage
                } else {
                    if let Pipeline::Graphics(graphics) = &self.interface.pipelines[*pipeline] {
                        let count = match &graphics.invocation.draw {
                            DrawCall::Indexed {
                                index_count, indices, ..
                            } => Some(self.draw_count(*pipeline, index_count, indices)?),
                            DrawCall::Indirect {
                                draw_count, commands, ..
                            }
                            | DrawCall::IndexedIndirect {
                                draw_count, commands, ..
                            } => Some(self.draw_count(*pipeline, draw_count, commands)?),
                            DrawCall::Direct { .. } => None,
                        };
                        if let Some(count) = count {
                            count.dependencies(&mut dependencies)?;
                        }
                    }
                    0
                };
                let Some(pass) = self
                    .interface
                    .frame_graph
                    .passes
                    .iter()
                    .find(|p| p.pipeline_index == *pipeline && p.stage_index == stage)
                else {
                    return Err(HostError::Invalid(
                        "operation has no resource access summary".into(),
                    ));
                };
                dependencies
                    .resources
                    .extend(pass.reads.iter().chain(&pass.writes).map(|r| ResourceId(r.resource)));
                dependencies.writes.extend(pass.writes.iter().map(|r| ResourceId(r.resource)));
            }
        }
        Ok(dependencies)
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
                        // Empty arrays still need one physical element for shader binding.
                        BufferLen::LikeInput {
                            set,
                            binding,
                            elem_bytes,
                            src_elem_bytes,
                        } => Expr::Max(
                            Box::new(Expr::Integer(1)),
                            Box::new(
                                Expr::BufferSize(self.slot_resource(r.pipeline_index, *set, *binding)?)
                                    .floor(*src_elem_bytes)?,
                            ),
                        )
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
                            Expr::Max(Box::new(Expr::Integer(1)), Box::new(domain))
                                .multiply(Expr::Integer((*elem_bytes).into()))
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

#[cfg(test)]
#[path = "program_tests.rs"]
mod tests;
