use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

use crate::{
    Binding, BufferLen, BufferUsage, DepthTest, DispatchLen, DispatchSize, DrawBufferRef, DrawCall,
    DrawCount, FrameResource, FrameResourceKind, HostSizeInput, HostSizeScalar, ModuleInterface, Pipeline,
    SizeExpr, SizeOp, StorageTextureSize,
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

/// Mathematical integer expressions used for capacities and launch dimensions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Expr {
    Integer(i64),
    Input(String),
    BufferSize(ResourceId),
    ReadScalar {
        resource: ResourceId,
        offset: u32,
        signed: bool,
    },
    TextureDimension {
        resource: ResourceId,
        axis: usize,
    },
    Add(Box<Expr>, Box<Expr>),
    Subtract(Box<Expr>, Box<Expr>),
    Multiply(Box<Expr>, Box<Expr>),
    Floor(Box<Expr>, Box<Expr>),
    Ceiling(Box<Expr>, Box<Expr>),
    Mod(Box<Expr>, Box<Expr>),
    Min(Box<Expr>, Box<Expr>),
    Max(Box<Expr>, Box<Expr>),
}

impl Expr {
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

    pub(crate) fn resources(&self, result: &mut BTreeSet<ResourceId>) {
        match self {
            Self::BufferSize(id)
            | Self::ReadScalar { resource: id, .. }
            | Self::TextureDimension { resource: id, .. } => {
                result.insert(*id);
            }
            Self::Add(a, b)
            | Self::Subtract(a, b)
            | Self::Multiply(a, b)
            | Self::Mod(a, b)
            | Self::Floor(a, b)
            | Self::Ceiling(a, b)
            | Self::Min(a, b)
            | Self::Max(a, b) => {
                a.resources(result);
                b.resources(result);
            }
            Self::Integer(_) | Self::Input(_) => {}
        }
    }

    pub(crate) fn inputs(&self, result: &mut BTreeSet<String>) {
        match self {
            Self::Input(name) => {
                result.insert(name.clone());
            }
            Self::Add(a, b)
            | Self::Subtract(a, b)
            | Self::Multiply(a, b)
            | Self::Mod(a, b)
            | Self::Floor(a, b)
            | Self::Ceiling(a, b)
            | Self::Min(a, b)
            | Self::Max(a, b) => {
                a.inputs(result);
                b.inputs(result);
            }
            Self::Integer(_)
            | Self::BufferSize(_)
            | Self::ReadScalar { .. }
            | Self::TextureDimension { .. } => {}
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
        interface.rebuild_frame_graph();
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
            .topological_order()
            .map_err(|cycle| HostError::Invalid(format!("cyclic pass dependencies: {cycle:?}")))?;
        let mut entries = BTreeMap::<String, Entry>::new();
        let mut graphics = BTreeMap::<usize, usize>::new();
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
                            groups: program.groups(pipeline, &stage.dispatch_size, stage.workgroup_size)?,
                        },
                    )
                }
                Pipeline::Graphics(g) => {
                    let visited = graphics.entry(pipeline).or_default();
                    *visited += 1;
                    if *visited < g.stages.len() {
                        continue;
                    }
                    let Some(stage) = g.stages.first() else {
                        return Err(HostError::Invalid("graphics pipeline without stages".into()));
                    };
                    (stage.owner.clone(), Operation::Draw { pipeline })
                }
            };
            let entry = entries.entry(owner.clone()).or_insert_with(|| Entry {
                name: owner,
                inputs: BTreeSet::new(),
                scalar_inputs: BTreeSet::new(),
                allocations: vec![],
                operations: vec![],
                results: vec![],
            });
            entry.operations.push(operation);
        }
        for (_, mut entry) in entries {
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

    pub fn logical_count(&self, pipeline: usize, len: &DispatchLen) -> Result<Expr, HostError> {
        match *len {
            DispatchLen::Fixed { count } => Ok(Expr::Integer(count.into())),
            DispatchLen::InputBinding {
                set,
                binding,
                elem_bytes,
            } => Expr::BufferSize(self.slot_resource(pipeline, set, binding)?).floor(elem_bytes),
            DispatchLen::PushConstant { offset } => {
                let Some((index, base)) =
                    self.bindings(pipeline).iter().enumerate().find_map(|(i, b)| match b {
                        Binding::PushConstant {
                            offset: base, size, ..
                        } if offset >= *base
                            && u64::from(offset) + 4 <= u64::from(*base) + u64::from(*size) =>
                        {
                            Some((i, *base))
                        }
                        _ => None,
                    })
                else {
                    return Err(HostError::Invalid(format!(
                        "missing scalar at push offset {offset}"
                    )));
                };
                Ok(Expr::ReadScalar {
                    resource: self.binding_resource(pipeline, index)?,
                    offset: offset - base,
                    signed: false,
                })
            }
            DispatchLen::StorageBuffer { set, binding, offset } => Ok(Expr::ReadScalar {
                resource: self.slot_resource(pipeline, set, binding)?,
                offset,
                signed: false,
            }),
            DispatchLen::StorageImage { set, binding } => {
                let resource = self.slot_resource(pipeline, set, binding)?;
                Ok(Expr::TextureDimension { resource, axis: 0 }
                    .multiply(Expr::TextureDimension { resource, axis: 1 }))
            }
        }
    }

    fn groups(
        &self,
        pipeline: usize,
        size: &DispatchSize,
        workgroup: (u32, u32, u32),
    ) -> Result<[Expr; 3], HostError> {
        let one = Expr::Integer(1);
        match size {
            DispatchSize::Fixed { x, y, z, .. } => Ok([
                Expr::Integer((*x).into()),
                Expr::Integer((*y).into()),
                Expr::Integer((*z).into()),
            ]),
            DispatchSize::DerivedFrom {
                len: DispatchLen::StorageImage { set, binding },
                ..
            } => {
                let resource = self.slot_resource(pipeline, *set, *binding)?;
                Ok([
                    Expr::TextureDimension { resource, axis: 0 }.ceiling(workgroup.0)?,
                    Expr::TextureDimension { resource, axis: 1 }.ceiling(workgroup.1)?,
                    one,
                ])
            }
            DispatchSize::DerivedFrom { len, workgroup_size } => Ok([
                Expr::Min(
                    Box::new(Expr::Max(
                        Box::new(self.logical_count(pipeline, len)?.ceiling(*workgroup_size)?),
                        Box::new(one.clone()),
                    )),
                    Box::new(Expr::Integer(65_535)),
                ),
                one.clone(),
                one,
            ]),
        }
    }

    fn prepare_entry(&self, entry: &mut Entry) -> Result<(), HostError> {
        let mut used = BTreeSet::new();
        let mut pipelines = BTreeSet::new();
        for op in &entry.operations {
            let pipeline = match op {
                Operation::Dispatch { pipeline, groups, .. } => {
                    for e in groups {
                        e.resources(&mut used);
                        e.inputs(&mut entry.scalar_inputs);
                    }
                    *pipeline
                }
                Operation::Draw { pipeline } => *pipeline,
            };
            pipelines.insert(pipeline);
            for index in 0..self.bindings(pipeline).len() {
                used.insert(self.binding_resource(pipeline, index)?);
            }
            if let Pipeline::Graphics(g) = &self.interface.pipelines[pipeline] {
                for a in &g.vertex_inputs {
                    let Some(id) =
                        self.interface.frame_graph.resources.iter().position(|r| r.name == a.name)
                    else {
                        return Err(HostError::Invalid(format!("missing vertex input {}", a.name)));
                    };
                    used.insert(ResourceId(id));
                }
                if g.invocation.fragment_state.depth_test != DepthTest::Disabled {
                    used.insert(self.depth_target(pipeline)?);
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
                    used.insert(ResourceId(id));
                }
                if let Some(reference) = g.invocation.draw.indices() {
                    used.insert(self.draw_resource(pipeline, reference)?);
                }
                if let Some(reference) = g.invocation.draw.indirect_commands() {
                    used.insert(self.draw_resource(pipeline, reference)?);
                }
                match &g.invocation.draw {
                    DrawCall::Indexed {
                        index_count: DrawCount::BufferLength,
                        indices,
                        ..
                    } => {
                        entry.scalar_inputs.insert(format!(
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
                        entry.scalar_inputs.insert(format!(
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
        used.extend(entry.results.iter().copied());
        let mut pending = BTreeMap::new();
        for id in used.clone() {
            if let Some(allocation) = self.allocation(id, &pipelines)? {
                for expr in allocation.expressions() {
                    expr.resources(&mut used);
                    expr.inputs(&mut entry.scalar_inputs);
                }
                pending.insert(id, allocation);
            }
        }
        entry.inputs = used.difference(&pending.keys().copied().collect()).copied().collect();
        let mut available = entry.inputs.clone();
        while !pending.is_empty() {
            let next = pending.iter().find_map(|(&id, a)| {
                let mut refs = BTreeSet::new();
                for e in a.expressions() {
                    e.resources(&mut refs);
                }
                refs.is_subset(&available).then_some(id)
            });
            let Some(id) = next else {
                return Err(HostError::Invalid("cyclic resource allocation sizes".into()));
            };
            let Some(allocation) = pending.remove(&id) else {
                return Err(HostError::Invalid("missing planned allocation".into()));
            };
            available.insert(id);
            entry.allocations.push(allocation);
        }
        Ok(())
    }

    pub fn size_expression(&self, pipeline: usize, expr: &SizeExpr) -> Result<Expr, HostError> {
        Ok(match expr {
            SizeExpr::Integer(n) => Expr::Integer(*n),
            SizeExpr::BufferLength { set, binding, stride } => {
                Expr::BufferSize(self.slot_resource(pipeline, *set, *binding)?).floor(*stride)?
            }
            SizeExpr::Scalar(input) => {
                let (mut expression, scalar) = match input {
                    HostSizeInput::Uniform {
                        set,
                        binding,
                        offset,
                        scalar,
                        ..
                    } => (
                        Expr::ReadScalar {
                            resource: self.slot_resource(pipeline, *set, *binding)?,
                            offset: *offset,
                            signed: false,
                        },
                        scalar,
                    ),
                    HostSizeInput::PushConstant {
                        push_constant_offset,
                        scalar,
                        ..
                    } => (
                        self.logical_count(
                            pipeline,
                            &DispatchLen::PushConstant {
                                offset: *push_constant_offset,
                            },
                        )?,
                        scalar,
                    ),
                };
                match scalar {
                    HostSizeScalar::I32 => {
                        if let Expr::ReadScalar { signed, .. } = &mut expression {
                            *signed = true;
                        }
                    }
                    HostSizeScalar::U32 => {}
                    HostSizeScalar::F32 => {
                        return Err(HostError::Invalid(
                            "floating-point capacity needs an explicit conversion".into(),
                        ))
                    }
                }
                expression
            }
            SizeExpr::Binary { op, left, right } => {
                let a = Box::new(self.size_expression(pipeline, left)?);
                let b = Box::new(self.size_expression(pipeline, right)?);
                match op {
                    SizeOp::Add => Expr::Add(a, b),
                    SizeOp::Subtract => Expr::Subtract(a, b),
                    SizeOp::Multiply => Expr::Multiply(a, b),
                    SizeOp::Floor => Expr::Floor(a, b),
                    SizeOp::Ceiling => Expr::Ceiling(a, b),
                    SizeOp::Mod => Expr::Mod(a, b),
                    SizeOp::Min => Expr::Min(a, b),
                    SizeOp::Max => Expr::Max(a, b),
                }
            }
        })
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
                                DispatchSize::DerivedFrom { len, .. } => Some(len),
                                DispatchSize::Fixed { .. } => None,
                            });
                            let Some(domain) = domain else {
                                return Ok(None);
                            };
                            self.logical_count(r.pipeline_index, domain)?
                                .multiply(Expr::Integer((*elem_bytes).into()))
                        }
                        BufferLen::HostProvided { .. } => return Ok(None),
                        BufferLen::Computed { bytes } => self.size_expression(r.pipeline_index, bytes)?,
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
