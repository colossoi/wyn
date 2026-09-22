use std::fmt::Write;

use crate::{
    Access, Allocation, Binding, BlendMode, BufferLen, CullMode, DepthTest, DrawCall, DrawCount, Entry,
    Expr, FillMode, FrameResourceKind, FrontFace, HostError, IndexFormat, IntegerOp, Operation, Pipeline,
    PrimitiveTopology, Program, ResourceId, ResultLayout, ResultScalar, SamplerBindingType, Scissor,
    ShaderFormat, ShaderStage, StorageImageFormat, TextureSampleType, TextureViewDimension, UniformMember,
    VertexFormat, Viewport,
};

pub(crate) fn symbol(name: &str) -> String {
    let mut out = String::from("source-");
    for byte in name.bytes() {
        if byte.is_ascii_alphanumeric() && !byte.is_ascii_uppercase() {
            out.push(char::from(byte));
        } else {
            out.push_str(&format!("-{:02x}-", byte));
        }
    }
    out
}

fn string(value: &str) -> String {
    format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
}
fn resource(id: ResourceId) -> String {
    format!("resource-{}", id.0)
}
fn access(value: &Access) -> &'static str {
    match value {
        Access::ReadOnly => ":read",
        Access::WriteOnly => ":write",
        Access::ReadWrite => ":read-write",
    }
}
pub(crate) fn format_name(format: StorageImageFormat) -> &'static str {
    match format {
        StorageImageFormat::Rgba8Unorm => "rgba8unorm",
        StorageImageFormat::Rgba16Float => "rgba16float",
        StorageImageFormat::Rgba32Float => "rgba32float",
        StorageImageFormat::R32Float => "r32float",
    }
}
fn dimension(d: &TextureViewDimension) -> &'static str {
    match d {
        TextureViewDimension::D1 => ":d1",
        TextureViewDimension::D2 => ":d2",
        TextureViewDimension::D2Array => ":d2-array",
        TextureViewDimension::Cube => ":cube",
        TextureViewDimension::CubeArray => ":cube-array",
        TextureViewDimension::D3 => ":d3",
    }
}
fn sampler(k: &SamplerBindingType) -> &'static str {
    match k {
        SamplerBindingType::Filtering => ":filtering",
        SamplerBindingType::NonFiltering => ":non-filtering",
        SamplerBindingType::Comparison => ":comparison",
    }
}
pub(crate) fn comparison(d: DepthTest) -> &'static str {
    match d {
        DepthTest::Disabled => "disabled",
        DepthTest::Never => "never",
        DepthTest::Less => "less",
        DepthTest::LessEqual => "less-equal",
        DepthTest::Equal => "equal",
        DepthTest::GreaterEqual => "greater-equal",
        DepthTest::Greater => "greater",
        DepthTest::Always => "always",
    }
}
fn vertex_type(v: &VertexFormat) -> String {
    let (scalar, count) = match v {
        VertexFormat::Float32 => ("f32", 1),
        VertexFormat::Float32x2 => ("f32", 2),
        VertexFormat::Float32x3 => ("f32", 3),
        VertexFormat::Float32x4 => ("f32", 4),
        VertexFormat::Sint32 => ("i32", 1),
        VertexFormat::Sint32x2 => ("i32", 2),
        VertexFormat::Sint32x3 => ("i32", 3),
        VertexFormat::Sint32x4 => ("i32", 4),
        VertexFormat::Uint32 => ("u32", 1),
        VertexFormat::Uint32x2 => ("u32", 2),
        VertexFormat::Uint32x3 => ("u32", 3),
        VertexFormat::Uint32x4 => ("u32", 4),
    };
    if count == 1 {
        format!(":{scalar}")
    } else {
        format!("(:vector :{scalar} {count})")
    }
}

impl Expr {
    pub fn to_whl(&self) -> String {
        match self {
            Self::I32 { op, left, right } | Self::U32 { op, left, right } => {
                let ty = if matches!(self, Self::I32 { .. }) { "i32" } else { "u32" };
                let op = match op {
                    IntegerOp::Add => "add",
                    IntegerOp::Subtract => "sub",
                    IntegerOp::Multiply => "mul",
                };
                format!(
                    "(i64 ({ty}-{op} ({ty} {}) ({ty} {})))",
                    left.to_whl(),
                    right.to_whl()
                )
            }
            Self::Integer(n) => format!("(i64 {n})"),
            Self::Input(n) => format!("(i64 {n})"),
            Self::BufferSize(r) => format!("(i64 (gpu-buffer-size {}))", resource(*r)),
            Self::ReadScalar {
                resource: r,
                offset,
                signed,
            } => format!(
                "(i64 (gpu-read-scalar {} {offset} '{}))",
                resource(*r),
                if *signed { "i32" } else { "u32" }
            ),
            Self::TextureDimension { resource: r, axis } => format!(
                "(i64 (gpu-texture-dimension {} 0 '{}))",
                resource(*r),
                match axis {
                    0 => "width",
                    1 => "height",
                    _ => "depth-or-layers",
                }
            ),
            Self::Add(a, b) => format!("(+ {} {})", a.to_whl(), b.to_whl()),
            Self::Subtract(a, b) => format!("(- {} {})", a.to_whl(), b.to_whl()),
            Self::Multiply(a, b) => format!("(* {} {})", a.to_whl(), b.to_whl()),
            Self::Floor(a, b) => format!("(floor {} {})", a.to_whl(), b.to_whl()),
            Self::Ceiling(a, b) => format!("(ceiling {} {})", a.to_whl(), b.to_whl()),
            Self::Mod(a, b) => format!("(mod {} {})", a.to_whl(), b.to_whl()),
            Self::Min(a, b) => format!("(min {} {})", a.to_whl(), b.to_whl()),
            Self::Max(a, b) => format!("(max {} {})", a.to_whl(), b.to_whl()),
        }
    }
}

impl Program {
    /// Render a host artifact referencing a sibling shader module.
    pub fn to_whl(&self, module_path: &str, format: ShaderFormat) -> Result<String, HostError> {
        let mut out = String::from("(define-host-program :version 1)\n");
        writeln!(
            out,
            "(define-gpu-module 'shaders :format :{} :path {})\n",
            match format {
                ShaderFormat::Spirv => "spirv",
                ShaderFormat::Wgsl => "wgsl",
            },
            string(module_path)
        )?;
        for (p, pipeline) in self.interface.pipelines.iter().enumerate() {
            match pipeline {
                Pipeline::Compute(c) => {
                    for (s, stage) in c.stages.iter().enumerate() {
                        writeln!(out,"(define-gpu-kernel 'kernel-{p}-{s}\n  :module 'shaders :entry {}\n  :workgroup-size '({} {} {})",string(&stage.entry_point),stage.workgroup_size.0,stage.workgroup_size.1,stage.workgroup_size.2)?;
                        self.write_parameters(&mut out, p, Some(s), format)?;
                        writeln!(out, ")\n")?;
                    }
                }
                Pipeline::Graphics(g) => {
                    writeln!(out, "(define-gpu-graphics 'graphics-{p}")?;
                    for (kind, label) in [
                        (ShaderStage::Vertex, "vertex"),
                        (ShaderStage::Fragment, "fragment"),
                    ] {
                        match g.stages.iter().find(|s| s.stage == kind) {
                            Some(stage) => {
                                writeln!(out, "  :{label} '(shaders {})", string(&stage.entry_point))?
                            }
                            None => writeln!(out, "  :{label} nil")?,
                        }
                    }
                    self.write_parameters(&mut out, p, None, format)?;
                    write!(out, "\n  :vertex-inputs '(")?;
                    for a in &g.vertex_inputs {
                        write!(
                            out,
                            "({} {} {} {} 0 :vertex)",
                            a.slot,
                            symbol(&a.name),
                            vertex_type(&a.format),
                            a.format.byte_size()
                        )?;
                    }
                    write!(out, ")\n  :color-outputs '(")?;
                    for a in &g.fragment_outputs {
                        write!(out, "({} :caller)", a.location)?;
                    }
                    let r = g.invocation.raster_state;
                    let f = g.invocation.fragment_state;
                    writeln!(
                        out,
                        ")\n  :depth-format {} :samples 1",
                        if f.depth_test == DepthTest::Disabled { "nil" } else { ":depth32float" }
                    )?;
                    writeln!(
                        out,
                        "  :topology :{} :front-face :{} :cull :{} :fill :{}",
                        match g.invocation.topology {
                            PrimitiveTopology::TriangleList => "triangle-list",
                            PrimitiveTopology::TriangleStrip => "triangle-strip",
                            PrimitiveTopology::LineList => "line-list",
                            PrimitiveTopology::LineStrip => "line-strip",
                            PrimitiveTopology::PointList => "point-list",
                        },
                        match r.front_face {
                            FrontFace::Clockwise => "clockwise",
                            FrontFace::CounterClockwise => "counter-clockwise",
                        },
                        match r.cull {
                            CullMode::None => "none",
                            CullMode::Front => "front",
                            CullMode::Back => "back",
                        },
                        match r.fill {
                            FillMode::Fill => "fill",
                            FillMode::Line => "line",
                            FillMode::Point => "point",
                        }
                    )?;
                    writeln!(
                        out,
                        "  :depth-test :{} :depth-write {} :blend :{} :color-write {})\n",
                        comparison(f.depth_test),
                        if f.depth_write { "t" } else { "nil" },
                        match f.blend {
                            BlendMode::Replace => "replace",
                            BlendMode::SourceOver => "source-over",
                            BlendMode::Add => "add",
                        },
                        if f.color_write { "t" } else { "nil" }
                    )?;
                }
            }
        }
        for entry in &self.entries {
            self.write_entry(&mut out, entry)?;
        }
        Ok(out)
    }

    pub fn parameter_indices(&self, p: usize, stage: Option<usize>) -> Vec<usize> {
        match (&self.interface.pipelines[p], stage) {
            (Pipeline::Compute(c), Some(s)) => {
                let uses = &c.stages[s].uses;
                let mut indices = uses.reads.iter().chain(&uses.writes).copied().collect::<Vec<_>>();
                indices.sort_unstable();
                indices.dedup();
                indices
            }
            _ => (0..self.bindings(p).len()).collect(),
        }
    }

    fn write_parameters(
        &self,
        out: &mut String,
        p: usize,
        stage: Option<usize>,
        format: ShaderFormat,
    ) -> Result<(), HostError> {
        let indices = self.parameter_indices(p, stage);
        write!(out, "  :parameters '(")?;
        for &b in &indices {
            let binding = self.shader_binding(p, stage, b, format)?;
            write!(out, "\n    (argument-{b} {})", self.binding_description(&binding))?;
        }
        write!(out, ")\n  :abi '(")?;
        for &b in &indices {
            let binding = &self.bindings(p)[b];
            match binding {
                Binding::PushConstant { offset, size, .. } => {
                    write!(out, "\n    (argument-{b} :push-constant {offset} {size})")?
                }
                _ => {
                    let Some((set, slot)) = binding.slot() else {
                        return Err(HostError::Invalid("resource binding has no location".into()));
                    };
                    let kind = match binding {
                        Binding::StorageBuffer { .. } => "storage",
                        Binding::Uniform { .. } => "uniform",
                        Binding::Texture { .. } => "sampled-texture",
                        Binding::StorageTexture { .. } => "storage-texture",
                        Binding::Sampler { .. } => "sampler",
                        Binding::PushConstant { .. } => unreachable!(),
                    };
                    write!(out, "\n    (argument-{b} :{kind} {set} {slot})")?;
                }
            }
        }
        write!(out, ")")?;
        Ok(())
    }

    fn binding_description(&self, b: &Binding) -> String {
        match b {
            Binding::StorageBuffer {
                access: a,
                members,
                length,
                ..
            } => {
                if members.is_empty() {
                    match length {
                        Some(BufferLen::Fixed { bytes }) => {
                            format!(":buffer {} :element :u8 :stride 1 :min-bytes {bytes}", access(a))
                        }
                        _ => format!(":buffer {} :element :u8 :stride 1", access(a)),
                    }
                } else {
                    let size = match length {
                        Some(BufferLen::Fixed { bytes }) => *bytes,
                        _ => members
                            .iter()
                            .map(|m| u64::from(m.offset) + u64::from(m.size))
                            .max()
                            .unwrap_or(0),
                    };
                    format!(":buffer {} :layout {}", access(a), layout(size, members))
                }
            }
            Binding::Uniform { size, members, .. } => {
                format!(":buffer :read :layout {}", layout((*size).into(), members))
            }
            Binding::PushConstant { size, .. } => format!(":host-buffer :read :min-bytes {size}"),
            Binding::Texture {
                sample_type,
                view_dimension,
                multisampled,
                ..
            } => format!(
                ":texture :read :dimension {} :format :caller :samples {} :sample-type {}",
                dimension(view_dimension),
                if *multisampled { ":multisampled" } else { "1" },
                match sample_type {
                    TextureSampleType::Float { filterable: true } => ":filterable-float",
                    TextureSampleType::Float { filterable: false } => ":float",
                    TextureSampleType::Sint => ":sint",
                    TextureSampleType::Uint => ":uint",
                    TextureSampleType::Depth => ":depth",
                }
            ),
            Binding::StorageTexture {
                format, access: a, ..
            } => format!(
                ":texture {} :dimension :d2 :format :{} :samples 1",
                access(a),
                format_name(*format)
            ),
            Binding::Sampler { binding_type, .. } => format!(":sampler :kind {}", sampler(binding_type)),
        }
    }

    fn resource_description(&self, id: ResourceId) -> String {
        if self.depth_targets.values().any(|&depth| depth == id) {
            return ":texture :read-write :dimension :d2 :format :depth32float :samples 1".into();
        }
        match self.resource_binding(id) {
            Some(b) => {
                let mut description = self.binding_description(b);
                let passes = &self.interface.frame_graph.passes;
                let readable = passes.iter().any(|p| p.reads.iter().any(|r| r.resource == id.0));
                let writable = passes.iter().any(|p| p.writes.iter().any(|r| r.resource == id.0));
                if writable {
                    description = description.replacen(":read ", ":read-write ", 1);
                }
                if readable {
                    description = description.replacen(":write ", ":read-write ", 1);
                }
                description
            }
            None => match self.interface.frame_graph.resources[id.0].kind {
                FrameResourceKind::Texture | FrameResourceKind::StorageTexture => {
                    ":texture :read-write :dimension :d2 :format :caller :samples 1".into()
                }
                FrameResourceKind::Sampler => ":sampler :kind :filtering".into(),
                FrameResourceKind::StorageBuffer
                | FrameResourceKind::Uniform
                | FrameResourceKind::PushConstant => ":buffer :read :element :u8 :stride 1".into(),
            },
        }
    }

    fn write_entry(&self, out: &mut String, entry: &Entry) -> Result<(), HostError> {
        let name = symbol(&entry.name);
        writeln!(
            out,
            "(define-host-entry '{name}\n  :function 'host-{name}\n  :source-name {}\n  :parameters '(",
            string(&entry.name)
        )?;
        for &id in &entry.inputs {
            writeln!(
                out,
                "    ({} {} :source-name {})",
                resource(id),
                self.resource_description(id),
                string(&self.interface.frame_graph.resources[id.0].name)
            )?;
        }
        for scalar in &entry.scalar_inputs {
            writeln!(out, "    ({scalar} :u32)")?;
        }
        write!(out, "  )\n  :results '(")?;
        let mut source_results =
            self.interface.source_results.iter().filter(|r| r.entry == entry.name).collect::<Vec<_>>();
        source_results.sort_by_key(|r| r.result);
        for (i, &id) in entry.results.iter().enumerate() {
            let source_result = source_results.get(i);
            let source_name = source_result
                .map(|r| r.name.as_str())
                .unwrap_or(&self.interface.frame_graph.resources[id.0].name);
            write!(
                out,
                "\n    ({} {} :source-name {} :ownership {}",
                symbol(source_name),
                self.resource_description(id),
                string(source_name),
                if entry.inputs.contains(&id) { ":borrowed" } else { ":owned" }
            )?;
            if let Some(result) = source_result {
                write!(out, " :value-layout {}", result.layout.to_whl())?;
            }
            if entry.inputs.contains(&id) {
                write!(out, " :alias {}", resource(id))?;
            }
            write!(out, ")")?;
        }
        writeln!(
            out,
            "))\n\n(defun host-{name} ({})",
            entry
                .inputs
                .iter()
                .map(|r| resource(*r))
                .chain(entry.scalar_inputs.iter().cloned())
                .collect::<Vec<_>>()
                .join(" ")
        )?;
        writeln!(out, "  (let* (")?;
        for allocation in &entry.allocations {
            match allocation {
                Allocation::Buffer { resource: r, bytes } => {
                    writeln!(out, "    ({} (gpu-alloc {}))", resource(*r), bytes.to_whl())?
                }
                Allocation::Texture {
                    resource: r,
                    width,
                    height,
                } => {
                    let Some(Binding::StorageTexture { format, .. }) = self.texture_binding(*r) else {
                        return Err(HostError::Invalid("allocated texture has no format".into()));
                    };
                    writeln!(out,"    ({} (gpu-alloc-texture :dimension :d2 :size (list {} {} 1) :format :{} :mip-levels 1 :samples 1 :usage '(:storage :sampled :render-target :copy)))",resource(*r),width.to_whl(),height.to_whl(),format_name(*format))?;
                }
            }
        }
        for p in entry
            .operations
            .iter()
            .map(|op| match op {
                Operation::Dispatch { pipeline, .. }
                | Operation::Draw { pipeline }
                | Operation::Scalar { pipeline, .. } => *pipeline,
            })
            .collect::<std::collections::BTreeSet<_>>()
        {
            for (b, binding) in self.bindings(p).iter().enumerate() {
                let usage = match binding {
                    Binding::Texture { .. } => Some("sampled"),
                    Binding::StorageTexture { .. } => Some("storage"),
                    _ => None,
                };
                if let Some(usage) = usage {
                    let id = self.binding_resource(p, b)?;
                    let view_dimension = match binding {
                        Binding::Texture { view_dimension, .. } => dimension(view_dimension),
                        _ => ":d2",
                    };
                    let layers = match binding {
                        Binding::Texture {
                            view_dimension: TextureViewDimension::D2Array | TextureViewDimension::CubeArray,
                            ..
                        } => format!("(gpu-texture-dimension {} 0 'depth-or-layers)", resource(id)),
                        Binding::Texture {
                            view_dimension: TextureViewDimension::Cube,
                            ..
                        } => "6".into(),
                        _ => "1".into(),
                    };
                    let mips = if usage == "sampled" {
                        format!("(gpu-texture-mip-levels {})", resource(id))
                    } else {
                        "1".into()
                    };
                    writeln!(out,"    (view-{p}-{b} (gpu-texture-view {} :usage :{usage} :dimension {view_dimension} :mip 0 :mip-count {mips} :layer 0 :layer-count {layers}))",resource(id))?;
                }
            }
            if let Pipeline::Graphics(g) = &self.interface.pipelines[p] {
                if g.invocation.fragment_state.depth_test != DepthTest::Disabled {
                    let id = self.depth_target(p)?;
                    writeln!(out,"    (depth-view-{p} (gpu-texture-view {} :usage :render-target :dimension :d2 :mip 0 :mip-count 1 :layer 0 :layer-count 1))",resource(id))?;
                }
                for target in &g.fragment_outputs {
                    let id = self.target_resource(&target.name)?;
                    writeln!(out,"    (target-{p}-{} (gpu-texture-view {} :usage :render-target :dimension :d2 :mip 0 :mip-count 1 :layer 0 :layer-count 1))",target.location,resource(id))?;
                }
            }
        }
        writeln!(out, "  )")?;
        for op in &entry.operations {
            match op{
            Operation::Scalar{pipeline,task}=>writeln!(out,"    {}",self.whl_scalar_task(*pipeline,&self.interface.scalar_tasks[*task])?)?,
            Operation::Dispatch{pipeline:p,stage:s,groups}=>writeln!(out,"    (gpu-dispatch 'kernel-{p}-{s}\n      :groups (list {} {} {})\n      :args (list {}))",groups[0].to_whl(),groups[1].to_whl(),groups[2].to_whl(),self.arguments(*p,Some(*s))?)?,
            Operation::Draw{pipeline:p}=>self.write_draw(out,*p)?,
        }
        }
        for a in &entry.allocations {
            let id = a.resource();
            if !entry.results.contains(&id) {
                writeln!(out, "    (gpu-free {})", resource(id))?;
            }
        }
        let results = entry.results.iter().map(|r| resource(*r)).collect::<Vec<_>>();
        writeln!(
            out,
            "    {}))\n",
            match results.as_slice() {
                [] => "nil".into(),
                [one] => one.clone(),
                _ => format!("(list {})", results.join(" ")),
            }
        )?;
        Ok(())
    }

    pub fn texture_binding(&self, id: ResourceId) -> Option<&Binding> {
        self.interface.frame_graph.resources[id.0].bindings.iter().find_map(|r| {
            let b = &self.bindings(r.pipeline_index)[r.binding_index];
            matches!(b, Binding::StorageTexture { .. }).then_some(b)
        })
    }

    pub fn target_resource(&self, name: &str) -> Result<ResourceId, HostError> {
        let Some(id) = self.interface.frame_graph.resources.iter().position(|r| {
            r.name == name
                && matches!(
                    r.kind,
                    FrameResourceKind::Texture | FrameResourceKind::StorageTexture
                )
        }) else {
            return Err(HostError::Invalid(format!("missing texture {name}")));
        };
        Ok(ResourceId(id))
    }

    fn arguments(&self, p: usize, s: Option<usize>) -> Result<String, HostError> {
        self.parameter_indices(p, s)
            .into_iter()
            .map(|b| match self.bindings(p)[b] {
                Binding::Texture { .. } | Binding::StorageTexture { .. } => Ok(format!("view-{p}-{b}")),
                _ => Ok(resource(self.binding_resource(p, b)?)),
            })
            .collect::<Result<Vec<_>, HostError>>()
            .map(|a| a.join(" "))
    }

    fn write_draw(&self, out: &mut String, p: usize) -> Result<(), HostError> {
        let Pipeline::Graphics(g) = &self.interface.pipelines[p] else {
            return Err(HostError::Invalid("draw references compute".into()));
        };
        writeln!(
            out,
            "    (gpu-draw 'graphics-{p}\n      :args (list {})",
            self.arguments(p, None)?
        )?;
        write!(out, "      :vertices (list")?;
        for a in &g.vertex_inputs {
            let Some(id) = self.interface.frame_graph.resources.iter().position(|r| r.name == a.name)
            else {
                return Err(HostError::Invalid(format!("missing vertex attribute {}", a.name)));
            };
            write!(out, " {}", resource(ResourceId(id)))?;
        }
        write!(out, ")\n      :colors (list")?;
        for target in &g.fragment_outputs {
            write!(
                out,
                " (list {} target-{p}-{} :load :store nil)",
                target.location, target.location
            )?;
        }
        writeln!(out, ")")?;
        if g.invocation.fragment_state.depth_test == DepthTest::Disabled {
            writeln!(out, "      :depth nil")?;
        } else {
            writeln!(out, "      :depth (list depth-view-{p} :load :store nil)")?;
        }
        let r = g.invocation.raster_state;
        match r.viewport {
            Viewport::Target => writeln!(out, "      :viewport :target")?,
            Viewport::Custom {
                origin,
                extent,
                depth,
            } => writeln!(
                out,
                "      :viewport '({:?} {:?} {:?} {:?} {:?} {:?})",
                origin[0], origin[1], extent[0], extent[1], depth[0], depth[1]
            )?,
        }
        match r.scissor {
            Scissor::Target => writeln!(out, "      :scissor :target")?,
            Scissor::Custom { origin, extent } => writeln!(
                out,
                "      :scissor '({} {} {} {})",
                origin[0], origin[1], extent[0], extent[1]
            )?,
        }
        write!(out, "      :draw (list ")?;
        let index_name = |f: &IndexFormat| match f {
            IndexFormat::Uint16 => ":u16",
            IndexFormat::Uint32 => ":u32",
        };
        let count = |c: &DrawCount, id: ResourceId| match c {
            DrawCount::Fixed(n) => n.to_string(),
            DrawCount::BufferLength => format!("count-resource-{}", id.0),
        };
        match &g.invocation.draw {
            DrawCall::Direct {
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            } => write!(
                out,
                ":direct {vertex_count} {instance_count} {first_vertex} {first_instance}"
            )?,
            DrawCall::Indexed {
                indices,
                index_format,
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            } => {
                let id = self.draw_resource(p, indices)?;
                write!(
                    out,
                    ":indexed {} {} {} {instance_count} {first_index} {vertex_offset} {first_instance}",
                    resource(id),
                    index_name(index_format),
                    count(index_count, id)
                )?;
            }
            DrawCall::Indirect {
                commands,
                offset,
                draw_count,
            } => {
                let id = self.draw_resource(p, commands)?;
                write!(
                    out,
                    ":indirect {} {offset} {} 16",
                    resource(id),
                    count(draw_count, id)
                )?;
            }
            DrawCall::IndexedIndirect {
                indices,
                index_format,
                commands,
                offset,
                draw_count,
            } => {
                let id = self.draw_resource(p, commands)?;
                write!(
                    out,
                    ":indexed-indirect {} {} {} {offset} {} 20",
                    resource(self.draw_resource(p, indices)?),
                    index_name(index_format),
                    resource(id),
                    count(draw_count, id)
                )?;
            }
        }
        writeln!(out, "))")?;
        Ok(())
    }
}

impl ResultLayout {
    pub fn to_whl(&self) -> String {
        match self {
            Self::Scalar(scalar) => match scalar {
                ResultScalar::I8 => ":i8",
                ResultScalar::I16 => ":i16",
                ResultScalar::I32 => ":i32",
                ResultScalar::I64 => ":i64",
                ResultScalar::U8 => ":u8",
                ResultScalar::U16 => ":u16",
                ResultScalar::U32 => ":u32",
                ResultScalar::U64 => ":u64",
                ResultScalar::F32 => ":f32",
                ResultScalar::F64 => ":f64",
                ResultScalar::Bool => ":bool",
            }
            .into(),
            Self::Sequence {
                element,
                count,
                stride,
            } => format!(
                "(:sequence :count {count} :stride {stride} :element {})",
                element.to_whl()
            ),
            Self::Array {
                element,
                stride,
                length,
            } => format!(
                "(:array :length {} :stride {stride} :range :caller :element {})",
                length.map(|n| n.to_string()).unwrap_or_else(|| ":dynamic".into()),
                element.to_whl()
            ),
            Self::Record { fields, size } | Self::Tuple { fields, size } => format!(
                "({} :size {size} :fields ({}))",
                if matches!(self, Self::Record { .. }) { ":record" } else { ":tuple" },
                fields
                    .iter()
                    .map(|f| format!(
                        "({} :offset {} :layout {})",
                        string(&f.name),
                        f.offset,
                        f.layout.to_whl()
                    ))
                    .collect::<Vec<_>>()
                    .join(" ")
            ),
            Self::Unsupported(description) => format!("(:unsupported {})", string(description)),
        }
    }
}

fn layout(size: u64, members: &[UniformMember]) -> String {
    format!(
        "(:size {size} :alignment 4 :fields ({}))",
        members
            .iter()
            .map(|m| format!("({} (:bytes {}) {})", symbol(&m.name), m.size, m.offset))
            .collect::<Vec<_>>()
            .join(" ")
    )
}
