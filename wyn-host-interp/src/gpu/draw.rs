use super::bindings::comparison;
use super::{gpu_error, options, Resource, WgpuBackend};
use crate::{Program, Result, Value};
use wgpu::{
    BlendComponent, BlendFactor, BlendOperation, BlendState, Color, ColorTargetState, ColorWrites,
    DepthStencilState, Face, FragmentState, FrontFace, IndexFormat, LoadOp, MultisampleState, Operations,
    PipelineLayoutDescriptor, PolygonMode, PrimitiveState, PrimitiveTopology, RenderPassColorAttachment,
    RenderPassDepthStencilAttachment, RenderPassDescriptor, RenderPipelineDescriptor, ShaderStages,
    StoreOp, TextureFormat, TextureView, VertexAttribute, VertexBufferLayout, VertexFormat, VertexState,
    VertexStepMode,
};

fn topology(name: &str) -> Result<PrimitiveTopology> {
    Ok(match name {
        ":triangle-list" => PrimitiveTopology::TriangleList,
        ":triangle-strip" => PrimitiveTopology::TriangleStrip,
        ":line-list" => PrimitiveTopology::LineList,
        ":line-strip" => PrimitiveTopology::LineStrip,
        ":point-list" => PrimitiveTopology::PointList,
        _ => return Err(gpu_error(format!("invalid topology {name}"))),
    })
}

fn vertex_format(value: &Value) -> Result<VertexFormat> {
    let (scalar, count) = match value {
        Value::Symbol(name) => (name.as_str(), 1),
        _ => {
            let [kind, scalar, count] = value.list()? else {
                return Err(gpu_error("invalid vertex format"));
            };
            if kind.text()? != ":vector" {
                return Err(gpu_error("invalid vertex type"));
            }
            (scalar.text()?, count.u32()?)
        }
    };
    Ok(match (scalar, count) {
        (":f32", 1) => VertexFormat::Float32,
        (":f32", 2) => VertexFormat::Float32x2,
        (":f32", 3) => VertexFormat::Float32x3,
        (":f32", 4) => VertexFormat::Float32x4,
        (":i32", 1) => VertexFormat::Sint32,
        (":i32", 2) => VertexFormat::Sint32x2,
        (":i32", 3) => VertexFormat::Sint32x3,
        (":i32", 4) => VertexFormat::Sint32x4,
        (":u32", 1) => VertexFormat::Uint32,
        (":u32", 2) => VertexFormat::Uint32x2,
        (":u32", 3) => VertexFormat::Uint32x3,
        (":u32", 4) => VertexFormat::Uint32x4,
        _ => return Err(gpu_error("unsupported vertex format")),
    })
}

fn store(value: &Value) -> Result<StoreOp> {
    match value.text()? {
        ":store" => Ok(StoreOp::Store),
        ":discard" => Ok(StoreOp::Discard),
        other => Err(gpu_error(format!("invalid store operation {other}"))),
    }
}

fn load<T>(value: &Value, clear: T) -> Result<LoadOp<T>> {
    match value.text()? {
        ":load" => Ok(LoadOp::Load),
        ":clear" | ":discard" => Ok(LoadOp::Clear(clear)),
        other => Err(gpu_error(format!("invalid load operation {other}"))),
    }
}

fn end(first: u32, count: u32) -> Result<u32> {
    first.checked_add(count).ok_or_else(|| gpu_error("draw range overflow"))
}

impl WgpuBackend {
    fn attachment(&self, value: &Value) -> Result<(TextureView, TextureFormat, u32, u32, u32)> {
        let Resource::View {
            view, width, height, ..
        } = self.resource(value)?
        else {
            return Err(gpu_error("attachment must be a texture view"));
        };
        let texture = self.texture(value)?;
        Ok((
            view.clone(),
            texture.format(),
            *width,
            *height,
            texture.sample_count(),
        ))
    }

    pub(super) fn draw(&mut self, program: &Program, args: &[Value]) -> Result<Value> {
        let o = options(
            args,
            1,
            &[
                ":args",
                ":vertices",
                ":colors",
                ":depth",
                ":viewport",
                ":scissor",
                ":draw",
            ],
        )?;
        let name = args[0].text()?;
        let Some(graphics) = program.graphics.get(name) else {
            return Err(gpu_error(format!("unknown graphics declaration {name}")));
        };
        let state = &graphics.options;
        let color_specs = o.get(":colors")?.list()?;
        let mut colors = Vec::new();
        let mut extent = None;
        let samples = state.u32(":samples")?;
        for spec in color_specs {
            let [slot, view, loading, storing, clear] = spec.list()? else {
                return Err(gpu_error("invalid color attachment"));
            };
            let (view, format, width, height, count) = self.attachment(view)?;
            if count != samples || extent.is_some_and(|size| size != (width, height)) {
                return Err(gpu_error("attachment sizes or samples do not match"));
            }
            extent = Some((width, height));
            let color = if matches!(clear, Value::Nil) {
                Color::TRANSPARENT
            } else {
                let [r, g, b, a] = clear.list()? else {
                    return Err(gpu_error("color clear needs four components"));
                };
                Color {
                    r: r.number()?.real(),
                    g: g.number()?.real(),
                    b: b.number()?.real(),
                    a: a.number()?.real(),
                }
            };
            colors.push((
                slot.u32()?,
                view,
                format,
                Operations {
                    load: load(loading, color)?,
                    store: store(storing)?,
                },
            ));
        }
        let depth = if matches!(o.get(":depth")?, Value::Nil) {
            None
        } else {
            let [view, loading, storing, clear] = o.get(":depth")?.list()? else {
                return Err(gpu_error("invalid depth attachment"));
            };
            let (view, format, width, height, count) = self.attachment(view)?;
            if count != samples
                || extent.is_some_and(|size| size != (width, height))
                || format != TextureFormat::Depth32Float
            {
                return Err(gpu_error("incompatible depth attachment"));
            }
            extent = Some((width, height));
            let clear = if matches!(clear, Value::Nil) { 1.0 } else { clear.number()?.real() as f32 };
            if !(0.0..=1.0).contains(&clear) {
                return Err(gpu_error("depth clear outside [0,1]"));
            }
            Some((
                view,
                Operations {
                    load: load(loading, clear)?,
                    store: store(storing)?,
                },
            ))
        };
        let Some((width, height)) = extent else {
            return Err(gpu_error("draw has no attachments"));
        };
        let outputs = state.get(":color-outputs")?.list()?;
        if outputs.len() != colors.len() {
            return Err(gpu_error("wrong number of color attachments"));
        }
        let color_count = colors.iter().map(|(slot, ..)| *slot as usize + 1).max().unwrap_or(0);
        if color_count > self.device.limits().max_color_attachments as usize {
            return Err(gpu_error("too many color attachments"));
        }
        let mut formats = vec![None; color_count];
        for (slot, _, format, _) in &colors {
            if formats[*slot as usize].replace(*format).is_some() {
                return Err(gpu_error("duplicate color attachment"));
            }
            if !outputs.iter().any(|output| {
                output.list().is_ok_and(|parts| {
                    parts.first().is_some_and(|v| v.u32().is_ok_and(|value| value == *slot))
                })
            }) {
                return Err(gpu_error("undeclared color attachment"));
            }
        }
        let attributes = state
            .get(":vertex-inputs")?
            .list()?
            .iter()
            .map(|spec| {
                let [location, _, format, stride, offset, step] = spec.list()? else {
                    return Err(gpu_error("invalid vertex input"));
                };
                let step = match step.text()? {
                    ":vertex" => VertexStepMode::Vertex,
                    ":instance" => VertexStepMode::Instance,
                    _ => return Err(gpu_error("invalid vertex step")),
                };
                Ok((
                    VertexAttribute {
                        shader_location: location.u32()?,
                        offset: offset.u64()?,
                        format: vertex_format(format)?,
                    },
                    stride.u64()?,
                    step,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let vertex_values = o.get(":vertices")?.list()?;
        if vertex_values.len() != attributes.len() {
            return Err(gpu_error("wrong vertex buffer count"));
        }
        let vertices = vertex_values.iter().map(|v| self.buffer(v).cloned()).collect::<Result<Vec<_>>>()?;
        let key = format!("{name}:{formats:?}:{samples}:{:?}", self.topology);
        if !self.renders.contains_key(&key) {
            let (layouts, pushes) = self.layouts(graphics, ShaderStages::VERTEX_FRAGMENT)?;
            let references = layouts.iter().collect::<Vec<_>>();
            let layout = self.device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some(name),
                bind_group_layouts: &references,
                push_constant_ranges: &pushes,
            });
            let [vertex_module, vertex_name] = state.get(":vertex")?.list()? else {
                return Err(gpu_error("missing vertex stage"));
            };
            let Some(vertex_module) = self.modules.get(vertex_module.text()?) else {
                return Err(gpu_error("missing vertex module"));
            };
            let blend = match state.text(":blend")? {
                ":replace" => None,
                ":source-over" => Some(BlendState::ALPHA_BLENDING),
                ":add" => Some(BlendState {
                    color: BlendComponent {
                        src_factor: BlendFactor::One,
                        dst_factor: BlendFactor::One,
                        operation: BlendOperation::Add,
                    },
                    alpha: BlendComponent {
                        src_factor: BlendFactor::One,
                        dst_factor: BlendFactor::One,
                        operation: BlendOperation::Add,
                    },
                }),
                _ => return Err(gpu_error("invalid blend mode")),
            };
            let targets = formats
                .iter()
                .map(|format| {
                    format.map(|format| ColorTargetState {
                        format,
                        blend,
                        write_mask: if state.get(":color-write").is_ok_and(Value::truth) {
                            ColorWrites::ALL
                        } else {
                            ColorWrites::empty()
                        },
                    })
                })
                .collect::<Vec<_>>();
            let fragment = if matches!(state.get(":fragment")?, Value::Nil) {
                None
            } else {
                let [module, entry] = state.get(":fragment")?.list()? else {
                    return Err(gpu_error("invalid fragment stage"));
                };
                let Some(module) = self.modules.get(module.text()?) else {
                    return Err(gpu_error("missing fragment module"));
                };
                Some(FragmentState {
                    module,
                    entry_point: Some(entry.text()?),
                    compilation_options: Default::default(),
                    targets: &targets,
                })
            };
            let depth_stencil = if state.text(":depth-test")? == ":disabled" {
                if state.get(":depth-write")?.truth() {
                    return Err(gpu_error("depth writes require depth testing"));
                }
                None
            } else {
                if depth.is_none() {
                    return Err(gpu_error("depth test needs an attachment"));
                }
                Some(DepthStencilState {
                    format: TextureFormat::Depth32Float,
                    depth_write_enabled: state.get(":depth-write")?.truth(),
                    depth_compare: comparison(state.text(":depth-test")?)?,
                    stencil: Default::default(),
                    bias: Default::default(),
                })
            };
            let buffers = attributes
                .iter()
                .map(|(attribute, stride, step)| VertexBufferLayout {
                    array_stride: *stride,
                    step_mode: *step,
                    attributes: std::slice::from_ref(attribute),
                })
                .collect::<Vec<_>>();
            let pipeline = self.device.create_render_pipeline(&RenderPipelineDescriptor {
                label: Some(name),
                layout: Some(&layout),
                vertex: VertexState {
                    module: vertex_module,
                    entry_point: Some(vertex_name.text()?),
                    compilation_options: Default::default(),
                    buffers: &buffers,
                },
                fragment,
                primitive: PrimitiveState {
                    topology: self.topology.unwrap_or(topology(state.text(":topology")?)?),
                    front_face: match state.text(":front-face")? {
                        ":clockwise" => FrontFace::Cw,
                        ":counter-clockwise" => FrontFace::Ccw,
                        _ => return Err(gpu_error("invalid winding")),
                    },
                    cull_mode: match state.text(":cull")? {
                        ":none" => None,
                        ":front" => Some(Face::Front),
                        ":back" => Some(Face::Back),
                        _ => return Err(gpu_error("invalid cull mode")),
                    },
                    polygon_mode: match state.text(":fill")? {
                        ":fill" => PolygonMode::Fill,
                        ":line" => PolygonMode::Line,
                        ":point" => PolygonMode::Point,
                        _ => return Err(gpu_error("invalid fill mode")),
                    },
                    ..Default::default()
                },
                depth_stencil,
                multisample: MultisampleState {
                    count: samples,
                    ..Default::default()
                },
                multiview: None,
                cache: None,
            });
            self.renders.insert(key.clone(), (pipeline, layouts));
        }
        let (pipeline, layouts) = &self.renders[&key];
        let (groups, pushes) = self.bind_groups(graphics, o.get(":args")?.list()?, layouts)?;
        let mut attachments = vec![None; color_count];
        for (slot, view, _, ops) in &colors {
            attachments[*slot as usize] = Some(RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: *ops,
            });
        }
        let depth_attachment = depth.as_ref().map(|(view, ops)| RenderPassDepthStencilAttachment {
            view,
            depth_ops: Some(*ops),
            stencil_ops: None,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some(name),
                color_attachments: &attachments,
                depth_stencil_attachment: depth_attachment,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(pipeline);
            for (set, group) in groups.iter().enumerate() {
                pass.set_bind_group(set as u32, group, &[]);
            }
            for (offset, bytes) in pushes {
                pass.set_push_constants(ShaderStages::VERTEX_FRAGMENT, offset, &bytes);
            }
            for (slot, buffer) in vertices.iter().enumerate() {
                pass.set_vertex_buffer(slot as u32, buffer.slice(..));
            }
            if o.get(":viewport")? != &Value::Symbol(":target".into()) {
                let values = o
                    .get(":viewport")?
                    .list()?
                    .iter()
                    .map(|v| Ok(v.number()?.real() as f32))
                    .collect::<Result<Vec<_>>>()?;
                let [x, y, w, h, min, max] = values.as_slice() else {
                    return Err(gpu_error("viewport needs six values"));
                };
                pass.set_viewport(*x, *y, *w, *h, *min, *max);
            }
            if o.get(":scissor")? != &Value::Symbol(":target".into()) {
                let values = o
                    .get(":scissor")?
                    .list()?
                    .iter()
                    .map(|v| v.number()?.integer())
                    .collect::<Result<Vec<_>>>()?;
                let [x, y, w, h] = values.as_slice() else {
                    return Err(gpu_error("scissor needs four values"));
                };
                if *w < 0 || *h < 0 {
                    return Err(gpu_error("negative scissor extent"));
                }
                let left = (*x).clamp(0, width.into());
                let top = (*y).clamp(0, height.into());
                let right = (x + w).clamp(left, width.into());
                let bottom = (y + h).clamp(top, height.into());
                if right == left || bottom == top {
                    return Ok(Value::Nil);
                }
                pass.set_scissor_rect(
                    left as u32,
                    top as u32,
                    (right - left) as u32,
                    (bottom - top) as u32,
                );
            }
            let draw = o.get(":draw")?.list()?;
            if let Some(indices) = &self.index_buffer {
                pass.set_index_buffer(self.buffer(indices)?.slice(..), IndexFormat::Uint32);
                pass.draw_indexed(
                    0..u32::try_from(self.buffer_size(indices)? / 4).map_err(gpu_error)?,
                    0,
                    0..1,
                );
            } else {
                match draw {
                    [kind, count, instances, first, first_instance] if kind.text()? == ":direct" => {
                        let first = first.u32()?;
                        let first_instance = first_instance.u32()?;
                        pass.draw(
                            first..end(first, self.vertex_count.unwrap_or(count.u32()?))?,
                            first_instance..end(first_instance, instances.u32()?)?,
                        );
                    }
                    [kind, indices, format, count, instances, first, base, first_instance]
                        if kind.text()? == ":indexed" =>
                    {
                        let format = match format.text()? {
                            ":u16" => IndexFormat::Uint16,
                            ":u32" => IndexFormat::Uint32,
                            _ => return Err(gpu_error("invalid index type")),
                        };
                        pass.set_index_buffer(self.buffer(indices)?.slice(..), format);
                        let first = first.u32()?;
                        let first_instance = first_instance.u32()?;
                        pass.draw_indexed(
                            first..end(first, count.u32()?)?,
                            i32::try_from(base.number()?.integer()?).map_err(gpu_error)?,
                            first_instance..end(first_instance, instances.u32()?)?,
                        );
                    }
                    [kind, commands, offset, count, stride] if kind.text()? == ":indirect" => {
                        let offset = offset.u64()?;
                        let stride = stride.u64()?;
                        for i in 0..count.u32()? {
                            let offset = u64::from(i)
                                .checked_mul(stride)
                                .and_then(|i| offset.checked_add(i))
                                .ok_or_else(|| gpu_error("indirect range overflow"))?;
                            pass.draw_indirect(self.buffer(commands)?, offset);
                        }
                    }
                    [kind, indices, format, commands, offset, count, stride]
                        if kind.text()? == ":indexed-indirect" =>
                    {
                        let format = match format.text()? {
                            ":u16" => IndexFormat::Uint16,
                            ":u32" => IndexFormat::Uint32,
                            _ => return Err(gpu_error("invalid index type")),
                        };
                        pass.set_index_buffer(self.buffer(indices)?.slice(..), format);
                        let offset = offset.u64()?;
                        let stride = stride.u64()?;
                        for i in 0..count.u32()? {
                            let offset = u64::from(i)
                                .checked_mul(stride)
                                .and_then(|i| offset.checked_add(i))
                                .ok_or_else(|| gpu_error("indirect range overflow"))?;
                            pass.draw_indexed_indirect(self.buffer(commands)?, offset);
                        }
                    }
                    _ => return Err(gpu_error("invalid draw command")),
                }
            }
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(Value::Nil)
    }
}
