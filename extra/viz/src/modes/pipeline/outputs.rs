use super::Runner;
use anyhow::{anyhow, Context, Result};
use half::f16;
use serde_json::{json, Map, Value as Json};
use std::fs;
use std::path::Path;
use wgpu::{
    BufferDescriptor, BufferUsages, MapMode, PollType, TexelCopyBufferInfo, TexelCopyBufferLayout,
    TextureFormat, COPY_BYTES_PER_ROW_ALIGNMENT,
};
use wyn_host_interp::{Options, Value};

fn take<const N: usize>(bytes: &[u8]) -> Result<[u8; N]> {
    let Some(bytes) = bytes.get(..N) else {
        return Err(anyhow!("result is smaller than its layout"));
    };
    Ok(bytes.try_into()?)
}

fn scalar(name: &str, bytes: &[u8]) -> Result<Json> {
    Ok(match name {
        ":i8" => json!(i8::from_le_bytes(take(bytes)?)),
        ":u8" => json!(u8::from_le_bytes(take(bytes)?)),
        ":i16" => json!(i16::from_le_bytes(take(bytes)?)),
        ":u16" => json!(u16::from_le_bytes(take(bytes)?)),
        ":i32" => json!(i32::from_le_bytes(take(bytes)?)),
        ":u32" => json!(u32::from_le_bytes(take(bytes)?)),
        ":i64" => json!(i64::from_le_bytes(take(bytes)?)),
        ":u64" => json!(u64::from_le_bytes(take(bytes)?)),
        ":f32" => float_json(f32::from_le_bytes(take(bytes)?) as f64),
        ":f64" => float_json(f64::from_le_bytes(take(bytes)?)),
        ":bool" => json!(u32::from_le_bytes(take(bytes)?) != 0),
        other => return Err(anyhow!("unsupported result scalar {other}")),
    })
}

fn float_json(value: f64) -> Json {
    if value.is_finite() {
        json!(value)
    } else {
        json!(value.to_string())
    }
}

fn decode(layout: &Value, bytes: &[u8], depth: usize) -> Result<Json> {
    if depth > 64 {
        return Err(anyhow!("result layout is too deeply nested"));
    }
    if let Value::Symbol(name) = layout {
        return scalar(name, bytes);
    }
    let Some((kind, options)) = layout.list()?.split_first() else {
        return Err(anyhow!("empty result layout"));
    };
    match kind.text()? {
        ":sequence" | ":array" => {
            let options = Options::parse(options, &[":count", ":length", ":stride", ":range", ":element"])?;
            let stride = usize::try_from(options.get(":stride")?.u64()?)?;
            if stride == 0 {
                return Err(anyhow!("zero result element stride"));
            }
            let length = if kind.text()? == ":sequence" {
                options.get(":count")?
            } else {
                options.get(":length")?
            };
            let dynamic = length == &Value::Symbol(":dynamic".into());
            let array = kind.text()? == ":array";
            let declared_count = if dynamic { None } else { Some(usize::try_from(length.u64()?)?) };
            if declared_count.is_some_and(|count| count > bytes.len() / stride) {
                return Err(anyhow!("result length exceeds its backing buffer"));
            }
            let count = if array { bytes.len() / stride } else { usize::try_from(length.u64()?)? };
            let Some(size) = count.checked_mul(stride) else {
                return Err(anyhow!("result length overflow"));
            };
            let Some(bytes) = bytes.get(..size) else {
                return Err(anyhow!("result length exceeds its backing buffer"));
            };
            let values = bytes
                .chunks_exact(stride)
                .map(|part| decode(options.get(":element")?, part, depth + 1))
                .collect::<Result<Vec<_>>>()?;
            if array {
                let mut output = json!({"backing_buffer":values,"byte_length":bytes.len()});
                if let Some(count) = declared_count {
                    output["declared_count"] = json!(count);
                }
                Ok(output)
            } else {
                Ok(Json::Array(values))
            }
        }
        ":record" | ":tuple" => {
            let options = Options::parse(options, &[":size", ":fields"])?;
            let Some(bytes) = bytes.get(..usize::try_from(options.get(":size")?.u64()?)?) else {
                return Err(anyhow!("result record exceeds buffer"));
            };
            let mut fields = Map::new();
            let mut items = Vec::new();
            for field in options.get(":fields")?.list()? {
                let Some((name, rest)) = field.list()?.split_first() else {
                    return Err(anyhow!("empty result field"));
                };
                let field = Options::parse(rest, &[":offset", ":layout"])?;
                let Some(bytes) = bytes.get(usize::try_from(field.get(":offset")?.u64()?)?..) else {
                    return Err(anyhow!("result field exceeds buffer"));
                };
                let value = decode(field.get(":layout")?, bytes, depth + 1)?;
                if kind.text()? == ":tuple" {
                    items.push(value);
                } else {
                    fields.insert(name.text()?.into(), value);
                }
            }
            if kind.text()? == ":tuple" {
                Ok(Json::Array(items))
            } else {
                Ok(Json::Object(fields))
            }
        }
        other => Err(anyhow!("unsupported result layout {other}")),
    }
}

impl Runner {
    pub fn output(&mut self, print_default: bool) -> Result<()> {
        for name in self.spec.outputs.keys() {
            if !self.entry.results.iter().any(|p| p.source_name() == name) {
                return Err(anyhow!(
                    "unknown output {name}; available: {}",
                    self.entry.results.iter().map(|p| p.source_name()).collect::<Vec<_>>().join(", ")
                ));
            }
        }
        let mut printed = Map::new();
        for (parameter, value) in self.entry.results.iter().zip(&self.results) {
            let path = self.spec.outputs.get(parameter.source_name());
            if path.is_none() && (!print_default || !self.spec.outputs.is_empty()) {
                continue;
            }
            let output = match parameter.kind.as_str() {
                ":buffer" | ":host-buffer" => {
                    let bytes = self.backend.read_buffer(value, 0, self.backend.buffer_size(value)?)?;
                    if let Some(path) = path.filter(|p| p.extension().is_some_and(|e| e == "bin")) {
                        fs::write(path, &bytes)?;
                        continue;
                    }
                    if let Some(layout) = parameter.options.optional(":value-layout") {
                        decode(layout, &bytes, 0)?
                    } else {
                        json!({"bytes":bytes})
                    }
                }
                ":texture" => {
                    if let Some(path) = path {
                        self.dump_texture(value, path)?;
                        continue;
                    }
                    let texture = self.backend.texture(value)?;
                    json!({"texture":{"width":texture.width(),"height":texture.height(),"format":format!("{:?}",texture.format())}})
                }
                _ => match value {
                    Value::Number(number) => {
                        if let Ok(integer) = number.integer() {
                            serde_json::from_str(&integer.to_string())?
                        } else {
                            float_json(number.real())
                        }
                    }
                    Value::True => json!(true),
                    Value::Nil => json!(false),
                    _ => return Err(anyhow!("unsupported host result")),
                },
            };
            if let Some(path) = path {
                fs::write(path, serde_json::to_string_pretty(&output)?)
                    .with_context(|| format!("writing {}", path.display()))?;
            } else {
                printed.insert(parameter.source_name().into(), output);
            }
        }
        if !printed.is_empty() {
            println!("{}", serde_json::to_string_pretty(&printed)?);
        }
        for (name, path) in &self.spec.opts.dump_textures {
            let value = self
                .entry
                .results
                .iter()
                .zip(&self.results)
                .find(|(p, _)| p.source_name() == name)
                .or_else(|| {
                    self.entry.parameters.iter().zip(&self.arguments).find(|(p, _)| p.source_name() == name)
                });
            let Some((_, value)) = value else {
                return Err(anyhow!("unknown texture {name}"));
            };
            self.dump_texture(value, path)?;
        }
        Ok(())
    }

    fn dump_texture(&self, value: &Value, path: &Path) -> Result<()> {
        let texture = self.backend.texture(value)?;
        let format = texture.format();
        let texel_size = match format {
            TextureFormat::Rgba8Unorm
            | TextureFormat::Rgba8UnormSrgb
            | TextureFormat::Bgra8Unorm
            | TextureFormat::Bgra8UnormSrgb
            | TextureFormat::R32Float => 4,
            TextureFormat::Rgba16Float => 8,
            TextureFormat::Rgba32Float => 16,
            _ => return Err(anyhow!("cannot dump {format:?} as PNG")),
        };
        if texture.sample_count() != 1 {
            return Err(anyhow!("PNG readback needs a single-sample texture"));
        }
        let row = texture.width() * texel_size;
        let pitch = row.div_ceil(COPY_BYTES_PER_ROW_ALIGNMENT) * COPY_BYTES_PER_ROW_ALIGNMENT;
        let size = u64::from(pitch) * u64::from(texture.height());
        let buffer = self.backend.device.create_buffer(&BufferDescriptor {
            label: Some("PNG readback"),
            size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.backend.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            TexelCopyBufferInfo {
                buffer: &buffer,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(pitch),
                    rows_per_image: Some(texture.height()),
                },
            },
            texture.size(),
        );
        self.backend.queue.submit(Some(encoder.finish()));
        let slice = buffer.slice(..);
        let (send, receive) = std::sync::mpsc::channel();
        slice.map_async(MapMode::Read, move |result| {
            if send.send(result).is_err() { /* The waiting caller has already returned an error. */ }
        });
        self.backend.device.poll(PollType::wait_indefinitely())?;
        receive.recv()??;
        let mapped = slice.get_mapped_range();
        let mut rgba = Vec::with_capacity((texture.width() as usize) * (texture.height() as usize) * 4);
        for row in mapped.chunks_exact(pitch as usize) {
            for pixel in
                row[..texture.width() as usize * texel_size as usize].chunks_exact(texel_size as usize)
            {
                match format {
                    TextureFormat::Rgba8Unorm | TextureFormat::Rgba8UnormSrgb => {
                        rgba.extend_from_slice(pixel)
                    }
                    TextureFormat::Bgra8Unorm | TextureFormat::Bgra8UnormSrgb => {
                        rgba.extend_from_slice(&[pixel[2], pixel[1], pixel[0], pixel[3]])
                    }
                    TextureFormat::R32Float => {
                        let v = (f32::from_le_bytes(take(pixel)?).clamp(0.0, 1.0) * 255.0).round() as u8;
                        rgba.extend_from_slice(&[v, v, v, 255]);
                    }
                    TextureFormat::Rgba16Float => {
                        for part in pixel.chunks_exact(2) {
                            rgba.push(
                                (f16::from_le_bytes(take(part)?).to_f32().clamp(0.0, 1.0) * 255.0).round()
                                    as u8,
                            );
                        }
                    }
                    TextureFormat::Rgba32Float => {
                        for part in pixel.chunks_exact(4) {
                            rgba.push(
                                (f32::from_le_bytes(take(part)?).clamp(0.0, 1.0) * 255.0).round() as u8
                            );
                        }
                    }
                    _ => return Err(anyhow!("unsupported PNG texture format")),
                }
            }
        }
        drop(mapped);
        buffer.unmap();
        image::save_buffer(
            path,
            &rgba,
            texture.width(),
            texture.height(),
            image::ColorType::Rgba8,
        )?;
        Ok(())
    }
}

#[cfg(test)]
#[path = "outputs_tests.rs"]
mod tests;
