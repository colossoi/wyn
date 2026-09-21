use anyhow::{Context, Result};
use wgpu::{
    Adapter, Device, DeviceDescriptor, Features, Instance, InstanceDescriptor, InstanceFlags, Limits,
    PowerPreference, Queue, RequestAdapterOptions, Surface,
};

pub struct GpuContext {
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
    pub surface: Option<Surface<'static>>,
}

pub struct DeviceRequest<'a> {
    pub instance_flags: InstanceFlags,
    pub desired_features: Features,
    pub limits_overlay: Option<Box<dyn FnOnce(&mut Limits, &Adapter) + 'a>>,
    pub surface_target: Option<Box<dyn FnOnce(&Instance) -> Result<Surface<'static>> + 'a>>,
}

impl Default for DeviceRequest<'_> {
    fn default() -> Self {
        Self {
            instance_flags: InstanceFlags::VALIDATION,
            desired_features: Features::FLOAT32_FILTERABLE
                | Features::VERTEX_WRITABLE_STORAGE
                | Features::PUSH_CONSTANTS
                | Features::POLYGON_MODE_LINE
                | Features::POLYGON_MODE_POINT
                | Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
            limits_overlay: None,
            surface_target: None,
        }
    }
}

impl GpuContext {
    pub async fn request(request: DeviceRequest<'_>) -> Result<Self> {
        let instance = Instance::new(&InstanceDescriptor {
            flags: request.instance_flags,
            ..Default::default()
        });
        let surface = request.surface_target.map(|create| create(&instance)).transpose()?;
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: surface.as_ref(),
                force_fallback_adapter: false,
            })
            .await
            .context("request_adapter failed")?;
        let features =
            adapter.features() & request.desired_features & !Features::EXPERIMENTAL_PASSTHROUGH_SHADERS;
        let mut limits = adapter.limits();
        if let Some(overlay) = request.limits_overlay {
            overlay(&mut limits, &adapter);
        }
        if !features.contains(Features::PUSH_CONSTANTS) {
            limits.max_push_constant_size = 0;
        }
        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label: Some("viz"),
                required_features: features,
                required_limits: limits,
                ..Default::default()
            })
            .await
            .context("request_device failed")?;
        Ok(Self {
            adapter,
            device,
            queue,
            surface,
        })
    }
}

#[derive(Clone, Copy)]
pub enum BufferInitSpec {
    Zero,
    Rng,
}

pub fn initialize(bytes: &mut [u8], spec: BufferInitSpec) {
    if matches!(spec, BufferInitSpec::Zero) {
        bytes.fill(0);
        return;
    }
    let mut seed = 0x9e3779b9u32;
    for chunk in bytes.chunks_exact_mut(4) {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        chunk.copy_from_slice(&((seed as f64 / 4294967296.0) as f32).to_le_bytes());
    }
}
