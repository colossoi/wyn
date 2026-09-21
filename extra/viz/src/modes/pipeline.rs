use wgpu::{Device, InstanceFlags, PresentMode, PrimitiveTopology, Queue, Texture};
mod inputs;
mod outputs;

use anyhow::{anyhow, Context, Result};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::PathBuf;
use wyn_host_interp::gpu::WgpuBackend;
use wyn_host_interp::{Backend, Entry, Number, Program, Value};

use crate::app::App;
use crate::config::FeedbackSpec;
use crate::gpu::{BufferInitSpec, DeviceRequest, GpuContext};
use crate::specs::{PushConstantSpec, UniformSpec};

pub struct InteractiveOpts {
    pub storage_dir: Option<PathBuf>,
    pub buffer_inits: HashMap<String, BufferInitSpec>,
    pub storage_bytes: HashMap<String, u64>,
    pub framebuffers: HashMap<String, FramebufferFormat>,
    pub index_buffer: Option<PathBuf>,
    pub present_mode: PresentMode,
    pub validate: bool,
    pub size: Option<(u32, u32)>,
    pub max_frames: Option<u32>,
    pub vertex_count: Option<u32>,
    pub topology: Option<PrimitiveTopology>,
    pub images: HashMap<String, PathBuf>,
    pub dump_textures: HashMap<String, PathBuf>,
    pub uniform_values: Vec<UniformSpec>,
    pub entry: Option<String>,
    pub headless: bool,
}

impl Default for InteractiveOpts {
    fn default() -> Self {
        Self {
            storage_dir: None,
            buffer_inits: HashMap::new(),
            storage_bytes: HashMap::new(),
            framebuffers: HashMap::new(),
            index_buffer: None,
            present_mode: PresentMode::Fifo,
            validate: true,
            size: None,
            max_frames: None,
            vertex_count: None,
            topology: None,
            images: HashMap::new(),
            dump_textures: HashMap::new(),
            uniform_values: Vec::new(),
            entry: None,
            headless: false,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub enum FramebufferFormat {
    #[default]
    Vec4F32,
}

impl FramebufferFormat {
    pub fn bytes_per_texel(self) -> u64 {
        16
    }
}
impl std::str::FromStr for FramebufferFormat {
    type Err = anyhow::Error;
    fn from_str(value: &str) -> Result<Self> {
        if value == "vec4f32" {
            Ok(Self::Vec4F32)
        } else {
            Err(anyhow!("unsupported framebuffer format {value}"))
        }
    }
}

pub struct RunSpec {
    pub program: Program,
    pub base: PathBuf,
    pub sources: Option<BTreeMap<String, Vec<u8>>>,
    pub inputs: HashMap<String, PathBuf>,
    pub outputs: HashMap<String, PathBuf>,
    pub constants: Vec<PushConstantSpec>,
    pub dispatch: BTreeMap<String, [u32; 3]>,
    pub feedback: Vec<FeedbackSpec>,
    pub opts: InteractiveOpts,
    pub verbose: bool,
}

pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub time: f32,
    pub delta: f32,
    pub index: u32,
    pub mouse: [f32; 4],
    pub keyboard: [u8; 768],
}

pub struct Runner {
    pub backend: WgpuBackend,
    spec: RunSpec,
    entry: Entry,
    arguments: Vec<Value>,
    updates: Vec<inputs::Update>,
    targets: Vec<usize>,
    present: Option<usize>,
    pub results: Vec<Value>,
    width: u32,
    height: u32,
}

pub async fn run_pipeline(
    host_path: PathBuf,
    inputs: HashMap<String, PathBuf>,
    outputs: HashMap<String, PathBuf>,
    constants: &[PushConstantSpec],
    dispatch: &HashMap<String, (u32, u32, u32)>,
    feedback: &[FeedbackSpec],
    opts: InteractiveOpts,
    verbose: bool,
) -> Result<()> {
    let source =
        fs::read_to_string(&host_path).with_context(|| format!("reading {}", host_path.display()))?;
    let program = Program::parse(&source)?;
    let base = host_path.parent().unwrap_or_else(|| std::path::Path::new(".")).to_path_buf();
    let graphical = !program.graphics.is_empty() && !opts.headless;
    let spec = RunSpec {
        program,
        base,
        sources: None,
        inputs,
        outputs,
        constants: constants.to_vec(),
        dispatch: dispatch.iter().map(|(name, (x, y, z))| (name.clone(), [*x, *y, *z])).collect(),
        feedback: feedback.to_vec(),
        opts,
        verbose,
    };
    if graphical {
        return App::run(spec);
    }
    let context = GpuContext::request(DeviceRequest {
        instance_flags: if spec.opts.validate { InstanceFlags::VALIDATION } else { InstanceFlags::empty() },
        ..Default::default()
    })
    .await?;
    if spec.verbose {
        eprintln!("GPU: {}", context.adapter.get_info().name);
    }
    let (width, height) = spec.opts.size.unwrap_or((800, 600));
    let count = spec.opts.max_frames.unwrap_or(1);
    if count == 0 {
        return Err(anyhow!("--max-frames must be positive"));
    }
    let mut runner = Runner::new(spec, context.device, context.queue, width, height)?;
    for index in 0..count {
        runner.run_frame(
            &Frame {
                width,
                height,
                time: index as f32 / 60.0,
                delta: 1.0 / 60.0,
                index,
                mouse: [0.0; 4],
                keyboard: [0; 768],
            },
            None,
        )?;
    }
    runner.output(true)
}

impl Runner {
    pub fn new(spec: RunSpec, device: Device, queue: Queue, width: u32, height: u32) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(anyhow!("target dimensions must be positive"));
        }
        let entry = if let Some(name) = &spec.opts.entry {
            spec.program.entry(name)?.clone()
        } else if spec.program.entries.len() == 1 {
            let Some(entry) = spec.program.entries.values().next() else {
                return Err(anyhow!("no host entry"));
            };
            entry.clone()
        } else {
            return Err(anyhow!(
                "select --entry from: {}",
                spec.program
                    .entries
                    .values()
                    .map(|e| e.source_name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        };
        let mut backend = match &spec.sources {
            Some(sources) => WgpuBackend::with_sources(device, queue, &spec.program, sources)?,
            None => WgpuBackend::new(device, queue, &spec.program, &spec.base)?,
        };
        backend.dispatch_overrides = spec.dispatch.clone();
        backend.vertex_count = spec.opts.vertex_count;
        backend.topology = spec.opts.topology;
        let mut runner = Self {
            backend,
            spec,
            entry,
            arguments: Vec::new(),
            updates: Vec::new(),
            targets: Vec::new(),
            present: None,
            results: Vec::new(),
            width,
            height,
        };
        runner.prepare()?;
        Ok(runner)
    }

    pub fn run_frame(&mut self, frame: &Frame, screen: Option<Texture>) -> Result<()> {
        if frame.width != self.width || frame.height != self.height {
            self.width = frame.width;
            self.height = frame.height;
            self.resize_targets()?;
        }
        if let (Some(index), Some(texture)) = (self.present, screen) {
            self.arguments[index] = self.backend.import_texture(texture);
        }
        self.update(frame)?;
        let result = self.spec.program.run(&self.entry.source_name, &self.arguments, &mut self.backend)?;
        self.results = match self.entry.results.len() {
            0 => Vec::new(),
            1 => vec![result],
            _ => result.list()?.to_vec(),
        };
        for feedback in &self.spec.feedback {
            let Some(value) = self.results.get(feedback.result) else {
                return Err(anyhow!("unknown feedback result {}", feedback.result));
            };
            match self.destination(&feedback.input)? {
                inputs::Destination::Resource(index) => self.arguments[index] = value.clone(),
                inputs::Destination::Field {
                    argument,
                    offset,
                    size,
                } => {
                    self.backend.call(
                        &self.spec.program,
                        "gpu-copy",
                        &[
                            self.arguments[argument].clone(),
                            Value::Number(Number::U64(offset)),
                            value.clone(),
                            Value::Number(Number::U64(0)),
                            Value::Number(Number::U64(size)),
                        ],
                    )?;
                }
            }
        }
        let mut retained = self.arguments.clone();
        retained.extend(self.results.iter().cloned());
        if let Some(indices) = &self.backend.index_buffer {
            retained.push(indices.clone());
        }
        self.backend.retain_resources(&retained);
        Ok(())
    }
}
