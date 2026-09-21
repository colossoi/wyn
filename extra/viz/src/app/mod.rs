use crate::gpu::{DeviceRequest, GpuContext};
use crate::modes::pipeline::{Frame, RunSpec, Runner};
use anyhow::{anyhow, Context, Result};
use std::sync::Arc;
use std::time::Instant;
use wgpu::{InstanceFlags, PresentMode, Surface, SurfaceConfiguration, SurfaceError, TextureUsages};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

pub struct App {
    spec: Option<RunSpec>,
    state: Option<State>,
    failure: Option<anyhow::Error>,
}

struct State {
    window: Arc<Window>,
    surface: Surface<'static>,
    config: SurfaceConfiguration,
    runner: Runner,
    started: Instant,
    previous: Instant,
    frame: Frame,
    maximum: Option<u32>,
    mouse_down: bool,
}

impl App {
    pub fn run(spec: RunSpec) -> Result<()> {
        let event_loop = EventLoop::new().context("creating event loop")?;
        let mut app = Self {
            spec: Some(spec),
            state: None,
            failure: None,
        };
        event_loop.run_app(&mut app)?;
        if let Some(error) = app.failure {
            return Err(error);
        }
        Ok(())
    }

    fn initialize(&mut self, event_loop: &ActiveEventLoop) -> Result<()> {
        let Some(spec) = self.spec.take() else {
            return Ok(());
        };
        if spec.opts.max_frames == Some(0) {
            return Err(anyhow!("--max-frames must be positive"));
        }
        let (width, height) = spec.opts.size.unwrap_or((800, 600));
        let window = Arc::new(
            event_loop.create_window(
                Window::default_attributes()
                    .with_title("wyn viz")
                    .with_inner_size(PhysicalSize::new(width, height)),
            )?,
        );
        let target = window.clone();
        let context = pollster::block_on(GpuContext::request(DeviceRequest {
            surface_target: Some(Box::new(move |instance| Ok(instance.create_surface(target)?))),
            instance_flags: if spec.opts.validate {
                InstanceFlags::VALIDATION
            } else {
                InstanceFlags::empty()
            },
            ..Default::default()
        }))?;
        if spec.verbose {
            eprintln!("GPU: {}", context.adapter.get_info().name);
        }
        let Some(surface) = context.surface else {
            return Err(anyhow!("missing window surface"));
        };
        let capabilities = surface.get_capabilities(&context.adapter);
        let Some(&format) =
            capabilities.formats.iter().find(|format| !format.is_srgb()).or(capabilities.formats.first())
        else {
            return Err(anyhow!("surface has no supported formats"));
        };
        let Some(&alpha_mode) = capabilities.alpha_modes.first() else {
            return Err(anyhow!("surface has no alpha modes"));
        };
        let usage = TextureUsages::RENDER_ATTACHMENT | (capabilities.usages & TextureUsages::COPY_SRC);
        let config = SurfaceConfiguration {
            usage,
            format,
            width: width.max(1),
            height: height.max(1),
            present_mode: if capabilities.present_modes.contains(&spec.opts.present_mode) {
                spec.opts.present_mode
            } else {
                PresentMode::Fifo
            },
            desired_maximum_frame_latency: 2,
            alpha_mode,
            view_formats: Vec::new(),
        };
        surface.configure(&context.device, &config);
        let maximum = spec.opts.max_frames;
        let runner = Runner::new(spec, context.device, context.queue, config.width, config.height)?;
        let now = Instant::now();
        self.state = Some(State {
            window,
            surface,
            config,
            runner,
            started: now,
            previous: now,
            frame: Frame {
                width,
                height,
                time: 0.0,
                delta: 0.0,
                index: 0,
                mouse: [0.0; 4],
                keyboard: [0; 768],
            },
            maximum,
            mouse_down: false,
        });
        Ok(())
    }

    fn fail(&mut self, event_loop: &ActiveEventLoop, error: anyhow::Error) {
        self.failure = Some(error);
        event_loop.exit();
    }
}

impl State {
    fn draw(&mut self) -> Result<bool> {
        if self.frame.width == 0 || self.frame.height == 0 {
            return Ok(false);
        }
        let frame = match self.surface.get_current_texture() {
            Ok(frame) => frame,
            Err(SurfaceError::Lost | SurfaceError::Outdated) => {
                self.surface.configure(&self.runner.backend.device, &self.config);
                return Ok(false);
            }
            Err(SurfaceError::Timeout) => return Ok(false),
            Err(error) => return Err(error.into()),
        };
        let now = Instant::now();
        self.frame.time = now.duration_since(self.started).as_secs_f32();
        self.frame.delta = now.duration_since(self.previous).as_secs_f32();
        self.previous = now;
        self.runner.run_frame(&self.frame, Some(frame.texture.clone()))?;
        self.frame.index = self.frame.index.checked_add(1).context("frame count overflow")?;
        self.frame.keyboard[256..512].fill(0);
        let finished = self.maximum.is_some_and(|max| self.frame.index >= max);
        if finished {
            self.runner.output(false)?;
        }
        frame.present();
        Ok(finished)
    }

    fn event(&mut self, event: WindowEvent) -> Result<bool> {
        match event {
            WindowEvent::CloseRequested => {
                self.runner.output(false)?;
                return Ok(true);
            }
            WindowEvent::Resized(size) => {
                self.frame.width = size.width;
                self.frame.height = size.height;
                if size.width != 0 && size.height != 0 {
                    self.config.width = size.width;
                    self.config.height = size.height;
                    self.surface.configure(&self.runner.backend.device, &self.config);
                }
            }
            WindowEvent::RedrawRequested => return self.draw(),
            WindowEvent::CursorMoved { position, .. } => {
                self.frame.mouse[0] = position.x as f32;
                self.frame.mouse[1] = self.frame.height as f32 - position.y as f32;
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                self.mouse_down = state == ElementState::Pressed;
                if self.mouse_down {
                    self.frame.mouse[2] = self.frame.mouse[0];
                    self.frame.mouse[3] = self.frame.mouse[1];
                } else {
                    self.frame.mouse[2] = -self.frame.mouse[2].abs();
                    self.frame.mouse[3] = -self.frame.mouse[3].abs();
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.logical_key == Key::Named(NamedKey::Escape) && event.state == ElementState::Pressed
                {
                    self.runner.output(false)?;
                    return Ok(true);
                }
                if let Some(code) = key_code(&event.logical_key) {
                    let down = event.state == ElementState::Pressed;
                    self.frame.keyboard[code] = if down { 255 } else { 0 };
                    if down && !event.repeat {
                        self.frame.keyboard[256 + code] = 255;
                        self.frame.keyboard[512 + code] ^= 255;
                    }
                }
            }
            _ => {}
        }
        Ok(false)
    }
}

fn key_code(key: &Key) -> Option<usize> {
    Some(match key {
        Key::Character(text) if text.len() == 1 && text.is_ascii() => {
            usize::from(text.as_bytes()[0].to_ascii_uppercase())
        }
        Key::Named(name) => match name {
            NamedKey::Backspace => 8,
            NamedKey::Tab => 9,
            NamedKey::Enter => 13,
            NamedKey::Shift => 16,
            NamedKey::Control => 17,
            NamedKey::Alt => 18,
            NamedKey::Escape => 27,
            NamedKey::Space => 32,
            NamedKey::ArrowLeft => 37,
            NamedKey::ArrowUp => 38,
            NamedKey::ArrowRight => 39,
            NamedKey::ArrowDown => 40,
            _ => return None,
        },
        _ => return None,
    })
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if let Err(error) = self.initialize(event_loop) {
            self.fail(event_loop, error);
        }
    }
    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else {
            return;
        };
        match state.event(event) {
            Ok(true) => event_loop.exit(),
            Ok(false) => {}
            Err(error) => self.fail(event_loop, error),
        }
    }
    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.window.request_redraw();
        }
    }
}
