//! `vertex-fragment` subcommand — interactive (winit) viewer that
//! drives a SPIR-V module with separate vertex and fragment entry
//! points, optionally binding shadertoy-style uniforms.

use std::path::PathBuf;

use anyhow::{Context, Result, anyhow};
use wgpu::PresentMode;
use winit::event_loop::EventLoop;

use crate::app::{App, PipelineSpec};

pub fn run_vertex_fragment(
    path: PathBuf,
    vertex: String,
    fragment: String,
    shadertoy: bool,
    max_frames: Option<u32>,
    verbose: bool,
    validate: bool,
    present_mode: PresentMode,
    difficulty: i32,
    size: Option<(u32, u32)>,
) -> Result<()> {
    let spec = PipelineSpec::VertexFragment {
        path,
        vertex,
        fragment,
        shadertoy,
        max_frames,
        verbose,
        validate,
        present_mode,
        difficulty,
        size,
    };

    let event_loop = EventLoop::new().context("failed to create event loop")?;
    let mut app = App::new(spec);
    event_loop.run_app(&mut app).map_err(|e| anyhow!(e)).context("winit event loop errored")?;
    Ok(())
}
