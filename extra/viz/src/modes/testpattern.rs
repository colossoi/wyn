//! `testpattern` subcommand — interactive viewer running a built-in
//! WGSL test pattern (no shader file required, always validates).

use anyhow::{anyhow, Context, Result};
use winit::event_loop::EventLoop;

use wgpu::PresentMode;

use crate::app::{App, PipelineSpec, Shader};

const TEST_PATTERN_SHADER: &str = r#"
// Resolution uniform (16-byte aligned)
struct Globals {
    resolution: vec3<f32>,
    _pad: f32,
};

@group(0) @binding(0)
var<uniform> globals: Globals;

// Vertex shader - fullscreen big triangle
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );
    return vec4<f32>(pos[vertex_index], 0.0, 1.0);
}

// Fragment shader - colored test pattern
@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let res = globals.resolution.xy;
    let fragCoord = pos.xy;
    let uv = fragCoord / res;

    // Checkerboard using integer math (fragCoord >= 0, so truncation == floor)
    let grid_size = 64.0;
    let gx = i32(fragCoord.x / grid_size);
    let gy = i32(fragCoord.y / grid_size);
    let checker = f32((gx + gy) & 1);

    // Color gradient + diagonal stripes
    let r = uv.x;
    let g = uv.y;
    let b = checker * 0.5 + 0.5;
    let stripe = sin((uv.x + uv.y) * 20.0) * 0.5 + 0.5;

    return vec4<f32>(r * stripe, g, b * (1.0 - stripe * 0.3), 1.0);
}
"#;

pub fn run_test_pattern(max_frames: Option<u32>, verbose: bool) -> Result<()> {
    eprintln!("[viz] Test pattern mode - built-in WGSL shader");
    let spec = PipelineSpec {
        shader: Shader(TEST_PATTERN_SHADER),
        max_frames,
        verbose,
        validate: true,
        present_mode: PresentMode::Fifo,
        size: None,
    };
    let event_loop = EventLoop::new().context("failed to create event loop")?;
    let mut app = App::new(spec);
    event_loop.run_app(&mut app).map_err(|e| anyhow!(e)).context("winit event loop errored")?;
    Ok(())
}
