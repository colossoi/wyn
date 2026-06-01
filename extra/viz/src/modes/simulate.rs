//! `simulate` subcommand — interactive viewer that drives a SPIR-V
//! module containing both a compute entry and a graphics
//! (vertex+fragment) entry, ping-ponging two storage buffers each
//! frame: each frame dispatches compute writing into the "other"
//! buffer, then renders the fragment reading that "other" buffer.
//!
//! Motivating shader is Conway's Game of Life
//! (`testfiles/playground/conway.wyn`), but the mode is generic:
//! works for any SPIR-V module whose compiled descriptor exposes one
//! `Compute` pipeline + one `Graphics` pipeline whose compute output
//! and fragment input have matching element shape.

mod patterns;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use wgpu::PresentMode;
use winit::event_loop::EventLoop;

use crate::app::{App, SimulateSpec};
use wyn_pipeline_descriptor::{
    Access, Binding, BufferUsage, ComputePipeline, GraphicsPipeline, Pipeline, PipelineDescriptor,
    ShaderStage,
};

pub use patterns::{Pattern, parse as parse_pattern};

#[allow(clippy::too_many_arguments)]
pub fn run_simulate(
    path: PathBuf,
    compute: Option<String>,
    vertex: Option<String>,
    fragment: Option<String>,
    grid: (u32, u32),
    pattern: Pattern,
    seed: Option<u64>,
    max_frames: Option<u32>,
    verbose: bool,
    validate: bool,
    present_mode: PresentMode,
    size: Option<(u32, u32)>,
    vertex_count: u32,
) -> Result<()> {
    // Sidecar: same path as the SPIR-V with .json extension.
    let descriptor = load_descriptor(&path)?;

    let compute_pipeline = pick_compute_pipeline(&descriptor, compute.as_deref())?;
    let graphics_pipeline = pick_graphics_pipeline(&descriptor)?;

    let resolved_vertex = resolve_stage_entry(&descriptor, ShaderStage::Vertex, vertex)?;
    let resolved_fragment = resolve_stage_entry(&descriptor, ShaderStage::Fragment, fragment)?;

    let (input_binding, output_binding) = identify_ping_pong_pair(compute_pipeline)?;
    let display_binding = identify_display_storage(graphics_pipeline)?;

    if verbose {
        eprintln!(
            "[viz simulate] compute = {} (set {} bindings {} → {}), \
             fragment = {} (set {} binding {})",
            compute_pipeline.entry_point,
            input_binding.set,
            input_binding.binding,
            output_binding.binding,
            resolved_fragment,
            display_binding.set,
            display_binding.binding,
        );
    }

    let initial_board = patterns::build_initial_board(&pattern, seed, grid)?;

    let spec = SimulateSpec {
        shader_path: path,
        compute_entry: compute_pipeline.entry_point.clone(),
        vertex_entry: resolved_vertex,
        fragment_entry: resolved_fragment,
        grid,
        initial_board,
        compute_pipeline: compute_pipeline.clone(),
        graphics_pipeline: graphics_pipeline.clone(),
        input_binding,
        output_binding,
        display_binding,
        max_frames,
        verbose,
        validate,
        present_mode,
        size,
        vertex_count,
    };

    let event_loop = EventLoop::new().context("failed to create event loop")?;
    let mut app = App::new_simulate(spec);
    event_loop.run_app(&mut app).map_err(|e| anyhow!(e)).context("winit event loop errored")?;
    Ok(())
}

#[derive(Debug, Clone, Copy)]
pub struct BindingLoc {
    pub set: u32,
    pub binding: u32,
}

fn load_descriptor(spv_path: &Path) -> Result<PipelineDescriptor> {
    let json_path = spv_path.with_extension("json");
    let content = std::fs::read_to_string(&json_path)
        .with_context(|| format!("read pipeline descriptor sidecar {:?}", json_path))?;
    serde_json::from_str(&content)
        .with_context(|| format!("parse pipeline descriptor sidecar {:?}", json_path))
}

fn pick_compute_pipeline<'a>(
    descriptor: &'a PipelineDescriptor,
    requested: Option<&str>,
) -> Result<&'a ComputePipeline> {
    let computes: Vec<&ComputePipeline> = descriptor
        .pipelines
        .iter()
        .filter_map(|p| if let Pipeline::Compute(c) = p { Some(c) } else { None })
        .collect();
    if let Some(name) = requested {
        computes
            .iter()
            .copied()
            .find(|c| c.entry_point == name)
            .ok_or_else(|| anyhow!("compute entry {:?} not found in pipeline descriptor", name))
    } else {
        match computes.as_slice() {
            [c] => Ok(*c),
            [] => Err(anyhow!(
                "no compute pipeline in descriptor — simulate mode needs one compute entry"
            )),
            many => Err(anyhow!(
                "multiple compute pipelines in descriptor ({}); pass --compute <entry> to pick one",
                many.len()
            )),
        }
    }
}

fn pick_graphics_pipeline(descriptor: &PipelineDescriptor) -> Result<&GraphicsPipeline> {
    // A graphics pipeline in the descriptor carries one stage. To run we
    // need both a vertex and a fragment stage, in either the same pipeline
    // or paired pipelines. Wyn emits one Graphics pipeline per entry
    // point, so prefer a pipeline that already has both stages; otherwise
    // synthesize a logical "graphics pipeline" view by merging the
    // vertex-only and fragment-only pipelines. For this pass we accept
    // either: (a) a single pipeline with both stages, or (b) exactly one
    // vertex-only + one fragment-only pipeline; we return the fragment
    // pipeline (its `bindings` are the render-side ones).
    let graphics: Vec<&GraphicsPipeline> = descriptor
        .pipelines
        .iter()
        .filter_map(|p| if let Pipeline::Graphics(g) = p { Some(g) } else { None })
        .collect();
    if graphics.is_empty() {
        bail!("no graphics pipeline in descriptor — simulate mode needs vertex + fragment entries");
    }
    let combined = graphics.iter().find(|g| {
        g.stages.iter().any(|s| matches!(s.stage, ShaderStage::Vertex))
            && g.stages.iter().any(|s| matches!(s.stage, ShaderStage::Fragment))
    });
    if let Some(g) = combined {
        return Ok(*g);
    }
    // Fall back to the fragment-only pipeline — it carries the render
    // bindings the host needs. The vertex entry name is resolved
    // separately via `resolve_stage_entry`.
    let fragment_only = graphics
        .iter()
        .copied()
        .find(|g| g.stages.iter().any(|s| matches!(s.stage, ShaderStage::Fragment)));
    fragment_only.ok_or_else(|| {
        anyhow!("graphics pipeline has no fragment stage — simulate mode needs a fragment entry")
    })
}

fn resolve_stage_entry(
    descriptor: &PipelineDescriptor,
    target: ShaderStage,
    requested: Option<String>,
) -> Result<String> {
    if let Some(name) = requested {
        return Ok(name);
    }
    // Search every graphics pipeline's stages — the compiler emits one
    // pipeline per entry point, so the vertex and fragment stages
    // typically live in separate pipelines (one stage each).
    let mut matches: Vec<&str> = Vec::new();
    for p in &descriptor.pipelines {
        let Pipeline::Graphics(g) = p else { continue };
        for stage in &g.stages {
            if std::mem::discriminant(&stage.stage) == std::mem::discriminant(&target) {
                matches.push(&stage.entry_point);
            }
        }
    }
    let stage_name = match target {
        ShaderStage::Vertex => "vertex",
        ShaderStage::Fragment => "fragment",
    };
    match matches.as_slice() {
        [one] => Ok((*one).to_string()),
        [] => Err(anyhow!(
            "no {} stage found in any graphics pipeline — pass --{} <entry> to name one",
            stage_name,
            stage_name
        )),
        many => Err(anyhow!(
            "multiple {} stages found ({}): {} — pass --{} <entry> to pick one",
            stage_name,
            many.len(),
            many.join(", "),
            stage_name
        )),
    }
}

/// Find the read-only storage input and write-only storage output that
/// form the ping-pong pair on the compute side.
fn identify_ping_pong_pair(compute: &ComputePipeline) -> Result<(BindingLoc, BindingLoc)> {
    let mut input: Option<BindingLoc> = None;
    let mut output: Option<BindingLoc> = None;
    for b in &compute.bindings {
        if let Binding::StorageBuffer {
            set,
            binding,
            access,
            usage,
            ..
        } = b
        {
            match (access, usage) {
                (Access::ReadOnly, BufferUsage::Input) => {
                    if input.is_some() {
                        bail!(
                            "compute pipeline has multiple read-only storage inputs; \
                             simulate mode needs exactly one ping-pong input"
                        );
                    }
                    input = Some(BindingLoc {
                        set: *set,
                        binding: *binding,
                    });
                }
                (Access::WriteOnly, BufferUsage::Output) => {
                    if output.is_some() {
                        bail!(
                            "compute pipeline has multiple write-only storage outputs; \
                             simulate mode needs exactly one ping-pong output"
                        );
                    }
                    output = Some(BindingLoc {
                        set: *set,
                        binding: *binding,
                    });
                }
                _ => {}
            }
        }
    }
    let input = input.ok_or_else(|| {
        anyhow!("compute pipeline has no read-only storage input — simulate mode needs one")
    })?;
    let output = output.ok_or_else(|| {
        anyhow!("compute pipeline has no write-only storage output — simulate mode needs one")
    })?;
    Ok((input, output))
}

/// Find the read-only storage input on the graphics side that the
/// fragment reads — the same physical buffer the compute writes.
fn identify_display_storage(graphics: &GraphicsPipeline) -> Result<BindingLoc> {
    let mut display: Option<BindingLoc> = None;
    for b in &graphics.bindings {
        if let Binding::StorageBuffer {
            set, binding, access, ..
        } = b
        {
            if matches!(access, Access::ReadOnly) {
                if display.is_some() {
                    bail!(
                        "graphics pipeline has multiple read-only storage inputs; \
                         simulate mode needs exactly one display-buffer input"
                    );
                }
                display = Some(BindingLoc {
                    set: *set,
                    binding: *binding,
                });
            }
        }
    }
    display.ok_or_else(|| {
        anyhow!("graphics pipeline has no read-only storage input — simulate mode needs one")
    })
}
