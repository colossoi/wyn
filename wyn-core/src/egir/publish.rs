//! Publish per-entry state from EGIR into the host-runtime-shared
//! `PipelineDescriptor`. Two distinct jobs, exposed through one
//! extension trait so the call site reads as `pipeline.publish_*`:
//!
//!   1. `publish_implicit_bindings` — surface the
//!      compiler-allocated `(set, binding)` slots that
//!      `convert_entry_point` parked on each entry's `EntryInput`s
//!      and `EntryOutput`s. The descriptor sees per-pipeline binding
//!      declarations that the developer never wrote — both routed
//!      user attributes (`#[storage]`, `#[uniform]`, `#[texture]`,
//!      `#[sampler]`, push constants) and compiler intermediates
//!      (`lift_gathers` gather buffers, parallelize phase IO).
//!
//!   2. `publish_graphics_io` — thread `#[location(N)]` graphics
//!      attributes from a vertex / fragment entry's inputs / outputs
//!      into the descriptor's `vertex_inputs` / `fragment_outputs`.
//!
//! Why an extension trait: `PipelineDescriptor` lives in the
//! `wyn-pipeline-descriptor` crate (shared with host runtimes), so
//! it can't grow `EgirEntry`-aware methods directly. Rust's orphan
//! rule blocks a regular `impl` block here too. A trait owned by
//! `wyn-core` is the standard workaround.

use std::collections::HashSet;

use crate::egir::program::EgirEntry;
use crate::pipeline_descriptor::{
    Access, Binding, BufferUsage, FragmentOutput, Pipeline, PipelineDescriptor, SamplerBindingType,
    TextureSampleType, TextureViewDimension, VertexAttribute,
};
use crate::ssa::types::{ExecutionModel, IoDecoration};

pub trait PipelineDescriptorPublish {
    /// Append `Binding::StorageBuffer` / `Uniform` / `PushConstant` /
    /// `Texture` / `Sampler` entries to the descriptor's per-pipeline
    /// bindings list for each `(set, binding)` recorded on the entry's
    /// `EntryInput`s, `EntryOutput`s, and `storage_bindings` (gather
    /// intermediates). Bindings already present (e.g. those a
    /// `MultiCompute` parallelization path pre-populated) are skipped.
    fn publish_implicit_bindings(&mut self, entries: &[EgirEntry]);

    /// Populate `vertex_inputs` and `fragment_outputs` on graphics
    /// pipelines from `#[location(N)]` decorations on the matching
    /// vertex/fragment entry's inputs/outputs.
    fn publish_graphics_io(&mut self, entries: &[EgirEntry]);
}

impl PipelineDescriptorPublish for PipelineDescriptor {
    fn publish_implicit_bindings(&mut self, entries: &[EgirEntry]) {
        for entry in entries {
            // Find the bindings list backing this entry. Compute entries
            // match a single-stage `Compute` by `entry_point` or any stage
            // of a `MultiCompute` (whose bindings are shared across stages,
            // covering parallel reduce / redomap / scan phases). Graphics
            // entries match a single-stage `Graphics` by stage entry_point.
            let bindings: &mut Vec<Binding> = match self.pipelines.iter_mut().find(|p| match p {
                Pipeline::Compute(cp) => cp.entry_point == entry.name,
                Pipeline::MultiCompute(mc) => mc.stages.iter().any(|s| s.entry_point == entry.name),
                Pipeline::Graphics(gp) => gp.stages.iter().any(|s| s.entry_point == entry.name),
            }) {
                Some(Pipeline::Compute(cp)) => &mut cp.bindings,
                Some(Pipeline::MultiCompute(mc)) => &mut mc.bindings,
                Some(Pipeline::Graphics(gp)) => &mut gp.bindings,
                _ => continue,
            };

            // Snapshot existing (set, binding) and push-constant offsets to
            // skip what `parallelize` already surfaced.
            let mut claimed: HashSet<(u32, u32)> = bindings
                .iter()
                .filter_map(|b| match b {
                    Binding::StorageBuffer { set, binding, .. } => Some((*set, *binding)),
                    Binding::Uniform { set, binding, .. } => Some((*set, *binding)),
                    _ => None,
                })
                .collect();
            let claimed_pc_offsets: HashSet<u32> = bindings
                .iter()
                .filter_map(|b| match b {
                    Binding::PushConstant { offset, .. } => Some(*offset),
                    _ => None,
                })
                .collect();

            for input in &entry.inputs {
                if let Some((set, binding)) = input.uniform_binding {
                    if claimed.contains(&(set, binding)) {
                        continue;
                    }
                    bindings.push(Binding::Uniform {
                        set,
                        binding,
                        name: input.name.clone(),
                    });
                } else if let Some((set, binding)) = input.storage_binding {
                    if claimed.contains(&(set, binding)) {
                        continue;
                    }
                    bindings.push(Binding::StorageBuffer {
                        set,
                        binding,
                        access: Access::ReadOnly,
                        usage: BufferUsage::Input,
                        name: input.name.clone(),
                        length: None,
                    });
                } else if let Some(offset) = input.push_constant_offset {
                    if claimed_pc_offsets.contains(&offset) {
                        continue;
                    }
                    let size = crate::ssa::layout::type_byte_size(&input.ty).unwrap_or(4) as u32;
                    bindings.push(Binding::PushConstant {
                        offset,
                        size,
                        name: input.name.clone(),
                    });
                } else if let Some((set, binding)) = input.texture_binding {
                    if claimed.contains(&(set, binding)) {
                        continue;
                    }
                    bindings.push(Binding::Texture {
                        set,
                        binding,
                        name: input.name.clone(),
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    });
                } else if let Some((set, binding)) = input.sampler_binding {
                    if claimed.contains(&(set, binding)) {
                        continue;
                    }
                    bindings.push(Binding::Sampler {
                        set,
                        binding,
                        name: input.name.clone(),
                        binding_type: SamplerBindingType::Filtering,
                    });
                }
            }

            // Compiler-managed gather buffers: storage bindings carrying an
            // explicit `length` (a `lift_gathers` intermediate, referenced via
            // `storage_index` rather than a param). Emit these *before* outputs so
            // the producer's matching `EntryOutput` (same set/binding) doesn't
            // also claim it as a host-read `Output`. The producer declares it
            // Output (it writes) and the consumer Input (it reads); both surface
            // as a compiler-managed `Intermediate`, with access from the role.
            for decl in &entry.storage_bindings {
                if decl.length.is_none() {
                    continue;
                }
                if !claimed.insert((decl.set, decl.binding)) {
                    continue;
                }
                let access = match decl.role {
                    crate::interface::StorageRole::Output => Access::WriteOnly,
                    _ => Access::ReadOnly,
                };
                bindings.push(Binding::StorageBuffer {
                    set: decl.set,
                    binding: decl.binding,
                    access,
                    usage: BufferUsage::Intermediate,
                    name: format!("{}_gather_b{}", entry.name, decl.binding),
                    length: decl.length.clone(),
                });
            }

            for (i, output) in entry.outputs.iter().enumerate() {
                let Some((set, binding)) = output.storage_binding else {
                    continue;
                };
                if !claimed.insert((set, binding)) {
                    continue;
                }
                // EntryOutput has no name field; synthesize from the entry
                // name + position. Single-output is the common case and
                // gets the cleaner `<entry>_output` form.
                let name = if entry.outputs.len() == 1 {
                    format!("{}_output", entry.name)
                } else {
                    format!("{}_output_{}", entry.name, i)
                };
                bindings.push(Binding::StorageBuffer {
                    set,
                    binding,
                    access: Access::WriteOnly,
                    usage: BufferUsage::Output,
                    name,
                    length: output.length.clone(),
                });
            }
        }
    }

    fn publish_graphics_io(&mut self, entries: &[EgirEntry]) {
        for entry in entries {
            match entry.execution_model {
                ExecutionModel::Vertex => publish_vertex_inputs(self, entry),
                ExecutionModel::Fragment => publish_fragment_outputs(self, entry),
                _ => {}
            }
        }
    }
}

/// Populate `vertex_inputs` of the Graphics pipeline backing a vertex
/// entry from its `#[location(n)]` parameters. Each `IoDecoration::Location`
/// input becomes a `VertexAttribute` carrying the location, name, and
/// the format derived from the input's type. The type checker guarantees
/// every such input has a valid vertex format, so `vertex_format`
/// returning `None` here is a compiler bug.
fn publish_vertex_inputs(pipeline: &mut PipelineDescriptor, entry: &EgirEntry) {
    let vertex_inputs = match pipeline.pipelines.iter_mut().find(|p| match p {
        Pipeline::Graphics(gp) => gp.stages.iter().any(|s| s.entry_point == entry.name),
        _ => false,
    }) {
        Some(Pipeline::Graphics(gp)) => &mut gp.vertex_inputs,
        _ => return,
    };

    for input in &entry.inputs {
        let Some(IoDecoration::Location(location)) = input.decoration else {
            continue;
        };
        let format = crate::ssa::layout::vertex_format(&input.ty).expect(
            "vertex #[location] param must have a valid vertex format \
             (the type checker enforces this)",
        );
        vertex_inputs.push(VertexAttribute {
            location,
            name: input.name.clone(),
            format,
        });
    }
}

/// Populate `fragment_outputs` of the Graphics pipeline backing a
/// fragment entry from its `#[location(n)]` outputs. Each
/// `IoDecoration::Location` output becomes a `FragmentOutput` carrying
/// the location and a synthesized name. `EntryOutput` has no name
/// field, so the name is derived from the entry name + position.
fn publish_fragment_outputs(pipeline: &mut PipelineDescriptor, entry: &EgirEntry) {
    let fragment_outputs = match pipeline.pipelines.iter_mut().find(|p| match p {
        Pipeline::Graphics(gp) => gp.stages.iter().any(|s| s.entry_point == entry.name),
        _ => false,
    }) {
        Some(Pipeline::Graphics(gp)) => &mut gp.fragment_outputs,
        _ => return,
    };

    let multi = entry.outputs.len() > 1;
    for (i, output) in entry.outputs.iter().enumerate() {
        let Some(IoDecoration::Location(location)) = output.decoration else {
            continue;
        };
        let name =
            if multi { format!("{}_output_{}", entry.name, i) } else { format!("{}_output", entry.name) };
        fragment_outputs.push(FragmentOutput { location, name });
    }
}
