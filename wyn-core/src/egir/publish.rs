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

    /// Workgroup size the parallelizer chose for the compute entry
    /// `entry_name`, or `(64, 1, 1)` when the entry isn't in the
    /// descriptor (e.g. graphics entries — non-compute call sites skip
    /// this anyway).
    fn workgroup_size_of(&self, entry_name: &str) -> (u32, u32, u32);
}

impl PipelineDescriptorPublish for PipelineDescriptor {
    fn publish_implicit_bindings(&mut self, entries: &[EgirEntry]) {
        // SPIR-V `NonWritable` is a module-level decoration on
        // `OpVariable`, not per-entry-point. The SPIR-V backend (see
        // `spirv/mod.rs` `written_bindings`) only emits `NonWritable`
        // when no entry writes the binding. Naga / wgpu reads that
        // module-level access as the binding's class: any binding
        // written by some entry becomes `read_write`. The descriptor
        // must agree, otherwise wgpu rejects pipeline creation with
        // `Storage class Storage{LOAD} doesn't match shader Storage{LOAD
        // | STORE}`. Compute the set of module-writable bindings here
        // and promote `ReadOnly → ReadWrite` in a final post-pass below.
        let written_bindings: HashSet<(u32, u32)> = entries
            .iter()
            .flat_map(|e| e.outputs.iter().filter_map(|o| o.storage_binding))
            .map(|br| (br.set, br.binding))
            .collect();

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
                if let Some(br) = input.uniform_binding {
                    if claimed.contains(&(br.set, br.binding)) {
                        continue;
                    }
                    bindings.push(Binding::Uniform {
                        set: br.set,
                        binding: br.binding,
                        name: input.name.clone(),
                    });
                } else if let Some(br) = input.storage_binding {
                    if claimed.contains(&(br.set, br.binding)) {
                        continue;
                    }
                    bindings.push(Binding::StorageBuffer {
                        set: br.set,
                        binding: br.binding,
                        access: Access::ReadOnly,
                        usage: BufferUsage::Input,
                        name: input.name.clone(),
                        length: None,
                    });
                } else if let Some(pc) = input.push_constant {
                    if claimed_pc_offsets.contains(&pc.offset) {
                        continue;
                    }
                    bindings.push(Binding::PushConstant {
                        offset: pc.offset,
                        size: pc.size,
                        name: input.name.clone(),
                    });
                } else if let Some(br) = input.texture_binding {
                    if claimed.contains(&(br.set, br.binding)) {
                        continue;
                    }
                    bindings.push(Binding::Texture {
                        set: br.set,
                        binding: br.binding,
                        name: input.name.clone(),
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    });
                } else if let Some(br) = input.sampler_binding {
                    if claimed.contains(&(br.set, br.binding)) {
                        continue;
                    }
                    bindings.push(Binding::Sampler {
                        set: br.set,
                        binding: br.binding,
                        name: input.name.clone(),
                        binding_type: SamplerBindingType::Filtering,
                    });
                } else if let Some((br, format, access)) = input.storage_image_binding {
                    if claimed.contains(&(br.set, br.binding)) {
                        continue;
                    }
                    bindings.push(Binding::StorageTexture {
                        set: br.set,
                        binding: br.binding,
                        name: input.name.clone(),
                        format,
                        access: match access {
                            crate::interface::StorageAccess::ReadOnly => Access::ReadOnly,
                            crate::interface::StorageAccess::WriteOnly => Access::WriteOnly,
                            crate::interface::StorageAccess::ReadWrite => Access::ReadWrite,
                        },
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
                if !claimed.insert((decl.binding.set, decl.binding.binding)) {
                    continue;
                }
                let access = match decl.role {
                    crate::interface::StorageRole::Output => Access::WriteOnly,
                    _ => Access::ReadOnly,
                };
                bindings.push(Binding::StorageBuffer {
                    set: decl.binding.set,
                    binding: decl.binding.binding,
                    access,
                    usage: BufferUsage::Intermediate,
                    name: format!("{}_gather_b{}", entry.name, decl.binding.binding),
                    length: decl.length.clone(),
                });
            }

            for (i, output) in entry.outputs.iter().enumerate() {
                let Some(br) = output.storage_binding else {
                    continue;
                };
                if !claimed.insert((br.set, br.binding)) {
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
                    set: br.set,
                    binding: br.binding,
                    access: Access::WriteOnly,
                    usage: BufferUsage::Output,
                    name,
                    length: output.length.clone(),
                });
            }
        }

        // Promote any `ReadOnly` storage binding whose `(set, binding)`
        // is written by some entry in the module — see comment at the
        // top of this function for why. Monotonic: only widens. Existing
        // `ReadWrite` / `WriteOnly` are left as-is (`WriteOnly` already
        // maps to `read_only=false` at the wgpu layer).
        for pipeline in self.pipelines.iter_mut() {
            let bindings: &mut Vec<Binding> = match pipeline {
                Pipeline::Compute(cp) => &mut cp.bindings,
                Pipeline::MultiCompute(mc) => &mut mc.bindings,
                Pipeline::Graphics(gp) => &mut gp.bindings,
            };
            for b in bindings.iter_mut() {
                if let Binding::StorageBuffer {
                    set, binding, access, ..
                } = b
                {
                    if *access == Access::ReadOnly && written_bindings.contains(&(*set, *binding)) {
                        *access = Access::ReadWrite;
                    }
                }
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

    fn workgroup_size_of(&self, entry_name: &str) -> (u32, u32, u32) {
        for p in &self.pipelines {
            match p {
                Pipeline::Compute(cp) if cp.entry_point == entry_name => return cp.workgroup_size,
                Pipeline::MultiCompute(mp) => {
                    if let Some(stage) = mp.stages.iter().find(|s| s.entry_point == entry_name) {
                        return stage.workgroup_size;
                    }
                }
                _ => {}
            }
        }
        (64, 1, 1)
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
