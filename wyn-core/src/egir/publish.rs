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
//!   2. `publish_graphics_io` — thread a vertex entry's `#[vertex_slot(n)]`
//!      inputs and a fragment entry's `#[target(name)]` outputs into the
//!      descriptor's `vertex_inputs` / `fragment_outputs`.
//!
//! Why an extension trait: `PipelineDescriptor` lives in the
//! `wyn-pipeline-descriptor` crate (shared with host runtimes), so
//! it can't grow `EgirEntry`-aware methods directly. Rust's orphan
//! rule blocks a regular `impl` block here too. A trait owned by
//! `wyn-core` is the standard workaround.

use crate::{LookupMap, LookupSet};

use crate::egir::program::EgirEntry;
use crate::pipeline_descriptor::{
    Access, BackingRef, Binding, BufferUsage, FragmentOutput, Pipeline, PipelineDescriptor,
    SamplerBindingType, TextureSampleType, TextureViewDimension, VertexAttribute,
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
    /// pipelines from a vertex entry's `#[vertex_slot(n)]` inputs and a
    /// fragment entry's `#[target(name)]` outputs.
    fn publish_graphics_io(&mut self, entries: &[EgirEntry]);

    /// Workgroup size the parallelizer chose for the compute entry
    /// `entry_name`, or `(64, 1, 1)` when the entry isn't in the
    /// descriptor (e.g. graphics entries — non-compute call sites skip
    /// this anyway).
    fn workgroup_size_of(&self, entry_name: &str) -> (u32, u32, u32);

    /// Restore source-parameter names onto input storage bindings. The
    /// parallelization path names its storage inputs positionally
    /// (`input_0`, `input_1`, …); `names` maps each `(set, binding)` back
    /// to the name the source declared. Only `BufferUsage::Input` storage
    /// buffers are touched, so outputs and intermediates keep their
    /// synthesized names even if some other entry's input shares a slot.
    fn relabel_input_storage_names(&mut self, names: &LookupMap<(u32, u32), String>);
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
        let written_bindings: LookupSet<(u32, u32)> = entries
            .iter()
            .flat_map(|e| e.outputs.iter().filter_map(|o| o.storage_binding))
            .map(|br| (br.set, br.binding))
            // A `#[storage(access=write/readwrite)]` input is written in place
            // (e.g. a `scatter` destination), so its binding is module-writable
            // too — the promotion below widens it (and any sibling read of the
            // same slot) to `ReadWrite`.
            .chain(entries.iter().flat_map(|e| {
                e.inputs.iter().filter_map(|i| {
                    let br = i.storage_binding?;
                    match i.storage_access {
                        Some(
                            crate::interface::StorageAccess::WriteOnly
                            | crate::interface::StorageAccess::ReadWrite,
                        ) => Some((br.set, br.binding)),
                        _ => None,
                    }
                })
            }))
            .chain(entries.iter().flat_map(|e| {
                e.storage_bindings.iter().filter_map(|decl| {
                    matches!(
                        decl.role,
                        crate::interface::StorageRole::Output | crate::interface::StorageRole::Intermediate
                    )
                    .then_some((decl.binding.set, decl.binding.binding))
                })
            }))
            .collect();

        for entry in entries {
            // Find the bindings list backing this entry. A compute
            // entry matches any stage in a `Compute` (covers both the
            // common single-stage case and multi-stage parallel
            // reduce / scan / ordered-prefix schedules);
            // graphics entries match a stage of a `Graphics`.
            let bindings: &mut Vec<Binding> = match self.pipelines.iter_mut().find(|p| match p {
                Pipeline::Compute(cp) => cp.stages.iter().any(|s| s.entry_point == entry.name),
                Pipeline::Graphics(gp) => gp.stages.iter().any(|s| s.entry_point == entry.name),
            }) {
                Some(Pipeline::Compute(cp)) => &mut cp.bindings,
                Some(Pipeline::Graphics(gp)) => &mut gp.bindings,
                _ => continue,
            };

            // Snapshot existing (set, binding) and push-constant offsets to
            // skip what `parallelize` (or an earlier publication pass) already
            // surfaced. Every slotted kind must be collected — a kind missing
            // here gets re-pushed on the next pass and the duplicate slot
            // fails bind-group-layout creation at runtime.
            let mut claimed: LookupSet<(u32, u32)> = bindings
                .iter()
                .filter_map(|b| match b {
                    Binding::StorageBuffer { set, binding, .. } => Some((*set, *binding)),
                    Binding::Uniform { set, binding, .. } => Some((*set, *binding)),
                    Binding::Texture { set, binding, .. } => Some((*set, *binding)),
                    Binding::Sampler { set, binding, .. } => Some((*set, *binding)),
                    Binding::StorageTexture { set, binding, .. } => Some((*set, *binding)),
                    Binding::PushConstant { .. } => None,
                })
                .collect();
            let claimed_pc_offsets: LookupSet<u32> = bindings
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
                    let (size, members) = uniform_block_members(&input.ty);
                    bindings.push(Binding::Uniform {
                        set: br.set,
                        binding: br.binding,
                        name: input.name.clone(),
                        size,
                        members,
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
                        length: input.length.clone(),
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
                        backing: input.texture_backing.map(|b| BackingRef {
                            set: b.set,
                            binding: b.binding,
                        }),
                        resource: input.texture_resource.clone(),
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
                } else if let Some((br, format, access, size)) = input.storage_image_binding {
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
                        size,
                        resource: input.storage_image_resource.clone(),
                    });
                }
            }

            // Compiler-managed storage declarations, including gather buffers,
            // scalar-prepass links, and EGIR-scheduled phase scratch. Emit these
            // *before* outputs so
            // the producer's matching `EntryOutput` (same set/binding) doesn't
            // also claim it as a host-read `Output`. The producer declares it
            // Output (it writes) and the consumer Input (it reads); both surface
            // as a compiler-managed `Intermediate`, with access from the role.
            for decl in &entry.storage_bindings {
                if !claimed.insert((decl.binding.set, decl.binding.binding)) {
                    continue;
                }
                let access = match decl.role {
                    crate::interface::StorageRole::Output => Access::WriteOnly,
                    crate::interface::StorageRole::Input => Access::ReadOnly,
                    crate::interface::StorageRole::Intermediate => Access::ReadWrite,
                };
                bindings.push(Binding::StorageBuffer {
                    set: decl.binding.set,
                    binding: decl.binding.binding,
                    access,
                    usage: BufferUsage::Intermediate,
                    name: if decl.length.is_some() {
                        format!("{}_gather_b{}", entry.name, decl.binding.binding)
                    } else {
                        format!("{}_intermediate_b{}", entry.name, decl.binding.binding)
                    },
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
            if let Pipeline::Compute(cp) = p {
                if let Some(stage) = cp.stages.iter().find(|s| s.entry_point == entry_name) {
                    return stage.workgroup_size;
                }
            }
        }
        (64, 1, 1)
    }

    fn relabel_input_storage_names(&mut self, names: &LookupMap<(u32, u32), String>) {
        if names.is_empty() {
            return;
        }
        for pipeline in self.pipelines.iter_mut() {
            let bindings: &mut Vec<Binding> = match pipeline {
                Pipeline::Compute(cp) => &mut cp.bindings,
                Pipeline::Graphics(gp) => &mut gp.bindings,
            };
            for b in bindings.iter_mut() {
                if let Binding::StorageBuffer {
                    set,
                    binding,
                    usage: BufferUsage::Input,
                    name,
                    ..
                } = b
                {
                    if let Some(real) = names.get(&(*set, *binding)) {
                        *name = real.clone();
                    }
                }
            }
        }
    }
}

/// Populate `vertex_inputs` of the Graphics pipeline backing a vertex
/// entry from its `#[vertex_slot(n)]` parameters. Each becomes a
/// `VertexAttribute` carrying the slot, name, and the format derived from
/// the input's type. The type checker guarantees every such input has a
/// valid vertex format, so `vertex_format` returning `None` here is a
/// compiler bug.
fn publish_vertex_inputs(pipeline: &mut PipelineDescriptor, entry: &EgirEntry) {
    let vertex_inputs = match pipeline.pipelines.iter_mut().find(|p| match p {
        Pipeline::Graphics(gp) => gp.stages.iter().any(|s| s.entry_point == entry.name),
        _ => false,
    }) {
        Some(Pipeline::Graphics(gp)) => &mut gp.vertex_inputs,
        _ => return,
    };

    for input in &entry.inputs {
        let Some(IoDecoration::Location(slot)) = input.decoration else {
            continue;
        };
        let format = crate::ssa::layout::vertex_format(&input.ty).expect(
            "vertex #[vertex_slot] param must have a valid vertex format \
             (the type checker enforces this)",
        );
        vertex_inputs.push(VertexAttribute {
            slot,
            name: input.name.clone(),
            format,
        });
    }
}

/// Populate `fragment_outputs` of the Graphics pipeline backing a
/// fragment entry from its `#[target(name)]` outputs. Each targeted output
/// becomes a `FragmentOutput` naming the render-target resource, with the
/// color-attachment slot taken from the output's position in the return tuple.
fn publish_fragment_outputs(pipeline: &mut PipelineDescriptor, entry: &EgirEntry) {
    let fragment_outputs = match pipeline.pipelines.iter_mut().find(|p| match p {
        Pipeline::Graphics(gp) => gp.stages.iter().any(|s| s.entry_point == entry.name),
        _ => false,
    }) {
        Some(Pipeline::Graphics(gp)) => &mut gp.fragment_outputs,
        _ => return,
    };

    for (i, output) in entry.outputs.iter().enumerate() {
        let Some(name) = output.target.clone() else {
            continue;
        };
        fragment_outputs.push(FragmentOutput {
            location: i as u32,
            name,
        });
    }
}

/// std140 block size + member layout for a uniform binding's value
/// type. Record uniforms publish one member per field (source field
/// names); tuples publish `f0..fn`; bare scalars/vectors publish a
/// single member at offset 0 named after nothing the host needs to
/// qualify — size 0 / empty members when the type has no block layout
/// (hosts fall back to their known-name tables).
fn uniform_block_members(
    ty: &polytype::Type<crate::ast::TypeName>,
) -> (u32, Vec<wyn_pipeline_descriptor::UniformMember>) {
    use crate::ast::TypeName;
    use crate::ssa::layout::{block_layout, type_byte_size, LayoutRules};
    let Some(layout) = block_layout(ty, LayoutRules::Std140) else {
        return (0, Vec::new());
    };
    let (names, field_tys): (Vec<String>, Vec<&polytype::Type<crate::ast::TypeName>>) = match ty {
        polytype::Type::Constructed(TypeName::Record(fields), args) => {
            (fields.iter().cloned().collect(), args.iter().collect())
        }
        polytype::Type::Constructed(TypeName::Tuple(_), args) => (
            (0..args.len()).map(|i| format!("f{i}")).collect(),
            args.iter().collect(),
        ),
        other => (vec![String::new()], vec![other]),
    };
    let members = names
        .into_iter()
        .zip(field_tys)
        .zip(&layout.member_offsets)
        .map(
            |((name, field_ty), &offset)| wyn_pipeline_descriptor::UniformMember {
                name,
                offset,
                size: type_byte_size(field_ty).unwrap_or(0),
            },
        )
        .collect();
    (layout.size, members)
}
