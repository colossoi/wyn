//! Publish compiler entry interfaces into the host pipeline descriptor.

use crate::ast::TypeName;
use crate::host;
use crate::host::UniformMember;
use crate::interface::results::result_layout;
use crate::interface::StorageLayout;
use crate::ssa::layout::{block_layout, type_byte_size, vertex_format};
use crate::EntryId;
use crate::{BindingRef, LookupMap, LookupSet};
use polytype::Type;

use crate::flow::ExecutionModel;
use crate::host::{
    Access, BackingRef, Binding, BufferUsage, FragmentOutput, ModuleInterface, Pipeline, ResultKind,
    SamplerBindingType, SourceResultBinding, StageBindingUses, TextureSampleType, TextureViewDimension,
    VertexAttribute,
};
use crate::interface::{IoDecoration, StorageAccess, TextureSource};
use crate::ssa::types::EntryPoint;

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct DescriptorError(String);

/// Compiler-owned companion to the host ABI descriptor. Each outer index is a
/// descriptor pipeline index and each inner index is a stage index.
pub(crate) type StageEntryAssociations = Vec<Vec<EntryId>>;

fn entries_by_id<'a>(entries: &[&'a EntryPoint]) -> LookupMap<EntryId, &'a EntryPoint> {
    entries.iter().map(|entry| (entry.id, *entry)).collect()
}

pub trait ModuleInterfacePublish {
    /// Append `Binding::StorageBuffer` / `Uniform` / `PushConstant` /
    /// `Texture` / `Sampler` entries to the descriptor's per-pipeline
    /// bindings list for each `(set, binding)` recorded on the entry's
    /// `EntryInput`s, `EntryOutput`s, and `storage_bindings`. Generated
    /// resource metadata is resolved directly from entries in the same source
    /// family when the final binding is constructed.
    fn publish_implicit_bindings<'a>(
        &mut self,
        entries: &[&EntryPoint],
        associations: &StageEntryAssociations,
        source: impl Fn(&EntryPoint) -> &'a str,
    ) -> Result<(), DescriptorError>;

    /// Populate `vertex_inputs` and `fragment_outputs` on graphics
    /// pipelines from a vertex entry's `#[vertex_slot(n)]` inputs and a
    /// fragment entry's `#[target(name)]` outputs.
    fn publish_graphics_io(&mut self, entries: &[&EntryPoint], associations: &StageEntryAssociations);

    /// Apply authored input names to their final descriptor slots.
    fn relabel_input_storage_names(&mut self, names: &LookupMap<(u32, u32), String>);
}

pub(crate) fn reconcile_storage_binding_access<'a>(
    bindings: &mut [Binding],
    stages: impl IntoIterator<Item = &'a StageBindingUses>,
) {
    let stages = stages.into_iter().collect::<Vec<_>>();
    for (index, binding) in bindings.iter_mut().enumerate() {
        let Binding::StorageBuffer { access, .. } = binding else {
            continue;
        };
        let reads = stages.iter().any(|stage| stage.reads.contains(&index));
        let writes = stages.iter().any(|stage| stage.writes.contains(&index));
        if reads || writes {
            *access = match (reads, writes) {
                (true, true) => Access::ReadWrite,
                (true, false) => Access::ReadOnly,
                (false, true) => Access::WriteOnly,
                (false, false) => unreachable!(),
            };
        }
    }
}

impl ModuleInterfacePublish for ModuleInterface {
    fn publish_implicit_bindings<'a>(
        &mut self,
        entries: &[&EntryPoint],
        associations: &StageEntryAssociations,
        source: impl Fn(&EntryPoint) -> &'a str,
    ) -> Result<(), DescriptorError> {
        let mut layout = DescriptorLayout::from_pipeline(self)?;

        for entry in entries {
            // The descriptor keeps emitted names only. egglog's companion map
            // supplies the structural stage-to-entry association.
            let Some((pipeline_index, _)) =
                associations.iter().enumerate().find(|(_, stages)| stages.contains(&entry.id))
            else {
                continue;
            };
            let bindings: &mut Vec<Binding> = match &mut self.pipelines[pipeline_index] {
                Pipeline::Compute(cp) => &mut cp.bindings,
                Pipeline::Graphics(gp) => &mut gp.bindings,
            };

            let claimed_pc_offsets: LookupSet<u32> = bindings
                .iter()
                .filter_map(|b| match b {
                    Binding::PushConstant { offset, .. } => Some(*offset),
                    _ => None,
                })
                .collect();
            let mut local_claimed: LookupSet<BindingRef> =
                bindings.iter().filter_map(binding_slot).collect();

            let entry_source = source(entry);
            let storage = |slot: BindingRef| {
                let mut selected: Option<&crate::interface::StorageBindingDecl> = None;
                for declaration in entries
                    .iter()
                    .filter(|candidate| source(candidate) == entry_source)
                    .flat_map(|entry| &entry.storage_bindings)
                    .filter(|declaration| declaration.binding == slot)
                {
                    if declaration.logical_resource.is_none() || declaration.length.is_none() {
                        return Err(DescriptorError(
                            "generated storage has no resource identity or capacity".into(),
                        ));
                    }
                    if let Some(previous) = selected {
                        if previous.logical_resource != declaration.logical_resource
                            || previous.length != declaration.length
                        {
                            return Err(DescriptorError(format!(
                                "generated resource identities disagree for {}: {slot}",
                                entry_source
                            )));
                        }
                    }
                    selected = Some(declaration);
                }
                Ok(selected)
            };

            for input in &entry.inputs {
                if let Some(br) = input.uniform_binding() {
                    let (size, members) = uniform_block_members(&input.ty);
                    let binding = Binding::Uniform {
                        set: br.set,
                        binding: br.binding,
                        name: input.name.clone(),
                        size,
                        members,
                    };
                    let Some(slot) = binding_slot(&binding) else {
                        continue;
                    };
                    layout.reserve(&binding)?;
                    if !local_claimed.insert(slot) {
                        continue;
                    }
                    bindings.push(binding);
                } else if let Some(br) = input.storage_binding() {
                    let generated = storage(br)?;
                    let binding = Binding::StorageBuffer {
                        set: br.set,
                        binding: br.binding,
                        access: Access::ReadOnly,
                        usage: if generated.is_some() {
                            BufferUsage::Intermediate
                        } else {
                            BufferUsage::Input
                        },
                        name: input.name.clone(),
                        resource: generated.and_then(|decl| decl.logical_resource.clone()),
                        length: generated
                            .and_then(|decl| decl.length.clone())
                            .or_else(|| input.storage_length().cloned()),
                        members: Vec::new(),
                    };
                    let Some(slot) = binding_slot(&binding) else {
                        continue;
                    };
                    layout.reserve(&binding)?;
                    if !local_claimed.insert(slot) {
                        continue;
                    }
                    bindings.push(binding);
                } else if let Some(pc) = input.push_constant() {
                    if claimed_pc_offsets.contains(&pc.offset) {
                        continue;
                    }
                    bindings.push(Binding::PushConstant {
                        offset: pc.offset,
                        size: pc.size,
                        name: input.name.clone(),
                    });
                } else if let Some(br) = input.texture_binding() {
                    let (backing, resource) = match input.texture_source() {
                        Some(TextureSource::Backing(binding)) => (
                            Some(BackingRef {
                                set: binding.set,
                                binding: binding.binding,
                            }),
                            None,
                        ),
                        Some(TextureSource::Resource { name, backing }) => (
                            backing.map(|binding| BackingRef {
                                set: binding.set,
                                binding: binding.binding,
                            }),
                            Some(name.clone()),
                        ),
                        Some(TextureSource::External) | None => (None, None),
                    };
                    let binding = Binding::Texture {
                        set: br.set,
                        binding: br.binding,
                        name: input.name.clone(),
                        // Backend preparation refines this from final SSA uses.
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                        backing,
                        resource,
                    };
                    let Some(slot) = binding_slot(&binding) else {
                        continue;
                    };
                    layout.reserve(&binding)?;
                    if !local_claimed.insert(slot) {
                        continue;
                    }
                    bindings.push(binding);
                } else if let Some(br) = input.sampler_binding() {
                    let binding = Binding::Sampler {
                        set: br.set,
                        binding: br.binding,
                        name: input.name.clone(),
                        binding_type: SamplerBindingType::Filtering,
                    };
                    let Some(slot) = binding_slot(&binding) else {
                        continue;
                    };
                    layout.reserve(&binding)?;
                    if !local_claimed.insert(slot) {
                        continue;
                    }
                    bindings.push(binding);
                } else if let Some((br, format, access, size)) = input.storage_image_binding() {
                    let binding = Binding::StorageTexture {
                        set: br.set,
                        binding: br.binding,
                        name: input.name.clone(),
                        format,
                        access: access.into(),
                        size,
                        resource: input.storage_image_resource().map(str::to_owned),
                    };
                    let Some(slot) = binding_slot(&binding) else {
                        continue;
                    };
                    layout.reserve(&binding)?;
                    if !local_claimed.insert(slot) {
                        continue;
                    }
                    bindings.push(binding);
                }
            }

            // Compiler-managed storage declarations, including gather buffers,
            // scalar-prepass links, and egglog-scheduled phase scratch. Emit these
            // *before* outputs so
            // the producer's matching `EntryOutput` (same set/binding) doesn't
            // also claim it as a host-read `Output`. The producer declares it
            // Output (it writes) and the consumer Input (it reads); both surface
            // as a compiler-managed `Intermediate`, with access from the role.
            for decl in &entry.storage_bindings {
                storage(decl.binding)?;
                let access = Access::from(StorageAccess::from(decl.role));
                let binding = Binding::StorageBuffer {
                    set: decl.binding.set,
                    binding: decl.binding.binding,
                    access,
                    usage: BufferUsage::Intermediate,
                    name: if decl.length.is_some() {
                        format!("{}_gather_b{}", entry.name, decl.binding.binding)
                    } else {
                        format!("{}_intermediate_b{}", entry.name, decl.binding.binding)
                    },
                    resource: decl.logical_resource.clone(),
                    length: decl.length.clone(),
                    members: Vec::new(),
                };
                let Some(slot) = binding_slot(&binding) else {
                    continue;
                };
                layout.reserve(&binding)?;
                if !local_claimed.insert(slot) {
                    continue;
                }
                bindings.push(binding);
            }

            for (i, output) in entry.outputs.iter().enumerate() {
                let Some(br) = output.storage_binding() else {
                    continue;
                };
                let source_result = SourceResultBinding {
                    entry: entry.name.clone(),
                    kind: if entry.outputs.len() == 1 { ResultKind::Value } else { ResultKind::TupleField },
                    name: output.target().map(str::to_owned).unwrap_or_else(|| {
                        if entry.outputs.len() == 1 {
                            entry.name.clone()
                        } else {
                            format!("result_{i}")
                        }
                    }),
                    layout: result_layout(&output.ty),
                    result: i,
                    pipeline_index,
                    set: br.set,
                    binding: br.binding,
                };
                if !self.source_results.contains(&source_result) {
                    self.source_results.push(source_result);
                }
                // This name is the buffer's frame-graph identity — a reader
                // binding the same name reads the same resource — and is what
                // `viz --output <name>` takes on the command line.
                // `#[target(name)]` sets it. Otherwise it is derived from the
                // entry, with the position omitted when there is only one
                // output.
                let name = output.target().map(str::to_owned).unwrap_or_else(|| {
                    if entry.outputs.len() == 1 {
                        format!("{}_output", entry.name)
                    } else {
                        format!("{}_output_{}", entry.name, i)
                    }
                });
                let generated = storage(br)?;
                let binding = Binding::StorageBuffer {
                    set: br.set,
                    binding: br.binding,
                    access: Access::WriteOnly,
                    usage: BufferUsage::Output,
                    name,
                    resource: generated.and_then(|decl| decl.logical_resource.clone()),
                    length: generated
                        .and_then(|decl| decl.length.clone())
                        .or_else(|| output.storage_length().cloned()),
                    members: Vec::new(),
                };
                let Some(slot) = binding_slot(&binding) else {
                    continue;
                };
                layout.reserve(&binding)?;
                if !local_claimed.insert(slot) {
                    continue;
                }
                bindings.push(binding);
            }
        }

        Ok(())
    }

    fn publish_graphics_io(&mut self, entries: &[&EntryPoint], associations: &StageEntryAssociations) {
        let entries = entries_by_id(entries);
        for (pipeline_index, stage_ids) in associations.iter().enumerate() {
            let Some(Pipeline::Graphics(graphics)) = self.pipelines.get_mut(pipeline_index) else {
                continue;
            };
            for id in stage_ids {
                let Some(entry) = entries.get(id).copied() else {
                    continue;
                };
                match entry.execution_model {
                    ExecutionModel::Vertex => append_vertex_inputs(&mut graphics.vertex_inputs, entry),
                    ExecutionModel::Fragment => {
                        append_fragment_outputs(&mut graphics.fragment_outputs, entry)
                    }
                    _ => {}
                }
            }
        }
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

#[derive(Debug, Clone, PartialEq)]
enum DescriptorShape {
    StorageBuffer,
    Uniform {
        size: u32,
    },
    Texture {
        sample_type: TextureSampleType,
        view_dimension: TextureViewDimension,
        multisampled: bool,
    },
    Sampler {
        binding_type: SamplerBindingType,
    },
    StorageTexture {
        format: host::StorageImageFormat,
    },
}

impl DescriptorShape {
    fn label(&self) -> &'static str {
        match self {
            DescriptorShape::StorageBuffer => "storage buffer",
            DescriptorShape::Uniform { .. } => "uniform buffer",
            DescriptorShape::Texture { .. } => "texture",
            DescriptorShape::Sampler { .. } => "sampler",
            DescriptorShape::StorageTexture { .. } => "storage texture",
        }
    }
}

struct DescriptorLayout {
    slots: LookupMap<BindingRef, DescriptorShape>,
}

impl DescriptorLayout {
    fn from_pipeline(pipeline: &ModuleInterface) -> Result<Self, DescriptorError> {
        let mut layout = Self {
            slots: LookupMap::new(),
        };
        for pipeline in &pipeline.pipelines {
            let bindings = match pipeline {
                Pipeline::Compute(cp) => &cp.bindings,
                Pipeline::Graphics(gp) => &gp.bindings,
            };
            for binding in bindings {
                layout.reserve(binding)?;
            }
        }
        Ok(layout)
    }

    fn reserve(&mut self, binding: &Binding) -> Result<bool, DescriptorError> {
        let Some(slot) = binding_slot(binding) else {
            return Ok(true);
        };
        let Some(shape) = binding_shape(binding) else {
            return Ok(true);
        };
        match self.slots.get(&slot) {
            Some(existing) if existing == &shape => Ok(false),
            Some(existing) => Err(DescriptorError(format!(
                "descriptor binding collision at {slot}: existing {} conflicts with {}",
                existing.label(),
                shape.label()
            ))),
            None => {
                self.slots.insert(slot, shape);
                Ok(true)
            }
        }
    }
}

fn binding_slot(binding: &Binding) -> Option<BindingRef> {
    binding.slot().map(|(set, binding)| BindingRef::new(set, binding))
}

fn binding_shape(binding: &Binding) -> Option<DescriptorShape> {
    Some(match binding {
        Binding::StorageBuffer { .. } => DescriptorShape::StorageBuffer,
        Binding::Uniform { size, .. } => DescriptorShape::Uniform { size: *size },
        Binding::Texture {
            sample_type,
            view_dimension,
            multisampled,
            ..
        } => DescriptorShape::Texture {
            sample_type: sample_type.clone(),
            view_dimension: view_dimension.clone(),
            multisampled: *multisampled,
        },
        Binding::Sampler { binding_type, .. } => DescriptorShape::Sampler {
            binding_type: binding_type.clone(),
        },
        Binding::StorageTexture { format, .. } => DescriptorShape::StorageTexture { format: *format },
        Binding::PushConstant { .. } => return None,
    })
}

/// Populate `vertex_inputs` of the Graphics pipeline backing a vertex
/// entry from its `#[vertex_slot(n)]` parameters. Each becomes a
/// `VertexAttribute` carrying the slot, name, and the format derived from
/// the input's type. The type checker guarantees every such input has a
/// valid vertex format, so `vertex_format` returning `None` here is a
/// compiler bug.
fn append_vertex_inputs(vertex_inputs: &mut Vec<VertexAttribute>, entry: &EntryPoint) {
    for input in &entry.inputs {
        let Some(IoDecoration::Location(slot)) = input.decoration() else {
            continue;
        };
        let Some(format) = vertex_format(&input.ty) else {
            unreachable!(
                "type-checked vertex input {} has no vertex format: {:?}",
                input.name, input.ty
            );
        };
        vertex_inputs.push(VertexAttribute {
            slot,
            name: input.name.clone(),
            format,
        });
    }
}

fn append_fragment_outputs(fragment_outputs: &mut Vec<FragmentOutput>, entry: &EntryPoint) {
    for (i, output) in entry.outputs.iter().enumerate() {
        if let Some(name) = output.target().map(str::to_owned) {
            fragment_outputs.push(FragmentOutput {
                location: i as u32,
                name,
            });
        }
    }
}
/// std140 block size + member layout for a uniform binding's value
/// type. Record uniforms publish one member per field (source field
/// names); tuples publish `f0..fn`; bare scalars/vectors publish a
/// single member at offset 0 named after nothing the host needs to
/// qualify — size 0 / empty members when the type has no block layout
/// (hosts fall back to their known-name tables).
fn uniform_block_members(ty: &Type<TypeName>) -> (u32, Vec<UniformMember>) {
    let Some(layout) = block_layout(ty, StorageLayout::Std140) else {
        return (0, Vec::new());
    };
    let (names, field_tys): (Vec<String>, Vec<&Type<TypeName>>) = match ty {
        Type::Constructed(TypeName::Record(fields), args) => {
            (fields.iter().cloned().collect(), args.iter().collect())
        }
        Type::Constructed(TypeName::Tuple(_), args) => (
            (0..args.len()).map(|i| format!("f{i}")).collect(),
            args.iter().collect(),
        ),
        other => (vec![String::new()], vec![other]),
    };
    let members = names
        .into_iter()
        .zip(field_tys)
        .zip(&layout.member_offsets)
        .map(|((name, field_ty), &offset)| UniformMember {
            name,
            offset,
            size: type_byte_size(field_ty).unwrap_or(0),
        })
        .collect();
    (layout.size, members)
}
