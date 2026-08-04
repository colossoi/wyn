//! Build source entry shells and structural stage associations for EGIR scheduling.
//!
//! This conversion preserves source entry metadata without choosing generated
//! entries, resources, output grouping, or dispatch phases.

use crate::interface::EntryKind;
use crate::pipeline_descriptor::*;
use crate::tlc::DefMeta as GenericDefMeta;
use crate::SymbolId;

type Program = crate::tlc::stage::InputSliceBoundsInferred;
type DefMeta = GenericDefMeta<crate::tlc::data::EntryInputBounds>;

pub(super) struct PipelineSeed {
    pub pipeline: PipelineDescriptor,
    pub stage_symbols: Vec<Vec<SymbolId>>,
}

pub(super) fn build(program: &Program) -> PipelineSeed {
    let mut pipelines: Vec<Pipeline> = Vec::new();
    let mut stage_symbols: Vec<Vec<SymbolId>> = Vec::new();
    let mut unified_graphics = crate::LookupMap::<String, usize>::new();

    for def in &program.defs {
        let DefMeta::EntryPoint(entry) = &def.meta else {
            continue;
        };
        let decl = &entry.declaration;
        let name = crate::symbol_name_or_bug(&program.symbols, def.name).to_string();
        let feedback = decl
            .feedback
            .iter()
            .map(|pair| FeedbackPair {
                read_set: pair.read.set,
                read_binding: pair.read.binding,
                write_set: pair.write.set,
                write_binding: pair.write.binding,
            })
            .collect();

        if matches!(decl.entry_kind, EntryKind::Root | EntryKind::Compute) {
            let dispatch_size = decl
                .compute_dispatch
                .map(|grid| DispatchSize::Fixed {
                    x: grid.x,
                    y: grid.y,
                    z: grid.z,
                    explicit: true,
                })
                .unwrap_or(DispatchSize::Fixed {
                    x: 1,
                    y: 1,
                    z: 1,
                    explicit: false,
                });
            pipelines.push(Pipeline::Compute(ComputePipeline {
                bindings: Vec::new(),
                stages: vec![ComputeStage {
                    entry_point: name.clone(),
                    owner: name,
                    workgroup_size: (64, 1, 1),
                    dispatch_size,
                    uses: StageBindingUses::default(),
                }],
                default_total_threads: None,
                feedback,
            }));
            stage_symbols.push(vec![def.name]);
        } else if let Some((owner, stage)) = unified_stage_name(&name) {
            if let Some(&pipeline_index) = unified_graphics.get(&owner) {
                let Pipeline::Graphics(graphics) = &mut pipelines[pipeline_index] else {
                    unreachable!("unified graphics owner was registered as compute")
                };
                graphics.stages.push(GraphicsStage {
                    entry_point: name,
                    owner,
                    stage,
                    uses: StageBindingUses::default(),
                });
                stage_symbols[pipeline_index].push(def.name);
            } else {
                let pipeline_index = pipelines.len();
                unified_graphics.insert(owner.clone(), pipeline_index);
                pipelines.push(Pipeline::Graphics(GraphicsPipeline {
                    invocation: decl.graphics_invocation.clone().unwrap_or_default(),
                    stages: vec![GraphicsStage {
                        entry_point: name,
                        owner,
                        stage,
                        uses: StageBindingUses::default(),
                    }],
                    bindings: Vec::new(),
                    vertex_inputs: Vec::new(),
                    fragment_outputs: Vec::new(),
                    feedback,
                }));
                stage_symbols.push(vec![def.name]);
            }
        } else {
            let stage = if decl.entry_kind == EntryKind::Vertex {
                ShaderStage::Vertex
            } else {
                ShaderStage::Fragment
            };
            pipelines.push(Pipeline::Graphics(GraphicsPipeline {
                invocation: GraphicsInvocation::default(),
                stages: vec![GraphicsStage {
                    entry_point: name.clone(),
                    owner: name,
                    stage,
                    uses: StageBindingUses::default(),
                }],
                bindings: Vec::new(),
                vertex_inputs: Vec::new(),
                fragment_outputs: Vec::new(),
                feedback,
            }));
            stage_symbols.push(vec![def.name]);
        }
    }

    PipelineSeed {
        pipeline: PipelineDescriptor {
            pipelines,
            frame_graph: Default::default(),
        },
        stage_symbols,
    }
}

fn unified_stage_name(name: &str) -> Option<(String, ShaderStage)> {
    let rest = name.strip_prefix("_w_stage_")?;
    if let Some(owner) = rest.strip_suffix("__vertex") {
        return Some((owner.to_string(), ShaderStage::Vertex));
    }
    if let Some(owner) = rest.strip_suffix("__fragment") {
        return Some((owner.to_string(), ShaderStage::Fragment));
    }
    None
}
