//! Build source entry shells for EGIR scheduling.
//!
//! This conversion preserves source entry metadata without choosing generated
//! entries, resources, output grouping, or dispatch phases.

use crate::ast::TypeName;
use crate::interface::{Attribute, EntryParamBindingKind};
use crate::pipeline_descriptor::*;
use crate::tlc::{DefMeta, Program, Term, TermKind};
use crate::{LookupMap, LookupSet, SymbolId};
use polytype::Type;

pub(super) struct PipelineSeed {
    pub pipeline: PipelineDescriptor,
    pub input_names: LookupMap<(u32, u32), String>,
}

fn peel_lambda_params(term: &Term) -> (Vec<(SymbolId, Type<TypeName>)>, &Term) {
    match &term.kind {
        TermKind::Lambda(lam) => {
            let (mut inner, body) = peel_lambda_params(&lam.body);
            let mut params = lam.params.clone();
            params.append(&mut inner);
            (params, body)
        }
        _ => (vec![], term),
    }
}

fn collect_entry_input_names(program: &Program) -> LookupMap<(u32, u32), String> {
    let mut out = LookupMap::new();
    let mut ambiguous = LookupSet::new();
    let mut put = |key: (u32, u32), name: String| {
        if ambiguous.contains(&key) {
            return;
        }
        match out.get(&key) {
            Some(existing) if *existing != name => {
                out.remove(&key);
                ambiguous.insert(key);
            }
            Some(_) => {}
            None => {
                out.insert(key, name);
            }
        }
    };

    for def in &program.defs {
        let DefMeta::EntryPoint(decl) = &def.meta else {
            continue;
        };
        if !decl.entry_type.is_compute() {
            continue;
        }
        let (params, _) = peel_lambda_params(&def.body);
        for (i, (sym, _)) in params.iter().enumerate() {
            let name = crate::symbol_name_or_bug(&program.symbols, *sym).to_string();
            if let Some(binding) =
                decl.params.get(i).and_then(crate::binding_layout::extract_storage_binding)
            {
                put((binding.set, binding.binding), name);
                continue;
            }
            let Some(binding) = decl.param_bindings.get(i).and_then(Option::as_ref) else {
                continue;
            };
            match &binding.kind {
                EntryParamBindingKind::Single { binding, .. } => {
                    put((binding.set, binding.binding), name);
                }
                EntryParamBindingKind::TupleOfViews(fields) => {
                    for (index, field) in fields.iter().enumerate() {
                        put(
                            (field.binding.set, field.binding.binding),
                            format!("{name}_{index}"),
                        );
                    }
                }
            }
        }
    }
    out
}

pub(super) fn run(program: &Program) -> PipelineSeed {
    let input_names = collect_entry_input_names(program);
    let mut pipelines = Vec::new();

    for def in &program.defs {
        let DefMeta::EntryPoint(decl) = &def.meta else {
            continue;
        };
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

        if decl.entry_type.is_compute() {
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
                    entry_point: name,
                    workgroup_size: (64, 1, 1),
                    dispatch_size,
                    reads: Vec::new(),
                    writes: Vec::new(),
                }],
                default_total_threads: None,
                feedback,
            }));
        } else {
            let stage = if decl.entry_type == Attribute::Vertex {
                ShaderStage::Vertex
            } else {
                ShaderStage::Fragment
            };
            pipelines.push(Pipeline::Graphics(GraphicsPipeline {
                stages: vec![GraphicsStage {
                    entry_point: name,
                    stage,
                }],
                bindings: Vec::new(),
                vertex_inputs: Vec::new(),
                fragment_outputs: Vec::new(),
                feedback,
            }));
        }
    }

    PipelineSeed {
        pipeline: PipelineDescriptor {
            pipelines,
            frame_graph: Default::default(),
        },
        input_names,
    }
}
