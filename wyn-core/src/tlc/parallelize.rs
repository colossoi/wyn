//! TLC-level SOAC parallelization.
//!
//! Stage A: Analyze compute entry points to find parallelizable SOACs.
//! Stage B: Restructure the program — create new entry points with chunked SOACs,
//!          allocate intermediate storage buffers, build pipeline descriptor.
//!
//! Loop creation and storage lowering stay in SSA (`to_ssa` + `soac_lower`).

use crate::SymbolId;
use crate::ast::{self, Attribute, TypeName};
use crate::pipeline_descriptor::*;
use polytype::Type;
use std::collections::HashMap;

use super::{ArrayExpr, Def, DefMeta, Lambda, Program, ReduceProps, SoacOp, Term, TermId, TermKind};

// =============================================================================
// Analysis types
// =============================================================================

/// Where a SOAC's input array comes from.
#[derive(Debug, Clone)]
pub enum ArrayProvenance {
    /// From a storage buffer entry parameter.
    Storage {
        set: u32,
        binding: u32,
        elem_ty: Type<TypeName>,
    },
    /// From a range/iota.
    Range,
}

/// A parallelizable SOAC found in a compute entry point.
#[derive(Debug, Clone)]
pub enum ParallelizableSoac {
    Map {
        lam: Lambda,
        inputs: Vec<ArrayExpr>,
        provenances: Vec<ArrayProvenance>,
        output_elem_type: Type<TypeName>,
    },
    Reduce {
        op: Lambda,
        ne: Box<Term>,
        input: ArrayExpr,
        provenance: ArrayProvenance,
        elem_type: Type<TypeName>,
    },
    Redomap {
        op: Lambda,
        reduce_op: Lambda,
        ne: Box<Term>,
        inputs: Vec<ArrayExpr>,
        provenances: Vec<ArrayProvenance>,
        acc_type: Type<TypeName>,
    },
    Scan {
        op: Lambda,
        ne: Box<Term>,
        input: ArrayExpr,
        provenance: ArrayProvenance,
        elem_type: Type<TypeName>,
    },
}

/// Result of analyzing a compute entry point.
#[derive(Debug, Clone)]
struct EntryAnalysis {
    def_name: SymbolId,
    soac: ParallelizableSoac,
    /// Let-binding prefix before the SOAC.
    prefix_lets: Vec<(SymbolId, Type<TypeName>, Term)>,
}

// =============================================================================
// Stage A: Analysis
// =============================================================================

fn analyze_program(program: &Program) -> HashMap<SymbolId, EntryAnalysis> {
    let mut results = HashMap::new();

    for def in &program.defs {
        let DefMeta::EntryPoint(ref entry_decl) = def.meta else {
            continue;
        };
        if !entry_decl.entry_type.is_compute() {
            continue;
        }

        if let Some(analysis) = analyze_entry(def) {
            results.insert(def.name, analysis);
        }
    }

    results
}

fn analyze_entry(def: &Def) -> Option<EntryAnalysis> {
    let mut prefix_lets = Vec::new();
    let mut current = &def.body;

    loop {
        match &current.kind {
            TermKind::Let {
                name,
                name_ty,
                rhs,
                body,
            } => {
                prefix_lets.push((*name, name_ty.clone(), (**rhs).clone()));
                current = body;
            }
            TermKind::Soac(soac) => {
                let parallelizable = analyze_soac(soac, &current.ty)?;
                return Some(EntryAnalysis {
                    def_name: def.name,
                    soac: parallelizable,
                    prefix_lets,
                });
            }
            _ => return None,
        }
    }
}

fn analyze_soac(soac: &SoacOp, result_ty: &Type<TypeName>) -> Option<ParallelizableSoac> {
    match soac {
        SoacOp::Map { lam, inputs } => {
            let provenances: Vec<ArrayProvenance> =
                inputs.iter().map(classify_input).collect::<Option<Vec<_>>>()?;
            let output_elem_type = elem_type_of_result(result_ty);
            Some(ParallelizableSoac::Map {
                lam: lam.clone(),
                inputs: inputs.clone(),
                provenances,
                output_elem_type,
            })
        }
        SoacOp::Reduce { op, ne, input, .. } => {
            let provenance = classify_input(input)?;
            Some(ParallelizableSoac::Reduce {
                op: op.clone(),
                ne: Box::new((**ne).clone()),
                input: input.clone(),
                provenance,
                elem_type: result_ty.clone(),
            })
        }
        SoacOp::Redomap {
            op,
            reduce_op,
            ne,
            inputs,
            ..
        } => {
            let provenances: Vec<ArrayProvenance> =
                inputs.iter().map(classify_input).collect::<Option<Vec<_>>>()?;
            Some(ParallelizableSoac::Redomap {
                op: op.clone(),
                reduce_op: reduce_op.clone(),
                ne: Box::new((**ne).clone()),
                inputs: inputs.clone(),
                provenances,
                acc_type: result_ty.clone(),
            })
        }
        SoacOp::Scan { op, ne, input } => {
            let provenance = classify_input(input)?;
            Some(ParallelizableSoac::Scan {
                op: op.clone(),
                ne: Box::new((**ne).clone()),
                input: input.clone(),
                provenance,
                elem_type: elem_type_of_result(result_ty),
            })
        }
        _ => None,
    }
}

fn classify_input(input: &ArrayExpr) -> Option<ArrayProvenance> {
    match input {
        ArrayExpr::StorageBuffer {
            set,
            binding,
            elem_ty,
            ..
        } => Some(ArrayProvenance::Storage {
            set: *set,
            binding: *binding,
            elem_ty: elem_ty.clone(),
        }),
        ArrayExpr::Range { .. } => Some(ArrayProvenance::Range),
        _ => None,
    }
}

fn elem_type_of_result(ty: &Type<TypeName>) -> Type<TypeName> {
    match ty {
        Type::Constructed(TypeName::Array, args) if !args.is_empty() => args[0].clone(),
        _ => ty.clone(),
    }
}

// =============================================================================
// Stage B: Restructuring
// =============================================================================

const LOCAL_SIZE: (u32, u32, u32) = (64, 1, 1);
const TOTAL_THREADS: u32 = 64;

pub struct ParallelizationResult {
    pub program: Program,
    pub pipeline: PipelineDescriptor,
}

/// Parallelize SOACs in compute entry points.
pub fn parallelize_soacs(mut program: Program) -> ParallelizationResult {
    let analyses = analyze_program(&program);

    if analyses.is_empty() {
        let pipeline = build_default_pipeline(&program);
        return ParallelizationResult { program, pipeline };
    }

    let mut pipelines = Vec::new();
    let mut new_defs = Vec::new();
    let mut removed_entries: Vec<SymbolId> = Vec::new();

    // Default pipelines for non-parallelized compute entries.
    for def in &program.defs {
        if let DefMeta::EntryPoint(ref decl) = def.meta {
            if decl.entry_type.is_compute() && !analyses.contains_key(&def.name) {
                let name = program.symbols.get(def.name).cloned().unwrap_or_default();
                let input_bindings = collect_input_bindings(&analyses, def.name);
                pipelines.push(Pipeline::Compute(ComputePipeline {
                    entry_point: name,
                    workgroup_size: LOCAL_SIZE,
                    dispatch_size: DispatchSize::DerivedFromInputLength {
                        workgroup_size: TOTAL_THREADS,
                    },
                    bindings: input_bindings,
                }));
            }
        }
    }

    // Track max binding across all storage decls for fresh binding allocation.
    let mut next_binding: u32 = program.storage.iter().map(|s| s.binding + 1).max().unwrap_or(0);

    for (_sym, analysis) in &analyses {
        let entry_name = program.symbols.get(analysis.def_name).cloned().unwrap_or_default();

        match &analysis.soac {
            ParallelizableSoac::Map { .. } => {
                // Single dispatch — the SOAC stays as-is.
                let input_bindings = collect_soac_bindings(&analysis.soac);
                pipelines.push(Pipeline::Compute(ComputePipeline {
                    entry_point: entry_name,
                    workgroup_size: LOCAL_SIZE,
                    dispatch_size: DispatchSize::DerivedFromInputLength {
                        workgroup_size: TOTAL_THREADS,
                    },
                    bindings: input_bindings,
                }));
            }

            ParallelizableSoac::Reduce {
                op, ne, elem_type, ..
            } => {
                let partials_binding = (0, next_binding);
                let result_binding = (0, next_binding + 1);
                next_binding += 2;

                let (entries, pipeline) = build_two_phase_entries(
                    &entry_name,
                    analysis,
                    op,
                    ne,
                    elem_type,
                    partials_binding,
                    result_binding,
                    &mut program,
                );
                removed_entries.push(analysis.def_name);
                new_defs.extend(entries);
                pipelines.push(pipeline);
            }

            ParallelizableSoac::Redomap {
                reduce_op,
                ne,
                acc_type,
                ..
            } => {
                let partials_binding = (0, next_binding);
                let result_binding = (0, next_binding + 1);
                next_binding += 2;

                let (entries, pipeline) = build_two_phase_entries(
                    &entry_name,
                    analysis,
                    reduce_op,
                    ne,
                    acc_type,
                    partials_binding,
                    result_binding,
                    &mut program,
                );
                removed_entries.push(analysis.def_name);
                new_defs.extend(entries);
                pipelines.push(pipeline);
            }

            ParallelizableSoac::Scan {
                op,
                ne,
                input,
                elem_type,
                ..
            } => {
                let output_binding = (0, next_binding);
                let block_sums_binding = (0, next_binding + 1);
                let block_offsets_binding = (0, next_binding + 2);
                next_binding += 3;

                let (entries, pipeline) = build_scan_entries(
                    &entry_name,
                    analysis,
                    op,
                    ne,
                    input,
                    elem_type,
                    output_binding,
                    block_sums_binding,
                    block_offsets_binding,
                    &mut program,
                );
                removed_entries.push(analysis.def_name);
                new_defs.extend(entries);
                pipelines.push(pipeline);
            }
        }
    }

    // Graphics pipelines.
    for def in &program.defs {
        if let DefMeta::EntryPoint(ref decl) = def.meta {
            if !decl.entry_type.is_compute() {
                let name = program.symbols.get(def.name).cloned().unwrap_or_default();
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
                    bindings: vec![],
                    vertex_inputs: vec![],
                    fragment_outputs: vec![],
                }));
            }
        }
    }

    program.defs.retain(|d| !removed_entries.contains(&d.name));
    program.defs.extend(new_defs);

    ParallelizationResult {
        program,
        pipeline: PipelineDescriptor { pipelines },
    }
}

// =============================================================================
// Two-phase entry builder (Reduce / Redomap)
// =============================================================================

fn build_two_phase_entries(
    entry_name: &str,
    analysis: &EntryAnalysis,
    reduce_op: &Lambda,
    ne: &Term,
    elem_type: &Type<TypeName>,
    partials_binding: (u32, u32),
    result_binding: (u32, u32),
    program: &mut Program,
) -> (Vec<Def>, Pipeline) {
    let span = ne.span;

    // Phase 1: the original SOAC (reduce or redomap) over the full input.
    // soac_lower + parallelization codegen will handle chunking.
    let phase1_name = format!("{}_phase1_chunks", entry_name);
    let phase1_body =
        rebuild_soac_with_lets(&analysis.soac, &analysis.prefix_lets, elem_type.clone(), span);
    let phase1_def = make_entry_def(&phase1_name, phase1_body, elem_type.clone(), program);

    // Phase 2: reduce over the partials buffer.
    let phase2_name = format!("{}_phase2_combine", entry_name);
    let partials_input = ArrayExpr::StorageBuffer {
        set: partials_binding.0,
        binding: partials_binding.1,
        offset: Box::new(uint_lit(0, span)),
        len: Box::new(uint_lit(TOTAL_THREADS as u64, span)),
        elem_ty: elem_type.clone(),
    };
    let phase2_soac = SoacOp::Reduce {
        op: reduce_op.clone(),
        ne: Box::new(ne.clone()),
        input: partials_input,
        props: ReduceProps::default(),
    };
    let phase2_body = soac_term(phase2_soac, elem_type.clone(), span);
    let phase2_def = make_entry_def(&phase2_name, phase2_body, elem_type.clone(), program);

    // Collect input bindings from the SOAC analysis.
    let input_bindings = collect_soac_bindings(&analysis.soac);
    let partials_idx = input_bindings.len();
    let result_idx = input_bindings.len() + 1;

    let mut all_bindings = input_bindings;
    all_bindings.push(Binding::StorageBuffer {
        set: partials_binding.0,
        binding: partials_binding.1,
        access: Access::ReadWrite,
        usage: BufferUsage::Intermediate,
        name: format!("{}_partials", entry_name),
    });
    all_bindings.push(Binding::StorageBuffer {
        set: result_binding.0,
        binding: result_binding.1,
        access: Access::WriteOnly,
        usage: BufferUsage::Output,
        name: format!("{}_result", entry_name),
    });

    let input_indices: Vec<usize> = (0..partials_idx).collect();

    let pipeline = Pipeline::MultiCompute(MultiComputePipeline {
        bindings: all_bindings,
        stages: vec![
            ComputeStage {
                entry_point: phase1_name.clone(),
                workgroup_size: LOCAL_SIZE,
                dispatch_size: DispatchSize::DerivedFromInputLength {
                    workgroup_size: TOTAL_THREADS,
                },
                reads: input_indices,
                writes: vec![partials_idx],
            },
            ComputeStage {
                entry_point: phase2_name.clone(),
                workgroup_size: (1, 1, 1),
                dispatch_size: DispatchSize::Fixed { x: 1, y: 1, z: 1 },
                reads: vec![partials_idx],
                writes: vec![result_idx],
            },
        ],
    });

    (vec![phase1_def, phase2_def], pipeline)
}

// =============================================================================
// Three-phase entry builder (Scan)
// =============================================================================

fn build_scan_entries(
    entry_name: &str,
    analysis: &EntryAnalysis,
    op: &Lambda,
    ne: &Term,
    _input: &ArrayExpr,
    elem_type: &Type<TypeName>,
    output_binding: (u32, u32),
    block_sums_binding: (u32, u32),
    block_offsets_binding: (u32, u32),
    program: &mut Program,
) -> (Vec<Def>, Pipeline) {
    let span = ne.span;

    // Phase 1: local scans per chunk.
    let phase1_name = format!("{}_phase1_local_scans", entry_name);
    let phase1_body =
        rebuild_soac_with_lets(&analysis.soac, &analysis.prefix_lets, elem_type.clone(), span);
    let phase1_def = make_entry_def(&phase1_name, phase1_body, elem_type.clone(), program);

    // Phase 2: scan the block sums.
    let phase2_name = format!("{}_phase2_scan_sums", entry_name);
    let block_sums_input = ArrayExpr::StorageBuffer {
        set: block_sums_binding.0,
        binding: block_sums_binding.1,
        offset: Box::new(uint_lit(0, span)),
        len: Box::new(uint_lit(TOTAL_THREADS as u64, span)),
        elem_ty: elem_type.clone(),
    };
    let phase2_soac = SoacOp::Scan {
        op: op.clone(),
        ne: Box::new(ne.clone()),
        input: block_sums_input,
    };
    let phase2_body = soac_term(phase2_soac, elem_type.clone(), span);
    let phase2_def = make_entry_def(&phase2_name, phase2_body, elem_type.clone(), program);

    // Phase 3: add block offsets to each element.
    let phase3_name = format!("{}_phase3_add_offsets", entry_name);
    let output_input = ArrayExpr::StorageBuffer {
        set: output_binding.0,
        binding: output_binding.1,
        offset: Box::new(uint_lit(0, span)),
        len: Box::new(uint_lit(0, span)), // runtime length
        elem_ty: elem_type.clone(),
    };
    // Phase 3 maps the scan op over the output, combining with block offsets.
    let phase3_soac = SoacOp::Map {
        lam: op.clone(),
        inputs: vec![output_input],
    };
    let phase3_body = soac_term(phase3_soac, elem_type.clone(), span);
    let phase3_def = make_entry_def(&phase3_name, phase3_body, elem_type.clone(), program);

    // Pipeline.
    let input_bindings = collect_soac_bindings(&analysis.soac);
    let output_idx = input_bindings.len();
    let block_sums_idx = input_bindings.len() + 1;
    let block_offsets_idx = input_bindings.len() + 2;

    let mut all_bindings = input_bindings;
    all_bindings.push(Binding::StorageBuffer {
        set: output_binding.0,
        binding: output_binding.1,
        access: Access::ReadWrite,
        usage: BufferUsage::Output,
        name: format!("{}_output", entry_name),
    });
    all_bindings.push(Binding::StorageBuffer {
        set: block_sums_binding.0,
        binding: block_sums_binding.1,
        access: Access::ReadWrite,
        usage: BufferUsage::Intermediate,
        name: format!("{}_block_sums", entry_name),
    });
    all_bindings.push(Binding::StorageBuffer {
        set: block_offsets_binding.0,
        binding: block_offsets_binding.1,
        access: Access::ReadWrite,
        usage: BufferUsage::Intermediate,
        name: format!("{}_block_offsets", entry_name),
    });

    let input_indices: Vec<usize> = (0..output_idx).collect();

    let pipeline = Pipeline::MultiCompute(MultiComputePipeline {
        bindings: all_bindings,
        stages: vec![
            ComputeStage {
                entry_point: phase1_name.clone(),
                workgroup_size: LOCAL_SIZE,
                dispatch_size: DispatchSize::DerivedFromInputLength {
                    workgroup_size: TOTAL_THREADS,
                },
                reads: input_indices.clone(),
                writes: vec![output_idx, block_sums_idx],
            },
            ComputeStage {
                entry_point: phase2_name.clone(),
                workgroup_size: (1, 1, 1),
                dispatch_size: DispatchSize::Fixed { x: 1, y: 1, z: 1 },
                reads: vec![block_sums_idx],
                writes: vec![block_offsets_idx],
            },
            ComputeStage {
                entry_point: phase3_name.clone(),
                workgroup_size: LOCAL_SIZE,
                dispatch_size: DispatchSize::DerivedFromInputLength {
                    workgroup_size: TOTAL_THREADS,
                },
                reads: vec![block_offsets_idx],
                writes: vec![output_idx],
            },
        ],
    });

    (vec![phase1_def, phase2_def, phase3_def], pipeline)
}

// =============================================================================
// Helpers
// =============================================================================

/// Rebuild a SOAC term with its prefix let-bindings.
fn rebuild_soac_with_lets(
    soac: &ParallelizableSoac,
    prefix_lets: &[(SymbolId, Type<TypeName>, Term)],
    result_ty: Type<TypeName>,
    span: ast::Span,
) -> Term {
    let soac_op = match soac {
        ParallelizableSoac::Map { lam, inputs, .. } => SoacOp::Map {
            lam: lam.clone(),
            inputs: inputs.clone(),
        },
        ParallelizableSoac::Reduce { op, ne, input, .. } => SoacOp::Reduce {
            op: op.clone(),
            ne: ne.clone(),
            input: input.clone(),
            props: ReduceProps::default(),
        },
        ParallelizableSoac::Redomap {
            op,
            reduce_op,
            ne,
            inputs,
            ..
        } => SoacOp::Redomap {
            op: op.clone(),
            reduce_op: reduce_op.clone(),
            ne: ne.clone(),
            inputs: inputs.clone(),
            props: ReduceProps::default(),
        },
        ParallelizableSoac::Scan { op, ne, input, .. } => SoacOp::Scan {
            op: op.clone(),
            ne: ne.clone(),
            input: input.clone(),
        },
    };

    let mut body = soac_term(soac_op, result_ty.clone(), span);

    for (name, ty, rhs) in prefix_lets.iter().rev() {
        body = Term {
            id: TermId(0),
            ty: result_ty.clone(),
            span,
            kind: TermKind::Let {
                name: *name,
                name_ty: ty.clone(),
                rhs: Box::new(rhs.clone()),
                body: Box::new(body),
            },
        };
    }

    body
}

fn soac_term(soac: SoacOp, ty: Type<TypeName>, span: ast::Span) -> Term {
    Term {
        id: TermId(0),
        ty,
        span,
        kind: TermKind::Soac(soac),
    }
}

fn uint_lit(val: u64, span: ast::Span) -> Term {
    Term {
        id: TermId(0),
        ty: Type::Constructed(TypeName::UInt(32), vec![]),
        span,
        kind: TermKind::IntLit(val.to_string()),
    }
}

fn make_entry_def(name: &str, body: Term, return_ty: Type<TypeName>, program: &mut Program) -> Def {
    let sym = program.symbols.alloc(name.to_string());
    program.def_syms.insert(name.to_string(), sym);

    let dummy_span = ast::Span::new(0, 0, 0, 0);
    let dummy_expr = ast::Node {
        h: ast::Header {
            id: ast::NodeId(0),
            span: dummy_span,
        },
        kind: ast::ExprKind::Unit,
    };

    Def {
        name: sym,
        ty: return_ty,
        body,
        meta: DefMeta::EntryPoint(Box::new(ast::EntryDecl {
            entry_type: Attribute::Compute,
            name: name.to_string(),
            name_span: dummy_span,
            size_params: vec![],
            type_params: vec![],
            params: vec![],
            outputs: vec![],
            body: dummy_expr,
        })),
        arity: 0,
    }
}

/// Collect storage buffer bindings from a SOAC's input provenance.
fn collect_soac_bindings(soac: &ParallelizableSoac) -> Vec<Binding> {
    let provenances = match soac {
        ParallelizableSoac::Map { provenances, .. } => provenances.as_slice(),
        ParallelizableSoac::Reduce { provenance, .. } => std::slice::from_ref(provenance),
        ParallelizableSoac::Redomap { provenances, .. } => provenances.as_slice(),
        ParallelizableSoac::Scan { provenance, .. } => std::slice::from_ref(provenance),
    };

    let mut bindings = Vec::new();
    for (i, p) in provenances.iter().enumerate() {
        if let ArrayProvenance::Storage { set, binding, .. } = p {
            bindings.push(Binding::StorageBuffer {
                set: *set,
                binding: *binding,
                access: Access::ReadOnly,
                usage: BufferUsage::Input,
                name: format!("input_{}", i),
            });
        }
    }
    bindings
}

fn collect_input_bindings(_analyses: &HashMap<SymbolId, EntryAnalysis>, _entry: SymbolId) -> Vec<Binding> {
    // For non-parallelized entries, we don't have detailed binding info at TLC level.
    // The SSA pass will fill this in.
    vec![]
}

fn build_default_pipeline(program: &Program) -> PipelineDescriptor {
    let mut pipelines = Vec::new();

    for def in &program.defs {
        if let DefMeta::EntryPoint(ref decl) = def.meta {
            let name = program.symbols.get(def.name).cloned().unwrap_or_default();
            if decl.entry_type.is_compute() {
                pipelines.push(Pipeline::Compute(ComputePipeline {
                    entry_point: name,
                    workgroup_size: LOCAL_SIZE,
                    dispatch_size: DispatchSize::DerivedFromInputLength {
                        workgroup_size: TOTAL_THREADS,
                    },
                    bindings: vec![],
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
                    bindings: vec![],
                    vertex_inputs: vec![],
                    fragment_outputs: vec![],
                }));
            }
        }
    }

    PipelineDescriptor { pipelines }
}
