//! TLC-level SOAC parallelization.
//!
//! Stage A: Analyze compute entry points to find parallelizable SOACs.
//! Stage B: Restructure the program — create new entry points with chunked SOACs,
//!          allocate intermediate storage buffers, build pipeline descriptor.
//!
//! Loop creation and storage lowering stay in SSA (`to_ssa` + `soac_lower`).

use crate::ast::{self, TypeName};
use crate::interface::{self, Attribute};
use crate::pipeline_descriptor::*;
use crate::{SymbolId, SymbolTable};
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
///
/// Holds the original `SoacOp` — callers that need per-variant logic
/// pattern-match `original`. `provenances` carries one entry per input
/// (length 1 for Reduce/Scan, N for Map/Redomap with N inputs).
///
/// `analyze_soac` is the only constructor and guarantees `original` is
/// one of `Map`/`Reduce`/`Redomap`/`Scan` — the non-parallelizable
/// variants (Filter, Scatter, ReduceByIndex) never appear here.
#[derive(Debug, Clone)]
pub struct SoacAnalysis {
    pub original: SoacOp,
    pub provenances: Vec<ArrayProvenance>,
}

impl SoacAnalysis {
    /// Every input ArrayExpr the SOAC consumes, in source order.
    pub fn inputs(&self) -> Vec<&ArrayExpr> {
        match &self.original {
            SoacOp::Map { inputs, .. } | SoacOp::Redomap { inputs, .. } => inputs.iter().collect(),
            SoacOp::Reduce { input, .. } | SoacOp::Scan { input, .. } => vec![input],
            _ => unreachable!("non-parallelizable SoacOp in SoacAnalysis"),
        }
    }

    /// Neutral/initial value — present for Reduce/Redomap/Scan, `None` for Map.
    pub fn ne(&self) -> Option<&Term> {
        match &self.original {
            SoacOp::Map { .. } => None,
            SoacOp::Reduce { ne, .. } | SoacOp::Redomap { ne, .. } | SoacOp::Scan { ne, .. } => Some(ne),
            _ => unreachable!("non-parallelizable SoacOp in SoacAnalysis"),
        }
    }

    /// Element type of one iteration's output — Map/Scan elem type (from the
    /// per-element lambda's ret_ty); Reduce/Redomap acc type (from `ne.ty`).
    pub fn result_elem_type(&self) -> Type<TypeName> {
        match &self.original {
            SoacOp::Map { lam, .. } => lam.ret_ty.clone(),
            SoacOp::Reduce { ne, .. } | SoacOp::Redomap { ne, .. } | SoacOp::Scan { ne, .. } => {
                ne.ty.clone()
            }
            _ => unreachable!("non-parallelizable SoacOp in SoacAnalysis"),
        }
    }
}

/// Result of analyzing a compute entry point.
#[derive(Debug, Clone)]
pub(super) struct EntryAnalysis {
    pub def_name: SymbolId,
    pub soac: SoacAnalysis,
    /// Let-binding prefix before the SOAC.
    pub prefix_lets: Vec<(SymbolId, Type<TypeName>, Term)>,
    /// The subset of the original entry's params that the SOAC and
    /// `prefix_lets` actually reference. Phase entries must re-declare
    /// these as params so the references don't leak out as free
    /// globals during SPIR-V emission.
    pub required_params: Vec<(SymbolId, Type<TypeName>)>,
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

        if let Some(analysis) = analyze_entry(def, &program.symbols) {
            results.insert(def.name, analysis);
        }
    }

    results
}

/// Find the SOAC (if any) that structurally produces the entry's return
/// value, along with the `prefix_lets` that must be rebuilt around each
/// restructured phase body.
///
/// Walks tail position through transparent wrappers:
///   - `Lambda`  — entry params come as a flat Lambda around the body.
///   - `Force`   — no-op marker, peeled transparently.
///   - `Let`     — collected into `prefix_lets`, continue into the body.
///   - `Var(x)`  — follow through to `x`'s defining let (removing that
///                 binding from `prefix_lets`, because the SOAC is now
///                 inlined in its place).
///
/// Anything else at tail — `If`, `Loop`, `App`, `BinOp`, `Tuple`,
/// literals, etc. — means the SOAC (if any) is not structurally the
/// entry's result; returns `None`.
/// Scope stack used by `analyze_entry` to track what's in scope as the
/// tail-position walk descends. Each frame is one binding; we never pop
/// (walking strictly in tail position), so the "stack" only grows.
///
/// Variants carry binding data:
/// - `LambdaParam` — from a peeled outer Lambda. These are the candidates
///   for `EntryAnalysis.required_params`.
/// - `Let` — collected as the walk passes through let-chains. These end
///   up as `EntryAnalysis.prefix_lets` (unless consumed by Var-follow).
#[derive(Debug)]
enum ScopeFrame {
    LambdaParam {
        sym: SymbolId,
        ty: Type<TypeName>,
    },
    Let {
        sym: SymbolId,
        ty: Type<TypeName>,
        rhs: Term,
    },
}

#[derive(Debug, Default)]
struct ScopeStack {
    frames: Vec<ScopeFrame>,
}

impl ScopeStack {
    fn push_lambda_params<'a>(&mut self, params: impl IntoIterator<Item = &'a (SymbolId, Type<TypeName>)>) {
        for (s, t) in params {
            self.frames.push(ScopeFrame::LambdaParam {
                sym: *s,
                ty: t.clone(),
            });
        }
    }
    fn push_let(&mut self, sym: SymbolId, ty: Type<TypeName>, rhs: Term) {
        self.frames.push(ScopeFrame::Let { sym, ty, rhs });
    }
    /// Remove and return the innermost `Let` binding for `sym`, or `None`
    /// if `sym` isn't let-bound in any frame. Used by Var-follow.
    fn remove_let(&mut self, sym: SymbolId) -> Option<(Type<TypeName>, Term)> {
        let idx =
            self.frames.iter().rposition(|f| matches!(f, ScopeFrame::Let { sym: s, .. } if *s == sym))?;
        match self.frames.remove(idx) {
            ScopeFrame::Let { ty, rhs, .. } => Some((ty, rhs)),
            _ => unreachable!(),
        }
    }
    fn is_lambda_param(&self, sym: SymbolId) -> bool {
        self.frames.iter().any(|f| matches!(f, ScopeFrame::LambdaParam { sym: s, .. } if *s == sym))
    }
    /// Return captured Lambda params in insertion order (outermost first).
    fn captured_params(&self) -> Vec<(SymbolId, Type<TypeName>)> {
        self.frames
            .iter()
            .filter_map(|f| match f {
                ScopeFrame::LambdaParam { sym, ty } => Some((*sym, ty.clone())),
                _ => None,
            })
            .collect()
    }
    /// Consume the stack and return its `Let` frames in insertion order
    /// (outermost first), i.e. the `prefix_lets` array for the phase bodies.
    fn into_prefix_lets(self) -> Vec<(SymbolId, Type<TypeName>, Term)> {
        self.frames
            .into_iter()
            .filter_map(|f| match f {
                ScopeFrame::Let { sym, ty, rhs } => Some((sym, ty, rhs)),
                _ => None,
            })
            .collect()
    }
}

pub(super) fn analyze_entry(def: &Def, symbols: &SymbolTable) -> Option<EntryAnalysis> {
    let mut scope = ScopeStack::default();
    let mut current: Term = def.body.clone();

    loop {
        let Term { ty, kind, .. } = current;
        match kind {
            TermKind::Lambda(lam) => {
                scope.push_lambda_params(&lam.params);
                current = *lam.body;
            }
            TermKind::Force(inner) => {
                current = *inner;
            }
            TermKind::Let {
                name,
                name_ty,
                rhs,
                body,
            } => {
                scope.push_let(name, name_ty, *rhs);
                current = *body;
            }
            TermKind::Soac(soac) => {
                let parallelizable = analyze_soac(&soac, &ty, symbols)?;
                let captured_params = scope.captured_params();
                let prefix_lets = scope.into_prefix_lets();
                let required_params =
                    compute_required_params(&parallelizable, &prefix_lets, &captured_params, symbols);
                return Some(EntryAnalysis {
                    def_name: def.name,
                    soac: parallelizable,
                    prefix_lets,
                    required_params,
                });
            }
            TermKind::Var(sym) => {
                // Var-follow: the tail is an alias. If `sym` is an entry
                // param, the entry returns a param — not a SOAC, reject.
                if scope.is_lambda_param(sym) {
                    return None;
                }
                // Otherwise try to consume a let binding.
                match scope.remove_let(sym) {
                    Some((_ty, rhs)) => current = rhs,
                    None => return None,
                }
            }
            _ => return None,
        }
    }
}

/// Compute the subset of outer entry params that the tail SOAC + retained
/// prefix_lets actually reference. Phase entries must re-declare these
/// as their own params or the references leak out as undefined globals
/// during SPIR-V emission.
fn compute_required_params(
    soac: &SoacAnalysis,
    prefix_lets: &[(SymbolId, Type<TypeName>, Term)],
    captured_params: &[(SymbolId, Type<TypeName>)],
    symbols: &SymbolTable,
) -> Vec<(SymbolId, Type<TypeName>)> {
    let mut fv = FreeVarSet::new();

    // Each prefix RHS is evaluated with all *previous* prefix names in
    // scope. The SOAC sees the full prefix in scope.
    for (name, _ty, rhs) in prefix_lets {
        fv.add_term(rhs, symbols);
        fv.bind(*name);
    }
    fv.add_soac(&soac.original, symbols);

    captured_params.iter().filter(|(s, _)| fv.contains(*s)).cloned().collect()
}

/// Thin wrapper around `defunctionalize::collect_free_vars` that
/// accumulates user-level free SymbolIds — passing empty `top_level` and
/// `known_defs` sets and stripping the `Term`s down to bare SymbolIds.
/// `bind` mutates the in-scope set so subsequent `add_*` calls treat
/// that name as bound.
struct FreeVarSet {
    bound: std::collections::HashSet<SymbolId>,
    free_syms: std::collections::HashSet<SymbolId>,
    free: Vec<Term>,                           // walker-required scratch
    seen: std::collections::HashSet<SymbolId>, // walker-required scratch
    top_level: std::collections::HashSet<SymbolId>,
    known_defs: std::collections::HashSet<String>,
}

impl FreeVarSet {
    fn new() -> Self {
        FreeVarSet {
            bound: std::collections::HashSet::new(),
            free_syms: std::collections::HashSet::new(),
            free: Vec::new(),
            seen: std::collections::HashSet::new(),
            top_level: std::collections::HashSet::new(),
            known_defs: std::collections::HashSet::new(),
        }
    }
    fn bind(&mut self, sym: SymbolId) {
        self.bound.insert(sym);
    }
    fn contains(&self, sym: SymbolId) -> bool {
        self.free_syms.contains(&sym)
    }
    fn add_term(&mut self, term: &Term, symbols: &SymbolTable) {
        super::defunctionalize::collect_free_vars(
            term,
            &self.bound,
            &self.top_level,
            &self.known_defs,
            symbols,
            &mut self.free,
            &mut self.seen,
        );
        self.harvest();
    }
    fn add_soac(&mut self, soac: &SoacOp, symbols: &SymbolTable) {
        // Wrap in a throwaway Term so we can reuse `collect_free_vars`.
        let term = Term {
            id: TermId(0),
            ty: Type::Variable(0),
            span: ast::Span::new(0, 0, 0, 0),
            kind: TermKind::Soac(soac.clone()),
        };
        self.add_term(&term, symbols);
    }
    fn harvest(&mut self) {
        for t in self.free.drain(..) {
            if let TermKind::Var(s) = &t.kind {
                self.free_syms.insert(*s);
            }
        }
    }
}

/// Analyze a SOAC, rejecting non-parallelizable variants (Filter,
/// Scatter, ReduceByIndex). Inputs that are `Ref(App(_w_range, ...))`
/// get normalized into `ArrayExpr::Range` so Stage B's builders can use
/// their existing Range-aware paths. Returns a `SoacAnalysis` that
/// holds the (possibly-normalized) `SoacOp` plus one provenance per
/// input.
fn analyze_soac(soac: &SoacOp, _result_ty: &Type<TypeName>, symbols: &SymbolTable) -> Option<SoacAnalysis> {
    let normalized: SoacOp = match soac {
        SoacOp::Map { lam, inputs } => {
            let (norm_inputs, _) = classify_inputs(inputs, symbols)?;
            SoacOp::Map {
                lam: lam.clone(),
                inputs: norm_inputs,
            }
        }
        SoacOp::Reduce { op, ne, input, props } => {
            let (norm_input, _) = classify_single(input, symbols)?;
            SoacOp::Reduce {
                op: op.clone(),
                ne: ne.clone(),
                input: norm_input,
                props: props.clone(),
            }
        }
        SoacOp::Redomap {
            op,
            reduce_op,
            ne,
            inputs,
            props,
        } => {
            let (norm_inputs, _) = classify_inputs(inputs, symbols)?;
            SoacOp::Redomap {
                op: op.clone(),
                reduce_op: reduce_op.clone(),
                ne: ne.clone(),
                inputs: norm_inputs,
                props: props.clone(),
            }
        }
        SoacOp::Scan { op, ne, input } => {
            let (norm_input, _) = classify_single(input, symbols)?;
            SoacOp::Scan {
                op: op.clone(),
                ne: ne.clone(),
                input: norm_input,
            }
        }
        _ => return None,
    };

    // Re-derive provenances from the normalized inputs. Build the analysis
    // so its `inputs()` method is the single source of truth for "what are
    // the inputs".
    let analysis_stub = SoacAnalysis {
        original: normalized,
        provenances: Vec::new(),
    };
    let provenances: Vec<ArrayProvenance> =
        analysis_stub.inputs().iter().map(|ae| classify_input(ae)).collect::<Option<Vec<_>>>()?;
    Some(SoacAnalysis {
        original: analysis_stub.original,
        provenances,
    })
}

/// Classify a list of inputs and return their normalized ArrayExprs
/// along with their provenances. Normalization rewrites
/// `Ref(App(_w_range, [start, end, kind]))` → `Range { start, len }`.
fn classify_inputs(
    inputs: &[ArrayExpr],
    symbols: &SymbolTable,
) -> Option<(Vec<ArrayExpr>, Vec<ArrayProvenance>)> {
    let mut norm_inputs = Vec::with_capacity(inputs.len());
    let mut provenances = Vec::with_capacity(inputs.len());
    for input in inputs {
        let (ni, pv) = classify_single(input, symbols)?;
        norm_inputs.push(ni);
        provenances.push(pv);
    }
    Some((norm_inputs, provenances))
}

fn classify_single(input: &ArrayExpr, symbols: &SymbolTable) -> Option<(ArrayExpr, ArrayProvenance)> {
    // First, try to normalize Ref(App(_w_range, ...)) into Range.
    let normalized = normalize_range_ref(input, symbols).unwrap_or_else(|| input.clone());
    let provenance = classify_input(&normalized)?;
    Some((normalized, provenance))
}

/// If `input` is `Ref(App(Var(sym), args))` with sym named `_w_range` or
/// `_w_range_step`, synthesize `ArrayExpr::Range { start, len }` where
/// `len` is `end - start` (or a step-aware variant for `_w_range_step`).
///
/// Returns `None` if `input` doesn't match the range-App shape; the caller
/// keeps the original ArrayExpr.
fn normalize_range_ref(input: &ArrayExpr, symbols: &SymbolTable) -> Option<ArrayExpr> {
    let inner = match input {
        ArrayExpr::Ref(t) => t.as_ref(),
        _ => return None,
    };
    let (func, args) = match &inner.kind {
        TermKind::App { func, args } => (func, args),
        _ => return None,
    };
    let sym = match &func.kind {
        TermKind::Var(s) => *s,
        _ => return None,
    };
    let name = symbols.get(sym)?;
    match name.as_str() {
        "_w_range" if args.len() == 3 => {
            // args = [start, end, kind]. len = end - start.
            let start = args[0].clone();
            let end = &args[1];
            let len = binop("-", end.clone(), start.clone(), end.ty.clone(), end.span);
            Some(ArrayExpr::Range {
                start: Box::new(start),
                len: Box::new(len),
            })
        }
        "_w_range_step" if args.len() == 4 => {
            // args = [start, step, end, kind]. len = (end - start) / step.
            let start = args[0].clone();
            let step = &args[1];
            let end = &args[2];
            let diff = binop("-", end.clone(), start.clone(), end.ty.clone(), end.span);
            let len = binop("/", diff, step.clone(), end.ty.clone(), end.span);
            Some(ArrayExpr::Range {
                start: Box::new(start),
                len: Box::new(len),
            })
        }
        _ => None,
    }
}

fn binop(op: &str, lhs: Term, rhs: Term, ty: Type<TypeName>, span: ast::Span) -> Term {
    Term {
        id: TermId(0),
        ty: ty.clone(),
        span,
        kind: TermKind::App {
            func: Box::new(Term {
                id: TermId(0),
                ty: Type::Constructed(
                    TypeName::Arrow,
                    vec![
                        ty.clone(),
                        Type::Constructed(TypeName::Arrow, vec![ty.clone(), ty.clone()]),
                    ],
                ),
                span,
                kind: TermKind::BinOp(ast::BinaryOp { op: op.to_string() }),
            }),
            args: vec![lhs, rhs],
        },
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
                // For non-parallelized compute entries, the SSA-stage
                // codegen derives binding info from the entry's param
                // types. No binding info is extractable at this TLC
                // stage, so we emit an empty vec.
                let input_bindings: Vec<Binding> = vec![];
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
        let plan = make_lowering_plan(analysis, &entry_name, next_binding, &mut program);
        next_binding += plan.extra_bindings_used;
        if let Some(removed) = plan.removed_entry {
            removed_entries.push(removed);
        }
        new_defs.extend(plan.new_defs);
        pipelines.push(plan.pipeline);
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
// Lowering plan dispatcher
// =============================================================================

struct LoweringPlan {
    removed_entry: Option<SymbolId>,
    new_defs: Vec<Def>,
    pipeline: Pipeline,
    extra_bindings_used: u32,
}

fn make_lowering_plan(
    analysis: &EntryAnalysis,
    entry_name: &str,
    next_binding: u32,
    program: &mut Program,
) -> LoweringPlan {
    match &analysis.soac.original {
        SoacOp::Map { .. } => make_map_plan(analysis, entry_name),
        SoacOp::Reduce { op, ne, .. } => {
            make_two_phase_plan(analysis, entry_name, op, ne, next_binding, program)
        }
        SoacOp::Redomap { reduce_op, ne, .. } => {
            make_two_phase_plan(analysis, entry_name, reduce_op, ne, next_binding, program)
        }
        SoacOp::Scan { op, ne, input } => {
            make_scan_plan(analysis, entry_name, op, ne, input, next_binding, program)
        }
        _ => unreachable!("analyze_soac rejected non-parallelizable variants"),
    }
}

fn make_map_plan(analysis: &EntryAnalysis, entry_name: &str) -> LoweringPlan {
    let input_bindings = collect_soac_bindings(&analysis.soac);
    let pipeline = Pipeline::Compute(ComputePipeline {
        entry_point: entry_name.to_string(),
        workgroup_size: LOCAL_SIZE,
        dispatch_size: DispatchSize::DerivedFromInputLength {
            workgroup_size: TOTAL_THREADS,
        },
        bindings: input_bindings,
    });
    LoweringPlan {
        removed_entry: None,
        new_defs: Vec::new(),
        pipeline,
        extra_bindings_used: 0,
    }
}

fn make_two_phase_plan(
    analysis: &EntryAnalysis,
    entry_name: &str,
    reduce_op: &Lambda,
    ne: &Term,
    next_binding: u32,
    program: &mut Program,
) -> LoweringPlan {
    let partials_binding = (0, next_binding);
    let result_binding = (0, next_binding + 1);
    let elem_type = analysis.soac.result_elem_type();
    let (entries, pipeline) = build_two_phase_entries(
        entry_name,
        analysis,
        reduce_op,
        ne,
        &elem_type,
        partials_binding,
        result_binding,
        program,
    );
    LoweringPlan {
        removed_entry: Some(analysis.def_name),
        new_defs: entries,
        pipeline,
        extra_bindings_used: 2,
    }
}

fn make_scan_plan(
    analysis: &EntryAnalysis,
    entry_name: &str,
    op: &Lambda,
    ne: &Term,
    input: &ArrayExpr,
    next_binding: u32,
    program: &mut Program,
) -> LoweringPlan {
    let output_binding = (0, next_binding);
    let block_sums_binding = (0, next_binding + 1);
    let block_offsets_binding = (0, next_binding + 2);
    let elem_type = analysis.soac.result_elem_type();
    let (entries, pipeline) = build_scan_entries(
        entry_name,
        analysis,
        op,
        ne,
        input,
        &elem_type,
        output_binding,
        block_sums_binding,
        block_offsets_binding,
        program,
    );
    LoweringPlan {
        removed_entry: Some(analysis.def_name),
        new_defs: entries,
        pipeline,
        extra_bindings_used: 3,
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

    let unit_ty = Type::Constructed(TypeName::Unit, vec![]);

    // Phase 1: chunked SOAC, each thread writes its partial to partials[tid].
    let phase1_name = format!("{}_phase1_chunks", entry_name);
    let phase1_body = build_chunked_soac_body(
        &analysis.soac,
        &analysis.prefix_lets,
        elem_type.clone(),
        span,
        program,
        Some(partials_binding),
    );
    // Phase 1 storage interface: reads whatever the input SOAC declares
    // (those come in via the TLC body already), writes `partials` at `tid`.
    let phase1_bindings = vec![interface::StorageBindingDecl {
        set: partials_binding.0,
        binding: partials_binding.1,
        role: interface::StorageRole::Intermediate,
        elem_ty: elem_type.clone(),
    }];
    let phase1_def = make_entry_def(
        &phase1_name,
        phase1_body,
        unit_ty.clone(),
        &analysis.required_params,
        phase1_bindings,
        program,
    );

    // Phase 2: reduce over the partials buffer, write result to result_binding[0].
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
    let phase2_soac_term = soac_term(phase2_soac, elem_type.clone(), span);
    let r_sym = program.symbols.alloc("_par_out".into());
    let phase2_store = intrinsic_term(
        "_w_intrinsic_storage_store",
        vec![
            uint_lit(result_binding.0 as u64, span),
            uint_lit(result_binding.1 as u64, span),
            uint_lit(0, span),
            var_term(r_sym, elem_type.clone(), span),
        ],
        unit_ty.clone(),
        span,
        program,
    );
    let phase2_body = let_term(r_sym, elem_type.clone(), phase2_soac_term, phase2_store, span);
    // Phase 2 storage interface: reads `partials` (as an Intermediate) and
    // writes the final user-visible `result`.
    let phase2_bindings = vec![
        interface::StorageBindingDecl {
            set: partials_binding.0,
            binding: partials_binding.1,
            role: interface::StorageRole::Intermediate,
            elem_ty: elem_type.clone(),
        },
        interface::StorageBindingDecl {
            set: result_binding.0,
            binding: result_binding.1,
            role: interface::StorageRole::Output,
            elem_ty: elem_type.clone(),
        },
    ];
    let phase2_def = make_entry_def(
        &phase2_name,
        phase2_body,
        unit_ty.clone(),
        &analysis.required_params,
        phase2_bindings,
        program,
    );

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
    let phase1_body = build_chunked_soac_body(
        &analysis.soac,
        &analysis.prefix_lets,
        elem_type.clone(),
        span,
        program,
        None,
    );
    let phase1_def = make_entry_def(
        &phase1_name,
        phase1_body,
        elem_type.clone(),
        &analysis.required_params,
        Vec::new(),
        program,
    );

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
    let phase2_def = make_entry_def(
        &phase2_name,
        phase2_body,
        elem_type.clone(),
        &analysis.required_params,
        Vec::new(),
        program,
    );

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
    let phase3_def = make_entry_def(
        &phase3_name,
        phase3_body,
        elem_type.clone(),
        &analysis.required_params,
        Vec::new(),
        program,
    );

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
// Chunked SOAC body builder
// =============================================================================

/// Build a chunked SOAC body for a parallel entry point.
///
/// Generates:
/// ```text
/// let tid = _w_intrinsic_thread_id() in
/// let total = 64 in
/// let chunk_size = (input_len + total - 1) / total in
/// let chunk_start = tid * chunk_size in
/// let chunk_len = u32.min(chunk_size, input_len - chunk_start) in
/// <soac over chunked inputs>
/// ```
fn build_chunked_soac_body(
    soac: &SoacAnalysis,
    prefix_lets: &[(SymbolId, Type<TypeName>, Term)],
    result_ty: Type<TypeName>,
    span: ast::Span,
    program: &mut Program,
    // If Some((set, binding)), wrap the inner SOAC with
    //   let r = <soac> in _w_intrinsic_storage_store(set, binding, tid, r)
    // so each thread's partial lands at partials[tid]. Return type then
    // becomes Unit rather than `result_ty`.
    write_partial_to: Option<(u32, u32)>,
) -> Term {
    let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);

    // Allocate symbols for chunk arithmetic bindings.
    let tid_sym = program.symbols.alloc("_par_tid".into());
    let total_sym = program.symbols.alloc("_par_total".into());
    let input_len_sym = program.symbols.alloc("_par_input_len".into());
    let chunk_size_sym = program.symbols.alloc("_par_chunk_size".into());
    let chunk_start_sym = program.symbols.alloc("_par_chunk_start".into());
    let chunk_len_sym = program.symbols.alloc("_par_chunk_len".into());

    // Build chunked input ArrayExprs (replace storage buffer offset/len with chunk range).
    let chunk_start_var = var_term(chunk_start_sym, u32_ty.clone(), span);
    let chunk_len_var = var_term(chunk_len_sym, u32_ty.clone(), span);

    // Rebuild the SOAC with inputs rebased to (chunk_start, chunk_len).
    // Pattern-match on the original SoacOp — `analyze_soac` guarantees one
    // of the four parallel variants.
    let chunked_soac = match &soac.original {
        SoacOp::Map { lam, inputs } => {
            let chunked_inputs = inputs
                .iter()
                .map(|input| chunk_array_expr(input, &chunk_start_var, &chunk_len_var))
                .collect();
            SoacOp::Map {
                lam: lam.clone(),
                inputs: chunked_inputs,
            }
        }
        SoacOp::Reduce { op, ne, input, props } => SoacOp::Reduce {
            op: op.clone(),
            ne: ne.clone(),
            input: chunk_array_expr(input, &chunk_start_var, &chunk_len_var),
            props: props.clone(),
        },
        SoacOp::Redomap {
            op,
            reduce_op,
            ne,
            inputs,
            props,
        } => {
            let chunked_inputs = inputs
                .iter()
                .map(|input| chunk_array_expr(input, &chunk_start_var, &chunk_len_var))
                .collect();
            SoacOp::Redomap {
                op: op.clone(),
                reduce_op: reduce_op.clone(),
                ne: ne.clone(),
                inputs: chunked_inputs,
                props: props.clone(),
            }
        }
        SoacOp::Scan { op, ne, input } => SoacOp::Scan {
            op: op.clone(),
            ne: ne.clone(),
            input: chunk_array_expr(input, &chunk_start_var, &chunk_len_var),
        },
        _ => unreachable!("analyze_soac rejected non-parallelizable variants"),
    };

    // Get the input length term from the first input's provenance.
    let input_len_term = get_input_len(soac, span);

    // Build the body bottom-up: SOAC first, then wrap with let bindings.
    let mut body = soac_term(chunked_soac, result_ty.clone(), span);

    // If requested, wrap the SOAC with `let r = <soac> in store(set, binding, tid, r)`
    // so each thread's partial result lands in its own slot. tid_sym is bound
    // further out in the let-chain and is in scope here.
    if let Some((set, binding)) = write_partial_to {
        let unit_ty = Type::Constructed(TypeName::Unit, vec![]);
        let r_sym = program.symbols.alloc("_par_out".into());
        let tid_var = var_term(tid_sym, u32_ty.clone(), span);
        let r_var = var_term(r_sym, result_ty.clone(), span);
        let store = intrinsic_term(
            "_w_intrinsic_storage_store",
            vec![
                uint_lit(set as u64, span),
                uint_lit(binding as u64, span),
                tid_var,
                r_var,
            ],
            unit_ty,
            span,
            program,
        );
        body = let_term(r_sym, result_ty.clone(), body, store, span);
    }

    // Wrap with prefix lets (from the original entry body).
    for (name, ty, rhs) in prefix_lets.iter().rev() {
        body = let_term(*name, ty.clone(), rhs.clone(), body, span);
    }

    // Wrap with chunk arithmetic lets.
    // chunk_len = if chunk_size < (input_len - chunk_start)
    //             then chunk_size else (input_len - chunk_start)
    // (inlined min — there is no backend-known `_w_u32_min` intrinsic).
    let len_minus_start = binop(
        "-",
        var_term(input_len_sym, u32_ty.clone(), span),
        var_term(chunk_start_sym, u32_ty.clone(), span),
        u32_ty.clone(),
        span,
    );
    let bool_ty = Type::Constructed(TypeName::Bool, vec![]);
    let cond = binop(
        "<",
        var_term(chunk_size_sym, u32_ty.clone(), span),
        len_minus_start.clone(),
        bool_ty,
        span,
    );
    let min_expr = Term {
        id: TermId(0),
        ty: u32_ty.clone(),
        span,
        kind: TermKind::If {
            cond: Box::new(cond),
            then_branch: Box::new(var_term(chunk_size_sym, u32_ty.clone(), span)),
            else_branch: Box::new(len_minus_start),
        },
    };
    body = let_term(chunk_len_sym, u32_ty.clone(), min_expr, body, span);

    // chunk_start = tid * chunk_size
    let chunk_start_rhs = binop(
        "*",
        var_term(tid_sym, u32_ty.clone(), span),
        var_term(chunk_size_sym, u32_ty.clone(), span),
        u32_ty.clone(),
        span,
    );
    body = let_term(chunk_start_sym, u32_ty.clone(), chunk_start_rhs, body, span);

    // chunk_size = (input_len + total - 1) / total
    let total_minus_1 = binop(
        "-",
        var_term(total_sym, u32_ty.clone(), span),
        uint_lit(1, span),
        u32_ty.clone(),
        span,
    );
    let len_plus = binop(
        "+",
        var_term(input_len_sym, u32_ty.clone(), span),
        total_minus_1,
        u32_ty.clone(),
        span,
    );
    let chunk_size_rhs = binop(
        "/",
        len_plus,
        var_term(total_sym, u32_ty.clone(), span),
        u32_ty.clone(),
        span,
    );
    body = let_term(chunk_size_sym, u32_ty.clone(), chunk_size_rhs, body, span);

    // input_len = <from provenance>
    body = let_term(input_len_sym, u32_ty.clone(), input_len_term, body, span);

    // total = TOTAL_THREADS
    body = let_term(
        total_sym,
        u32_ty.clone(),
        uint_lit(TOTAL_THREADS as u64, span),
        body,
        span,
    );

    // tid = _w_intrinsic_thread_id()
    let tid_rhs = intrinsic_term("_w_intrinsic_thread_id", vec![], u32_ty.clone(), span, program);
    body = let_term(tid_sym, u32_ty.clone(), tid_rhs, body, span);

    body
}

/// Replace a storage buffer's offset/len with chunk-relative values.
fn chunk_array_expr(input: &ArrayExpr, chunk_start: &Term, chunk_len: &Term) -> ArrayExpr {
    match input {
        ArrayExpr::StorageBuffer {
            set,
            binding,
            offset,
            elem_ty,
            ..
        } => {
            // New offset = original_offset + chunk_start
            let new_offset = if is_literal_zero(offset) {
                chunk_start.clone()
            } else {
                binop(
                    "+",
                    (**offset).clone(),
                    chunk_start.clone(),
                    chunk_start.ty.clone(),
                    chunk_start.span,
                )
            };
            ArrayExpr::StorageBuffer {
                set: *set,
                binding: *binding,
                offset: Box::new(new_offset),
                len: Box::new(chunk_len.clone()),
                elem_ty: elem_ty.clone(),
            }
        }
        ArrayExpr::Range { start, len: _ } => {
            // Range: chunk_start..chunk_len starting from original start
            let new_start = binop(
                "+",
                (**start).clone(),
                chunk_start.clone(),
                chunk_start.ty.clone(),
                chunk_start.span,
            );
            ArrayExpr::Range {
                start: Box::new(new_start),
                len: Box::new(chunk_len.clone()),
            }
        }
        other => other.clone(), // shouldn't happen for parallelizable inputs
    }
}

/// Get the input length term from the SOAC's first input.
fn get_input_len(soac: &SoacAnalysis, span: ast::Span) -> Term {
    get_array_expr_len(soac.inputs()[0], span)
}

fn get_array_expr_len(ae: &ArrayExpr, span: ast::Span) -> Term {
    match ae {
        ArrayExpr::StorageBuffer { len, .. } => (**len).clone(),
        ArrayExpr::Range { len, .. } => (**len).clone(),
        _ => uint_lit(0, span), // fallback
    }
}

fn is_literal_zero(term: &Term) -> bool {
    matches!(&term.kind, TermKind::IntLit(s) if s == "0")
}

// =============================================================================
// Term-building helpers
// =============================================================================

fn var_term(sym: SymbolId, ty: Type<TypeName>, span: ast::Span) -> Term {
    Term {
        id: TermId(0),
        ty,
        span,
        kind: TermKind::Var(sym),
    }
}

fn let_term(name: SymbolId, name_ty: Type<TypeName>, rhs: Term, body: Term, span: ast::Span) -> Term {
    let ty = body.ty.clone();
    Term {
        id: TermId(0),
        ty,
        span,
        kind: TermKind::Let {
            name,
            name_ty,
            rhs: Box::new(rhs),
            body: Box::new(body),
        },
    }
}

/// Build an intrinsic call term. Creates a symbol for the intrinsic name.
fn intrinsic_term(
    name: &str,
    args: Vec<Term>,
    ret_ty: Type<TypeName>,
    span: ast::Span,
    program: &mut Program,
) -> Term {
    let sym = if let Some(&existing) = program.def_syms.get(name) {
        existing
    } else {
        let sym = program.symbols.alloc(name.to_string());
        program.def_syms.insert(name.to_string(), sym);
        sym
    };
    Term {
        id: TermId(0),
        ty: ret_ty,
        span,
        kind: TermKind::App {
            func: Box::new(var_term(sym, Type::Variable(0), span)),
            args,
        },
    }
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

fn make_entry_def(
    name: &str,
    body: Term,
    return_ty: Type<TypeName>,
    required_params: &[(SymbolId, Type<TypeName>)],
    storage_bindings: Vec<interface::StorageBindingDecl>,
    program: &mut Program,
) -> Def {
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

    // Wrap the body in a single flat Lambda carrying `required_params`, mirroring
    // `transform_entry`'s convention. Compute the full arrow type.
    let (full_ty, body) = if required_params.is_empty() {
        (return_ty.clone(), body)
    } else {
        let mut ty = return_ty.clone();
        for (_, pt) in required_params.iter().rev() {
            ty = Type::Constructed(TypeName::Arrow, vec![pt.clone(), ty]);
        }
        let lam_body = Term {
            id: TermId(0),
            ty: ty.clone(),
            span: dummy_span,
            kind: TermKind::Lambda(Lambda {
                params: required_params.to_vec(),
                body: Box::new(body),
                ret_ty: return_ty.clone(),
                captures: vec![],
            }),
        };
        (ty, lam_body)
    };

    // Build ast::Pattern entries — from_tlc's entry conversion reads these
    // for IO decoration and size_hint, but phase-entry params need neither,
    // so bare `PatternKind::Name` is enough.
    let ast_params: Vec<ast::Pattern> = required_params
        .iter()
        .map(|(s, _)| {
            let pname = program.symbols.get(*s).cloned().unwrap_or_else(|| format!("p{}", s.0));
            ast::Pattern {
                h: ast::Header {
                    id: ast::NodeId(0),
                    span: dummy_span,
                },
                kind: ast::PatternKind::Name(pname),
            }
        })
        .collect();

    // Declare one anonymous output so `build_entry_outputs` in from_tlc
    // treats the return type as a single storage-bound output and
    // `emit_compute_output_stores` writes it into the partials/result
    // buffer rather than leaving a Return(value) that SPIR-V rejects.
    // (The ty field here is a placeholder — build_entry_outputs reads the
    //  TLC ret_type, not this.)
    let outputs = vec![interface::EntryOutput {
        ty: Type::Constructed(TypeName::Unit, vec![]),
        attribute: None,
    }];

    Def {
        name: sym,
        ty: full_ty,
        body,
        meta: DefMeta::EntryPoint(Box::new(interface::EntryDecl {
            entry_type: Attribute::Compute,
            name: name.to_string(),
            name_span: dummy_span,
            size_params: vec![],
            type_params: vec![],
            params: ast_params,
            outputs,
            storage_bindings,
            body: dummy_expr,
        })),
        arity: required_params.len(),
    }
}

/// Collect storage buffer bindings from a SOAC's input provenances.
fn collect_soac_bindings(soac: &SoacAnalysis) -> Vec<Binding> {
    soac.provenances
        .iter()
        .enumerate()
        .filter_map(|(i, p)| match p {
            ArrayProvenance::Storage { set, binding, .. } => Some(Binding::StorageBuffer {
                set: *set,
                binding: *binding,
                access: Access::ReadOnly,
                usage: BufferUsage::Input,
                name: format!("input_{}", i),
            }),
            _ => None,
        })
        .collect()
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
