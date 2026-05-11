//! TLC-level SOAC parallelization.
//!
//! Stage A: Analyze compute entry points to find parallelizable SOACs.
//! Stage B: Restructure the program — create new entry points with chunked SOACs,
//!          allocate intermediate storage buffers, build pipeline descriptor.
//!
//! Loop creation and storage lowering stay in SSA (`to_ssa` + `soac_lower`).

use crate::ast::{self, TypeName};
use crate::egir::from_tlc::AUTO_STORAGE_SET;
use crate::interface::{self, Attribute};
use crate::pipeline_descriptor::*;
use crate::{SymbolId, SymbolTable};
use polytype::Type;
use std::collections::{HashMap, HashSet};

use super::{
    ArrayExpr, Def, DefMeta, Lambda, LoopKind, Program, ReduceProps, SoacOp, Term, TermId, TermKind,
};

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

/// Every input ArrayExpr a parallelizable SOAC consumes, in source order.
/// Free-function form so callers (`analyze_soac`) can use it on a raw
/// `SoacOp` without building a throwaway `SoacAnalysis`.
fn soac_inputs(soac: &SoacOp) -> Vec<&ArrayExpr> {
    match soac {
        SoacOp::Map { inputs, .. } | SoacOp::Redomap { inputs, .. } => inputs.iter().collect(),
        SoacOp::Reduce { input, .. } | SoacOp::Scan { input, .. } => vec![input],
        _ => unreachable!("non-parallelizable SoacOp in SoacAnalysis"),
    }
}

impl SoacAnalysis {
    /// Every input ArrayExpr the SOAC consumes, in source order.
    pub fn inputs(&self) -> Vec<&ArrayExpr> {
        soac_inputs(&self.original)
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
            SoacOp::Map { lam, .. } => lam.lam.ret_ty.clone(),
            SoacOp::Reduce { ne, .. } | SoacOp::Redomap { ne, .. } | SoacOp::Scan { ne, .. } => {
                ne.ty.clone()
            }
            _ => unreachable!("non-parallelizable SoacOp in SoacAnalysis"),
        }
    }
}

/// Result of analyzing a compute entry point.
#[derive(Debug, Clone)]
struct EntryAnalysis {
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

fn analyze_entry(def: &Def, symbols: &SymbolTable) -> Option<EntryAnalysis> {
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
            TermKind::Var(crate::tlc::VarRef::Symbol(sym)) => {
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
///
/// Reuses `closure_convert::collect_free_vars` with empty `top_level` /
/// `known_defs` sets — no top-level filtering, just "bound vs free"
/// within the fragment we walk.
fn compute_required_params(
    soac: &SoacAnalysis,
    prefix_lets: &[(SymbolId, Type<TypeName>, Term)],
    captured_params: &[(SymbolId, Type<TypeName>)],
    symbols: &SymbolTable,
) -> Vec<(SymbolId, Type<TypeName>)> {
    use std::collections::HashSet;
    let empty_top: HashSet<SymbolId> = HashSet::new();
    let empty_defs: HashSet<String> = HashSet::new();
    let mut bound: HashSet<SymbolId> = HashSet::new();
    let mut free: Vec<Term> = Vec::new();
    let mut seen: HashSet<SymbolId> = HashSet::new();

    // Each prefix RHS is evaluated with all *previous* prefix names in
    // scope. The SOAC sees the full prefix in scope.
    for (name, _ty, rhs) in prefix_lets {
        super::closure_convert::collect_free_vars(
            rhs,
            &bound,
            &empty_top,
            &empty_defs,
            symbols,
            &mut free,
            &mut seen,
        );
        bound.insert(*name);
    }
    // Wrap the SOAC in a throwaway Term so we can reuse the same walker.
    let soac_term = Term {
        id: TermId(0),
        ty: Type::Variable(0),
        span: ast::Span::new(0, 0, 0, 0),
        kind: TermKind::Soac(soac.original.clone()),
    };
    super::closure_convert::collect_free_vars(
        &soac_term,
        &bound,
        &empty_top,
        &empty_defs,
        symbols,
        &mut free,
        &mut seen,
    );

    let free_syms: HashSet<SymbolId> = free
        .iter()
        .filter_map(|t| {
            if let TermKind::Var(crate::tlc::VarRef::Symbol(s)) = &t.kind { Some(*s) } else { None }
        })
        .collect();
    captured_params.iter().filter(|(s, _)| free_syms.contains(s)).cloned().collect()
}

/// Analyze a SOAC, rejecting non-parallelizable variants (Filter,
/// Scatter, ReduceByIndex). Inputs that are `Ref(App(_w_range, ...))`
/// get normalized into `ArrayExpr::Range` so Stage B's builders can use
/// their existing Range-aware paths. Returns a `SoacAnalysis` that
/// holds the (possibly-normalized) `SoacOp` plus one provenance per
/// input.
fn analyze_soac(soac: &SoacOp, _result_ty: &Type<TypeName>, symbols: &SymbolTable) -> Option<SoacAnalysis> {
    let normalized: SoacOp = match soac {
        SoacOp::Map {
            lam,
            inputs,
            consumes_input,
        } => {
            let (norm_inputs, _) = classify_inputs(inputs, symbols)?;
            SoacOp::Map {
                lam: lam.clone(),
                inputs: norm_inputs,
                consumes_input: *consumes_input,
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

    // Re-derive provenances from the normalized inputs. `soac_inputs`
    // is the single source of truth for "what are the inputs".
    let provenances: Vec<ArrayProvenance> =
        soac_inputs(&normalized).iter().map(|ae| classify_input(ae)).collect::<Option<Vec<_>>>()?;
    Some(SoacAnalysis {
        original: normalized,
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
    // Peel off leading `let` wrappers so we can recognize `let N = 256 in
    // _w_range(0, N, ...)` that inline_small doesn't always fold.
    let mut inner = inner;
    while let TermKind::Let { body, .. } = &inner.kind {
        inner = body.as_ref();
    }
    let (func, args) = match &inner.kind {
        TermKind::App { func, args } => (func, args),
        _ => return None,
    };
    let sym = match &func.kind {
        TermKind::Var(crate::tlc::VarRef::Symbol(s)) => *s,
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

// =============================================================================
// Graphical-entry SOAC lifting
// =============================================================================

/// For each graphical entry, walk its body's outer let-chain and hoist
/// reduce/redomap bindings whose RHS is invariant with respect to the
/// entry's per-invocation params. Each lift:
///   * allocates a fresh storage buffer for the scalar result,
///   * emits a compute pre-pass entry `<entry>_prepass_<n>` that
///     evaluates the SOAC and stores the result at index 0,
///   * rewrites the original let-binding's RHS to
///     `_w_intrinsic_storage_index(set, binding, 0)`,
///   * adds an `Input`-role `StorageBindingDecl` to the graphical
///     entry's interface so the backend's binding allowlist admits
///     the load.
///
/// The pre-pass entries land in `program.defs` and will be picked up
/// by `analyze_program` + Stage B in the usual way, producing the
/// two-phase compute pipeline that justifies "multi-stage" — one
/// source file compiles to a chunk/combine pair plus the original
/// vertex+fragment stages.
///
/// Scope (MVP): only reduce/redomap whose result is a scalar. Scan/Map
/// (array results) and deeply nested lets are left for a follow-up.
fn lift_graphical_invariant_soacs(
    program: &mut Program,
    next_binding: &mut u32,
    prepass_result_bindings: &mut HashMap<SymbolId, (u32, u32)>,
) {
    use std::collections::HashSet;

    // Snapshot indices of graphical entry defs — we'll mutate program.defs
    // in the loop, but only the def at `idx` (its body + storage_bindings).
    let indices: Vec<usize> = program
        .defs
        .iter()
        .enumerate()
        .filter_map(|(i, d)| match &d.meta {
            DefMeta::EntryPoint(decl) if !decl.entry_type.is_compute() => Some(i),
            _ => None,
        })
        .collect();

    let mut new_defs: Vec<Def> = Vec::new();

    for idx in indices {
        let entry_name = program.symbols.get(program.defs[idx].name).cloned().unwrap_or_default();
        let entry_params: HashSet<SymbolId> = {
            let (params, _) = peel_lambda_params(&program.defs[idx].body);
            params.iter().map(|(s, _)| *s).collect()
        };

        let body = program.defs[idx].body.clone();
        let mut added_decls: Vec<interface::StorageBindingDecl> = Vec::new();
        let new_body = lift_in_term(
            body,
            &entry_name,
            &entry_params,
            next_binding,
            &mut added_decls,
            &mut new_defs,
            prepass_result_bindings,
            program,
        );

        program.defs[idx].body = new_body;
        if let DefMeta::EntryPoint(ref mut decl) = program.defs[idx].meta {
            decl.storage_bindings.extend(added_decls);
        }
    }

    program.defs.extend(new_defs);
}

/// Return a `Term`'s (possibly-wrapped) lambda params by peeling
/// outer `TermKind::Lambda` layers. Mirrors `extract_params` in
/// `buffer_specialize.rs`.
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

/// Walk outer `Lambda`s and `Let`s, hoisting eligible SOAC-RHSs. Stops
/// descending at the first non-Lambda-non-Let term — that's the tail
/// computation and isn't a lift site.
fn lift_in_term(
    term: Term,
    entry_name: &str,
    entry_params: &std::collections::HashSet<SymbolId>,
    next_binding: &mut u32,
    added_decls: &mut Vec<interface::StorageBindingDecl>,
    new_defs: &mut Vec<Def>,
    prepass_result_bindings: &mut HashMap<SymbolId, (u32, u32)>,
    program: &mut Program,
) -> Term {
    match term.kind {
        TermKind::Lambda(lam) => {
            let Lambda { params, body, ret_ty } = lam;
            let new_body = lift_in_term(
                *body,
                entry_name,
                entry_params,
                next_binding,
                added_decls,
                new_defs,
                prepass_result_bindings,
                program,
            );
            Term {
                id: term.id,
                ty: term.ty,
                span: term.span,
                kind: TermKind::Lambda(Lambda {
                    params,
                    body: Box::new(new_body),
                    ret_ty,
                }),
            }
        }
        TermKind::Let {
            name,
            name_ty,
            rhs,
            body,
        } => {
            let new_rhs = maybe_hoist(
                *rhs,
                entry_name,
                &name_ty,
                entry_params,
                next_binding,
                added_decls,
                new_defs,
                prepass_result_bindings,
                program,
            );
            let new_body = lift_in_term(
                *body,
                entry_name,
                entry_params,
                next_binding,
                added_decls,
                new_defs,
                prepass_result_bindings,
                program,
            );
            Term {
                id: term.id,
                ty: term.ty,
                span: term.span,
                kind: TermKind::Let {
                    name,
                    name_ty,
                    rhs: Box::new(new_rhs),
                    body: Box::new(new_body),
                },
            }
        }
        _ => term,
    }
}

/// If `rhs` is a scalar-returning SOAC (reduce/redomap) whose free
/// vars don't reference entry params, allocate a storage binding,
/// emit a pre-pass compute entry, and replace `rhs` with a load of
/// that binding. Otherwise return `rhs` unchanged.
fn maybe_hoist(
    rhs: Term,
    entry_name: &str,
    name_ty: &Type<TypeName>,
    entry_params: &std::collections::HashSet<SymbolId>,
    next_binding: &mut u32,
    added_decls: &mut Vec<interface::StorageBindingDecl>,
    new_defs: &mut Vec<Def>,
    prepass_result_bindings: &mut HashMap<SymbolId, (u32, u32)>,
    program: &mut Program,
) -> Term {
    // TODO: extend to array-result SOACs (Scan, Map). A pre-pass
    // emitting an array would need to write N slots to storage, the
    // fragment would read back by index instead of at position 0, and
    // Stage B's two-phase plan would grow an array-sized output path.
    // For the scalar-result cases (Reduce, Redomap) the single-slot
    // shape already works end-to-end.
    let is_scalar_soac = matches!(
        &rhs.kind,
        TermKind::Soac(SoacOp::Reduce { .. }) | TermKind::Soac(SoacOp::Redomap { .. })
    );
    if !is_scalar_soac {
        return rhs;
    }

    // Invariance check: none of `rhs`'s free vars may be an entry param.
    if rhs_references_entry_param(&rhs, entry_params, &program.symbols) {
        return rhs;
    }

    // TODO: polymorphic-size free vars. Free vars whose type contains
    // a Size type variable (e.g. `iota(N)` where `N` is a `<[n]>`
    // parameter) pass the entry-param check — `N` isn't an entry
    // param — but the generated pre-pass doesn't have `N` in scope
    // either. Silently emitting the lift in that state produces a
    // pre-pass that references `@size` as an undeclared global and
    // fails backend validation. Panic loudly instead until the lift
    // either (a) refuses the hoist when polymorphic sizes are present
    // or (b) captures the size binding alongside the hoisted SOAC.
    assert_hoist_free_vars_are_grounded(&rhs, entry_params, &program.symbols);

    // Pre-allocate the binding the fragment will load from. Stage B's
    // make_two_phase_plan will use this as the prepass's result_binding
    // (via the prepass_result_bindings map), so phase 2's final store
    // goes exactly here.
    let binding = (AUTO_STORAGE_SET, *next_binding);
    *next_binding += 1;

    let span = rhs.span;
    let prepass_name = format!("{}_prepass_{}", entry_name, added_decls.len());
    let prepass_def = build_prepass_def(&prepass_name, rhs, name_ty.clone(), program);
    prepass_result_bindings.insert(prepass_def.name, binding);
    new_defs.push(prepass_def);

    added_decls.push(interface::StorageBindingDecl {
        set: binding.0,
        binding: binding.1,
        role: interface::StorageRole::Input,
        elem_ty: name_ty.clone(),
    });

    // Rewrite the let RHS to a storage load at position 0.
    intrinsic_term_by_id(
        crate::builtins::catalog().known().storage_index,
        vec![
            uint_lit(binding.0 as u64, span),
            uint_lit(binding.1 as u64, span),
            uint_lit(0, span),
        ],
        name_ty.clone(),
        span,
    )
}

/// True if `term` has any free SymbolId that names an entry param.
/// Uses `closure_convert::collect_free_vars` with empty
/// `top_level`/`known_defs` sets (same style as
/// `compute_required_params`).
fn rhs_references_entry_param(
    term: &Term,
    entry_params: &std::collections::HashSet<SymbolId>,
    symbols: &SymbolTable,
) -> bool {
    use std::collections::HashSet;
    let bound: HashSet<SymbolId> = HashSet::new();
    let empty_top: HashSet<SymbolId> = HashSet::new();
    let empty_defs: HashSet<String> = HashSet::new();
    let mut free: Vec<Term> = Vec::new();
    let mut seen: HashSet<SymbolId> = HashSet::new();
    super::closure_convert::collect_free_vars(
        term,
        &bound,
        &empty_top,
        &empty_defs,
        symbols,
        &mut free,
        &mut seen,
    );
    free.iter().any(
        |t| matches!(&t.kind, TermKind::Var(crate::tlc::VarRef::Symbol(s)) if entry_params.contains(s)),
    )
}

/// Panic if any free variable of `term` has a type that carries an
/// unresolved Size type variable. See the TODO at `maybe_hoist` —
/// these vars aren't in scope inside the generated pre-pass, and
/// silently emitting the lift produces a broken shader.
fn assert_hoist_free_vars_are_grounded(
    term: &Term,
    entry_params: &std::collections::HashSet<SymbolId>,
    symbols: &SymbolTable,
) {
    use std::collections::HashSet;
    let bound: HashSet<SymbolId> = HashSet::new();
    let empty_top: HashSet<SymbolId> = HashSet::new();
    let empty_defs: HashSet<String> = HashSet::new();
    let mut free: Vec<Term> = Vec::new();
    let mut seen: HashSet<SymbolId> = HashSet::new();
    super::closure_convert::collect_free_vars(
        term,
        &bound,
        &empty_top,
        &empty_defs,
        symbols,
        &mut free,
        &mut seen,
    );
    for t in &free {
        if let TermKind::Var(crate::tlc::VarRef::Symbol(sym)) = &t.kind {
            if entry_params.contains(sym) {
                continue;
            }
            if type_contains_type_variable(&t.ty) {
                let name = symbols.get(*sym).map(|s| s.as_str()).unwrap_or("<unknown>");
                panic!(
                    "parallelize::maybe_hoist: hoisted SOAC references free var `{}` \
                     whose type `{:?}` contains an unresolved Size type variable. \
                     The generated pre-pass would reference an undeclared @size \
                     global and fail backend validation. Fix the lift site before \
                     enabling this path.",
                    name, t.ty
                );
            }
        }
    }
}

/// True if `ty` transitively contains a `Type::Variable(_)` — the
/// wyn representation of an unresolved Size (or other) type variable.
fn type_contains_type_variable(ty: &Type<TypeName>) -> bool {
    match ty {
        Type::Variable(_) => true,
        Type::Constructed(_, args) => args.iter().any(type_contains_type_variable),
    }
}

/// Build a compute entry Def whose body is the bare `soac_term` at the
/// tail. No input params, no output storage bindings declared —
/// `run()` keeps a `prepass_result_bindings` map telling Stage B's
/// `make_two_phase_plan` which result binding to use for this entry,
/// so phase 2 writes to the binding the fragment reads from.
///
/// The tail-SOAC shape is important: `analyze_entry` recognizes it as
/// a parallelizable entry and feeds it to Stage B for multi-staging.
fn build_prepass_def(
    entry_name: &str,
    soac_term: Term,
    elem_ty: Type<TypeName>,
    program: &mut Program,
) -> Def {
    make_entry_def(entry_name, soac_term, elem_ty, &[], Vec::new(), program)
}

/// Parallelize SOACs in compute entry points.
///
/// `disable` short-circuits the whole pass — every compute entry runs
/// as a single sequential loop, graphical entries receive no pre-pass
/// lifting, and the pipeline descriptor is built from the untouched
/// program. Useful for debugging (keeps the SSA close to the source)
/// and for backends that can't handle multi-entry pipelines.
pub fn run(mut program: Program, disable: bool) -> ParallelizationResult {
    if disable {
        let pipeline = build_default_pipeline(&program);
        return ParallelizationResult { program, pipeline };
    }

    // Track max binding across every `(set, binding)` the program already
    // uses — including implicit `ArrayExpr::StorageBuffer` bindings
    // introduced by buffer_specialize/mono for SOAC inputs, not just
    // user-declared `#[storage]` entries. Filtering only by
    // `program.storage` would collide fresh intermediates with input
    // buffers; see PLAN_scan_stage_b.md.
    let mut next_binding: u32 =
        collect_all_used_bindings(&program).iter().map(|(_, b)| b + 1).max().unwrap_or(0);

    // Hoist invariant SOACs out of graphical entry bodies into generated
    // compute pre-pass entries. Each pre-pass writes its SOAC result to
    // a fresh storage buffer; the graphical entry's body is rewritten to
    // read from that buffer. The pre-pass entries are added to
    // `program.defs` so the compute Stage A/B analysis below picks them
    // up and multi-stages them.
    //
    // `prepass_result_bindings` maps each hoisted pre-pass's def symbol
    // to the storage binding the graphical entry reads from; Stage B's
    // `make_two_phase_plan` consults this so phase 2's result store goes
    // exactly there (instead of a freshly-allocated binding).
    let mut prepass_result_bindings: HashMap<SymbolId, (u32, u32)> = HashMap::new();
    lift_graphical_invariant_soacs(&mut program, &mut next_binding, &mut prepass_result_bindings);

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

    for (_sym, analysis) in &analyses {
        let entry_name = program.symbols.get(analysis.def_name).cloned().unwrap_or_default();
        let forced = prepass_result_bindings.get(&analysis.def_name).copied();
        let plan = make_lowering_plan(analysis, &entry_name, next_binding, forced, &mut program);
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
                    bindings: collect_program_resource_bindings(&program),
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
    forced_result_binding: Option<(u32, u32)>,
    program: &mut Program,
) -> LoweringPlan {
    match &analysis.soac.original {
        SoacOp::Map { .. } => make_map_plan(analysis, entry_name),
        SoacOp::Reduce { op, ne, .. } => make_two_phase_plan(
            analysis,
            entry_name,
            op,
            ne,
            next_binding,
            forced_result_binding,
            program,
        ),
        SoacOp::Redomap { reduce_op, ne, .. } => make_two_phase_plan(
            analysis,
            entry_name,
            reduce_op,
            ne,
            next_binding,
            forced_result_binding,
            program,
        ),
        SoacOp::Scan { op, ne, .. } => make_scan_plan(analysis, entry_name, op, ne, next_binding, program),
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
    reduce_op: &super::SoacBody,
    ne: &Term,
    next_binding: u32,
    forced_result_binding: Option<(u32, u32)>,
    program: &mut Program,
) -> LoweringPlan {
    // Partials always consumes one fresh binding. When the caller has
    // pre-allocated a result binding (graphical-entry lift), use it; the
    // lift step also added the Input-role decl on the graphical entry,
    // so this keeps the two sides in sync. Without a forced binding the
    // plan allocates its own.
    let (partials_binding, result_binding, extra_used) = match forced_result_binding {
        Some(result) => ((AUTO_STORAGE_SET, next_binding), result, 1),
        None => (
            (AUTO_STORAGE_SET, next_binding),
            (AUTO_STORAGE_SET, next_binding + 1),
            2,
        ),
    };
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
        extra_bindings_used: extra_used,
    }
}

fn make_scan_plan(
    analysis: &EntryAnalysis,
    entry_name: &str,
    op: &super::SoacBody,
    ne: &Term,
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
    reduce_op: &super::SoacBody,
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
    // (Input role), writes `partials` at `tid` (Intermediate). Input
    // bindings must be declared explicitly — the backend's storage-buffer
    // validation lists the entry's `storage_bindings` as the allowlist
    // for any `storage(set, binding)` reference in the body. Previously
    // this worked only because `partials_binding` accidentally aliased
    // the input buffer at (0, 0) under the too-shallow allocator.
    let mut phase1_bindings = input_storage_decls(&analysis.soac);
    phase1_bindings.push(interface::StorageBindingDecl {
        set: partials_binding.0,
        binding: partials_binding.1,
        role: interface::StorageRole::Intermediate,
        elem_ty: elem_type.clone(),
    });
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
    let phase2_store = intrinsic_term_by_id(
        crate::builtins::catalog().known().storage_store,
        vec![
            uint_lit(result_binding.0 as u64, span),
            uint_lit(result_binding.1 as u64, span),
            uint_lit(0, span),
            var_term(r_sym, elem_type.clone(), span),
        ],
        unit_ty.clone(),
        span,
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

    let mut all_bindings = collect_soac_bindings(&analysis.soac);
    let input_indices: Vec<usize> = (0..all_bindings.len()).collect();
    let partials_idx = push_storage_binding(
        &mut all_bindings,
        partials_binding,
        Access::ReadWrite,
        BufferUsage::Intermediate,
        format!("{}_partials", entry_name),
    );
    let result_idx = push_storage_binding(
        &mut all_bindings,
        result_binding,
        Access::WriteOnly,
        BufferUsage::Output,
        format!("{}_result", entry_name),
    );

    let pipeline = Pipeline::MultiCompute(MultiComputePipeline {
        bindings: all_bindings,
        stages: vec![
            derived_stage(phase1_name.clone(), input_indices, vec![partials_idx]),
            fixed_stage(phase2_name.clone(), vec![partials_idx], vec![result_idx]),
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
    op: &super::SoacBody,
    ne: &Term,
    elem_type: &Type<TypeName>,
    output_binding: (u32, u32),
    block_sums_binding: (u32, u32),
    block_offsets_binding: (u32, u32),
    program: &mut Program,
) -> (Vec<Def>, Pipeline) {
    let span = ne.span;

    // Resolve the input BufferRef from the SOAC's storage provenance.
    let input_buf = match analysis.soac.provenances.first() {
        Some(ArrayProvenance::Storage {
            set,
            binding,
            elem_ty,
        }) => BufferRef {
            set: *set,
            binding: *binding,
            elem_ty: elem_ty.clone(),
        },
        _ => panic!("BUG: build_scan_entries called with non-storage SOAC input"),
    };

    let plan = ScanPlan {
        combiner: op.clone(),
        neutral: ne.clone(),
        elem_ty: elem_type.clone(),
        input: input_buf,
        output: BufferRef::from_tuple(output_binding, elem_type.clone()),
        block_sums: BufferRef::from_tuple(block_sums_binding, elem_type.clone()),
        block_offsets: BufferRef::from_tuple(block_offsets_binding, elem_type.clone()),
        prefix_lets: analysis.prefix_lets.clone(),
        required_params: analysis.required_params.clone(),
        span,
    };

    let phase1_name = format!("{}_phase1_local_scans", entry_name);
    let phase2_name = format!("{}_phase2_scan_sums", entry_name);
    let phase3_name = format!("{}_phase3_add_offsets", entry_name);

    let (phase1_def, _phase1_bindings) =
        build_scan_phase_def(&plan, ScanPhase::LocalScan, &phase1_name, program);
    let (phase2_def, _phase2_bindings) =
        build_scan_phase_def(&plan, ScanPhase::SumsPrefixScan, &phase2_name, program);
    let (phase3_def, _phase3_bindings) =
        build_scan_phase_def(&plan, ScanPhase::ApplyBlockOffsets, &phase3_name, program);

    // Pipeline descriptor — indexes into a shared bindings Vec the host
    // uses to set up the WebGPU pipeline layout.
    let mut all_bindings = collect_soac_bindings(&analysis.soac);
    let input_indices: Vec<usize> = (0..all_bindings.len()).collect();
    let output_idx = push_storage_binding(
        &mut all_bindings,
        output_binding,
        Access::ReadWrite,
        BufferUsage::Output,
        format!("{}_output", entry_name),
    );
    let block_sums_idx = push_storage_binding(
        &mut all_bindings,
        block_sums_binding,
        Access::ReadWrite,
        BufferUsage::Intermediate,
        format!("{}_block_sums", entry_name),
    );
    let block_offsets_idx = push_storage_binding(
        &mut all_bindings,
        block_offsets_binding,
        Access::ReadWrite,
        BufferUsage::Intermediate,
        format!("{}_block_offsets", entry_name),
    );

    let pipeline = Pipeline::MultiCompute(MultiComputePipeline {
        bindings: all_bindings,
        stages: vec![
            derived_stage(phase1_name, input_indices, vec![output_idx, block_sums_idx]),
            fixed_stage(phase2_name, vec![block_sums_idx], vec![block_offsets_idx]),
            derived_stage(phase3_name, vec![block_offsets_idx], vec![output_idx]),
        ],
    });

    (vec![phase1_def, phase2_def, phase3_def], pipeline)
}

// =============================================================================
// Shared per-thread chunk-arithmetic scaffolding
// =============================================================================

/// The fresh symbol set for a parallel entry's per-thread chunk arithmetic.
/// One phase-entry body has a single `ChunkArithmetic` that `wrap()` turns
/// into a let-chain outside the body:
///
/// ```text
/// let tid         = _w_intrinsic_thread_id() in
/// let total       = 64 in
/// let input_len   = <input_len_term> in
/// let chunk_size  = (input_len + total - 1) / total in
/// let chunk_start = tid * chunk_size in
/// let chunk_len   = if chunk_size < (input_len - chunk_start)
///                     then chunk_size else (input_len - chunk_start) in
/// <body>
/// ```
///
/// Reused by both `build_chunked_soac_body` (reduce/map/redomap) and
/// `build_scan_phase_def` (scan phases 1 and 3).
struct ChunkArithmetic {
    /// Type of `total`, `input_len`, `chunk_size`, `chunk_start`,
    /// `chunk_len`. `tid` is always u32 (from `_w_intrinsic_thread_id`)
    /// and is cast to `index_ty` at the `chunk_start = tid * chunk_size`
    /// site when the two types differ.
    index_ty: Type<TypeName>,
    tid_sym: SymbolId,
    total_sym: SymbolId,
    input_len_sym: SymbolId,
    chunk_size_sym: SymbolId,
    chunk_start_sym: SymbolId,
    chunk_len_sym: SymbolId,
}

impl ChunkArithmetic {
    fn alloc_for(index_ty: Type<TypeName>, program: &mut Program) -> Self {
        ChunkArithmetic {
            index_ty,
            tid_sym: program.symbols.alloc("_par_tid".into()),
            total_sym: program.symbols.alloc("_par_total".into()),
            input_len_sym: program.symbols.alloc("_par_input_len".into()),
            chunk_size_sym: program.symbols.alloc("_par_chunk_size".into()),
            chunk_start_sym: program.symbols.alloc("_par_chunk_start".into()),
            chunk_len_sym: program.symbols.alloc("_par_chunk_len".into()),
        }
    }

    /// The raw u32 thread-id. Use for storage-op indices (`partials[tid]`,
    /// `block_sums[tid]`, etc.) since `_w_intrinsic_storage_store`'s
    /// index arg is u32.
    fn tid_u32(&self, span: ast::Span) -> Term {
        var_term(self.tid_sym, u32_ty(), span)
    }
    fn chunk_start(&self, span: ast::Span) -> Term {
        var_term(self.chunk_start_sym, self.index_ty.clone(), span)
    }
    fn chunk_len(&self, span: ast::Span) -> Term {
        var_term(self.chunk_len_sym, self.index_ty.clone(), span)
    }

    /// Wrap `body` with:
    /// ```text
    /// let tid         = _w_intrinsic_thread_id()  (u32)
    /// let total       = TOTAL_THREADS             (index_ty)
    /// let input_len   = <input_len_term>          (index_ty)
    /// let chunk_size  = (input_len + total - 1) / total  (index_ty)
    /// let chunk_start = cast(tid) * chunk_size    (index_ty)
    /// let chunk_len   = min(chunk_size, input_len - chunk_start)  (index_ty)
    /// ```
    /// `cast(tid)` is the identity when `index_ty` is u32, otherwise an
    /// `i32.u32` bitcast. `input_len_term` must already be typed as
    /// `self.index_ty`; callers derive it from the SOAC input's length
    /// term.
    fn wrap(&self, body: Term, input_len_term: Term, span: ast::Span, program: &mut Program) -> Term {
        let ity = self.index_ty.clone();
        let bool_ty = Type::Constructed(TypeName::Bool, vec![]);
        let one_lit = int_lit_of(1, &ity, span);
        let total_lit = int_lit_of(TOTAL_THREADS as i64, &ity, span);

        let len_minus_start = binop(
            "-",
            var_term(self.input_len_sym, ity.clone(), span),
            var_term(self.chunk_start_sym, ity.clone(), span),
            ity.clone(),
            span,
        );
        let cond = binop(
            "<",
            var_term(self.chunk_size_sym, ity.clone(), span),
            len_minus_start.clone(),
            bool_ty,
            span,
        );
        let min_expr = Term {
            id: TermId(0),
            ty: ity.clone(),
            span,
            kind: TermKind::If {
                cond: Box::new(cond),
                then_branch: Box::new(var_term(self.chunk_size_sym, ity.clone(), span)),
                else_branch: Box::new(len_minus_start),
            },
        };
        let body = let_term(self.chunk_len_sym, ity.clone(), min_expr, body, span);

        let tid_as_idx = if ity == u32_ty() {
            var_term(self.tid_sym, u32_ty(), span)
        } else {
            intrinsic_term(
                "i32.u32",
                vec![var_term(self.tid_sym, u32_ty(), span)],
                ity.clone(),
                span,
                program,
            )
        };
        let chunk_start_rhs = binop(
            "*",
            tid_as_idx,
            var_term(self.chunk_size_sym, ity.clone(), span),
            ity.clone(),
            span,
        );
        let body = let_term(self.chunk_start_sym, ity.clone(), chunk_start_rhs, body, span);

        let total_minus_1 = binop(
            "-",
            var_term(self.total_sym, ity.clone(), span),
            one_lit,
            ity.clone(),
            span,
        );
        let len_plus = binop(
            "+",
            var_term(self.input_len_sym, ity.clone(), span),
            total_minus_1,
            ity.clone(),
            span,
        );
        let chunk_size_rhs = binop(
            "/",
            len_plus,
            var_term(self.total_sym, ity.clone(), span),
            ity.clone(),
            span,
        );
        let body = let_term(self.chunk_size_sym, ity.clone(), chunk_size_rhs, body, span);

        let body = let_term(self.input_len_sym, ity.clone(), input_len_term, body, span);
        let body = let_term(self.total_sym, ity, total_lit, body, span);

        let tid_rhs = intrinsic_term_by_id(
            crate::builtins::catalog().known().thread_id,
            vec![],
            u32_ty(),
            span,
        );
        let_term(self.tid_sym, u32_ty(), tid_rhs, body, span)
    }
}

fn u32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::UInt(32), vec![])
}

/// Index type for `ChunkArithmetic` to use when chunking this SOAC.
/// Range inputs carry their own index type (usually i32 from `0..<N`
/// syntax); every other input kind (storage views, zips, generators,
/// literals) is indexed by u32.
fn soac_input_index_ty(soac: &SoacOp) -> Type<TypeName> {
    let first = match soac {
        SoacOp::Map { inputs, .. } | SoacOp::Redomap { inputs, .. } => inputs.first(),
        SoacOp::Reduce { input, .. } | SoacOp::Scan { input, .. } => Some(input),
        _ => None,
    };
    match first {
        Some(ArrayExpr::Range { start, .. }) => start.ty.clone(),
        _ => u32_ty(),
    }
}

/// Integer literal term typed as `ty`. `ty` is expected to be one of
/// `Int(32)` / `UInt(32)`; other widths/types aren't used by
/// `ChunkArithmetic` today.
fn int_lit_of(value: i64, ty: &Type<TypeName>, span: ast::Span) -> Term {
    Term {
        id: TermId(0),
        ty: ty.clone(),
        span,
        kind: TermKind::IntLit(value.to_string()),
    }
}

// =============================================================================
// ScanPlan — first-class data representation of a parallelized scan
// =============================================================================

/// Reference to a buffer used by a scan phase. `set`/`binding` identify it
/// inside shader entry metadata; `elem_ty` drives `StorageBindingDecl`.
#[derive(Debug, Clone)]
struct BufferRef {
    set: u32,
    binding: u32,
    elem_ty: Type<TypeName>,
}

impl BufferRef {
    fn from_tuple(t: (u32, u32), elem_ty: Type<TypeName>) -> Self {
        BufferRef {
            set: t.0,
            binding: t.1,
            elem_ty,
        }
    }

    fn decl(&self, role: interface::StorageRole) -> interface::StorageBindingDecl {
        interface::StorageBindingDecl {
            set: self.set,
            binding: self.binding,
            role,
            elem_ty: self.elem_ty.clone(),
        }
    }
}

/// Execution plan for a parallelized scan: every buffer, combiner, and
/// the input-length source are fields. Per-phase builders consume it,
/// each emitting a hand-rolled TLC body that writes to known buffers.
/// No reliance on the from_tlc Map/Scan → OutputView auto-rewrite
/// (which can't target a specific binding).
struct ScanPlan {
    combiner: super::SoacBody,
    neutral: Term,
    elem_ty: Type<TypeName>,
    input: BufferRef,
    output: BufferRef,
    block_sums: BufferRef,
    block_offsets: BufferRef,
    /// Let-bindings from the entry prefix that must wrap every phase body.
    prefix_lets: Vec<(SymbolId, Type<TypeName>, Term)>,
    /// Entry-level lambda params the phase bodies reference.
    required_params: Vec<(SymbolId, Type<TypeName>)>,
    span: ast::Span,
}

enum ScanPhase {
    /// Phase 1: each thread scans its chunk of the input into the
    /// corresponding slice of `output`, and writes its chunk total
    /// to `block_sums[tid]`.
    LocalScan,
    /// Phase 2: a single sequential scan of `block_sums` into
    /// `block_offsets` (replicated across threads; write is
    /// idempotent). Dispatched with workgroup (1,1,1).
    SumsPrefixScan,
    /// Phase 3: each thread reads its `block_offsets[tid]` and
    /// combines it with every already-written element in
    /// `output[chunk_range]`, writing the final result back in place.
    ApplyBlockOffsets,
}

/// Invoke a SOAC-op SoacBody on explicit argument terms. After
/// defunctionalize, the lambda's body is a `Var` naming the lifted
/// function; the call is emitted as `App(body, args_with_captures)`
/// following `from_tlc::convert_soac_*`'s convention that SOAC
/// combiners accept their original params followed by their captures.
fn invoke_soac_lambda(sb: &super::SoacBody, args: Vec<Term>, span: ast::Span) -> Term {
    assert_eq!(
        sb.lam.params.len(),
        args.len(),
        "BUG: parallelize invoking SOAC lambda: {} params vs {} args",
        sb.lam.params.len(),
        args.len()
    );
    let mut call_args = args;
    // Trailing captures — `convert_soac_map`/`..._scan` pass them after
    // the original-param args.
    for (sym, ty, _val) in &sb.captures {
        call_args.push(var_term(*sym, ty.clone(), span));
    }
    Term {
        id: TermId(0),
        ty: sb.lam.ret_ty.clone(),
        span,
        kind: TermKind::App {
            func: Box::new((*sb.lam.body).clone()),
            args: call_args,
        },
    }
}

fn emit_storage_load(buf: &BufferRef, index: Term, span: ast::Span, _program: &mut Program) -> Term {
    intrinsic_term_by_id(
        crate::builtins::catalog().known().storage_index,
        vec![
            uint_lit(buf.set as u64, span),
            uint_lit(buf.binding as u64, span),
            index,
        ],
        buf.elem_ty.clone(),
        span,
    )
}

fn emit_storage_store(
    buf: &BufferRef,
    index: Term,
    value: Term,
    span: ast::Span,
    _program: &mut Program,
) -> Term {
    let unit_ty = Type::Constructed(TypeName::Unit, vec![]);
    intrinsic_term_by_id(
        crate::builtins::catalog().known().storage_store,
        vec![
            uint_lit(buf.set as u64, span),
            uint_lit(buf.binding as u64, span),
            index,
            value,
        ],
        unit_ty,
        span,
    )
}

fn emit_storage_len(buf: &BufferRef, span: ast::Span, _program: &mut Program) -> Term {
    intrinsic_term_by_id(
        crate::builtins::catalog().known().storage_len,
        vec![uint_lit(buf.set as u64, span), uint_lit(buf.binding as u64, span)],
        u32_ty(),
        span,
    )
}

/// Sequence two effects: `let _dummy = first in second`. `second`'s type
/// is the result type of the whole expression; `first` must be
/// Unit-typed (the body doesn't reference the let-bound var).
fn seq_unit_effect(first: Term, second: Term, span: ast::Span, program: &mut Program) -> Term {
    let unit_ty = Type::Constructed(TypeName::Unit, vec![]);
    let dummy = program.symbols.alloc("_par_seq".into());
    let_term(dummy, unit_ty, first, second, span)
}

/// The load/combine/store/yield micro-pattern shared by every scan-phase
/// body. Emits:
///
/// ```text
/// let v   = load(src, index) in
/// let new = combiner(<combine_args(v)>) in
/// let _   = store(dst, index, new) in
/// new
/// ```
///
/// `combine_args(v_term)` builds the combiner's arg list — callers
/// choose the order (`(acc, v)` for phase 1/2; `(v, off)` for phase 3).
/// Returns a Term of type `elem_ty` usable as the loop body's new-acc.
fn emit_load_combine_store(
    src: &BufferRef,
    dst: &BufferRef,
    index: Term,
    combine_args: impl FnOnce(/*v_var:*/ Term) -> Vec<Term>,
    combiner: &super::SoacBody,
    elem_ty: &Type<TypeName>,
    span: ast::Span,
    program: &mut Program,
) -> Term {
    let v_sym = program.symbols.alloc("_lcs_v".into());
    let new_sym = program.symbols.alloc("_lcs_new".into());

    let v_load = emit_storage_load(src, index.clone(), span, program);
    let v_var = var_term(v_sym, elem_ty.clone(), span);
    let combiner_app = invoke_soac_lambda(combiner, combine_args(v_var), span);
    let store = emit_storage_store(
        dst,
        index,
        var_term(new_sym, elem_ty.clone(), span),
        span,
        program,
    );
    let tail = seq_unit_effect(store, var_term(new_sym, elem_ty.clone(), span), span, program);
    let new_binding = let_term(new_sym, elem_ty.clone(), combiner_app, tail, span);
    let_term(v_sym, elem_ty.clone(), v_load, new_binding, span)
}

/// Build a scan phase's Def. Dispatches on `ScanPhase` and returns
/// `(Def, storage_bindings)` for `build_scan_entries` to collect.
fn build_scan_phase_def(
    plan: &ScanPlan,
    phase: ScanPhase,
    entry_name: &str,
    program: &mut Program,
) -> (Def, Vec<interface::StorageBindingDecl>) {
    let span = plan.span;
    let (body, bindings) = match phase {
        ScanPhase::LocalScan => build_scan_local_body(plan, span, program),
        ScanPhase::SumsPrefixScan => build_scan_sums_body(plan, span, program),
        ScanPhase::ApplyBlockOffsets => build_scan_apply_body(plan, span, program),
    };
    // Wrap with the entry's prefix_lets (outer-scope bindings the phase
    // body may reference).
    let body = plan.prefix_lets.iter().rev().fold(body, |acc, (name, ty, rhs)| {
        let_term(*name, ty.clone(), rhs.clone(), acc, span)
    });
    let unit_ty = Type::Constructed(TypeName::Unit, vec![]);
    let def = make_entry_def(
        entry_name,
        body,
        unit_ty,
        &plan.required_params,
        bindings.clone(),
        program,
    );
    (def, bindings)
}

/// Phase 1 body: per-thread chunk scan writing to `output[chunk_range]`
/// plus `block_sums[tid] = final_acc`. Uses a `while` loop so the u32
/// counter type matches `chunk_len` (ForRange hardcodes i32 counters,
/// which would clash with ChunkArithmetic's u32 chunk-size arithmetic).
fn build_scan_local_body(
    plan: &ScanPlan,
    span: ast::Span,
    program: &mut Program,
) -> (Term, Vec<interface::StorageBindingDecl>) {
    let chunk = ChunkArithmetic::alloc_for(u32_ty(), program);

    let final_acc = emit_u32_counter_scan_loop(
        &plan.elem_ty,
        plan.neutral.clone(),
        chunk.chunk_len(span),
        |acc_var, i_var, program| {
            let abs_idx = binop("+", chunk.chunk_start(span), i_var, u32_ty(), span);
            emit_load_combine_store(
                &plan.input,
                &plan.output,
                abs_idx,
                |v| vec![acc_var, v],
                &plan.combiner,
                &plan.elem_ty,
                span,
                program,
            )
        },
        span,
        program,
    );

    // After the loop: store block_sums[tid] = final_acc.
    let final_acc_sym = program.symbols.alloc("_scan_final_acc".into());
    let store_block_sum = emit_storage_store(
        &plan.block_sums,
        chunk.tid_u32(span),
        var_term(final_acc_sym, plan.elem_ty.clone(), span),
        span,
        program,
    );
    let phase_body = let_term(
        final_acc_sym,
        plan.elem_ty.clone(),
        final_acc,
        store_block_sum,
        span,
    );

    // Wrap with chunk arithmetic scaffolding.
    let input_len_term = emit_storage_len(&plan.input, span, program);
    let body = chunk.wrap(phase_body, input_len_term, span, program);

    let bindings = vec![
        plan.input.decl(interface::StorageRole::Input),
        plan.output.decl(interface::StorageRole::Intermediate),
        plan.block_sums.decl(interface::StorageRole::Intermediate),
    ];
    (body, bindings)
}

/// Emit a `while i < bound` loop with state `(acc: acc_ty, i: u32)`.
/// `mk_body(acc_var, i_var, program)` returns the new `acc` value for
/// the next iteration. The returned Term is `_w_tuple_proj(loop, 0)` —
/// the final accumulator (projected out of the state tuple). Phase 2
/// and phase 3 discard it; phase 1 uses it to write `block_sums[tid]`.
fn emit_u32_counter_scan_loop(
    acc_ty: &Type<TypeName>,
    init_acc: Term,
    bound: Term,
    mk_body: impl FnOnce(/*acc_var:*/ Term, /*i_var:*/ Term, &mut Program) -> Term,
    span: ast::Span,
    program: &mut Program,
) -> Term {
    let u32_ty_v = u32_ty();
    let state_ty = Type::Constructed(TypeName::Tuple(2), vec![acc_ty.clone(), u32_ty_v.clone()]);
    let state_sym = program.symbols.alloc("_loop_state".into());
    let acc_sym = program.symbols.alloc("_loop_acc".into());
    let i_sym = program.symbols.alloc("_loop_i".into());

    let init = tuple_term(vec![init_acc, uint_lit(0, span)], state_ty.clone(), span, program);

    let acc_proj = tuple_proj(
        var_term(state_sym, state_ty.clone(), span),
        0,
        acc_ty.clone(),
        span,
        program,
    );
    let i_proj = tuple_proj(
        var_term(state_sym, state_ty.clone(), span),
        1,
        u32_ty_v.clone(),
        span,
        program,
    );

    let cond = binop(
        "<",
        var_term(i_sym, u32_ty_v.clone(), span),
        bound,
        Type::Constructed(TypeName::Bool, vec![]),
        span,
    );

    let new_acc = mk_body(
        var_term(acc_sym, acc_ty.clone(), span),
        var_term(i_sym, u32_ty_v.clone(), span),
        program,
    );
    let new_acc_sym = program.symbols.alloc("_loop_new_acc".into());
    let next_i = binop(
        "+",
        var_term(i_sym, u32_ty_v.clone(), span),
        uint_lit(1, span),
        u32_ty_v.clone(),
        span,
    );
    let new_state_tuple = tuple_term(
        vec![var_term(new_acc_sym, acc_ty.clone(), span), next_i],
        state_ty.clone(),
        span,
        program,
    );
    let body = let_term(new_acc_sym, acc_ty.clone(), new_acc, new_state_tuple, span);

    let the_loop = Term {
        id: TermId(0),
        ty: state_ty.clone(),
        span,
        kind: TermKind::Loop {
            loop_var: state_sym,
            loop_var_ty: state_ty.clone(),
            init: Box::new(init),
            init_bindings: vec![(acc_sym, acc_ty.clone(), acc_proj), (i_sym, u32_ty_v, i_proj)],
            kind: LoopKind::While { cond: Box::new(cond) },
            body: Box::new(body),
        },
    };

    // The loop's value is the state tuple; project .0 for the final acc.
    tuple_proj(the_loop, 0, acc_ty.clone(), span, program)
}

/// Phase 2 body: sequential scan over `block_sums` into `block_offsets`.
/// Every thread runs the same computation; writes are idempotent.
/// Dispatched with workgroup (1, 1, 1) so effectively a single thread.
fn build_scan_sums_body(
    plan: &ScanPlan,
    span: ast::Span,
    program: &mut Program,
) -> (Term, Vec<interface::StorageBindingDecl>) {
    let num_blocks = TOTAL_THREADS as u64;

    let final_acc = emit_u32_counter_scan_loop(
        &plan.elem_ty,
        plan.neutral.clone(),
        uint_lit(num_blocks, span),
        |acc_var, i_var, program| {
            emit_load_combine_store(
                &plan.block_sums,
                &plan.block_offsets,
                i_var,
                |v| vec![acc_var, v],
                &plan.combiner,
                &plan.elem_ty,
                span,
                program,
            )
        },
        span,
        program,
    );

    // The loop body already wrote every block_offsets[i]. Use a
    // harmless duplicate write of `final_acc` to block_offsets[N-1]
    // (the loop's last iteration already put the same value there) as
    // the Unit-typed tail of the phase body.
    let final_acc_sym = program.symbols.alloc("_sums_final_acc".into());
    let redundant_store = emit_storage_store(
        &plan.block_offsets,
        uint_lit(num_blocks - 1, span),
        var_term(final_acc_sym, plan.elem_ty.clone(), span),
        span,
        program,
    );
    let body = let_term(
        final_acc_sym,
        plan.elem_ty.clone(),
        final_acc,
        redundant_store,
        span,
    );

    let bindings = vec![
        plan.block_sums.decl(interface::StorageRole::Intermediate),
        plan.block_offsets.decl(interface::StorageRole::Intermediate),
    ];
    (body, bindings)
}

/// Phase 3 body: read `off = block_offsets[tid]`, then for each i in the
/// chunk range apply `output[chunk_start + i] = combiner(output[chunk_start + i], off)`.
fn build_scan_apply_body(
    plan: &ScanPlan,
    span: ast::Span,
    program: &mut Program,
) -> (Term, Vec<interface::StorageBindingDecl>) {
    let chunk = ChunkArithmetic::alloc_for(u32_ty(), program);
    let off_sym = program.symbols.alloc("_apply_off".into());

    let final_acc = emit_u32_counter_scan_loop(
        &plan.elem_ty,
        plan.neutral.clone(),
        chunk.chunk_len(span),
        |_acc_var /* unused — phase 3 carries no cross-iter state */, i_var, program| {
            let abs_idx = binop("+", chunk.chunk_start(span), i_var, u32_ty(), span);
            let off_var = var_term(off_sym, plan.elem_ty.clone(), span);
            // Phase 3 combines (prior, off), not (acc, v) — the `v` from
            // the load is this iteration's prior output element.
            emit_load_combine_store(
                &plan.output,
                &plan.output,
                abs_idx,
                |v| vec![v, off_var],
                &plan.combiner,
                &plan.elem_ty,
                span,
                program,
            )
        },
        span,
        program,
    );

    // Loop result is discarded; wrap in a Unit-typed tail — idempotent
    // write of `off` back to block_offsets[tid].
    let loop_result_sym = program.symbols.alloc("_apply_loop_res".into());
    let terminal_store = emit_storage_store(
        &plan.block_offsets,
        chunk.tid_u32(span),
        var_term(off_sym, plan.elem_ty.clone(), span),
        span,
        program,
    );
    let phase_body_after_loop = let_term(
        loop_result_sym,
        plan.elem_ty.clone(),
        final_acc,
        terminal_store,
        span,
    );

    // Prepend: let off = block_offsets[tid]
    let off_load = emit_storage_load(&plan.block_offsets, chunk.tid_u32(span), span, program);
    let phase_body = let_term(
        off_sym,
        plan.elem_ty.clone(),
        off_load,
        phase_body_after_loop,
        span,
    );

    // Wrap with chunk arithmetic. Use output's length so phase 3 doesn't
    // need to admit the input binding.
    let input_len_term = emit_storage_len(&plan.output, span, program);
    let body = chunk.wrap(phase_body, input_len_term, span, program);

    let bindings = vec![
        plan.output.decl(interface::StorageRole::Output),
        plan.block_offsets.decl(interface::StorageRole::Intermediate),
    ];
    (body, bindings)
}

// =============================================================================
// Chunked SOAC body builder
// =============================================================================

/// Build a chunked SOAC body for reduce/redomap/map phase 1. The SOAC's
/// input is rebased to `(chunk_start, chunk_len)` (per-thread range), and
/// the result is optionally stored into `partials[tid]`.
///
/// Scan is deliberately not handled here; `build_scan_phase_def` emits
/// scan's phase bodies as hand-rolled TLC loops because scan's phases
/// don't fit the "one SOAC = one entry" shape reduce/map do.
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
    // ChunkArithmetic's `index_ty` matches the SOAC input's index type:
    // storage-view inputs use u32 (from `_w_intrinsic_storage_len`);
    // range inputs use the range's declared `start.ty` (typically i32).
    // This keeps `chunk_array_expr`'s rebuilt Range consistent with the
    // original source-level types, avoiding a bitcast boundary.
    let index_ty = soac_input_index_ty(&soac.original);
    let chunk = ChunkArithmetic::alloc_for(index_ty, program);
    let chunk_start_var = chunk.chunk_start(span);
    let chunk_len_var = chunk.chunk_len(span);

    let chunked_soac = match &soac.original {
        SoacOp::Map {
            lam,
            inputs,
            consumes_input,
        } => {
            let chunked_inputs = inputs
                .iter()
                .map(|input| chunk_array_expr(input, &chunk_start_var, &chunk_len_var))
                .collect();
            SoacOp::Map {
                lam: lam.clone(),
                inputs: chunked_inputs,
                consumes_input: *consumes_input,
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
        SoacOp::Scan { .. } => {
            unreachable!("Scan is lowered via build_scan_phase_def, not build_chunked_soac_body")
        }
        _ => unreachable!("analyze_soac rejected non-parallelizable variants"),
    };

    // Get the input length term from the first input's provenance.
    let input_len_term = get_input_len(soac, span);

    let mut body = soac_term(chunked_soac, result_ty.clone(), span);

    // If requested, wrap the SOAC with `let r = <soac> in store(set, binding, tid, r)`
    // so each thread's partial result lands in its own slot.
    if let Some((set, binding)) = write_partial_to {
        let unit_ty = Type::Constructed(TypeName::Unit, vec![]);
        let r_sym = program.symbols.alloc("_par_out".into());
        let tid_var = chunk.tid_u32(span);
        let r_var = var_term(r_sym, result_ty.clone(), span);
        let store = intrinsic_term_by_id(
            crate::builtins::catalog().known().storage_store,
            vec![
                uint_lit(set as u64, span),
                uint_lit(binding as u64, span),
                tid_var,
                r_var,
            ],
            unit_ty,
            span,
        );
        body = let_term(r_sym, result_ty.clone(), body, store, span);
    }

    // Wrap with prefix lets (from the original entry body).
    for (name, ty, rhs) in prefix_lets.iter().rev() {
        body = let_term(*name, ty.clone(), rhs.clone(), body, span);
    }

    chunk.wrap(body, input_len_term, span, program)
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
            // Range: chunk_start..chunk_len starting from original start.
            // Chunk values are u32 (from ChunkArithmetic); the original
            // range is typically i32. Coercion to match the range's type
            // happens at the caller — `chunk_array_expr` stays a simple
            // structural rewrite.
            let range_ty = start.ty.clone();
            let new_start = binop(
                "+",
                (**start).clone(),
                chunk_start.clone(),
                range_ty.clone(),
                start.span,
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
        kind: TermKind::Var(crate::tlc::VarRef::Symbol(sym)),
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

/// Build an `App(Var(Builtin { id, overload_idx: 0 }), args)` term for
/// a catalog intrinsic.
fn intrinsic_term_by_id(
    id: crate::builtins::BuiltinId,
    args: Vec<Term>,
    ret_ty: Type<TypeName>,
    span: ast::Span,
) -> Term {
    let func_term = Term {
        id: TermId(0),
        ty: Type::Variable(0),
        span,
        kind: TermKind::Var(crate::tlc::VarRef::Builtin { id, overload_idx: 0 }),
    };
    Term {
        id: TermId(0),
        ty: ret_ty,
        span,
        kind: TermKind::App {
            func: Box::new(func_term),
            args,
        },
    }
}

fn intrinsic_term(
    name: &str,
    args: Vec<Term>,
    ret_ty: Type<TypeName>,
    span: ast::Span,
    program: &mut Program,
) -> Term {
    // Catalog-resolved builtins emit `Var(Builtin(id))` so downstream
    // passes dispatch structurally. The assert mirrors `tlc::build_call`
    // and `buffer_specialize::make_app`: synthesised IR sites target
    // single-overload entries.
    let func_var = if let Some(def) = crate::builtins::catalog().lookup_by_any_name(name) {
        assert_eq!(
            def.overloads().len(),
            1,
            "parallelize::intrinsic_term({:?}) targets a multi-overload catalog entry; \
             caller must specify overload_idx explicitly",
            name
        );
        crate::tlc::VarRef::Builtin {
            id: def.id,
            overload_idx: 0,
        }
    } else {
        let sym = if let Some(&existing) = program.def_syms.get(name) {
            existing
        } else {
            let sym = program.symbols.alloc(name.to_string());
            program.def_syms.insert(name.to_string(), sym);
            sym
        };
        crate::tlc::VarRef::Symbol(sym)
    };
    let func_term = Term {
        id: TermId(0),
        ty: Type::Variable(0),
        span,
        kind: TermKind::Var(func_var),
    };
    Term {
        id: TermId(0),
        ty: ret_ty,
        span,
        kind: TermKind::App {
            func: Box::new(func_term),
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

/// Construct a `TermKind::Tuple` term with the given tuple type.
fn tuple_term(components: Vec<Term>, ty: Type<TypeName>, span: ast::Span, _program: &mut Program) -> Term {
    Term {
        id: TermId(0),
        ty,
        span,
        kind: TermKind::Tuple(components),
    }
}

/// Project `base.index` via `TermKind::TupleProj`.
fn tuple_proj(
    base: Term,
    index: u32,
    elem_ty: Type<TypeName>,
    span: ast::Span,
    _program: &mut Program,
) -> Term {
    Term {
        id: TermId(0),
        ty: elem_ty,
        span,
        kind: TermKind::TupleProj {
            tuple: Box::new(base),
            idx: index as usize,
        },
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

// =============================================================================
// Pipeline-descriptor construction helpers
// =============================================================================

/// Append a storage-buffer binding to `bindings` and return its index.
fn push_storage_binding(
    bindings: &mut Vec<Binding>,
    (set, binding): (u32, u32),
    access: Access,
    usage: BufferUsage,
    name: String,
) -> usize {
    let idx = bindings.len();
    bindings.push(Binding::StorageBuffer {
        set,
        binding,
        access,
        usage,
        name,
    });
    idx
}

/// A `ComputeStage` with the default workgroup + DerivedFromInputLength
/// dispatch. Used by per-element phases (phase 1 of reduce; phases 1+3
/// of scan).
fn derived_stage(entry_point: String, reads: Vec<usize>, writes: Vec<usize>) -> ComputeStage {
    ComputeStage {
        entry_point,
        workgroup_size: LOCAL_SIZE,
        dispatch_size: DispatchSize::DerivedFromInputLength {
            workgroup_size: TOTAL_THREADS,
        },
        reads,
        writes,
    }
}

/// A `ComputeStage` with a `(1, 1, 1)` workgroup and fixed `1×1×1`
/// dispatch — used by single-threaded combine phases (phase 2 of
/// reduce; phase 2 of scan).
fn fixed_stage(entry_point: String, reads: Vec<usize>, writes: Vec<usize>) -> ComputeStage {
    ComputeStage {
        entry_point,
        workgroup_size: (1, 1, 1),
        dispatch_size: DispatchSize::Fixed { x: 1, y: 1, z: 1 },
        reads,
        writes,
    }
}

/// Iterate over the SOAC's storage-backed inputs, yielding
/// `(positional_index, set, binding, elem_ty)` for each. Provenances
/// that aren't `ArrayProvenance::Storage` (e.g. ranges) are skipped.
/// Single source of truth for the two adapters below.
fn storage_inputs(soac: &SoacAnalysis) -> impl Iterator<Item = (usize, u32, u32, &Type<TypeName>)> + '_ {
    soac.provenances.iter().enumerate().filter_map(|(i, p)| match p {
        ArrayProvenance::Storage {
            set,
            binding,
            elem_ty,
        } => Some((i, *set, *binding, elem_ty)),
        _ => None,
    })
}

/// Pipeline-level `Binding` descriptors for the SOAC's storage inputs.
fn collect_soac_bindings(soac: &SoacAnalysis) -> Vec<Binding> {
    storage_inputs(soac)
        .map(|(i, set, binding, _)| Binding::StorageBuffer {
            set,
            binding,
            access: Access::ReadOnly,
            usage: BufferUsage::Input,
            name: format!("input_{}", i),
        })
        .collect()
}

/// Per-entry `StorageBindingDecl`s for the SOAC's storage inputs. Every
/// phase entry whose body reads those buffers must declare them so the
/// backend's binding allowlist admits the references.
fn input_storage_decls(soac: &SoacAnalysis) -> Vec<interface::StorageBindingDecl> {
    storage_inputs(soac)
        .map(|(_, set, binding, elem_ty)| interface::StorageBindingDecl {
            set,
            binding,
            role: interface::StorageRole::Input,
            elem_ty: elem_ty.clone(),
        })
        .collect()
}

/// Collect every `(set, binding)` pair already claimed anywhere in the
/// program so `parallelize::run` can hand out fresh intermediate bindings
/// that don't collide with anything. Includes user-declared resources
/// (`program.storage`, `program.uniforms`) *and* implicit bindings
/// attached by earlier passes as `ArrayExpr::StorageBuffer` inside SOAC
/// inputs. Consulting only `program.storage` would miss the implicit
/// ones — scan's three-way collision (partials, result, input) would
/// emit a broken shader.
fn collect_all_used_bindings(program: &Program) -> HashSet<(u32, u32)> {
    let mut used: HashSet<(u32, u32)> = HashSet::new();
    for u in &program.uniforms {
        used.insert((u.set, u.binding));
    }
    for s in &program.storage {
        used.insert((s.set, s.binding));
    }
    for def in &program.defs {
        collect_bindings_in_term(&def.body, &mut used);
    }
    used
}

fn collect_bindings_in_term(term: &Term, used: &mut HashSet<(u32, u32)>) {
    // At a TermKind wrapping ArrayExpr/Soac, inspect the wrapped shape so
    // `StorageBuffer` bindings aren't skipped by `for_each_child` (which
    // only visits Term children and can't extract u32 binding fields).
    match &term.kind {
        TermKind::ArrayExpr(ae) => collect_bindings_in_ae(ae, used),
        TermKind::Soac(op) => collect_bindings_in_soac(op, used),
        _ => {}
    }
    term.for_each_child(&mut |c| collect_bindings_in_term(c, used));
}

fn collect_bindings_in_ae(ae: &ArrayExpr, used: &mut HashSet<(u32, u32)>) {
    if let ArrayExpr::StorageBuffer { set, binding, .. } = ae {
        used.insert((*set, *binding));
    }
    match ae {
        ArrayExpr::Zip(aes) => {
            for a in aes {
                collect_bindings_in_ae(a, used);
            }
        }
        ArrayExpr::Soac(op) => collect_bindings_in_soac(op, used),
        _ => {}
    }
}

fn collect_bindings_in_soac(op: &SoacOp, used: &mut HashSet<(u32, u32)>) {
    match op {
        SoacOp::Map { inputs, .. } | SoacOp::Redomap { inputs, .. } => {
            for ae in inputs {
                collect_bindings_in_ae(ae, used);
            }
        }
        SoacOp::Reduce { input, .. } | SoacOp::Scan { input, .. } | SoacOp::Filter { input, .. } => {
            collect_bindings_in_ae(input, used);
        }
        SoacOp::Scatter { indices, values, .. } | SoacOp::ReduceByIndex { indices, values, .. } => {
            collect_bindings_in_ae(indices, used);
            collect_bindings_in_ae(values, used);
        }
    }
}

/// Graphics pipelines and non-parallelized compute entries need to declare
/// every resource the shader references so the host can build a matching
/// pipeline layout. Without this, viz/wgpu rejects the pipeline with
/// "Binding is missing from the pipeline layout".
fn collect_program_resource_bindings(program: &Program) -> Vec<Binding> {
    let mut bindings = Vec::new();
    for u in &program.uniforms {
        bindings.push(Binding::Uniform {
            set: u.set,
            binding: u.binding,
            name: u.name.clone(),
        });
    }
    for s in &program.storage {
        let access = match s.access {
            interface::StorageAccess::ReadOnly => Access::ReadOnly,
            interface::StorageAccess::WriteOnly => Access::WriteOnly,
            interface::StorageAccess::ReadWrite => Access::ReadWrite,
        };
        bindings.push(Binding::StorageBuffer {
            set: s.set,
            binding: s.binding,
            access,
            usage: BufferUsage::Input,
            name: s.name.clone(),
        });
    }
    bindings
}

fn build_default_pipeline(program: &Program) -> PipelineDescriptor {
    let mut pipelines = Vec::new();

    let resource_bindings = collect_program_resource_bindings(program);
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
                    bindings: resource_bindings.clone(),
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
                    bindings: resource_bindings.clone(),
                    vertex_inputs: vec![],
                    fragment_outputs: vec![],
                }));
            }
        }
    }

    PipelineDescriptor { pipelines }
}

#[cfg(test)]
#[path = "parallelize_tests.rs"]
mod parallelize_tests;
