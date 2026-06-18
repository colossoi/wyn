//! TLC inlining pass.
//!
//! Runs after monomorphization, before SoA transform and SSA conversion.
//! Inlines small function bodies at call sites and into SOAC lambda bodies.
//! Everything is first-order and monomorphic at this point.

use super::VarRef;
use super::{
    Def, DefMeta, Program, Term, TermIdSource, TermKind, collect_var_refs, extract_lambda_params, term_size,
};
use crate::ast::{Span, TypeName};
use crate::{SymbolId, SymbolTable};
use polytype::Type;
use std::collections::{HashMap, HashSet};

/// Maximum term size for a user function to be inlined.
const INLINE_SIZE_THRESHOLD: usize = 30;

/// Force-inline every helper whose body contains a SOAC anywhere in its
/// term tree. This mirrors Futhark's "inline array/parallel callees" rule:
/// a helper that performs SOAC work behind a call boundary blocks fusion
/// from seeing the producer/consumer relationship across the boundary,
/// so we expose it by inlining. Size threshold is ignored — a helper
/// with a SOAC is by definition critical to fusion, regardless of source
/// LOC.
///
/// Iterates to a fixpoint so chains like `clump → center → sum` (each a
/// SOAC helper) fully expand: one round inlines `center`, the next sees
/// `sum` calls inside the freshly-expanded clump body and inlines those
/// too.
pub fn run_force_soac_helpers(mut program: Program) -> Program {
    // Bound iterations to guard against pathological recursion through
    // hand-crafted call graphs; typical wyn helper depth is 2–3.
    for _ in 0..8 {
        let candidates = build_soac_helper_candidates(&program);
        if candidates.is_empty() {
            return program;
        }
        // Stop when nothing in the program calls any current candidate.
        // (Inlining one round may expose new candidates — e.g. inlining
        // `sum` into `center`'s body makes `center` SOAC-bearing and a
        // candidate next round — so we re-detect candidates each iter.)
        if !any_def_calls_candidate(&program, &candidates) {
            return program;
        }
        let mut term_ids = TermIdSource::new();
        let new_defs: Vec<Def> = program
            .defs
            .into_iter()
            .map(|def| {
                let body = inline_term(def.body, &candidates, &mut term_ids);
                Def { body, ..def }
            })
            .collect();
        let new_defs = dead_code_eliminate(new_defs);
        program = Program {
            defs: new_defs,
            symbols: program.symbols,
            ..program
        };
    }
    program
}

fn build_soac_helper_candidates(program: &Program) -> HashMap<SymbolId, InlineBody> {
    let mut candidates = HashMap::new();
    for def in &program.defs {
        if !matches!(def.meta, DefMeta::Function) {
            continue;
        }
        // Entry points are roots, never call sites.
        let (params, body) = extract_lambda_params(&def.body);
        if params.is_empty() {
            continue;
        }
        if has_control_flow(&body) {
            continue;
        }
        if !contains_soac(&body) {
            continue;
        }
        // Skip helpers whose body carries an unresolved polytype `Variable`
        // (a free *type* parameter — distinct from a `SizeVar`). The
        // prelude's `unzip<[n], A, B>` is the load-bearing example: its
        // body contains SOAC `map` calls, so it qualifies as a SOAC
        // helper, but if we force-inline it the body's free `A`/`B`
        // type variables go nowhere — `monomorphize` only resolves
        // polymorphism at *function-definition* boundaries, not inside
        // already-inlined bodies, so the variables ride all the way to
        // the SPIR-V backend and panic on "Unresolved type variable".
        // Defer such helpers to the canonical
        // `monomorphize → fold_generated_lambdas → inline_small` path
        // that actually unifies their call sites.
        if term_contains_free_type_variable(&body)
            || params.iter().any(|(_, ty)| type_contains_variable(ty))
        {
            continue;
        }
        candidates.insert(def.name, InlineBody { params, body });
    }
    candidates
}

/// True if `ty` references any polytype `Type::Variable(_)` — an
/// unresolved type parameter. Doesn't trigger on size-vars or any
/// other nullary `TypeName::*`-marker.
fn type_contains_variable(ty: &Type<TypeName>) -> bool {
    match ty {
        Type::Variable(_) => true,
        Type::Constructed(_, args) => args.iter().any(type_contains_variable),
    }
}

/// True if any subterm's carried type contains a polytype `Variable` —
/// or, equivalently, if the def's body still carries free type
/// parameters that monomorphize would resolve.
fn term_contains_free_type_variable(term: &Term) -> bool {
    if type_contains_variable(&term.ty) {
        return true;
    }
    let mut found = false;
    term.for_each_child(&mut |c| {
        if !found {
            found = term_contains_free_type_variable(c);
        }
    });
    found
}

/// True if `term` contains anything fusion's classifier treats as a use of an
/// array: a `Soac` node, *or* a call to the `length` builtin. Helpers that
/// hide either behind a non-inlined call boundary block multi-consumer
/// fusion the same way, so force-inline catches both.
fn contains_soac(term: &Term) -> bool {
    if matches!(&term.kind, TermKind::Soac(_)) {
        return true;
    }
    if is_length_intrinsic_call(term) {
        return true;
    }
    let mut found = false;
    term.for_each_child(&mut |c| {
        if !found {
            found = contains_soac(c);
        }
    });
    found
}

fn is_length_intrinsic_call(term: &Term) -> bool {
    let TermKind::App { func, args } = &term.kind else {
        return false;
    };
    if args.len() != 1 {
        return false;
    }
    let TermKind::Var(super::VarRef::Builtin { id, .. }) = &func.kind else {
        return false;
    };
    *id == crate::builtins::catalog().known().length
}

fn any_def_calls_candidate(program: &Program, candidates: &HashMap<SymbolId, InlineBody>) -> bool {
    fn walk(term: &Term, cs: &HashMap<SymbolId, InlineBody>) -> bool {
        if let TermKind::App { func, .. } = &term.kind {
            if let TermKind::Var(VarRef::Symbol(s)) = &func.kind {
                if cs.contains_key(s) {
                    return true;
                }
            }
        }
        let mut found = false;
        term.for_each_child(&mut |c| {
            if !found {
                found = walk(c, cs);
            }
        });
        found
    }
    program.defs.iter().any(|def| walk(&def.body, candidates))
}

// =============================================================================
// Public API
// =============================================================================

/// Eliminate unreachable defs from a TLC program.
///
/// Preserves entry points, extern defs, and their transitive dependencies.
pub fn run_reachable(program: Program) -> Program {
    let defs = dead_code_eliminate(program.defs);
    let result = Program {
        defs,
        symbols: program.symbols,
        ..program
    };
    super::hof_specialize::verify_hof_specialized(&result).unwrap_or_else(|e| {
        panic!(
            "hof-specialization verifier failed after filter_reachable: {:?}",
            e
        )
    });
    result
}

/// Inline small user functions and constants at their call/reference sites.
///
/// This is the TLC equivalent of `ssa::ssa_inline::inline_small_functions`.
/// Inlines:
/// - Small user functions (term size ≤ threshold, no control flow or SOACs)
/// - Constants (arity-0 defs, substituted at Var reference sites)
///
/// Skips `LiftedLambda` defs (SOAC bodies) — handled by `inline()`.
pub fn run_small(program: Program) -> Program {
    let all_constants = find_all_constants(&program);
    let mut small_candidates = find_small_candidates(&program.defs, &program.symbols);

    if small_candidates.is_empty() && all_constants.is_empty() {
        return program;
    }

    // Inline constants into small candidate bodies so that when we inline
    // the candidate into a call site, the inlined body doesn't carry stale
    // Var references to constants.
    for (_sym, candidate) in &mut small_candidates {
        candidate.body = inline_constants(candidate.body.clone(), &all_constants);
    }

    let mut term_ids = TermIdSource::new();
    let defs: Vec<Def> = program
        .defs
        .into_iter()
        .map(|def| {
            // Constants are pure — inline them everywhere, including lambda bodies.
            let body = inline_constants(def.body, &all_constants);
            let body = inline_term(body, &small_candidates, &mut term_ids);
            Def { body, ..def }
        })
        .collect();

    let defs = dead_code_eliminate(defs);

    Program {
        defs,
        symbols: program.symbols,
        ..program
    }
}

/// Inline compiler-generated lifted-lambda defs (`DefMeta::LiftedLambda`) in a TLC program.
pub fn run_large(program: Program) -> Program {
    let inline_candidates = find_inline_candidates(&program.defs, &program.symbols);

    let mut term_ids = TermIdSource::new();
    let defs: Vec<Def> = program
        .defs
        .into_iter()
        .map(|def| {
            let body = inline_term(def.body, &inline_candidates, &mut term_ids);
            Def { body, ..def }
        })
        .collect();

    // DCE: remove defs not referenced by any entry point or reachable def.
    let defs = dead_code_eliminate(defs);

    let result = Program {
        defs,
        symbols: program.symbols,
        ..program
    };
    result.assert_flat_apps();
    result
}

// =============================================================================
// Inline candidate analysis
// =============================================================================

/// A function body ready for inlining: flat params + inner body.
#[derive(Clone)]
struct InlineBody {
    params: Vec<(SymbolId, Type<TypeName>)>,
    body: Term,
}

/// Find small user functions suitable for inlining.
///
/// A function qualifies if:
/// - It's a `DefMeta::Function` (lifted lambdas are handled by `run_large`)
/// - It has parameters (not a constant)
/// - Its body is small (term_size ≤ threshold)
/// - Its body has no control flow (If, Loop) or SOACs
fn find_small_candidates(defs: &[Def], _symbols: &SymbolTable) -> HashMap<SymbolId, InlineBody> {
    let mut candidates = HashMap::new();

    for def in defs {
        if !matches!(def.meta, DefMeta::Function) {
            continue;
        }

        let (params, body) = extract_lambda_params(&def.body);
        if params.is_empty() {
            continue; // constant, not a function
        }

        // Size check.
        if term_size(&body) > INLINE_SIZE_THRESHOLD {
            continue;
        }

        // Skip functions with control flow (If, Loop) — they'd become multi-block in SSA.
        // SOACs are fine to inline — they're single instructions in SSA.
        if has_control_flow(&body) {
            continue;
        }

        candidates.insert(def.name, InlineBody { params, body });
    }

    candidates
}

/// Find all constant defs, indexed by every SymbolId that could reference them.
///
/// A constant is an arity-0, non-entry, non-extern function def.
/// After monomorphization, the same constant name may be referenced through
/// different SymbolIds, so we index by name as well as by def SymbolId.
fn find_all_constants(program: &Program) -> HashMap<SymbolId, Term> {
    // Find the canonical constant defs.
    let mut by_sym: HashMap<SymbolId, Term> = HashMap::new();
    let mut by_name: HashMap<String, Term> = HashMap::new();

    for def in &program.defs {
        if !matches!(def.meta, DefMeta::Function) {
            continue;
        }
        if matches!(def.body.kind, TermKind::Extern(_)) {
            continue;
        }
        if def.arity != 0 {
            continue;
        }
        by_sym.insert(def.name, def.body.clone());
        if let Some(name) = program.symbols.get(def.name) {
            by_name.insert(name.clone(), def.body.clone());
        }
    }

    // Also map any other SymbolId that resolves to a constant's name.
    for def in &program.defs {
        if let Some(name) = program.symbols.get(def.name) {
            if let Some(body) = by_name.get(name) {
                by_sym.entry(def.name).or_insert_with(|| body.clone());
            }
        }
    }
    for (name, body) in &by_name {
        if let Some(&def_sym) = program.def_syms.get(name) {
            by_sym.entry(def_sym).or_insert_with(|| body.clone());
        }
    }

    by_sym
}

/// Check if a term contains control flow (If, Loop) or SOACs.
fn has_control_flow(term: &Term) -> bool {
    match &term.kind {
        TermKind::If { .. } | TermKind::Loop { .. } => true,
        _ => {
            let mut found = false;
            term.for_each_child(&mut |child| {
                if !found {
                    found = has_control_flow(child);
                }
            });
            found
        }
    }
}

/// Replace `Var(sym)` references with the constant body when `sym` is a constant candidate.
fn inline_constants(term: Term, constants: &HashMap<SymbolId, Term>) -> Term {
    let term = term.map_children(&mut |child| inline_constants(child, constants));

    if let TermKind::Var(VarRef::Symbol(sym)) = &term.kind {
        if let Some(body) = constants.get(sym) {
            return body.clone();
        }
    }

    term
}

/// Determine which defs are candidates for inlining.
/// Inlines all `DefMeta::LiftedLambda` defs — closure_convert-produced lifted
/// lambdas that we're putting back where they came from.
fn find_inline_candidates(defs: &[Def], _symbols: &SymbolTable) -> HashMap<SymbolId, InlineBody> {
    let mut candidates = HashMap::new();

    for def in defs {
        if !matches!(def.meta, DefMeta::LiftedLambda) {
            continue;
        }

        let (params, body) = extract_lambda_params(&def.body);

        if params.is_empty() {
            continue;
        }

        candidates.insert(
            def.name,
            InlineBody {
                params,
                body: body.clone(),
            },
        );
    }

    candidates
}

// =============================================================================
// Term inlining (via map_children)
// =============================================================================

/// Bottom-up: recurse into all children, then try to inline App nodes.
///
/// SOAC lambda bodies are bare Var refs to lifted defs after defunctionalization,
/// so recursing into them via map_children is harmless — the inline rewrite only
/// fires on fully-saturated App nodes matching candidates.
fn inline_term(term: Term, candidates: &HashMap<SymbolId, InlineBody>, ids: &mut TermIdSource) -> Term {
    let term = term.map_children(&mut |child| inline_term(child, candidates, ids));

    // Only App nodes can be inline sites.
    let TermKind::App { ref func, ref args } = term.kind else {
        return term;
    };

    // If the head is a Var referencing an inline candidate, inline it.
    if let TermKind::Var(VarRef::Symbol(sym)) = &func.kind {
        if let Some(ib) = candidates.get(sym) {
            if args.len() == ib.params.len() {
                let span = term.span;
                let TermKind::App { func: _, args } = term.kind else {
                    unreachable!()
                };
                return build_inline_lets(&ib.params, &args, ib.body.clone(), span, ids);
            }
        }
    }

    // Not inlineable — return as-is.
    term
}

// =============================================================================
// Small helpers
// =============================================================================

fn mk(ids: &mut TermIdSource, ty: Type<TypeName>, span: Span, kind: TermKind) -> Term {
    Term {
        id: ids.next_id(),
        ty,
        span,
        kind,
    }
}

/// Build Let bindings to substitute params with args, wrapping the inlined body.
///
/// The let's `name_ty` comes from the *arg*'s concrete type rather than the
/// param's declared type. The param's declared type may be polymorphic
/// (e.g. `[]T` is `Array[T, Abstract, Skolem, …]`); the arg at the call site
/// has the concrete instantiation. Using the arg type avoids dragging the
/// Abstract array variant into post-inline let chains where it would later
/// hit `egir::verify_no_abstract` at backend lowering.
///
/// Special case: when the arg is a bare `Var(SymbolId)` reference (i.e. the
/// caller is passing an in-scope binding straight through), we substitute
/// `param_sym → arg_sym` into the body instead of emitting a redundant
/// `let param_sym = arg_var in body`. The alias-let is correct in
/// principle but downstream passes — `egir::from_tlc::convert_soac_filter`
/// in particular — read attributes (storage region, ownership) directly
/// off the original symbol and don't follow let-bound aliases, so a
/// post-inline `let arr = xs in filter(p, arr)` loses the entry-param's
/// region info. This is *not* general beta-reduction — only the trivial
/// `let x = y in body` case where `y` is a `Var`. Non-Var args still
/// get the `let` wrap so we don't duplicate side-effecting computation
/// under multi-use params.
pub(crate) fn build_inline_lets(
    params: &[(SymbolId, Type<TypeName>)],
    args: &[Term],
    body: Term,
    span: Span,
    ids: &mut TermIdSource,
) -> Term {
    let mut result = body;
    for ((sym, _param_ty), arg) in params.iter().rev().zip(args.iter().rev()) {
        if let TermKind::Var(VarRef::Symbol(arg_sym)) = &arg.kind {
            // Substituting the *symbol* alone is not enough: the param's
            // declared type may be polymorphic (e.g. `[]T` with a region
            // type-variable), while the call-site arg carries the
            // concrete instantiation (a `Region(set, binding)`). If we
            // only rewrite the symbol, the substituted `Var` still
            // carries the param's polymorphic type and downstream type
            // walks (notably the SPIR-V backend's view-region check)
            // see an unresolved type variable. So replace both: the
            // var ref *and* its type, at every occurrence.
            result = substitute_sym_and_retype(result, *sym, *arg_sym, &arg.ty, ids);
            continue;
        }
        result = mk(
            ids,
            result.ty.clone(),
            span,
            TermKind::Let {
                name: *sym,
                name_ty: arg.ty.clone(),
                rhs: Box::new(arg.clone()),
                body: Box::new(result),
            },
        );
    }
    result
}

/// `substitute_sym` variant that *also* overrides the substituted
/// `Var(old)`'s carried type with `new_ty`. Used by `build_inline_lets`
/// for the bare-Var-arg fast path: when we forward a call-site
/// concrete-region arg through a polymorphic param, the substituted
/// `Var` must end up carrying the concrete type, not the polymorphic
/// param type that was on the original term.
fn substitute_sym_and_retype(
    term: Term,
    old: SymbolId,
    new: SymbolId,
    new_ty: &Type<TypeName>,
    term_ids: &mut TermIdSource,
) -> Term {
    match term.kind {
        TermKind::Var(VarRef::Symbol(s)) if s == old => Term {
            id: term_ids.next_id(),
            ty: new_ty.clone(),
            kind: TermKind::Var(VarRef::Symbol(new)),
            span: term.span,
        },
        TermKind::Var(VarRef::Symbol(_))
        | TermKind::Var(VarRef::Builtin { .. })
        | TermKind::BinOp(_)
        | TermKind::UnOp(_)
        | TermKind::IntLit(_)
        | TermKind::FloatLit(_)
        | TermKind::BoolLit(_)
        | TermKind::Extern(_) => term,

        TermKind::Lambda(ref lam) if lam.params.iter().any(|(p, _)| *p == old) => term,

        TermKind::Let {
            name,
            name_ty,
            rhs,
            body,
        } => {
            let new_rhs = substitute_sym_and_retype(*rhs, old, new, new_ty, term_ids);
            let new_body = if name == old {
                *body
            } else {
                substitute_sym_and_retype(*body, old, new, new_ty, term_ids)
            };
            Term {
                id: term_ids.next_id(),
                kind: TermKind::Let {
                    name,
                    name_ty,
                    rhs: Box::new(new_rhs),
                    body: Box::new(new_body),
                },
                ..term
            }
        }

        _ => term.map_children(&mut |child| substitute_sym_and_retype(child, old, new, new_ty, term_ids)),
    }
}

// =============================================================================
// Dead code elimination
// =============================================================================

/// Remove defs that are not referenced by any entry point or reachable def.
///
/// Preserves:
/// - All entry points and their transitive dependencies
/// - Extern defs (linked SPIR-V functions, needed even if not directly referenced)
pub fn dead_code_eliminate(defs: Vec<Def>) -> Vec<Def> {
    let mut reachable: HashSet<SymbolId> = HashSet::new();
    let mut worklist: Vec<SymbolId> = Vec::new();

    // Seed with entry points and extern defs.
    for def in &defs {
        let is_root =
            matches!(def.meta, DefMeta::EntryPoint(_)) || matches!(def.body.kind, TermKind::Extern(_));
        if is_root {
            reachable.insert(def.name);
            worklist.push(def.name);
        }
    }

    let def_map: HashMap<SymbolId, &Def> = defs.iter().map(|d| (d.name, d)).collect();

    while let Some(sym) = worklist.pop() {
        if let Some(def) = def_map.get(&sym) {
            for r in collect_var_refs(&def.body) {
                if def_map.contains_key(&r) && reachable.insert(r) {
                    worklist.push(r);
                }
            }
        }
    }

    defs.into_iter().filter(|d| reachable.contains(&d.name)).collect()
}
