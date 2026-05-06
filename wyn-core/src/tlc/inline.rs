//! TLC inlining pass.
//!
//! Runs after monomorphization, before SoA transform and SSA conversion.
//! Inlines small function bodies at call sites and into SOAC lambda bodies.
//! Everything is first-order and monomorphic at this point.

use super::{
    Def, DefMeta, Program, Term, TermIdSource, TermKind, collect_var_refs, extract_lambda_params, term_size,
};
use crate::ast::{Span, TypeName};
use crate::{SymbolId, SymbolTable};
use polytype::Type;
use std::collections::{HashMap, HashSet};

/// Maximum term size for a user function to be inlined.
const INLINE_SIZE_THRESHOLD: usize = 30;

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
        uniforms: program.uniforms,
        storage: program.storage,
        symbols: program.symbols,
        def_syms: program.def_syms,
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
/// Skips `_w_lambda_*` defs (SOAC bodies) — handled by `inline()`.
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
        uniforms: program.uniforms,
        storage: program.storage,
        symbols: program.symbols,
        def_syms: program.def_syms,
    }
}

/// Inline compiler-generated lambda defs (`_w_lambda_*`) in a TLC program.
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

    // DCE: remove defs no longer referenced by any entry point or reachable def.
    let defs = dead_code_eliminate(defs);

    let result = Program {
        defs,
        uniforms: program.uniforms,
        storage: program.storage,
        symbols: program.symbols,
        def_syms: program.def_syms,
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
/// - It's not a `_w_lambda_*` (those are handled by `inline()`)
/// - It's not an entry point
/// - It has parameters (not a constant)
/// - Its body is small (term_size ≤ threshold)
/// - Its body has no control flow (If, Loop) or SOACs
fn find_small_candidates(defs: &[Def], symbols: &SymbolTable) -> HashMap<SymbolId, InlineBody> {
    let mut candidates = HashMap::new();

    for def in defs {
        if !matches!(def.meta, DefMeta::Function) {
            continue;
        }

        let name = match symbols.get(def.name) {
            Some(n) => n,
            None => continue,
        };

        // Skip lambda defs — handled by inline().
        if name.starts_with("_w_lambda_") {
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

    if let TermKind::Var(sym) = &term.kind {
        if let Some(body) = constants.get(sym) {
            return body.clone();
        }
    }

    term
}

/// Determine which defs are candidates for inlining.
/// Inlines all `_w_lambda_*` defs — these are compiler-generated lifted lambdas
/// from defunctionalization that we're putting back where they came from.
fn find_inline_candidates(defs: &[Def], symbols: &SymbolTable) -> HashMap<SymbolId, InlineBody> {
    let mut candidates = HashMap::new();

    for def in defs {
        let name = match symbols.get(def.name) {
            Some(n) => n,
            None => continue,
        };

        if !name.starts_with("_w_lambda_") {
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
    if let TermKind::Var(sym) = &func.kind {
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
fn build_inline_lets(
    params: &[(SymbolId, Type<TypeName>)],
    args: &[Term],
    body: Term,
    span: Span,
    ids: &mut TermIdSource,
) -> Term {
    let mut result = body;
    for ((sym, param_ty), arg) in params.iter().rev().zip(args.iter().rev()) {
        result = mk(
            ids,
            result.ty.clone(),
            span,
            TermKind::Let {
                name: *sym,
                name_ty: param_ty.clone(),
                rhs: Box::new(arg.clone()),
                body: Box::new(result),
            },
        );
    }
    result
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
