//! Entry-boundary SOAC exposure / normalization.
//!
//! This pass does NOT parallelize or lift anything itself. It makes SOACs that
//! a user factored into a helper *visible at the entry boundary*, so the
//! downstream passes that already do that work (`lift_gathers`, `parallelize`)
//! can see them. Those passes only recognize a SOAC that physically sits in the
//! entry's own body; a `def stencil(xs) = scan(..)` called as
//! `entry e(xs) = stencil(xs)` hides the `scan` behind a call. Here we inline
//! such producer calls into the entry so the SOAC is present.
//!
//! Detection reuses `array_semantics` function summaries (a function whose
//! *result* is a SOAC is a "producer"); materialization reuses `inline.rs`'s
//! `build_inline_lets`.
//!
//! LOAD-BEARING INVARIANT: the walk visits only the entry's outer `Lambda`
//! chain + its top-level `Let` chain + the tail. It MUST NOT descend into a
//! SOAC operand lambda (or any lambda). A SOAC helper called *per element*
//! (`map(|x| helper_scan(x), xs)`) lives inside such a lambda; materializing it
//! there would expose a per-element scan to parallelization and wreck the
//! program's cost semantics. This is the same nested-vs-tail line
//! `array_semantics::analyze_body` draws.

use std::collections::HashMap;

use polytype::Type;

use crate::SymbolId;
use crate::ast::TypeName;

use super::array_semantics::{FunctionSummary, ResultSemantics, summarize_program};
use super::fusion::{build_sym_to_def, substitute_sym};
use super::inline::build_inline_lets;
use super::{DefMeta, Lambda, Program, Term, TermIdSource, TermKind, VarRef, extract_lambda_params};

/// Bound on inline-chain depth (chained/recursive producers). A real producer
/// chain is short; the cap only stops a pathological self-recursive helper from
/// looping forever — it leaves a residual call, which lowers normally.
const MAX_DEPTH: usize = 64;

/// A producer helper: its params + body (the `λparams. SOAC…`'s inner body).
type Producer = (Vec<(SymbolId, Type<TypeName>)>, Term);

pub fn run(mut program: Program) -> Program {
    let summaries = summarize_program(&program);
    let sym_to_def = build_sym_to_def(&program.symbols, &program.def_syms);

    // A "producer" is a non-extern `Function` def whose *result* (tail) is a
    // SOAC — `ResultSemantics::Produces`. The summary's tail-isolation is what
    // keeps per-element-internal SOACs (whose tail is the outer `map`) from
    // counting, so we never materialize a genuinely-nested SOAC.
    let producers: HashMap<SymbolId, Producer> = program
        .defs
        .iter()
        .filter(|d| matches!(d.meta, DefMeta::Function) && !matches!(d.body.kind, TermKind::Extern(_)))
        .filter(|d| {
            matches!(
                summaries.get(&d.name),
                Some(FunctionSummary {
                    result: ResultSemantics::Produces(_),
                    ..
                })
            )
        })
        .map(|d| {
            let (params, body) = extract_lambda_params(&d.body);
            (d.name, (params, body))
        })
        .collect();

    if producers.is_empty() {
        return program;
    }

    let mut ids = TermIdSource::new();
    for def in program.defs.iter_mut() {
        if !matches!(def.meta, DefMeta::EntryPoint(_)) {
            continue;
        }
        let body = def.body.clone();
        def.body = expose(body, &producers, &sym_to_def, &mut ids, 0);
    }

    // Inlined-then-dead producer helpers are removed by the later reachability
    // DCE (`inline::run_reachable`); we deliberately don't DCE here, so a lone
    // producer def with no entry caller (as in unit-test fragments) survives.
    program
}

/// Walk the entry's outer `Lambda` chain + top-level `Let` chain + tail,
/// inlining producer calls. Never descends into a lambda body other than the
/// entry's own outer param-binding lambda(s).
fn expose(
    term: Term,
    producers: &HashMap<SymbolId, Producer>,
    sym_to_def: &HashMap<SymbolId, SymbolId>,
    ids: &mut TermIdSource,
    depth: usize,
) -> Term {
    let Term { id, ty, span, kind } = term;
    match kind {
        // The entry's outer param-binding lambda(s): descend the body only.
        TermKind::Lambda(lam) => {
            let body = expose(*lam.body, producers, sym_to_def, ids, depth);
            Term {
                id,
                ty,
                span,
                kind: TermKind::Lambda(Lambda {
                    params: lam.params,
                    body: Box::new(body),
                    ret_ty: lam.ret_ty,
                }),
            }
        }
        // Top-level let chain: a producer call may be the bound value; the body
        // continues the chain. (We do NOT recurse into arbitrary sub-terms of
        // the rhs — only inline it if the rhs itself is a producer call.)
        TermKind::Let {
            name,
            name_ty,
            rhs,
            body,
        } => {
            let new_rhs = maybe_inline(*rhs, producers, sym_to_def, ids, depth);
            let new_body = expose(*body, producers, sym_to_def, ids, depth);
            Term {
                id,
                ty,
                span,
                kind: TermKind::Let {
                    name,
                    name_ty,
                    rhs: Box::new(new_rhs),
                    body: Box::new(new_body),
                },
            }
        }
        // Tail position.
        other => maybe_inline(
            Term {
                id,
                ty,
                span,
                kind: other,
            },
            producers,
            sym_to_def,
            ids,
            depth,
        ),
    }
}

/// If `term` is a saturated call to a producer helper, inline it (params→args)
/// and re-`expose` the result so chained / arg-position producers materialize
/// too. Otherwise return `term` unchanged — crucially without recursing into
/// SOAC operands or lambdas.
fn maybe_inline(
    term: Term,
    producers: &HashMap<SymbolId, Producer>,
    sym_to_def: &HashMap<SymbolId, SymbolId>,
    ids: &mut TermIdSource,
    depth: usize,
) -> Term {
    if depth >= MAX_DEPTH {
        return term;
    }
    if let TermKind::App { func, args } = &term.kind {
        if let TermKind::Var(VarRef::Symbol(f)) = &func.kind {
            let def_sym = sym_to_def.get(f).copied().unwrap_or(*f);
            if let Some((params, body)) = producers.get(&def_sym) {
                if params.len() == args.len() {
                    let inlined = inline_call(params, args, body.clone(), term.span, ids);
                    return expose(inlined, producers, sym_to_def, ids, depth + 1);
                }
            }
        }
    }
    term
}

/// Inline a producer call: substitute each `Var` argument directly into the
/// body (beta-reduction, so the SOAC stays a directly-bound value the
/// downstream passes recognize), and bind any non-`Var` argument with a `let`.
/// A param-binding `let` around the SOAC would hide it from
/// `fusion`/`lift_gathers`/`analyze_entry`, which match on a directly-bound SOAC.
fn inline_call(
    params: &[(SymbolId, Type<TypeName>)],
    args: &[Term],
    body: Term,
    span: crate::ast::Span,
    ids: &mut TermIdSource,
) -> Term {
    let mut inlined = body;
    let mut let_params: Vec<(SymbolId, Type<TypeName>)> = Vec::new();
    let mut let_args: Vec<Term> = Vec::new();
    for ((psym, pty), arg) in params.iter().zip(args.iter()) {
        if let TermKind::Var(VarRef::Symbol(asym)) = &arg.kind {
            inlined = substitute_sym(inlined, *psym, *asym, ids);
        } else {
            let_params.push((*psym, pty.clone()));
            let_args.push(arg.clone());
        }
    }
    if let_params.is_empty() {
        inlined
    } else {
        build_inline_lets(&let_params, &let_args, inlined, span, ids)
    }
}
