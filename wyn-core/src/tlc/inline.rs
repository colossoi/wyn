//! TLC inlining pass.
//!
//! Runs after monomorphization, before SoA transform and SSA conversion.
//! Inlines small function bodies at call sites and into SOAC lambda bodies.
//! Everything is first-order and monomorphic at this point.

use super::{
    ArrayExpr, Def, DefMeta, Lambda, LoopKind, Program, SoacOp, Term, TermId, TermIdSource, TermKind,
    collect_var_refs, extract_lambda_params,
};
use crate::ast::{Span, TypeName};
use crate::{SymbolId, SymbolTable};
use polytype::Type;
use std::collections::{HashMap, HashSet};

// =============================================================================
// Public API
// =============================================================================

/// Inline compiler-generated lambda defs (`_w_lambda_*`) in a TLC program.
pub fn inline(program: Program) -> Program {
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

    Program {
        defs,
        uniforms: program.uniforms,
        storage: program.storage,
        symbols: program.symbols,
    }
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
// Term inlining
// =============================================================================

/// Inline within a term, rewriting call sites and SOAC lambda bodies.
fn inline_term(term: Term, candidates: &HashMap<SymbolId, InlineBody>, ids: &mut TermIdSource) -> Term {
    let ty = term.ty;
    let span = term.span;

    match term.kind {
        // Try to inline curried application chains: f(a)(b)(c)
        TermKind::App { func, arg } => {
            let func = inline_term(*func, candidates, ids);
            let arg = inline_term(*arg, candidates, ids);

            // Collect the full application spine.
            let (head, args) = collect_app_spine_owned(func, arg);

            // If the head is a Var referencing an inline candidate, inline it.
            if let TermKind::Var(sym) = &head.kind {
                if let Some(ib) = candidates.get(sym) {
                    if args.len() == ib.params.len() {
                        return build_inline_lets(&ib.params, &args, ib.body.clone(), span, ids);
                    }
                }
            }

            // Not inlineable — rebuild the App chain.
            rebuild_app_chain(head, args, ty, span)
        }

        TermKind::Let {
            name,
            name_ty,
            rhs,
            body,
        } => {
            let rhs = inline_term(*rhs, candidates, ids);
            let body = inline_term(*body, candidates, ids);
            mk(
                ids,
                ty,
                span,
                TermKind::Let {
                    name,
                    name_ty,
                    rhs: Box::new(rhs),
                    body: Box::new(body),
                },
            )
        }

        TermKind::Lambda(lam) => {
            let lam = inline_lambda(lam, candidates, ids);
            mk(ids, ty, span, TermKind::Lambda(lam))
        }

        TermKind::If {
            cond,
            then_branch,
            else_branch,
        } => {
            let cond = inline_term(*cond, candidates, ids);
            let then_branch = inline_term(*then_branch, candidates, ids);
            let else_branch = inline_term(*else_branch, candidates, ids);
            mk(
                ids,
                ty,
                span,
                TermKind::If {
                    cond: Box::new(cond),
                    then_branch: Box::new(then_branch),
                    else_branch: Box::new(else_branch),
                },
            )
        }

        TermKind::Loop {
            loop_var,
            loop_var_ty,
            init,
            init_bindings,
            kind,
            body,
        } => {
            let init = inline_term(*init, candidates, ids);
            let init_bindings = init_bindings
                .into_iter()
                .map(|(s, t, e)| (s, t, inline_term(e, candidates, ids)))
                .collect();
            let kind = inline_loop_kind(kind, candidates, ids);
            let body = inline_term(*body, candidates, ids);
            mk(
                ids,
                ty,
                span,
                TermKind::Loop {
                    loop_var,
                    loop_var_ty,
                    init: Box::new(init),
                    init_bindings,
                    kind,
                    body: Box::new(body),
                },
            )
        }

        TermKind::Soac(soac) => {
            let soac = inline_soac(soac, candidates, ids);
            mk(ids, ty, span, TermKind::Soac(soac))
        }

        TermKind::ArrayExpr(ae) => {
            let ae = inline_array_expr(ae, candidates, ids);
            mk(ids, ty, span, TermKind::ArrayExpr(ae))
        }

        TermKind::Force(inner) => {
            let inner = inline_term(*inner, candidates, ids);
            mk(ids, ty, span, TermKind::Force(Box::new(inner)))
        }

        TermKind::Pack {
            exists_ty,
            dims,
            value,
        } => {
            let value = inline_term(*value, candidates, ids);
            mk(
                ids,
                ty,
                span,
                TermKind::Pack {
                    exists_ty,
                    dims,
                    value: Box::new(value),
                },
            )
        }

        TermKind::Unpack {
            scrut,
            dim_binders,
            value_binder,
            body,
        } => {
            let scrut = inline_term(*scrut, candidates, ids);
            let body = inline_term(*body, candidates, ids);
            mk(
                ids,
                ty,
                span,
                TermKind::Unpack {
                    scrut: Box::new(scrut),
                    dim_binders,
                    value_binder,
                    body: Box::new(body),
                },
            )
        }

        // Leaves.
        _ => Term { ty, span, ..term },
    }
}

// =============================================================================
// SOAC lambda inlining
// =============================================================================

fn inline_lambda(
    lam: Lambda,
    candidates: &HashMap<SymbolId, InlineBody>,
    ids: &mut TermIdSource,
) -> Lambda {
    Lambda {
        body: Box::new(inline_term(*lam.body, candidates, ids)),
        captures: lam
            .captures
            .into_iter()
            .map(|(s, t, e)| (s, t, inline_term(e, candidates, ids)))
            .collect(),
        ..lam
    }
}

fn inline_soac_lambda(
    lam: Lambda,
    candidates: &HashMap<SymbolId, InlineBody>,
    ids: &mut TermIdSource,
) -> Lambda {
    inline_lambda(lam, candidates, ids)
}

// =============================================================================
// Recursive helpers for SOAC / ArrayExpr / LoopKind
// =============================================================================

/// Inline within a SOAC's non-lambda subterms. SOAC lambdas are left
/// untouched — they are Var references to lifted Defs and must stay that way
/// for to_ssa, which emits calls to those Defs in the loop body.
fn inline_soac(soac: SoacOp, candidates: &HashMap<SymbolId, InlineBody>, ids: &mut TermIdSource) -> SoacOp {
    match soac {
        SoacOp::Map { lam, inputs } => SoacOp::Map {
            lam,
            inputs: inputs.into_iter().map(|ae| inline_array_expr(ae, candidates, ids)).collect(),
        },
        SoacOp::Reduce { op, ne, input, props } => SoacOp::Reduce {
            op,
            ne: Box::new(inline_term(*ne, candidates, ids)),
            input: inline_array_expr(input, candidates, ids),
            props,
        },
        SoacOp::Scan { op, ne, input } => SoacOp::Scan {
            op,
            ne: Box::new(inline_term(*ne, candidates, ids)),
            input: inline_array_expr(input, candidates, ids),
        },
        SoacOp::Filter { pred, input } => SoacOp::Filter {
            pred,
            input: inline_array_expr(input, candidates, ids),
        },
        SoacOp::Scatter {
            dest,
            indices,
            values,
        } => SoacOp::Scatter {
            dest,
            indices: inline_array_expr(indices, candidates, ids),
            values: inline_array_expr(values, candidates, ids),
        },
        SoacOp::ReduceByIndex {
            dest,
            op,
            ne,
            indices,
            values,
            props,
        } => SoacOp::ReduceByIndex {
            dest,
            op,
            ne: Box::new(inline_term(*ne, candidates, ids)),
            indices: inline_array_expr(indices, candidates, ids),
            values: inline_array_expr(values, candidates, ids),
            props,
        },
    }
}

fn inline_array_expr(
    ae: ArrayExpr,
    candidates: &HashMap<SymbolId, InlineBody>,
    ids: &mut TermIdSource,
) -> ArrayExpr {
    match ae {
        ArrayExpr::Ref(t) => ArrayExpr::Ref(Box::new(inline_term(*t, candidates, ids))),
        ArrayExpr::Zip(aes) => {
            ArrayExpr::Zip(aes.into_iter().map(|ae| inline_array_expr(ae, candidates, ids)).collect())
        }
        ArrayExpr::Soac(op) => ArrayExpr::Soac(Box::new(inline_soac(*op, candidates, ids))),
        ArrayExpr::Generate {
            shape,
            index_fn,
            elem_ty,
        } => ArrayExpr::Generate {
            shape,
            index_fn: inline_soac_lambda(index_fn, candidates, ids),
            elem_ty,
        },
        ArrayExpr::Literal(terms) => {
            ArrayExpr::Literal(terms.into_iter().map(|t| inline_term(t, candidates, ids)).collect())
        }
        ArrayExpr::Range { start, len } => ArrayExpr::Range {
            start: Box::new(inline_term(*start, candidates, ids)),
            len: Box::new(inline_term(*len, candidates, ids)),
        },
    }
}

fn inline_loop_kind(
    kind: LoopKind,
    candidates: &HashMap<SymbolId, InlineBody>,
    ids: &mut TermIdSource,
) -> LoopKind {
    match kind {
        LoopKind::For { var, var_ty, iter } => LoopKind::For {
            var,
            var_ty,
            iter: Box::new(inline_term(*iter, candidates, ids)),
        },
        LoopKind::ForRange { var, var_ty, bound } => LoopKind::ForRange {
            var,
            var_ty,
            bound: Box::new(inline_term(*bound, candidates, ids)),
        },
        LoopKind::While { cond } => LoopKind::While {
            cond: Box::new(inline_term(*cond, candidates, ids)),
        },
    }
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

/// Owning version of collect_app_spine: destructures the App chain.
fn collect_app_spine_owned(func: Term, arg: Term) -> (Term, Vec<Term>) {
    let mut args = vec![arg];
    let mut current = func;
    loop {
        match current.kind {
            TermKind::App { func, arg } => {
                args.push(*arg);
                current = *func;
            }
            _ => {
                args.reverse();
                return (current, args);
            }
        }
    }
}

/// Rebuild a curried App chain from head + args.
fn rebuild_app_chain(head: Term, args: Vec<Term>, final_ty: Type<TypeName>, span: Span) -> Term {
    if args.is_empty() {
        return head;
    }
    let n = args.len();
    let mut result = head;
    for (i, arg) in args.into_iter().enumerate() {
        let ty = if i == n - 1 {
            final_ty.clone()
        } else {
            // Intermediate arrow type — peel the return type.
            match &result.ty {
                Type::Constructed(TypeName::Arrow, ref a) if a.len() == 2 => a[1].clone(),
                other => other.clone(),
            }
        };
        result = Term {
            id: TermId(0),
            ty,
            span,
            kind: TermKind::App {
                func: Box::new(result),
                arg: Box::new(arg),
            },
        };
    }
    result
}

// =============================================================================
// Dead code elimination
// =============================================================================

/// Remove defs that are not referenced by any entry point or reachable def.
fn dead_code_eliminate(defs: Vec<Def>) -> Vec<Def> {
    let mut reachable: HashSet<SymbolId> = HashSet::new();
    let mut worklist: Vec<SymbolId> = Vec::new();

    // Seed with entry points.
    for def in &defs {
        if matches!(def.meta, DefMeta::EntryPoint(_)) {
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
