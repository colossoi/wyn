//! TLC inlining pass.
//!
//! Runs after monomorphization, before SoA transform and SSA conversion.
//! Inlines small function bodies at call sites and into SOAC lambda bodies.
//! Everything is first-order and monomorphic at this point.

use super::{
    Def, DefMeta, Program, Term, TermId, TermIdSource, TermKind,
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
// Term inlining (via map_children)
// =============================================================================

/// Bottom-up: recurse into all children, then try to inline App spines.
///
/// SOAC lambda bodies are bare Var refs to lifted defs after defunctionalization,
/// so recursing into them via map_children is harmless — the inline rewrite only
/// fires on fully-saturated App spines matching candidates.
fn inline_term(term: Term, candidates: &HashMap<SymbolId, InlineBody>, ids: &mut TermIdSource) -> Term {
    let term = term.map_children(&mut |child| inline_term(child, candidates, ids));

    // Only App nodes can be inline sites.
    let TermKind::App { ref func, .. } = term.kind else {
        return term;
    };

    // Collect the full application spine: f(a)(b)(c) → (f, [a, b, c])
    // We need to destructure by value, so re-match after the ref check.
    let ty = term.ty.clone();
    let span = term.span;
    let TermKind::App { func, arg } = term.kind else {
        unreachable!()
    };
    let (head, args) = collect_app_spine_owned(*func, *arg);

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
