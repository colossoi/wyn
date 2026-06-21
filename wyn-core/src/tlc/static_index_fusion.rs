//! Static-index fusion.
//!
//! Indexing an elementwise producer at a constant — `map(f, src)[k]` — demands a
//! single element, not the whole array. Materializing `src` to a buffer just to
//! read one known slot is the wrong default; instead fuse the index into the
//! producer:
//!
//! ```text
//! map(f, src)[k]  ⟶  let p = src[k] in <f's body>
//! ```
//!
//! For `[g(256)[3]]` (with `g` inlined to `map(|i| f32.i32(i), 0..<256)`) this
//! collapses to `let i = (0..<256)[3] in f32.i32(i)` — a virtual-array access
//! plus a scalar convert, materializing nothing.
//!
//! Scope: the index must be a literal, and the producer must be a *directly
//! nested* `Soac(Map)` under the `Index` (an inlined helper's body, possibly
//! wrapped in `let`s). A producer reached through a `Var` — i.e. let-bound and
//! potentially read more than once — is deliberately left alone; fusing it would
//! duplicate the producer per index, so multi-consumer and runtime-indexed cases
//! belong to the gather/materialization path instead.
//!
//! Runs post-materialize (from `parallelize::run`), where an inlined helper's
//! producer is a directly-nested `Soac(Map)` under the `Index`.

use super::{ArrayExpr, Lambda, Program, SoacBody, SoacOp, Term, TermIdSource, TermKind};
use crate::ast::TypeName;
use crate::SymbolId;
use polytype::Type;

/// Fuse every `Index(<directly-nested elementwise producer>, <literal>)` in the
/// program into a scalar element computation.
pub fn run(mut program: Program) -> Program {
    let mut ids = TermIdSource::new();
    for def in &mut program.defs {
        let body = def.body.clone();
        def.body = fuse(body, &mut ids);
    }
    program
}

fn fuse(term: Term, ids: &mut TermIdSource) -> Term {
    // Bottom-up: fuse inner indices first so an outer fusion sees the simplified
    // children.
    let term = term.map_children(&mut |c| fuse(c, ids));
    if let TermKind::Index { array, index } = &term.kind {
        if matches!(index.kind, TermKind::IntLit(_)) {
            if let Some(fused) = try_fuse(array, index, &term.ty, ids) {
                return fused;
            }
        }
    }
    term
}

/// Push `[index]` through enclosing `let`s and, at a `Soac(Map)`, replace the
/// whole `Index` with the lambda body bound to the indexed input elements.
/// `result_ty` is the element type (the original `Index`'s type). `None` if
/// `array` isn't an elementwise producer we can index through.
fn try_fuse(
    array: &Term,
    index: &Term,
    result_ty: &Type<TypeName>,
    ids: &mut TermIdSource,
) -> Option<Term> {
    match &array.kind {
        // Index(let n = r in b, k) ≡ let n = r in Index(b, k) — fuse the inner.
        TermKind::Let {
            name,
            name_ty,
            rhs,
            body,
        } => {
            let inner = try_fuse(body, index, result_ty, ids)
                .unwrap_or_else(|| make_index(body, index, result_ty, ids));
            Some(Term {
                id: ids.next_id(),
                ty: result_ty.clone(),
                span: array.span,
                kind: TermKind::Let {
                    name: *name,
                    name_ty: name_ty.clone(),
                    rhs: rhs.clone(),
                    body: Box::new(inner),
                },
            })
        }
        // map(f, src)[k] → bind each param to its input's [k] element, then run
        // f's body. Captured free values are rebound first so the body's
        // references stay in scope.
        TermKind::Soac(SoacOp::Map { lam, inputs, .. }) => {
            let SoacBody {
                lam: Lambda { params, body, .. },
                captures,
            } = lam;
            if params.len() != inputs.len() {
                return None;
            }
            let mut elems = Vec::with_capacity(params.len());
            for ((_, pty), input) in params.iter().zip(inputs) {
                elems.push(index_elem(input, pty.clone(), index, ids)?);
            }
            let mut result = (**body).clone();
            for ((psym, pty), elem) in params.iter().zip(elems).rev() {
                result = make_let(*psym, pty.clone(), elem, result, array.span, ids);
            }
            for (csym, cty, cterm) in captures.iter().rev() {
                result = make_let(*csym, cty.clone(), cterm.clone(), result, array.span, ids);
            }
            Some(result)
        }
        _ => None,
    }
}

/// The element of `input` at `[index]` — `Index(t, index)` for a `Ref(t)`. Other
/// input shapes (raw `Range`/`Zip`/`Literal` not wrapped in `Ref`) aren't fused
/// here; returning `None` leaves the original `Index` in place.
fn index_elem(
    input: &ArrayExpr,
    elem_ty: Type<TypeName>,
    index: &Term,
    ids: &mut TermIdSource,
) -> Option<Term> {
    match input {
        ArrayExpr::Ref(t) => Some(Term {
            id: ids.next_id(),
            ty: elem_ty,
            span: t.span,
            kind: TermKind::Index {
                array: t.clone(),
                index: Box::new(index.clone()),
            },
        }),
        _ => None,
    }
}

fn make_index(array: &Term, index: &Term, result_ty: &Type<TypeName>, ids: &mut TermIdSource) -> Term {
    Term {
        id: ids.next_id(),
        ty: result_ty.clone(),
        span: array.span,
        kind: TermKind::Index {
            array: Box::new(array.clone()),
            index: Box::new(index.clone()),
        },
    }
}

fn make_let(
    name: SymbolId,
    name_ty: Type<TypeName>,
    rhs: Term,
    body: Term,
    span: crate::ast::Span,
    ids: &mut TermIdSource,
) -> Term {
    let body_ty = body.ty.clone();
    Term {
        id: ids.next_id(),
        ty: body_ty,
        span,
        kind: TermKind::Let {
            name,
            name_ty,
            rhs: Box::new(rhs),
            body: Box::new(body),
        },
    }
}

#[cfg(test)]
#[path = "static_index_fusion_tests.rs"]
mod static_index_fusion_tests;
