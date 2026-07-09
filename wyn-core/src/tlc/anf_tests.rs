use super::check;
use crate::ast::{Span, TypeName};
use crate::tlc::{
    ArrayExpr, Def, DefMeta, Lambda, Program, SoacBody, SoacDestination, SoacOp, Term, TermIdSource,
    TermKind, VarRef,
};
use crate::{LookupMap, SymbolId, SymbolTable};
use polytype::Type;

fn span() -> Span {
    Span::dummy()
}

fn i32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Int(32), vec![])
}

fn bool_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Bool, vec![])
}

fn arr_ty() -> Type<TypeName> {
    Type::Constructed(
        TypeName::Array,
        vec![
            i32_ty(),
            Type::Constructed(TypeName::ArrayVariantComposite, vec![]),
            Type::Variable(0),
            crate::types::no_buffer(),
        ],
    )
}

fn term(kind: TermKind, ty: Type<TypeName>, ids: &mut TermIdSource) -> Term {
    Term {
        id: ids.next_id(),
        ty,
        span: span(),
        kind,
    }
}

/// A trivial single-param lambda returning `lit` — stands in as a Filter
/// predicate or a Map's elementwise body.
fn trivial_lam(ret: Type<TypeName>, lit: TermKind, ids: &mut TermIdSource) -> SoacBody {
    SoacBody {
        lam: Lambda {
            params: vec![(SymbolId::from(99), i32_ty())],
            body: Box::new(term(lit, ret.clone(), ids)),
            ret_ty: ret,
        },
        captures: vec![],
    }
}

/// A named array input atom over symbol `sym`.
fn arr_var(sym: u32) -> ArrayExpr {
    ArrayExpr::Var(VarRef::Symbol(SymbolId::from(sym)), arr_ty())
}

/// `map(λx. 0, xs)` — a producer over a named input `xs` (itself ANF).
fn map_producer(xs: u32, ids: &mut TermIdSource) -> SoacOp {
    SoacOp::Map {
        lam: trivial_lam(i32_ty(), TermKind::IntLit("0".into()), ids),
        inputs: vec![arr_var(xs)],
        destination: SoacDestination::Fresh,
    }
}

/// `filter(λ_. true, <input>)`.
fn filter_term(input: ArrayExpr, ids: &mut TermIdSource) -> Term {
    term(
        TermKind::Soac(SoacOp::Filter {
            map_lam: None,
            pred: trivial_lam(bool_ty(), TermKind::BoolLit(true), ids),
            input,
            destination: SoacDestination::Fresh,
        }),
        arr_ty(),
        ids,
    )
}

fn prog(body: Term) -> Program {
    // The validator never reads `symbols` (it's a structural check), so an empty
    // table and a raw def id are fine.
    Program {
        defs: vec![Def {
            name: SymbolId::from(0),
            ty: arr_ty(),
            body,
            meta: DefMeta::Function,
            arity: 1,
            param_diets: vec![],
            return_diet: crate::types::Diet::observing(),
        }],
        symbols: SymbolTable::new(),
        def_syms: LookupMap::new(),
    }
}

#[test]
fn named_filter_input_is_anf() {
    // filter(p, ys) where ys is a bare name — the canonical ANF input.
    let mut ids = TermIdSource::new();
    let body = filter_term(arr_var(0), &mut ids);
    assert!(check(&prog(body)).is_ok());
}

// An inline producer in a SOAC input — `filter(p, map(f, xs))` — is
// unrepresentable: the `ArrayExpr` type admits only atoms, so the type itself
// rules out that position. The validator's job is the `Index`/`App` Term
// positions below, which the type cannot constrain.

#[test]
fn inline_producer_in_index_is_rejected() {
    // (map(f, xs))[i] — an inline producer in an `Index` array operand.
    let mut ids = TermIdSource::new();
    let map_t = term(TermKind::Soac(map_producer(0, &mut ids)), arr_ty(), &mut ids);
    let idx = term(TermKind::IntLit("0".into()), i32_ty(), &mut ids);
    let body = term(
        TermKind::Index {
            array: Box::new(map_t),
            index: Box::new(idx),
        },
        i32_ty(),
        &mut ids,
    );
    let err = check(&prog(body)).unwrap_err();
    assert!(err.contains("Index"), "{err}");
}

#[test]
fn let_bound_producer_then_named_consumer_is_anf() {
    // let ys = map(f, xs) in filter(p, ys) — producer in a binding position,
    // consumer over the name. No violation.
    let mut ids = TermIdSource::new();
    let rhs = term(TermKind::Soac(map_producer(0, &mut ids)), arr_ty(), &mut ids);
    let consumer = filter_term(arr_var(1), &mut ids);
    let body = term(
        TermKind::Let {
            name: SymbolId::from(1),
            name_ty: arr_ty(),
            rhs: Box::new(rhs),
            body: Box::new(consumer),
        },
        arr_ty(),
        &mut ids,
    );
    assert!(check(&prog(body)).is_ok());
}

// ---- Real-program checks (compile a source through the full TLC pipeline) ----

/// A single `map` over a parameter is already ANF — its input is a bare name.
/// Sanity that `check` passes on ordinary pipeline output.
#[test]
fn real_program_single_map_is_anf() {
    let reachable =
        crate::compile_thru_tlc("#[compute]\nentry e(xs: []i32) []i32 = map(|x: i32| x + 1, xs)\n")
            .expect("compile_thru_tlc");
    assert!(
        check(&reachable.tlc).is_ok(),
        "{}",
        check(&reachable.tlc).unwrap_err()
    );
}

/// Inline `filter(p, map(...))` leaves an inline producer in the filter's input
/// position. EXPECTED RED until ANF Phase 2 floats the producer into a `let`;
/// this assertion is the red checkpoint for that work.
#[test]
fn real_program_inline_filter_over_map_is_anf() {
    let reachable = crate::compile_thru_tlc(
        "open f32\n#[compute]\nentry e(xs: []u32) ?k. [k]u32 =\n  filter(|y: u32| y < 100u32, map(|x: u32| x + 1u32, xs))\n",
    )
    .expect("compile_thru_tlc");
    assert!(
        check(&reachable.tlc).is_ok(),
        "{}",
        check(&reachable.tlc).unwrap_err()
    );
}
