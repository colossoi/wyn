//! A-Normal-Form validator for array producers.
//!
//! Asserts the TLC invariant that **every array a SOAC consumes is named, never
//! an inline producer**: a SOAC's array inputs are *atoms* (`Ref(Var)`,
//! `StorageView`, `Range`, `Literal`, `Zip` of atoms), and no `Index` array
//! operand or `App` argument is a bare inline SOAC. Producers live only in
//! binding positions — a `let` rhs, a def/lambda tail, an `OutputSlotStore`
//! value.
//!
//! This is a `debug_assert`-only invariant: in a correct compiler it never
//! fires, so it carries no symbol names or rich diagnostics — a terse static
//! reason is enough to point at the offending construct. The `ArrayExpr` type
//! makes inline producers unrepresentable in SOAC-*input* positions; this
//! checker guards the `Term`-typed positions the type cannot reach (`Index`
//! operand, `App` argument), where `runtime_index_producers` / `normalize` do
//! the floating.

use super::{ArrayExpr, Program, SoacOp, Term, TermKind};

/// Verify the array-producer ANF invariant for every def body. Returns the
/// first violating construct, or `Ok(())`.
pub fn check(program: &Program) -> Result<(), &'static str> {
    for def in &program.defs {
        walk(&def.body)?;
    }
    Ok(())
}

fn walk(term: &Term) -> Result<(), &'static str> {
    match &term.kind {
        TermKind::Soac(soac) => check_soac_inputs(soac)?,
        TermKind::ArrayExpr(ArrayExpr::Soac(soac)) => check_soac_inputs(soac)?,
        TermKind::Index { array, .. } if is_inline_producer(array) => {
            return Err("Index array operand is an inline producer (not ANF)");
        }
        TermKind::App { args, .. } if args.iter().any(is_inline_producer) => {
            return Err("App argument is an inline producer (not ANF)");
        }
        _ => {}
    }

    // A producer in a binding position (let rhs, tail, OutputSlotStore value) is
    // fine; the checks above only fire on non-binding positions, so recursing
    // never double-flags a legal producer.
    let mut result = Ok(());
    term.for_each_child(&mut |child| {
        if result.is_ok() {
            result = walk(child);
        }
    });
    result
}

/// Every array input of `soac` must be an atom.
fn check_soac_inputs(soac: &SoacOp) -> Result<(), &'static str> {
    if soac_inputs(soac).into_iter().any(|ae| !is_atom(ae)) {
        Err("SOAC array input is an inline producer (not ANF)")
    } else {
        Ok(())
    }
}

/// The array-input `ArrayExpr`s of a SOAC. `Scatter`'s destination is a `Place`,
/// not an array input, so it is excluded.
fn soac_inputs(soac: &SoacOp) -> Vec<&ArrayExpr> {
    match soac {
        SoacOp::Map { inputs, .. } | SoacOp::Screma { inputs, .. } | SoacOp::Scatter { inputs, .. } => {
            inputs.iter().collect()
        }
        SoacOp::Reduce { input, .. } | SoacOp::Scan { input, .. } | SoacOp::Filter { input, .. } => {
            vec![input]
        }
        SoacOp::ReduceByIndex { indices, values, .. } => vec![indices, values],
    }
}

/// An atom is a named reference or a leaf view — never an inline producer or a
/// compound term. `Zip` is an atom iff all its children are.
fn is_atom(ae: &ArrayExpr) -> bool {
    match ae {
        ArrayExpr::Ref(t) => matches!(t.kind, TermKind::Var(_)),
        ArrayExpr::StorageView(_) | ArrayExpr::Range { .. } | ArrayExpr::Literal(_) => true,
        ArrayExpr::Zip(children) => children.iter().all(is_atom),
        ArrayExpr::Soac(_) => false,
    }
}

/// A `Term` in an `Index`/`App` operand position that is a bare inline SOAC
/// producer (which `runtime_index_producers` / `normalize` should have floated).
fn is_inline_producer(t: &Term) -> bool {
    matches!(
        &t.kind,
        TermKind::Soac(_) | TermKind::ArrayExpr(ArrayExpr::Soac(_))
    )
}

#[cfg(test)]
#[path = "anf_tests.rs"]
mod anf_tests;
