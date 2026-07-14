//! A-Normal-Form validator for array producers.
//!
//! Asserts the TLC invariant that **every array a SOAC consumes is named, never
//! an inline producer**: a SOAC's array inputs are *atoms* (`Var`,
//! `Range`, `Literal`, `Zip` of atoms), and no `Index` array
//! operand or `App` argument is a bare inline SOAC. Producers live only in
//! binding positions — a `let` rhs or a def/lambda tail
//! value.
//!
//! This is a `debug_assert`-only invariant: in a correct compiler it never
//! fires, so it carries no symbol names or rich diagnostics — a terse static
//! reason is enough to point at the offending construct. The `ArrayExpr` type
//! makes inline producers unrepresentable in SOAC-*input* positions; this
//! checker guards the `Term`-typed positions the type cannot reach (`Index`
//! operand, `App` argument), where `runtime_index_producers` / `normalize` do
//! the floating.

use super::{Program, Term, TermKind};
use crate::ast::TypeName;
use polytype::Type;

/// Verify the array-producer ANF invariant for every def body. Returns the
/// first violating construct, or `Ok(())`.
pub fn check(program: &Program) -> Result<(), &'static str> {
    for def in &program.defs {
        walk(&def.body)?;
    }
    Ok(())
}

/// `debug_assert` the ANF invariant at a pass boundary. A no-op in release;
/// in debug/test builds it fires the moment a pass leaves an inline producer in
/// an `Index`/`App` Term position. Call at the end of every `run()` from
/// `normalize`/`runtime_index_producers` onward (the passes that establish and
/// must preserve the floated form). `stage` names the pass for the panic.
pub fn debug_check(program: &Program, stage: &'static str) {
    debug_assert!(
        check(program).is_ok(),
        "anf::check failed after {}: {}",
        stage,
        check(program).err().unwrap_or("")
    );
}

fn walk(term: &Term) -> Result<(), &'static str> {
    match &term.kind {
        TermKind::Index { array, .. } if is_inline_producer(array) => {
            return Err("Index array operand is an inline producer (not ANF)");
        }
        TermKind::App { args, .. } if args.iter().any(is_inline_producer) => {
            return Err("App argument is an inline array producer (not ANF)");
        }
        _ => {}
    }

    // A producer in a binding position (let rhs or tail) is
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

/// A `Term` in an `Index`/`App` operand position that is a bare inline *array*
/// producer (which `runtime_index_producers` / `normalize` float into a `let`).
/// A SOAC *input* is an atom by type, so only these `Term`-typed positions —
/// which the `ArrayExpr` type cannot constrain — need a runtime guard.
///
/// The array-type gate matters for `App` args: a scalar-producing `Reduce`
/// (`f32.sum(xs) / n` lowers to `(/) (reduce …) n`) is a perfectly fine scalar
/// operand, not an unmaterialized array, and the builtin→SOAC lowering that
/// creates it runs after the last `normalize`. Only an *array* producer in a
/// `Term` operand position is the thing the backend can't place. An `Index`
/// array operand is array-typed by construction, so this gate is a no-op there.
fn is_inline_producer(t: &Term) -> bool {
    matches!(&t.kind, TermKind::Soac(_)) && matches!(&t.ty, Type::Constructed(TypeName::Array, _))
}

#[cfg(test)]
#[path = "anf_tests.rs"]
mod anf_tests;
