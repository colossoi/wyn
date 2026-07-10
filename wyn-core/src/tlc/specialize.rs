//! Specialization pass for TLC.
//!
//! Specializes polymorphic intrinsic names based on argument types.
//! For example: `sign(x)` where `x: f32` becomes `f32.sign(x)`.

use super::VarRef;
use super::{Program, Term, TermIdSource, TermKind};
use crate::ast::TypeName;
use crate::builtins::catalog::KnownBuiltinIds;
use crate::builtins::{catalog, BuiltinId};
use crate::types::TypeExt;
use crate::SymbolTable;
use polytype::Type;

/// Specialize polymorphic intrinsics in a TLC program.
pub fn run(program: &mut Program, term_ids: &mut TermIdSource) {
    for def in &mut program.defs {
        specialize_term(&mut def.body, &mut program.symbols, term_ids);
    }
}

/// Bottom-up: recurse into children, then specialize App nodes in place.
fn specialize_term(term: &mut Term, symbols: &mut SymbolTable, term_ids: &mut TermIdSource) {
    term.for_each_child_mut(&mut |child| specialize_term(child, symbols, term_ids));

    // Only App nodes need specialization.
    let TermKind::App { func, args } = &mut term.kind else {
        return;
    };

    if args.is_empty() {
        return;
    }

    // Only catalog references specialize. A `Var(Symbol)` is always a
    // user binding (or compiler-synthesized fresh symbol) and must
    // pass through unchanged — user shadows of catalog names like
    // `let mul = ...` would otherwise be silently rewritten.
    let Some(id) = crate::tlc::var_term_builtin_id(func, symbols) else {
        return;
    };
    let known = catalog().known();

    // `mul` → `BinOp("*")` rewrite (works for both scalar and
    // vec/matrix overloads — overload selection at the catalog
    // level becomes irrelevant once it's a BinOp).
    if id == known.mul && args.len() == 2 {
        let binop = Term {
            id: term_ids.next_id(),
            ty: func.ty.clone(),
            span: func.span,
            kind: TermKind::BinOp(crate::ast::BinaryOp { op: "*".to_string() }),
        };
        **func = binop;
        return;
    }

    if let Some(specialized_name) = specialize_name(id, known, &args[0].ty) {
        // Per-type ops (`f32.abs`, `i32.min`, …) are 4 prefixes × 6
        // ops = 24 catalog entries, dynamically picked from the arg
        // type. Cheap enough to keep a per-call surface-name lookup
        // for now; the lookup result is a `BuiltinId` from a
        // catalog-known entry, so it still flows structurally.
        let def = catalog().lookup_by_surface_name(&specialized_name).unwrap_or_else(|| {
            panic!(
                "BUG: specialize emitted name '{}' not in catalog",
                specialized_name
            )
        });
        let new_func = Term {
            id: term_ids.next_id(),
            ty: func.ty.clone(),
            span: func.span,
            kind: TermKind::Var(VarRef::Builtin {
                id: def.id,
                overload_idx: 0,
            }),
        };
        **func = new_func;
    }
}

/// Specialize a polymorphic-op catalog reference into a per-type
/// surface name. Only fires for catalog ops in the specialize set
/// (`abs`, `sign`, `min`, `max`, `clamp`); everything else returns
/// `None` and passes through unchanged.
fn specialize_name(id: BuiltinId, known: &KnownBuiltinIds, arg_ty: &Type<TypeName>) -> Option<String> {
    let base = if id == known.abs {
        "abs"
    } else if id == known.sign {
        "sign"
    } else if id == known.min {
        "min"
    } else if id == known.max {
        "max"
    } else if id == known.clamp {
        "clamp"
    } else {
        return None;
    };
    type_prefix(arg_ty).map(|prefix| format!("{}.{}", prefix, base))
}

/// Get the type prefix for specialization (f32, i32, u32, etc.)
fn type_prefix(ty: &Type<TypeName>) -> Option<String> {
    let elem_ty = ty.elem_type().filter(|_| ty.is_vec()).unwrap_or(ty);
    match elem_ty {
        Type::Constructed(TypeName::Float(bits), _) => Some(format!("f{}", bits)),
        Type::Constructed(TypeName::Int(bits), _) => Some(format!("i{}", bits)),
        Type::Constructed(TypeName::UInt(bits), _) => Some(format!("u{}", bits)),
        _ => None,
    }
}

#[cfg(test)]
#[path = "specialize_tests.rs"]
mod specialize_tests;
