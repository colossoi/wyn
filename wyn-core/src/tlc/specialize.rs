//! Specialization pass for TLC.
//!
//! Specializes polymorphic intrinsic names based on argument types.
//! For example: `sign(x)` where `x: f32` becomes `f32.sign(x)`.

use super::{Def, Program, Term, TermIdSource, TermKind};
use crate::SymbolTable;
use crate::ast::TypeName;
use crate::types::TypeExt;
use polytype::Type;

/// Specialize polymorphic intrinsics in a TLC program.
pub fn run(program: Program) -> Program {
    let mut symbols = program.symbols;
    let mut term_ids = TermIdSource::new();

    let defs = program
        .defs
        .into_iter()
        .map(|def| {
            let body = specialize_term(def.body, &mut symbols, &mut term_ids);
            Def { body, ..def }
        })
        .collect();

    Program {
        defs,
        uniforms: program.uniforms,
        storage: program.storage,
        symbols,
        def_syms: program.def_syms,
    }
}

/// Bottom-up: recurse into children, then specialize App nodes.
fn specialize_term(term: Term, symbols: &mut SymbolTable, term_ids: &mut TermIdSource) -> Term {
    let term = term.map_children(&mut |child| specialize_term(child, symbols, term_ids));

    // Only App nodes need specialization.
    let TermKind::App { ref func, ref args } = term.kind else {
        return term;
    };

    if args.is_empty() {
        return term;
    }

    // Resolve the func's canonical name regardless of whether it's a
    // `Var(Symbol)` or `Var(Builtin)` — `var_term_canonical_name`
    // returns the catalog's surface_name for builtins and the
    // symbol's name for user-defined symbols.
    let Some(name) = crate::tlc::var_term_canonical_name(func, symbols) else {
        return term;
    };

    // `mul` → `BinOp("*")` rewrite (works for both scalar and
    // vec/matrix overloads — overload selection at the catalog
    // level becomes irrelevant once it's a BinOp).
    if name == "mul" && args.len() == 2 {
        let binop = Term {
            id: term_ids.next_id(),
            ty: func.ty.clone(),
            span: func.span,
            kind: TermKind::BinOp(crate::ast::BinaryOp { op: "*".to_string() }),
        };
        let TermKind::App { func: _, args } = term.kind else {
            unreachable!()
        };
        return Term {
            id: term_ids.next_id(),
            kind: TermKind::App {
                func: Box::new(binop),
                args,
            },
            ..term
        };
    }

    if let Some(specialized_name) = specialize_name(name, &args[0].ty) {
        let specialized_sym = symbols.alloc(specialized_name);
        let new_func = Term {
            id: term_ids.next_id(),
            ty: func.ty.clone(),
            span: func.span,
            kind: TermKind::Var(crate::tlc::VarRef::Symbol(specialized_sym)),
        };
        let TermKind::App { func: _, args } = term.kind else {
            unreachable!()
        };
        Term {
            id: term_ids.next_id(),
            kind: TermKind::App {
                func: Box::new(new_func),
                args,
            },
            ..term
        }
    } else {
        term
    }
}

/// Specialize a function name based on argument type.
fn specialize_name(name: &str, arg_ty: &Type<TypeName>) -> Option<String> {
    match name {
        "abs" | "sign" | "min" | "max" | "clamp" => {
            type_prefix(arg_ty).map(|prefix| format!("{}.{}", prefix, name))
        }
        _ => None,
    }
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
