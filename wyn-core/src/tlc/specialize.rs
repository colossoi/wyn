//! Type-directed specialization of polymorphic intrinsic names.
//!
//! This is the first, private step of monomorphization. For example,
//! `sign(x)` at `f32` becomes the structural catalog reference `f32.sign`.

use super::data::Empty;
use super::soa::SoaNormalized;
use super::{Program, RewriteDecision, Term, TermId, TermIdSource, TermKind, TermRewriter, VarRef};
use crate::ast::TypeName;
use crate::builtins::catalog::KnownBuiltinIds;
use crate::builtins::{catalog, BuiltinId};
use crate::types::TypeExt;
use crate::SymbolTable;
use polytype::Type;

pub(super) fn run(program: &mut Program<SoaNormalized>) {
    let (defs, symbols, term_ids) = (&mut program.defs, &program.symbols, &mut program.term_ids);
    let mut specializer = IntrinsicSpecializer { symbols, term_ids };
    for def in defs {
        specializer.rewrite_tracked(&mut def.body);
    }
}

struct IntrinsicSpecializer<'symbols, 'ids> {
    symbols: &'symbols SymbolTable,
    term_ids: &'ids mut TermIdSource,
}

impl TermRewriter<Empty, Empty> for IntrinsicSpecializer<'_, '_> {
    fn next_term_id(&mut self) -> TermId {
        self.term_ids.next_id()
    }

    fn rewrite_node(&mut self, term: &mut Term<Empty, Empty>) -> RewriteDecision {
        let TermKind::App { func, args } = &mut term.kind else {
            return RewriteDecision::Unchanged;
        };
        let Some(first_arg) = args.first() else {
            return RewriteDecision::Unchanged;
        };

        // A Symbol is always a user or compiler binding and may shadow a
        // catalog name. Only structural builtin references specialize.
        let Some(id) = crate::tlc::var_term_builtin_id(func, self.symbols) else {
            return RewriteDecision::Unchanged;
        };
        let known = catalog().known();

        // Multiplication no longer needs overload selection after it becomes a
        // structural binary operator.
        if id == known.mul && args.len() == 2 {
            func.kind = TermKind::BinOp(crate::ast::BinaryOp { op: "*".to_string() });
            func.id = self.term_ids.next_id();
            return RewriteDecision::Changed;
        }

        let Some(specialized_name) = specialize_name(id, known, &first_arg.ty) else {
            return RewriteDecision::Unchanged;
        };
        let def = catalog()
            .lookup_by_surface_name(&specialized_name)
            .unwrap_or_else(|| panic!("BUG: specialize emitted name '{specialized_name}' not in catalog"));
        func.kind = TermKind::Var(VarRef::Builtin {
            id: def.id,
            overload_idx: 0,
        });
        func.id = self.term_ids.next_id();
        RewriteDecision::Changed
    }
}

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
    type_prefix(arg_ty).map(|prefix| format!("{prefix}.{base}"))
}

fn type_prefix(ty: &Type<TypeName>) -> Option<String> {
    let elem_ty = ty.elem_type().filter(|_| ty.is_vec()).unwrap_or(ty);
    match elem_ty {
        Type::Constructed(TypeName::Float(bits), _) => Some(format!("f{bits}")),
        Type::Constructed(TypeName::Int(bits), _) => Some(format!("i{bits}")),
        Type::Constructed(TypeName::UInt(bits), _) => Some(format!("u{bits}")),
        _ => None,
    }
}

#[cfg(test)]
#[path = "specialize_tests.rs"]
mod specialize_tests;
