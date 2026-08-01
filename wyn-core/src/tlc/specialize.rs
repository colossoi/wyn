//! Type-directed specialization of polymorphic intrinsic names.
//!
//! This is the first, private step of monomorphization. For example,
//! `sign(x)` at `f32` becomes the structural catalog reference `f32.sign`.

use super::data::Empty;
use super::soa::SoaNormalized;
use super::{RewriteDecision, Term, TermId, TermIdSource, TermKind, TermRewriter, VarRef};
use crate::builtins::catalog;
use crate::types::TypeExt;
use polytype::Type;

pub(super) fn specialize_intrinsics(program: &mut SoaNormalized) {
    let (defs, term_ids) = (&mut program.defs, &mut program.term_ids);
    let mut specializer = IntrinsicSpecializer { term_ids };
    for def in defs {
        specializer.rewrite_tracked(&mut def.body);
    }
}

struct IntrinsicSpecializer<'ids> {
    term_ids: &'ids mut TermIdSource,
}

impl TermRewriter<Empty, Empty> for IntrinsicSpecializer<'_> {
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
        let TermKind::Var(VarRef::Builtin { id, .. }) = &func.kind else {
            return RewriteDecision::Unchanged;
        };
        let known = catalog().known();

        // Multiplication becomes a structural binary operator and needs no
        // overload-bearing callee.
        if *id == known.mul && args.len() == 2 {
            func.kind = TermKind::BinOp(crate::ast::BinaryOp {
                op: crate::op::BinaryOperator::Multiply,
            });
            func.id = self.term_ids.next_id();
            return RewriteDecision::Changed;
        }

        let scalar_ty = first_arg.ty.elem_type().filter(|_| first_arg.ty.is_vec()).unwrap_or(&first_arg.ty);
        let Type::Constructed(scalar, type_args) = scalar_ty else {
            return RewriteDecision::Unchanged;
        };
        if !type_args.is_empty() {
            return RewriteDecision::Unchanged;
        }
        let Some(specialized) = catalog().specialize_numeric(*id, scalar) else {
            return RewriteDecision::Unchanged;
        };
        func.kind = TermKind::Var(VarRef::Builtin {
            id: specialized,
            overload_idx: 0,
        });
        func.id = self.term_ids.next_id();
        RewriteDecision::Changed
    }
}

#[cfg(test)]
#[path = "specialize_tests.rs"]
mod specialize_tests;
