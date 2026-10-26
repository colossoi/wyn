//! Type-directed specialization of polymorphic intrinsic names.
//!
//! Called during monomorphization after type substitution. For example,
//! `sign(x)` at `f32` becomes the structural catalog reference `f32.sign`.

use super::data::Empty;
use super::{RewriteDecision, Term, TermIdSource, TermKind, VarRef};
use crate::ast;
use crate::builtins::catalog;
use crate::op;
use crate::types::TypeExt;
use polytype::Type;

pub(super) fn specialize_intrinsic(
    term: &mut Term<Empty, Empty>,
    term_ids: &mut TermIdSource,
) -> RewriteDecision {
    let TermKind::App { func, args } = &mut term.kind else {
        return RewriteDecision::Unchanged;
    };
    let Some(first_arg) = args.first() else {
        return RewriteDecision::Unchanged;
    };

    // A Symbol is always a user or compiler binding and may shadow a
    // catalog name. Only structural builtin references specialize.
    let TermKind::Var(VarRef::Builtin { id, overload_idx }) = &func.kind else {
        return RewriteDecision::Unchanged;
    };
    let known = catalog().known();

    // A scalar specialization has no record of which arguments require
    // broadcasting. Keep the overload; backends resolve its numeric opcode.
    if matches!(
        catalog().get(*id).overloads()[*overload_idx].lowering,
        crate::builtins::BuiltinLowering::ExtInstSplat { .. }
    ) {
        return RewriteDecision::Unchanged;
    }

    // Multiplication becomes a structural binary operator and needs no
    // overload-bearing callee.
    if *id == known.mul && args.len() == 2 {
        func.kind = TermKind::BinOp(ast::BinaryOp {
            op: op::BinaryOperator::Multiply,
        });
        func.id = term_ids.next_id();
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
    func.id = term_ids.next_id();
    RewriteDecision::Changed
}

#[cfg(test)]
#[path = "specialize_tests.rs"]
mod specialize_tests;
