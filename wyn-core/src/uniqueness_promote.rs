//! Uniqueness-based promotion of `ExprKind::ArrayWith` to in-place mutation.
//!
//! Consumes the alias checker's `liveness` map. For every `a with [i] = v`
//! node where the source array `a` is the last use and alias-free, we flip
//! `inplace = true`. The TLC transformer then routes the node to
//! `_w_intrinsic_array_with_inplace` instead of `_w_intrinsic_array_with`,
//! and the backends emit mutation instead of copy + patch.
//!
//! The promotion is a hint, always semantics-preserving: if we fail to set
//! the flag when we could have, the output is correct but does a redundant
//! copy. If we set the flag when we shouldn't have (bug), the alias
//! checker's contract is violated — but the checker already rejects those
//! programs, so reaching this pass guarantees the info is sound.
//!
//! Only mutates the AST; returns nothing.
//!
//! Example:
//! ```text
//! let b = a with [0] = 42 in ... b ...   -- last use of `a`: promote
//! let b = a with [0] = 42 in (a, b)      -- `a` used after: leave functional
//! ```
//!
//! This pass runs after alias-checking and before TLC transformation.

#[cfg(test)]
#[path = "uniqueness_promote_tests.rs"]
mod uniqueness_promote_tests;

use crate::alias_checker::AliasCheckResult;
use crate::ast::{Decl, Declaration, ExprKind, Expression, Program};
use crate::interface::EntryDecl;

/// Flip `inplace = true` on every `ExprKind::ArrayWith` whose source array
/// was proved dead (released + alias-free) by the alias checker.
pub fn run(program: &mut Program, alias: &AliasCheckResult) {
    let mut promoter = Promoter { alias };
    for decl in &mut program.declarations {
        promoter.visit_declaration(decl);
    }
}

struct Promoter<'a> {
    alias: &'a AliasCheckResult,
}

impl<'a> Promoter<'a> {
    fn visit_declaration(&mut self, decl: &mut Declaration) {
        match decl {
            Declaration::Decl(d) => self.visit_decl(d),
            Declaration::Entry(e) => self.visit_entry_decl(e),
            _ => {}
        }
    }

    fn visit_decl(&mut self, d: &mut Decl) {
        self.visit_expr(&mut d.body);
    }

    fn visit_entry_decl(&mut self, e: &mut EntryDecl) {
        self.visit_expr(&mut e.body);
    }

    fn visit_expr(&mut self, expr: &mut Expression) {
        match &mut expr.kind {
            ExprKind::ArrayWith {
                array,
                index,
                value,
                inplace,
            } => {
                // Recurse first so nested ArrayWiths are considered.
                self.visit_expr(array);
                self.visit_expr(index);
                self.visit_expr(value);

                if let Some(liveness) = self.alias.liveness.get(&array.h.id) {
                    if liveness.released && liveness.alias_free {
                        *inplace = true;
                    }
                }
            }
            ExprKind::VecWith { target, value, .. } => {
                // Vec swizzle update is purely functional — we always
                // build a fresh vec, so there's no in-place flag to
                // promote. Just recurse.
                self.visit_expr(target);
                self.visit_expr(value);
            }
            ExprKind::ArrayLiteral(elements) | ExprKind::VecMatLiteral(elements) => {
                for elem in elements {
                    self.visit_expr(elem);
                }
            }
            ExprKind::ArrayIndex(array, index) => {
                self.visit_expr(array);
                self.visit_expr(index);
            }
            ExprKind::BinaryOp(_, lhs, rhs) => {
                self.visit_expr(lhs);
                self.visit_expr(rhs);
            }
            ExprKind::UnaryOp(_, operand) => self.visit_expr(operand),
            ExprKind::Tuple(elements) => {
                for elem in elements {
                    self.visit_expr(elem);
                }
            }
            ExprKind::Constructor(_, args) => {
                for arg in args {
                    self.visit_expr(arg);
                }
            }
            ExprKind::RecordLiteral(fields) => {
                for (_, v) in fields {
                    self.visit_expr(v);
                }
            }
            ExprKind::Lambda(lambda) => self.visit_expr(&mut lambda.body),
            ExprKind::Application(func, args) => {
                self.visit_expr(func);
                for a in args {
                    self.visit_expr(a);
                }
            }
            ExprKind::LetIn(let_in) => {
                self.visit_expr(&mut let_in.value);
                self.visit_expr(&mut let_in.body);
            }
            ExprKind::FieldAccess(obj, _) => self.visit_expr(obj),
            ExprKind::If(if_expr) => {
                self.visit_expr(&mut if_expr.condition);
                self.visit_expr(&mut if_expr.then_branch);
                self.visit_expr(&mut if_expr.else_branch);
            }
            ExprKind::Loop(loop_expr) => {
                if let Some(init) = &mut loop_expr.init {
                    self.visit_expr(init);
                }
                self.visit_expr(&mut loop_expr.body);
            }
            ExprKind::Match(match_expr) => {
                self.visit_expr(&mut match_expr.scrutinee);
                for case in &mut match_expr.cases {
                    self.visit_expr(&mut case.body);
                }
            }
            ExprKind::Range(range) => {
                self.visit_expr(&mut range.start);
                if let Some(step) = &mut range.step {
                    self.visit_expr(step);
                }
                self.visit_expr(&mut range.end);
            }
            ExprKind::Slice(slice) => {
                self.visit_expr(&mut slice.array);
                if let Some(start) = &mut slice.start {
                    self.visit_expr(start);
                }
                if let Some(end) = &mut slice.end {
                    self.visit_expr(end);
                }
            }
            ExprKind::TypeAscription(inner, _) | ExprKind::TypeCoercion(inner, _) => {
                self.visit_expr(inner);
            }
            ExprKind::IntLiteral(_)
            | ExprKind::FloatLiteral(_)
            | ExprKind::BoolLiteral(_)
            | ExprKind::Unit
            | ExprKind::Identifier(_, _)
            | ExprKind::TypeHole => {}
        }
    }
}
