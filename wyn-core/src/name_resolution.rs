//! Name resolution pass
//!
//! Resolves module-qualified names by rewriting:
//!   `FieldAccess(Identifier(module), field)` → `Identifier([module], field)`
//! when `module` is a known module name.
//!
//! The same AST-walking machinery is reused by the module-elaboration path
//! (`module_manager::ModuleManager::resolve_names_in_expr`) — the walker is
//! generic over a `ResolveContext` that decides what a given identifier /
//! field-access means in the current mode. See `Resolver` below.

use crate::ast::{Declaration, ExprKind, Expression, Program};
use crate::error::Result;
use crate::module_manager::ModuleManager;
use crate::scope::{ScopeStack, for_each_pattern_name};

/// Insert every name bound by `pattern` into `scope`.
pub(crate) fn collect_pattern_bindings(pattern: &crate::ast::Pattern, scope: &mut ScopeStack<()>) {
    for_each_pattern_name(pattern, &mut |name| {
        scope.insert(name.to_string(), ());
    });
}

// ---------------------------------------------------------------------------
// Visitor — shared AST traversal, mode-specific identifier / field-access
// rewrites plugged in via `ResolveContext`.
// ---------------------------------------------------------------------------

/// Policy interface for a resolver pass. Consumers implement it to inject
/// their mode-specific identifier- and field-access- rewrite logic. The
/// walker handles everything else (recursion, scope push/pop, pattern
/// binding collection).
pub trait ResolveContext {
    /// Called for each `ExprKind::Identifier(quals, name)` leaf. May mutate
    /// `quals` / `name` in place (e.g. to qualify an intra-module ref). The
    /// `scope` argument reflects locals visible at this expression — used
    /// so intra-module rewrites don't shadow a lambda param of the same
    /// name.
    fn resolve_identifier(&self, _quals: &mut Vec<String>, _name: &mut String, _scope: &ScopeStack<()>) {}

    /// Called for each `ExprKind::FieldAccess(obj, field)` where `obj` is
    /// a plain `Identifier(obj_quals, obj_name)`. Return `Some(ExprKind)`
    /// to replace the entire FieldAccess expression (typical case:
    /// `mod.name` collapses to `Identifier([mod], name)`); return `None`
    /// to leave the FieldAccess alone — the walker will then recurse into
    /// `obj` as a regular expression.
    fn resolve_field_access(
        &self,
        _obj_quals: &[String],
        _obj_name: &str,
        _field: &str,
        _scope: &ScopeStack<()>,
    ) -> Option<ExprKind> {
        None
    }
}

/// Walk `expr` in place, applying `ctx`'s rewrite policies. `scope` is
/// the set of locally-bound names visible at `expr`. Callers typically
/// start with an empty `scope` at each declaration boundary.
pub fn walk_expr<C: ResolveContext>(
    expr: &mut Expression,
    ctx: &C,
    scope: &mut ScopeStack<()>,
) -> Result<()> {
    match &mut expr.kind {
        ExprKind::Identifier(quals, name) => {
            ctx.resolve_identifier(quals, name, scope);
        }
        ExprKind::FieldAccess(obj, field) => {
            // Peek at obj to see if it's the `Identifier . name` shape the
            // resolver hook wants to rewrite. If so, replace the whole
            // expression; otherwise fall through and recurse into obj.
            let rewrite = if let ExprKind::Identifier(obj_quals, obj_name) = &obj.kind {
                ctx.resolve_field_access(obj_quals, obj_name, field, scope)
            } else {
                None
            };
            if let Some(new_kind) = rewrite {
                expr.kind = new_kind;
            } else {
                walk_expr(obj, ctx, scope)?;
            }
        }
        ExprKind::Application(func, args) => {
            walk_expr(func, ctx, scope)?;
            for a in args {
                walk_expr(a, ctx, scope)?;
            }
        }
        ExprKind::Lambda(lambda) => {
            scope.push_scope();
            for p in &lambda.params {
                collect_pattern_bindings(p, scope);
            }
            walk_expr(&mut lambda.body, ctx, scope)?;
            scope.pop_scope();
        }
        ExprKind::LetIn(let_in) => {
            walk_expr(&mut let_in.value, ctx, scope)?;
            scope.push_scope();
            collect_pattern_bindings(&let_in.pattern, scope);
            walk_expr(&mut let_in.body, ctx, scope)?;
            scope.pop_scope();
        }
        ExprKind::If(if_expr) => {
            walk_expr(&mut if_expr.condition, ctx, scope)?;
            walk_expr(&mut if_expr.then_branch, ctx, scope)?;
            walk_expr(&mut if_expr.else_branch, ctx, scope)?;
        }
        ExprKind::BinaryOp(_, lhs, rhs) => {
            walk_expr(lhs, ctx, scope)?;
            walk_expr(rhs, ctx, scope)?;
        }
        ExprKind::UnaryOp(_, operand) => {
            walk_expr(operand, ctx, scope)?;
        }
        ExprKind::Tuple(exprs) | ExprKind::ArrayLiteral(exprs) | ExprKind::VecMatLiteral(exprs) => {
            for e in exprs {
                walk_expr(e, ctx, scope)?;
            }
        }
        ExprKind::ArrayIndex(arr, idx) => {
            walk_expr(arr, ctx, scope)?;
            walk_expr(idx, ctx, scope)?;
        }
        ExprKind::ArrayWith {
            array, index, value, ..
        } => {
            walk_expr(array, ctx, scope)?;
            walk_expr(index, ctx, scope)?;
            walk_expr(value, ctx, scope)?;
        }
        ExprKind::VecWith { target, value, .. } => {
            walk_expr(target, ctx, scope)?;
            walk_expr(value, ctx, scope)?;
        }
        ExprKind::RecordWith { record, value, .. } => {
            walk_expr(record, ctx, scope)?;
            walk_expr(value, ctx, scope)?;
        }
        ExprKind::RecordLiteral(fields) => {
            for (_, e) in fields {
                walk_expr(e, ctx, scope)?;
            }
        }
        ExprKind::Loop(loop_expr) => {
            scope.push_scope();
            collect_pattern_bindings(&loop_expr.pattern, scope);
            if let Some(ref mut init) = loop_expr.init {
                walk_expr(init, ctx, scope)?;
            }
            match &mut loop_expr.form {
                crate::ast::LoopForm::While(cond) => {
                    walk_expr(cond, ctx, scope)?;
                }
                crate::ast::LoopForm::For(idx_var, bound) => {
                    scope.insert(idx_var.clone(), ());
                    walk_expr(bound, ctx, scope)?;
                }
                crate::ast::LoopForm::ForIn(elem_pat, iter) => {
                    collect_pattern_bindings(elem_pat, scope);
                    walk_expr(iter, ctx, scope)?;
                }
            }
            walk_expr(&mut loop_expr.body, ctx, scope)?;
            scope.pop_scope();
        }
        ExprKind::Match(match_expr) => {
            walk_expr(&mut match_expr.scrutinee, ctx, scope)?;
            for case in &mut match_expr.cases {
                scope.push_scope();
                collect_pattern_bindings(&case.pattern, scope);
                walk_expr(&mut case.body, ctx, scope)?;
                scope.pop_scope();
            }
        }
        ExprKind::TypeAscription(e, _) | ExprKind::TypeCoercion(e, _) => {
            walk_expr(e, ctx, scope)?;
        }
        ExprKind::Range(range) => {
            walk_expr(&mut range.start, ctx, scope)?;
            walk_expr(&mut range.end, ctx, scope)?;
            if let Some(ref mut step) = range.step {
                walk_expr(step, ctx, scope)?;
            }
        }
        ExprKind::Slice(slice) => {
            walk_expr(&mut slice.array, ctx, scope)?;
            if let Some(ref mut start) = slice.start {
                walk_expr(start, ctx, scope)?;
            }
            if let Some(ref mut end) = slice.end {
                walk_expr(end, ctx, scope)?;
            }
        }
        ExprKind::Constructor(_, args) => {
            for a in args {
                walk_expr(a, ctx, scope)?;
            }
        }
        ExprKind::IntLiteral(_)
        | ExprKind::FloatLiteral(_)
        | ExprKind::BoolLiteral(_)
        | ExprKind::Unit
        | ExprKind::TypeHole => {}
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Program-mode resolver (top-level pass over user code)
// ---------------------------------------------------------------------------

/// Context used by the program-level name-resolution pass: rewrites
/// `mod.name` to `Identifier([mod], name)` when `mod` is a registered
/// module.
struct ProgramResolver<'a> {
    module_manager: &'a ModuleManager,
}

impl<'a> ResolveContext for ProgramResolver<'a> {
    fn resolve_field_access(
        &self,
        obj_quals: &[String],
        obj_name: &str,
        field: &str,
        _scope: &ScopeStack<()>,
    ) -> Option<ExprKind> {
        if obj_quals.is_empty() && self.module_manager.is_known_module(obj_name) {
            Some(ExprKind::Identifier(
                vec![obj_name.to_string()],
                field.to_string(),
            ))
        } else {
            None
        }
    }
}

/// Resolve names in a program by rewriting FieldAccess -> QualifiedName.
pub fn run(program: &mut Program, module_manager: &ModuleManager) -> Result<()> {
    for decl in &mut program.declarations {
        resolve_declaration(decl, module_manager)?;
    }
    Ok(())
}

/// Resolve names in a single Decl (for prelude functions).
pub fn resolve_decl(decl: &mut crate::ast::Decl, module_manager: &ModuleManager) -> Result<()> {
    let ctx = ProgramResolver { module_manager };
    let mut scope = ScopeStack::new();
    walk_expr(&mut decl.body, &ctx, &mut scope)
}

fn resolve_declaration(decl: &mut Declaration, module_manager: &ModuleManager) -> Result<()> {
    let ctx = ProgramResolver { module_manager };
    let mut scope = ScopeStack::new();
    match decl {
        Declaration::Decl(d) => walk_expr(&mut d.body, &ctx, &mut scope),
        Declaration::Entry(entry) => walk_expr(&mut entry.body, &ctx, &mut scope),
        Declaration::Sig(_) => Ok(()),
        _ => Ok(()),
    }
}
