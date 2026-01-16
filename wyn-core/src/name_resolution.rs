//! Name resolution pass
//!
//! Resolves module-qualified names by rewriting:
//!   FieldAccess(Identifier(module), field) -> QualifiedName([module], field)
//! when `module` is a known module name.
//!
//! Also resolves builtin aliases (e.g., magnitude -> _w_intrinsic_magnitude)
//! for identifiers that are not shadowed by local bindings.

use crate::ast::{Declaration, ExprKind, Expression, PatternKind, Program};
use crate::error::Result;
use crate::module_manager::ModuleManager;
use crate::scope::ScopeStack;

/// Collect bound names from a pattern into the scope
fn collect_pattern_bindings(pattern: &crate::ast::Pattern, scope: &mut ScopeStack<()>) {
    match &pattern.kind {
        PatternKind::Name(name) => {
            scope.insert(name.clone(), ());
        }
        PatternKind::Tuple(patterns) | PatternKind::Constructor(_, patterns) => {
            for p in patterns {
                collect_pattern_bindings(p, scope);
            }
        }
        PatternKind::Record(fields) => {
            for field in fields {
                if let Some(ref pat) = field.pattern {
                    collect_pattern_bindings(pat, scope);
                } else {
                    // Shorthand: { name } binds `name`
                    scope.insert(field.field.clone(), ());
                }
            }
        }
        PatternKind::Attributed(_, inner) | PatternKind::Typed(inner, _) => {
            collect_pattern_bindings(inner, scope);
        }
        PatternKind::Wildcard | PatternKind::Unit | PatternKind::Literal(_) => {}
    }
}

/// Resolve names in a program by rewriting FieldAccess -> QualifiedName
pub fn resolve_program(program: &mut Program, module_manager: &ModuleManager) -> Result<()> {
    for decl in &mut program.declarations {
        resolve_declaration(decl, module_manager)?;
    }
    Ok(())
}

fn resolve_declaration(decl: &mut Declaration, module_manager: &ModuleManager) -> Result<()> {
    let mut scope = ScopeStack::new();
    match decl {
        Declaration::Decl(d) => {
            resolve_expr(&mut d.body, module_manager, &mut scope)?;
        }
        Declaration::Entry(entry) => {
            resolve_expr(&mut entry.body, module_manager, &mut scope)?;
        }
        Declaration::Sig(_) => {
            // SigDecl has no body, only a type signature
        }
        _ => {}
    }
    Ok(())
}

fn resolve_expr(
    expr: &mut Expression,
    module_manager: &ModuleManager,
    scope: &mut ScopeStack<()>,
) -> Result<()> {
    match &mut expr.kind {
        ExprKind::FieldAccess(obj, field) => {
            // Check if this is module.name pattern
            if let ExprKind::Identifier(quals, name) = &obj.kind {
                if quals.is_empty() && module_manager.is_known_module(name) {
                    // Build the qualified name
                    let module = name.clone();
                    let func_name = field.clone();

                    // Rewrite to qualified Identifier
                    expr.kind = ExprKind::Identifier(vec![module.clone()], func_name);
                    return Ok(());
                }
            }
            // Otherwise, it's a real field access - recurse into object
            resolve_expr(obj, module_manager, scope)?;
        }
        ExprKind::Application(func, args) => {
            resolve_expr(func, module_manager, scope)?;
            for arg in args {
                resolve_expr(arg, module_manager, scope)?;
            }
        }
        ExprKind::Lambda(lambda) => {
            // Lambda params create a new scope
            scope.push_scope();
            for param in &lambda.params {
                collect_pattern_bindings(param, scope);
            }
            resolve_expr(&mut lambda.body, module_manager, scope)?;
            scope.pop_scope();
        }
        ExprKind::LetIn(let_in) => {
            // Resolve value in current scope
            resolve_expr(&mut let_in.value, module_manager, scope)?;
            // Add binding then resolve body
            scope.push_scope();
            collect_pattern_bindings(&let_in.pattern, scope);
            resolve_expr(&mut let_in.body, module_manager, scope)?;
            scope.pop_scope();
        }
        ExprKind::If(if_expr) => {
            resolve_expr(&mut if_expr.condition, module_manager, scope)?;
            resolve_expr(&mut if_expr.then_branch, module_manager, scope)?;
            resolve_expr(&mut if_expr.else_branch, module_manager, scope)?;
        }
        ExprKind::BinaryOp(_, lhs, rhs) => {
            resolve_expr(lhs, module_manager, scope)?;
            resolve_expr(rhs, module_manager, scope)?;
        }
        ExprKind::UnaryOp(_, operand) => {
            resolve_expr(operand, module_manager, scope)?;
        }
        ExprKind::Tuple(exprs) | ExprKind::ArrayLiteral(exprs) | ExprKind::VecMatLiteral(exprs) => {
            for e in exprs {
                resolve_expr(e, module_manager, scope)?;
            }
        }
        ExprKind::ArrayIndex(arr, idx) => {
            resolve_expr(arr, module_manager, scope)?;
            resolve_expr(idx, module_manager, scope)?;
        }
        ExprKind::ArrayWith { array, index, value } => {
            resolve_expr(array, module_manager, scope)?;
            resolve_expr(index, module_manager, scope)?;
            resolve_expr(value, module_manager, scope)?;
        }
        ExprKind::RecordLiteral(fields) => {
            for (_, e) in fields {
                resolve_expr(e, module_manager, scope)?;
            }
        }
        ExprKind::Loop(loop_expr) => {
            scope.push_scope();
            // Loop accumulator variable binding
            collect_pattern_bindings(&loop_expr.pattern, scope);
            if let Some(ref mut init) = loop_expr.init {
                resolve_expr(init, module_manager, scope)?;
            }
            match &mut loop_expr.form {
                crate::ast::LoopForm::While(cond) => {
                    resolve_expr(cond, module_manager, scope)?;
                }
                crate::ast::LoopForm::For(idx_var, bound) => {
                    scope.insert(idx_var.clone(), ());
                    resolve_expr(bound, module_manager, scope)?;
                }
                crate::ast::LoopForm::ForIn(elem_pat, iter) => {
                    collect_pattern_bindings(elem_pat, scope);
                    resolve_expr(iter, module_manager, scope)?;
                }
            }
            resolve_expr(&mut loop_expr.body, module_manager, scope)?;
            scope.pop_scope();
        }
        ExprKind::Match(match_expr) => {
            resolve_expr(&mut match_expr.scrutinee, module_manager, scope)?;
            for case in &mut match_expr.cases {
                scope.push_scope();
                collect_pattern_bindings(&case.pattern, scope);
                resolve_expr(&mut case.body, module_manager, scope)?;
                scope.pop_scope();
            }
        }
        ExprKind::TypeAscription(e, _) | ExprKind::TypeCoercion(e, _) => {
            resolve_expr(e, module_manager, scope)?;
        }
        ExprKind::Range(range) => {
            resolve_expr(&mut range.start, module_manager, scope)?;
            resolve_expr(&mut range.end, module_manager, scope)?;
            if let Some(ref mut step) = range.step {
                resolve_expr(step, module_manager, scope)?;
            }
        }
        ExprKind::Slice(slice) => {
            resolve_expr(&mut slice.array, module_manager, scope)?;
            if let Some(ref mut start) = slice.start {
                resolve_expr(start, module_manager, scope)?;
            }
            if let Some(ref mut end) = slice.end {
                resolve_expr(end, module_manager, scope)?;
            }
        }
        // Resolve builtin aliases (e.g., magnitude -> _w_intrinsic_magnitude)
        // Only resolve if the name is NOT shadowed by a local binding
        ExprKind::Identifier(quals, name) if quals.is_empty() && scope.lookup(name).is_none() => {
            use crate::intrinsics::IntrinsicSource;
            if let Some(resolved) = IntrinsicSource::resolve_alias(name) {
                *name = resolved.to_string();
            }
        }
        ExprKind::Identifier(_, _) => {}
        ExprKind::IntLiteral(_)
        | ExprKind::FloatLiteral(_)
        | ExprKind::BoolLiteral(_)
        | ExprKind::StringLiteral(_)
        | ExprKind::Unit
        | ExprKind::TypeHole => {}
    }
    Ok(())
}
