//! Name resolution pass
//!
//! Resolves module-qualified names by rewriting:
//!   FieldAccess(Identifier(module), field) -> QualifiedName([module], field)
//! when `module` is a known module name.

use crate::ast::{Declaration, ExprKind, Expression, Program};
use crate::error::Result;
use crate::module_manager::ModuleManager;

/// Resolve names in a program by rewriting FieldAccess -> QualifiedName
pub fn resolve_program(program: &mut Program, module_manager: &ModuleManager) -> Result<()> {
    for decl in &mut program.declarations {
        resolve_declaration(decl, module_manager)?;
    }
    Ok(())
}

fn resolve_declaration(decl: &mut Declaration, module_manager: &ModuleManager) -> Result<()> {
    match decl {
        Declaration::Decl(d) => {
            resolve_expr(&mut d.body, module_manager)?;
        }
        Declaration::Entry(entry) => {
            resolve_expr(&mut entry.body, module_manager)?;
        }
        Declaration::Sig(_) => {
            // SigDecl has no body, only a type signature
        }
        _ => {}
    }
    Ok(())
}

fn resolve_expr(expr: &mut Expression, module_manager: &ModuleManager) -> Result<()> {
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
            resolve_expr(obj, module_manager)?;
        }
        ExprKind::Application(func, args) => {
            resolve_expr(func, module_manager)?;
            for arg in args {
                resolve_expr(arg, module_manager)?;
            }
        }
        ExprKind::Lambda(lambda) => {
            resolve_expr(&mut lambda.body, module_manager)?;
        }
        ExprKind::LetIn(let_in) => {
            resolve_expr(&mut let_in.value, module_manager)?;
            resolve_expr(&mut let_in.body, module_manager)?;
        }
        ExprKind::If(if_expr) => {
            resolve_expr(&mut if_expr.condition, module_manager)?;
            resolve_expr(&mut if_expr.then_branch, module_manager)?;
            resolve_expr(&mut if_expr.else_branch, module_manager)?;
        }
        ExprKind::BinaryOp(_, lhs, rhs) => {
            resolve_expr(lhs, module_manager)?;
            resolve_expr(rhs, module_manager)?;
        }
        ExprKind::UnaryOp(_, operand) => {
            resolve_expr(operand, module_manager)?;
        }
        ExprKind::Tuple(exprs) | ExprKind::ArrayLiteral(exprs) | ExprKind::VecMatLiteral(exprs) => {
            for e in exprs {
                resolve_expr(e, module_manager)?;
            }
        }
        ExprKind::ArrayIndex(arr, idx) => {
            resolve_expr(arr, module_manager)?;
            resolve_expr(idx, module_manager)?;
        }
        ExprKind::ArrayWith { array, index, value } => {
            resolve_expr(array, module_manager)?;
            resolve_expr(index, module_manager)?;
            resolve_expr(value, module_manager)?;
        }
        ExprKind::RecordLiteral(fields) => {
            for (_, e) in fields {
                resolve_expr(e, module_manager)?;
            }
        }
        ExprKind::Loop(loop_expr) => {
            if let Some(ref mut init) = loop_expr.init {
                resolve_expr(init, module_manager)?;
            }
            match &mut loop_expr.form {
                crate::ast::LoopForm::While(cond) => {
                    resolve_expr(cond, module_manager)?;
                }
                crate::ast::LoopForm::For(_, bound) => {
                    resolve_expr(bound, module_manager)?;
                }
                crate::ast::LoopForm::ForIn(_, iter) => {
                    resolve_expr(iter, module_manager)?;
                }
            }
            resolve_expr(&mut loop_expr.body, module_manager)?;
        }
        ExprKind::Match(match_expr) => {
            resolve_expr(&mut match_expr.scrutinee, module_manager)?;
            for case in &mut match_expr.cases {
                resolve_expr(&mut case.body, module_manager)?;
            }
        }
        ExprKind::TypeAscription(e, _) | ExprKind::TypeCoercion(e, _) => {
            resolve_expr(e, module_manager)?;
        }
        ExprKind::Assert(cond, body) => {
            resolve_expr(cond, module_manager)?;
            resolve_expr(body, module_manager)?;
        }
        ExprKind::Range(range) => {
            resolve_expr(&mut range.start, module_manager)?;
            resolve_expr(&mut range.end, module_manager)?;
            if let Some(ref mut step) = range.step {
                resolve_expr(step, module_manager)?;
            }
        }
        ExprKind::Slice(slice) => {
            resolve_expr(&mut slice.array, module_manager)?;
            if let Some(ref mut start) = slice.start {
                resolve_expr(start, module_manager)?;
            }
            if let Some(ref mut end) = slice.end {
                resolve_expr(end, module_manager)?;
            }
        }
        // Check unqualified identifiers against implicitly opened modules
        ExprKind::Identifier(quals, name) => {
            if quals.is_empty() {
                // Check if this name exists in an implicitly opened module
                if let Some(module_name) = module_manager.resolve_implicit_open(name) {
                    // Qualify the identifier with the module name
                    *quals = vec![module_name.to_string()];
                }
            }
        }
        // Base cases - no sub-expressions to resolve
        ExprKind::IntLiteral(_)
        | ExprKind::FloatLiteral(_)
        | ExprKind::BoolLiteral(_)
        | ExprKind::StringLiteral(_)
        | ExprKind::Unit
        | ExprKind::TypeHole => {}
    }
    Ok(())
}
