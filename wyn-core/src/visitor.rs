//! AST visitor pattern for traversing the Wyn syntax tree
//!
//! This module provides a centralized traversal mechanism for the AST.
//! Each pass (type checking, defunctionalization, etc.) can implement the
//! Visitor trait and override only the hooks they need, while the walk_*
//! functions handle the actual tree traversal.

use crate::NodeId;
use crate::ast::*;
use std::ops::ControlFlow;

/// Visitor trait for traversing the AST
///
/// All methods have default implementations that delegate to walk_* functions.
/// Implementors can override specific hooks to customize behavior.
///
/// The Break associated type allows visitors to return errors or other data
/// when they need to short-circuit traversal.
pub trait Visitor: Sized {
    type Break;

    // --- Top-level program ---
    fn visit_program(&mut self, p: &Program) -> ControlFlow<Self::Break> {
        walk_program(self, p)
    }

    fn visit_declaration(&mut self, d: &Declaration) -> ControlFlow<Self::Break> {
        walk_declaration(self, d)
    }

    // --- Declarations ---
    fn visit_decl(&mut self, d: &Decl) -> ControlFlow<Self::Break> {
        walk_decl(self, d)
    }

    fn visit_entry_decl(&mut self, e: &EntryDecl) -> ControlFlow<Self::Break> {
        walk_entry_decl(self, e)
    }

    fn visit_uniform_decl(&mut self, u: &UniformDecl) -> ControlFlow<Self::Break> {
        walk_uniform_decl(self, u)
    }

    fn visit_storage_decl(&mut self, s: &StorageDecl) -> ControlFlow<Self::Break> {
        walk_storage_decl(self, s)
    }

    fn visit_sig_decl(&mut self, v: &SigDecl) -> ControlFlow<Self::Break> {
        walk_sig_decl(self, v)
    }

    // --- Expressions ---
    fn visit_expression(&mut self, e: &Expression) -> ControlFlow<Self::Break> {
        walk_expression(self, e)
    }

    fn visit_expr_int_literal(&mut self, _id: NodeId, _n: i32) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }

    fn visit_expr_float_literal(&mut self, _id: NodeId, _f: f32) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }

    fn visit_expr_bool_literal(&mut self, _id: NodeId, _b: bool) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }

    fn visit_expr_identifier(
        &mut self,
        _id: NodeId,
        _quals: &[String],
        _name: &str,
    ) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }

    fn visit_expr_array_literal(
        &mut self,
        _id: NodeId,
        elements: &[Expression],
    ) -> ControlFlow<Self::Break> {
        walk_expr_array_literal(self, elements)
    }

    fn visit_expr_array_index(
        &mut self,
        _id: NodeId,
        array: &Expression,
        index: &Expression,
    ) -> ControlFlow<Self::Break> {
        walk_expr_array_index(self, array, index)
    }

    fn visit_expr_binary_op(
        &mut self,
        _id: NodeId,
        _op: &BinaryOp,
        left: &Expression,
        right: &Expression,
    ) -> ControlFlow<Self::Break> {
        walk_expr_binary_op(self, left, right)
    }

    fn visit_expr_function_call(
        &mut self,
        _id: NodeId,
        _name: &str,
        args: &[Expression],
    ) -> ControlFlow<Self::Break> {
        walk_expr_function_call(self, args)
    }

    fn visit_expr_tuple(&mut self, _id: NodeId, elements: &[Expression]) -> ControlFlow<Self::Break> {
        walk_expr_tuple(self, elements)
    }

    fn visit_expr_lambda(&mut self, _id: NodeId, lambda: &LambdaExpr) -> ControlFlow<Self::Break> {
        walk_expr_lambda(self, lambda)
    }

    fn visit_expr_application(
        &mut self,
        _id: NodeId,
        func: &Expression,
        args: &[Expression],
    ) -> ControlFlow<Self::Break> {
        walk_expr_application(self, func, args)
    }

    fn visit_expr_let_in(&mut self, _id: NodeId, let_in: &LetInExpr) -> ControlFlow<Self::Break> {
        walk_expr_let_in(self, let_in)
    }

    fn visit_expr_field_access(
        &mut self,
        _id: NodeId,
        expr: &Expression,
        _field: &str,
    ) -> ControlFlow<Self::Break> {
        walk_expr_field_access(self, expr)
    }

    fn visit_expr_if(&mut self, _id: NodeId, if_expr: &IfExpr) -> ControlFlow<Self::Break> {
        walk_expr_if(self, if_expr)
    }

    fn visit_expr_slice(&mut self, _id: NodeId, slice: &SliceExpr) -> ControlFlow<Self::Break> {
        walk_expr_slice(self, slice)
    }

    fn visit_expr_range(&mut self, _id: NodeId, range: &RangeExpr) -> ControlFlow<Self::Break> {
        walk_expr_range(self, range)
    }

    // --- Types ---
    fn visit_type(&mut self, _t: &Type) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }

    fn visit_pattern(&mut self, p: &Pattern) -> ControlFlow<Self::Break> {
        walk_pattern(self, p)
    }

    fn visit_pattern_name(&mut self, _id: NodeId, _name: &str) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }

    fn visit_pattern_wildcard(&mut self, _id: NodeId) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }

    fn visit_pattern_unit(&mut self, _id: NodeId) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }

    fn visit_pattern_literal(&mut self, _id: NodeId, _lit: &PatternLiteral) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }

    fn visit_pattern_tuple(&mut self, _id: NodeId, patterns: &[Pattern]) -> ControlFlow<Self::Break> {
        walk_pattern_tuple(self, patterns)
    }

    fn visit_pattern_record(
        &mut self,
        _id: NodeId,
        fields: &[RecordPatternField],
    ) -> ControlFlow<Self::Break> {
        walk_pattern_record(self, fields)
    }

    fn visit_pattern_constructor(
        &mut self,
        _id: NodeId,
        _name: &str,
        patterns: &[Pattern],
    ) -> ControlFlow<Self::Break> {
        walk_pattern_constructor(self, patterns)
    }

    fn visit_pattern_typed(&mut self, _id: NodeId, inner: &Pattern, ty: &Type) -> ControlFlow<Self::Break> {
        walk_pattern_typed(self, inner, ty)
    }

    fn visit_pattern_attributed(
        &mut self,
        _id: NodeId,
        _attrs: &[Attribute],
        inner: &Pattern,
    ) -> ControlFlow<Self::Break> {
        walk_pattern_attributed(self, inner)
    }

    fn visit_parameter(&mut self, p: &Parameter) -> ControlFlow<Self::Break> {
        walk_parameter(self, p)
    }
}

// --- Walk functions: canonical traversal ---

pub fn walk_program<V: Visitor>(v: &mut V, p: &Program) -> ControlFlow<V::Break> {
    for decl in &p.declarations {
        v.visit_declaration(decl)?;
    }
    ControlFlow::Continue(())
}

pub fn walk_declaration<V: Visitor>(v: &mut V, d: &Declaration) -> ControlFlow<V::Break> {
    match d {
        Declaration::Decl(decl) => v.visit_decl(decl),
        Declaration::Entry(entry) => v.visit_entry_decl(entry),
        Declaration::Uniform(uniform) => v.visit_uniform_decl(uniform),
        Declaration::Storage(storage) => v.visit_storage_decl(storage),
        Declaration::Sig(sig) => v.visit_sig_decl(sig),
        Declaration::TypeBind(_) => {
            unimplemented!("Type bindings are not yet supported in visitor")
        }
        Declaration::ModuleBind(_) => {
            unimplemented!("Module bindings are not yet supported in visitor")
        }
        Declaration::ModuleTypeBind(_) => {
            unimplemented!("Module type bindings are not yet supported in visitor")
        }
        Declaration::Open(_) => {
            unimplemented!("Open declarations are not yet supported in visitor")
        }
        Declaration::Import(_) => {
            unimplemented!("Import declarations are not yet supported in visitor")
        }
    }
}

pub fn walk_decl<V: Visitor>(v: &mut V, d: &Decl) -> ControlFlow<V::Break> {
    // Visit parameters (visit patterns)
    for param in &d.params {
        v.visit_pattern(param)?;
    }

    // Visit type annotation if present
    if let Some(ty) = &d.ty {
        v.visit_type(ty)?;
    }

    // Visit body
    v.visit_expression(&d.body)
}

pub fn walk_entry_decl<V: Visitor>(v: &mut V, e: &EntryDecl) -> ControlFlow<V::Break> {
    // Visit parameters
    for param in &e.params {
        v.visit_pattern(param)?;
    }

    // Visit output types
    for output in &e.outputs {
        v.visit_type(&output.ty)?;
    }

    // Visit body
    v.visit_expression(&e.body)
}

pub fn walk_uniform_decl<V: Visitor>(v: &mut V, u: &UniformDecl) -> ControlFlow<V::Break> {
    v.visit_type(&u.ty)
}

pub fn walk_storage_decl<V: Visitor>(v: &mut V, s: &StorageDecl) -> ControlFlow<V::Break> {
    v.visit_type(&s.ty)
}

pub fn walk_sig_decl<V: Visitor>(v: &mut V, sig: &SigDecl) -> ControlFlow<V::Break> {
    v.visit_type(&sig.ty)
}

pub fn walk_parameter<V: Visitor>(v: &mut V, p: &Parameter) -> ControlFlow<V::Break> {
    v.visit_type(&p.ty)
}

pub fn walk_pattern<V: Visitor>(v: &mut V, pat: &Pattern) -> ControlFlow<V::Break> {
    let id = pat.h.id;
    match &pat.kind {
        PatternKind::Name(name) => v.visit_pattern_name(id, name),
        PatternKind::Wildcard => v.visit_pattern_wildcard(id),
        PatternKind::Unit => v.visit_pattern_unit(id),
        PatternKind::Literal(lit) => v.visit_pattern_literal(id, lit),
        PatternKind::Tuple(patterns) => v.visit_pattern_tuple(id, patterns),
        PatternKind::Record(fields) => v.visit_pattern_record(id, fields),
        PatternKind::Constructor(name, patterns) => v.visit_pattern_constructor(id, name, patterns),
        PatternKind::Typed(inner, ty) => v.visit_pattern_typed(id, inner, ty),
        PatternKind::Attributed(attrs, inner) => v.visit_pattern_attributed(id, attrs, inner),
    }
}

pub fn walk_pattern_tuple<V: Visitor>(v: &mut V, patterns: &[Pattern]) -> ControlFlow<V::Break> {
    for p in patterns {
        v.visit_pattern(p)?;
    }
    ControlFlow::Continue(())
}

pub fn walk_pattern_record<V: Visitor>(v: &mut V, fields: &[RecordPatternField]) -> ControlFlow<V::Break> {
    for field in fields {
        if let Some(p) = &field.pattern {
            v.visit_pattern(p)?;
        }
    }
    ControlFlow::Continue(())
}

pub fn walk_pattern_constructor<V: Visitor>(v: &mut V, patterns: &[Pattern]) -> ControlFlow<V::Break> {
    for p in patterns {
        v.visit_pattern(p)?;
    }
    ControlFlow::Continue(())
}

pub fn walk_pattern_typed<V: Visitor>(v: &mut V, inner: &Pattern, ty: &Type) -> ControlFlow<V::Break> {
    v.visit_pattern(inner)?;
    v.visit_type(ty)
}

pub fn walk_pattern_attributed<V: Visitor>(v: &mut V, inner: &Pattern) -> ControlFlow<V::Break> {
    v.visit_pattern(inner)
}

pub fn walk_expression<V: Visitor>(v: &mut V, e: &Expression) -> ControlFlow<V::Break> {
    let id = e.h.id;
    match &e.kind {
        ExprKind::RecordLiteral(fields) => {
            for (_name, field_expr) in fields {
                walk_expression(v, field_expr)?;
            }
            ControlFlow::Continue(())
        }
        ExprKind::IntLiteral(n) => v.visit_expr_int_literal(id, *n),
        ExprKind::FloatLiteral(f) => v.visit_expr_float_literal(id, *f),
        ExprKind::BoolLiteral(b) => v.visit_expr_bool_literal(id, *b),
        ExprKind::StringLiteral(_) => ControlFlow::Continue(()),
        ExprKind::Unit => ControlFlow::Continue(()),
        ExprKind::Identifier(quals, name) => v.visit_expr_identifier(id, quals, name),
        ExprKind::ArrayLiteral(elements) => v.visit_expr_array_literal(id, elements),
        ExprKind::ArrayIndex(array, index) => v.visit_expr_array_index(id, array, index),
        ExprKind::ArrayWith { array, index, value } => {
            v.visit_expression(array)?;
            v.visit_expression(index)?;
            v.visit_expression(value)
        }
        ExprKind::BinaryOp(op, left, right) => v.visit_expr_binary_op(id, op, left, right),
        ExprKind::Tuple(elements) => v.visit_expr_tuple(id, elements),
        ExprKind::Lambda(lambda) => v.visit_expr_lambda(id, lambda),
        ExprKind::Application(func, args) => v.visit_expr_application(id, func, args),
        ExprKind::LetIn(let_in) => v.visit_expr_let_in(id, let_in),
        ExprKind::FieldAccess(expr, field) => v.visit_expr_field_access(id, expr, field),
        ExprKind::If(if_expr) => v.visit_expr_if(id, if_expr),

        ExprKind::TypeHole => ControlFlow::Continue(()),
        ExprKind::UnaryOp(_, operand) => v.visit_expression(operand),
        ExprKind::Loop(loop_expr) => {
            // Visit the loop condition/iterator
            match &loop_expr.form {
                LoopForm::For(_, iterator) => v.visit_expression(iterator)?,
                LoopForm::ForIn(pattern, iterator) => {
                    v.visit_pattern(pattern)?;
                    v.visit_expression(iterator)?
                }
                LoopForm::While(condition) => v.visit_expression(condition)?,
            }
            // Visit the loop body
            v.visit_expression(&loop_expr.body)
        }
        ExprKind::Match(match_expr) => {
            v.visit_expression(&match_expr.scrutinee)?;
            for case in &match_expr.cases {
                v.visit_pattern(&case.pattern)?;
                v.visit_expression(&case.body)?;
            }
            ControlFlow::Continue(())
        }
        ExprKind::Range(range_expr) => v.visit_expr_range(id, range_expr),
        ExprKind::Slice(slice_expr) => v.visit_expr_slice(id, slice_expr),
        ExprKind::TypeAscription(expr, _) => v.visit_expression(expr),
        ExprKind::TypeCoercion(expr, _) => v.visit_expression(expr),
        ExprKind::Assert(cond, expr) => {
            v.visit_expression(cond)?;
            v.visit_expression(expr)
        }
        ExprKind::VecMatLiteral(elements) => {
            for elem in elements {
                v.visit_expression(elem)?;
            }
            ControlFlow::Continue(())
        }
    } // NEWCASESHERE - add new cases before this closing brace
}

pub fn walk_expr_array_literal<V: Visitor>(v: &mut V, elements: &[Expression]) -> ControlFlow<V::Break> {
    for elem in elements {
        v.visit_expression(elem)?;
    }
    ControlFlow::Continue(())
}

pub fn walk_expr_array_index<V: Visitor>(
    v: &mut V,
    array: &Expression,
    index: &Expression,
) -> ControlFlow<V::Break> {
    v.visit_expression(array)?;
    v.visit_expression(index)
}

pub fn walk_expr_binary_op<V: Visitor>(
    v: &mut V,
    left: &Expression,
    right: &Expression,
) -> ControlFlow<V::Break> {
    v.visit_expression(left)?;
    v.visit_expression(right)
}

pub fn walk_expr_function_call<V: Visitor>(v: &mut V, args: &[Expression]) -> ControlFlow<V::Break> {
    for arg in args {
        v.visit_expression(arg)?;
    }
    ControlFlow::Continue(())
}

pub fn walk_expr_tuple<V: Visitor>(v: &mut V, elements: &[Expression]) -> ControlFlow<V::Break> {
    for elem in elements {
        v.visit_expression(elem)?;
    }
    ControlFlow::Continue(())
}

pub fn walk_expr_lambda<V: Visitor>(v: &mut V, lambda: &LambdaExpr) -> ControlFlow<V::Break> {
    // Visit parameter patterns
    for param in &lambda.params {
        v.visit_pattern(param)?;
    }

    // Visit return type
    if let Some(ty) = &lambda.return_type {
        v.visit_type(ty)?;
    }

    // Visit body
    v.visit_expression(&lambda.body)
}

pub fn walk_expr_application<V: Visitor>(
    v: &mut V,
    func: &Expression,
    args: &[Expression],
) -> ControlFlow<V::Break> {
    v.visit_expression(func)?;
    for arg in args {
        v.visit_expression(arg)?;
    }
    ControlFlow::Continue(())
}

pub fn walk_expr_let_in<V: Visitor>(v: &mut V, let_in: &LetInExpr) -> ControlFlow<V::Break> {
    // Visit type annotation if present
    if let Some(ty) = &let_in.ty {
        v.visit_type(ty)?;
    }

    // Visit value
    v.visit_expression(&let_in.value)?;

    // Visit body
    v.visit_expression(&let_in.body)
}

pub fn walk_expr_field_access<V: Visitor>(v: &mut V, expr: &Expression) -> ControlFlow<V::Break> {
    v.visit_expression(expr)
}

pub fn walk_expr_if<V: Visitor>(v: &mut V, if_expr: &IfExpr) -> ControlFlow<V::Break> {
    v.visit_expression(&if_expr.condition)?;
    v.visit_expression(&if_expr.then_branch)?;
    v.visit_expression(&if_expr.else_branch)
}

pub fn walk_expr_slice<V: Visitor>(v: &mut V, slice: &SliceExpr) -> ControlFlow<V::Break> {
    v.visit_expression(&slice.array)?;
    if let Some(start) = &slice.start {
        v.visit_expression(start)?;
    }
    if let Some(end) = &slice.end {
        v.visit_expression(end)?;
    }
    if let Some(step) = &slice.step {
        v.visit_expression(step)?;
    }
    ControlFlow::Continue(())
}

pub fn walk_expr_range<V: Visitor>(v: &mut V, range: &RangeExpr) -> ControlFlow<V::Break> {
    v.visit_expression(&range.start)?;
    if let Some(step) = &range.step {
        v.visit_expression(step)?;
    }
    v.visit_expression(&range.end)
}
