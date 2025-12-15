//! MIR visitor pattern for traversing and transforming the Wyn MIR.
//!
//! This module provides a centralized traversal mechanism for the MIR.
//! Each pass (optimization, lowering, analysis, etc.) can implement the
//! `MirVisitor` trait and override only the hooks they need, while the
//! `walk_*` functions handle the actual tree traversal.
//!
//! The visitor is a tree-mutating visitor: it consumes and returns values.
//! Read-only passes can just return the input unchanged.

use crate::ast::{NodeId, Span, TypeName};
use crate::mir::*;
use polytype::Type;

/// Visitor trait for traversing and transforming the MIR.
///
/// All methods have default implementations that delegate to `walk_*` functions.
/// Implementors can override specific hooks to customize behavior.
///
/// The visitor consumes and returns values, making it suitable for both:
/// - Transformation passes (monomorphization, constant folding, etc.)
/// - Read-only passes (just return the input unchanged)
///
/// The `Error` associated type allows visitors to propagate errors.
/// The `Ctx` associated type provides per-branch context state.
pub trait MirFolder: Sized {
    type Error;
    /// Context type for per-branch state. Use `()` if not needed.
    type Ctx;

    // --- Top-level program ---

    fn visit_program(&mut self, p: Program, ctx: &mut Self::Ctx) -> Result<Program, Self::Error> {
        walk_program(self, p, ctx)
    }

    fn visit_def(&mut self, d: Def, ctx: &mut Self::Ctx) -> Result<Def, Self::Error> {
        walk_def(self, d, ctx)
    }

    fn visit_function(
        &mut self,
        id: NodeId,
        name: String,
        params: Vec<Param>,
        ret_type: Type<TypeName>,
        attributes: Vec<Attribute>,
        body: Expr,
        span: Span,
        ctx: &mut Self::Ctx,
    ) -> Result<Def, Self::Error> {
        walk_function(self, id, name, params, ret_type, attributes, body, span, ctx)
    }

    fn visit_constant(
        &mut self,
        id: NodeId,
        name: String,
        ty: Type<TypeName>,
        attributes: Vec<Attribute>,
        body: Expr,
        span: Span,
        ctx: &mut Self::Ctx,
    ) -> Result<Def, Self::Error> {
        walk_constant(self, id, name, ty, attributes, body, span, ctx)
    }

    fn visit_uniform(
        &mut self,
        id: NodeId,
        name: String,
        ty: Type<TypeName>,
        set: u32,
        binding: u32,
        ctx: &mut Self::Ctx,
    ) -> Result<Def, Self::Error> {
        walk_uniform(self, id, name, ty, set, binding, ctx)
    }

    fn visit_storage(
        &mut self,
        id: NodeId,
        name: String,
        ty: Type<TypeName>,
        set: u32,
        binding: u32,
        layout: crate::ast::StorageLayout,
        access: crate::ast::StorageAccess,
        ctx: &mut Self::Ctx,
    ) -> Result<Def, Self::Error> {
        walk_storage(self, id, name, ty, set, binding, layout, access, ctx)
    }

    fn visit_entry_point(
        &mut self,
        id: NodeId,
        name: String,
        execution_model: ExecutionModel,
        inputs: Vec<EntryInput>,
        outputs: Vec<EntryOutput>,
        body: Expr,
        span: Span,
        ctx: &mut Self::Ctx,
    ) -> Result<Def, Self::Error> {
        walk_entry_point(self, id, name, execution_model, inputs, outputs, body, span, ctx)
    }

    fn visit_param(&mut self, p: Param, ctx: &mut Self::Ctx) -> Result<Param, Self::Error> {
        walk_param(self, p, ctx)
    }

    fn visit_attribute(&mut self, a: Attribute, ctx: &mut Self::Ctx) -> Result<Attribute, Self::Error> {
        walk_attribute(self, a, ctx)
    }

    fn visit_type(
        &mut self,
        ty: Type<TypeName>,
        _ctx: &mut Self::Ctx,
    ) -> Result<Type<TypeName>, Self::Error> {
        Ok(ty)
    }

    // --- Expressions ---

    fn visit_expr(&mut self, e: Expr, ctx: &mut Self::Ctx) -> Result<Expr, Self::Error> {
        walk_expr(self, e, ctx)
    }

    fn visit_expr_literal(
        &mut self,
        lit: Literal,
        expr: Expr,
        ctx: &mut Self::Ctx,
    ) -> Result<Expr, Self::Error> {
        let lit = walk_literal(self, lit, ctx)?;
        Ok(Expr {
            kind: ExprKind::Literal(lit),
            ..expr
        })
    }

    fn visit_expr_var(
        &mut self,
        _name: String,
        expr: Expr,
        _ctx: &mut Self::Ctx,
    ) -> Result<Expr, Self::Error> {
        Ok(expr)
    }

    fn visit_expr_bin_op(
        &mut self,
        op: String,
        lhs: Expr,
        rhs: Expr,
        expr: Expr,
        ctx: &mut Self::Ctx,
    ) -> Result<Expr, Self::Error> {
        walk_expr_bin_op(self, op, lhs, rhs, expr, ctx)
    }

    fn visit_expr_unary_op(
        &mut self,
        op: String,
        operand: Expr,
        expr: Expr,
        ctx: &mut Self::Ctx,
    ) -> Result<Expr, Self::Error> {
        walk_expr_unary_op(self, op, operand, expr, ctx)
    }

    fn visit_expr_if(
        &mut self,
        cond: Expr,
        then_branch: Expr,
        else_branch: Expr,
        expr: Expr,
        ctx: &mut Self::Ctx,
    ) -> Result<Expr, Self::Error> {
        walk_expr_if(self, cond, then_branch, else_branch, expr, ctx)
    }

    fn visit_expr_let(
        &mut self,
        name: String,
        binding_id: u64,
        value: Expr,
        body: Expr,
        expr: Expr,
        ctx: &mut Self::Ctx,
    ) -> Result<Expr, Self::Error> {
        walk_expr_let(self, name, binding_id, value, body, expr, ctx)
    }

    fn visit_expr_loop(
        &mut self,
        loop_var: String,
        init: Expr,
        init_bindings: Vec<(String, Expr)>,
        kind: LoopKind,
        body: Expr,
        expr: Expr,
        ctx: &mut Self::Ctx,
    ) -> Result<Expr, Self::Error> {
        walk_expr_loop(self, loop_var, init, init_bindings, kind, body, expr, ctx)
    }

    fn visit_expr_call(
        &mut self,
        func: String,
        args: Vec<Expr>,
        expr: Expr,
        ctx: &mut Self::Ctx,
    ) -> Result<Expr, Self::Error> {
        walk_expr_call(self, func, args, expr, ctx)
    }

    fn visit_expr_intrinsic(
        &mut self,
        name: String,
        args: Vec<Expr>,
        expr: Expr,
        ctx: &mut Self::Ctx,
    ) -> Result<Expr, Self::Error> {
        walk_expr_intrinsic(self, name, args, expr, ctx)
    }

    fn visit_expr_attributed(
        &mut self,
        attributes: Vec<Attribute>,
        inner: Expr,
        expr: Expr,
        ctx: &mut Self::Ctx,
    ) -> Result<Expr, Self::Error> {
        walk_expr_attributed(self, attributes, inner, expr, ctx)
    }

    fn visit_expr_materialize(
        &mut self,
        inner: Expr,
        expr: Expr,
        ctx: &mut Self::Ctx,
    ) -> Result<Expr, Self::Error> {
        walk_expr_materialize(self, inner, expr, ctx)
    }

    fn visit_expr_closure(
        &mut self,
        lambda_name: String,
        captures: Vec<Expr>,
        expr: Expr,
        ctx: &mut Self::Ctx,
    ) -> Result<Expr, Self::Error> {
        walk_expr_closure(self, lambda_name, captures, expr, ctx)
    }

    fn visit_expr_range(
        &mut self,
        start: Expr,
        step: Option<Expr>,
        end: Expr,
        kind: RangeKind,
        expr: Expr,
        ctx: &mut Self::Ctx,
    ) -> Result<Expr, Self::Error> {
        walk_expr_range(self, start, step, end, kind, expr, ctx)
    }

    // --- Literals ---

    fn visit_literal_int(&mut self, value: String, _ctx: &mut Self::Ctx) -> Result<Literal, Self::Error> {
        Ok(Literal::Int(value))
    }

    fn visit_literal_float(&mut self, value: String, _ctx: &mut Self::Ctx) -> Result<Literal, Self::Error> {
        Ok(Literal::Float(value))
    }

    fn visit_literal_bool(&mut self, value: bool, _ctx: &mut Self::Ctx) -> Result<Literal, Self::Error> {
        Ok(Literal::Bool(value))
    }

    fn visit_literal_string(
        &mut self,
        value: String,
        _ctx: &mut Self::Ctx,
    ) -> Result<Literal, Self::Error> {
        Ok(Literal::String(value))
    }

    fn visit_literal_tuple(
        &mut self,
        elements: Vec<Expr>,
        ctx: &mut Self::Ctx,
    ) -> Result<Literal, Self::Error> {
        walk_literal_tuple(self, elements, ctx)
    }

    fn visit_literal_array(
        &mut self,
        elements: Vec<Expr>,
        ctx: &mut Self::Ctx,
    ) -> Result<Literal, Self::Error> {
        walk_literal_array(self, elements, ctx)
    }

    // --- Loops ---

    fn visit_loop_kind(&mut self, kind: LoopKind, ctx: &mut Self::Ctx) -> Result<LoopKind, Self::Error> {
        walk_loop_kind(self, kind, ctx)
    }

    fn visit_for_loop(
        &mut self,
        var: String,
        iter: Expr,
        ctx: &mut Self::Ctx,
    ) -> Result<LoopKind, Self::Error> {
        walk_for_loop(self, var, iter, ctx)
    }

    fn visit_for_range_loop(
        &mut self,
        var: String,
        bound: Expr,
        ctx: &mut Self::Ctx,
    ) -> Result<LoopKind, Self::Error> {
        walk_for_range_loop(self, var, bound, ctx)
    }

    fn visit_while_loop(&mut self, cond: Expr, ctx: &mut Self::Ctx) -> Result<LoopKind, Self::Error> {
        walk_while_loop(self, cond, ctx)
    }
}

// --- Walk functions: canonical traversal ---

pub fn walk_program<V: MirFolder>(v: &mut V, p: Program, ctx: &mut V::Ctx) -> Result<Program, V::Error> {
    let Program {
        defs,
        lambda_registry,
    } = p;

    let defs = defs.into_iter().map(|d| v.visit_def(d, ctx)).collect::<Result<Vec<_>, _>>()?;

    Ok(Program {
        defs,
        lambda_registry,
    })
}

pub fn walk_def<V: MirFolder>(v: &mut V, d: Def, ctx: &mut V::Ctx) -> Result<Def, V::Error> {
    match d {
        Def::Function {
            id,
            name,
            params,
            ret_type,
            attributes,
            body,
            span,
        } => v.visit_function(id, name, params, ret_type, attributes, body, span, ctx),
        Def::Constant {
            id,
            name,
            ty,
            attributes,
            body,
            span,
        } => v.visit_constant(id, name, ty, attributes, body, span, ctx),
        Def::Uniform {
            id,
            name,
            ty,
            set,
            binding,
        } => v.visit_uniform(id, name, ty, set, binding, ctx),
        Def::Storage {
            id,
            name,
            ty,
            set,
            binding,
            layout,
            access,
        } => v.visit_storage(id, name, ty, set, binding, layout, access, ctx),
        Def::EntryPoint {
            id,
            name,
            execution_model,
            inputs,
            outputs,
            body,
            span,
        } => v.visit_entry_point(id, name, execution_model, inputs, outputs, body, span, ctx),
    }
}

pub fn walk_function<V: MirFolder>(
    v: &mut V,
    id: NodeId,
    name: String,
    params: Vec<Param>,
    ret_type: Type<TypeName>,
    attributes: Vec<Attribute>,
    body: Expr,
    span: Span,
    ctx: &mut V::Ctx,
) -> Result<Def, V::Error> {
    let params = params.into_iter().map(|p| v.visit_param(p, ctx)).collect::<Result<Vec<_>, _>>()?;
    let ret_type = v.visit_type(ret_type, ctx)?;
    let attributes =
        attributes.into_iter().map(|a| v.visit_attribute(a, ctx)).collect::<Result<Vec<_>, _>>()?;
    let body = v.visit_expr(body, ctx)?;

    Ok(Def::Function {
        id,
        name,
        params,
        ret_type,
        attributes,
        body,
        span,
    })
}

pub fn walk_constant<V: MirFolder>(
    v: &mut V,
    id: NodeId,
    name: String,
    ty: Type<TypeName>,
    attributes: Vec<Attribute>,
    body: Expr,
    span: Span,
    ctx: &mut V::Ctx,
) -> Result<Def, V::Error> {
    let ty = v.visit_type(ty, ctx)?;

    let attributes =
        attributes.into_iter().map(|a| v.visit_attribute(a, ctx)).collect::<Result<Vec<_>, _>>()?;

    let body = v.visit_expr(body, ctx)?;

    Ok(Def::Constant {
        id,
        name,
        ty,
        attributes,
        body,
        span,
    })
}

pub fn walk_uniform<V: MirFolder>(
    v: &mut V,
    id: NodeId,
    name: String,
    ty: Type<TypeName>,
    set: u32,
    binding: u32,
    ctx: &mut V::Ctx,
) -> Result<Def, V::Error> {
    let ty = v.visit_type(ty, ctx)?;
    Ok(Def::Uniform {
        id,
        name,
        ty,
        set,
        binding,
    })
}

pub fn walk_storage<V: MirFolder>(
    v: &mut V,
    id: NodeId,
    name: String,
    ty: Type<TypeName>,
    set: u32,
    binding: u32,
    layout: crate::ast::StorageLayout,
    access: crate::ast::StorageAccess,
    ctx: &mut V::Ctx,
) -> Result<Def, V::Error> {
    let ty = v.visit_type(ty, ctx)?;
    Ok(Def::Storage {
        id,
        name,
        ty,
        set,
        binding,
        layout,
        access,
    })
}

pub fn walk_entry_point<V: MirFolder>(
    v: &mut V,
    id: NodeId,
    name: String,
    execution_model: ExecutionModel,
    inputs: Vec<EntryInput>,
    outputs: Vec<EntryOutput>,
    body: Expr,
    span: Span,
    ctx: &mut V::Ctx,
) -> Result<Def, V::Error> {
    // Visit types in inputs
    let inputs = inputs
        .into_iter()
        .map(|input| {
            let ty = v.visit_type(input.ty, ctx)?;
            Ok(EntryInput {
                name: input.name,
                ty,
                decoration: input.decoration,
            })
        })
        .collect::<Result<Vec<_>, V::Error>>()?;

    // Visit types in outputs
    let outputs = outputs
        .into_iter()
        .map(|output| {
            let ty = v.visit_type(output.ty, ctx)?;
            Ok(EntryOutput {
                ty,
                decoration: output.decoration,
            })
        })
        .collect::<Result<Vec<_>, V::Error>>()?;

    // Visit body
    let body = v.visit_expr(body, ctx)?;

    Ok(Def::EntryPoint {
        id,
        name,
        execution_model,
        inputs,
        outputs,
        body,
        span,
    })
}

pub fn walk_param<V: MirFolder>(v: &mut V, p: Param, ctx: &mut V::Ctx) -> Result<Param, V::Error> {
    let Param { name, ty } = p;
    let ty = v.visit_type(ty, ctx)?;
    Ok(Param { name, ty })
}

pub fn walk_attribute<V: MirFolder>(
    _v: &mut V,
    a: Attribute,
    _ctx: &mut V::Ctx,
) -> Result<Attribute, V::Error> {
    Ok(a)
}

// --- Expressions ---

pub fn walk_expr<V: MirFolder>(v: &mut V, e: Expr, ctx: &mut V::Ctx) -> Result<Expr, V::Error> {
    let Expr { id, ty, kind, span } = e;
    let ty = v.visit_type(ty, ctx)?;

    match kind {
        ExprKind::Literal(lit) => {
            let expr = Expr {
                id,
                ty,
                kind: ExprKind::Literal(lit.clone()),
                span,
            };
            v.visit_expr_literal(lit, expr, ctx)
        }
        ExprKind::Unit => Ok(Expr {
            id,
            ty,
            kind: ExprKind::Unit,
            span,
        }),
        ExprKind::Var(ref name) => {
            let expr = Expr {
                id,
                ty,
                kind: ExprKind::Var(name.clone()),
                span,
            };
            v.visit_expr_var(name.clone(), expr, ctx)
        }
        ExprKind::BinOp { op, lhs, rhs } => {
            let expr = Expr {
                id,
                ty,
                kind: ExprKind::Var(String::new()), // Dummy kind, won't be used
                span,
            };
            v.visit_expr_bin_op(op, *lhs, *rhs, expr, ctx)
        }
        ExprKind::UnaryOp { op, operand } => {
            let expr = Expr {
                id,
                ty,
                kind: ExprKind::Var(String::new()),
                span,
            };
            v.visit_expr_unary_op(op, *operand, expr, ctx)
        }
        ExprKind::If {
            cond,
            then_branch,
            else_branch,
        } => {
            let expr = Expr {
                id,
                ty,
                kind: ExprKind::Var(String::new()),
                span,
            };
            v.visit_expr_if(*cond, *then_branch, *else_branch, expr, ctx)
        }
        ExprKind::Let {
            name,
            binding_id,
            value,
            body,
        } => {
            let expr = Expr {
                id,
                ty,
                kind: ExprKind::Var(String::new()),
                span,
            };
            v.visit_expr_let(name, binding_id, *value, *body, expr, ctx)
        }
        ExprKind::Loop {
            loop_var,
            init,
            init_bindings,
            kind,
            body,
        } => {
            let expr = Expr {
                id,
                ty,
                kind: ExprKind::Var(String::new()),
                span,
            };
            v.visit_expr_loop(loop_var, *init, init_bindings, kind, *body, expr, ctx)
        }
        ExprKind::Call { func, args } => {
            let expr = Expr {
                id,
                ty,
                kind: ExprKind::Var(String::new()),
                span,
            };
            v.visit_expr_call(func, args, expr, ctx)
        }
        ExprKind::Intrinsic { name, args } => {
            let expr = Expr {
                id,
                ty,
                kind: ExprKind::Var(String::new()),
                span,
            };
            v.visit_expr_intrinsic(name, args, expr, ctx)
        }
        ExprKind::Attributed {
            attributes,
            expr: inner,
        } => {
            let expr = Expr {
                id,
                ty,
                kind: ExprKind::Var(String::new()),
                span,
            };
            v.visit_expr_attributed(attributes, *inner, expr, ctx)
        }
        ExprKind::Materialize(inner) => {
            let expr = Expr {
                id,
                ty,
                kind: ExprKind::Var(String::new()),
                span,
            };
            v.visit_expr_materialize(*inner, expr, ctx)
        }
        ExprKind::Closure {
            lambda_name,
            captures,
        } => {
            let expr = Expr {
                id,
                ty,
                kind: ExprKind::Var(String::new()),
                span,
            };
            v.visit_expr_closure(lambda_name, captures, expr, ctx)
        }
        ExprKind::Range {
            start,
            step,
            end,
            kind,
        } => {
            let expr = Expr {
                id,
                ty,
                kind: ExprKind::Var(String::new()),
                span,
            };
            v.visit_expr_range(*start, step.map(|s| *s), *end, kind, expr, ctx)
        }
    }
}

pub fn walk_expr_bin_op<V: MirFolder>(
    v: &mut V,
    op: String,
    lhs: Expr,
    rhs: Expr,
    expr: Expr,
    ctx: &mut V::Ctx,
) -> Result<Expr, V::Error> {
    let lhs = v.visit_expr(lhs, ctx)?;
    let rhs = v.visit_expr(rhs, ctx)?;
    Ok(Expr {
        kind: ExprKind::BinOp {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        },
        ..expr
    })
}

pub fn walk_expr_unary_op<V: MirFolder>(
    v: &mut V,
    op: String,
    operand: Expr,
    expr: Expr,
    ctx: &mut V::Ctx,
) -> Result<Expr, V::Error> {
    let operand = v.visit_expr(operand, ctx)?;
    Ok(Expr {
        kind: ExprKind::UnaryOp {
            op,
            operand: Box::new(operand),
        },
        ..expr
    })
}

pub fn walk_expr_if<V: MirFolder>(
    v: &mut V,
    cond: Expr,
    then_branch: Expr,
    else_branch: Expr,
    expr: Expr,
    ctx: &mut V::Ctx,
) -> Result<Expr, V::Error> {
    let cond = v.visit_expr(cond, ctx)?;
    let then_branch = v.visit_expr(then_branch, ctx)?;
    let else_branch = v.visit_expr(else_branch, ctx)?;
    Ok(Expr {
        kind: ExprKind::If {
            cond: Box::new(cond),
            then_branch: Box::new(then_branch),
            else_branch: Box::new(else_branch),
        },
        ..expr
    })
}

pub fn walk_expr_let<V: MirFolder>(
    v: &mut V,
    name: String,
    binding_id: u64,
    value: Expr,
    body: Expr,
    expr: Expr,
    ctx: &mut V::Ctx,
) -> Result<Expr, V::Error> {
    let value = v.visit_expr(value, ctx)?;
    let body = v.visit_expr(body, ctx)?;
    Ok(Expr {
        kind: ExprKind::Let {
            name,
            binding_id,
            value: Box::new(value),
            body: Box::new(body),
        },
        ..expr
    })
}

pub fn walk_expr_loop<V: MirFolder>(
    v: &mut V,
    loop_var: String,
    init: Expr,
    init_bindings: Vec<(String, Expr)>,
    kind: LoopKind,
    body: Expr,
    expr: Expr,
    ctx: &mut V::Ctx,
) -> Result<Expr, V::Error> {
    let init = v.visit_expr(init, ctx)?;
    let init_bindings = init_bindings
        .into_iter()
        .map(|(name, e)| Ok((name, v.visit_expr(e, ctx)?)))
        .collect::<Result<Vec<_>, _>>()?;

    let kind = v.visit_loop_kind(kind, ctx)?;
    let body = v.visit_expr(body, ctx)?;

    Ok(Expr {
        kind: ExprKind::Loop {
            loop_var,
            init: Box::new(init),
            init_bindings,
            kind,
            body: Box::new(body),
        },
        ..expr
    })
}

pub fn walk_expr_call<V: MirFolder>(
    v: &mut V,
    func: String,
    args: Vec<Expr>,
    expr: Expr,
    ctx: &mut V::Ctx,
) -> Result<Expr, V::Error> {
    let args = args.into_iter().map(|arg| v.visit_expr(arg, ctx)).collect::<Result<Vec<_>, _>>()?;
    Ok(Expr {
        kind: ExprKind::Call { func, args },
        ..expr
    })
}

pub fn walk_expr_intrinsic<V: MirFolder>(
    v: &mut V,
    name: String,
    args: Vec<Expr>,
    expr: Expr,
    ctx: &mut V::Ctx,
) -> Result<Expr, V::Error> {
    let args = args.into_iter().map(|arg| v.visit_expr(arg, ctx)).collect::<Result<Vec<_>, _>>()?;
    Ok(Expr {
        kind: ExprKind::Intrinsic { name, args },
        ..expr
    })
}

pub fn walk_expr_attributed<V: MirFolder>(
    v: &mut V,
    attributes: Vec<Attribute>,
    inner: Expr,
    expr: Expr,
    ctx: &mut V::Ctx,
) -> Result<Expr, V::Error> {
    let attributes =
        attributes.into_iter().map(|a| v.visit_attribute(a, ctx)).collect::<Result<Vec<_>, _>>()?;

    let inner = v.visit_expr(inner, ctx)?;

    Ok(Expr {
        kind: ExprKind::Attributed {
            attributes,
            expr: Box::new(inner),
        },
        ..expr
    })
}

pub fn walk_expr_materialize<V: MirFolder>(
    v: &mut V,
    inner: Expr,
    expr: Expr,
    ctx: &mut V::Ctx,
) -> Result<Expr, V::Error> {
    let inner = v.visit_expr(inner, ctx)?;
    Ok(Expr {
        kind: ExprKind::Materialize(Box::new(inner)),
        ..expr
    })
}

pub fn walk_expr_closure<V: MirFolder>(
    v: &mut V,
    lambda_name: String,
    captures: Vec<Expr>,
    expr: Expr,
    ctx: &mut V::Ctx,
) -> Result<Expr, V::Error> {
    let captures = captures.into_iter().map(|e| v.visit_expr(e, ctx)).collect::<Result<Vec<_>, _>>()?;
    Ok(Expr {
        kind: ExprKind::Closure {
            lambda_name,
            captures,
        },
        ..expr
    })
}

pub fn walk_expr_range<V: MirFolder>(
    v: &mut V,
    start: Expr,
    step: Option<Expr>,
    end: Expr,
    kind: RangeKind,
    expr: Expr,
    ctx: &mut V::Ctx,
) -> Result<Expr, V::Error> {
    let start = v.visit_expr(start, ctx)?;
    let step = step.map(|s| v.visit_expr(s, ctx)).transpose()?;
    let end = v.visit_expr(end, ctx)?;
    Ok(Expr {
        kind: ExprKind::Range {
            start: Box::new(start),
            step: step.map(Box::new),
            end: Box::new(end),
            kind,
        },
        ..expr
    })
}

// --- Literals ---

pub fn walk_literal<V: MirFolder>(v: &mut V, lit: Literal, ctx: &mut V::Ctx) -> Result<Literal, V::Error> {
    match lit {
        Literal::Int(s) => v.visit_literal_int(s, ctx),
        Literal::Float(s) => v.visit_literal_float(s, ctx),
        Literal::Bool(b) => v.visit_literal_bool(b, ctx),
        Literal::String(s) => v.visit_literal_string(s, ctx),
        Literal::Tuple(elems) => v.visit_literal_tuple(elems, ctx),
        Literal::Array(elems) => v.visit_literal_array(elems, ctx),
        Literal::Vector(elems) => {
            let elems = elems.into_iter().map(|e| v.visit_expr(e, ctx)).collect::<Result<Vec<_>, _>>()?;
            Ok(Literal::Vector(elems))
        }
        Literal::Matrix(rows) => {
            let rows = rows
                .into_iter()
                .map(|row| row.into_iter().map(|e| v.visit_expr(e, ctx)).collect::<Result<Vec<_>, _>>())
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Literal::Matrix(rows))
        }
    }
}

pub fn walk_literal_tuple<V: MirFolder>(
    v: &mut V,
    elements: Vec<Expr>,
    ctx: &mut V::Ctx,
) -> Result<Literal, V::Error> {
    let elements = elements.into_iter().map(|e| v.visit_expr(e, ctx)).collect::<Result<Vec<_>, _>>()?;
    Ok(Literal::Tuple(elements))
}

pub fn walk_literal_array<V: MirFolder>(
    v: &mut V,
    elements: Vec<Expr>,
    ctx: &mut V::Ctx,
) -> Result<Literal, V::Error> {
    let elements = elements.into_iter().map(|e| v.visit_expr(e, ctx)).collect::<Result<Vec<_>, _>>()?;
    Ok(Literal::Array(elements))
}

// --- Loop kinds ---

pub fn walk_loop_kind<V: MirFolder>(
    v: &mut V,
    kind: LoopKind,
    ctx: &mut V::Ctx,
) -> Result<LoopKind, V::Error> {
    match kind {
        LoopKind::For { var, iter } => v.visit_for_loop(var, *iter, ctx),
        LoopKind::ForRange { var, bound } => v.visit_for_range_loop(var, *bound, ctx),
        LoopKind::While { cond } => v.visit_while_loop(*cond, ctx),
    }
}

pub fn walk_for_loop<V: MirFolder>(
    v: &mut V,
    var: String,
    iter: Expr,
    ctx: &mut V::Ctx,
) -> Result<LoopKind, V::Error> {
    let iter = v.visit_expr(iter, ctx)?;
    Ok(LoopKind::For {
        var,
        iter: Box::new(iter),
    })
}

pub fn walk_for_range_loop<V: MirFolder>(
    v: &mut V,
    var: String,
    bound: Expr,
    ctx: &mut V::Ctx,
) -> Result<LoopKind, V::Error> {
    let bound = v.visit_expr(bound, ctx)?;
    Ok(LoopKind::ForRange {
        var,
        bound: Box::new(bound),
    })
}

pub fn walk_while_loop<V: MirFolder>(
    v: &mut V,
    cond: Expr,
    ctx: &mut V::Ctx,
) -> Result<LoopKind, V::Error> {
    let cond = v.visit_expr(cond, ctx)?;
    Ok(LoopKind::While { cond: Box::new(cond) })
}
