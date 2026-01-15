//! TLC to MIR transformation.
//!
//! Transforms a lifted TLC program (where all lambdas are top-level) into MIR.

use super::{Def as TlcDef, FunctionName, Program as TlcProgram, Term, TermKind};
use crate::ast::{NodeId, Span, TypeName};
use crate::mir::{self, Body, Def as MirDef, Expr, ExprId, LocalDecl, LocalId, LocalKind};
use polytype::Type;
use std::collections::HashMap;

/// Transforms TLC to MIR.
pub struct TlcToMir {
    /// Maps TLC variable names to MIR LocalIds (within current body)
    locals: HashMap<String, LocalId>,
    /// Maps top-level function names to their definitions
    top_level: HashMap<String, TlcDef>,
}

impl TlcToMir {
    /// Transform a lifted TLC program to MIR.
    pub fn transform(program: &TlcProgram) -> mir::Program {
        // Collect top-level names
        let top_level: HashMap<String, TlcDef> = program
            .defs
            .iter()
            .map(|d| (d.name.clone(), d.clone()))
            .collect();

        let mut transformer = Self {
            locals: HashMap::new(),
            top_level,
        };

        let defs: Vec<MirDef> = program
            .defs
            .iter()
            .map(|def| transformer.transform_def(def))
            .collect();

        mir::Program {
            defs,
            lambda_registry: crate::IdArena::new(),
        }
    }

    fn transform_def(&mut self, def: &TlcDef) -> MirDef {
        self.locals.clear();

        let mut body = Body::new();

        // Extract parameters from nested Lams
        let (params, inner_body) = self.extract_params(&def.body);

        // Register parameters as locals
        let param_ids: Vec<LocalId> = params
            .iter()
            .map(|(name, ty, span)| {
                let local_id = body.alloc_local(LocalDecl {
                    name: name.clone(),
                    span: *span,
                    ty: ty.clone(),
                    kind: LocalKind::Param,
                });
                self.locals.insert(name.clone(), local_id);
                local_id
            })
            .collect();

        // Transform the body
        let root = self.transform_term(inner_body, &mut body);
        body.set_root(root);

        if params.is_empty() {
            // Constant (no parameters)
            MirDef::Constant {
                id: NodeId(0), // TLC doesn't track NodeIds, use placeholder
                name: def.name.clone(),
                ty: def.ty.clone(),
                attributes: vec![],
                body,
                span: def.body.span,
            }
        } else {
            // Function with parameters
            MirDef::Function {
                id: NodeId(0),
                name: def.name.clone(),
                params: param_ids,
                ret_type: inner_body.ty.clone(),
                scheme: None,
                attributes: vec![],
                body,
                span: def.body.span,
            }
        }
    }

    /// Extract curried parameters from nested Lams.
    /// Returns (params, inner_body) where params is [(name, type, span), ...]
    fn extract_params<'a>(&self, term: &'a Term) -> (Vec<(String, Type<TypeName>, Span)>, &'a Term) {
        match &term.kind {
            TermKind::Lam {
                param,
                param_ty,
                body,
            } => {
                let (mut params, inner) = self.extract_params(body);
                params.insert(0, (param.clone(), param_ty.clone(), term.span));
                (params, inner)
            }
            _ => (vec![], term),
        }
    }

    fn transform_term(&mut self, term: &Term, body: &mut Body) -> ExprId {
        let ty = term.ty.clone();
        let span = term.span;
        let node_id = NodeId(term.id.0); // Use term ID as node ID

        match &term.kind {
            TermKind::Var(name) => {
                if let Some(&local_id) = self.locals.get(name) {
                    // Local variable reference
                    body.alloc_expr(Expr::Local(local_id), ty, span, node_id)
                } else if self.top_level.contains_key(name) {
                    // Global function reference
                    body.alloc_expr(Expr::Global(name.clone()), ty, span, node_id)
                } else {
                    // Unknown variable - could be an intrinsic
                    body.alloc_expr(Expr::Global(name.clone()), ty, span, node_id)
                }
            }

            TermKind::IntLit(s) => body.alloc_expr(Expr::Int(s.clone()), ty, span, node_id),

            TermKind::FloatLit(f) => body.alloc_expr(Expr::Float(f.to_string()), ty, span, node_id),

            TermKind::BoolLit(b) => body.alloc_expr(Expr::Bool(*b), ty, span, node_id),

            TermKind::StringLit(s) => body.alloc_expr(Expr::String(s.clone()), ty, span, node_id),

            TermKind::Let {
                name,
                name_ty,
                rhs,
                body: let_body,
            } => {
                let rhs_id = self.transform_term(rhs, body);
                let local_id = body.alloc_local(LocalDecl {
                    name: name.clone(),
                    span: rhs.span,
                    ty: name_ty.clone(),
                    kind: LocalKind::Let,
                });
                self.locals.insert(name.clone(), local_id);
                let body_id = self.transform_term(let_body, body);
                self.locals.remove(name);

                body.alloc_expr(
                    Expr::Let {
                        local: local_id,
                        rhs: rhs_id,
                        body: body_id,
                    },
                    ty,
                    span,
                    node_id,
                )
            }

            TermKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let cond_id = self.transform_term(cond, body);
                let then_id = self.transform_term(then_branch, body);
                let else_id = self.transform_term(else_branch, body);

                body.alloc_expr(
                    Expr::If {
                        cond: cond_id,
                        then_: then_id,
                        else_: else_id,
                    },
                    ty,
                    span,
                    node_id,
                )
            }

            TermKind::App { func, arg } => self.transform_app(func, arg, ty, span, node_id, body),

            TermKind::Lam { .. } => {
                // Lambdas should have been lifted - if we see one here, it's an error
                panic!(
                    "Unexpected lambda in TLC to MIR transformation at {:?}. \
                     All lambdas should have been lifted to top-level.",
                    span
                )
            }
        }
    }

    /// Transform an application. Handles curried BinOp/UnOp specially.
    fn transform_app(
        &mut self,
        func: &FunctionName,
        arg: &Term,
        ty: Type<TypeName>,
        span: Span,
        node_id: NodeId,
        body: &mut Body,
    ) -> ExprId {
        match func {
            FunctionName::BinOp(_op) => {
                // BinOp applied to first arg - need to find the second arg
                // This is a partial application: (+) x
                // The outer App will complete it
                // For now, we need to handle the pattern: App(Term(App(BinOp(+), lhs)), rhs)

                // This is the first application (BinOp to LHS)
                // We return a marker that the outer transform_app will recognize
                // Actually, let's look at the structure: the full binop is:
                // App { func: Term(App { func: BinOp(+), arg: lhs }), arg: rhs }
                //
                // So when we see BinOp directly, we're in the inner App and just have the LHS
                // We need to emit the full BinOp when we see the outer App with Term containing this

                // For the inner application, just transform the arg and return it
                // The outer FunctionName::Term case will handle completing the binop
                let _lhs_id = self.transform_term(arg, body);

                // We can't emit a BinOp here because we only have one operand
                // Return a special "partial binop" - but MIR doesn't have this concept
                // We need to restructure: detect the pattern at the outer level

                // Actually, let's handle this differently: when we see FunctionName::Term
                // containing an App with BinOp, we know it's a complete binary operation

                // For now, panic - we need to handle this at the App level
                panic!(
                    "Direct BinOp application should be handled by outer App at {:?}",
                    span
                );
            }

            FunctionName::UnOp(op) => {
                // UnOp is complete with one argument
                let operand_id = self.transform_term(arg, body);
                body.alloc_expr(
                    Expr::UnaryOp {
                        op: op.op.clone(),
                        operand: operand_id,
                    },
                    ty,
                    span,
                    node_id,
                )
            }

            FunctionName::Var(name) => {
                // Function call - but in curried form, this is partial application
                // Collect all args by walking the nested Apps
                let args = self.collect_curried_args(arg, body);
                body.alloc_expr(
                    Expr::Call {
                        func: name.clone(),
                        args,
                    },
                    ty,
                    span,
                    node_id,
                )
            }

            FunctionName::Term(inner_term) => {
                // Check if inner_term is a binop partial application
                if let TermKind::App {
                    func: inner_func,
                    arg: lhs,
                } = &inner_term.kind
                {
                    if let FunctionName::BinOp(op) = inner_func.as_ref() {
                        // Complete binary operation: we have both lhs and rhs (arg)
                        let lhs_id = self.transform_term(lhs, body);
                        let rhs_id = self.transform_term(arg, body);
                        return body.alloc_expr(
                            Expr::BinOp {
                                op: op.op.clone(),
                                lhs: lhs_id,
                                rhs: rhs_id,
                            },
                            ty,
                            span,
                            node_id,
                        );
                    }
                }

                // Generic higher-order application
                // First transform the function term, then apply
                let func_id = self.transform_term(inner_term, body);
                let arg_id = self.transform_term(arg, body);

                // Check if it's a call to a known function
                if let Expr::Global(func_name) = body.get_expr(func_id) {
                    let func_name = func_name.clone();
                    // Collect all curried args
                    body.alloc_expr(
                        Expr::Call {
                            func: func_name,
                            args: vec![arg_id],
                        },
                        ty,
                        span,
                        node_id,
                    )
                } else if let Expr::Call { func: func_name, args: existing_args } = body.get_expr(func_id).clone() {
                    // Extend the call with more arguments
                    let mut args = existing_args;
                    args.push(arg_id);
                    body.alloc_expr(
                        Expr::Call {
                            func: func_name,
                            args,
                        },
                        ty,
                        span,
                        node_id,
                    )
                } else {
                    // True higher-order application - need closure apply
                    // For now, emit as intrinsic call
                    body.alloc_expr(
                        Expr::Intrinsic {
                            name: "_w_apply".to_string(),
                            args: vec![func_id, arg_id],
                        },
                        ty,
                        span,
                        node_id,
                    )
                }
            }
        }
    }

    /// Collect arguments from a curried application chain.
    fn collect_curried_args(&mut self, term: &Term, body: &mut Body) -> Vec<ExprId> {
        // For now, just transform the single argument
        // A more complete implementation would walk nested Apps
        vec![self.transform_term(term, body)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tlc::TermIdSource;

    fn make_span(line: usize, col: usize) -> Span {
        Span {
            start_line: line,
            start_col: col,
            end_line: line,
            end_col: col + 1,
        }
    }

    #[test]
    fn test_transform_simple_function() {
        // def add(x, y) = x + y
        // In TLC (after lifting): def add = |x| |y| (+) x y
        let mut ids = TermIdSource::new();
        let span = make_span(1, 1);

        let x_var = Term {
            id: ids.next_id(),
            ty: Type::Constructed(TypeName::Int(32), vec![]),
            span,
            kind: TermKind::Var("x".to_string()),
        };

        let y_var = Term {
            id: ids.next_id(),
            ty: Type::Constructed(TypeName::Int(32), vec![]),
            span,
            kind: TermKind::Var("y".to_string()),
        };

        // Build: (+) x y as App(App(BinOp(+), x), y)
        use crate::ast::BinaryOp;
        let binop_x = Term {
            id: ids.next_id(),
            ty: Type::Constructed(
                TypeName::Arrow,
                vec![
                    Type::Constructed(TypeName::Int(32), vec![]),
                    Type::Constructed(TypeName::Int(32), vec![]),
                ],
            ),
            span,
            kind: TermKind::App {
                func: Box::new(FunctionName::BinOp(BinaryOp {
                    op: "+".to_string(),
                })),
                arg: Box::new(x_var),
            },
        };

        let add_body = Term {
            id: ids.next_id(),
            ty: Type::Constructed(TypeName::Int(32), vec![]),
            span,
            kind: TermKind::App {
                func: Box::new(FunctionName::Term(Box::new(binop_x))),
                arg: Box::new(y_var),
            },
        };

        // |y| body
        let lam_y = Term {
            id: ids.next_id(),
            ty: Type::Constructed(
                TypeName::Arrow,
                vec![
                    Type::Constructed(TypeName::Int(32), vec![]),
                    Type::Constructed(TypeName::Int(32), vec![]),
                ],
            ),
            span,
            kind: TermKind::Lam {
                param: "y".to_string(),
                param_ty: Type::Constructed(TypeName::Int(32), vec![]),
                body: Box::new(add_body),
            },
        };

        // |x| |y| body
        let lam_x = Term {
            id: ids.next_id(),
            ty: Type::Constructed(
                TypeName::Arrow,
                vec![
                    Type::Constructed(TypeName::Int(32), vec![]),
                    Type::Constructed(
                        TypeName::Arrow,
                        vec![
                            Type::Constructed(TypeName::Int(32), vec![]),
                            Type::Constructed(TypeName::Int(32), vec![]),
                        ],
                    ),
                ],
            ),
            span,
            kind: TermKind::Lam {
                param: "x".to_string(),
                param_ty: Type::Constructed(TypeName::Int(32), vec![]),
                body: Box::new(lam_y),
            },
        };

        let program = TlcProgram {
            defs: vec![TlcDef {
                name: "add".to_string(),
                ty: lam_x.ty.clone(),
                body: lam_x,
            }],
        };

        let mir = TlcToMir::transform(&program);

        assert_eq!(mir.defs.len(), 1);
        match &mir.defs[0] {
            MirDef::Function { name, params, .. } => {
                assert_eq!(name, "add");
                assert_eq!(params.len(), 2);
            }
            _ => panic!("Expected Function"),
        }
    }
}
