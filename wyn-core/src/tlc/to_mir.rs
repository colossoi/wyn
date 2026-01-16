//! TLC to MIR transformation.
//!
//! Transforms a lifted TLC program (where all lambdas are top-level) into MIR.

use super::{Def as TlcDef, DefMeta, FunctionName, Program as TlcProgram, Term, TermKind};
use crate::ast::{self, NodeId, PatternKind, Span, TypeName};
use crate::mir::{self, Body, Def as MirDef, Expr, ExprId, LocalDecl, LocalId, LocalKind, LoopKind};
use polytype::Type;
use std::collections::HashMap;

/// Transforms TLC to MIR.
pub struct TlcToMir {
    /// Maps TLC variable names to MIR LocalIds (within current body)
    locals: HashMap<String, LocalId>,
    /// Maps top-level function names to their definitions
    top_level: HashMap<String, TlcDef>,
    /// Static value tracking: maps local variable names to their known function targets.
    /// When we have `let f = some_lambda`, this records `f -> "some_lambda"`.
    /// This enables direct calls instead of dynamic `_w_apply`.
    static_values: HashMap<String, String>,
}

impl TlcToMir {
    /// Transform a lifted TLC program to MIR.
    pub fn transform(program: &TlcProgram) -> mir::Program {
        // Collect top-level names
        let top_level: HashMap<String, TlcDef> =
            program.defs.iter().map(|d| (d.name.clone(), d.clone())).collect();

        let mut transformer = Self {
            locals: HashMap::new(),
            top_level,
            static_values: HashMap::new(),
        };

        let mut defs: Vec<MirDef> = program.defs.iter().map(|def| transformer.transform_def(def)).collect();

        // Add uniform declarations
        for uniform in &program.uniforms {
            defs.push(MirDef::Uniform {
                id: NodeId(0),
                name: uniform.name.clone(),
                ty: uniform.ty.clone(),
                set: uniform.set,
                binding: uniform.binding,
            });
        }

        // Add storage declarations
        for storage in &program.storage {
            defs.push(MirDef::Storage {
                id: NodeId(0),
                name: storage.name.clone(),
                ty: storage.ty.clone(),
                set: storage.set,
                binding: storage.binding,
                layout: storage.layout,
                access: storage.access,
            });
        }

        mir::Program {
            defs,
            lambda_registry: crate::IdArena::new(),
        }
    }

    fn transform_def(&mut self, def: &TlcDef) -> MirDef {
        self.locals.clear();
        self.static_values.clear();

        match &def.meta {
            DefMeta::Function => self.transform_function_def(def),
            DefMeta::EntryPoint(entry) => self.transform_entry_def(def, entry),
        }
    }

    fn transform_function_def(&mut self, def: &TlcDef) -> MirDef {
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
                id: NodeId(0),
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

    fn transform_entry_def(&mut self, def: &TlcDef, entry: &ast::EntryDecl) -> MirDef {
        let mut body = Body::new();

        // Extract parameters from nested Lams
        let (params, inner_body) = self.extract_params(&def.body);

        // Build inputs with decorations from AST
        let mut inputs = Vec::new();
        for (i, (name, ty, span)) in params.iter().enumerate() {
            let local_id = body.alloc_local(LocalDecl {
                name: name.clone(),
                span: *span,
                ty: ty.clone(),
                kind: LocalKind::Param,
            });
            self.locals.insert(name.clone(), local_id);

            // Get decoration from original AST pattern
            let decoration = entry.params.get(i).and_then(|p| self.extract_io_decoration(p));

            inputs.push(mir::EntryInput {
                local: local_id,
                name: name.clone(),
                ty: ty.clone(),
                decoration,
            });
        }

        // Transform the body
        let root = self.transform_term(inner_body, &mut body);
        body.set_root(root);

        // Convert entry type to ExecutionModel
        let execution_model = match &entry.entry_type {
            ast::Attribute::Vertex => mir::ExecutionModel::Vertex,
            ast::Attribute::Fragment => mir::ExecutionModel::Fragment,
            ast::Attribute::Compute => mir::ExecutionModel::Compute {
                local_size: (64, 1, 1),
            },
            _ => panic!("Invalid entry type attribute: {:?}", entry.entry_type),
        };

        // Build outputs from AST
        let ret_type = inner_body.ty.clone();
        let outputs = self.build_entry_outputs(entry, &ret_type);

        MirDef::EntryPoint {
            id: NodeId(0),
            name: def.name.clone(),
            execution_model,
            inputs,
            outputs,
            body,
            span: def.body.span,
        }
    }

    /// Extract I/O decoration from a pattern
    fn extract_io_decoration(&self, pattern: &ast::Pattern) -> Option<mir::IoDecoration> {
        match &pattern.kind {
            PatternKind::Attributed(attrs, inner) => {
                for attr in attrs {
                    match attr {
                        ast::Attribute::BuiltIn(builtin) => {
                            return Some(mir::IoDecoration::BuiltIn(*builtin));
                        }
                        ast::Attribute::Location(loc) => {
                            return Some(mir::IoDecoration::Location(*loc));
                        }
                        _ => {}
                    }
                }
                self.extract_io_decoration(inner)
            }
            PatternKind::Typed(inner, _) => self.extract_io_decoration(inner),
            _ => None,
        }
    }

    /// Build entry outputs from AST entry declaration
    fn build_entry_outputs(
        &self,
        entry: &ast::EntryDecl,
        ret_type: &Type<TypeName>,
    ) -> Vec<mir::EntryOutput> {
        if entry.outputs.iter().all(|o| o.attribute.is_none()) && entry.outputs.len() == 1 {
            // Single output without explicit decoration
            if !matches!(ret_type, Type::Constructed(TypeName::Unit, _)) {
                vec![mir::EntryOutput {
                    ty: ret_type.clone(),
                    decoration: None,
                }]
            } else {
                vec![]
            }
        } else if let Type::Constructed(TypeName::Tuple(_), component_types) = ret_type {
            // Multiple outputs with decorations (tuple return)
            entry
                .outputs
                .iter()
                .zip(component_types.iter())
                .map(|(output, ty)| mir::EntryOutput {
                    ty: ty.clone(),
                    decoration: output.attribute.as_ref().and_then(|a| self.convert_to_io_decoration(a)),
                })
                .collect()
        } else {
            // Single output with decoration
            vec![mir::EntryOutput {
                ty: ret_type.clone(),
                decoration: entry
                    .outputs
                    .first()
                    .and_then(|o| o.attribute.as_ref())
                    .and_then(|a| self.convert_to_io_decoration(a)),
            }]
        }
    }

    /// Convert AST attribute to MIR IoDecoration
    fn convert_to_io_decoration(&self, attr: &ast::Attribute) -> Option<mir::IoDecoration> {
        match attr {
            ast::Attribute::BuiltIn(builtin) => Some(mir::IoDecoration::BuiltIn(*builtin)),
            ast::Attribute::Location(loc) => Some(mir::IoDecoration::Location(*loc)),
            _ => None,
        }
    }

    /// Extract the static function value from a term, if known.
    /// Returns Some(function_name) if the term is a reference to a known function.
    fn extract_static_value(&self, term: &Term) -> Option<String> {
        match &term.kind {
            TermKind::Var(name) => {
                // Check if it's a top-level function (not a constant)
                if let Some(def) = self.top_level.get(name) {
                    if def.arity > 0 {
                        Some(name.clone())
                    } else {
                        None // Constants are not callable functions
                    }
                } else {
                    // Check if it's a local bound to a known function
                    self.static_values.get(name).cloned()
                }
            }
            _ => None,
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
                } else if let Some(def) = self.top_level.get(name) {
                    if def.arity > 0 {
                        // Top-level function used as value → Closure with no captures
                        body.alloc_expr(
                            Expr::Closure {
                                lambda_name: name.clone(),
                                captures: vec![],
                            },
                            ty,
                            span,
                            node_id,
                        )
                    } else {
                        // Top-level constant → Global reference
                        body.alloc_expr(Expr::Global(name.clone()), ty, span, node_id)
                    }
                } else {
                    // Unknown variable - could be an intrinsic or global constant
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
                // Extract static value before transforming (for function tracking)
                let static_val = self.extract_static_value(rhs);

                let rhs_id = self.transform_term(rhs, body);
                let local_id = body.alloc_local(LocalDecl {
                    name: name.clone(),
                    span: rhs.span,
                    ty: name_ty.clone(),
                    kind: LocalKind::Let,
                });
                self.locals.insert(name.clone(), local_id);

                // Record static value if RHS is a known function
                if let Some(func_name) = static_val {
                    self.static_values.insert(name.clone(), func_name);
                }

                let body_id = self.transform_term(let_body, body);

                self.locals.remove(name);
                self.static_values.remove(name);

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

            FunctionName::Var(name) | FunctionName::Intrinsic(name) => {
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
                // Check if inner_term is a variable with a known static function value
                if let Some(func_name) = self.extract_static_value(inner_term) {
                    // Direct call to the known function
                    let arg_id = self.transform_term(arg, body);
                    return body.alloc_expr(
                        Expr::Call {
                            func: func_name,
                            args: vec![arg_id],
                        },
                        ty,
                        span,
                        node_id,
                    );
                }

                // Check if inner_term is a binop partial application
                if let TermKind::App {
                    func: inner_func,
                    arg: first_arg,
                } = &inner_term.kind
                {
                    if let FunctionName::BinOp(op) = inner_func.as_ref() {
                        // Complete binary operation: we have both lhs and rhs (arg)
                        let lhs_id = self.transform_term(first_arg, body);
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

                    // Check for two-arg intrinsics: _w_index, _w_tuple_proj
                    if let FunctionName::Intrinsic(name) = inner_func.as_ref() {
                        match name.as_str() {
                            "_w_index" | "_w_tuple_proj" => {
                                // Two-arg intrinsic: complete with both args
                                let first_id = self.transform_term(first_arg, body);
                                let second_id = self.transform_term(arg, body);
                                return body.alloc_expr(
                                    Expr::Intrinsic {
                                        name: name.clone(),
                                        args: vec![first_id, second_id],
                                    },
                                    ty,
                                    span,
                                    node_id,
                                );
                            }
                            _ => {}
                        }
                    }

                    // Check for three-arg intrinsics like _w_loop_while
                    // Pattern: App(App(App(Var("_w_loop_while"), init), cond), body)
                    // At this level: inner_func = Term(App(Var("_w_loop_while"), init)), first_arg = cond
                    if let FunctionName::Term(inner_inner_term) = inner_func.as_ref() {
                        if let TermKind::App {
                            func: innermost_func,
                            arg: init_arg,
                        } = &inner_inner_term.kind
                        {
                            if let FunctionName::Intrinsic(intrinsic_name) = innermost_func.as_ref() {
                                if intrinsic_name == "_w_loop_while" {
                                    return self.transform_loop_while(
                                        init_arg,  // init
                                        first_arg, // cond_func
                                        arg,       // body_func
                                        ty, span, node_id, body,
                                    );
                                }
                            }
                        }
                    }
                }

                // Generic higher-order application
                // First transform the function term, then apply
                let func_id = self.transform_term(inner_term, body);
                let arg_id = self.transform_term(arg, body);

                // Check if it's a call to a known function
                if let Expr::Global(func_name) = body.get_expr(func_id) {
                    let func_name = func_name.clone();
                    // Check for literal intrinsics that should become MIR literals/intrinsics
                    match func_name.as_str() {
                        "_w_vec_lit" => body.alloc_expr(Expr::Vector(vec![arg_id]), ty, span, node_id),
                        "_w_array_lit" => {
                            // For arrays, we need the size expression
                            let size = body.alloc_expr(
                                Expr::Int("1".to_string()),
                                Type::Constructed(TypeName::Int(32), vec![]),
                                span,
                                node_id,
                            );
                            body.alloc_expr(
                                Expr::Array {
                                    backing: mir::ArrayBacking::Literal(vec![arg_id]),
                                    size,
                                },
                                ty,
                                span,
                                node_id,
                            )
                        }
                        "_w_tuple" => body.alloc_expr(Expr::Tuple(vec![arg_id]), ty, span, node_id),
                        // Intrinsics that need special handling
                        name if name.starts_with("_w_") => body.alloc_expr(
                            Expr::Intrinsic {
                                name: func_name,
                                args: vec![arg_id],
                            },
                            ty,
                            span,
                            node_id,
                        ),
                        _ => body.alloc_expr(
                            Expr::Call {
                                func: func_name,
                                args: vec![arg_id],
                            },
                            ty,
                            span,
                            node_id,
                        ),
                    }
                } else if let Expr::Call {
                    func: func_name,
                    args: existing_args,
                } = body.get_expr(func_id).clone()
                {
                    // Extend the call with more arguments
                    let mut args = existing_args;
                    args.push(arg_id);

                    // Check for literal intrinsics that should become MIR literals
                    match func_name.as_str() {
                        "_w_vec_lit" => body.alloc_expr(Expr::Vector(args), ty, span, node_id),
                        "_w_array_lit" => {
                            let len = args.len();
                            let size = body.alloc_expr(
                                Expr::Int(len.to_string()),
                                Type::Constructed(TypeName::Int(32), vec![]),
                                span,
                                node_id,
                            );
                            body.alloc_expr(
                                Expr::Array {
                                    backing: mir::ArrayBacking::Literal(args),
                                    size,
                                },
                                ty,
                                span,
                                node_id,
                            )
                        }
                        "_w_tuple" => body.alloc_expr(Expr::Tuple(args), ty, span, node_id),
                        _ => body.alloc_expr(
                            Expr::Call {
                                func: func_name,
                                args,
                            },
                            ty,
                            span,
                            node_id,
                        ),
                    }
                } else if let Expr::Intrinsic {
                    name: intrinsic_name,
                    args: existing_args,
                } = body.get_expr(func_id).clone()
                {
                    // Extend an intrinsic with more arguments
                    let mut args = existing_args;
                    args.push(arg_id);
                    body.alloc_expr(
                        Expr::Intrinsic {
                            name: intrinsic_name,
                            args,
                        },
                        ty,
                        span,
                        node_id,
                    )
                } else if let Expr::Tuple(existing_elems) = body.get_expr(func_id).clone() {
                    // Extend an existing tuple literal
                    let mut elems = existing_elems;
                    elems.push(arg_id);
                    body.alloc_expr(Expr::Tuple(elems), ty, span, node_id)
                } else if let Expr::Vector(existing_elems) = body.get_expr(func_id).clone() {
                    // Extend an existing vector literal
                    let mut elems = existing_elems;
                    elems.push(arg_id);
                    body.alloc_expr(Expr::Vector(elems), ty, span, node_id)
                } else if let Expr::Array {
                    backing: mir::ArrayBacking::Literal(existing_elems),
                    ..
                } = body.get_expr(func_id).clone()
                {
                    // Extend an existing array literal
                    let mut elems = existing_elems;
                    elems.push(arg_id);
                    let len = elems.len();
                    let size = body.alloc_expr(
                        Expr::Int(len.to_string()),
                        Type::Constructed(TypeName::Int(32), vec![]),
                        span,
                        node_id,
                    );
                    body.alloc_expr(
                        Expr::Array {
                            backing: mir::ArrayBacking::Literal(elems),
                            size,
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

    /// Transform a _w_loop_while intrinsic into an Expr::Loop.
    ///
    /// The TLC form is: _w_loop_while init cond_func body_func
    /// Where:
    /// - init is the initial accumulator value (usually a tuple)
    /// - cond_func is a function from accumulator → bool
    /// - body_func is a function from accumulator → accumulator
    fn transform_loop_while(
        &mut self,
        init: &Term,
        cond_func: &Term,
        body_func: &Term,
        result_ty: Type<TypeName>,
        span: Span,
        node_id: NodeId,
        body: &mut Body,
    ) -> ExprId {
        // Transform the initial value
        let init_id = self.transform_term(init, body);
        let init_ty = init.ty.clone();

        // Create a loop variable for the accumulator
        let loop_var_name = format!("_loop_acc_{}", node_id.0);
        let loop_var = body.alloc_local(LocalDecl {
            name: loop_var_name.clone(),
            ty: init_ty.clone(),
            kind: LocalKind::LoopVar,
            span,
        });

        // Add to locals map so it can be referenced in cond/body
        self.locals.insert(loop_var_name.clone(), loop_var);

        // Create a reference to the loop variable
        let loop_var_ref = body.alloc_expr(Expr::Local(loop_var), init_ty.clone(), span, node_id);

        // Transform the condition: call cond_func with loop_var
        // We need to figure out the function name and any captures
        let bool_ty = Type::Constructed(TypeName::Str("bool"), vec![]);
        let cond_id = self.emit_func_call_with_arg(cond_func, loop_var_ref, bool_ty, span, node_id, body);

        // Create another reference to the loop variable for the body
        let loop_var_ref2 = body.alloc_expr(Expr::Local(loop_var), init_ty, span, node_id);

        // Transform the body: call body_func with loop_var
        let body_expr_id =
            self.emit_func_call_with_arg(body_func, loop_var_ref2, result_ty.clone(), span, node_id, body);

        // Remove the temporary local binding
        self.locals.remove(&loop_var_name);

        // Create the Loop expression
        body.alloc_expr(
            Expr::Loop {
                loop_var,
                init: init_id,
                init_bindings: vec![], // No pattern destructuring at MIR level
                kind: LoopKind::While { cond: cond_id },
                body: body_expr_id,
            },
            result_ty,
            span,
            node_id,
        )
    }

    /// Emit a function call that applies `func_term` to an additional `extra_arg`.
    ///
    /// Handles two cases:
    /// 1. func_term is Var("name") -> Call { func: name, args: [extra_arg] }
    /// 2. func_term is App(App(...Var("name")..., cap1), cap2) -> Call { func: name, args: [cap1, cap2, extra_arg] }
    fn emit_func_call_with_arg(
        &mut self,
        func_term: &Term,
        extra_arg: ExprId,
        result_ty: Type<TypeName>,
        span: Span,
        node_id: NodeId,
        body: &mut Body,
    ) -> ExprId {
        // Collect the function name and capture arguments
        let (func_name, capture_args) = self.collect_func_and_captures(func_term, body);

        // Build args: captures first, then the extra arg
        let mut args = capture_args;
        args.push(extra_arg);

        body.alloc_expr(
            Expr::Call {
                func: func_name,
                args,
            },
            result_ty,
            span,
            node_id,
        )
    }

    /// Collect function name and capture arguments from a term.
    ///
    /// For Var("name"): returns ("name", [])
    /// For App(App(Var("name"), cap1), cap2): returns ("name", [cap1_id, cap2_id])
    fn collect_func_and_captures(&mut self, term: &Term, body: &mut Body) -> (String, Vec<ExprId>) {
        match &term.kind {
            TermKind::Var(name) => (name.clone(), vec![]),
            TermKind::App { func, arg } => {
                // Recursively collect from the function position
                let (func_name, mut caps) = match func.as_ref() {
                    FunctionName::Var(name) | FunctionName::Intrinsic(name) => (name.clone(), vec![]),
                    FunctionName::Term(inner) => self.collect_func_and_captures(inner, body),
                    _ => panic!("Unexpected function form in loop lambda: {:?}", func),
                };
                // Transform and add this argument
                let arg_id = self.transform_term(arg, body);
                caps.push(arg_id);
                (func_name, caps)
            }
            _ => panic!("Unexpected term form in loop lambda: {:?}", term.kind),
        }
    }
}
