//! TLC to MIR transformation.
//!
//! Transforms a lifted TLC program (where all lambdas are top-level) into MIR.

use super::{Def as TlcDef, DefMeta, LoopKind as TlcLoopKind, Program as TlcProgram, Term, TermKind};
use crate::ast::{self, NodeId, PatternKind, Span, TypeName};
use crate::mir::{
    self, ArrayBacking, Body, Def as MirDef, Expr, ExprId, LocalDecl, LocalId, LocalKind,
    LoopKind as MirLoopKind,
};
use crate::types::TypeScheme;
use polytype::Type;
use std::collections::HashMap;

/// Transforms TLC to MIR.
pub struct TlcToMir<'a> {
    /// Maps TLC variable names to MIR LocalIds (within current body)
    locals: HashMap<String, LocalId>,
    /// Maps top-level function names to their definitions
    top_level: HashMap<String, TlcDef>,
    /// Type schemes for functions (for monomorphization)
    schemes: &'a HashMap<String, TypeScheme>,
}

impl<'a> TlcToMir<'a> {
    /// Transform a lifted TLC program to MIR.
    pub fn transform(program: &TlcProgram, schemes: &'a HashMap<String, TypeScheme>) -> mir::Program {
        // Collect top-level names
        let top_level: HashMap<String, TlcDef> =
            program.defs.iter().map(|d| (d.name.clone(), d.clone())).collect();

        let mut transformer = Self {
            locals: HashMap::new(),
            top_level,
            schemes,
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
            // Look up type scheme for this function (if it's a prelude function)
            let scheme = self.schemes.get(&def.name).cloned();
            MirDef::Function {
                id: NodeId(0),
                name: def.name.clone(),
                params: param_ids,
                ret_type: inner_body.ty.clone(),
                scheme,
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

    /// Extract curried parameters from nested Lams.
    /// Returns (params, inner_body) where params is [(name, type, span), ...]
    fn extract_params<'b>(&self, term: &'b Term) -> (Vec<(String, Type<TypeName>, Span)>, &'b Term) {
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
        let node_id = NodeId(0); // Synthetic term - no AST origin

        match &term.kind {
            TermKind::Var(name) => {
                if let Some(&local_id) = self.locals.get(name) {
                    // Local variable reference
                    body.alloc_expr(Expr::Local(local_id), ty, span, node_id)
                } else if self.top_level.contains_key(name) {
                    // Top-level function or constant → Global reference
                    body.alloc_expr(Expr::Global(name.clone()), ty, span, node_id)
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

            TermKind::Loop {
                loop_var,
                loop_var_ty,
                init,
                init_bindings,
                kind,
                body: loop_body,
            } => {
                // Transform init (outside loop scope)
                let init_id = self.transform_term(init, body);

                // Allocate loop variable
                let loop_var_local = body.alloc_local(LocalDecl {
                    name: loop_var.clone(),
                    span,
                    ty: loop_var_ty.clone(),
                    kind: LocalKind::LoopVar,
                });
                self.locals.insert(loop_var.clone(), loop_var_local);

                // Transform init_bindings and allocate their locals
                let mir_init_bindings: Vec<(LocalId, ExprId)> = init_bindings
                    .iter()
                    .map(|(name, binding_ty, expr)| {
                        let expr_id = self.transform_term(expr, body);
                        let local_id = body.alloc_local(LocalDecl {
                            name: name.clone(),
                            span,
                            ty: binding_ty.clone(),
                            kind: LocalKind::Let,
                        });
                        self.locals.insert(name.clone(), local_id);
                        (local_id, expr_id)
                    })
                    .collect();

                // Transform loop kind
                let mir_kind = match kind {
                    TlcLoopKind::For { var, var_ty, iter } => {
                        let iter_id = self.transform_term(iter, body);
                        let var_local = body.alloc_local(LocalDecl {
                            name: var.clone(),
                            span,
                            ty: var_ty.clone(),
                            kind: LocalKind::Let,
                        });
                        self.locals.insert(var.clone(), var_local);
                        MirLoopKind::For {
                            var: var_local,
                            iter: iter_id,
                        }
                    }
                    TlcLoopKind::ForRange { var, var_ty, bound } => {
                        let bound_id = self.transform_term(bound, body);
                        let var_local = body.alloc_local(LocalDecl {
                            name: var.clone(),
                            span,
                            ty: var_ty.clone(),
                            kind: LocalKind::Let,
                        });
                        self.locals.insert(var.clone(), var_local);
                        MirLoopKind::ForRange {
                            var: var_local,
                            bound: bound_id,
                        }
                    }
                    TlcLoopKind::While { cond } => {
                        let cond_id = self.transform_term(cond, body);
                        MirLoopKind::While { cond: cond_id }
                    }
                };

                // Transform body
                let body_id = self.transform_term(loop_body, body);

                // Clean up locals
                self.locals.remove(loop_var);
                for (name, _, _) in init_bindings {
                    self.locals.remove(name);
                }
                match kind {
                    TlcLoopKind::For { var, .. } | TlcLoopKind::ForRange { var, .. } => {
                        self.locals.remove(var);
                    }
                    TlcLoopKind::While { .. } => {}
                }

                body.alloc_expr(
                    Expr::Loop {
                        loop_var: loop_var_local,
                        init: init_id,
                        init_bindings: mir_init_bindings,
                        kind: mir_kind,
                        body: body_id,
                    },
                    ty,
                    span,
                    node_id,
                )
            }

            TermKind::Lam { .. } => {
                // Lambdas should have been lifted - if we see one here, it's an error
                panic!(
                    "Unexpected lambda in TLC to MIR transformation at {:?}. \
                     All lambdas should have been lifted to top-level.",
                    span
                )
            }

            // Operators as values - these should always be applied
            TermKind::BinOp(_) | TermKind::UnOp(_) => {
                panic!(
                    "Unexpected bare operator in TLC to MIR transformation at {:?}. \
                     Operators should always be applied to arguments.",
                    span
                )
            }
        }
    }

    /// Collect the spine of a nested application chain.
    /// Given `App(App(App(f, a), b), c)`, returns `(base_term, [a, b, c])`.
    fn collect_application_spine<'t>(func: &'t Term, arg: &'t Term) -> (&'t Term, Vec<&'t Term>) {
        let mut args = vec![arg];
        let mut current = func;

        // Walk up the chain of App to collect all arguments
        loop {
            match &current.kind {
                TermKind::App {
                    func: inner_func,
                    arg: inner_arg,
                } => {
                    // Another application - add its arg and continue
                    args.push(inner_arg.as_ref());
                    current = inner_func.as_ref();
                }
                _ => {
                    // Not an App - this is the base
                    args.reverse();
                    return (current, args);
                }
            }
        }
    }

    /// Transform an application using spine-based collection.
    /// Collects all arguments from nested curried applications first,
    /// then emits a single Call or Closure (never intermediate closures).
    fn transform_app(
        &mut self,
        func: &Term,
        arg: &Term,
        ty: Type<TypeName>,
        span: Span,
        node_id: NodeId,
        body: &mut Body,
    ) -> ExprId {
        // Collect the full application spine first
        let (base_term, args) = Self::collect_application_spine(func, arg);

        match &base_term.kind {
            TermKind::BinOp(op) => {
                assert!(
                    args.len() == 2,
                    "BinOp requires exactly 2 arguments, got {}",
                    args.len()
                );
                let lhs_id = self.transform_term(args[0], body);
                let rhs_id = self.transform_term(args[1], body);
                body.alloc_expr(
                    Expr::BinOp {
                        op: op.op.clone(),
                        lhs: lhs_id,
                        rhs: rhs_id,
                    },
                    ty,
                    span,
                    node_id,
                )
            }

            TermKind::UnOp(op) => {
                assert!(
                    args.len() == 1,
                    "UnOp requires exactly 1 argument, got {}",
                    args.len()
                );
                let operand_id = self.transform_term(args[0], body);
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

            TermKind::Var(name) => self.transform_var_application(name, &args, ty, span, node_id, body),

            _ => {
                // The base is a computed function value (not a simple Var/BinOp/UnOp)
                self.transform_term_application(base_term, &args, ty, span, node_id, body)
            }
        }
    }

    /// Transform an application where the base is a Var (named function).
    /// Only handles arity checking for known defs; intrinsic lowering is in emit_call_or_intrinsic.
    fn transform_var_application(
        &mut self,
        name: &str,
        args: &[&Term],
        ty: Type<TypeName>,
        span: Span,
        node_id: NodeId,
        body: &mut Body,
    ) -> ExprId {
        // Transform all arguments
        let arg_ids: Vec<_> = args.iter().map(|a| self.transform_term(a, body)).collect();

        // Check if this is a known function with arity info
        if let Some(def) = self.top_level.get(name) {
            let arity = def.arity;
            if arg_ids.len() == arity {
                // Full call
                body.alloc_expr(
                    Expr::Call {
                        func: name.to_string(),
                        args: arg_ids,
                    },
                    ty,
                    span,
                    node_id,
                )
            } else if arg_ids.len() < arity {
                panic!(
                    "Partial application not supported: {} expects {} args, got {}. \
                     SOAC closure flattening should handle this at TLC level.",
                    name,
                    arity,
                    arg_ids.len()
                )
            } else {
                panic!(
                    "Too many arguments for function {}: expected {}, got {}",
                    name,
                    arity,
                    arg_ids.len()
                )
            }
        } else {
            // Intrinsic or unknown function
            self.emit_call_or_intrinsic(name, arg_ids, ty, span, node_id, body)
        }
    }

    /// Transform an application where the base is a Term (computed function value).
    fn transform_term_application(
        &mut self,
        term: &Term,
        args: &[&Term],
        ty: Type<TypeName>,
        span: Span,
        node_id: NodeId,
        body: &mut Body,
    ) -> ExprId {
        // Transform all arguments
        let arg_ids: Vec<_> = args.iter().map(|a| self.transform_term(a, body)).collect();

        // Transform the function term
        let func_id = self.transform_term(term, body);

        // After defunctionalization, all function applications should resolve to Global refs
        match body.get_expr(func_id).clone() {
            Expr::Global(func_name) => {
                self.emit_call_or_intrinsic(&func_name, arg_ids, ty, span, node_id, body)
            }
            other => {
                panic!(
                    "BUG: Non-global function application after defunctionalization: {:?}",
                    other
                );
            }
        }
    }

    /// Emit either an Intrinsic or Call expression based on function name.
    /// All intrinsic-to-MIR-construct lowering happens here.
    fn emit_call_or_intrinsic(
        &self,
        name: &str,
        args: Vec<ExprId>,
        ty: Type<TypeName>,
        span: Span,
        node_id: NodeId,
        body: &mut Body,
    ) -> ExprId {
        // Handle intrinsics that become MIR constructs
        match name {
            "_w_vec_lit" => {
                return body.alloc_expr(Expr::Vector(args), ty, span, node_id);
            }
            "_w_array_lit" => {
                let len = args.len();
                let size = body.alloc_expr(
                    Expr::Int(len.to_string()),
                    Type::Constructed(TypeName::Int(32), vec![]),
                    span,
                    node_id,
                );
                return body.alloc_expr(
                    Expr::Array {
                        backing: ArrayBacking::Literal(args),
                        size,
                    },
                    ty,
                    span,
                    node_id,
                );
            }
            "_w_tuple" => {
                return body.alloc_expr(Expr::Tuple(args), ty, span, node_id);
            }
            "_w_range" if args.len() == 3 => {
                let kind = self.extract_range_kind(body, args[2]);
                return body.alloc_expr(
                    Expr::Array {
                        backing: ArrayBacking::Range {
                            start: args[0],
                            step: None,
                            kind,
                        },
                        size: args[1],
                    },
                    ty,
                    span,
                    node_id,
                );
            }
            "_w_range_step" if args.len() == 4 => {
                let kind = self.extract_range_kind(body, args[3]);
                return body.alloc_expr(
                    Expr::Array {
                        backing: ArrayBacking::Range {
                            start: args[0],
                            step: Some(args[1]),
                            kind,
                        },
                        size: args[2],
                    },
                    ty,
                    span,
                    node_id,
                );
            }
            "_w_slice" if args.len() == 3 => {
                let size = body.alloc_expr(
                    Expr::BinOp {
                        op: "-".to_string(),
                        lhs: args[2], // end
                        rhs: args[1], // start
                    },
                    Type::Constructed(TypeName::Int(32), vec![]),
                    span,
                    node_id,
                );
                return body.alloc_expr(
                    Expr::Array {
                        backing: ArrayBacking::View {
                            base: args[0],
                            offset: args[1],
                        },
                        size,
                    },
                    ty,
                    span,
                    node_id,
                );
            }
            "_w_index" if args.len() == 2 => {
                let array_arg = args[0];
                let index_arg = args[1];

                // Check if index is a literal constant
                let is_const_index = matches!(body.get_expr(index_arg), Expr::Int(_));

                let final_array_arg = if is_const_index {
                    // Constant index - no materialization needed (will use OpCompositeExtract)
                    array_arg
                } else {
                    // Runtime index - wrap array in Materialize for potential hoisting
                    let array_ty = body.get_type(array_arg).clone();
                    let ptr_ty = Type::Constructed(
                        TypeName::Pointer,
                        vec![array_ty, Type::Constructed(TypeName::AddressFunction, vec![])],
                    );
                    body.alloc_expr(Expr::Materialize(array_arg), ptr_ty, span, node_id)
                };

                return body.alloc_expr(
                    Expr::Intrinsic {
                        name: "_w_index".to_string(),
                        args: vec![final_array_arg, index_arg],
                    },
                    ty,
                    span,
                    node_id,
                );
            }
            _ => {}
        }

        if name.starts_with("_w_") {
            // Intrinsic call
            body.alloc_expr(
                Expr::Intrinsic {
                    name: name.to_string(),
                    args,
                },
                ty,
                span,
                node_id,
            )
        } else {
            // Regular function call
            body.alloc_expr(
                Expr::Call {
                    func: name.to_string(),
                    args,
                },
                ty,
                span,
                node_id,
            )
        }
    }

    /// Extract RangeKind from an integer literal expression.
    fn extract_range_kind(&self, body: &Body, expr_id: ExprId) -> mir::RangeKind {
        if let Expr::Int(s) = body.get_expr(expr_id) {
            match s.as_str() {
                "0" => mir::RangeKind::Inclusive,
                "1" => mir::RangeKind::Exclusive,
                "2" => mir::RangeKind::ExclusiveLt,
                "3" => mir::RangeKind::ExclusiveGt,
                _ => mir::RangeKind::Exclusive, // Default
            }
        } else {
            mir::RangeKind::Exclusive // Default
        }
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
        let binop_ty = Type::Constructed(
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
        );
        let binop_term = Term {
            id: ids.next_id(),
            ty: binop_ty,
            span,
            kind: TermKind::BinOp(BinaryOp { op: "+".to_string() }),
        };

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
                func: Box::new(binop_term),
                arg: Box::new(x_var),
            },
        };

        let add_body = Term {
            id: ids.next_id(),
            ty: Type::Constructed(TypeName::Int(32), vec![]),
            span,
            kind: TermKind::App {
                func: Box::new(binop_x),
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
                meta: super::DefMeta::Function,
                arity: 2,
            }],
            uniforms: vec![],
            storage: vec![],
        };

        let schemes = std::collections::HashMap::new();
        let mir = TlcToMir::transform(&program, &schemes);

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
