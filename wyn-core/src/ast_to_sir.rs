//! AST to SIR lowering pass.
//!
//! Converts typed AST to SIR (SOAC Intermediate Representation).
//! This pass:
//! - Recognizes SOAC patterns (map, reduce, scan, iota, replicate)
//! - Computes lambda captures via free variable analysis
//! - Converts let bindings to statement sequences

use std::collections::{HashMap, HashSet};

use crate::ast::{self, ExprKind, Expression, NodeId, PatternKind, Span, TypeName};
use crate::error::Result;
use crate::sir::builder::SirBuilder;
use crate::sir::types::{AssocInfo, ScalarTy, Size};
use crate::sir::{
    self, Body, Def, EntryInput, EntryOutput, ExecutionModel, Exp, IoDecoration, Lambda, Map,
    ParallelizationConfig, Param, Pat, Prim, Reduce, Scan, SirType, Soac, Statement, VarId,
};
use crate::types::TypeScheme;
use crate::{bail_flatten, err_parse_at};
use polytype::Type;

/// Lowers typed AST to SIR.
pub struct AstToSir {
    /// Builder for SIR construction.
    builder: SirBuilder,
    /// Type table from type checking.
    type_table: HashMap<NodeId, TypeScheme>,
    /// Mapping from variable names to VarIds in current scope.
    scope: Vec<HashMap<String, VarId>>,
    /// Set of builtin names (not captured as free variables).
    builtins: HashSet<String>,
    /// Generated definitions.
    defs: Vec<Def>,
    /// Current statement accumulator.
    current_stms: Vec<Statement>,
    /// Lambda registry: stores all lambdas for defunctionalization.
    lambdas: HashMap<sir::LambdaId, Lambda>,
}

impl AstToSir {
    pub fn new(type_table: HashMap<NodeId, TypeScheme>, builtins: HashSet<String>) -> Self {
        AstToSir {
            builder: SirBuilder::new(),
            type_table,
            scope: vec![HashMap::new()],
            builtins,
            defs: Vec::new(),
            current_stms: Vec::new(),
            lambdas: HashMap::new(),
        }
    }

    /// Lower an entire program.
    pub fn lower_program(mut self, program: &ast::Program) -> Result<sir::Program> {
        for decl in &program.declarations {
            self.lower_declaration(decl)?;
        }
        Ok(sir::Program {
            defs: self.defs,
            lambdas: self.lambdas,
        })
    }

    fn lower_declaration(&mut self, decl: &ast::Declaration) -> Result<()> {
        match decl {
            ast::Declaration::Decl(d) => self.lower_decl(d),
            ast::Declaration::Entry(e) => self.lower_entry(e),
            ast::Declaration::Uniform(u) => self.lower_uniform(u),
            ast::Declaration::Storage(s) => self.lower_storage(s),
            // Module and imports are handled earlier in the pipeline
            ast::Declaration::Module(_)
            | ast::Declaration::Import(_)
            | ast::Declaration::ModuleTypeBind(_)
            | ast::Declaration::Sig(_)
            | ast::Declaration::TypeBind(_)
            | ast::Declaration::Open(_) => Ok(()),
        }
    }

    fn lower_decl(&mut self, decl: &ast::Decl) -> Result<()> {
        self.push_scope();

        // Create parameters from patterns
        let params: Vec<_> =
            decl.params.iter().map(|p| self.lower_pattern_to_param(p)).collect::<Result<_>>()?;

        // Add params to scope
        for param in &params {
            self.bind_var(&param.name_hint, param.var);
        }

        // Lower body
        let body = self.lower_expr_to_body(&decl.body)?;
        let ret_ty = self.get_expr_type(&decl.body);

        self.pop_scope();

        // Use the body's NodeId for source tracing
        let id = decl.body.h.id;
        let span = decl.body.h.span;

        if params.is_empty() {
            // Constant
            self.defs.push(Def::Constant {
                id,
                name: decl.name.clone(),
                ty: ret_ty,
                body,
                span,
            });
        } else {
            // Function
            self.defs.push(Def::Function {
                id,
                name: decl.name.clone(),
                params,
                ret_ty,
                body,
                span,
            });
        }

        Ok(())
    }

    fn lower_entry(&mut self, entry: &ast::EntryDecl) -> Result<()> {
        self.push_scope();

        // Determine execution model from entry_type attribute
        let execution_model = match &entry.entry_type {
            ast::Attribute::Vertex => ExecutionModel::Vertex,
            ast::Attribute::Fragment => ExecutionModel::Fragment,
            ast::Attribute::Compute(explicit_size) => {
                // Use explicit size if provided, otherwise derive from size hints
                let local_size = match explicit_size {
                    Some(size) => *size,
                    None => {
                        // Extract size hints from params and use the largest for derivation
                        let size_hint = Self::extract_max_size_hint(&entry.params);
                        ParallelizationConfig::default().derive_workgroup_size(size_hint)
                    }
                };
                ExecutionModel::Compute { local_size }
            }
            _ => bail_flatten!("Invalid entry type attribute"),
        };

        // Create inputs from params
        let mut inputs = Vec::new();
        for param in &entry.params {
            let p = self.lower_pattern_to_param(param)?;
            self.bind_var(&p.name_hint, p.var);
            inputs.push(EntryInput {
                var: p.var,
                name: p.name_hint.clone(),
                ty: p.ty.clone(),
                decoration: None, // TODO: extract from pattern attributes
            });
        }

        // Lower body
        let body = self.lower_expr_to_body(&entry.body)?;

        // Extract outputs - get types from the type table (body type was unified with outputs)
        let body_type = self.type_table.get(&entry.body.h.id)
            .map(|scheme| self.scheme_to_type(scheme))
            .expect("Entry body should have type in type_table");

        let output_types: Vec<SirType> = if entry.outputs.len() == 1 {
            vec![body_type]
        } else {
            // Multiple outputs: body type is a tuple
            match body_type {
                Type::Constructed(TypeName::Tuple(_), args) => args,
                _ => panic!("Multiple outputs but body type is not a tuple: {:?}", body_type),
            }
        };

        let outputs: Vec<_> = entry
            .outputs
            .iter()
            .zip(output_types)
            .map(|(o, ty)| {
                let decoration = o.attribute.as_ref().and_then(|a| match a {
                    ast::Attribute::BuiltIn(b) => Some(IoDecoration::BuiltIn(*b)),
                    ast::Attribute::Location(l) => Some(IoDecoration::Location(*l)),
                    _ => None,
                });
                EntryOutput { ty, decoration }
            })
            .collect();

        self.pop_scope();

        // Use the body's NodeId for source tracing
        let id = entry.body.h.id;
        let span = entry.body.h.span;

        self.defs.push(Def::EntryPoint {
            id,
            name: entry.name.clone(),
            execution_model,
            inputs,
            outputs,
            body,
            span,
        });

        Ok(())
    }

    fn lower_uniform(&mut self, uniform: &ast::UniformDecl) -> Result<()> {
        let ty = self.ast_type_to_sir(&uniform.ty);
        // Uniforms are declarations without expressions, so no meaningful NodeId exists.
        // Use a placeholder - this is acceptable since uniforms don't generate executable code.
        let id = NodeId::new(0);
        self.defs.push(Def::Uniform {
            id,
            name: uniform.name.clone(),
            ty,
            set: uniform.set,
            binding: uniform.binding,
        });
        Ok(())
    }

    fn lower_storage(&mut self, storage: &ast::StorageDecl) -> Result<()> {
        let ty = self.ast_type_to_sir(&storage.ty);
        // Storage buffers are declarations without expressions, so no meaningful NodeId exists.
        // Use a placeholder - this is acceptable since storage decls don't generate executable code.
        let id = NodeId::new(0);
        self.defs.push(Def::Storage {
            id,
            name: storage.name.clone(),
            ty,
            set: storage.set,
            binding: storage.binding,
        });
        Ok(())
    }

    // =========================================================================
    // Expression Lowering
    // =========================================================================

    /// Lower an expression to a Body (sequence of statements + result).
    fn lower_expr_to_body(&mut self, expr: &Expression) -> Result<Body> {
        let old_stms = std::mem::take(&mut self.current_stms);
        let result_var = self.lower_expr(expr)?;
        let stms = std::mem::replace(&mut self.current_stms, old_stms);
        Ok(Body {
            statements: stms,
            result: vec![result_var],
        })
    }

    /// Lower a lambda body with tuple pattern destructuring.
    fn lower_lambda_body(
        &mut self,
        patterns: &[ast::Pattern],
        params: &[Param],
        body_expr: &Expression,
    ) -> Result<Body> {
        let old_stms = std::mem::take(&mut self.current_stms);

        // Emit tuple destructuring for tuple pattern parameters
        for (pattern, param) in patterns.iter().zip(params.iter()) {
            if matches!(self.get_inner_pattern_kind(pattern), PatternKind::Tuple(_)) {
                self.bind_pattern(pattern, param.var)?;
            }
        }

        // Lower the body expression
        let result_var = self.lower_expr(body_expr)?;
        let stms = std::mem::replace(&mut self.current_stms, old_stms);
        Ok(Body {
            statements: stms,
            result: vec![result_var],
        })
    }

    /// Lower an expression, returning the VarId of the result.
    /// May emit statements to current_stms.
    fn lower_expr(&mut self, expr: &Expression) -> Result<VarId> {
        let span = expr.h.span;
        let ty = self.get_expr_type(expr);

        match &expr.kind {
            ExprKind::IntLiteral(n) => {
                let val: i64 = n
                    .as_str()
                    .parse()
                    .map_err(|_| err_parse_at!(span, "Invalid integer literal: {}", n.as_str()))?;
                self.emit_stm(Exp::Prim(Prim::ConstI32(val as i32)), ty, span)
            }

            ExprKind::FloatLiteral(f) => self.emit_stm(Exp::Prim(Prim::ConstF32(*f)), ty, span),

            ExprKind::BoolLiteral(b) => self.emit_stm(Exp::Prim(Prim::ConstBool(*b)), ty, span),

            ExprKind::Unit => {
                // Unit value - emit as intrinsic so VarId is registered
                self.emit_stm(
                    Exp::Prim(Prim::Intrinsic {
                        name: "__unit".to_string(),
                        args: vec![],
                    }),
                    ty,
                    span,
                )
            }

            ExprKind::StringLiteral(_s) => {
                // Strings not directly supported in SIR yet
                bail_flatten!("String literals not yet supported in SIR")
            }

            ExprKind::Identifier(quals, name) => {
                let full_name =
                    if quals.is_empty() { name.clone() } else { format!("{}.{}", quals.join("."), name) };

                // Check if it's a local variable
                if let Some(var) = self.lookup_var(&full_name) {
                    Ok(var)
                } else {
                    // Global reference - emit as Apply with no args
                    self.emit_stm(
                        Exp::Apply {
                            func: full_name,
                            args: vec![],
                        },
                        ty,
                        span,
                    )
                }
            }

            ExprKind::BinaryOp(op, lhs, rhs) => {
                let lhs_var = self.lower_expr(lhs)?;
                let rhs_var = self.lower_expr(rhs)?;
                let prim = self.binop_to_prim(&op.op, lhs_var, rhs_var);
                self.emit_stm(Exp::Prim(prim), ty, span)
            }

            ExprKind::UnaryOp(op, operand) => {
                let operand_var = self.lower_expr(operand)?;
                let prim = self.unaryop_to_prim(&op.op, operand_var);
                self.emit_stm(Exp::Prim(prim), ty, span)
            }

            ExprKind::If(if_expr) => {
                let cond_var = self.lower_expr(&if_expr.condition)?;
                let then_body = self.lower_expr_to_body(&if_expr.then_branch)?;
                let else_body = self.lower_expr_to_body(&if_expr.else_branch)?;
                self.emit_stm(
                    Exp::If {
                        cond: cond_var,
                        then_body,
                        else_body,
                    },
                    ty,
                    span,
                )
            }

            ExprKind::LetIn(let_in) => self.lower_let_in(let_in, ty, span),

            ExprKind::Lambda(lambda) => self.lower_lambda(lambda, ty, span),

            ExprKind::Application(func, args) => self.lower_application(func, args, ty, span),

            ExprKind::Tuple(elems) => {
                let elem_vars: Vec<_> = elems.iter().map(|e| self.lower_expr(e)).collect::<Result<_>>()?;
                self.emit_stm(Exp::Tuple(elem_vars), ty, span)
            }

            ExprKind::ArrayLiteral(elems) => {
                let elem_vars: Vec<_> = elems.iter().map(|e| self.lower_expr(e)).collect::<Result<_>>()?;
                self.emit_stm(
                    Exp::Prim(Prim::Intrinsic {
                        name: "array_literal".to_string(),
                        args: elem_vars,
                    }),
                    ty,
                    span,
                )
            }

            ExprKind::ArrayIndex(arr, idx) => {
                let arr_var = self.lower_expr(arr)?;
                let idx_var = self.lower_expr(idx)?;
                self.emit_stm(
                    Exp::Prim(Prim::Index {
                        arr: arr_var,
                        idx: idx_var,
                    }),
                    ty,
                    span,
                )
            }

            ExprKind::FieldAccess(base_expr, field) => {
                let base_var = self.lower_expr(base_expr)?;
                let base_ty = self.get_expr_type(base_expr);
                // Check if it's a tuple field access (numeric index)
                if let Ok(index) = field.parse::<usize>() {
                    self.emit_stm(
                        Exp::TupleProj {
                            tuple: base_var,
                            index,
                        },
                        ty,
                        span,
                    )
                } else if matches!(base_ty, Type::Constructed(TypeName::Tuple(_), _)) {
                    // Tuple with named field - convert to index
                    let index = self.field_to_index(&base_ty, field);
                    self.emit_stm(
                        Exp::TupleProj {
                            tuple: base_var,
                            index,
                        },
                        ty,
                        span,
                    )
                } else {
                    // Record/struct field access
                    self.emit_stm(
                        Exp::Prim(Prim::Intrinsic {
                            name: format!("__field_{}", field),
                            args: vec![base_var],
                        }),
                        ty,
                        span,
                    )
                }
            }

            ExprKind::Match(m) => {
                // For now, emit as intrinsic - proper match lowering is complex
                let scrutinee_var = self.lower_expr(&m.scrutinee)?;
                self.emit_stm(
                    Exp::Prim(Prim::Intrinsic {
                        name: "__match".to_string(),
                        args: vec![scrutinee_var],
                    }),
                    ty,
                    span,
                )
            }

            ExprKind::Loop(l) => {
                // Lower loop - for now emit as intrinsic
                let init_args =
                    if let Some(init_expr) = &l.init { vec![self.lower_expr(init_expr)?] } else { vec![] };
                self.emit_stm(
                    Exp::Prim(Prim::Intrinsic {
                        name: "__loop".to_string(),
                        args: init_args,
                    }),
                    ty,
                    span,
                )
            }

            ExprKind::VecMatLiteral(elems) => {
                let elem_vars: Vec<_> = elems.iter().map(|e| self.lower_expr(e)).collect::<Result<_>>()?;
                self.emit_stm(
                    Exp::Prim(Prim::Intrinsic {
                        name: "vec_literal".to_string(),
                        args: elem_vars,
                    }),
                    ty,
                    span,
                )
            }

            ExprKind::RecordLiteral(fields) => {
                let elem_vars: Vec<_> =
                    fields.iter().map(|(_, e)| self.lower_expr(e)).collect::<Result<_>>()?;
                // Emit as intrinsic for now (could add Exp::Record later)
                self.emit_stm(
                    Exp::Prim(Prim::Intrinsic {
                        name: "__record".to_string(),
                        args: elem_vars,
                    }),
                    ty,
                    span,
                )
            }

            ExprKind::ArrayWith { array, index, value } => {
                let arr_var = self.lower_expr(array)?;
                let idx_var = self.lower_expr(index)?;
                let val_var = self.lower_expr(value)?;
                self.emit_stm(
                    Exp::Prim(Prim::Intrinsic {
                        name: "_w_array_with".to_string(),
                        args: vec![arr_var, idx_var, val_var],
                    }),
                    ty,
                    span,
                )
            }

            ExprKind::TypeAscription(inner, _) => self.lower_expr(inner),

            ExprKind::TypeCoercion(inner, _) => self.lower_expr(inner),

            ExprKind::Range(r) => {
                let start_var = self.lower_expr(&r.start)?;
                let end_var = self.lower_expr(&r.end)?;
                let intrinsic_name = match r.kind {
                    ast::RangeKind::Inclusive => "__range_inclusive",
                    ast::RangeKind::Exclusive => "__range",
                    ast::RangeKind::ExclusiveLt => "__range",
                    ast::RangeKind::ExclusiveGt => "__range_gt",
                };
                self.emit_stm(
                    Exp::Prim(Prim::Intrinsic {
                        name: intrinsic_name.to_string(),
                        args: vec![start_var, end_var],
                    }),
                    ty,
                    span,
                )
            }

            ExprKind::Slice(s) => {
                let arr_var = self.lower_expr(&s.array)?;
                let start_var = s.start.as_ref().map(|e| self.lower_expr(e)).transpose()?;
                let end_var = s.end.as_ref().map(|e| self.lower_expr(e)).transpose()?;

                // Use explicit intrinsic names to avoid ambiguity
                let (name, args) = match (start_var, end_var) {
                    (Some(start), Some(end)) => ("__slice_range", vec![arr_var, start, end]),
                    (Some(start), None) => ("__slice_from", vec![arr_var, start]),
                    (None, Some(end)) => ("__slice_to", vec![arr_var, end]),
                    (None, None) => ("__slice_full", vec![arr_var]),
                };

                self.emit_stm(
                    Exp::Prim(Prim::Intrinsic {
                        name: name.to_string(),
                        args,
                    }),
                    ty,
                    span,
                )
            }

            ExprKind::TypeHole => {
                bail_flatten!("Type holes not allowed in SIR lowering")
            }
        }
    }

    /// Lower a let-in expression.
    fn lower_let_in(&mut self, let_in: &ast::LetInExpr, _ty: SirType, _span: Span) -> Result<VarId> {
        // Lower the RHS
        let rhs_var = self.lower_expr(&let_in.value)?;

        // If pattern is a simple name, update the name hint of the defining statement
        if let PatternKind::Name(name) = &let_in.pattern.kind {
            self.update_name_hint(rhs_var, name.clone());
        } else if let PatternKind::Typed(inner, _) = &let_in.pattern.kind {
            if let PatternKind::Name(name) = &inner.kind {
                self.update_name_hint(rhs_var, name.clone());
            }
        }

        // Bind pattern
        self.push_scope();
        self.bind_pattern(&let_in.pattern, rhs_var)?;

        // Lower body
        let result = self.lower_expr(&let_in.body)?;
        self.pop_scope();

        Ok(result)
    }

    /// Update the name hint of the statement that defines a variable.
    fn update_name_hint(&mut self, var: VarId, name: String) {
        for stm in self.current_stms.iter_mut().rev() {
            if let Some(single_var) = stm.pat.single_var() {
                if single_var == var {
                    stm.pat.binds[0].name_hint = name;
                    return;
                }
            }
        }
    }

    /// Lower a lambda expression.
    fn lower_lambda(&mut self, lambda: &ast::LambdaExpr, ty: SirType, span: Span) -> Result<VarId> {
        self.push_scope();

        // Create parameters from patterns
        let params: Vec<_> =
            lambda.params.iter().map(|p| self.lower_pattern_to_param(p)).collect::<Result<_>>()?;

        // Bind parameter names (simple binding, no statements emitted yet)
        for param in &params {
            self.bind_var(&param.name_hint, param.var);
        }

        // Compute captures (free variables in lambda body)
        let captures = self.compute_captures(&lambda.body, &params);

        // Lower body with tuple pattern destructuring
        let body = self.lower_lambda_body(&lambda.params, &params, &lambda.body)?;

        // Extract return type
        let ret_tys = vec![self.get_expr_type(&lambda.body)];

        self.pop_scope();

        // Create lambda
        let lam = Lambda {
            id: self.builder.fresh_lambda(),
            params,
            captures: captures.clone(),
            body,
            ret_tys,
            span,
        };

        // Register lambda for defunctionalization
        let lambda_id = lam.id;
        self.lambdas.insert(lambda_id, lam);

        // Emit lambda as intrinsic - will be handled in SIR→MIR
        self.emit_stm(
            Exp::Prim(Prim::Intrinsic {
                name: format!("__lambda_{}", lambda_id.0),
                args: captures,
            }),
            ty,
            span,
        )
    }

    /// Lower a function application.
    fn lower_application(
        &mut self,
        func: &Expression,
        args: &[Expression],
        ty: SirType,
        span: Span,
    ) -> Result<VarId> {
        // Check if this is a SOAC call
        if let Some(soac) = self.try_lower_soac(func, args, &ty, span)? {
            return self.emit_stm(Exp::Op(sir::Op::Soac(soac)), ty, span);
        }

        // Regular function call
        let arg_vars: Vec<_> = args.iter().map(|a| self.lower_expr(a)).collect::<Result<_>>()?;

        // Get function name
        let func_name = match &func.kind {
            ExprKind::Identifier(quals, name) => {
                if quals.is_empty() {
                    // Desugar overloaded function names based on argument types
                    self.desugar_function_name(name, args)
                } else {
                    format!("{}.{}", quals.join("."), name)
                }
            }
            _ => {
                // Higher-order function call
                let func_var = self.lower_expr(func)?;
                return self.emit_stm(
                    Exp::Prim(Prim::Intrinsic {
                        name: "__apply".to_string(),
                        args: std::iter::once(func_var).chain(arg_vars).collect(),
                    }),
                    ty,
                    span,
                );
            }
        };

        self.emit_stm(
            Exp::Apply {
                func: func_name,
                args: arg_vars,
            },
            ty,
            span,
        )
    }

    /// Try to recognize and lower a SOAC call.
    fn try_lower_soac(
        &mut self,
        func: &Expression,
        args: &[Expression],
        ty: &SirType,
        span: Span,
    ) -> Result<Option<Soac>> {
        let func_name = match &func.kind {
            ExprKind::Identifier(quals, name) if quals.is_empty() => name.as_str(),
            _ => return Ok(None),
        };

        match func_name {
            "map" if args.len() == 2 => {
                let f = self.lower_lambda_arg(&args[0], span)?;
                let arr_var = self.lower_expr(&args[1])?;
                let w = self.infer_array_size(&args[1]);
                Ok(Some(Soac::Map(Map {
                    w,
                    f,
                    arrs: vec![arr_var],
                })))
            }

            "reduce" if args.len() == 3 => {
                let f = self.lower_lambda_arg(&args[0], span)?;
                let neutral_var = self.lower_expr(&args[1])?;
                let arr_var = self.lower_expr(&args[2])?;
                let w = self.infer_array_size(&args[2]);
                Ok(Some(Soac::Reduce(Reduce {
                    w,
                    f,
                    neutral: neutral_var,
                    arr: arr_var,
                    assoc: AssocInfo::unknown(),
                })))
            }

            "scan" if args.len() == 3 => {
                let f = self.lower_lambda_arg(&args[0], span)?;
                let neutral_var = self.lower_expr(&args[1])?;
                let arr_var = self.lower_expr(&args[2])?;
                let w = self.infer_array_size(&args[2]);
                Ok(Some(Soac::Scan(Scan {
                    w,
                    f,
                    neutral: neutral_var,
                    arr: arr_var,
                    assoc: AssocInfo::unknown(),
                })))
            }

            "iota" if args.len() == 1 => {
                let n = self.lower_expr_to_size(&args[0])?;
                let elem_ty = self.infer_iota_elem_type(ty);
                Ok(Some(Soac::Iota { n, elem_ty }))
            }

            "replicate" if args.len() == 2 => {
                let n = self.lower_expr_to_size(&args[0])?;
                let value_var = self.lower_expr(&args[1])?;
                Ok(Some(Soac::Replicate { n, value: value_var }))
            }

            _ => Ok(None),
        }
    }

    /// Lower a lambda argument (either inline lambda or function reference).
    fn lower_lambda_arg(&mut self, expr: &Expression, span: Span) -> Result<Lambda> {
        match &expr.kind {
            ExprKind::Lambda(lambda) => {
                self.push_scope();

                let params: Vec<_> =
                    lambda.params.iter().map(|p| self.lower_pattern_to_param(p)).collect::<Result<_>>()?;

                for param in &params {
                    self.bind_var(&param.name_hint, param.var);
                }

                let captures = self.compute_captures(&lambda.body, &params);
                let body = self.lower_expr_to_body(&lambda.body)?;
                let ret_tys = vec![self.get_expr_type(&lambda.body)];

                self.pop_scope();

                Ok(Lambda {
                    id: self.builder.fresh_lambda(),
                    params,
                    captures,
                    body,
                    ret_tys,
                    span,
                })
            }

            ExprKind::Identifier(quals, name) => {
                // Function reference - create a lambda that calls it
                let full_name =
                    if quals.is_empty() { name.clone() } else { format!("{}.{}", quals.join("."), name) };

                // Infer param types from function type
                let func_ty = self.get_expr_type(expr);
                let (param_tys, ret_ty) = self.extract_func_type(&func_ty);

                let params: Vec<_> = param_tys
                    .into_iter()
                    .enumerate()
                    .map(|(i, ty)| Param {
                        name_hint: format!("_arg{}", i),
                        var: self.builder.fresh_var(),
                        ty,
                        span,
                    })
                    .collect();

                let param_vars: Vec<_> = params.iter().map(|p| p.var).collect();

                // Create body: call the function with params
                let call_var = self.builder.fresh_var();
                let call_stm = Statement {
                    id: self.builder.fresh_stm(),
                    pat: Pat::single(call_var, ret_ty.clone(), "_result".to_string()),
                    exp: Exp::Apply {
                        func: full_name,
                        args: param_vars,
                    },
                    ty: ret_ty.clone(),
                    span,
                };

                Ok(Lambda {
                    id: self.builder.fresh_lambda(),
                    params,
                    captures: vec![],
                    body: Body {
                        statements: vec![call_stm],
                        result: vec![call_var],
                    },
                    ret_tys: vec![ret_ty],
                    span,
                })
            }

            _ => bail_flatten!("Expected lambda or function name, got {:?}", expr.kind),
        }
    }

    // =========================================================================
    // Helper Methods
    // =========================================================================

    fn emit_stm(&mut self, exp: Exp, ty: SirType, span: Span) -> Result<VarId> {
        let var = self.builder.fresh_var();
        let pat = Pat::single(var, ty.clone(), format!("_w_{}", var.0));
        let stm = Statement {
            id: self.builder.fresh_stm(),
            pat,
            exp,
            ty,
            span,
        };
        self.current_stms.push(stm);
        Ok(var)
    }

    fn push_scope(&mut self) {
        self.scope.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scope.pop();
    }

    fn bind_var(&mut self, name: &str, var: VarId) {
        if let Some(scope) = self.scope.last_mut() {
            scope.insert(name.to_string(), var);
        }
    }

    fn lookup_var(&self, name: &str) -> Option<VarId> {
        for scope in self.scope.iter().rev() {
            if let Some(var) = scope.get(name) {
                return Some(*var);
            }
        }
        None
    }

    /// Desugar overloaded function names based on argument types.
    fn desugar_function_name(&self, name: &str, args: &[Expression]) -> String {
        match name {
            "mul" if args.len() == 2 => {
                let arg1_ty = self.get_expr_type(&args[0]);
                let arg2_ty = self.get_expr_type(&args[1]);
                let is_mat = |ty: &SirType| matches!(ty, Type::Constructed(TypeName::Mat, _));
                let is_vec = |ty: &SirType| matches!(ty, Type::Constructed(TypeName::Vec, _));
                if is_mat(&arg1_ty) && is_mat(&arg2_ty) {
                    "mul_mat_mat".to_string()
                } else if is_mat(&arg1_ty) && is_vec(&arg2_ty) {
                    "mul_mat_vec".to_string()
                } else if is_vec(&arg1_ty) && is_mat(&arg2_ty) {
                    "mul_vec_mat".to_string()
                } else {
                    name.to_string()
                }
            }
            _ => name.to_string(),
        }
    }

    fn bind_pattern(&mut self, pattern: &ast::Pattern, var: VarId) -> Result<()> {
        match &pattern.kind {
            PatternKind::Name(name) => {
                self.bind_var(name, var);
                Ok(())
            }
            PatternKind::Wildcard => Ok(()),
            PatternKind::Typed(inner, _) => self.bind_pattern(inner, var),
            PatternKind::Tuple(patterns) => {
                // Emit TupleProj for each element
                for (i, pat) in patterns.iter().enumerate() {
                    let proj_ty = self.get_pattern_type(pat);
                    let proj_var =
                        self.emit_stm(Exp::TupleProj { tuple: var, index: i }, proj_ty, pattern.h.span)?;
                    self.bind_pattern(pat, proj_var)?;
                }
                Ok(())
            }
            _ => bail_flatten!("Pattern kind {:?} not yet supported", pattern.kind),
        }
    }

    fn lower_pattern_to_param(&mut self, pattern: &ast::Pattern) -> Result<Param> {
        let name = self.extract_pattern_name(pattern)?;
        let ty = self.get_pattern_type(pattern);
        let var = self.builder.fresh_var();
        Ok(Param {
            name_hint: name,
            var,
            ty,
            span: pattern.h.span,
        })
    }

    fn extract_pattern_name(&self, pattern: &ast::Pattern) -> Result<String> {
        match &pattern.kind {
            PatternKind::Name(name) => Ok(name.clone()),
            PatternKind::Typed(inner, _) => self.extract_pattern_name(inner),
            PatternKind::Attributed(_, inner) => self.extract_pattern_name(inner),
            PatternKind::Wildcard => Ok("_".to_string()),
            PatternKind::Tuple(_) => Ok("_tuple".to_string()),
            PatternKind::Unit => Ok("_unit".to_string()),
            _ => bail_flatten!("Complex parameter patterns not yet supported: {:?}", pattern.kind),
        }
    }

    /// Get the inner pattern kind, stripping Typed and Attributed wrappers.
    fn get_inner_pattern_kind<'a>(&self, pattern: &'a ast::Pattern) -> &'a PatternKind {
        match &pattern.kind {
            PatternKind::Typed(inner, _) => self.get_inner_pattern_kind(inner),
            PatternKind::Attributed(_, inner) => self.get_inner_pattern_kind(inner),
            other => other,
        }
    }

    /// Extract size hints from patterns and return the maximum.
    ///
    /// Used to derive workgroup size for compute shaders when not explicitly specified.
    fn extract_max_size_hint(params: &[ast::Pattern]) -> Option<u32> {
        let mut max_hint = None;

        for param in params {
            if let Some(hint) = Self::extract_size_hint_from_pattern(param) {
                max_hint = Some(max_hint.map_or(hint, |prev: u32| prev.max(hint)));
            }
        }

        max_hint
    }

    /// Extract size hint from a single pattern, recursively checking wrappers.
    fn extract_size_hint_from_pattern(pattern: &ast::Pattern) -> Option<u32> {
        match &pattern.kind {
            PatternKind::Typed(inner, _) => Self::extract_size_hint_from_pattern(inner),
            PatternKind::Attributed(attrs, inner) => {
                // Check for SizeHint in this pattern's attributes
                let hint_here = attrs.iter().find_map(|attr| {
                    if let ast::Attribute::SizeHint(n) = attr { Some(*n) } else { None }
                });
                // If not found here, check inner pattern
                hint_here.or_else(|| Self::extract_size_hint_from_pattern(inner))
            }
            _ => None,
        }
    }

    fn get_expr_type(&self, expr: &Expression) -> SirType {
        self.type_table
            .get(&expr.h.id)
            .map(|scheme| self.scheme_to_type(scheme))
            .unwrap_or_else(|| panic!("No type found for expression {:?}", expr.h.id))
    }

    fn scheme_to_type(&self, scheme: &TypeScheme) -> SirType {
        match scheme {
            TypeScheme::Monotype(ty) => ty.clone(),
            TypeScheme::Polytype { body, .. } => self.scheme_to_type(body),
        }
    }

    fn get_pattern_type(&self, pattern: &ast::Pattern) -> SirType {
        self.type_table.get(&pattern.h.id).map(|scheme| self.scheme_to_type(scheme)).unwrap_or_else(|| {
            // For typed patterns, extract from the annotation
            if let PatternKind::Typed(_, ty) = &pattern.kind {
                return ty.clone();
            }
            panic!("No type found for pattern {:?}", pattern.h.id)
        })
    }

    fn ast_type_to_sir(&self, ty: &ast::Type) -> SirType {
        // Reuse the polytype type directly
        ty.clone()
    }

    fn compute_captures(&self, expr: &Expression, params: &[Param]) -> Vec<VarId> {
        // Collect all free variable names from the expression
        let mut free_names = HashSet::new();
        let mut bound = HashSet::new();

        // Lambda parameters are bound
        for p in params {
            bound.insert(p.name_hint.clone());
        }

        self.collect_free_vars(expr, &mut bound, &mut free_names);

        // Look up VarIds for free names (excluding builtins)
        let mut captures = Vec::new();
        for name in free_names {
            if !self.builtins.contains(&name) {
                if let Some(var) = self.lookup_var(&name) {
                    captures.push(var);
                }
            }
        }
        captures
    }

    /// Recursively collect free variable names from an AST expression.
    fn collect_free_vars(
        &self,
        expr: &Expression,
        bound: &mut HashSet<String>,
        free: &mut HashSet<String>,
    ) {
        match &expr.kind {
            ExprKind::Identifier(_, name) => {
                if !bound.contains(name) {
                    free.insert(name.clone());
                }
            }
            ExprKind::IntLiteral(_)
            | ExprKind::FloatLiteral(_)
            | ExprKind::BoolLiteral(_)
            | ExprKind::StringLiteral(_)
            | ExprKind::Unit
            | ExprKind::TypeHole => {}
            ExprKind::ArrayLiteral(elems) | ExprKind::VecMatLiteral(elems) | ExprKind::Tuple(elems) => {
                for e in elems {
                    self.collect_free_vars(e, bound, free);
                }
            }
            ExprKind::ArrayIndex(arr, idx) => {
                self.collect_free_vars(arr, bound, free);
                self.collect_free_vars(idx, bound, free);
            }
            ExprKind::ArrayWith { array, index, value } => {
                self.collect_free_vars(array, bound, free);
                self.collect_free_vars(index, bound, free);
                self.collect_free_vars(value, bound, free);
            }
            ExprKind::BinaryOp(_, lhs, rhs) => {
                self.collect_free_vars(lhs, bound, free);
                self.collect_free_vars(rhs, bound, free);
            }
            ExprKind::UnaryOp(_, operand) => {
                self.collect_free_vars(operand, bound, free);
            }
            ExprKind::RecordLiteral(fields) => {
                for (_, e) in fields {
                    self.collect_free_vars(e, bound, free);
                }
            }
            ExprKind::Lambda(lambda) => {
                let mut inner_bound = bound.clone();
                for p in &lambda.params {
                    self.collect_pattern_names_into(p, &mut inner_bound);
                }
                self.collect_free_vars(&lambda.body, &mut inner_bound, free);
            }
            ExprKind::Application(func, args) => {
                self.collect_free_vars(func, bound, free);
                for a in args {
                    self.collect_free_vars(a, bound, free);
                }
            }
            ExprKind::LetIn(let_in) => {
                self.collect_free_vars(&let_in.value, bound, free);
                let mut inner_bound = bound.clone();
                self.collect_pattern_names_into(&let_in.pattern, &mut inner_bound);
                self.collect_free_vars(&let_in.body, &mut inner_bound, free);
            }
            ExprKind::FieldAccess(base, _) => {
                self.collect_free_vars(base, bound, free);
            }
            ExprKind::If(if_expr) => {
                self.collect_free_vars(&if_expr.condition, bound, free);
                self.collect_free_vars(&if_expr.then_branch, bound, free);
                self.collect_free_vars(&if_expr.else_branch, bound, free);
            }
            ExprKind::Slice(slice) => {
                self.collect_free_vars(&slice.array, bound, free);
                if let Some(start) = &slice.start {
                    self.collect_free_vars(start, bound, free);
                }
                if let Some(end) = &slice.end {
                    self.collect_free_vars(end, bound, free);
                }
            }
            ExprKind::Range(range) => {
                self.collect_free_vars(&range.start, bound, free);
                self.collect_free_vars(&range.end, bound, free);
            }
            ExprKind::Loop(loop_expr) => {
                if let Some(init) = &loop_expr.init {
                    self.collect_free_vars(init, bound, free);
                }
                let mut inner_bound = bound.clone();
                self.collect_pattern_names_into(&loop_expr.pattern, &mut inner_bound);
                match &loop_expr.form {
                    ast::LoopForm::For(_, expr) => self.collect_free_vars(expr, bound, free),
                    ast::LoopForm::ForIn(pat, expr) => {
                        self.collect_free_vars(expr, bound, free);
                        self.collect_pattern_names_into(pat, &mut inner_bound);
                    }
                    ast::LoopForm::While(cond) => self.collect_free_vars(cond, &mut inner_bound, free),
                }
                self.collect_free_vars(&loop_expr.body, &mut inner_bound, free);
            }
            ExprKind::Match(match_expr) => {
                self.collect_free_vars(&match_expr.scrutinee, bound, free);
                for case in &match_expr.cases {
                    let mut inner_bound = bound.clone();
                    self.collect_pattern_names_into(&case.pattern, &mut inner_bound);
                    self.collect_free_vars(&case.body, &mut inner_bound, free);
                }
            }
            ExprKind::TypeAscription(inner, _) | ExprKind::TypeCoercion(inner, _) => {
                self.collect_free_vars(inner, bound, free);
            }
        }
    }

    /// Collect all bound names from a pattern into the set.
    fn collect_pattern_names_into(&self, pattern: &ast::Pattern, bound: &mut HashSet<String>) {
        match &pattern.kind {
            PatternKind::Name(name) => {
                bound.insert(name.clone());
            }
            PatternKind::Typed(inner, _) => self.collect_pattern_names_into(inner, bound),
            PatternKind::Wildcard | PatternKind::Literal(_) | PatternKind::Unit => {}
            PatternKind::Tuple(pats) => {
                for p in pats {
                    self.collect_pattern_names_into(p, bound);
                }
            }
            PatternKind::Record(fields) => {
                for f in fields {
                    if let Some(pat) = &f.pattern {
                        self.collect_pattern_names_into(pat, bound);
                    } else {
                        bound.insert(f.field.clone());
                    }
                }
            }
            PatternKind::Constructor(_, pats) => {
                for p in pats {
                    self.collect_pattern_names_into(p, bound);
                }
            }
            PatternKind::Attributed(_, inner) => {
                self.collect_pattern_names_into(inner, bound);
            }
        }
    }

    fn binop_to_prim(&self, op: &str, lhs: VarId, rhs: VarId) -> Prim {
        match op {
            "+" => Prim::Add(lhs, rhs),
            "-" => Prim::Sub(lhs, rhs),
            "*" => Prim::Mul(lhs, rhs),
            "/" => Prim::Div(lhs, rhs),
            "%" => Prim::Mod(lhs, rhs),
            "==" => Prim::Eq(lhs, rhs),
            "!=" => Prim::Ne(lhs, rhs),
            "<" => Prim::Lt(lhs, rhs),
            "<=" => Prim::Le(lhs, rhs),
            ">" => Prim::Gt(lhs, rhs),
            ">=" => Prim::Ge(lhs, rhs),
            "&&" => Prim::And(lhs, rhs),
            "||" => Prim::Or(lhs, rhs),
            _ => Prim::Intrinsic {
                name: format!("binop_{}", op),
                args: vec![lhs, rhs],
            },
        }
    }

    fn unaryop_to_prim(&self, op: &str, operand: VarId) -> Prim {
        match op {
            "-" => Prim::Neg(operand),
            "!" => Prim::Not(operand),
            _ => Prim::Intrinsic {
                name: format!("unaryop_{}", op),
                args: vec![operand],
            },
        }
    }

    fn field_to_index(&self, ty: &SirType, field: &str) -> usize {
        // This is only called for named field access on Tuple types.
        // Currently, tuples in wyn are positional-only - named field access
        // should be rejected by type checking. If we reach here, it's a compiler bug.
        panic!(
            "Named field access '{}' on tuple type {:?} - should have been rejected by type checker",
            field, ty
        )
    }

    fn infer_array_size(&mut self, _expr: &Expression) -> Size {
        // TODO: implement size inference
        Size::Sym(self.builder.fresh_size_var())
    }

    fn lower_expr_to_size(&mut self, expr: &Expression) -> Result<Size> {
        match &expr.kind {
            ExprKind::IntLiteral(n) => {
                let val: u64 = n
                    .as_str()
                    .parse()
                    .map_err(|_| err_parse_at!(expr.h.span, "Invalid size literal: {}", n.as_str()))?;
                Ok(Size::Const(val))
            }
            _ => {
                // Symbolic size
                Ok(Size::Sym(self.builder.fresh_size_var()))
            }
        }
    }

    fn infer_iota_elem_type(&self, ty: &SirType) -> ScalarTy {
        match ty {
            Type::Constructed(TypeName::Array, args) if !args.is_empty() => self.type_to_scalar(&args[0]),
            _ => ScalarTy::I32,
        }
    }

    fn type_to_scalar(&self, ty: &SirType) -> ScalarTy {
        match ty {
            Type::Constructed(TypeName::Int(32), _) => ScalarTy::I32,
            Type::Constructed(TypeName::Int(64), _) => ScalarTy::I64,
            Type::Constructed(TypeName::UInt(32), _) => ScalarTy::U32,
            Type::Constructed(TypeName::UInt(64), _) => ScalarTy::U64,
            Type::Constructed(TypeName::Float(32), _) => ScalarTy::F32,
            Type::Constructed(TypeName::Float(64), _) => ScalarTy::F64,
            _ => ScalarTy::I32,
        }
    }

    fn extract_func_type(&self, ty: &SirType) -> (Vec<SirType>, SirType) {
        match ty {
            Type::Constructed(TypeName::Arrow, args) if args.len() == 2 => {
                let param_ty = args[0].clone();
                let ret_ty = args[1].clone();
                (vec![param_ty], ret_ty)
            }
            _ => (vec![], ty.clone()),
        }
    }
}

/// Lower an alias-checked program to SIR.
pub fn lower_to_sir(
    program: &ast::Program,
    type_table: HashMap<NodeId, TypeScheme>,
    builtins: HashSet<String>,
) -> Result<sir::Program> {
    let lowerer = AstToSir::new(type_table, builtins);
    lowerer.lower_program(program)
}
