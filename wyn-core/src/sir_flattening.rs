//! SIR Flattening pass: SIR -> MIR
//!
//! This pass performs:
//! - Defunctionalization: lambdas become top-level functions with closure records
//! - Pattern flattening: SIR patterns become simple let bindings
//! - Statement sequencing: SIR Body (stm list + result) becomes nested MIR Let expressions
//! - SOAC lowering: Map/Reduce/Scan become MIR calls or array constructs

use std::collections::{HashMap, HashSet};

use crate::ast::{NodeCounter, NodeId, Span, TypeName};
use crate::error::Result;
use crate::mir::{self, ArrayBacking, ExprId, LocalDecl, LocalId, LocalKind, RangeKind};
use crate::scope::ScopeStack;
use crate::sir::{self, Body, Def, Exp, Lambda, Prim, SirType, Soac, Stm, VarId};
// TypeScheme used in original flattening but not needed here yet
use crate::{bail_flatten, IdArena};
use polytype::Type;

/// Flattens SIR to MIR with defunctionalization.
pub struct SirFlattener {
    /// Counter for generating unique names
    next_id: usize,
    /// Counter for generating unique MIR node IDs
    node_counter: NodeCounter,
    /// Generated lambda functions (collected during flattening)
    generated_functions: Vec<mir::Def>,
    /// Lambda registry: all lambdas (source and synthesized)
    lambda_registry: IdArena<mir::LambdaId, mir::LambdaInfo>,
    /// Set of builtin names to exclude from free variable capture
    _builtins: HashSet<String>,
    /// Current MIR body being built
    current_body: mir::Body,
    /// Mapping from SIR VarId to MIR LocalId in current scope
    var_to_local: HashMap<VarId, LocalId>,
    /// Mapping from variable names to LocalIds (for generated code)
    name_to_local: ScopeStack<LocalId>,
    /// SIR lambdas available for defunctionalization
    sir_lambdas: HashMap<sir::LambdaId, Lambda>,
}

impl SirFlattener {
    pub fn new(builtins: HashSet<String>) -> Self {
        SirFlattener {
            next_id: 0,
            node_counter: NodeCounter::new(),
            generated_functions: Vec::new(),
            lambda_registry: IdArena::new(),
            _builtins: builtins,
            current_body: mir::Body::new(),
            var_to_local: HashMap::new(),
            name_to_local: ScopeStack::new(),
            sir_lambdas: HashMap::new(),
        }
    }

    /// Generate a fresh unique name.
    fn fresh_name(&mut self, prefix: &str) -> String {
        let name = format!("{}_{}", prefix, self.next_id);
        self.next_id += 1;
        name
    }

    /// Get a fresh NodeId.
    fn next_node_id(&mut self) -> NodeId {
        self.node_counter.next_id()
    }

    /// Start a new body. Returns the old body.
    fn begin_body(&mut self) -> mir::Body {
        std::mem::take(&mut self.current_body)
    }

    /// Finish the current body, returning it and restoring the old body.
    fn end_body(&mut self, old_body: mir::Body) -> mir::Body {
        std::mem::replace(&mut self.current_body, old_body)
    }

    /// Allocate a new local variable in the current body.
    fn alloc_local(&mut self, name: String, ty: Type<TypeName>, kind: LocalKind, span: Span) -> LocalId {
        let decl = LocalDecl {
            name: name.clone(),
            span,
            ty,
            kind,
        };
        let local_id = self.current_body.alloc_local(decl);
        self.name_to_local.insert(name, local_id);
        local_id
    }

    /// Allocate a new expression in the current body.
    fn alloc_expr(&mut self, expr: mir::Expr, ty: Type<TypeName>, span: Span) -> ExprId {
        let node_id = self.next_node_id();
        self.current_body.alloc_expr(expr, ty, span, node_id)
    }

    /// Flatten a complete SIR program to MIR.
    pub fn flatten_program(mut self, program: sir::Program) -> Result<mir::Program> {
        // Store SIR lambdas for defunctionalization
        self.sir_lambdas = program.lambdas;

        let mut defs = Vec::new();

        for def in program.defs {
            match def {
                Def::Function { id, name, params, ret_ty, body, span } => {
                    let mir_def = self.flatten_function(id, name, params, ret_ty, body, span)?;
                    defs.push(mir_def);
                }
                Def::EntryPoint { id, name, execution_model, inputs, outputs, body, span } => {
                    let mir_def = self.flatten_entry_point(id, name, execution_model, inputs, outputs, body, span)?;
                    defs.push(mir_def);
                }
                Def::Constant { id, name, ty, body, span } => {
                    let mir_def = self.flatten_constant(id, name, ty, body, span)?;
                    defs.push(mir_def);
                }
                Def::Uniform { id, name, ty, set, binding } => {
                    defs.push(mir::Def::Uniform { id, name, ty, set, binding });
                }
                Def::Storage { id, name, ty, set, binding } => {
                    // SIR Storage doesn't have layout/access, use defaults
                    defs.push(mir::Def::Storage {
                        id,
                        name,
                        ty,
                        set,
                        binding,
                        layout: crate::ast::StorageLayout::Std430,
                        access: crate::ast::StorageAccess::ReadWrite,
                    });
                }
            }
        }

        // Add generated lambda functions
        defs.extend(self.generated_functions);

        Ok(mir::Program {
            defs,
            lambda_registry: self.lambda_registry,
        })
    }

    fn flatten_function(
        &mut self,
        id: NodeId,
        name: String,
        params: Vec<sir::Param>,
        ret_ty: SirType,
        body: Body,
        span: Span,
    ) -> Result<mir::Def> {
        let old_body = self.begin_body();
        self.var_to_local.clear();
        self.name_to_local.push_scope();

        // Allocate parameters as locals
        let mut param_locals = Vec::new();
        for param in &params {
            let local_id = self.alloc_local(
                param.name_hint.clone(),
                param.ty.clone(),
                LocalKind::Param,
                param.span,
            );
            self.var_to_local.insert(param.var, local_id);
            param_locals.push(local_id);
        }

        // Flatten body
        let root_id = self.flatten_body(&body, span)?;
        self.current_body.set_root(root_id);

        self.name_to_local.pop_scope();
        let mir_body = self.end_body(old_body);

        Ok(mir::Def::Function {
            id,
            name,
            params: param_locals,
            ret_type: ret_ty,
            scheme: None,
            attributes: vec![],
            body: mir_body,
            span,
        })
    }

    fn flatten_entry_point(
        &mut self,
        id: NodeId,
        name: String,
        execution_model: sir::ExecutionModel,
        inputs: Vec<sir::EntryInput>,
        outputs: Vec<sir::EntryOutput>,
        body: Body,
        span: Span,
    ) -> Result<mir::Def> {
        let old_body = self.begin_body();
        self.var_to_local.clear();
        self.name_to_local.push_scope();

        // Allocate inputs as locals
        let mut mir_inputs = Vec::new();
        for input in &inputs {
            let local_id = self.alloc_local(
                input.name.clone(),
                input.ty.clone(),
                LocalKind::Param,
                span,
            );
            self.var_to_local.insert(input.var, local_id);
            mir_inputs.push(mir::EntryInput {
                local: local_id,
                name: input.name.clone(),
                ty: input.ty.clone(),
                decoration: input.decoration.as_ref().map(|d| self.convert_io_decoration(d)),
            });
        }

        // Convert outputs
        let mir_outputs: Vec<_> = outputs
            .iter()
            .map(|o| mir::EntryOutput {
                ty: o.ty.clone(),
                decoration: o.decoration.as_ref().map(|d| self.convert_io_decoration(d)),
            })
            .collect();

        // Convert execution model
        let mir_exec_model = match execution_model {
            sir::ExecutionModel::Vertex => mir::ExecutionModel::Vertex,
            sir::ExecutionModel::Fragment => mir::ExecutionModel::Fragment,
            sir::ExecutionModel::Compute { local_size } => mir::ExecutionModel::Compute { local_size },
        };

        // Flatten body
        let root_id = self.flatten_body(&body, span)?;
        self.current_body.set_root(root_id);

        self.name_to_local.pop_scope();
        let mir_body = self.end_body(old_body);

        Ok(mir::Def::EntryPoint {
            id,
            name,
            execution_model: mir_exec_model,
            inputs: mir_inputs,
            outputs: mir_outputs,
            body: mir_body,
            span,
        })
    }

    fn flatten_constant(
        &mut self,
        id: NodeId,
        name: String,
        ty: SirType,
        body: Body,
        span: Span,
    ) -> Result<mir::Def> {
        let old_body = self.begin_body();
        self.var_to_local.clear();
        self.name_to_local.push_scope();

        let root_id = self.flatten_body(&body, span)?;
        self.current_body.set_root(root_id);

        self.name_to_local.pop_scope();
        let mir_body = self.end_body(old_body);

        Ok(mir::Def::Constant {
            id,
            name,
            ty,
            attributes: vec![],
            body: mir_body,
            span,
        })
    }

    fn convert_io_decoration(&self, dec: &sir::IoDecoration) -> mir::IoDecoration {
        match dec {
            sir::IoDecoration::BuiltIn(b) => mir::IoDecoration::BuiltIn(*b),
            sir::IoDecoration::Location(l) => mir::IoDecoration::Location(*l),
        }
    }

    /// Flatten a SIR Body to a single MIR ExprId.
    ///
    /// SIR Body is a sequence of statements followed by result VarIds.
    /// This becomes nested Let expressions in MIR.
    fn flatten_body(&mut self, body: &Body, span: Span) -> Result<ExprId> {
        // First pass: pre-allocate LocalIds for all pattern bindings
        // This ensures var_to_local is populated before we reference any VarIds
        for stm in &body.stms {
            self.pre_allocate_pattern_bindings(stm)?;
        }

        // If no statements, just return the result
        if body.stms.is_empty() {
            return self.flatten_result(&body.result, span);
        }

        // Build from the inside out: start with result, wrap with lets
        let result_id = self.flatten_result(&body.result, span)?;

        // Process statements in reverse order to build nested lets
        let mut current = result_id;
        for stm in body.stms.iter().rev() {
            current = self.flatten_stm(stm, current)?;
        }

        Ok(current)
    }

    /// Pre-allocate LocalIds for all pattern bindings in a statement.
    fn pre_allocate_pattern_bindings(&mut self, stm: &Stm) -> Result<()> {
        for bind in &stm.pat.binds {
            if !self.var_to_local.contains_key(&bind.var) {
                let local_id = self.alloc_local(
                    bind.name_hint.clone(),
                    bind.ty.clone(),
                    LocalKind::Let,
                    stm.span,
                );
                self.var_to_local.insert(bind.var, local_id);
            }
        }
        Ok(())
    }

    /// Flatten result VarIds to a MIR expression.
    fn flatten_result(&mut self, result: &[VarId], span: Span) -> Result<ExprId> {
        match result.len() {
            0 => {
                // Unit result
                Ok(self.alloc_expr(mir::Expr::Unit, Type::Constructed(TypeName::Unit, vec![]), span))
            }
            1 => {
                // Single result - just reference the local
                let local_id = self.var_to_local.get(&result[0])
                    .copied()
                    .ok_or_else(|| crate::err_flatten!("Unknown variable in result"))?;
                let ty = self.current_body.get_local(local_id).ty.clone();
                Ok(self.alloc_expr(mir::Expr::Local(local_id), ty, span))
            }
            _ => {
                // Multi-result: function returns multiple values (flattened from tuple).
                // Create a tuple via intrinsic at the return boundary.
                let elem_ids: Vec<_> = result
                    .iter()
                    .map(|v| {
                        let local_id = self.var_to_local.get(v).copied()
                            .ok_or_else(|| crate::err_flatten!("Unknown variable in result"))?;
                        let ty = self.current_body.get_local(local_id).ty.clone();
                        Ok(self.alloc_expr(mir::Expr::Local(local_id), ty, span))
                    })
                    .collect::<Result<Vec<_>>>()?;

                let elem_tys: Vec<_> = elem_ids.iter()
                    .map(|id| self.current_body.get_type(*id).clone())
                    .collect();
                let tuple_ty = Type::Constructed(TypeName::Tuple(elem_tys.len()), elem_tys);

                Ok(self.alloc_expr(
                    mir::Expr::Intrinsic {
                        name: "tuple".to_string(),
                        args: elem_ids,
                    },
                    tuple_ty,
                    span,
                ))
            }
        }
    }

    /// Flatten a single SIR statement, wrapping the body in a Let.
    fn flatten_stm(&mut self, stm: &Stm, body_id: ExprId) -> Result<ExprId> {
        let span = stm.span;

        // Flatten the expression
        let rhs_id = self.flatten_exp(&stm.exp, &stm.ty, span)?;

        // Handle the pattern - use pre-allocated local
        if stm.pat.binds.len() == 1 {
            let bind = &stm.pat.binds[0];
            // Use the pre-allocated local from pre_allocate_pattern_bindings
            let local_id = *self.var_to_local.get(&bind.var)
                .ok_or_else(|| crate::err_flatten!("Missing pre-allocated local for {:?}", bind.var))?;

            let body_ty = self.current_body.get_type(body_id).clone();
            Ok(self.alloc_expr(
                mir::Expr::Let {
                    local: local_id,
                    rhs: rhs_id,
                    body: body_id,
                },
                body_ty,
                span,
            ))
        } else if stm.pat.binds.is_empty() {
            // No bindings - just sequence the expressions
            // Create a dummy local for the rhs
            let dummy_name = self.fresh_name("_");
            let local_id = self.alloc_local(
                dummy_name,
                stm.ty.clone(),
                LocalKind::Let,
                span,
            );
            let body_ty = self.current_body.get_type(body_id).clone();
            Ok(self.alloc_expr(
                mir::Expr::Let {
                    local: local_id,
                    rhs: rhs_id,
                    body: body_id,
                },
                body_ty,
                span,
            ))
        } else {
            // Tuple pattern - destructure
            self.flatten_tuple_pattern(stm, rhs_id, body_id, span)
        }
    }

    /// Flatten a tuple pattern into nested tuple_access + let bindings.
    fn flatten_tuple_pattern(
        &mut self,
        stm: &Stm,
        rhs_id: ExprId,
        body_id: ExprId,
        span: Span,
    ) -> Result<ExprId> {
        // First bind the tuple to a temp
        let tuple_name = self.fresh_name("tup");
        let tuple_local = self.alloc_local(
            tuple_name,
            stm.ty.clone(),
            LocalKind::Let,
            span,
        );

        // Build from inside out
        let mut current = body_id;

        // For each binding, extract from tuple and bind
        for (i, bind) in stm.pat.binds.iter().enumerate().rev() {
            // Use the pre-allocated local
            let local_id = *self.var_to_local.get(&bind.var)
                .ok_or_else(|| crate::err_flatten!("Missing pre-allocated local for tuple bind {:?}", bind.var))?;

            // tuple_access(tuple, index)
            let tuple_ref = self.alloc_expr(
                mir::Expr::Local(tuple_local),
                stm.ty.clone(),
                span,
            );
            let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
            let index_expr = self.alloc_expr(
                mir::Expr::Int(i.to_string()),
                i32_ty,
                span,
            );
            let access_expr = self.alloc_expr(
                mir::Expr::Intrinsic {
                    name: "tuple_access".to_string(),
                    args: vec![tuple_ref, index_expr],
                },
                bind.ty.clone(),
                span,
            );

            let body_ty = self.current_body.get_type(current).clone();
            current = self.alloc_expr(
                mir::Expr::Let {
                    local: local_id,
                    rhs: access_expr,
                    body: current,
                },
                body_ty,
                span,
            );
        }

        // Wrap with the tuple binding
        let body_ty = self.current_body.get_type(current).clone();
        Ok(self.alloc_expr(
            mir::Expr::Let {
                local: tuple_local,
                rhs: rhs_id,
                body: current,
            },
            body_ty,
            span,
        ))
    }

    /// Flatten a SIR expression to MIR.
    fn flatten_exp(&mut self, exp: &Exp, ty: &SirType, span: Span) -> Result<ExprId> {
        match exp {
            Exp::Prim(prim) => self.flatten_prim(prim, ty, span),

            Exp::Var(var_id) => {
                let local_id = self.var_to_local.get(var_id)
                    .copied()
                    .ok_or_else(|| crate::err_flatten!("Unknown variable {:?}", var_id))?;
                let local_ty = self.current_body.get_local(local_id).ty.clone();
                Ok(self.alloc_expr(mir::Expr::Local(local_id), local_ty, span))
            }

            Exp::If { cond, then_body, else_body } => {
                let cond_local = self.var_to_local.get(cond)
                    .copied()
                    .ok_or_else(|| crate::err_flatten!("Unknown condition variable"))?;
                let cond_ty = self.current_body.get_local(cond_local).ty.clone();
                let cond_id = self.alloc_expr(mir::Expr::Local(cond_local), cond_ty, span);

                let then_id = self.flatten_body(then_body, span)?;
                let else_id = self.flatten_body(else_body, span)?;

                Ok(self.alloc_expr(
                    mir::Expr::If {
                        cond: cond_id,
                        then_: then_id,
                        else_: else_id,
                    },
                    ty.clone(),
                    span,
                ))
            }

            Exp::Loop { params: _, init, body: _ } => {
                // Convert to MIR Loop
                // For now, emit as intrinsic - proper loop lowering is complex
                let init_ids: Vec<_> = init
                    .iter()
                    .map(|v| {
                        let local_id = self.var_to_local.get(v).copied().unwrap();
                        let ty = self.current_body.get_local(local_id).ty.clone();
                        self.alloc_expr(mir::Expr::Local(local_id), ty, span)
                    })
                    .collect();

                Ok(self.alloc_expr(
                    mir::Expr::Intrinsic {
                        name: "__loop".to_string(),
                        args: init_ids,
                    },
                    ty.clone(),
                    span,
                ))
            }

            Exp::Op(op) => self.flatten_op(op, ty, span),

            Exp::Apply { func, args } => {
                if args.is_empty() {
                    // No-args Apply is a global reference (uniform, storage, constant)
                    Ok(self.alloc_expr(
                        mir::Expr::Global(func.clone()),
                        ty.clone(),
                        span,
                    ))
                } else {
                    let arg_ids: Vec<_> = args
                        .iter()
                        .map(|v| {
                            let local_id = self.var_to_local.get(v).copied().unwrap();
                            let ty = self.current_body.get_local(local_id).ty.clone();
                            self.alloc_expr(mir::Expr::Local(local_id), ty, span)
                        })
                        .collect();

                    Ok(self.alloc_expr(
                        mir::Expr::Call {
                            func: func.clone(),
                            args: arg_ids,
                        },
                        ty.clone(),
                        span,
                    ))
                }
            }

            Exp::Tuple(elems) => {
                let elem_ids: Vec<_> = elems
                    .iter()
                    .map(|v| {
                        let local_id = self.var_to_local.get(v).copied().unwrap();
                        let ty = self.current_body.get_local(local_id).ty.clone();
                        self.alloc_expr(mir::Expr::Local(local_id), ty, span)
                    })
                    .collect();
                Ok(self.alloc_expr(mir::Expr::Tuple(elem_ids), ty.clone(), span))
            }

            Exp::TupleProj { tuple, index } => {
                let tuple_local = self.var_to_local.get(tuple).copied().unwrap();
                let tuple_ty = self.current_body.get_local(tuple_local).ty.clone();
                let tuple_id = self.alloc_expr(mir::Expr::Local(tuple_local), tuple_ty, span);
                Ok(self.alloc_expr(
                    mir::Expr::TupleProj { tuple: tuple_id, index: *index },
                    ty.clone(),
                    span,
                ))
            }
        }
    }

    /// Flatten a primitive expression.
    fn flatten_prim(&mut self, prim: &Prim, ty: &SirType, span: Span) -> Result<ExprId> {
        match prim {
            Prim::ConstBool(b) => {
                Ok(self.alloc_expr(mir::Expr::Bool(*b), ty.clone(), span))
            }
            Prim::ConstI32(n) => {
                Ok(self.alloc_expr(mir::Expr::Int(n.to_string()), ty.clone(), span))
            }
            Prim::ConstI64(n) => {
                Ok(self.alloc_expr(mir::Expr::Int(n.to_string()), ty.clone(), span))
            }
            Prim::ConstU32(n) => {
                Ok(self.alloc_expr(mir::Expr::Int(n.to_string()), ty.clone(), span))
            }
            Prim::ConstU64(n) => {
                Ok(self.alloc_expr(mir::Expr::Int(n.to_string()), ty.clone(), span))
            }
            Prim::ConstF32(n) => {
                Ok(self.alloc_expr(mir::Expr::Float(n.to_string()), ty.clone(), span))
            }
            Prim::ConstF64(n) => {
                Ok(self.alloc_expr(mir::Expr::Float(n.to_string()), ty.clone(), span))
            }

            Prim::Add(a, b) => self.flatten_binop("+", *a, *b, ty, span),
            Prim::Sub(a, b) => self.flatten_binop("-", *a, *b, ty, span),
            Prim::Mul(a, b) => self.flatten_binop("*", *a, *b, ty, span),
            Prim::Div(a, b) => self.flatten_binop("/", *a, *b, ty, span),
            Prim::Mod(a, b) => self.flatten_binop("%", *a, *b, ty, span),

            Prim::Eq(a, b) => self.flatten_binop("==", *a, *b, ty, span),
            Prim::Ne(a, b) => self.flatten_binop("!=", *a, *b, ty, span),
            Prim::Lt(a, b) => self.flatten_binop("<", *a, *b, ty, span),
            Prim::Le(a, b) => self.flatten_binop("<=", *a, *b, ty, span),
            Prim::Gt(a, b) => self.flatten_binop(">", *a, *b, ty, span),
            Prim::Ge(a, b) => self.flatten_binop(">=", *a, *b, ty, span),

            Prim::And(a, b) => self.flatten_binop("&&", *a, *b, ty, span),
            Prim::Or(a, b) => self.flatten_binop("||", *a, *b, ty, span),

            Prim::Neg(a) => {
                let local_id = self.var_to_local.get(a).copied().unwrap();
                let operand_ty = self.current_body.get_local(local_id).ty.clone();
                let operand_id = self.alloc_expr(mir::Expr::Local(local_id), operand_ty, span);
                Ok(self.alloc_expr(
                    mir::Expr::UnaryOp {
                        op: "-".to_string(),
                        operand: operand_id,
                    },
                    ty.clone(),
                    span,
                ))
            }
            Prim::Not(a) => {
                let local_id = self.var_to_local.get(a).copied().unwrap();
                let operand_ty = self.current_body.get_local(local_id).ty.clone();
                let operand_id = self.alloc_expr(mir::Expr::Local(local_id), operand_ty, span);
                Ok(self.alloc_expr(
                    mir::Expr::UnaryOp {
                        op: "!".to_string(),
                        operand: operand_id,
                    },
                    ty.clone(),
                    span,
                ))
            }

            Prim::Index { arr, idx } => {
                let arr_local = self.var_to_local.get(arr).copied().unwrap();
                let arr_ty = self.current_body.get_local(arr_local).ty.clone();
                let arr_id = self.alloc_expr(mir::Expr::Local(arr_local), arr_ty, span);

                let idx_local = self.var_to_local.get(idx).copied().unwrap();
                let idx_ty = self.current_body.get_local(idx_local).ty.clone();
                let idx_id = self.alloc_expr(mir::Expr::Local(idx_local), idx_ty, span);

                Ok(self.alloc_expr(
                    mir::Expr::Intrinsic {
                        name: "index".to_string(),
                        args: vec![arr_id, idx_id],
                    },
                    ty.clone(),
                    span,
                ))
            }

            Prim::Intrinsic { name, args } => {
                // Handle special intrinsics that need to be converted to MIR constructs
                if name == "array_literal" {
                    // array_literal(a, b, c, ...) -> Array with Literal backing
                    let elem_ids: Vec<_> = args
                        .iter()
                        .map(|v| {
                            let local_id = self.var_to_local.get(v).copied().unwrap();
                            let ty = self.current_body.get_local(local_id).ty.clone();
                            self.alloc_expr(mir::Expr::Local(local_id), ty, span)
                        })
                        .collect();

                    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
                    let size_id = self.alloc_expr(mir::Expr::Int(elem_ids.len().to_string()), i32_ty, span);

                    return Ok(self.alloc_expr(
                        mir::Expr::Array {
                            backing: ArrayBacking::Literal(elem_ids),
                            size: size_id,
                        },
                        ty.clone(),
                        span,
                    ));
                }

                if name == "vec_literal" {
                    // vec_literal(a, b, c, ...) -> Vector([a, b, c, ...])
                    let elem_ids: Vec<_> = args
                        .iter()
                        .map(|v| {
                            let local_id = self.var_to_local.get(v).copied().unwrap();
                            let ty = self.current_body.get_local(local_id).ty.clone();
                            self.alloc_expr(mir::Expr::Local(local_id), ty, span)
                        })
                        .collect();
                    return Ok(self.alloc_expr(
                        mir::Expr::Vector(elem_ids),
                        ty.clone(),
                        span,
                    ));
                }

                // __lambda_N(captures...) -> defunctionalize and create Closure
                if name.starts_with("__lambda_") {
                    let lambda_id_str = &name[9..]; // strip "__lambda_"
                    if let Ok(lambda_id_num) = lambda_id_str.parse::<u32>() {
                        let lambda_id = sir::LambdaId(lambda_id_num);
                        if let Some(lambda) = self.sir_lambdas.get(&lambda_id).cloned() {
                            // Defunctionalize the lambda
                            let lambda_name = self.defunctionalize_lambda(&lambda, span)?;

                            // Build capture expressions from args (which are the capture VarIds)
                            let capture_ids: Vec<_> = args
                                .iter()
                                .map(|v| {
                                    let local_id = self.var_to_local.get(v).copied().unwrap();
                                    let ty = self.current_body.get_local(local_id).ty.clone();
                                    self.alloc_expr(mir::Expr::Local(local_id), ty, span)
                                })
                                .collect();

                            // Return a Closure expression
                            return Ok(self.alloc_expr(
                                mir::Expr::Closure {
                                    lambda_name,
                                    captures: capture_ids,
                                },
                                ty.clone(),
                                span,
                            ));
                        }
                    }
                }

                if name == "__range" && args.len() == 2 {
                    // __range(start, end) -> Array with Range backing
                    let start_local = self.var_to_local.get(&args[0]).copied().unwrap();
                    let start_ty = self.current_body.get_local(start_local).ty.clone();
                    let start_id = self.alloc_expr(mir::Expr::Local(start_local), start_ty, span);

                    let end_local = self.var_to_local.get(&args[1]).copied().unwrap();
                    let end_ty = self.current_body.get_local(end_local).ty.clone();
                    let end_id = self.alloc_expr(mir::Expr::Local(end_local), end_ty, span);

                    return Ok(self.alloc_expr(
                        mir::Expr::Array {
                            backing: ArrayBacking::Range {
                                start: start_id,
                                step: None,
                                kind: RangeKind::Exclusive,
                            },
                            size: end_id,
                        },
                        ty.clone(),
                        span,
                    ));
                }

                if name == "__range_inclusive" && args.len() == 2 {
                    // __range_inclusive(start, end) -> Array with Range backing (inclusive)
                    let start_local = self.var_to_local.get(&args[0]).copied().unwrap();
                    let start_ty = self.current_body.get_local(start_local).ty.clone();
                    let start_id = self.alloc_expr(mir::Expr::Local(start_local), start_ty, span);

                    let end_local = self.var_to_local.get(&args[1]).copied().unwrap();
                    let end_ty = self.current_body.get_local(end_local).ty.clone();
                    let end_id = self.alloc_expr(mir::Expr::Local(end_local), end_ty, span);

                    return Ok(self.alloc_expr(
                        mir::Expr::Array {
                            backing: ArrayBacking::Range {
                                start: start_id,
                                step: None,
                                kind: RangeKind::Inclusive,
                            },
                            size: end_id,
                        },
                        ty.clone(),
                        span,
                    ));
                }

                // Slice intrinsics: __slice_range, __slice_from, __slice_to, __slice_full
                if name == "__slice_range" && args.len() == 3 {
                    // arr[start..end]
                    let base_local = self.var_to_local.get(&args[0]).copied().unwrap();
                    let base_ty = self.current_body.get_local(base_local).ty.clone();
                    let base_id = self.alloc_expr(mir::Expr::Local(base_local), base_ty, span);

                    let start_local = self.var_to_local.get(&args[1]).copied().unwrap();
                    let start_ty = self.current_body.get_local(start_local).ty.clone();
                    let offset_id = self.alloc_expr(mir::Expr::Local(start_local), start_ty, span);

                    let end_local = self.var_to_local.get(&args[2]).copied().unwrap();
                    let end_ty = self.current_body.get_local(end_local).ty.clone();
                    let end_id = self.alloc_expr(mir::Expr::Local(end_local), end_ty, span);

                    // length = end - start
                    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
                    let len_id = self.alloc_expr(
                        mir::Expr::BinOp {
                            op: "-".to_string(),
                            lhs: end_id,
                            rhs: offset_id,
                        },
                        i32_ty,
                        span,
                    );

                    return Ok(self.alloc_expr(
                        mir::Expr::Array {
                            backing: ArrayBacking::View {
                                base: base_id,
                                offset: offset_id,
                            },
                            size: len_id,
                        },
                        ty.clone(),
                        span,
                    ));
                }

                if name == "__slice_from" && args.len() == 2 {
                    // arr[start..] - from start to end of array
                    // For now, emit as intrinsic since we'd need array length
                    let arg_ids: Vec<_> = args
                        .iter()
                        .map(|v| {
                            let local_id = self.var_to_local.get(v).copied().unwrap();
                            let ty = self.current_body.get_local(local_id).ty.clone();
                            self.alloc_expr(mir::Expr::Local(local_id), ty, span)
                        })
                        .collect();
                    return Ok(self.alloc_expr(
                        mir::Expr::Intrinsic {
                            name: "slice_from".to_string(),
                            args: arg_ids,
                        },
                        ty.clone(),
                        span,
                    ));
                }

                if name == "__slice_to" && args.len() == 2 {
                    // arr[..end] - from 0 to end
                    let base_local = self.var_to_local.get(&args[0]).copied().unwrap();
                    let base_ty = self.current_body.get_local(base_local).ty.clone();
                    let base_id = self.alloc_expr(mir::Expr::Local(base_local), base_ty, span);

                    let end_local = self.var_to_local.get(&args[1]).copied().unwrap();
                    let end_ty = self.current_body.get_local(end_local).ty.clone();
                    let end_id = self.alloc_expr(mir::Expr::Local(end_local), end_ty, span);

                    // offset = 0
                    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
                    let offset_id = self.alloc_expr(mir::Expr::Int("0".to_string()), i32_ty.clone(), span);

                    return Ok(self.alloc_expr(
                        mir::Expr::Array {
                            backing: ArrayBacking::View {
                                base: base_id,
                                offset: offset_id,
                            },
                            size: end_id,
                        },
                        ty.clone(),
                        span,
                    ));
                }

                if name == "__slice_full" && args.len() == 1 {
                    // arr[..] - full slice (identity)
                    let base_local = self.var_to_local.get(&args[0]).copied().unwrap();
                    let base_ty = self.current_body.get_local(base_local).ty.clone();
                    return Ok(self.alloc_expr(mir::Expr::Local(base_local), base_ty, span));
                }

                // _w_array_with should be emitted as a Call for alias checker compatibility
                if name == "_w_array_with" && args.len() == 3 {
                    let arg_ids: Vec<_> = args
                        .iter()
                        .map(|v| {
                            let local_id = self.var_to_local.get(v).copied().unwrap();
                            let ty = self.current_body.get_local(local_id).ty.clone();
                            self.alloc_expr(mir::Expr::Local(local_id), ty, span)
                        })
                        .collect();
                    return Ok(self.alloc_expr(
                        mir::Expr::Call {
                            func: "_w_array_with".to_string(),
                            args: arg_ids,
                        },
                        ty.clone(),
                        span,
                    ));
                }

                let arg_ids: Vec<_> = args
                    .iter()
                    .map(|v| {
                        let local_id = self.var_to_local.get(v).copied().unwrap();
                        let ty = self.current_body.get_local(local_id).ty.clone();
                        self.alloc_expr(mir::Expr::Local(local_id), ty, span)
                    })
                    .collect();

                Ok(self.alloc_expr(
                    mir::Expr::Intrinsic {
                        name: name.clone(),
                        args: arg_ids,
                    },
                    ty.clone(),
                    span,
                ))
            }
        }
    }

    fn flatten_binop(&mut self, op: &str, lhs: VarId, rhs: VarId, ty: &SirType, span: Span) -> Result<ExprId> {
        let lhs_local = self.var_to_local.get(&lhs).copied().unwrap();
        let lhs_ty = self.current_body.get_local(lhs_local).ty.clone();
        let lhs_id = self.alloc_expr(mir::Expr::Local(lhs_local), lhs_ty, span);

        let rhs_local = self.var_to_local.get(&rhs).copied().unwrap();
        let rhs_ty = self.current_body.get_local(rhs_local).ty.clone();
        let rhs_id = self.alloc_expr(mir::Expr::Local(rhs_local), rhs_ty, span);

        Ok(self.alloc_expr(
            mir::Expr::BinOp {
                op: op.to_string(),
                lhs: lhs_id,
                rhs: rhs_id,
            },
            ty.clone(),
            span,
        ))
    }

    /// Flatten a SOAC or Launch operation.
    fn flatten_op(&mut self, op: &sir::Op, ty: &SirType, span: Span) -> Result<ExprId> {
        match op {
            sir::Op::Soac(soac) => self.flatten_soac(soac, ty, span),
            sir::Op::Launch(_launch) => {
                // Launch is post-kernelization, shouldn't appear yet
                bail_flatten!("Launch operations not yet supported in flattening")
            }
        }
    }

    /// Flatten a SOAC to MIR.
    fn flatten_soac(&mut self, soac: &Soac, ty: &SirType, span: Span) -> Result<ExprId> {
        match soac {
            Soac::Map(map) => {
                // Lower map to _w_intrinsic_map call
                let lambda_name = self.defunctionalize_lambda(&map.f, span)?;

                // Build closure for the lambda
                let capture_ids: Vec<_> = map.f.captures
                    .iter()
                    .map(|v| {
                        let local_id = self.var_to_local.get(v).copied().unwrap();
                        let ty = self.current_body.get_local(local_id).ty.clone();
                        self.alloc_expr(mir::Expr::Local(local_id), ty, span)
                    })
                    .collect();

                // Closure type is the function type
                let closure_ty = self.build_closure_type(&map.f);
                let closure_id = self.alloc_expr(
                    mir::Expr::Closure {
                        lambda_name,
                        captures: capture_ids,
                    },
                    closure_ty,
                    span,
                );

                // Build array arguments
                let arr_ids: Vec<_> = map.arrs
                    .iter()
                    .map(|v| {
                        let local_id = self.var_to_local.get(v).copied().unwrap();
                        let ty = self.current_body.get_local(local_id).ty.clone();
                        self.alloc_expr(mir::Expr::Local(local_id), ty, span)
                    })
                    .collect();

                let mut args = vec![closure_id];
                args.extend(arr_ids);

                Ok(self.alloc_expr(
                    mir::Expr::Call {
                        func: "_w_intrinsic_map".to_string(),
                        args,
                    },
                    ty.clone(),
                    span,
                ))
            }

            Soac::Reduce(reduce) => {
                let lambda_name = self.defunctionalize_lambda(&reduce.f, span)?;

                let capture_ids: Vec<_> = reduce.f.captures
                    .iter()
                    .map(|v| {
                        let local_id = self.var_to_local.get(v).copied().unwrap();
                        let ty = self.current_body.get_local(local_id).ty.clone();
                        self.alloc_expr(mir::Expr::Local(local_id), ty, span)
                    })
                    .collect();

                // Closure type is the function type
                let closure_ty = self.build_closure_type(&reduce.f);
                let closure_id = self.alloc_expr(
                    mir::Expr::Closure {
                        lambda_name,
                        captures: capture_ids,
                    },
                    closure_ty,
                    span,
                );

                let neutral_local = self.var_to_local.get(&reduce.neutral).copied().unwrap();
                let neutral_ty = self.current_body.get_local(neutral_local).ty.clone();
                let neutral_id = self.alloc_expr(mir::Expr::Local(neutral_local), neutral_ty, span);

                let arr_local = self.var_to_local.get(&reduce.arr).copied().unwrap();
                let arr_ty = self.current_body.get_local(arr_local).ty.clone();
                let arr_id = self.alloc_expr(mir::Expr::Local(arr_local), arr_ty, span);

                Ok(self.alloc_expr(
                    mir::Expr::Call {
                        func: "_w_intrinsic_reduce".to_string(),
                        args: vec![closure_id, neutral_id, arr_id],
                    },
                    ty.clone(),
                    span,
                ))
            }

            Soac::Scan(scan) => {
                let lambda_name = self.defunctionalize_lambda(&scan.f, span)?;

                let capture_ids: Vec<_> = scan.f.captures
                    .iter()
                    .map(|v| {
                        let local_id = self.var_to_local.get(v).copied().unwrap();
                        let ty = self.current_body.get_local(local_id).ty.clone();
                        self.alloc_expr(mir::Expr::Local(local_id), ty, span)
                    })
                    .collect();

                // Closure type is the function type
                let closure_ty = self.build_closure_type(&scan.f);
                let closure_id = self.alloc_expr(
                    mir::Expr::Closure {
                        lambda_name,
                        captures: capture_ids,
                    },
                    closure_ty,
                    span,
                );

                let neutral_local = self.var_to_local.get(&scan.neutral).copied().unwrap();
                let neutral_ty = self.current_body.get_local(neutral_local).ty.clone();
                let neutral_id = self.alloc_expr(mir::Expr::Local(neutral_local), neutral_ty, span);

                let arr_local = self.var_to_local.get(&scan.arr).copied().unwrap();
                let arr_ty = self.current_body.get_local(arr_local).ty.clone();
                let arr_id = self.alloc_expr(mir::Expr::Local(arr_local), arr_ty, span);

                Ok(self.alloc_expr(
                    mir::Expr::Call {
                        func: "_w_intrinsic_scan".to_string(),
                        args: vec![closure_id, neutral_id, arr_id],
                    },
                    ty.clone(),
                    span,
                ))
            }

            Soac::Iota { n, elem_ty: _ } => {
                // Lower iota to Array with Range backing
                let size_id = self.flatten_size(n, span)?;
                let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
                let start_id = self.alloc_expr(mir::Expr::Int("0".to_string()), i32_ty, span);

                Ok(self.alloc_expr(
                    mir::Expr::Array {
                        backing: ArrayBacking::Range {
                            start: start_id,
                            step: None,
                            kind: RangeKind::Exclusive,
                        },
                        size: size_id,
                    },
                    ty.clone(),
                    span,
                ))
            }

            Soac::Replicate { n, value } => {
                // Lower replicate to Array with IndexFn (constant function)
                let size_id = self.flatten_size(n, span)?;

                let value_local = self.var_to_local.get(value).copied().unwrap();
                let value_ty = self.current_body.get_local(value_local).ty.clone();
                let value_id = self.alloc_expr(mir::Expr::Local(value_local), value_ty.clone(), span);

                // Create a constant function that returns the value
                // Type is i32 -> value_ty (takes index, returns value)
                let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
                let closure_ty = Type::Constructed(TypeName::Arrow, vec![i32_ty, value_ty]);
                let fn_id = self.alloc_expr(
                    mir::Expr::Closure {
                        lambda_name: "__replicate_fn".to_string(),
                        captures: vec![value_id],
                    },
                    closure_ty,
                    span,
                );

                Ok(self.alloc_expr(
                    mir::Expr::Array {
                        backing: ArrayBacking::IndexFn { index_fn: fn_id },
                        size: size_id,
                    },
                    ty.clone(),
                    span,
                ))
            }

            Soac::Reshape { new_shape: _, arr } => {
                // For now, reshape is identity (shape info is in type)
                let arr_local = self.var_to_local.get(arr).copied().unwrap();
                let arr_ty = self.current_body.get_local(arr_local).ty.clone();
                Ok(self.alloc_expr(mir::Expr::Local(arr_local), arr_ty, span))
            }

            // Segmented operations - emit as intrinsics for now
            Soac::SegMap(_) | Soac::SegReduce(_) | Soac::SegScan(_) => {
                Ok(self.alloc_expr(
                    mir::Expr::Intrinsic {
                        name: "__segmented_op".to_string(),
                        args: vec![],
                    },
                    ty.clone(),
                    span,
                ))
            }
        }
    }

    /// Flatten a Size to an ExprId.
    fn flatten_size(&mut self, size: &sir::Size, span: Span) -> Result<ExprId> {
        let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
        match size {
            sir::Size::Const(n) => {
                Ok(self.alloc_expr(mir::Expr::Int(n.to_string()), i32_ty, span))
            }
            sir::Size::Sym(_var) => {
                // Symbolic size - would need to look up the actual variable
                // For now, emit a placeholder
                Ok(self.alloc_expr(
                    mir::Expr::Intrinsic {
                        name: "__symbolic_size".to_string(),
                        args: vec![],
                    },
                    i32_ty,
                    span,
                ))
            }
            sir::Size::Add(a, b) => {
                let a_id = self.flatten_size(a, span)?;
                let b_id = self.flatten_size(b, span)?;
                Ok(self.alloc_expr(
                    mir::Expr::BinOp {
                        op: "+".to_string(),
                        lhs: a_id,
                        rhs: b_id,
                    },
                    i32_ty,
                    span,
                ))
            }
            sir::Size::Mul(a, b) => {
                let a_id = self.flatten_size(a, span)?;
                let b_id = self.flatten_size(b, span)?;
                Ok(self.alloc_expr(
                    mir::Expr::BinOp {
                        op: "*".to_string(),
                        lhs: a_id,
                        rhs: b_id,
                    },
                    i32_ty,
                    span,
                ))
            }
        }
    }

    /// Build the Arrow type for a lambda's closure.
    /// The closure type is: param1 -> param2 -> ... -> ret_ty
    fn build_closure_type(&self, lambda: &Lambda) -> SirType {
        // Get the return type (single return or unit if empty)
        let ret_ty = if lambda.ret_tys.len() == 1 {
            lambda.ret_tys[0].clone()
        } else if lambda.ret_tys.is_empty() {
            Type::Constructed(TypeName::Unit, vec![])
        } else {
            // Multiple returns - use tuple type with arity
            Type::Constructed(TypeName::Tuple(lambda.ret_tys.len()), lambda.ret_tys.clone())
        };

        // Build curried Arrow type from right to left
        let mut result = ret_ty;
        for param in lambda.params.iter().rev() {
            result = Type::Constructed(TypeName::Arrow, vec![param.ty.clone(), result]);
        }

        result
    }

    /// Defunctionalize a lambda, creating a top-level function and returning its name.
    fn defunctionalize_lambda(&mut self, lambda: &Lambda, span: Span) -> Result<String> {
        let lambda_name = self.fresh_name("lambda");

        // Save current state
        let old_body = self.begin_body();
        let old_var_to_local = std::mem::take(&mut self.var_to_local);
        self.name_to_local.push_scope();

        // Create parameters: captures first, then regular params
        let mut param_locals = Vec::new();

        // Add capture parameters - get types from enclosing scope
        for (i, capture_var) in lambda.captures.iter().enumerate() {
            let capture_name = format!("_cap_{}", i);
            let capture_ty = old_var_to_local
                .get(capture_var)
                .map(|local_id| old_body.get_local(*local_id).ty.clone())
                .unwrap_or_else(|| panic!("Capture {:?} not found in enclosing scope", capture_var));
            let local_id = self.alloc_local(capture_name, capture_ty, LocalKind::Param, span);
            self.var_to_local.insert(*capture_var, local_id);
            param_locals.push(local_id);
        }

        // Add regular parameters
        for param in &lambda.params {
            let local_id = self.alloc_local(
                param.name_hint.clone(),
                param.ty.clone(),
                LocalKind::Param,
                param.span,
            );
            self.var_to_local.insert(param.var, local_id);
            param_locals.push(local_id);
        }

        // Flatten lambda body
        let root_id = self.flatten_body(&lambda.body, span)?;
        self.current_body.set_root(root_id);

        self.name_to_local.pop_scope();
        let lambda_body = self.end_body(old_body);

        // Restore state
        self.var_to_local = old_var_to_local;

        // Determine return type
        let ret_type = if lambda.ret_tys.len() == 1 {
            lambda.ret_tys[0].clone()
        } else if lambda.ret_tys.is_empty() {
            Type::Constructed(TypeName::Unit, vec![])
        } else {
            Type::Constructed(TypeName::Tuple(lambda.ret_tys.len()), lambda.ret_tys.clone())
        };

        // Create the function def
        let func_def = mir::Def::Function {
            id: self.next_node_id(),
            name: lambda_name.clone(),
            params: param_locals,
            ret_type,
            scheme: None,
            attributes: vec![],
            body: lambda_body,
            span,
        };

        self.generated_functions.push(func_def);

        // Register in lambda registry
        let lambda_info = mir::LambdaInfo {
            name: lambda_name.clone(),
            arity: lambda.params.len(),
        };
        self.lambda_registry.alloc(lambda_info);

        Ok(lambda_name)
    }

    /// Get the NodeCounter for use after flattening.
    pub fn into_node_counter(self) -> NodeCounter {
        self.node_counter
    }
}
