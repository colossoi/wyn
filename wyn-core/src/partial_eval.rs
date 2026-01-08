//! Partial evaluation pass for MIR using normalization-by-evaluation (NBE).
//!
//! This pass evaluates expressions at compile time when their operands are known,
//! inlines function calls with constant arguments, and unrolls loops with known bounds.
//! It replaces the simpler `constant_folding` pass when enabled.

use crate::ast::{NodeId, Span, TypeName};
use crate::error::Result;
use crate::mir::{Body, Def, Expr, ExprId, LocalId, LocalKind, LoopKind, Program};
use crate::{bail_type_at, err_type_at};
use polytype::Type;
use std::collections::HashMap;

// =============================================================================
// Scoped local mapping
// =============================================================================

/// Maps LocalIds from input bodies to LocalIds in the output body.
///
/// This needs scoping because when we inline a function or evaluate a constant,
/// we switch to evaluating a different input body. Each input body has its own
/// LocalId namespace, so we need separate mappings for each.
///
/// Example: If outer function has LocalId(0) = "x" and inlined function has
/// LocalId(0) = "y", these need different output LocalIds.
#[derive(Debug, Default)]
struct ScopedLocalMap {
    /// Stack of local mappings, one per input body being evaluated.
    /// The top of the stack is the current scope.
    frames: Vec<HashMap<LocalId, LocalId>>,
}

impl ScopedLocalMap {
    fn new() -> Self {
        ScopedLocalMap {
            frames: vec![HashMap::new()],
        }
    }

    /// Enter a new scope (when starting to evaluate a different body)
    fn push_scope(&mut self) {
        self.frames.push(HashMap::new());
    }

    /// Leave the current scope (when done evaluating a body)
    fn pop_scope(&mut self) {
        self.frames.pop();
    }

    /// Look up a LocalId in the current scope only
    fn get(&self, id: LocalId) -> Option<LocalId> {
        self.frames.last()?.get(&id).copied()
    }

    /// Insert a mapping in the current scope
    fn insert(&mut self, old: LocalId, new: LocalId) {
        if let Some(frame) = self.frames.last_mut() {
            frame.insert(old, new);
        }
    }

    /// Clear all scopes and start fresh (for a new top-level definition)
    fn reset(&mut self) {
        self.frames.clear();
        self.frames.push(HashMap::new());
    }
}

// =============================================================================
// Value representation
// =============================================================================

/// Semantic values for partial evaluation.
/// Represents what we know about a computation at compile time.
#[derive(Debug, Clone)]
pub enum Value {
    /// Known scalar value
    Scalar(ScalarValue),

    /// Known aggregate with all elements known
    Aggregate {
        kind: AggregateKind,
        elements: Vec<Value>,
    },

    /// Known closure (lambda with captured values)
    Closure {
        lambda_name: String,
        captures: Vec<Value>,
    },

    /// Unit value
    Unit,

    /// Completely unknown at compile time - must compute at runtime.
    /// The ExprId points to code in the OUTPUT body that computes this value.
    Unknown(ExprId),
}

/// Scalar value types
#[derive(Debug, Clone)]
pub enum ScalarValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
}

/// Kind of aggregate value
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregateKind {
    Tuple,
    Array,
    Vector,
    Matrix,
}

impl Value {
    /// Check if this value is fully known at compile time
    pub fn is_known(&self) -> bool {
        match self {
            Value::Scalar(_) | Value::Unit => true,
            Value::Aggregate { elements, .. } => elements.iter().all(|e| e.is_known()),
            Value::Closure { captures, .. } => captures.iter().all(|c| c.is_known()),
            Value::Unknown(_) => false,
        }
    }
}

// =============================================================================
// Environment
// =============================================================================

/// Environment mapping local variables to their values
#[derive(Debug, Clone)]
struct Env {
    bindings: HashMap<LocalId, Value>,
}

impl Env {
    fn new() -> Self {
        Env {
            bindings: HashMap::new(),
        }
    }

    fn extend(&mut self, local: LocalId, value: Value) {
        self.bindings.insert(local, value);
    }

    fn lookup(&self, local: LocalId) -> Option<&Value> {
        self.bindings.get(&local)
    }

    /// Create a child environment for entering a new scope
    fn child(&self) -> Self {
        Env {
            bindings: self.bindings.clone(),
        }
    }
}

// =============================================================================
// Cost budget
// =============================================================================

/// Cost budget to prevent unbounded expansion
#[derive(Debug, Clone)]
struct CostBudget {
    /// Maximum loop iterations to unroll
    max_loop_unroll: usize,
    /// Maximum function inlining depth
    max_inline_depth: usize,
    /// Current inline depth
    inline_depth: usize,
}

impl Default for CostBudget {
    fn default() -> Self {
        CostBudget {
            max_loop_unroll: 16,
            max_inline_depth: 8,
            inline_depth: 0,
        }
    }
}

// =============================================================================
// Partial Evaluator
// =============================================================================

/// Cached function/constant body for inlining
#[derive(Clone)]
struct FunctionInfo {
    params: Vec<LocalId>,
    body: Body,
}

/// Partial evaluator state
struct PartialEvaluator {
    /// Map from function name to its body (for inlining)
    func_cache: HashMap<String, FunctionInfo>,

    /// Map from constant name to its body (for evaluation)
    const_cache: HashMap<String, Body>,

    /// Output body being constructed (for the current definition)
    output: Body,

    /// Cost budget
    cost_budget: CostBudget,

    /// Scoped mapping from old LocalId to new LocalId.
    /// Supports nested scopes for inlined functions and constant evaluation.
    local_map: ScopedLocalMap,
}

impl PartialEvaluator {
    fn new(program: &Program) -> Self {
        let mut func_cache = HashMap::new();
        let mut const_cache = HashMap::new();

        for def in &program.defs {
            match def {
                Def::Function {
                    name, params, body, ..
                } => {
                    func_cache.insert(
                        name.clone(),
                        FunctionInfo {
                            params: params.clone(),
                            body: body.clone(),
                        },
                    );
                }
                Def::Constant { name, body, .. } => {
                    const_cache.insert(name.clone(), body.clone());
                }
                _ => {}
            }
        }

        PartialEvaluator {
            func_cache,
            const_cache,
            output: Body::new(),
            cost_budget: CostBudget::default(),
            local_map: ScopedLocalMap::new(),
        }
    }

    /// Emit an expression to the output body
    fn emit(&mut self, expr: Expr, ty: Type<TypeName>, span: Span, node_id: NodeId) -> ExprId {
        self.output.alloc_expr(expr, ty, span, node_id)
    }

    /// Convert a Value back to an ExprId in the output body (residualization)
    fn reify(&mut self, value: &Value, ty: Type<TypeName>, span: Span, node_id: NodeId) -> ExprId {
        match value {
            Value::Scalar(ScalarValue::Int(n)) => self.emit(Expr::Int(n.to_string()), ty, span, node_id),
            Value::Scalar(ScalarValue::Float(f)) => {
                self.emit(Expr::Float(f.to_string()), ty, span, node_id)
            }
            Value::Scalar(ScalarValue::Bool(b)) => self.emit(Expr::Bool(*b), ty, span, node_id),
            Value::Scalar(ScalarValue::String(s)) => self.emit(Expr::String(s.clone()), ty, span, node_id),
            Value::Unit => self.emit(Expr::Unit, ty, span, node_id),
            Value::Aggregate { kind, elements } => {
                let elem_ids: Vec<ExprId> = elements
                    .iter()
                    .enumerate()
                    .map(|(i, v)| {
                        let elem_ty = self.get_element_type(&ty, i);
                        self.reify(v, elem_ty, span, node_id)
                    })
                    .collect();

                match kind {
                    AggregateKind::Tuple => self.emit(Expr::Tuple(elem_ids), ty, span, node_id),
                    AggregateKind::Array => self.emit(Expr::Array(elem_ids), ty, span, node_id),
                    AggregateKind::Vector => self.emit(Expr::Vector(elem_ids), ty, span, node_id),
                    AggregateKind::Matrix => {
                        // Reconstruct the matrix structure from flat elements
                        // Matrix type is Mat(Size(cols), Size(rows), ElemType)
                        // The lowering expects Expr::Matrix(rows) where each row is
                        // a Vec of scalar ExprIds - it creates the row vectors itself
                        let (cols, rows) = self.get_matrix_dimensions(&ty);

                        // Group flat scalar elements into rows
                        let matrix_rows: Vec<Vec<ExprId>> = (0..rows)
                            .map(|row_idx| {
                                let row_start = row_idx * cols;
                                let row_end = row_start + cols;
                                elem_ids[row_start..row_end].to_vec()
                            })
                            .collect();

                        self.emit(Expr::Matrix(matrix_rows), ty, span, node_id)
                    }
                }
            }
            Value::Closure {
                lambda_name,
                captures,
            } => {
                // Reify each capture individually
                // We don't have type info for each capture at this point, so we use Unit as placeholder
                // (the actual capture type comes from the original expression in most cases)
                let capture_ids: Vec<ExprId> = captures
                    .iter()
                    .map(|cap| {
                        let cap_ty = Type::Constructed(TypeName::Unit, vec![]);
                        self.reify(cap, cap_ty, span, node_id)
                    })
                    .collect();
                self.emit(
                    Expr::Closure {
                        lambda_name: lambda_name.clone(),
                        captures: capture_ids,
                    },
                    ty,
                    span,
                    node_id,
                )
            }
            Value::Unknown(expr_id) => *expr_id,
        }
    }

    /// Get the element type at a given index for a tuple/array/vector/matrix type
    fn get_element_type(&self, ty: &Type<TypeName>, index: usize) -> Type<TypeName> {
        match ty {
            Type::Constructed(TypeName::Tuple(_), args) => {
                // Tuple: args are the element types in order
                args.get(index).cloned().unwrap_or_else(|| ty.clone())
            }
            Type::Constructed(TypeName::Array, args) if args.len() >= 2 => {
                // Array: args[0] = Size(n), args[1] = element type
                args[1].clone()
            }
            Type::Constructed(TypeName::Vec, args) if args.len() >= 2 => {
                // Vec: args[0] = Size(n), args[1] = element type
                args[1].clone()
            }
            Type::Constructed(TypeName::Mat, args) if args.len() >= 3 => {
                // Mat: args[0] = Size(cols), args[1] = Size(rows), args[2] = element type
                // For flattened matrix values, each element is the scalar element type
                args[2].clone()
            }
            Type::Constructed(TypeName::Record(_fields), args) => {
                // Record: args are the field types in source order
                args.get(index).cloned().unwrap_or_else(|| ty.clone())
            }
            _ => ty.clone(),
        }
    }

    /// Evaluate an expression, returning a Value
    fn eval(&mut self, body: &Body, expr_id: ExprId, env: &mut Env) -> Result<Value> {
        let expr = body.get_expr(expr_id).clone();
        let ty = body.get_type(expr_id).clone();
        let span = body.get_span(expr_id);
        let node_id = body.get_node_id(expr_id);

        match expr {
            // Literals - always known
            Expr::Int(s) => {
                let val: i64 = s.parse().map_err(|_| err_type_at!(span, "Invalid integer literal"))?;
                Ok(Value::Scalar(ScalarValue::Int(val)))
            }
            Expr::Float(s) => {
                let val: f64 = s.parse().map_err(|_| err_type_at!(span, "Invalid float literal"))?;
                Ok(Value::Scalar(ScalarValue::Float(val)))
            }
            Expr::Bool(b) => Ok(Value::Scalar(ScalarValue::Bool(b))),
            Expr::Unit => Ok(Value::Unit),
            Expr::String(s) => Ok(Value::Scalar(ScalarValue::String(s))),

            // Local variable reference
            Expr::Local(local_id) => {
                if let Some(value) = env.lookup(local_id) {
                    Ok(value.clone())
                } else {
                    // Unknown local (e.g., function parameter) - residualize
                    let new_local = self.map_local(body, local_id);
                    let new_id = self.emit(Expr::Local(new_local), ty, span, node_id);
                    Ok(Value::Unknown(new_id))
                }
            }

            // Global reference
            Expr::Global(name) => {
                // Check if it's a constant we can evaluate
                if let Some(const_body) = self.const_cache.get(&name).cloned() {
                    // Push a new scope for the constant's body (different LocalId namespace)
                    self.local_map.push_scope();
                    let mut const_env = Env::new();
                    let result = self.eval(&const_body, const_body.root, &mut const_env);
                    self.local_map.pop_scope();
                    return result;
                }
                // Otherwise, residualize
                let new_id = self.emit(Expr::Global(name), ty, span, node_id);
                Ok(Value::Unknown(new_id))
            }

            // Let binding
            Expr::Let {
                local,
                rhs,
                body: let_body,
            } => {
                let rhs_val = self.eval(body, rhs, env)?;

                // Map the local for residualization purposes
                let new_local = self.map_local(body, local);

                if rhs_val.is_known() {
                    // Bind the known value in the environment (will be inlined)
                    env.extend(local, rhs_val);
                    // Evaluate body without emitting a let
                    self.eval(body, let_body, env)
                } else {
                    // Emit a let binding for the unknown value
                    if let Value::Unknown(rhs_id) = &rhs_val {
                        // Get the RHS type - the local has the type of its RHS, not the let body
                        let rhs_ty = self.output.get_type(*rhs_id).clone();

                        // Bind the local to a reference, not the rhs directly
                        let local_ref_id = self.emit(Expr::Local(new_local), rhs_ty, span, node_id);
                        env.extend(local, Value::Unknown(local_ref_id));

                        // Evaluate the body
                        let body_val = self.eval(body, let_body, env)?;

                        // Emit the let expression wrapping the body
                        let body_id = self.reify(&body_val, ty, span, node_id);
                        let let_id = self.emit(
                            Expr::Let {
                                local: new_local,
                                rhs: *rhs_id,
                                body: body_id,
                            },
                            self.output.get_type(body_id).clone(),
                            span,
                            node_id,
                        );
                        Ok(Value::Unknown(let_id))
                    } else {
                        // Shouldn't happen, but fallback to old behavior
                        self.eval(body, let_body, env)
                    }
                }
            }

            // Binary operation
            Expr::BinOp { op, lhs, rhs } => {
                let lhs_val = self.eval(body, lhs, env)?;
                let rhs_val = self.eval(body, rhs, env)?;
                self.eval_binop(&op, lhs_val, rhs_val, ty, span, node_id)
            }

            // Unary operation
            Expr::UnaryOp { op, operand } => {
                let operand_val = self.eval(body, operand, env)?;
                self.eval_unaryop(&op, operand_val, ty, span, node_id)
            }

            // If expression
            Expr::If { cond, then_, else_ } => {
                let cond_val = self.eval(body, cond, env)?;
                self.eval_if(body, cond_val, then_, else_, env, ty, span, node_id)
            }

            // Tuple
            Expr::Tuple(elems) => {
                let elem_vals: Vec<Value> =
                    elems.iter().map(|e| self.eval(body, *e, env)).collect::<Result<_>>()?;

                if elem_vals.iter().all(|v| v.is_known()) {
                    Ok(Value::Aggregate {
                        kind: AggregateKind::Tuple,
                        elements: elem_vals,
                    })
                } else {
                    // Residualize
                    let elem_ids: Vec<ExprId> = elem_vals
                        .iter()
                        .enumerate()
                        .map(|(i, v)| {
                            let elem_ty = self.get_tuple_element_type(&ty, i);
                            self.reify(v, elem_ty, span, node_id)
                        })
                        .collect();
                    let new_id = self.emit(Expr::Tuple(elem_ids), ty, span, node_id);
                    Ok(Value::Unknown(new_id))
                }
            }

            // Array
            Expr::Array(elems) => {
                let elem_vals: Vec<Value> =
                    elems.iter().map(|e| self.eval(body, *e, env)).collect::<Result<_>>()?;

                if elem_vals.iter().all(|v| v.is_known()) {
                    Ok(Value::Aggregate {
                        kind: AggregateKind::Array,
                        elements: elem_vals,
                    })
                } else {
                    // Residualize
                    let elem_ids: Vec<ExprId> = elem_vals
                        .iter()
                        .map(|v| {
                            let elem_ty = self.get_array_element_type(&ty);
                            self.reify(v, elem_ty, span, node_id)
                        })
                        .collect();
                    let new_id = self.emit(Expr::Array(elem_ids), ty, span, node_id);
                    Ok(Value::Unknown(new_id))
                }
            }

            // Vector
            Expr::Vector(elems) => {
                let elem_vals: Vec<Value> =
                    elems.iter().map(|e| self.eval(body, *e, env)).collect::<Result<_>>()?;

                if elem_vals.iter().all(|v| v.is_known()) {
                    Ok(Value::Aggregate {
                        kind: AggregateKind::Vector,
                        elements: elem_vals,
                    })
                } else {
                    // Residualize
                    let elem_ids: Vec<ExprId> = elem_vals
                        .iter()
                        .map(|v| {
                            let elem_ty = self.get_vector_element_type(&ty);
                            self.reify(v, elem_ty, span, node_id)
                        })
                        .collect();
                    let new_id = self.emit(Expr::Vector(elem_ids), ty, span, node_id);
                    Ok(Value::Unknown(new_id))
                }
            }

            // Matrix
            Expr::Matrix(rows) => {
                // Flatten matrix into elements
                let mut all_known = true;
                let mut elem_vals = Vec::new();
                let mut row_ids: Vec<Vec<ExprId>> = Vec::new();

                for row in &rows {
                    let row_vals: Vec<Value> =
                        row.iter().map(|e| self.eval(body, *e, env)).collect::<Result<_>>()?;

                    if !row_vals.iter().all(|v| v.is_known()) {
                        all_known = false;
                    }

                    let row_elem_ids: Vec<ExprId> = row_vals
                        .iter()
                        .map(|v| {
                            let elem_ty = self.get_matrix_element_type(&ty);
                            self.reify(v, elem_ty, span, node_id)
                        })
                        .collect();
                    row_ids.push(row_elem_ids);
                    elem_vals.extend(row_vals);
                }

                if all_known {
                    Ok(Value::Aggregate {
                        kind: AggregateKind::Matrix,
                        elements: elem_vals,
                    })
                } else {
                    let new_id = self.emit(Expr::Matrix(row_ids), ty, span, node_id);
                    Ok(Value::Unknown(new_id))
                }
            }

            // Function call
            Expr::Call { func, args } => {
                let arg_vals: Vec<Value> =
                    args.iter().map(|a| self.eval(body, *a, env)).collect::<Result<_>>()?;
                // Pass original arg ExprIds so we can get their types when reifying
                self.eval_call(&func, arg_vals, &args, body, ty, span, node_id)
            }

            // Intrinsic call
            Expr::Intrinsic { name, args } => {
                let arg_vals: Vec<Value> =
                    args.iter().map(|a| self.eval(body, *a, env)).collect::<Result<_>>()?;
                // Pass original arg ExprIds so we can get their types when reifying
                self.eval_intrinsic(&name, arg_vals, &args, body, ty, span, node_id)
            }

            // Closure
            Expr::Closure {
                lambda_name,
                captures,
            } => {
                let captures_vals: Vec<Value> = captures
                    .iter()
                    .map(|c| self.eval(body, *c, env))
                    .collect::<Result<_>>()?;
                Ok(Value::Closure {
                    lambda_name,
                    captures: captures_vals,
                })
            }

            // Loop
            Expr::Loop {
                loop_var,
                init,
                init_bindings,
                kind,
                body: loop_body,
            } => self.eval_loop(
                body,
                loop_var,
                init,
                &init_bindings,
                &kind,
                loop_body,
                env,
                ty,
                span,
                node_id,
            ),

            // Range
            Expr::Range {
                start,
                step,
                end,
                kind,
            } => {
                let start_val = self.eval(body, start, env)?;
                let step_val = if let Some(s) = step { Some(self.eval(body, s, env)?) } else { None };
                let end_val = self.eval(body, end, env)?;

                // Residualize ranges for now
                let start_id = self.reify(&start_val, ty.clone(), span, node_id);
                let step_id = step_val.map(|v| self.reify(&v, ty.clone(), span, node_id));
                let end_id = self.reify(&end_val, ty.clone(), span, node_id);

                let new_id = self.emit(
                    Expr::Range {
                        start: start_id,
                        step: step_id,
                        end: end_id,
                        kind,
                    },
                    ty,
                    span,
                    node_id,
                );
                Ok(Value::Unknown(new_id))
            }

            // Materialize
            Expr::Materialize(inner) => {
                let inner_val = self.eval(body, inner, env)?;
                // ty is Ptr(inner_type), unwrap to get the actual inner type for reification
                let inner_ty = match &ty {
                    Type::Constructed(TypeName::Pointer, args) if !args.is_empty() => args[0].clone(),
                    _ => ty.clone(), // Fallback if not a pointer (shouldn't happen)
                };
                let inner_id = self.reify(&inner_val, inner_ty, span, node_id);
                let new_id = self.emit(Expr::Materialize(inner_id), ty, span, node_id);
                Ok(Value::Unknown(new_id))
            }

            // Attributed
            Expr::Attributed { attributes, expr } => {
                let inner_val = self.eval(body, expr, env)?;
                if inner_val.is_known() {
                    // Attributes don't affect known values
                    Ok(inner_val)
                } else {
                    let inner_id = self.reify(&inner_val, ty.clone(), span, node_id);
                    let new_id = self.emit(
                        Expr::Attributed {
                            attributes,
                            expr: inner_id,
                        },
                        ty,
                        span,
                        node_id,
                    );
                    Ok(Value::Unknown(new_id))
                }
            }

            // Slices - residualize for now
            Expr::OwnedSlice { data, len } => {
                let data_val = self.eval(body, data, env)?;
                let len_val = self.eval(body, len, env)?;

                let data_id = self.reify(&data_val, ty.clone(), span, node_id);
                let len_ty = Type::Constructed(TypeName::Named("i32".to_string()), vec![]);
                let len_id = self.reify(&len_val, len_ty, span, node_id);

                let new_id = self.emit(
                    Expr::OwnedSlice {
                        data: data_id,
                        len: len_id,
                    },
                    ty,
                    span,
                    node_id,
                );
                Ok(Value::Unknown(new_id))
            }

            Expr::BorrowedSlice { base, offset, len } => {
                let base_val = self.eval(body, base, env)?;
                let offset_val = self.eval(body, offset, env)?;
                let len_val = self.eval(body, len, env)?;

                let base_id = self.reify(&base_val, ty.clone(), span, node_id);
                let i32_ty = Type::Constructed(TypeName::Named("i32".to_string()), vec![]);
                let offset_id = self.reify(&offset_val, i32_ty.clone(), span, node_id);
                let len_id = self.reify(&len_val, i32_ty, span, node_id);

                let new_id = self.emit(
                    Expr::BorrowedSlice {
                        base: base_id,
                        offset: offset_id,
                        len: len_id,
                    },
                    ty,
                    span,
                    node_id,
                );
                Ok(Value::Unknown(new_id))
            }
        }
    }

    /// Map a local from the input body to the output body (in current scope)
    fn map_local(&mut self, body: &Body, local_id: LocalId) -> LocalId {
        if let Some(new_id) = self.local_map.get(local_id) {
            return new_id;
        }
        let decl = body.locals[local_id.index()].clone();
        let new_id = self.output.alloc_local(decl);
        self.local_map.insert(local_id, new_id);
        new_id
    }

    /// Extract element type from Tuple at given index
    fn get_tuple_element_type(&self, ty: &Type<TypeName>, index: usize) -> Type<TypeName> {
        match ty {
            Type::Constructed(TypeName::Tuple(_), args) => {
                args.get(index).cloned().unwrap_or_else(|| ty.clone())
            }
            _ => ty.clone(),
        }
    }

    /// Extract element type from Array<Size(n), ElemType>
    fn get_array_element_type(&self, ty: &Type<TypeName>) -> Type<TypeName> {
        match ty {
            Type::Constructed(TypeName::Array, args) if args.len() >= 2 => {
                // args[0] = Size(n), args[1] = element type
                args[1].clone()
            }
            _ => ty.clone(),
        }
    }

    /// Extract element type from Vec<Size(n), ElemType>
    fn get_vector_element_type(&self, ty: &Type<TypeName>) -> Type<TypeName> {
        match ty {
            Type::Constructed(TypeName::Vec, args) if args.len() >= 2 => {
                // args[0] = Size(n), args[1] = element type
                args[1].clone()
            }
            _ => ty.clone(),
        }
    }

    /// Extract element type from Mat<Size(cols), Size(rows), ElemType>
    fn get_matrix_element_type(&self, ty: &Type<TypeName>) -> Type<TypeName> {
        match ty {
            Type::Constructed(TypeName::Mat, args) if args.len() >= 3 => {
                // args[0] = Size(cols), args[1] = Size(rows), args[2] = element type
                args[2].clone()
            }
            _ => ty.clone(),
        }
    }

    /// Extract matrix dimensions (cols, rows) from Mat<Size(cols), Size(rows), ElemType>
    fn get_matrix_dimensions(&self, ty: &Type<TypeName>) -> (usize, usize) {
        match ty {
            Type::Constructed(TypeName::Mat, args) if args.len() >= 2 => {
                let cols = match &args[0] {
                    Type::Constructed(TypeName::Size(n), _) => *n,
                    _ => 1,
                };
                let rows = match &args[1] {
                    Type::Constructed(TypeName::Size(n), _) => *n,
                    _ => 1,
                };
                (cols, rows)
            }
            _ => (1, 1),
        }
    }

    /// Evaluate a binary operation
    fn eval_binop(
        &mut self,
        op: &str,
        lhs: Value,
        rhs: Value,
        ty: Type<TypeName>,
        span: Span,
        node_id: NodeId,
    ) -> Result<Value> {
        match (&lhs, &rhs) {
            // Both known scalars - compute result
            (Value::Scalar(l), Value::Scalar(r)) => {
                if let Some(result) = self.compute_binop(op, l, r, span)? {
                    return Ok(result);
                }
            }

            // Algebraic identities with known operands
            // x + 0 = x, 0 + x = x
            (Value::Scalar(ScalarValue::Int(0)), _) if op == "+" => return Ok(rhs),
            (_, Value::Scalar(ScalarValue::Int(0))) if op == "+" => return Ok(lhs),
            (Value::Scalar(ScalarValue::Float(f)), _) if op == "+" && *f == 0.0 => return Ok(rhs),
            (_, Value::Scalar(ScalarValue::Float(f))) if op == "+" && *f == 0.0 => return Ok(lhs),

            // x - 0 = x
            (_, Value::Scalar(ScalarValue::Int(0))) if op == "-" => return Ok(lhs),
            (_, Value::Scalar(ScalarValue::Float(f))) if op == "-" && *f == 0.0 => return Ok(lhs),

            // x * 1 = x, 1 * x = x
            (Value::Scalar(ScalarValue::Int(1)), _) if op == "*" => return Ok(rhs),
            (_, Value::Scalar(ScalarValue::Int(1))) if op == "*" => return Ok(lhs),
            (Value::Scalar(ScalarValue::Float(f)), _) if op == "*" && *f == 1.0 => return Ok(rhs),
            (_, Value::Scalar(ScalarValue::Float(f))) if op == "*" && *f == 1.0 => return Ok(lhs),

            // x * 0 = 0, 0 * x = 0
            (Value::Scalar(ScalarValue::Int(0)), _) if op == "*" => {
                return Ok(Value::Scalar(ScalarValue::Int(0)));
            }
            (_, Value::Scalar(ScalarValue::Int(0))) if op == "*" => {
                return Ok(Value::Scalar(ScalarValue::Int(0)));
            }
            (Value::Scalar(ScalarValue::Float(f)), _) if op == "*" && *f == 0.0 => {
                return Ok(Value::Scalar(ScalarValue::Float(0.0)));
            }
            (_, Value::Scalar(ScalarValue::Float(f))) if op == "*" && *f == 0.0 => {
                return Ok(Value::Scalar(ScalarValue::Float(0.0)));
            }

            // x / 1 = x
            (_, Value::Scalar(ScalarValue::Int(1))) if op == "/" => return Ok(lhs),
            (_, Value::Scalar(ScalarValue::Float(f))) if op == "/" && *f == 1.0 => return Ok(lhs),

            // Short-circuit boolean operations
            (Value::Scalar(ScalarValue::Bool(true)), _) if op == "||" => {
                return Ok(Value::Scalar(ScalarValue::Bool(true)));
            }
            (Value::Scalar(ScalarValue::Bool(false)), _) if op == "&&" => {
                return Ok(Value::Scalar(ScalarValue::Bool(false)));
            }
            (Value::Scalar(ScalarValue::Bool(false)), _) if op == "||" => return Ok(rhs),
            (Value::Scalar(ScalarValue::Bool(true)), _) if op == "&&" => return Ok(rhs),

            _ => {}
        }

        // Residualize
        let lhs_ty = ty.clone();
        let rhs_ty = ty.clone();
        let lhs_id = self.reify(&lhs, lhs_ty, span, node_id);
        let rhs_id = self.reify(&rhs, rhs_ty, span, node_id);
        let new_id = self.emit(
            Expr::BinOp {
                op: op.to_string(),
                lhs: lhs_id,
                rhs: rhs_id,
            },
            ty,
            span,
            node_id,
        );
        Ok(Value::Unknown(new_id))
    }

    /// Compute a binary operation on two known scalar values
    fn compute_binop(
        &self,
        op: &str,
        lhs: &ScalarValue,
        rhs: &ScalarValue,
        span: Span,
    ) -> Result<Option<Value>> {
        match (lhs, rhs) {
            (ScalarValue::Int(l), ScalarValue::Int(r)) => {
                let result = match op {
                    "+" => Some(Value::Scalar(ScalarValue::Int(l + r))),
                    "-" => Some(Value::Scalar(ScalarValue::Int(l - r))),
                    "*" => Some(Value::Scalar(ScalarValue::Int(l * r))),
                    "/" => {
                        if *r == 0 {
                            bail_type_at!(span, "Division by zero in constant expression");
                        }
                        Some(Value::Scalar(ScalarValue::Int(l / r)))
                    }
                    "%" => {
                        if *r == 0 {
                            bail_type_at!(span, "Modulo by zero in constant expression");
                        }
                        Some(Value::Scalar(ScalarValue::Int(l % r)))
                    }
                    "==" => Some(Value::Scalar(ScalarValue::Bool(l == r))),
                    "!=" => Some(Value::Scalar(ScalarValue::Bool(l != r))),
                    "<" => Some(Value::Scalar(ScalarValue::Bool(l < r))),
                    "<=" => Some(Value::Scalar(ScalarValue::Bool(l <= r))),
                    ">" => Some(Value::Scalar(ScalarValue::Bool(l > r))),
                    ">=" => Some(Value::Scalar(ScalarValue::Bool(l >= r))),
                    _ => None,
                };
                Ok(result)
            }
            (ScalarValue::Float(l), ScalarValue::Float(r)) => {
                let result = match op {
                    "+" => Some(Value::Scalar(ScalarValue::Float(l + r))),
                    "-" => Some(Value::Scalar(ScalarValue::Float(l - r))),
                    "*" => Some(Value::Scalar(ScalarValue::Float(l * r))),
                    "/" => {
                        if *r == 0.0 {
                            bail_type_at!(span, "Division by zero in constant expression");
                        }
                        Some(Value::Scalar(ScalarValue::Float(l / r)))
                    }
                    "==" => Some(Value::Scalar(ScalarValue::Bool(l == r))),
                    "!=" => Some(Value::Scalar(ScalarValue::Bool(l != r))),
                    "<" => Some(Value::Scalar(ScalarValue::Bool(l < r))),
                    "<=" => Some(Value::Scalar(ScalarValue::Bool(l <= r))),
                    ">" => Some(Value::Scalar(ScalarValue::Bool(l > r))),
                    ">=" => Some(Value::Scalar(ScalarValue::Bool(l >= r))),
                    _ => None,
                };
                Ok(result)
            }
            (ScalarValue::Bool(l), ScalarValue::Bool(r)) => {
                let result = match op {
                    "&&" => Some(Value::Scalar(ScalarValue::Bool(*l && *r))),
                    "||" => Some(Value::Scalar(ScalarValue::Bool(*l || *r))),
                    "==" => Some(Value::Scalar(ScalarValue::Bool(l == r))),
                    "!=" => Some(Value::Scalar(ScalarValue::Bool(l != r))),
                    _ => None,
                };
                Ok(result)
            }
            _ => Ok(None),
        }
    }

    /// Evaluate a unary operation
    fn eval_unaryop(
        &mut self,
        op: &str,
        operand: Value,
        ty: Type<TypeName>,
        span: Span,
        node_id: NodeId,
    ) -> Result<Value> {
        match (&operand, op) {
            (Value::Scalar(ScalarValue::Int(v)), "-") => Ok(Value::Scalar(ScalarValue::Int(-v))),
            (Value::Scalar(ScalarValue::Float(v)), "-") => Ok(Value::Scalar(ScalarValue::Float(-v))),
            (Value::Scalar(ScalarValue::Bool(v)), "!") => Ok(Value::Scalar(ScalarValue::Bool(!v))),
            _ => {
                // Residualize
                let operand_id = self.reify(&operand, ty.clone(), span, node_id);
                let new_id = self.emit(
                    Expr::UnaryOp {
                        op: op.to_string(),
                        operand: operand_id,
                    },
                    ty,
                    span,
                    node_id,
                );
                Ok(Value::Unknown(new_id))
            }
        }
    }

    /// Evaluate an if expression
    fn eval_if(
        &mut self,
        body: &Body,
        cond_val: Value,
        then_id: ExprId,
        else_id: ExprId,
        env: &mut Env,
        ty: Type<TypeName>,
        span: Span,
        node_id: NodeId,
    ) -> Result<Value> {
        match cond_val {
            // Known condition - evaluate only the taken branch
            Value::Scalar(ScalarValue::Bool(true)) => self.eval(body, then_id, env),
            Value::Scalar(ScalarValue::Bool(false)) => self.eval(body, else_id, env),

            // Unknown condition - evaluate both branches and residualize
            Value::Unknown(cond_expr_id) => {
                let mut then_env = env.child();
                let mut else_env = env.child();

                let then_val = self.eval(body, then_id, &mut then_env)?;
                let else_val = self.eval(body, else_id, &mut else_env)?;

                let then_result_id = self.reify(&then_val, ty.clone(), span, node_id);
                let else_result_id = self.reify(&else_val, ty.clone(), span, node_id);

                let if_id = self.emit(
                    Expr::If {
                        cond: cond_expr_id,
                        then_: then_result_id,
                        else_: else_result_id,
                    },
                    ty,
                    span,
                    node_id,
                );

                Ok(Value::Unknown(if_id))
            }

            _ => {
                // Non-boolean condition shouldn't happen after type checking
                bail_type_at!(span, "Non-boolean condition in if expression");
            }
        }
    }

    /// Evaluate a function call
    fn eval_call(
        &mut self,
        func: &str,
        args: Vec<Value>,
        arg_expr_ids: &[ExprId],
        body: &Body,
        ty: Type<TypeName>,
        span: Span,
        node_id: NodeId,
    ) -> Result<Value> {
        let all_known = args.iter().all(|v| v.is_known());

        // Try to inline if all arguments are known and we haven't exceeded inline depth
        if all_known && self.cost_budget.inline_depth < self.cost_budget.max_inline_depth {
            if let Some(func_info) = self.func_cache.get(func).cloned() {
                self.cost_budget.inline_depth += 1;

                // Push a new scope for the inlined function's body (different LocalId namespace)
                self.local_map.push_scope();

                // Build environment with argument bindings
                let mut call_env = Env::new();
                for (param_id, arg_val) in func_info.params.iter().zip(args.iter()) {
                    call_env.extend(*param_id, arg_val.clone());
                }

                // Evaluate the function body
                let result = self.eval(&func_info.body, func_info.body.root, &mut call_env);

                self.local_map.pop_scope();
                self.cost_budget.inline_depth -= 1;

                return result;
            }
        }

        // Can't inline - residualize using each argument's original type
        let arg_ids: Vec<ExprId> = args
            .iter()
            .zip(arg_expr_ids.iter())
            .map(|(v, &orig_id)| {
                let arg_ty = body.get_type(orig_id).clone();
                self.reify(v, arg_ty, span, node_id)
            })
            .collect();

        let call_id = self.emit(
            Expr::Call {
                func: func.to_string(),
                args: arg_ids,
            },
            ty,
            span,
            node_id,
        );

        Ok(Value::Unknown(call_id))
    }

    /// Evaluate an intrinsic call
    fn eval_intrinsic(
        &mut self,
        name: &str,
        args: Vec<Value>,
        arg_expr_ids: &[ExprId],
        body: &Body,
        ty: Type<TypeName>,
        span: Span,
        node_id: NodeId,
    ) -> Result<Value> {
        let all_known = args.iter().all(|v| v.is_known());

        // Try to evaluate pure intrinsics on known arguments
        if all_known {
            match name {
                // Tuple access
                "tuple_access" if args.len() == 2 => {
                    if let (
                        Value::Aggregate {
                            kind: AggregateKind::Tuple,
                            elements,
                        },
                        Value::Scalar(ScalarValue::Int(idx)),
                    ) = (&args[0], &args[1])
                    {
                        if let Some(elem) = elements.get(*idx as usize) {
                            return Ok(elem.clone());
                        }
                    }
                }

                // Math intrinsics on floats
                "sqrt" if args.len() == 1 => {
                    if let Value::Scalar(ScalarValue::Float(v)) = &args[0] {
                        return Ok(Value::Scalar(ScalarValue::Float(v.sqrt())));
                    }
                }
                "abs" if args.len() == 1 => match &args[0] {
                    Value::Scalar(ScalarValue::Float(v)) => {
                        return Ok(Value::Scalar(ScalarValue::Float(v.abs())));
                    }
                    Value::Scalar(ScalarValue::Int(v)) => {
                        return Ok(Value::Scalar(ScalarValue::Int(v.abs())));
                    }
                    _ => {}
                },
                "sin" if args.len() == 1 => {
                    if let Value::Scalar(ScalarValue::Float(v)) = &args[0] {
                        return Ok(Value::Scalar(ScalarValue::Float(v.sin())));
                    }
                }
                "cos" if args.len() == 1 => {
                    if let Value::Scalar(ScalarValue::Float(v)) = &args[0] {
                        return Ok(Value::Scalar(ScalarValue::Float(v.cos())));
                    }
                }
                "tan" if args.len() == 1 => {
                    if let Value::Scalar(ScalarValue::Float(v)) = &args[0] {
                        return Ok(Value::Scalar(ScalarValue::Float(v.tan())));
                    }
                }
                "exp" if args.len() == 1 => {
                    if let Value::Scalar(ScalarValue::Float(v)) = &args[0] {
                        return Ok(Value::Scalar(ScalarValue::Float(v.exp())));
                    }
                }
                "log" if args.len() == 1 => {
                    if let Value::Scalar(ScalarValue::Float(v)) = &args[0] {
                        return Ok(Value::Scalar(ScalarValue::Float(v.ln())));
                    }
                }
                "floor" if args.len() == 1 => {
                    if let Value::Scalar(ScalarValue::Float(v)) = &args[0] {
                        return Ok(Value::Scalar(ScalarValue::Float(v.floor())));
                    }
                }
                "ceil" if args.len() == 1 => {
                    if let Value::Scalar(ScalarValue::Float(v)) = &args[0] {
                        return Ok(Value::Scalar(ScalarValue::Float(v.ceil())));
                    }
                }
                "min" if args.len() == 2 => match (&args[0], &args[1]) {
                    (Value::Scalar(ScalarValue::Float(a)), Value::Scalar(ScalarValue::Float(b))) => {
                        return Ok(Value::Scalar(ScalarValue::Float(a.min(*b))));
                    }
                    (Value::Scalar(ScalarValue::Int(a)), Value::Scalar(ScalarValue::Int(b))) => {
                        return Ok(Value::Scalar(ScalarValue::Int(*a.min(b))));
                    }
                    _ => {}
                },
                "max" if args.len() == 2 => match (&args[0], &args[1]) {
                    (Value::Scalar(ScalarValue::Float(a)), Value::Scalar(ScalarValue::Float(b))) => {
                        return Ok(Value::Scalar(ScalarValue::Float(a.max(*b))));
                    }
                    (Value::Scalar(ScalarValue::Int(a)), Value::Scalar(ScalarValue::Int(b))) => {
                        return Ok(Value::Scalar(ScalarValue::Int(*a.max(b))));
                    }
                    _ => {}
                },
                "clamp" if args.len() == 3 => match (&args[0], &args[1], &args[2]) {
                    (
                        Value::Scalar(ScalarValue::Float(v)),
                        Value::Scalar(ScalarValue::Float(lo)),
                        Value::Scalar(ScalarValue::Float(hi)),
                    ) => {
                        return Ok(Value::Scalar(ScalarValue::Float(v.clamp(*lo, *hi))));
                    }
                    (
                        Value::Scalar(ScalarValue::Int(v)),
                        Value::Scalar(ScalarValue::Int(lo)),
                        Value::Scalar(ScalarValue::Int(hi)),
                    ) => {
                        return Ok(Value::Scalar(ScalarValue::Int(*v.clamp(lo, hi))));
                    }
                    _ => {}
                },

                _ => {}
            }
        }

        // Residualize - use each argument's original type, not the result type
        let arg_ids: Vec<ExprId> = args
            .iter()
            .zip(arg_expr_ids.iter())
            .map(|(v, &orig_id)| {
                let arg_ty = body.get_type(orig_id).clone();
                self.reify(v, arg_ty, span, node_id)
            })
            .collect();

        let intrinsic_id = self.emit(
            Expr::Intrinsic {
                name: name.to_string(),
                args: arg_ids,
            },
            ty,
            span,
            node_id,
        );

        Ok(Value::Unknown(intrinsic_id))
    }

    /// Evaluate a loop
    fn eval_loop(
        &mut self,
        body: &Body,
        loop_var: LocalId,
        init: ExprId,
        init_bindings: &[(LocalId, ExprId)],
        kind: &LoopKind,
        loop_body: ExprId,
        env: &mut Env,
        ty: Type<TypeName>,
        span: Span,
        node_id: NodeId,
    ) -> Result<Value> {
        let init_val = self.eval(body, init, env)?;

        // Check if we can unroll the loop
        match kind {
            LoopKind::ForRange { var, bound } => {
                let bound_val = self.eval(body, *bound, env)?;

                // If bound is known and small enough, unroll
                if let Value::Scalar(ScalarValue::Int(n)) = bound_val {
                    if n >= 0 && (n as usize) <= self.cost_budget.max_loop_unroll {
                        return self.unroll_for_range(
                            body,
                            loop_var,
                            init_val,
                            init_bindings,
                            *var,
                            n as usize,
                            loop_body,
                            env,
                            ty,
                            span,
                            node_id,
                        );
                    }
                }
            }

            LoopKind::For { var, iter } => {
                let iter_val = self.eval(body, *iter, env)?;

                // If array is fully known and small, unroll
                if let Value::Aggregate {
                    kind: AggregateKind::Array,
                    elements,
                } = &iter_val
                {
                    if elements.len() <= self.cost_budget.max_loop_unroll {
                        return self.unroll_for_in(
                            body,
                            loop_var,
                            init_val,
                            init_bindings,
                            *var,
                            elements,
                            loop_body,
                            env,
                            ty,
                            span,
                            node_id,
                        );
                    }
                }
            }

            LoopKind::While { .. } => {
                // While loops are harder to unroll - skip for now
            }
        }

        // Can't unroll - residualize the entire loop
        self.residualize_loop(
            body,
            loop_var,
            init_val,
            init_bindings,
            kind,
            loop_body,
            env,
            ty,
            span,
            node_id,
        )
    }

    /// Unroll a for-range loop
    fn unroll_for_range(
        &mut self,
        body: &Body,
        loop_var: LocalId,
        mut acc: Value,
        init_bindings: &[(LocalId, ExprId)],
        iter_var: LocalId,
        bound: usize,
        loop_body: ExprId,
        env: &mut Env,
        _ty: Type<TypeName>,
        _span: Span,
        _node_id: NodeId,
    ) -> Result<Value> {
        for i in 0..bound {
            // Bind loop accumulator
            env.extend(loop_var, acc.clone());

            // Bind init_bindings (these extract from loop_var)
            for (local, binding_expr) in init_bindings {
                let binding_val = self.eval(body, *binding_expr, env)?;
                env.extend(*local, binding_val);
            }

            // Bind iteration variable
            env.extend(iter_var, Value::Scalar(ScalarValue::Int(i as i64)));

            // Evaluate body
            acc = self.eval(body, loop_body, env)?;
        }

        Ok(acc)
    }

    /// Unroll a for-in loop
    fn unroll_for_in(
        &mut self,
        body: &Body,
        loop_var: LocalId,
        mut acc: Value,
        init_bindings: &[(LocalId, ExprId)],
        iter_var: LocalId,
        elements: &[Value],
        loop_body: ExprId,
        env: &mut Env,
        _ty: Type<TypeName>,
        _span: Span,
        _node_id: NodeId,
    ) -> Result<Value> {
        for elem in elements {
            // Bind loop accumulator
            env.extend(loop_var, acc.clone());

            // Bind init_bindings (these extract from loop_var)
            for (local, binding_expr) in init_bindings {
                let binding_val = self.eval(body, *binding_expr, env)?;
                env.extend(*local, binding_val);
            }

            // Bind iteration variable
            env.extend(iter_var, elem.clone());

            // Evaluate body
            acc = self.eval(body, loop_body, env)?;
        }

        Ok(acc)
    }

    /// Residualize a loop (emit it to output when we can't unroll)
    fn residualize_loop(
        &mut self,
        body: &Body,
        loop_var: LocalId,
        init_val: Value,
        init_bindings: &[(LocalId, ExprId)],
        kind: &LoopKind,
        loop_body: ExprId,
        env: &mut Env,
        ty: Type<TypeName>,
        span: Span,
        node_id: NodeId,
    ) -> Result<Value> {
        // Map all the locals
        let new_loop_var = self.map_local(body, loop_var);

        // Reify init value
        let loop_var_ty = body.locals[loop_var.index()].ty.clone();
        let init_id = self.reify(&init_val, loop_var_ty.clone(), span, node_id);

        // Process init bindings
        let mut new_init_bindings = Vec::new();
        for (local, binding_expr) in init_bindings {
            let new_local = self.map_local(body, *local);
            // Evaluate the binding expression in the context where loop_var is unknown
            env.extend(
                loop_var,
                Value::Unknown(self.emit(Expr::Local(new_loop_var), loop_var_ty.clone(), span, node_id)),
            );
            let binding_val = self.eval(body, *binding_expr, env)?;
            let binding_id = self.reify(&binding_val, ty.clone(), span, node_id);
            new_init_bindings.push((new_local, binding_id));
            env.extend(*local, binding_val);
        }

        // Map loop kind
        let new_kind = match kind {
            LoopKind::For { var, iter } => {
                let new_var = self.map_local(body, *var);
                let iter_val = self.eval(body, *iter, env)?;
                let iter_id = self.reify(&iter_val, ty.clone(), span, node_id);
                LoopKind::For {
                    var: new_var,
                    iter: iter_id,
                }
            }
            LoopKind::ForRange { var, bound } => {
                let new_var = self.map_local(body, *var);
                let bound_val = self.eval(body, *bound, env)?;
                let i32_ty = Type::Constructed(TypeName::Named("i32".to_string()), vec![]);
                let bound_id = self.reify(&bound_val, i32_ty, span, node_id);
                LoopKind::ForRange {
                    var: new_var,
                    bound: bound_id,
                }
            }
            LoopKind::While { cond } => {
                let cond_val = self.eval(body, *cond, env)?;
                let bool_ty = Type::Constructed(TypeName::Named("bool".to_string()), vec![]);
                let cond_id = self.reify(&cond_val, bool_ty, span, node_id);
                LoopKind::While { cond: cond_id }
            }
        };

        // Evaluate loop body with unknowns for loop variables
        env.extend(
            loop_var,
            Value::Unknown(self.emit(Expr::Local(new_loop_var), loop_var_ty, span, node_id)),
        );
        let body_val = self.eval(body, loop_body, env)?;
        let body_id = self.reify(&body_val, ty.clone(), span, node_id);

        let loop_id = self.emit(
            Expr::Loop {
                loop_var: new_loop_var,
                init: init_id,
                init_bindings: new_init_bindings,
                kind: new_kind,
                body: body_id,
            },
            ty,
            span,
            node_id,
        );

        Ok(Value::Unknown(loop_id))
    }

    /// Evaluate a function body, returning the new body
    fn eval_body(&mut self, old_body: Body) -> Result<Body> {
        self.output = Body::new();
        self.local_map.reset();

        // Copy locals (parameters and let-bindings)
        for local in &old_body.locals {
            self.output.alloc_local(local.clone());
        }

        // Set up the local_map for identity mapping initially
        for i in 0..old_body.locals.len() {
            let id = LocalId(i as u32);
            self.local_map.insert(id, id);
        }

        // Create environment with parameters as unknowns
        let mut env = Env::new();
        for (i, local) in old_body.locals.iter().enumerate() {
            if local.kind == LocalKind::Param {
                let local_id = LocalId(i as u32);
                let expr_id = self.emit(Expr::Local(local_id), local.ty.clone(), local.span, NodeId(0));
                env.extend(local_id, Value::Unknown(expr_id));
            }
        }

        // Evaluate the body
        let result = self.eval(&old_body, old_body.root, &mut env)?;

        // Reify the result as the new root
        let root_ty = old_body.get_type(old_body.root).clone();
        let root_span = old_body.get_span(old_body.root);
        let root_node_id = old_body.get_node_id(old_body.root);
        self.output.root = self.reify(&result, root_ty, root_span, root_node_id);

        Ok(std::mem::take(&mut self.output))
    }
}

// =============================================================================
// Entry point
// =============================================================================

/// Partially evaluate a MIR program.
pub fn partial_eval(program: Program) -> Result<Program> {
    let mut evaluator = PartialEvaluator::new(&program);
    let mut new_defs = Vec::new();

    for def in program.defs {
        let new_def = match def {
            Def::Function {
                id,
                name,
                params,
                ret_type,
                scheme,
                attributes,
                body,
                span,
            } => {
                let new_body = evaluator.eval_body(body)?;
                Def::Function {
                    id,
                    name,
                    params,
                    ret_type,
                    scheme,
                    attributes,
                    body: new_body,
                    span,
                }
            }
            Def::Constant {
                id,
                name,
                ty,
                attributes,
                body,
                span,
            } => {
                let new_body = evaluator.eval_body(body)?;
                Def::Constant {
                    id,
                    name,
                    ty,
                    attributes,
                    body: new_body,
                    span,
                }
            }
            Def::EntryPoint {
                id,
                name,
                execution_model,
                inputs,
                outputs,
                body,
                span,
            } => {
                let new_body = evaluator.eval_body(body)?;
                Def::EntryPoint {
                    id,
                    name,
                    execution_model,
                    inputs,
                    outputs,
                    body: new_body,
                    span,
                }
            }
            // Uniforms and storage have no body to evaluate
            Def::Uniform { .. } | Def::Storage { .. } => def,
        };
        new_defs.push(new_def);
    }

    Ok(Program {
        defs: new_defs,
        lambda_registry: program.lambda_registry,
    })
}
