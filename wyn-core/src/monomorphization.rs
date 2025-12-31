//! Monomorphization pass for MIR
//!
//! This pass takes polymorphic functions (with size/type variables) and creates
//! specialized monomorphic copies for each concrete instantiation that's actually called.
//!
//! Example:
//!   def sum [n] (arr:[n]f32) : f32 = ...
//!
//! When called with [4]f32, creates:
//!   def sum$n4 (arr:[4]f32) : f32 = ...
//!
//! This happens after type checking and flattening, before lowering.

use crate::IdArena;
use crate::ast::TypeName;
use crate::error::Result;
use crate::mir::{Body, Def, Expr, ExprId, LocalDecl, LocalId, MemBinding, Program};
use crate::mir::{LambdaId, LambdaInfo};
use crate::types::TypeScheme;
use polytype::Type;
use std::collections::{HashMap, HashSet, VecDeque};

/// A substitution mapping type variables to concrete types
type Substitution = HashMap<usize, Type<TypeName>>;

/// Monomorphize a MIR program
pub fn monomorphize(program: Program) -> Result<Program> {
    let mono = Monomorphizer::new(program);
    mono.run()
}

struct Monomorphizer {
    /// Original polymorphic functions by name
    poly_functions: HashMap<String, Def>,
    /// Generated monomorphic functions
    mono_functions: Vec<Def>,
    /// Map from (function_name, spec_key) to specialized name
    specializations: HashMap<(String, SpecKey), String>,
    /// Worklist of functions to process
    worklist: VecDeque<WorkItem>,
    /// Functions already processed
    processed: HashSet<String>,
    /// Lambda registry from original program
    lambda_registry: IdArena<LambdaId, LambdaInfo>,
}

struct WorkItem {
    /// Name of the function to process
    name: String,
    /// The function definition
    def: Def,
}

/// A key for looking up specializations
/// We use a sorted Vec instead of HashMap for deterministic ordering
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct SubstKey(Vec<(usize, TypeKey)>);

/// Combined specialization key: type substitution + memory bindings.
/// A function may need specialization due to type variables, memory bindings, or both.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SpecKey {
    /// Type substitution (may be empty for mem-only specialization)
    type_subst: SubstKey,
    /// Memory binding for each parameter (None = value/SSA)
    mem_bindings: Vec<Option<MemBinding>>,
}

/// A simplified representation of types for use as hash keys
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum TypeKey {
    Var(usize),
    Size(usize),
    Constructed(String, Vec<TypeKey>),
    Record(Vec<(String, TypeKey)>),
    Sum(Vec<(String, Vec<TypeKey>)>),
    Existential(Vec<String>, Box<TypeKey>),
}

impl SubstKey {
    fn from_subst(subst: &Substitution) -> Self {
        let mut items: Vec<_> = subst.iter().map(|(k, v)| (*k, TypeKey::from_type(v))).collect();
        items.sort();
        SubstKey(items)
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Convert back to a Substitution for use in specialize_def
    fn to_subst(&self) -> Substitution {
        self.0.iter().map(|(k, v)| (*k, v.to_type())).collect()
    }
}

impl SpecKey {
    fn new(subst: &Substitution, mem_bindings: Vec<Option<MemBinding>>) -> Self {
        SpecKey {
            type_subst: SubstKey::from_subst(subst),
            mem_bindings,
        }
    }

    /// Returns true if this represents a non-trivial specialization
    fn needs_specialization(&self) -> bool {
        !self.type_subst.is_empty() || self.mem_bindings.iter().any(|m| m.is_some())
    }
}

impl TypeKey {
    fn from_type(ty: &Type<TypeName>) -> Self {
        match ty {
            Type::Variable(id) => TypeKey::Var(*id),
            Type::Constructed(name, args) => {
                // Handle types with nested structure that need full representation
                match name {
                    TypeName::Size(n) => return TypeKey::Size(*n),
                    TypeName::Record(fields) => {
                        // Field names are in RecordFields, field types are in args
                        let mut key_fields: Vec<_> = fields
                            .iter()
                            .zip(args.iter())
                            .map(|(k, v)| (k.clone(), TypeKey::from_type(v)))
                            .collect();
                        key_fields.sort_by(|a, b| a.0.cmp(&b.0));
                        return TypeKey::Record(key_fields);
                    }
                    TypeName::Sum(variants) => {
                        let key_variants: Vec<_> = variants
                            .iter()
                            .map(|(name, types)| {
                                (name.clone(), types.iter().map(TypeKey::from_type).collect())
                            })
                            .collect();
                        return TypeKey::Sum(key_variants);
                    }
                    TypeName::Existential(vars) => {
                        let inner = &args[0];
                        return TypeKey::Existential(vars.clone(), Box::new(TypeKey::from_type(inner)));
                    }
                    _ => {}
                }

                // For other constructed types, use a string name + args
                let name_str = match name {
                    TypeName::Str(s) => s.to_string(),
                    TypeName::Float(bits) => format!("f{}", bits),
                    TypeName::UInt(bits) => format!("u{}", bits),
                    TypeName::Int(bits) => format!("i{}", bits),
                    TypeName::Array => "array".to_string(),
                    TypeName::Vec => "vec".to_string(),
                    TypeName::Mat => "mat".to_string(),
                    TypeName::Unsized => "unsized".to_string(),
                    TypeName::Arrow => "arrow".to_string(),
                    TypeName::SizeVar(s) => format!("sizevar_{}", s),
                    TypeName::UserVar(s) => format!("uservar_{}", s),
                    TypeName::Named(s) => s.clone(),
                    TypeName::Unique => {
                        return TypeKey::Constructed(
                            "unique".to_string(),
                            args.iter().map(TypeKey::from_type).collect(),
                        );
                    }
                    TypeName::Unit => "unit".to_string(),
                    TypeName::Tuple(n) => format!("tuple{}", n),
                    TypeName::Pointer => "ptr".to_string(),
                    _ => unreachable!("Should have been handled above: {:?}", name),
                };
                TypeKey::Constructed(name_str, args.iter().map(TypeKey::from_type).collect())
            }
        }
    }

    /// Convert back to a Type. Used for reconstructing substitutions.
    fn to_type(&self) -> Type<TypeName> {
        match self {
            TypeKey::Var(id) => Type::Variable(*id),
            TypeKey::Size(n) => Type::Constructed(TypeName::Size(*n), vec![]),
            TypeKey::Record(fields) => {
                let field_names: Vec<_> = fields.iter().map(|(k, _)| k.clone()).collect();
                let field_types: Vec<_> = fields.iter().map(|(_, v)| v.to_type()).collect();
                Type::Constructed(TypeName::Record(field_names.into()), field_types)
            }
            TypeKey::Sum(variants) => {
                let variant_types: Vec<_> = variants
                    .iter()
                    .map(|(name, types)| (name.clone(), types.iter().map(|t| t.to_type()).collect()))
                    .collect();
                Type::Constructed(TypeName::Sum(variant_types), vec![])
            }
            TypeKey::Existential(vars, inner) => {
                Type::Constructed(TypeName::Existential(vars.clone()), vec![inner.to_type()])
            }
            TypeKey::Constructed(name, args) => {
                let type_args: Vec<_> = args.iter().map(|a| a.to_type()).collect();
                let type_name = match name.as_str() {
                    "f16" => TypeName::Float(16),
                    "f32" => TypeName::Float(32),
                    "f64" => TypeName::Float(64),
                    "u8" => TypeName::UInt(8),
                    "u16" => TypeName::UInt(16),
                    "u32" => TypeName::UInt(32),
                    "u64" => TypeName::UInt(64),
                    "i8" => TypeName::Int(8),
                    "i16" => TypeName::Int(16),
                    "i32" => TypeName::Int(32),
                    "i64" => TypeName::Int(64),
                    "array" => TypeName::Array,
                    "vec" => TypeName::Vec,
                    "mat" => TypeName::Mat,
                    "unsized" => TypeName::Unsized,
                    "arrow" => TypeName::Arrow,
                    "unit" => TypeName::Unit,
                    "ptr" => TypeName::Pointer,
                    "unique" => TypeName::Unique,
                    s if s.starts_with("tuple") => {
                        let n: usize = s[5..].parse().unwrap_or(0);
                        TypeName::Tuple(n)
                    }
                    s if s.starts_with("sizevar_") => TypeName::SizeVar(s[8..].to_string()),
                    s if s.starts_with("uservar_") => TypeName::UserVar(s[8..].to_string()),
                    s => TypeName::Named(s.to_string()),
                };
                Type::Constructed(type_name, type_args)
            }
        }
    }
}

// =============================================================================
// Scheme Instantiation Helpers
// =============================================================================

/// Unwrap a TypeScheme to get the inner monotype.
/// The scheme's bound variables are already unique within the scheme,
/// so we can use them directly for unification and substitution.
fn unwrap_scheme(scheme: &TypeScheme) -> &Type<TypeName> {
    match scheme {
        TypeScheme::Monotype(ty) => ty,
        TypeScheme::Polytype { body, .. } => unwrap_scheme(body),
    }
}

/// Split a function type into (param_types, return_type).
/// Handles curried function types: (A -> B -> C) becomes ([A, B], C)
fn split_function_type(ty: &Type<TypeName>) -> (Vec<Type<TypeName>>, Type<TypeName>) {
    let mut params = Vec::new();
    let mut current = ty.clone();

    while let Type::Constructed(TypeName::Arrow, ref args) = current {
        if args.len() == 2 {
            params.push(args[0].clone());
            current = args[1].clone();
        } else {
            break;
        }
    }

    (params, current)
}

impl Monomorphizer {
    fn new(program: Program) -> Self {
        // First pass: collect storage buffer declarations
        let storage_buffers: HashMap<String, (u32, u32)> = program
            .defs
            .iter()
            .filter_map(|def| match def {
                Def::Storage {
                    name, set, binding, ..
                } => Some((name.clone(), (*set, *binding))),
                _ => None,
            })
            .collect();

        // Second pass: build function map and collect entry points with mem bindings
        let mut poly_functions = HashMap::new();
        let mut entry_points = Vec::new();

        for mut def in program.defs {
            let name = match &def {
                Def::Function { name, .. } => name.clone(),
                Def::Constant { name, .. } => name.clone(),
                Def::Uniform { name, .. } => name.clone(),
                Def::Storage { name, .. } => name.clone(),
                Def::EntryPoint { name, .. } => name.clone(),
            };

            // For entry points, set mem bindings on storage-backed parameters
            if let Def::EntryPoint { inputs, body, .. } = &mut def {
                for input in inputs {
                    if let Some(&(set, binding)) = storage_buffers.get(&input.name) {
                        body.get_local_mut(input.local).mem = Some(MemBinding::Storage { set, binding });
                    }
                }
                entry_points.push(WorkItem {
                    name: name.clone(),
                    def: def.clone(),
                });
            }

            poly_functions.insert(name, def);
        }

        let mut worklist = VecDeque::new();
        worklist.extend(entry_points);

        Monomorphizer {
            poly_functions,
            mono_functions: Vec::new(),
            specializations: HashMap::new(),
            worklist,
            processed: HashSet::new(),
            lambda_registry: program.lambda_registry,
        }
    }

    fn run(mut self) -> Result<Program> {
        while let Some(work_item) = self.worklist.pop_front() {
            if self.processed.contains(&work_item.name) {
                continue;
            }
            self.processed.insert(work_item.name.clone());

            // Process this function: look for calls and specialize callees
            let def = self.process_function(work_item.def)?;
            self.mono_functions.push(def);
        }

        Ok(Program {
            defs: self.mono_functions,
            lambda_registry: self.lambda_registry,
        })
    }

    /// Ensure a definition is in the worklist (for monomorphic callees and constants)
    fn ensure_in_worklist(&mut self, name: &str, def: Def) {
        if !self.processed.contains(name) {
            // Check if it's already in the worklist
            let already_queued = self.worklist.iter().any(|w| w.name == name);
            if !already_queued {
                self.worklist.push_back(WorkItem {
                    name: name.to_string(),
                    def,
                });
            }
        }
    }

    fn process_function(&mut self, def: Def) -> Result<Def> {
        match def {
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
                let body = self.process_body(body)?;
                Ok(Def::Function {
                    id,
                    name,
                    params,
                    ret_type,
                    scheme,
                    attributes,
                    body,
                    span,
                })
            }
            Def::Constant {
                id,
                name,
                ty,
                attributes,
                body,
                span,
            } => {
                let body = self.process_body(body)?;
                Ok(Def::Constant {
                    id,
                    name,
                    ty,
                    attributes,
                    body,
                    span,
                })
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
                let body = self.process_body(body)?;
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
            Def::Uniform { .. } => Ok(def),
            Def::Storage { .. } => Ok(def),
        }
    }

    fn process_body(&mut self, old_body: Body) -> Result<Body> {
        let mut new_body = Body::new();

        // Copy locals
        for local in &old_body.locals {
            new_body.alloc_local(local.clone());
        }

        // Map old ExprIds to new ExprIds
        let mut expr_map: HashMap<ExprId, ExprId> = HashMap::new();

        // Process expressions in order
        for (old_idx, old_expr) in old_body.exprs.iter().enumerate() {
            let old_id = ExprId(old_idx as u32);
            let ty = old_body.get_type(old_id).clone();
            let span = old_body.get_span(old_id);
            let node_id = old_body.get_node_id(old_id);

            let new_expr = self.process_expr(&old_body, old_expr, &expr_map, &ty)?;
            let new_id = new_body.alloc_expr(new_expr, ty, span, node_id);
            expr_map.insert(old_id, new_id);
        }

        new_body.root = expr_map[&old_body.root];
        Ok(new_body)
    }

    fn process_expr(
        &mut self,
        body: &Body,
        expr: &Expr,
        expr_map: &HashMap<ExprId, ExprId>,
        expr_type: &Type<TypeName>,
    ) -> Result<Expr> {
        match expr {
            Expr::Call { func, args } => {
                // Map arguments
                let new_args: Vec<_> = args.iter().map(|a| expr_map[a]).collect();

                // Extract mem bindings from arguments
                let mem_bindings: Vec<Option<MemBinding>> = args
                    .iter()
                    .map(|&arg_id| {
                        if let Expr::Local(local_id) = body.get_expr(arg_id) {
                            body.get_local(*local_id).mem
                        } else {
                            None
                        }
                    })
                    .collect();

                // Check if this is a call to a user-defined function
                if let Some(poly_def) = self.poly_functions.get(func).cloned() {
                    // Check if this is a trivial wrapper (body just forwards to another call)
                    // If so, inline it by calling the inner function directly
                    if let Some((inner_func, _inner_args)) = get_trivial_wrapper_target(&poly_def) {
                        // Look up the inner function
                        if let Some(inner_def) = self.poly_functions.get(&inner_func).cloned() {
                            let arg_types: Vec<_> =
                                args.iter().map(|a| body.get_type(*a).clone()).collect();
                            let subst = self.infer_substitution(&inner_def, &arg_types)?;
                            let spec_key = SpecKey::new(&subst, mem_bindings.clone());

                            if spec_key.needs_specialization() {
                                // Specialize the inner function directly
                                let specialized_name =
                                    self.get_or_create_specialization(&inner_func, &spec_key, &inner_def)?;
                                return Ok(Expr::Call {
                                    func: specialized_name,
                                    args: new_args,
                                });
                            } else {
                                // Monomorphic call to inner function
                                self.ensure_in_worklist(&inner_func, inner_def);
                                return Ok(Expr::Call {
                                    func: inner_func,
                                    args: new_args,
                                });
                            }
                        }
                        // Inner function is a builtin/intrinsic - call it directly
                        return Ok(Expr::Call {
                            func: inner_func,
                            args: new_args,
                        });
                    }

                    // Get argument types to infer substitution
                    let arg_types: Vec<_> = args.iter().map(|a| body.get_type(*a).clone()).collect();
                    let subst = self.infer_substitution(&poly_def, &arg_types)?;
                    let spec_key = SpecKey::new(&subst, mem_bindings);

                    if spec_key.needs_specialization() {
                        // This call needs specialization (type vars and/or mem bindings)
                        let specialized_name =
                            self.get_or_create_specialization(func, &spec_key, &poly_def)?;
                        return Ok(Expr::Call {
                            func: specialized_name,
                            args: new_args,
                        });
                    } else {
                        // Monomorphic call with no mem bindings - ensure the callee is in the worklist
                        self.ensure_in_worklist(func, poly_def);
                    }
                }

                Ok(Expr::Call {
                    func: func.clone(),
                    args: new_args,
                })
            }

            // Map subexpressions for compound expressions
            Expr::BinOp { op, lhs, rhs } => Ok(Expr::BinOp {
                op: op.clone(),
                lhs: expr_map[lhs],
                rhs: expr_map[rhs],
            }),
            Expr::UnaryOp { op, operand } => Ok(Expr::UnaryOp {
                op: op.clone(),
                operand: expr_map[operand],
            }),
            Expr::If { cond, then_, else_ } => Ok(Expr::If {
                cond: expr_map[cond],
                then_: expr_map[then_],
                else_: expr_map[else_],
            }),
            Expr::Let {
                local,
                rhs,
                body: let_body,
            } => Ok(Expr::Let {
                local: *local,
                rhs: expr_map[rhs],
                body: expr_map[let_body],
            }),
            Expr::Loop {
                loop_var,
                init,
                init_bindings,
                kind,
                body: loop_body,
            } => {
                let new_init_bindings: Vec<_> =
                    init_bindings.iter().map(|(local, e)| (*local, expr_map[e])).collect();
                Ok(Expr::Loop {
                    loop_var: *loop_var,
                    init: expr_map[init],
                    init_bindings: new_init_bindings,
                    kind: map_loop_kind(kind, expr_map),
                    body: expr_map[loop_body],
                })
            }
            Expr::Intrinsic { name, args } => {
                let new_args: Vec<_> = args.iter().map(|a| expr_map[a]).collect();
                Ok(Expr::Intrinsic {
                    name: name.clone(),
                    args: new_args,
                })
            }
            Expr::Attributed {
                attributes,
                expr: inner,
            } => Ok(Expr::Attributed {
                attributes: attributes.clone(),
                expr: expr_map[inner],
            }),
            Expr::Materialize(inner) => Ok(Expr::Materialize(expr_map[inner])),
            Expr::Tuple(elems) => {
                let new_elems: Vec<_> = elems.iter().map(|e| expr_map[e]).collect();
                Ok(Expr::Tuple(new_elems))
            }
            Expr::Array(elems) => {
                let new_elems: Vec<_> = elems.iter().map(|e| expr_map[e]).collect();
                Ok(Expr::Array(new_elems))
            }
            Expr::Vector(elems) => {
                let new_elems: Vec<_> = elems.iter().map(|e| expr_map[e]).collect();
                Ok(Expr::Vector(new_elems))
            }
            Expr::Matrix(rows) => {
                let new_rows: Vec<Vec<_>> =
                    rows.iter().map(|row| row.iter().map(|e| expr_map[e]).collect()).collect();
                Ok(Expr::Matrix(new_rows))
            }
            Expr::Closure {
                lambda_name,
                captures,
            } => {
                let new_captures = expr_map[captures];

                // Try to specialize the lambda based on the closure's concrete type
                if let Some(def) = self.poly_functions.get(lambda_name).cloned() {
                    // expr_type is the function type of this closure (Arrow type)
                    // Extract concrete param and return types from it
                    let (concrete_params, concrete_ret) = split_function_type(expr_type);

                    // Build substitution by unifying lambda params AND return type
                    // against concrete types. This is necessary because lambdas may have
                    // different variable IDs for params vs return type (e.g., zip3's lambda
                    // has ((A,B),C) -> (A,B,C) where param uses different vars than return)
                    let subst = self.infer_lambda_substitution(&def, &concrete_params, &concrete_ret)?;

                    // Lambdas don't receive storage-backed arrays directly, so no mem bindings
                    let lambda_mem_bindings = vec![None; concrete_params.len()];
                    let spec_key = SpecKey::new(&subst, lambda_mem_bindings);

                    if !subst.is_empty() {
                        // Specialize the lambda with concrete types
                        let specialized_name =
                            self.get_or_create_specialization(lambda_name, &spec_key, &def)?;
                        return Ok(Expr::Closure {
                            lambda_name: specialized_name,
                            captures: new_captures,
                        });
                    } else {
                        // Check if lambda body has unresolved variables even with empty subst
                        // This can happen for prelude lambdas whose param types are concrete
                        // but whose internal expressions have stale type variables
                        if let Def::Function { body, .. } = &def {
                            if body_has_unresolved_variables(body) {
                                // Force specialization even with empty subst to trigger
                                // the variable resolution in specialize_def
                                let specialized_name =
                                    self.get_or_create_specialization(lambda_name, &spec_key, &def)?;
                                return Ok(Expr::Closure {
                                    lambda_name: specialized_name,
                                    captures: new_captures,
                                });
                            }
                        }
                        self.ensure_in_worklist(lambda_name, def);
                    }
                }

                Ok(Expr::Closure {
                    lambda_name: lambda_name.clone(),
                    captures: new_captures,
                })
            }
            Expr::Range {
                start,
                step,
                end,
                kind,
            } => Ok(Expr::Range {
                start: expr_map[start],
                step: step.map(|s| expr_map[&s]),
                end: expr_map[end],
                kind: *kind,
            }),
            Expr::Global(name) => {
                // Global reference might refer to a top-level constant
                if let Some(def) = self.poly_functions.get(name).cloned() {
                    self.ensure_in_worklist(name, def);
                }
                Ok(Expr::Global(name.clone()))
            }

            // Leaf nodes - just clone
            Expr::Local(id) => Ok(Expr::Local(*id)),
            Expr::Int(s) => Ok(Expr::Int(s.clone())),
            Expr::Float(s) => Ok(Expr::Float(s.clone())),
            Expr::Bool(b) => Ok(Expr::Bool(*b)),
            Expr::String(s) => Ok(Expr::String(s.clone())),
            Expr::Unit => Ok(Expr::Unit),

            // Slices - map subexpressions
            Expr::OwnedSlice { data, len } => Ok(Expr::OwnedSlice {
                data: expr_map[data],
                len: expr_map[len],
            }),
            Expr::BorrowedSlice { base, offset, len } => Ok(Expr::BorrowedSlice {
                base: expr_map[base],
                offset: expr_map[offset],
                len: expr_map[len],
            }),
        }
    }

    /// Infer the substitution needed for a polymorphic function call.
    /// Uses scheme's canonical variable IDs for consistency with specialize_def.
    fn infer_substitution(&self, poly_def: &Def, arg_types: &[Type<TypeName>]) -> Result<Substitution> {
        let mut subst = Substitution::new();

        let _func_name = match poly_def {
            Def::Function { name, .. } => name.as_str(),
            _ => "<unknown>",
        };

        match poly_def {
            Def::Function {
                scheme: Some(scheme), ..
            } => {
                // Use canonical scheme variable IDs (consistent with specialize_def)
                let func_type = unwrap_scheme(scheme);
                let (param_types, _ret_type) = split_function_type(func_type);

                // Unify ONLY against scheme params (not body params)
                for (param_ty, arg_ty) in param_types.iter().zip(arg_types.iter()) {
                    self.unify_for_subst(param_ty, arg_ty, &mut subst)?;
                }
            }
            Def::Function {
                scheme: None,
                params,
                body,
                ..
            } => {
                // Fallback for functions without schemes (user functions, lambdas)
                let body_param_types: Vec<_> = params.iter().map(|&p| &body.get_local(p).ty).collect();
                for (param_ty, arg_ty) in body_param_types.iter().zip(arg_types.iter()) {
                    self.unify_for_subst(param_ty, arg_ty, &mut subst)?;
                }
            }
            Def::EntryPoint { inputs, .. } => {
                let param_types: Vec<_> = inputs.iter().map(|i| &i.ty).collect();
                for (param_ty, arg_ty) in param_types.iter().zip(arg_types.iter()) {
                    self.unify_for_subst(param_ty, arg_ty, &mut subst)?;
                }
            }
            Def::Constant { .. } | Def::Uniform { .. } | Def::Storage { .. } => {
                // No parameters
            }
        }

        Ok(subst)
    }

    /// Infer substitution for a lambda using both param types and return type from the closure's concrete type
    fn infer_lambda_substitution(
        &self,
        lambda_def: &Def,
        concrete_params: &[Type<TypeName>],
        concrete_ret: &Type<TypeName>,
    ) -> Result<Substitution> {
        let mut subst = Substitution::new();

        if let Def::Function { params, body, .. } = lambda_def {
            // Unify lambda's body param types against concrete param types
            let body_param_types: Vec<_> = params.iter().map(|&p| &body.get_local(p).ty).collect();
            for (param_ty, arg_ty) in body_param_types.iter().zip(concrete_params.iter()) {
                self.unify_for_subst(param_ty, arg_ty, &mut subst)?;
            }

            // Also unify lambda's return type against concrete return type
            let body_ret = body.get_type(body.root);
            self.unify_for_subst(body_ret, concrete_ret, &mut subst)?;
        }

        Ok(subst)
    }

    /// Unify two types to build a substitution
    fn unify_for_subst(
        &self,
        expected: &Type<TypeName>,
        actual: &Type<TypeName>,
        subst: &mut Substitution,
    ) -> Result<()> {
        match (expected, actual) {
            (Type::Variable(id), concrete) => {
                // This is a type variable in the polymorphic function
                // Map it to the concrete type from the call site
                if !contains_variables(concrete) {
                    subst.insert(*id, concrete.clone());
                }
            }
            (Type::Constructed(name1, args1), Type::Constructed(name2, args2)) => {
                // Recurse into constructed types
                if std::mem::discriminant(name1) == std::mem::discriminant(name2) {
                    for (a1, a2) in args1.iter().zip(args2.iter()) {
                        self.unify_for_subst(a1, a2, subst)?;
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Get or create a specialized version of a function
    fn get_or_create_specialization(
        &mut self,
        func_name: &str,
        spec_key: &SpecKey,
        poly_def: &Def,
    ) -> Result<String> {
        let cache_key = (func_name.to_string(), spec_key.clone());

        if let Some(specialized_name) = self.specializations.get(&cache_key) {
            return Ok(specialized_name.clone());
        }

        // Build substitution from type_subst
        let subst = spec_key.type_subst.to_subst();

        // Create new specialized name including both type and mem info
        let type_suffix = format_subst(&subst);
        let mem_suffix = format_mem_bindings(&spec_key.mem_bindings);
        let specialized_name = if type_suffix.is_empty() && mem_suffix.is_empty() {
            func_name.to_string()
        } else if type_suffix.is_empty() {
            format!("{}${}", func_name, mem_suffix)
        } else if mem_suffix.is_empty() {
            format!("{}${}", func_name, type_suffix)
        } else {
            format!("{}${}${}", func_name, type_suffix, mem_suffix)
        };

        // Clone and specialize the function with both type subst and mem bindings
        let specialized_def = specialize_def(
            poly_def.clone(),
            &subst,
            &spec_key.mem_bindings,
            &specialized_name,
        )?;

        // Add to worklist to process its body
        self.worklist.push_back(WorkItem {
            name: specialized_name.clone(),
            def: specialized_def,
        });

        self.specializations.insert(cache_key, specialized_name.clone());
        Ok(specialized_name)
    }
}

fn map_loop_kind(kind: &crate::mir::LoopKind, expr_map: &HashMap<ExprId, ExprId>) -> crate::mir::LoopKind {
    use crate::mir::LoopKind;
    match kind {
        LoopKind::For { var, iter } => LoopKind::For {
            var: *var,
            iter: expr_map[iter],
        },
        LoopKind::ForRange { var, bound } => LoopKind::ForRange {
            var: *var,
            bound: expr_map[bound],
        },
        LoopKind::While { cond } => LoopKind::While { cond: expr_map[cond] },
    }
}

/// Check if a type contains type variables
fn contains_variables(ty: &Type<TypeName>) -> bool {
    match ty {
        Type::Variable(_) => true,
        Type::Constructed(name, args) => {
            // Check for unresolved variables in TypeName itself
            match name {
                TypeName::SizeVar(_) | TypeName::UserVar(_) => return true,
                TypeName::Record(_fields) => {
                    // Check types inside record fields (stored in args)
                    if args.iter().any(contains_variables) {
                        return true;
                    }
                }
                TypeName::Sum(variants) => {
                    // Check types inside sum variants
                    if variants.iter().any(|(_, types)| types.iter().any(contains_variables)) {
                        return true;
                    }
                }
                TypeName::Existential(_) => {
                    // Inner type is now in args[0], checked below
                }
                _ => {}
            }
            // Check type arguments
            args.iter().any(contains_variables)
        }
    }
}

/// Check if a function body contains any unresolved type variables
fn body_has_unresolved_variables(body: &Body) -> bool {
    // Check all expression types
    for idx in 0..body.exprs.len() {
        let ty = body.get_type(ExprId(idx as u32));
        if contains_variables(ty) {
            return true;
        }
    }
    // Check all local declarations
    for local in &body.locals {
        if contains_variables(&local.ty) {
            return true;
        }
    }
    false
}

/// Check if a function is a trivial wrapper (body just forwards all params to another call).
/// Returns the target function name if so.
fn get_trivial_wrapper_target(def: &Def) -> Option<(String, Vec<LocalId>)> {
    if let Def::Function { params, body, .. } = def {
        // The body's root must be a Call expression
        let root_expr = body.get_expr(body.root);
        if let Expr::Call { func, args } = root_expr {
            // Check if all args are just Local references to params (in order)
            if args.len() != params.len() {
                return None;
            }
            for (i, &arg_id) in args.iter().enumerate() {
                let arg_expr = body.get_expr(arg_id);
                if let Expr::Local(local_id) = arg_expr {
                    if *local_id != params[i] {
                        return None;
                    }
                } else {
                    return None;
                }
            }
            // This is a trivial wrapper - it just forwards params to another function
            return Some((func.clone(), params.clone()));
        }
    }
    None
}

/// Collect variable mappings by unifying body type with concrete type.
/// When body has a variable and concrete has a non-variable type,
/// add the mapping body_var → concrete.
fn collect_body_var_mappings(
    body_ty: &Type<TypeName>,
    concrete_ty: &Type<TypeName>,
    extended: &mut Substitution,
) {
    match (body_ty, concrete_ty) {
        (Type::Variable(body_id), concrete) => {
            // Body has a var, concrete is the resolved type - map directly
            if !contains_variables(concrete) {
                extended.insert(*body_id, concrete.clone());
            }
        }
        (Type::Constructed(_, body_args), Type::Constructed(_, concrete_args)) => {
            // Recurse into type arguments
            for (b, c) in body_args.iter().zip(concrete_args.iter()) {
                collect_body_var_mappings(b, c, extended);
            }
        }
        _ => {}
    }
}

/// Create a specialized version of a function by applying substitution and mem bindings
fn specialize_def(
    def: Def,
    subst: &Substitution,
    mem_bindings: &[Option<MemBinding>],
    new_name: &str,
) -> Result<Def> {
    use crate::mir::{EntryInput, EntryOutput};

    match def {
        Def::Function {
            id,
            params,
            ret_type,
            scheme,
            attributes,
            body,
            span,
            ..
        } => {
            // Build extended substitution that includes body variable IDs
            // Body expression types come from type_table (different var IDs than scheme)
            // We need to map body vars to concrete types by comparing with scheme-derived types
            let full_subst = if let Some(ref scheme) = scheme {
                let mut extended = subst.clone();
                let func_type = unwrap_scheme(scheme);
                let (scheme_params, scheme_ret) = split_function_type(func_type);

                // Apply substitution to scheme types to get concrete types
                let concrete_params: Vec<_> = scheme_params.iter().map(|t| apply_subst(t, subst)).collect();
                let concrete_ret = apply_subst(&scheme_ret, subst);

                // Collect variable mappings from params and all body expressions
                //
                // The key insight: param LOCAL DECLARATIONS use scheme variable IDs
                // (from flattening with prelude_schemes), while body expressions may use
                // different variable IDs from type_table. We collect from:
                // 1. Local declarations (canonical scheme vars)
                // 2. Local reference expressions (may have same or different vars)
                // 3. All other expressions (to catch nested lambdas etc.)

                for (i, &param_id) in params.iter().enumerate() {
                    if let Some(concrete_ty) = concrete_params.get(i) {
                        // Collect from local declaration (canonical type with scheme vars)
                        if let Some(local) = body.locals.get(param_id.0 as usize) {
                            collect_body_var_mappings(&local.ty, concrete_ty, &mut extended);
                        }

                        // Also collect from expressions that reference this local
                        for (expr_idx, expr) in body.exprs.iter().enumerate() {
                            if matches!(expr, Expr::Local(id) if *id == param_id) {
                                let expr_ty = body.get_type(ExprId(expr_idx as u32));
                                collect_body_var_mappings(expr_ty, concrete_ty, &mut extended);
                            }
                        }
                    }
                }

                // Also map body return type vars
                let body_ret = body.get_type(body.root);
                collect_body_var_mappings(body_ret, &concrete_ret, &mut extended);

                // Now walk ALL expressions to propagate mappings transitively.
                // This catches nested lambda types that use the same variables as params.
                // We iterate until no new mappings are found (fixed point).
                let mut changed = true;
                while changed {
                    changed = false;
                    for (expr_idx, _) in body.exprs.iter().enumerate() {
                        let expr_ty = body.get_type(ExprId(expr_idx as u32));
                        // Try to resolve this expression's type using current mappings
                        let resolved = apply_subst_partial(expr_ty, &extended);
                        // If we got a fully concrete type, map any remaining vars
                        if !contains_variables(&resolved) && contains_variables(expr_ty) {
                            let before = extended.len();
                            collect_body_var_mappings(expr_ty, &resolved, &mut extended);
                            if extended.len() > before {
                                changed = true;
                            }
                        }
                    }
                }

                extended
            } else {
                // For functions without schemes (lambdas), build substitution from params
                let mut extended = subst.clone();
                // Collect variable mappings from param local declarations
                for &param_id in params.iter() {
                    if let Some(local) = body.locals.get(param_id.0 as usize) {
                        // Try to resolve the param type using current substitution
                        let resolved = apply_subst_partial(&local.ty, &extended);
                        if !contains_variables(&resolved) && contains_variables(&local.ty) {
                            collect_body_var_mappings(&local.ty, &resolved, &mut extended);
                        }
                    }
                }

                // Collect from return type
                let body_ret = body.get_type(body.root);
                let resolved_ret = apply_subst_partial(body_ret, &extended);
                if !contains_variables(&resolved_ret) && contains_variables(body_ret) {
                    collect_body_var_mappings(body_ret, &resolved_ret, &mut extended);
                }

                // Propagate mappings through all expressions
                let mut changed = true;
                while changed {
                    changed = false;
                    for (expr_idx, _) in body.exprs.iter().enumerate() {
                        let expr_ty = body.get_type(ExprId(expr_idx as u32));
                        let resolved = apply_subst_partial(expr_ty, &extended);
                        if !contains_variables(&resolved) && contains_variables(expr_ty) {
                            let before = extended.len();
                            collect_body_var_mappings(expr_ty, &resolved, &mut extended);
                            if extended.len() > before {
                                changed = true;
                            }
                        }
                    }
                }

                extended
            };

            // If the function has a scheme, use it for consistent return type
            let ret_type = if let Some(ref scheme) = scheme {
                // Unwrap the scheme to get the inner function type
                // Use the same bound variable IDs as infer_substitution
                let func_type = unwrap_scheme(scheme);
                let (_, scheme_ret_type) = split_function_type(func_type);
                apply_subst(&scheme_ret_type, subst)
            } else {
                // Fallback: use the original ret_type
                let ret_context = format!("{}::ret_type", new_name);
                apply_subst_with_context(&ret_type, &full_subst, &ret_context)
            };

            // Apply the extended substitution to the body
            let mut body = apply_subst_body_with_context(body, &full_subst, new_name);

            // Apply mem bindings to parameters
            for (i, &param_id) in params.iter().enumerate() {
                if let Some(mem) = mem_bindings.get(i).copied().flatten() {
                    body.get_local_mut(param_id).mem = Some(mem);
                }
            }

            Ok(Def::Function {
                id,
                name: new_name.to_string(),
                params,
                ret_type,
                scheme: None, // Specialized functions are monomorphic
                attributes,
                body,
                span,
            })
        }
        Def::EntryPoint {
            id,
            execution_model,
            inputs,
            outputs,
            body,
            span,
            ..
        } => {
            let inputs = inputs
                .into_iter()
                .map(|i| {
                    let ctx = format!("{}::input({})", new_name, i.name);
                    EntryInput {
                        local: i.local,
                        name: i.name,
                        ty: apply_subst_with_context(&i.ty, subst, &ctx),
                        decoration: i.decoration,
                    }
                })
                .collect();
            let outputs = outputs
                .into_iter()
                .enumerate()
                .map(|(idx, o)| {
                    let ctx = format!("{}::output({})", new_name, idx);
                    EntryOutput {
                        ty: apply_subst_with_context(&o.ty, subst, &ctx),
                        decoration: o.decoration,
                    }
                })
                .collect();
            let body = apply_subst_body_with_context(body, subst, new_name);

            Ok(Def::EntryPoint {
                id,
                name: new_name.to_string(),
                execution_model,
                inputs,
                outputs,
                body,
                span,
            })
        }
        Def::Constant {
            id,
            ty,
            attributes,
            body,
            span,
            ..
        } => Ok(Def::Constant {
            id,
            name: new_name.to_string(),
            ty: apply_subst(&ty, subst),
            attributes,
            body: apply_subst_body(body, subst),
            span,
        }),
        Def::Uniform {
            id, ty, set, binding, ..
        } => Ok(Def::Uniform {
            id,
            name: new_name.to_string(),
            ty: apply_subst(&ty, subst),
            set,
            binding,
        }),
        Def::Storage {
            id,
            ty,
            set,
            binding,
            layout,
            access,
            ..
        } => Ok(Def::Storage {
            id,
            name: new_name.to_string(),
            ty: apply_subst(&ty, subst),
            set,
            binding,
            layout,
            access,
        }),
    }
}

/// Apply a substitution to a type
fn apply_subst(ty: &Type<TypeName>, subst: &Substitution) -> Type<TypeName> {
    apply_subst_with_context(ty, subst, "unknown")
}

/// Apply a substitution to a type, keeping unresolved variables as-is.
/// This is used for collecting variable mappings where we may not have all mappings yet.
fn apply_subst_partial(ty: &Type<TypeName>, subst: &Substitution) -> Type<TypeName> {
    match ty {
        Type::Variable(id) => subst.get(id).cloned().unwrap_or_else(|| ty.clone()),
        Type::Constructed(name, args) => {
            let new_args = args.iter().map(|arg| apply_subst_partial(arg, subst)).collect();
            Type::Constructed(name.clone(), new_args)
        }
    }
}

/// Apply a substitution to a type with context for debugging
fn apply_subst_with_context(ty: &Type<TypeName>, subst: &Substitution, context: &str) -> Type<TypeName> {
    match ty {
        Type::Variable(id) => subst.get(id).cloned().unwrap_or_else(|| {
            eprintln!(
                "DEBUG: Unresolved type variable Variable({}) in context: {}\n\
                 Substitution contains {} mappings: {:?}",
                id, context, subst.len(), subst
            );
            panic!(
                "BUG: Unresolved type variable Variable({}) during monomorphization. \
                 The substitution is incomplete - this variable should have been resolved during type checking \
                 or added to the substitution during monomorphization.\n\
                 Context: {}\n\
                 Substitution contains: {:?}",
                id, context, subst
            )
        }),
        Type::Constructed(name, args) => {
            // Recursively apply substitution to type arguments
            let new_args = args.iter().map(|arg| apply_subst_with_context(arg, subst, context)).collect();

            // Also apply substitution to types nested inside TypeName
            let new_name = match name {
                TypeName::Record(fields) => TypeName::Record(fields.clone()),
                TypeName::Sum(variants) => {
                    let new_variants = variants
                        .iter()
                        .map(|(name, types)| {
                            (
                                name.clone(),
                                types.iter().map(|t| apply_subst_with_context(t, subst, context)).collect(),
                            )
                        })
                        .collect();
                    TypeName::Sum(new_variants)
                }
                TypeName::Existential(vars) => {
                    // Inner type is now in args, will be substituted via new_args
                    TypeName::Existential(vars.clone())
                }
                _ => name.clone(),
            };

            Type::Constructed(new_name, new_args)
        }
    }
}

/// Apply a substitution to a body
fn apply_subst_body(old_body: Body, subst: &Substitution) -> Body {
    apply_subst_body_with_context(old_body, subst, "body")
}

/// Apply a substitution to a body with context for debugging
fn apply_subst_body_with_context(old_body: Body, subst: &Substitution, func_name: &str) -> Body {
    let mut new_body = Body::new();

    // Apply substitution to locals
    for local in &old_body.locals {
        let context = format!("{}::local({})", func_name, local.name);
        new_body.alloc_local(LocalDecl {
            name: local.name.clone(),
            span: local.span,
            ty: apply_subst_with_context(&local.ty, subst, &context),
            kind: local.kind,
            mem: local.mem,
        });
    }

    // Copy expressions with substituted types
    for (idx, expr) in old_body.exprs.iter().enumerate() {
        let old_id = ExprId(idx as u32);
        let context = format!("{}::expr({}, {:?})", func_name, idx, expr);
        let ty = apply_subst_with_context(old_body.get_type(old_id), subst, &context);
        let span = old_body.get_span(old_id);
        let node_id = old_body.get_node_id(old_id);
        new_body.alloc_expr(expr.clone(), ty, span, node_id);
    }

    new_body.root = old_body.root;
    new_body
}

/// Format a substitution for use in specialized function names
fn format_subst(subst: &Substitution) -> String {
    let mut items: Vec<_> = subst.iter().collect();
    items.sort_by_key(|(k, _)| *k);

    items.iter().map(|(_, ty)| format_type_compact(ty)).collect::<Vec<_>>().join("_")
}

fn format_type_compact(ty: &Type<TypeName>) -> String {
    match ty {
        Type::Variable(id) => format!("v{}", id),
        Type::Constructed(TypeName::Size(n), _) => format!("n{}", n),
        Type::Constructed(TypeName::Str(s), args) if args.is_empty() => s.to_string(),
        Type::Constructed(TypeName::Array, args) if args.len() == 2 => {
            format!(
                "arr{}{}",
                format_type_compact(&args[0]),
                format_type_compact(&args[1])
            )
        }
        _ => "ty".to_string(),
    }
}

/// Format mem bindings for use in specialized function names
fn format_mem_bindings(bindings: &[Option<MemBinding>]) -> String {
    // If all bindings are None, no suffix needed
    if bindings.iter().all(|m| m.is_none()) {
        return String::new();
    }

    let parts: Vec<_> = bindings
        .iter()
        .map(|m| match m {
            Some(MemBinding::Storage { set, binding }) => format!("s{}b{}", set, binding),
            None => "_".to_string(),
        })
        .collect();

    format!("mem_{}", parts.join("_"))
}
