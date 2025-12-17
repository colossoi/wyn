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

use crate::ast::TypeName;
use crate::error::Result;
use crate::mir::{Body, Def, Expr, ExprId, LocalDecl, Program};
use crate::IdArena;
use crate::mir::{LambdaId, LambdaInfo};
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
    /// Map from (function_name, substitution) to specialized name
    specializations: HashMap<(String, SubstKey), String>,
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
                    TypeName::Existential(vars, inner) => {
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
                        return TypeKey::Constructed("unique".to_string(), args.iter().map(TypeKey::from_type).collect());
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
}

impl Monomorphizer {
    fn new(program: Program) -> Self {
        // Build map of polymorphic functions
        let mut poly_functions = HashMap::new();
        let mut entry_points = Vec::new();

        for def in program.defs {
            let name = match &def {
                Def::Function { name, .. } => name.clone(),
                Def::Constant { name, .. } => name.clone(),
                Def::Uniform { name, .. } => name.clone(),
                Def::Storage { name, .. } => name.clone(),
                Def::EntryPoint { name, .. } => name.clone(),
            };

            // Check if this is an entry point
            if let Def::EntryPoint { .. } = &def {
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

            let new_expr = self.process_expr(&old_body, old_expr, &expr_map)?;
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
    ) -> Result<Expr> {
        match expr {
            Expr::Call { func, args } => {
                // Map arguments
                let new_args: Vec<_> = args.iter().map(|a| expr_map[a]).collect();

                // Check if this is a call to a user-defined function
                if let Some(poly_def) = self.poly_functions.get(func).cloned() {
                    // Get argument types to infer substitution
                    let arg_types: Vec<_> = args.iter().map(|a| body.get_type(*a).clone()).collect();
                    let subst = self.infer_substitution(&poly_def, &arg_types)?;

                    if !subst.is_empty() {
                        // This is a polymorphic call - specialize it
                        let specialized_name =
                            self.get_or_create_specialization(func, &subst, &poly_def)?;
                        return Ok(Expr::Call {
                            func: specialized_name,
                            args: new_args,
                        });
                    } else {
                        // Monomorphic call - ensure the callee is in the worklist
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
            Expr::Let { local, rhs, body: let_body } => Ok(Expr::Let {
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
                let new_init_bindings: Vec<_> = init_bindings
                    .iter()
                    .map(|(local, e)| (*local, expr_map[e]))
                    .collect();
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
            Expr::Attributed { attributes, expr: inner } => Ok(Expr::Attributed {
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
                let new_rows: Vec<Vec<_>> = rows
                    .iter()
                    .map(|row| row.iter().map(|e| expr_map[e]).collect())
                    .collect();
                Ok(Expr::Matrix(new_rows))
            }
            Expr::Closure { lambda_name, captures } => {
                // Ensure lambda function is in worklist
                if let Some(def) = self.poly_functions.get(lambda_name).cloned() {
                    self.ensure_in_worklist(lambda_name, def);
                }
                let new_captures: Vec<_> = captures.iter().map(|c| expr_map[c]).collect();
                Ok(Expr::Closure {
                    lambda_name: lambda_name.clone(),
                    captures: new_captures,
                })
            }
            Expr::Range { start, step, end, kind } => Ok(Expr::Range {
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
        }
    }

    /// Infer the substitution needed for a polymorphic function call
    fn infer_substitution(&self, poly_def: &Def, arg_types: &[Type<TypeName>]) -> Result<Substitution> {
        let mut subst = Substitution::new();

        // Get parameter types from the definition
        let param_types: Vec<_> = match poly_def {
            Def::Function { params, body, .. } => params.iter().map(|&p| &body.get_local(p).ty).collect(),
            Def::EntryPoint { inputs, .. } => inputs.iter().map(|i| &i.ty).collect(),
            Def::Constant { .. } => return Ok(subst), // No parameters
            Def::Uniform { .. } => return Ok(subst),  // No parameters
            Def::Storage { .. } => return Ok(subst),  // No parameters
        };

        // Match argument types against parameter types
        for (param_ty, arg_ty) in param_types.iter().zip(arg_types.iter()) {
            self.unify_for_subst(param_ty, arg_ty, &mut subst)?;
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
        subst: &Substitution,
        poly_def: &Def,
    ) -> Result<String> {
        let key = (func_name.to_string(), SubstKey::from_subst(subst));

        if let Some(specialized_name) = self.specializations.get(&key) {
            return Ok(specialized_name.clone());
        }

        // Create new specialized name
        let specialized_name = format!("{}${}", func_name, format_subst(subst));

        // Clone and specialize the function
        let specialized_def = specialize_def(poly_def.clone(), subst, &specialized_name)?;

        // Add to worklist to process its body
        self.worklist.push_back(WorkItem {
            name: specialized_name.clone(),
            def: specialized_def,
        });

        self.specializations.insert(key, specialized_name.clone());
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
        LoopKind::While { cond } => LoopKind::While {
            cond: expr_map[cond],
        },
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
                TypeName::Existential(_, inner) => {
                    // Check the inner type
                    if contains_variables(inner) {
                        return true;
                    }
                }
                _ => {}
            }
            // Check type arguments
            args.iter().any(contains_variables)
        }
    }
}

/// Create a specialized version of a function by applying substitution
fn specialize_def(def: Def, subst: &Substitution, new_name: &str) -> Result<Def> {
    use crate::mir::{EntryInput, EntryOutput};

    match def {
        Def::Function {
            id,
            params,
            ret_type,
            attributes,
            body,
            span,
            ..
        } => {
            // params is Vec<LocalId> - indices into body.locals
            // apply_subst_body handles substituting types in the locals
            let ret_type = apply_subst(&ret_type, subst);
            let body = apply_subst_body(body, subst);

            Ok(Def::Function {
                id,
                name: new_name.to_string(),
                params,
                ret_type,
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
                .map(|i| EntryInput {
                    local: i.local,
                    name: i.name,
                    ty: apply_subst(&i.ty, subst),
                    decoration: i.decoration,
                })
                .collect();
            let outputs = outputs
                .into_iter()
                .map(|o| EntryOutput {
                    ty: apply_subst(&o.ty, subst),
                    decoration: o.decoration,
                })
                .collect();
            let body = apply_subst_body(body, subst);

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
    match ty {
        Type::Variable(id) => subst.get(id).cloned().unwrap_or_else(|| {
            panic!(
                "BUG: Unresolved type variable Variable({}) during monomorphization. \
                 The substitution is incomplete - this variable should have been resolved during type checking \
                 or added to the substitution during monomorphization.\n\
                 Substitution contains: {:?}",
                id, subst
            )
        }),
        Type::Constructed(name, args) => {
            // Recursively apply substitution to type arguments
            let new_args = args.iter().map(|arg| apply_subst(arg, subst)).collect();

            // Also apply substitution to types nested inside TypeName
            let new_name = match name {
                TypeName::Record(fields) => TypeName::Record(fields.clone()),
                TypeName::Sum(variants) => {
                    let new_variants = variants
                        .iter()
                        .map(|(name, types)| {
                            (
                                name.clone(),
                                types.iter().map(|t| apply_subst(t, subst)).collect(),
                            )
                        })
                        .collect();
                    TypeName::Sum(new_variants)
                }
                TypeName::Existential(vars, inner) => {
                    TypeName::Existential(vars.clone(), Box::new(apply_subst(inner, subst)))
                }
                _ => name.clone(),
            };

            Type::Constructed(new_name, new_args)
        }
    }
}

/// Apply a substitution to a body
fn apply_subst_body(old_body: Body, subst: &Substitution) -> Body {
    let mut new_body = Body::new();

    // Apply substitution to locals
    for local in &old_body.locals {
        new_body.alloc_local(LocalDecl {
            name: local.name.clone(),
            span: local.span,
            ty: apply_subst(&local.ty, subst),
            kind: local.kind,
        });
    }

    // Copy expressions with substituted types
    for (idx, expr) in old_body.exprs.iter().enumerate() {
        let old_id = ExprId(idx as u32);
        let ty = apply_subst(old_body.get_type(old_id), subst);
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

    items
        .iter()
        .map(|(_, ty)| format_type_compact(ty))
        .collect::<Vec<_>>()
        .join("_")
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
