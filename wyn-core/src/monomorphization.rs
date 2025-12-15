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
use crate::ast::{Type, TypeName};
use crate::error::Result;
use crate::mir::folder::MirFolder;
use crate::mir::{Def, Expr, ExprKind, LambdaId, LambdaInfo, Param, Program};
use crate::types::TypeExt;
use polytype::Type as PolyType;
use std::collections::{HashMap, HashSet, VecDeque};

/// A substitution mapping type variables to concrete types
type Substitution = HashMap<usize, Type>;

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
    fn from_type(ty: &Type) -> Self {
        // Handle unique types first via dedicated API
        if let Some(inner) = ty.as_unique_inner() {
            return TypeKey::Constructed("unique".to_string(), vec![TypeKey::from_type(inner)]);
        }
        match ty {
            PolyType::Variable(id) => TypeKey::Var(*id),
            PolyType::Constructed(name, args) => {
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
                    TypeName::Unique => unreachable!("Handled above via as_unique_inner()"),
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
                let body = self.process_expr(body)?;
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
                let body = self.process_expr(body)?;
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
                let body = self.process_expr(body)?;
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
            Def::Uniform { .. } => {
                // Uniforms have no body to process
                Ok(def)
            }
            Def::Storage { .. } => {
                // Storage buffers have no body to process
                Ok(def)
            }
        }
    }

    fn process_expr(&mut self, expr: Expr) -> Result<Expr> {
        let kind = match expr.kind {
            ExprKind::Call { func, args } => {
                // Process arguments first
                let args: Result<Vec<_>> = args.into_iter().map(|arg| self.process_expr(arg)).collect();
                let args = args?;

                // Check if this is a call to a polymorphic function
                if let Some(poly_def) = self.poly_functions.get(&func).cloned() {
                    let subst = self.infer_substitution(&poly_def, &args)?;

                    if !subst.is_empty() {
                        // This is a polymorphic call - specialize it
                        let specialized_name =
                            self.get_or_create_specialization(&func, &subst, &poly_def)?;
                        ExprKind::Call {
                            func: specialized_name,
                            args,
                        }
                    } else {
                        // Monomorphic call - ensure the callee is in the worklist
                        self.ensure_in_worklist(&func, poly_def);
                        ExprKind::Call { func, args }
                    }
                } else {
                    ExprKind::Call { func, args }
                }
            }
            ExprKind::BinOp { op, lhs, rhs } => ExprKind::BinOp {
                op,
                lhs: Box::new(self.process_expr(*lhs)?),
                rhs: Box::new(self.process_expr(*rhs)?),
            },
            ExprKind::UnaryOp { op, operand } => ExprKind::UnaryOp {
                op,
                operand: Box::new(self.process_expr(*operand)?),
            },
            ExprKind::If {
                cond,
                then_branch,
                else_branch,
            } => ExprKind::If {
                cond: Box::new(self.process_expr(*cond)?),
                then_branch: Box::new(self.process_expr(*then_branch)?),
                else_branch: Box::new(self.process_expr(*else_branch)?),
            },
            ExprKind::Let {
                name,
                binding_id,
                value,
                body,
            } => ExprKind::Let {
                name,
                binding_id,
                value: Box::new(self.process_expr(*value)?),
                body: Box::new(self.process_expr(*body)?),
            },
            ExprKind::Loop {
                loop_var,
                init,
                init_bindings,
                kind,
                body,
            } => {
                let init_bindings: Result<Vec<_>> = init_bindings
                    .into_iter()
                    .map(|(name, expr)| Ok((name, self.process_expr(expr)?)))
                    .collect();
                ExprKind::Loop {
                    loop_var,
                    init: Box::new(self.process_expr(*init)?),
                    init_bindings: init_bindings?,
                    kind,
                    body: Box::new(self.process_expr(*body)?),
                }
            }
            ExprKind::Intrinsic { name, args } => {
                let args: Result<Vec<_>> = args.into_iter().map(|arg| self.process_expr(arg)).collect();
                ExprKind::Intrinsic { name, args: args? }
            }
            ExprKind::Attributed { attributes, expr } => ExprKind::Attributed {
                attributes,
                expr: Box::new(self.process_expr(*expr)?),
            },
            ExprKind::Materialize(inner) => ExprKind::Materialize(Box::new(self.process_expr(*inner)?)),
            // Leaf nodes - no recursion needed
            ExprKind::Var(ref name) => {
                // Variable reference might refer to a top-level constant
                // NOTE: This can't distinguish local variables from global constants,
                // so it may unnecessarily queue constants when a local variable shadows
                // a constant name. This is safe but suboptimal - it won't change the
                // semantics, just might do extra work.
                // A proper fix would require tracking local scope or using resolved
                // symbol IDs instead of strings.
                if let Some(def) = self.poly_functions.get(name).cloned() {
                    self.ensure_in_worklist(name, def);
                }
                expr.kind
            }
            ExprKind::Literal(ref lit) => {
                // Process tuple elements recursively
                if let crate::mir::Literal::Tuple(elems) = lit {
                    for elem in elems {
                        let _ = self.process_expr(elem.clone())?;
                    }
                }
                expr.kind
            }
            ExprKind::Closure {
                ref lambda_name,
                ref captures,
            } => {
                // Ensure lambda function is in worklist
                if let Some(def) = self.poly_functions.get(lambda_name).cloned() {
                    self.ensure_in_worklist(lambda_name, def);
                }
                // Process captures recursively
                for cap in captures {
                    let _ = self.process_expr(cap.clone())?;
                }
                expr.kind
            }
            ExprKind::Range {
                start,
                step,
                end,
                kind,
            } => ExprKind::Range {
                start: Box::new(self.process_expr(*start)?),
                step: step.map(|s| self.process_expr(*s)).transpose()?.map(Box::new),
                end: Box::new(self.process_expr(*end)?),
                kind,
            },
            ExprKind::Unit => ExprKind::Unit,
        };

        Ok(Expr {
            id: expr.id,
            ty: expr.ty,
            kind,
            span: expr.span,
        })
    }

    /// Infer the substitution needed for a polymorphic function call
    fn infer_substitution(&self, poly_def: &Def, args: &[Expr]) -> Result<Substitution> {
        let mut subst = Substitution::new();

        // Get parameter types from the definition
        let param_types = match poly_def {
            Def::Function { params, .. } => params.iter().map(|p| &p.ty).collect::<Vec<_>>(),
            Def::EntryPoint { inputs, .. } => inputs.iter().map(|i| &i.ty).collect::<Vec<_>>(),
            Def::Constant { .. } => return Ok(subst), // No parameters
            Def::Uniform { .. } => return Ok(subst),  // No parameters
            Def::Storage { .. } => return Ok(subst),  // No parameters
        };

        // Match argument types against parameter types
        for (param_ty, arg) in param_types.iter().zip(args.iter()) {
            self.unify_for_subst(param_ty, &arg.ty, &mut subst)?;
        }

        Ok(subst)
    }

    /// Unify two types to build a substitution
    fn unify_for_subst(&self, expected: &Type, actual: &Type, subst: &mut Substitution) -> Result<()> {
        match (expected, actual) {
            (PolyType::Variable(id), concrete) => {
                // This is a type variable in the polymorphic function
                // Map it to the concrete type from the call site
                if !contains_variables(concrete) {
                    subst.insert(*id, concrete.clone());
                }
            }
            (PolyType::Constructed(name1, args1), PolyType::Constructed(name2, args2)) => {
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
        let specialized_def = self.specialize_def(poly_def.clone(), subst, &specialized_name)?;

        // Add to worklist to process its body
        self.worklist.push_back(WorkItem {
            name: specialized_name.clone(),
            def: specialized_def,
        });

        self.specializations.insert(key, specialized_name.clone());
        Ok(specialized_name)
    }

    /// Create a specialized version of a function by applying substitution
    fn specialize_def(&self, def: Def, subst: &Substitution, new_name: &str) -> Result<Def> {
        use crate::mir::EntryInput;
        use crate::mir::EntryOutput;

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
                let params = params
                    .into_iter()
                    .map(|p| Param {
                        name: p.name,
                        ty: apply_subst(&p.ty, subst),
                    })
                    .collect();
                let ret_type = apply_subst(&ret_type, subst);
                let body = apply_subst_expr(body, subst);

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
                let body = apply_subst_expr(body, subst);

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
                body: apply_subst_expr(body, subst),
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
}

/// Check if a type contains type variables
fn contains_variables(ty: &Type) -> bool {
    match ty {
        PolyType::Variable(_) => true,
        PolyType::Constructed(name, args) => {
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

/// Visitor that applies type substitutions to expressions
struct SubstitutionVisitor<'a> {
    subst: &'a Substitution,
}

impl<'a> MirFolder for SubstitutionVisitor<'a> {
    type Error = std::convert::Infallible;
    type Ctx = ();

    fn visit_type(&mut self, ty: Type, _ctx: &mut Self::Ctx) -> std::result::Result<Type, Self::Error> {
        Ok(apply_subst(&ty, self.subst))
    }
}

/// Apply a substitution to a type
fn apply_subst(ty: &Type, subst: &Substitution) -> Type {
    match ty {
        PolyType::Variable(id) => subst.get(id).cloned().unwrap_or_else(|| {
            panic!(
                "BUG: Unresolved type variable Variable({}) during monomorphization. \
                 The substitution is incomplete - this variable should have been resolved during type checking \
                 or added to the substitution during monomorphization.\n\
                 Substitution contains: {:?}",
                id, subst
            )
        }),
        PolyType::Constructed(name, args) => {
            // Recursively apply substitution to type arguments
            let new_args = args.iter().map(|arg| apply_subst(arg, subst)).collect();

            // Also apply substitution to types nested inside TypeName
            let new_name = match name {
                TypeName::Record(fields) => {
                    // Field names don't contain types, so no substitution needed
                    // Types are in args which are already substituted above
                    TypeName::Record(fields.clone())
                }
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

            PolyType::Constructed(new_name, new_args)
        }
    }
}

/// Apply a substitution to an expression
fn apply_subst_expr(expr: Expr, subst: &Substitution) -> Expr {
    let mut visitor = SubstitutionVisitor { subst };
    visitor.visit_expr(expr, &mut ()).unwrap()
}

/// Format a substitution for use in specialized function names
fn format_subst(subst: &Substitution) -> String {
    let mut items: Vec<_> = subst.iter().collect();
    items.sort_by_key(|(k, _)| *k);

    items.iter().map(|(_, ty)| format_type_compact(ty)).collect::<Vec<_>>().join("_")
}

fn format_type_compact(ty: &Type) -> String {
    match ty {
        PolyType::Variable(id) => format!("v{}", id),
        PolyType::Constructed(TypeName::Size(n), _) => format!("n{}", n),
        PolyType::Constructed(TypeName::Str(s), args) if args.is_empty() => s.to_string(),
        PolyType::Constructed(TypeName::Array, args) if args.len() == 2 => {
            format!(
                "arr{}{}",
                format_type_compact(&args[0]),
                format_type_compact(&args[1])
            )
        }
        _ => "ty".to_string(),
    }
}
