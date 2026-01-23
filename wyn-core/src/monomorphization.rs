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
use crate::err_type;
use crate::error::Result;
use crate::mir::{Body, Def, Expr, ExprId, LocalDecl, Program};
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
    /// Processed (original_name, spec_key) pairs
    /// Using the conceptual key rather than serialized name avoids dedup bugs
    /// if name formatting accidentally collides for distinct specializations.
    processed: HashSet<(String, SpecKey)>,
    /// Lambda registry from original program
    lambda_registry: IdArena<LambdaId, LambdaInfo>,
}

struct WorkItem {
    /// Original function name (before specialization)
    original_name: String,
    /// Specialization key (empty for monomorphic functions)
    spec_key: SpecKey,
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
    /// Type substitution for specialization
    type_subst: SubstKey,
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
    /// Create an empty spec key (for monomorphic functions)
    fn empty() -> Self {
        SpecKey {
            type_subst: SubstKey(Vec::new()),
        }
    }

    fn new(subst: &Substitution) -> Self {
        SpecKey {
            type_subst: SubstKey::from_subst(subst),
        }
    }

    /// Returns true if this represents a non-trivial specialization
    fn needs_specialization(&self) -> bool {
        !self.type_subst.is_empty()
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
                    TypeName::SizePlaceholder => "size_placeholder".to_string(),
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
                    TypeName::AddressFunction => "function".to_string(),
                    TypeName::AddressStorage => "storage".to_string(),
                    TypeName::AddressPlaceholder => "address_placeholder".to_string(),
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
                    "size_placeholder" => TypeName::SizePlaceholder,
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

/// Build a curried function type from parameter types and return type.
/// The inverse of split_function_type: ([A, B], C) becomes (A -> B -> C)
fn build_function_type(param_types: &[Type<TypeName>], ret_type: &Type<TypeName>) -> Type<TypeName> {
    param_types.iter().rev().fold(ret_type.clone(), |acc, param_ty| {
        Type::Constructed(TypeName::Arrow, vec![param_ty.clone(), acc])
    })
}

impl Monomorphizer {
    fn new(program: Program) -> Self {
        // TODO(Phase 6): Storage buffer declarations are now tracked via types.
        // The old mem binding approach is being phased out.

        // Build function map and collect entry points
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

            // For entry points, add to worklist
            // TODO(Phase 6): Address space is now tracked in types (Slice[elem, Storage/Function])
            // rather than via mem bindings on LocalDecl
            if let Def::EntryPoint { .. } = &def {
                entry_points.push(WorkItem {
                    original_name: name.clone(),
                    spec_key: SpecKey::empty(),
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
            let key = (work_item.original_name.clone(), work_item.spec_key.clone());
            if self.processed.contains(&key) {
                continue;
            }
            self.processed.insert(key);

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
        let key = (name.to_string(), SpecKey::empty());
        if !self.processed.contains(&key) {
            // Check if it's already in the worklist
            let already_queued =
                self.worklist.iter().any(|w| w.original_name == name && !w.spec_key.needs_specialization());
            if !already_queued {
                self.worklist.push_back(WorkItem {
                    original_name: name.to_string(),
                    spec_key: SpecKey::empty(),
                    def,
                });
            }
        }
    }

    fn process_function(&mut self, mut def: Def) -> Result<Def> {
        match &mut def {
            Def::Function { body, .. } | Def::Constant { body, .. } | Def::EntryPoint { body, .. } => {
                self.process_body(body)?;
            }
            Def::Uniform { .. } | Def::Storage { .. } => {}
        }
        Ok(def)
    }

    /// Process a body in-place, rewriting Call/Global names for specialization.
    /// Only modifies func names - ExprIds and structure remain unchanged.
    fn process_body(&mut self, body: &mut Body) -> Result<()> {
        // We need to iterate by index because we modify exprs and call methods on self
        for idx in 0..body.exprs.len() {
            let expr_id = ExprId(idx as u32);

            match &body.exprs[idx] {
                Expr::Call { func, args } => {
                    // Check if this is a call to a user-defined function that needs specialization
                    if let Some(poly_def) = self.poly_functions.get(func).cloned() {
                        let arg_types: Vec<_> = args.iter().map(|a| body.get_type(*a).clone()).collect();
                        let subst = self.infer_substitution(&poly_def, &arg_types)?;
                        let spec_key = SpecKey::new(&subst);

                        if spec_key.needs_specialization() {
                            let specialized_name =
                                self.get_or_create_specialization(&func.clone(), &spec_key, &poly_def)?;
                            // Update func name in place
                            if let Expr::Call { func, .. } = &mut body.exprs[idx] {
                                *func = specialized_name;
                            }
                        } else {
                            self.ensure_in_worklist(&func.clone(), poly_def);
                        }
                    }
                }

                Expr::Global(name) => {
                    // Global reference might refer to a top-level function/constant
                    if let Some(poly_def) = self.poly_functions.get(name).cloned() {
                        let expr_type = body.get_type(expr_id).clone();
                        if let Some(subst) = self.infer_global_substitution(&poly_def, &expr_type) {
                            let spec_key = SpecKey::new(&subst);
                            if spec_key.needs_specialization() {
                                let specialized_name =
                                    self.get_or_create_specialization(&name.clone(), &spec_key, &poly_def)?;
                                // Update name in place
                                if let Expr::Global(name) = &mut body.exprs[idx] {
                                    *name = specialized_name;
                                }
                            } else {
                                self.ensure_in_worklist(&name.clone(), poly_def);
                            }
                        } else {
                            self.ensure_in_worklist(&name.clone(), poly_def);
                        }
                    }
                }

                // All other expressions don't need modification - ExprIds stay the same
                _ => {}
            }
        }
        Ok(())
    }

    /// Infer the substitution needed for a polymorphic function call.
    /// Uses scheme's canonical variable IDs for consistency with specialize_def.
    fn infer_substitution(&self, poly_def: &Def, arg_types: &[Type<TypeName>]) -> Result<Substitution> {
        let mut subst = Substitution::new();

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

    /// Infer the substitution needed for a global reference based on its concrete type.
    /// This handles lambdas passed to HOFs where the lambda's polymorphic type
    /// must be specialized to match the concrete function type in context.
    fn infer_global_substitution(
        &self,
        poly_def: &Def,
        concrete_type: &Type<TypeName>,
    ) -> Option<Substitution> {
        let mut subst = Substitution::new();

        // Get the polymorphic type from the definition
        let poly_type = match poly_def {
            Def::Function {
                scheme: Some(scheme), ..
            } => unwrap_scheme(scheme).clone(),
            Def::Function {
                scheme: None,
                ret_type,
                params,
                body,
                ..
            } => {
                // Build function type from params and return type
                let param_types: Vec<_> = params.iter().map(|&p| body.get_local(p).ty.clone()).collect();
                build_function_type(&param_types, ret_type)
            }
            Def::Constant { body, .. } => body.get_type(body.root).clone(),
            _ => return None,
        };

        // Unify the polymorphic type with the concrete type
        if self.unify_for_subst(&poly_type, concrete_type, &mut subst).is_ok() {
            if !subst.is_empty() {
                return Some(subst);
            }
        }

        None
    }

    /// Unify two types to build a substitution.
    ///
    /// Requires that `actual` is fully concrete (no type variables).
    /// At monomorphization time, all call sites should have concrete argument types.
    fn unify_for_subst(
        &self,
        expected: &Type<TypeName>,
        actual: &Type<TypeName>,
        subst: &mut Substitution,
    ) -> Result<()> {
        // Invariant: call argument types should be concrete at monomorphization time
        if contains_variables(actual) {
            return Err(err_type!(
                "monomorphization: actual type still contains variables: {}",
                format_type_compact(actual)
            ));
        }

        match (expected, actual) {
            (Type::Variable(id), concrete) => {
                // Bind the scheme variable to the concrete type
                subst.insert(*id, concrete.clone());
                Ok(())
            }
            (Type::Constructed(name1, args1), Type::Constructed(name2, args2)) => {
                // Recurse into constructed types only if type constructors match exactly
                if name1 != name2 {
                    return Err(err_type!(
                        "monomorphization: type constructor mismatch: {:?} vs {:?}",
                        name1,
                        name2
                    ));
                }
                if args1.len() != args2.len() {
                    return Err(err_type!(
                        "monomorphization: arity mismatch for {:?}: {} vs {}",
                        name1,
                        args1.len(),
                        args2.len()
                    ));
                }
                for (a1, a2) in args1.iter().zip(args2.iter()) {
                    self.unify_for_subst(a1, a2, subst)?;
                }
                Ok(())
            }
            _ => Err(err_type!(
                "monomorphization: structural mismatch between {} and {}",
                format_type_compact(expected),
                format_type_compact(actual)
            )),
        }
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

        // Create new specialized name from type substitution
        let type_suffix = format_subst(&subst);
        let specialized_name = if type_suffix.is_empty() {
            func_name.to_string()
        } else {
            format!("{}${}", func_name, type_suffix)
        };

        // Clone and specialize the function
        let specialized_def = specialize_def(poly_def.clone(), &subst, &specialized_name)?;

        // Add to worklist to process its body
        self.worklist.push_back(WorkItem {
            original_name: func_name.to_string(),
            spec_key: spec_key.clone(),
            def: specialized_def,
        });

        self.specializations.insert(cache_key, specialized_name.clone());
        Ok(specialized_name)
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

/// Create a specialized version of a function by applying substitution.
///
/// After canonicalize_vars, body type variables match scheme type variables,
/// so we can directly apply the substitution without complex fixed-point propagation.
fn specialize_def(def: Def, subst: &Substitution, new_name: &str) -> Result<Def> {
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
            // Compute return type from scheme if available
            let ret_type = if let Some(ref scheme) = scheme {
                let func_type = unwrap_scheme(scheme);
                let (_, scheme_ret_type) = split_function_type(func_type);
                apply_subst(&scheme_ret_type, subst)
            } else {
                // Fallback: use the original ret_type
                let ret_context = format!("{}::ret_type", new_name);
                apply_subst_with_context(&ret_type, subst, &ret_context)
            };

            // Apply substitution directly to body
            // (canonicalize_vars ensures body vars match scheme vars)
            let body = apply_subst_body_with_context(body, subst, new_name);

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

/// Apply a substitution to a type with context for debugging
fn apply_subst_with_context(ty: &Type<TypeName>, subst: &Substitution, context: &str) -> Type<TypeName> {
    match ty {
        Type::Variable(id) => subst.get(id).cloned().unwrap_or_else(|| {
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
        Type::Constructed(TypeName::Str(s), args) => {
            let args_str = args.iter().map(format_type_compact).collect::<Vec<_>>().join("_");
            format!("{}_{}", s, args_str)
        }
        Type::Constructed(TypeName::Array, args) => {
            assert!(args.len() == 3);
            format!(
                "arr{}_{}{}",
                format_type_compact(&args[0]),
                format_type_compact(&args[1]),
                format_type_compact(&args[2])
            )
        }
        Type::Constructed(TypeName::Tuple(arity), args) => {
            let args_str = args.iter().map(format_type_compact).collect::<Vec<_>>().join("_");
            format!("tup{}_{}", arity, args_str)
        }
        Type::Constructed(TypeName::Vec, args) => {
            let args_str = args.iter().map(format_type_compact).collect::<Vec<_>>().join("_");
            format!("vec_{}", args_str)
        }
        Type::Constructed(TypeName::Float(bits), _) => format!("f{}", bits),
        Type::Constructed(TypeName::Int(bits), _) => format!("i{}", bits),
        Type::Constructed(TypeName::UInt(bits), _) => format!("u{}", bits),
        Type::Constructed(TypeName::Arrow, args) => {
            let args_str = args.iter().map(format_type_compact).collect::<Vec<_>>().join("_");
            format!("fn_{}", args_str)
        }
        Type::Constructed(TypeName::Unit, _) => "unit".to_string(),
        Type::Constructed(TypeName::Named(name), args) if args.is_empty() => name.clone(),
        Type::Constructed(TypeName::Named(name), args) => {
            let args_str = args.iter().map(format_type_compact).collect::<Vec<_>>().join("_");
            format!("{}_{}", name, args_str)
        }
        Type::Constructed(name, args) => {
            // Fallback for other constructed types
            let args_str = args.iter().map(format_type_compact).collect::<Vec<_>>().join("_");
            if args_str.is_empty() { format!("{:?}", name) } else { format!("{:?}_{}", name, args_str) }
        }
    }
}
