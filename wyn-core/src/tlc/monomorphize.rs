//! Monomorphization pass for TLC.
//!
//! This pass takes polymorphic functions (with size/type variables) and creates
//! specialized monomorphic copies for each concrete instantiation that's actually called.
//!
//! This happens at the TLC level, before MIR lowering, so that all functions are
//! monomorphic by the time we reach codegen.
//!
//! Example:
//!   def sum [n] (arr:[n]f32) : f32 = ...
//!
//! When called with [4]f32, creates:
//!   def sum$n4 (arr:[4]f32) : f32 = ...

use super::{Def, DefMeta, LoopKind, Program, Term, TermIdSource, TermKind};
use crate::ast::TypeName;
use crate::types::TypeScheme;
use crate::{SymbolId, SymbolTable};
use polytype::Type;
use std::collections::{HashMap, HashSet, VecDeque};

/// A substitution mapping type variables to concrete types
pub(crate) type Substitution = HashMap<usize, Type<TypeName>>;

/// Monomorphize a TLC program.
///
/// This walks through all definitions starting from entry points, finds calls
/// to polymorphic functions, and creates specialized versions with concrete types.
pub fn monomorphize(program: Program, schemes: &HashMap<String, TypeScheme>) -> Program {
    let mono = Monomorphizer::new(program, schemes);
    mono.run()
}

pub(crate) struct Monomorphizer<'a> {
    /// Symbol table for name lookup and allocation
    symbols: SymbolTable,
    /// Original polymorphic functions by symbol
    poly_functions: HashMap<SymbolId, Def>,
    /// Generated monomorphic functions
    mono_functions: Vec<Def>,
    /// Map from (function_sym, spec_key) to specialized symbol
    specializations: HashMap<(SymbolId, SpecKey), SymbolId>,
    /// Worklist of functions to process
    worklist: VecDeque<WorkItem>,
    /// Processed (original_sym, spec_key) pairs
    processed: HashSet<(SymbolId, SpecKey)>,
    /// Type schemes for polymorphic functions (keyed by string name)
    schemes: &'a HashMap<String, TypeScheme>,
    /// Term ID source for creating new terms
    term_ids: TermIdSource,
    /// Uniform declarations (passed through unchanged)
    uniforms: Vec<crate::ast::UniformDecl>,
    /// Storage declarations (passed through unchanged)
    storage: Vec<crate::ast::StorageDecl>,
}

struct WorkItem {
    /// Original function symbol (before specialization)
    original_sym: SymbolId,
    /// Specialization key (empty for monomorphic functions)
    spec_key: SpecKey,
    /// The function definition
    def: Def,
}

/// A key for looking up specializations
/// We use a sorted Vec instead of HashMap for deterministic ordering
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct SubstKey(Vec<(usize, TypeKey)>);

/// Combined specialization key: type substitution.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct SpecKey {
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
    pub(crate) fn empty() -> Self {
        SpecKey {
            type_subst: SubstKey(Vec::new()),
        }
    }

    pub(crate) fn new(subst: &Substitution) -> Self {
        SpecKey {
            type_subst: SubstKey::from_subst(subst),
        }
    }

    /// Returns true if this represents a non-trivial specialization
    pub(crate) fn needs_specialization(&self) -> bool {
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
                    TypeName::SizePlaceholder => {
                        panic!("SizePlaceholder should be resolved before monomorphization")
                    }
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
                    TypeName::ArrayVariantComposite => "array_composite".to_string(),
                    TypeName::ArrayVariantView => "array_view".to_string(),
                    TypeName::AddressPlaceholder => {
                        panic!("AddressPlaceholder should be resolved before monomorphization")
                    }
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
                    "arrow" => TypeName::Arrow,
                    "unit" => TypeName::Unit,
                    "ptr" => TypeName::Pointer,
                    "unique" => TypeName::Unique,
                    "array_view" => TypeName::ArrayVariantView,
                    "array_composite" => TypeName::ArrayVariantComposite,
                    s if s.starts_with("tuple") => {
                        let n: usize = s[5..]
                            .parse()
                            .unwrap_or_else(|_| panic!("BUG: invalid tuple arity in mangled name: {}", s));
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

impl<'a> Monomorphizer<'a> {
    fn new(program: Program, schemes: &'a HashMap<String, TypeScheme>) -> Self {
        // Build function map and collect entry points
        let mut poly_functions = HashMap::new();
        let mut entry_points = Vec::new();

        for def in program.defs.iter() {
            let sym = def.name;

            // For entry points, add to worklist
            if matches!(&def.meta, DefMeta::EntryPoint(_)) {
                entry_points.push(WorkItem {
                    original_sym: sym,
                    spec_key: SpecKey::empty(),
                    def: def.clone(),
                });
            }

            poly_functions.insert(sym, def.clone());
        }

        let mut worklist = VecDeque::new();
        worklist.extend(entry_points);

        Monomorphizer {
            symbols: program.symbols,
            poly_functions,
            mono_functions: Vec::new(),
            specializations: HashMap::new(),
            worklist,
            processed: HashSet::new(),
            schemes,
            term_ids: TermIdSource::new(),
            uniforms: program.uniforms,
            storage: program.storage,
        }
    }

    fn run(mut self) -> Program {
        while let Some(work_item) = self.worklist.pop_front() {
            let key = (work_item.original_sym, work_item.spec_key.clone());
            if self.processed.contains(&key) {
                continue;
            }
            self.processed.insert(key);

            // Process this function: look for calls and specialize callees
            let def = self.process_def(work_item.def);
            self.mono_functions.push(def);
        }

        Program {
            defs: self.mono_functions,
            uniforms: self.uniforms,
            storage: self.storage,
            symbols: self.symbols,
        }
    }

    /// Ensure a definition is in the worklist (for monomorphic callees and constants)
    fn ensure_in_worklist(&mut self, sym: SymbolId, def: Def) {
        let key = (sym, SpecKey::empty());
        if !self.processed.contains(&key) {
            // Check if it's already in the worklist
            let already_queued =
                self.worklist.iter().any(|w| w.original_sym == sym && !w.spec_key.needs_specialization());
            if !already_queued {
                self.worklist.push_back(WorkItem {
                    original_sym: sym,
                    spec_key: SpecKey::empty(),
                    def,
                });
            }
        }
    }

    fn process_def(&mut self, def: Def) -> Def {
        let new_body = self.process_term(&def.body);
        Def {
            body: new_body,
            ..def
        }
    }

    /// Process a term, rewriting calls to polymorphic functions.
    fn process_term(&mut self, term: &Term) -> Term {
        let kind = match &term.kind {
            TermKind::App { func, arg } => {
                // First recursively process the function and argument
                let processed_arg = self.process_term(arg);

                // Collect application spine to detect function calls
                let (base, args) = Self::collect_application_spine(func, arg);

                // Check if base is a variable referencing a known function
                if let TermKind::Var(sym) = &base.kind {
                    let sym = *sym;
                    if let Some(poly_def) = self.poly_functions.get(&sym).cloned() {
                        // Infer substitution from argument types
                        let arg_types: Vec<_> = args.iter().map(|a| a.ty.clone()).collect();
                        let subst = self.infer_substitution(&poly_def, &arg_types);
                        let spec_key = SpecKey::new(&subst);

                        if spec_key.needs_specialization() {
                            // Get or create specialized version
                            let specialized_sym =
                                self.get_or_create_specialization(sym, &spec_key, &poly_def);
                            // Rebuild the application with the specialized function symbol
                            let new_func = self.rewrite_var_sym(func, sym, specialized_sym);
                            return Term {
                                id: self.term_ids.next_id(),
                                ty: term.ty.clone(),
                                span: term.span,
                                kind: TermKind::App {
                                    func: Box::new(self.process_term(&new_func)),
                                    arg: Box::new(processed_arg),
                                },
                            };
                        } else {
                            // Monomorphic call - ensure callee is in worklist
                            self.ensure_in_worklist(sym, poly_def);
                        }
                    }
                }

                // Default: just process recursively
                let processed_func = self.process_term(func);
                TermKind::App {
                    func: Box::new(processed_func),
                    arg: Box::new(processed_arg),
                }
            }

            TermKind::Var(sym) => {
                let sym = *sym;
                // Check if this is a reference to a polymorphic function
                // This handles cases like `let f = some_poly_fn in ...`
                if let Some(poly_def) = self.poly_functions.get(&sym).cloned() {
                    // Try to infer specialization from the term's type
                    if let Some(subst) = self.infer_var_substitution(&poly_def, &term.ty) {
                        let spec_key = SpecKey::new(&subst);
                        if spec_key.needs_specialization() {
                            let specialized_sym =
                                self.get_or_create_specialization(sym, &spec_key, &poly_def);
                            return Term {
                                id: self.term_ids.next_id(),
                                ty: term.ty.clone(),
                                span: term.span,
                                kind: TermKind::Var(specialized_sym),
                            };
                        } else {
                            self.ensure_in_worklist(sym, poly_def);
                        }
                    } else {
                        self.ensure_in_worklist(sym, poly_def);
                    }
                }
                TermKind::Var(sym)
            }

            TermKind::Lam {
                param,
                param_ty,
                body,
            } => TermKind::Lam {
                param: param.clone(),
                param_ty: param_ty.clone(),
                body: Box::new(self.process_term(body)),
            },

            TermKind::Let {
                name,
                name_ty,
                rhs,
                body,
            } => TermKind::Let {
                name: name.clone(),
                name_ty: name_ty.clone(),
                rhs: Box::new(self.process_term(rhs)),
                body: Box::new(self.process_term(body)),
            },

            TermKind::If {
                cond,
                then_branch,
                else_branch,
            } => TermKind::If {
                cond: Box::new(self.process_term(cond)),
                then_branch: Box::new(self.process_term(then_branch)),
                else_branch: Box::new(self.process_term(else_branch)),
            },

            TermKind::Loop {
                loop_var,
                loop_var_ty,
                init,
                init_bindings,
                kind,
                body,
            } => {
                let new_init_bindings = init_bindings
                    .iter()
                    .map(|(name, ty, expr)| (name.clone(), ty.clone(), self.process_term(expr)))
                    .collect();
                let new_kind = match kind {
                    LoopKind::For { var, var_ty, iter } => LoopKind::For {
                        var: var.clone(),
                        var_ty: var_ty.clone(),
                        iter: Box::new(self.process_term(iter)),
                    },
                    LoopKind::ForRange { var, var_ty, bound } => LoopKind::ForRange {
                        var: var.clone(),
                        var_ty: var_ty.clone(),
                        bound: Box::new(self.process_term(bound)),
                    },
                    LoopKind::While { cond } => LoopKind::While {
                        cond: Box::new(self.process_term(cond)),
                    },
                };
                TermKind::Loop {
                    loop_var: loop_var.clone(),
                    loop_var_ty: loop_var_ty.clone(),
                    init: Box::new(self.process_term(init)),
                    init_bindings: new_init_bindings,
                    kind: new_kind,
                    body: Box::new(self.process_term(body)),
                }
            }

            // Leaves unchanged
            k @ (TermKind::IntLit(_)
            | TermKind::FloatLit(_)
            | TermKind::BoolLit(_)
            | TermKind::StringLit(_)
            | TermKind::BinOp(_)
            | TermKind::UnOp(_)
            | TermKind::Extern(_)) => k.clone(),
        };

        Term {
            id: self.term_ids.next_id(),
            ty: term.ty.clone(),
            span: term.span,
            kind,
        }
    }

    /// Collect the spine of a nested application chain.
    /// Given `App(App(App(f, a), b), c)`, returns `(f, [a, b, c])`.
    pub(crate) fn collect_application_spine<'t>(
        func: &'t Term,
        arg: &'t Term,
    ) -> (&'t Term, Vec<&'t Term>) {
        let mut args = vec![arg];
        let mut current = func;

        loop {
            match &current.kind {
                TermKind::App {
                    func: inner_func,
                    arg: inner_arg,
                } => {
                    args.push(inner_arg.as_ref());
                    current = inner_func.as_ref();
                }
                _ => {
                    args.reverse();
                    return (current, args);
                }
            }
        }
    }

    /// Rewrite Var nodes that match old_sym to use new_sym.
    /// This traverses nested App nodes to find the function symbol.
    fn rewrite_var_sym(&mut self, term: &Term, old_sym: SymbolId, new_sym: SymbolId) -> Term {
        match &term.kind {
            TermKind::Var(sym) if *sym == old_sym => Term {
                id: self.term_ids.next_id(),
                ty: term.ty.clone(),
                span: term.span,
                kind: TermKind::Var(new_sym),
            },
            TermKind::App { func, arg } => Term {
                id: self.term_ids.next_id(),
                ty: term.ty.clone(),
                span: term.span,
                kind: TermKind::App {
                    func: Box::new(self.rewrite_var_sym(func, old_sym, new_sym)),
                    arg: arg.clone(),
                },
            },
            _ => term.clone(),
        }
    }

    /// Infer the substitution needed for a polymorphic function call.
    fn infer_substitution(&self, poly_def: &Def, arg_types: &[Type<TypeName>]) -> Substitution {
        let mut subst = Substitution::new();

        // Get the type scheme for this function (look up by string name)
        let name_str = self.symbols.get(poly_def.name).expect("BUG: def symbol not in table");
        if let Some(scheme) = self.schemes.get(name_str) {
            let func_type = unwrap_scheme(scheme);
            let (param_types, _ret_type) = split_function_type(func_type);

            // Unify parameter types with argument types
            for (param_ty, arg_ty) in param_types.iter().zip(arg_types.iter()) {
                self.unify_for_subst(param_ty, arg_ty, &mut subst);
            }
        } else {
            // No scheme - try using the def's function type directly
            let (param_types, _ret_type) = split_function_type(&poly_def.ty);
            for (param_ty, arg_ty) in param_types.iter().zip(arg_types.iter()) {
                self.unify_for_subst(param_ty, arg_ty, &mut subst);
            }
        }

        subst
    }

    /// Infer substitution for a variable reference based on its concrete type.
    fn infer_var_substitution(
        &self,
        poly_def: &Def,
        concrete_type: &Type<TypeName>,
    ) -> Option<Substitution> {
        let mut subst = Substitution::new();

        // Get the polymorphic type from scheme or def (look up by string name)
        let name_str = self.symbols.get(poly_def.name).expect("BUG: def symbol not in table");
        let poly_type = if let Some(scheme) = self.schemes.get(name_str) {
            unwrap_scheme(scheme).clone()
        } else {
            poly_def.ty.clone()
        };

        // Unify the polymorphic type with the concrete type
        self.unify_for_subst(&poly_type, concrete_type, &mut subst);
        if !subst.is_empty() {
            return Some(subst);
        }

        None
    }

    /// Unify two types to build a substitution.
    fn unify_for_subst(
        &self,
        expected: &Type<TypeName>,
        actual: &Type<TypeName>,
        subst: &mut Substitution,
    ) {
        match (expected, actual) {
            (Type::Variable(id), concrete) => {
                // Bind the scheme variable to the concrete type
                subst.insert(*id, concrete.clone());
            }
            (Type::Constructed(name1, args1), Type::Constructed(name2, args2)) => {
                // Recurse into constructed types only if type constructors match exactly
                if name1 != name2 {
                    return; // Type mismatch - just no binding
                }
                if args1.len() != args2.len() {
                    return; // Arity mismatch
                }
                for (a1, a2) in args1.iter().zip(args2.iter()) {
                    self.unify_for_subst(a1, a2, subst);
                }
            }
            _ => {} // Other cases - no binding
        }
    }

    /// Get or create a specialized version of a function
    fn get_or_create_specialization(
        &mut self,
        func_sym: SymbolId,
        spec_key: &SpecKey,
        poly_def: &Def,
    ) -> SymbolId {
        let cache_key = (func_sym, spec_key.clone());

        if let Some(specialized_sym) = self.specializations.get(&cache_key) {
            return *specialized_sym;
        }

        // Build substitution from type_subst
        let subst = spec_key.type_subst.to_subst();

        // Create new specialized name from type substitution
        let func_name = self.symbols.get(func_sym).expect("BUG: func symbol not in table");
        let type_suffix = format_subst(&subst);
        let specialized_name = if type_suffix.is_empty() {
            func_name.to_string()
        } else {
            format!("{}${}", func_name, type_suffix)
        };
        let specialized_sym = self.symbols.alloc(specialized_name);

        // Clone and specialize the function
        let specialized_def = self.specialize_def(poly_def.clone(), &subst, specialized_sym);

        // Add to worklist to process its body
        self.worklist.push_back(WorkItem {
            original_sym: func_sym,
            spec_key: spec_key.clone(),
            def: specialized_def,
        });

        self.specializations.insert(cache_key, specialized_sym);
        specialized_sym
    }

    /// Create a specialized version of a function by applying substitution.
    fn specialize_def(&mut self, def: Def, subst: &Substitution, new_sym: SymbolId) -> Def {
        Def {
            name: new_sym,
            ty: apply_subst(&def.ty, subst),
            body: self.apply_subst_term(&def.body, subst),
            meta: def.meta,
            arity: def.arity,
        }
    }

    /// Apply a substitution to a term, recursively updating all types.
    fn apply_subst_term(&mut self, term: &Term, subst: &Substitution) -> Term {
        let new_ty = apply_subst(&term.ty, subst);
        let new_kind = match &term.kind {
            TermKind::Var(sym) => TermKind::Var(*sym),
            TermKind::IntLit(s) => TermKind::IntLit(s.clone()),
            TermKind::FloatLit(f) => TermKind::FloatLit(*f),
            TermKind::BoolLit(b) => TermKind::BoolLit(*b),
            TermKind::StringLit(s) => TermKind::StringLit(s.clone()),
            TermKind::BinOp(op) => TermKind::BinOp(op.clone()),
            TermKind::UnOp(op) => TermKind::UnOp(op.clone()),
            TermKind::Extern(s) => TermKind::Extern(s.clone()),

            TermKind::Lam {
                param,
                param_ty,
                body,
            } => TermKind::Lam {
                param: param.clone(),
                param_ty: apply_subst(param_ty, subst),
                body: Box::new(self.apply_subst_term(body, subst)),
            },

            TermKind::App { func, arg } => TermKind::App {
                func: Box::new(self.apply_subst_term(func, subst)),
                arg: Box::new(self.apply_subst_term(arg, subst)),
            },

            TermKind::Let {
                name,
                name_ty,
                rhs,
                body,
            } => TermKind::Let {
                name: name.clone(),
                name_ty: apply_subst(name_ty, subst),
                rhs: Box::new(self.apply_subst_term(rhs, subst)),
                body: Box::new(self.apply_subst_term(body, subst)),
            },

            TermKind::If {
                cond,
                then_branch,
                else_branch,
            } => TermKind::If {
                cond: Box::new(self.apply_subst_term(cond, subst)),
                then_branch: Box::new(self.apply_subst_term(then_branch, subst)),
                else_branch: Box::new(self.apply_subst_term(else_branch, subst)),
            },

            TermKind::Loop {
                loop_var,
                loop_var_ty,
                init,
                init_bindings,
                kind,
                body,
            } => {
                let new_init_bindings = init_bindings
                    .iter()
                    .map(|(name, ty, expr)| {
                        (
                            name.clone(),
                            apply_subst(ty, subst),
                            self.apply_subst_term(expr, subst),
                        )
                    })
                    .collect();
                let new_kind = match kind {
                    LoopKind::For { var, var_ty, iter } => LoopKind::For {
                        var: var.clone(),
                        var_ty: apply_subst(var_ty, subst),
                        iter: Box::new(self.apply_subst_term(iter, subst)),
                    },
                    LoopKind::ForRange { var, var_ty, bound } => LoopKind::ForRange {
                        var: var.clone(),
                        var_ty: apply_subst(var_ty, subst),
                        bound: Box::new(self.apply_subst_term(bound, subst)),
                    },
                    LoopKind::While { cond } => LoopKind::While {
                        cond: Box::new(self.apply_subst_term(cond, subst)),
                    },
                };
                TermKind::Loop {
                    loop_var: loop_var.clone(),
                    loop_var_ty: apply_subst(loop_var_ty, subst),
                    init: Box::new(self.apply_subst_term(init, subst)),
                    init_bindings: new_init_bindings,
                    kind: new_kind,
                    body: Box::new(self.apply_subst_term(body, subst)),
                }
            }
        };

        Term {
            id: self.term_ids.next_id(),
            ty: new_ty,
            span: term.span,
            kind: new_kind,
        }
    }
}

/// Apply a substitution to a type
fn apply_subst(ty: &Type<TypeName>, subst: &Substitution) -> Type<TypeName> {
    match ty {
        Type::Variable(id) => subst.get(id).cloned().unwrap_or_else(|| ty.clone()),
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
                TypeName::Existential(vars) => TypeName::Existential(vars.clone()),
                _ => name.clone(),
            };

            Type::Constructed(new_name, new_args)
        }
    }
}

/// Format a substitution for use in specialized function names
fn format_subst(subst: &Substitution) -> String {
    let mut items: Vec<_> = subst.iter().collect();
    items.sort_by_key(|(k, _)| *k);

    items.iter().map(|(_, ty)| format_type_compact(ty)).collect::<Vec<_>>().join("_")
}

pub(crate) fn format_type_compact(ty: &Type<TypeName>) -> String {
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
        Type::Constructed(TypeName::ArrayVariantView, _) => "array_view".to_string(),
        Type::Constructed(TypeName::ArrayVariantComposite, _) => "array_composite".to_string(),
        Type::Constructed(TypeName::ArrayVariantVirtual, _) => todo!(),
        Type::Constructed(name, args) => {
            // Fallback for other constructed types
            let args_str = args.iter().map(format_type_compact).collect::<Vec<_>>().join("_");
            if args_str.is_empty() { format!("{:?}", name) } else { format!("{:?}_{}", name, args_str) }
        }
    }
}
