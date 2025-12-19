// Polymorphic builtin registry
// Provides type schemes for truly polymorphic functions (map, magnitude, matrix ops, etc.)
// Implementations come from ImplSource

use crate::ast::{Type, TypeName, TypeScheme};
use crate::type_checker::TypeVarGenerator;
use polytype::Context;
use std::collections::HashMap;

/// Entry for a builtin with its type scheme
#[derive(Debug, Clone)]
pub struct BuiltinEntry {
    /// Type scheme (e.g., "forall a. a -> a")
    pub scheme: TypeScheme,
}

impl BuiltinEntry {
    /// Compute the arity (number of arguments) of this builtin
    pub fn arity(&self) -> usize {
        fn count_arrows(ty: &Type) -> usize {
            match ty {
                Type::Constructed(TypeName::Arrow, args) if args.len() == 2 => 1 + count_arrows(&args[1]),
                _ => 0,
            }
        }

        fn scheme_arity(scheme: &TypeScheme) -> usize {
            match scheme {
                TypeScheme::Monotype(ty) => count_arrows(ty),
                TypeScheme::Polytype { body, .. } => scheme_arity(body),
            }
        }

        scheme_arity(&self.scheme)
    }
}

/// Result of looking up a builtin - either a single entry or an overload set
pub enum BuiltinLookup<'a> {
    /// Single builtin with no overloads
    Single(&'a BuiltinEntry),
    /// Multiple overloads that need resolution based on argument types
    Overloaded(OverloadSet<'a>),
}

/// A set of overloaded builtins that need type-based resolution
pub struct OverloadSet<'a> {
    entries: &'a [BuiltinEntry],
    arity: usize,
}

impl<'a> OverloadSet<'a> {
    /// Create a fresh type for this overload set: ?0 -> ?1 -> ... -> ?n
    /// Since all overloads have the same arity, we can defer resolution to unification time.
    pub fn fresh_type(&self, ctx: &mut impl TypeVarGenerator) -> Type {
        let mut ty = ctx.new_variable();
        for _ in 0..self.arity {
            let param = ctx.new_variable();
            ty = Type::arrow(param, ty);
        }
        ty
    }

    /// Choose the correct overload based on argument types.
    /// Returns the matching entry and the resolved return type.
    /// Uses backtracking: saves context, tries each overload, restores on failure.
    pub fn choose(
        &self,
        arg_types: &[Type],
        ctx: &mut Context<TypeName>,
    ) -> Option<(&'a BuiltinEntry, Type)> {
        for entry in self.entries {
            // Save context for backtracking
            let saved_context = ctx.clone();

            // Instantiate this overload with fresh type variables
            let func_type = entry.scheme.instantiate(ctx);

            // Try to unify parameter types with argument types
            if let Some(return_type) = Self::try_unify(&func_type, arg_types, ctx) {
                return Some((entry, return_type));
            }

            // Restore context and try next overload
            *ctx = saved_context;
        }
        None
    }

    /// Try to unify a function type with the given argument types.
    /// Returns the return type if successful, None otherwise.
    fn try_unify(func_type: &Type, arg_types: &[Type], ctx: &mut Context<TypeName>) -> Option<Type> {
        let mut current_type = func_type.clone();

        for arg_type in arg_types {
            // Decompose the function type: should be param_ty -> rest
            let param_ty = ctx.new_variable();
            let rest_ty = ctx.new_variable();
            let expected_arrow = Type::arrow(param_ty.clone(), rest_ty.clone());

            // Unify current function type with the expected arrow type
            if ctx.unify(&current_type, &expected_arrow).is_err() {
                return None;
            }

            // Unify the parameter type with the argument type
            let param_ty = param_ty.apply(ctx);
            if ctx.unify(&param_ty, arg_type).is_err() {
                return None;
            }

            // Continue with the rest of the function type
            current_type = rest_ty.apply(ctx);
        }

        Some(current_type)
    }

    /// Get the entries (for error messages)
    pub fn entries(&self) -> &[BuiltinEntry] {
        self.entries
    }
}

/// Registry for polymorphic builtin types
pub struct PolyBuiltins {
    /// Maps function name to entry or entries (for overloads)
    builtins: HashMap<String, Vec<BuiltinEntry>>,
}

impl PolyBuiltins {
    pub fn new(ctx: &mut impl TypeVarGenerator) -> Self {
        let mut registry = PolyBuiltins {
            builtins: HashMap::new(),
        };

        registry.register_scalar_math_functions(ctx);
        registry.register_vector_operations(ctx);
        registry.register_matrix_operations(ctx);
        registry.register_higher_order_functions(ctx);

        registry
    }

    /// Add an overload for a builtin.
    /// Asserts that all overloads for the same name have the same arity.
    fn add_overload(&mut self, name: String, entry: BuiltinEntry) {
        let new_arity = entry.arity();
        let entries = self.builtins.entry(name.clone()).or_insert_with(Vec::new);
        if let Some(existing) = entries.first() {
            let existing_arity = existing.arity();
            assert_eq!(
                existing_arity, new_arity,
                "BUG: Overload for '{}' has arity {} but existing overloads have arity {}",
                name, new_arity, existing_arity
            );
        }
        entries.push(entry);
    }

    /// Register a polymorphic builtin function
    fn register_poly(&mut self, name: &str, param_types: Vec<Type>, return_type: Type) {
        let mut func_type = return_type;
        for param_type in param_types.iter().rev() {
            func_type = Type::arrow(param_type.clone(), func_type);
        }

        // Collect all type variables in the function type
        let type_vars = collect_type_vars(&func_type);

        // Wrap in nested Polytype for each type variable (proper quantification)
        let mut scheme = TypeScheme::Monotype(func_type);
        for var_id in type_vars.into_iter().rev() {
            scheme = TypeScheme::Polytype {
                variable: var_id,
                body: Box::new(scheme),
            };
        }

        let entry = BuiltinEntry { scheme };
        self.add_overload(name.to_string(), entry);
    }

    /// Get a builtin by name
    pub fn get(&self, name: &str) -> Option<BuiltinLookup<'_>> {
        self.builtins.get(name).map(|entries| {
            if entries.len() == 1 {
                BuiltinLookup::Single(&entries[0])
            } else {
                let arity = entries[0].arity();
                BuiltinLookup::Overloaded(OverloadSet { entries, arity })
            }
        })
    }

    /// Register scalar math functions (abs, sign, floor, ceil, fract, min, max, clamp, mix, smoothstep)
    fn register_scalar_math_functions(&mut self, ctx: &mut impl TypeVarGenerator) {
        // Unary: abs, sign, floor, ceil, fract
        for name in ["abs", "sign", "floor", "ceil", "fract"] {
            let a = ctx.new_variable();
            self.register_poly(name, vec![a.clone()], a);
        }

        // Binary: min, max
        for name in ["min", "max"] {
            let a = ctx.new_variable();
            self.register_poly(name, vec![a.clone(), a.clone()], a);
        }

        // Ternary: clamp lo hi x, mix a b t, smoothstep edge0 edge1 x
        for name in ["clamp", "mix", "smoothstep"] {
            let a = ctx.new_variable();
            self.register_poly(name, vec![a.clone(), a.clone(), a.clone()], a);
        }
    }

    /// Register vector operations (magnitude, normalize, dot, cross, etc.)
    fn register_vector_operations(&mut self, ctx: &mut impl TypeVarGenerator) {
        // magnitude : ∀n a. vec<n,a> -> a
        let n = ctx.new_variable();
        let a = ctx.new_variable();
        let vec_n_a = Type::Constructed(TypeName::Vec, vec![n, a.clone()]);
        self.register_poly("magnitude", vec![vec_n_a], a);

        // normalize : ∀n a. vec<n,a> -> vec<n,a>
        let n = ctx.new_variable();
        let a = ctx.new_variable();
        let vec_n_a = Type::Constructed(TypeName::Vec, vec![n, a]);
        self.register_poly("normalize", vec![vec_n_a.clone()], vec_n_a);

        // dot : ∀n a. vec<n,a> -> vec<n,a> -> a
        let n = ctx.new_variable();
        let a = ctx.new_variable();
        let vec_n_a = Type::Constructed(TypeName::Vec, vec![n, a.clone()]);
        self.register_poly("dot", vec![vec_n_a.clone(), vec_n_a], a);

        // cross : vec<3,f32> -> vec<3,f32> -> vec<3,f32>
        let vec3f32 = Type::Constructed(
            TypeName::Vec,
            vec![
                Type::Constructed(TypeName::Size(3), vec![]),
                Type::Constructed(TypeName::Float(32), vec![]),
            ],
        );
        self.register_poly("cross", vec![vec3f32.clone(), vec3f32.clone()], vec3f32);

        // distance : ∀n a. vec<n,a> -> vec<n,a> -> a
        let n = ctx.new_variable();
        let a = ctx.new_variable();
        let vec_n_a = Type::Constructed(TypeName::Vec, vec![n, a.clone()]);
        self.register_poly("distance", vec![vec_n_a.clone(), vec_n_a], a);

        // reflect : ∀n a. vec<n,a> -> vec<n,a> -> vec<n,a>
        let n = ctx.new_variable();
        let a = ctx.new_variable();
        let vec_n_a = Type::Constructed(TypeName::Vec, vec![n, a]);
        self.register_poly("reflect", vec![vec_n_a.clone(), vec_n_a.clone()], vec_n_a);

        // refract : ∀n a. vec<n,a> -> vec<n,a> -> a -> vec<n,a>
        let n = ctx.new_variable();
        let a = ctx.new_variable();
        let vec_n_a = Type::Constructed(TypeName::Vec, vec![n, a.clone()]);
        self.register_poly("refract", vec![vec_n_a.clone(), vec_n_a.clone(), a], vec_n_a);

        // Component-wise unary: abs, sign, floor, ceil, fract : vec<n,a> -> vec<n,a>
        for name in ["abs", "sign", "floor", "ceil", "fract"] {
            let n = ctx.new_variable();
            let a = ctx.new_variable();
            let vec_n_a = Type::Constructed(TypeName::Vec, vec![n, a]);
            self.register_poly(name, vec![vec_n_a.clone()], vec_n_a);
        }

        // Component-wise binary: min, max : vec<n,a> -> vec<n,a> -> vec<n,a>
        for name in ["min", "max"] {
            let n = ctx.new_variable();
            let a = ctx.new_variable();
            let vec_n_a = Type::Constructed(TypeName::Vec, vec![n, a]);
            self.register_poly(name, vec![vec_n_a.clone(), vec_n_a.clone()], vec_n_a);
        }

        // clamp : a -> a -> vec<n,a> -> vec<n,a> (currying-friendly: bounds first)
        let n = ctx.new_variable();
        let a = ctx.new_variable();
        let vec_n_a = Type::Constructed(TypeName::Vec, vec![n, a.clone()]);
        self.register_poly("clamp", vec![a.clone(), a, vec_n_a.clone()], vec_n_a);

        // Note: vector mix with scalar interpolant doesn't work with GLSL FMix
        // (requires all operands same type). Use scalar mix component-wise instead.

        // smoothstep : a -> a -> vec<n,a> -> vec<n,a>
        let n = ctx.new_variable();
        let a = ctx.new_variable();
        let vec_n_a = Type::Constructed(TypeName::Vec, vec![n, a.clone()]);
        self.register_poly("smoothstep", vec![a.clone(), a, vec_n_a.clone()], vec_n_a);
    }

    /// Register matrix operations
    fn register_matrix_operations(&mut self, ctx: &mut impl TypeVarGenerator) {
        // determinant : ∀n a. mat<n,n,a> -> a (square matrix only)
        let n = ctx.new_variable();
        let a = ctx.new_variable();
        let mat_n_n_a = Type::Constructed(TypeName::Mat, vec![n, a.clone(), a.clone()]);
        self.register_poly("determinant", vec![mat_n_n_a], a);

        // inverse : ∀n a. mat<n,n,a> -> mat<n,n,a> (square matrix only)
        let n = ctx.new_variable();
        let a = ctx.new_variable();
        let mat_n_n_a = Type::Constructed(TypeName::Mat, vec![n, a.clone(), a]);
        self.register_poly("inverse", vec![mat_n_n_a.clone()], mat_n_n_a);

        // outer : ∀n m a. vec<n,a> -> vec<m,a> -> mat<n,m,a>
        let n = ctx.new_variable();
        let m = ctx.new_variable();
        let a = ctx.new_variable();
        let vec_n_a = Type::Constructed(TypeName::Vec, vec![n.clone(), a.clone()]);
        let vec_m_a = Type::Constructed(TypeName::Vec, vec![m.clone(), a.clone()]);
        let mat_n_m_a = Type::Constructed(TypeName::Mat, vec![n, m, a]);
        self.register_poly("outer", vec![vec_n_a, vec_m_a], mat_n_m_a);

        // Surface "mul" overloads (desugared to mul_mat_mat, mul_mat_vec, mul_vec_mat in flattening)
        // mul : ∀n m p a. mat<n,m,a> -> mat<m,p,a> -> mat<n,p,a>
        let n = ctx.new_variable();
        let m = ctx.new_variable();
        let p = ctx.new_variable();
        let a = ctx.new_variable();
        let mat_n_m_a = Type::Constructed(TypeName::Mat, vec![n.clone(), m.clone(), a.clone()]);
        let mat_m_p_a = Type::Constructed(TypeName::Mat, vec![m, p.clone(), a.clone()]);
        let mat_n_p_a = Type::Constructed(TypeName::Mat, vec![n, p, a]);
        self.register_poly("mul", vec![mat_n_m_a, mat_m_p_a], mat_n_p_a);

        // mul : ∀n m a. mat<n,m,a> -> vec<m,a> -> vec<n,a>
        let n = ctx.new_variable();
        let m = ctx.new_variable();
        let a = ctx.new_variable();
        let vec_m_a = Type::Constructed(TypeName::Vec, vec![m.clone(), a.clone()]);
        let vec_n_a = Type::Constructed(TypeName::Vec, vec![n.clone(), a.clone()]);
        let mat_n_m = Type::Constructed(TypeName::Mat, vec![n, m, a]);
        self.register_poly("mul", vec![mat_n_m, vec_m_a], vec_n_a);

        // mul : ∀n m a. vec<n,a> -> mat<n,m,a> -> vec<m,a>
        let n = ctx.new_variable();
        let m = ctx.new_variable();
        let a = ctx.new_variable();
        let vec_n_a = Type::Constructed(TypeName::Vec, vec![n.clone(), a.clone()]);
        let vec_m_a = Type::Constructed(TypeName::Vec, vec![m.clone(), a.clone()]);
        let mat_n_m = Type::Constructed(TypeName::Mat, vec![n, m, a]);
        self.register_poly("mul", vec![vec_n_a, mat_n_m], vec_m_a);
    }

    /// Register higher-order functions like map
    fn register_higher_order_functions(&mut self, ctx: &mut impl TypeVarGenerator) {
        // map : ∀a b n. (a -> b) -> [n]a -> [n]b
        let a = ctx.new_variable();
        let b = ctx.new_variable();
        let n = ctx.new_variable();
        let func_type = Type::arrow(a.clone(), b.clone());
        let array_a = Type::Constructed(TypeName::Array, vec![n.clone(), a]);
        let array_b = Type::Constructed(TypeName::Array, vec![n, b]);
        self.register_poly("map", vec![func_type, array_a], array_b);

        // length : ∀a n. [n]a -> i32
        let a = ctx.new_variable();
        let n = ctx.new_variable();
        let array_a = Type::Constructed(TypeName::Array, vec![n, a]);
        let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
        self.register_poly("length", vec![array_a], i32_ty);

        // replicate : ∀a n. n -> a -> [n]a
        let a = ctx.new_variable();
        let n = ctx.new_variable();
        let array_a = Type::Constructed(TypeName::Array, vec![n.clone(), a.clone()]);
        self.register_poly("replicate", vec![n, a], array_a);

        // _w_uninit : ∀a. () -> a
        let a = ctx.new_variable();
        let unit_ty = Type::Constructed(TypeName::Unit, vec![]);
        self.register_poly("_w_uninit", vec![unit_ty], a);

        // _w_array_with : ∀a n. [n]a -> i32 -> a -> [n]a
        let a = ctx.new_variable();
        let n = ctx.new_variable();
        let array_a = Type::Constructed(TypeName::Array, vec![n, a.clone()]);
        let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
        self.register_poly("_w_array_with", vec![array_a.clone(), i32_ty, a], array_a);

        // reduce : ∀a n. (a -> a -> a) -> a -> [n]a -> a
        let a = ctx.new_variable();
        let n = ctx.new_variable();
        let op_type = Type::arrow(a.clone(), Type::arrow(a.clone(), a.clone())); // a -> a -> a
        let array_a = Type::Constructed(TypeName::Array, vec![n, a.clone()]);
        self.register_poly("reduce", vec![op_type, a.clone(), array_a], a);

        // filter : ∀a n. (a -> bool) -> [n]a -> ?k. [k]a
        let a = ctx.new_variable();
        let n = ctx.new_variable();
        let bool_ty = Type::Constructed(TypeName::Str("bool"), vec![]);
        let pred_type = Type::arrow(a.clone(), bool_ty); // a -> bool
        let array_a = Type::Constructed(TypeName::Array, vec![n, a.clone()]);
        // Existential return type: ?k. [k]a
        let k = "k".to_string();
        let k_var = Type::Constructed(TypeName::SizeVar(k.clone()), vec![]);
        let result_array = Type::Constructed(TypeName::Array, vec![k_var, a.clone()]);
        let existential_result = Type::Constructed(TypeName::Existential(vec![k]), vec![result_array]);
        self.register_poly("filter", vec![pred_type, array_a], existential_result);

        // TODO: scatter, etc.

        // Misc utility intrinsics
        self.register_misc_intrinsics();
    }

    /// Register miscellaneous utility intrinsics
    fn register_misc_intrinsics(&mut self) {
        let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);

        // _w_bitcast_i32_to_u32 : i32 -> u32
        self.register_poly(
            "_w_bitcast_i32_to_u32",
            vec![i32_ty],
            Type::Constructed(TypeName::UInt(32), vec![]),
        );
    }
}

impl Default for PolyBuiltins {
    fn default() -> Self {
        let mut ctx = polytype::Context::<TypeName>::default();
        Self::new(&mut ctx)
    }
}

/// Collect all type variable IDs from a type (in order of first occurrence)
fn collect_type_vars(ty: &Type) -> Vec<polytype::Variable> {
    let mut vars = Vec::new();
    collect_type_vars_inner(ty, &mut vars);
    vars
}

fn collect_type_vars_inner(ty: &Type, vars: &mut Vec<polytype::Variable>) {
    match ty {
        Type::Variable(id) => {
            if !vars.contains(id) {
                vars.push(*id);
            }
        }
        Type::Constructed(_, args) => {
            for arg in args {
                collect_type_vars_inner(arg, vars);
            }
        }
    }
}
