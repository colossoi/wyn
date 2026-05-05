use crate::ast::{Type, TypeName, TypeScheme};
use crate::type_checker::TypeVarGenerator;

/// A scheme builder is a function that produces a fresh `TypeScheme` —
/// fresh type variables drawn from the supplied generator. Schemes can't
/// be statically materialized because each instantiation needs its own
/// type variables; a function pointer is the smallest declarative shape
/// that lets the static catalog table stay declarative.
pub type SchemeBuilder = fn(&mut dyn TypeVarGenerator) -> TypeScheme;

/// Wrap a monotype in nested `Polytype` quantifiers, one per type
/// variable that appears in the type. Variables are quantified in order
/// of first occurrence.
pub fn quantify(ty: Type) -> TypeScheme {
    let vars = collect_type_vars(&ty);
    let mut scheme = TypeScheme::Monotype(ty);
    for var_id in vars.into_iter().rev() {
        scheme = TypeScheme::Polytype {
            variable: var_id,
            body: Box::new(scheme),
        };
    }
    scheme
}

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

// ---------------------------------------------------------------------------
// Common scheme shapes
// ---------------------------------------------------------------------------

/// `∀n a. vec<n,a> -> vec<n,a>` — used by every unary `vec.*` op
/// (sin/cos/abs/floor/etc.).
pub fn vec_unary_same(ctx: &mut dyn TypeVarGenerator) -> TypeScheme {
    let n = ctx.new_variable();
    let a = ctx.new_variable();
    let vec_n_a = Type::Constructed(TypeName::Vec, vec![a, n]);
    quantify(Type::arrow(vec_n_a.clone(), vec_n_a))
}

/// `∀n a. vec<n,a> -> vec<n,a> -> vec<n,a>` — binary `vec.*` ops
/// (pow/atan2/mod/min/max).
pub fn vec_binary_same(ctx: &mut dyn TypeVarGenerator) -> TypeScheme {
    let n = ctx.new_variable();
    let a = ctx.new_variable();
    let vec_n_a = Type::Constructed(TypeName::Vec, vec![a, n]);
    quantify(Type::arrow(
        vec_n_a.clone(),
        Type::arrow(vec_n_a.clone(), vec_n_a),
    ))
}

/// `∀n a. vec<n,a> -> vec<n,a> -> vec<n,a> -> vec<n,a>` — ternary
/// `vec.*` ops (clamp/mix/smoothstep).
pub fn vec_ternary_same(ctx: &mut dyn TypeVarGenerator) -> TypeScheme {
    let n = ctx.new_variable();
    let a = ctx.new_variable();
    let vec_n_a = Type::Constructed(TypeName::Vec, vec![a, n]);
    quantify(Type::arrow(
        vec_n_a.clone(),
        Type::arrow(vec_n_a.clone(), Type::arrow(vec_n_a.clone(), vec_n_a)),
    ))
}
