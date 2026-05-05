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
// Scalar shapes
// ---------------------------------------------------------------------------

/// `∀a. a -> a`
pub fn scalar_unary(ctx: &mut dyn TypeVarGenerator) -> TypeScheme {
    let a = ctx.new_variable();
    quantify(Type::arrow(a.clone(), a))
}

/// `∀a. a -> a -> a`
pub fn scalar_binary(ctx: &mut dyn TypeVarGenerator) -> TypeScheme {
    let a = ctx.new_variable();
    quantify(Type::arrow(a.clone(), Type::arrow(a.clone(), a)))
}

/// `∀a. a -> a -> a -> a`
pub fn scalar_ternary(ctx: &mut dyn TypeVarGenerator) -> TypeScheme {
    let a = ctx.new_variable();
    quantify(Type::arrow(
        a.clone(),
        Type::arrow(a.clone(), Type::arrow(a.clone(), a)),
    ))
}

// ---------------------------------------------------------------------------
// Vector shapes
// ---------------------------------------------------------------------------

/// `∀n a. vec<n,a> -> vec<n,a>`
pub fn vec_unary_same(ctx: &mut dyn TypeVarGenerator) -> TypeScheme {
    let n = ctx.new_variable();
    let a = ctx.new_variable();
    let vec_n_a = Type::Constructed(TypeName::Vec, vec![a, n]);
    quantify(Type::arrow(vec_n_a.clone(), vec_n_a))
}

/// `∀n a. vec<n,a> -> vec<n,a> -> vec<n,a>`
pub fn vec_binary_same(ctx: &mut dyn TypeVarGenerator) -> TypeScheme {
    let n = ctx.new_variable();
    let a = ctx.new_variable();
    let vec_n_a = Type::Constructed(TypeName::Vec, vec![a, n]);
    quantify(Type::arrow(
        vec_n_a.clone(),
        Type::arrow(vec_n_a.clone(), vec_n_a),
    ))
}

/// `∀n a. vec<n,a> -> vec<n,a> -> vec<n,a> -> vec<n,a>`
pub fn vec_ternary_same(ctx: &mut dyn TypeVarGenerator) -> TypeScheme {
    let n = ctx.new_variable();
    let a = ctx.new_variable();
    let vec_n_a = Type::Constructed(TypeName::Vec, vec![a, n]);
    quantify(Type::arrow(
        vec_n_a.clone(),
        Type::arrow(vec_n_a.clone(), Type::arrow(vec_n_a.clone(), vec_n_a)),
    ))
}

/// `∀n a. vec<n,a> -> a` — magnitude / length-squared shape.
pub fn vec_to_scalar(ctx: &mut dyn TypeVarGenerator) -> TypeScheme {
    let n = ctx.new_variable();
    let a = ctx.new_variable();
    let vec_n_a = Type::Constructed(TypeName::Vec, vec![a.clone(), n]);
    quantify(Type::arrow(vec_n_a, a))
}

/// `∀n a. vec<n,a> -> vec<n,a> -> a` — dot, distance.
pub fn vec_binary_to_scalar(ctx: &mut dyn TypeVarGenerator) -> TypeScheme {
    let n = ctx.new_variable();
    let a = ctx.new_variable();
    let vec_n_a = Type::Constructed(TypeName::Vec, vec![a.clone(), n]);
    quantify(Type::arrow(vec_n_a.clone(), Type::arrow(vec_n_a, a)))
}

/// `vec<3,f32> -> vec<3,f32> -> vec<3,f32>` — cross product (monomorphic).
pub fn vec3f32_binary(_ctx: &mut dyn TypeVarGenerator) -> TypeScheme {
    let vec3f32 = Type::Constructed(
        TypeName::Vec,
        vec![
            Type::Constructed(TypeName::Float(32), vec![]),
            Type::Constructed(TypeName::Size(3), vec![]),
        ],
    );
    TypeScheme::Monotype(Type::arrow(
        vec3f32.clone(),
        Type::arrow(vec3f32.clone(), vec3f32),
    ))
}

/// `∀n a. vec<n,a> -> vec<n,a> -> a -> vec<n,a>` — refract.
pub fn vec_vec_scalar_to_vec(ctx: &mut dyn TypeVarGenerator) -> TypeScheme {
    let n = ctx.new_variable();
    let a = ctx.new_variable();
    let vec_n_a = Type::Constructed(TypeName::Vec, vec![a.clone(), n]);
    quantify(Type::arrow(
        vec_n_a.clone(),
        Type::arrow(vec_n_a.clone(), Type::arrow(a, vec_n_a)),
    ))
}

/// `∀n a. a -> a -> vec<n,a> -> vec<n,a>` — clamp(lo, hi, x) with scalar
/// bounds and vector x.
pub fn vec_clamp_scalar_lohi(ctx: &mut dyn TypeVarGenerator) -> TypeScheme {
    let n = ctx.new_variable();
    let a = ctx.new_variable();
    let vec_n_a = Type::Constructed(TypeName::Vec, vec![a.clone(), n]);
    quantify(Type::arrow(
        a.clone(),
        Type::arrow(a, Type::arrow(vec_n_a.clone(), vec_n_a)),
    ))
}

/// `∀n a. vec<n,a> -> vec<n,a> -> a -> vec<n,a>` — mix(x, y, t) with
/// vector x/y and scalar interpolant.
pub fn vec_mix_scalar_interp(ctx: &mut dyn TypeVarGenerator) -> TypeScheme {
    let n = ctx.new_variable();
    let a = ctx.new_variable();
    let vec_n_a = Type::Constructed(TypeName::Vec, vec![a.clone(), n]);
    quantify(Type::arrow(
        vec_n_a.clone(),
        Type::arrow(vec_n_a.clone(), Type::arrow(a, vec_n_a)),
    ))
}

/// `∀n a. a -> a -> vec<n,a> -> vec<n,a>` — smoothstep(edge0, edge1, x)
/// with scalar edges and vector x.
pub fn vec_smoothstep_scalar_edges(ctx: &mut dyn TypeVarGenerator) -> TypeScheme {
    let n = ctx.new_variable();
    let a = ctx.new_variable();
    let vec_n_a = Type::Constructed(TypeName::Vec, vec![a.clone(), n]);
    quantify(Type::arrow(
        a.clone(),
        Type::arrow(a, Type::arrow(vec_n_a.clone(), vec_n_a)),
    ))
}

// ---------------------------------------------------------------------------
// Matrix shapes
// ---------------------------------------------------------------------------

/// `∀n a. mat<n,n,a> -> a` — determinant on a square matrix.
pub fn mat_square_to_scalar(ctx: &mut dyn TypeVarGenerator) -> TypeScheme {
    let n = ctx.new_variable();
    let a = ctx.new_variable();
    let mat_n_n_a = Type::Constructed(TypeName::Mat, vec![a.clone(), n.clone(), n]);
    quantify(Type::arrow(mat_n_n_a, a))
}

/// `∀n a. mat<n,n,a> -> mat<n,n,a>` — inverse on a square matrix.
pub fn mat_square_to_mat(ctx: &mut dyn TypeVarGenerator) -> TypeScheme {
    let n = ctx.new_variable();
    let a = ctx.new_variable();
    let mat_n_n_a = Type::Constructed(TypeName::Mat, vec![a, n.clone(), n]);
    quantify(Type::arrow(mat_n_n_a.clone(), mat_n_n_a))
}

/// `∀n m a. vec<n,a> -> vec<m,a> -> mat<n,m,a>` — outer product.
pub fn vec_vec_outer(ctx: &mut dyn TypeVarGenerator) -> TypeScheme {
    let n = ctx.new_variable();
    let m = ctx.new_variable();
    let a = ctx.new_variable();
    let vec_n_a = Type::Constructed(TypeName::Vec, vec![a.clone(), n.clone()]);
    let vec_m_a = Type::Constructed(TypeName::Vec, vec![a.clone(), m.clone()]);
    let mat_n_m_a = Type::Constructed(TypeName::Mat, vec![a, n, m]);
    quantify(Type::arrow(vec_n_a, Type::arrow(vec_m_a, mat_n_m_a)))
}

/// `∀n m p a. mat<n,m,a> -> mat<m,p,a> -> mat<n,p,a>` — `mul` overload.
pub fn mat_x_mat(ctx: &mut dyn TypeVarGenerator) -> TypeScheme {
    let n = ctx.new_variable();
    let m = ctx.new_variable();
    let p = ctx.new_variable();
    let a = ctx.new_variable();
    let mat_n_m_a = Type::Constructed(TypeName::Mat, vec![a.clone(), n.clone(), m.clone()]);
    let mat_m_p_a = Type::Constructed(TypeName::Mat, vec![a.clone(), m, p.clone()]);
    let mat_n_p_a = Type::Constructed(TypeName::Mat, vec![a, n, p]);
    quantify(Type::arrow(mat_n_m_a, Type::arrow(mat_m_p_a, mat_n_p_a)))
}

/// `∀n m a. mat<n,m,a> -> vec<m,a> -> vec<n,a>` — `mul` overload.
pub fn mat_x_vec(ctx: &mut dyn TypeVarGenerator) -> TypeScheme {
    let n = ctx.new_variable();
    let m = ctx.new_variable();
    let a = ctx.new_variable();
    let vec_m_a = Type::Constructed(TypeName::Vec, vec![a.clone(), m.clone()]);
    let vec_n_a = Type::Constructed(TypeName::Vec, vec![a.clone(), n.clone()]);
    let mat_n_m = Type::Constructed(TypeName::Mat, vec![a, n, m]);
    quantify(Type::arrow(mat_n_m, Type::arrow(vec_m_a, vec_n_a)))
}

/// `∀n m a. vec<n,a> -> mat<n,m,a> -> vec<m,a>` — `mul` overload.
pub fn vec_x_mat(ctx: &mut dyn TypeVarGenerator) -> TypeScheme {
    let n = ctx.new_variable();
    let m = ctx.new_variable();
    let a = ctx.new_variable();
    let vec_n_a = Type::Constructed(TypeName::Vec, vec![a.clone(), n.clone()]);
    let vec_m_a = Type::Constructed(TypeName::Vec, vec![a.clone(), m.clone()]);
    let mat_n_m = Type::Constructed(TypeName::Mat, vec![a, n, m]);
    quantify(Type::arrow(vec_n_a, Type::arrow(mat_n_m, vec_m_a)))
}

// ---------------------------------------------------------------------------
// Array / HOF shapes
// ---------------------------------------------------------------------------

fn array_type(elem: Type, addrspace: Type, size: Type) -> Type {
    Type::Constructed(TypeName::Array, vec![elem, size, addrspace])
}

/// `∀a n s. Array[a, s, n] -> i32` — array length.
pub fn array_to_i32(ctx: &mut dyn TypeVarGenerator) -> TypeScheme {
    let a = ctx.new_variable();
    let n = ctx.new_variable();
    let s = ctx.new_variable();
    let arr = array_type(a, s, n);
    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
    quantify(Type::arrow(arr, i32_ty))
}

/// `∀a. () -> a` — uninit/poison value.
pub fn unit_to_t(ctx: &mut dyn TypeVarGenerator) -> TypeScheme {
    let a = ctx.new_variable();
    let unit_ty = Type::Constructed(TypeName::Unit, vec![]);
    quantify(Type::arrow(unit_ty, a))
}

/// `∀a n s. Array[a, s, n] -> i32 -> a -> Array[a, s, n]` — array_with.
pub fn array_index_value_to_array(ctx: &mut dyn TypeVarGenerator) -> TypeScheme {
    let a = ctx.new_variable();
    let n = ctx.new_variable();
    let s = ctx.new_variable();
    let arr = array_type(a.clone(), s, n);
    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
    quantify(Type::arrow(arr.clone(), Type::arrow(i32_ty, Type::arrow(a, arr))))
}
