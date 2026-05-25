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
// Type-construction helpers
// ---------------------------------------------------------------------------

fn vec_type(elem: Type, size: Type) -> Type {
    Type::Constructed(TypeName::Vec, vec![elem, size])
}

fn mat_type(elem: Type, n: Type, m: Type) -> Type {
    Type::Constructed(TypeName::Mat, vec![elem, n, m])
}

fn array_type(elem: Type, addrspace: Type, size: Type) -> Type {
    Type::Constructed(TypeName::Array, vec![elem, size, addrspace])
}

fn arrow_chain(params: &[Type], ret: Type) -> Type {
    let mut t = ret;
    for p in params.iter().rev() {
        t = Type::arrow(p.clone(), t);
    }
    t
}

fn unit_ty() -> Type {
    Type::Constructed(TypeName::Unit, vec![])
}
fn i32_ty() -> Type {
    Type::Constructed(TypeName::Int(32), vec![])
}
fn u32_ty() -> Type {
    Type::Constructed(TypeName::UInt(32), vec![])
}

// ---------------------------------------------------------------------------
// Scalar shapes
// ---------------------------------------------------------------------------

/// `∀a. a -> a`
pub fn scalar_unary(ctx: &mut dyn TypeVarGenerator) -> TypeScheme {
    let a = ctx.new_variable();
    quantify(arrow_chain(&[a.clone()], a))
}

/// `∀a. a -> a -> a`
pub fn scalar_binary(ctx: &mut dyn TypeVarGenerator) -> TypeScheme {
    let a = ctx.new_variable();
    quantify(arrow_chain(&[a.clone(), a.clone()], a))
}

/// `∀a. a -> a -> a -> a`
pub fn scalar_ternary(ctx: &mut dyn TypeVarGenerator) -> TypeScheme {
    let a = ctx.new_variable();
    quantify(arrow_chain(&[a.clone(), a.clone(), a.clone()], a))
}

// ---------------------------------------------------------------------------
// Vector shapes
// ---------------------------------------------------------------------------

/// `∀n a. vec<n,a> -> vec<n,a>`
pub fn vec_unary_same(ctx: &mut dyn TypeVarGenerator) -> TypeScheme {
    let (n, a) = (ctx.new_variable(), ctx.new_variable());
    let v = vec_type(a, n);
    quantify(arrow_chain(&[v.clone()], v))
}

/// `∀n a. vec<n,a> -> vec<n,a> -> vec<n,a>`
pub fn vec_binary_same(ctx: &mut dyn TypeVarGenerator) -> TypeScheme {
    let (n, a) = (ctx.new_variable(), ctx.new_variable());
    let v = vec_type(a, n);
    quantify(arrow_chain(&[v.clone(), v.clone()], v))
}

/// `∀n a. vec<n,a> -> vec<n,a> -> vec<n,a> -> vec<n,a>`
pub fn vec_ternary_same(ctx: &mut dyn TypeVarGenerator) -> TypeScheme {
    let (n, a) = (ctx.new_variable(), ctx.new_variable());
    let v = vec_type(a, n);
    quantify(arrow_chain(&[v.clone(), v.clone(), v.clone()], v))
}

/// `∀n a. vec<n,a> -> a` — magnitude.
pub fn vec_to_scalar(ctx: &mut dyn TypeVarGenerator) -> TypeScheme {
    let (n, a) = (ctx.new_variable(), ctx.new_variable());
    quantify(arrow_chain(&[vec_type(a.clone(), n)], a))
}

/// `∀n a. vec<n,a> -> vec<n,a> -> a` — dot, distance.
pub fn vec_binary_to_scalar(ctx: &mut dyn TypeVarGenerator) -> TypeScheme {
    let (n, a) = (ctx.new_variable(), ctx.new_variable());
    let v = vec_type(a.clone(), n);
    quantify(arrow_chain(&[v.clone(), v], a))
}

/// `vec<3,f32> -> vec<3,f32> -> vec<3,f32>` — cross product.
pub fn vec3f32_binary(_ctx: &mut dyn TypeVarGenerator) -> TypeScheme {
    let v = vec_type(
        Type::Constructed(TypeName::Float(32), vec![]),
        Type::Constructed(TypeName::Size(3), vec![]),
    );
    TypeScheme::Monotype(arrow_chain(&[v.clone(), v.clone()], v))
}

/// `∀n a. vec<n,a> -> vec<n,a> -> a -> vec<n,a>` — refract / mix.
pub fn vec_vec_scalar_to_vec(ctx: &mut dyn TypeVarGenerator) -> TypeScheme {
    let (n, a) = (ctx.new_variable(), ctx.new_variable());
    let v = vec_type(a.clone(), n);
    quantify(arrow_chain(&[v.clone(), v.clone(), a], v))
}

/// `∀n a. a -> vec<n,a> -> vec<n,a>` — step(edge, x) with a scalar
/// edge that broadcasts across a vector x.
pub fn vec_scalar_edge_to_vec(ctx: &mut dyn TypeVarGenerator) -> TypeScheme {
    let (n, a) = (ctx.new_variable(), ctx.new_variable());
    let v = vec_type(a.clone(), n);
    quantify(arrow_chain(&[a, v.clone()], v))
}

/// `∀n a. a -> a -> vec<n,a> -> vec<n,a>` — clamp(lo, hi, x) /
/// smoothstep(edge0, edge1, x) with scalar bounds.
pub fn vec_clamp_scalar_lohi(ctx: &mut dyn TypeVarGenerator) -> TypeScheme {
    let (n, a) = (ctx.new_variable(), ctx.new_variable());
    let v = vec_type(a.clone(), n);
    quantify(arrow_chain(&[a.clone(), a, v.clone()], v))
}

/// Same shape as `vec_clamp_scalar_lohi` but kept distinct in case the
/// names diverge later.
pub use self::vec_clamp_scalar_lohi as vec_smoothstep_scalar_edges;

/// `∀n a. vec<n,a> -> vec<n,a> -> a -> vec<n,a>` — mix(x, y, t).
pub use self::vec_vec_scalar_to_vec as vec_mix_scalar_interp;

// ---------------------------------------------------------------------------
// Matrix shapes
// ---------------------------------------------------------------------------

/// `∀n a. mat<n,n,a> -> a` — determinant.
pub fn mat_square_to_scalar(ctx: &mut dyn TypeVarGenerator) -> TypeScheme {
    let (n, a) = (ctx.new_variable(), ctx.new_variable());
    quantify(arrow_chain(&[mat_type(a.clone(), n.clone(), n)], a))
}

/// `∀n a. mat<n,n,a> -> mat<n,n,a>` — inverse.
pub fn mat_square_to_mat(ctx: &mut dyn TypeVarGenerator) -> TypeScheme {
    let (n, a) = (ctx.new_variable(), ctx.new_variable());
    let m = mat_type(a, n.clone(), n);
    quantify(arrow_chain(&[m.clone()], m))
}

/// `∀n m a. vec<n,a> -> vec<m,a> -> mat<n,m,a>` — outer product.
pub fn vec_vec_outer(ctx: &mut dyn TypeVarGenerator) -> TypeScheme {
    let (n, m, a) = (ctx.new_variable(), ctx.new_variable(), ctx.new_variable());
    quantify(arrow_chain(
        &[vec_type(a.clone(), n.clone()), vec_type(a.clone(), m.clone())],
        mat_type(a, n, m),
    ))
}

/// `∀n m p a. mat<n,m,a> -> mat<m,p,a> -> mat<n,p,a>` — `mul`.
pub fn mat_x_mat(ctx: &mut dyn TypeVarGenerator) -> TypeScheme {
    let (n, m, p, a) = (
        ctx.new_variable(),
        ctx.new_variable(),
        ctx.new_variable(),
        ctx.new_variable(),
    );
    quantify(arrow_chain(
        &[
            mat_type(a.clone(), n.clone(), m.clone()),
            mat_type(a.clone(), m, p.clone()),
        ],
        mat_type(a, n, p),
    ))
}

/// `∀n m a. mat<n,m,a> -> vec<m,a> -> vec<n,a>` — `mul`.
pub fn mat_x_vec(ctx: &mut dyn TypeVarGenerator) -> TypeScheme {
    let (n, m, a) = (ctx.new_variable(), ctx.new_variable(), ctx.new_variable());
    quantify(arrow_chain(
        &[mat_type(a.clone(), n.clone(), m.clone()), vec_type(a.clone(), m)],
        vec_type(a, n),
    ))
}

/// `∀n m a. vec<n,a> -> mat<n,m,a> -> vec<m,a>` — `mul`.
pub fn vec_x_mat(ctx: &mut dyn TypeVarGenerator) -> TypeScheme {
    let (n, m, a) = (ctx.new_variable(), ctx.new_variable(), ctx.new_variable());
    quantify(arrow_chain(
        &[vec_type(a.clone(), n.clone()), mat_type(a.clone(), n, m.clone())],
        vec_type(a, m),
    ))
}

// ---------------------------------------------------------------------------
// Array / HOF shapes
// ---------------------------------------------------------------------------

/// `∀a n s. Array[a, s, n] -> i32` — array length.
pub fn array_to_i32(ctx: &mut dyn TypeVarGenerator) -> TypeScheme {
    let (a, n, s) = (ctx.new_variable(), ctx.new_variable(), ctx.new_variable());
    quantify(arrow_chain(&[array_type(a, s, n)], i32_ty()))
}

/// `∀a. () -> a` — uninit/poison value.
pub fn unit_to_t(ctx: &mut dyn TypeVarGenerator) -> TypeScheme {
    let a = ctx.new_variable();
    quantify(arrow_chain(&[unit_ty()], a))
}

/// `∀a n s. Array[a, s, n] -> i32 -> a -> Array[a, s, n]` — array_with.
pub fn array_index_value_to_array(ctx: &mut dyn TypeVarGenerator) -> TypeScheme {
    let (a, n, s) = (ctx.new_variable(), ctx.new_variable(), ctx.new_variable());
    let arr = array_type(a.clone(), s, n);
    quantify(arrow_chain(&[arr.clone(), i32_ty(), a], arr))
}

/// `∀a n s. i32 -> a -> Array[a, s, n]` — replicate.
pub fn replicate_scheme(ctx: &mut dyn TypeVarGenerator) -> TypeScheme {
    let (a, n, s) = (ctx.new_variable(), ctx.new_variable(), ctx.new_variable());
    quantify(arrow_chain(&[i32_ty(), a.clone()], array_type(a, s, n)))
}

/// `∀a n s. (a -> a -> a) -> a -> Array[a, s, n] -> a` — reduce.
pub fn reduce_scheme(ctx: &mut dyn TypeVarGenerator) -> TypeScheme {
    let (a, n, s) = (ctx.new_variable(), ctx.new_variable(), ctx.new_variable());
    let op = arrow_chain(&[a.clone(), a.clone()], a.clone());
    let arr = array_type(a.clone(), s, n);
    quantify(arrow_chain(&[op, a.clone(), arr], a))
}

/// `∀a n s. (a -> a -> a) -> a -> Array[a, s, n] -> Array[a, s, n]` — scan.
pub fn scan_scheme(ctx: &mut dyn TypeVarGenerator) -> TypeScheme {
    let (a, n, s) = (ctx.new_variable(), ctx.new_variable(), ctx.new_variable());
    let op = arrow_chain(&[a.clone(), a.clone()], a.clone());
    let arr_in = array_type(a.clone(), s.clone(), n.clone());
    let arr_out = array_type(a.clone(), s, n);
    quantify(arrow_chain(&[op, a, arr_in], arr_out))
}

/// `∀a b n s. (a -> b) -> Array[a, s, n] -> Array[b, s, n]` — map.
pub fn map_scheme(ctx: &mut dyn TypeVarGenerator) -> TypeScheme {
    let (a, b, n, s) = (
        ctx.new_variable(),
        ctx.new_variable(),
        ctx.new_variable(),
        ctx.new_variable(),
    );
    let f = arrow_chain(&[a.clone()], b.clone());
    quantify(arrow_chain(
        &[f, array_type(a, s.clone(), n.clone())],
        array_type(b, s, n),
    ))
}

/// `∀a b n m s1 s2. (a -> b) -> Array[a, s1, n] -> Array[b, s2, m] -> i32 -> ()`
/// — map_into (effectful).
pub fn map_into_scheme(ctx: &mut dyn TypeVarGenerator) -> TypeScheme {
    let (a, b, n, m, s1, s2) = (
        ctx.new_variable(),
        ctx.new_variable(),
        ctx.new_variable(),
        ctx.new_variable(),
        ctx.new_variable(),
        ctx.new_variable(),
    );
    let f = arrow_chain(&[a.clone()], b.clone());
    quantify(arrow_chain(
        &[f, array_type(a, s1, n), array_type(b, s2, m), i32_ty()],
        unit_ty(),
    ))
}

/// `u32 -> u32 -> u32` — rotr32.
pub fn u32_binary(_ctx: &mut dyn TypeVarGenerator) -> TypeScheme {
    let u = u32_ty();
    TypeScheme::Monotype(arrow_chain(&[u.clone(), u.clone()], u))
}

// ---------------------------------------------------------------------------
// Textures / samplers
// ---------------------------------------------------------------------------

fn texture2d_ty() -> Type {
    Type::Constructed(TypeName::Texture2D, vec![])
}
fn sampler_ty() -> Type {
    Type::Constructed(TypeName::Sampler, vec![])
}
fn f32_ty() -> Type {
    Type::Constructed(TypeName::Float(32), vec![])
}
fn vec_n(elem: Type, n: usize) -> Type {
    vec_type(elem, Type::Constructed(TypeName::Size(n), vec![]))
}

/// `texture2d -> vec2<i32> -> i32 -> vec4<f32>` — raw texel fetch
/// (`texture_load`). Monomorphic; pure (no derivatives).
pub fn texture_load_scheme(_ctx: &mut dyn TypeVarGenerator) -> TypeScheme {
    let coord = vec_n(i32_ty(), 2);
    let result = vec_n(f32_ty(), 4);
    TypeScheme::Monotype(arrow_chain(&[texture2d_ty(), coord, i32_ty()], result))
}

/// `texture2d -> sampler -> vec2<f32> -> f32 -> vec4<f32>` — filtered
/// sample at an EXPLICIT LOD (`texture_sample`). The trailing `f32` is
/// the mip level. Explicit LOD keeps this referentially transparent (no
/// screen-space-derivative dependence); see the texture plan's v2 note
/// for gradient-based filtering.
pub fn texture_sample_scheme(_ctx: &mut dyn TypeVarGenerator) -> TypeScheme {
    let uv = vec_n(f32_ty(), 2);
    let result = vec_n(f32_ty(), 4);
    TypeScheme::Monotype(arrow_chain(&[texture2d_ty(), sampler_ty(), uv, f32_ty()], result))
}
