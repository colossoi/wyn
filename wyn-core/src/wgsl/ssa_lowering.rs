//! WGSL SSA-to-text lowering.
//!
//! Entry points carry `@vertex` / `@fragment` / `@compute` attributes;
//! module-scope types and bindings precede function definitions.
//!
//! This file provides:
//! - [`wgsl_mangle`]: injective identifier mangler, aware of WGSL reserved words.
//! - [`validate_wgsl_identifier`]: host-contract identifier validator.
//! - [`TypeEmitter`]: Wyn polytype → WGSL type string, with cached tuple structs.
//! - [`lower`]: entry point.

use crate::builtins::{by_id, catalog};
use crate::{LookupMap, LookupSet};
use std::fmt::Write as _;

use polytype::Type as PolyType;

use crate::ast::{Span, TypeName};
use crate::builtins::lowering::{BuiltinLowering, PrimOp};
use crate::error::Result;
use crate::ssa::types::{
    EntryPoint, ExecutionModel, FuncBody, Function, InstKind, IoDecoration, Program, ValueId, ValueRef,
    WynInstNode,
};
use crate::types::TypeExt;
use crate::BindingRef;

/// Lower an SSA program to a WGSL module. The module contains all entry
/// points (distinguished by `@vertex` / `@fragment` / `@compute`
/// attributes), module-scope types, bindings, and helper functions.
pub fn lower(program: &Program) -> Result<String> {
    let mut ctx = LowerCtx::new(program);
    ctx.lower_program()
}

// -----------------------------------------------------------------------------
// Identifier mangling
// -----------------------------------------------------------------------------

/// Injective mangle from any Rust/Wyn identifier to a legal WGSL identifier.
pub fn wgsl_mangle(name: &str) -> String {
    let leads_with_alnum =
        name.chars().next().is_some_and(|c| matches!(c, 'A'..='Z' | 'a'..='z' | '0'..='9'));
    let mut out = String::with_capacity(name.len() + 2);
    out.push('w');
    if leads_with_alnum {
        out.push('_');
    }
    for c in name.chars() {
        match c {
            'A'..='Z' | 'a'..='z' | '0'..='9' => out.push(c),
            '_' => out.push_str("_U"),
            '.' => out.push_str("_D"),
            '$' => out.push_str("_S"),
            _ => {
                let _ = write!(out, "_X{:x}_", c as u32);
            }
        }
    }
    if wgsl_keyword_set().contains(out.as_str()) {
        out.push_str("_R");
    }
    out
}

/// Validate that a host-contract name is a legal WGSL identifier.
pub fn validate_wgsl_identifier(name: &str) -> core::result::Result<(), String> {
    if name.is_empty() {
        return Err("identifier must not be empty".to_string());
    }
    let mut chars = name.chars();
    match chars.next().unwrap() {
        'A'..='Z' | 'a'..='z' | '_' => {}
        c => {
            return Err(format!(
                "identifier '{}' must start with a letter or underscore, got '{}'",
                name, c
            ));
        }
    }
    for c in chars {
        if !matches!(c, 'A'..='Z' | 'a'..='z' | '0'..='9' | '_') {
            return Err(format!(
                "identifier '{}' contains illegal character '{}'",
                name, c
            ));
        }
    }
    if name == "_" || name.starts_with("__") {
        return Err(format!("identifier '{}' is reserved by WGSL", name));
    }
    if wgsl_keyword_set().contains(name) {
        return Err(format!("identifier '{}' is a WGSL keyword", name));
    }
    Ok(())
}

fn wgsl_keyword_set() -> &'static LookupSet<&'static str> {
    use std::sync::OnceLock;
    static SET: OnceLock<LookupSet<&'static str>> = OnceLock::new();
    SET.get_or_init(|| {
        [
            // Keywords (WGSL §2.4)
            "alias",
            "break",
            "case",
            "const",
            "const_assert",
            "continue",
            "continuing",
            "default",
            "diagnostic",
            "discard",
            "else",
            "enable",
            "false",
            "fn",
            "for",
            "if",
            "let",
            "loop",
            "override",
            "requires",
            "return",
            "struct",
            "switch",
            "true",
            "var",
            "while",
            // Type predeclared names (§6)
            "array",
            "atomic",
            "bool",
            "f16",
            "f32",
            "i32",
            "mat2x2",
            "mat2x3",
            "mat2x4",
            "mat3x2",
            "mat3x3",
            "mat3x4",
            "mat4x2",
            "mat4x3",
            "mat4x4",
            "ptr",
            "sampler",
            "sampler_comparison",
            "texture_1d",
            "texture_2d",
            "texture_2d_array",
            "texture_3d",
            "texture_cube",
            "texture_cube_array",
            "texture_depth_2d",
            "texture_depth_2d_array",
            "texture_depth_cube",
            "texture_depth_cube_array",
            "texture_depth_multisampled_2d",
            "texture_external",
            "texture_multisampled_2d",
            "texture_storage_1d",
            "texture_storage_2d",
            "texture_storage_2d_array",
            "texture_storage_3d",
            "u32",
            "vec2",
            "vec3",
            "vec4",
            // Reserved words (§2.5) — practical subset.
            "NULL",
            "Self",
            "abstract",
            "active",
            "addressof",
            "as",
            "async",
            "attribute",
            "auto",
            "await",
            "become",
            "binding_array",
            "cast",
            "catch",
            "class",
            "co_await",
            "co_return",
            "co_yield",
            "coherent",
            "column_major",
            "common",
            "compile",
            "compile_fragment",
            "concept",
            "const_cast",
            "consteval",
            "constexpr",
            "constinit",
            "crate",
            "debugger",
            "decltype",
            "delete",
            "demote",
            "demote_to_helper",
            "do",
            "dynamic_cast",
            "enum",
            "explicit",
            "export",
            "extends",
            "extern",
            "external",
            "fallthrough",
            "filter",
            "final",
            "finally",
            "friend",
            "from",
            "fxgroup",
            "get",
            "goto",
            "groupshared",
            "highp",
            "impl",
            "implements",
            "import",
            "inline",
            "instanceof",
            "interface",
            "layout",
            "lowp",
            "macro",
            "macro_rules",
            "match",
            "mediump",
            "meta",
            "mod",
            "module",
            "move",
            "mut",
            "mutable",
            "namespace",
            "new",
            "nil",
            "noexcept",
            "noinline",
            "nointerpolation",
            "noperspective",
            "null",
            "nullptr",
            "of",
            "operator",
            "package",
            "packoffset",
            "partition",
            "pass",
            "patch",
            "pixelfragment",
            "precise",
            "precision",
            "premerge",
            "priv",
            "protected",
            "pub",
            "public",
            "readonly",
            "ref",
            "regardless",
            "register",
            "reinterpret_cast",
            "require",
            "resource",
            "restrict",
            "self",
            "set",
            "shared",
            "sizeof",
            "smooth",
            "snorm",
            "static",
            "static_assert",
            "static_cast",
            "std",
            "subroutine",
            "super",
            "target",
            "template",
            "this",
            "thread_local",
            "throw",
            "trait",
            "try",
            "type",
            "typedef",
            "typeid",
            "typename",
            "typeof",
            "union",
            "unless",
            "unorm",
            "unsafe",
            "unsized",
            "use",
            "using",
            "varying",
            "virtual",
            "volatile",
            "wgsl",
            "where",
            "with",
            "writeonly",
            "yield",
        ]
        .iter()
        .copied()
        .collect()
    })
}

// -----------------------------------------------------------------------------
// Type lowering
// -----------------------------------------------------------------------------

/// Lower Wyn polytypes to WGSL type strings, caching generated tuple structs.
pub struct TypeEmitter {
    tuple_type_cache: LookupMap<String, String>,
    pub tuple_structs: LookupMap<String, Vec<String>>,
    tuple_counter: usize,
    /// Virtual-array range structs, cached by element type string. A
    /// virtual array (produced by `ArrayRange`) lowers to a struct
    /// `{ f0: T, f1: T, f2: T }` holding `(start, step, len)`. The `fN`
    /// names match `Project`'s `.f{index}` emission so start/step/len
    /// project naturally; `Index` and `_w_intrinsic_length` have
    /// dedicated handling.
    virtual_range_cache: LookupMap<String, String>,
    pub virtual_range_structs: Vec<(String, String)>, // (struct_name, elem_ty)
    virtual_range_counter: usize,
    /// `array_variant_bounded` arrays — function-local fixed-capacity
    /// buffer plus a runtime length. Lowers to `{ buffer: array<T, N>,
    /// len: u32 }`. Key: `(elem_wgsl_ty, capacity)`.
    bounded_cache: LookupMap<(String, u32), String>,
    /// `(struct_name, elem_ty, capacity)` for emission.
    pub bounded_structs: Vec<(String, String, u32)>,
    bounded_counter: usize,
}

impl Default for TypeEmitter {
    fn default() -> Self {
        Self::new()
    }
}

impl TypeEmitter {
    pub fn new() -> Self {
        Self {
            tuple_type_cache: LookupMap::new(),
            tuple_structs: LookupMap::new(),
            tuple_counter: 0,
            virtual_range_cache: LookupMap::new(),
            virtual_range_structs: Vec::new(),
            virtual_range_counter: 0,
            bounded_cache: LookupMap::new(),
            bounded_structs: Vec::new(),
            bounded_counter: 0,
        }
    }

    /// Look up or create the WGSL struct name for a bounded array with
    /// element type `elem` (already WGSL-lowered) and capacity `n`.
    fn bounded_struct(&mut self, elem: &str, n: u32) -> String {
        let key = (elem.to_string(), n);
        if let Some(name) = self.bounded_cache.get(&key) {
            return name.clone();
        }
        let name = format!("Bounded{}", self.bounded_counter);
        self.bounded_counter += 1;
        self.bounded_cache.insert(key, name.clone());
        self.bounded_structs.push((name.clone(), elem.to_string(), n));
        name
    }

    /// Look up or create the WGSL struct name for a virtual-array range
    /// whose start/step/len are all of `elem`'s WGSL type.
    fn virtual_range_struct(&mut self, elem: &str) -> String {
        if let Some(name) = self.virtual_range_cache.get(elem) {
            return name.clone();
        }
        let name = format!("VirtRange{}", self.virtual_range_counter);
        self.virtual_range_counter += 1;
        self.virtual_range_cache.insert(elem.to_string(), name.clone());
        self.virtual_range_structs.push((name.clone(), elem.to_string()));
        name
    }

    pub fn type_to_wgsl(&mut self, ty: &PolyType<TypeName>) -> Result<String> {
        match ty {
            PolyType::Constructed(name, args) => match name {
                TypeName::Float(32) => Ok("f32".to_string()),
                TypeName::Int(32) => Ok("i32".to_string()),
                TypeName::UInt(32) => Ok("u32".to_string()),
                TypeName::Bool => Ok("bool".to_string()),
                // Opaque GPU resources (v1: 2D float texture + sampler).
                TypeName::Texture2D => Ok("texture_2d<f32>".to_string()),
                TypeName::Sampler => Ok("sampler".to_string()),
                TypeName::StorageTexture => Err(crate::err_wgsl!(
                    "StorageTexture reached runtime WGSL type lowering; terminal EGIR resource \
                     erasure must remove image handles from SSA"
                )),
                TypeName::Float(bits) | TypeName::Int(bits) | TypeName::UInt(bits) if *bits != 32 => Err(
                    crate::err_wgsl!("WGSL does not support {}-bit scalars (found {:?})", bits, ty),
                ),
                TypeName::Unit => Ok("/* unit */".to_string()),
                TypeName::Tuple(_) => {
                    let elem_types: Result<Vec<String>> =
                        args.iter().map(|a| self.type_to_wgsl(a)).collect();
                    let elem_types = elem_types?;
                    let sig = elem_types.join(",");
                    if let Some(name) = self.tuple_type_cache.get(&sig) {
                        return Ok(name.clone());
                    }
                    let struct_name = format!("T{}", self.tuple_counter);
                    self.tuple_counter += 1;
                    self.tuple_structs.insert(struct_name.clone(), elem_types);
                    self.tuple_type_cache.insert(sig, struct_name.clone());
                    Ok(struct_name)
                }
                TypeName::Vec => {
                    let elem = self.type_to_wgsl(
                        ty.elem_type()
                            .ok_or_else(|| crate::err_wgsl!("Vec type missing elem arg: {:?}", ty))?,
                    )?;
                    let n = ty
                        .vec_size()
                        .ok_or_else(|| crate::err_wgsl!("Vec type has non-concrete size: {:?}", ty))?;
                    if !(2..=4).contains(&n) {
                        return Err(crate::err_wgsl!("WGSL vector size must be 2/3/4, got {}", n));
                    }
                    Ok(format!("vec{}<{}>", n, elem))
                }
                TypeName::Mat => {
                    let elem = self.type_to_wgsl(
                        ty.elem_type()
                            .ok_or_else(|| crate::err_wgsl!("Mat type missing elem arg: {:?}", ty))?,
                    )?;
                    let cols = ty
                        .mat_cols()
                        .ok_or_else(|| crate::err_wgsl!("Mat type has non-concrete cols: {:?}", ty))?;
                    let rows = ty
                        .mat_rows()
                        .ok_or_else(|| crate::err_wgsl!("Mat type has non-concrete rows: {:?}", ty))?;
                    if elem != "f32" {
                        return Err(crate::err_wgsl!("WGSL matrices are f32-only (got {})", elem));
                    }
                    Ok(format!("mat{}x{}<{}>", cols, rows, elem))
                }
                TypeName::Array => {
                    let elem = self.type_to_wgsl(
                        ty.elem_type()
                            .ok_or_else(|| crate::err_wgsl!("Array type missing elem arg: {:?}", ty))?,
                    )?;
                    // Virtual arrays (ranges) are `{start, step, len}` triples
                    // in the same elem type; lower to a generated struct.
                    if let Some(PolyType::Constructed(TypeName::ArrayVariantVirtual, _)) =
                        ty.array_variant()
                    {
                        return Ok(self.virtual_range_struct(&elem));
                    }
                    // View arrays are runtime `{offset, len}` handles into a
                    // backing storage buffer. The buffer is static (recovered
                    // from the type's region), so the VALUE is just the runtime
                    // pair — modelled as `vec2<u32>` (`.x` = offset, `.y` = len).
                    // (The storage *binding* is declared `array<T>` separately;
                    // this is the type of a view VALUE that flows through lets,
                    // function params, and block params.)
                    if let Some(PolyType::Constructed(TypeName::ArrayVariantView, _)) = ty.array_variant() {
                        return Ok("vec2<u32>".to_string());
                    }
                    // Bounded arrays are `{buffer: array<T, N>, len: u32}` —
                    // function-local fixed-capacity buffer plus a runtime
                    // count.
                    if let Some(PolyType::Constructed(TypeName::ArrayVariantBounded, _)) =
                        ty.array_variant()
                    {
                        let n = match ty.array_size() {
                            Some(PolyType::Constructed(TypeName::Size(n), _)) => *n as u32,
                            _ => {
                                return Err(crate::err_wgsl!(
                                    "Bounded array must have Size(N) capacity, got {:?}",
                                    ty
                                ));
                            }
                        };
                        return Ok(self.bounded_struct(&elem, n));
                    }
                    match ty.array_size() {
                        Some(PolyType::Constructed(TypeName::Size(n), _)) => {
                            Ok(format!("array<{}, {}>", elem, n))
                        }
                        // Runtime-sized `array<T>`. Legal in WGSL only at
                        // storage-binding sites; naga will reject it
                        // elsewhere, which is the correct signal.
                        _ => Ok(format!("array<{}>", elem)),
                    }
                }
                TypeName::Record(fields) => Err(crate::err_wgsl!(
                    "Record type reached WGSL lowering (should have been lowered earlier): {:?}",
                    fields
                )),
                _ => Err(crate::err_wgsl!("unsupported type in WGSL lowering: {:?}", ty)),
            },
            _ => Err(crate::err_wgsl!("unsupported type in WGSL lowering: {:?}", ty)),
        }
    }
}

// -----------------------------------------------------------------------------
// Module-level lowering
// -----------------------------------------------------------------------------

fn wgsl_var(id: ValueId) -> String {
    use crate::ssa::framework::Key;
    let ffi = id.data().as_ffi();
    let idx = ffi & 0xFFFFFFFF;
    let ver = ffi >> 32;
    format!("v{}_{}", idx, ver)
}

fn storage_image_global_name(binding: BindingRef) -> String {
    format!("_img_{}_{}", binding.set, binding.binding)
}

fn wgsl_place(id: crate::ssa::types::PlaceId) -> String {
    use crate::ssa::framework::Key;
    let ffi = id.data().as_ffi();
    let idx = ffi & 0xFFFFFFFF;
    let ver = ffi >> 32;
    format!("p{}_{}", idx, ver)
}

/// True if `ty` is a view-variant unsized array (i.e. backed by a
/// storage buffer, accessed through a `@group @binding` global rather
/// than a function parameter).
fn is_view_array_ty(ty: &polytype::Type<TypeName>) -> bool {
    let polytype::Type::Constructed(TypeName::Array, args) = ty else {
        return false;
    };
    if args.len() != 3 {
        return false;
    }
    // args = [elem, variant, size]
    crate::types::is_array_variant_view(&args[1])
}

struct LowerCtx<'a> {
    program: &'a Program,
    type_emitter: TypeEmitter,
    lowered: LookupSet<String>,
    indent: usize,
    /// Track mangled names for collision detection.
    mangled_names: LookupMap<String, String>,
    /// Entry-point output structs keyed by their field signature
    /// ("attr0+ty0,attr1+ty1,..."). Maps sig → (struct_name, fields).
    /// `fields` is an ordered list of (field_name, attribute_prefix,
    /// wgsl_type) tuples.
    output_structs: LookupMap<String, (String, Vec<(String, String, String)>)>,
    output_struct_counter: usize,
    /// Push-constant block info per entry-point name. Compute entries
    /// whose inputs carry `push_constant_offset` are backed by a uniform
    /// block in WGSL (WebGPU has no push constants). Each block holds
    /// one field per push-constant input; fields are keyed by the input
    /// index in `entry.inputs` so `lower_entry_point` can route the
    /// corresponding SSA `ValueId` to `<block_var>.<field_name>`.
    pc_blocks: LookupMap<String, PcBlock>,
    /// If the current compute entry's source declared its own
    /// `#[builtin(global_invocation_id)]` param, this holds that param's
    /// mangled WGSL name. `_w_intrinsic_thread_id()` lowering reads from
    /// this in preference to the synthesized `_wgsl_gid`, since WGSL
    /// rejects duplicate `@builtin` declarations on the same entry.
    wgsl_gid_alias: Option<String>,
    /// Compile-time-constant arrays promoted to module-scope `var<private>`
    /// globals so a runtime index addresses one shared materialization
    /// instead of a per-occurrence `var<function>` (the WGSL analog of the
    /// SPIR-V `Private`-global hoist). Keyed by the initializer expression,
    /// which is already value-deduped, so equal arrays collapse to one
    /// global. Value is `(global_name, wgsl_type)`.
    private_const_globals: LookupMap<String, (String, String)>,
}

/// Uniform-block stand-in for a compute entry's push-constant inputs.
struct PcBlock {
    struct_name: String,
    var_name: String,
    /// Synthesized `@group(set) @binding(binding)`.
    set: u32,
    binding: u32,
    /// Per push-constant input: index into `entry.inputs`, the WGSL
    /// field name, and the WGSL type string. Preserves input order.
    fields: Vec<(usize, String, String)>,
}

impl<'a> LowerCtx<'a> {
    fn new(program: &'a Program) -> Self {
        Self {
            program,
            type_emitter: TypeEmitter::new(),
            lowered: LookupSet::new(),
            indent: 0,
            mangled_names: LookupMap::new(),
            output_structs: LookupMap::new(),
            output_struct_counter: 0,
            pc_blocks: LookupMap::new(),
            wgsl_gid_alias: None,
            private_const_globals: LookupMap::new(),
        }
    }

    /// Get or create an entry-output struct for a list of
    /// `(attribute_prefix, wgsl_type)` pairs. Attribute prefix includes
    /// the trailing space (e.g. `"@builtin(position) "`).
    fn get_or_create_output_struct(&mut self, fields: Vec<(String, String)>) -> String {
        let sig = fields.iter().map(|(a, t)| format!("{}{}", a, t)).collect::<Vec<_>>().join(",");
        if let Some((name, _)) = self.output_structs.get(&sig) {
            return name.clone();
        }
        let name = format!("VsOut{}", self.output_struct_counter);
        self.output_struct_counter += 1;
        let numbered: Vec<(String, String, String)> =
            fields.into_iter().enumerate().map(|(i, (attr, ty))| (format!("f{}", i), attr, ty)).collect();
        self.output_structs.insert(sig, (name.clone(), numbered));
        name
    }

    fn indent_str(&self) -> String {
        "    ".repeat(self.indent)
    }

    fn mangle_tracked(&mut self, name: &str) -> Result<String> {
        let mangled = wgsl_mangle(name);
        if let Some(existing) = self.mangled_names.get(&mangled) {
            if existing != name {
                return Err(crate::err_wgsl!(
                    "Identifier mangling collision: '{}' and '{}' both produced '{}'",
                    existing,
                    name,
                    mangled
                ));
            }
        } else {
            self.mangled_names.insert(mangled.clone(), name.to_string());
        }
        Ok(mangled)
    }

    fn lower_program(&mut self) -> Result<String> {
        // Body: emit referenced functions + entry points into `code`, then
        // prepend accumulated tuple struct declarations and bindings.
        let mut code = String::new();

        // Emit all non-extern functions first (entry points reference them).
        for func in &self.program.functions {
            self.lower_function(func, &mut code)?;
        }

        // Emit entry points.
        for entry in &self.program.entry_points {
            self.lower_entry_point(entry, &mut code)?;
        }

        let mut output = String::new();
        writeln!(output, "// Generated by the Wyn compiler — WGSL backend").unwrap();
        writeln!(output).unwrap();

        // Tuple struct declarations (cached from type_to_wgsl calls).
        let mut structs: Vec<_> = self.type_emitter.tuple_structs.iter().collect();
        structs.sort_by_key(|(name, _)| (*name).clone());
        for (name, field_types) in structs {
            writeln!(output, "struct {} {{", name).unwrap();
            for (i, ft) in field_types.iter().enumerate() {
                writeln!(output, "    f{}: {},", i, ft).unwrap();
            }
            writeln!(output, "}}").unwrap();
            writeln!(output).unwrap();
        }

        // Virtual-array range struct declarations (from ArrayRange
        // creation sites / `type_to_wgsl` on `Array[_, _, Virtual]`).
        // Field layout mirrors the tuple layout (`f0`/`f1`/`f2`) so
        // `Project` can emit `.fN` without special-casing, matching
        // SPIR-V's `composite_extract` indices 0/1/2 for start/step/len.
        for (name, elem) in &self.type_emitter.virtual_range_structs {
            writeln!(output, "struct {} {{", name).unwrap();
            writeln!(output, "    f0: {},", elem).unwrap();
            writeln!(output, "    f1: {},", elem).unwrap();
            writeln!(output, "    f2: {},", elem).unwrap();
            writeln!(output, "}}").unwrap();
            writeln!(output).unwrap();
        }

        // Bounded-array struct declarations. Field layout mirrors the
        // tuple `f0`/`f1` convention so `Project` reaches buffer/len via
        // `.f0`/`.f1` without special-casing, matching SPIR-V's
        // `composite_extract` indices 0/1.
        for (name, elem, n) in &self.type_emitter.bounded_structs {
            writeln!(output, "struct {} {{", name).unwrap();
            writeln!(output, "    f0: array<{}, {}>,", elem, n).unwrap();
            writeln!(output, "    f1: i32,").unwrap();
            writeln!(output, "}}").unwrap();
            writeln!(output).unwrap();
        }

        // Entry-point output struct declarations — each field carries
        // its `@builtin(...)` or `@location(N)` attribute inline,
        // since WGSL requires the decoration to sit on the struct
        // member rather than a free module-scope variable.
        let mut structs: Vec<_> = self.output_structs.values().collect();
        structs.sort_by_key(|(name, _)| name.clone());
        for (name, fields) in structs {
            writeln!(output, "struct {} {{", name).unwrap();
            for (field_name, attr, ty) in fields {
                writeln!(output, "    {}{}: {},", attr, field_name, ty).unwrap();
            }
            writeln!(output, "}}").unwrap();
            writeln!(output).unwrap();
        }

        // Compile-time-constant arrays hoisted to shared module-scope
        // `var<private>` globals (the WGSL analog of the SPIR-V Private-global
        // hoist): a runtime index addresses one materialization instead of a
        // per-occurrence `var<function>`. Populated while lowering bodies
        // above; emitted here, after the struct declarations its types may
        // reference. Sorted by name for deterministic output.
        let mut const_globals: Vec<_> = self.private_const_globals.iter().collect();
        const_globals.sort_by_key(|(_, (name, _))| name.clone());
        for (init, (name, ty)) in const_globals {
            writeln!(output, "var<private> {}: {} = {};", name, ty, init).unwrap();
            writeln!(output).unwrap();
        }

        // Compute push-constant blocks — compiled to `var<storage, read>`
        // bindings since WGSL has no push constants and its uniform
        // address space imposes 16-byte array stride alignment (an
        // `array<u32, N>` field would be rejected). Storage-read
        // bindings accept natural scalar/array strides. Each compute
        // entry with `push_constant_offset`-tagged inputs gets its own
        // struct and binding. Sorted by binding for determinism.
        let mut blocks: Vec<_> = self.pc_blocks.values().collect();
        blocks.sort_by_key(|b| (b.set, b.binding));
        for b in blocks {
            writeln!(output, "struct {} {{", b.struct_name).unwrap();
            for (_, field_name, ty_str) in &b.fields {
                writeln!(output, "    {}: {},", field_name, ty_str).unwrap();
            }
            writeln!(output, "}}").unwrap();
            writeln!(output).unwrap();
            writeln!(
                output,
                "@group({}) @binding({}) var<storage, read> {}: {};",
                b.set, b.binding, b.var_name, b.struct_name
            )
            .unwrap();
            writeln!(output).unwrap();
        }

        // Compiler-introduced storage bindings + entry-level storage-
        // backed I/O. WGSL needs these at module scope; dedupe by
        // (set, binding) and coalesce access modes so an (in, out) pair
        // on the same slot becomes `read_write`.
        let mut synth: LookupMap<BindingRef, (String, String, bool, bool)> = LookupMap::new();
        // Key → (elem_ty_str, module_name, has_read, has_write).
        let is_declared = |_: BindingRef| false;
        for entry in &self.program.entry_points {
            // Explicit compiler-inserted bindings (e.g. parallelize's
            // partial-sum buffer).
            for sb in &entry.storage_bindings {
                if is_declared(sb.binding) {
                    continue;
                }
                let ty_str = self.type_emitter.type_to_wgsl(&sb.elem_ty)?;
                let entry_ref = synth.entry(sb.binding).or_insert_with(|| {
                    let name = format!("_buf_{}_{}", sb.binding.set, sb.binding.binding);
                    (ty_str.clone(), name, false, false)
                });
                match sb.role {
                    crate::interface::StorageRole::Input => entry_ref.2 = true,
                    crate::interface::StorageRole::Output => entry_ref.3 = true,
                    crate::interface::StorageRole::Intermediate => {
                        entry_ref.2 = true;
                        entry_ref.3 = true;
                    }
                }
            }
            // Entry inputs marked with storage_binding — compute shader
            // runtime-sized array parameters. The element type is the
            // array element type; the WGSL binding holds the full
            // `array<T>`.
            for input in &entry.inputs {
                if let Some(br) = input.storage_binding {
                    if is_declared(br) {
                        continue;
                    }
                    let elem_ty = input
                        .ty
                        .elem_type()
                        .ok_or_else(|| {
                            crate::err_wgsl!("storage-bound input '{}' has no element type", input.name)
                        })?
                        .clone();
                    let ty_str = self.type_emitter.type_to_wgsl(&elem_ty)?;
                    let entry_ref = synth.entry(br).or_insert_with(|| {
                        let name = format!("_buf_{}_{}", br.set, br.binding);
                        (ty_str.clone(), name, false, false)
                    });
                    // Entry inputs are read by convention.
                    entry_ref.2 = true;
                    // A `#[storage(..., access=write/readwrite)]` input is
                    // written in place (e.g. a `scatter` destination); WGSL
                    // needs `read_write` for the Store.
                    if matches!(
                        input.storage_access,
                        Some(crate::interface::StorageAccess::WriteOnly)
                            | Some(crate::interface::StorageAccess::ReadWrite)
                    ) {
                        entry_ref.3 = true;
                    }
                }
            }
            // Entry outputs likewise. For scalar-valued compute outputs
            // (e.g. reduce → f32), the user-level type isn't an array
            // but the underlying binding still holds a runtime-sized
            // array of that scalar — the SOAC parallelize pass packs
            // the result into a single-element slot.
            for out in &entry.outputs {
                if let Some(br) = out.storage_binding {
                    if is_declared(br) {
                        continue;
                    }
                    // Array-shaped output (`[]T`) → elem is `T`; scalar / vec /
                    // struct-valued compute output (e.g. reduce → f32) packs
                    // into a single-element `[]T` binding where `out.ty` IS
                    // the elem. `array_elem` returns None for the latter.
                    let elem_ty = match crate::types::array_elem(&out.ty) {
                        Some(elem) => elem.clone(),
                        None => out.ty.clone(),
                    };
                    let ty_str = self.type_emitter.type_to_wgsl(&elem_ty)?;
                    let entry_ref = synth.entry(br).or_insert_with(|| {
                        let name = format!("_buf_{}_{}", br.set, br.binding);
                        (ty_str.clone(), name, false, false)
                    });
                    entry_ref.3 = true;
                }
            }
        }
        // Sort for determinism.
        let mut synth_sorted: Vec<_> = synth.into_iter().collect();
        synth_sorted.sort_by_key(|(br, _)| (br.set, br.binding));
        for (br, (elem_ty, name, has_in, has_out)) in synth_sorted {
            let (set, binding) = (br.set, br.binding);
            let access = match (has_in, has_out) {
                (true, true) => "read_write",
                (true, false) => "read",
                (false, true) => "read_write", // write-only + WGSL needs read_write for Store
                (false, false) => "read",
            };
            writeln!(
                output,
                "@group({}) @binding({}) var<storage, {}> {}: array<{}>;",
                set, binding, access, name, elem_ty
            )
            .unwrap();
            writeln!(output).unwrap();
        }

        // Workgroup-shared arrays (phase2 tree reduce): one module-scope
        // `var<workgroup> _wg_<id>: array<T, count>` per distinct id, found by
        // pre-scanning every entry body for `StorageView(Workgroup{id,count})`.
        // ids are globally unique (assigned by the phase2 synthesis).
        let mut wg_arrays: std::collections::BTreeMap<u32, (String, u32)> =
            std::collections::BTreeMap::new();
        for entry in &self.program.entry_points {
            for (_, inst) in entry.body.inner.insts.iter() {
                if let InstKind::Op {
                    tag: crate::op::OpTag::StorageView(crate::op::PureViewSource::Workgroup { id, count }),
                    ..
                } = &inst.data
                {
                    let result = inst.result.expect("StorageView(Workgroup) must have a result");
                    let view_ty = entry.body.get_value_type(result);
                    // Array-shaped workgroup view → elem; scalar / vec /
                    // struct-shaped (single-element reduce) → view IS the elem.
                    let elem_ty = match crate::types::array_elem(view_ty) {
                        Some(elem) => elem.clone(),
                        None => view_ty.clone(),
                    };
                    let elem_str = self.type_emitter.type_to_wgsl(&elem_ty)?;
                    wg_arrays.entry(*id).or_insert((elem_str, *count));
                }
            }
        }
        for (id, (elem_str, count)) in wg_arrays {
            writeln!(
                output,
                "var<workgroup> _wg_{}: array<{}, {}>;",
                id, elem_str, count
            )
            .unwrap();
            writeln!(output).unwrap();
        }

        // Entry-param `#[uniform(set, binding)]` declarations become
        // module-scope `var<uniform>` in WGSL. The variable's name is
        // the mangled form of the param name so it matches body
        // references. Dedupe by mangled name across entries — same
        // name in two entries must agree on (set, binding, type).
        let mut uniforms: LookupMap<String, (u32, u32, String)> = LookupMap::new();
        for entry in &self.program.entry_points {
            for input in &entry.inputs {
                if let Some(br) = input.uniform_binding {
                    let (set, binding) = (br.set, br.binding);
                    let ty_str = self.type_emitter.type_to_wgsl(&input.ty)?;
                    let key = self.mangle_tracked(&input.name)?;
                    if let Some(prev) = uniforms.get(&key) {
                        if *prev != (set, binding, ty_str.clone()) {
                            return Err(crate::err_wgsl!(
                                "uniform '{}' declared with conflicting (set, binding, type) across entries",
                                input.name
                            ));
                        }
                    }
                    uniforms.insert(key, (set, binding, ty_str));
                }
            }
        }
        let mut uniforms_sorted: Vec<_> = uniforms.into_iter().collect();
        uniforms_sorted.sort_by_key(|(_, (set, binding, _))| (*set, *binding));
        for (name, (set, binding, ty_str)) in uniforms_sorted {
            writeln!(
                output,
                "@group({}) @binding({}) var<uniform> {}: {};",
                set, binding, name, ty_str
            )
            .unwrap();
            writeln!(output).unwrap();
        }

        // Entry-param `#[texture]` / `#[sampler]` declarations become
        // module-scope handle vars (no address space). Same dedupe-by-name
        // discipline as uniforms. The WGSL type comes from the param type
        // (`texture_2d<f32>` / `sampler`).
        let mut handles: LookupMap<String, (u32, u32, String)> = LookupMap::new();
        for entry in &self.program.entry_points {
            for input in &entry.inputs {
                let set_binding = input.texture_binding.or(input.sampler_binding);
                if let Some(br) = set_binding {
                    let (set, binding) = (br.set, br.binding);
                    let ty_str = self.type_emitter.type_to_wgsl(&input.ty)?;
                    let key = self.mangle_tracked(&input.name)?;
                    if let Some(prev) = handles.get(&key) {
                        if *prev != (set, binding, ty_str.clone()) {
                            return Err(crate::err_wgsl!(
                                "texture/sampler '{}' declared with conflicting (set, binding, type) \
                                 across entries",
                                input.name
                            ));
                        }
                    }
                    handles.insert(key, (set, binding, ty_str));
                }
            }
        }
        let mut handles_sorted: Vec<_> = handles.into_iter().collect();
        handles_sorted.sort_by_key(|(_, (set, binding, _))| (*set, *binding));
        for (name, (set, binding, ty_str)) in handles_sorted {
            writeln!(
                output,
                "@group({}) @binding({}) var {}: {};",
                set, binding, name, ty_str
            )
            .unwrap();
            writeln!(output).unwrap();
        }

        // Storage-image bindings. WGSL renders the format and access
        // mode on the binding's type: `texture_storage_2d<format,
        // access>`. Resource identity is the binding, not an entry-local
        // parameter name, so shared bindings receive one canonical global.
        let mut storage_images: LookupMap<BindingRef, String> = LookupMap::new();
        for entry in &self.program.entry_points {
            for input in &entry.inputs {
                let Some((br, format, access, _size)) = input.storage_image_binding else {
                    continue;
                };
                let ty_str = format!(
                    "texture_storage_2d<{}, {}>",
                    wgsl_storage_image_format(format),
                    wgsl_storage_access(access),
                );
                if let Some(previous_ty) = storage_images.get(&br) {
                    if previous_ty != &ty_str {
                        return Err(crate::err_wgsl!(
                            "storage_image binding ({}, {}) has conflicting format/access across entries",
                            br.set,
                            br.binding
                        ));
                    }
                }
                storage_images.insert(br, ty_str);
            }
        }
        let mut storage_images_sorted: Vec<_> = storage_images.into_iter().collect();
        storage_images_sorted.sort_by_key(|(binding, _)| (binding.set, binding.binding));
        for (binding, ty_str) in storage_images_sorted {
            writeln!(
                output,
                "@group({}) @binding({}) var {}: {};",
                binding.set,
                binding.binding,
                storage_image_global_name(binding),
                ty_str
            )
            .unwrap();
            writeln!(output).unwrap();
        }

        output.push_str(&code);
        Ok(output)
    }

    fn lower_function(&mut self, func: &Function, output: &mut String) -> Result<()> {
        if func.linkage_name.is_some() {
            // Extern functions cross-reference another shader language; we
            // can't emit them from SSA. Skip — the Wyn frontend should have
            // warned already.
            return Ok(());
        }
        if self.lowered.contains(&func.name) {
            return Ok(());
        }
        self.lowered.insert(func.name.clone());

        let body = &func.body;
        let name = self.mangle_tracked(&func.name)?;
        let ret_ty = self.type_emitter.type_to_wgsl(&body.return_ty)?;
        // Skip params whose type is a view-array (`[?]T`). WGSL function
        // signatures can't take bare `array<T>` — they'd need
        // `ptr<storage, array<T>, read>`. In practice these params are
        // dead: buffer_specialize either rewrote them into explicit
        // (offset, len) tuples upstream, or the body was rewritten to
        // reach the buffer through its `@group @binding` global and
        // ignore the param entirely (e.g. a `map` lambda after
        // partial-evaluation). If the body *did* reference it, a
        // later name-lookup error at the reference site surfaces the
        // bug cleanly.
        let params: Result<Vec<String>> = body
            .params
            .iter()
            .filter(|(_, ty, _)| !is_view_array_ty(ty))
            .map(|(_, ty, pname)| {
                let ty_str = self.type_emitter.type_to_wgsl(ty)?;
                let pname = self.mangle_tracked(pname)?;
                Ok(format!("{}: {}", pname, ty_str))
            })
            .collect();
        let params = params?;

        writeln!(output, "fn {}({}) -> {} {{", name, params.join(", "), ret_ty).unwrap();
        self.indent += 1;

        let mut body_ctx = BodyLowerCtx::new(self, body, func.span);
        let result = body_ctx.lower(output)?;
        writeln!(output, "{}return {};", self.indent_str(), result).unwrap();

        self.indent -= 1;
        writeln!(output, "}}").unwrap();
        writeln!(output).unwrap();
        Ok(())
    }

    fn lower_entry_point(&mut self, entry: &EntryPoint, output: &mut String) -> Result<()> {
        let body = &entry.body;

        // Build the entry attribute (`@vertex`, `@fragment`, `@compute ...`).
        let entry_attr = match &entry.execution_model {
            ExecutionModel::Vertex => "@vertex".to_string(),
            ExecutionModel::Fragment => "@fragment".to_string(),
            ExecutionModel::Compute { local_size } => format!(
                "@compute @workgroup_size({}, {}, {})",
                local_size.0, local_size.1, local_size.2
            ),
        };
        let stage_is_fragment = matches!(entry.execution_model, ExecutionModel::Fragment);

        // Parameters: one per entry input. Each gets an attribute based on
        // its decoration (@builtin(...) or @location(N)). The SSA body's
        // params[i] shares the user-written name; we mangle for WGSL.
        //
        // When a builtin's WGSL-mandated type differs from Wyn's declared
        // type (e.g. vertex_index must be `u32`, but Wyn may declare it
        // `i32`), we emit the param under an internal name with the WGSL
        // type and queue a cast statement at the start of the body that
        // binds the user's name to the converted value.
        let mut param_strs: Vec<String> = Vec::new();
        // Each entry: (user_mangled_name, internal_name, user_ty_str).
        let mut builtin_casts: Vec<(String, String, String)> = Vec::new();
        // Build a push-constant block for this entry if any input carries
        // `push_constant_offset`. SPIR-V packs these into a PushConstant
        // storage-class struct; WGSL has no push constants, so we emit
        // a uniform block instead. Collected here so `lower_program`'s
        // module-scope pass can emit the struct + `var<uniform>` decl.
        let pc_inputs: Vec<(usize, &crate::ssa::types::EntryInput)> =
            entry.inputs.iter().enumerate().filter(|(_, inp)| inp.push_constant.is_some()).collect();
        let pc_block: Option<PcBlock> = if !pc_inputs.is_empty() {
            // Synthetic (set, binding) chosen to avoid colliding with
            // user-declared storage/uniform bindings: set = 1 (user
            // bindings conventionally sit at set 0), binding counts up
            // per entry that needs a block.
            let binding = self.pc_blocks.len() as u32;
            let struct_name = format!("_PcBlock{}", binding);
            let var_name = format!("_pc{}", binding);
            let mut fields: Vec<(usize, String, String)> = Vec::new();
            for (i, inp) in &pc_inputs {
                let raw_name =
                    body.params.get(*i).map(|(_, _, n)| n.clone()).unwrap_or_else(|| inp.name.clone());
                // Mangle through the same pass the rest of the backend
                // uses — raw user names may collide with WGSL reserved
                // words like `target` or `loop`.
                let field_name = self.mangle_tracked(&raw_name)?;
                let ty_str = self.type_emitter.type_to_wgsl(&inp.ty)?;
                fields.push((*i, field_name, ty_str));
            }
            Some(PcBlock {
                struct_name,
                var_name,
                set: 1,
                binding,
                fields,
            })
        } else {
            None
        };

        for (i, input) in entry.inputs.iter().enumerate() {
            // Storage-backed inputs (compute shader runtime-sized array
            // params) become module-scope bindings, not function params.
            if input.storage_binding.is_some() {
                continue;
            }
            // Push-constant inputs are routed through the synthesized
            // uniform block — no function parameter emitted.
            if input.push_constant.is_some() {
                continue;
            }
            // `#[uniform]`-attributed inputs become module-scope
            // `var<uniform>` declarations; the body references them by
            // name directly.
            if input.uniform_binding.is_some() {
                continue;
            }
            // `#[texture]` / `#[sampler]` / `#[storage_image]` inputs
            // become module-scope handle vars (emitted in `lower_program`);
            // the body references them by the same mangled name, so no
            // function param.
            if input.texture_binding.is_some()
                || input.sampler_binding.is_some()
                || input.storage_image_binding.is_some()
            {
                continue;
            }
            let param_name =
                body.params.get(i).map(|(_, _, n)| n.clone()).unwrap_or_else(|| input.name.clone());
            let mangled_name = self.mangle_tracked(&param_name)?;
            let user_ty_str = self.type_emitter.type_to_wgsl(&input.ty)?;
            let (attr, param_ty_str, internal_name) = match &input.decoration {
                Some(IoDecoration::BuiltIn(b)) => {
                    let wgsl_b = map_builtin_to_wgsl(b, stage_is_fragment).ok_or_else(|| {
                        crate::err_wgsl!(
                            "entry input {}: WGSL has no @builtin mapping for {:?}",
                            param_name,
                            b
                        )
                    })?;
                    let attr = format!("@builtin({}) ", wgsl_b);
                    match wgsl_builtin_type(b) {
                        Some(required) if required != user_ty_str => {
                            let internal = format!("_builtin_{}", wgsl_b);
                            builtin_casts.push((
                                mangled_name.clone(),
                                internal.clone(),
                                user_ty_str.clone(),
                            ));
                            (attr, required.to_string(), internal)
                        }
                        _ => (attr, user_ty_str.clone(), mangled_name.clone()),
                    }
                }
                Some(IoDecoration::Location(n)) => (
                    format!("@location({}) ", n),
                    user_ty_str.clone(),
                    mangled_name.clone(),
                ),
                None => (String::new(), user_ty_str.clone(), mangled_name.clone()),
            };
            param_strs.push(format!("{}{}: {}", attr, internal_name, param_ty_str));
        }

        // Return type: either a single output with a decoration, or a
        // generated struct for multi-output fragment shaders. Compute
        // shaders have no function-return value (outputs bind at module
        // scope via `@group/@binding`).
        //
        // Filter out storage-backed outputs before counting — they're
        // written via `Store` to module-scope bindings, not returned.
        // Indices (into `entry.outputs`) of the non-storage outputs, so
        // we can route `OutputPtr { index: N }` to the right struct field
        // for multi-output entries.
        let non_storage_outputs: Vec<(usize, &crate::ssa::types::EntryOutput)> =
            entry.outputs.iter().enumerate().filter(|(_, o)| o.storage_binding.is_none()).collect();
        // For multi-output: the generated struct name and the per-output
        // field mapping (orig_index → field_name), which pre-declares
        // `var _out_struct: VsOutN;` in the body prelude and routes
        // `OutputPtr` targets into its fields.
        let mut multi_output_struct: Option<(String, LookupMap<usize, String>)> = None;
        let (ret_type_str, is_compute_void) = match entry.execution_model {
            ExecutionModel::Compute { .. } => (String::new(), true),
            _ => {
                if non_storage_outputs.is_empty() {
                    return Err(crate::err_wgsl!(
                        "entry '{}' has no non-storage outputs but is not a compute shader",
                        entry.name
                    ));
                }
                if non_storage_outputs.len() == 1 {
                    let (_, out) = non_storage_outputs[0];
                    let ty_str = self.type_emitter.type_to_wgsl(&out.ty)?;
                    let attr = match &out.decoration {
                        Some(IoDecoration::BuiltIn(b)) => {
                            let wgsl_b = map_builtin_to_wgsl(b, stage_is_fragment).ok_or_else(|| {
                                crate::err_wgsl!("entry output: WGSL has no @builtin mapping for {:?}", b)
                            })?;
                            format!("@builtin({}) ", wgsl_b)
                        }
                        Some(IoDecoration::Location(n)) => format!("@location({}) ", n),
                        None => String::new(),
                    };
                    (format!("{}{}", attr, ty_str), false)
                } else {
                    // Multi-output: pack outputs into a generated struct.
                    // Each field carries its own `@builtin(...)` or
                    // `@location(N)` attribute (WGSL requires these on
                    // struct members, not on the return type).
                    let mut field_specs: Vec<(String, String)> = Vec::new();
                    let mut index_to_field: LookupMap<usize, String> = LookupMap::new();
                    for (orig_index, out) in &non_storage_outputs {
                        let ty_str = self.type_emitter.type_to_wgsl(&out.ty)?;
                        let attr = match &out.decoration {
                            Some(IoDecoration::BuiltIn(b)) => {
                                let wgsl_b =
                                    map_builtin_to_wgsl(b, stage_is_fragment).ok_or_else(|| {
                                        crate::err_wgsl!(
                                            "entry output: WGSL has no @builtin mapping for {:?}",
                                            b
                                        )
                                    })?;
                                format!("@builtin({}) ", wgsl_b)
                            }
                            Some(IoDecoration::Location(n)) => format!("@location({}) ", n),
                            None => {
                                return Err(crate::err_wgsl!(
                                    "entry '{}' output #{} has no decoration; multi-output \
                                     WGSL entries require `@builtin(...)` or `@location(N)` on \
                                     every field",
                                    entry.name,
                                    orig_index
                                ));
                            }
                        };
                        field_specs.push((attr, ty_str));
                    }
                    let struct_name = self.get_or_create_output_struct(field_specs);
                    // Field names (`f0`, `f1`, ...) are assigned in
                    // `get_or_create_output_struct` by struct-field
                    // position, which mirrors the order we pushed above
                    // (i.e. the order of `non_storage_outputs`).
                    for (pos, (orig_index, _)) in non_storage_outputs.iter().enumerate() {
                        index_to_field.insert(*orig_index, format!("_out_struct.f{}", pos));
                    }
                    multi_output_struct = Some((struct_name.clone(), index_to_field));
                    (struct_name, false)
                }
            }
        };

        // Compute shaders always get a synthetic
        // `@builtin(global_invocation_id) _wgsl_gid: vec3<u32>` parameter
        // so `_w_intrinsic_thread_id()` can read `_wgsl_gid.x`. Skip if
        // the user already declared one with `#[builtin(global_invocation_id)]`
        // — WGSL rejects duplicate `@builtin` declarations on the same
        // entry, and the intrinsic dispatch can read from the user's
        // param via the alias added below.
        self.wgsl_gid_alias = None;
        if matches!(entry.execution_model, ExecutionModel::Compute { .. }) {
            let user_gid = entry.inputs.iter().find(|i| {
                matches!(
                    &i.decoration,
                    Some(IoDecoration::BuiltIn(spirv::BuiltIn::GlobalInvocationId))
                )
            });
            if let Some(input) = user_gid {
                let mangled = self.mangle_tracked(&input.name)?;
                self.wgsl_gid_alias = Some(mangled);
            } else {
                param_strs.push("@builtin(global_invocation_id) _wgsl_gid: vec3<u32>".to_string());
            }
            // Likewise `_w_intrinsic_local_id()` → `_wgsl_lid.x`, the thread's
            // index within its workgroup (used by the workgroup-parallel
            // phase2 reduce for shared-memory addressing).
            param_strs.push("@builtin(local_invocation_id) _wgsl_lid: vec3<u32>".to_string());
            // Likewise for `_w_intrinsic_num_workgroups()` → `_wgsl_nwg.x`,
            // the runtime dispatch grid width the parallelize chunk math
            // divides the input by.
            param_strs.push("@builtin(num_workgroups) _wgsl_nwg: vec3<u32>".to_string());
        }

        // Entry-point names are kept verbatim so the pipeline descriptor's
        // `entry_point` strings match what's actually in the WGSL output.
        // Internal symbols still go through `mangle_tracked`; entries are
        // user-written and the parser's identifier rules already restrict
        // them to a subset of legal WGSL identifiers — we re-check here.
        let name = entry.name.clone();
        validate_wgsl_identifier(&name).map_err(|e| {
            crate::err_wgsl!(
                "entry-point name '{}' is not a legal WGSL identifier: {}",
                name,
                e
            )
        })?;
        if let Some(prev) = self.mangled_names.insert(name.clone(), name.clone()) {
            if prev != name {
                return Err(crate::err_wgsl!(
                    "entry-point name '{}' collides with mangled symbol from '{}'",
                    name,
                    prev
                ));
            }
        }
        writeln!(output, "{}", entry_attr).unwrap();
        if is_compute_void {
            writeln!(output, "fn {}({}) {{", name, param_strs.join(", ")).unwrap();
        } else {
            writeln!(
                output,
                "fn {}({}) -> {} {{",
                name,
                param_strs.join(", "),
                ret_type_str
            )
            .unwrap();
        }

        self.indent += 1;

        // Emit builtin-type casts first — `let <user_name>: <user_ty> = <user_ty>(<internal>);`
        // — so the body sees the user-declared name bound to the cast.
        // Later code inside the body refers to the user's mangled name
        // via its ValueId in value_map.
        for (user_name, internal_name, user_ty_str) in &builtin_casts {
            writeln!(
                output,
                "{}let {}: {} = {}({});",
                self.indent_str(),
                user_name,
                user_ty_str,
                user_ty_str,
                internal_name
            )
            .unwrap();
        }

        // Pre-declare `var<function>` locals for each location-decorated
        // output so the body's `OutputPtr` can alias into them. Skip
        // storage-bound outputs — those are written via direct Store
        // to module-scope bindings, not returned through the function.
        // Multi-output entries share a single `_out_struct` var and
        // alias each `OutputPtr { index }` into a distinct field.
        let mut output_locals: Vec<(usize, String, String)> = Vec::new(); // (index, name, wgsl_type)
        if !is_compute_void {
            match &multi_output_struct {
                Some((struct_name, _)) => {
                    writeln!(output, "{}var _out_struct: {};", self.indent_str(), struct_name).unwrap();
                }
                None => {
                    for (i, out) in entry.outputs.iter().enumerate() {
                        if out.storage_binding.is_some() {
                            continue;
                        }
                        let ty_str = self.type_emitter.type_to_wgsl(&out.ty)?;
                        let name = format!("_out{}", i);
                        writeln!(output, "{}var {}: {};", self.indent_str(), name, ty_str).unwrap();
                        output_locals.push((i, name, ty_str));
                    }
                }
            }
        }

        let mut body_ctx = BodyLowerCtx::new(self, body, entry.span);
        if let Some((_, index_to_field)) = &multi_output_struct {
            body_ctx.output_target_names = index_to_field.clone();
        }
        // Pre-seed `value_map` for push-constant inputs. Push-constant
        // fields don't appear as function parameters in the emitted
        // WGSL; the body refers to them as `<pc_var>.<field>` instead.
        if let Some(pc) = &pc_block {
            for (input_idx, field_name, _) in &pc.fields {
                if let Some((value_id, _, _)) = body.params.get(*input_idx) {
                    let expr = format!("{}.{}", pc.var_name, field_name);
                    body_ctx.value_map.insert(*value_id, ValueBinding::Alias(expr));
                }
            }
        }
        let result = body_ctx.lower(output)?;
        let uses_output_ptrs = body_ctx.uses_output_ptrs;

        if !is_compute_void {
            if uses_output_ptrs {
                // OutputPtr-based return: single output returns `_out0`;
                // multi-output returns the packed `_out_struct`.
                if multi_output_struct.is_some() {
                    writeln!(output, "{}return _out_struct;", self.indent_str()).unwrap();
                } else if output_locals.len() == 1 {
                    writeln!(output, "{}return _out0;", self.indent_str()).unwrap();
                } else {
                    return Err(crate::err_wgsl!(
                        "multi-output entries with OutputPtr not yet implemented (entry '{}')",
                        entry.name
                    ));
                }
            } else {
                writeln!(output, "{}return {};", self.indent_str(), result).unwrap();
            }
        }
        self.indent -= 1;
        writeln!(output, "}}").unwrap();
        writeln!(output).unwrap();
        // Register the PC block (after emitting the fn body so state
        // additions can't accidentally leak into the fn emission).
        if let Some(pc) = pc_block {
            self.pc_blocks.insert(entry.name.clone(), pc);
        }
        Ok(())
    }
}

/// Map a SPIR-V `BuiltIn` decoration to its WGSL `@builtin(...)` spelling.
/// Returns `None` for builtins WGSL doesn't expose (caller should error).
fn map_builtin_to_wgsl(b: &spirv::BuiltIn, _stage_is_fragment: bool) -> Option<&'static str> {
    Some(match b {
        spirv::BuiltIn::Position => "position",
        spirv::BuiltIn::FragCoord => "position",
        spirv::BuiltIn::VertexIndex => "vertex_index",
        spirv::BuiltIn::InstanceIndex => "instance_index",
        spirv::BuiltIn::FrontFacing => "front_facing",
        spirv::BuiltIn::FragDepth => "frag_depth",
        spirv::BuiltIn::PointSize => return None, // no WGSL equivalent
        spirv::BuiltIn::GlobalInvocationId => "global_invocation_id",
        spirv::BuiltIn::LocalInvocationId => "local_invocation_id",
        spirv::BuiltIn::LocalInvocationIndex => "local_invocation_index",
        spirv::BuiltIn::WorkgroupId => "workgroup_id",
        spirv::BuiltIn::NumWorkgroups => "num_workgroups",
        _ => return None,
    })
}

/// The WGSL-mandated type for a given `@builtin(...)` parameter. WGSL
/// fixes these by spec (vertex_index is always `u32`, position is always
/// `vec4<f32>`, etc.). Wyn's user-level types may differ (e.g., `i32`
/// for vertex_index); the entry-point wrapper inserts a cast when they
/// don't match.
fn wgsl_builtin_type(b: &spirv::BuiltIn) -> Option<&'static str> {
    Some(match b {
        spirv::BuiltIn::Position | spirv::BuiltIn::FragCoord => "vec4<f32>",
        spirv::BuiltIn::VertexIndex | spirv::BuiltIn::InstanceIndex => "u32",
        spirv::BuiltIn::LocalInvocationIndex => "u32",
        spirv::BuiltIn::FrontFacing => "bool",
        spirv::BuiltIn::FragDepth => "f32",
        spirv::BuiltIn::GlobalInvocationId
        | spirv::BuiltIn::LocalInvocationId
        | spirv::BuiltIn::WorkgroupId
        | spirv::BuiltIn::NumWorkgroups => "vec3<u32>",
        _ => return None,
    })
}

// -----------------------------------------------------------------------------
// Body-level lowering
// -----------------------------------------------------------------------------

/// Value-category tag on each ValueId's emitted WGSL expression.
/// `Alias` is safe to substitute into any rvalue context. `Place`
/// names an lvalue whose read is side-effectful or expensive (a
/// storage-buffer element, `buf[idx]`) and must be materialized on
/// Load; Store writes through either variant directly.
#[derive(Clone)]
enum ValueBinding {
    /// Either a local `let`/`var` name (`v3_1`) or a buffer identifier
    /// that `ViewIndex` dereferences into a place expression.
    Alias(String),
}

impl ValueBinding {
    fn expr(&self) -> &str {
        match self {
            ValueBinding::Alias(s) => s,
        }
    }
}

struct BodyLowerCtx<'a, 'b> {
    ctx: &'a mut LowerCtx<'b>,
    body: &'a FuncBody,
    /// Emitted WGSL expression (or var name) per ValueId, tagged with
    /// its value category (rvalue-safe alias vs. lvalue place).
    value_map: LookupMap<ValueId, ValueBinding>,
    /// Set of names declared with `let`/`var` in the current scope.
    declared: LookupSet<String>,
    /// Workgroup view name (`_wg_<id>`) keyed by `StorageView` result ValueId.
    /// Storage views recover their buffer name from the type's region; only
    /// workgroup views (whose `_wg_<id>` isn't in any type) need this.
    workgroup_view_name: LookupMap<ValueId, String>,
    /// WGSL place expression per `PlaceId` — output variable name,
    /// `_alloca_N` for function-local `Alloca`s, or `buf[offset+idx]`
    /// for `ViewIndex`. Consumed by `Load` / `Store` in `emit_nodes`.
    place_targets: LookupMap<crate::ssa::types::PlaceId, String>,
    /// Set to `true` if at least one `OutputSlot` was lowered; the entry
    /// wrapper then returns the declared `_out0` (or builds a return
    /// struct for multi-output) instead of emitting a `return <expr>;`
    /// from the body's terminator result.
    uses_output_ptrs: bool,
    /// Override for the name each `OutputSlot { index }` resolves to. If
    /// an index is present here, that name is used as the store target;
    /// otherwise the default `_out{index}` is used. Populated by
    /// `lower_entry_point` when it's packing outputs into a struct —
    /// then the name is `_out_struct.f{index}`.
    output_target_names: LookupMap<usize, String>,
    /// Span of the instruction currently being lowered.
    current_span: Option<Span>,
    func_span: Span,
}

impl<'a, 'b> BodyLowerCtx<'a, 'b> {
    fn new(ctx: &'a mut LowerCtx<'b>, body: &'a FuncBody, func_span: Span) -> Self {
        Self {
            ctx,
            body,
            value_map: LookupMap::new(),
            declared: LookupSet::new(),
            workgroup_view_name: LookupMap::new(),
            place_targets: LookupMap::new(),
            uses_output_ptrs: false,
            output_target_names: LookupMap::new(),
            current_span: None,
            func_span,
        }
    }

    /// Resolve a ValueRef to a compile-time integer, if possible. Returns
    /// None for runtime values. Used by storage-intrinsic dispatch where
    /// `set` and `binding` must be compile-time constants.
    fn resolve_const_u32(&self, v: ValueRef) -> Option<u32> {
        match v {
            ValueRef::Const(crate::ssa::types::ConstantValue::I32(n)) => Some(n as u32),
            ValueRef::Const(crate::ssa::types::ConstantValue::U32(n)) => Some(n),
            ValueRef::Ssa(id) => {
                // Walk the body's insts looking for one whose result is
                // `id` and whose data is a literal Int. Rare path —
                // called only for `_w_intrinsic_storage_len`'s set and
                // binding args, which monomorphization folds to constants.
                for (_, inst) in self.body.inner.insts.iter() {
                    if inst.result == Some(id) {
                        if let InstKind::Op {
                            tag: crate::op::OpTag::Int(s) | crate::op::OpTag::Uint(s),
                            ..
                        } = &inst.data
                        {
                            return s.parse::<u32>().ok();
                        }
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// True iff `v` is a compile-time constant: a numeric/bool literal, or a
    /// composite (array / vector / tuple / matrix) whose elements are all
    /// recursively constant. Such a value can be promoted to a module-scope
    /// `var<private>` whose initializer is a WGSL const-expression.
    fn is_const_value(&self, v: ValueRef) -> bool {
        use crate::op::OpTag;
        let id = match v {
            ValueRef::Const(_) => return true,
            ValueRef::Ssa(id) => id,
        };
        let Some(val) = self.body.inner.values.get(id) else {
            return false;
        };
        let inst_id = match val.def {
            crate::ssa::framework::ValueDef::Inst { inst } => inst,
            _ => return false,
        };
        let Some(inst) = self.body.inner.insts.get(inst_id) else {
            return false;
        };
        match &inst.data {
            InstKind::Op { tag, operands } => match tag {
                OpTag::Int(_) | OpTag::Uint(_) | OpTag::Float(_) | OpTag::Bool(_) | OpTag::Unit => true,
                OpTag::Tuple(_) | OpTag::Vector(_) | OpTag::ArrayLit(_) | OpTag::Matrix { .. } => {
                    operands.iter().all(|o| self.is_const_value(*o))
                }
                _ => false,
            },
            _ => false,
        }
    }

    /// Emit a constant value as a fully-inlined WGSL const-expression
    /// (literals + `vecN<T>(…)` / `array<T,N>(…)` / struct constructors),
    /// independent of the per-occurrence `var` bindings the body otherwise
    /// uses. Required for a module-scope `var<private>` initializer, which
    /// must be a const-expression and can't name function-local bindings.
    /// Caller guarantees `is_const_value(v)`.
    fn const_expr_of(&mut self, v: ValueRef) -> Result<String> {
        use crate::op::OpTag;
        let id = match v {
            ValueRef::Const(c) => return self.format_constant(&c),
            ValueRef::Ssa(id) => id,
        };
        let result_ty = self.body.get_value_type(id).clone();
        let val = self
            .body
            .inner
            .values
            .get(id)
            .ok_or_else(|| crate::err_wgsl_at!(self.blame_span(), "const hoist: value not found"))?;
        let inst_id = match val.def {
            crate::ssa::framework::ValueDef::Inst { inst } => inst,
            _ => {
                return Err(crate::err_wgsl_at!(
                    self.blame_span(),
                    "const hoist: value has no instruction"
                ))
            }
        };
        let (tag, operands) = match self.body.inner.insts.get(inst_id).map(|i| &i.data) {
            Some(InstKind::Op { tag, operands }) => (tag.clone(), operands.clone()),
            _ => return Err(crate::err_wgsl_at!(self.blame_span(), "const hoist: not an Op")),
        };
        match &tag {
            OpTag::Int(s) | OpTag::Uint(s) => {
                if matches!(result_ty, PolyType::Constructed(TypeName::UInt(32), _)) {
                    Ok(format!("{}u", s))
                } else {
                    Ok(format!("{}i", s))
                }
            }
            OpTag::Float(s) => {
                if s.contains('.') || s.contains('e') || s.contains('E') {
                    Ok(format!("{}f", s))
                } else {
                    Ok(format!("{}.0f", s))
                }
            }
            OpTag::Bool(b) => Ok((if *b { "true" } else { "false" }).to_string()),
            OpTag::Vector(_) | OpTag::ArrayLit(_) | OpTag::Tuple(_) => {
                let wgsl_ty = self.ctx.type_emitter.type_to_wgsl(&result_ty)?;
                let parts: Result<Vec<_>> = operands.iter().map(|o| self.const_expr_of(*o)).collect();
                Ok(format!("{}({})", wgsl_ty, parts?.join(", ")))
            }
            _ => Err(crate::err_wgsl_at!(
                self.blame_span(),
                "const hoist: unsupported op"
            )),
        }
    }

    /// Resolve a storage binding (set, binding) to its module-scope name.
    /// Uses the synthesized `_buf_{set}_{binding}` naming used for
    /// compiler-introduced compute-entry bindings.
    /// The WGSL buffer identifier a view value reads. A storage view's
    /// descriptor is the concrete `Region(set, binding)` in its type, so the
    /// name comes from there — authoritative regardless of how the view was
    /// derived (slice, block param, call). A workgroup view's type is
    /// `NoRegion`; its `_wg_<id>` name isn't in any type, so it is recovered
    /// from the `ViewHandle` set when the view was created.
    fn view_buffer_name(&self, view_ssa: ValueId) -> Result<String> {
        match crate::types::array_view_region(self.body.get_value_type(view_ssa)) {
            Some(br) => self.storage_name(br.set, br.binding),
            None => self.workgroup_view_name.get(&view_ssa).cloned().ok_or_else(|| {
                crate::err_wgsl_at!(
                    self.blame_span(),
                    "view {:?} has neither a concrete buffer region nor a workgroup name",
                    view_ssa
                )
            }),
        }
    }

    fn storage_name(&self, set: u32, binding: u32) -> Result<String> {
        let br = BindingRef::new(set, binding);
        for entry in &self.ctx.program.entry_points {
            if entry.storage_bindings.iter().any(|sb| sb.binding == br)
                || entry.inputs.iter().any(|i| i.storage_binding == Some(br))
                || entry.outputs.iter().any(|o| o.storage_binding == Some(br))
            {
                return Ok(format!("_buf_{}_{}", set, binding));
            }
        }
        Err(crate::err_wgsl_at!(
            self.blame_span(),
            "no storage binding at (set={}, binding={})",
            set,
            binding
        ))
    }

    fn storage_image_name(&self, binding: BindingRef) -> Result<String> {
        let declared =
            self.ctx.program.entry_points.iter().flat_map(|entry| &entry.inputs).any(|input| {
                input.storage_image_binding.is_some_and(|(candidate, ..)| candidate == binding)
            });
        if !declared {
            return Err(crate::err_wgsl_at!(
                self.blame_span(),
                "no storage-image binding at (set={}, binding={})",
                binding.set,
                binding.binding
            ));
        }
        Ok(storage_image_global_name(binding))
    }

    fn blame_span(&self) -> Span {
        self.current_span.unwrap_or(self.func_span)
    }

    /// Resolve a ValueRef to its WGSL expression text. Constants inline;
    /// SSA values look up the `value_map`.
    fn get_value(&self, v: ValueRef) -> Result<String> {
        match v {
            ValueRef::Ssa(id) => self.get_value_ref(id),
            ValueRef::Const(c) => self.format_constant(&c),
        }
    }

    fn get_value_ref(&self, id: ValueId) -> Result<String> {
        self.binding(id).map(|b| b.expr().to_string())
    }

    /// Resolve `v` to a WGSL expression, inserting an explicit scalar
    /// cast if its type doesn't already match `result_ty`. Only coerces
    /// between the scalar integer / float types — vectors, matrices,
    /// and structs pass through unchanged (a non-scalar mismatch is a
    /// real error, not a coercion target).
    fn coerce_operand_to_result_ty(
        &mut self,
        v: ValueRef,
        result_ty: Option<&polytype::Type<TypeName>>,
    ) -> Result<String> {
        let expr = self.get_value(v)?;
        let (Some(result_ty), ValueRef::Ssa(id)) = (result_ty, v) else {
            return Ok(expr);
        };
        let operand_ty = self.body.get_value_type(id);
        if operand_ty == result_ty {
            return Ok(expr);
        }
        let both_scalar = matches!(
            result_ty,
            polytype::Type::Constructed(TypeName::Int(_) | TypeName::UInt(_) | TypeName::Float(_), _)
        ) && matches!(
            operand_ty,
            polytype::Type::Constructed(TypeName::Int(_) | TypeName::UInt(_) | TypeName::Float(_), _)
        );
        if !both_scalar {
            return Ok(expr);
        }
        let cast_ty = self.ctx.type_emitter.type_to_wgsl(result_ty)?;
        Ok(format!("{}({})", cast_ty, expr))
    }

    fn binding(&self, id: ValueId) -> Result<&ValueBinding> {
        self.value_map.get(&id).ok_or_else(|| {
            crate::err_wgsl_at!(self.blame_span(), "value {:?} not bound in WGSL value_map", id)
        })
    }

    fn format_constant(&self, c: &crate::ssa::types::ConstantValue) -> Result<String> {
        use crate::ssa::types::ConstantValue;
        Ok(match c {
            ConstantValue::I32(v) => format!("{}i", v),
            ConstantValue::U32(v) => format!("{}u", v),
            ConstantValue::F32(bits) => {
                let v = f32::from_bits(*bits);
                if v.is_nan() || v.is_infinite() {
                    return Err(crate::err_wgsl_at!(
                        self.blame_span(),
                        "WGSL does not support NaN/Infinity constants: {}",
                        v
                    ));
                }
                let s = format!("{}", v);
                if s.contains('.') || s.contains('e') || s.contains('E') {
                    format!("{}f", s)
                } else {
                    format!("{}.0f", s)
                }
            }
            ConstantValue::Bool(b) => (if *b { "true" } else { "false" }).to_string(),
        })
    }

    fn lower(&mut self, output: &mut String) -> Result<String> {
        // Seed parameter names into value_map. Callers (entry-point
        // lowering) may have pre-inserted entries for params routed
        // through module-scope bindings (e.g. push-constant inputs →
        // `<block_var>.<field>`); skip those so the mangled name
        // doesn't clobber the override.
        for (value_id, _, name) in &self.body.params {
            if self.value_map.contains_key(value_id) {
                continue;
            }
            let mangled = self.ctx.mangle_tracked(name)?;
            self.value_map.insert(*value_id, ValueBinding::Alias(mangled.clone()));
            self.declared.insert(mangled);
        }

        // Structured walk via the shared structurize pass.
        let nodes = crate::structured::structurize(self.body);
        self.emit_nodes(&nodes, output)
    }

    fn emit_nodes(&mut self, nodes: &[crate::structured::Node], output: &mut String) -> Result<String> {
        use crate::structured::Node;
        let mut result_var = String::new();

        for node in nodes {
            match node {
                Node::Inst(inst_id) => {
                    let inst = self.body.get_inst(*inst_id);
                    self.current_span = inst.span;

                    // Memory-style operations emit statements (not
                    // expressions that bind into `let`), so we handle
                    // them at the emit_nodes level.
                    match &inst.data {
                        // Alloca: `var<function> x: T;` — register the
                        // local var's name as the place expression.
                        InstKind::Alloca { elem_ty, result } => {
                            let ty = self.ctx.type_emitter.type_to_wgsl(elem_ty)?;
                            let var = wgsl_place(*result);
                            writeln!(output, "{}var {}: {};", self.ctx.indent_str(), var, ty).unwrap();
                            self.declared.insert(var.clone());
                            self.place_targets.insert(*result, var);
                            continue;
                        }

                        // OutputSlot: bind the place to the entry's
                        // output variable name (`_out0` or struct-field
                        // override).
                        InstKind::OutputSlot { index, result } => {
                            let out_name = self
                                .output_target_names
                                .get(index)
                                .cloned()
                                .unwrap_or_else(|| format!("_out{}", index));
                            self.uses_output_ptrs = true;
                            self.place_targets.insert(*result, out_name);
                            continue;
                        }

                        // PlaceIndex: `{parent}[idx]` — index into another
                        // place's expression. Addresses one element of an
                        // Alloca'd function-local array without going
                        // through a whole-array Load.
                        InstKind::PlaceIndex { place, index, result } => {
                            let parent = self.place_targets.get(place).cloned().ok_or_else(|| {
                                crate::err_wgsl_at!(
                                    self.blame_span(),
                                    "PlaceIndex: parent place {:?} has no target expression",
                                    place
                                )
                            })?;
                            let idx = self.get_value(*index)?;
                            let expr = format!("{}[{}]", parent, idx);
                            self.place_targets.insert(*result, expr);
                            continue;
                        }

                        // ViewIndex: `buf[base_offset + idx]`.
                        InstKind::ViewIndex { view, index, result } => {
                            let view_id = view.as_ssa().ok_or_else(|| {
                                crate::err_wgsl_at!(
                                    self.blame_span(),
                                    "ViewIndex: view operand must be an SSA value"
                                )
                            })?;
                            // Buffer name from the type's region; offset is the
                            // `.x` of the view's `vec2<u32>` value.
                            let buffer_name = self.view_buffer_name(view_id)?;
                            let view_val = self.get_value(*view)?;
                            let idx = self.get_value(*index)?;
                            let expr = format!("{}[(i32(({}).x)) + (i32({}))]", buffer_name, view_val, idx);
                            self.place_targets.insert(*result, expr);
                            continue;
                        }

                        // Load: materialize the place's current value
                        // into a cached `var` so subsequent uses go
                        // through a stable binding.
                        InstKind::Load { place } => {
                            let result_id = inst.result.ok_or_else(|| {
                                crate::err_wgsl_at!(self.blame_span(), "Load must have a result")
                            })?;
                            let target = self.place_targets.get(place).cloned().ok_or_else(|| {
                                crate::err_wgsl_at!(
                                    self.blame_span(),
                                    "Load: place {:?} has no target expression",
                                    place
                                )
                            })?;
                            let ty =
                                self.ctx.type_emitter.type_to_wgsl(self.body.get_value_type(result_id))?;
                            let var = wgsl_var(result_id);
                            writeln!(
                                output,
                                "{}var {}: {} = {};",
                                self.ctx.indent_str(),
                                var,
                                ty,
                                target
                            )
                            .unwrap();
                            self.declared.insert(var.clone());
                            self.value_map.insert(result_id, ValueBinding::Alias(var));
                            continue;
                        }

                        // Store: `target = value;` where target is the
                        // place's WGSL expression (output var name,
                        // alloca var, or `buf[idx]`).
                        InstKind::Store { place, value } => {
                            let target = self.place_targets.get(place).cloned().ok_or_else(|| {
                                crate::err_wgsl_at!(
                                    self.blame_span(),
                                    "Store: place {:?} has no target expression",
                                    place
                                )
                            })?;
                            let val = self.get_value(*value)?;
                            writeln!(output, "{}{} = {};", self.ctx.indent_str(), target, val).unwrap();
                            continue;
                        }

                        InstKind::ControlBarrier => {
                            writeln!(output, "{}workgroupBarrier();", self.ctx.indent_str()).unwrap();
                            continue;
                        }

                        // `_w_intrinsic_uninit()` returns an
                        // uninitialized composite. In WGSL that's just a
                        // `var<function> x: T;` — no initializer, no
                        // function call.
                        InstKind::Op {
                            tag: crate::op::OpTag::Intrinsic { id, .. },
                            ..
                        } if *id == catalog().known().uninit => {
                            let result_id = inst.result.ok_or_else(|| {
                                crate::err_wgsl_at!(
                                    self.blame_span(),
                                    "_w_intrinsic_uninit must have a result"
                                )
                            })?;
                            let ty =
                                self.ctx.type_emitter.type_to_wgsl(self.body.get_value_type(result_id))?;
                            let var = wgsl_var(result_id);
                            writeln!(output, "{}var {}: {};", self.ctx.indent_str(), var, ty).unwrap();
                            self.declared.insert(var.clone());
                            self.value_map.insert(result_id, ValueBinding::Alias(var));
                            continue;
                        }

                        // `_w_intrinsic_array_with_inplace(arr, i, v)`:
                        // alias the result to the source array and emit
                        // `arr[i] = v;`. SOAC-generated output arrays
                        // are loop-carried, so mutating the live copy
                        // is always safe (previous iteration's value
                        // dies on the back-edge phi).
                        //
                        // `_w_intrinsic_array_with(arr, i, v)`:
                        // functional update. Declare a fresh `var` copy
                        // of the source, then patch the one element.
                        InstKind::Op {
                            tag: crate::op::OpTag::Intrinsic { id, .. },
                            operands,
                        } if *id == catalog().known().array_with_in_place
                            || *id == catalog().known().array_with =>
                        {
                            let known = catalog().known();
                            let is_inplace = *id == known.array_with_in_place;
                            let func_name = by_id(*id).dispatch_name();
                            if operands.len() != 3 {
                                return Err(crate::err_wgsl_at!(
                                    self.blame_span(),
                                    "{} expects 3 args, got {}",
                                    func_name,
                                    operands.len()
                                ));
                            }
                            let arr_src = self.get_value(operands[0])?;
                            let idx = self.get_value(operands[1])?;
                            let val = self.get_value(operands[2])?;
                            let result_id = inst.result.ok_or_else(|| {
                                crate::err_wgsl_at!(self.blame_span(), "{} must have a result", func_name)
                            })?;
                            if is_inplace {
                                writeln!(
                                    output,
                                    "{}{}[{}] = {};",
                                    self.ctx.indent_str(),
                                    arr_src,
                                    idx,
                                    val
                                )
                                .unwrap();
                                // Alias-only: the result is the
                                // (now-mutated) source array.
                                self.value_map.insert(result_id, ValueBinding::Alias(arr_src));
                            } else {
                                let ty = self
                                    .ctx
                                    .type_emitter
                                    .type_to_wgsl(self.body.get_value_type(result_id))?;
                                let var = wgsl_var(result_id);
                                writeln!(
                                    output,
                                    "{}var {}: {} = {};",
                                    self.ctx.indent_str(),
                                    var,
                                    ty,
                                    arr_src
                                )
                                .unwrap();
                                writeln!(output, "{}{}[{}] = {};", self.ctx.indent_str(), var, idx, val)
                                    .unwrap();
                                self.declared.insert(var.clone());
                                self.value_map.insert(result_id, ValueBinding::Alias(var));
                            }
                            continue;
                        }

                        // Materialize: `var<function> x: T = expr;` so
                        // the value becomes subscriptable via
                        // DynamicExtract's `x[i]`. WGSL forbids dynamic
                        // indexing of `let`-bound values, so this must
                        // be a `var`. A *constant* array is hoisted once
                        // to a shared module-scope `var<private>` instead
                        // of being rebuilt per occurrence.
                        InstKind::Op {
                            tag: crate::op::OpTag::Materialize,
                            operands,
                        } => {
                            let result_id = inst.result.ok_or_else(|| {
                                crate::err_wgsl_at!(self.blame_span(), "Materialize must have a result")
                            })?;
                            let ty =
                                self.ctx.type_emitter.type_to_wgsl(self.body.get_value_type(result_id))?;
                            if self.is_const_value(operands[0]) {
                                // Promote to a deduped module-scope `var<private>`
                                // and alias the result to it; the runtime index then
                                // addresses one shared materialization. `var<private>`
                                // (a reference) is dynamically indexable, unlike a
                                // `const` value. The initializer is the fully-inlined
                                // const-expression, which also serves as the value-
                                // based dedup key (equal arrays → one global).
                                let init = self.const_expr_of(operands[0])?;
                                let existing =
                                    self.ctx.private_const_globals.get(&init).map(|(name, _)| name.clone());
                                let name = match existing {
                                    Some(name) => name,
                                    None => {
                                        let name = format!(
                                            "_const_global_{}",
                                            self.ctx.private_const_globals.len()
                                        );
                                        self.ctx
                                            .private_const_globals
                                            .insert(init.clone(), (name.clone(), ty));
                                        name
                                    }
                                };
                                self.value_map.insert(result_id, ValueBinding::Alias(name));
                            } else {
                                let var = wgsl_var(result_id);
                                let val = self.get_value(operands[0])?;
                                writeln!(output, "{}var {}: {} = {};", self.ctx.indent_str(), var, ty, val)
                                    .unwrap();
                                self.declared.insert(var.clone());
                                self.value_map.insert(result_id, ValueBinding::Alias(var));
                            }
                            continue;
                        }

                        _ => {}
                    }

                    let expr = self.lower_inst(inst)?;
                    if let Some(result) = inst.result {
                        // StorageView: the result is a handle value whose
                        // name captures the buffer identity; alias it so
                        // later ViewIndex lookups can resolve via
                        // `view_handles`.
                        if matches!(
                            inst.data,
                            InstKind::Op {
                                tag: crate::op::OpTag::StorageView(_),
                                ..
                            }
                        ) {
                            self.value_map.insert(result, ValueBinding::Alias(expr));
                            continue;
                        }
                        // Unit / SideEffect results have no usable value
                        // — emit the expression as a bare statement
                        // (`textureStore(...);`) rather than a
                        // `var name: type = expr;` that WGSL would
                        // reject (the comment-typed `var v: /* unit */`
                        // shape fails to parse).
                        let result_ty = self.body.get_value_type(result);
                        if matches!(
                            result_ty,
                            polytype::Type::Constructed(TypeName::Unit, _)
                                | polytype::Type::Constructed(TypeName::SideEffect, _)
                        ) {
                            writeln!(output, "{}{};", self.ctx.indent_str(), expr).unwrap();
                            continue;
                        }
                        let var = wgsl_var(result);
                        let ty = self.ctx.type_emitter.type_to_wgsl(result_ty)?;
                        writeln!(output, "{}var {}: {} = {};", self.ctx.indent_str(), var, ty, expr)
                            .unwrap();
                        self.declared.insert(var.clone());
                        self.value_map.insert(result, ValueBinding::Alias(var));
                    }
                }

                Node::Assign { target, value } => {
                    let val = self.get_value_ref(*value)?;
                    let var = wgsl_var(*target);
                    if self.declared.contains(&var) {
                        writeln!(output, "{}{} = {};", self.ctx.indent_str(), var, val).unwrap();
                    } else {
                        let ty = self.ctx.type_emitter.type_to_wgsl(self.body.get_value_type(*target))?;
                        writeln!(output, "{}var {}: {} = {};", self.ctx.indent_str(), var, ty, val)
                            .unwrap();
                        self.declared.insert(var.clone());
                    }
                    self.value_map.insert(*target, ValueBinding::Alias(var));
                }

                Node::If {
                    cond,
                    then_body,
                    then_args,
                    else_body,
                    else_args,
                    merge_params,
                } => {
                    // Pre-declare merge params as `var` so both branches can
                    // assign into them.
                    for param in merge_params {
                        let var = wgsl_var(*param);
                        if !self.declared.contains(&var) {
                            let ty =
                                self.ctx.type_emitter.type_to_wgsl(self.body.get_value_type(*param))?;
                            writeln!(output, "{}var {}: {};", self.ctx.indent_str(), var, ty).unwrap();
                            self.declared.insert(var.clone());
                        }
                        self.value_map.insert(*param, ValueBinding::Alias(var));
                    }

                    let cond_val = self.get_value_ref(*cond)?;
                    writeln!(output, "{}if {} {{", self.ctx.indent_str(), cond_val).unwrap();
                    self.ctx.indent += 1;
                    self.emit_nodes(then_body, output)?;
                    for (param, arg) in merge_params.iter().zip(then_args.iter()) {
                        let arg_val = self.get_value_ref(*arg)?;
                        let var = wgsl_var(*param);
                        writeln!(output, "{}{} = {};", self.ctx.indent_str(), var, arg_val).unwrap();
                    }
                    self.ctx.indent -= 1;
                    writeln!(output, "{}}} else {{", self.ctx.indent_str()).unwrap();
                    self.ctx.indent += 1;
                    self.emit_nodes(else_body, output)?;
                    for (param, arg) in merge_params.iter().zip(else_args.iter()) {
                        let arg_val = self.get_value_ref(*arg)?;
                        let var = wgsl_var(*param);
                        writeln!(output, "{}{} = {};", self.ctx.indent_str(), var, arg_val).unwrap();
                    }
                    self.ctx.indent -= 1;
                    writeln!(output, "{}}}", self.ctx.indent_str()).unwrap();
                }

                Node::Loop {
                    state_vars,
                    init_args,
                    header_insts,
                    cond,
                    cond_is_continue,
                    body,
                } => {
                    // Initialize loop state as `var`.
                    for (var, init) in state_vars.iter().zip(init_args.iter()) {
                        let init_val = self.get_value_ref(*init)?;
                        let var_name = wgsl_var(*var);
                        let ty = self.ctx.type_emitter.type_to_wgsl(self.body.get_value_type(*var))?;
                        writeln!(
                            output,
                            "{}var {}: {} = {};",
                            self.ctx.indent_str(),
                            var_name,
                            ty,
                            init_val
                        )
                        .unwrap();
                        self.declared.insert(var_name.clone());
                        self.value_map.insert(*var, ValueBinding::Alias(var_name));
                    }
                    writeln!(output, "{}loop {{", self.ctx.indent_str()).unwrap();
                    self.ctx.indent += 1;
                    for inst_id in header_insts {
                        let inst = self.body.get_inst(*inst_id);
                        self.current_span = inst.span;
                        let expr = self.lower_inst(inst)?;
                        if let Some(result) = inst.result {
                            if matches!(
                                inst.data,
                                InstKind::Op {
                                    tag: crate::op::OpTag::StorageView(_),
                                    ..
                                }
                            ) {
                                self.value_map.insert(result, ValueBinding::Alias(expr));
                                continue;
                            }
                            let var = wgsl_var(result);
                            let ty =
                                self.ctx.type_emitter.type_to_wgsl(self.body.get_value_type(result))?;
                            writeln!(output, "{}var {}: {} = {};", self.ctx.indent_str(), var, ty, expr)
                                .unwrap();
                            self.declared.insert(var.clone());
                            self.value_map.insert(result, ValueBinding::Alias(var));
                        }
                    }
                    let cond_val = self.get_value_ref(*cond)?;
                    if *cond_is_continue {
                        // cond is "continue condition"; break when it's false.
                        writeln!(output, "{}if !({}) {{ break; }}", self.ctx.indent_str(), cond_val)
                            .unwrap();
                    } else {
                        // cond is "break condition"; break when it's true.
                        writeln!(output, "{}if {} {{ break; }}", self.ctx.indent_str(), cond_val).unwrap();
                    }
                    self.emit_nodes(body, output)?;
                    self.ctx.indent -= 1;
                    writeln!(output, "{}}}", self.ctx.indent_str()).unwrap();
                }

                Node::Return(value) => {
                    result_var = match value {
                        Some(v) => self.get_value_ref(*v)?,
                        None => String::new(),
                    };
                }
            }
        }

        Ok(result_var)
    }

    fn lower_inst(&mut self, inst: &WynInstNode) -> Result<String> {
        let result_ty = inst.result.map(|r| self.body.get_value_type(r).clone());
        self.current_span = inst.span;

        match &inst.data {
            InstKind::Op { tag, operands } => match tag {
                // Integer literals carry their type in the suffix:
                //   `Nu` for `u32`, `Ni` for `i32`. WGSL has no implicit
                //   int conversion, so respecting the SSA value's type is
                //   load-bearing for subsequent uses.
                crate::op::OpTag::Int(s) | crate::op::OpTag::Uint(s) => match result_ty.as_ref() {
                    Some(PolyType::Constructed(TypeName::UInt(32), _)) => Ok(format!("{}u", s)),
                    Some(PolyType::Constructed(TypeName::Int(32), _)) | _ => Ok(format!("{}i", s)),
                },
                crate::op::OpTag::Float(s) => {
                    let suffix = "f";
                    if s.contains('.') || s.contains('e') || s.contains('E') {
                        Ok(format!("{}{}", s, suffix))
                    } else {
                        Ok(format!("{}.0{}", s, suffix))
                    }
                }
                crate::op::OpTag::Bool(b) => Ok((if *b { "true" } else { "false" }).to_string()),
                crate::op::OpTag::Unit => Err(crate::err_wgsl_at!(
                    self.blame_span(),
                    "unit values aren't materializable in WGSL"
                )),

                crate::op::OpTag::BinOp(op) => {
                    let lhs = operands[0];
                    let rhs = operands[1];
                    // WGSL has no implicit numeric coercion (`i32 + u32` is an
                    // error), so when an operand's type doesn't match the
                    // BinOp's declared result type we wrap it in an explicit
                    // cast. This comes up e.g. when a slice's offset (i32
                    // literal) gets added to a parent `StorageView` offset
                    // (u32 arrayLength result).
                    let l = self.coerce_operand_to_result_ty(lhs, result_ty.as_ref())?;
                    let r = self.coerce_operand_to_result_ty(rhs, result_ty.as_ref())?;
                    match op.as_str() {
                        "**" => Ok(format!("pow({}, {})", l, r)),
                        _ => Ok(format!("({} {} {})", l, op, r)),
                    }
                }

                crate::op::OpTag::UnaryOp(op) => {
                    let inner = self.get_value(operands[0])?;
                    Ok(format!("({}{})", op, inner))
                }

                crate::op::OpTag::Tuple(_) => {
                    if operands.is_empty() {
                        return Err(crate::err_wgsl_at!(
                            self.blame_span(),
                            "empty tuple not supported in WGSL"
                        ));
                    }
                    let parts: Result<Vec<_>> = operands.iter().map(|e| self.get_value(*e)).collect();
                    let ty = result_ty.as_ref().ok_or_else(|| {
                        crate::err_wgsl_at!(self.blame_span(), "Tuple must have a result type")
                    })?;
                    let struct_name = self.ctx.type_emitter.type_to_wgsl(ty)?;
                    Ok(format!("{}({})", struct_name, parts?.join(", ")))
                }

                crate::op::OpTag::Vector(_) => {
                    let parts: Result<Vec<_>> = operands.iter().map(|e| self.get_value(*e)).collect();
                    let ty = result_ty.as_ref().ok_or_else(|| {
                        crate::err_wgsl_at!(self.blame_span(), "Vector must have a result type")
                    })?;
                    let wgsl_ty = self.ctx.type_emitter.type_to_wgsl(ty)?;
                    Ok(format!("{}({})", wgsl_ty, parts?.join(", ")))
                }

                crate::op::OpTag::ArrayLit(_) => {
                    let parts: Result<Vec<_>> = operands.iter().map(|e| self.get_value(*e)).collect();
                    let ty = result_ty.as_ref().ok_or_else(|| {
                        crate::err_wgsl_at!(self.blame_span(), "ArrayLit must have a result type")
                    })?;
                    let wgsl_ty = self.ctx.type_emitter.type_to_wgsl(ty)?;
                    // `array<T, N>(e0, e1, ...)` constructor.
                    Ok(format!("{}({})", wgsl_ty, parts?.join(", ")))
                }

                crate::op::OpTag::Project { index } => {
                    let base = operands[0];
                    let base_val = self.get_value(base)?;
                    // Determine base type: SSA → look up via body; Const's type
                    // isn't accessible that way so we error (projects on consts
                    // would be folded earlier).
                    let base_id = base.as_ssa().ok_or_else(|| {
                        crate::err_wgsl_at!(self.blame_span(), "Project base must be an SSA value")
                    })?;
                    let base_ty = self.body.get_value_type(base_id);
                    if matches!(base_ty, PolyType::Constructed(TypeName::Vec, _)) {
                        let swizzle = match index {
                            0 => "x",
                            1 => "y",
                            2 => "z",
                            3 => "w",
                            _ => {
                                return Err(crate::err_wgsl_at!(
                                    self.blame_span(),
                                    "invalid vector swizzle index: {}",
                                    index
                                ));
                            }
                        };
                        Ok(format!("{}.{}", base_val, swizzle))
                    } else if matches!(
                        crate::types::get_array_variant(base_ty),
                        Some(TypeName::ArrayVariantComposite)
                    ) {
                        // A Composite array is a WGSL `array<T, N>`: project by
                        // subscript. (View/Bounded/Virtual variants are emitted
                        // as structs, so they keep the `.fN` field access below.)
                        Ok(format!("{}[{}]", base_val, index))
                    } else {
                        Ok(format!("{}.f{}", base_val, index))
                    }
                }

                crate::op::OpTag::Index => {
                    let base = operands[0];
                    let index = operands[1];
                    let base_val = self.get_value(base)?;
                    let index_val = self.get_value(index)?;
                    // Virtual arrays are `{start, step, len}` triples in a
                    // generated struct; indexing computes `start + i*step`
                    // (matching SPIR-V's `lower_virtual_index`). Composite
                    // and view arrays just subscript normally.
                    if let Some(id) = base.as_ssa() {
                        let base_ty = self.body.get_value_type(id);
                        if let Some(PolyType::Constructed(TypeName::ArrayVariantVirtual, _)) =
                            base_ty.array_variant()
                        {
                            return Ok(format!("({}.f0 + {} * {}.f1)", base_val, index_val, base_val));
                        }
                        if let Some(PolyType::Constructed(TypeName::ArrayVariantBounded, _)) =
                            base_ty.array_variant()
                        {
                            // Bounded `{buffer, len}`: index into the buffer
                            // member (no bounds check against `len` — same
                            // contract as Composite).
                            return Ok(format!("{}.f0[{}]", base_val, index_val));
                        }
                    }
                    Ok(format!("{}[{}]", base_val, index_val))
                }

                crate::op::OpTag::Global(name) => {
                    // Constants like iResolution/iTime are emitted at module scope
                    // and referenced by their user-facing names (validated as
                    // legal WGSL identifiers). Wyn-internal defs go through the
                    // mangler.
                    if self.ctx.program.constants.iter().any(|c| c.name == *name) {
                        Ok(name.clone())
                    } else {
                        self.ctx.mangle_tracked(name)
                    }
                }

                crate::op::OpTag::Call(func) => {
                    let args: &[ValueRef] = operands;
                    // Route well-known builtins (type casts, math functions)
                    // through the same dispatch as `OpTag::Intrinsic`; fall
                    // back to a mangled user-function call.
                    let raw_strs: Result<Vec<_>> = args.iter().map(|a| self.get_value(*a)).collect();
                    let raw_strs = raw_strs?;
                    // A user-defined function shadows any same-named builtin:
                    // resolve the callee first and only fall back to builtin
                    // dispatch when no user function owns this name. Without
                    // this, a user `def step(...)` would be hijacked by the
                    // builtin `step` lowering (matched purely by surface name).
                    let callee = self.ctx.program.functions.iter().find(|f| f.name == *func);
                    if callee.is_none() {
                        // Structural dispatch: if this name is registered
                        // in `impl_source`, route through the `BuiltinLowering`
                        // so the qualifier prefix doesn't matter (`f32.cos`,
                        // `vec.cos`, `_w_intrinsic_cos` all share a `PrimOp`).
                        if let Some(lowered) =
                            self.try_lower_via_impl_source(func, &raw_strs, result_ty.as_ref())?
                        {
                            return Ok(lowered);
                        }
                        if let Some(lowered) = try_lower_wgsl_builtin(func, &raw_strs) {
                            return Ok(lowered);
                        }
                    }
                    // Mirror `lower_function`: drop arguments whose parameter
                    // slot on the callee is a view-array. The callee's
                    // signature has those params filtered out, so we skip
                    // the corresponding args here too.
                    let arg_strs: Vec<String> = match callee {
                        Some(f) => args
                            .iter()
                            .zip(f.body.params.iter())
                            .filter_map(|(arg, (_, pty, _))| {
                                if is_view_array_ty(pty) {
                                    None
                                } else {
                                    Some(self.get_value(*arg))
                                }
                            })
                            .collect::<Result<Vec<_>>>()?,
                        None => raw_strs,
                    };
                    let mangled = self.ctx.mangle_tracked(func)?;
                    Ok(format!("{}({})", mangled, arg_strs.join(", ")))
                }

                crate::op::OpTag::Intrinsic { id, overload_idx } => {
                    let args: &[ValueRef] = operands;
                    let known = catalog().known();
                    let arg_strs: Result<Vec<_>> = args.iter().map(|a| self.get_value(*a)).collect();
                    let arg_strs = arg_strs?;

                    // Honor the type-checker's overload choice for the
                    // operand-splatting ext-insts (`smoothstep(scalar,
                    // scalar, vec)`, `step(scalar, vec)`): mirror the SPIR-V
                    // backend by splatting the named scalar operands to the
                    // result vector width before emitting the same WGSL call.
                    // The chosen overload's scheme guarantees those operands
                    // are scalar, so the splat is unconditional. The name-keyed
                    // `lower_intrinsic` below only ever sees overload 0, so it
                    // can't reach this case on its own.
                    if let BuiltinLowering::ExtInstSplat { ext, splat_args } =
                        &crate::builtins::by_id(*id).overloads()[*overload_idx].lowering
                    {
                        if let (Some(name), Some(rty)) = (glsl_std450_wgsl_name(*ext), result_ty.as_ref()) {
                            let rty_str = self.ctx.type_emitter.type_to_wgsl(rty)?;
                            let mut splatted = arg_strs.clone();
                            for &pos in *splat_args {
                                splatted[pos] = format!("{}({})", rty_str, arg_strs[pos]);
                            }
                            return Ok(format!("{}({})", name, splatted.join(", ")));
                        }
                    }
                    // `_w_intrinsic_storage_len(set, binding)` → runtime
                    // length of the storage buffer at those coordinates.
                    // Both args are compile-time integer constants.
                    // WGSL's `arrayLength(&x)` returns `u32`. We cast to `i32`
                    // only when the result slot actually wants `i32`, which
                    // keeps the emitted declaration type consistent with the
                    // expression type (a mismatch is a naga parse error, not
                    // just a style issue).
                    let wants_i32 = matches!(
                        result_ty.as_ref(),
                        Some(PolyType::Constructed(TypeName::Int(32), _))
                    );
                    // `_w_intrinsic_thread_id()` → `_wgsl_gid.x` from the
                    // compute entry's auto-injected builtin parameter, or
                    // from the user-declared global-invocation-id param
                    // when one exists (`wgsl_gid_alias`). The SSA result
                    // type is always `u32` (see
                    // `parallelize::intrinsic_term(..., u32_ty)`), so no
                    // cast is needed.
                    if *id == known.thread_id && args.is_empty() {
                        let base = self.ctx.wgsl_gid_alias.as_deref().unwrap_or("_wgsl_gid");
                        return Ok(format!("{base}.x"));
                    }
                    // `_w_intrinsic_local_id()` → `_wgsl_lid.x` (also u32).
                    if *id == known.local_id && args.is_empty() {
                        return Ok("_wgsl_lid.x".to_string());
                    }
                    // `_w_intrinsic_num_workgroups()` → `_wgsl_nwg.x` (also u32).
                    if *id == known.num_workgroups && args.is_empty() {
                        return Ok("_wgsl_nwg.x".to_string());
                    }
                    // texture_load(tex, coord, lod) → textureLoad. Raw texel
                    // fetch (no filtering); referentially transparent.
                    if *id == known.texture_load && args.len() == 3 {
                        return Ok(format!(
                            "textureLoad({}, {}, {})",
                            arg_strs[0], arg_strs[1], arg_strs[2]
                        ));
                    }
                    // texture_sample(tex, samp, uv, lod) → textureSampleLevel.
                    // v1 uses EXPLICIT LOD (the trailing arg), not implicit
                    // `textureSample`, so the result is a pure function of its
                    // arguments — referentially transparent and valid in any
                    // stage. See the texture plan's v2 note for gradient-based
                    // filtering (`texture_sample_grad` → `textureSampleGrad`).
                    if *id == known.texture_sample && args.len() == 4 {
                        return Ok(format!(
                            "textureSampleLevel({}, {}, {}, {})",
                            arg_strs[0], arg_strs[1], arg_strs[2], arg_strs[3]
                        ));
                    }
                    if *id == known.storage_len && args.len() == 2 {
                        let set = self.resolve_const_u32(args[0]);
                        let binding = self.resolve_const_u32(args[1]);
                        match (set, binding) {
                            (Some(s), Some(b)) => {
                                let name = self.storage_name(s, b)?;
                                let expr = format!("arrayLength(&{})", name);
                                return Ok(if wants_i32 { format!("i32({})", expr) } else { expr });
                            }
                            _ => {
                                return Err(crate::err_wgsl_at!(
                                    self.blame_span(),
                                    "_w_intrinsic_storage_len expects const set/binding args"
                                ));
                            }
                        }
                    }
                    // `_w_intrinsic_length(arr)` is array length, semantically
                    // distinct from WGSL's vector-magnitude `length`. For a
                    // fixed-size composite array we emit the statically-known
                    // size as a literal; runtime-sized (storage) arrays route
                    // through WGSL's `arrayLength(&x)` (u32). Cast only when
                    // the SSA result expects `i32`.
                    // `_w_intrinsic_slice(arr, start, end)` → sub-view or
                    // materialized sub-array. Three cases mirror SPIR-V's
                    // `"slice"` arm: view→view (new handle), view→composite
                    // (materialize a `array<T,N>(...)` literal), and
                    // composite→composite (also a literal). Start/end are
                    // constants for the materialization cases; runtime
                    // start/end are allowed for view→view.
                    if *id == known.slice && args.len() == 3 {
                        let arr_id = args[0].as_ssa().ok_or_else(|| {
                            crate::err_wgsl_at!(
                                self.blame_span(),
                                "_w_intrinsic_slice: array arg must be an SSA value"
                            )
                        })?;
                        let result_ty_ref = result_ty.as_ref().ok_or_else(|| {
                            crate::err_wgsl_at!(
                                self.blame_span(),
                                "_w_intrinsic_slice must have a result type"
                            )
                        })?;
                        let result_is_composite = matches!(
                            result_ty_ref.array_variant(),
                            Some(PolyType::Constructed(TypeName::ArrayVariantComposite, _))
                        );
                        // Source is a view iff its type is the View variant.
                        let src_is_view = self
                            .body
                            .get_value_type(arr_id)
                            .array_variant()
                            .map(crate::types::is_array_variant_view)
                            .unwrap_or(false);
                        if src_is_view {
                            // Buffer name from the source view's type region; its
                            // runtime offset is the `.x` of its `vec2<u32>` value.
                            let buffer_name = self.view_buffer_name(arr_id)?;
                            let view_val = self.get_value(args[0])?;
                            if result_is_composite {
                                // View → Composite: materialize as
                                // `array<T,N>(buf[off+s], buf[off+s+1], ...)`.
                                let start = self.resolve_const_u32(args[1]).ok_or_else(|| {
                                    crate::err_wgsl_at!(
                                        self.blame_span(),
                                        "_w_intrinsic_slice(view → composite): start must be a constant"
                                    )
                                })?;
                                let end = self.resolve_const_u32(args[2]).ok_or_else(|| {
                                    crate::err_wgsl_at!(
                                        self.blame_span(),
                                        "_w_intrinsic_slice(view → composite): end must be a constant"
                                    )
                                })?;
                                let ty_str = self.ctx.type_emitter.type_to_wgsl(result_ty_ref)?;
                                let elems: Vec<String> = (start..end)
                                    .map(|i| format!("{}[(i32(({}).x) + {}i)]", buffer_name, view_val, i))
                                    .collect();
                                return Ok(format!("{}({})", ty_str, elems.join(", ")));
                            } else {
                                // View → View: a fresh `vec2<u32>` whose offset is
                                // `source.x + start` and len is `end - start`. Its
                                // type carries the same region, so downstream
                                // consumers recover the buffer from it.
                                let new_offset =
                                    format!("u32(i32(({}).x) + i32({}))", view_val, arg_strs[1]);
                                let new_len = format!("u32(i32({}) - i32({}))", arg_strs[2], arg_strs[1]);
                                return Ok(format!("vec2<u32>({}, {})", new_offset, new_len));
                            }
                        }
                        // Composite → Composite: `array<T,N>(arr[s], arr[s+1], ...)`.
                        if result_is_composite {
                            let start = self.resolve_const_u32(args[1]).ok_or_else(|| {
                                crate::err_wgsl_at!(
                                    self.blame_span(),
                                    "_w_intrinsic_slice: start must be a constant for composite slice"
                                )
                            })?;
                            let end = self.resolve_const_u32(args[2]).ok_or_else(|| {
                                crate::err_wgsl_at!(
                                    self.blame_span(),
                                    "_w_intrinsic_slice: end must be a constant for composite slice"
                                )
                            })?;
                            let ty_str = self.ctx.type_emitter.type_to_wgsl(result_ty_ref)?;
                            let elems: Vec<String> =
                                (start..end).map(|i| format!("{}[{}i]", arg_strs[0], i)).collect();
                            return Ok(format!("{}({})", ty_str, elems.join(", ")));
                        }
                        return Err(crate::err_wgsl_at!(
                            self.blame_span(),
                            "_w_intrinsic_slice: unsupported slice shape (src not view, dst not composite)"
                        ));
                    }
                    if *id == known.length && args.len() == 1 {
                        if let Some(id) = args[0].as_ssa() {
                            let ty = self.body.get_value_type(id);
                            // Virtual arrays carry their length in the `f2`
                            // field of the range struct, element-typed (so
                            // `0u32..<n` yields a `u32` field). `emit_length`
                            // upstream asks for i32, so cast when needed.
                            if let Some(PolyType::Constructed(TypeName::ArrayVariantVirtual, _)) =
                                ty.array_variant()
                            {
                                let expr = format!("{}.f2", arg_strs[0]);
                                return Ok(if wants_i32 { format!("i32({})", expr) } else { expr });
                            }
                            // Bounded arrays carry their runtime length in
                            // the `f1` field of the `{buffer, len}` struct
                            // (already i32).
                            if let Some(PolyType::Constructed(TypeName::ArrayVariantBounded, _)) =
                                ty.array_variant()
                            {
                                let expr = format!("{}.f1", arg_strs[0]);
                                return Ok(if wants_i32 { expr } else { format!("u32({})", expr) });
                            }
                            if let Some(PolyType::Constructed(TypeName::Size(n), _)) = ty.array_size() {
                                return Ok(if wants_i32 { format!("{}i", n) } else { format!("{}u", n) });
                            }
                            // View: the length is the `.y` of its `vec2<u32>`
                            // value (a whole-buffer entry view's `.y` is the
                            // buffer's `arrayLength`).
                            if let Some(PolyType::Constructed(TypeName::ArrayVariantView, _)) =
                                ty.array_variant()
                            {
                                let expr = format!("({}).y", arg_strs[0]);
                                return Ok(if wants_i32 { format!("i32({})", expr) } else { expr });
                            }
                            // Runtime-sized storage array: `arrayLength(&x)` is u32.
                            let expr = format!("arrayLength(&{})", arg_strs[0]);
                            return Ok(if wants_i32 { format!("i32({})", expr) } else { expr });
                        }
                        return Err(crate::err_wgsl_at!(
                            self.blame_span(),
                            "_w_intrinsic_length requires an SSA array argument"
                        ));
                    }
                    let name = by_id(*id).dispatch_name();
                    self.lower_intrinsic(name, &arg_strs, result_ty.as_ref())
                }

                crate::op::OpTag::StorageImageLoad(binding) => {
                    let image = self.storage_image_name(*binding)?;
                    let coord = self.get_value(operands[0])?;
                    Ok(format!("textureLoad({}, {})", image, coord))
                }

                crate::op::OpTag::StorageImageStore(binding) => {
                    let image = self.storage_image_name(*binding)?;
                    let coord = self.get_value(operands[0])?;
                    let texel = self.get_value(operands[1])?;
                    Ok(format!("textureStore({}, {}, {})", image, coord, texel))
                }

                crate::op::OpTag::Extern(linkage) => Err(crate::err_wgsl_at!(
                    self.blame_span(),
                    "Extern functions are not supported in WGSL (linkage: {})",
                    linkage
                )),

                crate::op::OpTag::ArrayRange { has_step } => {
                    let start = operands[0];
                    let len = operands[1];
                    // Virtual array lowered as a `VirtRangeN` struct whose
                    // fields are (start, step, len) — field order matches
                    // SPIR-V's composite_construct so `Project { index: 0/1/2 }`
                    // extracts start/step/len naturally. Default step is `1`
                    // in the element's type when absent.
                    let start_s = self.get_value(start)?;
                    let len_s = self.get_value(len)?;
                    let ty = result_ty.as_ref().ok_or_else(|| {
                        crate::err_wgsl_at!(self.blame_span(), "ArrayRange must have a result type")
                    })?;
                    let elem_ty = ty.elem_type().ok_or_else(|| {
                        crate::err_wgsl_at!(self.blame_span(), "ArrayRange result missing elem type")
                    })?;
                    let elem_str = self.ctx.type_emitter.type_to_wgsl(elem_ty)?;
                    let struct_name = self.ctx.type_emitter.type_to_wgsl(ty)?;
                    let step_s = if *has_step {
                        self.get_value(operands[2])?
                    } else {
                        match elem_ty {
                            PolyType::Constructed(TypeName::UInt(_), _) => "1u".to_string(),
                            PolyType::Constructed(TypeName::Float(_), _) => "1.0f".to_string(),
                            _ => "1i".to_string(),
                        }
                    };
                    // Fields emit in struct order: f0=start, f1=step, f2=len.
                    // `elem_str` is used only for its existence (ensures the
                    // type is cached before constructor emits).
                    let _ = elem_str;
                    Ok(format!("{}({}, {}, {})", struct_name, start_s, step_s, len_s))
                }

                crate::op::OpTag::Matrix { .. } => Err(crate::err_wgsl_at!(
                    self.blame_span(),
                    "Matrix literals are not yet implemented in WGSL lowering"
                )),

                // Materialize is handled in emit_nodes so it becomes a
                // `var<function>` (subscriptable) instead of a `let`.
                crate::op::OpTag::Materialize => Err(crate::err_wgsl_at!(
                    self.blame_span(),
                    "internal: Materialize should be handled in emit_nodes"
                )),

                crate::op::OpTag::DynamicExtract => {
                    let base = operands[0];
                    let index = operands[1];
                    let base_val = self.get_value(base)?;
                    let index_val = self.get_value(index)?;
                    // For a Bounded base the materialized value is a
                    // `{f0: array<T,N>, f1: i32}` struct, so the dynamic
                    // index has to traverse the `f0` member first.
                    if let Some(id) = base.as_ssa() {
                        let base_ty = self.body.get_value_type(id);
                        if matches!(
                            base_ty.array_variant(),
                            Some(PolyType::Constructed(TypeName::ArrayVariantBounded, _))
                        ) {
                            return Ok(format!("{}.f0[{}]", base_val, index_val));
                        }
                    }
                    Ok(format!("{}[{}]", base_val, index_val))
                }

                // Storage view: a runtime `{offset, len}` handle into a backing
                // buffer, reified as a `vec2<u32>` value (`.x` = offset, `.y` =
                // len). The backing buffer is static — recovered from the value
                // type's region at the consumer (`view_buffer_name`) — so it's
                // NOT part of the value. Workgroup views are the one exception:
                // their `_wg_<id>` name isn't in any type, so record it.
                crate::op::OpTag::StorageView(src) => {
                    let offset = operands[0];
                    let len = operands[1];
                    let result_id = inst.result.ok_or_else(|| {
                        crate::err_wgsl_at!(self.blame_span(), "StorageView must have a result")
                    })?;
                    if let crate::op::PureViewSource::Workgroup { id, .. } = src {
                        self.workgroup_view_name.insert(result_id, format!("_wg_{}", id));
                    }
                    let offset_expr = self.get_value(offset)?;
                    let len_expr = self.get_value(len)?;
                    Ok(format!("vec2<u32>(u32({}), u32({}))", offset_expr, len_expr))
                }

                crate::op::OpTag::StorageViewLen => {
                    // Length is the `.y` of the view's `vec2<u32>` value.
                    let view = operands[0];
                    let view_val = self.get_value(view)?;
                    Ok(format!("({}).y", view_val))
                }

                crate::op::OpTag::ViewIndex
                | crate::op::OpTag::PlaceIndex
                | crate::op::OpTag::OutputSlot { .. } => {
                    unreachable!("OpTag::{:?} is EGIR-only and must not reach SSA backend", tag)
                }
            },

            // OutputSlot / Alloca / ViewIndex / PlaceIndex / Load / Store are
            // all handled specially in `emit_nodes` (they produce places or
            // emit statements, not let-binding expressions). Reaching
            // lower_inst for any of them is an internal bug.
            InstKind::OutputSlot { .. }
            | InstKind::ViewIndex { .. }
            | InstKind::PlaceIndex { .. }
            | InstKind::Alloca { .. }
            | InstKind::Load { .. }
            | InstKind::Store { .. }
            | InstKind::ControlBarrier => Err(crate::err_wgsl_at!(
                self.blame_span(),
                "internal: {:?} should be handled in emit_nodes",
                inst.data
            )),
        }
    }

    fn lower_intrinsic(
        &mut self,
        name: &str,
        args: &[String],
        ret_ty: Option<&PolyType<TypeName>>,
    ) -> Result<String> {
        if let Some(lowered) = self.try_lower_via_impl_source(name, args, ret_ty)? {
            return Ok(lowered);
        }
        if let Some(lowered) = try_lower_wgsl_builtin(name, args) {
            return Ok(lowered);
        }
        Err(crate::err_wgsl_at!(
            self.blame_span(),
            "intrinsic '{}' is not yet implemented in WGSL lowering",
            name
        ))
    }

    /// Look up `name` in `impl_source` and, if it's a `PrimOp`, lower
    /// it via `lower_primop_wgsl`. Returns `Ok(None)` when the name
    /// isn't registered or the impl isn't a `PrimOp` we lower (linked
    /// SPIR-V functions and `Intrinsic`s are out of scope here).
    fn try_lower_via_impl_source(
        &mut self,
        name: &str,
        args: &[String],
        result_ty: Option<&PolyType<TypeName>>,
    ) -> Result<Option<String>> {
        let Some(builtin) = catalog().lookup_lowering(name) else {
            return Ok(None);
        };
        let prim_op = match builtin {
            BuiltinLowering::PrimOp(p) => p,
            BuiltinLowering::LinkedSpirv(_)
            | BuiltinLowering::ByBuiltinId
            | BuiltinLowering::ExtInstSplat { .. }
            | BuiltinLowering::NotLowered => return Ok(None),
        };
        let result_ty_str = match result_ty {
            Some(ty) => Some(self.ctx.type_emitter.type_to_wgsl(ty)?),
            None => None,
        };
        Ok(lower_primop_wgsl(prim_op, args, result_ty_str.as_deref()))
    }
}

/// Lower a `BuiltinLowering::PrimOp` to its WGSL expression. Mirrors the
/// SPIR-V backend's `lower_primop` — both backends consume the same
/// `BuiltinLowering` map from `impl_source`, so the qualifier prefix on the
/// surface name (`f32.cos`, `vec.cos`, `_w_intrinsic_cos`) is invisible
/// here: only the structural `PrimOp` matters.
///
/// `result_ty_str` is the WGSL spelling of the call's result type
/// (e.g. `"f32"`, `"i32"`, `"vec3<f32>"`). Required for type-cast ops
/// (`SIToFP`, `Bitcast`, …) where the cast target comes from the
/// result slot, not the operand.
///
/// Returns `None` for `PrimOp`s without a defined WGSL emission (e.g.
/// `IsNan`, `OuterProduct`); the caller surfaces an error or falls
/// back to a user-function call.
fn wgsl_storage_image_format(format: crate::pipeline_descriptor::StorageImageFormat) -> &'static str {
    use crate::pipeline_descriptor::StorageImageFormat;
    match format {
        StorageImageFormat::Rgba8Unorm => "rgba8unorm",
        StorageImageFormat::Rgba16Float => "rgba16float",
        StorageImageFormat::Rgba32Float => "rgba32float",
        StorageImageFormat::R32Float => "r32float",
    }
}

fn wgsl_storage_access(access: crate::interface::StorageAccess) -> &'static str {
    use crate::interface::StorageAccess;
    match access {
        StorageAccess::ReadOnly => "read",
        StorageAccess::WriteOnly => "write",
        StorageAccess::ReadWrite => "read_write",
    }
}

/// GLSL.std.450 ext-inst number → its WGSL builtin spelling. Shared by
/// the `GlslExt` PrimOp arm and the `ExtInstSplat` lowering (which splats
/// scalar operands before emitting the same call). Returns `None` for
/// numbers with no direct WGSL equivalent.
fn glsl_std450_wgsl_name(n: u32) -> Option<&'static str> {
    Some(match n {
        1 => "round",
        3 => "trunc",
        4 | 5 => "abs",
        6 | 7 => "sign",
        8 => "floor",
        9 => "ceil",
        10 => "fract",
        11 => "radians",
        12 => "degrees",
        13 => "sin",
        14 => "cos",
        15 => "tan",
        16 => "asin",
        17 => "acos",
        18 => "atan",
        19 => "sinh",
        20 => "cosh",
        21 => "tanh",
        22 => "asinh",
        23 => "acosh",
        24 => "atanh",
        25 => "atan2",
        26 => "pow",
        27 => "exp",
        28 => "log",
        29 => "exp2",
        30 => "log2",
        31 => "sqrt",
        32 => "inverseSqrt",
        33 => "determinant",
        34 => "inverse",
        37 | 38 | 39 => "min",
        40 | 41 | 42 => "max",
        43 | 44 | 45 => "clamp",
        46 => "mix",
        48 => "step",
        49 => "smoothstep",
        50 => "fma",
        53 => "ldexp",
        66 => "length",
        67 => "distance",
        68 => "cross",
        69 => "normalize",
        71 => "reflect",
        72 => "refract",
        _ => return None,
    })
}

fn lower_primop_wgsl(prim_op: &PrimOp, args: &[String], result_ty_str: Option<&str>) -> Option<String> {
    use PrimOp::*;
    match prim_op {
        // GLSL.std.450 ops with direct WGSL builtin equivalents.
        // Numbers come from the GLSL.std.450 extended-instruction set;
        // WGSL spells most of them identically (sin/cos/floor/…) and
        // collapses signed/unsigned/float variants into a single
        // overloaded builtin (FMin/UMin/SMin → `min`).
        GlslExt(n) => {
            let name = glsl_std450_wgsl_name(*n)?;
            Some(format!("{}({})", name, args.join(", ")))
        }

        // Wyn's `mod` is GLSL-style FMod (always non-negative when the
        // divisor is positive). WGSL has no direct equivalent — `%` is
        // FRem semantics — so synthesize via the math identity:
        //   mod(x, y) = x - y * floor(x / y)
        // Works for scalar and same-shape vector operands (WGSL's `/`,
        // `*`, and `floor` are all componentwise).
        FMod => {
            if args.len() == 2 {
                Some(format!("({0} - {1} * floor({0} / {1}))", args[0], args[1]))
            } else {
                None
            }
        }

        // Screen-space derivatives map to WGSL's native spellings.
        Fwidth => Some(format!("fwidth({})", args.join(", "))),
        DPdx => Some(format!("dpdx({})", args.join(", "))),
        DPdy => Some(format!("dpdy({})", args.join(", "))),

        // WGSL has no `isNan` / `isInf` builtins, so synthesize them from
        // the standard floating-point identities. `x != x` is true only
        // for NaN; a finite |x| ceiling detects ±Inf. Both spellings are
        // componentwise, so they also work for vector operands.
        IsNan => Some(format!("({0} != {0})", args[0])),
        IsInf => Some(format!("(abs({0}) > 3.4028235e38)", args[0])),

        Dot => Some(format!("dot({})", args.join(", "))),

        // Matrix / vector multiplications: WGSL's `*` operator handles
        // matrix×matrix, matrix×vector, vector×scalar, etc., picking
        // the right overload from operand types.
        MatrixTimesMatrix | MatrixTimesVector | VectorTimesMatrix | VectorTimesScalar
        | MatrixTimesScalar => {
            if args.len() == 2 {
                Some(format!("({} * {})", args[0], args[1]))
            } else {
                None
            }
        }

        // Type conversions: WGSL spells the cast as `<target_ty>(x)`.
        // `Bitcast` is special — WGSL needs explicit `bitcast<T>(x)`.
        SIToFP | UIToFP | FPToSI | FPToUI | FPConvert | SConvert | UConvert => {
            let result_ty = result_ty_str?;
            if args.len() == 1 {
                Some(format!("{}({})", result_ty, args[0]))
            } else {
                None
            }
        }
        Bitcast => {
            let result_ty = result_ty_str?;
            if args.len() == 1 {
                Some(format!("bitcast<{}>({})", result_ty, args[0]))
            } else {
                None
            }
        }

        // `OuterProduct`, `IsNan`, `IsInf`, and the arithmetic /
        // comparison / bitwise ops don't reach this path under current
        // codegen (arithmetic flows through `InstKind::BinOp`/`UnaryOp`
        // with infix emission; nan/inf and outer aren't used by any
        // testfile yet). Return `None` so the caller can surface a
        // clear "not yet implemented" error if one shows up.
        _ => None,
    }
}

/// Inline `f32.pi` / `f32.e` (and the `f64` analogues) as WGSL float
/// literals. Those are prelude `def`s — not `PrimOp`s — so they can't
/// dispatch through `lower_primop_wgsl`; without this shortcut they'd
/// each compile to a mangled user-function call returning a constant.
///
/// All math ops, type casts, vector ops, etc. now route through
/// `lower_primop_wgsl` via `impl_source.get(name)`; the surface name's
/// qualifier prefix (`f32.cos`, `vec.cos`, `_w_intrinsic_cos`) is no
/// longer load-bearing here.
fn try_lower_wgsl_builtin(name: &str, _args: &[String]) -> Option<String> {
    let (to, from) = name.split_once('.')?;
    if matches!(to, "f32" | "f64") {
        match from {
            "pi" => return Some("3.14159265358979323846f".to_string()),
            "e" => return Some("2.71828182845904523536f".to_string()),
            _ => {}
        }
    }
    None
}

#[cfg(test)]
#[path = "ssa_lowering_tests.rs"]
mod ssa_lowering_tests;
