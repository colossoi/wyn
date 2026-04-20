//! WGSL SSA-to-text lowering.
//!
//! The backend is built against WGSL directly (not as a GLSL dialect).
//! Entry points carry `@vertex` / `@fragment` / `@compute` attributes;
//! module-scope types and bindings precede function definitions.
//!
//! This file provides:
//! - [`wgsl_mangle`]: injective identifier mangler, aware of WGSL reserved words.
//! - [`validate_wgsl_identifier`]: host-contract identifier validator.
//! - [`TypeEmitter`]: Wyn polytype → WGSL type string, with cached tuple structs.
//! - [`lower`]: entry point.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

use polytype::Type as PolyType;

use crate::ast::{Span, TypeName};
use crate::error::Result;
use crate::ssa::types::{
    EntryPoint, ExecutionModel, FuncBody, Function, InstKind, IoDecoration, Program, ValueId, ValueRef,
    ViewSource, WynInstNode,
};
use crate::types::TypeExt;

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

fn wgsl_keyword_set() -> &'static std::collections::HashSet<&'static str> {
    use std::sync::OnceLock;
    static SET: OnceLock<std::collections::HashSet<&'static str>> = OnceLock::new();
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
    tuple_type_cache: HashMap<String, String>,
    pub tuple_structs: HashMap<String, Vec<String>>,
    tuple_counter: usize,
}

impl Default for TypeEmitter {
    fn default() -> Self {
        Self::new()
    }
}

impl TypeEmitter {
    pub fn new() -> Self {
        Self {
            tuple_type_cache: HashMap::new(),
            tuple_structs: HashMap::new(),
            tuple_counter: 0,
        }
    }

    pub fn type_to_wgsl(&mut self, ty: &PolyType<TypeName>) -> Result<String> {
        match ty {
            PolyType::Constructed(name, args) => match name {
                TypeName::Float(32) => Ok("f32".to_string()),
                TypeName::Int(32) => Ok("i32".to_string()),
                TypeName::UInt(32) => Ok("u32".to_string()),
                TypeName::Bool => Ok("bool".to_string()),
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

struct LowerCtx<'a> {
    program: &'a Program,
    type_emitter: TypeEmitter,
    lowered: HashSet<String>,
    indent: usize,
    /// Track mangled names for collision detection.
    mangled_names: HashMap<String, String>,
    /// Entry-point output structs keyed by their field signature
    /// ("attr0+ty0,attr1+ty1,..."). Maps sig → (struct_name, fields).
    /// `fields` is an ordered list of (field_name, attribute_prefix,
    /// wgsl_type) tuples.
    output_structs: HashMap<String, (String, Vec<(String, String, String)>)>,
    output_struct_counter: usize,
}

impl<'a> LowerCtx<'a> {
    fn new(program: &'a Program) -> Self {
        Self {
            program,
            type_emitter: TypeEmitter::new(),
            lowered: HashSet::new(),
            indent: 0,
            mangled_names: HashMap::new(),
            output_structs: HashMap::new(),
            output_struct_counter: 0,
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
        let numbered: Vec<(String, String, String)> = fields
            .into_iter()
            .enumerate()
            .map(|(i, (attr, ty))| (format!("f{}", i), attr, ty))
            .collect();
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
        if !self.type_emitter.tuple_structs.is_empty() {
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
        }

        // Entry-point output struct declarations — each field carries
        // its `@builtin(...)` or `@location(N)` attribute inline,
        // since WGSL requires the decoration to sit on the struct
        // member rather than a free module-scope variable.
        if !self.output_structs.is_empty() {
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
        }

        // Uniform declarations: `@group(G) @binding(B) var<uniform> name: T;`.
        // Uniforms must be host-shared with the user-written name, so we
        // validate rather than mangle.
        for u in &self.program.uniforms {
            validate_wgsl_identifier(&u.name).map_err(|e| crate::err_wgsl!("uniform {}: {}", u.name, e))?;
            let ty_str = self.type_emitter.type_to_wgsl(&u.ty)?;
            writeln!(
                output,
                "@group({}) @binding({}) var<uniform> {}: {};",
                u.set, u.binding, u.name, ty_str
            )
            .unwrap();
        }
        if !self.program.uniforms.is_empty() {
            writeln!(output).unwrap();
        }

        // Storage buffer declarations from user-declared `#[storage]`
        // bindings. Access mode → second parameter of `var<storage, ...>`.
        for s in &self.program.storage {
            validate_wgsl_identifier(&s.name)
                .map_err(|e| crate::err_wgsl!("storage buffer {}: {}", s.name, e))?;
            let ty_str = self.type_emitter.type_to_wgsl(&s.ty)?;
            let access = match s.access {
                crate::interface::StorageAccess::ReadOnly => "read",
                crate::interface::StorageAccess::WriteOnly => "write",
                crate::interface::StorageAccess::ReadWrite => "read_write",
            };
            writeln!(
                output,
                "@group({}) @binding({}) var<storage, {}> {}: {};",
                s.set, s.binding, access, s.name, ty_str
            )
            .unwrap();
        }

        // Compiler-introduced storage bindings + entry-level storage-
        // backed I/O. WGSL needs these at module scope; dedupe by
        // (set, binding) and coalesce access modes so an (in, out) pair
        // on the same slot becomes `read_write`.
        let mut synth: HashMap<(u32, u32), (String, String, bool, bool)> = HashMap::new();
        // Key → (elem_ty_str, module_name, has_read, has_write).
        let is_declared =
            |set, binding| self.program.storage.iter().any(|s| s.set == set && s.binding == binding);
        for entry in &self.program.entry_points {
            // Explicit compiler-inserted bindings (e.g. parallelize's
            // partial-sum buffer).
            for sb in &entry.storage_bindings {
                if is_declared(sb.set, sb.binding) {
                    continue;
                }
                let key = (sb.set, sb.binding);
                let ty_str = self.type_emitter.type_to_wgsl(&sb.elem_ty)?;
                let entry_ref = synth.entry(key).or_insert_with(|| {
                    let name = format!("_buf_{}_{}", sb.set, sb.binding);
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
                if let Some((set, binding)) = input.storage_binding {
                    if is_declared(set, binding) {
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
                    let entry_ref = synth.entry((set, binding)).or_insert_with(|| {
                        let name = format!("_buf_{}_{}", set, binding);
                        (ty_str.clone(), name, false, false)
                    });
                    // Entry inputs are read by convention.
                    entry_ref.2 = true;
                }
            }
            // Entry outputs likewise. For scalar-valued compute outputs
            // (e.g. reduce → f32), the user-level type isn't an array
            // but the underlying binding still holds a runtime-sized
            // array of that scalar — the SOAC parallelize pass packs
            // the result into a single-element slot.
            for out in &entry.outputs {
                if let Some((set, binding)) = out.storage_binding {
                    if is_declared(set, binding) {
                        continue;
                    }
                    let elem_ty = out.ty.elem_type().cloned().unwrap_or_else(|| out.ty.clone());
                    let ty_str = self.type_emitter.type_to_wgsl(&elem_ty)?;
                    let entry_ref = synth.entry((set, binding)).or_insert_with(|| {
                        let name = format!("_buf_{}_{}", set, binding);
                        (ty_str.clone(), name, false, false)
                    });
                    entry_ref.3 = true;
                }
            }
        }
        // Sort for determinism.
        let mut synth_sorted: Vec<_> = synth.into_iter().collect();
        synth_sorted.sort_by_key(|((set, binding), _)| (*set, *binding));
        for ((set, binding), (elem_ty, name, has_in, has_out)) in synth_sorted {
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
        }

        if !self.program.storage.is_empty()
            || self.program.entry_points.iter().any(|e| !e.storage_bindings.is_empty())
        {
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
        let params: Result<Vec<String>> = body
            .params
            .iter()
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
        for (i, input) in entry.inputs.iter().enumerate() {
            // Storage-backed inputs (compute shader runtime-sized array
            // params) become module-scope bindings, not function params.
            if input.storage_binding.is_some() {
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
        let non_storage_outputs: Vec<(usize, &crate::ssa::types::EntryOutput)> = entry
            .outputs
            .iter()
            .enumerate()
            .filter(|(_, o)| o.storage_binding.is_none())
            .collect();
        // For multi-output: the generated struct name and the per-output
        // field mapping (orig_index → field_name), used to pre-declare
        // `var _out_struct: VsOutN;` in the body prelude and to route
        // `OutputPtr` targets into its fields.
        let mut multi_output_struct: Option<(String, HashMap<usize, String>)> = None;
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
                    let mut index_to_field: HashMap<usize, String> = HashMap::new();
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
                        index_to_field
                            .insert(*orig_index, format!("_out_struct.f{}", pos));
                    }
                    multi_output_struct = Some((struct_name.clone(), index_to_field));
                    (struct_name, false)
                }
            }
        };

        let name = self.mangle_tracked(&entry.name)?;
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
                    writeln!(output, "{}var _out_struct: {};", self.indent_str(), struct_name)
                        .unwrap();
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

/// A storage-view handle tracked per ValueId: the name of the underlying
/// `@group @binding` storage declaration plus the offset and length
/// expressions the view was created with. When a `StorageViewIndex` or
/// `StorageViewLen` reaches lowering, we resolve the view's ValueId to
/// one of these and emit `name[offset + idx]` or the recorded len expr.
#[derive(Clone)]
struct ViewHandle {
    buffer_name: String,
    offset_expr: String,
    len_expr: String,
}

struct BodyLowerCtx<'a, 'b> {
    ctx: &'a mut LowerCtx<'b>,
    body: &'a FuncBody,
    /// Map from ValueId to WGSL expression text.
    value_map: HashMap<ValueId, String>,
    /// Set of names declared with `let`/`var` in the current scope.
    declared: HashSet<String>,
    /// Storage view handles keyed by their `StorageView` result ValueId.
    view_handles: HashMap<ValueId, ViewHandle>,
    /// OutputPtr bookkeeping: for an entry with `@location(N)` outputs we
    /// declare a `var<function> _out{N}: T` at function entry and alias
    /// the `OutputPtr` result's ValueId to that name. `Store` to such a
    /// pointer becomes `_out{N} = value;`. At function exit the entry
    /// lowering returns whichever output the entry produces (for
    /// single-output entries, `_out0`).
    output_ptrs: HashMap<ValueId, String>,
    /// Set to `true` if at least one `OutputPtr` was lowered; the entry
    /// wrapper then returns the declared `_out0` (or builds a return
    /// struct for multi-output) instead of emitting a `return <expr>;`
    /// from the body's terminator result.
    uses_output_ptrs: bool,
    /// Override for the name each `OutputPtr { index }` resolves to. If an
    /// index is present here, that name is used as the store target (and
    /// cached in `output_ptrs`); otherwise the default `_out{index}` is
    /// used. Populated by `lower_entry_point` when it's packing outputs
    /// into a struct — then the name is `_out_struct.f{index}`.
    output_target_names: HashMap<usize, String>,
    /// Span of the instruction currently being lowered.
    current_span: Option<Span>,
    func_span: Span,
}

impl<'a, 'b> BodyLowerCtx<'a, 'b> {
    fn new(ctx: &'a mut LowerCtx<'b>, body: &'a FuncBody, func_span: Span) -> Self {
        Self {
            ctx,
            body,
            value_map: HashMap::new(),
            declared: HashSet::new(),
            view_handles: HashMap::new(),
            output_ptrs: HashMap::new(),
            uses_output_ptrs: false,
            output_target_names: HashMap::new(),
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
                        if let InstKind::Int(s) = &inst.data {
                            return s.parse::<u32>().ok();
                        }
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Resolve a storage binding (set, binding) to its module-scope name.
    /// Checks user-declared `#[storage]` buffers first, falls back to the
    /// synthesized `_buf_{set}_{binding}` naming used for compiler-
    /// introduced compute-entry bindings.
    fn storage_name(&self, set: u32, binding: u32) -> Result<String> {
        if let Some(name) = self
            .ctx
            .program
            .storage
            .iter()
            .find(|s| s.set == set && s.binding == binding)
            .map(|s| s.name.clone())
        {
            return Ok(name);
        }
        // Synthesized binding — matches the naming in `lower_program`.
        for entry in &self.ctx.program.entry_points {
            if entry.storage_bindings.iter().any(|sb| sb.set == set && sb.binding == binding)
                || entry.inputs.iter().any(|i| i.storage_binding == Some((set, binding)))
                || entry.outputs.iter().any(|o| o.storage_binding == Some((set, binding)))
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
        self.value_map.get(&id).cloned().ok_or_else(|| {
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
        // Seed parameter names into value_map.
        for (value_id, _, name) in &self.body.params {
            let mangled = self.ctx.mangle_tracked(name)?;
            self.value_map.insert(*value_id, mangled.clone());
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
                        // Alloca: `var<function> x: T;` — no initializer.
                        InstKind::Alloca { elem_ty } => {
                            let result_id = inst.result.ok_or_else(|| {
                                crate::err_wgsl_at!(self.blame_span(), "Alloca must have a result")
                            })?;
                            let ty = self.ctx.type_emitter.type_to_wgsl(elem_ty)?;
                            let var = wgsl_var(result_id);
                            writeln!(output, "{}var {}: {};", self.ctx.indent_str(), var, ty).unwrap();
                            self.declared.insert(var.clone());
                            self.value_map.insert(result_id, var);
                            continue;
                        }

                        // Load through a `var<function>` is reading the
                        // variable by name — WGSL's function-scope
                        // pointer model collapses pointer and value at
                        // the syntactic level.
                        InstKind::Load { ptr } => {
                            let result_id = inst.result.ok_or_else(|| {
                                crate::err_wgsl_at!(self.blame_span(), "Load must have a result")
                            })?;
                            let name = match ptr {
                                ValueRef::Ssa(id) => self.get_value_ref(*id)?,
                                ValueRef::Const(_) => {
                                    return Err(crate::err_wgsl_at!(
                                        self.blame_span(),
                                        "Load from a constant pointer is not meaningful"
                                    ));
                                }
                            };
                            self.value_map.insert(result_id, name);
                            continue;
                        }

                        // Store: `target = value;`. Target is either
                        // `_outN` for an OutputPtr or the alloca'd
                        // var<function> name.
                        InstKind::Store { ptr, value } => {
                            if let Some(ptr_id) = ptr.as_ssa() {
                                let target = if let Some(out_name) = self.output_ptrs.get(&ptr_id).cloned()
                                {
                                    out_name
                                } else {
                                    self.get_value_ref(ptr_id)?
                                };
                                let val = self.get_value(*value)?;
                                writeln!(output, "{}{} = {};", self.ctx.indent_str(), target, val).unwrap();
                                continue;
                            }
                            return Err(crate::err_wgsl_at!(
                                self.blame_span(),
                                "Store ptr must be an SSA value"
                            ));
                        }

                        // `_w_intrinsic_uninit()` returns an
                        // uninitialized composite. In WGSL that's just a
                        // `var<function> x: T;` — no initializer, no
                        // function call. Emit the declaration directly
                        // rather than falling through to the Call
                        // handler (which would mangle the intrinsic
                        // name as if it were a user function).
                        InstKind::Call { func, .. } if func == "_w_intrinsic_uninit" => {
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
                            self.value_map.insert(result_id, var);
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
                        InstKind::Call { func, args }
                            if func == "_w_intrinsic_array_with_inplace"
                                || func == "_w_intrinsic_array_with" =>
                        {
                            if args.len() != 3 {
                                return Err(crate::err_wgsl_at!(
                                    self.blame_span(),
                                    "{} expects 3 args, got {}",
                                    func,
                                    args.len()
                                ));
                            }
                            let arr_src = self.get_value(args[0])?;
                            let idx = self.get_value(args[1])?;
                            let val = self.get_value(args[2])?;
                            let result_id = inst.result.ok_or_else(|| {
                                crate::err_wgsl_at!(self.blame_span(), "{} must have a result", func)
                            })?;
                            if func == "_w_intrinsic_array_with_inplace" {
                                writeln!(
                                    output,
                                    "{}{}[{}] = {};",
                                    self.ctx.indent_str(),
                                    arr_src,
                                    idx,
                                    val
                                )
                                .unwrap();
                                self.value_map.insert(result_id, arr_src);
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
                                self.value_map.insert(result_id, var);
                            }
                            continue;
                        }

                        // Materialize: `var<function> x: T = expr;` so
                        // the value becomes subscriptable via
                        // DynamicExtract's `x[i]`. Distinct from the
                        // normal `let` path because WGSL forbids
                        // dynamic indexing of `let`-bound values.
                        InstKind::Materialize { value } => {
                            let result_id = inst.result.ok_or_else(|| {
                                crate::err_wgsl_at!(self.blame_span(), "Materialize must have a result")
                            })?;
                            let ty =
                                self.ctx.type_emitter.type_to_wgsl(self.body.get_value_type(result_id))?;
                            let var = wgsl_var(result_id);
                            let val = self.get_value(*value)?;
                            writeln!(output, "{}var {}: {} = {};", self.ctx.indent_str(), var, ty, val)
                                .unwrap();
                            self.declared.insert(var.clone());
                            self.value_map.insert(result_id, var);
                            continue;
                        }

                        _ => {}
                    }

                    let expr = self.lower_inst(inst)?;
                    if let Some(result) = inst.result {
                        // Alias-only InstKinds — the `result` is a handle
                        // pointing at something already declared at module
                        // scope or elsewhere; we only record the alias in
                        // value_map and emit nothing. Copying into a local
                        // `var` would be nonsense (for StorageView, it'd
                        // try to clone a runtime-sized storage array into
                        // a function local, which WGSL rejects).
                        if matches!(
                            inst.data,
                            InstKind::OutputPtr { .. } | InstKind::StorageView { .. }
                        ) {
                            self.value_map.insert(result, expr);
                            continue;
                        }
                        let var = wgsl_var(result);
                        if self.declared.contains(&var) {
                            writeln!(output, "{}{} = {};", self.ctx.indent_str(), var, expr).unwrap();
                        } else {
                            // `var` (not `let`) because `structurize` can
                            // emit a merge block's instructions twice —
                            // once at the tail of the last arm and once
                            // via the post-if fall-through in `lower_arm`.
                            // GLSL's re-assignable function locals tolerate
                            // that; WGSL's `let` is immutable, so we always
                            // declare with `var` to allow a subsequent
                            // same-value re-assignment.
                            let ty =
                                self.ctx.type_emitter.type_to_wgsl(self.body.get_value_type(result))?;
                            writeln!(output, "{}var {}: {} = {};", self.ctx.indent_str(), var, ty, expr)
                                .unwrap();
                            self.declared.insert(var.clone());
                        }
                        self.value_map.insert(result, var);
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
                    self.value_map.insert(*target, var);
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
                        self.value_map.insert(*param, var);
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
                        self.value_map.insert(*var, var_name.clone());
                        self.declared.insert(var_name);
                    }
                    writeln!(output, "{}loop {{", self.ctx.indent_str()).unwrap();
                    self.ctx.indent += 1;
                    // Emit header insts (condition-test insts).
                    for inst_id in header_insts {
                        let inst = self.body.get_inst(*inst_id);
                        self.current_span = inst.span;
                        let expr = self.lower_inst(inst)?;
                        if let Some(result) = inst.result {
                            // Mirror the alias-only check from the
                            // main per-inst dispatch: OutputPtr and
                            // StorageView never produce a local
                            // declaration — the expression text is
                            // the handle.
                            if matches!(
                                inst.data,
                                InstKind::OutputPtr { .. } | InstKind::StorageView { .. }
                            ) {
                                self.value_map.insert(result, expr);
                                continue;
                            }
                            let var = wgsl_var(result);
                            let ty =
                                self.ctx.type_emitter.type_to_wgsl(self.body.get_value_type(result))?;
                            writeln!(output, "{}var {}: {} = {};", self.ctx.indent_str(), var, ty, expr)
                                .unwrap();
                            self.declared.insert(var.clone());
                            self.value_map.insert(result, var);
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
            // Integer literals carry their type in the suffix:
            //   `Nu` for `u32`, `Ni` for `i32`. WGSL has no implicit
            //   int conversion, so respecting the SSA value's type is
            //   load-bearing for subsequent uses.
            InstKind::Int(s) => match result_ty.as_ref() {
                Some(PolyType::Constructed(TypeName::UInt(32), _)) => Ok(format!("{}u", s)),
                Some(PolyType::Constructed(TypeName::Int(32), _)) | _ => {
                    Ok(format!("{}i", s))
                }
            },
            InstKind::Float(s) => {
                let suffix = "f";
                if s.contains('.') || s.contains('e') || s.contains('E') {
                    Ok(format!("{}{}", s, suffix))
                } else {
                    Ok(format!("{}.0{}", s, suffix))
                }
            }
            InstKind::Bool(b) => Ok((if *b { "true" } else { "false" }).to_string()),
            InstKind::String(_) => Err(crate::err_wgsl_at!(
                self.blame_span(),
                "string literals aren't representable in WGSL"
            )),
            InstKind::Unit => Err(crate::err_wgsl_at!(
                self.blame_span(),
                "unit values aren't materializable in WGSL"
            )),

            InstKind::BinOp { op, lhs, rhs } => {
                let l = self.get_value(*lhs)?;
                let r = self.get_value(*rhs)?;
                match op.as_str() {
                    "**" => Ok(format!("pow({}, {})", l, r)),
                    _ => Ok(format!("({} {} {})", l, op, r)),
                }
            }

            InstKind::UnaryOp { op, operand } => {
                let inner = self.get_value(*operand)?;
                Ok(format!("({}{})", op, inner))
            }

            InstKind::Tuple(elems) => {
                if elems.is_empty() {
                    return Err(crate::err_wgsl_at!(
                        self.blame_span(),
                        "empty tuple not supported in WGSL"
                    ));
                }
                let parts: Result<Vec<_>> = elems.iter().map(|e| self.get_value(*e)).collect();
                let ty = result_ty.as_ref().ok_or_else(|| {
                    crate::err_wgsl_at!(self.blame_span(), "Tuple must have a result type")
                })?;
                let struct_name = self.ctx.type_emitter.type_to_wgsl(ty)?;
                Ok(format!("{}({})", struct_name, parts?.join(", ")))
            }

            InstKind::Vector(elems) => {
                let parts: Result<Vec<_>> = elems.iter().map(|e| self.get_value(*e)).collect();
                let ty = result_ty.as_ref().ok_or_else(|| {
                    crate::err_wgsl_at!(self.blame_span(), "Vector must have a result type")
                })?;
                let wgsl_ty = self.ctx.type_emitter.type_to_wgsl(ty)?;
                Ok(format!("{}({})", wgsl_ty, parts?.join(", ")))
            }

            InstKind::ArrayLit { elements } => {
                let parts: Result<Vec<_>> = elements.iter().map(|e| self.get_value(*e)).collect();
                let ty = result_ty.as_ref().ok_or_else(|| {
                    crate::err_wgsl_at!(self.blame_span(), "ArrayLit must have a result type")
                })?;
                let wgsl_ty = self.ctx.type_emitter.type_to_wgsl(ty)?;
                // `array<T, N>(e0, e1, ...)` constructor.
                Ok(format!("{}({})", wgsl_ty, parts?.join(", ")))
            }

            InstKind::Project { base, index } => {
                let base_val = self.get_value(*base)?;
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
                } else {
                    Ok(format!("{}.f{}", base_val, index))
                }
            }

            InstKind::Index { base, index } => {
                let base_val = self.get_value(*base)?;
                let index_val = self.get_value(*index)?;
                Ok(format!("{}[{}]", base_val, index_val))
            }

            InstKind::Global(name) => {
                // Constants like iResolution/iTime are emitted at module scope
                // and referenced by their user-facing names (validated as
                // legal WGSL identifiers). Wyn-internal defs go through the
                // mangler.
                if self.ctx.program.uniforms.iter().any(|u| u.name == *name)
                    || self.ctx.program.constants.iter().any(|c| c.name == *name)
                {
                    Ok(name.clone())
                } else {
                    self.ctx.mangle_tracked(name)
                }
            }

            InstKind::Call { func, args } => {
                let arg_strs: Result<Vec<_>> = args.iter().map(|a| self.get_value(*a)).collect();
                let arg_strs = arg_strs?;
                // Route well-known builtins (type casts, math functions)
                // through the same dispatch as `InstKind::Intrinsic`; fall
                // back to a mangled user-function call.
                if let Some(lowered) = try_lower_wgsl_builtin(func, &arg_strs) {
                    return Ok(lowered);
                }
                let mangled = self.ctx.mangle_tracked(func)?;
                Ok(format!("{}({})", mangled, arg_strs.join(", ")))
            }

            InstKind::Intrinsic { name, args } => {
                let arg_strs: Result<Vec<_>> = args.iter().map(|a| self.get_value(*a)).collect();
                let arg_strs = arg_strs?;
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
                if name == "_w_intrinsic_storage_len" && args.len() == 2 {
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
                if name == "_w_intrinsic_length" && args.len() == 1 {
                    if let Some(id) = args[0].as_ssa() {
                        let ty = self.body.get_value_type(id);
                        if let Some(PolyType::Constructed(TypeName::Size(n), _)) = ty.array_size() {
                            return Ok(if wants_i32 {
                                format!("{}i", n)
                            } else {
                                format!("{}u", n)
                            });
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
                self.lower_intrinsic(name, &arg_strs, result_ty.as_ref())
            }

            InstKind::Extern(linkage) => Err(crate::err_wgsl_at!(
                self.blame_span(),
                "Extern functions are not supported in WGSL (linkage: {})",
                linkage
            )),

            InstKind::ArrayRange { .. } => Err(crate::err_wgsl_at!(
                self.blame_span(),
                "ArrayRange should be expanded before reaching WGSL lowering"
            )),

            InstKind::Matrix(_) => Err(crate::err_wgsl_at!(
                self.blame_span(),
                "Matrix literals are not yet implemented in WGSL lowering"
            )),

            InstKind::OutputPtr { index } => {
                let result_id = inst.result.ok_or_else(|| {
                    crate::err_wgsl_at!(self.blame_span(), "OutputPtr must have a result")
                })?;
                let name = self
                    .output_target_names
                    .get(index)
                    .cloned()
                    .unwrap_or_else(|| format!("_out{}", index));
                self.output_ptrs.insert(result_id, name.clone());
                self.uses_output_ptrs = true;
                // Nothing to emit: the `var<function>` declaration happens
                // in the entry-point wrapper, not here.
                Ok(name)
            }

            InstKind::Store { ptr, value } => {
                // Two supported shapes:
                //   1. Store to an OutputPtr → becomes `_outN = value;`.
                //   2. Store to a local var (Alloca) — TODO, not yet
                //      implemented; we'll land Alloca/Load/Store together
                //      in the memory-ops commit.
                let ptr_id = ptr.as_ssa().ok_or_else(|| {
                    crate::err_wgsl_at!(self.blame_span(), "Store ptr must be an SSA value")
                })?;
                if let Some(out_name) = self.output_ptrs.get(&ptr_id).cloned() {
                    let val = self.get_value(*value)?;
                    return Err(crate::err_wgsl_at!(
                        self.blame_span(),
                        "Store to OutputPtr must be handled in emit_nodes, not lower_inst (internal bug): out={}, val={}",
                        out_name,
                        val
                    ));
                }
                Err(crate::err_wgsl_at!(
                    self.blame_span(),
                    "Store to non-output pointer is not yet implemented in WGSL lowering"
                ))
            }

            // Alloca/Load/Store are handled specially in `emit_nodes` so
            // they become `var<function>` declarations and assignments
            // rather than let-bindings. Reaching lower_inst with one of
            // these is an internal bug.
            InstKind::Alloca { .. } | InstKind::Load { .. } => Err(crate::err_wgsl_at!(
                self.blame_span(),
                "internal: {:?} should be handled in emit_nodes",
                inst.data
            )),

            // Materialize is handled in emit_nodes so it becomes a
            // `var<function>` (subscriptable) instead of a `let`.
            InstKind::Materialize { .. } => Err(crate::err_wgsl_at!(
                self.blame_span(),
                "internal: Materialize should be handled in emit_nodes"
            )),

            InstKind::DynamicExtract { base, index } => {
                let base_val = self.get_value(*base)?;
                let index_val = self.get_value(*index)?;
                Ok(format!("{}[{}]", base_val, index_val))
            }

            // Storage view: remember (buffer_name, offset, len) against
            // the view's ValueId so subsequent StorageViewIndex /
            // StorageViewLen can resolve through it. The "value" of a
            // view node is the buffer name — uses outside of
            // Index/Len (rare) get the binding directly.
            InstKind::StorageView { source, offset, len } => {
                let buffer_name = match source {
                    ViewSource::Storage { set, binding } => self.storage_name(*set, *binding)?,
                    ViewSource::Inherited { parent } => {
                        // Inherit the parent view's underlying binding
                        // name (offset/len come fresh from this Node).
                        self.view_handles.get(parent).map(|h| h.buffer_name.clone()).ok_or_else(|| {
                            crate::err_wgsl_at!(self.blame_span(), "Inherited view's parent has no handle")
                        })?
                    }
                };
                let offset_expr = self.get_value(*offset)?;
                let len_expr = self.get_value(*len)?;
                let result_id = inst.result.ok_or_else(|| {
                    crate::err_wgsl_at!(self.blame_span(), "StorageView must have a result")
                })?;
                self.view_handles.insert(
                    result_id,
                    ViewHandle {
                        buffer_name: buffer_name.clone(),
                        offset_expr,
                        len_expr,
                    },
                );
                // Return the buffer name itself. Downstream uses resolve
                // through view_handles for offset/len.
                Ok(buffer_name)
            }

            InstKind::StorageViewIndex { view, index } => {
                let view_id = view.as_ssa().ok_or_else(|| {
                    crate::err_wgsl_at!(self.blame_span(), "StorageViewIndex view must be SSA")
                })?;
                let handle = self.view_handles.get(&view_id).cloned().ok_or_else(|| {
                    crate::err_wgsl_at!(
                        self.blame_span(),
                        "StorageViewIndex references view without a known handle"
                    )
                })?;
                let idx = self.get_value(*index)?;
                // `buf[i32(offset) + i32(idx)]`. Both operands are cast
                // to `i32` because WGSL has no implicit int-int coercion
                // and offset/idx can independently come in as either
                // signed or unsigned (parallelize emits u32 offsets;
                // Wyn indices are commonly i32). Casting both keeps the
                // expression monomorphic; naga folds casts of literals
                // at validation time.
                Ok(format!(
                    "{}[(i32({}) + i32({}))]",
                    handle.buffer_name, handle.offset_expr, idx
                ))
            }

            InstKind::StorageViewLen { view } => {
                let view_id = view.as_ssa().ok_or_else(|| {
                    crate::err_wgsl_at!(self.blame_span(), "StorageViewLen view must be SSA")
                })?;
                let handle = self.view_handles.get(&view_id).cloned().ok_or_else(|| {
                    crate::err_wgsl_at!(
                        self.blame_span(),
                        "StorageViewLen references view without a known handle"
                    )
                })?;
                Ok(handle.len_expr)
            }
        }
    }

    fn lower_intrinsic(
        &mut self,
        name: &str,
        args: &[String],
        _ret_ty: Option<&PolyType<TypeName>>,
    ) -> Result<String> {
        if let Some(lowered) = try_lower_wgsl_builtin(name, args) {
            return Ok(lowered);
        }
        Err(crate::err_wgsl_at!(
            self.blame_span(),
            "intrinsic '{}' is not yet implemented in WGSL lowering",
            name
        ))
    }
}

/// Map a Wyn builtin/intrinsic name to its WGSL emission. Returns `None`
/// when the name isn't recognized (caller falls back to a mangled user
/// function call or raises an error, per context).
///
/// Covers:
///   - Scalar/vector math built-ins with 1:1 WGSL names (`dot`, `sin`, …).
///   - Type-cast intrinsics of the form `f32.i32`/`i32.f32`/… → `f32(x)`.
///   - A few names Wyn spells differently from WGSL (`f32.sqrt`, `f32.pi`, …).
fn try_lower_wgsl_builtin(name: &str, args: &[String]) -> Option<String> {
    // Strip the `_w_intrinsic_` prefix so the per-op tables work on
    // both "sin" and "_w_intrinsic_sin" inputs.
    let stripped = name.strip_prefix("_w_intrinsic_").unwrap_or(name);

    // Special-case aliases where the Wyn name differs from WGSL's.
    match stripped {
        "magnitude" => {
            // `length` in WGSL (distinct from array length; called on
            // vectors / scalars).
            return Some(format!("length({})", args.join(", ")));
        }
        "mod" => {
            // Wyn's `mod` is FMod. WGSL's `%` on floats is FMod too.
            if args.len() == 2 {
                return Some(format!("({} % {})", args[0], args[1]));
            }
        }
        _ => {}
    }

    // Strip a trailing type suffix like `_i32`, `_u32`, `_f32`, `_f64`,
    // `_bool` so `_w_intrinsic_max_i32` dispatches as `max`.
    let bare = {
        let candidates = ["_i32", "_u32", "_f32", "_f64", "_bool"];
        candidates.iter().find_map(|suf| stripped.strip_suffix(suf)).unwrap_or(stripped)
    };

    // WGSL built-ins with matching names.
    const DIRECT: &[&str] = &[
        "dot",
        "cross",
        "normalize",
        "length",
        "distance",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "atan2",
        "exp",
        "exp2",
        "log",
        "log2",
        "sqrt",
        "inverseSqrt",
        "abs",
        "sign",
        "floor",
        "ceil",
        "fract",
        "round",
        "trunc",
        "min",
        "max",
        "clamp",
        "mix",
        "step",
        "smoothstep",
        "pow",
        "radians",
        "degrees",
        "reflect",
        "refract",
    ];
    if DIRECT.contains(&bare) {
        return Some(format!("{}({})", bare, args.join(", ")));
    }

    // Type-cast intrinsics: `f32.i32(x)` → `f32(x)`. The source type in
    // the suffix is informational; WGSL's constructor-style cast reads
    // only the target type.
    if let Some((to, _from)) = name.split_once('.') {
        match to {
            "f32" | "i32" | "u32" | "bool" => {
                if args.len() == 1 {
                    return Some(format!("{}({})", to, args[0]));
                }
            }
            _ => {}
        }

        // Dotted math built-ins: Wyn spells many of these as `f32.sin`,
        // `f32.sqrt`, etc. WGSL uses the bare name.
        if matches!(to, "f32" | "f64") {
            let bare = &name[to.len() + 1..];
            if DIRECT.contains(&bare) {
                return Some(format!("{}({})", bare, args.join(", ")));
            }
            // f32.pi / f32.e are constants in Wyn.
            match bare {
                "pi" => return Some("3.14159265358979323846f".to_string()),
                "e" => return Some("2.71828182845904523536f".to_string()),
                "max" | "min" => return Some(format!("{}({})", bare, args.join(", "))),
                _ => {}
            }
        }
    }

    None
}

#[cfg(test)]
#[path = "ssa_lowering_tests.rs"]
mod ssa_lowering_tests;
