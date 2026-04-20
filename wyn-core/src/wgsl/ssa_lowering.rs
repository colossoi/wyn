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
    EntryPoint, FuncBody, Function, InstKind, Program, ValueId, ValueRef, WynInstNode,
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
                    if let Some(PolyType::Constructed(TypeName::Size(n), _)) = ty.array_size() {
                        Ok(format!("array<{}, {}>", elem, n))
                    } else {
                        Err(crate::err_wgsl!(
                            "WGSL array must have concrete size for standalone use: {:?}",
                            ty
                        ))
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
}

impl<'a> LowerCtx<'a> {
    fn new(program: &'a Program) -> Self {
        Self {
            program,
            type_emitter: TypeEmitter::new(),
            lowered: HashSet::new(),
            indent: 0,
            mangled_names: HashMap::new(),
        }
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
        // prepend the accumulated tuple struct declarations.
        let mut code = String::new();

        // Emit all non-extern functions first (entry points reference them).
        for func in &self.program.functions {
            self.lower_function(func, &mut code)?;
        }

        // Emit entry points.
        for entry in &self.program.entry_points {
            self.lower_entry_point(entry, &mut code)?;
        }

        // Prepend generated structs.
        let mut output = String::new();
        writeln!(output, "// Generated by the Wyn compiler — WGSL backend").unwrap();
        writeln!(output).unwrap();
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
        // v1: full entry-point support will land in a follow-up commit
        // covering attribute emission, storage/uniform bindings, and
        // compute workgroup sizes. For now, error loudly so the driver
        // surfaces it.
        let _ = entry;
        let _ = output;
        Err(crate::err_wgsl!(
            "WGSL entry-point emission is not yet implemented (function '{}')",
            entry.name
        ))
    }
}

// -----------------------------------------------------------------------------
// Body-level lowering
// -----------------------------------------------------------------------------

struct BodyLowerCtx<'a, 'b> {
    ctx: &'a mut LowerCtx<'b>,
    body: &'a FuncBody,
    /// Map from ValueId to WGSL expression text.
    value_map: HashMap<ValueId, String>,
    /// Set of names declared with `let`/`var` in the current scope.
    declared: HashSet<String>,
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
            current_span: None,
            func_span,
        }
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
                    let expr = self.lower_inst(inst)?;
                    if let Some(result) = inst.result {
                        let var = wgsl_var(result);
                        if self.declared.contains(&var) {
                            writeln!(output, "{}{} = {};", self.ctx.indent_str(), var, expr).unwrap();
                        } else {
                            let ty =
                                self.ctx.type_emitter.type_to_wgsl(self.body.get_value_type(result))?;
                            writeln!(output, "{}let {}: {} = {};", self.ctx.indent_str(), var, ty, expr)
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
                        let expr = self.lower_inst(inst)?;
                        if let Some(result) = inst.result {
                            let var = wgsl_var(result);
                            let ty =
                                self.ctx.type_emitter.type_to_wgsl(self.body.get_value_type(result))?;
                            writeln!(output, "{}let {}: {} = {};", self.ctx.indent_str(), var, ty, expr)
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
            InstKind::Int(s) => Ok(format!("{}i", s)),
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
                let mangled = self.ctx.mangle_tracked(func)?;
                Ok(format!("{}({})", mangled, arg_strs?.join(", ")))
            }

            InstKind::Intrinsic { name, args } => {
                let arg_strs: Result<Vec<_>> = args.iter().map(|a| self.get_value(*a)).collect();
                self.lower_intrinsic(name, &arg_strs?, result_ty.as_ref())
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

            InstKind::Alloca { .. }
            | InstKind::Load { .. }
            | InstKind::Store { .. }
            | InstKind::OutputPtr { .. } => Err(crate::err_wgsl_at!(
                self.blame_span(),
                "memory operations ({:?}) are not yet implemented in WGSL lowering",
                inst.data
            )),

            InstKind::Materialize { .. } | InstKind::DynamicExtract { .. } => Err(crate::err_wgsl_at!(
                self.blame_span(),
                "dynamic indexing ({:?}) is not yet implemented in WGSL lowering",
                inst.data
            )),

            InstKind::StorageView { .. }
            | InstKind::StorageViewIndex { .. }
            | InstKind::StorageViewLen { .. } => Err(crate::err_wgsl_at!(
                self.blame_span(),
                "storage view operations ({:?}) are not yet implemented in WGSL lowering",
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
        // Minimal builtin dispatch: pass through a curated set of WGSL-
        // equivalent built-ins. Extend as we port testfiles.
        let direct_builtins = &[
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
        if direct_builtins.contains(&name) {
            return Ok(format!("{}({})", name, args.join(", ")));
        }
        // Explicit conversion intrinsics: `f32.i32(x)` maps to `f32(x)`, etc.
        // Wyn spells these as intrinsic names like `f32.i32`. WGSL uses
        // type-as-constructor.
        if let Some((to, _from)) = name.split_once('.') {
            match to {
                "f32" | "i32" | "u32" | "bool" => {
                    if args.len() == 1 {
                        return Ok(format!("{}({})", to, args[0]));
                    }
                }
                _ => {}
            }
        }
        // Fall back to a user-level error rather than silently dropping.
        let _ = ret_ty;
        Err(crate::err_wgsl_at!(
            self.blame_span(),
            "intrinsic '{}' is not yet implemented in WGSL lowering",
            name
        ))
    }
}

#[cfg(test)]
#[path = "ssa_lowering_tests.rs"]
mod ssa_lowering_tests;
