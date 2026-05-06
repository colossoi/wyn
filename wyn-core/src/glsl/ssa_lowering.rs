//! SSA to GLSL lowering.
//!
//! This module converts SSA programs to GLSL shader source code.
//! It generates separate strings for vertex and fragment shaders.

#[cfg(test)]
#[path = "ssa_lowering_tests.rs"]
mod ssa_lowering_tests;

use crate::ast::{Span, TypeName};
use crate::builtins::lowering::{BuiltinLowering, Intrinsic, PrimOp};
use crate::builtins::names::{INTRINSIC_ARRAY_WITH, INTRINSIC_ARRAY_WITH_INPLACE, INTRINSIC_UNINIT};
use crate::error::Result;
use crate::lowering_common::ShaderStage;
use crate::ssa::types::{ConstantValue, FuncBody, InstKind, ValueId, ValueRef, WynInstNode};
use crate::ssa::types::{EntryPoint, ExecutionModel, Function, IoDecoration, Program};
use crate::types::TypeExt;
use crate::{bail_glsl, bail_glsl_at};
use polytype::Type as PolyType;
use rspirv::spirv;
use std::collections::{HashMap, HashSet};
use std::fmt::Write;

/// Output from GLSL lowering - separate shader strings
#[derive(Debug, Clone)]
pub struct GlslOutput {
    /// Vertex shader source (None if no vertex entry point)
    pub vertex: Option<String>,
    /// Fragment shader source (None if no fragment entry point)
    pub fragment: Option<String>,
}

/// Lower an SSA program to GLSL
pub fn lower(program: &Program) -> Result<GlslOutput> {
    let mut ctx = LowerCtx::new(program);
    ctx.lower_program()
}

/// Lower an SSA program to Shadertoy-compatible GLSL
/// Returns just the fragment shader with mainImage entry point
pub fn lower_shadertoy(program: &Program) -> Result<String> {
    let mut ctx = LowerCtx::new(program);
    ctx.lower_shadertoy()
}

/// Context for lowering SSA to GLSL
struct LowerCtx<'a> {
    program: &'a Program,
    /// Functions by name
    func_index: HashMap<String, usize>,
    /// Functions that have been lowered
    lowered: HashSet<String>,
    /// Current indentation level
    indent: usize,
    /// Tuple types that need struct definitions (keyed by struct name)
    tuple_structs: HashMap<String, Vec<String>>,
    /// Counter for unique tuple struct names
    tuple_counter: usize,
    /// Cache from tuple type signature to struct name
    tuple_type_cache: HashMap<String, String>,
    /// Tracks every compiler-internal name that's been mangled this
    /// compilation (mangled → original). Detects collisions as a defensive
    /// invariant — the sanitizer is injective by construction, but this
    /// map catches bugs: missing escape cases, accidental double-mangling,
    /// future scheme changes that violate assumptions.
    mangled_names: HashMap<String, String>,
}

/// Sanitize a compiler-internal name into a legal, injective GLSL identifier.
///
/// GLSL identifiers match `[A-Za-z_][A-Za-z0-9_]*`. Our encoding uses `_` as
/// the exclusive escape prefix — underscore never appears unescaped in the
/// output — so every `_` is unambiguously the start of an escape sequence.
///
/// | Input char      | Output      | Purpose                                 |
/// |-----------------|-------------|-----------------------------------------|
/// | `A-Z a-z 0-9`   | self        | already legal                           |
/// | `_`             | `_U`        | escape underscore itself                |
/// | `.`             | `_D`        | module dot (e.g. `materials.foo`)       |
/// | `$`             | `_S`        | specialization separator                |
/// | other           | `_X<hex>_`  | fallback; terminated to bound hex digits|
///
/// Every mangled identifier is prefixed with `w_`, which guarantees
/// first-char validity and keeps the output out of GLSL keyword space
/// and the reserved `gl_` prefix.
///
/// The scheme is injective in practice by construction: in the body, `_`
/// always starts an escape. To avoid producing the GLSL-reserved `__`
/// sequence where a leading source underscore collides with the prefix
/// terminator, the prefix shortens to `w` when the input starts with a
/// non-alphanumeric char (which would otherwise encode to `_<escape>`).
/// Pathological collisions (e.g. a user-level ident named `Ufoo` aliasing
/// mangled `_foo`) are caught by the `mangled_names` tracker in
/// `LowerCtx`, surfacing as a compile error. We don't decode — this is
/// one-way, strictly for code generation.
fn glsl_mangle(name: &str) -> String {
    use std::fmt::Write as _;
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
    out
}

/// GLSL-reserved keywords and type names. Names in this set are rejected
/// when they appear at an entry-point I/O site (where we preserve the
/// user's spelling verbatim for host-contract reasons).
fn glsl_keyword_set() -> &'static std::collections::HashSet<&'static str> {
    use std::sync::OnceLock;
    static SET: OnceLock<std::collections::HashSet<&'static str>> = OnceLock::new();
    SET.get_or_init(|| {
        [
            // Storage qualifiers / layout
            "attribute",
            "const",
            "uniform",
            "varying",
            "buffer",
            "shared",
            "coherent",
            "volatile",
            "restrict",
            "readonly",
            "writeonly",
            "layout",
            "centroid",
            "flat",
            "smooth",
            "noperspective",
            "patch",
            "sample",
            "invariant",
            "precise",
            "subroutine",
            "in",
            "out",
            "inout",
            // Control flow
            "break",
            "continue",
            "do",
            "for",
            "while",
            "switch",
            "case",
            "default",
            "if",
            "else",
            "discard",
            "return",
            // Types
            "void",
            "bool",
            "true",
            "false",
            "float",
            "double",
            "int",
            "uint",
            "vec2",
            "vec3",
            "vec4",
            "dvec2",
            "dvec3",
            "dvec4",
            "ivec2",
            "ivec3",
            "ivec4",
            "uvec2",
            "uvec3",
            "uvec4",
            "bvec2",
            "bvec3",
            "bvec4",
            "mat2",
            "mat3",
            "mat4",
            "mat2x2",
            "mat2x3",
            "mat2x4",
            "mat3x2",
            "mat3x3",
            "mat3x4",
            "mat4x2",
            "mat4x3",
            "mat4x4",
            "dmat2",
            "dmat3",
            "dmat4",
            "dmat2x2",
            "dmat2x3",
            "dmat2x4",
            "dmat3x2",
            "dmat3x3",
            "dmat3x4",
            "dmat4x2",
            "dmat4x3",
            "dmat4x4",
            "atomic_uint",
            "struct",
            // Precision qualifiers
            "lowp",
            "mediump",
            "highp",
            "precision",
            // Sampler & image types (enumerated where they matter; not exhaustive)
            "sampler1D",
            "sampler2D",
            "sampler3D",
            "samplerCube",
            "sampler1DArray",
            "sampler2DArray",
            "sampler2DRect",
            "samplerCubeArray",
            "samplerBuffer",
            "sampler2DMS",
            "sampler2DMSArray",
            "isampler1D",
            "isampler2D",
            "isampler3D",
            "isamplerCube",
            "usampler1D",
            "usampler2D",
            "usampler3D",
            "usamplerCube",
            "image1D",
            "image2D",
            "image3D",
            "imageCube",
            "image1DArray",
            "image2DArray",
        ]
        .into_iter()
        .collect()
    })
}

/// Validate that a host-contract name (uniform / vertex attribute / fragment
/// output) is a legal GLSL identifier. Used where we preserve user-written
/// spelling rather than mangling — we reject at compile time rather than
/// let GLSL fail downstream.
fn validate_glsl_identifier(name: &str) -> core::result::Result<(), String> {
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
    if name.starts_with("gl_") {
        return Err(format!("identifier '{}' starts with reserved prefix 'gl_'", name));
    }
    if glsl_keyword_set().contains(name) {
        return Err(format!("identifier '{}' is a GLSL keyword", name));
    }
    Ok(())
}

impl<'a> LowerCtx<'a> {
    fn new(program: &'a Program) -> Self {
        let mut func_index = HashMap::new();
        for (i, func) in program.functions.iter().enumerate() {
            func_index.insert(func.name.clone(), i);
        }

        LowerCtx {
            program,
            func_index,
            lowered: HashSet::new(),
            indent: 0,
            tuple_structs: HashMap::new(),
            tuple_counter: 0,
            tuple_type_cache: HashMap::new(),
            mangled_names: HashMap::new(),
        }
    }

    fn lower_program(&mut self) -> Result<GlslOutput> {
        let mut vertex_shader = None;
        let mut fragment_shader = None;

        for entry in &self.program.entry_points {
            match &entry.execution_model {
                ExecutionModel::Vertex => {
                    vertex_shader = Some(self.lower_shader(&entry.name, ShaderStage::Vertex)?);
                }
                ExecutionModel::Fragment => {
                    fragment_shader = Some(self.lower_shader(&entry.name, ShaderStage::Fragment)?);
                }
                ExecutionModel::Compute { .. } => {
                    bail_glsl!("Compute shaders are not supported in GLSL output format");
                }
            }
        }

        Ok(GlslOutput {
            vertex: vertex_shader,
            fragment: fragment_shader,
        })
    }

    fn lower_shadertoy(&mut self) -> Result<String> {
        // Find fragment entry point
        let entry = self
            .program
            .entry_points
            .iter()
            .find(|e| matches!(e.execution_model, ExecutionModel::Fragment))
            .ok_or_else(|| {
                crate::error::CompilerError::GlslError("No fragment entry point found".to_string(), None)
            })?;

        // Clear state
        self.tuple_structs.clear();
        self.tuple_type_cache.clear();
        self.tuple_counter = 0;
        self.lowered.clear();

        let mut code = String::new();

        // Lower helper functions first
        let deps = self.collect_dependencies(&entry.name)?;
        for dep_name in &deps {
            if dep_name != &entry.name {
                if let Some(&idx) = self.func_index.get(dep_name) {
                    self.lower_function(&self.program.functions[idx].clone(), &mut code)?;
                }
            }
        }

        // Lower Shadertoy entry point
        self.lower_shadertoy_entry_point(entry, &mut code)?;

        // Build final output with struct definitions first
        let mut output = String::new();
        writeln!(output, "// Generated by Wyn compiler for Shadertoy").unwrap();
        writeln!(output).unwrap();

        // Emit struct definitions for tuple types
        if !self.tuple_structs.is_empty() {
            for (struct_name, field_types) in &self.tuple_structs {
                writeln!(output, "struct {} {{", struct_name).unwrap();
                for (i, field_type) in field_types.iter().enumerate() {
                    writeln!(output, "    {} f{};", field_type, i).unwrap();
                }
                writeln!(output, "}};").unwrap();
            }
            writeln!(output).unwrap();
        }

        output.push_str(&code);
        Ok(output)
    }

    fn lower_shadertoy_entry_point(&mut self, entry: &EntryPoint, output: &mut String) -> Result<()> {
        // Find the fragCoord parameter. Both `position` and `frag_coord`
        // Wyn-side builtin decorations map to the GLSL pixel-coordinate slot
        // (`BuiltIn::Position` and `BuiltIn::FragCoord` respectively); accept
        // either so the wrapper's `vec4 <name> = vec4(fc.x, iResolution.y -
        // fc.y, 0.0, 1.0);` declaration always lands.
        let mut frag_coord_name = None;
        for input in &entry.inputs {
            if matches!(
                &input.decoration,
                Some(IoDecoration::BuiltIn(
                    spirv::BuiltIn::Position | spirv::BuiltIn::FragCoord
                ))
            ) {
                frag_coord_name = Some(input.name.clone());
            }
        }

        writeln!(output, "void mainImage(out vec4 fragColor, in vec2 fc) {{").unwrap();
        self.indent += 1;

        // If the shader expects vec4 fragCoord, convert from vec2
        if let Some(ref name) = frag_coord_name {
            writeln!(
                output,
                "{}vec4 {} = vec4(fc.x, iResolution.y - fc.y, 0.0, 1.0);",
                self.indent_str(),
                name
            )
            .unwrap();
        }

        // Map output index 0 → fragColor for OutputPtr+Store handling
        let mut body_ctx = BodyLowerCtx::new(self, &entry.body, entry.span);
        body_ctx.entry_output_names.insert(0, "fragColor".to_string());
        let result = body_ctx.lower(output)?;
        let used_output_ptrs = body_ctx.uses_output_ptrs;

        if !used_output_ptrs {
            writeln!(output, "{}fragColor = {};", self.indent_str(), result).unwrap();
        }

        self.indent -= 1;
        writeln!(output, "}}").unwrap();

        Ok(())
    }

    fn lower_shader(&mut self, entry_name: &str, stage: ShaderStage) -> Result<String> {
        self.tuple_structs.clear();
        self.tuple_type_cache.clear();
        self.tuple_counter = 0;
        self.lowered.clear();

        let mut code = String::new();

        // Emit uniforms. These names are host-contract (e.g. Shadertoy's
        // `iResolution`) and must appear verbatim, so validate rather than
        // mangle.
        for uniform in &self.program.uniforms {
            self.validate_io_name(&uniform.name)?;
            writeln!(
                code,
                "layout(set = {}, binding = {}) uniform {} {};",
                uniform.set,
                uniform.binding,
                self.type_to_glsl(&uniform.ty),
                uniform.name
            )
            .unwrap();
        }
        writeln!(code).unwrap();

        // Lower helper functions
        let deps = self.collect_dependencies(entry_name)?;
        for dep_name in &deps {
            if dep_name != entry_name {
                if let Some(&idx) = self.func_index.get(dep_name) {
                    self.lower_function(&self.program.functions[idx].clone(), &mut code)?;
                }
            }
        }

        // Find and lower entry point
        let entry = self.program.entry_points.iter().find(|e| e.name == entry_name).ok_or_else(|| {
            crate::error::CompilerError::GlslError(format!("Entry point '{}' not found", entry_name), None)
        })?;

        self.lower_entry_point(entry, stage, &mut code)?;

        // Build final output
        let mut output = String::new();
        writeln!(output, "#version 450").unwrap();
        writeln!(output, "#extension GL_ARB_shading_language_420pack : enable").unwrap();
        writeln!(output).unwrap();

        // Emit struct definitions
        if !self.tuple_structs.is_empty() {
            writeln!(output, "// Tuple struct definitions").unwrap();
            let mut structs: Vec<_> = self.tuple_structs.iter().collect();
            structs.sort_by_key(|(name, _)| *name);
            for (struct_name, field_types) in structs {
                writeln!(output, "struct {} {{", struct_name).unwrap();
                for (i, field_type) in field_types.iter().enumerate() {
                    writeln!(output, "    {} f{};", field_type, i).unwrap();
                }
                writeln!(output, "}};").unwrap();
            }
            writeln!(output).unwrap();
        }

        output.push_str(&code);
        Ok(output)
    }

    fn collect_dependencies(&self, name: &str) -> Result<Vec<String>> {
        let mut deps = Vec::new();
        let mut visited = HashSet::new();
        self.collect_deps_recursive(name, &mut deps, &mut visited)?;
        Ok(deps)
    }

    fn collect_deps_recursive(
        &self,
        name: &str,
        deps: &mut Vec<String>,
        visited: &mut HashSet<String>,
    ) -> Result<()> {
        if visited.contains(name) {
            return Ok(());
        }
        visited.insert(name.to_string());

        // Check if it's a function
        if let Some(&idx) = self.func_index.get(name) {
            let func = &self.program.functions[idx];
            self.collect_body_deps(&func.body, deps, visited)?;
        }

        // Check if it's an entry point
        for entry in &self.program.entry_points {
            if entry.name == name {
                self.collect_body_deps(&entry.body, deps, visited)?;
            }
        }

        deps.push(name.to_string());
        Ok(())
    }

    fn collect_body_deps(
        &self,
        body: &FuncBody,
        deps: &mut Vec<String>,
        visited: &mut HashSet<String>,
    ) -> Result<()> {
        for (_bid, block) in &body.inner.blocks {
            for &inst_id in &block.insts {
                let inst = body.get_inst(inst_id);
                if let InstKind::Call { func, .. } = &inst.data {
                    if self.func_index.contains_key(func)
                        && crate::builtins::catalog().lookup_lowering(func).is_none()
                    {
                        self.collect_deps_recursive(func, deps, visited)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn lower_function(&mut self, func: &Function, output: &mut String) -> Result<()> {
        if func.linkage_name.is_some() {
            // Skip extern functions
            return Ok(());
        }

        if self.lowered.contains(&func.name) {
            return Ok(());
        }
        self.lowered.insert(func.name.clone());

        let body = &func.body;

        // Build parameter list
        let params: Vec<String> =
            body.params.iter().map(|(_, ty, name)| format!("{} {}", self.type_to_glsl(ty), name)).collect();

        let ret_ty = self.type_to_glsl(&body.return_ty);
        let glsl_name = self.glsl_mangle_tracked(&func.name)?;
        writeln!(output, "{} {}({}) {{", ret_ty, glsl_name, params.join(", ")).unwrap();

        self.indent += 1;
        let result = self.lower_body(body, func.span, output)?;
        writeln!(output, "{}return {};", self.indent_str(), result).unwrap();
        self.indent -= 1;

        writeln!(output, "}}").unwrap();
        writeln!(output).unwrap();

        Ok(())
    }

    fn lower_entry_point(
        &mut self,
        entry: &EntryPoint,
        stage: ShaderStage,
        output: &mut String,
    ) -> Result<()> {
        let body = &entry.body;

        // Emit input declarations. Names are user-written (vertex attribute
        // names, Shadertoy-style uniforms) and must remain verbatim, but we
        // reject illegal chars / reserved names at compile time.
        let mut builtin_assignments = Vec::new();
        for input in &entry.inputs {
            self.validate_io_name(&input.name)?;
            match &input.decoration {
                Some(IoDecoration::Location(loc)) => {
                    writeln!(
                        output,
                        "layout(location = {}) in {} {};",
                        loc,
                        self.type_to_glsl(&input.ty),
                        input.name
                    )
                    .unwrap();
                }
                Some(IoDecoration::BuiltIn(builtin)) => {
                    let gl_var = match builtin {
                        spirv::BuiltIn::Position => {
                            if stage == ShaderStage::Fragment {
                                "gl_FragCoord"
                            } else {
                                "gl_Position"
                            }
                        }
                        spirv::BuiltIn::VertexIndex => "gl_VertexID",
                        spirv::BuiltIn::InstanceIndex => "gl_InstanceID",
                        spirv::BuiltIn::FragCoord => "gl_FragCoord",
                        spirv::BuiltIn::FrontFacing => "gl_FrontFacing",
                        spirv::BuiltIn::PointSize => "gl_PointSize",
                        spirv::BuiltIn::FragDepth => "gl_FragDepth",
                        _ => continue,
                    };
                    builtin_assignments.push((
                        input.name.clone(),
                        self.type_to_glsl(&input.ty),
                        gl_var.to_string(),
                    ));
                }
                None => {}
            }
        }

        // Emit output declarations
        let mut location_outputs: Vec<(usize, u32)> = Vec::new();
        let is_tuple_return = entry.outputs.len() > 1;
        for (i, out) in entry.outputs.iter().enumerate() {
            if let Some(IoDecoration::Location(loc)) = &out.decoration {
                writeln!(
                    output,
                    "layout(location = {}) out {} o{};",
                    loc,
                    self.type_to_glsl(&out.ty),
                    i
                )
                .unwrap();
                location_outputs.push((i, *loc));
            }
        }

        // Build output name map for OutputPtr handling
        let mut output_names: HashMap<usize, String> = HashMap::new();
        for (i, out) in entry.outputs.iter().enumerate() {
            if let Some(IoDecoration::BuiltIn(spirv::BuiltIn::Position)) = &out.decoration {
                output_names.insert(i, "gl_Position".to_string());
            } else if let Some(IoDecoration::Location(_)) = &out.decoration {
                output_names.insert(i, format!("o{}", i));
            }
        }

        writeln!(output).unwrap();
        writeln!(output, "void main() {{").unwrap();
        self.indent += 1;

        // Emit builtin assignments
        for (name, ty, gl_var) in &builtin_assignments {
            writeln!(output, "{}{} {} = {};", self.indent_str(), ty, name, gl_var).unwrap();
        }

        let mut body_ctx = BodyLowerCtx::new(self, body, entry.span);
        // Pre-populate output pointer names so OutputPtr+Store become GLSL assignments
        for (idx, name) in &output_names {
            body_ctx.entry_output_names.insert(*idx, name.clone());
        }
        let result = body_ctx.lower(output)?;
        let used_output_ptrs = body_ctx.uses_output_ptrs;

        if !used_output_ptrs {
            // Legacy path: assign return value to outputs
            for (tuple_idx, _loc) in &location_outputs {
                if is_tuple_return {
                    writeln!(
                        output,
                        "{}o{} = {}.f{};",
                        self.indent_str(),
                        tuple_idx,
                        result,
                        tuple_idx
                    )
                    .unwrap();
                } else {
                    writeln!(output, "{}_out0 = {};", self.indent_str(), result).unwrap();
                }
            }

            if stage == ShaderStage::Vertex {
                for (i, out) in entry.outputs.iter().enumerate() {
                    if let Some(IoDecoration::BuiltIn(spirv::BuiltIn::Position)) = &out.decoration {
                        if is_tuple_return {
                            writeln!(output, "{}gl_Position = {}.f{};", self.indent_str(), result, i)
                                .unwrap();
                        } else {
                            writeln!(output, "{}gl_Position = {};", self.indent_str(), result).unwrap();
                        }
                    }
                }
            }
        }

        self.indent -= 1;
        writeln!(output, "}}").unwrap();

        Ok(())
    }

    /// Lower an SSA function body to GLSL.
    ///
    /// This walks the CFG and generates GLSL code. For now, we use a simple
    /// approach that emits variables for each SSA value and handles control
    /// flow by detecting patterns.
    fn lower_body(&mut self, body: &FuncBody, span: Span, output: &mut String) -> Result<String> {
        let mut body_ctx = BodyLowerCtx::new(self, body, span);
        body_ctx.lower(output)
    }

    fn type_to_glsl(&mut self, ty: &PolyType<TypeName>) -> String {
        match ty {
            PolyType::Constructed(name, args) => match name {
                TypeName::Float(32) => "float".to_string(),
                TypeName::Float(64) => "double".to_string(),
                TypeName::Int(32) => "int".to_string(),
                TypeName::Int(64) => "int64_t".to_string(),
                TypeName::UInt(32) => "uint".to_string(),
                TypeName::UInt(64) => "uint64_t".to_string(),
                TypeName::Bool => "bool".to_string(),
                TypeName::Unit => "void".to_string(),
                TypeName::Tuple(_) => {
                    let elem_types: Vec<String> = args.iter().map(|a| self.type_to_glsl(a)).collect();
                    let sig = elem_types.join(",");

                    if let Some(name) = self.tuple_type_cache.get(&sig) {
                        return name.clone();
                    }

                    let struct_name = format!("T{}", self.tuple_counter);
                    self.tuple_counter += 1;

                    self.tuple_structs.insert(struct_name.clone(), elem_types);
                    self.tuple_type_cache.insert(sig, struct_name.clone());

                    struct_name
                }
                TypeName::Vec => {
                    let n = ty.vec_size().unwrap_or(4);
                    let elem = self.type_to_glsl(ty.elem_type().expect("Vec has elem"));
                    match elem.as_str() {
                        "float" => format!("vec{}", n),
                        "double" => format!("dvec{}", n),
                        "int" => format!("ivec{}", n),
                        "uint" => format!("uvec{}", n),
                        "bool" => format!("bvec{}", n),
                        _ => format!("vec{}", n),
                    }
                }
                TypeName::Mat => {
                    let cols = ty.mat_cols().unwrap_or(4);
                    let rows = ty.mat_rows().unwrap_or(4);
                    let elem = self.type_to_glsl(ty.elem_type().expect("Mat has elem"));
                    match elem.as_str() {
                        "float" => {
                            if rows == cols {
                                format!("mat{}", rows)
                            } else {
                                format!("mat{}x{}", cols, rows)
                            }
                        }
                        "double" => {
                            if rows == cols {
                                format!("dmat{}", rows)
                            } else {
                                format!("dmat{}x{}", cols, rows)
                            }
                        }
                        _ => format!("mat{}", rows),
                    }
                }
                TypeName::Array => {
                    let elem = self.type_to_glsl(ty.elem_type().expect("Array has elem"));
                    // GLSL requires function-parameter arrays to be sized, and
                    // sized constructors need the size too. Emit `T[N]` when
                    // the array size is a concrete `Size(N)`; fall back to
                    // `T[]` only when the size is a variable (rare — mostly
                    // for unmonomorphized types that shouldn't reach here).
                    if let Some(PolyType::Constructed(TypeName::Size(n), _)) = ty.array_size() {
                        format!("{}[{}]", elem, n)
                    } else {
                        format!("{}[]", elem)
                    }
                }
                TypeName::Record(fields) => {
                    panic!("BUG: Record type reached GLSL lowering. Fields: {:?}", fields);
                }
                _ => panic!("BUG: unsupported type in GLSL lowering: {:?}", ty),
            },
            _ => panic!("BUG: unsupported type in GLSL lowering: {:?}", ty),
        }
    }

    fn indent_str(&self) -> String {
        "    ".repeat(self.indent)
    }

    /// Mangle a compiler-internal name with collision detection.
    ///
    /// The sanitizer itself is injective, so two different originals should
    /// never produce the same mangled output. If they do, it's a bug in the
    /// scheme (or accidental double-mangling) and we bail loudly rather
    /// than silently emit colliding GLSL.
    fn glsl_mangle_tracked(&mut self, name: &str) -> Result<String> {
        let mangled = glsl_mangle(name);
        match self.mangled_names.get(&mangled) {
            Some(prev) if prev != name => bail_glsl!(
                "GLSL identifier collision: '{}' and '{}' both mangle to '{}'",
                prev,
                name,
                mangled
            ),
            Some(_) => {}
            None => {
                self.mangled_names.insert(mangled.clone(), name.to_string());
            }
        }
        Ok(mangled)
    }

    /// Validate a host-contract identifier; wraps [`validate_glsl_identifier`]
    /// with a `GlslError` conversion.
    fn validate_io_name(&self, name: &str) -> Result<()> {
        validate_glsl_identifier(name).map_err(|msg| crate::err_glsl!("{}", msg))
    }
}

/// Format a ValueId as a short, valid GLSL identifier.
fn glsl_var(id: ValueId) -> String {
    use crate::ssa::framework::Key;
    let ffi = id.data().as_ffi();
    let idx = ffi & 0xFFFFFFFF;
    let ver = ffi >> 32;
    format!("v{}_{}", idx, ver)
}

/// Context for lowering a single function body.
struct BodyLowerCtx<'a, 'b> {
    ctx: &'a mut LowerCtx<'b>,
    body: &'a FuncBody,
    /// Map from ValueId to GLSL expression string.
    value_map: HashMap<ValueId, String>,
    /// Set of declared variables
    declared: HashSet<String>,
    /// Map from OutputPtr ValueIds to their GLSL output variable names.
    output_ptrs: HashMap<crate::ssa::types::PlaceId, String>,
    /// Map from output index to GLSL variable name (set by entry point lowering).
    entry_output_names: HashMap<usize, String>,
    /// Whether this body used OutputPtr+Store (so post-body assignment is skipped).
    uses_output_ptrs: bool,
    /// Span of the instruction currently being lowered. Set by `lower_inst`
    /// before recursing into helpers; consumed by `bail_glsl_at!` so backend
    /// errors blame the source line of the originating expression.
    current_span: Option<Span>,
    /// Function-level span fallback when an instruction has no span of its own.
    func_span: Span,
}

impl<'a, 'b> BodyLowerCtx<'a, 'b> {
    fn new(ctx: &'a mut LowerCtx<'b>, body: &'a FuncBody, func_span: Span) -> Self {
        BodyLowerCtx {
            ctx,
            body,
            value_map: HashMap::new(),
            declared: HashSet::new(),
            output_ptrs: HashMap::new(),
            entry_output_names: HashMap::new(),
            uses_output_ptrs: false,
            current_span: None,
            func_span,
        }
    }

    /// Span used to blame an instruction's lowering errors. Falls back to
    /// the function-level span if the instruction has no span of its own.
    fn blame_span(&self) -> Span {
        self.current_span.unwrap_or(self.func_span)
    }

    fn lower(&mut self, output: &mut String) -> Result<String> {
        // Map function parameters to their names
        for (value_id, _, name) in &self.body.params {
            self.value_map.insert(*value_id, name.clone());
            self.declared.insert(name.clone());
        }

        // Convert CFG to structured control flow tree, then emit
        let nodes = crate::structured::structurize(self.body);
        self.emit_nodes(&nodes, output)
    }

    /// Emit structured control flow nodes as GLSL.
    fn emit_nodes(&mut self, nodes: &[crate::structured::Node], output: &mut String) -> Result<String> {
        use crate::structured::Node;
        let mut result_var = String::new();

        for node in nodes {
            match node {
                Node::Inst(inst_id) => {
                    let inst = self.body.get_inst(*inst_id);

                    // Special-case the array-update intrinsics: they don't fit the
                    // "one expression = one statement" model.
                    if self.try_emit_array_intrinsic(inst, output)? {
                        continue;
                    }

                    let is_side_effect =
                        matches!(inst.data, InstKind::OutputSlot { .. } | InstKind::Store { .. });
                    let expr = self.lower_inst(inst, output)?;

                    if let Some(result) = inst.result {
                        if !is_side_effect {
                            let var_name = glsl_var(result);
                            if self.declared.contains(&var_name) {
                                writeln!(output, "{}{} = {};", self.ctx.indent_str(), var_name, expr)
                                    .unwrap();
                            } else {
                                let ty = self.ctx.type_to_glsl(self.body.inner.value_type(result));
                                writeln!(
                                    output,
                                    "{}{} {} = {};",
                                    self.ctx.indent_str(),
                                    ty,
                                    var_name,
                                    expr
                                )
                                .unwrap();
                                self.declared.insert(var_name.clone());
                            }
                            self.value_map.insert(result, var_name);
                        }
                    }
                }

                Node::Assign { target, value } => {
                    let val = self.get_value(*value)?;
                    let var_name = glsl_var(*target);
                    // Elide redundant self-copies. The ArrayWithInPlace path
                    // aliases the result's GLSL var to the source array's GLSL
                    // var; the resulting `x = x;` is skipped here.
                    if val == var_name {
                        self.value_map.insert(*target, var_name);
                        continue;
                    }
                    if self.declared.contains(&var_name) {
                        writeln!(output, "{}{} = {};", self.ctx.indent_str(), var_name, val).unwrap();
                    } else {
                        let ty = self.ctx.type_to_glsl(self.body.inner.value_type(*target));
                        writeln!(output, "{}{} {} = {};", self.ctx.indent_str(), ty, var_name, val)
                            .unwrap();
                        self.declared.insert(var_name.clone());
                    }
                    self.value_map.insert(*target, var_name);
                }

                Node::If {
                    cond,
                    then_body,
                    then_args,
                    else_body,
                    else_args,
                    merge_params,
                } => {
                    // Declare merge params before the if
                    for param in merge_params {
                        let var_name = glsl_var(*param);
                        if !self.declared.contains(&var_name) {
                            let ty = self.ctx.type_to_glsl(self.body.inner.value_type(*param));
                            writeln!(output, "{}{} {};", self.ctx.indent_str(), ty, var_name).unwrap();
                            self.declared.insert(var_name.clone());
                        }
                        self.value_map.insert(*param, var_name);
                    }

                    let cond_val = self.get_value(*cond)?;
                    writeln!(output, "{}if ({}) {{", self.ctx.indent_str(), cond_val).unwrap();
                    self.ctx.indent += 1;

                    self.emit_nodes(then_body, output)?;
                    for (param, arg) in merge_params.iter().zip(then_args.iter()) {
                        let arg_val = self.get_value(*arg)?;
                        let var_name = glsl_var(*param);
                        writeln!(output, "{}{} = {};", self.ctx.indent_str(), var_name, arg_val).unwrap();
                    }

                    self.ctx.indent -= 1;
                    writeln!(output, "{}}} else {{", self.ctx.indent_str()).unwrap();
                    self.ctx.indent += 1;

                    self.emit_nodes(else_body, output)?;
                    for (param, arg) in merge_params.iter().zip(else_args.iter()) {
                        let arg_val = self.get_value(*arg)?;
                        let var_name = glsl_var(*param);
                        writeln!(output, "{}{} = {};", self.ctx.indent_str(), var_name, arg_val).unwrap();
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
                    // Declare and initialize state variables
                    for (var, init) in state_vars.iter().zip(init_args.iter()) {
                        let init_val = self.get_value(*init)?;
                        let var_name = glsl_var(*var);
                        let ty = self.ctx.type_to_glsl(self.body.inner.value_type(*var));
                        writeln!(
                            output,
                            "{}{} {} = {};",
                            self.ctx.indent_str(),
                            ty,
                            var_name,
                            init_val
                        )
                        .unwrap();
                        self.value_map.insert(*var, var_name.clone());
                        self.declared.insert(var_name);
                    }

                    // First evaluation of header (computes condition)
                    self.emit_header_insts(header_insts, output)?;

                    let cond_val = self.get_value(*cond)?;
                    let cond_expr = if *cond_is_continue { cond_val } else { format!("!({})", cond_val) };
                    writeln!(output, "{}while ({}) {{", self.ctx.indent_str(), cond_expr).unwrap();
                    self.ctx.indent += 1;

                    self.emit_nodes(body, output)?;

                    // Re-evaluate header (update condition for next iteration)
                    self.emit_header_insts(header_insts, output)?;

                    self.ctx.indent -= 1;
                    writeln!(output, "{}}}", self.ctx.indent_str()).unwrap();
                }

                Node::Return(val) => {
                    if let Some(v) = val {
                        result_var = self.get_value(*v)?;
                    }
                }
            }
        }

        Ok(result_var)
    }

    /// Emit header instructions, using assignment for already-declared variables.
    fn emit_header_insts(
        &mut self,
        insts: &[crate::ssa::framework::InstId],
        output: &mut String,
    ) -> Result<()> {
        for &inst_id in insts {
            let inst = self.body.get_inst(inst_id);
            let expr = self.lower_inst(inst, output)?;
            if let Some(result) = inst.result {
                let var_name = glsl_var(result);
                if self.declared.contains(&var_name) {
                    writeln!(output, "{}{} = {};", self.ctx.indent_str(), var_name, expr).unwrap();
                } else {
                    let ty = self.ctx.type_to_glsl(self.body.inner.value_type(result));
                    writeln!(output, "{}{} {} = {};", self.ctx.indent_str(), ty, var_name, expr).unwrap();
                    self.declared.insert(var_name.clone());
                }
                self.value_map.insert(result, var_name);
            }
        }
        Ok(())
    }

    fn lower_inst(&mut self, inst: &WynInstNode, _output: &mut String) -> Result<String> {
        let result_ty = inst.result.map(|r| self.body.inner.value_type(r));
        self.current_span = inst.span;
        match &inst.data {
            InstKind::Int(s) => Ok(s.clone()),
            InstKind::Float(s) => {
                if s.contains('.') || s.contains('e') || s.contains('E') {
                    Ok(s.clone())
                } else {
                    Ok(format!("{}.0", s))
                }
            }
            InstKind::Bool(b) => Ok(if *b { "true" } else { "false" }.to_string()),
            InstKind::Unit => {
                unreachable!(
                    "InstKind::Unit should never reach GLSL codegen; unit values are not materializable"
                )
            }

            InstKind::BinOp { op, lhs, rhs } => {
                let l = self.get_value_ref(*lhs)?;
                let r = self.get_value_ref(*rhs)?;
                match op.as_str() {
                    "**" => Ok(format!("pow({}, {})", l, r)),
                    _ => Ok(format!("({} {} {})", l, op, r)),
                }
            }

            InstKind::UnaryOp { op, operand } => {
                let inner = self.get_value_ref(*operand)?;
                Ok(format!("({}{})", op, inner))
            }

            InstKind::Tuple(elems) => {
                if elems.is_empty() {
                    bail_glsl_at!(self.blame_span(), "Empty tuple in GLSL lowering");
                }
                let parts: Result<Vec<_>> = elems.iter().map(|e| self.get_value_ref(*e)).collect();
                let struct_name = self.ctx.type_to_glsl(result_ty.expect("Tuple must have result"));
                Ok(format!("{}({})", struct_name, parts?.join(", ")))
            }

            InstKind::ArrayLit { elements } => {
                let parts: Result<Vec<_>> = elements.iter().map(|e| self.get_value_ref(*e)).collect();
                // GLSL array literal syntax is `ElemType[](v0, v1, ...)` —
                // note the `[]` belongs to the literal, not the type. If
                // we used type_to_glsl on the array type it would already
                // include a `[]` suffix, producing `Elem[][](...)` which
                // is arrays-of-arrays.
                let arr_ty = result_ty.expect("ArrayLit must have result");
                let elem_ty = arr_ty.elem_type().expect("ArrayLit result is an array");
                let elem_str = self.ctx.type_to_glsl(elem_ty);
                Ok(format!("{}[]({})", elem_str, parts?.join(", ")))
            }

            InstKind::Vector(elems) => {
                let parts: Result<Vec<_>> = elems.iter().map(|e| self.get_value_ref(*e)).collect();
                let ty = self.ctx.type_to_glsl(result_ty.expect("Vector must have result"));
                Ok(format!("{}({})", ty, parts?.join(", ")))
            }

            InstKind::Index { base, index } => {
                let base_val = self.get_value_ref(*base)?;
                let index_val = self.get_value_ref(*index)?;
                Ok(format!("{}[{}]", base_val, index_val))
            }

            InstKind::Project { base, index } => {
                let base_val = self.get_value_ref(*base)?;
                let base_ty = match base.as_ssa() {
                    Some(id) => self.body.get_value_type(id),
                    None => bail_glsl_at!(self.blame_span(), "Project base must be SSA value"),
                };

                // Check if it's a vector type - use swizzle
                if matches!(base_ty, PolyType::Constructed(TypeName::Vec, _)) {
                    let swizzle = match index {
                        0 => "x",
                        1 => "y",
                        2 => "z",
                        3 => "w",
                        _ => bail_glsl_at!(self.blame_span(), "Invalid vector swizzle index: {}", index),
                    };
                    Ok(format!("{}.{}", base_val, swizzle))
                } else {
                    // Struct field access
                    Ok(format!("{}.f{}", base_val, index))
                }
            }

            InstKind::Call { func, args } => {
                let arg_strs: Result<Vec<_>> = args.iter().map(|a| self.get_value_ref(*a)).collect();
                let arg_strs = arg_strs?;

                // Check if it's a builtin
                if let Some(impl_) = crate::builtins::catalog().lookup_lowering(func) {
                    self.lower_builtin_call(&impl_, &arg_strs, result_ty.expect("Call must have result"))
                } else {
                    let mangled = self.ctx.glsl_mangle_tracked(func)?;
                    Ok(format!("{}({})", mangled, arg_strs.join(", ")))
                }
            }

            InstKind::Intrinsic {
                id,
                overload_idx: _,
                args,
            } => {
                let arg_strs: Result<Vec<_>> = args.iter().map(|a| self.get_value_ref(*a)).collect();
                let ssa_args: Vec<ValueId> = args.iter().filter_map(|a| a.as_ssa()).collect();
                let name = crate::builtins::by_id(*id).dispatch_name();
                self.lower_intrinsic(
                    name,
                    &arg_strs?,
                    &ssa_args,
                    result_ty.expect("Intrinsic must have result"),
                )
            }

            InstKind::ArrayRange { .. } => {
                bail_glsl_at!(self.blame_span(), "ArrayRange not supported in GLSL")
            }

            InstKind::Global(name) => Ok(name.clone()),

            InstKind::Extern(linkage) => {
                bail_glsl_at!(
                    self.blame_span(),
                    "Extern functions not supported in GLSL: {}",
                    linkage
                )
            }

            InstKind::Matrix(_) => {
                bail_glsl_at!(
                    self.blame_span(),
                    "Matrix literals not yet implemented in GLSL lowering"
                )
            }

            // TODO: GLSL backend does not support storage places today. To
            // lift this, route `Alloca` to a function-scope GLSL array,
            // teach `Load` / `Store` to walk `PlaceId` → place expression
            // (mirror WGSL's `place_targets`), and implement StorageView
            // via SSBO syntax. Until then the GLSL target rejects any
            // view / Alloca / Load path.
            InstKind::StorageView { .. }
            | InstKind::StorageViewLen { .. }
            | InstKind::ViewIndex { .. }
            | InstKind::Alloca { .. }
            | InstKind::Load { .. } => bail_glsl_at!(
                self.blame_span(),
                "GLSL target does not support storage places (StorageView / ViewIndex / Alloca / Load)"
            ),

            InstKind::Store { place, value } => {
                if let Some(out_name) = self.output_ptrs.get(place) {
                    let val = self.get_value_ref(*value)?;
                    writeln!(_output, "{}{} = {};", self.ctx.indent_str(), out_name, val).unwrap();
                    Ok(String::new())
                } else {
                    bail_glsl_at!(
                        self.blame_span(),
                        "GLSL target: Store target is not an entry-point OutputSlot"
                    )
                }
            }

            InstKind::OutputSlot { index, result } => {
                if let Some(name) = self.entry_output_names.get(&(*index as usize)) {
                    self.output_ptrs.insert(*result, name.clone());
                    self.uses_output_ptrs = true;
                    Ok(String::new())
                } else {
                    bail_glsl_at!(
                        self.blame_span(),
                        "OutputSlot index {} has no corresponding GLSL output",
                        index
                    )
                }
            }

            InstKind::Materialize { .. } | InstKind::DynamicExtract { .. } => {
                bail_glsl_at!(
                    self.blame_span(),
                    "Materialize/DynamicExtract not supported in GLSL (SPIR-V-specific lowering)"
                )
            }
        }
    }

    fn get_value(&self, val: ValueId) -> Result<String> {
        self.value_map
            .get(&val)
            .cloned()
            .ok_or_else(|| crate::err_glsl_at!(self.blame_span(), "Value {:?} not found in value_map", val))
    }

    fn get_value_ref(&self, vr: ValueRef) -> Result<String> {
        match vr {
            ValueRef::Ssa(id) => self.get_value(id),
            ValueRef::Const(c) => match c {
                ConstantValue::I32(v) => Ok(format!("{}", v)),
                ConstantValue::U32(v) => Ok(format!("{}u", v)),
                ConstantValue::F32(bits) => {
                    let f = f32::from_bits(bits);
                    let s = format!("{}", f);
                    if s.contains('.') || s.contains('e') || s.contains('E') {
                        Ok(s)
                    } else {
                        Ok(format!("{}.0", s))
                    }
                }
                ConstantValue::Bool(b) => Ok(if b { "true" } else { "false" }.to_string()),
            },
        }
    }

    /// Intercept the array-update intrinsics — `Uninit`, `ArrayWith`,
    /// `ArrayWithInPlace` — which don't fit the expression-returns-a-string
    /// model. They emit statements directly and alias the result's GLSL
    /// variable to the source array (for ArrayWithInPlace) or to a freshly
    /// declared local (for Uninit / ArrayWith).
    ///
    /// Returns `true` when the inst was handled (caller should skip the
    /// generic path), `false` otherwise.
    fn try_emit_array_intrinsic(&mut self, inst: &WynInstNode, output: &mut String) -> Result<bool> {
        use crate::ssa::types::InstKind;
        let (intrinsic_name, args) = match &inst.data {
            InstKind::Call { func, args } => (func.as_str(), args),
            _ => return Ok(false),
        };
        let intr = match intrinsic_name {
            INTRINSIC_UNINIT => Intrinsic::Uninit,
            INTRINSIC_ARRAY_WITH => Intrinsic::ArrayWith,
            INTRINSIC_ARRAY_WITH_INPLACE => Intrinsic::ArrayWithInPlace,
            _ => return Ok(false),
        };
        let result = inst.result.ok_or_else(|| {
            crate::err_glsl_at!(self.blame_span(), "Array intrinsic {:?} missing result", intr)
        })?;
        let var_name = glsl_var(result);
        let result_ty = self.body.inner.value_type(result).clone();
        let sized_decl_prefix =
            self.sized_array_decl_prefix(&result_ty).unwrap_or_else(|| self.ctx.type_to_glsl(&result_ty));
        match intr {
            Intrinsic::Uninit => {
                // GLSL ES 3.00+ allows uninitialized locals; matches SPIR-V
                // Function-storage semantics (contents undefined until written).
                // The SOAC contract is that the array is fully overwritten before
                // it's read.
                writeln!(
                    output,
                    "{}{} {};",
                    self.ctx.indent_str(),
                    sized_decl_prefix,
                    var_name
                )
                .unwrap();
                self.declared.insert(var_name.clone());
                self.value_map.insert(result, var_name);
                Ok(true)
            }
            Intrinsic::ArrayWithInPlace => {
                // Alias the result's GLSL var to the source array's GLSL var:
                // we're mutating in place. The subsequent Node::Assign that
                // would have copied this result into the loop-carried variable
                // becomes a self-copy and is elided.
                if args.len() != 3 {
                    bail_glsl_at!(self.blame_span(), "ArrayWithInPlace expects 3 args");
                }
                let arr = self.get_value_ref(args[0])?;
                let idx = self.get_value_ref(args[1])?;
                let val = self.get_value_ref(args[2])?;
                writeln!(output, "{}{}[{}] = {};", self.ctx.indent_str(), arr, idx, val).unwrap();
                self.value_map.insert(result, arr);
                Ok(true)
            }
            Intrinsic::ArrayWith => {
                // Functional update: declare a fresh local, copy the source
                // array into it, then patch the one element.
                if args.len() != 3 {
                    bail_glsl_at!(self.blame_span(), "ArrayWith expects 3 args");
                }
                let arr = self.get_value_ref(args[0])?;
                let idx = self.get_value_ref(args[1])?;
                let val = self.get_value_ref(args[2])?;
                writeln!(
                    output,
                    "{}{} {} = {};",
                    self.ctx.indent_str(),
                    sized_decl_prefix,
                    var_name,
                    arr
                )
                .unwrap();
                writeln!(
                    output,
                    "{}{}[{}] = {};",
                    self.ctx.indent_str(),
                    var_name,
                    idx,
                    val
                )
                .unwrap();
                self.declared.insert(var_name.clone());
                self.value_map.insert(result, var_name);
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    /// Format a type as a GLSL array declaration prefix with known size,
    /// e.g. `float[8]`. Returns `None` for non-sized or non-array types.
    fn sized_array_decl_prefix(&mut self, ty: &PolyType<TypeName>) -> Option<String> {
        use crate::types::TypeExt;
        let size = ty.array_size()?;
        let n = match size {
            PolyType::Constructed(TypeName::Size(n), _) => *n as u32,
            _ => return None,
        };
        let elem = ty.elem_type()?;
        Some(format!("{}[{}]", self.ctx.type_to_glsl(elem), n))
    }

    fn lower_builtin_call(
        &mut self,
        impl_: &BuiltinLowering,
        args: &[String],
        ret_ty: &PolyType<TypeName>,
    ) -> Result<String> {
        match impl_ {
            BuiltinLowering::PrimOp(op) => self.lower_primop(op, args, ret_ty),
            BuiltinLowering::Intrinsic(intr) => match intr {
                Intrinsic::Length => Ok(format!("int({}.length())", args[0])),
                Intrinsic::Uninit | Intrinsic::ArrayWith | Intrinsic::ArrayWithInPlace => {
                    bail_glsl_at!(
                        self.blame_span(),
                        "Intrinsic {:?} must be handled at Node::Inst level",
                        intr
                    )
                }
                Intrinsic::Slice => {
                    bail_glsl_at!(self.blame_span(), "GLSL backend does not support array slicing")
                }
                Intrinsic::StorageLen => {
                    bail_glsl_at!(
                        self.blame_span(),
                        "GLSL backend does not support storage-buffer length queries"
                    )
                }
                Intrinsic::ExtInstSplat { .. } => {
                    bail_glsl_at!(
                        self.blame_span(),
                        "GLSL backend does not support ExtInstSplat (vec mix/clamp/smoothstep)"
                    )
                }
                Intrinsic::ThreadId => {
                    bail_glsl_at!(
                        self.blame_span(),
                        "GLSL backend does not support compute-shader thread ID"
                    )
                }
            },
            BuiltinLowering::LinkedSpirv(name) => {
                bail_glsl_at!(
                    self.blame_span(),
                    "Linked SPIR-V function '{}' not supported in GLSL",
                    name
                )
            }
            BuiltinLowering::NotLowered => {
                bail_glsl_at!(
                    self.blame_span(),
                    "NotLowered builtin should not reach backend dispatch"
                )
            }
        }
    }

    fn lower_primop(
        &mut self,
        op: &PrimOp,
        args: &[String],
        ret_ty: &PolyType<TypeName>,
    ) -> Result<String> {
        use PrimOp::*;
        match op {
            GlslExt(id) => {
                let func = glsl_ext_to_name(*id);
                Ok(format!("{}({})", func, args.join(", ")))
            }
            Dot => Ok(format!("dot({}, {})", args[0], args[1])),
            OuterProduct => Ok(format!("outerProduct({}, {})", args[0], args[1])),
            MatrixTimesMatrix | MatrixTimesVector | VectorTimesMatrix => {
                Ok(format!("({} * {})", args[0], args[1]))
            }
            VectorTimesScalar | MatrixTimesScalar => Ok(format!("({} * {})", args[0], args[1])),
            FAdd | IAdd => Ok(format!("({} + {})", args[0], args[1])),
            FSub | ISub => Ok(format!("({} - {})", args[0], args[1])),
            FMul | IMul => Ok(format!("({} * {})", args[0], args[1])),
            FDiv | SDiv | UDiv => Ok(format!("({} / {})", args[0], args[1])),
            FRem | FMod | SRem | SMod | UMod => Ok(format!("mod({}, {})", args[0], args[1])),
            FOrdEqual | IEqual => Ok(format!("({} == {})", args[0], args[1])),
            FOrdNotEqual | INotEqual => Ok(format!("({} != {})", args[0], args[1])),
            FOrdLessThan | SLessThan | ULessThan => Ok(format!("({} < {})", args[0], args[1])),
            FOrdGreaterThan | SGreaterThan | UGreaterThan => Ok(format!("({} > {})", args[0], args[1])),
            FOrdLessThanEqual | SLessThanEqual | ULessThanEqual => {
                Ok(format!("({} <= {})", args[0], args[1]))
            }
            FOrdGreaterThanEqual | SGreaterThanEqual | UGreaterThanEqual => {
                Ok(format!("({} >= {})", args[0], args[1]))
            }
            BitwiseAnd => Ok(format!("({} & {})", args[0], args[1])),
            BitwiseOr => Ok(format!("({} | {})", args[0], args[1])),
            BitwiseXor => Ok(format!("({} ^ {})", args[0], args[1])),
            Not => Ok(format!("(~{})", args[0])),
            ShiftLeftLogical => Ok(format!("({} << {})", args[0], args[1])),
            ShiftRightArithmetic | ShiftRightLogical => Ok(format!("({} >> {})", args[0], args[1])),
            FPToSI => Ok(format!("int({})", args[0])),
            FPToUI => Ok(format!("uint({})", args[0])),
            SIToFP | UIToFP => Ok(format!("float({})", args[0])),
            FPConvert => Ok(format!("float({})", args[0])),
            SConvert => {
                let target = self.ctx.type_to_glsl(ret_ty);
                Ok(format!("{}({})", target, args[0]))
            }
            UConvert => {
                let target = self.ctx.type_to_glsl(ret_ty);
                Ok(format!("{}({})", target, args[0]))
            }
            IsNan => Ok(format!("isnan({})", args[0])),
            IsInf => Ok(format!("isinf({})", args[0])),
            Bitcast => match ret_ty {
                PolyType::Constructed(TypeName::Int(32), _) => Ok(format!("floatBitsToInt({})", args[0])),
                PolyType::Constructed(TypeName::UInt(32), _) => Ok(format!("floatBitsToUint({})", args[0])),
                PolyType::Constructed(TypeName::Float(32), _) => Ok(format!("intBitsToFloat({})", args[0])),
                _ => bail_glsl_at!(
                    self.blame_span(),
                    "Unsupported bitcast target type: {}",
                    self.ctx.type_to_glsl(ret_ty)
                ),
            },
        }
    }

    fn lower_intrinsic(
        &mut self,
        name: &str,
        args: &[String],
        _arg_ids: &[ValueId],
        ret_ty: &PolyType<TypeName>,
    ) -> Result<String> {
        // Catalog-driven dispatch for `_w_intrinsic_*` builtins.
        if let Some(impl_) = crate::builtins::catalog().lookup_lowering(name) {
            return self.lower_builtin_call(&impl_, args, ret_ty);
        }
        bail_glsl_at!(self.blame_span(), "Unknown intrinsic: {}", name)
    }
}

/// Map GLSLstd450 extended instruction opcodes to GLSL function names
fn glsl_ext_to_name(id: u32) -> &'static str {
    match id {
        1 => "round",
        2 => "roundEven",
        3 => "trunc",
        4 => "abs",
        5 => "abs",
        6 => "sign",
        7 => "sign",
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
        25 => "atan",
        26 => "pow",
        27 => "exp",
        28 => "log",
        29 => "exp2",
        30 => "log2",
        31 => "sqrt",
        32 => "inversesqrt",
        33 => "determinant",
        34 => "inverse",
        37 => "min",
        38 => "min",
        39 => "min",
        40 => "max",
        41 => "max",
        42 => "max",
        43 => "clamp",
        44 => "clamp",
        45 => "clamp",
        46 => "mix",
        48 => "step",
        49 => "smoothstep",
        50 => "fma",
        53 => "ldexp",
        66 => "length",
        67 => "distance",
        68 => "cross",
        69 => "normalize",
        70 => "faceforward",
        71 => "reflect",
        72 => "refract",
        _ => "/* unknown_glsl_ext */",
    }
}
