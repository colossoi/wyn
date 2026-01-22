//! GLSL Lowering
//!
//! This module converts MIR to GLSL shader source code.
//! It generates separate strings for vertex and fragment shaders.

use crate::ast::TypeName;
use crate::bail_glsl;
use crate::error::Result;
use crate::impl_source::{BuiltinImpl, ImplSource, PrimOp};
use crate::lowering_common::ShaderStage;
use crate::mir::{ArrayBacking, Body, Def, ExecutionModel, Expr, ExprId, LocalId, LoopKind, Program};
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

/// Lower a MIR program to GLSL
pub fn lower(program: &Program) -> Result<GlslOutput> {
    let mut ctx = LowerCtx::new(program);
    ctx.lower_program()
}

/// Lower a MIR program to Shadertoy-compatible GLSL
/// Returns just the fragment shader with mainImage entry point
pub fn lower_shadertoy(program: &Program) -> Result<String> {
    let mut ctx = LowerCtx::new(program);
    ctx.lower_shadertoy()
}

/// Context for lowering MIR to GLSL
struct LowerCtx<'a> {
    program: &'a Program,
    /// Map from definition name to its index
    def_index: HashMap<String, usize>,
    /// Functions that have been lowered
    lowered: HashSet<String>,
    /// Builtin implementations
    impl_source: ImplSource,
    /// Current indentation level
    indent: usize,
    /// Tuple types that need struct definitions (keyed by struct name)
    tuple_structs: HashMap<String, Vec<String>>,
    /// Counter for unique tuple struct names
    tuple_counter: usize,
    /// Cache from tuple type signature to struct name
    tuple_type_cache: HashMap<String, String>,
    /// Variables declared in the current function (to avoid redeclarations)
    declared_vars: HashSet<String>,
    /// Counter for generating unique temporary names
    temp_counter: usize,
}

impl<'a> LowerCtx<'a> {
    fn new(program: &'a Program) -> Self {
        let mut def_index = HashMap::new();
        for (i, def) in program.defs.iter().enumerate() {
            let name = match def {
                Def::Function { name, .. } => name,
                Def::Constant { name, .. } => name,
                Def::Uniform { name, .. } => name,
                Def::Storage { name, .. } => name,
                Def::EntryPoint { name, .. } => name,
            };
            def_index.insert(name.clone(), i);
        }

        LowerCtx {
            program,
            def_index,
            lowered: HashSet::new(),
            impl_source: ImplSource::default(),
            indent: 0,
            tuple_structs: HashMap::new(),
            tuple_counter: 0,
            tuple_type_cache: HashMap::new(),
            declared_vars: HashSet::new(),
            temp_counter: 0,
        }
    }

    fn fresh_id(&mut self) -> usize {
        let id = self.temp_counter;
        self.temp_counter += 1;
        id
    }

    /// Check if a type is a struct (tuple or record) that can't be used in ternary
    fn is_struct_type(&self, ty: &PolyType<TypeName>) -> bool {
        matches!(
            ty,
            PolyType::Constructed(TypeName::Tuple(_), _) | PolyType::Constructed(TypeName::Record(_), _)
        )
    }

    fn lower_program(&mut self) -> Result<GlslOutput> {
        let mut vertex_shader = None;
        let mut fragment_shader = None;

        // Find entry points and generate shaders
        for def in &self.program.defs {
            if let Def::EntryPoint {
                name,
                execution_model,
                ..
            } = def
            {
                match execution_model {
                    ExecutionModel::Vertex => {
                        vertex_shader = Some(self.lower_shader(name, ShaderStage::Vertex)?);
                    }
                    ExecutionModel::Fragment => {
                        fragment_shader = Some(self.lower_shader(name, ShaderStage::Fragment)?);
                    }
                    ExecutionModel::Compute { .. } => {
                        bail_glsl!("Compute shaders are not supported in GLSL output format");
                    }
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
        let mut entry_name = None;
        for def in &self.program.defs {
            if let Def::EntryPoint {
                name,
                execution_model: ExecutionModel::Fragment,
                ..
            } = def
            {
                entry_name = Some(name.clone());
                break;
            }
        }

        let entry_name = entry_name.ok_or_else(|| {
            crate::error::CompilerError::GlslError("No fragment entry point found".to_string(), None)
        })?;

        // Clear state
        self.tuple_structs.clear();
        self.tuple_type_cache.clear();
        self.tuple_counter = 0;
        self.lowered.clear();

        // First pass: lower code to discover tuple types
        let mut code = String::new();

        // Collect dependencies
        let deps = self.collect_dependencies(&entry_name)?;

        // Skip uniforms - Shadertoy provides iResolution, iTime, iMouse, etc.

        // Emit helper functions (non-entry points first)
        for dep_name in &deps {
            if dep_name != &entry_name {
                if let Some(&idx) = self.def_index.get(dep_name) {
                    self.lower_def(&self.program.defs[idx].clone(), &mut code)?;
                }
            }
        }

        // Emit Shadertoy entry point
        if let Some(&idx) = self.def_index.get(&entry_name) {
            self.lower_shadertoy_entry_point(&self.program.defs[idx].clone(), &mut code)?;
        }

        // Build final output with struct definitions first
        let mut output = String::new();

        // Shadertoy comment header
        writeln!(output, "// Generated by Wyn compiler for Shadertoy").unwrap();
        writeln!(output).unwrap();

        // Emit struct definitions for tuple types
        if !self.tuple_structs.is_empty() {
            for (struct_name, field_types) in &self.tuple_structs {
                writeln!(output, "struct {} {{", struct_name).unwrap();
                for (i, field_type) in field_types.iter().enumerate() {
                    writeln!(output, "    {} _{};", field_type, i).unwrap();
                }
                writeln!(output, "}};").unwrap();
            }
            writeln!(output).unwrap();
        }

        // Constants are emitted by lower_def when processing dependencies

        // Append the code
        output.push_str(&code);

        Ok(output)
    }

    fn lower_shadertoy_entry_point(&mut self, def: &Def, output: &mut String) -> Result<()> {
        if let Def::EntryPoint { inputs, body, .. } = def {
            self.declared_vars.clear();

            // Find the fragCoord parameter (builtin position in fragment shader)
            let mut frag_coord_name = None;
            for input in inputs {
                if let Some(crate::mir::IoDecoration::BuiltIn(spirv::BuiltIn::Position)) = &input.decoration
                {
                    frag_coord_name = Some(input.name.clone());
                }
            }

            // Shadertoy entry point signature - use _st_fragCoord to avoid collision
            writeln!(
                output,
                "void mainImage(out vec4 fragColor, in vec2 _st_fragCoord) {{"
            )
            .unwrap();
            self.indent += 1;

            // If the shader expects vec4 fragCoord, convert from vec2
            // Pre-flip Y so that Vulkan's Y-flip in the shader cancels out
            // (Vulkan: Y=0 at top, Shadertoy: Y=0 at bottom)
            if let Some(ref name) = frag_coord_name {
                writeln!(
                    output,
                    "{}vec4 {} = vec4(_st_fragCoord.x, iResolution.y - _st_fragCoord.y, 0.0, 1.0);",
                    self.indent_str(),
                    name
                )
                .unwrap();
                self.declared_vars.insert(name.clone());
            }

            let result = self.lower_expr(body, body.root, output)?;

            // Write to fragColor output
            writeln!(output, "{}fragColor = {};", self.indent_str(), result).unwrap();

            self.indent -= 1;
            writeln!(output, "}}").unwrap();
        }
        Ok(())
    }

    fn lower_shader(&mut self, entry_name: &str, stage: ShaderStage) -> Result<String> {
        // Clear tuple structs from previous shader
        self.tuple_structs.clear();
        self.tuple_type_cache.clear();
        self.tuple_counter = 0;

        // First pass: lower code to discover tuple types
        let mut code = String::new();

        // Collect dependencies and emit them
        self.lowered.clear();
        let deps = self.collect_dependencies(entry_name)?;

        // Emit uniforms
        for def in &self.program.defs {
            if let Def::Uniform {
                name,
                ty,
                set,
                binding,
                ..
            } = def
            {
                writeln!(
                    code,
                    "layout(set = {}, binding = {}) uniform {} {};",
                    set,
                    binding,
                    self.type_to_glsl(ty),
                    name
                )
                .unwrap();
            }
        }
        writeln!(code).unwrap();

        // Emit helper functions (non-entry points first)
        for dep_name in &deps {
            if dep_name != entry_name {
                if let Some(&idx) = self.def_index.get(dep_name) {
                    self.lower_def(&self.program.defs[idx].clone(), &mut code)?;
                }
            }
        }

        // Emit entry point
        if let Some(&idx) = self.def_index.get(entry_name) {
            self.lower_entry_point(&self.program.defs[idx].clone(), stage, &mut code)?;
        }

        // Now build the final output with struct definitions first
        let mut output = String::new();

        // GLSL version and extensions
        writeln!(output, "#version 450").unwrap();
        writeln!(output, "#extension GL_ARB_shading_language_420pack : enable").unwrap();
        writeln!(output).unwrap();

        // Emit struct definitions for tuple types
        if !self.tuple_structs.is_empty() {
            writeln!(output, "// Tuple struct definitions").unwrap();
            // Sort by name for deterministic output
            let mut structs: Vec<_> = self.tuple_structs.iter().collect();
            structs.sort_by_key(|(name, _)| *name);
            for (struct_name, field_types) in structs {
                writeln!(output, "struct {} {{", struct_name).unwrap();
                for (i, field_type) in field_types.iter().enumerate() {
                    writeln!(output, "    {} _{};", field_type, i).unwrap();
                }
                writeln!(output, "}};").unwrap();
            }
            writeln!(output).unwrap();
        }

        // Append the code
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

        if let Some(&idx) = self.def_index.get(name) {
            let def = &self.program.defs[idx];
            match def {
                Def::Function { body, .. } => {
                    self.collect_body_deps(body, deps, visited)?;
                }
                Def::Constant { body, .. } => {
                    self.collect_body_deps(body, deps, visited)?;
                }
                Def::EntryPoint { body, .. } => {
                    self.collect_body_deps(body, deps, visited)?;
                }
                Def::Uniform { .. } => {}
                Def::Storage { .. } => {}
            }
        }

        deps.push(name.to_string());
        Ok(())
    }

    fn collect_body_deps(
        &self,
        body: &Body,
        deps: &mut Vec<String>,
        visited: &mut HashSet<String>,
    ) -> Result<()> {
        // Collect deps from all expressions in the body
        for expr in body.iter_exprs() {
            self.collect_expr_deps(body, expr, deps, visited)?;
        }
        Ok(())
    }

    fn collect_expr_deps(
        &self,
        _body: &Body,
        expr: &Expr,
        deps: &mut Vec<String>,
        visited: &mut HashSet<String>,
    ) -> Result<()> {
        match expr {
            Expr::Call { func, .. } => {
                // If it's a user function (not a builtin), collect it
                if self.def_index.contains_key(func) && self.impl_source.get(func).is_none() {
                    self.collect_deps_recursive(func, deps, visited)?;
                }
            }
            Expr::Global(name) => {
                // Check if this references a constant
                if let Some(&idx) = self.def_index.get(name) {
                    if matches!(self.program.defs[idx], Def::Constant { .. }) {
                        self.collect_deps_recursive(name, deps, visited)?;
                    }
                }
            }
            // Other expressions don't introduce dependencies
            _ => {}
        }
        Ok(())
    }

    fn lower_def(&mut self, def: &Def, output: &mut String) -> Result<()> {
        match def {
            Def::EntryPoint { .. } => {
                // Entry points handled separately via lower_shader
                return Ok(());
            }
            Def::Function {
                name,
                params,
                ret_type,
                body,
                ..
            } => {
                // Clear declared variables for this function
                self.declared_vars.clear();

                // Function signature
                write!(output, "{} {}(", self.type_to_glsl(ret_type), name).unwrap();
                for (i, &param_id) in params.iter().enumerate() {
                    if i > 0 {
                        write!(output, ", ").unwrap();
                    }
                    let param = body.get_local(param_id);
                    write!(output, "{} {}", self.type_to_glsl(&param.ty), param.name).unwrap();
                    // Track params as declared
                    self.declared_vars.insert(param.name.clone());
                }
                writeln!(output, ") {{").unwrap();

                self.indent += 1;
                let result = self.lower_expr(body, body.root, output)?;
                writeln!(output, "{}return {};", self.indent_str(), result).unwrap();
                self.indent -= 1;

                writeln!(output, "}}").unwrap();
                writeln!(output).unwrap();
            }
            Def::Constant { name, ty, body, .. } => {
                write!(output, "const {} {} = ", self.type_to_glsl(ty), name).unwrap();
                let val = self.lower_expr(body, body.root, output)?;
                writeln!(output, "{};", val).unwrap();
            }
            Def::Uniform { .. } => {
                // Already emitted at top of shader
            }
            Def::Storage { .. } => {
                // Already emitted at top of shader
            }
        }
        Ok(())
    }

    fn lower_entry_point(&mut self, def: &Def, stage: ShaderStage, output: &mut String) -> Result<()> {
        if let Def::EntryPoint {
            inputs,
            outputs,
            body,
            ..
        } = def
        {
            // Clear declared variables for this entry point
            self.declared_vars.clear();

            // Collect input declarations and builtin mappings
            let mut builtin_assignments = Vec::new();
            for input in inputs {
                match &input.decoration {
                    Some(crate::mir::IoDecoration::Location(loc)) => {
                        writeln!(
                            output,
                            "layout(location = {}) in {} {};",
                            loc,
                            self.type_to_glsl(&input.ty),
                            input.name
                        )
                        .unwrap();
                    }
                    Some(crate::mir::IoDecoration::BuiltIn(builtin)) => {
                        // Map builtin to GLSL gl_* variable
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
            let mut location_outputs: Vec<(usize, u32)> = Vec::new(); // (output_index, location)
            let is_tuple_return = outputs.len() > 1;
            for (i, out) in outputs.iter().enumerate() {
                if let Some(crate::mir::IoDecoration::Location(loc)) = &out.decoration {
                    writeln!(
                        output,
                        "layout(location = {}) out {} _out{};",
                        loc,
                        self.type_to_glsl(&out.ty),
                        i
                    )
                    .unwrap();
                    location_outputs.push((i, *loc));
                }
            }

            writeln!(output).unwrap();
            writeln!(output, "void main() {{").unwrap();
            self.indent += 1;

            // Emit builtin variable assignments
            for (name, ty, gl_var) in &builtin_assignments {
                writeln!(output, "{}{} {} = {};", self.indent_str(), ty, name, gl_var).unwrap();
                self.declared_vars.insert(name.clone());
            }

            let result = self.lower_expr(body, body.root, output)?;

            // Assign to outputs - handle tuple returns by extracting components
            for (tuple_idx, _loc) in &location_outputs {
                if is_tuple_return {
                    writeln!(
                        output,
                        "{}_out{} = {}._{};",
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

            // Handle gl_Position for vertex shaders - extract from tuple if needed
            if stage == ShaderStage::Vertex {
                for (i, out) in outputs.iter().enumerate() {
                    if let Some(crate::mir::IoDecoration::BuiltIn(spirv::BuiltIn::Position)) =
                        &out.decoration
                    {
                        if is_tuple_return {
                            writeln!(output, "{}gl_Position = {}._{};", self.indent_str(), result, i)
                                .unwrap();
                        } else {
                            writeln!(output, "{}gl_Position = {};", self.indent_str(), result).unwrap();
                        }
                    }
                }
            }

            self.indent -= 1;
            writeln!(output, "}}").unwrap();
        }
        Ok(())
    }

    fn lower_expr(&mut self, body: &Body, expr_id: ExprId, output: &mut String) -> Result<String> {
        let expr = body.get_expr(expr_id);
        let ty = body.get_type(expr_id);

        match expr {
            // --- Literals ---
            Expr::Int(s) => Ok(s.clone()),

            Expr::Float(s) => {
                // Ensure float has decimal point
                if s.contains('.') || s.contains('e') || s.contains('E') {
                    Ok(s.clone())
                } else {
                    Ok(format!("{}.0", s))
                }
            }

            Expr::Bool(b) => Ok(if *b { "true" } else { "false" }.to_string()),

            Expr::String(s) => Ok(format!("\"{}\"", s)),

            Expr::Unit => Ok("".to_string()),

            // --- Variables ---
            Expr::Local(local_id) => {
                let local = body.get_local(*local_id);
                Ok(local.name.clone())
            }

            Expr::Global(name) => Ok(name.clone()),

            // --- Aggregates ---
            Expr::Tuple(elems) => {
                // Empty tuples should not reach lowering - they indicate a bug
                // (Unit values and empty closures should be handled at call sites)
                if elems.is_empty() {
                    bail_glsl!(
                        "BUG: Empty tuple reached GLSL lowering. Empty tuples/unit values should \
                         be handled at call sites (let _ = ..., map with empty closures, etc.)"
                    );
                }

                // Emit tuple as struct constructor
                let mut parts = Vec::new();
                for &e in elems {
                    parts.push(self.lower_expr(body, e, output)?);
                }
                let struct_name = self.type_to_glsl(ty);
                Ok(format!("{}({})", struct_name, parts.join(", ")))
            }

            Expr::Array { backing, size: _ } => match backing {
                ArrayBacking::Literal(elems) => {
                    let mut parts = Vec::new();
                    for &e in elems {
                        parts.push(self.lower_expr(body, e, output)?);
                    }
                    Ok(format!("{}[]({})", self.type_to_glsl(ty), parts.join(", ")))
                }
                ArrayBacking::Range { .. } => {
                    bail_glsl!("Range arrays not supported in GLSL lowering")
                }
                ArrayBacking::IndexFn { .. } => {
                    bail_glsl!("Index function arrays not supported in GLSL lowering")
                }
                ArrayBacking::View { .. } => {
                    bail_glsl!("View arrays not supported in GLSL lowering")
                }
                ArrayBacking::Owned { .. } => {
                    bail_glsl!("Owned arrays not supported in GLSL lowering")
                }
                ArrayBacking::Storage { .. } => {
                    bail_glsl!("Storage arrays not supported in GLSL lowering")
                }
            },

            Expr::Vector(elems) => {
                let mut parts = Vec::new();
                for &e in elems {
                    parts.push(self.lower_expr(body, e, output)?);
                }
                Ok(format!("{}({})", self.type_to_glsl(ty), parts.join(", ")))
            }

            Expr::Matrix(rows) => {
                if rows.is_empty() {
                    bail_glsl!("BUG: Empty matrix (no rows) reached GLSL lowering");
                }
                let num_cols = rows[0].len();
                if num_cols == 0 {
                    bail_glsl!("BUG: Empty matrix row (no columns) reached GLSL lowering");
                }
                let mut parts = Vec::new();
                // GLSL matrices are column-major, so iterate column-by-column
                for col in 0..num_cols {
                    for row in rows {
                        if col < row.len() {
                            parts.push(self.lower_expr(body, row[col], output)?);
                        }
                    }
                }
                Ok(format!("{}({})", self.type_to_glsl(ty), parts.join(", ")))
            }

            // --- Operations ---
            Expr::BinOp { op, lhs, rhs } => {
                let l = self.lower_expr(body, *lhs, output)?;
                let r = self.lower_expr(body, *rhs, output)?;
                // Handle operators that don't exist in GLSL
                match op.as_str() {
                    "**" => Ok(format!("pow({}, {})", l, r)),
                    _ => Ok(format!("({} {} {})", l, op, r)),
                }
            }

            Expr::UnaryOp { op, operand } => {
                let inner = self.lower_expr(body, *operand, output)?;
                Ok(format!("({}{})", op, inner))
            }

            // --- Control Flow ---
            Expr::If { cond, then_, else_ } => {
                // Use if-else statements for struct/tuple types (better compatibility)
                // Use ternary for simple scalar types
                if self.is_struct_type(ty) {
                    let result_var = format!("_w_if_{}", self.fresh_id());
                    let ty_str = self.type_to_glsl(ty);

                    // Declare result variable
                    writeln!(output, "{}{} {};", self.indent_str(), ty_str, result_var).unwrap();
                    self.declared_vars.insert(result_var.clone());

                    // Emit if-else
                    let c = self.lower_expr(body, *cond, output)?;
                    writeln!(output, "{}if ({}) {{", self.indent_str(), c).unwrap();
                    self.indent += 1;
                    let t = self.lower_expr(body, *then_, output)?;
                    writeln!(output, "{}{} = {};", self.indent_str(), result_var, t).unwrap();
                    self.indent -= 1;
                    writeln!(output, "{}}} else {{", self.indent_str()).unwrap();
                    self.indent += 1;
                    let e = self.lower_expr(body, *else_, output)?;
                    writeln!(output, "{}{} = {};", self.indent_str(), result_var, e).unwrap();
                    self.indent -= 1;
                    writeln!(output, "{}}}", self.indent_str()).unwrap();

                    Ok(result_var)
                } else {
                    let c = self.lower_expr(body, *cond, output)?;
                    let t = self.lower_expr(body, *then_, output)?;
                    let e = self.lower_expr(body, *else_, output)?;
                    Ok(format!("({} ? {} : {})", c, t, e))
                }
            }

            Expr::Let {
                local,
                rhs,
                body: let_body,
            } => {
                let local_decl = body.get_local(*local);
                let name = &local_decl.name;
                let v = self.lower_expr(body, *rhs, output)?;

                if self.declared_vars.contains(name) {
                    // Variable already declared, just assign
                    writeln!(output, "{}{} = {};", self.indent_str(), name, v).unwrap();
                } else {
                    // New variable, declare with type
                    let rhs_ty = body.get_type(*rhs);
                    writeln!(
                        output,
                        "{}{} {} = {};",
                        self.indent_str(),
                        self.type_to_glsl(rhs_ty),
                        name,
                        v
                    )
                    .unwrap();
                    self.declared_vars.insert(name.clone());
                }
                self.lower_expr(body, *let_body, output)
            }

            Expr::Loop {
                loop_var,
                init,
                init_bindings,
                kind,
                body: loop_body,
            } => self.lower_loop(body, *loop_var, *init, init_bindings, kind, *loop_body, output),

            // --- Calls ---
            Expr::Call { func, args } => {
                // Special case for reduce - generate a loop
                if func == "reduce" {
                    return self.lower_reduce(body, args, ty, output);
                }

                let mut arg_strs = Vec::new();
                for &arg in args {
                    arg_strs.push(self.lower_expr(body, arg, output)?);
                }

                // Check if it's a builtin
                if let Some(impl_) = self.impl_source.get(func) {
                    self.lower_builtin_call(impl_, &arg_strs, ty)
                } else {
                    Ok(format!("{}({})", func, arg_strs.join(", ")))
                }
            }

            Expr::Intrinsic { name, args } => {
                let mut arg_strs = Vec::new();
                let arg_ids: Vec<ExprId> = args.clone();
                for &arg in args {
                    arg_strs.push(self.lower_expr(body, arg, output)?);
                }
                self.lower_intrinsic(body, name, &arg_strs, &arg_ids, ty)
            }

            // --- Special ---
            Expr::Materialize(inner) => self.lower_expr(body, *inner, output),

            Expr::Attributed { expr: inner, .. } => self.lower_expr(body, *inner, output),

            // --- Memory operations ---
            Expr::Load { .. } | Expr::Store { .. } => {
                bail_glsl!("Load/Store expressions not yet implemented in GLSL lowering")
            }
        }
    }

    fn lower_builtin_call(
        &self,
        impl_: &BuiltinImpl,
        args: &[String],
        ret_ty: &PolyType<TypeName>,
    ) -> Result<String> {
        match impl_ {
            BuiltinImpl::PrimOp(op) => self.lower_primop(op, args, ret_ty),
            BuiltinImpl::Intrinsic(intr) => {
                bail_glsl!("Intrinsic {:?} not supported in GLSL", intr)
            }
        }
    }

    fn lower_primop(&self, op: &PrimOp, args: &[String], ret_ty: &PolyType<TypeName>) -> Result<String> {
        use PrimOp::*;
        match op {
            // GLSL.std.450 extended instructions map to GLSL functions
            GlslExt(id) => {
                let func = glsl_ext_to_name(*id);
                Ok(format!("{}({})", func, args.join(", ")))
            }

            // Math operations
            Dot => Ok(format!("dot({}, {})", args[0], args[1])),
            OuterProduct => Ok(format!("outerProduct({}, {})", args[0], args[1])),
            MatrixTimesMatrix | MatrixTimesVector | VectorTimesMatrix => {
                Ok(format!("({} * {})", args[0], args[1]))
            }
            VectorTimesScalar | MatrixTimesScalar => Ok(format!("({} * {})", args[0], args[1])),

            // Arithmetic
            FAdd | IAdd => Ok(format!("({} + {})", args[0], args[1])),
            FSub | ISub => Ok(format!("({} - {})", args[0], args[1])),
            FMul | IMul => Ok(format!("({} * {})", args[0], args[1])),
            FDiv | SDiv | UDiv => Ok(format!("({} / {})", args[0], args[1])),
            FRem | FMod | SRem | SMod => Ok(format!("mod({}, {})", args[0], args[1])),

            // Comparisons
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

            // Bitwise
            BitwiseAnd => Ok(format!("({} & {})", args[0], args[1])),
            BitwiseOr => Ok(format!("({} | {})", args[0], args[1])),
            BitwiseXor => Ok(format!("({} ^ {})", args[0], args[1])),
            Not => Ok(format!("(~{})", args[0])),
            ShiftLeftLogical => Ok(format!("({} << {})", args[0], args[1])),
            ShiftRightArithmetic | ShiftRightLogical => Ok(format!("({} >> {})", args[0], args[1])),

            // Conversions
            FPToSI => Ok(format!("int({})", args[0])),
            FPToUI => Ok(format!("uint({})", args[0])),
            SIToFP | UIToFP => Ok(format!("float({})", args[0])),
            FPConvert => Ok(format!("float({})", args[0])),
            SConvert | UConvert => Ok(format!("int({})", args[0])),
            Bitcast => {
                // Choose the correct GLSL bitcast function based on target type
                let func = match ret_ty {
                    PolyType::Constructed(TypeName::Int(32), _) => "floatBitsToInt",
                    PolyType::Constructed(TypeName::UInt(32), _) => "floatBitsToUint",
                    PolyType::Constructed(TypeName::Float(32), _) => "intBitsToFloat",
                    _ => "floatBitsToInt", // Default fallback
                };
                Ok(format!("{}({})", func, args[0]))
            }
        }
    }

    fn lower_intrinsic(
        &self,
        body: &Body,
        name: &str,
        args: &[String],
        arg_ids: &[ExprId],
        _ret_ty: &PolyType<TypeName>,
    ) -> Result<String> {
        match name {
            "tuple_access" => {
                // args[0] is the tuple/vector, args[1] is the index
                if let Expr::Int(idx_str) = body.get_expr(arg_ids[1]) {
                    let idx: usize = idx_str
                        .parse()
                        .map_err(|_| crate::err_glsl!("BUG: Invalid tuple index literal: {}", idx_str))?;
                    // Check if this is a vector type - use swizzle syntax
                    let arg_ty = body.get_type(arg_ids[0]);
                    if self.is_vector_type(arg_ty) {
                        let swizzle = match idx {
                            0 => "x",
                            1 => "y",
                            2 => "z",
                            3 => "w",
                            _ => bail_glsl!("BUG: Invalid vector swizzle index: {} (max is 3)", idx),
                        };
                        Ok(format!("{}.{}", args[0], swizzle))
                    } else {
                        // Tuple - use struct field syntax ._N
                        Ok(format!("{}._{}", args[0], idx))
                    }
                } else {
                    // Fallback for dynamic index
                    Ok(format!("{}[{}]", args[0], args[1]))
                }
            }
            "record_access" => {
                // args[0] is the record, args[1] is the field name (as string literal)
                if let Expr::String(field) = body.get_expr(arg_ids[1]) {
                    Ok(format!("{}.{}", args[0], field))
                } else {
                    Ok(format!("{}.{}", args[0], args[1]))
                }
            }
            "index" => {
                // Check if argument is an array (unified Array[elem, addrspace, size] type)
                let arg_ty = body.get_type(arg_ids[0]);
                match arg_ty {
                    PolyType::Constructed(TypeName::Array, type_args) => {
                        assert!(type_args.len() == 3);
                        // Check if unsized (slice-like) - size is at index 2
                        let is_unsized =
                            matches!(&type_args[2], PolyType::Constructed(TypeName::Unsized, _));
                        if is_unsized {
                            // Unsized array (slice) - check expression kind
                            match body.get_expr(arg_ids[0]) {
                                Expr::Array {
                                    backing: ArrayBacking::Owned { .. },
                                    ..
                                } => Ok(format!("{}.data[{}]", args[0], args[1])),
                                Expr::Array {
                                    backing: ArrayBacking::View { .. },
                                    ..
                                } => Ok(format!("{}.base[{}.offset + {}]", args[0], args[0], args[1])),
                                Expr::Array {
                                    backing: ArrayBacking::Storage { name, .. },
                                    ..
                                } => Ok(format!("{}[{}.offset + {}]", name, args[0], args[1])),
                                _ => Ok(format!("{}.data[{}]", args[0], args[1])),
                            }
                        } else {
                            // Sized array - regular indexing
                            Ok(format!("{}[{}]", args[0], args[1]))
                        }
                    }
                    _ => Ok(format!("{}[{}]", args[0], args[1])),
                }
            }
            "length" => {
                // Check if argument is an array
                let arg_ty = body.get_type(arg_ids[0]);
                match arg_ty {
                    PolyType::Constructed(TypeName::Array, type_args) => {
                        assert!(type_args.len() == 3);
                        // Check if unsized (slice-like)
                        let is_unsized =
                            matches!(&type_args[2], PolyType::Constructed(TypeName::Unsized, _));
                        if is_unsized {
                            // Unsized array: access len field
                            Ok(format!("{}.len", args[0]))
                        } else {
                            // Sized array: use GLSL .length() method
                            Ok(format!("{}.length()", args[0]))
                        }
                    }
                    _ => bail_glsl!("length called on non-array type: {:?}", arg_ty),
                }
            }
            "_w_tuple_proj" => {
                // Same as tuple_access - args[0] is the tuple/vector, args[1] is the index
                if let Expr::Int(idx_str) = body.get_expr(arg_ids[1]) {
                    let idx: usize = idx_str
                        .parse()
                        .map_err(|_| crate::err_glsl!("BUG: Invalid tuple index literal: {}", idx_str))?;
                    let arg_ty = body.get_type(arg_ids[0]);
                    if self.is_vector_type(arg_ty) {
                        let swizzle = match idx {
                            0 => "x",
                            1 => "y",
                            2 => "z",
                            3 => "w",
                            _ => bail_glsl!("Invalid vector component index: {}", idx),
                        };
                        Ok(format!("{}.{}", args[0], swizzle))
                    } else {
                        // Struct/tuple field access
                        Ok(format!("{}._{}", args[0], idx))
                    }
                } else {
                    bail_glsl!("_w_tuple_proj requires constant index")
                }
            }
            "_w_index" => {
                // Same as index - array indexing
                let arg_ty = body.get_type(arg_ids[0]);
                match arg_ty {
                    PolyType::Constructed(TypeName::Array, type_args) => {
                        assert!(type_args.len() == 3);
                        let is_unsized =
                            matches!(&type_args[2], PolyType::Constructed(TypeName::Unsized, _));
                        if is_unsized {
                            match body.get_expr(arg_ids[0]) {
                                Expr::Array {
                                    backing: ArrayBacking::Owned { .. },
                                    ..
                                } => Ok(format!("{}.data[{}]", args[0], args[1])),
                                Expr::Array {
                                    backing: ArrayBacking::View { .. },
                                    ..
                                } => Ok(format!("{}.base[{}.offset + {}]", args[0], args[0], args[1])),
                                Expr::Array {
                                    backing: ArrayBacking::Storage { name, .. },
                                    ..
                                } => Ok(format!("{}[{}.offset + {}]", name, args[0], args[1])),
                                _ => Ok(format!("{}.data[{}]", args[0], args[1])),
                            }
                        } else {
                            Ok(format!("{}[{}]", args[0], args[1]))
                        }
                    }
                    _ => Ok(format!("{}[{}]", args[0], args[1])),
                }
            }
            _ => bail_glsl!("Unknown intrinsic: {}", name),
        }
    }

    fn lower_loop(
        &mut self,
        body: &Body,
        loop_var: LocalId,
        init: ExprId,
        init_bindings: &[(LocalId, ExprId)],
        kind: &LoopKind,
        loop_body: ExprId,
        output: &mut String,
    ) -> Result<String> {
        let loop_var_name = body.get_local(loop_var).name.clone();

        // Emit init
        let init_val = self.lower_expr(body, init, output)?;
        let init_ty = body.get_type(init);
        if self.declared_vars.contains(&loop_var_name) {
            writeln!(output, "{}{} = {};", self.indent_str(), loop_var_name, init_val).unwrap();
        } else {
            writeln!(
                output,
                "{}{} {} = {};",
                self.indent_str(),
                self.type_to_glsl(init_ty),
                loop_var_name,
                init_val
            )
            .unwrap();
            self.declared_vars.insert(loop_var_name.clone());
        }

        // Emit init bindings
        for (local_id, expr_id) in init_bindings {
            let local_name = body.get_local(*local_id).name.clone();
            let val = self.lower_expr(body, *expr_id, output)?;
            let binding_ty = body.get_type(*expr_id);
            if self.declared_vars.contains(&local_name) {
                writeln!(output, "{}{} = {};", self.indent_str(), local_name, val).unwrap();
            } else {
                writeln!(
                    output,
                    "{}{} {} = {};",
                    self.indent_str(),
                    self.type_to_glsl(binding_ty),
                    local_name,
                    val
                )
                .unwrap();
                self.declared_vars.insert(local_name.clone());
            }
        }

        match kind {
            LoopKind::While { cond } => {
                // Use while(true) with break to ensure condition is re-evaluated each iteration
                writeln!(output, "{}while (true) {{", self.indent_str()).unwrap();
                self.indent += 1;
                let cond_str = self.lower_expr(body, *cond, output)?;
                writeln!(output, "{}if (!{}) break;", self.indent_str(), cond_str).unwrap();
                self.indent -= 1;
            }
            LoopKind::ForRange { var, bound } => {
                let var_name = body.get_local(*var).name.clone();
                let bound_str = self.lower_expr(body, *bound, output)?;
                writeln!(
                    output,
                    "{}for (int {} = 0; {} < {}; {}++) {{",
                    self.indent_str(),
                    var_name,
                    var_name,
                    bound_str,
                    var_name
                )
                .unwrap();
            }
            LoopKind::For { var, iter } => {
                let var_decl = body.get_local(*var);
                let var_name = var_decl.name.clone();
                let var_ty = &var_decl.ty;
                let iter_str = self.lower_expr(body, *iter, output)?;
                writeln!(
                    output,
                    "{}for (int _i = 0; _i < {}.length(); _i++) {{",
                    self.indent_str(),
                    iter_str
                )
                .unwrap();
                self.indent += 1;
                writeln!(
                    output,
                    "{}{} {} = {}[_i];",
                    self.indent_str(),
                    self.type_to_glsl(var_ty),
                    var_name,
                    iter_str
                )
                .unwrap();
                self.indent -= 1;
            }
        }

        self.indent += 1;
        let body_result = self.lower_expr(body, loop_body, output)?;
        writeln!(
            output,
            "{}{} = {};",
            self.indent_str(),
            loop_var_name,
            body_result
        )
        .unwrap();

        // Re-extract loop bindings from the updated tuple
        for (local_id, expr_id) in init_bindings {
            let local_name = body.get_local(*local_id).name.clone();
            let val = self.lower_expr(body, *expr_id, output)?;
            writeln!(output, "{}{} = {};", self.indent_str(), local_name, val).unwrap();
        }

        self.indent -= 1;

        writeln!(output, "{}}}", self.indent_str()).unwrap();

        Ok(loop_var_name)
    }

    /// Lower a reduce call to a GLSL for loop
    /// reduce op ne array -> scalar
    fn lower_reduce(
        &mut self,
        body: &Body,
        args: &[ExprId],
        ret_ty: &PolyType<TypeName>,
        output: &mut String,
    ) -> Result<String> {
        if args.len() != 3 {
            bail_glsl!("reduce requires 3 args (op, ne, array), got {}", args.len());
        }

        // Extract function name from the operator reference
        let op_func_name = match body.get_expr(args[0]) {
            Expr::Global(name) => name.clone(),
            other => bail_glsl!(
                "reduce operator must be a function reference (Global), got {:?}",
                other
            ),
        };

        // Lower neutral element and array
        let neutral_val = self.lower_expr(body, args[1], output)?;
        let array_val = self.lower_expr(body, args[2], output)?;

        // Get array size from type
        let arr_ty = body.get_type(args[2]);
        let array_size = match arr_ty {
            PolyType::Constructed(TypeName::Array, type_args) => {
                assert!(type_args.len() == 3);
                match &type_args[2] {
                    PolyType::Constructed(TypeName::Size(n), _) => *n,
                    _ => bail_glsl!("Invalid array size type for reduce"),
                }
            }
            _ => bail_glsl!("reduce input must be array type"),
        };

        // Generate accumulator variable
        let acc_name = format!("_w_reduce_acc_{}", self.temp_counter);
        self.temp_counter += 1;
        let acc_ty_str = self.type_to_glsl(ret_ty);

        // Emit: type acc = ne;
        writeln!(
            output,
            "{}{} {} = {};",
            self.indent_str(),
            acc_ty_str,
            acc_name,
            neutral_val
        )
        .unwrap();

        // Emit: for (int i = 0; i < N; i++) { acc = op(acc, array[i]); }
        let loop_var = format!("_w_reduce_i_{}", self.temp_counter);
        self.temp_counter += 1;

        writeln!(
            output,
            "{}for (int {} = 0; {} < {}; {}++) {{",
            self.indent_str(),
            loop_var,
            loop_var,
            array_size,
            loop_var
        )
        .unwrap();

        self.indent += 1;
        writeln!(
            output,
            "{}{} = {}({}, {}[{}]);",
            self.indent_str(),
            acc_name,
            op_func_name,
            acc_name,
            array_val,
            loop_var
        )
        .unwrap();
        self.indent -= 1;

        writeln!(output, "{}}}", self.indent_str()).unwrap();

        // Return the accumulator variable name
        Ok(acc_name)
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
                TypeName::Str(s) if *s == "bool" => "bool".to_string(),
                TypeName::Unit => "void".to_string(),
                // Tuple: generate a struct type
                TypeName::Tuple(_) => {
                    // Get element types
                    let elem_types: Vec<String> = args.iter().map(|a| self.type_to_glsl(a)).collect();
                    let sig = elem_types.join(",");

                    // Check cache
                    if let Some(name) = self.tuple_type_cache.get(&sig) {
                        return name.clone();
                    }

                    // Generate new struct name
                    let struct_name = format!("_Tuple{}", self.tuple_counter);
                    self.tuple_counter += 1;

                    // Register the struct
                    self.tuple_structs.insert(struct_name.clone(), elem_types);
                    self.tuple_type_cache.insert(sig, struct_name.clone());

                    struct_name
                }
                // Vec: args[0] is Size(n), args[1] is element type
                TypeName::Vec if args.len() >= 2 => {
                    let n = match &args[0] {
                        PolyType::Constructed(TypeName::Size(n), _) => *n,
                        _ => 4,
                    };
                    let elem = self.type_to_glsl(&args[1]);
                    match elem.as_str() {
                        "float" => format!("vec{}", n),
                        "double" => format!("dvec{}", n),
                        "int" => format!("ivec{}", n),
                        "uint" => format!("uvec{}", n),
                        "bool" => format!("bvec{}", n),
                        _ => format!("vec{}", n),
                    }
                }
                // Mat: args[0] is Size(cols), args[1] is Size(rows), args[2] is element type
                TypeName::Mat if args.len() >= 3 => {
                    let cols = match &args[0] {
                        PolyType::Constructed(TypeName::Size(n), _) => *n,
                        _ => 4,
                    };
                    let rows = match &args[1] {
                        PolyType::Constructed(TypeName::Size(n), _) => *n,
                        _ => 4,
                    };
                    let elem = self.type_to_glsl(&args[2]);
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
                // Array: args[0] is elem_type, args[1] is addrspace, args[2] is size
                TypeName::Array => {
                    assert!(args.len() == 3);
                    format!("{}[]", self.type_to_glsl(&args[0]))
                }
                // Record types (closures) should be eliminated before GLSL lowering
                TypeName::Record(fields) => {
                    panic!(
                        "BUG: Record type reached GLSL lowering. \
                         This should have been eliminated during defunctionalization. \
                         Fields: {:?}",
                        fields
                    );
                }
                _ => "/* unknown */".to_string(),
            },
            _ => "/* unknown */".to_string(),
        }
    }

    fn indent_str(&self) -> String {
        "    ".repeat(self.indent)
    }

    fn is_vector_type(&self, ty: &PolyType<TypeName>) -> bool {
        matches!(ty, PolyType::Constructed(TypeName::Vec, _))
    }
}

/// Map GLSLstd450 extended instruction opcodes to GLSL function names
fn glsl_ext_to_name(id: u32) -> &'static str {
    match id {
        1 => "round",        // Round
        2 => "roundEven",    // RoundEven
        3 => "trunc",        // Trunc
        4 => "abs",          // FAbs
        5 => "abs",          // SAbs
        6 => "sign",         // FSign
        7 => "sign",         // SSign
        8 => "floor",        // Floor
        9 => "ceil",         // Ceil
        10 => "fract",       // Fract
        11 => "radians",     // Radians
        12 => "degrees",     // Degrees
        13 => "sin",         // Sin
        14 => "cos",         // Cos
        15 => "tan",         // Tan
        16 => "asin",        // Asin
        17 => "acos",        // Acos
        18 => "atan",        // Atan
        19 => "sinh",        // Sinh
        20 => "cosh",        // Cosh
        21 => "tanh",        // Tanh
        22 => "asinh",       // Asinh
        23 => "acosh",       // Acosh
        24 => "atanh",       // Atanh
        25 => "atan",        // Atan2
        26 => "pow",         // Pow
        27 => "exp",         // Exp
        28 => "log",         // Log
        29 => "exp2",        // Exp2
        30 => "log2",        // Log2
        31 => "sqrt",        // Sqrt
        32 => "inversesqrt", // InverseSqrt
        33 => "determinant", // Determinant
        34 => "inverse",     // MatrixInverse
        37 => "min",         // FMin
        38 => "min",         // UMin
        39 => "min",         // SMin
        40 => "max",         // FMax
        41 => "max",         // UMax
        42 => "max",         // SMax
        43 => "clamp",       // FClamp
        44 => "clamp",       // UClamp
        45 => "clamp",       // SClamp
        46 => "mix",         // FMix
        48 => "step",        // Step
        49 => "smoothstep",  // SmoothStep
        50 => "fma",         // Fma
        53 => "ldexp",       // Ldexp
        // TODO: isnan/isinf use opcodes 66/67 in impl_source.rs but those are Length/Distance
        // IsNan should be 149, IsInf should be 148 - need to fix impl_source.rs and add mappings here
        66 => "length",      // Length (also incorrectly used for isnan)
        67 => "distance",    // Distance (also incorrectly used for isinf)
        68 => "cross",       // Cross
        69 => "normalize",   // Normalize
        70 => "faceforward", // FaceForward
        71 => "reflect",     // Reflect
        72 => "refract",     // Refract
        _ => "/* unknown_glsl_ext */",
    }
}
