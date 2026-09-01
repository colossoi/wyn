use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
use wyn_core::error::CompilerError;
use wyn_core::{
    CodegenTarget, CompilationFailure, Compiler, CompilerOptions, LoweringProfile, ParsedModules,
    PipelineTopologyPolicy, SchedulePolicy,
};
use wyn_module_graph::{
    BuildError, BuildFailure, LocalSourceError, LocalSources, ModuleKey, ModulePath, PackageIdentity,
    PackagePlan, PackagePlanBuilder, SourceFingerprint,
};

/// Get the compiler version string
#[wasm_bindgen]
pub fn version() -> String {
    "005".to_string()
}

fn single_source_plan(source: &str) -> Result<(PackagePlan, LocalSources), String> {
    let root_path = ModulePath::new("main.wyn").map_err(|error| error.to_string())?;
    let fingerprint =
        SourceFingerprint::new("wasm-source").map_err(|error| error.to_string())?;
    let identity = PackageIdentity::new("wasm/root", "v0.0.0", fingerprint)
        .map_err(|error| error.to_string())?;
    let mut builder = PackagePlanBuilder::new();
    let package = builder
        .add_package(identity, root_path.clone())
        .map_err(|error| error.to_string())?;
    let root = ModuleKey::new(package, root_path);
    builder.set_root(root.clone()).map_err(|error| error.to_string())?;
    let plan = builder.build().map_err(|error| error.to_string())?;
    let mut sources = LocalSources::new();
    sources.add_override(root, source).map_err(|error| error.to_string())?;
    Ok((plan, sources))
}

fn load_source_modules(
    source: &str,
    options: CompilerOptions,
) -> Result<ParsedModules, SourceModulesError> {
    let compiler = Compiler::new(options).map_err(SourceModulesError::Compiler)?;
    let (plan, mut sources) = single_source_plan(source).map_err(SourceModulesError::Setup)?;
    compiler
        .load_modules(plan, &mut sources)
        .map_err(SourceModulesError::Build)
}

#[derive(Debug)]
enum SourceModulesError {
    Compiler(CompilerError),
    Setup(String),
    Build(BuildFailure<CompilerError, LocalSourceError>),
}

// =============================================================================
// Tree Node for IR visualization
// =============================================================================

/// A node in a tree representation of IR.
/// Format: { name: "label", children: [...] }
#[derive(Serialize, Deserialize, Clone)]
pub struct TreeNode {
    pub name: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<TreeNode>,
}

impl TreeNode {
    fn leaf(name: impl Into<String>) -> Self {
        TreeNode {
            name: name.into(),
            children: vec![],
        }
    }

    fn branch(name: impl Into<String>, children: Vec<TreeNode>) -> Self {
        TreeNode {
            name: name.into(),
            children,
        }
    }
}

// =============================================================================
// TLC to Tree conversion
// =============================================================================

mod tlc_tree {
    use super::TreeNode;
    use wyn_core::ast::TypeName;
    use wyn_core::tlc::{Def, DefMeta, Family, LoopKind, Payload, Program, Term, TermKind};

    fn fmt_ty(ty: &polytype::Type<TypeName>) -> String {
        wyn_core::diags::format_type(ty)
    }

    pub fn program_to_tree<Tag, F: Family, GlobalContext>(
        program: &Program<Tag, F, GlobalContext>,
    ) -> Vec<TreeNode> {
        program.defs.iter().map(def_to_tree).collect()
    }

    fn def_to_tree<F: Family>(def: &Def<F>) -> TreeNode {
        let meta = match &def.meta {
            DefMeta::Function => "fn",
            DefMeta::EntryPoint(_) => "entry",
            DefMeta::LiftedLambda => "lifted",
        };
        let label = format!("{} {} : {}", meta, def.name, fmt_ty(&def.ty));
        TreeNode::branch(label, vec![term_to_tree(&def.body)])
    }

    fn term_to_tree<C: Payload, S: Payload>(term: &Term<C, S>) -> TreeNode {
        let ty = fmt_ty(&term.ty);
        match &term.kind {
            TermKind::Var(wyn_core::tlc::VarRef::Symbol(name)) => {
                TreeNode::leaf(format!("Var({}) : {}", name, ty))
            }
            TermKind::BinOp(op) => TreeNode::leaf(format!("BinOp({:?}) : {}", op, ty)),
            TermKind::UnOp(op) => TreeNode::leaf(format!("UnOp({:?}) : {}", op, ty)),
            TermKind::Lambda(ref lam) => {
                let params_str: Vec<String> =
                    lam.params.iter().map(|(p, ty)| format!("{}: {}", p, fmt_ty(ty))).collect();
                let label = format!("Lambda({}) : {}", params_str.join(", "), ty);
                TreeNode::branch(label, vec![term_to_tree(&lam.body)])
            }
            TermKind::App { func, args } => {
                let mut children = vec![TreeNode::branch("func", vec![term_to_tree(func)])];
                for (i, arg) in args.iter().enumerate() {
                    children.push(TreeNode::branch(format!("arg{}", i), vec![term_to_tree(arg)]));
                }
                TreeNode::branch(format!("App : {}", ty), children)
            }
            TermKind::Let {
                name,
                name_ty,
                rhs,
                body,
            } => {
                let label = format!("Let({}: {})", name, fmt_ty(name_ty));
                TreeNode::branch(
                    label,
                    vec![
                        TreeNode::branch("rhs", vec![term_to_tree(rhs)]),
                        TreeNode::branch("body", vec![term_to_tree(body)]),
                    ],
                )
            }
            TermKind::IntLit(s) => TreeNode::leaf(format!("Int({}) : {}", s, ty)),
            TermKind::FloatLit(f) => TreeNode::leaf(format!("Float({}) : {}", f, ty)),
            TermKind::BoolLit(b) => TreeNode::leaf(format!("Bool({}) : {}", b, ty)),
            TermKind::Extern(link) => TreeNode::leaf(format!("Extern({}) : {}", link, ty)),
            TermKind::If {
                cond,
                then_branch,
                else_branch,
            } => TreeNode::branch(
                format!("If : {}", ty),
                vec![
                    TreeNode::branch("cond", vec![term_to_tree(cond)]),
                    TreeNode::branch("then", vec![term_to_tree(then_branch)]),
                    TreeNode::branch("else", vec![term_to_tree(else_branch)]),
                ],
            ),
            TermKind::Loop {
                loop_var,
                loop_var_ty,
                init,
                init_bindings,
                kind,
                body,
            } => {
                let label = format!("Loop({}: {})", loop_var, fmt_ty(loop_var_ty));
                let mut children = vec![TreeNode::branch("init", vec![term_to_tree(init)])];
                if !init_bindings.is_empty() {
                    let bindings: Vec<TreeNode> = init_bindings
                        .iter()
                        .map(|(n, t, e)| {
                            TreeNode::branch(format!("{}: {}", n, fmt_ty(t)), vec![term_to_tree(e)])
                        })
                        .collect();
                    children.push(TreeNode::branch("bindings", bindings));
                }
                children.push(loop_kind_to_tree(kind));
                children.push(TreeNode::branch("body", vec![term_to_tree(body)]));
                TreeNode::branch(label, children)
            }
            TermKind::Closure(_) => TreeNode::leaf(format!("<closure> : {}", ty)),
            TermKind::Soac(_) | TermKind::ArrayExpr(_) => TreeNode::leaf(format!("<soac> : {}", ty)),
            other => TreeNode::leaf(format!("{:?} : {}", other, ty)),
        }
    }

    fn loop_kind_to_tree<C: Payload, S: Payload>(kind: &LoopKind<C, S>) -> TreeNode {
        match kind {
            LoopKind::For { var, var_ty, iter } => TreeNode::branch(
                format!("for {} : {}", var, fmt_ty(var_ty)),
                vec![term_to_tree(iter)],
            ),
            LoopKind::ForRange { var, var_ty, bound } => TreeNode::branch(
                format!("for_range {} : {}", var, fmt_ty(var_ty)),
                vec![term_to_tree(bound)],
            ),
            LoopKind::While { cond } => TreeNode::branch("while", vec![term_to_tree(cond)]),
        }
    }
}

/// Initialize the compiler cache. Call this once at startup.
/// Returns true on success.
#[wasm_bindgen]
pub fn init_compiler() -> bool {
    console_error_panic_hook::set_once();
    match Compiler::new(CompilerOptions::default()) {
        Ok(_) => true,
        Err(error) => {
            web_sys::console::error_1(&format!("Failed to initialize compiler: {error}").into());
            false
        }
    }
}

/// Source location for an error
#[derive(Serialize, Deserialize, Clone)]
pub struct ErrorLocation {
    pub start_line: usize,
    pub start_col: usize,
    pub end_line: usize,
    pub end_col: usize,
}

/// Structured error information
#[derive(Serialize, Deserialize, Clone)]
pub struct ErrorInfo {
    pub message: String,
    pub location: Option<ErrorLocation>,
}

fn source_position(source: &str, offset: u32) -> Option<(usize, usize)> {
    let offset = usize::try_from(offset).ok()?;
    if offset > source.len() || !source.is_char_boundary(offset) {
        return None;
    }
    let prefix = &source[..offset];
    let line = prefix.bytes().filter(|byte| *byte == b'\n').count() + 1;
    let current_line = prefix.rsplit_once('\n').map_or(prefix, |(_, current_line)| current_line);
    Some((line, current_line.chars().count() + 1))
}

fn error_location(source: &str, e: &CompilerError) -> Option<ErrorLocation> {
    let span = e.span()?;
    span.module()?;
    let range = span.range();
    let (start_line, start_col) = source_position(source, range.start())?;
    let (end_line, end_col) = source_position(source, range.end())?;
    Some(ErrorLocation {
        start_line,
        start_col,
        end_line,
        end_col,
    })
}

fn format_error(e: &CompilerError) -> String {
    match e {
        CompilerError::ParseError(msg, _) => format!("Parse error: {}", msg),
        CompilerError::TypeError(msg, _) => format!("Type error: {}", msg),
        CompilerError::UndefinedVariable(name, _) => format!("Undefined variable: '{}'", name),
        CompilerError::AliasError(msg, _) => format!("Alias error: {}", msg),
        CompilerError::SpirvError(msg, _) => format!("SPIR-V error: {}", msg),
        CompilerError::WgslError(msg, _) => format!("WGSL error: {}", msg),
        CompilerError::ModuleError(msg, _) => format!("Module error: {}", msg),
        CompilerError::FlatteningError(msg, _) => format!("Flatten error: {}", msg),
        CompilerError::IoError(err) => format!("IO error: {}", err),
        CompilerError::SpirvBuilderError(msg) => format!("SPIR-V builder error: {}", msg),
        CompilerError::TypeHole(msg) => format!("Type hole: {}", msg),
        CompilerError::FormattingError(err) => format!("Formatting error: {err}"),
        CompilerError::Internal(msg) => format!("Internal compiler error: {msg}"),
    }
}

// =============================================================================
// Program interface metadata (for WebGPU binding + pipeline visualization)
// =============================================================================

/// Compact description of a program's entry points and resource bindings,
/// serializable to JSON for the JS side to drive WebGPU setup and the
/// pipeline-visualization UI.
#[derive(Serialize, Deserialize, Clone)]
pub struct ProgramInterface {
    pub entries: Vec<EntryInterface>,
    pub uniforms: Vec<ResourceBinding>,
    pub storage: Vec<ResourceBinding>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct EntryInterface {
    pub name: String,
    /// WGSL entry-point name — this is what WebGPU's
    /// `entryPoint:` in `createRenderPipeline` / `createComputePipeline`
    /// needs. Entry points are emitted verbatim by the WGSL backend.
    pub wgsl_name: String,
    /// One of "vertex" / "fragment" / "compute".
    pub kind: String,
    /// `[x, y, z]` workgroup size for compute entries; omitted otherwise.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub workgroup_size: Option<[u32; 3]>,
    pub inputs: Vec<EntryBinding>,
    pub outputs: Vec<EntryBinding>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct EntryBinding {
    pub name: String,
    pub ty: String,
    /// `"builtin(<name>)"`, `"slot(<n>)"`, `"target(<name>)"`,
    /// `"storage(<set>,<binding>)"`, `"uniform(<set>,<binding>)"`,
    /// `"texture(<set>,<binding>)"`, `"sampler(<set>,<binding>)"`,
    /// `"push_constant(<offset>)"`, or `"unknown"`.
    pub decoration: String,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ResourceBinding {
    pub name: String,
    pub set: u32,
    pub binding: u32,
    pub ty: String,
    /// For storage bindings: `"read"` / `"write"` / `"read_write"`. Empty
    /// for uniforms.
    #[serde(skip_serializing_if = "String::is_empty", default)]
    pub access: String,
}

fn fmt_ssa_type(ty: &polytype::Type<wyn_core::ast::TypeName>) -> String {
    wyn_core::diags::format_type(ty)
}

fn entry_binding_from_input(input: &wyn_core::interface::EntryInput) -> EntryBinding {
    use wyn_core::interface::IoDecoration;
    use wyn_core::BindingRef;
    let decoration = if let Some(BindingRef { set, binding }) = input.storage_binding() {
        format!("storage({},{})", set, binding)
    } else if let Some(BindingRef { set, binding }) = input.uniform_binding() {
        format!("uniform({},{})", set, binding)
    } else if let Some(BindingRef { set, binding }) = input.texture_binding() {
        format!("texture({},{})", set, binding)
    } else if let Some(BindingRef { set, binding }) = input.sampler_binding() {
        format!("sampler({},{})", set, binding)
    } else if let Some(pc) = input.push_constant() {
        format!("push_constant({})", pc.offset)
    } else {
        match input.decoration() {
            Some(IoDecoration::BuiltIn(b)) => format!("builtin({:?})", b),
            Some(IoDecoration::Location(n)) => format!("slot({})", n),
            None => "unknown".to_string(),
        }
    };
    EntryBinding {
        name: input.name.clone(),
        ty: fmt_ssa_type(&input.ty),
        decoration,
    }
}

fn entry_binding_from_output(idx: usize, output: &wyn_core::interface::EntryOutput) -> EntryBinding {
    use wyn_core::interface::IoDecoration;
    use wyn_core::BindingRef;
    let decoration = if let Some(BindingRef { set, binding }) = output.storage_binding() {
        format!("storage({},{})", set, binding)
    } else if let Some(name) = output.target() {
        format!("target({})", name)
    } else {
        match output.decoration() {
            Some(IoDecoration::BuiltIn(b)) => format!("builtin({:?})", b),
            Some(IoDecoration::Location(n)) => format!("slot({})", n),
            None => "unknown".to_string(),
        }
    };
    EntryBinding {
        name: format!("out{}", idx),
        ty: fmt_ssa_type(&output.ty),
        decoration,
    }
}

fn program_interface<Tag, GlobalContext>(
    program: &wyn_core::ssa::Program<Tag, GlobalContext>,
) -> ProgramInterface {
    use wyn_core::flow::ExecutionModel;
    use wyn_core::types::TypeExt;
    let entries = program
        .entry_points
        .iter()
        .map(|e| {
            let (kind, workgroup_size) = match &e.execution_model {
                ExecutionModel::Vertex => ("vertex".to_string(), None),
                ExecutionModel::Fragment => ("fragment".to_string(), None),
                ExecutionModel::Compute { local_size } => (
                    "compute".to_string(),
                    Some([local_size.0, local_size.1, local_size.2]),
                ),
            };
            let mut inputs: Vec<EntryBinding> = e.inputs.iter().map(entry_binding_from_input).collect();
            // Compiler-introduced storage bindings that aren't already in
            // inputs/outputs — surface them so the pipeline viz can show
            // the full buffer interface.
            for sb in &e.storage_bindings {
                let already = e.inputs.iter().any(|i| i.storage_binding() == Some(sb.binding))
                    || e.outputs.iter().any(|o| o.storage_binding() == Some(sb.binding));
                if already {
                    continue;
                }
                let role = match sb.role {
                    wyn_core::interface::StorageRole::Input => "in",
                    wyn_core::interface::StorageRole::Output => "out",
                    wyn_core::interface::StorageRole::InputOutput => "inout",
                    wyn_core::interface::StorageRole::Intermediate => "tmp",
                };
                inputs.push(EntryBinding {
                    name: format!("_buf_{}_{}_{}", sb.binding.set, sb.binding.binding, role),
                    ty: fmt_ssa_type(&sb.elem_ty),
                    decoration: format!("storage({},{})", sb.binding.set, sb.binding.binding),
                });
            }
            let outputs: Vec<EntryBinding> =
                e.outputs.iter().enumerate().map(|(i, o)| entry_binding_from_output(i, o)).collect();
            EntryInterface {
                name: e.name.clone(),
                wgsl_name: e.name.clone(),
                kind,
                workgroup_size,
                inputs,
                outputs,
            }
        })
        .collect();
    // Uniforms: every entry input carrying a `#[uniform(set, binding)]`
    // attribution. Deduplicate by slot — the same uniform is referenced from
    // each entry that uses it (e.g. a vertex + fragment pair sharing iTime).
    let mut uniforms_by_slot: std::collections::BTreeMap<(u32, u32), ResourceBinding> =
        std::collections::BTreeMap::new();
    for entry in &program.entry_points {
        for input in &entry.inputs {
            if let Some(br) = input.uniform_binding() {
                uniforms_by_slot.entry((br.set, br.binding)).or_insert_with(|| ResourceBinding {
                    name: input.name.clone(),
                    set: br.set,
                    binding: br.binding,
                    ty: fmt_ssa_type(&input.ty),
                    access: String::new(),
                });
            }
        }
    }
    let uniforms: Vec<ResourceBinding> = uniforms_by_slot.into_values().collect();

    // Storage bindings, coalesced across entries — a phase-1 writer and a
    // phase-2 reader of the same slot yields `read_write`. Both the user's
    // declared storage params and the compiler-introduced buffers (e.g.
    // parallelize's partials + result) live on the entries' inputs/outputs
    // and `storage_bindings`, so they all flow through here.
    let mut storage: Vec<ResourceBinding> = Vec::new();
    let mut synth: std::collections::BTreeMap<
        (u32, u32),
        (polytype::Type<wyn_core::ast::TypeName>, bool, bool),
    > = std::collections::BTreeMap::new();
    let mark = |synth: &mut std::collections::BTreeMap<_, _>,
                set: u32,
                binding: u32,
                elem_ty: polytype::Type<wyn_core::ast::TypeName>,
                reads: bool,
                writes: bool| {
        let e: &mut (_, bool, bool) =
            synth.entry((set, binding)).or_insert_with(|| (elem_ty, false, false));
        e.1 |= reads;
        e.2 |= writes;
    };
    for entry in &program.entry_points {
        for sb in &entry.storage_bindings {
            let (r, w) = match sb.role {
                wyn_core::interface::StorageRole::Input => (true, false),
                wyn_core::interface::StorageRole::Output => (false, true),
                wyn_core::interface::StorageRole::InputOutput
                | wyn_core::interface::StorageRole::Intermediate => (true, true),
            };
            mark(
                &mut synth,
                sb.binding.set,
                sb.binding.binding,
                sb.elem_ty.clone(),
                r,
                w,
            );
        }
        for input in &entry.inputs {
            if let Some(br) = input.storage_binding() {
                let elem_ty = input.ty.elem_type().cloned().unwrap_or_else(|| input.ty.clone());
                mark(&mut synth, br.set, br.binding, elem_ty, true, false);
            }
        }
        for out in &entry.outputs {
            if let Some(br) = out.storage_binding() {
                let elem_ty = out.ty.elem_type().cloned().unwrap_or_else(|| out.ty.clone());
                mark(&mut synth, br.set, br.binding, elem_ty, false, true);
            }
        }
    }
    for ((set, binding), (elem_ty, has_read, has_write)) in synth {
        let access = match (has_read, has_write) {
            (true, true) | (false, true) => "read_write",
            (true, false) => "read",
            (false, false) => "read",
        };
        storage.push(ResourceBinding {
            name: format!("_buf_{}_{}", set, binding),
            set,
            binding,
            ty: wyn_core::diags::format_type(&elem_ty),
            access: access.to_string(),
        });
    }

    ProgramInterface {
        entries,
        uniforms,
        storage,
    }
}

// =============================================================================
// WGSL compilation
// =============================================================================

#[derive(Serialize, Deserialize, Clone)]
pub struct CompileResultWgsl {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wgsl: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub interface: Option<ProgramInterface>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mir: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tlc: Option<Vec<TreeNode>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ErrorInfo>,
}

impl CompileResultWgsl {
    fn err(source: &str, e: CompilerError) -> Self {
        CompileResultWgsl {
            success: false,
            wgsl: None,
            interface: None,
            mir: None,
            tlc: None,
            error: Some(ErrorInfo {
                message: format_error(&e),
                location: error_location(source, &e),
            }),
        }
    }
    fn err_msg(message: String) -> Self {
        CompileResultWgsl {
            success: false,
            wgsl: None,
            interface: None,
            mir: None,
            tlc: None,
            error: Some(ErrorInfo {
                message,
                location: None,
            }),
        }
    }

    fn frontend_err(source: &str, failure: CompilationFailure) -> Self {
        let location = failure
            .error()
            .span()
            .filter(|span| span.module() == Some(failure.source_graph().root()))
            .and_then(|_| error_location(source, failure.error()));
        Self {
            success: false,
            wgsl: None,
            interface: None,
            mir: None,
            tlc: None,
            error: Some(ErrorInfo {
                message: failure.to_string(),
                location,
            }),
        }
    }

    fn source_modules_err(source: &str, error: SourceModulesError) -> Self {
        match error {
            SourceModulesError::Compiler(error) => Self::err(source, error),
            SourceModulesError::Setup(message) => Self::err_msg(message),
            SourceModulesError::Build(failure) => {
                let location = match failure.error() {
                    BuildError::Parse { source: error, .. } => error_location(source, error),
                    _ => None,
                };
                Self {
                    success: false,
                    wgsl: None,
                    interface: None,
                    mir: None,
                    tlc: None,
                    error: Some(ErrorInfo {
                        message: failure.to_string(),
                        location,
                    }),
                }
            }
        }
    }
}

/// Compile Wyn source to WGSL + emit the program interface (entries,
/// uniforms, storage) as structured JSON for WebGPU setup and for the
/// pipeline-visualization UI.
#[wasm_bindgen]
pub fn compile_to_wgsl(source: &str) -> JsValue {
    compile_to_wgsl_with_options(source, true, false)
}

/// Compile with explicit source-language and direct-output policy. The
/// playground should pass `(true, true)` so its graphical vocabulary is
/// enabled while hidden prepasses/resources remain forbidden.
#[wasm_bindgen]
pub fn compile_to_wgsl_with_options(source: &str, graphics: bool, direct: bool) -> JsValue {
    console_error_panic_hook::set_once();
    init_compiler();
    let result = compile_to_wgsl_impl(source, graphics, direct);
    serde_wasm_bindgen::to_value(&result).unwrap_or_else(|e| {
        let err = CompileResultWgsl::err_msg(format!("Serialization error: {}", e));
        serde_wasm_bindgen::to_value(&err).unwrap()
    })
}

fn compile_to_wgsl_impl(source: &str, graphics: bool, direct: bool) -> CompileResultWgsl {
    let modules = match load_source_modules(source, CompilerOptions { graphics }) {
        Ok(modules) => modules,
        Err(error) => return CompileResultWgsl::source_modules_err(source, error),
    };

    // Frontend → TLC → semantic EGIR → target-aware SSA lowering → WGSL.
    let program = match modules.type_check() {
        Ok(p) => p,
        Err(failure) => return CompileResultWgsl::frontend_err(source, failure),
    };
    let program = match wyn_core::ast_type_holes::reject_type_holes(program) {
        Ok(p) => p,
        Err(e) => return CompileResultWgsl::err(source, e),
    };

    let program = match wyn_core::tlc::lower_from_ast(program) {
        Ok(t) => t,
        Err(e) => return CompileResultWgsl::err(source, e),
    };
    let program = match wyn_core::tlc::pin_entry_buffers(program) {
        Ok(t) => t,
        Err(e) => return CompileResultWgsl::err(source, e),
    };
    let program = match wyn_core::tlc::validate_ownership(program) {
        Ok(t) => t,
        Err(e) => return CompileResultWgsl::err(source, e),
    };
    let program = wyn_core::tlc::partial_eval(program);
    let tlc_tree = tlc_tree::program_to_tree(&program);

    let program = wyn_core::tlc::normalize_soacs(program);
    let program = match wyn_core::tlc::monomorphize(program) {
        Ok(t) => t,
        Err(e) => return CompileResultWgsl::err(source, e),
    };
    let program = wyn_core::tlc::rep_specialize(program);
    let program = wyn_core::tlc::inline_small(program);
    let program = wyn_core::tlc::force_inline_soac_helpers(program);
    let program = wyn_core::tlc::renormalize_inlined_soa(program);
    let program = wyn_core::tlc::canonicalize_conditional_producers(program);
    let program = wyn_core::tlc::normalize_soacs_to_anf(program);
    let program = wyn_core::tlc::float_runtime_index_nested_producers(program);
    let program = wyn_core::tlc::defunctionalize(program);
    let program = wyn_core::tlc::fold_generated_lambdas(program);
    let program = wyn_core::tlc::apply_ownership(program);
    let program = wyn_core::tlc::filter_reachable(program);
    let program = wyn_core::tlc::infer_input_slice_bounds(program);
    let program = match wyn_core::to_egraph(program) {
        Ok(s) => s,
        Err(e) => return CompileResultWgsl::err_msg(format!("SSA conversion error: {:?}", e)),
    };
    let profile = if direct {
        LoweringProfile::with_topology(
            CodegenTarget::Wgsl,
            SchedulePolicy::Serial,
            PipelineTopologyPolicy::AuthoredOnly,
        )
    } else {
        LoweringProfile::new(CodegenTarget::Wgsl, SchedulePolicy::Parallel)
    };
    let lower = || -> Result<_, wyn_core::egir::from_tlc::ConvertError> {
        let program = wyn_core::egir::reify_soacs(program);
        let program = wyn_core::egir::optimize_semantic_operations(program)
            .map_err(|error| wyn_core::egir::from_tlc::ConvertError::Internal(error.to_string()))?;
        let program = wyn_core::egir::apply_pipeline_topology_policy(program, profile.topology);
        let program = wyn_core::egir::plan_logical_resources_with_policy(program, profile.topology)?;
        let program = wyn_core::egir::plan(program, profile)?;
        wyn_core::lower_egir_to_ssa(program)
    };
    let ssa = match lower() {
        Ok(s) => s,
        Err(e) => return CompileResultWgsl::err_msg(format!("SSA lowering error: {e}")),
    };
    let mir = wyn_core::ssa::print::format_program(&ssa);
    let interface = program_interface(&ssa);

    match wyn_core::lower_ssa_to_wgsl(ssa) {
        Ok(wgsl) => CompileResultWgsl {
            success: true,
            wgsl: Some(wgsl),
            interface: Some(interface),
            mir: Some(mir),
            tlc: Some(tlc_tree),
            error: None,
        },
        Err(e) => CompileResultWgsl::err(source, e),
    }
}

#[cfg(test)]
#[path = "lib_tests.rs"]
mod lib_tests;

/// Get a simple example program to start with
#[wasm_bindgen]
pub fn get_example_program() -> String {
    r#"-- Shadertoy-style Wyn example: one graphical operation that
-- rasterizes a full-screen triangle and shades the covered pixels.

def verts: [3]vec4f32 =
  [@[-1.0, -1.0, 0.0, 1.0],
   @[3.0, -1.0, 0.0, 1.0],
   @[-1.0, 3.0, 0.0, 1.0]]

def vertex_main(vertex: vertex_invocation) vertex<vec2f32> =
  vertex_output(verts[i32(vertex.vertex_index)], @[0.0, 0.0])

def fragment_main(iResolution: vec3f32,
                  iTime: f32,
                  iTimeDelta: f32,
                  iFrameRate: f32,
                  iFrame: i32,
                  iChannelTime: [4]f32,
                  iChannelResolution: [4]vec3f32,
                  iMouse: vec4f32,
                  iDate: vec4f32,
                  iSampleRate: f32,
                  fragment: fragment_invocation<vec2f32>) vec4f32 =
  let uv = fragment.position.xy / iResolution.xy in
  let phase = iTime in
  let r = 0.5 + 0.5 * f32.cos(phase + uv.x * 3.0 + 0.0) in
  let g = 0.5 + 0.5 * f32.cos(phase + uv.y * 3.0 + 2.0) in
  let b = 0.5 + 0.5 * f32.cos(phase + (uv.x + uv.y) * 1.5 + 4.0) in
  @[r, g, b, 1.0]

entry image(iResolution: vec3f32,
            iTime: f32,
            iTimeDelta: f32,
            iFrameRate: f32,
            iFrame: i32,
            iChannelTime: [4]f32,
            iChannelResolution: [4]vec3f32,
            iMouse: vec4f32,
            iDate: vec4f32,
            iSampleRate: f32,
            screen: render_target<vec4f32>) render_target<vec4f32> =
  let raster = rasterize_triangles(direct_draw(3u32, 1u32), vertex_main) in
  shade(screen, raster, |fragment|
    fragment_main(iResolution, iTime, iTimeDelta, iFrameRate, iFrame,
                  iChannelTime, iChannelResolution, iMouse, iDate,
                  iSampleRate, fragment))
"#
    .to_string()
}
