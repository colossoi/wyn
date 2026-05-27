use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use wasm_bindgen::prelude::*;
use wyn_core::ast::NodeCounter;
use wyn_core::error::CompilerError;
use wyn_core::module_manager::{ModuleManager, PreElaboratedPrelude};

/// Cached prelude and starting node counter
/// Creating the prelude parses all prelude files, which is expensive.
/// We cache this and create fresh FrontEnds from it for each compilation.
struct PreludeCache {
    prelude: PreElaboratedPrelude,
    start_node_counter: NodeCounter,
}

thread_local! {
    static PRELUDE_CACHE: RefCell<Option<PreludeCache>> = RefCell::new(None);
}

/// Get the compiler version string
#[wasm_bindgen]
pub fn version() -> String {
    "004".to_string()
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
    use wyn_core::tlc::{Def, DefMeta, LoopKind, Program, Term, TermKind};

    fn fmt_ty(ty: &polytype::Type<TypeName>) -> String {
        wyn_core::diags::format_type(ty)
    }

    pub fn program_to_tree(program: &Program) -> Vec<TreeNode> {
        program.defs.iter().map(def_to_tree).collect()
    }

    fn def_to_tree(def: &Def) -> TreeNode {
        let meta = match &def.meta {
            DefMeta::Function => "fn",
            DefMeta::EntryPoint(_) => "entry",
            DefMeta::LiftedLambda => "lifted",
        };
        let label = format!("{} {} : {}", meta, def.name, fmt_ty(&def.ty));
        TreeNode::branch(label, vec![term_to_tree(&def.body)])
    }

    fn term_to_tree(term: &Term) -> TreeNode {
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
            TermKind::Soac(_) | TermKind::ArrayExpr(_) => TreeNode::leaf(format!("<soac> : {}", ty)),
            other => TreeNode::leaf(format!("{:?} : {}", other, ty)),
        }
    }

    fn loop_kind_to_tree(kind: &LoopKind) -> TreeNode {
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

// =============================================================================
// MIR to Tree conversion (removed — old MIR types no longer exist)
// =============================================================================

/// Initialize the compiler cache. Call this once at startup.
/// Returns true on success.
#[wasm_bindgen]
pub fn init_compiler() -> bool {
    console_error_panic_hook::set_once();

    PRELUDE_CACHE.with(|cache| {
        if cache.borrow().is_some() {
            return true; // Already initialized
        }

        let mut node_counter = NodeCounter::new();
        match ModuleManager::create_prelude(&mut node_counter) {
            Ok(prelude) => {
                *cache.borrow_mut() = Some(PreludeCache {
                    prelude,
                    start_node_counter: node_counter,
                });
                true
            }
            Err(e) => {
                web_sys::console::error_1(&format!("Failed to initialize prelude: {:?}", e).into());
                false
            }
        }
    })
}

/// Build a fresh `(NodeCounter, ModuleManager)` pair from the cached prelude.
fn create_compiler_init() -> Option<(NodeCounter, ModuleManager)> {
    PRELUDE_CACHE.with(|cache| {
        let cache_ref = cache.borrow();
        let cached = cache_ref.as_ref()?;
        Some(wyn_core::init_compiler_from_prelude(
            cached.prelude.clone(),
            cached.start_node_counter.clone(),
        ))
    })
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

fn error_location(e: &CompilerError) -> Option<ErrorLocation> {
    e.span().map(|s| ErrorLocation {
        start_line: s.start_line,
        start_col: s.start_col,
        end_line: s.end_line,
        end_col: s.end_col,
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
    /// WGSL-mangled entry-point name — this is what WebGPU's
    /// `entryPoint:` in `createRenderPipeline` / `createComputePipeline`
    /// needs, since the WGSL backend mangles all user identifiers.
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
    /// `"builtin(<name>)"`, `"location(<n>)"`, `"storage(<set>,<binding>)"`,
    /// `"uniform(<set>,<binding>)"`, `"texture(<set>,<binding>)"`,
    /// `"sampler(<set>,<binding>)"`, `"push_constant(<offset>)"`, or
    /// `"unknown"`.
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

fn entry_binding_from_input(input: &wyn_core::ssa::types::EntryInput) -> EntryBinding {
    use wyn_core::ssa::types::IoDecoration;
    let decoration = if let Some((s, b)) = input.storage_binding {
        format!("storage({},{})", s, b)
    } else if let Some((s, b)) = input.uniform_binding {
        format!("uniform({},{})", s, b)
    } else if let Some((s, b)) = input.texture_binding {
        format!("texture({},{})", s, b)
    } else if let Some((s, b)) = input.sampler_binding {
        format!("sampler({},{})", s, b)
    } else if let Some(off) = input.push_constant_offset {
        format!("push_constant({})", off)
    } else {
        match &input.decoration {
            Some(IoDecoration::BuiltIn(b)) => format!("builtin({:?})", b),
            Some(IoDecoration::Location(n)) => format!("location({})", n),
            None => "unknown".to_string(),
        }
    };
    EntryBinding {
        name: input.name.clone(),
        ty: fmt_ssa_type(&input.ty),
        decoration,
    }
}

fn entry_binding_from_output(idx: usize, output: &wyn_core::ssa::types::EntryOutput) -> EntryBinding {
    use wyn_core::ssa::types::IoDecoration;
    let decoration = if let Some((s, b)) = output.storage_binding {
        format!("storage({},{})", s, b)
    } else {
        match &output.decoration {
            Some(IoDecoration::BuiltIn(b)) => format!("builtin({:?})", b),
            Some(IoDecoration::Location(n)) => format!("location({})", n),
            None => "unknown".to_string(),
        }
    };
    EntryBinding {
        name: format!("out{}", idx),
        ty: fmt_ssa_type(&output.ty),
        decoration,
    }
}

fn program_interface(program: &wyn_core::ssa::types::Program) -> ProgramInterface {
    use wyn_core::ssa::types::ExecutionModel;
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
                let already = e.inputs.iter().any(|i| i.storage_binding == Some((sb.set, sb.binding)))
                    || e.outputs.iter().any(|o| o.storage_binding == Some((sb.set, sb.binding)));
                if already {
                    continue;
                }
                let role = match sb.role {
                    wyn_core::interface::StorageRole::Input => "in",
                    wyn_core::interface::StorageRole::Output => "out",
                    wyn_core::interface::StorageRole::Intermediate => "tmp",
                };
                inputs.push(EntryBinding {
                    name: format!("_buf_{}_{}_{}", sb.set, sb.binding, role),
                    ty: fmt_ssa_type(&sb.elem_ty),
                    decoration: format!("storage({},{})", sb.set, sb.binding),
                });
            }
            let outputs: Vec<EntryBinding> =
                e.outputs.iter().enumerate().map(|(i, o)| entry_binding_from_output(i, o)).collect();
            EntryInterface {
                name: e.name.clone(),
                wgsl_name: wyn_core::wgsl::ssa_lowering::wgsl_mangle(&e.name),
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
            if let Some((set, binding)) = input.uniform_binding {
                uniforms_by_slot.entry((set, binding)).or_insert_with(|| ResourceBinding {
                    name: input.name.clone(),
                    set,
                    binding,
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
                wyn_core::interface::StorageRole::Intermediate => (true, true),
            };
            mark(&mut synth, sb.set, sb.binding, sb.elem_ty.clone(), r, w);
        }
        for input in &entry.inputs {
            if let Some((set, binding)) = input.storage_binding {
                let elem_ty = input.ty.elem_type().cloned().unwrap_or_else(|| input.ty.clone());
                mark(&mut synth, set, binding, elem_ty, true, false);
            }
        }
        for out in &entry.outputs {
            if let Some((set, binding)) = out.storage_binding {
                let elem_ty = out.ty.elem_type().cloned().unwrap_or_else(|| out.ty.clone());
                mark(&mut synth, set, binding, elem_ty, false, true);
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
    fn err(e: CompilerError) -> Self {
        CompileResultWgsl {
            success: false,
            wgsl: None,
            interface: None,
            mir: None,
            tlc: None,
            error: Some(ErrorInfo {
                message: format_error(&e),
                location: error_location(&e),
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
}

/// Compile Wyn source to WGSL + emit the program interface (entries,
/// uniforms, storage) as structured JSON for WebGPU setup and for the
/// pipeline-visualization UI.
#[wasm_bindgen]
pub fn compile_to_wgsl(source: &str) -> JsValue {
    console_error_panic_hook::set_once();
    init_compiler();
    let result = compile_to_wgsl_impl(source);
    serde_wasm_bindgen::to_value(&result).unwrap_or_else(|e| {
        let err = CompileResultWgsl::err_msg(format!("Serialization error: {}", e));
        serde_wasm_bindgen::to_value(&err).unwrap()
    })
}

fn compile_to_wgsl_impl(source: &str) -> CompileResultWgsl {
    let (mut node_counter, mut module_manager) = match create_compiler_init() {
        Some(f) => f,
        None => return CompileResultWgsl::err_msg("Compiler not initialized".to_string()),
    };

    // Frontend pipeline: parse → elaborate → resolve → fold → type-check →
    // TLC → EGIR (`expand_soacs(true) → materialize → optimize_skeleton →
    // elaborate`) → WGSL.
    let parsed = match wyn_core::Compiler::parse(source, &mut node_counter) {
        Ok(p) => p,
        Err(e) => return CompileResultWgsl::err(e),
    };
    let parsed = match parsed.elaborate_modules(&mut module_manager, &mut node_counter) {
        Ok(p) => p,
        Err(e) => return CompileResultWgsl::err(e),
    };
    let resolved = match parsed.resolve(&module_manager) {
        Ok(r) => r,
        Err(e) => return CompileResultWgsl::err(e),
    };
    let ast_folded = resolved.fold_ast_constants();
    let type_checked = match ast_folded.type_check(&mut module_manager) {
        Ok(t) => t,
        Err(e) => return CompileResultWgsl::err(e),
    };

    let tlc_program = type_checked.to_tlc(&module_manager, false);
    let tlc_after_partial_eval = tlc_program.partial_eval();
    let tlc_tree = tlc_tree::program_to_tree(&tlc_after_partial_eval.tlc);

    let tlc_with_ownership = match tlc_after_partial_eval.normalize_soacs().fuse_maps().apply_ownership() {
        Ok(t) => t,
        Err(e) => return CompileResultWgsl::err_msg(format!("apply_ownership: {:?}", e)),
    };
    let tlc_parallelized = match tlc_with_ownership
        .lift_gathers()
        .defunctionalize()
        .monomorphize()
        .buffer_specialize()
        .fold_generated_lambdas()
        .inline_small()
        .parallelize_soacs(false)
    {
        Ok(t) => t,
        Err(e) => return CompileResultWgsl::err_msg(format!("parallelize_soacs: {:?}", e)),
    };
    let raw = match tlc_parallelized.filter_reachable().to_egraph() {
        Ok(s) => s,
        Err(e) => return CompileResultWgsl::err_msg(format!("SSA conversion error: {:?}", e)),
    };
    let ssa = raw.expand_soacs(true).materialize().optimize_skeleton().elaborate();
    let mir = wyn_core::ssa::print::format_program(&ssa.ssa);
    let interface = program_interface(&ssa.ssa);

    match ssa.lower_wgsl() {
        Ok(wgsl) => CompileResultWgsl {
            success: true,
            wgsl: Some(wgsl),
            interface: Some(interface),
            mir: Some(mir),
            tlc: Some(tlc_tree),
            error: None,
        },
        Err(e) => CompileResultWgsl::err(e),
    }
}

#[cfg(test)]
#[path = "lib_tests.rs"]
mod lib_tests;

/// Get a simple example program to start with
#[wasm_bindgen]
pub fn get_example_program() -> String {
    r#"-- Wyn Shader Example
-- This compiles to WGSL for WebGPU

------------------------------------------------------------
-- Uniforms
------------------------------------------------------------
#[uniform(set=1, binding=0)] def iResolution: vec3f32
#[uniform(set=1, binding=1)] def iTime: f32

------------------------------------------------------------
-- Vertex shader: full-screen triangle
------------------------------------------------------------
def verts: [3]vec4f32 =
  [@[-1.0, -1.0, 0.0, 1.0],
   @[3.0, -1.0, 0.0, 1.0],
   @[-1.0, 3.0, 0.0, 1.0]]

#[vertex]
entry vertex_main(#[builtin(vertex_index)] vertex_id: i32)  #[builtin(position)] vec4f32 =
  verts[vertex_id]

------------------------------------------------------------
-- Fragment shader
------------------------------------------------------------
#[fragment]
entry fragment_main(#[builtin(position)] fragCoord: vec4f32) #[location(0)] vec4f32 =
  -- Flip Y for Vulkan
  let coord = @[fragCoord.x, iResolution.y - fragCoord.y] in
  let uv = @[coord.x / iResolution.x, coord.y / iResolution.y] in

  -- Colorful animated gradient
  let phase = iTime in
  let r = 0.5 + 0.5 * f32.cos(phase + uv.x * 3.0 + 0.0) in
  let g = 0.5 + 0.5 * f32.cos(phase + uv.y * 3.0 + 2.0) in
  let b = 0.5 + 0.5 * f32.cos(phase + (uv.x + uv.y) * 1.5 + 4.0) in
  @[r, g, b, 1.0]
"#
    .to_string()
}
