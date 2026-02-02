use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use wasm_bindgen::prelude::*;
use wyn_core::FrontEnd;
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
        TreeNode { name: name.into(), children: vec![] }
    }

    fn branch(name: impl Into<String>, children: Vec<TreeNode>) -> Self {
        TreeNode { name: name.into(), children }
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
        };
        let label = format!("{} {} : {}", meta, def.name, fmt_ty(&def.ty));
        TreeNode::branch(label, vec![term_to_tree(&def.body)])
    }

    fn term_to_tree(term: &Term) -> TreeNode {
        let ty = fmt_ty(&term.ty);
        match &term.kind {
            TermKind::Var(name) => TreeNode::leaf(format!("Var({}) : {}", name, ty)),
            TermKind::BinOp(op) => TreeNode::leaf(format!("BinOp({:?}) : {}", op, ty)),
            TermKind::UnOp(op) => TreeNode::leaf(format!("UnOp({:?}) : {}", op, ty)),
            TermKind::Lam { param, param_ty, body } => {
                let label = format!("Lam({}: {}) : {}", param, fmt_ty(param_ty), ty);
                TreeNode::branch(label, vec![term_to_tree(body)])
            }
            TermKind::App { func, arg } => {
                TreeNode::branch(format!("App : {}", ty), vec![
                    TreeNode::branch("func", vec![term_to_tree(func)]),
                    TreeNode::branch("arg", vec![term_to_tree(arg)]),
                ])
            }
            TermKind::Let { name, name_ty, rhs, body } => {
                let label = format!("Let({}: {})", name, fmt_ty(name_ty));
                TreeNode::branch(label, vec![
                    TreeNode::branch("rhs", vec![term_to_tree(rhs)]),
                    TreeNode::branch("body", vec![term_to_tree(body)]),
                ])
            }
            TermKind::IntLit(s) => TreeNode::leaf(format!("Int({}) : {}", s, ty)),
            TermKind::FloatLit(f) => TreeNode::leaf(format!("Float({}) : {}", f, ty)),
            TermKind::BoolLit(b) => TreeNode::leaf(format!("Bool({}) : {}", b, ty)),
            TermKind::StringLit(s) => TreeNode::leaf(format!("String({:?}) : {}", s, ty)),
            TermKind::Extern(link) => TreeNode::leaf(format!("Extern({}) : {}", link, ty)),
            TermKind::If { cond, then_branch, else_branch } => {
                TreeNode::branch(format!("If : {}", ty), vec![
                    TreeNode::branch("cond", vec![term_to_tree(cond)]),
                    TreeNode::branch("then", vec![term_to_tree(then_branch)]),
                    TreeNode::branch("else", vec![term_to_tree(else_branch)]),
                ])
            }
            TermKind::Loop { loop_var, loop_var_ty, init, init_bindings, kind, body } => {
                let label = format!("Loop({}: {})", loop_var, fmt_ty(loop_var_ty));
                let mut children = vec![TreeNode::branch("init", vec![term_to_tree(init)])];
                if !init_bindings.is_empty() {
                    let bindings: Vec<TreeNode> = init_bindings
                        .iter()
                        .map(|(n, t, e)| TreeNode::branch(format!("{}: {}", n, fmt_ty(t)), vec![term_to_tree(e)]))
                        .collect();
                    children.push(TreeNode::branch("bindings", bindings));
                }
                children.push(loop_kind_to_tree(kind));
                children.push(TreeNode::branch("body", vec![term_to_tree(body)]));
                TreeNode::branch(label, children)
            }
        }
    }

    fn loop_kind_to_tree(kind: &LoopKind) -> TreeNode {
        match kind {
            LoopKind::For { var, var_ty, iter } => {
                TreeNode::branch(format!("for {} : {}", var, fmt_ty(var_ty)), vec![term_to_tree(iter)])
            }
            LoopKind::ForRange { var, var_ty, bound } => {
                TreeNode::branch(format!("for_range {} : {}", var, fmt_ty(var_ty)), vec![term_to_tree(bound)])
            }
            LoopKind::While { cond } => TreeNode::branch("while", vec![term_to_tree(cond)]),
        }
    }
}

// =============================================================================
// MIR to Tree conversion
// =============================================================================

mod mir_tree {
    use super::TreeNode;
    use wyn_core::ast::TypeName;
    use wyn_core::mir::{ArrayBacking, Body, Def, Expr, ExprId, LoopKind, Program};

    fn fmt_ty(ty: &polytype::Type<TypeName>) -> String {
        wyn_core::diags::format_type(ty)
    }

    pub fn program_to_tree(program: &Program) -> Vec<TreeNode> {
        program.defs.iter().map(def_to_tree).collect()
    }

    fn def_to_tree(def: &Def) -> TreeNode {
        match def {
            Def::Function { name, params, ret_type, body, .. } => {
                let params_str = params
                    .iter()
                    .map(|lid| format!("_{}", lid.0))
                    .collect::<Vec<_>>()
                    .join(", ");
                let label = format!("fn {}({}) -> {}", name, params_str, fmt_ty(ret_type));
                TreeNode::branch(label, vec![body_to_tree(body)])
            }
            Def::Constant { name, ty, body, .. } => {
                let label = format!("const {}: {}", name, fmt_ty(ty));
                TreeNode::branch(label, vec![body_to_tree(body)])
            }
            Def::Uniform { name, ty, set, binding, .. } => {
                TreeNode::leaf(format!("uniform {} : {} @set={} @binding={}", name, fmt_ty(ty), set, binding))
            }
            Def::Storage { name, ty, set, binding, .. } => {
                TreeNode::leaf(format!("storage {} : {} @set={} @binding={}", name, fmt_ty(ty), set, binding))
            }
            Def::EntryPoint { name, execution_model, body, .. } => {
                let label = format!("entry {} ({:?})", name, execution_model);
                TreeNode::branch(label, vec![body_to_tree(body)])
            }
        }
    }

    fn body_to_tree(body: &Body) -> TreeNode {
        expr_to_tree(body, body.root)
    }

    fn expr_to_tree(body: &Body, id: ExprId) -> TreeNode {
        let expr = &body.exprs[id.index()];
        let ty = fmt_ty(&body.types[id.index()]);

        match expr {
            Expr::Local(lid) => {
                let name = body.get_local(*lid).name.as_str();
                TreeNode::leaf(format!("Local(_{} {}) : {}", lid.0, name, ty))
            }
            Expr::Global(name) => TreeNode::leaf(format!("Global({}) : {}", name, ty)),
            Expr::Extern(name) => TreeNode::leaf(format!("Extern({}) : {}", name, ty)),
            Expr::Int(s) => TreeNode::leaf(format!("Int({}) : {}", s, ty)),
            Expr::Float(s) => TreeNode::leaf(format!("Float({}) : {}", s, ty)),
            Expr::Bool(b) => TreeNode::leaf(format!("Bool({}) : {}", b, ty)),
            Expr::Unit => TreeNode::leaf(format!("Unit : {}", ty)),
            Expr::String(s) => TreeNode::leaf(format!("String({:?}) : {}", s, ty)),
            Expr::Tuple(elems) => {
                let children: Vec<_> = elems.iter().map(|e| expr_to_tree(body, *e)).collect();
                TreeNode::branch(format!("Tuple : {}", ty), children)
            }
            Expr::Array { backing, size } => {
                TreeNode::branch(format!("Array : {}", ty), vec![
                    backing_to_tree(body, backing),
                    TreeNode::branch("size", vec![expr_to_tree(body, *size)]),
                ])
            }
            Expr::Vector(elems) => {
                let children: Vec<_> = elems.iter().map(|e| expr_to_tree(body, *e)).collect();
                TreeNode::branch(format!("Vector : {}", ty), children)
            }
            Expr::Matrix(rows) => {
                let children: Vec<_> = rows
                    .iter()
                    .enumerate()
                    .map(|(i, row)| {
                        let row_children: Vec<_> = row.iter().map(|e| expr_to_tree(body, *e)).collect();
                        TreeNode::branch(format!("row{}", i), row_children)
                    })
                    .collect();
                TreeNode::branch(format!("Matrix : {}", ty), children)
            }
            Expr::BinOp { op, lhs, rhs } => {
                TreeNode::branch(format!("BinOp({}) : {}", op, ty), vec![
                    expr_to_tree(body, *lhs),
                    expr_to_tree(body, *rhs),
                ])
            }
            Expr::UnaryOp { op, operand } => {
                TreeNode::branch(format!("UnaryOp({}) : {}", op, ty), vec![expr_to_tree(body, *operand)])
            }
            Expr::Let { local, rhs, body: let_body } => {
                let name = body.get_local(*local).name.as_str();
                TreeNode::branch(format!("Let(_{} {})", local.0, name), vec![
                    TreeNode::branch("rhs", vec![expr_to_tree(body, *rhs)]),
                    TreeNode::branch("body", vec![expr_to_tree(body, *let_body)]),
                ])
            }
            Expr::If { cond, then_, else_ } => {
                TreeNode::branch(format!("If : {}", ty), vec![
                    TreeNode::branch("cond", vec![expr_to_tree(body, *cond)]),
                    TreeNode::branch("then", vec![expr_to_tree(body, *then_)]),
                    TreeNode::branch("else", vec![expr_to_tree(body, *else_)]),
                ])
            }
            Expr::Loop { loop_var, init, init_bindings, kind, body: loop_body } => {
                let name = body.get_local(*loop_var).name.as_str();
                let mut children = vec![TreeNode::branch("init", vec![expr_to_tree(body, *init)])];
                if !init_bindings.is_empty() {
                    let bindings: Vec<_> = init_bindings
                        .iter()
                        .map(|(lid, e)| {
                            let bname = body.get_local(*lid).name.as_str();
                            TreeNode::branch(format!("_{} {}", lid.0, bname), vec![expr_to_tree(body, *e)])
                        })
                        .collect();
                    children.push(TreeNode::branch("bindings", bindings));
                }
                children.push(loop_kind_to_tree(body, kind));
                children.push(TreeNode::branch("body", vec![expr_to_tree(body, *loop_body)]));
                TreeNode::branch(format!("Loop(_{} {})", loop_var.0, name), children)
            }
            Expr::Call { func, args } => {
                let children: Vec<_> = args.iter().map(|a| expr_to_tree(body, *a)).collect();
                TreeNode::branch(format!("Call({}) : {}", func, ty), children)
            }
            Expr::Intrinsic { name, args } => {
                let children: Vec<_> = args.iter().map(|a| expr_to_tree(body, *a)).collect();
                TreeNode::branch(format!("Intrinsic({}) : {}", name, ty), children)
            }
            Expr::Materialize(e) => {
                TreeNode::branch(format!("Materialize : {}", ty), vec![expr_to_tree(body, *e)])
            }
            Expr::Attributed { attributes, expr } => {
                let attrs = attributes.iter().map(|a| format!("{:?}", a)).collect::<Vec<_>>().join(", ");
                TreeNode::branch(format!("Attributed [{}] : {}", attrs, ty), vec![expr_to_tree(body, *expr)])
            }
            Expr::Load { ptr } => {
                TreeNode::branch(format!("Load : {}", ty), vec![expr_to_tree(body, *ptr)])
            }
            Expr::Store { ptr, value } => {
                TreeNode::branch(format!("Store : {}", ty), vec![
                    TreeNode::branch("ptr", vec![expr_to_tree(body, *ptr)]),
                    TreeNode::branch("value", vec![expr_to_tree(body, *value)]),
                ])
            }
            Expr::StorageView { set, binding, offset, len } => {
                TreeNode::branch(format!("StorageView(@{},{}) : {}", set, binding, ty), vec![
                    TreeNode::branch("offset", vec![expr_to_tree(body, *offset)]),
                    TreeNode::branch("len", vec![expr_to_tree(body, *len)]),
                ])
            }
            Expr::SliceStorageView { view, start, len } => {
                TreeNode::branch(format!("SliceStorageView : {}", ty), vec![
                    TreeNode::branch("view", vec![expr_to_tree(body, *view)]),
                    TreeNode::branch("start", vec![expr_to_tree(body, *start)]),
                    TreeNode::branch("len", vec![expr_to_tree(body, *len)]),
                ])
            }
            Expr::StorageViewIndex { view, index } => {
                TreeNode::branch(format!("StorageViewIndex : {}", ty), vec![
                    TreeNode::branch("view", vec![expr_to_tree(body, *view)]),
                    TreeNode::branch("index", vec![expr_to_tree(body, *index)]),
                ])
            }
            Expr::StorageViewLen { view } => {
                TreeNode::branch(format!("StorageViewLen : {}", ty), vec![expr_to_tree(body, *view)])
            }
        }
    }

    fn backing_to_tree(body: &Body, backing: &ArrayBacking) -> TreeNode {
        match backing {
            ArrayBacking::Literal(elems) => {
                let children: Vec<_> = elems.iter().map(|e| expr_to_tree(body, *e)).collect();
                TreeNode::branch("Literal", children)
            }
            ArrayBacking::Range { start, step, kind } => {
                let mut children = vec![TreeNode::branch("start", vec![expr_to_tree(body, *start)])];
                if let Some(s) = step {
                    children.push(TreeNode::branch("step", vec![expr_to_tree(body, *s)]));
                }
                TreeNode::branch(format!("Range({:?})", kind), children)
            }
        }
    }

    fn loop_kind_to_tree(body: &Body, kind: &LoopKind) -> TreeNode {
        match kind {
            LoopKind::For { var, iter } => {
                let name = body.get_local(*var).name.as_str();
                TreeNode::branch(format!("for _{} {}", var.0, name), vec![expr_to_tree(body, *iter)])
            }
            LoopKind::ForRange { var, bound } => {
                let name = body.get_local(*var).name.as_str();
                TreeNode::branch(format!("for_range _{} {}", var.0, name), vec![expr_to_tree(body, *bound)])
            }
            LoopKind::While { cond } => TreeNode::branch("while", vec![expr_to_tree(body, *cond)]),
        }
    }
}

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

/// Create a fresh FrontEnd using the cached prelude
fn create_frontend() -> Option<FrontEnd> {
    PRELUDE_CACHE.with(|cache| {
        let cache_ref = cache.borrow();
        let cached = cache_ref.as_ref()?;
        Some(FrontEnd::new_from_prelude(
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
#[derive(Serialize, Deserialize)]
pub struct ErrorInfo {
    pub message: String,
    pub location: Option<ErrorLocation>,
}

/// Result of compiling Wyn source to GLSL
#[derive(Serialize, Deserialize)]
pub struct CompileResult {
    pub success: bool,
    pub glsl: Option<String>,
    pub error: Option<ErrorInfo>,
}

impl CompileResult {
    fn ok(glsl: String) -> Self {
        CompileResult {
            success: true,
            glsl: Some(glsl),
            error: None,
        }
    }

    fn err(e: CompilerError) -> Self {
        let location = e.span().map(|s| ErrorLocation {
            start_line: s.start_line,
            start_col: s.start_col,
            end_line: s.end_line,
            end_col: s.end_col,
        });

        // Get just the message without the span debug info
        let message = match &e {
            CompilerError::ParseError(msg, _) => format!("Parse error: {}", msg),
            CompilerError::TypeError(msg, _) => format!("Type error: {}", msg),
            CompilerError::UndefinedVariable(name, _) => format!("Undefined variable: '{}'", name),
            CompilerError::AliasError(msg, _) => format!("Alias error: {}", msg),
            CompilerError::SpirvError(msg, _) => format!("SPIR-V error: {}", msg),
            CompilerError::GlslError(msg, _) => format!("GLSL error: {}", msg),
            CompilerError::ModuleError(msg, _) => format!("Module error: {}", msg),
            CompilerError::FlatteningError(msg, _) => format!("Flatten error: {}", msg),
            CompilerError::IoError(err) => format!("IO error: {}", err),
            CompilerError::SpirvBuilderError(msg) => format!("SPIR-V builder error: {}", msg),
        };

        CompileResult {
            success: false,
            glsl: None,
            error: Some(ErrorInfo { message, location }),
        }
    }

    fn err_msg(message: String) -> Self {
        CompileResult {
            success: false,
            glsl: None,
            error: Some(ErrorInfo {
                message,
                location: None,
            }),
        }
    }
}

/// Extended result with IR trees for visualization
#[derive(Serialize, Deserialize)]
pub struct CompileResultWithIR {
    pub success: bool,
    pub glsl: Option<String>,
    pub tlc: Option<Vec<TreeNode>>,
    pub initial_mir: Option<Vec<TreeNode>>,
    pub final_mir: Option<Vec<TreeNode>>,
    pub error: Option<ErrorInfo>,
}

fn format_error(e: &CompilerError) -> String {
    match e {
        CompilerError::ParseError(msg, _) => format!("Parse error: {}", msg),
        CompilerError::TypeError(msg, _) => format!("Type error: {}", msg),
        CompilerError::UndefinedVariable(name, _) => format!("Undefined variable: '{}'", name),
        CompilerError::AliasError(msg, _) => format!("Alias error: {}", msg),
        CompilerError::SpirvError(msg, _) => format!("SPIR-V error: {}", msg),
        CompilerError::GlslError(msg, _) => format!("GLSL error: {}", msg),
        CompilerError::ModuleError(msg, _) => format!("Module error: {}", msg),
        CompilerError::FlatteningError(msg, _) => format!("Flatten error: {}", msg),
        CompilerError::IoError(err) => format!("IO error: {}", err),
        CompilerError::SpirvBuilderError(msg) => format!("SPIR-V builder error: {}", msg),
    }
}

fn error_location(e: &CompilerError) -> Option<ErrorLocation> {
    e.span().map(|s| ErrorLocation {
        start_line: s.start_line,
        start_col: s.start_col,
        end_line: s.end_line,
        end_col: s.end_col,
    })
}

impl CompileResultWithIR {
    fn err(e: CompilerError) -> Self {
        CompileResultWithIR {
            success: false,
            glsl: None,
            tlc: None,
            initial_mir: None,
            final_mir: None,
            error: Some(ErrorInfo {
                message: format_error(&e),
                location: error_location(&e),
            }),
        }
    }

    fn err_msg(message: String) -> Self {
        CompileResultWithIR {
            success: false,
            glsl: None,
            tlc: None,
            initial_mir: None,
            final_mir: None,
            error: Some(ErrorInfo { message, location: None }),
        }
    }
}

/// Compile Wyn source code to Shadertoy-compatible GLSL.
///
/// Returns a JSON-serialized CompileResult.
/// Note: init_compiler() should be called first, but this will auto-initialize if needed.
#[wasm_bindgen]
pub fn compile_to_shadertoy(source: &str) -> JsValue {
    // Set up panic hook for better error messages
    console_error_panic_hook::set_once();

    // Auto-initialize if not already done
    init_compiler();

    let result = compile_impl(source);
    serde_wasm_bindgen::to_value(&result).unwrap_or_else(|e| {
        let error_result = CompileResult::err_msg(format!("Serialization error: {}", e));
        serde_wasm_bindgen::to_value(&error_result).unwrap()
    })
}

fn compile_impl(source: &str) -> CompileResult {
    // Create frontend from cached prelude
    let mut frontend = match create_frontend() {
        Some(f) => f,
        None => return CompileResult::err_msg("Compiler not initialized".to_string()),
    };

    // Parse
    let parsed = match wyn_core::Compiler::parse(source, &mut frontend.node_counter) {
        Ok(p) => p,
        Err(e) => return CompileResult::err(e),
    };

    // Desugar
    let desugared = match parsed.desugar(&mut frontend.node_counter) {
        Ok(d) => d,
        Err(e) => return CompileResult::err(e),
    };

    // Resolve names
    let resolved = match desugared.resolve(&frontend.module_manager) {
        Ok(r) => r,
        Err(e) => return CompileResult::err(e),
    };

    // Fold AST constants
    let ast_folded = resolved.fold_ast_constants();

    // Type check
    let type_checked = match ast_folded.type_check(&mut frontend.module_manager, &mut frontend.schemes) {
        Ok(t) => t,
        Err(e) => return CompileResult::err(e),
    };

    // Alias check
    let alias_checked = match type_checked.alias_check() {
        Ok(a) => a,
        Err(e) => return CompileResult::err(e),
    };

    if alias_checked.has_alias_errors() {
        return CompileResult::err_msg("Alias checking failed".to_string());
    }

    // Build builtins set for lambda lifting
    let builtins = wyn_core::build_known_defs(&alias_checked.ast, &frontend.module_manager);

    // Transform to TLC, monomorphize, then to MIR
    let flattened = alias_checked
        .to_tlc(builtins, &frontend.schemes, &frontend.module_manager)
        .partial_eval()
        .defunctionalize()
        .monomorphize()
        .to_mir();

    // MIR passes
    let hoisted = flattened.hoist_materializations();
    let normalized = hoisted.normalize();
    let defaulted = normalized.default_address_spaces();
    let parallelized = defaulted.parallelize_soacs();
    let reachable = parallelized.filter_reachable();
    let lifted = reachable.lift_bindings();

    // Lower to Shadertoy GLSL
    match lifted.lower_shadertoy() {
        Ok(glsl) => CompileResult::ok(glsl),
        Err(e) => CompileResult::err(e),
    }
}

/// Compile Wyn source code and return IR trees along with GLSL.
///
/// Returns a JSON-serialized CompileResultWithIR.
#[wasm_bindgen]
pub fn compile_with_ir(source: &str) -> JsValue {
    console_error_panic_hook::set_once();
    init_compiler();

    let result = compile_with_ir_impl(source);
    serde_wasm_bindgen::to_value(&result).unwrap_or_else(|e| {
        let error_result = CompileResultWithIR::err_msg(format!("Serialization error: {}", e));
        serde_wasm_bindgen::to_value(&error_result).unwrap()
    })
}

fn compile_with_ir_impl(source: &str) -> CompileResultWithIR {
    let mut frontend = match create_frontend() {
        Some(f) => f,
        None => return CompileResultWithIR::err_msg("Compiler not initialized".to_string()),
    };

    // Parse
    let parsed = match wyn_core::Compiler::parse(source, &mut frontend.node_counter) {
        Ok(p) => p,
        Err(e) => return CompileResultWithIR::err(e),
    };

    // Desugar
    let desugared = match parsed.desugar(&mut frontend.node_counter) {
        Ok(d) => d,
        Err(e) => return CompileResultWithIR::err(e),
    };

    // Resolve names
    let resolved = match desugared.resolve(&frontend.module_manager) {
        Ok(r) => r,
        Err(e) => return CompileResultWithIR::err(e),
    };

    // Fold AST constants
    let ast_folded = resolved.fold_ast_constants();

    // Type check
    let type_checked = match ast_folded.type_check(&mut frontend.module_manager, &mut frontend.schemes) {
        Ok(t) => t,
        Err(e) => return CompileResultWithIR::err(e),
    };

    // Alias check
    let alias_checked = match type_checked.alias_check() {
        Ok(a) => a,
        Err(e) => return CompileResultWithIR::err(e),
    };

    if alias_checked.has_alias_errors() {
        return CompileResultWithIR::err_msg("Alias checking failed".to_string());
    }

    // Build builtins set
    let builtins = wyn_core::build_known_defs(&alias_checked.ast, &frontend.module_manager);

    // Transform to TLC
    let tlc_program = alias_checked.to_tlc(builtins, &frontend.schemes, &frontend.module_manager);

    // Capture TLC tree (after partial eval, before defunctionalization)
    let tlc_after_partial_eval = tlc_program.partial_eval();
    let tlc_tree = tlc_tree::program_to_tree(&tlc_after_partial_eval.tlc);

    // Continue pipeline: defunctionalize, monomorphize, then to MIR
    let flattened = tlc_after_partial_eval
        .defunctionalize()
        .monomorphize()
        .to_mir();

    // Capture initial MIR (after flattening, before MIR passes)
    let initial_mir_tree = mir_tree::program_to_tree(&flattened.mir);

    // MIR passes
    let hoisted = flattened.hoist_materializations();
    let normalized = hoisted.normalize();
    let defaulted = normalized.default_address_spaces();
    let parallelized = defaulted.parallelize_soacs();
    let reachable = parallelized.filter_reachable();
    let lifted = reachable.lift_bindings();

    // Capture final MIR
    let final_mir_tree = mir_tree::program_to_tree(&lifted.mir);

    // Lower to Shadertoy GLSL
    match lifted.lower_shadertoy() {
        Ok(glsl) => CompileResultWithIR {
            success: true,
            glsl: Some(glsl),
            tlc: Some(tlc_tree),
            initial_mir: Some(initial_mir_tree),
            final_mir: Some(final_mir_tree),
            error: None,
        },
        Err(e) => CompileResultWithIR::err(e),
    }
}

/// Get a simple example program to start with
#[wasm_bindgen]
pub fn get_example_program() -> String {
    r#"-- Wyn Shader Example
-- This compiles to Shadertoy-compatible GLSL

------------------------------------------------------------
-- Uniforms
------------------------------------------------------------
#[uniform(set=0, binding=0)] def iResolution: vec2f32
#[uniform(set=0, binding=1)] def iTime: f32

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
