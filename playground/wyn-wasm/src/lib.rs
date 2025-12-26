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
    "003".to_string()
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
            &cached.prelude,
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
    let type_checked = match ast_folded.type_check(&frontend.module_manager, &mut frontend.schemes) {
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

    // Flatten to MIR
    let (flattened, _backend) = match alias_checked.flatten(&frontend.module_manager, &frontend.schemes) {
        Ok(f) => f,
        Err(e) => return CompileResult::err(e),
    };

    // MIR passes
    let hoisted = flattened.hoist_materializations();
    let normalized = hoisted.normalize();
    let monomorphized = match normalized.monomorphize() {
        Ok(m) => m,
        Err(e) => return CompileResult::err(e),
    };

    // Use partial eval for better optimization
    let folded = match monomorphized.partial_eval() {
        Ok(f) => f,
        Err(e) => return CompileResult::err(e),
    };

    let reachable = folded.filter_reachable();
    let lifted = reachable.lift_bindings();

    // Lower to Shadertoy GLSL
    match lifted.lower_shadertoy() {
        Ok(glsl) => CompileResult::ok(glsl),
        Err(e) => CompileResult::err(e),
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
def vertex_main(#[builtin(vertex_index)] vertex_id: i32) -> #[builtin(position)] vec4f32 = verts[vertex_id]

------------------------------------------------------------
-- Fragment shader
------------------------------------------------------------
#[fragment]
def fragment_main(#[builtin(position)] fragCoord: vec4f32) -> #[location(0)] vec4f32 =
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
