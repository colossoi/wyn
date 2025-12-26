use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// Result of compiling Wyn source to GLSL
#[derive(Serialize, Deserialize)]
pub struct CompileResult {
    pub success: bool,
    pub glsl: Option<String>,
    pub error: Option<String>,
}

/// Compile Wyn source code to Shadertoy-compatible GLSL.
///
/// Returns a JSON-serialized CompileResult.
#[wasm_bindgen]
pub fn compile_to_shadertoy(source: &str) -> JsValue {
    // Set up panic hook for better error messages
    console_error_panic_hook::set_once();

    let result = compile_impl(source);
    serde_wasm_bindgen::to_value(&result).unwrap_or_else(|e| {
        let error_result = CompileResult {
            success: false,
            glsl: None,
            error: Some(format!("Serialization error: {}", e)),
        };
        serde_wasm_bindgen::to_value(&error_result).unwrap()
    })
}

fn compile_impl(source: &str) -> CompileResult {
    // Create frontend (loads prelude, sets up module manager)
    let mut frontend = wyn_core::FrontEnd::new();

    // Parse
    let parsed = match wyn_core::Compiler::parse(source, &mut frontend.node_counter) {
        Ok(p) => p,
        Err(e) => {
            return CompileResult {
                success: false,
                glsl: None,
                error: Some(format!("Parse error: {}", e)),
            };
        }
    };

    // Desugar
    let desugared = match parsed.desugar(&mut frontend.node_counter) {
        Ok(d) => d,
        Err(e) => {
            return CompileResult {
                success: false,
                glsl: None,
                error: Some(format!("Desugar error: {}", e)),
            };
        }
    };

    // Resolve names
    let resolved = match desugared.resolve(&frontend.module_manager) {
        Ok(r) => r,
        Err(e) => {
            return CompileResult {
                success: false,
                glsl: None,
                error: Some(format!("Name resolution error: {}", e)),
            };
        }
    };

    // Fold AST constants
    let ast_folded = resolved.fold_ast_constants();

    // Type check
    let type_checked = match ast_folded.type_check(&frontend.module_manager, &mut frontend.schemes) {
        Ok(t) => t,
        Err(e) => {
            return CompileResult {
                success: false,
                glsl: None,
                error: Some(format!("Type error: {}", e)),
            };
        }
    };

    // Alias check
    let alias_checked = match type_checked.alias_check() {
        Ok(a) => a,
        Err(e) => {
            return CompileResult {
                success: false,
                glsl: None,
                error: Some(format!("Alias check error: {}", e)),
            };
        }
    };

    if alias_checked.has_alias_errors() {
        return CompileResult {
            success: false,
            glsl: None,
            error: Some("Alias checking failed".to_string()),
        };
    }

    // Flatten to MIR
    let (flattened, _backend) = match alias_checked.flatten(&frontend.module_manager, &frontend.schemes) {
        Ok(f) => f,
        Err(e) => {
            return CompileResult {
                success: false,
                glsl: None,
                error: Some(format!("Flatten error: {}", e)),
            };
        }
    };

    // MIR passes
    let hoisted = flattened.hoist_materializations();
    let normalized = hoisted.normalize();
    let monomorphized = match normalized.monomorphize() {
        Ok(m) => m,
        Err(e) => {
            return CompileResult {
                success: false,
                glsl: None,
                error: Some(format!("Monomorphization error: {}", e)),
            };
        }
    };

    // Use partial eval for better optimization
    let folded = match monomorphized.partial_eval() {
        Ok(f) => f,
        Err(e) => {
            return CompileResult {
                success: false,
                glsl: None,
                error: Some(format!("Partial eval error: {}", e)),
            };
        }
    };

    let reachable = folded.filter_reachable();
    let lifted = reachable.lift_bindings();

    // Lower to Shadertoy GLSL
    match lifted.lower_shadertoy() {
        Ok(glsl) => CompileResult {
            success: true,
            glsl: Some(glsl),
            error: None,
        },
        Err(e) => CompileResult {
            success: false,
            glsl: None,
            error: Some(format!("GLSL lowering error: {}", e)),
        },
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
