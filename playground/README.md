# Wyn Playground

A web-based shader editor for Wyn, similar to Shadertoy. Write Wyn code, compile it to GLSL, and see your shaders run in real-time.

## Prerequisites

- [Deno](https://deno.land/) (for the web server)
- [wasm-pack](https://rustwasm.github.io/wasm-pack/) (for building the WASM compiler)
- Rust toolchain with `wasm32-unknown-unknown` target

## Setup

1. Build the WASM compiler module:

```bash
cd wyn-wasm
wasm-pack build --target web --out-dir ../web/static/pkg
```

2. Start the development server:

```bash
cd web
deno task dev
```

3. Open http://localhost:8080 in your browser

## Project Structure

```
playground/
├── wyn-wasm/           # Rust crate that compiles to WASM
│   ├── Cargo.toml
│   └── src/
│       └── lib.rs      # WASM bindings for wyn-core
├── web/                # Deno web server + frontend
│   ├── deno.json
│   ├── server.ts       # Simple static file server
│   └── static/
│       ├── index.html  # Main UI
│       ├── app.js      # JavaScript application
│       └── pkg/        # Built WASM files (generated)
└── README.md
```

## Usage

- Write your Wyn shader code in the editor
- Press "Compile & Run" or Ctrl+Enter to compile and display
- The generated GLSL is shown in the output panel
- Supported Shadertoy uniforms: `iResolution`, `iTime`, `iMouse`

## Example Shader

```wyn
#[uniform(set=0, binding=0)] def iResolution: vec2f32
#[uniform(set=0, binding=1)] def iTime: f32

def verts: [3]vec4f32 =
  [@[-1.0, -1.0, 0.0, 1.0],
   @[3.0, -1.0, 0.0, 1.0],
   @[-1.0, 3.0, 0.0, 1.0]]

#[vertex]
def vertex_main(#[builtin(vertex_index)] vertex_id: i32) -> #[builtin(position)] vec4f32 =
  verts[vertex_id]

#[fragment]
def fragment_main(#[builtin(position)] fragCoord: vec4f32) -> #[location(0)] vec4f32 =
  let uv = @[fragCoord.x / iResolution.x, fragCoord.y / iResolution.y] in
  let r = 0.5 + 0.5 * f32.cos(iTime + uv.x * 3.0) in
  let g = 0.5 + 0.5 * f32.cos(iTime + uv.y * 3.0 + 2.0) in
  let b = 0.5 + 0.5 * f32.cos(iTime + (uv.x + uv.y) * 1.5 + 4.0) in
  @[r, g, b, 1.0]
```

## Rebuilding After Changes

If you modify `wyn-core`, rebuild the WASM module:

```bash
cd wyn-wasm
wasm-pack build --target web --out-dir ../web/static/pkg
```

Then refresh the browser to load the updated compiler.
