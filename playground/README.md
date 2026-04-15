# Wyn Playground

Browser-based playground for the Wyn shader language: write Wyn, compile to GLSL in-browser via WASM, see the shader run on a WebGL canvas in real time. Built on React Router v7 (SSR) deployed to Cloudflare Pages via a Worker.

## Prerequisites

- Node.js 20+
- [wasm-pack](https://rustwasm.github.io/wasm-pack/) and a Rust toolchain with `wasm32-unknown-unknown`
- Optional, for deploys: a Cloudflare account + `wrangler` (installed as a devDep)

## Setup

```bash
cd playground
npm install
npm run build:wasm        # produces public/pkg/* via wasm-pack
```

## Development

```bash
npm run dev               # Vite dev server with HMR — http://localhost:5173
```

## Production build + local preview

```bash
npm run build             # rebuilds wasm + react-router build → build/{client,server}/
npm run preview           # wrangler dev — serves the Worker against the built bundle
```

## Deploy to Cloudflare Pages

**CLI**:
```bash
npm run deploy            # npm run build && wrangler deploy
```
First run prompts for Cloudflare auth.

**Git integration**: point the Pages project at this repo with build command `cd playground && npm install && npm run build` and output directory `playground/dist/client`. The Worker entry is picked up from `wrangler.jsonc` automatically.

## Project structure

```
playground/
├── app/                  # React Router app
│   ├── root.tsx          # html shell + CodeMirror CDN script
│   ├── routes/home.tsx   # playground orchestrator (state + flow)
│   ├── components/       # Editor, Preview, IRTree, StatusBar
│   ├── lib/wasm.ts       # WASM init + typed bindings
│   ├── lib/webgl.ts      # shader compile + RAF loop
│   └── app.css           # global stylesheet
├── workers/app.ts        # Cloudflare Worker entry — runs the SSR handler
├── public/
│   ├── pkg/              # wasm-pack output (gitignored, rebuilt by build:wasm)
│   └── _headers          # Cloudflare COOP/COEP headers
├── wyn-wasm/             # Rust crate compiled to WASM
├── react-router.config.ts
├── vite.config.ts
├── wrangler.jsonc
└── package.json
```

## Rebuilding after changes to `wyn-core`

```bash
npm run build:wasm        # picks up the latest wyn-core via the cargo workspace
```

The dev server hot-reloads the JS/TSX side automatically; only the WASM rebuild needs an explicit step.

## Example shader

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

Supported Shadertoy uniforms: `iResolution`, `iTime`, `iMouse`.
