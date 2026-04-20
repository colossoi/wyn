// WebGPU render pipeline for WGSL previews.
// Pure functions + a minimal RAF-loop builder; no React in here.
//
// The flow mirrors `webgl.ts`:
//   setupContext → createRenderPipeline → startRenderLoop → stop()
//
// Shadertoy-style uniforms (iResolution: vec3<f32>, iTime: f32,
// iMouse: vec4<f32>) get auto-populated per frame; other uniforms are
// left untouched (the backing buffer is zero-initialized).

import type { ProgramInterface, ResourceBinding } from "./wasm";

export interface WebGPUContext {
  device: GPUDevice;
  canvasContext: GPUCanvasContext;
  format: GPUTextureFormat;
}

/** Initialize a WebGPU adapter+device and configure the canvas context. */
export async function setupContext(canvas: HTMLCanvasElement): Promise<WebGPUContext> {
  if (!navigator.gpu) {
    throw new Error("WebGPU not supported in this browser");
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("No GPU adapter available");
  }
  const device = await adapter.requestDevice();
  const canvasContext = canvas.getContext("webgpu");
  if (!canvasContext) {
    throw new Error("canvas.getContext('webgpu') returned null");
  }
  const format = navigator.gpu.getPreferredCanvasFormat();
  canvasContext.configure({
    device,
    format,
    alphaMode: "opaque",
  });
  return { device, canvasContext, format };
}

// -----------------------------------------------------------------------------
// Pipeline + bind-group construction
// -----------------------------------------------------------------------------

/** Bytes a scalar/vector/matrix WGSL type occupies, for laying out uniform
 * buffers. Returns `null` for types we don't recognize — those bindings get
 * a zero-filled buffer sized by the declared type string is impossible, so
 * we allocate a minimum-size fallback. */
function uniformSizeBytes(ty: string): number {
  // Strip spaces and be defensive about variations like "vec3f" vs "vec3<f32>".
  const t = ty.replace(/\s+/g, "");
  if (t === "f32" || t === "i32" || t === "u32") return 4;
  if (t === "vec2<f32>" || t === "vec2f") return 8;
  if (t === "vec3<f32>" || t === "vec3f") return 12;
  if (t === "vec4<f32>" || t === "vec4f") return 16;
  if (t === "vec2<i32>" || t === "vec2<u32>") return 8;
  if (t === "vec3<i32>" || t === "vec3<u32>") return 12;
  if (t === "vec4<i32>" || t === "vec4<u32>") return 16;
  // Fallback — WebGPU requires ≥ 4 bytes per uniform buffer.
  return 16;
}

/** Round up to a multiple of 16 (WebGPU uniform-buffer binding-size
 *  minimum; also avoids alignment surprises on array-of-vec types). */
function pad16(n: number): number {
  return Math.max(16, Math.ceil(n / 16) * 16);
}

export interface RenderResources {
  pipeline: GPURenderPipeline;
  /** Map from `(set, binding)` key → GPUBuffer for every uniform binding. */
  uniformBuffers: Map<string, GPUBuffer>;
  /** Bind groups keyed by descriptor set. */
  bindGroups: Map<number, GPUBindGroup>;
  /** Subset of uniforms we recognize and update per-frame. */
  shadertoy: {
    iResolution?: GPUBuffer;
    iTime?: GPUBuffer;
    iMouse?: GPUBuffer;
  };
}

function bindingKey(set: number, binding: number): string {
  return `${set}:${binding}`;
}

/** Build a render pipeline + bind groups from a compiled WGSL module.
 * Requires the program to have exactly one @vertex and one @fragment
 * entry; any compute entries are ignored for rendering (the pipeline-viz
 * panel still shows them). */
export function createRenderPipeline(
  ctx: WebGPUContext,
  wgsl: string,
  iface: ProgramInterface,
): RenderResources {
  const { device, format } = ctx;

  const module = device.createShaderModule({ code: wgsl });

  const vertexEntry = iface.entries.find((e) => e.kind === "vertex");
  const fragmentEntry = iface.entries.find((e) => e.kind === "fragment");
  if (!vertexEntry || !fragmentEntry) {
    throw new Error("WGSL render preview requires both a @vertex and a @fragment entry");
  }

  // Allocate a uniform buffer per `@group(G) @binding(B)` declaration.
  // The Wyn WGSL backend emits one binding per uniform, so there's a
  // 1:1 mapping between `iface.uniforms` and WGSL bindings.
  const uniformBuffers = new Map<string, GPUBuffer>();
  const shadertoy: RenderResources["shadertoy"] = {};
  for (const u of iface.uniforms) {
    const size = pad16(uniformSizeBytes(u.ty));
    const buf = device.createBuffer({
      label: `uniform ${u.name}`,
      size,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    uniformBuffers.set(bindingKey(u.set, u.binding), buf);
    if (u.name === "iResolution") shadertoy.iResolution = buf;
    else if (u.name === "iTime") shadertoy.iTime = buf;
    else if (u.name === "iMouse") shadertoy.iMouse = buf;
  }

  // Bind groups: one per distinct `set`, containing every uniform whose
  // `set` matches. We let the shader-module reflection drive the layout
  // (pipeline.getBindGroupLayout) so the descriptor matches whatever
  // naga inferred from the WGSL source.
  const pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module,
      entryPoint: vertexEntry.wgsl_name,
    },
    fragment: {
      module,
      entryPoint: fragmentEntry.wgsl_name,
      targets: [{ format }],
    },
    primitive: { topology: "triangle-list" },
  });

  const bindGroups = new Map<number, GPUBindGroup>();
  const setsInUse = new Set<number>();
  for (const u of iface.uniforms) setsInUse.add(u.set);
  for (const set of setsInUse) {
    const entries = iface.uniforms
      .filter((u) => u.set === set)
      .map<GPUBindGroupEntry>((u) => ({
        binding: u.binding,
        resource: { buffer: uniformBuffers.get(bindingKey(u.set, u.binding))! },
      }));
    bindGroups.set(
      set,
      device.createBindGroup({
        layout: pipeline.getBindGroupLayout(set),
        entries,
      }),
    );
  }

  return { pipeline, uniformBuffers, bindGroups, shadertoy };
}

// -----------------------------------------------------------------------------
// Render loop
// -----------------------------------------------------------------------------

export interface RenderLoop {
  stop: () => void;
}

/** Drive the render pipeline once per RAF tick. Each frame:
 *   - update known Shadertoy uniforms (iResolution, iTime, iMouse)
 *   - encode a render pass drawing a 3-vertex full-screen triangle
 *     (we always draw 3 vertices; Wyn vertex entries typically use
 *     `vertex_index` to pick positions from a constant array)
 *   - submit and loop. */
export function startRenderLoop(
  ctx: WebGPUContext,
  canvas: HTMLCanvasElement,
  res: RenderResources,
  onFps: (fps: number) => void,
): RenderLoop {
  const { device, canvasContext } = ctx;
  let animationId: number | null = null;
  const startTime = performance.now();
  let frameCount = 0;
  let lastFpsUpdate = startTime;

  const resolutionBytes = new Float32Array(3); // vec3<f32>
  const timeBytes = new Float32Array(1);
  const mouseBytes = new Float32Array(4);

  function tick(time: number) {
    const currentTime = (time - startTime) / 1000;

    // Update recognized uniforms.
    if (res.shadertoy.iResolution) {
      resolutionBytes[0] = canvas.width;
      resolutionBytes[1] = canvas.height;
      resolutionBytes[2] = 1.0;
      device.queue.writeBuffer(res.shadertoy.iResolution, 0, resolutionBytes);
    }
    if (res.shadertoy.iTime) {
      timeBytes[0] = currentTime;
      device.queue.writeBuffer(res.shadertoy.iTime, 0, timeBytes);
    }
    if (res.shadertoy.iMouse) {
      device.queue.writeBuffer(res.shadertoy.iMouse, 0, mouseBytes);
    }

    const encoder = device.createCommandEncoder();
    const view = canvasContext.getCurrentTexture().createView();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view,
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    pass.setPipeline(res.pipeline);
    for (const [set, group] of res.bindGroups) {
      pass.setBindGroup(set, group);
    }
    pass.draw(3);
    pass.end();
    device.queue.submit([encoder.finish()]);

    frameCount++;
    if (time - lastFpsUpdate > 1000) {
      onFps(Math.round((frameCount * 1000) / (time - lastFpsUpdate)));
      frameCount = 0;
      lastFpsUpdate = time;
    }
    animationId = requestAnimationFrame(tick);
  }

  animationId = requestAnimationFrame(tick);
  return {
    stop() {
      if (animationId !== null) {
        cancelAnimationFrame(animationId);
        animationId = null;
      }
    },
  };
}

// Suppress unused warning for ResourceBinding export.
export type _unused = ResourceBinding;
