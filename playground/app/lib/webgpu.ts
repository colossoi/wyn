// WebGPU render + compute pipeline driver for WGSL previews.
// Pure functions + a minimal RAF-loop builder; no React in here.
//
// Flow:
//   setupContext → createPipelines → startRenderLoop → stop()
//
// Shadertoy-style uniforms (iResolution: vec3<f32>, iTime: f32,
// iMouse: vec4<f32>) get auto-populated per frame; other uniforms are
// left untouched (the backing buffer is zero-initialized).
//
// Storage buffers (both user-declared and compiler-introduced by the
// parallelize pass for lifted pre-pass partials/results) are allocated
// at a generous default size. Every compute entry becomes a compute
// pipeline dispatched once per frame (single workgroup) before the
// render pass — the parallelize pass emits 64-thread workgroups sized
// to cover the whole problem in one dispatch.

import type { ProgramInterface, ResourceBinding } from "./wasm";

export interface WebGPUContext {
  device: GPUDevice;
  canvasContext: GPUCanvasContext;
  format: GPUTextureFormat;
}

/** Initialize a WebGPU adapter+device and configure the canvas context. */
export async function setupContext(
  canvas: HTMLCanvasElement,
): Promise<WebGPUContext> {
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

/** Default allocation for every runtime-sized `array<T>` storage binding.
 *  64 KiB handles parallelize's 64-element partials buffers and the
 *  single-scalar result slot with plenty of headroom. */
const STORAGE_BUFFER_BYTES = 64 * 1024;

/** Bytes a scalar/vector/matrix WGSL type occupies. Returns `null` for
 *  types we don't recognize — those bindings get a 16-byte fallback. */
function uniformSizeBytes(ty: string): number {
  const t = ty.replace(/\s+/g, "");
  if (t === "f32" || t === "i32" || t === "u32") return 4;
  if (t === "vec2<f32>" || t === "vec2f") return 8;
  if (t === "vec3<f32>" || t === "vec3f") return 12;
  if (t === "vec4<f32>" || t === "vec4f") return 16;
  if (t === "vec2<i32>" || t === "vec2<u32>") return 8;
  if (t === "vec3<i32>" || t === "vec3<u32>") return 12;
  if (t === "vec4<i32>" || t === "vec4<u32>") return 16;
  return 16;
}

/** Round up to a multiple of 16 (WebGPU uniform-buffer binding-size
 *  minimum; also avoids alignment surprises on array-of-vec types). */
function pad16(n: number): number {
  return Math.max(16, Math.ceil(n / 16) * 16);
}

export interface ComputeDispatch {
  pipeline: GPUComputePipeline;
  workgroups: [number, number, number];
}

export interface RenderResources {
  /** Render pipeline for the vertex+fragment pair, if the program has
   *  both. Absent for compute-only programs. */
  pipeline: GPURenderPipeline | null;
  /** Every compute entry in dispatch order (interface order), dispatched
   *  once per frame before the render pass. */
  computeDispatches: ComputeDispatch[];
  /** Bind groups for the render pipeline, keyed by `set`. Storage
   *  bindings here are `read-only-storage`; WebGPU disallows writable
   *  storage on vertex/fragment stages. */
  renderBindGroups: Map<number, GPUBindGroup>;
  /** Bind groups for compute pipelines, keyed by `set`. Storage
   *  bindings here are read-write; visibility is COMPUTE-only. */
  computeBindGroups: Map<number, GPUBindGroup>;
  /** Uniform backing buffers keyed by `"set:binding"`. */
  uniformBuffers: Map<string, GPUBuffer>;
  /** Storage backing buffers keyed by `"set:binding"`. */
  storageBuffers: Map<string, GPUBuffer>;
  /** Shortcut handles for auto-populated uniforms. */
  shadertoy: {
    iResolution?: GPUBuffer;
    iTime?: GPUBuffer;
    iMouse?: GPUBuffer;
  };
}

function bindingKey(set: number, binding: number): string {
  return `${set}:${binding}`;
}

/** Build render + compute pipelines and the shared bind groups from a
 *  compiled WGSL module. Every compute entry becomes a `ComputeDispatch`
 *  issued once per frame before the render pass; if the program is
 *  compute-only (no vertex+fragment pair), the render pipeline is null
 *  and only the compute passes run. */
export function createPipelines(
  ctx: WebGPUContext,
  wgsl: string,
  iface: ProgramInterface,
): RenderResources {
  const { device, format } = ctx;
  const module = device.createShaderModule({ code: wgsl });

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

  const storageBuffers = new Map<string, GPUBuffer>();
  for (const s of iface.storage) {
    const buf = device.createBuffer({
      label: `storage ${s.name} @(${s.set},${s.binding})`,
      size: STORAGE_BUFFER_BYTES,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    });
    storageBuffers.set(bindingKey(s.set, s.binding), buf);
  }

  // Render and compute pipelines use separate bind group layouts over
  // the same buffers because the layout type must match the shader's
  // declared storage access (the WGSL emitter declares a slot
  // `read_write` when any stage writes it, so `storage` is required
  // everywhere), and WebGPU additionally forbids `storage` (read-write)
  // on VERTEX visibility. Storage slots therefore appear in the render
  // layout at FRAGMENT-only visibility; uniforms span VERTEX|FRAGMENT.
  const setsInUse = new Set<number>();
  for (const u of iface.uniforms) setsInUse.add(u.set);
  for (const s of iface.storage) setsInUse.add(s.set);
  const sortedSets = Array.from(setsInUse).sort((a, b) => a - b);

  // Storage bindings referenced by any vertex/fragment entry — these
  // must appear in the render layout as read-only-storage.
  const renderStorageSlots = new Set<string>();
  for (const e of iface.entries) {
    if (e.kind !== "vertex" && e.kind !== "fragment") continue;
    for (const b of e.inputs) {
      const m = /^storage\((\d+),(\d+)\)$/.exec(b.decoration);
      if (m) renderStorageSlots.add(bindingKey(Number(m[1]), Number(m[2])));
    }
    for (const b of e.outputs) {
      const m = /^storage\((\d+),(\d+)\)$/.exec(b.decoration);
      if (m) renderStorageSlots.add(bindingKey(Number(m[1]), Number(m[2])));
    }
  }

  const uniformVis = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT;
  const renderStorageVis = GPUShaderStage.FRAGMENT;
  const computeVis = GPUShaderStage.COMPUTE;

  const renderLayouts: GPUBindGroupLayout[] = sortedSets.map((set) => {
    const entries: GPUBindGroupLayoutEntry[] = [];
    for (const u of iface.uniforms) {
      if (u.set === set) {
        entries.push({
          binding: u.binding,
          visibility: uniformVis,
          buffer: { type: "uniform" },
        });
      }
    }
    for (const s of iface.storage) {
      if (
        s.set === set &&
        renderStorageSlots.has(bindingKey(s.set, s.binding))
      ) {
        const type: GPUBufferBindingType =
          s.access === "read" ? "read-only-storage" : "storage";
        entries.push({
          binding: s.binding,
          visibility: renderStorageVis,
          buffer: { type },
        });
      }
    }
    entries.sort((a, b) => a.binding - b.binding);
    return device.createBindGroupLayout({ entries });
  });

  const computeLayouts: GPUBindGroupLayout[] = sortedSets.map((set) => {
    const entries: GPUBindGroupLayoutEntry[] = [];
    for (const u of iface.uniforms) {
      if (u.set === set) {
        entries.push({
          binding: u.binding,
          visibility: computeVis,
          buffer: { type: "uniform" },
        });
      }
    }
    for (const s of iface.storage) {
      if (s.set === set) {
        entries.push({
          binding: s.binding,
          visibility: computeVis,
          buffer: { type: "storage" },
        });
      }
    }
    entries.sort((a, b) => a.binding - b.binding);
    return device.createBindGroupLayout({ entries });
  });

  const renderPipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: renderLayouts,
  });
  const computePipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: computeLayouts,
  });

  const makeBindGroups = (
    layouts: GPUBindGroupLayout[],
    includeStorage: (set: number, binding: number) => boolean,
  ): Map<number, GPUBindGroup> => {
    const out = new Map<number, GPUBindGroup>();
    for (let i = 0; i < sortedSets.length; i++) {
      const set = sortedSets[i];
      const entries: GPUBindGroupEntry[] = [];
      for (const u of iface.uniforms) {
        if (u.set === set) {
          entries.push({
            binding: u.binding,
            resource: {
              buffer: uniformBuffers.get(bindingKey(u.set, u.binding))!,
            },
          });
        }
      }
      for (const s of iface.storage) {
        if (s.set === set && includeStorage(s.set, s.binding)) {
          entries.push({
            binding: s.binding,
            resource: {
              buffer: storageBuffers.get(bindingKey(s.set, s.binding))!,
            },
          });
        }
      }
      entries.sort((a, b) => a.binding - b.binding);
      out.set(set, device.createBindGroup({ layout: layouts[i], entries }));
    }
    return out;
  };

  const renderBindGroups = makeBindGroups(renderLayouts, (set, binding) =>
    renderStorageSlots.has(bindingKey(set, binding)),
  );
  const computeBindGroups = makeBindGroups(computeLayouts, () => true);

  const vertexEntry = iface.entries.find((e) => e.kind === "vertex");
  const fragmentEntry = iface.entries.find((e) => e.kind === "fragment");
  let pipeline: GPURenderPipeline | null = null;
  if (vertexEntry && fragmentEntry) {
    pipeline = device.createRenderPipeline({
      layout: renderPipelineLayout,
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
  }

  const computeDispatches: ComputeDispatch[] = iface.entries
    .filter((e) => e.kind === "compute")
    .map((e) => ({
      pipeline: device.createComputePipeline({
        layout: computePipelineLayout,
        compute: {
          module,
          entryPoint: e.wgsl_name,
        },
      }),
      // parallelize emits 64-thread workgroups sized to cover the whole
      // problem in one dispatch; keep the prototype simple by always
      // dispatching a single workgroup. User-authored compute entries
      // that need multiple workgroups aren't handled yet.
      workgroups: [1, 1, 1],
    }));

  return {
    pipeline,
    computeDispatches,
    renderBindGroups,
    computeBindGroups,
    uniformBuffers,
    storageBuffers,
    shadertoy,
  };
}

/** Back-compat alias — older call sites import `createRenderPipeline`. */
export const createRenderPipeline = createPipelines;

// -----------------------------------------------------------------------------
// Render loop
// -----------------------------------------------------------------------------

export interface RenderLoop {
  stop: () => void;
}

/** Drive pipelines once per RAF tick. Each frame:
 *   - update known Shadertoy uniforms (iResolution, iTime, iMouse)
 *   - dispatch every compute entry (single workgroup each)
 *   - encode a render pass drawing a 3-vertex full-screen triangle, if
 *     the program has vertex+fragment entries
 *   - submit and loop. */
export function startRenderLoop(
  ctx: WebGPUContext,
  canvas: HTMLCanvasElement,
  res: RenderResources,
  onFps: (fps: number) => void,
  onElapsed?: (seconds: number) => void,
  isPaused?: () => boolean,
): RenderLoop {
  const { device, canvasContext } = ctx;
  let animationId: number | null = null;
  let startTime = performance.now();
  let pausedAt: number | null = null;
  let frameCount = 0;
  let lastFpsUpdate = startTime;

  const resolutionBytes = new Float32Array(3); // vec3<f32>
  const timeBytes = new Float32Array(1);
  const mouseBytes = new Float32Array(4);

  function tick(time: number) {
    // Freeze elapsed time while paused; on resume, shift startTime so
    // iTime keeps ticking seamlessly from where it was.
    if (isPaused?.()) {
      if (pausedAt === null) pausedAt = time;
      animationId = requestAnimationFrame(tick);
      return;
    }
    if (pausedAt !== null) {
      startTime += time - pausedAt;
      pausedAt = null;
    }
    const currentTime = (time - startTime) / 1000;
    onElapsed?.(currentTime);

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

    // One pass per dispatch so consecutive phases get a memory barrier
    // — WebGPU guarantees ordering between passes in the same encoder,
    // but dispatches within a single pass have no implicit barrier.
    for (const cd of res.computeDispatches) {
      const cpass = encoder.beginComputePass();
      cpass.setPipeline(cd.pipeline);
      for (const [set, group] of res.computeBindGroups) {
        cpass.setBindGroup(set, group);
      }
      cpass.dispatchWorkgroups(
        cd.workgroups[0],
        cd.workgroups[1],
        cd.workgroups[2],
      );
      cpass.end();
    }

    if (res.pipeline) {
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
      for (const [set, group] of res.renderBindGroups) {
        pass.setBindGroup(set, group);
      }
      pass.draw(3);
      pass.end();
    }

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
