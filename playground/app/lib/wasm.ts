// WASM bindings for the wyn-wasm crate.
//
// The wasm-pack output lives in `app/lib/wasm-pkg/` (built via
// `npm run build:wasm`). Vite handles the `.js` glue and bundles the
// `.wasm` binary as an asset.
//
// Initialization is cached: the first caller drives `wasm-bindgen init` and
// `init_compiler()`; subsequent callers await the same promise.

export interface ErrorLocation {
  start_line: number;
  start_col: number;
  end_line: number;
  end_col: number;
}

export interface ErrorInfo {
  message: string;
  location: ErrorLocation | null;
}

export interface IRTreeNode {
  name: string;
  children?: IRTreeNode[];
}

export interface CompileResult {
  success: boolean;
  glsl?: string;
  tlc?: IRTreeNode[];
  mir?: string;
  error?: ErrorInfo;
}

// -----------------------------------------------------------------------------
// WGSL compilation output — includes a structured program interface the JS
// side uses to drive WebGPU buffer/binding setup and the pipeline-viz panel.
// -----------------------------------------------------------------------------

export interface EntryBinding {
  name: string;
  ty: string;
  /** "builtin(<name>)" / "location(<n>)" / "storage(<set>,<binding>)" / "push_constant(<off>)" */
  decoration: string;
}

export interface EntryInterface {
  name: string;
  /** WGSL-mangled entry-point name — pass this as `entryPoint:` to
   *  WebGPU's pipeline creation. */
  wgsl_name: string;
  /** "vertex" | "fragment" | "compute" */
  kind: string;
  /** `[x, y, z]` for compute entries; undefined otherwise. */
  workgroup_size?: [number, number, number];
  inputs: EntryBinding[];
  outputs: EntryBinding[];
}

export interface ResourceBinding {
  name: string;
  set: number;
  binding: number;
  ty: string;
  /** "read" / "write" / "read_write" for storage; empty for uniforms. */
  access?: string;
}

export interface ProgramInterface {
  entries: EntryInterface[];
  uniforms: ResourceBinding[];
  storage: ResourceBinding[];
}

export interface CompileResultWgsl {
  success: boolean;
  wgsl?: string;
  interface?: ProgramInterface;
  mir?: string;
  tlc?: IRTreeNode[];
  error?: ErrorInfo;
}

export interface WynWasm {
  version: () => string;
  compile_with_ir: (source: string) => CompileResult;
  compile_to_shadertoy: (source: string) => CompileResult;
  compile_to_wgsl: (source: string) => CompileResultWgsl;
  get_example_program: () => string;
}

let cached: Promise<WynWasm> | null = null;

export function initWasm(): Promise<WynWasm> {
  if (typeof window === "undefined") {
    return Promise.reject(new Error("initWasm called on the server"));
  }
  if (!cached) {
    cached = (async () => {
      // @ts-ignore - generated module, present after `npm run build:wasm`
      const mod = await import("./wasm-pkg/wyn_wasm.js");
      await mod.default();
      if (!mod.init_compiler()) {
        throw new Error("Failed to initialize compiler");
      }
      return {
        version: mod.version,
        compile_with_ir: mod.compile_with_ir,
        compile_to_shadertoy: mod.compile_to_shadertoy,
        compile_to_wgsl: mod.compile_to_wgsl,
        get_example_program: mod.get_example_program,
      };
    })();
  }
  return cached;
}
