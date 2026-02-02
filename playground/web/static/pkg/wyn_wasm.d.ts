/* tslint:disable */
/* eslint-disable */

/**
 * Compile Wyn source code to Shadertoy-compatible GLSL.
 *
 * Returns a JSON-serialized CompileResult.
 * Note: init_compiler() should be called first, but this will auto-initialize if needed.
 */
export function compile_to_shadertoy(source: string): any;

/**
 * Compile Wyn source code and return IR trees along with GLSL.
 *
 * Returns a JSON-serialized CompileResultWithIR.
 */
export function compile_with_ir(source: string): any;

/**
 * Get a simple example program to start with
 */
export function get_example_program(): string;

/**
 * Initialize the compiler cache. Call this once at startup.
 * Returns true on success.
 */
export function init_compiler(): boolean;

/**
 * Get the compiler version string
 */
export function version(): string;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly compile_to_shadertoy: (a: number, b: number) => any;
  readonly compile_with_ir: (a: number, b: number) => any;
  readonly get_example_program: () => [number, number];
  readonly init_compiler: () => number;
  readonly version: () => [number, number];
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
  readonly __wbindgen_externrefs: WebAssembly.Table;
  readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
