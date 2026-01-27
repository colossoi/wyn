# Linked SPIR-V Implementation

This document describes the work-in-progress support for linking external SPIR-V modules into Wyn-compiled shaders.

## Motivation

Some functions (like SHA256 compression) are complex enough that generating them inline during compilation is expensive and produces large output. SPIR-V supports a linkage capability that allows:

1. Compiling a function separately (via `spirv-as` from hand-written assembly or another compiler)
2. Marking it with `Export` linkage
3. Importing it from another module with `Import` linkage
4. Linking the modules together with `spirv-link`

This allows us to:
- Write performance-critical functions in hand-optimized SPIR-V assembly
- Reuse the same compiled function across multiple shaders
- Reduce compilation time by not regenerating complex code

## Implementation

### New `BuiltinImpl` Variant

Added `LinkedSpirv(String)` to `BuiltinImpl` enum in `impl_source.rs`:

```rust
pub enum BuiltinImpl {
    PrimOp(PrimOp),
    Intrinsic(Intrinsic),
    LinkedSpirv(String),  // linkage name for import
}
```

### SPIR-V Lowering

When the compiler encounters a call to a `LinkedSpirv` function:

1. Enables `OpCapability Linkage` (once per module)
2. Declares an empty function with the correct signature
3. Decorates it with `OpDecorate %func LinkageAttributes "name" Import`
4. Generates `OpFunctionCall` to invoke it

The resulting SPIR-V is valid but incomplete - it must be linked with the external module before use.

### External SPIR-V Modules

External functions live in `lib/` as `.spvasm` files:

- `lib/sha256_compress.spvasm` - SHA256 compression function

These are assembled with `spirv-as` and linked with `spirv-link`:

```bash
# Assemble the library
spirv-as lib/sha256_compress.spvasm -o lib/sha256_compress.spv

# Compile Wyn code (produces incomplete SPIR-V with imports)
cargo run --bin wyn -- compile myshader.wyn -o myshader_unlinked.spv

# Link them together
spirv-link myshader_unlinked.spv lib/sha256_compress.spv -o myshader.spv
```

## Current Status (WIP)

- [x] Add `LinkedSpirv` variant to `BuiltinImpl`
- [x] Implement import declaration generation in SPIR-V lowering
- [x] Write SHA256 compression in SPIR-V assembly (fully unrolled, 64 rounds)
- [x] Add GLSL lowering error for unsupported linked functions
- [ ] Integrate linking into the compilation pipeline
- [ ] Add CLI flag to specify library modules to link
- [ ] Test end-to-end with actual GPU execution

## Files

- `wyn-core/src/impl_source.rs` - `LinkedSpirv` variant and registration
- `wyn-core/src/spirv/lowering.rs` - Import declaration generation
- `wyn-core/src/glsl/lowering.rs` - Error handling for GLSL target
- `lib/sha256_compress.spvasm` - Hand-written SHA256 compression
- `scripts/gen_sha256_spvasm.py` - Generator script for the SPIR-V assembly
