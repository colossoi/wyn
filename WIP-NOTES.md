# WIP: SIR Parallelization for Compute Shaders

## Overview

Implementing GPU parallelization transforms on SIR that lower high-level SOACs (like `map`) to efficient GPU kernels. The goal is to transform:

```
entry double(arr: []f32) []f32 = map((|x| x * 2.0f32), arr)
```

Into a form where each GPU thread processes one element via slicing.

## Current State

### What's Working

1. **Parallelization transform in `sir/parallelize.rs`**:
   - Detects top-level `map` on unsized storage arrays
   - Transforms to: `global_invocation_id` -> extract x component -> slice input at that index -> map on size-1 slice
   - Uses `__slice_range(arr, idx, idx+1)` to hook into existing slice infrastructure

2. **GlobalInvocationId builtin handling in `flattening.rs`**:
   - Added `require_builtin()` method that creates entry inputs for SPIR-V builtins
   - `_w_global_invocation_id` intrinsic gets flattened to a local load from the builtin input
   - `build_entry_point()` automatically appends discovered builtins to entry inputs

3. **Storage buffer handling in `lowering.rs`**:
   - Storage buffers create RuntimeArray in Block struct for SPIR-V
   - Added storage class tracking to `current_input_vars`
   - Storage buffers are NOT loaded directly (skipped in load loop) - they're accessed via slices

### Current Problem

When flattening `__slice_range(arr, idx, end_idx)` where `arr` is a storage buffer:

- The existing code creates `ArrayBacking::View { base, offset }` which expects `base` to be a loaded value
- But storage buffers can't be loaded as a whole (they're RuntimeArrays)
- Storage buffers need `ArrayBacking::Storage { name, offset }` instead, which references the buffer by name

**The fix needed** in `flattening.rs` `__slice_range` handling:
1. Check if `base_ty` is a storage buffer array (has `AddressStorage` in type)
2. If storage buffer: use `ArrayBacking::Storage { name: base_name, offset: offset_id }`
3. If regular array: use `ArrayBacking::View { base: base_id, offset: offset_id }` (existing behavior)

The type information is already available on the local (`base_decl.ty`), so no additional tracking is needed.

## Files Modified

- `wyn-core/src/sir/parallelize.rs` - Transform logic
- `wyn-core/src/flattening.rs` - `require_builtin()`, `build_entry_point()`, `_w_global_invocation_id` handling
- `wyn-core/src/spirv/lowering.rs` - Storage class tracking, skip loading storage buffers
- `testfiles/double.wyn` - Test compute shader

## Test Command

```bash
cargo run --bin wyn -- compile testfiles/double.wyn -o /tmp/double.spv
spirv-val /tmp/double.spv
```

## Next Steps

1. Modify `__slice_range` in flattening.rs to detect storage buffers and use `ArrayBacking::Storage`
2. Verify the lowering of `ArrayBacking::Storage` slices works correctly
3. Handle output writing (the map result needs to be stored back to output buffer at the correct index)
