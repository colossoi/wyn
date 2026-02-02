# Storage View Refactor (WIP)

## Problem

SPIR-V's logical addressing model doesn't support pointer arithmetic on storage buffer pointers in the way we were using it. `OpPtrAccessChain` requires the base pointer to point into an array with `ArrayStride` decoration. Our original "fat pointer" representation `{ptr, len}` for Views was not tenable because:

1. We can't do `ptr + offset` on an arbitrary element pointer
2. The pointer must be to the buffer itself, not an offset element

## Solution

Change Views from fat pointers to "buffer slice descriptors":

**Old representation:** `{ptr, len}` where ptr could be offset
**New representation:** `{buffer_ptr, offset, len}` where buffer_ptr is always the storage buffer variable

## MIR Changes

Replaced old expressions:
- `View { ptr, len }`
- `ViewPtr { view }`
- `ViewLen { view }`
- `PtrAdd { ptr, offset }`

With new expressions:
- `StorageView { set: u32, binding: u32, offset: ExprId, len: ExprId }` - create view with known set/binding
- `SliceStorageView { view: ExprId, start: ExprId, len: ExprId }` - slice existing view
- `StorageViewIndex { view: ExprId, index: ExprId }` - get pointer to element
- `StorageViewLen { view: ExprId }` - get length

## SPIR-V Lowering Strategy

At SPIR-V level, the view is a struct `{buffer_ptr, offset, len}`:
- `buffer_ptr`: Pointer to the storage buffer variable (looked up from set/binding at `StorageView` creation)
- `offset`: u32 offset into the buffer
- `len`: u32 length of the view

For `StorageViewIndex { view, index }`:
```
buffer_ptr = view[0]
offset = view[1]
final_index = offset + index
result = OpAccessChain(buffer_ptr, [0, final_index])
```

## Files Modified

### Complete:
- `mir/mod.rs` - New expression variants
- `materialize_hoisting.rs` - Expression traversal updated
- `dps_transform.rs` - Expression copying updated
- `inplace_rewriter.rs` - Expression rewriting updated
- `soac_parallelize.rs` - Changed view/chunk creation to use new expressions
- `glsl/lowering.rs` - Error message updated

### In Progress:
- `spirv/lowering.rs` - Lowering logic needs fixing:
  - Currently stores set/binding as runtime integers (wrong)
  - Should store buffer_ptr directly, looked up at StorageView creation time
  - `find_storage_buffer_by_binding` method doesn't exist

### Not Yet Updated:
- `tlc/to_mir.rs` - Slice implementation still uses old ViewPtr/PtrAdd/View pattern

## TODO

1. Fix `spirv/lowering.rs`:
   - `StorageView`: Look up buffer from `storage_buffers` HashMap using set/binding, store buffer_ptr in struct
   - `SliceStorageView`: Extract buffer_ptr from existing view, adjust offset
   - `StorageViewIndex`: Extract buffer_ptr and offset, do `OpAccessChain(buffer_ptr, [0, offset + index])`
   - `StorageViewLen`: Extract len field

2. Update `tlc/to_mir.rs`:
   - Change slice intrinsic handling to emit `SliceStorageView` instead of ViewPtr/PtrAdd/View

3. Run `cargo build` and fix remaining compilation errors

4. Run `cargo test` and fix test failures

5. Test with actual .wyn files (miner.wyn, etc.)

## Design Notes

- `set` and `binding` in `StorageView` are compile-time constants (u32), used to look up the buffer variable
- Once lowered to SPIR-V, the view struct contains the actual buffer pointer, not set/binding
- `SliceStorageView` preserves the buffer_ptr, only adjusts offset
- `OpArrayLength` returns u32, so view lengths are u32
