# Array Type Unification - Work in Progress

## Goal
Unify `ValueArray` and `Slice` into a single `Array[elem, addrspace, size]` type constructor to properly support multi-buffer compute shaders where lambdas capture storage buffers.

## The Problem
Previously, `[]f32` in entry parameters was ambiguous - it could mean a value array or a storage slice. SPIR-V requires different representations:
- Value arrays: inline data with known size
- Storage slices: pointer + length struct for runtime-sized buffers

## New Type Format
```
Array[elem_type, address_space, size]
```

Where:
- `elem_type`: The element type (e.g., `f32`)
- `address_space`: `Storage`, `Function`, or `AddressUnknown` (marker)
- `size`: `Size(n)`, `SizeVar("n")`, or `Unsized` (marker)

## Implementation Status

### Completed
1. **TypeName changes**: Added `AddressUnknown`, `Function` variants; removed `Slice`, `RuntimeArray`
2. **Parser updates**: `[]f32` emits `Array[f32, AddressUnknown, Unsized]`; `[5]f32` emits `Array[f32, AddressUnknown, Size(5)]`
3. **Type checker pre-pass**: Converts `AddressUnknown` and `Unsized` markers to fresh type variables before unification
4. **Entry point handling**: Constrains array params to `Storage` address space
5. **Intrinsics**: Updated all array-related intrinsics to 3-arg format
6. **Helper functions**: Updated `sized_array`, `unsized_array`, `array_elem`, etc.

### Current Blocker
The prelude `filter` function fails type checking:

```wyn
def filter<[n], A>(p: A -> bool, xs: [n]A) ?k. [k]A =
  _w_intrinsic_filter(p, xs)
```

**Error**: During generalization, the type contains `SizeVar("k")` in positions where type variables should be:
- Predicate arg shows `SizeVar("k")` instead of type variable `A`
- Input array shows 2 args (old format) while output shows 3 args (new format)
- All type variables appear unified with `SizeVar("k")`

**Hypothesis**: During unification of the intrinsic's existential return type `?k. Array[a, s, SizeVar("k")]` with the declared return type `?k. Array[A, ?, SizeVar("k")]`, something is incorrectly propagating `SizeVar("k")` to other type variables.

## Files Changed
- `types/mod.rs` - TypeName variants, helper functions
- `parser.rs` - Array type parsing
- `types/checker.rs` - Pre-pass for marker conversion, entry param constraints
- `intrinsics.rs` - 3-arg array types for all intrinsics
- `flattening.rs`, `spirv/lowering.rs`, `glsl/lowering.rs` - Updated for new format
- Various other files for Slice/RuntimeArray removal

## Next Steps
1. Debug why `SizeVar("k")` leaks during existential type unification
2. May need to handle existential types specially in the type checker
3. Consider if the 2-arg array in the error indicates incomplete migration somewhere
