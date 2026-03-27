# SOAC Map Unrolling — Work In Progress

## Motivation

When `map f arr` operates on a small fixed-size array, the SOAC lowering currently emits a for-range loop with an accumulator. Each iteration updates one element via `array_with`, which in SPIR-V requires:

1. `OpStore` the whole array to a local variable
2. `OpAccessChain` to the element
3. `OpStore` the new element value
4. `OpLoad` the whole array back

That's 4 memory ops per field update × N fields per iteration. For a 3-field struct-of-arrays with 12 elements, that's 12 memory ops per iteration × 12 iterations = 144 memory ops.

The alternative for small arrays: unroll the map. Emit N individual function calls, then construct the result array directly with `OpCompositeConstruct`. No loop, no accumulator, no copy-modify cycle.

## Implementation (partial)

Added to `soac_lower.rs::expand_map`:

```rust
const UNROLL_THRESHOLD: usize = 16;
if let Some(n) = array_size {
    if n <= UNROLL_THRESHOLD {
        return expand_map_unrolled(...);
    }
}
```

`expand_map_unrolled` emits:
- For each index 0..N: index each input array at that position, call the map function, collect the result
- Construct the output array from the N results via `ArrayLit`

This works for simple array types.

## Issue: SoA (Structure of Arrays) transform

The SoA transform converts `[N](A, B, C)` into `([N]A, [N]B, [N]C)` — a tuple of arrays instead of an array of tuples. This is applied before SOAC lowering.

When SoA is active, the map's result type is `([12]float, [12]int, [12]vec3)`, not `[12](float, int, vec3)`. The unrolled map produces 12 `(float, int, vec3)` values, but packing them into an `ArrayLit` fails because the result type expects 3 arrays, not 12 structs.

The fix requires **transposing** the results:
1. From the N `(A, B, C)` results, extract field 0 from each → pack into `[N]A`
2. Extract field 1 from each → pack into `[N]B`
3. Extract field 2 from each → pack into `[N]C`
4. Pack the 3 arrays into `([N]A, [N]B, [N]C)`

This is straightforward but adds complexity. The helper `is_soa_tuple` and `soa_elem_type` from `soa_helpers.rs` detect and describe the SoA layout.

## Options

1. **Implement the transpose** — handle SoA in `expand_map_unrolled` by detecting `is_soa_tuple` on the result type and transposing. ~20 lines of additional code.

2. **Lower the threshold** — set `UNROLL_THRESHOLD` to 4 to only catch vec2/vec3/vec4 maps where SoA is unlikely. The raytrace's 12-sphere map would still use the loop path.

3. **Run unrolling before SoA** — if the unroll happens before SoA transform, the result type is still `[N](A,B,C)` and `ArrayLit` works directly. But this requires pipeline reordering.

## Current state

The code change was reverted. The `expand_map_unrolled` function exists conceptually but needs the SoA transpose to handle the raytrace case. All other test files (which don't use SoA on their map results) passed validation.
