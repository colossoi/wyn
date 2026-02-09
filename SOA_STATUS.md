# SoA Transform — Status

## What's Done (committed)

1. **SoA transform pass** (`soa_transform.rs`): rewrites `[n](A,B)` → `([n]A, [n]B)` as a TLC-to-TLC pass after monomorphize.
2. **Pipeline wiring**: `TlcMonomorphized` → `.soa_transform()` → `TlcSoaTransformed` → `.to_ssa()`. All call sites updated.
3. **SoA-aware SSA lowering helpers** in `to_ssa.rs`:
   - Free functions: `is_soa_tuple()`, `soa_elem_type()`, `extract_array_size()`
   - Converter methods: `soa_length()`, `soa_index()`, `soa_array_with()`, `soa_uninit()`
   - `_w_index` handler delegates to `soa_index`
   - `convert_for_in_loop`, `convert_soac_map`, `convert_soac_reduce` all use the helpers
4. **Multi-arg eta-expansion fix** (`tlc/mod.rs`): `term_to_lambda` now decomposes the full arrow chain, fixing the `vecAdd` panic in raytrace.wyn.
5. **500/500 unit tests pass.**

## What's In Progress (uncommitted, this commit)

Fixing nested SoA types for raytrace.wyn. The original code produced `([8](int,vec3), [8]vec3)` for `[8]((int,vec3),vec3)` instead of the fully-distributed `(([8]int, [8]vec3), [8]vec3)`.

### Changes in this commit

**`soa_transform.rs`**:
- `soa_type()` now recursively applies `soa_type` to distributed component arrays
- `rewrite_index_aot`, `rewrite_array_with_aot`, `rewrite_array_lit_aot` wrap constructed array types in `soa_type()`

**`to_ssa.rs`**:
- `is_soa_tuple()` now recognizes nested SoA tuples (component can be a tuple-of-arrays, not just a plain array)
- `soa_elem_type()` recurses on nested SoA components
- `soa_index()` computes `comp_elem_ty` correctly for nested SoA via `soa_elem_type`
- `soa_array_with()` recurses instead of emitting `_w_intrinsic_array_with` on nested SoA components
- `soa_uninit()` recurses instead of emitting `_w_intrinsic_uninit` on nested SoA components
- `soa_length()` already recursed (done earlier)

## Known Issue

raytrace.wyn still fails with a SPIR-V error ("Index called on non-array/non-pointer type" on a deeply nested SoA tuple). The recursive fixes above address the root cause identified via debug tracing (`push_index` was being called from `soa_index` → `convert_soac_map` on a 3-level nested SoA tuple because `is_soa_tuple` didn't recognize nested tuples). **This has not been re-tested after the final round of fixes.**

## Plan Going Forward

- Phase 2/3 (eliminating tuples from late pipeline) are **dropped** — tuples are allowed for composite return types and loop vars.
- Next: verify raytrace.wyn compiles and passes spirv-val after these nested SoA fixes.
- If further issues surface, they'll likely be in SPIR-V lowering encountering SoA tuple types it doesn't expect.
