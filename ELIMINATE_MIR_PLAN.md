# Plan: Eliminate MIR, Lower TLC Directly to SSA

## Goal

Remove MIR as a separate intermediate representation.

**Before:**
```
TLC → MIR → SSA → SPIR-V
```

**After:**
```
TLC → SSA → SPIR-V
```

## Current Status

This branch contains work-in-progress toward eliminating MIR. The code **will not compile** - many MIR-dependent files have been removed but not all references updated.

### Completed

1. **Phase 1: Created `tlc/to_ssa.rs`** ✓
   - Direct TLC → SSA conversion (bypassing MIR)
   - Handles all TermKind variants: Var, IntLit, FloatLit, BoolLit, Let, If, App, Loop, Extern
   - Expands SOACs (`_w_intrinsic_map`, `_w_intrinsic_reduce`) to explicit loops
   - Added `Clone` derives to `SsaProgram`, `SsaFunction`, `SsaEntryPoint`

2. **Phase 2: Updated pipeline stages in `lib.rs`** ✓
   - Added `TlcMonomorphized::to_ssa()` method
   - Added `SsaConverted` stage holding `SsaProgram`
   - Added `SsaConverted::parallelize_soacs()` → `SsaParallelized`

3. **Phase 3: SSA-level SOAC analysis and parallelization** ✓
   - Created `mir/ssa_soac_analysis.rs`:
     - Loop detection via back-edge detection in CFG
     - Map loop pattern recognition
     - Array provenance tracking to entry storage buffers
   - Created `mir/ssa_parallelize.rs`:
     - Thread chunking transformation for parallelizable compute shaders
     - Storage view creation for input/output
     - Single-thread fallback for non-parallelizable shaders
   - Removed redundant `tlc/soac_analysis.rs` and tests

4. **Phase 4: SPIR-V lowering (partial)**
   - Added `lower_ssa_program()` function in `spirv/ssa_lowering.rs`
   - Made Constructor methods public: `new()`, `begin_function()`, `create_uniform_block_type()`
   - Renamed `ast_type_to_spirv` → `polytype_to_spirv` for clarity

### Files Removed (This Branch)

These files were part of the MIR-based pipeline and have been removed:

| File | Purpose |
|------|---------|
| `mir/soac_analysis.rs` | MIR-level SOAC analysis |
| `mir/to_ssa.rs` | MIR → SSA conversion |
| `mir/transform.rs` | MIR transformation utilities |
| `mir/parallelism.rs` | MIR parallelism utilities |
| `tlc/to_mir.rs` | TLC → MIR conversion |
| `tlc/to_mir_tests.rs` | Tests for TLC → MIR |
| `tlc/soac_analysis.rs` | TLC-level SOAC analysis (redundant) |
| `tlc/soac_analysis_tests.rs` | Tests for TLC SOAC analysis |
| `soac_parallelize.rs` | MIR-level SOAC parallelization |
| `default_address_spaces.rs` | MIR address space defaulting |
| `mir_transform_tests.rs` | MIR transform tests |

### Files Added

| File | Purpose |
|------|---------|
| `tlc/to_ssa.rs` | Direct TLC → SSA conversion |
| `mir/ssa_soac_analysis.rs` | SSA-level SOAC pattern detection |
| `mir/ssa_parallelize.rs` | SSA-level SOAC parallelization |

### Files Modified

| File | Changes |
|------|---------|
| `lib.rs` | Removed MIR pipeline stages, added SSA stages |
| `mir/mod.rs` | Removed MIR-specific module declarations |
| `tlc/mod.rs` | Removed `to_mir` module |
| `spirv/ssa_lowering.rs` | Added `lower_ssa_program()` for direct SSA → SPIR-V |
| `spirv/lowering.rs` | Made Constructor methods `pub(crate)` |

## Remaining Work

### Phase 4: Complete SPIR-V Lowering

1. Wire up `SsaParallelized.lower()` to call `spirv::ssa_lowering::lower_ssa_program()`
2. Update `spirv/lowering.rs` to remove MIR-specific code paths
3. Handle GLSL lowering path (or defer)

### Phase 5: Fix Compilation Errors

1. Update test files that reference `to_mir()`:
   - `integration_tests.rs`
   - `desugar_tests.rs`
   - `spirv/lowering_tests.rs`

2. Update any code that references removed modules:
   - `dps_transform.rs` - operates on MIR, needs rewrite or removal
   - `reachability.rs` - operates on MIR, needs rewrite for SSA
   - `pipeline.rs` - may reference MIR types
   - `glsl/lowering.rs` - operates on MIR

3. Clean up MIR types in `mir/mod.rs`:
   - Keep: `ssa.rs`, `ssa_builder.rs`, `ssa_verify.rs`, `ssa_soac_analysis.rs`, `ssa_parallelize.rs`
   - Decide what to do with remaining MIR types (may still be needed for some lowering)

### Phase 6: Validation

1. Get all tests passing
2. Verify SPIR-V output matches for test shaders
3. Run validation script on testfiles

## New Pipeline

```
TLC Program
    ↓ tlc/to_ssa.rs
SSA Program (SsaConverted)
    ↓ mir/ssa_soac_analysis.rs (detect patterns)
    ↓ mir/ssa_parallelize.rs (for compute shaders)
SSA Program (SsaParallelized)
    ↓ spirv/ssa_lowering.rs
SPIR-V
```

## Design Decisions

1. **SOAC analysis at SSA level**: Detects map loop patterns in the CFG directly, tracks array provenance to storage buffers.

2. **Parallelization at SSA level**: Transforms SSA CFG to add thread chunking - cleaner than expression tree manipulation.

3. **No interprocedural analysis yet**: Current SSA analysis only finds maps directly in entry point bodies. Maps inside helper functions aren't detected (future work).

4. **DPS transformation**: Needs decision - rewrite for SSA or handle differently?

## Known Issues

- Compilation errors throughout (expected - files removed)
- `dps_transform.rs` still operates on MIR
- GLSL lowering path not updated
- Many tests reference removed `to_mir()` method
