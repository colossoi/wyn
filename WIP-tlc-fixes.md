# WIP: TLC Path Fixes

## Current State

Working on fixing the TLC (Typed Lambda Calculus) transformation path to handle higher-order functions, polymorphic intrinsics, and lambda lifting correctly.

## Completed

1. **Builtins fix for lambda lifting** (committed: f614188)
   - Lambda lifting was incorrectly capturing global intrinsics (sign, abs, etc.) as free variables
   - Added `builtins` parameter to `LambdaLifter` to exclude these names
   - Added `build_builtins()` helper in `lib.rs`

2. **Specialization pass** (committed: 9f08d2b)
   - Added `tlc/specialize.rs` to transform polymorphic intrinsics based on argument types
   - `sign(x)` where `x: f32` → `f32.sign(x)`
   - Same for `abs`, `min`, `max`, `clamp`

3. **Lambda lifting fix for nested lambdas** (committed: 5f3b71b)
   - Bug: `λ(a). λ(b). a + b` was being split incorrectly
   - Inner lambda was lifted separately with `a` appearing free
   - Fix: Extract ALL nested lambda params together before lifting
   - Added `extract_lambda_params()`, `rebuild_nested_lam()`, `wrap_lam_with_captures()`

4. **Matrix literal types** (tlc/mod.rs)
   - Inner ArrayLiterals in VecMatLiteral now correctly transform to vectors (`_w_vec_lit`) instead of arrays (`_w_array_lit`) when parent type is Mat
   - Added `build_intrinsic_call_from_terms` helper for pre-transformed elements

5. **Global constants vs functions** (tlc/to_mir.rs)
   - Fixed distinction between top-level constants (arity=0) and functions (arity>0)
   - Constants now correctly emit `Expr::Global` instead of `Expr::Closure`
   - Fixed `extract_static_value` to handle constants properly

6. **Extended TLC intrinsic specialization** (tlc/specialize.rs)
   - Added `floor`, `ceil`, `fract` to specialization list (was only abs, sign, min, max, clamp)
   - Added `specialize_unary_type` to fix expression types for specialized intrinsics
   - Added `specialize_mul_partial_type` for mul_vec_mat/mul_mat_vec type fixing

7. **Impl source registration** (impl_source.rs)
   - Added `f32.fract` registration (was missing type-qualified version)

8. **Unit-type capture filtering** (tlc/lift.rs)
   - Added `is_unit_type` helper and filtering of unit-type captures
   - Removed dead code (`is_bound`, `wrap_with_captures`)

9. **Cleanup**
   - Removed debug output from lib.rs

## Validation Status

**Passing (7 files):**
- entrylevel.wyn
- kuko.wyn
- lava.wyn
- red_triangle.wyn
- red_triangle_curried.wyn
- seascape.wyn
- simple_compute.wyn

**Failing (4 files):**
- holodice.wyn - SPIR-V validation error: void parameters in function type
  - Unit-type filtering not catching all cases
  - Likely unit values propagating through loop body transformations
- da_rasterizer.wyn - Unknown function "map"
- primitives.wyn - UserVar('T') type variable reaching lowering
- raytrace.wyn - Undefined global "f32.pi"

## Underlying Issues

The current approach of special-casing each polymorphic intrinsic in TLC specialization is fragile. A more systematic solution might involve:

1. **Type propagation**: Propagating concrete types through the TLC tree more thoroughly, so that by the time we reach lambda lifting, all types are concrete
2. **Uniform intrinsic handling**: Handling polymorphic intrinsics uniformly rather than by name - perhaps by having the type checker annotate them with their concrete instantiation
3. **Earlier type resolution**: Ensuring type variables are resolved before lambda lifting, not during monomorphization
4. **Unit type elision**: Better integration to properly elide unit types throughout the entire pipeline (not just in capture filtering)

The void parameter issue in holodice suggests unit-type values are being passed through multiple stages - needs deeper investigation into where they originate (likely loop body destructuring with `_` patterns).

## Files Modified

- `wyn-core/src/tlc/mod.rs` - Matrix literal handling
- `wyn-core/src/tlc/to_mir.rs` - Global constant vs function closure handling, loop transformation
- `wyn-core/src/tlc/specialize.rs` - Extended intrinsic specialization with type fixing
- `wyn-core/src/tlc/lift.rs` - Unit-type capture filtering, dead code removal
- `wyn-core/src/impl_source.rs` - Added f32.fract registration
- `wyn-core/src/lib.rs` - Removed debug output

## Next Steps

1. Investigate the holodice void parameter issue more deeply - trace where unit types enter the pipeline
2. Consider a more systematic approach to polymorphic intrinsic handling
3. Address the remaining test failures (map, UserVar, f32.pi)
