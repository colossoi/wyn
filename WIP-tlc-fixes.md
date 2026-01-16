# WIP: TLC Path Fixes

## Current State

Working on fixing the TLC (Typed Lambda Calculus) transformation path to handle higher-order functions correctly.

## Completed

1. **Builtins fix for lambda lifting** (committed: f614188)
   - Lambda lifting was incorrectly capturing global intrinsics (sign, abs, etc.) as free variables
   - Added `builtins` parameter to `LambdaLifter` to exclude these names
   - Added `build_builtins()` helper in `lib.rs`

2. **Specialization pass** (committed: 9f08d2b)
   - Added `tlc/specialize.rs` to transform polymorphic intrinsics based on argument types
   - `sign(x)` where `x: f32` → `f32.sign(x)`
   - Same for `abs`, `min`, `max`, `clamp`

3. **Lambda lifting fix for nested lambdas** (in progress)
   - Bug: `λ(a). λ(b). a + b` was being split incorrectly
   - Inner lambda was lifted separately with `a` appearing free
   - Fix: Extract ALL nested lambda params together before lifting
   - Added `extract_lambda_params()`, `rebuild_nested_lam()`, `wrap_lam_with_captures()`

## In Progress

4. **Function references as values**
   - When a lifted lambda like `_lambda_0` is passed to `reduce`, it becomes `Expr::Global("_lambda_0")`
   - SPIR-V lowering doesn't handle `Expr::Global` for functions (only constants/uniforms)
   - Fix: Changed `to_mir.rs` to emit `Expr::Closure { lambda_name, captures: [] }` for top-level function references
   - **Status**: Just made this change, needs testing

## Validation Status

Passing:
- red_triangle.wyn
- red_triangle_curried.wyn
- simple_compute.wyn

Failing with various errors:
- test_2d_array.wyn - was "Undefined global: _lambda_0", testing closure fix
- entrylevel.wyn - "Undefined global: p" (likely same class of issue)
- Others have similar "Undefined global" errors or HOF issues

## Next Steps

1. Test the closure fix for function references
2. If that works, run full validation
3. May need to handle captures for lambdas that close over variables
4. May need special handling for map/reduce/filter intrinsics

## Debug Output

There's temporary debug output in `lib.rs:to_mir()` that prints lifted TLC:
```rust
eprintln!("=== LIFTED TLC ===\n{}\n==================", self.tlc);
```
Remove this before final commit.
