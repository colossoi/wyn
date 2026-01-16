# WIP: Fix validation failures (raytrace.wyn, da_rasterizer.wyn)

## Status
In progress - intrinsic arities added to to_mir, need to fix remaining issues.

## Completed
- Consolidated alias resolution to name_resolution.rs (committed)
- Added intrinsic arity support to TLC-to-MIR (uncommitted)

## Remaining Issues

### Issue 1: `normalize` alias not resolved in raytrace.wyn
- **Error**: `UndefinedVariable("normalize")` at line 118
- **Root Cause**: `name_resolution.rs` doesn't handle `Declaration::Module` - skips with `_ => {}`
- **Fix**: Add Module handling in `resolve_declaration` to recurse into module body declarations

### Issue 2: Lambda closure construction in da_rasterizer.wyn
- **Error**: `Expected closure or function reference, got Call { func: "_lambda_3", args: [...] }`
- **Root Cause**: `apply_to_captures` creates App chains that become `Expr::Call`, but SPIR-V expects `Expr::Closure`
- **Fix**: In `to_mir.rs`, detect when partial application of `_lambda_*` with arrow result type = closure construction

## Files to Modify
- `wyn-core/src/name_resolution.rs` - Add Module handling
- `wyn-core/src/tlc/to_mir.rs` - Detect closure construction pattern
