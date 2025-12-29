# Work In Progress - Inline Modules

## What's Done

### Inline Module Support in Compiler
- Added `elaborate_modules` method to `ModuleManager` that processes `module Name = { ... }` declarations in user programs
- Added `elaborate_modules` step to the compilation pipeline in `lib.rs` and `main.rs`
- Module bindings are elaborated and then removed from the AST before type checking
- Updated type checker tests to include the elaboration step

### Raytracer Refactored with Inline Module
- Created `module materials = { ... }` containing PBR and glass material functions:
  - `pbrFresnelSchlick`, `pbrDistributionGGX`, `pbrGeometrySchlickGGX`, `pbrGeometrySmith`
  - `brdf`, `hash`, `fresnelGlass`, `refractRay`
- Functions are called as `materials.brdf(...)`, `materials.fresnelGlass(...)`, etc.
- Module defines its own `epsilon` constant (modules are self-contained, can't see outer scope)

### Prelude Fixes
- Fixed `cbrt` to use `x ** (1.0f32 / 3.0f32)` instead of non-existent `f32.pow`
- Removed `to_i64` and `to_f64` functions that had type errors
- Removed `get_bit` and `set_bit` from the `float` module type signature (stubs not needed)

## What's Left

### Prelude Stubs to Implement
In the f32 module, these are still stubs:

| Function | Current | Needed |
|----------|---------|--------|
| `get_bit` | stub returning 0 | **remove entirely** |
| `set_bit` | stub returning x | **remove entirely** |
| `ldexp(x, n)` | returns x | `x * 2^n` |
| `erf(x)` | returns x | error function (musl algorithm) |
| `erfc(x)` | returns x | complementary error function |
| `gamma(x)` | returns x | gamma function (Lanczos approx) |
| `lgamma(x)` | returns x | log gamma function |
| `nextafter(x, y)` | returns x | next representable float |

### Module Scoping
Modules are intentionally self-contained (Futhark-style):
- Cannot reference outer scope
- Options: define inside module, pass as parameter, or use parameterized module (functor)

## Current Error
After prelude fixes, need to:
1. Remove `get_bit`/`set_bit` defs from f32 module
2. Implement `ldexp`, `erf`, `erfc`, `gamma`, `lgamma`
3. Then raytracer should compile

## Files Changed
- `wyn-core/src/module_manager/mod.rs` - module elaboration
- `wyn-core/src/lib.rs` - pipeline changes
- `wyn/src/main.rs` - CLI changes
- `wyn-core/src/types/checker.rs` - type checking with module decls
- `wyn-core/src/types/checker_tests.rs` - test helper updates
- `prelude/math.wyn` - prelude fixes
- `testfiles/raytrace.wyn` - inline materials module
