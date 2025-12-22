# ModuleManager Refactoring Notes

## Completed

1. **Removed type-checking from `create_prelude()`** - PreElaboratedPrelude now only contains module data (no type tables or schemes)

2. **Removed `prelude_type_table` and `prelude_schemes` from PreElaboratedPrelude and ModuleManager** - These fields are no longer needed since type-checking happens during the pipeline

3. **Added `check_prelude_functions()` call to `check_program()`** - Prelude functions are now type-checked as part of normal type-checking flow

4. **Updated pipeline API** - `type_check()` and `flatten()` now take specific fields instead of `&mut FrontEnd`:
   - `type_check(module_manager, &mut schemes)`
   - `flatten(module_manager, &schemes)`

5. **Removed getter methods from ModuleManager**:
   - `get_prelude_type_table()`
   - `get_prelude_schemes()`
   - `get_prelude_scheme()`

## Remaining Work

### Remove polytype import from ModuleManager

The file `wyn-core/src/module_manager/mod.rs` still imports polytype:
```rust
use polytype::{Context, TypeScheme};
```

These are used by:
- `get_module_function_type()` - takes `&mut Context<TypeName>`, returns `TypeScheme`
- `get_prelude_function_type()` - takes `&mut Context<TypeName>`, returns `TypeScheme`
- `convert_to_polytype()` helper
- `build_function_type_from_decl()` helper

To remove polytype from ModuleManager:
1. Move `convert_to_polytype()` and related helpers to TypeChecker
2. Have TypeChecker compute function types from Decl on demand
3. Remove the `get_*_function_type()` methods from ModuleManager
4. Update TypeChecker to use its own methods for looking up module/prelude function types

### Clean up FrontEnd struct

FrontEnd currently has fields that may not be needed:
- `context` - only used to create IntrinsicSource
- `type_table` - currently empty, not populated
- `intrinsics` - created fresh, not shared with TypeChecker
- `module_schemes` - cache for per-module function schemes (not yet used)

Consider whether FrontEnd should just hold:
- `node_counter`
- `module_manager`
- `schemes` (populated by type_check)

### Arc<ElaboratedModule> cache pattern

The original plan included making ModuleManager return `Arc<ElaboratedModule>` for lazy loading. This hasn't been implemented yet. Current state:
- `elaborated_modules: HashMap<String, ElaboratedModule>` (owned, not Arc)

### Test flakiness

There's a flaky test `spirv::lowering::tests::test_map_variants` that passes when run alone but sometimes fails when run with all tests. This appears to be related to the static prelude cache and Context/type variable allocation across tests.
