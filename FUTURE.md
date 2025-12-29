# Future Work

Wishlist of future improvements and known incomplete features.

## Module System

### Functor argument type checking

Currently functor application `module M = F(Arg)` does not verify that `Arg` satisfies the module type declared for `F`'s parameter. The elaboration proceeds assuming the argument is valid.

**Needed:** Check that the argument module provides all types and functions required by the parameter's module type signature.

### Module body vs signature checking

When a module declares a signature `module M : SomeType = { ... }`, we don't verify that the body actually implements everything in `SomeType` with compatible types.

**Needed:** After elaborating a module body, check that it provides all items declared in the signature with matching types.

### Open declaration semantics

`open M` is parsed but has no effect at the value level. The `NameResolver` struct exists but is marked `#[allow(dead_code)]` and unused.

**Current state:**
- `Declaration::Open(_)` in `elaborate_module_body` is a TODO stub
- Name resolution is done ad-hoc in `resolve_names_in_expr` using:
  - `local_bindings` (function params, let bindings)
  - `module_functions` (intra-module definitions)
  - `known_modules` for `Module.field` patterns via FieldAccess rewrites
  - `param_bindings` for functor parameter references
- Prelude functions are found via `is_prelude_function` / `get_prelude_function` at later stages

**Result:** Unqualified names can only resolve to local bindings or same-module functions. `open SomeModule` does nothing.

**Options:**
1. Wire `NameResolver` into `resolve_names_in_expr` and handle `Declaration::Open`
2. Remove `NameResolver` until ready to implement properly

Either way, document that `open` is not functional to avoid confusion.
