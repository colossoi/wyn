# WIP Status: Prelude Type-Checking and SOAC Flattening

## Current State

The prelude functions (map, reduce, scan, filter, scatter, zip) are now:
1. Parsed from `prelude/soacs.wyn`
2. Type-checked during `create_prelude()`
3. Their type table is stored in `PreElaboratedPrelude`

However, **tests are failing** because the defun analysis pass doesn't include prelude functions.

## The Problem

In `lib.rs:flatten()`:
```rust
let defun_analysis = defun_analysis::analyze_program(&self.ast, &type_table, &builtins);
```

This only analyzes the user's AST, not the prelude function declarations. When we later try to flatten prelude functions, they reference NodeIds that have no defun classification, causing:

```
BUG: No defun classification for NodeId(...). Analysis pass missed this node.
```

## Fix Needed

Extend defun analysis to also cover prelude functions. Options:
1. Add `analyze_declarations(&[Decl])` method to DefunAnalyzer
2. Call it for prelude functions after analyzing user program
3. Pass combined declarations to a single analysis call

## Files Modified So Far

- `wyn-core/src/module_manager/mod.rs` - Added `prelude_type_table` field, type-checking in `create_prelude()`
- `wyn-core/src/types/checker.rs` - Added `check_prelude_functions()` and `into_type_table()`
- `wyn-core/src/lib.rs` - Merge type tables, flatten prelude functions

---

# Future Work: Liveness Data from Alias Checker

## Background

The alias checker (`wyn-core/src/alias_checker.rs`) performs dataflow analysis to track:
- Which values are "live" (will be used again)
- Which values are "last use" (consumed, can be moved)
- Aliasing relationships between values

Currently this information is computed but **not communicated out** of the alias checker for use by later passes.

## Why We Need Liveness Data

For efficient SPIR-V/GPU code generation, we need to know:

1. **Last-use optimization**: When a value is used for the last time, we can:
   - Avoid unnecessary copies
   - Reuse the underlying storage
   - Generate more efficient in-place operations

2. **Array mutation**: SOACs like `scatter` modify arrays. Knowing liveness helps determine:
   - Whether the input array can be mutated in-place
   - Whether we need to copy before mutation

3. **Memory management**: GPU memory is limited. Knowing when values die allows:
   - Earlier deallocation
   - Better register allocation in SPIR-V

## Current State

The alias checker computes `LivenessInfo` per-block:
```rust
pub struct LivenessInfo {
    pub live_in: HashSet<MirId>,
    pub live_out: HashSet<MirId>,
    pub last_uses: HashMap<MirId, /* statement index */>,
}
```

But this is only used internally for validation, not exposed for codegen.

## Proposed Solution

1. **Return liveness data from alias checker**:
   ```rust
   pub fn check_program(mir: &Mir) -> Result<LivenessResults> {
       // ... existing analysis ...
       Ok(LivenessResults {
           block_liveness: computed_liveness,
           global_last_uses: last_use_map,
       })
   }
   ```

2. **Use in SPIR-V lowering**:
   - Mark last-use loads as "consumable"
   - Enable in-place array operations
   - Guide OpVariable placement

3. **Annotate MIR** (alternative):
   - Add `is_last_use: bool` field to `Operand`
   - Populated by alias checker
   - Simpler for lowering to consume

## Implementation Steps

1. Change alias checker return type to include liveness
2. Thread liveness data through to SPIR-V lowering
3. Use liveness info when generating OpLoad/OpStore
4. Add tests for last-use optimization
