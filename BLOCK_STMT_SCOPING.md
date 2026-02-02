# Block Statement Scoping Issue (WIP)

## Problem Summary

Loop variables are undefined during SPIR-V lowering. The validation script shows failures like:

```
Error: CompilationError(SpirvError("Undefined local variable: i (local_id=LocalId(2), env has: [\"p\"])", None))
```

This affects most test files that use loops with let bindings inside the body.

## Root Cause

The MIR representation uses `Block { stmts, result }` for loop bodies and if branches. The issue is that:

1. **`to_mir.rs`** creates loops with `Block::new(body_id)` which has empty `stmts`
2. **`normalize.rs`** also creates `Block::new(new_loop_body)` with empty `stmts`
3. Bindings inside loop bodies are emitted as **function-level** statements in `body.stmts`
4. **SPIR-V lowering** processes function-level statements once at the start via `lower_body_with_stmts()`
5. When lowering the loop body via `lower_block()`, only `Block.stmts` (empty) are processed

The result: bindings that depend on loop variables (like `i`) are evaluated once at function entry where `i` doesn't exist, rather than on each loop iteration.

## Example

```wyn
def sdfScene(p: vec3f32) f32 =
  let (_, d) = loop (i, d) = (0, 2.0) while i < 16 do
    let fi = f32.i32(i) in    -- fi depends on i, but emitted at function level
    ...
    (i + 1, d2)
  in d
```

The binding `fi = f32.i32(i)` ends up in `body.stmts` (function level), but `i` is only defined inside the loop header. When `lower_body_with_stmts()` tries to evaluate `fi`, `i` isn't in the environment yet.

## Where the Problem Occurs

### 1. `to_mir.rs` (lines ~536-547)

```rust
body.alloc_expr(
    Expr::Loop {
        loop_var: loop_var_local,
        init: init_id,
        init_bindings: mir_init_bindings,
        kind: mir_kind,
        body: Block::new(body_id),  // <-- Empty stmts
    },
    ...
)
```

### 2. `normalize.rs` (lines ~441-448)

```rust
Expr::Loop {
    loop_var: *loop_var,
    init: atom_init,
    init_bindings: new_init_bindings,
    kind: new_kind,
    body: Block::new(new_loop_body),  // <-- Empty stmts, discards original
}
```

### 3. SPIR-V lowering (`spirv/lowering.rs`)

`lower_body_with_stmts()` processes all function-level statements first, then `lower_expr()` is called on the root. When it encounters a Loop, it calls `lower_block()` which only processes the (empty) `Block.stmts`.

## Fix Options

### Option A: Put loop-scoped bindings in Block.stmts

Change `to_mir.rs` to identify which statements are "inside" the loop and put them in `Block.stmts` rather than function-level `body.stmts`.

**Challenge**: The current flattening approach creates all statements at function level. Would require dependency analysis to determine which statements belong inside loops.

### Option B: Defer function-level statement evaluation

Change SPIR-V lowering to not pre-evaluate all statements. Instead, evaluate each statement lazily when its LHS local is first referenced.

**Challenge**: Requires tracking which statements have been evaluated. May affect code generation order.

### Option C: Inline expansion of loop bodies

Before lowering, inline all the statements that a loop body depends on into the loop. Essentially "unflatten" the loop body.

**Challenge**: Duplicates statements if used both inside and outside loop.

### Option D: Restore Let expressions for loop bodies

Keep `Let` expressions for bindings inside loops/ifs instead of flattening them. Only flatten truly sequential bindings at function level.

**Challenge**: Partially reverses the flattening work.

## Current Investigation

Looking at the MIR output for `lava.wyn`:

```
def sdfScene (p: vec3f32): f32 =
    locals:
    0: p
    1: _loop_108
    2: i          <-- loop iteration variable
    3: d
    4: fi         <-- depends on i
    ...
  stmts:
    [function-level statements including fi = f32.i32(local_2)]
  exprs:
    e0: local_2   <-- reference to i
    e1: f32.i32(e0)
    ...
```

The statement `fi = f32.i32(local_2)` is at function level, but `local_2` (i) only exists inside the loop.

## Files Involved

- `wyn-core/src/tlc/to_mir.rs` - Creates MIR from TLC, emits all stmts at function level
- `wyn-core/src/normalize.rs` - ANF normalization, creates `Block::new()` with empty stmts
- `wyn-core/src/mir/mod.rs` - Block struct definition
- `wyn-core/src/spirv/lowering.rs` - `lower_body_with_stmts()`, `lower_block()`, `lower_expr()`

## Test Files Affected

Most files with loops containing let bindings:
- `lava.wyn` - While loop with let bindings
- `seascape.wyn` - For loops with let bindings
- `holodice.wyn` - Nested loops
- `primitives.wyn` - Various loop patterns
- `kuko.wyn` - Complex scene with loops
- `entrylevel.wyn` - Entry point with loops
- etc.

Only simple files like `red_triangle.wyn` (no loops) pass validation.

## Next Steps

1. Decide on fix approach (A, B, C, or D)
2. Implement the fix
3. Run validation script to verify all test files pass
