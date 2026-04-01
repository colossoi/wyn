# SSA-Level Function Inlining — WIP

## Goal

Inline small user-defined functions at their call sites in SSA, so that
trivial helpers (e.g., a function that selects the Nth element from a
global array) don't survive as separate SPIR-V functions with
`OpFunctionCall` overhead.

## What's done

### wyn-ssa: `split_block_at` helper (`wyn-ssa/src/lib.rs`)

Generic utility that splits a block at a given instruction:
- Removes the instruction
- Creates a continuation block with the removed instruction's result as
  a block parameter (uses rewritten automatically)
- Moves subsequent instructions and the original terminator to the
  continuation block
- Leaves the original block's terminator as `Unreachable` (caller sets it)

### wyn-core: `ssa_inline` pass (`wyn-core/src/ssa/ssa_inline.rs`)

SSA-level inlining pass:
- Candidates: functions with ≤ 30 instructions, excluding `_w_lambda_*`
  (SOAC loop bodies) and functions containing SOAC instructions
- Does NOT inline into `_w_lambda_*` functions
- Uses `split_block_at` to split the caller at the call site
- Copies callee blocks into caller with fresh BlockIds/ValueIds
- Maps callee function params to call args via `ValueMapping` enum:
  - `Ssa(ValueId)` — direct mapping for SSA args
  - `Const(ConstantValue)` — keeps constants as `ValueRef::Const` in
    copied instructions via `substitute_values`, avoiding materialization
  - Constants are only materialized when they appear in terminators
    (which require raw `ValueId`)
- Rewrites callee `Return` terminators into branches to the continuation
- Runs `forward_single_pred_params` + `eliminate_empty_blocks` cleanup
- DCE removes functions that become unreferenced after inlining

### Pipeline position

```
to_ssa → inline_small → parallelize_soacs → filter_reachable → optimize → lower_soacs → materialize → lower
```

### soac_lower: RPO block iteration fix (`wyn-core/src/ssa/soac_lower.rs`)

After inlining, blocks in the slotmap are no longer in dominator order.
soac_lower's rebuild loop processes blocks sequentially and builds a
`value_map` as it goes — if a block is processed before the block that
defines a value it uses, the lookup panics.

Fix: compute RPO of the function body and iterate in that order instead
of slotmap insertion order.

## Current status

531 of 532 tests pass. 1 failure remains:

```
test_spirv_raytrace — panics in soac_lower with "no entry found for key"
```

The RPO fix resolved most block-ordering issues, but there's still a
value lookup failure in the raytrace shader specifically. This likely
needs the RPO iteration to also cover the pre-creation pass (currently
the pre-creation loop still uses slotmap order, which is fine for block
params but may interact badly with inlined blocks).

## Next steps

1. **Fix the remaining raytrace failure.** Debug which value is missing
   from `value_map` in soac_lower. Likely the pre-creation pass or RPO
   computation needs adjustment.

2. **Unit test: constant propagation through inlining.** Write a test
   with two functions:
   - A helper that takes an `i32` and extracts that element from a
     constant global array: `def helper(idx: i32) f32 = globalData[idx]`
   - A main function that calls it with a constant:
     `let x = helper(9) in ...`
   After inlining, this should produce `globalData[9]` directly — a
   constant index into a constant array. The compiler should resolve
   this at compile time with **no materialization** of the integer `9`
   as an SSA instruction. New constant-folding code in the SSA optimizer
   may be needed to support this (fold `Index { base: Global, index: Const }`
   into the extracted element).

3. **Tune the threshold.** The current 30-instruction limit is a
   starting point. Profile the raytrace shader output to find the sweet
   spot between code size and call overhead elimination.
