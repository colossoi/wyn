# Partial Evaluation Status

## Current Implementation

The partial evaluator (NBE - Normalization by Evaluation) handles:
- Scalar constants (int, float, bool)
- Binary operations with algebraic identities (x + 0, x * 1, etc.)
- Function inlining when all arguments are known
- Let bindings
- Conditionals with known conditions (dead branch elimination)
- Tuple/array/vector construction and access
- Basic intrinsics (arithmetic, comparisons)

## Known Limitation: `_w_intrinsic_map` not evaluated

The `_w_intrinsic_map` SOAC intrinsic is **not yet handled** by partial evaluation.

Example MIR (from `map(double, [1.0, 2.0, 3.0])`):
```
e5: [1, 2, 3]
e6: _w_intrinsic_map(double, e5)
e8: @tuple_access(e6, 0)
```

Expected after partial eval:
```
e0: 2    # double(1.0) = 2.0
```

Actual: The `_w_intrinsic_map` call is preserved, and the SPIR-V lowerer handles it.

## Why it still works

With `--partial-eval`, compilation succeeds because the SPIR-V lowerer can handle `_w_intrinsic_map` at lowering time. Without the flag, it fails with "Global constants must be literals" because some other path in the pipeline requires folded constants.

## Future Work

To fully evaluate `map` at compile time, add handling in `eval_intrinsic()`:
1. Check if the function argument is a known closure/function reference
2. Check if the array argument is a fully known `Value::Aggregate`
3. If both known, iterate and apply the function to each element
4. Return `Value::Aggregate` with the results

Similar handling needed for other SOACs: `reduce`, `scan`, `filter`, etc.
