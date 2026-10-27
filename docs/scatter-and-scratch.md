# Scatter and scratch

`scatter(destination, indices, values)` ignores out-of-bounds indices. If multiple inputs target the same valid index, they must supply identical values. Conflicting values have unspecified behavior; there is no last-writer guarantee. This is the caller's obligation, not an injectivity proof performed by the compiler.

`duplicate_scratch(xs)` creates a fresh array with the shape and element type
of `xs`, without copying its elements. `scratch(n)` creates an uninitialized
array of length `n`; its element type is inferred from context or a type annotation.

```wyn
let output = duplicate_scratch(xs) in
scatter(output, destinations, xs)
```

```wyn
let output: []i32 = scratch(n) in
scatter(output, destinations, values)
```

Both forms use the same allocation machinery. The compatibility spelling
`#[scratch] expression` also retains the expression's shape and element type
without requiring initialization, including for `map` and `replicate` expressions.

Every element must be written before it is read. A full permutation scatter satisfies this condition; a partial scatter only permits subsequent reads of the written positions. The compiler does not prove full coverage. Normal, unannotated destinations retain their initial values at unwritten positions.

Scatter writes are parallel when its callback is safe and there is no detected in-place destination read conflict. Collision correctness and storage dependencies are separate obligations: the collision contract does not permit racing reads of a destination with its writes.

Scratch destinations skip both the fill and the initialization copy. Separate scatters from the same scratch expression still have independent results. Array lengths can be dynamic; invocation-local scratch retains the compiler's existing restrictions on local allocation sizes.
