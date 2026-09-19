# Scatter and scratch

`scatter(destination, indices, values)` ignores out-of-bounds indices. If multiple inputs target the same valid index, they must supply identical values. Conflicting values have unspecified behavior; there is no last-writer guarantee. This is the caller's obligation, not an injectivity proof performed by the compiler.

`#[scratch] expression` retains an array expression's shape and element type while permitting uninitialized storage. For example:

```wyn
let output = #[scratch] replicate(length(xs), 0i32) in
scatter(output, destinations, xs)
```

The `0i32` determines the element type; it does not cause an initialization pass. This follows the array semantics of Futhark's [scratch attribute](https://futhark.readthedocs.io/en/latest/language-reference.html#scratch). Wyn currently accepts this attribute on array expressions. It is not a public allocation function.

Every element must be written before it is read. A full permutation scatter satisfies this condition; a partial scatter only permits subsequent reads of the written positions. The compiler does not prove full coverage. Normal, unannotated destinations retain their initial values at unwritten positions.

Scatter writes are parallel when its callback is safe and there is no detected in-place destination read conflict. Collision correctness and storage dependencies are separate obligations: the collision contract does not permit racing reads of a destination with its writes.

Scratch destinations skip both the fill and the initialization copy. Separate scatters from the same scratch expression still have independent results. Array lengths can be dynamic; invocation-local scratch retains the compiler's existing restrictions on local allocation sizes.
