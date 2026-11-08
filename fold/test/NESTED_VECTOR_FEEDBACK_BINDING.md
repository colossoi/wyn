# Selected nested vector feedback has no external binding

Compile only, from the repository root:

```sh
target/release/wyn build fold/test/nested_vector_feedback_binding.wyn --graphics -t wgsl --max-warnings 0 -o /tmp/nested-vector-feedback.wgsl
```

Fixed by master commit `66e3afca` (Preserve storage dependencies for nested
vector projections). Verified with a rebuilt release compiler: the reproducer
now compiles successfully.

Expected: successful compilation. Previously, with a rebuilt compiler at master
`8e9312a3`:

```text
error: egglog output: external storage Value(561) has no binding; type Some(Constructed(Float(32), []))
```

The value number depends on the surrounding program. The complete Fold build
reports `Value(43324)` with the same error and type.

This single-file case selects an array-bearing state from reset or input,
uses its position array in a vertex callback, and packs a nested vector/count
record into feedback. The geometry is intentionally degenerate: this is only
a compilation reproducer. No runtime or browser execution is needed. No fix
or application workaround is included.
