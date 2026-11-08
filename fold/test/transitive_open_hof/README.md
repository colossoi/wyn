# Transitive import HOF specialization regression

Compile only, from the repository root:

```sh
target/release/wyn build fold/test/transitive_open_hof/main.wyn -t wgsl --max-warnings 0 -o /tmp/transitive-open-hof.wgsl
```

Expected: successful compilation, producing `[sqrt(2), 2 + sqrt(2)]`.

Fixed by master commit `3beb084e` (Resolve opened builtin members in nested
module paths). Verified with a rebuilt release compiler: this two-file
reproducer now compiles successfully.

Previously on master `74433605`, the compiler exited 101,
with a panic at `wyn-core/src/tlc/inline.rs:274`:

```text
hof-specialization verifier failed after fold_generated_lambdas:
definition sym… parameter 1 still has a function type after HOF specialization
```

Reduced to two files: `helper.wyn` contains a nested module and re-exports
its array constant. Importing that file previously panicked. Putting the same nested
modules entirely in one file compiles. A direct import without the nested
module also compiles after `74433605`.

No graphics, simulation execution, browser, compiler changes, or application
workarounds are involved.
