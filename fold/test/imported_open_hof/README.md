# Imported open-resolved math call fails HOF specialization

Compile only, from the repository root; no simulation or browser is involved:

```sh
target/release/wyn build fold/test/imported_open_hof/main.wyn -t wgsl --max-warnings 0 -o /tmp/imported-open-hof.wgsl
```

Expected: compile successfully, producing `[sqrt(2), 2 + sqrt(2)]`.

Fixed by master commit `74433605` (Preserve builtin identity through module
re-exports). Verified with a rebuilt release compiler: this reproducer now
compiles successfully.

Before that fix, the compiler exited 101 and panicked at
`wyn-core/src/tlc/inline.rs:274`:

```text
hof-specialization verifier failed after fold_generated_lambdas:
definition sym… parameter 1 still has a function type after HOF specialization
```

The symbol number varies with the surrounding source graph. This is the same
failure encountered compiling Fold's topology generator, reduced to two files
and two output elements. It requires no Fold simulation, graphics, or runtime.

Isolation checks: the equivalent single-file program compiles; qualifying the
helper's call as `f32.sqrt` also compiles. Those are diagnostic controls, not
changes to the application. No compiler fix is included.
