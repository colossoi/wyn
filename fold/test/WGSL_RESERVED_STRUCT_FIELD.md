# Reserved generated WGSL struct field

Compile and validate without a browser or GPU execution:

```sh
target/release/wyn build fold/test/wgsl_reserved_struct_field.wyn -t wgsl -o /tmp/reserved-field.wgsl
fold/tools/validate-wgsl/target/release/fold-validate-wgsl /tmp/reserved-field.wgsl
```

Fixed by master commit `32f7a1a9` (Use legal WGSL identifiers for all generated
struct fields). Verified with a rebuilt compiler and Naga validation: this
reproducer and the complete Fold WGSL now pass validation.

Previously on master `66e3afca`, Wyn compilation succeeded but Naga rejected the output:

```text
error: name `f16` is a reserved keyword
20 |     f16: f32,
```

A record with 17 fields triggers the generated field name `f16`; the same
problem occurs in the complete Fold shader, where that field has type `bool`.
The source field names (`a0` through `a16`) are valid. This is an emitter issue,
not a requirement to change the application record layout. No workaround is
included.
