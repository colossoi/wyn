# Discriminate `EntryOutput` By Shader Stage

*Part of the "make invalid states unrepresentable" theme — same spirit as
`c06bf458`.*

## Context

`ssa/types.rs:579`:

```rust
pub struct EntryOutput {
    pub ty: Type<TypeName>,
    pub decoration: Option<IoDecoration>,
    pub storage_binding: Option<BindingRef>,   // Some for compute, None for graphics
    pub length: Option<BufferLen>,             // correlated with storage_binding
}
```

Whether an entry output has a storage binding is determined by the *execution
model*: a compute output writes to a storage buffer (must have one); a vertex /
fragment output is an interface variable (must not). The flat struct lets the
two fields float free of the stage.

## Invalid states currently representable

- A compute `EntryOutput` with `storage_binding: None`.
- A graphics `EntryOutput` with `storage_binding: Some(_)`.
- `length: Some(_)` with `storage_binding: None` (orphaned length).

## How it's enforced today (runtime)

- `egir/realize_outputs/mod.rs:104,192` —
  `output.storage_binding.expect("BUG: compute output without storage binding")`.
- `egir/realize_outputs/dispatch.rs:607` —
  `output.storage_binding.expect("compute output has a storage binding")`.

## The tightening

```rust
pub enum EntryOutput {
    Vertex   { ty: Type<TypeName>, decoration: Option<IoDecoration> },
    Fragment { ty: Type<TypeName>, decoration: Option<IoDecoration> },
    Compute  { ty: Type<TypeName>, storage_binding: BindingRef,      // required
               length: Option<BufferLen> },
}
```

The compute binding is non-optional by construction; the graphics variants
can't carry one; `length` lives only where it is meaningful.

## Blast radius

~30 sites: construction in `from_tlc.rs`, all iteration in `realize_outputs/`,
`publish.rs`, the SPIR-V backend, tests. Medium. Removes the three
storage-binding `.expect`s and makes the stage/binding correlation a compile
error to violate.

## Verification

1. `cargo test -p wyn-core`.
2. `scripts/validate_testfiles.sh` (SPIR-V) and `--wgsl` — both unchanged
   (graphics + compute entries exercised).
3. `grep -n 'storage_binding.expect' wyn-core/src/egir/` → zero.
