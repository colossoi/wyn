# Production panic-extraction audit

This audit covers production Rust targets in the workspace. The semantic scan is:

```text
cargo clippy --workspace --lib --bins -- \
  -W clippy::unwrap_used \
  -W clippy::expect_used \
  -W clippy::panic
```

A companion scan enables `clippy::unreachable`, `clippy::todo`, and
`clippy::unimplemented`. Production `assert!`, `assert_eq!`, and `assert_ne!`
calls were reviewed with a source scan because Clippy has no general lint for
them.

Unlike a text search, this excludes test-only code and methods named `expect`
that are not `Option::expect` or `Result::expect`. Safe fallback methods such as
`unwrap_or` are intentionally out of scope. Direct indexing is not counted, so
the same structural remedies below must also be applied to adjacent `map[key]`
and `values[index]` operations when a cluster is changed.

## Snapshot

The starting tree contained 584 production panic points detected by this scan:
466 `unwrap`/`expect` extractions and 118 explicit `panic!` calls.

This change removes 56 of them. The remaining 528 are:

| Kind | Count |
| --- | ---: |
| `Option`/`Result::expect` | 306 |
| `Option`/`Result::unwrap` | 104 |
| `panic!` | 118 |

The remaining sites are concentrated by compiler layer:

| Layer | Count |
| --- | ---: |
| EGIR | 224 |
| TLC | 89 |
| SPIR-V lowering | 46 |
| type checker/type model | 35 |
| other `wyn-core` code | 107 |
| SSA | 18 |
| frontend | 4 |
| other crates | 5 |

The companion scan found 84 `unreachable!` calls, no `todo!` or
`unimplemented!` calls, and 46 production assertions. The largest
`unreachable!` clusters are EGIR fusion/expansion (25), other EGIR code (26),
TLC (16), SPIR-V lowering (7), and frontend/type code (9); one more is in
`wyn-staged-ir`. Assertions are tracked separately because some guard public
mutation APIs or process-exhaustion invariants, but assertion-based validation
of source-derived or pass-derived state should migrate with the same clusters.
`debug_assert!` is excluded because it is not a release-build panic path.

## Completed clusters

- Target parallelization: all 26 extraction sites were removed. Malformed
  callable/operand/result boundaries now return `ParallelizeError`. Scan
  lambdas and their callable bodies are stored together rather than in two
  vectors joined by a truncating `zip`. Reduction inputs are validated and
  retained together rather than validated and looked up again.
- WGSL text lowering: three non-formatting extraction sites were removed.
  Invalid identifiers, missing instruction results, and missing push-constant
  ABI slots are diagnostics. Formatting writes are intentionally deferred.
- Typed SPIR-V builder: 9 builder-state assertions now return `dr::Error`.
- Generic graph algorithms: 4 coupled-map lookups now use entry-oriented
  mutation; dominator intersection no longer extracts a presumed first item.
- Parser: seven locally checked parser shapes now return parse errors. Numeric
  literal conversion and unsupported hexadecimal floats remain in the lexer
  for a separate design pass.
- Resource/open resolution: missing bindings, feedback, modules, or scope
  frames now return compiler errors.

No-extraction Clippy gates now protect the completed planner, graph crate, and
typed SPIR-V builder scopes.

## Remaining remedies

The remaining calls should not receive one mechanical treatment.

### 1. Replace encoded type layouts with aggregate views

Clusters in `ssa/layout.rs`, `types/mod.rs`, `diags.rs`, `tlc/soa.rs`,
`tlc/monomorphize.rs`, and `spirv/types_lowering.rs` first test `is_array`,
`is_vec`, or `is_mat` and then independently extract element, size, variant,
and region fields. This is a tuple/SoA representation problem.

Introduce borrowed and owned shape records such as `ArrayType { elem, variant,
size, region }`, `VectorType { elem, size }`, and `MatrixType { elem, rows,
cols }`, returned by one pattern match. At the EGIR boundary, use the narrower
rank-1 array aggregate already called out in `typestate-invariant-backlog.md`.
Do not replace each field extraction with a separate `ok_or_else`; that keeps
partially valid type shapes representable.

### 2. Co-locate IDs with the data that makes them valid

Repeated symbol/function/node lookups dominate `tlc/from_ast.rs`,
`egir/from_tlc.rs`, `egir/elaborate.rs`, and several fusion/reification files.
The high-value changes are:

- make `SymbolTable` the only allocator of `SymbolId`;
- return arena-backed handles or aggregate records where an ID is always used
  with metadata from a parallel map;
- move per-call/per-result facts into records keyed once, instead of several
  maps whose key sets are expected to stay equal.

These changes remove both the reported extraction and neighboring indexing
operations. Adding an error to every internal lookup would merely turn a data
ownership bug into repetitive plumbing.

### 3. Make pass-specific variants structural

The TLC phase-only variants and post-pass assumptions listed in
`typestate-invariant-backlog.md` account for many `panic!`/`expect` arms in
`tlc/from_ast.rs`, pattern lowering, defunctionalization, and EGIR conversion.
Split phase-specific enums or add phase-typed wrappers so a pass cannot receive
variants that should have been eliminated earlier.

### 4. Propagate failures at validation and backend boundaries

Verifiers, source-driven lowering, builder calls, resource publication, and
external ABI construction should return contextual `CompilerError` or their
existing local error type. This applies especially to the remaining SPIR-V
lowering cluster and EGIR physical-call/reification boundaries. These are real
fallible boundaries; changing their signatures is preferable to asserting an
upstream phase ran correctly.

### 5. Retain only explicit process-exhaustion invariants

`wyn-base::IdSource` still has one checked-add assertion for exhausting the
32-bit compiler ID space. It is not source-recoverable with the current
infallible allocator API. Keep it explicit until allocator APIs return a
fallible allocation result; do not replace it with wrapping or saturation,
which would silently alias IDs.

## Next slices

1. Land the aggregate type-shape accessors and migrate the six type-layout
   clusters together.
2. Make `SymbolTable` own symbol allocation, then remove the associated TLC and
   TLC-to-EGIR lookup assertions.
3. Introduce aggregate call/result boundary records in EGIR elaboration and
   reification.
4. Split phase-scoped TLC variants, starting with array expressions.
5. Enable the no-extraction Clippy gate for each migrated module, and finally
   for `wyn-core` as a whole.
6. Design a WGSL emission sink that makes writes to an in-memory `String`
   structurally infallible at call sites. Avoid both per-line `unwrap()` and
   propagating `fmt::Error` through every lowering routine; migrate the
   formatting cluster only after that abstraction exists.
