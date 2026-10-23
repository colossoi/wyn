# SSA simplification results

All seven candidates from the October 23, 2026 analysis have been implemented.
The cleanup removes **409 net production Rust lines** (+383 / -792), excluding
test files, fixtures, documentation files, and pre-existing changes. The count includes only
the three-line adjustment to the already-modified WASM inspector, not its other
changes.

| Removed or simplified representation | Required capability and resulting representation |
| --- | --- |
| `OpTag::Tuple(usize)`, `Vector(usize)`, `ArrayLit(usize)` | Construction needs operands and a result type. These are now unit tags; constructors no longer compute or copy counts. Their distinct tags retain the storage/ownership distinctions used by SSA passes. |
| `OpTag::Matrix { rows, cols }` | No production constructor existed. Removed the entire tag, mapping, printing, classification, and backend branches. Matrix types and operations produced through the existing vector/intrinsic paths remain supported. |
| `EntryInput.size_hint` | No lowering or publication consumer existed. Removed the SSA field and extraction. The source-language `#[size_hint(...)]` attribute remains available in TLC parameter metadata throughout egglog optimization, then is omitted when constructing SSA. It currently does not affect planning. |
| `ValueInfo.name` | SSA identity suffices for function and block parameters. Removed all value names, named block-builder APIs, generated `arg{i}` strings, and WGSL name-uniquification machinery. WGSL helper parameters use SSA IDs; entry interface names still supply host binding aliases. |
| Publication-only entry metadata | Buffer lengths, texture backing/resource identity, image allocation size/resource identity, and generated-storage identity/capacity are needed by the host descriptor. They now live in publication-only payloads that become zero-sized `()` in returned shader SSA. Removed the old getters and the intermediate texture-source enum. |
| `EntryOutputDestination::Target(String)` | Publication needs the target name; shaders need its location. Names stay in publication metadata and the shader interface carries a numeric location. The WASM SSA inspector consequently reports `slot(n)`. |
| `InstKind::Alloca.elem_ty` | Typed allocation uses `FuncBody.places[result].elem_ty`. Removed the duplicate field, producer clones, and backend reads of the duplicate. Local-storage analysis now obtains/clones the type only when creating a new place. |
| `Program.constants`, `Constant`, `GlobalId`, `OpTag::Global` | The production SSA producer always supplied an empty constant table. Removed the definition/reference path, constant reachability, traversal loops, SPIR-V declarations/emission, and WGSL lookup. Reachability follows function calls only. |
| Record labels and existential binders in backend types | Shader representation is positional. Program construction recursively converts records to tuples and removes existential wrappers after host publication, covering parameters, results, instruction values, places, inputs, outputs, and generated storage types. Backend record-specific branches were removed or converted to tuple handling. |

The constant removal also eliminates fallback function-emission maps and the
module-wide access union used for functions reached through constant bodies.
Every emitted helper now has a concrete entry context. Stage-specific and
pipeline-wide storage accesses remain separate because they have distinct
backend uses.

Generated storage publication now requires a resource name and capacity in its
type, removing impossible missing-value branches. Cross-stage identity/capacity
consistency checks remain. An unused publication renaming method was also removed.

The shared frontend `TypeName` enum remains necessary for source checking and
host layout publication. Backend program construction erases its source metadata
once; proof-state transitions do not repeat that work. Record-update validation
and emission use tuple shape and field types. Host result labels and uniform
member names are published before erasure. The active
`Program.addressable_constants` mechanism remains available for addressable
aggregate constants.

Source locations:

- [Operation vocabulary](../wyn-core/src/op.rs), [SSA values](../wyn-core/src/ssa/ir/mod.rs), and [SSA builder](../wyn-core/src/ssa/builder.rs).
- [Program and type boundary](../wyn-core/src/ssa/types.rs), [entry interface](../wyn-core/src/interface.rs), [ABI production](../wyn-core/src/egglog/abi.rs), and [host publication](../wyn-core/src/interface/publish.rs).
- [Function reachability](../wyn-core/src/ssa/reachability.rs), [function emission selection](../wyn-core/src/ssa/storage_function_variants.rs), [SPIR-V lowering](../wyn-core/src/spirv/lower.rs), and [WGSL lowering](../wyn-core/src/wgsl/ssa_lowering.rs).

The source-language `#[size_hint(...)]` syntax and positive-integer validation
are preserved. Direct SSA callers must use the smaller constructor/parameter APIs; helper
parameter spellings and some SSA diagnostic type spellings change. Host resource
names and result-field names remain part of the published ABI.

The size-hint regression verifies that the attribute survives the complete
egglog pipeline for reductions, scans, and generated graphics stages, and that
each example lowers to WGSL and SPIR-V. Graphics capture extraction now checks
for a resource binding rather than an empty attribute list, so a hint cannot
suppress automatic binding assignment.

The commit candidate was validated in an isolated checkout containing only the
SSA cleanup; pre-existing local inspection changes were excluded. All gates passed:

- Workspace and WASM formatting checks.
- `cargo build --workspace --all-targets`.
- `cargo test --workspace`: 1,663 passed, 16 ignored, including 1,513 core tests.
- `cargo test --manifest-path wyn-wasm/Cargo.toml --locked`: 3 passed.
- `scripts/check_wasm.ps1`: locked `wasm32-unknown-unknown` build check passed.
- `cargo test --manifest-path extra/viz/Cargo.toml --locked`: 10 passed.
- Tracked SPIR-V testfile validation: 123 passed.
- Tracked WGSL testfile validation: 122 passed; the script intentionally skips
  `miner.wyn`, which depends on linked SPIR-V helpers.
- `git diff --cached --check`.

Windows test threads used `RUST_MIN_STACK=16777216` for the planning suite.
