# Ranked iteration and bucket scatter plan

Status: active follow-up plan

Last updated: 2026-08-11

Implementation baseline: `492a1b59` (`Add rank-aware bucket scatter lowering`)

## Objective

Make ranked iteration a first-class compiler concept while keeping
`bucket_scatter_1d` through `bucket_scatter_4d` as thin temporary frontend
entry points over one ranked backend operation.

The immediate Equihash boundary is usable after `492a1b59`: generated ranked
items can feed bucket insertion without materializing the complete candidate
array, storage-bound ranked tuple arrays compile as AoS views, negative keys
are discardable, and fixed rank-3 domains dispatch over WebGPU x/y/z.

This is not the final ranked-producer design. The current implementation uses
ranked metadata on the existing TLC and EGIR SOAC representations. A reusable
producer abstraction, general dispatch topology, and `expand`/`flat_map`
remain follow-up work.

## Compiler invariants

1. Logical rank is explicit compiler data. It must not be recovered by peeling
   a post-fusion operand type.
2. Logical domain dimensions remain separate through dispatch planning.
3. Generated arrays remain generators until a consumer explicitly requires
   materialization.
4. A consumer that requires fused generation must either consume the generator
   directly or fail with a clear diagnostic. It must not silently allocate an
   impractically large intermediate.
5. Logical array shape and physical layout are independent. Storage AoS,
   generated/local SoA, and future strided field views must preserve the same
   ranked semantics.
6. An inactive emission touches neither bucket counts nor overflow state.
7. Size products use checked wide arithmetic during planning. Conversion to
   target-sized values happens only after target limits are validated.

## Landed baseline

### Unified frontend and TLC operation

- The type checker exposes `bucket_scatter_1d` through
  `bucket_scatter_4d` with ranked item types.
- All four names lower to one `SoacOp::BucketScatter` carrying a domain rank,
  an envelope lambda, inputs, and input-to-domain dimension mappings.
- There is no duplicated rank-specific backend lowering.

Relevant code:

- `wyn-core/src/types/checker.rs`
- `wyn-core/src/tlc/from_ast.rs`
- `wyn-core/src/tlc/mod.rs`

### Ranked iteration metadata

- EGIR `SoacInputType` now carries logical dimension mappings.
- `SegSpace` retains a vector of extents instead of one prematurely flattened
  length.
- Reification derives each logical extent independently and scheduling can map
  those extents onto multiple physical dispatch axes.

This is the current functional equivalent of part of the proposed
`RankedProducer` and `CoordinateMap` model. It deliberately reuses the existing
SOAC lambda/input representation; it is not yet a standalone producer IR.

Relevant code:

- `wyn-core/src/egir/ir.rs`
- `wyn-core/src/egir/reify.rs`
- `wyn-core/src/egir/types.rs`

### Generated ranked items

- A rectangular nest of maps is composed into the bucket insertion envelope.
- Each map input records which logical coordinate indexes it.
- The generated leaf computation runs in the insertion shader, so the tested
  Equihash-shaped candidate domain has no materialized candidate array.

Current limitation: map-array operands at an inner level cannot themselves
depend on an outer map parameter. If that composition is needed, the current
fusion pattern declines it instead of representing the dependency as a
general coordinate map.

Relevant code:

- `wyn-core/src/tlc/runtime_index_producers.rs`
- `wyn-core/src/tlc/runtime_index_producers_tests.rs`

### Storage-bound ranked arrays

- Fixed arrays that are too large for portable push constants, and consuming
  fixed-array parameters, receive storage bindings.
- Storage-bound arrays of tuples retain their physical AoS representation.
- Ranked reads use explicit coordinates and recognize a storage-view producer
  even when the surface array type remains composite.

Relevant code:

- `wyn-core/src/binding_layout.rs`
- `wyn-core/src/tlc/soa.rs`
- `wyn-core/src/egir/soac_expand/array_io.rs`

### Bucket insertion semantics

The dedicated insertion lowering implements:

```text
key < 0
    discard without touching counts or overflow

0 <= key < bucket_count
    slot = atomic_add(counts[key], 1)
    if slot < capacity
        write destination[key][slot]
    else
        set overflow

key >= bucket_count
    set overflow without indexing counts
```

Counts therefore report the total population of each valid bucket, including
items beyond capacity. Ordering within a bucket remains unspecified.

The negative-key guard currently lives in the purpose-built bucket insertion
lowering. It has not yet been promoted to a general guarded-emission IR.

Relevant code:

- `wyn-core/src/egir/soac/hist.rs`
- `wyn-core/src/egir/soac_expand/hist_lowering.rs`

### Multidimensional dispatch

- Fixed ranked domains retain their dimensions until scheduling.
- The innermost dimension maps to x, the next dimension maps to y, and any
  remaining outer prefix maps to z.
- Shader reads use the resulting logical coordinate vector directly rather
  than reconstructing the full domain through one signed flattened index.
- `global_invocation_id.y` and `.z` are supported consistently by WGSL and
  SPIR-V lowering.

The regression domain `[4096][658][2016]` publishes workgroups
`{x: 32, y: 658, z: 4096}` for local size `(64, 1, 1)`. Its total logical
population exceeds both the old x-only dispatch limit and a 32-bit flattened
index.

Current limitation: the topology is a fixed ranked policy, not a general
partitioner. It does not yet support resource-derived y/z expressions or a
grid-stride tile quotient when no direct partition fits WebGPU limits.

Relevant code:

- `wyn-core/src/egir/parallelize/hist.rs`
- `wyn-core/src/egir/parallelize/schedule/mod.rs`
- `wyn-core/src/egir/soac_expand/hist_lowering.rs`
- `wyn-core/src/wgsl/ssa_lowering.rs`
- `wyn-core/src/spirv/lower_builtin.rs`

### Runtime contract and descriptors

- Bucket scatter is emitted as init, insert, and finish stages.
- Destination storage is returned by alias instead of copied to a second huge
  output buffer.
- Counts and overflow are routed directly to output resources.
- Scalar outputs now publish their fixed byte length, so overflow is described
  as a four-byte output rather than `length: null`.
- `--single-stage` reports that bucket scatter requires its pipeline instead
  of reaching a backend panic.

Relevant code:

- `wyn-core/src/egir/realize_outputs/mod.rs`
- `wyn-core/src/egir/realize_outputs/dispatch.rs`
- `wyn-core/src/egir/from_tlc.rs`

## Acceptance status

| Gate | Status | Evidence or remaining work |
| --- | --- | --- |
| Literal rank-2 input compiles | Complete | Existing literal path remains supported. |
| Map-produced rank-2 input compiles without panic | Complete | Ranked TLC fusion regression. |
| Storage-bound rank-2 AoS input compiles without panic | Complete | WGSL/Naga and SPIR-V regression. |
| Generated computation is in the insertion shader | Complete | Three-stage WGSL integration regression. |
| No collision-candidate storage is materialized | Complete for supported rectangular map nests | Add a direct descriptor assertion as hardening. |
| Negative keys touch neither counts nor overflow | Implemented structurally | Add an executable GPU or IR-interpreter semantic test. |
| Capacity overflow still increments total count and sets overflow | Implemented structurally | Add an executable semantic test. |
| Nonnegative invalid key sets overflow safely | Implemented structurally | Add an executable semantic test. |
| Rank-3 dispatch uses x/y/z | Complete | Descriptor and generated-WGSL assertions. |
| Total domain beyond old x-only limit compiles | Complete | `[4096][658][2016]` regression. |
| Rank-1 behavior remains accepted | Complete | Rank-1 SPIR-V regression. |
| `_1d` through `_4d` share one backend lowering | Complete | One ranked TLC/EGIR operation. |
| Literal, generated, and bound inputs produce identical runtime results | Pending | Requires an executable backend comparison test. |

The committed baseline passes the full Rust workspace suite with
`RUST_MIN_STACK=16777216`, validates all 88 active SPIR-V testfiles, and
validates 87 WGSL testfiles with one documented linked-helper skip.

## Remaining implementation plan

### Phase 1: Semantic hardening and diagnostics

1. Add executable tests for discard, valid insertion, capacity overflow,
   invalid-key overflow, and total counts.
2. Add a direct descriptor/resource assertion proving that a fused generated
   candidate stream has no materialized candidate buffer.
3. Replace ranked-shape and layout `panic!`/`expect` paths reachable from source
   programs with source diagnostics.
4. Introduce an internal guarded emission form, conceptually:

   ```text
   HistEmission {
       active,
       key,
       value,
   }
   ```

   Lower today's negative-key rule to `active = key >= 0`. Keep atomic counter
   and overflow operations strictly inside the active path.

Exit gate: all bucket semantics are verified by execution, and malformed
ranked inputs fail diagnostically rather than panicking.

### Phase 2: General ranked producer consumption

1. Decide whether to introduce a standalone `RankedProducer` or to promote the
   current TLC envelope plus dimension mappings into an equivalent shared IR.
2. Represent coordinate-dependent producer inputs explicitly, including inner
   map-array operands derived from outer logical coordinates.
3. Canonicalize rectangular nested maps into that representation independently
   of bucket scatter.
4. Make direct generator consumption a required property for consumers that
   declare it. If composition is impossible, emit a clear diagnostic rather
   than materializing the candidate stream.
5. Reuse the representation for histogram, map, reduce, and future stream
   consumers instead of adding new bucket-specific fusion patterns.

Exit gate: supported ranked generators compose through consumer boundaries
without consumer-specific rank peeling or candidate allocation.

### Phase 3: General dispatch topology

Introduce a target-aware `DispatchTopology` that maps a logical `SegSpace` to
at most three physical axes.

1. Keep the innermost/high-volume dimension on x when profitable.
2. Partition contiguous logical dimensions across x/y/z.
3. Decode grouped logical dimensions in the shader.
4. Use checked `u64` products during planning.
5. Support fixed and resource-derived axis expressions.
6. When no direct partition fits WebGPU's per-axis workgroup limits, use a
   grid-stride tile quotient instead of reverting to whole-domain x flattening.
7. Extend the descriptor only when dynamic y/z expressions require it; the
   existing fixed `{x, y, z}` form remains sufficient for current rank-3 work.

Exit gate: arbitrary supported ranks dispatch legally or fail with a precise
target-limit diagnostic; no path silently overflows or returns to x-only
whole-domain flattening.

### Phase 4: First-class physical array layout

1. Introduce explicit physical layout metadata equivalent to
   `ArrayLayout::{Aos, Soa, Generated, StridedFields}`.
2. Keep layout selection independent of logical rank and producer fusion.
3. Model strided field views before splitting storage AoS tuples into component
   arrays.
4. Add parity tests for AoS storage, SoA generated values, and mixed captures.

Exit gate: ranked indexing is expressed once and parameterized by layout;
storage representation is never inferred from a transformed logical type.

### Phase 5: Rank-polymorphic surface typing

Replace the temporary `_1d` through `_4d` names with one rank-polymorphic
surface operation when Wyn's type system can quantify over rank. Keep the
existing names as compatibility wrappers during migration. No backend change
should be required.

### Phase 6: Variable-cardinality candidate generation

Design an `expand`/`flat_map`-style SOAC for later Equihash rounds, where each
source bucket produces a variable number of collision rows. Its output should
compose with the same guarded ranked producer and bucket insertion machinery.

This phase is intentionally separate: bucket scatter solves destination
bucketing once candidate rows exist; it does not itself construct a
variable-sized candidate stream.

## Scope boundaries

- Do not expose general source-level atomics.
- Do not promise stable ordering within a bucket.
- Do not treat bucket scatter alone as completion of the Equihash solver.
- Do not broaden the frontend API beyond `_1d` through `_4d` until true rank
  polymorphism is available.
- Prefer extensions to shared ranked producer, layout, and dispatch machinery
  over new rank-specific lowering branches.
