# Plan: materialize randomly-indexed computed arrays into a buffer

> **Status (gather-materialize branch):** Phases 1–3 (Map gather + real buffer
> sizing + scan producers, with the forced-binding channel unified) **landed**.
> **Phase 1:** `tlc/lift_gathers.rs` + the `lift_gathers` typestate step
> (between `apply_ownership` and `defunctionalize`) detect `let arr = map(f,
> src) in …arr[i]`, split the producer into a `<entry>_gather_<n>` compute
> pre-pass writing a forced storage binding, and rewrite the consumer's
> `arr[i]` to a `storage_index` load. `parallelize::make_map_plan` honors the
> pre-pass's Output decl to pin its result buffer.
> **Phase 2:** the gather buffer is now a `BufferUsage::Intermediate`
> (compiler-managed scratch, no host file) carrying a `BufferLen` sizing policy
> (`pipeline_descriptor`). `lift_gathers` emits `BufferLen::LikeInput{src, …}`
> (a map preserves element count); `enrich_pipeline_with_auto_bindings` surfaces
> any length-bearing storage binding as an Intermediate (access from role:
> producer writes, consumer reads); `viz/gpu.rs` sizes it from the policy
> (two-pass: inputs first, then resolve `LikeInput`), retiring the 1024 default
> for it. Verified at the descriptor level (spirv-val + naga + binding/length
> tests); GPU runtime sizing is in place but not exercised here (no adapter).
> Covered by `lift_gathers_tests.rs`, `integration_tests.rs`,
> `wgsl/ssa_lowering_tests.rs`, and `BufferLen` tests in the descriptor crate.
>
> **Phase 3:** scan producers + a unification of the forced-output channel.
> *Unification (done first):* the per-SOAC forced-result-binding paths collapsed
> to one channel — `parallelize::run` folds gather pre-passes' `Output` decls
> into the same `prepass_result_bindings` map the scalar lift uses, and
> `PlannedBindings::forced_output()` lets `from_tlc` honor a pinned output for
> Map *and* Scan uniformly (removed the Map-specific `forced_output_binding_from_decl`).
> *Scan:* `lift_gathers` accepts a `scan` producer; `make_scan_plan` honors the
> forced output (symmetric with `make_two_phase_plan`) and places its
> block-sum/offset intermediates above it. Verified: `let o = scan(op,ne,xs) in
> …o[i]` over an input array compiles + validates (spirv-val + naga).
>
> **Fused map→scan lowering (fixed):** `SoacOp::Scan` / `PendingSoac::Scan`
> gained a pure `reduce_func` (mirroring `Redomap`), threaded through every TLC
> SOAC-reconstruction pass and the egir scan synthesis. The parallel scan uses
> the fused element-step (`op`/`func`) for phase 1 and the pure combiner
> (`reduce_op`/`reduce_func`) for phases 2 & 3 (which merge already-transformed
> block sums). A *type-preserving* fused scan (`scan(op, ne, map(g, xs))`,
> `g: T -> T`) now compiles and validates (spirv-val + naga), as does a gather
> over one. The gather lift admits these (its `scan_input_is_direct` gate keys
> on input-vs-result element type).
>
> **Known gaps:**
> - A runtime-sized *output* buffer (e.g. the consumer's `iota(6144)` result)
>   still uses viz's 1024 default — pre-existing, affects any runtime-sized
>   compute output, not gather-specific.
> - A *type-changing* fused scan (`g: A -> T`, e.g. the original chained repro
>   `let counts = map(|h:vec4f32| .. : i32) in scan(.., counts)`) still
>   mis-lowers: the scan's phase 1 access-chains the raw input with the
>   accumulator's element type (an `OpAccessChain` int-vs-vector mismatch). This
>   is a separate scan-lowering gap (a standalone such scan also fails), so the
>   gather lift declines these and leaves the diagnostic. Fixing it means
>   distinguishing the scan's *input* element type from its *accumulator*
>   element type in phase 1.
>
> **Next:** type-changing fused-scan lowering (unblocks the chained repro);
> Phase 4 (gather CSE).

## Context

Indexing a **computed** array (a `map`/`scan` result) at a runtime index from
inside another `map` fails to compile:

```wyn
entry gen(bh: []vec4f32) []i32 =
  let counts = map(|h:vec4f32| 4 + 5*(if h.x>4.0 then 3 else 1), bh) in
  map(|i:i32| counts[i % 256], iota(6144))        -- gather from a computed array
```
→ `spirv/mod.rs:374` "Composite variant unsized arrays not supported".

```wyn
  let offsets = scan(|a,b| a+b, 0, counts) in
  map(|i:i32| offsets[i % 256], iota(6144))
```
→ `egir/elaborate.rs:314` "unexpanded PendingSoac in skeleton: Scan{Fresh}".

Indexing an **input** array (`vals: []T`, `vals[i % 256]`) works because inputs
have a `StorageView` over a real buffer. A computed array lowers to an
in-register loop building a runtime-sized **Composite** array (MIR: `%2: [?]i32`,
loop-carried + captured by the consumer lambda) — no SPIR-V value
representation, so it can't be randomly indexed. Random indexing isn't pointwise
(can't fuse), and the consumer is parallelized (one output element per thread),
so the producer must run as its **own stage** that fills a storage buffer first;
the consumer then reads that buffer by index.

Intended outcome: both repros compile and validate on SPIR-V + WGSL, with the
computed array materialized into a compiler-allocated intermediate buffer.

## Key reuse: the documented array-result follow-up to the scalar lift

`tlc/parallelize.rs::lift_graphical_invariant_soacs` + `maybe_hoist` already do
this shape for **scalar** SOAC results: allocate an intermediate storage
binding, emit a `<entry>_prepass_<n>` compute stage that evaluates the SOAC and
stores its result, add an `Input` `StorageBindingDecl` to the consumer, and
rewrite the binding RHS to `_w_intrinsic_storage_index(set, binding, 0)`. Its
own docstring (`maybe_hoist`, ~L881) names the array case as the explicit TODO:
"extend to array-result SOACs (Scan, Map) … write N slots to storage, read back
by index instead of at position 0." This plan implements that follow-up, plus a
new trigger (random-indexed computed array in a **compute** entry; the scalar
lift triggers on entry-invariance in **graphical** entries).

## Verified facts

- `TermKind::Index { array, index }` (`tlc/mod.rs:288`) is the indexing node —
  the rewrite site. After `defunctionalize` the computed array becomes a
  `SoacBody.capture` `(SymbolId, Type, Term)` on the consumer SOAC, so the lift
  must run **before defunc** while `arr` is still a plain `Var` and the producer
  is a recognizable `Soac`.
- `egir/from_tlc.rs:897` already lowers `Index` on a **View**-variant array to
  `ViewIndex + Load` (buffer-backed). So once `arr` reads from a storage view,
  the gather lowers correctly with **no Index/materialize changes**. (Non-view
  arrays go to `PureOp::Index` → materialize → the runtime-sized panic.)
- `interface::StorageBindingDecl = { set, binding, role, elem_ty }` has **no
  size field**. Intermediate element count is decided at runtime:
  `viz/gpu.rs::create_binding_buffers` hardcodes `1024*64 = 65536` for
  intermediates; `tephra/miner.rs` sizes `partials` to `total_threads`. Sizing
  the gather buffer to its true length is the one genuinely new runtime piece.
- `wyn_pipeline_descriptor::{MultiComputePipeline, ComputeStage}` carry stages +
  shared `bindings` + `reads`/`writes`; Stage B already builds these for the
  scalar lift's pre-pass.

## Design

### 1. Detection — `lift_indexed_compute_arrays` (new, pre-defunc pass)
Walk each compute entry's outer let-chain; `let arr = rhs in body` is a lift site
iff `rhs` is a **`Soac(Map)`** producing a runtime-sized array (Phase 1 — Scan /
Redomap deferred to Phase 3) **and** `arr` is consumed by a dynamic
`Index { array: Var(arr), index }` in `body` (directly or inside a SOAC lambda
body) with a non-constant index. Conservative: require the simple shape `let arr
= map(...) in … arr[i]`; skip pointwise-only uses (already fused) and the
entry-output case (already streams to OutputView). Reuse `collect_free_vars`; add
an `Index`-use walk.

### 2. Materialize the producer (reuse/generalize `maybe_hoist`/`build_prepass_def`)
Per lift site: allocate `binding = (AUTO_STORAGE_SET, next_binding++)`,
`elem_ty = arr` element type; build a `<entry>_gather_<n>` compute entry whose
body is `rhs` (the map) returning the array, bound to `binding` (the normal
"compute entry whose result is a runtime array" path; `assign_outputs` streams
the map into its output view). Record `prepass_result_bindings[gather_def.name] =
binding`.

### 3. Rewrite the consumer = `Index → storage_index`
Rewrite each `Index{Var(arr), idx}` in `body` to
`_w_intrinsic_storage_index(set, binding, idx)` (exactly what `buffer_specialize`
emits for a view-param index, and what the scalar lift does with `idx=0`), then
drop the now-dead `let arr`. Add an `Input`-role `StorageBindingDecl` to the
consumer entry's interface. (Do *not* put `ArrayExpr::StorageBuffer` — SOAC-input
syntax — at a `let`-RHS Term position.)

### 4. Stage ordering
The gather pre-pass + rewritten consumer form a `MultiCompute` pipeline (gather
first, writing `binding`; consumer reads it), picked up by `analyze_program` +
Stage B exactly as the scalar lift's pre-pass is today (via
`prepass_result_bindings` → `forced` in `parallelize::run`).

### 5. Runtime intermediate sizing (new)
The intermediate holds N = the producer's output length (here `length(bh)`), not
65536. Add `StorageBindingDecl.elem_count: Option<BufferLen>` (`Fixed(u32)` |
`SameAsInput(binding)`), emit it from the lift, and size the buffer from it in
`viz/gpu.rs` + `tephra`. This also retires the existing 65536 hack. (MVP fallback
if deferring: size intermediates to the largest input length — valid when N ≤ an
input length.)

## Refinements found during implementation setup

- **Pipeline position: a new pass *before* `defunctionalize`.** The analogous
  `lift_graphical_invariant_soacs` runs inside `parallelize` (post-defunc,
  post-`buffer_specialize`), where `arr` has become a closure capture and inputs
  are already storage-specialized — awkward to rewrite. Run the gather lift as a
  new typestate step between `apply_ownership` and `defunctionalize` (lib.rs
  chain `fuse_maps → apply_ownership → defunctionalize`), where `arr` is a plain
  `Var`, the producer is a clean `Soac(Map)`, and the input `bh` is still an
  unspecialized `[]T` param. `buffer_specialize` then lowers the rewritten
  consumer + the pre-pass uniformly downstream.
- **Consumer rewrite = `Index → storage_index` (confirmed).** Per the IR-layer
  correction: don't keep a view value. Rewrite `Index{Var(arr), idx}` to
  `_w_intrinsic_storage_index(set, binding, idx)`, then drop the dead `let arr`.
  Phase 1 conservative: only when `arr` is used *solely* via `Index`.
- **Pre-pass captures the producer's input arrays, not just uniforms.** The
  scalar lift captures uniforms (scalars); the gather producer `map(f, bh)` reads
  an input array `bh`, so the `<entry>_gather_<n>` pre-pass must re-declare `bh`
  as its own input param at the same storage binding (extend
  `build_prepass_def`/`make_entry_def` capture to input-view free vars).

## Staging

Land this incrementally; each phase is independently shippable.

- **Phase 1 — Map-only gather.** Conservative detection (`let arr = map(...) in
  …arr[i]`), the producer pre-pass (§2), the consumer rewrite (§3), and Stage-B
  wiring (§4). Use the **existing oversized intermediate default** (65536) rather
  than blocking on sizing — correct as long as the producer output ≤ that
  default. Proves the prepass + pipeline end-to-end and fixes the first repro.
- **Phase 2 — real `BufferLen` plumbing (§5).** Replace the 65536 hack with an
  explicit `elem_count` on the binding, threaded through descriptor + runtime.
- **Phase 3 — Scan / Redomap producers.** A parallelized `scan` may itself
  expand into multiple stages before the gather consumer — sequence it after the
  map path is proven.
- **Phase 4 — CSE / shared gather buffers.** Coalesce multiple gathers over the
  same computed array into one intermediate (optimization; one-buffer-per-array
  is already correct).

## Files

- `wyn-core/src/lib.rs` — new typestate step (between `apply_ownership` and
  `defunctionalize`) calling the gather lift.
- `wyn-core/src/tlc/lift_gathers.rs` (new) — `lift_indexed_compute_arrays`
  driver, the producer pre-pass builder (reusing `build_prepass_def`), the
  consumer `Index→storage_index` rewrite, and the dynamic-`Index`-use detection.
- `wyn-core/src/interface.rs` + `wyn-pipeline-descriptor/src/lib.rs` —
  `elem_count` / `BufferLen` on the storage binding (Phase 2).
- `extra/viz/src/gpu.rs`, `extra/tephra/src/miner.rs` — size intermediate
  buffers from `elem_count` (Phase 2).
- `wyn-core/src/integration_tests.rs` + `wyn-core/src/wgsl/ssa_lowering_tests.rs`
  — both repros compile (SPIR-V + WGSL) and a value-correctness check.

## Risks

- **Detection precision** — lift only genuine random gather, not pointwise
  fusion or const-index. Over-lift = needless buffer+stage; under-lift = panic.
- **Scan producer** composes with its own parallel-scan phase1/2/3 — land the
  **map** producer first, then scan.
- **Runtime sizing** — buffer must be ≥ N or the gather reads OOB.
- **Pass ordering** — must run before `defunctionalize`.
- **Two distinct lengths** — consumer dispatch length (`iota(6144)`) is
  independent of the gathered buffer length (`length(bh)`); keep them separate.

## Verification

- Both repros compile to SPIR-V (`spirv-val`) and WGSL (`naga`).
- A value test: gather a known computed array, assert output == CPU reference.
- `cargo test --lib -p wyn-core` green; `bash scripts/validate_testfiles.sh`
  (+ `--wgsl`) no regressions.
- If a GPU is available, a multi-stage tephra run with the sized intermediate.

## Out of scope

- Scatter / write-indexed computed arrays.
- Const-index into a runtime-sized computed array (rare; fixed-size already
  handled by materialize + DynamicExtract locally).
- CSE of multiple gathers over the same array into one buffer (later optimization;
  one buffer per lifted array is correct).
