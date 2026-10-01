# Radix port fidelity and compiler support — WIP

Status at 2026-09-30: **not ready to merge**. All six sorting entry points
compile in the probes described below, but the latest compiler test run has
two failures. GPU correctness and stability have not been established for all
six functions. Fusion is not yet demonstrated to match or exceed Futhark.

This records outstanding differences and current evidence, not historical
compiler failures that have already been fixed.

## Source and public API

The reference is `pkg/sort/upstream/radix_sort.fut`, from diku-dk/sorts v0.7.3,
commit `a473651d67d8ed0fafcea315f0226676721e2bdf`. The translation is
`pkg/sort/src/radix_sort.wyn`.

The translation preserves the two-bit digit, local pairwise helpers, repeated
bin expression, four-component scan, independent three-component totals
reduction, destination formula, scratch-annotated copy/scatter, counted loop,
by-key permutation/gather, and signed/float bit transformations. It does not
unroll the loop or obtain totals from the final scan element. Odd bit counts
still request both bits of the final digit, as upstream does.

Differences and limitations:

- Array sizes, indices, prefix ranks, bucket counts, and by-key permutation
  indices use Wyn i32 instead of Futhark i64. This is the agreed native-index
  adaptation; it restricts supported lengths/counts to Wyn's signed index
  range. It is not equivalent for arrays exceeding that range.
- Bin labels use i32 instead of upstream i8, omitting the explicit narrowing
  conversion. For valid get_bit callbacks returning 0 or 1, labels remain
  0–3 and the sorting computation is unchanged. This is an additional
  representation adaptation, with different storage cost; arbitrary invalid
  callback values need not behave identically.
- Upstream marks `radix_sort_step` and `by_key_wrapper` local. Wyn currently
  declares both as ordinary module definitions. `with_indices` is public in
  upstream too; the “six public functions” refers to the six sorting APIs.
- **The package manifest still selects the old `src/lib.wyn`.** That separate
  implementation uses 15 explicit passes, sorts only 30-bit keys, reads totals
  from the final prefix, uses replicate for scratch, and requires nonempty
  input. It is not the faithful port. Current port probes explicitly import
  `src/radix_sort`. Package integration remains unfinished.
- Presence of all six definitions is not proof of executable API parity.
  This WIP supplies integral num_bits and f32.get_bit. It does not establish
  parity for every numeric module or width: f16/f64 bit-introspection contracts
  and backend support still need auditing. WGSL does not support native f64.
- The source float transform matches upstream, but signed-zero, NaN payload,
  float ordering, and stable by-key behavior still require end-to-end evidence
  across the intended backends. Do not interpret source equivalence as a GPU
  validation result.

## Changes in this WIP

The original host-loop patch has been strengthened to publish explicit setup,
body, and completion stage identities instead of reconstructing loops from
shader names. Body stages follow dependency order. Compiler stage dependencies
are now carried into the host frame graph, including control edges that buffer
read/write analysis alone cannot reconstruct. This is needed to keep the final
by-key gather after the complete loop.

The dynamic-array work generalizes free representation variables for unsized
arrays, separates loop-carried representation from initializer representation,
and uses the inferred loop type during TLC conversion. Host-loop eligibility
now includes runtime-sized arrays and structure-of-arrays tuple state.

Host scalar expressions can obtain a whole input buffer's element count from
its byte size divided by element stride. Dynamic input length is not a separate
Rust argument: the entire supplied buffer is the logical array. Subranges and
spare capacity are not expressed by this API. Fixed-size array inputs now get
an exact byte-size check before command recording; both undersized and
oversized buffers return HostError::Invalid. That generated check was inspected,
but its runtime rejection paths have not yet been exercised.

Loop initialization is skipped when the logical carry length is zero. It is
still performed for nonempty arrays even when the iteration count is zero.
The direct dynamic integer sort passed an empty-input GPU check. This does not
yet establish empty-input support for by-key paths, which have additional
pre-loop and post-loop kernels.

## Validation at the WIP checkpoint

No new repository tests were added. Existing interface fixtures received empty
dependency lists for the new metadata field. Temporary compile and GPU probes
were kept outside the repository.

- Earlier explicit-host-loop baseline: 1,319 core tests and 5 host tests passed;
  the 25-million-i32 (100 MB) GPU output matched CPU sorting.
- Dynamic direct integer sort: nine nonempty input lengths passed GPU comparison
  against CPU sorting, including 25 million elements and reuse of the same
  context for a smaller input afterward. Empty input subsequently passed with
  the initialization guard. These runs preceded the final by-key support changes.
- All six sorting functions compiled to WGSL plus Rust/WGPU with fixed `[8]`
  and dynamic `[]` arguments. Integer callbacks used i32.get_bit; float callbacks
  used f32.get_bit. A separate i32.num_bits probe also compiled.
- A temporary harness for executing all six generated dynamic sorts was built,
  but was **not run** before this checkpoint. By-key stability and float runtime
  correctness are therefore not validated by that harness.
- **Latest core suite: 1,317 passed, 2 failed, 14 ignored.** The failed tests are
  `egglog::to_ssa::tests::loop_local_tuple_collectives_preserve_component_arrays`
  and `egglog::to_ssa::tests::nested_tuple_loop_state_preserves_component_arrays`.
  Both report `invalid projection 0` on an array-of-tuples storage view. Broadening
  host-loop eligibility exposes a mismatch between packed loop storage and
  component-array projections. Fix the representation boundary; do not skip
  these tests or exclude tuple carries to hide the failure.
- The latest combined test command stopped on the core failures, so it did not
  provide a fresh host-suite result. Full gates and validate_testfiles have not
  been rerun for this checkpoint. Earlier gate results do not validate this WIP.

Useful commands (run from the repository):

```sh
cargo test -p wyn-core -p wyn-host --lib
cargo fmt --all -- --check
git diff --check
```

To reproduce compilation, import `pkg/sort/src/radix_sort` from a temporary
entry file and instantiate each of radix_sort, radix_sort_by_key,
radix_sort_int, radix_sort_int_by_key, radix_sort_float, and
radix_sort_float_by_key. Compile using:

```sh
wyn build -O --target wgsl --target-double rust-wgpu probe.wyn -o probe.wgsl
```

Use both fixed and dynamic input types. Identity keys establish compilation,
not stability: runtime checks must also use distinct payloads with equal keys.

## Fusion comparison: not yet equal or better

Reference: the earlier inspected Futhark 0.27.1 GPU IR and backend code recorded
in [radix-fusion-wip.md](radix-fusion-wip.md). The current check inspected newly
generated WGSL/Rust for the dynamic direct i32 sort and the saved Futhark GPU
IR. Futhark was not rebuilt or benchmarked during this audit.

| Property | Current Wyn | Recorded Futhark |
|---|---|---|
| Bin map plus totals reduction | One partial-reduction traversal emits bins and totals | Shared traversal emits bins and totals |
| Reuse of emitted bins | **Missing:** scan and final scatter recompute bins | Scan reads the retained bins |
| Scan prefix adjustment, index calculation, scatter | Fused final kernel; no separate destination-index array | Fused scan output action |
| Scratch initialization per radix pass | No separate initialization launch | Eliminated |
| Reduction | Two launches: partials and combine | One reduction launch, then a scalar-copy launch |
| Scan | Three launches; full-length partial-prefix storage | OpenCL: three launches and full-length prefixes; CUDA/HIP: status clear plus lookback scan |
| Launches per nonempty 32-bit pass | 5 | OpenCL: 5; CUDA/HIP: 4 |

The current five-loop-kernel sequence is partial reduction, reduction combine,
partial scan, scan carry combine, and adjusted-prefix scatter. There is also
one initialization copy outside the loop. The publication-only completion
stage is not dispatched. Counts exclude uploads, readback, scalar uploads,
and context/cache management.

A concrete missed sharing opportunity is visible in generated WGSL: the
reduction writes the bin-label buffer, but that buffer has no reads. Both the
scan and scatter instead read the carried keys and repeat digit extraction.
That is dead full-array storage traffic plus repeated computation, not evidence
of better fusion. Fix the producer/consumer routing, then re-inspect the actual
shader loads/stores to confirm the array is either reused or eliminated under
an explicitly justified plan.

Equal launch counts do not imply equivalent fusion, memory traffic, or speed.
Wyn also still retains full-length scan-prefix storage, unlike Futhark's
CUDA/HIP lookback path. The i32/i64 counter-width difference further prevents
raw byte counts from being attributed solely to fusion quality.

Before merging: fix both tuple-loop regressions; validate all six APIs and
stable by-key payloads (including empty, boundary-sized, signed and float
cases); audit the remaining numeric contracts and package entry; rerun full
gates; and compare fusion and intermediate traffic for the validated variants.
No claim of “same or better than Futhark” is justified at this checkpoint.
