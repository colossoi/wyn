# Radix port fidelity and compiler support — WIP

Status at 2026-10-01: **still WIP**. The tuple-loop regressions are fixed, and
all six sorting APIs pass the GPU cases below through generated Rust/WGPU and
WGSL on an RX 580. Numeric-width coverage and package integration remain
incomplete. Fusion is not yet demonstrated to match or exceed Futhark.

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
- Signed-zero, NaN payload, float bit-pattern ordering, and stable by-key
  behavior are verified for the f32 WGSL cases below. Other numeric widths,
  devices, and backend paths remain outside that GPU evidence.

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

Tuple-array output publication resolves each logical component to its physical
array and element projection. The output-copy loop loads an element before
projecting its fields; it does not rebuild temporary component arrays. Already
separate component arrays keep their own lengths. The host executes completion
stages that write outputs after the loop; a completion with no writes remains
metadata only. WGSL parameter legalization also covers loop bounds and initial
lengths, so scalar loop controls reach generated host code.

Host scalar expressions can obtain a whole input buffer's element count from
its byte size divided by element stride. Dynamic input length is not a separate
Rust argument: the entire supplied buffer is the logical array. Subranges and
spare capacity are not expressed by this API. Fixed-size array inputs now get
an exact byte-size check before command recording; both undersized and
oversized buffers return HostError::Invalid. This includes fixed arrays lowered
as WGSL parameter blocks. GPU-harness calls verify rejection at lengths 0, 7,
and 9 for an eight-element input, and successful sorting at length 8.
Checks use only the called entry's pipelines; a fixed input of another entry
sharing the same resource name does not constrain this call.

Loop initialization is skipped when the logical carry length is zero. It is
still performed for nonempty arrays even when the iteration count is zero.
Derived dispatch grids allow zero workgroups. Rust/WGPU and the WHL GPU backend
skip zero-work dispatches before constructing buffer bindings. This permits
zero-byte inputs to the by-key setup and gather phases, as verified below.

## Validation at the WIP checkpoint

[The GPU harness](../pkg/sort/test/radix_gpu.rs),
[source fixture](../pkg/sort/test/radix_gpu.wyn), and
[runner](../scripts/test_radix_gpu.ps1) reproduce the release-mode checks.
On the RX 580 Vulkan backend, **244 radix cases and 30 tuple-loop cases pass**:

- All six APIs run against stable CPU reference ordering. Integer callbacks
  use i32.get_bit; float callbacks use f32.get_bit. Two additional entries cover
  an odd five-bit request (which sorts six bits, as upstream does) and zero bits.
- Lengths are 1, 0, 2, 63, 64, 65, 255, 256, 257, 4097, 65537, 100003, 8, 0,
  and 1, with the final smaller calls reusing the same context after larger
  allocations. Every length runs randomized and duplicate-key inputs through
  all eight entries: 240 cases. Every output word and the unchanged input are
  checked. Distinct low-byte payloads verify stability among equal high-bit keys.
- Float comparisons check exact bits, including both zeros, infinities,
  subnormals, and positive/negative quiet and signaling NaN patterns.
- Four further cases check fixed-size sorting and rejection of invalid lengths.
- Both original tuple-loop fixtures execute on the GPU for 0, 1, 2, 3, and 5
  iterations. All six entries' outputs are checked, including tuple scans,
  nested boolean fields, and component-array publication.
- The existing tuple-loop compiler tests remain enabled. An additional host
  regression checks scalar-bound legalization and completion dispatches through
  both SPIR-V and WGSL host generation.

These are correctness checks, not fresh performance measurements. They do not
establish all numeric-width or cross-device/backend parity. In particular, the
current GPU harness runs generated Rust/WGPU with WGSL, not native SPIR-V.

Compiler gates pass: 1,320 core tests (14 ignored), 5 host tests, and 8 interpreter
tests. The interpreter also passes its 8 tests with its WGPU feature enabled
in a separate invocation. The tracked SPIR-V testfile gate passes 120 files.
The existing generated-host GPU suite also passes for both WGSL and SPIR-V;
three fixture calls now match the emitted parameter lists.
Formatting and whitespace checks pass. Running the compiler and WGPU-enabled
interpreter suites together currently triggers a Naga 26/27 dependency-feature
conflict in codespan-reporting; the separate commands below avoid that build
conflict without changing dependency versions.

Useful commands (run from the repository):

```sh
cargo test --release -p wyn-core -p wyn-host -p wyn-host-interp --lib
cargo test --release -p wyn-host-interp --features wgpu --lib
cargo fmt --all -- --check
git diff --check
pwsh -File scripts/test_radix_gpu.ps1
```

To reproduce compilation, import `pkg/sort/src/radix_sort` from a temporary
entry file and instantiate each of radix_sort, radix_sort_by_key,
radix_sort_int, radix_sort_int_by_key, radix_sort_float, and
radix_sort_float_by_key. Compile using:

```sh
wyn build -O --target wgsl --target-double rust-wgpu probe.wyn -o probe.wgsl
```

The runner defaults to Vulkan unless WGPU_BACKEND is already set. Generated
modules, the temporary Cargo crate, and local run logs live under
`target/radix-validation`.

## Fusion comparison: not yet equal or better

Reference: the earlier inspected Futhark 0.27.1 GPU IR and backend code recorded
in [radix-fusion-wip.md](radix-fusion-wip.md). The current check inspected newly
generated WGSL/Rust for the dynamic direct i32 sort and the saved Futhark GPU
IR. Futhark was not rebuilt or benchmarked during this audit.

| Property | Current Wyn | Recorded Futhark |
|---|---|---|
| Bin map plus totals reduction | Bin extraction is fused into the partial reduction; bins are not stored | Shared traversal emits bins and totals |
| Bin reuse | Separate source bin maps feed reduction and scan; final scatter recomputes its bin | Scan reads retained bins |
| Scan prefix adjustment, index calculation, scatter | Fused final kernel; no separate destination-index array | Fused scan output action |
| Scratch initialization per radix pass | No initializer stores or allocation | Eliminated |
| Reduction | Two launches: partials and combine | One reduction launch, then a scalar-copy launch |
| Scan | Three launches; full-length partial-prefix storage | OpenCL: three launches and full-length prefixes; CUDA/HIP: status clear plus lookback scan |
| Launches per nonempty 32-bit pass | 5 | OpenCL: 5; CUDA/HIP: 4 |

The current five-loop-kernel sequence is partial reduction, reduction combine,
partial scan, scan carry combine, and adjusted-prefix scatter. There is also
one initialization copy outside the loop. The publication-only completion
stage is not dispatched. Counts exclude uploads, readback, scalar uploads,
and context/cache management.

The earlier audit misidentified an unused array as bin storage. Tracing
`PlanResult` back to `SourceOperationValue` shows that it was the `copy(xs)`
initializer of the scratch destination. Its shader store copied the original
key, not the extracted digit. The original source has two separate `map(num,xs)`
expressions. Their computations are fused independently; no bin result is
retained across the reduction and scan groups.

The unused copy came from an unconditional operand demand for an operation
that the planner rematerializes from shape metadata. That demand is now
restricted to operations that actually execute at that boundary. Existing
residency rules consequently omit the copy's allocation and stores. Kernel
lowering also skips element computation for unallocated outputs. A radix-step
regression checks that only the four-component prefix array and sorted result
have allocations proportional to input length.

This removes one key-sized array allocation and one full key-array write per
pass. It does not establish that recomputing bins is preferable to retaining
them; a general cost decision across independently written but equivalent
producers remains open.

## Producer-sharing comparison

The regression input is `testfiles/rust_host_sharing.wyn`. The comparison below
uses generated Wyn shaders and host plans against the fusion mechanisms in
`extra/futhark/src/Futhark/Optimise/Fusion.hs` and `Fusion/Composing.hs`; it is
not a fresh Futhark compilation or performance comparison. Futhark composes
producer outputs into consumer inputs, removes duplicate inputs, repeats
vertical/horizontal/inner fusion, and removes unused outputs after convergence.

| Case | Wyn behavior and remaining gap |
|---|---|
| Diamond with multiple map consumers | One traversal; a shared producer element is evaluated once. |
| Producer used by two reductions | One input load and producer evaluation feed both accumulators; their guarded evaluation now shares one cache. |
| Reduction followed by a dependent map | Producer storage survives the reduction boundary and the later map reads it. |
| Sliced producer | Slice offsets are preserved and elements load directly at the adjusted index. The producer and sliced consumer still use separate dispatches; slice fusion remains a gap. |
| Array loop body | Sharing works within an iteration and across its reduction boundary; host iterations remain ordered. This does not establish arbitrary nested parallelism parity. |
| Scan output action | Radix fuses adjusted-prefix scatter. The simpler `action` fixture still has four launches: a shape observer of the horizontally fused scratch initializer prevents the contraction and leaves a stored index array. |
| Scan also returned | The observable scan output remains available; scatter stays separate. Broader output-action composition remains open. |
| Shape-only scratch initializer | Unused initializer storage and arithmetic are omitted; observed sibling results remain. |

The collective cache is scoped to one valid input element, before any workgroup
barrier. Slice traversal uses a separate cache for its adjusted index so two
different slices cannot alias one cached element. These changes use existing
source identities and residency facts; they introduce no expression IR or
Rust dependency graph.

Validation on 2026-10-01: 1,324 core tests passed (14 existing ignored tests),
and all 34 pipeline tests passed after the final runtime-sized fixture update.
The generated Rust host GPU suite passed on RX 580 / Vulkan with both WGSL and
SPIR-V. Its new matrix covers eight entries, three input patterns, and lengths
8, 65, and 257: 144 entry/backend cases, including unchanged-input checks.
The WGSL/Rust-host radix runner also passed all 244 radix and 30 tuple-loop
cases. Logs are `target/sharing-core.log`, `target/sharing-final-pipeline.log`,
`target/sharing-host-gpu.log`, and `target/sharing-radix-gpu.log`.

Equal launch counts do not imply equivalent fusion, memory traffic, or speed.
Wyn also still retains full-length scan-prefix storage, unlike Futhark's
CUDA/HIP lookback path. The i32/i64 counter-width difference further prevents
raw byte counts from being attributed solely to fusion quality.

Before merging: audit the remaining numeric contracts and package entry;
broaden backend/device coverage; and compare fusion and intermediate traffic
for the validated variants.
No claim of “same or better than Futhark” is justified at this checkpoint.
