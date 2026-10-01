# Radix sort and fusion WIP

**Current checkpoint:** see [port fidelity and validation gaps](radix-port-fidelity-wip.md).
The latest compiler work has two failing tuple-loop tests and has not established
end-to-end parity for all six sorts or fusion equal to Futhark. Historical
measurements below do not validate that work.

This branch contains the Futhark radix port, a cooperative scan/reduction
implementation, and a remaining fusion/host-loop sketch. The sketch executes
radix passes as repeated parallel GPU dispatches through WHL and Rust/WGPU.
The cooperative collective change is isolated
from the sketch and validated separately against master.

## Source and port contract

[The Wyn port](../pkg/sort/src/radix_sort.wyn) translates
[diku-dk/sorts](https://github.com/diku-dk/sorts) v0.7.3, commit
`a473651d67d8ed0fafcea315f0226676721e2bdf`. The
[original source](../pkg/sort/upstream/radix_sort.fut) and
[ISC license](../pkg/sort/FUTHARK-LICENSE) accompany it.

The port preserves the two-bit radix step, local higher-order helpers, runtime
loop, separate totals reduction, and scratch scatter. All six public sorts are
present: unsigned bit-pattern, signed integer, and floating-point sorts, each
with a by-key variant. Key types and upstream bit-pattern behavior are retained,
including signed zero, NaNs, and the final two-bit request for odd bit counts.
Array dimensions, indices, ranks, and bucket counts use idiomatic Wyn `i32`
instead of Futhark `i64`, as agreed. That also bounds representable array sizes
and counts to Wyn's index range. Bucket labels now also use `i32` instead of
Futhark's `i8`, as requested, for compatibility with WGSL and the normal Naga
runtime. Their values remain 0-3. Key and payload types are unchanged.

The package's existing `src/lib.wyn` remains its default library; this new module
is imported directly by the existing
[smoke probe](../pkg/sort/test/radix_futhark.wyn). Do not inline helpers by hand,
unroll the source loop, replace
the scratch producer, or derive totals from the last scan element to bypass
compiler limitations.

## Futhark execution tree and scheduling

### Evidence and scope

On 2026-09-30, the exact upstream source was compiled again with **Futhark
0.27.1**, compiler commit `f547a39abb1f3773f7cdb324ec8ea7cb8cf13ea0`, using:

```futhark
entry main (xs: []i32) = radix_sort_int i32.num_bits i32.get_bit xs
```

The optimized SOAC IR, GPU IR, and generated OpenCL, CUDA, and HIP host/device
sources were inspected. Reproduction artifacts are in `target/futhark-radix`
in the primary checkout. This is inspection of generated code, not a Futhark
GPU timing measurement. Launch counts below apply to this entry, version, and
nonempty input; they are not universal properties of scan or radix sort.

The read-only `extra/futhark` submodule is a **different revision**:
`3a5bc16d91763a1d5aa0c66dea0bd0b4287ab68a`. Its implementation locations are
listed below. Numerical claims here come from the generated 0.27.1 code,
not an assumption that the checkout matches the installed compiler.

To reproduce, append the entry above to a copy of
`pkg/sort/upstream/radix_sort.fut`, saved as `target/futhark-radix/main.fut`:

```sh
futhark --version
futhark dev --standard target/futhark-radix/main.fut > target/futhark-radix/standard.ir
futhark dev --gpu target/futhark-radix/main.fut > target/futhark-radix/gpu.ir
futhark opencl --library target/futhark-radix/main.fut -o target/futhark-radix/radix_opencl
futhark cuda --library target/futhark-radix/main.fut -o target/futhark-radix/radix_cuda
futhark hip --library target/futhark-radix/main.fut -o target/futhark-radix/radix_hip
```

### Source dependencies and stable ordering

For length `n`, each pass extracts two key bits into a bin in `0..3`. It
computes three global totals `(na, nb, nc)` for bins 0, 1, and 2, and four
inclusive prefix counts `(a, b, c, d)`. The fourth global total is unnecessary:
only the populations of earlier bins determine a bucket's base address.

An element's destination is its inclusive count within its bin, minus one,
plus the populations of all preceding bins. Every element therefore has a
distinct destination, destinations cover the output, and input order within
each bin is preserved. This makes the pass stable and justifies uninitialized
scratch. Input and destination cannot generally alias while parallel threads
are still reading input elements.

```text
host entry
`-- n == 0 ? zero iterations : 16 two-bit passes for i32
    `-- host loop: digit = 2 * pass; carry current array
        |-- GPU map/reduction: current array -> bins + (na, nb, nc)
        |-- obtain fresh scratch destination of n elements
        |-- GPU scan with output action
        |   |-- bins -> four indicators -> four inclusive prefix counts
        |   `-- final prefix + bin + totals + original element
        |       -> destination index -> write element into destination
        `-- carry destination as the next pass's input
```

The outer loop is generated **host control**, not one GPU thread running all
array operations. Passes remain sequentially dependent; work within each
reduction/scan is distributed across GPU threads and blocks. Arrays stay in
device memory. The generated loop contains **no scalar device-to-host
readbacks** of bucket totals. Host reference assignment at the backedge does
not copy array data. Empty input skips the body, although setup outside the
loop may still occur.

### What fusion removes and retains

The optimized SOAC IR has one `redomap` and one `screma` inside `with_acc`:

| Source work | Optimized representation | Materialized output |
|---|---|---|
| Both `map(num, xs)` occurrences and three-count reduction | One `redomap`, computing each bin once and retaining it alongside the reduction | `bins[n]` and three totals |
| Four equality tests and boolean-to-count conversion | Scan input mapper | No separate flag arrays |
| Four component scans | One tuple scan in `screma` | Backend-dependent internal scan storage |
| `map2(f, bins, offsets)` | Scan output mapper | No separate final destination-index array |
| `scatter` | `update_acc` in the scan output mapper, enclosed by `with_acc` | Next array |
| `#[scratch] copy(xs)` | `scratch(i32, n)` | Allocation without copying/initializing elements |

`with_acc` establishes access to destination storage. The output mapper updates
that accumulator once a globally adjusted prefix is available. This is
composition of a scan with an output action, without a radix-specific IR node.
The compiler author's [scan/scatter fusion explanation](https://www.futhark-lang.org/blog/2026-03-24-scan-scatter-fusion.html)
describes the representation and motivation.

Reduction and scan remain distinct because the scan's final output action
needs all three totals. Replacing the reduction with an index into the last
scan result would introduce a completion dependency on that result array and
obstruct this fusion. Removing the reduction requires a different collective
schedule, not just another map-fusion rule.

### Actual generated GPU launches

GPU IR contains the main `segred` and `segscan`, plus a small `gpu` body that
reads the three reduction results into three fresh device scalar slots. This
survives as a **one-thread copy kernel**.

For OpenCL, every nonempty radix pass executes:

```text
1. segred_nonseg  -- emit bins and reduce three totals in parallel
2. gpuseq         -- one thread copies three i64 totals into three device slots
3. scan_stage1    -- local scans; store four prefix streams and forwarded bins
4. scan_stage2    -- one workgroup scans block/chunk carries
5. scan_stage3    -- apply carries, calculate destinations, scatter payloads
```

This is **5 kernels/pass, or 80 for a nonempty 32-bit sort**, excluding context
initialization and caller upload/readback. The source module is named
`SegScan.TwoPass`, but this generated scan has three launches. Module names and
SOAC counts are not reliable physical launch counts.

The reduction is a single launch despite two logical reduction phases. Blocks
write partial sums; the last block, identified with an atomic counter, performs
the final reduction. That counter is initialized during context setup and
reset by the reduction completion protocol.

CUDA and HIP select the single-pass scan for this primitive tuple-addition
operator. Their per-pass schedule is:

```text
1. segred_nonseg  -- bins and totals
2. gpuseq         -- copy the three totals
3. replicate_i8   -- clear the scan's per-tile status flags
4. segscan        -- decoupled-lookback scan with scatter output action
```

This is **4 kernels/pass, or 64 for a nonempty 32-bit sort**, excluding setup
and caller transfers. Counting only direct entry kernel calls would give the
wrong answer: the status-clear launch is inside a `replicate_i8` helper.
Single-pass describes the main scan traversal, not the absence of setup or
synchronization. CUDA/HIP code generation was inspected; neither backend has
been executed or benchmarked in this work.

The backend queue/stream orders launches so later readers see earlier writes.
No CPU readback of the array or totals is needed between passes. Workgroup
barriers and inter-block synchronization remain part of the scan/reduction
algorithms; fusion does not remove these requirements.

### Buffers and lifetimes

Let `R` be the reduction block count and `V` the number of scan tiles. These
are logical payload sizes, not measured allocator peaks or device-allocation
traffic:

| Storage | OpenCL 0.27.1 | CUDA/HIP 0.27.1 |
|---|---|---|
| Current i32 input and next destination | `4n` bytes each | `4n` bytes each |
| Retained bin labels | `n` bytes (`i8`) | `n` bytes (`i8`) |
| Three totals plus three copied scalar slots | 48 bytes across six allocations | Same |
| Three reduction partial streams | `24R` bytes | Same |
| Four full-length scan prefix streams | `32n` bytes | Absent |
| Forwarded bins between scan phases | Additional `n` bytes | No full-length forwarded stream |
| Lookback state | Not used | `V` status bytes plus `64V` bytes for four aggregate and four inclusive-prefix streams |

There is also shared/local memory and constant-size counter state. The caller's
original input may remain live after the loop advances to a different current
array; current-plus-destination is not a claim about total peak memory.

Bins and the six total slots are allocated before the loop. The generated body
still calls the allocator for reduction scratch, destination, and scan scratch
on each iteration. These are Futhark caching/reference-counting allocator calls,
**not evidence of fresh driver allocations each time**. The carried reference
keeps the preceding input alive while the next destination is obtained. The
code does not explicitly preallocate two destination buffers and merely swap
host pointers.

OpenCL fusion eliminates the *public* prefix/index intermediates, but internal
full-length prefix traffic remains. CUDA/HIP keeps tile summaries instead.
This distinction is why fused scan/scatter alone does not establish optimal
memory traffic.

### Assessment of suboptimal work

These are engineering judgments about the observed code. Proposed speedups
have not been measured.

| Area | Assessment and possible improvement |
|---|---|
| Shared bins/totals | Good fusion: repeated bin computation coalesces and one traversal both retains bins and computes totals. |
| Scan/index/scatter | Good fusion: no separate destination-index array or scatter traversal; scratch initialization also disappears. |
| One-thread totals copy | Appears avoidable: copy 24 bytes unchanged, pay one launch/pass, allocate three more scalar slots. Letting the output action read original reduction buffers should remove it if alias/lifetime invariants allow. That would remove 16 launches from this 32-bit sort; timing benefit is unmeasured. |
| OpenCL scan | Less efficient than the CUDA/HIP path in launch count and prefix storage. A portable single-pass replacement needs sound memory ordering and forward-progress guarantees; importing a CUDA synchronization strategy blindly is not valid. |
| Forwarded OpenCL bins | Appears redundant: original bins remain available, but stage 1 writes another `n`-byte copy for stage 3. Forwarding the original storage directly should avoid that allocation and traffic. |
| Four prefix counters | The fourth inclusive count can be reconstructed as `(index + 1) - first - second - third`. Exploiting that could save accumulator/shared-memory pressure and an `8n`-byte OpenCL stream. This is an algorithmic/relational optimization, not ordinary producer-consumer fusion or a proposed source workaround for Wyn. |
| Separate totals reduction | A genuine global dependency, not an obvious missed fusion. A specialized radix algorithm can organize work differently, but needs a different synchronization protocol. Preserve this algorithm while fixing Wyn. |
| Two-bit radix | Sixteen passes is conservative. Wider digits reduce passes but increase counters and register/shared-memory demand. The best width requires measurements for representative sizes, payloads, and devices. |
| Loop allocation calls | Invariant scratch reuse and alternating destinations could reduce bookkeeping. Current caching means source allocation-call counts alone overstate probable driver cost. Measure before complicating allocation policy. |
| Generic launch configuration | Block sizes/counts and scan chunk sizes are tunable. Small arrays may be launch-bound and larger ones traffic/occupancy-bound. A small-array path or tuning may help; no optimal parameters are established here. |

The by-key source first builds `(key, original_index)` pairs, sorts those, then
gathers original payloads. This avoids moving large payloads through every
pass but adds index traffic and a final gather. The counts/sizes above are for
the directly sorted i32 entry, not every by-key or float specialization.

### Futhark implementation locations

In the read-only `extra/futhark` checkout:

- `src/Futhark/CodeGen/ImpGen.hs`: carries loop merge parameters through
  imperative `for`/`while`; associates `WithAcc` parameters with destination
  arrays.
- `src/Futhark/CodeGen/ImpGen/GPU.hs`: dispatches `SegRed`/`SegScan` codegen.
- `src/Futhark/CodeGen/ImpGen/GPU/SegRed.hs`: retained map outputs, block partials,
  and the final-block reduction protocol.
- `src/Futhark/CodeGen/ImpGen/GPU/SegScan.hs`: chooses the scan implementation
  according to backend and supported operator.
- `src/Futhark/CodeGen/ImpGen/GPU/SegScan/SinglePass.hs`: lookback and per-tile
  synchronization state.
- `src/Futhark/CodeGen/ImpGen/GPU/SegScan/TwoPass.hs`: general fallback with
  stored prefixes and carry propagation.

## What Wyn should take from this

The general mechanisms are a collective's composable output action and host
control that repeats a scheduled parallel region while carrying scalar state
and array resources. They belong in ordinary fusion, placement, scheduling,
and host-program lowering. There should be no radix-only operation or source
pattern recognizer. Matching Futhark's useful structure does not require copying
its extra totals-copy kernel or forwarded-bin allocation.

## Implemented cooperative collectives

The scheduler selects a chunk grid and workgroup width before allocation and
SSA lowering. That same choice supplies the chunk dispatch, partial/offset
buffer capacities, and contiguous input partitioning. The default is 256
workgroups of 256 lanes; the value is a scheduling policy, not a duplicated
assumption in Rust. An explicit entry grid supplies the chunk grid, including
all three axes. Element phases also use that grid. Combine and stable compaction
are single-workgroup recipes and remain one group.

Each chunk group scans whole tiles in order, carrying its aggregate between
tiles. One cooperative combine group scans the partials, processing multiple
tiles when the authored grid has more groups than fit in one workgroup. Scan
output kernels add the appropriate exclusive group carry. Empty chunks publish
the neutral element. Shared-memory barriers protect bank changes and reuse
between tiles. Operand order is preserved for associative, noncommutative
operators; floating-point regrouping still has the usual parallel-reduction
rounding consequences.

The same workgroup-scan implementation serves scan, reduction, and stable
filter compaction. It replaces the duplicated filter implementation and the
serial carry loop. Unused mapped-stream scratch rules and their unreachable
lowering branches are removed.

Implementation locations:

- [schedule.egg](../wyn-core/src/egglog/schedule.egg): collective launch policy,
  stage grids, and scratch extents.
- [plan.rs](../wyn-core/src/egglog/to_ssa/plan.rs),
  [sizes.rs](../wyn-core/src/egglog/to_ssa/sizes.rs), and
  [publication.rs](../wyn-core/src/egglog/to_ssa/publication.rs): decode and
  publish the selected launches without replacing their grids afterward.
- [screma.rs](../wyn-core/src/egglog/to_ssa/kernels/screma.rs): cooperative tile
  scan, reduction, and carry propagation.
- [filter.rs](../wyn-core/src/egglog/to_ssa/kernels/filter.rs): shared scan helper
  used for stable compaction.

GPU validation on the RX 580 passed 250 cases: scalar scan/reduction,
noncommutative tuple scan/reduction, and stable filtering, each across five grid
choices and ten lengths. Grids were automatic, 1x1x1, 2x3x4, 257x1x1, and 3x7x17.
Lengths were 0, 1, 63, 64, 65, 255, 256, 257, 4097, and 100003. Every returned
value was checked against a CPU reference. Tuple scans were consumed by a map
to scalar values, avoiding the separate direct tuple-array publication issue.
The local reproduction harness and logs are in `target/collective-scheduling`.

## Remaining fusion and host-loop sketch

### Actual radix schedule

The generated host program copies the initial input into carried storage once,
then executes sixteen two-bit passes for nonempty i32 input. Each pass has five
GPU dispatches:

1. Reduce per-group bucket totals.
2. Scan per-group prefixes.
3. Cooperatively combine the reduction partials.
4. Cooperatively scan the prefix carries.
5. Apply carries, compute destination indices, and scatter into the next buffer.

Steps 1 and 2 precede their respective combine phases; the final scatter needs
both results. The buffers are allocated before the host loop. The loop updates
its GPU-visible digit index and swaps current/next buffer references at the
backedge. There are no CPU readbacks of bucket totals between passes.

This matches the inspected Futhark OpenCL launch count of five per pass, with
different work in those launches: Wyn uses a separate reduction-combine kernel
and omits Futhark's totals-copy kernel. It does not implement the CUDA/HIP
single-pass lookback scan. Full-length internal prefix storage remains:
four i32 streams require `16n` bytes. Fusion removes the final public prefix and
destination-index arrays, but does not eliminate that internal traffic.

### Fusion and sharing

[fusion.egg](../wyn-core/src/egglog/fusion/fusion.egg) and
[composition.egg](../wyn-core/src/egglog/fusion/composition.egg) permit an
unobserved scan to feed a scatter into fresh scratch. The scan's final output
phase invokes the indexed-write lowering with globally adjusted prefixes.
The original separate totals reduction and source algorithm remain intact.

Cheap bucket extraction is recomputed in the reduction, scan input, and final
output phase. Futhark retains byte-sized bins; Wyn uses i32 bins and currently
recomputes them. An earlier experiment retaining i32 bins was slower on the
RX 580 (about 231 ms versus 206.5 ms for 100 MB), and that experimental code was
removed. This is evidence about this expression, representation, and device;
it is not a general rule to recompute arbitrary shared producers. Purity,
observers, aliasing, and the cost of recomputation still need a production review.

### Host loops

The sketch recognizes counted loops carrying one fixed-size array. Egglog
placement exposes the body's parallel region, and allocation assigns distinct
current/next buffers. [loops.rs](../wyn-core/src/egglog/to_ssa/loops.rs) publishes
loop metadata. [program.rs](../wyn-host/src/program.rs) reconstructs the loop
body from named stages, and [whl.rs](../wyn-host/src/whl.rs) emits the repeated
dispatches and buffer swap.

That stage-name reconstruction and the single-array carry restriction are
unfinished design work. Loop bodies and carried resources should be represented
directly in the host program. [rust_wgpu.rs](../wyn-host/src/rust_wgpu.rs) now
emits the same counted loop, ordered index uploads, body dispatches, and buffer
swaps. Index uploads use encoder copies rather than queue writes, preserving
their order within batched submissions. Both carried buffers are allocated
outside the loop and excluded from the context's reusable scratch cache because
either buffer can become a returned result. Zero iterations,
odd/even iteration counts, nested control, and retained aliases need explicit
coverage before broadening this implementation.

### Measurements and limits

Earlier release-mode measurements of the cooperative radix sketch on an RX 580
were **30.341 ms for 10 MB** and **206.521 ms for 100 MB** of random i32 data
(decimal MB), using the median of five warm runs and checking every output
against the CPU sort. These timings exclude module setup and caller upload and
readback; they include host-loop submission and waiting for GPU completion.
They predate the final grid-generalization change and are not fresh timings of
this commit. The independent scalar CPU port is in
[radix_cpu.rs](../pkg/sort/radix_cpu.rs).

Futhark-generated code was inspected, not benchmarked on this GPU. The results
therefore establish a working parallel Wyn sketch, not a measured speedup over
Futhark or CUDA/HIP scan parity. Full validation of all six radix variants,
stable by-key ordering, float bit patterns, and odd bit counts remains open.

## Validation and next steps

The Rust host-loop emitter was exercised on 2026-09-30 with generated WGSL
through WGPU/Vulkan on a GTX 1660 Ti. A release-mode run sorted 100,000,000
bytes (25,000,000 pseudorandom i32 values, xorshift32 seed `0x12345678`). All
outputs from the final run matched a CPU sort. Five warm runs took 156.679,
148.352, 147.551, 147.929, and 147.841 ms: median **147.929 ms**. Timing includes
host encoding/submission, per-call allocation, and GPU completion, but excludes
context setup, input upload, readback, and CPU verification. The CPU reference
sort took 846.293 ms. This device differs from the earlier RX 580 measurements.
The single-threaded Rust two-bit radix implementation in `pkg/sort/radix_cpu.rs`
was then built with `rustc --edition=2021 -O -C target-cpu=native` and run on the
same 100 MB input on an AMD Ryzen 7 2700X. Five warm runs took 2085.895,
2059.962, 2084.843, 2088.262, and 2085.427 ms: median **2085.427 ms**. Every
run matched the CPU standard-library sort. CPU radix timing includes input
copying and scratch allocation, but excludes generation and verification.
The measured GPU sort interval is about **14.1x faster** than this sequential
two-bit radix implementation; that ratio excludes GPU transfers and is not a
comparison with an optimized wider-digit or parallel CPU radix algorithm.
Including upload and readback on the same GTX 1660 Ti, five warm runs measured
189.726, 188.499, 188.749, 187.583, and 188.656 ms until all 100 MB was mapped
and CPU-readable: median **188.656 ms**. Including an additional copy into an
owned CPU byte vector gave a median of **225.279 ms** (runs: 229.092, 225.279,
213.695, 220.498, 227.931 ms). These intervals include fresh input/readback
buffer allocation, input upload, host encoding/submission, sorting, and readback
synchronization. Device/pipeline setup, input generation, and verification remain
outside the interval. Every output in all six runs (one warmup plus five timed)
matched the CPU reference. Relative to the 2085.427 ms CPU radix median, these
are approximately **11.1x** and **9.3x** faster, respectively.

The Rayon version in `pkg/sort/radix_rayon.rs` retains sixteen stable two-bit
passes. Workers count buckets in contiguous chunks, then scatter into disjoint
output slices assigned by bucket and input-chunk order. This preserves stability
without unsafe code or atomics. A small serial prefix over chunk histograms
assigns those slices; it does not scan the full input serially.
Build with `RUSTFLAGS="-C target-cpu=native" cargo build --release
--manifest-path pkg/sort/cpu-bench/Cargo.toml` (Rayon 1.12.0 in the lockfile).
Run the resulting `radix-rayon 25000000 5` with `RAYON_NUM_THREADS=8` or `16`.

On the same Ryzen 7 2700X and identical input, five warm runs gave:

| Rayon workers | Run times (ms) | Median (ms) |
|---|---|---|
| 8 | 462.158, 449.882, 452.555, 445.006, 456.272 | 452.555 |
| 16 | 298.722, 300.815, 289.369, 287.774, 290.381 | 290.381 |

All 25 million values were verified after every run. Timing includes input
copying, scratch allocation, histograms, and all passes; thread-pool startup,
input generation, reference sorting, and verification are excluded. Sixteen
workers are **7.2x faster** than the earlier sequential port. Compared with
that Rayon median, GPU sorting is **2.0x faster** without transfers, **1.5x**
including upload and mapped readback, and **1.3x** including a copy into an owned
CPU vector. These are measured comparisons of these implementations, not a
claim that two-bit radix is optimal for either processor.

The existing five wyn-host tests pass after filling in their missing
`dispatch_loops` descriptor fields; no test cases were added. The compiler
build and Rust host generation for both WGSL and SPIR-V passed. A small WGSL
run also verified two calls recorded in one encoder on software Vulkan.

The feature checkout passes the 69 lowering tests. Its WGSL suite passes 76 of
78 tests; the known failures are captured fixed-array typing and scatter into
a storage buffer. These belong to the remaining sketch. The collective patch
passed the complete core unit suite independently against master: 1,319 passed,
14 ignored, no failures. Those sketch changes are not needed to adopt it. GPU
checks described above exercise the generalized scheduling.

Next:

1. Resolve the captured-array typing and scatter-lowering regressions, and
   review the WIP polymorphic-lambda/prelude contract changes.
2. Replace stage-name loop reconstruction with explicit host loop structure
   and carried resources.
3. Review scan-output fusion legality and producer retention/recomputation,
   including still-observed results and destination aliasing.
4. Validate all radix variants and boundary cases, then rerun release timings
   and compare launch count, allocations, live storage, and transferred bytes
   with the inspected Futhark schedules. Reducing internal prefix traffic is
   the largest remaining structural performance opportunity; a portable
   single-pass protocol requires a separate synchronization design.
