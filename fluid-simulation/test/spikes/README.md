# Optimization viability spikes — 2026-10-27

These began as isolated probes. The first two lowerings are now implemented in
the compiler; see the production results below. Wyn remains the algorithm source. `patch_generated.py` models two possible compiler lowerings by editing
emitted WGSL; these shaders are not proposed as simulator implementations.

## Initial spike results

GeForce GTX 1660 Ti, two rounds of 16 frames, first two discarded. Second round
reverses variant order. Values are median milliseconds per round, including host
execution/allocation and waiting for GPU completion, excluding final readback.
Other tests were running; initial load averages were 4.75/5.04/5.60. Raw samples
are in `results.json`. These are probe timings, not simulator frame speedups.

| Probe | Current emission | Experimental emission | Conclusion |
|---|---:|---:|---|
| 4,096 particle neighbor outputs | 2.148 / 2.086 | 0.163 / 0.149 | Direct tiled output is worthwhile |
| Same, only reuse the query for counts | 2.148 / 2.086 | 2.091 / 1.893 | Reuse alone explains little of the improvement |
| 32,768 independent scans of 32 u32 values | 6.414 / 6.158 | 0.439 / 0.427 | Native subgroup scan is worthwhile |

All output elements, counts, and padding were checked against CPU references.
The neighbor input is 64 isolated groups of 64 collinear particles, with at most
seven neighbors per particle. It exercises actual narrow-phase code and tiled
output, not octree traversal. The full simulator's neighbor density may differ.

### Neighbor output

Both Wyn spellings (`baseline` and `candidate`) still emit two dispatches and a
1,605,632-byte row intermediate. The experimental lowering retains the first
neighbor query, writes directly into the existing tiled output, and writes its
count once. It removes the row allocation and transpose dispatch. The separate
reuse control retains the row/transpose and removes only the repeated query.

Next implementation: producer-to-output layout fusion, using existing scheduling
and materialization machinery. Merely spelling the permutation as a scatter in
Wyn does not currently achieve it. These timings do not isolate memory traffic,
register pressure, and coalescing from one another.

### Subgroups

Current emission uses 256 threads and workgroup memory for each 32-element scan.
The experimental lowering uses one 32-thread workgroup and `subgroupInclusiveAdd`,
with no shared-memory scan or workgroup barriers. It requires 32-lane subgroups
and tests only associative u32 addition. General lowering must select supported
operations/widths, handle active lanes, and retain a workgroup fallback. It does
not prove whole traversal speedups or arbitrary user-defined scan support.

The temporary viewer enables `Features::SUBGROUP`. Naga 27 recognizes the
operation without an `enable subgroups` directive (it rejects that directive).

### Blocked radix

`blocked_radix.wyn` expresses one stable four-bit pass using 64-key blocks, packed
histogram scans for two local two-bit passes, block histograms, and global scatter.
For 4,096 keys its logical block count/prefix tables each contain 1,024 integers
(4 KiB), versus 65,536 integers (256 KiB) for a per-key 16-bin prefix table.
This is an algorithmic storage comparison, not measured generated allocation.
It is a smaller decomposition probe, not upstream's complete 256-key block kernel.

**Updated compiler check, 2026-10-27:** `/home/jonah/.cargo/bin/wyn` now compiles
`../repro_nested_scan_reduce_capture.wyn` and `blocked_local_sort.wyn` on both
backends. GPU outputs pass: the minimal example returns `[[5,7],[11,15]]`, and
all 64 local-sort rows match stable CPU sorting for random, equal, and reverse
keys. Results are recorded in `updated-compiler-checks.json`.

**Second updated compiler check, 2026-10-27:** the allocation failure is also fixed.
`../repro_nested_scan_flatten.wyn` compiles and returns `[1,3,3,7]` for `[1,2,3,4]`
on both backends. The full blocked pass and baseline match stable CPU sorting for
all 4,096 keys on random, equal, and reverse inputs. No source workaround was
introduced. Raw checks and samples are in `blocked-compiler-checks.json`.

The blocked decomposition is correct, but the current lowering is slower on this
small probe. Random-input medians (16 frames, first two discarded):

| Backend | Existing four-bit pass | Blocked pass |
|---|---:|---:|
| WGSL | 0.198 ms | 1.346 ms |
| SPIR-V | 0.169 ms | 1.776 ms |

These include host execution/allocation and GPU completion, as above. Generated
host programs allocate 344,128 bytes for the baseline versus 108,736 for the
blocked pass, including the output; dispatch counts are three versus six. This
proves the decomposition and reduced allocation, not a performance win or parity
with upstream's complete blocked implementation.

## Reproduction

From the repository root:

```sh
target/release/wyn build fluid-simulation/test/spikes/neighbor_output.wyn -t wgsl -o /tmp/spike-neighbor.wgsl
target/release/wyn build fluid-simulation/test/spikes/subgroup_scan.wyn -t wgsl -o /tmp/spike-subgroup.wgsl
python3 fluid-simulation/test/spikes/patch_generated.py
python3 fluid-simulation/test/spikes/measure.py extra/viz/target/release/fluid-timing-viz
# With the updated installed compiler, local sorting compiles:
/home/jonah/.cargo/bin/wyn build fluid-simulation/test/spikes/blocked_local_sort.wyn -t wgsl -o /tmp/spike-local.wgsl
# Both now compile and pass GPU checks with the updated installed compiler:
/home/jonah/.cargo/bin/wyn build fluid-simulation/test/spikes/blocked_radix.wyn -t wgsl -o /tmp/spike-blocked.wgsl
/home/jonah/.cargo/bin/wyn build fluid-simulation/test/repro_nested_scan_flatten.wyn -t wgsl -o /tmp/spike-flatten.wgsl
```

The shader patcher deliberately asserts current emitted names. Regeneration with
another compiler may require adapting it. `measure.py` writes raw outputs/logs
and `spike-results.json` under `/tmp`.

The measurement viewer is an isolated copy in `/tmp/wyn-fluid-timing-viz`.
`timing-viewer.patch` adds a GPU-completion wall-clock measurement to the current
viewer, which already requests subgroup support. To recreate it, copy `extra/viz` into a
temporary directory, apply that patch with `patch -p1`, rename its Cargo package
`fluid-timing-viz`, make its path dependency on `wyn-host-interp` absolute, and add
an empty `[workspace]` section. Build with:

```sh
cargo build --release --offline -j 2 --manifest-path /tmp/wyn-fluid-timing-viz/Cargo.toml --target-dir /home/jonah/wyn/extra/viz/target
```

## Production implementation — 2026-10-27

The compiler now recognizes a complete tiled gather and routes its row producer
straight to the tiled destination. This works with fused metadata maps and
reductions, and preserves row storage when another consumer needs it. Partial
tiles retain the original gather. Selection and safety checks live in Egglog;
SSA emission only implements the selected address mapping. The simulator loses
one dispatch and its 1,693,440-byte `[4320][98]` intermediate without a Wyn rewrite.

`wyn build --subgroups` enables native scalar 32-bit integer/float addition in
nested scans and reductions on both backends. Subgroup ranks define logical lane
order; shared carries join multiple subgroups, including partial groups. This
does not assume 32-lane hardware. Unsupported operators keep workgroup lowering;
without the flag, all collectives remain portable. The simulator launcher uses
the flag and viz requests the adapter's supported subgroup feature.

On the same GTX 1660 Ti, generated production WGSL passes exact CPU checks:
neighbor output 0.162 ms, nested scan 0.869 ms (16 frames, discard two). These
replace the spike's approximately 2.1 ms and 6.3 ms baseline kernels. The general
subgroup implementation includes carry synchronization absent from the original
single-subgroup shader experiment.

Back-to-back full simulator runs, reversing order, measured baseline medians
27.831/25.822 ms and optimized medians 24.233/23.421 ms. A subsequent comparison
recorded in `production-results.json` also checks identical particle, velocity,
and octree output buffers after 16 frames. These include host execution and GPU
completion, under concurrent CPU work; the full-frame improvement is smaller
than the isolated kernel improvements.

Regressions: `python3 fluid-simulation/test/kernel_outputs_check.py` checks both
backends, multi-subgroup/tail cases, signed/floating addition, custom reducers,
metadata reductions, preserved rows, and padded tiles. Set `WYN` and `VIZ` to
select the compiler and viewer builds. Planning/backend unit tests additionally
check feature selection and shader validation.


Final gates: workspace all-target debug build; 1,670 workspace tests passed with
14 existing ignored tests (release test binaries, `RUST_MIN_STACK=67108864`);
all documented standalone crate builds/tests and browser-target check; 124
SPIR-V testfiles and 123 WGSL testfiles with one expected WGSL skip; package
validation and all simulator/radix GPU checks. The full suite exposed an existing
inclusive stepped-range count bug, now fixed with a regression: `0..2..=4` has
three elements, matching its inferred type.
