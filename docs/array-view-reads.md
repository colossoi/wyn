# Avoid whole-row copies for scalar array reads

`rows[i][j]`, including a row projected from a tuple or read by a captured
callback, now keeps its address through scalar evaluation and SSA lowering.
The compiler loads the requested element instead of materializing the entire
intermediate array. Already evaluated values and captures remain boundaries,
preserving snapshots across updates. SPIR-V storage types also agree when a
nested input array retains an unresolved representation qualifier.

An earlier isolated WGSL comparison on the Radeon RX 580, using the same
32-thread workgroups and measurement host on both sides, measured these
compute-pass GPU medians for 100,000 tuple-based lookups:

| Row capacity | Before | After |
|---:|---:|---:|
| 64 | 0.13360 ms | 0.01504 ms |
| 256 | 2.12976 ms | 0.01504 ms |
| 1,024 | 11.41160 ms | 0.01696 ms |

These are lookup microbenchmarks, measured in three batches of 100 calls after
warmup; upload, compilation and readback are excluded. The change removes the
copying cost that grew with row capacity. Compiler fixtures check the absence
of private row arrays on both backends, and a GPU regression checks numerical
results and observable old/new values across an update.

The current fluid simulator gains no speedup from this change. Compiling the
default 4,320-particle program with `--graphics --subgroups` before and after
produced byte-identical SPIR-V (SHA-256
`0c80dc859ec52b9a72f23b47c275963e5c863e358b7d4b7caf9b988b5c65d339`),
and identical host programs apart from the shader filename. This is a compiler
improvement for the demonstrated lookup pattern, not a fluid performance result.

Run the focused gates after building the release compiler:

```text
cargo test --release --offline -p wyn-core --lib
cargo test --release --offline -p wyn --test compiler_fixtures
python scripts/test_view_reads_gpu.py
```

The GPU check requires a release `extra/viz` executable and `spirv-val`; `WYN`
and `VIZ` can override the executable paths. It exercises both scalar
optimization policies and both shader backends, including empty inputs,
partial workgroups and 100,000 lookups.

Validation on 2026-10-29: workspace formatting and all-target builds passed;
workspace tests passed after increasing `RUST_MIN_STACK` to 67108864 and
rerunning the package-manager test with `TEMP`/`TMP` inside the workspace.
Workspace doctests, all nine standalone crate builds/tests listed in the
README, and the browser-target WebAssembly check passed. Testfile validation
passed 123 SPIR-V programs and 122 WGSL programs; WGSL's existing linked-helper
miner exclusion accounts for its one skip. The core suite passed 1,355 tests
with 14 existing ignores.

The viewer's Naga-to-Vulkan translation currently produces an `ArrayStride`
decoration on a function-local fixed array (`VUID-StandaloneSpirv-None-10684`).
This reproduces with the pre-change compiler on a simple nested map; the
compiler's original SPIR-V passes `spirv-val`. Strict viewer validation
therefore does not pass on this machine. All 72 numerical GPU cases passed
with `python scripts/test_view_reads_gpu.py --no-validate`; this explicit
option disables viewer GPU layers while retaining `spirv-val` validation of
compiler output. It is not a clean Vulkan-layer validation result.

With the same viewer-layer workaround and separate `spirv-val` checks, the
fluid step, tiled-neighbor step, linked-neighbor and renderer checks passed.
The spatial check timed out after 120 seconds on `repro_large_tree_pipeline`;
that fixture's SPIR-V is also byte-identical before and after this change.
The full GPU gate set is therefore not clean, independently of the passing
compiler gates and numerical lookup tests.
