# SSA local-storage optimization

Fluid's local-storage phase fell from **1,475.8 ms to 152.9 ms (9.65x)**.
Storage-rule execution fell from **1,352.8 ms to 60.5 ms (22.35x)**.
Whole-process compilation fell from **4.161 s to 2.804 s**, a **32.6% reduction**.

These are paired measurements against the saved, instrumented compiler from
the [fine profile](compiler-profile-fine-2026-10-28.md). Both binaries use
upstream revision `1e0269814a19c0e0edbae9847485c0546a82d3c9` plus the local
profiling work. The updated binary adds the storage optimization described here.

## Before and after

Median of three fresh processes per binary and workload, after one warmup each.
Runs were sequential, with before/after order alternating between rounds. No
builds or tests ran during measurement. Project options match the previous
reports: optimized Tinyporto with a Rust/WGPU host, and fluid with subgroups
and its normal unoptimized configuration. Both target SPIR-V.

| Work | Tinyporto before | Tinyporto after | Fluid before | Fluid after |
| --- | ---: | ---: | ---: | ---: |
| Local-storage phase | 44.6 ms | 35.4 ms | 1,475.8 ms | 152.9 ms |
| Storage rules, included above | 13.7 ms | 5.7 ms | 1,352.8 ms | 60.5 ms |
| Entire SSA/SPIR-V backend | 202 ms | 190 ms | 1,565 ms | 243 ms |
| Whole compiler process | 7.552 s | 7.619 s | 4.161 s | 2.804 s |

Tinyporto's local-storage phase improved by 20.7%. Its total median increased
0.9%, within the overlapping observed ranges: 7.503-7.823 s before and
7.597-7.637 s after. This establishes no total-time improvement for Tinyporto.
Fluid's process ranges were 4.070-4.204 s before and 2.759-2.806 s after.
Nested rows must not be added together.

## Granularity

The original analysis assigned a separate point to every SSA instruction and
propagated allocation-origin reachability even for arrays that could not reuse
an existing allocation. Its tables carried unnecessary control-flow detail.

The updated importer keeps array definitions/uses and accesses to allocations
that could supply a local array, including all derived pointers. It contracts
scalar-only and unrelated-memory steps. Within a basic block, accesses between
array definitions and allocation boundaries share an event group: their order
cannot change the liveness or origin-reachability decisions queried here.
Every CFG block/edge and edge-translated array argument remains represented.

Allocation-origin reachability now runs after reuse selection, only for selected,
single-origin array webs whose source is an actual allocation. The same alias,
later-access, interference, escape, and reallocation checks still decide reuse.

| Imported granularity | Tinyporto | Fluid |
| --- | ---: | ---: |
| SSA instructions previously represented individually | 1,020 | 16,233 |
| Storage-event groups now represented | 17 | 533 |
| CFG blocks retained | 85 | 785 |
| Local array values retained | 17 | 317 |

For example, `fluid__compute_7_combine` has 3,565 instructions but only 99
storage-event groups, with all 70 blocks retained. The two partial kernels each
use 81 groups instead of roughly 3,000 instruction steps. Group counts exclude
block-boundary points; they are not total relation-row counts.

Fluid now spends approximately 48 ms loading schemas, 8 ms importing facts,
and 61 ms executing storage rules. Collecting the smaller representation costs
2.8 ms, versus 0.3 ms before. This is still not a proof that all remaining
structural analysis costs are minimal.

## Production line count

Relative to the saved compiler immediately before this storage optimization:

| Production source | Added | Removed | Net |
| --- | ---: | ---: | ---: |
| `wyn-core/src/ssa/local_storage.rs` | 124 | 38 | +86 |
| `wyn-core/src/ssa/local_storage.egg` | 10 | 7 | +3 |
| **Total** | **134** | **45** | **+89** |

These are source-line counts including comments and blank lines, excluding
tests, documentation, and the earlier profiling instrumentation.

## Validation

All documented full gates passed:

- Workspace formatting, build with all targets, and tests: **1,500 passed,
  zero failed, 14 existing ignores**, including all 127 CLI fixture files.
- All-target builds and tests for `wyn-wasm` and the documented standalone
  crates: viz, tephra, spirv-validator, wyn-doc, mdbook-wyn-highlight,
  treereduce-wyn, and both tree-sitter-wyn manifests.
- WASM formatting and locked `wasm32-unknown-unknown` check.
- Tracked shader sweeps: **123 SPIR-V passed**; **122 WGSL passed**, one expected
  linked-SPIR-V-helper skip.
- Library package compile/validation runner and `git diff --check`.

Targeted tests verify contraction of scalar padding and unrelated storage,
grouping of repeated accesses, preservation of late derived-pointer accesses,
eligibility filtering, and live values across source-allocation reexecution.

Both large outputs pass `spirv-val`. Tinyporto's shader is byte-identical;
fluid differs only in interface-ID order on 59 `OpEntryPoint` instructions.
All other 27,472 fluid disassembly lines match. Both generated hosts are
byte-identical. GPU execution was not performed.

Windows test threads used `RUST_MIN_STACK=67108864`. The final workspace run
used a writable directory under `tmp/` for `TMP` and `TEMP`, resolving a
sandbox directory-rename denial in a package-cache test. A package-runner
executable lock and a parser/sweep temporary-file race passed sequential reruns.
No production change was needed for those environment/concurrency issues.

[Evidence](profiles/2026-10-28-compiler-storage.json) records every measured run,
raw benchmark logs, binary hashes, harnesses, the optimization patch, granularity
counts, output comparisons, and gate commands/results. Local binaries, shader
outputs, and full gate logs are under `tmp/compiler-storage-opt-2026-10-28`.
