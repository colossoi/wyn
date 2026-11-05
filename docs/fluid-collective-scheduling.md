# Size collective launches to the useful work

The fluid simulator's default 4,320-particle Hilbert key/index sort is about
twice as fast on the RX 580 after this compiler change. This is the existing
`Sort.FourBit` implementation on the simulator's colliding-block preset data.
The sorting algorithm and fluid source are unchanged.

## Compiler change

Previously, automatic global scans/reductions always launched 256 chunk
workgroups and a 256-thread carry workgroup. For the fluid sort, each chunk
contains 64 elements, so only 68 of those workgroups have input.

The existing scheduling pipeline now has an explicit selection point after
shape/layout facts and recipe discovery, before phase/storage finalization.
`planning/collectives.rs` reads those facts and selects:

- At most one workgroup per useful input tile, retaining the existing cap of 256.
- A power-of-two carry width covering those workgroups, between 32 and 256.
- The existing schedule for unknown input sizes and explicit grids.

Existing egglog rules derive launch geometry, partial/carry capacities and
dependencies from these choices. The shared lowering already supports these
sizes, so neither backend changes. Fixed fusion is retained. This is a small
work-count policy, not a calibrated hardware cost predictor.

The **full fluid build** receives the same change: the particle-sort chunk
grid becomes 68 groups, the carry width becomes 128, and each of its two
partial buffers and its carry buffer shrinks from 16,384 to 4,352 bytes.
The 100,000-key control retains 256 groups and a 256-thread carry stage.

## Measurement

Baseline: commit `57cf20b8`. Device: Radeon RX 580, AMD driver 23.11.1,
Vulkan. Both shader targets use `--subgroups`, matching the fluid launcher.
The benchmark generates the preset's Hilbert key/index pairs outside timing,
then calls the same 32-bit FourBit sort used by the simulator.

Three batches per size, five warmups per batch, then 20 measured calls per
compiler per batch. Before/after order alternates. Every result is checked
against a stable CPU sort; input buffers remain unchanged. GPU timestamps span
the generated sort call, including its clears/copies/dispatches, but exclude
pipeline setup, input generation, readback and verification. Resident wall time
also includes CPU encoding, submission and GPU completion. Vulkan validation
layers are disabled for timing; compiler SPIR-V is checked with `spirv-val`.

| Target | Keys | GPU before | GPU after | Resident before | Resident after |
|---|---:|---:|---:|---:|---:|
| WGSL | 4,320 | 1.14880 ms | 0.57096 ms | 1.83245 ms | 1.26055 ms |
| SPIR-V | 4,320 | 1.16072 ms | 0.57504 ms | 1.83595 ms | 1.24915 ms |
| WGSL | 100,000 | 1.93360 ms | 1.93296 ms | 2.71670 ms | 2.72695 ms |
| SPIR-V | 100,000 | 1.95264 ms | 1.94680 ms | 2.74985 ms | 2.71610 ms |

These are measured sorting-stage improvements, approximately 0.58 ms saved per
default-size sort. They are not full-frame timings: the separate serial
tree-building path still needs work.

## Validation

Planner tests check matching chunk grids, carry widths and temporary capacities,
including empty, partial and multi-tile inputs. The fluid radix test checks its
68/128 schedule; unknown-size scans retain their previous defaults.

`scripts/test_collective_launch_gpu.py` checks prefixes and totals for an
associative, noncommutative affine-composition operator, across both targets and
both scalar optimization policies. Its 40 numerical cases passed. On this
machine the documented pre-existing Naga/Vulkan array-decoration issue requires
`--no-validate` for viewer execution; compiler SPIR-V validation remains enabled.
This does not count as passing Vulkan-layer validation.

Gates run on 2026-10-29:

- Workspace formatting, release all-target build and the full workspace test
  suite passed, including 1,357 core tests (14 existing ignores) and all compiler
  fixtures. Tests used a 64 MiB thread stack and a workspace-local temporary
  directory. Build attempts interrupted by Windows executable locks passed when
  rerun after the compiler/viewer-using checks finished.
- All nine README-listed standalone crate builds/tests and the browser-target
  WebAssembly check passed.
- Testfile validation passed 123 SPIR-V cases and 122 WGSL cases; the existing
  linked-helper miner exclusion accounts for the one WGSL skip. The complete
  fluid program also builds and validates on both shader targets.
- Forty scan/total GPU cases passed numerically, as did every warmup/measured
  stable-sort result in both benchmark runs. The four fluid step, tiled-neighbor,
  linked-neighbor and renderer checks passed with viewer layers disabled and
  separate compiler-SPIR-V validation.
- The spatial gate again timed out after 120 seconds in the pre-existing
  4,097-slot tree case. Its host program still dispatches the tree operations
  as single invocations. Together with the viewer's Naga/Vulkan decoration
  failure, this prevents claiming a clean full GPU gate set or a measured
  full-frame speedup.
