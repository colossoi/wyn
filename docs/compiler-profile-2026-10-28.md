# CLI compiler phase timings

Follow-up: [detailed Egglog and backend timings](compiler-profile-fine-2026-10-28.md)
separate storage-rule execution, evaluation planning, extraction, and SPIR-V
emission. The measurements below retain the original coarse baseline.

Measured on October 28, 2026. Tinyporto compiles in **7.13 seconds**, with
**78% of process time in Egglog through SSA generation**. The fluid simulator
compiles in **3.93 seconds**, spending **49% in Egglog and 38% in the SSA/SPIR-V
backend**. AST and TLC are smaller contributors in both programs.

These are compile-time measurements of each project's normal SPIR-V build.
They exclude Rust host compilation, GPU pipeline creation, and application execution.

## Coarse results

Median milliseconds across three fresh CLI processes after one warmup:

| Phase | Tinyporto with `-O` | Fluid with `--subgroups`, without `-O` |
| --- | ---: | ---: |
| AST loading and parsing | 45 | 20 |
| AST semantic frontend | 284 | 64 |
| TLC lowering and transformations | 473 | 102 |
| Egglog through SSA generation | 5,581 | 1,921 |
| SSA preparation and SPIR-V generation | 187 | 1,496 |
| Artifact writing and host emission | 504 | 277 |
| Other process time | 62 | 46 |
| **Total process wall time** | **7,127** | **3,928** |

Tinyporto's measured wall times ranged from 7.106 to 7.202 seconds; fluid's
ranged from 3.927 to 3.937 seconds. Columns contain independent phase medians,
so their sum can differ slightly from the median total.

`load_modules` includes frontend/prelude initialization, reading the imported
source closure, and parsing. `type_check` includes import resolution, module
elaboration, name/resource resolution, AST constant folding, and type checking;
it is not an isolated type-inference measurement. TLC is the sum of the 14
sequential timers from `to_tlc` through `apply_ownership`.

`lower` includes SSA storage preparation, floating-expression placement,
reachability, backend legalization/validation, SPIR-V emission, and host program
construction. `write_artifacts` includes host emission, so `host_emit` must not
be added again. Other process time is external elapsed time minus the six
non-overlapping measured groups; it includes startup/shutdown, package
preparation, diagnostics, uninstrumented work, and integer-timer rounding.

The backend group also invokes Egglog: `ssa::prepare_spirv` calls
`local_storage::apply`, which builds separate function-local graphs from SSA
and runs storage-web, liveness, selection, and allocation rules. This work is
inside `lower`, not the timer named `egglog`. Querying the original compiler
graph to construct SSA is already included in the earlier Egglog-through-SSA
group. The coarse measurements therefore do not isolate all Egglog engine
time, and the 1,496 ms fluid backend figure cannot yet be attributed between
local-storage analysis, other SSA work, and SPIR-V emission.

## Where the time goes

| Selected Egglog phase | Tinyporto ms | Fluid ms |
| --- | ---: | ---: |
| Import | 255 | 92 |
| Structural analysis | 77 | 24 |
| Fusion | 240 | 770 |
| Placement | 445 | 210 |
| Scheduling | 619 | 349 |
| Interface preparation | 237 | 122 |
| Scalar optimization | 2,155 | 145 |
| Host preparation | 17 | 4 |
| Evaluation planning | 1,183 | 75 |
| SSA emission | 224 | 77 |

For Tinyporto, scalar optimization and evaluation planning together account for
about **3.34 seconds, or 47% of the whole compile**. Within scalar optimization,
extraction takes 1.226 seconds, including 419 ms computing compact costs and
666 ms computing inlined costs. The scalar rewrite fixed point takes 525 ms.
These sub-timers are nested and must not be summed with their parents.

For fluid, the largest single coarse group after Egglog is the **1.496-second
backend**. Within Egglog, fusion takes 770 ms, including 722 ms in its schedule.
The scalar optimizer is only 145 ms with the normal build options. The two
applications therefore suggest different next investigations.

Within TLC, helper inlining is the largest measured pass: 162 ms for Tinyporto
and 34 ms for fluid. Tinyporto's Rust host emission takes 207 ms inside its
504-ms artifact-writing group; fluid emits a WHL host in less than the timer's
one-millisecond resolution.

## Effect of scalar optimization

The control builds change only `-O` relative to their application's normal flags.
Each has one warmup and three measured runs.

| Program | Without `-O` | With `-O` | Added compile time |
| --- | ---: | ---: | ---: |
| Tinyporto | 4.705 s | 7.127 s | 2.422 s, 51.5% |
| Fluid | 3.928 s | 4.190 s | 0.262 s, 6.7% |

Tinyporto's scalar optimization rises from 533 to 2,155 ms and evaluation
planning from 576 to 1,183 ms. Its emitted SPIR-V grows from 434,568 to 533,280
bytes. Fluid's SPIR-V changes from 495,044 to 491,568 bytes. These observations
do not establish GPU runtime quality; they are not a recommendation to disable
`-O`.

## Reproduce

From the Wyn repository root in PowerShell:

```powershell
cargo build --release --locked -p wyn
New-Item -ItemType Directory -Force tmp/compiler-profile | Out-Null
$env:RUST_MIN_STACK = '67108864'
& .\target\release\wyn.exe build ..\tinyporto\wyn\main.wyn --graphics -O --target-double rust-wgpu --target spirv -v -o tmp/compiler-profile/tinyporto.spv
& .\target\release\wyn.exe build fluid-simulation --graphics --subgroups --target spirv -v -o tmp/compiler-profile/fluid.spv
```

Run each once to warm the filesystem cache, then measure three fresh processes
serially, redirecting diagnostics to files. Remove `-O` for the Tinyporto
control; add it for the fluid control. The existing `-v` timers were sufficient:
no compiler instrumentation or behavior changes were made for this investigation.

[Complete measurement evidence](profiles/2026-10-28-compiler.json) contains every
command, warmup and measured result, raw diagnostics, individual pass timings,
shader sizes/hashes, and the Python measurement harness. The local working
harness and generated shaders are also retained in the ignored
`tmp/compiler-profile-2026-10-28` directory.

## Environment and checkout state

Intel Core i7-9700 at its nominal 3.00 GHz, eight logical processors, Windows
build 26200; Rust 1.96.0. The compiler was built with `cargo build --release
--locked -p wyn`. Runs used warm filesystem caches, fresh compiler processes,
normal warning reporting, and captured verbose diagnostics. No other builds or
benchmarks were launched concurrently. CPU frequency and unrelated desktop
activity were not controlled; these are coarse wall-clock profiles, not CPU samples.

Wyn was fast-forwarded seven commits from `7ddcd513` to
`1e0269814a19c0e0edbae9847485c0546a82d3c9`, matching `origin/master`.
The pre-existing Egglog inspection/viewer edits, WASM changes, triangulation
package, patch, and tilt-shift example were preserved. Tracked edits also remain
backed up in the stash named `pre-upstream-2026-10-28-local-inspection-work`.
Submodules were not updated or edited. The release CLI includes the restored
inspection module, whose inspection entry points are not called by this path.

Tinyporto uses the existing sibling checkout at
`213c53c4e8c5f42c127feefeb530c6e9e430f921`, including its existing staged
`wyn/water_spectrum.wyn` edit. Its checkout was not updated or modified.
Fluid uses the newly pulled source in this repository, including the normal
`--subgroups` option from `fluid-simulation/run.sh`.

All 16 compiler invocations succeeded, and all four final SPIR-V variants
passed `spirv-val`. GPU execution, generated Rust host compilation, WGSL, and
the full compiler regression suite were not run for this profiling-only task.

## Next investigation

Start with Tinyporto's scalar extraction cost computation and evaluation
planning, which together offer the largest measured compile-time target.
For fluid, split `lower_ssa_to_spirv` into SSA preparation, placement,
`prepare_spirv` (especially its separate Egglog local-storage analysis), and
SPIR-V emission timers before choosing a backend algorithm to optimize. The
current coarse timer cannot distinguish those costs.
