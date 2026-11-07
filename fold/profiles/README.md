# Native Fold profile — 2026-11-07

## Fix: native CPU execution

The default solver now executes the Wyn-generated kernel as native CPU code,
matching upstream’s CPU-simulation/GPU-rendering architecture. The generated
control flow and equations are retained through a Clang-vector adapter; no
handwritten replacement solver, reduced iteration count, or timestep change
is involved.

- Same six-step checkpoint: 1.2–1.5 ms/step, versus 109–116 ms on the GPU.
- 60 interactive frames at 360: 54.3 FPS, 1.277 ms average solver step.
- 60 interactive frames at 900: 55.2 FPS, 1.276 ms average solver step.
- Five upstream fixtures pass, including nonbinary weights and all 2,483 particles.
- Workgroup-position caching was tested and rejected: about 160 ms/step.

See `cpu-360.json` and `cpu-900.json` for the interactive measurement summaries.
The earlier GPU measurements and inspection below explain the change. Use
`--gpu-solver` when reproducing those historical measurements; `--profile-copy`
is only meaningful with that flag.

## Original GPU bottleneck

AMD Radeon Pro 570. Native checkpoint at simulation time 2.0 seconds, 572 active
particles (44 rows). Each sample advances six 1/120-second steps from the same
checkpoint. Every simulation process retains its five-second SIGKILL deadline;
shader compilation occurs before that deadline starts.

These are synchronized CPU wall-clock timings. This adapter does not expose
WGPU timestamp queries, so there is no independent GPU-counter duration. The
copy-only baseline separates transfer/readback cost from constraint computation.

| Measurement | Time |
| --- | ---: |
| Host prepare | 0.009–0.025 ms/step, apart from initial allocation noise |
| Solver + upload + readback, 360 render size | 109–116 ms/step |
| Solver + upload + readback, 900 render size | 109–114 ms/step |
| Copy-only upload/readback baseline | 1.8–2.0 ms/step warm; first 3.9 ms |
| CPU shell generation | 1.4–1.6 ms/frame |
| Render + GPU completion, 360 | 12.4 ms |
| Render + GPU completion, 900 | 9.5 ms |
| Full scalar optimization (-O), solver | 117–123 ms/step |
| Remove redundant zero initialization, solver | 105–115 ms/step |

The render figures are individual warm samples, not a resolution scaling curve.
The clear conclusion is that simulation dominates. Six solver steps cost about
0.67 seconds before rendering. The interactive accumulator caps elapsed time to
0.05 seconds and consequently performs up to six fixed steps per displayed frame.
It cannot keep pace with the authored 120 Hz simulation at these kernel timings.

## Inspected code

`solver.spvasm` is SPIR-V exported from the exact storage-lowered WGSL via Naga
27, then disassembled with spirv-dis. `solver.metal` is Naga's Metal-source
export of the same module. These are inspection artifacts, not AMD machine ISA.
Native execution uses WGPU/Metal, not this exported SPIR-V. Default standalone
Naga export options need not match every internal WGPU backend option.
The installed command-line developer tools lack `metal`/`metal-objdump`, so no
claim is made about actual AMD ISA, register allocation, or backend load CSE.

Observed issues:

1. **A single invocation does all work.** `@workgroup_size(1,1,1)` and one
   dispatched workgroup serialize every edge/contact constraint. The hot
   positions, lambdas and hash arrays are in a 177,824-byte storage allocation.
   This preserves Gauss–Seidel ordering, but leaves dependent storage accesses
   with essentially no other active work to hide latency.
2. **Inactive edges remain in the loop.** All 13,880 edges are visited four times
   (55,520 visits/step). Only 3,149 connect active particles at the measured state:
   77.31% are irrelevant. The generated code loads both endpoint vec4s before
   checking the active count, unlike upstream's short-circuit rejection.
3. **Repeated loads before either endpoint write.** The generated active edge
   block loads each endpoint vec4 six times (12 loads for two old values), plus
   two loads of the same lambda. The Metal export retains these loads. A driver
   may eliminate some; this is not a claim that all reach hardware memory.
4. **Invariants are sunk into hot loops.** Bend compliance `pow(10,...)` is inside
   the edge loop; exclusion-radius `ceil(thickness/spacing)` is inside linked-list
   neighbor traversal. Row division and cells loads are repeated there too.
5. **Double initialization.** The storage lowering faithfully adds the zero
   initialization required by WGSL local declarations, then emitted initializer
   loops overwrite each full array. About 260 KB of unnecessary zero stores per
   step result. A temporary variant omitting just these zero fills reduced solve
   time by only about 6 ms; it is not installed as the production shader.
6. **Backend loop guards.** Naga's Metal export adds a uint2 loop-bound counter,
   decrement and check to each loop. This is backend safety lowering, not an
   accidental large-array copy. It is additional scalar work in the nested loops.

Static SPIR-V counts: 244 OpLoad, 132 OpStore, 127 OpAccessChain, 25 OpLoopMerge,
28 OpFunctionCall, and zero OpCopyMemory/OpCopyMemorySized. Counts are static
instructions, not executed traffic. This confirms that the former whole-array
copy issue is not present; scalar memory behavior and serialization remain.

## Next changes worth pursuing

- Restrict traversal to the active edge prefix while preserving its original
  forward/reverse ordering; prove omitted edges are all no-ops.
- Hoist timestep/stiffness/contact constants and snapshot each old endpoint once.
- Put hot positions in lower-latency memory. At default maximum size, packed XYZ
  needs 29,796 bytes and could fit a 32 KiB workgroup allocation, unlike vec4's
  39,728 bytes. This needs careful compiler/driver allocation work and validation
  of weights/error reporting; it is a proposed optimization, not implemented.
- Preserve sequential update semantics. Simply parallelizing dependent edges
  changes the solver; reducing iteration counts or fixed-step frequency also
  changes the authored simulation.

## Reproduce

```sh
python3 fold/run.py --headless --profile --steps 6 --size 360 --load-state /tmp/wyn-fold-state.json
python3 fold/run.py --headless --profile --profile-copy --steps 6 --size 360 --load-state /tmp/wyn-fold-state.json
fold/tools/validate-wgsl/target/release/fold-validate-wgsl fold/generated/constraints.wgsl --spirv /tmp/fold.spv --msl /tmp/fold.metal
spirv-dis /tmp/fold.spv -o /tmp/fold.spvasm
```

Create checkpoints using the instructions in the project README. `--profile-copy`
is a diagnostic baseline that bypasses constraints; it must not be used as a
simulation. `FOLD_SHADER` can select a temporary WGSL variant for investigation.
Default simulation behavior is unchanged.
