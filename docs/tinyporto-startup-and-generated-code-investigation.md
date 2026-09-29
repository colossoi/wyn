# Tinyporto startup and generated-code investigation

These investigation notes were supplied on September 27, 2026. The measurements
describe the 30,128-instruction Tinyporto snapshot, not the unfinished expression-DAG
implementation in this branch. The targeted cleanup described here was an
experiment on generated SPIR-V; it did not change production compiler or app code.
See [the scalar-port checkpoint](spirv-reuse-wip-20260926.md) for the separate
implementation status and the reproducible source comparison.

## Startup

The host eagerly creates 17 compute pipelines, then warms graphics pipelines
before showing the scene. Temporal GI is the largest compute shader, at **7,292
reachable instructions**. It includes the reference path tracer even when
reference mode is disabled.

Splitting reference GI into a separate, lazily created pipeline is the strongest
structural opportunity identified by this investigation. All pipeline
constructors also use `cache: None`; persistent pipeline caching is worth
benchmarking.

These observations identify work to investigate, not measured startup savings.

## Generated-code cleanup

Targeted cleanup produced the following reduction:

| Metric | Original | Targeted cleanup |
| --- | ---: | ---: |
| Binary size | 574,652 bytes | 419,928 bytes |
| Instructions | 30,128 | 22,771 |
| Extract/construct instructions | 7,900 | 3,975 |

That is **24.4% fewer instructions**, with SPIR-V and Naga validation passing in
the reported experiment. Generic optimizer `-O` and `-Os` runs instead increased
code size through inlining. These flags refer to that SPIR-V optimization
experiment, not a recommendation to disable Wyn's own Egglog `-O` option.

The 7,900 figure counts **all** extract/construct instructions, not exclusively
redundant ones. The supplied notes do not include the exact targeted pass list,
optimizer version, or cleaned binary; preserve those when reproducing this
experiment before treating the cleanup result as a regression target.

## Concrete camera packing/unpacking example

Tinyporto's `wyn/camera.wyn` converts between two camera records:

```wyn
def shared_orbit(o: orbit) gfx.camera.orbit.state = {
  target = o.target,
  azimuth = o.az,
  elevation = o.elev,
  distance = o.dist,
}
```

Together with `camera_from_frame`, this produces the following sequence in the
analyzed temporal GI shader:

```text
%9209 = OpCompositeConstruct %249 %9205 %9206 %9207 %9208 %9196
%9210 = OpCompositeExtract %109 %9209 0
%9211 = OpCompositeExtract %5   %9209 1
%9212 = OpCompositeExtract %5   %9209 2
%9213 = OpCompositeExtract %5   %9209 3
%9214 = OpCompositeExtract %70  %9209 4
%9215 = OpCompositeConstruct %249 %9210 %9211 %9212 %9213 %9214
```

It constructs a camera record, extracts every field, then reconstructs the
identical record. `%9215` can reuse `%9209`, eliminating six instructions. The
four-field camera conversion subsequently repeats this pattern. The IDs above
are specific to the analyzed binary.

This is sensible Wyn source exposing a compiler simplification gap. These SSA
operations are not necessarily physical memory copies.

## Likely origin and why it survives

The likely origin is generic record conversion introduced after the earlier
simplifier has run. In
[`egglog/to_ssa/values.rs`](../wyn-core/src/egglog/to_ssa/values.rs), `value_state`
recursively normalizes records by:

1. Extracting each field.
2. Converting it to the required representation.
3. Constructing a record from the converted fields.

That is useful when fields contain arrays whose representations must change.
For a camera containing ordinary vectors and floats, however, each conversion
can be a no-op, leaving an unnecessary unpack/repack. The `cast` helper also
reconstructs records when their compiler-level types differ, even if their
eventual SPIR-V representations match.

The `field` helper emits an extraction without checking whether its input was
just constructed. The SPIR-V backend translates that extraction literally.
Earlier simplification rules cannot catch operations introduced afterward.

Ordinary common-subexpression elimination alone is insufficient:

```text
original = construct(a, b)
rebuilt  = construct(extract(original, 0), extract(original, 1))
```

The constructions have different operand IDs until aggregate simplification
proves the extracts equal `a` and `b`.

The working diagnosis is therefore a **pass-ordering and representation-conversion
gap**. The investigation did not trace `%9215` through intermediate compiler
dumps, so the exact conversion responsible remains unproven. The disassembly
demonstrates the redundancy; the compiler source explains plausible origins
and why it can survive.

## Other concrete opportunities

- **Reuse repeated math:** opaque lighting computes identical camera `sin`/`cos`
  expressions seven times within one basic block.
- **Forward aggregate fields:** simplify extractions from freshly constructed
  records, then eliminate unused constructions and exact reconstructions.
- **Reuse loop-invariant sizes:** generated GI loops repeatedly rebuild
  resolution-derived dimensions and array-view metadata.

Prefer exposing helper expressions to the Egglog DAG before SSA lowering.
Representation conversions introduced afterward still need construction-time
simplification or a narrowly justified later cleanup. Preserve type contracts,
floating-point evaluation order, runtime lengths, and guards on partial operations.

## Validation limits

The reported experiment validated SPIR-V and Naga output and measured static
size reductions. No GPU was exposed in that investigation environment, so it
does **not** establish a startup-time or frame-time speedup. GPU timing and
pipeline-cache experiments remain follow-up work.
