# WIP: SPIR-V reuse and the Egglog scalar port

## Current checkpoint: September 27

Work is on `codex/finish-egglog-scalar-port`, based on master `07e8693b`.
Master itself is unchanged. **This checkpoint is not ready to merge.**

The latest experiment moves scalar helper expansion before expression insertion
so the existing Egglog DAG, simplification, and placement can see the expanded
bodies. Scheduling also instantiates small pure callbacks into that DAG using
fresh parameter bindings for generated arguments. Helper expansion in `to_ssa`
has been removed. The SSA emission-cache experiment was removed as well.

Current validation and next steps:

- `cargo build -p wyn` passes for the latest source-DAG implementation.
- The focused camera/rotation placement tests passed after moving expansion out
  of SSA; the full suite has not been rerun on this latest implementation.
- The preceding port-on-master workspace run had 1,524 core tests pass and one
  failure: `mapped_array_length_does_not_require_its_discarded_elements` panics
  with `expected array: Discarded` in the scheduled test interpreter. That
  failure also reproduces with the initial source-DAG implementation.
- The first source-DAG Tinyporto build failed with a reachable expression
  referring to inactive `OperationId(718)`. A follow-up normalizes transitive
  call replacements before rewriting roots; it builds but has not yet been
  checked against Tinyporto. Check replacement coverage and binding identity
  before relying on it.
- Recheck scope/guard preservation, eager argument evaluation, callback argument
  types, and sharing across invocations. Finish DAG simplification and placement
  coverage, rerun Tinyporto and the workspace tests, and update the README and
  measurements before considering the port complete.

All measurements and passing full-suite results below describe earlier
checkpoints, not the unfinished source-DAG implementation above.

## Historical checkpoint

Status recorded September 26, 2026. The compiler implementation is commit
`1d3b9e86` ("Move scalar helper expansion into the Egglog SSA handoff").
The comparison baseline is `0cff30a2`, which already includes trivial phi
removal and WGSL lexical declarations.

This is a working checkpoint. Correctness and validation gates pass, but the
port currently emits more Tinyporto instructions than the baseline. Early SSA
constant folding and SSA dominance reuse remain deliberately disabled for the
comparison. Recovering that optimization coverage is unfinished work.

## What changed

- Small helper expansion runs after Egglog scheduling, during the SSA handoff,
  so it also covers callbacks created by scheduling.
- Egglog placement retains expressions evaluated by conditions, loop
  initializers, and direct-call arguments for reuse at dominated uses.
- Expression emission preserves values evaluated before a conditional while
  discarding branch-local cache additions at the join.
- Safe arithmetic exposed by inlining uses the existing floating-instruction
  scheduler during the handoff. This placement still operates on SSA data.
- The former SSA helper inliner and pure-expression interning were removed.
  Dynamic-index materialization and late backend preparation remain in SSA.
- The [README pass tables](../README.md#compiler-architecture) describe the
  ordering and the subpasses inside SSA orchestration.

The implementation commit changes production Rust by **+256/-258, net -2
lines**, excluding test files. Its complete diff, including tests and README,
is +458/-615, net -157 lines.

## Measurements and validation

Counts include each reachable function once for the temporal GI comparison;
whole-module counts include the entire shader module.

| Reproducer | Baseline `0cff30a2` | WIP `1d3b9e86` |
| --- | ---: | ---: |
| Map, whole module | 84 | 84 |
| Tinyporto, whole module | 27,722 | 30,171 |
| Tinyporto temporal GI and reachable functions | 7,108 | 7,304 |

Both current reproducers have **zero single-input phis**. The original map
report counted 85 instructions before phi removal; the target remains 81.
The Tinyporto source is the saved `tmp/temporal-gi-20260925/control/` snapshot
in the sibling Tinyporto checkout. The original categorization is in that
checkout's `docs/spirv-reuse-categories-20260925.md`.

The map reproducer is:

```wyn
entry repro(xs: []f32) []f32 = map(|x| x + 1.0, xs)
```

Validation completed for the implementation commit:

- Workspace: 1,638 passed, zero failed, 18 ignored; the core suite accounts
  for 1,500 passed and 16 ignored.
- Tracked SPIR-V corpus: 116 passed.
- Tracked WGSL corpus: 115 passed, one expected skip.
- WebAssembly release check, formatting, and diff whitespace checks passed.
- 32 headless GPU cases passed across SPIR-V and WGSL, covering loop swaps,
  nested loops, guarded division, array state, and zero iterations.

The additional call-argument sharing reduced an intermediate WIP result from
32,716 to 30,171 whole-module instructions. That improvement does not erase
the remaining regression against the baseline. Fewer outlined functions and
changed sharing make instruction count alone insufficient to assign the
entire difference to one pass; representative duplicates still need tracing.

## Can duplicates be prevented at the lowering boundary?

**Yes. Generation-time elimination should be the first experiment for the
operations that lowering itself creates.** It can avoid generating work whose
operands or result are already known, before considering a later reuse pass.

### Keep known view lengths available

[`length_cached`](../wyn-core/src/egglog/to_ssa/values.rs) already handles
static lengths and scheduled view-length metadata. Its fallback emits a
`length` intrinsic returning `i32`, followed by a conversion to `u32` for
generated indexing. The [SPIR-V length handler](../wyn-core/src/spirv/lower_builtin.rs)
extracts a view's `u32` length field and converts it to the intrinsic's result
type. Repeating this route can create repeated extracts and casts.

First check whether the view's length is already available at the call site.
Forward that existing value, or reuse a previously emitted query for the same
bound view, instead of reconstructing the query and conversions. Preserve
runtime live lengths for bounded arrays: capacity is not a substitute for
length. Do not change signedness or conversion semantics merely to remove an
instruction. The immediate acceptance target is the map's **84 -> 81**.

### Simplify projections while constructing instructions

The handoff's `field` helper currently emits a projection. When its input is a
known tuple or vector construction, it can instead return the corresponding
existing operand, subject to matching types. Similarly, an exact reconstruction
of a value's components can return that value rather than emit another
aggregate. These are candidates for small construction helpers, not a new
whole-program representation.

Some related simplifications already exist: `cast` returns an unchanged value
when its type already matches, and Egglog has basic projection-of-constructor
rules. The missing cases to investigate are those introduced after those rules
have run, especially by helper expansion and representation conversion.

### Share identical generated operations where they are emitted

[`Body::op`](../wyn-core/src/egglog/to_ssa.rs) currently appends an instruction
for each request. The expression cache is keyed by source `ExprId`, so it does
not automatically share different lowering requests that produce the same
typed operation on the same SSA operands.

A narrowly scoped emission helper could return an existing result keyed by
operation, result type, and operands. Start with duplicates in one block, or
values explicitly known to dominate the use. This is reuse during construction
and still needs a scope/lifetime rule; a function-wide unqualified cache would
be incorrect across sibling branches or different bindings. Reusing a partial
operation's already evaluated result must not move its first evaluation outside
its guard. Loads, opaque calls, and context-dependent intrinsics stay excluded.

If an operation is emitted only once in SSA but its backend expansion repeats
the same view-field extraction or cast, address that expansion at the backend
boundary. Establish where the duplication first appears before adding caches
at both boundaries. Keep any new machinery within the code-reduction goal.

### Keep later-created duplicates separate

[Late if-conversion](../wyn-core/src/ssa/if_conversion.rs) creates selects and
brings arm computations together after the Egglog handoff. An emitter cache
used only at the earlier boundary cannot see those changes. Check whether the
same construction helper can serve that pass, or whether a small late reuse
pass is justified by the residual cases. The current evidence does not establish
that generation-time elimination replaces every later cleanup, nor that all
reported duplicates require a later pass.

## Status and remaining work

1. **Trivial phi removal: done.** Commit `0cff30a2` replaces single-input block
   parameters with their dominating incoming values while retaining control
   structure. Current measurements confirm no single-input phis in either
   reproducer. Keep the existing loop and WGSL scope regressions as coverage.

2. **Repeated calculations and generated view handling: partially done; next.**
   Helper expansion and some guarded and call-argument sharing have moved to
   Egglog. The 29 reported selects, neighbor-index arithmetic, and generated
   extracts/casts are not all resolved. First test the generation-time changes
   above, starting with view length and the map. Trace remaining examples at
   the scheduled representation, emitted SSA, after if-conversion, and SPIR-V
   expansion. Recover the Tinyporto baseline before claiming further savings.
   Decide the remaining late reuse work from those results; do not simply
   restore every disabled pass or accept an instruction-count assertion change.

3. **Aggregate simplification followed by reuse: pending.** Extend exact
   extraction/construction simplifications and canonicalize equivalent record
   fields and reconstructed vectors. Apply them where the relevant aggregates
   become visible, then reuse the exposed calculations. Test the camera example
   against the seven redundant normalizations in the original report. Preserve
   floating-point evaluation order and the guards on partial operations.

4. **Unused-component elimination: pending.** Propagate component demand
   backward through projections, constructors, elementwise operations, and
   loop state. Remove calculations feeding only unused components, including
   the reported clamps, divides, and remainders. Respect cross-component
   dependencies such as normalization and dot products, and retain effects.

5. **Load reuse: deferred, last.** Start with proven read-only storage and
   identical addresses. Require an available dominating load and proof that
   writes, aliases, calls, or synchronization cannot invalidate the result.
   The three reported load removals offer less benefit than the preceding work.

## Continuation: generated view lengths

The handoff now forwards the existing `u32` length operand of a constructed
`StorageView` when generated code asks for its length. This is a local
constructor simplification: it introduces no cache and moves no evaluation
across control-flow boundaries. Other representations and non-`u32` operands
retain the existing intrinsic/conversion path; bounded-array capacity is never
used as a live length.

The map reproducer now emits **78 whole-module instructions**, down from 84.
Both its loop bound and its output view reuse the input view's length, removing
two composite extracts and four bitcasts. The new SPIR-V regression checks this
result, and construction coverage checks zero, `u32::MAX`, and runtime lengths.
Existing bounded-array and rebound-array tests retain coverage of live lengths.
Validation: `cargo test --workspace` passed (1,640 passed, zero failed, 18 ignored);
`cargo fmt --all -- --check` and `git diff --check` also passed.

## Restored Tinyporto comparison

The untracked scratch snapshot is gone, but Tinyporto commit
`32ec4b8aa7765d0cf1e5a293875667a4c4db1497` with Wyn packages from
`0cff30a2dd5b13f52baa7332c0dd7a269b3cd2f5` reproduces both historical baseline
counts exactly. The source and dependencies can now be exported directly from
Git using [compare_tinyporto.py](../scripts/compare_tinyporto.py), without changing
either repository's working tree. The old categorization document remains absent.

Build the baseline compiler from a Git archive of `0cff30a2` and retain its `wyn`
binary separately from the compiler being measured. Then run:

```bash
python3 scripts/compare_tinyporto.py \
  --tinyporto-repo ../tinyporto \
  --output-dir /tmp/tinyporto-comparison-new \
  --compiler baseline=/absolute/path/to/baseline/wyn \
  --compiler current=/absolute/path/to/current/wyn
```

Use a fresh output directory. The script pins source and package revisions,
compiles with `--graphics -O --target spirv --target-double rust-wgpu`, saves
compiler logs and artifacts, and records whole-module and per-entry reachable
function counts in `counts.json`. Reachable functions are counted once per entry,
including function headers, parameters, labels, and ends. The temporal GI entry
is `tinyporto_frame_compute_5_compute_4`.

| Compiler | Whole module | Temporal GI and reachable functions |
| --- | ---: | ---: |
| Historical baseline `0cff30a2`, reproduced | 27,722 | 7,108 |
| Master `07e8693b`, before port | 28,061 | 7,235 |
| WIP `1d3b9e86` plus generated length forwarding | 30,128 | 7,292 |
| Port and length forwarding on master `07e8693b` | 30,466 | 7,424 |
| Discarded SSA construction-cache experiment on master | 27,005 | 6,090 |

The newer master also contains intervening compiler fixes, so comparisons must
use the corresponding baseline. The construction-cache experiment established
that generated duplicates account for much of the regression, but was removed
in favor of using Egglog's expression DAG before SSA lowering. The length
change improves the WIP's historical 30,171 / 7,304 counts by 43 / 12 instructions.
The remaining instruction-count regression is unresolved; these measurements do
not establish a runtime performance regression or improvement.
