# Durable select intrinsic and conditional lowering

Status: implemented. The design below records the scope and rationale.

The internal `_w_intrinsic_select` now survives Egglog, host extraction, and SSA,
with a shared conditional-value query for fusion dependency analysis. Its
operands are eager and use WGSL order: false value, true value, condition.
Backend preparation converts private scalar/vector diamonds when all additional
work is safe to speculate and fits a four-operation budget. Unsafe and expensive
arms remain guarded. Both shader backends emit their native select operation.

At commit `309cf8e4`, the default local tinyporto SPIR-V build contained 500 `OpSelect`, 679
`OpBranchConditional`, 603 `OpSelectionMerge`, and 827 `OpPhi` instructions across
37 entry points; the 638,068-byte module passes Vulkan 1.3 validation. These are
code-generation counts, not a measured GPU speedup.

Focused regressions cover analysis/extraction, eager host evaluation, work
accounting, malformed types, nested and loop-local diamonds, guarded fallbacks,
multiple results, and floating-point bit preservation. The GPU regression script
`scripts/test_select_gpu.ps1` compares SPIR-V and WGSL outputs against a CPU
reference with algebraic optimization on/off and lengths 0, 1, 63, 64, 65, 257.
All 24 cases pass, including division guarded against zero. Direct compilation
is checked separately with scalar output because dynamic invocation-local array
allocation is unsupported by the direct path.

The initial tracked shader corpus validated with 115 SPIR-V passes and 114 WGSL passes;
WGSL skips the existing `miner.wyn` fixture because it links SPIR-V helpers.
Workspace and standalone Wasm tests cover the remaining compiler/host paths.
That core run passed 1,482 unit tests (16 existing ignores), its package
integration test, and four compile-fail documentation tests. The remaining
workspace crates, CLI integration tests, and three standalone Wasm tests pass.

Public syntax, vector masks, aggregate/wide types, and profitability tuning based
on GPU timings remain follow-up work.

## Early formation and select simplification

Safe value conditionals become the durable intrinsic during expression import,
after fusion and before scalar EqSat and placement. Both early and late
conversion use the same native-type contract, cheap primitive whitelist, and
four-operation budget. Early conversion proves the whole newly eager arm DAG,
counts shared computations once, and admits only literals and parameters as
leaves. Memory/execution results require availability proofs and remain opaque
to this early pass; the late SSA pass still handles additional opportunities.

Select-specific EqSat rules simplify boolean choices, negated predicates,
repeated predicates in nested choices, and projections through vector choices.
Complementary integer choices such as `(if c then a else b) + (if c then b else a)`
reduce to `a + b`. Typed comparisons whose two literal alternatives agree can
become constant even when the selected value is dynamic. Thus choosing 4 or 8
and testing for zero exposes an unreachable branch to the existing SSA cleanup
introduced in `bab89169`, including loops inside that branch.

Rules which remove eager operands require an unconditional proof that the whole
expression can be discarded. Facts from one branch use are never installed as
global equalities for a shared expression. Partial operands and partial
condition computations retain their evaluation. Correlated arithmetic rules
are restricted to integers; floating-point operand order is preserved.

Select alternatives run for four rounds alongside saturated constant folding;
optional algebra has four additional rounds. This bounds exploration through
cyclic e-classes. Extraction decides whether projection alternatives are cheaper.
The GPU fixture exercises correlated choices, an unreachable loop, and guarded
division together, with algebra enabled and disabled on both shader targets.

Host scalar arithmetic uses the same wrapping integer and IEEE floating-point
semantics as device arithmetic. Its typed WHL operations and generated Rust
therefore tolerate overflow in a newly eager, unchosen arm. General WHL capacity
arithmetic remains checked.

Select recognition uses the typed builtin signature without joining all typed
representations of each operand e-class. Those redundant joins caused a
Cartesian-product slowdown during scalar saturation; a regression exercises
256 equivalent terms per operand. Compile-time comparisons use separate compiler
binaries with their optimization behavior checked against the select fixture.
The fixed tinyporto control input, compiled in release mode with graphics,
algebra, SPIR-V and Rust host output enabled, took 4.24/4.10 seconds versus
4.92/5.26 seconds at `bab89169` in alternating runs. Arithmetic saturation took
289/322 milliseconds. Its validated SPIR-V changed from 491 to 336 selects,
640 to 601 conditional branches, and 616,792 to 517,180 bytes. These are local
compile-time and code-size measurements, not GPU timing results.

Validation passes for the release workspace suite (1,496 core unit tests, 16
existing ignores, the package integration test, and four compile-fail doc tests),
three standalone Wasm tests, and all 24 GPU cases. The tracked shader corpus
passes 116 SPIR-V and 115 WGSL validations; WGSL retains the existing `miner.wyn`
skip for its linked SPIR-V helpers.

Introduce an explicitly eager, durable select intrinsic, then use it to replace
eligible value-producing conditionals in shared SSA. Preserve conditional-result
information in analyses without treating eager operand evaluation as conditional
execution.

## Starting point

- `egglog/to_ssa/values.rs` lowers every `ExprKind::If` to a selection header,
  two arms, and a merge block parameter. Its `select` helper also constructs a
  diamond even though its arguments are already values; boolean-to-number
  conversion calls this helper.
- SPIR-V emits the CFG and merge parameters as branches and phi instructions.
  WGSL structurizes it and emits `if` statements assigning merge variables.
- `ssa::optimize` already floats operations that are safe to speculate;
  `place_floating` gives them concrete blocks. Consequently, some remaining
  diamonds can be removed without evaluating any additional operand work.
- `BuiltinLowering::is_speculatable` distinguishes total operations from merely
  pure ones. Dead-code discardability and dominance-based reuse are separate
  permissions and must stay separate.
- Egglog's existing constructor named `Select` represents **lazy** `ExprKind::If`.
  It is not an eager intrinsic, despite its name.

## 1. Establish a durable operation

Use the existing builtin catalog: one internal builtin identity with
`BuiltinLowering::PrimOp(PrimOp::Select)`. Carry it through the existing
`ExprKind::PureApp` and `InstKind::Op { tag: OpTag::Intrinsic, ... }` machinery.
This avoids adding a second, parallel SSA operation identity for the same
intrinsic. Provide typed construction and recognition helpers so analyses do not
depend on a string name or manually unpack argument positions.

Use `select(false_value, true_value, condition)` consistently, matching WGSL.
The condition is initially scalar `bool`; both value operands and the result
have exactly the same supported type. A future vector-mask overload can be
added separately. The intrinsic chooses an existing value; it is not `mix`,
arithmetic masking, or a short-circuit operation.

Keep it compiler-internal for the first implementation. A public Wyn `select`
can subsequently expose this same identity and document eager evaluation; public
syntax is not needed to optimize existing programs.

Rename the existing lazy Egglog constructor to `IfThenElse`, updating its schema,
import/export, rules, and tests. Do not reinterpret existing `Select` expressions
as eager operations. The new intrinsic remains identifiable through its catalog
identity in the pure application representation.

The primitive itself is pure, discardable, reusable, and speculatable **on
already available operands**. Those properties do not prove that its operand
expressions may be moved or newly evaluated. Extend type/arity validation,
constant evaluation, printing, builtin metadata, and serialization paths as
required by the existing intrinsic machinery.

## 2. Preserve the analysis information currently attached to if

Introduce a small conditional-value query returning condition, true value, false
value, and evaluation mode (`Lazy` or `Eager`). It recognizes both `ExprKind::If`
and the select builtin. Use it only where an analysis needs value-choice
semantics; structured control-flow analyses should continue recognizing actual
conditional regions.

| Consumer | Required treatment |
| --- | --- |
| `egglog/fusion/analysis/summary.rs` | Recognize select before the generic `PureApp` case and preserve `sink.choice` / `ChoiceDependency`, including field projections. Its current generic path uses `sink.all` and loses this structure. |
| `egglog/visit.rs`, expression dependency rules, reference/capture discovery | Both forms reference condition and both values. Existing `PureApp` traversal covers select, but add coverage to prove all inputs survive. |
| `egglog/planning/expressions.rs`, size and residency consumers | Preserve all input dependencies and any conditional-result facts the analysis already understands. Do not invent mutually exclusive execution for eager operands. |
| `egglog/scalar/hoist.rs` and `hoist/analysis.rs` | Select is ordinary eager dataflow. Keep lazy-arm common-work placement specific to lazy if; prove availability and safe motion across the entire moved operand graph. |
| `egglog/execution/work.rs` | Charge select plus both operand computations, counting shared work according to placement. This file models code work, not expected branch latency; do not repurpose it as a GPU cost model. |
| `arithmetic.egg`, `egglog/scalar/fold.rs`, SSA constant folding | Add constant-condition and identical-value simplifications with eager-evaluation accounting. Replacing a result must not silently erase required operand evaluation; remove producers only under the existing discardability rules. |
| `egglog/host.rs`, host scalar IR/emission/interpreter, `schedule_test_exec.rs` | Support the eager operation when it reaches host extraction or evaluation. Evaluate operands before choosing; do not translate their expression trees directly to lazy `ScalarExpr::If`. Account for the boolean condition's different type. |
| TLC ownership, stage extraction, conditional producer normalization; Egglog region scheduling | Keep branch-region handling specific to `TermKind::If` / `OperationKind::If`. Internal select has value operands, not alternative effectful regions. Verify ordinary builtin traversal and typing suffice. |
| SSA reuse, DCE, floating placement, backend validation | Reuse the catalog-based intrinsic handling, adding explicit select validation/folding where needed. Selecting values does not grant permission to move their producers. |

An analysis can share the equation “the result is one of these two values”
without sharing the assumption “only one operand expression executes.” In
particular, facts implied by the condition may describe the selected result but
must not justify evaluating either operand unconditionally.

## 3. Implement both backend mappings and the first producer

- SPIR-V: add `PrimOp::Select` handling in `spirv/lower_ops.rs`, emitting
  `OpSelect(result_type, condition, true_value, false_value)`. Add a typed builder
  method in `wyn-spirv` consistent with the builder's preferred typed API.
- WGSL: add handling in `wgsl/ssa_lowering.rs`, emitting
  `select(false_value, true_value, condition)` through normal SSA bindings.
- Replace the boolean-to-number helper diamond in `egglog/to_ssa/values.rs`
  with the intrinsic for supported result types. Preserve guarded conversions:
  an arbitrary conversion formerly performed in an arm cannot automatically be
  performed before select.
- Start automatic conversion with the common boolean/numeric scalar and vector
  types supported by both targets. Centralize the eligibility check. Keep tuple,
  struct, matrix, array, view, pointer, and opaque-handle choices as branches.
  Add explicit coverage before accepting native wider SPIR-V values or WGSL
  emulated `u64`; do not treat an emulated physical type as an ordinary source
  scalar without checking its legalization.

WGSL select supports scalars and vectors and evaluates its arguments eagerly.
SPIR-V permits additional composite result types from version 1.4; Wyn's builder
currently emits version 1.5. Composite support can be added later, with bounded
field/column selection for WGSL and explicit handling of physical representations.
There is no need to increase the SPIR-V version for the initial implementation.
See the [WGSL select specification](https://www.w3.org/TR/WGSL/#select-builtin),
[WGSL builtin evaluation rules](https://www.w3.org/TR/WGSL/#builtin-functions), and
[SPIR-V OpSelect specification](https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpSelect).

## 4. Add conservative shared SSA if-conversion

Add `ssa/if_conversion.rs`. Initially invoke it from a common preparation helper
used by both `prepare_spirv` and `prepare_wgsl`, after floating placement and the
initial dead-value cleanup, before publishing texture requirements and final
backend validation. Keep WGSL's addressable-constant promotion in its existing
preparation ordering. Cover helper functions, entry points, and constant bodies.

This late insertion preserves existing upstream branch analyses while the new
intrinsic is made durable throughout the compiler. Moving automatic conversion
earlier can be considered after analysis parity and measurements are established.

First accept only canonical acyclic diamonds: a selection header, private arms
without nested control flow, and a common merge with no unrelated predecessors.
Require no effects, calls, memory access, or other required work in the arms.
All selected values must already dominate the header. This captures constants,
parameters, and arithmetic already placed above the branch with zero additional
speculation. Convert all live merge results together; if any is unsupported,
leave the diamond intact.

Then extend to short straight-line arms whose entire newly executed dependency
graph is safe and cheap to speculate. Reuse and audit the existing speculation
policy instead of equating purity or discardability with safety. In particular,
the SSA binary-operator policy and the intrinsic policy differ for shifts today;
unify this decision before letting a new pass rely on it.

Keep guarded division/remainder, domain-sensitive math and conversions, dynamic
indexing, loads/stores, atomics, barriers, texture operations, derivatives, and
unknown calls in control flow. A hazardous computation already dominating the
header is different: choosing its existing result introduces no new evaluation.

Make profitability a separate check. Begin with a named budget of four added
cheap scalar-equivalent operations per converted selection, counting shared
dependencies once and vector width conservatively. Exclude expensive math even
when it is speculatable. Calibrate this threshold with measurements; branch
removal alone is not evidence of a speedup.

Move eligible instructions in dependency order, emit one select per differing
merge result, and substitute identical results directly. Update placements,
block-parameter definitions, argument lists, and selection metadata. Remove only
private arm blocks; never remove another construct's merge/continue target.
Eliminate trivial single-predecessor merge parameters so WGSL does not retain
unnecessary merge variables and SPIR-V does not retain trivial phi nodes.

Permit diamonds within loops, but keep all moved work inside the same loop and
iteration. Preserve enclosing guards, loop headers, exits, and continue blocks.
Recompute invalidated dominance information between conversions, and finish with
folding/dead-value cleanup. A bounded worklist can expose nested opportunities
without repeatedly scanning the entire program.

Keep structural-target checks explicit when removing private arm blocks. Reuse
the SSA substitution and dead-value cleanup APIs for value replacement and
instruction deletion. Constant conditions and identical alternatives fold
without retaining unnecessary selects.

## 5. Validate semantics, analysis parity, and output

1. Intrinsic tests: operand order, both condition values, scalar/vector types,
   validation failures, constant folding, and selected floating-point bit
   patterns. Verify serialization/extraction round trips retain eager select.
2. Analysis tests: equivalent safe if/select expressions retain conditional
   fusion dependencies and capture/size inputs. Verify select counts both
   operands and never inherits lazy branch guards. Cover host extraction and
   interpretation with eager operand evaluation.
3. SSA tests: constant/already-available operands, cheap arms, multiple results,
   identical results, nested selections, and loop-local diamonds. Negative cases
   cover unsafe/expensive arms, shared blocks, effects, unsupported result types,
   structural targets, and possibly empty loops.
4. Backend tests: the targeted diamond becomes `OpSelect` / WGSL `select`, with
   no corresponding branch/phi/merge variable. Validate SPIR-V and WGSL with the
   existing validators. Preserve tests that require guarded execution. Update
   `vertex_conditional_is_evaluated_once_for_nested_varyings` to test one shared
   conditional result rather than requiring one `OpSelectionMerge`.
5. End-to-end gates: relevant crate tests, workspace checks, separate `wyn-wasm`
   tests, and `scripts/validate_testfiles.ps1` in SPIR-V and WGSL modes. Exercise
   optimized/unoptimized and scheduled/direct compilation paths. Compare GPU
   results for representative safe conversions and guarded fallback cases.
6. Measure a fixed local shader corpus, including the existing tinyporto fixture
   when available: conditional branches, phi/merge variables, selects, generated
   size, compile time, and representative GPU timings. Read submodule fixtures
   only; keep generated artifacts outside submodules. Use an internal test option
   to disable conversion for before/after comparisons.

The implementation includes durable intrinsic and analysis/backend support,
boolean conversion, zero-additional-work SSA conversion, and budgeted arm
speculation, plus early expression conversion and select-aware EqSat. Broader
aggregates, vector masks, and public syntax require additional eligibility and
evaluation tests.
