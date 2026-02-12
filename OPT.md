# SPIR-V Output Optimization Opportunities

Analysis of SPIR-V output from the Wyn compiler, identifying patterns that produce suboptimal code.

## Current state (post vec-scalar splatting)

After adding vec-scalar arithmetic splatting and `mul()` → `*` unification, testfile stats:

| Testfile       | Lines | Construct | Extract | VecTimesScalar |
|----------------|-------|-----------|---------|----------------|
| seascape       | 848   | 58        | 140     | 14             |
| kuko           | 486   | 30        | 51      | 2              |
| da_rasterizer  | 808   | 92        | 57      | 8              |
| entrylevel     | 589   | 52        | 108     | 5              |
| lava           | 424   | 22        | 58      | 1              |

The high Extract:Construct ratios (especially seascape 140:58, entrylevel 108:52) indicate widespread scalarization — vectors are constructed then immediately decomposed for scalar ops.

---

## 1. Scalarized Vector Operations (highest impact)

The compiler decomposes vector operations into per-component scalar operations. Example from seascape:

```spirv
%81 = OpCompositeConstruct %v2float %72 %80    ; pack into vec2
%82 = OpCompositeExtract %float %81 0          ; immediately unpack .x
%84 = OpFMul %float %82 %float_2               ; scalar mul
%86 = OpCompositeExtract %float %81 1          ; immediately unpack .y
%87 = OpFMul %float %86 %float_2               ; scalar mul
%89 = OpCompositeConstruct %v2float %85 %88    ; repack into vec2
```

Should be: `OpVectorTimesScalar %v2float %81 %float_2`.

**Root cause**: The TLC→SSA lowering scalarizes tuples/vectors, generating per-element code. The SPIR-V backend then faithfully lowers each scalar op.

**Fix options**:
- (a) Keep vectors intact through SSA lowering (architectural, high effort)
- (b) SPIR-V peephole pass: detect construct→extract→scalar-op→construct and reassemble into vector ops (medium effort, post-hoc)
- (c) SSA-level optimization pass that recognizes parallel scalar ops on extracted components and merges them back (medium effort)

**Priority**: 1 — affects every shader, accounts for the majority of instruction bloat.

---

## 2. Redundant Construct-then-Extract

Closely related to #1. `OpCompositeConstruct` immediately followed by `OpCompositeExtract` on the same value. The construct is unnecessary; just use the original scalar directly.

```spirv
%50 = OpCompositeConstruct %v3float %a %b %c
%51 = OpCompositeExtract %float %50 0          ; == %a, just use %a
```

**Fix**: Peephole optimization — when extracting from a known construct, forward the original component. This is straightforward and can be done at either SSA or SPIR-V level.

**Priority**: 2 — high impact, medium difficulty. Would chain well with #1.

---

## 3. Duplicate Uniform Loads

Same `OpAccessChain` + `OpLoad` with identical operands repeated in the same basic block:

```spirv
%67 = OpAccessChain %_ptr_Uniform_v2float %10 %int_0
%68 = OpLoad %v2float %67
%69 = OpAccessChain %_ptr_Uniform_v2float %10 %int_0   ; duplicate!
%70 = OpLoad %v2float %69                              ; duplicate!
```

**Fix**: Basic-block-level CSE on AccessChain+Load pairs. Straightforward.

**Priority**: 3 — medium impact, low difficulty.

---

## 4. Trivial/Empty Blocks

Blocks containing only `OpLabel` + `OpBranch` (empty fallthroughs). Every function has at least one due to the SSA lowering emitting an entry block that immediately branches.

**Fix**: Simple CFG pass to merge blocks with unconditional branches into their successors (when the successor has a single predecessor).

**Priority**: 4 — medium impact, low difficulty.

---

## 5. Single-Predecessor Phi Nodes

`OpPhi` with one incoming edge — the phi is just the value itself.

**Fix**: Trivially replaceable with a direct reference. Can be done in a single pass.

**Priority**: 5 — low-medium impact, low difficulty.

---

## 6. Write-Once/Read-Once Function Variables

Variables that are stored to once and loaded once — an artifact of the SSA lowering's variable materialization.

**Fix**: mem2reg-like pass promoting these to direct SSA value references.

**Priority**: 6 — medium impact, medium difficulty.

---

## 7. Struct Pack/Unpack for Loop State

Loop bodies construct a struct with `OpCompositeConstruct` and then the loop header immediately extracts all fields with `OpCompositeExtract`. This is the loop state tuple being materialized.

**Fix**: Scalar replacement of aggregates — carry loop state as individual SSA values instead of packing/unpacking structs.

**Priority**: 7 — medium impact, medium difficulty.

---

## 8. Small Inlinable Functions

Many small utility functions (constant-returning, single-operation, field accessors) that could be inlined at the SPIR-V level. Especially notable in miner (SHA-256 round constants) and raytrace (material accessors).

**Fix**: Inline functions below a size/complexity threshold. Note: TLC already has an inlining pass; this may indicate functions that escape it.

**Priority**: 8 — low-medium impact, medium difficulty.

---

## Priority Summary

| # | Optimization                       | Impact    | Difficulty                         |
|---|------------------------------------|-----------|------------------------------------|
| 1 | Scalarized vector ops              | Very High | High (SSA-level redesign)          |
| 2 | Construct-then-extract elimination | High      | Medium (peephole on SSA or SPIR-V) |
| 3 | Duplicate uniform load CSE         | Medium    | Low (basic block CSE)              |
| 4 | Trivial block merging              | Medium    | Low (simple CFG pass)              |
| 5 | Single-pred phi simplification     | Low-Med   | Low (trivial rewrite)              |
| 6 | Write-once var elimination         | Medium    | Medium (mem2reg-like)              |
| 7 | Loop state scalar replacement      | Medium    | Medium                             |
| 8 | Function inlining                  | Low-Med   | Medium                             |

The biggest win would be keeping vector operations in vector form through the SSA lowering rather than scalarizing them — this affects every shader and accounts for the majority of instruction bloat.
