# WIP: Prelude Functions and Monomorphization

## The Challenge

Validation of testfiles fails with "Unknown function: map" errors during SPIR-V lowering.

### Root Cause Analysis

1. **Prelude functions are thin wrappers around intrinsics**:
   ```wyn
   def map<[n], A, B>(f: A -> B, xs: [n]A) [n]B =
     _w_intrinsic_map(f, xs)
   ```

2. **SPIR-V lowering only knows about intrinsics** (`_w_intrinsic_map`), not the user-facing wrappers (`map`).

3. **Prelude functions are intentionally excluded from TLC transformation** (see comment in lib.rs:586-588). They never become MIR definitions, so the lowerer can't find them.

### Attempted Fix: Include Prelude in TLC

Modified `tlc::transform` to accept prelude declarations and transform them alongside user code. This way prelude functions go through all optimization passes (partial eval, lambda lifting, etc.) together with user code.

**Result**: Monomorphization fails with "Unresolved type variable Variable(425)".

### Why Monomorphization Fails

Prelude functions like `map` have polymorphic types:
- `map<[n], A, B>(f: A -> B, xs: [n]A) [n]B`
- Type variables: `n`, `A`, `B`

When `map` is called from user code with concrete types, the monomorphizer should:
1. Find the call site with concrete types (e.g., `map(f, arr)` where `f: f32 -> f32`, `arr: [10]f32`)
2. Specialize `map` for those concrete types
3. Replace the polymorphic definition with concrete instances

**The problem**: Even with pre-monomorphization DCE filtering out unused prelude functions, the *reachable* polymorphic functions (like `map`, which IS called) still have unresolved type variables in their definitions.

The monomorphizer encounters the `map` definition with types like `A`, `B`, `n` and fails because it expects all types to be concrete before specialization.

## Options

### Option 1: Inline Prelude Wrappers During TLC→MIR

Recognize prelude wrapper patterns during TLC→MIR transformation and inline them to their underlying intrinsics.

**Approach**: When transforming a call like `map(f, xs)`, detect that `map` is a prelude wrapper and emit `_w_intrinsic_map(f, xs)` instead.

**Pros**:
- Simple, targeted fix
- No changes to monomorphization needed
- Prelude functions become zero-cost abstractions

**Cons**:
- Requires maintaining a list of prelude wrappers
- Won't work for user-defined polymorphic functions
- Loses the ability to optimize prelude code

### Option 2: Fix Monomorphization Type Propagation

The monomorphizer should only process polymorphic function *definitions* when specializing from call sites, using the concrete types from the call.

**Approach**:
1. Don't process polymorphic definitions directly
2. When encountering a call to a polymorphic function, instantiate a specialized version using the call's concrete types
3. Track instantiation requests and process them lazily

**Pros**:
- Proper solution that handles all polymorphic functions
- Would enable user-defined polymorphic library functions

**Cons**:
- More complex change to monomorphization
- May require restructuring how types flow through the pipeline

### Option 3: On-Demand Prelude Inclusion

Only include prelude functions that are actually called, and transform them with concrete type information from their call sites.

**Approach**:
1. Analyze user code to find prelude function calls
2. For each call, instantiate the prelude function with the call's concrete types
3. Transform only those concrete instances to TLC/MIR

**Pros**:
- Avoids polymorphism entirely
- Minimal code generation

**Cons**:
- Requires type information propagation during TLC transformation
- Essentially duplicates monomorphization logic

### Option 4: Treat Prelude Names as Intrinsic Aliases in Lowering

Handle `map`, `reduce`, `zip`, etc. in SPIR-V lowering as aliases for their intrinsic counterparts.

**Approach**: In the lowerer, when encountering a call to `map`, treat it the same as `_w_intrinsic_map`.

**Pros**:
- Minimal change
- No pipeline modifications needed

**Cons**:
- Hardcoded knowledge of prelude in lowerer
- Prelude functions can't be optimized or modified
- Doesn't scale to user-defined library functions

## Recommendation

**Option 2 (Fix Monomorphization)** is the most principled solution but requires understanding why type information isn't flowing correctly from call sites to polymorphic definitions.

**Option 1 (Inline Wrappers)** is a pragmatic short-term fix that would unblock progress while the monomorphization issue is investigated.

## Investigation Notes

- DCE before monomorphization works: 78 defs → 29 reachable
- The issue is that reachable polymorphic functions still have unresolved type variables
- Monomorphization entry point is `normalized.monomorphize()` in main.rs
- Type variables like `Variable(425)` suggest the type table has unresolved constraints
