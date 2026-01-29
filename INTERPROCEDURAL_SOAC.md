# Interprocedural SOAC Analysis for Compute Shader Parallelization

## Goal

Extend the MIR SOAC analysis to find maps nested inside user-defined functions, not just maps directly in the entry point body. This enables parallelization of compute shaders that call helper functions containing the actual map operation.

## Problem Statement

Given:
```wyn
def double(x: f32) f32 = x * 2.0
def apply_double(arr: []f32) []f32 = map(double, arr)

#[compute]
entry main(data: []f32) []f32 = apply_double(data)
```

The parallelization pass needs to:
1. Find the map inside `apply_double`
2. Create a chunk View of the entry input `data`
3. Pass the chunk to `apply_double` instead of the raw storage array

## What's Been Implemented

### 1. Provenance-Tracking SOAC Analysis (`soac_analysis.rs`)

Rewrote the analysis to track array provenance through the call graph:

```rust
pub enum ArrayProvenance {
    /// Entry input storage buffer
    EntryStorage { name: String, local: LocalId },

    /// Range/iota expression (for future bitcoin miner case)
    Range { start: ExprId, end: ExprId, step: Option<ExprId>, kind: RangeKind, inline_path: Vec<String> },

    /// Unknown - cannot parallelize
    Unknown,
}

pub struct ParallelizableMap {
    pub source: ArrayProvenance,        // Where the mapped array comes from
    pub closure_name: String,           // Function being mapped
    pub entry_call: Option<CallInfo>,   // If interprocedural, the call to transform
    pub is_inplace: bool,
}
```

The analyzer:
- Walks the entry point body tracking `LocalId -> ArrayProvenance`
- When entering a call, propagates argument provenance to callee parameters
- When finding a map, returns the provenance traced to its origin

### 2. Updated Parallelization Pass (`soac_parallelize.rs`)

Now handles both direct and interprocedural maps:

**Direct map**: Creates `map_into(closure, chunk_view, output, offset)`

**Interprocedural**: Calls `create_call_with_chunk()` which copies the original call but replaces the array argument with the chunk View

### 3. Monomorphization Fix (`monomorphization.rs`)

Added missing cases for address space type parsing:
```rust
"storage" => TypeName::AddressStorage,
"function" => TypeName::AddressFunction,
```

Without this, `storage` fell through to `TypeName::Named("storage")` causing lowering failures.

## Current Blocking Issue

The MIR output shows the transformation is working:
```
entry main:
  e25: @view(base=e6, offset=e24, len=e23)      -- chunk View created
  e26: apply_double$AddressStorage_v225(e25)   -- passed to helper

def apply_double$AddressStorage_v225:
  e2: @_w_intrinsic_map(e0, e1)                -- e1 = local_0 (arr parameter)
```

But lowering fails because:
- The helper's `arr` parameter is just `Expr::Local(0)`
- The type says `Array[f32, AddressStorage, ?225]` (storage array)
- But there's no `ArrayBacking` information for how to access it
- The lowering expects either a `View` expression or a direct `Storage` backing

## The Representation Question

### How array types flow through the system:

1. **Type checking**: `Array[elem, ?addrspace, ?size]` with type variables
2. **After resolution**: `Array[f32, AddressStorage, ?n]` for storage arrays
3. **MIR generation**: `ArrayBacking` captures representation:
   - `Storage { name, offset }` for entry point parameters
   - `View { base, offset }` for slices
   - etc.
4. **Monomorphization**: Creates `apply_double$AddressStorage_v225`

### The gap:

- Entry point parameters → associated with storage buffers via `EntryInput`
- Function parameters → just Locals with a type, **no backing info**

When `apply_double` receives a View, the parameter is typed as a storage array, but the MIR body just sees `Local(0)`. The fact that it's a View (fat pointer `{ptr, len}`) is lost.

## Options for Fixing

### Option 1: Normalize all dynamic storage access through Views

Add a pass that rewrites:
```
map(f, local_0)  -- where local_0 is storage array parameter
```
To:
```
map(f, View { base: local_0, offset: 0, size: length(local_0) })
```

Pro: Lowering only sees Views for dynamic storage
Con: Feels like a hack, "wrapping" things

### Option 2: Add backing info to LocalDecl for parameters

Extend `LocalDecl` to optionally include `ArrayBacking` for parameters that are arrays.

Pro: Explicit representation
Con: Duplicates type information, parameters don't really have "backings"

### Option 3: Convention-based lowering

All storage array parameters in non-entry functions are implicitly fat pointers. The lowering handles `Local` expressions with storage array type specially.

Pro: No MIR changes needed
Con: Implicit convention, lowering has to "figure out" what it's dealing with

### Option 4: Distinguish parameter arrays in the type system

Add a marker to distinguish "this is a reference I received" vs "this is a buffer I own".

Pro: Explicit in the type
Con: More type complexity

## Files Changed

- `wyn-core/src/mir/soac_analysis.rs` - Provenance-tracking analysis
- `wyn-core/src/soac_parallelize.rs` - Updated to use provenance
- `wyn-core/src/monomorphization.rs` - Address space parsing fix

## Test File

`testfiles/interprocedural_soac.wyn` - Simple test case for the feature

## Next Steps

Decide on the representation approach (Options 1-4 above) and implement it.
