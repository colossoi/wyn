# Defunctionalization in Wyn: Research Notes

## Background

This document captures research on how Futhark handles higher-order functions (HOFs) through defunctionalization, and how this applies to Wyn's TLC transformation.

## The Problem

When a lambda captures variables from its environment, those captures need to be passed somehow when the lambda is called. For example:

```
def sum_offset(arr, y) =
  map(|x| x + y, arr)   // y is captured
```

After lambda lifting, the lambda becomes a top-level function with `y` as an extra parameter:

```
_lambda_0(x, y) = x + y
```

But now `map` receives `_lambda_0` which expects 2 arguments, while `map` only knows to pass 1 (the array element). How do we get `y` to `_lambda_0`?

## Two Approaches

### 1. Closure Structs (Reynolds-style)

- Represent closures as `{ tag: function_id, captures: [values...] }`
- SOACs/HOFs receive the closure struct
- At call time, extract tag and captures, dispatch to the right function

**Problem**: Requires runtime dispatch or special handling in each HOF.

### 2. Flattened Arguments + Specialization (Futhark-style)

- Captures become trailing parameters on the lifted lambda
- HOFs are **specialized at each call site** based on which function is passed
- Captures flow through as extra parameters

**Key insight**: No runtime dispatch needed because every call site is specialized.

## Futhark's Type Restrictions

Futhark ensures every applied function is **statically known** through type restrictions:

1. Functions may NOT be stored inside arrays
2. Functions may NOT be returned from conditional branches
3. Functions may NOT be loop parameters

These restrictions guarantee the compiler always knows which specific function is being called at every application site.

## How Specialization Works

Given:
```
def apply(f, x) = f(x)

// Call site:
apply(|a| a + y, 5)   // y is captured
```

After lambda lifting:
```
_lambda_0(a, y) = a + y   // arity 2, capture y at end
```

Specialized `apply` for this call site:
```
apply_specialized(x, y) = _lambda_0(x, y)
```

The captures (`y`) become parameters of the specialized function. No closure struct needed.

## Implications for Wyn

### Current State (broken)

The current TLC defunctionalization:
1. Lifts lambdas with captures as extra parameters (now at end)
2. Has special-case handling for SOACs to flatten captures
3. Non-SOAC HOFs don't work

### Proposed Fix

Treat ALL HOF call sites uniformly:
1. Lift lambdas with captures as trailing parameters
2. Track which function is passed at each HOF call site (via StaticVal)
3. **Specialize the HOF** for that specific function, threading captures through
4. No special SOAC handling needed - SOACs are just HOFs

This is essentially **monomorphization for HOFs** - similar to how we specialize polymorphic functions for different types, we specialize HOFs for different function arguments.

### Example Transformation

Input:
```
def my_map(f, arr) =
  // body that calls f on each element

def main(arr, y) =
  my_map(|x| x + y, arr)
```

After transformation:
```
_lambda_0(x, y) = x + y

// Specialized for _lambda_0 with 1 capture
def my_map_lambda_0(arr, y) =
  // body that calls _lambda_0(elem, y) on each element

def main(arr, y) =
  my_map_lambda_0(arr, y)
```

## References

- [The Futhark Language](https://futhark-book.readthedocs.io/en/latest/language.html)
- [High-Performance Defunctionalisation in Futhark (Paper)](https://www.researchgate.net/publication/332598125_High-Performance_Defunctionalisation_in_Futhark)
- [High-Performance Defunctionalisation in Futhark (Slides)](https://www.cse.chalmers.se/~myreen/tfp2018/slides/Anders_Kiel_Hovgaard.pdf)
- [Defunctionalization - Wikipedia](https://en.wikipedia.org/wiki/Defunctionalization)
- `extra/defunc.txt` - Original Futhark defunctionalization paper notes

## Open Questions

1. How to handle the specialization pass? New TLC pass or extend existing defunctionalization?
2. Do we need to enforce Futhark's type restrictions, or can we rely on existing limitations?
3. How does this interact with monomorphization of polymorphic functions?
