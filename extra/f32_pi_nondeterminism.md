# f32.pi nondeterminism investigation

## Symptom

`testfiles/raytrace.wyn` fails ~50% of the time with:

```
Error: in function 'materials.brdf': SPIR-V generation error:
  Block(BlockId(1v1)): Unknown global: f32.pi
```

Confirmed by running release build 1000 times: 515 pass, 485 fail.

## Root cause

Rust's `HashMap` uses `RandomState` (SipHash with per-process random seed) for
DoS resistance. Any code path that iterates a HashMap and feeds results into an
ordered structure (like `Vec<Def>`) produces nondeterministic ordering.

## Evidence

Dumping TLC output (`--output-tlc`) across multiple runs shows:
- Different file sizes (16244 vs 16272 bytes)
- Different SymbolId assignments (`sym285 = 3.1415927` vs `sym152 = 3.1415927`)
- Definition ordering shuffled between runs

The value `f32.pi = 3.14159265358979323846f32` is defined in `prelude/math.wyn`
as a module constant. After module elaboration it becomes `f32.pi`. The value is
always present in TLC, but gets different SymbolIds each run.

## Likely source

The nondeterminism originates somewhere in the TLC pipeline where a HashMap is
iterated to produce def ordering. Candidates identified so far:

- `tlc/partial_eval.rs:90` — `defs: HashMap<SymbolId, Def>` (used for lookup only, iteration is over input Vec — probably safe)
- `tlc/buffer_specialize.rs:59` — `def_map: HashMap<SymbolId, Def>` (used for lookup only — probably safe)

The source has NOT been conclusively identified yet. It could be in:
- Module elaboration (`elaborate_modules`)
- Name resolution / symbol table construction
- Monomorphization's `SpecKey` HashMap
- Defunctionalization's capture ordering

## Why it causes the error

When def ordering changes, `to_ssa`'s Phase 1 constant detection (lines 124-163)
processes defs in a different order. It builds `pure_constant_names: HashSet<String>`
incrementally — each constant sees only previously-detected constants. If `f32.pi`
is processed in an order where it's not recognized as purely constant (unlikely
for a literal, but possible if its SymbolId lookup fails due to the name being
under a different SymbolId), it falls through to emit `Global("f32.pi")` which
SPIR-V can't resolve.

More likely: the TLC constant inliner (`inline_small`) uses `find_all_constants`
which builds `HashMap<SymbolId, Term>`. If `f32.pi` has a SymbolId that differs
from what the referencing function uses (post-monomorphization divergence), and
the name-based fallback in `find_all_constants` doesn't catch it due to
`def_syms: HashMap<String, SymbolId>` iteration order, the constant doesn't get
inlined into `materials.brdf`.

## Fix approach

1. Find the exact HashMap iteration that produces nondeterministic def ordering
2. Either:
   a. Replace `HashMap` with `IndexMap` (preserves insertion order) at that site
   b. Sort defs by name or SymbolId after the HashMap-based pass
   c. Use `BTreeMap` instead of `HashMap` where iteration order matters
