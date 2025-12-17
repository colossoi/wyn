# MIR 2.0: Arena-Based Refactor

## Summary

Refactoring the MIR (Mid-level Intermediate Representation) from a recursive `Box<Expr>` structure to a flat arena-based representation with explicit `ExprId` and `LocalId` indices.

## Completed Work

### Phase 1: Replace MIR Types (Complete)

**Files modified:**
- `wyn-core/src/mir/mod.rs` - Complete rewrite

**Key changes:**
- Added `ExprId(u32)` and `LocalId(u32)` newtype indices
- Added `LocalKind` enum (Param, Let, LoopVar)
- Added `LocalDecl` struct with name, span, ty, kind
- Replaced recursive `Expr` with flat enum using `ExprId` references:
  - `Expr::Local(LocalId)` - local variable reference
  - `Expr::Global(String)` - global definition reference
  - `Expr::Int(String)`, `Expr::Float(String)`, `Expr::Bool(bool)`, `Expr::Unit`, `Expr::String(String)` - literals
  - `Expr::Tuple(Vec<ExprId>)`, `Expr::Array(Vec<ExprId>)`, `Expr::Vector(Vec<ExprId>)`, `Expr::Matrix(Vec<Vec<ExprId>>)` - aggregates
  - `Expr::BinOp { op, lhs: ExprId, rhs: ExprId }`, `Expr::UnaryOp { op, operand: ExprId }` - operations
  - `Expr::Let { local: LocalId, rhs: ExprId, body: ExprId }` - let bindings
  - `Expr::If { cond, then_, else_: ExprId }` - conditionals
  - `Expr::Loop { loop_var: LocalId, init: ExprId, init_bindings: Vec<(LocalId, ExprId)>, kind: LoopKind, body: ExprId }` - loops
  - `Expr::Call { func, args: Vec<ExprId> }`, `Expr::Intrinsic { name, args: Vec<ExprId> }` - calls
  - `Expr::Closure { lambda_name, captures: Vec<ExprId> }` - closures
  - `Expr::Range { start, step, end: ExprId, kind }` - ranges
  - `Expr::Materialize(ExprId)`, `Expr::Attributed { attributes, expr: ExprId }` - special
- Added `Body` struct with:
  - `locals: Vec<LocalDecl>` - local variable declarations
  - `exprs: Vec<Expr>` - expression arena
  - `types: Vec<Type<TypeName>>` - type per expression (parallel to exprs)
  - `spans: Vec<Span>` - span per expression (parallel to exprs)
  - `node_ids: Vec<NodeId>` - NodeId per expression (parallel to exprs)
  - `root: ExprId` - root expression
  - Builder methods: `alloc_local()`, `alloc_expr()`, `get_expr()`, `get_type()`, etc.
- Updated `Def` variants:
  - `Def::Function` now has `params: Vec<LocalId>` and `body: Body`
  - `Def::Constant` now has `body: Body`
  - `Def::EntryPoint` now has `body: Body`
- Updated `LoopKind` to use `LocalId` and `ExprId`:
  - `LoopKind::For { var: LocalId, iter: ExprId }`
  - `LoopKind::ForRange { var: LocalId, bound: ExprId }`
  - `LoopKind::While { cond: ExprId }`

### Phase 2: Rewrite Flattening (Complete)

**Files modified:**
- `wyn-core/src/flattening.rs` - Complete rewrite

**Key changes:**
- `Flattener` now builds `Body` instead of `Box<Expr>`
- `flatten_expr()` returns `ExprId` (allocates into body.exprs)
- Local variables tracked via `LocalId`:
  - `alloc_local()` creates `LocalDecl` and returns `LocalId`
  - Environment maps variable names to `LocalId`
- Removed `binding_id: u64` tracking (LocalId provides unique identification)
- Changed `needs_backing_store: HashSet<LocalId>` (was `HashSet<u64>`)
- Updated `backing_store_name()` to take `LocalId`

### Phase 3: SPIR-V Lowering (Complete)

**Files modified:**
- `wyn-core/src/spirv/lowering.rs` - Complete rewrite of expression lowering
- `wyn-core/src/lib.rs` - Uncommented `pub mod spirv;`

**Key changes:**
- Updated function signatures:
  - `lower_regular_function(constructor, name, params: &[LocalId], ret_type, body: &Body)`
  - `lower_entry_point_from_def(constructor, name, inputs, outputs, body: &Body)`
  - `lower_expr(constructor, body: &Body, expr_id: ExprId) -> Result<spirv::Word>`
  - `lower_const_expr(constructor, body: &Body, expr_id: ExprId) -> Result<spirv::Word>`
  - `try_extract_const_int(body: &Body, expr_id: ExprId) -> Option<i32>`
- `ensure_deps_lowered()` now iterates over `body.iter_exprs()`
- Pattern matching changed from `ExprKind::*` to `Expr::*`
- Type access changed from `expr.ty` to `body.get_type(expr_id)`
- Recursive calls changed from `lower_expr(constructor, &expr)` to `lower_expr(constructor, body, expr_id)`
- Variable lookup uses `body.get_local(local_id).name`
- Removed old `lower_literal()` and `lower_const_literal()` functions (literals handled directly)
- Added handling for `Expr::Tuple`, `Expr::Array`, `Expr::Vector`, `Expr::Matrix` in `lower_expr`

**Build status:** Compiles successfully with only warnings

**Test status:** 398 tests passing, 19 failing (alias_checker and desugar tests)

## Remaining Work

### Phase 4: Add Back MIR Passes

The following passes are currently commented out and need to be rewritten:

#### 4a. Reachability (`reachability.rs`)
- **Complexity:** Low (read-only traversal)
- **Changes needed:**
  - Traverse `Body` to find callees
  - Use `body.iter_exprs()` to walk expressions
  - Match on `Expr::Global`, `Expr::Call`, `Expr::Closure` to find dependencies
  - No transformation, just filtering

#### 4b. Constant Folding (`constant_folding.rs`)
- **Complexity:** Medium (stateless transformation)
- **Changes needed:**
  - Create new `Body` with folded expressions
  - Use `body.get_expr(id)` and `body.get_type(id)` for analysis
  - Allocate new expressions via `body.alloc_expr()`
  - Handle `Expr::Int`, `Expr::Float`, `Expr::Bool` for constant evaluation

#### 4c. Normalize (`normalize.rs`)
- **Complexity:** Medium-High (ANF transformation)
- **Changes needed:**
  - Needs fresh `LocalId`s for temporaries
  - Build new `Body` with normalized expressions
  - Use `body.alloc_local()` for new bindings
  - Transform complex expressions into let-bound temporaries

#### 4d. Materialize Hoisting (`materialize_hoisting.rs`)
- **Complexity:** Medium-High
- **Changes needed:**
  - Structural equality via arena traversal
  - Extract common materializations to Let bindings
  - Use `ExprId` for expression identity

#### 4e. Monomorphization (`monomorphization.rs`)
- **Complexity:** High
- **Changes needed:**
  - Type substitution on `Body`
  - Create specialized copies of functions
  - Update all type references in `body.types`

#### 4f. Binding Lifter (`binding_lifter.rs`)
- **Complexity:** High (loop-invariant code motion)
- **Changes needed:**
  - Track free variables via `LocalId`
  - Analyze loop bodies for invariant expressions
  - Move Let bindings outside loops

### Phase 5: GLSL Lowering

**File:** `wyn-core/src/glsl/lowering.rs`

Similar changes to SPIR-V lowering:
- Update function signatures to take `&Body, ExprId`
- Change pattern matching from `ExprKind::*` to `Expr::*`
- Use `body.get_type()`, `body.get_expr()`, `body.get_local()`

### Phase 6: Fix Failing Tests and Cleanup

#### Alias Checker Tests (6 failing)
- `test_inplace_map_simple_dead_after`
- `test_inplace_map_in_let`
- `test_inplace_map_nested`
- `test_inplace_with_simple_dead_after`
- `test_inplace_with_discarded`
- `test_inplace_with_chained`

**Issue:** `alias_checker.rs` needs to be updated for new MIR types.

#### Desugar Tests (13 failing)
- Various range and slice tests

**Issue:** Likely type mismatches or missing handling in flattening.

#### Cleanup Tasks
- Remove dead code warnings
- Re-enable MIR folder (`mir/folder.rs`) or remove if no longer needed
- Update test files for MIR passes
- Run full validation suite: `./scripts/validate_testfiles.sh`

## Architecture Notes

### Benefits of Arena-Based MIR
1. **Memory efficiency:** Expressions stored contiguously, no Box overhead
2. **Cache-friendly:** Linear traversal of expression array
3. **Easy copying:** Clone by value (just copy indices)
4. **Parallel access:** Multiple readers via indices, no reference lifetimes
5. **Explicit locals:** Clear tracking of all variables in function

### Key Patterns for Pass Authors
```rust
// Walking expressions
for expr in body.iter_exprs() {
    match expr {
        Expr::Call { func, args } => { /* ... */ }
        _ => {}
    }
}

// Getting type of expression
let ty = body.get_type(expr_id);

// Creating new expression
let new_id = body.alloc_expr(expr, ty.clone(), span, node_id);

// Creating new local
let local_id = body.alloc_local(LocalDecl {
    name: "temp".to_string(),
    span,
    ty: some_type,
    kind: LocalKind::Let,
});

// Looking up local info
let local = body.get_local(local_id);
let name = &local.name;
let ty = &local.ty;
```

## Files Modified Summary

| File | Status | Notes |
|------|--------|-------|
| `mir/mod.rs` | Rewritten | New arena-based types |
| `flattening.rs` | Rewritten | Produces new MIR |
| `spirv/lowering.rs` | Rewritten | Consumes new MIR |
| `lib.rs` | Updated | Pipeline changes |
| `alias_checker.rs` | Needs update | Uses old patterns |
| `materialize_hoisting.rs` | Commented out | Phase 4 |
| `normalize.rs` | Commented out | Phase 4 |
| `monomorphization.rs` | Commented out | Phase 4 |
| `reachability.rs` | Commented out | Phase 4 |
| `constant_folding.rs` | Commented out | Phase 4 |
| `binding_lifter.rs` | Commented out | Phase 4 |
| `mir/folder.rs` | Commented out | May deprecate |
| `glsl/lowering.rs` | Commented out | Phase 5 |
