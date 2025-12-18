# MIR 2.0: Arena-Based Refactor

## Summary

Refactoring the MIR (Mid-level Intermediate Representation) from a recursive `Box<Expr>` structure to a flat arena-based representation with explicit `ExprId` and `LocalId` indices.

## Current Status

- **486 tests passing**
- **7/8 testfiles validating** (seascape.wyn fails - constant folding issue)
- All Phase 4 MIR passes ported
- GLSL lowering ported

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
  - `Expr::Closure { lambda_name, captures: ExprId }` - closures (captures is single ExprId pointing to tuple/unit)
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
- **Fixed:** `flatten_application` no longer allocates dead `Expr::Global` for direct function calls

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
- Added handling for `Expr::Tuple`, `Expr::Array`, `Expr::Vector`, `Expr::Matrix` in `lower_expr`

### Phase 4: MIR Passes (Complete)

All MIR passes have been ported to the arena-based structure:

#### 4a. Reachability (`reachability.rs`) - Complete
- Traverses `Body` to find callees
- Uses arena iteration to walk expressions

#### 4b. Constant Folding (`constant_folding.rs`) - Complete
- Creates new `Body` with folded expressions
- Handles literal evaluation

#### 4c. Normalize (`normalize.rs`) - Complete
- ANF transformation with fresh `LocalId`s for temporaries
- Builds new `Body` with normalized expressions
- Fixed loop kind bindings and let-tuple variable scoping

#### 4d. Materialize Hoisting (`materialize_hoisting.rs`) - Complete
- Structural equality via arena traversal
- Extracts common materializations to Let bindings

#### 4e. Monomorphization (`monomorphization.rs`) - Complete
- Type substitution on `Body`
- Creates specialized copies of polymorphic functions
- Fixed: only specializes polymorphic functions when called, not when referenced

#### 4f. Binding Lifter (`binding_lifter.rs`) - Complete
- Tracks free variables via `LocalId`
- Analyzes loop bodies for invariant expressions

### Phase 5: GLSL Lowering (Complete)

**File:** `wyn-core/src/glsl/lowering.rs`

- Updated function signatures to take `&Body, ExprId`
- Changed pattern matching from `ExprKind::*` to `Expr::*`
- Uses `body.get_type()`, `body.get_expr()`, `body.get_local()`

## Remaining Work

### seascape.wyn constant folding issue

Error: `Global constants must be literals or compile-time foldable expressions`

The SPIR-V lowering's `lower_const_expr` doesn't handle all expression types that could appear in global constants. Needs investigation into which expression type is failing.

### Cleanup Tasks
- Remove dead code warnings
- Remove or deprecate `mir/folder.rs` (no longer used)
- Full validation suite

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
| `mir/mod.rs` | Complete | New arena-based types |
| `flattening.rs` | Complete | Produces new MIR |
| `spirv/lowering.rs` | Complete | Consumes new MIR |
| `glsl/lowering.rs` | Complete | Consumes new MIR |
| `lib.rs` | Complete | Pipeline changes |
| `alias_checker.rs` | Complete | Updated for new MIR |
| `materialize_hoisting.rs` | Complete | Ported to arena |
| `normalize.rs` | Complete | Ported to arena |
| `monomorphization.rs` | Complete | Ported to arena |
| `reachability.rs` | Complete | Ported to arena |
| `constant_folding.rs` | Complete | Ported to arena |
| `binding_lifter.rs` | Complete | Ported to arena |
| `mir/folder.rs` | Deprecated | No longer used |

## Testfile Validation

| File | Status |
|------|--------|
| `da_rasterizer.wyn` | PASS |
| `entrylevel.wyn` | PASS |
| `holodice.wyn` | PASS |
| `lava.wyn` | PASS |
| `primitives.wyn` | PASS |
| `red_triangle.wyn` | PASS |
| `red_triangle_curried.wyn` | PASS |
| `seascape.wyn` | FAIL (constant folding) |
