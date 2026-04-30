use crate::Compiler;
use crate::alias_checker::{AliasCheckResult, AliasChecker};

fn check_alias(source: &str) -> AliasCheckResult {
    let mut frontend = crate::cached_frontend();
    let parsed = Compiler::parse(source, &mut frontend.node_counter).expect("parse failed");
    let desugared = parsed.desugar(&mut frontend.node_counter).expect("desugar failed");
    let resolved = desugared.resolve(&mut frontend.module_manager).expect("resolve failed");
    let folded = resolved.fold_ast_constants();
    let type_checked =
        folded.type_check(&mut frontend.module_manager, &mut frontend.schemes).expect("type_check failed");

    let checker = AliasChecker::new(&type_checked.type_table, &type_checked.span_table);
    checker.check_program(&type_checked.ast).expect("alias check failed")
}

/// Helper to run through the pipeline up to alias checking (returns the AliasChecked stage)
fn alias_check_pipeline(source: &str) -> crate::AliasChecked {
    let mut frontend = crate::cached_frontend();
    let parsed = Compiler::parse(source, &mut frontend.node_counter).expect("parse failed");
    parsed
        .desugar(&mut frontend.node_counter)
        .expect("desugar failed")
        .resolve(&mut frontend.module_manager)
        .expect("resolve failed")
        .fold_ast_constants()
        .type_check(&mut frontend.module_manager, &mut frontend.schemes)
        .expect("type_check failed")
        .alias_check()
        .expect("alias_check failed")
}

#[test]
fn test_no_error_simple() {
    let result = check_alias(r#"def main(x: i32) i32 = x + 1"#);
    assert!(!result.has_errors());
}

#[test]
fn test_use_after_move() {
    let source = r#"
def consume(arr: *[4]i32) i32 = arr[0]

def main(arr: [4]i32) i32 =
    let _ = consume(arr) in
    arr[0]
"#;
    let result = check_alias(source);
    assert!(result.has_errors(), "Expected use-after-move error");
}

#[test]
fn test_alias_through_let() {
    let source = r#"
def consume(arr: *[4]i32) i32 = arr[0]

def main(arr: [4]i32) i32 =
    let alias = arr in
    let _ = consume(arr) in
    alias[0]
"#;
    let result = check_alias(source);
    assert!(result.has_errors(), "Expected use-after-move error for alias");
}

#[test]
fn test_copy_type_no_tracking() {
    // i32 is copy, should work fine
    let source = r#"
def main(x: i32) i32 =
    let y = x in
    let z = x in
    y + z
"#;
    let result = check_alias(source);
    assert!(!result.has_errors());
}

// === Tricky edge cases ===

#[test]
fn test_transitive_aliasing() {
    // a -> b -> c, consume c, use a should error
    let source = r#"
def consume(arr: *[4]i32) i32 = arr[0]

def main(arr: [4]i32) i32 =
    let a = arr in
    let b = a in
    let c = b in
    let _ = consume(c) in
    a[0]
"#;
    let result = check_alias(source);
    assert!(
        result.has_errors(),
        "Expected error: a transitively aliases c which was consumed"
    );
}

#[test]
fn test_consume_alias_use_original() {
    // Consume the alias, then try to use the original - should error
    let source = r#"
def consume(arr: *[4]i32) i32 = arr[0]

def main(arr: [4]i32) i32 =
    let alias = arr in
    let _ = consume(alias) in
    arr[0]
"#;
    let result = check_alias(source);
    assert!(
        result.has_errors(),
        "Expected error: arr's backing store was consumed via alias"
    );
}

#[test]
fn test_shadowing_does_not_affect_outer() {
    // Inner variable shadows outer with same name, consume inner, use outer
    // This should be OK because they're different backing stores
    let source = r#"
def consume(arr: *[4]i32) i32 = arr[0]

def main(arr: [4]i32) i32 =
    let x = arr in
    let result =
        let x = [1, 2, 3, 4] in
        consume(x)
    in
    x[0]
"#;
    let result = check_alias(source);
    assert!(
        !result.has_errors(),
        "Should be OK: inner x is different from outer x"
    );
}

#[test]
fn test_if_branches_independent() {
    // Consume in one branch shouldn't affect use in other branch
    // (they're mutually exclusive execution paths)
    let source = r#"
def consume(arr: *[4]i32) i32 = arr[0]

def main(t: (bool, [4]i32)) i32 =
    let (cond, arr) = t in
    if cond then consume(arr) else arr[0]
"#;
    let result = check_alias(source);
    // This SHOULD be OK since the branches are mutually exclusive
    // But a conservative implementation might reject it
    assert!(
        !result.has_errors(),
        "Should be OK: branches are mutually exclusive"
    );
}

#[test]
fn test_use_after_if_that_consumes() {
    // Use after an if expression where one branch consumed - should error
    let source = r#"
def consume(arr: *[4]i32) i32 = arr[0]

def main(t: (bool, [4]i32)) i32 =
    let (cond, arr) = t in
    let _ = if cond then consume(arr) else 0 in
    arr[0]
"#;
    let result = check_alias(source);
    assert!(
        result.has_errors(),
        "Expected error: arr might have been consumed in if branch"
    );
}

#[test]
fn test_return_aliases_non_consumed_params() {
    // id3 takes: a (consumed via *), b (not consumed), c (not consumed)
    // Returns non-unique [4]i32, which should alias b and c
    // After the call: x is consumed, y and z are aliased by result
    // Use separate array literals so each has its own backing store

    let base = r#"
def id3(a: *[4]i32, b: [4]i32, c: [4]i32) [4]i32 = c
def consume(arr: *[4]i32) i32 = arr[0]
def main: i32 =
    let x = [1, 2, 3, 4] in
    let y = [5, 6, 7, 8] in
    let z = [9, 10, 11, 12] in
    let result = id3(x, y, z) in
"#;

    // Try consuming x - should error (x was already consumed by id3)
    let source_x = format!("{}    consume(x)", base);
    let result_x = check_alias(&source_x);
    assert!(result_x.has_errors(), "Expected error: x was consumed by id3");

    // Consume result, then try to use y - should error (result aliases y)
    let source_result_then_y = format!("{}    let _ = consume(result) in y[0]", base);
    let result_y = check_alias(&source_result_then_y);
    assert!(
        result_y.has_errors(),
        "Expected error: result aliases y, consuming result invalidates y"
    );

    // Consume result, then try to use z - should error (result aliases z)
    let source_result_then_z = format!("{}    let _ = consume(result) in z[0]", base);
    let result_z = check_alias(&source_result_then_z);
    assert!(
        result_z.has_errors(),
        "Expected error: result aliases z, consuming result invalidates z"
    );
}

#[test]
fn test_return_aliases_shared_backing_store() {
    // Using tuple destructuring - all three variables share a backing store
    // from the tuple. x, y, z all reference the same underlying memory.
    // Error messages now include alias information to help users understand
    // why seemingly unrelated variables are affected.

    let base = r#"
def id3(a: *[4]i32, b: [4]i32, c: [4]i32) [4]i32 = c
def consume(arr: *[4]i32) i32 = arr[0]
def main(t: ([4]i32, [4]i32, [4]i32)) i32 =
    let (x, y, z) = t in
    let result = id3(x, y, z) in
"#;

    // All three should error because they share a backing store with x,
    // which was consumed by id3
    let source_x = format!("{}    consume(x)", base);
    let result_x = check_alias(&source_x);
    assert!(result_x.has_errors(), "Expected error: x was consumed by id3");

    let source_y = format!("{}    consume(y)", base);
    let result_y = check_alias(&source_y);
    assert!(
        result_y.has_errors(),
        "Expected error: y shares backing store with consumed x"
    );

    let source_z = format!("{}    consume(z)", base);
    let result_z = check_alias(&source_z);
    assert!(
        result_z.has_errors(),
        "Expected error: z shares backing store with consumed x"
    );
}

// =============================================================================
// Comprehensive uniqueness/aliasing coverage
// =============================================================================
//
// The 11 tests above cover the core "consumed-then-used" patterns. The
// following tests exercise:
//   - basic legitimate uses that should *not* error
//   - more elaborate consumption violations (lambdas, loops, slicing,
//     match arms, swizzle-with on unique vecs, multi-dim arrays)
//   - subtle aliasing introduced by tuple/record construction and
//     destructuring
//   - "devious" cases: self-aliasing in args, branching aliases, dead
//     bindings, polymorphism over uniqueness
//
// Tests marked `#[ignore]` document behaviors I expect from the design
// but that the implementation either doesn't support yet or handles in
// a way I'd want to confirm with the maintainer before treating as
// authoritative. Each ignored test has an inline note about which side
// it falls on.

// --- A. Basic legitimate uses that should not error -------------------------

#[test]
fn test_unique_pass_through() {
    // The simplest valid use: a function that just returns its unique input.
    let source = r#"
def pass(x: *[4]i32) *[4]i32 = x

def main(arr: *[4]i32) i32 = (pass(arr))[0]
"#;
    let result = check_alias(source);
    assert!(!result.has_errors(), "pass-through of unique arg should be legal");
}

#[test]
fn test_unique_modify_and_return() {
    // Standard "consume, modify, return" pattern.
    let source = r#"
def bump(x: *[4]i32) *[4]i32 = x with [0] = 99

def main(arr: *[4]i32) i32 = (bump(arr))[0]
"#;
    let result = check_alias(source);
    assert!(!result.has_errors(), "in-place modify-and-return should be legal");
}

#[test]
fn test_call_unique_with_fresh_literal() {
    // An array literal has no prior alias, so feeding it to a unique
    // parameter is the canonical safe case.
    let source = r#"
def consume(arr: *[4]i32) i32 = arr[0]

def main: i32 = consume([1, 2, 3, 4])
"#;
    let result = check_alias(source);
    assert!(
        !result.has_errors(),
        "passing a fresh array literal to a unique param should be legal"
    );
}

#[test]
#[ignore = "TODO: missing `*T → T` weakening in return position. \
            step2 returns `*[4]i32` but main's declared return is `[4]i32` — \
            this should typecheck (uniqueness can always be discarded), \
            but the type checker currently rejects it."]
fn test_unique_chain_through_calls() {
    // Receive unique, hand off to another unique consumer.
    // The interesting bit isn't the chain — it's that the final
    // result, a `*[4]i32`, should be usable where a `[4]i32` is
    // expected.
    let source = r#"
def step1(x: *[4]i32) *[4]i32 = x with [0] = 1
def step2(x: *[4]i32) *[4]i32 = x with [1] = 2

def main(arr: *[4]i32) [4]i32 = step2(step1(arr))
"#;
    let result = check_alias(source);
    assert!(!result.has_errors(), "chained unique consumers should be legal");
}

#[test]
fn test_two_distinct_unique_params() {
    // Two different unique sources to two different unique parameters.
    let source = r#"
def combine(a: *[4]i32, b: *[4]i32) i32 = a[0] + b[0]

def main: i32 = combine([1, 2, 3, 4], [5, 6, 7, 8])
"#;
    let result = check_alias(source);
    assert!(
        !result.has_errors(),
        "two distinct unique sources should be legal"
    );
}

#[test]
#[ignore = "TODO: missing `*T → T` weakening when constructing tuples. \
            arr is `*[4]i32` but main's tuple return type wants a `[4]i32` \
            element — uniqueness should be discardable here, but the type \
            checker rejects."]
fn test_unique_used_then_returned() {
    // Read from `arr` (allowed), then return it. No consumption
    // happens, so the read of `arr[0]` and the return of `arr` are
    // both legal. The point of the test is the *legitimate
    // weakening* of `*[4]i32` to `[4]i32` inside the tuple.
    let source = r#"
def main(arr: *[4]i32) ([4]i32, i32) =
    let x = arr[0] in
    (arr, x)
"#;
    let result = check_alias(source);
    assert!(
        !result.has_errors(),
        "indexing without consuming, then returning, should be legal"
    );
}

// --- B. Self-aliasing and double-consume violations -------------------------

#[test]
fn test_self_aliasing_in_unique_args() {
    // Passing the same array to two unique parameters should error —
    // it gets consumed twice with overlapping memory.
    let source = r#"
def both(a: *[4]i32, b: *[4]i32) i32 = a[0] + b[0]

def main(arr: [4]i32) i32 = both(arr, arr)
"#;
    let result = check_alias(source);
    assert!(
        result.has_errors(),
        "passing the same array as two unique args should be rejected"
    );
}

#[test]
fn test_double_consume_sequential() {
    // Consuming twice in sequence — second call must error.
    let source = r#"
def consume(arr: *[4]i32) i32 = arr[0]

def main(arr: [4]i32) i32 =
    let _ = consume(arr) in
    consume(arr)
"#;
    let result = check_alias(source);
    assert!(result.has_errors(), "second consume of arr should be rejected");
}

#[test]
fn test_consume_then_pass_to_non_unique() {
    // After consumption, the value can't be observed at all — even by a
    // non-unique-expecting function.
    let source = r#"
def consume(arr: *[4]i32) i32 = arr[0]
def peek(arr: [4]i32) i32 = arr[0]

def main(arr: [4]i32) i32 =
    let _ = consume(arr) in
    peek(arr)
"#;
    let result = check_alias(source);
    assert!(result.has_errors(), "post-consume read should be rejected");
}

// --- C. Slicing and multi-dim aliasing --------------------------------------

#[test]
fn test_consume_slice_invalidates_source() {
    // Slice aliases the source; consuming the slice should also kill
    // the source. (Marked ignore until I confirm whether slicing emits
    // an alias relationship in the alias checker today.)
    let source = r#"
def consume(arr: *[2]i32) i32 = arr[0]

def main(arr: [4]i32) i32 =
    let s = arr[0..2] in
    let _ = consume(s) in
    arr[0]
"#;
    let result = check_alias(source);
    assert!(
        result.has_errors(),
        "consuming a slice should invalidate the source array"
    );
}

#[test]
fn test_slice_two_then_consume_one() {
    // Two slices of the same array: consuming one slice should
    // invalidate the other (overlapping or not, conservative
    // implementations reject).
    let source = r#"
def consume(arr: *[2]i32) i32 = arr[0]

def main(arr: [4]i32) i32 =
    let s1 = arr[0..2] in
    let s2 = arr[2..4] in
    let _ = consume(s1) in
    s2[0]
"#;
    let result = check_alias(source);
    assert!(
        result.has_errors(),
        "consuming a slice should invalidate the source array's other slices"
    );
}

#[test]
#[ignore = "compiler gap: indexing a 2D array doesn't propagate the alias \
            relationship from the inner row back to the outer grid. After \
            `let row = grid[0]; consume(row)`, the alias checker thinks \
            `grid` is still live — it isn't, because grid[0]'s memory has \
            been consumed."]
fn test_consume_inner_row_invalidates_outer() {
    // Indexing a 2D unique array yields an inner array. Consuming it
    // should invalidate the outer.
    let source = r#"
def consume(row: *[4]i32) i32 = row[0]

def main(grid: [3][4]i32) i32 =
    let row = grid[0] in
    let _ = consume(row) in
    grid[1][0]
"#;
    let result = check_alias(source);
    assert!(
        result.has_errors(),
        "consuming an inner row should invalidate the outer 2D array"
    );
}

// --- D. Lambdas and higher-order functions ----------------------------------

#[test]
#[ignore = "compiler gap: a lambda body that consumes a captured unique \
            value should be rejected — the lambda could be invoked \
            zero or more times, which makes a single-consumption \
            guarantee impossible. Today the alias checker accepts the \
            program."]
fn test_lambda_body_consumes_capture() {
    // A lambda that consumes a captured variable. This is conceptually
    // tricky: the lambda might be called more than once, which would be
    // a double-consume. The conservative answer is to reject any
    // lambda body that consumes a captured unique value.
    let source = r#"
def consume(arr: *[4]i32) i32 = arr[0]

def main(arr: *[4]i32) i32 =
    let f = |dummy: i32| consume(arr) in
    f(0)
"#;
    let result = check_alias(source);
    assert!(
        result.has_errors(),
        "a lambda capturing and consuming a unique value should be rejected"
    );
}

#[test]
fn test_map_over_array_does_not_consume_array() {
    // `map(f, xs)` reads `xs`; it doesn't consume it. After the map,
    // `xs` should still be live.
    let source = r#"
def main(xs: [4]i32) ([4]i32, i32) =
    let ys = map(|x: i32| x + 1, xs) in
    (xs, ys[0])
"#;
    let result = check_alias(source);
    assert!(!result.has_errors(), "map should not consume the source array");
}

// --- E. Match and loop control flow ----------------------------------------

#[test]
fn test_use_after_match_with_consuming_arm() {
    // Same shape as `test_use_after_if_that_consumes` but with `match`.
    let source = r#"
def consume(arr: *[4]i32) i32 = arr[0]

def main(t: (#some(i32) | #none, [4]i32)) i32 =
    let (tag, arr) = t in
    let _ = match tag
            case #some(_) -> consume(arr)
            case #none -> 0 in
    arr[0]
"#;
    let result = check_alias(source);
    assert!(
        result.has_errors(),
        "post-match use of arr should be rejected when one arm consumes"
    );
}

#[test]
fn test_loop_carried_unique_consumed_once_per_iteration() {
    // `loop (acc) = arr while ...` carries `acc` across iterations.
    // The body produces a new `acc` and the old one is dead — should
    // be legal.
    let source = r#"
def main(arr: *[4]i32) i32 =
    let r = loop (acc) = arr while false do
              acc with [0] = 1
    in r[0]
"#;
    let result = check_alias(source);
    assert!(
        !result.has_errors(),
        "loop-carrying a unique value through `with` should be legal"
    );
}

#[test]
#[ignore = "compiler gap: a loop body that consumes a non-loop-carried \
            unique value should be rejected — the body runs every \
            iteration, so the second iteration would re-consume. The \
            alias checker should treat anything captured by a loop body \
            the same way it treats lambda captures (a closure invoked \
            from inside the loop)."]
fn test_loop_body_consumes_outer_unique() {
    // Body consumes `outside`, but the body runs N times — second
    // iteration would re-consume.
    let source = r#"
def consume(arr: *[4]i32) i32 = arr[0]

def main(outside: *[4]i32) i32 =
    let r = loop (i) = 0 while i < 10 do
              let _ = consume(outside) in i + 1
    in r
"#;
    let result = check_alias(source);
    assert!(
        result.has_errors(),
        "loop body consuming an outer unique should be rejected"
    );
}

// --- F. Tuples / records / vec containers ----------------------------------

#[test]
fn test_consume_one_element_of_tuple() {
    // A tuple `(unique_a, b)`: consuming `a` should leave `b` alone
    // since they're separate elements with separate stores. The
    // existing test_return_aliases_shared_backing_store covers a
    // tuple-destructuring case — this one builds a tuple from
    // separate sources first.
    let source = r#"
def consume(arr: *[4]i32) i32 = arr[0]

def main: i32 =
    let a = [1, 2, 3, 4] in
    let b = [5, 6, 7, 8] in
    let _ = consume(a) in
    b[0]
"#;
    let result = check_alias(source);
    assert!(
        !result.has_errors(),
        "consuming one element should not invalidate another from a different literal"
    );
}

#[test]
#[ignore = "compiler gap (type checker): VecWith doesn't strip the unique \
            wrapper from its target type. `*vec3f32` should be acceptable \
            anywhere `vec3f32` is, but the swizzle-with arm fails with \
            'requires a vector target, got *[Vec[f32, 3]]'. Once that's \
            fixed, the alias checker should also reject the use of `v` \
            after the with consumed it."]
fn test_swizzle_with_consumes_unique_target() {
    // The new `with .swizzle = e` form rebuilds the vec; if the
    // target is unique and dead after, the source should be
    // consumed.
    let source = r#"
def main(v: *vec3f32) f32 =
    let v2 = v with .yz = @[1.0f32, 2.0f32] in
    v[0]
"#;
    let result = check_alias(source);
    assert!(
        result.has_errors(),
        "VecWith should consume its unique source like ArrayWith"
    );
}

// --- G. Devious cases ------------------------------------------------------

#[test]
fn test_branching_alias_invalidates_both_sources() {
    // `y` is bound to either `a` or `b`. Conservatively, consuming
    // `y` invalidates both (the alias checker can't know at compile
    // time which branch ran).
    let source = r#"
def consume(arr: *[4]i32) i32 = arr[0]

def main(t: (bool, [4]i32, [4]i32)) i32 =
    let (c, a, b) = t in
    let y = if c then a else b in
    let _ = consume(y) in
    a[0]
"#;
    let result = check_alias(source);
    assert!(
        result.has_errors(),
        "consuming a value bound from an if-expression should invalidate both branches' sources"
    );
}

#[test]
fn test_wildcard_let_does_not_consume() {
    // Binding to `_` is not a consumption; it's effectively a
    // sequencing point but doesn't take ownership.
    let source = r#"
def main(arr: *[4]i32) i32 =
    let _ = arr in
    arr[0]
"#;
    let result = check_alias(source);
    assert!(!result.has_errors(), "let _ = arr should not consume arr");
}

#[test]
fn test_rebinding_does_not_share_store() {
    // Two unrelated arrays bound to the same name — the second
    // shadow has its own backing store.
    let source = r#"
def consume(arr: *[4]i32) i32 = arr[0]

def main: i32 =
    let x = [1, 2, 3, 4] in
    let _ = consume(x) in
    let x = [5, 6, 7, 8] in
    consume(x)
"#;
    let result = check_alias(source);
    assert!(
        !result.has_errors(),
        "shadowing introduces a fresh store, so consuming the new x is legal"
    );
}

#[test]
fn test_polymorphic_unique_param_consumes() {
    // A polymorphic function with a `*T` parameter still consumes its
    // argument — uniqueness has to flow through type variables.
    // Returns i32 (a Copy type) to avoid the unrelated `*T → T`
    // weakening gap that other tests cover.
    let source = r#"
def consume_poly<T>(x: *T) i32 = 0

def main(arr: [4]i32) i32 =
    let _ = consume_poly(arr) in
    arr[0]
"#;
    let result = check_alias(source);
    assert!(
        result.has_errors(),
        "polymorphic *T parameter should still trigger consumption"
    );
}

#[test]
fn test_polymorphic_unique_pass_through() {
    // Identity over `*T`: consume in, return out. Uniqueness should be
    // preserved through the type variable so the caller's site sees
    // the result as unique (and the original input as consumed).
    let source = r#"
def id_unique<T>(x: *T) *T = x

def main(arr: [4]i32) i32 =
    let r = id_unique(arr) in
    r[0]
"#;
    let result = check_alias(source);
    assert!(
        !result.has_errors(),
        "polymorphic *T identity is legal — caller consumed arr, then read r"
    );
}
