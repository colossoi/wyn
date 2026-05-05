use super::{Origin, OwnershipModel, analyze, build, eligible_consuming_soacs};
use crate::Compiler;
use crate::tlc::{Program, Term, TermKind};

#[test]
fn origin_mutability() {
    assert!(Origin::Fresh.is_mutable());
    assert!(Origin::UniqueParam.is_mutable());
    assert!(Origin::Entry.is_mutable());
    assert!(Origin::BorrowedMutableElement.is_mutable());
    assert!(!Origin::NonUniqueParam.is_mutable());
    assert!(!Origin::Borrowed.is_mutable());
}

#[test]
fn empty_model_lookups() {
    let model = OwnershipModel::new();
    assert!(model.var_to_owner.is_empty());
    assert!(model.origins.is_empty());
    assert!(model.uses.is_empty());
    assert!(model.kills.is_empty());
    assert!(model.defs.is_empty());
    assert!(model.live_out.is_empty());
}

fn compile_to_tlc(source: &str) -> Program {
    let mut frontend = crate::cached_frontend();
    let parsed = Compiler::parse(source, &mut frontend.node_counter).expect("parse");
    let type_checked = parsed
        .desugar(&mut frontend.node_counter)
        .expect("desugar")
        .resolve(&mut frontend.module_manager)
        .expect("resolve")
        .fold_ast_constants()
        .type_check(&mut frontend.module_manager, &mut frontend.schemes)
        .expect("type_check");
    let tlc = type_checked
        .to_tlc(&frontend.schemes, &frontend.module_manager, false)
        .partial_eval()
        .normalize_soacs();
    tlc.0.tlc
}

fn find_def<'a>(program: &'a Program, name: &str) -> &'a crate::tlc::Def {
    program
        .defs
        .iter()
        .find(|d| program.symbols.get(d.name).map(|s| s.as_str()) == Some(name))
        .unwrap_or_else(|| panic!("no def named {name}"))
}

fn param_origin(model: &OwnershipModel, def: &crate::tlc::Def, param_index: usize) -> Origin {
    let lam = match &def.body.kind {
        crate::tlc::TermKind::Lambda(l) => l,
        _ => panic!("def body is not a lambda"),
    };
    let (sym, _) = lam.params[param_index];
    let owner = model.owner_of(sym).expect("expected non-copy param to have an owner");
    model.origin(owner).expect("owner should have an origin")
}

#[test]
fn unique_param_origin_is_unique_param() {
    let program = compile_to_tlc(
        r#"
def f(a: *[4]i32) i32 = a[0]
"#,
    );
    let model = build(&program);
    let def = find_def(&program, "f");
    assert_eq!(param_origin(&model, def, 0), Origin::UniqueParam);
}

#[test]
fn non_unique_array_param_origin_is_non_unique_param() {
    let program = compile_to_tlc(
        r#"
def f(a: [4]i32) i32 = a[0]
"#,
    );
    let model = build(&program);
    let def = find_def(&program, "f");
    assert_eq!(param_origin(&model, def, 0), Origin::NonUniqueParam);
}

#[test]
fn copy_param_has_no_owner() {
    let program = compile_to_tlc(
        r#"
def f(x: i32) i32 = x + 1
"#,
    );
    let model = build(&program);
    let def = find_def(&program, "f");
    let lam = match &def.body.kind {
        crate::tlc::TermKind::Lambda(l) => l,
        _ => panic!("def body is not a lambda"),
    };
    let (sym, _) = lam.params[0];
    assert!(model.owner_of(sym).is_none());
}

/// Build a Let term by hand and run `build` on a synthesized program.
/// Bypasses partial_eval, which would otherwise inline trivial
/// `let x = y in body` aliases away before they reach our pass.
fn synth_program_with_alias_let() -> (Program, crate::SymbolId, crate::SymbolId) {
    use crate::ast::{Span, TypeName};
    use crate::tlc::{Def, DefMeta, Lambda, Term, TermIdSource, TermKind};
    use polytype::Type;

    let mut symbols = crate::SymbolTable::new();
    let mut ids = TermIdSource::new();

    let f_sym = symbols.alloc("f".to_string());
    let a_sym = symbols.alloc("a".to_string());
    let b_sym = symbols.alloc("b".to_string());

    // Type: *[4]i32  —  Unique(Array<i32>)
    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
    let arr_ty = Type::Constructed(TypeName::Array, vec![i32_ty.clone()]);
    let unique_arr_ty = Type::Constructed(TypeName::Unique, vec![arr_ty.clone()]);

    // Body: Let b = a in 0    (rhs `a` aliases `b`; `0` body keeps test trivial)
    let var_a = Term {
        id: ids.next_id(),
        ty: unique_arr_ty.clone(),
        span: Span::dummy(),
        kind: TermKind::Var(a_sym),
    };
    let body_zero = Term {
        id: ids.next_id(),
        ty: i32_ty.clone(),
        span: Span::dummy(),
        kind: TermKind::IntLit("0".to_string()),
    };
    let let_term = Term {
        id: ids.next_id(),
        ty: i32_ty.clone(),
        span: Span::dummy(),
        kind: TermKind::Let {
            name: b_sym,
            name_ty: unique_arr_ty.clone(),
            rhs: Box::new(var_a),
            body: Box::new(body_zero),
        },
    };

    let lam_body_id = let_term.id;
    let lambda = Lambda {
        params: vec![(a_sym, unique_arr_ty.clone())],
        body: Box::new(let_term),
        ret_ty: i32_ty.clone(),
    };
    let lambda_term = Term {
        id: ids.next_id(),
        ty: Type::Constructed(TypeName::Arrow, vec![unique_arr_ty, i32_ty]),
        span: Span::dummy(),
        kind: TermKind::Lambda(lambda),
    };

    let def = Def {
        name: f_sym,
        ty: lambda_term.ty.clone(),
        body: lambda_term,
        meta: DefMeta::Function,
        arity: 1,
    };
    let program = Program {
        defs: vec![def],
        uniforms: vec![],
        storage: vec![],
        symbols,
        def_syms: Default::default(),
    };
    let _ = lam_body_id; // silence unused (kept for documentation)
    (program, a_sym, b_sym)
}

#[test]
fn let_aliases_existing_owner() {
    let (program, a_sym, b_sym) = synth_program_with_alias_let();
    let model = build(&program);

    let a_owner = model.owner_of(a_sym).expect("`a` should have an owner");
    let b_owner = model.owner_of(b_sym).expect("`b` should have an owner");
    assert_eq!(a_owner, b_owner, "alias-let should share owner");

    // No defs anywhere — `b` aliases existing owner, never allocates fresh.
    assert!(
        model.defs.values().all(|s| s.is_empty()),
        "alias-let should not record a fresh owner under defs"
    );

    // Origin should be UniqueParam (inherited from `a`).
    assert_eq!(model.origin(a_owner), Some(super::Origin::UniqueParam));
}

#[test]
fn app_with_unique_arg_records_kill() {
    let program = compile_to_tlc(
        r#"
def consume(x: *[4]i32) i32 = x[0]
def main(a: *[4]i32) i32 = consume(a)
"#,
    );
    let model = build(&program);

    let main_def = find_def(&program, "main");
    let lam = match &main_def.body.kind {
        crate::tlc::TermKind::Lambda(l) => l,
        _ => panic!(),
    };
    let a_owner = model.owner_of(lam.params[0].0).unwrap();

    // The `consume(a)` App must record a kill of `a`'s owner.
    let killed_anywhere = model.kills.values().any(|set| set.contains(&a_owner));
    assert!(
        killed_anywhere,
        "expected `a`'s owner to be killed by the consume(a) App"
    );
}

#[test]
fn app_with_non_unique_arg_records_no_kill() {
    let program = compile_to_tlc(
        r#"
def borrow(x: [4]i32) i32 = x[0]
def main(a: *[4]i32) i32 = borrow(a)
"#,
    );
    let model = build(&program);

    let main_def = find_def(&program, "main");
    let lam = match &main_def.body.kind {
        crate::tlc::TermKind::Lambda(l) => l,
        _ => panic!(),
    };
    let a_owner = model.owner_of(lam.params[0].0).unwrap();

    // borrow takes T (no `*`) — no kill recorded for `a`.
    let killed_anywhere = model.kills.values().any(|set| set.contains(&a_owner));
    assert!(
        !killed_anywhere,
        "expected no kill — borrow takes a non-unique parameter"
    );
}

#[test]
fn soac_element_param_gets_borrowed_mutable_element_when_non_copy() {
    // Map over a 2D array — the lambda's element param has type [4]i32
    // (non-copy), so it should be tracked as a per-iteration view of a
    // mutable input.
    let program = compile_to_tlc(
        r#"
def f(rows: *[3][4]i32) [3][4]i32 = map(|row| row, rows)
"#,
    );
    let model = build(&program);

    let elem_count =
        model.origins.values().filter(|o| **o == super::Origin::BorrowedMutableElement).count();
    let unique_count = model.origins.values().filter(|o| **o == super::Origin::UniqueParam).count();
    assert!(unique_count >= 1, "expected at least one UniqueParam owner");
    assert!(
        elem_count >= 1,
        "expected at least one BorrowedMutableElement owner for the SOAC element param"
    );
}

#[test]
fn var_use_recorded_for_tracked_owner() {
    let program = compile_to_tlc(
        r#"
def f(a: *[4]i32) i32 = a[0]
"#,
    );
    let model = build(&program);

    let def = find_def(&program, "f");
    let lam = match &def.body.kind {
        crate::tlc::TermKind::Lambda(l) => l,
        _ => panic!(),
    };
    let owner = model.owner_of(lam.params[0].0).unwrap();

    // Some term in the body must record `owner` as a use.
    let used_anywhere = model.uses.values().any(|set| set.contains(&owner));
    assert!(used_anywhere, "expected owner of `a` to appear in a uses set");
}

// =============================================================================
// Backward liveness
// =============================================================================

/// Locate the App term whose func is a `Var` resolving to any of the
/// given names. Accepts both array_with intrinsic variants so the
/// helper works whether or not promotion has fired.
fn find_app_call_to<'a>(body: &'a Term, names: &[&str], program: &Program) -> Option<&'a Term> {
    fn name_of_var<'a>(t: &Term, program: &'a Program) -> Option<&'a str> {
        if let TermKind::Var(sym) = &t.kind { program.symbols.get(*sym).map(|s| s.as_str()) } else { None }
    }
    fn walk<'a>(t: &'a Term, names: &[&str], program: &Program) -> Option<&'a Term> {
        if let TermKind::App { func, args } = &t.kind {
            if let Some(n) = name_of_var(func, program) {
                if names.contains(&n) {
                    return Some(t);
                }
            }
            if let Some(hit) = walk(func, names, program) {
                return Some(hit);
            }
            for a in args {
                if let Some(hit) = walk(a, names, program) {
                    return Some(hit);
                }
            }
            return None;
        }
        match &t.kind {
            TermKind::Let { rhs, body, .. } => {
                walk(rhs, names, program).or_else(|| walk(body, names, program))
            }
            TermKind::If {
                cond,
                then_branch,
                else_branch,
            } => walk(cond, names, program)
                .or_else(|| walk(then_branch, names, program))
                .or_else(|| walk(else_branch, names, program)),
            TermKind::Lambda(lam) => walk(&lam.body, names, program),
            TermKind::Loop {
                init,
                init_bindings,
                body,
                ..
            } => {
                if let Some(hit) = walk(init, names, program) {
                    return Some(hit);
                }
                for (_, _, e) in init_bindings {
                    if let Some(hit) = walk(e, names, program) {
                        return Some(hit);
                    }
                }
                walk(body, names, program)
            }
            TermKind::Force(inner) => walk(inner, names, program),
            _ => None,
        }
    }
    walk(body, names, program)
}

const ARRAY_WITH_INTRINSICS: &[&str] = &["_w_intrinsic_array_with", "_w_intrinsic_array_with_inplace"];

#[test]
fn liveness_dead_after_array_with_when_source_is_last_use() {
    // a is used only inside the array_with — should not appear in
    // live_out at the array_with's term.
    let program = compile_to_tlc(
        r#"
def f(a: *[4]i32) *[4]i32 = a with [0] = 1
"#,
    );
    let model = analyze(&program);

    let def = find_def(&program, "f");
    let lam = match &def.body.kind {
        TermKind::Lambda(l) => l,
        _ => panic!(),
    };
    let a_owner = model.owner_of(lam.params[0].0).unwrap();

    let aw = find_app_call_to(&lam.body, ARRAY_WITH_INTRINSICS, &program)
        .expect("expected an array_with call in f's body");
    let live = model.live_out.get(&aw.id).expect("live_out missing");
    assert!(
        !live.contains(&a_owner),
        "expected `a`'s owner to be dead-after the array_with (live={:?})",
        live
    );
}

#[test]
fn liveness_alive_after_array_with_when_source_used_after() {
    // Non-unique source: `with` makes a fresh copy, doesn't consume.
    // The else-branch returns `a`, so liveness must report `a` as
    // live at the array_with's term. The promotion check would also
    // reject because `a`'s origin is NonUniqueParam, but we're
    // testing the liveness fact independently.
    let program = compile_to_tlc(
        r#"
def f(a: [4]i32, k: i32) [4]i32 =
    let b = a with [0] = 1 in
    if k > 0 then b else a
"#,
    );
    let model = analyze(&program);

    let def = find_def(&program, "f");
    let lam = match &def.body.kind {
        TermKind::Lambda(l) => l,
        _ => panic!(),
    };
    let a_owner = model.owner_of(lam.params[0].0).unwrap();

    let aw =
        find_app_call_to(&lam.body, ARRAY_WITH_INTRINSICS, &program).expect("expected an array_with call");
    let live = model.live_out.get(&aw.id).expect("live_out missing");
    assert!(
        live.contains(&a_owner),
        "expected `a`'s owner live after the array_with (else branch returns it); live={:?}",
        live
    );
}

#[test]
fn liveness_kill_inside_one_branch_does_not_leak_to_other() {
    // Else branch consumes `a`; then-branch reads `a[0]` without
    // consuming. After the if the function returns, so live_out at
    // the if = ∅. At the consume call (in else), live_out = ∅
    // (else's tail) — confirms the kill site sees no surviving uses.
    // At the if's own term, live_out = ∅ (function exit). Live_in at
    // the if's cond should still include `a`'s owner because both
    // branches reference it.
    let program = compile_to_tlc(
        r#"
def consume(x: *[4]i32) i32 = x[0]
def f(cond: bool, a: *[4]i32) i32 =
    if cond then a[0] else consume(a)
"#,
    );
    let model = analyze(&program);

    let def = find_def(&program, "f");
    let lam = match &def.body.kind {
        TermKind::Lambda(l) => l,
        _ => panic!(),
    };
    let a_owner = model.owner_of(lam.params[1].0).unwrap();

    let body_live = model.live_out.get(&lam.body.id).unwrap();
    assert!(body_live.is_empty(), "function body's live_out should be ∅");

    let consume_call = find_app_call_to(&lam.body, &["consume"], &program)
        .expect("expected a consume call in the else branch");
    let live = model.live_out.get(&consume_call.id).expect("live_out missing");
    assert!(
        !live.contains(&a_owner),
        "expected `a` dead after consume(a) — nothing in its branch's tail uses it; live={:?}",
        live
    );
}

#[test]
fn liveness_loop_body_use_propagates_via_fixed_point() {
    // Read `a` inside a for-loop body without consuming. The body
    // re-iterates and reads `a` again on each pass, so `a`'s owner
    // must be live across the body — that's the fixed-point's job.
    let program = compile_to_tlc(
        r#"
def f(a: [4]i32, n: i32) i32 =
    loop acc = 0 for i < n do
        acc + a[i]
"#,
    );
    let model = analyze(&program);

    let def = find_def(&program, "f");
    let lam = match &def.body.kind {
        TermKind::Lambda(l) => l,
        _ => panic!(),
    };
    let a_owner = model.owner_of(lam.params[0].0).unwrap();

    // Find the loop term inside the body. (The Lambda body's first
    // term should be the Loop — partial_eval can't strip it.)
    let loop_term = match &lam.body.kind {
        TermKind::Loop { .. } => &lam.body,
        _ => {
            // If the body is a Let or wrapper, drill in. Conservative:
            // walk via the structured matcher we already have.
            panic!("expected a loop at the body's top level");
        }
    };
    let loop_live = model.live_out.get(&loop_term.id).expect("loop live_out");

    // Outer caller has nothing live after the loop returns (function
    // exits with the loop's value), so loop_live is ∅ at the loop's
    // OWN site. The interesting fact is inside the body: the body's
    // live_in must contain `a`'s owner (uses on every iteration).
    assert!(
        loop_live.is_empty(),
        "loop's outer live_out should be ∅; got {:?}",
        loop_live
    );

    // Verify `a`'s owner appears in some live_out inside the body —
    // the fixed-point should have propagated it.
    let any_alive = model.live_out.values().any(|set| set.contains(&a_owner));
    assert!(
        any_alive,
        "expected `a`'s owner to be live somewhere inside the loop body via fixed-point"
    );
}

// =============================================================================
// Use-after-move diagnostics — exercise `tlc::ownership::check`
// =============================================================================

/// Run the front-end through to the point where ownership analysis
/// can be invoked, then return whether the program has a
/// use-after-move violation.
fn has_use_after_move(source: &str) -> bool {
    let mut frontend = crate::cached_frontend();
    let parsed = Compiler::parse(source, &mut frontend.node_counter).expect("parse");
    let type_checked = parsed
        .desugar(&mut frontend.node_counter)
        .expect("desugar")
        .resolve(&mut frontend.module_manager)
        .expect("resolve")
        .fold_ast_constants()
        .type_check(&mut frontend.module_manager, &mut frontend.schemes)
        .expect("type_check");
    let tlc = type_checked
        .to_tlc(&frontend.schemes, &frontend.module_manager, false)
        .partial_eval()
        .normalize_soacs();
    super::check(&tlc.0.tlc).is_err()
}

#[test]
fn no_error_simple_arithmetic() {
    assert!(!has_use_after_move(r#"def main(x: i32) i32 = x + 1"#));
}

#[test]
fn use_after_move_consume_then_use() {
    let source = r#"
def consume(arr: *[4]i32) i32 = arr[0]
def main(arr: *[4]i32) i32 =
    let _ = consume(arr) in
    arr[0]
"#;
    assert!(has_use_after_move(source));
}

#[test]
fn use_after_move_via_let_alias() {
    let source = r#"
def consume(arr: *[4]i32) i32 = arr[0]
def main(arr: *[4]i32) i32 =
    let alias = arr in
    let _ = consume(arr) in
    alias[0]
"#;
    assert!(has_use_after_move(source));
}

#[test]
fn copy_type_no_use_after_move() {
    let source = r#"
def main(x: i32) i32 =
    let y = x in
    let z = x in
    y + z
"#;
    assert!(!has_use_after_move(source));
}

// Inner shadowed `x` is its own owner; the outer `x` is untouched
// by `consume(inner_x)`. The TLC transformer's `LetIn` arm
// snapshots and restores `scope` around the body so the inner
// `let x` doesn't leak out and pollute the outer body's `Var(x)`
// resolution.
#[test]
fn shadowing_does_not_violate() {
    let source = r#"
def consume(arr: *[4]i32) i32 = arr[0]
def main(arr: *[4]i32) i32 =
    let x = arr in
    let result =
        let x = [1, 2, 3, 4] in
        consume(x)
    in
    x[0]
"#;
    assert!(!has_use_after_move(source));
}

#[test]
fn self_aliasing_in_unique_args() {
    // Passing the same backing store to two `*T` parameters in the
    // same call consumes it twice. Sound treatment is to reject.
    let source = r#"
def both(a: *[4]i32, b: *[4]i32) i32 = a[0] + b[0]
def main(arr: *[4]i32) i32 = both(arr, arr)
"#;
    assert!(
        has_use_after_move(source),
        "passing arr to two `*T` params in one call should be rejected"
    );
}

#[test]
fn lambda_kill_of_capture() {
    // Lambdas may be invoked any number of times. A body that kills
    // a captured store turns one consumption into N. Sound treatment
    // is to reject any kill of a capture.
    let source = r#"
def consume(arr: *[4]i32) i32 = arr[0]
def main(arr: *[4]i32) i32 =
    let f = |x: i32| consume(arr) in
    f(0) + f(1)
"#;
    assert!(
        has_use_after_move(source),
        "a lambda body that consumes a capture should be rejected"
    );
}

#[test]
fn use_after_move_consume_alias_use_original() {
    let source = r#"
def consume(arr: *[4]i32) i32 = arr[0]
def main(arr: *[4]i32) i32 =
    let alias = arr in
    let _ = consume(alias) in
    arr[0]
"#;
    assert!(has_use_after_move(source));
}

// =============================================================================
// Aliasing intrinsics + Borrowed origin
// =============================================================================

/// Walk `f`'s body looking for a `Let { name, ... }` whose name's
/// symbol resolves to `var_name`, and return that name's owner +
/// origin. Scoping the search inside the named def avoids
/// false-positive matches against prelude symbols with the same
/// surface name.
fn binder_origin(program: &Program, fn_name: &str, var_name: &str) -> (super::OwnerId, Origin) {
    fn find_let_sym(t: &Term, var_name: &str, program: &Program) -> Option<crate::SymbolId> {
        if let TermKind::Let { name, .. } = &t.kind {
            if program.symbols.get(*name).map(|s| s.as_str()) == Some(var_name) {
                return Some(*name);
            }
        }
        let mut found = None;
        match &t.kind {
            TermKind::Let { rhs, body, .. } => {
                found =
                    find_let_sym(rhs, var_name, program).or_else(|| find_let_sym(body, var_name, program));
            }
            TermKind::App { func, args } => {
                found = find_let_sym(func, var_name, program);
                for a in args {
                    if found.is_some() {
                        break;
                    }
                    found = find_let_sym(a, var_name, program);
                }
            }
            TermKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                found = find_let_sym(cond, var_name, program)
                    .or_else(|| find_let_sym(then_branch, var_name, program))
                    .or_else(|| find_let_sym(else_branch, var_name, program));
            }
            TermKind::Lambda(lam) => {
                found = find_let_sym(&lam.body, var_name, program);
            }
            TermKind::Loop { init, body, .. } => {
                found =
                    find_let_sym(init, var_name, program).or_else(|| find_let_sym(body, var_name, program));
            }
            TermKind::Force(inner) => {
                found = find_let_sym(inner, var_name, program);
            }
            _ => {}
        }
        found
    }

    let f_def = find_def(program, fn_name);
    let model = build(program);
    let sym = find_let_sym(&f_def.body, var_name, program).unwrap_or_else(|| {
        panic!(
            "no let-binding named `{}` in `{}` (partial_eval may have inlined it)",
            var_name, fn_name
        )
    });
    let owner = model
        .owner_of(sym)
        .unwrap_or_else(|| panic!("`{}` should have an owner in `{}`", var_name, fn_name));
    let origin =
        model.origin(owner).unwrap_or_else(|| panic!("owner of `{}` should have an origin", var_name));
    (owner, origin)
}

#[test]
fn array_index_aliases_base_when_result_is_non_copy() {
    // `let row = grid[i]` — row should share grid's owner because
    // `_w_index` returns a view, not a fresh allocation.
    let program = compile_to_tlc(
        r#"
def f(grid: *[3][4]i32, i: i32) i32 =
    let row = grid[i] in row[0]
"#,
    );
    let model = build(&program);

    let f_def = find_def(&program, "f");
    let lam = match &f_def.body.kind {
        TermKind::Lambda(l) => l,
        _ => panic!(),
    };
    let grid_sym = lam.params[0].0;
    let grid_owner = model.owner_of(grid_sym).unwrap();

    let row_sym = program
        .symbols
        .iter()
        .find_map(|(s, n)| if n == "row" { Some(*s) } else { None })
        .expect("symbol `row` not found");
    let row_owner = model.owner_of(row_sym).expect("`row` should have an owner");

    assert_eq!(grid_owner, row_owner, "row = grid[i] should alias grid's owner",);
}

#[test]
fn unrecognized_call_returning_non_copy_is_borrowed() {
    // `view` returns its arg; we don't track that interprocedurally,
    // so `r = view(arr)` must be Borrowed (immutable) — refusing
    // promotion is sound under-approximation.
    let program = compile_to_tlc(
        r#"
def view(a: [4]i32) [4]i32 = a
def main(arr: [4]i32) i32 =
    let r = view(arr) in r[0]
"#,
    );
    let (_, origin) = binder_origin(&program, "main", "r");
    assert_eq!(
        origin,
        Origin::Borrowed,
        "result of an unrecognized non-copy call should be Borrowed",
    );
}

#[test]
fn unique_returning_call_is_fresh() {
    // A function that declares `*[4]i32` return is fresh by Wyn's
    // uniqueness contract — no inter-procedural analysis required.
    let program = compile_to_tlc(
        r#"
def bump(a: *[4]i32) *[4]i32 = a with [0] = 1
def main(a: *[4]i32) i32 =
    let r = bump(a) in r[0]
"#,
    );
    let (_, origin) = binder_origin(&program, "main", "r");
    assert_eq!(
        origin,
        Origin::Fresh,
        "result of a `*T`-returning call should be Fresh",
    );
}

#[test]
fn array_literal_let_is_fresh() {
    // `_w_array_lit` is a recognized fresh-producer, so promotion
    // stays open downstream. Two distinct indices keep partial_eval
    // from inlining the let away.
    let program = compile_to_tlc(
        r#"
def main(i: i32, j: i32) i32 =
    let arr = [1, 2, 3, 4] in arr[i] + arr[j]
"#,
    );
    let (_, origin) = binder_origin(&program, "main", "arr");
    assert_eq!(origin, Origin::Fresh);
}

#[test]
fn map_element_param_mutable_when_input_unique() {
    // Symmetric to the gap test: map over a `*` 2D array. The
    // element view aliases mutable memory the caller surrendered, so
    // the element param is itself mutable — preserves promotion of
    // an inner `with` inside the body for in-place SOACs.
    //
    // The declared return is `[3][4]i32`, not `*[3][4]i32`, because
    // map's polymorphic signature currently strips `*` at the
    // parameter site, so `map(|row| row, *arr)` infers a non-unique
    // result. Once DPS codegen for in-place SOACs preserves `*` on
    // the output (or the type checker grows uniqueness inference
    // through SOACs), the return type here can become `*[3][4]i32`.
    let program = compile_to_tlc(
        r#"
def main(arr: *[3][4]i32) [3][4]i32 = map(|row| row, arr)
"#,
    );
    let model = build(&program);
    let main_def = find_def(&program, "main");
    fn first_map_elem_param(t: &Term) -> Option<crate::SymbolId> {
        if let TermKind::Soac(crate::tlc::SoacOp::Map { lam, .. }) = &t.kind {
            return Some(lam.lam.params[0].0);
        }
        let mut found = None;
        t.for_each_child(&mut |child| {
            if found.is_none() {
                found = first_map_elem_param(child);
            }
        });
        found
    }
    let row_sym = first_map_elem_param(&main_def.body).expect("expected a Map SOAC");
    let row_owner = model.owner_of(row_sym).expect("row should have an owner");
    let row_origin = model.origin(row_owner).expect("origin");
    assert!(
        row_origin.is_mutable(),
        "map's element param over `*` input should be mutable; got {:?}",
        row_origin,
    );
}

#[test]
fn map_element_param_inherits_input_mutability() {
    // Map over a non-unique 2D array. The element param `row` IS
    // a slot in the caller's `arr` — mutating it would clobber the
    // caller's data. Element-param origin must reflect arr's
    // non-mutability (Borrowed), not bare Fresh.
    let program = compile_to_tlc(
        r#"
def main(arr: [3][4]i32) [3][4]i32 = map(|row| row, arr)
"#,
    );
    let model = build(&program);
    let main_def = find_def(&program, "main");
    // Walk main's body looking for a Soac::Map and inspect its
    // first lambda param's owner.
    fn first_map_elem_param(t: &Term) -> Option<crate::SymbolId> {
        if let TermKind::Soac(crate::tlc::SoacOp::Map { lam, .. }) = &t.kind {
            return Some(lam.lam.params[0].0);
        }
        let mut found = None;
        t.for_each_child(&mut |child| {
            if found.is_none() {
                found = first_map_elem_param(child);
            }
        });
        found
    }
    let row_sym = first_map_elem_param(&main_def.body).expect("expected a Map SOAC in main's body");
    let row_owner = model.owner_of(row_sym).expect("row should have an owner");
    let row_origin = model.origin(row_owner).expect("row owner should have an origin");
    assert!(
        !row_origin.is_mutable(),
        "map's element param over non-unique input should be immutable; got {:?}",
        row_origin,
    );
}

// =============================================================================
// Consuming-SOAC eligibility query (Phase B)
// =============================================================================

#[test]
fn eligible_consuming_map_when_input_dead_and_unique() {
    // `*[N]T` input, no further use after the map. Should qualify.
    let program = compile_to_tlc(
        r#"
def f(a: *[3][4]i32) [3][4]i32 = map(|row| row, a)
"#,
    );
    let model = analyze(&program);
    let eligible = eligible_consuming_soacs(&program, &model);
    assert_eq!(
        eligible.len(),
        1,
        "expected exactly one eligible Map; got {}: {:?}",
        eligible.len(),
        eligible,
    );
}

#[test]
fn not_eligible_when_input_non_unique() {
    // Same shape but with a non-unique input. Element view is
    // Borrowed (caller still owns), so the input owner isn't
    // mutable from this function's perspective.
    let program = compile_to_tlc(
        r#"
def f(a: [3][4]i32) [3][4]i32 = map(|row| row, a)
"#,
    );
    let model = analyze(&program);
    let eligible = eligible_consuming_soacs(&program, &model);
    assert!(
        eligible.is_empty(),
        "non-unique input should not be eligible; got {:?}",
        eligible,
    );
}

#[test]
fn not_eligible_when_input_used_after() {
    // `*[N]T` input but a borrowing function reads `a` after the
    // map, so the input is live after the SOAC's term.
    let program = compile_to_tlc(
        r#"
def borrow(b: [3][4]i32) i32 = b[0][0]
def f(a: *[3][4]i32) i32 =
    let m = map(|row| row, a) in
    borrow(a) + m[0][0]
"#,
    );
    let model = analyze(&program);
    let eligible = eligible_consuming_soacs(&program, &model);
    assert!(
        eligible.is_empty(),
        "input still alive after map should not be eligible; got {:?}",
        eligible,
    );
}

#[test]
fn not_eligible_when_body_changes_element_type() {
    // Output element type (bool) differs from input element type
    // (i32). In-place rewrite cannot apply — different size, can't
    // reuse the buffer.
    let program = compile_to_tlc(
        r#"
def f(a: *[8]i32) [8]bool = map(|x: i32| x > 0, a)
"#,
    );
    let model = analyze(&program);
    let eligible = eligible_consuming_soacs(&program, &model);
    assert!(
        eligible.is_empty(),
        "element-type-changing map should not be eligible; got {:?}",
        eligible,
    );
}

#[test]
fn not_eligible_when_body_reads_input_outside_elem_param() {
    // Stencil pattern: `map(|x| x + a[i-1], a)`. In-place mutation
    // at index i would change later iterations' reads of `a[j]` for
    // j > i. The pointwise check rejects.
    let program = compile_to_tlc(
        r#"
def f(a: *[8]i32) [8]i32 =
    map(|x: i32| x + a[0], a)
"#,
    );
    let model = analyze(&program);
    let eligible = eligible_consuming_soacs(&program, &model);
    assert!(
        eligible.is_empty(),
        "map with capture-read of input should not be eligible; got {:?}",
        eligible,
    );
}

#[test]
fn not_eligible_when_soac_is_entry_output() {
    // Compute entry whose result is a Map and is bound to a
    // storage output. The output-side rewrite handles this; an
    // input-side rewrite would clobber the runtime output buffer.
    // Conservative skip.
    let program = compile_to_tlc(
        r#"
#[compute]
entry double(arr: []i32) []i32 = map(|x: i32| x + 1, arr)
"#,
    );
    let model = analyze(&program);
    let eligible = eligible_consuming_soacs(&program, &model);
    assert!(
        eligible.is_empty(),
        "entry-output Map should not be eligible (output-side rewrite handles it); got {:?}",
        eligible,
    );
}

// =============================================================================
// `consumes_input` flag flipping by apply_ownership (Phase C)
// =============================================================================

/// Drive the program through to `TlcOwnershipApplied` (post-fusion,
/// post-apply_ownership). Returns the post-rewrite Program for
/// inspecting `SoacOp::Map { consumes_input, .. }`.
fn compile_to_owned(source: &str) -> Program {
    let mut frontend = crate::cached_frontend();
    let parsed = Compiler::parse(source, &mut frontend.node_counter).expect("parse");
    let type_checked = parsed
        .desugar(&mut frontend.node_counter)
        .expect("desugar")
        .resolve(&mut frontend.module_manager)
        .expect("resolve")
        .fold_ast_constants()
        .type_check(&mut frontend.module_manager, &mut frontend.schemes)
        .expect("type_check");
    let owned = type_checked
        .to_tlc(&frontend.schemes, &frontend.module_manager, false)
        .partial_eval()
        .normalize_soacs()
        .fuse_maps()
        .apply_ownership()
        .expect("apply_ownership");
    owned.0.tlc
}

fn map_consumes_input(program: &Program, fn_name: &str) -> Option<bool> {
    fn walk(t: &Term) -> Option<bool> {
        if let TermKind::Soac(crate::tlc::SoacOp::Map { consumes_input, .. }) = &t.kind {
            return Some(*consumes_input);
        }
        let mut found = None;
        t.for_each_child(&mut |child| {
            if found.is_none() {
                found = walk(child);
            }
        });
        found
    }
    let def = find_def(program, fn_name);
    walk(&def.body)
}

#[test]
fn consumes_input_flag_set_for_eligible_map() {
    let program = compile_to_owned(
        r#"
def f(a: *[3][4]i32) [3][4]i32 = map(|row| row, a)
"#,
    );
    assert_eq!(
        map_consumes_input(&program, "f"),
        Some(true),
        "eligible Map should have consumes_input = true after apply_ownership",
    );
}

#[test]
fn consumes_input_flag_not_set_for_non_unique_input() {
    let program = compile_to_owned(
        r#"
def f(a: [3][4]i32) [3][4]i32 = map(|row| row, a)
"#,
    );
    assert_eq!(
        map_consumes_input(&program, "f"),
        Some(false),
        "Map over non-unique input should keep consumes_input = false",
    );
}

// =============================================================================
// SoacBody captures (SOAC envelope shape)
// =============================================================================
//
// `Lambda` no longer carries captures — those live on the `SoacBody`
// envelope used by SOAC nodes. The previous test harness here built a
// hand-rolled "post-defunc free-standing lambda" with populated
// `Lambda::captures` to exercise the analysis seam; that shape is
// architecturally unrepresentable now, so the harness and its two
// tests are gone. The SOAC-side capture handling is exercised by the
// end-to-end SOAC tests (e.g. `compute_shader_*` in the integration
// suite) that lower a Map / Reduce with closures through defunc into
// `SoacBody { captures: [...] }`.

// =============================================================================
// SOAC body fixed-point: per-iteration locals must not loop back
// =============================================================================

#[test]
fn map_body_consuming_element_param_is_accepted() {
    // Each iteration of `map` receives a fresh element. A body
    // that consumes that element via a `*T` call is sound: the
    // consumption applies to one runtime value per iteration, and
    // the next iteration receives a different value.
    //
    // This test is the false-positive case for an over-conservative
    // SOAC fixed-point: if `lambda_body_fixed_point` doesn't
    // subtract the element-param owner from the loop-back set, it
    // treats the param as carried across iterations and the
    // body's kill conflicts with the carried-live owner.
    let source = r#"
def consume(x: *[4]i32) i32 = x[0]
def main(rows: *[3][4]i32) [3]i32 = map(|row: [4]i32| consume(row), rows)
"#;
    assert!(
        !has_use_after_move(source),
        "SOAC body consuming its per-iteration element param is sound \
         (each iteration's element is a fresh runtime value); \
         analysis should accept",
    );
}

// =============================================================================
// `_w_intrinsic_array_with` promotion threads through aliasing intrinsics
// =============================================================================

/// Synthesize a TLC program shaped like:
///
/// ```text
/// def f(grid: *[2][4]i32) [4]i32 =
///     _w_intrinsic_array_with(_w_index(grid, 0), 0, 99)
/// ```
///
/// The functional `_w_intrinsic_array_with`'s source-arg is *not* a
/// plain `Var(sym)` — it's an aliasing intrinsic application
/// (`_w_index(grid, 0)`). The build pass's `alias_target` recognizes
/// `_w_index` and threads through it to grid's owner. Promotion at
/// the with-call must do the same query if it is to remain consistent
/// with the rest of the model: the underlying owner is `grid`, which
/// is `UniqueParam` (mutable), and it's dead after the call (the
/// function returns the with's result). So the call should rewrite
/// to `_w_intrinsic_array_with_inplace`.
fn synth_program_with_with_through_index() -> Program {
    use crate::ast::{Span, TypeName};
    use crate::tlc::{Def, DefMeta, Lambda, Term, TermIdSource, TermKind};
    use polytype::Type;

    let mut symbols = crate::SymbolTable::new();
    let mut def_syms = std::collections::HashMap::new();
    let mut ids = TermIdSource::new();

    // Top-level symbols
    let f_sym = symbols.alloc("f".to_string());
    def_syms.insert("f".to_string(), f_sym);
    let with_sym = symbols.alloc(crate::intrinsics::INTRINSIC_ARRAY_WITH.to_string());
    def_syms.insert(crate::intrinsics::INTRINSIC_ARRAY_WITH.to_string(), with_sym);
    let index_sym = symbols.alloc("_w_index".to_string());
    def_syms.insert("_w_index".to_string(), index_sym);

    // Local symbols
    let grid_sym = symbols.alloc("grid".to_string());

    // Types
    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
    let inner_arr_ty = Type::Constructed(TypeName::Array, vec![i32_ty.clone()]);
    let outer_arr_ty = Type::Constructed(TypeName::Array, vec![inner_arr_ty.clone()]);
    let unique_outer_ty = Type::Constructed(TypeName::Unique, vec![outer_arr_ty.clone()]);

    // _w_index : *[2][4]i32 -> i32 -> [4]i32
    let index_fn_ty = Type::Constructed(
        TypeName::Arrow,
        vec![
            unique_outer_ty.clone(),
            Type::Constructed(TypeName::Arrow, vec![i32_ty.clone(), inner_arr_ty.clone()]),
        ],
    );
    // _w_intrinsic_array_with : [4]i32 -> i32 -> i32 -> [4]i32
    let with_fn_ty = Type::Constructed(
        TypeName::Arrow,
        vec![
            inner_arr_ty.clone(),
            Type::Constructed(
                TypeName::Arrow,
                vec![
                    i32_ty.clone(),
                    Type::Constructed(TypeName::Arrow, vec![i32_ty.clone(), inner_arr_ty.clone()]),
                ],
            ),
        ],
    );

    // _w_index(grid, 0)
    let var_index = Term {
        id: ids.next_id(),
        ty: index_fn_ty.clone(),
        span: Span::dummy(),
        kind: TermKind::Var(index_sym),
    };
    let var_grid = Term {
        id: ids.next_id(),
        ty: unique_outer_ty.clone(),
        span: Span::dummy(),
        kind: TermKind::Var(grid_sym),
    };
    let zero_idx = Term {
        id: ids.next_id(),
        ty: i32_ty.clone(),
        span: Span::dummy(),
        kind: TermKind::IntLit("0".to_string()),
    };
    let index_call = Term {
        id: ids.next_id(),
        ty: inner_arr_ty.clone(),
        span: Span::dummy(),
        kind: TermKind::App {
            func: Box::new(var_index),
            args: vec![var_grid, zero_idx],
        },
    };

    // _w_intrinsic_array_with(index_call, 0, 99)
    let var_with = Term {
        id: ids.next_id(),
        ty: with_fn_ty.clone(),
        span: Span::dummy(),
        kind: TermKind::Var(with_sym),
    };
    let zero_with_idx = Term {
        id: ids.next_id(),
        ty: i32_ty.clone(),
        span: Span::dummy(),
        kind: TermKind::IntLit("0".to_string()),
    };
    let val_99 = Term {
        id: ids.next_id(),
        ty: i32_ty.clone(),
        span: Span::dummy(),
        kind: TermKind::IntLit("99".to_string()),
    };
    let with_call = Term {
        id: ids.next_id(),
        ty: inner_arr_ty.clone(),
        span: Span::dummy(),
        kind: TermKind::App {
            func: Box::new(var_with),
            args: vec![index_call, zero_with_idx, val_99],
        },
    };

    // Lambda(grid: *[2][4]i32) -> [4]i32 = with_call
    let lambda = Lambda {
        params: vec![(grid_sym, unique_outer_ty.clone())],
        body: Box::new(with_call),
        ret_ty: inner_arr_ty.clone(),
    };
    let f_ty = Type::Constructed(
        TypeName::Arrow,
        vec![unique_outer_ty.clone(), inner_arr_ty.clone()],
    );
    let lambda_term = Term {
        id: ids.next_id(),
        ty: f_ty.clone(),
        span: Span::dummy(),
        kind: TermKind::Lambda(lambda),
    };
    let f_def = Def {
        name: f_sym,
        ty: f_ty,
        body: lambda_term,
        meta: DefMeta::Function,
        arity: 1,
    };

    Program {
        defs: vec![f_def],
        uniforms: vec![],
        storage: vec![],
        symbols,
        def_syms,
    }
}

#[test]
fn array_with_promotes_when_source_is_aliasing_intrinsic() {
    // Pre-rewrite: the body's outer App is `_w_intrinsic_array_with(...)`.
    // Post-rewrite: `apply_ownership` should rewrite the func var to
    // `_w_intrinsic_array_with_inplace` because the source's underlying
    // owner (grid) is mutable and dead-after.
    let program = synth_program_with_with_through_index();

    // Sanity: pre-rewrite uses the functional form.
    let pre_func_name = {
        let f_def = find_def(&program, "f");
        let lam = match &f_def.body.kind {
            TermKind::Lambda(l) => l,
            _ => panic!("f's body is not a lambda"),
        };
        let TermKind::App { func, .. } = &lam.body.kind else {
            panic!("expected lambda body to be an App");
        };
        let TermKind::Var(s) = &func.kind else {
            panic!("expected App's func to be a Var");
        };
        program.symbols.get(*s).cloned().unwrap_or_default()
    };
    assert_eq!(
        pre_func_name,
        crate::intrinsics::INTRINSIC_ARRAY_WITH,
        "test setup: outer App must call the functional with intrinsic"
    );

    let rewritten = super::apply_ownership(program).expect("apply_ownership");
    let f_def = find_def(&rewritten, "f");
    let lam = match &f_def.body.kind {
        TermKind::Lambda(l) => l,
        _ => panic!("f's body is not a lambda"),
    };
    let TermKind::App { func, .. } = &lam.body.kind else {
        panic!("expected lambda body to be an App after rewrite");
    };
    let TermKind::Var(s) = &func.kind else {
        panic!("expected rewritten App's func to be a Var");
    };
    let post_func_name = rewritten.symbols.get(*s).cloned().unwrap_or_default();
    assert_eq!(
        post_func_name,
        crate::intrinsics::INTRINSIC_ARRAY_WITH_INPLACE,
        "promotion must thread through `_w_index` to grid's owner — \
         source-arg is `_w_index(grid, 0)`, which aliases grid \
         (UniqueParam, dead-after); is_promotable should follow \
         alias_target and rewrite, not bail because the source arg \
         isn't a bare Var",
    );
}

// =============================================================================
// SoacBody captures (SOAC envelope shape, post-defunc)
// =============================================================================
//
// Captures live on the SoacBody wrapper (alongside the inner Lambda),
// not on the Lambda struct itself. These tests construct a
// SoacOp::Map whose `lam: SoacBody` carries a populated `captures`
// list, then exercise the ownership analysis to confirm capture-kill
// detection and capture-term liveness propagate through.

fn synth_program_with_populated_soac_captures() -> Program {
    use crate::ast::{Span, TypeName};
    use crate::tlc::{Def, DefMeta, Lambda, SoacBody, Term, TermIdSource, TermKind};
    use polytype::Type;

    let mut symbols = crate::SymbolTable::new();
    let mut ids = TermIdSource::new();
    let mut def_syms = std::collections::HashMap::new();

    let consume_sym = symbols.alloc("consume".to_string());
    def_syms.insert("consume".to_string(), consume_sym);
    let main_sym = symbols.alloc("main".to_string());
    def_syms.insert("main".to_string(), main_sym);

    let consume_arg_sym = symbols.alloc("x".to_string());
    let outer_sym = symbols.alloc("outer".to_string());
    let f_sym = symbols.alloc("f".to_string());
    let lambda_param_sym = symbols.alloc("_x".to_string());
    let cap_sym = symbols.alloc("cap".to_string());

    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
    let arr_ty = Type::Constructed(TypeName::Array, vec![i32_ty.clone()]);
    let unique_arr_ty = Type::Constructed(TypeName::Unique, vec![arr_ty.clone()]);
    let consume_ty = Type::Constructed(TypeName::Arrow, vec![unique_arr_ty.clone(), i32_ty.clone()]);

    let consume_body = Term {
        id: ids.next_id(),
        ty: i32_ty.clone(),
        span: Span::dummy(),
        kind: TermKind::IntLit("0".to_string()),
    };
    let consume_lam = Lambda {
        params: vec![(consume_arg_sym, unique_arr_ty.clone())],
        body: Box::new(consume_body),
        ret_ty: i32_ty.clone(),
    };
    let consume_lam_term = Term {
        id: ids.next_id(),
        ty: consume_ty.clone(),
        span: Span::dummy(),
        kind: TermKind::Lambda(consume_lam),
    };
    let consume_def = Def {
        name: consume_sym,
        ty: consume_ty.clone(),
        body: consume_lam_term,
        meta: DefMeta::Function,
        arity: 1,
    };

    let var_consume = Term {
        id: ids.next_id(),
        ty: consume_ty.clone(),
        span: Span::dummy(),
        kind: TermKind::Var(consume_sym),
    };
    let var_cap = Term {
        id: ids.next_id(),
        ty: unique_arr_ty.clone(),
        span: Span::dummy(),
        kind: TermKind::Var(cap_sym),
    };
    let consume_call = Term {
        id: ids.next_id(),
        ty: i32_ty.clone(),
        span: Span::dummy(),
        kind: TermKind::App {
            func: Box::new(var_consume),
            args: vec![var_cap],
        },
    };

    let var_outer = Term {
        id: ids.next_id(),
        ty: unique_arr_ty.clone(),
        span: Span::dummy(),
        kind: TermKind::Var(outer_sym),
    };

    let lambda = Lambda {
        params: vec![(lambda_param_sym, i32_ty.clone())],
        body: Box::new(consume_call),
        ret_ty: i32_ty.clone(),
    };

    let zero_lit = Term {
        id: ids.next_id(),
        ty: i32_ty.clone(),
        span: Span::dummy(),
        kind: TermKind::IntLit("0".to_string()),
    };
    let one_lit = Term {
        id: ids.next_id(),
        ty: i32_ty.clone(),
        span: Span::dummy(),
        kind: TermKind::IntLit("1".to_string()),
    };
    let range_input = crate::tlc::ArrayExpr::Range {
        start: Box::new(zero_lit),
        len: Box::new(one_lit),
    };

    // SoacBody carries (cap_sym, *[4]i32, Var(outer_sym)) — the capture
    // local plus the outer term that supplies its value at SOAC creation.
    let map_term = Term {
        id: ids.next_id(),
        ty: Type::Constructed(TypeName::Array, vec![i32_ty.clone()]),
        span: Span::dummy(),
        kind: TermKind::Soac(crate::tlc::SoacOp::Map {
            lam: SoacBody {
                lam: lambda,
                captures: vec![(cap_sym, unique_arr_ty.clone(), var_outer)],
            },
            inputs: vec![range_input],
            consumes_input: false,
        }),
    };

    let inner_let_body = Term {
        id: ids.next_id(),
        ty: i32_ty.clone(),
        span: Span::dummy(),
        kind: TermKind::IntLit("0".to_string()),
    };
    let inner_let = Term {
        id: ids.next_id(),
        ty: i32_ty.clone(),
        span: Span::dummy(),
        kind: TermKind::Let {
            name: f_sym,
            name_ty: Type::Constructed(TypeName::Array, vec![i32_ty.clone()]),
            rhs: Box::new(map_term),
            body: Box::new(inner_let_body),
        },
    };

    fn int_lit(ids: &mut TermIdSource, n: &str) -> Term {
        Term {
            id: ids.next_id(),
            ty: Type::Constructed(TypeName::Int(32), vec![]),
            span: Span::dummy(),
            kind: TermKind::IntLit(n.to_string()),
        }
    }
    let arr_lit = Term {
        id: ids.next_id(),
        ty: unique_arr_ty.clone(),
        span: Span::dummy(),
        kind: TermKind::ArrayExpr(crate::tlc::ArrayExpr::Literal(vec![
            int_lit(&mut ids, "1"),
            int_lit(&mut ids, "2"),
            int_lit(&mut ids, "3"),
            int_lit(&mut ids, "4"),
        ])),
    };
    let outer_let = Term {
        id: ids.next_id(),
        ty: i32_ty.clone(),
        span: Span::dummy(),
        kind: TermKind::Let {
            name: outer_sym,
            name_ty: unique_arr_ty.clone(),
            rhs: Box::new(arr_lit),
            body: Box::new(inner_let),
        },
    };

    let main_def = Def {
        name: main_sym,
        ty: i32_ty.clone(),
        body: outer_let,
        meta: DefMeta::Function,
        arity: 0,
    };

    Program {
        defs: vec![consume_def, main_def],
        uniforms: vec![],
        storage: vec![],
        symbols,
        def_syms,
    }
}

#[test]
fn soac_with_populated_captures_detects_capture_kill() {
    // SOAC envelope shape: SoacOp::Map { lam: SoacBody { lam, captures }, .. }.
    // The lambda body consumes `cap` (a populated capture); analysis must
    // propagate the kill to `outer`'s owner and reject the program — a
    // SOAC body that consumes a capture across hypothetical re-invocations
    // is a use-after-move.
    let program = synth_program_with_populated_soac_captures();
    let result = super::check(&program);
    assert!(
        result.is_err(),
        "SOAC body that consumes a populated capture should be rejected; got {:?}",
        result,
    );
}

#[test]
fn soac_capture_term_is_analyzed_for_liveness() {
    // The capture term itself is a sub-expression with its own
    // TermId. Liveness must flow through it — otherwise reads/kills
    // inside the capture term go unchecked. We assert by building a
    // program with a populated capture and confirming `live_out` is
    // populated for the capture term's id after `analyze`.
    use crate::ast::{Span, TypeName};
    use crate::tlc::{Def, DefMeta, Lambda, SoacBody, Term, TermIdSource, TermKind};
    use polytype::Type;

    let mut symbols = crate::SymbolTable::new();
    let mut ids = TermIdSource::new();

    let main_sym = symbols.alloc("main".to_string());
    let outer_sym = symbols.alloc("outer".to_string());
    let f_sym = symbols.alloc("f".to_string());
    let cap_sym = symbols.alloc("cap".to_string());
    let lambda_param_sym = symbols.alloc("_x".to_string());

    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
    let arr_ty = Type::Constructed(TypeName::Array, vec![i32_ty.clone()]);

    let var_outer = Term {
        id: ids.next_id(),
        ty: arr_ty.clone(),
        span: Span::dummy(),
        kind: TermKind::Var(outer_sym),
    };
    let capture_term_id = var_outer.id;

    let body_zero = Term {
        id: ids.next_id(),
        ty: i32_ty.clone(),
        span: Span::dummy(),
        kind: TermKind::IntLit("0".to_string()),
    };

    let lambda = Lambda {
        params: vec![(lambda_param_sym, i32_ty.clone())],
        body: Box::new(body_zero),
        ret_ty: i32_ty.clone(),
    };

    let zero_lit = Term {
        id: ids.next_id(),
        ty: i32_ty.clone(),
        span: Span::dummy(),
        kind: TermKind::IntLit("0".to_string()),
    };
    let one_lit = Term {
        id: ids.next_id(),
        ty: i32_ty.clone(),
        span: Span::dummy(),
        kind: TermKind::IntLit("1".to_string()),
    };
    let range_input = crate::tlc::ArrayExpr::Range {
        start: Box::new(zero_lit),
        len: Box::new(one_lit),
    };

    let map_term = Term {
        id: ids.next_id(),
        ty: arr_ty.clone(),
        span: Span::dummy(),
        kind: TermKind::Soac(crate::tlc::SoacOp::Map {
            lam: SoacBody {
                lam: lambda,
                captures: vec![(cap_sym, arr_ty.clone(), var_outer)],
            },
            inputs: vec![range_input],
            consumes_input: false,
        }),
    };

    let zero_body = Term {
        id: ids.next_id(),
        ty: i32_ty.clone(),
        span: Span::dummy(),
        kind: TermKind::IntLit("0".to_string()),
    };
    let inner_let = Term {
        id: ids.next_id(),
        ty: i32_ty.clone(),
        span: Span::dummy(),
        kind: TermKind::Let {
            name: f_sym,
            name_ty: arr_ty.clone(),
            rhs: Box::new(map_term),
            body: Box::new(zero_body),
        },
    };

    fn int_lit(ids: &mut TermIdSource, n: &str) -> Term {
        Term {
            id: ids.next_id(),
            ty: Type::Constructed(TypeName::Int(32), vec![]),
            span: Span::dummy(),
            kind: TermKind::IntLit(n.to_string()),
        }
    }
    let arr_lit = Term {
        id: ids.next_id(),
        ty: arr_ty.clone(),
        span: Span::dummy(),
        kind: TermKind::ArrayExpr(crate::tlc::ArrayExpr::Literal(vec![
            int_lit(&mut ids, "1"),
            int_lit(&mut ids, "2"),
            int_lit(&mut ids, "3"),
            int_lit(&mut ids, "4"),
        ])),
    };
    let outer_let = Term {
        id: ids.next_id(),
        ty: i32_ty.clone(),
        span: Span::dummy(),
        kind: TermKind::Let {
            name: outer_sym,
            name_ty: arr_ty.clone(),
            rhs: Box::new(arr_lit),
            body: Box::new(inner_let),
        },
    };

    let main_def = Def {
        name: main_sym,
        ty: i32_ty.clone(),
        body: outer_let,
        meta: DefMeta::Function,
        arity: 0,
    };

    let program = Program {
        defs: vec![main_def],
        uniforms: vec![],
        storage: vec![],
        symbols,
        def_syms: std::collections::HashMap::new(),
    };

    let model = super::analyze(&program);
    assert!(
        model.live_out.contains_key(&capture_term_id),
        "capture term's TermId should appear in live_out after analyze — \
         otherwise reads/kills inside the capture term aren't being walked",
    );
}
