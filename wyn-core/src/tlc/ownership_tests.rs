use super::{Origin, OwnershipModel, analyze, build};
use crate::Compiler;
use crate::tlc::{Program, Term, TermKind};

#[test]
fn origin_mutability() {
    assert!(Origin::Fresh.is_mutable());
    assert!(Origin::UniqueParam.is_mutable());
    assert!(Origin::Entry.is_mutable());
    assert!(!Origin::NonUniqueParam.is_mutable());
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
        captures: vec![],
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
fn soac_element_param_gets_fresh_owner_when_non_copy() {
    // Map over a 2D array — the lambda's element param has type [4]i32
    // (non-copy), so it should be tracked as a fresh per-iteration owner.
    let program = compile_to_tlc(
        r#"
def f(rows: *[3][4]i32) [3][4]i32 = map(|row| row, rows)
"#,
    );
    let model = build(&program);

    // The model should contain at least one Origin::Fresh that's
    // distinct from the rows' UniqueParam. (If the lambda body had no
    // tracked references, the per-iteration owner is still allocated.)
    let fresh_count = model.origins.values().filter(|o| **o == super::Origin::Fresh).count();
    let unique_count = model.origins.values().filter(|o| **o == super::Origin::UniqueParam).count();
    assert!(unique_count >= 1, "expected at least one UniqueParam owner");
    assert!(
        fresh_count >= 1,
        "expected at least one Fresh owner for the SOAC element param"
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

// Inner shadowed `x` should be its own owner; the outer `x` is
// untouched by `consume(inner_x)`. Currently a false positive: the
// AST→TLC transformer (`tlc::Transformer::define`) overwrites
// `scope[name]` without push/pop, so the trailing `x[0]` resolves
// to the inner `x`'s SymbolId and the consume looks like it kills
// the still-live outer slot. Fix is in the transformer's scope
// management, not in this pass — pinning the case here so the bug
// has a home.
#[test]
#[ignore = "TLC transform doesn't push/pop scope for nested let — see comment"]
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
