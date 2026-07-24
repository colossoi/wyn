//! Tests for AST→TLC sum-type lowering, focused on the `build_blank`
//! generator that fills dead variant slots at Constructor expressions.

use super::*;
use crate::Compiler;

fn unit_test_term(ids: &mut TermIdSource, kind: TermKind) -> Term {
    Term {
        id: ids.next_id(),
        ty: Type::Constructed(TypeName::Unit, vec![]),
        span: Span::dummy(),
        kind,
    }
}

#[test]
fn mapping_children_assigns_caller_provided_parent_id() {
    let mut ids = TermIdSource::new();
    let child = unit_test_term(&mut ids, TermKind::UnitLit);
    let original_child_id = child.id;
    let parent = unit_test_term(&mut ids, TermKind::Tuple(vec![child]));
    let original_parent_id = parent.id;

    let fresh_child_id = ids.next_id();
    let fresh_parent_id = ids.next_id();
    let rewritten = parent.map_children(fresh_parent_id, &mut |mut child| {
        child.id = fresh_child_id;
        child
    });
    let TermKind::Tuple(children) = rewritten.kind else {
        panic!("expected tuple");
    };

    assert_eq!(children[0].id, fresh_child_id);
    assert_eq!(rewritten.id, fresh_parent_id);
    assert_ne!(children[0].id, original_child_id);
    assert_ne!(rewritten.id, original_parent_id);
}

/// Compile a source string down to the raw TLC program produced by
/// `to_tlc`, skipping every post-TLC pass so the Constructor lowering
/// is observable verbatim.
fn compile_to_tlc_raw(source: &str) -> Program<stage::Transformed> {
    let (mut node_counter, mut module_manager) = crate::cached_compiler_init();
    let parsed = Compiler::parse(source, &mut node_counter).expect("parse");
    let type_checked = parsed
        .resolve(&mut module_manager)
        .expect("resolve")
        .fold_ast_constants()
        .type_check(&mut module_manager)
        .expect("type_check");
    type_checked.to_tlc(&module_manager, false)
}

fn find_def_body<'a>(program: &'a Program<stage::Transformed>, name: &str) -> &'a Term {
    let def = program
        .defs
        .iter()
        .find(|d| program.symbols.get(d.name).map(|s| s.as_str()) == Some(name))
        .unwrap_or_else(|| panic!("no def named {name}"));
    &def.body
}

/// Sum constructor expressions lower to a flat Tuple. Returns the
/// (tag-term, payload-slot-terms) destructured from the def body.
fn assert_constructor_tuple<'a>(body: &'a Term) -> &'a [Term] {
    match &body.kind {
        TermKind::Tuple(slots) => slots,
        other => panic!(
            "expected Constructor to lower to TermKind::Tuple, got {:?}",
            other
        ),
    }
}

#[test]
fn blank_for_array_payload() {
    // #small is active; #big's [4]i32 payload slot must be blank-filled
    // with an ArrayExpr::Literal of four zero i32 terms.
    let program = compile_to_tlc_raw(
        r#"
def make: #big([4]i32) | #small(i32) | #empty = #small(7)
"#,
    );
    let slots = assert_constructor_tuple(find_def_body(&program, "make"));
    // Slots: [tag=1, big_payload(blank [4]i32), small_payload(7)]
    assert_eq!(
        slots.len(),
        3,
        "expected tag + 2 payload slots, got {}",
        slots.len()
    );

    // tag = 1 (#small is source-index 1)
    assert!(
        matches!(&slots[0].kind, TermKind::IntLit(s) if s == "1"),
        "tag slot: {:?}",
        slots[0].kind
    );

    // big_payload (dead): ArrayExpr::Literal of 4 IntLit("0")
    let elems = match &slots[1].kind {
        TermKind::ArrayExpr(ArrayExpr::Literal(es)) => es,
        other => panic!(
            "expected ArrayExpr::Literal for blank array slot, got {:?}",
            other
        ),
    };
    assert_eq!(elems.len(), 4, "array blank should have 4 elements");
    for (i, e) in elems.iter().enumerate() {
        assert!(
            matches!(&e.kind, TermKind::IntLit(s) if s == "0"),
            "elem {i} should be IntLit(\"0\"), got {:?}",
            e.kind
        );
    }

    // small_payload (live): IntLit("7")
    assert!(
        matches!(&slots[2].kind, TermKind::IntLit(s) if s == "7"),
        "live slot: {:?}",
        slots[2].kind
    );
}

#[test]
fn blank_for_nested_sum_payload() {
    // Nested sums: lower_type rewrites the inner Sum to a Tuple before
    // build_blank sees it, so the outer blank for #outer's payload is
    // itself a Tuple matching the inner sum's lowered layout.
    let program = compile_to_tlc_raw(
        r#"
def make: #outer(#inner(i32) | #other) | #flat(i32) = #flat(1)
"#,
    );
    let slots = assert_constructor_tuple(find_def_body(&program, "make"));
    // Slots: [tag=1, outer_payload(blank inner-sum-as-tuple), flat_payload(1)]
    assert_eq!(slots.len(), 3);

    assert!(matches!(&slots[0].kind, TermKind::IntLit(s) if s == "1"));

    // Inner-sum slot is itself a Tuple. Inner sum has layout:
    //   slot 0: u32 tag
    //   slot 1: i32 (#inner's payload)
    // (#other has no payload, so no further slots.)
    // Blank for this lowered tuple type is (IntLit("0"), IntLit("0")).
    let inner_slots = match &slots[1].kind {
        TermKind::Tuple(s) => s,
        other => panic!("expected nested Tuple for blanked inner sum, got {:?}", other),
    };
    assert_eq!(
        inner_slots.len(),
        2,
        "inner sum lowered tuple should have 2 slots"
    );
    for (i, e) in inner_slots.iter().enumerate() {
        assert!(
            matches!(&e.kind, TermKind::IntLit(s) if s == "0"),
            "inner slot {i} should be IntLit(\"0\"), got {:?}",
            e.kind
        );
    }

    assert!(matches!(&slots[2].kind, TermKind::IntLit(s) if s == "1"));
}

#[test]
fn blank_for_vec_payload() {
    // vec3f32 payload: dead slot is VecLit of 3 zero floats.
    let program = compile_to_tlc_raw(
        r#"
def make: #v(vec3f32) | #s(i32) = #s(1)
"#,
    );
    let slots = assert_constructor_tuple(find_def_body(&program, "make"));
    // Slots: [tag=1, v_payload(blank vec3), s_payload(1)]
    assert_eq!(slots.len(), 3);

    assert!(matches!(&slots[0].kind, TermKind::IntLit(s) if s == "1"));

    let vec_elems = match &slots[1].kind {
        TermKind::VecLit(es) => es,
        other => panic!("expected VecLit for blank vec slot, got {:?}", other),
    };
    assert_eq!(vec_elems.len(), 3, "vec3 blank should have 3 elements");
    for (i, e) in vec_elems.iter().enumerate() {
        assert!(
            matches!(&e.kind, TermKind::FloatLit(v) if *v == 0.0),
            "vec elem {i} should be FloatLit(0.0), got {:?}",
            e.kind
        );
    }

    assert!(matches!(&slots[2].kind, TermKind::IntLit(s) if s == "1"));
}

#[test]
fn blank_recursive_array_of_arrays() {
    // [2][3]i32 payload: outer Literal of 2 inner Literals of 3 IntLit("0").
    let program = compile_to_tlc_raw(
        r#"
def make: #m([2][3]i32) | #s(i32) = #s(0)
"#,
    );
    let slots = assert_constructor_tuple(find_def_body(&program, "make"));
    assert_eq!(slots.len(), 3);

    assert!(matches!(&slots[0].kind, TermKind::IntLit(s) if s == "1"));

    let outer_elems = match &slots[1].kind {
        TermKind::ArrayExpr(ArrayExpr::Literal(es)) => es,
        other => panic!(
            "expected outer ArrayExpr::Literal for [2][3]i32 blank, got {:?}",
            other
        ),
    };
    assert_eq!(outer_elems.len(), 2, "outer dimension should have 2 elements");
    for (i, row) in outer_elems.iter().enumerate() {
        let inner = match &row.kind {
            TermKind::ArrayExpr(ArrayExpr::Literal(es)) => es,
            other => panic!("row {i} should be inner ArrayExpr::Literal, got {:?}", other),
        };
        assert_eq!(inner.len(), 3, "inner dimension should have 3 elements");
        for (j, e) in inner.iter().enumerate() {
            assert!(
                matches!(&e.kind, TermKind::IntLit(s) if s == "0"),
                "row {i} col {j} should be IntLit(\"0\"), got {:?}",
                e.kind
            );
        }
    }
}

// A top-level `def` whose name collides with a SOAC (`map`) type-checks as the
// user signature, and TLC lowers the call to match: SOAC identity is a
// structural tag set by the frontend resolver (which honours shadowing — local
// and top-level), so `map(y)` is an ordinary application of the user def, not a
// `Soac(Map)` node. Before the resolver-driven tag, AST→TLC re-derived SOAC-ness
// by string match (excluding only *locally* bound names) and the 1-arg call was
// routed to `transform_soac_map`, which `assert!(args.len() >= 2)` and panicked.
#[test]
fn user_def_shadowing_soac_map_is_a_normal_call() {
    let program = compile_to_tlc_raw(
        r#"
def map(x: i32) i32 = x
def use_it(y: i32) i32 = map(y)
"#,
    );
    let body = find_def_body(&program, "use_it");
    let call = match &body.kind {
        TermKind::Lambda(lam) => &lam.body,
        other => panic!("expected `use_it` to be a Lambda, got {:?}", other),
    };
    match &call.kind {
        TermKind::App { func, .. } => assert!(
            matches!(&func.kind, TermKind::Var(VarRef::Symbol(_))),
            "callee should be the user `map` symbol, not a SOAC/builtin; got {:?}",
            func.kind
        ),
        other => panic!(
            "user `def map` should lower to a normal application, not a SOAC; got {:?}",
            other
        ),
    }
}

/// Walk a let-chain body to the first `scatter` SOAC.
fn find_scatter(mut t: &Term) -> &SoacOp {
    loop {
        match &t.kind {
            TermKind::Soac(op @ SoacOp::Scatter { .. }) => return op,
            TermKind::Lambda(lam) => t = &lam.body,
            TermKind::Let { rhs, body, .. } => {
                if let TermKind::Soac(op @ SoacOp::Scatter { .. }) = &rhs.kind {
                    return op;
                }
                t = body;
            }
            other => panic!("no scatter SOAC in body; got {other:?}"),
        }
    }
}

// `scatter(dest, is, vs)` lowers to a `Scatter` carrying the identity envelope
// `λ(i, v) → (i, v)` over `inputs = [is, vs]`. Map→scatter fusion later composes
// producers into this envelope, so its exact shape is load-bearing.
#[test]
fn plain_scatter_carries_identity_envelope() {
    let program = compile_to_tlc_raw(
        r#"
def N:i32 = 5
#[compute]
entry rasterize(#[storage(set=2, binding=0, access=read)] positions: []vec4f32,
                #[storage(set=2, binding=1, access=write)] fb: []vec4f32) () =
  let pts  = positions[0..N] in
  let idxs = map(|p:vec4f32| i32.f32(p.y) * 512 + i32.f32(p.x), pts) in
  let vals = map(|p:vec4f32| @[1.0, 1.0, 1.0, 1.0], pts) in
  let _ = scatter(fb, idxs, vals) in ()
"#,
    );
    let body = find_def_body(&program, "rasterize");
    let SoacOp::Scatter { lam, inputs, .. } = find_scatter(body) else {
        unreachable!("find_scatter returns a Scatter")
    };

    // Two raw inputs: the index array and the value array.
    assert_eq!(inputs.len(), 2, "plain scatter keeps [indices, values]");

    // Identity envelope: two params, returning the 2-tuple of those same params.
    assert_eq!(lam.lam.params.len(), 2, "envelope takes (i, v)");
    assert_eq!(
        lam.data,
        (),
        "pre-defunctionalization envelope has no capture payload"
    );
    assert!(
        matches!(&lam.lam.ret_ty, Type::Constructed(TypeName::Tuple(2), _)),
        "envelope returns a 2-tuple, got {:?}",
        lam.lam.ret_ty
    );
    let i_sym = lam.lam.params[0].0;
    let v_sym = lam.lam.params[1].0;
    match &lam.lam.body.kind {
        TermKind::Tuple(parts) => {
            assert_eq!(parts.len(), 2, "body is the pair (i, v)");
            assert!(
                matches!(&parts[0].kind, TermKind::Var(VarRef::Symbol(s)) if *s == i_sym),
                "tuple slot 0 is the index param"
            );
            assert!(
                matches!(&parts[1].kind, TermKind::Var(VarRef::Symbol(s)) if *s == v_sym),
                "tuple slot 1 is the value param"
            );
        }
        other => panic!("identity envelope body must be a 2-tuple, got {other:?}"),
    }
}
