use super::*;
use crate::tlc::{clone_term_with_fresh_ids, Lambda, LoopKind};

fn term(ids: &mut TermIdSource, kind: TermKind) -> Term {
    Term::fresh(
        ids,
        Type::Constructed(TypeName::Int(32), vec![]),
        Span::generated(),
        kind,
    )
}

fn int(ids: &mut TermIdSource, n: i32) -> Term {
    term(ids, TermKind::IntLit(n.to_string()))
}

fn var(ids: &mut TermIdSource, name: SymbolId) -> Term {
    term(ids, TermKind::Var(VarRef::Symbol(name)))
}

fn raw_let(ids: &mut TermIdSource, name: SymbolId, rhs: Term, body: Term) -> Term {
    Term::fresh(
        ids,
        body.ty.clone(),
        Span::generated(),
        TermKind::Let {
            name,
            name_ty: rhs.ty.clone(),
            rhs: Box::new(rhs),
            body: Box::new(body),
        },
    )
}

fn prefix(term: &Term) -> (Vec<(SymbolId, &Term)>, &Term) {
    let mut bindings = vec![];
    let mut tail = term;
    while let TermKind::Let { name, rhs, body, .. } = &tail.kind {
        assert!(!matches!(rhs.kind, TermKind::Let { .. }), "nested binding RHS");
        bindings.push((*name, &**rhs));
        tail = body;
    }
    (bindings, tail)
}

#[test]
fn nested_rhs_bindings_preserve_dependency_order_and_value_identity() {
    let mut ids = TermIdSource::new();
    let mut symbols = SymbolTable::new();
    let names: Vec<_> =
        ["a", "b", "result", "after"].into_iter().map(|s| symbols.alloc(s.into())).collect();
    let a_value = int(&mut ids, 7);
    let a_id = a_value.id;
    let a_ref = var(&mut ids, names[0]);
    let b_rhs = raw_let(&mut ids, names[0], a_value, a_ref);
    let b_ref = var(&mut ids, names[1]);
    let rhs = raw_let(&mut ids, names[1], b_rhs, b_ref);
    let mut bindings = Bindings::new();
    bindings.push(LetBinding {
        name: names[2],
        name_ty: rhs.ty.clone(),
        rhs,
        span: Span::generated(),
    });
    let after = int(&mut ids, 9);
    bindings.push(LetBinding {
        name: names[3],
        name_ty: after.ty.clone(),
        rhs: after,
        span: Span::generated(),
    });
    let result = var(&mut ids, names[2]);
    let result_id = result.id;
    let built = bindings.finish(result, &mut ids);
    let (prefix, tail) = prefix(&built);
    assert_eq!(prefix.iter().map(|(name, _)| *name).collect::<Vec<_>>(), names);
    assert_eq!(prefix[0].1.id, a_id);
    assert_eq!(tail.id, result_id);
    assert!(matches!(prefix[1].1.kind, TermKind::Var(VarRef::Symbol(s)) if s == names[0]));
    assert!(matches!(prefix[2].1.kind, TermKind::Var(VarRef::Symbol(s)) if s == names[1]));
}

#[test]
fn naming_a_value_twice_retains_one_evaluation() {
    let mut ids = TermIdSource::new();
    let mut symbols = SymbolTable::new();
    let callee = symbols.alloc("opaque_call".into());
    let func = var(&mut ids, callee);
    let call = term(
        &mut ids,
        TermKind::App {
            func: Box::new(func),
            args: vec![],
        },
    );
    let call_id = call.id;
    let mut bindings = Bindings::new();
    let first = bindings.name(call, "value", &mut symbols, &mut ids);
    let second = clone_term_with_fresh_ids(&first, &mut ids);
    let second_id = second.id;
    let second = bindings.name(second, "alias", &mut symbols, &mut ids);
    assert_eq!(second.id, second_id);
    let tuple = term(&mut ids, TermKind::Tuple(vec![first, second]));
    let built = bindings.finish(tuple, &mut ids);
    let (prefix, tail) = prefix(&built);
    assert_eq!(prefix.len(), 1);
    assert_eq!(prefix[0].1.id, call_id);
    let TermKind::Tuple(values) = &tail.kind else {
        panic!("tuple result")
    };
    for value in values {
        assert!(matches!(value.kind, TermKind::Var(VarRef::Symbol(s)) if s == prefix[0].0));
    }
}

#[test]
fn bindings_stay_inside_branches_callbacks_and_loops() {
    let mut ids = TermIdSource::new();
    let mut symbols = SymbolTable::new();
    let local = symbols.alloc("local".into());
    let param = symbols.alloc("param".into());
    let loop_var = symbols.alloc("acc".into());
    let index = symbols.alloc("i".into());
    let value = int(&mut ids, 1);
    let result = var(&mut ids, local);
    let inner = raw_let(&mut ids, local, value, result);
    let ty = inner.ty.clone();
    let cond = Term::fresh(
        &mut ids,
        Type::Constructed(TypeName::Bool, vec![]),
        Span::generated(),
        TermKind::BoolLit(false),
    );
    let fallback = int(&mut ids, 0);
    let then_branch = clone_term_with_fresh_ids(&inner, &mut ids);
    let branch = term(
        &mut ids,
        TermKind::If {
            cond: Box::new(cond),
            then_branch: Box::new(then_branch),
            else_branch: Box::new(fallback),
        },
    );
    let callback_body = clone_term_with_fresh_ids(&inner, &mut ids);
    let callback = Term::fresh(
        &mut ids,
        Type::Constructed(TypeName::Arrow, vec![ty.clone(), ty.clone()]),
        Span::generated(),
        TermKind::Lambda(Lambda {
            params: vec![(param, ty.clone())],
            body: Box::new(callback_body),
            ret_ty: ty.clone(),
        }),
    );
    let init = int(&mut ids, 0);
    let bound = int(&mut ids, 3);
    let loop_term = term(
        &mut ids,
        TermKind::Loop {
            loop_var,
            loop_var_ty: ty.clone(),
            init: Box::new(init),
            init_bindings: vec![],
            kind: LoopKind::ForRange {
                var: index,
                var_ty: ty,
                bound: Box::new(bound),
            },
            body: Box::new(inner),
        },
    );
    for expression in [branch, callback, loop_term] {
        let original = format!("{expression:?}");
        let mut bindings = Bindings::new();
        let result = bindings.name(expression, "outer", &mut symbols, &mut ids);
        let built = bindings.finish(result, &mut ids);
        let (prefix, _) = prefix(&built);
        assert_eq!(prefix.len(), 1, "an inner binding escaped its evaluation scope");
        assert_eq!(format!("{:?}", prefix[0].1), original);
    }
}

#[test]
fn inline_arguments_are_flat_and_keep_left_to_right_order() {
    let mut ids = TermIdSource::new();
    let mut symbols = SymbolTable::new();
    let names: Vec<_> = ["left_local", "left_param", "right_local", "right_param"]
        .into_iter()
        .map(|s| symbols.alloc(s.into()))
        .collect();
    let mut args = vec![];
    let mut params = vec![];
    for pair in names.chunks_exact(2) {
        let value = int(&mut ids, 7);
        let local_ref = var(&mut ids, pair[0]);
        args.push(raw_let(&mut ids, pair[0], value, local_ref));
        params.push((pair[1], Type::Constructed(TypeName::Int(32), vec![])));
    }
    let result = var(&mut ids, names[1]);
    let inlined = crate::tlc::inline::build_inline_lets(&params, args, result, Span::generated(), &mut ids);
    let (prefix, _) = prefix(&inlined);
    assert_eq!(prefix.iter().map(|(name, _)| *name).collect::<Vec<_>>(), names);
    let original_id = inlined.id;
    let (unchanged, changed) = flatten_nested_let(inlined, &mut ids);
    assert!(!changed, "inlining must emit flat bindings without an ANF repair");
    assert_eq!(unchanged.id, original_id);
}

#[test]
fn soac_inputs_retain_atoms_and_name_computed_values() {
    let mut ids = TermIdSource::new();
    let mut symbols = SymbolTable::new();
    let mut bindings = Bindings::new();
    let element = int(&mut ids, 1);
    let array = ArrayExpr::Literal(vec![element]);
    let ty = array.array_type();
    let literal = Term::fresh(
        &mut ids,
        ty.clone(),
        Span::generated(),
        TermKind::ArrayExpr(array),
    );
    assert!(matches!(
        bindings.input(literal, &mut symbols, &mut ids),
        ArrayExpr::Literal(_)
    ));
    let callee = symbols.alloc("array_helper".into());
    let func = var(&mut ids, callee);
    let call = Term::fresh(
        &mut ids,
        ty,
        Span::generated(),
        TermKind::App {
            func: Box::new(func),
            args: vec![],
        },
    );
    let call_id = call.id;
    let input = bindings.input(call, &mut symbols, &mut ids);
    let named = input.as_named_ref().expect("computed input must be named");
    let result = int(&mut ids, 0);
    let built = bindings.finish(result, &mut ids);
    let (prefix, _) = prefix(&built);
    assert_eq!(prefix.len(), 1);
    assert_eq!(prefix[0].0, named);
    assert_eq!(prefix[0].1.id, call_id);
}
