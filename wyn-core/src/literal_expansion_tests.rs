//! Small source rejection cases and literal-expansion invariants.
//! Broader acceptance and backend scenarios live in tests/compiler/cases/.
use crate::tlc::{ArrayExpr, TermKind};
use crate::{compile_thru_frontend, compile_thru_tlc};

fn assert_rejected_sources(sources: &[&str], diagnostic: &str) {
    for source in sources {
        let error = compile_thru_frontend(source).expect_err(source).to_string();
        assert!(
            error.contains(diagnostic),
            "{source}: expected {diagnostic:?}, got {error}"
        );
    }
}

#[test]
fn literal_expansion_rejects_nonvectors_mixed_types_and_excess_width() {
    assert_rejected_sources(
        &[
            "let bad = @[1...]",
            "let bad = @[@[1, 2].x...]",
            "let bad = @[[1, 2]...]",
            "let bad = @[@[[1, 2], [3, 4]]...]",
            "let bad = @[@[1, 2]..., 3.0]",
            "let bad = [@[1, 2]..., true]",
            "let bad = @[@[1, 2, 3, 4]..., 5]",
        ],
        "Type error:",
    );
}

#[test]
fn literal_expansion_is_only_valid_in_literals() {
    assert_rejected_sources(
        &[
            "let x = @[1, 2]...",
            "let x = (@[1, 2]..., 3)",
            "def f(x: vec2i32) vec2i32 = x\nlet x = f(@[1, 2]...)",
            "let x = {a = @[1, 2]...}",
            "let x = 0...3",
            "let x = [0...3]",
            "let x = @[@[1, 2]... + 1]",
        ],
        "Parse error:",
    );
}

#[test]
fn literal_expansion_evaluates_its_producer_once() {
    use crate::tlc::{Payload, SoacOp, Term, TermKind};
    fn count_reduces<C: Payload, S: Payload>(t: &Term<C, S>) -> usize {
        let mut count = usize::from(matches!(&t.kind, TermKind::Soac(SoacOp::Reduce { .. })));
        t.for_each_child(&mut |child| count += count_reduces(child));
        count
    }
    for literal in [
        "@[reduce(|a: vec3f32, b: vec3f32| a + b, @[0.0, 0.0, 0.0], xs)..., 1.0]",
        "@[reduce(|a: vec3f32, b: vec3f32| a + b, @[0.0, 0.0, 0.0], xs).yx..., 1.0, 2.0]",
    ] {
        let source = format!("entry e(xs: [8]vec3f32) vec4f32 = {literal}");
        let tlc = compile_thru_tlc(&source).unwrap();
        let total: usize = tlc.defs.iter().map(|d| count_reduces(&d.body)).sum();
        assert_eq!(total, 1, "duplicated producer: {source}");
    }
}

#[test]
fn literal_expansion_preserves_component_order() {
    use crate::tlc::{ArrayExpr, Payload, Term, TermKind, VarRef};
    use std::collections::HashMap;

    // Interpret the constant aggregate subset so this checks values without
    // depending on whether the optimizer chooses to inline a let binding.
    fn values<C: Payload, S: Payload>(
        term: &Term<C, S>,
        env: &mut HashMap<crate::SymbolId, Vec<i32>>,
    ) -> Vec<i32> {
        match &term.kind {
            TermKind::IntLit(value) => vec![value.parse().unwrap()],
            TermKind::VecLit(parts) | TermKind::ArrayExpr(ArrayExpr::Literal(parts)) => {
                parts.iter().flat_map(|part| values(part, env)).collect()
            }
            TermKind::Var(VarRef::Symbol(name)) => env[name].clone(),
            TermKind::Let { name, rhs, body, .. } => {
                let value = values(rhs, env);
                let old = env.insert(*name, value);
                let result = values(body, env);
                if let Some(old) = old {
                    env.insert(*name, old);
                } else {
                    env.remove(name);
                }
                result
            }
            TermKind::TupleProj { tuple, idx } => vec![values(tuple, env)[*idx]],
            TermKind::Index { array, index } => {
                let index = values(index, env)[0] as usize;
                vec![values(array, env)[index]]
            }
            _ => panic!("unexpected constant expression: {term:?}"),
        }
    }
    for (literal, expected) in [
        ("@[v..., 9]", vec![1, 2, 3, 9]),
        ("@[v.yx..., v.zz...]", vec![2, 1, 3, 3]),
        ("@[0, v.zyx...]", vec![0, 3, 2, 1]),
        (
            "let a = [0, v.yx..., 9] in @[a[0], a[1], a[2], a[3]]",
            vec![0, 2, 1, 9],
        ),
    ] {
        let source = format!("entry e() vec4i32 = let v = @[1, 2, 3] in {literal}");
        let tlc = compile_thru_tlc(&source).unwrap();
        assert_eq!(tlc.defs.len(), 1);
        let actual = values(&tlc.defs[0].body, &mut HashMap::new());
        assert_eq!(actual, expected, "{literal}");
    }
}

#[test]
fn inclusive_stepped_ranges_count_the_endpoint_after_dividing() {
    for (range, count) in [("0..2..=4", 3), ("0..2..=5", 3), ("1..3..=10", 4), ("2..=2", 1)] {
        let source = format!("entry main() [{count}]i32 = {range}");
        let tlc = compile_thru_tlc(&source).unwrap();
        let TermKind::ArrayExpr(ArrayExpr::Range { len, .. }) = &tlc.defs[0].body.kind else {
            panic!("expected a virtual range: {source}");
        };
        let TermKind::IntLit(length) = &len.kind else {
            panic!("range length was not folded: {source}");
        };
        assert_eq!(length.parse::<usize>().unwrap(), count, "{source}");
    }
}
