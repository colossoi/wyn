//! Literal expansion and the disjoint inclusive-range spelling.
use crate::{
    compile_thru_frontend, compile_thru_spirv, compile_thru_ssa, compile_thru_tlc, lower_ssa_to_wgsl,
};

#[test]
fn literal_expansion_accepts_all_vector_element_types_and_widths() {
    for scalar in [
        "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "f16", "f32", "f64", "bool",
    ] {
        for width in 2..=4 {
            let source = format!("def vector(v: vec{width}{scalar}) vec{width}{scalar} = @[v...]\ndef array(v: vec{width}{scalar}) [{width}]{scalar} = [v...,]");
            compile_thru_frontend(&source).unwrap_or_else(|e| panic!("{source}: {e:?}"));
        }
    }
    compile_thru_frontend("def f(v: vec3f32) vec4f32 = @[v..., 1.0]\ndef g(v: vec3f32) vec3f32 = @[v.xy..., 1.0]\ndef h(v: vec2f32) vec4f32 = @[v.yx..., v.xx...,]").unwrap();
}

#[test]
fn literal_expansion_rejects_nonvectors_mixed_types_and_excess_width() {
    for expression in [
        "@[1...]",
        "@[@[1, 2].x...]",
        "@[[1, 2]...]",
        "@[@[[1, 2], [3, 4]]...]",
        "@[@[1, 2]..., 3.0]",
        "[@[1, 2]..., true]",
        "@[@[1, 2, 3, 4]..., 5]",
    ] {
        let err = compile_thru_frontend(&format!("let bad = {expression}")).expect_err(expression);
        assert!(
            matches!(err, crate::error::CompilerError::TypeError(_, _)),
            "{expression}: {err:?}"
        );
    }
}

#[test]
fn literal_expansion_is_only_valid_in_literals() {
    for source in [
        "let x = @[1, 2]...",
        "let x = (@[1, 2]..., 3)",
        "def f(x: vec2i32) vec2i32 = x\nlet x = f(@[1, 2]...)",
        "let x = {a = @[1, 2]...}",
        "let x = 0...3",
        "let x = [0...3]",
        "let x = @[@[1, 2]... + 1]",
    ] {
        assert!(compile_thru_frontend(source).is_err(), "accepted {source}");
    }
}

#[test]
fn inclusive_range_preserves_exclusive_ranges_and_slices() {
    compile_thru_frontend("let a: [4]i32 = 0..=3\nlet b: [3]i32 = 0..3\nlet c: [3]i32 = 0..<3\nlet d: [3]i32 = 0..2..=4\ndef slice(a: [4]i32) [3]i32 = a[0..3]").unwrap();
}

#[test]
fn literal_expansion_lowers_to_both_shader_backends() {
    for source in [
        "entry e(v: vec3f32) vec4f32 = @[v..., 1.0]",
        "entry e(v: vec3f32) vec4f32 = @[v.yx..., v.zz...]",
        "entry e(v: vec3f32) f32 = let a: [5]f32 = [0.0, v..., 1.0] in a[2]",
        "entry e(v: vec2f32) mat2x2f32 = @[[v.yx...], [v...]]",
        "entry e() i32 = let r = 0..2..=4 in r[2]",
    ] {
        compile_thru_spirv(source).unwrap_or_else(|e| panic!("SPIR-V: {source}: {e:?}"));
        lower_ssa_to_wgsl(compile_thru_ssa(source).unwrap())
            .unwrap_or_else(|e| panic!("WGSL: {source}: {e:?}"));
    }
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
