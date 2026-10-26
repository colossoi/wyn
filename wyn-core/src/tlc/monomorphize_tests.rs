#![cfg(test)]

use super::{format_type_compact, SpecKey};
use crate::ast::TypeName;
use crate::tlc::{apply_type_substitution, TypeSubstitution};
use polytype::Type;

#[test]
fn intrinsic_specialization_uses_instantiated_helper_types() {
    use crate::tlc::{self, Term, TermKind, VarRef, WalkDecision};

    let source = r#"
        def minimum<T>(x: T, y: T) T = min(x, y)
        entry main(x: i32, y: i32, a: f32, b: f32) (i32, f32) =
            (minimum(x, y), minimum(a, b))
    "#;
    let typed = crate::compile_thru_frontend(source).expect("type check");
    let typed = crate::ast_type_holes::reject_type_holes(typed).expect("type holes");
    let program = tlc::lower_from_ast(typed).expect("lower to TLC");
    let program = tlc::validate_ownership(program).expect("ownership");
    let program = tlc::specialize_types(tlc::partial_eval(program)).expect("specialize types");

    let mut intrinsics = Vec::new();
    for def in &program.defs {
        def.body.walk(&mut |term: &Term| {
            if let TermKind::App { func, .. } = &term.kind {
                if let TermKind::Var(VarRef::Builtin { id, .. }) = func.kind {
                    intrinsics.push(crate::builtins::by_id(id).raw.surface_name);
                }
            }
            WalkDecision::Recurse
        });
    }
    intrinsics.sort();
    assert_eq!(intrinsics, ["f32.min", "i32.min"]);
}

#[test]
fn nested_array_helpers_are_exposed_before_buffer_specialization() {
    use crate::tlc::{self, DefMeta, SoacOp, Term, TermKind, WalkDecision};

    let mut source = String::from(
        r#"
        def total(xs: []i32) i32 = reduce(|a, b| a + b, 0, xs)
        def summarized(xs: []i32) i32 =
            let kept = filter(|x| x > 0, xs) in total(kept) + length(kept)
    "#,
    );
    // Long call chains must not depend on a second inlining pass after extraction.
    let mut callee = String::from("summarized");
    for depth in 0..12 {
        source.push_str(&format!("def helper{depth}(xs: []i32) i32 = {callee}(xs) + 1\n"));
        callee = format!("helper{depth}");
    }
    source.push_str(&format!(
        "entry main(xs: []i32, ys: []i32) (i32, i32) = ({callee}(xs), {callee}(ys))"
    ));
    let unified = crate::test_pipeline::compile_thru_unified_helpers(&source);
    assert!(unified.defs.iter().all(|def| matches!(def.meta, DefMeta::EntryPoint(_))));
    let extracted = tlc::extract_stages(unified).expect("extract stages");
    let pinned = tlc::pin_entry_buffers(extracted).expect("pin buffers");
    let specialized = tlc::monomorphize(pinned).expect("monomorphize");

    let mut filters = 0;
    let mut reductions = 0;
    specialized.defs[0].body.walk(&mut |term: &Term| {
        match term.kind {
            TermKind::Soac(SoacOp::Filter { .. }) => filters += 1,
            TermKind::Soac(SoacOp::Reduce { .. }) => reductions += 1,
            _ => {}
        }
        WalkDecision::Recurse
    });
    assert_eq!((filters, reductions), (2, 2));
}

#[test]
fn test_format_type_compact() {
    let f32_ty = Type::Constructed(TypeName::Float(32), vec![]);
    assert_eq!(format_type_compact(&f32_ty), "f32");

    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
    assert_eq!(format_type_compact(&i32_ty), "i32");

    let size_ty = Type::Constructed(TypeName::Size(4), vec![]);
    assert_eq!(format_type_compact(&size_ty), "n4");
}

#[test]
fn test_spec_key_empty() {
    let key = SpecKey::empty();
    assert!(!key.needs_specialization());
}

#[test]
fn test_spec_key_with_subst() {
    let mut subst = TypeSubstitution::new();
    subst.insert(0, Type::Constructed(TypeName::Float(32), vec![]));
    subst.insert(1, Type::Constructed(TypeName::Bool, vec![]));

    let key = SpecKey::new(&subst);
    assert!(key.needs_specialization());
    assert_eq!(key.type_subst.to_subst(), subst);
}

#[test]
fn shared_type_substitution_reaches_sum_variant_payloads() {
    let mut subst = TypeSubstitution::new();
    let concrete = Type::Constructed(TypeName::UInt(32), vec![]);
    subst.insert(0, concrete.clone());
    let sum = Type::Constructed(
        TypeName::Sum(vec![("some".to_string(), vec![Type::Variable(0)])]),
        vec![],
    );

    let rewritten = apply_type_substitution(&sum, &subst);
    let Type::Constructed(TypeName::Sum(variants), _) = rewritten else {
        panic!("expected sum type");
    };
    assert_eq!(variants[0].1, vec![concrete]);
}
