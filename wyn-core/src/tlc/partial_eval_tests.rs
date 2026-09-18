//! Tests for TLC partial evaluation.

use super::{partial_eval, VarRef};
use crate::ast::{BinaryOp, Span, TypeName};
use crate::builtins;
use crate::op::BinaryOperator;
use crate::tlc;
use crate::tlc::context::TransformedGlobal;
use crate::tlc::data::{Empty, PolymorphicDefinition};
use crate::tlc::ownership::OwnershipValidated;
use crate::tlc::test_support::TestBuilder;
use crate::tlc::{Def, DefMeta, Lambda, LoopKind, Program, Term, TermIdSource, TermKind};
use crate::types;
use crate::{SymbolId, SymbolTable};
use polytype::Type;

fn input_ae(boxed: Box<Term<Empty, Empty>>) -> tlc::ArrayExpr<Empty, Empty> {
    use crate::tlc::{ArrayExpr, TermKind};
    let t = *boxed;
    match t.kind {
        TermKind::Var(vr) => ArrayExpr::Var(vr, t.ty),
        TermKind::ArrayExpr(ae) => ae,
        other => panic!("test SOAC input must be a variable or array expr, got {other:?}"),
    }
}
fn make_span() -> Span {
    Span::generated()
}

fn make_program(
    name_sym: SymbolId,
    body: Term<Empty, Empty>,
    (symbols, term_ids): (SymbolTable, TermIdSource),
) -> OwnershipValidated {
    Program::from_parts(
        vec![Def {
            data: PolymorphicDefinition { scheme: None },
            name: name_sym,
            package: None,
            ty: body.ty.clone(),
            body,
            meta: DefMeta::Function,
            arity: 0,
            param_diets: vec![],
            return_diet: types::Diet::observing(),
        }],
        symbols,
        term_ids,
        transformed_global(),
    )
}

fn transformed_global() -> TransformedGlobal {
    TransformedGlobal {
        known_defs: Default::default(),
        auto_storage_binding_ids: Default::default(),
    }
}

fn make_int(ids: &mut TermIdSource, n: i64) -> Term<Empty, Empty> {
    Term {
        id: ids.next_id(),
        ty: Type::Constructed(TypeName::Int(32), vec![]),
        span: make_span(),
        kind: TermKind::IntLit(n.to_string()),
    }
}

fn make_bool(ids: &mut TermIdSource, b: bool) -> Term<Empty, Empty> {
    Term {
        id: ids.next_id(),
        ty: Type::Constructed(TypeName::Bool, vec![]),
        span: make_span(),
        kind: TermKind::BoolLit(b),
    }
}

fn float_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Float(32), vec![])
}

fn make_float(ids: &mut TermIdSource, value: f32) -> Term<Empty, Empty> {
    Term {
        id: ids.next_id(),
        ty: float_ty(),
        span: make_span(),
        kind: TermKind::FloatLit(value),
    }
}

fn make_float_builtin_call(ids: &mut TermIdSource, name: &str, args: &[f32]) -> Term<Empty, Empty> {
    let builtin = builtins::catalog()
        .lookup_by_surface_name(name)
        .unwrap_or_else(|| panic!("missing test builtin `{name}`"));
    let func_ty = args.iter().fold(float_ty(), |result, _| arrow_ty(float_ty(), result));
    Term {
        id: ids.next_id(),
        ty: float_ty(),
        span: make_span(),
        kind: TermKind::App {
            func: Box::new(Term {
                id: ids.next_id(),
                ty: func_ty,
                span: make_span(),
                kind: TermKind::Var(VarRef::Builtin {
                    id: builtin.id,
                    overload_idx: 0,
                }),
            }),
            args: args.iter().map(|value| make_float(ids, *value)).collect(),
        },
    }
}

fn eval_float_builtin(name: &str, args: &[f32]) -> Term<Empty, Empty> {
    let mut b = TestBuilder::new();
    let test_sym = b.sym("test");
    let body = make_float_builtin_call(&mut b.ids, name, args);
    let mut program = partial_eval(make_program(test_sym, body, b.finish()));
    program.defs.pop().expect("test definition disappeared").body
}

fn make_binop(
    ids: &mut TermIdSource,
    op: BinaryOperator,
    lhs: Term<Empty, Empty>,
    rhs: Term<Empty, Empty>,
) -> Term<Empty, Empty> {
    let result_ty = lhs.ty.clone();
    let partial_ty = Type::Constructed(TypeName::Arrow, vec![result_ty.clone(), result_ty.clone()]);
    let binop_ty = Type::Constructed(TypeName::Arrow, vec![result_ty.clone(), partial_ty.clone()]);

    Term {
        id: ids.next_id(),
        ty: result_ty,
        span: make_span(),
        kind: TermKind::App {
            func: Box::new(Term {
                id: ids.next_id(),
                ty: binop_ty,
                span: make_span(),
                kind: TermKind::BinOp(BinaryOp { op }),
            }),
            args: vec![lhs, rhs],
        },
    }
}

#[test]
fn test_constant_folding_add() {
    let mut b = TestBuilder::new();
    let test_sym = b.sym("test");

    let lhs = make_int(&mut b.ids, 2);
    let rhs = make_int(&mut b.ids, 3);
    let term = make_binop(&mut b.ids, BinaryOperator::Add, lhs, rhs);

    let program = make_program(test_sym, term, b.finish());

    let result = partial_eval(program);
    assert_eq!(result.defs.len(), 1);

    match &result.defs[0].body.kind {
        TermKind::IntLit(s) => assert_eq!(s, "5"),
        other => panic!("Expected IntLit(5), got {:?}", other),
    }
}

#[test]
fn test_constant_folding_mul() {
    let mut b = TestBuilder::new();
    let test_sym = b.sym("test");

    let lhs = make_int(&mut b.ids, 4);
    let rhs = make_int(&mut b.ids, 7);
    let term = make_binop(&mut b.ids, BinaryOperator::Multiply, lhs, rhs);

    let program = make_program(test_sym, term, b.finish());

    let result = partial_eval(program);

    match &result.defs[0].body.kind {
        TermKind::IntLit(s) => assert_eq!(s, "28"),
        other => panic!("Expected IntLit(28), got {:?}", other),
    }
}

fn float_math_cases() -> Vec<(&'static str, Vec<f32>, f32)> {
    vec![
        ("f32.sin", vec![0.0], 0.0),
        ("f32.cos", vec![0.0], 1.0),
        ("f32.tan", vec![0.0], 0.0),
        ("f32.asin", vec![0.0], 0.0),
        ("f32.acos", vec![1.0], 0.0),
        ("f32.atan", vec![0.0], 0.0),
        ("f32.sinh", vec![0.0], 0.0),
        ("f32.cosh", vec![0.0], 1.0),
        ("f32.tanh", vec![0.0], 0.0),
        ("f32.asinh", vec![0.0], 0.0),
        ("f32.acosh", vec![1.0], 0.0),
        ("f32.atanh", vec![0.0], 0.0),
        ("f32.atan2", vec![1.0, 0.0], std::f32::consts::FRAC_PI_2),
        ("f32.pow", vec![2.0, 3.0], 8.0),
        ("f32.exp", vec![0.0], 1.0),
        ("f32.log", vec![1.0], 0.0),
        ("f32.exp2", vec![3.0], 8.0),
        ("f32.log2", vec![8.0], 3.0),
        ("f32.sqrt", vec![9.0], 3.0),
        ("f32.rsqrt", vec![4.0], 0.5),
        ("f32.radians", vec![180.0], std::f32::consts::PI),
        ("f32.degrees", vec![std::f32::consts::PI], 180.0),
        ("f32.floor", vec![1.75], 1.0),
        ("f32.ceil", vec![1.25], 2.0),
        ("f32.round", vec![2.5], 2.0),
        ("f32.round", vec![3.5], 4.0),
        ("f32.round", vec![-2.5], -2.0),
        ("f32.trunc", vec![-1.75], -1.0),
        ("f32.fract", vec![-1.75], 0.25),
        ("f32.abs", vec![-2.0], 2.0),
        ("f32.sign", vec![-2.0], -1.0),
        ("f32.sign", vec![0.0], 0.0),
        ("f32.sign", vec![2.0], 1.0),
        ("f32.min", vec![3.0, -2.0], -2.0),
        ("f32.max", vec![3.0, -2.0], 3.0),
        ("f32.clamp", vec![-2.0, -1.0, 1.0], -1.0),
        ("f32.clamp", vec![2.0, -1.0, 1.0], 1.0),
        ("f32.lerp", vec![2.0, 6.0, 0.25], 3.0),
        ("f32.fma", vec![2.0, 3.0, 4.0], 10.0),
        ("f32.mod", vec![-3.5, 2.0], 0.5),
        ("f32.mod", vec![3.5, -2.0], -0.5),
        ("mix", vec![2.0, 6.0, 0.25], 3.0),
        ("smoothstep", vec![0.0, 1.0, 0.25], 0.15625),
        ("smoothstep", vec![0.0, 1.0, -2.0], 0.0),
        ("smoothstep", vec![0.0, 1.0, 2.0], 1.0),
        ("step", vec![0.5, 0.25], 0.0),
        ("step", vec![0.5, 0.5], 1.0),
    ]
}

#[test]
fn scalar_glsl_math_builtins_constant_fold() {
    for (name, args, expected) in float_math_cases() {
        match eval_float_builtin(name, &args).kind {
            TermKind::FloatLit(actual) => {
                let tolerance = 1.0e-5 * expected.abs().max(1.0);
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "{name}{args:?}: expected {expected}, got {actual}"
                );
            }
            other => panic!("{name}{args:?}: expected folded FloatLit, got {other:?}"),
        }
    }
}

#[test]
fn scalar_glsl_math_keeps_poison_and_non_finite_results_residual() {
    for (name, args) in [
        ("f32.atan2", vec![0.0, 0.0]),
        ("f32.pow", vec![-2.0, 2.0]),
        ("f32.log", vec![0.0]),
        ("f32.sqrt", vec![-1.0]),
        ("f32.exp", vec![100.0]),
        ("f32.asin", vec![2.0]),
        ("f32.acosh", vec![0.0]),
        ("f32.atanh", vec![1.0]),
        ("f32.rsqrt", vec![0.0]),
        ("f32.pow", vec![0.0, 0.0]),
        ("f32.clamp", vec![0.0, 2.0, 1.0]),
        ("smoothstep", vec![1.0, 1.0, 0.5]),
        ("smoothstep", vec![2.0, 1.0, 0.5]),
        ("f32.mod", vec![2.0, 0.0]),
        ("f32.min", vec![f32::NAN, 1.0]),
        ("f32.max", vec![1.0, f32::INFINITY]),
    ] {
        assert!(
            matches!(eval_float_builtin(name, &args).kind, TermKind::App { .. }),
            "{name}{args:?} must remain a runtime call"
        );
    }
}

#[test]
fn scalar_glsl_math_folds_inside_lambda_body() {
    let mut b = TestBuilder::new();
    let test_sym = b.sym("test");
    let ignored_sym = b.sym("ignored");
    let body = make_float_builtin_call(&mut b.ids, "f32.cos", &[0.0]);
    let lambda_ty = arrow_ty(float_ty(), float_ty());
    let lambda = Term {
        id: b.next_id(),
        ty: lambda_ty.clone(),
        span: b.span(),
        kind: TermKind::Lambda(Lambda {
            params: vec![(ignored_sym, float_ty())],
            body: Box::new(body),
            ret_ty: float_ty(),
        }),
    };
    let (symbols, term_ids) = b.finish();
    let program = Program::from_parts(
        vec![Def {
            data: PolymorphicDefinition { scheme: None },
            name: test_sym,
            package: None,
            ty: lambda_ty,
            body: lambda,
            meta: DefMeta::Function,
            arity: 1,
            param_diets: vec![types::Diet::observing()],
            return_diet: types::Diet::observing(),
        }],
        symbols,
        term_ids,
        transformed_global(),
    );

    let program = partial_eval(program);
    match &program.defs[0].body.kind {
        TermKind::Lambda(lam) => assert!(matches!(&lam.body.kind, TermKind::FloatLit(1.0))),
        other => panic!("expected Lambda, got {other:?}"),
    }
}

fn make_builtin_call(
    ids: &mut TermIdSource,
    name: &str,
    overload_idx: usize,
    args: Vec<Term<Empty, Empty>>,
    result_ty: Type<TypeName>,
) -> Term<Empty, Empty> {
    let builtin = builtins::catalog().lookup_by_surface_name(name).expect(name);
    let func_ty = args.iter().rev().fold(result_ty.clone(), |ret, arg| arrow_ty(arg.ty.clone(), ret));
    let func = Term {
        id: ids.next_id(),
        ty: func_ty,
        span: make_span(),
        kind: TermKind::Var(VarRef::Builtin {
            id: builtin.id,
            overload_idx,
        }),
    };
    Term {
        id: ids.next_id(),
        ty: result_ty,
        span: make_span(),
        kind: TermKind::App {
            func: Box::new(func),
            args,
        },
    }
}

fn make_float_vector(ids: &mut TermIdSource, values: &[f32]) -> Term<Empty, Empty> {
    let parts = values.iter().map(|value| make_float(ids, *value)).collect();
    Term {
        id: ids.next_id(),
        ty: types::vec(values.len(), float_ty()),
        span: make_span(),
        kind: TermKind::VecLit(parts),
    }
}

fn assert_float_vector(term: &Term<Empty, Empty>, expected: &[f32]) {
    let TermKind::VecLit(parts) = &term.kind else {
        panic!("expected constant vector: {term:?}")
    };
    assert_eq!(parts.len(), expected.len());
    assert_eq!(term.ty, types::vec(expected.len(), float_ty()));
    for (part, expected) in parts.iter().zip(expected) {
        let TermKind::FloatLit(actual) = part.kind else {
            panic!("expected scalar component: {part:?}")
        };
        assert_eq!(part.ty, float_ty());
        assert!(
            (actual - expected).abs() <= 1.0e-5 * expected.abs().max(1.0),
            "{actual} != {expected}"
        );
    }
}

#[test]
fn vector_math_folds_all_existing_componentwise_overloads() {
    for (scalar_name, args, expected) in float_math_cases() {
        let name = match scalar_name {
            "f32.lerp" => "vec.mix".to_owned(),
            "f32.fma" => continue, // No vector overload is published in the catalog.
            "mix" => "vec.mix".to_owned(),
            "smoothstep" => "vec.smoothstep".to_owned(),
            "step" => "step".to_owned(),
            name => name.replacen("f32.", "vec.", 1),
        };
        // Exercise the same dispatch both during evaluation and inside a
        // retained function body, which uses the residual-tree folder.
        for retained in [false, true] {
            let mut b = TestBuilder::new();
            let test_sym = b.sym("test");
            let ignored = b.sym("ignored");
            let args = args.iter().map(|value| make_float_vector(&mut b.ids, &[*value; 3])).collect();
            let mut body = make_builtin_call(&mut b.ids, &name, 0, args, types::vec(3, float_ty()));
            if retained {
                body = Term {
                    id: b.next_id(),
                    ty: arrow_ty(float_ty(), body.ty.clone()),
                    span: make_span(),
                    kind: TermKind::Lambda(Lambda {
                        params: vec![(ignored, float_ty())],
                        ret_ty: body.ty.clone(),
                        body: Box::new(body),
                    }),
                };
            }
            let result = partial_eval(make_program(test_sym, body, b.finish()));
            let mut body = &result.defs[0].body;
            if let TermKind::Lambda(lambda) = &body.kind {
                body = &lambda.body;
            }
            assert_float_vector(body, &[expected; 3]);
        }
    }
}

fn partial_source_body(source: &str, name: &str) -> Term<Empty, Empty> {
    let typed = crate::compile_thru_frontend(source).expect("type check");
    let typed = crate::ast_type_holes::reject_type_holes(typed).expect("type holes");
    let tlc = tlc::lower_from_ast(typed).expect("lower to TLC");
    let tlc = tlc::validate_ownership(tlc).expect("ownership");
    let tlc = partial_eval(tlc);
    let def = tlc
        .defs
        .iter()
        .find(|def| tlc.symbols.get(def.name).is_some_and(|symbol| symbol == name))
        .expect(name);
    let mut body = &def.body;
    while let TermKind::Lambda(lambda) = &body.kind {
        body = &lambda.body;
    }
    body.clone()
}

#[test]
fn vector_math_folds_source_constants_chains_and_scalar_broadcasts() {
    for (expression, expected) in [
        ("vec.sin(@[0.0, 0.0])", vec![0.0, 0.0]),
        ("vec.pow(@[2.0, 3.0], @[3.0, 2.0])", vec![8.0, 9.0]),
        ("vec.floor(vec.exp2(@[1.0, 2.0]))", vec![2.0, 4.0]),
        (
            "vec.abs(@[-1.0, -2.0, -3.0, -4.0].wzyx)",
            vec![4.0, 3.0, 2.0, 1.0],
        ),
        ("vec.floor(vec2f32(@[1i32, 3i32]))", vec![1.0, 3.0]),
        ("clamp(@[-2.0, 0.5, 3.0], 0.0, 1.0)", vec![0.0, 0.5, 1.0]),
        ("mix(@[0.0, 4.0], @[4.0, 8.0], 0.25)", vec![1.0, 5.0]),
        ("smoothstep(0.0, 1.0, @[0.25, 0.75])", vec![0.15625, 0.84375]),
        ("step(0.5, @[0.25, 0.5, 0.75])", vec![0.0, 1.0, 1.0]),
        ("let v = @[-2.0, 4.0] in vec.max(v, @[0.0, 3.0])", vec![0.0, 4.0]),
        (
            "let v = vec.exp2(@[1.0, 2.0]) in vec.sqrt(v)",
            vec![2.0f32.sqrt(), 2.0],
        ),
    ] {
        let source = format!("entry e(x:f32) vec{}f32 = {expression}", expected.len());
        assert_float_vector(&partial_source_body(&source, "e"), &expected);
        crate::compile_thru_spirv(&source).unwrap_or_else(|error| panic!("{source}: {error:?}"));
        let ssa = crate::compile_thru_ssa(&source).expect("SSA");
        let wgsl = crate::lower_ssa_to_wgsl(ssa).expect("WGSL");
        validate_math_wgsl(&wgsl);
    }
    let source = "def v:vec2f32 = vec.exp2(@[1.0, 2.0])\nentry e(x:f32) vec2f32 = vec.max(v, @[3.0, 3.0])";
    assert_float_vector(&partial_source_body(source, "e"), &[3.0, 4.0]);
}

fn validate_math_wgsl(source: &str) {
    let module = naga::front::wgsl::parse_str(source).unwrap_or_else(|error| panic!("{error}\n{source}"));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|error| panic!("{error:?}\n{source}"));
}

#[test]
fn clamp_runtime_vector_broadcast_matches_constant_folding() {
    for source in [
        "entry e(v:vec3f32) vec3f32 = clamp(v, 0.0, 1.0)",
        "entry e(v:vec3i32) vec3i32 = clamp(v, 0i32, 10i32)",
        "entry e(v:vec3u32) vec3u32 = clamp(v, 0u32, 10u32)",
    ] {
        let spirv = crate::compile_thru_spirv(source).expect("SPIR-V");
        let options = naga::front::spv::Options::default();
        let module = naga::front::spv::Frontend::new(spirv.spirv.into_iter(), &options)
            .parse()
            .unwrap_or_else(|error| panic!("{source}: {error:?}"));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|error| panic!("{source}: {error:?}"));
        let ssa = crate::compile_thru_ssa(source).expect("SSA");
        validate_math_wgsl(&crate::lower_ssa_to_wgsl(ssa).expect("WGSL"));
    }
}

#[test]
fn generic_integer_math_folds_scalar_and_vector_constants() {
    let source = "entry e(x:i32) vec3i32 = max(abs(@[-2i32, -3i32, 4i32]), @[1i32, 5i32, 2i32])";
    let term = partial_source_body(source, "e");
    let TermKind::VecLit(parts) = term.kind else {
        panic!("expected integer vector: {term:?}")
    };
    let values: Vec<_> = parts
        .iter()
        .map(|part| {
            let TermKind::IntLit(value) = &part.kind else {
                panic!("{part:?}")
            };
            value.parse::<i32>().unwrap()
        })
        .collect();
    assert_eq!(values, [2, 5, 4]);
    assert!(
        matches!(partial_source_body("entry e(x:i32) i32 = min(-2i32, 3i32)", "e").kind,
        TermKind::IntLit(value) if value == "-2")
    );
}

#[test]
fn constant_vectors_remain_valid_arguments_to_polymorphic_helpers() {
    let source = "def pair(x) = (x, x)\ndef p = pair(@[1.0, 2.0])\nentry e() (vec2f32, vec2f32) = p";
    crate::compile_thru_spirv(source).expect("constant vector through generic helper");
}

#[test]
fn scalar_float_classification_folds_to_booleans() {
    for (name, input, expected) in [
        ("f32.isnan", f32::NAN, true),
        ("f32.isnan", 1.0, false),
        ("f32.isinf", f32::NEG_INFINITY, true),
        ("f32.isinf", f32::NAN, false),
    ] {
        let mut b = TestBuilder::new();
        let test_sym = b.sym("test");
        let arg = make_float(&mut b.ids, input);
        let body = make_builtin_call(
            &mut b.ids,
            name,
            0,
            vec![arg],
            Type::Constructed(TypeName::Bool, vec![]),
        );
        let result = partial_eval(make_program(test_sym, body, b.finish()));
        assert!(matches!(result.defs[0].body.kind, TermKind::BoolLit(value) if value == expected));
    }
}

#[test]
fn vector_math_keeps_unknown_lanes_and_invalid_domains_residual() {
    for expression in [
        "vec.sin(@[x, 0.0])",
        "vec.sqrt(@[4.0, -1.0])",
        "vec.pow(@[2.0, -2.0], @[2.0, 2.0])",
        "clamp(@[0.0, 1.0], 2.0, 1.0)",
        "smoothstep(1.0, 1.0, @[0.0, 2.0])",
    ] {
        let source = format!("entry e(x:f32) vec2f32 = {expression}");
        assert!(
            matches!(partial_source_body(&source, "e").kind, TermKind::App { .. }),
            "{source}"
        );
    }
    // Geometry combines lanes and must not go through componentwise lifting.
    let source = "entry e(x:f32) vec2f32 = normalize(@[3.0, 4.0])";
    assert!(matches!(
        partial_source_body(source, "e").kind,
        TermKind::App { .. }
    ));
}

#[test]
fn scalar_math_preserves_signed_zero_and_declared_precision() {
    for (name, args, expected) in [
        ("f32.min", vec![0.0, -0.0], -0.0f32),
        ("f32.max", vec![-0.0, 0.0], 0.0),
        ("f32.abs", vec![-0.0], 0.0),
        ("f32.sign", vec![-0.0], 0.0),
        ("f32.round", vec![-0.5], -0.0),
        ("f32.fract", vec![-0.0], 0.0),
    ] {
        let TermKind::FloatLit(value) = eval_float_builtin(name, &args).kind else {
            panic!("{name}")
        };
        assert_eq!(value.to_bits(), expected.to_bits(), "{name}");
    }
    for (bits, input, expected) in [(16, 1.0004, Some(0.0)), (16, 16.0, None), (64, 1.0, None)] {
        let ty = Type::Constructed(TypeName::Float(bits), vec![]);
        let name = if input == 16.0 {
            "f16.exp2"
        } else if bits == 16 {
            "f16.fract"
        } else {
            "f64.sin"
        };
        let mut b = TestBuilder::new();
        let test_sym = b.sym("test");
        let mut arg = make_float(&mut b.ids, input);
        arg.ty = ty.clone();
        let body = make_builtin_call(&mut b.ids, name, 0, vec![arg], ty);
        let result = partial_eval(make_program(test_sym, body, b.finish()));
        match expected {
            Some(expected) => {
                assert!(matches!(result.defs[0].body.kind, TermKind::FloatLit(v) if v == expected))
            }
            None => assert!(matches!(result.defs[0].body.kind, TermKind::App { .. })),
        }
    }
}

#[test]
fn integer_math_respects_width_signedness_and_clamp_domains() {
    for (name, ty, args, expected) in [
        ("i8.abs", TypeName::Int(8), vec![-128], Some(-128)),
        ("i64.abs", TypeName::Int(64), vec![i64::MIN], Some(i64::MIN)),
        ("i32.sign", TypeName::Int(32), vec![-100], Some(-1)),
        ("i32.min", TypeName::Int(32), vec![-5, 3], Some(-5)),
        ("i32.max", TypeName::Int(32), vec![-5, 3], Some(3)),
        ("u64.max", TypeName::UInt(64), vec![-1, 3], Some(-1)),
        ("u64.min", TypeName::UInt(64), vec![-1, 3], Some(3)),
        ("u64.clamp", TypeName::UInt(64), vec![-1, 0, 100], Some(100)),
        ("i32.clamp", TypeName::Int(32), vec![0, 2, 1], None),
    ] {
        let mut b = TestBuilder::new();
        let test_sym = b.sym("test");
        let ty = Type::Constructed(ty, vec![]);
        let args = args
            .into_iter()
            .map(|value| {
                let mut arg = make_int(&mut b.ids, value);
                arg.ty = ty.clone();
                arg
            })
            .collect();
        let body = make_builtin_call(&mut b.ids, name, 0, args, ty);
        let result = partial_eval(make_program(test_sym, body, b.finish()));
        match expected {
            Some(expected) => assert!(
                matches!(&result.defs[0].body.kind, TermKind::IntLit(v) if v == &expected.to_string()),
                "{name}"
            ),
            None => assert!(matches!(result.defs[0].body.kind, TermKind::App { .. }), "{name}"),
        }
    }
}

#[test]
fn test_algebraic_add_zero() {
    let mut b = TestBuilder::new();
    let x_sym = b.sym("x");
    let test_sym = b.sym("test");

    let x = Term {
        id: b.next_id(),
        ty: Type::Constructed(TypeName::Int(32), vec![]),
        span: b.span(),
        kind: TermKind::Var(VarRef::Symbol(x_sym)),
    };
    let zero = make_int(&mut b.ids, 0);
    let term = make_binop(&mut b.ids, BinaryOperator::Add, x, zero);

    let program = make_program(test_sym, term, b.finish());

    let result = partial_eval(program);

    // x + 0 should simplify to just x
    match &result.defs[0].body.kind {
        TermKind::Var(VarRef::Symbol(sym)) => {
            let name = result.symbols.get(*sym).expect("BUG: symbol not in table");
            assert_eq!(name, "x");
        }
        other => panic!("Expected Var(x), got {:?}", other),
    }
}

#[test]
fn test_algebraic_mul_one() {
    let mut b = TestBuilder::new();
    let x_sym = b.sym("x");
    let test_sym = b.sym("test");

    let one = make_int(&mut b.ids, 1);
    let x = Term {
        id: b.next_id(),
        ty: Type::Constructed(TypeName::Int(32), vec![]),
        span: b.span(),
        kind: TermKind::Var(VarRef::Symbol(x_sym)),
    };
    let term = make_binop(&mut b.ids, BinaryOperator::Multiply, one, x);

    let program = make_program(test_sym, term, b.finish());

    let result = partial_eval(program);

    // 1 * x should simplify to just x
    match &result.defs[0].body.kind {
        TermKind::Var(VarRef::Symbol(sym)) => {
            let name = result.symbols.get(*sym).expect("BUG: symbol not in table");
            assert_eq!(name, "x");
        }
        other => panic!("Expected Var(x), got {:?}", other),
    }
}

#[test]
fn test_algebraic_mul_zero() {
    let mut b = TestBuilder::new();
    let x_sym = b.sym("x");
    let test_sym = b.sym("test");

    let x = Term {
        id: b.next_id(),
        ty: Type::Constructed(TypeName::Int(32), vec![]),
        span: b.span(),
        kind: TermKind::Var(VarRef::Symbol(x_sym)),
    };
    let zero = make_int(&mut b.ids, 0);
    let term = make_binop(&mut b.ids, BinaryOperator::Multiply, x, zero);

    let program = make_program(test_sym, term, b.finish());

    let result = partial_eval(program);

    // x * 0 should simplify to 0
    match &result.defs[0].body.kind {
        TermKind::IntLit(s) => assert_eq!(s, "0"),
        other => panic!("Expected IntLit(0), got {:?}", other),
    }
}

#[test]
fn test_if_true_elimination() {
    let mut b = TestBuilder::new();
    let test_sym = b.sym("test");

    let cond = make_bool(&mut b.ids, true);
    let then_branch = make_int(&mut b.ids, 1);
    let else_branch = make_int(&mut b.ids, 2);
    let term = Term {
        id: b.next_id(),
        ty: Type::Constructed(TypeName::Int(32), vec![]),
        span: b.span(),
        kind: TermKind::If {
            cond: Box::new(cond),
            then_branch: Box::new(then_branch),
            else_branch: Box::new(else_branch),
        },
    };

    let program = make_program(test_sym, term, b.finish());

    let result = partial_eval(program);

    // if true then 1 else 2 should simplify to 1
    match &result.defs[0].body.kind {
        TermKind::IntLit(s) => assert_eq!(s, "1"),
        other => panic!("Expected IntLit(1), got {:?}", other),
    }
}

#[test]
fn test_if_false_elimination() {
    let mut b = TestBuilder::new();
    let test_sym = b.sym("test");

    let cond = make_bool(&mut b.ids, false);
    let then_branch = make_int(&mut b.ids, 1);
    let else_branch = make_int(&mut b.ids, 2);
    let term = Term {
        id: b.next_id(),
        ty: Type::Constructed(TypeName::Int(32), vec![]),
        span: b.span(),
        kind: TermKind::If {
            cond: Box::new(cond),
            then_branch: Box::new(then_branch),
            else_branch: Box::new(else_branch),
        },
    };

    let program = make_program(test_sym, term, b.finish());

    let result = partial_eval(program);

    // if false then 1 else 2 should simplify to 2
    match &result.defs[0].body.kind {
        TermKind::IntLit(s) => assert_eq!(s, "2"),
        other => panic!("Expected IntLit(2), got {:?}", other),
    }
}

#[test]
fn test_let_constant_propagation() {
    let mut b = TestBuilder::new();
    let x_sym = b.sym("x");
    let test_sym = b.sym("test");

    // let x = 5 in x + 3
    let rhs = make_int(&mut b.ids, 5);
    let x_var = Term {
        id: b.next_id(),
        ty: Type::Constructed(TypeName::Int(32), vec![]),
        span: b.span(),
        kind: TermKind::Var(VarRef::Symbol(x_sym)),
    };
    let three = make_int(&mut b.ids, 3);
    let body_expr = make_binop(&mut b.ids, BinaryOperator::Add, x_var, three);
    let term = Term {
        id: b.next_id(),
        ty: Type::Constructed(TypeName::Int(32), vec![]),
        span: b.span(),
        kind: TermKind::Let {
            name: x_sym,
            name_ty: Type::Constructed(TypeName::Int(32), vec![]),
            rhs: Box::new(rhs),
            body: Box::new(body_expr),
        },
    };

    let program = make_program(test_sym, term, b.finish());

    let result = partial_eval(program);

    // let x = 5 in x + 3 should simplify to 8
    match &result.defs[0].body.kind {
        TermKind::IntLit(s) => assert_eq!(s, "8"),
        other => panic!("Expected IntLit(8), got {:?}", other),
    }
}

#[test]
fn test_function_inlining() {
    // def foo(a, b) = a + b
    // def bar = foo(8, 9)
    // bar should evaluate to 17
    let mut b = TestBuilder::new();
    let int_ty = Type::Constructed(TypeName::Int(32), vec![]);

    let a_sym = b.sym("a");
    let b_sym = b.sym("b");
    let foo_sym = b.sym("foo");
    let bar_sym = b.sym("bar");

    // Build foo: |a| |b| a + b
    let a_var = Term {
        id: b.next_id(),
        ty: int_ty.clone(),
        span: b.span(),
        kind: TermKind::Var(VarRef::Symbol(a_sym)),
    };
    let b_var = Term {
        id: b.next_id(),
        ty: int_ty.clone(),
        span: b.span(),
        kind: TermKind::Var(VarRef::Symbol(b_sym)),
    };
    let a_plus_b = make_binop(&mut b.ids, BinaryOperator::Add, a_var, b_var);

    let foo_body = Term {
        id: b.next_id(),
        ty: Type::Constructed(
            TypeName::Arrow,
            vec![
                int_ty.clone(),
                Type::Constructed(TypeName::Arrow, vec![int_ty.clone(), int_ty.clone()]),
            ],
        ),
        span: b.span(),
        kind: TermKind::Lambda(Lambda {
            params: vec![(a_sym, int_ty.clone()), (b_sym, int_ty.clone())],
            body: Box::new(a_plus_b),
            ret_ty: int_ty.clone(),
        }),
    };

    // Build bar: foo 8 9
    let eight = make_int(&mut b.ids, 8);
    let nine = make_int(&mut b.ids, 9);

    let foo_ref = Term {
        id: b.next_id(),
        ty: foo_body.ty.clone(),
        span: b.span(),
        kind: TermKind::Var(VarRef::Symbol(foo_sym)),
    };

    let bar_body = Term {
        id: b.next_id(),
        ty: int_ty.clone(),
        span: b.span(),
        kind: TermKind::App {
            func: Box::new(foo_ref),
            args: vec![eight, nine],
        },
    };

    let (symbols, term_ids) = b.finish();

    let program = Program::from_parts(
        vec![
            Def {
                data: PolymorphicDefinition { scheme: None },
                name: foo_sym,
                package: None,
                ty: foo_body.ty.clone(),
                body: foo_body,
                meta: DefMeta::Function,
                arity: 2,
                param_diets: vec![],
                return_diet: types::Diet::observing(),
            },
            Def {
                data: PolymorphicDefinition { scheme: None },
                name: bar_sym,
                package: None,
                ty: int_ty.clone(),
                body: bar_body,
                meta: DefMeta::Function,
                arity: 0,
                param_diets: vec![],
                return_diet: types::Diet::observing(),
            },
        ],
        symbols,
        term_ids,
        transformed_global(),
    );

    let result = partial_eval(program);

    // bar should be inlined and folded to 17
    let bar_def = result
        .defs
        .iter()
        .find(|d| result.symbols.get(d.name).expect("BUG: symbol not in table") == "bar")
        .unwrap();
    match &bar_def.body.kind {
        TermKind::IntLit(s) => assert_eq!(s, "17"),
        other => panic!("Expected IntLit(17), got {:?}", other),
    }
}

fn int_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Int(32), vec![])
}

fn arrow_ty(from: Type<TypeName>, to: Type<TypeName>) -> Type<TypeName> {
    Type::Constructed(TypeName::Arrow, vec![from, to])
}

/// Test that `let f = g in f x` is inlined to `g x`
#[test]
fn test_function_alias_inlining() {
    let mut b = TestBuilder::new();

    let y_sym = b.sym("y");
    let f_sym = b.sym("f");
    let g_sym = b.sym("g");
    let main_sym = b.sym("main");

    let span = b.span();

    // Build: def g = |y| y  (identity function)
    let g_body = Term {
        id: b.next_id(),
        ty: arrow_ty(int_ty(), int_ty()),
        span,
        kind: TermKind::Lambda(Lambda {
            params: vec![(y_sym, int_ty())],
            body: Box::new(Term {
                id: b.next_id(),
                ty: int_ty(),
                span,
                kind: TermKind::Var(VarRef::Symbol(y_sym)),
            }),
            ret_ty: int_ty(),
        }),
    };

    // Build: def main = let f = g in f 42
    // This is: Let { name: "f", rhs: Var("g"), body: App(Var("f"), 42) }
    let main_body = Term {
        id: b.next_id(),
        ty: int_ty(),
        span,
        kind: TermKind::Let {
            name: f_sym,
            name_ty: arrow_ty(int_ty(), int_ty()),
            rhs: Box::new(Term {
                id: b.next_id(),
                ty: arrow_ty(int_ty(), int_ty()),
                span,
                kind: TermKind::Var(VarRef::Symbol(g_sym)),
            }),
            body: Box::new(Term {
                id: b.next_id(),
                ty: int_ty(),
                span,
                kind: TermKind::App {
                    func: Box::new(Term {
                        id: b.next_id(),
                        ty: arrow_ty(int_ty(), int_ty()),
                        span,
                        kind: TermKind::Var(VarRef::Symbol(f_sym)),
                    }),
                    args: vec![Term {
                        id: b.next_id(),
                        ty: int_ty(),
                        span,
                        kind: TermKind::IntLit("42".to_string()),
                    }],
                },
            }),
        },
    };

    let (symbols, term_ids) = b.finish();

    let program = Program::from_parts(
        vec![
            Def {
                data: PolymorphicDefinition { scheme: None },
                name: g_sym,
                package: None,
                ty: g_body.ty.clone(),
                body: g_body,
                meta: DefMeta::Function,
                arity: 1,
                param_diets: vec![],
                return_diet: types::Diet::observing(),
            },
            Def {
                data: PolymorphicDefinition { scheme: None },
                name: main_sym,
                package: None,
                ty: int_ty(),
                body: main_body,
                meta: DefMeta::Function,
                arity: 0,
                param_diets: vec![],
                return_diet: types::Diet::observing(),
            },
        ],
        symbols,
        term_ids,
        transformed_global(),
    );

    let result = partial_eval(program);

    // Find main's body - it should be simplified to just `42`
    // because g is identity and f aliases g, so f 42 = g 42 = 42
    let main_def = result
        .defs
        .iter()
        .find(|d| result.symbols.get(d.name).expect("BUG: symbol not in table") == "main")
        .unwrap();

    // The result should be IntLit("42") since g is identity
    match &main_def.body.kind {
        TermKind::IntLit(s) => assert_eq!(s, "42"),
        other => panic!("Expected IntLit(42), got {:?}", other),
    }
}

/// Test that function alias without full application still uses correct name
#[test]
fn test_function_alias_partial_application() {
    let mut b = TestBuilder::new();

    let x_sym = b.sym("x");
    let y_sym = b.sym("y");
    let f_sym = b.sym("f");
    let g_sym = b.sym("g");
    let main_sym = b.sym("main");

    let span = b.span();

    // Build: def g = |x| |y| x  (const function, arity 2)
    let g_body = Term {
        id: b.next_id(),
        ty: arrow_ty(int_ty(), arrow_ty(int_ty(), int_ty())),
        span,
        kind: TermKind::Lambda(Lambda {
            params: vec![(x_sym, int_ty()), (y_sym, int_ty())],
            body: Box::new(Term {
                id: b.next_id(),
                ty: int_ty(),
                span,
                kind: TermKind::Var(VarRef::Symbol(x_sym)),
            }),
            ret_ty: int_ty(),
        }),
    };

    // Build: def main = let f = g in f 1 2
    // f aliases g, so f 1 2 should become g 1 2 = 1
    let main_body = Term {
        id: b.next_id(),
        ty: int_ty(),
        span,
        kind: TermKind::Let {
            name: f_sym,
            name_ty: arrow_ty(int_ty(), arrow_ty(int_ty(), int_ty())),
            rhs: Box::new(Term {
                id: b.next_id(),
                ty: arrow_ty(int_ty(), arrow_ty(int_ty(), int_ty())),
                span,
                kind: TermKind::Var(VarRef::Symbol(g_sym)),
            }),
            body: Box::new(Term {
                id: b.next_id(),
                ty: int_ty(),
                span,
                kind: TermKind::App {
                    func: Box::new(Term {
                        id: b.next_id(),
                        ty: arrow_ty(int_ty(), arrow_ty(int_ty(), int_ty())),
                        span,
                        kind: TermKind::Var(VarRef::Symbol(f_sym)),
                    }),
                    args: vec![
                        Term {
                            id: b.next_id(),
                            ty: int_ty(),
                            span,
                            kind: TermKind::IntLit("1".to_string()),
                        },
                        Term {
                            id: b.next_id(),
                            ty: int_ty(),
                            span,
                            kind: TermKind::IntLit("2".to_string()),
                        },
                    ],
                },
            }),
        },
    };

    let (symbols, term_ids) = b.finish();

    let program = Program::from_parts(
        vec![
            Def {
                data: PolymorphicDefinition { scheme: None },
                name: g_sym,
                package: None,
                ty: g_body.ty.clone(),
                body: g_body,
                meta: DefMeta::Function,
                arity: 2,
                param_diets: vec![],
                return_diet: types::Diet::observing(),
            },
            Def {
                data: PolymorphicDefinition { scheme: None },
                name: main_sym,
                package: None,
                ty: int_ty(),
                body: main_body,
                meta: DefMeta::Function,
                arity: 0,
                param_diets: vec![],
                return_diet: types::Diet::observing(),
            },
        ],
        symbols,
        term_ids,
        transformed_global(),
    );

    let result = partial_eval(program);

    // Find main's body - it should be simplified to `1`
    // because g x y = x, so f 1 2 = g 1 2 = 1
    let main_def = result
        .defs
        .iter()
        .find(|d| result.symbols.get(d.name).expect("BUG: symbol not in table") == "main")
        .unwrap();

    match &main_def.body.kind {
        TermKind::IntLit(s) => assert_eq!(s, "1"),
        other => panic!("Expected IntLit(1), got {:?}", other),
    }
}

/// Test that `let f = f32.sin in f x` becomes `f32.sin x`
#[test]
fn test_intrinsic_alias_inlining() {
    let mut b = TestBuilder::new();

    let f_sym = b.sym("f");
    let f32_sin_sym = b.sym("f32.sin");
    let main_sym = b.sym("main");

    let span = b.span();
    let float_ty = Type::Constructed(TypeName::Float(32), vec![]);

    // Build: def main = let f = f32.sin in f 0.5
    // f32.sin is an intrinsic (not in defs), so it evaluates to Unknown(Var("f32.sin"))
    let main_body = Term {
        id: b.next_id(),
        ty: float_ty.clone(),
        span,
        kind: TermKind::Let {
            name: f_sym,
            name_ty: arrow_ty(float_ty.clone(), float_ty.clone()),
            rhs: Box::new(Term {
                id: b.next_id(),
                ty: arrow_ty(float_ty.clone(), float_ty.clone()),
                span,
                kind: TermKind::Var(VarRef::Symbol(f32_sin_sym)),
            }),
            body: Box::new(Term {
                id: b.next_id(),
                ty: float_ty.clone(),
                span,
                kind: TermKind::App {
                    func: Box::new(Term {
                        id: b.next_id(),
                        ty: arrow_ty(float_ty.clone(), float_ty.clone()),
                        span,
                        kind: TermKind::Var(VarRef::Symbol(f_sym)),
                    }),
                    args: vec![Term {
                        id: b.next_id(),
                        ty: float_ty.clone(),
                        span,
                        kind: TermKind::FloatLit(0.5),
                    }],
                },
            }),
        },
    };

    let (symbols, term_ids) = b.finish();

    let program = Program::from_parts(
        vec![Def {
            data: PolymorphicDefinition { scheme: None },
            name: main_sym,
            package: None,
            ty: float_ty.clone(),
            body: main_body,
            meta: DefMeta::Function,
            arity: 0,
            param_diets: vec![],
            return_diet: types::Diet::observing(),
        }],
        symbols,
        term_ids,
        transformed_global(),
    );

    let result = partial_eval(program);
    let main_def = result
        .defs
        .iter()
        .find(|d| result.symbols.get(d.name).expect("BUG: symbol not in table") == "main")
        .unwrap();

    // The result should be App(f32.sin, 0.5) - the alias `f` should be resolved to `f32.sin`
    match &main_def.body.kind {
        TermKind::App { func, args } => {
            // Check that the function is now f32.sin (not f)
            match &func.kind {
                TermKind::Var(VarRef::Symbol(sym)) => {
                    let name = result.symbols.get(*sym).expect("BUG: symbol not in table");
                    assert_eq!(name, "f32.sin");
                }
                other => panic!("Expected Var(f32.sin), got {:?}", other),
            }
            // Check the argument is still 0.5
            assert_eq!(args.len(), 1);
            match &args[0].kind {
                TermKind::FloatLit(f) => assert!((*f - 0.5).abs() < 0.001),
                other => panic!("Expected FloatLit(0.5), got {:?}", other),
            }
        }
        other => panic!("Expected App, got {:?}", other),
    }
}

// =============================================================================
// Substitution survives through SOAC residualization
// =============================================================================
//
// When residualizing a SOAC or ArrayExpr, free variables bound in the
// partial-evaluation environment must be substituted with their let RHS.
// Lambda parameters and inner let-bound names shadow that environment.

use crate::tlc::{ArrayExpr, SoacBody, SoacOp};

/// Assert that `target_sym` does not occur as a free `Var` anywhere
/// in `term`. Recurses via `map_children`, so it sees through Soac,
/// ArrayExpr, App, Tuple, etc. without needing per-variant handling.
/// Binder-aware via the `bound` set: Lambda params, Let names, etc.
/// shadow the assertion (we treat any matching symbol added by an
/// enclosing binder as a different identifier).
fn assert_no_free_reference_to(term: &Term<Empty, Empty>, target_sym: SymbolId) {
    fn walk(t: &Term<Empty, Empty>, target: SymbolId, shadowed: bool) {
        if shadowed {
            return;
        }
        match &t.kind {
            TermKind::Var(VarRef::Symbol(sym)) => {
                assert!(
                    *sym != target,
                    "free Var(sym={:?}) — let-bound symbol not substituted through SOAC",
                    sym.0,
                );
            }
            TermKind::Let { name, rhs, body, .. } => {
                walk(rhs, target, false);
                walk(body, target, *name == target);
            }
            TermKind::Lambda(lam) => {
                let shadow = lam.params.iter().any(|(p, _)| *p == target);
                walk(&lam.body, target, shadow);
            }
            _ => {
                t.for_each_child(&mut |child| walk(child, target, false));
            }
        }
    }
    walk(term, target_sym, false);
}

#[test]
fn let_bound_array_substituted_through_soac_input() {
    // Construct (paraphrased):
    //   def test() [3]i32 =
    //       let m: [3]i32 = [1, 2, 3] in
    //       map(|x: i32| x, m)
    //
    // After partial_eval, the let may be eliminated — but if so, every
    // reference to `m` inside the SOAC must be substituted with the literal.
    let mut b = TestBuilder::new();
    let m_sym = b.sym("m");
    let x_sym = b.sym("x");
    let test_sym = b.sym("test");

    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
    let arr_ty = Type::Constructed(
        TypeName::Array,
        vec![
            i32_ty.clone(),
            Type::Constructed(TypeName::Size(3), vec![]),
            Type::Constructed(TypeName::ArrayVariantComposite, vec![]),
            types::no_buffer(),
        ],
    );

    // [1, 2, 3] as ArrayExpr::Literal
    let lit_elems = vec![
        make_int(&mut b.ids, 1),
        make_int(&mut b.ids, 2),
        make_int(&mut b.ids, 3),
    ];
    let arr_lit_term = Term {
        id: b.next_id(),
        ty: arr_ty.clone(),
        span: b.span(),
        kind: TermKind::ArrayExpr(ArrayExpr::Literal(lit_elems)),
    };

    // `m`-typed var reference inside the SOAC input
    let m_var = Term {
        id: b.next_id(),
        ty: arr_ty.clone(),
        span: b.span(),
        kind: TermKind::Var(VarRef::Symbol(m_sym)),
    };

    // Identity lambda |x: i32| x
    let x_var = Term {
        id: b.next_id(),
        ty: i32_ty.clone(),
        span: b.span(),
        kind: TermKind::Var(VarRef::Symbol(x_sym)),
    };
    let lam = Lambda {
        params: vec![(x_sym, i32_ty.clone())],
        body: Box::new(x_var),
        ret_ty: i32_ty.clone(),
    };

    let soac_term = Term {
        id: b.next_id(),
        ty: arr_ty.clone(),
        span: b.span(),
        kind: TermKind::Soac(SoacOp::Map {
            lam: SoacBody { lam, data: () },
            inputs: vec![input_ae(Box::new(m_var))],
            destination: types::SoacOwnership::Fresh,
        }),
    };

    let let_term = Term {
        id: b.next_id(),
        ty: arr_ty.clone(),
        span: b.span(),
        kind: TermKind::Let {
            name: m_sym,
            name_ty: arr_ty.clone(),
            rhs: Box::new(arr_lit_term),
            body: Box::new(soac_term),
        },
    };

    let program = make_program(test_sym, let_term, b.finish());
    let result = partial_eval(program);

    assert_eq!(result.defs.len(), 1);
    assert_no_free_reference_to(&result.defs[0].body, m_sym);
}

#[test]
fn let_bound_value_is_substituted_through_residual_loop() {
    let mut b = TestBuilder::new();
    let n_sym = b.sym("n");
    let acc_sym = b.sym("acc");
    let index_sym = b.sym("i");
    let test_sym = b.sym("test");
    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);

    let bound = Term {
        id: b.next_id(),
        ty: i32_ty.clone(),
        span: b.span(),
        kind: TermKind::Var(VarRef::Symbol(n_sym)),
    };
    let body = Term {
        id: b.next_id(),
        ty: i32_ty.clone(),
        span: b.span(),
        kind: TermKind::Var(VarRef::Symbol(acc_sym)),
    };
    let loop_term = Term {
        id: b.next_id(),
        ty: i32_ty.clone(),
        span: b.span(),
        kind: TermKind::Loop {
            loop_var: acc_sym,
            loop_var_ty: i32_ty.clone(),
            init: Box::new(make_int(&mut b.ids, 0)),
            init_bindings: vec![],
            kind: LoopKind::ForRange {
                var: index_sym,
                var_ty: i32_ty.clone(),
                bound: Box::new(bound),
            },
            body: Box::new(body),
        },
    };
    let term = Term {
        id: b.next_id(),
        ty: i32_ty.clone(),
        span: b.span(),
        kind: TermKind::Let {
            name: n_sym,
            name_ty: i32_ty,
            rhs: Box::new(make_int(&mut b.ids, 10)),
            body: Box::new(loop_term),
        },
    };

    let result = partial_eval(make_program(test_sym, term, b.finish()));
    assert_no_free_reference_to(&result.defs[0].body, n_sym);
}

#[test]
fn parses_full_u64_literal_range_as_bit_patterns() {
    let ty = Type::Constructed(TypeName::UInt(64), vec![]);
    assert_eq!(super::parse_integer_value("0", &ty).unwrap(), 0);
    assert_eq!(
        super::parse_integer_value("9223372036854775808", &ty).unwrap(),
        i64::MIN
    );
    assert_eq!(
        super::parse_integer_value("18446744073709551615", &ty).unwrap(),
        -1
    );
    assert_eq!(super::parse_integer_value("-1", &ty).unwrap(), -1);
}

#[test]
fn signed_literals_keep_signed_range_checks() {
    let ty = Type::Constructed(TypeName::Int(64), vec![]);
    assert_eq!(
        super::parse_integer_value(i64::MAX.to_string().as_str(), &ty).unwrap(),
        i64::MAX
    );
    assert!(super::parse_integer_value("18446744073709551615", &ty).is_err());
}
