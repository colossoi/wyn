use super::*;
use crate::ast::{Span, TypeName};
use crate::egir;
use crate::egir::program::{semantic_program_for_test, Func, ProgramIdentities};
use crate::egir::soac::screma;
use crate::egir::types::{
    by_value_function_result, callable_parameter, CallEffects, OperandRef, PureOp, SegBody, Semantic,
    SoacEffect, SoacOwnership, WynLanguage,
};
use crate::pipeline_descriptor::PipelineDescriptor;
use crate::types;
use crate::BindingRef;
use crate::FunctionId;
use polytype::Type;
use smallvec::smallvec;

fn op(id: u32) -> SemanticOpId {
    SemanticOpId::for_test(id)
}

fn dep(
    producer: &SemanticOpId,
    consumer: &SemanticOpId,
    kind: SemanticDependencyKind,
) -> SemanticDependency {
    SemanticDependency {
        producer: producer.clone(),
        consumer: consumer.clone(),
        kind,
    }
}

#[test]
fn oracle_distinguishes_conflict_flow_and_value_edges() {
    let a = op(0);
    let b = op(1);
    let c = op(2);

    // a --Resource--> b, b --Effect--> c, a --Value--> c
    let deps = vec![
        dep(&a, &b, SemanticDependencyKind::Resource),
        dep(&b, &c, SemanticDependencyKind::Effect),
        dep(&a, &c, SemanticDependencyKind::Value),
    ];
    let g = SemanticGraph::new(&deps);

    // Resource and effect edges are reordering conflicts (both directions);
    // value edges are represented separately.
    assert!(g.conflicts(&a, &b), "resource edge is a conflict");
    assert!(g.conflicts(&b, &a));
    assert!(g.conflicts(&b, &c), "effect ordering is a reordering conflict");
    assert!(
        !g.conflicts(&a, &c),
        "a value edge alone is fusable, not a conflict"
    );

    // Reachability follows value edges only, not effect ordering or resource.
    assert!(g.reachable_between(&a, &c), "a --Value--> c");
    assert!(
        !g.reachable_between(&b, &c),
        "b --Effect--> c is ordering, not a value chain"
    );
    assert!(
        !g.reachable_between(&a, &b),
        "a--b is resource-only, not a value edge"
    );
    assert!(!g.reachable_between(&c, &a), "no back edge");
    assert_eq!(g.value_consumer_count(&a), 1);
    assert_eq!(g.value_consumer_count(&b), 0);
}

#[test]
fn unknown_ops_have_no_edges() {
    let a = op(0);
    let lonely = op(1);
    let g = SemanticGraph::new(&[dep(&a, &a, SemanticDependencyKind::Value)]);
    assert!(!g.conflicts(&lonely, &a));
    assert!(!g.reachable_between(&lonely, &a));
}

fn append_capturing_map(graph: &mut EGraph<Semantic>, id: u32, captures: Vec<ValueId>) {
    let ty = Type::Constructed(TypeName::Unit, vec![]);
    let result = graph.alloc_side_effect_result(ty.clone());
    let result_binding = graph.value_result(result);
    let block = graph.skeleton.entry;
    graph.skeleton.blocks[block].side_effects.push(SideEffect {
        kind: SideEffectKind::Soac(SoacEffect(
            op(id),
            Soac::Screma(screma::Op {
                inputs: vec![],
                form: screma::ScremaForm {
                    pre: screma::Lambda::region(
                        SegBody {
                            region: FunctionId::from_index(0),
                            captures: captures.into_iter().map(OperandRef::Value).collect(),
                        },
                        vec![],
                        vec![ty.clone()],
                    ),
                    scans: vec![],
                    reductions: vec![],
                    post: screma::Lambda::identity(vec![ty]),
                },
                result_state: vec![screma::ResultState {
                    ownership: SoacOwnership::Fresh,
                }],
                state: screma::SemanticState::Serial,
            }),
        )),
        operands: smallvec![],
        result: Some(result_binding),
        effects: None,
        span: None,
    });
}

fn array(element: Type<TypeName>) -> Type<TypeName> {
    Type::Constructed(
        TypeName::Array,
        vec![
            element,
            Type::Constructed(TypeName::ArrayVariantComposite, vec![]),
            Type::Constructed(TypeName::Size(4), vec![]),
            types::no_buffer(),
        ],
    )
}

fn screma_verification_program(operator: screma::Lambda, neutral_is_bool: bool) -> egir::reify::Segmented {
    let i32_type = Type::Constructed(TypeName::Int(32), vec![]);
    let array_type = array(i32_type.clone());
    let result_type = Type::Constructed(TypeName::Tuple(1), vec![array_type.clone()]);
    let mut graph = EGraph::new();
    let input = graph.add_test_value_parameter(0, array_type.clone());
    let neutral = if neutral_is_bool {
        graph.intern_pure(
            PureOp::Bool(false),
            smallvec![],
            Type::Constructed(TypeName::Bool, vec![]),
            None,
        )
    } else {
        graph.intern_pure(PureOp::Int("0".into()), smallvec![], i32_type.clone(), None)
    };
    let result = graph.alloc_side_effect_result(result_type.clone());
    let result_binding = graph.value_result(result);
    let block = graph.skeleton.entry;
    graph.skeleton.blocks[block].side_effects.push(SideEffect {
        kind: SideEffectKind::Soac(SoacEffect(
            op(0),
            Soac::Screma(screma::Op {
                inputs: vec![egir::types::SoacInputType::array(array_type.clone())],
                form: screma::ScremaForm {
                    pre: screma::Lambda::identity(vec![i32_type.clone()]),
                    scans: vec![screma::Scan {
                        operator,
                        neutral: vec![neutral],
                    }],
                    reductions: vec![],
                    post: screma::Lambda::identity(vec![i32_type]),
                },
                result_state: vec![screma::ResultState {
                    ownership: SoacOwnership::Fresh,
                }],
                state: screma::SemanticState::Serial,
            }),
        )),
        operands: smallvec![OperandRef::Value(input)],
        result: Some(result_binding),
        effects: None,
        span: None,
    });

    let mut identities = ProgramIdentities::default();
    let region = identities.alloc_function("malformed_screma".into());
    let function = Func::<Semantic>::new(
        region,
        "malformed_screma".to_string(),
        Span::dummy(),
        None,
        vec![callable_parameter::<BindingRef, WynLanguage>(
            "xs".into(),
            array_type,
        )],
        by_value_function_result::<WynLanguage>(result_type),
        CallEffects::General,
        graph,
    );
    semantic_program_for_test(
        vec![function],
        vec![],
        vec![],
        vec![],
        PipelineDescriptor::default(),
        identities,
    )
}

#[test]
fn verifier_rejects_identity_screma_operator_without_panicking() {
    let i32_type = Type::Constructed(TypeName::Int(32), vec![]);
    let program =
        screma_verification_program(screma::Lambda::identity(vec![i32_type.clone(), i32_type]), false);

    let error = verify(&program).expect_err("identity collective operator must be rejected");
    assert!(
        error.contains("scan 0 operator is identity"),
        "unexpected error: {error}"
    );
}

#[test]
fn verifier_rejects_screma_neutral_type_mismatch() {
    let i32_type = Type::Constructed(TypeName::Int(32), vec![]);
    let program = screma_verification_program(
        screma::Lambda::region(
            SegBody {
                region: FunctionId::from_index(0),
                captures: vec![],
            },
            vec![i32_type.clone(), i32_type.clone()],
            vec![i32_type],
        ),
        true,
    );

    let error = verify(&program).expect_err("neutral type mismatch must be rejected");
    assert!(error.contains("neutral 0"), "unexpected error: {error}");
}
#[test]
fn scheduled_operations_expose_shared_prelude_inputs() {
    let mut egir = EGraph::<Semantic>::new();
    let source = egir.add_test_value_parameter(0, Type::Constructed(TypeName::Unit, vec![]));
    append_capturing_map(&mut egir, 10, vec![source, source]);
    append_capturing_map(&mut egir, 11, vec![source]);

    let graph = SemanticGraph::with_operation_captures(&[], &egir);
    assert_eq!(graph.captured_values().collect::<Vec<_>>(), vec![source]);
    assert_eq!(
        graph.capture_consumers(source).collect::<Vec<_>>(),
        vec![op(10), op(11)]
    );
    assert_eq!(graph.operation_site(&op(10)).map(|site| site.index), Some(0));
    assert_eq!(graph.operation_site(&op(11)).map(|site| site.index), Some(1));
}
