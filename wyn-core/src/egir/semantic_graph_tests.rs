use super::*;
use crate::ast::{Span, TypeName};
use crate::egir;
use crate::egir::ir::ResultBinding;
use crate::egir::program::{semantic_program_for_test, Func, ProgramIdentities};
use crate::egir::soac::screma;
use crate::egir::soac::Lambda;
use crate::egir::types::{
    by_value_function_result, callable_parameter, CallEffects, EffectOp, OperandRef, Parameters, PureOp,
    SegBody, Semantic, SoacEffect, SoacOwnership, WynLanguage,
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

#[test]
fn consumers_follow_value_incidences_and_deduplicate_captures() {
    let mut egir = EGraph::<Semantic>::new();
    let a = append_capturing_map(&mut egir, 0, vec![]);
    append_capturing_map(&mut egir, 1, vec![]);
    let c = append_capturing_map(&mut egir, 2, vec![a, a]);
    append_capturing_map(&mut egir, 3, vec![c]);
    let graph = SemanticGraph::new(&egir);
    assert_eq!(
        graph.value_consumers(&op(0)).collect::<Vec<_>>(),
        vec![op(2), op(3)]
    );
    assert_eq!(graph.value_consumer_count(&op(0)), 2);
    assert_eq!(graph.value_consumer_count(&op(1)), 0);
}

#[test]
fn unknown_ops_have_no_edges() {
    let egir = EGraph::<Semantic>::new();
    let graph = SemanticGraph::new(&egir);
    assert_eq!(graph.value_consumer_count(&op(1)), 0);
    assert_eq!(graph.value_consumers(&op(1)).count(), 0);
}

fn append_scalar_effect(graph: &mut EGraph<Semantic>, input: ValueId) -> ValueId {
    let result = graph.alloc_side_effect_result(graph.nodes[input].ty.clone());
    let binding = graph.value_result(result);
    graph.skeleton.blocks[graph.skeleton.entry].side_effects.push(SideEffect {
        kind: SideEffectKind::Effect(EffectOp::Op {
            tag: PureOp::Materialize,
        }),
        operands: smallvec![OperandRef::Value(input)],
        result: Some(binding),
        effects: None,
        span: None,
    });
    result
}

#[test]
fn scalar_only_bodies_have_no_semantic_dependencies() {
    let mut graph = EGraph::<Semantic>::new();
    let input = graph.intern_pure(
        PureOp::Int("1".into()),
        smallvec![],
        Type::Constructed(TypeName::Int(32), vec![]),
        None,
    );
    append_scalar_effect(&mut graph, input);
    let empty = EGraph::<Semantic>::new();
    let scalar_analysis = GraphAnalysis::new(&graph);
    let empty_analysis = GraphAnalysis::new(&empty);
    for index in [
        SemanticGraph::new(&graph),
        SemanticGraph::for_bodies([
            (BodySite::Entry(0), &empty_analysis),
            (BodySite::Entry(1), &scalar_analysis),
        ]),
    ] {
        assert!(index.sites.is_empty());
        assert!(index.consumers.is_empty());
        assert!(index.captures.is_empty());
        assert!(index.array_residency_demands.is_empty());
    }
}

#[test]
fn mixed_bodies_keep_dependencies_through_ordinary_effects() {
    let mut scalar = EGraph::<Semantic>::new();
    let value = scalar.add_block_param(scalar.skeleton.entry, Type::Constructed(TypeName::Unit, vec![]));
    append_scalar_effect(&mut scalar, value);

    let mut mixed = EGraph::<Semantic>::new();
    let producer = append_capturing_map(&mut mixed, 0, vec![]);
    let intermediate = append_scalar_effect(&mut mixed, producer);
    append_capturing_map(&mut mixed, 1, vec![intermediate]);
    let scalar_analysis = GraphAnalysis::new(&scalar);
    let mixed_analysis = GraphAnalysis::new(&mixed);
    let bodies = [
        (BodySite::Entry(0), &scalar_analysis),
        (BodySite::Entry(1), &mixed_analysis),
    ];
    let index = SemanticGraph::for_bodies(bodies);
    assert_eq!(index.value_consumers(&op(0)).collect::<Vec<_>>(), vec![op(1)]);
    assert_eq!(index.operation_site(&op(1)).map(|site| site.index), Some(2));
    assert_eq!(
        index
            .capture_consumers(SourceValue {
                body: BodySite::Entry(1),
                value: intermediate
            })
            .collect::<Vec<_>>(),
        vec![op(1)]
    );

    let mut facts = Facts::new();
    for (body, analysis) in bodies {
        facts.add_body(body, analysis, []).unwrap();
    }
    let full = SemanticGraph::from_facts(facts);
    assert_eq!(index.sites, full.sites);
    assert_eq!(index.consumers, full.consumers);
    assert_eq!(index.captures, full.captures);
    assert_eq!(index.array_residency_demands, full.array_residency_demands);
}

fn append_capturing_map(graph: &mut EGraph<Semantic>, id: u32, captures: Vec<ValueId>) -> ValueId {
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
                    pre: Lambda::region(
                        SegBody {
                            region: FunctionId::from_index(0),
                            captures: captures.into_iter().map(OperandRef::Value).collect(),
                        },
                        vec![],
                        vec![ty.clone()],
                    ),
                    scans: vec![],
                    reductions: vec![],
                    post: Lambda::identity(vec![ty]),
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
    result
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

fn screma_verification_program(operator: Lambda, neutral_is_bool: bool) -> egir::reify::Segmented {
    let i32_type = Type::Constructed(TypeName::Int(32), vec![]);
    let array_type = array(i32_type.clone());
    let result_type = Type::Constructed(TypeName::Tuple(1), vec![array_type.clone()]);
    let params = Parameters::from_ordered([callable_parameter::<BindingRef, WynLanguage>(
        "xs".into(),
        array_type.clone(),
    )]);
    let mut graph = EGraph::new();
    let input = graph.add_test_value_parameter(params.ids().next().unwrap(), array_type.clone());
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
                    pre: Lambda::identity(vec![i32_type.clone()]),
                    scans: vec![screma::Scan {
                        operator,
                        neutral: vec![neutral],
                    }],
                    reductions: vec![],
                    post: Lambda::identity(vec![i32_type]),
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
        Span::generated(),
        None,
        params,
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
    let program = screma_verification_program(Lambda::identity(vec![i32_type.clone(), i32_type]), false);

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
        Lambda::region(
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
    let source = egir.add_block_param(egir.skeleton.entry, Type::Constructed(TypeName::Unit, vec![]));
    append_capturing_map(&mut egir, 10, vec![source, source]);
    append_capturing_map(&mut egir, 11, vec![source]);

    let graph = SemanticGraph::new(&egir);
    assert_eq!(
        graph.captured_values(BodySite::Entry(0)).collect::<Vec<_>>(),
        vec![source]
    );
    assert_eq!(
        graph
            .capture_consumers(SourceValue {
                body: BodySite::Entry(0),
                value: source
            })
            .collect::<Vec<_>>(),
        vec![op(10), op(11)]
    );
    assert_eq!(graph.operation_site(&op(10)).map(|site| site.index), Some(0));
    assert_eq!(graph.operation_site(&op(11)).map(|site| site.index), Some(1));
}

#[test]
fn shared_facts_preserve_cross_block_observers_without_contraction_edges() {
    let mut egir = EGraph::<Semantic>::new();
    let a = append_capturing_map(&mut egir, 0, vec![]);
    append_capturing_map(&mut egir, 1, vec![a]);
    let entry = egir.skeleton.entry;
    let next = egir.skeleton.create_block();
    let consumer = egir.skeleton.blocks[entry].side_effects.pop().unwrap();
    egir.skeleton.blocks[next].side_effects.push(consumer);
    let mut facts = Facts::new();
    facts
        .add_body(
            BodySite::Entry(0),
            &crate::egir::analysis::GraphAnalysis::new(&egir),
            [],
        )
        .unwrap();
    let groups = facts.operations.keys().copied().collect::<Vec<_>>();
    let producer = facts.builder.outputs(groups[0]).unwrap()[0];
    let input = facts.ports[&((BodySite::Entry(0), next), a)];
    assert_eq!(facts.external[&input], vec![producer]);
    let graph = facts.builder.clone().finish(groups.iter().map(|id| (*id, ()))).unwrap();
    assert!(graph.producers(input).unwrap().is_empty());
    assert!(graph.boundary(&[groups[0]], &[]).unwrap().outputs.contains(&producer));
    assert_eq!(graph.boundary(&groups, &[]), Err(wyn_fusion::Error::Scope));
    assert_eq!(
        SemanticGraph::from_facts(facts).value_consumers(&op(0)).collect::<Vec<_>>(),
        vec![op(1)]
    );
}

#[test]
fn shared_facts_qualify_capture_sources_by_body() {
    let mut first = EGraph::<Semantic>::new();
    let value = first.add_block_param(first.skeleton.entry, Type::Constructed(TypeName::Unit, vec![]));
    let mut second = first.clone();
    append_capturing_map(&mut first, 0, vec![value]);
    append_capturing_map(&mut second, 1, vec![value]);
    let mut facts = Facts::new();
    facts
        .add_body(
            BodySite::Entry(0),
            &crate::egir::analysis::GraphAnalysis::new(&first),
            [],
        )
        .unwrap();
    facts
        .add_body(
            BodySite::Entry(1),
            &crate::egir::analysis::GraphAnalysis::new(&second),
            [],
        )
        .unwrap();
    let graph = SemanticGraph::from_facts(facts);
    for index in 0..2 {
        let source = SourceValue {
            body: BodySite::Entry(index),
            value,
        };
        assert_eq!(
            graph.capture_consumers(source).collect::<Vec<_>>(),
            vec![op(index as u32)]
        );
    }
}

#[test]
fn loop_carried_flow_is_visible_to_residency_without_a_contraction_cycle() {
    let source = r#"
entry repeated(xs: [4]i32) [4]i32 =
  let seed = map(|x: i32| x + 1, xs) in
  loop values = seed for i < 3 do map(|x: i32| x * 2, values)
"#;
    let tlc = crate::tlc::infer_input_slice_bounds(crate::compile_thru_tlc(source).unwrap());
    let program = egir::reify_soacs(crate::to_egraph(tlc).unwrap());
    let mut facts = Facts::new();
    facts
        .add_body(
            BodySite::Entry(0),
            &crate::egir::analysis::GraphAnalysis::new(&program.entry_points[0].graph),
            [],
        )
        .unwrap();
    let groups = facts.operations.keys().copied().collect::<Vec<_>>();
    let graph = facts.builder.clone().finish(groups.iter().map(|id| (*id, ()))).unwrap();
    assert!(graph.order().is_ok());
    let maps = facts.operations.values().filter_map(|op| op.semantic_id).collect::<Vec<_>>();
    assert_eq!(maps.len(), 2);
    let index = SemanticGraph::from_facts(facts);
    assert_eq!(index.value_consumers(&maps[0]).collect::<Vec<_>>(), vec![maps[1]]);
    assert_eq!(index.value_consumer_count(&maps[1]), 0);
}

#[test]
fn result_ports_preserve_occurrence_order_and_direct_precedence() {
    let scalar = Type::Constructed(TypeName::UInt(32), vec![]);
    let pair = Type::Constructed(TypeName::Tuple(2), vec![scalar.clone(), scalar.clone()]);
    let mut graph = EGraph::<Semantic>::new();
    let values = (0..4).map(|_| graph.alloc_side_effect_result(scalar.clone())).collect::<Vec<_>>();
    let first = ResultBinding::product(
        pair.clone(),
        values[..2].iter().map(|value| graph.value_result(*value)),
    );
    let second = ResultBinding::product(
        pair.clone(),
        values[2..].iter().map(|value| graph.value_result(*value)),
    );
    let packed = graph.intern_pure(
        PureOp::Tuple(2),
        smallvec![values[0], values[1]],
        pair.clone(),
        None,
    );
    graph.register_result_origin(packed, second.clone());
    graph.register_result_origin(packed, first.clone());
    let aliased = graph.alloc_side_effect_result(pair.clone());
    let direct = graph.alloc_side_effect_result(pair.clone());
    graph.install_aliases([(aliased, packed), (direct, packed)]);
    let result = ResultBinding::product(
        Type::Constructed(TypeName::Tuple(4), vec![pair; 4]),
        [first.clone(), second, first, graph.value_result(direct)],
    );
    let site = SideEffectSite {
        block: graph.skeleton.entry,
        index: 0,
    };
    graph.skeleton.blocks[site.block].side_effects.push(SideEffect {
        kind: SideEffectKind::Effect(EffectOp::Op {
            tag: PureOp::Materialize,
        }),
        operands: smallvec![],
        result: Some(result),
        effects: None,
        span: None,
    });

    let scope = (BodySite::Entry(0), site.block);
    let mut facts = Facts::new();
    facts
        .add_body(
            scope.0,
            &GraphAnalysis::new(&graph),
            [packed, aliased, direct, values[0], values[1]].map(|value| (site.block, value)),
        )
        .unwrap();
    let group = *facts.operations.keys().next().unwrap();
    let outputs = facts.builder.outputs(group).unwrap();
    assert_eq!(facts.ports[&(scope, packed)], outputs[0]);
    assert_eq!(facts.ports[&(scope, aliased)], outputs[0]);
    assert_eq!(facts.ports[&(scope, direct)], outputs[3]);
    for (index, value) in values[..2].iter().enumerate() {
        let port = facts.ports[&(scope, *value)];
        let Incidence::Project { base, path } = &facts.values[&port].incidence else {
            panic!("nested return must project from its producer");
        };
        assert_eq!(*base, outputs[2]);
        assert_eq!(path, &[index]);
    }
}
