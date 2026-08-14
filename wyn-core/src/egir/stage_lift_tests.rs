use super::*;

use crate::ast::{Span, TypeName};
use crate::egir::program::{
    semantic_program_for_test, ProgramIdentities, SemanticEntry, SemanticOpId,
    SemanticResourceRef,
};
use crate::egir::reify::Segmented;
use crate::egir::soac::screma;
use crate::egir::types::{
    by_value_function_result, callable_parameter, CallEffects, FuncParam, OperandRef, ParameterId,
    SegExtent, SegSpace, Semantic, SideEffect, SkeletonTerminator, Soac, SoacDestination,
    SoacEffect, SoacInputType, WynLanguage,
};
use crate::flow::ExecutionModel;
use crate::interface::{BindingExposure, EntryInput, EntryInputKind, StorageAccess};
use crate::op::BinaryOperator;
use crate::pipeline_descriptor::PipelineDescriptor;
use crate::ssa::types::ConstantValue;
use crate::{BindingRef, FunctionId};
use polytype::Type;
use smallvec::smallvec;

use super::super::stage_variance::{DependenceSource, Uniformity};

fn u32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::UInt(32), vec![])
}

fn semantic_params(
    specs: impl IntoIterator<Item = (String, Type<TypeName>)>,
) -> Vec<FuncParam<SemanticResourceRef, Type<TypeName>>> {
    specs
        .into_iter()
        .map(|(name, ty)| callable_parameter::<SemanticResourceRef, WynLanguage>(name, ty))
        .collect()
}

fn semantic_function(
    id: FunctionId,
    name: &str,
    graph: EGraph<Semantic>,
    parameter_count: usize,
) -> SemanticFunc {
    let ty = u32_ty();
    SemanticFunc::new(
        id,
        name.into(),
        Span::dummy(),
        None,
        semantic_params((0..parameter_count).map(|index| (format!("p{index}"), ty.clone()))),
        by_value_function_result::<WynLanguage>(ty),
        CallEffects::Pure,
        graph,
    )
}

fn mixed_callee(id: FunctionId) -> SemanticFunc {
    let ty = u32_ty();
    let mut graph = EGraph::<Semantic>::new();
    let lane = graph.add_test_value_parameter(0, ty.clone());
    let invariant = graph.add_test_value_parameter(1, ty.clone());
    let square = graph.intern_pure(
        PureOp::BinOp(BinaryOperator::Multiply),
        smallvec![invariant, invariant],
        ty.clone(),
        None,
    );
    let result = graph.intern_pure(
        PureOp::BinOp(BinaryOperator::Add),
        smallvec![lane, square],
        ty,
        None,
    );
    graph.skeleton.blocks[graph.skeleton.entry].term =
        SkeletonTerminator::Return(Some(graph.value_result(result)));
    semantic_function(id, "mixed", graph, 2)
}

fn calling_body(id: FunctionId, mixed: FunctionId) -> SemanticFunc {
    let ty = u32_ty();
    let mut graph = EGraph::<Semantic>::new();
    let lane = graph.add_test_value_parameter(0, ty.clone());
    let invariant = graph.add_test_value_parameter(1, ty.clone());
    let params = semantic_params([
        ("lane".into(), ty.clone()),
        ("invariant".into(), ty.clone()),
    ]);
    let result = graph
        .add_call(
            mixed,
            &params,
            &by_value_function_result::<WynLanguage>(ty.clone()),
            [OperandRef::Value(lane), OperandRef::Value(invariant)],
            CallEffects::Pure,
            None,
        )
        .unwrap()
        .1
        .single_value()
        .unwrap();
    graph.skeleton.blocks[graph.skeleton.entry].term =
        SkeletonTerminator::Return(Some(graph.value_result(result)));
    semantic_function(id, "map_body", graph, 2)
}

fn enclosing_uniform(graph: &mut EGraph<Semantic>) -> ValueId {
    graph.add_test_value_parameter(0, u32_ty())
}

fn analyze_enclosing(graph: &EGraph<Semantic>) -> StageDependenceAnalysis {
    StageDependenceAnalysis::for_graph(
        graph,
        &[StageDependence::from_source(
            Uniformity::StageUniform,
            DependenceSource::Uniform,
        )],
    )
    .unwrap()
}

fn empty_program(functions: Vec<SemanticFunc>, identities: ProgramIdentities) -> Segmented {
    semantic_program_for_test(
        functions,
        vec![],
        vec![],
        vec![],
        PipelineDescriptor::default(),
        identities,
    )
}

fn array_ty(element: Type<TypeName>, variant: TypeName) -> Type<TypeName> {
    Type::Constructed(
        TypeName::Array,
        vec![
            element,
            Type::Constructed(variant, vec![]),
            Type::Constructed(TypeName::Size(64), vec![]),
            crate::types::no_buffer(),
        ],
    )
}

#[test]
fn mixed_stage_call_uses_generic_inlining_then_lifts_its_uniform_subgraph() {
    let mut identities = ProgramIdentities::default();
    let mixed = identities.alloc_function("mixed".into());
    let map_body = identities.alloc_function("map_body".into());
    let mut program = empty_program(
        vec![mixed_callee(mixed), calling_body(map_body, mixed)],
        identities,
    );
    let mut enclosing = EGraph::<Semantic>::new();
    let capture = enclosing_uniform(&mut enclosing);
    let body = SegBody {
        region: map_body,
        captures: vec![OperandRef::Value(capture)],
    };
    let enclosing_analysis = analyze_enclosing(&enclosing);

    let prepared = prepare_lift(&program, &enclosing_analysis, &body)
        .unwrap()
        .expect("mixed call exposes a liftable uniform square");
    assert_eq!(prepared.calls_inlined, 1);
    assert_eq!(prepared.frontier.len(), 1);

    let (mut specialized, mut specialized_body) = apply_lift(&mut enclosing, prepared).unwrap();
    assert_eq!(specialized.params.len(), 2);
    assert_eq!(specialized_body.captures.len(), 1);
    let lifted = specialized_body.captures[0].value().unwrap();
    assert!(matches!(
        &enclosing.nodes[lifted].kind,
        ValueKind::Pure {
            op: PureOp::BinOp(name),
            ..
        } if *name == BinaryOperator::Multiply
    ));
    assert!(
        !graph_ops::reachable_execution_values(&specialized.graph).into_iter().any(|node| {
            match &specialized.graph.nodes[node].kind {
                ValueKind::CallResult { .. } => true,
                ValueKind::Pure {
                    op: PureOp::BinOp(name),
                    ..
                } => *name == BinaryOperator::Multiply,
                _ => false,
            }
        })
    );
    assert!(enclosing.verify_hash_cons().is_ok());
    assert!(specialized.graph.verify_hash_cons().is_ok());

    specialized.name = "map_body_stage_lift".into();
    let specialized_region = program.data.identities.alloc_function(specialized.name.clone());
    specialized.region = specialized_region;
    specialized_body.region = specialized_region;
    program = program.extend_functions([specialized]);
    let enclosing_analysis = analyze_enclosing(&enclosing);
    assert!(
        prepare_lift(&program, &enclosing_analysis, &specialized_body).unwrap().is_none(),
        "a captured uniform value must make the lift idempotent"
    );
}

#[test]
fn multiple_uniform_frontier_values_share_one_aggregate_capture() {
    let ty = u32_ty();
    let mut graph = EGraph::<Semantic>::new();
    let lane = graph.add_test_value_parameter(0, ty.clone());
    let invariant = graph.add_test_value_parameter(1, ty.clone());
    let one = graph.intern_constant(ConstantValue::U32(1), ty.clone());
    let square = graph.intern_pure(
        PureOp::BinOp(BinaryOperator::Multiply),
        smallvec![invariant, invariant],
        ty.clone(),
        None,
    );
    let increment = graph.intern_pure(
        PureOp::BinOp(BinaryOperator::Add),
        smallvec![invariant, one],
        ty.clone(),
        None,
    );
    let varying = graph.intern_pure(
        PureOp::BinOp(BinaryOperator::Add),
        smallvec![lane, square],
        ty.clone(),
        None,
    );
    let result = graph.intern_pure(
        PureOp::BinOp(BinaryOperator::Multiply),
        smallvec![varying, increment],
        ty,
        None,
    );
    graph.skeleton.blocks[graph.skeleton.entry].term =
        SkeletonTerminator::Return(Some(graph.value_result(result)));

    let mut identities = ProgramIdentities::default();
    let callee_id = identities.alloc_function("multi_mixed".into());
    let map_body_id = identities.alloc_function("map_body".into());
    let callee = semantic_function(callee_id, "multi_mixed", graph, 2);
    let mut body_graph = EGraph::<Semantic>::new();
    let lane = body_graph.add_test_value_parameter(0, u32_ty());
    let invariant = body_graph.add_test_value_parameter(1, u32_ty());
    let params = semantic_params([
        ("lane".into(), u32_ty()),
        ("invariant".into(), u32_ty()),
    ]);
    let result = body_graph
        .add_call(
            callee_id,
            &params,
            &by_value_function_result::<WynLanguage>(u32_ty()),
            [OperandRef::Value(lane), OperandRef::Value(invariant)],
            CallEffects::Pure,
            None,
        )
        .unwrap()
        .1
        .single_value()
        .unwrap();
    body_graph.skeleton.blocks[body_graph.skeleton.entry].term =
        SkeletonTerminator::Return(Some(body_graph.value_result(result)));
    let mut program = empty_program(
        vec![callee, semantic_function(map_body_id, "map_body", body_graph, 2)],
        identities,
    );
    let mut enclosing = EGraph::<Semantic>::new();
    let capture = enclosing_uniform(&mut enclosing);
    let body = SegBody {
        region: map_body_id,
        captures: vec![OperandRef::Value(capture)],
    };
    let enclosing_analysis = analyze_enclosing(&enclosing);
    let prepared = prepare_lift(&program, &enclosing_analysis, &body)
        .unwrap()
        .expect("two uniform boundary values are liftable");
    assert_eq!(prepared.frontier.len(), 2);

    let (mut specialized, mut specialized_body) = apply_lift(&mut enclosing, prepared).unwrap();
    assert_eq!(specialized.params.len(), 2);
    assert_eq!(specialized_body.captures.len(), 1);
    assert!(matches!(
        &enclosing.nodes[specialized_body.captures[0].value().unwrap()].kind,
        ValueKind::Pure {
            op: PureOp::Tuple(2),
            operands,
        } if operands.len() == 2
    ));
    assert_eq!(
        graph_ops::reachable_execution_values(&specialized.graph)
            .into_iter()
            .filter(|node| matches!(
                specialized.graph.nodes[*node].kind,
                ValueKind::Pure {
                    op: PureOp::Project { .. },
                    ..
                }
            ))
            .count(),
        2
    );

    specialized.name = "map_body_stage_lift".into();
    let specialized_region = program.data.identities.alloc_function(specialized.name.clone());
    specialized.region = specialized_region;
    specialized_body.region = specialized_region;
    program = program.extend_functions([specialized]);
    let enclosing_analysis = analyze_enclosing(&enclosing);
    assert!(
        prepare_lift(&program, &enclosing_analysis, &specialized_body).unwrap().is_none(),
        "aggregate projections are the already-lifted boundary"
    );
}

#[test]
fn parallel_soac_use_is_specialized_and_captures_the_lifted_value() {
    let element_ty = u32_ty();
    let input_ty = array_ty(element_ty.clone(), TypeName::ArrayVariantView);
    let result_ty = array_ty(element_ty.clone(), TypeName::ArrayVariantComposite);
    let mut entry_graph = EGraph::<Semantic>::new();
    let input = entry_graph.add_parameter(
        ParameterId::new(0),
        semantic_params([("points".into(), input_ty.clone())])[0].representation(),
    );
    let input = input.value().unwrap();
    let camera = entry_graph.add_test_value_parameter(1, element_ty.clone());
    let result = entry_graph.alloc_side_effect_result(result_ty.clone());

    let mut identities = ProgramIdentities::default();
    let mixed = identities.alloc_function("mixed".into());
    let original_region = identities.alloc_function("map_body".into());
    let entry_id = identities.alloc_entry("compute".into());
    let mut program = empty_program(
        vec![mixed_callee(mixed), calling_body(original_region, mixed)],
        identities,
    );
    let effect = SideEffect {
        kind: SideEffectKind::Soac(SoacEffect(
            SemanticOpId::for_test(0),
            Soac::Screma(screma::Op {
                inputs: vec![SoacInputType::array(input_ty.clone())],
                form: screma::ScremaForm {
                    pre: screma::Lambda::region(
                        SegBody {
                            region: original_region,
                            captures: vec![OperandRef::Value(camera)],
                        },
                        vec![element_ty.clone()],
                        vec![element_ty.clone()],
                    ),
                    scans: vec![],
                    reductions: vec![],
                    post: screma::Lambda::identity(vec![element_ty.clone()]),
                },
                result_state: vec![screma::ResultState {
                    destination: SoacDestination::fresh(),
                }],
                state: screma::SemanticState::Segmented {
                    space: SegSpace::new(SegExtent::Fixed(64)),
                    placement: screma::Placement::Kernel,
                    output_slots: vec![],
                    resources: vec![],
                },
            }),
        )),
        operands: smallvec![entry_graph.operand_ref(input)],
        result: Some(entry_graph.value_result(result)),
        effects: None,
        span: None,
    };
    let block = entry_graph.skeleton.entry;
    entry_graph.skeleton.blocks[block].side_effects.push(effect);
    entry_graph.skeleton.blocks[block].term =
        SkeletonTerminator::Return(Some(entry_graph.value_result(result)));
    let entry = SemanticEntry::new_with_resources(
        "compute".into(),
        entry_id,
        Span::dummy(),
        ExecutionModel::Compute {
            local_size: (64, 1, 1),
        },
        vec![
            EntryInput {
                name: "points".into(),
                ty: input_ty.clone(),
                size_hint: None,
                kind: EntryInputKind::Storage {
                    exposure: BindingExposure::Host(BindingRef::new(0, 0)),
                    access: StorageAccess::ReadOnly,
                    length: None,
                },
            },
            EntryInput {
                name: "frame".into(),
                ty: element_ty.clone(),
                size_hint: None,
                kind: EntryInputKind::Uniform {
                    binding: BindingRef::new(1, 0),
                },
            },
        ],
        vec![],
        vec![],
        semantic_params([
            ("points".into(), input_ty),
            ("frame".into(), element_ty),
        ]),
        by_value_function_result::<WynLanguage>(result_ty),
        entry_graph,
    );
    program.entry_points.push(entry);

    let (program, stats) = run_with_stats(program).unwrap();
    assert_eq!(
        stats,
        StageLiftStats {
            bodies_specialized: 1,
            calls_inlined: 1,
            values_lifted: 1,
        }
    );
    let entry = &program.entry_points[0];
    let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) =
        &entry.graph.skeleton.blocks[entry.graph.skeleton.entry].side_effects[0].kind
    else {
        panic!("entry no longer contains its parallel map")
    };
    let body = op.form.pre.seg_body().expect("map pre-lambda region");
    assert_ne!(body.region, original_region);
    assert_eq!(body.captures.len(), 1);
    assert!(matches!(
        &entry.graph.nodes[body.captures[0].value().unwrap()].kind,
        ValueKind::Pure {
            op: PureOp::BinOp(name),
            ..
        } if *name == BinaryOperator::Multiply
    ));
    assert_eq!(program.region(body.region).unwrap().params.len(), 2);
    assert!(super::super::semantic_graph::verify(&program).is_ok());
}
