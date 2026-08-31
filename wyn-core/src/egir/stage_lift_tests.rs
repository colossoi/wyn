use super::*;
use crate::types;

use crate::ast::{Span, TypeName};
use crate::egir::program::{semantic_program_for_test, Entry, ProgramIdentities, SemanticOpId};
use crate::egir::reify::Segmented;
use crate::egir::soac::screma;
use crate::egir::types::{
    by_value_function_result, callable_parameter, CallEffects, OperandRef, Parameters, SegExtent, SegSpace,
    Semantic, SideEffect, SkeletonTerminator, Soac, SoacEffect, SoacInputType, SoacOwnership, WynLanguage,
};
use crate::flow::ExecutionModel;
use crate::interface::{BindingExposure, EntryInput, EntryInputKind, StorageAccess};
use crate::op::BinaryOperator;
use crate::pipeline_descriptor::PipelineDescriptor;
use crate::ssa::types::ConstantValue;
use crate::{BindingRef, FunctionId, LookupMap};
use polytype::Type;
use smallvec::smallvec;

use super::super::stage_variance::{DependenceSource, Uniformity};

fn u32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::UInt(32), vec![])
}

fn semantic_params(
    specs: impl IntoIterator<Item = (String, Type<TypeName>)>,
) -> Parameters<BindingRef, Type<TypeName>> {
    specs
        .into_iter()
        .map(|(name, ty)| callable_parameter::<BindingRef, WynLanguage>(name, ty))
        .collect()
}

fn semantic_function(
    id: FunctionId,
    name: &str,
    graph: EGraph<Semantic>,
    params: Parameters<BindingRef, Type<TypeName>>,
) -> Func {
    let ty = u32_ty();
    Func::<Semantic>::new(
        id,
        name.into(),
        Span::generated(),
        None,
        params,
        by_value_function_result::<WynLanguage>(ty),
        CallEffects::Pure,
        graph,
    )
}

fn mixed_callee(id: FunctionId) -> Func {
    let ty = u32_ty();
    let params = semantic_params([("lane".into(), ty.clone()), ("invariant".into(), ty.clone())]);
    let parameter_ids = params.ids().collect::<Vec<_>>();
    let mut graph = EGraph::<Semantic>::new();
    let lane = graph.add_test_value_parameter(parameter_ids[0], ty.clone());
    let invariant = graph.add_test_value_parameter(parameter_ids[1], ty.clone());
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
    semantic_function(id, "mixed", graph, params)
}

fn calling_body(id: FunctionId, mixed: &Func) -> Func {
    let ty = u32_ty();
    let params = semantic_params([("lane".into(), ty.clone()), ("invariant".into(), ty.clone())]);
    let parameter_ids = params.ids().collect::<Vec<_>>();
    let mut graph = EGraph::<Semantic>::new();
    let lane = graph.add_test_value_parameter(parameter_ids[0], ty.clone());
    let invariant = graph.add_test_value_parameter(parameter_ids[1], ty.clone());
    let result = graph
        .emit_call(
            graph.skeleton.entry,
            mixed.region,
            mixed.params(),
            &by_value_function_result::<WynLanguage>(ty.clone()),
            [OperandRef::Value(lane), OperandRef::Value(invariant)],
            CallEffects::Pure,
            None,
            None,
        )
        .unwrap()
        .1
        .single_value()
        .unwrap();
    graph.skeleton.blocks[graph.skeleton.entry].term =
        SkeletonTerminator::Return(Some(graph.value_result(result)));
    semantic_function(id, "map_body", graph, params)
}

fn enclosing_uniform(graph: &mut EGraph<Semantic>) -> (ValueId, Parameters<BindingRef, Type<TypeName>>) {
    let params = semantic_params([("uniform".into(), u32_ty())]);
    let value = graph.add_test_value_parameter(params.ids().next().unwrap(), u32_ty());
    (value, params)
}

fn analyze_enclosing(
    graph: &EGraph<Semantic>,
    params: &Parameters<BindingRef, Type<TypeName>>,
) -> StageDependenceAnalysis {
    StageDependenceAnalysis::for_graph(
        graph,
        &params
            .ids()
            .zip([StageDependence::from_source(
                Uniformity::StageUniform,
                DependenceSource::Uniform,
            )])
            .collect::<LookupMap<_, _>>(),
    )
    .unwrap()
}

fn empty_program(functions: Vec<Func<Semantic>>, identities: ProgramIdentities) -> Segmented {
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
            types::no_buffer(),
        ],
    )
}

#[test]
fn mixed_stage_call_uses_generic_inlining_then_lifts_its_uniform_subgraph() {
    let mut identities = ProgramIdentities::default();
    let mixed = identities.alloc_function("mixed".into());
    let map_body = identities.alloc_function("map_body".into());
    let mixed_function = mixed_callee(mixed);
    let map_function = calling_body(map_body, &mixed_function);
    let mut program = empty_program(vec![mixed_function, map_function], identities);
    let mut enclosing = EGraph::<Semantic>::new();
    let (capture, enclosing_params) = enclosing_uniform(&mut enclosing);
    let body = SegBody {
        region: map_body,
        captures: vec![OperandRef::Value(capture)],
    };
    let enclosing_analysis = analyze_enclosing(&enclosing, &enclosing_params);

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
    let enclosing_analysis = analyze_enclosing(&enclosing, &enclosing_params);
    assert!(
        prepare_lift(&program, &enclosing_analysis, &specialized_body).unwrap().is_none(),
        "a captured uniform value must make the lift idempotent"
    );
}

#[test]
fn multiple_uniform_frontier_values_share_one_aggregate_capture() {
    let ty = u32_ty();
    let callee_params = semantic_params([("lane".into(), ty.clone()), ("invariant".into(), ty.clone())]);
    let callee_parameter_ids = callee_params.ids().collect::<Vec<_>>();
    let mut graph = EGraph::<Semantic>::new();
    let lane = graph.add_test_value_parameter(callee_parameter_ids[0], ty.clone());
    let invariant = graph.add_test_value_parameter(callee_parameter_ids[1], ty.clone());
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
    let callee = semantic_function(callee_id, "multi_mixed", graph, callee_params);
    let body_params = semantic_params([("lane".into(), u32_ty()), ("invariant".into(), u32_ty())]);
    let body_parameter_ids = body_params.ids().collect::<Vec<_>>();
    let mut body_graph = EGraph::<Semantic>::new();
    let lane = body_graph.add_test_value_parameter(body_parameter_ids[0], u32_ty());
    let invariant = body_graph.add_test_value_parameter(body_parameter_ids[1], u32_ty());
    let result = body_graph
        .emit_call(
            body_graph.skeleton.entry,
            callee_id,
            callee.params(),
            &by_value_function_result::<WynLanguage>(u32_ty()),
            [OperandRef::Value(lane), OperandRef::Value(invariant)],
            CallEffects::Pure,
            None,
            None,
        )
        .unwrap()
        .1
        .single_value()
        .unwrap();
    body_graph.skeleton.blocks[body_graph.skeleton.entry].term =
        SkeletonTerminator::Return(Some(body_graph.value_result(result)));
    let mut program = empty_program(
        vec![
            callee,
            semantic_function(map_body_id, "map_body", body_graph, body_params),
        ],
        identities,
    );
    let mut enclosing = EGraph::<Semantic>::new();
    let (capture, enclosing_params) = enclosing_uniform(&mut enclosing);
    let body = SegBody {
        region: map_body_id,
        captures: vec![OperandRef::Value(capture)],
    };
    let enclosing_analysis = analyze_enclosing(&enclosing, &enclosing_params);
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
    let enclosing_analysis = analyze_enclosing(&enclosing, &enclosing_params);
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
    let entry_params = semantic_params([
        ("points".into(), input_ty.clone()),
        ("frame".into(), element_ty.clone()),
    ]);
    let entry_parameter_ids = entry_params.ids().collect::<Vec<_>>();
    let mut entry_graph = EGraph::<Semantic>::new();
    let input = entry_graph.add_parameter(
        entry_parameter_ids[0],
        entry_params.get(entry_parameter_ids[0]).unwrap().representation(),
    );
    let input = input.value().unwrap();
    let camera = entry_graph.add_test_value_parameter(entry_parameter_ids[1], element_ty.clone());
    let result = entry_graph.alloc_side_effect_result(result_ty.clone());

    let mut identities = ProgramIdentities::default();
    let mixed = identities.alloc_function("mixed".into());
    let original_region = identities.alloc_function("map_body".into());
    let entry_id = identities.alloc_entry("compute".into());
    let mixed_function = mixed_callee(mixed);
    let map_function = calling_body(original_region, &mixed_function);
    let mut program = empty_program(vec![mixed_function, map_function], identities);
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
                    ownership: SoacOwnership::Fresh,
                }],
                state: screma::SemanticState::Segmented {
                    space: SegSpace::new(SegExtent::Fixed(64)),
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
    let entry = Entry::<Semantic>::new_with_resources(
        "compute".into(),
        entry_id,
        Span::generated(),
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
        entry_params,
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
