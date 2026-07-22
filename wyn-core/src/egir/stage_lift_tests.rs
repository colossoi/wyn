use super::*;

use crate::ast::{Span, TypeName};
use crate::egir::program::{RegionInterner, SemanticEntry, SemanticOpId};
use crate::egir::soac::screma;
use crate::egir::types::{
    SegExtent, SegSpace, Semantic, SideEffect, SkeletonTerminator, Soac, SoacDestination, SoacEffect,
    SoacInputType,
};
use crate::flow::ExecutionModel;
use crate::interface::{BindingExposure, EntryInput, EntryInputKind, StorageAccess};
use crate::pipeline_descriptor::PipelineDescriptor;
use crate::ssa::types::ConstantValue;
use crate::{BindingRef, LookupMap};
use polytype::Type;
use smallvec::smallvec;

use super::super::stage_variance::{DependenceSource, Uniformity};

fn u32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::UInt(32), vec![])
}

fn semantic_function(name: &str, graph: EGraph<Semantic>, parameter_count: usize) -> SemanticFunc {
    let ty = u32_ty();
    SemanticFunc::new(
        name.into(),
        Span::dummy(),
        None,
        (0..parameter_count).map(|index| (ty.clone(), format!("p{index}"))).collect(),
        ty,
        graph,
        LookupMap::new(),
    )
}

fn mixed_callee() -> SemanticFunc {
    let ty = u32_ty();
    let mut graph = EGraph::<Semantic>::new();
    let lane = graph.add_func_param(0, ty.clone());
    let invariant = graph.add_func_param(1, ty.clone());
    let square = graph.intern_pure(
        PureOp::BinOp("*".into()),
        smallvec![invariant, invariant],
        ty.clone(),
        None,
    );
    let result = graph.intern_pure(PureOp::BinOp("+".into()), smallvec![lane, square], ty, None);
    graph.skeleton.blocks[graph.skeleton.entry].term = SkeletonTerminator::Return(Some(result));
    semantic_function("mixed", graph, 2)
}

fn calling_body() -> SemanticFunc {
    let ty = u32_ty();
    let mut graph = EGraph::<Semantic>::new();
    let lane = graph.add_func_param(0, ty.clone());
    let invariant = graph.add_func_param(1, ty.clone());
    let result = graph.intern_pure(PureOp::Call("mixed".into()), smallvec![lane, invariant], ty, None);
    graph.skeleton.blocks[graph.skeleton.entry].term = SkeletonTerminator::Return(Some(result));
    semantic_function("map_body", graph, 2)
}

fn enclosing_uniform(graph: &mut EGraph<Semantic>) -> NodeId {
    graph.add_func_param(0, u32_ty())
}

fn analyze_enclosing(graph: &EGraph<Semantic>) -> StageDependenceAnalysis {
    StageDependenceAnalysis::for_graph(
        graph,
        &LookupMap::new(),
        &[StageDependence::from_source(
            Uniformity::StageUniform,
            DependenceSource::Uniform,
        )],
    )
    .unwrap()
}

fn empty_program(functions: Vec<SemanticFunc>) -> SemanticProgram {
    SemanticProgram::new(
        functions,
        vec![],
        vec![],
        vec![],
        PipelineDescriptor::default(),
        RegionInterner::default(),
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
    let mut program = empty_program(vec![mixed_callee(), calling_body()]);
    let mut enclosing = EGraph::<Semantic>::new();
    let capture = enclosing_uniform(&mut enclosing);
    let body = SegBody {
        region: program.region_interner.get("map_body").unwrap(),
        captures: vec![capture],
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
    let lifted = specialized_body.captures[0];
    assert!(matches!(
        &enclosing.nodes[lifted],
        ENode::Pure {
            op: PureOp::BinOp(name),
            ..
        } if name == "*"
    ));
    assert!(
        !graph_ops::reachable_execution_values(&specialized.graph).into_iter().any(|node| {
            match &specialized.graph.nodes[node] {
                ENode::Pure {
                    op: PureOp::Call(_), ..
                } => true,
                ENode::Pure {
                    op: PureOp::BinOp(name),
                    ..
                } => name == "*",
                _ => false,
            }
        })
    );
    assert!(enclosing.verify_hash_cons().is_ok());
    assert!(specialized.graph.verify_hash_cons().is_ok());

    specialized.name = "map_body_stage_lift".into();
    specialized_body.region = program.define_region(specialized);
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
    let lane = graph.add_func_param(0, ty.clone());
    let invariant = graph.add_func_param(1, ty.clone());
    let one = graph.intern_constant(ConstantValue::U32(1), ty.clone());
    let square = graph.intern_pure(
        PureOp::BinOp("*".into()),
        smallvec![invariant, invariant],
        ty.clone(),
        None,
    );
    let increment = graph.intern_pure(
        PureOp::BinOp("+".into()),
        smallvec![invariant, one],
        ty.clone(),
        None,
    );
    let varying = graph.intern_pure(
        PureOp::BinOp("+".into()),
        smallvec![lane, square],
        ty.clone(),
        None,
    );
    let result = graph.intern_pure(PureOp::BinOp("*".into()), smallvec![varying, increment], ty, None);
    graph.skeleton.blocks[graph.skeleton.entry].term = SkeletonTerminator::Return(Some(result));

    let callee = semantic_function("multi_mixed", graph, 2);
    let mut body_graph = EGraph::<Semantic>::new();
    let lane = body_graph.add_func_param(0, u32_ty());
    let invariant = body_graph.add_func_param(1, u32_ty());
    let result = body_graph.intern_pure(
        PureOp::Call("multi_mixed".into()),
        smallvec![lane, invariant],
        u32_ty(),
        None,
    );
    body_graph.skeleton.blocks[body_graph.skeleton.entry].term = SkeletonTerminator::Return(Some(result));
    let mut program = empty_program(vec![callee, semantic_function("map_body", body_graph, 2)]);
    let mut enclosing = EGraph::<Semantic>::new();
    let capture = enclosing_uniform(&mut enclosing);
    let body = SegBody {
        region: program.region_interner.get("map_body").unwrap(),
        captures: vec![capture],
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
        &enclosing.nodes[specialized_body.captures[0]],
        ENode::Pure {
            op: PureOp::Tuple(2),
            operands,
        } if operands.len() == 2
    ));
    assert_eq!(
        graph_ops::reachable_execution_values(&specialized.graph)
            .into_iter()
            .filter(|node| matches!(
                specialized.graph.nodes[*node],
                ENode::Pure {
                    op: PureOp::Project { .. },
                    ..
                }
            ))
            .count(),
        2
    );

    specialized.name = "map_body_stage_lift".into();
    specialized_body.region = program.define_region(specialized);
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
    let input = entry_graph.add_func_param(0, input_ty.clone());
    let camera = entry_graph.add_func_param(1, element_ty.clone());
    let result = entry_graph.alloc_side_effect_result(result_ty.clone());

    let mut program = empty_program(vec![mixed_callee(), calling_body()]);
    let original_region = program.region_interner.get("map_body").unwrap();
    let effect = SideEffect {
        kind: SideEffectKind::Soac(SoacEffect(
            SemanticOpId::for_test(0),
            Soac::Screma(screma::Op::Map {
                lanes: screma::Lanes {
                    inputs: vec![SoacInputType {
                        array: input_ty.clone(),
                    }],
                    maps: vec![screma::Map {
                        body: SegBody {
                            region: original_region,
                            captures: vec![camera],
                        },
                        input_indices: vec![screma::InputId(0)],
                        output_element_type: element_ty.clone(),
                        destination: SoacDestination::fresh(),
                        result_type: result_ty.clone(),
                    }],
                },
                state: screma::SemanticState::Segmented {
                    space: SegSpace {
                        dims: vec![SegExtent::Fixed(64)],
                    },
                    placement: screma::Placement::Kernel,
                    output_slots: vec![],
                    resources: vec![],
                },
            }),
        )),
        operand_nodes: smallvec![input],
        result: Some(result),
        effects: None,
        span: None,
    };
    let block = entry_graph.skeleton.entry;
    entry_graph.skeleton.blocks[block].side_effects.push(effect);
    entry_graph.skeleton.blocks[block].term = SkeletonTerminator::Return(Some(result));
    let entry = SemanticEntry::new_with_resources(
        "compute".into(),
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
        vec![(input_ty, "points".into()), (element_ty, "frame".into())],
        result_ty,
        entry_graph,
        LookupMap::new(),
    );
    program.entry_points.push(entry);

    let stats = run(&mut program).unwrap();
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
    let body = &op.lanes().maps[0].body;
    assert_ne!(body.region, original_region);
    assert_eq!(body.captures.len(), 1);
    assert!(matches!(
        &entry.graph.nodes[body.captures[0]],
        ENode::Pure {
            op: PureOp::BinOp(name),
            ..
        } if name == "*"
    ));
    assert_eq!(program.region(body.region).unwrap().params.len(), 2);
    assert!(super::super::semantic_graph::verify(&program).is_ok());
}
