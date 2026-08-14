use super::*;

use crate::ast::{Span, TypeName};
use crate::egir::program::{
    semantic_program_for_test, ProgramIdentities, SemanticFunc, SemanticResourceRef,
};
use crate::egir::types::{
    by_value_function_result, callable_parameter, CallEffects, FuncParam, OperandRef, Semantic,
    SkeletonTerminator, WynLanguage,
};
use crate::flow::ExecutionModel;
use crate::interface::{EntryInput, IoDecoration};
use crate::pipeline_descriptor::PipelineDescriptor;
use crate::ssa::types::ConstantValue;
use crate::{BindingRef, EntryId, FunctionId};
use polytype::Type;
use smallvec::smallvec;

fn u32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::UInt(32), vec![])
}

fn semantic_params(
    specs: impl IntoIterator<Item = (&'static str, Type<TypeName>)>,
) -> Vec<FuncParam<SemanticResourceRef, Type<TypeName>>> {
    specs
        .into_iter()
        .map(|(name, ty)| callable_parameter::<SemanticResourceRef, WynLanguage>(name.into(), ty))
        .collect()
}

#[test]
fn entry_uniforms_seed_invariance_and_calls_report_mixed_arguments() {
    let ty = u32_ty();
    let project = FunctionId::from_index(0);
    let mut graph = EGraph::<Semantic>::new();
    let uniform = graph.add_test_value_parameter(0, ty.clone());
    let stage_input = graph.add_test_value_parameter(1, ty.clone());
    let one = graph.intern_constant(ConstantValue::U32(1), ty.clone());
    let uniform_sum = graph.intern_pure(
        PureOp::BinOp(crate::op::BinaryOperator::Add),
        smallvec![uniform, one],
        ty.clone(),
        None,
    );
    let project_params = semantic_params([
        ("value", ty.clone()),
        ("offset", ty.clone()),
    ]);
    let call = graph
        .add_call(
            project,
            &project_params,
            &by_value_function_result::<WynLanguage>(ty.clone()),
            [OperandRef::Value(stage_input), OperandRef::Value(uniform_sum)],
            CallEffects::Pure,
            None,
        )
        .unwrap()
        .1
        .single_value()
        .unwrap();
    graph.skeleton.blocks[graph.skeleton.entry].term =
        SkeletonTerminator::Return(Some(graph.value_result(call)));

    let inputs = vec![
        EntryInput {
            name: "frame".into(),
            ty: ty.clone(),
            size_hint: None,
            kind: EntryInputKind::Uniform {
                binding: BindingRef::new(1, 0),
            },
        },
        EntryInput {
            name: "position".into(),
            ty: ty.clone(),
            size_hint: None,
            kind: EntryInputKind::Value {
                decoration: Some(IoDecoration::Location(0)),
            },
        },
    ];
    let entry = SemanticEntry::new_with_resources(
        "vertex".into(),
        EntryId::from_index(0),
        Span::dummy(),
        ExecutionModel::Vertex,
        inputs,
        vec![],
        vec![],
        semantic_params([
            ("resolved_uniform", ty.clone()),
            ("resolved_position", ty.clone()),
        ]),
        by_value_function_result::<WynLanguage>(ty),
        graph,
    );

    let analysis = StageDependenceAnalysis::for_entry(&entry).unwrap();
    assert_eq!(
        analysis.dependence(uniform).uniformity(),
        Uniformity::StageUniform
    );
    assert!(analysis.dependence(uniform).depends_on(DependenceSource::Uniform));
    assert_eq!(
        analysis.dependence(uniform_sum).uniformity(),
        Uniformity::StageUniform
    );
    assert_eq!(
        analysis.dependence(stage_input).uniformity(),
        Uniformity::InvocationVarying
    );
    assert_eq!(
        analysis.dependence(call).uniformity(),
        Uniformity::InvocationVarying
    );

    let call_variance = analysis.call_arguments(&entry.graph, call).unwrap();
    assert_eq!(call_variance.callee, project);
    assert_eq!(
        call_variance.arguments.as_slice(),
        &[
            (stage_input, analysis.dependence(stage_input)),
            (uniform_sum, analysis.dependence(uniform_sum)),
        ]
    );
    assert!(call_variance.has_mixed_stage_variance());
}

#[test]
fn block_parameters_include_incoming_control_variance() {
    let ty = u32_ty();
    let bool_ty = Type::Constructed(TypeName::Bool, vec![]);
    let mut graph = EGraph::<Semantic>::new();
    let varying_condition = graph.add_test_value_parameter(0, bool_ty);
    let entry = graph.skeleton.entry;
    let then_block = graph.skeleton.create_block();
    let else_block = graph.skeleton.create_block();
    let merge = graph.skeleton.create_block();
    let one = graph.intern_constant(ConstantValue::U32(1), ty.clone());
    let two = graph.intern_constant(ConstantValue::U32(2), ty.clone());
    graph.skeleton.blocks[entry].term = SkeletonTerminator::CondBranch {
        cond: varying_condition,
        then_target: then_block,
        then_args: vec![],
        else_target: else_block,
        else_args: vec![],
    };
    let then_args = graph.admit_flow_values([one]);
    graph.skeleton.blocks[then_block].term = SkeletonTerminator::Branch {
        target: merge,
        args: then_args,
    };
    let else_args = graph.admit_flow_values([two]);
    graph.skeleton.blocks[else_block].term = SkeletonTerminator::Branch {
        target: merge,
        args: else_args,
    };
    let selected = graph.add_block_param(merge, ty);
    graph.skeleton.blocks[merge].term = SkeletonTerminator::Return(Some(graph.value_result(selected)));

    let analysis = StageDependenceAnalysis::for_graph(
        &graph,
        &[StageDependence::from_source(
            Uniformity::InvocationVarying,
            DependenceSource::StageInput,
        )],
    )
    .unwrap();
    assert!(analysis.dependence(one).is_compile_time_constant());
    assert!(analysis.dependence(two).is_compile_time_constant());
    assert_eq!(
        analysis.dependence(selected).uniformity(),
        Uniformity::InvocationVarying
    );
}

#[test]
fn invariant_loop_carried_values_converge_through_the_cfg_cycle() {
    let ty = u32_ty();
    let bool_ty = Type::Constructed(TypeName::Bool, vec![]);
    let mut graph = EGraph::<Semantic>::new();
    let entry = graph.skeleton.entry;
    let header = graph.skeleton.create_block();
    let exit = graph.skeleton.create_block();
    let zero = graph.intern_constant(ConstantValue::U32(0), ty.clone());
    let one = graph.intern_constant(ConstantValue::U32(1), ty.clone());
    let condition = graph.intern_constant(ConstantValue::Bool(true), bool_ty);
    let initial_args = graph.admit_flow_values([zero]);
    graph.skeleton.blocks[entry].term = SkeletonTerminator::Branch {
        target: header,
        args: initial_args,
    };
    let current = graph.add_block_param(header, ty.clone());
    let next = graph.intern_pure(
        PureOp::BinOp(crate::op::BinaryOperator::Add),
        smallvec![current, one],
        ty.clone(),
        None,
    );
    let next_args = graph.admit_flow_values([next]);
    let current_args = graph.admit_flow_values([current]);
    graph.skeleton.blocks[header].term = SkeletonTerminator::CondBranch {
        cond: condition,
        then_target: header,
        then_args: next_args,
        else_target: exit,
        else_args: current_args,
    };
    let result = graph.add_block_param(exit, ty);
    graph.skeleton.blocks[exit].term = SkeletonTerminator::Return(Some(graph.value_result(result)));

    graph.skeleton.blocks[header].control_header = Some(crate::flow::ControlHeader::Loop {
        merge: exit,
        continue_block: header,
    });
    let analysis = StageDependenceAnalysis::for_graph(&graph, &[]).unwrap();
    for value in [current, next, result] {
        assert!(analysis.dependence(value).is_stage_invariant());
        assert!(!analysis.dependence(value).is_loop_invariant(header));
        assert!(analysis.dependence(value).loop_dependencies().contains(&header));
    }
}

#[test]
fn storage_provenance_is_independent_of_index_uniformity() {
    let ty = u32_ty();
    let array_ty = Type::Constructed(TypeName::Array, vec![]);
    let mut graph = EGraph::<Semantic>::new();
    let storage = graph.add_test_value_parameter(0, array_ty);
    let varying_index = graph.add_test_value_parameter(1, ty.clone());
    let zero = graph.intern_constant(ConstantValue::U32(0), ty.clone());
    let uniform_load = graph.intern_pure(PureOp::Index, smallvec![storage, zero], ty.clone(), None);
    let varying_load = graph.intern_pure(PureOp::Index, smallvec![storage, varying_index], ty, None);
    let analysis = StageDependenceAnalysis::for_graph(
        &graph,
        &[
            StageDependence::from_source(Uniformity::StageUniform, DependenceSource::Storage),
            StageDependence::from_source(Uniformity::InvocationVarying, DependenceSource::StageInput),
        ],
    )
    .unwrap();

    let uniform_facts = analysis.dependence(uniform_load);
    assert_eq!(uniform_facts.uniformity(), Uniformity::StageUniform);
    assert_eq!(
        uniform_facts.sources(),
        &[DependenceSource::Storage].into_iter().collect()
    );

    let varying_facts = analysis.dependence(varying_load);
    assert_eq!(varying_facts.uniformity(), Uniformity::InvocationVarying);
    assert!(varying_facts.depends_on(DependenceSource::Storage));
    assert!(varying_facts.depends_on(DependenceSource::StageInput));
}

#[test]
fn invocation_intrinsics_are_varying_without_operands() {
    let ty = u32_ty();
    let mut graph = EGraph::<Semantic>::new();
    let known = catalog().known();
    let thread_id = graph.intern_pure(
        PureOp::Intrinsic {
            id: known.thread_id,
            overload_idx: 0,
        },
        smallvec![],
        ty.clone(),
        None,
    );
    let num_workgroups = graph.intern_pure(
        PureOp::Intrinsic {
            id: known.num_workgroups,
            overload_idx: 0,
        },
        smallvec![],
        ty,
        None,
    );

    let analysis = StageDependenceAnalysis::for_graph(&graph, &[]).unwrap();
    assert_eq!(
        analysis.dependence(thread_id).uniformity(),
        Uniformity::InvocationVarying
    );
    assert!(analysis.dependence(thread_id).depends_on(DependenceSource::InvocationBuiltin));
    assert_eq!(
        analysis.dependence(num_workgroups).uniformity(),
        Uniformity::StageUniform
    );
    assert!(analysis.dependence(num_workgroups).depends_on(DependenceSource::DispatchBuiltin));
}

#[test]
fn repeated_region_captures_are_analyzed_per_use() {
    let ty = u32_ty();
    let mut region_graph = EGraph::<Semantic>::new();
    let lane_value = region_graph.add_test_value_parameter(0, ty.clone());
    let capture = region_graph.add_test_value_parameter(1, ty.clone());
    let result = region_graph.intern_pure(
        PureOp::BinOp(crate::op::BinaryOperator::Add),
        smallvec![lane_value, capture],
        ty.clone(),
        None,
    );
    region_graph.skeleton.blocks[region_graph.skeleton.entry].term =
        SkeletonTerminator::Return(Some(region_graph.value_result(result)));
    let mut identities = ProgramIdentities::default();
    let region_id: FunctionId = identities.alloc_function("map_body".into());
    let region = SemanticFunc::new(
        region_id,
        "map_body".into(),
        Span::dummy(),
        None,
        semantic_params([("lane", ty.clone()), ("capture", ty.clone())]),
        by_value_function_result::<WynLanguage>(ty.clone()),
        CallEffects::Pure,
        region_graph,
    );
    let program = semantic_program_for_test(
        vec![region],
        vec![],
        vec![],
        vec![],
        PipelineDescriptor::default(),
        identities,
    );

    let mut enclosing_graph = EGraph::<Semantic>::new();
    let invariant_capture = enclosing_graph.intern_constant(ConstantValue::U32(7), ty.clone());
    let varying_capture = enclosing_graph.add_test_value_parameter(0, ty);
    let enclosing = StageDependenceAnalysis::for_graph(
        &enclosing_graph,
        &[StageDependence::from_source(
            Uniformity::InvocationVarying,
            DependenceSource::StageInput,
        )],
    )
    .unwrap();

    let invariant_use = StageDependenceAnalysis::for_seg_body(
        &program,
        &enclosing,
        &SegBody {
            region: region_id,
            captures: vec![OperandRef::Value(invariant_capture)],
        },
    )
    .unwrap();
    let varying_use = StageDependenceAnalysis::for_seg_body(
        &program,
        &enclosing,
        &SegBody {
            region: region_id,
            captures: vec![OperandRef::Value(varying_capture)],
        },
    )
    .unwrap();

    assert_eq!(
        invariant_use.dependence(lane_value).uniformity(),
        Uniformity::InvocationVarying
    );
    assert!(invariant_use.dependence(lane_value).depends_on(DependenceSource::RepeatedRegionInput));
    assert!(invariant_use.dependence(capture).is_compile_time_constant());
    assert_eq!(
        varying_use.dependence(capture).uniformity(),
        Uniformity::InvocationVarying
    );
}
