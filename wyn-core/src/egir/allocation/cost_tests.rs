use super::*;
use crate::ast::{Span, TypeName};
use crate::egir::analysis::GraphAnalysis;
use crate::egir::program::SemanticResourceRef;
use crate::egir::stage_variance::StageDependenceAnalysis;
use crate::egir::types::{by_value_function_result, callable_parameter, Parameters, PureOp, WynLanguage};
use crate::flow::ExecutionModel;
use crate::interface::{BindingExposure, EntryInput, IoDecoration};
use crate::op;
use crate::BindingRef;
use crate::EntryId;
use polytype::Type;
use smallvec::smallvec;

fn i32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Int(32), vec![])
}

fn u32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::UInt(32), vec![])
}

#[test]
fn stage_invariance_and_scalar_relocation_legality_remain_separate() {
    let ty = u32_ty();
    let parameters = ["uniform", "read_only", "read_write", "dispatch_size"]
        .into_iter()
        .map(|name| callable_parameter::<SemanticResourceRef, WynLanguage>(name.into(), ty.clone()))
        .collect::<Parameters<_, _>>();
    let mut graph = EGraph::new();
    let params = parameters
        .ids()
        .map(|parameter| graph.add_test_value_parameter(parameter, ty.clone()))
        .collect::<Vec<_>>();
    graph.skeleton.blocks[graph.skeleton.entry].term =
        SkeletonTerminator::Return(Some(graph.value_result(params[0])));
    let inputs = vec![
        EntryInput {
            name: "uniform".into(),
            ty: ty.clone(),
            size_hint: None,
            kind: EntryInputKind::Uniform {
                binding: BindingRef::new(1, 0),
            },
        },
        EntryInput {
            name: "read_only".into(),
            ty: ty.clone(),
            size_hint: None,
            kind: EntryInputKind::Storage {
                exposure: BindingExposure::Host(BindingRef::new(1, 1)),
                access: StorageAccess::ReadOnly,
                length: None,
            },
        },
        EntryInput {
            name: "read_write".into(),
            ty: ty.clone(),
            size_hint: None,
            kind: EntryInputKind::Storage {
                exposure: BindingExposure::Host(BindingRef::new(1, 2)),
                access: StorageAccess::ReadWrite,
                length: None,
            },
        },
        EntryInput {
            name: "dispatch_size".into(),
            ty: ty.clone(),
            size_hint: None,
            kind: EntryInputKind::Value {
                decoration: Some(IoDecoration::BuiltIn(spirv::BuiltIn::NumWorkgroups)),
            },
        },
    ];
    let entry = AllocatedEntry::new_with_resources(
        "compute".into(),
        EntryId::from_index(0),
        Span::generated(),
        ExecutionModel::Compute {
            local_size: (1, 1, 1),
        },
        inputs,
        vec![],
        vec![],
        parameters,
        by_value_function_result::<WynLanguage>(ty),
        graph,
    );

    let dependence = StageDependenceAnalysis::for_entry(&entry, &GraphAnalysis::new(&entry.graph)).unwrap();
    assert!(params.iter().all(|parameter| dependence.dependence(*parameter).is_stage_invariant()));
    assert!(entry_parameter_is_scalar_relocatable(&entry, 0));
    assert!(entry_parameter_is_scalar_relocatable(&entry, 1));
    assert!(!entry_parameter_is_scalar_relocatable(&entry, 2));
    assert!(!entry_parameter_is_scalar_relocatable(&entry, 3));
}

#[test]
fn profitability_includes_launch_loads_and_margin() {
    assert!(!materialization_is_profitable(1, 128, 1));
    assert!(materialization_is_profitable(256, 64, 1));

    let cost = 20;
    let invocations = 64;
    let recompute = cost * invocations;
    let handoff = SINGLETON_LAUNCH_COST + cost + STORAGE_LOAD_COST * invocations;
    assert_eq!(
        materialization_is_profitable(cost, invocations, 1),
        4 * recompute >= 5 * handoff
    );

    let two_output_handoff = SINGLETON_LAUNCH_COST + cost + 2 * STORAGE_LOAD_COST * invocations;
    assert_eq!(
        materialization_is_profitable(cost, invocations, 2),
        4 * recompute >= 5 * two_output_handoff
    );
}

#[test]
fn canonical_fixed_range_loop_recovers_trip_count() {
    let mut graph = EGraph::new();
    let entry = graph.skeleton.entry;
    let header = graph.skeleton.create_block();
    let body = graph.skeleton.create_block();
    let merge = graph.skeleton.create_block();
    let zero = graph.intern_pure(PureOp::Int("0".into()), smallvec![], i32_ty(), None);
    let one = graph.intern_pure(PureOp::Int("1".into()), smallvec![], i32_ty(), None);
    let bound = graph.intern_pure(PureOp::Int("32".into()), smallvec![], i32_ty(), None);
    let index = graph.add_block_param(header, i32_ty());
    let zero_args = graph.admit_flow_values([zero]);
    graph.skeleton.blocks[entry].term = SkeletonTerminator::Branch {
        target: header,
        args: zero_args,
    };
    let cond = graph.intern_pure(
        PureOp::BinOp(op::BinaryOperator::Less),
        smallvec![index, bound],
        Type::Constructed(TypeName::Bool, vec![]),
        None,
    );
    graph.skeleton.blocks[header].term = SkeletonTerminator::CondBranch {
        cond,
        then_target: body,
        then_args: vec![],
        else_target: merge,
        else_args: vec![],
    };
    let next = graph.intern_pure(
        PureOp::BinOp(op::BinaryOperator::Add),
        smallvec![index, one],
        i32_ty(),
        None,
    );
    let next_args = graph.admit_flow_values([next]);
    graph.skeleton.blocks[body].term = SkeletonTerminator::Branch {
        target: header,
        args: next_args,
    };
    graph.skeleton.blocks[merge].term = SkeletonTerminator::Return(None);

    assert_eq!(fixed_loop_trip_count(&graph, header, body, merge), Some(32));
}
