use super::*;

fn i32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Int(32), vec![])
}

fn u32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::UInt(32), vec![])
}

fn effect(result: NodeId) -> SideEffect {
    SideEffect {
        kind: SideEffectKind::Effect(EffectOp::ControlBarrier),
        operand_nodes: SmallVec::new(),
        result: Some(result),
        effects: None,
        span: None,
    }
}

#[derive(Clone, Debug)]
struct TestPhase;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum TestConst {
    FortyTwo,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct TestLanguage;

impl Language for TestLanguage {
    type Const = TestConst;
    type Ty = String;
}

impl EgirPhase for TestPhase {
    type Resource = ();
    type ResourceDecl = u16;
    type SoacId = ();
    type MapState = ();
    type ReduceState = ();
    type ScanState = ();
    type CompositeState = ();
    type FilterState = ();
    type HistState = ();
}

#[test]
fn soac_placement_preserves_logical_ownership() {
    let mut destination = SoacDestination::unique_input();
    destination.place(SoacPlacement::OutputView);

    assert_eq!(destination.ownership, SoacOwnership::UniqueInput);
    assert!(destination.is_output_view());
    assert!(!destination.is_unplaced_unique_input());
}

#[test]
fn graph_accepts_non_wyn_payloads() {
    let mut graph = super::super::ir::EGraph::<Semantic, TestLanguage>::new();
    let node = graph.intern_pure(PureOp::Unit, SmallVec::new(), "unit".to_string(), None);
    let constant = graph.intern_constant(TestConst::FortyTwo, "number".to_string());

    assert_eq!(graph.types[&node], "unit");
    assert!(matches!(
        graph.nodes[constant],
        super::super::ir::ENode::Constant(TestConst::FortyTwo)
    ));
}

#[test]
fn adding_block_params_registers_them_in_order() {
    let mut graph = super::super::ir::EGraph::<TestPhase, TestLanguage>::new();
    let block = graph.skeleton.create_block();

    let first = graph.add_block_param(block, "first".to_string());
    let second = graph.add_block_param(block, "second".to_string());

    assert_eq!(graph.skeleton.blocks[block].params, [first, second]);
    assert!(matches!(
        graph.nodes[first],
        super::super::ir::ENode::BlockParam { block: owner, index: 0 } if owner == block
    ));
    assert!(matches!(
        graph.nodes[second],
        super::super::ir::ENode::BlockParam { block: owner, index: 1 } if owner == block
    ));
    assert_eq!(graph.types[&first], "first");
    assert_eq!(graph.types[&second], "second");
}

#[test]
fn removing_block_param_slots_updates_incoming_edges_and_indices() {
    let mut graph = super::super::ir::EGraph::<TestPhase, TestLanguage>::new();
    let entry = graph.skeleton.entry;
    let branch_predecessor = graph.skeleton.create_block();
    let target = graph.skeleton.create_block();

    let first = graph.add_block_param(target, "first".to_string());
    let second = graph.add_block_param(target, "second".to_string());
    let third = graph.add_block_param(target, "third".to_string());

    let args = (0..9).map(|index| graph.add_func_param(index, format!("arg-{index}"))).collect::<Vec<_>>();
    graph.skeleton.blocks[entry].term = SkeletonTerminator::CondBranch {
        cond: args[0],
        then_target: target,
        then_args: vec![args[1], args[2], args[3]],
        else_target: target,
        else_args: vec![args[4], args[5], args[6]],
    };
    graph.skeleton.blocks[branch_predecessor].term = SkeletonTerminator::Branch {
        target,
        args: vec![args[6], args[7], args[8]],
    };

    let slots = [2, 0, 2].into_iter().collect::<crate::SortedSet<_>>();
    let removed = graph.remove_block_param_slots(target, &slots);

    assert_eq!(removed, [first, third]);
    assert_eq!(graph.skeleton.blocks[target].params, [second]);
    assert!(matches!(
        graph.nodes[second],
        super::super::ir::ENode::BlockParam { block, index: 0 } if block == target
    ));
    assert!(graph.nodes.contains_key(first));
    assert!(graph.nodes.contains_key(third));
    match &graph.skeleton.blocks[entry].term {
        SkeletonTerminator::CondBranch {
            then_args, else_args, ..
        } => {
            assert_eq!(then_args, &[args[2]]);
            assert_eq!(else_args, &[args[5]]);
        }
        other => panic!("{other:?}"),
    }
    match &graph.skeleton.blocks[branch_predecessor].term {
        SkeletonTerminator::Branch {
            args: branch_args, ..
        } => {
            assert_eq!(branch_args, &[args[7]]);
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn splitting_block_moves_effect_suffix_and_original_terminator() {
    let mut graph: EGraph = EGraph::new();
    let unit = Type::Constructed(TypeName::Unit, vec![]);
    let first = graph.alloc_side_effect_result(unit.clone());
    let second = graph.alloc_side_effect_result(unit.clone());
    let third = graph.alloc_side_effect_result(unit);
    let entry = graph.skeleton.entry;
    graph.skeleton.blocks[entry].side_effects = vec![effect(first), effect(second), effect(third)];
    graph.skeleton.blocks[entry].term = SkeletonTerminator::Return(Some(third));

    let continuation = graph.skeleton.split_block_before_effect(entry, 1);

    assert_eq!(
        graph.skeleton.blocks[entry]
            .side_effects
            .iter()
            .filter_map(|effect| effect.result)
            .collect::<Vec<_>>(),
        [first]
    );
    assert!(matches!(
        &graph.skeleton.blocks[entry].term,
        SkeletonTerminator::Branch { target, args }
            if *target == continuation && args.is_empty()
    ));
    assert_eq!(
        graph.skeleton.blocks[continuation]
            .side_effects
            .iter()
            .filter_map(|effect| effect.result)
            .collect::<Vec<_>>(),
        [second, third]
    );
    assert!(matches!(
        graph.skeleton.blocks[continuation].term,
        SkeletonTerminator::Return(Some(result)) if result == third
    ));
}

#[test]
fn entry_and_program_accept_non_wyn_resource_metadata() {
    let graph = super::super::ir::EGraph::<TestPhase, TestLanguage>::new();
    let entry = super::super::ir::Entry::<TestPhase, TestLanguage>::new_with_resources(
        "custom".to_string(),
        crate::ast::Span::new(0, 0, 0, 0),
        crate::flow::ExecutionModel::Compute {
            local_size: (1, 1, 1),
        },
        vec![],
        vec![],
        vec![7],
        vec![],
        "unit".to_string(),
        graph,
        crate::LookupMap::new(),
    );
    assert_eq!(entry.resource_declarations, [7]);

    let program = super::super::ir::Program::<TestPhase, TestLanguage>::new(
        vec![],
        vec![],
        vec![entry],
        vec![],
        crate::pipeline_descriptor::PipelineDescriptor::default(),
        super::super::ir::RegionInterner::default(),
    );
    assert_eq!(program.entry_points[0].resource_declarations, [7]);
}

#[test]
fn indexes_results_across_skeleton_blocks() {
    let mut graph: EGraph = EGraph::new();
    let unit = Type::Constructed(TypeName::Unit, vec![]);
    let first = graph.alloc_side_effect_result(unit.clone());
    let second = graph.alloc_side_effect_result(unit);
    let entry = graph.skeleton.entry;
    let other = graph.skeleton.create_block();
    graph.skeleton.blocks[entry].side_effects.push(effect(first));
    graph.skeleton.blocks[other].side_effects.push(effect(second));

    let index = graph.side_effect_index();
    assert_eq!(
        index.site(first),
        Some(SideEffectSite {
            block: entry,
            index: 0
        })
    );
    assert_eq!(
        index.site(second),
        Some(SideEffectSite {
            block: other,
            index: 0
        })
    );
    assert_eq!(
        index.effect(&graph, second).and_then(|effect| effect.result),
        Some(second)
    );
}

#[test]
fn replace_all_references_does_not_leave_stale_hash_cons_key() {
    let mut graph = EGraph::new();
    let int = i32_ty();
    let a = graph.intern_pure(PureOp::Int("1".into()), smallvec::smallvec![], int.clone(), None);
    let b = graph.intern_pure(PureOp::Int("2".into()), smallvec::smallvec![], int.clone(), None);
    let old_call = graph.intern_pure(
        PureOp::Call("__test_hash_cons".into()),
        smallvec::smallvec![a, b],
        int.clone(),
        None,
    );

    crate::egir::graph_ops::replace_all_references(&mut graph, b, a);

    let reinterned_old_call = graph.intern_pure(
        PureOp::Call("__test_hash_cons".into()),
        smallvec::smallvec![a, b],
        int,
        None,
    );

    assert_ne!(old_call, reinterned_old_call);
    assert!(graph.verify_hash_cons().is_ok());
}

#[test]
fn retype_node_does_not_leave_stale_hash_cons_key() {
    let mut graph = EGraph::<Semantic>::new();
    let int = i32_ty();
    let uint = u32_ty();
    let arg = graph.intern_pure(PureOp::Int("1".into()), smallvec::smallvec![], int.clone(), None);
    let old_call = graph.intern_pure(
        PureOp::Call("__test_hash_cons_retype".into()),
        smallvec::smallvec![arg],
        int.clone(),
        None,
    );

    graph.retype_node(old_call, uint);

    let reinterned_old_call = graph.intern_pure(
        PureOp::Call("__test_hash_cons_retype".into()),
        smallvec::smallvec![arg],
        int,
        None,
    );

    assert_ne!(old_call, reinterned_old_call);
    assert!(graph.verify_hash_cons().is_ok());
}
