use super::*;

fn i32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Int(32), vec![])
}

fn u32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::UInt(32), vec![])
}

fn effect(result: ValueId) -> SideEffect {
    SideEffect {
        kind: SideEffectKind::Effect(EffectOp::ControlBarrier),
        operands: SmallVec::new(),
        result: Some(ResultBinding::destination(
            Type::Constructed(TypeName::Unit, vec![]),
            ResultDestination::ReturnValue(result),
        )),
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

    fn is_materialized_aggregate(_: &Self::Ty) -> bool {
        false
    }

    fn is_view(_: &Self::Ty) -> bool {
        false
    }

    fn product_fields(_: &Self::Ty) -> Option<&[Self::Ty]> {
        None
    }
}

impl Family for TestPhase {
    type Resource = ();
    type Soac = ();

    fn remap_soac_values(_: &mut Self::Soac, _: &mut dyn FnMut(ValueId) -> ValueId) {}
}

#[derive(Debug)]
enum TestTag {}
type TestProgramFamily = super::super::ir::ProgramFamily<TestPhase, u16, (), ()>;

#[test]
fn graph_accepts_non_wyn_payloads() {
    let mut graph = super::super::ir::EGraph::<TestPhase, TestLanguage>::new();
    let node = graph.intern_pure(PureOp::Unit, SmallVec::new(), "unit".to_string(), None);
    let constant = graph.intern_constant(TestConst::FortyTwo, "number".to_string());
    let entry = graph.skeleton.entry;
    graph.skeleton.blocks[entry].side_effects.push(super::super::ir::SideEffect {
        kind: super::super::ir::SideEffectKind::Soac(()),
        operands: SmallVec::new(),
        result: None,
        effects: None,
        span: None,
    });

    assert_eq!(graph.nodes[node].ty, "unit");
    assert!(matches!(
        graph.nodes[constant].kind,
        super::super::ir::ValueKind::Constant(TestConst::FortyTwo)
    ));
    assert!(matches!(
        graph.skeleton.blocks[entry].side_effects[0].kind,
        super::super::ir::SideEffectKind::Soac(())
    ));
}

#[test]
fn adding_block_params_registers_them_in_order() {
    let mut graph = super::super::ir::EGraph::<TestPhase, TestLanguage>::new();
    let block = graph.skeleton.create_block();

    let first = graph.add_block_param(block, "first".to_string());
    let second = graph.add_block_param(block, "second".to_string());

    assert_eq!(
        graph.skeleton.blocks[block].params.iter().map(|parameter| parameter.value()).collect::<Vec<_>>(),
        [first, second]
    );
    assert!(matches!(
        graph.nodes[first].kind,
        super::super::ir::ValueKind::BlockParam { block: owner, index: 0 } if owner == block
    ));
    assert!(matches!(
        graph.nodes[second].kind,
        super::super::ir::ValueKind::BlockParam { block: owner, index: 1 } if owner == block
    ));
    assert_eq!(graph.nodes[first].ty, "first");
    assert_eq!(graph.nodes[second].ty, "second");
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

    let args = (0..9)
        .map(|index| graph.add_test_value_parameter(index, format!("arg-{index}")))
        .collect::<Vec<_>>();
    graph.skeleton.blocks[entry].term = super::super::ir::SkeletonTerminator::<TestLanguage>::CondBranch {
        cond: args[0],
        then_target: target,
        then_args: graph.admit_flow_values([args[1], args[2], args[3]]),
        else_target: target,
        else_args: graph.admit_flow_values([args[4], args[5], args[6]]),
    };
    graph.skeleton.blocks[branch_predecessor].term =
        super::super::ir::SkeletonTerminator::<TestLanguage>::Branch {
            target,
            args: graph.admit_flow_values([args[6], args[7], args[8]]),
        };

    let slots = [2, 0, 2].into_iter().collect::<crate::SortedSet<_>>();
    let removed = graph.remove_block_param_slots(target, &slots);

    assert_eq!(removed, [first, third]);
    assert_eq!(graph.skeleton.blocks[target].params[0].value(), second);
    assert!(matches!(
        graph.nodes[second].kind,
        super::super::ir::ValueKind::BlockParam { block, index: 0 } if block == target
    ));
    assert!(graph.nodes.contains_key(first));
    assert!(graph.nodes.contains_key(third));
    match &graph.skeleton.blocks[entry].term {
        super::super::ir::SkeletonTerminator::<TestLanguage>::CondBranch {
            then_args, else_args, ..
        } => {
            assert_eq!(then_args[0].value(), args[2]);
            assert_eq!(else_args[0].value(), args[5]);
        }
        other => panic!("{other:?}"),
    }
    match &graph.skeleton.blocks[branch_predecessor].term {
        super::super::ir::SkeletonTerminator::<TestLanguage>::Branch {
            args: branch_args, ..
        } => {
            assert_eq!(branch_args[0].value(), args[7]);
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
    graph.skeleton.blocks[entry].term = SkeletonTerminator::Return(Some(graph.value_result(third)));
    graph.skeleton.blocks[entry].control_header =
        Some(crate::flow::ControlHeader::Selection { merge: entry });

    let continuation = graph.skeleton.split_block_before_effect(entry, 1);

    assert_eq!(
        graph.skeleton.blocks[entry]
            .side_effects
            .iter()
            .filter_map(|effect| effect.value_result())
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
            .filter_map(|effect| effect.value_result())
            .collect::<Vec<_>>(),
        [second, third]
    );
    assert!(matches!(
        &graph.skeleton.blocks[continuation].term,
        SkeletonTerminator::Return(Some(result)) if result.single_value() == Some(third)
    ));
    assert!(graph.skeleton.blocks[entry].control_header.is_none());
    assert!(matches!(
        graph.skeleton.blocks[continuation].control_header.as_ref(),
        Some(crate::flow::ControlHeader::Selection { merge }) if *merge == entry
    ));
}

#[test]
fn entry_and_program_accept_non_wyn_resource_metadata() {
    let graph = super::super::ir::EGraph::<TestPhase, TestLanguage>::new();
    let entry = super::super::ir::Entry::<TestPhase, u16, (), TestLanguage>::new_with_resources(
        "custom".to_string(),
        crate::EntryId::from_index(0),
        crate::ast::Span::new(0, 0, 0, 0),
        crate::flow::ExecutionModel::Compute {
            local_size: (1, 1, 1),
        },
        vec![],
        vec![],
        vec![7],
        vec![],
        by_value_function_result::<TestLanguage>("unit".to_string()),
        graph,
    );
    assert_eq!(entry.resource_declarations, [7]);

    let program = super::super::ir::Program::<TestTag, TestProgramFamily, (), TestLanguage>::from_parts(
        vec![],
        vec![],
        vec![entry],
        vec![],
        (),
        (),
    );
    assert_eq!(program.entry_points[0].resource_declarations, [7]);
}

#[test]
fn retaining_entry_parameter_indices_compacts_interface_and_nodes() {
    let mut graph = super::super::ir::EGraph::<TestPhase, TestLanguage>::new();
    let first = graph.add_test_value_parameter(0, "first".to_string());
    let removed = graph.add_test_value_parameter(1, "removed".to_string());
    let third = graph.add_test_value_parameter(2, "third".to_string());
    let inputs = ["first", "removed", "third"]
        .into_iter()
        .map(|name| crate::interface::EntryInput {
            name: name.to_string(),
            ty: name.to_string(),
            size_hint: None,
            kind: crate::interface::EntryInputKind::Value { decoration: None },
        })
        .collect();
    let params = ["first", "removed", "third"]
        .into_iter()
        .map(|name| FuncParam::value(name.to_string(), name.to_string()))
        .collect();
    let mut entry = super::super::ir::Entry::<TestPhase, (), (), TestLanguage>::new_with_resources(
        "compact".to_string(),
        crate::EntryId::from_index(0),
        crate::ast::Span::dummy(),
        crate::flow::ExecutionModel::Compute {
            local_size: (1, 1, 1),
        },
        inputs,
        vec![],
        vec![],
        params,
        by_value_function_result::<TestLanguage>("unit".to_string()),
        graph,
    );

    entry.retain_parameter_indices(&[0, 2].into_iter().collect());

    assert_eq!(
        entry.inputs.iter().map(|input| input.name.as_str()).collect::<Vec<_>>(),
        ["first", "third"]
    );
    assert_eq!(
        entry.params.iter().map(|parameter| parameter.name()).collect::<Vec<_>>(),
        ["first", "third"]
    );
    assert!(matches!(
        entry.graph.nodes[first].kind,
        super::super::ir::ValueKind::FuncParam { parameter } if parameter.index() == 0
    ));
    assert!(!entry.graph.nodes.contains_key(removed));
    assert!(matches!(
        entry.graph.nodes[third].kind,
        super::super::ir::ValueKind::FuncParam { parameter } if parameter.index() == 1
    ));
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
        index.effect(&graph, second).and_then(SideEffect::value_result),
        Some(second)
    );
}

#[test]
fn replace_all_references_does_not_leave_stale_hash_cons_key() {
    let mut graph: EGraph = EGraph::new();
    let int = i32_ty();
    let a = graph.intern_pure(PureOp::Int("1".into()), smallvec::smallvec![], int.clone(), None);
    let b = graph.intern_pure(PureOp::Int("2".into()), smallvec::smallvec![], int.clone(), None);
    let old_call = graph.intern_pure(
        PureOp::BinOp(crate::op::BinaryOperator::Add),
        smallvec::smallvec![a, b],
        int.clone(),
        None,
    );

    graph.replace_value_references(b, a);

    let reinterned_old_call = graph.intern_pure(
        PureOp::BinOp(crate::op::BinaryOperator::Add),
        smallvec::smallvec![a, b],
        int,
        None,
    );

    assert_ne!(old_call, reinterned_old_call);
    assert!(graph.verify_hash_cons().is_ok());
}

#[test]
fn removing_func_param_clears_its_metadata() {
    let mut graph = super::super::ir::EGraph::<TestPhase, TestLanguage>::new();
    let span = crate::ast::Span::new(1, 2, 3, 4);
    let param = graph.add_test_value_parameter(0, "number".to_string());
    graph.nodes[param].span = Some(span);

    assert!(graph.remove_func_param(param));
    assert!(!graph.nodes.contains_key(param));
}

#[test]
fn retype_node_does_not_leave_stale_hash_cons_key() {
    let mut graph = EGraph::<Semantic>::new();
    let int = i32_ty();
    let uint = u32_ty();
    let arg = graph.intern_pure(PureOp::Int("1".into()), smallvec::smallvec![], int.clone(), None);
    let old_call = graph.intern_pure(PureOp::Materialize, smallvec::smallvec![arg], int.clone(), None);

    graph.retype_node(old_call, uint);

    let reinterned_old_call = graph.intern_pure(PureOp::Materialize, smallvec::smallvec![arg], int, None);

    assert_ne!(old_call, reinterned_old_call);
    assert!(graph.verify_hash_cons().is_ok());
}
