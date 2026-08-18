use super::*;
use crate::ast::TypeName;
use crate::egir::types::{
    EffectOp, EffectToken, OperandRef, PlaceAccess, PlaceId, PlaceRegion, PlaceType, PureOp, SideEffectKind,
};
use crate::op;
use crate::ssa::types::ConstantValue;
use polytype::Type;
use smallvec::smallvec;

fn u32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::UInt(32), vec![])
}

fn bool_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Bool, vec![])
}

fn test_place(graph: &mut EGraph) -> PlaceId {
    graph.add_alloca_place(
        PlaceType {
            pointee: u32_ty(),
            region: PlaceRegion::Function,
            access: PlaceAccess::ReadWrite,
        },
        None,
    )
}

fn load_effect(
    graph: &EGraph,
    place: PlaceId,
    result: ValueId,
    effects: (EffectToken, EffectToken),
) -> SideEffect {
    SideEffect::new(
        SideEffectKind::Effect(EffectOp::Load { place }),
        smallvec![],
        Some(graph.value_result(result)),
        Some(effects),
        None,
    )
}

fn store_effect(place: PlaceId, value: ValueId, effects: (EffectToken, EffectToken)) -> SideEffect {
    SideEffect::new(
        SideEffectKind::Effect(EffectOp::Store { place }),
        smallvec![OperandRef::Value(value)],
        None,
        Some(effects),
        None,
    )
}

fn value_effect(
    graph: &EGraph,
    input: ValueId,
    result: ValueId,
    effects: (EffectToken, EffectToken),
) -> SideEffect {
    SideEffect::new(
        SideEffectKind::Effect(EffectOp::Op {
            tag: PureOp::Materialize,
        }),
        smallvec![OperandRef::Value(input)],
        Some(graph.value_result(result)),
        Some(effects),
        None,
    )
}

#[test]
fn selected_projection_remaps_cfg_aliases_and_value_producers() {
    let mut graph = EGraph::<Semantic>::new();
    let entry = graph.skeleton.entry;
    let body = graph.skeleton.create_block();
    let exit = graph.skeleton.create_block();
    let cond = graph.intern_constant(
        ConstantValue::Bool(true),
        Type::Constructed(TypeName::Bool, vec![]),
    );
    let place = test_place(&mut graph);
    let produced = graph.alloc_side_effect_result(u32_ty());
    let effect = load_effect(
        &graph,
        place,
        produced,
        (EffectToken::from(0), EffectToken::from(1)),
    );
    graph.skeleton.blocks[entry].side_effects.push(effect);
    let unrelated = graph.alloc_side_effect_result(u32_ty());
    let effect = load_effect(
        &graph,
        place,
        unrelated,
        (EffectToken::from(1), EffectToken::from(2)),
    );
    graph.skeleton.blocks[entry].side_effects.push(effect);
    let body_param = graph.add_block_param(body, u32_ty());
    let effect = store_effect(place, body_param, (EffectToken::from(2), EffectToken::from(3)));
    graph.skeleton.blocks[body].side_effects.push(effect);
    let produced_args = graph.admit_flow_values([produced]);
    graph.skeleton.blocks[entry].term = SkeletonTerminator::CondBranch {
        cond,
        then_target: body,
        then_args: produced_args,
        else_target: exit,
        else_args: vec![],
    };
    graph.skeleton.blocks[body].term = SkeletonTerminator::Branch {
        target: exit,
        args: vec![],
    };
    graph.skeleton.blocks[exit].term = SkeletonTerminator::Return(None);
    graph.skeleton.blocks[entry].control_header = Some(ControlHeader::Selection { merge: exit });
    let alias = graph.intern_constant(ConstantValue::U32(0), u32_ty());
    graph.nodes[produced].alias = Some(alias);
    graph.nodes[unrelated].alias = Some(alias);

    let projected = GraphProjector::new(&graph)
        .selected(HashSet::from([SideEffectSite {
            block: body,
            index: 0,
        }]))
        .expect("projection");
    assert_eq!(
        projected.graph.skeleton.blocks.iter().map(|(_, block)| block.side_effects.len()).sum::<usize>(),
        1,
        "the selected store consumes the canonical alias without retaining either replaced load"
    );
    assert!(projected.node(produced).is_some());
    assert!(projected.node(unrelated).is_none());
    assert_eq!(projected.node(produced), projected.node(alias));
    assert_eq!(
        projected.graph.nodes[projected.node(produced).unwrap()].alias,
        None
    );
    assert!(matches!(
        projected.graph.skeleton.blocks[projected.block(entry).unwrap()]
            .control_header
            .as_ref(),
        Some(ControlHeader::Selection { merge }) if *merge == projected.block(exit).unwrap()
    ));
}

#[test]
fn complete_projection_remaps_loop_headers_and_parameters() {
    let mut graph = EGraph::<Semantic>::new();
    let entry = graph.skeleton.entry;
    let header = graph.skeleton.create_block();
    let exit = graph.skeleton.create_block();
    let zero = graph.intern_constant(ConstantValue::U32(0), u32_ty());
    let zero_args = graph.admit_flow_values([zero]);
    graph.skeleton.blocks[entry].term = SkeletonTerminator::Branch {
        target: header,
        args: zero_args,
    };
    let _index = graph.add_block_param(header, u32_ty());
    graph.skeleton.blocks[header].term = SkeletonTerminator::Branch {
        target: exit,
        args: vec![],
    };
    graph.skeleton.blocks[exit].term = SkeletonTerminator::Return(None);
    graph.skeleton.blocks[header].control_header = Some(ControlHeader::Loop {
        merge: exit,
        continue_block: header,
    });
    let projected = GraphProjector::new(&graph).all().expect("complete projection");
    assert_eq!(projected.graph.skeleton.blocks.len(), 3);
    assert_eq!(
        projected.graph.skeleton.blocks[projected.block(header).unwrap()].params.len(),
        1
    );
    assert!(projected.node(zero).is_some());
    assert!(matches!(
        projected.graph.skeleton.blocks[projected.block(header).unwrap()]
            .control_header
            .as_ref(),
        Some(ControlHeader::Loop { merge, continue_block })
            if *merge == projected.block(exit).unwrap()
                && *continue_block == projected.block(header).unwrap()
    ));
}

#[test]
fn captured_value_recipe_projects_a_structured_loop_prefix() {
    let mut graph = EGraph::<Semantic>::new();
    let entry = graph.skeleton.entry;
    let header = graph.skeleton.create_block();
    let body = graph.skeleton.create_block();
    let continuation = graph.skeleton.create_block();
    let zero = graph.intern_constant(ConstantValue::U32(0), u32_ty());
    let one = graph.intern_constant(ConstantValue::U32(1), u32_ty());
    let bound = graph.intern_constant(ConstantValue::U32(32), u32_ty());
    let acc = graph.add_block_param(header, u32_ty());
    let index = graph.add_block_param(header, u32_ty());
    let result = graph.add_block_param(continuation, u32_ty());
    let initial_args = graph.admit_flow_values([zero, zero]);
    graph.skeleton.blocks[entry].term = SkeletonTerminator::Branch {
        target: header,
        args: initial_args,
    };
    let cond = graph.intern_pure(
        PureOp::BinOp(op::BinaryOperator::Less),
        smallvec![index, bound],
        bool_ty(),
        None,
    );
    let exit_args = graph.admit_flow_values([acc]);
    graph.skeleton.blocks[header].term = SkeletonTerminator::CondBranch {
        cond,
        then_target: body,
        then_args: vec![],
        else_target: continuation,
        else_args: exit_args,
    };
    let next_acc = graph.intern_pure(
        PureOp::BinOp(op::BinaryOperator::Add),
        smallvec![acc, one],
        u32_ty(),
        None,
    );
    let next_index = graph.intern_pure(
        PureOp::BinOp(op::BinaryOperator::Add),
        smallvec![index, one],
        u32_ty(),
        None,
    );
    let loop_args = graph.admit_flow_values([next_acc, next_index]);
    graph.skeleton.blocks[body].term = SkeletonTerminator::Branch {
        target: header,
        args: loop_args,
    };
    graph.skeleton.blocks[continuation].term = SkeletonTerminator::Return(None);
    graph.skeleton.blocks[header].control_header = Some(ControlHeader::Loop {
        merge: continuation,
        continue_block: body,
    });

    let recipe = GraphProjector::new(&graph)
        .captured_value_recipe(
            result,
            SideEffectSite {
                block: continuation,
                index: 0,
            },
        )
        .expect("structured loop recipe");
    assert_eq!(recipe.projection.graph.skeleton.blocks.len(), 4);
    assert_eq!(
        recipe.result_block,
        recipe.projection.block(continuation).unwrap()
    );
    assert!(matches!(
        recipe.source,
        ValueRecipeSource::StructuredPrefix { continuation: block } if block == continuation
    ));
    assert!(matches!(
        recipe.projection.graph.skeleton.blocks[recipe.projection.block(header).unwrap()]
            .control_header
            .as_ref(),
        Some(ControlHeader::Loop { merge, continue_block })
            if *merge == recipe.projection.block(continuation).unwrap()
                && *continue_block == recipe.projection.block(body).unwrap()
    ));
    recipe.projection.graph.skeleton.verify_branch_arities().expect("projected loop branch arity");
}

#[test]
fn captured_value_recipe_projects_a_structured_selection_prefix() {
    let mut graph = EGraph::<Semantic>::new();
    let entry = graph.skeleton.entry;
    let then_block = graph.skeleton.create_block();
    let else_block = graph.skeleton.create_block();
    let continuation = graph.skeleton.create_block();
    let cond = graph.intern_constant(ConstantValue::Bool(true), bool_ty());
    let left = graph.intern_constant(ConstantValue::U32(1), u32_ty());
    let right = graph.intern_constant(ConstantValue::U32(2), u32_ty());
    let result = graph.add_block_param(continuation, u32_ty());
    graph.skeleton.blocks[entry].term = SkeletonTerminator::CondBranch {
        cond,
        then_target: then_block,
        then_args: vec![],
        else_target: else_block,
        else_args: vec![],
    };
    let left_args = graph.admit_flow_values([left]);
    graph.skeleton.blocks[then_block].term = SkeletonTerminator::Branch {
        target: continuation,
        args: left_args,
    };
    let right_args = graph.admit_flow_values([right]);
    graph.skeleton.blocks[else_block].term = SkeletonTerminator::Branch {
        target: continuation,
        args: right_args,
    };
    graph.skeleton.blocks[continuation].term = SkeletonTerminator::Return(None);
    graph.skeleton.blocks[entry].control_header = Some(ControlHeader::Selection { merge: continuation });

    let recipe = GraphProjector::new(&graph)
        .captured_value_recipe(
            result,
            SideEffectSite {
                block: continuation,
                index: 0,
            },
        )
        .expect("structured selection recipe");
    assert_eq!(recipe.projection.graph.skeleton.blocks.len(), 4);
    assert!(matches!(
        recipe.projection.graph.skeleton.blocks[recipe.projection.graph.skeleton.entry]
            .control_header
            .as_ref(),
        Some(ControlHeader::Selection { merge })
            if *merge == recipe.projection.block(continuation).unwrap()
    ));
    recipe.projection.graph.skeleton.verify_branch_arities().expect("projected selection branch arity");
}

#[test]
fn captured_recipe_reports_selected_effect_result_used_by_retained_terminator() {
    let mut graph = EGraph::<Semantic>::new();
    let entry = graph.skeleton.entry;
    let continuation = graph.skeleton.create_block();
    let then_block = graph.skeleton.create_block();
    let else_block = graph.skeleton.create_block();
    let place = test_place(&mut graph);
    let live_out = graph.alloc_side_effect_result(bool_ty());
    let effect = load_effect(
        &graph,
        place,
        live_out,
        (EffectToken::from(0), EffectToken::from(1)),
    );
    graph.skeleton.blocks[entry].side_effects.push(effect);
    let boundary_source = graph.alloc_side_effect_result(u32_ty());
    let effect = load_effect(
        &graph,
        place,
        boundary_source,
        (EffectToken::from(1), EffectToken::from(2)),
    );
    graph.skeleton.blocks[entry].side_effects.push(effect);
    let boundary = graph.add_block_param(continuation, u32_ty());
    let boundary_args = graph.admit_flow_values([boundary_source]);
    graph.skeleton.blocks[entry].term = SkeletonTerminator::Branch {
        target: continuation,
        args: boundary_args,
    };
    graph.skeleton.blocks[continuation].term = SkeletonTerminator::CondBranch {
        cond: live_out,
        then_target: then_block,
        then_args: vec![],
        else_target: else_block,
        else_args: vec![],
    };
    graph.skeleton.blocks[then_block].term = SkeletonTerminator::Return(None);
    graph.skeleton.blocks[else_block].term = SkeletonTerminator::Return(None);

    let recipe = GraphProjector::new(&graph)
        .captured_value_recipe(
            boundary,
            SideEffectSite {
                block: continuation,
                index: 0,
            },
        )
        .expect("structured value recipe");

    assert_eq!(recipe.live_outs().collect::<Vec<_>>(), vec![live_out]);
}

#[test]
fn entry_recipe_reports_selected_effect_result_used_by_external_value() {
    let mut graph = EGraph::<Semantic>::new();
    let entry = graph.skeleton.entry;
    let place = test_place(&mut graph);
    let live_out = graph.alloc_side_effect_result(u32_ty());
    let effect = load_effect(
        &graph,
        place,
        live_out,
        (EffectToken::from(0), EffectToken::from(1)),
    );
    graph.skeleton.blocks[entry].side_effects.push(effect);
    let root = graph.alloc_side_effect_result(u32_ty());
    let effect = value_effect(
        &graph,
        live_out,
        root,
        (EffectToken::from(1), EffectToken::from(2)),
    );
    graph.skeleton.blocks[entry].side_effects.push(effect);
    graph.skeleton.blocks[entry].term = SkeletonTerminator::Return(None);
    let external = graph.intern_pure(
        PureOp::BinOp(op::BinaryOperator::Add),
        smallvec![live_out, root],
        u32_ty(),
        None,
    );
    let projector = GraphProjector::new(&graph);

    let internal_only = projector.entry_value_recipe(root).expect("entry value recipe");
    assert!(internal_only.live_outs().next().is_none());

    let externally_observed = projector
        .entry_value_recipe_with_retained_values(root, [external])
        .expect("entry recipe with external observer");
    assert_eq!(
        externally_observed.live_outs().collect::<Vec<_>>(),
        vec![live_out]
    );
}

#[test]
fn entry_recipe_projects_multiple_requested_values_as_one_component() {
    let mut graph = EGraph::<Semantic>::new();
    let entry = graph.skeleton.entry;
    let parameter = graph.add_test_value_parameter(0, u32_ty());
    let one = graph.intern_constant(ConstantValue::U32(1), u32_ty());
    let first = graph.intern_pure(
        PureOp::BinOp(op::BinaryOperator::Add),
        smallvec![parameter, one],
        u32_ty(),
        None,
    );
    let second = graph.intern_pure(
        PureOp::BinOp(op::BinaryOperator::Multiply),
        smallvec![parameter, first],
        u32_ty(),
        None,
    );
    graph.skeleton.blocks[entry].term = SkeletonTerminator::Return(None);

    let recipe = GraphProjector::new(&graph)
        .entry_values_recipe([first, second, first])
        .expect("multi-value entry recipe");

    assert_eq!(
        recipe.values,
        vec![
            recipe.projection.node(first).unwrap(),
            recipe.projection.node(second).unwrap()
        ]
    );
    assert_eq!(recipe.projection.graph.skeleton.blocks.len(), 1);
    recipe.projection.graph.verify_hash_cons().expect("projected recipe hash-conses");
}

#[test]
fn structured_value_recipe_leaves_independent_continuation_effect_in_source() {
    let mut graph = EGraph::<Semantic>::new();
    let entry = graph.skeleton.entry;
    let continuation = graph.skeleton.create_block();
    let place = test_place(&mut graph);
    let prefix_value = graph.alloc_side_effect_result(u32_ty());
    let effect = load_effect(
        &graph,
        place,
        prefix_value,
        (EffectToken::from(0), EffectToken::from(1)),
    );
    graph.skeleton.blocks[entry].side_effects.push(effect);
    let result = graph.add_block_param(continuation, u32_ty());
    let prefix_args = graph.admit_flow_values([prefix_value]);
    graph.skeleton.blocks[entry].term = SkeletonTerminator::Branch {
        target: continuation,
        args: prefix_args,
    };
    let independent = graph.alloc_side_effect_result(u32_ty());
    let effect = load_effect(
        &graph,
        place,
        independent,
        (EffectToken::from(1), EffectToken::from(2)),
    );
    graph.skeleton.blocks[continuation].side_effects.push(effect);
    let effect = store_effect(place, result, (EffectToken::from(2), EffectToken::from(3)));
    graph.skeleton.blocks[continuation].side_effects.push(effect);
    graph.skeleton.blocks[continuation].term = SkeletonTerminator::Return(None);

    let recipe = GraphProjector::new(&graph)
        .captured_value_recipe(
            result,
            SideEffectSite {
                block: continuation,
                index: 1,
            },
        )
        .expect("structured recipe");
    assert!(recipe.projection.source_effects().contains(&SideEffectSite {
        block: entry,
        index: 0,
    }));
    assert!(!recipe.projection.source_effects().contains(&SideEffectSite {
        block: continuation,
        index: 0,
    }));
    assert!(recipe.projection.node(prefix_value).is_some());
    assert!(recipe.projection.node(independent).is_none());
}

#[test]
fn selected_operation_recipe_detaches_an_independent_continuation_effect() {
    let mut graph = EGraph::<Semantic>::new();
    let entry = graph.skeleton.entry;
    let continuation = graph.skeleton.create_block();
    let zero = graph.intern_constant(ConstantValue::U32(0), u32_ty());
    let _result = graph.add_block_param(continuation, u32_ty());
    let zero_args = graph.admit_flow_values([zero]);
    graph.skeleton.blocks[entry].term = SkeletonTerminator::Branch {
        target: continuation,
        args: zero_args,
    };
    let produced = graph.alloc_side_effect_result(u32_ty());
    let effect = value_effect(
        &graph,
        zero,
        produced,
        (EffectToken::from(0), EffectToken::from(1)),
    );
    graph.skeleton.blocks[continuation].side_effects.push(effect);
    graph.skeleton.blocks[continuation].term = SkeletonTerminator::Return(None);

    let projected = GraphProjector::new(&graph)
        .selected_operation_recipe(HashSet::from([SideEffectSite {
            block: continuation,
            index: 0,
        }]))
        .expect("detached operation recipe");
    assert_eq!(projected.graph.skeleton.blocks.len(), 1);
    assert!(projected.graph.skeleton.blocks[projected.graph.skeleton.entry].params.is_empty());
    assert_eq!(
        projected.block(continuation),
        Some(projected.graph.skeleton.entry)
    );
    assert!(projected.block(entry).is_none());
    assert!(projected.node(produced).is_some());
}

#[test]
fn selected_operation_recipe_rejects_a_continuation_parameter_dependency() {
    let mut graph = EGraph::<Semantic>::new();
    let entry = graph.skeleton.entry;
    let continuation = graph.skeleton.create_block();
    let zero = graph.intern_constant(ConstantValue::U32(0), u32_ty());
    let result = graph.add_block_param(continuation, u32_ty());
    let zero_args = graph.admit_flow_values([zero]);
    graph.skeleton.blocks[entry].term = SkeletonTerminator::Branch {
        target: continuation,
        args: zero_args,
    };
    let produced = graph.alloc_side_effect_result(u32_ty());
    let effect = value_effect(
        &graph,
        result,
        produced,
        (EffectToken::from(0), EffectToken::from(1)),
    );
    graph.skeleton.blocks[continuation].side_effects.push(effect);
    graph.skeleton.blocks[continuation].term = SkeletonTerminator::Return(None);

    let projection =
        GraphProjector::new(&graph).selected_operation_recipe(HashSet::from([SideEffectSite {
            block: continuation,
            index: 0,
        }]));
    assert!(matches!(
        projection,
        Err(error) if error.contains("block parameter")
    ));
}

#[test]
fn selected_component_detaches_an_independent_continuation_value() {
    let mut graph = EGraph::<Semantic>::new();
    let entry = graph.skeleton.entry;
    let continuation = graph.skeleton.create_block();
    let zero = graph.intern_constant(ConstantValue::U32(0), u32_ty());
    let _prefix_result = graph.add_block_param(continuation, u32_ty());
    let zero_args = graph.admit_flow_values([zero]);
    graph.skeleton.blocks[entry].term = SkeletonTerminator::Branch {
        target: continuation,
        args: zero_args,
    };
    let produced = graph.alloc_side_effect_result(u32_ty());
    let effect = value_effect(
        &graph,
        zero,
        produced,
        (EffectToken::from(0), EffectToken::from(1)),
    );
    graph.skeleton.blocks[continuation].side_effects.push(effect);
    graph.skeleton.blocks[continuation].term = SkeletonTerminator::Return(None);

    let projected = GraphProjector::new(&graph)
        .selected_component_with_values(
            HashSet::from([SideEffectSite {
                block: continuation,
                index: 0,
            }]),
            vec![produced],
        )
        .expect("independent output component");
    assert_eq!(projected.graph.skeleton.blocks.len(), 1);
    assert_eq!(
        projected.block(continuation),
        Some(projected.graph.skeleton.entry)
    );
    assert!(projected.block(entry).is_none());
    assert!(projected.node(produced).is_some());
}

#[test]
fn selected_component_retains_cfg_for_a_continuation_parameter_dependency() {
    let mut graph = EGraph::<Semantic>::new();
    let entry = graph.skeleton.entry;
    let continuation = graph.skeleton.create_block();
    let zero = graph.intern_constant(ConstantValue::U32(0), u32_ty());
    let prefix_result = graph.add_block_param(continuation, u32_ty());
    let zero_args = graph.admit_flow_values([zero]);
    graph.skeleton.blocks[entry].term = SkeletonTerminator::Branch {
        target: continuation,
        args: zero_args,
    };
    let produced = graph.alloc_side_effect_result(u32_ty());
    let effect = value_effect(
        &graph,
        prefix_result,
        produced,
        (EffectToken::from(0), EffectToken::from(1)),
    );
    graph.skeleton.blocks[continuation].side_effects.push(effect);
    graph.skeleton.blocks[continuation].term = SkeletonTerminator::Return(None);

    let projected = GraphProjector::new(&graph)
        .selected_component_with_values(
            HashSet::from([SideEffectSite {
                block: continuation,
                index: 0,
            }]),
            vec![produced],
        )
        .expect("dependent output component retains its control prefix");
    assert_eq!(projected.graph.skeleton.blocks.len(), 2);
    assert!(projected.block(entry).is_some());
    let projected_continuation = projected.block(continuation).expect("projected continuation");
    assert_eq!(
        projected.graph.skeleton.blocks[projected_continuation].params.len(),
        1
    );
    assert!(projected.node(prefix_result).is_some());
}

#[test]
fn projection_does_not_resurrect_eliminated_block_parameters() {
    let mut graph = EGraph::<Semantic>::new();
    let entry = graph.skeleton.entry;
    let continuation = graph.skeleton.create_block();
    let eliminated = graph.add_block_param(continuation, u32_ty());
    graph.skeleton.blocks[continuation].params.clear();
    graph.skeleton.blocks[entry].term = SkeletonTerminator::Branch {
        target: continuation,
        args: vec![],
    };
    graph.skeleton.blocks[continuation].term = SkeletonTerminator::Return(None);

    let projected =
        GraphProjector::new(&graph).all().expect("projection with an eliminated historical parameter");
    assert!(projected.node(eliminated).is_none());
    assert!(projected.graph.skeleton.blocks[projected.block(continuation).unwrap()].params.is_empty());
    projected
        .graph
        .skeleton
        .verify_branch_arities()
        .expect("projection keeps eliminated parameter arity");
}

#[test]
fn value_flow_projection_prunes_unrelated_cfg_lanes_and_parameters() {
    let mut graph = EGraph::<Semantic>::new();
    let entry = graph.skeleton.entry;
    let then_block = graph.skeleton.create_block();
    let else_block = graph.skeleton.create_block();
    let merge = graph.skeleton.create_block();
    let cond = graph.add_test_value_parameter(0, bool_ty());
    let then_value = graph.add_test_value_parameter(1, u32_ty());
    let else_value = graph.add_test_value_parameter(2, u32_ty());
    let unrelated = graph.add_test_value_parameter(3, u32_ty());
    let selected = graph.add_block_param(merge, u32_ty());
    let omitted = graph.add_block_param(merge, u32_ty());

    graph.skeleton.blocks[entry].control_header = Some(ControlHeader::Selection { merge });
    graph.skeleton.blocks[entry].term = SkeletonTerminator::CondBranch {
        cond,
        then_target: then_block,
        then_args: vec![],
        else_target: else_block,
        else_args: vec![],
    };
    let then_args = graph.admit_flow_values([then_value, unrelated]);
    graph.skeleton.blocks[then_block].term = SkeletonTerminator::Branch {
        target: merge,
        args: then_args,
    };
    let else_args = graph.admit_flow_values([else_value, unrelated]);
    graph.skeleton.blocks[else_block].term = SkeletonTerminator::Branch {
        target: merge,
        args: else_args,
    };
    graph.skeleton.blocks[merge].term = SkeletonTerminator::Return(Some(graph.value_result(selected)));

    let projected =
        GraphProjector::new(&graph).value_flow(vec![selected]).expect("pure value-flow projection");
    let projected_merge = projected.block(merge).expect("projected merge");
    assert_eq!(projected.graph.skeleton.blocks[projected_merge].params.len(), 1);
    assert!(projected.node(selected).is_some());
    assert!(projected.node(omitted).is_none());
    assert!(projected.node(unrelated).is_none());
    assert!(projected.node(cond).is_some());
    assert!(projected.node(then_value).is_some());
    assert!(projected.node(else_value).is_some());
    projected
        .graph
        .skeleton
        .verify_branch_arities()
        .expect("value-flow projection keeps branch lanes aligned");
}
