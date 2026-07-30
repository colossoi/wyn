use super::*;
use crate::ast::TypeName;
use crate::egir::types::{EffectOp, EffectToken, PureOp, SideEffectKind};
use crate::ssa::types::ConstantValue;
use polytype::Type;
use smallvec::smallvec;

fn u32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::UInt(32), vec![])
}

fn bool_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Bool, vec![])
}

#[test]
fn selected_projection_remaps_cfg_aliases_and_value_producers() {
    let mut graph = EGraph::new();
    let entry = graph.skeleton.entry;
    let body = graph.skeleton.create_block();
    let exit = graph.skeleton.create_block();
    let cond = graph.intern_constant(
        ConstantValue::Bool(true),
        Type::Constructed(TypeName::Bool, vec![]),
    );
    let place = graph.intern_constant(ConstantValue::U32(0), u32_ty());
    let produced = graph.alloc_side_effect_result(u32_ty());
    graph.skeleton.blocks[entry].side_effects.push(SideEffect {
        kind: SideEffectKind::Effect(EffectOp::Load),
        operand_nodes: smallvec![place],
        result: Some(produced),
        effects: Some((EffectToken::from(0), EffectToken::from(1))),
        span: None,
    });
    let unrelated = graph.alloc_side_effect_result(u32_ty());
    graph.skeleton.blocks[entry].side_effects.push(SideEffect {
        kind: SideEffectKind::Effect(EffectOp::Load),
        operand_nodes: smallvec![place],
        result: Some(unrelated),
        effects: Some((EffectToken::from(1), EffectToken::from(2))),
        span: None,
    });
    let body_param = graph.add_block_param(body, u32_ty());
    graph.skeleton.blocks[body].side_effects.push(SideEffect {
        kind: SideEffectKind::Effect(EffectOp::Store),
        operand_nodes: smallvec![place, body_param],
        result: None,
        effects: Some((EffectToken::from(2), EffectToken::from(3))),
        span: None,
    });
    graph.skeleton.blocks[entry].term = SkeletonTerminator::CondBranch {
        cond,
        then_target: body,
        then_args: vec![produced],
        else_target: exit,
        else_args: vec![],
    };
    graph.skeleton.blocks[body].term = SkeletonTerminator::Branch {
        target: exit,
        args: vec![],
    };
    graph.skeleton.blocks[exit].term = SkeletonTerminator::Return(None);
    graph.skeleton.blocks[entry].control_header = Some(ControlHeader::Selection { merge: exit });
    graph.nodes[produced].alias = Some(place);
    graph.nodes[unrelated].alias = Some(place);

    let projected = GraphProjector::new(&graph)
        .selected(HashSet::from([SideEffectSite {
            block: body,
            index: 0,
        }]))
        .expect("projection");
    assert_eq!(
        projected.graph.skeleton.blocks.iter().map(|(_, block)| block.side_effects.len()).sum::<usize>(),
        2,
        "selected store and its load producer survive; unrelated load does not"
    );
    assert!(projected.node(produced).is_some());
    assert!(projected.node(unrelated).is_none());
    assert_eq!(
        projected.graph.nodes[projected.node(produced).unwrap()].alias,
        projected.node(place)
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
    let mut graph = EGraph::new();
    let entry = graph.skeleton.entry;
    let header = graph.skeleton.create_block();
    let exit = graph.skeleton.create_block();
    let zero = graph.intern_constant(ConstantValue::U32(0), u32_ty());
    graph.skeleton.blocks[entry].term = SkeletonTerminator::Branch {
        target: header,
        args: vec![zero],
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
    let mut graph = EGraph::new();
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
    graph.skeleton.blocks[entry].term = SkeletonTerminator::Branch {
        target: header,
        args: vec![zero, zero],
    };
    let cond = graph.intern_pure(
        PureOp::BinOp(crate::op::BinaryOperator::Less),
        smallvec![index, bound],
        bool_ty(),
        None,
    );
    graph.skeleton.blocks[header].term = SkeletonTerminator::CondBranch {
        cond,
        then_target: body,
        then_args: vec![],
        else_target: continuation,
        else_args: vec![acc],
    };
    let next_acc = graph.intern_pure(
        PureOp::BinOp(crate::op::BinaryOperator::Add),
        smallvec![acc, one],
        u32_ty(),
        None,
    );
    let next_index = graph.intern_pure(
        PureOp::BinOp(crate::op::BinaryOperator::Add),
        smallvec![index, one],
        u32_ty(),
        None,
    );
    graph.skeleton.blocks[body].term = SkeletonTerminator::Branch {
        target: header,
        args: vec![next_acc, next_index],
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
    let mut graph = EGraph::new();
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
    graph.skeleton.blocks[then_block].term = SkeletonTerminator::Branch {
        target: continuation,
        args: vec![left],
    };
    graph.skeleton.blocks[else_block].term = SkeletonTerminator::Branch {
        target: continuation,
        args: vec![right],
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
    let mut graph = EGraph::new();
    let entry = graph.skeleton.entry;
    let continuation = graph.skeleton.create_block();
    let then_block = graph.skeleton.create_block();
    let else_block = graph.skeleton.create_block();
    let place = graph.intern_constant(ConstantValue::U32(0), u32_ty());
    let live_out = graph.alloc_side_effect_result(bool_ty());
    graph.skeleton.blocks[entry].side_effects.push(SideEffect {
        kind: SideEffectKind::Effect(EffectOp::Load),
        operand_nodes: smallvec![place],
        result: Some(live_out),
        effects: Some((EffectToken::from(0), EffectToken::from(1))),
        span: None,
    });
    let boundary_source = graph.alloc_side_effect_result(u32_ty());
    graph.skeleton.blocks[entry].side_effects.push(SideEffect {
        kind: SideEffectKind::Effect(EffectOp::Load),
        operand_nodes: smallvec![place],
        result: Some(boundary_source),
        effects: Some((EffectToken::from(1), EffectToken::from(2))),
        span: None,
    });
    let boundary = graph.add_block_param(continuation, u32_ty());
    graph.skeleton.blocks[entry].term = SkeletonTerminator::Branch {
        target: continuation,
        args: vec![boundary_source],
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
    let mut graph = EGraph::new();
    let entry = graph.skeleton.entry;
    let place = graph.intern_constant(ConstantValue::U32(0), u32_ty());
    let live_out = graph.alloc_side_effect_result(u32_ty());
    graph.skeleton.blocks[entry].side_effects.push(SideEffect {
        kind: SideEffectKind::Effect(EffectOp::Load),
        operand_nodes: smallvec![place],
        result: Some(live_out),
        effects: Some((EffectToken::from(0), EffectToken::from(1))),
        span: None,
    });
    let root = graph.alloc_side_effect_result(u32_ty());
    graph.skeleton.blocks[entry].side_effects.push(SideEffect {
        kind: SideEffectKind::Effect(EffectOp::Load),
        operand_nodes: smallvec![live_out],
        result: Some(root),
        effects: Some((EffectToken::from(1), EffectToken::from(2))),
        span: None,
    });
    graph.skeleton.blocks[entry].term = SkeletonTerminator::Return(None);
    let external = graph.intern_pure(
        PureOp::BinOp(crate::op::BinaryOperator::Add),
        smallvec![live_out, place],
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
    let mut graph = EGraph::new();
    let entry = graph.skeleton.entry;
    let parameter = graph.add_func_param(0, u32_ty());
    let one = graph.intern_constant(ConstantValue::U32(1), u32_ty());
    let first = graph.intern_pure(
        PureOp::BinOp(crate::op::BinaryOperator::Add),
        smallvec![parameter, one],
        u32_ty(),
        None,
    );
    let second = graph.intern_pure(
        PureOp::BinOp(crate::op::BinaryOperator::Multiply),
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
    let mut graph = EGraph::new();
    let entry = graph.skeleton.entry;
    let continuation = graph.skeleton.create_block();
    let place = graph.intern_constant(ConstantValue::U32(0), u32_ty());
    let prefix_value = graph.alloc_side_effect_result(u32_ty());
    graph.skeleton.blocks[entry].side_effects.push(SideEffect {
        kind: SideEffectKind::Effect(EffectOp::Load),
        operand_nodes: smallvec![place],
        result: Some(prefix_value),
        effects: Some((EffectToken::from(0), EffectToken::from(1))),
        span: None,
    });
    let result = graph.add_block_param(continuation, u32_ty());
    graph.skeleton.blocks[entry].term = SkeletonTerminator::Branch {
        target: continuation,
        args: vec![prefix_value],
    };
    let independent = graph.alloc_side_effect_result(u32_ty());
    graph.skeleton.blocks[continuation].side_effects.push(SideEffect {
        kind: SideEffectKind::Effect(EffectOp::Load),
        operand_nodes: smallvec![place],
        result: Some(independent),
        effects: Some((EffectToken::from(1), EffectToken::from(2))),
        span: None,
    });
    graph.skeleton.blocks[continuation].side_effects.push(SideEffect {
        kind: SideEffectKind::Effect(EffectOp::Store),
        operand_nodes: smallvec![place, result],
        result: None,
        effects: Some((EffectToken::from(2), EffectToken::from(3))),
        span: None,
    });
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
    let mut graph = EGraph::new();
    let entry = graph.skeleton.entry;
    let continuation = graph.skeleton.create_block();
    let zero = graph.intern_constant(ConstantValue::U32(0), u32_ty());
    let _result = graph.add_block_param(continuation, u32_ty());
    graph.skeleton.blocks[entry].term = SkeletonTerminator::Branch {
        target: continuation,
        args: vec![zero],
    };
    let produced = graph.alloc_side_effect_result(u32_ty());
    graph.skeleton.blocks[continuation].side_effects.push(SideEffect {
        kind: SideEffectKind::Effect(EffectOp::Load),
        operand_nodes: smallvec![zero],
        result: Some(produced),
        effects: Some((EffectToken::from(0), EffectToken::from(1))),
        span: None,
    });
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
    let mut graph = EGraph::new();
    let entry = graph.skeleton.entry;
    let continuation = graph.skeleton.create_block();
    let zero = graph.intern_constant(ConstantValue::U32(0), u32_ty());
    let result = graph.add_block_param(continuation, u32_ty());
    graph.skeleton.blocks[entry].term = SkeletonTerminator::Branch {
        target: continuation,
        args: vec![zero],
    };
    let produced = graph.alloc_side_effect_result(u32_ty());
    graph.skeleton.blocks[continuation].side_effects.push(SideEffect {
        kind: SideEffectKind::Effect(EffectOp::Load),
        operand_nodes: smallvec![result],
        result: Some(produced),
        effects: Some((EffectToken::from(0), EffectToken::from(1))),
        span: None,
    });
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
    let mut graph = EGraph::new();
    let entry = graph.skeleton.entry;
    let continuation = graph.skeleton.create_block();
    let zero = graph.intern_constant(ConstantValue::U32(0), u32_ty());
    let _prefix_result = graph.add_block_param(continuation, u32_ty());
    graph.skeleton.blocks[entry].term = SkeletonTerminator::Branch {
        target: continuation,
        args: vec![zero],
    };
    let produced = graph.alloc_side_effect_result(u32_ty());
    graph.skeleton.blocks[continuation].side_effects.push(SideEffect {
        kind: SideEffectKind::Effect(EffectOp::Load),
        operand_nodes: smallvec![zero],
        result: Some(produced),
        effects: Some((EffectToken::from(0), EffectToken::from(1))),
        span: None,
    });
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
    let mut graph = EGraph::new();
    let entry = graph.skeleton.entry;
    let continuation = graph.skeleton.create_block();
    let zero = graph.intern_constant(ConstantValue::U32(0), u32_ty());
    let prefix_result = graph.add_block_param(continuation, u32_ty());
    graph.skeleton.blocks[entry].term = SkeletonTerminator::Branch {
        target: continuation,
        args: vec![zero],
    };
    let produced = graph.alloc_side_effect_result(u32_ty());
    graph.skeleton.blocks[continuation].side_effects.push(SideEffect {
        kind: SideEffectKind::Effect(EffectOp::Load),
        operand_nodes: smallvec![prefix_result],
        result: Some(produced),
        effects: Some((EffectToken::from(0), EffectToken::from(1))),
        span: None,
    });
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
    let mut graph = EGraph::new();
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
    let mut graph = EGraph::new();
    let entry = graph.skeleton.entry;
    let then_block = graph.skeleton.create_block();
    let else_block = graph.skeleton.create_block();
    let merge = graph.skeleton.create_block();
    let cond = graph.add_func_param(0, bool_ty());
    let then_value = graph.add_func_param(1, u32_ty());
    let else_value = graph.add_func_param(2, u32_ty());
    let unrelated = graph.add_func_param(3, u32_ty());
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
    graph.skeleton.blocks[then_block].term = SkeletonTerminator::Branch {
        target: merge,
        args: vec![then_value, unrelated],
    };
    graph.skeleton.blocks[else_block].term = SkeletonTerminator::Branch {
        target: merge,
        args: vec![else_value, unrelated],
    };
    graph.skeleton.blocks[merge].term = SkeletonTerminator::Return(Some(selected));

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
