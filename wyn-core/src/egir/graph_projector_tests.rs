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
    let headers = LookupMap::from([(entry, ControlHeader::Selection { merge: exit })]);
    let aliases = LookupMap::from([(produced, place), (unrelated, place)]);

    let projected = GraphProjector::new(&graph, &headers)
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
    assert_eq!(projected.remap_aliases(&aliases).len(), 1);
    assert!(matches!(
        projected.control_headers.get(&projected.block(entry).unwrap()),
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
    let headers = LookupMap::from([(
        header,
        ControlHeader::Loop {
            merge: exit,
            continue_block: header,
        },
    )]);
    let projected = GraphProjector::new(&graph, &headers).all().expect("complete projection");
    assert_eq!(projected.graph.skeleton.blocks.len(), 3);
    assert_eq!(
        projected.graph.skeleton.blocks[projected.block(header).unwrap()].params.len(),
        1
    );
    assert!(projected.node(zero).is_some());
    assert!(matches!(
        projected.control_headers.get(&projected.block(header).unwrap()),
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
        PureOp::BinOp("<".into()),
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
    let next_acc = graph.intern_pure(PureOp::BinOp("+".into()), smallvec![acc, one], u32_ty(), None);
    let next_index = graph.intern_pure(PureOp::BinOp("+".into()), smallvec![index, one], u32_ty(), None);
    graph.skeleton.blocks[body].term = SkeletonTerminator::Branch {
        target: header,
        args: vec![next_acc, next_index],
    };
    graph.skeleton.blocks[continuation].term = SkeletonTerminator::Return(None);
    let headers = LookupMap::from([(
        header,
        ControlHeader::Loop {
            merge: continuation,
            continue_block: body,
        },
    )]);

    let recipe = GraphProjector::new(&graph, &headers)
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
        recipe
            .projection
            .control_headers
            .get(&recipe.projection.block(header).unwrap()),
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
    let headers = LookupMap::from([(entry, ControlHeader::Selection { merge: continuation })]);

    let recipe = GraphProjector::new(&graph, &headers)
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
        recipe
            .projection
            .control_headers
            .get(&recipe.projection.graph.skeleton.entry),
        Some(ControlHeader::Selection { merge })
            if *merge == recipe.projection.block(continuation).unwrap()
    ));
    recipe.projection.graph.skeleton.verify_branch_arities().expect("projected selection branch arity");
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

    let projected = GraphProjector::new(&graph, &LookupMap::new())
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
        GraphProjector::new(&graph, &LookupMap::new()).selected_operation_recipe(HashSet::from([
            SideEffectSite {
                block: continuation,
                index: 0,
            },
        ]));
    assert!(matches!(
        projection,
        Err(error) if error.contains("block parameter")
    ));
}

#[test]
fn projection_does_not_resurrect_eliminated_block_parameters() {
    let mut graph = EGraph::new();
    let entry = graph.skeleton.entry;
    let continuation = graph.skeleton.create_block();
    let eliminated = graph.nodes.insert(ENode::BlockParam {
        block: continuation,
        index: 0,
    });
    graph.types.insert(eliminated, u32_ty());
    graph.skeleton.blocks[entry].term = SkeletonTerminator::Branch {
        target: continuation,
        args: vec![],
    };
    graph.skeleton.blocks[continuation].term = SkeletonTerminator::Return(None);

    let projected = GraphProjector::new(&graph, &LookupMap::new())
        .all()
        .expect("projection with an eliminated historical parameter");
    assert!(projected.node(eliminated).is_none());
    assert!(projected.graph.skeleton.blocks[projected.block(continuation).unwrap()].params.is_empty());
    projected
        .graph
        .skeleton
        .verify_branch_arities()
        .expect("projection keeps eliminated parameter arity");
}
