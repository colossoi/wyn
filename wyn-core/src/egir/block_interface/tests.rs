use super::*;
use crate::ast::TypeName;
use crate::egir::types::{EGraph, Physical, SkeletonTerminator};
use crate::ssa::types::ConstantValue;
use polytype::Type;

fn ty() -> Type<TypeName> {
    Type::Constructed(TypeName::UInt(32), vec![])
}

fn fixture() -> (EGraph, BlockId, BlockId, Vec<ValueId>, Vec<ValueId>) {
    let mut graph = EGraph::new();
    let entry = graph.skeleton.entry;
    let predecessor = graph.skeleton.create_block();
    let merge = graph.skeleton.create_block();
    let exit = graph.skeleton.create_block();
    let parameters = (0..5).map(|_| graph.add_block_param(merge, ty())).collect::<Vec<_>>();
    graph.add_block_param(exit, ty());
    graph.add_block_param(exit, ty());
    let values =
        (0..15).map(|value| graph.intern_constant(ConstantValue::U32(value), ty())).collect::<Vec<_>>();
    let condition = graph.intern_constant(
        ConstantValue::Bool(true),
        Type::Constructed(TypeName::Bool, vec![]),
    );
    graph.skeleton.blocks[entry].term = SkeletonTerminator::CondBranch {
        cond: condition,
        then_target: merge,
        then_args: graph.admit_flow_values(values[..5].iter().copied()),
        else_target: merge,
        else_args: graph.admit_flow_values(values[5..10].iter().copied()),
    };
    graph.skeleton.blocks[predecessor].term = SkeletonTerminator::Branch {
        target: merge,
        args: graph.admit_flow_values(values[10..].iter().copied()),
    };
    graph.skeleton.blocks[merge].term = SkeletonTerminator::Branch {
        target: exit,
        args: graph.admit_flow_values([parameters[4], parameters[1]]),
    };
    (graph, merge, exit, parameters, values)
}

#[test]
fn batch_selection_permutates_parameters_arguments_and_node_indices() {
    let (mut graph, merge, exit, parameters, values) = fixture();
    let exit_removed = graph.skeleton.blocks[exit].params[0].value();
    let removed = select_columns(&mut graph, |block, interface| {
        Ok(if block == merge {
            vec![4, 1, 3]
        } else if block == exit {
            vec![1]
        } else {
            (0..interface.columns().len()).collect()
        })
    })
    .unwrap();
    assert_eq!(
        removed.into_iter().collect::<LookupSet<_>>(),
        [parameters[0], parameters[2], exit_removed].into_iter().collect()
    );
    let interfaces = extract(&graph).unwrap();
    assert_eq!(
        interfaces[&merge].parameters().map(FlowValueId::value).collect::<Vec<_>>(),
        [parameters[4], parameters[1], parameters[3]]
    );
    let emitted = interfaces[&merge]
        .rows()
        .map(|(_, args)| args.into_iter().map(FlowValueId::value).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    assert_eq!(
        emitted,
        [
            vec![values[4], values[1], values[3]],
            vec![values[9], values[6], values[8]],
            vec![values[14], values[11], values[13]]
        ]
    );
    assert_eq!(
        interfaces[&exit].columns()[0].common_argument().unwrap().value(),
        parameters[1]
    );
    assert!(graph.nodes.contains_key(parameters[0]));
    for (&block, interface) in &interfaces {
        for (slot, parameter) in interface.parameters().enumerate() {
            assert!(
                matches!(graph.nodes[parameter.value()].kind(), ValueKind::BlockParam { block: owner, index } if *owner == block && *index == slot)
            );
        }
    }
}

#[test]
fn a_failed_last_selection_does_not_partially_apply_the_batch() {
    let (mut graph, _, _, _, _) = fixture();
    let original = extract(&graph).unwrap();
    let mut calls = 0;
    assert!(select_columns(&mut graph, |_, _| {
        calls += 1;
        Ok(if calls == original.len() { vec![usize::MAX] } else { vec![] })
    })
    .is_err());
    assert_eq!(calls, original.len());
    assert_eq!(extract(&graph).unwrap(), original);
}

#[test]
fn extraction_rejects_invalid_membership_arity_and_destinations() {
    let (mut graph, merge, _, parameters, _) = fixture();
    graph.nodes[parameters[0]].kind = ValueKind::BlockParam {
        block: merge,
        index: 1,
    };
    assert!(extract(&graph).unwrap_err().contains("mismatched block parameter"));
    graph.nodes[parameters[0]].kind = ValueKind::BlockParam {
        block: graph.skeleton.entry,
        index: 0,
    };
    assert!(extract(&graph).is_err());
    graph.nodes[parameters[0]].kind = ValueKind::BlockParam {
        block: merge,
        index: 0,
    };
    let duplicate = graph.admit_flow_value(parameters[0]);
    graph.skeleton.blocks[merge].params.push(duplicate);
    assert!(extract(&graph).is_err());
    graph.skeleton.blocks[merge].params.pop();
    let entry = graph.skeleton.entry;
    IncomingEdge::Then(entry, parameters[0]).arguments_mut(&mut graph.skeleton.blocks[entry].term).pop();
    assert!(extract(&graph).unwrap_err().contains("Arity"));
    graph.skeleton.blocks.remove(merge);
    assert!(extract(&graph).unwrap_err().contains("absent block"));
}

#[test]
fn physical_flow_admission_is_rechecked_after_remapping() {
    let mut graph = EGraph::<Physical>::new();
    let entry = graph.skeleton.entry;
    let target = graph.skeleton.create_block();
    let parameter = graph.add_block_param(target, ty());
    let aggregate = graph.alloc_side_effect_result(crate::types::sized_array(4, ty()));
    let argument = graph.admit_flow_value(parameter).try_remap(|_| Ok::<_, String>(aggregate)).unwrap();
    graph.skeleton.blocks[entry].term = SkeletonTerminator::Branch {
        target,
        args: vec![argument],
    };
    assert!(extract(&graph).unwrap_err().contains("materialized flow"));
    graph.nodes.remove(aggregate);
    assert!(extract(&graph).unwrap_err().contains("missing flow value"));
}

#[test]
fn edge_identity_is_source_and_arm_independent_of_condition_metadata() {
    let (graph, merge, _, parameters, values) = fixture();
    let entry = graph.skeleton.entry;
    let argument = graph.admit_flow_value(values[0]);
    assert_eq!(
        BlockInterface::new(
            [argument],
            [
                (IncomingEdge::Then(entry, parameters[0]), vec![argument]),
                (IncomingEdge::Then(entry, parameters[1]), vec![argument]),
            ]
        ),
        Err(wyn_block_interface::Error::DuplicateEdge)
    );
    assert_eq!(extract(&graph).unwrap()[&merge].edges().len(), 3);
}

#[test]
fn zero_width_interfaces_retain_both_arms_and_control_conditions() {
    let (mut graph, merge, _, _, _) = fixture();
    select_columns(&mut graph, |_, _| Ok(vec![])).unwrap();
    let interfaces = extract(&graph).unwrap();
    let edges = interfaces[&merge].edges();
    assert_eq!(edges.len(), 3);
    assert!(matches!(edges[0], IncomingEdge::Then(_, _)));
    assert!(matches!(edges[1], IncomingEdge::Else(_, _)));
    assert_eq!(edges[0].source(), edges[1].source());
    assert_eq!(edges[0].condition(), edges[1].condition());
    assert!(edges[0].condition().is_some());
    assert!(interfaces[&merge].rows().all(|(_, args)| args.is_empty()));
}
