use super::*;
use crate::ast::TypeName;
use crate::egir;
use crate::egir::analysis::GraphAnalysis;
use crate::egir::program::SemanticOpId;
use crate::egir::soac::screma;
use crate::egir::types::{PureOp, SegSpace, SideEffect, Soac, SoacEffect, SoacInputType, SoacOwnership};
use crate::FunctionId;
use polytype::Type;
use smallvec::smallvec;

fn dead_producer_graph() -> EGraph<Semantic> {
    let mut graph = EGraph::<Semantic>::new();
    let int = Type::Constructed(TypeName::Int(32), vec![]);
    let tuple = Type::Constructed(TypeName::Tuple(1), vec![int.clone()]);
    let result = graph.alloc_side_effect_result(tuple);
    let result_binding = graph.value_result(result);
    let _dead_project =
        graph.intern_pure(PureOp::Project { index: 0 }, smallvec![result], int.clone(), None);
    graph.skeleton.blocks[graph.skeleton.entry].side_effects.push(SideEffect {
        kind: SideEffectKind::Soac(SoacEffect(
            SemanticOpId::for_test(0),
            Soac::Screma(screma::Op {
                inputs: Vec::<SoacInputType>::new(),
                form: screma::ScremaForm {
                    pre: screma::Lambda::region(
                        egir::types::SegBody {
                            region: FunctionId::from_index(0),
                            captures: vec![],
                        },
                        vec![],
                        vec![int.clone()],
                    ),
                    scans: vec![],
                    reductions: vec![],
                    post: screma::Lambda::identity(vec![int]),
                },
                result_state: vec![screma::ResultState {
                    ownership: SoacOwnership::Fresh,
                }],
                state: screma::SemanticState::Segmented {
                    space: SegSpace::new(egir::types::SegExtent::Fixed(1)),
                    output_slots: vec![],
                    resources: vec![],
                },
            }),
        )),
        operands: smallvec![],
        result: Some(result_binding),
        effects: None,
        span: None,
    });
    graph
}

#[test]
fn unreachable_project_does_not_keep_dead_segop_alive() {
    let mut graph = dead_producer_graph();
    assert!(eliminate_dead_seg_ops_in_graph(&mut graph, []));
    assert!(graph.skeleton.blocks[graph.skeleton.entry].side_effects.is_empty());
}

#[test]
fn dead_producer_chain_disappears_in_one_application() {
    let mut graph = dead_producer_graph();
    let block = graph.skeleton.entry;
    let first = graph.skeleton.blocks[block].side_effects[0].clone();
    let first_value = first.result.as_ref().unwrap().values()[0];
    let second_value = graph.alloc_side_effect_result(graph.nodes[first_value].ty.clone());
    let mut second = first;
    second.result = Some(graph.value_result(second_value));
    second.operands.push(crate::egir::types::OperandRef::Value(first_value));
    graph.skeleton.blocks[block].side_effects.push(second);
    assert!(dead_seg_ops_in_graph(&graph, [second_value]).is_empty());
    assert!(eliminate_dead_seg_ops_in_graph(&mut graph, []));
    assert!(graph.skeleton.blocks[block].side_effects.is_empty());
}

#[test]
fn tuple_projection_keeps_only_the_observed_producer() {
    let mut graph = dead_producer_graph();
    let block = graph.skeleton.entry;
    let first_effect = graph.skeleton.blocks[block].side_effects[0].clone();
    let first = first_effect.result.as_ref().unwrap().values()[0];
    let second = graph.alloc_side_effect_result(graph.nodes[first].ty.clone());
    let mut second_effect = first_effect;
    second_effect.result = Some(graph.value_result(second));
    graph.skeleton.blocks[block].side_effects.push(second_effect);
    let ty = graph.nodes[first].ty.clone();
    let tuple = graph.intern_pure(
        PureOp::Tuple(2),
        smallvec![first, second],
        Type::Constructed(TypeName::Tuple(2), vec![ty.clone(), ty.clone()]),
        None,
    );
    let projected = graph.intern_pure(PureOp::Project { index: 1 }, smallvec![tuple], ty, None);
    let graph_analysis = GraphAnalysis::new(&graph);
    let facts = graph_analysis.slice();
    assert!(!facts.pure_reaches(first, projected));
    assert!(facts.pure_reaches(second, projected));
    assert!(eliminate_dead_seg_ops_in_graph(&mut graph, [projected]));
    assert_eq!(graph.skeleton.blocks[block].side_effects.len(), 1);
    assert_eq!(
        graph.skeleton.blocks[block].side_effects[0].result.as_ref().unwrap().values(),
        vec![second]
    );
}
