use super::*;
use crate::ast::TypeName;
use crate::egir::soac::screma;
use crate::egir::types::{RegionId, SegBody, Semantic, SoacDestination};
use polytype::Type;
use smallvec::smallvec;

fn op(id: u32) -> SemanticOpId {
    SemanticOpId(id)
}

fn dep(
    producer: &SemanticOpId,
    consumer: &SemanticOpId,
    kind: SemanticDependencyKind,
) -> SemanticDependency {
    SemanticDependency {
        producer: producer.clone(),
        consumer: consumer.clone(),
        kind,
    }
}

#[test]
fn oracle_distinguishes_conflict_flow_and_value_edges() {
    let a = op(0);
    let b = op(1);
    let c = op(2);

    // a --Resource--> b, b --Effect--> c, a --Value--> c
    let deps = vec![
        dep(&a, &b, SemanticDependencyKind::Resource),
        dep(&b, &c, SemanticDependencyKind::Effect),
        dep(&a, &c, SemanticDependencyKind::Value),
    ];
    let g = SemanticGraph::new(&deps);

    // Resource and effect edges are reordering conflicts (both directions);
    // value edges are represented separately.
    assert!(g.conflicts(&a, &b), "resource edge is a conflict");
    assert!(g.conflicts(&b, &a));
    assert!(g.conflicts(&b, &c), "effect ordering is a reordering conflict");
    assert!(
        !g.conflicts(&a, &c),
        "a value edge alone is fusable, not a conflict"
    );

    // Reachability follows value edges only, not effect ordering or resource.
    assert!(g.reachable_between(&a, &c), "a --Value--> c");
    assert!(
        !g.reachable_between(&b, &c),
        "b --Effect--> c is ordering, not a value chain"
    );
    assert!(
        !g.reachable_between(&a, &b),
        "a--b is resource-only, not a value edge"
    );
    assert!(!g.reachable_between(&c, &a), "no back edge");
    assert_eq!(g.value_consumer_count(&a), 1);
    assert_eq!(g.value_consumer_count(&b), 0);
}

#[test]
fn unknown_ops_have_no_edges() {
    let a = op(0);
    let lonely = op(1);
    let g = SemanticGraph::new(&[dep(&a, &a, SemanticDependencyKind::Value)]);
    assert!(!g.conflicts(&lonely, &a));
    assert!(!g.reachable_between(&lonely, &a));
}

fn append_capturing_map(graph: &mut EGraph<Semantic>, id: u32, captures: Vec<NodeId>) {
    let ty = Type::Constructed(TypeName::Unit, vec![]);
    let result = graph.alloc_side_effect_result(ty.clone());
    let block = graph.skeleton.entry;
    graph.skeleton.blocks[block].side_effects.push(SideEffect {
        kind: SideEffectKind::Soac(
            op(id),
            Soac::Screma(screma::Op::Map {
                lanes: screma::Lanes {
                    inputs: vec![],
                    maps: vec![screma::Map {
                        body: SegBody {
                            region: RegionId::from_index(0),
                            captures,
                        },
                        input_indices: vec![],
                        output_element_type: ty.clone(),
                        destination: SoacDestination::fresh(),
                        result_type: ty,
                    }],
                },
                state: screma::SemanticState::Serial,
            }),
        ),
        operand_nodes: smallvec![],
        result: Some(result),
        effects: None,
        span: None,
    });
}

#[test]
fn scheduled_operations_expose_shared_prelude_inputs() {
    let mut egir = EGraph::<Semantic>::new();
    let source = egir.add_func_param(0, Type::Constructed(TypeName::Unit, vec![]));
    append_capturing_map(&mut egir, 10, vec![source, source]);
    append_capturing_map(&mut egir, 11, vec![source]);

    let graph = SemanticGraph::with_operation_captures(&[], &egir);
    assert_eq!(graph.operations().collect::<Vec<_>>(), vec![op(10), op(11)]);
    assert_eq!(
        graph.operation_captures(&op(10)).collect::<Vec<_>>(),
        vec![source]
    );
    assert_eq!(
        graph.operation_captures(&op(11)).collect::<Vec<_>>(),
        vec![source]
    );
    assert_eq!(graph.operation_site(&op(10)).map(|site| site.index), Some(0));
    assert_eq!(graph.operation_site(&op(11)).map(|site| site.index), Some(1));
}
