use super::*;
use crate::ast::TypeName;
use crate::egir::program::SemanticOpId;
use crate::egir::soac::screma;
use crate::egir::types::{PureOp, SegSpace, SideEffect, Soac, SoacEffect, SoacInputType, SoacOwnership};
use polytype::Type;
use smallvec::smallvec;

#[test]
fn unreachable_project_does_not_keep_dead_segop_alive() {
    let mut graph = EGraph::new();
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
                        crate::egir::types::SegBody {
                            region: crate::egir::types::RegionId::from_index(0),
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
                    space: SegSpace::new(crate::egir::types::SegExtent::Fixed(1)),
                    placement: screma::Placement::LaneLocal,
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
    assert!(eliminate_dead_seg_ops_in_graph(&mut graph, []));
    assert!(graph.skeleton.blocks[graph.skeleton.entry].side_effects.is_empty());
}
