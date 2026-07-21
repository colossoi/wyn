use super::*;
use crate::ast::Span;
use crate::egir::program::SemanticEntry;
use crate::egir::types::{EffectOp, SideEffect, SkeletonTerminator};
use crate::ssa::types::ConstantValue;

fn bool_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Bool, vec![])
}

fn unit_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Unit, vec![])
}

fn u32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::UInt(32), vec![])
}

fn entry_with_graph(graph: EGraph) -> SemanticEntry {
    SemanticEntry::new_with_resources(
        "terminator_live_out".into(),
        Span::dummy(),
        ExecutionModel::Compute {
            local_size: (1, 1, 1),
        },
        vec![],
        vec![],
        vec![],
        vec![],
        unit_ty(),
        graph,
        crate::LookupMap::new(),
    )
}

#[test]
fn retained_terminator_observes_selected_effect_result() {
    let mut graph = EGraph::new();
    let entry = graph.skeleton.entry;
    let continuation = graph.skeleton.create_block();
    let then_block = graph.skeleton.create_block();
    let else_block = graph.skeleton.create_block();
    let place = graph.intern_constant(ConstantValue::U32(0), u32_ty());
    let produced = graph.alloc_side_effect_result(bool_ty());
    graph.skeleton.blocks[entry].side_effects.push(SideEffect {
        kind: SideEffectKind::Effect(EffectOp::Load),
        operand_nodes: smallvec::smallvec![place],
        result: Some(produced),
        effects: Some((EffectToken::from(0), EffectToken::from(1))),
        span: None,
    });
    graph.skeleton.blocks[entry].term = SkeletonTerminator::Branch {
        target: continuation,
        args: vec![],
    };
    graph.skeleton.blocks[continuation].term = SkeletonTerminator::CondBranch {
        cond: produced,
        then_target: then_block,
        then_args: vec![],
        else_target: else_block,
        else_args: vec![],
    };
    graph.skeleton.blocks[then_block].term = SkeletonTerminator::Return(None);
    graph.skeleton.blocks[else_block].term = SkeletonTerminator::Return(None);
    let entry = entry_with_graph(graph);
    let producer_effects = HashSet::from([SideEffectSite {
        block: entry.graph.skeleton.entry,
        index: 0,
    }]);
    let retained = retained_prelude_terminator_blocks(
        &entry,
        &crate::egir::graph_projector::ValueRecipeSource::StructuredPrefix { continuation },
    );

    assert!(!retained.contains(&entry.graph.skeleton.entry));
    assert!(retained.contains(&continuation));
    assert!(value_is_observed_outside(
        &entry,
        produced,
        &producer_effects,
        &retained,
    ));
    assert!(!value_is_observed_outside(
        &entry,
        produced,
        &producer_effects,
        &HashSet::new(),
    ));
}
