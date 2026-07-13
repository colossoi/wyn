use super::*;
use crate::ast::Span;
use crate::egir::program::{OutputRoute, OutputSlotId, SlotSource};
use crate::egir::types::EffectToken;
use crate::ssa::types::ExecutionModel;
use crate::tlc::ScremaAccumulator;

/// Region indices used by the operator fixtures below: step is region 0,
/// combine is region 1. Opaque indices stand in for the named regions a
/// real conversion would intern.
const STEP_REGION: RegionId = RegionId::from_index(0);
const COMBINE_REGION: RegionId = RegionId::from_index(1);

fn accumulator(
    kind: ScremaAccumulator,
    step_captures: Vec<NodeId>,
    combine_captures: Vec<NodeId>,
) -> ScremaOperator {
    ScremaOperator {
        kind,
        step: SegBody {
            region: STEP_REGION,
            captures: step_captures,
        },
        combine: SegBody {
            region: COMBINE_REGION,
            captures: combine_captures,
        },
        input_indices: vec![],
    }
}

fn neutral(graph: &mut EGraph, index: usize) -> NodeId {
    graph.add_func_param(index, Type::Constructed(TypeName::Unit, vec![]))
}

#[test]
fn output_ownership_comes_from_explicit_route_writer() {
    let mut graph = EGraph::new();
    let block = graph.skeleton.entry;
    let source = neutral(&mut graph, 0);
    let writer = EffectToken(9);
    graph.skeleton.blocks[block].side_effects.push(SideEffect {
        semantic_id: None,
        kind: SideEffectKind::Inst(InstKind::Store {
            place: Default::default(),
            value: crate::ssa::types::ValueRef::Ssa(Default::default()),
        }),
        operand_nodes: smallvec![],
        result: None,
        effects: Some((EffectToken(8), writer)),
        span: None,
    });
    let mut entry = SemanticEntry::new(
        crate::interface::EntryOrigin::Source,
        "route_test".into(),
        Span::dummy(),
        ExecutionModel::Compute {
            local_size: (1, 1, 1),
        },
        vec![],
        vec![],
        vec![],
        vec![],
        Type::Constructed(TypeName::Unit, vec![]),
        graph,
        LookupMap::new(),
    );
    entry.output_routes.push(OutputRoute {
        source: SlotSource { block, value: source },
        slot: OutputSlotId(3),
        writers: vec![OutputWriter::Effect(writer)],
    });

    assert_eq!(
        side_effect_output_slots(&entry, &entry.graph.skeleton.blocks[block].side_effects[0]),
        vec![3]
    );
}

#[test]
fn reduction_accumulator_reifies_as_seg_red_operator() {
    let mut graph = EGraph::new();
    let ne = neutral(&mut graph, 0);
    let kind = reify_seg_kind(
        &[accumulator(ScremaAccumulator::Reduce, vec![ne], vec![ne, ne])],
        &[ne],
        1,
    );
    let SegOpKind::SegRed { operators } = kind else {
        panic!("reduction must reify as SegRed")
    };
    assert_eq!(operators.len(), 1);
    assert_eq!(operators[0].step.region, STEP_REGION);
    assert_eq!(operators[0].combine.region, COMBINE_REGION);
    assert_eq!(operators[0].neutral, ne);
    assert!(operators[0].shape.is_empty());
    assert_eq!(operators[0].step.captures, vec![ne]);
    assert_eq!(operators[0].combine.captures, vec![ne, ne]);
    assert!(
        !operators[0].commutative,
        "Wyn does not yet declare commutativity"
    );
}

#[test]
fn scan_accumulator_reifies_as_seg_scan_operator() {
    let mut graph = EGraph::new();
    let ne = neutral(&mut graph, 0);
    let kind = reify_seg_kind(&[accumulator(ScremaAccumulator::Scan, vec![], vec![])], &[ne], 1);
    assert!(matches!(kind, SegOpKind::SegScan { operators } if operators.len() == 1));
}

#[test]
fn mixed_reduce_and_scan_stays_serial_until_joint_scheduler_exists() {
    let accumulators = [
        accumulator(ScremaAccumulator::Reduce, vec![], vec![]),
        accumulator(ScremaAccumulator::Scan, vec![], vec![]),
    ];
    let mut graph = EGraph::new();
    let neutrals = [neutral(&mut graph, 0), neutral(&mut graph, 1)];
    assert!(matches!(
        reify_seg_kind(&accumulators, &neutrals, 1),
        SegOpKind::SegComposite { operators } if operators.len() == 2
    ));
}

#[test]
fn idle_chunk_start_is_clamped_before_remaining_subtraction() {
    let mut graph = EGraph::new();
    let len = graph.add_func_param(0, Type::Constructed(TypeName::UInt(32), vec![]));
    let (_, start, _) =
        emit_chunk_arithmetic(&mut graph, REDUCE_PHASE1_WIDTH, len).expect("u32 chunk arithmetic");
    assert!(matches!(
        &graph.nodes[start],
        super::super::types::ENode::Pure {
            op: PureOp::Intrinsic { .. },
            operands,
        } if operands.as_slice().last() == Some(&len)
    ));
}

#[test]
fn scan_phase2_writes_exclusive_prefix_before_combining_current_block() {
    let elem_ty = Type::Constructed(TypeName::Int(32), vec![]);
    let mut phase1 = EGraph::new();
    let neutral = phase1.intern_pure(PureOp::Int("0".into()), smallvec![], elem_ty.clone());
    let sums = BindingRef::new(0, 40);
    let offsets = BindingRef::new(0, 41);
    let phase2 = synthesize_phase2_scan(
        "prefix",
        "combine".into(),
        elem_ty,
        &phase1,
        neutral,
        sums,
        offsets,
        None,
    )
    .expect("phase2 synthesis");

    let stored_value = phase2
        .graph
        .skeleton
        .blocks
        .iter()
        .flat_map(|(_, block)| &block.side_effects)
        .find_map(|effect| {
            if !matches!(effect.kind, SideEffectKind::Inst(InstKind::Store { .. })) {
                return None;
            }
            let place = *effect.operand_nodes.first()?;
            (storage_binding_under(&phase2.graph, place) == Some(offsets)).then(|| effect.operand_nodes[1])
        })
        .expect("block-offset store");
    assert!(matches!(
        phase2.graph.nodes[stored_value],
        super::super::types::ENode::BlockParam { .. }
    ));
}
