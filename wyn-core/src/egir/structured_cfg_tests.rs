use super::*;
use crate::egir::types::{EffectOp, EffectToken, Language, Physical, SideEffectKind, WynLanguage};
use crate::ssa::types::ConstantValue;
use smallvec::smallvec;

fn effect(id: u32) -> SideEffect<Physical> {
    SideEffect::new(
        SideEffectKind::Effect(EffectOp::ControlBarrier),
        smallvec![],
        None,
        Some((EffectToken::from(id * 2), EffectToken::from(id * 2 + 1))),
        None,
    )
}

fn bool_value(graph: &mut EGraph<Physical>, value: bool) -> ValueId {
    graph.intern_constant(
        ConstantValue::Bool(value),
        Type::Constructed(TypeName::Bool, vec![]),
    )
}

#[test]
fn guarded_effect_replacement_preserves_suffix_terminator_and_control() {
    let mut graph = EGraph::<Physical>::new();
    let entry = graph.skeleton.entry;
    let old_then = graph.skeleton.create_block();
    let old_merge = graph.skeleton.create_block();
    let old_condition = bool_value(&mut graph, true);
    graph.skeleton.blocks[entry].side_effects = vec![effect(0), effect(1), effect(2)];
    install_selection(&mut graph, entry, old_condition, old_then, old_merge, old_merge);
    let new_condition = bool_value(&mut graph, false);

    let replacement = replace_effect_with_guarded_selection(
        &mut graph,
        SideEffectSite {
            block: entry,
            index: 1,
        },
        |_| Ok(new_condition),
    )
    .unwrap();

    assert_eq!(replacement.effect.effects(), effect(1).effects());
    assert_eq!(graph.skeleton.blocks[entry].side_effects.len(), 1);
    assert_eq!(
        graph.skeleton.blocks[replacement.continuation].side_effects.len(),
        1
    );
    assert!(matches!(
        graph.skeleton.blocks[entry].control_header,
        Some(ControlHeader::Selection { merge }) if merge == replacement.continuation
    ));
    assert!(matches!(
        graph.skeleton.blocks[replacement.continuation].control_header,
        Some(ControlHeader::Selection { merge }) if merge == old_merge
    ));
    assert!(matches!(
        graph.skeleton.blocks[replacement.continuation].term,
        SkeletonTerminator::CondBranch { cond, .. } if cond == old_condition
    ));
}

#[test]
fn counted_effect_replacement_keeps_continuation_and_control_in_sync() {
    let mut graph = EGraph::<Physical>::new();
    let entry = graph.skeleton.entry;
    graph.skeleton.blocks[entry].side_effects = vec![effect(0), effect(1)];
    graph.skeleton.blocks[entry].term = SkeletonTerminator::Return(None);
    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
    let initial = graph.intern_constant(ConstantValue::I32(7), i32_ty.clone());
    let trip_count = graph.intern_constant(ConstantValue::I32(4), i32_ty.clone());
    let mut exit = None;
    let mut effect_ids = IdSource::new();
    let carried = graph.value_result(initial);

    let loop_cfg = replace_effect_with_counted_loop(
        &mut graph,
        SideEffectSite {
            block: entry,
            index: 0,
        },
        &[carried],
        &[0],
        &mut effect_ids,
        |_| trip_count,
        |_, _, binding| {
            exit = binding.single_value();
            Ok(())
        },
    )
    .unwrap();

    assert!(exit.is_some());
    assert_eq!(graph.skeleton.blocks[loop_cfg.continuation].side_effects.len(), 1);
    assert!(matches!(
        graph.skeleton.blocks[loop_cfg.continuation].term,
        SkeletonTerminator::Return(None)
    ));
    assert!(matches!(
        graph.skeleton.blocks[loop_cfg.carried.block()].control_header,
        Some(ControlHeader::Loop { merge, continue_block })
            if merge == loop_cfg.continuation && continue_block == loop_cfg.body
    ));
    assert!(matches!(
        graph.skeleton.blocks[loop_cfg.carried.block()].term,
        SkeletonTerminator::CondBranch { then_target, else_target, .. }
            if then_target == loop_cfg.body && else_target == loop_cfg.continuation
    ));
    graph.skeleton.verify_branch_arities().unwrap();
}

#[test]
fn counted_loop_constructs_materialized_state_in_one_fixed_place() {
    let mut graph = EGraph::<Physical>::new();
    let entry = graph.skeleton.entry;
    graph.skeleton.blocks[entry].side_effects = vec![effect(0)];
    graph.skeleton.blocks[entry].term = SkeletonTerminator::Return(None);
    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
    let array_ty = Type::Constructed(
        TypeName::Array,
        vec![
            i32_ty.clone(),
            crate::types::array_variant_composite(),
            Type::Constructed(TypeName::Size(1), vec![]),
            crate::types::no_buffer(),
        ],
    );
    let zero = graph.intern_constant(ConstantValue::I32(0), i32_ty.clone());
    let initial = graph.intern_pure(PureOp::ArrayLit(1), smallvec![zero], array_ty, None);
    let trip_count = graph.intern_constant(ConstantValue::I32(4), i32_ty);
    let carried = graph.value_result(initial);
    let mut exit = None;
    let mut effect_ids = IdSource::new();

    let loop_cfg = replace_effect_with_counted_loop(
        &mut graph,
        SideEffectSite {
            block: entry,
            index: 0,
        },
        &[carried],
        &[0],
        &mut effect_ids,
        |_| trip_count,
        |_, _, binding| {
            exit = Some(binding.clone());
            Ok(())
        },
    )
    .unwrap();

    let carried = loop_cfg.carried.bindings()[0].result();
    let exit = exit.unwrap();
    assert_eq!(carried.places(), exit.places());
    assert_eq!(carried.places().len(), 1);
    assert!(graph.skeleton.blocks.values().all(|block| {
        block
            .params
            .iter()
            .all(|parameter| !WynLanguage::contains_materialized_flow(graph.value(parameter.value()).ty()))
    }));
}

#[test]
fn inline_effect_replacement_moves_structured_metadata_to_its_tail() {
    let mut graph = EGraph::<Physical>::new();
    let entry = graph.skeleton.entry;
    let old_then = graph.skeleton.create_block();
    let old_merge = graph.skeleton.create_block();
    let condition = bool_value(&mut graph, true);
    graph.skeleton.blocks[entry].side_effects = vec![effect(0), effect(1)];
    install_selection(&mut graph, entry, condition, old_then, old_merge, old_merge);
    let tail = graph.skeleton.create_block();

    let continuation = detach_effect_for_inline_replacement(
        &mut graph,
        SideEffectSite {
            block: entry,
            index: 0,
        },
    )
    .unwrap();
    restore_inline_effect_continuation(&mut graph, tail, continuation);

    assert!(graph.skeleton.blocks[entry].control_header.is_none());
    assert!(matches!(
        graph.skeleton.blocks[tail].control_header,
        Some(ControlHeader::Selection { merge }) if merge == old_merge
    ));
    assert_eq!(graph.skeleton.blocks[tail].side_effects.len(), 1);
    assert!(matches!(
        graph.skeleton.blocks[tail].term,
        SkeletonTerminator::CondBranch { cond, .. } if cond == condition
    ));
}
