use super::*;
use crate::compile_thru_tlc;
use crate::egir;
use crate::egir::soac::SegmentedMetadata;
use crate::egir::types::{SideEffectKind, Soac, SoacEffect};
use crate::lower_egir_to_ssa;
use crate::tlc;
use crate::to_egraph;
use crate::LoweringProfile;

fn reified(source: &str) -> egir::reify::Segmented {
    let program = compile_thru_tlc(source).expect("compile through TLC");
    let program = tlc::infer_input_slice_bounds(program);
    let program = to_egraph(program).expect("convert to raw EGIR");
    egir::reify_soacs(program)
}

fn force_horizontal_then_vertical(source: &str) -> egir::ResourcesAllocated {
    let program = reified(source);
    let function_count = program.functions.len();
    let (snapshot, catalog) = snapshot::Snapshot::build(&program).unwrap();
    let planned = planner::plan_with_priority(snapshot, None, &[5, 1, 0, 2, 3, 4]).unwrap();
    assert!(
        planned.plan.actions().len() >= 2,
        "horizontal and vertical actions must occur before emission"
    );
    assert_eq!(program.functions.len(), function_count);
    let (program, _) = emit::apply(program, planned, catalog).unwrap();
    let program = egir::optimize_semantic_operations(program).expect("semantic EGIR optimization failed");
    let optimized = egir::lift_stage_uniform_values(program);
    egir::plan_logical_resources(optimized).expect("allocate the vertically normalized Screma")
}

fn assert_screma_and_lower(allocated: egir::ResourcesAllocated, scans: usize) {
    let scremas = allocated
        .data
        .stages
        .stages()
        .map(|(_, stage)| stage.body())
        .flat_map(|entry| entry.graph.skeleton.blocks.iter().flat_map(|(_, block)| &block.side_effects))
        .filter_map(|effect| {
            let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &effect.kind else {
                return None;
            };
            Some(op)
        })
        .collect::<Vec<_>>();
    assert_eq!(scremas.len(), 1);
    assert_eq!(scremas[0].form.layout().scan_input_count(), scans);
    assert_eq!(scremas[0].form.layout().reduction_result_count(), 1);
    assert_eq!(scremas[0].form.post.result_types.len(), 1);
    assert!(scremas[0].validate().is_ok());

    let planned =
        egir::plan(allocated, LoweringProfile::PORTABLE).expect("plan the vertically normalized Screma");
    lower_egir_to_ssa(planned).expect("lower the vertically normalized Screma");
}

#[test]
fn cross_barrier_projection_handles_conditional_lambda_results() {
    let program = reified(
        r#"
entry scan_map_reduce(xs: [4]i32) ([4]i32, i32) =
  let prefixes = scan(|a: i32, b: i32| a + b, 0, xs) in
  let values = map(|x: i32| x + 1, xs) in
  let paired = map(
    |(prefix, x): (i32, i32)| (if x < 0 then 0 - x else x, prefix * 2),
    zip(prefixes, values)
  ) in
  let (magnitudes, doubled_prefixes) = unzip(paired) in
  let total = reduce(|a: i32, b: i32| a + b, 0, magnitudes) in
  (doubled_prefixes, total)
"#,
    );
    let (snapshot, catalog) = snapshot::Snapshot::build(&program).unwrap();
    let planned = planner::plan(snapshot, None).unwrap();
    assert_eq!(
        planned.plan.groups().count(),
        1,
        "final symbolic operations: {:#?}",
        planned.plan.groups().map(|(_, group)| group.payload()).collect::<Vec<_>>()
    );
    let (fused, _) = emit::apply(program, planned, catalog).expect("fusion application");

    let scremas = fused
        .entry_points
        .iter()
        .flat_map(|entry| entry.graph.skeleton.blocks.iter().flat_map(|(_, block)| &block.side_effects))
        .filter_map(|effect| {
            let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &effect.kind else {
                return None;
            };
            Some(op)
        })
        .collect::<Vec<_>>();
    assert_eq!(
        scremas.len(),
        1,
        "independent collective work crosses the scan barrier:\n{}",
        egir::semantic_graph::summary(&fused)
    );
    assert_eq!(scremas[0].form.scan_count(), 1);
    assert_eq!(scremas[0].form.reduction_count(), 1);
    assert_eq!(scremas[0].form.post.result_types.len(), 1);
    assert!(scremas[0].validate().is_ok());
    assert!(
        fused.functions.iter().any(
            |function| function.name.contains("projection") && function.graph.skeleton.blocks.len() > 1
        ),
        "conditional result projection synthesizes an explicit CFG helper"
    );

    let optimized: egir::Optimized = fused.retag();
    let allocated =
        egir::plan_logical_resources(optimized).expect("allocate the cross-barrier conditional Screma");
    let planned = egir::plan(allocated, LoweringProfile::PORTABLE)
        .expect("plan the cross-barrier conditional Screma");
    lower_egir_to_ssa(planned).expect("lower the cross-barrier conditional Screma");
}
#[test]
fn vertical_normalization_accepts_a_reduction_bearing_producer() {
    let allocated = force_horizontal_then_vertical(
        r#"
entry redomap_then_map<[n]>(xs: [n]i32) (i32, [n]i32) =
  let mapped = map(|x: i32| x + 1, xs) in
  let total = reduce(|a: i32, b: i32| a + b, 0, xs) in
  let consumed = map(|x: i32| x * 2, mapped) in
  (total, consumed)
"#,
    );
    assert_screma_and_lower(allocated, 0);
}

#[test]
fn vertical_normalization_accepts_a_scan_and_reduction_producer() {
    let allocated = force_horizontal_then_vertical(
        r#"
entry scan_redomap_then_map<[n]>(xs: [n]i32) (i32, [n]i32) =
  let prefixes = scan(|a: i32, b: i32| a + b, 0, xs) in
  let total = reduce(|a: i32, b: i32| a + b, 0, xs) in
  let consumed = map(|x: i32| x * 2, prefixes) in
  (total, consumed)
"#,
    );
    assert_screma_and_lower(allocated, 1);
}

#[test]
fn snapshot_preserves_opaque_barriers_without_effect_tokens() {
    use crate::egir::types::{EffectOp, SideEffect};
    let mut program = reified(
        r#"entry chain(xs: [4]i32) [4]i32 =
      let a = map(|x: i32| x + 1, xs) in map(|x: i32| x * 2, a)"#,
    );
    let graph = &mut program.entry_points[0].graph;
    graph.skeleton.blocks[graph.skeleton.entry].side_effects.insert(
        1,
        SideEffect {
            kind: SideEffectKind::Effect(EffectOp::ControlBarrier),
            operands: smallvec::smallvec![],
            result: None,
            effects: None,
            span: None,
        },
    );
    let (snapshot, _) = snapshot::Snapshot::build(&program).unwrap();
    let planned = planner::plan(snapshot, None).unwrap();
    assert!(planned.plan.actions().is_empty());
    assert_eq!(planned.plan.groups().count(), 3);
}

#[test]
fn snapshot_keeps_cross_block_uses_as_external_observers() {
    use crate::egir::types::SkeletonTerminator;
    let mut program = reified(
        r#"entry chain(xs: [4]i32) [4]i32 =
      let a = map(|x: i32| x + 1, xs) in map(|x: i32| x * 2, a)"#,
    );
    let graph = &mut program.entry_points[0].graph;
    let entry = graph.skeleton.entry;
    let next = graph.skeleton.create_block();
    let consumer = graph.skeleton.blocks[entry].side_effects.pop().unwrap();
    graph.skeleton.blocks[next].side_effects.push(consumer);
    graph.skeleton.blocks[next].term = graph.skeleton.blocks[entry].term.clone();
    graph.skeleton.blocks[entry].term = SkeletonTerminator::Branch {
        target: next,
        args: vec![],
    };
    for route in program.entry_points[0].routes_mut() {
        route.source.block = next;
    }
    let (snapshot, _) = snapshot::Snapshot::build(&program).unwrap();
    let groups = snapshot.graph.order().unwrap();
    assert_eq!(
        snapshot.graph.boundary(&groups, &[]),
        Err(wyn_fusion::Error::Scope)
    );
    assert!(!snapshot.graph.boundary(&[groups[0]], &[]).unwrap().outputs.is_empty());
    assert!(planner::plan(snapshot, None).unwrap().plan.actions().is_empty());
}

#[test]
fn snapshot_qualifies_values_by_body_and_accepts_loop_parameters() {
    let program = reified(
        r#"
entry first(xs: [4]i32) [4]i32 = map(|x: i32| x + 1, xs)
entry second(xs: [4]i32) [4]i32 =
  loop values = xs for i < 3 do map(|x: i32| x + 1, values)
"#,
    );
    let (snapshot, catalog) = snapshot::Snapshot::build(&program).unwrap();
    let sources = catalog.values.values().collect::<Vec<_>>();
    assert!(sources
        .iter()
        .any(|left| sources.iter().any(|right| left.value == right.value && left.body != right.body)));
    assert!(snapshot.graph.order().is_ok());
    for (id, group) in snapshot.graph.groups() {
        for input in group.inputs() {
            assert!(!snapshot.graph.producers(*input).unwrap().contains(&id));
        }
    }
}

#[test]
fn horizontal_fusion_rejects_input_resource_write_hazards() {
    use crate::egir::types::ResourceAccess;
    let mut program = reified(
        r#"entry siblings(xs: []i32) ([]i32, []i32) =
      (map(|x: i32| x + 1, xs), map(|x: i32| x * 2, xs))"#,
    );
    let graph = &mut program.entry_points[0].graph;
    let effect = &mut graph.skeleton.blocks[graph.skeleton.entry].side_effects[0];
    let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &mut effect.kind else {
        panic!("map");
    };
    let crate::egir::soac::screma::SemanticState::Segmented(SegmentedMetadata { resources, .. }) =
        &mut op.state
    else {
        panic!("segmented map");
    };
    resources.iter_mut().find(|access| access.access == ResourceAccess::Read).unwrap().access =
        ResourceAccess::ReadWrite;
    let (snapshot, _) = snapshot::Snapshot::build(&program).unwrap();
    assert!(planner::plan(snapshot, None).unwrap().plan.actions().is_empty());
}

#[test]
fn indexed_tuple_fields_keep_distinct_boundaries() {
    let program = reified(
        r#"
entry tuple_item(xs: [4]i32) (i32, i32) =
  let paired = map(|x: i32| (x + 1, [x, x * 2, x * 3]), xs) in
  let (a, ys) = paired[2] in (a, ys[1])
"#,
    );
    let (snapshot, catalog) = snapshot::Snapshot::build(&program).unwrap();
    let fields = snapshot
        .values
        .values()
        .filter_map(|fact| match &fact.usage {
            snapshot::Use::Project { base, path } => Some((*base, path.clone())),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert!(fields
        .iter()
        .any(|(base, path)| fields.iter().any(|(other, other_path)| base == other && path != other_path)));
    let planned = planner::plan(snapshot, None).unwrap();
    assert_eq!(planned.plan.actions().len(), 1);
    assert!(planned.plan.groups().any(|(_, group)| matches!(&group.payload().kind, snapshot::Kind::Indexed { demands, .. } if demands.iter().any(|(_, _, path)| !path.is_empty()))));
    let (program, _) = emit::apply(program, planned, catalog).unwrap();
    let optimized = egir::optimize_semantic_operations(program).unwrap();
    let allocated = egir::plan_logical_resources(egir::lift_stage_uniform_values(optimized)).unwrap();
    let planned = egir::plan(allocated, LoweringProfile::PORTABLE).unwrap();
    lower_egir_to_ssa(planned).unwrap();
}

#[test]
fn fused_scan_allocates_output_when_its_unique_input_is_absorbed() {
    let program = reified(include_str!("../../../../testfiles/gather_scan_chain.wyn"));
    let (snapshot, catalog) = snapshot::Snapshot::build(&program).unwrap();
    let planned = planner::plan(snapshot, None).unwrap();
    assert_eq!(planned.plan.actions().len(), 1);
    let scan = planned
        .plan
        .groups()
        .find_map(|(_, group)| {
            let operation = group.payload();
            matches!(&operation.kind, snapshot::Kind::Screma(form) if !form.scans.is_empty())
                .then_some(operation)
        })
        .unwrap();
    assert!(scan.result_state.iter().all(|state| state.ownership == crate::types::SoacOwnership::Fresh));
    let (program, _) = emit::apply(program, planned, catalog).unwrap();
    let optimized = egir::optimize_semantic_operations(program).unwrap();
    let allocated = egir::plan_logical_resources(egir::lift_stage_uniform_values(optimized)).unwrap();
    let planned = egir::plan(allocated, LoweringProfile::PORTABLE).unwrap();
    lower_egir_to_ssa(planned).unwrap();
}
