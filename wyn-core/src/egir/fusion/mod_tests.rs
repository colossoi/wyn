use super::*;
use crate::compile_thru_tlc;
use crate::egir;
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
    let dependencies = egir::semantic_graph::dependencies(&program);
    let oracle = SemanticGraph::new(&dependencies);
    let horizontal = horizontal::analyze(&program, &oracle)
        .expect("the sibling collective and array producer should fuse horizontally");
    let program = horizontal::apply(program, horizontal).expect("horizontal fusion failed");

    let producer = program
        .entry_points
        .iter()
        .flat_map(|entry| entry.graph.skeleton.blocks.iter().flat_map(|(_, block)| &block.side_effects))
        .find_map(|effect| {
            let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &effect.kind else {
                return None;
            };
            (!op.form.reductions.is_empty() && !op.form.post.result_types.is_empty()).then_some(op)
        })
        .expect("horizontal fusion should construct a reduction-bearing producer");
    assert_eq!(producer.form.reduction_result_count(), 1);
    assert_eq!(producer.form.post.result_types.len(), 1);

    let dependencies = egir::semantic_graph::dependencies(&program);
    let oracle = SemanticGraph::new(&dependencies);
    let vertical = vertical::analyze(&program, &oracle).unwrap_or_else(|| {
            panic!(
                "the reduction-bearing producer should fuse into its map consumer:\n{}\ndependencies: {dependencies:#?}",
                egir::semantic_graph::summary(&program)
            )
        });
    let program = vertical::apply(program, vertical).expect("vertical fusion failed");
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
    assert_eq!(scremas[0].form.scan_input_count(), scans);
    assert_eq!(scremas[0].form.reduction_result_count(), 1);
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
    let mut fused = program;
    loop {
        let dependencies = egir::semantic_graph::dependencies(&fused);
        let oracle = SemanticGraph::new(&dependencies);
        let (next, changed) = rewrite_once(fused, &oracle).expect("fusion rewrite failed");
        fused = next;
        if !changed {
            break;
        }
    }

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
        fused.functions.iter().any(|function| function.name.contains("vertical_middle_consumer_pre")),
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
