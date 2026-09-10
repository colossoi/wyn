use super::*;
use crate::egir::{self, fusion::emit};

fn contract(snapshot: &mut Snapshot, sources: Vec<GroupId>, composition: Composition) -> GroupId {
    let boundary = snapshot.graph.boundary(&sources, &composition.absorbed).unwrap();
    snapshot
        .graph
        .contract(Proposal {
            sources,
            absorbed_values: composition.absorbed,
            results: composition.results,
            accounted_constraints: boundary.constraints,
            payload: composition.operation,
        })
        .unwrap()
        .target
}

#[test]
fn reduction_producer_fuses_into_unprojectable_collective_pre() {
    let source = r#"
entry nested_redomap(xs: [4]i32) (i32, i32) =
  let mapped = map(|x: i32| x + 1, xs) in
  let total = reduce(|a: i32, b: i32| a + b, 0, xs) in
  let nested = map(|x: i32| reduce(|a: i32, b: i32| a + b, 0, 0 ..< x), mapped) in
  let consumed = reduce(|a: i32, b: i32| a + b, 0, nested) in
  (total, consumed)
"#;
    let tlc = crate::tlc::infer_input_slice_bounds(crate::compile_thru_tlc(source).unwrap());
    let program = egir::reify_soacs(crate::to_egraph(tlc).unwrap());
    let function_count = program.functions.len();
    let (mut snapshot, catalog) = Snapshot::build(&program).unwrap();
    let groups = snapshot
        .graph
        .order()
        .unwrap()
        .into_iter()
        .filter(|id| {
            matches!(
                snapshot.graph.group(*id).unwrap().scope().0,
                egir::ir::BodySite::Entry(_)
            )
        })
        .collect::<Vec<_>>();
    let [mapped, total, nested, consumed] = groups[..] else {
        panic!("expected four entry operations, got {groups:?}");
    };
    let mut recipes = std::mem::take(&mut snapshot.recipes);
    let producer = horizontal(&snapshot, &mut recipes, mapped, total).unwrap();
    let producer = contract(&mut snapshot, vec![mapped, total], producer);
    let consumer = vertical(&snapshot, &mut recipes, nested, consumed).unwrap();
    let consumer = contract(&mut snapshot, vec![nested, consumed], consumer);
    let Kind::Screma(form) = &snapshot.graph.group(consumer).unwrap().payload().kind else {
        panic!("expected composed consumer Screma");
    };
    assert_eq!(form.reduction_result_count(), 1);
    assert!(!form.pre.projectable(0..form.operator_input_count()));
    snapshot.recipes = recipes;

    let planned = plan(snapshot, None).unwrap();
    assert!(planned
        .plan
        .actions()
        .iter()
        .any(|action| action.sources.contains(&producer) && action.sources.contains(&consumer)));
    assert_eq!(program.functions.len(), function_count);
    let (program, _) = emit::apply(program, planned, catalog).unwrap();
    let scremas = program.entry_points[0]
        .graph
        .skeleton
        .blocks
        .iter()
        .flat_map(|(_, block)| &block.side_effects)
        .filter_map(|effect| match &effect.kind {
            egir::types::SideEffectKind::Soac(egir::types::SoacEffect(
                _,
                egir::types::Soac::Screma(op),
            )) => Some(op),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(scremas.len(), 1);
    assert_eq!(scremas[0].form.reduction_result_count(), 2);
    assert!(scremas[0].form.scans.is_empty());
    scremas[0].validate().unwrap();
    let optimized = egir::optimize_semantic_operations(program).unwrap();
    let allocated = egir::plan_logical_resources(egir::lift_stage_uniform_values(optimized)).unwrap();
    let planned = egir::plan(allocated, crate::LoweringProfile::PORTABLE).unwrap();
    crate::lower_ssa_to_spirv(crate::lower_egir_to_ssa(planned).unwrap()).unwrap();
}
