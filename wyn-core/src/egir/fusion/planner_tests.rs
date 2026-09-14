use super::*;
use crate::egir::{self, fusion::emit};

fn entry_snapshot(source: &str) -> (Snapshot, Vec<GroupId>) {
    let tlc = crate::tlc::infer_input_slice_bounds(crate::compile_thru_tlc(source).unwrap());
    let program = egir::reify_soacs(crate::to_egraph(tlc).unwrap());
    let (snapshot, _) = Snapshot::build(&program).unwrap();
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
        .collect();
    (snapshot, groups)
}

#[test]
fn malformed_candidates_report_errors_without_changing_the_graph() {
    let (mut snapshot, groups) = entry_snapshot(
        r#"
entry siblings(xs: [4]i32) ([4]i32, [4]i32) =
  (map(|x: i32| x + 1, xs), map(|x: i32| x * 2, xs))
"#,
    );
    let [left, right] = groups[..] else {
        panic!("expected two entry operations, got {groups:?}");
    };
    let before = format!("{:?}", snapshot.graph);
    for (sources, expected) in [
        (vec![left, right], wyn_fusion::Error::Routing),
        (vec![left, left], wyn_fusion::Error::Membership),
        (vec![left], wyn_fusion::Error::Routing),
    ] {
        let proposal = Proposal {
            sources: sources.clone(),
            absorbed_values: vec![],
            results: vec![],
            accounted_constraints: vec![],
            payload: operation(&snapshot, left),
        };
        let error = apply_candidate(&mut snapshot, proposal).unwrap_err().to_string();
        assert!(error.contains(&expected.to_string()), "{error}");
        assert!(error.contains(&format!("{sources:?}")), "{error}");
        assert_eq!(format!("{:?}", snapshot.graph), before);
    }
}

#[test]
fn cycle_candidates_are_skipped_without_changing_the_graph() {
    let (mut snapshot, groups) = entry_snapshot(
        r#"
entry chain(xs: [4]i32) [4]i32 =
  let a = map(|x: i32| x + 1, xs) in
  let b = map(|x: i32| x * 2, a) in
  map(|x: i32| x - 3, b)
"#,
    );
    let [first, _, last] = groups[..] else {
        panic!("expected three entry operations, got {groups:?}");
    };
    let sources = vec![first, last];
    let boundary = snapshot.graph.boundary(&sources, &[]).unwrap();
    let proposal = Proposal {
        sources,
        absorbed_values: vec![],
        results: boundary.outputs.into_iter().map(|port| vec![port]).collect(),
        accounted_constraints: boundary.constraints,
        payload: operation(&snapshot, first),
    };
    let before = format!("{:?}", snapshot.graph);
    assert!(!apply_candidate(&mut snapshot, proposal).unwrap());
    assert_eq!(format!("{:?}", snapshot.graph), before);
}

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
    assert_eq!(form.layout().reduction_result_count(), 1);
    assert!(!form.pre.projectable(0..form.layout().operator_input_count()));
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
    assert_eq!(scremas[0].form.layout().reduction_result_count(), 2);
    assert!(scremas[0].form.scans.is_empty());
    scremas[0].validate().unwrap();
    let optimized = egir::optimize_semantic_operations(program).unwrap();
    let allocated = egir::lift_stage_uniform_values(optimized);
    let planned = egir::plan(allocated, crate::LoweringProfile::PORTABLE).unwrap();
    crate::lower_ssa_to_spirv(crate::lower_egir_to_ssa(planned).unwrap()).unwrap();
}
