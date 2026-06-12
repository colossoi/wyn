//! Tests for the report-only producer-consumer planner: assert the strategy
//! chosen for the producer in each anchor shape.

use super::*;
use crate::Compiler;

/// Compile a source string down to the TLC program at the planner's pipeline
/// position (after `materialize_entry_soacs`, before `rep_specialize`), so the
/// plan is built on exactly the IR the real pass would see.
fn plan_src(source: &str) -> (Program, Vec<EntryPlan>) {
    let (mut node_counter, mut module_manager) = crate::cached_compiler_init();
    let parsed = Compiler::parse(source, &mut node_counter).expect("parse");
    let tc = parsed
        .resolve(&mut module_manager)
        .expect("resolve")
        .fold_ast_constants()
        .type_check(&mut module_manager)
        .expect("type_check");
    // Drive the typed-TLC pipeline up to the planner's slot (after
    // `materialize_entry_soacs`, before `rep_specialize`).
    let late = tc
        .to_tlc(&module_manager, false)
        .pin_entry_regions()
        .expect("pin_entry_regions")
        .partial_eval()
        .normalize_soacs()
        .fuse_maps()
        .apply_ownership()
        .expect("apply_ownership")
        .normalize_outputs()
        .expect("normalize_outputs")
        .lift_gathers()
        .defunctionalize()
        .monomorphize()
        .fold_generated_lambdas()
        .inline_small()
        .materialize_entry_soacs();
    let program = late.0.tlc.clone();
    let plans = plan_program(&program);
    (program, plans)
}

/// The planned producer whose let-binding has the given `name` (matched by the
/// binding symbol's name, since the pipeline may introduce several symbols that
/// share a source name).
fn producer<'a>(program: &Program, plans: &'a [EntryPlan], name: &str) -> &'a PlannedProducer {
    plans
        .iter()
        .flat_map(|p| &p.producers)
        .find(|pp| pp.binding.and_then(|s| program.symbols.get(s)).map(|n| n.as_str()) == Some(name))
        .unwrap_or_else(|| panic!("no planned producer bound to a symbol named `{name}`"))
}

#[test]
fn scalar_reduce_captured_in_map_is_scalar_broadcast() {
    let (program, plans) = plan_src(
        r#"
#[compute]
entry e(xs: []i32) []i32 =
    let s = reduce(|x: i32, y: i32| x + y, 0, xs) in
    map(|i: i32| i + s, iota(2048))
"#,
    );
    let p = producer(&program, &plans, "s");
    assert!(!p.produces_array, "`s` is a scalar reduce");
    assert_eq!(p.demand, Demand::PerElement, "`s` is read per element in the map");
    assert_eq!(p.strategy, Strategy::StoragePrepass(PrepassKind::ScalarBroadcast));
}

#[test]
fn two_scalar_reduces_captured_in_map_each_broadcast() {
    // Both reduces are scalars consumed per element inside the tail map — each
    // should hoist to its own one-element prepass buffer. This is the shape the
    // `aspiration_two_scalar_reduces_in_let_rhs_should_each_hoist` test wants
    // the planner to drive in Stage 3.
    let (program, plans) = plan_src(
        r#"
#[compute]
entry e(xs: []i32) []i32 =
    let s = reduce(|x: i32, y: i32| x + y, 0, xs) in
    let p = reduce(|x: i32, y: i32| x * y, 1, xs) in
    map(|i: i32| i + s + p, iota(2048))
"#,
    );
    for name in ["s", "p"] {
        let pp = producer(&program, &plans, name);
        assert!(!pp.produces_array, "`{name}` is a scalar reduce");
        assert_eq!(
            pp.demand,
            Demand::PerElement,
            "`{name}` is read per element in the map"
        );
        assert_eq!(
            pp.strategy,
            Strategy::StoragePrepass(PrepassKind::ScalarBroadcast),
            "`{name}` should be a scalar-broadcast prepass"
        );
    }
}
