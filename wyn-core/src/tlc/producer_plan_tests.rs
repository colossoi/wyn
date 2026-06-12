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
fn filter_over_static_input_plans_bounded_aggregate() {
    // A filter whose input is a statically-sized array is a fixed-capacity
    // Bounded aggregate — the same decision `rep_specialize` executes as
    // `ConcreteVariant::Bounded { capacity }`, sourced from the shared
    // `filter_variant`. The 5-element literal ANF-binds to a `[5]i32` Ref.
    let (program, plans) = plan_src(
        r#"
def sum<[n]>(arr: [n]i32) i32 = reduce(|a: i32, b: i32| a + b, 0, arr)
def center(arr: []i32) i32 = sum(arr)
#[compute]
entry tick() i32 =
    let kept = filter(|x: i32| x > 0, [1, -2, 3, -4, 5]) in
    center(kept)
"#,
    );
    let p = producer(&program, &plans, "kept");
    assert!(p.produces_array, "`kept` is a filter result array");
    assert_eq!(p.strategy, Strategy::BoundedAggregate { capacity: 5 });
}

#[test]
fn filter_over_runtime_input_plans_view() {
    // A filter over a runtime-sized storage input has no static capacity —
    // it's a `{offset,len}` View (and `rep_specialize` leaves the size slot
    // Skolem). Distinguishes the View case from Bounded above.
    let (program, plans) = plan_src(
        r#"
def sum<[n]>(arr: [n]i32) i32 = reduce(|a: i32, b: i32| a + b, 0, arr)
def center(arr: []i32) i32 = sum(arr)
#[compute]
entry tick(#[storage(set=2, binding=0, access=read)] xs: []i32) i32 =
    let kept = filter(|x: i32| x > 0, xs) in
    center(kept)
"#,
    );
    let p = producer(&program, &plans, "kept");
    assert!(p.produces_array, "`kept` is a filter result array");
    assert_eq!(p.strategy, Strategy::View);
}

#[test]
fn filter_over_unreadable_input_yields_no_variant_not_view() {
    // A filter input whose array type can't be read (a fused-chain ArrayExpr,
    // not a Ref/StorageView) has no producer-derived variant: `filter_variant`
    // returns None. The planner maps that to `Strategy::LeaveAsIs` and
    // `rep_specialize`'s `?` maps it to "no specialization" — report ==
    // executor. Reporting `View` here (the prior bug) would disagree with the
    // executor, which does nothing for this case.
    assert_eq!(filter_variant(&ArrayExpr::Literal(vec![])), None);
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
