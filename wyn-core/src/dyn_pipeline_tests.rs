//! Proving-out tests for alternative TLC pass orderings, driven by `Driver`.
//!
//! These are exploratory: a panic mid-order is a *useful* result — it pinpoints
//! the first pass that an ordering violates. The proposed order under test runs
//! defunctionalize + monomorphize BEFORE fusion (the inverse of production), to
//! probe whether fusion can be made purely intraprocedural.

use super::Driver;
use crate::tlc::{SoacOp, Term, TermKind};

/// Does any subterm contain a `Screma` (the shape a fused `map → reduce` lowers
/// to)? Walks the whole term, including SOAC operator lambdas.
fn contains_screma(term: &Term) -> bool {
    if matches!(&term.kind, TermKind::Soac(SoacOp::Screma { .. })) {
        return true;
    }
    let mut found = false;
    term.for_each_child(&mut |c| found |= contains_screma(c));
    found
}

/// The front half of the candidate reordering: monomorphize the whole program
/// first, inline everything, THEN fuse — so fusion never has to reason across a
/// call. Stops right after fusion + DCE, which is where the thesis lives.
fn through_fusion(d: Driver) -> Driver {
    d.partial_eval()
        .normalize_soacs()
        // --- specialize + inline everything before fusion ---
        .defunctionalize()
        .monomorphize()
        .rep_specialize()
        .fold_generated_lambdas()
        .inline_small()
        .force_inline_soac_helpers()
        .normalize_soacs() // inlining exposed new tuple/zip/map structure
        // --- canonicalize, then fuse ---
        .if_over_producer()
        .fuse_maps()
        .reachable()
}

/// The full candidate order: front half + residency/indexing + finalize.
fn proposed_order(d: Driver) -> Driver {
    through_fusion(d)
        // --- residency / indexing ---
        .expose_entry_producers()
        .fuse_static_indices()
        .float_runtime_index_producers()
        .lift_gathers()
        .reachable()
        // --- finalize ---
        .normalize_outputs()
        .apply_ownership()
}

const MAP_REDUCE_SRC: &str = r#"
def globalArr: [4]f32 = [10.0, 20.0, 30.0, 40.0]

def myMap(ro: f32, rd: f32) [4]f32 =
  map(|x: f32| x + ro + rd, globalArr)

def myReduce(hits: [4]f32) f32 =
  reduce(|acc: f32, x: f32| if acc < x then acc else x, 999.0, hits)

#[fragment]
entry fragment_main() #[location(0)] vec4f32 =
  let hits = myMap(1.0, 2.0) in
  let closest = myReduce(hits) in
  @[closest, 0.0, 0.0, 1.0]
"#;

#[test]
fn proposed_order_runs_map_reduce_to_completion() {
    let d = proposed_order(Driver::from_source(MAP_REDUCE_SRC));
    eprintln!("final defs: {:?}", d.def_names());
    assert!(!d.program().defs.is_empty(), "pipeline produced an empty program");
}

/// The payoff probe: with monomorphize + inlining moved BEFORE fusion, the
/// cross-call `myMap` → `myReduce` should fuse *intraprocedurally* (both inlined
/// into `fragment_main` first), producing a `Screma` — no interprocedural
/// summary path required.
/// A polymorphic, higher-order, `Produces`-shaped helper (`myTab` = `map(f, 0..<n)`)
/// consumed by a `reduce`. In the PRODUCTION order this is the entangled case:
/// force-inline skips `myTab` (free type var `A`), so it survives to fusion and
/// only fuses via the interprocedural summary path. Under the proposed order it
/// is monomorphized + inlined BEFORE fusion, so — if the reorder holds — the
/// whole `reduce(map(f, 0..<n))` chain fuses intraprocedurally with no summary.
const POLY_HELPER_SRC: &str = r#"
def myTab<[n], A>(size: i32, f: i32 -> A) [n]A = map(f, 0..<size)

#[compute]
entry sum_tab(n: i32) f32 =
  reduce(|a: f32, b: f32| a + b, 0.0, myTab(n, |i: i32| f32.i32(i) * 2.0))
"#;

#[test]
fn proposed_order_fuses_polymorphic_higher_order_helper() {
    let d = proposed_order(Driver::from_source(POLY_HELPER_SRC));
    eprintln!("final defs: {:?}", d.def_names());
    let fused = d.program().defs.iter().any(|def| contains_screma(&def.body));
    assert!(
        fused,
        "expected the polymorphic helper to specialize+inline+fuse to a Screma; defs = {:?}",
        d.def_names()
    );
}

#[test]
fn proposed_order_fuses_map_reduce_intraprocedurally() {
    let d = proposed_order(Driver::from_source(MAP_REDUCE_SRC));
    let fused = d.program().defs.iter().any(|def| contains_screma(&def.body));
    assert!(fused, "expected a fused Screma after the reordered pipeline; defs = {:?}", d.def_names());
}
