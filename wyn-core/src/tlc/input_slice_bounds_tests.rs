//! Unit tests for the TLC slice-bound inference. See module-level docs
//! on `input_slice_bounds` for the contract.

use crate::pipeline_descriptor::BufferLen;
use crate::tlc::{self, DefMeta, Program};

fn program_from(src: &str) -> Program<tlc::stage::InputSliceBoundsInferred> {
    tlc::infer_input_slice_bounds(crate::test_pipeline::compile_to_reachable(src))
}

fn bounds_for_entry<'a>(
    program: &'a Program<tlc::stage::InputSliceBoundsInferred>,
    name: &str,
) -> &'a crate::LookupMap<crate::SymbolId, BufferLen> {
    let def = program
        .defs
        .iter()
        .find(|def| program.symbols.get(def.name).is_some_and(|def_name| def_name == name))
        .unwrap_or_else(|| panic!("missing entry {name}"));
    let DefMeta::EntryPoint(entry) = &def.meta else {
        panic!("{name} is not an entry");
    };
    &entry.data.by_symbol
}

fn collect_term_ids<C: crate::tlc::Payload, S: crate::tlc::Payload>(
    term: &crate::tlc::Term<C, S>,
    ids: &mut Vec<crate::tlc::TermId>,
) {
    ids.push(term.id);
    term.for_each_child(&mut |child| collect_term_ids(child, ids));
}

#[test]
fn transition_reuses_entry_term_tree() {
    let src = r#"
#[compute]
entry e(#[storage(set=2, binding=0, access=read)] xs: []i32) i32 =
  length(xs)
"#;
    let reachable = crate::test_pipeline::compile_to_reachable(src);
    let mut before = Vec::new();
    for def in &reachable.defs {
        collect_term_ids(&def.body, &mut before);
    }

    let bounded = tlc::infer_input_slice_bounds(reachable);
    let mut after = Vec::new();
    for def in &bounded.defs {
        collect_term_ids(&def.body, &mut after);
    }

    assert_eq!(after, before, "entry-data transition must not rebuild term nodes");
}

/// `param[0..K]` with `K` a compile-time `IntLit` (folded from a
/// `def N` constant), the symbol referenced *only* through that slice
/// → the analyzer reports `Fixed { bytes: K * sizeof(elem) }`.
#[test]
fn slice_only_param_gets_bound() {
    let src = r#"
def N:i32 = 8
#[compute]
entry e(#[storage(set=2, binding=0, access=read)] xs: []vec4f32) vec4f32 =
  let xs = xs[0..N] in
  reduce(|a,b| a+b, @[0.0,0.0,0.0,0.0], xs)
"#;
    let prog = program_from(src);
    let e = bounds_for_entry(&prog, "e");
    assert_eq!(e.len(), 1, "exactly one tracked input: {:?}", e);
    let bytes = e.values().next().unwrap();
    assert_eq!(
        bytes,
        &BufferLen::Fixed { bytes: 8 * 16 },
        "K=8 × sizeof(vec4f32)=16 → 128 bytes",
    );
}

/// `length(param)` in the body is a disqualifying raw `Var(sym)`
/// reference — symbol drops out of the bounds map.
#[test]
fn length_call_disqualifies() {
    let src = r#"
#[compute]
entry e(#[storage(set=2, binding=0, access=read)] xs: []vec4f32) i32 =
  length(xs)
"#;
    let prog = program_from(src);
    let bounds = bounds_for_entry(&prog, "e");
    assert!(
        bounds.is_empty(),
        "length(xs) is a raw use → no bound: {:?}",
        bounds,
    );
}

/// `let xs = xs[0..N] in ... xs ...` — the outer storage-param symbol
/// only appears in the slice; the inner `xs` is a fresh let-bound
/// `SymbolId`. The analyzer's tracked set never sees the inner `xs`,
/// so the outer slice bound stands.
#[test]
fn let_shadowing_does_not_disqualify_outer() {
    let src = r#"
def N:i32 = 4
#[compute]
entry e(#[storage(set=2, binding=0, access=read)] xs: []vec4f32) vec4f32 =
  let xs = xs[0..N] in
  reduce(|a,b| a+b, @[0.0,0.0,0.0,0.0], xs)
"#;
    let prog = program_from(src);
    let e = bounds_for_entry(&prog, "e");
    assert_eq!(
        e.values().next().unwrap(),
        &BufferLen::Fixed { bytes: 4 * 16 },
        "outer slice [0..4] preserved through inner `let xs` shadow"
    );
}

/// Multiple `param[0..K_i]` slices in the body → analyzer reports
/// `Fixed { bytes: max(K_i) * sizeof(elem) }`.
#[test]
fn multiple_slices_take_max() {
    let src = r#"
def SMALL:i32 = 4
def BIG:i32 = 16
#[compute]
entry e(#[storage(set=2, binding=0, access=read)] xs: []vec4f32) vec4f32 =
  let a = xs[0..SMALL] in
  let b = xs[0..BIG] in
  reduce(|x,y| x+y, @[0.0,0.0,0.0,0.0], a) +
    reduce(|x,y| x+y, @[0.0,0.0,0.0,0.0], b)
"#;
    let prog = program_from(src);
    let e = bounds_for_entry(&prog, "e");
    assert_eq!(
        e.values().next().unwrap(),
        &BufferLen::Fixed { bytes: 16 * 16 },
        "max(SMALL=4, BIG=16) × sizeof(vec4f32)=16 → 256 bytes"
    );
}
