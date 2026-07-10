//! Tests for the gather-materialization pass (`lift_gathers`).

use super::{DefMeta, Program, SoacOp, Term, TermKind, VarRef};
use crate::builtins::{catalog, BuiltinId};
use crate::interface::StorageRole;

const GATHER_SRC: &str = "\
#[compute]
entry gen(bh: []vec4f32) []i32 =
  let counts = map(|h:vec4f32| 4 + 5*(if h.x>4.0 then 3 else 1), bh) in
  map(|i:i32| counts[i % 256], iota(6144))
";

/// Run the front-end through `float_runtime_index_nested_producers` — the slot
/// just before `plan_execute_gather_residency` (which runs `lift_gathers::run`)
/// in the residency cluster — returning the TLC program the pass consumes.
fn pre_residency(src: &str) -> Program {
    crate::test_pipeline::compile_thru_runtime_index(src)
}

fn def_named<'a>(program: &'a Program, name: &str) -> &'a super::Def {
    program
        .defs
        .iter()
        .find(|d| program.symbols.get(d.name).map(|n| n == name).unwrap_or(false))
        .unwrap_or_else(|| panic!("no def named {name}"))
}

/// Count applications of the `storage_index` builtin anywhere in `term`.
fn count_storage_index(term: &Term, id: BuiltinId) -> usize {
    let mut n = 0;
    walk(term, &mut |t| {
        if let TermKind::App { func, .. } = &t.kind {
            if matches!(&func.kind, TermKind::Var(VarRef::Builtin { id: fid, .. }) if *fid == id) {
                n += 1;
            }
        }
    });
    n
}

/// True if any reachable `let` binds a `Soac(Map)` (the un-lifted producer).
fn has_map_producing_let(term: &Term) -> bool {
    let mut found = false;
    walk(term, &mut |t| {
        if let TermKind::Let { rhs, .. } = &t.kind {
            if matches!(&rhs.kind, TermKind::Soac(SoacOp::Map { .. })) {
                found = true;
            }
        }
    });
    found
}

fn walk(term: &Term, f: &mut impl FnMut(&Term)) {
    f(term);
    term.for_each_child(&mut |c| walk(c, f));
}

#[test]
fn lifts_map_gather_into_prepass_and_storage_index() {
    let program = pre_residency(GATHER_SRC);
    assert!(
        has_map_producing_let(&def_named(&program, "gen").body),
        "precondition: the consumer starts with a map-producing `let counts`"
    );

    let lifted = super::run(program, &mut crate::IdSource::<u32>::new());
    let storage_index = catalog().known().storage_index;

    // The producer map became its own compute pre-pass with an Output decl.
    let gather = lifted
        .defs
        .iter()
        .find(|d| lifted.symbols.get(d.name).map(|n| n.contains("_gather_0")).unwrap_or(false))
        .expect("a `gen_gather_0` pre-pass def must be created");
    let DefMeta::EntryPoint(gather_decl) = &gather.meta else {
        panic!("gather pre-pass must be an entry point");
    };
    assert!(
        gather_decl.entry_type.is_compute(),
        "gather pre-pass must be a compute entry"
    );
    let out_binding = gather_decl
        .storage_bindings
        .iter()
        .find(|b| matches!(b.role, StorageRole::Output))
        .expect("gather pre-pass must pin its result via an Output storage binding");

    // The consumer dropped the `let counts = map(..)` and now reads the
    // gather buffer once per use via `storage_index` at the same binding.
    let gen = def_named(&lifted, "gen");
    assert!(
        !has_map_producing_let(&gen.body),
        "the producer `let` must be gone from the consumer after lifting"
    );
    assert_eq!(
        count_storage_index(&gen.body, storage_index),
        1,
        "the single `counts[i % 256]` use must become one storage_index load"
    );
    let DefMeta::EntryPoint(gen_decl) = &gen.meta else {
        panic!("gen must be an entry point");
    };
    let in_binding = gen_decl
        .storage_bindings
        .iter()
        .find(|b| matches!(b.role, StorageRole::Input))
        .expect("consumer must declare the gather buffer as an Input binding");

    // Producer and consumer agree on the gather buffer's (set, binding).
    assert_eq!(
        out_binding.binding, in_binding.binding,
        "pre-pass output and consumer input must name the same gather buffer"
    );
}

fn has_gather_prepass(program: &Program) -> bool {
    program
        .defs
        .iter()
        .any(|d| program.symbols.get(d.name).map(|n| n.contains("_gather_")).unwrap_or(false))
}

#[test]
fn lifts_scan_over_input_array() {
    // A scan over an entry-param array is a valid gather producer (scan
    // preserves element count and type), so it's lifted like a map.
    let src = "\
#[compute]
entry g(xs: []i32) []i32 =
  let o = scan(|a:i32,b:i32| a+b, 0, xs) in
  map(|i:i32| o[i % 256], iota(6144))
";
    let lifted = super::run(pre_residency(src), &mut crate::IdSource::<u32>::new());
    assert!(
        has_gather_prepass(&lifted),
        "scan over an input array must be lifted into a gather pre-pass"
    );
}

#[test]
fn lifts_scan_over_computed_array() {
    // A scan over a *computed* array: fusion folds the producer map into the
    // scan (`scan(op, ne, map(g, bh))`), so the scan reads the entry param
    // `bh` directly and the lift can pull it into a self-contained pre-pass —
    // even though `g` changes the element type (vec4f32 -> i32).
    let src = "\
#[compute]
entry gen(bh: []vec4f32) []i32 =
  let counts = map(|h:vec4f32| 4 + 5*(if h.x>4.0 then 3 else 1), bh) in
  let offsets = scan(|a:i32,b:i32| a+b, 0, counts) in
  map(|i:i32| offsets[i % 256], iota(6144))
";
    let lifted = super::run(pre_residency(src), &mut crate::IdSource::<u32>::new());
    assert!(
        has_gather_prepass(&lifted),
        "a scan over a (map-fused) computed array must be lifted into a gather pre-pass"
    );
}

#[test]
fn lifts_chained_scan_producers() {
    // Chained let-bound scans: `mid` is a scan over the entry param `xs` and is
    // itself consumed as the `outer` scan's input; `outer`'s result is randomly
    // indexed. `mid` is lifted because a SOAC input is a materializable use; the
    // resulting `mid` gather buffer then sources `outer`'s pre-pass (one
    // `chained_intermediates` Input decl). Two gather pre-passes — one per
    // scan — with `mid`'s output feeding `outer`'s input.
    let src = "\
#[compute]
entry gen(xs: []i32) []i32 =
  let mid = scan(|a:i32,b:i32| a+b, 0, xs) in
  let outer = scan(|a:i32,b:i32| a*b, 1, mid) in
  map(|i:i32| outer[i % 256], iota(6144))
";
    let lifted = super::run(pre_residency(src), &mut crate::IdSource::<u32>::new());
    let prepasses = lifted
        .defs
        .iter()
        .filter(|d| {
            matches!(d.meta, super::DefMeta::EntryPoint(_))
                && lifted.symbols.get(d.name).is_some_and(|n| n.contains("_gather_"))
        })
        .count();
    assert_eq!(
        prepasses, 2,
        "expected one gather pre-pass per chained scan (mid + outer)"
    );
}

#[test]
fn lifts_chained_scan_with_fused_gather_input() {
    // `s2` is a scan over a map that dynamically indexes `s1`; fusion composes
    // the map into the scan. The first gather lift materializes `s1`, and the
    // second must still see `s2` as a liftable scan producer.
    let src = "\
#[compute]
entry gen(bh: []vec4f32) []i32 =
  let counts = map(|h:vec4f32| 4 + 5*(if h.x>4.0 then 3 else 1), bh) in
  let s1 = scan(|a:i32,b:i32| a+b, 0, counts) in
  let s2 = scan(|a:i32,b:i32| a+b, 0,
      map(|i:i32| s1[i % 256], iota(6144))) in
  map(|i:i32| s2[i % 128], iota(2048))
";
    let lifted = super::run(pre_residency(src), &mut crate::IdSource::<u32>::new());
    let prepasses = lifted
        .defs
        .iter()
        .filter(|d| {
            matches!(d.meta, super::DefMeta::EntryPoint(_))
                && lifted.symbols.get(d.name).is_some_and(|n| n.contains("_gather_"))
        })
        .count();
    assert_eq!(
        prepasses, 2,
        "expected gather pre-passes for both s1 and the fused s2 scan"
    );
}

#[test]
fn declines_bare_var_in_non_materializable_position() {
    // `counts` appears both as `counts[i % 256]` (materializable → would
    // become `storage_index`) and as a bare `Var(counts)` argument to the
    // `length` builtin (non-materializable: we don't know how to lower a
    // runtime-sized Composite threaded through arbitrary term positions, and
    // `length` reading a hypothetical gather buffer's length isn't part of
    // the current materialization story). The bail must win: the lift
    // declines the whole binding, leaving the un-lifted `let counts = map(...)`
    // for downstream passes. Asserts that we did not silently rewrite some
    // uses and forget others — the bail is all-or-nothing.
    let src = "\
#[compute]
entry gen(xs: []i32) []i32 =
  let counts = map(|x: i32| x + 1, xs) in
  let n: i32 = length(counts) in
  map(|i: i32| counts[i % 256] + n, iota(6144))
";
    let lifted = super::run(pre_residency(src), &mut crate::IdSource::<u32>::new());
    assert!(
        !has_gather_prepass(&lifted),
        "a bare Var(counts) in a non-materializable position must trip the bail \
         and decline the lift even when a materializable use is present"
    );
    assert!(
        has_map_producing_let(&def_named(&lifted, "gen").body),
        "with the lift declined, the original `let counts = map(...)` must remain"
    );
}

#[test]
fn leaves_pointwise_map_chains_untouched() {
    // No random indexing: `counts` is consumed pointwise, so fusion/lowering
    // handle it — there's nothing to lift, and no pre-pass should appear.
    let src = "\
#[compute]
entry gen(bh: []i32) []i32 =
  let counts = map(|x:i32| x + 1, bh) in
  map(|c:i32| c * 2, counts)
";
    let program = pre_residency(src);
    let n_before = program.defs.len();
    let lifted = super::run(program, &mut crate::IdSource::<u32>::new());
    assert_eq!(
        lifted.defs.len(),
        n_before,
        "pointwise chains must not spawn a gather pre-pass"
    );
    assert!(
        !lifted.defs.iter().any(|d| lifted
            .symbols
            .get(d.name)
            .map(|n| n.contains("_gather_"))
            .unwrap_or(false)),
        "no gather pre-pass for a non-indexed computed array"
    );
}

#[test]
fn lifts_gather_inside_tuple_slot_operand() {
    // Multi-output entry where the gather candidate (`let counts =
    // map(...) in map(|i| counts[i % 256], iota(...))`) lives inline as a
    // tuple operand, not above the tuple. After normalize_outputs wraps
    // the slot operand in `OutputSlotStore { value: ... }`, the lifter
    // must still descend into `value` to find the producer. Regression:
    // before the fix, `lift_in_term`'s `_ =>` catch-all dropped the
    // OutputSlotStore subtree, and the gather producer survived
    // unlifted into codegen.
    let src = "\
#[compute]
entry gen(bh: []vec4f32) ([]i32, []i32) =
  (let counts = map(|h:vec4f32| 4 + 5*(if h.x>4.0 then 3 else 1), bh) in
     map(|i:i32| counts[i % 256], iota(6144)),
   iota(100))
";
    let mut program = pre_residency(src);
    crate::tlc::normalize_outputs::run(&mut program).expect("normalize_outputs");
    let lifted = super::run(program, &mut crate::IdSource::<u32>::new());
    assert!(
        has_gather_prepass(&lifted),
        "gather producer inside an OutputSlotStore.value must still be lifted"
    );
    assert!(
        !has_map_producing_let(&def_named(&lifted, "gen").body),
        "the producer `let` inside the tuple operand must be gone after lifting"
    );
}
