//! Tests for the gather-materialization pass (`lift_gathers`).

use super::{DefMeta, Program, SoacOp, Term, TermKind, VarRef};
use crate::Compiler;
use crate::builtins::{BuiltinId, catalog};
use crate::interface::StorageRole;

const GATHER_SRC: &str = "\
#[compute]
entry gen(bh: []vec4f32) []i32 =
  let counts = map(|h:vec4f32| 4 + 5*(if h.x>4.0 then 3 else 1), bh) in
  map(|i:i32| counts[i % 256], iota(6144))
";

/// Run the front-end through `apply_ownership` (the stage `lift_gathers`
/// consumes), returning the TLC program just before the pass.
fn ownership_applied(src: &str) -> Program {
    let (mut node_counter, mut module_manager) = crate::cached_compiler_init();
    let type_checked = Compiler::parse(src, &mut node_counter)
        .expect("parse")
        .resolve(&module_manager)
        .expect("resolve")
        .fold_ast_constants()
        .type_check(&mut module_manager)
        .expect("type_check");
    type_checked
        .to_tlc(&module_manager, false)
        .partial_eval()
        .normalize_soacs()
        .fuse_maps()
        .apply_ownership()
        .expect("apply_ownership")
        .0
        .tlc
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
    let program = ownership_applied(GATHER_SRC);
    assert!(
        has_map_producing_let(&def_named(&program, "gen").body),
        "precondition: the consumer starts with a map-producing `let counts`"
    );

    let lifted = super::run(program);
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
        (out_binding.set, out_binding.binding),
        (in_binding.set, in_binding.binding),
        "pre-pass output and consumer input must name the same gather buffer"
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
    let program = ownership_applied(src);
    let n_before = program.defs.len();
    let lifted = super::run(program);
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
