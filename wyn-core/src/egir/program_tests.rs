//! Tests for the region arena / interner invariants that the opaque
//! `RegionId` representation depends on.

use super::*;
use crate::ast::{Span, TypeName};
use crate::egir::types::{EGraph, RegionId};
use crate::pipeline_descriptor::PipelineDescriptor;
use polytype::Type;

fn unit_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Unit, vec![])
}

fn empty_func(name: &str) -> EgirFunc {
    EgirFunc::new(
        name.to_string(),
        Span::dummy(),
        None,
        vec![],
        unit_ty(),
        EGraph::new(),
        LookupMap::new(),
    )
}

#[test]
fn interner_assigns_dense_indices_and_deduplicates_names() {
    let mut interner = RegionInterner::default();
    let foo = interner.intern("foo");
    let bar = interner.intern("bar");
    assert_eq!(
        foo,
        interner.intern("foo"),
        "same name re-interns to the same index"
    );
    assert_ne!(foo, bar);
    assert_eq!(foo.index(), 0);
    assert_eq!(bar.index(), 1);
    assert_eq!(interner.name(foo), "foo");
    assert_eq!(interner.name(bar), "bar");
    assert_eq!(interner.get("foo"), Some(foo));
    assert_eq!(interner.get("absent"), None);
}

/// Correctness risk #1 — index agreement. A `SegBody` built during TLC→EGIR
/// conversion interns its callee *before* `EgirInner::new` walks the function
/// list, and the callee may appear later in that list than other functions.
/// The arena entry for the callee must still land on the index the `SegBody`
/// already holds.
#[test]
fn segbody_index_resolves_to_its_arena_function() {
    // `op` is referenced (interned) first, as a SegBody operator region would
    // be, even though it is the *second* function in the program.
    let mut interner = RegionInterner::default();
    let op_id: RegionId = interner.intern("op");

    let inner = EgirInner::new(
        vec![empty_func("main"), empty_func("op")],
        vec![],
        vec![],
        vec![],
        PipelineDescriptor::default(),
        interner,
    );

    // The pre-interned index is preserved and points at the right body.
    assert_eq!(inner.region_interner.get("op"), Some(op_id));
    assert!(
        inner.regions.contains_key(&op_id),
        "callee region must be in the arena"
    );
    assert_eq!(inner.region_name(op_id), "op");
    assert_eq!(inner.regions.get(&op_id).unwrap().name, "op");
    // `main` was assigned its index by `EgirInner::new`, after `op`.
    let main_id = inner.region_interner.get("main").expect("main interned");
    assert_ne!(main_id, op_id);
    assert_eq!(inner.region_name(main_id), "main");
}
