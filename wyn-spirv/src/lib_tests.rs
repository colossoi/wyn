use super::*;

/// `Id<K>` for distinct `K`s are distinct types — a function expecting
/// `TypeId` cannot be called with a `ValueId`. Compile-only check; if
/// this file builds, the safety invariant holds.
#[test]
fn id_kinds_are_distinct_types() {
    fn takes_type(_: TypeId) {}
    fn takes_value(_: ValueId) {}
    let t: TypeId = Id::new(7);
    let v: ValueId = Id::new(11);
    takes_type(t);
    takes_value(v);
    // `takes_type(v)` here would fail to compile — that's the point.
}

#[test]
fn deref_extracts_raw_word() {
    let t: TypeId = Id::new(42);
    assert_eq!(*t, 42);
    // Function signatures wanting `spirv::Word` still need explicit
    // `*id` — auto-coercion via Deref doesn't apply to function args.
    fn takes_word(_: spirv::Word) {}
    takes_word(*t);
}

#[test]
fn id_is_copy_eq_hash() {
    use std::collections::HashSet;
    let a: TypeId = Id::new(1);
    let b: TypeId = Id::new(1);
    let c: TypeId = Id::new(2);
    assert_eq!(a, b);
    assert_ne!(a, c);
    let mut set = HashSet::new();
    set.insert(a);
    set.insert(b);
    set.insert(c);
    assert_eq!(set.len(), 2);
    // `a` is still usable after insertion — proves `Copy`.
    let _ = a;
}

#[test]
fn builder_emits_minimal_valid_module() {
    let mut b = SpirvBuilder::new();
    // Add an OpTypeVoid + a no-op function so the assembled module
    // has *something* in it. Pure smoke test that the setup is right.
    let void = b.void_type();
    let (_fn_id, _params, _code_block) = b.begin_function("main", &[], void).expect("begin_function");
    b.ret().expect("ret");
    b.end_function().expect("end_function");
    let module = b.into_module();
    assert!(
        !module.functions.is_empty(),
        "module should contain at least one function"
    );
}
