use super::*;

#[test]
fn basic_scoping() {
    let mut m = ScopedMap::new();
    m.insert(1, "a");
    assert_eq!(m.get(&1), Some("a"));

    m.push_scope();
    assert_eq!(m.get(&1), Some("a")); // visible from parent
    m.insert(2, "b");
    assert_eq!(m.get(&2), Some("b"));

    m.pop_scope();
    assert_eq!(m.get(&1), Some("a"));
    assert_eq!(m.get(&2), None); // gone after pop
}

#[test]
fn shadow_and_restore() {
    let mut m = ScopedMap::new();
    m.insert(1, 10);
    m.push_scope();
    m.insert(1, 20); // shadow
    assert_eq!(m.get(&1), Some(20));
    m.pop_scope();
    assert_eq!(m.get(&1), Some(10)); // restored
}

#[test]
fn sibling_scopes_independent() {
    let mut m = ScopedMap::new();
    m.insert(0, "root");

    // Child A
    m.push_scope();
    m.insert(1, "a");
    assert_eq!(m.get(&1), Some("a"));
    m.pop_scope();

    // Child B — should not see child A's insertions
    m.push_scope();
    assert_eq!(m.get(&1), None);
    m.insert(1, "b");
    assert_eq!(m.get(&1), Some("b"));
    m.pop_scope();

    assert_eq!(m.get(&0), Some("root"));
    assert_eq!(m.get(&1), None);
}
