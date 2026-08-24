use super::{IdArena, IdSource};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct TestId(u32);

impl From<u32> for TestId {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

#[test]
fn id_source_allocates_monotonically_and_peek_does_not_consume() {
    let mut source = IdSource::<TestId>::new();

    assert_eq!(source.peek_id(), TestId(0));
    assert_eq!(source.peek_id(), TestId(0));
    assert_eq!(source.next_id(), TestId(0));
    assert_eq!(source.next_id(), TestId(1));
}

#[test]
fn arena_supports_deferred_insertion_and_preserves_insertion_order() {
    let mut arena = IdArena::<TestId, &'static str>::new();
    let deferred = arena.alloc_id();
    let immediate = arena.alloc("immediate");
    arena.insert(deferred, "deferred");

    assert_eq!(arena.get(deferred), Some(&"deferred"));
    assert_eq!(arena[immediate], "immediate");
    assert_eq!(
        arena.iter().map(|(&id, &value)| (id, value)).collect::<Vec<_>>(),
        vec![(immediate, "immediate"), (deferred, "deferred")]
    );
}

#[test]
fn arena_supports_mutation_and_owned_iteration() {
    let mut arena = IdArena::<TestId, String>::default();
    let first = arena.alloc("first".to_owned());
    let second = arena.alloc("second".to_owned());

    arena[first].push('!');
    arena.get_mut(second).unwrap().push('?');

    assert_eq!(arena.ids().collect::<Vec<_>>(), vec![first, second]);
    assert_eq!(
        arena.values().map(String::as_str).collect::<Vec<_>>(),
        vec!["first!", "second?"]
    );
    assert_eq!(
        arena.into_iter().collect::<Vec<_>>(),
        vec![(first, "first!".to_owned()), (second, "second?".to_owned())]
    );
}
