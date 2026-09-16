use super::{IdArena, IdSource, InternIndex, Interner};

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

#[test]
fn interner_reuses_borrowed_values_and_finishes_as_an_arena() {
    let mut values = Interner::<TestId, String>::new();
    let first = values.intern("first");
    let second = values.intern("second");
    assert_eq!(values.intern("first"), first);
    assert_eq!(values.get("second"), Some(second));
    assert_eq!(values.resolve(first), "first");
    assert_eq!(values.arena().len(), 2);
    let arena = values.into_arena();
    assert_eq!(arena[first], "first");
    assert_eq!(arena.ids().collect::<Vec<_>>(), vec![first, second]);
}

#[test]
fn interner_checks_equality_when_hashes_collide() {
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct Collision(u32);
    impl std::hash::Hash for Collision {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            state.write_u8(0);
        }
    }
    let mut values = Interner::<TestId, Collision>::new();
    let first = values.intern(&Collision(1));
    let second = values.intern(&Collision(2));
    assert_ne!(first, second);
    assert_eq!(values.intern(&Collision(1)), first);
    assert_eq!(values.resolve(second), &Collision(2));
}

#[test]
fn interning_an_existing_arena_preserves_ids_and_allocation_state() {
    let mut arena = IdArena::<TestId, String>::new();
    let reserved = arena.alloc_id();
    let first = arena.alloc("first".into());
    let duplicate = arena.alloc("first".into());
    arena.insert(reserved, "reserved".into());
    let mut index = InternIndex::from_arena(&arena);
    assert_eq!(index.intern(&mut arena, "first"), duplicate);
    assert_eq!(index.intern(&mut arena, "reserved"), reserved);
    let next = index.intern(&mut arena, "next");
    assert_eq!(next, TestId(3));
    assert_eq!(index.intern(&mut arena, "next"), next);
    assert_eq!(arena.len(), 4);
    assert_eq!(arena[first], "first");
    let rebuilt = InternIndex::from_arena(&arena);
    assert_eq!(rebuilt.get("next"), Some(next));
}
