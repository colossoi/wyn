use crate::scope::ScopeStack;

#[test]
fn test_basic_scope_operations() {
    let mut scope_stack: ScopeStack<i32> = ScopeStack::new();

    // Insert in global scope
    scope_stack.insert("x".to_string(), 1);
    assert_eq!(scope_stack.lookup("x"), Some(&1));

    // Push new scope and shadow variable
    scope_stack.push_scope();
    scope_stack.insert("x".to_string(), 2);
    scope_stack.insert("y".to_string(), 3);

    assert_eq!(scope_stack.lookup("x"), Some(&2)); // Shadows outer x
    assert_eq!(scope_stack.lookup("y"), Some(&3));

    // Pop scope
    scope_stack.pop_scope();
    assert_eq!(scope_stack.lookup("x"), Some(&1)); // Back to outer x
    assert!(scope_stack.lookup("y").is_none()); // y is gone
}

#[test]
fn test_free_variables() {
    let mut scope_stack: ScopeStack<i32> = ScopeStack::new();

    // Global scope
    scope_stack.insert("global_var".to_string(), 1);

    // Outer function scope
    scope_stack.push_scope();
    scope_stack.insert("outer_param".to_string(), 2);

    // Inner lambda scope
    scope_stack.push_scope();
    scope_stack.insert("inner_param".to_string(), 3);

    let used_names = vec![
        "inner_param".to_string(), // Defined in current scope
        "outer_param".to_string(), // Free variable from outer scope
        "global_var".to_string(),  // Free variable from global scope
        "undefined".to_string(),   // Not defined anywhere
    ];

    let free_vars = scope_stack.collect_free_variables(&used_names);

    // Should include variables from outer scopes, not current scope or undefined
    assert!(free_vars.contains(&"outer_param".to_string()));
    assert!(free_vars.contains(&"global_var".to_string()));
    assert!(!free_vars.contains(&"inner_param".to_string()));
    assert!(!free_vars.contains(&"undefined".to_string()));
}

#[test]
fn test_manual_scope_management() {
    let mut scope_stack: ScopeStack<i32> = ScopeStack::new();
    scope_stack.insert("x".to_string(), 1);

    // Manual scope push
    scope_stack.push_scope();
    scope_stack.insert("x".to_string(), 2);
    assert_eq!(scope_stack.lookup("x"), Some(&2));

    // Manual scope pop
    scope_stack.pop_scope();
    assert_eq!(scope_stack.lookup("x"), Some(&1));
}

// ---------------------------------------------------------------------------
// pattern_bound_names — one walker shared by name resolution and module
// elaboration. Covers every PatternKind variant directly so either caller
// can't silently drift from the other.
// ---------------------------------------------------------------------------

mod pattern_bound_names {
    use crate::ast::{Header, NodeId, Pattern, PatternKind, RecordPatternField, Span};
    use crate::scope::pattern_bound_names;
    use polytype::Type;

    // Build Patterns without going through the parser. We use a shared
    // fake NodeId of 0 — the walker never looks at it.
    fn pat(kind: PatternKind) -> Pattern {
        Pattern {
            h: Header {
                id: NodeId::new(0),
                span: Span::new(0, 0, 0, 0),
            },
            kind,
        }
    }
    fn name(s: &str) -> Pattern {
        pat(PatternKind::Name(s.into()))
    }
    fn wild() -> Pattern {
        pat(PatternKind::Wildcard)
    }
    fn i32_ty() -> Type<crate::ast::TypeName> {
        Type::Constructed(crate::ast::TypeName::Int(32), vec![])
    }

    #[test]
    fn name_pattern_yields_its_name() {
        assert_eq!(pattern_bound_names(&name("x")), vec!["x".to_string()]);
    }

    #[test]
    fn wildcard_unit_and_literal_yield_no_names() {
        assert!(pattern_bound_names(&wild()).is_empty());
        assert!(pattern_bound_names(&pat(PatternKind::Unit)).is_empty());
        assert!(
            pattern_bound_names(&pat(PatternKind::Literal(crate::ast::PatternLiteral::Bool(true))))
                .is_empty()
        );
    }

    #[test]
    fn tuple_pattern_recurses() {
        let p = pat(PatternKind::Tuple(vec![name("a"), name("b"), wild(), name("c")]));
        assert_eq!(pattern_bound_names(&p), vec!["a", "b", "c"]);
    }

    #[test]
    fn constructor_pattern_recurses() {
        let p = pat(PatternKind::Constructor("some".into(), vec![name("x")]));
        assert_eq!(pattern_bound_names(&p), vec!["x"]);
    }

    #[test]
    fn typed_wrapper_recurses() {
        let p = pat(PatternKind::Typed(Box::new(name("n")), i32_ty()));
        assert_eq!(pattern_bound_names(&p), vec!["n"]);
    }

    #[test]
    fn attributed_wrapper_recurses() {
        let p = pat(PatternKind::Attributed(vec![], Box::new(name("x"))));
        assert_eq!(pattern_bound_names(&p), vec!["x"]);
    }

    #[test]
    fn record_pattern_with_explicit_patterns_recurses() {
        let p = pat(PatternKind::Record(vec![
            RecordPatternField {
                field: "a".into(),
                pattern: Some(name("alias_a")),
            },
            RecordPatternField {
                field: "b".into(),
                pattern: Some(name("alias_b")),
            },
        ]));
        assert_eq!(pattern_bound_names(&p), vec!["alias_a", "alias_b"]);
    }

    #[test]
    fn record_shorthand_binds_field_name() {
        // `{ x }` binds the identifier `x`; the walker treats a field with no
        // nested pattern as if the pattern were `Name(field)`.
        let p = pat(PatternKind::Record(vec![
            RecordPatternField {
                field: "x".into(),
                pattern: None,
            },
            RecordPatternField {
                field: "y".into(),
                pattern: Some(name("renamed_y")),
            },
        ]));
        assert_eq!(pattern_bound_names(&p), vec!["x", "renamed_y"]);
    }

    #[test]
    fn deeply_nested_pattern_visits_every_leaf() {
        // `#[foo] (#some(x), {k}: i32, _)` — attributed + tuple + constructor +
        // record-shorthand + typed + wildcard; all in one.
        let inner = pat(PatternKind::Tuple(vec![
            pat(PatternKind::Constructor("some".into(), vec![name("x")])),
            pat(PatternKind::Typed(
                Box::new(pat(PatternKind::Record(vec![RecordPatternField {
                    field: "k".into(),
                    pattern: None,
                }]))),
                i32_ty(),
            )),
            wild(),
        ]));
        let p = pat(PatternKind::Attributed(vec![], Box::new(inner)));
        assert_eq!(pattern_bound_names(&p), vec!["x", "k"]);
    }
}
