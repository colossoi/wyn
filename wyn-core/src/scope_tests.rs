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
