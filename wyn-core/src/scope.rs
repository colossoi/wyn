use std::collections::HashMap;

/// A single scope containing variable bindings
#[derive(Debug, Clone)]
pub struct Scope<T> {
    bindings: HashMap<String, T>,
}

impl<T: Clone> Default for Scope<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone> Scope<T> {
    pub fn new() -> Self {
        Scope {
            bindings: HashMap::new(),
        }
    }

    pub fn insert(&mut self, name: String, value: T) {
        self.bindings.insert(name, value);
    }

    /// Get a binding.
    pub fn get(&self, name: &str) -> Option<&T> {
        self.bindings.get(name)
    }

    pub fn contains_key(&self, name: &str) -> bool {
        self.bindings.contains_key(name)
    }

    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.bindings.keys()
    }
}

/// A stack-based scope manager that tracks nested scopes
#[derive(Debug, Clone)]
pub struct ScopeStack<T> {
    scopes: Vec<Scope<T>>,
}

impl<T: Clone> Default for ScopeStack<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone> ScopeStack<T> {
    /// Create a new scope stack with a global scope
    pub fn new() -> Self {
        ScopeStack {
            scopes: vec![Scope::new()],
        }
    }

    /// Push a new scope onto the stack
    pub fn push_scope(&mut self) {
        self.scopes.push(Scope::new());
    }

    /// Pop the current scope from the stack
    /// Returns None if trying to pop the global scope
    pub fn pop_scope(&mut self) -> Option<Scope<T>> {
        if self.scopes.len() > 1 { self.scopes.pop() } else { None }
    }

    /// Insert a binding in the current (innermost) scope
    pub fn insert(&mut self, name: String, value: T) {
        if let Some(current_scope) = self.scopes.last_mut() {
            current_scope.insert(name, value);
        }
    }

    /// Look up a binding, searching from innermost to outermost scope.
    pub fn lookup(&self, name: &str) -> Option<&T> {
        for scope in self.scopes.iter().rev() {
            if let Some(value) = scope.get(name) {
                return Some(value);
            }
        }
        None
    }

    /// Check if a name is defined in the current scope (not outer scopes)
    pub fn is_defined_in_current_scope(&self, name: &str) -> bool {
        self.scopes.last().map(|scope| scope.contains_key(name)).unwrap_or(false)
    }

    /// Check if a name is defined in any scope (ignoring consumed state)
    pub fn is_defined(&self, name: &str) -> bool {
        self.scopes.iter().rev().any(|scope| scope.contains_key(name))
    }

    /// Get the current scope depth (0 = global scope)
    pub fn depth(&self) -> usize {
        self.scopes.len().saturating_sub(1)
    }

    /// Iterate over all bindings in all scopes (from outermost to innermost)
    /// Calls the provided closure for each (name, value) pair
    pub fn for_each_binding<F>(&self, mut f: F)
    where
        F: FnMut(&str, &T),
    {
        for scope in &self.scopes {
            for (name, value) in &scope.bindings {
                f(name, value);
            }
        }
    }

    /// Collect all names that are defined in outer scopes but not current scope
    /// This is useful for free variable analysis
    pub fn collect_free_variables(&self, used_names: &[String]) -> Vec<String> {
        let mut free_vars = Vec::new();

        if let Some(current_scope) = self.scopes.last() {
            for name in used_names {
                // If the name is used but not defined in current scope,
                // check if it's defined in outer scopes
                if !current_scope.contains_key(name) {
                    // Search in outer scopes
                    for outer_scope in self.scopes[..self.scopes.len() - 1].iter().rev() {
                        if outer_scope.contains_key(name) {
                            free_vars.push(name.clone());
                            break;
                        }
                    }
                }
            }
        }

        free_vars
    }
}
