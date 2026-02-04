//! Alias checker for tracking backing stores and detecting use-after-move errors.
//!
//! This module implements a visitor-based approach that tracks "backing stores" -
//! the underlying memory that variables reference. When a variable is consumed
//! (passed to a function with a `*T` parameter), all variables referencing the
//! same backing store become invalid.

use crate::ast::*;
use crate::error::Result;
use crate::types::TypeExt;
use crate::visitor::{self, Visitor};
use crate::{NodeId, SpanTable, TypeTable};
use polytype::TypeScheme;
use std::collections::{HashMap, HashSet};
use std::ops::ControlFlow;

/// Unique identifier for a backing store (the actual memory/array)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BackingStoreId(u32);

/// State of a backing store
#[derive(Debug, Clone, PartialEq)]
pub enum StoreState {
    Live,
    Consumed {
        at: NodeId,
        var_name: String,
    },
}

/// What an expression evaluates to in terms of aliasing
#[derive(Debug, Clone, Default)]
pub struct AliasInfo {
    /// The backing stores this expression references
    pub stores: HashSet<BackingStoreId>,
}

impl AliasInfo {
    pub fn copy() -> Self {
        Self {
            stores: HashSet::new(),
        }
    }

    pub fn fresh(id: BackingStoreId) -> Self {
        let mut stores = HashSet::new();
        stores.insert(id);
        Self { stores }
    }

    pub fn references(stores: HashSet<BackingStoreId>) -> Self {
        Self { stores }
    }

    pub fn is_copy(&self) -> bool {
        self.stores.is_empty()
    }

    /// Merge two AliasInfos (for if/else branches, function returns)
    pub fn union(&self, other: &AliasInfo) -> AliasInfo {
        let mut stores = self.stores.clone();
        stores.extend(other.stores.iter().cloned());
        AliasInfo { stores }
    }
}

/// Liveness information for an expression
#[derive(Debug, Clone, Default)]
pub struct ExprLivenessInfo {
    /// True if no other live variable references the same backing store(s)
    pub alias_free: bool,
    /// True if this is the last use of the value (not needed after)
    pub released: bool,
}

/// An alias-related error
#[derive(Debug, Clone)]
pub struct AliasError {
    pub kind: AliasErrorKind,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum AliasErrorKind {
    UseAfterMove {
        variable: String,
        consumed_var: String,
        consumed_at: NodeId,
        /// Other variables that alias the same backing store (for better error messages)
        aliases: Vec<String>,
    },
}

/// The alias checker that walks the AST using the visitor pattern
pub struct AliasChecker<'a> {
    type_table: &'a TypeTable,
    span_table: &'a SpanTable,
    /// All backing stores and their states
    stores: HashMap<BackingStoreId, StoreState>,
    /// Stack of scopes, each mapping variable names to their backing stores
    scopes: Vec<HashMap<String, HashSet<BackingStoreId>>>,
    /// Reverse mapping: backing store -> variables that reference it
    store_to_vars: HashMap<BackingStoreId, HashSet<String>>,
    /// Counter for generating unique store IDs
    next_store_id: u32,
    /// Computed AliasInfo for each expression node
    results: HashMap<NodeId, AliasInfo>,
    /// Collected errors
    errors: Vec<AliasError>,
    /// Liveness info for expressions at function call sites
    liveness: HashMap<NodeId, ExprLivenessInfo>,
    /// Pre-computed variable uses for determining "last use" (ordered by program order)
    var_uses: HashMap<String, Vec<NodeId>>,
}

impl<'a> AliasChecker<'a> {
    pub fn new(type_table: &'a TypeTable, span_table: &'a SpanTable) -> Self {
        Self {
            type_table,
            span_table,
            stores: HashMap::new(),
            scopes: vec![HashMap::new()],
            store_to_vars: HashMap::new(),
            next_store_id: 0,
            results: HashMap::new(),
            errors: Vec::new(),
            liveness: HashMap::new(),
            var_uses: HashMap::new(),
        }
    }

    /// Look up the span for a NodeId, falling back to a dummy span if not found
    fn get_span(&self, id: NodeId) -> Span {
        self.span_table.get(&id).copied().unwrap_or_else(|| Span::new(0, 0, 0, 0))
    }

    /// Create a new backing store and return its ID
    fn new_store(&mut self) -> BackingStoreId {
        let id = BackingStoreId(self.next_store_id);
        self.next_store_id += 1;
        self.stores.insert(id, StoreState::Live);
        id
    }

    /// Push a new scope
    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    /// Pop the current scope and clean up reverse mapping
    fn pop_scope(&mut self) {
        if self.scopes.len() > 1 {
            if let Some(scope) = self.scopes.pop() {
                // Clean up reverse mapping for variables going out of scope
                for (name, stores) in scope {
                    for store_id in stores {
                        if let Some(vars) = self.store_to_vars.get_mut(&store_id) {
                            vars.remove(&name);
                        }
                    }
                }
            }
        }
    }

    /// Bind a variable to backing stores in the current scope
    fn bind_variable(&mut self, name: &str, info: &AliasInfo) {
        if !info.stores.is_empty() {
            // Update forward mapping (variable -> stores)
            if let Some(scope) = self.scopes.last_mut() {
                scope.insert(name.to_string(), info.stores.clone());
            }
            // Update reverse mapping (store -> variables)
            for store_id in &info.stores {
                self.store_to_vars.entry(*store_id).or_default().insert(name.to_string());
            }
        }
    }

    /// Look up a variable's backing stores
    fn lookup_variable(&self, name: &str) -> Option<HashSet<BackingStoreId>> {
        for scope in self.scopes.iter().rev() {
            if let Some(stores) = scope.get(name) {
                return Some(stores.clone());
            }
        }
        None
    }

    /// Check if any of the given stores have been consumed
    fn check_stores_live(&self, stores: &HashSet<BackingStoreId>) -> Option<(&str, NodeId)> {
        for store_id in stores {
            if let Some(StoreState::Consumed { at, var_name }) = self.stores.get(store_id) {
                return Some((var_name.as_str(), *at));
            }
        }
        None
    }

    /// Get all variable names that alias the given backing stores, excluding a specific variable
    fn get_aliases(&self, stores: &HashSet<BackingStoreId>, exclude: &str) -> Vec<String> {
        let mut aliases: HashSet<String> = HashSet::new();
        for store_id in stores {
            if let Some(vars) = self.store_to_vars.get(store_id) {
                for var in vars {
                    if var != exclude {
                        aliases.insert(var.clone());
                    }
                }
            }
        }
        let mut result: Vec<_> = aliases.into_iter().collect();
        result.sort(); // Deterministic ordering for tests
        result
    }

    /// Consume all given backing stores
    fn consume_stores(&mut self, stores: &HashSet<BackingStoreId>, at: NodeId, var_name: &str) {
        for store_id in stores {
            self.stores.insert(
                *store_id,
                StoreState::Consumed {
                    at,
                    var_name: var_name.to_string(),
                },
            );
        }
    }

    /// Store the result for a node
    fn set_result(&mut self, id: NodeId, info: AliasInfo) {
        self.results.insert(id, info);
    }

    /// Get the result for a node (defaults to Copy if not found)
    fn get_result(&self, id: NodeId) -> AliasInfo {
        self.results.get(&id).cloned().unwrap_or_default()
    }

    /// Check a program for alias errors
    pub fn check_program(mut self, program: &Program) -> Result<AliasCheckResult> {
        // Pass 1: Collect all variable uses in program order
        for decl in &program.declarations {
            match decl {
                Declaration::Decl(d) => collect_variable_uses_in_expr(&d.body, &mut self.var_uses),
                Declaration::Entry(e) => collect_variable_uses_in_expr(&e.body, &mut self.var_uses),
                _ => {}
            }
        }

        // Pass 2: Main alias checking with liveness computation
        for decl in &program.declarations {
            match decl {
                Declaration::Decl(d) => self.check_decl(d),
                Declaration::Entry(e) => self.check_entry(e),
                _ => {} // Skip other declarations
            }
        }

        Ok(AliasCheckResult {
            errors: self.errors,
            liveness: self.liveness,
        })
    }

    fn check_decl(&mut self, decl: &Decl) {
        self.push_scope();

        // Bind parameters - each gets a fresh backing store if non-copy
        for param in &decl.params {
            self.bind_pattern_params(param);
        }

        // Check the body using visitor
        let _ = self.visit_expression(&decl.body);

        self.pop_scope();
    }

    fn check_entry(&mut self, entry: &EntryDecl) {
        self.push_scope();

        for param in &entry.params {
            self.bind_pattern_params(param);
        }

        let _ = self.visit_expression(&entry.body);

        self.pop_scope();
    }

    /// Bind pattern parameters, creating fresh backing stores for non-copy types
    fn bind_pattern_params(&mut self, pattern: &Pattern) {
        let names = pattern.collect_names();
        for name in names {
            if !self.node_is_copy_type(pattern.h.id) {
                let store_id = self.new_store();
                self.bind_variable(&name, &AliasInfo::fresh(store_id));
            }
        }
    }

    /// Check if a node's type is Copy
    fn node_is_copy_type(&self, node_id: NodeId) -> bool {
        if let Some(scheme) = self.type_table.get(&node_id) {
            is_copy_type_scheme(scheme)
        } else {
            true // Conservative: treat unknown as copy
        }
    }

    /// Check if the i-th parameter of a function is consuming
    fn is_param_consuming(&self, func_id: NodeId, param_index: usize) -> bool {
        if let Some(scheme) = self.type_table.get(&func_id) {
            let ty = unwrap_scheme(scheme);
            is_param_consuming_in_type(ty, param_index)
        } else {
            false
        }
    }

    /// Check if a function's return type is alias-free
    fn return_is_fresh(&self, func_id: NodeId) -> bool {
        if let Some(scheme) = self.type_table.get(&func_id) {
            let ty = unwrap_scheme(scheme);
            get_return_type_is_fresh(ty)
        } else {
            false
        }
    }

    /// Check if an expression has no aliases (only one variable references its backing stores)
    fn is_alias_free(&self, info: &AliasInfo) -> bool {
        info.stores
            .iter()
            .all(|store_id| self.store_to_vars.get(store_id).map(|vars| vars.len() <= 1).unwrap_or(true))
    }

    /// Check if a node's type is an array type
    fn is_array_type(&self, node_id: NodeId) -> bool {
        if let Some(scheme) = self.type_table.get(&node_id) {
            let ty = unwrap_scheme(scheme);
            ty.is_array()
        } else {
            false
        }
    }

    /// Collect all array-typed variable names used in an expression
    fn collect_array_vars_in_expr(&self, expr: &Expression) -> HashSet<String> {
        let mut vars = HashSet::new();
        self.collect_array_vars_recursive(expr, &mut vars);
        vars
    }

    fn collect_array_vars_recursive(&self, expr: &Expression, vars: &mut HashSet<String>) {
        match &expr.kind {
            ExprKind::Identifier(quals, name) => {
                if quals.is_empty() && self.is_array_type(expr.h.id) {
                    vars.insert(name.clone());
                }
            }
            ExprKind::ArrayLiteral(elems) => {
                for elem in elems {
                    self.collect_array_vars_recursive(elem, vars);
                }
            }
            ExprKind::ArrayIndex(arr, idx) => {
                self.collect_array_vars_recursive(arr, vars);
                self.collect_array_vars_recursive(idx, vars);
            }
            ExprKind::BinaryOp(_, left, right) => {
                self.collect_array_vars_recursive(left, vars);
                self.collect_array_vars_recursive(right, vars);
            }
            ExprKind::UnaryOp(_, operand) => {
                self.collect_array_vars_recursive(operand, vars);
            }
            ExprKind::Tuple(elems) => {
                for elem in elems {
                    self.collect_array_vars_recursive(elem, vars);
                }
            }
            ExprKind::Application(func, args) => {
                self.collect_array_vars_recursive(func, vars);
                for arg in args {
                    self.collect_array_vars_recursive(arg, vars);
                }
            }
            ExprKind::LetIn(let_in) => {
                self.collect_array_vars_recursive(&let_in.value, vars);
                self.collect_array_vars_recursive(&let_in.body, vars);
            }
            ExprKind::If(if_expr) => {
                self.collect_array_vars_recursive(&if_expr.condition, vars);
                self.collect_array_vars_recursive(&if_expr.then_branch, vars);
                self.collect_array_vars_recursive(&if_expr.else_branch, vars);
            }
            ExprKind::Lambda(lambda) => {
                self.collect_array_vars_recursive(&lambda.body, vars);
            }
            ExprKind::FieldAccess(expr, _) => {
                self.collect_array_vars_recursive(expr, vars);
            }
            _ => {}
        }
    }

    /// Check if all array variables in an expression are at their last use
    fn expr_is_released(&self, expr: &Expression) -> bool {
        let array_vars = self.collect_array_vars_in_expr(expr);
        array_vars.iter().all(|var| {
            if let Some(uses) = self.var_uses.get(var) {
                // Find the current use position
                if let Some(pos) = uses.iter().position(|&id| id == expr.h.id) {
                    pos == uses.len() - 1
                } else {
                    // This expression ID not found in uses - conservative: not released
                    false
                }
            } else {
                false // No tracked uses - conservative: not released
            }
        })
    }
}

impl<'a> Visitor for AliasChecker<'a> {
    type Break = ();

    fn visit_expr_int_literal(
        &mut self,
        id: NodeId,
        _n: &crate::lexer::IntString,
    ) -> ControlFlow<Self::Break> {
        self.set_result(id, AliasInfo::copy());
        ControlFlow::Continue(())
    }

    fn visit_expr_float_literal(&mut self, id: NodeId, _f: f32) -> ControlFlow<Self::Break> {
        self.set_result(id, AliasInfo::copy());
        ControlFlow::Continue(())
    }

    fn visit_expr_bool_literal(&mut self, id: NodeId, _b: bool) -> ControlFlow<Self::Break> {
        self.set_result(id, AliasInfo::copy());
        ControlFlow::Continue(())
    }

    fn visit_expr_identifier(
        &mut self,
        id: NodeId,
        _quals: &[String],
        name: &str,
    ) -> ControlFlow<Self::Break> {
        if self.node_is_copy_type(id) {
            self.set_result(id, AliasInfo::copy());
        } else if let Some(stores) = self.lookup_variable(name) {
            // Check if any backing store has been consumed
            if let Some((consumed_var, consumed_at)) = self.check_stores_live(&stores) {
                // Collect other variables that alias the same backing store
                let aliases = self.get_aliases(&stores, name);
                self.errors.push(AliasError {
                    kind: AliasErrorKind::UseAfterMove {
                        variable: name.to_string(),
                        consumed_var: consumed_var.to_string(),
                        consumed_at,
                        aliases,
                    },
                    span: self.get_span(id),
                });
            }
            self.set_result(id, AliasInfo::references(stores));
        } else {
            self.set_result(id, AliasInfo::copy());
        }
        ControlFlow::Continue(())
    }

    fn visit_expr_array_literal(
        &mut self,
        id: NodeId,
        elements: &[Expression],
    ) -> ControlFlow<Self::Break> {
        // Visit all elements first
        for elem in elements {
            self.visit_expression(elem)?;
        }
        // Array literal creates a fresh backing store
        let store_id = self.new_store();
        self.set_result(id, AliasInfo::fresh(store_id));
        ControlFlow::Continue(())
    }

    fn visit_expr_array_index(
        &mut self,
        id: NodeId,
        array: &Expression,
        index: &Expression,
    ) -> ControlFlow<Self::Break> {
        self.visit_expression(array)?;
        self.visit_expression(index)?;
        // Element type determines if copy or not
        if self.node_is_copy_type(id) {
            self.set_result(id, AliasInfo::copy());
        } else {
            // Non-copy element could alias the array
            // For simplicity, treat as copy for now
            self.set_result(id, AliasInfo::copy());
        }
        ControlFlow::Continue(())
    }

    fn visit_expr_binary_op(
        &mut self,
        id: NodeId,
        _op: &BinaryOp,
        left: &Expression,
        right: &Expression,
    ) -> ControlFlow<Self::Break> {
        self.visit_expression(left)?;
        self.visit_expression(right)?;
        self.set_result(id, AliasInfo::copy());
        ControlFlow::Continue(())
    }

    fn visit_expr_tuple(&mut self, id: NodeId, elements: &[Expression]) -> ControlFlow<Self::Break> {
        let mut all_stores = HashSet::new();
        for elem in elements {
            self.visit_expression(elem)?;
            all_stores.extend(self.get_result(elem.h.id).stores);
        }
        self.set_result(id, AliasInfo::references(all_stores));
        ControlFlow::Continue(())
    }

    fn visit_expr_let_in(&mut self, id: NodeId, let_in: &LetInExpr) -> ControlFlow<Self::Break> {
        // Visit the value expression
        self.visit_expression(&let_in.value)?;
        let value_info = self.get_result(let_in.value.h.id);

        // Push scope for the body
        self.push_scope();

        // Bind the pattern to the value's alias info
        let names = let_in.pattern.collect_names();
        for name in names {
            self.bind_variable(&name, &value_info);
        }

        // Visit the body
        self.visit_expression(&let_in.body)?;
        let body_info = self.get_result(let_in.body.h.id);

        self.pop_scope();

        self.set_result(id, body_info);
        ControlFlow::Continue(())
    }

    fn visit_expr_if(&mut self, id: NodeId, if_expr: &IfExpr) -> ControlFlow<Self::Break> {
        self.visit_expression(&if_expr.condition)?;

        // Save store state before branches - each branch sees the pre-branch state
        let pre_branch_stores = self.stores.clone();

        // Visit then branch
        self.visit_expression(&if_expr.then_branch)?;
        let then_stores = self.stores.clone();

        // Restore pre-branch state for else branch
        self.stores = pre_branch_stores;

        // Visit else branch
        self.visit_expression(&if_expr.else_branch)?;
        let else_stores = self.stores.clone();

        // Merge: a store is consumed after the if if it was consumed in EITHER branch
        // (conservative for code after the if-expression)
        for (store_id, state) in then_stores {
            if matches!(state, StoreState::Consumed { .. }) {
                self.stores.insert(store_id, state);
            }
        }
        for (store_id, state) in else_stores {
            if matches!(state, StoreState::Consumed { .. }) {
                self.stores.insert(store_id, state);
            }
        }

        let then_info = self.get_result(if_expr.then_branch.h.id);
        let else_info = self.get_result(if_expr.else_branch.h.id);

        // Result aliases union of both branches
        self.set_result(id, then_info.union(&else_info));
        ControlFlow::Continue(())
    }

    fn visit_expr_lambda(&mut self, id: NodeId, lambda: &LambdaExpr) -> ControlFlow<Self::Break> {
        self.push_scope();

        for param in &lambda.params {
            self.bind_pattern_params(param);
        }

        self.visit_expression(&lambda.body)?;

        self.pop_scope();

        // Lambdas are copy types
        self.set_result(id, AliasInfo::copy());
        ControlFlow::Continue(())
    }

    fn visit_expr_application(
        &mut self,
        id: NodeId,
        func: &Expression,
        args: &[Expression],
    ) -> ControlFlow<Self::Break> {
        // Visit function
        self.visit_expression(func)?;

        // Start with any alias info the function itself carries
        // (important for curried calls where partial applications carry aliasing)
        let func_info = self.get_result(func.h.id);
        let mut observing_stores: HashSet<BackingStoreId> = func_info.stores.clone();

        // Visit and process each argument
        for (i, arg) in args.iter().enumerate() {
            self.visit_expression(arg)?;
            let arg_info = self.get_result(arg.h.id);

            // Compute liveness info for array-typed arguments
            if self.is_array_type(arg.h.id) {
                let alias_free = self.is_alias_free(&arg_info);
                let released = self.expr_is_released(arg);
                self.liveness.insert(arg.h.id, ExprLivenessInfo { alias_free, released });
            }

            // Check if this parameter is consuming (*T)
            let param_is_consuming = self.is_param_consuming(func.h.id, i);

            if param_is_consuming {
                // Consume the argument's backing stores
                if !arg_info.stores.is_empty() {
                    let var_name = if let ExprKind::Identifier(quals, name) = &arg.kind {
                        if quals.is_empty() {
                            name.clone()
                        } else {
                            format!("{}.{}", quals.join("."), name)
                        }
                    } else {
                        "<expr>".to_string()
                    };
                    self.consume_stores(&arg_info.stores, arg.h.id, &var_name);
                }
            } else {
                // Non-consuming parameter - result might alias this arg
                observing_stores.extend(arg_info.stores);
            }
        }

        // Check if return type is alias-free (*T)
        if self.return_is_fresh(func.h.id) {
            let store_id = self.new_store();
            self.set_result(id, AliasInfo::fresh(store_id));
        } else if observing_stores.is_empty() {
            self.set_result(id, AliasInfo::copy());
        } else {
            // Result aliases all observing arguments (including those from partial applications)
            self.set_result(id, AliasInfo::references(observing_stores));
        }

        ControlFlow::Continue(())
    }

    fn visit_expr_field_access(
        &mut self,
        id: NodeId,
        expr: &Expression,
        _field: &str,
    ) -> ControlFlow<Self::Break> {
        self.visit_expression(expr)?;
        let base_info = self.get_result(expr.h.id);

        if self.node_is_copy_type(id) {
            self.set_result(id, AliasInfo::copy());
        } else {
            // Field access aliases the base
            self.set_result(id, base_info);
        }
        ControlFlow::Continue(())
    }

    fn visit_expr_slice(&mut self, id: NodeId, slice: &SliceExpr) -> ControlFlow<Self::Break> {
        // Visit the array and index expressions
        self.visit_expression(&slice.array)?;
        if let Some(start) = &slice.start {
            self.visit_expression(start)?;
        }
        if let Some(end) = &slice.end {
            self.visit_expression(end)?;
        }

        // Borrowed slices alias the original array - the slice is a view into the base.
        // The slice inherits the stores from the base array.
        let base_info = self.get_result(slice.array.h.id);
        self.set_result(id, AliasInfo::references(base_info.stores));
        ControlFlow::Continue(())
    }

    fn visit_expr_range(&mut self, id: NodeId, range: &RangeExpr) -> ControlFlow<Self::Break> {
        // Visit the range expressions
        self.visit_expression(&range.start)?;
        if let Some(step) = &range.step {
            self.visit_expression(step)?;
        }
        self.visit_expression(&range.end)?;

        // Ranges are desugared to iota/map which creates a fresh array.
        let store_id = self.new_store();
        self.set_result(id, AliasInfo::fresh(store_id));
        ControlFlow::Continue(())
    }

    // Default implementation handles other cases
    fn visit_expression(&mut self, e: &Expression) -> ControlFlow<Self::Break> {
        // Use the walk function which dispatches to specific handlers
        visitor::walk_expression(self, e)?;

        // If no specific handler set a result, default to copy
        if !self.results.contains_key(&e.h.id) {
            self.set_result(e.h.id, AliasInfo::copy());
        }

        ControlFlow::Continue(())
    }
}

// --- Helper functions for type checking ---

fn is_copy_type(ty: &polytype::Type<TypeName>) -> bool {
    // Unique types are not copy
    if ty.is_unique() {
        return false;
    }
    match ty {
        polytype::Type::Constructed(name, args) => match name {
            TypeName::Int(_) => true,
            TypeName::UInt(_) => true,
            TypeName::Float(_) => true,
            TypeName::Str("bool") => true,
            TypeName::Str("unit") => true,
            TypeName::Array => false,
            TypeName::Vec => false,
            TypeName::Mat => false,
            TypeName::Tuple(_) => args.iter().all(is_copy_type),
            TypeName::Arrow => true, // Functions are copy
            TypeName::Unique => unreachable!("Handled above via is_unique()"),
            _ => true, // Conservative: treat unknown as copy
        },
        polytype::Type::Variable(_) => true,
    }
}

/// Collect all variable uses in an expression, recording them in program order.
/// This is used to determine which use of a variable is the "last use".
fn collect_variable_uses_in_expr(expr: &Expression, uses: &mut HashMap<String, Vec<NodeId>>) {
    match &expr.kind {
        ExprKind::Identifier(quals, name) => {
            if quals.is_empty() {
                uses.entry(name.clone()).or_default().push(expr.h.id);
            }
        }
        ExprKind::IntLiteral(_)
        | ExprKind::FloatLiteral(_)
        | ExprKind::BoolLiteral(_)
        | ExprKind::StringLiteral(_) => {}
        ExprKind::ArrayLiteral(elems) => {
            for elem in elems {
                collect_variable_uses_in_expr(elem, uses);
            }
        }
        ExprKind::ArrayIndex(arr, idx) => {
            collect_variable_uses_in_expr(arr, uses);
            collect_variable_uses_in_expr(idx, uses);
        }
        ExprKind::BinaryOp(_, left, right) => {
            collect_variable_uses_in_expr(left, uses);
            collect_variable_uses_in_expr(right, uses);
        }
        ExprKind::UnaryOp(_, operand) => {
            collect_variable_uses_in_expr(operand, uses);
        }
        ExprKind::Tuple(elems) => {
            for elem in elems {
                collect_variable_uses_in_expr(elem, uses);
            }
        }
        ExprKind::Application(func, args) => {
            collect_variable_uses_in_expr(func, uses);
            for arg in args {
                collect_variable_uses_in_expr(arg, uses);
            }
        }
        ExprKind::LetIn(let_in) => {
            collect_variable_uses_in_expr(&let_in.value, uses);
            collect_variable_uses_in_expr(&let_in.body, uses);
        }
        ExprKind::If(if_expr) => {
            collect_variable_uses_in_expr(&if_expr.condition, uses);
            collect_variable_uses_in_expr(&if_expr.then_branch, uses);
            collect_variable_uses_in_expr(&if_expr.else_branch, uses);
        }
        ExprKind::Lambda(lambda) => {
            collect_variable_uses_in_expr(&lambda.body, uses);
        }
        ExprKind::FieldAccess(expr, _) => {
            collect_variable_uses_in_expr(expr, uses);
        }
        ExprKind::RecordLiteral(fields) => {
            for (_, value) in fields {
                collect_variable_uses_in_expr(value, uses);
            }
        }
        ExprKind::Match(match_expr) => {
            collect_variable_uses_in_expr(&match_expr.scrutinee, uses);
            for case in &match_expr.cases {
                collect_variable_uses_in_expr(&case.body, uses);
            }
        }
        ExprKind::Loop(loop_expr) => {
            if let Some(init) = &loop_expr.init {
                collect_variable_uses_in_expr(init, uses);
            }
            match &loop_expr.form {
                LoopForm::For(_, bound) | LoopForm::ForIn(_, bound) => {
                    collect_variable_uses_in_expr(bound, uses);
                }
                LoopForm::While(cond) => {
                    collect_variable_uses_in_expr(cond, uses);
                }
            }
            collect_variable_uses_in_expr(&loop_expr.body, uses);
        }
        ExprKind::ArrayWith { array, index, value } => {
            collect_variable_uses_in_expr(array, uses);
            collect_variable_uses_in_expr(index, uses);
            collect_variable_uses_in_expr(value, uses);
        }
        ExprKind::Range(range) => {
            collect_variable_uses_in_expr(&range.start, uses);
            collect_variable_uses_in_expr(&range.end, uses);
            if let Some(step) = &range.step {
                collect_variable_uses_in_expr(step, uses);
            }
        }
        ExprKind::Slice(slice) => {
            collect_variable_uses_in_expr(&slice.array, uses);
            if let Some(start) = &slice.start {
                collect_variable_uses_in_expr(start, uses);
            }
            if let Some(end) = &slice.end {
                collect_variable_uses_in_expr(end, uses);
            }
        }
        ExprKind::TypeAscription(expr, _) | ExprKind::TypeCoercion(expr, _) => {
            collect_variable_uses_in_expr(expr, uses);
        }
        ExprKind::VecMatLiteral(elems) => {
            for elem in elems {
                collect_variable_uses_in_expr(elem, uses);
            }
        }
        ExprKind::Unit | ExprKind::TypeHole => {}
    }
}

fn is_copy_type_scheme(scheme: &TypeScheme<TypeName>) -> bool {
    is_copy_type(unwrap_scheme(scheme))
}

fn is_unique_type(ty: &polytype::Type<TypeName>) -> bool {
    ty.is_unique()
}

fn unwrap_scheme(scheme: &TypeScheme<TypeName>) -> &polytype::Type<TypeName> {
    match scheme {
        TypeScheme::Monotype(ty) => ty,
        TypeScheme::Polytype { body, .. } => unwrap_scheme(body),
    }
}

fn is_param_consuming_in_type(ty: &polytype::Type<TypeName>, param_index: usize) -> bool {
    match ty {
        polytype::Type::Constructed(TypeName::Arrow, args) if args.len() == 2 => {
            if param_index == 0 {
                is_unique_type(&args[0])
            } else {
                is_param_consuming_in_type(&args[1], param_index - 1)
            }
        }
        _ => false,
    }
}

fn get_return_type_is_fresh(ty: &polytype::Type<TypeName>) -> bool {
    match ty {
        polytype::Type::Constructed(TypeName::Arrow, args) if args.len() == 2 => {
            get_return_type_is_fresh(&args[1])
        }
        _ => ty.is_unique(),
    }
}

/// Result of alias checking
#[derive(Debug)]
pub struct AliasCheckResult {
    pub errors: Vec<AliasError>,
    /// Liveness info for array-typed expressions at function call sites
    pub liveness: HashMap<NodeId, ExprLivenessInfo>,
}

impl AliasCheckResult {
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    pub fn print_errors(&self) {
        for error in &self.errors {
            match &error.kind {
                AliasErrorKind::UseAfterMove {
                    variable,
                    consumed_var,
                    aliases,
                    ..
                } => {
                    eprintln!("error: use of moved value `{}`", variable);
                    eprintln!("  --> {:?}", error.span);
                    eprintln!("  = note: value was moved when `{}` was consumed", consumed_var);
                    if !aliases.is_empty() {
                        eprintln!(
                            "  = note: `{}` shares backing store with: {}",
                            variable,
                            aliases.join(", ")
                        );
                    }
                }
            }
        }
    }
}
