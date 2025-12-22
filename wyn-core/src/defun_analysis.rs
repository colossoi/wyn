//! Defunctionalization analysis pass
//!
//! This pass walks the AST and computes StaticValue classifications for each expression.
//! - Lambda expressions are classified as Closure with computed free variables
//! - Other expressions are classified as Dyn
//!
//! The analysis output is used by the flattening pass to generate MIR.

use crate::ast::{self, ExprKind, Expression, NodeId, Type, TypeName};
use crate::pattern;
use crate::scope::ScopeStack;
use polytype::TypeScheme;
use std::collections::{HashMap, HashSet};

/// Static value classification for defunctionalization.
/// This is the analysis-only version without binding_id (that's assigned during flattening).
#[derive(Debug, Clone)]
pub enum StaticValue {
    /// Dynamic runtime value
    Dyn,
    /// Defunctionalized closure with known call target and captured free variables
    Closure {
        /// Name of the generated lambda function
        lam_name: String,
        /// Free variables captured by this closure, with their types
        free_vars: Vec<(String, Type)>,
    },
}

/// Pre-computed defunctionalization analysis results.
/// This is passed to the flattening pass.
#[derive(Debug)]
pub struct DefunAnalysis {
    /// Expression NodeId -> StaticValue classification
    pub classifications: HashMap<NodeId, StaticValue>,
}

impl DefunAnalysis {
    /// Look up the classification for an expression.
    /// Returns None if the expression wasn't analyzed (which is a bug).
    pub fn get(&self, node_id: NodeId) -> Option<&StaticValue> {
        self.classifications.get(&node_id)
    }

    /// Look up the classification for an expression, panicking if not found.
    pub fn get_or_panic(&self, node_id: NodeId) -> &StaticValue {
        self.classifications.get(&node_id).unwrap_or_else(|| {
            panic!(
                "BUG: No defun classification for NodeId({:?}). Analysis pass missed this node.",
                node_id
            )
        })
    }
}

/// Analyzer for computing defunctionalization classifications.
struct DefunAnalyzer<'a> {
    /// Counter for generating unique lambda IDs
    next_id: usize,

    /// Stack of enclosing declaration names for lambda naming
    enclosing_decl_stack: Vec<String>,

    /// Type table from type checking - maps NodeId to TypeScheme
    type_table: &'a HashMap<NodeId, TypeScheme<TypeName>>,

    /// Set of builtin names to exclude from free variable capture
    builtins: &'a HashSet<String>,

    /// Accumulated classifications
    classifications: HashMap<NodeId, StaticValue>,

    /// Scope tracking for name -> StaticValue (propagating Closure through let bindings)
    scope: ScopeStack<StaticValue>,
}

impl<'a> DefunAnalyzer<'a> {
    fn new(type_table: &'a HashMap<NodeId, TypeScheme<TypeName>>, builtins: &'a HashSet<String>) -> Self {
        DefunAnalyzer {
            next_id: 0,
            enclosing_decl_stack: Vec::new(),
            type_table,
            builtins,
            classifications: HashMap::new(),
            scope: ScopeStack::new(),
        }
    }

    fn fresh_id(&mut self) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Get the type of an expression from the type table
    fn get_expr_type(&self, expr: &Expression) -> Option<Type> {
        self.type_table.get(&expr.h.id).and_then(|scheme| self.get_monotype(scheme)).cloned()
    }

    /// Extract monotype from TypeScheme (must be fully instantiated)
    fn get_monotype<'b>(&self, scheme: &'b TypeScheme<TypeName>) -> Option<&'b Type> {
        match scheme {
            TypeScheme::Monotype(ty) => Some(ty),
            TypeScheme::Polytype { body, .. } => self.get_monotype(body),
        }
    }

    /// Analyze a program and produce DefunAnalysis
    fn analyze_program(&mut self, program: &ast::Program) {
        for decl in &program.declarations {
            self.analyze_declaration(decl);
        }
    }

    /// Analyze a single Decl (function/constant declaration)
    fn analyze_decl(&mut self, d: &ast::Decl) {
        self.enclosing_decl_stack.push(d.name.clone());
        // Register params in scope as Dyn
        self.scope.push_scope();
        for param in &d.params {
            if let Some(name) = param.simple_name() {
                self.scope.insert(name.to_string(), StaticValue::Dyn);
            }
        }
        self.analyze_expr(&d.body);
        self.scope.pop_scope();
        self.enclosing_decl_stack.pop();
    }

    fn analyze_declaration(&mut self, decl: &ast::Declaration) {
        match decl {
            ast::Declaration::Decl(d) => {
                self.analyze_decl(d);
            }
            ast::Declaration::Entry(entry) => {
                self.enclosing_decl_stack.push(entry.name.clone());
                // Register params in scope as Dyn
                self.scope.push_scope();
                for param in &entry.params {
                    if let Some(name) = param.simple_name() {
                        self.scope.insert(name.to_string(), StaticValue::Dyn);
                    }
                }
                self.analyze_expr(&entry.body);
                self.scope.pop_scope();
                self.enclosing_decl_stack.pop();
            }
            // These don't contain expressions that need analysis
            ast::Declaration::Uniform(_)
            | ast::Declaration::Storage(_)
            | ast::Declaration::Sig(_)
            | ast::Declaration::TypeBind(_)
            | ast::Declaration::ModuleBind(_)
            | ast::Declaration::ModuleTypeBind(_)
            | ast::Declaration::Open(_)
            | ast::Declaration::Import(_) => {}
        }
    }

    fn analyze_expr(&mut self, expr: &Expression) {
        match &expr.kind {
            ExprKind::Lambda(lambda) => {
                // Lambda is classified as Closure
                let free_vars = self.compute_free_vars_with_types(lambda, expr);
                let id = self.fresh_id();
                let enclosing = self.enclosing_decl_stack.last().map(|s| s.as_str()).unwrap_or("anon");
                let lam_name = format!("_w_lam_{}_{}", enclosing, id);

                // Store classification (lambda registration happens during flattening)
                self.classifications.insert(expr.h.id, StaticValue::Closure { lam_name, free_vars });

                // Register lambda params in scope as Dyn, then analyze body
                self.scope.push_scope();
                for param in &lambda.params {
                    if let Some(name) = param.simple_name() {
                        self.scope.insert(name.to_string(), StaticValue::Dyn);
                    }
                }
                self.analyze_expr(&lambda.body);
                self.scope.pop_scope();
            }

            // Identifiers: look up in scope to propagate Closure classification
            ExprKind::Identifier(quals, name) => {
                // Only look up unqualified names (qualified names like M.f are always Dyn)
                let sv = if quals.is_empty() {
                    self.scope.lookup(name).cloned().unwrap_or(StaticValue::Dyn)
                } else {
                    StaticValue::Dyn
                };
                self.classifications.insert(expr.h.id, sv);
            }

            // Literals are always Dyn
            ExprKind::IntLiteral(_)
            | ExprKind::FloatLiteral(_)
            | ExprKind::BoolLiteral(_)
            | ExprKind::StringLiteral(_)
            | ExprKind::Unit
            | ExprKind::TypeHole => {
                self.classifications.insert(expr.h.id, StaticValue::Dyn);
            }

            ExprKind::BinaryOp(_, lhs, rhs) => {
                self.analyze_expr(lhs);
                self.analyze_expr(rhs);
                self.classifications.insert(expr.h.id, StaticValue::Dyn);
            }

            ExprKind::UnaryOp(_, operand) => {
                self.analyze_expr(operand);
                self.classifications.insert(expr.h.id, StaticValue::Dyn);
            }

            ExprKind::If(if_expr) => {
                self.analyze_expr(&if_expr.condition);
                self.analyze_expr(&if_expr.then_branch);
                self.analyze_expr(&if_expr.else_branch);
                self.classifications.insert(expr.h.id, StaticValue::Dyn);
            }

            ExprKind::LetIn(let_in) => {
                // Analyze the value first
                self.analyze_expr(&let_in.value);

                // Get the value's classification to propagate through simple name bindings
                let value_sv = self.classifications.get(&let_in.value.h.id).cloned();

                // For simple name patterns, propagate the Closure classification
                self.scope.push_scope();
                if let ast::PatternKind::Name(name) = &let_in.pattern.kind {
                    if let Some(sv) = value_sv {
                        self.scope.insert(name.clone(), sv);
                    }
                }

                // Analyze the body with the new scope
                self.analyze_expr(&let_in.body);
                self.scope.pop_scope();

                // The let expression itself is Dyn (it evaluates to its body)
                self.classifications.insert(expr.h.id, StaticValue::Dyn);
            }

            ExprKind::Application(func, args) => {
                self.analyze_expr(func);
                for arg in args {
                    self.analyze_expr(arg);
                }
                self.classifications.insert(expr.h.id, StaticValue::Dyn);
            }

            ExprKind::Tuple(elems) | ExprKind::ArrayLiteral(elems) | ExprKind::VecMatLiteral(elems) => {
                for elem in elems {
                    self.analyze_expr(elem);
                }
                self.classifications.insert(expr.h.id, StaticValue::Dyn);
            }

            ExprKind::RecordLiteral(fields) => {
                for (_, field_expr) in fields {
                    self.analyze_expr(field_expr);
                }
                self.classifications.insert(expr.h.id, StaticValue::Dyn);
            }

            ExprKind::ArrayIndex(arr, idx) => {
                self.analyze_expr(arr);
                self.analyze_expr(idx);
                self.classifications.insert(expr.h.id, StaticValue::Dyn);
            }

            ExprKind::ArrayWith { array, index, value } => {
                self.analyze_expr(array);
                self.analyze_expr(index);
                self.analyze_expr(value);
                self.classifications.insert(expr.h.id, StaticValue::Dyn);
            }

            ExprKind::FieldAccess(obj, _) => {
                self.analyze_expr(obj);
                self.classifications.insert(expr.h.id, StaticValue::Dyn);
            }

            ExprKind::Loop(loop_expr) => {
                if let Some(init) = &loop_expr.init {
                    self.analyze_expr(init);
                }
                match &loop_expr.form {
                    ast::LoopForm::While(cond) => self.analyze_expr(cond),
                    ast::LoopForm::For(_, bound_expr) => self.analyze_expr(bound_expr),
                    ast::LoopForm::ForIn(_, iter) => self.analyze_expr(iter),
                }
                self.analyze_expr(&loop_expr.body);
                self.classifications.insert(expr.h.id, StaticValue::Dyn);
            }

            ExprKind::TypeAscription(inner, _) | ExprKind::TypeCoercion(inner, _) => {
                self.analyze_expr(inner);
                self.classifications.insert(expr.h.id, StaticValue::Dyn);
            }

            ExprKind::Match(match_expr) => {
                self.analyze_expr(&match_expr.scrutinee);
                for case in &match_expr.cases {
                    self.analyze_expr(&case.body);
                }
                self.classifications.insert(expr.h.id, StaticValue::Dyn);
            }

            ExprKind::Range(range) => {
                self.analyze_expr(&range.start);
                self.analyze_expr(&range.end);
                if let Some(step) = &range.step {
                    self.analyze_expr(step);
                }
                self.classifications.insert(expr.h.id, StaticValue::Dyn);
            }
            ExprKind::Slice(slice) => {
                self.analyze_expr(&slice.array);
                if let Some(start) = &slice.start {
                    self.analyze_expr(start);
                }
                if let Some(end) = &slice.end {
                    self.analyze_expr(end);
                }
                self.classifications.insert(expr.h.id, StaticValue::Dyn);
            }
        }
    }

    /// Compute free variables for a lambda, including their types
    fn compute_free_vars_with_types(
        &self,
        lambda: &ast::LambdaExpr,
        _lambda_expr: &Expression,
    ) -> Vec<(String, Type)> {
        // Collect bound parameters
        let mut bound = HashSet::new();
        for param in &lambda.params {
            if let Some(name) = param.simple_name() {
                bound.insert(name.to_string());
            }
        }

        // Find free variables
        let free_names = self.find_free_variables(&lambda.body, &bound);

        // Sort for deterministic ordering
        let mut sorted_names: Vec<_> = free_names.into_iter().collect();
        sorted_names.sort();

        // Get types for each free variable
        sorted_names
            .into_iter()
            .filter_map(|name| {
                let ty = self.find_var_type_in_expr(&lambda.body, &name)?;
                Some((name, ty))
            })
            .collect()
    }

    /// Find free variables in an expression
    fn find_free_variables(&self, expr: &Expression, bound: &HashSet<String>) -> HashSet<String> {
        let mut free = HashSet::new();
        self.collect_free_vars(expr, bound, &mut free);
        free
    }

    fn collect_free_vars(&self, expr: &Expression, bound: &HashSet<String>, free: &mut HashSet<String>) {
        match &expr.kind {
            ExprKind::Identifier(quals, name) => {
                // Only unqualified names can be free variables
                if quals.is_empty() && !bound.contains(name) && !self.builtins.contains(name) {
                    free.insert(name.clone());
                }
            }
            ExprKind::IntLiteral(_)
            | ExprKind::FloatLiteral(_)
            | ExprKind::BoolLiteral(_)
            | ExprKind::StringLiteral(_)
            | ExprKind::Unit
            | ExprKind::TypeHole => {}
            ExprKind::BinaryOp(_, lhs, rhs) => {
                self.collect_free_vars(lhs, bound, free);
                self.collect_free_vars(rhs, bound, free);
            }
            ExprKind::UnaryOp(_, operand) => {
                self.collect_free_vars(operand, bound, free);
            }
            ExprKind::If(if_expr) => {
                self.collect_free_vars(&if_expr.condition, bound, free);
                self.collect_free_vars(&if_expr.then_branch, bound, free);
                self.collect_free_vars(&if_expr.else_branch, bound, free);
            }
            ExprKind::LetIn(let_in) => {
                self.collect_free_vars(&let_in.value, bound, free);
                let mut extended = bound.clone();
                for name in pattern::bound_names(&let_in.pattern) {
                    extended.insert(name);
                }
                self.collect_free_vars(&let_in.body, &extended, free);
            }
            ExprKind::Lambda(lambda) => {
                let mut extended = bound.clone();
                for param in &lambda.params {
                    if let Some(name) = param.simple_name() {
                        extended.insert(name.to_string());
                    }
                }
                self.collect_free_vars(&lambda.body, &extended, free);
            }
            ExprKind::Application(func, args) => {
                self.collect_free_vars(func, bound, free);
                for arg in args {
                    self.collect_free_vars(arg, bound, free);
                }
            }
            ExprKind::Tuple(elems) | ExprKind::ArrayLiteral(elems) | ExprKind::VecMatLiteral(elems) => {
                for elem in elems {
                    self.collect_free_vars(elem, bound, free);
                }
            }
            ExprKind::RecordLiteral(fields) => {
                for (_, expr) in fields {
                    self.collect_free_vars(expr, bound, free);
                }
            }
            ExprKind::ArrayIndex(arr, idx) => {
                self.collect_free_vars(arr, bound, free);
                self.collect_free_vars(idx, bound, free);
            }
            ExprKind::ArrayWith { array, index, value } => {
                self.collect_free_vars(array, bound, free);
                self.collect_free_vars(index, bound, free);
                self.collect_free_vars(value, bound, free);
            }
            ExprKind::FieldAccess(obj, _) => {
                self.collect_free_vars(obj, bound, free);
            }
            ExprKind::Loop(loop_expr) => {
                if let Some(init) = &loop_expr.init {
                    self.collect_free_vars(init, bound, free);
                }
                let mut extended = bound.clone();
                for name in pattern::bound_names(&loop_expr.pattern) {
                    extended.insert(name);
                }
                match &loop_expr.form {
                    ast::LoopForm::While(cond) => {
                        self.collect_free_vars(cond, &extended, free);
                    }
                    ast::LoopForm::For(var, bound_expr) => {
                        extended.insert(var.clone());
                        self.collect_free_vars(bound_expr, &extended, free);
                    }
                    ast::LoopForm::ForIn(pat, iter) => {
                        self.collect_free_vars(iter, bound, free);
                        for name in pattern::bound_names(pat) {
                            extended.insert(name);
                        }
                    }
                }
                self.collect_free_vars(&loop_expr.body, &extended, free);
            }
            ExprKind::TypeAscription(inner, _) | ExprKind::TypeCoercion(inner, _) => {
                self.collect_free_vars(inner, bound, free);
            }
            ExprKind::Match(match_expr) => {
                self.collect_free_vars(&match_expr.scrutinee, bound, free);
                for case in &match_expr.cases {
                    let mut extended = bound.clone();
                    for name in pattern::bound_names(&case.pattern) {
                        extended.insert(name);
                    }
                    self.collect_free_vars(&case.body, &extended, free);
                }
            }
            ExprKind::Range(range) => {
                self.collect_free_vars(&range.start, bound, free);
                self.collect_free_vars(&range.end, bound, free);
                if let Some(step) = &range.step {
                    self.collect_free_vars(step, bound, free);
                }
            }
            ExprKind::Slice(slice) => {
                self.collect_free_vars(&slice.array, bound, free);
                if let Some(start) = &slice.start {
                    self.collect_free_vars(start, bound, free);
                }
                if let Some(end) = &slice.end {
                    self.collect_free_vars(end, bound, free);
                }
            }
        }
    }

    /// Find the type of a variable by searching for its occurrence in an expression
    fn find_var_type_in_expr(&self, expr: &Expression, var_name: &str) -> Option<Type> {
        match &expr.kind {
            ExprKind::Identifier(quals, name) if quals.is_empty() && name == var_name => {
                self.get_expr_type(expr)
            }
            ExprKind::BinaryOp(_, lhs, rhs) => self
                .find_var_type_in_expr(lhs, var_name)
                .or_else(|| self.find_var_type_in_expr(rhs, var_name)),
            ExprKind::UnaryOp(_, operand) => self.find_var_type_in_expr(operand, var_name),
            ExprKind::If(if_expr) => self
                .find_var_type_in_expr(&if_expr.condition, var_name)
                .or_else(|| self.find_var_type_in_expr(&if_expr.then_branch, var_name))
                .or_else(|| self.find_var_type_in_expr(&if_expr.else_branch, var_name)),
            ExprKind::LetIn(let_in) => self
                .find_var_type_in_expr(&let_in.value, var_name)
                .or_else(|| self.find_var_type_in_expr(&let_in.body, var_name)),
            ExprKind::Application(func, args) => self
                .find_var_type_in_expr(func, var_name)
                .or_else(|| args.iter().find_map(|a| self.find_var_type_in_expr(a, var_name))),
            ExprKind::Tuple(elems) | ExprKind::ArrayLiteral(elems) | ExprKind::VecMatLiteral(elems) => {
                elems.iter().find_map(|e| self.find_var_type_in_expr(e, var_name))
            }
            ExprKind::ArrayIndex(arr, idx) => self
                .find_var_type_in_expr(arr, var_name)
                .or_else(|| self.find_var_type_in_expr(idx, var_name)),
            ExprKind::ArrayWith { array, index, value } => self
                .find_var_type_in_expr(array, var_name)
                .or_else(|| self.find_var_type_in_expr(index, var_name))
                .or_else(|| self.find_var_type_in_expr(value, var_name)),
            ExprKind::FieldAccess(obj, _) => self.find_var_type_in_expr(obj, var_name),
            ExprKind::RecordLiteral(fields) => {
                fields.iter().find_map(|(_, e)| self.find_var_type_in_expr(e, var_name))
            }
            ExprKind::Lambda(lambda) => self.find_var_type_in_expr(&lambda.body, var_name),
            ExprKind::Loop(loop_expr) => {
                if let Some(init) = &loop_expr.init {
                    if let Some(ty) = self.find_var_type_in_expr(init, var_name) {
                        return Some(ty);
                    }
                }
                match &loop_expr.form {
                    ast::LoopForm::While(cond) => {
                        if let Some(ty) = self.find_var_type_in_expr(cond, var_name) {
                            return Some(ty);
                        }
                    }
                    ast::LoopForm::For(_, bound_expr) => {
                        if let Some(ty) = self.find_var_type_in_expr(bound_expr, var_name) {
                            return Some(ty);
                        }
                    }
                    ast::LoopForm::ForIn(_, iter) => {
                        if let Some(ty) = self.find_var_type_in_expr(iter, var_name) {
                            return Some(ty);
                        }
                    }
                }
                self.find_var_type_in_expr(&loop_expr.body, var_name)
            }
            ExprKind::TypeAscription(inner, _) | ExprKind::TypeCoercion(inner, _) => {
                self.find_var_type_in_expr(inner, var_name)
            }
            ExprKind::Match(match_expr) => {
                if let Some(ty) = self.find_var_type_in_expr(&match_expr.scrutinee, var_name) {
                    return Some(ty);
                }
                match_expr.cases.iter().find_map(|c| self.find_var_type_in_expr(&c.body, var_name))
            }
            ExprKind::Range(range) => self
                .find_var_type_in_expr(&range.start, var_name)
                .or_else(|| self.find_var_type_in_expr(&range.end, var_name))
                .or_else(|| range.step.as_ref().and_then(|s| self.find_var_type_in_expr(s, var_name))),
            _ => None,
        }
    }

    fn into_analysis(self) -> DefunAnalysis {
        DefunAnalysis {
            classifications: self.classifications,
        }
    }
}

/// Analyze a program for defunctionalization.
/// Returns the analysis results to be passed to the flattening pass.
pub fn analyze_program(
    program: &ast::Program,
    type_table: &HashMap<NodeId, TypeScheme<TypeName>>,
    builtins: &HashSet<String>,
) -> DefunAnalysis {
    let mut analyzer = DefunAnalyzer::new(type_table, builtins);
    analyzer.analyze_program(program);
    analyzer.into_analysis()
}

/// Analyze a program and additional declarations (e.g., prelude functions).
/// This is used when there are declarations that aren't part of the main AST
/// but still need defunctionalization analysis.
pub fn analyze_program_with_decls(
    program: &ast::Program,
    extra_decls: &[&ast::Decl],
    type_table: &HashMap<NodeId, TypeScheme<TypeName>>,
    builtins: &HashSet<String>,
) -> DefunAnalysis {
    let mut analyzer = DefunAnalyzer::new(type_table, builtins);
    analyzer.analyze_program(program);
    // Analyze additional declarations
    for decl in extra_decls {
        analyzer.analyze_decl(decl);
    }
    analyzer.into_analysis()
}
