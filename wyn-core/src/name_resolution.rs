//! Name resolution pass
//!
//! Resolves module-qualified names by rewriting:
//!   `FieldAccess(Identifier(module), field)` → `Identifier([module], field)`
//! when `module` is a known module name.
//!
//! The same AST-walking machinery is reused by the module-elaboration path
//! (`module_manager::ModuleManager::resolve_names_in_expr`) — the walker is
//! generic over a `ResolveContext` that decides what a given identifier /
//! field-access means in the current mode. See `Resolver` below.
//!
//! After the rewrite pass, `build_name_resolution` populates a
//! `NameResolution` side table that classifies each `Identifier` NodeId as
//! `Builtin(BuiltinId)` (single catalog match) or
//! `OverloadedBuiltin(Vec<BuiltinId>)` (multiple catalog matches under
//! the same surface name). Identifiers not in the catalog (locals,
//! top-level defs, module values) are absent from the side table — the
//! type checker handles them via scope/module lookup.

use std::collections::HashMap;

use crate::ast::{Declaration, ExprKind, Expression, NodeId, Program};
use crate::builtins::{BuiltinCatalog, BuiltinId};
use crate::error::Result;
use crate::module_manager::ModuleManager;
use crate::scope::{ScopeStack, for_each_pattern_name};

/// Insert every name bound by `pattern` into `scope`.
pub(crate) fn collect_pattern_bindings(pattern: &crate::ast::Pattern, scope: &mut ScopeStack<()>) {
    for_each_pattern_name(pattern, &mut |name| {
        scope.insert(name.to_string(), ());
    });
}

// ---------------------------------------------------------------------------
// Visitor — shared AST traversal, mode-specific identifier / field-access
// rewrites plugged in via `ResolveContext`.
// ---------------------------------------------------------------------------

/// Policy interface for a resolver pass. Consumers implement it to inject
/// their mode-specific identifier- and field-access- rewrite logic. The
/// walker handles everything else (recursion, scope push/pop, pattern
/// binding collection).
pub trait ResolveContext {
    /// Called for each `ExprKind::Identifier(quals, name)` leaf. May mutate
    /// `quals` / `name` in place (e.g. to qualify an intra-module ref). The
    /// `scope` argument reflects locals visible at this expression — used
    /// so intra-module rewrites don't shadow a lambda param of the same
    /// name.
    fn resolve_identifier(&self, _quals: &mut Vec<String>, _name: &mut String, _scope: &ScopeStack<()>) {}

    /// Called for each `ExprKind::FieldAccess(obj, field)` where `obj` is
    /// a plain `Identifier(obj_quals, obj_name)`. Return `Some(ExprKind)`
    /// to replace the entire FieldAccess expression (typical case:
    /// `mod.name` collapses to `Identifier([mod], name)`); return `None`
    /// to leave the FieldAccess alone — the walker will then recurse into
    /// `obj` as a regular expression.
    fn resolve_field_access(
        &self,
        _obj_quals: &[String],
        _obj_name: &str,
        _field: &str,
        _scope: &ScopeStack<()>,
    ) -> Option<ExprKind> {
        None
    }
}

/// Walk `expr` in place, applying `ctx`'s rewrite policies. `scope` is
/// the set of locally-bound names visible at `expr`. Callers typically
/// start with an empty `scope` at each declaration boundary.
pub fn walk_expr<C: ResolveContext>(
    expr: &mut Expression,
    ctx: &C,
    scope: &mut ScopeStack<()>,
) -> Result<()> {
    match &mut expr.kind {
        ExprKind::Identifier(quals, name) => {
            ctx.resolve_identifier(quals, name, scope);
        }
        ExprKind::FieldAccess(obj, field) => {
            // Peek at obj to see if it's the `Identifier . name` shape the
            // resolver hook wants to rewrite. If so, replace the whole
            // expression; otherwise fall through and recurse into obj.
            let rewrite = if let ExprKind::Identifier(obj_quals, obj_name) = &obj.kind {
                ctx.resolve_field_access(obj_quals, obj_name, field, scope)
            } else {
                None
            };
            if let Some(new_kind) = rewrite {
                expr.kind = new_kind;
            } else {
                walk_expr(obj, ctx, scope)?;
            }
        }
        ExprKind::Application(func, args) => {
            walk_expr(func, ctx, scope)?;
            for a in args {
                walk_expr(a, ctx, scope)?;
            }
        }
        ExprKind::Lambda(lambda) => {
            scope.push_scope();
            for p in &lambda.params {
                collect_pattern_bindings(p, scope);
            }
            walk_expr(&mut lambda.body, ctx, scope)?;
            scope.pop_scope();
        }
        ExprKind::LetIn(let_in) => {
            walk_expr(&mut let_in.value, ctx, scope)?;
            scope.push_scope();
            collect_pattern_bindings(&let_in.pattern, scope);
            walk_expr(&mut let_in.body, ctx, scope)?;
            scope.pop_scope();
        }
        ExprKind::If(if_expr) => {
            walk_expr(&mut if_expr.condition, ctx, scope)?;
            walk_expr(&mut if_expr.then_branch, ctx, scope)?;
            walk_expr(&mut if_expr.else_branch, ctx, scope)?;
        }
        ExprKind::BinaryOp(_, lhs, rhs) => {
            walk_expr(lhs, ctx, scope)?;
            walk_expr(rhs, ctx, scope)?;
        }
        ExprKind::UnaryOp(_, operand) => {
            walk_expr(operand, ctx, scope)?;
        }
        ExprKind::Tuple(exprs) | ExprKind::ArrayLiteral(exprs) | ExprKind::VecMatLiteral(exprs) => {
            for e in exprs {
                walk_expr(e, ctx, scope)?;
            }
        }
        ExprKind::ArrayIndex(arr, idx) => {
            walk_expr(arr, ctx, scope)?;
            walk_expr(idx, ctx, scope)?;
        }
        ExprKind::ArrayWith {
            array, index, value, ..
        } => {
            walk_expr(array, ctx, scope)?;
            walk_expr(index, ctx, scope)?;
            walk_expr(value, ctx, scope)?;
        }
        ExprKind::VecWith { target, value, .. } => {
            walk_expr(target, ctx, scope)?;
            walk_expr(value, ctx, scope)?;
        }
        ExprKind::RecordWith { record, value, .. } => {
            walk_expr(record, ctx, scope)?;
            walk_expr(value, ctx, scope)?;
        }
        ExprKind::RecordLiteral(fields) => {
            for (_, e) in fields {
                walk_expr(e, ctx, scope)?;
            }
        }
        ExprKind::Loop(loop_expr) => {
            scope.push_scope();
            collect_pattern_bindings(&loop_expr.pattern, scope);
            if let Some(ref mut init) = loop_expr.init {
                walk_expr(init, ctx, scope)?;
            }
            match &mut loop_expr.form {
                crate::ast::LoopForm::While(cond) => {
                    walk_expr(cond, ctx, scope)?;
                }
                crate::ast::LoopForm::For(idx_var, bound) => {
                    scope.insert(idx_var.clone(), ());
                    walk_expr(bound, ctx, scope)?;
                }
                crate::ast::LoopForm::ForIn(elem_pat, iter) => {
                    collect_pattern_bindings(elem_pat, scope);
                    walk_expr(iter, ctx, scope)?;
                }
            }
            walk_expr(&mut loop_expr.body, ctx, scope)?;
            scope.pop_scope();
        }
        ExprKind::Match(match_expr) => {
            walk_expr(&mut match_expr.scrutinee, ctx, scope)?;
            for case in &mut match_expr.cases {
                scope.push_scope();
                collect_pattern_bindings(&case.pattern, scope);
                walk_expr(&mut case.body, ctx, scope)?;
                scope.pop_scope();
            }
        }
        ExprKind::TypeAscription(e, _) | ExprKind::TypeCoercion(e, _) => {
            walk_expr(e, ctx, scope)?;
        }
        ExprKind::Range(range) => {
            walk_expr(&mut range.start, ctx, scope)?;
            walk_expr(&mut range.end, ctx, scope)?;
            if let Some(ref mut step) = range.step {
                walk_expr(step, ctx, scope)?;
            }
        }
        ExprKind::Slice(slice) => {
            walk_expr(&mut slice.array, ctx, scope)?;
            if let Some(ref mut start) = slice.start {
                walk_expr(start, ctx, scope)?;
            }
            if let Some(ref mut end) = slice.end {
                walk_expr(end, ctx, scope)?;
            }
        }
        ExprKind::Constructor(_, args) => {
            for a in args {
                walk_expr(a, ctx, scope)?;
            }
        }
        ExprKind::IntLiteral(_)
        | ExprKind::FloatLiteral(_)
        | ExprKind::BoolLiteral(_)
        | ExprKind::Unit
        | ExprKind::TypeHole => {}
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Program-mode resolver (top-level pass over user code)
// ---------------------------------------------------------------------------

/// Context used by the program-level name-resolution pass: rewrites
/// `mod.name` to `Identifier([mod], name)` when `mod` is a registered
/// module.
struct ProgramResolver<'a> {
    module_manager: &'a ModuleManager,
}

impl<'a> ResolveContext for ProgramResolver<'a> {
    fn resolve_field_access(
        &self,
        obj_quals: &[String],
        obj_name: &str,
        field: &str,
        _scope: &ScopeStack<()>,
    ) -> Option<ExprKind> {
        if obj_quals.is_empty() && self.module_manager.is_known_module(obj_name) {
            Some(ExprKind::Identifier(
                vec![obj_name.to_string()],
                field.to_string(),
            ))
        } else {
            None
        }
    }
}

/// Resolve names in a program by rewriting FieldAccess -> QualifiedName.
pub fn run(program: &mut Program, module_manager: &ModuleManager) -> Result<()> {
    for decl in &mut program.declarations {
        resolve_declaration(decl, module_manager)?;
    }
    Ok(())
}

/// Resolve names in a single Decl (for prelude functions).
pub fn resolve_decl(decl: &mut crate::ast::Decl, module_manager: &ModuleManager) -> Result<()> {
    let ctx = ProgramResolver { module_manager };
    let mut scope = ScopeStack::new();
    walk_expr(&mut decl.body, &ctx, &mut scope)
}

fn resolve_declaration(decl: &mut Declaration, module_manager: &ModuleManager) -> Result<()> {
    let ctx = ProgramResolver { module_manager };
    let mut scope = ScopeStack::new();
    match decl {
        Declaration::Decl(d) => walk_expr(&mut d.body, &ctx, &mut scope),
        Declaration::Entry(entry) => walk_expr(&mut entry.body, &ctx, &mut scope),
        Declaration::Sig(_) => Ok(()),
        _ => Ok(()),
    }
}

// ---------------------------------------------------------------------------
// Side-table builder: classify each Identifier NodeId against the catalog
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum ResolvedValueRef {
    /// Catalog entry matching this identifier's surface name. The
    /// entry's `overloads` slice may contain one or many overloads;
    /// overload selection happens later (Phase 3) once arg types are
    /// known.
    Builtin(BuiltinId),
}

/// Side table populated by `build_name_resolution`. Maps Identifier
/// NodeIds to their catalog classification. Identifiers not in the
/// catalog (locals, top-level defs, module values) are absent.
#[derive(Debug, Default)]
pub struct NameResolution {
    pub values: HashMap<NodeId, ResolvedValueRef>,
}

impl NameResolution {
    pub fn get(&self, id: NodeId) -> Option<&ResolvedValueRef> {
        self.values.get(&id)
    }
}

/// Walk the program after module-qualification rewrite and build the
/// side table. Top-level def names are pushed into scope first, so
/// user code that shadows a builtin (e.g. `def length = ...`) is
/// classified as a non-builtin.
pub fn build_name_resolution(program: &Program, catalog: &BuiltinCatalog) -> NameResolution {
    let mut nr = NameResolution::default();
    let mut top_level: ScopeStack<()> = ScopeStack::new();
    collect_top_level_names(&program.declarations, &mut top_level);

    for decl in &program.declarations {
        let mut scope = top_level.clone();
        match decl {
            Declaration::Decl(d) => {
                scope.push_scope();
                for p in &d.params {
                    collect_pattern_bindings(p, &mut scope);
                }
                walk_resolution(&d.body, catalog, &mut scope, &mut nr);
                scope.pop_scope();
            }
            Declaration::Entry(entry) => {
                scope.push_scope();
                for p in &entry.params {
                    collect_pattern_bindings(p, &mut scope);
                }
                walk_resolution(&entry.body, catalog, &mut scope, &mut nr);
                scope.pop_scope();
            }
            _ => {}
        }
    }
    nr
}

fn collect_top_level_names(decls: &[Declaration], scope: &mut ScopeStack<()>) {
    for decl in decls {
        match decl {
            Declaration::Decl(d) => {
                scope.insert(d.name.clone(), ());
            }
            Declaration::Entry(entry) => {
                scope.insert(entry.name.clone(), ());
            }
            _ => {}
        }
    }
}

/// Walk an expression tree, recording catalog classifications for each
/// `ExprKind::Identifier` whose surface name is not lexically shadowed.
fn walk_resolution(
    expr: &Expression,
    catalog: &BuiltinCatalog,
    scope: &mut ScopeStack<()>,
    nr: &mut NameResolution,
) {
    match &expr.kind {
        ExprKind::Identifier(quals, name) => {
            // Unqualified name shadowed by a local? Skip — let the
            // checker resolve via scope.
            if quals.is_empty() && scope.lookup(name).is_some() {
                return;
            }
            let full_name =
                if quals.is_empty() { name.clone() } else { format!("{}.{}", quals.join("."), name) };
            if let Some(def) = catalog.lookup_by_surface_name(&full_name) {
                nr.values.insert(expr.h.id, ResolvedValueRef::Builtin(def.id));
            }
        }
        ExprKind::Application(func, args) => {
            walk_resolution(func, catalog, scope, nr);
            for arg in args {
                walk_resolution(arg, catalog, scope, nr);
            }
        }
        ExprKind::Lambda(lambda) => {
            scope.push_scope();
            for p in &lambda.params {
                collect_pattern_bindings(p, scope);
            }
            walk_resolution(&lambda.body, catalog, scope, nr);
            scope.pop_scope();
        }
        ExprKind::LetIn(let_in) => {
            walk_resolution(&let_in.value, catalog, scope, nr);
            scope.push_scope();
            collect_pattern_bindings(&let_in.pattern, scope);
            walk_resolution(&let_in.body, catalog, scope, nr);
            scope.pop_scope();
        }
        ExprKind::If(if_expr) => {
            walk_resolution(&if_expr.condition, catalog, scope, nr);
            walk_resolution(&if_expr.then_branch, catalog, scope, nr);
            walk_resolution(&if_expr.else_branch, catalog, scope, nr);
        }
        ExprKind::BinaryOp(_, lhs, rhs) => {
            walk_resolution(lhs, catalog, scope, nr);
            walk_resolution(rhs, catalog, scope, nr);
        }
        ExprKind::UnaryOp(_, operand) => walk_resolution(operand, catalog, scope, nr),
        ExprKind::Tuple(exprs) | ExprKind::ArrayLiteral(exprs) | ExprKind::VecMatLiteral(exprs) => {
            for e in exprs {
                walk_resolution(e, catalog, scope, nr);
            }
        }
        ExprKind::ArrayIndex(arr, idx) => {
            walk_resolution(arr, catalog, scope, nr);
            walk_resolution(idx, catalog, scope, nr);
        }
        ExprKind::ArrayWith {
            array, index, value, ..
        } => {
            walk_resolution(array, catalog, scope, nr);
            walk_resolution(index, catalog, scope, nr);
            walk_resolution(value, catalog, scope, nr);
        }
        ExprKind::VecWith { target, value, .. } => {
            walk_resolution(target, catalog, scope, nr);
            walk_resolution(value, catalog, scope, nr);
        }
        ExprKind::RecordWith { record, value, .. } => {
            walk_resolution(record, catalog, scope, nr);
            walk_resolution(value, catalog, scope, nr);
        }
        ExprKind::RecordLiteral(fields) => {
            for (_, e) in fields {
                walk_resolution(e, catalog, scope, nr);
            }
        }
        ExprKind::Loop(loop_expr) => {
            use crate::ast::LoopForm;
            scope.push_scope();
            collect_pattern_bindings(&loop_expr.pattern, scope);
            if let Some(init) = &loop_expr.init {
                walk_resolution(init, catalog, scope, nr);
            }
            match &loop_expr.form {
                LoopForm::While(cond) => walk_resolution(cond, catalog, scope, nr),
                LoopForm::For(idx_var, bound) => {
                    scope.insert(idx_var.clone(), ());
                    walk_resolution(bound, catalog, scope, nr);
                }
                LoopForm::ForIn(elem_pat, iter) => {
                    collect_pattern_bindings(elem_pat, scope);
                    walk_resolution(iter, catalog, scope, nr);
                }
            }
            walk_resolution(&loop_expr.body, catalog, scope, nr);
            scope.pop_scope();
        }
        ExprKind::Match(match_expr) => {
            walk_resolution(&match_expr.scrutinee, catalog, scope, nr);
            for case in &match_expr.cases {
                scope.push_scope();
                collect_pattern_bindings(&case.pattern, scope);
                walk_resolution(&case.body, catalog, scope, nr);
                scope.pop_scope();
            }
        }
        ExprKind::TypeAscription(e, _) | ExprKind::TypeCoercion(e, _) => {
            walk_resolution(e, catalog, scope, nr);
        }
        ExprKind::FieldAccess(obj, _) => {
            walk_resolution(obj, catalog, scope, nr);
        }
        ExprKind::Range(range) => {
            walk_resolution(&range.start, catalog, scope, nr);
            if let Some(step) = &range.step {
                walk_resolution(step, catalog, scope, nr);
            }
            walk_resolution(&range.end, catalog, scope, nr);
        }
        ExprKind::Slice(slice) => {
            walk_resolution(&slice.array, catalog, scope, nr);
            if let Some(start) = &slice.start {
                walk_resolution(start, catalog, scope, nr);
            }
            if let Some(end) = &slice.end {
                walk_resolution(end, catalog, scope, nr);
            }
        }
        ExprKind::Constructor(_, args) => {
            for a in args {
                walk_resolution(a, catalog, scope, nr);
            }
        }
        ExprKind::IntLiteral(_)
        | ExprKind::FloatLiteral(_)
        | ExprKind::BoolLiteral(_)
        | ExprKind::Unit
        | ExprKind::TypeHole => {}
    }
}
