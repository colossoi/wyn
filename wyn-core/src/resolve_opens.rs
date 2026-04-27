//! `open M` name-resolution pass.
//!
//! Runs after elaboration / `resolve_placeholders` and before type
//! checking. Walks every `Declaration` and `Expression`, maintaining a
//! stack of currently-open module names, and rewrites unqualified
//! identifiers to fully-qualified form whenever a unique opened module
//! provides the name. Downstream passes (type checker, TLC) only ever
//! see qualified names produced by this pass.
//!
//! Resolution rules per identifier — first match wins:
//!
//! 1. **Already qualified** (`M.x`): leave alone.
//! 2. **Locally bound** (lambda param, `let`, function name in scope):
//!    leave alone. Local bindings always shadow opens.
//! 3. **Opened-module candidates**, searched innermost-first: collect
//!    every opened `M` such that `M.name` exists in the value index.
//!    * exactly one candidate → rewrite `Identifier([], name)` →
//!      `Identifier([M], name)`.
//!    * multiple candidates → ambiguous-name error.
//! 4. **None of the above**: leave bare. Downstream passes resolve via
//!    the top-level `INTRINSIC_RENAMES` shortcut, prelude defs, or
//!    surface "undefined variable".
//!
//! `open M` itself is validated: an unknown module name errors at the
//! `open` site rather than letting a typo cascade into confusing
//! "undefined `cos`" diagnostics later.

use crate::ast::{self, Declaration, ExprKind, Expression, ModuleExpression, Pattern, Program};
use crate::error::Result;
use std::collections::HashSet;

#[cfg(test)]
#[path = "resolve_opens_tests.rs"]
mod tests;

/// Member-source abstraction. Decouples the resolver from where module
/// member info lives so later changes (real value exports, user
/// modules, etc.) update only the index construction.
#[derive(Debug, Default, Clone)]
pub struct OpenIndex {
    /// Modules known to the program. Anything not in this set fails
    /// `open` validation regardless of whether some member of that name
    /// happens to be registered elsewhere.
    modules: HashSet<String>,
    /// Value-namespace members per module. `(module, name)` pairs.
    members: HashSet<(String, String)>,
}

impl OpenIndex {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn has_module(&self, m: &str) -> bool {
        self.modules.contains(m)
    }

    pub fn has_member(&self, m: &str, name: &str) -> bool {
        self.members.contains(&(m.to_string(), name.to_string()))
    }

    /// Insert a `(module, name)` pair, also marking the module as
    /// known. Used to build the index from spec_schemes / impl_source.
    pub fn add_member(&mut self, module: &str, name: &str) {
        self.modules.insert(module.to_string());
        self.members.insert((module.to_string(), name.to_string()));
    }

    /// Build an index from any iterator of `"M.name"`-style keys (the
    /// shape of `PlaceholderResolver::spec_schemes` and
    /// `ImplSource`'s registry). Keys without exactly one dot are
    /// ignored — they're function names not module members.
    pub fn from_qualified_names<I, S>(keys: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut idx = OpenIndex::new();
        for key in keys {
            let k = key.as_ref();
            // Exactly one '.' splits a key into (module, member). The
            // ImplSource also registers width-conversion entries like
            // "i32.f32"; those still parse as (module=i32, name=f32)
            // and become legitimately reachable through `open i32`.
            if let Some((m, n)) = k.split_once('.') {
                if !m.is_empty() && !n.is_empty() && !n.contains('.') {
                    idx.add_member(m, n);
                }
            }
        }
        idx
    }
}

/// Resolver state — open stack + locals frame stack + index.
pub struct OpenResolver<'a> {
    index: &'a OpenIndex,
    /// Modules opened at the current point in the walk, innermost-last.
    opens: Vec<String>,
    /// Stack of frames; each frame holds the names introduced by
    /// patterns / let-bindings / function params at that nesting
    /// level. Lookup walks frames top-down.
    locals: Vec<HashSet<String>>,
}

impl<'a> OpenResolver<'a> {
    pub fn new(index: &'a OpenIndex) -> Self {
        Self {
            index,
            opens: Vec::new(),
            locals: vec![HashSet::new()],
        }
    }

    /// Top-level entry: rewrite identifiers in every declaration of a
    /// program in source order.
    pub fn resolve_program(&mut self, program: &mut Program) -> Result<()> {
        // Top-level `def` / `entry` / `extern` names are all visible
        // in each other's bodies (mutual recursion). Pre-load the
        // outermost `locals` frame with every top-level def/entry/sig
        // name so a top-level function never accidentally rewrites to
        // an opened module member.
        for decl in &program.declarations {
            if let Some(name) = top_level_name(decl) {
                self.locals.last_mut().unwrap().insert(name);
            }
        }
        for decl in &mut program.declarations {
            self.resolve_declaration(decl)?;
        }
        Ok(())
    }

    fn resolve_declaration(&mut self, decl: &mut Declaration) -> Result<()> {
        match decl {
            Declaration::Open(mod_exp) => self.handle_open(mod_exp),
            Declaration::Decl(d) => {
                self.locals.push(HashSet::new());
                for p in &d.params {
                    self.bind_pattern(p);
                }
                self.resolve_expression(&mut d.body)?;
                self.locals.pop();
                Ok(())
            }
            Declaration::Entry(e) => {
                self.locals.push(HashSet::new());
                for p in &e.params {
                    self.bind_pattern(p);
                }
                self.resolve_expression(&mut e.body)?;
                self.locals.pop();
                Ok(())
            }
            // Opens are only rewritten in expression position; module
            // declarations and signatures don't contain expressions
            // that reach the value namespace.
            Declaration::Module(_)
            | Declaration::ModuleTypeBind(_)
            | Declaration::Sig(_)
            | Declaration::TypeBind(_)
            | Declaration::Uniform(_)
            | Declaration::Storage(_)
            | Declaration::Extern(_)
            | Declaration::Import(_) => Ok(()),
        }
    }

    fn handle_open(&mut self, mod_exp: &ModuleExpression) -> Result<()> {
        match mod_exp {
            ModuleExpression::Name(name) => {
                if !self.index.has_module(name) {
                    return Err(crate::err_module!(
                        "open: unknown module '{}'. No module by that name is in scope.",
                        name
                    ));
                }
                self.opens.push(name.clone());
                Ok(())
            }
            // Other module-expression forms parse but we don't
            // support them as `open` targets yet — fail loudly so
            // typos and unsupported shapes don't silently no-op.
            _ => Err(crate::err_module!(
                "open: only `open <ModuleName>` is supported in this resolver pass; other module expressions (lambda, application, struct, import, ascription) are not yet handled."
            )),
        }
    }

    fn bind_pattern(&mut self, pat: &Pattern) {
        let frame = self.locals.last_mut().unwrap();
        for n in pat.bound_names() {
            frame.insert(n);
        }
    }

    fn locally_bound(&self, name: &str) -> bool {
        self.locals.iter().any(|frame| frame.contains(name))
    }

    fn resolve_expression(&mut self, expr: &mut Expression) -> Result<()> {
        // Identifier rewrite — the whole point of the pass.
        if let ExprKind::Identifier(quals, name) = &expr.kind {
            // Rule 1: already qualified.
            if !quals.is_empty() {
                return Ok(());
            }
            // Rule 2: locally bound.
            if self.locally_bound(name) {
                return Ok(());
            }
            // Rule 3: opened-module candidates.
            let candidates: Vec<String> =
                self.opens.iter().rev().filter(|m| self.index.has_member(m, name)).cloned().collect();
            match candidates.len() {
                0 => {} // Rule 4: leave bare for downstream resolution.
                1 => {
                    let m = candidates.into_iter().next().unwrap();
                    if let ExprKind::Identifier(quals, _) = &mut expr.kind {
                        quals.push(m);
                    }
                }
                _ => {
                    let cand_list = candidates
                        .iter()
                        .map(|m| format!("`{}.{}`", m, name))
                        .collect::<Vec<_>>()
                        .join(", ");
                    return Err(crate::err_module_at!(
                        expr.h.span,
                        "ambiguous reference '{}'; opened modules provide: {}. Qualify the reference explicitly.",
                        name,
                        cand_list
                    ));
                }
            }
            return Ok(());
        }

        // Compound forms — descend, pushing/popping local frames at
        // binding sites so introduced names shadow opens correctly.
        match &mut expr.kind {
            ExprKind::Lambda(lambda) => {
                self.locals.push(HashSet::new());
                for p in &lambda.params {
                    self.bind_pattern(p);
                }
                self.resolve_expression(&mut lambda.body)?;
                self.locals.pop();
            }
            ExprKind::Application(func, args) => {
                self.resolve_expression(func)?;
                for a in args {
                    self.resolve_expression(a)?;
                }
            }
            ExprKind::LetIn(let_in) => {
                // Value is in the parent frame's scope.
                self.resolve_expression(&mut let_in.value)?;
                // Body sees the let-bound names.
                self.locals.push(HashSet::new());
                self.bind_pattern(&let_in.pattern);
                self.resolve_expression(&mut let_in.body)?;
                self.locals.pop();
            }
            ExprKind::If(if_expr) => {
                self.resolve_expression(&mut if_expr.condition)?;
                self.resolve_expression(&mut if_expr.then_branch)?;
                self.resolve_expression(&mut if_expr.else_branch)?;
            }
            ExprKind::BinaryOp(_, l, r) => {
                self.resolve_expression(l)?;
                self.resolve_expression(r)?;
            }
            ExprKind::UnaryOp(_, op) => {
                self.resolve_expression(op)?;
            }
            ExprKind::ArrayIndex(arr, idx) => {
                self.resolve_expression(arr)?;
                self.resolve_expression(idx)?;
            }
            ExprKind::ArrayWith {
                array, index, value, ..
            } => {
                self.resolve_expression(array)?;
                self.resolve_expression(index)?;
                self.resolve_expression(value)?;
            }
            ExprKind::Slice(slice) => {
                self.resolve_expression(&mut slice.array)?;
                if let Some(s) = &mut slice.start {
                    self.resolve_expression(s)?;
                }
                if let Some(e) = &mut slice.end {
                    self.resolve_expression(e)?;
                }
            }
            ExprKind::ArrayLiteral(es) | ExprKind::VecMatLiteral(es) | ExprKind::Tuple(es) => {
                for e in es {
                    self.resolve_expression(e)?;
                }
            }
            ExprKind::RecordLiteral(fields) => {
                for (_, v) in fields {
                    self.resolve_expression(v)?;
                }
            }
            ExprKind::FieldAccess(inner, _) => {
                self.resolve_expression(inner)?;
            }
            ExprKind::Range(range) => {
                self.resolve_expression(&mut range.start)?;
                if let Some(s) = &mut range.step {
                    self.resolve_expression(s)?;
                }
                self.resolve_expression(&mut range.end)?;
            }
            ExprKind::Match(m) => {
                self.resolve_expression(&mut m.scrutinee)?;
                for case in &mut m.cases {
                    self.locals.push(HashSet::new());
                    self.bind_pattern(&case.pattern);
                    self.resolve_expression(&mut case.body)?;
                    self.locals.pop();
                }
            }
            ExprKind::TypeAscription(e, _) | ExprKind::TypeCoercion(e, _) => {
                self.resolve_expression(e)?;
            }
            ExprKind::Loop(loop_expr) => {
                if let Some(init) = &mut loop_expr.init {
                    self.resolve_expression(init)?;
                }
                // Loop pattern binds names visible in form + body.
                self.locals.push(HashSet::new());
                self.bind_pattern(&loop_expr.pattern);
                match &mut loop_expr.form {
                    ast::LoopForm::For(_, bound) => {
                        self.resolve_expression(bound)?;
                    }
                    ast::LoopForm::ForIn(pat, iter) => {
                        self.bind_pattern(pat);
                        self.resolve_expression(iter)?;
                    }
                    ast::LoopForm::While(cond) => {
                        self.resolve_expression(cond)?;
                    }
                }
                self.resolve_expression(&mut loop_expr.body)?;
                self.locals.pop();
            }
            // Leaves — no expressions inside.
            ExprKind::Identifier(_, _)
            | ExprKind::IntLiteral(_)
            | ExprKind::FloatLiteral(_)
            | ExprKind::BoolLiteral(_)
            | ExprKind::Unit
            | ExprKind::TypeHole => {}
        }

        Ok(())
    }
}

/// Top-level public entry point. Builds the open index from the given
/// `qualified_keys` (typically the union of `spec_schemes`'s keys and
/// `ImplSource::all_names()`) and rewrites the program.
pub fn run<I, S>(program: &mut Program, qualified_keys: I) -> Result<()>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let index = OpenIndex::from_qualified_names(qualified_keys);
    let mut r = OpenResolver::new(&index);
    r.resolve_program(program)
}

fn top_level_name(decl: &Declaration) -> Option<String> {
    match decl {
        Declaration::Decl(d) => Some(d.name.clone()),
        Declaration::Entry(e) => Some(e.name.clone()),
        Declaration::Sig(s) => Some(s.name.clone()),
        Declaration::Extern(e) => Some(e.name.clone()),
        Declaration::Uniform(u) => Some(u.name.clone()),
        Declaration::Storage(s) => Some(s.name.clone()),
        // No value-namespace name to register.
        Declaration::Module(_)
        | Declaration::ModuleTypeBind(_)
        | Declaration::TypeBind(_)
        | Declaration::Open(_)
        | Declaration::Import(_) => None,
    }
}
