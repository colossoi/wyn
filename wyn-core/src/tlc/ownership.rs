//! TLC-level ownership and liveness analysis.
//!
//! Three layers:
//!
//! 1. **Build** — assign an `OwnerId` to every non-copy runtime slot
//!    (function parameter, let-binding, loop carry, SOAC element
//!    parameter). Record per-term `uses`, `kills`, and `defs` sets.
//!    Variables that *alias* an existing owner share its `OwnerId`.
//!
//! 2. **Liveness** — backward dataflow over the structured TLC tree,
//!    with fixed-point iteration over loops and SOAC bodies. Records
//!    per-term `live_out`.
//!
//! 3. **Use** — two consumers read the populated model:
//!    - use-after-move checking: an owner in `kills[T] ∩ live_out[T]`
//!      is consumed at `T` while a successor still needs it.
//!    - in-place promotion: at each `_w_intrinsic_array_with` call,
//!      promote to `_w_intrinsic_array_with_inplace` when the source's
//!      owner is mutable and absent from `live_out`.

use std::collections::{HashMap, HashSet};

use crate::SymbolId;
use crate::ast::{Span, TypeName};
use crate::tlc::{ArrayExpr, Def, DefMeta, Lambda, LoopKind, Program, SoacOp, Term, TermId, TermKind};
use crate::types;
use polytype::Type;

/// A unique identifier for an owned (non-copy) runtime slot.
///
/// One owner per allocation event:
/// - parameter slot for each non-copy parameter
/// - let-binding whose rhs introduces a fresh value
/// - loop carry / loop variable
/// - per-iteration SOAC element parameter
///
/// Variables aliasing an existing slot share the slot's owner (e.g.
/// `let b = a in ...` registers `b → owner(a)`, no new owner created).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OwnerId(u32);

impl OwnerId {
    pub fn raw(self) -> u32 {
        self.0
    }
}

/// Provenance of an owner. Drives the mutability decision used by
/// promotion: only mutable owners are eligible for in-place
/// promotion, regardless of liveness.
///
/// `NonUniqueParam` represents a `T` parameter (no `*`). The caller
/// retains ownership, so mutating it would clobber the caller's
/// memory. Promotion must reject these.
///
/// `Entry` is the implicit-unique rule — entry parameters are
/// handed by the host as exclusive ownership at the boundary, so they
/// behave as `*T` regardless of how they're written in source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Origin {
    /// Allocated by this function: literal, range, generator, SOAC
    /// output, function call returning `*T`.
    Fresh,
    /// Bound from a `*T` parameter — caller surrendered ownership.
    UniqueParam,
    /// Bound from a `T` parameter — caller still owns. Not mutable.
    NonUniqueParam,
    /// Entry parameter — implicitly unique.
    Entry,
    /// Result of an unrecognized non-copy producer (e.g. a user
    /// function whose body might return an alias of one of its args,
    /// or a lambda whose body returns a capture). Without
    /// inter-procedural summaries we can't know what this aliases,
    /// so promotion treats it as immutable. Sound under-approximation
    /// of "borrowing" — refusing the optimization is always safe.
    Borrowed,
}

impl Origin {
    pub fn is_mutable(self) -> bool {
        matches!(self, Origin::Fresh | Origin::UniqueParam | Origin::Entry)
    }
}

/// The result of the ownership analysis.
///
/// Build phase fills `var_to_owner`, `origins`, `uses`, `kills`,
/// `defs`, `term_spans`, and `owner_to_var`. Liveness fills
/// `live_out`. Use-after-move checking and promotion both consume
/// the populated model.
#[derive(Default, Debug)]
pub struct OwnershipModel {
    /// SymbolId → OwnerId. Populated as bindings are visited.
    pub var_to_owner: HashMap<SymbolId, OwnerId>,
    /// First binder name for each owner — used to make
    /// use-after-move error messages user-readable. Populated when an
    /// owner is created, never overwritten.
    pub owner_to_var: HashMap<OwnerId, SymbolId>,
    /// Per-owner provenance.
    pub origins: HashMap<OwnerId, Origin>,
    /// Per-term span — for error reporting.
    pub term_spans: HashMap<TermId, Span>,
    /// Per-term: owners read here.
    pub uses: HashMap<TermId, HashSet<OwnerId>>,
    /// Per-term: owners consumed/moved here (e.g. function arg with
    /// `*T` parameter type).
    pub kills: HashMap<TermId, HashSet<OwnerId>>,
    /// Per-term: owners newly introduced here.
    pub defs: HashMap<TermId, HashSet<OwnerId>>,
    /// Per-term: live-out set. Empty until liveness runs.
    pub live_out: HashMap<TermId, HashSet<OwnerId>>,
}

impl OwnershipModel {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn owner_of(&self, sym: SymbolId) -> Option<OwnerId> {
        self.var_to_owner.get(&sym).copied()
    }

    pub fn origin(&self, owner: OwnerId) -> Option<Origin> {
        self.origins.get(&owner).copied()
    }
}

/// Build per-term effect sets, then compute backward liveness with
/// fixed-point over loops and SOAC bodies. The returned model has
/// `live_out` populated for every term reachable from a function body.
pub fn analyze(program: &Program) -> OwnershipModel {
    let mut model = build(program);
    let mut analyzer = Liveness { model: &mut model };
    for def in &program.defs {
        analyzer.analyze_def(def);
    }
    model
}

/// Walk the program and populate the model's effect sets, leaving
/// `live_out` empty. Exposed for testing the build step in isolation;
/// production code should call `analyze` instead.
pub fn build(program: &Program) -> OwnershipModel {
    let mut builder = Builder::new(program);
    for def in &program.defs {
        builder.visit_def(def);
    }
    builder.model
}

struct Builder<'p> {
    model: OwnershipModel,
    next_owner: u32,
    program: &'p Program,
}

impl<'p> Builder<'p> {
    fn new(program: &'p Program) -> Self {
        Self {
            model: OwnershipModel::new(),
            next_owner: 0,
            program,
        }
    }

    fn fresh_owner(&mut self, origin: Origin) -> OwnerId {
        let id = OwnerId(self.next_owner);
        self.next_owner += 1;
        self.model.origins.insert(id, origin);
        id
    }

    fn bind(&mut self, sym: SymbolId, owner: OwnerId) {
        self.model.var_to_owner.insert(sym, owner);
        // First binder wins — keeps the most "canonical" name (param
        // or fresh-let) for error messages.
        self.model.owner_to_var.entry(owner).or_insert(sym);
    }

    fn visit_def(&mut self, def: &Def) {
        // The function's parameters live on the top-level Lambda's
        // params field — Defs whose body is anything else are zero-arity
        // constants and have no slots to track here.
        if let TermKind::Lambda(lam) = &def.body.kind {
            self.bind_params(lam, matches!(def.meta, DefMeta::EntryPoint(_)));
            self.visit_term(&lam.body);
        } else {
            self.visit_term(&def.body);
        }
    }

    fn bind_params(&mut self, lam: &Lambda, is_entry: bool) {
        for (sym, ty) in &lam.params {
            if types::is_copy(ty) {
                continue;
            }
            let origin = if is_entry {
                Origin::Entry
            } else if types::is_unique(ty) {
                Origin::UniqueParam
            } else {
                Origin::NonUniqueParam
            };
            let owner = self.fresh_owner(origin);
            self.bind(*sym, owner);
        }
    }

    /// Recursive descent over a term tree. Records per-term effect
    /// sets (`uses`, `defs`, `kills`) and threads alias bindings
    /// through `Let`s. Constructs that introduce per-iteration or
    /// per-call bindings (Lambda, Loop, SOACs) are handled
    /// explicitly so their parameters are bound before the body is
    /// visited.
    fn visit_term(&mut self, term: &Term) {
        self.model.term_spans.insert(term.id, term.span);
        match &term.kind {
            TermKind::Var(sym) => {
                if let Some(owner) = self.model.owner_of(*sym) {
                    self.model.uses.entry(term.id).or_default().insert(owner);
                }
            }
            TermKind::Let {
                name,
                name_ty,
                rhs,
                body,
            } => {
                self.visit_term(rhs);
                if !types::is_copy(name_ty) {
                    let owner = match self.alias_target(rhs) {
                        Some(target) => target,
                        None => {
                            let origin = self.origin_for_unaliased(rhs);
                            let fresh = self.fresh_owner(origin);
                            self.model.defs.entry(term.id).or_default().insert(fresh);
                            fresh
                        }
                    };
                    self.bind(*name, owner);
                }
                self.visit_term(body);
            }
            TermKind::App { func, args } => {
                self.visit_term(func);
                let param_tys = collect_param_types(&func.ty, args.len());
                for (arg, param_ty) in args.iter().zip(&param_tys) {
                    self.visit_term(arg);
                    if types::is_unique(param_ty) {
                        if let Some(owner) = self.alias_target(arg) {
                            self.model.kills.entry(term.id).or_default().insert(owner);
                        }
                    }
                }
            }
            TermKind::Lambda(lam) => self.visit_lambda(lam),
            TermKind::Soac(op) => self.visit_soac(op),
            TermKind::ArrayExpr(ae) => self.visit_array_expr(ae),
            TermKind::Loop {
                loop_var,
                loop_var_ty,
                init,
                init_bindings,
                kind,
                body,
            } => {
                self.visit_term(init);

                // The loop carry: each iteration binds `loop_var` to the
                // running accumulator. Treat it as a fresh per-iteration
                // owner, inheriting the init's owner if init is a direct
                // alias of one. The liveness fixed-point still iterates
                // correctly.
                if !types::is_copy(loop_var_ty) {
                    let owner = match self.alias_target(init) {
                        Some(target) => target,
                        None => {
                            let origin = self.origin_for_unaliased(init);
                            let fresh = self.fresh_owner(origin);
                            self.model.defs.entry(term.id).or_default().insert(fresh);
                            fresh
                        }
                    };
                    self.bind(*loop_var, owner);
                }

                // Sub-bindings extracted from loop_var (e.g. tuple
                // destructuring). Each is `(name, ty, extraction_expr)`.
                for (name, ty, extract) in init_bindings {
                    self.visit_term(extract);
                    if !types::is_copy(ty) {
                        let owner = match self.alias_target(extract) {
                            Some(target) => target,
                            None => {
                                let origin = self.origin_for_unaliased(extract);
                                let fresh = self.fresh_owner(origin);
                                self.model.defs.entry(term.id).or_default().insert(fresh);
                                fresh
                            }
                        };
                        self.bind(*name, owner);
                    }
                }

                // Loop kind: for-array binds an iteration element; for-range
                // binds an i32; while is just a condition expression.
                match kind {
                    LoopKind::For { var, var_ty, iter } => {
                        self.visit_term(iter);
                        if !types::is_copy(var_ty) {
                            let owner = self.fresh_owner(Origin::Fresh);
                            self.bind(*var, owner);
                        }
                    }
                    LoopKind::ForRange { var, var_ty, bound } => {
                        self.visit_term(bound);
                        if !types::is_copy(var_ty) {
                            let owner = self.fresh_owner(Origin::Fresh);
                            self.bind(*var, owner);
                        }
                    }
                    LoopKind::While { cond } => {
                        self.visit_term(cond);
                    }
                }

                self.visit_term(body);
            }
            _ => {
                term.for_each_child(&mut |child| self.visit_term(child));
            }
        }
    }

    /// Bind every non-copy lambda param to a fresh per-call owner,
    /// then visit the body. Captures' carrier terms are visited so
    /// they record uses on the closure value's parent scope.
    fn visit_lambda(&mut self, lam: &Lambda) {
        for (sym, ty) in &lam.params {
            if !types::is_copy(ty) {
                let owner = self.fresh_owner(Origin::Fresh);
                self.bind(*sym, owner);
            }
        }
        for (_, _, capture_term) in &lam.captures {
            self.visit_term(capture_term);
        }
        self.visit_term(&lam.body);
    }

    /// SOACs: each input array contributes uses; each lambda's element
    /// param gets a fresh per-iteration owner.
    fn visit_soac(&mut self, op: &SoacOp) {
        match op {
            SoacOp::Map { lam, inputs } => {
                for ae in inputs {
                    self.visit_array_expr(ae);
                }
                self.visit_lambda(lam);
            }
            SoacOp::Reduce { op, ne, input, .. } => {
                self.visit_term(ne);
                self.visit_array_expr(input);
                self.visit_lambda(op);
            }
            SoacOp::Scan { op, ne, input } => {
                self.visit_term(ne);
                self.visit_array_expr(input);
                self.visit_lambda(op);
            }
            SoacOp::Filter { pred, input } => {
                self.visit_array_expr(input);
                self.visit_lambda(pred);
            }
            SoacOp::Scatter { indices, values, .. } => {
                self.visit_array_expr(indices);
                self.visit_array_expr(values);
            }
            SoacOp::ReduceByIndex {
                op,
                ne,
                indices,
                values,
                ..
            } => {
                self.visit_term(ne);
                self.visit_array_expr(indices);
                self.visit_array_expr(values);
                self.visit_lambda(op);
            }
            SoacOp::Redomap {
                op,
                reduce_op,
                ne,
                inputs,
                ..
            } => {
                self.visit_term(ne);
                for ae in inputs {
                    self.visit_array_expr(ae);
                }
                self.visit_lambda(op);
                self.visit_lambda(reduce_op);
            }
        }
    }

    fn visit_array_expr(&mut self, ae: &ArrayExpr) {
        match ae {
            ArrayExpr::Ref(t) => self.visit_term(t),
            ArrayExpr::Zip(aes) => {
                for ae in aes {
                    self.visit_array_expr(ae);
                }
            }
            ArrayExpr::Soac(op) => self.visit_soac(op),
            ArrayExpr::Generate { index_fn, .. } => self.visit_lambda(index_fn),
            ArrayExpr::Literal(terms) => {
                for t in terms {
                    self.visit_term(t);
                }
            }
            ArrayExpr::Range { start, len } => {
                self.visit_term(start);
                self.visit_term(len);
            }
            ArrayExpr::StorageBuffer { offset, len, .. } => {
                self.visit_term(offset);
                self.visit_term(len);
            }
        }
    }

    /// If a term is a direct alias of an existing tracked owner,
    /// return that owner. Two patterns count as direct aliases:
    ///
    /// - `Var(v)` where `v` is bound to an owner.
    /// - `App(intrinsic, args)` for a known aliasing intrinsic
    ///   (`_w_index`, `_w_tuple_proj`, `_w_intrinsic_array_with_inplace`):
    ///   recurse into the arg position the intrinsic aliases.
    ///   Handles nesting like `grid[i][j]` naturally.
    ///
    /// Returns `None` for fresh allocations and unrecognized compound
    /// RHSs. The caller decides what owner/origin to mint based on
    /// whether the rhs is a recognized fresh-producer.
    fn alias_target(&self, term: &Term) -> Option<OwnerId> {
        match &term.kind {
            TermKind::Var(sym) => self.model.owner_of(*sym),
            TermKind::App { func, args } => {
                let TermKind::Var(s) = &func.kind else {
                    return None;
                };
                let name = self.program.symbols.get(*s)?;
                let i = intrinsic_aliasing_arg(name)?;
                self.alias_target(args.get(i)?)
            }
            _ => None,
        }
    }

    /// The origin to assign when a Let / Loop binder produces a fresh
    /// owner (i.e., `alias_target` returned `None`). `Fresh` for
    /// recognized producers (literals, ranges, generators, fresh
    /// intrinsics, calls returning `*T`); `Borrowed` for anything we
    /// can't classify (user function returning non-unique non-copy,
    /// lambda invocation, etc.).
    fn origin_for_unaliased(&self, rhs: &Term) -> Origin {
        if types::is_unique(&rhs.ty) {
            return Origin::Fresh;
        }
        if rhs_is_fresh_producer(rhs, self.program) {
            return Origin::Fresh;
        }
        Origin::Borrowed
    }
}

/// Intrinsics whose result aliases one of their arguments. The
/// returned index is the arg position the result aliases.
fn intrinsic_aliasing_arg(name: &str) -> Option<usize> {
    match name {
        // arr[i] — view into arr
        "_w_index" => Some(0),
        // tuple_proj(t, idx) — projection into t
        "_w_tuple_proj" => Some(0),
        // in-place with returns the same buffer it consumed
        "_w_intrinsic_array_with_inplace" => Some(0),
        _ => None,
    }
}

/// Recognize forms that *definitely* produce a fresh non-copy value.
/// Used by `origin_for_unaliased` to keep promotion paths open for
/// known fresh-producers while staying conservative about unknowns.
fn rhs_is_fresh_producer(term: &Term, program: &Program) -> bool {
    match &term.kind {
        TermKind::ArrayExpr(_) => true,
        TermKind::App { func, .. } => {
            let TermKind::Var(s) = &func.kind else {
                return false;
            };
            program.symbols.get(*s).map(|name| is_fresh_producer_intrinsic(name)).unwrap_or(false)
        }
        _ => false,
    }
}

/// Compiler-internal intrinsics that build a fresh non-copy value.
/// Functional `with` allocates a new array; aliasing intrinsics like
/// `_w_index` are deliberately absent.
fn is_fresh_producer_intrinsic(name: &str) -> bool {
    matches!(
        name,
        "_w_array_lit" | "_w_vec_lit" | "_w_tuple" | "_w_intrinsic_array_with"
    )
}

/// Walk a curried function type `a1 -> a2 -> ... -> aN -> ret`, returning
/// the parameter types `[a1, a2, ..., aN]` (one per applied argument).
///
/// Stops after `expected` unwraps; if the type doesn't have enough arrows
/// (e.g. type variable that hasn't been instantiated yet), returns what was
/// found so far. Callers use the result for per-position effect classification
/// (uniqueness, etc.) and gracefully handle a short result.
fn collect_param_types(func_ty: &Type<TypeName>, expected: usize) -> Vec<Type<TypeName>> {
    let mut out = Vec::with_capacity(expected);
    let mut current = func_ty.clone();
    for _ in 0..expected {
        match current {
            Type::Constructed(TypeName::Arrow, args) if args.len() == 2 => {
                out.push(args[0].clone());
                current = args[1].clone();
            }
            _ => break,
        }
    }
    out
}

// =============================================================================
// Backward liveness with fixed-point over loops and SOAC bodies
// =============================================================================

/// Walks the TLC tree in reverse evaluation order, computing the set
/// of owners reachable from some future use at every program point.
///
/// At each term, records `live_out[term.id]` — the live-out set the
/// caller's continuation receives. Callers query this map at promotion
/// candidates (`_w_intrinsic_array_with` calls, in-place SOAC inputs)
/// to ask "is the source's owner dead-after?"
///
/// Iteration is structured-recursive: the tree's shape encodes the
/// successor relation. Loops and iterating SOACs run a fixed-point
/// over the body so back-edges propagate.
struct Liveness<'m> {
    model: &'m mut OwnershipModel,
}

type LiveSet = HashSet<OwnerId>;

impl<'m> Liveness<'m> {
    fn analyze_def(&mut self, def: &Def) {
        // The function's exit takes the body's value by value; from the
        // body's perspective, no owners need to outlive the return.
        // Uses of the returned value are recorded at the body's leaf
        // Var(s), which is enough for promotion checks within the body.
        self.analyze(&def.body, LiveSet::new());
    }

    /// Analyze a term in reverse: given the live-out set, return the
    /// live-in set. Records `live_out[term.id]` along the way.
    fn analyze(&mut self, term: &Term, live_after: LiveSet) -> LiveSet {
        self.model.live_out.insert(term.id, live_after.clone());

        match &term.kind {
            TermKind::Var(_)
            | TermKind::IntLit(_)
            | TermKind::FloatLit(_)
            | TermKind::BoolLit(_)
            | TermKind::BinOp(_)
            | TermKind::UnOp(_)
            | TermKind::Extern(_) => self.transfer(term.id, live_after),

            TermKind::Let { rhs, body, .. } => {
                let live_in_body = self.analyze(body, live_after);
                let defs_here = self.defs(term.id);
                let live_after_rhs = sub(&live_in_body, &defs_here);
                self.analyze(rhs, live_after_rhs)
            }

            TermKind::App { func, args } => {
                let mut live = self.transfer(term.id, live_after);
                for arg in args.iter().rev() {
                    live = self.analyze(arg, live);
                }
                self.analyze(func, live)
            }

            TermKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let lb_then = self.analyze(then_branch, live_after.clone());
                let lb_else = self.analyze(else_branch, live_after);
                let merged = union(&lb_then, &lb_else);
                self.analyze(cond, merged)
            }

            TermKind::Loop {
                init,
                init_bindings,
                kind,
                body,
                ..
            } => {
                let defs_here = self.defs(term.id);
                let mut live_after_body = live_after.clone();
                loop {
                    let live_in_body = self.analyze(body, live_after_body.clone());
                    let next = union(&live_after, &sub(&live_in_body, &defs_here));
                    if next == live_after_body {
                        break;
                    }
                    live_after_body = next;
                }
                let live_after_body_final = live_after_body;
                let live_in_body_final = self.analyze(body, live_after_body_final.clone());

                let live_after_kind = sub(&live_in_body_final, &defs_here);
                let live_in_kind = self.analyze_loop_kind(kind, live_after_kind);

                let mut live = live_in_kind;
                for (_name, _ty, extract) in init_bindings.iter().rev() {
                    live = self.analyze(extract, live);
                }
                self.analyze(init, live)
            }

            TermKind::Lambda(lam) => self.analyze_lambda(lam, live_after),

            TermKind::Soac(op) => self.analyze_soac(op, live_after, term.id),

            TermKind::ArrayExpr(ae) => self.analyze_array_expr(ae, live_after),

            TermKind::Force(inner) => self.analyze(inner, live_after),
        }
    }

    fn analyze_loop_kind(&mut self, kind: &LoopKind, live_after: LiveSet) -> LiveSet {
        match kind {
            LoopKind::For { iter, .. } => self.analyze(iter, live_after),
            LoopKind::ForRange { bound, .. } => self.analyze(bound, live_after),
            LoopKind::While { cond } => self.analyze(cond, live_after),
        }
    }

    /// A lambda value carries its captures: any owner referenced in the
    /// body that isn't bound by the body itself is live at the lambda
    /// creation site. We compute this by analyzing the body with
    /// `live_after = ∅`; the result is the body's live_in, which (after
    /// removing locally-bound owners — in practice fresh per-iteration
    /// params recorded under `defs`) gives the captures. The lambda's
    /// own term contributes those captures to its parent's live_in.
    fn analyze_lambda(&mut self, lam: &Lambda, live_after: LiveSet) -> LiveSet {
        let live_in_body = self.analyze(&lam.body, LiveSet::new());
        // The body's live_in includes any owner the body references.
        // Locally-bound owners (lambda params) are tracked under their
        // own ids; they don't appear in live_in_body if no Var leaf
        // outside the body's scope references them — and Var leaves
        // *inside* the body use the locally-bound symbols. Treat the
        // result as captures-needed.
        union(&live_after, &live_in_body)
    }

    fn analyze_soac(&mut self, op: &SoacOp, live_after: LiveSet, _soac_id: TermId) -> LiveSet {
        match op {
            SoacOp::Map { lam, inputs } => {
                let live_in_lam = self.lambda_body_fixed_point(lam);
                let mut live = union(&live_after, &live_in_lam);
                for ae in inputs.iter().rev() {
                    live = self.analyze_array_expr(ae, live);
                }
                live
            }
            SoacOp::Reduce { op, ne, input, .. } => {
                let live_in_op = self.lambda_body_fixed_point(op);
                let after_input = union(&live_after, &live_in_op);
                let after_ne = self.analyze_array_expr(input, after_input);
                self.analyze(ne, after_ne)
            }
            SoacOp::Scan { op, ne, input } => {
                let live_in_op = self.lambda_body_fixed_point(op);
                let after_input = union(&live_after, &live_in_op);
                let after_ne = self.analyze_array_expr(input, after_input);
                self.analyze(ne, after_ne)
            }
            SoacOp::Filter { pred, input } => {
                let live_in_pred = self.lambda_body_fixed_point(pred);
                let after_input = union(&live_after, &live_in_pred);
                self.analyze_array_expr(input, after_input)
            }
            SoacOp::Scatter { indices, values, .. } => {
                let after_values = self.analyze_array_expr(values, live_after);
                self.analyze_array_expr(indices, after_values)
            }
            SoacOp::ReduceByIndex {
                op,
                ne,
                indices,
                values,
                ..
            } => {
                let live_in_op = self.lambda_body_fixed_point(op);
                let after_values = union(&live_after, &live_in_op);
                let after_indices = self.analyze_array_expr(values, after_values);
                let after_ne = self.analyze_array_expr(indices, after_indices);
                self.analyze(ne, after_ne)
            }
            SoacOp::Redomap {
                op,
                reduce_op,
                ne,
                inputs,
                ..
            } => {
                let live_in_op = self.lambda_body_fixed_point(op);
                let live_in_reduce = self.lambda_body_fixed_point(reduce_op);
                let mut live = union(&union(&live_after, &live_in_op), &live_in_reduce);
                for ae in inputs.iter().rev() {
                    live = self.analyze_array_expr(ae, live);
                }
                self.analyze(ne, live)
            }
        }
    }

    /// SOAC bodies iterate. Run a fixed-point over the body so any
    /// owner used in iteration N+1 stays live across iteration N.
    fn lambda_body_fixed_point(&mut self, lam: &Lambda) -> LiveSet {
        let mut live_after_body = LiveSet::new();
        loop {
            let live_in_body = self.analyze(&lam.body, live_after_body.clone());
            if live_in_body == live_after_body {
                break;
            }
            live_after_body = live_in_body;
        }
        // Final pass to lock in the recorded live_outs.
        self.analyze(&lam.body, live_after_body.clone())
    }

    fn analyze_array_expr(&mut self, ae: &ArrayExpr, live_after: LiveSet) -> LiveSet {
        match ae {
            ArrayExpr::Ref(t) => self.analyze(t, live_after),
            ArrayExpr::Zip(aes) => {
                let mut live = live_after;
                for ae in aes.iter().rev() {
                    live = self.analyze_array_expr(ae, live);
                }
                live
            }
            ArrayExpr::Soac(op) => self.analyze_soac(op, live_after, TermId(u32::MAX)),
            ArrayExpr::Generate { index_fn, .. } => self.analyze_lambda(index_fn, live_after),
            ArrayExpr::Literal(terms) => {
                let mut live = live_after;
                for t in terms.iter().rev() {
                    live = self.analyze(t, live);
                }
                live
            }
            ArrayExpr::Range { start, len } => {
                let after_start = self.analyze(len, live_after);
                self.analyze(start, after_start)
            }
            ArrayExpr::StorageBuffer { offset, len, .. } => {
                let after_offset = self.analyze(len, live_after);
                self.analyze(offset, after_offset)
            }
        }
    }

    /// Standard transfer: `live_in = (live_out − kills) ∪ uses`.
    fn transfer(&self, id: TermId, live_after: LiveSet) -> LiveSet {
        let kills = self.model.kills.get(&id);
        let uses = self.model.uses.get(&id);
        let mut live = live_after;
        if let Some(k) = kills {
            for owner in k {
                live.remove(owner);
            }
        }
        if let Some(u) = uses {
            for owner in u {
                live.insert(*owner);
            }
        }
        live
    }

    fn defs(&self, id: TermId) -> LiveSet {
        self.model.defs.get(&id).cloned().unwrap_or_default()
    }
}

fn union(a: &LiveSet, b: &LiveSet) -> LiveSet {
    a.union(b).copied().collect()
}

fn sub(a: &LiveSet, b: &LiveSet) -> LiveSet {
    a.difference(b).copied().collect()
}

// =============================================================================
// TLC-level promotion of array_with → array_with_inplace
// =============================================================================

/// Walks every `_w_intrinsic_array_with` call in the program and
/// rewrites the func to `_w_intrinsic_array_with_inplace` when the
/// source array's owner is mutable (Origin::Fresh / UniqueParam /
/// Entry) and absent from `live_out` at the call site.
///
/// Returns an error if the analysis detects a use-after-move: a
/// `*T` consumption at a program point where the consumed owner is
/// still in `live_out`.
pub fn promote_inplace(mut program: Program) -> crate::error::Result<Program> {
    let model = analyze(&program);
    if let Some(err) = check_use_after_move(&program, &model) {
        return Err(err);
    }
    let functional_sym = match program.def_syms.get(crate::intrinsics::INTRINSIC_ARRAY_WITH).copied() {
        Some(s) => s,
        None => return Ok(program),
    };
    let inplace_sym = match program.def_syms.get(crate::intrinsics::INTRINSIC_ARRAY_WITH_INPLACE).copied() {
        Some(s) => s,
        None => {
            let s = program.symbols.alloc(crate::intrinsics::INTRINSIC_ARRAY_WITH_INPLACE.to_string());
            program.def_syms.insert(crate::intrinsics::INTRINSIC_ARRAY_WITH_INPLACE.to_string(), s);
            s
        }
    };

    let mut rewriter = Rewriter {
        model: &model,
        functional_sym,
        inplace_sym,
    };
    let new_defs: Vec<Def> = program
        .defs
        .into_iter()
        .map(|def| Def {
            body: rewriter.rewrite(def.body),
            ..def
        })
        .collect();
    Ok(Program {
        defs: new_defs,
        ..program
    })
}

/// Run the ownership analysis and report a use-after-move error if
/// any owner is consumed at a program point where a successor still
/// reads it. Used by the `wyn check` subcommand and indirectly by
/// `promote_inplace`.
pub fn check(program: &Program) -> crate::error::Result<()> {
    let model = analyze(program);
    if let Some(err) = check_use_after_move(program, &model) {
        return Err(err);
    }
    Ok(())
}

/// Walk the model for any owner that is killed at a term while still
/// being in that term's `live_out` — i.e., a successor still reads it.
/// Returns the first such violation as a CompilerError; later
/// violations are not reported in this pass (one diagnostic per
/// compile is consistent with how the rest of the pipeline reports).
fn check_use_after_move(program: &Program, model: &OwnershipModel) -> Option<crate::error::CompilerError> {
    use crate::error::CompilerError;
    let mut violations: Vec<(TermId, OwnerId)> = model
        .kills
        .iter()
        .flat_map(|(id, killed)| {
            let live = model.live_out.get(id);
            killed.iter().filter(move |o| live.map_or(false, |s| s.contains(o))).map(move |o| (*id, *o))
        })
        .collect();
    violations.sort_by_key(|(id, _)| id.0);
    let (term_id, owner) = violations.into_iter().next()?;
    let span = model.term_spans.get(&term_id).copied();
    let var_name = model
        .owner_to_var
        .get(&owner)
        .and_then(|s| program.symbols.get(*s).cloned())
        .unwrap_or_else(|| "<value>".to_string());
    Some(CompilerError::AliasError(
        format!("use of moved value `{}`", var_name),
        span,
    ))
}

struct Rewriter<'m> {
    model: &'m OwnershipModel,
    functional_sym: SymbolId,
    inplace_sym: SymbolId,
}

impl<'m> Rewriter<'m> {
    fn rewrite(&mut self, term: Term) -> Term {
        if let TermKind::App { func, args } = &term.kind {
            let calls_functional = matches!(&func.kind, TermKind::Var(s) if *s == self.functional_sym);
            if calls_functional && args.len() == 3 && self.is_promotable(term.id, &args[0]) {
                let TermKind::App { func, args } = term.kind else {
                    unreachable!()
                };
                let new_func = Term {
                    kind: TermKind::Var(self.inplace_sym),
                    ..*func
                };
                let new_args: Vec<Term> = args.into_iter().map(|a| self.rewrite(a)).collect();
                return Term {
                    id: term.id,
                    ty: term.ty,
                    span: term.span,
                    kind: TermKind::App {
                        func: Box::new(new_func),
                        args: new_args,
                    },
                };
            }
        }
        term.map_children(&mut |child| self.rewrite(child))
    }

    fn is_promotable(&self, call_id: TermId, source_arg: &Term) -> bool {
        let TermKind::Var(sym) = &source_arg.kind else {
            return false;
        };
        let Some(owner) = self.model.owner_of(*sym) else {
            return false;
        };
        let Some(origin) = self.model.origin(owner) else {
            return false;
        };
        if !origin.is_mutable() {
            return false;
        }
        // No live_out recorded for this call ⇒ analyze didn't reach
        // it (e.g. dead-code def we never traversed). Be conservative:
        // don't promote.
        let live = match self.model.live_out.get(&call_id) {
            Some(l) => l,
            None => return false,
        };
        !live.contains(&owner)
    }
}

#[cfg(test)]
#[path = "ownership_tests.rs"]
mod ownership_tests;
