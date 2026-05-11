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
    /// Per-iteration element view of a mutable-input SOAC body.
    /// The element is loaded into a local each iteration; mutations
    /// stay local. Mutable like `Fresh` for promotion purposes, but
    /// distinguished so future code can tell "I own an array" from
    /// "I have a per-iteration view into a mutable array I'm
    /// iterating over."
    BorrowedMutableElement,
}

impl Origin {
    pub fn is_mutable(self) -> bool {
        matches!(
            self,
            Origin::Fresh | Origin::UniqueParam | Origin::Entry | Origin::BorrowedMutableElement
        )
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
    /// Diagnostics detected during the build walk that don't fall
    /// out of the standard `kills ∩ live_out` check. Currently
    /// intra-call duplicate consumption (`f(arr, arr)` to two `*T`
    /// params). Stored as (message, span) so we don't need
    /// `CompilerError: Clone`; the check function reconstructs.
    pub build_errors: Vec<(String, Option<Span>)>,
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
            TermKind::Var(crate::tlc::VarRef::Symbol(sym)) => {
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
                let mut killed_this_call: HashSet<OwnerId> = HashSet::new();
                for (arg, param_ty) in args.iter().zip(&param_tys) {
                    self.visit_term(arg);
                    if types::is_unique(param_ty) {
                        if let Some(owner) = self.alias_target(arg) {
                            // A second `*T` arg resolving to the same
                            // owner consumes a store the prior arg
                            // already moved — use-after-move within
                            // this call. Record at the offending
                            // arg's term so the diagnostic span
                            // points at the second occurrence.
                            if !killed_this_call.insert(owner) {
                                let var_name = self
                                    .model
                                    .owner_to_var
                                    .get(&owner)
                                    .and_then(|s| self.program.symbols.get(*s).cloned())
                                    .unwrap_or_else(|| "<value>".to_string());
                                let span = self.model.term_spans.get(&arg.id).copied();
                                self.model
                                    .build_errors
                                    .push((format!("use of moved value `{}`", var_name), span));
                            }
                            self.model.kills.entry(term.id).or_default().insert(owner);
                        }
                    }
                }
            }
            TermKind::Lambda(lam) => self.visit_lambda(lam),
            TermKind::Soac(op) => self.visit_soac(op, term.id),
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

    /// Free-standing lambda binding: without a SOAC context to tie
    /// param mutability to a specific input, every non-copy param
    /// defaults to `Borrowed`. Sound under-approximation — a
    /// free-standing lambda might be called with caller-owned data,
    /// so the body must not mutate its params in place.
    fn visit_lambda(&mut self, lam: &Lambda) {
        for (sym, ty) in &lam.params {
            if !types::is_copy(ty) {
                let owner = self.fresh_owner(Origin::Borrowed);
                self.bind(*sym, owner);
            }
        }
        self.visit_term(&lam.body);
    }

    /// Visit a SoacBody: bind its captures, then walk the body.
    /// Used by the SOAC arms (after binding params with input-derived
    /// origins).
    ///
    /// Captures pre-defunctionalization are empty — the body
    /// references outer SymbolIds directly and the existing scope
    /// resolves them. Post-defunc, each capture has its own
    /// `(capture_sym, ty, capture_term)`: the body is rewritten to
    /// reference `capture_sym`, which we must bind here so its
    /// reads/kills connect back to the outer owner. Without this,
    /// a body that consumes a captured store would record no kill
    /// (the capture-local sym is not in `var_to_owner`) and the
    /// analysis would silently approve a use-after-move.
    fn visit_soac_body(&mut self, sb: &super::SoacBody) {
        for (capture_sym, capture_ty, capture_term) in &sb.captures {
            self.visit_term(capture_term);
            if !types::is_copy(capture_ty) {
                let owner = match self.alias_target(capture_term) {
                    Some(target) => target,
                    None => self.fresh_owner(self.origin_for_unaliased(capture_term)),
                };
                self.bind(*capture_sym, owner);
            }
        }
        self.visit_term(&sb.lam.body);
    }

    /// SOACs: each input array contributes uses; element params
    /// inherit mutability from their matched input (Map/Redomap), or
    /// from the single input (Reduce/Scan/Filter/ReduceByIndex).
    /// Accumulator params (Reduce/Scan/Redomap) are the body's
    /// per-iteration output — Fresh.
    ///
    /// Each per-iteration owner introduced here is also recorded
    /// under `defs[soac_id]` so the liveness fixed-point can
    /// subtract them from the loop-back set: a body that consumes
    /// its own element/accumulator param consumes a *fresh* runtime
    /// value each iteration, not a value carried across.
    fn visit_soac(&mut self, op: &SoacOp, soac_id: TermId) {
        match op {
            SoacOp::Map { lam, inputs, .. } => {
                for ae in inputs {
                    self.visit_array_expr(ae);
                }
                for ((sym, ty), input) in lam.lam.params.iter().zip(inputs.iter()) {
                    if !types::is_copy(ty) {
                        let origin = self.element_origin_from_input(input);
                        let owner = self.fresh_owner(origin);
                        self.bind(*sym, owner);
                        self.record_per_call_def(soac_id, owner);
                    }
                }
                self.visit_soac_body(lam);
            }
            SoacOp::Reduce { op, ne, input, .. } => {
                self.visit_term(ne);
                self.visit_array_expr(input);
                self.bind_reducer_params(&op.lam, input, soac_id);
                self.visit_soac_body(op);
            }
            SoacOp::Scan { op, ne, input } => {
                self.visit_term(ne);
                self.visit_array_expr(input);
                self.bind_reducer_params(&op.lam, input, soac_id);
                self.visit_soac_body(op);
            }
            SoacOp::Filter { pred, input } => {
                self.visit_array_expr(input);
                if let Some((sym, ty)) = pred.lam.params.first() {
                    if !types::is_copy(ty) {
                        let origin = self.element_origin_from_input(input);
                        let owner = self.fresh_owner(origin);
                        self.bind(*sym, owner);
                        self.record_per_call_def(soac_id, owner);
                    }
                }
                self.visit_soac_body(pred);
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
                self.bind_reducer_params(&op.lam, values, soac_id);
                self.visit_soac_body(op);
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
                // op has shape (acc, x1, ..., xN). Bind acc as Fresh,
                // each xi from inputs[i].
                if let Some(((acc_sym, acc_ty), elem_params)) = op.lam.params.split_first() {
                    if !types::is_copy(acc_ty) {
                        let owner = self.fresh_owner(Origin::Fresh);
                        self.bind(*acc_sym, owner);
                        self.record_per_call_def(soac_id, owner);
                    }
                    for ((sym, ty), input) in elem_params.iter().zip(inputs.iter()) {
                        if !types::is_copy(ty) {
                            let origin = self.element_origin_from_input(input);
                            let owner = self.fresh_owner(origin);
                            self.bind(*sym, owner);
                            self.record_per_call_def(soac_id, owner);
                        }
                    }
                }
                self.visit_soac_body(op);
                // reduce_op is the parallel-phase combiner: (acc, acc) → acc.
                // Both params are accumulator-typed; Fresh per call.
                for (sym, ty) in &reduce_op.lam.params {
                    if !types::is_copy(ty) {
                        let owner = self.fresh_owner(Origin::Fresh);
                        self.bind(*sym, owner);
                        self.record_per_call_def(soac_id, owner);
                    }
                }
                self.visit_soac_body(reduce_op);
            }
        }
    }

    fn record_per_call_def(&mut self, soac_id: TermId, owner: OwnerId) {
        self.model.defs.entry(soac_id).or_default().insert(owner);
    }

    /// Bind a reducer lambda's two params: (acc, elem). `acc` is the
    /// body's per-iteration output (Fresh). `elem` inherits
    /// mutability from the input. Both are per-call, recorded under
    /// `defs[soac_id]` so the liveness fixed-point doesn't carry
    /// them across iterations.
    fn bind_reducer_params(&mut self, op: &Lambda, input: &ArrayExpr, soac_id: TermId) {
        let mut params = op.params.iter();
        if let Some((sym, ty)) = params.next() {
            if !types::is_copy(ty) {
                let owner = self.fresh_owner(Origin::Fresh);
                self.bind(*sym, owner);
                self.record_per_call_def(soac_id, owner);
            }
        }
        if let Some((sym, ty)) = params.next() {
            if !types::is_copy(ty) {
                let origin = self.element_origin_from_input(input);
                let owner = self.fresh_owner(origin);
                self.bind(*sym, owner);
                self.record_per_call_def(soac_id, owner);
            }
        }
    }

    /// Decide the origin to give a SOAC element param given the
    /// input ArrayExpr it iterates over. Mutable inputs
    /// (Fresh-allocated arrays, `*T`-typed array refs) yield mutable
    /// element views; everything else borrows.
    fn element_origin_from_input(&self, ae: &ArrayExpr) -> Origin {
        match ae {
            ArrayExpr::Ref(t) => {
                if let Some(owner) = self.alias_target(t) {
                    if self.model.origin(owner).map(|o| o.is_mutable()).unwrap_or(false) {
                        return Origin::BorrowedMutableElement;
                    }
                    return Origin::Borrowed;
                }
                // No tracked owner — fall back to the term's static type.
                if types::is_unique(&t.ty) { Origin::BorrowedMutableElement } else { Origin::Borrowed }
            }
            // Fresh-producer ArrayExprs: literal/generate/range/soac
            // synthesize a new array, so element views are mutable.
            ArrayExpr::Literal(_)
            | ArrayExpr::Generate { .. }
            | ArrayExpr::Range { .. }
            | ArrayExpr::Soac(_) => Origin::BorrowedMutableElement,
            // Storage-buffer-backed views: conservative borrow.
            ArrayExpr::StorageBuffer { .. } => Origin::Borrowed,
            // Zip is a phase-scoped sentinel that should be absorbed
            // by `tlc::soa::run` before we get here. If one survives,
            // be conservative.
            ArrayExpr::Zip(_) => Origin::Borrowed,
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
            // Nested SOAC inside an ArrayExpr: there is no
            // dedicated TermId for it (the SOAC isn't wrapped in a
            // Term here), so per-call defs land under a sentinel.
            // `lambda_body_fixed_point` looks up by the call site's
            // own id, so this sentinel collision is harmless.
            ArrayExpr::Soac(op) => self.visit_soac(op, TermId(u32::MAX)),
            ArrayExpr::Generate { index_fn, .. } => self.visit_soac_body(index_fn),
            ArrayExpr::Literal(terms) => {
                for t in terms {
                    self.visit_term(t);
                }
            }
            ArrayExpr::Range { start, len, step } => {
                self.visit_term(start);
                self.visit_term(len);
                if let Some(s) = step {
                    self.visit_term(s);
                }
            }
            ArrayExpr::StorageBuffer { offset, len, .. } => {
                self.visit_term(offset);
                self.visit_term(len);
            }
        }
    }

    fn alias_target(&self, term: &Term) -> Option<OwnerId> {
        alias_target_of(term, &self.model, self.program)
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

/// If `term` is a direct alias of an existing tracked owner, return
/// that owner. Two patterns count as direct aliases:
///
/// - `Var(v)` where `v` is bound to an owner.
/// - `App(intrinsic, args)` for a known aliasing intrinsic
///   (`_w_index`, `_w_tuple_proj`, `_w_intrinsic_array_with_inplace`):
///   recurse into the arg position the intrinsic aliases. Handles
///   nesting like `grid[i][j]` naturally.
///
/// Returns `None` for fresh allocations and unrecognized compound
/// terms. Used during build (to thread aliasing through let/loop
/// binders) and during promotion (to identify the underlying owner
/// of an `_w_intrinsic_array_with` source that's a projection or
/// indexing expression rather than a plain `Var`).
pub(super) fn alias_target_of(term: &Term, model: &OwnershipModel, program: &Program) -> Option<OwnerId> {
    match &term.kind {
        TermKind::Var(crate::tlc::VarRef::Symbol(sym)) => model.owner_of(*sym),
        // `Index` and `TupleProj` are projection variants whose result
        // aliases the base.
        TermKind::Index { array, .. } => alias_target_of(array, model, program),
        TermKind::TupleProj { tuple, .. } => alias_target_of(tuple, model, program),
        // In-place `array_with` returns the buffer it consumed.
        TermKind::App { func, args } => {
            if crate::tlc::var_term_builtin_id(func, &program.symbols)
                == Some(crate::builtins::catalog().known().array_with_in_place)
            {
                return alias_target_of(args.first()?, model, program);
            }
            None
        }
        _ => None,
    }
}

/// Recognize forms that *definitely* produce a fresh non-copy value.
/// Used by `origin_for_unaliased` to keep promotion paths open for
/// known fresh-producers while staying conservative about unknowns.
///
/// Functional `array_with` (catalog `BuiltinId`) allocates a new array;
/// the in-place variant is deliberately absent — that one aliases its
/// input. Structural literal/tuple constructors (`Tuple`, `VecLit`,
/// `ArrayExpr`) also allocate fresh.
fn rhs_is_fresh_producer(term: &Term, program: &Program) -> bool {
    match &term.kind {
        TermKind::ArrayExpr(_) | TermKind::Tuple(_) | TermKind::VecLit(_) => true,
        TermKind::App { func, .. } => {
            crate::tlc::var_term_builtin_id(func, &program.symbols)
                == Some(crate::builtins::catalog().known().array_with)
        }
        _ => false,
    }
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
        //
        // Walk the Lambda's inner body directly — bypassing
        // `analyze_lambda` and its fixed-point. A top-level Def is
        // not a stored lambda value: each call gets fresh args, so
        // captures-across-invocations doesn't apply at this level.
        // Nested lambdas (let-bound, returned, etc.) still go through
        // `analyze_lambda` and pick up the fixed-point.
        let body = match &def.body.kind {
            TermKind::Lambda(lam) => &*lam.body,
            _ => &def.body,
        };
        self.analyze(body, LiveSet::new());
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

            TermKind::Tuple(parts) | TermKind::VecLit(parts) => {
                let mut live = self.transfer(term.id, live_after);
                for p in parts.iter().rev() {
                    live = self.analyze(p, live);
                }
                live
            }
            TermKind::TupleProj { tuple, .. } => {
                let live = self.transfer(term.id, live_after);
                self.analyze(tuple, live)
            }
            TermKind::Index { array, index } => {
                let live = self.transfer(term.id, live_after);
                let live = self.analyze(index, live);
                self.analyze(array, live)
            }
        }
    }

    fn analyze_loop_kind(&mut self, kind: &LoopKind, live_after: LiveSet) -> LiveSet {
        match kind {
            LoopKind::For { iter, .. } => self.analyze(iter, live_after),
            LoopKind::ForRange { bound, .. } => self.analyze(bound, live_after),
            LoopKind::While { cond } => self.analyze(cond, live_after),
        }
    }

    /// A lambda value can be invoked any number of times after
    /// creation, so the body must be analyzed as if it could iterate.
    /// Reuse the SOAC-body fixed-point: a body that kills its own
    /// referenced state across hypothetical re-invocations is detected
    /// because the kill conflicts with the still-live owner.
    fn analyze_lambda(&mut self, lam: &Lambda, live_after: LiveSet) -> LiveSet {
        // Free-standing lambdas have no per-iteration locals to
        // exclude — every reference inside the body must remain
        // live across hypothetical re-invocations.
        let no_per_call_defs = LiveSet::new();
        let live_in_body = self.lambda_body_fixed_point(lam, &no_per_call_defs);
        union(&live_after, &live_in_body)
    }

    /// SOAC-body variant of `analyze_lambda`: same fixed-point as
    /// above, but also threads liveness through capture terms.
    ///
    /// Capture terms are program points themselves — they may
    /// contain reads or kills that need to land at the lambda
    /// creation site. We thread liveness through them in reverse
    /// (last-evaluated first under backward dataflow), starting
    /// from the post-body live set so any uses inside a capture
    /// term flow back to the parent's live_in.
    fn analyze_soac_body(&mut self, sb: &super::SoacBody, live_after: LiveSet) -> LiveSet {
        let no_per_call_defs = LiveSet::new();
        let live_in_body = self.lambda_body_fixed_point(&sb.lam, &no_per_call_defs);
        let mut live = union(&live_after, &live_in_body);
        for (_, _, capture_term) in sb.captures.iter().rev() {
            live = self.analyze(capture_term, live);
        }
        live
    }

    fn analyze_soac(&mut self, op: &SoacOp, live_after: LiveSet, soac_id: TermId) -> LiveSet {
        let per_call_defs = self.model.defs.get(&soac_id).cloned().unwrap_or_default();
        match op {
            SoacOp::Map { lam, inputs, .. } => {
                let mut live = self.soac_envelope_fixed_point(lam, &per_call_defs, live_after);
                for ae in inputs.iter().rev() {
                    live = self.analyze_array_expr(ae, live);
                }
                live
            }
            SoacOp::Reduce { op, ne, input, .. } => {
                let after_op = self.soac_envelope_fixed_point(op, &per_call_defs, live_after);
                let after_input = self.analyze_array_expr(input, after_op);
                self.analyze(ne, after_input)
            }
            SoacOp::Scan { op, ne, input } => {
                let after_op = self.soac_envelope_fixed_point(op, &per_call_defs, live_after);
                let after_input = self.analyze_array_expr(input, after_op);
                self.analyze(ne, after_input)
            }
            SoacOp::Filter { pred, input } => {
                let after_pred = self.soac_envelope_fixed_point(pred, &per_call_defs, live_after);
                self.analyze_array_expr(input, after_pred)
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
                let after_op = self.soac_envelope_fixed_point(op, &per_call_defs, live_after);
                let after_values = self.analyze_array_expr(values, after_op);
                let after_indices = self.analyze_array_expr(indices, after_values);
                self.analyze(ne, after_indices)
            }
            SoacOp::Redomap {
                op,
                reduce_op,
                ne,
                inputs,
                ..
            } => {
                let after_op = self.soac_envelope_fixed_point(op, &per_call_defs, live_after);
                let after_reduce = self.soac_envelope_fixed_point(reduce_op, &per_call_defs, after_op);
                let mut live = after_reduce;
                for ae in inputs.iter().rev() {
                    live = self.analyze_array_expr(ae, live);
                }
                self.analyze(ne, live)
            }
        }
    }

    /// Run the lambda-body fixed-point honoring per-call defs, then
    /// thread liveness through the captures (in reverse, mirroring
    /// backward dataflow). Captures are program points whose own
    /// reads/kills must reach the parent's live_in; without this,
    /// `live_out[capture_term.id]` would never be set.
    fn soac_envelope_fixed_point(
        &mut self,
        sb: &super::SoacBody,
        per_call_defs: &LiveSet,
        live_after: LiveSet,
    ) -> LiveSet {
        let live_in_body = self.lambda_body_fixed_point(&sb.lam, per_call_defs);
        let mut live = union(&live_after, &live_in_body);
        for (_, _, capture_term) in sb.captures.iter().rev() {
            live = self.analyze(capture_term, live);
        }
        live
    }

    /// SOAC / lambda bodies iterate (or may be invoked multiple
    /// times). Run a fixed-point over the body so any owner used in
    /// iteration N+1 stays live across iteration N — *except*
    /// `per_call_defs`, which are owners freshly produced each
    /// invocation (SOAC element/accumulator params, loop_var). For
    /// free-standing lambdas with no per-call locals to subtract,
    /// pass an empty set.
    fn lambda_body_fixed_point(&mut self, lam: &Lambda, per_call_defs: &LiveSet) -> LiveSet {
        let mut live_after_body = LiveSet::new();
        loop {
            let live_in_body = self.analyze(&lam.body, live_after_body.clone());
            let next = sub(&live_in_body, per_call_defs);
            if next == live_after_body {
                break;
            }
            live_after_body = next;
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
            ArrayExpr::Generate { index_fn, .. } => self.analyze_soac_body(index_fn, live_after),
            ArrayExpr::Literal(terms) => {
                let mut live = live_after;
                for t in terms.iter().rev() {
                    live = self.analyze(t, live);
                }
                live
            }
            ArrayExpr::Range { start, len, step } => {
                let after = if let Some(s) = step { self.analyze(s, live_after) } else { live_after };
                let after_start = self.analyze(len, after);
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
pub fn apply_ownership(mut program: Program) -> crate::error::Result<Program> {
    let model = analyze(&program);
    if let Some(err) = check_use_after_move(&program, &model) {
        return Err(err);
    }
    let consuming_soacs: HashSet<TermId> = eligible_consuming_soacs(&program, &model).into_iter().collect();

    // Promotion of `array_with` → `array_with_inplace` is keyed by the
    // catalog (BuiltinId for the in-place form is looked up at the
    // rewrite site), not by symbol-table identity. We always run the
    // rewriter — even with no consuming SOACs, an `array_with` whose
    // source is a unique single-use binding is still promotable.

    let defs_in = std::mem::take(&mut program.defs);
    let mut rewriter = Rewriter {
        model: &model,
        program: &program,
        consuming_soacs: &consuming_soacs,
    };
    let new_defs: Vec<Def> = defs_in
        .into_iter()
        .map(|def| Def {
            body: rewriter.rewrite(def.body),
            ..def
        })
        .collect();
    drop(rewriter);
    program.defs = new_defs;
    Ok(program)
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
    if let Some((msg, span)) = model.build_errors.first() {
        return Some(CompilerError::AliasError(msg.clone(), *span));
    }
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
    program: &'m Program,
    consuming_soacs: &'m HashSet<TermId>,
}

impl<'m> Rewriter<'m> {
    fn rewrite(&mut self, term: Term) -> Term {
        // array_with → array_with_inplace: rewrite the App's `func`
        // before descending, since the rewrite swaps the function var.
        if let TermKind::App { func, args } = &term.kind {
            let known = crate::builtins::catalog().known();
            let calls_functional =
                crate::tlc::var_term_builtin_id(func, &self.program.symbols) == Some(known.array_with);
            if calls_functional && args.len() == 3 && self.is_promotable(term.id, &args[0]) {
                let inplace_id = known.array_with_in_place;
                let TermKind::App { func, args } = term.kind else {
                    unreachable!()
                };
                let new_func = Term {
                    kind: TermKind::Var(crate::tlc::VarRef::Builtin {
                        id: inplace_id,
                        overload_idx: 0,
                    }),
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
        // Recurse children first; any consuming-Map mark is a leaf
        // mutation on the SOAC node itself.
        let id = term.id;
        let mut rewritten = term.map_children(&mut |child| self.rewrite(child));
        if self.consuming_soacs.contains(&id) {
            if let TermKind::Soac(SoacOp::Map { consumes_input, .. }) = &mut rewritten.kind {
                *consumes_input = true;
            }
        }
        rewritten
    }

    fn is_promotable(&self, call_id: TermId, source_arg: &Term) -> bool {
        let Some(owner) = alias_target_of(source_arg, self.model, self.program) else {
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

// =============================================================================
// Consuming-SOAC eligibility (input-side DPS)
// =============================================================================

/// Return the term ids of `Map` SOACs that are eligible for input-side
/// destination-passing — i.e. the Map could mutate its input buffer in
/// place rather than allocating a fresh output.
///
/// A Map qualifies only when *all* of:
///
/// 1. The input is a single `ArrayExpr::Ref(Var(sym))` whose owner is
///    mutable and absent from `live_out` at the SOAC's term.
/// 2. The lambda body's return type matches the lambda's element-param
///    type (pointwise: same shape in, same shape out).
/// 3. The body does not read the input owner outside of the element
///    parameter — no captured stencil reads. `map(|x| x + a[i-1], a)`
///    is rejected because in-place mutation at index `i` would change
///    later iterations' reads.
/// 4. The Map is not in tail position of an `EntryPoint` def with
///    compute outputs. The output-side rewrite handles those, and
///    in-place input mutation would clobber the runtime contract on
///    the output buffer. Sound under-approximation; the overlap case
///    is a separate rewrite.
///
/// Pure analysis. Does not mutate the program. The caller decides
/// whether to act on the result.
pub fn eligible_consuming_soacs(program: &Program, model: &OwnershipModel) -> Vec<TermId> {
    let entry_output_soacs = collect_entry_output_soac_ids(program);
    let mut out = Vec::new();
    for def in &program.defs {
        walk_for_eligible_maps(&def.body, model, program, &entry_output_soacs, &mut out);
    }
    out
}

fn collect_entry_output_soac_ids(program: &Program) -> HashSet<TermId> {
    let mut out = HashSet::new();
    for def in &program.defs {
        let entry = match &def.meta {
            DefMeta::EntryPoint(e) => e,
            _ => continue,
        };
        // Compute entries with bound storage outputs are the targets
        // of the output-side rewrite. Drill through the def body to
        // any tail-position SOAC and collect its term id.
        if entry.entry_type.is_compute() && !entry.outputs.is_empty() {
            let body = match &def.body.kind {
                TermKind::Lambda(lam) => &*lam.body,
                _ => &def.body,
            };
            collect_tail_soac_ids(body, &mut out);
        }
    }
    out
}

fn collect_tail_soac_ids(term: &Term, out: &mut HashSet<TermId>) {
    match &term.kind {
        TermKind::Soac(_) => {
            out.insert(term.id);
        }
        TermKind::Let { body, .. } => collect_tail_soac_ids(body, out),
        TermKind::Force(inner) => collect_tail_soac_ids(inner, out),
        TermKind::If {
            then_branch,
            else_branch,
            ..
        } => {
            collect_tail_soac_ids(then_branch, out);
            collect_tail_soac_ids(else_branch, out);
        }
        _ => {}
    }
}

fn walk_for_eligible_maps(
    term: &Term,
    model: &OwnershipModel,
    program: &Program,
    entry_output_soacs: &HashSet<TermId>,
    out: &mut Vec<TermId>,
) {
    if let TermKind::Soac(SoacOp::Map { lam, inputs, .. }) = &term.kind {
        if map_is_eligible(term.id, &lam.lam, inputs, model, entry_output_soacs) {
            out.push(term.id);
        }
    }
    term.for_each_child(&mut |child| {
        walk_for_eligible_maps(child, model, program, entry_output_soacs, out)
    });
}

fn map_is_eligible(
    soac_id: TermId,
    lam: &Lambda,
    inputs: &[ArrayExpr],
    model: &OwnershipModel,
    entry_output_soacs: &HashSet<TermId>,
) -> bool {
    // 4 — output-bound check first: cheap.
    if entry_output_soacs.contains(&soac_id) {
        return false;
    }
    // 1 — single Var input, owner mutable, dead-after.
    if inputs.len() != 1 {
        return false;
    }
    let input_term = match &inputs[0] {
        ArrayExpr::Ref(t) => &**t,
        _ => return false,
    };
    let input_sym = match &input_term.kind {
        TermKind::Var(crate::tlc::VarRef::Symbol(s)) => *s,
        _ => return false,
    };
    let owner = match model.owner_of(input_sym) {
        Some(o) => o,
        None => return false,
    };
    let origin = match model.origin(owner) {
        Some(o) => o,
        None => return false,
    };
    if !origin.is_mutable() {
        return false;
    }
    let live_out = match model.live_out.get(&soac_id) {
        Some(l) => l,
        None => return false,
    };
    if live_out.contains(&owner) {
        return false;
    }
    // 2 — body type matches element param.
    if lam.params.len() != 1 {
        return false;
    }
    if lam.params[0].1 != lam.ret_ty {
        return false;
    }
    // 3 — pointwise: body does not reference the input symbol
    //     outside the element-param substitution. Since the element
    //     param has its own SymbolId distinct from `input_sym`,
    //     scanning for any `Var(input_sym)` in the body suffices.
    if body_references_sym(&lam.body, input_sym) {
        return false;
    }
    true
}

fn body_references_sym(term: &Term, sym: SymbolId) -> bool {
    if let TermKind::Var(crate::tlc::VarRef::Symbol(s)) = &term.kind {
        if *s == sym {
            return true;
        }
    }
    let mut found = false;
    term.for_each_child(&mut |child| {
        if !found {
            found = body_references_sym(child, sym);
        }
    });
    found
}

#[cfg(test)]
#[path = "ownership_tests.rs"]
mod ownership_tests;
