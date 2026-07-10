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

use super::VarRef;
use crate::builtins::catalog;
use crate::error::CompilerError;
use crate::tlc::var_term_builtin_id;
use crate::{LookupMap, LookupSet};

use crate::ast::{Span, TypeName};
use crate::tlc::{
    ArrayExpr, Def, DefMeta, Lambda, LoopKind, Program, SoacDestination, SoacOp, Term, TermId, TermKind,
};
use crate::types;
use crate::types::Diet;
use crate::types::TypeExt;
use crate::SymbolId;
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

/// Component-sensitive alias information for aggregate values. Futhark tracks
/// tuple/record aliases per field: consuming the whole aggregate flattens all
/// fields, while a projection sees only the selected component.
#[derive(Clone, Debug, Default)]
struct AliasValue {
    owners: LookupSet<OwnerId>,
    components: Option<Vec<AliasValue>>,
}

impl AliasValue {
    fn leaf(owners: LookupSet<OwnerId>) -> Self {
        Self {
            owners,
            components: None,
        }
    }

    fn aggregate(components: Vec<AliasValue>) -> Self {
        Self {
            owners: LookupSet::new(),
            components: Some(components),
        }
    }

    fn flattened(&self) -> LookupSet<OwnerId> {
        let mut owners = self.owners.clone();
        if let Some(components) = &self.components {
            for component in components {
                owners.extend(component.flattened());
            }
        }
        owners
    }

    fn project(&self, index: usize) -> Self {
        let mut projected = self
            .components
            .as_ref()
            .and_then(|components| components.get(index))
            .cloned()
            .unwrap_or_else(|| Self::leaf(self.flattened()));
        projected.owners.extend(self.owners.iter().copied());
        projected
    }

    fn union_with(&mut self, other: &AliasValue) {
        self.owners.extend(other.owners.iter().copied());
        match (&mut self.components, &other.components) {
            (Some(left), Some(right)) if left.len() == right.len() => {
                for (left, right) in left.iter_mut().zip(right) {
                    left.union_with(right);
                }
            }
            (Some(_), Some(_)) | (Some(_), None) | (None, Some(_)) => {
                self.owners.extend(other.flattened());
            }
            (None, None) => {}
        }
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
/// Entry parameters follow the same explicit-`*` rule as ordinary
/// function parameters: a plain `T` entry input is observing — the
/// host retains ownership of the buffer — and only a `*T` entry
/// parameter (`Entry`) hands the shader exclusive ownership at the
/// boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Origin {
    /// Allocated by this function: literal, range, generator, SOAC
    /// output, function call returning `*T`.
    Fresh,
    /// Bound from a `*T` parameter — caller surrendered ownership.
    UniqueParam,
    /// Bound from a `T` parameter — caller still owns. Not mutable.
    NonUniqueParam,
    /// Bound from a `*T` entry parameter — the host surrendered
    /// ownership of the buffer at the pipeline boundary.
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
    pub var_to_owner: LookupMap<SymbolId, OwnerId>,
    /// Every backing owner a value may alias. Aggregate values can carry
    /// several owners (one per non-copy component); `var_to_owner` remains the
    /// single-owner fast path used by in-place promotion.
    pub var_aliases: LookupMap<SymbolId, LookupSet<OwnerId>>,
    /// Component-sensitive form backing `var_aliases`. Kept private so
    /// downstream optimization APIs continue to consume flat owner sets.
    var_alias_values: LookupMap<SymbolId, AliasValue>,
    /// First binder name for each owner — makes use-after-move error
    /// messages user-readable. Populated when an owner is created, never
    /// overwritten.
    pub owner_to_var: LookupMap<OwnerId, SymbolId>,
    /// Per-owner provenance.
    pub origins: LookupMap<OwnerId, Origin>,
    /// Per-term span — for error reporting.
    pub term_spans: LookupMap<TermId, Span>,
    /// Per-term: owners read here.
    pub uses: LookupMap<TermId, LookupSet<OwnerId>>,
    /// Per-term: owners consumed/moved here (e.g. function arg with
    /// `*T` parameter type).
    pub kills: LookupMap<TermId, LookupSet<OwnerId>>,
    /// Per-term: owners newly introduced here.
    pub defs: LookupMap<TermId, LookupSet<OwnerId>>,
    /// Per-term: live-out set. Empty until liveness runs.
    pub live_out: LookupMap<TermId, LookupSet<OwnerId>>,
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

    pub fn aliases_of(&self, sym: SymbolId) -> LookupSet<OwnerId> {
        self.var_aliases.get(&sym).cloned().unwrap_or_default()
    }

    fn alias_value_of(&self, sym: SymbolId) -> AliasValue {
        self.var_alias_values.get(&sym).cloned().unwrap_or_else(|| AliasValue::leaf(self.aliases_of(sym)))
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
    /// Def symbol → index into `program.defs`, so a call site can read
    /// its callee's consumption diet by symbol.
    def_index: LookupMap<SymbolId, usize>,
    /// Memoized `alias_value` results, cleared per def (term ids are
    /// unique only within one). Sound because every query runs after
    /// the term's free variables are bound and a symbol's alias value
    /// is written exactly once.
    alias_cache: LookupMap<TermId, AliasValue>,
    /// We deliberately do not model captured storage-image handles as
    /// linearly consumable across parallel SOAC lanes in v1.
    soac_body_depth: usize,
}

impl<'p> Builder<'p> {
    fn new(program: &'p Program) -> Self {
        let def_index = program.defs.iter().enumerate().map(|(i, d)| (d.name, i)).collect();
        Self {
            model: OwnershipModel::new(),
            next_owner: 0,
            program,
            def_index,
            alias_cache: LookupMap::new(),
            soac_body_depth: 0,
        }
    }

    /// The consumption diet of the function `func` resolves to: the
    /// callee def's `(param_diets, return_diet)` for a `Var(Symbol)`, or a
    /// builtin's diet. Consuming builtins are the two `array_with`
    /// intrinsics (first parameter consuming); every other callee is
    /// observing. Returns `None` when the callee can't be resolved to a
    /// named function (e.g. a variable of function type before
    /// defunctionalization) — the caller then treats it as observing.
    fn callee_diet(&self, func: &Term) -> Option<(&'p [Diet], &'p Diet)> {
        if let TermKind::Var(VarRef::Symbol(sym)) = &func.kind {
            if let Some(&i) = self.def_index.get(sym) {
                let def = &self.program.defs[i];
                return Some((&def.param_diets, &def.return_diet));
            }
        }
        None
    }

    /// Whether the callee consumes (moves out) its `index`-th argument.
    /// Most builtins do not; `image_with` is the linear storage-image update
    /// form and consumes its source handle.
    fn callee_param_consumes(&self, func: &Term, index: usize) -> bool {
        if var_term_builtin_id(func, &self.program.symbols) == Some(catalog().known().image_with) {
            return index == 0;
        }
        self.callee_diet(func).and_then(|(p, _)| p.get(index)).map(Diet::is_consuming).unwrap_or(false)
    }

    /// The callee's return diet — `Diet::observing()` when unknown.
    fn callee_return_diet(&self, func: &Term) -> Diet {
        let known = catalog().known();
        if var_term_builtin_id(func, &self.program.symbols) == Some(known.array_with) {
            // A functional `array_with` returns a fresh (alias-free) array.
            return Diet::Leaf(true);
        }
        if var_term_builtin_id(func, &self.program.symbols) == Some(known.image_with) {
            // A storage-image update returns the next linear handle.
            return Diet::Leaf(true);
        }
        self.callee_diet(func).map(|(_, r)| r.clone()).unwrap_or_else(Diet::observing)
    }

    fn fresh_owner(&mut self, origin: Origin) -> OwnerId {
        let id = OwnerId(self.next_owner);
        self.next_owner += 1;
        self.model.origins.insert(id, origin);
        id
    }

    fn bind(&mut self, sym: SymbolId, owner: OwnerId) {
        self.bind_alias_value(sym, AliasValue::leaf(std::iter::once(owner).collect()));
    }

    fn bind_alias_value(&mut self, sym: SymbolId, value: AliasValue) {
        let aliases = value.flattened();
        self.model.var_alias_values.insert(sym, value);
        self.model.var_aliases.insert(sym, aliases.clone());
        if aliases.len() == 1 {
            let owner = *aliases.iter().next().expect("one alias");
            self.model.var_to_owner.insert(sym, owner);
        } else {
            self.model.var_to_owner.remove(&sym);
        }
        for owner in aliases {
            self.model.owner_to_var.entry(owner).or_insert(sym);
        }
    }

    /// A value of `ty` with every non-copy component owned by a fresh
    /// owner of `origin`: the type's empty skeleton with all missing
    /// owners materialized.
    fn fresh_alias_value_for_type(&mut self, ty: &Type<TypeName>, origin: Origin) -> AliasValue {
        let mut value = Self::empty_alias_value_for_type(ty);
        self.materialize_all_missing(&mut value, ty, origin);
        value
    }

    /// Bind `sym` to the alias value produced by `rhs`, minting fresh
    /// owners for components `rhs` allocates. Owners introduced here
    /// (rather than inherited from existing bindings) are recorded as
    /// defs of `def_site` so the liveness fixed-point subtracts them
    /// from loop-back sets. SOAC captures pass `None`: a capture
    /// renames an outer store, it introduces no per-call value.
    fn bind_value_from_term(
        &mut self,
        sym: SymbolId,
        ty: &Type<TypeName>,
        rhs: &Term,
        def_site: Option<TermId>,
    ) {
        let mut value = self.alias_value(rhs);
        let inherited = value.flattened();
        // Materialize fresh owners against the producer's `*` structure
        // (its diet), read from the callee's signature for a call.
        let (shape_ty, shape_diet) = self.producer_shape(rhs);
        self.materialize_term_value(&mut value, &shape_ty, &shape_diet, rhs);
        let mut owners = value.flattened();
        if owners.is_empty() {
            let origin = self.origin_for_unaliased(rhs);
            value = self.fresh_alias_value_for_type(ty, origin);
            owners = value.flattened();
        }
        if let Some(def_site) = def_site {
            let introduced: LookupSet<OwnerId> = owners.difference(&inherited).copied().collect();
            self.model.defs.entry(def_site).or_default().extend(introduced);
        }
        self.bind_alias_value(sym, value);
    }

    /// The type and diet describing what `rhs` produces: a call's fresh
    /// components come from the callee's return type and diet; other
    /// producers are observing over their own type.
    fn producer_shape(&self, rhs: &Term) -> (Type<TypeName>, Diet) {
        match &rhs.kind {
            TermKind::App { func, args } => (
                callee_return_type(&func.ty, args.len()),
                self.callee_return_diet(func),
            ),
            TermKind::Loop { .. } if matches!(rhs.ty, Type::Constructed(TypeName::StorageTexture, _)) => {
                (rhs.ty.clone(), Diet::Leaf(true))
            }
            TermKind::Coerce { inner, .. } => self.producer_shape(inner),
            _ => (rhs.ty.clone(), Diet::observing()),
        }
    }

    fn visit_def(&mut self, def: &Def) {
        // Term ids are unique within a def but restart across defs, so
        // the memo table must not outlive one.
        self.alias_cache = LookupMap::new();
        // The function's parameters live on the top-level Lambda's
        // params field — Defs whose body is anything else are zero-arity
        // constants and have no slots to track here.
        if let TermKind::Lambda(lam) = &def.body.kind {
            let is_entry = matches!(def.meta, DefMeta::EntryPoint(_));
            self.bind_params(lam, &def.param_diets, is_entry);
            self.visit_term(&lam.body);
            if !is_entry {
                self.check_alias_free_return(&lam.ret_ty, &def.return_diet, &lam.body);
            }
        } else {
            self.visit_term(&def.body);
        }
    }

    /// A function that declares a `*` (alias-free) return must actually
    /// produce one: the positive evidence the checker cannot supply once
    /// uniqueness leaves expression types. Each `*`-marked return
    /// component must be backed only by mutable storage — freshly
    /// allocated, or ownership surrendered by a consuming parameter —
    /// never by an observing parameter or a shared global.
    fn check_alias_free_return(&mut self, ret_ty: &Type<TypeName>, ret_diet: &Diet, body: &Term) {
        if !ret_diet.is_consuming() {
            return;
        }
        let value = self.alias_value(body);
        if !self.return_component_ok(ret_ty, ret_diet, Some(body), &value) {
            let span = self.model.term_spans.get(&body.id).copied();
            self.model.build_errors.push((
                "cannot prove the `*` (alias-free) result is alias-free; \
                 it must be freshly allocated or a consumed parameter (take any \
                 shared input as an explicit parameter, or use `copy`)"
                    .to_string(),
                span,
            ));
        }
    }

    /// Recursively check that every `*`-marked (per the diet) component of
    /// the return is alias-free. `term` is the sub-expression producing
    /// this component when syntactically available (used to recognize a
    /// fresh producer for an untracked component).
    fn return_component_ok(
        &mut self,
        ret_ty: &Type<TypeName>,
        ret_diet: &Diet,
        term: Option<&Term>,
        value: &AliasValue,
    ) -> bool {
        if ret_diet.is_consuming_at_root() {
            let owners = value.flattened();
            return if owners.is_empty() {
                // No tracked storage: accept only a recognized fresh
                // producer (an untracked component of a call result is
                // vouched for by the callee's own checked signature).
                term.map_or(true, |t| self.is_definitely_alias_free(t))
            } else {
                owners
                    .iter()
                    .all(|owner| self.model.origin(*owner).map(Origin::is_mutable).unwrap_or(false))
            };
        }
        match ret_ty {
            Type::Constructed(TypeName::Tuple(_), args) | Type::Constructed(TypeName::Record(_), args) => {
                let parts = match term.map(|t| &t.kind) {
                    Some(TermKind::Tuple(parts)) if parts.len() == args.len() => Some(parts.as_slice()),
                    _ => None,
                };
                args.iter().enumerate().all(|(i, arg_ty)| {
                    let component_term = parts.and_then(|p| p.get(i));
                    let component_value = value.project(i);
                    self.return_component_ok(
                        arg_ty,
                        &ret_diet.component(i),
                        component_term,
                        &component_value,
                    )
                })
            }
            _ => true,
        }
    }

    fn bind_params(&mut self, lam: &Lambda, param_diets: &[Diet], is_entry: bool) {
        for (index, (sym, ty)) in lam.params.iter().enumerate() {
            if types::is_copy(ty) {
                continue;
            }
            // Entry params are mutable iff explicitly marked `*`. A plain
            // `[]T` view read by a compute entry is immutable, mirroring a
            // non-consuming function param; an unsoundly-mutable
            // `Origin::Entry` here would let ownership rewrite non-tail
            // SOACs to InputBuffer destinations that clobber the caller.
            let consuming = param_diets.get(index).map(Diet::is_consuming).unwrap_or(false);
            let origin = if is_entry && consuming {
                Origin::Entry
            } else if consuming {
                Origin::UniqueParam
            } else {
                Origin::NonUniqueParam
            };
            let value = self.fresh_alias_value_for_type(ty, origin);
            self.bind_alias_value(*sym, value);
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
            TermKind::Var(VarRef::Symbol(sym)) => {
                let aliases = self.model.aliases_of(*sym);
                self.model.uses.entry(term.id).or_default().extend(aliases);
            }
            TermKind::Let {
                name,
                name_ty,
                rhs,
                body,
            } => {
                self.visit_term(rhs);
                if !types::is_copy(name_ty) {
                    self.bind_value_from_term(*name, name_ty, rhs, Some(term.id));
                }
                self.visit_term(body);
            }
            TermKind::App { func, args } => {
                self.visit_term(func);
                if var_term_builtin_id(func, &self.program.symbols) == Some(catalog().known().image_with)
                    && self.soac_body_depth > 0
                {
                    let span = self.model.term_spans.get(&term.id).copied();
                    self.model.build_errors.push((
                        "`storage_image with [coord] = value` is linear and cannot be used inside a SOAC body in v1; use a serial loop or a future structured bulk image writer"
                            .to_string(),
                        span,
                    ));
                }
                let mut killed_this_call: LookupSet<OwnerId> = LookupSet::new();
                for (arg_index, arg) in args.iter().enumerate() {
                    self.visit_term(arg);
                    if self.callee_param_consumes(func, arg_index) {
                        let aliases = self.alias_targets(arg);
                        if !aliases.is_empty() {
                            let rejected: Vec<OwnerId> = aliases
                                .iter()
                                .copied()
                                .filter(|owner| {
                                    !self.model.origin(*owner).map(Origin::is_mutable).unwrap_or(false)
                                })
                                .collect();
                            if !rejected.is_empty() {
                                for owner in rejected {
                                    let var_name = owner_display_name(&self.model, self.program, owner);
                                    let span = self.model.term_spans.get(&arg.id).copied();
                                    self.model.build_errors.push((
                                        format!(
                                            "cannot consume observing value `{}`; \
                                             consuming parameter requires an alias-free argument",
                                            var_name
                                        ),
                                        span,
                                    ));
                                }
                            } else {
                                for owner in aliases.iter().copied() {
                                    // A second `*T` arg resolving to an owner the
                                    // call already consumed is a duplicate move.
                                    if !killed_this_call.insert(owner) {
                                        let var_name = owner_display_name(&self.model, self.program, owner);
                                        let span = self.model.term_spans.get(&arg.id).copied();
                                        self.model
                                            .build_errors
                                            .push((format!("use of moved value `{}`", var_name), span));
                                    }
                                }
                                self.model.kills.entry(term.id).or_default().extend(aliases);
                            }
                        } else if !self.is_definitely_alias_free(arg) {
                            let span = self.model.term_spans.get(&arg.id).copied();
                            self.model.build_errors.push((
                                "cannot consume observing value; consuming parameter requires an alias-free argument"
                                    .to_string(),
                                span,
                            ));
                        }
                    }
                }
            }
            TermKind::Lambda(lam) => self.visit_lambda(lam),
            TermKind::Soac(op) => self.visit_soac(op, term.id),
            TermKind::ArrayExpr(ae) => self.visit_array_expr(ae),
            TermKind::TupleProj { tuple, .. }
                if matches!(&tuple.kind, TermKind::Var(VarRef::Symbol(_))) =>
            {
                let aliases = self.alias_targets(term);
                self.model.uses.entry(term.id).or_default().extend(aliases);
            }
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
                    if matches!(loop_var_ty, Type::Constructed(TypeName::StorageTexture, _)) {
                        let init_aliases = self.alias_targets(init);
                        let owner = self.fresh_owner(Origin::Fresh);
                        self.model.defs.entry(term.id).or_default().insert(owner);
                        self.model.kills.entry(term.id).or_default().extend(init_aliases);
                        self.bind(*loop_var, owner);
                    } else {
                        self.bind_value_from_term(*loop_var, loop_var_ty, init, Some(term.id));
                    }
                }

                // Sub-bindings extracted from loop_var (e.g. tuple
                // destructuring). Each is `(name, ty, extraction_expr)`.
                for (name, ty, extract) in init_bindings {
                    self.visit_term(extract);
                    if !types::is_copy(ty) {
                        self.bind_value_from_term(*name, ty, extract, Some(term.id));
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
                self.bind_value_from_term(*capture_sym, capture_ty, capture_term, None);
            }
        }
        self.soac_body_depth += 1;
        self.visit_term(&sb.lam.body);
        self.soac_body_depth -= 1;
    }

    /// SOACs: each input array contributes uses; element params
    /// inherit mutability from their matched input (Map/Screma), or
    /// from the single input (Reduce/Scan/Filter/ReduceByIndex).
    /// Accumulator params (Reduce/Scan/Screma) are the body's
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
            SoacOp::Scan { op, ne, input, .. } => {
                self.visit_term(ne);
                self.visit_array_expr(input);
                self.bind_reducer_params(&op.lam, input, soac_id);
                self.visit_soac_body(op);
            }
            SoacOp::Filter {
                map_lam, pred, input, ..
            } => {
                self.visit_array_expr(input);
                // A fused producer map reads the input element and may capture or
                // consume outer values; bind its element param and track its body
                // so those dependencies are visible to liveness / move checking.
                if let Some(map_lam) = map_lam {
                    if let Some((sym, ty)) = map_lam.lam.params.first() {
                        if !types::is_copy(ty) {
                            let origin = self.element_origin_from_input(input);
                            let owner = self.fresh_owner(origin);
                            self.bind(*sym, owner);
                            self.record_per_call_def(soac_id, owner);
                        }
                    }
                    self.visit_soac_body(map_lam);
                }
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
            SoacOp::Scatter { lam, inputs, .. } => {
                self.visit_soac_body(lam);
                for input in inputs {
                    self.visit_array_expr(input);
                }
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
            SoacOp::Screma {
                lanes,
                accumulators,
                inputs,
            } => {
                for acc in accumulators {
                    self.visit_term(&acc.ne);
                }
                for ae in inputs {
                    self.visit_array_expr(ae);
                }
                for lane in lanes {
                    for ((sym, ty), input) in lane.lam.lam.params.iter().zip(inputs.iter()) {
                        if !types::is_copy(ty) {
                            let origin = self.element_origin_from_input(input);
                            let owner = self.fresh_owner(origin);
                            self.bind(*sym, owner);
                            self.record_per_call_def(soac_id, owner);
                        }
                    }
                }
                for acc in accumulators {
                    if let Some(((acc_sym, acc_ty), elem_params)) = acc.step_lam.lam.params.split_first() {
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
                    for (sym, ty) in &acc.reduce_op.lam.params {
                        if !types::is_copy(ty) {
                            let owner = self.fresh_owner(Origin::Fresh);
                            self.bind(*sym, owner);
                            self.record_per_call_def(soac_id, owner);
                        }
                    }
                }
                for lane in lanes {
                    self.visit_soac_body(&lane.lam);
                }
                for acc in accumulators {
                    self.visit_soac_body(&acc.step_lam);
                    self.visit_soac_body(&acc.reduce_op);
                }
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
    fn element_origin_from_input(&mut self, ae: &ArrayExpr) -> Origin {
        match ae {
            ArrayExpr::Var(vr, ty) => {
                let t = super::synthetic_atom_var_term(*vr, ty.clone());
                let aliases = self.alias_targets(&t);
                if !aliases.is_empty() {
                    if aliases
                        .iter()
                        .all(|owner| self.model.origin(*owner).map(|o| o.is_mutable()).unwrap_or(false))
                    {
                        return Origin::BorrowedMutableElement;
                    }
                    return Origin::Borrowed;
                }
                // No tracked owner (an untracked SOAC input): observe it.
                // A `*` input is a bound parameter, so it always has a
                // tracked owner handled above.
                Origin::Borrowed
            }
            // Fresh-producer ArrayExprs: literal/range synthesize a new array,
            // so element views are mutable.
            ArrayExpr::Literal(_) | ArrayExpr::Range { .. } => Origin::BorrowedMutableElement,
            // Storage-buffer-backed views: conservative borrow.
            ArrayExpr::StorageView(_) => Origin::Borrowed,
            // Zip is a phase-scoped sentinel that should be absorbed
            // by `tlc::soa::run` before we get here. If one survives,
            // be conservative.
            ArrayExpr::Zip(_) => Origin::Borrowed,
        }
    }

    fn visit_array_expr(&mut self, ae: &ArrayExpr) {
        match ae {
            // A named SOAC-input atom carries no `TermId` of its own, so its use
            // can't be recorded in the id-keyed `uses` table without fabricating
            // a placeholder id (which would alias a real term and corrupt
            // `transfer`). Liveness reads this use natively in
            // `Liveness::analyze_array_expr` instead; nothing else consumes it.
            ArrayExpr::Var(..) => {}
            ArrayExpr::Zip(aes) => {
                for ae in aes {
                    self.visit_array_expr(ae);
                }
            }
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
            ArrayExpr::StorageView(crate::tlc::StorageView { offset, len, .. }) => {
                self.visit_term(offset);
                self.visit_term(len);
            }
        }
    }

    /// Every pre-existing backing owner that may be reachable through the
    /// value produced by `term`. Aggregate projections deliberately retain
    /// the complete set: this may suppress an optimization, but cannot make
    /// observing storage consumable.
    fn alias_targets(&mut self, term: &Term) -> LookupSet<OwnerId> {
        self.alias_value(term).flattened()
    }

    fn alias_value(&mut self, term: &Term) -> AliasValue {
        // Vars resolve by symbol lookup, outside the cache: synthetic
        // probe terms (`atom_var_term` with a fresh `TermIdSource` in
        // `element_origin_from_input`) carry ids that collide with real
        // terms, and a Var lookup is as cheap as a cache hit anyway.
        if let TermKind::Var(vr) = &term.kind {
            return match vr {
                VarRef::Symbol(sym) => self.model.alias_value_of(*sym),
                _ => Self::empty_alias_value_for_type(&term.ty),
            };
        }
        if let Some(cached) = self.alias_cache.get(&term.id) {
            return cached.clone();
        }
        let value = self.compute_alias_value(term);
        self.alias_cache.insert(term.id, value.clone());
        value
    }

    fn compute_alias_value(&mut self, term: &Term) -> AliasValue {
        match &term.kind {
            TermKind::Var(VarRef::Symbol(sym)) => self.model.alias_value_of(*sym),
            TermKind::Var(_) => Self::empty_alias_value_for_type(&term.ty),
            TermKind::Coerce { inner, .. } => self.alias_value(inner),
            TermKind::Index { array, .. } => AliasValue::leaf(self.alias_targets(array)),
            TermKind::TupleProj { tuple, idx } => self.alias_value(tuple).project(*idx),
            TermKind::Tuple(parts) => {
                AliasValue::aggregate(parts.iter().map(|part| self.alias_value(part)).collect())
            }
            TermKind::VecLit(parts) => AliasValue::leaf(self.aliases_of_terms(parts.iter())),
            TermKind::If {
                then_branch,
                else_branch,
                ..
            } => {
                // A branch yielding a storage image yields *the* handle: one
                // arm updates it, another may pass it through unchanged. The
                // result is the sole live handle either way, so it is a fresh
                // owner rather than an alias of the arms' inputs — otherwise
                // the updating arm's `with` would consume an owner the merged
                // result keeps live.
                if matches!(term.ty, Type::Constructed(TypeName::StorageTexture, _)) {
                    return Self::empty_alias_value_for_type(&term.ty);
                }
                let mut value = self.alias_value(then_branch);
                value.union_with(&self.alias_value(else_branch));
                value
            }
            TermKind::Loop { init, body, .. } => {
                if matches!(term.ty, Type::Constructed(TypeName::StorageTexture, _)) {
                    return Self::empty_alias_value_for_type(&term.ty);
                }
                let mut value = self.alias_value(init);
                value.union_with(&self.alias_value(body));
                value
            }
            TermKind::Let { body, .. } => self.alias_value(body),
            TermKind::App { func, args } => {
                let known = catalog().known();
                if var_term_builtin_id(func, &self.program.symbols) == Some(known.array_with_in_place) {
                    return args
                        .first()
                        .map(|arg| self.alias_value(arg))
                        .unwrap_or_else(|| Self::empty_alias_value_for_type(&term.ty));
                }
                if var_term_builtin_id(func, &self.program.symbols) == Some(known.array_with) {
                    return Self::empty_alias_value_for_type(&term.ty);
                }

                // The result's per-component uniqueness comes from the
                // callee's return diet (a `*` component is fresh/alias-free);
                // an observing component aliases every argument passed to a
                // non-consuming parameter.
                let return_ty = callee_return_type(&func.ty, args.len());
                let return_diet = self.callee_return_diet(func);
                let mut aliases = LookupSet::new();
                for (index, arg) in args.iter().enumerate() {
                    if !self.callee_param_consumes(func, index) {
                        aliases.extend(self.alias_targets(arg));
                    }
                }
                Self::observing_alias_value_for_type(&return_ty, &return_diet, &aliases)
            }
            TermKind::ArrayExpr(ae) => AliasValue::leaf(self.array_expr_aliases(ae)),
            TermKind::Soac(op) => {
                // A SOAC result carries no `*`; it is observing over the
                // aliases its inputs/accumulators contribute.
                let aliases = self.soac_result_aliases(op, &term.ty);
                Self::observing_alias_value_for_type(&term.ty, &Diet::observing(), &aliases)
            }
            TermKind::Lambda(_)
            | TermKind::IntLit(_)
            | TermKind::FloatLit(_)
            | TermKind::BoolLit(_)
            | TermKind::UnitLit
            | TermKind::BinOp(_)
            | TermKind::UnOp(_)
            | TermKind::Extern(_)
            | TermKind::OutputSlotStore { .. } => Self::empty_alias_value_for_type(&term.ty),
        }
    }

    /// The ownerless skeleton of `ty`: aggregates become component
    /// trees, everything else an empty leaf.
    fn empty_alias_value_for_type(ty: &Type<TypeName>) -> AliasValue {
        match ty {
            Type::Constructed(TypeName::Tuple(_), args) | Type::Constructed(TypeName::Record(_), args) => {
                AliasValue::aggregate(args.iter().map(Self::empty_alias_value_for_type).collect())
            }
            _ => AliasValue::default(),
        }
    }

    /// Futhark's interprocedural rule: every non-alias-free result component
    /// aliases all arguments passed to observing parameters. Components the
    /// `diet` marks `*` are fresh (empty skeleton) and receive fresh owners
    /// when bound.
    fn observing_alias_value_for_type(
        ty: &Type<TypeName>,
        diet: &Diet,
        aliases: &LookupSet<OwnerId>,
    ) -> AliasValue {
        if diet.is_consuming_at_root() {
            return Self::empty_alias_value_for_type(ty);
        }
        match ty {
            Type::Constructed(TypeName::Tuple(_), args) | Type::Constructed(TypeName::Record(_), args) => {
                AliasValue::aggregate(
                    args.iter()
                        .enumerate()
                        .map(|(i, arg)| {
                            Self::observing_alias_value_for_type(arg, &diet.component(i), aliases)
                        })
                        .collect(),
                )
            }
            _ if types::is_copy(ty) => AliasValue::default(),
            _ => AliasValue::leaf(aliases.clone()),
        }
    }

    fn materialize_term_value(
        &mut self,
        value: &mut AliasValue,
        ty: &Type<TypeName>,
        diet: &Diet,
        term: &Term,
    ) {
        // A `*` at this node means the whole value is freshly owned.
        if diet.is_consuming_at_root() {
            self.materialize_all_missing(value, ty, Origin::Fresh);
            if value.flattened().is_empty() {
                value.owners.insert(self.fresh_owner(Origin::Fresh));
            }
            return;
        }
        match ty {
            Type::Constructed(TypeName::Tuple(_), args) | Type::Constructed(TypeName::Record(_), args) => {
                if value.components.is_none() {
                    *value = Self::empty_alias_value_for_type(ty);
                }
                if let Some(components) = value.components.as_mut() {
                    let term_parts = match &term.kind {
                        TermKind::Tuple(parts) if parts.len() == components.len() => Some(parts.as_slice()),
                        _ => None,
                    };
                    for (index, (component, component_ty)) in components.iter_mut().zip(args).enumerate() {
                        let component_term = term_parts.and_then(|parts| parts.get(index)).unwrap_or(term);
                        self.materialize_term_value(
                            component,
                            component_ty,
                            &diet.component(index),
                            component_term,
                        );
                    }
                }
            }
            _ if types::is_copy(ty) => {}
            _ => {
                if self.allocates_fresh_outer(term, ty)
                    || value.owners.is_empty() && self.is_definitely_alias_free(term)
                {
                    value.owners.insert(self.fresh_owner(Origin::Fresh));
                }
            }
        }
    }

    /// Fill every ownerless non-copy leaf of `value` with a fresh owner
    /// of `origin`; an all-copy subtree under a `*` boundary gets one
    /// owner at the boundary itself.
    fn materialize_all_missing(&mut self, value: &mut AliasValue, ty: &Type<TypeName>, origin: Origin) {
        match ty {
            Type::Constructed(TypeName::Tuple(_), args) | Type::Constructed(TypeName::Record(_), args) => {
                if value.components.is_none() {
                    *value = Self::empty_alias_value_for_type(ty);
                }
                if let Some(components) = value.components.as_mut() {
                    for (component, component_ty) in components.iter_mut().zip(args) {
                        self.materialize_all_missing(component, component_ty, origin);
                    }
                }
            }
            _ if types::is_copy(ty) => {}
            _ if value.owners.is_empty() => {
                value.owners.insert(self.fresh_owner(origin));
            }
            _ => {}
        }
    }

    /// Whether `term`'s outermost value is a freshly allocated store:
    /// array/range literals, the fresh-array SOACs, an array-typed
    /// Screma result, and the functional `array_with` builtin. The
    /// single allocator list shared by leaf materialization and
    /// `is_definitely_alias_free`.
    fn allocates_fresh_outer(&self, term: &Term, ty: &Type<TypeName>) -> bool {
        match &term.kind {
            TermKind::ArrayExpr(ArrayExpr::Literal(_) | ArrayExpr::Range { .. }) => true,
            TermKind::Soac(SoacOp::Map { .. } | SoacOp::Scan { .. } | SoacOp::Filter { .. }) => true,
            TermKind::Soac(SoacOp::Screma { .. }) => ty.is_array(),
            TermKind::App { func, .. } => {
                var_term_builtin_id(func, &self.program.symbols) == Some(catalog().known().array_with)
            }
            _ => false,
        }
    }

    fn aliases_of_terms<'a>(&mut self, terms: impl IntoIterator<Item = &'a Term>) -> LookupSet<OwnerId> {
        let mut aliases = LookupSet::new();
        for term in terms {
            aliases.extend(self.alias_targets(term));
        }
        aliases
    }

    fn array_expr_aliases(&self, ae: &ArrayExpr) -> LookupSet<OwnerId> {
        match ae {
            ArrayExpr::Var(VarRef::Symbol(sym), _) => self.model.aliases_of(*sym),
            ArrayExpr::Var(VarRef::Builtin { .. }, _) | ArrayExpr::StorageView(_) => LookupSet::new(),
            ArrayExpr::Zip(parts) => {
                let mut aliases = LookupSet::new();
                for part in parts {
                    aliases.extend(self.array_expr_aliases(part));
                }
                aliases
            }
            ArrayExpr::Literal(_) => LookupSet::new(),
            ArrayExpr::Range { .. } => LookupSet::new(),
        }
    }

    /// SOACs allocate fresh outer arrays. Existing aliases matter when a
    /// result can itself contain non-copy elements, or when the SOAC returns
    /// an accumulator directly.
    fn soac_result_aliases(&mut self, op: &SoacOp, _result_ty: &Type<TypeName>) -> LookupSet<OwnerId> {
        match op {
            SoacOp::Map { .. } | SoacOp::Filter { .. } | SoacOp::Scan { .. } => LookupSet::new(),
            SoacOp::Scatter { inputs, .. } => self.aliases_of_array_exprs(inputs),
            SoacOp::Screma { accumulators, .. } => {
                let mut aliases = LookupSet::new();
                for acc in accumulators {
                    aliases.extend(self.alias_targets(&acc.ne));
                }
                aliases
            }
            SoacOp::Reduce { ne, input, .. } => {
                let mut aliases = self.alias_targets(ne);
                aliases.extend(self.array_expr_aliases(input));
                aliases
            }
            SoacOp::ReduceByIndex {
                ne, indices, values, ..
            } => {
                let mut aliases = self.alias_targets(ne);
                aliases.extend(self.array_expr_aliases(indices));
                aliases.extend(self.array_expr_aliases(values));
                aliases
            }
        }
    }

    fn aliases_of_array_exprs(&self, aes: &[ArrayExpr]) -> LookupSet<OwnerId> {
        let mut aliases = LookupSet::new();
        for ae in aes {
            aliases.extend(self.array_expr_aliases(ae));
        }
        aliases
    }

    /// Whether an owner-less expression is known to construct an alias-free
    /// value. Expressions with tracked aliases are judged by those aliases;
    /// this predicate only justifies a new owner for allocating forms.
    fn is_definitely_alias_free(&mut self, term: &Term) -> bool {
        if types::is_copy(&term.ty) {
            return true;
        }
        if !self.alias_targets(term).is_empty() {
            return false;
        }
        match &term.kind {
            _ if self.allocates_fresh_outer(term, &term.ty) => true,
            TermKind::Tuple(parts) | TermKind::VecLit(parts) => {
                parts.iter().all(|part| types::is_copy(&part.ty) || self.is_definitely_alias_free(part))
            }
            TermKind::If {
                then_branch,
                else_branch,
                ..
            } => self.is_definitely_alias_free(then_branch) && self.is_definitely_alias_free(else_branch),
            // A zero-trip loop returns init unchanged, so the result is
            // alias-free only when both init and body are.
            TermKind::Loop { init, body, .. } => {
                if matches!(term.ty, Type::Constructed(TypeName::StorageTexture, _)) {
                    return true;
                }
                self.is_definitely_alias_free(init) && self.is_definitely_alias_free(body)
            }
            // A call result is alias-free only when the callee declares a
            // `*` return; otherwise it may alias storage reachable through
            // the callee (top-level constants, captured state) that no
            // argument-derived alias set can see.
            TermKind::App { func, .. } => self.callee_return_diet(func).is_consuming(),
            TermKind::Let { body, .. } => self.is_definitely_alias_free(body),
            // A non-array Screma result is the accumulator; its aliases
            // are the `ne` aliases, which the empty-alias guard covered.
            TermKind::Soac(SoacOp::Screma { .. }) => true,
            TermKind::Coerce { inner, .. } => self.is_definitely_alias_free(inner),
            _ => false,
        }
    }

    /// The origin to assign when a Let / Loop binder produces a fresh
    /// owner (i.e., `alias_target` returned `None`). `Fresh` for
    /// recognized producers (literals, ranges, generators, fresh
    /// intrinsics, calls returning `*T`); `Borrowed` for anything we
    /// can't classify (user function returning non-unique non-copy,
    /// lambda invocation, etc.).
    fn origin_for_unaliased(&mut self, rhs: &Term) -> Origin {
        if self.is_definitely_alias_free(rhs) {
            return Origin::Fresh;
        }
        Origin::Borrowed
    }
}

/// The user-facing name for an owner in diagnostics: its first binder,
/// or `<value>` when no binder ever named it.
fn owner_display_name(model: &OwnershipModel, program: &Program, owner: OwnerId) -> String {
    model
        .owner_to_var
        .get(&owner)
        .and_then(|s| program.symbols.get(*s).cloned())
        .unwrap_or_else(|| "<value>".to_string())
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
        TermKind::Var(VarRef::Symbol(sym)) => model.owner_of(*sym),
        // `Index` and `TupleProj` are projection variants whose result
        // aliases the base.
        TermKind::Index { array, .. } => alias_target_of(array, model, program),
        TermKind::TupleProj { tuple, .. } => alias_target_of(tuple, model, program),
        // In-place `array_with` returns the buffer it consumed.
        TermKind::App { func, args } => {
            if var_term_builtin_id(func, &program.symbols) == Some(catalog().known().array_with_in_place) {
                return alias_target_of(args.first()?, model, program);
            }
            None
        }
        _ => None,
    }
}

/// The result type of applying `func_ty` to `n_args` arguments — the
/// residue after peeling `n_args` arrows. This is where a function's
/// `*` return contract lives now that expression types no longer carry
/// uniqueness: ownership reads "does this call produce a fresh value"
/// from the callee's signature, not the call's inferred type.
fn callee_return_type(func_ty: &Type<TypeName>, n_args: usize) -> Type<TypeName> {
    let mut current = func_ty.clone();
    for _ in 0..n_args {
        match current {
            Type::Constructed(TypeName::Arrow, args) if args.len() == 2 => {
                current = args[1].clone();
            }
            _ => break,
        }
    }
    current
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

type LiveSet = LookupSet<OwnerId>;

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
            | TermKind::UnitLit
            | TermKind::BinOp(_)
            | TermKind::UnOp(_)
            | TermKind::Extern(_) => self.transfer(term.id, live_after),

            TermKind::Coerce { inner, .. } => self.analyze(inner, live_after),

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
                let live_after = self.transfer(term.id, live_after);
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

            TermKind::Tuple(parts) | TermKind::VecLit(parts) => {
                let mut live = self.transfer(term.id, live_after);
                for p in parts.iter().rev() {
                    live = self.analyze(p, live);
                }
                live
            }
            TermKind::TupleProj { tuple, .. } => {
                let live = self.transfer(term.id, live_after);
                if matches!(&tuple.kind, TermKind::Var(VarRef::Symbol(_))) {
                    live
                } else {
                    self.analyze(tuple, live)
                }
            }
            TermKind::Index { array, index } => {
                let live = self.transfer(term.id, live_after);
                let live = self.analyze(index, live);
                self.analyze(array, live)
            }
            TermKind::OutputSlotStore { .. } => {
                unreachable!("OutputSlotStore introduced by tlc::normalize_outputs (post-ownership)")
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
            SoacOp::Scan { op, ne, input, .. } => {
                let after_op = self.soac_envelope_fixed_point(op, &per_call_defs, live_after);
                let after_input = self.analyze_array_expr(input, after_op);
                self.analyze(ne, after_input)
            }
            SoacOp::Filter {
                map_lam, pred, input, ..
            } => {
                // Backward dataflow: input → map_lam → pred. A fused map's
                // captures are live before the filter, so thread its envelope
                // between the predicate and the input.
                let after_pred = self.soac_envelope_fixed_point(pred, &per_call_defs, live_after);
                let after_map = match map_lam {
                    Some(ml) => self.soac_envelope_fixed_point(ml, &per_call_defs, after_pred),
                    None => after_pred,
                };
                self.analyze_array_expr(input, after_map)
            }
            SoacOp::Scatter { lam, inputs, .. } => {
                let mut live = self.soac_envelope_fixed_point(lam, &per_call_defs, live_after);
                for ae in inputs.iter().rev() {
                    live = self.analyze_array_expr(ae, live);
                }
                live
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
            SoacOp::Screma {
                lanes,
                accumulators,
                inputs,
            } => {
                let mut live = live_after;
                for lane in lanes {
                    live = self.soac_envelope_fixed_point(&lane.lam, &per_call_defs, live);
                }
                for acc in accumulators {
                    live = self.soac_envelope_fixed_point(&acc.step_lam, &per_call_defs, live);
                    live = self.soac_envelope_fixed_point(&acc.reduce_op, &per_call_defs, live);
                }
                for ae in inputs.iter().rev() {
                    live = self.analyze_array_expr(ae, live);
                }
                for acc in accumulators.iter().rev() {
                    live = self.analyze(&acc.ne, live);
                }
                live
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
            // A named SOAC-input atom is a use of `sym`, so its owner is live
            // before the SOAC. It has no `TermId`, so add the use directly rather
            // than routing through `transfer` on a fabricated (and colliding)
            // placeholder id.
            ArrayExpr::Var(VarRef::Symbol(sym), _) => {
                let mut live = live_after;
                live.extend(self.model.aliases_of(*sym));
                live
            }
            ArrayExpr::Var(VarRef::Builtin { .. }, _) => live_after,
            ArrayExpr::Zip(aes) => {
                let mut live = live_after;
                for ae in aes.iter().rev() {
                    live = self.analyze_array_expr(ae, live);
                }
                live
            }
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
            ArrayExpr::StorageView(crate::tlc::StorageView { offset, len, .. }) => {
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
pub fn apply_ownership(program: &mut Program) -> crate::error::Result<()> {
    let model = analyze(program);
    if let Some(err) = check_use_after_move(program, &model) {
        return Err(err);
    }
    if let Some(err) = check_linear_image_results(program, &model) {
        return Err(err);
    }
    let consuming_soacs: LookupSet<TermId> =
        eligible_consuming_soacs(program, &model).into_iter().collect();

    // Promotion of `array_with` → `array_with_inplace` is keyed by the
    // catalog (BuiltinId for the in-place form is looked up at the
    // rewrite site), not by symbol-table identity. We always run the
    // rewriter — even with no consuming SOACs, an `array_with` whose
    // source is a unique single-use binding is still promotable.

    let defs_in = std::mem::take(&mut program.defs);
    let mut rewriter = Rewriter {
        model: &model,
        program,
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
    Ok(())
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
    if let Some(err) = check_linear_image_results(program, &model) {
        return Err(err);
    }
    Ok(())
}

/// Walk the model for any owner that is killed at a term while still
/// being in that term's `live_out` — i.e., a successor still reads it.
/// Returns the first such violation as a CompilerError; later
/// violations are not reported in this pass (one diagnostic per
/// compile is consistent with how the rest of the pipeline reports).
fn check_use_after_move(program: &Program, model: &OwnershipModel) -> Option<CompilerError> {
    use CompilerError;
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
    let var_name = owner_display_name(model, program, owner);
    Some(CompilerError::AliasError(
        format!("use of moved value `{}`", var_name),
        span,
    ))
}

fn check_linear_image_results(program: &Program, model: &OwnershipModel) -> Option<CompilerError> {
    for def in &program.defs {
        if let Some(err) =
            check_linear_image_results_in_term(&def.body, program, model, LinearImageUseContext::Used)
        {
            return Some(err);
        }
    }
    None
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LinearImageUseContext {
    Used,
    Discarded,
}

fn is_image_with_app(term: &Term, program: &Program) -> bool {
    match &term.kind {
        TermKind::App { func, args } if args.len() == 3 => {
            var_term_builtin_id(func, &program.symbols) == Some(catalog().known().image_with)
        }
        _ => false,
    }
}

fn is_image_load_app(term: &Term, program: &Program) -> bool {
    match &term.kind {
        TermKind::App { func, args } if args.len() == 2 => {
            var_term_builtin_id(func, &program.symbols) == Some(catalog().known().image_load)
        }
        _ => false,
    }
}

fn image_update_drop_error(term: &Term, model: &OwnershipModel) -> CompilerError {
    CompilerError::AliasError(
        "linear storage-image update result must be threaded to another update, observed, returned, or used as the unit entry tail"
            .to_string(),
        model.term_spans.get(&term.id).copied().or(Some(term.span)),
    )
}

fn check_linear_image_results_in_term(
    term: &Term,
    program: &Program,
    model: &OwnershipModel,
    context: LinearImageUseContext,
) -> Option<CompilerError> {
    if is_image_with_app(term, program) && context == LinearImageUseContext::Discarded {
        return Some(image_update_drop_error(term, model));
    }

    if let TermKind::Let {
        name_ty, rhs, body, ..
    } = &term.kind
    {
        if matches!(name_ty, Type::Constructed(TypeName::StorageTexture, _)) {
            let introduced: LookupSet<OwnerId> = model
                .defs
                .get(&term.id)
                .into_iter()
                .flatten()
                .copied()
                .filter(|owner| model.origin(*owner) == Some(Origin::Fresh))
                .collect();
            if !introduced.is_empty() && !linear_image_owner_is_threaded(body, &introduced, model) {
                let span = model.term_spans.get(&rhs.id).copied().or(Some(rhs.span));
                return Some(CompilerError::AliasError(
                    "linear storage-image update result must be threaded to another update, observed, returned, or used as the unit entry tail"
                        .to_string(),
                    span,
                ));
            }
        }
    }

    match &term.kind {
        TermKind::Let {
            name_ty, rhs, body, ..
        } => {
            let rhs_context = if matches!(name_ty, Type::Constructed(TypeName::StorageTexture, _)) {
                LinearImageUseContext::Used
            } else {
                LinearImageUseContext::Discarded
            };
            check_linear_image_results_in_term(rhs, program, model, rhs_context)
                .or_else(|| check_linear_image_results_in_term(body, program, model, context))
        }
        TermKind::Lambda(lam) => {
            check_linear_image_results_in_term(&lam.body, program, model, LinearImageUseContext::Used)
        }
        TermKind::If {
            cond,
            then_branch,
            else_branch,
        } => check_linear_image_results_in_term(cond, program, model, LinearImageUseContext::Discarded)
            .or_else(|| check_linear_image_results_in_term(then_branch, program, model, context))
            .or_else(|| check_linear_image_results_in_term(else_branch, program, model, context)),
        TermKind::Loop {
            init,
            init_bindings,
            kind,
            body,
            ..
        } => {
            let init_err =
                check_linear_image_results_in_term(init, program, model, LinearImageUseContext::Used);
            if init_err.is_some() {
                return init_err;
            }
            for (_, _, extract) in init_bindings {
                if let Some(err) =
                    check_linear_image_results_in_term(extract, program, model, LinearImageUseContext::Used)
                {
                    return Some(err);
                }
            }
            let kind_err = match kind {
                LoopKind::For { iter, .. } => check_linear_image_results_in_term(
                    iter,
                    program,
                    model,
                    LinearImageUseContext::Discarded,
                ),
                LoopKind::ForRange { bound, .. } => check_linear_image_results_in_term(
                    bound,
                    program,
                    model,
                    LinearImageUseContext::Discarded,
                ),
                LoopKind::While { cond } => check_linear_image_results_in_term(
                    cond,
                    program,
                    model,
                    LinearImageUseContext::Discarded,
                ),
            };
            kind_err.or_else(|| {
                check_linear_image_results_in_term(body, program, model, LinearImageUseContext::Used)
            })
        }
        TermKind::App { func, args } => {
            if let Some(err) =
                check_linear_image_results_in_term(func, program, model, LinearImageUseContext::Discarded)
            {
                return Some(err);
            }
            for (index, arg) in args.iter().enumerate() {
                let arg_context = if (is_image_load_app(term, program) || is_image_with_app(term, program))
                    && index == 0
                {
                    LinearImageUseContext::Used
                } else if matches!(arg.ty, Type::Constructed(TypeName::StorageTexture, _)) {
                    LinearImageUseContext::Used
                } else {
                    LinearImageUseContext::Discarded
                };
                if let Some(err) = check_linear_image_results_in_term(arg, program, model, arg_context) {
                    return Some(err);
                }
            }
            None
        }
        TermKind::Tuple(parts) | TermKind::VecLit(parts) => {
            for part in parts {
                let part_context = if matches!(part.ty, Type::Constructed(TypeName::StorageTexture, _)) {
                    context
                } else {
                    LinearImageUseContext::Discarded
                };
                if let Some(err) = check_linear_image_results_in_term(part, program, model, part_context) {
                    return Some(err);
                }
            }
            None
        }
        TermKind::TupleProj { tuple, idx } => {
            if let TermKind::Tuple(parts) = &tuple.kind {
                for (i, part) in parts.iter().enumerate() {
                    let part_context = if i == *idx { context } else { LinearImageUseContext::Discarded };
                    if let Some(err) =
                        check_linear_image_results_in_term(part, program, model, part_context)
                    {
                        return Some(err);
                    }
                }
                None
            } else {
                check_linear_image_results_in_term(tuple, program, model, context)
            }
        }
        TermKind::Index { array, index } => {
            check_linear_image_results_in_term(array, program, model, LinearImageUseContext::Discarded)
                .or_else(|| {
                    check_linear_image_results_in_term(
                        index,
                        program,
                        model,
                        LinearImageUseContext::Discarded,
                    )
                })
        }
        TermKind::Coerce { inner, .. } => {
            check_linear_image_results_in_term(inner, program, model, context)
        }
        TermKind::ArrayExpr(ae) => check_linear_image_results_in_array_expr(ae, program, model),
        TermKind::Soac(op) => check_linear_image_results_in_soac(op, program, model),
        TermKind::OutputSlotStore { value, .. } => {
            check_linear_image_results_in_term(value, program, model, LinearImageUseContext::Discarded)
        }
        TermKind::Var(_)
        | TermKind::IntLit(_)
        | TermKind::FloatLit(_)
        | TermKind::BoolLit(_)
        | TermKind::UnitLit
        | TermKind::BinOp(_)
        | TermKind::UnOp(_)
        | TermKind::Extern(_) => None,
    }
}

fn check_linear_image_results_in_array_expr(
    ae: &ArrayExpr,
    program: &Program,
    model: &OwnershipModel,
) -> Option<CompilerError> {
    match ae {
        ArrayExpr::Literal(terms) => {
            for term in terms {
                if let Some(err) = check_linear_image_results_in_term(
                    term,
                    program,
                    model,
                    LinearImageUseContext::Discarded,
                ) {
                    return Some(err);
                }
            }
            None
        }
        ArrayExpr::Range { start, len, step } => {
            check_linear_image_results_in_term(start, program, model, LinearImageUseContext::Discarded)
                .or_else(|| {
                    check_linear_image_results_in_term(
                        len,
                        program,
                        model,
                        LinearImageUseContext::Discarded,
                    )
                })
                .or_else(|| {
                    step.as_ref().and_then(|s| {
                        check_linear_image_results_in_term(
                            s,
                            program,
                            model,
                            LinearImageUseContext::Discarded,
                        )
                    })
                })
        }
        ArrayExpr::StorageView(crate::tlc::StorageView { offset, len, .. }) => {
            check_linear_image_results_in_term(offset, program, model, LinearImageUseContext::Discarded)
                .or_else(|| {
                    check_linear_image_results_in_term(
                        len,
                        program,
                        model,
                        LinearImageUseContext::Discarded,
                    )
                })
        }
        ArrayExpr::Zip(parts) => {
            for part in parts {
                if let Some(err) = check_linear_image_results_in_array_expr(part, program, model) {
                    return Some(err);
                }
            }
            None
        }
        ArrayExpr::Var(..) => None,
    }
}

fn check_linear_image_results_in_soac(
    op: &SoacOp,
    program: &Program,
    model: &OwnershipModel,
) -> Option<CompilerError> {
    let check_body = |sb: &super::SoacBody| {
        for (_, _, capture) in &sb.captures {
            if let Some(err) = check_linear_image_results_in_term(
                capture,
                program,
                model,
                LinearImageUseContext::Discarded,
            ) {
                return Some(err);
            }
        }
        check_linear_image_results_in_term(&sb.lam.body, program, model, LinearImageUseContext::Used)
    };
    match op {
        SoacOp::Map { lam, inputs, .. } => {
            for input in inputs {
                if let Some(err) = check_linear_image_results_in_array_expr(input, program, model) {
                    return Some(err);
                }
            }
            check_body(lam)
        }
        SoacOp::Reduce { op, ne, input } => {
            check_linear_image_results_in_term(ne, program, model, LinearImageUseContext::Discarded)
                .or_else(|| check_linear_image_results_in_array_expr(input, program, model))
                .or_else(|| check_body(op))
        }
        SoacOp::Scan { op, ne, input, .. } => {
            check_linear_image_results_in_term(ne, program, model, LinearImageUseContext::Discarded)
                .or_else(|| check_linear_image_results_in_array_expr(input, program, model))
                .or_else(|| check_body(op))
        }
        SoacOp::Filter {
            map_lam, pred, input, ..
        } => check_linear_image_results_in_array_expr(input, program, model)
            .or_else(|| map_lam.as_ref().and_then(|m| check_body(m)))
            .or_else(|| check_body(pred)),
        SoacOp::Scatter { lam, inputs, .. } => {
            for input in inputs {
                if let Some(err) = check_linear_image_results_in_array_expr(input, program, model) {
                    return Some(err);
                }
            }
            check_body(lam)
        }
        SoacOp::ReduceByIndex {
            op,
            ne,
            indices,
            values,
            ..
        } => check_linear_image_results_in_term(ne, program, model, LinearImageUseContext::Discarded)
            .or_else(|| check_linear_image_results_in_array_expr(indices, program, model))
            .or_else(|| check_linear_image_results_in_array_expr(values, program, model))
            .or_else(|| check_body(op)),
        SoacOp::Screma {
            lanes,
            accumulators,
            inputs,
        } => {
            for input in inputs {
                if let Some(err) = check_linear_image_results_in_array_expr(input, program, model) {
                    return Some(err);
                }
            }
            for lane in lanes {
                if let Some(err) = check_body(&lane.lam) {
                    return Some(err);
                }
            }
            for acc in accumulators {
                if let Some(err) = check_linear_image_results_in_term(
                    &acc.ne,
                    program,
                    model,
                    LinearImageUseContext::Discarded,
                ) {
                    return Some(err);
                }
                if let Some(err) = check_body(&acc.step_lam) {
                    return Some(err);
                }
                if let Some(err) = check_body(&acc.reduce_op) {
                    return Some(err);
                }
            }
            None
        }
    }
}

fn linear_image_owner_is_threaded(
    term: &Term,
    owners: &LookupSet<OwnerId>,
    model: &OwnershipModel,
) -> bool {
    if term_consumes_owner_on_all_paths(term, owners, model) {
        return true;
    }
    if term_observes_owner_on_all_paths(term, owners, model) {
        return true;
    }
    term_returns_owner_on_all_paths(term, owners, model)
}

fn owner_sets_intersect(a: &LookupSet<OwnerId>, b: &LookupSet<OwnerId>) -> bool {
    a.iter().any(|owner| b.contains(owner))
}

fn term_kills_owner(term: &Term, owners: &LookupSet<OwnerId>, model: &OwnershipModel) -> bool {
    model.kills.get(&term.id).is_some_and(|kills| owner_sets_intersect(kills, owners))
}

fn term_consumes_owner_on_all_paths(
    term: &Term,
    owners: &LookupSet<OwnerId>,
    model: &OwnershipModel,
) -> bool {
    if term_kills_owner(term, owners, model) {
        return true;
    }
    match &term.kind {
        TermKind::Let { rhs, body, .. } => {
            term_consumes_owner_on_all_paths(rhs, owners, model)
                || term_consumes_owner_on_all_paths(body, owners, model)
        }
        TermKind::If {
            then_branch,
            else_branch,
            ..
        } => {
            term_consumes_owner_on_all_paths(then_branch, owners, model)
                && term_consumes_owner_on_all_paths(else_branch, owners, model)
        }
        TermKind::Coerce { inner, .. } => term_consumes_owner_on_all_paths(inner, owners, model),
        _ => false,
    }
}

fn term_observes_owner_on_all_paths(
    term: &Term,
    owners: &LookupSet<OwnerId>,
    model: &OwnershipModel,
) -> bool {
    match &term.kind {
        TermKind::App { func, args } => {
            let observes_here = matches!(&func.kind, TermKind::Var(VarRef::Builtin { id, .. }) if *id == catalog().known().image_load)
                && args.first().is_some_and(|arg| owner_sets_intersect(&arg_aliases(arg, model), owners));
            observes_here || args.iter().any(|arg| term_observes_owner_on_all_paths(arg, owners, model))
        }
        TermKind::Let { rhs, body, .. } => {
            if term_consumes_owner_on_all_paths(rhs, owners, model) {
                false
            } else {
                term_observes_owner_on_all_paths(rhs, owners, model)
                    || term_observes_owner_on_all_paths(body, owners, model)
            }
        }
        TermKind::If {
            then_branch,
            else_branch,
            ..
        } => {
            term_observes_owner_on_all_paths(then_branch, owners, model)
                && term_observes_owner_on_all_paths(else_branch, owners, model)
        }
        TermKind::Coerce { inner, .. } => term_observes_owner_on_all_paths(inner, owners, model),
        _ => false,
    }
}

fn arg_aliases(term: &Term, model: &OwnershipModel) -> LookupSet<OwnerId> {
    match &term.kind {
        TermKind::Var(VarRef::Symbol(sym)) => model.aliases_of(*sym),
        TermKind::Coerce { inner, .. } => arg_aliases(inner, model),
        _ => LookupSet::new(),
    }
}

fn term_returns_owner_on_all_paths(
    term: &Term,
    owners: &LookupSet<OwnerId>,
    model: &OwnershipModel,
) -> bool {
    match &term.kind {
        TermKind::Var(VarRef::Symbol(sym)) => owner_sets_intersect(&model.aliases_of(*sym), owners),
        TermKind::Coerce { inner, .. } => term_returns_owner_on_all_paths(inner, owners, model),
        TermKind::Let { rhs, body, .. } => {
            if term_consumes_owner_on_all_paths(rhs, owners, model) {
                false
            } else {
                term_returns_owner_on_all_paths(body, owners, model)
            }
        }
        TermKind::If {
            then_branch,
            else_branch,
            ..
        } => {
            term_returns_owner_on_all_paths(then_branch, owners, model)
                && term_returns_owner_on_all_paths(else_branch, owners, model)
        }
        _ => false,
    }
}

struct Rewriter<'m> {
    model: &'m OwnershipModel,
    program: &'m Program,
    consuming_soacs: &'m LookupSet<TermId>,
}

impl<'m> Rewriter<'m> {
    fn rewrite(&mut self, term: Term) -> Term {
        // array_with → array_with_inplace: rewrite the App's `func`
        // before descending, since the rewrite swaps the function var.
        if let TermKind::App { func, args } = &term.kind {
            let known = catalog().known();
            let calls_functional =
                var_term_builtin_id(func, &self.program.symbols) == Some(known.array_with);
            if calls_functional && args.len() == 3 && self.is_promotable(term.id, &args[0]) {
                let inplace_id = known.array_with_in_place;
                let TermKind::App { func, args } = term.kind else {
                    unreachable!()
                };
                let new_func = Term {
                    kind: TermKind::Var(VarRef::Builtin {
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
            match &mut rewritten.kind {
                TermKind::Soac(SoacOp::Map { destination, .. })
                | TermKind::Soac(SoacOp::Scan { destination, .. })
                | TermKind::Soac(SoacOp::Filter { destination, .. }) => {
                    *destination = SoacDestination::InputBuffer;
                }
                _ => {}
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
        walk_for_eligible_soacs(&def.body, model, program, &entry_output_soacs, &mut out);
    }
    out
}

fn collect_entry_output_soac_ids(program: &Program) -> LookupSet<TermId> {
    let mut out = LookupSet::new();
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

fn collect_tail_soac_ids(term: &Term, out: &mut LookupSet<TermId>) {
    match &term.kind {
        TermKind::Soac(_) => {
            out.insert(term.id);
        }
        TermKind::Let { body, .. } => collect_tail_soac_ids(body, out),
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

fn walk_for_eligible_soacs(
    term: &Term,
    model: &OwnershipModel,
    program: &Program,
    entry_output_soacs: &LookupSet<TermId>,
    out: &mut Vec<TermId>,
) {
    match &term.kind {
        TermKind::Soac(SoacOp::Map { lam, inputs, .. }) => {
            // Multi-input map isn't eligible (the body reads parallel
            // streams; the consume rewrite would only own one of them).
            if inputs.len() == 1 && !entry_output_soacs.contains(&term.id) {
                if let Some(input_sym) = input_is_dead_unique_var(term.id, &inputs[0], model) {
                    if map_body_ok(&lam.lam) && !body_references_sym(&lam.lam.body, input_sym) {
                        out.push(term.id);
                    }
                }
            }
        }
        TermKind::Soac(SoacOp::Scan { op, input, .. }) => {
            // No tail-of-compute-entry exclusion. The Screma scan path
            // in egir::parallelize knows how to consume an `InputBuffer`
            // destination: phase 1 and phase 3 route their writes back
            // to the input binding, and the pipeline descriptor skips
            // the auto-output slot.
            if let Some(input_sym) = input_is_dead_unique_var(term.id, input, model) {
                if scan_body_ok(&op.lam) && !body_references_sym(&op.lam.body, input_sym) {
                    out.push(term.id);
                }
            }
        }
        TermKind::Soac(SoacOp::Filter {
            map_lam, pred, input, ..
        }) => {
            // A fused producer map can change the element type (so `f(x)` no
            // longer fits the input slot) and may read the input array at other
            // indices, so reusing the input buffer in place is only sound for a
            // plain filter.
            if map_lam.is_none() && !entry_output_soacs.contains(&term.id) {
                if let Some(input_sym) = input_is_dead_unique_var(term.id, input, model) {
                    if filter_body_ok(&pred.lam) && !body_references_sym(&pred.lam.body, input_sym) {
                        out.push(term.id);
                    }
                }
            }
        }
        _ => {}
    }
    term.for_each_child(&mut |child| {
        walk_for_eligible_soacs(child, model, program, entry_output_soacs, out)
    });
}

/// Shared input-side eligibility check: returns the input's
/// SymbolId if the SOAC's input is a single `Var` reference whose
/// owner is mutable and absent from `live_out`. Each caller applies
/// the entry-output exclusion separately where applicable and the
/// SOAC-specific body-shape and pointwise checks.
fn input_is_dead_unique_var(
    soac_id: TermId,
    input: &ArrayExpr,
    model: &OwnershipModel,
) -> Option<SymbolId> {
    let input_sym = match input {
        ArrayExpr::Var(VarRef::Symbol(s), ty) => {
            // In-place consumption writes the result over the input's buffer.
            // A Virtual array (a range / `iota`) has no buffer to write into,
            // so a map over one must allocate a Fresh result rather than be
            // marked consuming — otherwise the backend has no buffer to retarget.
            if matches!(
                ty.array_variant(),
                Some(Type::Constructed(TypeName::ArrayVariantVirtual, _))
            ) {
                return None;
            }
            *s
        }
        _ => return None,
    };
    let owner = model.owner_of(input_sym)?;
    let origin = model.origin(owner)?;
    if !origin.is_mutable() {
        return None;
    }
    let live_out = model.live_out.get(&soac_id)?;
    if live_out.contains(&owner) {
        return None;
    }
    Some(input_sym)
}

/// Map's body shape: single param whose type matches the lambda's
/// return — so the per-iteration write fits back into the input's
/// element slot.
fn map_body_ok(lam: &Lambda) -> bool {
    lam.params.len() == 1 && lam.params[0].1 == lam.ret_ty
}

/// Scan's body shape: `|acc, elem| _` where the elem-param type
/// matches the lambda's return (= accumulator type = element type).
fn scan_body_ok(lam: &Lambda) -> bool {
    lam.params.len() == 2 && lam.params[1].1 == lam.ret_ty
}

/// Filter's body shape: single param, returns `bool`. The pred's
/// param type already matches the input's element type by
/// type-checking; we just confirm the boolean return.
fn filter_body_ok(lam: &Lambda) -> bool {
    lam.params.len() == 1 && matches!(&lam.ret_ty, Type::Constructed(TypeName::Bool, _))
}

fn body_references_sym(term: &Term, sym: SymbolId) -> bool {
    if let TermKind::Var(VarRef::Symbol(s)) = &term.kind {
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
