//! Representation specialization (Phase 2 of the array-variant-abstract
//! project).
//!
//! `ArrayVariantAbstract` is a representation-polymorphic variant
//! introduced for `filter`'s existential return. Phase 1 made it
//! survive monomorphize verbatim and added a backend-boundary verifier
//! (`egir::verify_no_abstract`). The verifier surfaced the canonical
//! failure mode: when a user-defined size-poly helper does **not**
//! inline (e.g. `center(filter_result)` where `center` is small but
//! contains an inner call), monomorphize specializes the callee for
//! `Abstract` and that signature leaks into SSA. Backends can't lower
//! representation-polymorphic SSA functions.
//!
//! This pass closes the gap. At each `App(Var(callee), args)` where an
//! arg is a let-bound value whose producer's concrete representation
//! is statically known, we clone the callee, substitute `Abstract` →
//! producer's concrete variant in the matching param's type (and
//! everywhere downstream in the body), recursively rewrite the cloned
//! body so nested calls specialize too, and rewrite the original call
//! to invoke the specialization. Caches per `(orig_sym, spec_key)`.
//!
//! Mirrors `tlc::buffer_specialize` (per-buffer specialization) and
//! `tlc::hof_specialize` (per-callable specialization) — same shape,
//! representation axis.
//!
//! Producer recognition: `filter(pred, arr)` produces Bounded when
//! `arr` has a static `Size(_)` size, View otherwise. Other producers
//! (partition, take_while if/when added) are out of scope; the
//! verifier will flag them.

use super::{Def, DefMeta, Lambda, Program, SoacOp, Term, TermIdSource, TermKind, VarRef};
use crate::ast::TypeName;
use crate::tlc::ArrayExpr;
use crate::LookupMap;
use crate::{SymbolId, SymbolTable};
use polytype::Type;

/// Concrete array representation variant chosen by a known producer.
/// `Bounded` carries the producer's static capacity — the consumer's
/// param type also gets its (Skolem) size slot rewritten to that
/// `Size(N)`, because Bounded values require a static capacity at the
/// backend. `View` has a runtime length and leaves the size slot alone.
/// `Composite` and `Virtual` round out the lattice for future
/// producers (none today).
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
#[allow(dead_code)]
enum ConcreteVariant {
    Bounded {
        capacity: usize,
    },
    View,
    Composite,
    Virtual,
}

impl ConcreteVariant {
    fn variant_type(self) -> Type<TypeName> {
        let name = match self {
            ConcreteVariant::Bounded { .. } => TypeName::ArrayVariantBounded,
            ConcreteVariant::View => TypeName::ArrayVariantView,
            ConcreteVariant::Composite => TypeName::ArrayVariantComposite,
            ConcreteVariant::Virtual => TypeName::ArrayVariantVirtual,
        };
        Type::Constructed(name, vec![])
    }

    /// Concrete size to substitute into the param type, when the
    /// variant constrains it. `Bounded` needs `Size(capacity)` —
    /// without it the backend can't lay the buffer out. `View` and
    /// the others leave the size slot alone (runtime length).
    fn size_type(self) -> Option<Type<TypeName>> {
        match self {
            ConcreteVariant::Bounded { capacity } => {
                Some(Type::Constructed(TypeName::Size(capacity), vec![]))
            }
            _ => None,
        }
    }

    fn key_str(self) -> String {
        match self {
            ConcreteVariant::Bounded { capacity } => format!("bounded{}", capacity),
            ConcreteVariant::View => "view".to_string(),
            ConcreteVariant::Composite => "composite".to_string(),
            ConcreteVariant::Virtual => "virtual".to_string(),
        }
    }
}

type SpecKey = Vec<Option<ConcreteVariant>>;

struct RepSpecializer<'a, 'ids> {
    symbols: &'a mut SymbolTable,
    term_ids: &'ids mut TermIdSource,
    def_map: LookupMap<SymbolId, Def>,
    /// Cache: `(orig_callee_sym, spec_key) → specialized_sym`.
    specializations: LookupMap<(SymbolId, SpecKey), SymbolId>,
    /// Newly generated specialized defs, appended to the program.
    new_defs: Vec<Def>,
    /// Scoped per-def env. Stack of frames pushed/popped at every Let
    /// (and at every nested Lambda — captures don't leak across defs).
    /// Each frame maps a bound symbol to the producer-derived variant.
    env_stack: Vec<LookupMap<SymbolId, ConcreteVariant>>,
}

/// Entry point: rewrite every def's body in place to insert call-edge
/// representation specializations, appending the new specialized defs
/// to the program.
pub fn run(program: &mut Program, term_ids: &mut TermIdSource) {
    // Snapshot of the original defs: specialization clones a callee's
    // body while the defs themselves are being rewritten.
    let def_map: LookupMap<SymbolId, Def> = program.defs.iter().map(|d| (d.name, d.clone())).collect();
    let mut s = RepSpecializer {
        symbols: &mut program.symbols,
        term_ids,
        def_map,
        specializations: LookupMap::new(),
        new_defs: Vec::new(),
        env_stack: Vec::new(),
    };

    for def in &mut program.defs {
        s.rewrite_def_body(&mut def.body);
    }
    program.defs.append(&mut s.new_defs);
}

impl<'a, 'ids> RepSpecializer<'a, 'ids> {
    // ------------------------------------------------------------------
    // Scoped producer env
    // ------------------------------------------------------------------

    fn push_scope(&mut self) {
        self.env_stack.push(LookupMap::new());
    }

    fn pop_scope(&mut self) {
        self.env_stack.pop();
    }

    fn bind(&mut self, sym: SymbolId, variant: ConcreteVariant) {
        if let Some(frame) = self.env_stack.last_mut() {
            frame.insert(sym, variant);
        }
    }

    fn lookup(&self, sym: SymbolId) -> Option<ConcreteVariant> {
        self.env_stack.iter().rev().find_map(|frame| frame.get(&sym).copied())
    }

    // ------------------------------------------------------------------
    // Driver
    // ------------------------------------------------------------------

    /// Preserve the outer parameter-spine `Lambda` nodes; rewrite their
    /// inner body. A fresh scope is pushed for the def's body.
    fn rewrite_def_body(&mut self, term: &mut Term) {
        self.push_scope();
        self.rewrite_def_body_inner(term);
        self.pop_scope();
    }

    fn rewrite_def_body_inner(&mut self, term: &mut Term) {
        match &mut term.kind {
            TermKind::Lambda(Lambda { body, .. }) => {
                self.rewrite_def_body_inner(body);
            }
            _ => {
                self.rewrite_term(term);
                return;
            }
        }
        term.id = self.term_ids.next_id();
    }

    /// Recursive in-place term walker. Two hooks fire:
    ///   * `Let { rhs = filter(_, arr), ... }` — record the bound name
    ///     as a known producer in the current scope (Bounded/View based
    ///     on `arr`'s size).
    ///   * `App { func = Var(Symbol(callee_sym)), args }` — if any arg
    ///     is a Var of a known-producer name and the callee's param
    ///     has `Abstract` at that position, specialize the callee and
    ///     rewrite the call site.
    fn rewrite_term(&mut self, term: &mut Term) {
        match &mut term.kind {
            TermKind::Lambda(lam) => {
                // Nested lambdas open a fresh scope — captures are
                // tracked by the surrounding Let.
                self.push_scope();
                self.rewrite_term(&mut lam.body);
                self.pop_scope();
            }
            TermKind::Let { name, rhs, body, .. } => {
                self.rewrite_term(rhs);
                let producer_variant = self.detect_producer_variant(rhs);
                self.push_scope();
                if let Some(v) = producer_variant {
                    self.bind(*name, v);
                }
                self.rewrite_term(body);
                self.pop_scope();
            }
            TermKind::Soac(soac) => self.rewrite_soac(soac),
            _ => {
                term.for_each_child_mut(&mut |child| self.rewrite_term(child));
            }
        }
        term.id = self.term_ids.next_id();
        self.maybe_specialize_call(term);
    }

    fn rewrite_soac(&mut self, soac: &mut SoacOp) {
        match soac {
            SoacOp::Map { lam, inputs, .. } => {
                self.rewrite_soac_body(lam);
                for ae in inputs {
                    self.rewrite_array_expr(ae);
                }
            }
            SoacOp::Reduce { op, ne, input } => {
                self.rewrite_soac_body(op);
                self.rewrite_term(ne);
                self.rewrite_array_expr(input);
            }
            SoacOp::Scan {
                op,
                reduce_op,
                ne,
                input,
                ..
            } => {
                self.rewrite_soac_body(op);
                self.rewrite_soac_body(reduce_op);
                self.rewrite_term(ne);
                self.rewrite_array_expr(input);
            }
            SoacOp::Filter {
                map_lam, pred, input, ..
            } => {
                if let Some(ml) = map_lam {
                    self.rewrite_soac_body(ml);
                }
                self.rewrite_soac_body(pred);
                self.rewrite_array_expr(input);
            }
            SoacOp::Scatter { lam, inputs, .. } => {
                self.rewrite_soac_body(lam);
                for ae in inputs {
                    self.rewrite_array_expr(ae);
                }
            }
            SoacOp::ReduceByIndex {
                op,
                ne,
                indices,
                values,
                ..
            } => {
                self.rewrite_soac_body(op);
                self.rewrite_term(ne);
                self.rewrite_array_expr(indices);
                self.rewrite_array_expr(values);
            }
            SoacOp::Screma {
                lanes,
                accumulators,
                inputs,
            } => {
                for lane in lanes {
                    self.rewrite_soac_body(&mut lane.lam);
                }
                for acc in accumulators {
                    self.rewrite_soac_body(&mut acc.step_lam);
                    self.rewrite_soac_body(&mut acc.reduce_op);
                    self.rewrite_term(&mut acc.ne);
                }
                for ae in inputs {
                    self.rewrite_array_expr(ae);
                }
            }
        }
    }

    fn rewrite_soac_body(&mut self, sb: &mut super::SoacBody) {
        self.push_scope();
        self.rewrite_term(&mut sb.lam.body);
        self.pop_scope();
        for (_, _, e) in &mut sb.captures {
            self.rewrite_term(e);
        }
    }

    fn rewrite_array_expr(&mut self, ae: &mut ArrayExpr) {
        match ae {
            ArrayExpr::Var(..) => {}
            ArrayExpr::Zip(aes) => {
                for a in aes {
                    self.rewrite_array_expr(a);
                }
            }
            ArrayExpr::Literal(terms) => {
                for t in terms {
                    self.rewrite_term(t);
                }
            }
            ArrayExpr::Range { start, len, step } => {
                self.rewrite_term(start);
                self.rewrite_term(len);
                if let Some(s) = step {
                    self.rewrite_term(s);
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Producer recognition
    // ------------------------------------------------------------------

    /// Recognise let-bound producers whose concrete representation is
    /// derivable from the TLC type at this point. Today, only
    /// `SoacOp::Filter` — and `filter` is a SOAC at TLC level
    /// (`transform_soac_filter` lowers the surface call). Producer
    /// rule:
    ///   * input array size statically `Size(_)` ⇒ Bounded.
    ///   * anything else (Variable / Skolem / SizeVar) ⇒ View.
    fn detect_producer_variant(&self, rhs: &Term) -> Option<ConcreteVariant> {
        match &rhs.kind {
            TermKind::Soac(SoacOp::Filter { input, .. }) => {
                let input_ty = input.array_type();
                let size = array_size(&input_ty)?;
                Some(match size {
                    Type::Constructed(TypeName::Size(n), _) => ConcreteVariant::Bounded { capacity: *n },
                    _ => ConcreteVariant::View,
                })
            }
            // `let arr = Var(other)` aliases — propagate the source's
            // variant. Specifically, `open_existential` lowers
            // `let kept: ?k.[k]T = filter(...)` to `let _anf = filter(...);
            // let kept = _anf`, where `kept` and `_anf` are distinct
            // SymbolIds but `kept` has the open-existential type.
            // Without this we'd lose the variant at the open site.
            TermKind::Var(VarRef::Symbol(other_sym)) => self.lookup(*other_sym),
            _ => None,
        }
    }

    // ------------------------------------------------------------------
    // Call-site dispatch
    // ------------------------------------------------------------------

    /// Call-site hook, applied after a term's children are rewritten:
    /// if this is an `App(Var(Symbol), args)` whose args match known
    /// producer variants at representation-polymorphic callee params,
    /// specialize the callee and repoint `func` at the specialization.
    fn maybe_specialize_call(&mut self, term: &mut Term) {
        let span = term.span;
        let TermKind::App { func, args } = &mut term.kind else {
            return;
        };
        let TermKind::Var(VarRef::Symbol(callee_sym)) = &func.kind else {
            return;
        };
        let callee_sym = *callee_sym;

        // Build a positional spec key. None at every position with no
        // known concrete variant; Some(variant) when the arg is a `Var`
        // of a binding currently in the producer env.
        let spec_key: SpecKey = args
            .iter()
            .map(|a| match &a.kind {
                TermKind::Var(VarRef::Symbol(s)) => self.lookup(*s),
                _ => None,
            })
            .collect();
        if spec_key.iter().all(Option::is_none) {
            return;
        }

        // Check the callee actually has a representation-polymorphic array
        // at one of the matched positions; otherwise it's already concrete
        // for this arg, and there's nothing to specialize.
        let Some(callee_def) = self.def_map.get(&callee_sym).cloned() else {
            return;
        };
        if !matches!(callee_def.meta, DefMeta::Function | DefMeta::LiftedLambda) {
            return;
        }
        let (params, _body) = super::extract_lambda_params(&callee_def.body);
        let mut effective_key = spec_key;
        for (i, slot) in effective_key.iter_mut().enumerate() {
            if slot.is_some() {
                let param_ty = match params.get(i) {
                    Some((_, t)) => t,
                    None => {
                        *slot = None;
                        continue;
                    }
                };
                if !type_has_specializable_array_variant(param_ty) {
                    *slot = None;
                }
            }
        }
        if effective_key.iter().all(Option::is_none) {
            return;
        }

        let spec_sym = self.get_or_create_specialization(callee_sym, &callee_def, &effective_key);

        let func_ty = match self.def_map.get(&spec_sym) {
            Some(d) => d.ty.clone(),
            None => self
                .new_defs
                .iter()
                .find(|d| d.name == spec_sym)
                .map(|d| d.ty.clone())
                .unwrap_or(func.ty.clone()),
        };
        let TermKind::App { func, .. } = &mut term.kind else {
            unreachable!()
        };
        **func = Term {
            id: self.term_ids.next_id(),
            ty: func_ty,
            span,
            kind: TermKind::Var(VarRef::Symbol(spec_sym)),
        };
    }

    // ------------------------------------------------------------------
    // Specialization
    // ------------------------------------------------------------------

    fn get_or_create_specialization(
        &mut self,
        callee_sym: SymbolId,
        callee_def: &Def,
        spec_key: &SpecKey,
    ) -> SymbolId {
        if let Some(&existing) = self.specializations.get(&(callee_sym, spec_key.clone())) {
            return existing;
        }

        let orig_name =
            self.symbols.get(callee_sym).cloned().unwrap_or_else(|| format!("def_{:?}", callee_sym));
        let suffix: String = spec_key
            .iter()
            .enumerate()
            .filter_map(|(i, v)| v.map(|v| format!("_p{}{}", i, v.key_str())))
            .collect();
        let spec_name = format!("{}_rep{}", orig_name, suffix);
        let spec_sym = self.symbols.alloc(spec_name);

        // Insert the cache entry BEFORE recursive rewrite so a
        // recursive call to the same `(callee, spec_key)` resolves to
        // the in-progress spec sym instead of looping.
        self.specializations.insert((callee_sym, spec_key.clone()), spec_sym);

        // Walk the callee's parameter spine, rewriting param types at
        // matched positions, then recursively rewrite the inner body
        // for nested call-edge specialization.
        let new_body = self.specialize_def_body(&callee_def.body, spec_key);

        // `rebuild_nested_lam` (called by `specialize_def_body`) wraps
        // the inner body in `Lambda` nodes and computes the outer term's
        // type as the curried arrow over the rewritten params + the
        // inner body's return type. That outer type IS the function
        // type — do NOT clone the original `callee_def.ty`, which still
        // carries `ArrayVariantAbstract` in the matched param's slot.
        // Mirroring the wrong type into the new def would leak Abstract
        // back into downstream consumers that read `Def.ty` (call-site
        // synthesis, EGIR lowering).
        let new_def = Def {
            name: spec_sym,
            ty: new_body.ty.clone(),
            body: new_body,
            meta: DefMeta::Function,
            arity: callee_def.arity,
            param_diets: callee_def.param_diets.clone(),
            return_diet: callee_def.return_diet.clone(),
        };
        self.new_defs.push(new_def);
        spec_sym
    }

    /// Walk a callee's param-spine `Lambda`s and rewrite the matched
    /// param types; then recurse into the inner body. Inner body walk
    /// uses `rewrite_def_body` so nested calls trigger further
    /// specialization — and the new-variant params get pushed into the
    /// env so calls inside the body that use those params as args
    /// also see their concrete variant.
    fn specialize_def_body(&mut self, body: &Term, spec_key: &SpecKey) -> Term {
        let (params, mut inner) = super::extract_lambda_params(body);
        let new_params: Vec<(SymbolId, Type<TypeName>)> = params
            .iter()
            .enumerate()
            .map(|(i, (s, ty))| {
                let new_ty = match spec_key.get(i).copied().flatten() {
                    Some(variant) => substitute_specializable_variant_in_type(ty, variant),
                    None => ty.clone(),
                };
                (*s, new_ty)
            })
            .collect();

        // Bind the specialized params into a fresh env scope so a
        // recursive `App(Var(callee), [Var(param)])` inside the body
        // sees `param` as a known-variant value and triggers further
        // specialization.
        self.push_scope();
        for (i, (sym, _)) in params.iter().enumerate() {
            if let Some(Some(variant)) = spec_key.get(i) {
                self.bind(*sym, *variant);
            }
        }
        self.rewrite_def_body(&mut inner);
        self.pop_scope();

        super::closure_convert::rebuild_nested_lam(&new_params, inner, body.span, &mut self.term_ids)
    }
}

// ----------------------------------------------------------------------
// Type-level helpers
// ----------------------------------------------------------------------

fn array_size(ty: &Type<TypeName>) -> Option<&Type<TypeName>> {
    if let Type::Constructed(TypeName::Array, args) = ty {
        return args.get(2);
    }
    None
}

/// Best-effort array-type extraction for the shapes a `SoacOp::Filter`
/// input can take. This pass cares about `Ref` (a bound name with an
/// Array-typed term). Other variants (Soac, Zip, Literal, Range) appear in
/// fused chains; we don't need them for the simple producer-detection
/// case, so return `None` — the call site falls through to "no
/// producer-derived variant".
/// Return `true` if any representation-polymorphic array variant appears
/// anywhere in the type tree. `ArrayVariantAbstract` comes from filter's
/// existential result; a `Type::Variable` in the variant slot comes from a
/// plain representation-polymorphic helper such as `def sum(arr: [n]i32)`.
fn type_has_specializable_array_variant(ty: &Type<TypeName>) -> bool {
    match ty {
        Type::Variable(_) => false,
        Type::Constructed(TypeName::Array, args) if args.len() >= 4 => {
            matches!(
                &args[1],
                Type::Constructed(TypeName::ArrayVariantAbstract, _) | Type::Variable(_)
            ) || args.iter().any(type_has_specializable_array_variant)
        }
        Type::Constructed(_, args) => args.iter().any(type_has_specializable_array_variant),
    }
}

/// Substitute every representation-polymorphic array variant in any `Array`
/// subtree with the chosen concrete variant. For variants that constrain the
/// size (currently only `Bounded`, which needs a static `Size(N)` capacity),
/// also rewrite the size slot to the producer's concrete size when the
/// existing slot is a non-literal (Skolem / Variable / SizeVar). This is what
/// makes the consumer's specialized signature match the runtime layout the
/// producer's EGIR lowering emits.
fn substitute_specializable_variant_in_type(
    ty: &Type<TypeName>,
    target: ConcreteVariant,
) -> Type<TypeName> {
    match ty {
        Type::Variable(_) => ty.clone(),
        Type::Constructed(TypeName::Array, args) if args.len() >= 4 => {
            let mut new_args: Vec<Type<TypeName>> =
                args.iter().map(|a| substitute_specializable_variant_in_type(a, target)).collect();
            if matches!(
                &args[1],
                Type::Constructed(TypeName::ArrayVariantAbstract, _) | Type::Variable(_)
            ) {
                new_args[1] = target.variant_type();
                if let Some(size_ty) = target.size_type() {
                    // Only rewrite the size when the current slot is
                    // non-literal — a Size(N) already there is the
                    // user's intent and stays.
                    if !matches!(&new_args[2], Type::Constructed(TypeName::Size(_), _)) {
                        new_args[2] = size_ty;
                    }
                }
            }
            Type::Constructed(TypeName::Array, new_args)
        }
        Type::Constructed(name, args) => Type::Constructed(
            name.clone(),
            args.iter().map(|a| substitute_specializable_variant_in_type(a, target)).collect(),
        ),
    }
}
