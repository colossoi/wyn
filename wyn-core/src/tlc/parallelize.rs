//! TLC-level SOAC parallelization.
//!
//! Stage A: Analyze compute entry points to find parallelizable SOACs.
//! Stage B: Restructure the program — create new entry points with chunked SOACs,
//!          allocate intermediate storage buffers, build pipeline descriptor.
//!
//! Loop creation and storage lowering stay in SSA (`to_ssa` + `soac_lower`).

use super::closure_convert::{collect_free_vars, compute_free_vars};
use super::VarRef;
use crate::ast::{self, TypeName};
use crate::builtins::catalog;
use crate::egir::from_tlc::AUTO_STORAGE_SET;
use crate::interface::{self, Attribute, EntryParamBinding, EntryParamBindingKind};
use crate::pipeline_descriptor::*;
use crate::types::TypeExt;
use crate::StableMap;
use crate::{BindingRef, SymbolId, SymbolTable};
use crate::{LookupMap, LookupSet};
use polytype::Type;

use super::{
    ArrayExpr, Def, DefMeta, Lambda, Program, ScremaAccumulator, SoacDestination, SoacOp, Term,
    TermIdSource, TermKind,
};

// =============================================================================
// Analysis types
// =============================================================================

/// Where a SOAC's input array comes from.
#[derive(Debug, Clone)]
pub enum ArrayProvenance {
    /// From a storage buffer entry parameter. `elem_bytes` is captured
    /// at construction time alongside `elem_ty` so the dispatch-len
    /// resolver doesn't have to re-derive it from the type.
    Storage {
        binding: BindingRef,
        elem_ty: Type<TypeName>,
        elem_bytes: u32,
    },
    /// From a range/iota. Carries the bound expression so the dispatch
    /// length can be resolved from `iota(literal)` / `iota(param)` /
    /// `iota(length(arr))` without re-walking lowered IR.
    Range {
        bound: Term,
    },
    /// Anything else (entry-param array vars, SoA-tuple references, etc.).
    /// TLC analysis can't pin down `(set, binding)` for these, but EGIR
    /// can — the from_tlc / soac_expand machinery already handles
    /// `Ref(Var(sym))` and `as_soa_tuple` shapes correctly. Accepted
    /// by every SOAC's parallelization gate post-EGIR-migration of
    /// Map / Reduce / Scan / Screma.
    Opaque,
}

/// A parallelizable SOAC found in a compute entry point.
///
/// Holds the original `SoacOp` — callers that need per-variant logic
/// pattern-match `original`. `provenances` carries one entry per input
/// (length 1 for Reduce/Scan, N for Map/Screma with N inputs).
///
/// `analyze_soac` is the only constructor and guarantees `original` is
/// one of `Map`/`Reduce`/`Scan`/`Screma` — the non-parallelizable
/// variants (Filter, Scatter, ReduceByIndex) never appear here.
#[derive(Debug, Clone)]
pub struct SoacAnalysis {
    pub original: SoacOp,
    pub provenances: Vec<ArrayProvenance>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParallelSoacFlavor {
    Map,
    Reduce,
    Scan,
    Screma,
}

struct ParallelSoacShape<'a> {
    flavor: ParallelSoacFlavor,
    inputs: Vec<&'a ArrayExpr>,
    ne: Option<&'a Term>,
    result_elem_type: Type<TypeName>,
    lowerable_today: bool,
}

fn parallel_soac_shape(soac: &SoacOp) -> Option<ParallelSoacShape<'_>> {
    match soac {
        SoacOp::Map { lam, inputs, .. } => Some(ParallelSoacShape {
            flavor: ParallelSoacFlavor::Map,
            inputs: inputs.iter().collect(),
            ne: None,
            result_elem_type: lam.lam.ret_ty.clone(),
            lowerable_today: true,
        }),
        SoacOp::Reduce { ne, input, .. } => Some(ParallelSoacShape {
            flavor: ParallelSoacFlavor::Reduce,
            inputs: vec![input],
            ne: Some(ne),
            result_elem_type: ne.ty.clone(),
            lowerable_today: true,
        }),
        SoacOp::Scan { ne, input, .. } => Some(ParallelSoacShape {
            flavor: ParallelSoacFlavor::Scan,
            inputs: vec![input],
            ne: Some(ne),
            result_elem_type: ne.ty.clone(),
            lowerable_today: true,
        }),
        SoacOp::Screma {
            inputs,
            lanes,
            accumulators,
        } => {
            let result_elem_type = accumulators
                .first()
                .map(|acc| acc.ne.ty.clone())
                .or_else(|| lanes.first().map(|lane| lane.lam.lam.ret_ty.clone()))
                .unwrap_or_else(|| Type::Constructed(TypeName::Unit, vec![]));
            // Pointwise (no accumulators, all maps) routes through the
            // multi-output map path. Mixed Screma (accumulators + maps) is
            // recognised by analysis so make_lowering_plan can dispatch a
            // serial single-thread compute pipeline — the EGIR-side
            // parallel transform lights up in a follow-up. A Screma with
            // no maps and no accumulators is meaningless; reject it.
            let lowerable_today = !inputs.is_empty() && (!accumulators.is_empty() || !lanes.is_empty());
            Some(ParallelSoacShape {
                flavor: ParallelSoacFlavor::Screma,
                inputs: inputs.iter().collect(),
                ne: None,
                result_elem_type,
                lowerable_today,
            })
        }
        SoacOp::Filter { .. } | SoacOp::Scatter { .. } | SoacOp::ReduceByIndex { .. } => None,
    }
}

/// Every input ArrayExpr a parallelizable SOAC consumes, in source order.
/// Free-function form so callers (`analyze_soac`) can use it on a raw
/// `SoacOp` without building a throwaway `SoacAnalysis`.
fn soac_inputs(soac: &SoacOp) -> Vec<&ArrayExpr> {
    parallel_soac_shape(soac).expect("non-parallel SOAC in SoacAnalysis").inputs
}

impl SoacAnalysis {
    /// Every input ArrayExpr the SOAC consumes, in source order.
    pub fn inputs(&self) -> Vec<&ArrayExpr> {
        soac_inputs(&self.original)
    }

    /// Neutral/initial value — present for Reduce/Scan, `None` for Map/Screma.
    pub fn ne(&self) -> Option<&Term> {
        parallel_soac_shape(&self.original).expect("non-parallel SOAC in SoacAnalysis").ne
    }

    /// Element type of one iteration's output — Map/Scan elem type (from the
    /// per-element lambda's ret_ty); Reduce acc type (from `ne.ty`).
    pub fn result_elem_type(&self) -> Type<TypeName> {
        parallel_soac_shape(&self.original).expect("non-parallel SOAC in SoacAnalysis").result_elem_type
    }
}

/// One entry of an `EntryAnalysis::required_params` list: a free var
/// from the SOAC + prefix_lets fragment, paired with the attribute and
/// `EntryParamBinding` carried forward from the original entry. Phase
/// entries reproduce all four so a captured param routes through EGIR
/// the same way the original did (a runtime-sized array without the
/// `#[storage]` attribute and `EntryParamBinding` falls into the
/// push-constant path and fails its static-layout check).
#[derive(Debug, Clone)]
pub(crate) struct RequiredParam {
    pub sym: SymbolId,
    pub ty: Type<TypeName>,
    pub attr: Option<Attribute>,
    pub binding: Option<interface::EntryParamBinding>,
}

/// Result of analyzing a compute entry point.
#[derive(Debug, Clone)]
struct EntryAnalysis {
    // TODO(parallelize-egir): once EGIR derives domains and dependencies from
    // `slot_sources`, delete the legacy lowering-only fields below (and their
    // unused imports/helpers) under a warning-free build test.  Keeping a dead
    // shadow planner here risks the TLC and EGIR analyses drifting apart.
    pub def_name: SymbolId,
    pub soac: SoacAnalysis,
    /// Let-bound symbol whose RHS was followed to find the tail SOAC.
    /// Preserved so ordered-prefix scheduling can distinguish work that
    /// must run after the tail has materialized its output buffer.
    pub tail_alias: Option<(SymbolId, Type<TypeName>)>,
    /// Let-binding prefix before the SOAC.
    pub prefix_lets: Vec<(SymbolId, Type<TypeName>, Term)>,
    /// The subset of the original entry's params that the SOAC and
    /// `prefix_lets` actually reference, each annotated with its
    /// original attribute + `EntryParamBinding`. Phase entries must
    /// re-declare these as params so the references don't leak out as
    /// free globals during SPIR-V emission.
    pub required_params: Vec<RequiredParam>,
    /// `(slot_index, value)` for each output slot whose value isn't
    /// the SOAC itself. The SOAC consumes slot 0's value; remaining
    /// `OutputSlotStore` terms from the post-SOAC let-chain land here
    /// so the TLC-side two-phase synthesis can append direct
    /// `storage_store` calls for them to phase 2's body.
    pub extra_slots: Vec<(usize, Term)>,
    /// Every normalized output slot after following direct local aliases to
    /// SOAC producers. Downstream per-slot planning consumes this canonical
    /// list instead of rediscovering the original syntax from the source def.
    pub output_slots: Vec<(usize, Term)>,
}

// =============================================================================
// Stage A: Analysis
// =============================================================================

fn analyze_program(program: &Program) -> StableMap<SymbolId, EntryAnalysis> {
    // StableMap (not LookupMap): the iteration order in `run` drives binding
    // allocation through the module-wide `IdSource`, so a LookupMap's
    // randomized iteration would shuffle binding numbers on every compile.
    let mut results = StableMap::new();

    for def in &program.defs {
        let DefMeta::EntryPoint(ref entry_decl) = def.meta else {
            continue;
        };
        if !entry_decl.entry_type.is_compute() {
            continue;
        }

        if let Some(analysis) = analyze_entry(def, &program.symbols) {
            results.insert(def.name, analysis);
        }
    }

    results
}

/// Find the SOAC (if any) that structurally produces the entry's return
/// value, along with the `prefix_lets` that must be rebuilt around each
/// restructured phase body.
///
/// Walks tail position through transparent wrappers:
///   - `Lambda`  — entry params come as a flat Lambda around the body.
///   - `Force`   — no-op marker, peeled transparently.
///   - `Let`     — collected into `prefix_lets`, continue into the body.
///   - `Var(x)`  — follow through to `x`'s defining let (removing that
///                 binding from `prefix_lets`, because the SOAC is now
///                 inlined in its place).
///
/// Anything else at tail — `If`, `Loop`, `App`, `BinOp`, `Tuple`,
/// literals, etc. — means the SOAC (if any) is not structurally the
/// entry's result; returns `None`.
/// Scope stack used by `analyze_entry` to track what's in scope as the
/// tail-position walk descends. Each frame is one binding; we never pop
/// (walking strictly in tail position), so the "stack" only grows.
///
/// Variants carry binding data:
/// - `LambdaParam` — from a peeled outer Lambda. These are the candidates
///   for `EntryAnalysis.required_params`.
/// - `Let` — collected as the walk passes through let-chains. These end
///   up as `EntryAnalysis.prefix_lets` (unless consumed by Var-follow).
#[derive(Debug)]
enum ScopeFrame {
    LambdaParam {
        sym: SymbolId,
        ty: Type<TypeName>,
    },
    Let {
        sym: SymbolId,
        ty: Type<TypeName>,
        rhs: Term,
    },
}

#[derive(Debug, Default)]
struct ScopeStack {
    frames: Vec<ScopeFrame>,
}

impl ScopeStack {
    fn push_lambda_params<'a>(&mut self, params: impl IntoIterator<Item = &'a (SymbolId, Type<TypeName>)>) {
        for (s, t) in params {
            self.frames.push(ScopeFrame::LambdaParam {
                sym: *s,
                ty: t.clone(),
            });
        }
    }
    fn push_let(&mut self, sym: SymbolId, ty: Type<TypeName>, rhs: Term) {
        self.frames.push(ScopeFrame::Let { sym, ty, rhs });
    }
    /// Remove and return the innermost `Let` binding for `sym`, or `None`
    /// if `sym` isn't let-bound in any frame. Used by Var-follow.
    fn remove_let(&mut self, sym: SymbolId) -> Option<(Type<TypeName>, Term)> {
        let idx =
            self.frames.iter().rposition(|f| matches!(f, ScopeFrame::Let { sym: s, .. } if *s == sym))?;
        match self.frames.remove(idx) {
            ScopeFrame::Let { ty, rhs, .. } => Some((ty, rhs)),
            _ => unreachable!(),
        }
    }
    fn get_let(&self, sym: SymbolId) -> Option<(Type<TypeName>, Term)> {
        self.frames.iter().rev().find_map(|frame| match frame {
            ScopeFrame::Let { sym: s, ty, rhs } if *s == sym => Some((ty.clone(), rhs.clone())),
            _ => None,
        })
    }
    fn is_lambda_param(&self, sym: SymbolId) -> bool {
        self.frames.iter().any(|f| matches!(f, ScopeFrame::LambdaParam { sym: s, .. } if *s == sym))
    }
    /// Return captured Lambda params in insertion order (outermost first).
    fn captured_params(&self) -> Vec<(SymbolId, Type<TypeName>)> {
        self.frames
            .iter()
            .filter_map(|f| match f {
                ScopeFrame::LambdaParam { sym, ty } => Some((*sym, ty.clone())),
                _ => None,
            })
            .collect()
    }
    /// Consume the stack and return its `Let` frames in insertion order
    /// (outermost first), i.e. the `prefix_lets` array for the phase bodies.
    fn into_prefix_lets(self) -> Vec<(SymbolId, Type<TypeName>, Term)> {
        self.frames
            .into_iter()
            .filter_map(|f| match f {
                ScopeFrame::Let { sym, ty, rhs } => Some((sym, ty, rhs)),
                _ => None,
            })
            .collect()
    }
}

fn analyze_entry(def: &Def, symbols: &SymbolTable) -> Option<EntryAnalysis> {
    let mut scope = ScopeStack::default();
    let mut current: Term = def.body.clone();
    let mut extra_slots: Vec<(usize, Term)> = Vec::new();
    let mut output_slots: Vec<(usize, Term)> = Vec::new();
    let mut tail_alias: Option<(SymbolId, Type<TypeName>)> = None;

    // The entry's binding layout, which resolves `Ref(Var(sym))` SOAC inputs
    // back to their assigned (set, binding). Empty for non-compute entries.
    let entry_slots: &[Option<EntryParamBinding>] =
        if let DefMeta::EntryPoint(decl) = &def.meta { &decl.param_bindings } else { &[] };

    loop {
        let Term { ty, kind, .. } = current;
        match kind {
            TermKind::Lambda(lam) => {
                scope.push_lambda_params(&lam.params);
                current = *lam.body;
            }
            TermKind::Let {
                name,
                name_ty,
                rhs,
                body,
            } => {
                // `tlc::normalize_outputs` emits the entry tail as a chain
                // of `let _ = OutputSlotStore(i, <slot_value>) in …`.
                // Treat slot 0's value as the entry's parallelizable tail
                // — same primary-slot policy the old direct `Tuple(...)`
                // tail had — and keep walking the body for prefix lets.
                if let TermKind::OutputSlotStore {
                    slot_index: 0, value, ..
                } = rhs.kind
                {
                    // The let's `body` is the rest of the
                    // `normalize_outputs` sequencing chain (slots 1..N
                    // stores ending in `UnitLit`). Collect those sibling
                    // stores so the parallelizable tail can be chosen from
                    // *any* output slot, not just slot 0 — a streamed SOAC
                    // in a later slot must still parallelize even when an
                    // earlier slot holds a fixed value.
                    let mut slot0_value = *value;
                    let mut siblings: Vec<(usize, Term)> = Vec::new();
                    collect_extra_slot_stores(&body, &mut siblings);

                    // Classify slots by their producer, not by whether the
                    // normalized store happens to contain that producer
                    // inline. This applies the same direct-alias following to
                    // every output position that tail analysis already gives
                    // its chosen primary SOAC.
                    let mut consumed_aliases_by_slot: LookupMap<usize, Vec<(SymbolId, Type<TypeName>)>> =
                        LookupMap::new();
                    if let Some((resolved, aliases, consume_aliases)) =
                        resolve_output_alias(&slot0_value, &scope)
                    {
                        slot0_value = resolved;
                        if consume_aliases {
                            consumed_aliases_by_slot.insert(0, aliases);
                        }
                    }
                    for (slot_index, value) in &mut siblings {
                        if let Some((resolved, aliases, consume_aliases)) =
                            resolve_output_alias(value, &scope)
                        {
                            *value = resolved;
                            if consume_aliases {
                                consumed_aliases_by_slot.insert(*slot_index, aliases);
                            }
                        }
                    }
                    for aliases in consumed_aliases_by_slot.values() {
                        for (symbol, _) in aliases {
                            let _ = scope.remove_let(*symbol);
                        }
                    }
                    output_slots =
                        std::iter::once((0, slot0_value.clone())).chain(siblings.iter().cloned()).collect();

                    let sibling_soac_count =
                        siblings.iter().filter(|(_, v)| is_soac_output_candidate(v, &scope)).count();
                    let sibling_map_count =
                        siblings.iter().filter(|(_, v)| is_map_output_candidate(v, &scope)).count();

                    // Slot 0 is the tail itself (a SOAC) or follows through to
                    // one (`Var` alias / `TupleProj`). Keep the original
                    // primary-slot policy verbatim: every sibling is an extra
                    // slot, and a multi-SOAC entry must be all-pointwise-map
                    // (the multidomain split handles those; any other sibling
                    // shape — reduce / scan / filter / scatter / in-place — has
                    // no parallel split yet, so keep the entry serial).
                    if is_soac_output_candidate(&slot0_value, &scope) {
                        if let Some(aliases) = consumed_aliases_by_slot.get(&0) {
                            if let Some(alias) = aliases.first() {
                                tail_alias.get_or_insert(alias.clone());
                            }
                        }
                        if siblings.iter().any(|(_, v)| is_soac_output_candidate(v, &scope)) {
                            let all_pointwise = is_map_output_candidate(&slot0_value, &scope)
                                && siblings.iter().all(|(_, v)| is_map_output_candidate(v, &scope));
                            if !all_pointwise {
                                return None;
                            }
                        }
                        extra_slots = siblings;
                        current = slot0_value;
                        continue;
                    }

                    // Slot 0 is a plain fixed value (no input domain). Promote a
                    // sibling to the primary tail so output order doesn't force a
                    // serial lowering, demoting slot 0 (and other fixed siblings)
                    // to extra slots:
                    //   * exactly one sibling SOAC -> it is the primary tail
                    //     (any flavor); other slots ride as fixed extras.
                    //   * >= 2 sibling SOACs, all pointwise maps -> multidomain:
                    //     the first map is primary, the rest (maps + fixed) are
                    //     extras. The EGIR split gives each map its own domain
                    //     and each fixed slot a 1x1x1 constant-write stage.
                    let promote: Option<usize> = if sibling_soac_count == 1 {
                        siblings.iter().position(|(_, v)| is_soac_output_candidate(v, &scope))
                    } else if sibling_soac_count >= 2 && sibling_map_count == sibling_soac_count {
                        siblings.iter().position(|(_, v)| is_map_output_candidate(v, &scope))
                    } else {
                        None
                    };

                    match promote {
                        Some(pos) => {
                            if let Some(aliases) = consumed_aliases_by_slot.get(&siblings[pos].0) {
                                if let Some(alias) = aliases.first() {
                                    tail_alias.get_or_insert(alias.clone());
                                }
                            }
                            extra_slots.push((0, slot0_value));
                            let mut primary_value = None;
                            for (idx, (i, v)) in siblings.into_iter().enumerate() {
                                if idx == pos {
                                    primary_value = Some(v);
                                } else {
                                    extra_slots.push((i, v));
                                }
                            }
                            current = primary_value.expect("promoted slot present");
                        }
                        None => {
                            // All outputs fixed (nothing to parallelize) or an
                            // unsupported mix: retain every sibling as an extra
                            // slot and fall through to slot 0, which the non-SOAC
                            // tail arm rejects.
                            extra_slots = siblings;
                            current = slot0_value;
                        }
                    }
                    continue;
                }
                scope.push_let(name, name_ty, *rhs);
                current = *body;
            }
            TermKind::Soac(soac) => {
                let parallelizable = analyze_soac(&soac, &ty, symbols, &entry_slots)?;
                let captured_params = scope.captured_params();
                let prefix_lets = scope.into_prefix_lets();
                let required_params = compute_required_params(
                    &parallelizable,
                    &prefix_lets,
                    &extra_slots,
                    &captured_params,
                    def,
                    symbols,
                );
                return Some(EntryAnalysis {
                    def_name: def.name,
                    soac: parallelizable,
                    tail_alias,
                    prefix_lets,
                    required_params,
                    extra_slots,
                    output_slots,
                });
            }
            TermKind::Var(VarRef::Symbol(sym)) => {
                // Var-follow: the tail is an alias. If `sym` is an entry
                // param, the entry returns a param — not a SOAC, reject.
                if scope.is_lambda_param(sym) {
                    return None;
                }
                // Otherwise try to consume a let binding.
                match scope.remove_let(sym) {
                    Some((alias_ty, rhs)) => {
                        tail_alias.get_or_insert((sym, alias_ty));
                        current = rhs;
                    }
                    None => return None,
                }
            }
            // A genuine multi-output `Screma` let-bound and projected at field 0
            // (`proj(sym, 0)`). A fused `map → reduce` is scalar-output — a bare
            // `Soac(Screma)` inline, or an alias `Var(sym)` when let-bound — so
            // the `Soac`/`Var` arms handle it, not this one.
            TermKind::TupleProj { tuple, idx: 0 } => match tuple.kind {
                TermKind::Var(VarRef::Symbol(sym)) => {
                    if scope.is_lambda_param(sym) {
                        return None;
                    }
                    match scope.remove_let(sym) {
                        Some((alias_ty, rhs))
                            if matches!(rhs.kind, TermKind::Soac(SoacOp::Screma { .. })) =>
                        {
                            tail_alias.get_or_insert((sym, alias_ty));
                            current = rhs;
                        }
                        _ => return None,
                    }
                }
                _ => return None,
            },
            _ => return None,
        }
    }
}

/// Resolve a direct local alias chain for output-slot classification. Aliases
/// leading to a SOAC (including a shared-Screma projection) are marked for
/// consumption; fixed aliases remain available for sibling-stage captures.
fn resolve_output_alias(
    term: &Term,
    scope: &ScopeStack,
) -> Option<(Term, Vec<(SymbolId, Type<TypeName>)>, bool)> {
    let mut current = term.clone();
    let mut aliases = Vec::new();
    let mut seen = LookupSet::new();
    loop {
        match current.kind {
            TermKind::Var(VarRef::Symbol(symbol)) => {
                if !seen.insert(symbol) {
                    return None;
                }
                let (ty, rhs) = scope.get_let(symbol)?;
                aliases.push((symbol, ty));
                current = rhs;
            }
            _ if aliases.is_empty() => return None,
            _ => {
                let consume_aliases = is_soac_output_candidate(&current, scope);
                return Some((current, aliases, consume_aliases));
            }
        }
    }
}

/// Whether a normalized output value is itself the parallel producer. A tuple
/// projection qualifies only for the established field-0 shared-Screma shape;
/// ordinary variables and projections are fixed outputs.
fn is_soac_output_candidate(term: &Term, scope: &ScopeStack) -> bool {
    match &term.kind {
        TermKind::Soac(_) => true,
        TermKind::TupleProj { tuple, idx: 0 } => {
            let TermKind::Var(VarRef::Symbol(symbol)) = tuple.kind else {
                return false;
            };
            scope
                .get_let(symbol)
                .is_some_and(|(_, rhs)| matches!(rhs.kind, TermKind::Soac(SoacOp::Screma { .. })))
        }
        _ => false,
    }
}

/// Whether an output candidate is pointwise and safe for the per-slot map
/// path. A field-0 projection represents its entire shared Screma producer;
/// sibling projections are outputs of that producer, not additional SOACs.
fn is_map_output_candidate(term: &Term, scope: &ScopeStack) -> bool {
    match &term.kind {
        TermKind::Soac(_) => is_map_only_fresh_soac_term(term),
        TermKind::TupleProj { tuple, idx: 0 } => {
            let TermKind::Var(VarRef::Symbol(symbol)) = tuple.kind else {
                return false;
            };
            scope.get_let(symbol).is_some_and(|(_, rhs)| {
                matches!(rhs.kind, TermKind::Soac(SoacOp::Screma { ref accumulators, .. }) if accumulators.is_empty())
            })
        }
        _ => false,
    }
}

/// Walk the body half of slot 0's sequencing let, picking off each
/// `OutputSlotStore { slot_index, value, .. }` along the let-chain
/// into `out`. The chain shape from `normalize_outputs` is `let _seq =
/// OutputSlotStore(i, v) in <rest>` terminating in `UnitLit`.
pub(crate) fn collect_extra_slot_stores(term: &Term, out: &mut Vec<(usize, Term)>) {
    let mut cur = term;
    loop {
        match &cur.kind {
            TermKind::Let { rhs, body, .. } => {
                if let TermKind::OutputSlotStore {
                    slot_index, value, ..
                } = &rhs.kind
                {
                    out.push((*slot_index, (**value).clone()));
                }
                cur = body;
            }
            _ => break,
        }
    }
}

/// Compute the subset of outer entry params that the tail SOAC + retained
/// prefix_lets actually reference, each annotated with the attribute +
/// `EntryParamBinding` carried forward from the original entry. Phase
/// entries must re-declare these as params (or the references leak out
/// as undefined globals during SPIR-V emission) AND reproduce the
/// original binding metadata (or a runtime-sized array param falls into
/// EGIR's push-constant path and fails its static-layout check).
///
/// Reuses `closure_convert::collect_free_vars` with empty `top_level` /
/// `known_defs` sets — no top-level filtering, just "bound vs free"
/// within the fragment we walk.
fn compute_required_params(
    soac: &SoacAnalysis,
    prefix_lets: &[(SymbolId, Type<TypeName>, Term)],
    extra_slots: &[(usize, Term)],
    captured_params: &[(SymbolId, Type<TypeName>)],
    def: &Def,
    symbols: &SymbolTable,
) -> Vec<RequiredParam> {
    use crate::LookupSet;
    let empty_top: LookupSet<SymbolId> = LookupSet::new();
    let empty_defs: LookupSet<String> = LookupSet::new();
    let mut bound: LookupSet<SymbolId> = LookupSet::new();
    let mut free: Vec<Term> = Vec::new();
    let mut seen: LookupSet<SymbolId> = LookupSet::new();

    // Each prefix RHS is evaluated with all *previous* prefix names in
    // scope. The SOAC sees the full prefix in scope.
    for (name, _ty, rhs) in prefix_lets {
        collect_free_vars(
            rhs,
            &bound,
            &empty_top,
            &empty_defs,
            symbols,
            &mut free,
            &mut seen,
        );
        bound.insert(*name);
    }
    // Wrap the SOAC in a throwaway Term so we can reuse the same walker.
    // `collect_free_vars` reads the term's `kind` and structure but never
    // queries the id; a local TermIdSource keeps the "no placeholders"
    // invariant without threading a source down through the analysis API.
    let mut throwaway_ids = TermIdSource::new();
    let soac_term = Term {
        id: throwaway_ids.next_id(),
        ty: Type::Variable(0),
        span: ast::Span::new(0, 0, 0, 0),
        kind: TermKind::Soac(soac.original.clone()),
    };
    collect_free_vars(
        &soac_term,
        &bound,
        &empty_top,
        &empty_defs,
        symbols,
        &mut free,
        &mut seen,
    );

    // Sibling slot values evaluate in phase 2 alongside the reduce
    // result store; their free vars must also be re-declared on the
    // phase entries.
    for (_, sval) in extra_slots {
        collect_free_vars(
            sval,
            &bound,
            &empty_top,
            &empty_defs,
            symbols,
            &mut free,
            &mut seen,
        );
    }

    let free_syms: LookupSet<SymbolId> = free
        .iter()
        .filter_map(
            |t| {
                if let TermKind::Var(VarRef::Symbol(s)) = &t.kind {
                    Some(*s)
                } else {
                    None
                }
            },
        )
        .collect();

    // Zip in each captured param's attribute and `EntryParamBinding`
    // from the original entry. `peel_lambda_params` indexes the entry
    // body's outer Lambda; that index aligns with `EntryDecl.params`
    // and `EntryDecl.param_bindings`.
    let decl = match &def.meta {
        DefMeta::EntryPoint(d) => Some(d),
        _ => None,
    };
    let (orig_params, _) = peel_lambda_params(&def.body);

    captured_params
        .iter()
        .filter(|(s, _)| free_syms.contains(s))
        .map(|(sym, ty)| {
            let orig_idx = orig_params.iter().position(|(s, _)| s == sym);
            let attr = decl
                .and_then(|d| orig_idx.and_then(|i| d.params.get(i)))
                .and_then(forwardable_binding_attribute);
            let binding =
                decl.and_then(|d| orig_idx.and_then(|i| d.param_bindings.get(i).cloned())).flatten();
            RequiredParam {
                sym: *sym,
                ty: ty.clone(),
                attr,
                binding,
            }
        })
        .collect()
}

/// Find the first `#[storage]` / `#[uniform]` / `#[texture]` / `#[sampler]`
/// / `#[storage_image]` attribute on an entry-param pattern, peeling
/// through `Typed` and `Attributed` wrappers to reach it. Phase entries
/// re-attach this attribute on the corresponding captured param so EGIR
/// routes the param to the same binding the host provided.
fn forwardable_binding_attribute(pat: &ast::Pattern) -> Option<Attribute> {
    match &pat.kind {
        ast::PatternKind::Attributed(attrs, inner) => {
            for attr in attrs {
                if matches!(
                    attr,
                    Attribute::Storage { .. }
                        | Attribute::Uniform { .. }
                        | Attribute::Texture { .. }
                        | Attribute::Sampler { .. }
                        | Attribute::StorageImage { .. }
                ) {
                    return Some(attr.clone());
                }
            }
            forwardable_binding_attribute(inner)
        }
        ast::PatternKind::Typed(inner, _) => forwardable_binding_attribute(inner),
        _ => None,
    }
}

/// Analyze a SOAC, rejecting non-parallelizable variants (Filter,
/// Scatter, ReduceByIndex). Returns a `SoacAnalysis` holding the
/// `SoacOp` plus one provenance per input.
fn analyze_soac(
    soac: &SoacOp,
    _result_ty: &Type<TypeName>,
    _symbols: &SymbolTable,
    entry_slots: &[Option<EntryParamBinding>],
) -> Option<SoacAnalysis> {
    let shape = parallel_soac_shape(soac)?;
    if !shape.lowerable_today {
        return None;
    }
    debug_assert!(matches!(
        shape.flavor,
        ParallelSoacFlavor::Map
            | ParallelSoacFlavor::Reduce
            | ParallelSoacFlavor::Scan
            | ParallelSoacFlavor::Screma
    ));

    let normalized: SoacOp = match soac {
        SoacOp::Map {
            lam,
            inputs,
            destination,
        } => {
            // Map is migrated to the EGIR-side lowering path, which
            // rediscovers input shapes natively (storage buffers, SoA
            // tuples, ranges, view refs). No TLC-side input gate needed.
            SoacOp::Map {
                lam: lam.clone(),
                inputs: inputs.clone(),
                destination: *destination,
            }
        }
        SoacOp::Reduce { op, ne, input } => {
            // The EGIR-side reduce migration rediscovers concrete input
            // shapes from the entry body, just like Map. Accept whatever
            // `input` is; non-StorageBuffer cases get `Opaque` provenance
            // and the EGIR pass walks the actual EgirEntry's StorageView
            // node for set/binding extraction.
            SoacOp::Reduce {
                op: op.clone(),
                ne: ne.clone(),
                input: input.clone(),
            }
        }
        SoacOp::Screma {
            lanes,
            accumulators,
            inputs,
        } => SoacOp::Screma {
            lanes: lanes.clone(),
            accumulators: accumulators.clone(),
            inputs: inputs.clone(),
        },
        SoacOp::Scan {
            op,
            reduce_op,
            ne,
            input,
            destination,
        } => SoacOp::Scan {
            op: op.clone(),
            reduce_op: reduce_op.clone(),
            ne: ne.clone(),
            input: input.clone(),
            destination: *destination,
        },
        _ => return None,
    };

    // Re-derive provenances from the normalized inputs. `soac_inputs`
    // is the single source of truth for "what are the inputs". Map
    // inputs that don't classify (e.g. `Ref(Var(tuple_sym))` for SoA
    // tuple entry params) fall back to `Opaque` — the EGIR lowering
    // path rediscovers their concrete shape. Reduce/Scan/Screma reject
    // such inputs at their own gates above; reaching this loop with
    // Opaque can only happen for Map.
    let provenances: Vec<ArrayProvenance> = soac_inputs(&normalized)
        .iter()
        .map(|ae| classify_input(ae, entry_slots).unwrap_or(ArrayProvenance::Opaque))
        .collect();
    Some(SoacAnalysis {
        original: normalized,
        provenances,
    })
}

fn binop(
    op: &str,
    lhs: Term,
    rhs: Term,
    ty: Type<TypeName>,
    span: ast::Span,
    term_ids: &mut TermIdSource,
) -> Term {
    let func = Term {
        id: term_ids.next_id(),
        ty: Type::Constructed(
            TypeName::Arrow,
            vec![
                ty.clone(),
                Type::Constructed(TypeName::Arrow, vec![ty.clone(), ty.clone()]),
            ],
        ),
        span,
        kind: TermKind::BinOp(ast::BinaryOp { op: op.to_string() }),
    };
    Term {
        id: term_ids.next_id(),
        ty,
        span,
        kind: TermKind::App {
            func: Box::new(func),
            args: vec![lhs, rhs],
        },
    }
}

fn classify_input(input: &ArrayExpr, entry_slots: &[Option<EntryParamBinding>]) -> Option<ArrayProvenance> {
    match input {
        ArrayExpr::StorageView(crate::tlc::StorageView { binding, elem_ty, .. }) => {
            let elem_bytes = crate::ssa::layout::type_byte_size(elem_ty).expect(
                "ArrayExpr::StorageView elem_ty must have a static byte layout — \
                 every constructor (lift_gathers / buffer_specialize / parallelize \
                 synthesis) is responsible for only ever producing sized elem types",
            );
            Some(ArrayProvenance::Storage {
                binding: *binding,
                elem_ty: elem_ty.clone(),
                elem_bytes,
            })
        }
        ArrayExpr::Var(_, ty) => {
            // A bare entry-param storage-buffer reference. Resolve the
            // assigned (set, binding) via the entry's binding layout — the
            // same lookup `default_entry_dispatch_len` uses. Tuple-of-views
            // params resolve to their first slot (same element count).
            if let Some(sym) = input.as_named_ref() {
                if let Some(slot) = entry_slots.iter().flatten().find(|s| s.param_sym == sym) {
                    let (buf, elem_ty, elem_bytes) = slot.first_buffer();
                    return Some(ArrayProvenance::Storage {
                        binding: buf,
                        elem_ty: elem_ty.clone(),
                        elem_bytes,
                    });
                }
            }
            let ref_ty = crate::types::strip_unique(ty);
            if let Some(binding) = crate::types::array_view_region(&ref_ty) {
                if let Some(elem_ty) = ref_ty.elem_type() {
                    let elem_bytes = crate::ssa::layout::type_byte_size(elem_ty)?;
                    return Some(ArrayProvenance::Storage {
                        binding,
                        elem_ty: elem_ty.clone(),
                        elem_bytes,
                    });
                }
            }
            None
        }
        // An `iota(N)` input is a `Range`; its `len` is the dispatch length.
        ArrayExpr::Range { len, .. } => Some(ArrayProvenance::Range {
            bound: (**len).clone(),
        }),
        _ => None,
    }
}

/// A SOAC output slot the multi-domain map split can lower as its own parallel
/// stage: a map-only SOAC producing a fresh array. A plain `Map` qualifies, as
/// does a `Screma` with map lambdas and no accumulators. Reduce / scan / filter
/// / scatter and in-place maps do not. A map-only `Screma` may have several
/// inputs; the split dispatches over `inputs.first()`, relying on the type
/// checker's invariant that a SOAC's inputs all share one length.
fn is_map_only_fresh_soac_term(t: &Term) -> bool {
    match &t.kind {
        TermKind::Soac(SoacOp::Map {
            destination: SoacDestination::Fresh,
            ..
        }) => true,
        TermKind::Soac(SoacOp::Screma { accumulators, .. }) => accumulators.is_empty(),
        _ => false,
    }
}

/// The parallel strategy recognized for a compute entry's tail SOAC. EGIR
/// drives the actual lowering (Seg creation, phasing, scratch allocation) from
/// this plus the entry body; TLC only recognizes the shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelStrategy {
    Map,
    Reduce,
    Scan,
    Screma,
}

/// Per-entry recognition output. The two host-facing facts (`dispatch_sized`,
/// `forced_output`) are what `egir::from_tlc` needs to size outputs at
/// conversion; `strategy` is what `egir::parallelize` keys lowering on. No
/// binding numbers — EGIR allocates all scratch itself.
#[derive(Debug, Clone)]
pub struct EntryRecognition {
    pub strategy: ParallelStrategy,
    /// Output is one element per thread (Map/Scan) vs a single scalar (Reduce).
    pub dispatch_sized: bool,
    /// A pre-pinned output binding (a gather pre-pass's result); else EGIR
    /// auto-allocates the entry output.
    pub forced_output: Option<BindingRef>,
}

pub struct ParallelizationResult {
    pub program: Program,
    pub pipeline: PipelineDescriptor,
    /// Per-entry recognition keyed by surface name (matches `EgirEntry::name`).
    /// Present only for parallelizable compute entries.
    pub recognitions: LookupMap<String, EntryRecognition>,
    /// Source-parameter name for each storage `(set, binding)`, captured
    /// from the original compute entries before they are replaced by phase
    /// entries. The relabel pass in `to_egraph` restores these onto the
    /// finalized descriptor's input bindings, which the parallel path would
    /// otherwise name positionally (`input_0`, `input_1`, …).
    pub input_names: LookupMap<(u32, u32), String>,
}

pub(crate) fn let_term(
    name: SymbolId,
    name_ty: Type<TypeName>,
    rhs: Term,
    body: Term,
    span: ast::Span,
    term_ids: &mut TermIdSource,
) -> Term {
    let ty = body.ty.clone();
    Term {
        id: term_ids.next_id(),
        ty,
        span,
        kind: TermKind::Let {
            name,
            name_ty,
            rhs: Box::new(rhs),
            body: Box::new(body),
        },
    }
}

/// Build an `App(Var(Builtin { id, overload_idx: 0 }), args)` term for

fn peel_lambda_params(term: &Term) -> (Vec<(SymbolId, Type<TypeName>)>, &Term) {
    match &term.kind {
        TermKind::Lambda(lam) => {
            let (mut inner, body) = peel_lambda_params(&lam.body);
            let mut params = lam.params.clone();
            params.append(&mut inner);
            (params, body)
        }
        _ => (vec![], term),
    }
}

/// Walk outer `Lambda`s and `Let`s, hoisting eligible SOAC-RHSs. Stops
/// descending at the first non-Lambda-non-Let term — that's the tail

fn collect_entry_input_names(program: &Program) -> LookupMap<(u32, u32), String> {
    let mut out: LookupMap<(u32, u32), String> = LookupMap::new();
    let mut ambiguous: LookupSet<(u32, u32)> = LookupSet::new();
    let mut put = |out: &mut LookupMap<(u32, u32), String>, key: (u32, u32), name: String| {
        if ambiguous.contains(&key) {
            return;
        }
        match out.get(&key) {
            Some(existing) if *existing != name => {
                out.remove(&key);
                ambiguous.insert(key);
            }
            Some(_) => {}
            None => {
                out.insert(key, name);
            }
        }
    };
    for def in &program.defs {
        let DefMeta::EntryPoint(decl) = &def.meta else {
            continue;
        };
        if !decl.entry_type.is_compute() {
            continue;
        }
        let (params, _) = peel_lambda_params(&def.body);
        // Capture both binding kinds: explicit `#[storage(set, binding)]`
        // attributes (which the auto-allocator leaves alone) and the
        // auto-allocated slots cached on `decl.param_bindings`.
        let layout = &decl.param_bindings;
        for (i, (sym, _)) in params.iter().enumerate() {
            let name = crate::symbol_name_or_bug(&program.symbols, *sym).to_string();
            if let Some(br) = decl.params.get(i).and_then(crate::binding_layout::extract_storage_binding) {
                put(&mut out, (br.set, br.binding), name);
                continue;
            }
            let Some(pb) = layout.get(i).and_then(|slot| slot.as_ref()) else {
                continue;
            };
            match &pb.kind {
                EntryParamBindingKind::Single { binding, .. } => {
                    put(&mut out, (binding.set, binding.binding), name);
                }
                EntryParamBindingKind::TupleOfViews(fields) => {
                    for (idx, f) in fields.iter().enumerate() {
                        put(
                            &mut out,
                            (f.binding.set, f.binding.binding),
                            format!("{}_{}", name, idx),
                        );
                    }
                }
            }
        }
    }
    out
}

/// Parallelize SOACs in compute entry points.
///
/// `disable` short-circuits the whole pass — every compute entry runs as a
/// single sequential loop and the pipeline descriptor is built from the
/// untouched program. The canonical caller also uses this flag to skip the

pub(crate) fn make_entry_def(
    name: &str,
    body: Term,
    return_ty: Type<TypeName>,
    required_params: &[RequiredParam],
    storage_bindings: Vec<interface::StorageBindingDecl>,
    program: &mut Program,
    term_ids: &mut TermIdSource,
) -> Def {
    let sym = program.symbols.alloc(name.to_string());
    program.def_syms.insert(name.to_string(), sym);

    let dummy_span = ast::Span::new(0, 0, 0, 0);
    let dummy_expr = ast::Node {
        h: ast::Header {
            id: ast::NodeId(0),
            span: dummy_span,
        },
        kind: ast::ExprKind::Unit,
    };

    // Wrap the body in a single flat Lambda carrying `required_params`, mirroring
    // `transform_entry`'s convention. Compute the full arrow type.
    let lambda_params: Vec<(SymbolId, Type<TypeName>)> =
        required_params.iter().map(|r| (r.sym, r.ty.clone())).collect();
    let (full_ty, body) = if required_params.is_empty() {
        (return_ty.clone(), body)
    } else {
        let mut ty = return_ty.clone();
        for r in required_params.iter().rev() {
            ty = Type::Constructed(TypeName::Arrow, vec![r.ty.clone(), ty]);
        }
        let lam_body = Term {
            id: term_ids.next_id(),
            ty: ty.clone(),
            span: dummy_span,
            kind: TermKind::Lambda(Lambda {
                params: lambda_params,
                body: Box::new(body),
                ret_ty: return_ty.clone(),
            }),
        };
        (ty, lam_body)
    };

    // Build ast::Pattern entries. Each required_param that originated from
    // an attributed entry param (`#[storage(...)]` / `#[uniform(...)]` /
    // etc.) gets its original attribute re-attached, so EGIR conversion
    // routes it to the same binding the host provided rather than
    // falling back to push constants.
    let ast_params: Vec<ast::Pattern> = required_params
        .iter()
        .map(|r| {
            let pname = crate::symbol_name_or_bug(&program.symbols, r.sym).to_string();
            let name_pat = ast::Pattern {
                h: ast::Header {
                    id: ast::NodeId(0),
                    span: dummy_span,
                },
                kind: ast::PatternKind::Name(pname),
            };
            match &r.attr {
                Some(attr) => ast::Pattern {
                    h: ast::Header {
                        id: ast::NodeId(0),
                        span: dummy_span,
                    },
                    kind: ast::PatternKind::Attributed(vec![attr.clone()], Box::new(name_pat)),
                },
                None => name_pat,
            }
        })
        .collect();
    let param_bindings: Vec<Option<interface::EntryParamBinding>> =
        required_params.iter().map(|r| r.binding.clone()).collect();

    // Declare one anonymous output so `build_entry_outputs` in from_tlc
    // treats the return type as a single storage-bound output and
    // `emit_compute_output_stores` writes it into the partials/result
    // buffer rather than leaving a Return(value) that SPIR-V rejects.
    // (The ty field here is a placeholder — build_entry_outputs reads the
    //  TLC ret_type, not this.)
    let outputs = vec![interface::EntryOutput {
        ty: Type::Constructed(TypeName::Unit, vec![]),
        attribute: None,
    }];

    Def {
        name: sym,
        ty: full_ty,
        body,
        meta: DefMeta::EntryPoint(Box::new(interface::EntryDecl {
            entry_type: Attribute::Compute,
            name: name.to_string(),
            name_span: dummy_span,
            size_params: vec![],
            type_params: vec![],
            param_bindings,
            params: ast_params,
            outputs,
            storage_bindings,
            feedback: vec![],
            body: dummy_expr,
        })),
        arity: required_params.len(),
    }
}

/// Carries the program through with a serial default pipeline and no
/// parallelization plans.
pub fn run(
    mut program: Program,
    _disable: bool,
    _binding_ids: &mut crate::IdSource<u32>,
) -> crate::error::Result<ParallelizationResult> {
    let input_names = collect_entry_input_names(&program);

    // Recognition: collapse equal-domain sibling output maps, then analyze each
    // compute entry's tail SOAC. EGIR does the lowering from these recognitions.
    let mut term_ids = TermIdSource::new();
    super::fusion::fuse_equal_domain_sibling_maps(&mut program, &mut term_ids);
    let analyses = analyze_program(&program);

    let mut recognitions: LookupMap<String, EntryRecognition> = LookupMap::new();
    for (def_name, analysis) in &analyses {
        let (strategy, dispatch_sized) = match &analysis.soac.original {
            SoacOp::Map { .. } => (ParallelStrategy::Map, true),
            SoacOp::Reduce { .. } => (ParallelStrategy::Reduce, false),
            SoacOp::Scan { .. } => (ParallelStrategy::Scan, true),
            SoacOp::Screma { accumulators, .. } => {
                let has_scan = accumulators.iter().any(|a| matches!(a.kind, ScremaAccumulator::Scan));
                (ParallelStrategy::Screma, accumulators.is_empty() || has_scan)
            }
            _ => continue,
        };
        let forced_output = program.defs.iter().find(|d| d.name == *def_name).and_then(|d| match &d.meta {
            DefMeta::EntryPoint(decl) => decl
                .storage_bindings
                .iter()
                .find(|b| matches!(b.role, interface::StorageRole::Output))
                .map(|b| b.binding),
            _ => None,
        });
        let name = crate::symbol_name_or_bug(&program.symbols, *def_name).to_string();
        recognitions.insert(
            name,
            EntryRecognition {
                strategy,
                dispatch_sized,
                forced_output,
            },
        );
    }

    let mut pipelines = Vec::new();
    for def in &program.defs {
        let DefMeta::EntryPoint(decl) = &def.meta else {
            continue;
        };
        let name = crate::symbol_name_or_bug(&program.symbols, def.name).to_string();
        if decl.entry_type.is_compute() {
            pipelines.push(Pipeline::Compute(ComputePipeline {
                bindings: Vec::new(),
                stages: vec![ComputeStage {
                    entry_point: name,
                    workgroup_size: (64, 1, 1),
                    dispatch_size: DispatchSize::Fixed { x: 1, y: 1, z: 1 },
                    reads: vec![],
                    writes: vec![],
                }],
                default_total_threads: None,
                feedback: Vec::new(),
            }));
        } else {
            let stage = if decl.entry_type == Attribute::Vertex {
                ShaderStage::Vertex
            } else {
                ShaderStage::Fragment
            };
            pipelines.push(Pipeline::Graphics(GraphicsPipeline {
                stages: vec![GraphicsStage {
                    entry_point: name,
                    stage,
                }],
                bindings: Vec::new(),
                vertex_inputs: vec![],
                fragment_outputs: vec![],
                feedback: Vec::new(),
            }));
        }
    }
    Ok(ParallelizationResult {
        program,
        pipeline: PipelineDescriptor { pipelines },
        recognitions,
        input_names,
    })
}

/// Currently performs no scalar-reduction outlining.
pub fn hoist_scalar_prepasses(program: Program, _binding_ids: &mut crate::IdSource<u32>) -> Program {
    program
}
