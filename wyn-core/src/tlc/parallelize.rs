//! TLC-level SOAC parallelization.
//!
//! Stage A: Analyze compute entry points to find parallelizable SOACs.
//! Stage B: Restructure the program — create new entry points with chunked SOACs,
//!          retain the source-level pipeline shells consumed by terminal EGIR lowering.
//!
//! Loop creation and storage lowering stay in SSA (`to_ssa` + `soac_lower`).

use super::closure_convert::compute_free_vars;
use super::VarRef;
use crate::ast::{self, TypeName};
use crate::builtins::catalog;
use crate::egir::from_tlc::AUTO_STORAGE_SET;
use crate::interface::{self, Attribute, EntryParamBinding, EntryParamBindingKind};
use crate::pipeline_descriptor::*;
use crate::types::TypeExt;
use crate::{BindingRef, SymbolId, SymbolTable};
use crate::{LookupMap, LookupSet};
use polytype::Type;

use super::{
    ArrayExpr, Def, DefMeta, Lambda, Program, SoacDestination, SoacOp, Term, TermIdSource, TermKind,
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

/// Result of analyzing a compute entry point: its recognized tail SOAC. EGIR
/// derives domains, dependencies, and per-slot lowering from the entry's
/// `slot_sources`, so recognition keeps only the SOAC classification.
#[derive(Debug, Clone)]
struct EntryAnalysis {
    pub soac: SoacAnalysis,
}

// =============================================================================
// Stage A: Analysis
// =============================================================================

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
        for (s, _) in params {
            self.frames.push(ScopeFrame::LambdaParam { sym: *s });
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
}

fn analyze_entry(def: &Def, symbols: &SymbolTable) -> Option<EntryAnalysis> {
    let mut scope = ScopeStack::default();
    let mut current: Term = def.body.clone();
    let mut extra_slots: Vec<(usize, Term)> = Vec::new();
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
                return Some(EntryAnalysis { soac: parallelizable });
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

fn classify_input(input: &ArrayExpr, entry_slots: &[Option<EntryParamBinding>]) -> Option<ArrayProvenance> {
    match input {
        ArrayExpr::StorageView(crate::tlc::StorageView { binding, elem_ty, .. }) => {
            let elem_bytes = crate::ssa::layout::storage_elem_stride(elem_ty).expect(
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

pub struct ParallelizationResult {
    pub program: Program,
    pub pipeline: PipelineDescriptor,
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
    origin: interface::EntryOrigin,
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
            origin,
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

    // Equal-domain sibling maps remain a source-level fusion; EGIR derives
    // semantic placement and scheduling from the resulting program.
    let mut term_ids = TermIdSource::new();
    super::fusion::fuse_equal_domain_sibling_maps(&mut program, &mut term_ids);

    let mut pipelines = Vec::new();
    for def in &program.defs {
        let DefMeta::EntryPoint(decl) = &def.meta else {
            continue;
        };
        let name = crate::symbol_name_or_bug(&program.symbols, def.name).to_string();
        // The entry's `previous`-view ping-pong pairs (recorded by
        // `resolve_resources`) seed the pipeline here; `KernelSchedule`
        // carries them through scheduling into the published descriptor.
        let feedback: Vec<crate::pipeline_descriptor::FeedbackPair> = decl
            .feedback
            .iter()
            .map(|pair| crate::pipeline_descriptor::FeedbackPair {
                read_set: pair.read.set,
                read_binding: pair.read.binding,
                write_set: pair.write.set,
                write_binding: pair.write.binding,
            })
            .collect();
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
                feedback,
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
                feedback,
            }));
        }
    }
    Ok(ParallelizationResult {
        program,
        pipeline: PipelineDescriptor { pipelines },
        input_names,
    })
}

type LocalPrepassBinding = (SymbolId, Type<TypeName>, Term);

enum ScalarPrepassEntryPolicy {
    Graphics {
        tainted: LookupSet<SymbolId>,
        uniforms: LookupMap<SymbolId, BindingRef>,
    },
    Compute {
        source_def: Def,
        params: LookupSet<SymbolId>,
    },
}

struct ScalarPrepassHoister<'a> {
    entry_name: &'a str,
    policy: ScalarPrepassEntryPolicy,
    top_level: &'a LookupSet<SymbolId>,
    locals: Vec<LocalPrepassBinding>,
    binding_ids: &'a mut crate::IdSource<u32>,
    added_decls: Vec<interface::StorageBindingDecl>,
    new_defs: Vec<Def>,
    program: &'a mut Program,
    term_ids: &'a mut TermIdSource,
}

/// Outline scalar reductions before closure conversion.
///
/// The generated compute entry carries the transitive, reproducible portion of
/// the surrounding `let` scope that its reduction needs. Its result binding is
/// declared as an Output on the pre-pass itself, allowing the later
/// parallelization recognition to discover the forced result binding without a
/// side table. Defunctionalization then sees the generated entry as ordinary
/// source-shaped TLC and attaches lambda captures in the correct scope.
pub fn hoist_scalar_prepasses(mut program: Program, binding_ids: &mut crate::IdSource<u32>) -> Program {
    let top_level_syms: LookupSet<SymbolId> = program.defs.iter().map(|d| d.name).collect();
    let entry_indices: Vec<usize> = program
        .defs
        .iter()
        .enumerate()
        .filter_map(|(index, def)| matches!(def.meta, DefMeta::EntryPoint(_)).then_some(index))
        .collect();

    let mut term_ids = TermIdSource::new();
    let mut new_defs = Vec::new();

    for index in entry_indices {
        let entry_name = crate::symbol_name_or_bug(&program.symbols, program.defs[index].name).to_string();
        let body = program.defs[index].body.clone();
        let (params, _) = peel_lambda_params(&body);
        let policy = match &program.defs[index].meta {
            DefMeta::EntryPoint(decl) if decl.entry_type.is_compute() => {
                if !compute_entry_can_broadcast_scalar_prepasses(&program.defs[index], &program.symbols) {
                    continue;
                }
                ScalarPrepassEntryPolicy::Compute {
                    source_def: program.defs[index].clone(),
                    params: params.iter().map(|(symbol, _)| *symbol).collect(),
                }
            }
            DefMeta::EntryPoint(decl) => {
                let mut varying = LookupSet::new();
                let mut uniforms = LookupMap::new();
                for (param_index, (symbol, _)) in params.iter().enumerate() {
                    match decl
                        .params
                        .get(param_index)
                        .and_then(crate::binding_layout::extract_uniform_binding)
                    {
                        Some(binding) => {
                            uniforms.insert(*symbol, binding);
                        }
                        None => {
                            varying.insert(*symbol);
                        }
                    }
                }
                ScalarPrepassEntryPolicy::Graphics {
                    tainted: compute_taint_set(&body, &varying, &program.symbols),
                    uniforms,
                }
            }
            _ => unreachable!("entry_indices contains only entry points"),
        };

        let (new_body, added_decls, mut entry_defs) = {
            let mut hoister = ScalarPrepassHoister {
                entry_name: &entry_name,
                policy,
                top_level: &top_level_syms,
                locals: Vec::new(),
                binding_ids,
                added_decls: Vec::new(),
                new_defs: Vec::new(),
                program: &mut program,
                term_ids: &mut term_ids,
            };
            let body = hoister.rewrite(body);
            (body, hoister.added_decls, hoister.new_defs)
        };
        new_defs.append(&mut entry_defs);

        program.defs[index].body = new_body;
        if let DefMeta::EntryPoint(decl) = &mut program.defs[index].meta {
            decl.storage_bindings.extend(added_decls);
        }
    }

    program.defs.extend(new_defs);
    super::anf::debug_check(&program, "hoist_scalar_prepasses");
    program
}

impl ScalarPrepassHoister<'_> {
    fn rewrite(&mut self, term: Term) -> Term {
        match term.kind {
            TermKind::Lambda(Lambda { params, body, ret_ty }) => Term {
                kind: TermKind::Lambda(Lambda {
                    params,
                    body: Box::new(self.rewrite(*body)),
                    ret_ty,
                }),
                ..term
            },
            TermKind::Let {
                name,
                name_ty,
                rhs,
                body,
            } => {
                // Keep the source-shaped RHS for dependency slicing. A SOAC
                // dependency is rejected rather than duplicated.
                let dependency = (*rhs).clone();
                let rhs = self.try_hoist(*rhs, &name_ty);
                self.locals.push((name, name_ty.clone(), dependency));
                let body = self.rewrite(*body);
                self.locals.pop();
                Term {
                    kind: TermKind::Let {
                        name,
                        name_ty,
                        rhs: Box::new(rhs),
                        body: Box::new(body),
                    },
                    ..term
                }
            }
            _ => term,
        }
    }
}

/// Find the transitive local definition slice required by `rhs`. Only pure,
/// non-SOAC definitions are reproducible in a separate dispatch; anything else
/// makes the candidate stay inline.
impl ScalarPrepassHoister<'_> {
    fn close_over_local_deps(&mut self, rhs: &Term) -> Option<Term> {
        close_term_over_carryable_locals(
            rhs,
            &self.locals,
            &self.program.symbols,
            |symbol| match &self.policy {
                ScalarPrepassEntryPolicy::Graphics { tainted, uniforms } => {
                    !tainted.contains(&symbol)
                        && (uniforms.contains_key(&symbol) || self.top_level.contains(&symbol))
                }
                ScalarPrepassEntryPolicy::Compute { params, .. } => params.contains(&symbol),
            },
            self.term_ids,
        )
    }
}

impl ScalarPrepassHoister<'_> {
    fn try_hoist(&mut self, rhs: Term, result_ty: &Type<TypeName>) -> Term {
        if !is_scalar_reduction(&rhs) {
            return rhs;
        }
        let Some(prepass_body) = self.close_over_local_deps(&rhs) else {
            return rhs;
        };

        let span = rhs.span;
        let required_params = match &self.policy {
            ScalarPrepassEntryPolicy::Graphics { uniforms, .. } => {
                // A uniform may occur only inside a pulled dependency.
                collect_uniform_required_params(&prepass_body, uniforms, &self.program.symbols)
            }
            ScalarPrepassEntryPolicy::Compute { source_def, .. } => {
                compute_broadcast_required_params(&prepass_body, source_def, &self.program.symbols)
                    .expect("compute pre-pass dependencies were classified as forwardable")
            }
        };
        let binding = BindingRef::new(AUTO_STORAGE_SET, self.binding_ids.next_id());
        let name = format!("{}_prepass_{}", self.entry_name, self.added_decls.len());
        self.new_defs.push(make_entry_def(
            &name,
            interface::EntryOrigin::ScalarPrepass,
            prepass_body,
            result_ty.clone(),
            &required_params,
            vec![interface::StorageBindingDecl {
                binding,
                role: interface::StorageRole::Output,
                elem_ty: result_ty.clone(),
                length: None,
            }],
            self.program,
            self.term_ids,
        ));
        self.added_decls.push(interface::StorageBindingDecl {
            binding,
            role: interface::StorageRole::Input,
            elem_ty: result_ty.clone(),
            length: None,
        });

        intrinsic_term_by_id(
            catalog().known().storage_index,
            vec![
                uint_lit(binding.set as u64, span, self.term_ids),
                uint_lit(binding.binding as u64, span, self.term_ids),
                uint_lit(0, span, self.term_ids),
            ],
            result_ty.clone(),
            span,
            self.term_ids,
        )
    }
}

fn close_term_over_carryable_locals(
    term: &Term,
    locals: &[LocalPrepassBinding],
    symbols: &SymbolTable,
    allow_external: impl FnMut(SymbolId) -> bool,
    term_ids: &mut TermIdSource,
) -> Option<Term> {
    let dependencies = collect_carryable_local_deps(term, locals, symbols, allow_external)?;
    Some(wrap_prepass_local_deps(term.clone(), &dependencies, term_ids))
}

fn collect_carryable_local_deps(
    term: &Term,
    locals: &[LocalPrepassBinding],
    symbols: &SymbolTable,
    mut allow_external: impl FnMut(SymbolId) -> bool,
) -> Option<Vec<LocalPrepassBinding>> {
    let local_syms: LookupSet<_> = locals.iter().map(|(symbol, _, _)| *symbol).collect();
    let empty_syms = LookupSet::new();
    let empty_defs = LookupSet::new();
    let free_symbols = |term: &Term| {
        compute_free_vars(term, &empty_syms, &empty_syms, &empty_defs, symbols)
            .into_iter()
            .filter_map(|term| match term.kind {
                TermKind::Var(VarRef::Symbol(symbol)) => Some(symbol),
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    let mut needed = LookupSet::new();
    let mut classify = |symbol, needed: &mut LookupSet<SymbolId>| {
        if local_syms.contains(&symbol) {
            needed.insert(symbol);
            true
        } else {
            allow_external(symbol)
        }
    };
    for symbol in free_symbols(term) {
        if !classify(symbol, &mut needed) {
            return None;
        }
    }

    let mut dependencies = Vec::new();
    for (symbol, ty, rhs) in locals.iter().rev() {
        if !needed.remove(symbol) {
            continue;
        }
        if !is_pullable_prepass_dependency(rhs) {
            return None;
        }
        for free_symbol in free_symbols(rhs) {
            if !classify(free_symbol, &mut needed) {
                return None;
            }
        }
        dependencies.push((*symbol, ty.clone(), rhs.clone()));
    }
    if !needed.is_empty() {
        return None;
    }
    dependencies.reverse();
    Some(dependencies)
}

fn is_pullable_prepass_dependency(term: &Term) -> bool {
    if matches!(term.kind, TermKind::Soac(_) | TermKind::OutputSlotStore { .. }) {
        return false;
    }
    let mut pullable = true;
    term.for_each_child(&mut |child| {
        if pullable {
            pullable = is_pullable_prepass_dependency(child);
        }
    });
    pullable
}

fn wrap_prepass_local_deps(
    mut body: Term,
    dependencies: &[(SymbolId, Type<TypeName>, Term)],
    term_ids: &mut TermIdSource,
) -> Term {
    for (symbol, ty, rhs) in dependencies.iter().rev() {
        body = Term {
            id: term_ids.next_id(),
            ty: body.ty.clone(),
            span: rhs.span,
            kind: TermKind::Let {
                name: *symbol,
                name_ty: ty.clone(),
                rhs: Box::new(rhs.clone()),
                body: Box::new(body),
            },
        };
    }
    body
}

fn compute_entry_can_broadcast_scalar_prepasses(def: &Def, symbols: &SymbolTable) -> bool {
    let Some(analysis) = analyze_entry(def, symbols) else {
        return false;
    };
    matches!(analysis.soac.original, SoacOp::Map { .. } | SoacOp::Scan { .. })
}

fn compute_broadcast_required_params(
    term: &Term,
    source_def: &Def,
    symbols: &SymbolTable,
) -> Option<Vec<RequiredParam>> {
    let empty_syms = LookupSet::new();
    let empty_defs = LookupSet::new();
    let free = compute_free_vars(term, &empty_syms, &empty_syms, &empty_defs, symbols);

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
    let decl = match &source_def.meta {
        DefMeta::EntryPoint(decl) => decl,
        _ => return None,
    };
    let (orig_params, _) = peel_lambda_params(&source_def.body);
    let orig_param_syms: LookupSet<SymbolId> = orig_params.iter().map(|(sym, _)| *sym).collect();
    if !free_syms.iter().all(|sym| orig_param_syms.contains(sym)) {
        return None;
    }

    Some(
        orig_params
            .iter()
            .enumerate()
            .filter(|(_, (sym, _))| free_syms.contains(sym))
            .map(|(i, (sym, ty))| RequiredParam {
                sym: *sym,
                ty: ty.clone(),
                attr: decl.params.get(i).and_then(forwardable_binding_attribute),
                binding: decl.param_bindings.get(i).cloned().flatten(),
            })
            .collect(),
    )
}

/// A scalar-result reduction: either a bare `Reduce`, or a fused `map → reduce`
/// — a scalar-output, single-`Reduce`, no-map `Screma`.
fn is_scalar_reduction(term: &Term) -> bool {
    match &term.kind {
        TermKind::Soac(SoacOp::Reduce { .. }) => true,
        TermKind::Soac(SoacOp::Screma {
            lanes, accumulators, ..
        }) => super::is_scalar_reduce_screma(lanes, accumulators),
        _ => false,
    }
}

/// Compute the *transitive* taint set of symbols that depend on entry
/// params. Starts from `entry_params` and grows by walking the body's
/// `Let` chain in source order: a let-bound symbol joins the set iff
/// its RHS has any free var already in the set. Stops descending once
/// the term is neither a Lambda nor a Let — the tail isn't a hoist
/// site, so taint propagation past it doesn't matter.
fn compute_taint_set(
    term: &Term,
    entry_params: &LookupSet<SymbolId>,
    symbols: &SymbolTable,
) -> LookupSet<SymbolId> {
    let mut tainted = entry_params.clone();
    walk_taint(term, &mut tainted, symbols);
    tainted
}

fn walk_taint(term: &Term, tainted: &mut LookupSet<SymbolId>, symbols: &SymbolTable) {
    match &term.kind {
        TermKind::Lambda(lam) => {
            walk_taint(&lam.body, tainted, symbols);
        }
        TermKind::Let { name, rhs, body, .. } => {
            if rhs_references_entry_param(rhs, tainted, symbols) {
                tainted.insert(*name);
            }
            walk_taint(body, tainted, symbols);
        }
        _ => {}
    }
}

/// True if `term` has any free SymbolId that's in the given taint set
/// (entry params plus everything transitively derived from them).
fn rhs_references_entry_param(
    term: &Term,
    entry_params: &LookupSet<SymbolId>,
    symbols: &SymbolTable,
) -> bool {
    let empty_syms = LookupSet::new();
    let empty_defs = LookupSet::new();
    let free = compute_free_vars(term, &empty_syms, &empty_syms, &empty_defs, symbols);
    free.iter().any(|t| matches!(&t.kind, TermKind::Var(VarRef::Symbol(s)) if entry_params.contains(s)))
}

/// Collect `term`'s free vars that are uniform entry params, as
/// `(symbol, type, binding)`. Re-declares the uniforms a hoisted SOAC
/// reads on the generated pre-pass entry. Deduplicated by symbol.
fn collect_uniform_required_params(
    term: &Term,
    uniform_params: &LookupMap<SymbolId, BindingRef>,
    symbols: &SymbolTable,
) -> Vec<RequiredParam> {
    let empty_syms = LookupSet::new();
    let empty_defs = LookupSet::new();
    let free = compute_free_vars(term, &empty_syms, &empty_syms, &empty_defs, symbols);

    let mut out: Vec<RequiredParam> = Vec::new();
    let mut added: LookupSet<SymbolId> = LookupSet::new();
    for t in &free {
        if let TermKind::Var(VarRef::Symbol(s)) = &t.kind {
            if let Some(binding) = uniform_params.get(s) {
                if added.insert(*s) {
                    out.push(RequiredParam {
                        sym: *s,
                        ty: t.ty.clone(),
                        attr: Some(Attribute::Uniform {
                            set: binding.set,
                            binding: binding.binding,
                        }),
                        binding: None,
                    });
                }
            }
        }
    }
    out
}

pub(crate) fn intrinsic_term_by_id(
    id: crate::builtins::BuiltinId,
    args: Vec<Term>,
    ret_ty: Type<TypeName>,
    span: ast::Span,
    term_ids: &mut TermIdSource,
) -> Term {
    let func_term = Term {
        id: term_ids.next_id(),
        ty: Type::Variable(0),
        span,
        kind: TermKind::Var(VarRef::Builtin { id, overload_idx: 0 }),
    };
    Term {
        id: term_ids.next_id(),
        ty: ret_ty,
        span,
        kind: TermKind::App {
            func: Box::new(func_term),
            args,
        },
    }
}

fn uint_lit(val: u64, span: ast::Span, term_ids: &mut TermIdSource) -> Term {
    Term {
        id: term_ids.next_id(),
        ty: Type::Constructed(TypeName::UInt(32), vec![]),
        span,
        kind: TermKind::IntLit(val.to_string()),
    }
}

#[cfg(test)]
#[path = "parallelize_tests.rs"]
mod parallelize_tests;
