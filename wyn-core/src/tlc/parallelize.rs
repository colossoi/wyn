//! TLC-level SOAC parallelization.
//!
//! Stage A: Analyze compute entry points to find parallelizable SOACs.
//! Stage B: Restructure the program — create new entry points with chunked SOACs,
//!          allocate intermediate storage buffers, build pipeline descriptor.
//!
//! Loop creation and storage lowering stay in SSA (`to_ssa` + `soac_lower`).

use super::VarRef;
use super::closure_convert::collect_free_vars;
use crate::ast::{self, TypeName};
use crate::builtins::catalog;
use crate::egir::from_tlc::AUTO_STORAGE_SET;
use crate::interface::{self, Attribute, EntryParamBinding, EntryParamBindingKind};
use crate::pipeline_descriptor::*;
use crate::{BindingRef, SymbolId, SymbolTable};
use polytype::Type;
use std::collections::{HashMap, HashSet};

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
    /// Map / Reduce / Redomap / Scan.
    Opaque,
}

/// A parallelizable SOAC found in a compute entry point.
///
/// Holds the original `SoacOp` — callers that need per-variant logic
/// pattern-match `original`. `provenances` carries one entry per input
/// (length 1 for Reduce/Scan, N for Map/Redomap with N inputs).
///
/// `analyze_soac` is the only constructor and guarantees `original` is
/// one of `Map`/`Reduce`/`Redomap`/`Scan` — the non-parallelizable
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
    Redomap,
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
        SoacOp::Redomap { ne, inputs, .. } => Some(ParallelSoacShape {
            flavor: ParallelSoacFlavor::Redomap,
            inputs: inputs.iter().collect(),
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
            map_lams,
            accumulators,
        } => {
            let result_elem_type = accumulators
                .first()
                .map(|acc| acc.ne.ty.clone())
                .or_else(|| map_lams.first().map(|lam| lam.lam.ret_ty.clone()))
                .unwrap_or_else(|| Type::Constructed(TypeName::Unit, vec![]));
            // Pointwise (no accumulators, all maps) routes through the
            // multi-output map path. Mixed Screma (accumulators + maps) is
            // recognised by analysis so make_lowering_plan can dispatch a
            // serial single-thread compute pipeline — the EGIR-side
            // parallel transform lights up in a follow-up. A Screma with
            // no maps and no accumulators is meaningless; reject it.
            let lowerable_today = !inputs.is_empty() && (!accumulators.is_empty() || !map_lams.is_empty());
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

    /// Neutral/initial value — present for Reduce/Redomap/Scan, `None` for Map.
    pub fn ne(&self) -> Option<&Term> {
        parallel_soac_shape(&self.original).expect("non-parallel SOAC in SoacAnalysis").ne
    }

    /// Element type of one iteration's output — Map/Scan elem type (from the
    /// per-element lambda's ret_ty); Reduce/Redomap acc type (from `ne.ty`).
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
    pub def_name: SymbolId,
    pub soac: SoacAnalysis,
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
}

// =============================================================================
// Stage A: Analysis
// =============================================================================

fn analyze_program(program: &Program) -> HashMap<SymbolId, EntryAnalysis> {
    let mut results = HashMap::new();

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

    // The entry's binding layout, which resolves `Ref(Var(sym))` SOAC inputs
    // back to their assigned (set, binding). Empty for non-compute entries.
    let entry_slots = if let DefMeta::EntryPoint(decl) = &def.meta {
        let (params, _) = peel_lambda_params(&def.body);
        crate::binding_layout::compute_entry_binding_layout(&params, decl, AUTO_STORAGE_SET)
    } else {
        Vec::new()
    };

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
                    // stores ending in `UnitLit`). Walk it here so the
                    // sibling slot values reach phase synthesis; the
                    // analysis itself descends into slot 0's value.
                    collect_extra_slot_stores(&body, &mut extra_slots);
                    if extra_slots.iter().any(|(_, v)| matches!(v.kind, TermKind::Soac(_))) {
                        // Sibling slot computed by another SOAC would
                        // need its own parallel kernel; not handled by
                        // the two-phase synthesis yet — bail and let
                        // the entry stay unparallelized.
                        return None;
                    }
                    current = *value;
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
                    prefix_lets,
                    required_params,
                    extra_slots,
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
                    Some((_ty, rhs)) => current = rhs,
                    None => return None,
                }
            }
            TermKind::TupleProj { tuple, idx: 0 } => {
                let TermKind::Var(VarRef::Symbol(sym)) = tuple.kind else {
                    return None;
                };
                if scope.is_lambda_param(sym) {
                    return None;
                }
                match scope.remove_let(sym) {
                    Some((_ty, rhs)) if matches!(rhs.kind, TermKind::Soac(SoacOp::Screma { .. })) => {
                        current = rhs;
                    }
                    _ => return None,
                }
            }
            _ => return None,
        }
    }
}

/// Walk the body half of slot 0's sequencing let, picking off each
/// `OutputSlotStore { slot_index, value, .. }` along the let-chain
/// into `out`. The chain shape from `normalize_outputs` is `let _seq =
/// OutputSlotStore(i, v) in <rest>` terminating in `UnitLit`.
fn collect_extra_slot_stores(term: &Term, out: &mut Vec<(usize, Term)>) {
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
    use std::collections::HashSet;
    let empty_top: HashSet<SymbolId> = HashSet::new();
    let empty_defs: HashSet<String> = HashSet::new();
    let mut bound: HashSet<SymbolId> = HashSet::new();
    let mut free: Vec<Term> = Vec::new();
    let mut seen: HashSet<SymbolId> = HashSet::new();

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

    let free_syms: HashSet<SymbolId> = free
        .iter()
        .filter_map(
            |t| {
                if let TermKind::Var(VarRef::Symbol(s)) = &t.kind { Some(*s) } else { None }
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
            | ParallelSoacFlavor::Redomap
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
        SoacOp::Redomap {
            op,
            reduce_op,
            ne,
            inputs,
        } => SoacOp::Redomap {
            op: op.clone(),
            reduce_op: reduce_op.clone(),
            ne: ne.clone(),
            inputs: inputs.clone(),
        },
        SoacOp::Screma {
            map_lams,
            accumulators,
            inputs,
        } => SoacOp::Screma {
            map_lams: map_lams.clone(),
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
    // path rediscovers their concrete shape. Reduce/Scan/Redomap reject
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
        ArrayExpr::Ref(t) => {
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
            // `iota(N)` desugars to
            // `let arg = N in Range{ start: 0, len: arg - 0, step: None }`.
            // The dispatch length is `N` itself; lift it out as the Range bound.
            if let Some(bound) = extract_iota_bound(t) {
                return Some(ArrayProvenance::Range { bound });
            }
            None
        }
        _ => None,
    }
}

/// Recognize the `iota(N)` desugaring
/// `let arg = N in Range { start: 0, len: arg - 0 }` and recover `N`.
fn extract_iota_bound(t: &Term) -> Option<Term> {
    let TermKind::Let { name, rhs, body, .. } = &t.kind else {
        return None;
    };
    let TermKind::ArrayExpr(ArrayExpr::Range { start, len, .. }) = &body.kind else {
        return None;
    };
    let TermKind::App { func, args } = &len.kind else {
        return None;
    };
    let [arg, zero] = args.as_slice() else { return None };
    let TermKind::BinOp(op) = &func.kind else {
        return None;
    };
    let arg_is_name = matches!(&arg.kind, TermKind::Var(VarRef::Symbol(s)) if s == name);
    (op.op == "-" && is_zero_int(start) && is_zero_int(zero) && arg_is_name).then(|| (**rhs).clone())
}

fn is_zero_int(t: &Term) -> bool {
    matches!(&t.kind, TermKind::IntLit(s) if s == "0")
}

// =============================================================================
// Stage B: Restructuring
// =============================================================================

const DEFAULT_WORKGROUP_X: u32 = 64;

/// Workgroup count for a reduce/redomap phase 1 grid. The grid is fixed
/// (not derived from input length) and sized to roughly saturate the GPU;
/// the kernel grid-strides over the input via the `num_workgroups`
/// intrinsic, so this bounds the `partials` count to one per worker
/// (`PHASE1_SATURATING_GROUPS * workgroup_width`) regardless of input size.
const PHASE1_SATURATING_GROUPS: u32 = 1024;

/// Per-entry pipeline sizing derived from `#[size_hint(N)]`. The
/// `workgroup` drives the shader's `local_size` and the chunk-arithmetic
/// `total_threads`; `default_total_threads` ships to the pipeline
/// descriptor as a host-runtime dispatch default.
#[derive(Clone, Copy)]
struct PipelineSizing {
    workgroup: (u32, u32, u32),
    default_total_threads: Option<std::num::NonZeroU32>,
}

impl PipelineSizing {
    /// Sizing for an entry being parallelized — `analysis` lets us
    /// pick the size_hint of the array that actually drives the
    /// dispatch (the SOAC's input).
    fn for_analyzed_entry(program: &Program, analysis: &EntryAnalysis) -> Self {
        let hint = entry_size_hint(program, analysis.def_name, Some(analysis));
        PipelineSizing {
            workgroup: pick_workgroup_size(hint),
            default_total_threads: hint,
        }
    }

    /// Sizing for an entry with no SOAC analysis (a non-parallelized
    /// compute entry or the `disable=true` fast path). Falls back to
    /// the first view-array param's hint.
    fn for_default_entry(program: &Program, def_name: SymbolId) -> Self {
        let hint = entry_size_hint(program, def_name, None);
        PipelineSizing {
            workgroup: pick_workgroup_size(hint),
            default_total_threads: hint,
        }
    }
}

/// Pick the compute-shader workgroup size (X, 1, 1) for a parallelized
/// entry based on its (optional) `#[size_hint(N)]`. Interpretation A
/// from `issues/size-hint-design.md`: the hint is a host-runtime
/// default for dispatch sizing and a compile-time signal for
/// workgroup-size selection. The hint is not load-bearing for
/// correctness; only for picking a better workgroup size.
///
/// Buckets:
/// - hint < 64       → `next_power_of_two(hint)`
/// - hint in 64..=64K → 64 (current default)
/// - hint > 64K       → 256 for better occupancy
/// - None            → 64
fn pick_workgroup_size(size_hint: Option<std::num::NonZeroU32>) -> (u32, u32, u32) {
    let x = match size_hint.map(|n| n.get()) {
        Some(n) if n < 64 => n.next_power_of_two(),
        Some(n) if n <= 65_536 => 64,
        Some(_) => 256,
        None => DEFAULT_WORKGROUP_X,
    };
    (x, 1, 1)
}

/// Lookup the `#[size_hint(N)]` of the array that drives this entry's
/// dispatch — i.e., the SOAC's input array. Other params' hints (e.g.
/// `#[size_hint(8)]` on a small lookup table alongside the main input)
/// are intentionally ignored, since they don't drive workgroup-size
/// selection or the host-runtime default thread count.
///
/// Identification: `buffer_specialize` assigns set=0, binding=k to the
/// k-th view-array entry param in source order. The SOAC's input
/// provenance reports the (set, binding) it reads from. Match the two.
fn entry_size_hint(
    program: &Program,
    def_name: SymbolId,
    analysis: Option<&EntryAnalysis>,
) -> Option<std::num::NonZeroU32> {
    let def = program.defs.iter().find(|d| d.name == def_name)?;
    let decl = match &def.meta {
        DefMeta::EntryPoint(decl) => decl,
        _ => return None,
    };
    // If we have a SOAC analysis, prefer the param that maps to the
    // SOAC's first storage input. Use buffer_specialize's binding-
    // assignment scheme (view-array params get sequential bindings on
    // set=0 in source order) to translate.
    if let Some(analysis) = analysis {
        match analysis.soac.provenances.first() {
            Some(ArrayProvenance::Storage { binding, .. }) if binding.set == 0 => {
                let mut idx: u32 = 0;
                for p in &decl.params {
                    if pattern_binds_view_array(p) {
                        if idx == binding.binding {
                            return crate::egir::from_tlc::extract_size_hint(p);
                        }
                        idx += 1;
                    }
                }
                return None;
            }
            Some(ArrayProvenance::Opaque) => {
                return decl.params.iter().filter_map(crate::egir::from_tlc::extract_size_hint).next();
            }
            _ => return None,
        }
    }
    // No analysis (non-parallelized compute entry): fall back to the
    // first view-array param's hint. There's no canonical "SOAC input"
    // here; this is the most conservative choice.
    decl.params.iter().filter_map(crate::egir::from_tlc::extract_size_hint).next()
}

/// True iff the pattern declares a view-typed array (i.e. `[]T` in
/// source — the param shape that `buffer_specialize` rewrites into an
/// `(offset, len)` pair). Mirrors `buffer_specialize::is_view_array`
/// at the AST level.
fn pattern_binds_view_array(p: &crate::ast::Pattern) -> bool {
    use crate::types::TypeExt;
    let Some(ty) = p.pattern_type() else { return false };
    if !ty.is_array() {
        return false;
    }
    let is_view = ty
        .array_variant()
        .map(|v| matches!(v, Type::Constructed(TypeName::ArrayVariantView, _)))
        .unwrap_or(false);
    let is_unsized = ty.array_size().map(|s| matches!(s, Type::Variable(_))).unwrap_or(false);
    is_view && is_unsized
}

pub struct ParallelizationResult {
    pub program: Program,
    pub pipeline: PipelineDescriptor,
    /// Compiler-internal per-entry plans for EGIR to consume. Keyed by the
    /// entry's surface name (matches `egir::EgirEntry::name`). Empty for
    /// graphics entries, non-parallelized compute entries, and (today)
    /// reduce/redomap/scan entries — those still lower through the old
    /// TLC path. Map planning is the only strategy populated until the
    /// EGIR migration broadens.
    pub plans: HashMap<String, ParallelizationPlan>,
}

/// Per-entry, compiler-internal description of how EGIR should lower a
/// parallelized SOAC. The plan stays declarative: it picks a strategy,
/// declares the dispatch shape, and reserves binding numbers. It does
/// not encode element access — EGIR rediscovers the SOAC's inputs,
/// captures, lambda, and output view from the entry body itself.
#[derive(Debug, Clone)]
pub struct ParallelizationPlan {
    /// Entry surface name; matches `egir::EgirEntry::name`.
    pub entry: String,
    pub dispatch: DispatchModel,
    pub bindings: PlannedBindings,
}

/// Dispatch shape from the planner + sizing. Carried explicitly so
/// the host pipeline descriptor, the planning record, and the generated
/// kernel can be verified to agree.
#[derive(Debug, Clone, Copy)]
pub enum DispatchModel {
    /// Host computes `ceil(N / workgroup_size)` groups at submit time,
    /// where `N` is the length of the SOAC's `input_index`th input.
    DerivedFromInputLength {
        input_index: usize,
        workgroup_size: u32,
    },
    /// Fixed group + local size. Used by reduce/scan combine phases.
    Fixed {
        groups: [u32; 3],
        local_size: [u32; 3],
    },
}

/// Unified Screma binding layout that covers every parallel SOAC
/// shape the EGIR-native path emits. Pointwise Map collapses to
/// `{ map_outputs: [_], accumulators: [] }`; Reduce/Redomap collapses
/// to `{ map_outputs: [], accumulators: [Reduce{partials, result}] }`;
/// Scan collapses to `{ map_outputs: [], accumulators: [Scan{output,
/// block_sums, block_offsets}] }`; mixed Screma populates both.
#[derive(Debug, Clone)]
pub struct PlannedBindings {
    /// One entry per mapped output. `None` means EGIR auto-allocates the
    /// entry output view for that field.
    pub map_outputs: Vec<Option<BindingRef>>,
    pub accumulators: Vec<PlannedScremaAccumulator>,
}

#[derive(Debug, Clone)]
pub struct PlannedScremaAccumulator {
    pub kind: PlannedScremaAccumulatorKind,
    pub partials: Option<BindingRef>,
    pub result: Option<BindingRef>,
    pub output: Option<BindingRef>,
    pub block_sums: Option<BindingRef>,
    pub block_offsets: Option<BindingRef>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlannedScremaAccumulatorKind {
    Reduce,
    Scan,
}

impl PlannedBindings {
    /// The pinned result binding for shapes whose result is a single
    /// `EntryOutput` buffer (pointwise Screma == Map, single-Scan Screma).
    /// `from_tlc::build_entry_outputs` honors this so the EGIR output lands
    /// on the buffer the consumer reads. All-Reduce Screma manages its
    /// result bindings via partials+result allocations (Stores, not
    /// `EntryOutput`), so it reports `None`.
    pub fn forced_output(&self) -> Option<BindingRef> {
        // Pointwise Screma: forced output from the single map slot.
        if self.accumulators.is_empty() && self.map_outputs.len() == 1 {
            return self.map_outputs[0];
        }
        // Single Scan accumulator: forced output is the scan's output binding.
        if self.accumulators.len() == 1
            && matches!(self.accumulators[0].kind, PlannedScremaAccumulatorKind::Scan)
        {
            return self.accumulators[0].output;
        }
        None
    }
}

// =============================================================================
// Graphical-entry SOAC lifting
// =============================================================================

/// For each graphical entry, walk its body's outer let-chain and hoist
/// reduce/redomap bindings whose RHS is invariant with respect to the
/// entry's per-invocation params. Each lift:
///   * allocates a fresh storage buffer for the scalar result,
///   * emits a compute pre-pass entry `<entry>_prepass_<n>` that
///     evaluates the SOAC and stores the result at index 0,
///   * rewrites the original let-binding's RHS to
///     `_w_intrinsic_storage_index(set, binding, 0)`,
///   * adds an `Input`-role `StorageBindingDecl` to the graphical
///     entry's interface so the backend's binding allowlist admits
///     the load.
///
/// The pre-pass entries land in `program.defs` and will be picked up
/// by `analyze_program` + Stage B in the usual way, producing the
/// two-phase compute pipeline that justifies "multi-stage" — one
/// source file compiles to a chunk/combine pair plus the original
/// vertex+fragment stages.
///
/// Scope (MVP): only reduce/redomap whose result is a scalar. Scan/Map
/// (array results) and deeply nested lets are left for a follow-up.
enum ScalarPrepassPolicy {
    GraphicalInvariant {
        tainted: HashSet<SymbolId>,
        uniform_params: HashMap<SymbolId, BindingRef>,
    },
    ComputeBroadcast {
        source_def: Def,
    },
}

fn lift_scalar_soac_prepasses(
    program: &mut Program,
    next_binding: &mut u32,
    prepass_result_bindings: &mut HashMap<SymbolId, BindingRef>,
    term_ids: &mut TermIdSource,
) {
    use std::collections::HashSet;

    // Snapshot indices of graphical entry defs — we'll mutate program.defs
    // in the loop, but only the def at `idx` (its body + storage_bindings).
    let indices: Vec<usize> = program
        .defs
        .iter()
        .enumerate()
        .filter_map(|(i, d)| match &d.meta {
            DefMeta::EntryPoint(_) => Some(i),
            _ => None,
        })
        .collect();

    let mut new_defs: Vec<Def> = Vec::new();

    for idx in indices {
        let entry_name = crate::symbol_name_or_bug(&program.symbols, program.defs[idx].name).to_string();

        let body = program.defs[idx].body.clone();
        let policy = match &program.defs[idx].meta {
            DefMeta::EntryPoint(decl) if decl.entry_type.is_compute() => {
                if !compute_entry_can_broadcast_scalar_prepasses(&program.defs[idx], &program.symbols) {
                    continue;
                }
                ScalarPrepassPolicy::ComputeBroadcast {
                    source_def: program.defs[idx].clone(),
                }
            }
            DefMeta::EntryPoint(_) => {
                let decl_params: Vec<ast::Pattern> = match &program.defs[idx].meta {
                    DefMeta::EntryPoint(decl) => decl.params.clone(),
                    _ => Vec::new(),
                };
                let (peeled, _) = peel_lambda_params(&body);
                let mut entry_params: HashSet<SymbolId> = HashSet::new();
                let mut uniform_params: HashMap<SymbolId, BindingRef> = HashMap::new();
                for (i, (sym, _)) in peeled.iter().enumerate() {
                    match decl_params.get(i).and_then(crate::binding_layout::extract_uniform_binding) {
                        Some(br) => {
                            uniform_params.insert(*sym, br);
                        }
                        None => {
                            entry_params.insert(*sym);
                        }
                    }
                }
                ScalarPrepassPolicy::GraphicalInvariant {
                    tainted: compute_taint_set(&body, &entry_params, &program.symbols),
                    uniform_params,
                }
            }
            _ => continue,
        };
        let mut added_decls: Vec<interface::StorageBindingDecl> = Vec::new();
        let new_body = lift_in_term(
            body,
            &entry_name,
            &policy,
            next_binding,
            &mut added_decls,
            &mut new_defs,
            prepass_result_bindings,
            program,
            term_ids,
        );

        program.defs[idx].body = new_body;
        if let DefMeta::EntryPoint(ref mut decl) = program.defs[idx].meta {
            decl.storage_bindings.extend(added_decls);
        }
    }

    program.defs.extend(new_defs);
}

/// Return a `Term`'s (possibly-wrapped) lambda params by peeling
/// outer `TermKind::Lambda` layers. Mirrors `extract_params` in
/// `buffer_specialize.rs`.
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
/// computation and isn't a lift site.
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
    let bound: HashSet<SymbolId> = HashSet::new();
    let empty_top: HashSet<SymbolId> = HashSet::new();
    let empty_defs: HashSet<String> = HashSet::new();
    let mut free: Vec<Term> = Vec::new();
    let mut seen: HashSet<SymbolId> = HashSet::new();
    collect_free_vars(
        term,
        &bound,
        &empty_top,
        &empty_defs,
        symbols,
        &mut free,
        &mut seen,
    );

    let free_syms: HashSet<SymbolId> = free
        .iter()
        .filter_map(
            |t| {
                if let TermKind::Var(VarRef::Symbol(s)) = &t.kind { Some(*s) } else { None }
            },
        )
        .collect();
    let decl = match &source_def.meta {
        DefMeta::EntryPoint(decl) => decl,
        _ => return None,
    };
    let (orig_params, _) = peel_lambda_params(&source_def.body);
    let orig_param_syms: HashSet<SymbolId> = orig_params.iter().map(|(sym, _)| *sym).collect();
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

fn lift_in_term(
    term: Term,
    entry_name: &str,
    policy: &ScalarPrepassPolicy,
    next_binding: &mut u32,
    added_decls: &mut Vec<interface::StorageBindingDecl>,
    new_defs: &mut Vec<Def>,
    prepass_result_bindings: &mut HashMap<SymbolId, BindingRef>,
    program: &mut Program,
    term_ids: &mut TermIdSource,
) -> Term {
    match term.kind {
        TermKind::Lambda(lam) => {
            let Lambda { params, body, ret_ty } = lam;
            let new_body = lift_in_term(
                *body,
                entry_name,
                policy,
                next_binding,
                added_decls,
                new_defs,
                prepass_result_bindings,
                program,
                term_ids,
            );
            Term {
                id: term.id,
                ty: term.ty,
                span: term.span,
                kind: TermKind::Lambda(Lambda {
                    params,
                    body: Box::new(new_body),
                    ret_ty,
                }),
            }
        }
        TermKind::Let {
            name,
            name_ty,
            rhs,
            body,
        } => {
            let new_rhs = maybe_hoist(
                *rhs,
                entry_name,
                &name_ty,
                policy,
                next_binding,
                added_decls,
                new_defs,
                prepass_result_bindings,
                program,
                term_ids,
            );
            let new_body = lift_in_term(
                *body,
                entry_name,
                policy,
                next_binding,
                added_decls,
                new_defs,
                prepass_result_bindings,
                program,
                term_ids,
            );
            Term {
                id: term.id,
                ty: term.ty,
                span: term.span,
                kind: TermKind::Let {
                    name,
                    name_ty,
                    rhs: Box::new(new_rhs),
                    body: Box::new(new_body),
                },
            }
        }
        _ => term,
    }
}

/// If `rhs` is a scalar-returning SOAC (reduce/redomap) whose free
/// vars don't reference entry params, allocate a storage binding,
/// emit a pre-pass compute entry, and replace `rhs` with a load of
/// that binding. Otherwise return `rhs` unchanged.
fn maybe_hoist(
    rhs: Term,
    entry_name: &str,
    name_ty: &Type<TypeName>,
    policy: &ScalarPrepassPolicy,
    next_binding: &mut u32,
    added_decls: &mut Vec<interface::StorageBindingDecl>,
    new_defs: &mut Vec<Def>,
    prepass_result_bindings: &mut HashMap<SymbolId, BindingRef>,
    program: &mut Program,
    term_ids: &mut TermIdSource,
) -> Term {
    // TODO: extend to array-result SOACs (Scan, Map). A pre-pass
    // emitting an array would need to write N slots to storage, the
    // fragment would read back by index instead of at position 0, and
    // Stage B's two-phase plan would grow an array-sized output path.
    // For the scalar-result cases (Reduce, Redomap) the single-slot
    // shape already works end-to-end.
    let is_scalar_soac = matches!(
        &rhs.kind,
        TermKind::Soac(SoacOp::Reduce { .. }) | TermKind::Soac(SoacOp::Redomap { .. })
    );
    if !is_scalar_soac {
        return rhs;
    }

    let required_params = match policy {
        ScalarPrepassPolicy::GraphicalInvariant {
            tainted,
            uniform_params,
        } => {
            if rhs_references_entry_param(&rhs, tainted, &program.symbols) {
                return rhs;
            }
            assert_hoist_free_vars_are_grounded(&rhs, tainted, &program.symbols);
            collect_uniform_required_params(&rhs, uniform_params, &program.symbols)
        }
        ScalarPrepassPolicy::ComputeBroadcast { source_def } => {
            match compute_broadcast_required_params(&rhs, source_def, &program.symbols) {
                Some(required_params) => required_params,
                None => return rhs,
            }
        }
    };

    // Invariance check: none of `rhs`'s free vars may be an entry param.
    // TODO: polymorphic-size free vars. Free vars whose type contains
    // a Size type variable (e.g. `iota(N)` where `N` is a `<[n]>`
    // parameter) pass the entry-param check — `N` isn't an entry
    // param — but the generated pre-pass doesn't have `N` in scope
    // either. Silently emitting the lift in that state produces a
    // pre-pass that references `@size` as an undeclared global and
    // fails backend validation. Panic loudly instead until the lift
    // either (a) refuses the hoist when polymorphic sizes are present
    // or (b) captures the size binding alongside the hoisted SOAC.
    // Pre-allocate the binding the fragment will load from. Stage B's
    // make_two_phase_plan will use this as the prepass's result_binding
    // (via the prepass_result_bindings map), so phase 2's final store
    // goes exactly here.
    let binding = BindingRef::new(AUTO_STORAGE_SET, *next_binding);
    *next_binding += 1;

    // Capture the SOAC's uniform free vars. The pre-pass is a separate
    // entry, so it must re-declare each uniform it reads as its own
    // `#[uniform]` param at the same (set, binding) the original entry used.
    let span = rhs.span;
    let prepass_name = format!("{}_prepass_{}", entry_name, added_decls.len());
    let prepass_def = build_prepass_def(
        &prepass_name,
        rhs,
        name_ty.clone(),
        &required_params,
        program,
        term_ids,
    );
    prepass_result_bindings.insert(prepass_def.name, binding);
    new_defs.push(prepass_def);

    added_decls.push(interface::StorageBindingDecl {
        binding,
        role: interface::StorageRole::Input,
        elem_ty: name_ty.clone(),
        length: None,
    });

    // Rewrite the let RHS to a storage load at position 0.
    intrinsic_term_by_id(
        catalog().known().storage_index,
        vec![
            uint_lit(binding.set as u64, span, term_ids),
            uint_lit(binding.binding as u64, span, term_ids),
            uint_lit(0, span, term_ids),
        ],
        name_ty.clone(),
        span,
        term_ids,
    )
}

/// Compute the *transitive* taint set of symbols that depend on entry
/// params. Starts from `entry_params` and grows by walking the body's
/// `Let` chain in source order: a let-bound symbol joins the set iff
/// its RHS has any free var already in the set. Stops descending once
/// the term is neither a Lambda nor a Let — the tail isn't a hoist
/// site, so taint propagation past it doesn't matter.
fn compute_taint_set(
    term: &Term,
    entry_params: &std::collections::HashSet<SymbolId>,
    symbols: &SymbolTable,
) -> std::collections::HashSet<SymbolId> {
    let mut tainted = entry_params.clone();
    walk_taint(term, &mut tainted, symbols);
    tainted
}

fn walk_taint(term: &Term, tainted: &mut std::collections::HashSet<SymbolId>, symbols: &SymbolTable) {
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
/// Uses `closure_convert::collect_free_vars` with empty
/// `top_level`/`known_defs` sets (same style as
/// `compute_required_params`).
fn rhs_references_entry_param(
    term: &Term,
    entry_params: &std::collections::HashSet<SymbolId>,
    symbols: &SymbolTable,
) -> bool {
    use std::collections::HashSet;
    let bound: HashSet<SymbolId> = HashSet::new();
    let empty_top: HashSet<SymbolId> = HashSet::new();
    let empty_defs: HashSet<String> = HashSet::new();
    let mut free: Vec<Term> = Vec::new();
    let mut seen: HashSet<SymbolId> = HashSet::new();
    collect_free_vars(
        term,
        &bound,
        &empty_top,
        &empty_defs,
        symbols,
        &mut free,
        &mut seen,
    );
    free.iter().any(|t| matches!(&t.kind, TermKind::Var(VarRef::Symbol(s)) if entry_params.contains(s)))
}

/// Collect `term`'s free vars that are uniform entry params, as
/// `(symbol, type, binding)`. Re-declares the uniforms a hoisted SOAC
/// reads on the generated pre-pass entry. Deduplicated by symbol.
fn collect_uniform_required_params(
    term: &Term,
    uniform_params: &HashMap<SymbolId, BindingRef>,
    symbols: &SymbolTable,
) -> Vec<RequiredParam> {
    let bound: HashSet<SymbolId> = HashSet::new();
    let empty_top: HashSet<SymbolId> = HashSet::new();
    let empty_defs: HashSet<String> = HashSet::new();
    let mut free: Vec<Term> = Vec::new();
    let mut seen: HashSet<SymbolId> = HashSet::new();
    collect_free_vars(
        term,
        &bound,
        &empty_top,
        &empty_defs,
        symbols,
        &mut free,
        &mut seen,
    );

    let mut out: Vec<RequiredParam> = Vec::new();
    let mut added: HashSet<SymbolId> = HashSet::new();
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

/// Panic if any free variable of `term` has a type that carries an
/// unresolved Size type variable. See the TODO at `maybe_hoist` —
/// these vars aren't in scope inside the generated pre-pass, and
/// silently emitting the lift produces a broken shader.
fn assert_hoist_free_vars_are_grounded(
    term: &Term,
    entry_params: &std::collections::HashSet<SymbolId>,
    symbols: &SymbolTable,
) {
    use std::collections::HashSet;
    let bound: HashSet<SymbolId> = HashSet::new();
    let empty_top: HashSet<SymbolId> = HashSet::new();
    let empty_defs: HashSet<String> = HashSet::new();
    let mut free: Vec<Term> = Vec::new();
    let mut seen: HashSet<SymbolId> = HashSet::new();
    collect_free_vars(
        term,
        &bound,
        &empty_top,
        &empty_defs,
        symbols,
        &mut free,
        &mut seen,
    );
    for t in &free {
        if let TermKind::Var(VarRef::Symbol(sym)) = &t.kind {
            if entry_params.contains(sym) {
                continue;
            }
            if type_contains_type_variable(&t.ty) {
                let name = symbols.get(*sym).map(|s| s.as_str()).unwrap_or("<unknown>");
                panic!(
                    "parallelize::maybe_hoist: hoisted SOAC references free var `{}` \
                     whose type `{:?}` contains an unresolved Size type variable. \
                     The generated pre-pass would reference an undeclared @size \
                     global and fail backend validation. Fix the lift site before \
                     enabling this path.",
                    name, t.ty
                );
            }
        }
    }
}

/// True if `ty` transitively contains a `Type::Variable(_)` — the
/// wyn representation of an unresolved Size (or other) type variable.
fn type_contains_type_variable(ty: &Type<TypeName>) -> bool {
    match ty {
        Type::Variable(_) => true,
        Type::Constructed(_, args) => args.iter().any(type_contains_type_variable),
    }
}

/// Build a compute entry Def whose body is the bare `soac_term` at the
/// tail. No input params, no output storage bindings declared —
/// `run()` keeps a `prepass_result_bindings` map telling Stage B's
/// `make_two_phase_plan` which result binding to use for this entry,
/// so phase 2 writes to the binding the fragment reads from.
///
/// The tail-SOAC shape is important: `analyze_entry` recognizes it as
/// a parallelizable entry and feeds it to Stage B for multi-staging.
fn build_prepass_def(
    entry_name: &str,
    soac_term: Term,
    elem_ty: Type<TypeName>,
    required_params: &[RequiredParam],
    program: &mut Program,
    term_ids: &mut TermIdSource,
) -> Def {
    make_entry_def(
        entry_name,
        soac_term,
        elem_ty,
        required_params,
        Vec::new(),
        program,
        term_ids,
    )
}

/// Parallelize SOACs in compute entry points.
///
/// `disable` short-circuits the whole pass — every compute entry runs
/// as a single sequential loop, graphical entries receive no pre-pass
/// lifting, and the pipeline descriptor is built from the untouched
/// program. Useful for debugging (keeps the SSA close to the source)
/// and for backends that can't handle multi-entry pipelines.
pub fn run(mut program: Program, disable: bool) -> crate::error::Result<ParallelizationResult> {
    if disable {
        let pipeline = build_default_pipeline(&program);
        return Ok(ParallelizationResult {
            program,
            pipeline,
            plans: HashMap::new(),
        });
    }

    // Track max binding across every `(set, binding)` the program already
    // uses — including implicit `ArrayExpr::StorageBuffer` bindings
    // introduced by lift_gathers / buffer_specialize / mono for SOAC inputs.
    // Missing these would let fresh intermediates collide with input buffers.
    let mut next_binding: u32 =
        collect_all_used_bindings(&program).iter().map(|br| br.binding + 1).max().unwrap_or(0);

    // Pass-local TermIdSource. TLC TermIds on synthesized terms aren't
    // load-bearing past parallelize, but every synthesized term still
    // gets a real ID (no `TermId(0)` placeholder).
    let mut term_ids = TermIdSource::new();

    // Hoist invariant SOACs out of graphical entry bodies into generated
    // compute pre-pass entries. Each pre-pass writes its SOAC result to
    // a fresh storage buffer; the graphical entry's body is rewritten to
    // read from that buffer. The pre-pass entries are added to
    // `program.defs` so the compute Stage A/B analysis below picks them
    // up and multi-stages them.
    //
    // `prepass_result_bindings` maps each hoisted pre-pass's def symbol
    // to the storage binding the graphical entry reads from; Stage B's
    // `make_two_phase_plan` consults this so phase 2's result store goes
    // exactly there (instead of a freshly-allocated binding).
    let mut prepass_result_bindings: HashMap<SymbolId, BindingRef> = HashMap::new();
    lift_scalar_soac_prepasses(
        &mut program,
        &mut next_binding,
        &mut prepass_result_bindings,
        &mut term_ids,
    );

    // Gather pre-passes (from the pre-defunc `lift_gathers`) pin their result
    // by declaring it as an Output-role storage binding on the entry. Fold
    // those into the same forced-binding map, so every SOAC kind's lowering
    // honors a forced result through one channel — there's no separate
    // per-kind reader.
    for def in &program.defs {
        if let DefMeta::EntryPoint(decl) = &def.meta {
            if let Some(b) =
                decl.storage_bindings.iter().find(|b| matches!(b.role, interface::StorageRole::Output))
            {
                prepass_result_bindings.entry(def.name).or_insert(b.binding);
            }
        }
    }

    let analyses = analyze_program(&program);

    if analyses.is_empty() {
        let pipeline = build_default_pipeline(&program);
        return Ok(ParallelizationResult {
            program,
            pipeline,
            plans: HashMap::new(),
        });
    }

    let mut pipelines = Vec::new();
    let mut new_defs = Vec::new();
    let mut removed_entries: Vec<SymbolId> = Vec::new();
    let mut plans: HashMap<String, ParallelizationPlan> = HashMap::new();

    // Default pipelines for non-parallelized compute entries.
    for def in &program.defs {
        if let DefMeta::EntryPoint(ref decl) = def.meta {
            if decl.entry_type.is_compute() && !analyses.contains_key(&def.name) {
                let name = crate::symbol_name_or_bug(&program.symbols, def.name).to_string();
                // For non-parallelized compute entries, the SSA-stage
                // codegen derives binding info from the entry's param
                // types. No binding info is extractable at this TLC
                // stage, so we emit an empty vec.
                let input_bindings: Vec<Binding> = vec![];
                let sizing = PipelineSizing::for_default_entry(&program, def.name);
                let len = default_entry_dispatch_len(&program, def.name);
                pipelines.push(Pipeline::Compute(ComputePipeline {
                    entry_point: name,
                    workgroup_size: sizing.workgroup,
                    dispatch_size: DispatchSize::DerivedFrom {
                        len,
                        workgroup_size: sizing.workgroup.0,
                    },
                    bindings: input_bindings,
                    default_total_threads: sizing.default_total_threads,
                }));
            }
        }
    }

    for (_sym, analysis) in &analyses {
        let entry_name = crate::symbol_name_or_bug(&program.symbols, analysis.def_name).to_string();
        let forced = prepass_result_bindings.get(&analysis.def_name).copied();
        let plan = make_lowering_plan(
            analysis,
            &entry_name,
            next_binding,
            forced,
            &mut program,
            &mut term_ids,
        );
        next_binding += plan.extra_bindings_used;
        if let Some(removed) = plan.removed_entry {
            removed_entries.push(removed);
        }
        new_defs.extend(plan.new_defs);
        pipelines.push(plan.pipeline);
        if let Some(parallel_plan) = plan.parallel_plan {
            plans.insert(parallel_plan.entry.clone(), parallel_plan);
        }
    }

    // Graphics pipelines.
    for def in &program.defs {
        if let DefMeta::EntryPoint(ref decl) = def.meta {
            if !decl.entry_type.is_compute() {
                let name = crate::symbol_name_or_bug(&program.symbols, def.name).to_string();
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
                }));
            }
        }
    }

    program.defs.retain(|d| !removed_entries.contains(&d.name));
    program.defs.extend(new_defs);

    Ok(ParallelizationResult {
        program,
        pipeline: PipelineDescriptor { pipelines },
        plans,
    })
}

// =============================================================================
// Lowering plan dispatcher
// =============================================================================

struct LoweringPlan {
    removed_entry: Option<SymbolId>,
    new_defs: Vec<Def>,
    pipeline: Pipeline,
    extra_bindings_used: u32,
    /// EGIR-bound parallelization plan, if this strategy migrated to the
    /// EGIR-side path. Populated for Map today; None for reduce / scan /
    /// redomap (still TLC-lowered until their EGIR migration).
    parallel_plan: Option<ParallelizationPlan>,
}

fn term_has_ordered_side_effect_soac(term: &Term) -> bool {
    if let TermKind::Soac(soac) = &term.kind {
        if soac_has_ordered_side_effect(soac) {
            return true;
        }
    }
    let mut found = false;
    term.for_each_child(&mut |child| {
        if !found {
            found = term_has_ordered_side_effect_soac(child);
        }
    });
    found
}

fn soac_has_ordered_side_effect(soac: &SoacOp) -> bool {
    match soac {
        SoacOp::Map { destination, .. }
        | SoacOp::Scan { destination, .. }
        | SoacOp::Filter { destination, .. } => *destination != SoacDestination::Fresh,
        SoacOp::Scatter { .. } | SoacOp::ReduceByIndex { .. } => true,
        SoacOp::Reduce { .. } | SoacOp::Redomap { .. } | SoacOp::Screma { .. } => false,
    }
}

fn retained_prefix_has_ordered_side_effect(analysis: &EntryAnalysis) -> bool {
    analysis.prefix_lets.iter().any(|(_, _, rhs)| term_has_ordered_side_effect_soac(rhs))
        || analysis.extra_slots.iter().any(|(_, value)| term_has_ordered_side_effect_soac(value))
}

fn make_serial_compute_plan(
    analysis: &EntryAnalysis,
    entry_name: &str,
    sizing: PipelineSizing,
) -> LoweringPlan {
    LoweringPlan {
        removed_entry: None,
        new_defs: Vec::new(),
        pipeline: Pipeline::Compute(ComputePipeline {
            entry_point: entry_name.to_string(),
            workgroup_size: sizing.workgroup,
            dispatch_size: DispatchSize::Fixed { x: 1, y: 1, z: 1 },
            bindings: collect_soac_bindings(&analysis.soac),
            default_total_threads: sizing.default_total_threads,
        }),
        extra_bindings_used: 0,
        parallel_plan: None,
    }
}

fn make_lowering_plan(
    analysis: &EntryAnalysis,
    entry_name: &str,
    next_binding: u32,
    forced_result_binding: Option<BindingRef>,
    program: &mut Program,
    term_ids: &mut TermIdSource,
) -> LoweringPlan {
    let sizing = PipelineSizing::for_analyzed_entry(program, analysis);
    let shape = parallel_soac_shape(&analysis.soac.original).expect("analyzed SOAC has a parallel shape");
    if retained_prefix_has_ordered_side_effect(analysis) {
        return make_serial_compute_plan(analysis, entry_name, sizing);
    }
    match shape.flavor {
        ParallelSoacFlavor::Map => {
            // `forced_result_binding` pins the map's output to a specific
            // buffer when this is a gather pre-pass (the consumer reads it via
            // `storage_index`); ordinary maps pass `None` and auto-allocate.
            make_map_plan(
                analysis,
                entry_name,
                next_binding,
                sizing,
                forced_result_binding,
                program,
            )
        }
        ParallelSoacFlavor::Reduce | ParallelSoacFlavor::Redomap => {
            let (reduce_op, ne) =
                accumulator_phase_combiner(&analysis.soac.original).expect("reduce-like SOAC has combiner");
            make_two_phase_plan(
                analysis,
                entry_name,
                reduce_op,
                ne,
                next_binding,
                forced_result_binding,
                program,
                sizing,
                term_ids,
            )
        }
        ParallelSoacFlavor::Scan => {
            let (op, ne) = scan_phase_combiner(&analysis.soac.original).expect("scan SOAC has combiner");
            make_scan_plan(
                analysis,
                entry_name,
                op,
                ne,
                next_binding,
                forced_result_binding,
                program,
                sizing,
            )
        }
        ParallelSoacFlavor::Screma => {
            let is_mixed = match &analysis.soac.original {
                SoacOp::Screma { accumulators, .. } => !accumulators.is_empty(),
                _ => false,
            };
            if is_mixed {
                // Mixed Screma: dispatched to make_screma_plan, which picks
                // an EGIR-parallel two-phase shape when the shape is
                // supported, else falls back to a serial 1×1×1 Compute
                // pipeline.
                make_screma_plan(analysis, entry_name, next_binding, sizing, program)
            } else {
                // Pointwise-only Screma is a multi-output Map: one lane
                // per element, no cross-lane phases.
                make_map_plan(
                    analysis,
                    entry_name,
                    next_binding,
                    sizing,
                    forced_result_binding,
                    program,
                )
            }
        }
    }
}

fn accumulator_phase_combiner(soac: &SoacOp) -> Option<(&super::SoacBody, &Term)> {
    match soac {
        SoacOp::Reduce { op, ne, .. } => Some((op, ne)),
        SoacOp::Redomap { reduce_op, ne, .. } => Some((reduce_op, ne)),
        _ => None,
    }
}

fn scan_phase_combiner(soac: &SoacOp) -> Option<(&super::SoacBody, &Term)> {
    match soac {
        SoacOp::Scan { op, ne, .. } => Some((op, ne)),
        _ => None,
    }
}

fn make_map_plan(
    analysis: &EntryAnalysis,
    entry_name: &str,
    _next_binding: u32,
    sizing: PipelineSizing,
    forced_output_binding: Option<BindingRef>,
    program: &Program,
) -> LoweringPlan {
    // Plan: don't pre-allocate the output binding. The EGIR side already
    // auto-allocates it via `build_entry_outputs` and walks the entry's
    // params to lay out input bindings; predicting either at TLC requires
    // mirroring buffer_specialize + from_tlc layout logic. The plan just
    // tags the entry as a parallel Map; the pipeline descriptor's binding
    // list gets enriched post-EGIR by `enrich_pipeline_with_auto_bindings`
    // (see `from_tlc.rs:164`).
    let bindings = collect_soac_bindings(&analysis.soac);
    let pipeline = Pipeline::Compute(ComputePipeline {
        entry_point: entry_name.to_string(),
        workgroup_size: sizing.workgroup,
        dispatch_size: DispatchSize::DerivedFrom {
            len: resolve_dispatch_len(analysis, 0, program),
            workgroup_size: sizing.workgroup.0,
        },
        bindings,
        default_total_threads: sizing.default_total_threads,
    });
    let parallel_plan = ParallelizationPlan {
        entry: entry_name.to_string(),
        dispatch: DispatchModel::DerivedFromInputLength {
            input_index: 0,
            workgroup_size: sizing.workgroup.0,
        },
        bindings: PlannedBindings {
            map_outputs: vec![forced_output_binding],
            accumulators: vec![],
        },
    };
    LoweringPlan {
        removed_entry: None,
        new_defs: Vec::new(),
        pipeline,
        extra_bindings_used: 0,
        parallel_plan: Some(parallel_plan),
    }
}

/// True when EGIR's `parallelize_entry` can parallelize this
/// mixed Screma today. Supported shapes:
/// - 0+ map outputs (no captures) + N>=1 Reduce accumulators (no
///   captures, arbitrary pure NE) — emits N+1 stages
/// - 0+ map outputs (no captures) + exactly 1 Scan accumulator (no
///   captures, arbitrary pure NE) — emits 3 stages
/// Mixed Reduce+Scan in the same Screma and multi-Scan still gate out.
/// Arbitrary NE subgraphs are handled by both phase 2 paths
/// (`synthesize_phase2_reduce_cloning_ne_named` and
/// `synthesize_phase2_scan`) via `graph_ops::clone_pure_subgraph`.
fn egir_parallelizable(soac: &SoacOp) -> bool {
    let SoacOp::Screma {
        map_lams,
        accumulators,
        ..
    } = soac
    else {
        return false;
    };
    if accumulators.is_empty() {
        return false;
    }
    if map_lams.iter().any(|m| !m.captures.is_empty()) {
        return false;
    }
    let all_reduce = accumulators.iter().all(|a| matches!(a.kind, super::ScremaAccumulator::Reduce));
    let single_scan =
        accumulators.len() == 1 && matches!(accumulators[0].kind, super::ScremaAccumulator::Scan);
    if !(all_reduce || single_scan) {
        return false;
    }
    for acc in accumulators {
        if !acc.step_lam.captures.is_empty() || !acc.reduce_op.captures.is_empty() {
            return false;
        }
    }
    true
}

/// Mixed-Screma planner. Two branches:
///
/// - **EGIR-parallel**: when `egir_parallelizable` matches, emit
///   a two-stage MultiCompute pipeline (phase 1 = the entry, chunked
///   in-place by `egir::parallelize::parallelize_entry`; phase 2
///   = synthesized tree-reduce combiner) and a `ParallelizationPlan`
///   whose `PlannedBindings` carries the partials/result bindings.
/// - **Serial fallback**: 1×1×1 Compute pipeline so the existing serial
///   soac_expand lowering produces correct output. `parallel_plan` is
///   None — the EGIR Screma arm short-circuits and the entry stays as
///   the original PendingSoac::Screma.
fn make_screma_plan(
    analysis: &EntryAnalysis,
    entry_name: &str,
    next_binding: u32,
    sizing: PipelineSizing,
    program: &Program,
) -> LoweringPlan {
    if !egir_parallelizable(&analysis.soac.original) {
        let bindings = collect_soac_bindings(&analysis.soac);
        let pipeline = Pipeline::Compute(ComputePipeline {
            entry_point: entry_name.to_string(),
            workgroup_size: sizing.workgroup,
            dispatch_size: DispatchSize::Fixed { x: 1, y: 1, z: 1 },
            bindings,
            default_total_threads: sizing.default_total_threads,
        });
        return LoweringPlan {
            removed_entry: None,
            new_defs: Vec::new(),
            pipeline,
            extra_bindings_used: 0,
            parallel_plan: None,
        };
    }

    // Binding layout. The entry has, in this order, M map outputs and N
    // accumulator outputs (one per accumulator, all auto-allocated by
    // from_tlc::build_entry_outputs above the AIC input bindings):
    //
    //   [0..AIC)        input bindings (auto-allocated)
    //   [AIC..AIC+M)    map output bindings (auto-allocated; phase 1
    //                   keeps writing here through chunked views)
    //   [AIC+M..AIC+M+N)  accumulator auto-output bindings (planner
    //                   aliases each as partials_i / scan_output;
    //                   phase 1 clears the slot on entry.outputs so
    //                   the host doesn't try to read it as the "real"
    //                   output)
    //   [AIC+M+N..)     planner-fresh bindings (result_i for Reduce;
    //                   block_sums + block_offsets for Scan)
    //
    // Single-reduce (M=0, N=1) collapses to make_two_phase_plan's
    // layout: partials = AIC, result = AIC+1.
    let auto_input_count = count_view_param_bindings(program, analysis.def_name);
    let (n_maps, accumulators_src) = match &analysis.soac.original {
        SoacOp::Screma {
            map_lams,
            accumulators,
            ..
        } => (map_lams.len() as u32, accumulators.clone()),
        _ => unreachable!(),
    };
    let n_accs = accumulators_src.len() as u32;
    let auto_outputs_base = next_binding + auto_input_count + n_maps;
    let fresh_base = auto_outputs_base + n_accs;

    let elem_type = analysis.soac.result_elem_type();
    let acc_kind = accumulators_src[0].kind;
    let all_reduce = accumulators_src.iter().all(|a| matches!(a.kind, super::ScremaAccumulator::Reduce));

    let (planned_bindings, pipeline, extra_used) = if all_reduce {
        let mut planned_accumulators = Vec::with_capacity(n_accs as usize);
        for i in 0..n_accs {
            let partials = BindingRef::new(AUTO_STORAGE_SET, auto_outputs_base + i);
            let result = BindingRef::new(AUTO_STORAGE_SET, fresh_base + i);
            planned_accumulators.push(PlannedScremaAccumulator {
                kind: PlannedScremaAccumulatorKind::Reduce,
                partials: Some(partials),
                result: Some(result),
                output: None,
                block_sums: None,
                block_offsets: None,
            });
        }
        let pipeline = build_screma_reduce_pipeline_descriptor(
            entry_name,
            &analysis.soac,
            &elem_type,
            &planned_accumulators,
            sizing,
        );
        (
            PlannedBindings {
                map_outputs: vec![None; n_maps as usize],
                accumulators: planned_accumulators,
            },
            pipeline,
            n_accs, // result bindings only; partials reuse auto-outputs
        )
    } else {
        // single-Scan path; gated by egir_parallelizable.
        debug_assert_eq!(n_accs, 1);
        debug_assert!(matches!(acc_kind, super::ScremaAccumulator::Scan));
        let scan_output_binding = BindingRef::new(AUTO_STORAGE_SET, auto_outputs_base);
        let block_sums_binding = BindingRef::new(AUTO_STORAGE_SET, fresh_base);
        let block_offsets_binding = BindingRef::new(AUTO_STORAGE_SET, fresh_base + 1);
        let pipeline = build_screma_scan_pipeline_descriptor(
            entry_name,
            analysis,
            scan_output_binding,
            block_sums_binding,
            block_offsets_binding,
            sizing,
            program,
        );
        let planned_accumulators = vec![PlannedScremaAccumulator {
            kind: PlannedScremaAccumulatorKind::Scan,
            partials: None,
            result: None,
            output: Some(scan_output_binding),
            block_sums: Some(block_sums_binding),
            block_offsets: Some(block_offsets_binding),
        }];
        (
            PlannedBindings {
                map_outputs: vec![None; n_maps as usize],
                accumulators: planned_accumulators,
            },
            pipeline,
            2, // block_sums + block_offsets
        )
    };

    let parallel_plan = ParallelizationPlan {
        entry: entry_name.to_string(),
        dispatch: DispatchModel::Fixed {
            groups: [1, 1, 1],
            local_size: [sizing.workgroup.0, sizing.workgroup.1, sizing.workgroup.2],
        },
        bindings: planned_bindings,
    };
    LoweringPlan {
        removed_entry: None,
        new_defs: Vec::new(),
        pipeline,
        extra_bindings_used: extra_used,
        parallel_plan: Some(parallel_plan),
    }
}

/// MultiCompute pipeline for a Screma with N Reduce accumulators.
/// One phase 1 stage (reads inputs, writes M map outputs + N partials),
/// then N phase 2 stages (one tree-reduce per accumulator).
fn build_screma_reduce_pipeline_descriptor(
    entry_name: &str,
    analysis: &SoacAnalysis,
    elem_type: &Type<TypeName>,
    accumulators: &[PlannedScremaAccumulator],
    sizing: PipelineSizing,
) -> Pipeline {
    let _ = elem_type;
    let workgroup = sizing.workgroup;
    let mut all_bindings = collect_soac_bindings(analysis);
    let input_indices: Vec<usize> = (0..all_bindings.len()).collect();
    let mut partials_indices = Vec::with_capacity(accumulators.len());
    let mut result_indices = Vec::with_capacity(accumulators.len());
    for (i, acc) in accumulators.iter().enumerate() {
        let partials = acc.partials.expect("Reduce reserves partials");
        let result = acc.result.expect("Reduce reserves result");
        let p_idx = push_storage_binding(
            &mut all_bindings,
            partials,
            Access::ReadWrite,
            BufferUsage::Intermediate,
            format!("{}_partials_{}", entry_name, i),
        );
        let r_idx = push_storage_binding(
            &mut all_bindings,
            result,
            Access::WriteOnly,
            BufferUsage::Output,
            format!("{}_result_{}", entry_name, i),
        );
        partials_indices.push(p_idx);
        result_indices.push(r_idx);
    }
    let mut stages = Vec::with_capacity(1 + accumulators.len());
    stages.push(saturating_stage(
        entry_name.to_string(),
        input_indices,
        partials_indices.clone(),
        workgroup,
    ));
    for (i, (p_idx, r_idx)) in partials_indices.iter().zip(result_indices.iter()).enumerate() {
        let phase2_name = if accumulators.len() == 1 {
            format!("{}_phase2_combine", entry_name)
        } else {
            format!("{}_phase2_combine_{}", entry_name, i)
        };
        stages.push(tree_phase2_stage(phase2_name, vec![*p_idx], vec![*r_idx]));
    }
    Pipeline::MultiCompute(MultiComputePipeline {
        bindings: all_bindings,
        stages,
        default_total_threads: sizing.default_total_threads,
    })
}

/// Build the `Pipeline::MultiCompute` descriptor for a Screma with a
/// single Scan accumulator. Three stages: phase 1 chunks the input +
/// writes block_sums; phase 2 sequentially scans block_sums into
/// block_offsets; phase 3 reads block_offsets and applies each chunk's
/// offset back over the scan output.
fn build_screma_scan_pipeline_descriptor(
    entry_name: &str,
    analysis: &EntryAnalysis,
    scan_output_binding: BindingRef,
    block_sums_binding: BindingRef,
    block_offsets_binding: BindingRef,
    sizing: PipelineSizing,
    program: &Program,
) -> Pipeline {
    let workgroup = sizing.workgroup;
    let mut all_bindings = collect_soac_bindings(&analysis.soac);
    let input_indices: Vec<usize> = (0..all_bindings.len()).collect();
    let output_idx = push_storage_binding(
        &mut all_bindings,
        scan_output_binding,
        Access::ReadWrite,
        BufferUsage::Output,
        format!("{}_scan_output", entry_name),
    );
    let block_sums_idx = push_storage_binding(
        &mut all_bindings,
        block_sums_binding,
        Access::ReadWrite,
        BufferUsage::Intermediate,
        format!("{}_block_sums", entry_name),
    );
    let block_offsets_idx = push_storage_binding(
        &mut all_bindings,
        block_offsets_binding,
        Access::ReadWrite,
        BufferUsage::Intermediate,
        format!("{}_block_offsets", entry_name),
    );

    let phase1_name = entry_name.to_string();
    let phase2_name = format!("{}_phase2_scan_sums", entry_name);
    let phase3_name = format!("{}_phase3_add_offsets", entry_name);

    let scan_len = resolve_dispatch_len(analysis, 0, program);
    Pipeline::MultiCompute(MultiComputePipeline {
        bindings: all_bindings,
        stages: vec![
            derived_stage(
                phase1_name,
                input_indices,
                vec![output_idx, block_sums_idx],
                workgroup,
                scan_len.clone(),
            ),
            fixed_stage(phase2_name, vec![block_sums_idx], vec![block_offsets_idx]),
            derived_stage(
                phase3_name,
                vec![block_offsets_idx],
                vec![output_idx],
                workgroup,
                scan_len,
            ),
        ],
        default_total_threads: sizing.default_total_threads,
    })
}

fn make_two_phase_plan(
    analysis: &EntryAnalysis,
    entry_name: &str,
    reduce_op: &super::SoacBody,
    ne: &Term,
    next_binding: u32,
    forced_result_binding: Option<BindingRef>,
    program: &mut Program,
    sizing: PipelineSizing,
    term_ids: &mut TermIdSource,
) -> LoweringPlan {
    // Partials always consumes one fresh binding. When the caller has
    // pre-allocated a result binding (graphical-entry lift), use it; the
    // lift step also added the Input-role decl on the graphical entry,
    // so this keeps the two sides in sync. Without a forced binding the
    // plan allocates its own.
    let (partials_binding, result_binding, extra_used) = match forced_result_binding {
        Some(result) => (BindingRef::new(AUTO_STORAGE_SET, next_binding), result, 1),
        None => (
            BindingRef::new(AUTO_STORAGE_SET, next_binding),
            BindingRef::new(AUTO_STORAGE_SET, next_binding + 1),
            2,
        ),
    };
    let elem_type = analysis.soac.result_elem_type();

    // Reduce / Redomap both qualify for the EGIR-side migration when
    // their combiner has no captures and the result is a scalar (for
    // tuple results, `emit_compute_output_stores` emits per-component
    // Stores that phase1's `find_store_of` doesn't yet locate).
    // Reduce additionally requires a scalar-literal NE (phase2 re-emits
    // via `intern_constant`); Redomap clones the NE subgraph at EGIR,
    // so any pure NE works.
    let scalar_result = matches!(
        &elem_type,
        Type::Constructed(TypeName::Int(_), _)
            | Type::Constructed(TypeName::UInt(_), _)
            | Type::Constructed(TypeName::Float(_), _)
            | Type::Constructed(TypeName::Bool, _),
    );
    // Tuple results whose output Stores the phase1 rewrite can fold into one
    // whole-tuple partials Store route through the EGIR chunking path alongside
    // scalars.
    let routable_result = scalar_result || tuple_can_use_whole_store_partials(&elem_type);
    let can_route = match &analysis.soac.original {
        SoacOp::Reduce { .. } => {
            routable_result && reduce_op.captures.is_empty() && is_simple_constant_term(ne)
        }
        SoacOp::Redomap { .. } => routable_result && reduce_op.captures.is_empty(),
        _ => false,
    };
    let egir_native = forced_result_binding.is_none() && can_route;

    if egir_native {
        // `from_tlc::convert_entry_point` allocates one auto-storage
        // binding per view-typed entry param; partials/result for the
        // EGIR path have to live above that range to avoid colliding
        // with the input buffers.
        let auto_input_count = count_view_param_bindings(program, analysis.def_name);
        let partials_binding = BindingRef::new(AUTO_STORAGE_SET, next_binding + auto_input_count);
        let result_binding = BindingRef::new(AUTO_STORAGE_SET, next_binding + auto_input_count + 1);
        let pipeline = build_two_phase_pipeline_descriptor(
            entry_name,
            &analysis.soac,
            &elem_type,
            partials_binding,
            result_binding,
            sizing,
        );
        let bindings = PlannedBindings {
            map_outputs: vec![],
            accumulators: vec![PlannedScremaAccumulator {
                kind: PlannedScremaAccumulatorKind::Reduce,
                partials: Some(partials_binding),
                result: Some(result_binding),
                output: None,
                block_sums: None,
                block_offsets: None,
            }],
        };
        let parallel_plan = ParallelizationPlan {
            entry: entry_name.to_string(),
            dispatch: DispatchModel::Fixed {
                groups: [1, 1, 1],
                local_size: [sizing.workgroup.0, sizing.workgroup.1, sizing.workgroup.2],
            },
            bindings,
        };
        return LoweringPlan {
            removed_entry: None,
            new_defs: Vec::new(),
            pipeline,
            extra_bindings_used: extra_used,
            parallel_plan: Some(parallel_plan),
        };
    }

    // TLC-side synthesis path (used by Redomap and complex Reduce
    // shapes not yet covered by the EGIR-side migration). Allocates
    // one extra Output binding per `extra_slots` entry so the original
    // entry's multi-output shape survives into phase 2.
    let extra_slot_bindings: Vec<BindingRef> = (0..analysis.extra_slots.len() as u32)
        .map(|i| BindingRef::new(AUTO_STORAGE_SET, next_binding + extra_used + i))
        .collect();
    let extra_used = extra_used + analysis.extra_slots.len() as u32;
    let (entries, pipeline) = build_two_phase_entries(
        entry_name,
        analysis,
        reduce_op,
        ne,
        &elem_type,
        partials_binding,
        result_binding,
        &extra_slot_bindings,
        program,
        sizing,
        term_ids,
    );
    LoweringPlan {
        removed_entry: Some(analysis.def_name),
        new_defs: entries,
        pipeline,
        extra_bindings_used: extra_used,
        parallel_plan: None,
    }
}

/// Count the storage bindings `from_tlc::convert_entry_point` will
/// auto-allocate for this entry's view-typed params. Plain unsized
/// arrays contribute 1; tuples-of-unsized-arrays contribute one per
/// component. Mirrors the allocator at `from_tlc.rs:395`+.
/// True for a tuple result the EGIR phase1 store-rewrite can collapse into a
/// single whole-tuple `partials[tid]` Store. `emit_compute_output_stores`
/// SoA-decomposes the result into per-component / per-element Stores; the
/// rewrite reconstructs one whole-result Store from them, which works as long
/// as every component is a scalar or fixed-size array. An unsized-array
/// component is split into its own runtime-sized binding that can't be folded
/// back into one slot, so such tuples stay on the TLC path.
fn tuple_can_use_whole_store_partials(elem_ty: &Type<TypeName>) -> bool {
    match elem_ty {
        Type::Constructed(TypeName::Tuple(_), comps) => {
            comps.iter().all(|c| !crate::types::is_unsized_array(c))
        }
        _ => false,
    }
}

fn count_view_param_bindings(program: &Program, def_sym: SymbolId) -> u32 {
    let def = match program.defs.iter().find(|d| d.name == def_sym) {
        Some(d) => d,
        None => return 0,
    };
    match &def.meta {
        DefMeta::EntryPoint(entry) => entry.param_bindings.iter().flatten().map(|b| b.buffer_count()).sum(),
        _ => 0,
    }
}

/// True for `Term`s whose value at EGIR time will be a `ConstantValue`
/// EGIR can reconstruct (`IntLit`, `FloatLit`, `BoolLit`). Used by the
/// EGIR-side reduce path to gate on initializers it can re-emit.
fn is_simple_constant_term(t: &Term) -> bool {
    matches!(
        t.kind,
        TermKind::IntLit(_) | TermKind::FloatLit(_) | TermKind::BoolLit(_)
    )
}

/// Build the `Pipeline::MultiCompute` descriptor for an EGIR-side
/// two-phase reduce. Phase 1 stage keeps the entry's original name
/// (the EGIR pass rewrites the body in place); phase 2 is named
/// `<entry>_phase2_combine`.
fn build_two_phase_pipeline_descriptor(
    entry_name: &str,
    analysis: &SoacAnalysis,
    _elem_type: &Type<TypeName>,
    partials_binding: BindingRef,
    result_binding: BindingRef,
    sizing: PipelineSizing,
) -> Pipeline {
    let workgroup = sizing.workgroup;
    let phase2_name = format!("{}_phase2_combine", entry_name);
    let mut all_bindings = collect_soac_bindings(analysis);
    let input_indices: Vec<usize> = (0..all_bindings.len()).collect();
    let partials_idx = push_storage_binding(
        &mut all_bindings,
        partials_binding,
        Access::ReadWrite,
        BufferUsage::Intermediate,
        format!("{}_partials", entry_name),
    );
    let result_idx = push_storage_binding(
        &mut all_bindings,
        result_binding,
        Access::WriteOnly,
        BufferUsage::Output,
        format!("{}_result", entry_name),
    );
    Pipeline::MultiCompute(MultiComputePipeline {
        bindings: all_bindings,
        stages: vec![
            saturating_stage(
                entry_name.to_string(),
                input_indices,
                vec![partials_idx],
                workgroup,
            ),
            tree_phase2_stage(phase2_name, vec![partials_idx], vec![result_idx]),
        ],
        default_total_threads: sizing.default_total_threads,
    })
}

fn make_scan_plan(
    analysis: &EntryAnalysis,
    entry_name: &str,
    op: &super::SoacBody,
    _ne: &Term,
    next_binding: u32,
    forced_result_binding: Option<BindingRef>,
    program: &mut Program,
    sizing: PipelineSizing,
) -> LoweringPlan {
    let elem_type = analysis.soac.result_elem_type();

    // Path B (EGIR-side) gate: scan with no combiner captures, where
    // the input is either an explicit `ArrayExpr::StorageBuffer` or a
    // bare `Var(entry_param)` that `from_tlc` will convert into a
    // `PureOp::StorageView`. Captures fail today because phase 2/3
    // don't plumb them through; non-Var/non-Storage inputs (Range,
    // literal, etc.) fail because phase 1 needs a `(set, binding)`
    // pair to chunk.
    let input_likely_storage = match analysis.soac.inputs().first() {
        Some(ArrayExpr::StorageView(_)) => true,
        Some(ArrayExpr::Ref(t)) => matches!(&t.kind, TermKind::Var(VarRef::Symbol(_))),
        _ => false,
    };
    let can_route = op.captures.is_empty() && input_likely_storage;

    if !can_route {
        let pipeline = Pipeline::Compute(ComputePipeline {
            entry_point: entry_name.to_string(),
            workgroup_size: sizing.workgroup,
            // Serial single-thread scan, not parallelized — no EGIR pass fills
            dispatch_size: DispatchSize::Fixed { x: 1, y: 1, z: 1 },
            bindings: vec![],
            default_total_threads: sizing.default_total_threads,
        });
        return LoweringPlan {
            removed_entry: None,
            new_defs: Vec::new(),
            pipeline,
            extra_bindings_used: 0,
            parallel_plan: None,
        };
    }

    // When the original scan has `destination = true`, phase 1 and
    // phase 3 write back to the input buffer; the auto-bound output
    // binding is unused. The pipeline descriptor reroutes
    // accordingly.
    let consuming = matches!(
        &analysis.soac.original,
        SoacOp::Scan {
            destination: SoacDestination::InputBuffer,
            ..
        },
    );

    // `from_tlc::convert_entry_point` auto-allocates one binding per
    // view-typed entry param plus one for the array output. Our
    // intermediates have to live above that range.
    let auto_input_count = count_view_param_bindings(program, analysis.def_name);
    // A gather pre-pass pins its result to the buffer the consumer reads;
    // otherwise the output sits just past the input-view bindings (matching
    // `from_tlc::build_entry_outputs`). The two intermediates start one past
    // the output binding so they never collide with it — a forced output may
    // sit above the fresh range. (Non-forced layout is unchanged: output at
    // `auto_input_count`, then `+1`, `+2`.)
    let auto_output = next_binding + auto_input_count;
    let output_binding = forced_result_binding.unwrap_or(BindingRef::new(AUTO_STORAGE_SET, auto_output));
    let scratch_start = output_binding.binding.max(auto_output) + 1;
    let block_sums_binding = BindingRef::new(AUTO_STORAGE_SET, scratch_start);
    let block_offsets_binding = BindingRef::new(AUTO_STORAGE_SET, scratch_start + 1);

    let pipeline = build_scan_pipeline_descriptor(
        entry_name,
        analysis,
        &elem_type,
        consuming,
        output_binding,
        block_sums_binding,
        block_offsets_binding,
        sizing,
        program,
    );

    let bindings = PlannedBindings {
        map_outputs: vec![],
        accumulators: vec![PlannedScremaAccumulator {
            kind: PlannedScremaAccumulatorKind::Scan,
            partials: None,
            result: None,
            output: forced_result_binding,
            block_sums: Some(block_sums_binding),
            block_offsets: Some(block_offsets_binding),
        }],
    };
    let parallel_plan = ParallelizationPlan {
        entry: entry_name.to_string(),
        dispatch: DispatchModel::Fixed {
            groups: [1, 1, 1],
            local_size: [sizing.workgroup.0, sizing.workgroup.1, sizing.workgroup.2],
        },
        bindings,
    };
    LoweringPlan {
        removed_entry: None,
        new_defs: Vec::new(),
        pipeline,
        extra_bindings_used: 2,
        parallel_plan: Some(parallel_plan),
    }
}

/// Build the `Pipeline::MultiCompute` descriptor for a parallel scan.
/// Three stages share three or four bindings depending on whether the
/// scan consumes its input:
///
/// - Non-consuming (`consuming = false`): four bindings — input, the
///   entry's auto-bound output, and the two synthesized intermediates
///   (block_sums, block_offsets). Phase 1 reads input, writes output +
///   block_sums; phase 2 (1×1×1) reads block_sums, writes
///   block_offsets; phase 3 reads block_offsets, writes output in
///   place.
/// - Consuming (`consuming = true`): three bindings — input (promoted
///   to ReadWrite), block_sums, and block_offsets. The auto-output
///   slot is unused. All phase writes that would have hit the output
///   binding land on the input binding instead; the host reads the
///   result from the input buffer.
fn build_scan_pipeline_descriptor(
    entry_name: &str,
    analysis: &EntryAnalysis,
    elem_type: &Type<TypeName>,
    consuming: bool,
    output_binding: BindingRef,
    block_sums_binding: BindingRef,
    block_offsets_binding: BindingRef,
    sizing: PipelineSizing,
    program: &Program,
) -> Pipeline {
    let soac = &analysis.soac;
    let workgroup = sizing.workgroup;
    let _ = elem_type;
    let mut all_bindings = collect_soac_bindings(soac);
    // When `collect_soac_bindings` returned no Storage bindings (the
    // SoAC's input was a bare `Var(entry_param)` whose provenance fell
    // through to Opaque), but the consuming path needs the input
    // binding's index, synthesize it now. from_tlc allocates view
    // entry params at (AUTO_STORAGE_SET, 0..auto_input_count); the
    // scan's input is the first view input, so binding 0. Enrichment
    // later deduplicates against the entry's real `storage_binding`.
    if consuming && all_bindings.is_empty() {
        push_storage_binding(
            &mut all_bindings,
            BindingRef::new(AUTO_STORAGE_SET, 0),
            Access::ReadWrite,
            BufferUsage::Input,
            "input_0".to_string(),
        );
    }
    let input_indices: Vec<usize> = (0..all_bindings.len()).collect();
    let output_idx = if consuming {
        // Phase 1 + phase 3 write back to the input binding. Promote
        // the input's access to ReadWrite and reuse its index for
        // every spot the separate output binding would have occupied.
        let idx = input_indices[0];
        if let Binding::StorageBuffer { access, .. } = &mut all_bindings[idx] {
            *access = Access::ReadWrite;
        }
        idx
    } else {
        // Phase 1 writes to the entry's auto-bound output binding.
        push_storage_binding(
            &mut all_bindings,
            output_binding,
            Access::ReadWrite,
            BufferUsage::Output,
            format!("{}_output", entry_name),
        )
    };
    let block_sums_idx = push_storage_binding(
        &mut all_bindings,
        block_sums_binding,
        Access::ReadWrite,
        BufferUsage::Intermediate,
        format!("{}_block_sums", entry_name),
    );
    let block_offsets_idx = push_storage_binding(
        &mut all_bindings,
        block_offsets_binding,
        Access::ReadWrite,
        BufferUsage::Intermediate,
        format!("{}_block_offsets", entry_name),
    );

    let phase1_name = entry_name.to_string();
    let phase2_name = format!("{}_phase2_scan_sums", entry_name);
    let phase3_name = format!("{}_phase3_add_offsets", entry_name);

    // Phases 1 and 3 iterate one thread per scan element — the scan's
    // single input determines the length.
    let scan_len = resolve_dispatch_len(analysis, 0, program);
    Pipeline::MultiCompute(MultiComputePipeline {
        bindings: all_bindings,
        stages: vec![
            derived_stage(
                phase1_name,
                input_indices,
                vec![output_idx, block_sums_idx],
                workgroup,
                scan_len.clone(),
            ),
            fixed_stage(phase2_name, vec![block_sums_idx], vec![block_offsets_idx]),
            derived_stage(
                phase3_name,
                vec![block_offsets_idx],
                vec![output_idx],
                workgroup,
                scan_len,
            ),
        ],
        default_total_threads: sizing.default_total_threads,
    })
}

// =============================================================================
// Two-phase entry builder (Reduce / Redomap)
// =============================================================================

fn build_two_phase_entries(
    entry_name: &str,
    analysis: &EntryAnalysis,
    reduce_op: &super::SoacBody,
    ne: &Term,
    elem_type: &Type<TypeName>,
    partials_binding: BindingRef,
    result_binding: BindingRef,
    extra_slot_bindings: &[BindingRef],
    program: &mut Program,
    sizing: PipelineSizing,
    term_ids: &mut TermIdSource,
) -> (Vec<Def>, Pipeline) {
    debug_assert_eq!(
        extra_slot_bindings.len(),
        analysis.extra_slots.len(),
        "one binding per extra slot"
    );
    let workgroup = sizing.workgroup;
    let span = ne.span;

    let unit_ty = Type::Constructed(TypeName::Unit, vec![]);

    // Phase 1: chunked SOAC, each thread writes its partial to partials[tid].
    let phase1_name = format!("{}_phase1_chunks", entry_name);
    let phase1_body = build_chunked_soac_body(
        &analysis.soac,
        &analysis.prefix_lets,
        elem_type.clone(),
        span,
        program,
        Some(partials_binding),
        workgroup.0,
        term_ids,
    );
    // Phase 1 storage interface: reads whatever the input SOAC declares
    // (Input role), writes `partials` at `tid` (Intermediate). Input
    // bindings must be declared explicitly — the backend's storage-buffer
    // validation lists the entry's `storage_bindings` as the allowlist
    // for any `storage(set, binding)` reference in the body.
    let mut phase1_bindings = input_storage_decls(&analysis.soac);
    phase1_bindings.push(interface::StorageBindingDecl {
        binding: BindingRef::new(partials_binding.set, partials_binding.binding),
        role: interface::StorageRole::Intermediate,
        elem_ty: elem_type.clone(),
        length: None,
    });
    let phase1_def = make_entry_def(
        &phase1_name,
        phase1_body,
        unit_ty.clone(),
        &analysis.required_params,
        phase1_bindings,
        program,
        term_ids,
    );

    // Phase 2: reduce over the partials buffer, write result to result_binding[0].
    let phase2_name = format!("{}_phase2_combine", entry_name);
    let partials_input = ArrayExpr::StorageView(crate::tlc::StorageView {
        binding: partials_binding,
        offset: Box::new(uint_lit(0, span, term_ids)),
        len: Box::new(uint_lit(workgroup.0 as u64, span, term_ids)),
        elem_ty: elem_type.clone(),
    });
    let phase2_soac = SoacOp::Reduce {
        op: reduce_op.clone(),
        ne: Box::new(ne.clone()),
        input: partials_input,
    };
    let phase2_soac_term = soac_term(phase2_soac, elem_type.clone(), span, term_ids);
    let r_sym = program.symbols.alloc("_par_out".into());
    let r_var = var_term(r_sym, elem_type.clone(), span, term_ids);
    let phase2_store = intrinsic_term_by_id(
        catalog().known().storage_store,
        vec![
            uint_lit(result_binding.set as u64, span, term_ids),
            uint_lit(result_binding.binding as u64, span, term_ids),
            uint_lit(0, span, term_ids),
            r_var,
        ],
        unit_ty.clone(),
        span,
        term_ids,
    );
    // Sequence the reduce's result store with one `storage_store` per
    // extra slot. Each extra slot is a non-SOAC term (`analyze_entry`
    // rejects SOAC-valued extra slots) so evaluating it in phase 2's
    // single-thread context is safe.
    let mut phase2_tail = phase2_store;
    for ((_, slot_value), slot_binding) in analysis.extra_slots.iter().zip(extra_slot_bindings.iter()).rev()
    {
        let store = intrinsic_term_by_id(
            catalog().known().storage_store,
            vec![
                uint_lit(slot_binding.set as u64, span, term_ids),
                uint_lit(slot_binding.binding as u64, span, term_ids),
                uint_lit(0, span, term_ids),
                slot_value.clone(),
            ],
            unit_ty.clone(),
            span,
            term_ids,
        );
        let seq_sym = program.symbols.alloc("_seq".into());
        phase2_tail = let_term(seq_sym, unit_ty.clone(), store, phase2_tail, span, term_ids);
    }
    let phase2_body = let_term(
        r_sym,
        elem_type.clone(),
        phase2_soac_term,
        phase2_tail,
        span,
        term_ids,
    );
    // Phase 2 storage interface: reads `partials` (as an Intermediate),
    // writes the final user-visible `result`, plus one Output binding
    // per extra slot so the original entry's multi-output shape
    // survives into phase 2.
    let mut phase2_bindings = vec![
        interface::StorageBindingDecl {
            binding: BindingRef::new(partials_binding.set, partials_binding.binding),
            role: interface::StorageRole::Intermediate,
            elem_ty: elem_type.clone(),
            length: None,
        },
        interface::StorageBindingDecl {
            binding: BindingRef::new(result_binding.set, result_binding.binding),
            role: interface::StorageRole::Output,
            elem_ty: elem_type.clone(),
            length: None,
        },
    ];
    for ((_, slot_value), slot_binding) in analysis.extra_slots.iter().zip(extra_slot_bindings.iter()) {
        phase2_bindings.push(interface::StorageBindingDecl {
            binding: *slot_binding,
            role: interface::StorageRole::Output,
            elem_ty: slot_value.ty.clone(),
            length: None,
        });
    }
    let phase2_def = make_entry_def(
        &phase2_name,
        phase2_body,
        unit_ty.clone(),
        &analysis.required_params,
        phase2_bindings,
        program,
        term_ids,
    );

    let mut all_bindings = collect_soac_bindings(&analysis.soac);
    let input_indices: Vec<usize> = (0..all_bindings.len()).collect();
    let partials_idx = push_storage_binding(
        &mut all_bindings,
        partials_binding,
        Access::ReadWrite,
        BufferUsage::Intermediate,
        format!("{}_partials", entry_name),
    );
    let result_idx = push_storage_binding(
        &mut all_bindings,
        result_binding,
        Access::WriteOnly,
        BufferUsage::Output,
        format!("{}_result", entry_name),
    );
    let mut phase2_output_indices = vec![result_idx];
    for (i, slot_binding) in extra_slot_bindings.iter().enumerate() {
        let idx = push_storage_binding(
            &mut all_bindings,
            *slot_binding,
            Access::WriteOnly,
            BufferUsage::Output,
            format!("{}_slot{}", entry_name, analysis.extra_slots[i].0),
        );
        phase2_output_indices.push(idx);
    }

    let pipeline = Pipeline::MultiCompute(MultiComputePipeline {
        bindings: all_bindings,
        stages: vec![
            saturating_stage(phase1_name.clone(), input_indices, vec![partials_idx], workgroup),
            tree_phase2_stage(phase2_name.clone(), vec![partials_idx], phase2_output_indices),
        ],
        default_total_threads: sizing.default_total_threads,
    });

    (vec![phase1_def, phase2_def], pipeline)
}

// =============================================================================
// Shared per-thread chunk-arithmetic scaffolding
// =============================================================================

/// The fresh symbol set for a parallel entry's per-thread chunk arithmetic.
/// One phase-entry body has a single `ChunkArithmetic` that `wrap()` turns
/// into a let-chain outside the body:
///
/// ```text
/// let tid         = _w_intrinsic_thread_id() in
/// let total       = 64 in
/// let input_len   = <input_len_term> in
/// let chunk_size  = (input_len + total - 1) / total in
/// let chunk_start = tid * chunk_size in
/// let chunk_len   = if chunk_size < (input_len - chunk_start)
///                     then chunk_size else (input_len - chunk_start) in
/// <body>
/// ```
///
/// Reused by both `build_chunked_soac_body` (reduce/map/redomap) and
/// `build_scan_phase_def` (scan phases 1 and 3).
struct ChunkArithmetic {
    /// Type of `total`, `input_len`, `chunk_size`, `chunk_start`,
    /// `chunk_len`. `tid` is always u32 (from `_w_intrinsic_thread_id`)
    /// and is cast to `index_ty` at the `chunk_start = tid * chunk_size`
    /// site when the two types differ.
    index_ty: Type<TypeName>,
    /// Total thread count for this phase. Equals the workgroup X
    /// dimension under the single-workgroup design used by reduce /
    /// redomap / scan today; chosen from `pick_workgroup_size`.
    total_threads: u32,
    tid_sym: SymbolId,
    total_sym: SymbolId,
    input_len_sym: SymbolId,
    chunk_size_sym: SymbolId,
    chunk_start_sym: SymbolId,
    chunk_len_sym: SymbolId,
}

impl ChunkArithmetic {
    fn alloc_for(index_ty: Type<TypeName>, total_threads: u32, program: &mut Program) -> Self {
        ChunkArithmetic {
            index_ty,
            total_threads,
            tid_sym: program.symbols.alloc("_par_tid".into()),
            total_sym: program.symbols.alloc("_par_total".into()),
            input_len_sym: program.symbols.alloc("_par_input_len".into()),
            chunk_size_sym: program.symbols.alloc("_par_chunk_size".into()),
            chunk_start_sym: program.symbols.alloc("_par_chunk_start".into()),
            chunk_len_sym: program.symbols.alloc("_par_chunk_len".into()),
        }
    }

    /// The raw u32 thread-id. Use for storage-op indices (`partials[tid]`,
    /// `block_sums[tid]`, etc.) since `_w_intrinsic_storage_store`'s
    /// index arg is u32.
    fn tid_u32(&self, span: ast::Span, term_ids: &mut TermIdSource) -> Term {
        var_term(self.tid_sym, u32_ty(), span, term_ids)
    }
    fn chunk_start(&self, span: ast::Span, term_ids: &mut TermIdSource) -> Term {
        var_term(self.chunk_start_sym, self.index_ty.clone(), span, term_ids)
    }
    fn chunk_len(&self, span: ast::Span, term_ids: &mut TermIdSource) -> Term {
        var_term(self.chunk_len_sym, self.index_ty.clone(), span, term_ids)
    }

    /// Wrap `body` with:
    /// ```text
    /// let tid         = _w_intrinsic_thread_id()  (u32)
    /// let total       = TOTAL_THREADS             (index_ty)
    /// let input_len   = <input_len_term>          (index_ty)
    /// let chunk_size  = (input_len + total - 1) / total  (index_ty)
    /// let chunk_start = cast(tid) * chunk_size    (index_ty)
    /// let chunk_len   = min(chunk_size, input_len - chunk_start)  (index_ty)
    /// ```
    /// `cast(tid)` is the identity when `index_ty` is u32, otherwise an
    /// `i32.u32` bitcast. `input_len_term` must already be typed as
    /// `self.index_ty`; callers derive it from the SOAC input's length
    /// term.
    fn wrap(
        &self,
        body: Term,
        input_len_term: Term,
        span: ast::Span,
        program: &mut Program,
        term_ids: &mut TermIdSource,
    ) -> Term {
        let ity = self.index_ty.clone();
        let bool_ty = Type::Constructed(TypeName::Bool, vec![]);
        let one_lit = int_lit_of(1, &ity, span, term_ids);
        let total_lit = int_lit_of(self.total_threads as i64, &ity, span, term_ids);

        let len_minus_start = binop(
            "-",
            var_term(self.input_len_sym, ity.clone(), span, term_ids),
            var_term(self.chunk_start_sym, ity.clone(), span, term_ids),
            ity.clone(),
            span,
            term_ids,
        );
        let cond = binop(
            "<",
            var_term(self.chunk_size_sym, ity.clone(), span, term_ids),
            len_minus_start.clone(),
            bool_ty,
            span,
            term_ids,
        );
        let then_branch = var_term(self.chunk_size_sym, ity.clone(), span, term_ids);
        let min_expr = Term {
            id: term_ids.next_id(),
            ty: ity.clone(),
            span,
            kind: TermKind::If {
                cond: Box::new(cond),
                then_branch: Box::new(then_branch),
                else_branch: Box::new(len_minus_start),
            },
        };
        let body = let_term(self.chunk_len_sym, ity.clone(), min_expr, body, span, term_ids);

        let tid_as_idx = if ity == u32_ty() {
            var_term(self.tid_sym, u32_ty(), span, term_ids)
        } else {
            let tid_var = var_term(self.tid_sym, u32_ty(), span, term_ids);
            intrinsic_term("i32.u32", vec![tid_var], ity.clone(), span, program, term_ids)
        };
        let chunk_start_rhs = binop(
            "*",
            tid_as_idx,
            var_term(self.chunk_size_sym, ity.clone(), span, term_ids),
            ity.clone(),
            span,
            term_ids,
        );
        let body = let_term(
            self.chunk_start_sym,
            ity.clone(),
            chunk_start_rhs,
            body,
            span,
            term_ids,
        );

        let total_minus_1 = binop(
            "-",
            var_term(self.total_sym, ity.clone(), span, term_ids),
            one_lit,
            ity.clone(),
            span,
            term_ids,
        );
        let len_plus = binop(
            "+",
            var_term(self.input_len_sym, ity.clone(), span, term_ids),
            total_minus_1,
            ity.clone(),
            span,
            term_ids,
        );
        let chunk_size_rhs = binop(
            "/",
            len_plus,
            var_term(self.total_sym, ity.clone(), span, term_ids),
            ity.clone(),
            span,
            term_ids,
        );
        let body = let_term(
            self.chunk_size_sym,
            ity.clone(),
            chunk_size_rhs,
            body,
            span,
            term_ids,
        );

        let body = let_term(
            self.input_len_sym,
            ity.clone(),
            input_len_term,
            body,
            span,
            term_ids,
        );
        let body = let_term(self.total_sym, ity, total_lit, body, span, term_ids);

        let tid_rhs = intrinsic_term_by_id(catalog().known().thread_id, vec![], u32_ty(), span, term_ids);
        let_term(self.tid_sym, u32_ty(), tid_rhs, body, span, term_ids)
    }
}

fn u32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::UInt(32), vec![])
}

/// Index type for `ChunkArithmetic` to use when chunking this SOAC.
/// Range inputs carry their own index type (usually i32 from `0..<N`
/// syntax); every other input kind (storage views, zips, generators,
/// literals) is indexed by u32.
fn soac_input_index_ty(soac: &SoacOp) -> Type<TypeName> {
    let first = match soac {
        SoacOp::Map { inputs, .. } | SoacOp::Redomap { inputs, .. } => inputs.first(),
        SoacOp::Reduce { input, .. } | SoacOp::Scan { input, .. } => Some(input),
        _ => None,
    };
    match first {
        Some(ArrayExpr::Range { start, .. }) => start.ty.clone(),
        _ => u32_ty(),
    }
}

/// Integer literal term typed as `ty`. `ty` is expected to be one of
/// `Int(32)` / `UInt(32)`; other widths/types aren't used by
/// `ChunkArithmetic` today.
fn int_lit_of(value: i64, ty: &Type<TypeName>, span: ast::Span, term_ids: &mut TermIdSource) -> Term {
    Term {
        id: term_ids.next_id(),
        ty: ty.clone(),
        span,
        kind: TermKind::IntLit(value.to_string()),
    }
}

// =============================================================================
// Chunked SOAC body builder
// =============================================================================

/// Build a chunked SOAC body for reduce/redomap/map phase 1. The SOAC's
/// input is rebased to `(chunk_start, chunk_len)` (per-thread range), and
/// the result is optionally stored into `partials[tid]`. Scan never
/// reaches this builder — its parallelization is fully EGIR-side via
/// the Screma scan path.
fn build_chunked_soac_body(
    soac: &SoacAnalysis,
    prefix_lets: &[(SymbolId, Type<TypeName>, Term)],
    result_ty: Type<TypeName>,
    span: ast::Span,
    program: &mut Program,
    // If Some((set, binding)), wrap the inner SOAC with
    //   let r = <soac> in _w_intrinsic_storage_store(set, binding, tid, r)
    // so each thread's partial lands at partials[tid]. Return type then
    // becomes Unit rather than `result_ty`.
    write_partial_to: Option<BindingRef>,
    total_threads: u32,
    term_ids: &mut TermIdSource,
) -> Term {
    // ChunkArithmetic's `index_ty` matches the SOAC input's index type:
    // storage-view inputs use u32 (from `_w_intrinsic_storage_len`);
    // range inputs use the range's declared `start.ty` (typically i32).
    // This keeps `chunk_array_expr`'s rebuilt Range consistent with the
    // original source-level types, avoiding a bitcast boundary.
    let index_ty = soac_input_index_ty(&soac.original);
    let chunk = ChunkArithmetic::alloc_for(index_ty, total_threads, program);
    let chunk_start_var = chunk.chunk_start(span, term_ids);
    let chunk_len_var = chunk.chunk_len(span, term_ids);

    let chunked_soac = match &soac.original {
        SoacOp::Map {
            lam,
            inputs,
            destination,
        } => {
            let chunked_inputs = inputs
                .iter()
                .map(|input| chunk_array_expr(input, &chunk_start_var, &chunk_len_var, term_ids))
                .collect();
            SoacOp::Map {
                lam: lam.clone(),
                inputs: chunked_inputs,
                destination: *destination,
            }
        }
        SoacOp::Reduce { op, ne, input } => SoacOp::Reduce {
            op: op.clone(),
            ne: ne.clone(),
            input: chunk_array_expr(input, &chunk_start_var, &chunk_len_var, term_ids),
        },
        SoacOp::Redomap {
            op,
            reduce_op,
            ne,
            inputs,
        } => {
            let chunked_inputs = inputs
                .iter()
                .map(|input| chunk_array_expr(input, &chunk_start_var, &chunk_len_var, term_ids))
                .collect();
            SoacOp::Redomap {
                op: op.clone(),
                reduce_op: reduce_op.clone(),
                ne: ne.clone(),
                inputs: chunked_inputs,
            }
        }
        SoacOp::Scan { .. } => {
            unreachable!("Scan is parallelized EGIR-side, never reaches build_chunked_soac_body")
        }
        _ => unreachable!("analyze_soac rejected non-parallelizable variants"),
    };

    // Get the input length term from the first input's provenance.
    let input_len_term = get_input_len(soac, span, term_ids);

    let mut body = soac_term(chunked_soac, result_ty.clone(), span, term_ids);

    // If requested, wrap the SOAC with `let r = <soac> in store(set, binding, tid, r)`
    // so each thread's partial result lands in its own slot.
    if let Some(br) = write_partial_to {
        let unit_ty = Type::Constructed(TypeName::Unit, vec![]);
        let r_sym = program.symbols.alloc("_par_out".into());
        let tid_var = chunk.tid_u32(span, term_ids);
        let r_var = var_term(r_sym, result_ty.clone(), span, term_ids);
        let store = intrinsic_term_by_id(
            catalog().known().storage_store,
            vec![
                uint_lit(br.set as u64, span, term_ids),
                uint_lit(br.binding as u64, span, term_ids),
                tid_var,
                r_var,
            ],
            unit_ty,
            span,
            term_ids,
        );
        body = let_term(r_sym, result_ty.clone(), body, store, span, term_ids);
    }

    // Wrap with prefix lets (from the original entry body).
    for (name, ty, rhs) in prefix_lets.iter().rev() {
        body = let_term(*name, ty.clone(), rhs.clone(), body, span, term_ids);
    }

    chunk.wrap(body, input_len_term, span, program, term_ids)
}

/// Replace a storage buffer's offset/len with chunk-relative values.
fn chunk_array_expr(
    input: &ArrayExpr,
    chunk_start: &Term,
    chunk_len: &Term,
    term_ids: &mut TermIdSource,
) -> ArrayExpr {
    match input {
        ArrayExpr::StorageView(crate::tlc::StorageView {
            binding,
            offset,
            elem_ty,
            ..
        }) => {
            // New offset = original_offset + chunk_start
            let new_offset = if is_literal_zero(offset) {
                chunk_start.clone()
            } else {
                binop(
                    "+",
                    (**offset).clone(),
                    chunk_start.clone(),
                    chunk_start.ty.clone(),
                    chunk_start.span,
                    term_ids,
                )
            };
            ArrayExpr::StorageView(crate::tlc::StorageView {
                binding: *binding,
                offset: Box::new(new_offset),
                len: Box::new(chunk_len.clone()),
                elem_ty: elem_ty.clone(),
            })
        }
        ArrayExpr::Range { start, len: _, step } => {
            // Range: chunk_start..chunk_len starting from original start.
            // Chunk values are u32 (from ChunkArithmetic); the original
            // range is typically i32. Coercion to match the range's type
            // happens at the caller — `chunk_array_expr` stays a simple
            // structural rewrite.
            let range_ty = start.ty.clone();
            let new_start = binop(
                "+",
                (**start).clone(),
                chunk_start.clone(),
                range_ty.clone(),
                start.span,
                term_ids,
            );
            ArrayExpr::Range {
                start: Box::new(new_start),
                len: Box::new(chunk_len.clone()),
                step: step.clone(),
            }
        }
        other => other.clone(), // shouldn't happen for parallelizable inputs
    }
}

/// Get the input length term from the SOAC's first input.
fn get_input_len(soac: &SoacAnalysis, span: ast::Span, term_ids: &mut TermIdSource) -> Term {
    get_array_expr_len(soac.inputs()[0], span, term_ids)
}

fn get_array_expr_len(ae: &ArrayExpr, span: ast::Span, term_ids: &mut TermIdSource) -> Term {
    match ae {
        ArrayExpr::StorageView(crate::tlc::StorageView { len, .. }) => (**len).clone(),
        ArrayExpr::Range { len, .. } => (**len).clone(),
        _ => uint_lit(0, span, term_ids), // fallback
    }
}

fn is_literal_zero(term: &Term) -> bool {
    matches!(&term.kind, TermKind::IntLit(s) if s == "0")
}

// =============================================================================
// Term-building helpers
// =============================================================================

fn var_term(sym: SymbolId, ty: Type<TypeName>, span: ast::Span, term_ids: &mut TermIdSource) -> Term {
    Term {
        id: term_ids.next_id(),
        ty,
        span,
        kind: TermKind::Var(VarRef::Symbol(sym)),
    }
}

fn let_term(
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
/// a catalog intrinsic.
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

fn intrinsic_term(
    name: &str,
    args: Vec<Term>,
    ret_ty: Type<TypeName>,
    span: ast::Span,
    program: &mut Program,
    term_ids: &mut TermIdSource,
) -> Term {
    // Catalog-resolved builtins emit `Var(Builtin(id))` so downstream
    // passes dispatch structurally. The assert mirrors `tlc::build_call`
    // and `buffer_specialize::make_app`: synthesized IR sites target
    // single-overload entries.
    let func_var = if let Some(def) = catalog().lookup_by_any_name(name) {
        assert_eq!(
            def.overloads().len(),
            1,
            "parallelize::intrinsic_term({:?}) targets a multi-overload catalog entry; \
             caller must specify overload_idx explicitly",
            name
        );
        VarRef::Builtin {
            id: def.id,
            overload_idx: 0,
        }
    } else {
        let sym = if let Some(&existing) = program.def_syms.get(name) {
            existing
        } else {
            let sym = program.symbols.alloc(name.to_string());
            program.def_syms.insert(name.to_string(), sym);
            sym
        };
        VarRef::Symbol(sym)
    };
    let func_term = Term {
        id: term_ids.next_id(),
        ty: Type::Variable(0),
        span,
        kind: TermKind::Var(func_var),
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

fn soac_term(soac: SoacOp, ty: Type<TypeName>, span: ast::Span, term_ids: &mut TermIdSource) -> Term {
    Term {
        id: term_ids.next_id(),
        ty,
        span,
        kind: TermKind::Soac(soac),
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
            body: dummy_expr,
        })),
        arity: required_params.len(),
    }
}

// =============================================================================
// Pipeline-descriptor construction helpers
// =============================================================================

/// Append a storage-buffer binding to `bindings` and return its index.
fn push_storage_binding(
    bindings: &mut Vec<Binding>,
    br: BindingRef,
    access: Access,
    usage: BufferUsage,
    name: String,
) -> usize {
    let idx = bindings.len();
    bindings.push(Binding::StorageBuffer {
        set: br.set,
        binding: br.binding,
        access,
        usage,
        name,
        length: None,
    });
    idx
}

/// Resolve a parallel SOAC's `DispatchLen` from its provenance: a buffer
/// view → that buffer's binding; a range with a recognizable bound → the
/// bound's source (`Fixed` / `InputBinding(length(arr))`); a fixed-size
/// array → its static count; otherwise the entry's first input buffer
/// (computed/loaded arrays share its element count).
fn resolve_dispatch_len(analysis: &EntryAnalysis, input_index: usize, program: &Program) -> DispatchLen {
    let inputs = analysis.soac.inputs();
    match analysis.soac.provenances.get(input_index) {
        Some(ArrayProvenance::Storage {
            binding, elem_bytes, ..
        }) => DispatchLen::InputBinding {
            set: binding.set,
            binding: binding.binding,
            elem_bytes: *elem_bytes,
        },
        Some(ArrayProvenance::Range { bound }) => resolve_range_bound(bound, analysis.def_name, program)
            .unwrap_or_else(|| default_entry_dispatch_len(program, analysis.def_name)),
        Some(ArrayProvenance::Opaque) | None => inputs
            .get(input_index)
            .and_then(|ae| match ae {
                ArrayExpr::Ref(t) => fixed_array_count(&t.ty),
                _ => None,
            })
            .map(|count| DispatchLen::Fixed { count })
            .unwrap_or_else(|| default_entry_dispatch_len(program, analysis.def_name)),
    }
}

/// `iota(N)` count from the range's bound:
/// - integer literal → `Fixed{count}`;
/// - a `storage_len(IntLit set, IntLit binding)` call (or its `i32.u32`
///   scalar-cast wrapper, as emitted by `buffer_specialize` from
///   `length(view_param)`) → that binding's `InputBinding`;
/// otherwise `None` and the caller falls back to the entry's first buffer.
fn resolve_range_bound(bound: &Term, def_name: SymbolId, program: &Program) -> Option<DispatchLen> {
    if let TermKind::IntLit(s) = &bound.kind {
        return s.parse::<u32>().ok().map(|count| DispatchLen::Fixed { count });
    }
    let target = recognize_storage_len(bound)?;
    let elem_bytes =
        entry_binding_slots(program, def_name).into_iter().flatten().find_map(|b| match b.kind {
            EntryParamBindingKind::Single {
                binding, elem_bytes, ..
            } if binding == target => Some(elem_bytes),
            EntryParamBindingKind::TupleOfViews(fields) => {
                fields.into_iter().find(|f| f.binding == target).map(|f| f.elem_bytes)
            }
            _ => None,
        })?;
    Some(DispatchLen::InputBinding {
        set: target.set,
        binding: target.binding,
        elem_bytes,
    })
}

/// Match `storage_len(IntLit set, IntLit binding)` (the shape that
/// `buffer_specialize` emits for `length(view_param)`), unwrapping a
/// surrounding `i32.u32` cast if present. Returns the recognized binding.
fn recognize_storage_len(term: &Term) -> Option<BindingRef> {
    let TermKind::App { func, args } = &term.kind else {
        return None;
    };
    let id = match &func.kind {
        TermKind::Var(VarRef::Builtin { id, .. }) => *id,
        _ => return None,
    };
    if id == catalog().known().storage_len {
        if args.len() != 2 {
            return None;
        }
        let set = u32_int_lit(&args[0])?;
        let binding = u32_int_lit(&args[1])?;
        return Some(BindingRef::new(set, binding));
    }
    if catalog().lookup_by_any_name("i32.u32").map(|d| d.id) == Some(id) {
        if args.len() != 1 {
            return None;
        }
        return recognize_storage_len(&args[0]);
    }
    None
}

fn u32_int_lit(term: &Term) -> Option<u32> {
    match &term.kind {
        TermKind::IntLit(s) => s.parse::<u32>().ok(),
        _ => None,
    }
}

/// Static element count of a fixed-size `[N]T`, if any.
fn fixed_array_count(ty: &Type<TypeName>) -> Option<u32> {
    match crate::types::array_size(ty)? {
        Type::Constructed(TypeName::Size(n), _) => Some(*n as u32),
        _ => None,
    }
}

/// Entry's auto-storage binding slots, or empty for non-entry / non-compute
/// defs. Single source of truth for the param→binding lookups used by both
/// `default_entry_dispatch_len` and the SOAC-input provenance resolver.
fn entry_binding_slots(program: &Program, def_name: SymbolId) -> Vec<Option<EntryParamBinding>> {
    let Some(def) = program.defs.iter().find(|d| d.name == def_name) else {
        return Vec::new();
    };
    let DefMeta::EntryPoint(decl) = &def.meta else {
        return Vec::new();
    };
    let (params, _) = peel_lambda_params(&def.body);
    crate::binding_layout::compute_entry_binding_layout(&params, decl, AUTO_STORAGE_SET)
}

/// Dispatch length for a non-parallelized compute entry:
///   1. its first input-view param binding, if any (the previous
///      "derive from the first input buffer" behavior, now explicit);
///   2. else, between its first explicit `#[storage(set, binding, ...)]`
///      view-array param and its first `#[storage_image]` param,
///      whichever matches the entry's *shape*:
///        - an entry that **produces an array** (`map`-shaped, even when
///          a side effect kept it off the parallel path) iterates that
///          input view — derive from the explicit storage view-array;
///        - an entry that **returns `()`** is a manually-`gid`-driven
///          image writer — derive per texel from the storage image. An
///          explicit `#[storage]` param here is usually a fixed-index
///          *side input* (e.g. the keyboard-state buffer), not the
///          iteration domain.
///      (A `map` that iterates a storage view *and* writes an image and
///      stays on the parallel path resolves its length from the map's
///      input provenance — `resolve_dispatch_len` — so it never reaches
///      here.)
///   3. else a single-element grid (the conservative fallback for
///      entries that drive themselves entirely from push constants /
///      uniforms).
///
/// Shaders that need a different sizing source than this heuristic
/// picks can override it at runtime with `--dispatch ENTRY:WxH`.
fn default_entry_dispatch_len(program: &Program, def_name: SymbolId) -> DispatchLen {
    let slots = entry_binding_slots(program, def_name);
    if let Some(binding) = slots.iter().flatten().next() {
        let (buf, _elem_ty, elem_bytes) = binding.first_buffer();
        return DispatchLen::InputBinding {
            set: buf.set,
            binding: buf.binding,
            elem_bytes,
        };
    }
    let def = program.defs.iter().find(|d| d.name == def_name);
    if let Some(def) = def {
        if let DefMeta::EntryPoint(decl) = &def.meta {
            let (body_params, _) = peel_lambda_params(&def.body);

            // First explicit `#[storage]` view-array param (host-wired;
            // first declared wins) and first written `#[storage_image]`.
            let explicit_view = decl.params.iter().enumerate().find_map(|(i, pattern)| {
                let br = crate::binding_layout::extract_storage_binding(pattern)?;
                let (_, ty) = body_params.get(i)?;
                let (_elem_ty, elem_bytes) = crate::binding_layout::runtime_sized_array_elem(ty)?;
                Some(DispatchLen::InputBinding {
                    set: br.set,
                    binding: br.binding,
                    elem_bytes,
                })
            });
            let storage_image = decl.params.iter().find_map(|pattern| {
                let (br, _fmt, _access, _size) =
                    crate::binding_layout::extract_storage_image_binding(pattern)?;
                Some(DispatchLen::StorageImage {
                    set: br.set,
                    binding: br.binding,
                })
            });

            // An array-producing entry (`map`-shaped) iterates its input
            // view; a `()`-returning one writes the image per texel. Order
            // the two candidates by that shape, then take the first present.
            let produces_array = decl.outputs.iter().any(|o| crate::types::array_size(&o.ty).is_some());
            let (primary, secondary) = if produces_array {
                (explicit_view, storage_image)
            } else {
                (storage_image, explicit_view)
            };
            if let Some(len) = primary.or(secondary) {
                return len;
            }
        }
    }
    DispatchLen::Fixed { count: 1 }
}

/// A `ComputeStage` with `workgroup` + a `DerivedFrom(len)` dispatch. Used by
/// per-element phases (phase 1 of reduce; phases 1+3 of scan).
fn derived_stage(
    entry_point: String,
    reads: Vec<usize>,
    writes: Vec<usize>,
    workgroup: (u32, u32, u32),
    len: DispatchLen,
) -> ComputeStage {
    ComputeStage {
        entry_point,
        workgroup_size: workgroup,
        dispatch_size: DispatchSize::DerivedFrom {
            len,
            workgroup_size: workgroup.0,
        },
        reads,
        writes,
    }
}

/// A `ComputeStage` for a reduce/redomap phase 1: a fixed, hardware-saturating
/// grid (`PHASE1_SATURATING_GROUPS` workgroups), *not* derived from input
/// length. The kernel grid-strides over the input via the `num_workgroups`
/// intrinsic, so a bounded grid yields a bounded `partials` count — one partial
/// per worker — instead of one per input element.
fn saturating_stage(
    entry_point: String,
    reads: Vec<usize>,
    writes: Vec<usize>,
    workgroup: (u32, u32, u32),
) -> ComputeStage {
    ComputeStage {
        entry_point,
        workgroup_size: workgroup,
        dispatch_size: DispatchSize::Fixed {
            x: PHASE1_SATURATING_GROUPS,
            y: 1,
            z: 1,
        },
        reads,
        writes,
    }
}

/// A `ComputeStage` with a `(1, 1, 1)` workgroup and fixed `1×1×1`
/// dispatch — used by single-threaded combine phases (phase 2 of
/// reduce; phase 2 of scan).
fn fixed_stage(entry_point: String, reads: Vec<usize>, writes: Vec<usize>) -> ComputeStage {
    ComputeStage {
        entry_point,
        workgroup_size: (1, 1, 1),
        dispatch_size: DispatchSize::Fixed { x: 1, y: 1, z: 1 },
        reads,
        writes,
    }
}

/// A reduce/redomap phase 2 stage: one workgroup of `PHASE2_WIDTH` threads
/// (`LocalSize(W,1,1)`, dispatch `[1,1,1]`) that tree-reduces the partials in
/// workgroup-shared memory. Mirrors the `W` baked into the synthesized phase2
/// entry (`egir::parallelize::build_tree_reduce_phase2`).
fn tree_phase2_stage(entry_point: String, reads: Vec<usize>, writes: Vec<usize>) -> ComputeStage {
    ComputeStage {
        entry_point,
        workgroup_size: (crate::egir::parallelize::PHASE2_WIDTH, 1, 1),
        dispatch_size: DispatchSize::Fixed { x: 1, y: 1, z: 1 },
        reads,
        writes,
    }
}

/// Iterate over the SOAC's storage-backed inputs, yielding
/// `(positional_index, set, binding, elem_ty)` for each. Provenances
/// that aren't `ArrayProvenance::Storage` (e.g. ranges) are skipped.
/// Single source of truth for the two adapters below.
fn storage_inputs(soac: &SoacAnalysis) -> impl Iterator<Item = (usize, BindingRef, &Type<TypeName>)> + '_ {
    soac.provenances.iter().enumerate().filter_map(|(i, p)| match p {
        ArrayProvenance::Storage { binding, elem_ty, .. } => Some((i, *binding, elem_ty)),
        _ => None,
    })
}

/// Pipeline-level `Binding` descriptors for the SOAC's storage inputs.
fn collect_soac_bindings(soac: &SoacAnalysis) -> Vec<Binding> {
    storage_inputs(soac)
        .map(|(i, br, _)| Binding::StorageBuffer {
            set: br.set,
            binding: br.binding,
            access: Access::ReadOnly,
            usage: BufferUsage::Input,
            name: format!("input_{}", i),
            length: None,
        })
        .collect()
}

/// Per-entry `StorageBindingDecl`s for the SOAC's storage inputs. Every
/// phase entry whose body reads those buffers must declare them so the
/// backend's binding allowlist admits the references.
fn input_storage_decls(soac: &SoacAnalysis) -> Vec<interface::StorageBindingDecl> {
    storage_inputs(soac)
        .map(|(_, br, elem_ty)| interface::StorageBindingDecl {
            binding: br,
            role: interface::StorageRole::Input,
            elem_ty: elem_ty.clone(),
            length: None,
        })
        .collect()
}

/// Collect every `BindingRef` already claimed anywhere in the
/// program so `parallelize::run` can hand out fresh intermediate bindings
/// that don't collide with anything. Walks every term, picking up
/// implicit bindings attached by earlier passes as `ArrayExpr::StorageView`
/// inside SOAC inputs — scan's three-way collision (partials, result,
/// input) would otherwise emit a broken shader.
fn collect_all_used_bindings(program: &Program) -> HashSet<BindingRef> {
    let mut used: HashSet<BindingRef> = HashSet::new();
    for def in &program.defs {
        collect_bindings_in_term(&def.body, &mut used);
    }
    used
}

fn collect_bindings_in_term(term: &Term, used: &mut HashSet<BindingRef>) {
    // At a TermKind wrapping ArrayExpr/Soac, inspect the wrapped shape so
    // `StorageView` bindings aren't skipped by `for_each_child` (which
    // only visits Term children and can't extract binding fields).
    match &term.kind {
        TermKind::ArrayExpr(ae) => collect_bindings_in_ae(ae, used),
        TermKind::Soac(op) => collect_bindings_in_soac(op, used),
        _ => {}
    }
    term.for_each_child(&mut |c| collect_bindings_in_term(c, used));
}

fn collect_bindings_in_ae(ae: &ArrayExpr, used: &mut HashSet<BindingRef>) {
    if let ArrayExpr::StorageView(crate::tlc::StorageView { binding: br, .. }) = ae {
        used.insert(*br);
    }
    match ae {
        ArrayExpr::Zip(aes) => {
            for a in aes {
                collect_bindings_in_ae(a, used);
            }
        }
        ArrayExpr::Soac(op) => collect_bindings_in_soac(op, used),
        _ => {}
    }
}

fn collect_bindings_in_soac(op: &SoacOp, used: &mut HashSet<BindingRef>) {
    match op {
        SoacOp::Map { inputs, .. } | SoacOp::Redomap { inputs, .. } | SoacOp::Screma { inputs, .. } => {
            for ae in inputs {
                collect_bindings_in_ae(ae, used);
            }
        }
        SoacOp::Reduce { input, .. } | SoacOp::Scan { input, .. } | SoacOp::Filter { input, .. } => {
            collect_bindings_in_ae(input, used);
        }
        SoacOp::Scatter { inputs, .. } => {
            for ae in inputs {
                collect_bindings_in_ae(ae, used);
            }
        }
        SoacOp::ReduceByIndex { indices, values, .. } => {
            collect_bindings_in_ae(indices, used);
            collect_bindings_in_ae(values, used);
        }
    }
}

fn build_default_pipeline(program: &Program) -> PipelineDescriptor {
    let mut pipelines = Vec::new();

    for def in &program.defs {
        if let DefMeta::EntryPoint(ref decl) = def.meta {
            let name = crate::symbol_name_or_bug(&program.symbols, def.name).to_string();
            if decl.entry_type.is_compute() {
                let sizing = PipelineSizing::for_default_entry(program, def.name);
                let len = default_entry_dispatch_len(program, def.name);
                pipelines.push(Pipeline::Compute(ComputePipeline {
                    entry_point: name,
                    workgroup_size: sizing.workgroup,
                    dispatch_size: DispatchSize::DerivedFrom {
                        len,
                        workgroup_size: sizing.workgroup.0,
                    },
                    default_total_threads: sizing.default_total_threads,
                    bindings: Vec::new(),
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
                }));
            }
        }
    }

    PipelineDescriptor { pipelines }
}

#[cfg(test)]
#[path = "parallelize_tests.rs"]
mod parallelize_tests;
