//! Materialize a randomly-indexed computed array into a storage buffer.
//!
//! Indexing a *computed* array (a `map` result) at a runtime index from
//! inside another SOAC has no SPIR-V value form — a computed array lowers to
//! an in-register, runtime-sized Composite array that can't be randomly
//! indexed (`spirv/mod.rs`: "Composite variant unsized arrays not supported").
//!
//! ```text
//! let counts = map(f, bh) in
//! map(|i| counts[i % 256], iota(N))      -- gather from a computed array
//! ```
//!
//! This pass splits the producer `map` into its own `<entry>_gather_<n>`
//! compute entry that writes a fresh storage binding, and rewrites each
//! `counts[idx]` in the consumer to a `_w_intrinsic_storage_index(set,
//! binding, idx)` load from that buffer. `from_tlc` already lowers an Index
//! over a buffer-backed view to `ViewIndex + Load`, so once the consumer
//! reads from a storage view the gather lowers with no further changes.
//!
//! It runs *before* `defunctionalize`, while the producer is still a clean
//! `Soac(Map)` and the consumed array is a plain `Var` (post-defunc it would
//! be a closure capture, awkward to recognize and rewrite).
//!
//! Phase 1 is conservative: only `let arr = map(..) in <body>` where `arr` is
//! used *solely* via dynamic `Index`, and the producer's free variables are
//! all input arrays (no captured uniforms/sizes yet).

use std::collections::{HashMap, HashSet};

use super::closure_convert::collect_free_vars;
use super::parallelize::make_entry_def;
use super::{
    ArrayExpr, Def, DefMeta, Lambda, Program, SoacBody, SoacOp, StorageView, Term, TermIdSource, TermKind,
    VarRef,
};
use crate::BindingRef;
use crate::ast::TypeName;
use crate::egir::from_tlc::AUTO_STORAGE_SET;
use crate::interface::{EntryParamBindingKind, StorageBindingDecl, StorageRole};
use crate::{SymbolId, SymbolTable};
use polytype::Type;

/// Indices of every compute entry def in `program`.
fn compute_entry_indices(program: &Program) -> Vec<usize> {
    program
        .defs
        .iter()
        .enumerate()
        .filter_map(|(i, d)| match &d.meta {
            DefMeta::EntryPoint(decl) if decl.entry_type.is_compute() => Some(i),
            _ => None,
        })
        .collect()
}

/// Lift every randomly-indexed computed array out of each compute entry into
/// a gather pre-pass + a `storage_index` read.
pub fn run(mut program: Program) -> Program {
    let mut new_defs: Vec<Def> = Vec::new();
    for idx in compute_entry_indices(&program) {
        lift_entry(&mut program, idx, &mut new_defs);
    }
    program.defs.extend(new_defs);
    program
}

/// Visibility normalization for post-materialize gather planning — the first
/// half of the post-`inline_small` / `materialize_entry_soacs` gather lift.
///
/// The pre-defunc [`run`] can't see a runtime-sized array produced inside a
/// helper — the producer only surfaces in the entry body once inlining exposes
/// it (the ordering hazard). Inlining a consumer leaves trivial alias lets —
/// `f32.sum(xs)` becomes `let p = xs in reduce(.., p)` — and [`rewrite_uses`]
/// bails on the bare `Var(xs)` in `p`'s RHS; a runtime-indexed *nested* producer
/// (`map(.., src)[j]`) isn't let-bound at all. This pass collapses those aliases
/// and floats nested indexed producers to entry-level lets, so a producer-plan
/// computed *after* it can see and authorize every late-exposed producer. It is
/// a no-op for entries fully lifted pre-defunc (only storage reads remain).
///
/// Split from the executor [`execute_gathers`] precisely so the authoritative
/// `producer_plan::plan_program` runs *between* them — normalization makes the
/// producers visible, the plan authorizes them, the executor materializes them.
pub fn normalize_for_gather(mut program: Program) -> Program {
    for idx in compute_entry_indices(&program) {
        program.defs[idx].body = inline_trivial_aliases(program.defs[idx].body.clone());
        float_nested_indexed_producers(&mut program, idx);
    }
    program
}

/// Executor half of the post-materialize gather lift: materialize each
/// late-exposed producer into a gather pre-pass + `storage_index` reads. Runs
/// after [`normalize_for_gather`] has exposed the producers (and, from Stage 7,
/// after the planner has authorized them).
pub fn execute_gathers(mut program: Program) -> Program {
    let mut new_defs: Vec<Def> = Vec::new();
    for idx in compute_entry_indices(&program) {
        lift_entry(&mut program, idx, &mut new_defs);
    }
    program.defs.extend(new_defs);
    program
}

/// Float a runtime-indexed *nested* elementwise producer to an entry-level
/// `let`, so `lift_entry`'s let-chain walk can materialize it.
///
/// `Index(map(.., src), j)` with a non-literal `j` has no fused form (only
/// constant indices fuse, in `static_index_fusion`), and the producer isn't
/// let-bound, so the gather lift never sees it and the runtime-sized array
/// panics in the backend. Rewrite `Index(<producer>, j)` →
/// `let t = <producer> in Index(Var(t), j)` with `t` pulled to the top of the
/// entry body. Only producers whose free vars are all entry params are floated
/// — those are safe to evaluate at entry top; anything referencing an
/// inner-bound var is left as-is (a remaining gap, not a miscompile).
fn float_nested_indexed_producers(program: &mut Program, idx: usize) {
    let (params, _) = peel_lambda_params(&program.defs[idx].body);
    let entry_params: HashSet<SymbolId> = params.iter().map(|(s, _)| *s).collect();

    // Pass 1 (immutable): collect hoistable `Index` sites by TermId.
    let body = program.defs[idx].body.clone();
    let mut sites: Vec<(super::TermId, Term)> = Vec::new();
    collect_float_sites(
        &body,
        &entry_params,
        &program.symbols,
        &program.def_syms,
        &mut sites,
    );
    if sites.is_empty() {
        return;
    }

    // Allocate a fresh binder per site, then rewrite each site's `Index` array
    // to `Var(fresh)` and wrap the body in the producer `let`s.
    let mut ids = TermIdSource::new();
    let floated: Vec<(super::TermId, SymbolId, Term)> = sites
        .into_iter()
        .enumerate()
        .map(|(i, (tid, producer))| {
            let sym = program.symbols.alloc(format!("_float_{}_{}", idx, i));
            (tid, sym, producer)
        })
        .collect();
    let id_to_sym: HashMap<super::TermId, SymbolId> =
        floated.iter().map(|(tid, sym, _)| (*tid, *sym)).collect();

    let mut wrapped = rewrite_float_sites(body, &id_to_sym, &mut ids);
    for (_, sym, producer) in floated.into_iter().rev() {
        let name_ty = producer.ty.clone();
        let span = producer.span;
        let body_ty = wrapped.ty.clone();
        wrapped = Term {
            id: ids.next_id(),
            ty: body_ty,
            span,
            kind: TermKind::Let {
                name: sym,
                name_ty,
                rhs: Box::new(producer),
                body: Box::new(wrapped),
            },
        };
    }
    program.defs[idx].body = wrapped;
}

/// Fully inline every `let x = v in body` into `body[x := v]`. Used only on a
/// small floated producer term, so the per-use duplication of `v` is bounded.
fn inline_lets(term: Term) -> Term {
    match term.kind {
        TermKind::Let { name, rhs, body, .. } => {
            let rhs = inline_lets(*rhs);
            let body = inline_lets(*body);
            subst_term(body, name, &rhs)
        }
        _ => term.map_children(&mut inline_lets),
    }
}

/// Replace every `Var(name)` in `term` with a clone of `repl`. `name` is one
/// globally-unique SymbolId, so the substitution needs no shadowing check.
fn subst_term(term: Term, name: SymbolId, repl: &Term) -> Term {
    if let TermKind::Var(VarRef::Symbol(s)) = &term.kind {
        if *s == name {
            return repl.clone();
        }
    }
    term.map_children(&mut |c| subst_term(c, name, repl))
}

/// True if `t`, after peeling enclosing `let`s, is a directly-nested
/// `Soac(Map)` — an elementwise producer that isn't reached through a `Var`.
fn peels_to_map(mut t: &Term) -> bool {
    loop {
        match &t.kind {
            TermKind::Let { body, .. } => t = body,
            TermKind::Soac(SoacOp::Map { .. }) => return true,
            _ => return false,
        }
    }
}

fn collect_float_sites(
    term: &Term,
    entry_params: &HashSet<SymbolId>,
    symbols: &SymbolTable,
    def_syms: &HashMap<String, SymbolId>,
    out: &mut Vec<(super::TermId, Term)>,
) {
    if let TermKind::Index { array, index } = &term.kind {
        // Constant indices are handled by `static_index_fusion`; only runtime
        // indices into a directly-nested producer need materialization here.
        let runtime_index = !matches!(index.kind, TermKind::IntLit(_));
        if runtime_index && peels_to_map(array) {
            // Inline the producer's internal `let`s so the floated `let t =`
            // binds a bare `Soac(Map)` directly (what `try_lift` recognizes) and
            // the map carries no free scalar binders (e.g. `let n = 256`) that
            // would defeat the gather lift's free-var contract.
            let producer = inline_lets((**array).clone());
            let frees = free_symbol_vars(&producer, symbols, def_syms);
            if frees.iter().all(|(s, _)| entry_params.contains(s)) {
                out.push((term.id, producer));
            }
        }
    }
    term.for_each_child(&mut |c| collect_float_sites(c, entry_params, symbols, def_syms, out));
}

fn rewrite_float_sites(
    term: Term,
    id_to_sym: &HashMap<super::TermId, SymbolId>,
    ids: &mut TermIdSource,
) -> Term {
    let term = term.map_children(&mut |c| rewrite_float_sites(c, id_to_sym, ids));
    if let TermKind::Index { array, index } = &term.kind {
        if let Some(&sym) = id_to_sym.get(&term.id) {
            let var = Term {
                id: ids.next_id(),
                ty: array.ty.clone(),
                span: array.span,
                kind: TermKind::Var(VarRef::Symbol(sym)),
            };
            return Term {
                id: term.id,
                ty: term.ty.clone(),
                span: term.span,
                kind: TermKind::Index {
                    array: Box::new(var),
                    index: index.clone(),
                },
            };
        }
    }
    term
}

/// Collapse `let p = q in body` where `q` is a bare `Var` — an alias inlining
/// leaves behind — into `body[p := q]`. SymbolIds are globally unique, so a
/// blanket `p → q` substitution needs no shadowing check.
fn inline_trivial_aliases(term: Term) -> Term {
    match term.kind {
        TermKind::Let {
            name,
            name_ty,
            rhs,
            body,
        } => {
            let rhs = inline_trivial_aliases(*rhs);
            if let TermKind::Var(VarRef::Symbol(q)) = rhs.kind {
                let body = subst_var(*body, name, q);
                return inline_trivial_aliases(body);
            }
            Term {
                kind: TermKind::Let {
                    name,
                    name_ty,
                    rhs: Box::new(rhs),
                    body: Box::new(inline_trivial_aliases(*body)),
                },
                ..term
            }
        }
        _ => term.map_children(&mut inline_trivial_aliases),
    }
}

/// Replace every `Var(from)` in `term` with `Var(to)`. Unconditional — `from`
/// is one globally-unique SymbolId bound exactly once, so it can't be shadowed.
fn subst_var(term: Term, from: SymbolId, to: SymbolId) -> Term {
    if let TermKind::Var(VarRef::Symbol(s)) = &term.kind {
        if *s == from {
            return Term {
                kind: TermKind::Var(VarRef::Symbol(to)),
                ..term
            };
        }
    }
    term.map_children(&mut |c| subst_var(c, from, to))
}

/// The gather-relevant storage layout of a compute entry: the `Var(sym) →
/// (set, binding)` map for its input-array params (gather producers reference
/// these by bare `Var`; tuple-of-views params aren't gather sources), and the
/// total input-view buffer count. `None` if `program.defs[idx]` isn't an entry.
/// Shared by [`lift_entry`] (the executor) and [`gather_decision`] callers.
pub(crate) fn entry_layout(program: &Program, idx: usize) -> Option<(HashMap<SymbolId, (u32, u32)>, u32)> {
    let (params, _tail) = peel_lambda_params(&program.defs[idx].body);
    let DefMeta::EntryPoint(decl) = &program.defs[idx].meta else {
        return None;
    };
    let slots = crate::binding_layout::compute_entry_binding_layout(&params, decl, AUTO_STORAGE_SET);
    let param_bindings: HashMap<SymbolId, (u32, u32)> = slots
        .iter()
        .flatten()
        .filter_map(|b| match &b.kind {
            EntryParamBindingKind::Single { binding, .. } => {
                Some((b.param_sym, (binding.set, binding.binding)))
            }
            EntryParamBindingKind::TupleOfViews(_) => None,
        })
        .collect();
    let view_count: u32 = slots.iter().flatten().map(|b| b.buffer_count()).sum();
    Some((param_bindings, view_count))
}

/// Lift gather sites out of a single compute entry at `program.defs[idx]`.
fn lift_entry(program: &mut Program, idx: usize, new_defs: &mut Vec<Def>) {
    let entry_name = crate::symbol_name_or_bug(&program.symbols, program.defs[idx].name).to_string();
    let body = program.defs[idx].body.clone();

    // Gather buffers must sit above the consumer's own auto-allocated
    // bindings: input-view params occupy `0..view_count`, and `from_tlc`
    // places this entry's storage outputs at `view_count..view_count +
    // out_count` (see `build_entry_outputs`). So the first free binding for a
    // gather intermediate is `view_count + out_count`.
    let Some((param_bindings, view_count)) = entry_layout(program, idx) else {
        return;
    };
    let decl = match &program.defs[idx].meta {
        DefMeta::EntryPoint(d) => (**d).clone(),
        _ => return,
    };
    // Number of storage-output slots the entry declared. Reading off
    // `decl.outputs.len()` is exact — one entry per declared output
    // (tuple-returning entries get one EntryOutput per field).
    //
    // We can NOT read this off `def.ty`'s arrow-return position even
    // though that LOOKS like the source of truth: `tlc::normalize_outputs`
    // rewrites the def.ty's return slot to `SideEffect` to match the
    // body's new `OutputSlotStore` tail, so any pass running after
    // normalize_outputs that reads `def.ty.return` sees `SideEffect`
    // and undercounts.
    //
    // Why this mattered: if `out_count` undercounts by N, the gather
    // intermediate `next_gather = view_count + out_count` lands on a
    // binding slot the (N-th) output expected, and the descriptor
    // emitter overwrites the output's binding entry with the gather's
    // intermediate role — the output slot silently vanishes from the
    // JSON descriptor (regression first surfaced by an entry returning
    // `([]vec4f32, [5]i32)` whose `[5]i32` literal indexed into a
    // `scan` result, gather-lifting the scan).
    let out_count = decl.outputs.len() as u32;
    // Gather buffers sit above the entry's input views and outputs, and above
    // any gather Input decls a prior lift run already attached to this entry:
    // the gather lift runs once pre-defunc (`run`) and again post-materialize
    // (`execute_gathers`), when helper-inlined producers first become
    // visible — the second run must not reuse the first run's binding numbers.
    let existing_max = decl
        .storage_bindings
        .iter()
        .filter(|b| b.binding.set == AUTO_STORAGE_SET)
        .map(|b| b.binding.binding + 1)
        .max()
        .unwrap_or(0);
    let mut next_gather = (view_count + out_count).max(existing_max);

    let mut added_decls: Vec<StorageBindingDecl> = Vec::new();
    let new_body = lift_in_term(
        body,
        &entry_name,
        &param_bindings,
        &mut next_gather,
        &mut added_decls,
        new_defs,
        program,
    );

    program.defs[idx].body = new_body;
    if let DefMeta::EntryPoint(ref mut d) = program.defs[idx].meta {
        d.storage_bindings.extend(added_decls);
    }
}

/// Walk the outer `Lambda`/`Let` chain, lifting eligible producer lets. Stops
/// descending at the first non-Lambda/non-Let term (the tail computation).
fn lift_in_term(
    term: Term,
    entry_name: &str,
    param_bindings: &HashMap<SymbolId, (u32, u32)>,
    next_gather: &mut u32,
    added_decls: &mut Vec<StorageBindingDecl>,
    new_defs: &mut Vec<Def>,
    program: &mut Program,
) -> Term {
    match term.kind {
        TermKind::Lambda(lam) => {
            let Lambda { params, body, ret_ty } = lam;
            let new_body = lift_in_term(
                *body,
                entry_name,
                param_bindings,
                next_gather,
                added_decls,
                new_defs,
                program,
            );
            Term {
                kind: TermKind::Lambda(Lambda {
                    params,
                    body: Box::new(new_body),
                    ret_ty,
                }),
                ..term
            }
        }
        TermKind::Let {
            name,
            name_ty,
            rhs,
            body,
        } => {
            if let Some((prepass, decl, rewritten_body)) = try_lift(
                name,
                &name_ty,
                &rhs,
                *body.clone(),
                entry_name,
                param_bindings,
                *next_gather,
                new_defs.len(),
                program,
            ) {
                *next_gather += 1;
                new_defs.push(prepass);
                added_decls.push(decl);
                // The dropped `let` is gone; keep lifting in the rewritten body.
                return lift_in_term(
                    rewritten_body,
                    entry_name,
                    param_bindings,
                    next_gather,
                    added_decls,
                    new_defs,
                    program,
                );
            }
            let new_body = lift_in_term(
                *body,
                entry_name,
                param_bindings,
                next_gather,
                added_decls,
                new_defs,
                program,
            );
            // After normalize_outputs, the rhs of a sequencing let is an
            // OutputSlotStore that may wrap further gather candidates
            // (`let _ = OutputSlotStore(i, let counts = map(...) in …)`).
            // Descend so the inner producer still gets lifted.
            let new_rhs = if matches!(rhs.kind, TermKind::OutputSlotStore { .. }) {
                Box::new(lift_in_term(
                    *rhs,
                    entry_name,
                    param_bindings,
                    next_gather,
                    added_decls,
                    new_defs,
                    program,
                ))
            } else {
                rhs
            };
            Term {
                kind: TermKind::Let {
                    name,
                    name_ty,
                    rhs: new_rhs,
                    body: Box::new(new_body),
                },
                ..term
            }
        }
        TermKind::OutputSlotStore { slot_index, value } => {
            let new_value = lift_in_term(
                *value,
                entry_name,
                param_bindings,
                next_gather,
                added_decls,
                new_defs,
                program,
            );
            Term {
                kind: TermKind::OutputSlotStore {
                    slot_index,
                    value: Box::new(new_value),
                },
                ..term
            }
        }
        _ => term,
    }
}

/// The gather-residency decision for one `let name = rhs in body`. The single
/// authority over *whether* a computed array becomes a gather buffer: the
/// executor ([`try_lift`]) trusts it, and `producer_plan` surfaces it as the
/// planner's `Strategy::StoragePrepass(Gather)`. Total — it never silently
/// declines a demanded-but-impossible gather:
///
///   * [`GatherDecision::Gather`] — semantic *and* capability gates pass; the
///     executor is then guaranteed to materialize it (no `bail` downstream).
///   * [`GatherDecision::Unsupported`] — the array is demanded as a gather
///     (randomly indexed / multi-consumed) but a capability gate fails (a
///     producer free var isn't an entry-param array, or a use can't be rewritten
///     to a storage read). Carries a reason; the executor leaves it in place for
///     the Stage-1 `from_tlc` clean-reject to catch.
///   * [`GatherDecision::NotGather`] — not a gather producer at all (not a
///     runtime-sized map/scan, or consumed only wholesale, not per element).
pub(crate) enum GatherDecision {
    Gather(GatherPlan),
    Unsupported(String),
    NotGather,
}

/// Everything the executor needs to materialize a [`GatherDecision::Gather`],
/// computed once by [`gather_decision`] so the executor re-derives nothing.
pub(crate) struct GatherPlan {
    /// `body` with every `name[idx]` rewritten to a `storage_index` load and
    /// every `Ref(Var(name))` SOAC input to a storage-view read.
    rewritten: Term,
    frees: Vec<(SymbolId, Type<TypeName>)>,
    elem_ty: Type<TypeName>,
    length: Option<crate::pipeline_descriptor::BufferLen>,
    chained: Vec<(u32, u32, Type<TypeName>)>,
    binding: (u32, u32),
    name_ty: Type<TypeName>,
    rhs: Term,
}

/// Decide a `let name = rhs in body` gather site — read-only over `program`, so
/// it doubles as the planner's pure query and the executor's gate. `binding_num`
/// is the storage binding the rewrite would target; its value doesn't affect the
/// decision (only the rewritten body the executor consumes).
pub(crate) fn gather_decision(
    name: SymbolId,
    name_ty: &Type<TypeName>,
    rhs: &Term,
    body: Term,
    param_bindings: &HashMap<SymbolId, (u32, u32)>,
    binding_num: u32,
    program: &Program,
) -> GatherDecision {
    // Producer must be an array-yielding SOAC (`map` or `scan`) of a
    // runtime-sized array. Both preserve element count, so the gather buffer
    // tracks the producer's input length.
    let is_array_producer = matches!(
        &rhs.kind,
        TermKind::Soac(SoacOp::Map { .. }) | TermKind::Soac(SoacOp::Scan { .. })
    );
    if !is_array_producer || !is_runtime_sized_array(name_ty) {
        return GatherDecision::NotGather;
    }
    let Some(elem_ty) = crate::types::array_elem(name_ty).cloned() else {
        return GatherDecision::NotGather;
    };

    // The pre-pass re-declares the producer's free variables as its own input
    // params, so every one must be an entry-param input array (Phase 1). A
    // free var that's a let-bound computed array or a uniform/size would need
    // its definition pulled in (or a uniform re-declared) — deferred. This
    // keeps the pre-pass self-contained over real buffers. (Fusion has already
    // folded any `map` producer into the scan/map, so a `scan(op, ne, map(g,
    // xs))` arrives reading `xs` directly.)
    let frees = free_symbol_vars(rhs, &program.symbols, &program.def_syms);
    if let Some((s, _)) =
        frees.iter().find(|(s, ty)| !(is_runtime_sized_array(ty) && param_bindings.contains_key(s)))
    {
        let fname = crate::symbol_name_or_bug(&program.symbols, *s);
        return GatherDecision::Unsupported(format!(
            "gather producer free var `{fname}` is not a runtime-sized entry-param array"
        ));
    }

    // Rewrite `name[idx]` → storage_index and `ArrayExpr::Ref(Var(name))`
    // → `ArrayExpr::StorageBuffer{…}`; bail if `name` is used any other way (a
    // bare Var in a non-SOAC-input position), or if no use is a dynamic index
    // nor a SOAC input. Multi-consumer is fine: every downstream Index *and*
    // every downstream SOAC input gets routed to the same gather buffer.
    let binding = (AUTO_STORAGE_SET, binding_num);
    let mut bail = false;
    let mut dyn_uses = 0usize;
    let mut soac_uses = 0usize;
    // Local ID source for the synthesized literals + App nodes. Per-pass
    // restart matches the rest of the lift-gathers / parallelize style;
    // TLC TermIds aren't load-bearing past parallelize.
    let mut term_ids = TermIdSource::new();
    let rewritten = rewrite_uses(
        body,
        name,
        binding,
        &elem_ty,
        &mut bail,
        &mut dyn_uses,
        &mut soac_uses,
        &mut term_ids,
    );
    if bail {
        let cname = crate::symbol_name_or_bug(&program.symbols, name);
        return GatherDecision::Unsupported(format!(
            "computed array `{cname}` is used in a position that can't read from a storage buffer"
        ));
    }
    if dyn_uses + soac_uses == 0 {
        // Consumed only wholesale (never randomly indexed nor as a SOAC input):
        // not a gather. Some other strategy (fuse / view) owns it.
        return GatherDecision::NotGather;
    }

    // The gather buffer holds `map(f, src)`'s output: one element per `src`
    // element, so its length tracks `src`'s element count (element sizes may
    // differ). `from_tlc` allocates the host buffer from this policy.
    let length = gather_length(&elem_ty, &frees, param_bindings);

    // Chained intermediates: if the producer reads any `StorageBuffer{set,
    // binding, …}` directly (e.g. its input was itself a previously-lifted
    // gather buffer), the pre-pass must declare each as its own Input so the
    // descriptor wires up the cross-stage read.
    let chained = producer_storage_inputs(rhs);

    GatherDecision::Gather(GatherPlan {
        rewritten,
        frees,
        elem_ty,
        length,
        chained,
        binding,
        name_ty: name_ty.clone(),
        rhs: rhs.clone(),
    })
}

/// Execute the gather residency decision for `let name = rhs in body`. Returns
/// the gather pre-pass def, the consumer's new Input binding decl, and the
/// rewritten body, or `None` when [`gather_decision`] declines (`NotGather`) or
/// reports the site `Unsupported` (left in place for the Stage-1 clean-reject).
/// On `Gather` the executor *trusts* the decision: `gather_decision` already
/// rewrote the body, so no `bail` can occur here.
fn try_lift(
    name: SymbolId,
    name_ty: &Type<TypeName>,
    rhs: &Term,
    body: Term,
    entry_name: &str,
    param_bindings: &HashMap<SymbolId, (u32, u32)>,
    binding_num: u32,
    gather_idx: usize,
    program: &mut Program,
) -> Option<(Def, StorageBindingDecl, Term)> {
    let plan = match gather_decision(name, name_ty, rhs, body, param_bindings, binding_num, program) {
        GatherDecision::Gather(plan) => plan,
        GatherDecision::Unsupported(reason) => {
            log::debug!("lift_gathers: {reason}");
            return None;
        }
        GatherDecision::NotGather => return None,
    };

    let mut term_ids = TermIdSource::new();
    let prepass = build_gather_prepass(
        entry_name,
        gather_idx,
        plan.rhs,
        plan.name_ty,
        &plan.frees,
        plan.binding,
        plan.length.clone(),
        &plan.chained,
        program,
        &mut term_ids,
    );
    // The consumer *reads* the gather buffer: an Input-role decl carrying the
    // sizing policy. `from_tlc` emits any length-bearing storage binding as a
    // compiler-managed Intermediate in the descriptor (read-only here).
    let decl = StorageBindingDecl {
        binding: crate::BindingRef::new(plan.binding.0, plan.binding.1),
        role: StorageRole::Input,
        elem_ty: plan.elem_ty,
        length: plan.length,
    };
    Some((prepass, decl, plan.rewritten))
}

/// Sizing policy for the gather buffer: `LikeInput` of the producer's first
/// input array (a `map` preserves element count). `None` if that input isn't
/// a known param binding or element sizes can't be computed — the runtime
/// then falls back to its default intermediate size.
fn gather_length(
    elem_ty: &Type<TypeName>,
    producer_inputs: &[(SymbolId, Type<TypeName>)],
    param_bindings: &HashMap<SymbolId, (u32, u32)>,
) -> Option<crate::pipeline_descriptor::BufferLen> {
    let elem_bytes = crate::ssa::layout::type_byte_size(elem_ty)?;
    let (src_sym, src_ty) = producer_inputs.first()?;
    let (set, binding) = param_bindings.get(src_sym).copied()?;
    let src_elem_ty = crate::types::array_elem(src_ty)?;
    let src_elem_bytes = crate::ssa::layout::type_byte_size(src_elem_ty)?;
    Some(crate::pipeline_descriptor::BufferLen::LikeInput {
        set,
        binding,
        elem_bytes,
        src_elem_bytes,
    })
}

/// Recursively replace each use of `arr` with a read from the gather buffer.
/// The walk is **context-aware** via mutual recursion with
/// [`rewrite_soac`] / [`rewrite_array_input`]:
///
///   * In a generic `Term` position (here): an `Index { array: Var(arr), index
///     }` is rewritten to `_w_intrinsic_storage_index(set, binding, index)`
///     (tallied in `dyn_uses`); a bare `Var(arr)` sets `bail` (we can't lower
///     a runtime-sized Composite threaded through arbitrary term positions);
///     a `Soac` is split into "inputs → materializable" + "lambda bodies / ne
///     → generic", handled by [`rewrite_soac`].
///   * In an *ArrayExpr input* position (see [`rewrite_array_input`]): a
///     `Ref(Var(arr))` at any nesting depth — including nested inside a `Zip`
///     or another `Soac`'s own input — is rewritten to
///     `StorageBuffer{set,binding,offset:0,len:storage_len,elem_ty}` (tallied
///     in `soac_uses`). The materializable-position frame propagates down
///     through `Zip` and nested `Soac` inputs but flips back to a generic
///     `Term` context when we cross into a `Ref(non-Var)`'s wrapped Term, a
///     `Range`'s `start`/`len`/`step`, a `Literal`'s inner terms, or a
///     `StorageBuffer`'s `offset`/`len` — i.e. wherever the surrounding
///     position is no longer "a SOAC reading this array."
fn rewrite_uses(
    term: Term,
    arr: SymbolId,
    binding: (u32, u32),
    elem_ty: &Type<TypeName>,
    bail: &mut bool,
    dyn_uses: &mut usize,
    soac_uses: &mut usize,
    term_ids: &mut TermIdSource,
) -> Term {
    if let TermKind::Index { array, index } = &term.kind {
        if matches!(&array.kind, TermKind::Var(VarRef::Symbol(s)) if *s == arr) {
            let idx = rewrite_uses(
                (**index).clone(),
                arr,
                binding,
                elem_ty,
                bail,
                dyn_uses,
                soac_uses,
                term_ids,
            );
            if !matches!(idx.kind, TermKind::IntLit(_)) {
                *dyn_uses += 1;
            }
            return super::storage_index_call(
                crate::BindingRef::new(binding.0, binding.1),
                idx,
                elem_ty.clone(),
                term.span,
                term_ids,
            );
        }
    }

    // SOAC: hand off to the context-aware soac walker so inputs are treated
    // as a materializable ArrayExpr position (intercepting `Ref(Var(arr))` at
    // any nesting depth) while lambda bodies / ne / etc. stay in the generic
    // Term context.
    if let TermKind::Soac(_) = &term.kind {
        let Term { id, ty, span, kind } = term;
        let TermKind::Soac(soac) = kind else {
            unreachable!()
        };
        let new_soac = rewrite_soac(
            soac, arr, binding, elem_ty, bail, dyn_uses, soac_uses, span, term_ids,
        );
        return Term {
            id,
            ty,
            span,
            kind: TermKind::Soac(new_soac),
        };
    }

    if matches!(&term.kind, TermKind::Var(VarRef::Symbol(s)) if *s == arr) {
        *bail = true;
        return term;
    }
    term.map_children(&mut |c| rewrite_uses(c, arr, binding, elem_ty, bail, dyn_uses, soac_uses, term_ids))
}

/// Walk a `SoacOp`, splitting inputs (materializable ArrayExpr position) from
/// other components (generic Term position). Inputs go through
/// [`rewrite_array_input`]; lambda bodies / ne / etc. go through
/// [`rewrite_uses`].
#[allow(clippy::too_many_arguments)]
fn rewrite_soac(
    soac: SoacOp,
    arr: SymbolId,
    binding: (u32, u32),
    elem_ty: &Type<TypeName>,
    bail: &mut bool,
    dyn_uses: &mut usize,
    soac_uses: &mut usize,
    span: crate::ast::Span,
    term_ids: &mut TermIdSource,
) -> SoacOp {
    match soac {
        SoacOp::Map {
            lam,
            inputs,
            destination,
        } => SoacOp::Map {
            lam: rewrite_soac_body(lam, arr, binding, elem_ty, bail, dyn_uses, soac_uses, term_ids),
            inputs: inputs
                .into_iter()
                .map(|ae| {
                    rewrite_array_input(
                        ae, arr, binding, elem_ty, bail, dyn_uses, soac_uses, span, term_ids,
                    )
                })
                .collect(),
            destination,
        },
        SoacOp::Reduce { op, ne, input } => SoacOp::Reduce {
            op: rewrite_soac_body(op, arr, binding, elem_ty, bail, dyn_uses, soac_uses, term_ids),
            ne: Box::new(rewrite_uses(
                *ne, arr, binding, elem_ty, bail, dyn_uses, soac_uses, term_ids,
            )),
            input: rewrite_array_input(
                input, arr, binding, elem_ty, bail, dyn_uses, soac_uses, span, term_ids,
            ),
        },
        SoacOp::Scan {
            op,
            reduce_op,
            ne,
            input,
            destination,
        } => SoacOp::Scan {
            op: rewrite_soac_body(op, arr, binding, elem_ty, bail, dyn_uses, soac_uses, term_ids),
            reduce_op: rewrite_soac_body(
                reduce_op, arr, binding, elem_ty, bail, dyn_uses, soac_uses, term_ids,
            ),
            ne: Box::new(rewrite_uses(
                *ne, arr, binding, elem_ty, bail, dyn_uses, soac_uses, term_ids,
            )),
            input: rewrite_array_input(
                input, arr, binding, elem_ty, bail, dyn_uses, soac_uses, span, term_ids,
            ),
            destination,
        },
        SoacOp::Redomap {
            op,
            reduce_op,
            ne,
            inputs,
        } => SoacOp::Redomap {
            op: rewrite_soac_body(op, arr, binding, elem_ty, bail, dyn_uses, soac_uses, term_ids),
            reduce_op: rewrite_soac_body(
                reduce_op, arr, binding, elem_ty, bail, dyn_uses, soac_uses, term_ids,
            ),
            ne: Box::new(rewrite_uses(
                *ne, arr, binding, elem_ty, bail, dyn_uses, soac_uses, term_ids,
            )),
            inputs: inputs
                .into_iter()
                .map(|ae| {
                    rewrite_array_input(
                        ae, arr, binding, elem_ty, bail, dyn_uses, soac_uses, span, term_ids,
                    )
                })
                .collect(),
        },
        SoacOp::Filter {
            pred,
            input,
            destination,
        } => SoacOp::Filter {
            pred: rewrite_soac_body(pred, arr, binding, elem_ty, bail, dyn_uses, soac_uses, term_ids),
            input: rewrite_array_input(
                input, arr, binding, elem_ty, bail, dyn_uses, soac_uses, span, term_ids,
            ),
            destination,
        },
        // Scatter / ReduceByIndex carry ArrayExpr indices+values too. Defer
        // their materializable-position treatment until those SOACs are
        // wired up end-to-end; for now fall back to the generic descent so
        // an `arr` reference inside one cleanly bails.
        other => super::map_soac_children(other, &mut |t| {
            rewrite_uses(t, arr, binding, elem_ty, bail, dyn_uses, soac_uses, term_ids)
        }),
    }
}

/// Walk a `SoacBody` (a SOAC's lambda + captures). The lambda body and each
/// capture term sit in the generic `Term` context.
#[allow(clippy::too_many_arguments)]
fn rewrite_soac_body(
    sb: SoacBody,
    arr: SymbolId,
    binding: (u32, u32),
    elem_ty: &Type<TypeName>,
    bail: &mut bool,
    dyn_uses: &mut usize,
    soac_uses: &mut usize,
    term_ids: &mut TermIdSource,
) -> SoacBody {
    SoacBody {
        lam: rewrite_lambda(sb.lam, arr, binding, elem_ty, bail, dyn_uses, soac_uses, term_ids),
        captures: sb
            .captures
            .into_iter()
            .map(|(s, ty, t)| {
                (
                    s,
                    ty,
                    rewrite_uses(t, arr, binding, elem_ty, bail, dyn_uses, soac_uses, term_ids),
                )
            })
            .collect(),
    }
}

#[allow(clippy::too_many_arguments)]
fn rewrite_lambda(
    lam: Lambda,
    arr: SymbolId,
    binding: (u32, u32),
    elem_ty: &Type<TypeName>,
    bail: &mut bool,
    dyn_uses: &mut usize,
    soac_uses: &mut usize,
    term_ids: &mut TermIdSource,
) -> Lambda {
    Lambda {
        params: lam.params,
        body: Box::new(rewrite_uses(
            *lam.body, arr, binding, elem_ty, bail, dyn_uses, soac_uses, term_ids,
        )),
        ret_ty: lam.ret_ty,
    }
}

/// Walk an `ArrayExpr` in **materializable position** (a SOAC input slot,
/// directly or nested through `Zip` / nested `Soac` inputs). A bare
/// `Ref(Var(arr))` here is the canonical "consume `arr` as a SOAC input"
/// pattern: rewrite it to a `StorageView` read of the gather buffer (and
/// bump `soac_uses`) — at *any* depth, including under a `Zip` or another
/// `Soac`'s `Map(inputs=[Ref(Var(arr))])`. The materializable frame
/// propagates down `Zip` and `Soac`-input children; everywhere else (the
/// wrapped term inside a non-matching `Ref`, a `Range`'s components, a
/// `Literal`'s elements, a `StorageView`'s `offset`/`len`) we flip back to
/// the generic `Term` context and route through [`rewrite_uses`].
#[allow(clippy::too_many_arguments)]
fn rewrite_array_input(
    ae: ArrayExpr,
    arr: SymbolId,
    binding: (u32, u32),
    elem_ty: &Type<TypeName>,
    bail: &mut bool,
    dyn_uses: &mut usize,
    soac_uses: &mut usize,
    span: crate::ast::Span,
    term_ids: &mut TermIdSource,
) -> ArrayExpr {
    match ae {
        ArrayExpr::Ref(t) => {
            if matches!(&t.kind, TermKind::Var(VarRef::Symbol(s)) if *s == arr) {
                *soac_uses += 1;
                let bref = BindingRef::new(binding.0, binding.1);
                let len = super::storage_len_call(bref, span, term_ids);
                ArrayExpr::StorageView(StorageView {
                    binding: bref,
                    offset: Box::new(uint_lit(0, span, term_ids)),
                    len: Box::new(len),
                    elem_ty: elem_ty.clone(),
                })
            } else {
                // The wrapped term is a generic Term context (not itself an
                // input position) — descend via `rewrite_uses`.
                ArrayExpr::Ref(Box::new(rewrite_uses(
                    *t, arr, binding, elem_ty, bail, dyn_uses, soac_uses, term_ids,
                )))
            }
        }
        ArrayExpr::Zip(children) => ArrayExpr::Zip(
            children
                .into_iter()
                .map(|c| {
                    rewrite_array_input(
                        c, arr, binding, elem_ty, bail, dyn_uses, soac_uses, span, term_ids,
                    )
                })
                .collect(),
        ),
        ArrayExpr::Soac(boxed) => ArrayExpr::Soac(Box::new(rewrite_soac(
            *boxed, arr, binding, elem_ty, bail, dyn_uses, soac_uses, span, term_ids,
        ))),
        ArrayExpr::Literal(terms) => ArrayExpr::Literal(
            terms
                .into_iter()
                .map(|t| rewrite_uses(t, arr, binding, elem_ty, bail, dyn_uses, soac_uses, term_ids))
                .collect(),
        ),
        ArrayExpr::Range { start, len, step } => ArrayExpr::Range {
            start: Box::new(rewrite_uses(
                *start, arr, binding, elem_ty, bail, dyn_uses, soac_uses, term_ids,
            )),
            len: Box::new(rewrite_uses(
                *len, arr, binding, elem_ty, bail, dyn_uses, soac_uses, term_ids,
            )),
            step: step.map(|s| {
                Box::new(rewrite_uses(
                    *s, arr, binding, elem_ty, bail, dyn_uses, soac_uses, term_ids,
                ))
            }),
        },
        ArrayExpr::StorageView(sv) => ArrayExpr::StorageView(StorageView {
            binding: sv.binding,
            offset: Box::new(rewrite_uses(
                *sv.offset, arr, binding, elem_ty, bail, dyn_uses, soac_uses, term_ids,
            )),
            len: Box::new(rewrite_uses(
                *sv.len, arr, binding, elem_ty, bail, dyn_uses, soac_uses, term_ids,
            )),
            elem_ty: sv.elem_ty,
        }),
    }
}

/// Build the `<entry>_gather_<n>` compute entry whose body is the producer
/// `map`, re-declaring each captured input array as its own view param and
/// pinning the result to `binding` via an Output storage decl (which
/// `parallelize::make_map_plan` reads to force the map's output buffer; the
/// `length` makes `from_tlc` emit it as a compiler-managed Intermediate).
fn build_gather_prepass(
    entry_name: &str,
    gather_idx: usize,
    producer: Term,
    result_ty: Type<TypeName>,
    captured_inputs: &[(SymbolId, Type<TypeName>)],
    binding: (u32, u32),
    length: Option<crate::pipeline_descriptor::BufferLen>,
    chained_intermediates: &[(u32, u32, Type<TypeName>)],
    program: &mut Program,
    term_ids: &mut TermIdSource,
) -> Def {
    let name = format!("{}_gather_{}", entry_name, gather_idx);
    let elem_ty = crate::types::array_elem(&result_ty)
        .cloned()
        .expect("try_lift's is_runtime_sized_array(name_ty) gate guarantees an array elem");
    let required_params: Vec<super::parallelize::RequiredParam> = captured_inputs
        .iter()
        .map(|(s, ty)| super::parallelize::RequiredParam {
            sym: *s,
            ty: ty.clone(),
            attr: None,
            binding: None,
        })
        .collect();
    let mut storage_bindings = vec![StorageBindingDecl {
        binding: BindingRef::new(binding.0, binding.1),
        role: StorageRole::Output,
        elem_ty,
        length,
    }];
    for (set, binding, elem_ty) in chained_intermediates {
        storage_bindings.push(StorageBindingDecl {
            binding: BindingRef::new(*set, *binding),
            role: StorageRole::Input,
            elem_ty: elem_ty.clone(),
            // Length policy was already attached to this binding by whichever
            // earlier `try_lift` created it; the descriptor uses that
            // canonical entry and falls back if absent (Phase 2 plumbing).
            length: None,
        });
    }
    make_entry_def(
        &name,
        producer,
        result_ty,
        &required_params,
        storage_bindings,
        program,
        term_ids,
    )
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Peel outer `Lambda` layers, returning the flattened params and the tail.
fn peel_lambda_params(term: &Term) -> (Vec<(SymbolId, Type<TypeName>)>, &Term) {
    match &term.kind {
        TermKind::Lambda(lam) => {
            let (mut inner, tail) = peel_lambda_params(&lam.body);
            let mut params = lam.params.clone();
            params.append(&mut inner);
            (params, tail)
        }
        _ => (vec![], term),
    }
}

/// True if `ty` is a runtime-sized array (size is a type variable or
/// placeholder) — mirrors `binding_layout::is_runtime_sized_array`.
pub(crate) fn is_runtime_sized_array(ty: &Type<TypeName>) -> bool {
    crate::types::array_size(&crate::types::strip_unique(ty))
        .map(|s| {
            matches!(
                s,
                Type::Variable(_) | Type::Constructed(TypeName::SizePlaceholder, _)
            )
        })
        .unwrap_or(false)
}

/// Collect the free `Var(Symbol)` references of `term` as `(sym, ty)`.
/// `def_syms` is the program's top-level def table — its keys identify
/// names that are globally accessible (top-level functions / constants)
/// and therefore not "free" in the sense the predicate cares about
/// (they don't need to be supplied as pre-pass inputs; they're emitted
/// as separate defs and called directly).
pub(crate) fn free_symbol_vars(
    term: &Term,
    symbols: &SymbolTable,
    def_syms: &HashMap<String, SymbolId>,
) -> Vec<(SymbolId, Type<TypeName>)> {
    let bound: HashSet<SymbolId> = HashSet::new();
    let empty_top: HashSet<SymbolId> = HashSet::new();
    let known_defs: HashSet<String> = def_syms.keys().cloned().collect();
    let mut free: Vec<Term> = Vec::new();
    let mut seen: HashSet<SymbolId> = HashSet::new();
    collect_free_vars(
        term,
        &bound,
        &empty_top,
        &known_defs,
        symbols,
        &mut free,
        &mut seen,
    );
    free.iter()
        .filter_map(|t| match &t.kind {
            TermKind::Var(VarRef::Symbol(s)) => Some((*s, t.ty.clone())),
            _ => None,
        })
        .collect()
}

/// Return every `ArrayExpr::StorageView` (deduped by binding) reachable from
/// `rhs` — i.e. the chained intermediates the producer reads. The walker
/// [`rewrite_array_input`] can plant a `StorageView` at any depth (nested
/// through `Zip` or inside another `Soac`'s own input), so a one-level scan
/// would miss them and the resulting pre-pass would forget to declare those
/// bindings as Inputs. Recursive descent + dedup keeps the declaration set
/// complete and minimal.
pub(crate) fn producer_storage_inputs(rhs: &Term) -> Vec<(u32, u32, Type<TypeName>)> {
    let mut out = Vec::<(u32, u32, Type<TypeName>)>::new();
    collect_storage_in_term(rhs, &mut out);
    out
}

fn collect_storage_in_term(term: &Term, out: &mut Vec<(u32, u32, Type<TypeName>)>) {
    if let TermKind::Soac(soac) = &term.kind {
        collect_storage_in_soac(soac, out);
    }
    term.for_each_child(&mut |c| collect_storage_in_term(c, out));
}

fn collect_storage_in_soac(soac: &SoacOp, out: &mut Vec<(u32, u32, Type<TypeName>)>) {
    match soac {
        SoacOp::Map { inputs, .. } | SoacOp::Redomap { inputs, .. } => {
            for ae in inputs {
                collect_storage_in_ae(ae, out);
            }
        }
        SoacOp::Reduce { input, .. } | SoacOp::Scan { input, .. } | SoacOp::Filter { input, .. } => {
            collect_storage_in_ae(input, out);
        }
        _ => {}
    }
}

fn collect_storage_in_ae(ae: &ArrayExpr, out: &mut Vec<(u32, u32, Type<TypeName>)>) {
    match ae {
        ArrayExpr::StorageView(sv) => {
            let key = (sv.binding.set, sv.binding.binding);
            if !out.iter().any(|(s, b, _)| (*s, *b) == key) {
                out.push((sv.binding.set, sv.binding.binding, sv.elem_ty.clone()));
            }
            // The offset/len Terms can themselves contain SOACs reading
            // other storage buffers — keep descending.
            collect_storage_in_term(&sv.offset, out);
            collect_storage_in_term(&sv.len, out);
        }
        ArrayExpr::Ref(t) => collect_storage_in_term(t, out),
        ArrayExpr::Zip(children) => {
            for c in children {
                collect_storage_in_ae(c, out);
            }
        }
        ArrayExpr::Soac(boxed) => collect_storage_in_soac(boxed, out),
        ArrayExpr::Literal(terms) => {
            for t in terms {
                collect_storage_in_term(t, out);
            }
        }
        ArrayExpr::Range { start, len, step } => {
            collect_storage_in_term(start, out);
            collect_storage_in_term(len, out);
            if let Some(s) = step {
                collect_storage_in_term(s, out);
            }
        }
    }
}

fn uint_lit(val: u64, span: crate::ast::Span, term_ids: &mut TermIdSource) -> Term {
    Term {
        id: term_ids.next_id(),
        ty: Type::Constructed(TypeName::UInt(32), vec![]),
        span,
        kind: TermKind::IntLit(val.to_string()),
    }
}
#[cfg(test)]
#[path = "lift_gathers_tests.rs"]
mod lift_gathers_tests;
