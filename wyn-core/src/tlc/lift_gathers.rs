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

use crate::{LookupMap, LookupSet};

use super::closure_convert::collect_free_vars;
use super::parallelize::make_entry_def;
use super::{
    ArrayExpr, Def, DefMeta, Lambda, Program, SoacBody, SoacOp, StorageView, Term, TermIdSource, TermKind,
    VarRef,
};
use crate::ast::TypeName;
use crate::egir::from_tlc::AUTO_STORAGE_SET;
use crate::interface::{EntryParamBinding, EntryParamBindingKind, StorageBindingDecl, StorageRole};
use crate::BindingRef;
use crate::{SymbolId, SymbolTable};
use polytype::Type;

/// Lift every randomly-indexed computed array out of each compute entry into
/// a gather pre-pass + a `storage_index` read.
pub fn run(mut program: Program, binding_ids: &mut crate::IdSource<u32>) -> Program {
    let entry_indices: Vec<usize> = program
        .defs
        .iter()
        .enumerate()
        .filter_map(|(i, d)| match &d.meta {
            DefMeta::EntryPoint(decl) if decl.entry_type.is_compute() => Some(i),
            _ => None,
        })
        .collect();

    let mut new_defs: Vec<Def> = Vec::new();
    for idx in entry_indices {
        lift_entry(&mut program, idx, &mut new_defs, binding_ids);
    }
    program.defs.extend(new_defs);
    super::anf::debug_check(&program, "lift_gathers");
    program
}

/// Lift gather sites out of a single compute entry at `program.defs[idx]`.
fn lift_entry(
    program: &mut Program,
    idx: usize,
    new_defs: &mut Vec<Def>,
    binding_ids: &mut crate::IdSource<u32>,
) {
    let entry_name = crate::symbol_name_or_bug(&program.symbols, program.defs[idx].name).to_string();
    let body = program.defs[idx].body.clone();

    let decl = match &program.defs[idx].meta {
        DefMeta::EntryPoint(d) => (**d).clone(),
        _ => return,
    };
    let outer_slots: &[Option<EntryParamBinding>] = &decl.param_bindings;

    let mut added_decls: Vec<StorageBindingDecl> = Vec::new();
    let new_body = lift_in_term(
        body,
        &entry_name,
        outer_slots,
        binding_ids,
        &mut added_decls,
        new_defs,
        program,
        &[],
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
    outer_slots: &[Option<EntryParamBinding>],
    binding_ids: &mut crate::IdSource<u32>,
    added_decls: &mut Vec<StorageBindingDecl>,
    new_defs: &mut Vec<Def>,
    program: &mut Program,
    local_lets: &[(SymbolId, Type<TypeName>, Term)],
) -> Term {
    match term.kind {
        TermKind::Lambda(lam) => {
            let Lambda { params, body, ret_ty } = lam;
            let new_body = lift_in_term(
                *body,
                entry_name,
                outer_slots,
                binding_ids,
                added_decls,
                new_defs,
                program,
                local_lets,
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
            // Peek the next binding id and offer it to `try_lift` as the
            // gather-buffer slot. Commit (advance the factory) only if the
            // lift succeeded — failed eligibility checks leave the id free
            // for the next attempt.
            let candidate_binding = binding_ids.peek_id();
            if let Some((prepass, decl, rewritten_body)) = try_lift(
                name,
                &name_ty,
                &rhs,
                *body.clone(),
                entry_name,
                outer_slots,
                candidate_binding,
                new_defs.len(),
                program,
                local_lets,
            ) {
                let _ = binding_ids.next_id();
                new_defs.push(prepass);
                added_decls.push(decl);
                // The dropped `let` is gone; keep lifting in the rewritten body.
                return lift_in_term(
                    rewritten_body,
                    entry_name,
                    outer_slots,
                    binding_ids,
                    added_decls,
                    new_defs,
                    program,
                    local_lets,
                );
            }
            let rhs_for_scope = (*rhs).clone();
            let mut body_local_lets = local_lets.to_vec();
            body_local_lets.push((name, name_ty.clone(), rhs_for_scope));
            let new_body = lift_in_term(
                *body,
                entry_name,
                outer_slots,
                binding_ids,
                added_decls,
                new_defs,
                program,
                &body_local_lets,
            );
            // After normalize_outputs, the rhs of a sequencing let is an
            // OutputSlotStore that may wrap further gather candidates
            // (`let _ = OutputSlotStore(i, let counts = map(...) in …)`).
            // Descend so the inner producer still gets lifted.
            let new_rhs = if matches!(rhs.kind, TermKind::OutputSlotStore { .. }) {
                Box::new(lift_in_term(
                    *rhs,
                    entry_name,
                    outer_slots,
                    binding_ids,
                    added_decls,
                    new_defs,
                    program,
                    local_lets,
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
                outer_slots,
                binding_ids,
                added_decls,
                new_defs,
                program,
                local_lets,
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

/// Attempt to lift `let name = rhs in body`. Returns the gather pre-pass def,
/// the consumer's new Input binding decl, and the rewritten body (with each
/// `name[idx]` replaced by a `storage_index` load and the `let` dropped), or
/// `None` if this isn't a liftable gather site.
fn try_lift(
    name: SymbolId,
    name_ty: &Type<TypeName>,
    rhs: &Term,
    body: Term,
    entry_name: &str,
    outer_slots: &[Option<EntryParamBinding>],
    binding_num: u32,
    gather_idx: usize,
    program: &mut Program,
    local_lets: &[(SymbolId, Type<TypeName>, Term)],
) -> Option<(Def, StorageBindingDecl, Term)> {
    // Producer must be an array-yielding SOAC (`map` or `scan`) of a
    // runtime-sized array. Both preserve element count, so the gather buffer
    // tracks the producer's input length.
    let is_array_producer = is_liftable_array_producer(rhs);
    if !is_array_producer || !is_runtime_sized_array(name_ty) {
        return None;
    }
    let elem_ty = crate::types::array_elem(name_ty).cloned()?;

    // The pre-pass re-declares the producer's free variables as its own input
    // params, so every one must be an entry-param input array (Phase 1). A
    // free var that's a let-bound computed array or a uniform/size would need
    // its definition pulled in (or a uniform re-declared) — deferred. This
    // keeps the pre-pass self-contained over real buffers. (Fusion has already
    // folded any `map` producer into the scan/map, so a `scan(op, ne, map(g,
    // xs))` arrives reading `xs` directly.)
    let local_deps =
        local_gather_dependencies(rhs, local_lets, outer_slots, &program.symbols, &program.def_syms)?;
    let producer = wrap_local_gather_dependencies(rhs.clone(), &local_deps);
    let frees = free_symbol_vars(&producer, &program.symbols, &program.def_syms);
    if !frees
        .iter()
        .all(|(s, ty)| is_runtime_sized_array(ty) && find_outer_single(outer_slots, *s).is_some())
    {
        return None;
    }

    // Rewrite `name[idx]` → storage_index and `ArrayExpr::Ref(Var(name))`
    // → `ArrayExpr::StorageBuffer{…}` on a trial copy; bail if `name` is used
    // any other way (a bare Var in a non-SOAC-input position), or if no use is
    // a dynamic index nor a SOAC input. Multi-consumer is fine: every
    // downstream Index *and* every downstream SOAC input gets routed to the
    // same gather buffer.
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
    if bail || (dyn_uses + soac_uses == 0) {
        return None;
    }

    // The gather buffer holds `map(f, src)`'s output: one element per `src`
    // element, so its length tracks `src`'s element count (element sizes may
    // differ). `from_tlc` allocates the host buffer from this policy.
    let length = gather_length(&elem_ty, &frees, outer_slots);

    // Chained intermediates: if the producer reads any `StorageBuffer{set,
    // binding, …}` directly (e.g. its input was itself an already-lifted
    // gather buffer), the pre-pass must declare each as its own Input so the
    // descriptor wires up the cross-stage read.
    let chained = producer_storage_inputs(&producer);

    let prepass = build_gather_prepass(
        entry_name,
        gather_idx,
        producer,
        name_ty.clone(),
        &frees,
        outer_slots,
        binding,
        length.clone(),
        &chained,
        program,
        &mut term_ids,
    );
    // The consumer *reads* the gather buffer: an Input-role decl carrying the
    // sizing policy. `from_tlc` emits any length-bearing storage binding as a
    // compiler-managed Intermediate in the descriptor (read-only here).
    let decl = StorageBindingDecl {
        binding: crate::BindingRef::new(binding.0, binding.1),
        role: StorageRole::Input,
        elem_ty,
        length,
    };
    Some((prepass, decl, rewritten))
}

/// Sizing policy for the gather buffer: `LikeInput` of the producer's first
/// input array (a `map` preserves element count). `None` if that input isn't
/// a known param binding or element sizes can't be computed — the runtime
/// then falls back to its default intermediate size.
fn gather_length(
    elem_ty: &Type<TypeName>,
    producer_inputs: &[(SymbolId, Type<TypeName>)],
    outer_slots: &[Option<EntryParamBinding>],
) -> Option<crate::pipeline_descriptor::BufferLen> {
    let elem_bytes = crate::ssa::layout::type_byte_size(elem_ty)?;
    let (src_sym, src_ty) = producer_inputs.first()?;
    let br = find_outer_single(outer_slots, *src_sym)?;
    let src_elem_ty = crate::types::array_elem(src_ty)?;
    let src_elem_bytes = crate::ssa::layout::type_byte_size(src_elem_ty)?;
    Some(crate::pipeline_descriptor::BufferLen::LikeInput {
        set: br.set,
        binding: br.binding,
        elem_bytes,
        src_elem_bytes,
    })
}

/// Scan the outer entry's cached param-binding slots for a Single-kind binding
/// matching `sym`. Tuple-of-views bindings are skipped — gather captures are
/// always bare `Var(sym)` references, never tuple projections.
fn local_gather_dependencies(
    term: &Term,
    local_lets: &[(SymbolId, Type<TypeName>, Term)],
    outer_slots: &[Option<EntryParamBinding>],
    symbols: &SymbolTable,
    def_syms: &LookupMap<String, SymbolId>,
) -> Option<Vec<(SymbolId, Type<TypeName>, Term)>> {
    let mut needed: LookupSet<SymbolId> = free_symbol_vars(term, symbols, def_syms)
        .into_iter()
        .filter_map(|(sym, ty)| {
            let outer_input = is_runtime_sized_array(&ty) && find_outer_single(outer_slots, sym).is_some();
            (!outer_input).then_some(sym)
        })
        .collect();
    if needed.is_empty() {
        return Some(Vec::new());
    }

    let mut deps = Vec::new();
    for (sym, ty, rhs) in local_lets.iter().rev() {
        if !needed.contains(sym) {
            continue;
        }
        if !is_local_gather_dependency(rhs) {
            return None;
        }
        needed.remove(sym);
        for (dep_sym, dep_ty) in free_symbol_vars(rhs, symbols, def_syms) {
            let outer_input =
                is_runtime_sized_array(&dep_ty) && find_outer_single(outer_slots, dep_sym).is_some();
            if !outer_input {
                needed.insert(dep_sym);
            }
        }
        deps.push((*sym, ty.clone(), rhs.clone()));
    }

    if !needed.is_empty() {
        return None;
    }
    deps.reverse();
    Some(deps)
}

/// A local `let` is pullable into the pre-pass if it is pure and reproducible:
/// any term free of a SOAC or output-store. Scalars (e.g. the length argument
/// of an `iota`) qualify alongside arrays — the producer's own free-variable
/// resolution downstream rejects anything that can't be grounded in the
/// pre-pass.
fn is_local_gather_dependency(term: &Term) -> bool {
    !contains_local_gather_dependency_blocker(term)
}

fn contains_local_gather_dependency_blocker(term: &Term) -> bool {
    if matches!(term.kind, TermKind::Soac(_) | TermKind::OutputSlotStore { .. }) {
        return true;
    }
    let mut blocked = false;
    term.for_each_child(&mut |child| {
        if !blocked {
            blocked = contains_local_gather_dependency_blocker(child);
        }
    });
    blocked
}

fn wrap_local_gather_dependencies(mut producer: Term, deps: &[(SymbolId, Type<TypeName>, Term)]) -> Term {
    for (name, name_ty, rhs) in deps.iter().rev() {
        producer = Term {
            id: producer.id,
            ty: producer.ty.clone(),
            span: rhs.span,
            kind: TermKind::Let {
                name: *name,
                name_ty: name_ty.clone(),
                rhs: Box::new(rhs.clone()),
                body: Box::new(producer),
            },
        };
    }
    producer
}

fn find_outer_single(
    outer_slots: &[Option<EntryParamBinding>],
    sym: SymbolId,
) -> Option<crate::BindingRef> {
    outer_slots.iter().flatten().find_map(|b| match &b.kind {
        EntryParamBindingKind::Single { binding, .. } if b.param_sym == sym => Some(*binding),
        _ => None,
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
///     position ceases to be "a SOAC reading this array."
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
        SoacOp::Screma {
            lanes,
            accumulators,
            inputs,
        } => SoacOp::Screma {
            lanes: lanes
                .into_iter()
                .map(|lane| super::ScremaLane {
                    lam: rewrite_soac_body(
                        lane.lam, arr, binding, elem_ty, bail, dyn_uses, soac_uses, term_ids,
                    ),
                    input_indices: lane.input_indices,
                })
                .collect(),
            accumulators: accumulators
                .into_iter()
                .map(|acc| super::ScremaAccumulatorSpec {
                    kind: acc.kind,
                    step_lam: rewrite_soac_body(
                        acc.step_lam,
                        arr,
                        binding,
                        elem_ty,
                        bail,
                        dyn_uses,
                        soac_uses,
                        term_ids,
                    ),
                    reduce_op: rewrite_soac_body(
                        acc.reduce_op,
                        arr,
                        binding,
                        elem_ty,
                        bail,
                        dyn_uses,
                        soac_uses,
                        term_ids,
                    ),
                    ne: Box::new(rewrite_uses(
                        *acc.ne, arr, binding, elem_ty, bail, dyn_uses, soac_uses, term_ids,
                    )),
                })
                .collect(),
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
            map_lam,
            pred,
            input,
            destination,
        } => SoacOp::Filter {
            map_lam: map_lam.map(|ml| {
                rewrite_soac_body(ml, arr, binding, elem_ty, bail, dyn_uses, soac_uses, term_ids)
            }),
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
        ArrayExpr::Var(vr, ty) => {
            if matches!(vr, VarRef::Symbol(s) if s == arr) {
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
                // A different named input — a leaf, nothing to rewrite.
                ArrayExpr::Var(vr, ty)
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
    outer_slots: &[Option<EntryParamBinding>],
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
    // Each captured input was a view-array param of the outer entry, so it
    // already has a binding pinned by `pin_entry_regions`. Re-use that binding
    // in the synthesized pre-pass so both entries reference the same SPIR-V
    // OpVariable at the same `(set, binding)`. Captures that don't match an
    // outer slot (scalars, etc.) keep `binding: None` and get push-constant
    // routed by `egir/from_tlc`.
    let required_params: Vec<super::parallelize::RequiredParam> = captured_inputs
        .iter()
        .map(|(s, ty)| super::parallelize::RequiredParam {
            sym: *s,
            ty: ty.clone(),
            attr: None,
            binding: outer_slots.iter().flatten().find(|b| b.param_sym == *s).cloned(),
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

/// True if `ty` is a runtime-sized array (size is a type variable or
/// placeholder) — mirrors `binding_layout::is_runtime_sized_array`.
fn is_runtime_sized_array(ty: &Type<TypeName>) -> bool {
    crate::types::array_size(&crate::types::strip_unique(ty))
        .map(|s| {
            matches!(
                s,
                Type::Variable(_) | Type::Constructed(TypeName::SizePlaceholder, _)
            )
        })
        .unwrap_or(false)
}

fn is_liftable_array_producer(term: &Term) -> bool {
    match &term.kind {
        TermKind::Let { body, .. } => is_liftable_array_producer(body),
        TermKind::Soac(SoacOp::Map { .. } | SoacOp::Scan { .. }) => true,
        // A single-output *array* Screma — one map lane, or one `Scan`
        // accumulator (a fused map→scan arrives this way) — is a liftable
        // producer just like a bare Map/Scan. A single `Reduce` is scalar.
        TermKind::Soac(SoacOp::Screma {
            lanes, accumulators, ..
        }) => {
            (lanes.len() == 1 && accumulators.is_empty())
                || (lanes.is_empty()
                    && accumulators.len() == 1
                    && matches!(accumulators[0].kind, super::ScremaAccumulator::Scan))
        }
        _ => false,
    }
}

/// Collect the free `Var(Symbol)` references of `term` as `(sym, ty)`.
/// `def_syms` is the program's top-level def table — its keys identify
/// names that are globally accessible (top-level functions / constants)
/// and therefore not "free" in the sense the predicate cares about
/// (they don't need to be supplied as pre-pass inputs; they're emitted
/// as separate defs and called directly).
fn free_symbol_vars(
    term: &Term,
    symbols: &SymbolTable,
    def_syms: &LookupMap<String, SymbolId>,
) -> Vec<(SymbolId, Type<TypeName>)> {
    let bound: LookupSet<SymbolId> = LookupSet::new();
    let empty_top: LookupSet<SymbolId> = LookupSet::new();
    let known_defs: LookupSet<String> = def_syms.keys().cloned().collect();
    let mut free: Vec<Term> = Vec::new();
    let mut seen: LookupSet<SymbolId> = LookupSet::new();
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
fn producer_storage_inputs(rhs: &Term) -> Vec<(u32, u32, Type<TypeName>)> {
    let mut out = Vec::<(u32, u32, Type<TypeName>)>::new();
    collect_storage_in_term(rhs, &mut out);
    out
}

fn collect_storage_in_term(term: &Term, out: &mut Vec<(u32, u32, Type<TypeName>)>) {
    if let Some((set, binding, elem_ty)) = storage_index_input(term) {
        push_storage_input(out, set, binding, elem_ty);
    }
    if let TermKind::Soac(soac) = &term.kind {
        collect_storage_in_soac(soac, out);
    }
    term.for_each_child(&mut |c| collect_storage_in_term(c, out));
}

fn collect_storage_in_soac(soac: &SoacOp, out: &mut Vec<(u32, u32, Type<TypeName>)>) {
    match soac {
        SoacOp::Map { inputs, .. } | SoacOp::Screma { inputs, .. } => {
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
            push_storage_input(out, sv.binding.set, sv.binding.binding, sv.elem_ty.clone());
            // The offset/len Terms can themselves contain SOACs reading
            // other storage buffers — keep descending.
            collect_storage_in_term(&sv.offset, out);
            collect_storage_in_term(&sv.len, out);
        }
        // A named input is a leaf — no storage view to collect.
        ArrayExpr::Var(_, _) => {}
        ArrayExpr::Zip(children) => {
            for c in children {
                collect_storage_in_ae(c, out);
            }
        }
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

fn storage_index_input(term: &Term) -> Option<(u32, u32, Type<TypeName>)> {
    let TermKind::App { func, args } = &term.kind else {
        return None;
    };
    let TermKind::Var(VarRef::Builtin { id, .. }) = &func.kind else {
        return None;
    };
    if *id != crate::builtins::catalog().known().storage_index || args.len() != 3 {
        return None;
    }
    Some((u32_int_lit(&args[0])?, u32_int_lit(&args[1])?, term.ty.clone()))
}

fn u32_int_lit(term: &Term) -> Option<u32> {
    match &term.kind {
        TermKind::IntLit(s) => s.parse::<u32>().ok(),
        _ => None,
    }
}

fn push_storage_input(
    out: &mut Vec<(u32, u32, Type<TypeName>)>,
    set: u32,
    binding: u32,
    elem_ty: Type<TypeName>,
) {
    if !out.iter().any(|(s, b, _)| *s == set && *b == binding) {
        out.push((set, binding, elem_ty));
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
