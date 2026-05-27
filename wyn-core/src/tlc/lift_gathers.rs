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

use std::collections::HashSet;

use super::closure_convert::collect_free_vars;
use super::parallelize::{intrinsic_term_by_id, make_entry_def};
use super::{Def, DefMeta, Lambda, Program, SoacOp, Term, TermKind, VarRef};
use crate::ast::TypeName;
use crate::builtins::catalog;
use crate::egir::from_tlc::AUTO_STORAGE_SET;
use crate::interface::{StorageBindingDecl, StorageRole};
use crate::{SymbolId, SymbolTable};
use polytype::Type;

/// Lift every randomly-indexed computed array out of each compute entry into
/// a gather pre-pass + a `storage_index` read.
pub fn run(mut program: Program) -> Program {
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
        lift_entry(&mut program, idx, &mut new_defs);
    }
    program.defs.extend(new_defs);
    program
}

/// Lift gather sites out of a single compute entry at `program.defs[idx]`.
fn lift_entry(program: &mut Program, idx: usize, new_defs: &mut Vec<Def>) {
    let entry_name = program.symbols.get(program.defs[idx].name).cloned().unwrap_or_default();
    let body = program.defs[idx].body.clone();

    // Gather buffers must sit above the consumer's own auto-allocated
    // bindings: input-view params occupy `0..view_count`, and `from_tlc`
    // places this entry's storage outputs at `view_count..view_count +
    // out_count` (see `build_entry_outputs`). So the first free binding for a
    // gather intermediate is `view_count + out_count`.
    let (params, tail) = peel_lambda_params(&body);
    let decl = match &program.defs[idx].meta {
        DefMeta::EntryPoint(d) => (**d).clone(),
        _ => return,
    };
    let view_count =
        crate::binding_layout::compute_entry_binding_layout(&params, &decl, AUTO_STORAGE_SET).len() as u32;
    let out_count = storage_output_count(&tail.ty);
    let mut next_gather = view_count + out_count;

    let mut added_decls: Vec<StorageBindingDecl> = Vec::new();
    let new_body = lift_in_term(
        body,
        &entry_name,
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
    next_gather: &mut u32,
    added_decls: &mut Vec<StorageBindingDecl>,
    new_defs: &mut Vec<Def>,
    program: &mut Program,
) -> Term {
    match term.kind {
        TermKind::Lambda(lam) => {
            let Lambda { params, body, ret_ty } = lam;
            let new_body = lift_in_term(*body, entry_name, next_gather, added_decls, new_defs, program);
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
                    next_gather,
                    added_decls,
                    new_defs,
                    program,
                );
            }
            let new_body = lift_in_term(*body, entry_name, next_gather, added_decls, new_defs, program);
            Term {
                kind: TermKind::Let {
                    name,
                    name_ty,
                    rhs,
                    body: Box::new(new_body),
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
    binding_num: u32,
    gather_idx: usize,
    program: &mut Program,
) -> Option<(Def, StorageBindingDecl, Term)> {
    // Producer must be a `map` yielding a runtime-sized array.
    if !matches!(&rhs.kind, TermKind::Soac(SoacOp::Map { .. })) || !is_runtime_sized_array(name_ty) {
        return None;
    }
    let elem_ty = crate::types::array_elem(name_ty).cloned()?;

    // Producer captures only input arrays (Phase 1). Any other free var
    // (uniform/size) would need re-declaration on the pre-pass — deferred.
    let frees = free_symbol_vars(rhs, &program.symbols);
    if !frees.iter().all(|(_, ty)| is_runtime_sized_array(ty)) {
        return None;
    }

    // Rewrite `name[idx]` → storage_index on a trial copy; bail if `name` is
    // used any other way, or if no use is a dynamic (non-constant) index.
    let binding = (AUTO_STORAGE_SET, binding_num);
    let mut bail = false;
    let mut dyn_uses = 0usize;
    let rewritten = rewrite_uses(body, name, binding, &elem_ty, &mut bail, &mut dyn_uses);
    if bail || dyn_uses == 0 {
        return None;
    }

    let prepass = build_gather_prepass(
        entry_name,
        gather_idx,
        rhs.clone(),
        name_ty.clone(),
        &frees,
        binding,
        program,
    );
    let decl = StorageBindingDecl {
        set: binding.0,
        binding: binding.1,
        role: StorageRole::Input,
        elem_ty,
    };
    Some((prepass, decl, rewritten))
}

/// Recursively replace `Index { array: Var(arr), index }` with
/// `_w_intrinsic_storage_index(set, binding, index)`. Sets `bail` if `arr`
/// appears in any other position (not a pure-gather use). Counts uses whose
/// index is not a constant literal in `dyn_uses`.
fn rewrite_uses(
    term: Term,
    arr: SymbolId,
    binding: (u32, u32),
    elem_ty: &Type<TypeName>,
    bail: &mut bool,
    dyn_uses: &mut usize,
) -> Term {
    if let TermKind::Index { array, index } = &term.kind {
        if matches!(&array.kind, TermKind::Var(VarRef::Symbol(s)) if *s == arr) {
            let idx = rewrite_uses((**index).clone(), arr, binding, elem_ty, bail, dyn_uses);
            if !matches!(idx.kind, TermKind::IntLit(_)) {
                *dyn_uses += 1;
            }
            return intrinsic_term_by_id(
                catalog().known().storage_index,
                vec![
                    uint_lit(binding.0 as u64, term.span),
                    uint_lit(binding.1 as u64, term.span),
                    idx,
                ],
                elem_ty.clone(),
                term.span,
            );
        }
    }
    if matches!(&term.kind, TermKind::Var(VarRef::Symbol(s)) if *s == arr) {
        *bail = true;
        return term;
    }
    term.map_children(&mut |c| rewrite_uses(c, arr, binding, elem_ty, bail, dyn_uses))
}

/// Build the `<entry>_gather_<n>` compute entry whose body is the producer
/// `map`, re-declaring each captured input array as its own view param and
/// pinning the result to `binding` via an Output storage decl (which
/// `parallelize::make_map_plan` reads to force the map's output buffer).
fn build_gather_prepass(
    entry_name: &str,
    gather_idx: usize,
    producer: Term,
    result_ty: Type<TypeName>,
    captured_inputs: &[(SymbolId, Type<TypeName>)],
    binding: (u32, u32),
    program: &mut Program,
) -> Def {
    let name = format!("{}_gather_{}", entry_name, gather_idx);
    let elem_ty = crate::types::array_elem(&result_ty).cloned().unwrap_or_else(|| result_ty.clone());
    let uniform_attrs = vec![None; captured_inputs.len()];
    let storage_bindings = vec![StorageBindingDecl {
        set: binding.0,
        binding: binding.1,
        role: StorageRole::Output,
        elem_ty,
    }];
    make_entry_def(
        &name,
        producer,
        result_ty,
        captured_inputs,
        &uniform_attrs,
        storage_bindings,
        program,
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

/// Number of storage-output bindings `from_tlc` allocates for a compute
/// entry returning `ret_ty`: one per tuple field, one for a plain non-unit
/// return, none for unit.
fn storage_output_count(ret_ty: &Type<TypeName>) -> u32 {
    match ret_ty {
        Type::Constructed(TypeName::Unit, _) => 0,
        Type::Constructed(TypeName::Tuple(_), fields) => fields.len() as u32,
        _ => 1,
    }
}

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

/// Collect the free `Var(Symbol)` references of `term` as `(sym, ty)`.
fn free_symbol_vars(term: &Term, symbols: &SymbolTable) -> Vec<(SymbolId, Type<TypeName>)> {
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
    free.iter()
        .filter_map(|t| match &t.kind {
            TermKind::Var(VarRef::Symbol(s)) => Some((*s, t.ty.clone())),
            _ => None,
        })
        .collect()
}

fn uint_lit(val: u64, span: crate::ast::Span) -> Term {
    Term {
        id: super::TermId(0),
        ty: Type::Constructed(TypeName::UInt(32), vec![]),
        span,
        kind: TermKind::IntLit(val.to_string()),
    }
}

#[cfg(test)]
#[path = "lift_gathers_tests.rs"]
mod lift_gathers_tests;
