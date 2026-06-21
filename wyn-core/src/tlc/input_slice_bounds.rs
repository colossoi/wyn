//! Minimum-required input buffer length inference from slice references.
//!
//! For each `(SymbolId, elem_type)` pair handed in — typically the
//! storage-bound input parameters of a compute / graphics entry — this
//! pass walks an entry body's `Term` and returns a
//! `BufferLen::Fixed { bytes: max_K * sizeof(elem) }` per symbol that
//! satisfies *all* of:
//!
//!   * Every `Var(Symbol(sym))` reference to the symbol in the body
//!     appears as the first arg of an
//!     `App(_w_intrinsic_slice, [Var(sym), IntLit(0), IntLit(K)])` call.
//!     The slice pattern is recognized atomically — the walker does NOT
//!     descend into `args[0]` as a bare reference, so the inner
//!     `Var(sym)` does not itself disqualify the symbol.
//!   * `K` parses as a non-negative `u64`.
//!
//! Symbols whose body has *any* reference outside that pattern (raw
//! `Var(sym)`, `length(sym)`, non-zero slice start, non-`IntLit` end,
//! …) are not present in the returned map. The host has to declare the
//! size another way.
//!
//! The walker is scope-aware: when it descends into a `Let` or `Lambda`
//! whose binder rebinds a tracked symbol, it drops that symbol from the
//! tracked set for the inner body. TLC currently allocates a fresh
//! `SymbolId` for `let x = ... in body` even when `x` shadows an outer
//! name, so this is a belt-and-braces guard against transformed code
//! that re-binds the same `SymbolId`. `Loop`-bound init vars are walked
//! via `for_each_child`; no current test exercises a loop that shadows
//! an input symbol.
//!
//! Read-only analysis — no rewrites — consumed by
//! `egir::from_tlc::convert_entry_point` when it builds the descriptor
//! entry for each input binding.

use std::collections::{HashMap, HashSet};

use polytype::Type;

use crate::ast::TypeName;
use crate::builtins::{catalog, BuiltinId};
use crate::pipeline_descriptor::BufferLen;
use crate::types::TypeExt;
use crate::SymbolId;

use super::{extract_lambda_params, DefMeta, Program, Term, TermKind, VarRef};

/// Per-entry-point per-symbol input slice bounds for an entire program.
/// Outer key: entry-point surface name. Inner key: the entry's input
/// parameter `SymbolId` for a storage-bound input. Inner value: inferred
/// minimum byte length (see `infer`).
pub type ProgramBounds = HashMap<String, HashMap<SymbolId, BufferLen>>;

/// Run `infer` on every entry point in `program`. Each entry's storage-
/// bound params are extracted from its outer lambda chain (paired with
/// each param's declared type) and fed to `infer` alongside the inner
/// body.
pub fn compute_for_program(program: &Program) -> ProgramBounds {
    let mut out: ProgramBounds = HashMap::new();
    for def in &program.defs {
        if !matches!(def.meta, DefMeta::EntryPoint(_)) {
            continue;
        }
        let entry_name = match program.symbols.get(def.name) {
            Some(s) => s.to_string(),
            None => continue,
        };
        let (params, inner_body) = extract_lambda_params(&def.body);
        let inputs: Vec<(SymbolId, Type<TypeName>)> =
            params.iter().map(|(sym, ty)| (*sym, crate::types::strip_unique(ty))).collect();
        let bounds = infer(&inner_body, &inputs);
        if !bounds.is_empty() {
            out.insert(entry_name, bounds);
        }
    }
    out
}

/// Per `(SymbolId, array_type)` input, infer a minimum-required buffer
/// length from how the entry body slices that input. See the module
/// docs for the exact contract.
pub fn infer(body: &Term, inputs: &[(SymbolId, Type<TypeName>)]) -> HashMap<SymbolId, BufferLen> {
    let slice_id = catalog().known().slice;

    let mut elem_bytes: HashMap<SymbolId, u64> = HashMap::new();
    for (sym, ty) in inputs {
        let Some(et) = ty.elem_type() else { continue };
        let Some(b) = crate::ssa::layout::type_byte_size(et) else {
            continue;
        };
        elem_bytes.insert(*sym, b as u64);
    }
    let tracked: HashSet<SymbolId> = elem_bytes.keys().copied().collect();

    let mut clean_max_k: HashMap<SymbolId, u64> = HashMap::new();
    let mut dirty: HashSet<SymbolId> = HashSet::new();
    walk(body, slice_id, &tracked, &mut clean_max_k, &mut dirty);

    clean_max_k
        .into_iter()
        .filter(|(sym, _)| !dirty.contains(sym))
        .filter_map(|(sym, k)| {
            elem_bytes.get(&sym).map(|&eb| {
                (
                    sym,
                    BufferLen::Fixed {
                        bytes: k.saturating_mul(eb),
                    },
                )
            })
        })
        .collect()
}

fn walk(
    term: &Term,
    slice_id: BuiltinId,
    tracked: &HashSet<SymbolId>,
    clean_max_k: &mut HashMap<SymbolId, u64>,
    dirty: &mut HashSet<SymbolId>,
) {
    // 1. Atomic recognition: `slice(Var(sym), IntLit(0), IntLit(K))`
    // for a tracked `sym`. Does NOT recurse into args[0] (the bare
    // Var is exactly what we're consuming as the slice's array
    // operand).
    if let TermKind::App { func, args } = &term.kind {
        if args.len() == 3 {
            let is_slice = matches!(
                &func.kind,
                TermKind::Var(VarRef::Builtin { id, .. }) if *id == slice_id
            );
            if is_slice {
                if let TermKind::Var(VarRef::Symbol(sym)) = &args[0].kind {
                    if tracked.contains(sym) {
                        let start_ok = matches!(
                            &args[1].kind,
                            TermKind::IntLit(s) if s.parse::<i64>() == Ok(0)
                        );
                        let end_k = if let TermKind::IntLit(s) = &args[2].kind {
                            s.parse::<i64>().ok().and_then(|i| if i >= 0 { Some(i as u64) } else { None })
                        } else {
                            None
                        };
                        if start_ok {
                            if let Some(k) = end_k {
                                let entry = clean_max_k.entry(*sym).or_insert(0);
                                *entry = (*entry).max(k);
                                walk(&args[1], slice_id, tracked, clean_max_k, dirty);
                                walk(&args[2], slice_id, tracked, clean_max_k, dirty);
                                return;
                            }
                        }
                    }
                }
            }
        }
    }

    // 2. Bare `Var(Symbol(sym))` for a tracked sym → mark dirty.
    if let TermKind::Var(VarRef::Symbol(sym)) = &term.kind {
        if tracked.contains(sym) {
            dirty.insert(*sym);
        }
        return;
    }

    // 3. Scope-aware recursion through binders. TLC `Let` is
    // non-recursive: `rhs` sees the outer scope.
    match &term.kind {
        TermKind::Let { name, rhs, body, .. } => {
            walk(rhs, slice_id, tracked, clean_max_k, dirty);
            if tracked.contains(name) {
                let mut sub = tracked.clone();
                sub.remove(name);
                walk(body, slice_id, &sub, clean_max_k, dirty);
            } else {
                walk(body, slice_id, tracked, clean_max_k, dirty);
            }
        }
        TermKind::Lambda(lam) => {
            let shadows: Vec<SymbolId> =
                lam.params.iter().map(|(p, _)| *p).filter(|p| tracked.contains(p)).collect();
            if shadows.is_empty() {
                walk(&lam.body, slice_id, tracked, clean_max_k, dirty);
            } else {
                let mut sub = tracked.clone();
                for s in shadows {
                    sub.remove(&s);
                }
                walk(&lam.body, slice_id, &sub, clean_max_k, dirty);
            }
        }
        _ => {
            term.for_each_child(&mut |child| walk(child, slice_id, tracked, clean_max_k, dirty));
        }
    }
}

#[cfg(test)]
#[path = "input_slice_bounds_tests.rs"]
mod input_slice_bounds_tests;
