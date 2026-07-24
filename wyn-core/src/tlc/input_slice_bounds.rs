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
//! Analysis is read-only. The phase transition then attaches each result
//! directly to its owning entry node; EGIR consumes that in-tree data when
//! it builds the descriptor entry for each input binding.

use crate::{LookupMap, LookupSet};

use polytype::Type;

use crate::ast::TypeName;
use crate::builtins::{catalog, BuiltinId};
use crate::pipeline_descriptor::BufferLen;
use crate::types::TypeExt;
use crate::SymbolId;

use super::{
    data, extract_lambda_params_ref, Def, DefMeta, EntryPoint, Payload, Program, Term, TermKind,
    TermVisitor, VarRef, WalkDecision,
};

/// Backend-ready TLC with input bounds embedded in entry definitions.
#[derive(Debug, Clone, Copy, Default)]
pub struct InputBounded;

impl super::Family for InputBounded {
    type DefinitionData = ();
    type EntryData = super::data::EntryInputBounds;
    type ClosureData = super::data::ExplicitClosurePayload;
    type SoacBodyData = super::data::ExplicitCapturesPayload;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct InputSliceBoundsInferred;

impl super::Stage for InputSliceBoundsInferred {
    type Family = InputBounded;
    type GlobalContext = super::context::BackendGlobal;
}

type SourceStage = super::stage::Reachable;
type EntryPatches = LookupMap<SymbolId, data::EntryInputBounds>;

/// Infer every entry patch, then consume the old phase into the phase whose
/// entry nodes own those bounds. Both families select the same closure and
/// SOAC-body variables, so every term body moves without traversal.
pub fn run(program: Program<SourceStage>) -> Program<InputSliceBoundsInferred> {
    let mut patches = analyze(&program);
    let rebuilt = program.rebuild::<InputSliceBoundsInferred>(std::convert::identity, |def, _term_ids| {
        attach_entry_bounds(def, &mut patches)
    });
    assert!(
        patches.is_empty(),
        "input-slice-bound patches targeted definitions absent from the rebuilt program: {:?}",
        patches.keys().collect::<Vec<_>>()
    );
    rebuilt
}

/// Build one patch for every entry, including entries for which no finite
/// minimum can be inferred. Definition `SymbolId`s are globally unique.
fn analyze(program: &Program<SourceStage>) -> EntryPatches {
    let mut out = EntryPatches::new();
    for def in &program.defs {
        if !matches!(def.meta, DefMeta::EntryPoint(_)) {
            continue;
        }
        let (inner_body, params) = extract_lambda_params_ref(&def.body);
        let inputs: Vec<(SymbolId, Type<TypeName>)> =
            params.iter().map(|(sym, ty)| (*sym, ty.clone())).collect();
        let previous = out.insert(
            def.name,
            data::EntryInputBounds {
                by_symbol: infer(inner_body, &inputs),
            },
        );
        assert!(
            previous.is_none(),
            "duplicate TLC definition SymbolId {:?}",
            def.name
        );
    }
    out
}

fn attach_entry_bounds(
    def: Def<super::defunctionalize::ClosureConverted>,
    patches: &mut EntryPatches,
) -> Def<InputBounded> {
    let Def {
        data: (),
        name,
        ty,
        body,
        meta,
        arity,
        param_diets,
        return_diet,
    } = def;
    let meta = match meta {
        DefMeta::Function => DefMeta::Function,
        DefMeta::LiftedLambda => DefMeta::LiftedLambda,
        DefMeta::EntryPoint(entry) => {
            let data = patches
                .remove(&name)
                .unwrap_or_else(|| panic!("missing input-slice-bound patch for entry {name:?}"));
            DefMeta::EntryPoint(EntryPoint {
                declaration: entry.declaration,
                data,
            })
        }
    };
    Def {
        data: (),
        name,
        ty,
        body,
        meta,
        arity,
        param_diets,
        return_diet,
    }
}

/// Per `(SymbolId, array_type)` input, infer a minimum-required buffer
/// length from how the entry body slices that input. See the module
/// docs for the exact contract.
pub fn infer<C: Payload, S: Payload>(
    body: &Term<C, S>,
    inputs: &[(SymbolId, Type<TypeName>)],
) -> LookupMap<SymbolId, BufferLen> {
    let slice_id = catalog().known().slice;

    let mut elem_bytes: LookupMap<SymbolId, u64> = LookupMap::new();
    for (sym, ty) in inputs {
        let Some(et) = ty.elem_type() else { continue };
        let Some(b) = crate::ssa::layout::type_byte_size(et) else {
            continue;
        };
        elem_bytes.insert(*sym, b as u64);
    }
    let mut walker = InputBoundWalker {
        slice_id,
        tracked: elem_bytes.keys().copied().collect(),
        clean_max_k: LookupMap::new(),
        dirty: LookupSet::new(),
    };
    body.walk(&mut walker);
    let InputBoundWalker {
        clean_max_k, dirty, ..
    } = walker;

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

struct InputBoundWalker {
    slice_id: BuiltinId,
    tracked: LookupSet<SymbolId>,
    clean_max_k: LookupMap<SymbolId, u64>,
    dirty: LookupSet<SymbolId>,
}

impl<C: Payload, S: Payload> TermVisitor<C, S> for InputBoundWalker {
    fn visit(&mut self, term: &Term<C, S>) -> WalkDecision {
        // 1. Atomic recognition: `slice(Var(sym), IntLit(0), IntLit(K))`
        // for a tracked `sym`. Do not recurse into args[0]: the bare Var is
        // exactly what this slice consumes as its array operand.
        if let TermKind::App { func, args } = &term.kind {
            if args.len() == 3 {
                let is_slice = matches!(
                    &func.kind,
                    TermKind::Var(VarRef::Builtin { id, .. }) if *id == self.slice_id
                );
                if is_slice {
                    if let TermKind::Var(VarRef::Symbol(sym)) = &args[0].kind {
                        if self.tracked.contains(sym) {
                            let start_ok = matches!(
                                &args[1].kind,
                                TermKind::IntLit(s) if s.parse::<i64>() == Ok(0)
                            );
                            let end_k = if let TermKind::IntLit(s) = &args[2].kind {
                                s.parse::<i64>()
                                    .ok()
                                    .and_then(|i| if i >= 0 { Some(i as u64) } else { None })
                            } else {
                                None
                            };
                            if start_ok {
                                if let Some(k) = end_k {
                                    let entry = self.clean_max_k.entry(*sym).or_insert(0);
                                    *entry = (*entry).max(k);
                                    args[1].walk(self);
                                    args[2].walk(self);
                                    return WalkDecision::Prune;
                                }
                            }
                        }
                    }
                }
            }
        }
        // 2. Bare `Var(Symbol(sym))` for a tracked sym → mark dirty.
        if let TermKind::Var(VarRef::Symbol(sym)) = &term.kind {
            if self.tracked.contains(sym) {
                self.dirty.insert(*sym);
            }
            return WalkDecision::Prune;
        }

        // 3. Scope-aware recursion through binders. TLC `Let` is
        // non-recursive: `rhs` sees the outer scope.
        match &term.kind {
            TermKind::Let { name, rhs, body, .. } => {
                rhs.walk(self);
                let shadowed = self.tracked.remove(name);
                body.walk(self);
                if shadowed {
                    self.tracked.insert(*name);
                }
                WalkDecision::Prune
            }
            TermKind::Lambda(lam) => {
                let shadowed: Vec<SymbolId> = lam
                    .params
                    .iter()
                    .map(|(param, _)| *param)
                    .filter(|param| self.tracked.remove(param))
                    .collect();
                lam.body.walk(self);
                for symbol in shadowed {
                    self.tracked.insert(symbol);
                }
                WalkDecision::Prune
            }
            _ => WalkDecision::Recurse,
        }
    }
}

#[cfg(test)]
#[path = "input_slice_bounds_tests.rs"]
mod input_slice_bounds_tests;
