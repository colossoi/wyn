//! Late ownership-driven TLC rewrites.
//!
//! This pass analyzes the exact post-closure tree, produces `TermId`-keyed
//! patches, and consumes the tree once to apply them.

use super::analysis::{alias_target_of, analyze, AnalysisState, Origin};
use crate::ast::TypeName;
use crate::builtins::catalog;
use crate::tlc::stage::GeneratedLambdasFolded;
use crate::tlc::{
    var_term_builtin_id, ArrayExpr, Def, Lambda, Payload, Program, RewriteDecision, SoacOp, Term, TermId,
    TermIdSource, TermKind, TermRewriter, VarRef, WalkDecision,
};
use crate::types::TypeExt;
use crate::{LookupMap, SymbolId, SymbolTable};
use polytype::Type;

#[derive(Debug, Clone, Copy, Default)]
pub struct OwnershipApplied;

impl crate::tlc::Stage for OwnershipApplied {
    type Family = crate::tlc::family::ClosureConverted;
    type GlobalContext = crate::tlc::context::PostClosureGlobal;
}

// =============================================================================
// TLC-level promotion of array_with → array_with_inplace
// =============================================================================

#[derive(Debug, Clone, Copy)]
pub(super) enum OwnershipPatch {
    PromoteArrayWith,
    MarkUniqueInput,
}

type OwnershipPatches = LookupMap<TermId, OwnershipPatch>;

fn insert_ownership_patch(patches: &mut OwnershipPatches, target: TermId, patch: OwnershipPatch) {
    let previous = patches.insert(target, patch);
    assert!(
        previous.is_none(),
        "multiple ownership patches targeted TLC term {target:?}: {previous:?} and {patch:?}"
    );
}

/// Analyze the exact tree that will be rebuilt and return every requested
/// ownership mutation as a `TermId`-keyed patch map.
fn analyze_application<St: crate::tlc::Stage>(program: &Program<St>) -> OwnershipPatches {
    let model = analyze(program);
    let mut patches = OwnershipPatches::new();
    {
        let mut collect_patch = |term: &Term<
            <<St as crate::tlc::Stage>::Family as crate::tlc::Family>::ClosureData,
            <<St as crate::tlc::Stage>::Family as crate::tlc::Family>::SoacBodyData,
        >| {
            if is_eligible_unique_input_soac(term, &model) {
                insert_ownership_patch(&mut patches, term.id, OwnershipPatch::MarkUniqueInput);
            }
            if let TermKind::App { func, args } = &term.kind {
                let known = catalog().known();
                let calls_functional =
                    var_term_builtin_id(func, &program.symbols) == Some(known.array_with);
                if calls_functional
                    && args.len() == 3
                    && array_with_is_promotable(term.id, &args[0], &model, &program.symbols)
                {
                    insert_ownership_patch(&mut patches, term.id, OwnershipPatch::PromoteArrayWith);
                }
            }
            WalkDecision::Recurse
        };
        for def in &program.defs {
            def.body.walk(&mut collect_patch);
        }
    }
    patches
}

/// Apply ownership analysis as a consuming phase transition.
///
/// Analysis produces every mutation before reconstruction begins. The generic
/// term rewriter consumes those patches, preserves untouched subtree IDs, and
/// allocates fresh IDs only along changed paths.
pub fn apply_ownership(program: Program<GeneratedLambdasFolded>) -> Program<OwnershipApplied> {
    apply_ownership_rewrite(program).into_stage()
}

pub(super) fn apply_ownership_rewrite<S: crate::tlc::Stage>(program: Program<S>) -> Program<S> {
    let mut patches = analyze_application(&program);
    let rebuilt = program.rebuild::<S>(std::convert::identity, |def, term_ids| {
        let mut rewriter = OwnershipRewriter {
            patches: &mut patches,
            term_ids,
        };
        Def {
            body: def.body.rewrite(&mut rewriter),
            ..def
        }
    });
    assert!(
        patches.is_empty(),
        "ownership patches targeted terms absent from the rebuilt program: {:?}",
        patches.keys().collect::<Vec<_>>()
    );
    rebuilt
}

pub(super) struct OwnershipRewriter<'a> {
    pub(super) patches: &'a mut OwnershipPatches,
    pub(super) term_ids: &'a mut TermIdSource,
}

impl<C: Payload, S: Payload> TermRewriter<C, S> for OwnershipRewriter<'_> {
    fn next_term_id(&mut self) -> TermId {
        self.term_ids.next_id()
    }

    fn rewrite_node(&mut self, term: &mut Term<C, S>) -> RewriteDecision {
        let Some(patch) = self.patches.remove(&term.id) else {
            return RewriteDecision::Unchanged;
        };

        match patch {
            OwnershipPatch::PromoteArrayWith => {
                let TermKind::App { func, .. } = &mut term.kind else {
                    panic!("array-with ownership patch targeted non-App term {:?}", term.id);
                };
                func.id = self.term_ids.next_id();
                func.kind = TermKind::Var(VarRef::Builtin {
                    id: catalog().known().array_with_in_place,
                    overload_idx: 0,
                });
            }
            OwnershipPatch::MarkUniqueInput => match &mut term.kind {
                TermKind::Soac(SoacOp::Map { destination, .. })
                | TermKind::Soac(SoacOp::Scan { destination, .. })
                | TermKind::Soac(SoacOp::Filter { destination, .. }) => {
                    *destination = crate::types::SoacOwnership::UniqueInput;
                }
                _ => panic!(
                    "unique-input ownership patch targeted ineligible term {:?}",
                    term.id
                ),
            },
        }
        RewriteDecision::Changed
    }
}

fn array_with_is_promotable<C: Payload, S: Payload>(
    call_id: TermId,
    source_arg: &Term<C, S>,
    model: &AnalysisState,
    symbols: &SymbolTable,
) -> bool {
    let Some(owner) = alias_target_of(source_arg, model, symbols) else {
        return false;
    };
    let Some(origin) = model.origin(owner) else {
        return false;
    };
    if !origin.is_mutable() {
        return false;
    }
    // No live_out recorded for this call ⇒ analyze didn't reach it
    // (e.g. a dead-code def). Be conservative and do not promote.
    let Some(live) = model.live_out.get(&call_id) else {
        return false;
    };
    !live.contains(&owner)
}

// =============================================================================
// SOAC unique-input eligibility
// =============================================================================

/// Return SOAC term ids whose primary input has a unique owner and whose
/// pointwise body would permit input-side reuse.
///
/// This records an ownership fact, not a physical reuse decision. EGIR may
/// fuse or reroute the operation before residency planning, and only then
/// resolves the candidate using the final liveness graph.
///
/// A Map qualifies only when *all* of:
///
/// 1. The input is a single buffered `ArrayExpr::Var` whose owner has a
///    unique, mutable source-level origin.
/// 2. The lambda body's return type matches the lambda's element-param
///    type (pointwise: same shape in, same shape out).
/// 3. The body does not read the input owner outside of the element
///    parameter — no captured stencil reads. `map(|x| x + a[i-1], a)`
///    is rejected because in-place mutation at index `i` would change
///    later iterations' reads.
/// Pure analysis. Does not mutate the program. The caller decides
/// whether to act on the result.
pub(super) fn eligible_unique_input_soacs<St: crate::tlc::Stage>(
    program: &Program<St>,
    model: &AnalysisState,
) -> Vec<TermId> {
    let mut eligible = Vec::new();
    {
        let mut collect = |term: &Term<
            <<St as crate::tlc::Stage>::Family as crate::tlc::Family>::ClosureData,
            <<St as crate::tlc::Stage>::Family as crate::tlc::Family>::SoacBodyData,
        >| {
            if is_eligible_unique_input_soac(term, model) {
                eligible.push(term.id);
            }
            WalkDecision::Recurse
        };
        for def in &program.defs {
            def.body.walk(&mut collect);
        }
    }
    eligible
}

fn is_eligible_unique_input_soac<C: Payload, S: Payload>(term: &Term<C, S>, model: &AnalysisState) -> bool {
    match &term.kind {
        TermKind::Soac(SoacOp::Map { lam, inputs, .. }) => {
            // Multi-input map isn't eligible (the body reads parallel
            // streams; the consume rewrite would only own one of them).
            inputs.len() == 1
                && unique_input_var(&inputs[0], model).is_some_and(|input_sym| {
                    map_body_ok(&lam.lam) && !body_references_sym(&lam.lam.body, input_sym)
                })
        }
        TermKind::Soac(SoacOp::Scan { op, input, .. }) => {
            unique_input_var(input, model).is_some_and(|input_sym| {
                scan_body_ok(&op.lam) && !body_references_sym(&op.lam.body, input_sym)
            })
        }
        TermKind::Soac(SoacOp::Filter { pred, input, .. }) => {
            unique_input_var(input, model).is_some_and(|input_sym| {
                filter_body_ok(&pred.lam) && !body_references_sym(&pred.lam.body, input_sym)
            })
        }
        _ => false,
    }
}

/// Shared ownership-side eligibility check: return the input symbol when the
/// SOAC sees one buffered variable with a unique mutable owner. Callers add
/// SOAC-specific body-shape and pointwise-safety checks. Liveness is
/// deliberately absent here because EGIR owns the physical reuse decision.
fn unique_input_var<C: Payload, S: Payload>(
    input: &ArrayExpr<C, S>,
    model: &AnalysisState,
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
    // Only an externally-backed unique array has a stable ownership identity
    // at this source boundary. EGIR may fuse or materialize the SOAC later, so
    // TLC records only `UniqueInput` and does not commit to `InputBuffer`.
    if !matches!(origin, Origin::Fresh | Origin::UniqueParam | Origin::Entry) {
        return None;
    }
    Some(input_sym)
}

/// Map's body shape: single param whose type matches the lambda's
/// return — so the per-iteration write fits back into the input's
/// element slot.
fn map_body_ok<C: Payload, S: Payload>(lam: &Lambda<C, S>) -> bool {
    lam.params.len() == 1 && lam.params[0].1 == lam.ret_ty
}

/// Scan's body shape: `|acc, elem| _` where the elem-param type
/// matches the lambda's return (= accumulator type = element type).
fn scan_body_ok<C: Payload, S: Payload>(lam: &Lambda<C, S>) -> bool {
    lam.params.len() == 2 && lam.params[1].1 == lam.ret_ty
}

/// Filter's body shape: single param, returns `bool`. The pred's
/// param type already matches the input's element type by
/// type-checking; we just confirm the boolean return.
fn filter_body_ok<C: Payload, S: Payload>(lam: &Lambda<C, S>) -> bool {
    lam.params.len() == 1 && matches!(&lam.ret_ty, Type::Constructed(TypeName::Bool, _))
}

fn body_references_sym<C: Payload, S: Payload>(term: &Term<C, S>, sym: SymbolId) -> bool {
    let mut found = false;
    term.walk(&mut |term: &Term<C, S>| {
        if matches!(&term.kind, TermKind::Var(VarRef::Symbol(candidate)) if *candidate == sym) {
            found = true;
        }
        if found {
            WalkDecision::Prune
        } else {
            WalkDecision::Recurse
        }
    });
    found
}
