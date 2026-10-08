//! TLC inlining passes.
//!
//! The monomorphic pass expands constants and forces array-work helpers across
//! call boundaries. After defunctionalization, the same generic inliner folds
//! compiler-generated lifted lambdas back into their call sites.

use super::data::{Empty, ExplicitCapturesPayload, ExplicitClosurePayload};
use super::defunctionalize::{ClosureConverted, Defunctionalized};
use super::rep_specialize::RepSpecialized;
use super::VarRef;
use super::{
    clone_term_with_fresh_ids, extract_lambda_params, Bindings, Def, DefMeta, LetBinding, Payload,
    RewriteDecision, Term, TermId, TermIdSource, TermKind, TermRewriter,
};
use crate::ast::{Span, TypeName};
use crate::builtins;
use crate::map_in_place;
use crate::{LookupMap, LookupSet};
use crate::{SymbolId, SymbolTable};
use polytype::Type;

#[derive(Debug, Clone, Copy)]
pub enum SoacHelpersInlinedTag {}
pub type SoacHelpersInlined =
    super::Program<SoacHelpersInlinedTag, super::monomorphize::Monomorphic, super::context::RewriteGlobal>;

#[derive(Debug, Clone, Copy)]
pub enum GeneratedLambdasFoldedTag {}
pub type GeneratedLambdasFolded = super::Program<
    GeneratedLambdasFoldedTag,
    super::defunctionalize::ClosureConverted,
    super::context::PostClosureGlobal,
>;

/// Expand constants, then force-inline helpers whose bodies contain array work.
/// Array computations and shape queries must be visible in the caller so
/// egglog can infer producer/consumer relationships and dispatch extents.
/// Scalar helper optimization is left to egglog.
///
/// Iterates to a fixpoint so chains like `clump → center → sum` (each a
/// SOAC helper) fully expand: one round inlines `center`, the next sees
/// `sum` calls inside the freshly-expanded clump body and inlines those
/// too.
pub fn force_inline_soac_helpers(mut program: RepSpecialized) -> SoacHelpersInlined {
    expand_constants(&mut program);
    super::dce::eliminate_unreachable_defs(&mut program.defs);
    force_inline_array_work_helpers_to_fixpoint(&mut program);
    debug_assert!(
        verify_array_work_helpers_inlined(&program).is_ok(),
        "force-inline left an array-work helper behind a call boundary; \
         egglog would need an interprocedural path: {:?}",
        verify_array_work_helpers_inlined(&program).err(),
    );
    program.retag()
}

#[derive(Debug, PartialEq, Eq)]
struct CalledArrayWorkHelper {
    caller: SymbolId,
    callee: SymbolId,
}

/// egglog fusion and dispatch inference are deliberately intraprocedural. Keep
/// source-level inlining as the one boundary operation that exposes every
/// array-work helper before conversion, then let egglog own all
/// producer/consumer and scheduling decisions.
fn verify_array_work_helpers_inlined(program: &RepSpecialized) -> Result<(), Vec<CalledArrayWorkHelper>> {
    let array_work_bearing: LookupSet<SymbolId> =
        program.defs.iter().filter(|def| contains_array_work(&def.body)).map(|def| def.name).collect();
    let mut violations = Vec::new();
    for def in &program.defs {
        collect_called_array_work_helpers(&def.body, def.name, &array_work_bearing, &mut violations);
    }
    if violations.is_empty() {
        Ok(())
    } else {
        Err(violations)
    }
}

fn collect_called_array_work_helpers(
    term: &Term<Empty, Empty>,
    caller: SymbolId,
    array_work_bearing: &LookupSet<SymbolId>,
    out: &mut Vec<CalledArrayWorkHelper>,
) {
    if let TermKind::App { func, .. } = &term.kind {
        if let TermKind::Var(VarRef::Symbol(callee)) = &func.kind {
            if array_work_bearing.contains(callee) {
                out.push(CalledArrayWorkHelper {
                    caller,
                    callee: *callee,
                });
            }
        }
    }
    term.for_each_child(&mut |child| {
        collect_called_array_work_helpers(child, caller, array_work_bearing, out)
    });
}

fn force_inline_array_work_helpers_to_fixpoint(program: &mut RepSpecialized) {
    // Bound iterations to guard against pathological recursion through
    // hand-crafted call graphs; typical wyn helper depth is 2–3.
    for _ in 0..8 {
        let candidates = build_array_work_helper_candidates(program);
        if candidates.is_empty() {
            return;
        }
        // Stop when nothing in the program calls any current candidate.
        // (Inlining one round may expose new candidates — e.g. inlining
        // `sum` into `center`'s body makes `center` SOAC-bearing and a
        // candidate next round — so we re-detect candidates each iter.)
        if !any_def_calls_candidate(program, &candidates) {
            return;
        }
        let term_ids = &mut program.term_ids;
        map_in_place(&mut program.defs, |def| {
            let body = inline_term(def.body, &candidates, term_ids);
            Def { body, ..def }
        });
        super::dce::eliminate_unreachable_defs(&mut program.defs);
    }
}

fn build_array_work_helper_candidates(
    program: &RepSpecialized,
) -> LookupMap<SymbolId, InlineBody<Empty, Empty>> {
    let mut candidates = LookupMap::new();
    for def in &program.defs {
        if !matches!(def.meta, DefMeta::Function) {
            continue;
        }
        // Entry points are roots, never call sites.
        let (params, body) = extract_lambda_params(&def.body);
        if params.is_empty() {
            continue;
        }
        // Any helper containing array work is a candidate, control flow or
        // not, so neither a SOAC nor an explicit range/literal producer is
        // reachable behind a call (`verify_array_work_helpers_inlined`).
        if !contains_array_work(&body) {
            continue;
        }
        candidates.insert(def.name, InlineBody { params, body });
    }
    candidates
}

/// True if `term` contains a SOAC, an explicit array producer, or a `length` / `#[scratch]`
/// operation. All must be visible in the caller so egglog can build complete
/// producer/use edges and derive dispatch extents without interprocedural
/// summaries.
pub(super) fn contains_array_work<C: Payload, S: Payload>(term: &Term<C, S>) -> bool {
    if matches!(&term.kind, TermKind::Soac(_) | TermKind::ArrayExpr(_)) {
        return true;
    }
    if is_array_shape_intrinsic_call(term) {
        return true;
    }
    let mut found = false;
    term.for_each_child(&mut |c| {
        if !found {
            found = contains_array_work(c);
        }
    });
    found
}

fn is_array_shape_intrinsic_call<C: Payload, S: Payload>(term: &Term<C, S>) -> bool {
    let TermKind::App { func, args } = &term.kind else {
        return false;
    };
    if args.len() != 1 {
        return false;
    }
    let TermKind::Var(super::VarRef::Builtin { id, .. }) = &func.kind else {
        return false;
    };
    *id == builtins::catalog().known().length || *id == builtins::catalog().known().scratch_annotation
}

fn any_def_calls_candidate(
    program: &RepSpecialized,
    candidates: &LookupMap<SymbolId, InlineBody<Empty, Empty>>,
) -> bool {
    fn walk(term: &Term<Empty, Empty>, cs: &LookupMap<SymbolId, InlineBody<Empty, Empty>>) -> bool {
        if let TermKind::App { func, .. } = &term.kind {
            if let TermKind::Var(VarRef::Symbol(s)) = &func.kind {
                if cs.contains_key(s) {
                    return true;
                }
            }
        }
        let mut found = false;
        term.for_each_child(&mut |c| {
            if !found {
                found = walk(c, cs);
            }
        });
        found
    }
    program.defs.iter().any(|def| walk(&def.body, candidates))
}

/// Expose constant bodies before selecting helpers that contain array work.
fn expand_constants(program: &mut RepSpecialized) {
    let all_constants = find_all_constants(program);
    let mut constants = ConstantInliner {
        constants: &all_constants,
        term_ids: &mut program.term_ids,
    };
    for def in &mut program.defs {
        constants.rewrite_tracked(&mut def.body);
    }
}

/// Inline compiler-generated lifted lambdas, remove unreachable definitions,
/// and verify that the remaining definitions have no function-typed parameters.
pub fn fold_generated_lambdas(mut program: Defunctionalized) -> GeneratedLambdasFolded {
    let inline_candidates = find_inline_candidates(&program.defs, &program.symbols);

    let term_ids = &mut program.term_ids;
    map_in_place(&mut program.defs, |def| {
        let body = inline_term(def.body, &inline_candidates, term_ids);
        Def { body, ..def }
    });

    // DCE: remove defs not referenced by any entry point or reachable def.
    super::dce::eliminate_unreachable_defs(&mut program.defs);

    program.assert_flat_apps();
    super::defunctionalize::verify_hof_specialized(&program).unwrap_or_else(|error| {
        panic!("hof-specialization verifier failed after fold_generated_lambdas: {error}")
    });
    program.retag()
}

// =============================================================================
// Inline candidate analysis
// =============================================================================

/// A function body ready for inlining: flat params + inner body.
struct InlineBody<C: Payload, S: Payload> {
    params: Vec<(SymbolId, Type<TypeName>)>,
    body: Term<C, S>,
}

/// Find zero-arity, non-entry, non-extern definitions by their SymbolId.
fn find_all_constants(program: &RepSpecialized) -> LookupMap<SymbolId, Term<Empty, Empty>> {
    program
        .defs
        .iter()
        .filter(|def| matches!(def.meta, DefMeta::Function))
        .filter(|def| !matches!(def.body.kind, TermKind::Extern(_)))
        .filter(|def| def.arity == 0)
        .map(|def| (def.name, def.body.clone()))
        .collect()
}

struct ConstantInliner<'a, 'ids> {
    constants: &'a LookupMap<SymbolId, Term<Empty, Empty>>,
    term_ids: &'ids mut TermIdSource,
}

impl TermRewriter<Empty, Empty> for ConstantInliner<'_, '_> {
    fn next_term_id(&mut self) -> TermId {
        self.term_ids.next_id()
    }

    fn rewrite_node_before_children(&mut self, term: &mut Term<Empty, Empty>) -> RewriteDecision {
        let TermKind::Var(VarRef::Symbol(symbol)) = &term.kind else {
            return RewriteDecision::Unchanged;
        };
        let Some(template) = self.constants.get(symbol) else {
            return RewriteDecision::Unchanged;
        };
        let mut replacement = clone_term_with_fresh_ids(template, self.term_ids);
        replacement.id = term.id;
        *term = replacement;
        RewriteDecision::Changed
    }
}

/// Determine which defs are candidates for inlining.
/// Inlines all `DefMeta::LiftedLambda` defs — defunctionalization-produced lifted
/// lambdas that we're putting back where they came from.
fn find_inline_candidates(
    defs: &[Def<ClosureConverted>],
    _symbols: &SymbolTable,
) -> LookupMap<SymbolId, InlineBody<ExplicitClosurePayload, ExplicitCapturesPayload>> {
    let mut candidates = LookupMap::new();

    for def in defs {
        if !matches!(def.meta, DefMeta::LiftedLambda) {
            continue;
        }

        let (params, body) = extract_lambda_params(&def.body);

        if params.is_empty() {
            continue;
        }

        candidates.insert(def.name, InlineBody { params, body });
    }

    candidates
}

// =============================================================================
// Term inlining
// =============================================================================

/// Bottom-up: recurse into all children, then try to inline App nodes.
///
/// SOAC lambda bodies are bare Var refs to lifted defs after defunctionalization,
/// so recursing into them is harmless — the inline rewrite only
/// fires on fully-saturated App nodes matching candidates.
fn inline_term<C: Payload, S: Payload>(
    term: Term<C, S>,
    candidates: &LookupMap<SymbolId, InlineBody<C, S>>,
    term_ids: &mut TermIdSource,
) -> Term<C, S> {
    term.rewrite_owned(&mut FunctionInliner { candidates, term_ids })
}

struct FunctionInliner<'a, 'ids, C: Payload, S: Payload> {
    candidates: &'a LookupMap<SymbolId, InlineBody<C, S>>,
    term_ids: &'ids mut TermIdSource,
}

impl<C: Payload, S: Payload> TermRewriter<C, S> for FunctionInliner<'_, '_, C, S> {
    fn next_term_id(&mut self) -> TermId {
        self.term_ids.next_id()
    }

    fn rewrite_owned_node(&mut self, term: Term<C, S>) -> (Term<C, S>, RewriteDecision) {
        let candidate = match &term.kind {
            TermKind::App { func, args } => match &func.kind {
                TermKind::Var(VarRef::Symbol(symbol)) => {
                    self.candidates.get(symbol).filter(|candidate| args.len() == candidate.params.len())
                }
                _ => None,
            },
            _ => None,
        };
        let Some(candidate) = candidate else {
            return (term, RewriteDecision::Unchanged);
        };
        let params = candidate.params.clone();
        let body = clone_term_with_fresh_ids(&candidate.body, self.term_ids);

        let Term {
            id,
            ty: _,
            span,
            kind,
        } = term;
        let TermKind::App { args, .. } = kind else {
            unreachable!()
        };
        let mut replacement = build_inline_lets(&params, args, body, span, self.term_ids);
        replacement.id = id;
        (replacement, RewriteDecision::Changed)
    }
}

// =============================================================================
// Small helpers
// =============================================================================

/// Build Let bindings to substitute params with args, wrapping the inlined body.
///
/// The let's `name_ty` comes from the *arg*'s concrete type rather than the
/// param's declared type. The param's declared type may be polymorphic
/// (e.g. `[]T` is `Array[T, Abstract, Skolem, …]`); the arg at the call site
/// has the concrete instantiation. Using the arg type avoids dragging the
/// Abstract array variant into post-inline let chains where it would later
/// hit `ssa::backend_validation` at backend lowering.
///
/// Special case: when the arg is a bare `Var(SymbolId)` reference (i.e. the
/// caller is passing an in-scope binding straight through), we substitute
/// `param_sym → arg_sym` into the body instead of emitting a redundant
/// `let param_sym = arg_var in body`. The alias-let is correct in
/// principle, but substituting the original symbol directly keeps its
/// storage-region and ownership metadata attached to downstream uses. This is *not* general beta-reduction — only the trivial
/// `let x = y in body` case where `y` is a `Var`. Non-Var args still
/// get the `let` wrap so we don't duplicate side-effecting computation
/// under multi-use params.
pub(crate) fn build_inline_lets<C: Payload, S: Payload>(
    params: &[(SymbolId, Type<TypeName>)],
    args: Vec<Term<C, S>>,
    body: Term<C, S>,
    span: Span,
    ids: &mut TermIdSource,
) -> Term<C, S> {
    let mut body_bindings = Bindings::new();
    let tail = body_bindings.append(body);
    let mut result = body_bindings.finish(tail, ids);
    for ((sym, _param_ty), arg) in params.iter().rev().zip(args.into_iter().rev()) {
        if let TermKind::Var(VarRef::Symbol(arg_sym)) = &arg.kind {
            // Substituting the *symbol* alone is not enough: the param's
            // declared type may be polymorphic (e.g. `[]T` with a region
            // type-variable), while the call-site arg carries the
            // concrete instantiation (a `Buffer(set, binding)`). If we
            // only rewrite the symbol, the substituted `Var` still
            // carries the param's polymorphic type and downstream type
            // walks (notably the SPIR-V backend's view-region check)
            // see an unresolved type variable. So replace both: the
            // var ref *and* its type, at every occurrence.
            result = substitute_sym_and_retype(result, *sym, *arg_sym, &arg.ty, ids);
            continue;
        }
        let mut bindings = Bindings::new();
        bindings.push(LetBinding {
            name: *sym,
            name_ty: arg.ty.clone(),
            rhs: arg,
            span,
        });
        result = bindings.finish(result, ids);
    }
    result
}

/// `substitute_sym` variant that *also* overrides the substituted
/// `Var(old)`'s carried type with `new_ty`. Used by `build_inline_lets`
/// for the bare-Var-arg fast path: when we forward a call-site
/// concrete-region arg through a polymorphic param, the substituted
/// `Var` must end up carrying the concrete type, not the polymorphic
/// param type that was on the original term.
fn substitute_sym_and_retype<C: Payload, S: Payload>(
    term: Term<C, S>,
    old: SymbolId,
    new: SymbolId,
    new_ty: &Type<TypeName>,
    term_ids: &mut TermIdSource,
) -> Term<C, S> {
    super::subst::substitute_with(
        term,
        old,
        &mut |occurrence, ids| {
            Term::fresh(
                ids,
                new_ty.clone(),
                occurrence.span,
                TermKind::Var(VarRef::Symbol(new)),
            )
        },
        term_ids,
    )
}
