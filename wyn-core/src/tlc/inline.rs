//! TLC inlining passes.
//!
//! The monomorphic passes inline small functions/constants and force array-work
//! helpers across call boundaries. After defunctionalization, the same generic
//! inliner folds compiler-generated lifted lambdas back into their call sites.

use super::data::{Empty, ExplicitCapturesPayload, ExplicitClosurePayload};
use super::defunctionalize::{ClosureConverted, Defunctionalized};
use super::rep_specialize::RepSpecialized;
use super::VarRef;
use super::{
    clone_term_with_fresh_ids, extract_lambda_params, term_size, Def, DefMeta, Payload, Program,
    RewriteDecision, Term, TermId, TermIdSource, TermKind, TermRewriter,
};
use crate::ast::{Span, TypeName};
use crate::{LookupMap, LookupSet};
use crate::{SymbolId, SymbolTable};
use polytype::Type;

#[derive(Debug, Clone, Copy, Default)]
pub struct SmallInlined;

impl super::Stage for SmallInlined {
    type Family = super::monomorphize::Monomorphic;
    type GlobalContext = super::context::RewriteGlobal;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SoacHelpersInlined;

impl super::Stage for SoacHelpersInlined {
    type Family = super::monomorphize::Monomorphic;
    type GlobalContext = super::context::RewriteGlobal;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct GeneratedLambdasFolded;

impl super::Stage for GeneratedLambdasFolded {
    type Family = super::defunctionalize::ClosureConverted;
    type GlobalContext = super::context::PostClosureGlobal;
}

/// Maximum term size for a user function to be inlined.
const INLINE_SIZE_THRESHOLD: usize = 30;

/// Force-inline every helper whose body contains a SOAC or an explicit array
/// producer anywhere in its term tree. This mirrors Futhark's "inline
/// array/parallel callees" rule: a helper that performs SOAC work or constructs
/// a range/literal behind a call boundary blocks EGIR from seeing the complete
/// producer/consumer relationship and dispatch extent, so we expose it by
/// inlining. Size threshold is ignored — such a helper is by definition
/// critical to fusion and scheduling, regardless of source LOC.
///
/// Iterates to a fixpoint so chains like `clump → center → sum` (each a
/// SOAC helper) fully expand: one round inlines `center`, the next sees
/// `sum` calls inside the freshly-expanded clump body and inlines those
/// too.
pub fn run_force_soac_helpers(mut program: Program<SmallInlined>) -> Program<SoacHelpersInlined> {
    force_inline_array_work_helpers_to_fixpoint(&mut program);
    debug_assert!(
        verify_array_work_helpers_inlined(&program).is_ok(),
        "force-inline left an array-work helper behind a call boundary; \
         semantic EGIR would need an interprocedural path: {:?}",
        verify_array_work_helpers_inlined(&program).err(),
    );
    program.into_stage()
}

#[derive(Debug, PartialEq, Eq)]
struct CalledArrayWorkHelper {
    caller: SymbolId,
    callee: SymbolId,
}

/// EGIR fusion and dispatch inference are deliberately intraprocedural. Keep
/// source-level inlining as the one boundary operation that exposes every
/// array-work helper before conversion, then let EGIR own all
/// producer/consumer and scheduling decisions.
fn verify_array_work_helpers_inlined(
    program: &Program<SmallInlined>,
) -> Result<(), Vec<CalledArrayWorkHelper>> {
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

fn force_inline_array_work_helpers_to_fixpoint(program: &mut Program<SmallInlined>) {
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
        crate::map_in_place(&mut program.defs, |def| {
            let body = inline_term(def.body, &candidates, term_ids);
            Def { body, ..def }
        });
        super::dce::eliminate_unreachable_defs(&mut program.defs);
    }
}

fn build_array_work_helper_candidates(
    program: &Program<SmallInlined>,
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
        // Skip helpers whose body carries an unresolved polytype `Variable`
        // (a free *type* parameter — distinct from a `SizeVar`). The
        // prelude's `unzip<[n], A, B>` is the load-bearing example: its
        // body contains SOAC `map` calls, so it qualifies as a SOAC
        // helper, but if we force-inline it the body's free `A`/`B`
        // type variables go nowhere — `monomorphize` only resolves
        // polymorphism at *function-definition* boundaries, not inside
        // already-inlined bodies, so the variables ride all the way to
        // the SPIR-V backend and panic on "Unresolved type variable".
        // Defer such helpers to the canonical
        // `monomorphize → fold_generated_lambdas → inline_small` path
        // that actually unifies their call sites.
        if term_contains_free_type_variable(&body)
            || params.iter().any(|(_, ty)| type_contains_variable(ty))
        {
            continue;
        }
        candidates.insert(def.name, InlineBody { params, body });
    }
    candidates
}

/// True if `ty` references an unresolved *value* type parameter (a free
/// polytype `Type::Variable`). For an array, only the element slot can carry
/// one; the variant / size / buffer slots carry representation, size, and
/// buffer variables that are resolved on their own axes and don't leak to the
/// backend as "unresolved type variable", so they are not counted.
fn type_contains_variable(ty: &Type<TypeName>) -> bool {
    match ty {
        Type::Variable(_) => true,
        Type::Constructed(TypeName::Array, args) => args.first().is_some_and(type_contains_variable),
        Type::Constructed(_, args) => args.iter().any(type_contains_variable),
    }
}

/// True if any subterm's carried type contains a polytype `Variable` —
/// or, equivalently, if the def's body still carries free type
/// parameters that monomorphize would resolve.
fn term_contains_free_type_variable<C: Payload, S: Payload>(term: &Term<C, S>) -> bool {
    if type_contains_variable(&term.ty) {
        return true;
    }
    let mut found = false;
    term.for_each_child(&mut |c| {
        if !found {
            found = term_contains_free_type_variable(c);
        }
    });
    found
}

/// True if `term` contains a SOAC, an explicit array producer, or a `length`
/// call. All must be visible in the caller so EGIR can build complete
/// producer/use edges and derive dispatch extents without interprocedural
/// summaries.
pub(super) fn contains_array_work<C: Payload, S: Payload>(term: &Term<C, S>) -> bool {
    if matches!(&term.kind, TermKind::Soac(_) | TermKind::ArrayExpr(_)) {
        return true;
    }
    if is_length_intrinsic_call(term) {
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

fn is_length_intrinsic_call<C: Payload, S: Payload>(term: &Term<C, S>) -> bool {
    let TermKind::App { func, args } = &term.kind else {
        return false;
    };
    if args.len() != 1 {
        return false;
    }
    let TermKind::Var(super::VarRef::Builtin { id, .. }) = &func.kind else {
        return false;
    };
    *id == crate::builtins::catalog().known().length
}

fn any_def_calls_candidate(
    program: &Program<SmallInlined>,
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

/// Inline small user functions and constants at their call/reference sites.
///
/// This is the TLC equivalent of `ssa::ssa_inline::inline_small_functions`.
/// Inlines:
/// - Small user functions (term size ≤ threshold, no control flow or SOACs)
/// - Constants (arity-0 defs, substituted at Var reference sites)
///
/// Skips `LiftedLambda` defs (SOAC bodies) — handled by `inline()`.
pub fn run_small(mut program: Program<RepSpecialized>) -> Program<SmallInlined> {
    let all_constants = find_all_constants(&program);
    let mut small_candidates = find_small_candidates(&program.defs, &program.symbols);

    if small_candidates.is_empty() && all_constants.is_empty() {
        return program.into_stage();
    }

    // Inline constants into small candidate bodies so that when we inline
    // the candidate into a call site, the inlined body doesn't carry stale
    // Var references to constants.
    small_candidates = small_candidates
        .into_iter()
        .map(|(symbol, mut candidate)| {
            candidate.body = inline_constants(candidate.body, &all_constants, &mut program.term_ids);
            (symbol, candidate)
        })
        .collect();

    let term_ids = &mut program.term_ids;
    crate::map_in_place(&mut program.defs, |def| {
        // Constants are pure — inline them everywhere, including lambda bodies.
        let body = inline_constants(def.body, &all_constants, term_ids);
        let body = inline_term(body, &small_candidates, term_ids);
        Def { body, ..def }
    });

    super::dce::eliminate_unreachable_defs(&mut program.defs);
    program.into_stage()
}

/// Inline compiler-generated lifted-lambda defs (`DefMeta::LiftedLambda`) in a TLC program.
pub fn fold_generated_lambdas(mut program: Program<Defunctionalized>) -> Program<GeneratedLambdasFolded> {
    let inline_candidates = find_inline_candidates(&program.defs, &program.symbols);

    let term_ids = &mut program.term_ids;
    crate::map_in_place(&mut program.defs, |def| {
        let body = inline_term(def.body, &inline_candidates, term_ids);
        Def { body, ..def }
    });

    // DCE: remove defs not referenced by any entry point or reachable def.
    super::dce::eliminate_unreachable_defs(&mut program.defs);

    program.assert_flat_apps();
    program.into_stage()
}

// =============================================================================
// Inline candidate analysis
// =============================================================================

/// A function body ready for inlining: flat params + inner body.
struct InlineBody<C: Payload, S: Payload> {
    params: Vec<(SymbolId, Type<TypeName>)>,
    body: Term<C, S>,
}

/// Find small user functions suitable for inlining.
///
/// A function qualifies if:
/// - It's a `DefMeta::Function` (lifted lambdas are handled by
///   `fold_generated_lambdas`)
/// - It has parameters (not a constant)
/// - Its body is small (term_size ≤ threshold)
/// - Its body has no control flow (If, Loop) or SOACs
fn find_small_candidates(
    defs: &[Def<super::monomorphize::Monomorphic>],
    _symbols: &SymbolTable,
) -> LookupMap<SymbolId, InlineBody<Empty, Empty>> {
    let mut candidates = LookupMap::new();

    for def in defs {
        if !matches!(def.meta, DefMeta::Function) {
            continue;
        }

        let (params, body) = extract_lambda_params(&def.body);
        if params.is_empty() {
            continue; // constant, not a function
        }

        // Size check.
        if term_size(&body) > INLINE_SIZE_THRESHOLD {
            continue;
        }

        // Skip functions with control flow (If, Loop) — they'd become multi-block in SSA.
        // SOACs are fine to inline — they're single instructions in SSA.
        if has_control_flow(&body) {
            continue;
        }

        candidates.insert(def.name, InlineBody { params, body });
    }

    candidates
}

/// Find all constant defs, indexed by every SymbolId that could reference them.
///
/// A constant is an arity-0, non-entry, non-extern function def.
/// After monomorphization, the same constant name may be referenced through
/// different SymbolIds, so we index by name as well as by def SymbolId.
fn find_all_constants(program: &Program<RepSpecialized>) -> LookupMap<SymbolId, Term<Empty, Empty>> {
    // Find the canonical constant defs.
    let mut by_sym: LookupMap<SymbolId, Term<Empty, Empty>> = LookupMap::new();
    let mut by_name: LookupMap<String, Term<Empty, Empty>> = LookupMap::new();

    for def in &program.defs {
        if !matches!(def.meta, DefMeta::Function) {
            continue;
        }
        if matches!(def.body.kind, TermKind::Extern(_)) {
            continue;
        }
        if def.arity != 0 {
            continue;
        }
        by_sym.insert(def.name, def.body.clone());
        if let Some(name) = program.symbols.get(def.name) {
            by_name.insert(name.clone(), def.body.clone());
        }
    }

    // Also map any other SymbolId that resolves to a constant's name.
    for def in &program.defs {
        if let Some(name) = program.symbols.get(def.name) {
            if let Some(body) = by_name.get(name) {
                by_sym.entry(def.name).or_insert_with(|| body.clone());
            }
        }
    }
    for (name, body) in &by_name {
        if let Some(&def_sym) = program.def_syms.get(name) {
            by_sym.entry(def_sym).or_insert_with(|| body.clone());
        }
    }

    by_sym
}

/// Check if a term contains control flow (If, Loop) or SOACs.
fn has_control_flow<C: Payload, S: Payload>(term: &Term<C, S>) -> bool {
    match &term.kind {
        TermKind::If { .. } | TermKind::Loop { .. } => true,
        _ => {
            let mut found = false;
            term.for_each_child(&mut |child| {
                if !found {
                    found = has_control_flow(child);
                }
            });
            found
        }
    }
}

/// Replace `Var(sym)` references with the constant body when `sym` is a constant candidate.
fn inline_constants(
    term: Term<Empty, Empty>,
    constants: &LookupMap<SymbolId, Term<Empty, Empty>>,
    term_ids: &mut TermIdSource,
) -> Term<Empty, Empty> {
    term.rewrite(&mut ConstantInliner { constants, term_ids })
}

struct ConstantInliner<'a, 'ids> {
    constants: &'a LookupMap<SymbolId, Term<Empty, Empty>>,
    term_ids: &'ids mut TermIdSource,
}

impl TermRewriter<Empty, Empty> for ConstantInliner<'_, '_> {
    fn next_term_id(&mut self) -> TermId {
        self.term_ids.next_id()
    }

    fn rewrite_node(&mut self, term: &mut Term<Empty, Empty>) -> RewriteDecision {
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
    term.rewrite(&mut FunctionInliner { candidates, term_ids })
}

struct FunctionInliner<'a, 'ids, C: Payload, S: Payload> {
    candidates: &'a LookupMap<SymbolId, InlineBody<C, S>>,
    term_ids: &'ids mut TermIdSource,
}

impl<C: Payload, S: Payload> TermRewriter<C, S> for FunctionInliner<'_, '_, C, S> {
    fn next_term_id(&mut self) -> TermId {
        self.term_ids.next_id()
    }

    fn rewrite_node(&mut self, term: &mut Term<C, S>) -> RewriteDecision {
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
            return RewriteDecision::Unchanged;
        };
        let params = candidate.params.clone();
        let body = clone_term_with_fresh_ids(&candidate.body, self.term_ids);

        let kind = std::mem::replace(&mut term.kind, TermKind::UnitLit);
        let TermKind::App { args, .. } = kind else {
            unreachable!()
        };
        let mut replacement = build_inline_lets(&params, args, body, term.span, self.term_ids);
        replacement.id = term.id;
        *term = replacement;
        RewriteDecision::Changed
    }
}

// =============================================================================
// Small helpers
// =============================================================================

fn mk<C: Payload, S: Payload>(
    ids: &mut TermIdSource,
    ty: Type<TypeName>,
    span: Span,
    kind: TermKind<C, S>,
) -> Term<C, S> {
    Term {
        id: ids.next_id(),
        ty,
        span,
        kind,
    }
}

/// Build Let bindings to substitute params with args, wrapping the inlined body.
///
/// The let's `name_ty` comes from the *arg*'s concrete type rather than the
/// param's declared type. The param's declared type may be polymorphic
/// (e.g. `[]T` is `Array[T, Abstract, Skolem, …]`); the arg at the call site
/// has the concrete instantiation. Using the arg type avoids dragging the
/// Abstract array variant into post-inline let chains where it would later
/// hit `egir::verify_no_abstract` at backend lowering.
///
/// Special case: when the arg is a bare `Var(SymbolId)` reference (i.e. the
/// caller is passing an in-scope binding straight through), we substitute
/// `param_sym → arg_sym` into the body instead of emitting a redundant
/// `let param_sym = arg_var in body`. The alias-let is correct in
/// principle but downstream passes — `egir::from_tlc::convert_soac_filter`
/// in particular — read attributes (storage region, ownership) directly
/// off the original symbol and don't follow let-bound aliases, so a
/// post-inline `let arr = xs in filter(p, arr)` loses the entry-param's
/// region info. This is *not* general beta-reduction — only the trivial
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
    let mut result = body;
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
        result = mk(
            ids,
            result.ty.clone(),
            span,
            TermKind::Let {
                name: *sym,
                name_ty: arg.ty.clone(),
                rhs: Box::new(arg),
                body: Box::new(result),
            },
        );
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
        &mut |occurrence, ids| Term {
            id: ids.next_id(),
            ty: new_ty.clone(),
            kind: TermKind::Var(VarRef::Symbol(new)),
            span: occurrence.span,
        },
        term_ids,
    )
}
