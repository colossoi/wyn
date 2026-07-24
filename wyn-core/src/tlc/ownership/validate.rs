//! Early source-level ownership validation.
//!
//! This pass runs before simplification and inlining can erase the call
//! boundaries that carry source `*T` contracts. It does not rewrite TLC.

use super::analysis::{analyze, owner_display_name, AnalysisState, Origin, OwnerId};
use crate::ast::TypeName;
use crate::builtins::catalog;
use crate::error::CompilerError;
use crate::tlc::data::Empty;
use crate::tlc::stage::BuffersPinned;
use crate::tlc::{
    var_term_builtin_id, ArrayExpr, LoopKind, Program, SoacBody, SoacOp, Term, TermId, TermKind, VarRef,
};
use crate::LookupSet;
use polytype::Type;

#[derive(Debug, Clone, Copy, Default)]
pub struct OwnershipValidated;

impl crate::tlc::Stage for OwnershipValidated {
    type Family = crate::tlc::family::Polymorphic;
    type GlobalContext = crate::tlc::context::RewriteGlobal;
}

pub fn validate(program: Program<BuffersPinned>) -> crate::error::Result<Program<OwnershipValidated>> {
    check(&program)?;
    Ok(program.into_stage())
}

/// Run the ownership analysis and report a use-after-move error if
/// any owner is consumed at a program point where a successor still
/// reads it. Used by the `wyn check` subcommand and indirectly by
/// `promote_inplace`.
pub fn check(program: &Program<BuffersPinned>) -> crate::error::Result<()> {
    let model = analyze(program);
    if let Some(err) = check_use_after_move(program, &model) {
        return Err(err);
    }
    if let Some(err) = check_linear_image_results(program, &model) {
        return Err(err);
    }
    Ok(())
}

/// Walk the model for any owner that is killed at a term while still
/// being in that term's `live_out` — i.e., a successor still reads it.
/// Returns the first such violation as a CompilerError; later
/// violations are not reported in this pass (one diagnostic per
/// compile is consistent with how the rest of the pipeline reports).
fn check_use_after_move(program: &Program<BuffersPinned>, model: &AnalysisState) -> Option<CompilerError> {
    use CompilerError;
    if let Some((msg, span)) = model.build_errors.first() {
        return Some(CompilerError::AliasError(msg.clone(), *span));
    }
    let mut violations: Vec<(TermId, OwnerId)> = model
        .kills
        .iter()
        .flat_map(|(id, killed)| {
            let live = model.live_out.get(id);
            killed.iter().filter(move |o| live.map_or(false, |s| s.contains(o))).map(move |o| (*id, *o))
        })
        .collect();
    violations.sort_by_key(|(id, _)| id.0);
    let (term_id, owner) = violations.into_iter().next()?;
    let span = model.term_spans.get(&term_id).copied();
    let var_name = owner_display_name(model, &program.symbols, owner);
    Some(CompilerError::AliasError(
        format!("use of moved value `{}`", var_name),
        span,
    ))
}

fn check_linear_image_results(
    program: &Program<BuffersPinned>,
    model: &AnalysisState,
) -> Option<CompilerError> {
    for def in &program.defs {
        if let Some(err) =
            check_linear_image_results_in_term(&def.body, program, model, LinearImageUseContext::Used)
        {
            return Some(err);
        }
    }
    None
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LinearImageUseContext {
    Used,
    Discarded,
}

fn is_image_with_app(term: &Term<Empty, Empty>, program: &Program<BuffersPinned>) -> bool {
    match &term.kind {
        TermKind::App { func, args } if args.len() == 3 => {
            var_term_builtin_id(func, &program.symbols) == Some(catalog().known().image_with)
        }
        _ => false,
    }
}

fn is_image_load_app(term: &Term<Empty, Empty>, program: &Program<BuffersPinned>) -> bool {
    match &term.kind {
        TermKind::App { func, args } if args.len() == 2 => {
            var_term_builtin_id(func, &program.symbols) == Some(catalog().known().image_load)
        }
        _ => false,
    }
}

fn image_update_drop_error(term: &Term<Empty, Empty>, model: &AnalysisState) -> CompilerError {
    CompilerError::AliasError(
        "linear storage-image update result must be threaded to another update, observed, returned, or used as the unit entry tail"
            .to_string(),
        model.term_spans.get(&term.id).copied().or(Some(term.span)),
    )
}

fn check_linear_image_results_in_term(
    term: &Term<Empty, Empty>,
    program: &Program<BuffersPinned>,
    model: &AnalysisState,
    context: LinearImageUseContext,
) -> Option<CompilerError> {
    if is_image_with_app(term, program) && context == LinearImageUseContext::Discarded {
        return Some(image_update_drop_error(term, model));
    }

    if let TermKind::Let {
        name_ty, rhs, body, ..
    } = &term.kind
    {
        if matches!(name_ty, Type::Constructed(TypeName::StorageTexture, _)) {
            let introduced: LookupSet<OwnerId> = model
                .defs
                .get(&term.id)
                .into_iter()
                .flatten()
                .copied()
                .filter(|owner| model.origin(*owner) == Some(Origin::Fresh))
                .collect();
            if !introduced.is_empty() && !linear_image_owner_is_threaded(body, &introduced, model) {
                let span = model.term_spans.get(&rhs.id).copied().or(Some(rhs.span));
                return Some(CompilerError::AliasError(
                    "linear storage-image update result must be threaded to another update, observed, returned, or used as the unit entry tail"
                        .to_string(),
                    span,
                ));
            }
        }
    }

    match &term.kind {
        TermKind::Let {
            name_ty, rhs, body, ..
        } => {
            let rhs_context = if matches!(name_ty, Type::Constructed(TypeName::StorageTexture, _)) {
                LinearImageUseContext::Used
            } else {
                LinearImageUseContext::Discarded
            };
            check_linear_image_results_in_term(rhs, program, model, rhs_context)
                .or_else(|| check_linear_image_results_in_term(body, program, model, context))
        }
        TermKind::Lambda(lam) => {
            check_linear_image_results_in_term(&lam.body, program, model, LinearImageUseContext::Used)
        }
        TermKind::If {
            cond,
            then_branch,
            else_branch,
        } => check_linear_image_results_in_term(cond, program, model, LinearImageUseContext::Discarded)
            .or_else(|| check_linear_image_results_in_term(then_branch, program, model, context))
            .or_else(|| check_linear_image_results_in_term(else_branch, program, model, context)),
        TermKind::Loop {
            init,
            init_bindings,
            kind,
            body,
            ..
        } => {
            let init_err =
                check_linear_image_results_in_term(init, program, model, LinearImageUseContext::Used);
            if init_err.is_some() {
                return init_err;
            }
            for (_, _, extract) in init_bindings {
                if let Some(err) =
                    check_linear_image_results_in_term(extract, program, model, LinearImageUseContext::Used)
                {
                    return Some(err);
                }
            }
            let kind_err = match kind {
                LoopKind::For { iter, .. } => check_linear_image_results_in_term(
                    iter,
                    program,
                    model,
                    LinearImageUseContext::Discarded,
                ),
                LoopKind::ForRange { bound, .. } => check_linear_image_results_in_term(
                    bound,
                    program,
                    model,
                    LinearImageUseContext::Discarded,
                ),
                LoopKind::While { cond } => check_linear_image_results_in_term(
                    cond,
                    program,
                    model,
                    LinearImageUseContext::Discarded,
                ),
            };
            kind_err.or_else(|| {
                check_linear_image_results_in_term(body, program, model, LinearImageUseContext::Used)
            })
        }
        TermKind::App { func, args } => {
            if let Some(err) =
                check_linear_image_results_in_term(func, program, model, LinearImageUseContext::Discarded)
            {
                return Some(err);
            }
            for (index, arg) in args.iter().enumerate() {
                let arg_context = if (is_image_load_app(term, program) || is_image_with_app(term, program))
                    && index == 0
                {
                    LinearImageUseContext::Used
                } else if matches!(arg.ty, Type::Constructed(TypeName::StorageTexture, _)) {
                    LinearImageUseContext::Used
                } else {
                    LinearImageUseContext::Discarded
                };
                if let Some(err) = check_linear_image_results_in_term(arg, program, model, arg_context) {
                    return Some(err);
                }
            }
            None
        }
        TermKind::Tuple(parts) | TermKind::VecLit(parts) => {
            for part in parts {
                let part_context = if matches!(part.ty, Type::Constructed(TypeName::StorageTexture, _)) {
                    context
                } else {
                    LinearImageUseContext::Discarded
                };
                if let Some(err) = check_linear_image_results_in_term(part, program, model, part_context) {
                    return Some(err);
                }
            }
            None
        }
        TermKind::TupleProj { tuple, idx } => {
            if let TermKind::Tuple(parts) = &tuple.kind {
                for (i, part) in parts.iter().enumerate() {
                    let part_context = if i == *idx { context } else { LinearImageUseContext::Discarded };
                    if let Some(err) =
                        check_linear_image_results_in_term(part, program, model, part_context)
                    {
                        return Some(err);
                    }
                }
                None
            } else {
                check_linear_image_results_in_term(tuple, program, model, context)
            }
        }
        TermKind::Index { array, index } => {
            check_linear_image_results_in_term(array, program, model, LinearImageUseContext::Discarded)
                .or_else(|| {
                    check_linear_image_results_in_term(
                        index,
                        program,
                        model,
                        LinearImageUseContext::Discarded,
                    )
                })
        }
        TermKind::Coerce { inner, .. } => {
            check_linear_image_results_in_term(inner, program, model, context)
        }
        TermKind::Closure(()) => None,
        TermKind::ArrayExpr(ae) => check_linear_image_results_in_array_expr(ae, program, model),
        TermKind::Soac(op) => check_linear_image_results_in_soac(op, program, model),
        TermKind::Var(_)
        | TermKind::IntLit(_)
        | TermKind::FloatLit(_)
        | TermKind::BoolLit(_)
        | TermKind::UnitLit
        | TermKind::BinOp(_)
        | TermKind::UnOp(_)
        | TermKind::Extern(_) => None,
    }
}

fn check_linear_image_results_in_array_expr(
    ae: &ArrayExpr<Empty, Empty>,
    program: &Program<BuffersPinned>,
    model: &AnalysisState,
) -> Option<CompilerError> {
    match ae {
        ArrayExpr::Literal(terms) => {
            for term in terms {
                if let Some(err) = check_linear_image_results_in_term(
                    term,
                    program,
                    model,
                    LinearImageUseContext::Discarded,
                ) {
                    return Some(err);
                }
            }
            None
        }
        ArrayExpr::Range { start, len, step } => {
            check_linear_image_results_in_term(start, program, model, LinearImageUseContext::Discarded)
                .or_else(|| {
                    check_linear_image_results_in_term(
                        len,
                        program,
                        model,
                        LinearImageUseContext::Discarded,
                    )
                })
                .or_else(|| {
                    step.as_ref().and_then(|s| {
                        check_linear_image_results_in_term(
                            s,
                            program,
                            model,
                            LinearImageUseContext::Discarded,
                        )
                    })
                })
        }
        ArrayExpr::Zip(parts) => {
            for part in parts {
                if let Some(err) = check_linear_image_results_in_array_expr(part, program, model) {
                    return Some(err);
                }
            }
            None
        }
        ArrayExpr::Var(..) => None,
    }
}

fn check_linear_image_results_in_soac(
    op: &SoacOp<Empty, Empty>,
    program: &Program<BuffersPinned>,
    model: &AnalysisState,
) -> Option<CompilerError> {
    let check_body = |sb: &SoacBody<Empty, Empty>| {
        check_linear_image_results_in_term(&sb.lam.body, program, model, LinearImageUseContext::Used)
    };
    match op {
        SoacOp::Map { lam, inputs, .. } => {
            for input in inputs {
                if let Some(err) = check_linear_image_results_in_array_expr(input, program, model) {
                    return Some(err);
                }
            }
            check_body(lam)
        }
        SoacOp::Reduce { op, ne, input } => {
            check_linear_image_results_in_term(ne, program, model, LinearImageUseContext::Discarded)
                .or_else(|| check_linear_image_results_in_array_expr(input, program, model))
                .or_else(|| check_body(op))
        }
        SoacOp::Scan { op, ne, input, .. } => {
            check_linear_image_results_in_term(ne, program, model, LinearImageUseContext::Discarded)
                .or_else(|| check_linear_image_results_in_array_expr(input, program, model))
                .or_else(|| check_body(op))
        }
        SoacOp::Filter { pred, input, .. } => {
            check_linear_image_results_in_array_expr(input, program, model).or_else(|| check_body(pred))
        }
        SoacOp::Scatter { lam, inputs, .. } => {
            for input in inputs {
                if let Some(err) = check_linear_image_results_in_array_expr(input, program, model) {
                    return Some(err);
                }
            }
            check_body(lam)
        }
        SoacOp::ReduceByIndex {
            op,
            ne,
            indices,
            values,
            ..
        } => check_linear_image_results_in_term(ne, program, model, LinearImageUseContext::Discarded)
            .or_else(|| check_linear_image_results_in_array_expr(indices, program, model))
            .or_else(|| check_linear_image_results_in_array_expr(values, program, model))
            .or_else(|| check_body(op)),
    }
}

fn linear_image_owner_is_threaded(
    term: &Term<Empty, Empty>,
    owners: &LookupSet<OwnerId>,
    model: &AnalysisState,
) -> bool {
    if term_consumes_owner_on_all_paths(term, owners, model) {
        return true;
    }
    if term_observes_owner_on_all_paths(term, owners, model) {
        return true;
    }
    term_returns_owner_on_all_paths(term, owners, model)
}

fn owner_sets_intersect(a: &LookupSet<OwnerId>, b: &LookupSet<OwnerId>) -> bool {
    a.iter().any(|owner| b.contains(owner))
}

fn term_kills_owner(term: &Term<Empty, Empty>, owners: &LookupSet<OwnerId>, model: &AnalysisState) -> bool {
    model.kills.get(&term.id).is_some_and(|kills| owner_sets_intersect(kills, owners))
}

fn term_consumes_owner_on_all_paths(
    term: &Term<Empty, Empty>,
    owners: &LookupSet<OwnerId>,
    model: &AnalysisState,
) -> bool {
    if term_kills_owner(term, owners, model) {
        return true;
    }
    match &term.kind {
        TermKind::Let { rhs, body, .. } => {
            term_consumes_owner_on_all_paths(rhs, owners, model)
                || term_consumes_owner_on_all_paths(body, owners, model)
        }
        TermKind::If {
            then_branch,
            else_branch,
            ..
        } => {
            term_consumes_owner_on_all_paths(then_branch, owners, model)
                && term_consumes_owner_on_all_paths(else_branch, owners, model)
        }
        TermKind::Coerce { inner, .. } => term_consumes_owner_on_all_paths(inner, owners, model),
        _ => false,
    }
}

fn term_observes_owner_on_all_paths(
    term: &Term<Empty, Empty>,
    owners: &LookupSet<OwnerId>,
    model: &AnalysisState,
) -> bool {
    match &term.kind {
        TermKind::App { func, args } => {
            let observes_here = matches!(&func.kind, TermKind::Var(VarRef::Builtin { id, .. }) if *id == catalog().known().image_load)
                && args.first().is_some_and(|arg| owner_sets_intersect(&arg_aliases(arg, model), owners));
            observes_here || args.iter().any(|arg| term_observes_owner_on_all_paths(arg, owners, model))
        }
        TermKind::Let { rhs, body, .. } => {
            if term_consumes_owner_on_all_paths(rhs, owners, model) {
                false
            } else {
                term_observes_owner_on_all_paths(rhs, owners, model)
                    || term_observes_owner_on_all_paths(body, owners, model)
            }
        }
        TermKind::If {
            then_branch,
            else_branch,
            ..
        } => {
            term_observes_owner_on_all_paths(then_branch, owners, model)
                && term_observes_owner_on_all_paths(else_branch, owners, model)
        }
        TermKind::Coerce { inner, .. } => term_observes_owner_on_all_paths(inner, owners, model),
        _ => false,
    }
}

fn arg_aliases(term: &Term<Empty, Empty>, model: &AnalysisState) -> LookupSet<OwnerId> {
    match &term.kind {
        TermKind::Var(VarRef::Symbol(sym)) => model.aliases_of(*sym),
        TermKind::Coerce { inner, .. } => arg_aliases(inner, model),
        _ => LookupSet::new(),
    }
}

fn term_returns_owner_on_all_paths(
    term: &Term<Empty, Empty>,
    owners: &LookupSet<OwnerId>,
    model: &AnalysisState,
) -> bool {
    match &term.kind {
        TermKind::Var(VarRef::Symbol(sym)) => owner_sets_intersect(&model.aliases_of(*sym), owners),
        TermKind::Coerce { inner, .. } => term_returns_owner_on_all_paths(inner, owners, model),
        TermKind::Let { rhs, body, .. } => {
            if term_consumes_owner_on_all_paths(rhs, owners, model) {
                false
            } else {
                term_returns_owner_on_all_paths(body, owners, model)
            }
        }
        TermKind::If {
            then_branch,
            else_branch,
            ..
        } => {
            term_returns_owner_on_all_paths(then_branch, owners, model)
                && term_returns_owner_on_all_paths(else_branch, owners, model)
        }
        _ => false,
    }
}
