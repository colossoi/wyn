use polytype::Context;

use crate::ast::{Type, TypeName};

/// Result of overload resolution: the index of the candidate that
/// unified successfully and the inferred return type.
#[derive(Debug)]
pub struct ResolvedOverload {
    pub winner_index: usize,
    pub return_type: Type,
}

/// No candidate function type unified with the given argument types.
#[derive(Debug)]
pub struct NoMatchingOverload;

/// Try to unify a single candidate function type with a sequence of
/// argument types, returning the resulting return type on success.
/// Mutates `ctx` (allocating fresh variables) regardless of outcome —
/// callers that intend to try multiple candidates should checkpoint
/// before each attempt and roll back on failure.
pub fn try_unify_with_args(
    func_type: &Type,
    arg_types: &[Type],
    ctx: &mut Context<TypeName>,
) -> Option<Type> {
    let mut current_type = func_type.clone();
    for arg_type in arg_types {
        let param_ty = ctx.new_variable();
        let rest_ty = ctx.new_variable();
        let expected_arrow = Type::arrow(param_ty.clone(), rest_ty.clone());
        if ctx.unify(&current_type, &expected_arrow).is_err() {
            return None;
        }
        let param_ty = param_ty.apply(ctx);
        if ctx.unify(&param_ty, arg_type).is_err() {
            return None;
        }
        current_type = rest_ty.apply(ctx);
    }
    Some(current_type)
}

/// Resolve an overloaded callee against argument types via backtracking
/// unification. Tries each candidate in order; the first that unifies
/// wins. Callers do post-resolution checks (e.g. partial-application);
/// since overloads of a single name share arity, partial-application
/// either rejects all or none.
pub fn resolve_overload(
    candidates: &[Type],
    arg_types: &[Type],
    ctx: &mut Context<TypeName>,
) -> Result<ResolvedOverload, NoMatchingOverload> {
    for (i, cand) in candidates.iter().enumerate() {
        let checkpoint = ctx.len();
        if let Some(return_type) = try_unify_with_args(cand, arg_types, ctx) {
            return Ok(ResolvedOverload {
                winner_index: i,
                return_type,
            });
        }
        ctx.rollback(checkpoint);
    }
    Err(NoMatchingOverload)
}
