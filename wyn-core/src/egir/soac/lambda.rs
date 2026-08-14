//! Shared construction utilities for canonical SOAC lambdas.
//!
//! Lambda representation stays in [`super::screma`]. This module owns the
//! repetitive EGIR mechanics needed by fusion and parallel lowering: emitting
//! calls, packing and unpacking multi-result values, and finalising generated
//! callable regions.

use polytype::Type;
use smallvec::smallvec;

use crate::ast::{Span, TypeName};
use crate::egir::program::{fresh_region_name, ProgramIdentities, SemanticFunc, SemanticResourceRef};
use crate::egir::types::{
    by_value_function_result, callable_parameter, CallEffects, EGraph, FuncParam, OperandRef,
    ParameterId, PureOp, RegionId, SegBody, SkeletonTerminator, ValueId, ValueKind, WynLanguage,
};
use crate::flow::BlockId;

use super::screma;

pub(crate) fn named_parameters(
    types: &[Type<TypeName>],
    prefix: &str,
) -> Vec<FuncParam<SemanticResourceRef, Type<TypeName>>> {
    types
        .iter()
        .enumerate()
        .map(|(index, ty)| {
            callable_parameter::<SemanticResourceRef, WynLanguage>(
                format!("{prefix}_{index}"),
                ty.clone(),
            )
        })
        .collect()
}

pub(crate) fn function_parameters(
    graph: &mut EGraph,
    params: &[FuncParam<SemanticResourceRef, Type<TypeName>>],
) -> Vec<OperandRef> {
    params
        .iter()
        .enumerate()
        .map(|(index, parameter)| {
            graph.add_parameter(ParameterId::new(index), parameter.representation())
        })
        .collect()
}
pub(crate) fn result_type(types: &[Type<TypeName>]) -> Type<TypeName> {
    match types {
        [ty] => ty.clone(),
        _ => Type::Constructed(TypeName::Tuple(types.len()), types.to_vec()),
    }
}

pub(crate) fn pack_results(graph: &mut EGraph, results: &[ValueId], types: &[Type<TypeName>]) -> ValueId {
    debug_assert_eq!(results.len(), types.len());
    match results {
        [result] => *result,
        _ => graph.intern_pure(
            PureOp::Tuple(results.len()),
            results.iter().copied().collect(),
            result_type(types),
            None,
        ),
    }
}

pub(crate) fn unpack_results(
    graph: &mut EGraph,
    result: ValueId,
    types: &[Type<TypeName>],
) -> Vec<ValueId> {
    match types {
        [_] => vec![result],
        _ => types
            .iter()
            .enumerate()
            .map(|(index, ty)| {
                let op = PureOp::Project { index: index as u32 };
                let operands = smallvec![result];
                graph
                    .try_algebraic_fold(&op, &operands, ty)
                    .unwrap_or_else(|| graph.intern_pure(op, operands, ty.clone(), None))
            })
            .collect(),
    }
}

/// Emit a lambda application whose region name has already been resolved.
///
/// Identity lambdas do not have a callable region and simply return their
/// arguments. Region-lambda callers must append captures to `arguments`.
pub(crate) fn emit_call(
    graph: &mut EGraph,
    lambda: &screma::Lambda,
    callee: Option<&SemanticFunc>,
    arguments: Vec<OperandRef>,
) -> Vec<ValueId> {
    if lambda.is_identity() {
        debug_assert_eq!(arguments.len(), lambda.result_types.len());
        return arguments
            .into_iter()
            .map(|argument| argument.value().expect("identity lambda arguments are values or views"))
            .collect();
    }
    let function = lambda.seg_body().expect("region lambda has no callable body").region;
    let callee = callee.expect("region lambda call requires its canonical function boundary");
    assert_eq!(callee.region, function, "lambda and function boundary disagree on region identity");
    assert_eq!(callee.effects(), CallEffects::Pure, "SOAC lambda call must be pure");
    let (_, result) = graph
        .add_call(
            function,
            callee.params(),
            callee.result(),
            arguments,
            callee.effects(),
            None,
        )
        .expect("lambda call must match its canonical function boundary");
    result.values()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn finish_function(
    mut graph: EGraph,
    return_block: BlockId,
    region: RegionId,
    name: String,
    span: Span,
    params: Vec<FuncParam<SemanticResourceRef, Type<TypeName>>>,
    result_types: &[Type<TypeName>],
    results: &[ValueId],
) -> SemanticFunc {
    let result = pack_results(&mut graph, results, result_types);
    let result_abi = by_value_function_result::<WynLanguage>(result_type(result_types));
    let result = crate::egir::graph_ops::bind_by_value_result(&mut graph, &result_abi, result);
    graph.skeleton.blocks[return_block].term = SkeletonTerminator::Return(Some(result));
    SemanticFunc::new(
        region,
        name,
        span,
        None,
        params,
        result_abi,
        CallEffects::Pure,
        graph,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn finish_region_lambda(
    identities: &mut ProgramIdentities,
    scope: &str,
    label: &str,
    span: Span,
    graph: EGraph,
    return_block: BlockId,
    params: Vec<FuncParam<SemanticResourceRef, Type<TypeName>>>,
    captures: Vec<OperandRef>,
    parameter_types: Vec<Type<TypeName>>,
    result_types: Vec<Type<TypeName>>,
    results: Vec<ValueId>,
    fold_identity: bool,
) -> (screma::Lambda, Option<SemanticFunc>) {
    let is_identity = fold_identity
        && captures.is_empty()
        && params.len() == parameter_types.len()
        && result_types == parameter_types
        && results.iter().enumerate().all(|(index, result)| {
            matches!(
                graph.nodes.get(*result).map(|node| &node.kind),
                Some(ValueKind::FuncParam { parameter }) if parameter.index() == index
            )
        });
    if is_identity {
        return (screma::Lambda::identity(parameter_types), None);
    }

    let name = fresh_region_name(identities, &format!("{scope}_{label}"));
    let region = identities.alloc_function(name.clone());
    let function = finish_function(
        graph,
        return_block,
        region,
        name,
        span,
        params,
        &result_types,
        &results,
    );
    (
        screma::Lambda::region(SegBody::new(region, captures), parameter_types, result_types),
        Some(function),
    )
}
