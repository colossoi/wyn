//! Shared construction utilities for canonical SOAC lambdas.
//!
//! Lambda representation stays in [`super::screma`]. This module owns the
//! repetitive EGIR mechanics needed by fusion and parallel lowering: emitting
//! calls, packing and unpacking multi-result values, and finalising generated
//! callable regions.

use crate::egir;
use polytype::Type;
use smallvec::smallvec;

use crate::ast::{Span, TypeName};
use crate::egir::program::{fresh_region_name, Func, ProgramIdentities, SemanticResourceRef};
use crate::egir::types::{
    by_value_function_result, callable_parameter, CallEffects, EGraph, FuncParam, OperandRef, ParameterId,
    PureOp, ResultBinding, SegBody, Semantic, SkeletonTerminator, ValueId, ValueKind, WynLanguage,
};
use crate::flow::BlockId;
use crate::FunctionId;

use super::screma;

pub(crate) fn named_parameters(
    types: &[Type<TypeName>],
    prefix: &str,
) -> Vec<FuncParam<SemanticResourceRef, Type<TypeName>>> {
    types
        .iter()
        .enumerate()
        .map(|(index, ty)| {
            callable_parameter::<SemanticResourceRef, WynLanguage>(format!("{prefix}_{index}"), ty.clone())
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
        .map(|(index, parameter)| graph.add_parameter(ParameterId::new(index), parameter.representation()))
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
        _ => {
            let ty = result_type(types);
            let binding = ResultBinding::product(
                ty.clone(),
                results.iter().zip(types).map(|(&value, ty)| {
                    egir::graph_ops::bind_physical_result_value(graph, ty.clone(), value)
                }),
            );
            let value = graph.intern_pure(
                PureOp::Tuple(results.len()),
                results.iter().copied().collect(),
                ty,
                None,
            );
            egir::graph_ops::register_result_origin_tree(graph, value, &binding);
            value
        }
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

pub(crate) fn logical_result_fields(
    result: &ResultBinding<Type<TypeName>>,
    result_types: &[Type<TypeName>],
) -> Vec<ResultBinding<Type<TypeName>>> {
    match result_types {
        [] => Vec::new(),
        [_] => vec![result.clone()],
        _ => {
            let fields = result.top_level_fields();
            assert_eq!(fields.len(), result_types.len());
            fields
        }
    }
}

pub(crate) fn result_argument_values<P: egir::ir::Family>(
    graph: &mut EGraph<P>,
    results: &[ResultBinding<Type<TypeName>>],
) -> Vec<ValueId> {
    results
        .iter()
        .map(|result| {
            egir::graph_ops::result_argument_value(graph, result)
                .expect("lambda result must have an argument representation")
        })
        .collect()
}

/// Emit a lambda application whose region name has already been resolved.
///
/// Identity lambdas do not have a callable region and simply return their
/// arguments. Region-lambda callers must append captures to `arguments`.
pub(crate) fn emit_call(
    graph: &mut EGraph,
    block: BlockId,
    lambda: &screma::Lambda,
    callee: Option<&Func<Semantic>>,
    arguments: Vec<OperandRef>,
) -> Vec<ResultBinding<Type<TypeName>>> {
    if lambda.is_identity() {
        debug_assert_eq!(arguments.len(), lambda.result_types.len());
        return arguments
            .into_iter()
            .zip(&lambda.result_types)
            .map(|(argument, ty)| {
                let value = argument.value().expect("identity lambda arguments are values or views");
                let abi = by_value_function_result::<WynLanguage>(ty.clone());
                egir::graph_ops::bind_by_value_result(graph, &abi, value)
            })
            .collect();
    }
    let function = lambda.seg_body().expect("region lambda has no callable body").region;
    let callee = callee.expect("region lambda call requires its canonical function boundary");
    assert_eq!(
        callee.region, function,
        "lambda and function boundary disagree on region identity"
    );
    let (_, result) = graph
        .emit_call(
            block,
            function,
            callee.params(),
            callee.result(),
            arguments,
            callee.effects(),
            None,
            None,
        )
        .expect("lambda call must match its canonical function boundary");
    logical_result_fields(&result, &lambda.result_types)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn finish_function(
    mut graph: EGraph,
    return_block: BlockId,
    region: FunctionId,
    name: String,
    span: Span,
    params: Vec<FuncParam<SemanticResourceRef, Type<TypeName>>>,
    result_types: &[Type<TypeName>],
    results: &[ValueId],
) -> Func {
    let result = pack_results(&mut graph, results, result_types);
    let result_abi = by_value_function_result::<WynLanguage>(result_type(result_types));
    let result = egir::graph_ops::bind_by_value_result(&mut graph, &result_abi, result);
    graph.skeleton.blocks[return_block].term = SkeletonTerminator::Return(Some(result));
    let effects = if graph.has_ordered_effects() { CallEffects::General } else { CallEffects::Pure };
    Func::<Semantic>::new(region, name, span, None, params, result_abi, effects, graph)
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
) -> (screma::Lambda, Option<Func<Semantic>>) {
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
