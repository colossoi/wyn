//! Shared construction utilities for canonical SOAC lambdas.
//!
//! Lambda representation stays in [`super::screma`]. This module owns the
//! repetitive EGIR mechanics needed by fusion and parallel lowering: emitting
//! calls, packing and unpacking multi-result values, and finalising generated
//! callable regions.

use polytype::Type;
use smallvec::{smallvec, SmallVec};

use crate::ast::{Span, TypeName};
use crate::egir::program::{fresh_region_name, RegionInterner, SemanticFunc};
use crate::egir::types::{EGraph, ENode, NodeId, PureOp, RegionId, SegBody, SkeletonTerminator};
use crate::flow::BlockId;

use super::screma;

pub(crate) fn named_parameters(types: &[Type<TypeName>], prefix: &str) -> Vec<(Type<TypeName>, String)> {
    types.iter().enumerate().map(|(index, ty)| (ty.clone(), format!("{prefix}_{index}"))).collect()
}

pub(crate) fn function_parameters(graph: &mut EGraph, params: &[(Type<TypeName>, String)]) -> Vec<NodeId> {
    params.iter().enumerate().map(|(index, (ty, _))| graph.add_func_param(index, ty.clone())).collect()
}
pub(crate) fn result_type(types: &[Type<TypeName>]) -> Type<TypeName> {
    match types {
        [ty] => ty.clone(),
        _ => Type::Constructed(TypeName::Tuple(types.len()), types.to_vec()),
    }
}

pub(crate) fn pack_results(graph: &mut EGraph, results: &[NodeId], types: &[Type<TypeName>]) -> NodeId {
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

pub(crate) fn unpack_results(graph: &mut EGraph, result: NodeId, types: &[Type<TypeName>]) -> Vec<NodeId> {
    match types {
        [_] => vec![result],
        _ => types
            .iter()
            .enumerate()
            .map(|(index, ty)| {
                graph.intern_pure(
                    PureOp::Project { index: index as u32 },
                    smallvec![result],
                    ty.clone(),
                    None,
                )
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
    callee: Option<&str>,
    arguments: Vec<NodeId>,
) -> Vec<NodeId> {
    if lambda.is_identity() {
        debug_assert_eq!(arguments.len(), lambda.result_types.len());
        return arguments;
    }
    let result = graph.intern_pure(
        PureOp::Call(callee.expect("region lambda has no resolved callee").to_owned()),
        SmallVec::from_vec(arguments),
        result_type(&lambda.result_types),
        None,
    );
    unpack_results(graph, result, &lambda.result_types)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn finish_function(
    mut graph: EGraph,
    return_block: BlockId,
    region: RegionId,
    name: String,
    span: Span,
    params: Vec<(Type<TypeName>, String)>,
    result_types: &[Type<TypeName>],
    results: &[NodeId],
) -> SemanticFunc {
    let result = pack_results(&mut graph, results, result_types);
    graph.skeleton.blocks[return_block].term = SkeletonTerminator::Return(Some(result));
    SemanticFunc::new(region, name, span, None, params, result_type(result_types), graph)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn finish_region_lambda(
    interner: &mut RegionInterner,
    scope: &str,
    label: &str,
    span: Span,
    graph: EGraph,
    return_block: BlockId,
    params: Vec<(Type<TypeName>, String)>,
    captures: Vec<NodeId>,
    parameter_types: Vec<Type<TypeName>>,
    result_types: Vec<Type<TypeName>>,
    results: Vec<NodeId>,
    fold_identity: bool,
) -> (screma::Lambda, Option<SemanticFunc>) {
    let is_identity = fold_identity
        && captures.is_empty()
        && params.len() == parameter_types.len()
        && result_types == parameter_types
        && results.iter().enumerate().all(|(index, result)| {
            matches!(
                graph.nodes.get(*result).map(|node| &node.kind),
                Some(ENode::FuncParam { index: parameter }) if *parameter == index
            )
        });
    if is_identity {
        return (screma::Lambda::identity(parameter_types), None);
    }

    let name = fresh_region_name(interner, &format!("{scope}_{label}"));
    let region = interner.intern(&name);
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
        screma::Lambda::region(SegBody { region, captures }, parameter_types, result_types),
        Some(function),
    )
}
