//! Adapt canonical SOAC lambdas to the physical callable ABI.

use super::{logical_result_fields, Lambda};
use crate::ast::TypeName;
use crate::egir::graph_ops::bind_by_value_result;
use crate::egir::physical_call_abi::emit_call;
use crate::egir::program::Func;
use crate::egir::types::{
    by_value_function_result, EGraph, EffectToken, Physical, ResultBinding, ValueId, WynLanguage,
};
use crate::flow::BlockId;
use crate::{types, FunctionId, LookupMap};
use polytype::Type;
use wyn_base::IdSource;

pub(crate) type PhysicalCallables = LookupMap<FunctionId, Func<Physical>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MappedCallMode {
    DirectDestinationPassing,
    StructuredStore,
}

fn mapped_call_mode(
    callee: &Func<Physical>,
    lambda: &Lambda,
    destinations: &[ResultBinding<Type<TypeName>>],
) -> Result<MappedCallMode, String> {
    let results = match lambda.result_types.as_slice() {
        [] => Vec::new(),
        [_] => vec![callee.result().clone()],
        _ => callee.result().top_level_fields(),
    };
    if results.len() != destinations.len() {
        return Err(format!(
            "mapped lambda has {} logical results but {} destinations",
            results.len(),
            destinations.len()
        ));
    }

    let mut has_structured_store = false;
    for (index, (result, destination)) in results.iter().zip(destinations).enumerate() {
        let result_leaves = result.destination_leaves();
        let destination_leaves = destination.destination_leaves();
        let direct = result_leaves.len() == destination_leaves.len()
            && result_leaves.iter().zip(&destination_leaves).all(|(result_leaf, destination_leaf)| {
                types::array_elem(destination_leaf.ty()) == Some(result_leaf.ty())
            });
        if direct {
            continue;
        }

        let structured = result.is_product()
            && destination.single_destination().and_then(|(array_ty, _)| types::array_elem(array_ty))
                == Some(result.ty());
        if structured {
            has_structured_store = true;
            continue;
        }

        return Err(format!(
            "mapped lambda result {index} of type {:?} does not match destination type {:?}",
            result.ty(),
            destination.ty()
        ));
    }

    Ok(if has_structured_store {
        MappedCallMode::StructuredStore
    } else {
        MappedCallMode::DirectDestinationPassing
    })
}

/// Resolve and invoke any SOAC family's lambda, appending its typed captures.
///
/// Identity applications return their arguments without emitting a call.
/// Mapped results use destination passing where their shape permits it; the
/// caller stores the returned bindings for the remaining result shapes.
pub(crate) fn emit_physical_call(
    graph: &mut EGraph<Physical>,
    block: BlockId,
    callables: &PhysicalCallables,
    lambda: &Lambda,
    arguments: Vec<ValueId>,
    mapped_destinations: Option<(&[ResultBinding<Type<TypeName>>], ValueId)>,
    next_effect: &mut IdSource<EffectToken>,
) -> Result<Vec<ResultBinding<Type<TypeName>>>, String> {
    lambda.validate("SOAC")?;
    if arguments.len() != lambda.parameter_types.len() {
        return Err(format!(
            "SOAC lambda requires {} arguments, found {}",
            lambda.parameter_types.len(),
            arguments.len()
        ));
    }
    if lambda.is_identity() {
        return Ok(arguments
            .into_iter()
            .zip(&lambda.result_types)
            .map(|(argument, ty)| {
                let abi = by_value_function_result::<WynLanguage>(ty.clone());
                bind_by_value_result(graph, &abi, argument)
            })
            .collect());
    }
    let body = lambda.seg_body().expect("non-identity SOAC lambda has a region");
    let callee = callables
        .get(&body.region)
        .ok_or_else(|| format!("SOAC lambda callable boundary {:?} is missing", body.region))?;
    let mut operands =
        arguments.into_iter().map(|argument| graph.operand_ref(argument)).collect::<Vec<_>>();
    operands.extend(body.captures.iter().copied());
    let result = match mapped_destinations {
        None => emit_call(graph, block, callee, operands, None, next_effect),
        Some((destinations, lane)) => match mapped_call_mode(callee, lambda, destinations)? {
            MappedCallMode::DirectDestinationPassing => emit_call(
                graph,
                block,
                callee,
                operands,
                Some((destinations, lane)),
                next_effect,
            ),
            MappedCallMode::StructuredStore => emit_call(graph, block, callee, operands, None, next_effect),
        },
    }?;
    Ok(logical_result_fields(&result, &lambda.result_types))
}
