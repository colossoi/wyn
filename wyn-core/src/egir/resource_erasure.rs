//! Erase compile-time-only GPU resource handles before SSA construction.
//!
//! Buffer monomorphization makes a `StorageTexture` parameter's binding part
//! of its type. Binding-qualified image operations therefore need no runtime
//! image value. This pass removes those zero-footprint parameters and the
//! matching call operands after all SegOps have expanded, when the complete
//! call graph is concrete.

/// Physical EGIR with compile-time-only resource handles erased.
#[derive(Debug, Clone, Copy)]
pub enum ResourcesErasedTag {}
pub type ResourcesErased = egir::program::PhysicalProgram<ResourcesErasedTag>;

#[cfg(test)]
#[path = "resource_erasure_tests.rs"]
mod resource_erasure_tests;

use crate::ast::TypeName;
use crate::egir;
use crate::egir::from_tlc::ConvertError;
use crate::egir::program::{Func, Program};
use crate::egir::types::{EGraph, Family, OperandType, Physical, ValueKind};
use crate::FunctionId;
use crate::{LookupMap, LookupSet};
use polytype::Type;

pub fn erase_resources(
    program: super::skel_opt::SkeletonOptimized,
) -> Result<ResourcesErased, ConvertError> {
    let erasures: LookupMap<FunctionId, Vec<bool>> = program
        .functions
        .iter()
        .map(|function| {
            (
                function.region,
                function
                    .params()
                    .iter()
                    .map(|parameter| {
                        matches!(parameter.representation(), OperandType::Value(ty) if is_storage_image(ty))
                    })
                    .collect(),
            )
        })
        .collect();

    let Program {
        functions,
        externs,
        entry_points,
        constants,
        data,
        global_context,
        state: _,
    } = program;
    let functions = functions
        .into_iter()
        .map(|function| erase_function_resources(function, &erasures))
        .collect::<Result<_, ConvertError>>()?;
    let entry_points = entry_points
        .into_iter()
        .map(|entry| entry.try_map_graph(|graph| rewrite_graph(graph, &erasures)))
        .collect::<Result<_, ConvertError>>()?;
    let constants = constants
        .into_iter()
        .map(|constant| constant.try_map_graph(|graph| rewrite_graph(graph, &erasures)))
        .collect::<Result<_, ConvertError>>()?;
    let program = Program::from_parts(functions, externs, entry_points, constants, data, global_context);
    debug_assert!(
        program.validate_kernel_bodies().is_ok(),
        "resource erasure broke physical kernel/body ownership"
    );
    Ok(program)
}

fn is_storage_image(ty: &Type<TypeName>) -> bool {
    matches!(ty, Type::Constructed(TypeName::StorageTexture, _))
}

fn rewrite_graph<P: Family>(
    mut graph: EGraph<P>,
    erasures: &LookupMap<FunctionId, Vec<bool>>,
) -> Result<EGraph<P>, ConvertError> {
    let calls = graph.side_effect_index().calls().map(|(call, _)| call).collect::<Vec<_>>();
    for site in calls {
        let call = graph.calls.get_mut(site).expect("explicit call has a boundary record");
        if let Some(erase) = erasures.get(&call.callee()) {
            let retain = erase.iter().map(|erase| !erase).collect::<Vec<_>>();
            call.retain_arguments(&retain).map_err(ConvertError::Internal)?;
        }
    }
    Ok(graph)
}

fn erase_function_resources(
    function: Func<Physical>,
    erasures: &LookupMap<FunctionId, Vec<bool>>,
) -> Result<Func<Physical>, ConvertError> {
    let egir::ir::Func {
        region,
        name,
        span,
        linkage_name,
        params,
        result,
        effects,
        graph,
    } = function;
    let mut graph = rewrite_graph(graph, erasures)?;
    let erase: Vec<bool> = params
        .iter()
        .map(|parameter| {
            matches!(parameter.representation(), OperandType::Value(ty) if is_storage_image(ty))
        })
        .collect();
    if !erase.iter().any(|erase| *erase) {
        return Ok(Func::<Physical>::new(
            region,
            name,
            span,
            linkage_name,
            params,
            result,
            effects,
            graph,
        ));
    }

    let erased_parameters = params
        .ids()
        .zip(&erase)
        .filter_map(|(parameter, erase)| (*erase).then_some(parameter))
        .collect::<LookupSet<_>>();

    let mut erased_nodes = Vec::new();
    for (node_id, node) in &mut graph.nodes {
        let ValueKind::FuncParam { parameter } = &mut node.kind else {
            continue;
        };
        if erased_parameters.contains(parameter) {
            erased_nodes.push(node_id);
        }
    }
    for place in graph.places.values_mut() {
        if let egir::ir::PlaceOp::Parameter { parameter } = place.op() {
            if erased_parameters.contains(parameter) {
                return Err(ConvertError::Internal(format!(
                    "addressable parameter {parameter:?} in `{name}` cannot be erased",
                )));
            }
        }
    }
    let result = result.try_map(
        &mut |ty| Ok::<_, ConvertError>(ty),
        &mut |slot| Ok::<_, ConvertError>(slot),
        &mut |parameter| {
            (!erased_parameters.contains(&parameter)).then_some(parameter).ok_or_else(|| {
                ConvertError::Internal(format!(
                    "result destination parameter {parameter:?} in `{name}` cannot be erased",
                ))
            })
        },
    )?;

    // A remaining use means a new storage-image value operation was added
    // without being reified to a binding-qualified operation. Fail here rather
    // than letting a backend recreate an opaque runtime handle.
    let live = live_nodes(&graph);
    for erased in erased_nodes {
        if live.contains(&erased) {
            return Err(ConvertError::Internal(format!(
                "storage-image parameter in `{}` still has a runtime EGIR use after resource erasure",
                name
            )));
        }
        // The declaration is gone, so the dead graph binding must go with it.
        graph.remove_func_param(erased);
    }

    let mut params = params;
    let retain = erase.into_iter().map(|erase| !erase).collect::<Vec<_>>();
    params.retain_abi_positions(&retain);
    Ok(Func::<Physical>::new(
        region,
        name,
        span,
        linkage_name,
        params,
        result,
        effects,
        graph,
    ))
}

fn live_nodes<P: Family>(graph: &EGraph<P>) -> LookupSet<egir::types::ValueId> {
    let mut roots = Vec::new();
    for (_, block) in &graph.skeleton.blocks {
        for effect in &block.side_effects {
            roots.extend(graph.effect_boundary_value_dependencies(effect));
        }
        roots.extend(block.term.referenced_nodes());
    }

    wyn_graph::reachable_set(roots, wyn_graph::WalkOrder::DepthFirst, |node, out| {
        out.extend(graph.nodes[node].children());
        match graph.nodes[node].kind() {
            ValueKind::CallResult { call, .. } => out.extend(graph.call_value_dependencies(*call)),
            ValueKind::PlaceLength { place } | ValueKind::PlaceView { place } => {
                out.extend(graph.place_value_dependencies(*place))
            }
            _ => {}
        }
    })
}
