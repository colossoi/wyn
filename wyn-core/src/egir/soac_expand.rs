//! Expand physical `SideEffectKind::Soac(SoacEffect(_, ...))` skeleton side-effects
//! into explicit loop subgraphs with pure ops in the sea and block params
//! carrying accumulators.
//!
//! Consumes target-planned physical EGIR before graph cleanup and SSA
//! elaboration. Every physical variant must be handled here; any SOAC left in
//! the skeleton after this stage is a bug.

/// Physical EGIR whose SOAC effects have been expanded into explicit CFGs.
#[derive(Debug, Clone, Copy)]
pub enum SoacsExpandedTag {}
pub type SoacsExpanded = super::program::PhysicalProgram<SoacsExpandedTag>;

use crate::flow::BlockId;
use wyn_base::IdSource;

use polytype::Type;

use super::graph_ops::{bind_by_value_result, load_result_value};
use super::program::Func;
use super::soac::screma;
use crate::ast::TypeName;
use crate::types::{is_array_variant_view, is_virtual_array};

use super::types::{
    as_soa_tuple, by_value_function_result, EGraph, EffectToken, OperandRef, Physical, ResultBinding,
    ResultDestination, SideEffectKind, Soac, SoacEffect, ValueId, WynLanguage,
};
use crate::{FunctionId, LookupMap};

type CallableMap = LookupMap<FunctionId, Func<Physical>>;

mod array_io;
mod filter_lowering;
mod flow_normalize;
mod hist_lowering;
mod loop_builder;
mod screma_lowering;

use filter_lowering::expand_filter;
use flow_normalize::normalize_place_backed_flow;
use hist_lowering::expand_hist;
use screma_lowering::expand_screma;

/// Expand every graph-bearing body and rebuild the program at the
/// post-expansion checkpoint.
pub fn expand_soacs(program: super::parallelize::Planned) -> Result<SoacsExpanded, String> {
    let mut program = program;
    let callables = program
        .functions
        .iter()
        .map(|function| (function.region, function.clone()))
        .collect::<CallableMap>();
    program = program.try_map_graphs_with_state(|_, graph, _, context| {
        run_one_body(graph, &callables, &mut context.effect_ids)
    })?;
    Ok(program.retag_physical())
}

/// Expand every physical SOAC in the skeleton.
pub fn run_one_body(
    mut graph: EGraph<Physical>,
    callables: &CallableMap,
    effect_ids: &mut IdSource<EffectToken>,
) -> Result<EGraph<Physical>, String> {
    // Re-scan after every expansion because splitting a block moves the
    // remaining suffix. Selecting the first operation preserves producer to
    // consumer order, so a resolved destination is visible to later updates.
    while let Some((bid, idx)) = graph.skeleton.blocks.iter().find_map(|(bid, block)| {
        block.side_effects.iter().position(|effect| is_handleable_soac(&effect.kind)).map(|idx| (bid, idx))
    }) {
        expand_one(&mut graph, bid, idx, effect_ids, callables)?;
    }
    if let Some((block, effect)) = graph.skeleton.blocks.iter().find_map(|(block, contents)| {
        contents
            .side_effects
            .iter()
            .find(|effect| matches!(effect.kind, SideEffectKind::Soac(_)))
            .map(|effect| (block, effect))
    }) {
        return Err(format!(
            "SOAC expansion left an unsupported physical operation in {block:?}: {:?}",
            effect.kind
        ));
    }
    normalize_place_backed_flow(&mut graph, effect_ids)?;
    Ok(graph)
}

/// Does this SOAC kind have a TLC→EGIR expansion implemented here?
fn is_handleable_soac(kind: &SideEffectKind<Physical>) -> bool {
    let SideEffectKind::Soac(SoacEffect(_, soac)) = kind else {
        return false;
    };
    match soac {
        Soac::Screma(op) if op.is_serial() => {
            op.inputs.iter().all(|input| is_plain_array_source(&input.array))
        }
        Soac::Filter(op) => {
            !op.body.inputs.is_empty()
                && op.body.inputs.iter().all(|input| is_plain_array_source(&input.array))
        }
        // Hist reads all input arrays per element; loop length comes from the
        // first input, but every input must support the read path.
        Soac::Hist(op) => {
            !op.inputs.is_empty() && op.inputs.iter().all(|input| is_plain_array_source(&input.array))
        }
        Soac::Screma(op) if matches!(op.state, screma::PhysicalState::Segmented(_)) && op.is_map() => {
            op.form.post.is_identity() && op.inputs.iter().all(|input| is_plain_array_source(&input.array))
        }
        // Scan and reduction recipes must lower the complete canonical
        // contract before this pass.
        Soac::Screma(_) => false,
    }
}

/// Element type to read from an input array: the buffer's own element type
/// (uniqueness stripped). For a map-fused scan/reduce the raw input element
/// differs from the accumulator element carried by `input_elem_type` (e.g.
/// `scan(+, 0, map(|h:vec4f32| ..:i32, bh))` reads `vec4f32` but accumulates
/// `i32`), so the read must follow the array type, not the accumulator.
/// Falls back to `acc_elem` when the array type has no extractable element
/// (e.g. a SoA-tuple source, handled separately).

/// Input-array shape handled today: rank-1 composite/view/virtual
/// arrays, or SoA tuples `([n]A, [n]B, ...)` (produced by `tlc::soa`)
/// whose components are themselves handleable.
fn is_plain_array_source(arr_ty: &Type<TypeName>) -> bool {
    // Rank-1 invariant: [elem, variant, size, region] (4 args).
    if matches!(arr_ty, Type::Constructed(TypeName::Array, args) if args.len() == 4) {
        return true;
    }
    if let Some(components) = as_soa_tuple(arr_ty) {
        return components.iter().all(is_plain_array_source);
    }
    false
}

/// If `ty` is a SoA tuple (tuple where every component is an Array or itself
/// a SoA tuple), return the component types. Mirrors the helper in
/// `ssa::soa_helpers`.

/// Element type of a SoA tuple: `([n]A, [n]B)` → `(A, B)`. Nested SoA tuples
/// recurse into their own element types.
fn is_view_source(arr_ty: &Type<TypeName>) -> bool {
    matches!(
        arr_ty,
        Type::Constructed(TypeName::Array, args)
            // args = [elem, variant, size, region]
            if args.len() == 4 && is_array_variant_view(&args[1])
    )
}

fn is_virtual_source(arr_ty: &Type<TypeName>) -> bool {
    is_virtual_array(arr_ty)
}

fn value_binding(
    graph: &mut EGraph<Physical>,
    ty: &Type<TypeName>,
    value: ValueId,
) -> ResultBinding<Type<TypeName>> {
    let abi = by_value_function_result::<WynLanguage>(ty.clone());
    bind_by_value_result(graph, &abi, value)
}

fn load_result_arguments(
    graph: &mut EGraph<Physical>,
    block: BlockId,
    results: &[ResultBinding<Type<TypeName>>],
    next_effect: &mut IdSource<EffectToken>,
) -> Vec<ValueId> {
    results.iter().map(|result| load_result_value(graph, block, result, next_effect)).collect()
}

fn result_is_addressable(graph: &EGraph<Physical>, result: &ResultBinding<Type<TypeName>>) -> bool {
    result.destination_count()
        == result
            .destination_leaves()
            .iter()
            .filter(|leaf| {
                leaf.single_destination().is_some_and(|(_, destination)| match destination {
                    ResultDestination::Place(_) => true,
                    ResultDestination::ReturnValue(value) => {
                        matches!(
                            graph.operand_ref(graph.canonical_value(*value)),
                            OperandRef::View(_)
                        )
                    }
                })
            })
            .count()
}

fn expand_one(
    graph: &mut EGraph<Physical>,
    bid: BlockId,
    idx: usize,
    next_effect: &mut IdSource<EffectToken>,
    callables: &CallableMap,
) -> Result<(), String> {
    let effect = graph.skeleton.blocks[bid]
        .side_effects
        .get(idx)
        .ok_or_else(|| format!("missing selected SOAC effect {idx} in {bid:?}"))?;
    let SideEffectKind::Soac(SoacEffect(_, soac)) = &effect.kind else {
        return Err("SOAC expansion target changed after selection".into());
    };
    match soac {
        Soac::Screma(_) => expand_screma(graph, bid, idx, next_effect, callables),
        Soac::Filter(_) => expand_filter(graph, bid, idx, next_effect, callables),
        Soac::Hist(_) => expand_hist(graph, bid, idx, next_effect, callables),
    }
}

#[cfg(test)]
#[path = "soac_expand_tests.rs"]
mod soac_expand_tests;
