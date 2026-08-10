//! Whole-graph output realization.
//!
//! After `from_tlc`, every entry has its writes laid out as one of:
//!
//!   * **Compute (slot-routed)**: each declared output has explicit
//!     `SlotSource`s derived from the original TLC tail during conversion.
//!     Sources are `(block, value)` pairs, one per CFG path that produces the
//!     slot. A simple body produces one source; branches can produce several.
//!   * **Graphics**: the entry's body terminates in `Return(Some(value))`
//!     where `value` is a scalar/vector/fixed aggregate (or a `Tuple`
//!     of them across multiple location-decorated outputs).
//!
//! This pass materialises both shapes into side-effect stores. It needs
//! the whole graph to do its job correctly: sibling `Index` consumers
//! of a slot's SOAC source can be produced anywhere in the body and
//! must all be rewritten to load from the slot's view, which requires
//! the consumer-finding walk over the completed graph. That's why this
//! is a separate phase from `from_tlc`, not a part of it.
//!
//! This pass establishes concrete output writers. The later residency planner
//! materializes any runtime composite arrays exposed by non-output consumers;
//! its verifier checks the completed representation boundary.
//!
//! Runs after `from_tlc::run`, before segmentation: the SOAC→OutputView
//! rewrite must precede SOAC wrapping/expansion.

/// EGIR whose entry outputs own their realized writer routes.
#[derive(Debug, Clone, Copy)]
pub enum OutputsRealizedTag {}
pub type OutputsRealized = super::program::Program<
    OutputsRealizedTag,
    super::ir::ProgramFamily<
        super::types::Raw,
        super::program::SemanticResourceDecl,
        super::ir::RealizedOutputRoute,
        super::program::CoreProgramData,
    >,
    super::program::RewriteGlobal,
>;

use crate::flow::{BlockId, ExecutionModel};
use crate::types::TypeExt;
use polytype::Type;
#[allow(unused_imports)]
use ExecutionModel as _;

use super::from_tlc::ConvertError;
use super::ir::{RealizedOutputRoute, UnrealizedOutputRoute};
use super::program::{
    host_resource_map, Entry, LogicalResourceArena, OutputWriter, Program, RawEntry, SemanticResourceRef,
    SlotSource,
};
use super::types::{
    EGraph, ENode, EffectToken, NodeId, PureOp, Raw, SideEffectIndex, SideEffectKind, SkeletonTerminator,
    Soac, SoacEffect,
};
use crate::ResourceId;
use std::collections::HashMap;

pub mod dispatch;
pub mod reconcile;
pub mod verify;

fn bucket_array_output(
    graph: &EGraph<Raw>,
    effect_index: &SideEffectIndex,
    source: NodeId,
) -> Option<(SemanticResourceRef, NodeId, Option<NodeId>)> {
    let ENode::Pure {
        op: PureOp::Project { index },
        operands,
    } = &graph.nodes[source].kind
    else {
        return None;
    };
    if *index > 1 {
        return None;
    }
    let [result] = operands.as_slice() else {
        return None;
    };
    let effect = effect_index.effect(graph, *result)?;
    let SideEffectKind::Soac(SoacEffect(_, Soac::Hist(op))) = &effect.kind else {
        return None;
    };
    let super::soac::hist::UpdatePolicy::BucketInsert { counts, .. } = op.body.update_policy else {
        return None;
    };
    if *index == 0 {
        let dest = super::graph_ops::extract_storage_view_source(graph, effect.operand_nodes[0])?;
        Some((dest, *result, Some(effect.operand_nodes[0])))
    } else {
        Some((counts, *result, None))
    }
}

/// Realize every entry's outputs into side-effect stores. After this
/// pass, `verify::check` confirms the invariant.
pub fn realize_outputs(program: super::from_tlc::Converted) -> Result<OutputsRealized, ConvertError> {
    let Program {
        functions,
        externs,
        entry_points,
        constants,
        mut data,
        mut global_context,
        state: _,
    } = program;
    let by_binding = host_resource_map(&data.resources);
    let entry_points = entry_points
        .into_iter()
        .map(|entry| {
            let entry = entry.map_output_routes(|UnrealizedOutputRoute { source }| RealizedOutputRoute {
                source,
                writers: Vec::new(),
            });
            realize_entry(
                entry,
                &by_binding,
                &mut data.resources,
                &mut global_context.effect_ids,
            )
        })
        .collect::<Result<_, ConvertError>>()?;
    let program = Program::from_parts(functions, externs, entry_points, constants, data, global_context);
    // Output retargeting can rewrite a captured `map` result from a Composite
    // array to a storage view; sync each capturing region's parameter type so
    // the region body lowers consistently.
    reconcile::run(program)
}

fn realize_entry(
    mut entry: RawEntry<RealizedOutputRoute>,
    by_binding: &HashMap<crate::BindingRef, ResourceId>,
    resources: &mut LogicalResourceArena,
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> Result<RawEntry<RealizedOutputRoute>, ConvertError> {
    if entry.outputs.is_empty() {
        return Ok(entry);
    }
    if entry.execution_model.is_compute() {
        if entry.routes().next().is_none() {
            synthesize_compute_routes(&mut entry);
        }
        realize_compute_slots(&mut entry, by_binding, resources, effect_ids)?;
        clear_compute_returns(&mut entry);
    } else {
        realize_graphics_returns(&mut entry, effect_ids)?;
    }
    Ok(entry)
}

/// Compute entry points publish exclusively through their output routes. Once
/// those writers exist, no value may remain on an entry terminator.
fn clear_compute_returns(entry: &mut RawEntry<RealizedOutputRoute>) {
    for (_, block) in &mut entry.graph.skeleton.blocks {
        if matches!(block.term, SkeletonTerminator::Return(Some(_))) {
            block.term = SkeletonTerminator::Return(None);
        }
    }
}

/// Compute slot-source path. Each
/// declared output's `SlotSource`s independently lower to a DPS write
/// into the shared `OutputView`. Multi-source slots (`If`-forks etc.)
/// share one view; runtime CFG picks which source's write fires.
fn realize_compute_slots(
    entry: &mut RawEntry<RealizedOutputRoute>,
    by_binding: &HashMap<crate::BindingRef, ResourceId>,
    resources: &mut LogicalResourceArena,
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> Result<(), ConvertError> {
    let Entry {
        graph,
        outputs,
        resource_declarations,
        ..
    } = entry;
    // One producer snapshot for the whole slot loop: everything below
    // appends side effects or rewrites them in place, so sites stay valid.
    let effect_index = graph.side_effect_index();

    for (slot_index, output) in outputs.iter_mut().enumerate() {
        let sources: Vec<SlotSource> = output.routes.iter().map(|route| route.source).collect();
        if sources.is_empty() {
            return Err(ConvertError::Unsupported(format!(
                "compute output #{} has no source — TLC-to-EGIR conversion \
                 must derive at least one route for every declared output",
                slot_index
            )));
        }

        // The two array fields of bucket_scatter are already resident: field
        // zero aliases the caller-supplied destination and field one aliases
        // the compiler-owned count buffer. Publishing another auto output and
        // copying either view would both waste bandwidth and misinterpret the
        // view's `(offset, len)` representation as array elements.
        if sources.len() == 1 {
            if let Some((resource, result, existing_view)) =
                bucket_array_output(graph, &effect_index, sources[0].value)
            {
                let view = existing_view.unwrap_or_else(|| {
                    super::graph_ops::intern_resource_view(
                        graph,
                        resource.0,
                        Type::Constructed(crate::ast::TypeName::UInt(32), vec![]),
                        None,
                    )
                });
                let source_ty = graph.nodes[view].ty.clone();
                let displaced_output = output.resource;
                output.resource = Some(resource);
                output.ty = source_ty.clone();
                output.make_storage_internal();
                output.routes[0].source.value = view;
                output.routes[0].writers = vec![OutputWriter::Value(result)];
                if displaced_output != Some(resource) {
                    resource_declarations
                        .retain(|declaration| Some(declaration.resource) != displaced_output);
                }
                continue;
            }
        }

        let binding = output.storage_binding().expect("BUG: compute output without storage binding");
        let resource = *by_binding.get(&binding).expect("compute output must have a semantic resource");

        // An existential SOAC field can retain its source-level Abstract
        // representation on the entry declaration even though the producer
        // has already chosen a concrete view. Output realization operates on
        // representation, so inherit that concrete source type before
        // classifying fixed versus runtime arrays.
        if sources.len() == 1
            && matches!(
                output.ty.array_variant(),
                Some(Type::Constructed(crate::ast::TypeName::ArrayVariantAbstract, _))
            )
        {
            let source_ty = graph.nodes[sources[0].value].ty.clone();
            if source_ty.is_array() {
                output.ty = source_ty;
            }
        }

        // A runtime `filter` whose result is this output retargets directly:
        // its serial loop compacts into the output buffer and writes a paired
        // length cell. No DPS store is emitted — the filter *is* the writer.
        if sources.len() == 1
            && dispatch::retarget_filter_output(
                graph,
                resource_declarations,
                resources,
                resource,
                output,
                sources[0].value,
            )?
        {
            output.routes[0].writers = vec![OutputWriter::Value(sources[0].value)];
            continue;
        }

        let multi_source = sources.len() > 1;
        let output_ty = output.ty.clone();
        for (route, src) in output.routes.iter_mut().zip(&sources) {
            let mut writers = source_value_writers(graph, &effect_index, src.value);
            writers.extend(dispatch::compute_slot_source(
                graph,
                &effect_index,
                effect_ids,
                src.block,
                src.value,
                slot_index,
                &output_ty,
                resource,
                multi_source,
            )?);
            dedup_output_writers(&mut writers);
            route.writers = writers;
        }
    }
    Ok(())
}

/// Convert a generated compute return into explicit output routes before
/// materializing its writers.
///
///
///   * Graphics entries (vertex / fragment) — outputs are scalar /
///     vector / matrix written to `OutputSlot { index }` places.
///   * Generated compute entries — outputs are storage-buffer-bound; the
///     SOAC at the tail may need retargeting via `compute_slot_source`.
fn synthesize_compute_routes(entry: &mut RawEntry<RealizedOutputRoute>) {
    let Entry { graph, outputs, .. } = entry;

    let Some((return_block, result)) = unique_value_return(graph) else {
        return;
    };
    let sources = output_sources(graph, result, outputs);
    for (slot, source) in sources.into_iter().enumerate() {
        outputs[slot].routes.push(RealizedOutputRoute {
            source: SlotSource {
                block: return_block,
                value: source,
            },
            writers: Vec::new(),
        });
    }
}

/// Graphics entries retain return values because their ABI is location-based
/// IO, not storage output routes.
fn realize_graphics_returns(
    entry: &mut RawEntry<RealizedOutputRoute>,
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> Result<(), ConvertError> {
    let Entry { graph, outputs, .. } = entry;
    let Some((return_block, result)) = unique_value_return(graph) else {
        return Ok(());
    };
    let effect_index = graph.side_effect_index();
    for (slot, source) in output_sources(graph, result, outputs).into_iter().enumerate() {
        let output = &mut outputs[slot];
        let mut writers = source_value_writers(graph, &effect_index, source);
        writers.push(dispatch::graphics_slot_source(
            graph,
            return_block,
            effect_ids,
            source,
            slot,
            &output.ty,
        ));
        dedup_output_writers(&mut writers);
        output.routes.push(RealizedOutputRoute {
            source: SlotSource {
                block: return_block,
                value: source,
            },
            writers,
        });
    }

    graph.skeleton.blocks[return_block].term = SkeletonTerminator::Return(None);
    Ok(())
}

fn unique_value_return(graph: &EGraph<Raw>) -> Option<(BlockId, NodeId)> {
    let mut returns = graph.skeleton.blocks.iter().filter_map(|(block, body)| {
        let SkeletonTerminator::Return(Some(value)) = body.term else {
            return None;
        };
        Some((block, value))
    });
    let result = returns.next();
    assert!(
        returns.next().is_none(),
        "realize_outputs: entry body has more than one Return(Some(..)) terminator"
    );
    result
}

fn source_value_writers(
    graph: &EGraph<Raw>,
    effect_index: &super::types::SideEffectIndex,
    source: NodeId,
) -> Vec<OutputWriter> {
    let mut writers = Vec::new();
    wyn_graph::for_each_reachable(
        [source],
        wyn_graph::WalkOrder::DepthFirst,
        |node, dependencies| {
            if effect_index.site(node).is_none() {
                dependencies.extend(graph.nodes[node].kind.children());
            }
        },
        |node| {
            if effect_index.effect(graph, node).is_some_and(|effect| {
                matches!(effect.kind, super::types::SideEffectKind::Soac(SoacEffect(_, _)))
            }) {
                writers.push(OutputWriter::Value(node));
            }
        },
    );
    writers
}

fn dedup_output_writers(writers: &mut Vec<OutputWriter>) {
    let mut seen = crate::LookupSet::new();
    writers.retain(|writer| seen.insert(*writer));
}

/// Per-output source nodes: the single result, the operands of a literal
/// `Tuple(n)` result, or `Project(result, i)` for an opaque tuple.
fn output_sources(
    graph: &mut EGraph<Raw>,
    result: NodeId,
    outputs: &[super::ir::EntryOutput<
        SemanticResourceRef,
        RealizedOutputRoute,
        super::types::WynLanguage,
    >],
) -> Vec<NodeId> {
    use super::types::{ENode, PureOp};
    use smallvec::smallvec;

    let n = outputs.len();
    if n == 1 {
        return vec![result];
    }
    if let ENode::Pure {
        op: PureOp::Tuple(k),
        operands,
    } = &graph.nodes[result].kind
    {
        if *k == n && operands.len() == n {
            return operands.to_vec();
        }
    }
    outputs
        .iter()
        .enumerate()
        .map(|(i, output)| {
            graph.intern_pure(
                PureOp::Project { index: i as u32 },
                smallvec![result],
                output.ty.clone(),
                None,
            )
        })
        .collect()
}
