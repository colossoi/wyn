//! Parallel scan candidate analysis, binding, and phase emission.

#![cfg_attr(not(test), deny(clippy::expect_used, clippy::unwrap_used))]

use super::*;
/// Complete graph-local scan recipe, consumed before entry mutation.
pub(super) struct ScanCandidate {
    pub block: BlockId,
    pub effect: usize,
    pub owner: SemanticOpId,
    pub scratch_type: Type<TypeName>,
    step_region: RegionId,
    combine_region: RegionId,
    step_captures: Vec<NodeId>,
    neutral: NodeId,
    input_view: NodeId,
    input_view_type: Type<TypeName>,
    map_output_view_operands: Vec<usize>,
    scan_output_view_operand: usize,
    scan_output_storage: SemanticResourceRef,
    scan_output_view_type: Type<TypeName>,
}

pub(super) struct ScanAnalysis {
    candidate: ScanCandidate,
    block_sums: ResourceId,
    block_offsets: ResourceId,
}

pub(super) fn analyze_scan_candidate(
    entry: &crate::egir::program::PlannedEntry,
) -> error::Result<RecipeSelection<ScanCandidate>> {
    let (block, effect, _) = segmented_recipe_effect(entry).ok_or_else(|| {
        error::ParallelizeError::Invalid("scan analysis has no selected segmented effect".into())
    })?;
    let side_effect = semantic_effect(&entry.graph, block, effect)?;
    match seg_recipe_family(side_effect) {
        Ok(SegScratchFamily::Scan) => {}
        Ok(SegScratchFamily::Reduce) => {
            return Err(error::ParallelizeError::Invalid(
                "scan analysis received a reduce operation".into(),
            ));
        }
        Err(reason) => return Ok(RecipeSelection::Serial(reason)),
    }
    let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(screma::Op::Scan { lanes, operators, .. }))) =
        &side_effect.kind
    else {
        return Err(error::ParallelizeError::Invalid(
            "selected scan effect changed operation kind during analysis".into(),
        ));
    };
    let operator = &operators.first;
    if !can_clone_pure_subgraph(&entry.graph, operator.neutral, &[]) {
        return Ok(RecipeSelection::Serial(FallbackReason::UnsupportedCaptures));
    }
    let input = *side_effect.operand_nodes.first().ok_or_else(|| {
        error::ParallelizeError::Invalid("selected scan effect has no input operand".into())
    })?;
    if !can_chunk_view(&entry.graph, input, ChunkInputKind::StorageOrRange) {
        return Ok(RecipeSelection::Serial(FallbackReason::UnsupportedViewShape));
    }
    let output_base = lanes.inputs.len();
    for index in 0..lanes.maps.len() {
        let view = *side_effect.operand_nodes.get(output_base + index).ok_or_else(|| {
            error::ParallelizeError::Invalid(format!("scan recipe is missing map output operand {index}"))
        })?;
        if !can_chunk_view(&entry.graph, view, ChunkInputKind::StorageOnly) {
            return Ok(RecipeSelection::Serial(FallbackReason::UnsupportedViewShape));
        }
    }
    let scan_output_view_operand = output_base + lanes.maps.len();
    let scan_output = *side_effect.operand_nodes.get(scan_output_view_operand).ok_or_else(|| {
        error::ParallelizeError::Invalid("scan recipe is missing its output-view operand".into())
    })?;
    let Some(scan_output_storage) = graph_ops::extract_storage_view_source(&entry.graph, scan_output)
    else {
        return Ok(RecipeSelection::Serial(FallbackReason::UnsupportedDestination));
    };
    let owner = side_effect.kind.soac_id().copied().ok_or_else(|| {
        error::ParallelizeError::Invalid("selected scan effect has no semantic operation id".into())
    })?;
    let scratch_type = semantic_node_type(&entry.graph, operator.neutral)?;
    if crate::ssa::layout::type_byte_size(&scratch_type).is_none() {
        return Ok(RecipeSelection::Serial(FallbackReason::UnsupportedScratchLayout));
    }
    let input_view_type = semantic_node_type(&entry.graph, input)?;
    let scan_output_view_type = semantic_node_type(&entry.graph, scan_output)?;
    Ok(RecipeSelection::Parallel(ScanCandidate {
        block,
        effect,
        owner,
        scratch_type,
        step_region: operator.step.region,
        combine_region: operator.combine.region,
        step_captures: operator.step.captures.clone(),
        neutral: operator.neutral,
        input_view: input,
        input_view_type,
        map_output_view_operands: (0..lanes.maps.len()).map(|index| output_base + index).collect(),
        scan_output_view_operand,
        scan_output_storage,
        scan_output_view_type,
    }))
}

pub(super) fn bind_scan_candidate(
    candidate: ScanCandidate,
    resources: &planning::ResourceIndex<'_>,
) -> error::Result<ScanAnalysis> {
    let block_sums = resources.exactly_one_at(candidate.owner, CompilerResourceKind::ScanBlockSums, 0)?.id;
    let block_offsets =
        resources.exactly_one_at(candidate.owner, CompilerResourceKind::ScanBlockOffsets, 1)?.id;
    Ok(ScanAnalysis {
        candidate,
        block_sums,
        block_offsets,
    })
}

pub(super) fn emit_scan_entry(
    entry: &mut crate::egir::program::PlannedEntry,
    analysis: ScanAnalysis,
    schedule: &mut schedule::KernelPlan,
    resources: &planning::ResourceIndex<'_>,
    policy: ParallelPolicy,
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> error::Result<Vec<crate::egir::program::PlannedEntry>> {
    debug_assert_eq!(
        segmented_recipe_effect(entry).map(|(block, effect, _)| (block, effect)),
        Some((analysis.candidate.block, analysis.candidate.effect))
    );
    let total_threads = policy.reduce_phase1_width;
    let ScanCandidate {
        block: block_id,
        effect: idx,
        scratch_type: elem_ty,
        step_region,
        combine_region,
        step_captures: step_capture_nodes,
        neutral: init_nid,
        input_view: input_view_nid,
        input_view_type: input_view_ty,
        map_output_view_operands: map_output_view_ops,
        scan_output_view_operand: scan_output_view_op,
        scan_output_storage,
        scan_output_view_type: orig_scan_output_view_ty,
        ..
    } = analysis.candidate;
    let (block_sums_resource, block_offsets_resource) = (analysis.block_sums, analysis.block_offsets);
    let op_func = schedule.callable_name(step_region).to_string();
    let reduce_func = schedule.callable_name(combine_region).to_string();

    // Chunk the input and the scan output view; swap them into the operand list.
    let chunked = chunk_soac_inputs(
        &mut entry.graph,
        &[(input_view_nid, input_view_ty.clone())],
        total_threads,
        ChunkInputKind::StorageOrRange,
        "SegScan",
    )?;
    let chunk_start = chunked.chunk_start;
    let chunk_len = chunked.chunk_len;
    let chunked_input_nid = chunked.views[0];
    {
        let operand = semantic_effect_mut(&mut entry.graph, block_id, idx)?
            .operand_nodes
            .first_mut()
            .ok_or_else(|| error::ParallelizeError::Invalid("scan recipe lost its input operand".into()))?;
        *operand = chunked_input_nid;
    }
    for (map_index, operand_index) in map_output_view_ops.iter().enumerate() {
        let original = *semantic_effect(&entry.graph, block_id, idx)?
            .operand_nodes
            .get(*operand_index)
            .ok_or_else(|| {
                error::ParallelizeError::Invalid(format!(
                    "scan recipe is missing map output operand {operand_index}"
                ))
            })?;
        let view_ty = semantic_node_type(&entry.graph, original)?;
        let chunked_view = chunk_view_like(
            &mut entry.graph,
            original,
            view_ty,
            chunk_start,
            chunk_len,
            ChunkInputKind::StorageOnly,
            &format!("SegScan map output {map_index}"),
        )?;
        let operand = semantic_effect_mut(&mut entry.graph, block_id, idx)?
            .operand_nodes
            .get_mut(*operand_index)
            .ok_or_else(|| {
                error::ParallelizeError::Invalid(format!(
                    "scan recipe lost map output operand {operand_index}"
                ))
            })?;
        *operand = chunked_view;
    }
    let chunked_scan_output = graph_ops::intern_chunked_resource_view(
        &mut entry.graph,
        scan_output_storage.0,
        chunk_start,
        chunk_len,
        orig_scan_output_view_ty,
        None,
    );
    {
        let operand = semantic_effect_mut(&mut entry.graph, block_id, idx)?
            .operand_nodes
            .get_mut(scan_output_view_op)
            .ok_or_else(|| {
                error::ParallelizeError::Invalid("scan recipe lost its output-view operand".into())
            })?;
        *operand = chunked_scan_output;
    }

    // Append a chunked reduce over the same input that stores each thread's
    // final accumulator to `block_sums[tid]`.
    {
        let next_semantic_op = entry
            .graph
            .skeleton
            .blocks
            .iter()
            .flat_map(|(_, block)| &block.side_effects)
            .filter_map(|effect| effect.kind.soac_id())
            .map(|id| id.0)
            .max()
            .map_or(0, |id| id + 1);
        // `[chunked_input, init]` — the step captures live on the SegBody below.
        let reduce_operands: smallvec::SmallVec<[NodeId; 4]> = smallvec![chunked_input_nid, init_nid];
        let tuple_ty = Type::Constructed(TypeName::Tuple(1), vec![elem_ty.clone()]);
        let screma_nid = graph_ops::emit_pending_soac(
            &mut entry.graph,
            block_id,
            SemanticOpId(next_semantic_op),
            Soac::Screma(screma::Op::Reduce {
                lanes: screma::Lanes {
                    inputs: vec![crate::egir::types::SoacInputType { array: input_view_ty }],
                    maps: vec![],
                },
                operators: screma::NonEmpty {
                    first: screma::Operator {
                        step: SegBody {
                            region: schedule.intern_callable(&op_func),
                            captures: step_capture_nodes,
                        },
                        combine: SegBody {
                            region: schedule.intern_callable(&op_func),
                            captures: vec![],
                        },
                        input_indices: vec![screma::InputId(0)],
                        neutral: init_nid,
                        shape: Vec::new(),
                        commutative: false,
                        destination: SoacDestination::fresh(),
                        result_type: elem_ty.clone(),
                    },
                    rest: Vec::new(),
                },
                state: screma::SemanticState::Serial,
            }),
            reduce_operands,
            tuple_ty,
            effect_ids,
            None,
        );
        let result_nid = entry.graph.intern_pure(
            crate::egir::types::PureOp::Project { index: 0 },
            smallvec![screma_nid],
            elem_ty.clone(),
            None,
        );
        let arr_ty = Type::Constructed(
            TypeName::Array,
            vec![
                elem_ty.clone(),
                Type::Constructed(TypeName::ArrayVariantView, vec![]),
                Type::Variable(0),
                crate::types::no_buffer(),
            ],
        );
        let block_sums_view =
            graph_ops::intern_resource_view(&mut entry.graph, block_sums_resource, arr_ty, None);
        graph_ops::emit_storage_store(
            &mut entry.graph,
            block_id,
            block_sums_view,
            chunked.tid,
            result_nid,
            elem_ty.clone(),
            effect_ids,
            None,
        );
    }

    // Both intermediates are declared on phase 1 (block_sums is written here,
    // block_offsets is read by phase 3) so the verifiers and `realize_outputs`
    // see a consistent interface.
    for resource in [block_sums_resource, block_offsets_resource] {
        entry.resource_declarations.push(SemanticResourceDecl {
            resource: SemanticResourceRef(resource),
            role: crate::interface::StorageRole::Intermediate,
            elem_ty: elem_ty.clone(),
            size: resources.get(resource)?.size.clone(),
        });
    }

    let mut phase2 = synthesize_phase2_scan(
        &entry.name,
        reduce_func.clone(),
        elem_ty.clone(),
        &entry.graph,
        init_nid,
        block_sums_resource,
        block_offsets_resource,
        None,
        effect_ids,
    )?;
    apply_manifest_resource_sizes(&mut phase2, resources)?;
    let swap_wrapper_name = format!("{}_scan_op_swap", entry.name);
    let swap_wrapper = synthesize_swap_wrapper(
        swap_wrapper_name.clone(),
        reduce_func,
        elem_ty.clone(),
        entry.span,
    );
    let swap_region = schedule.define_callable(swap_wrapper);
    let mut phase3 = synthesize_phase3_scan(
        &entry.name,
        swap_region,
        elem_ty,
        required_resource(scan_output_storage),
        block_offsets_resource,
        total_threads,
        effect_ids,
    )?;
    apply_manifest_resource_sizes(&mut phase3, resources)?;

    // Phase 1 is now a per-invocation Screma scan over the thread's chunk plus
    // the appended block-sum reduce; `soac_expand` lowers both.
    make_screma_serial(&mut entry.graph, block_id, idx)?;
    Ok(vec![phase2, phase3])
}

/// Synthesize phase 2 of a parallel scan: a single-invocation sequential
/// exclusive scan over `block_sums`. `block_offsets[i]` is the prefix of
/// blocks strictly before `i`, which phase 3 can safely prepend to chunk `i`.
pub(super) fn synthesize_phase2_scan(
    entry_name: &str,
    op_func: String,
    elem_ty: Type<TypeName>,
    phase1_graph: &crate::egir::types::EGraph,
    phase1_ne_nid: NodeId,
    block_sums_resource: ResourceId,
    block_offsets_resource: ResourceId,
    len_out: Option<ResourceId>,
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> Result<crate::egir::program::PlannedEntry, String> {
    use crate::egir::builder::EntryBuilder;
    let mut b =
        EntryBuilder::new_compute(format!("{}_phase2_scan_sums", entry_name), (1, 1, 1), effect_ids);
    let scratch_len = dispatch_worker_logical_size(&elem_ty);
    b.declare_intermediate_storage_sized(block_sums_resource, elem_ty.clone(), scratch_len.clone());
    b.declare_intermediate_storage_sized(block_offsets_resource, elem_ty.clone(), scratch_len);
    if let Some(len_out) = len_out {
        b.declare_output_storage_sized(
            len_out,
            elem_ty.clone(),
            crate::egir::program::LogicalSize::FixedBytes(4),
        );
    }

    let init_nid = graph_ops::clone_pure_subgraph(phase1_graph, b.graph_mut(), phase1_ne_nid)?;
    let phase2 = build_exclusive_scan_phase2(
        &mut b,
        op_func,
        elem_ty.clone(),
        init_nid,
        block_sums_resource,
        block_offsets_resource,
        len_out.is_some(),
    );
    // A runtime filter publishes the scan's grand total (its survivor count)
    // into the length cell. The generic scan builder above stays oblivious to
    // this; only the bridge that knows the filter's `len_out` wires it up.
    if let (Some(len_out), Some(total)) = (len_out, phase2.total) {
        let (graph, _, effect_ids) = b.construction_parts_mut();
        let len_view = graph_ops::intern_resource_view(graph, len_out, elem_ty.clone(), None);
        graph_ops::emit_storage_store(
            graph,
            phase2.after,
            len_view,
            phase2.zero,
            total,
            elem_ty,
            effect_ids,
            None,
        );
    }
    Ok(b.build())
}

/// What an exclusive-scan phase-2 loop hands back to a caller that wants to
/// append work (e.g. a runtime filter storing the survivor count) to the
/// post-loop `after` block. The loop itself is generic — it knows nothing
/// about where a total is stored.
struct ExclusiveScanPhase2 {
    /// The grand total of all block sums, exposed as an `after` block param.
    /// `Some` only when `want_total` was requested.
    total: Option<NodeId>,
    /// The post-loop block (also left as the builder's current block).
    after: BlockId,
    /// The interned `0` node, reusable as a store index.
    zero: NodeId,
}

fn build_exclusive_scan_phase2(
    b: &mut crate::egir::builder::EntryBuilder,
    op_func: String,
    elem_ty: Type<TypeName>,
    init_nid: NodeId,
    block_sums_resource: ResourceId,
    block_offsets_resource: ResourceId,
    want_total: bool,
) -> ExclusiveScanPhase2 {
    let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
    let bool_ty = Type::Constructed(TypeName::Bool, vec![]);
    let arr_ty = Type::Constructed(
        TypeName::Array,
        vec![
            elem_ty.clone(),
            Type::Constructed(TypeName::ArrayVariantView, vec![]),
            Type::Variable(0),
            crate::types::no_buffer(),
        ],
    );
    let entry_block = b.graph_mut().skeleton.entry;
    let (graph, control_headers, effect_ids) = b.construction_parts_mut();
    let sums = graph_ops::intern_resource_view(graph, block_sums_resource, arr_ty.clone(), None);
    let offsets = graph_ops::intern_resource_view(graph, block_offsets_resource, arr_ty, None);
    let len = emit_resource_len(graph, block_sums_resource);
    let zero = graph_ops::intern_u32(graph, 0, None);
    let one = graph_ops::intern_u32(graph, 1, None);

    let header = graph.skeleton.create_block();
    let body = graph.skeleton.create_block();
    let cont = graph.skeleton.create_block();
    let after = graph.skeleton.create_block();
    let acc = graph.add_block_param(header, elem_ty.clone());
    let index = graph.add_block_param(header, u32_ty.clone());
    graph.skeleton.blocks[entry_block].term = SkeletonTerminator::Branch {
        target: header,
        args: vec![init_nid, zero],
    };
    let condition = graph_ops::intern_binop(graph, "<", index, len, bool_ty, None);
    graph.skeleton.blocks[header].term = SkeletonTerminator::CondBranch {
        cond: condition,
        then_target: body,
        then_args: vec![],
        else_target: after,
        else_args: if want_total { vec![acc] } else { vec![] },
    };
    control_headers.insert(
        header,
        ControlHeader::Loop {
            merge: after,
            continue_block: cont,
        },
    );

    graph_ops::emit_storage_store(
        graph,
        body,
        offsets,
        index,
        acc,
        elem_ty.clone(),
        effect_ids,
        None,
    );
    let value = graph_ops::emit_view_load(graph, body, sums, index, elem_ty.clone(), effect_ids, None);
    let next_acc = graph.intern_pure(
        PureOp::Call(op_func),
        smallvec![acc, value],
        elem_ty.clone(),
        None,
    );
    graph.skeleton.blocks[body].term = SkeletonTerminator::Branch {
        target: cont,
        args: vec![next_acc],
    };
    let continued_acc = graph.add_block_param(cont, graph.types[&acc].clone());
    let next_index = graph_ops::intern_binop(graph, "+", index, one, u32_ty, None);
    graph.skeleton.blocks[cont].term = SkeletonTerminator::Branch {
        target: header,
        args: vec![continued_acc, next_index],
    };
    // Expose the grand total as an `after` block param when requested; the
    // caller (a filter compaction) appends its own store. The generic scan
    // never touches a length cell.
    let total = if want_total {
        let total = graph.add_block_param(after, elem_ty.clone());
        Some(total)
    } else {
        None
    };
    b.set_current_block(after);
    ExclusiveScanPhase2 { total, after, zero }
}

/// Synthesize phase 3 of a parallel scan: a chunked compute entry where each
/// thread reads `off = block_offsets[tid]` and applies `op(off, output[i])` to
/// every element of its chunk of `output`. Map's call convention is
/// `func(elem, ...captures)`, so phase 3 routes through `swap_wrapper_name`
/// (`\(elem, off) -> op(off, elem)`) to keep `off` in the accumulator slot for
/// non-commutative ops.
pub(super) fn synthesize_phase3_scan(
    entry_name: &str,
    swap_region: RegionId,
    elem_ty: Type<TypeName>,
    output_resource: ResourceId,
    block_offsets_resource: ResourceId,
    total_threads: u32,
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> Result<crate::egir::program::PlannedEntry, String> {
    use crate::egir::builder::EntryBuilder;
    let mut b = EntryBuilder::new_compute(
        format!("{}_phase3_add_offsets", entry_name),
        (total_threads, 1, 1),
        effect_ids,
    );
    b.declare_output_storage(output_resource, elem_ty.clone());
    b.declare_intermediate_storage_sized(
        block_offsets_resource,
        elem_ty.clone(),
        dispatch_worker_logical_size(&elem_ty),
    );

    let arr_ty = Type::Constructed(
        TypeName::Array,
        vec![
            elem_ty.clone(),
            Type::Constructed(TypeName::ArrayVariantView, vec![]),
            Type::Variable(0),
            crate::types::no_buffer(),
        ],
    );
    let _output_view = b.emit_storage_view(output_resource, arr_ty.clone());
    let block_offsets_view = b.emit_storage_view(block_offsets_resource, arr_ty.clone());

    let output_len = emit_resource_len(b.graph_mut(), output_resource);
    let (tid, chunk_start, chunk_len) = emit_chunk_arithmetic(b.graph_mut(), total_threads, output_len)?;

    let off_place = b.graph_mut().intern_pure(
        crate::egir::types::PureOp::ViewIndex,
        smallvec![block_offsets_view, tid],
        elem_ty.clone(),
        None,
    );
    let off = b.emit_load(off_place, elem_ty.clone());

    let chunked_output = graph_ops::intern_chunked_resource_view(
        b.graph_mut(),
        output_resource,
        chunk_start,
        chunk_len,
        arr_ty.clone(),
        None,
    );

    b.emit_pending_map_into(
        swap_region,
        chunked_output,
        arr_ty.clone(),
        elem_ty,
        vec![off],
        chunked_output,
        arr_ty,
    );
    Ok(b.build())
}
