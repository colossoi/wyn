//! Parallel scan candidate analysis, binding, and phase emission.

use super::*;
/// Complete graph-local scan recipe, consumed before entry mutation.
pub(super) struct ScanCandidate {
    pub site: SideEffectSite,
    pub owner: SemanticOpId,
    pub scratch_type: Type<TypeName>,
    serial: recognize::SerialScremaRecipe,
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

pub(super) struct BoundScan {
    candidate: ScanCandidate,
    block_sums: ResourceId,
    block_offsets: ResourceId,
}

pub(super) fn analyze_scan_candidate(
    entry: &crate::egir::program::PlannedEntry,
    located: ParallelScan<'_>,
) -> error::Result<RecipeSelection<ScanCandidate>> {
    let serial = located.serial_recipe();
    let (located, lanes, operators) = located.into_parts();
    let site = located.site;
    let side_effect = located.effect;
    let operator = &operators.first;
    if !can_clone_pure_subgraph(&entry.graph, operator.neutral, &[]) {
        return Ok(RecipeSelection::Serial(FallbackReason::UnsupportedCaptures));
    }
    let operands =
        screma::ScremaOperands::decode(located.op, &side_effect.operand_nodes, side_effect.result)?;
    let input = operands.input(0).node;
    if !can_chunk_view(&entry.graph, input, ChunkInputKind::StorageOrRange) {
        return Ok(RecipeSelection::Serial(FallbackReason::UnsupportedViewShape));
    }
    let mut map_output_view_operands = Vec::with_capacity(lanes.maps.len());
    for index in 0..lanes.maps.len() {
        let Some(output) = operands.output(index) else {
            return Ok(RecipeSelection::Serial(FallbackReason::UnsupportedDestination));
        };
        if !can_chunk_view(&entry.graph, output.node, ChunkInputKind::StorageOnly) {
            return Ok(RecipeSelection::Serial(FallbackReason::UnsupportedViewShape));
        }
        map_output_view_operands.push(output.slot);
    }
    let Some(scan_output) = operands.output(lanes.maps.len()) else {
        return Ok(RecipeSelection::Serial(FallbackReason::UnsupportedDestination));
    };
    let Some(scan_output_storage) = graph_ops::extract_storage_view_source(&entry.graph, scan_output.node)
    else {
        return Ok(RecipeSelection::Serial(FallbackReason::UnsupportedDestination));
    };
    let owner = located.owner;
    let scratch_type = semantic_node_type(&entry.graph, operator.neutral);
    if crate::ssa::layout::type_byte_size(&scratch_type).is_none() {
        return Ok(RecipeSelection::Serial(FallbackReason::UnsupportedScratchLayout));
    }
    let input_view_type = semantic_node_type(&entry.graph, input);
    let scan_output_view_type = semantic_node_type(&entry.graph, scan_output.node);
    Ok(RecipeSelection::Parallel(ScanCandidate {
        site,
        owner,
        scratch_type,
        serial,
        step_region: operator.step.region,
        combine_region: operator.combine.region,
        step_captures: operator.step.captures.clone(),
        neutral: operator.neutral,
        input_view: input,
        input_view_type,
        map_output_view_operands,
        scan_output_view_operand: scan_output.slot,
        scan_output_storage,
        scan_output_view_type,
    }))
}

impl BoundScan {
    pub(super) fn bind(candidate: ScanCandidate, resources: &super::planning::ScratchBindings) -> Self {
        let block_sums = resources.id(candidate.owner, CompilerResourceKind::ScanBlockSums, 0);
        let block_offsets = resources.id(candidate.owner, CompilerResourceKind::ScanBlockOffsets, 1);
        Self {
            candidate,
            block_sums,
            block_offsets,
        }
    }
}

impl KernelPlanBuilder<'_, '_> {
    pub(super) fn emit_scan_entry(
        &mut self,
        entry: &mut crate::egir::program::PlannedEntry,
        analysis: BoundScan,
    ) -> error::Result<Vec<crate::egir::program::PlannedEntry>> {
        let total_threads = self.policy.reduce_phase1_width;
        let ScanCandidate {
            site,
            scratch_type: elem_ty,
            serial,
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
        let block_id = site.block;
        let (block_sums_resource, block_offsets_resource) = (analysis.block_sums, analysis.block_offsets);
        let op_func = self.schedule.callable_name(step_region).to_string();
        let reduce_func = self.schedule.callable_name(combine_region).to_string();

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
            semantic_effect_mut(&mut entry.graph, site).operand_nodes[0] = chunked_input_nid;
        }
        for (map_index, operand_index) in map_output_view_ops.iter().enumerate() {
            let original = semantic_effect(&entry.graph, site).operand_nodes[*operand_index];
            let view_ty = semantic_node_type(&entry.graph, original);
            let chunked_view = chunk_view_like(
                &mut entry.graph,
                original,
                view_ty,
                chunk_start,
                chunk_len,
                ChunkInputKind::StorageOnly,
                &format!("SegScan map output {map_index}"),
            )?;
            semantic_effect_mut(&mut entry.graph, site).operand_nodes[*operand_index] = chunked_view;
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
            semantic_effect_mut(&mut entry.graph, site).operand_nodes[scan_output_view_op] =
                chunked_scan_output;
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
            // The neutral is owned by `Operator::neutral`; effect operands
            // contain only the input view.
            let reduce_operands: smallvec::SmallVec<[NodeId; 4]> = smallvec![chunked_input_nid];
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
                                region: self.schedule.intern_callable(&op_func),
                                captures: step_capture_nodes,
                            },
                            combine: SegBody {
                                region: self.schedule.intern_callable(&op_func),
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
                self.effect_ids,
                None,
            );
            let result_nid = entry.graph.intern_pure(
                crate::egir::types::PureOp::Project { index: 0 },
                smallvec![screma_nid],
                elem_ty.clone(),
                None,
            );
            let arr_ty =
                crate::types::view_array_with_size(&elem_ty, Type::Variable(0), crate::types::no_buffer());
            let block_sums_view =
                graph_ops::intern_resource_view(&mut entry.graph, block_sums_resource, arr_ty, None);
            graph_ops::emit_storage_store(
                &mut entry.graph,
                block_id,
                block_sums_view,
                chunked.tid,
                result_nid,
                elem_ty.clone(),
                self.effect_ids,
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
                size: self.resources.get(resource)?.size.clone(),
            });
        }

        let scan_scratch = ScanScratch {
            block_sums: block_sums_resource,
            block_offsets: block_offsets_resource,
        };
        let phase2 = ScanPhase2Spec {
            entry_name: entry.name.clone(),
            operator: reduce_func.clone(),
            elem_ty: elem_ty.clone(),
            source_graph: &entry.graph,
            neutral: init_nid,
            scratch: scan_scratch,
            total_out: None,
        };
        let mut phase2 = phase2.build(self.effect_ids)?;
        apply_manifest_resource_sizes(&mut phase2, &self.resources)?;
        let swap_wrapper_name = format!("{}_scan_op_swap", entry.name);
        let swap_wrapper = synthesize_swap_wrapper(
            swap_wrapper_name.clone(),
            reduce_func,
            elem_ty.clone(),
            entry.span,
        );
        let swap_region = self.schedule.define_callable(swap_wrapper);
        let phase3 = ScanPhase3Spec {
            entry_name: entry.name.clone(),
            swap_region,
            elem_ty,
            output_resource: scan_output_storage.0,
            block_offsets: block_offsets_resource,
            width: total_threads,
        };
        let mut phase3 = phase3.build(self.effect_ids)?;
        apply_manifest_resource_sizes(&mut phase3, &self.resources)?;

        // Phase 1 is now a per-invocation Screma scan over the thread's chunk plus
        // the appended block-sum reduce; `soac_expand` lowers both.
        make_screma_serial(&mut entry.graph, serial);
        Ok(vec![phase2, phase3])
    }
}
