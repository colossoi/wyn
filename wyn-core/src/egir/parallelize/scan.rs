//! Parallel scan candidate analysis, binding, and phase emission.

use super::model::REDUCE_PHASE1_WIDTH;
use super::*;

#[derive(Clone, Copy)]
pub(super) struct ScanScratch {
    pub block_sums: ResourceId,
    pub block_offsets: ResourceId,
}

/// Build a single-invocation exclusive scan over block sums.
pub(super) struct ScanPhase2Spec<'a> {
    pub entry_name: String,
    pub operator: String,
    pub elem_ty: Type<TypeName>,
    pub source_graph: &'a EGraph,
    pub neutral: NodeId,
    pub scratch: ScanScratch,
    pub total_out: Option<ResourceId>,
}

impl ScanPhase2Spec<'_> {
    pub(super) fn build(
        self,
        effect_ids: &mut crate::IdSource<EffectToken>,
    ) -> Result<crate::egir::program::PlannedEntry, String> {
        use crate::egir::builder::EntryBuilder;

        let mut builder = EntryBuilder::new_compute(
            format!("{}_phase2_scan_sums", self.entry_name),
            (1, 1, 1),
            effect_ids,
        );
        let scratch_len = dispatch_worker_logical_size(&self.elem_ty);
        builder.declare_intermediate_storage_sized(
            self.scratch.block_sums,
            self.elem_ty.clone(),
            scratch_len.clone(),
        );
        builder.declare_intermediate_storage_sized(
            self.scratch.block_offsets,
            self.elem_ty.clone(),
            scratch_len,
        );
        if let Some(len_out) = self.total_out {
            builder.declare_output_storage_sized(
                len_out,
                self.elem_ty.clone(),
                crate::egir::program::LogicalSize::FixedBytes(4),
            );
        }

        let neutral = graph_ops::clone_pure_subgraph(self.source_graph, builder.graph_mut(), self.neutral)?;
        let phase = self.emit_loop(&mut builder, neutral);
        if let (Some(len_out), Some(total)) = (self.total_out, phase.total) {
            let (graph, _, effect_ids) = builder.construction_parts_mut();
            let len_view = graph_ops::intern_resource_view(graph, len_out, self.elem_ty.clone(), None);
            graph_ops::emit_storage_store(
                graph,
                phase.after,
                len_view,
                phase.zero,
                total,
                self.elem_ty.clone(),
                effect_ids,
                None,
            );
        }
        Ok(builder.build())
    }

    fn emit_loop(
        &self,
        builder: &mut crate::egir::builder::EntryBuilder,
        neutral: NodeId,
    ) -> ExclusiveScanPhase2 {
        let elem_ty = self.elem_ty.clone();
        let want_total = self.total_out.is_some();
        let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
        let bool_ty = Type::Constructed(TypeName::Bool, vec![]);
        let arr_ty =
            crate::types::view_array_with_size(&elem_ty, Type::Variable(0), crate::types::no_buffer());
        let entry_block = builder.graph_mut().skeleton.entry;
        let (graph, control_headers, effect_ids) = builder.construction_parts_mut();
        let sums = graph_ops::intern_resource_view(graph, self.scratch.block_sums, arr_ty.clone(), None);
        let offsets = graph_ops::intern_resource_view(graph, self.scratch.block_offsets, arr_ty, None);
        let len = graph_ops::intern_resource_len(graph, self.scratch.block_sums, None);
        let zero = graph_ops::intern_u32(graph, 0, None);
        let one = graph_ops::intern_u32(graph, 1, None);

        let header = graph.skeleton.create_block();
        let body = graph.skeleton.create_block();
        let continuation = graph.skeleton.create_block();
        let after = graph.skeleton.create_block();
        let accumulator = graph.add_block_param(header, elem_ty.clone());
        let index = graph.add_block_param(header, u32_ty.clone());
        graph.skeleton.blocks[entry_block].term = SkeletonTerminator::Branch {
            target: header,
            args: vec![neutral, zero],
        };
        let condition = graph_ops::intern_binop(graph, "<", index, len, bool_ty, None);
        graph.skeleton.blocks[header].term = SkeletonTerminator::CondBranch {
            cond: condition,
            then_target: body,
            then_args: vec![],
            else_target: after,
            else_args: if want_total { vec![accumulator] } else { vec![] },
        };
        control_headers.insert(
            header,
            ControlHeader::Loop {
                merge: after,
                continue_block: continuation,
            },
        );

        graph_ops::emit_storage_store(
            graph,
            body,
            offsets,
            index,
            accumulator,
            elem_ty.clone(),
            effect_ids,
            None,
        );
        let value = graph_ops::emit_view_load(graph, body, sums, index, elem_ty.clone(), effect_ids, None);
        let next_accumulator = graph.intern_pure(
            PureOp::Call(self.operator.clone()),
            smallvec![accumulator, value],
            elem_ty.clone(),
            None,
        );
        graph.skeleton.blocks[body].term = SkeletonTerminator::Branch {
            target: continuation,
            args: vec![next_accumulator],
        };
        let continued_accumulator = graph.add_block_param(continuation, graph.types[&accumulator].clone());
        let next_index = graph_ops::intern_binop(graph, "+", index, one, u32_ty, None);
        graph.skeleton.blocks[continuation].term = SkeletonTerminator::Branch {
            target: header,
            args: vec![continued_accumulator, next_index],
        };
        let total = if want_total { Some(graph.add_block_param(after, elem_ty)) } else { None };
        builder.set_current_block(after);
        ExclusiveScanPhase2 { total, after, zero }
    }
}

struct ExclusiveScanPhase2 {
    total: Option<NodeId>,
    after: BlockId,
    zero: NodeId,
}

/// Build the chunked phase that prepends each block's exclusive offset.
pub(super) struct ScanPhase3Spec {
    pub entry_name: String,
    pub swap_region: RegionId,
    pub elem_ty: Type<TypeName>,
    pub output_resource: ResourceId,
    pub block_offsets: ResourceId,
    pub width: u32,
}

impl ScanPhase3Spec {
    pub(super) fn build(
        self,
        effect_ids: &mut crate::IdSource<EffectToken>,
    ) -> Result<crate::egir::program::PlannedEntry, String> {
        use crate::egir::builder::EntryBuilder;

        let mut builder = EntryBuilder::new_compute(
            format!("{}_phase3_add_offsets", self.entry_name),
            (self.width, 1, 1),
            effect_ids,
        );
        builder.declare_output_storage(self.output_resource, self.elem_ty.clone());
        builder.declare_intermediate_storage_sized(
            self.block_offsets,
            self.elem_ty.clone(),
            dispatch_worker_logical_size(&self.elem_ty),
        );
        let arr_ty =
            crate::types::view_array_with_size(&self.elem_ty, Type::Variable(0), crate::types::no_buffer());
        let _output_view = builder.emit_storage_view(self.output_resource, arr_ty.clone());
        let block_offsets_view = builder.emit_storage_view(self.block_offsets, arr_ty.clone());
        let output_len = graph_ops::intern_resource_len(builder.graph_mut(), self.output_resource, None);
        let (tid, chunk_start, chunk_len) =
            emit_chunk_arithmetic(builder.graph_mut(), self.width, output_len)?;
        let offset_place = builder.graph_mut().intern_pure(
            PureOp::ViewIndex,
            smallvec![block_offsets_view, tid],
            self.elem_ty.clone(),
            None,
        );
        let offset = builder.emit_load(offset_place, self.elem_ty.clone());
        let chunked_output = graph_ops::intern_chunked_resource_view(
            builder.graph_mut(),
            self.output_resource,
            chunk_start,
            chunk_len,
            arr_ty.clone(),
            None,
        );

        builder.emit_pending_map_into(
            self.swap_region,
            chunked_output,
            arr_ty.clone(),
            self.elem_ty,
            vec![offset],
            chunked_output,
            arr_ty,
        );
        Ok(builder.build())
    }
}

/// Complete graph-local scan recipe, consumed before entry mutation.
pub(super) struct ScanCandidate {
    pub site: SideEffectSite,
    pub owner: SemanticOpId,
    pub scratch_type: Type<TypeName>,
    serial: SerialScremaRecipe,
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
    phase1_width: u32,
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
    let scratch_type = entry.graph.types[&operator.neutral].clone();
    if crate::ssa::layout::type_byte_size(&scratch_type).is_none() {
        return Ok(RecipeSelection::Serial(FallbackReason::UnsupportedScratchLayout));
    }
    let input_view_type = entry.graph.types[&input].clone();
    let scan_output_view_type = entry.graph.types[&scan_output.node].clone();
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
        phase1_width: REDUCE_PHASE1_WIDTH,
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
        let ScanCandidate {
            site,
            owner,
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
            phase1_width: total_threads,
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
            entry.graph.skeleton.effect_mut(site).operand_nodes[0] = chunked_input_nid;
        }
        for (map_index, operand_index) in map_output_view_ops.iter().enumerate() {
            let original = entry.graph.skeleton.effect(site).operand_nodes[*operand_index];
            let view_ty = entry.graph.types[&original].clone();
            let chunked_view = chunk_view_like(
                &mut entry.graph,
                original,
                view_ty,
                chunk_start,
                chunk_len,
                ChunkInputKind::StorageOnly,
                &format!("SegScan map output {map_index}"),
            )?;
            entry.graph.skeleton.effect_mut(site).operand_nodes[*operand_index] = chunked_view;
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
            entry.graph.skeleton.effect_mut(site).operand_nodes[scan_output_view_op] = chunked_scan_output;
        }

        // Append a chunked reduce over the same input that stores each thread's
        // final accumulator to `block_sums[tid]`.
        {
            // The neutral is owned by `Operator::neutral`; effect operands
            // contain only the input view.
            let reduce_operands: smallvec::SmallVec<[NodeId; 4]> = smallvec![chunked_input_nid];
            let tuple_ty = Type::Constructed(TypeName::Tuple(1), vec![elem_ty.clone()]);
            let screma_nid = graph_ops::emit_pending_soac(
                &mut entry.graph,
                block_id,
                owner.implementation(0),
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
                size: self.resources[resource].size.clone(),
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
        apply_manifest_resource_sizes(&mut phase2, self.resources);
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
        apply_manifest_resource_sizes(&mut phase3, self.resources);

        // Phase 1 is now a per-invocation Screma scan over the thread's chunk plus
        // the appended block-sum reduce; `soac_expand` lowers both.
        make_screma_serial(&mut entry.graph, serial);
        Ok(vec![phase2, phase3])
    }
}
