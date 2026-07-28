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
        semantic_ids: &mut crate::egir::program::SemanticOpIdSource,
        effect_ids: &mut crate::IdSource<EffectToken>,
    ) -> Result<BuiltPhase, String> {
        use crate::egir::builder::EntryBuilder;

        let mut accesses = vec![
            schedule::ScheduledResource {
                resource: self.scratch.block_sums,
                access: crate::ResourceAccess::Read,
            },
            schedule::ScheduledResource {
                resource: self.scratch.block_offsets,
                access: crate::ResourceAccess::Write,
            },
        ];
        if let Some(resource) = self.total_out {
            accesses.push(schedule::ScheduledResource {
                resource,
                access: crate::ResourceAccess::Write,
            });
        }

        let mut builder = EntryBuilder::new_compute(
            format!("{}_phase2_scan_sums", self.entry_name),
            (1, 1, 1),
            semantic_ids,
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
            let (graph, effect_ids) = builder.construction_parts_mut();
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
        Ok(BuiltPhase::new(builder.build(), accesses))
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
        let (graph, effect_ids) = builder.construction_parts_mut();
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
        graph.skeleton.blocks[header].control_header = Some(ControlHeader::Loop {
            merge: after,
            continue_block: continuation,
        });

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
        let continued_accumulator =
            graph.add_block_param(continuation, graph.nodes[accumulator].ty.clone());
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

/// Optional final map applied after block offsets have made each prefix global.
pub(super) struct ScanPostPhaseSpec<'a> {
    pub region: RegionId,
    pub captures: Vec<NodeId>,
    pub source_graph: &'a EGraph,
    pub output_resource: ResourceId,
    pub output_elem_ty: Type<TypeName>,
    pub output_view_ty: Type<TypeName>,
}

/// Build the chunked phase that prepends each block's exclusive offset.
pub(super) struct ScanPhase3Spec<'a> {
    pub entry_name: String,
    pub swap_region: RegionId,
    pub elem_ty: Type<TypeName>,
    /// Holds local prefixes on entry and global prefixes after offsetting.
    pub output_resource: ResourceId,
    pub block_offsets: ResourceId,
    pub width: u32,
    pub post_maps: Vec<ScanPostPhaseSpec<'a>>,
}

impl ScanPhase3Spec<'_> {
    pub(super) fn build(
        self,
        semantic_ids: &mut crate::egir::program::SemanticOpIdSource,
        effect_ids: &mut crate::IdSource<EffectToken>,
    ) -> Result<BuiltPhase, String> {
        use crate::egir::builder::EntryBuilder;

        let mut resources = vec![
            schedule::ScheduledResource {
                resource: self.output_resource,
                access: crate::ResourceAccess::ReadWrite,
            },
            schedule::ScheduledResource {
                resource: self.block_offsets,
                access: crate::ResourceAccess::Read,
            },
        ];
        for post_map in &self.post_maps {
            resources.push(schedule::ScheduledResource {
                resource: post_map.output_resource,
                access: crate::ResourceAccess::Write,
            });
        }
        resources.sort_by_key(|resource| resource.resource);

        let mut builder = EntryBuilder::new_compute(
            format!("{}_phase3_add_offsets", self.entry_name),
            (self.width, 1, 1),
            semantic_ids,
            effect_ids,
        );
        if self.post_maps.is_empty() {
            builder.declare_output_storage(self.output_resource, self.elem_ty.clone());
        } else {
            builder.declare_intermediate_storage_sized(
                self.output_resource,
                self.elem_ty.clone(),
                dispatch_worker_logical_size(&self.elem_ty),
            );
        }
        for post_map in &self.post_maps {
            builder.declare_output_storage(post_map.output_resource, post_map.output_elem_ty.clone());
        }
        builder.declare_intermediate_storage_sized(
            self.block_offsets,
            self.elem_ty.clone(),
            dispatch_worker_logical_size(&self.elem_ty),
        );
        let arr_ty =
            crate::types::view_array_with_size(&self.elem_ty, Type::Variable(0), crate::types::no_buffer());
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
            arr_ty.clone(),
        );
        for post_map in self.post_maps {
            let mut captures = Vec::with_capacity(post_map.captures.len());
            for capture in post_map.captures {
                captures.push(graph_ops::clone_pure_subgraph(
                    post_map.source_graph,
                    builder.graph_mut(),
                    capture,
                )?);
            }
            let output_view = graph_ops::intern_chunked_resource_view(
                builder.graph_mut(),
                post_map.output_resource,
                chunk_start,
                chunk_len,
                post_map.output_view_ty.clone(),
                None,
            );
            builder.emit_pending_map_into(
                post_map.region,
                chunked_output,
                arr_ty.clone(),
                post_map.output_elem_ty,
                captures,
                output_view,
                post_map.output_view_ty,
            );
        }
        Ok(BuiltPhase::new(builder.build(), resources))
    }
}

/// Post-scan map facts that survive recipe selection and scratch binding.
struct ScanPostMap {
    region: RegionId,
    captures: Vec<NodeId>,
    output_elem_type: Type<TypeName>,
    output_resource: SemanticResourceRef,
    output_view_type: Type<TypeName>,
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
    scan_prefix_view_type: Type<TypeName>,
    post_maps: Vec<ScanPostMap>,
    phase1_width: u32,
    segment: screma::Segmented<SemanticResourceRef>,
}

impl ScanCandidate {
    pub(super) fn prefix_scratch_type(&self) -> Option<&Type<TypeName>> {
        (!self.post_maps.is_empty()).then_some(&self.scratch_type)
    }
}

pub(super) struct BoundScan {
    candidate: ScanCandidate,
    block_sums: ResourceId,
    block_offsets: ResourceId,
    scan_prefixes: Option<ResourceId>,
}

pub(super) fn analyze_scan_candidate(
    entry: &crate::egir::program::PlannedEntry,
    located: LocatedScrema<'_>,
) -> error::Result<Option<ScanCandidate>> {
    debug_assert_eq!(
        super::capabilities::ScremaRecipeCapabilities::analyze(located.op).recipe_class(),
        super::capabilities::ScremaRecipeClass::Scan
    );
    let segment = located.segmented()?;
    let serial = located.serial_recipe();
    let lanes = located.op.lanes();
    let operators = located.op.operators();
    let site = located.site;
    let side_effect = located.effect;
    let operator = &operators[0];
    if !can_clone_pure_subgraph(&entry.graph, operator.neutral, &[]) {
        return Ok(None);
    }
    if located.op.post_maps.iter().any(|map| {
        map.body.captures.iter().any(|capture| !can_clone_pure_subgraph(&entry.graph, *capture, &[]))
    }) {
        return Ok(None);
    }
    let operands =
        screma::ScremaOperands::decode(located.op, &side_effect.operand_nodes, side_effect.result)?;
    let input = operands.input(0).node;
    if !can_chunk_view(&entry.graph, input, ChunkInputKind::StorageOrRange) {
        return Ok(None);
    }
    let mut map_output_view_operands = Vec::with_capacity(lanes.maps.len());
    for index in 0..lanes.maps.len() {
        let Some(output) = operands.output(index) else {
            return Ok(None);
        };
        if !can_chunk_view(&entry.graph, output.node, ChunkInputKind::StorageOnly) {
            return Ok(None);
        }
        map_output_view_operands.push(output.slot);
    }
    let Some(scan_output) = operands.output(lanes.maps.len()) else {
        return Ok(None);
    };
    let Some(scan_output_storage) = graph_ops::extract_storage_view_source(&entry.graph, scan_output.node)
    else {
        return Ok(None);
    };
    let owner = located.owner;
    let scratch_type = entry.graph.nodes[operator.neutral].ty.clone();
    if crate::ssa::layout::type_byte_size(&scratch_type).is_none() {
        return Ok(None);
    }
    let input_view_type = entry.graph.nodes[input].ty.clone();
    let scan_output_view_type = entry.graph.nodes[scan_output.node].ty.clone();
    let scan_prefix_view_type = if located.op.post_maps.is_empty() {
        scan_output_view_type.clone()
    } else {
        crate::types::view_array_with_size(&scratch_type, Type::Variable(0), crate::types::no_buffer())
    };
    let mut post_maps = Vec::with_capacity(located.op.post_maps.len());
    for (index, map) in located.op.post_maps.iter().enumerate() {
        let Some(output) = operands.output(lanes.maps.len() + index) else {
            return Ok(None);
        };
        let Some(output_resource) = graph_ops::extract_storage_view_source(&entry.graph, output.node)
        else {
            return Ok(None);
        };
        post_maps.push(ScanPostMap {
            region: map.body.region,
            captures: map.body.captures.clone(),
            output_elem_type: map.output_element_type.clone(),
            output_resource,
            output_view_type: entry.graph.nodes[output.node].ty.clone(),
        });
    }
    Ok(Some(ScanCandidate {
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
        scan_prefix_view_type,
        post_maps,
        phase1_width: REDUCE_PHASE1_WIDTH,
        segment,
    }))
}

impl BoundScan {
    pub(super) fn segment(&self) -> &screma::Segmented<SemanticResourceRef> {
        &self.candidate.segment
    }

    pub(super) fn bind(candidate: ScanCandidate, resources: &super::planning::ScratchBindings) -> Self {
        let block_sums = resources.id(candidate.owner, CompilerResourceKind::ScanBlockSums, 0);
        let block_offsets = resources.id(candidate.owner, CompilerResourceKind::ScanBlockOffsets, 1);
        let scan_prefixes = candidate
            .prefix_scratch_type()
            .map(|_| resources.id(candidate.owner, CompilerResourceKind::ScanPrefixes, 2));
        Self {
            candidate,
            block_sums,
            block_offsets,
            scan_prefixes,
        }
    }
}

impl KernelPlanBuilder<'_, '_> {
    pub(super) fn emit_scan_entry(
        &mut self,
        mut entry: crate::egir::program::PlannedEntry,
        analysis: BoundScan,
    ) -> error::Result<[BuiltPhase; 3]> {
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
            scan_prefix_view_type,
            post_maps,
            phase1_width: total_threads,
            segment,
        } = analysis.candidate;
        let mut phase1_resources = merge_scheduled_resources(
            &declared_input_resources(&entry.resource_declarations),
            &segmented_resources(&segment),
        );
        let block_id = site.block;
        let (block_sums_resource, block_offsets_resource) = (analysis.block_sums, analysis.block_offsets);
        let scan_prefixes_resource = analysis.scan_prefixes.unwrap_or(scan_output_storage.0);
        if !post_maps.is_empty() {
            phase1_resources
                .retain(|resource| !post_maps.iter().any(|map| map.output_resource.0 == resource.resource));
        }
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
            let view_ty = entry.graph.nodes[original].ty.clone();
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
            scan_prefixes_resource,
            chunk_start,
            chunk_len,
            scan_prefix_view_type,
            None,
        );
        if post_maps.is_empty() {
            entry.graph.skeleton.effect_mut(site).operand_nodes[scan_output_view_op] = chunked_scan_output;
        } else {
            let operands = &mut entry.graph.skeleton.effect_mut(site).operand_nodes;
            operands.truncate(scan_output_view_op);
            operands.push(chunked_scan_output);
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
                Soac::Screma(screma::Op {
                    lanes: screma::Lanes {
                        inputs: vec![crate::egir::types::SoacInputType { array: input_view_ty }],
                        maps: vec![],
                    },
                    operators: vec![screma::Operator {
                        kind: screma::OperatorKind::Reduce,
                        step: SegBody {
                            region: step_region,
                            captures: step_capture_nodes,
                        },
                        combine: SegBody {
                            region: step_region,
                            captures: vec![],
                        },
                        input_indices: vec![screma::InputId(0)],
                        neutral: init_nid,
                        shape: Vec::new(),
                        commutative: false,
                        destination: SoacDestination::fresh(),
                        result_type: elem_ty.clone(),
                    }],
                    post_maps: Vec::new(),
                    hidden_scan_outputs: Vec::new(),
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
        for resource in
            [block_sums_resource, block_offsets_resource].into_iter().chain(analysis.scan_prefixes)
        {
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
        let combine_name = self.region_interner.resolve(combine_region).clone();
        let phase2 = ScanPhase2Spec {
            entry_name: entry.name.clone(),
            operator: combine_name.clone(),
            elem_ty: elem_ty.clone(),
            source_graph: &entry.graph,
            neutral: init_nid,
            scratch: scan_scratch,
            total_out: None,
        };
        let mut phase2 = phase2.build(self.semantic_ids, self.effect_ids)?;
        apply_manifest_resource_sizes(&mut phase2.body, self.resources);
        let swap_wrapper_name = format!("{}_scan_op_swap", entry.name);
        let swap_elem_ty = elem_ty.clone();
        let span = entry.span;
        let swap_region = self.define_callable(swap_wrapper_name, |region, name| {
            synthesize_swap_wrapper(region, name, combine_name, swap_elem_ty, span)
        })?;
        let phase3 = ScanPhase3Spec {
            entry_name: entry.name.clone(),
            swap_region,
            elem_ty,
            output_resource: scan_prefixes_resource,
            block_offsets: block_offsets_resource,
            width: total_threads,
            post_maps: post_maps
                .iter()
                .map(|map| ScanPostPhaseSpec {
                    region: map.region,
                    captures: map.captures.clone(),
                    source_graph: &entry.graph,
                    output_resource: map.output_resource.0,
                    output_elem_ty: map.output_elem_type.clone(),
                    output_view_ty: map.output_view_type.clone(),
                })
                .collect(),
        };
        let mut phase3 = phase3.build(self.semantic_ids, self.effect_ids)?;
        apply_manifest_resource_sizes(&mut phase3.body, self.resources);

        // Phase 1 is now a per-invocation Screma scan over the thread's chunk plus
        // the appended block-sum reduce; `soac_expand` lowers both.
        make_screma_serial(&mut entry.graph, serial);
        if !post_maps.is_empty() {
            let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) =
                &mut entry.graph.skeleton.effect_mut(site).kind
            else {
                unreachable!("selected scan site changed during phase construction");
            };
            op.post_maps.clear();
            op.hidden_scan_outputs.clear();
            op.operators_mut()[0].destination =
                SoacDestination::fresh().placed(crate::egir::types::SoacPlacement::OutputView);
            let result_ty =
                Type::Constructed(TypeName::Tuple(1), vec![op.operators()[0].result_type.clone()]);
            let result = entry
                .graph
                .skeleton
                .effect(site)
                .result
                .ok_or_else(|| "parallel scan phase 1 lost its result".to_owned())?;
            entry.graph.retype_node(result, result_ty);
        }
        phase1_resources.push(schedule::ScheduledResource {
            resource: block_sums_resource,
            access: crate::ResourceAccess::Write,
        });
        if analysis.scan_prefixes.is_some() {
            phase1_resources.push(schedule::ScheduledResource {
                resource: scan_prefixes_resource,
                access: crate::ResourceAccess::Write,
            });
        }
        phase1_resources.sort_by_key(|resource| resource.resource);
        Ok([BuiltPhase::new(entry, phase1_resources), phase2, phase3])
    }
}
