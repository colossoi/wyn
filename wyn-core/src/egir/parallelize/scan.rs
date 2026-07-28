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
    let _ = (entry, located);
    Ok(None)
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
        entry: crate::egir::program::PlannedEntry,
        analysis: BoundScan,
    ) -> error::Result<[BuiltPhase; 3]> {
        let _ = (self, entry, analysis);
        Err("parallel scan lowering has not yet been migrated to canonical Screma lambdas".into())
    }
}
