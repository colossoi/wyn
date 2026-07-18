//! Shared builders for the prefix-scan phases used by scans and filters.

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
