//! Parallel scan candidate analysis, binding, and phase emission.

use super::model::REDUCE_PHASE1_WIDTH;
use super::*;
use crate::egir::soac::lambda as lambda_ops;

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

/// One output of the canonical post-scan lambda.
pub(super) struct ScanPostOutput {
    pub resource: ResourceId,
    pub elem_ty: Type<TypeName>,
    pub view_ty: Type<TypeName>,
}

/// The one canonical map run after block offsets have made every prefix global.
pub(super) struct ScanPostPhaseSpec<'a> {
    pub pre: screma::Lambda,
    pub source_graph: &'a EGraph,
    pub inputs: Vec<(NodeId, crate::egir::types::SoacInputType)>,
    pub input_declarations: Vec<SemanticResourceDecl>,
    pub outputs: Vec<ScanPostOutput>,
}

/// Build the chunked phase that prepends each block's exclusive offset and,
/// when needed, evaluates the original whole post-barrier dataflow.
pub(super) struct ScanPhase3Spec<'a> {
    pub entry_name: String,
    pub swap_region: RegionId,
    pub elem_ty: Type<TypeName>,
    /// Holds local prefixes on entry and global prefixes after offsetting.
    pub output_resource: ResourceId,
    pub block_offsets: ResourceId,
    pub width: u32,
    pub post: Option<ScanPostPhaseSpec<'a>>,
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
        if let Some(post) = &self.post {
            resources =
                merge_scheduled_resources(&resources, &declared_input_resources(&post.input_declarations));
            resources = merge_scheduled_resources(
                &resources,
                &post
                    .outputs
                    .iter()
                    .map(|output| schedule::ScheduledResource {
                        resource: output.resource,
                        access: crate::ResourceAccess::Write,
                    })
                    .collect::<Vec<_>>(),
            );
        }

        let mut builder = EntryBuilder::new_compute(
            format!("{}_phase3_add_offsets", self.entry_name),
            (self.width, 1, 1),
            semantic_ids,
            effect_ids,
        );
        if self.post.is_none() {
            builder.declare_output_storage(self.output_resource, self.elem_ty.clone());
        } else {
            builder.declare_intermediate_storage_sized(
                self.output_resource,
                self.elem_ty.clone(),
                dispatch_worker_logical_size(&self.elem_ty),
            );
        }
        if let Some(post) = &self.post {
            for declaration in &post.input_declarations {
                builder.declare_input_storage_sized(
                    declaration.resource.0,
                    declaration.elem_ty.clone(),
                    declaration.size.clone(),
                );
            }
            for output in &post.outputs {
                builder.declare_output_storage(output.resource, output.elem_ty.clone());
            }
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
        if let Some(mut post) = self.post {
            if let Some(body) = post.pre.seg_body_mut() {
                body.captures = body
                    .captures
                    .iter()
                    .map(|capture| {
                        graph_ops::clone_pure_subgraph(post.source_graph, builder.graph_mut(), *capture)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
            }
            let mut inputs = Vec::with_capacity(1 + post.inputs.len());
            inputs.push((
                chunked_output,
                crate::egir::types::SoacInputType { array: arr_ty },
            ));
            for (index, (source, input)) in post.inputs.into_iter().enumerate() {
                let cloned =
                    graph_ops::clone_pure_subgraph(post.source_graph, builder.graph_mut(), source)?;
                let chunked = chunk_view_like(
                    builder.graph_mut(),
                    cloned,
                    input.array.clone(),
                    chunk_start,
                    chunk_len,
                    ChunkInputKind::StorageOrRange,
                    &format!("SegScan phase 3 input {index}"),
                )?;
                inputs.push((chunked, input));
            }
            let output_views = post
                .outputs
                .into_iter()
                .map(|output| {
                    let view = graph_ops::intern_chunked_resource_view(
                        builder.graph_mut(),
                        output.resource,
                        chunk_start,
                        chunk_len,
                        output.view_ty.clone(),
                        None,
                    );
                    (view, output.view_ty)
                })
                .collect();
            builder.emit_pending_map_into_views(inputs, post.pre, output_views);
        }
        Ok(BuiltPhase::new(builder.build(), resources))
    }
}

struct ScanOutput {
    elem_type: Type<TypeName>,
    resource: SemanticResourceRef,
    view_type: Type<TypeName>,
}
/// Complete graph-local scan recipe, consumed before entry mutation.
pub(super) struct ScanCandidate {
    pub site: SideEffectSite,
    pub owner: SemanticOpId,
    pub scratch_type: Type<TypeName>,
    serial: SerialScremaRecipe,
    pre: screma::Lambda,
    scan: screma::Scan,
    post: screma::Lambda,
    input_views: Vec<(NodeId, crate::egir::types::SoacInputType)>,
    result: NodeId,
    outputs: Vec<ScanOutput>,
    direct_output: bool,
    phase1_width: u32,
    segment: screma::Segmented<SemanticResourceRef>,
}

impl ScanCandidate {
    pub(super) fn prefix_scratch_type(&self) -> Option<&Type<TypeName>> {
        (!self.direct_output).then_some(&self.scratch_type)
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
    let [scan] = located.op.form.scans.as_slice() else {
        return Ok(None);
    };
    let [neutral] = scan.neutral.as_slice() else {
        return Ok(None);
    };
    if !can_clone_pure_subgraph(&entry.graph, *neutral, &[]) {
        return Ok(None);
    }

    let operands =
        screma::ScremaOperands::decode(located.op, &located.effect.operand_nodes, located.effect.result)?;
    let mut input_views = Vec::with_capacity(located.op.inputs.len());
    for (operand, input) in operands.inputs().zip(&located.op.inputs) {
        if !can_chunk_view(&entry.graph, operand.node, ChunkInputKind::StorageOrRange) {
            return Ok(None);
        }
        input_views.push((operand.node, input.clone()));
    }

    let mut outputs = Vec::with_capacity(located.op.result_count());
    for field in 0..located.op.result_count() {
        let Some(output) = operands.output(field) else {
            return Ok(None);
        };
        if !can_chunk_view(&entry.graph, output.node, ChunkInputKind::StorageOnly) {
            return Ok(None);
        }
        let Some(resource) = graph_ops::extract_storage_view_source(&entry.graph, output.node) else {
            return Ok(None);
        };
        outputs.push(ScanOutput {
            elem_type: located.op.form.post.result_types[field].clone(),
            resource,
            view_type: entry.graph.nodes[output.node].ty.clone(),
        });
    }
    if outputs.is_empty() {
        return Ok(None);
    }

    let scratch_type = entry.graph.nodes[*neutral].ty.clone();
    if crate::ssa::layout::type_byte_size(&scratch_type).is_none() {
        return Ok(None);
    }
    let direct_output = located.op.form.post.is_identity()
        && located.op.form.mapped_types().is_some_and(|mapped| mapped.is_empty())
        && outputs.len() == 1;
    if !direct_output {
        for capture in located
            .op
            .form
            .pre
            .seg_body()
            .into_iter()
            .flat_map(|body| &body.captures)
            .chain(located.op.form.post.seg_body().into_iter().flat_map(|body| &body.captures))
        {
            if !can_clone_pure_subgraph(&entry.graph, *capture, &[]) {
                return Ok(None);
            }
        }
    }
    Ok(Some(ScanCandidate {
        site: located.site,
        owner: located.owner,
        scratch_type,
        serial,
        pre: located.op.form.pre.clone(),
        scan: scan.clone(),
        post: located.op.form.post.clone(),
        input_views,
        result: operands.result(),
        outputs,
        direct_output,
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
        let BoundScan {
            candidate,
            block_sums: block_sums_resource,
            block_offsets: block_offsets_resource,
            scan_prefixes,
        } = analysis;
        let ScanCandidate {
            site,
            owner,
            scratch_type: elem_ty,
            serial,
            pre,
            scan,
            post,
            input_views,
            result: screma_result,
            outputs,
            direct_output,
            phase1_width: total_threads,
            segment,
        } = candidate;
        let block_id = site.block;
        let neutral = scan.neutral[0];
        let operator_region = scan
            .operator
            .seg_body()
            .ok_or_else(|| "parallel scan operator lost its region".to_owned())?
            .region;
        let combine_name = self.region_interner.resolve(operator_region).clone();
        let phase1_pre = if pre.result_types.len() == 1 {
            pre.clone()
        } else {
            let captures = pre
                .seg_body()
                .into_iter()
                .flat_map(|body| body.captures.iter().copied())
                .collect::<Vec<_>>();
            let capture_types =
                captures.iter().map(|capture| entry.graph.nodes[*capture].ty.clone()).collect::<Vec<_>>();
            let callee = pre.seg_body().map(|body| self.region_interner.resolve(body.region).clone());
            let source = pre.clone();
            let parameter_types = pre.parameter_types.clone();
            let result_type = elem_ty.clone();
            let span = entry.span;
            let region =
                self.define_callable(format!("{}_scan_pre", entry.name), move |region, name| {
                    synthesize_scan_input_function(
                        region,
                        name,
                        source,
                        callee,
                        capture_types,
                        result_type,
                        span,
                    )
                })?;
            screma::Lambda::region(
                SegBody { region, captures },
                parameter_types,
                vec![elem_ty.clone()],
            )
        };

        let post_lambda = if direct_output {
            None
        } else {
            let captures = pre
                .seg_body()
                .into_iter()
                .flat_map(|body| body.captures.iter().copied())
                .chain(post.seg_body().into_iter().flat_map(|body| body.captures.iter().copied()))
                .collect::<Vec<_>>();
            let capture_types =
                captures.iter().map(|capture| entry.graph.nodes[*capture].ty.clone()).collect::<Vec<_>>();
            let pre_callee = pre.seg_body().map(|body| self.region_interner.resolve(body.region).clone());
            let post_callee = post.seg_body().map(|body| self.region_interner.resolve(body.region).clone());
            let source_pre = pre.clone();
            let source_post = post.clone();
            let mut parameter_types = vec![elem_ty.clone()];
            parameter_types.extend(pre.parameter_types.iter().cloned());
            let result_types = post.result_types.clone();
            let span = entry.span;
            let region =
                self.define_callable(format!("{}_scan_post", entry.name), move |region, name| {
                    synthesize_scan_post_function(
                        region,
                        name,
                        source_pre,
                        pre_callee,
                        source_post,
                        post_callee,
                        capture_types,
                        span,
                    )
                })?;
            Some(screma::Lambda::region(
                SegBody { region, captures },
                parameter_types,
                result_types,
            ))
        };

        let mut phase1_resources = merge_scheduled_resources(
            &declared_input_resources(&entry.resource_declarations),
            &segmented_resources(&segment),
        );
        if !direct_output {
            phase1_resources
                .retain(|resource| !outputs.iter().any(|output| output.resource.0 == resource.resource));
        }
        let input_view_data =
            input_views.iter().map(|(node, input)| (*node, input.array.clone())).collect::<Vec<_>>();
        make_screma_serial(&mut entry.graph, serial);
        let chunked = chunk_soac_inputs(
            &mut entry.graph,
            &input_view_data,
            total_threads,
            ChunkInputKind::StorageOrRange,
            "SegScan",
        )?;
        let prefix_resource = scan_prefixes.unwrap_or(outputs[0].resource.0);
        let prefix_view_type = if scan_prefixes.is_some() {
            crate::types::view_array_with_size(&elem_ty, Type::Variable(0), crate::types::no_buffer())
        } else {
            outputs[0].view_type.clone()
        };
        let chunked_prefix = graph_ops::intern_chunked_resource_view(
            &mut entry.graph,
            prefix_resource,
            chunked.chunk_start,
            chunked.chunk_len,
            prefix_view_type.clone(),
            None,
        );
        {
            let effect = entry.graph.skeleton.effect_mut(site);
            effect.operand_nodes =
                chunked.views.iter().copied().chain(std::iter::once(chunked_prefix)).collect();
            let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &mut effect.kind else {
                return Err("selected scan site changed during phase construction".into());
            };
            op.form.pre = phase1_pre.clone();
            op.form.post = screma::Lambda::identity(vec![elem_ty.clone()]);
            op.result_state = vec![screma::ResultState {
                destination: SoacDestination::fresh().placed(crate::egir::types::SoacPlacement::OutputView),
            }];
        }
        entry.graph.retype_node(
            screma_result,
            Type::Constructed(TypeName::Tuple(1), vec![prefix_view_type]),
        );

        let reduce_result = graph_ops::emit_pending_soac(
            &mut entry.graph,
            block_id,
            owner.implementation(0),
            Soac::Screma(screma::Op {
                inputs: input_views.iter().map(|(_, input)| input.clone()).collect(),
                form: screma::ScremaForm {
                    pre: phase1_pre,
                    scans: Vec::new(),
                    reductions: vec![screma::Reduce {
                        operator: scan.operator.clone(),
                        neutral: scan.neutral.clone(),
                        commutative: false,
                    }],
                    post: screma::Lambda::identity(Vec::new()),
                },
                result_state: vec![screma::ResultState {
                    destination: SoacDestination::fresh(),
                }],
                state: screma::SemanticState::Serial,
            }),
            chunked.views.iter().copied().collect(),
            Type::Constructed(TypeName::Tuple(1), vec![elem_ty.clone()]),
            self.effect_ids,
            None,
        );
        let block_sum = entry.graph.intern_pure(
            PureOp::Project { index: 0 },
            smallvec![reduce_result],
            elem_ty.clone(),
            None,
        );
        let scratch_array_type =
            crate::types::view_array_with_size(&elem_ty, Type::Variable(0), crate::types::no_buffer());
        let block_sums_view = graph_ops::intern_resource_view(
            &mut entry.graph,
            block_sums_resource,
            scratch_array_type,
            None,
        );
        graph_ops::emit_storage_store(
            &mut entry.graph,
            block_id,
            block_sums_view,
            chunked.tid,
            block_sum,
            elem_ty.clone(),
            self.effect_ids,
            None,
        );

        for resource in [block_sums_resource, block_offsets_resource].into_iter().chain(scan_prefixes) {
            entry.resource_declarations.push(SemanticResourceDecl {
                resource: SemanticResourceRef(resource),
                role: crate::interface::StorageRole::Intermediate,
                elem_ty: elem_ty.clone(),
                size: self.resources[resource].size.clone(),
            });
        }

        let phase2 = ScanPhase2Spec {
            entry_name: entry.name.clone(),
            operator: combine_name.clone(),
            elem_ty: elem_ty.clone(),
            source_graph: &entry.graph,
            neutral,
            scratch: ScanScratch {
                block_sums: block_sums_resource,
                block_offsets: block_offsets_resource,
            },
            total_out: None,
        };
        let mut phase2 = phase2.build(self.semantic_ids, self.effect_ids)?;
        apply_manifest_resource_sizes(&mut phase2.body, self.resources);

        let swap_elem_ty = elem_ty.clone();
        let span = entry.span;
        let swap_region = self
            .define_callable(format!("{}_scan_op_swap", entry.name), |region, name| {
                synthesize_swap_wrapper(region, name, combine_name, swap_elem_ty, span)
            })?;
        let input_declarations = entry
            .resource_declarations
            .iter()
            .filter(|declaration| declaration.role == crate::interface::StorageRole::Input)
            .cloned()
            .collect();
        let post_phase = post_lambda.map(|pre| ScanPostPhaseSpec {
            pre,
            source_graph: &entry.graph,
            inputs: input_views,
            input_declarations,
            outputs: outputs
                .iter()
                .map(|output| ScanPostOutput {
                    resource: output.resource.0,
                    elem_ty: output.elem_type.clone(),
                    view_ty: output.view_type.clone(),
                })
                .collect(),
        });
        let phase3 = ScanPhase3Spec {
            entry_name: entry.name.clone(),
            swap_region,
            elem_ty,
            output_resource: prefix_resource,
            block_offsets: block_offsets_resource,
            width: total_threads,
            post: post_phase,
        };
        let mut phase3 = phase3.build(self.semantic_ids, self.effect_ids)?;
        apply_manifest_resource_sizes(&mut phase3.body, self.resources);

        phase1_resources = merge_scheduled_resources(
            &phase1_resources,
            &[schedule::ScheduledResource {
                resource: block_sums_resource,
                access: crate::ResourceAccess::Write,
            }],
        );
        if scan_prefixes.is_some() {
            phase1_resources = merge_scheduled_resources(
                &phase1_resources,
                &[schedule::ScheduledResource {
                    resource: prefix_resource,
                    access: crate::ResourceAccess::Write,
                }],
            );
        }
        Ok([BuiltPhase::new(entry, phase1_resources), phase2, phase3])
    }
}

fn synthesize_scan_input_function(
    region: RegionId,
    name: String,
    pre: screma::Lambda,
    callee: Option<String>,
    capture_types: Vec<Type<TypeName>>,
    result_type: Type<TypeName>,
    span: crate::ast::Span,
) -> SemanticFunc {
    let mut parameter_types = pre.parameter_types.clone();
    parameter_types.extend(capture_types);
    let params = lambda_ops::named_parameters(&parameter_types, "arg");
    let mut graph = EGraph::new();
    let arguments = lambda_ops::function_parameters(&mut graph, &params);
    let results = lambda_ops::emit_call(&mut graph, &pre, callee.as_deref(), arguments);
    let entry = graph.skeleton.entry;
    lambda_ops::finish_function(
        graph,
        entry,
        region,
        name,
        span,
        params,
        &[result_type],
        &results[..1],
    )
}

#[allow(clippy::too_many_arguments)]
fn synthesize_scan_post_function(
    region: RegionId,
    name: String,
    pre: screma::Lambda,
    pre_callee: Option<String>,
    post: screma::Lambda,
    post_callee: Option<String>,
    capture_types: Vec<Type<TypeName>>,
    span: crate::ast::Span,
) -> SemanticFunc {
    let mut element_types = vec![post.parameter_types[0].clone()];
    element_types.extend(pre.parameter_types.iter().cloned());
    let element_count = element_types.len();
    let pre_capture_count = pre.seg_body().map_or(0, |body| body.captures.len());
    let mut parameter_types = element_types;
    parameter_types.extend(capture_types);
    let params = lambda_ops::named_parameters(&parameter_types, "arg");
    let mut graph = EGraph::new();
    let arguments = lambda_ops::function_parameters(&mut graph, &params);

    let mut pre_arguments = arguments[1..element_count].to_vec();
    pre_arguments.extend_from_slice(&arguments[element_count..element_count + pre_capture_count]);
    let pre_results = lambda_ops::emit_call(&mut graph, &pre, pre_callee.as_deref(), pre_arguments);
    let mut post_arguments = vec![arguments[0]];
    post_arguments.extend_from_slice(&pre_results[1..]);
    post_arguments.extend_from_slice(&arguments[element_count + pre_capture_count..]);
    let post_results = lambda_ops::emit_call(&mut graph, &post, post_callee.as_deref(), post_arguments);
    let entry = graph.skeleton.entry;
    lambda_ops::finish_function(
        graph,
        entry,
        region,
        name,
        span,
        params,
        &post.result_types,
        &post_results,
    )
}
