//! Parallel scan candidate analysis, binding, and phase emission.

use super::kernel::cloneable_capture_inputs;
use super::model::REDUCE_PHASE1_WIDTH;
use super::*;
use crate::ast;
use crate::egir;
use crate::egir::soac::lambda as lambda_ops;
use crate::egir::types::{OperandRef, ResultBinding};
use crate::interface;
use crate::op;
use crate::ssa;
use crate::types;
use crate::IdSource;
use crate::ResourceAccess;

#[derive(Clone, Copy)]
pub(super) struct ScanScratch {
    pub block_sums: ResourceId,
    pub block_offsets: ResourceId,
}

/// Reduction results carried by the same product monoid as scan prefixes.
pub(super) struct ScanReductionOutputSpec<'a> {
    source_components: &'a [ValueId],
    component_offset: usize,
    component_types: &'a [Type<TypeName>],
    stores: &'a [super::reduce::RoutedReductionStore],
}
/// Build a single-invocation exclusive scan over block sums.
pub(super) struct ScanPhase2Spec<'a> {
    pub entry_name: String,
    pub operator: &'a Func<Semantic>,
    pub elem_ty: Type<TypeName>,
    pub source_graph: &'a EGraph,
    pub operator_captures: &'a [OperandRef],
    pub capture_inputs: &'a [SemanticResourceDecl],
    pub neutral: ValueId,
    pub scratch: ScanScratch,
    pub total_out: Option<ResourceId>,
    pub reduction_output: Option<ScanReductionOutputSpec<'a>>,
}

impl ScanPhase2Spec<'_> {
    pub(super) fn build(
        self,
        identities: &mut egir::program::ProgramIdentities,
        semantic_ids: &mut egir::program::SemanticOpIdSource,
        effect_ids: &mut IdSource<EffectToken>,
    ) -> Result<BuiltPhase, String> {
        use crate::egir::builder::EntryBuilder;

        let mut accesses = vec![
            SegResourceAccess::<ResourceId> {
                resource: self.scratch.block_sums,
                access: ResourceAccess::Read,
            },
            SegResourceAccess::<ResourceId> {
                resource: self.scratch.block_offsets,
                access: ResourceAccess::Write,
            },
        ];
        if let Some(resource) = self.total_out {
            accesses.push(SegResourceAccess::<ResourceId> {
                resource,
                access: ResourceAccess::Write,
            });
        }
        if let Some(output) = &self.reduction_output {
            accesses.extend(output.stores.iter().map(|store| SegResourceAccess::<ResourceId> {
                resource: store.output.0,
                access: ResourceAccess::Write,
            }));
        }

        accesses.extend(
            self.capture_inputs.iter().map(|declaration| SegResourceAccess::<ResourceId> {
                resource: declaration.resource.0,
                access: ResourceAccess::Read,
            }),
        );
        accesses = egir::ir::SegResourceAccess::merge(&accesses, &[]);

        let mut builder = EntryBuilder::new_compute(
            format!("{}_phase2_scan_sums", self.entry_name),
            (1, 1, 1),
            identities,
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
        for declaration in self.capture_inputs {
            builder.declare_input_storage_sized(
                declaration.resource.0,
                declaration.elem_ty.clone(),
                declaration.size.clone(),
            );
        }
        if let Some(len_out) = self.total_out {
            builder.declare_output_storage_sized(
                len_out,
                self.elem_ty.clone(),
                egir::program::LogicalSize::FixedBytes(4),
            );
        }
        if let Some(output) = &self.reduction_output {
            let mut declared = std::collections::HashSet::new();
            for store in output.stores {
                if declared.insert(store.output.0) {
                    builder.declare_output_storage_sized(
                        store.output.0,
                        store.output.1.clone(),
                        store.output.2.clone(),
                    );
                }
            }
        }

        let neutral = graph_ops::clone_pure_subgraph(self.source_graph, builder.graph_mut(), self.neutral)?;
        let operator_captures = self
            .operator_captures
            .iter()
            .map(|capture| {
                graph_ops::clone_operand_subgraph(self.source_graph, builder.graph_mut(), *capture)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let phase = self.emit_loop(&mut builder, neutral, &operator_captures);
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
        if let (Some(output), Some(total)) = (&self.reduction_output, phase.total) {
            let total_components =
                lambda_ops::unpack_results(builder.graph_mut(), total, output.component_types);
            let substitutions = output
                .source_components
                .iter()
                .copied()
                .zip(total_components.into_iter().skip(output.component_offset))
                .collect::<Vec<_>>();
            for store in output.stores {
                let cloned_value = graph_ops::clone_pure_subgraph_substituting(
                    self.source_graph,
                    builder.graph_mut(),
                    store.value,
                    &substitutions,
                )?;
                let (graph, effect_ids) = builder.construction_parts_mut();
                graph_ops::emit_resource_write(
                    graph,
                    phase.after,
                    store.output.0,
                    cloned_value,
                    &store.ty,
                    effect_ids,
                    None,
                )?;
            }
        }
        Ok(BuiltPhase::new(builder.build(), accesses))
    }

    fn emit_loop(
        &self,
        builder: &mut egir::builder::EntryBuilder,
        neutral: ValueId,
        operator_captures: &[OperandRef],
    ) -> ExclusiveScanPhase2 {
        let elem_ty = self.elem_ty.clone();
        let want_total = self.total_out.is_some() || self.reduction_output.is_some();
        let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
        let bool_ty = Type::Constructed(TypeName::Bool, vec![]);
        let arr_ty = types::view_array_with_size(&elem_ty, Type::Variable(0), types::no_buffer());
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
            args: graph.admit_flow_values([neutral, zero]),
        };
        let condition = graph_ops::intern_binop(graph, op::BinaryOperator::Less, index, len, bool_ty, None);
        graph.skeleton.blocks[header].term = SkeletonTerminator::CondBranch {
            cond: condition,
            then_target: body,
            then_args: vec![],
            else_target: after,
            else_args: if want_total { graph.admit_flow_values([accumulator]) } else { vec![] },
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
        let mut operator_arguments = vec![graph.operand_ref(accumulator), graph.operand_ref(value)];
        operator_arguments.extend_from_slice(operator_captures);
        let (_, result) = graph
            .emit_call(
                body,
                self.operator.region,
                self.operator.params(),
                self.operator.result(),
                operator_arguments,
                self.operator.effects(),
                None,
                None,
            )
            .expect("scan operator call must match its canonical boundary");
        let next_accumulator = graph_ops::pack_result_values(graph, &result)
            .expect("scan operator result is returned by value");
        graph.skeleton.blocks[body].term = SkeletonTerminator::Branch {
            target: continuation,
            args: graph.admit_flow_values([next_accumulator]),
        };
        let continued_accumulator =
            graph.add_block_param(continuation, graph.nodes[accumulator].ty.clone());
        let next_index = graph_ops::intern_binop(graph, op::BinaryOperator::Add, index, one, u32_ty, None);
        graph.skeleton.blocks[continuation].term = SkeletonTerminator::Branch {
            target: header,
            args: graph.admit_flow_values([continued_accumulator, next_index]),
        };
        let total = if want_total { Some(graph.add_block_param(after, elem_ty)) } else { None };
        builder.set_current_block(after);
        ExclusiveScanPhase2 { total, after, zero }
    }
}

struct ExclusiveScanPhase2 {
    total: Option<ValueId>,
    after: BlockId,
    zero: ValueId,
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
    pub inputs: Vec<(ValueId, egir::types::SoacInputType)>,
    pub input_declarations: Vec<SemanticResourceDecl>,
    pub outputs: Vec<ScanPostOutput>,
}

/// Build the chunked phase that prepends each block's exclusive offset and,
/// when needed, evaluates the original whole post-barrier dataflow.
pub(super) struct ScanPhase3Spec<'a> {
    pub entry_name: String,
    pub swap_region: FunctionId,
    pub elem_ty: Type<TypeName>,
    pub source_graph: &'a EGraph,
    pub operator_captures: Vec<OperandRef>,
    pub capture_inputs: Vec<SemanticResourceDecl>,
    /// Holds local prefixes on entry and global prefixes after offsetting.
    pub output_resource: ResourceId,
    pub block_offsets: ResourceId,
    pub width: u32,
    pub post: Option<ScanPostPhaseSpec<'a>>,
}

impl ScanPhase3Spec<'_> {
    pub(super) fn build(
        self,
        identities: &mut egir::program::ProgramIdentities,
        semantic_ids: &mut egir::program::SemanticOpIdSource,
        effect_ids: &mut IdSource<EffectToken>,
    ) -> Result<BuiltPhase, String> {
        use crate::egir::builder::EntryBuilder;

        let mut input_declarations = self.capture_inputs.clone();
        if let Some(post) = &self.post {
            for declaration in &post.input_declarations {
                if !input_declarations.iter().any(|existing| existing.resource == declaration.resource) {
                    input_declarations.push(declaration.clone());
                }
            }
        }
        let mut resources = vec![
            SegResourceAccess::<ResourceId> {
                resource: self.output_resource,
                access: ResourceAccess::ReadWrite,
            },
            SegResourceAccess::<ResourceId> {
                resource: self.block_offsets,
                access: ResourceAccess::Read,
            },
        ];
        resources = merge_scheduled_resources(&resources, &declared_input_resources(&input_declarations));
        if let Some(post) = &self.post {
            resources = merge_scheduled_resources(
                &resources,
                &post
                    .outputs
                    .iter()
                    .map(|output| SegResourceAccess::<ResourceId> {
                        resource: output.resource,
                        access: ResourceAccess::Write,
                    })
                    .collect::<Vec<_>>(),
            );
        }

        let mut builder = EntryBuilder::new_compute(
            format!("{}_phase3_add_offsets", self.entry_name),
            (self.width, 1, 1),
            identities,
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
        for declaration in &input_declarations {
            builder.declare_input_storage_sized(
                declaration.resource.0,
                declaration.elem_ty.clone(),
                declaration.size.clone(),
            );
        }
        if let Some(post) = &self.post {
            for output in &post.outputs {
                builder.declare_output_storage(output.resource, output.elem_ty.clone());
            }
        }
        builder.declare_intermediate_storage_sized(
            self.block_offsets,
            self.elem_ty.clone(),
            dispatch_worker_logical_size(&self.elem_ty),
        );
        let arr_ty = types::view_array_with_size(&self.elem_ty, Type::Variable(0), types::no_buffer());
        let block_offsets_view = builder.emit_storage_view(self.block_offsets, arr_ty.clone());
        let output_len = graph_ops::intern_resource_len(builder.graph_mut(), self.output_resource, None);
        let (tid, chunk_start, chunk_len) =
            emit_chunk_arithmetic(builder.graph_mut(), self.width, output_len)?;
        let offset_place = {
            let graph = builder.graph_mut();
            let view = graph.view_id(block_offsets_view);
            graph.add_view_index_place(view, tid, self.elem_ty.clone(), None)
        };
        let offset = builder.emit_load(offset_place, self.elem_ty.clone());
        let chunked_output = graph_ops::intern_chunked_resource_view(
            builder.graph_mut(),
            self.output_resource,
            chunk_start,
            chunk_len,
            arr_ty.clone(),
            None,
        );

        let mut operator_captures = vec![builder.graph_mut().operand_ref(offset)];
        operator_captures.extend(
            self.operator_captures
                .iter()
                .map(|capture| {
                    graph_ops::clone_operand_subgraph(self.source_graph, builder.graph_mut(), *capture)
                })
                .collect::<Result<Vec<_>, _>>()?,
        );
        builder.emit_pending_map_into(
            self.swap_region,
            chunked_output,
            arr_ty.clone(),
            self.elem_ty,
            operator_captures,
            chunked_output,
            arr_ty.clone(),
        );
        if let Some(mut post) = self.post {
            if let Some(body) = post.pre.seg_body_mut() {
                body.captures = body
                    .captures
                    .iter()
                    .map(|capture| {
                        graph_ops::clone_operand_subgraph(post.source_graph, builder.graph_mut(), *capture)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
            }
            let mut inputs = Vec::with_capacity(1 + post.inputs.len());
            inputs.push((chunked_output, egir::types::SoacInputType::array(arr_ty)));
            for (index, (source, input)) in post.inputs.into_iter().enumerate() {
                let cloned =
                    graph_ops::clone_pure_subgraph(post.source_graph, builder.graph_mut(), source)?;
                let chunked = chunk_view_like(
                    builder.graph_mut(),
                    cloned,
                    input.array.clone(),
                    chunk_start,
                    chunk_len,
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
    scans: Vec<screma::Scan>,
    reductions: Vec<screma::Reduce>,
    reduction_routing: super::reduce::ReductionRouting,
    operator_capture_inputs: Vec<SemanticResourceDecl>,
    post: screma::Lambda,
    input_views: Vec<(ValueId, egir::types::SoacInputType)>,
    results: Vec<ResultBinding<Type<TypeName>>>,
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
    entry: &egir::program::PlannedEntry,
    located: LocatedScrema<'_>,
    resources: &egir::program::LogicalResourceArena,
) -> error::Result<Option<ScanCandidate>> {
    debug_assert_eq!(
        super::capabilities::classify(located.op),
        super::capabilities::Strategy::Scan
    );
    let segment = located.segmented()?;
    let serial = located.serial_recipe();
    let scans = &located.op.form.scans;
    let reductions = &located.op.form.reductions;
    if scans.is_empty()
        || scans
            .iter()
            .flat_map(|scan| &scan.neutral)
            .chain(reductions.iter().flat_map(|reduction| &reduction.neutral))
            .any(|neutral| !can_clone_pure_subgraph(&entry.graph, *neutral, &[]))
    {
        return Ok(None);
    }
    let operator_captures = scans
        .iter()
        .map(|scan| &scan.operator)
        .chain(reductions.iter().map(|reduction| &reduction.operator))
        .flat_map(|operator| operator.captures().iter().copied())
        .collect::<Vec<_>>();
    let Some(operator_capture_inputs) = cloneable_capture_inputs(entry, &operator_captures) else {
        return Ok(None);
    };

    let operands = screma::ScremaOperands::decode(
        located.op,
        &located.effect.operands,
        located.effect.result.as_ref(),
    )?;
    let mut input_views = Vec::with_capacity(located.op.inputs.len());
    for (operand, input) in operands.inputs().zip(&located.op.inputs) {
        let Some(operand) = operand.operand.value() else {
            return Ok(None);
        };
        if !can_chunk_view(&entry.graph, operand) {
            return Ok(None);
        }
        input_views.push((operand, input.clone()));
    }

    let reduction_results = located.op.form.reduction_result_count();
    let results = operands.result_fields();
    let mut outputs = Vec::with_capacity(located.op.form.post.result_types.len());
    for post_field in 0..located.op.form.post.result_types.len() {
        let field = reduction_results + post_field;
        let Some(result) = results.get(field) else {
            return Ok(None);
        };
        let Some(resource) = entry.resource_for_result(result) else {
            return Ok(None);
        };
        outputs.push(ScanOutput {
            elem_type: located.op.form.post.result_types[post_field].clone(),
            resource,
            view_type: result.ty().clone(),
        });
    }
    if outputs.is_empty() {
        return Ok(None);
    }

    let component_types = scans
        .iter()
        .map(|scan| &scan.operator)
        .chain(reductions.iter().map(|reduction| &reduction.operator))
        .flat_map(|operator| operator.result_types.iter().cloned())
        .collect::<Vec<_>>();
    let scratch_type = lambda_ops::result_type(&component_types);
    if ssa::layout::type_byte_size(&scratch_type).is_none() {
        return Ok(None);
    }
    let direct_output = reductions.is_empty()
        && located.op.form.post.is_identity()
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
            let Some(capture) = capture.value() else {
                return Ok(None);
            };
            if !can_clone_pure_subgraph(&entry.graph, capture, &[]) {
                return Ok(None);
            }
        }
    }
    if results.len() < reduction_results + outputs.len() {
        return Ok(None);
    }
    let Some(reduction_routing) = super::reduce::analyze_reduction_routing(
        entry,
        located.op,
        &results[..reduction_results],
        resources,
    ) else {
        return Ok(None);
    };
    Ok(Some(ScanCandidate {
        site: located.site,
        owner: located.owner,
        scratch_type,
        serial,
        pre: located.op.form.pre.clone(),
        scans: scans.to_vec(),
        reductions: reductions.to_vec(),
        reduction_routing,
        operator_capture_inputs,
        post: located.op.form.post.clone(),
        input_views,
        results,
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
        mut entry: egir::program::PlannedEntry,
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
            scans,
            reductions,
            reduction_routing,
            operator_capture_inputs,
            post,
            input_views,
            results: screma_results,
            outputs,
            direct_output,
            phase1_width: total_threads,
            segment,
        } = candidate;
        let block_id = site.block;
        let component_types = scans
            .iter()
            .map(|scan| &scan.operator)
            .chain(reductions.iter().map(|reduction| &reduction.operator))
            .flat_map(|operator| operator.result_types.iter().cloned())
            .collect::<Vec<_>>();
        let scan_component_count = scans.iter().map(|scan| scan.operator.result_types.len()).sum::<usize>();
        let component_count = component_types.len();
        let operator_captures = scans
            .iter()
            .map(|scan| &scan.operator)
            .chain(reductions.iter().map(|reduction| &reduction.operator))
            .flat_map(|operator| operator.captures().iter().copied())
            .collect::<Vec<_>>();
        let original_operators = scans
            .iter()
            .map(|scan| scan.operator.clone())
            .chain(reductions.iter().map(|reduction| reduction.operator.clone()))
            .collect::<Vec<_>>();
        let original_operator_functions = original_operators
            .iter()
            .map(|operator| {
                let region = operator.seg_body().expect("scan operator has a callable region").region;
                self.callable(region).clone()
            })
            .collect::<Vec<_>>();
        let phase_operator = if reductions.is_empty() && scans.len() == 1 && component_count == 1 {
            scans[0].operator.clone()
        } else {
            let capture_types = operator_captures
                .iter()
                .map(|capture| {
                    entry.graph.nodes[capture.value().expect("scan operator capture is a value or view")]
                        .ty
                        .clone()
                })
                .collect::<Vec<_>>();
            let wrapper_component_types = component_types.clone();
            let wrapper_scratch_type = elem_ty.clone();
            let span = entry.span;
            let region = self.define_callable(
                format!("{}_scan_product_operator", entry.name),
                move |region, name| {
                    synthesize_packed_operator_function(
                        region,
                        name,
                        original_operators,
                        original_operator_functions,
                        wrapper_component_types,
                        capture_types,
                        wrapper_scratch_type,
                        span,
                    )
                },
            )?;
            screma::Lambda::region(
                SegBody {
                    region,
                    captures: operator_captures.clone(),
                },
                vec![elem_ty.clone(), elem_ty.clone()],
                vec![elem_ty.clone()],
            )
        };
        let operator_region =
            phase_operator.seg_body().expect("parallel scan phase operator has a region").region;
        let combine_region = operator_region;
        let neutrals = scans
            .iter()
            .flat_map(|scan| scan.neutral.iter().copied())
            .chain(reductions.iter().flat_map(|reduction| reduction.neutral.iter().copied()))
            .collect::<Vec<_>>();
        let neutral = lambda_ops::pack_results(&mut entry.graph, &neutrals, &component_types);
        let phase_scan = screma::Scan {
            operator: phase_operator,
            neutral: vec![neutral],
        };
        let phase1_pre = if pre.result_types.len() == 1 && component_count == 1 {
            pre.clone()
        } else {
            let captures = pre
                .seg_body()
                .into_iter()
                .flat_map(|body| body.captures.iter().copied())
                .collect::<Vec<_>>();
            let capture_types = captures
                .iter()
                .map(|capture| {
                    entry.graph.nodes[capture.value().expect("scan pre capture is a value or view")]
                        .ty
                        .clone()
                })
                .collect::<Vec<_>>();
            let source = pre.clone();
            let source_function = source.seg_body().map(|body| self.callable(body.region).clone());
            let parameter_types = pre.parameter_types.clone();
            let result_type = elem_ty.clone();
            let span = entry.span;
            let region =
                self.define_callable(format!("{}_scan_pre", entry.name), move |region, name| {
                    synthesize_scan_input_function(
                        region,
                        name,
                        source,
                        source_function,
                        capture_types,
                        component_count,
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
            let capture_types = captures
                .iter()
                .map(|capture| {
                    entry.graph.nodes[capture.value().expect("scan post capture is a value or view")]
                        .ty
                        .clone()
                })
                .collect::<Vec<_>>();
            let source_pre = pre.clone();
            let source_post = post.clone();
            let source_pre_function = source_pre.seg_body().map(|body| self.callable(body.region).clone());
            let source_post_function =
                source_post.seg_body().map(|body| self.callable(body.region).clone());
            let mut parameter_types = vec![elem_ty.clone()];
            parameter_types.extend(pre.parameter_types.iter().cloned());
            let result_types = post.result_types.clone();
            let post_component_types = component_types.clone();
            let post_scratch_type = elem_ty.clone();
            let span = entry.span;
            let region =
                self.define_callable(format!("{}_scan_post", entry.name), move |region, name| {
                    synthesize_scan_post_function(
                        region,
                        name,
                        source_pre,
                        source_pre_function,
                        source_post,
                        source_post_function,
                        post_component_types,
                        scan_component_count,
                        post_scratch_type,
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

        let reduction_component_types = reductions
            .iter()
            .flat_map(|reduction| reduction.operator.result_types.iter().cloned())
            .collect::<Vec<_>>();
        let reduction_result_components = screma_results[..reduction_component_types.len()]
            .iter()
            .map(|result| {
                graph_ops::pack_result_values(&mut entry.graph, result)
                    .expect("scan reduction component is returned by value")
            })
            .collect::<Vec<_>>();
        let reduction_stores = reduction_routing.stores;
        let reduction_output_declarations = reduction_stores.iter().map(|store| store.output.clone()).fold(
            Vec::new(),
            |mut outputs, output| {
                if !outputs.iter().any(|(resource, _, _)| *resource == output.0) {
                    outputs.push(output);
                }
                outputs
            },
        );
        let moved_reduction_outputs = reduction_output_declarations
            .iter()
            .map(|(resource, _, _)| *resource)
            .collect::<std::collections::HashSet<_>>();
        entry.outputs.retain(|output| {
            output.resource.is_none_or(|resource| !moved_reduction_outputs.contains(&resource.0))
        });
        entry.resource_declarations.retain(|declaration| {
            declaration.role != interface::StorageRole::Output
                || !moved_reduction_outputs.contains(&declaration.resource.0)
        });
        let mut phase1_resources = merge_scheduled_resources(
            &declared_input_resources(&entry.resource_declarations),
            &segmented_resources(&segment),
        );
        phase1_resources.retain_mut(|access| {
            if !moved_reduction_outputs.contains(&access.resource) {
                return true;
            }
            match access.access {
                ResourceAccess::Read => true,
                ResourceAccess::Write => false,
                ResourceAccess::ReadWrite => {
                    access.access = ResourceAccess::Read;
                    true
                }
            }
        });
        if !direct_output {
            phase1_resources
                .retain(|resource| !outputs.iter().any(|output| output.resource.0 == resource.resource));
        }
        let input_view_data =
            input_views.iter().map(|(node, input)| (*node, input.array.clone())).collect::<Vec<_>>();
        make_screma_serial(&mut entry.graph, serial);
        let chunked = chunk_soac_inputs(&mut entry.graph, &input_view_data, total_threads, "SegScan")?;
        let prefix_resource = scan_prefixes.unwrap_or(outputs[0].resource.0);
        let prefix_view_type = if scan_prefixes.is_some() {
            types::view_array_with_size(&elem_ty, Type::Variable(0), types::no_buffer())
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
        let prefix_binding = graph_ops::bind_physical_result_value(
            &mut entry.graph,
            prefix_view_type.clone(),
            chunked_prefix,
        );
        let phase1_result_ty = Type::Constructed(TypeName::Tuple(1), vec![prefix_view_type]);
        let phase1_result = egir::types::ResultBinding::product(phase1_result_ty, [prefix_binding]);
        {
            let operands = chunked.views.iter().map(|value| entry.graph.operand_ref(*value)).collect();
            let effect = entry.graph.skeleton.effect_mut(site);
            effect.operands = operands;
            let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &mut effect.kind else {
                return Err("selected scan site changed during phase construction".into());
            };
            op.form.pre = phase1_pre.clone();
            op.form.scans = vec![phase_scan.clone()];
            op.form.reductions.clear();
            op.form.post = screma::Lambda::identity(vec![elem_ty.clone()]);
            op.result_state = vec![screma::ResultState {
                ownership: types::SoacOwnership::Fresh,
            }];
            effect.result = Some(phase1_result);
        }

        let reduce_operands = chunked.views.iter().map(|view| entry.graph.operand_ref(*view)).collect();
        let reduce_result = graph_ops::alloc_by_value_effect_result(
            &mut entry.graph,
            Type::Constructed(TypeName::Tuple(1), vec![elem_ty.clone()]),
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
                        operator: phase_scan.operator.clone(),
                        neutral: phase_scan.neutral.clone(),
                        commutative: false,
                    }],
                    post: screma::Lambda::identity(Vec::new()),
                },
                result_state: vec![screma::ResultState {
                    ownership: types::SoacOwnership::Fresh,
                }],
                state: screma::SemanticState::Serial,
            }),
            reduce_operands,
            reduce_result,
            self.effect_ids,
            None,
        );
        let block_sum_result = reduce_result.field(0).expect("the block reduction has one logical result");
        let block_sum = graph_ops::pack_result_values(&mut entry.graph, &block_sum_result)
            .expect("the block reduction result is returned by value");
        let scratch_array_type =
            types::view_array_with_size(&elem_ty, Type::Variable(0), types::no_buffer());
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
                role: interface::StorageRole::Intermediate,
                elem_ty: elem_ty.clone(),
                size: self.resources[resource].size.clone(),
            });
        }

        let combine_function = self.callable(combine_region).clone();
        let phase2 = ScanPhase2Spec {
            entry_name: entry.name.clone(),
            operator: &combine_function,
            elem_ty: elem_ty.clone(),
            source_graph: &entry.graph,
            operator_captures: &operator_captures,
            capture_inputs: &operator_capture_inputs,
            neutral,
            scratch: ScanScratch {
                block_sums: block_sums_resource,
                block_offsets: block_offsets_resource,
            },
            total_out: None,
            reduction_output: (!reduction_result_components.is_empty()).then_some(
                ScanReductionOutputSpec {
                    source_components: &reduction_result_components,
                    component_offset: scan_component_count,
                    component_types: &component_types,
                    stores: &reduction_stores,
                },
            ),
        };
        let mut phase2 = phase2.build(&mut self.identities, self.semantic_ids, self.effect_ids)?;
        apply_manifest_resource_sizes(&mut phase2.body, self.resources);

        let swap_elem_ty = elem_ty.clone();
        let operator_capture_types = operator_captures
            .iter()
            .map(|capture| {
                entry.graph.nodes[capture.value().expect("scan operator capture is a value or view")]
                    .ty
                    .clone()
            })
            .collect::<Vec<_>>();
        let span = entry.span;
        let swap_region =
            self.define_callable(format!("{}_scan_op_swap", entry.name), |region, name| {
                synthesize_swap_wrapper(
                    region,
                    name,
                    &combine_function,
                    swap_elem_ty,
                    operator_capture_types,
                    span,
                )
            })?;
        let input_declarations = entry
            .resource_declarations
            .iter()
            .filter(|declaration| declaration.role == interface::StorageRole::Input)
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
            source_graph: &entry.graph,
            operator_captures,
            capture_inputs: operator_capture_inputs,
            output_resource: prefix_resource,
            block_offsets: block_offsets_resource,
            width: total_threads,
            post: post_phase,
        };
        let mut phase3 = phase3.build(&mut self.identities, self.semantic_ids, self.effect_ids)?;
        apply_manifest_resource_sizes(&mut phase3.body, self.resources);

        phase1_resources = merge_scheduled_resources(
            &phase1_resources,
            &[SegResourceAccess::<ResourceId> {
                resource: block_sums_resource,
                access: ResourceAccess::Write,
            }],
        );
        if scan_prefixes.is_some() {
            phase1_resources = merge_scheduled_resources(
                &phase1_resources,
                &[SegResourceAccess::<ResourceId> {
                    resource: prefix_resource,
                    access: ResourceAccess::Write,
                }],
            );
        }
        entry.outputs.clear();
        Ok([BuiltPhase::new(entry, phase1_resources), phase2, phase3])
    }
}

fn synthesize_packed_operator_function(
    region: FunctionId,
    name: String,
    operators: Vec<screma::Lambda>,
    operator_functions: Vec<Func<Semantic>>,
    component_types: Vec<Type<TypeName>>,
    capture_types: Vec<Type<TypeName>>,
    scratch_type: Type<TypeName>,
    span: ast::Span,
) -> Func<Semantic> {
    let mut parameter_types = vec![scratch_type.clone(), scratch_type.clone()];
    parameter_types.extend(capture_types);
    let params = lambda_ops::named_parameters(&parameter_types, "arg");
    let mut graph = EGraph::new();
    let arguments = lambda_ops::function_parameters(&mut graph, &params);
    let left = lambda_ops::unpack_results(
        &mut graph,
        arguments[0].value().expect("packed operator parameter is a value"),
        &component_types,
    );
    let right = lambda_ops::unpack_results(
        &mut graph,
        arguments[1].value().expect("packed operator parameter is a value"),
        &component_types,
    );
    let mut component_cursor = 0;
    let mut capture_cursor = 2;
    let mut results = Vec::with_capacity(component_types.len());
    for (operator, function) in operators.iter().zip(&operator_functions) {
        let component_end = component_cursor + operator.result_types.len();
        let capture_end = capture_cursor + operator.capture_count();
        let mut operator_arguments = left[component_cursor..component_end]
            .iter()
            .map(|value| graph.operand_ref(*value))
            .collect::<Vec<_>>();
        operator_arguments
            .extend(right[component_cursor..component_end].iter().map(|value| graph.operand_ref(*value)));
        operator_arguments.extend_from_slice(&arguments[capture_cursor..capture_end]);
        let entry = graph.skeleton.entry;
        let operator_results =
            lambda_ops::emit_call(&mut graph, entry, operator, Some(function), operator_arguments);
        results.extend(lambda_ops::result_argument_values(&mut graph, &operator_results));
        component_cursor = component_end;
        capture_cursor = capture_end;
    }
    debug_assert_eq!(component_cursor, component_types.len());
    debug_assert_eq!(capture_cursor, arguments.len());
    let packed = lambda_ops::pack_results(&mut graph, &results, &component_types);
    let entry = graph.skeleton.entry;
    lambda_ops::finish_function(
        graph,
        entry,
        region,
        name,
        span,
        params,
        &[scratch_type],
        &[packed],
    )
}

#[allow(clippy::too_many_arguments)]
fn synthesize_scan_input_function(
    region: FunctionId,
    name: String,
    pre: screma::Lambda,
    pre_function: Option<Func<Semantic>>,
    capture_types: Vec<Type<TypeName>>,
    component_count: usize,
    result_type: Type<TypeName>,
    span: ast::Span,
) -> Func<Semantic> {
    let mut parameter_types = pre.parameter_types.clone();
    parameter_types.extend(capture_types);
    let params = lambda_ops::named_parameters(&parameter_types, "arg");
    let mut graph = EGraph::new();
    let arguments = lambda_ops::function_parameters(&mut graph, &params);
    let entry = graph.skeleton.entry;
    let results = lambda_ops::emit_call(&mut graph, entry, &pre, pre_function.as_ref(), arguments);
    let results = lambda_ops::result_argument_values(&mut graph, &results);
    let packed = lambda_ops::pack_results(
        &mut graph,
        &results[..component_count],
        &pre.result_types[..component_count],
    );
    let entry = graph.skeleton.entry;
    lambda_ops::finish_function(
        graph,
        entry,
        region,
        name,
        span,
        params,
        &[result_type],
        &[packed],
    )
}

#[allow(clippy::too_many_arguments)]
fn synthesize_scan_post_function(
    region: FunctionId,
    name: String,
    pre: screma::Lambda,
    pre_function: Option<Func<Semantic>>,
    post: screma::Lambda,
    post_function: Option<Func<Semantic>>,
    component_types: Vec<Type<TypeName>>,
    scan_component_count: usize,
    scratch_type: Type<TypeName>,
    capture_types: Vec<Type<TypeName>>,
    span: ast::Span,
) -> Func<Semantic> {
    let mut element_types = vec![scratch_type];
    element_types.extend(pre.parameter_types.iter().cloned());
    let element_count = element_types.len();
    let pre_capture_count = pre.capture_count();
    let mut parameter_types = element_types;
    parameter_types.extend(capture_types);
    let params = lambda_ops::named_parameters(&parameter_types, "arg");
    let mut graph = EGraph::new();
    let arguments = lambda_ops::function_parameters(&mut graph, &params);

    let mut pre_arguments = arguments[1..element_count].to_vec();
    pre_arguments.extend_from_slice(&arguments[element_count..element_count + pre_capture_count]);
    let entry = graph.skeleton.entry;
    let pre_results = lambda_ops::emit_call(&mut graph, entry, &pre, pre_function.as_ref(), pre_arguments);
    let pre_results = lambda_ops::result_argument_values(&mut graph, &pre_results);
    let prefix_components = lambda_ops::unpack_results(
        &mut graph,
        arguments[0].value().expect("scan post prefix is a value"),
        &component_types,
    );
    let mut post_arguments = prefix_components[..scan_component_count]
        .iter()
        .map(|value| graph.operand_ref(*value))
        .collect::<Vec<_>>();
    post_arguments
        .extend(pre_results[component_types.len()..].iter().map(|value| graph.operand_ref(*value)));
    post_arguments.extend_from_slice(&arguments[element_count + pre_capture_count..]);
    let post_results =
        lambda_ops::emit_call(&mut graph, entry, &post, post_function.as_ref(), post_arguments);
    let post_results = lambda_ops::result_argument_values(&mut graph, &post_results);
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
