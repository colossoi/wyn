//! Parallel reduction candidate analysis, binding, and phase emission.

use super::kernel::{capture_declarations, cloneable_capture_inputs};
use super::model::{REDUCE_PHASE1_WIDTH, REDUCE_PHASE2_WIDTH};
use super::planning::{OperationRef, ScratchRef};
use super::*;
use crate::egir;
use crate::egir::soac::lambda as lambda_ops;
use crate::egir::types::{OperandRef, ResultBinding};
use crate::interface;
use crate::op;
use crate::types;
use crate::ResourceAccess;
use wyn_base::IdSource;
#[derive(Debug)]
pub struct ReduceRecipe<R> {
    pub operation: OperationRef,
    pub accumulators: Vec<ReductionAccumulator<R>>,
    pub routing: ReductionRouting,
}

#[derive(Debug)]
pub struct ReductionAccumulator<R> {
    pub capture_inputs: Vec<ResourceId>,
    pub partials: R,
}

#[derive(Debug)]
pub struct ReductionRouting {
    pub stores: Vec<ReductionStore>,
}

#[derive(Debug)]
pub struct ReductionStore {
    pub value: ValueId,
    pub accumulators: Vec<usize>,
    pub destination: ResourceId,
}

impl ReductionRouting {
    pub(super) fn resolve(
        self,
        entry: &egir::program::PlannedEntry,
        resources: &LogicalResourceArena,
    ) -> ParallelizeResult<Vec<RoutedReductionStore>> {
        self.stores
            .into_iter()
            .map(|store| {
                let resource = &resources[store.destination];
                Ok(RoutedReductionStore {
                    value: store.value,
                    ty: entry.graph.nodes[store.value].ty().clone(),
                    accumulators: store.accumulators,
                    output: (
                        store.destination,
                        resource.elem_ty().clone(),
                        resource.size().cloned().ok_or_else(|| {
                            ParallelizeError::Invalid("reduction output has no logical size".into())
                        })?,
                    ),
                })
            })
            .collect()
    }
}

#[derive(Clone)]
pub(super) struct RoutedReductionStore {
    pub value: ValueId,
    pub ty: Type<TypeName>,
    accumulators: Vec<usize>,
    pub output: (ResourceId, Type<TypeName>, egir::program::LogicalSize),
}

/// Transfer publication ownership to the phase that computes the final totals.
pub(super) fn retire_reduction_outputs(
    entry: &mut egir::program::PlannedEntry,
    stores: &[&RoutedReductionStore],
) -> HashSet<ResourceId> {
    let moved = stores.iter().map(|store| store.output.0).collect::<HashSet<_>>();
    let graph = &entry.graph;
    let sites = graph
        .skeleton
        .blocks
        .iter()
        .flat_map(|(block, body)| {
            body.side_effects.iter().enumerate().filter_map(move |(index, effect)| {
                let SideEffectKind::Effect(EffectOp::Store { place }) = effect.kind() else {
                    return None;
                };
                let egir::ir::PlaceOp::ViewIndex { view, .. } = graph.place(*place).op() else {
                    return None;
                };
                let resource = graph_ops::extract_storage_view_source(graph, view.value())?;
                let [operand] = effect.operands() else {
                    return None;
                };
                let value = graph.canonical_value(operand.value()?);
                stores
                    .iter()
                    .any(|store| {
                        store.output.0 == resource.0 && graph.canonical_value(store.value) == value
                    })
                    .then_some(SideEffectSite { block, index })
            })
        })
        .collect::<Vec<_>>();
    for site in sites.into_iter().rev() {
        entry.graph.skeleton.remove_effect_splicing_dependencies(site);
    }
    entry.outputs.retain(|output| output.resource.is_none_or(|resource| !moved.contains(&resource.0)));
    entry.internal_results.retain(|output| !moved.contains(&output.resource.0));
    entry.resource_declarations.retain_mut(|declaration| {
        if !moved.contains(&declaration.resource.0) {
            return true;
        }
        match declaration.role {
            interface::StorageRole::Output => false,
            interface::StorageRole::InputOutput => {
                declaration.role = interface::StorageRole::Input;
                true
            }
            _ => true,
        }
    });
    moved
}

struct EmissionAccumulator {
    component_types: Vec<Type<TypeName>>,
    scratch_type: Type<TypeName>,
    operator: Func<Semantic>,
    operator_captures: Vec<OperandRef>,
    capture_inputs: Vec<SemanticResourceDecl>,
    neutrals: Vec<ValueId>,
    stores: Vec<RoutedReductionStore>,
    partial: ResourceId,
}

fn analyze_reduction_operators(
    entry: &egir::program::PlannedEntry,
    analysis: &graph_ops::GraphAnalysis<'_, Semantic>,
    located: LocatedScrema<'_>,
) -> Option<Vec<ReductionAccumulator<ScratchRef>>> {
    located
        .op
        .form
        .reductions
        .iter()
        .enumerate()
        .map(|(slot, reduction)| {
            if reduction.neutral.iter().any(|neutral| !can_clone_pure_subgraph(&entry.graph, *neutral, &[]))
                || reduction.operator.seg_body().is_none()
            {
                return None;
            }
            let capture_inputs = cloneable_capture_inputs(entry, analysis, reduction.operator.captures())?;
            let partials = ScratchRef::dispatch(
                located.owner,
                CompilerResourceKind::ReducePartial,
                slot,
                lambda_ops::result_type(&reduction.operator.result_types),
            )
            .ok()?;
            Some(ReductionAccumulator {
                capture_inputs,
                partials,
            })
        })
        .collect()
}

pub(super) fn analyze_reduction_routing(
    entry: &egir::program::PlannedEntry,
    analysis: &graph_ops::GraphAnalysis<'_, Semantic>,
    op: &screma::Op<Semantic>,
    results: &[ResultBinding<Type<TypeName>>],
    resources: &egir::program::LogicalResourceArena,
) -> Option<ReductionRouting> {
    let field_accumulators = op
        .form
        .reductions
        .iter()
        .enumerate()
        .flat_map(|(accumulator, reduction)| std::iter::repeat_n(accumulator, reduction.neutral.len()))
        .collect::<Vec<_>>();
    let mut stores = Vec::new();
    let result_values = results.iter().map(ResultBinding::values).collect::<Vec<_>>();
    let supplied = result_values.iter().flatten().copied().collect::<Vec<_>>();
    for (resource, route) in entry.resource_routes() {
        let resource = resource.0;
        entry
            .resource_declarations
            .iter()
            .find(|declaration| declaration.role.writes() && declaration.resource.0 == resource)?;
        resources[resource].size()?;
        let destination = resource;
        let value = route.source.value;
        let producers = graph_ops::value_producer_closure(analysis, [value]);
        let mut accumulator_dependencies = Vec::new();
        for (field, values) in result_values.iter().enumerate() {
            if values.iter().any(|result| producers.values().contains(result)) {
                let accumulator = *field_accumulators.get(field)?;
                accumulator_dependencies.push(accumulator);
            }
        }
        accumulator_dependencies.sort_unstable();
        accumulator_dependencies.dedup();
        if accumulator_dependencies.is_empty() {
            continue;
        }
        if !can_clone_pure_subgraph(&entry.graph, value, &supplied) {
            return None;
        }
        stores.push(ReductionStore {
            value,
            accumulators: accumulator_dependencies,
            destination,
        });
    }
    if !(0..op.form.reductions.len())
        .all(|accumulator| stores.iter().any(|store| store.accumulators.contains(&accumulator)))
    {
        return None;
    }
    Some(ReductionRouting { stores })
}

pub(super) fn construct_reduce_recipe(
    entry: &egir::program::PlannedEntry,
    located: LocatedScrema<'_>,
    resources: &egir::program::LogicalResourceArena,
) -> ParallelizeResult<Option<ReduceRecipe<ScratchRef>>> {
    debug_assert_eq!(
        super::capabilities::classify(located.op),
        super::capabilities::Strategy::Reduce
    );
    let side_effect = located.effect;
    let reduction_results = located.op.form.layout().reduction_result_count();
    let n_maps = located.op.form.post.result_types.len();
    let operands =
        screma::ScremaOperands::decode(located.op, &side_effect.operands, side_effect.result.as_ref())?;
    for input in operands.inputs() {
        let Some(input) = input.operand.value() else {
            return Ok(None);
        };
        if !can_chunk_view(&entry.graph, input) {
            return Ok(None);
        }
    }
    let results = operands.result_fields();
    for index in 0..n_maps {
        let Some(result) = results.get(reduction_results + index) else {
            return Ok(None);
        };
        let Some(_) = entry.resource_for_result(result) else {
            return Ok(None);
        };
    }

    if results.len() < reduction_results {
        return Ok(None);
    }
    let reduction_values = results[..reduction_results].to_vec();
    let analysis = graph_ops::GraphAnalysis::new(&entry.graph);
    let Some(accumulators) = analyze_reduction_operators(entry, &analysis, located) else {
        return Ok(None);
    };
    let Some(routing) =
        analyze_reduction_routing(entry, &analysis, located.op, &reduction_values, resources)
    else {
        return Ok(None);
    };
    if routing.stores.iter().any(|store| store.accumulators.len() != 1) {
        return Ok(None);
    }
    Ok(Some(ReduceRecipe {
        operation: located.reference(),
        accumulators,
        routing,
    }))
}

impl KernelPlanBuilder<'_> {
    pub(super) fn emit_reduce_entry(
        &mut self,
        mut entry: egir::program::PlannedEntry,
        recipe: ReduceRecipe<ResourceId>,
    ) -> ParallelizeResult<(BuiltPhase, Vec<BuiltPhase>)> {
        let located = recipe.operation.screma(&entry)?;
        let site = located.site;
        let serial = recipe.operation;
        let segment = located.segmented()?;
        let phase1_width = REDUCE_PHASE1_WIDTH;
        let phase2_width = REDUCE_PHASE2_WIDTH;
        let operands = screma::ScremaOperands::decode(
            located.op,
            &located.effect.operands,
            located.effect.result.as_ref(),
        )?;
        let input_view_data = operands
            .inputs()
            .map(|input| {
                let value = input
                    .operand
                    .value()
                    .ok_or_else(|| ParallelizeError::Invalid("reduction input is not a value".into()))?;
                Ok((value, entry.graph.nodes[value].ty().clone()))
            })
            .collect::<ParallelizeResult<Vec<_>>>()?;
        let results = operands.result_fields();
        let reduction_results = located.op.form.layout().reduction_result_count();
        let map_outputs = results[reduction_results..]
            .iter()
            .map(|result| {
                let resource = entry.resource_for_result(result).ok_or_else(|| {
                    ParallelizeError::Invalid("mapped reduction result has no resource".into())
                })?;
                Ok((result.clone(), resource))
            })
            .collect::<ParallelizeResult<Vec<_>>>()?;
        let screma_results = results[..reduction_results].to_vec();
        let mut stores = recipe.routing.resolve(&entry, self.resources)?;
        let accumulators = recipe
            .accumulators
            .into_iter()
            .zip(&located.op.form.reductions)
            .enumerate()
            .map(|(index, (accumulator, reduction))| {
                let operator = reduction
                    .operator
                    .seg_body()
                    .ok_or_else(|| ParallelizeError::Invalid("reduction operator has no region".into()))?;
                let routed = stores.extract_if(.., |store| store.accumulators == [index]).collect();
                Ok(EmissionAccumulator {
                    component_types: reduction.operator.result_types.clone(),
                    scratch_type: self.resources[accumulator.partials].elem_ty().clone(),
                    operator: self.callable(operator.region)?.clone(),
                    operator_captures: reduction.operator.captures().to_vec(),
                    capture_inputs: capture_declarations(&entry, &accumulator.capture_inputs)?,
                    neutrals: reduction.neutral.clone(),
                    stores: routed,
                    partial: accumulator.partials,
                })
            })
            .collect::<ParallelizeResult<Vec<_>>>()?;
        let mut phase1_resources = merge_scheduled_resources(
            &declared_input_resources(&entry.resource_declarations),
            &segmented_resources(&segment),
        );
        let block_id = site.block;
        let total_threads = phase1_width;
        let n_accs = accumulators.len();
        // 3. Chunk every input view and bind mapped results to their matching
        // resource slices.
        let chunked = chunk_soac_inputs(&mut entry.graph, &input_view_data, total_threads, "SegRed")?;
        let chunk_start = chunked.chunk_start;
        let chunk_len = chunked.chunk_len;
        {
            let views = chunked.views.iter().map(|view| entry.graph.operand_ref(*view)).collect::<Vec<_>>();
            let se = entry.graph.skeleton.effect_mut(site);
            for (i, &new_view) in views.iter().enumerate() {
                se.operands[i] = new_view;
            }
        }
        for (map_index, (result, resource)) in map_outputs.iter().enumerate() {
            let chunked_view = graph_ops::intern_chunked_resource_view(
                &mut entry.graph,
                resource.0,
                chunk_start,
                chunk_len,
                result.ty().clone(),
                None,
            );
            let destination = graph_ops::bind_result_to_view(&mut entry.graph, result, chunked_view)?;
            graph_ops::rebind_result_value_references(&mut entry.graph, result, &destination)
                .map_err(|error| format!("SegRed map output {map_index}: {error}"))?;
        }

        // 5. Phase 1 stores each reduction operator's complete accumulator to
        // one partial buffer. Multi-component operators keep their canonical
        // flattened result ABI, but their scratch element is a packed tuple.
        let mut result_field = 0;
        let mut accumulator_values = Vec::with_capacity(n_accs);
        for accumulator in &accumulators {
            let end = result_field + accumulator.component_types.len();
            let results = screma_results[result_field..end].to_vec();
            let components = results
                .iter()
                .map(|result| {
                    graph_ops::pack_result_values(&mut entry.graph, result).map_err(ParallelizeError::from)
                })
                .collect::<ParallelizeResult<Vec<_>>>()?;
            result_field += components.len();
            let packed =
                lambda_ops::pack_results(&mut entry.graph, &components, &accumulator.component_types);
            accumulator_values.push((results, packed));
        }
        for (accumulator, (_, accumulator_value)) in accumulators.iter().zip(&accumulator_values) {
            let elem_ty = accumulator.scratch_type.clone();
            let arr_ty = types::view_array_with_size(&elem_ty, Type::Variable(0), types::no_buffer());
            let partials_view =
                graph_ops::intern_resource_view(&mut entry.graph, accumulator.partial, arr_ty, None);
            graph_ops::emit_storage_store(
                &mut entry.graph,
                block_id,
                partials_view,
                chunked.tid,
                *accumulator_value,
                elem_ty,
                self.effect_ids,
                None,
            );
            entry.resource_declarations.push(SemanticResourceDecl {
                resource: SemanticResourceRef(accumulator.partial),
                role: interface::StorageRole::Intermediate,
            });
        }
        let stores = accumulators.iter().flat_map(|accumulator| &accumulator.stores).collect::<Vec<_>>();
        let moved = retire_reduction_outputs(&mut entry, &stores);

        // 6. Synthesize one phase 2 entry per accumulator. Dropping the phase-1
        // stores leaves their pure place/value subgraphs available for projection.
        let mut phase2s = Vec::with_capacity(n_accs);
        for (acc_i, (accumulator, (accumulator_results, _))) in
            accumulators.iter().zip(accumulator_values).enumerate()
        {
            let phase2_name = if n_accs == 1 {
                format!("{}_phase2_combine", entry.name)
            } else {
                format!("{}_phase2_combine_{}", entry.name, acc_i)
            };
            let combine = ReduceCombineSpec {
                name: phase2_name,
                operator: &accumulator.operator,
                component_types: &accumulator.component_types,
                elem_ty: accumulator.scratch_type.clone(),
                source_graph: &entry.graph,
                operator_captures: &accumulator.operator_captures,
                capture_inputs: &accumulator.capture_inputs,
                neutrals: &accumulator.neutrals,
                partials: accumulator.partial,
                accumulator_results: &accumulator_results,
                output_stores: &accumulator.stores,
                width: phase2_width,
            };
            let phase2 = combine.build(&mut self.identities, self.semantic_ids, self.effect_ids)?;
            phase2s.push(phase2);
        }
        // Scheduling consumed the semantic SegRed. Phase 1 is now an ordinary
        // per-invocation Screma over the thread's chunk; `soac_lowering` lowers that
        // local loop while the synthesized phase-2 entries combine its partials.
        make_screma_serial(&mut entry.graph, serial);
        phase1_resources.retain_mut(|access| {
            if !moved.contains(&access.resource) {
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
        phase1_resources.extend(
            accumulators.iter().map(|accumulator| SegResourceAccess::<ResourceId> {
                resource: accumulator.partial,
                access: ResourceAccess::Write,
            }),
        );
        phase1_resources.sort_by_key(|resource| resource.resource);
        Ok((BuiltPhase::new(entry, phase1_resources), phase2s))
    }
}

/// Workgroup width for the single-workgroup tree-reduce phase 2: `W` threads
/// grid-stride the `T` partials into shared memory, then reduce in-shared with
/// a log-`W` tree. Kept modest so `W * sizeof(elem)` stays within the
/// workgroup shared-memory budget (256 × a 36-byte tuple ≈ 9 KB). The phase2
/// The published compute stage must dispatch this same width.
struct ReduceCombineSpec<'a> {
    name: String,
    operator: &'a Func<Semantic>,
    component_types: &'a [Type<TypeName>],
    elem_ty: Type<TypeName>,
    source_graph: &'a EGraph,
    operator_captures: &'a [OperandRef],
    capture_inputs: &'a [SemanticResourceDecl],
    neutrals: &'a [ValueId],
    partials: ResourceId,
    accumulator_results: &'a [ResultBinding<Type<TypeName>>],
    output_stores: &'a [RoutedReductionStore],
    width: u32,
}

impl ReduceCombineSpec<'_> {
    fn emit_operator(
        &self,
        graph: &mut EGraph,
        block: BlockId,
        left: ValueId,
        right: ValueId,
        captures: &[OperandRef],
    ) -> Result<ValueId, String> {
        let mut operands = lambda_ops::unpack_results(graph, left, self.component_types)
            .into_iter()
            .map(|value| graph.operand_ref(value))
            .collect::<Vec<_>>();
        operands.extend(
            lambda_ops::unpack_results(graph, right, self.component_types)
                .into_iter()
                .map(|value| graph.operand_ref(value)),
        );
        operands.extend_from_slice(captures);
        let (_, result) = graph
            .emit_call(
                block,
                self.operator.region,
                self.operator.params(),
                self.operator.result(),
                operands,
                self.operator.effects(),
                None,
                None,
            )
            .map_err(|cause| {
                format!("reduction operator call does not match its canonical boundary: {cause}")
            })?;
        graph_ops::pack_result_values(graph, &result)
    }

    fn emit_tree(
        &self,
        b: &mut egir::builder::EntryBuilder,
        init_nid: ValueId,
        operator_captures: &[OperandRef],
    ) -> Result<(), String> {
        let elem_ty = self.elem_ty.clone();
        let partials_resource = self.partials;
        let phase1_graph = self.source_graph;
        let accumulator_results = self.accumulator_results;
        let output_stores = self.output_stores;
        let width = self.width;
        let w = width;
        let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
        let bool_ty = Type::Constructed(TypeName::Bool, vec![]);
        let view_arr_ty = types::view_array_with_size(
            &elem_ty,
            Type::Variable(0),
            // Resource stamped by `intern_resource_view`.
            types::no_buffer(),
        );

        // ---- entry block: lid, partials view + length, shared view, result view ----
        let entry_bid = b.graph_mut().skeleton.entry;
        let (graph, eff) = b.construction_parts_mut();

        let lid = graph_ops::intern_intrinsic(
            graph,
            catalog().known().local_id,
            smallvec![],
            u32_ty.clone(),
            None,
        );
        let partials_view =
            graph_ops::intern_resource_view(graph, partials_resource, view_arr_ty.clone(), None);
        let len = graph_ops::intern_resource_len(graph, partials_resource, None);
        // Workgroup-shared `array<elem, W>` (id 0 within this entry).
        let shared_view = graph_ops::emit_workgroup_view(graph, 0, w, view_arr_ty.clone(), None);
        let w_nid = graph_ops::intern_u32(graph, w, None);
        let zero_u32 = graph_ops::intern_u32(graph, 0, None);

        // Contiguous per-thread chunk over `partials` (not strided): thread `lid`
        // reduces `partials[start .. end)`, so the tree combines `shared[0..W]` in
        // global order and the reduction stays valid for associative,
        // non-commutative operators.
        //   chunk = ceil(len / W);  start = lid * chunk;  end = min(start+chunk, len)
        let w_minus_1 = graph_ops::intern_u32(graph, w - 1, None);
        let len_plus = graph_ops::intern_binop(
            graph,
            op::BinaryOperator::Add,
            len,
            w_minus_1,
            u32_ty.clone(),
            None,
        );
        let chunk = graph_ops::intern_binop(
            graph,
            op::BinaryOperator::Divide,
            len_plus,
            w_nid,
            u32_ty.clone(),
            None,
        );
        let start = graph_ops::intern_binop(
            graph,
            op::BinaryOperator::Multiply,
            lid,
            chunk,
            u32_ty.clone(),
            None,
        );
        let start_plus =
            graph_ops::intern_binop(graph, op::BinaryOperator::Add, start, chunk, u32_ty.clone(), None);
        let u32_min = catalog()
            .specialize_numeric(catalog().known().min, &TypeName::UInt(32))
            .ok_or_else(|| "u32 min specialization is missing from the catalog".to_string())?;
        let end =
            graph_ops::intern_intrinsic(graph, u32_min, smallvec![start_plus, len], u32_ty.clone(), None);

        // ---- blocks ----
        let grid_header = graph.skeleton.create_block();
        let grid_body = graph.skeleton.create_block();
        let grid_cont = graph.skeleton.create_block();
        let grid_after = graph.skeleton.create_block();
        let tree_header = graph.skeleton.create_block();
        let tree_body = graph.skeleton.create_block();
        let tree_then = graph.skeleton.create_block();
        let tree_sel_merge = graph.skeleton.create_block();
        let tree_cont = graph.skeleton.create_block();
        let tree_after = graph.skeleton.create_block();
        let write_blk = graph.skeleton.create_block();
        let end_blk = graph.skeleton.create_block();

        // grid_header params: (acc, i)
        let acc_in = graph.add_block_param(grid_header, elem_ty.clone());
        let i_in = graph.add_block_param(grid_header, u32_ty.clone());

        // entry → grid_header(init, start)
        graph.skeleton.blocks[entry_bid].term = SkeletonTerminator::Branch {
            target: grid_header,
            args: graph.admit_flow_values([init_nid, start]),
        };

        // grid_header: i < end ? grid_body : grid_after(acc)
        let grid_cond =
            graph_ops::intern_binop(graph, op::BinaryOperator::Less, i_in, end, bool_ty.clone(), None);
        graph.skeleton.blocks[grid_header].term = SkeletonTerminator::CondBranch {
            cond: grid_cond,
            then_target: grid_body,
            then_args: vec![],
            else_target: grid_after,
            else_args: graph.admit_flow_values([acc_in]),
        };
        graph.skeleton.blocks[grid_header].control_header = Some(ControlHeader::Loop {
            merge: grid_after,
            continue_block: grid_cont,
        });

        // grid_body: acc' = op(acc, partials[i]); → grid_cont(acc')
        let elem_i =
            graph_ops::emit_view_load(graph, grid_body, partials_view, i_in, elem_ty.clone(), eff, None);
        let acc_next = self.emit_operator(graph, grid_body, acc_in, elem_i, operator_captures)?;
        graph.skeleton.blocks[grid_body].term = SkeletonTerminator::Branch {
            target: grid_cont,
            args: graph.admit_flow_values([acc_next]),
        };

        // grid_cont(acc_c): i_next = i + W; → grid_header(acc_c, i_next)
        let acc_c = graph.add_block_param(grid_cont, elem_ty.clone());
        let one_u32 = graph_ops::intern_u32(graph, 1, None);
        let i_next = graph_ops::intern_binop(
            graph,
            op::BinaryOperator::Add,
            i_in,
            one_u32,
            u32_ty.clone(),
            None,
        );
        graph.skeleton.blocks[grid_cont].term = SkeletonTerminator::Branch {
            target: grid_header,
            args: graph.admit_flow_values([acc_c, i_next]),
        };

        // grid_after(acc_final): shared[lid] = acc_final; barrier; → tree_header(1)
        let acc_final = graph.add_block_param(grid_after, elem_ty.clone());
        graph_ops::emit_storage_store(
            graph,
            grid_after,
            shared_view,
            lid,
            acc_final,
            elem_ty.clone(),
            eff,
            None,
        );
        graph_ops::emit_workgroup_barrier(graph, grid_after, eff);
        graph.skeleton.blocks[grid_after].term = SkeletonTerminator::Branch {
            target: tree_header,
            args: graph.admit_flow_values([one_u32]),
        };

        // Grow an adjacent-pair tree from stride 1. This preserves source order
        // for associative, non-commutative operators.
        let stride_in = graph.add_block_param(tree_header, u32_ty.clone());
        let stride_cond = graph_ops::intern_binop(
            graph,
            op::BinaryOperator::Less,
            stride_in,
            w_nid,
            bool_ty.clone(),
            None,
        );
        graph.skeleton.blocks[tree_header].term = SkeletonTerminator::CondBranch {
            cond: stride_cond,
            then_target: tree_body,
            then_args: vec![],
            else_target: tree_after,
            else_args: vec![],
        };
        graph.skeleton.blocks[tree_header].control_header = Some(ControlHeader::Loop {
            merge: tree_after,
            continue_block: tree_cont,
        });

        // Only the first lane in each adjacent pair combines the two runs.
        let two = graph_ops::intern_u32(graph, 2, None);
        let pair_width = graph_ops::intern_binop(
            graph,
            op::BinaryOperator::Multiply,
            stride_in,
            two,
            u32_ty.clone(),
            None,
        );
        let lane_in_pair = graph_ops::intern_binop(
            graph,
            op::BinaryOperator::Remainder,
            lid,
            pair_width,
            u32_ty.clone(),
            None,
        );
        let active = graph_ops::intern_binop(
            graph,
            op::BinaryOperator::Equal,
            lane_in_pair,
            zero_u32,
            bool_ty.clone(),
            None,
        );
        graph.skeleton.blocks[tree_body].term = SkeletonTerminator::CondBranch {
            cond: active,
            then_target: tree_then,
            then_args: vec![],
            else_target: tree_sel_merge,
            else_args: vec![],
        };
        graph.skeleton.blocks[tree_body].control_header = Some(ControlHeader::Selection {
            merge: tree_sel_merge,
        });

        // tree_then: shared[lid] = op(shared[lid], shared[lid+stride]); → tree_sel_merge
        let a = graph_ops::emit_view_load(graph, tree_then, shared_view, lid, elem_ty.clone(), eff, None);
        let lid_plus = graph_ops::intern_binop(
            graph,
            op::BinaryOperator::Add,
            lid,
            stride_in,
            u32_ty.clone(),
            None,
        );
        let bb = graph_ops::emit_view_load(
            graph,
            tree_then,
            shared_view,
            lid_plus,
            elem_ty.clone(),
            eff,
            None,
        );
        let combined = self.emit_operator(graph, tree_then, a, bb, operator_captures)?;
        graph_ops::emit_storage_store(
            graph,
            tree_then,
            shared_view,
            lid,
            combined,
            elem_ty.clone(),
            eff,
            None,
        );
        graph.skeleton.blocks[tree_then].term = SkeletonTerminator::Branch {
            target: tree_sel_merge,
            args: vec![],
        };

        // tree_sel_merge → tree_cont   (selection merge; barrier lives past it)
        graph.skeleton.blocks[tree_sel_merge].term = SkeletonTerminator::Branch {
            target: tree_cont,
            args: vec![],
        };

        // tree_cont: barrier; stride_next = stride*2; → tree_header(stride_next)
        graph_ops::emit_workgroup_barrier(graph, tree_cont, eff);
        let stride_next = graph_ops::intern_binop(
            graph,
            op::BinaryOperator::Multiply,
            stride_in,
            two,
            u32_ty.clone(),
            None,
        );
        graph.skeleton.blocks[tree_cont].term = SkeletonTerminator::Branch {
            target: tree_header,
            args: graph.admit_flow_values([stride_next]),
        };

        // tree_after: lid == 0 ? write_blk : end_blk   (selection)
        let is_zero = graph_ops::intern_binop(
            graph,
            op::BinaryOperator::Equal,
            lid,
            zero_u32,
            bool_ty.clone(),
            None,
        );
        graph.skeleton.blocks[tree_after].term = SkeletonTerminator::CondBranch {
            cond: is_zero,
            then_target: write_blk,
            then_args: vec![],
            else_target: end_blk,
            else_args: vec![],
        };
        graph.skeleton.blocks[tree_after].control_header =
            Some(ControlHeader::Selection { merge: end_blk });

        // write_blk: combined = shared[0]; replay each captured output store reading
        // `combined` in place of the per-thread accumulator value. A scalar reduce
        // has one store (`out[0] = combined`); a tuple-element reduce decomposes
        // across one store per field.
        let s0 = graph_ops::emit_view_load(
            graph,
            write_blk,
            shared_view,
            zero_u32,
            elem_ty.clone(),
            eff,
            None,
        );
        let combined_components = lambda_ops::unpack_results(graph, s0, self.component_types);
        let substitutions = accumulator_results
            .iter()
            .zip(&combined_components)
            .zip(self.component_types)
            .flat_map(|((old, combined), ty)| {
                let abi = egir::types::by_value_function_result::<egir::types::WynLanguage>(ty.clone());
                let new = graph_ops::bind_by_value_result(graph, &abi, *combined);
                old.values().into_iter().zip(new.values())
            })
            .collect::<Vec<_>>();
        for store in output_stores {
            let cloned_value = graph_ops::clone_pure_subgraph_substituting(
                phase1_graph,
                graph,
                store.value,
                &substitutions,
            )?;
            graph_ops::emit_resource_write(
                graph,
                write_blk,
                store.output.0,
                cloned_value,
                &store.ty,
                eff,
                None,
            )?;
        }
        graph.skeleton.blocks[write_blk].term = SkeletonTerminator::Branch {
            target: end_blk,
            args: vec![],
        };

        // end_blk is the exit; `build()` finalizes it with Return(None).
        b.set_current_block(end_blk);
        Ok(())
    }
}

/// Programmatic phase 2 synthesis where the neutral element is a
/// (possibly compound) pure subgraph cloned from phase 1. Used by the
/// Screma reduce path for any NE shape (scalar literal, tuple, array,
/// etc.).
/// Synthesize a reduce phase-2 combine entry. Its `partials` buffer is typed as
/// the (possibly tuple) accumulator element; the workgroup tree reduces them to
/// one combined value and replays the accumulator's captured output stores
/// (`output_stores`, `(place, value)` nodes from `phase1_graph`) against it,
/// substituting `accumulator_value` for the combined result. `output_decls`
/// declares the output bindings this entry writes. Screma's multi-accumulator
/// path passes a `_phase2_combine_{i}` `full_name` per combiner.
impl ReduceCombineSpec<'_> {
    fn build(
        self,
        identities: &mut egir::program::ProgramIdentities,
        semantic_ids: &mut egir::program::SemanticOpIdSource,
        effect_ids: &mut IdSource<EffectToken>,
    ) -> Result<BuiltPhase, String> {
        use crate::egir::builder::EntryBuilder;
        let mut resources = vec![SegResourceAccess::<ResourceId> {
            resource: self.partials,
            access: ResourceAccess::Read,
        }];
        resources.extend(
            self.capture_inputs.iter().map(|declaration| SegResourceAccess::<ResourceId> {
                resource: declaration.resource.0,
                access: ResourceAccess::Read,
            }),
        );
        let output_declarations = self.output_stores.iter().map(|store| store.output.clone()).fold(
            Vec::new(),
            |mut outputs, output| {
                if !outputs.iter().any(|(resource, _, _)| *resource == output.0) {
                    outputs.push(output);
                }
                outputs
            },
        );
        resources.extend(
            output_declarations.iter().map(|(resource, _, _)| SegResourceAccess::<ResourceId> {
                resource: *resource,
                access: ResourceAccess::Write,
            }),
        );
        resources = egir::ir::SegResourceAccess::merge(&resources, &[]);
        let mut b = EntryBuilder::new_compute(
            self.name.clone(),
            (self.width, 1, 1),
            identities,
            semantic_ids,
            effect_ids,
        );
        b.declare_intermediate_storage(self.partials);
        for declaration in self.capture_inputs {
            b.declare_input_storage(declaration.resource.0);
        }
        for (resource, _, _) in &output_declarations {
            b.declare_output_storage(*resource);
        }

        let neutrals = self
            .neutrals
            .iter()
            .map(|neutral| graph_ops::clone_pure_subgraph(self.source_graph, b.graph_mut(), *neutral))
            .collect::<Result<Vec<_>, _>>()?;
        let init_nid = lambda_ops::pack_results(b.graph_mut(), &neutrals, self.component_types);
        let operator_captures = self
            .operator_captures
            .iter()
            .map(|capture| graph_ops::clone_operand_subgraph(self.source_graph, b.graph_mut(), *capture))
            .collect::<Result<Vec<_>, _>>()?;
        self.emit_tree(&mut b, init_nid, &operator_captures)?;
        Ok(BuiltPhase::new(b.build(), resources))
    }
}
