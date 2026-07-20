//! Parallel reduction candidate analysis, binding, and phase emission.

use super::model::{REDUCE_PHASE1_WIDTH, REDUCE_PHASE2_WIDTH};
use super::*;
/// Complete graph-local reduction recipe, consumed before entry mutation.
pub(super) struct ReduceCandidate {
    pub site: SideEffectSite,
    pub owner: SemanticOpId,
    serial: SerialScremaRecipe,
    input_views: Vec<(NodeId, Type<TypeName>)>,
    map_output_view_operands: Vec<usize>,
    map_count: usize,
    result: NodeId,
    accumulators: Vec<ReduceAccumulator>,
    phase1_width: u32,
    phase2_width: u32,
}

struct ReduceAccumulator {
    scratch_type: Type<TypeName>,
    combine_region: RegionId,
    neutral: NodeId,
    stores: Vec<ReduceOutputStore>,
    outputs: Vec<(ResourceId, Type<TypeName>, crate::egir::program::LogicalSize)>,
}

pub(super) struct BoundReduce {
    candidate: ReduceCandidate,
    partials: Vec<ResourceId>,
}

struct EmissionAccumulator {
    scratch_type: Type<TypeName>,
    operator: String,
    neutral: NodeId,
    stores: Vec<(NodeId, NodeId)>,
    outputs: Vec<(ResourceId, Type<TypeName>, crate::egir::program::LogicalSize)>,
    partial: ResourceId,
}

struct ReduceOutputStore {
    location: (BlockId, usize),
    place: NodeId,
    value: NodeId,
    writer: Option<crate::egir::types::EffectToken>,
}

impl ReduceCandidate {
    pub(super) fn scratch_types(&self) -> impl Iterator<Item = &Type<TypeName>> {
        self.accumulators.iter().map(|accumulator| &accumulator.scratch_type)
    }
}

pub(super) fn analyze_reduce_candidate(
    entry: &crate::egir::program::PlannedEntry,
    located: ParallelReduce<'_>,
    resources: &crate::egir::program::LogicalResourceArena,
) -> error::Result<RecipeSelection<ReduceCandidate>> {
    let serial = located.serial_recipe();
    let (located, lanes, operators) = located.into_parts();
    let site = located.site;
    let side_effect = located.effect;
    let operators = operators.iter().collect::<Vec<_>>();
    let n_accs = operators.len();
    let n_maps = lanes.maps.len();
    let operands =
        screma::ScremaOperands::decode(located.op, &side_effect.operand_nodes, side_effect.result)?;
    for input in operands.inputs() {
        if !can_chunk_view(&entry.graph, input.node, ChunkInputKind::StorageOrRange) {
            return Ok(RecipeSelection::Serial(FallbackReason::UnsupportedViewShape));
        }
    }
    let mut map_output_view_operands = Vec::with_capacity(n_maps);
    for index in 0..n_maps {
        let Some(output) = operands.output(index) else {
            return Ok(RecipeSelection::Serial(FallbackReason::UnsupportedDestination));
        };
        if !can_chunk_view(&entry.graph, output.node, ChunkInputKind::StorageOnly) {
            return Ok(RecipeSelection::Serial(FallbackReason::UnsupportedViewShape));
        }
        map_output_view_operands.push(output.slot);
    }
    let result = operands.result();
    let owner = located.owner;
    if operators.iter().any(|operator| !can_clone_pure_subgraph(&entry.graph, operator.neutral, &[])) {
        return Ok(RecipeSelection::Serial(FallbackReason::UnsupportedCaptures));
    }
    let scratch_types =
        operators.iter().map(|operator| entry.graph.types[&operator.neutral].clone()).collect::<Vec<_>>();
    if scratch_types.iter().any(|ty| crate::ssa::layout::type_byte_size(ty).is_none()) {
        return Ok(RecipeSelection::Serial(FallbackReason::UnsupportedScratchLayout));
    }
    let input_views =
        operands.inputs().map(|input| (input.node, entry.graph.types[&input.node].clone())).collect();
    let mut stores = (0..n_accs).map(|_| Vec::new()).collect::<Vec<_>>();
    let mut outputs: Vec<Vec<(ResourceId, Type<TypeName>, crate::egir::program::LogicalSize)>> =
        vec![Vec::new(); n_accs];
    for (block_id, block) in &entry.graph.skeleton.blocks {
        for (effect_index, effect) in block.side_effects.iter().enumerate() {
            if !matches!(effect.kind, SideEffectKind::Effect(EffectOp::Store)) {
                continue;
            }
            let (Some(&place), Some(&value)) = (effect.operand_nodes.first(), effect.operand_nodes.get(1))
            else {
                continue;
            };
            let Some(root) = graph_ops::root_projection_index(&entry.graph, value, result)
                .or_else(|| (value == result && n_maps + n_accs == 1).then_some(0))
            else {
                continue;
            };
            let accumulator = root;
            if accumulator < n_maps || accumulator - n_maps >= n_accs {
                continue;
            }
            let accumulator = accumulator - n_maps;
            if !can_clone_pure_subgraph(&entry.graph, place, &[])
                || !can_clone_pure_subgraph(&entry.graph, value, &[result])
            {
                return Ok(RecipeSelection::Serial(FallbackReason::UnsupportedCaptures));
            }
            stores[accumulator].push(ReduceOutputStore {
                location: (block_id, effect_index),
                place,
                value,
                writer: effect.effects.map(|(_, writer)| writer),
            });
            if let Some(resource) =
                graph_ops::storage_resource_under(&entry.graph, place).map(|resource| resource.0)
            {
                let logical = &resources[resource];
                let output = entry.resource_declarations.iter().find(|declaration| {
                    declaration.role == crate::interface::StorageRole::Output
                        && declaration.resource.0 == resource
                });
                if let Some(output) = output {
                    if !outputs[accumulator].iter().any(|(candidate, _, _)| *candidate == resource) {
                        outputs[accumulator].push((resource, output.elem_ty.clone(), logical.size.clone()));
                    }
                }
            }
        }
    }
    if !(0..n_accs).all(|index| !stores[index].is_empty() && !outputs[index].is_empty()) {
        return Ok(RecipeSelection::Serial(FallbackReason::UnsupportedDestination));
    }
    let accumulators = operators
        .into_iter()
        .zip(scratch_types)
        .zip(stores)
        .zip(outputs)
        .map(
            |(((operator, scratch_type), stores), outputs)| ReduceAccumulator {
                scratch_type,
                combine_region: operator.combine.region,
                neutral: operator.neutral,
                stores,
                outputs,
            },
        )
        .collect();
    Ok(RecipeSelection::Parallel(ReduceCandidate {
        site,
        owner,
        serial,
        input_views,
        map_output_view_operands,
        map_count: n_maps,
        result,
        accumulators,
        phase1_width: REDUCE_PHASE1_WIDTH,
        phase2_width: REDUCE_PHASE2_WIDTH,
    }))
}

impl BoundReduce {
    pub(super) fn bind(candidate: ReduceCandidate, resources: &super::planning::ScratchBindings) -> Self {
        let partials = (0..candidate.accumulators.len())
            .map(|slot| resources.id(candidate.owner, CompilerResourceKind::ReducePartial, slot))
            .collect();
        Self { candidate, partials }
    }
}

impl KernelPlanBuilder<'_, '_> {
    pub(super) fn emit_reduce_entry(
        &mut self,
        entry: &mut crate::egir::program::PlannedEntry,
        bound: BoundReduce,
    ) -> error::Result<Vec<crate::egir::program::PlannedEntry>> {
        let BoundReduce {
            candidate,
            partials: partial_resources,
        } = bound;
        let ReduceCandidate {
            site,
            serial,
            input_views: input_view_data,
            map_output_view_operands,
            map_count: n_maps,
            result: screma_result_nid,
            accumulators,
            phase1_width,
            phase2_width,
            ..
        } = candidate;
        let block_id = site.block;
        let total_threads = phase1_width;
        let n_accs = accumulators.len();
        let mut drop_locations = Vec::new();
        let mut dropped_writers = std::collections::HashSet::new();
        let accumulators = accumulators
            .into_iter()
            .zip(partial_resources)
            .map(|(accumulator, partial)| {
                let mut stores = Vec::with_capacity(accumulator.stores.len());
                for store in accumulator.stores {
                    stores.push((store.place, store.value));
                    drop_locations.push(store.location);
                    dropped_writers.extend(store.writer);
                }
                EmissionAccumulator {
                    scratch_type: accumulator.scratch_type,
                    operator: self.schedule.callable_name(accumulator.combine_region).to_string(),
                    neutral: accumulator.neutral,
                    stores,
                    outputs: accumulator.outputs,
                    partial,
                }
            })
            .collect::<Vec<_>>();
        // 3. Chunk all input views and every map output view; swap them back
        // into the Screma operand list.
        let chunked = chunk_soac_inputs(
            &mut entry.graph,
            &input_view_data,
            total_threads,
            ChunkInputKind::StorageOrRange,
            "SegRed",
        )?;
        let chunk_start = chunked.chunk_start;
        let chunk_len = chunked.chunk_len;
        {
            let se = entry.graph.skeleton.effect_mut(site);
            for (i, &new_view) in chunked.views.iter().enumerate() {
                se.operand_nodes[i] = new_view;
            }
        }
        for (map_index, operand_index) in map_output_view_operands.iter().enumerate() {
            let orig_view = entry.graph.skeleton.effect(site).operand_nodes[*operand_index];
            let view_ty = entry.graph.types[&orig_view].clone();
            let chunked_view = chunk_view_like(
                &mut entry.graph,
                orig_view,
                view_ty,
                chunk_start,
                chunk_len,
                ChunkInputKind::StorageOnly,
                &format!("SegRed map output {map_index}"),
            )?;
            entry.graph.skeleton.effect_mut(site).operand_nodes[*operand_index] = chunked_view;
        }

        // 5. Phase 1 stores each thread's whole accumulator value to `partials[tid]`
        // and no longer writes the outputs. `accumulator_value` is the hash-consed
        // `Project{acc_pos}(screma_result)` node — phase 2 substitutes it for the
        // combined result when replaying the captured stores.
        let accumulator_values: Vec<NodeId> = (0..n_accs)
            .zip(&accumulators)
            .map(|(acc_i, accumulator)| {
                entry.graph.intern_pure(
                    crate::egir::types::PureOp::Project {
                        index: (n_maps + acc_i) as u32,
                    },
                    smallvec![screma_result_nid],
                    accumulator.scratch_type.clone(),
                    None,
                )
            })
            .collect();
        // Drop the decomposed output stores (highest index first per block).
        drop_locations.sort_by_key(|location| std::cmp::Reverse(location.1));
        for (bid, sx) in drop_locations {
            entry.graph.skeleton.blocks[bid].side_effects.remove(sx);
        }
        for route in &mut entry.output_routes {
            route.writers.retain(
            |writer| !matches!(writer, OutputWriter::Effect(effect) if dropped_writers.contains(effect)),
        );
        }
        for (accumulator, accumulator_value) in accumulators.iter().zip(&accumulator_values) {
            let elem_ty = accumulator.scratch_type.clone();
            let arr_ty =
                crate::types::view_array_with_size(&elem_ty, Type::Variable(0), crate::types::no_buffer());
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
            // Clear the moved output bindings from phase 1; register partials.
            for (resource, _, _) in &accumulator.outputs {
                let logical = &self.resources[*resource];
                if let Some(binding) = logical.host_binding() {
                    for output in &mut entry.outputs {
                        if output.storage_binding() == Some(binding) {
                            output.make_storage_internal();
                        }
                    }
                }
            }
            entry.resource_declarations.push(SemanticResourceDecl {
                resource: SemanticResourceRef(accumulator.partial),
                role: crate::interface::StorageRole::Intermediate,
                elem_ty: accumulator.scratch_type.clone(),
                size: self.resources[accumulator.partial].size.clone(),
            });
        }
        // A moved output binding may also carry an Output storage declaration (e.g. a
        // hoisted prepass result). Phase 1 no longer writes it; phase 2 owns it.
        let moved: std::collections::HashSet<ResourceId> = accumulators
            .iter()
            .flat_map(|accumulator| &accumulator.outputs)
            .map(|(resource, _, _)| *resource)
            .collect();
        entry.resource_declarations.retain(|declaration| {
            declaration.role != crate::interface::StorageRole::Output
                || !moved.contains(&declaration.resource.0)
        });

        // 6. Synthesize one phase 2 entry per accumulator. Dropping the phase-1
        // stores leaves their pure place/value subgraphs available for projection.
        let mut phase2s = Vec::with_capacity(n_accs);
        for (acc_i, (accumulator, accumulator_value)) in
            accumulators.iter().zip(accumulator_values).enumerate()
        {
            let phase2_name = if n_accs == 1 {
                format!("{}_phase2_combine", entry.name)
            } else {
                format!("{}_phase2_combine_{}", entry.name, acc_i)
            };
            let combine = ReduceCombineSpec {
                name: phase2_name,
                operator: accumulator.operator.clone(),
                elem_ty: accumulator.scratch_type.clone(),
                source_graph: &entry.graph,
                neutral: accumulator.neutral,
                partials: accumulator.partial,
                accumulator: accumulator_value,
                output_stores: &accumulator.stores,
                output_declarations: &accumulator.outputs,
                width: phase2_width,
            };
            let phase2 = combine.build(self.effect_ids)?;
            phase2s.push(phase2);
        }
        // Scheduling consumed the semantic SegRed. Phase 1 is now an ordinary
        // per-invocation Screma over the thread's chunk; `soac_expand` lowers that
        // local loop while the synthesized phase-2 entries combine its partials.
        make_screma_serial(&mut entry.graph, serial);
        Ok(phase2s)
    }
}

/// Workgroup width for the single-workgroup tree-reduce phase 2: `W` threads
/// grid-stride the `T` partials into shared memory, then reduce in-shared with
/// a log-`W` tree. Kept modest so `W * sizeof(elem)` stays within the
/// workgroup shared-memory budget (256 × a 36-byte tuple ≈ 9 KB). The phase2
/// The published compute stage must dispatch this same width.
struct ReduceCombineSpec<'a> {
    name: String,
    operator: String,
    elem_ty: Type<TypeName>,
    source_graph: &'a crate::egir::types::EGraph,
    neutral: NodeId,
    partials: ResourceId,
    accumulator: NodeId,
    output_stores: &'a [(NodeId, NodeId)],
    output_declarations: &'a [(ResourceId, Type<TypeName>, crate::egir::program::LogicalSize)],
    width: u32,
}

impl ReduceCombineSpec<'_> {
    fn emit_tree(
        &self,
        b: &mut crate::egir::builder::EntryBuilder,
        init_nid: NodeId,
    ) -> Result<(), String> {
        let op_func = self.operator.clone();
        let elem_ty = self.elem_ty.clone();
        let partials_resource = self.partials;
        let phase1_graph = self.source_graph;
        let accumulator_value = self.accumulator;
        let output_stores = self.output_stores;
        let width = self.width;
        let w = width;
        let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
        let bool_ty = Type::Constructed(TypeName::Bool, vec![]);
        let view_arr_ty = crate::types::view_array_with_size(
            &elem_ty,
            Type::Variable(0),
            // Resource stamped by `intern_resource_view`.
            crate::types::no_buffer(),
        );

        // ---- entry block: lid, partials view + length, shared view, result view ----
        let entry_bid = b.graph_mut().skeleton.entry;
        let (graph, control_headers, eff) = b.construction_parts_mut();

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
        let len_plus = graph_ops::intern_binop(graph, "+", len, w_minus_1, u32_ty.clone(), None);
        let chunk = graph_ops::intern_binop(graph, "/", len_plus, w_nid, u32_ty.clone(), None);
        let start = graph_ops::intern_binop(graph, "*", lid, chunk, u32_ty.clone(), None);
        let start_plus = graph_ops::intern_binop(graph, "+", start, chunk, u32_ty.clone(), None);
        let u32_min = catalog()
            .lookup_by_any_name("u32.min")
            .ok_or_else(|| "required builtin `u32.min` is missing from the catalog".to_string())?;
        let end = graph_ops::intern_intrinsic(
            graph,
            u32_min.id,
            smallvec![start_plus, len],
            u32_ty.clone(),
            None,
        );

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
            args: vec![init_nid, start],
        };

        // grid_header: i < end ? grid_body : grid_after(acc)
        let grid_cond = graph_ops::intern_binop(graph, "<", i_in, end, bool_ty.clone(), None);
        graph.skeleton.blocks[grid_header].term = SkeletonTerminator::CondBranch {
            cond: grid_cond,
            then_target: grid_body,
            then_args: vec![],
            else_target: grid_after,
            else_args: vec![acc_in],
        };
        control_headers.insert(
            grid_header,
            ControlHeader::Loop {
                merge: grid_after,
                continue_block: grid_cont,
            },
        );

        // grid_body: acc' = op(acc, partials[i]); → grid_cont(acc')
        let elem_i =
            graph_ops::emit_view_load(graph, grid_body, partials_view, i_in, elem_ty.clone(), eff, None);
        let acc_next = graph.intern_pure(
            PureOp::Call(op_func.clone()),
            smallvec![acc_in, elem_i],
            elem_ty.clone(),
            None,
        );
        graph.skeleton.blocks[grid_body].term = SkeletonTerminator::Branch {
            target: grid_cont,
            args: vec![acc_next],
        };

        // grid_cont(acc_c): i_next = i + W; → grid_header(acc_c, i_next)
        let acc_c = graph.add_block_param(grid_cont, elem_ty.clone());
        let one_u32 = graph_ops::intern_u32(graph, 1, None);
        let i_next = graph_ops::intern_binop(graph, "+", i_in, one_u32, u32_ty.clone(), None);
        graph.skeleton.blocks[grid_cont].term = SkeletonTerminator::Branch {
            target: grid_header,
            args: vec![acc_c, i_next],
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
            args: vec![one_u32],
        };

        // Grow an adjacent-pair tree from stride 1. This preserves source order
        // for associative, non-commutative operators.
        let stride_in = graph.add_block_param(tree_header, u32_ty.clone());
        let stride_cond = graph_ops::intern_binop(graph, "<", stride_in, w_nid, bool_ty.clone(), None);
        graph.skeleton.blocks[tree_header].term = SkeletonTerminator::CondBranch {
            cond: stride_cond,
            then_target: tree_body,
            then_args: vec![],
            else_target: tree_after,
            else_args: vec![],
        };
        control_headers.insert(
            tree_header,
            ControlHeader::Loop {
                merge: tree_after,
                continue_block: tree_cont,
            },
        );

        // Only the first lane in each adjacent pair combines the two runs.
        let two = graph_ops::intern_u32(graph, 2, None);
        let pair_width = graph_ops::intern_binop(graph, "*", stride_in, two, u32_ty.clone(), None);
        let lane_in_pair = graph_ops::intern_binop(graph, "%", lid, pair_width, u32_ty.clone(), None);
        let active = graph_ops::intern_binop(graph, "==", lane_in_pair, zero_u32, bool_ty.clone(), None);
        graph.skeleton.blocks[tree_body].term = SkeletonTerminator::CondBranch {
            cond: active,
            then_target: tree_then,
            then_args: vec![],
            else_target: tree_sel_merge,
            else_args: vec![],
        };
        control_headers.insert(
            tree_body,
            ControlHeader::Selection {
                merge: tree_sel_merge,
            },
        );

        // tree_then: shared[lid] = op(shared[lid], shared[lid+stride]); → tree_sel_merge
        let a = graph_ops::emit_view_load(graph, tree_then, shared_view, lid, elem_ty.clone(), eff, None);
        let lid_plus = graph_ops::intern_binop(graph, "+", lid, stride_in, u32_ty.clone(), None);
        let bb = graph_ops::emit_view_load(
            graph,
            tree_then,
            shared_view,
            lid_plus,
            elem_ty.clone(),
            eff,
            None,
        );
        let combined = graph.intern_pure(
            PureOp::Call(op_func.clone()),
            smallvec![a, bb],
            elem_ty.clone(),
            None,
        );
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
        let stride_next = graph_ops::intern_binop(graph, "*", stride_in, two, u32_ty.clone(), None);
        graph.skeleton.blocks[tree_cont].term = SkeletonTerminator::Branch {
            target: tree_header,
            args: vec![stride_next],
        };

        // tree_after: lid == 0 ? write_blk : end_blk   (selection)
        let is_zero = graph_ops::intern_binop(graph, "==", lid, zero_u32, bool_ty.clone(), None);
        graph.skeleton.blocks[tree_after].term = SkeletonTerminator::CondBranch {
            cond: is_zero,
            then_target: write_blk,
            then_args: vec![],
            else_target: end_blk,
            else_args: vec![],
        };
        control_headers.insert(tree_after, ControlHeader::Selection { merge: end_blk });

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
        for &(place, value) in output_stores {
            let cloned_place = graph_ops::clone_pure_subgraph(phase1_graph, graph, place)?;
            let cloned_value = graph_ops::clone_pure_subgraph_substituting(
                phase1_graph,
                graph,
                value,
                &[(accumulator_value, s0)],
            )?;
            graph_ops::emit_store(graph, write_blk, cloned_place, cloned_value, eff, None);
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
        effect_ids: &mut crate::IdSource<EffectToken>,
    ) -> Result<crate::egir::program::PlannedEntry, String> {
        use crate::egir::builder::EntryBuilder;
        let mut b = EntryBuilder::new_compute(self.name.clone(), (self.width, 1, 1), effect_ids);
        b.declare_intermediate_storage_sized(
            self.partials,
            self.elem_ty.clone(),
            dispatch_worker_logical_size(&self.elem_ty),
        );
        for (resource, ty, size) in self.output_declarations {
            b.declare_output_storage_sized(*resource, ty.clone(), size.clone());
        }

        let init_nid = graph_ops::clone_pure_subgraph(self.source_graph, b.graph_mut(), self.neutral)?;
        self.emit_tree(&mut b, init_nid)?;
        Ok(b.build())
    }
}
