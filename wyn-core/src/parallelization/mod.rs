//! Compute shader parallelization.
//!
//! This module transforms compute shaders with parallelizable patterns
//! (like map loops) to use thread chunking for parallel execution.
//!
//! The design uses strategy traits to separate input and output concerns:
//! - `InputStrategy`: How to get elements (from storage buffer, from range, etc.)
//! - `OutputStrategy`: How to store results (to storage buffer, etc.)
//!
//! This avoids a combinatorial explosion of functions for each input×output combination.

mod strategies;

use crate::ast::TypeName;
use crate::pipeline_descriptor::{
    Access, Binding, BufferUsage, ComputePipeline, ComputeStage, DispatchSize, MultiComputePipeline,
    Pipeline, PipelineDescriptor,
};
use crate::ssa::builder::FuncBuilder;
use crate::ssa::soac_analysis::{ArrayProvenance, ParallelizableSoac, analyze_program};
use crate::ssa::types::{EffectToken, FuncBody, InstKind, Terminator, ValueId};
use crate::ssa::types::{EntryInput, EntryOutput, EntryPoint, ExecutionModel, Program};
use polytype::Type;

use std::collections::HashMap;

pub use strategies::{
    InputStrategy, OutputStrategy, RangeHandle, RangeInput, StorageInput, StorageOutput, remap_value,
};

/// Context passed to strategies during parallelization.
/// Contains the builder and common utilities.
pub struct ParallelizeCtx<'a> {
    pub builder: FuncBuilder,
    pub entry: &'a EntryPoint,
    // Common types
    pub u32_ty: Type<TypeName>,
    pub i32_ty: Type<TypeName>,
    pub bool_ty: Type<TypeName>,
    pub unit_ty: Type<TypeName>,
}

impl<'a> ParallelizeCtx<'a> {
    pub fn new(builder: FuncBuilder, entry: &'a EntryPoint) -> Self {
        Self {
            builder,
            entry,
            u32_ty: Type::Constructed(TypeName::UInt(32), vec![]),
            i32_ty: Type::Constructed(TypeName::Int(32), vec![]),
            bool_ty: Type::Constructed(TypeName::Bool, vec![]),
            unit_ty: Type::Constructed(TypeName::Unit, vec![]),
        }
    }

    pub fn push_int(&mut self, value: &str) -> Option<ValueId> {
        self.builder.push_int(value, self.u32_ty.clone()).ok()
    }

    pub fn push_i32(&mut self, value: &str) -> Option<ValueId> {
        self.builder.push_int(value, self.i32_ty.clone()).ok()
    }

    pub fn push_binop(
        &mut self,
        op: &str,
        lhs: ValueId,
        rhs: ValueId,
        ty: Type<TypeName>,
    ) -> Option<ValueId> {
        self.builder.push_binop(op, lhs, rhs, ty).ok()
    }

    pub fn push_intrinsic(
        &mut self,
        name: &str,
        args: Vec<ValueId>,
        ty: Type<TypeName>,
    ) -> Option<ValueId> {
        self.builder.push_intrinsic(name, args, ty).ok()
    }

    pub fn push_call(&mut self, func: &str, args: Vec<ValueId>, ty: Type<TypeName>) -> Option<ValueId> {
        self.builder.push_call(func, args, ty).ok()
    }

    pub fn push_inst(&mut self, kind: crate::ssa::types::InstKind, ty: Type<TypeName>) -> Option<ValueId> {
        self.builder.push_inst(kind, ty).ok()
    }

    /// Get the *new* body's entry effect token (from `self.builder`, not the original entry).
    /// Effect tokens are unordered markers — the SPIR-V backend ignores them for ordering.
    /// We use this single token for all parallel iterations since they're independent.
    pub fn entry_effect(&self) -> EffectToken {
        self.builder.entry_effect()
    }

    /// Consume the context and finish building, returning the function body.
    pub fn finish(self) -> Option<FuncBody> {
        self.builder.finish().ok()
    }

    /// Get thread ID and calculate chunk bounds for the current thread.
    /// Returns (thread_id, chunk_start, chunk_end).
    pub fn compute_thread_chunk(
        &mut self,
        input_len: ValueId,
        total_threads: u32,
    ) -> Option<(ValueId, ValueId, ValueId)> {
        let thread_id = self.push_intrinsic("_w_intrinsic_thread_id", vec![], self.u32_ty.clone())?;
        let total_threads_val = self.push_int(&total_threads.to_string())?;
        let threads_minus_1 = self.push_int(&(total_threads - 1).to_string())?;

        // chunk_size = ceil(len / total_threads) = (len + total_threads - 1) / total_threads
        let len_plus = self.push_binop("+", input_len, threads_minus_1, self.u32_ty.clone())?;
        let chunk_size = self.push_binop("/", len_plus, total_threads_val, self.u32_ty.clone())?;

        // chunk_start = thread_id * chunk_size
        let chunk_start = self.push_binop("*", thread_id, chunk_size, self.u32_ty.clone())?;

        // chunk_end = min(chunk_start + chunk_size, len)
        let start_plus_size = self.push_binop("+", chunk_start, chunk_size, self.u32_ty.clone())?;
        let chunk_end = self.push_call("u32.min", vec![start_plus_size, input_len], self.u32_ty.clone())?;

        Some((thread_id, chunk_start, chunk_end))
    }

    /// Remap captured variables from the entry body into the current builder.
    /// Returns the remapped values in the same order as `captures`.
    pub fn remap_captures(&mut self, captures: &[ValueId]) -> Option<Vec<ValueId>> {
        let entry = self.entry;
        let mut remap_memo: HashMap<ValueId, ValueId> = entry
            .body
            .params
            .iter()
            .enumerate()
            .map(|(i, (src, _, _))| (*src, self.builder.get_param(i)))
            .collect();

        let mut result = Vec::new();
        for &capture in captures {
            let new_val = remap_value(&entry.body, capture, &mut self.builder, &mut remap_memo)?;
            result.push(new_val);
        }
        Some(result)
    }
}

/// Result of the parallelization pass.
pub struct ParallelizationResult {
    /// The (possibly modified) SSA program, which may contain additional entry points.
    pub program: Program,
    /// Pipeline descriptor describing how to execute the program.
    pub pipeline: PipelineDescriptor,
}

/// Parallelize SOACs in an SSA program.
/// Returns the modified program and a pipeline descriptor for the host.
pub fn parallelize_soacs(mut program: Program) -> ParallelizationResult {
    let analysis = analyze_program(&program);
    let mut descriptor = PipelineDescriptor::default();
    // New entry points to add (from multi-dispatch SOACs like reduce).
    // These replace the original entry point.
    let mut new_entries: Vec<EntryPoint> = Vec::new();
    let mut entries_to_remove: Vec<String> = Vec::new();

    for entry in &mut program.entry_points {
        if let ExecutionModel::Compute { local_size } = entry.execution_model {
            if let Some(entry_analysis) = analysis.by_entry.get(&entry.name) {
                if let Some(ref par_soac) = entry_analysis.parallelizable_soac {
                    match par_soac {
                        ParallelizableSoac::Map {
                            source,
                            map_function,
                            captures,
                            output_elem_type,
                        } => {
                            let storage_outputs: Vec<_> = entry
                                .outputs
                                .iter()
                                .filter(|o| o.storage_binding.is_some())
                                .cloned()
                                .collect();

                            if let Some((new_body, output_binding)) = parallelize_map_entry(
                                entry,
                                source,
                                map_function,
                                captures,
                                output_elem_type,
                                local_size,
                            ) {
                                entry.body = new_body;
                                if !storage_outputs.is_empty() {
                                    entry.outputs = storage_outputs.clone();
                                } else {
                                    let array_ty = Type::Constructed(
                                        TypeName::Array,
                                        vec![
                                            output_elem_type.clone(),
                                            Type::Constructed(TypeName::SizePlaceholder, vec![]),
                                            Type::Constructed(TypeName::ArrayVariantView, vec![]),
                                        ],
                                    );
                                    entry.outputs = vec![EntryOutput {
                                        ty: array_ty,
                                        decoration: None,
                                        storage_binding: Some(output_binding),
                                    }];
                                }

                                descriptor.pipelines.push(build_map_pipeline(
                                    entry,
                                    output_binding,
                                    local_size,
                                ));
                            }
                        }
                        ParallelizableSoac::Reduce {
                            source,
                            reduce_function,
                            init,
                            captures,
                            elem_type,
                        } => {
                            if let Some((reduce_entries, pipeline)) = parallelize_reduce_entry(
                                entry,
                                source,
                                reduce_function,
                                *init,
                                captures,
                                elem_type,
                                local_size,
                            ) {
                                entries_to_remove.push(entry.name.clone());
                                new_entries.extend(reduce_entries);
                                descriptor.pipelines.push(pipeline);
                            }
                        }
                        ParallelizableSoac::Scan { .. } => {
                            // TODO: Phase 3 — multi-entry scan parallelization
                        }
                    }
                }
            }
        }
    }

    // Replace original entries with multi-dispatch entries
    if !entries_to_remove.is_empty() {
        program.entry_points.retain(|e| !entries_to_remove.contains(&e.name));
        program.entry_points.extend(new_entries);
    }

    ParallelizationResult {
        program,
        pipeline: descriptor,
    }
}

/// Build a single-dispatch compute pipeline descriptor for a map.
fn build_map_pipeline(
    entry: &EntryPoint,
    output_binding: (u32, u32),
    local_size: (u32, u32, u32),
) -> Pipeline {
    let mut bindings = Vec::new();

    for input in &entry.inputs {
        if let Some((set, binding)) = input.storage_binding {
            bindings.push(Binding::StorageBuffer {
                set,
                binding,
                access: Access::ReadOnly,
                usage: BufferUsage::Input,
                name: input.name.clone(),
            });
        }
    }

    bindings.push(Binding::StorageBuffer {
        set: output_binding.0,
        binding: output_binding.1,
        access: Access::WriteOnly,
        usage: BufferUsage::Output,
        name: format!("{}_output", entry.name),
    });

    Pipeline::Compute(ComputePipeline {
        entry_point: entry.name.clone(),
        workgroup_size: local_size,
        dispatch_size: DispatchSize::DerivedFromInputLength {
            workgroup_size: local_size.0,
        },
        bindings,
    })
}

/// Create a parallelized map version of a compute entry point.
/// Returns the new body and the (set, binding) used for the output storage buffer.
fn parallelize_map_entry(
    entry: &EntryPoint,
    source: &ArrayProvenance,
    map_function: &str,
    captures: &[ValueId],
    output_elem_type: &Type<TypeName>,
    local_size: (u32, u32, u32),
) -> Option<(FuncBody, (u32, u32))> {
    let total_threads = local_size.0 * local_size.1 * local_size.2;
    if total_threads == 0 {
        return None;
    }

    // Derive output storage binding from entry's declared outputs (assigned by to_ssa).
    // Fall back to counting storage inputs if no output binding exists yet (synthesis).
    let output_binding = entry.outputs.iter().find_map(|o| o.storage_binding).unwrap_or_else(|| {
        let next = entry.inputs.iter().filter(|i| i.storage_binding.is_some()).count() as u32;
        (0, next)
    });

    // Build the parallelized function
    let unit_ty = Type::Constructed(TypeName::Unit, vec![]);
    let params: Vec<(Type<TypeName>, String)> =
        entry.inputs.iter().map(|i| (i.ty.clone(), i.name.clone())).collect();
    let builder = FuncBuilder::new(params, unit_ty.clone());
    let mut ctx = ParallelizeCtx::new(builder, entry);

    // Create strategies and build loop body based on provenance
    match source {
        ArrayProvenance::EntryStorage {
            param_index,
            storage_binding,
            ..
        } => {
            let mut input = StorageInput::new(*param_index, *storage_binding);
            let output = StorageOutput::new(output_binding.0, output_binding.1);
            build_map_body(
                &mut ctx,
                &mut input,
                &output,
                map_function,
                captures,
                output_elem_type,
                total_threads,
            )?;
        }
        ArrayProvenance::Range { value } => {
            let mut input = RangeInput::new(*value, &entry.body)?;
            let output = StorageOutput::new(output_binding.0, output_binding.1);
            build_map_body(
                &mut ctx,
                &mut input,
                &output,
                map_function,
                captures,
                output_elem_type,
                total_threads,
            )?;
        }
        ArrayProvenance::Unknown => return None,
    };

    let body = ctx.finish()?;
    Some((body, output_binding))
}

/// Build the parallel map loop body: setup strategies, chunk work, loop, call map function, store.
fn build_map_body<I: InputStrategy, O: OutputStrategy>(
    ctx: &mut ParallelizeCtx,
    input_strategy: &mut I,
    output_strategy: &O,
    map_function: &str,
    captures: &[ValueId],
    output_elem_type: &Type<TypeName>,
    total_threads: u32,
) -> Option<()> {
    // 1. Setup input and output strategies (resources created once)
    let (input_handle, input_len, _input_elem_ty) = input_strategy.setup(ctx)?;
    let output_handle = output_strategy.setup(ctx, output_elem_type)?;

    // 2. Get thread ID and calculate chunk bounds
    let (_thread_id, chunk_start, chunk_end) = ctx.compute_thread_chunk(input_len, total_threads)?;

    // 3. Create loop structure
    let u32_ty = ctx.u32_ty.clone();
    let bool_ty = ctx.bool_ty.clone();

    let (header, header_params) = ctx.builder.create_block_with_params(vec![u32_ty.clone()]);
    let loop_index = header_params[0];
    let body_block = ctx.builder.create_block();
    let exit_block = ctx.builder.create_block();
    ctx.builder.mark_loop_header(header, exit_block, body_block).ok()?;

    // Branch to header with initial index
    ctx.builder
        .terminate(Terminator::Branch {
            target: header,
            args: vec![chunk_start],
        })
        .ok()?;

    // Header: check i < chunk_end
    ctx.builder.switch_to_block(header).ok()?;
    let cond = ctx.builder.push_binop("<", loop_index, chunk_end, bool_ty.clone()).ok()?;
    ctx.builder
        .terminate(Terminator::CondBranch {
            cond,
            then_target: body_block,
            then_args: vec![],
            else_target: exit_block,
            else_args: vec![],
        })
        .ok()?;

    // 4. Loop body: get element, apply function, store result
    ctx.builder.switch_to_block(body_block).ok()?;

    let input_elem = input_strategy.get_element(ctx, input_handle, loop_index)?;

    // Build call args: element first, then remapped captures
    let mut call_args = vec![input_elem];
    let remapped_captures = ctx.remap_captures(captures)?;
    call_args.extend(remapped_captures);

    // Apply map function with the correct return type
    let result_elem = ctx.push_call(map_function, call_args, output_elem_type.clone())?;

    // Store result
    output_strategy.store_result(ctx, output_handle, loop_index, result_elem, output_elem_type)?;

    // Increment index and branch back to header
    let one = ctx.push_int("1")?;
    let next_i = ctx.push_binop("+", loop_index, one, u32_ty.clone())?;
    ctx.builder
        .terminate(Terminator::Branch {
            target: header,
            args: vec![next_i],
        })
        .ok()?;

    // 5. Exit: return unit
    ctx.builder.switch_to_block(exit_block).ok()?;
    ctx.builder.terminate(Terminator::Return(None)).ok()?;

    Some(())
}

// =============================================================================
// Reduce Parallelization (2 dispatches)
// =============================================================================

/// Parallelize a reduce SOAC into two entry points:
/// 1. `{name}_phase1_chunks` — each thread reduces its chunk → partials[thread_id]
/// 2. `{name}_phase2_combine` — single thread combines partials → result[0]
///
/// Returns the two new entry points (replacing the original).
fn parallelize_reduce_entry(
    entry: &EntryPoint,
    source: &ArrayProvenance,
    reduce_function: &str,
    init: ValueId,
    captures: &[ValueId],
    elem_type: &Type<TypeName>,
    local_size: (u32, u32, u32),
) -> Option<(Vec<EntryPoint>, Pipeline)> {
    let total_threads = local_size.0 * local_size.1 * local_size.2;
    if total_threads == 0 {
        return None;
    }

    // Allocate bindings for intermediate (partials) and output (result) buffers.
    // Input bindings come from the entry's storage inputs.
    let next_binding = entry
        .inputs
        .iter()
        .filter_map(|i| i.storage_binding)
        .chain(entry.outputs.iter().filter_map(|o| o.storage_binding))
        .map(|(_, b)| b + 1)
        .max()
        .unwrap_or(0);
    let partials_binding = (0u32, next_binding);
    let result_binding = (0u32, next_binding + 1);

    // --- Phase 1: each thread reduces its chunk ---
    let phase1 = build_reduce_phase1(
        entry,
        source,
        reduce_function,
        init,
        captures,
        elem_type,
        total_threads,
        partials_binding,
        local_size,
    )?;

    // --- Phase 2: thread 0 combines partial results ---
    let phase2 = build_reduce_phase2(
        entry,
        reduce_function,
        init,
        captures,
        elem_type,
        total_threads,
        partials_binding,
        result_binding,
    )?;

    // Build pipeline descriptor
    let mut bindings = Vec::new();

    // Input storage buffers
    let mut input_binding_indices = Vec::new();
    for input in &entry.inputs {
        if let Some((set, binding)) = input.storage_binding {
            input_binding_indices.push(bindings.len());
            bindings.push(Binding::StorageBuffer {
                set,
                binding,
                access: Access::ReadOnly,
                usage: BufferUsage::Input,
                name: input.name.clone(),
            });
        }
    }

    // Intermediate partials buffer
    let partials_idx = bindings.len();
    bindings.push(Binding::StorageBuffer {
        set: partials_binding.0,
        binding: partials_binding.1,
        access: Access::ReadWrite,
        usage: BufferUsage::Intermediate,
        name: format!("{}_partials", entry.name),
    });

    // Output result buffer
    let result_idx = bindings.len();
    bindings.push(Binding::StorageBuffer {
        set: result_binding.0,
        binding: result_binding.1,
        access: Access::WriteOnly,
        usage: BufferUsage::Output,
        name: format!("{}_result", entry.name),
    });

    let workgroup_size = local_size.0;
    let pipeline = Pipeline::MultiCompute(MultiComputePipeline {
        bindings,
        stages: vec![
            ComputeStage {
                entry_point: phase1.name.clone(),
                workgroup_size: local_size,
                dispatch_size: DispatchSize::DerivedFromInputLength { workgroup_size },
                reads: input_binding_indices,
                writes: vec![partials_idx],
            },
            ComputeStage {
                entry_point: phase2.name.clone(),
                workgroup_size: (1, 1, 1),
                dispatch_size: DispatchSize::Fixed { x: 1, y: 1, z: 1 },
                reads: vec![partials_idx],
                writes: vec![result_idx],
            },
        ],
    });

    Some((vec![phase1, phase2], pipeline))
}

/// Build Phase 1 of reduce: each thread reduces its chunk and writes to partials buffer.
fn build_reduce_phase1(
    entry: &EntryPoint,
    source: &ArrayProvenance,
    reduce_function: &str,
    init: ValueId,
    captures: &[ValueId],
    elem_type: &Type<TypeName>,
    total_threads: u32,
    partials_binding: (u32, u32),
    local_size: (u32, u32, u32),
) -> Option<EntryPoint> {
    let unit_ty = Type::Constructed(TypeName::Unit, vec![]);
    let params: Vec<(Type<TypeName>, String)> =
        entry.inputs.iter().map(|i| (i.ty.clone(), i.name.clone())).collect();
    let builder = FuncBuilder::new(params, unit_ty.clone());
    let mut ctx = ParallelizeCtx::new(builder, entry);

    // Setup input based on provenance
    let (input_len, input_strategy_data) = match source {
        ArrayProvenance::EntryStorage {
            param_index,
            storage_binding,
            ..
        } => {
            let mut input = StorageInput::new(*param_index, *storage_binding);
            let (handle, len, _elem_ty) = input.setup(&mut ctx)?;
            (len, ReduceInputData::Storage { input, handle })
        }
        ArrayProvenance::Range { value } => {
            let mut input = RangeInput::new(*value, &entry.body)?;
            let (handle, len, _elem_ty) = input.setup(&mut ctx)?;
            (len, ReduceInputData::Range { input, handle })
        }
        ArrayProvenance::Unknown => return None,
    };

    // Setup partials output buffer
    let partials_output = StorageOutput::new(partials_binding.0, partials_binding.1);
    let partials_view = partials_output.setup(&mut ctx, elem_type)?;

    // Get thread ID and chunk bounds
    let (thread_id, chunk_start, chunk_end) = ctx.compute_thread_chunk(input_len, total_threads)?;

    // Remap init value and captures from original entry body
    let remapped_init = strategies::remap_entry_value(&mut ctx, init)?;
    let remapped_captures = ctx.remap_captures(captures)?;

    // Build reduction loop: acc = init; for i in chunk_start..chunk_end: acc = f(acc, elem)
    let u32_ty = ctx.u32_ty.clone();
    let bool_ty = ctx.bool_ty.clone();

    let (header, header_params) =
        ctx.builder.create_block_with_params(vec![elem_type.clone(), u32_ty.clone()]);
    let acc = header_params[0];
    let loop_index = header_params[1];
    let body_block = ctx.builder.create_block();
    let (exit_block, exit_params) = ctx.builder.create_block_with_params(vec![elem_type.clone()]);
    let final_acc = exit_params[0];
    ctx.builder.mark_loop_header(header, exit_block, body_block).ok()?;

    // Branch to header with (init, chunk_start)
    ctx.builder
        .terminate(Terminator::Branch {
            target: header,
            args: vec![remapped_init, chunk_start],
        })
        .ok()?;

    // Header: check i < chunk_end
    ctx.builder.switch_to_block(header).ok()?;
    let cond = ctx.builder.push_binop("<", loop_index, chunk_end, bool_ty).ok()?;
    ctx.builder
        .terminate(Terminator::CondBranch {
            cond,
            then_target: body_block,
            then_args: vec![],
            else_target: exit_block,
            else_args: vec![acc],
        })
        .ok()?;

    // Body: get element, call reduce function
    ctx.builder.switch_to_block(body_block).ok()?;

    let input_elem = match &input_strategy_data {
        ReduceInputData::Storage { input, handle } => input.get_element(&mut ctx, *handle, loop_index)?,
        ReduceInputData::Range { input, handle } => input.get_element(&mut ctx, *handle, loop_index)?,
    };

    // call_args: (acc, elem, captures...)
    let mut call_args = vec![acc, input_elem];
    call_args.extend(remapped_captures.iter().copied());
    let new_acc = ctx.push_call(reduce_function, call_args, elem_type.clone())?;

    let one = ctx.push_int("1")?;
    let next_i = ctx.push_binop("+", loop_index, one, u32_ty.clone())?;
    ctx.builder
        .terminate(Terminator::Branch {
            target: header,
            args: vec![new_acc, next_i],
        })
        .ok()?;

    // Exit: store final_acc to partials[thread_id]
    ctx.builder.switch_to_block(exit_block).ok()?;
    partials_output.store_result(&mut ctx, partials_view, thread_id, final_acc, elem_type)?;
    ctx.builder.terminate(Terminator::Return(None)).ok()?;

    let body = ctx.finish()?;

    // Build the partials output entry output
    let array_view_ty = Type::Constructed(
        TypeName::Array,
        vec![
            elem_type.clone(),
            Type::Constructed(TypeName::SizePlaceholder, vec![]),
            Type::Constructed(TypeName::ArrayVariantView, vec![]),
        ],
    );

    Some(EntryPoint {
        name: format!("{}_phase1_chunks", entry.name),
        body,
        execution_model: ExecutionModel::Compute { local_size },
        inputs: entry.inputs.clone(),
        outputs: vec![EntryOutput {
            ty: array_view_ty,
            decoration: None,
            storage_binding: Some(partials_binding),
        }],
        span: entry.span,
    })
}

/// Build Phase 2 of reduce: single thread combines partial results.
fn build_reduce_phase2(
    entry: &EntryPoint,
    reduce_function: &str,
    init: ValueId,
    captures: &[ValueId],
    elem_type: &Type<TypeName>,
    total_threads: u32,
    partials_binding: (u32, u32),
    result_binding: (u32, u32),
) -> Option<EntryPoint> {
    let unit_ty = Type::Constructed(TypeName::Unit, vec![]);
    let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
    let bool_ty = Type::Constructed(TypeName::Bool, vec![]);

    // Phase 2 has no entry-level params (it reads from partials buffer directly).
    // But we need the function referenced by reduce_function to exist in the program,
    // and we need to remap captures. Captures may reference entry params, so
    // Phase 2 needs the same input params as the original entry.
    let params: Vec<(Type<TypeName>, String)> =
        entry.inputs.iter().map(|i| (i.ty.clone(), i.name.clone())).collect();
    let builder = FuncBuilder::new(params, unit_ty.clone());
    let mut ctx = ParallelizeCtx::new(builder, entry);

    // Remap init and captures
    let remapped_init = strategies::remap_entry_value(&mut ctx, init)?;
    let remapped_captures = ctx.remap_captures(captures)?;

    // Setup partials input buffer (read from) — emit storage view directly
    let partials_view_ty = Type::Constructed(
        TypeName::Array,
        vec![
            elem_type.clone(),
            Type::Constructed(TypeName::SizePlaceholder, vec![]),
            Type::Constructed(TypeName::ArrayVariantView, vec![]),
        ],
    );
    let partials_view = ctx
        .builder
        .emit_storage_view(partials_binding.0, partials_binding.1, partials_view_ty.clone())
        .ok()?;

    // Setup result output buffer
    let result_output = StorageOutput::new(result_binding.0, result_binding.1);
    let result_view = result_output.setup(&mut ctx, elem_type)?;

    // Loop: acc = init; for t in 0..total_threads: acc = f(acc, partials[t])
    let total = ctx.push_int(&total_threads.to_string())?;
    let zero = ctx.push_int("0")?;

    let (header, header_params) =
        ctx.builder.create_block_with_params(vec![elem_type.clone(), u32_ty.clone()]);
    let acc = header_params[0];
    let loop_index = header_params[1];
    let body_block = ctx.builder.create_block();
    let (exit_block, exit_params) = ctx.builder.create_block_with_params(vec![elem_type.clone()]);
    let final_acc = exit_params[0];
    ctx.builder.mark_loop_header(header, exit_block, body_block).ok()?;

    ctx.builder
        .terminate(Terminator::Branch {
            target: header,
            args: vec![remapped_init, zero],
        })
        .ok()?;

    // Header: check t < total_threads
    ctx.builder.switch_to_block(header).ok()?;
    let cond = ctx.builder.push_binop("<", loop_index, total, bool_ty).ok()?;
    ctx.builder
        .terminate(Terminator::CondBranch {
            cond,
            then_target: body_block,
            then_args: vec![],
            else_target: exit_block,
            else_args: vec![acc],
        })
        .ok()?;

    // Body: load partials[t], acc = f(acc, elem)
    ctx.builder.switch_to_block(body_block).ok()?;

    let ptr = ctx.push_inst(
        InstKind::StorageViewIndex {
            view: partials_view,
            index: loop_index,
        },
        elem_type.clone(),
    )?;
    let effect_in = ctx.entry_effect();
    let partial_elem = ctx.builder.push_load(ptr, elem_type.clone(), effect_in).ok()?;

    let mut call_args = vec![acc, partial_elem];
    call_args.extend(remapped_captures.iter().copied());
    let new_acc = ctx.push_call(reduce_function, call_args, elem_type.clone())?;

    let one = ctx.push_int("1")?;
    let next_t = ctx.push_binop("+", loop_index, one, u32_ty.clone())?;
    ctx.builder
        .terminate(Terminator::Branch {
            target: header,
            args: vec![new_acc, next_t],
        })
        .ok()?;

    // Exit: store final acc to result[0]
    ctx.builder.switch_to_block(exit_block).ok()?;
    let zero_idx = ctx.push_int("0")?;
    result_output.store_result(&mut ctx, result_view, zero_idx, final_acc, elem_type)?;
    ctx.builder.terminate(Terminator::Return(None)).ok()?;

    let body = ctx.finish()?;

    let result_array_ty = Type::Constructed(
        TypeName::Array,
        vec![
            elem_type.clone(),
            Type::Constructed(TypeName::SizePlaceholder, vec![]),
            Type::Constructed(TypeName::ArrayVariantView, vec![]),
        ],
    );

    // Phase 2 inputs: partials buffer + any push-constant inputs from the
    // original entry that captures/init might reference.
    let mut phase2_inputs = Vec::new();
    // Add the partials buffer as a storage input
    phase2_inputs.push(EntryInput {
        name: format!("{}_partials", entry.name),
        ty: partials_view_ty.clone(),
        decoration: None,
        size_hint: None,
        storage_binding: Some(partials_binding),
        push_constant_offset: None,
    });
    // Carry forward push-constant inputs (captures may reference them)
    for input in &entry.inputs {
        if input.push_constant_offset.is_some() {
            phase2_inputs.push(input.clone());
        }
    }

    Some(EntryPoint {
        name: format!("{}_phase2_combine", entry.name),
        body,
        execution_model: ExecutionModel::Compute {
            local_size: (1, 1, 1),
        },
        inputs: phase2_inputs,
        outputs: vec![EntryOutput {
            ty: result_array_ty,
            decoration: None,
            storage_binding: Some(result_binding),
        }],
        span: entry.span,
    })
}

/// Helper enum to hold input strategy data after setup (avoids trait object boxing).
enum ReduceInputData {
    Storage {
        input: StorageInput,
        handle: ValueId,
    },
    Range {
        input: RangeInput,
        handle: RangeHandle,
    },
}
