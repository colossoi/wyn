//! SSA-level SOAC parallelization for compute shaders.
//!
//! This pass transforms compute shaders with parallelizable map loops
//! to use thread chunking for parallel execution.
//!
//! For a parallelizable compute shader:
//! ```text
//! entry compute(arr: []f32) -> []f32 = map(f, arr)
//! ```
//!
//! The transformation produces:
//! ```text
//! entry compute(arr: storage_view) -> () =
//!     let thread_id = __builtin_thread_id()
//!     let chunk_size = ceil(len / num_threads)
//!     let chunk_start = thread_id * chunk_size
//!     let chunk_end = min(chunk_start + chunk_size, len)
//!     // loop over chunk_start..chunk_end
//!     for i in chunk_start..chunk_end:
//!         output[i] = f(arr[i])
//! ```

use crate::ast::{NodeId, Span, TypeName};
use crate::tlc::to_ssa::{EntryOutput, ExecutionModel, SsaEntryPoint, SsaProgram};
use polytype::Type;

use super::ssa::{FuncBody, InstKind, Terminator};
use super::ssa_builder::FuncBuilder;
use super::ssa_soac_analysis::{ArrayProvenance, ParallelizableMap, analyze_program};

/// Parallelize SOACs in an SSA program.
pub fn parallelize_soacs(mut program: SsaProgram) -> SsaProgram {
    // Run analysis
    let analysis = analyze_program(&program);

    // Transform each compute entry point
    for entry in &mut program.entry_points {
        if let ExecutionModel::Compute { local_size } = entry.execution_model {
            if let Some(entry_analysis) = analysis.by_entry.get(&entry.name) {
                // Collect output storage buffer info before transformation
                // We need to preserve both the binding AND the original type for storage buffer creation
                let storage_outputs: Vec<_> =
                    entry.outputs.iter().filter(|o| o.storage_binding.is_some()).cloned().collect();

                if let Some(ref par_map) = entry_analysis.parallelizable_map {
                    // Try to parallelize
                    if let Some(new_body) = parallelize_entry(entry, par_map, local_size) {
                        entry.body = new_body;
                        // Keep original storage buffer outputs (they still need to be created)
                        // plus a unit output marker for the actual return value
                        if storage_outputs.is_empty() {
                            entry.outputs = vec![EntryOutput {
                                ty: Type::Constructed(TypeName::Unit, vec![]),
                                decoration: None,
                                storage_binding: None,
                            }];
                        } else {
                            // Keep the storage outputs - they need the original type for buffer creation
                            entry.outputs = storage_outputs.clone();
                        }
                    }
                } else {
                    // No parallelizable map - create single-thread fallback
                    if let Some(new_body) = create_single_thread_fallback(entry, local_size) {
                        entry.body = new_body;
                        if storage_outputs.is_empty() {
                            entry.outputs = vec![EntryOutput {
                                ty: Type::Constructed(TypeName::Unit, vec![]),
                                decoration: None,
                                storage_binding: None,
                            }];
                        } else {
                            entry.outputs = storage_outputs.clone();
                        }
                    }
                }
            }
        }
    }

    program
}

/// Create a parallelized version of a compute entry point.
fn parallelize_entry(
    entry: &SsaEntryPoint,
    par_map: &ParallelizableMap,
    local_size: (u32, u32, u32),
) -> Option<FuncBody> {
    let total_threads = local_size.0 * local_size.1 * local_size.2;

    // Extract storage info from provenance
    let (_storage_name, param_index, (set, binding)) = match &par_map.source {
        ArrayProvenance::EntryStorage {
            name,
            param_index,
            storage_binding,
        } => (name.clone(), *param_index, *storage_binding),
        _ => return None,
    };

    let dummy_span = Span {
        start_line: 1,
        start_col: 1,
        end_line: 1,
        end_col: 1,
    };
    let dummy_node_id = NodeId(0);
    // Use u32 for indices and lengths (all non-negative values)
    let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
    let bool_ty = Type::Constructed(TypeName::Str("bool"), vec![]);
    let unit_ty = Type::Constructed(TypeName::Unit, vec![]);

    // Get the array type from inputs
    let array_ty = entry.inputs.get(param_index)?.ty.clone();
    let elem_ty = match &array_ty {
        Type::Constructed(TypeName::Array, args) if !args.is_empty() => args[0].clone(),
        _ => return None,
    };

    // Create new function builder
    // Parameters: same as original but storage buffers become views
    let params: Vec<(Type<TypeName>, String)> =
        entry.inputs.iter().map(|i| (i.ty.clone(), i.name.clone())).collect();
    let mut builder = FuncBuilder::new(params, unit_ty.clone());

    // Get the input array parameter value
    let _input_param = builder.get_param(param_index);

    // Build preamble: storage views, thread_id, chunking

    // 1. Create storage view for input
    let zero = builder.push_int("0", u32_ty.clone(), dummy_span, dummy_node_id).ok()?;

    // Get storage length via intrinsic
    let set_val = builder.push_int(&set.to_string(), u32_ty.clone(), dummy_span, dummy_node_id).ok()?;
    let binding_val =
        builder.push_int(&binding.to_string(), u32_ty.clone(), dummy_span, dummy_node_id).ok()?;
    let storage_len = builder
        .push_intrinsic(
            "_w_storage_len",
            vec![set_val, binding_val],
            u32_ty.clone(),
            dummy_span,
            dummy_node_id,
        )
        .ok()?;

    // Create input storage view
    let input_view = builder
        .push_inst(
            InstKind::StorageView {
                set,
                binding,
                offset: zero,
                len: storage_len,
            },
            array_ty.clone(),
            dummy_span,
            dummy_node_id,
        )
        .ok()?;

    // 2. Create output storage view (binding after inputs)
    let output_binding = entry.inputs.len() as u32;
    let output_binding_val = builder
        .push_int(
            &output_binding.to_string(),
            u32_ty.clone(),
            dummy_span,
            dummy_node_id,
        )
        .ok()?;
    let output_len = builder
        .push_intrinsic(
            "_w_storage_len",
            vec![set_val, output_binding_val],
            u32_ty.clone(),
            dummy_span,
            dummy_node_id,
        )
        .ok()?;

    let zero2 = builder.push_int("0", u32_ty.clone(), dummy_span, dummy_node_id).ok()?;
    let output_view = builder
        .push_inst(
            InstKind::StorageView {
                set: 0,
                binding: output_binding,
                offset: zero2,
                len: output_len,
            },
            array_ty.clone(),
            dummy_span,
            dummy_node_id,
        )
        .ok()?;

    // 3. Get thread ID
    let thread_id = builder
        .push_intrinsic(
            "__builtin_thread_id",
            vec![],
            u32_ty.clone(),
            dummy_span,
            dummy_node_id,
        )
        .ok()?;

    // 4. Calculate chunk_size = ceil(len / total_threads)
    //    = (len + total_threads - 1) / total_threads
    let total_threads_val = builder
        .push_int(
            &total_threads.to_string(),
            u32_ty.clone(),
            dummy_span,
            dummy_node_id,
        )
        .ok()?;
    let threads_minus_1 = builder
        .push_int(
            &(total_threads - 1).to_string(),
            u32_ty.clone(),
            dummy_span,
            dummy_node_id,
        )
        .ok()?;
    let len_plus = builder
        .push_binop(
            "+",
            storage_len,
            threads_minus_1,
            u32_ty.clone(),
            dummy_span,
            dummy_node_id,
        )
        .ok()?;
    let chunk_size = builder
        .push_binop(
            "/",
            len_plus,
            total_threads_val,
            u32_ty.clone(),
            dummy_span,
            dummy_node_id,
        )
        .ok()?;

    // 5. chunk_start = thread_id * chunk_size
    let chunk_start = builder
        .push_binop(
            "*",
            thread_id,
            chunk_size,
            u32_ty.clone(),
            dummy_span,
            dummy_node_id,
        )
        .ok()?;

    // 6. chunk_end = min(chunk_start + chunk_size, len)
    let start_plus_size = builder
        .push_binop(
            "+",
            chunk_start,
            chunk_size,
            u32_ty.clone(),
            dummy_span,
            dummy_node_id,
        )
        .ok()?;
    let chunk_end = builder
        .push_call(
            "u32.min",
            vec![start_plus_size, storage_len],
            u32_ty.clone(),
            dummy_span,
            dummy_node_id,
        )
        .ok()?;

    // 7. Create loop: for i in chunk_start..chunk_end
    // Build loop blocks manually - just index, no accumulator (side-effect only loop)
    let (header, header_params) = builder.create_block_with_params(vec![u32_ty.clone()]);
    let loop_index = header_params[0];
    let body = builder.create_block();
    let exit = builder.create_block();
    builder.mark_loop_header(header, exit, body).ok()?;

    // Branch to header with initial index
    builder
        .terminate(Terminator::Branch {
            target: header,
            args: vec![chunk_start],
        })
        .ok()?;

    // Header: check i < chunk_end
    builder.switch_to_block(header).ok()?;
    let cond = builder
        .push_binop(
            "<",
            loop_index,
            chunk_end,
            bool_ty.clone(),
            dummy_span,
            dummy_node_id,
        )
        .ok()?;
    builder
        .terminate(Terminator::CondBranch {
            cond,
            then_target: body,
            then_args: vec![],
            else_target: exit,
            else_args: vec![],
        })
        .ok()?;

    // Body: output[i] = f(input[i])
    builder.switch_to_block(body).ok()?;

    // Index into input view
    let input_ptr = builder
        .push_inst(
            InstKind::StorageViewIndex {
                view: input_view,
                index: loop_index,
            },
            elem_ty.clone(), // pointer type, simplified
            dummy_span,
            dummy_node_id,
        )
        .ok()?;

    // Load element - use entry_effect and alloc new effects
    let effect_in = builder.entry_effect();
    let effect_out = builder.alloc_effect();
    let input_elem = builder
        .push_inst(
            InstKind::Load {
                ptr: input_ptr,
                effect_in,
                effect_out,
            },
            elem_ty.clone(),
            dummy_span,
            dummy_node_id,
        )
        .ok()?;

    // Apply map function
    let result_elem = builder
        .push_call(
            &par_map.map_function,
            vec![input_elem],
            elem_ty.clone(),
            dummy_span,
            dummy_node_id,
        )
        .ok()?;

    // Index into output view
    let output_ptr = builder
        .push_inst(
            InstKind::StorageViewIndex {
                view: output_view,
                index: loop_index,
            },
            elem_ty.clone(),
            dummy_span,
            dummy_node_id,
        )
        .ok()?;

    // Store result - chain effect from load
    let effect_out2 = builder.alloc_effect();
    builder
        .push_inst(
            InstKind::Store {
                ptr: output_ptr,
                value: result_elem,
                effect_in: effect_out,
                effect_out: effect_out2,
            },
            unit_ty.clone(),
            dummy_span,
            dummy_node_id,
        )
        .ok()?;

    // Increment index and branch back to header
    let one = builder.push_int("1", u32_ty.clone(), dummy_span, dummy_node_id).ok()?;
    let next_i =
        builder.push_binop("+", loop_index, one, u32_ty.clone(), dummy_span, dummy_node_id).ok()?;
    builder
        .terminate(Terminator::Branch {
            target: header,
            args: vec![next_i],
        })
        .ok()?;

    // Exit: return unit
    builder.switch_to_block(exit).ok()?;
    builder.terminate(Terminator::ReturnUnit).ok()?;

    builder.finish().ok()
}

/// Create a single-thread fallback for non-parallelizable compute shaders.
/// Only thread 0 executes the body.
fn create_single_thread_fallback(entry: &SsaEntryPoint, _local_size: (u32, u32, u32)) -> Option<FuncBody> {
    let dummy_span = Span {
        start_line: 1,
        start_col: 1,
        end_line: 1,
        end_col: 1,
    };
    let dummy_node_id = NodeId(0);
    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
    let bool_ty = Type::Constructed(TypeName::Str("bool"), vec![]);
    let unit_ty = Type::Constructed(TypeName::Unit, vec![]);

    // Create new function with same parameters
    let params: Vec<(Type<TypeName>, String)> =
        entry.inputs.iter().map(|i| (i.ty.clone(), i.name.clone())).collect();
    let mut builder = FuncBuilder::new(params, unit_ty.clone());

    // Get thread ID
    let thread_id = builder
        .push_intrinsic(
            "__builtin_thread_id",
            vec![],
            i32_ty.clone(),
            dummy_span,
            dummy_node_id,
        )
        .ok()?;

    // Check if thread_id == 0
    let zero = builder.push_int("0", i32_ty.clone(), dummy_span, dummy_node_id).ok()?;
    let is_thread_zero =
        builder.push_binop("==", thread_id, zero, bool_ty.clone(), dummy_span, dummy_node_id).ok()?;

    // Create then/else/merge blocks
    let then_block = builder.create_block();
    let else_block = builder.create_block();
    let merge_block = builder.create_block();

    builder
        .terminate(Terminator::CondBranch {
            cond: is_thread_zero,
            then_target: then_block,
            then_args: vec![],
            else_target: else_block,
            else_args: vec![],
        })
        .ok()?;

    // Then block: execute original body (simplified - just return unit for now)
    // TODO: Copy the original body here
    builder.switch_to_block(then_block).ok()?;
    builder
        .terminate(Terminator::Branch {
            target: merge_block,
            args: vec![],
        })
        .ok()?;

    // Else block: skip
    builder.switch_to_block(else_block).ok()?;
    builder
        .terminate(Terminator::Branch {
            target: merge_block,
            args: vec![],
        })
        .ok()?;

    // Merge block: return unit
    builder.switch_to_block(merge_block).ok()?;
    builder.terminate(Terminator::ReturnUnit).ok()?;

    builder.finish().ok()
}
