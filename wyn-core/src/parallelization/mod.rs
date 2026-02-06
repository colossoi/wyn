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

use crate::ast::{NodeId, Span, TypeName};
use crate::mir::ssa::{EffectToken, FuncBody, InstKind, Terminator, ValueId};
use crate::mir::ssa_builder::FuncBuilder;
use crate::mir::ssa_soac_analysis::{ArrayProvenance, ParallelizableMap, analyze_program};
use crate::tlc::to_ssa::{EntryOutput, ExecutionModel, SsaEntryPoint, SsaProgram};
use polytype::Type;

use std::collections::HashMap;

pub use strategies::{
    InputStrategy, OutputStrategy, RangeInput, StorageInput, StorageOutput, remap_value,
};

/// Context passed to strategies during parallelization.
/// Contains the builder and common utilities.
pub struct ParallelizeCtx<'a> {
    pub builder: FuncBuilder,
    pub entry: &'a SsaEntryPoint,
    pub span: Span,
    pub node_id: NodeId,
    // Common types
    pub u32_ty: Type<TypeName>,
    pub i32_ty: Type<TypeName>,
    pub bool_ty: Type<TypeName>,
    pub unit_ty: Type<TypeName>,
}

impl<'a> ParallelizeCtx<'a> {
    pub fn new(builder: FuncBuilder, entry: &'a SsaEntryPoint) -> Self {
        let span = entry.span;
        let node_id = NodeId(0); // Dummy node ID for generated code
        Self {
            builder,
            entry,
            span,
            node_id,
            u32_ty: Type::Constructed(TypeName::UInt(32), vec![]),
            i32_ty: Type::Constructed(TypeName::Int(32), vec![]),
            bool_ty: Type::Constructed(TypeName::Str("bool"), vec![]),
            unit_ty: Type::Constructed(TypeName::Unit, vec![]),
        }
    }

    /// Push an integer constant.
    pub fn push_int(&mut self, value: &str) -> Option<ValueId> {
        self.builder.push_int(value, self.u32_ty.clone(), self.span, self.node_id).ok()
    }

    /// Push an i32 integer constant.
    pub fn push_i32(&mut self, value: &str) -> Option<ValueId> {
        self.builder.push_int(value, self.i32_ty.clone(), self.span, self.node_id).ok()
    }

    /// Push a binary operation.
    pub fn push_binop(
        &mut self,
        op: &str,
        lhs: ValueId,
        rhs: ValueId,
        ty: Type<TypeName>,
    ) -> Option<ValueId> {
        self.builder.push_binop(op, lhs, rhs, ty, self.span, self.node_id).ok()
    }

    /// Push an intrinsic call.
    pub fn push_intrinsic(
        &mut self,
        name: &str,
        args: Vec<ValueId>,
        ty: Type<TypeName>,
    ) -> Option<ValueId> {
        self.builder.push_intrinsic(name, args, ty, self.span, self.node_id).ok()
    }

    /// Push a function call.
    pub fn push_call(&mut self, func: &str, args: Vec<ValueId>, ty: Type<TypeName>) -> Option<ValueId> {
        self.builder.push_call(func, args, ty, self.span, self.node_id).ok()
    }

    /// Push an instruction.
    pub fn push_inst(&mut self, kind: InstKind, ty: Type<TypeName>) -> Option<ValueId> {
        self.builder.push_inst(kind, ty, self.span, self.node_id).ok()
    }

    /// Allocate an effect token.
    pub fn alloc_effect(&mut self) -> EffectToken {
        self.builder.alloc_effect()
    }

    /// Get the entry effect token.
    pub fn entry_effect(&self) -> EffectToken {
        self.builder.entry_effect()
    }

    /// Consume the context and finish building, returning the function body.
    pub fn finish(self) -> Option<FuncBody> {
        self.builder.finish().ok()
    }
}

/// Parallelize SOACs in an SSA program.
pub fn parallelize_soacs(mut program: SsaProgram) -> SsaProgram {
    let analysis = analyze_program(&program);

    for entry in &mut program.entry_points {
        if let ExecutionModel::Compute { local_size } = entry.execution_model {
            if let Some(entry_analysis) = analysis.by_entry.get(&entry.name) {
                let storage_outputs: Vec<_> =
                    entry.outputs.iter().filter(|o| o.storage_binding.is_some()).cloned().collect();

                if let Some(ref par_map) = entry_analysis.parallelizable_map {
                    if let Some(new_body) = parallelize_entry(entry, par_map, local_size) {
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
                    } else {
                    }
                }
                // Note: single-thread fallback removed - all compute shaders should be parallelizable
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

    // Create input and output strategies based on provenance
    let (input_strategy, output_strategy): (Box<dyn InputStrategy>, Box<dyn OutputStrategy>) =
        match &par_map.source {
            ArrayProvenance::EntryStorage {
                param_index,
                storage_binding,
                ..
            } => {
                let input = StorageInput::new(*param_index, *storage_binding);
                let output_binding = entry.inputs.len() as u32;
                let output = StorageOutput::new(0, output_binding);
                (Box::new(input), Box::new(output))
            }
            ArrayProvenance::Range { value } => {
                let input = RangeInput::new(*value, &entry.body)?;
                let output = StorageOutput::new(0, 0); // binding 0 for range-only maps
                (Box::new(input), Box::new(output))
            }
            ArrayProvenance::Unknown => return None,
        };

    // Build the parallelized function
    let unit_ty = Type::Constructed(TypeName::Unit, vec![]);
    let params: Vec<(Type<TypeName>, String)> =
        entry.inputs.iter().map(|i| (i.ty.clone(), i.name.clone())).collect();
    let builder = FuncBuilder::new(params, unit_ty.clone());

    let mut ctx = ParallelizeCtx::new(builder, entry);

    // 1. Setup input and output strategies
    let (input_len, _input_elem_ty) = input_strategy.setup(&mut ctx)?;
    let output_elem_ty = &par_map.map_loop.map_result_ty;
    let output_view = output_strategy.setup(&mut ctx, output_elem_ty)?;

    // 2. Get thread ID and calculate chunk bounds
    let thread_id = ctx.push_intrinsic("__builtin_thread_id", vec![], ctx.u32_ty.clone())?;

    let total_threads_val = ctx.push_int(&total_threads.to_string())?;
    let threads_minus_1 = ctx.push_int(&(total_threads - 1).to_string())?;

    // chunk_size = ceil(len / total_threads) = (len + total_threads - 1) / total_threads
    let len_plus = ctx.push_binop("+", input_len, threads_minus_1, ctx.u32_ty.clone())?;
    let chunk_size = ctx.push_binop("/", len_plus, total_threads_val, ctx.u32_ty.clone())?;

    // chunk_start = thread_id * chunk_size
    let chunk_start = ctx.push_binop("*", thread_id, chunk_size, ctx.u32_ty.clone())?;

    // chunk_end = min(chunk_start + chunk_size, len)
    let start_plus_size = ctx.push_binop("+", chunk_start, chunk_size, ctx.u32_ty.clone())?;
    let chunk_end = ctx.push_call("u32.min", vec![start_plus_size, input_len], ctx.u32_ty.clone())?;

    // 3. Create loop structure
    let u32_ty = ctx.u32_ty.clone();
    let bool_ty = ctx.bool_ty.clone();
    let span = ctx.span;
    let node_id = ctx.node_id;

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
    let cond = ctx.builder.push_binop("<", loop_index, chunk_end, bool_ty.clone(), span, node_id).ok()?;
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

    let input_elem = input_strategy.get_element(&mut ctx, loop_index)?;

    // Build the full argument list for the map function.
    // The original call may have captured variables (e.g. from defunctionalized lambdas)
    // in addition to the loop element. Remap each original arg's dependency cone into the
    // new builder, substituting the element arg position with the new input_elem.
    let result_ty = &par_map.map_loop.map_result_ty;
    let call_args = {
        // Shared memo pre-seeded with entry param → builder param mappings.
        // Shared across all captured args to deduplicate common dependencies.
        let mut remap_memo: HashMap<ValueId, ValueId> = entry
            .body
            .params
            .iter()
            .enumerate()
            .map(|(i, (src, _, _))| (*src, ctx.builder.get_param(i)))
            .collect();

        let span = ctx.span;
        let node_id = ctx.node_id;
        let mut args = Vec::new();
        for (i, &orig_arg) in par_map.map_loop.map_call_args.iter().enumerate() {
            if i == par_map.map_loop.element_arg_index {
                args.push(input_elem);
            } else {
                // Recursively remap the value and its dependency cone
                let new_val =
                    remap_value(&entry.body, orig_arg, &mut ctx.builder, &mut remap_memo, span, node_id)?;
                args.push(new_val);
            }
        }
        args
    };

    // Apply map function with the correct return type
    let result_elem = ctx.push_call(&par_map.map_function, call_args, result_ty.clone())?;

    // Store result using the map function's return type (not the input elem type)
    output_strategy.store_result(&mut ctx, output_view, loop_index, result_elem, result_ty)?;

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
    ctx.builder.terminate(Terminator::ReturnUnit).ok()?;

    ctx.finish()
}
