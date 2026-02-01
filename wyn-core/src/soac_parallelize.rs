//! SOAC parallelization pass for compute shaders.
//!
//! This pass transforms compute shaders that map over input arrays into
//! parallel versions where each thread processes a chunk of elements.
//!
//! Transformation for direct maps:
//! ```text
//! entry compute(arr: []f32) []f32 = map(|x| x * 2.0, arr)
//! ```
//! Becomes:
//! ```text
//! entry compute(arr: []f32) ()  =
//!   let __thread_id = __builtin_thread_id() in
//!   let __chunk_start = __thread_id * __chunk_size in
//!   let __chunk_end = min(__chunk_start + __chunk_size, length(arr)) in
//!   let __local_chunk = arr[__chunk_start..__chunk_end] in
//!   map_into(|x| x * 2.0, __local_chunk, _output, __chunk_start)
//! ```
//!
//! For interprocedural maps (map inside helper function):
//! ```text
//! def apply_double(arr: []f32) []f32 = map(double, arr)
//! entry main(data: []f32) []f32 = apply_double(data)
//! ```
//! Becomes:
//! ```text
//! entry main(data: []f32) () =
//!   let __thread_id = ... in
//!   let __local_chunk = data[__chunk_start..__chunk_end] in
//!   apply_double(__local_chunk)  -- helper receives the chunk View
//! ```

use crate::ast::{NodeId, Span, TypeName};
use crate::mir::{
    ArrayBacking, Body, Def, EntryInput, EntryOutput, ExecutionModel, Expr, ExprId, LocalDecl, LocalId,
    LocalKind, Program,
    soac_analysis::{self, ArrayProvenance, MirSoacAnalysis, ParallelizableMap},
};
use polytype::Type;
use std::collections::HashMap;

/// Parallelize SOACs in compute shaders.
pub fn parallelize_soacs(program: Program) -> Program {
    // Run MIR SOAC analysis
    let analysis = soac_analysis::analyze_program(&program);

    // Transform each definition
    let defs = program.defs.into_iter().map(|def| parallelize_def(def, &analysis)).collect();

    Program {
        defs,
        lambda_registry: program.lambda_registry,
    }
}

/// Parallelize a single definition if it's a compute entry point with a parallelizable map.
fn parallelize_def(def: Def, analysis: &MirSoacAnalysis) -> Def {
    match def {
        Def::EntryPoint {
            id,
            name,
            execution_model: ExecutionModel::Compute { local_size },
            inputs,
            outputs,
            body,
            span,
        } => {
            // Check if this entry point has a parallelizable map
            if let Some(map_info) = soac_analysis::find_root_parallelizable_map(analysis, &name) {
                // Try to transform the body
                if let Some((new_body, new_outputs)) =
                    try_parallelize_body(&body, &inputs, &outputs, map_info, local_size)
                {
                    return Def::EntryPoint {
                        id,
                        name,
                        execution_model: ExecutionModel::Compute { local_size },
                        inputs,
                        outputs: new_outputs,
                        body: new_body,
                        span,
                    };
                }
            }

            // Can't parallelize - create single-thread fallback
            let (new_body, new_outputs) = create_single_thread_fallback(&body, &inputs, &outputs);
            Def::EntryPoint {
                id,
                name,
                execution_model: ExecutionModel::Compute { local_size },
                inputs,
                outputs: new_outputs,
                body: new_body,
                span,
            }
        }
        other => other,
    }
}

/// Create a single-thread fallback for compute shaders that can't be parallelized.
///
/// This runs the shader body only on thread 0, with input parameters rewritten
/// to use storage-backed arrays instead of locals.
fn create_single_thread_fallback(
    body: &Body,
    inputs: &[EntryInput],
    _outputs: &[EntryOutput],
) -> (Body, Vec<EntryOutput>) {
    let mut new_body = Body::new();
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

    // Create local mapping for non-storage inputs
    let mut local_map: HashMap<LocalId, LocalId> = HashMap::new();
    let mut storage_locals: HashMap<LocalId, (String, Type<TypeName>)> = HashMap::new();

    for (old_idx, local) in body.locals.iter().enumerate() {
        let old_id = LocalId(old_idx as u32);

        // Check if this local is a storage input
        let is_storage_input = inputs.iter().any(|input| {
            input.local == old_id
                && matches!(
                    &input.ty,
                    Type::Constructed(TypeName::Array, args) if args.len() >= 2 && matches!(&args[1], Type::Constructed(TypeName::ArrayVariantView, _))
                )
        });

        if is_storage_input {
            // Find the input to get its name
            if let Some(input) = inputs.iter().find(|i| i.local == old_id) {
                storage_locals.insert(old_id, (input.name.clone(), input.ty.clone()));
            }
            // Still need to allocate the local for the mapping
            let new_id = new_body.alloc_local(local.clone());
            local_map.insert(old_id, new_id);
        } else {
            let new_id = new_body.alloc_local(local.clone());
            local_map.insert(old_id, new_id);
        }
    }

    // Create thread_id local
    let thread_id_local = new_body.alloc_local(LocalDecl {
        name: "__thread_id".to_string(),
        span: dummy_span,
        ty: i32_ty.clone(),
        kind: LocalKind::Let,
    });

    // 1. __builtin_thread_id()
    let thread_id_intrinsic = new_body.alloc_expr(
        Expr::Intrinsic {
            name: "__builtin_thread_id".to_string(),
            args: vec![],
        },
        i32_ty.clone(),
        dummy_span,
        dummy_node_id,
    );

    // 2. thread_id == 0 check
    let zero = new_body.alloc_expr(
        Expr::Int("0".to_string()),
        i32_ty.clone(),
        dummy_span,
        dummy_node_id,
    );
    let thread_id_ref = new_body.alloc_expr(
        Expr::Local(thread_id_local),
        i32_ty.clone(),
        dummy_span,
        dummy_node_id,
    );
    let is_thread_zero = new_body.alloc_expr(
        Expr::BinOp {
            op: "==".to_string(),
            lhs: thread_id_ref,
            rhs: zero,
        },
        bool_ty.clone(),
        dummy_span,
        dummy_node_id,
    );

    // 3. Copy body with storage rewrites
    let body_copy =
        copy_expr_tree_with_storage(&mut new_body, body, body.root, &local_map, &storage_locals);

    // 4. Unit value for early return
    let unit_expr = new_body.alloc_expr(Expr::Unit, unit_ty.clone(), dummy_span, dummy_node_id);

    // 5. if thread_id == 0 then body else ()
    let if_expr = new_body.alloc_expr(
        Expr::If {
            cond: is_thread_zero,
            then_: body_copy,
            else_: unit_expr,
        },
        unit_ty.clone(),
        dummy_span,
        dummy_node_id,
    );

    // 6. Wrap in let for thread_id
    let root_expr = new_body.alloc_expr(
        Expr::Let {
            local: thread_id_local,
            rhs: thread_id_intrinsic,
            body: if_expr,
        },
        unit_ty.clone(),
        dummy_span,
        dummy_node_id,
    );

    new_body.set_root(root_expr);

    // Output becomes unit since we're writing to storage directly (or just discarding)
    let new_outputs = vec![EntryOutput {
        ty: unit_ty,
        decoration: None,
    }];

    (new_body, new_outputs)
}

/// Copy an expression tree, rewriting storage locals to Storage-backed arrays.
fn copy_expr_tree_with_storage(
    dest: &mut Body,
    src: &Body,
    expr_id: ExprId,
    local_map: &HashMap<LocalId, LocalId>,
    storage_locals: &HashMap<LocalId, (String, Type<TypeName>)>,
) -> ExprId {
    let ty = src.get_type(expr_id).clone();
    let span = src.get_span(expr_id);
    let node_id = src.node_ids[expr_id.index()];
    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
    let dummy_node_id = NodeId(0);

    let new_expr = match src.get_expr(expr_id) {
        Expr::Local(local_id) => {
            // Check if this is a storage local that needs rewriting
            // Storage locals are now represented as View-typed Locals
            // Just use the mapped local directly
            let _ = storage_locals; // Storage locals are handled by lowering preamble
            Expr::Local(*local_map.get(local_id).unwrap_or(local_id))
        }
        Expr::Global(name) => Expr::Global(name.clone()),
        Expr::Extern(linkage) => Expr::Extern(linkage.clone()),
        Expr::Int(s) => Expr::Int(s.clone()),
        Expr::Float(s) => Expr::Float(s.clone()),
        Expr::Bool(b) => Expr::Bool(*b),
        Expr::Unit => Expr::Unit,
        Expr::String(s) => Expr::String(s.clone()),
        Expr::Array { backing, size } => {
            let new_size = copy_expr_tree_with_storage(dest, src, *size, local_map, storage_locals);
            let new_backing = match backing {
                ArrayBacking::Literal(elems) => {
                    let new_elems: Vec<ExprId> = elems
                        .iter()
                        .map(|e| copy_expr_tree_with_storage(dest, src, *e, local_map, storage_locals))
                        .collect();
                    ArrayBacking::Literal(new_elems)
                }
                ArrayBacking::Range { start, step, kind } => {
                    let new_start =
                        copy_expr_tree_with_storage(dest, src, *start, local_map, storage_locals);
                    let new_step =
                        step.map(|s| copy_expr_tree_with_storage(dest, src, s, local_map, storage_locals));
                    ArrayBacking::Range {
                        start: new_start,
                        step: new_step,
                        kind: *kind,
                    }
                }
            };
            Expr::Array {
                backing: new_backing,
                size: new_size,
            }
        }
        Expr::Tuple(elems) => {
            let new_elems: Vec<ExprId> = elems
                .iter()
                .map(|e| copy_expr_tree_with_storage(dest, src, *e, local_map, storage_locals))
                .collect();
            Expr::Tuple(new_elems)
        }
        Expr::Vector(elems) => {
            let new_elems: Vec<ExprId> = elems
                .iter()
                .map(|e| copy_expr_tree_with_storage(dest, src, *e, local_map, storage_locals))
                .collect();
            Expr::Vector(new_elems)
        }
        Expr::Matrix(rows) => {
            let new_rows: Vec<Vec<ExprId>> = rows
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|e| copy_expr_tree_with_storage(dest, src, *e, local_map, storage_locals))
                        .collect()
                })
                .collect();
            Expr::Matrix(new_rows)
        }
        Expr::BinOp { op, lhs, rhs } => {
            let new_lhs = copy_expr_tree_with_storage(dest, src, *lhs, local_map, storage_locals);
            let new_rhs = copy_expr_tree_with_storage(dest, src, *rhs, local_map, storage_locals);
            Expr::BinOp {
                op: op.clone(),
                lhs: new_lhs,
                rhs: new_rhs,
            }
        }
        Expr::UnaryOp { op, operand } => {
            let new_operand = copy_expr_tree_with_storage(dest, src, *operand, local_map, storage_locals);
            Expr::UnaryOp {
                op: op.clone(),
                operand: new_operand,
            }
        }
        Expr::Call { func, args } => {
            let new_args: Vec<ExprId> = args
                .iter()
                .map(|e| copy_expr_tree_with_storage(dest, src, *e, local_map, storage_locals))
                .collect();
            Expr::Call {
                func: func.clone(),
                args: new_args,
            }
        }
        Expr::Intrinsic { name, args } => {
            let new_args: Vec<ExprId> = args
                .iter()
                .map(|e| copy_expr_tree_with_storage(dest, src, *e, local_map, storage_locals))
                .collect();
            Expr::Intrinsic {
                name: name.clone(),
                args: new_args,
            }
        }
        Expr::If { cond, then_, else_ } => {
            let new_cond = copy_expr_tree_with_storage(dest, src, *cond, local_map, storage_locals);
            let new_then = copy_expr_tree_with_storage(dest, src, *then_, local_map, storage_locals);
            let new_else = copy_expr_tree_with_storage(dest, src, *else_, local_map, storage_locals);
            Expr::If {
                cond: new_cond,
                then_: new_then,
                else_: new_else,
            }
        }
        Expr::Let { local, rhs, body } => {
            let new_rhs = copy_expr_tree_with_storage(dest, src, *rhs, local_map, storage_locals);
            let new_body = copy_expr_tree_with_storage(dest, src, *body, local_map, storage_locals);
            Expr::Let {
                local: *local_map.get(local).unwrap_or(local),
                rhs: new_rhs,
                body: new_body,
            }
        }
        Expr::Materialize(inner) => {
            let new_inner = copy_expr_tree_with_storage(dest, src, *inner, local_map, storage_locals);
            Expr::Materialize(new_inner)
        }
        Expr::Load { ptr } => {
            let new_ptr = copy_expr_tree_with_storage(dest, src, *ptr, local_map, storage_locals);
            Expr::Load { ptr: new_ptr }
        }
        Expr::Store { ptr, value } => {
            let new_ptr = copy_expr_tree_with_storage(dest, src, *ptr, local_map, storage_locals);
            let new_value = copy_expr_tree_with_storage(dest, src, *value, local_map, storage_locals);
            Expr::Store {
                ptr: new_ptr,
                value: new_value,
            }
        }
        Expr::Loop { .. } => {
            // Loops are complex - just copy unchanged for now
            src.get_expr(expr_id).clone()
        }
        Expr::Attributed { attributes, expr } => {
            let new_expr_inner = copy_expr_tree_with_storage(dest, src, *expr, local_map, storage_locals);
            Expr::Attributed {
                attributes: attributes.clone(),
                expr: new_expr_inner,
            }
        }
        Expr::View { ptr, len } => {
            let new_ptr = copy_expr_tree_with_storage(dest, src, *ptr, local_map, storage_locals);
            let new_len = copy_expr_tree_with_storage(dest, src, *len, local_map, storage_locals);
            Expr::View { ptr: new_ptr, len: new_len }
        }
        Expr::ViewPtr { view } => {
            let new_view = copy_expr_tree_with_storage(dest, src, *view, local_map, storage_locals);
            Expr::ViewPtr { view: new_view }
        }
        Expr::ViewLen { view } => {
            let new_view = copy_expr_tree_with_storage(dest, src, *view, local_map, storage_locals);
            Expr::ViewLen { view: new_view }
        }
        Expr::PtrAdd { ptr, offset } => {
            let new_ptr = copy_expr_tree_with_storage(dest, src, *ptr, local_map, storage_locals);
            let new_offset = copy_expr_tree_with_storage(dest, src, *offset, local_map, storage_locals);
            Expr::PtrAdd { ptr: new_ptr, offset: new_offset }
        }
    };

    dest.alloc_expr(new_expr, ty, span, node_id)
}

/// Try to parallelize a compute shader body using the analysis results.
fn try_parallelize_body(
    body: &Body,
    inputs: &[EntryInput],
    outputs: &[EntryOutput],
    map_info: &ParallelizableMap,
    local_size: (u32, u32, u32),
) -> Option<(Body, Vec<EntryOutput>)> {
    // Extract info based on provenance
    let (storage_name, storage_local, array_ty) = match &map_info.source {
        ArrayProvenance::EntryStorage { name, local } => {
            let input = inputs.iter().find(|i| i.local == *local)?;
            (name.clone(), *local, input.ty.clone())
        }
        ArrayProvenance::Range { .. } => {
            // TODO: Handle range chunking (adjust bounds instead of creating View)
            return None;
        }
        ArrayProvenance::Unknown => return None,
    };

    // Calculate total threads from workgroup size
    let total_threads = local_size.0 * local_size.1 * local_size.2;

    // Find the size hint for the mapped array (if available)
    let size_hint: Option<u32> = inputs.iter().find(|i| i.local == storage_local).and_then(|i| i.size_hint);

    // Create new body with the transformation
    let mut new_body = Body::new();
    let dummy_span = Span {
        start_line: 1,
        start_col: 1,
        end_line: 1,
        end_col: 1,
    };
    let dummy_node_id = NodeId(0);
    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);

    // Copy locals from original body (parameters)
    let mut local_map: HashMap<LocalId, LocalId> = HashMap::new();
    for (old_idx, local) in body.locals.iter().enumerate() {
        let old_id = LocalId(old_idx as u32);
        let new_id = new_body.alloc_local(local.clone());
        local_map.insert(old_id, new_id);
    }

    // Create output local (for output View that lowering will set up)
    let output_local = new_body.alloc_local(LocalDecl {
        name: "_output".to_string(),
        span: dummy_span,
        ty: array_ty.clone(),
        kind: LocalKind::Param,
    });

    // Create locals for chunking
    let thread_id_local = new_body.alloc_local(LocalDecl {
        name: "__thread_id".to_string(),
        span: dummy_span,
        ty: i32_ty.clone(),
        kind: LocalKind::Let,
    });

    let chunk_start_local = new_body.alloc_local(LocalDecl {
        name: "__chunk_start".to_string(),
        span: dummy_span,
        ty: i32_ty.clone(),
        kind: LocalKind::Let,
    });

    let chunk_size_local = new_body.alloc_local(LocalDecl {
        name: "__chunk_size".to_string(),
        span: dummy_span,
        ty: i32_ty.clone(),
        kind: LocalKind::Let,
    });

    let chunk_end_local = new_body.alloc_local(LocalDecl {
        name: "__chunk_end".to_string(),
        span: dummy_span,
        ty: i32_ty.clone(),
        kind: LocalKind::Let,
    });

    let array_len_local = new_body.alloc_local(LocalDecl {
        name: "__array_len".to_string(),
        span: dummy_span,
        ty: i32_ty.clone(),
        kind: LocalKind::Let,
    });

    // The mapped local now holds a View {ptr, len} directly
    let mapped_storage_local = *local_map.get(&storage_local).unwrap_or(&storage_local);

    // 1. Create array length expression
    let array_len_expr = if let Some(hint) = size_hint {
        new_body.alloc_expr(
            Expr::Int(hint.to_string()),
            i32_ty.clone(),
            dummy_span,
            dummy_node_id,
        )
    } else {
        // Runtime: call length intrinsic on the storage buffer local
        let arr_ref = new_body.alloc_expr(
            Expr::Local(mapped_storage_local),
            array_ty.clone(),
            dummy_span,
            dummy_node_id,
        );
        new_body.alloc_expr(
            Expr::Intrinsic {
                name: "_w_intrinsic_length".to_string(),
                args: vec![arr_ref],
            },
            i32_ty.clone(),
            dummy_span,
            dummy_node_id,
        )
    };

    // 2. The base array is just the storage local (which holds a View)
    let base_array_expr_id = new_body.alloc_expr(
        Expr::Local(mapped_storage_local),
        array_ty.clone(),
        dummy_span,
        dummy_node_id,
    );

    // 3. __builtin_thread_id() intrinsic
    let thread_id_intrinsic = new_body.alloc_expr(
        Expr::Intrinsic {
            name: "__builtin_thread_id".to_string(),
            args: vec![],
        },
        i32_ty.clone(),
        dummy_span,
        dummy_node_id,
    );

    // 4. __chunk_size = ceil(array_len / total_threads)
    let total_threads_expr = new_body.alloc_expr(
        Expr::Int(total_threads.to_string()),
        i32_ty.clone(),
        dummy_span,
        dummy_node_id,
    );
    let total_threads_minus_1 = new_body.alloc_expr(
        Expr::Int((total_threads - 1).to_string()),
        i32_ty.clone(),
        dummy_span,
        dummy_node_id,
    );
    let array_len_ref_for_chunk = new_body.alloc_expr(
        Expr::Local(array_len_local),
        i32_ty.clone(),
        dummy_span,
        dummy_node_id,
    );
    let len_plus_threads_minus_1 = new_body.alloc_expr(
        Expr::BinOp {
            op: "+".to_string(),
            lhs: array_len_ref_for_chunk,
            rhs: total_threads_minus_1,
        },
        i32_ty.clone(),
        dummy_span,
        dummy_node_id,
    );
    let chunk_size_expr = new_body.alloc_expr(
        Expr::BinOp {
            op: "/".to_string(),
            lhs: len_plus_threads_minus_1,
            rhs: total_threads_expr,
        },
        i32_ty.clone(),
        dummy_span,
        dummy_node_id,
    );

    // 5. __chunk_start = __thread_id * __chunk_size
    let thread_id_ref = new_body.alloc_expr(
        Expr::Local(thread_id_local),
        i32_ty.clone(),
        dummy_span,
        dummy_node_id,
    );
    let chunk_size_ref = new_body.alloc_expr(
        Expr::Local(chunk_size_local),
        i32_ty.clone(),
        dummy_span,
        dummy_node_id,
    );
    let chunk_start_expr = new_body.alloc_expr(
        Expr::BinOp {
            op: "*".to_string(),
            lhs: thread_id_ref,
            rhs: chunk_size_ref,
        },
        i32_ty.clone(),
        dummy_span,
        dummy_node_id,
    );

    // 6. __chunk_end = min(__chunk_start + __chunk_size, array_len)
    let chunk_start_ref = new_body.alloc_expr(
        Expr::Local(chunk_start_local),
        i32_ty.clone(),
        dummy_span,
        dummy_node_id,
    );
    let chunk_size_ref2 = new_body.alloc_expr(
        Expr::Local(chunk_size_local),
        i32_ty.clone(),
        dummy_span,
        dummy_node_id,
    );
    let chunk_start_plus_size = new_body.alloc_expr(
        Expr::BinOp {
            op: "+".to_string(),
            lhs: chunk_start_ref,
            rhs: chunk_size_ref2,
        },
        i32_ty.clone(),
        dummy_span,
        dummy_node_id,
    );
    let array_len_for_min = new_body.alloc_expr(
        Expr::Local(array_len_local),
        i32_ty.clone(),
        dummy_span,
        dummy_node_id,
    );
    let chunk_end_expr = new_body.alloc_expr(
        Expr::Call {
            func: "i32.min".to_string(),
            args: vec![chunk_start_plus_size, array_len_for_min],
        },
        i32_ty.clone(),
        dummy_span,
        dummy_node_id,
    );

    // 7. Create local chunk as a View into the base array
    let chunk_start_ref2 = new_body.alloc_expr(
        Expr::Local(chunk_start_local),
        i32_ty.clone(),
        dummy_span,
        dummy_node_id,
    );
    let chunk_end_ref = new_body.alloc_expr(
        Expr::Local(chunk_end_local),
        i32_ty.clone(),
        dummy_span,
        dummy_node_id,
    );
    let chunk_len = new_body.alloc_expr(
        Expr::BinOp {
            op: "-".to_string(),
            lhs: chunk_end_ref,
            rhs: chunk_start_ref2,
        },
        i32_ty.clone(),
        dummy_span,
        dummy_node_id,
    );
    // Extract the ptr from the base array and add the chunk offset
    // Extract the ptr from the base array using ViewPtr
    let base_ptr = new_body.alloc_expr(
        Expr::ViewPtr { view: base_array_expr_id },
        i32_ty.clone(), // ptr type at MIR level
        dummy_span,
        dummy_node_id,
    );
    let chunk_start_ref3 = new_body.alloc_expr(
        Expr::Local(chunk_start_local),
        i32_ty.clone(),
        dummy_span,
        dummy_node_id,
    );
    // Compute chunk_ptr = base_ptr + chunk_start using PtrAdd
    let chunk_ptr = new_body.alloc_expr(
        Expr::PtrAdd {
            ptr: base_ptr,
            offset: chunk_start_ref3,
        },
        i32_ty.clone(),
        dummy_span,
        dummy_node_id,
    );
    // Create the local chunk view using Expr::View
    let local_chunk_expr = new_body.alloc_expr(
        Expr::View {
            ptr: chunk_ptr,
            len: chunk_len,
        },
        array_ty.clone(),
        dummy_span,
        dummy_node_id,
    );

    // 8. Create the final expression based on whether this is direct or interprocedural
    let unit_ty = Type::Constructed(TypeName::Unit, vec![]);

    let final_expr = if let Some(ref call_info) = map_info.entry_call {
        // Interprocedural: call the helper function with the chunk View
        // We need to copy the original call but replace the array argument with the chunk
        create_call_with_chunk(
            &mut new_body,
            body,
            call_info.call_expr,
            call_info.array_arg_index,
            local_chunk_expr,
            &local_map,
        )
    } else {
        // Direct map: create map_into(closure, chunk, output, offset)
        let closure_expr = new_body.alloc_expr(
            Expr::Global(map_info.closure_name.clone()),
            Type::Variable(0), // Type doesn't matter much here
            dummy_span,
            dummy_node_id,
        );

        // Output is a Local holding a View (set up by lowering preamble)
        let output_storage_expr = new_body.alloc_expr(
            Expr::Local(output_local),
            array_ty.clone(),
            dummy_span,
            dummy_node_id,
        );

        let chunk_start_for_map = new_body.alloc_expr(
            Expr::Local(chunk_start_local),
            i32_ty.clone(),
            dummy_span,
            dummy_node_id,
        );

        new_body.alloc_expr(
            Expr::Intrinsic {
                name: "_w_intrinsic_map_into".to_string(),
                args: vec![
                    closure_expr,
                    local_chunk_expr,
                    output_storage_expr,
                    chunk_start_for_map,
                ],
            },
            unit_ty.clone(),
            dummy_span,
            dummy_node_id,
        )
    };

    // 9. Wrap in let bindings (innermost to outermost)
    let let_chunk_end = new_body.alloc_expr(
        Expr::Let {
            local: chunk_end_local,
            rhs: chunk_end_expr,
            body: final_expr,
        },
        unit_ty.clone(),
        dummy_span,
        dummy_node_id,
    );

    let let_chunk_start = new_body.alloc_expr(
        Expr::Let {
            local: chunk_start_local,
            rhs: chunk_start_expr,
            body: let_chunk_end,
        },
        unit_ty.clone(),
        dummy_span,
        dummy_node_id,
    );

    let let_chunk_size = new_body.alloc_expr(
        Expr::Let {
            local: chunk_size_local,
            rhs: chunk_size_expr,
            body: let_chunk_start,
        },
        unit_ty.clone(),
        dummy_span,
        dummy_node_id,
    );

    let let_array_len = new_body.alloc_expr(
        Expr::Let {
            local: array_len_local,
            rhs: array_len_expr,
            body: let_chunk_size,
        },
        unit_ty.clone(),
        dummy_span,
        dummy_node_id,
    );

    let root_expr = new_body.alloc_expr(
        Expr::Let {
            local: thread_id_local,
            rhs: thread_id_intrinsic,
            body: let_array_len,
        },
        unit_ty.clone(),
        dummy_span,
        dummy_node_id,
    );

    new_body.set_root(root_expr);

    // Keep original outputs - the map still returns an array
    Some((new_body, outputs.to_vec()))
}

/// Create a call expression with one argument replaced by the chunk View.
fn create_call_with_chunk(
    new_body: &mut Body,
    old_body: &Body,
    call_expr_id: ExprId,
    array_arg_index: usize,
    chunk_expr: ExprId,
    local_map: &HashMap<LocalId, LocalId>,
) -> ExprId {
    let call_expr = old_body.get_expr(call_expr_id);
    let ty = old_body.get_type(call_expr_id).clone();
    let span = old_body.get_span(call_expr_id);
    let node_id = old_body.node_ids[call_expr_id.index()];

    match call_expr {
        Expr::Call { func, args } => {
            // Copy all arguments, but replace the array argument with the chunk
            let new_args: Vec<ExprId> = args
                .iter()
                .enumerate()
                .map(|(i, &arg)| {
                    if i == array_arg_index {
                        chunk_expr
                    } else {
                        copy_expr_tree(new_body, old_body, arg, local_map)
                    }
                })
                .collect();

            new_body.alloc_expr(
                Expr::Call {
                    func: func.clone(),
                    args: new_args,
                },
                ty,
                span,
                node_id,
            )
        }
        _ => {
            // Shouldn't happen - entry_call should always point to a Call expr
            copy_expr_tree(new_body, old_body, call_expr_id, local_map)
        }
    }
}

/// Copy an expression tree from one body to another, remapping locals.
fn copy_expr_tree(
    dest: &mut Body,
    src: &Body,
    expr_id: ExprId,
    local_map: &HashMap<LocalId, LocalId>,
) -> ExprId {
    let ty = src.get_type(expr_id).clone();
    let span = src.get_span(expr_id);
    let node_id = src.node_ids[expr_id.index()];

    let new_expr = match src.get_expr(expr_id) {
        Expr::Local(local_id) => Expr::Local(*local_map.get(local_id).unwrap_or(local_id)),
        Expr::Global(name) => Expr::Global(name.clone()),
        Expr::Extern(linkage) => Expr::Extern(linkage.clone()),
        Expr::Int(s) => Expr::Int(s.clone()),
        Expr::Float(s) => Expr::Float(s.clone()),
        Expr::Bool(b) => Expr::Bool(*b),
        Expr::Unit => Expr::Unit,
        Expr::String(s) => Expr::String(s.clone()),
        Expr::Array { backing, size } => {
            let new_size = copy_expr_tree(dest, src, *size, local_map);
            let new_backing = copy_backing(dest, src, backing, local_map);
            Expr::Array {
                backing: new_backing,
                size: new_size,
            }
        }
        Expr::Tuple(elems) => {
            let new_elems: Vec<ExprId> =
                elems.iter().map(|e| copy_expr_tree(dest, src, *e, local_map)).collect();
            Expr::Tuple(new_elems)
        }
        Expr::Vector(elems) => {
            let new_elems: Vec<ExprId> =
                elems.iter().map(|e| copy_expr_tree(dest, src, *e, local_map)).collect();
            Expr::Vector(new_elems)
        }
        Expr::Matrix(rows) => {
            let new_rows: Vec<Vec<ExprId>> = rows
                .iter()
                .map(|row| row.iter().map(|e| copy_expr_tree(dest, src, *e, local_map)).collect())
                .collect();
            Expr::Matrix(new_rows)
        }
        Expr::BinOp { op, lhs, rhs } => {
            let new_lhs = copy_expr_tree(dest, src, *lhs, local_map);
            let new_rhs = copy_expr_tree(dest, src, *rhs, local_map);
            Expr::BinOp {
                op: op.clone(),
                lhs: new_lhs,
                rhs: new_rhs,
            }
        }
        Expr::UnaryOp { op, operand } => {
            let new_operand = copy_expr_tree(dest, src, *operand, local_map);
            Expr::UnaryOp {
                op: op.clone(),
                operand: new_operand,
            }
        }
        Expr::Let { local, rhs, body } => {
            let new_rhs = copy_expr_tree(dest, src, *rhs, local_map);
            let new_body = copy_expr_tree(dest, src, *body, local_map);
            Expr::Let {
                local: *local_map.get(local).unwrap_or(local),
                rhs: new_rhs,
                body: new_body,
            }
        }
        Expr::If { cond, then_, else_ } => {
            let new_cond = copy_expr_tree(dest, src, *cond, local_map);
            let new_then = copy_expr_tree(dest, src, *then_, local_map);
            let new_else = copy_expr_tree(dest, src, *else_, local_map);
            Expr::If {
                cond: new_cond,
                then_: new_then,
                else_: new_else,
            }
        }
        Expr::Call { func, args } => {
            let new_args: Vec<ExprId> =
                args.iter().map(|a| copy_expr_tree(dest, src, *a, local_map)).collect();
            Expr::Call {
                func: func.clone(),
                args: new_args,
            }
        }
        Expr::Intrinsic { name, args } => {
            let new_args: Vec<ExprId> =
                args.iter().map(|a| copy_expr_tree(dest, src, *a, local_map)).collect();
            Expr::Intrinsic {
                name: name.clone(),
                args: new_args,
            }
        }
        Expr::Materialize(inner) => {
            let new_inner = copy_expr_tree(dest, src, *inner, local_map);
            Expr::Materialize(new_inner)
        }
        Expr::Attributed { attributes, expr } => {
            let new_expr = copy_expr_tree(dest, src, *expr, local_map);
            Expr::Attributed {
                attributes: attributes.clone(),
                expr: new_expr,
            }
        }
        Expr::Load { ptr } => {
            let new_ptr = copy_expr_tree(dest, src, *ptr, local_map);
            Expr::Load { ptr: new_ptr }
        }
        Expr::Store { ptr, value } => {
            let new_ptr = copy_expr_tree(dest, src, *ptr, local_map);
            let new_value = copy_expr_tree(dest, src, *value, local_map);
            Expr::Store {
                ptr: new_ptr,
                value: new_value,
            }
        }
        Expr::Loop { .. } => {
            // Loops are complex - just copy unchanged for now
            src.get_expr(expr_id).clone()
        }
        Expr::View { ptr, len } => {
            let new_ptr = copy_expr_tree(dest, src, *ptr, local_map);
            let new_len = copy_expr_tree(dest, src, *len, local_map);
            Expr::View { ptr: new_ptr, len: new_len }
        }
        Expr::ViewPtr { view } => {
            let new_view = copy_expr_tree(dest, src, *view, local_map);
            Expr::ViewPtr { view: new_view }
        }
        Expr::ViewLen { view } => {
            let new_view = copy_expr_tree(dest, src, *view, local_map);
            Expr::ViewLen { view: new_view }
        }
        Expr::PtrAdd { ptr, offset } => {
            let new_ptr = copy_expr_tree(dest, src, *ptr, local_map);
            let new_offset = copy_expr_tree(dest, src, *offset, local_map);
            Expr::PtrAdd { ptr: new_ptr, offset: new_offset }
        }
    };

    dest.alloc_expr(new_expr, ty, span, node_id)
}

/// Copy an array backing, remapping locals.
fn copy_backing(
    dest: &mut Body,
    src: &Body,
    backing: &ArrayBacking,
    local_map: &HashMap<LocalId, LocalId>,
) -> ArrayBacking {
    match backing {
        ArrayBacking::Literal(elems) => {
            let new_elems: Vec<ExprId> =
                elems.iter().map(|e| copy_expr_tree(dest, src, *e, local_map)).collect();
            ArrayBacking::Literal(new_elems)
        }
        ArrayBacking::Range { start, step, kind } => {
            let new_start = copy_expr_tree(dest, src, *start, local_map);
            let new_step = step.map(|s| copy_expr_tree(dest, src, s, local_map));
            ArrayBacking::Range {
                start: new_start,
                step: new_step,
                kind: *kind,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallelize_soacs_passthrough() {
        let program = Program {
            defs: vec![],
            lambda_registry: crate::IdArena::new(),
        };

        let result = parallelize_soacs(program);
        assert!(result.defs.is_empty());
    }
}
