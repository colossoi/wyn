//! SOAC parallelization pass for compute shaders.
//!
//! This pass transforms compute shaders that map over input arrays into
//! parallel versions where each thread processes one element.
//!
//! Transformation:
//! ```text
//! entry compute(arr: []f32) []f32 = map(|x| x * 2.0, arr)
//! ```
//! Becomes:
//! ```text
//! entry compute(arr: []f32) f32 =
//!   let idx = __thread_id in     // GlobalInvocationId.x
//!   let elem = arr[idx] in
//!   elem * 2.0                   // result stored to output[idx]
//! ```
//!
//! The lowering then:
//! - Sets up storage buffers for arr and output
//! - Maps `__thread_id` to GlobalInvocationId.x
//! - Stores the result to output[thread_id]

use crate::ast::{NodeId, Span, TypeName};
use crate::mir::{
    ArrayBacking, Body, Def, EntryInput, EntryOutput, ExecutionModel, Expr, ExprId, LocalDecl, LocalId,
    LocalKind, Program,
    soac_analysis::{self, MirSoacAnalysis, ParallelizableMap},
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

            // Can't parallelize - return unchanged
            Def::EntryPoint {
                id,
                name,
                execution_model: ExecutionModel::Compute { local_size },
                inputs,
                outputs,
                body,
                span,
            }
        }
        other => other,
    }
}

/// Try to parallelize a compute shader body using the analysis results.
///
/// Transforms:
/// ```text
/// map(closure, arr)
/// ```
/// Into:
/// ```text
/// let __thread_id = __builtin_thread_id() in
/// let __chunk_start = __thread_id * __chunk_size in
/// let __chunk_end = min(__chunk_start + __chunk_size, length(arr)) in
/// let __local_chunk = arr[__chunk_start..__chunk_end] in
/// map(closure, __local_chunk)
/// ```
fn try_parallelize_body(
    body: &Body,
    inputs: &[EntryInput],
    outputs: &[EntryOutput],
    map_info: &ParallelizableMap,
    local_size: (u32, u32, u32),
) -> Option<(Body, Vec<EntryOutput>)> {
    // Get the array type to determine element type
    let array_ty = body.get_type(map_info.array);
    let _elem_ty = soac_analysis::get_element_type(array_ty)?;

    // Calculate total threads from workgroup size
    let total_threads = local_size.0 * local_size.1 * local_size.2;

    // Find the size hint for the mapped array (if available)
    let size_hint: Option<u32> = match body.get_expr(map_info.array) {
        Expr::Local(local_id) => inputs.iter().find(|i| i.local == *local_id).and_then(|i| i.size_hint),
        _ => None,
    };

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

    // Build expressions bottom-up:

    // 1. First, compute the array length expression (we need this for both the base array and chunk calculations)
    // Find the input for the mapped array
    let array_input = match body.get_expr(map_info.array) {
        Expr::Local(local_id) => inputs.iter().find(|i| i.local == *local_id),
        _ => None,
    };

    // Create array length expression - either from size_hint or a runtime length call
    let array_len_expr = if let Some(hint) = size_hint {
        // Use compile-time size hint
        new_body.alloc_expr(
            Expr::Int(hint.to_string()),
            i32_ty.clone(),
            dummy_span,
            dummy_node_id,
        )
    } else if let Some(input) = array_input {
        // Runtime: call length intrinsic on the storage buffer
        // First create a temporary reference to the storage buffer (with size 0 as placeholder
        // since length() doesn't need the size, it queries the buffer directly)
        let offset_zero = new_body.alloc_expr(
            Expr::Int("0".to_string()),
            i32_ty.clone(),
            dummy_span,
            dummy_node_id,
        );
        let temp_size = new_body.alloc_expr(
            Expr::Int("0".to_string()),
            i32_ty.clone(),
            dummy_span,
            dummy_node_id,
        );
        let temp_arr_ref = new_body.alloc_expr(
            Expr::Array {
                backing: ArrayBacking::Storage {
                    name: input.name.clone(),
                    offset: offset_zero,
                },
                size: temp_size,
            },
            array_ty.clone(),
            dummy_span,
            dummy_node_id,
        );
        new_body.alloc_expr(
            Expr::Intrinsic {
                name: "_w_intrinsic_length".to_string(),
                args: vec![temp_arr_ref],
            },
            i32_ty.clone(),
            dummy_span,
            dummy_node_id,
        )
    } else {
        // Non-storage array - copy it and call length
        let arr_copy = copy_expr_tree(&mut new_body, body, map_info.array, &local_map);
        new_body.alloc_expr(
            Expr::Intrinsic {
                name: "_w_intrinsic_length".to_string(),
                args: vec![arr_copy],
            },
            i32_ty.clone(),
            dummy_span,
            dummy_node_id,
        )
    };

    // Add a local for array_len so we can reference it multiple times
    let array_len_local = new_body.alloc_local(LocalDecl {
        name: "__array_len".to_string(),
        span: dummy_span,
        ty: i32_ty.clone(),
        kind: LocalKind::Let,
    });

    // 2. Create the base array expression with the proper size
    let base_array_expr_id = match body.get_expr(map_info.array) {
        Expr::Local(local_id) => {
            // Check if this local is an entry input (storage buffer)
            if let Some(input) = inputs.iter().find(|i| i.local == *local_id) {
                // Create a Storage-backed array expression
                let zero = new_body.alloc_expr(
                    Expr::Int("0".to_string()),
                    i32_ty.clone(),
                    dummy_span,
                    dummy_node_id,
                );
                // Use the array_len_local as the size
                let size_ref = new_body.alloc_expr(
                    Expr::Local(array_len_local),
                    i32_ty.clone(),
                    dummy_span,
                    dummy_node_id,
                );
                new_body.alloc_expr(
                    Expr::Array {
                        backing: ArrayBacking::Storage {
                            name: input.name.clone(),
                            offset: zero,
                        },
                        size: size_ref,
                    },
                    array_ty.clone(),
                    dummy_span,
                    dummy_node_id,
                )
            } else {
                copy_expr_tree(&mut new_body, body, map_info.array, &local_map)
            }
        }
        _ => copy_expr_tree(&mut new_body, body, map_info.array, &local_map),
    };

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

    // 4. __chunk_size = (array_len + total_threads - 1) / total_threads
    //    This is ceiling division: ceil(array_len / total_threads)
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
    // Use array_len_local for the min call
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

    // 6. Create local chunk as a View into the base array
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
    // Size of chunk = end - start
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
    let chunk_start_ref3 = new_body.alloc_expr(
        Expr::Local(chunk_start_local),
        i32_ty.clone(),
        dummy_span,
        dummy_node_id,
    );
    let local_chunk_expr = new_body.alloc_expr(
        Expr::Array {
            backing: ArrayBacking::View {
                base: base_array_expr_id,
                offset: chunk_start_ref3,
            },
            size: chunk_len,
        },
        array_ty.clone(),
        dummy_span,
        dummy_node_id,
    );

    // 7. Copy the closure expression
    let closure_expr = copy_expr_tree(&mut new_body, body, map_info.closure, &local_map);

    // 8. Create the output storage buffer expression
    // The output buffer is named "_output" by convention (see pipeline.rs)
    let output_zero = new_body.alloc_expr(
        Expr::Int("0".to_string()),
        i32_ty.clone(),
        dummy_span,
        dummy_node_id,
    );
    let output_size_ref = new_body.alloc_expr(
        Expr::Local(array_len_local),
        i32_ty.clone(),
        dummy_span,
        dummy_node_id,
    );
    let output_storage_expr = new_body.alloc_expr(
        Expr::Array {
            backing: ArrayBacking::Storage {
                name: "_output".to_string(),
                offset: output_zero,
            },
            size: output_size_ref,
        },
        array_ty.clone(),
        dummy_span,
        dummy_node_id,
    );

    // 9. Create the map_into call: map_into(closure, input_view, output_storage, chunk_start)
    // This reads from input, applies the function, and writes to output
    let chunk_start_for_map = new_body.alloc_expr(
        Expr::Local(chunk_start_local),
        i32_ty.clone(),
        dummy_span,
        dummy_node_id,
    );
    let unit_ty = Type::Constructed(TypeName::Unit, vec![]);
    let map_expr = new_body.alloc_expr(
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
    );

    // 10. Wrap in let bindings (innermost to outermost)
    // The body returns () since map_into writes to output as a side effect
    let let_chunk_end_body = map_expr;

    let let_chunk_end = new_body.alloc_expr(
        Expr::Let {
            local: chunk_end_local,
            rhs: chunk_end_expr,
            body: let_chunk_end_body,
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
        ArrayBacking::IndexFn { index_fn } => {
            let new_fn = copy_expr_tree(dest, src, *index_fn, local_map);
            ArrayBacking::IndexFn { index_fn: new_fn }
        }
        ArrayBacking::View { base, offset } => {
            let new_base = copy_expr_tree(dest, src, *base, local_map);
            let new_offset = copy_expr_tree(dest, src, *offset, local_map);
            ArrayBacking::View {
                base: new_base,
                offset: new_offset,
            }
        }
        ArrayBacking::Owned { data } => {
            let new_data = copy_expr_tree(dest, src, *data, local_map);
            ArrayBacking::Owned { data: new_data }
        }
        ArrayBacking::Storage { name, offset } => {
            let new_offset = copy_expr_tree(dest, src, *offset, local_map);
            ArrayBacking::Storage {
                name: name.clone(),
                offset: new_offset,
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
