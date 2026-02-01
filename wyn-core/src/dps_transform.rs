//! Destination-Passing Style (DPS) transformation.
//!
//! Transforms functions that return runtime-sized arrays into DPS style where:
//! 1. The function takes an extra output buffer parameter
//! 2. Instead of returning the array, it writes to the destination buffer
//! 3. Return type becomes Unit
//!
//! This is necessary because SPIR-V can't return runtime-sized arrays.

use crate::ast::{NodeId, Span, TypeName};
use crate::mir::{ArrayBacking, Body, Def, Expr, ExprId, LocalDecl, LocalId, LocalKind, Program};
use polytype::Type;
use std::collections::{HashMap, HashSet};

/// Check if a type is a runtime-sized array (requires DPS transformation).
///
/// Runtime-sized arrays have the form `Array[elem, variant, Variable(_)]` where the
/// size argument is a type variable (runtime-determined). These cannot be returned
/// directly in SPIR-V and must use destination-passing style instead.
fn is_runtime_sized_array(ty: &Type<TypeName>) -> bool {
    match ty {
        Type::Constructed(TypeName::Array, args) if args.len() == 3 => {
            // Size is a type variable (runtime-determined)
            matches!(&args[2], Type::Variable(_))
        }
        _ => false,
    }
}

/// Apply DPS transformation to a MIR program.
pub fn apply_dps_transform(program: Program) -> Program {
    // Phase 1: Identify all functions that need DPS transformation
    // Skip prelude functions like 'iota' which return virtual arrays
    let dps_functions: HashSet<String> = program
        .defs
        .iter()
        .filter_map(|def| match def {
            Def::Function { name, ret_type, .. } => {
                // Skip prelude functions - they have special handling
                if name == "iota" || name.starts_with("iota$") {
                    return None;
                }
                if is_runtime_sized_array(ret_type) { Some(name.clone()) } else { None }
            }
            _ => None,
        })
        .collect();

    // If no functions need DPS, return unchanged
    if dps_functions.is_empty() {
        return program;
    }

    // Phase 2: Transform all definitions
    let defs = program.defs.into_iter().map(|def| transform_def(def, &dps_functions)).collect();

    Program {
        defs,
        lambda_registry: program.lambda_registry,
    }
}

/// Transform a single definition.
fn transform_def(def: Def, dps_functions: &HashSet<String>) -> Def {
    match def {
        Def::Function {
            id,
            name,
            params,
            ret_type,
            attributes,
            body,
            span,
            dps_output: _,
        } => {
            if dps_functions.contains(&name) {
                // Transform this function to DPS
                transform_function_to_dps(id, name, params, ret_type, attributes, body, span)
            } else {
                // Just transform call sites within this function
                let new_body = transform_body_call_sites(&body, dps_functions);
                Def::Function {
                    id,
                    name,
                    params,
                    ret_type,
                    attributes,
                    body: new_body,
                    span,
                    dps_output: None,
                }
            }
        }
        Def::EntryPoint {
            id,
            name,
            execution_model,
            inputs,
            outputs,
            body,
            span,
        } => {
            // Entry points may call DPS functions - transform their call sites
            let new_body = transform_entry_call_sites(&body, dps_functions, &outputs);
            Def::EntryPoint {
                id,
                name,
                execution_model,
                inputs,
                outputs,
                body: new_body,
                span,
            }
        }
        // Other definitions pass through unchanged
        other => other,
    }
}

/// Transform a function to DPS style.
fn transform_function_to_dps(
    id: NodeId,
    name: String,
    mut params: Vec<LocalId>,
    ret_type: Type<TypeName>,
    attributes: Vec<crate::mir::Attribute>,
    body: Body,
    span: Span,
) -> Def {
    let mut new_body = Body::new();
    let dummy_span = Span {
        start_line: 1,
        start_col: 1,
        end_line: 1,
        end_col: 1,
    };

    // Copy existing locals
    let mut local_map: HashMap<LocalId, LocalId> = HashMap::new();
    for (old_idx, local) in body.locals.iter().enumerate() {
        let old_id = LocalId(old_idx as u32);
        let new_id = new_body.alloc_local(local.clone());
        local_map.insert(old_id, new_id);
    }

    // Add output parameter
    let out_local = new_body.alloc_local(LocalDecl {
        name: "_out".to_string(),
        span: dummy_span,
        ty: ret_type.clone(),
        kind: LocalKind::Param,
    });

    // Update params list - the new output param is added at the end
    let new_params: Vec<LocalId> = params.iter().map(|p| *local_map.get(p).unwrap()).collect();
    params = new_params;
    params.push(out_local);

    // Transform the body to use DPS
    // For now, we handle the common case: body is a `map` expression
    let root = transform_body_to_dps(&body, &mut new_body, &local_map, out_local);
    new_body.set_root(root);

    Def::Function {
        id,
        name,
        params,
        ret_type: Type::Constructed(TypeName::Unit, vec![]), // Now returns Unit
        attributes,
        body: new_body,
        span,
        dps_output: Some(out_local),
    }
}

/// Transform a function body to DPS style.
/// The body currently evaluates to an array value; transform to write to output.
fn transform_body_to_dps(
    src: &Body,
    dest: &mut Body,
    local_map: &HashMap<LocalId, LocalId>,
    out_local: LocalId,
) -> ExprId {
    let root_expr = src.get_expr(src.root);
    let root_ty = src.get_type(src.root);
    let root_span = src.get_span(src.root);
    let dummy_node_id = NodeId(0);
    let unit_ty = Type::Constructed(TypeName::Unit, vec![]);
    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);

    // Check if the body is a map intrinsic - transform to map_into
    match root_expr {
        Expr::Intrinsic { name, args } if name == "_w_intrinsic_map" || name == "map" => {
            // Transform map(f, arr) to map_into(f, arr, _out, 0)
            // Copy the existing args
            let mut new_args: Vec<ExprId> =
                args.iter().map(|&e| copy_expr_tree(dest, src, e, local_map)).collect();

            // Add output buffer reference
            let out_ref =
                dest.alloc_expr(Expr::Local(out_local), root_ty.clone(), root_span, dummy_node_id);
            new_args.push(out_ref);

            // Add offset (0)
            let zero = dest.alloc_expr(Expr::Int("0".to_string()), i32_ty, root_span, dummy_node_id);
            new_args.push(zero);

            dest.alloc_expr(
                Expr::Intrinsic {
                    name: "_w_intrinsic_map_into".to_string(),
                    args: new_args,
                },
                unit_ty,
                root_span,
                dummy_node_id,
            )
        }
        Expr::Call { func, args } => {
            // If calling another function that returns array, it might also be DPS
            // For now, copy as-is and rely on the call site transformation
            let new_args: Vec<ExprId> =
                args.iter().map(|&e| copy_expr_tree(dest, src, e, local_map)).collect();

            dest.alloc_expr(
                Expr::Call {
                    func: func.clone(),
                    args: new_args,
                },
                root_ty.clone(),
                root_span,
                dummy_node_id,
            )
        }
        Expr::Let {
            local,
            rhs,
            body: let_body,
        } => {
            // Transform let bindings - preserve the structure
            let new_rhs = copy_expr_tree(dest, src, *rhs, local_map);
            let inner_body = transform_body_to_dps_inner(src, dest, *let_body, local_map, out_local);
            dest.alloc_expr(
                Expr::Let {
                    local: *local_map.get(local).unwrap_or(local),
                    rhs: new_rhs,
                    body: inner_body,
                },
                unit_ty,
                root_span,
                dummy_node_id,
            )
        }
        _ => {
            // For other expressions, we need to copy to the output buffer
            // This is more complex - for now, just copy and the SPIR-V lowering will handle it
            copy_expr_tree(dest, src, src.root, local_map)
        }
    }
}

/// Helper for transforming nested expressions within let bindings.
fn transform_body_to_dps_inner(
    src: &Body,
    dest: &mut Body,
    expr_id: ExprId,
    local_map: &HashMap<LocalId, LocalId>,
    out_local: LocalId,
) -> ExprId {
    let expr = src.get_expr(expr_id);
    let ty = src.get_type(expr_id);
    let span = src.get_span(expr_id);
    let dummy_node_id = NodeId(0);
    let unit_ty = Type::Constructed(TypeName::Unit, vec![]);
    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);

    match expr {
        Expr::Intrinsic { name, args } if name == "_w_intrinsic_map" || name == "map" => {
            // Transform map to map_into
            let mut new_args: Vec<ExprId> =
                args.iter().map(|&e| copy_expr_tree(dest, src, e, local_map)).collect();

            let out_ref = dest.alloc_expr(Expr::Local(out_local), ty.clone(), span, dummy_node_id);
            new_args.push(out_ref);

            let zero = dest.alloc_expr(Expr::Int("0".to_string()), i32_ty, span, dummy_node_id);
            new_args.push(zero);

            dest.alloc_expr(
                Expr::Intrinsic {
                    name: "_w_intrinsic_map_into".to_string(),
                    args: new_args,
                },
                unit_ty,
                span,
                dummy_node_id,
            )
        }
        Expr::Let {
            local,
            rhs,
            body: let_body,
        } => {
            let new_rhs = copy_expr_tree(dest, src, *rhs, local_map);
            let inner = transform_body_to_dps_inner(src, dest, *let_body, local_map, out_local);
            dest.alloc_expr(
                Expr::Let {
                    local: *local_map.get(local).unwrap_or(local),
                    rhs: new_rhs,
                    body: inner,
                },
                unit_ty,
                span,
                dummy_node_id,
            )
        }
        _ => copy_expr_tree(dest, src, expr_id, local_map),
    }
}

/// Check that a non-DPS function doesn't call any DPS functions.
/// Non-DPS functions cannot call DPS functions because there's no output buffer to pass.
/// Returns the body unchanged if valid, panics if a DPS call is found.
fn transform_body_call_sites(body: &Body, dps_functions: &HashSet<String>) -> Body {
    // Check all expressions for calls to DPS functions
    for expr in body.iter_exprs() {
        if let Expr::Call { func, .. } = expr {
            if dps_functions.contains(func) {
                panic!(
                    "Non-DPS function calls DPS function '{}'. \
                     Functions that call DPS functions must also be DPS \
                     (return runtime-sized arrays).",
                    func
                );
            }
        }
    }
    // No DPS calls found - return body unchanged
    body.clone()
}

/// Transform call sites within an entry point to pass output buffers to DPS callees.
/// Entry points have output storage buffers that can be passed to DPS functions.
fn transform_entry_call_sites(
    body: &Body,
    dps_functions: &HashSet<String>,
    outputs: &[crate::mir::EntryOutput],
) -> Body {
    // Check if there are any DPS calls in this entry point
    let has_dps_calls = body
        .iter_exprs()
        .any(|expr| matches!(expr, Expr::Call { func, .. } if dps_functions.contains(func)));

    if !has_dps_calls {
        return body.clone();
    }

    // For DPS calls, we need an output buffer. Currently support single array output.
    if outputs.len() != 1 {
        panic!(
            "Entry point with DPS calls must have exactly one output, found {}",
            outputs.len()
        );
    }

    let output_ty = &outputs[0].ty;
    if !is_runtime_sized_array(output_ty) {
        panic!("Entry point output must be a runtime-sized array for DPS");
    }

    // Build a new body with transformed call sites
    let mut new_body = Body::new();
    let dummy_span = Span {
        start_line: 1,
        start_col: 1,
        end_line: 1,
        end_col: 1,
    };
    let dummy_node_id = NodeId(0);

    // Copy locals
    let mut local_map: HashMap<LocalId, LocalId> = HashMap::new();
    for (old_idx, local) in body.locals.iter().enumerate() {
        let old_id = LocalId(old_idx as u32);
        let new_id = new_body.alloc_local(local.clone());
        local_map.insert(old_id, new_id);
    }

    // TODO: Create a View for the output buffer instead of Storage
    // For now, use a placeholder Local that will be set up by parallelization
    let output_local = new_body.alloc_local(LocalDecl {
        name: "_output_view".to_string(),
        span: dummy_span,
        ty: output_ty.clone(),
        kind: LocalKind::Param,
    });
    let output_storage_expr = new_body.alloc_expr(
        Expr::Local(output_local),
        output_ty.clone(),
        dummy_span,
        dummy_node_id,
    );

    // Transform expressions, adding output storage to DPS calls
    let new_root = transform_expr_for_entry_dps(
        body,
        &mut new_body,
        body.root,
        &local_map,
        dps_functions,
        output_storage_expr,
        output_ty,
    );
    new_body.set_root(new_root);

    new_body
}

/// Transform an expression tree, adding output buffer arguments to DPS calls.
fn transform_expr_for_entry_dps(
    src: &Body,
    dest: &mut Body,
    expr_id: ExprId,
    local_map: &HashMap<LocalId, LocalId>,
    dps_functions: &HashSet<String>,
    output_storage_expr: ExprId,
    output_ty: &Type<TypeName>,
) -> ExprId {
    let _ty = src.get_type(expr_id);
    let span = src.get_span(expr_id);
    let node_id = src.node_ids[expr_id.index()];

    match src.get_expr(expr_id) {
        Expr::Call { func, args } if dps_functions.contains(func) => {
            // Transform DPS call: add output storage as last argument
            let mut new_args: Vec<ExprId> = args
                .iter()
                .map(|&e| {
                    transform_expr_for_entry_dps(
                        src,
                        dest,
                        e,
                        local_map,
                        dps_functions,
                        output_storage_expr,
                        output_ty,
                    )
                })
                .collect();

            // Add output storage expression
            new_args.push(output_storage_expr);

            // DPS calls return Unit
            dest.alloc_expr(
                Expr::Call {
                    func: func.clone(),
                    args: new_args,
                },
                Type::Constructed(TypeName::Unit, vec![]),
                span,
                node_id,
            )
        }
        // For all other expressions, recursively transform children
        _ => copy_expr_tree_with_dps(
            src,
            dest,
            expr_id,
            local_map,
            dps_functions,
            output_storage_expr,
            output_ty,
        ),
    }
}

/// Copy an expression tree, transforming DPS calls along the way.
fn copy_expr_tree_with_dps(
    src: &Body,
    dest: &mut Body,
    expr_id: ExprId,
    local_map: &HashMap<LocalId, LocalId>,
    dps_functions: &HashSet<String>,
    output_storage_expr: ExprId,
    output_ty: &Type<TypeName>,
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
            let new_size = transform_expr_for_entry_dps(
                src,
                dest,
                *size,
                local_map,
                dps_functions,
                output_storage_expr,
                output_ty,
            );
            let new_backing = copy_backing_with_dps(
                src,
                dest,
                backing,
                local_map,
                dps_functions,
                output_storage_expr,
                output_ty,
            );
            Expr::Array {
                backing: new_backing,
                size: new_size,
            }
        }
        Expr::Tuple(elems) => {
            let new_elems: Vec<ExprId> = elems
                .iter()
                .map(|e| {
                    transform_expr_for_entry_dps(
                        src,
                        dest,
                        *e,
                        local_map,
                        dps_functions,
                        output_storage_expr,
                        output_ty,
                    )
                })
                .collect();
            Expr::Tuple(new_elems)
        }
        Expr::Vector(elems) => {
            let new_elems: Vec<ExprId> = elems
                .iter()
                .map(|e| {
                    transform_expr_for_entry_dps(
                        src,
                        dest,
                        *e,
                        local_map,
                        dps_functions,
                        output_storage_expr,
                        output_ty,
                    )
                })
                .collect();
            Expr::Vector(new_elems)
        }
        Expr::Matrix(rows) => {
            let new_rows: Vec<Vec<ExprId>> = rows
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|e| {
                            transform_expr_for_entry_dps(
                                src,
                                dest,
                                *e,
                                local_map,
                                dps_functions,
                                output_storage_expr,
                                output_ty,
                            )
                        })
                        .collect()
                })
                .collect();
            Expr::Matrix(new_rows)
        }
        Expr::BinOp { op, lhs, rhs } => {
            let new_lhs = transform_expr_for_entry_dps(
                src,
                dest,
                *lhs,
                local_map,
                dps_functions,
                output_storage_expr,
                output_ty,
            );
            let new_rhs = transform_expr_for_entry_dps(
                src,
                dest,
                *rhs,
                local_map,
                dps_functions,
                output_storage_expr,
                output_ty,
            );
            Expr::BinOp {
                op: op.clone(),
                lhs: new_lhs,
                rhs: new_rhs,
            }
        }
        Expr::UnaryOp { op, operand } => {
            let new_operand = transform_expr_for_entry_dps(
                src,
                dest,
                *operand,
                local_map,
                dps_functions,
                output_storage_expr,
                output_ty,
            );
            Expr::UnaryOp {
                op: op.clone(),
                operand: new_operand,
            }
        }
        Expr::Let { local, rhs, body } => {
            let new_rhs = transform_expr_for_entry_dps(
                src,
                dest,
                *rhs,
                local_map,
                dps_functions,
                output_storage_expr,
                output_ty,
            );
            let new_body = transform_expr_for_entry_dps(
                src,
                dest,
                *body,
                local_map,
                dps_functions,
                output_storage_expr,
                output_ty,
            );
            Expr::Let {
                local: *local_map.get(local).unwrap_or(local),
                rhs: new_rhs,
                body: new_body,
            }
        }
        Expr::If { cond, then_, else_ } => {
            let new_cond = transform_expr_for_entry_dps(
                src,
                dest,
                *cond,
                local_map,
                dps_functions,
                output_storage_expr,
                output_ty,
            );
            let new_then = transform_expr_for_entry_dps(
                src,
                dest,
                *then_,
                local_map,
                dps_functions,
                output_storage_expr,
                output_ty,
            );
            let new_else = transform_expr_for_entry_dps(
                src,
                dest,
                *else_,
                local_map,
                dps_functions,
                output_storage_expr,
                output_ty,
            );
            Expr::If {
                cond: new_cond,
                then_: new_then,
                else_: new_else,
            }
        }
        Expr::Call { func, args } if dps_functions.contains(func) => {
            // DPS call - add output storage
            let mut new_args: Vec<ExprId> = args
                .iter()
                .map(|a| {
                    transform_expr_for_entry_dps(
                        src,
                        dest,
                        *a,
                        local_map,
                        dps_functions,
                        output_storage_expr,
                        output_ty,
                    )
                })
                .collect();
            new_args.push(output_storage_expr);
            Expr::Call {
                func: func.clone(),
                args: new_args,
            }
        }
        Expr::Call { func, args } => {
            let new_args: Vec<ExprId> = args
                .iter()
                .map(|a| {
                    transform_expr_for_entry_dps(
                        src,
                        dest,
                        *a,
                        local_map,
                        dps_functions,
                        output_storage_expr,
                        output_ty,
                    )
                })
                .collect();
            Expr::Call {
                func: func.clone(),
                args: new_args,
            }
        }
        Expr::Intrinsic { name, args } => {
            let new_args: Vec<ExprId> = args
                .iter()
                .map(|a| {
                    transform_expr_for_entry_dps(
                        src,
                        dest,
                        *a,
                        local_map,
                        dps_functions,
                        output_storage_expr,
                        output_ty,
                    )
                })
                .collect();
            Expr::Intrinsic {
                name: name.clone(),
                args: new_args,
            }
        }
        Expr::Materialize(inner) => {
            let new_inner = transform_expr_for_entry_dps(
                src,
                dest,
                *inner,
                local_map,
                dps_functions,
                output_storage_expr,
                output_ty,
            );
            Expr::Materialize(new_inner)
        }
        Expr::Attributed { attributes, expr } => {
            let new_expr = transform_expr_for_entry_dps(
                src,
                dest,
                *expr,
                local_map,
                dps_functions,
                output_storage_expr,
                output_ty,
            );
            Expr::Attributed {
                attributes: attributes.clone(),
                expr: new_expr,
            }
        }
        Expr::Load { ptr } => {
            let new_ptr = transform_expr_for_entry_dps(
                src,
                dest,
                *ptr,
                local_map,
                dps_functions,
                output_storage_expr,
                output_ty,
            );
            Expr::Load { ptr: new_ptr }
        }
        Expr::Store { ptr, value } => {
            let new_ptr = transform_expr_for_entry_dps(
                src,
                dest,
                *ptr,
                local_map,
                dps_functions,
                output_storage_expr,
                output_ty,
            );
            let new_value = transform_expr_for_entry_dps(
                src,
                dest,
                *value,
                local_map,
                dps_functions,
                output_storage_expr,
                output_ty,
            );
            Expr::Store {
                ptr: new_ptr,
                value: new_value,
            }
        }
        Expr::Loop {
            loop_var,
            init,
            init_bindings,
            kind,
            body: loop_body,
        } => {
            let new_init = transform_expr_for_entry_dps(
                src,
                dest,
                *init,
                local_map,
                dps_functions,
                output_storage_expr,
                output_ty,
            );
            let new_init_bindings: Vec<(LocalId, ExprId)> = init_bindings
                .iter()
                .map(|(l, e)| {
                    (
                        *local_map.get(l).unwrap_or(l),
                        transform_expr_for_entry_dps(
                            src,
                            dest,
                            *e,
                            local_map,
                            dps_functions,
                            output_storage_expr,
                            output_ty,
                        ),
                    )
                })
                .collect();
            let new_kind = copy_loop_kind_with_dps(
                src,
                dest,
                kind,
                local_map,
                dps_functions,
                output_storage_expr,
                output_ty,
            );
            let new_body = transform_expr_for_entry_dps(
                src,
                dest,
                *loop_body,
                local_map,
                dps_functions,
                output_storage_expr,
                output_ty,
            );
            Expr::Loop {
                loop_var: *local_map.get(loop_var).unwrap_or(loop_var),
                init: new_init,
                init_bindings: new_init_bindings,
                kind: new_kind,
                body: new_body,
            }
        }
    };

    dest.alloc_expr(new_expr, ty, span, node_id)
}

/// Copy array backing with DPS transformation.
fn copy_backing_with_dps(
    src: &Body,
    dest: &mut Body,
    backing: &ArrayBacking,
    local_map: &HashMap<LocalId, LocalId>,
    dps_functions: &HashSet<String>,
    output_storage_expr: ExprId,
    output_ty: &Type<TypeName>,
) -> ArrayBacking {
    match backing {
        ArrayBacking::Literal(elems) => {
            let new_elems: Vec<ExprId> = elems
                .iter()
                .map(|e| {
                    transform_expr_for_entry_dps(
                        src,
                        dest,
                        *e,
                        local_map,
                        dps_functions,
                        output_storage_expr,
                        output_ty,
                    )
                })
                .collect();
            ArrayBacking::Literal(new_elems)
        }
        ArrayBacking::Range { start, step, kind } => {
            let new_start = transform_expr_for_entry_dps(
                src,
                dest,
                *start,
                local_map,
                dps_functions,
                output_storage_expr,
                output_ty,
            );
            let new_step = step.map(|s| {
                transform_expr_for_entry_dps(
                    src,
                    dest,
                    s,
                    local_map,
                    dps_functions,
                    output_storage_expr,
                    output_ty,
                )
            });
            ArrayBacking::Range {
                start: new_start,
                step: new_step,
                kind: *kind,
            }
        }
        ArrayBacking::View { ptr, len } => {
            let new_ptr = transform_expr_for_entry_dps(
                src,
                dest,
                *ptr,
                local_map,
                dps_functions,
                output_storage_expr,
                output_ty,
            );
            let new_len = transform_expr_for_entry_dps(
                src,
                dest,
                *len,
                local_map,
                dps_functions,
                output_storage_expr,
                output_ty,
            );
            ArrayBacking::View {
                ptr: new_ptr,
                len: new_len,
            }
        }
    }
}

/// Copy loop kind with DPS transformation.
fn copy_loop_kind_with_dps(
    src: &Body,
    dest: &mut Body,
    kind: &crate::mir::LoopKind,
    local_map: &HashMap<LocalId, LocalId>,
    dps_functions: &HashSet<String>,
    output_storage_expr: ExprId,
    output_ty: &Type<TypeName>,
) -> crate::mir::LoopKind {
    use crate::mir::LoopKind;
    match kind {
        LoopKind::For { var, iter } => LoopKind::For {
            var: *local_map.get(var).unwrap_or(var),
            iter: transform_expr_for_entry_dps(
                src,
                dest,
                *iter,
                local_map,
                dps_functions,
                output_storage_expr,
                output_ty,
            ),
        },
        LoopKind::ForRange { var, bound } => LoopKind::ForRange {
            var: *local_map.get(var).unwrap_or(var),
            bound: transform_expr_for_entry_dps(
                src,
                dest,
                *bound,
                local_map,
                dps_functions,
                output_storage_expr,
                output_ty,
            ),
        },
        LoopKind::While { cond } => LoopKind::While {
            cond: transform_expr_for_entry_dps(
                src,
                dest,
                *cond,
                local_map,
                dps_functions,
                output_storage_expr,
                output_ty,
            ),
        },
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
        Expr::Loop {
            loop_var,
            init,
            init_bindings,
            kind,
            body: loop_body,
        } => {
            let new_init = copy_expr_tree(dest, src, *init, local_map);
            let new_init_bindings: Vec<(LocalId, ExprId)> = init_bindings
                .iter()
                .map(|(l, e)| {
                    (
                        *local_map.get(l).unwrap_or(l),
                        copy_expr_tree(dest, src, *e, local_map),
                    )
                })
                .collect();
            let new_kind = copy_loop_kind(dest, src, kind, local_map);
            let new_body = copy_expr_tree(dest, src, *loop_body, local_map);
            Expr::Loop {
                loop_var: *local_map.get(loop_var).unwrap_or(loop_var),
                init: new_init,
                init_bindings: new_init_bindings,
                kind: new_kind,
                body: new_body,
            }
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
        ArrayBacking::View { ptr, len } => {
            let new_ptr = copy_expr_tree(dest, src, *ptr, local_map);
            let new_len = copy_expr_tree(dest, src, *len, local_map);
            ArrayBacking::View {
                ptr: new_ptr,
                len: new_len,
            }
        }
    }
}

/// Copy a loop kind, remapping locals.
fn copy_loop_kind(
    dest: &mut Body,
    src: &Body,
    kind: &crate::mir::LoopKind,
    local_map: &HashMap<LocalId, LocalId>,
) -> crate::mir::LoopKind {
    use crate::mir::LoopKind;
    match kind {
        LoopKind::For { var, iter } => LoopKind::For {
            var: *local_map.get(var).unwrap_or(var),
            iter: copy_expr_tree(dest, src, *iter, local_map),
        },
        LoopKind::ForRange { var, bound } => LoopKind::ForRange {
            var: *local_map.get(var).unwrap_or(var),
            bound: copy_expr_tree(dest, src, *bound, local_map),
        },
        LoopKind::While { cond } => LoopKind::While {
            cond: copy_expr_tree(dest, src, *cond, local_map),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_runtime_sized_array() {
        // Fixed-size array - not DPS
        let fixed = Type::Constructed(
            TypeName::Array,
            vec![
                Type::Constructed(TypeName::Float(32), vec![]),
                Type::Constructed(TypeName::ArrayVariantComposite, vec![]),
                Type::Constructed(TypeName::Size(10), vec![]),
            ],
        );
        assert!(!is_runtime_sized_array(&fixed));

        // Runtime-sized array - needs DPS
        let runtime = Type::Constructed(
            TypeName::Array,
            vec![
                Type::Constructed(TypeName::Float(32), vec![]),
                Type::Constructed(TypeName::ArrayVariantView, vec![]),
                Type::Variable(0),
            ],
        );
        assert!(is_runtime_sized_array(&runtime));
    }
}
