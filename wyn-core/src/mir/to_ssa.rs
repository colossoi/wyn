//! Conversion from old MIR Body to new SSA FuncBody.
//!
//! This module provides a bridge between the old expression-based MIR
//! and the new SSA form with explicit control flow graphs.
//!
//! The conversion handles:
//! - Mapping old LocalIds to new ValueIds
//! - Converting nested If/Loop expressions to explicit CFG blocks
//! - Preserving type and span information

use std::collections::HashMap;

use crate::ast::{NodeId, Span, TypeName};
use polytype::Type;

use super::ssa::{FuncBody, InstKind, Terminator, ValueId};
use super::ssa_builder::FuncBuilder;
use super::{ArrayBacking, Body, Expr, ExprId, LocalId, LocalKind, LoopKind};

/// Convert an old MIR Body to an SSA FuncBody.
///
/// # Arguments
/// * `body` - The old MIR body to convert
/// * `param_names` - Names for parameters (from the Def)
/// * `return_ty` - Return type of the function
///
/// # Returns
/// The converted SSA function body
pub fn convert_body(
    body: &Body,
    param_names: &[String],
    return_ty: Type<TypeName>,
) -> Result<FuncBody, ConvertError> {
    let converter = Converter::new(body, param_names, return_ty);
    converter.convert()
}

/// Error during conversion.
#[derive(Debug, Clone)]
pub enum ConvertError {
    /// Referenced an unknown local variable.
    UnknownLocal(LocalId),
    /// Referenced an unknown expression.
    UnknownExpr(ExprId),
    /// Builder error during construction.
    BuilderError(String),
}

impl std::fmt::Display for ConvertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConvertError::UnknownLocal(id) => write!(f, "Unknown local: {:?}", id),
            ConvertError::UnknownExpr(id) => write!(f, "Unknown expression: {:?}", id),
            ConvertError::BuilderError(msg) => write!(f, "Builder error: {}", msg),
        }
    }
}

impl std::error::Error for ConvertError {}

/// State for converting old MIR to SSA.
struct Converter<'a> {
    /// The old body being converted.
    old_body: &'a Body,
    /// The SSA function builder.
    builder: FuncBuilder,
    /// Mapping from old LocalId to new ValueId.
    local_map: HashMap<LocalId, ValueId>,
    /// Cache of converted expressions (old ExprId -> new ValueId).
    expr_cache: HashMap<ExprId, ValueId>,
}

impl<'a> Converter<'a> {
    fn new(body: &'a Body, param_names: &[String], return_ty: Type<TypeName>) -> Self {
        // Collect parameters from the old body
        let params: Vec<(Type<TypeName>, String)> = body
            .locals
            .iter()
            .enumerate()
            .filter(|(_, decl)| decl.kind == LocalKind::Param)
            .enumerate()
            .map(|(param_idx, (_, decl))| {
                let name = if param_idx < param_names.len() {
                    param_names[param_idx].clone()
                } else {
                    decl.name.clone()
                };
                (decl.ty.clone(), name)
            })
            .collect();

        let builder = FuncBuilder::new(params, return_ty);

        Converter {
            old_body: body,
            builder,
            local_map: HashMap::new(),
            expr_cache: HashMap::new(),
        }
    }

    fn convert(mut self) -> Result<FuncBody, ConvertError> {
        // Map parameter locals to their ValueIds
        let mut param_idx = 0;
        for (local_idx, decl) in self.old_body.locals.iter().enumerate() {
            if decl.kind == LocalKind::Param {
                let local_id = LocalId(local_idx as u32);
                let value_id = self.builder.get_param(param_idx);
                self.local_map.insert(local_id, value_id);
                param_idx += 1;
            }
        }

        // Process top-level statements
        for stmt in &self.old_body.stmts {
            let value = self.convert_expr(stmt.rhs)?;
            self.local_map.insert(stmt.local, value);
        }

        // Convert the root expression
        let result = self.convert_expr(self.old_body.root)?;

        // Terminate with return
        let return_ty = self.old_body.get_type(self.old_body.root).clone();
        if matches!(return_ty, Type::Constructed(TypeName::Unit, _)) {
            self.builder
                .terminate(Terminator::ReturnUnit)
                .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        } else {
            self.builder
                .terminate(Terminator::Return(result))
                .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        }

        self.builder.finish().map_err(|e| ConvertError::BuilderError(e.to_string()))
    }

    /// Convert an expression, returning the ValueId of the result.
    fn convert_expr(&mut self, expr_id: ExprId) -> Result<ValueId, ConvertError> {
        // Check cache first
        if let Some(&value) = self.expr_cache.get(&expr_id) {
            return Ok(value);
        }

        let expr = self.old_body.get_expr(expr_id).clone();
        let ty = self.old_body.get_type(expr_id).clone();
        let span = self.old_body.get_span(expr_id);
        let node_id = self.old_body.get_node_id(expr_id);

        let value = match expr {
            Expr::Local(local_id) => {
                *self.local_map.get(&local_id).ok_or(ConvertError::UnknownLocal(local_id))?
            }

            Expr::Global(name) => self
                .builder
                .push_global(&name, ty, span, node_id)
                .map_err(|e| ConvertError::BuilderError(e.to_string()))?,

            Expr::Extern(name) => self
                .builder
                .push_inst(InstKind::Extern(name), ty, span, node_id)
                .map_err(|e| ConvertError::BuilderError(e.to_string()))?,

            Expr::Int(s) => self
                .builder
                .push_int(&s, ty, span, node_id)
                .map_err(|e| ConvertError::BuilderError(e.to_string()))?,

            Expr::Float(s) => self
                .builder
                .push_float(&s, ty, span, node_id)
                .map_err(|e| ConvertError::BuilderError(e.to_string()))?,

            Expr::Bool(b) => self
                .builder
                .push_bool(b, span, node_id)
                .map_err(|e| ConvertError::BuilderError(e.to_string()))?,

            Expr::Unit => self
                .builder
                .push_inst(InstKind::Unit, ty, span, node_id)
                .map_err(|e| ConvertError::BuilderError(e.to_string()))?,

            Expr::String(s) => self
                .builder
                .push_inst(InstKind::String(s), ty, span, node_id)
                .map_err(|e| ConvertError::BuilderError(e.to_string()))?,

            Expr::Tuple(elements) => {
                let values =
                    elements.iter().map(|&e| self.convert_expr(e)).collect::<Result<Vec<_>, _>>()?;
                self.builder
                    .push_tuple(values, ty, span, node_id)
                    .map_err(|e| ConvertError::BuilderError(e.to_string()))?
            }

            Expr::Array { backing, size } => {
                let size_value = self.convert_expr(size)?;
                match backing {
                    ArrayBacking::Literal(elements) => {
                        let values = elements
                            .iter()
                            .map(|&e| self.convert_expr(e))
                            .collect::<Result<Vec<_>, _>>()?;
                        self.builder
                            .push_inst(InstKind::ArrayLit { elements: values }, ty, span, node_id)
                            .map_err(|e| ConvertError::BuilderError(e.to_string()))?
                    }
                    ArrayBacking::Range { start, step, kind: _ } => {
                        let start_value = self.convert_expr(start)?;
                        let step_value = match step {
                            Some(s) => Some(self.convert_expr(s)?),
                            None => None,
                        };
                        // For ranges, the "size" is the length
                        self.builder
                            .push_inst(
                                InstKind::ArrayRange {
                                    start: start_value,
                                    len: size_value,
                                    step: step_value,
                                },
                                ty,
                                span,
                                node_id,
                            )
                            .map_err(|e| ConvertError::BuilderError(e.to_string()))?
                    }
                }
            }

            Expr::Vector(elements) => {
                let values =
                    elements.iter().map(|&e| self.convert_expr(e)).collect::<Result<Vec<_>, _>>()?;
                self.builder
                    .push_inst(InstKind::Vector(values), ty, span, node_id)
                    .map_err(|e| ConvertError::BuilderError(e.to_string()))?
            }

            Expr::Matrix(rows) => {
                let row_values = rows
                    .iter()
                    .map(|row| row.iter().map(|&e| self.convert_expr(e)).collect::<Result<Vec<_>, _>>())
                    .collect::<Result<Vec<_>, _>>()?;
                self.builder
                    .push_inst(InstKind::Matrix(row_values), ty, span, node_id)
                    .map_err(|e| ConvertError::BuilderError(e.to_string()))?
            }

            Expr::BinOp { op, lhs, rhs } => {
                let lhs_value = self.convert_expr(lhs)?;
                let rhs_value = self.convert_expr(rhs)?;
                self.builder
                    .push_binop(&op, lhs_value, rhs_value, ty, span, node_id)
                    .map_err(|e| ConvertError::BuilderError(e.to_string()))?
            }

            Expr::UnaryOp { op, operand } => {
                let operand_value = self.convert_expr(operand)?;
                self.builder
                    .push_unary(&op, operand_value, ty, span, node_id)
                    .map_err(|e| ConvertError::BuilderError(e.to_string()))?
            }

            Expr::If { cond, then_, else_ } => self.convert_if(cond, &then_, &else_, ty, span, node_id)?,

            Expr::Loop {
                loop_var,
                init,
                init_bindings,
                kind,
                body,
            } => self.convert_loop(loop_var, init, &init_bindings, &kind, &body, ty, span, node_id)?,

            Expr::Call { func, args } => {
                let arg_values =
                    args.iter().map(|&e| self.convert_expr(e)).collect::<Result<Vec<_>, _>>()?;
                self.builder
                    .push_call(&func, arg_values, ty, span, node_id)
                    .map_err(|e| ConvertError::BuilderError(e.to_string()))?
            }

            Expr::Intrinsic { name, args } => {
                // Special-case intrinsics that map to first-class SSA instructions
                match name.as_str() {
                    "_w_index" if args.len() == 2 => {
                        // Convert to InstKind::Index
                        let base = self.convert_expr(args[0])?;
                        let index = self.convert_expr(args[1])?;
                        self.builder
                            .push_index(base, index, ty, span, node_id)
                            .map_err(|e| ConvertError::BuilderError(e.to_string()))?
                    }
                    "_w_tuple_proj" if args.len() == 2 => {
                        // Convert to InstKind::Project
                        // The index must be a constant integer literal
                        let base = self.convert_expr(args[0])?;
                        let index = match self.old_body.get_expr(args[1]) {
                            Expr::Int(s) => s.parse::<u32>().map_err(|_| {
                                ConvertError::BuilderError(format!(
                                    "_w_tuple_proj index '{}' is not a valid u32",
                                    s
                                ))
                            })?,
                            _ => {
                                return Err(ConvertError::BuilderError(
                                    "_w_tuple_proj requires constant integer index".to_string(),
                                ));
                            }
                        };
                        self.builder
                            .push_project(base, index, ty, span, node_id)
                            .map_err(|e| ConvertError::BuilderError(e.to_string()))?
                    }
                    "_w_intrinsic_reduce" => {
                        // Expand reduce to a loop
                        self.convert_reduce(&args, ty, span, node_id)?
                    }
                    "_w_intrinsic_map" => {
                        // Expand map to a loop
                        self.convert_map(&args, ty, span, node_id)?
                    }
                    _ => {
                        // Generic intrinsic
                        let arg_values =
                            args.iter().map(|&e| self.convert_expr(e)).collect::<Result<Vec<_>, _>>()?;
                        self.builder
                            .push_intrinsic(&name, arg_values, ty, span, node_id)
                            .map_err(|e| ConvertError::BuilderError(e.to_string()))?
                    }
                }
            }

            Expr::Materialize(inner) => {
                // Materialize becomes alloca + store + the pointer
                // For now, just convert the inner and treat it as the value
                // The actual lowering will handle materialization
                let inner_value = self.convert_expr(inner)?;
                let inner_ty = self.old_body.get_type(inner).clone();

                let effect_in = self.builder.entry_effect();
                let effect_out = self.builder.alloc_effect();

                // Alloca
                let ptr = self
                    .builder
                    .push_inst(
                        InstKind::Alloca {
                            elem_ty: inner_ty,
                            effect_in,
                            effect_out,
                        },
                        ty.clone(),
                        span,
                        node_id,
                    )
                    .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

                // Store
                let effect_in2 = effect_out;
                let effect_out2 = self.builder.alloc_effect();
                self.builder
                    .push_void_inst(
                        InstKind::Store {
                            ptr,
                            value: inner_value,
                            effect_in: effect_in2,
                            effect_out: effect_out2,
                        },
                        span,
                        node_id,
                    )
                    .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

                ptr
            }

            Expr::Attributed { attributes: _, expr } => {
                // For now, just convert the inner expression
                // Attributes can be preserved in metadata later
                self.convert_expr(expr)?
            }

            Expr::Load { ptr } => {
                let ptr_value = self.convert_expr(ptr)?;
                let effect_in = self.builder.entry_effect();
                let effect_out = self.builder.alloc_effect();
                self.builder
                    .push_inst(
                        InstKind::Load {
                            ptr: ptr_value,
                            effect_in,
                            effect_out,
                        },
                        ty,
                        span,
                        node_id,
                    )
                    .map_err(|e| ConvertError::BuilderError(e.to_string()))?
            }

            Expr::Store { ptr, value } => {
                let ptr_value = self.convert_expr(ptr)?;
                let val_value = self.convert_expr(value)?;
                let effect_in = self.builder.entry_effect();
                let effect_out = self.builder.alloc_effect();
                self.builder
                    .push_void_inst(
                        InstKind::Store {
                            ptr: ptr_value,
                            value: val_value,
                            effect_in,
                            effect_out,
                        },
                        span,
                        node_id,
                    )
                    .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
                // Store returns unit - allocate a unit value
                self.builder
                    .push_inst(InstKind::Unit, ty, span, node_id)
                    .map_err(|e| ConvertError::BuilderError(e.to_string()))?
            }

            Expr::StorageView {
                set,
                binding,
                offset,
                len,
            } => {
                let offset_value = self.convert_expr(offset)?;
                let len_value = self.convert_expr(len)?;
                self.builder
                    .push_inst(
                        InstKind::StorageView {
                            set,
                            binding,
                            offset: offset_value,
                            len: len_value,
                        },
                        ty,
                        span,
                        node_id,
                    )
                    .map_err(|e| ConvertError::BuilderError(e.to_string()))?
            }

            Expr::SliceStorageView { view, start, len } => {
                // Convert to StorageView with adjusted offset
                // This is simplified - actual implementation would compute new offset
                let view_value = self.convert_expr(view)?;
                let start_value = self.convert_expr(start)?;
                let len_value = self.convert_expr(len)?;
                // For now, emit as an intrinsic call that will be handled later
                self.builder
                    .push_intrinsic(
                        "_w_slice_storage_view",
                        vec![view_value, start_value, len_value],
                        ty,
                        span,
                        node_id,
                    )
                    .map_err(|e| ConvertError::BuilderError(e.to_string()))?
            }

            Expr::StorageViewIndex { view, index } => {
                let view_value = self.convert_expr(view)?;
                let index_value = self.convert_expr(index)?;
                self.builder
                    .push_inst(
                        InstKind::StorageViewIndex {
                            view: view_value,
                            index: index_value,
                        },
                        ty,
                        span,
                        node_id,
                    )
                    .map_err(|e| ConvertError::BuilderError(e.to_string()))?
            }

            Expr::StorageViewLen { view } => {
                let view_value = self.convert_expr(view)?;
                self.builder
                    .push_inst(InstKind::StorageViewLen { view: view_value }, ty, span, node_id)
                    .map_err(|e| ConvertError::BuilderError(e.to_string()))?
            }
        };

        // Cache the result
        self.expr_cache.insert(expr_id, value);
        Ok(value)
    }

    /// Convert an if-then-else expression to SSA blocks.
    fn convert_if(
        &mut self,
        cond: ExprId,
        then_block: &super::Block,
        else_block: &super::Block,
        result_ty: Type<TypeName>,
        _span: Span,
        _node_id: NodeId,
    ) -> Result<ValueId, ConvertError> {
        // Convert condition
        let cond_value = self.convert_expr(cond)?;

        // Create the block structure
        let if_blocks = self.builder.create_if_then_else(result_ty);

        // Branch to then/else
        self.builder
            .terminate(Terminator::CondBranch {
                cond: cond_value,
                then_target: if_blocks.then_block,
                then_args: vec![],
                else_target: if_blocks.else_block,
                else_args: vec![],
            })
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        // Convert then block
        self.builder
            .switch_to_block(if_blocks.then_block)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        let then_result = self.convert_block(then_block)?;
        self.builder
            .terminate(Terminator::Branch {
                target: if_blocks.merge_block,
                args: vec![then_result],
            })
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        // Convert else block
        self.builder
            .switch_to_block(if_blocks.else_block)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        let else_result = self.convert_block(else_block)?;
        self.builder
            .terminate(Terminator::Branch {
                target: if_blocks.merge_block,
                args: vec![else_result],
            })
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        // Switch to merge block for further instructions
        self.builder
            .switch_to_block(if_blocks.merge_block)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        Ok(if_blocks.result)
    }

    /// Convert a loop expression to SSA blocks.
    fn convert_loop(
        &mut self,
        loop_var: LocalId,
        init: ExprId,
        init_bindings: &[(LocalId, ExprId)],
        kind: &LoopKind,
        body: &super::Block,
        result_ty: Type<TypeName>,
        span: Span,
        node_id: NodeId,
    ) -> Result<ValueId, ConvertError> {
        match kind {
            LoopKind::While { cond } => self.convert_while_loop(
                loop_var,
                init,
                init_bindings,
                *cond,
                body,
                result_ty,
                span,
                node_id,
            ),
            LoopKind::ForRange { var, bound } => self.convert_for_range_loop(
                loop_var,
                init,
                init_bindings,
                *var,
                *bound,
                body,
                result_ty,
                span,
                node_id,
            ),
            LoopKind::For { var, iter } => self.convert_for_in_loop(
                loop_var,
                init,
                init_bindings,
                *var,
                *iter,
                body,
                result_ty,
                span,
                node_id,
            ),
        }
    }

    fn convert_while_loop(
        &mut self,
        loop_var: LocalId,
        init: ExprId,
        init_bindings: &[(LocalId, ExprId)],
        cond: ExprId,
        body: &super::Block,
        _result_ty: Type<TypeName>,
        _span: Span,
        _node_id: NodeId,
    ) -> Result<ValueId, ConvertError> {
        let acc_ty = self.old_body.get_local(loop_var).ty.clone();

        // Create loop blocks
        let loop_blocks = self.builder.create_while_loop(acc_ty);

        // Convert initial value and branch to header
        let init_value = self.convert_expr(init)?;
        self.builder
            .terminate(Terminator::Branch {
                target: loop_blocks.header,
                args: vec![init_value],
            })
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        // Header: map loop_var to header's acc parameter
        self.builder
            .switch_to_block(loop_blocks.header)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        self.local_map.insert(loop_var, loop_blocks.acc);

        // Process init_bindings (extract from loop_var)
        for (local, expr) in init_bindings {
            let value = self.convert_expr(*expr)?;
            self.local_map.insert(*local, value);
        }

        // Convert condition
        let cond_value = self.convert_expr(cond)?;

        // Branch to body or exit
        self.builder
            .terminate(Terminator::CondBranch {
                cond: cond_value,
                then_target: loop_blocks.body,
                then_args: vec![],
                else_target: loop_blocks.exit,
                else_args: vec![loop_blocks.acc],
            })
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        // Body: convert and branch back to header
        self.builder
            .switch_to_block(loop_blocks.body)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        let new_acc = self.convert_block(body)?;
        self.builder
            .terminate(Terminator::Branch {
                target: loop_blocks.header,
                args: vec![new_acc],
            })
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        // Exit: continue from here
        self.builder
            .switch_to_block(loop_blocks.exit)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        Ok(loop_blocks.result)
    }

    fn convert_for_range_loop(
        &mut self,
        loop_var: LocalId,
        init: ExprId,
        init_bindings: &[(LocalId, ExprId)],
        index_var: LocalId,
        bound: ExprId,
        body: &super::Block,
        _result_ty: Type<TypeName>,
        span: Span,
        node_id: NodeId,
    ) -> Result<ValueId, ConvertError> {
        let acc_ty = self.old_body.get_local(loop_var).ty.clone();
        let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
        let bool_ty = Type::Constructed(TypeName::Str("bool"), vec![]);

        // Create loop blocks
        let loop_blocks = self.builder.create_for_range_loop(acc_ty);

        // Convert initial value and bound
        let init_value = self.convert_expr(init)?;
        let bound_value = self.convert_expr(bound)?;

        // Branch to header with (init, 0)
        let zero = self
            .builder
            .push_int("0", i32_ty.clone(), span, node_id)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        self.builder
            .terminate(Terminator::Branch {
                target: loop_blocks.header,
                args: vec![init_value, zero],
            })
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        // Header: map loop_var and index_var
        self.builder
            .switch_to_block(loop_blocks.header)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        self.local_map.insert(loop_var, loop_blocks.acc);
        self.local_map.insert(index_var, loop_blocks.index);

        // Process init_bindings
        for (local, expr) in init_bindings {
            let value = self.convert_expr(*expr)?;
            self.local_map.insert(*local, value);
        }

        // Check i < bound
        let cond = self
            .builder
            .push_binop("<", loop_blocks.index, bound_value, bool_ty, span, node_id)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        self.builder
            .terminate(Terminator::CondBranch {
                cond,
                then_target: loop_blocks.body,
                then_args: vec![],
                else_target: loop_blocks.exit,
                else_args: vec![loop_blocks.acc],
            })
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        // Body: convert, increment i, branch back
        self.builder
            .switch_to_block(loop_blocks.body)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        let new_acc = self.convert_block(body)?;
        let one = self
            .builder
            .push_int("1", i32_ty, span, node_id)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        let next_i = self
            .builder
            .push_binop(
                "+",
                loop_blocks.index,
                one,
                Type::Constructed(TypeName::Int(32), vec![]),
                span,
                node_id,
            )
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        self.builder
            .terminate(Terminator::Branch {
                target: loop_blocks.header,
                args: vec![new_acc, next_i],
            })
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        // Exit
        self.builder
            .switch_to_block(loop_blocks.exit)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        Ok(loop_blocks.result)
    }

    fn convert_for_in_loop(
        &mut self,
        loop_var: LocalId,
        init: ExprId,
        init_bindings: &[(LocalId, ExprId)],
        elem_var: LocalId,
        iter: ExprId,
        body: &super::Block,
        _result_ty: Type<TypeName>,
        span: Span,
        node_id: NodeId,
    ) -> Result<ValueId, ConvertError> {
        let acc_ty = self.old_body.get_local(loop_var).ty.clone();
        let elem_ty = self.old_body.get_local(elem_var).ty.clone();
        let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
        let bool_ty = Type::Constructed(TypeName::Str("bool"), vec![]);

        // Create loop blocks (for-in is like for-range with indexing)
        let loop_blocks = self.builder.create_for_range_loop(acc_ty);

        // Convert initial value and iterator
        let init_value = self.convert_expr(init)?;
        let iter_value = self.convert_expr(iter)?;

        // Get length of iterator (we'll need an intrinsic for this)
        let _iter_ty = self.old_body.get_type(iter).clone();
        let len = self
            .builder
            .push_intrinsic("_w_intrinsic_length", vec![iter_value], i32_ty.clone(), span, node_id)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        // Branch to header with (init, 0)
        let zero = self
            .builder
            .push_int("0", i32_ty.clone(), span, node_id)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        self.builder
            .terminate(Terminator::Branch {
                target: loop_blocks.header,
                args: vec![init_value, zero],
            })
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        // Header
        self.builder
            .switch_to_block(loop_blocks.header)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        self.local_map.insert(loop_var, loop_blocks.acc);

        // Get element at index
        let elem = self
            .builder
            .push_index(iter_value, loop_blocks.index, elem_ty, span, node_id)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        self.local_map.insert(elem_var, elem);

        // Process init_bindings
        for (local, expr) in init_bindings {
            let value = self.convert_expr(*expr)?;
            self.local_map.insert(*local, value);
        }

        // Check i < len
        let cond = self
            .builder
            .push_binop("<", loop_blocks.index, len, bool_ty, span, node_id)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        self.builder
            .terminate(Terminator::CondBranch {
                cond,
                then_target: loop_blocks.body,
                then_args: vec![],
                else_target: loop_blocks.exit,
                else_args: vec![loop_blocks.acc],
            })
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        // Body
        self.builder
            .switch_to_block(loop_blocks.body)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        let new_acc = self.convert_block(body)?;
        let one = self
            .builder
            .push_int("1", i32_ty, span, node_id)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        let next_i = self
            .builder
            .push_binop(
                "+",
                loop_blocks.index,
                one,
                Type::Constructed(TypeName::Int(32), vec![]),
                span,
                node_id,
            )
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        self.builder
            .terminate(Terminator::Branch {
                target: loop_blocks.header,
                args: vec![new_acc, next_i],
            })
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        // Exit
        self.builder
            .switch_to_block(loop_blocks.exit)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        Ok(loop_blocks.result)
    }

    /// Convert a block's statements and return the result value.
    fn convert_block(&mut self, block: &super::Block) -> Result<ValueId, ConvertError> {
        // Process statements
        for stmt in &block.stmts {
            let value = self.convert_expr(stmt.rhs)?;
            self.local_map.insert(stmt.local, value);
        }

        // Convert and return the result expression
        self.convert_expr(block.result)
    }

    /// Expand _w_intrinsic_reduce to an explicit loop.
    /// reduce(op, init, arr) -> fold left: acc = init; for i < len(arr): acc = op(acc, arr[i])
    fn convert_reduce(
        &mut self,
        args: &[ExprId],
        result_ty: Type<TypeName>,
        span: Span,
        node_id: NodeId,
    ) -> Result<ValueId, ConvertError> {
        if args.len() < 3 {
            return Err(ConvertError::BuilderError(
                "_w_intrinsic_reduce requires 3 arguments (op, init, arr)".to_string(),
            ));
        }

        let op_expr = args[0];
        let init_expr = args[1];
        let arr_expr = args[2];

        // Get the function name from the op expression
        let op_name = match self.old_body.get_expr(op_expr) {
            Expr::Global(name) => name.clone(),
            _ => {
                return Err(ConvertError::BuilderError(
                    "_w_intrinsic_reduce: operator must be a function reference".to_string(),
                ));
            }
        };

        let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
        let bool_ty = Type::Constructed(TypeName::Str("bool"), vec![]);
        let acc_ty = result_ty.clone();

        // Get element type from array type
        let arr_ty = self.old_body.get_type(arr_expr);
        let elem_ty = match arr_ty {
            Type::Constructed(TypeName::Array, type_args) if !type_args.is_empty() => type_args[0].clone(),
            _ => acc_ty.clone(), // Fallback
        };

        // Convert array and init
        let arr_value = self.convert_expr(arr_expr)?;
        let init_value = self.convert_expr(init_expr)?;

        // Get array length
        let len = self
            .builder
            .push_intrinsic("_w_intrinsic_length", vec![arr_value], i32_ty.clone(), span, node_id)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        // Create loop blocks
        let loop_blocks = self.builder.create_for_range_loop(acc_ty.clone());

        // Branch to header with (init, 0)
        let zero = self
            .builder
            .push_int("0", i32_ty.clone(), span, node_id)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        self.builder
            .terminate(Terminator::Branch {
                target: loop_blocks.header,
                args: vec![init_value, zero],
            })
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        // Header: check i < len
        self.builder
            .switch_to_block(loop_blocks.header)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        let cond = self
            .builder
            .push_binop("<", loop_blocks.index, len, bool_ty, span, node_id)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        self.builder
            .terminate(Terminator::CondBranch {
                cond,
                then_target: loop_blocks.body,
                then_args: vec![],
                else_target: loop_blocks.exit,
                else_args: vec![loop_blocks.acc],
            })
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        // Body: new_acc = op(acc, arr[i])
        self.builder
            .switch_to_block(loop_blocks.body)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        let elem = self
            .builder
            .push_index(arr_value, loop_blocks.index, elem_ty, span, node_id)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        let new_acc = self
            .builder
            .push_call(&op_name, vec![loop_blocks.acc, elem], acc_ty, span, node_id)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        // Increment i and branch back
        let one = self
            .builder
            .push_int("1", i32_ty, span, node_id)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        let next_i = self
            .builder
            .push_binop(
                "+",
                loop_blocks.index,
                one,
                Type::Constructed(TypeName::Int(32), vec![]),
                span,
                node_id,
            )
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        self.builder
            .terminate(Terminator::Branch {
                target: loop_blocks.header,
                args: vec![new_acc, next_i],
            })
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        // Exit
        self.builder
            .switch_to_block(loop_blocks.exit)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        Ok(loop_blocks.result)
    }

    /// Expand _w_intrinsic_map to an explicit loop.
    /// map(f, arr) -> result where for i < len(arr): result[i] = f(arr[i])
    fn convert_map(
        &mut self,
        args: &[ExprId],
        result_ty: Type<TypeName>,
        span: Span,
        node_id: NodeId,
    ) -> Result<ValueId, ConvertError> {
        if args.len() < 2 {
            return Err(ConvertError::BuilderError(
                "_w_intrinsic_map requires 2 arguments (f, arr)".to_string(),
            ));
        }

        let f_expr = args[0];
        let arr_expr = args[1];

        // Get the function name from f expression
        let f_name = match self.old_body.get_expr(f_expr) {
            Expr::Global(name) => name.clone(),
            _ => {
                return Err(ConvertError::BuilderError(
                    "_w_intrinsic_map: function must be a function reference".to_string(),
                ));
            }
        };

        let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
        let bool_ty = Type::Constructed(TypeName::Str("bool"), vec![]);

        // Get element types from array types
        let arr_ty = self.old_body.get_type(arr_expr);
        let input_elem_ty = match arr_ty {
            Type::Constructed(TypeName::Array, type_args) if !type_args.is_empty() => type_args[0].clone(),
            _ => {
                return Err(ConvertError::BuilderError(
                    "_w_intrinsic_map: second argument must be an array".to_string(),
                ));
            }
        };
        let output_elem_ty = match &result_ty {
            Type::Constructed(TypeName::Array, type_args) if !type_args.is_empty() => type_args[0].clone(),
            _ => input_elem_ty.clone(), // Fallback
        };

        // Convert array
        let arr_value = self.convert_expr(arr_expr)?;

        // Get array length
        let len = self
            .builder
            .push_intrinsic("_w_intrinsic_length", vec![arr_value], i32_ty.clone(), span, node_id)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        // For map, we need to build up the result array element by element.
        // Use an accumulator that is the result array being built.
        // However, SPIR-V arrays are immutable - we need to use _w_array_with.
        // Start with an uninitialized array and update each element.
        let init_array = self
            .builder
            .push_call("_w_intrinsic_uninit", vec![], result_ty.clone(), span, node_id)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        // Create loop blocks (acc is the result array)
        let loop_blocks = self.builder.create_for_range_loop(result_ty.clone());

        // Branch to header with (init_array, 0)
        let zero = self
            .builder
            .push_int("0", i32_ty.clone(), span, node_id)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        self.builder
            .terminate(Terminator::Branch {
                target: loop_blocks.header,
                args: vec![init_array, zero],
            })
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        // Header: check i < len
        self.builder
            .switch_to_block(loop_blocks.header)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        let cond = self
            .builder
            .push_binop("<", loop_blocks.index, len, bool_ty, span, node_id)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        self.builder
            .terminate(Terminator::CondBranch {
                cond,
                then_target: loop_blocks.body,
                then_args: vec![],
                else_target: loop_blocks.exit,
                else_args: vec![loop_blocks.acc],
            })
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        // Body: elem = f(arr[i]); new_arr = arr_with(acc, i, elem)
        self.builder
            .switch_to_block(loop_blocks.body)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        let input_elem = self
            .builder
            .push_index(arr_value, loop_blocks.index, input_elem_ty, span, node_id)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        let output_elem = self
            .builder
            .push_call(&f_name, vec![input_elem], output_elem_ty, span, node_id)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        let new_arr = self
            .builder
            .push_call(
                "_w_intrinsic_array_with",
                vec![loop_blocks.acc, loop_blocks.index, output_elem],
                result_ty,
                span,
                node_id,
            )
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        // Increment i and branch back
        let one = self
            .builder
            .push_int("1", i32_ty, span, node_id)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        let next_i = self
            .builder
            .push_binop(
                "+",
                loop_blocks.index,
                one,
                Type::Constructed(TypeName::Int(32), vec![]),
                span,
                node_id,
            )
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;
        self.builder
            .terminate(Terminator::Branch {
                target: loop_blocks.header,
                args: vec![new_arr, next_i],
            })
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        // Exit
        self.builder
            .switch_to_block(loop_blocks.exit)
            .map_err(|e| ConvertError::BuilderError(e.to_string()))?;

        Ok(loop_blocks.result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::NodeId;
    use crate::mir::{Body, LocalDecl, LocalKind};

    fn dummy_span() -> Span {
        Span {
            start_line: 1,
            start_col: 1,
            end_line: 1,
            end_col: 1,
        }
    }

    fn dummy_node() -> NodeId {
        NodeId(0)
    }

    fn i32_ty() -> Type<TypeName> {
        Type::Constructed(TypeName::Int(32), vec![])
    }

    #[test]
    fn test_convert_simple_add() {
        // Build old MIR for: fn add(x: i32, y: i32) -> i32 = x + y
        let mut body = Body::new();

        // Parameters
        let x = body.alloc_local(LocalDecl {
            name: "x".to_string(),
            ty: i32_ty(),
            span: dummy_span(),
            kind: LocalKind::Param,
        });
        let y = body.alloc_local(LocalDecl {
            name: "y".to_string(),
            ty: i32_ty(),
            span: dummy_span(),
            kind: LocalKind::Param,
        });

        // Expressions
        let x_expr = body.alloc_expr(Expr::Local(x), i32_ty(), dummy_span(), dummy_node());
        let y_expr = body.alloc_expr(Expr::Local(y), i32_ty(), dummy_span(), dummy_node());
        let add_expr = body.alloc_expr(
            Expr::BinOp {
                op: "+".to_string(),
                lhs: x_expr,
                rhs: y_expr,
            },
            i32_ty(),
            dummy_span(),
            dummy_node(),
        );

        body.set_root(add_expr);

        // Convert
        let ssa_body = convert_body(&body, &["x".to_string(), "y".to_string()], i32_ty()).unwrap();

        // Verify
        assert_eq!(ssa_body.params.len(), 2);
        assert_eq!(ssa_body.num_blocks(), 1);

        // Verify with verifier
        let verify_result = crate::mir::ssa_verify::verify_func(&ssa_body);
        assert!(
            verify_result.is_ok(),
            "Verification failed: {:?}",
            verify_result.err()
        );
    }

    #[test]
    fn test_convert_with_let_binding() {
        // Build old MIR for: fn f(x: i32) -> i32 = let y = x + 1 in y * 2
        let mut body = Body::new();

        // Parameter
        let x = body.alloc_local(LocalDecl {
            name: "x".to_string(),
            ty: i32_ty(),
            span: dummy_span(),
            kind: LocalKind::Param,
        });

        // Let binding local
        let y = body.alloc_local(LocalDecl {
            name: "y".to_string(),
            ty: i32_ty(),
            span: dummy_span(),
            kind: LocalKind::Let,
        });

        // Expressions
        let x_expr = body.alloc_expr(Expr::Local(x), i32_ty(), dummy_span(), dummy_node());
        let one = body.alloc_expr(Expr::Int("1".to_string()), i32_ty(), dummy_span(), dummy_node());
        let add_expr = body.alloc_expr(
            Expr::BinOp {
                op: "+".to_string(),
                lhs: x_expr,
                rhs: one,
            },
            i32_ty(),
            dummy_span(),
            dummy_node(),
        );

        // Add statement: y = x + 1
        body.push_stmt(y, add_expr);

        // Result: y * 2
        let y_expr = body.alloc_expr(Expr::Local(y), i32_ty(), dummy_span(), dummy_node());
        let two = body.alloc_expr(Expr::Int("2".to_string()), i32_ty(), dummy_span(), dummy_node());
        let mul_expr = body.alloc_expr(
            Expr::BinOp {
                op: "*".to_string(),
                lhs: y_expr,
                rhs: two,
            },
            i32_ty(),
            dummy_span(),
            dummy_node(),
        );

        body.set_root(mul_expr);

        // Convert
        let ssa_body = convert_body(&body, &["x".to_string()], i32_ty()).unwrap();

        // Verify
        assert_eq!(ssa_body.params.len(), 1);

        // Verify with verifier
        let verify_result = crate::mir::ssa_verify::verify_func(&ssa_body);
        assert!(
            verify_result.is_ok(),
            "Verification failed: {:?}",
            verify_result.err()
        );
    }

    #[test]
    fn test_convert_if_then_else() {
        // Build old MIR for: fn abs(x: i32) -> i32 = if x < 0 then -x else x
        let mut body = Body::new();

        // Parameter
        let x = body.alloc_local(LocalDecl {
            name: "x".to_string(),
            ty: i32_ty(),
            span: dummy_span(),
            kind: LocalKind::Param,
        });

        // Expressions
        let x_expr1 = body.alloc_expr(Expr::Local(x), i32_ty(), dummy_span(), dummy_node());
        let zero = body.alloc_expr(Expr::Int("0".to_string()), i32_ty(), dummy_span(), dummy_node());
        let bool_ty = Type::Constructed(TypeName::Str("bool"), vec![]);
        let cond = body.alloc_expr(
            Expr::BinOp {
                op: "<".to_string(),
                lhs: x_expr1,
                rhs: zero,
            },
            bool_ty,
            dummy_span(),
            dummy_node(),
        );

        // Then branch: -x
        let x_expr2 = body.alloc_expr(Expr::Local(x), i32_ty(), dummy_span(), dummy_node());
        let neg_x = body.alloc_expr(
            Expr::UnaryOp {
                op: "-".to_string(),
                operand: x_expr2,
            },
            i32_ty(),
            dummy_span(),
            dummy_node(),
        );
        let then_block = super::super::Block::new(neg_x);

        // Else branch: x
        let x_expr3 = body.alloc_expr(Expr::Local(x), i32_ty(), dummy_span(), dummy_node());
        let else_block = super::super::Block::new(x_expr3);

        // If expression
        let if_expr = body.alloc_expr(
            Expr::If {
                cond,
                then_: then_block,
                else_: else_block,
            },
            i32_ty(),
            dummy_span(),
            dummy_node(),
        );

        body.set_root(if_expr);

        // Convert
        let ssa_body = convert_body(&body, &["x".to_string()], i32_ty()).unwrap();

        // Verify structure
        assert_eq!(ssa_body.params.len(), 1);
        assert_eq!(ssa_body.num_blocks(), 4); // entry, then, else, merge

        // Verify with verifier
        let verify_result = crate::mir::ssa_verify::verify_func(&ssa_body);
        assert!(
            verify_result.is_ok(),
            "Verification failed: {:?}",
            verify_result.err()
        );
    }
}
