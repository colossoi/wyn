//! Input and output strategies for parallelization.
//!
//! These traits abstract over how elements are read (input) and written (output)
//! in parallelized compute shaders.

use std::collections::HashMap;

use crate::ast::TypeName;
use crate::mir::ssa::{FuncBody, InstKind, ValueId};
use crate::types::is_virtual_array;
use polytype::Type;

use super::ParallelizeCtx;

/// Strategy for reading input elements in a parallel loop.
pub trait InputStrategy {
    /// Setup resources and return (length, element_type).
    /// Called once before the loop.
    fn setup(&self, ctx: &mut ParallelizeCtx) -> Option<(ValueId, Type<TypeName>)>;

    /// Get the element at the given index.
    /// Called inside the loop body.
    fn get_element(&self, ctx: &mut ParallelizeCtx, index: ValueId) -> Option<ValueId>;
}

/// Strategy for writing output elements in a parallel loop.
pub trait OutputStrategy {
    /// Setup resources and return the output view/handle.
    /// Called once before the loop.
    fn setup(&self, ctx: &mut ParallelizeCtx, elem_ty: &Type<TypeName>) -> Option<ValueId>;

    /// Store a result at the given index.
    /// Called inside the loop body.
    fn store_result(
        &self,
        ctx: &mut ParallelizeCtx,
        output_view: ValueId,
        index: ValueId,
        value: ValueId,
        elem_ty: &Type<TypeName>,
    ) -> Option<()>;
}

// =============================================================================
// Storage Buffer Input
// =============================================================================

/// Input strategy for reading from a storage buffer.
pub struct StorageInput {
    param_index: usize,
    set: u32,
    binding: u32,
}

impl StorageInput {
    pub fn new(param_index: usize, (set, binding): (u32, u32)) -> Self {
        Self {
            param_index,
            set,
            binding,
        }
    }
}

impl InputStrategy for StorageInput {
    fn setup(&self, ctx: &mut ParallelizeCtx) -> Option<(ValueId, Type<TypeName>)> {
        let array_ty = ctx.entry.inputs.get(self.param_index)?.ty.clone();
        let elem_ty = match &array_ty {
            Type::Constructed(TypeName::Array, args) if !args.is_empty() => args[0].clone(),
            _ => return None,
        };

        let _view =
            ctx.builder.emit_storage_view(self.set, self.binding, array_ty, ctx.span, ctx.node_id).ok()?;

        // Return storage length for loop bound calculation.
        // The view is recreated in get_element (TODO: store for reuse).
        let set_val = ctx.push_int(&self.set.to_string())?;
        let binding_val = ctx.push_int(&self.binding.to_string())?;
        let storage_len =
            ctx.push_intrinsic("_w_storage_len", vec![set_val, binding_val], ctx.u32_ty.clone())?;

        Some((storage_len, elem_ty))
    }

    fn get_element(&self, ctx: &mut ParallelizeCtx, index: ValueId) -> Option<ValueId> {
        let array_ty = ctx.entry.inputs.get(self.param_index)?.ty.clone();
        let elem_ty = match &array_ty {
            Type::Constructed(TypeName::Array, args) if !args.is_empty() => args[0].clone(),
            _ => return None,
        };

        // Recreate view (TODO: store from setup for reuse)
        let view =
            ctx.builder.emit_storage_view(self.set, self.binding, array_ty, ctx.span, ctx.node_id).ok()?;

        // Index into view
        let ptr = ctx.push_inst(InstKind::StorageViewIndex { view, index }, elem_ty.clone())?;

        // Load element
        let effect_in = ctx.entry_effect();
        let effect_out = ctx.alloc_effect();
        let elem = ctx.push_inst(
            InstKind::Load {
                ptr,
                effect_in,
                effect_out,
            },
            elem_ty,
        )?;

        Some(elem)
    }
}

// =============================================================================
// Range Input
// =============================================================================

/// Input strategy for computing elements from a range (iota).
pub struct RangeInput {
    /// Kind of range: either explicit ArrayRange or simple iota
    kind: RangeKind,
    /// Length of the range
    len: ValueId,
    /// Element type (usually i32)
    elem_ty: Type<TypeName>,
}

/// The kind of range input.
enum RangeKind {
    /// Full range with explicit start/step
    Explicit {
        start: ValueId,
        step: Option<ValueId>,
    },
    /// Simple iota(n): range from 0 to n with step 1
    Iota,
}

impl RangeInput {
    /// Create a RangeInput by extracting range parameters from the original body.
    pub fn new(range_value: ValueId, original_body: &FuncBody) -> Option<Self> {
        // Find the instruction that produces the range value
        for inst in &original_body.insts {
            if inst.result == Some(range_value) {
                match &inst.kind {
                    // Direct ArrayRange instruction
                    InstKind::ArrayRange { start, len, step } => {
                        let elem_ty = extract_array_elem_type(&inst.result_ty)?;
                        return Some(Self {
                            kind: RangeKind::Explicit {
                                start: *start,
                                step: *step,
                            },
                            len: *len,
                            elem_ty,
                        });
                    }
                    // Call that produces a virtual array (iota) - range from 0 to len with step 1
                    InstKind::Call { args, .. } if is_virtual_array(&inst.result_ty) && args.len() == 1 => {
                        let elem_ty = extract_array_elem_type(&inst.result_ty)?;
                        return Some(Self {
                            kind: RangeKind::Iota,
                            len: args[0],
                            elem_ty,
                        });
                    }
                    _ => {}
                }
            }
        }
        None
    }
}

/// Extract element type from an array type.
fn extract_array_elem_type(ty: &Type<TypeName>) -> Option<Type<TypeName>> {
    if let Type::Constructed(TypeName::Array, args) = ty {
        if !args.is_empty() {
            return Some(args[0].clone());
        }
    }
    None
}

impl InputStrategy for RangeInput {
    fn setup(&self, ctx: &mut ParallelizeCtx) -> Option<(ValueId, Type<TypeName>)> {
        // Remap the length value (may be a computed expression) to the new builder
        let new_len = remap_entry_value(ctx, self.len)?;

        // Convert to u32 for indexing
        let len_u32 = ctx.push_call("u32.i32", vec![new_len], ctx.u32_ty.clone())?;

        Some((len_u32, self.elem_ty.clone()))
    }

    fn get_element(&self, ctx: &mut ParallelizeCtx, index: ValueId) -> Option<ValueId> {
        // Convert index (u32) to element type (i32)
        let index_i32 = ctx.push_call("i32.u32", vec![index], ctx.i32_ty.clone())?;

        match &self.kind {
            RangeKind::Iota => {
                // iota(n): element at index i is just i
                Some(index_i32)
            }
            RangeKind::Explicit { start, step } => {
                // Remap start (and step if present) to new builder
                let new_start = remap_entry_value(ctx, *start)?;

                // Compute: start + index * step (or start + index if step is 1)
                let elem = if let Some(step) = step {
                    let new_step = remap_entry_value(ctx, *step)?;
                    let idx_times_step = ctx.push_binop("*", index_i32, new_step, self.elem_ty.clone())?;
                    ctx.push_binop("+", new_start, idx_times_step, self.elem_ty.clone())?
                } else {
                    ctx.push_binop("+", new_start, index_i32, self.elem_ty.clone())?
                };

                Some(elem)
            }
        }
    }
}

// =============================================================================
// Storage Buffer Output
// =============================================================================

/// Output strategy for writing to a storage buffer.
pub struct StorageOutput {
    set: u32,
    binding: u32,
}

impl StorageOutput {
    pub fn new(set: u32, binding: u32) -> Self {
        Self { set, binding }
    }
}

impl OutputStrategy for StorageOutput {
    fn setup(&self, ctx: &mut ParallelizeCtx, elem_ty: &Type<TypeName>) -> Option<ValueId> {
        let array_ty = Type::Constructed(
            TypeName::Array,
            vec![
                elem_ty.clone(),
                Type::Constructed(TypeName::ArrayVariantView, vec![]),
                Type::Constructed(TypeName::SizePlaceholder, vec![]),
            ],
        );
        ctx.builder.emit_storage_view(self.set, self.binding, array_ty, ctx.span, ctx.node_id).ok()
    }

    fn store_result(
        &self,
        ctx: &mut ParallelizeCtx,
        output_view: ValueId,
        index: ValueId,
        value: ValueId,
        elem_ty: &Type<TypeName>,
    ) -> Option<()> {
        let effect_in = ctx.entry_effect();
        ctx.builder
            .emit_storage_store(
                output_view,
                index,
                value,
                elem_ty.clone(),
                effect_in,
                ctx.span,
                ctx.node_id,
            )
            .ok()?;
        Some(())
    }
}

// =============================================================================
// Helpers
// =============================================================================

/// Recursively copy a value and its dependency cone from a source `FuncBody`
/// into a target `FuncBuilder`. Uses a memo cache to avoid duplicating shared
/// dependencies. Only pure (non-effectful) instructions are supported; returns
/// `None` if an effectful instruction is encountered.
pub fn remap_value(
    source: &FuncBody,
    value: ValueId,
    builder: &mut crate::mir::ssa_builder::FuncBuilder,
    memo: &mut HashMap<ValueId, ValueId>,
    span: crate::ast::Span,
    node_id: crate::ast::NodeId,
) -> Option<ValueId> {
    // Check memo first
    if let Some(&mapped) = memo.get(&value) {
        return Some(mapped);
    }

    // Find the instruction that produces this value
    let inst = source.insts.iter().find(|i| i.result == Some(value))?;

    let new_val = match &inst.kind {
        // Literals — push directly
        InstKind::Int(v) => builder.push_int(v, inst.result_ty.clone(), span, node_id).ok()?,
        InstKind::Float(v) => builder.push_float(v, inst.result_ty.clone(), span, node_id).ok()?,
        InstKind::Bool(v) => builder.push_bool(*v, span, node_id).ok()?,
        InstKind::Unit => builder.push_inst(InstKind::Unit, inst.result_ty.clone(), span, node_id).ok()?,
        InstKind::String(v) => {
            builder.push_inst(InstKind::String(v.clone()), inst.result_ty.clone(), span, node_id).ok()?
        }

        // Binary / unary ops
        InstKind::BinOp { op, lhs, rhs } => {
            let new_lhs = remap_value(source, *lhs, builder, memo, span, node_id)?;
            let new_rhs = remap_value(source, *rhs, builder, memo, span, node_id)?;
            builder.push_binop(op, new_lhs, new_rhs, inst.result_ty.clone(), span, node_id).ok()?
        }
        InstKind::UnaryOp { op, operand } => {
            let new_op = remap_value(source, *operand, builder, memo, span, node_id)?;
            builder.push_unary(op, new_op, inst.result_ty.clone(), span, node_id).ok()?
        }

        // Aggregates
        InstKind::Tuple(elems) => {
            let new_elems = remap_values(source, elems, builder, memo, span, node_id)?;
            builder.push_tuple(new_elems, inst.result_ty.clone(), span, node_id).ok()?
        }
        InstKind::Vector(elems) => {
            let new_elems = remap_values(source, elems, builder, memo, span, node_id)?;
            builder.push_inst(InstKind::Vector(new_elems), inst.result_ty.clone(), span, node_id).ok()?
        }
        InstKind::Matrix(rows) => {
            let new_rows: Option<Vec<Vec<ValueId>>> = rows
                .iter()
                .map(|row| remap_values(source, row, builder, memo, span, node_id))
                .collect();
            builder
                .push_inst(InstKind::Matrix(new_rows?), inst.result_ty.clone(), span, node_id)
                .ok()?
        }
        InstKind::ArrayLit { elements } => {
            let new_elems = remap_values(source, elements, builder, memo, span, node_id)?;
            builder
                .push_inst(InstKind::ArrayLit { elements: new_elems }, inst.result_ty.clone(), span, node_id)
                .ok()?
        }
        InstKind::ArrayRange { start, len, step } => {
            let new_start = remap_value(source, *start, builder, memo, span, node_id)?;
            let new_len = remap_value(source, *len, builder, memo, span, node_id)?;
            let new_step = match step {
                Some(s) => Some(remap_value(source, *s, builder, memo, span, node_id)?),
                None => None,
            };
            builder
                .push_inst(
                    InstKind::ArrayRange { start: new_start, len: new_len, step: new_step },
                    inst.result_ty.clone(),
                    span,
                    node_id,
                )
                .ok()?
        }

        // Access
        InstKind::Project { base, index } => {
            let new_base = remap_value(source, *base, builder, memo, span, node_id)?;
            builder.push_project(new_base, *index, inst.result_ty.clone(), span, node_id).ok()?
        }
        InstKind::Index { base, index } => {
            let new_base = remap_value(source, *base, builder, memo, span, node_id)?;
            let new_index = remap_value(source, *index, builder, memo, span, node_id)?;
            builder.push_index(new_base, new_index, inst.result_ty.clone(), span, node_id).ok()?
        }

        // Calls
        InstKind::Call { func, args } => {
            let new_args = remap_values(source, args, builder, memo, span, node_id)?;
            builder.push_call(func, new_args, inst.result_ty.clone(), span, node_id).ok()?
        }
        InstKind::Intrinsic { name, args } => {
            let new_args = remap_values(source, args, builder, memo, span, node_id)?;
            builder.push_intrinsic(name, new_args, inst.result_ty.clone(), span, node_id).ok()?
        }

        // References
        InstKind::Global(name) => {
            builder.push_global(name, inst.result_ty.clone(), span, node_id).ok()?
        }
        InstKind::Extern(name) => {
            builder
                .push_inst(InstKind::Extern(name.clone()), inst.result_ty.clone(), span, node_id)
                .ok()?
        }

        // Effectful — cannot remap
        InstKind::Alloca { .. }
        | InstKind::Load { .. }
        | InstKind::Store { .. }
        | InstKind::StorageView { .. }
        | InstKind::StorageViewIndex { .. }
        | InstKind::StorageViewLen { .. }
        | InstKind::OutputPtr { .. } => return None,
    };

    memo.insert(value, new_val);
    Some(new_val)
}

/// Remap a slice of values, returning `None` if any cannot be remapped.
fn remap_values(
    source: &FuncBody,
    values: &[ValueId],
    builder: &mut crate::mir::ssa_builder::FuncBuilder,
    memo: &mut HashMap<ValueId, ValueId>,
    span: crate::ast::Span,
    node_id: crate::ast::NodeId,
) -> Option<Vec<ValueId>> {
    values
        .iter()
        .map(|&v| remap_value(source, v, builder, memo, span, node_id))
        .collect()
}

/// Convenience wrapper: remap a single value from the entry body using a fresh memo
/// pre-seeded with entry param mappings.
pub fn remap_entry_value(ctx: &mut ParallelizeCtx, value: ValueId) -> Option<ValueId> {
    let body = ctx.entry.body.clone();
    let span = ctx.span;
    let node_id = ctx.node_id;

    let mut memo: HashMap<ValueId, ValueId> = body
        .params
        .iter()
        .enumerate()
        .map(|(i, (src, _, _))| (*src, ctx.builder.get_param(i)))
        .collect();

    remap_value(&body, value, &mut ctx.builder, &mut memo, span, node_id)
}
