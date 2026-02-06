//! Input and output strategies for parallelization.
//!
//! These traits abstract over how elements are read (input) and written (output)
//! in parallelized compute shaders.

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

        let _view = ctx.builder
            .emit_storage_view(self.set, self.binding, array_ty, ctx.span, ctx.node_id)
            .ok()?;

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
        let view = ctx.builder
            .emit_storage_view(self.set, self.binding, array_ty, ctx.span, ctx.node_id)
            .ok()?;

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
        // Copy the length constant to the new builder
        let new_len = copy_value_to_builder(ctx, &ctx.entry.body.clone(), self.len)?;

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
                // Copy start (and step if present) to new builder
                let body = ctx.entry.body.clone();
                let new_start = copy_value_to_builder(ctx, &body, *start)?;

                // Compute: start + index * step (or start + index if step is 1)
                let elem = if let Some(step) = step {
                    let new_step = copy_value_to_builder(ctx, &body, *step)?;
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
        ctx.builder
            .emit_storage_view(self.set, self.binding, array_ty, ctx.span, ctx.node_id)
            .ok()
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
            .emit_storage_store(output_view, index, value, elem_ty.clone(), effect_in, ctx.span, ctx.node_id)
            .ok()?;
        Some(())
    }
}

// =============================================================================
// Helpers
// =============================================================================

/// Copy a value from the original body to the new builder.
/// Handles constants (Int, Float) and entry point parameters.
pub fn copy_value_to_builder(
    ctx: &mut ParallelizeCtx,
    original_body: &FuncBody,
    value: ValueId,
) -> Option<ValueId> {
    // Check if the value is an entry point parameter
    for (i, (param_val, _, _)) in original_body.params.iter().enumerate() {
        if *param_val == value {
            return Some(ctx.builder.get_param(i));
        }
    }

    // Check if it's a constant instruction
    for inst in &original_body.insts {
        if inst.result == Some(value) {
            match &inst.kind {
                InstKind::Int(int_val) => {
                    return ctx
                        .builder
                        .push_int(int_val, inst.result_ty.clone(), ctx.span, ctx.node_id)
                        .ok();
                }
                InstKind::Float(float_val) => {
                    return ctx
                        .builder
                        .push_float(float_val, inst.result_ty.clone(), ctx.span, ctx.node_id)
                        .ok();
                }
                _ => return None,
            }
        }
    }
    None
}
