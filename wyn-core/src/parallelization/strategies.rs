//! Input and output strategies for parallelization.
//!
//! These traits abstract over how elements are read (input) and written (output)
//! in parallelized compute shaders. Each trait uses an associated `Handle` type
//! returned from `setup` and passed to per-element operations, so resources
//! (storage views, remapped values) are created once rather than per iteration.

use std::collections::HashMap;

use crate::ast::TypeName;
use crate::ssa::types::{FuncBody, InstKind, ValueId};
use crate::types::{TypeExt, is_virtual_array};
use polytype::Type;

use super::ParallelizeCtx;

/// Strategy for reading input elements in a parallel loop.
pub trait InputStrategy {
    type Handle: Copy;

    /// Setup resources and return (handle, length, element_type).
    /// Called once before the loop.
    fn setup(&mut self, ctx: &mut ParallelizeCtx) -> Option<(Self::Handle, ValueId, Type<TypeName>)>;

    /// Get the element at the given index using the handle from setup.
    /// Called inside the loop body.
    fn get_element(
        &self,
        ctx: &mut ParallelizeCtx,
        handle: Self::Handle,
        index: ValueId,
    ) -> Option<ValueId>;
}

/// Strategy for writing output elements in a parallel loop.
pub trait OutputStrategy {
    type Handle: Copy;

    /// Setup resources and return a handle for storing results.
    /// Called once before the loop.
    fn setup(&self, ctx: &mut ParallelizeCtx, elem_ty: &Type<TypeName>) -> Option<Self::Handle>;

    /// Store a result at the given index using the handle from setup.
    /// Called inside the loop body.
    fn store_result(
        &self,
        ctx: &mut ParallelizeCtx,
        handle: Self::Handle,
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
    /// Element type, populated by setup.
    elem_ty: Option<Type<TypeName>>,
}

impl StorageInput {
    pub fn new(param_index: usize, (set, binding): (u32, u32)) -> Self {
        Self {
            param_index,
            set,
            binding,
            elem_ty: None,
        }
    }
}

impl InputStrategy for StorageInput {
    /// The storage buffer view created during setup.
    type Handle = ValueId;

    fn setup(&mut self, ctx: &mut ParallelizeCtx) -> Option<(ValueId, ValueId, Type<TypeName>)> {
        let array_ty = ctx.entry.inputs.get(self.param_index)?.ty.clone();
        let elem_ty = array_ty.elem_type()?.clone();

        let view = ctx.builder.emit_storage_view(self.set, self.binding, array_ty).ok()?;

        let set_val = ctx.push_int(&self.set.to_string())?;
        let binding_val = ctx.push_int(&self.binding.to_string())?;
        let storage_len = ctx.push_intrinsic(
            "_w_intrinsic_storage_len",
            vec![set_val, binding_val],
            ctx.u32_ty.clone(),
        )?;

        self.elem_ty = Some(elem_ty.clone());
        Some((view, storage_len, elem_ty))
    }

    fn get_element(&self, ctx: &mut ParallelizeCtx, view: ValueId, index: ValueId) -> Option<ValueId> {
        let elem_ty = self.elem_ty.as_ref()?.clone();

        let ptr = ctx.push_inst(InstKind::StorageViewIndex { view, index }, elem_ty.clone())?;

        // Effect tokens are unordered markers (SPIR-V backend ignores them for ordering).
        // We use entry_effect() for all parallel iterations since they're independent.
        let effect_in = ctx.entry_effect();
        let elem = ctx.builder.push_load(ptr, elem_ty, effect_in).ok()?;

        Some(elem)
    }
}

// =============================================================================
// Range Input
// =============================================================================

/// Handle for range input, holding remapped start/step values.
#[derive(Clone, Copy)]
pub struct RangeHandle {
    /// Remapped start value. None for iota (starts at 0).
    start: Option<ValueId>,
    /// Remapped step value. None when step is 1.
    step: Option<ValueId>,
}

/// Input strategy for computing elements from a range (iota).
pub struct RangeInput {
    /// Kind of range: either explicit ArrayRange or simple iota
    kind: RangeKind,
    /// Length of the range (in the original body, before remapping)
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
    ///
    /// Only handles ranges produced by instructions (ArrayRange, iota calls).
    /// Values defined by block parameters (loop-carried ranges, phi-like merges)
    /// are not recognized and return `None`, conservatively skipping parallelization.
    pub fn new(range_value: ValueId, original_body: &FuncBody) -> Option<Self> {
        // Find the instruction that produces the range value.
        // Note: block-param-defined values won't be found here — that's intentional.
        for (_iid, inst) in &original_body.inner.insts {
            if inst.result == Some(range_value) {
                let result_ty = inst.result.map(|r| original_body.inner.value_type(r));
                match &inst.data {
                    // Direct ArrayRange instruction
                    InstKind::ArrayRange { start, len, step } => {
                        let elem_ty = result_ty.and_then(extract_array_elem_type)?;
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
                    InstKind::Call { args, .. }
                        if result_ty.is_some_and(|t| is_virtual_array(t)) && args.len() == 1 =>
                    {
                        let elem_ty = result_ty.and_then(extract_array_elem_type)?;
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
    ty.elem_type().filter(|_| ty.is_array()).cloned()
}

/// Return the scalar type name string (e.g. "i32", "u32", "f32") for a primitive type.
fn scalar_name(ty: &Type<TypeName>) -> Option<&'static str> {
    match ty {
        Type::Constructed(TypeName::Int(32), _) => Some("i32"),
        Type::Constructed(TypeName::Int(64), _) => Some("i64"),
        Type::Constructed(TypeName::UInt(32), _) => Some("u32"),
        Type::Constructed(TypeName::UInt(64), _) => Some("u64"),
        Type::Constructed(TypeName::Float(32), _) => Some("f32"),
        Type::Constructed(TypeName::Float(64), _) => Some("f64"),
        _ => None,
    }
}

impl InputStrategy for RangeInput {
    type Handle = RangeHandle;

    fn setup(&mut self, ctx: &mut ParallelizeCtx) -> Option<(RangeHandle, ValueId, Type<TypeName>)> {
        // Remap the length value (may be a computed expression) to the new builder
        let new_len = remap_entry_value(ctx, self.len)?;

        // Convert length to u32 for indexing (GPU invocation indices are u32)
        let elem_name = scalar_name(&self.elem_ty)?;
        let len_u32 = if elem_name == "u32" {
            new_len
        } else {
            ctx.push_call(&format!("u32.{elem_name}"), vec![new_len], ctx.u32_ty.clone())?
        };

        // Remap start/step once during setup instead of per-element
        let handle = match &self.kind {
            RangeKind::Iota => RangeHandle {
                start: None,
                step: None,
            },
            RangeKind::Explicit { start, step } => {
                let new_start = remap_entry_value(ctx, *start)?;
                let new_step = match step {
                    Some(s) => Some(remap_entry_value(ctx, *s)?),
                    None => None,
                };
                RangeHandle {
                    start: Some(new_start),
                    step: new_step,
                }
            }
        };

        Some((handle, len_u32, self.elem_ty.clone()))
    }

    fn get_element(
        &self,
        ctx: &mut ParallelizeCtx,
        handle: RangeHandle,
        index: ValueId,
    ) -> Option<ValueId> {
        // Convert index (u32) to element type for arithmetic
        let elem_name = scalar_name(&self.elem_ty)?;
        let index_elem = if elem_name == "u32" {
            index
        } else {
            ctx.push_call(&format!("{elem_name}.u32"), vec![index], self.elem_ty.clone())?
        };

        match handle.start {
            None => {
                // iota(n): element at index i is just i
                Some(index_elem)
            }
            Some(start) => {
                // Compute: start + index * step (or start + index if step is 1)
                let elem = if let Some(step) = handle.step {
                    let idx_times_step = ctx.push_binop("*", index_elem, step, self.elem_ty.clone())?;
                    ctx.push_binop("+", start, idx_times_step, self.elem_ty.clone())?
                } else {
                    ctx.push_binop("+", start, index_elem, self.elem_ty.clone())?
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
    /// The output storage buffer view.
    type Handle = ValueId;

    fn setup(&self, ctx: &mut ParallelizeCtx, elem_ty: &Type<TypeName>) -> Option<ValueId> {
        let array_ty = Type::Constructed(
            TypeName::Array,
            vec![
                elem_ty.clone(),
                Type::Constructed(TypeName::SizePlaceholder, vec![]),
                Type::Constructed(TypeName::ArrayVariantView, vec![]),
            ],
        );
        ctx.builder.emit_storage_view(self.set, self.binding, array_ty).ok()
    }

    fn store_result(
        &self,
        ctx: &mut ParallelizeCtx,
        output_view: ValueId,
        index: ValueId,
        value: ValueId,
        elem_ty: &Type<TypeName>,
    ) -> Option<()> {
        // Effect tokens are unordered markers (SPIR-V backend ignores them for ordering).
        // We use entry_effect() for all parallel iterations since they're independent.
        let effect_in = ctx.entry_effect();
        ctx.builder.emit_storage_store(output_view, index, value, elem_ty.clone(), effect_in).ok()?;
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
    builder: &mut crate::ssa::builder::FuncBuilder,
    memo: &mut HashMap<ValueId, ValueId>,
) -> Option<ValueId> {
    if let Some(&mapped) = memo.get(&value) {
        return Some(mapped);
    }

    // Find the instruction that produces this value.
    let inst = source.inner.insts.values().find(|i| i.result == Some(value))?;
    let rty = source.inner.value_type(value).clone();

    let new_val = match &inst.data {
        InstKind::Int(v) => builder.push_int(v, rty).ok()?,
        InstKind::Float(v) => builder.push_float(v, rty).ok()?,
        InstKind::Bool(v) => builder.push_bool(*v).ok()?,
        InstKind::Unit => builder.push_inst(InstKind::Unit, rty).ok()?,
        InstKind::String(v) => builder.push_inst(InstKind::String(v.clone()), rty).ok()?,

        InstKind::BinOp { op, lhs, rhs } => {
            let new_lhs = remap_value(source, *lhs, builder, memo)?;
            let new_rhs = remap_value(source, *rhs, builder, memo)?;
            builder.push_binop(op, new_lhs, new_rhs, rty).ok()?
        }
        InstKind::UnaryOp { op, operand } => {
            let new_op = remap_value(source, *operand, builder, memo)?;
            builder.push_unary(op, new_op, rty).ok()?
        }

        InstKind::Tuple(elems) => {
            let new_elems = remap_values(source, elems, builder, memo)?;
            builder.push_tuple(new_elems, rty).ok()?
        }
        InstKind::Vector(elems) => {
            let new_elems = remap_values(source, elems, builder, memo)?;
            builder.push_inst(InstKind::Vector(new_elems), rty).ok()?
        }
        InstKind::Matrix(rows) => {
            let new_rows: Option<Vec<Vec<ValueId>>> =
                rows.iter().map(|row| remap_values(source, row, builder, memo)).collect();
            builder.push_inst(InstKind::Matrix(new_rows?), rty).ok()?
        }
        InstKind::ArrayLit { elements } => {
            let new_elems = remap_values(source, elements, builder, memo)?;
            builder.push_inst(InstKind::ArrayLit { elements: new_elems }, rty).ok()?
        }
        InstKind::ArrayRange { start, len, step } => {
            let new_start = remap_value(source, *start, builder, memo)?;
            let new_len = remap_value(source, *len, builder, memo)?;
            let new_step = match step {
                Some(s) => Some(remap_value(source, *s, builder, memo)?),
                None => None,
            };
            builder
                .push_inst(
                    InstKind::ArrayRange {
                        start: new_start,
                        len: new_len,
                        step: new_step,
                    },
                    rty,
                )
                .ok()?
        }

        InstKind::Project { base, index } => {
            let new_base = remap_value(source, *base, builder, memo)?;
            builder.push_project(new_base, *index, rty).ok()?
        }
        InstKind::Index { base, index } => {
            let new_base = remap_value(source, *base, builder, memo)?;
            let new_index = remap_value(source, *index, builder, memo)?;
            builder.push_index(new_base, new_index, rty).ok()?
        }

        InstKind::Call { func, args } => {
            let new_args = remap_values(source, args, builder, memo)?;
            builder.push_call(func, new_args, rty).ok()?
        }
        InstKind::Intrinsic { name, args } => {
            let new_args = remap_values(source, args, builder, memo)?;
            builder.push_intrinsic(name, new_args, rty).ok()?
        }

        InstKind::Global(name) => builder.push_global(name, rty).ok()?,
        InstKind::Extern(name) => builder.push_inst(InstKind::Extern(name.clone()), rty).ok()?,

        // Effectful / opaque — cannot remap
        InstKind::Alloca { .. }
        | InstKind::Load { .. }
        | InstKind::Store { .. }
        | InstKind::StorageView { .. }
        | InstKind::StorageViewIndex { .. }
        | InstKind::StorageViewLen { .. }
        | InstKind::OutputPtr { .. }
        | InstKind::Soac(_) => return None,
    };

    memo.insert(value, new_val);
    Some(new_val)
}

fn remap_values(
    source: &FuncBody,
    values: &[ValueId],
    builder: &mut crate::ssa::builder::FuncBuilder,
    memo: &mut HashMap<ValueId, ValueId>,
) -> Option<Vec<ValueId>> {
    values.iter().map(|&v| remap_value(source, v, builder, memo)).collect()
}

/// Convenience wrapper: remap a single value from the entry body using a fresh memo
/// pre-seeded with entry param mappings.
pub fn remap_entry_value(ctx: &mut ParallelizeCtx, value: ValueId) -> Option<ValueId> {
    let body = ctx.entry.body.clone();
    let mut memo: HashMap<ValueId, ValueId> =
        body.params.iter().enumerate().map(|(i, (src, _, _))| (*src, ctx.builder.get_param(i))).collect();

    remap_value(&body, value, &mut ctx.builder, &mut memo)
}
