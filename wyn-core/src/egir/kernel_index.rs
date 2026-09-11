//! Scalar invocation indices and chunk bounds shared by scheduling and expansion.

use super::graph_ops;
use super::ir::Family;
use super::types::{EGraph, PureOp, ValueId};
use crate::ast::TypeName;
use crate::builtins::{catalog, BuiltinId};
use crate::op;
use polytype::Type;
use smallvec::smallvec;

/// Read a u32 dispatch builtin in the caller's i32 or u32 index type.
pub(crate) fn emit_invocation_index<P: Family>(
    graph: &mut EGraph<P>,
    builtin: BuiltinId,
    index_ty: &Type<TypeName>,
) -> Result<ValueId, String> {
    let value = graph_ops::intern_intrinsic(
        graph,
        builtin,
        smallvec![],
        Type::Constructed(TypeName::UInt(32), vec![]),
        None,
    );
    cast_u32_to_index(graph, value, index_ty)
}

/// Emit the chunk-arithmetic preamble (`tid`, `chunk_start`,
/// `chunk_len`) as pure nodes in `graph`. Caller supplies the
/// `input_len` ValueId (typed `i32` or `u32`) — for StorageView inputs that's a
/// `_w_intrinsic_storage_len(set, binding)` call; for Range inputs
/// it's the Range's own `len` operand. Returns
/// `(tid, chunk_start, chunk_len)`.
pub(crate) fn emit_chunk_arithmetic<P: Family>(
    graph: &mut EGraph<P>,
    workgroup_width: u32,
    input_len: ValueId,
) -> Result<(ValueId, ValueId, ValueId), String> {
    let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
    // The chunk arithmetic runs in the input's *index* type: storage-view
    // inputs index in u32 (`_w_intrinsic_storage_len`), Range inputs in the
    // range's own element type (typically i32). Computing in u32 and feeding
    // a u32 `chunk_start`/`chunk_len` into an i32 Range produced an
    // `OpCompositeConstruct` whose constituents didn't match the i32
    // `{start, step, len}` struct (spirv-val rejected it). Derive the index
    // type from `input_len` and emit all arithmetic there.
    let index_ty = graph
        .nodes
        .get(input_len)
        .map(|node| node.ty.clone())
        .ok_or_else(|| format!("chunk input length {input_len:?} has no type"))?;
    let is_u32 = index_ty == u32_ty;

    // `tid`/`num_workgroups` are u32 intrinsics. The returned `tid` stays u32
    // (callers use it as a `partials[tid]` storage index); the index-typed
    // copies feed the chunk math.
    let tid = graph_ops::intern_intrinsic(
        graph,
        catalog().known().thread_id,
        smallvec![],
        u32_ty.clone(),
        None,
    );
    let tid_idx = cast_u32_to_index(graph, tid, &index_ty)?;
    let nwg_idx = emit_invocation_index(graph, catalog().known().num_workgroups, &index_ty)?;

    // Runtime total thread count = num_workgroups.x * workgroup width. With a
    // `derived_from_input_length` dispatch (~ceil(n / width) workgroups) this
    // makes chunk_size ≈ 1, so each thread reduces ~one element — a saturating
    // grid rather than a fixed-width one. `workgroup_width` is the
    // compile-time per-workgroup width.
    let wg_width = intern_index_lit(graph, workgroup_width, &index_ty);
    let total = graph_ops::intern_binop(
        graph,
        op::BinaryOperator::Multiply,
        nwg_idx,
        wg_width,
        index_ty.clone(),
        None,
    );
    let one = intern_index_lit(graph, 1, &index_ty);
    let total_minus_one = graph_ops::intern_binop(
        graph,
        op::BinaryOperator::Subtract,
        total,
        one,
        index_ty.clone(),
        None,
    );
    let len_plus = graph_ops::intern_binop(
        graph,
        op::BinaryOperator::Add,
        input_len,
        total_minus_one,
        index_ty.clone(),
        None,
    );
    let chunk_size = graph_ops::intern_binop(
        graph,
        op::BinaryOperator::Divide,
        len_plus,
        total,
        index_ty.clone(),
        None,
    );
    let raw_chunk_start = graph_ops::intern_binop(
        graph,
        op::BinaryOperator::Multiply,
        tid_idx,
        chunk_size,
        index_ty.clone(),
        None,
    );
    let scalar_type = if is_u32 { TypeName::UInt(32) } else { TypeName::Int(32) };
    let min_id = catalog()
        .specialize_numeric(catalog().known().min, &scalar_type)
        .ok_or_else(|| format!("min specialization missing for {scalar_type:?}"))?;
    // Clamp idle workers to the end before subtraction. For n < workers this
    // produces `(start=n,len=0)` instead of underflowing `n-start`.
    let chunk_start = graph_ops::intern_intrinsic(
        graph,
        min_id,
        smallvec![raw_chunk_start, input_len],
        index_ty.clone(),
        None,
    );
    let remaining = graph_ops::intern_binop(
        graph,
        op::BinaryOperator::Subtract,
        input_len,
        chunk_start,
        index_ty.clone(),
        None,
    );
    let chunk_len =
        graph_ops::intern_intrinsic(graph, min_id, smallvec![chunk_size, remaining], index_ty, None);
    Ok((tid, chunk_start, chunk_len))
}

/// Integer literal `n` typed as `index_ty` (`u32` → `PureOp::Uint`, else
/// `PureOp::Int`).
fn intern_index_lit<P: Family>(graph: &mut EGraph<P>, n: u32, index_ty: &Type<TypeName>) -> ValueId {
    let op = match index_ty {
        Type::Constructed(TypeName::UInt(32), _) => PureOp::Uint(n.to_string()),
        _ => PureOp::Int(n.to_string()),
    };
    graph.intern_pure(op, smallvec![], index_ty.clone(), None)
}

/// Cast a u32 value into `index_ty`: identity for u32, else the per-type
/// bitcast intrinsic (`i32.u32`).
pub(crate) fn cast_u32_to_index<P: Family>(
    graph: &mut EGraph<P>,
    v: ValueId,
    index_ty: &Type<TypeName>,
) -> Result<ValueId, String> {
    match index_ty {
        Type::Constructed(TypeName::UInt(32), _) => Ok(v),
        Type::Constructed(TypeName::Int(32), _) => {
            let conversion = catalog()
                .conversion(&TypeName::Int(32), &TypeName::UInt(32))
                .ok_or_else(|| "u32-to-i32 conversion missing from catalog".to_string())?;
            Ok(graph_ops::intern_intrinsic(
                graph,
                conversion,
                smallvec![v],
                index_ty.clone(),
                None,
            ))
        }
        other => Err(format!("chunk arithmetic: unsupported index type {:?}", other)),
    }
}
