//! SPIKE: programmatic `EgirEntry` synthesis API.
//!
//! Question this module answers: can the EgirEntry construction logic
//! that today only lives inside `from_tlc::convert_entry_point` be
//! extracted as a reusable, TLC-independent builder?
//!
//! Motivation: the reduce / scan / redomap migration to the EGIR-side
//! parallelization boundary needs to synthesize multi-phase pipelines
//! (e.g. a `phase2_combine` entry for reduce) directly at EGIR, without
//! round-tripping through a fresh TLC `Def` + `from_tlc`. This spike
//! demonstrates the API surface that would be required.
//!
//! The spike validates one shape: a phase2-style entry that reads from
//! a runtime-sized storage view, runs a `PendingSoac::Reduce` over it,
//! and stores the scalar result to another storage binding.
//!
//! Scope: deliberately *not* land-ready. The methods here re-implement
//! primitives that already exist inside `Converter` in `from_tlc.rs`;
//! a real landing would factor a shared `GraphBuilder` out of
//! `Converter` and have both consumers (TLC conversion + EGIR phase
//! synthesis) call into it.

use std::collections::HashMap;

use polytype::Type;
use smallvec::smallvec;

use crate::BindingRef;
use crate::ast::{Span, TypeName};
use crate::interface;
use crate::ssa::framework::BlockId;
use crate::ssa::types::{ConstantValue, ControlHeader, ExecutionModel};

use super::graph_ops;
use super::program::EgirEntry;
use super::types::{EGraph, NodeId, PendingSoac, SkeletonTerminator, SoacDestination};
use crate::ssa::types::{EntryInput, EntryOutput};

/// Build a synthesized `EgirEntry` programmatically. Mirrors the primitive
/// operations `from_tlc::Converter` exposes, but holds no TLC-side state.
///
/// Typical usage (phase2 of a reduce):
/// ```text
/// let mut b = EntryBuilder::new_compute("compute_sum_phase2_combine".into(), (1, 1, 1));
/// b.declare_intermediate_storage(0, 1, f32_ty());
/// b.declare_output_storage(0, 2, f32_ty());
/// let view = b.emit_storage_view(0, 1, partials_arr_ty);
/// let init = b.emit_constant(ConstantValue::from_f32(0.0), f32_ty());
/// let r = b.emit_pending_reduce("op_add", view, partials_arr_ty, f32_ty(), init, vec![]);
/// let zero = b.emit_u32(0);
/// b.emit_storage_store(0, 2, zero, r, f32_ty());
/// let entry = b.build();
/// ```
pub struct EntryBuilder {
    graph: EGraph,
    control_headers: HashMap<BlockId, ControlHeader>,
    current_block: BlockId,
    name: String,
    span: Span,
    execution_model: ExecutionModel,
    inputs: Vec<EntryInput>,
    outputs: Vec<EntryOutput>,
    storage_bindings: Vec<interface::StorageBindingDecl>,
    params: Vec<(Type<TypeName>, String)>,
    return_ty: Type<TypeName>,
    next_effect: u32,
}

impl EntryBuilder {
    /// New compute-shader entry. Always returns Unit; effectful writes
    /// happen via `emit_storage_store`.
    pub fn new_compute(name: String, local_size: (u32, u32, u32)) -> Self {
        let graph = EGraph::new();
        let entry = graph.skeleton.entry;
        EntryBuilder {
            graph,
            control_headers: HashMap::new(),
            current_block: entry,
            name,
            span: Span::new(0, 0, 0, 0),
            execution_model: ExecutionModel::Compute { local_size },
            inputs: Vec::new(),
            outputs: Vec::new(),
            storage_bindings: Vec::new(),
            params: Vec::new(),
            return_ty: Type::Constructed(TypeName::Unit, vec![]),
            next_effect: 1,
        }
    }

    // ---- Storage interface declarations ----------------------------------

    pub fn declare_intermediate_storage(&mut self, set: u32, binding: u32, elem_ty: Type<TypeName>) {
        self.storage_bindings.push(interface::StorageBindingDecl {
            binding: BindingRef::new(set, binding),
            role: interface::StorageRole::Intermediate,
            elem_ty,
            length: None,
        });
    }

    pub fn declare_output_storage(&mut self, set: u32, binding: u32, elem_ty: Type<TypeName>) {
        self.storage_bindings.push(interface::StorageBindingDecl {
            binding: BindingRef::new(set, binding),
            role: interface::StorageRole::Output,
            elem_ty,
            length: None,
        });
    }

    // ---- Pure-op primitives ----------------------------------------------
    //
    // Thin wrappers over `graph_ops::*` that pre-fill the builder's
    // current span. All graph manipulation goes through the shared
    // module so the three EGIR-construction contexts stay consistent.

    fn span(&self) -> Option<Span> {
        Some(self.span)
    }

    /// Direct mutable access to the underlying EGraph — used when a
    /// caller needs `graph_ops` operations not yet wrapped on the
    /// builder (e.g. `clone_pure_subgraph` for copying a Redomap
    /// neutral-element subgraph across entries).
    pub fn graph_mut(&mut self) -> &mut EGraph {
        &mut self.graph
    }

    /// Mutable access to the control-header map — used when hand-building
    /// structured control flow (loops / selections) directly on the graph,
    /// e.g. the workgroup-parallel phase2 tree reduce.
    pub fn control_headers_mut(&mut self) -> &mut HashMap<BlockId, ControlHeader> {
        &mut self.control_headers
    }

    /// Repoint the "current" block. `build()` finalizes the current block
    /// with `Return(None)`, so a multi-block body must set this to its exit
    /// block before calling `build()`.
    pub fn set_current_block(&mut self, block: BlockId) {
        self.current_block = block;
    }

    pub fn emit_u32(&mut self, n: u32) -> NodeId {
        let span = self.span();
        graph_ops::intern_u32(&mut self.graph, n, span)
    }

    pub fn emit_constant(&mut self, value: ConstantValue, ty: Type<TypeName>) -> NodeId {
        graph_ops::intern_constant(&mut self.graph, value, ty)
    }

    pub fn emit_storage_view(&mut self, set: u32, binding: u32, view_ty: Type<TypeName>) -> NodeId {
        let span = self.span();
        graph_ops::intern_storage_view(&mut self.graph, set, binding, view_ty, span)
    }

    /// Emit a `PendingSoac::Reduce` side-effect. `soac_expand` will lower
    /// this into a serial reduce loop within the current block. Returns
    /// the result NodeId (consumable by downstream stores).
    ///
    /// Operand layout matches `convert_soac_reduce`:
    /// `[input_array, init, ...captures]`.
    pub fn emit_pending_reduce(
        &mut self,
        func: String,
        input_array_nid: NodeId,
        input_array_ty: Type<TypeName>,
        input_elem_ty: Type<TypeName>,
        init_nid: NodeId,
        captures: Vec<NodeId>,
    ) -> NodeId {
        let mut operands: smallvec::SmallVec<[NodeId; 4]> = smallvec![input_array_nid, init_nid];
        operands.extend(captures.into_iter());
        let span = self.span();
        graph_ops::emit_pending_soac(
            &mut self.graph,
            self.current_block,
            PendingSoac::Reduce {
                func,
                input_array_type: input_array_ty,
                input_elem_type: input_elem_ty.clone(),
            },
            operands,
            input_elem_ty,
            &mut self.next_effect,
            span,
        )
    }

    /// Emit a `PendingSoac::Scan { destination: OutputView }`. Operand layout
    /// matches the post-`rewrite_map_scan_to_into` shape:
    /// `[input_array, init, ...captures, output_view]`. Result is unit.
    pub fn emit_pending_scan_into(
        &mut self,
        func: String,
        input_array_nid: NodeId,
        input_array_ty: Type<TypeName>,
        input_elem_ty: Type<TypeName>,
        init_nid: NodeId,
        captures: Vec<NodeId>,
        output_view_nid: NodeId,
    ) -> NodeId {
        let mut operands: smallvec::SmallVec<[NodeId; 4]> = smallvec![input_array_nid, init_nid];
        operands.extend(captures.into_iter());
        operands.push(output_view_nid);
        let unit_ty = Type::Constructed(TypeName::Unit, vec![]);
        let span = self.span();
        graph_ops::emit_pending_soac(
            &mut self.graph,
            self.current_block,
            PendingSoac::Scan {
                // A serial into-scan (e.g. phase 2's block-sum scan) is never
                // re-parallelized, so its element-step and pure combiner are
                // the same function.
                reduce_func: func.clone(),
                func,
                input_array_type: input_array_ty,
                input_elem_type: input_elem_ty,
                destination: SoacDestination::OutputView,
            },
            operands,
            unit_ty,
            &mut self.next_effect,
            span,
        )
    }

    /// Emit a single-input `PendingSoac::Map { destination: OutputView }`.
    /// Operand layout: `[input_array, ...captures, output_view]`. Result is unit.
    pub fn emit_pending_map_into(
        &mut self,
        func: String,
        input_array_nid: NodeId,
        input_array_ty: Type<TypeName>,
        input_elem_ty: Type<TypeName>,
        output_elem_ty: Type<TypeName>,
        captures: Vec<NodeId>,
        output_view_nid: NodeId,
    ) -> NodeId {
        let mut operands: smallvec::SmallVec<[NodeId; 4]> = smallvec![input_array_nid];
        operands.extend(captures.into_iter());
        operands.push(output_view_nid);
        let unit_ty = Type::Constructed(TypeName::Unit, vec![]);
        let span = self.span();
        graph_ops::emit_pending_soac(
            &mut self.graph,
            self.current_block,
            PendingSoac::Map {
                func,
                input_array_types: vec![input_array_ty],
                input_elem_types: vec![input_elem_ty],
                output_elem_type: output_elem_ty,
                destination: SoacDestination::OutputView,
            },
            operands,
            unit_ty,
            &mut self.next_effect,
            span,
        )
    }

    /// Emit a `Load` from a place (typically a `ViewIndex` node).
    /// Returns the loaded value's NodeId.
    pub fn emit_load(&mut self, place_nid: NodeId, elem_ty: Type<TypeName>) -> NodeId {
        use super::graph_ops::alloc_effect;
        use super::types::{SideEffect, SideEffectKind};
        use crate::ssa::types::InstKind;
        let span = self.span();
        let result = self.graph.alloc_side_effect_result(elem_ty);
        let eff_in = alloc_effect(&mut self.next_effect);
        let eff_out = alloc_effect(&mut self.next_effect);
        self.graph.skeleton.blocks[self.current_block].side_effects.push(SideEffect {
            kind: SideEffectKind::Inst(InstKind::Load {
                place: Default::default(),
            }),
            operand_nodes: smallvec![place_nid],
            result: Some(result),
            effects: Some((eff_in, eff_out)),
            span,
        });
        result
    }

    /// Emit a `Store` of `value` to `storage[set, binding][index]`.
    pub fn emit_storage_store(
        &mut self,
        set: u32,
        binding: u32,
        index_nid: NodeId,
        value_nid: NodeId,
        elem_ty: Type<TypeName>,
    ) {
        let arr_ty = Type::Constructed(
            TypeName::Array,
            vec![
                elem_ty.clone(),
                Type::Variable(0),
                Type::Constructed(TypeName::ArrayVariantView, vec![]),
            ],
        );
        let view_nid = self.emit_storage_view(set, binding, arr_ty);
        let span = self.span();
        graph_ops::emit_storage_store(
            &mut self.graph,
            self.current_block,
            view_nid,
            index_nid,
            value_nid,
            elem_ty,
            &mut self.next_effect,
            span,
        );
    }

    /// Finalize: set the entry block's terminator to `Return(None)` and
    /// hand back an `EgirEntry`.
    pub fn build(mut self) -> EgirEntry {
        self.graph.skeleton.blocks[self.current_block].term = SkeletonTerminator::Return(None);
        EgirEntry::new(
            self.name,
            self.span,
            self.execution_model,
            self.inputs,
            self.outputs,
            self.storage_bindings,
            self.params,
            self.return_ty,
            self.graph,
            self.control_headers,
        )
    }
}

#[cfg(test)]
#[path = "builder_tests.rs"]
mod builder_tests;
