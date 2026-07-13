//! Programmatic planned-kernel synthesis API used by the EGIR-side
//! parallelization phases that synthesize extra compute entries (phase
//! 2 / phase 3 of the Screma transform).

use crate::LookupMap;

use polytype::Type;
use smallvec::smallvec;

use crate::ast::{Span, TypeName};
use crate::ssa::framework::BlockId;
use crate::ssa::types::{ConstantValue, ControlHeader, ExecutionModel};
use crate::{interface, ResourceId};

use super::graph_ops;
use super::program::{
    PhysicalEntry, PhysicalResourceTable, PlannedKernelBody, SemanticResourceDecl, SemanticResourceRef,
};
use super::types::{EGraph, EgirSoac, NodeId, SkeletonTerminator, SoacDestination};
use crate::ssa::types::{EntryInput, EntryOutput};

/// Build a closed `PlannedKernelBody` programmatically. Mirrors the
/// primitive operations `from_tlc::Converter` exposes, but holds no
/// TLC-side state.
pub struct EntryBuilder {
    graph: EGraph,
    control_headers: LookupMap<BlockId, ControlHeader>,
    current_block: BlockId,
    name: String,
    span: Span,
    execution_model: ExecutionModel,
    inputs: Vec<EntryInput>,
    outputs: Vec<EntryOutput>,
    resource_declarations: Vec<SemanticResourceDecl>,
    params: Vec<(Type<TypeName>, String)>,
    return_ty: Type<TypeName>,
    next_effect: u32,
}

/// Commit builder for a fully planned physical entry. It consumes either a
/// semantic projection or a closed materialization recipe rather than
/// exposing a partially initialized physical record.
pub struct PhysicalEntryBuilder<'a> {
    body: &'a PlannedKernelBody,
    resources: &'a PhysicalResourceTable,
}

struct PhysicalEntryDraft {
    name: String,
    span: Span,
    execution_model: ExecutionModel,
    inputs: Vec<EntryInput>,
    outputs: Vec<EntryOutput>,
    resources: Vec<SemanticResourceDecl>,
    params: Vec<(Type<TypeName>, String)>,
    return_ty: Type<TypeName>,
    graph: EGraph,
    control_headers: LookupMap<BlockId, ControlHeader>,
    aliases: LookupMap<NodeId, NodeId>,
    output_routes: Vec<super::program::OutputRoute>,
}

impl<'a> PhysicalEntryBuilder<'a> {
    pub fn new(body: &'a PlannedKernelBody, resources: &'a PhysicalResourceTable) -> Self {
        Self { body, resources }
    }

    pub fn build(self) -> Result<PhysicalEntry, String> {
        let mut draft = PhysicalEntryDraft {
            name: self.body.name.clone(),
            span: self.body.span,
            execution_model: self.body.execution_model.clone(),
            inputs: self.body.inputs.clone(),
            outputs: self.body.outputs.clone(),
            resources: self.body.resource_declarations.clone(),
            params: self.body.params.clone(),
            return_ty: self.body.return_ty.clone(),
            graph: self.body.graph.clone(),
            control_headers: self.body.control_headers.clone(),
            aliases: self.body.aliases.clone(),
            output_routes: self.body.output_routes.clone(),
        };
        super::program::physicalize_graph_resources(&mut draft.graph, self.resources)?;
        for input in &mut draft.inputs {
            super::program::physicalize_type_resources(&mut input.ty, self.resources);
        }
        for output in &mut draft.outputs {
            super::program::physicalize_type_resources(&mut output.ty, self.resources);
        }
        for (ty, _) in &mut draft.params {
            super::program::physicalize_type_resources(ty, self.resources);
        }
        super::program::physicalize_type_resources(&mut draft.return_ty, self.resources);
        for declaration in &mut draft.resources {
            super::program::physicalize_type_resources(&mut declaration.elem_ty, self.resources);
        }
        if draft.name.is_empty() {
            return Err("physical entry has no publication name".into());
        }
        for route in &draft.output_routes {
            if route.slot.0 >= draft.outputs.len() {
                return Err(format!(
                    "physical entry `{}` routes invalid output slot {}",
                    draft.name, route.slot.0
                ));
            }
        }
        let storage_bindings = draft
            .resources
            .into_iter()
            .map(|declaration| {
                let resource = declaration.resource.resource().ok_or_else(|| {
                    format!(
                        "physical entry `{}` contains a pending resource binding",
                        draft.name
                    )
                })?;
                let binding = self.resources.binding(resource).ok_or_else(|| {
                    format!(
                        "physical entry `{}` references missing resource {:?}",
                        draft.name, resource
                    )
                })?;
                Ok(interface::StorageBindingDecl {
                    binding,
                    role: declaration.role,
                    elem_ty: declaration.elem_ty,
                    length: super::program::buffer_len(&declaration.size, self.resources),
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(PhysicalEntry {
            name: draft.name,
            span: draft.span,
            execution_model: draft.execution_model,
            inputs: draft.inputs,
            outputs: draft.outputs,
            storage_bindings,
            params: draft.params,
            return_ty: draft.return_ty,
            graph: draft.graph,
            control_headers: draft.control_headers,
            aliases: draft.aliases,
            output_routes: draft.output_routes,
        })
    }
}

impl EntryBuilder {
    /// New compute-shader entry. Always returns Unit; effectful writes
    /// happen via `emit_storage_store`.
    pub fn new_compute(name: String, local_size: (u32, u32, u32)) -> Self {
        let graph = EGraph::new();
        let entry = graph.skeleton.entry;
        EntryBuilder {
            graph,
            control_headers: LookupMap::new(),
            current_block: entry,
            name,
            span: Span::new(0, 0, 0, 0),
            execution_model: ExecutionModel::Compute { local_size },
            inputs: Vec::new(),
            outputs: Vec::new(),
            resource_declarations: Vec::new(),
            params: Vec::new(),
            return_ty: Type::Constructed(TypeName::Unit, vec![]),
            next_effect: 1,
        }
    }

    // ---- Storage interface declarations ----------------------------------

    pub fn declare_intermediate_storage(&mut self, resource: ResourceId, elem_ty: Type<TypeName>) {
        self.declare_intermediate_storage_sized(resource, elem_ty, None);
    }

    pub fn declare_intermediate_storage_sized(
        &mut self,
        resource: ResourceId,
        elem_ty: Type<TypeName>,
        length: Option<crate::pipeline_descriptor::BufferLen>,
    ) {
        self.resource_declarations.push(SemanticResourceDecl {
            resource: SemanticResourceRef::Resource(resource),
            role: interface::StorageRole::Intermediate,
            elem_ty,
            size: super::program::pending_logical_size(length.as_ref()),
        });
    }

    pub fn declare_intermediate_resource(
        &mut self,
        resource: ResourceId,
        elem_ty: Type<TypeName>,
        size: super::program::LogicalSize,
    ) {
        self.resource_declarations.push(SemanticResourceDecl {
            resource: SemanticResourceRef::Resource(resource),
            role: interface::StorageRole::Intermediate,
            elem_ty,
            size,
        });
    }

    pub fn declare_output_storage(&mut self, resource: ResourceId, elem_ty: Type<TypeName>) {
        self.declare_output_storage_sized(resource, elem_ty, None);
    }

    pub fn declare_output_storage_sized(
        &mut self,
        resource: ResourceId,
        elem_ty: Type<TypeName>,
        length: Option<crate::pipeline_descriptor::BufferLen>,
    ) {
        self.resource_declarations.push(SemanticResourceDecl {
            resource: SemanticResourceRef::Resource(resource),
            role: interface::StorageRole::Output,
            elem_ty,
            size: super::program::pending_logical_size(length.as_ref()),
        });
    }

    pub fn declare_output_resource(
        &mut self,
        resource: ResourceId,
        elem_ty: Type<TypeName>,
        size: super::program::LogicalSize,
    ) {
        self.resource_declarations.push(SemanticResourceDecl {
            resource: SemanticResourceRef::Resource(resource),
            role: interface::StorageRole::Output,
            elem_ty,
            size,
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
    /// builder (e.g. `clone_pure_subgraph` for copying a reducing Screma's
    /// neutral-element subgraph across entries).
    pub fn graph_mut(&mut self) -> &mut EGraph {
        &mut self.graph
    }

    /// Mutable access to the control-header map — used when hand-building
    /// structured control flow (loops / selections) directly on the graph,
    /// e.g. the workgroup-parallel phase2 tree reduce.
    pub fn control_headers_mut(&mut self) -> &mut LookupMap<BlockId, ControlHeader> {
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

    pub fn emit_storage_view(&mut self, resource: ResourceId, view_ty: Type<TypeName>) -> NodeId {
        let span = self.span();
        graph_ops::intern_resource_view(&mut self.graph, resource, view_ty, span)
    }

    /// Emit a `EgirSoac::Screma { 0 maps, 1 Scan acc, OutputView }` —
    /// the consolidated shape used by `synthesize_phase2_scan` for the
    /// sequential block-sum scan. Operand layout:
    /// `[input_array, init, ...captures, output_view]`. Result is a
    /// 1-tuple of the output_view's type (Screma's expansion requires a
    /// tuple result).
    pub fn emit_pending_scan_into(
        &mut self,
        region: super::types::RegionId,
        input_array_nid: NodeId,
        input_array_ty: Type<TypeName>,
        input_elem_ty: Type<TypeName>,
        init_nid: NodeId,
        captures: Vec<NodeId>,
        output_view_nid: NodeId,
        output_view_ty: Type<TypeName>,
    ) -> NodeId {
        let step_body = super::types::SegBody { region, captures };
        // `[input, init, output_view]` — captures live on `step_body`.
        let operands: smallvec::SmallVec<[NodeId; 4]> =
            smallvec![input_array_nid, init_nid, output_view_nid];
        let tuple_ty = Type::Constructed(TypeName::Tuple(1), vec![output_view_ty]);
        let span = self.span();
        graph_ops::emit_pending_soac(
            &mut self.graph,
            self.current_block,
            EgirSoac::Screma {
                map_bodies: vec![],
                accumulators: vec![super::types::ScremaOperator {
                    kind: crate::tlc::ScremaAccumulator::Scan,
                    // A serial into-scan is never re-parallelized, so the step
                    // and combine reference the same region.
                    step: step_body,
                    combine: super::types::SegBody {
                        region,
                        captures: vec![],
                    },
                    input_indices: vec![0],
                }],
                input_array_types: vec![input_array_ty],
                input_elem_types: vec![input_elem_ty],
                map_output_elem_types: vec![],
                map_input_indices: vec![],
                map_destinations: vec![],
                acc_destinations: vec![SoacDestination::OutputView],
            },
            operands,
            tuple_ty,
            &mut self.next_effect,
            span,
        )
    }

    /// Emit a `EgirSoac::Screma { 1 map (OutputView), 0 accs }` — the
    /// consolidated shape used by `synthesize_phase3_scan` for the
    /// chunked apply-offsets pass. Operand layout:
    /// `[input_array, ...captures, output_view]`. Result is a 1-tuple
    /// of the output_view's type.
    pub fn emit_pending_map_into(
        &mut self,
        region: super::types::RegionId,
        input_array_nid: NodeId,
        input_array_ty: Type<TypeName>,
        input_elem_ty: Type<TypeName>,
        output_elem_ty: Type<TypeName>,
        captures: Vec<NodeId>,
        output_view_nid: NodeId,
        output_view_ty: Type<TypeName>,
    ) -> NodeId {
        let map_body = super::types::SegBody { region, captures };
        // `[input, output_view]` — captures live on `map_body`.
        let operands: smallvec::SmallVec<[NodeId; 4]> = smallvec![input_array_nid, output_view_nid];
        let tuple_ty = Type::Constructed(TypeName::Tuple(1), vec![output_view_ty]);
        let span = self.span();
        graph_ops::emit_pending_soac(
            &mut self.graph,
            self.current_block,
            EgirSoac::Screma {
                map_bodies: vec![map_body],
                accumulators: vec![],
                input_array_types: vec![input_array_ty],
                input_elem_types: vec![input_elem_ty],
                map_output_elem_types: vec![output_elem_ty],
                map_input_indices: vec![vec![0]],
                map_destinations: vec![SoacDestination::OutputView],
                acc_destinations: vec![],
            },
            operands,
            tuple_ty,
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
            semantic_id: None,
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

    /// Emit a `Store` of `value` to `storage[binding][index]`.
    pub fn emit_storage_store(
        &mut self,
        resource: ResourceId,
        index_nid: NodeId,
        value_nid: NodeId,
        elem_ty: Type<TypeName>,
    ) {
        let arr_ty = Type::Constructed(
            TypeName::Array,
            vec![
                elem_ty.clone(),
                Type::Constructed(TypeName::ArrayVariantView, vec![]),
                Type::Variable(0),
                // buffer stamped from the binding by emit_storage_view.
                crate::types::no_buffer(),
            ],
        );
        let view_nid = self.emit_storage_view(resource, arr_ty);
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
    /// hand back a closed planned-kernel body.
    pub fn build(mut self) -> PlannedKernelBody {
        self.graph.skeleton.blocks[self.current_block].term = SkeletonTerminator::Return(None);
        PlannedKernelBody {
            name: self.name,
            span: self.span,
            execution_model: self.execution_model,
            inputs: self.inputs,
            outputs: self.outputs,
            resource_declarations: self.resource_declarations,
            params: self.params,
            return_ty: self.return_ty,
            graph: self.graph,
            control_headers: self.control_headers,
            aliases: LookupMap::new(),
            output_routes: Vec::new(),
        }
    }
}

#[cfg(test)]
#[path = "builder_tests.rs"]
mod builder_tests;
