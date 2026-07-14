//! Focused construction API for planner-generated kernel entries.

use crate::ast::{Span, TypeName};
use crate::ssa::framework::BlockId;
use crate::ssa::types::{ControlHeader, EntryInput, EntryOutput, ExecutionModel};
use crate::{interface, LookupMap, ResourceId};
use polytype::Type;
use smallvec::smallvec;

use super::graph_ops;
use super::program::{LogicalSize, PlannedEntry, SemanticResourceDecl, SemanticResourceRef};
use super::soac::screma;
use super::types::{EGraph, NodeId, SkeletonTerminator, Soac, SoacDestination, SoacInputType};

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

impl EntryBuilder {
    pub fn new_compute(name: String, local_size: (u32, u32, u32)) -> Self {
        let graph = EGraph::new();
        let current_block = graph.skeleton.entry;
        Self {
            graph,
            control_headers: LookupMap::new(),
            current_block,
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

    fn declare(
        &mut self,
        resource: ResourceId,
        role: interface::StorageRole,
        elem_ty: Type<TypeName>,
        size: LogicalSize,
    ) {
        self.resource_declarations.push(SemanticResourceDecl {
            resource: SemanticResourceRef(resource),
            role,
            elem_ty,
            size,
        });
    }

    pub fn declare_intermediate_storage_sized(
        &mut self,
        resource: ResourceId,
        elem_ty: Type<TypeName>,
        size: LogicalSize,
    ) {
        self.declare(resource, interface::StorageRole::Intermediate, elem_ty, size);
    }

    pub fn declare_output_storage(&mut self, resource: ResourceId, elem_ty: Type<TypeName>) {
        self.declare_output_storage_sized(resource, elem_ty, LogicalSize::Unspecified);
    }

    pub fn declare_output_storage_sized(
        &mut self,
        resource: ResourceId,
        elem_ty: Type<TypeName>,
        size: LogicalSize,
    ) {
        self.declare(resource, interface::StorageRole::Output, elem_ty, size);
    }

    pub fn graph_mut(&mut self) -> &mut EGraph {
        &mut self.graph
    }

    pub fn control_headers_mut(&mut self) -> &mut LookupMap<BlockId, ControlHeader> {
        &mut self.control_headers
    }

    pub fn set_current_block(&mut self, block: BlockId) {
        self.current_block = block;
    }

    pub fn emit_storage_view(&mut self, resource: ResourceId, ty: Type<TypeName>) -> NodeId {
        graph_ops::intern_resource_view(&mut self.graph, resource, ty, Some(self.span))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn emit_pending_map_into(
        &mut self,
        region: super::types::RegionId,
        input_array: NodeId,
        input_array_ty: Type<TypeName>,
        input_elem_ty: Type<TypeName>,
        output_elem_ty: Type<TypeName>,
        captures: Vec<NodeId>,
        output_view: NodeId,
        output_view_ty: Type<TypeName>,
    ) -> NodeId {
        let tuple_ty = Type::Constructed(TypeName::Tuple(1), vec![output_view_ty.clone()]);
        graph_ops::emit_pending_soac(
            &mut self.graph,
            self.current_block,
            Soac::Screma(screma::Op {
                body: screma::Body {
                    inputs: vec![SoacInputType {
                        array: input_array_ty,
                        element: input_elem_ty,
                    }],
                    maps: vec![screma::Map {
                        body: super::types::SegBody { region, captures },
                        input_indices: vec![0],
                        output_element_type: output_elem_ty,
                        destination: SoacDestination::OutputView,
                        result_type: output_view_ty,
                    }],
                    kind: screma::Kind::Map,
                },
                state: screma::SemanticState::Serial,
            }),
            smallvec![input_array, output_view],
            tuple_ty,
            &mut self.next_effect,
            Some(self.span),
        )
    }

    pub fn emit_load(&mut self, place: NodeId, elem_ty: Type<TypeName>) -> NodeId {
        graph_ops::emit_load(
            &mut self.graph,
            self.current_block,
            place,
            elem_ty,
            &mut self.next_effect,
            Some(self.span),
        )
    }

    pub fn build(mut self) -> PlannedEntry {
        self.graph.skeleton.blocks[self.current_block].term = SkeletonTerminator::Return(None);
        PlannedEntry {
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
