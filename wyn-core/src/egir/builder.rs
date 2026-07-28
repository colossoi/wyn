//! Focused construction API for planner-generated kernel entries.

use crate::ast::{Span, TypeName};
use crate::flow::{BlockId, ExecutionModel};
use crate::interface::{self, EntryInput, EntryOutput};
use crate::ResourceId;
use polytype::Type;
use smallvec::SmallVec;

use super::graph_ops;
use super::program::{
    LogicalSize, PlannedEntry, SemanticOpIdSource, SemanticResourceDecl, SemanticResourceRef,
};
use super::soac::screma;
use super::types::{
    EGraph, EffectToken, NodeId, SkeletonTerminator, Soac, SoacDestination, SoacInputType, SoacPlacement,
};

pub struct EntryBuilder<'a> {
    graph: EGraph,
    current_block: BlockId,
    name: String,
    span: Span,
    execution_model: ExecutionModel,
    inputs: Vec<EntryInput>,
    outputs: Vec<EntryOutput>,
    resource_declarations: Vec<SemanticResourceDecl>,
    params: Vec<(Type<TypeName>, String)>,
    return_ty: Type<TypeName>,
    semantic_ids: &'a mut SemanticOpIdSource,
    effect_ids: &'a mut crate::IdSource<EffectToken>,
}

impl<'a> EntryBuilder<'a> {
    pub fn new_compute(
        name: String,
        local_size: (u32, u32, u32),
        semantic_ids: &'a mut SemanticOpIdSource,
        effect_ids: &'a mut crate::IdSource<EffectToken>,
    ) -> Self {
        let graph = EGraph::new();
        let current_block = graph.skeleton.entry;
        Self {
            graph,
            current_block,
            name,
            span: Span::new(0, 0, 0, 0),
            execution_model: ExecutionModel::Compute { local_size },
            inputs: Vec::new(),
            outputs: Vec::new(),
            resource_declarations: Vec::new(),
            params: Vec::new(),
            return_ty: Type::Constructed(TypeName::Unit, vec![]),
            semantic_ids,
            effect_ids,
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

    pub fn declare_input_storage_sized(
        &mut self,
        resource: ResourceId,
        elem_ty: Type<TypeName>,
        size: LogicalSize,
    ) {
        self.declare(resource, interface::StorageRole::Input, elem_ty, size);
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

    pub fn construction_parts_mut(&mut self) -> (&mut EGraph, &mut crate::IdSource<EffectToken>) {
        (&mut self.graph, self.effect_ids)
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
        output_elem_ty: Type<TypeName>,
        captures: Vec<NodeId>,
        output_view: NodeId,
        output_view_ty: Type<TypeName>,
    ) -> NodeId {
        let input = SoacInputType {
            array: input_array_ty,
        };
        let input_element_type = input.element();
        let pre = screma::Lambda::region(
            super::types::SegBody { region, captures },
            vec![input_element_type],
            vec![output_elem_ty.clone()],
        );
        self.emit_pending_map_into_views(
            vec![(input_array, input)],
            pre,
            vec![(output_view, output_view_ty)],
        )
    }

    /// Emit one canonical map whose complete pre-lambda writes all result
    /// fields to the corresponding output views.
    pub fn emit_pending_map_into_views(
        &mut self,
        inputs: Vec<(NodeId, SoacInputType)>,
        pre: screma::Lambda,
        output_views: Vec<(NodeId, Type<TypeName>)>,
    ) -> NodeId {
        debug_assert_eq!(inputs.len(), pre.parameter_types.len());
        debug_assert_eq!(output_views.len(), pre.result_types.len());
        let tuple_ty = Type::Constructed(
            TypeName::Tuple(output_views.len()),
            output_views.iter().map(|(_, ty)| ty.clone()).collect(),
        );
        let result_types = pre.result_types.clone();
        let operands = inputs
            .iter()
            .map(|(node, _)| *node)
            .chain(output_views.iter().map(|(node, _)| *node))
            .collect::<SmallVec<[NodeId; 4]>>();
        let id = self.semantic_ids.next_id();
        graph_ops::emit_pending_soac(
            &mut self.graph,
            self.current_block,
            id,
            Soac::Screma(screma::Op {
                inputs: inputs.into_iter().map(|(_, input)| input).collect(),
                form: screma::ScremaForm {
                    pre,
                    scans: Vec::new(),
                    reductions: Vec::new(),
                    post: screma::Lambda::identity(result_types.clone()),
                },
                result_state: result_types
                    .iter()
                    .map(|_| screma::ResultState {
                        destination: SoacDestination::fresh().placed(SoacPlacement::OutputView),
                    })
                    .collect(),
                state: screma::SemanticState::Serial,
            }),
            operands,
            tuple_ty,
            self.effect_ids,
            Some(self.span),
        )
    }

    pub fn emit_load(&mut self, place: NodeId, elem_ty: Type<TypeName>) -> NodeId {
        graph_ops::emit_load(
            &mut self.graph,
            self.current_block,
            place,
            elem_ty,
            self.effect_ids,
            Some(self.span),
        )
    }

    pub fn build(mut self) -> PlannedEntry {
        self.graph.skeleton.blocks[self.current_block].term = SkeletonTerminator::Return(None);
        PlannedEntry {
            name: self.name,
            span: self.span,
            execution_model: self.execution_model,
            inputs: self
                .inputs
                .into_iter()
                .map(|inner| super::ir::EntryInput {
                    inner,
                    resource: None,
                })
                .collect(),
            outputs: self
                .outputs
                .into_iter()
                .map(|inner| super::ir::EntryOutput {
                    inner,
                    resource: None,
                    routes: Vec::new(),
                })
                .collect(),
            resource_declarations: self.resource_declarations,
            params: self.params,
            return_ty: self.return_ty,
            graph: self.graph,
        }
    }
}
