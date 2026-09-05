//! Focused construction API for planner-generated kernel entries.

use crate::ast::{Span, TypeName};
use crate::flow::{BlockId, ExecutionModel};
use crate::interface::{self, EntryInput, EntryOutput};
use crate::types;
use crate::EntryId;
use crate::{FunctionId, ResourceId};
use polytype::Type;
use smallvec::SmallVec;
use wyn_base::IdSource;

use super::graph_ops;
use super::program::{
    AllocatedEntry, PlannedEntry, SemanticOpIdSource, SemanticResourceDecl, SemanticResourceRef,
};
use super::soac::screma;
use super::types::{
    EGraph, EffectToken, OperandRef, PlaceId, ResultBinding, Semantic, SkeletonTerminator, Soac,
    SoacInputType, ValueId, WynLanguage,
};

type AllocatedGraph = EGraph<Semantic<SemanticResourceRef>>;

pub struct EntryBuilder<'a> {
    graph: AllocatedGraph,
    id: EntryId,
    current_block: BlockId,
    name: String,
    span: Span,
    execution_model: ExecutionModel,
    inputs: Vec<EntryInput>,
    outputs: Vec<EntryOutput>,
    resource_declarations: Vec<SemanticResourceDecl>,
    semantic_ids: &'a mut SemanticOpIdSource,
    effect_ids: &'a mut IdSource<EffectToken>,
}

impl<'a> EntryBuilder<'a> {
    pub fn new_compute(
        name: String,
        local_size: (u32, u32, u32),
        identities: &mut super::program::ProgramIdentities,
        semantic_ids: &'a mut SemanticOpIdSource,
        effect_ids: &'a mut IdSource<EffectToken>,
    ) -> Self {
        let id = identities.alloc_entry(name.clone());
        let graph = EGraph::new();
        let current_block = graph.skeleton.entry;
        Self {
            graph,
            id,
            current_block,
            name,
            span: Span::generated(),
            execution_model: ExecutionModel::Compute { local_size },
            inputs: Vec::new(),
            outputs: Vec::new(),
            resource_declarations: Vec::new(),
            semantic_ids,
            effect_ids,
        }
    }

    fn declare(&mut self, resource: ResourceId, role: interface::StorageRole) {
        self.resource_declarations.push(SemanticResourceDecl {
            resource: SemanticResourceRef(resource),
            role,
        });
    }

    pub fn declare_intermediate_storage(&mut self, resource: ResourceId) {
        self.declare(resource, interface::StorageRole::Intermediate);
    }

    pub fn declare_input_storage(&mut self, resource: ResourceId) {
        self.declare(resource, interface::StorageRole::Input);
    }

    pub fn declare_output_storage(&mut self, resource: ResourceId) {
        self.declare(resource, interface::StorageRole::Output);
    }

    pub fn graph_mut(&mut self) -> &mut AllocatedGraph {
        &mut self.graph
    }

    pub fn construction_parts_mut(&mut self) -> (&mut AllocatedGraph, &mut IdSource<EffectToken>) {
        (&mut self.graph, self.effect_ids)
    }

    pub fn set_current_block(&mut self, block: BlockId) {
        self.current_block = block;
    }

    pub fn emit_storage_view(&mut self, resource: ResourceId, ty: Type<TypeName>) -> ValueId {
        graph_ops::intern_resource_view(&mut self.graph, resource, ty, Some(self.span))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn emit_pending_map_into(
        &mut self,
        region: FunctionId,
        input_array: ValueId,
        input_array_ty: Type<TypeName>,
        output_elem_ty: Type<TypeName>,
        captures: Vec<OperandRef>,
        output_view: ValueId,
        output_view_ty: Type<TypeName>,
    ) -> ValueId {
        let input = SoacInputType::array(input_array_ty);
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
        inputs: Vec<(ValueId, SoacInputType)>,
        pre: screma::Lambda,
        output_views: Vec<(ValueId, Type<TypeName>)>,
    ) -> ValueId {
        debug_assert_eq!(inputs.len(), pre.parameter_types.len());
        debug_assert_eq!(output_views.len(), pre.result_types.len());
        let tuple_ty = Type::Constructed(
            TypeName::Tuple(output_views.len()),
            output_views.iter().map(|(_, ty)| ty.clone()).collect(),
        );
        let result_types = pre.result_types.clone();
        let operands = inputs
            .iter()
            .map(|(node, _)| self.graph.operand_ref(*node))
            .collect::<SmallVec<[OperandRef; 4]>>();
        let result_abi = super::ir::by_value_function_result::<WynLanguage>(tuple_ty.clone());
        let result_fields = result_abi
            .top_level_fields()
            .into_iter()
            .zip(&output_views)
            .map(|(field, (view, _))| {
                graph_ops::bind_result_to_view(&mut self.graph, &field, *view)
                    .expect("a map destination must match its logical result shape")
            })
            .collect::<Vec<_>>();
        let result = ResultBinding::product(tuple_ty, result_fields);
        let id = self.semantic_ids.next_id();
        let result = graph_ops::emit_pending_soac(
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
                        ownership: types::SoacOwnership::Fresh,
                    })
                    .collect(),
                state: screma::SemanticState::Serial,
            }),
            operands,
            result,
            self.effect_ids,
            Some(self.span),
        );
        graph_ops::pack_result_values(&mut self.graph, &result)
            .expect("a view-routed map result can be assembled")
    }

    pub fn emit_load(&mut self, place: PlaceId, elem_ty: Type<TypeName>) -> ValueId {
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
        PlannedEntry::new(AllocatedEntry {
            id: self.id,
            name: self.name,
            span: self.span,
            execution_model: self.execution_model,
            parameter_inputs: (0..self.inputs.len())
                .map(|index| vec![super::program::InputSlotId(index)])
                .collect(),
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
            internal_results: Vec::new(),
            resource_declarations: self.resource_declarations,
            params: super::ir::Parameters::new(),
            result: super::ir::by_value_function_result::<WynLanguage>(Type::Constructed(
                TypeName::Unit,
                vec![],
            )),
            graph: self.graph,
        })
    }
}
