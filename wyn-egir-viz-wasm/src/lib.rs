use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::HashMap;
use wasm_bindgen::prelude::*;
use wyn_core::ast::{NodeCounter, Span};
use wyn_core::egir::ir::{OperandRef, PlaceOp, ProgramFamily, SideEffectKind};
use wyn_core::egir::program::{
    CompilerResourceKind, LogicalResource, LogicalSize, MaterializationRequirement, NoStorageDeclaration,
    OutputWriter, RealizedOutputRoute, ResourceId, ResourceOrigin, RewriteGlobal, SemanticOpId,
    SemanticProgramData, SemanticResourceRef,
};
use wyn_core::egir::soac::screma::{Lambda, ScremaOperands};
use wyn_core::egir::soac::{filter, hist, screma};
use wyn_core::egir::types::{
    EffectOp, GraphResource as WynGraphResource, PlaceDestination, Raw, ResultDestination, SegExtent,
    SegResourceAccess, SegSpace, Semantic, Soac, SoacEffect, ValueKind, WynSoacPhase,
};
use wyn_core::error::CompilerError;
use wyn_core::module_manager::{ModuleManager, PreElaboratedPrelude};
use wyn_core::{BindingRef, FunctionId, ResourceAccess};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InspectPass {
    OptimizeSemanticOperations,
    PlanLogicalResources,
    ReifySoacs,
}

impl InspectPass {
    const OPTIMIZE_SEMANTIC_OPERATIONS: &'static str = "egir::optimize_semantic_operations";
    const PLAN_LOGICAL_RESOURCES: &'static str = "egir::plan_logical_resources";
    const REIFY_SOACS: &'static str = "egir::reify_soacs";

    fn parse(value: &str) -> Option<Self> {
        match value {
            Self::OPTIMIZE_SEMANTIC_OPERATIONS => Some(Self::OptimizeSemanticOperations),
            Self::PLAN_LOGICAL_RESOURCES => Some(Self::PlanLogicalResources),
            Self::REIFY_SOACS => Some(Self::ReifySoacs),
            _ => None,
        }
    }

    fn id(self) -> &'static str {
        match self {
            Self::OptimizeSemanticOperations => Self::OPTIMIZE_SEMANTIC_OPERATIONS,
            Self::PlanLogicalResources => Self::PLAN_LOGICAL_RESOURCES,
            Self::ReifySoacs => Self::REIFY_SOACS,
        }
    }
}

struct PreludeCache {
    prelude: PreElaboratedPrelude,
    start_node_counter: NodeCounter,
}

thread_local! {
    static PRELUDE_CACHE: RefCell<Option<PreludeCache>> = const { RefCell::new(None) };
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SourceSpan {
    pub start_line: usize,
    pub start_col: usize,
    pub end_line: usize,
    pub end_col: usize,
}

impl From<Span> for SourceSpan {
    fn from(span: Span) -> Self {
        Self {
            start_line: span.start_line,
            start_col: span.start_col,
            end_line: span.end_line,
            end_col: span.end_col,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VizError {
    pub message: String,
    pub span: Option<SourceSpan>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphGroup {
    pub id: String,
    pub label: String,
    pub kind: String,
    pub outputs: Vec<GraphOutput>,
    pub resource_declarations: Vec<GraphResourceDeclaration>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphOutput {
    pub slot: usize,
    pub ty: String,
    pub binding: Option<GraphBinding>,
    pub resource: Option<String>,
    pub kind: GraphOutputKind,
    pub routes: Vec<GraphOutputRoute>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphOutputKind {
    pub variant: String,
    pub destination: Option<String>,
    pub exposure: Option<String>,
    pub binding: Option<GraphBinding>,
    pub length: Option<GraphSize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphBinding {
    pub set: u32,
    pub binding: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphSize {
    pub variant: String,
    pub bytes: Option<u64>,
    pub binding: Option<GraphBinding>,
    pub resource: Option<String>,
    pub elem_bytes: Option<u32>,
    pub src_elem_bytes: Option<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphResourceDeclaration {
    pub resource: String,
    pub role: String,
    pub elem_ty: String,
    pub size: GraphSize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphResource {
    pub id: String,
    pub elem_ty: String,
    pub origin: GraphResourceOrigin,
    pub size: GraphSize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphResourceOrigin {
    pub variant: String,
    pub binding: Option<GraphBinding>,
    pub name: Option<String>,
    pub compiler_kind: Option<String>,
    pub owner: Option<String>,
    pub slot: Option<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphOutputRoute {
    pub source_block: String,
    pub source_value: String,
    pub writers: Vec<GraphOutputWriter>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphOutputWriter {
    pub kind: String,
    pub id: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: String,
    pub group: String,
    pub label: String,
    pub category: String,
    pub variant: String,
    pub detail: String,
    pub ty: Option<String>,
    pub span: Option<SourceSpan>,
    pub operation: Option<GraphOperation>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphReference {
    pub id: String,
    pub kind: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphOperandGroup {
    pub role: String,
    pub values: Vec<GraphReference>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphRegion {
    pub role: String,
    pub symbol: Option<String>,
    pub identity: bool,
    pub captures: Vec<GraphReference>,
    pub parameter_types: Vec<String>,
    pub result_types: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphResult {
    pub path: Vec<usize>,
    pub ty: String,
    pub destination: String,
    pub references: Vec<GraphReference>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GraphOperation {
    pub semantic_id: Option<String>,
    pub operand_groups: Vec<GraphOperandGroup>,
    pub regions: Vec<GraphRegion>,
    pub results: Vec<GraphResult>,
    pub soac_state: Option<GraphSoacState>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphSoacState {
    pub phase: String,
    pub variant: String,
    pub space: Vec<GraphSegExtent>,
    pub output_slots: Vec<usize>,
    pub resources: Vec<GraphResourceAccess>,
    pub filter_output: Option<GraphFilterOutput>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphSegExtent {
    pub variant: String,
    pub fixed: Option<u32>,
    pub value: Option<GraphReference>,
    pub binding: Option<GraphBinding>,
    pub resource: Option<String>,
    pub offset: Option<u32>,
    pub elem_bytes: Option<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphResourceAccess {
    pub binding: Option<GraphBinding>,
    pub resource: Option<String>,
    pub access: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphFilterOutput {
    pub variant: String,
    pub capacity: GraphFilterCapacity,
    pub ownership: Option<String>,
    pub backing: Option<GraphFilterBacking>,
    pub length: Option<GraphFilterLength>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphFilterCapacity {
    pub variant: String,
    pub ty: Option<String>,
    pub input: Option<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphFilterBacking {
    pub variant: String,
    pub binding: Option<GraphBinding>,
    pub resource: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphFilterLength {
    pub variant: String,
    pub binding: Option<GraphBinding>,
    pub resource: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphEdge {
    pub id: String,
    pub source: String,
    pub target: String,
    pub kind: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphTerminator {
    pub kind: String,
    pub values: Vec<String>,
    pub targets: Vec<String>,
    pub target_args: Vec<Vec<String>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphBlock {
    pub id: String,
    pub group: String,
    pub params: Vec<String>,
    pub operations: Vec<String>,
    pub terminator: GraphTerminator,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GraphSnapshot {
    pub resources: Vec<GraphResource>,
    pub materializations: Vec<GraphMaterialization>,
    pub groups: Vec<GraphGroup>,
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
    pub blocks: Vec<GraphBlock>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphMaterialization {
    pub id: String,
    pub variant: String,
    pub entry_group: String,
    pub entry_name: String,
    pub space: Vec<GraphSegExtent>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeRelation {
    pub before: Vec<String>,
    pub after: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InspectResult {
    pub success: bool,
    pub pass: String,
    pub before: Option<GraphSnapshot>,
    pub after: Option<GraphSnapshot>,
    pub relations: Vec<NodeRelation>,
    pub error: Option<VizError>,
}

impl InspectResult {
    fn error(pass: impl Into<String>, message: impl Into<String>, span: Option<Span>) -> Self {
        Self {
            success: false,
            pass: pass.into(),
            before: None,
            after: None,
            relations: Vec::new(),
            error: Some(VizError {
                message: message.into(),
                span: span.map(Into::into),
            }),
        }
    }
}

#[wasm_bindgen]
pub fn init_compiler() -> bool {
    console_error_panic_hook::set_once();
    PRELUDE_CACHE.with(|cache| {
        if cache.borrow().is_some() {
            return true;
        }
        let mut node_counter = NodeCounter::new();
        match ModuleManager::create_prelude(&mut node_counter) {
            Ok(prelude) => {
                *cache.borrow_mut() = Some(PreludeCache {
                    prelude,
                    start_node_counter: node_counter,
                });
                true
            }
            Err(error) => {
                web_sys::console::error_1(&format!("failed to initialize Wyn prelude: {error:?}").into());
                false
            }
        }
    })
}

#[wasm_bindgen]
pub fn inspect_optimize_semantic_operations(source: &str) -> JsValue {
    inspect_pass(source, InspectPass::OPTIMIZE_SEMANTIC_OPERATIONS)
}

#[wasm_bindgen]
pub fn inspect_pass(source: &str, pass: &str) -> JsValue {
    console_error_panic_hook::set_once();
    let Some(pass) = InspectPass::parse(pass) else {
        return serde_wasm_bindgen::to_value(&InspectResult::error(
            pass,
            format!("unknown EGIR pass `{pass}`"),
            None,
        ))
        .expect("serialize unknown EGIR pass error");
    };
    let result = inspect_pass_impl(source, pass);
    serde_wasm_bindgen::to_value(&result).unwrap_or_else(|error| {
        serde_wasm_bindgen::to_value(&InspectResult::error(
            pass.id(),
            format!("failed to serialize EGIR snapshots: {error}"),
            None,
        ))
        .expect("serialize fallback EGIR visualization error")
    })
}

fn compiler_init() -> Option<(NodeCounter, ModuleManager)> {
    PRELUDE_CACHE.with(|cache| {
        let cache = cache.borrow();
        let cached = cache.as_ref()?;
        Some(wyn_core::init_compiler_from_prelude(
            cached.prelude.clone(),
            cached.start_node_counter.clone(),
        ))
    })
}

fn compiler_error(pass: InspectPass, error: CompilerError) -> InspectResult {
    let span = error.span();
    InspectResult::error(pass.id(), format_compiler_error(&error), span)
}

fn format_compiler_error(error: &CompilerError) -> String {
    match error {
        CompilerError::ParseError(message, _) => format!("Parse error: {message}"),
        CompilerError::TypeError(message, _) => format!("Type error: {message}"),
        CompilerError::UndefinedVariable(name, _) => format!("Undefined variable: `{name}`"),
        CompilerError::AliasError(message, _) => format!("Alias error: {message}"),
        CompilerError::SpirvError(message, _) => format!("SPIR-V error: {message}"),
        CompilerError::WgslError(message, _) => format!("WGSL error: {message}"),
        CompilerError::ModuleError(message, _) => format!("Module error: {message}"),
        CompilerError::FlatteningError(message, _) => format!("Flatten error: {message}"),
        CompilerError::IoError(error) => format!("I/O error: {error}"),
        CompilerError::SpirvBuilderError(message) => format!("SPIR-V builder error: {message}"),
        CompilerError::TypeHole(message) => format!("Type hole: {message}"),
    }
}

#[cfg(test)]
fn inspect_impl(source: &str) -> InspectResult {
    inspect_pass_impl(source, InspectPass::OptimizeSemanticOperations)
}

fn inspect_pass_impl(source: &str, pass: InspectPass) -> InspectResult {
    if !init_compiler() {
        return InspectResult::error(pass.id(), "failed to initialize the Wyn compiler", None);
    }
    let Some((node_counter, module_manager)) = compiler_init() else {
        return InspectResult::error(pass.id(), "compiler cache is unavailable", None);
    };

    macro_rules! try_compiler {
        ($expression:expr) => {
            match $expression {
                Ok(value) => value,
                Err(error) => return compiler_error(pass, error),
            }
        };
    }

    let program = try_compiler!(wyn_core::parser::parse(source, node_counter, module_manager));
    let program = try_compiler!(wyn_core::resolve_imports::resolve_imports(
        program,
        std::path::Path::new(".")
    ));
    let program = try_compiler!(wyn_core::elaborate_modules::elaborate_modules(program));
    let program = wyn_core::name_resolution::resolve_names(program);
    let program = try_compiler!(wyn_core::resolve_resources::resolve_resources(program));
    let program = wyn_core::ast_const_fold::fold_constants(program);
    let program = wyn_core::resolve_placeholders::resolve_type_placeholders(program);
    let program = try_compiler!(wyn_core::resolve_opens::resolve_opens(program));
    let program = try_compiler!(wyn_core::types::run::type_check(program));
    let program = try_compiler!(wyn_core::ast_type_holes::reject_type_holes(program));
    let program = try_compiler!(wyn_core::tlc::lower_from_ast(program));
    let program = try_compiler!(wyn_core::tlc::pin_entry_buffers(program));
    let program = try_compiler!(wyn_core::tlc::validate_ownership(program));
    let program = wyn_core::tlc::partial_eval(program);
    let program = wyn_core::tlc::normalize_soacs(program);
    let program = wyn_core::tlc::monomorphize(program);
    let program = wyn_core::tlc::rep_specialize(program);
    let program = wyn_core::tlc::inline_small(program);
    let program = wyn_core::tlc::force_inline_soac_helpers(program);
    let program = wyn_core::tlc::renormalize_inlined_soa(program);
    let program = wyn_core::tlc::canonicalize_conditional_producers(program);
    let program = wyn_core::tlc::normalize_soacs_to_anf(program);
    let program = wyn_core::tlc::float_runtime_index_nested_producers(program);
    let program = wyn_core::tlc::defunctionalize(program);
    let program = wyn_core::tlc::fold_generated_lambdas(program);
    let program = wyn_core::tlc::apply_ownership(program);
    let program = wyn_core::tlc::filter_reachable(program);
    let program = wyn_core::tlc::infer_input_slice_bounds(program);
    let program = match wyn_core::to_egraph(program) {
        Ok(program) => program,
        Err(error) => {
            return InspectResult::error(pass.id(), format!("EGIR conversion error: {error:?}"), None)
        }
    };

    if pass == InspectPass::ReifySoacs {
        let before = snapshot_program(&program);
        let program = wyn_core::egir::reify_soacs(program);
        let after = snapshot_program(&program);
        return InspectResult {
            success: true,
            pass: pass.id().to_string(),
            before: Some(before),
            after: Some(after),
            relations: Vec::new(),
            error: None,
        };
    }

    let segmented = wyn_core::egir::reify_soacs(program);
    let before = snapshot_program(&segmented);
    let (semantic_operations_optimized, trace) =
        wyn_core::egir::optimize_semantic_operations_with_trace(segmented);
    if pass == InspectPass::OptimizeSemanticOperations {
        let after = snapshot_program(&semantic_operations_optimized);
        let relations = trace
            .relations
            .into_iter()
            .map(|relation| NodeRelation {
                before: relation.before.into_iter().map(operation_node_id).collect(),
                after: relation.after.into_iter().map(operation_node_id).collect(),
            })
            .collect();
        return InspectResult {
            success: true,
            pass: pass.id().to_string(),
            before: Some(before),
            after: Some(after),
            relations,
            error: None,
        };
    }

    let optimized = wyn_core::egir::lift_stage_uniform_values(semantic_operations_optimized);
    if pass == InspectPass::PlanLogicalResources {
        let before = snapshot_program(&optimized);
        let allocated = match wyn_core::egir::plan_logical_resources(optimized) {
            Ok(program) => program,
            Err(error) => {
                return InspectResult::error(
                    pass.id(),
                    format!("EGIR logical-resource planning error: {error:?}"),
                    None,
                )
            }
        };
        return InspectResult {
            success: true,
            pass: pass.id().to_string(),
            before: Some(before),
            after: Some(snapshot_allocated_program(&allocated)),
            relations: Vec::new(),
            error: None,
        };
    }
    unreachable!("all inspector passes return at their checkpoint")
}

fn operation_node_id(id: SemanticOpId) -> String {
    match id.implementation_slot() {
        Some(slot) => format!("op:{}:{slot}", id.source_index()),
        None => format!("op:{}", id.source_index()),
    }
}

trait SnapshotResource: WynGraphResource {
    fn graph_reference(&self) -> (Option<GraphBinding>, Option<String>);
}

impl SnapshotResource for BindingRef {
    fn graph_reference(&self) -> (Option<GraphBinding>, Option<String>) {
        (Some(graph_binding(*self)), None)
    }
}

impl SnapshotResource for SemanticResourceRef {
    fn graph_reference(&self) -> (Option<GraphBinding>, Option<String>) {
        (None, Some(resource_name(*self)))
    }
}

trait SnapshotPhase: WynSoacPhase {
    fn graph_resource(resource: &Self::Resource) -> (Option<GraphBinding>, Option<String>);

    fn soac_node_id(id: &Self::SoacId, group: &str, block: wyn_core::flow::BlockId, index: usize)
        -> String;

    fn soac_detail(id: &Self::SoacId, soac: &Soac<Self>) -> String;

    fn semantic_id(id: &Self::SoacId) -> Option<String>;

    fn screma_state(group: &str, op: &screma::Op<Self>) -> GraphSoacState;

    fn filter_state(group: &str, op: &filter::Op<Self>) -> GraphSoacState;

    fn hist_state(group: &str, op: &hist::Op<Self>) -> GraphSoacState;
}

impl<R: SnapshotResource> SnapshotPhase for Raw<R> {
    fn graph_resource(resource: &Self::Resource) -> (Option<GraphBinding>, Option<String>) {
        resource.graph_reference()
    }

    fn soac_node_id(
        _id: &Self::SoacId,
        group: &str,
        block: wyn_core::flow::BlockId,
        index: usize,
    ) -> String {
        format!("{group}/effect/{block:?}/{index}")
    }

    fn soac_detail(_id: &Self::SoacId, soac: &Soac<Self>) -> String {
        format!("{soac:#?}")
    }

    fn semantic_id(_id: &Self::SoacId) -> Option<String> {
        None
    }

    fn screma_state(_group: &str, _op: &screma::Op<Self>) -> GraphSoacState {
        graph_raw_soac_state(None)
    }

    fn filter_state(_group: &str, op: &filter::Op<Self>) -> GraphSoacState {
        let output = match &op.state.output {
            filter::RawOutput::Local { capacity, ownership } => {
                graph_local_filter_output(capacity, *ownership)
            }
            filter::RawOutput::Runtime { capacity } => {
                graph_runtime_filter_output(graph_runtime_capacity(*capacity), None, None)
            }
        };
        graph_raw_soac_state(Some(output))
    }

    fn hist_state(_group: &str, _op: &hist::Op<Self>) -> GraphSoacState {
        graph_raw_soac_state(None)
    }
}

fn graph_raw_soac_state(filter_output: Option<GraphFilterOutput>) -> GraphSoacState {
    GraphSoacState {
        phase: "raw".to_string(),
        variant: "raw".to_string(),
        space: Vec::new(),
        output_slots: Vec::new(),
        resources: Vec::new(),
        filter_output,
    }
}

fn graph_semantic_soac_state<P: SnapshotPhase>(
    group: &str,
    variant: &str,
    space: Option<&SegSpace<P::Resource>>,
    output_slots: &[wyn_core::egir::program::OutputSlotId],
    resources: &[SegResourceAccess<P::Resource>],
    filter_output: Option<GraphFilterOutput>,
) -> GraphSoacState {
    GraphSoacState {
        phase: "semantic".to_string(),
        variant: variant.to_string(),
        space: space.map_or_else(Vec::new, |space| graph_seg_space::<P>(group, space)),
        output_slots: output_slots.iter().map(|slot| slot.0).collect(),
        resources: resources.iter().map(graph_resource_access::<P>).collect(),
        filter_output,
    }
}

fn graph_seg_space<P: SnapshotPhase>(group: &str, space: &SegSpace<P::Resource>) -> Vec<GraphSegExtent> {
    space
        .dims()
        .iter()
        .map(|extent| match extent {
            SegExtent::Fixed(value) => GraphSegExtent {
                variant: "fixed".to_string(),
                fixed: Some(*value),
                value: None,
                binding: None,
                resource: None,
                offset: None,
                elem_bytes: None,
            },
            SegExtent::PushConstant { node, offset } => GraphSegExtent {
                variant: "push_constant".to_string(),
                fixed: None,
                value: Some(value_reference(group, *node)),
                binding: None,
                resource: None,
                offset: Some(*offset),
                elem_bytes: None,
            },
            SegExtent::ResourceLength {
                view,
                resource,
                elem_bytes,
            } => {
                let (binding, resource) = P::graph_resource(resource);
                GraphSegExtent {
                    variant: "resource_length".to_string(),
                    fixed: None,
                    value: Some(view_reference(group, *view)),
                    binding,
                    resource,
                    offset: None,
                    elem_bytes: Some(*elem_bytes),
                }
            }
            SegExtent::Value(value) => GraphSegExtent {
                variant: "value".to_string(),
                fixed: None,
                value: Some(value_reference(group, *value)),
                binding: None,
                resource: None,
                offset: None,
                elem_bytes: None,
            },
        })
        .collect()
}

fn graph_resource_access<P: SnapshotPhase>(access: &SegResourceAccess<P::Resource>) -> GraphResourceAccess {
    let (binding, resource) = P::graph_resource(&access.resource);
    GraphResourceAccess {
        binding,
        resource,
        access: match access.access {
            ResourceAccess::Read => "read",
            ResourceAccess::Write => "write",
            ResourceAccess::ReadWrite => "read_write",
        }
        .to_string(),
    }
}

impl<R: SnapshotResource> SnapshotPhase for Semantic<R> {
    fn graph_resource(resource: &Self::Resource) -> (Option<GraphBinding>, Option<String>) {
        resource.graph_reference()
    }

    fn soac_node_id(
        id: &Self::SoacId,
        _group: &str,
        _block: wyn_core::flow::BlockId,
        _index: usize,
    ) -> String {
        operation_node_id(*id)
    }

    fn soac_detail(id: &Self::SoacId, soac: &Soac<Self>) -> String {
        format!("semantic op {}\n\n{soac:#?}", id.source_index())
    }

    fn semantic_id(id: &Self::SoacId) -> Option<String> {
        Some(operation_node_id(*id))
    }

    fn screma_state(group: &str, op: &screma::Op<Self>) -> GraphSoacState {
        match &op.state {
            screma::SemanticState::Serial => {
                graph_semantic_soac_state::<Self>(group, "serial", None, &[], &[], None)
            }
            screma::SemanticState::Segmented {
                space,
                output_slots,
                resources,
            } => graph_semantic_soac_state::<Self>(
                group,
                "segmented",
                Some(space),
                output_slots,
                resources,
                None,
            ),
        }
    }

    fn filter_state(group: &str, op: &filter::Op<Self>) -> GraphSoacState {
        graph_semantic_soac_state::<Self>(
            group,
            "segmented",
            Some(&op.state.space),
            &op.state.output_slots,
            &op.state.resources,
            Some(graph_filter_output::<Self>(&op.state.output)),
        )
    }

    fn hist_state(group: &str, op: &hist::Op<Self>) -> GraphSoacState {
        match &op.state {
            hist::SemanticState::Serial => {
                graph_semantic_soac_state::<Self>(group, "serial", None, &[], &[], None)
            }
            hist::SemanticState::Segmented(space) => {
                graph_semantic_soac_state::<Self>(group, "segmented", Some(space), &[], &[], None)
            }
        }
    }
}

fn graph_local_filter_output(
    capacity: &wyn_core::types::Type,
    ownership: wyn_core::egir::types::SoacOwnership,
) -> GraphFilterOutput {
    GraphFilterOutput {
        variant: "local".to_string(),
        capacity: GraphFilterCapacity {
            variant: "type".to_string(),
            ty: Some(wyn_core::diags::format_type(capacity)),
            input: None,
        },
        ownership: Some(
            match ownership {
                wyn_core::egir::types::SoacOwnership::Fresh => "fresh",
                wyn_core::egir::types::SoacOwnership::UniqueInput => "unique_input",
            }
            .to_string(),
        ),
        backing: None,
        length: None,
    }
}

fn graph_runtime_capacity(capacity: filter::RuntimeCapacity) -> GraphFilterCapacity {
    match capacity {
        filter::RuntimeCapacity::LikeInput { input } => GraphFilterCapacity {
            variant: "like_input".to_string(),
            ty: None,
            input: Some(input.0),
        },
    }
}

fn graph_runtime_filter_output(
    capacity: GraphFilterCapacity,
    backing: Option<GraphFilterBacking>,
    length: Option<GraphFilterLength>,
) -> GraphFilterOutput {
    GraphFilterOutput {
        variant: "runtime".to_string(),
        capacity,
        ownership: None,
        backing,
        length,
    }
}

fn graph_filter_output<P: SnapshotPhase>(output: &filter::Output<P::Resource>) -> GraphFilterOutput {
    match output {
        filter::Output::Local { capacity, ownership } => graph_local_filter_output(capacity, *ownership),
        filter::Output::Runtime(runtime) => graph_runtime_filter_output(
            graph_runtime_capacity(runtime.capacity),
            Some(match &runtime.backing {
                filter::RuntimeBacking::Deferred => GraphFilterBacking {
                    variant: "deferred".to_string(),
                    binding: None,
                    resource: None,
                },
                filter::RuntimeBacking::Bound(reference) => {
                    let (binding, resource) = P::graph_resource(reference);
                    GraphFilterBacking {
                        variant: "bound".to_string(),
                        binding,
                        resource,
                    }
                }
            }),
            Some(match &runtime.length {
                filter::RuntimeLength::Implicit => GraphFilterLength {
                    variant: "implicit".to_string(),
                    binding: None,
                    resource: None,
                },
                filter::RuntimeLength::Stored(reference) => {
                    let (binding, resource) = P::graph_resource(reference);
                    GraphFilterLength {
                        variant: "stored".to_string(),
                        binding,
                        resource,
                    }
                }
            }),
        ),
    }
}

fn graph_binding(binding: wyn_core::BindingRef) -> GraphBinding {
    GraphBinding {
        set: binding.set,
        binding: binding.binding,
    }
}

fn resource_id_name(resource: ResourceId) -> String {
    format!("$r{}", resource.index())
}

fn resource_name(resource: SemanticResourceRef) -> String {
    resource_id_name(resource.0)
}

fn graph_logical_size(size: &LogicalSize) -> GraphSize {
    match size {
        LogicalSize::FixedBytes(bytes) => GraphSize {
            variant: "fixed_bytes".to_string(),
            bytes: Some(*bytes),
            binding: None,
            resource: None,
            elem_bytes: None,
            src_elem_bytes: None,
        },
        LogicalSize::LikeResource {
            resource,
            elem_bytes,
            src_elem_bytes,
        } => GraphSize {
            variant: "like_resource".to_string(),
            bytes: None,
            binding: None,
            resource: Some(resource_id_name(*resource)),
            elem_bytes: Some(*elem_bytes),
            src_elem_bytes: Some(*src_elem_bytes),
        },
        LogicalSize::SameAsDispatch { elem_bytes } => GraphSize {
            variant: "same_as_dispatch".to_string(),
            bytes: None,
            binding: None,
            resource: None,
            elem_bytes: Some(*elem_bytes),
            src_elem_bytes: None,
        },
        LogicalSize::Unspecified => GraphSize {
            variant: "unspecified".to_string(),
            bytes: None,
            binding: None,
            resource: None,
            elem_bytes: None,
            src_elem_bytes: None,
        },
    }
}

fn storage_role(role: wyn_core::interface::StorageRole) -> String {
    match role {
        wyn_core::interface::StorageRole::Input => "input",
        wyn_core::interface::StorageRole::Output => "output",
        wyn_core::interface::StorageRole::InputOutput => "input_output",
        wyn_core::interface::StorageRole::Intermediate => "intermediate",
    }
    .to_string()
}

fn compiler_resource_kind(kind: CompilerResourceKind) -> String {
    match kind {
        CompilerResourceKind::GatherHandoff => "gather_handoff",
        CompilerResourceKind::ReducePartial => "reduce_partial",
        CompilerResourceKind::ScanBlockSums => "scan_block_sums",
        CompilerResourceKind::ScanBlockOffsets => "scan_block_offsets",
        CompilerResourceKind::ScanPrefixes => "scan_prefixes",
        CompilerResourceKind::FilterScratch => "filter_scratch",
        CompilerResourceKind::FilterLenCell => "filter_len_cell",
        CompilerResourceKind::FilterFlags => "filter_flags",
        CompilerResourceKind::FilterOffsets => "filter_offsets",
        CompilerResourceKind::FilterScanBlockSums => "filter_scan_block_sums",
        CompilerResourceKind::FilterScanBlockOffsets => "filter_scan_block_offsets",
        CompilerResourceKind::BucketCounts => "bucket_counts",
        CompilerResourceKind::BucketOverflow => "bucket_overflow",
        CompilerResourceKind::ScalarHandoff => "scalar_handoff",
        CompilerResourceKind::MultiConsumerArray => "multi_consumer_array",
    }
    .to_string()
}

fn graph_buffer_len(length: &wyn_core::pipeline_descriptor::BufferLen) -> GraphSize {
    use wyn_core::pipeline_descriptor::BufferLen;
    match length {
        BufferLen::Fixed { bytes } => GraphSize {
            variant: "fixed_bytes".to_string(),
            bytes: Some(*bytes),
            binding: None,
            resource: None,
            elem_bytes: None,
            src_elem_bytes: None,
        },
        BufferLen::LikeInput {
            set,
            binding,
            elem_bytes,
            src_elem_bytes,
        } => GraphSize {
            variant: "like_input".to_string(),
            bytes: None,
            binding: Some(GraphBinding {
                set: *set,
                binding: *binding,
            }),
            resource: None,
            elem_bytes: Some(*elem_bytes),
            src_elem_bytes: Some(*src_elem_bytes),
        },
        BufferLen::SameAsDispatch { elem_bytes } => GraphSize {
            variant: "same_as_dispatch".to_string(),
            bytes: None,
            binding: None,
            resource: None,
            elem_bytes: Some(*elem_bytes),
            src_elem_bytes: None,
        },
    }
}

fn graph_output_kind(kind: &wyn_core::interface::EntryOutputKind) -> GraphOutputKind {
    use wyn_core::interface::{BindingExposure, EntryOutputDestination, EntryOutputKind};
    match kind {
        EntryOutputKind::Value { destination } => GraphOutputKind {
            variant: "value".to_string(),
            destination: Some(match destination {
                EntryOutputDestination::Plain => "plain".to_string(),
                EntryOutputDestination::BuiltIn(value) => format!("builtin({value:?})"),
                EntryOutputDestination::Location(value) => format!("location({value})"),
                EntryOutputDestination::Target(value) => format!("target({value:?})"),
            }),
            exposure: None,
            binding: None,
            length: None,
        },
        EntryOutputKind::Storage { exposure, length } => {
            let (exposure, binding) = match exposure {
                BindingExposure::Host(binding) => ("host", Some(graph_binding(*binding))),
                BindingExposure::Internal => ("internal", None),
            };
            GraphOutputKind {
                variant: "storage".to_string(),
                destination: None,
                exposure: Some(exposure.to_string()),
                binding,
                length: length.as_ref().map(graph_buffer_len),
            }
        }
    }
}

fn snapshot_program<Tag, P>(
    program: &wyn_core::egir::program::Program<
        Tag,
        ProgramFamily<P, NoStorageDeclaration, RealizedOutputRoute, SemanticProgramData>,
        RewriteGlobal,
    >,
) -> GraphSnapshot
where
    P: SnapshotPhase,
{
    let mut snapshot = GraphSnapshot::default();
    let region_names = program
        .functions
        .iter()
        .map(|function| (function.region, function.name.clone()))
        .collect::<HashMap<_, _>>();
    for (index, entry) in program.entry_points.iter().enumerate() {
        let group = format!("entry:{index}");
        snapshot.groups.push(GraphGroup {
            id: group.clone(),
            label: format!("entry {}", entry.name),
            kind: "entry".to_string(),
            outputs: entry
                .outputs
                .iter()
                .enumerate()
                .map(|(slot, output)| GraphOutput {
                    slot,
                    ty: wyn_core::diags::format_type(&output.ty),
                    binding: output.resource.as_ref().and_then(|resource| P::graph_resource(resource).0),
                    resource: output.resource.as_ref().and_then(|resource| P::graph_resource(resource).1),
                    kind: graph_output_kind(&output.kind),
                    routes: output
                        .routes
                        .iter()
                        .map(|route| GraphOutputRoute {
                            source_block: format!("{group}/block/{:?}", route.source.block),
                            source_value: value_node_id(&group, route.source.value),
                            writers: route
                                .writers
                                .iter()
                                .map(|writer| match writer {
                                    OutputWriter::Value(value) => GraphOutputWriter {
                                        kind: "value".to_string(),
                                        id: value_node_id(&group, *value),
                                    },
                                    OutputWriter::Effect(effect) => GraphOutputWriter {
                                        kind: "effect".to_string(),
                                        id: effect.to_string(),
                                    },
                                })
                                .collect(),
                        })
                        .collect(),
                })
                .collect(),
            resource_declarations: Vec::new(),
        });
        snapshot_graph(&mut snapshot, &group, &entry.graph, &region_names);
    }
    for function in &program.functions {
        let group = format!("function:{:?}", function.region);
        snapshot.groups.push(GraphGroup {
            id: group.clone(),
            label: format!("fn {}", function.name),
            kind: "function".to_string(),
            outputs: Vec::new(),
            resource_declarations: Vec::new(),
        });
        snapshot_graph(&mut snapshot, &group, &function.graph, &region_names);
    }
    for (index, constant) in program.constants.iter().enumerate() {
        let group = format!("constant:{index}");
        snapshot.groups.push(GraphGroup {
            id: group.clone(),
            label: format!("const {}", constant.name),
            kind: "constant".to_string(),
            outputs: Vec::new(),
            resource_declarations: Vec::new(),
        });
        snapshot_graph(&mut snapshot, &group, &constant.graph, &region_names);
    }
    snapshot
}

fn snapshot_allocated_program(program: &wyn_core::egir::ResourcesAllocated) -> GraphSnapshot {
    let mut snapshot = GraphSnapshot::default();
    snapshot.resources = program
        .data
        .core
        .resources
        .iter()
        .map(|resource| {
            let origin = match &resource.origin {
                ResourceOrigin::Host(host) => GraphResourceOrigin {
                    variant: "host".to_string(),
                    binding: Some(graph_binding(host.binding)),
                    name: host.name.clone(),
                    compiler_kind: None,
                    owner: None,
                    slot: None,
                },
                ResourceOrigin::Compiler(compiler) => GraphResourceOrigin {
                    variant: "compiler".to_string(),
                    binding: None,
                    name: None,
                    compiler_kind: Some(compiler_resource_kind(compiler.kind)),
                    owner: compiler.owner.map(operation_node_id),
                    slot: Some(compiler.slot),
                },
            };
            GraphResource {
                id: resource_id_name(resource.id()),
                elem_ty: wyn_core::diags::format_type(&resource.elem_ty),
                origin,
                size: graph_logical_size(&resource.size),
            }
        })
        .collect();
    let region_names = program
        .functions
        .iter()
        .map(|function| (function.region, function.name.clone()))
        .collect::<HashMap<_, _>>();

    for (index, entry) in program.entry_points.iter().enumerate() {
        snapshot_allocated_entry(
            &mut snapshot,
            format!("entry:{index}"),
            "entry",
            entry,
            program.logical_resources(),
            &region_names,
        );
    }
    for id in program.data.materializations.ids() {
        let requirement = &program.data.materializations[id];
        let entry = requirement.entry();
        let group = format!("materialization:{}", id.0);
        snapshot.materializations.push(GraphMaterialization {
            id: format!("@m{}", id.0),
            variant: match requirement {
                MaterializationRequirement::SharedArray { .. } => "shared_array",
                MaterializationRequirement::Gather { .. } => "gather",
                MaterializationRequirement::RuntimeArray { .. } => "runtime_array",
                MaterializationRequirement::Scalar { .. } => "scalar",
            }
            .to_string(),
            entry_group: group.clone(),
            entry_name: entry.name.clone(),
            space: requirement.space().map_or_else(Vec::new, |space| {
                graph_seg_space::<Semantic<SemanticResourceRef>>(&group, space)
            }),
        });
        snapshot_allocated_entry(
            &mut snapshot,
            group,
            "materialization",
            entry,
            program.logical_resources(),
            &region_names,
        );
    }
    for function in &program.functions {
        let group = format!("function:{:?}", function.region);
        snapshot.groups.push(GraphGroup {
            id: group.clone(),
            label: format!("fn {}", function.name),
            kind: "function".to_string(),
            outputs: Vec::new(),
            resource_declarations: Vec::new(),
        });
        snapshot_graph(&mut snapshot, &group, &function.graph, &region_names);
    }
    for (index, constant) in program.constants.iter().enumerate() {
        let group = format!("constant:{index}");
        snapshot.groups.push(GraphGroup {
            id: group.clone(),
            label: format!("const {}", constant.name),
            kind: "constant".to_string(),
            outputs: Vec::new(),
            resource_declarations: Vec::new(),
        });
        snapshot_graph(&mut snapshot, &group, &constant.graph, &region_names);
    }
    snapshot
}

fn snapshot_allocated_entry(
    snapshot: &mut GraphSnapshot,
    group: String,
    kind: &str,
    entry: &wyn_core::egir::program::AllocatedEntry,
    resources: &[LogicalResource],
    region_names: &HashMap<FunctionId, String>,
) {
    snapshot.groups.push(GraphGroup {
        id: group.clone(),
        label: format!("{kind} {}", entry.name),
        kind: kind.to_string(),
        outputs: entry
            .outputs
            .iter()
            .enumerate()
            .map(|(slot, output)| GraphOutput {
                slot,
                ty: wyn_core::diags::format_type(&output.ty),
                binding: None,
                resource: output.resource.map(resource_name),
                kind: graph_output_kind(&output.kind),
                routes: output
                    .routes
                    .iter()
                    .map(|route| GraphOutputRoute {
                        source_block: format!("{group}/block/{:?}", route.source.block),
                        source_value: value_node_id(&group, route.source.value),
                        writers: route
                            .writers
                            .iter()
                            .map(|writer| match writer {
                                OutputWriter::Value(value) => GraphOutputWriter {
                                    kind: "value".to_string(),
                                    id: value_node_id(&group, *value),
                                },
                                OutputWriter::Effect(effect) => GraphOutputWriter {
                                    kind: "effect".to_string(),
                                    id: effect.to_string(),
                                },
                            })
                            .collect(),
                    })
                    .collect(),
            })
            .collect(),
        resource_declarations: entry
            .resource_declarations
            .iter()
            .map(|declaration| {
                let resource = &resources[declaration.resource.0.index()];
                GraphResourceDeclaration {
                    resource: resource_name(declaration.resource),
                    role: storage_role(declaration.role),
                    elem_ty: wyn_core::diags::format_type(&resource.elem_ty),
                    size: graph_logical_size(&resource.size),
                }
            })
            .collect(),
    });
    snapshot_graph(snapshot, &group, &entry.graph, region_names);
}

fn snapshot_graph<P: SnapshotPhase>(
    snapshot: &mut GraphSnapshot,
    group: &str,
    graph: &wyn_core::egir::types::EGraph<P>,
    region_names: &HashMap<FunctionId, String>,
) {
    for (value_id, value) in graph.values() {
        let id = value_node_id(group, value_id);
        let (label, variant) = value_label(value.kind());
        snapshot.nodes.push(GraphNode {
            id: id.clone(),
            group: group.to_string(),
            label,
            category: "value".to_string(),
            variant,
            detail: format!(
                "{:#?}\n\ntype: {}",
                value.kind(),
                wyn_core::diags::format_type(value.ty())
            ),
            ty: Some(wyn_core::diags::format_type(value.ty())),
            span: value.span().map(Into::into),
            operation: None,
        });
        for dependency in graph.value_dependencies(value_id) {
            if graph.values().contains_key(dependency) {
                push_edge(snapshot, value_node_id(group, dependency), id.clone(), "value");
            }
        }
        if let Some(alias) = value.alias() {
            if graph.values().contains_key(alias) {
                push_edge(snapshot, value_node_id(group, alias), id.clone(), "equivalent");
            }
        }
    }

    for (place_id, place) in graph.places() {
        let id = place_node_id(group, place_id);
        let (label, variant, operation) = place_display(group, place.op());
        snapshot.nodes.push(GraphNode {
            id,
            group: group.to_string(),
            label,
            category: "place".to_string(),
            variant,
            detail: format!("{:#?}\n\ntype: {:#?}", place.op(), place.ty()),
            ty: Some(wyn_core::diags::format_type(&place.ty().pointee)),
            span: place.span().map(Into::into),
            operation: Some(operation),
        });
    }

    for (block_id, block) in &graph.skeleton.blocks {
        let block_node = format!("{group}/block/{block_id:?}");
        snapshot.nodes.push(GraphNode {
            id: block_node.clone(),
            group: group.to_string(),
            label: format!("block {block_id:?}"),
            category: "block".to_string(),
            variant: "block".to_string(),
            detail: format!("{:#?}", block.term),
            ty: None,
            span: None,
            operation: None,
        });

        let mut operations = Vec::new();
        let mut previous_effect = None;
        for (index, effect) in block.side_effects.iter().enumerate() {
            let display = effect_display(group, block_id, index, effect, graph, region_names);
            let effect_id = display.id.clone();
            operations.push(effect_id.clone());
            snapshot.nodes.push(GraphNode {
                id: effect_id.clone(),
                group: group.to_string(),
                label: display.label,
                category: "operation".to_string(),
                variant: display.variant,
                detail: display.detail,
                ty: None,
                span: effect.span().map(Into::into),
                operation: display.operation,
            });
            push_edge(snapshot, block_node.clone(), effect_id.clone(), "block");
            if let Some(previous) = previous_effect.replace(effect_id.clone()) {
                push_edge(snapshot, previous, effect_id.clone(), "sequence");
            }
            for dependency in graph.effect_boundary_value_dependencies(effect) {
                if graph.values().contains_key(dependency) {
                    push_edge(
                        snapshot,
                        value_node_id(group, dependency),
                        effect_id.clone(),
                        "operand",
                    );
                }
            }
            if let Some(result) = graph.effect_result_binding(effect) {
                for value in result.values() {
                    if graph.values().contains_key(value) {
                        push_edge(snapshot, effect_id.clone(), value_node_id(group, value), "result");
                    }
                }
                for place in result.places() {
                    if graph.places().contains_key(place) {
                        push_edge(snapshot, effect_id.clone(), place_node_id(group, place), "result");
                    }
                }
            }
            if let SideEffectKind::Effect(wyn_core::egir::types::EffectOp::Alloca { result }) =
                effect.kind()
            {
                if graph.places().contains_key(*result) {
                    push_edge(
                        snapshot,
                        effect_id.clone(),
                        place_node_id(group, *result),
                        "result",
                    );
                }
            }
        }

        for value in block.term.referenced_nodes() {
            if graph.values().contains_key(value) {
                push_edge(
                    snapshot,
                    value_node_id(group, value),
                    block_node.clone(),
                    "terminator",
                );
            }
        }

        snapshot.blocks.push(GraphBlock {
            id: block_node.clone(),
            group: group.to_string(),
            params: block.params.iter().map(|parameter| value_node_id(group, parameter.value())).collect(),
            operations,
            terminator: graph_terminator(group, &block.term),
        });

        match &block.term {
            wyn_core::flow::Terminator::Branch { target, .. } => push_edge(
                snapshot,
                block_node.clone(),
                format!("{group}/block/{target:?}"),
                "control",
            ),
            wyn_core::flow::Terminator::CondBranch {
                then_target,
                else_target,
                ..
            } => {
                push_edge(
                    snapshot,
                    block_node.clone(),
                    format!("{group}/block/{then_target:?}"),
                    "control",
                );
                push_edge(
                    snapshot,
                    block_node,
                    format!("{group}/block/{else_target:?}"),
                    "control",
                );
            }
            wyn_core::flow::Terminator::Return(_) | wyn_core::flow::Terminator::Unreachable => {}
        }
    }
}

fn graph_terminator(
    group: &str,
    terminator: &wyn_core::egir::types::SkeletonTerminator,
) -> GraphTerminator {
    match terminator {
        wyn_core::flow::Terminator::Return(result) => GraphTerminator {
            kind: "return".to_string(),
            values: result
                .iter()
                .flat_map(|binding| binding.values())
                .map(|value| value_node_id(group, value))
                .collect(),
            targets: Vec::new(),
            target_args: Vec::new(),
        },
        wyn_core::flow::Terminator::Branch { target, args } => GraphTerminator {
            kind: "branch".to_string(),
            values: Vec::new(),
            targets: vec![format!("{group}/block/{target:?}")],
            target_args: vec![args.iter().map(|value| value_node_id(group, value.value())).collect()],
        },
        wyn_core::flow::Terminator::CondBranch {
            cond,
            then_target,
            then_args,
            else_target,
            else_args,
        } => GraphTerminator {
            kind: "cond_branch".to_string(),
            values: vec![value_node_id(group, *cond)],
            targets: vec![
                format!("{group}/block/{then_target:?}"),
                format!("{group}/block/{else_target:?}"),
            ],
            target_args: vec![
                then_args.iter().map(|value| value_node_id(group, value.value())).collect(),
                else_args.iter().map(|value| value_node_id(group, value.value())).collect(),
            ],
        },
        wyn_core::flow::Terminator::Unreachable => GraphTerminator {
            kind: "unreachable".to_string(),
            values: Vec::new(),
            targets: Vec::new(),
            target_args: Vec::new(),
        },
    }
}

fn value_node_id(group: &str, value: wyn_core::egir::types::ValueId) -> String {
    format!("{group}/value/{value:?}")
}

fn place_node_id(group: &str, place: wyn_core::egir::types::PlaceId) -> String {
    format!("{group}/place/{place:?}")
}

fn place_display(group: &str, op: &PlaceOp) -> (String, String, GraphOperation) {
    let operand_groups = match op {
        PlaceOp::Parameter { .. } | PlaceOp::AllocaResult | PlaceOp::OutputSlot { .. } => Vec::new(),
        PlaceOp::View { view } => vec![GraphOperandGroup {
            role: "view".to_string(),
            values: vec![view_reference(group, *view)],
        }],
        PlaceOp::Index { base, index } => vec![
            GraphOperandGroup {
                role: "base".to_string(),
                values: vec![place_reference(group, *base)],
            },
            GraphOperandGroup {
                role: "index".to_string(),
                values: vec![value_reference(group, *index)],
            },
        ],
        PlaceOp::Slice { base, start, length } => vec![
            GraphOperandGroup {
                role: "base".to_string(),
                values: vec![place_reference(group, *base)],
            },
            GraphOperandGroup {
                role: "start".to_string(),
                values: vec![value_reference(group, *start)],
            },
            GraphOperandGroup {
                role: "length".to_string(),
                values: vec![value_reference(group, *length)],
            },
        ],
        PlaceOp::ViewIndex { view, index } => vec![
            GraphOperandGroup {
                role: "view".to_string(),
                values: vec![view_reference(group, *view)],
            },
            GraphOperandGroup {
                role: "index".to_string(),
                values: vec![value_reference(group, *index)],
            },
        ],
    };
    let (label, variant) = match op {
        PlaceOp::Parameter { parameter } => (parameter_label(parameter), "parameter"),
        PlaceOp::View { .. } => ("place.view".to_string(), "view"),
        PlaceOp::AllocaResult => ("place.alloca_result".to_string(), "alloca-result"),
        PlaceOp::Index { .. } => ("place.index".to_string(), "index"),
        PlaceOp::Slice { .. } => ("place.slice".to_string(), "slice"),
        PlaceOp::ViewIndex { .. } => ("place.view_index".to_string(), "view-index"),
        PlaceOp::OutputSlot { index } => (format!("place.output_slot({index})"), "output-slot"),
    };
    (
        label,
        variant.to_string(),
        GraphOperation {
            semantic_id: None,
            operand_groups,
            regions: Vec::new(),
            results: Vec::new(),
            soac_state: None,
        },
    )
}

fn value_label<R: WynGraphResource>(kind: &ValueKind<R>) -> (String, String) {
    match kind {
        ValueKind::Pure { op, .. } => (inline_debug(op), "pure".to_string()),
        ValueKind::Union { .. } => ("union".to_string(), "union".to_string()),
        ValueKind::FuncParam { parameter } => (parameter_label(parameter), "parameter".to_string()),
        ValueKind::BlockParam { index, .. } => (format!("block param {index}"), "parameter".to_string()),
        ValueKind::CallResult { slot, .. } => (format!("call result {slot:?}"), "result".to_string()),
        ValueKind::PlaceLength { .. } => ("place length".to_string(), "place".to_string()),
        ValueKind::PlaceView { .. } => ("place view".to_string(), "place".to_string()),
        ValueKind::Constant(value) => (inline_debug(value), "constant".to_string()),
        ValueKind::SideEffectResult => ("effect result".to_string(), "result".to_string()),
    }
}

fn parameter_label(parameter: &wyn_core::egir::ir::ParameterId) -> String {
    let debug = format!("{parameter:?}");
    let index = debug
        .split_once('(')
        .and_then(|(_, rest)| rest.split_once('v'))
        .map(|(index, _)| index)
        .unwrap_or(debug.as_str());
    format!("param {index}")
}

struct EffectDisplay {
    id: String,
    label: String,
    variant: String,
    detail: String,
    operation: Option<GraphOperation>,
}

fn effect_display<P: SnapshotPhase>(
    group: &str,
    block: wyn_core::flow::BlockId,
    index: usize,
    effect: &wyn_core::egir::types::SideEffect<P>,
    graph: &wyn_core::egir::types::EGraph<P>,
    region_names: &HashMap<FunctionId, String>,
) -> EffectDisplay {
    match effect.kind() {
        SideEffectKind::Soac(SoacEffect(id, soac)) => {
            let (label, variant, mut operation) = match soac {
                Soac::Screma(op) => (
                    "soac.screma".to_string(),
                    if op.form.scan_count() > 0 {
                        "segscan"
                    } else if op.form.reduction_count() > 0 {
                        "segred"
                    } else {
                        "segmap"
                    }
                    .to_string(),
                    screma_operation(group, effect, graph, op, region_names),
                ),
                Soac::Filter(op) => (
                    "soac.filter".to_string(),
                    "filter".to_string(),
                    filter_operation(group, effect, op, region_names),
                ),
                Soac::Hist(op) => (
                    "soac.hist".to_string(),
                    "hist".to_string(),
                    hist_operation(group, effect, op, region_names),
                ),
            };
            operation.semantic_id = P::semantic_id(id);
            EffectDisplay {
                id: P::soac_node_id(id, group, block, index),
                label,
                variant,
                detail: P::soac_detail(id, soac),
                operation: Some(operation),
            }
        }
        SideEffectKind::Effect(operation) => {
            let (label, variant) = effect_operation_name(operation);
            EffectDisplay {
                id: format!("{group}/effect/{block:?}/{index}"),
                label,
                variant,
                detail: format!("{operation:#?}"),
                operation: Some(GraphOperation {
                    semantic_id: None,
                    operand_groups: (!effect.operands().is_empty())
                        .then(|| GraphOperandGroup {
                            role: "operands".to_string(),
                            values: effect
                                .operands()
                                .iter()
                                .copied()
                                .map(|operand| graph_reference(group, operand))
                                .collect(),
                        })
                        .into_iter()
                        .collect(),
                    regions: Vec::new(),
                    results: graph_results(group, graph.effect_result_binding(effect)),
                    soac_state: None,
                }),
            }
        }
    }
}

fn effect_operation_name<R: WynGraphResource>(operation: &EffectOp<R>) -> (String, String) {
    let (label, variant) = match operation {
        EffectOp::Call { .. } => ("func.call".to_string(), "call"),
        EffectOp::Op { tag } => (inline_debug(tag), "op"),
        EffectOp::Alloca { .. } => ("mem.alloca".to_string(), "alloca"),
        EffectOp::Load { .. } => ("mem.load".to_string(), "load"),
        EffectOp::Store { .. } => ("mem.store".to_string(), "store"),
        EffectOp::Atomic { .. } => ("mem.atomic".to_string(), "atomic"),
        EffectOp::ControlBarrier => ("sync.control_barrier".to_string(), "control-barrier"),
    };
    (label, variant.to_string())
}

fn screma_operation<P: SnapshotPhase>(
    group: &str,
    effect: &wyn_core::egir::types::SideEffect<P>,
    graph: &wyn_core::egir::types::EGraph<P>,
    op: &wyn_core::egir::soac::screma::Op<P>,
    region_names: &HashMap<FunctionId, String>,
) -> GraphOperation {
    let mut operand_groups = Vec::new();
    if let Ok(operands) = ScremaOperands::decode(op, effect.operands(), graph.effect_result_binding(effect))
    {
        operand_groups.push(GraphOperandGroup {
            role: "inputs".to_string(),
            values: operands.inputs().map(|operand| graph_reference(group, operand.operand)).collect(),
        });
    } else {
        operand_groups.push(GraphOperandGroup {
            role: "operands".to_string(),
            values: effect
                .operands()
                .iter()
                .copied()
                .map(|operand| graph_reference(group, operand))
                .collect(),
        });
    }

    for (index, scan) in op.form.scans.iter().enumerate() {
        operand_groups.push(GraphOperandGroup {
            role: format!("scan[{index}].neutral"),
            values: scan.neutral.iter().copied().map(|value| value_reference(group, value)).collect(),
        });
    }
    for (index, reduction) in op.form.reductions.iter().enumerate() {
        operand_groups.push(GraphOperandGroup {
            role: format!("reduce[{index}].neutral"),
            values: reduction.neutral.iter().copied().map(|value| value_reference(group, value)).collect(),
        });
    }

    let mut regions = vec![lambda_region("pre", &op.form.pre, group, region_names)];
    regions.extend(
        op.form.scans.iter().enumerate().map(|(index, scan)| {
            lambda_region(format!("scan[{index}]"), &scan.operator, group, region_names)
        }),
    );
    regions.extend(op.form.reductions.iter().enumerate().map(|(index, reduction)| {
        lambda_region(
            format!("reduce[{index}]"),
            &reduction.operator,
            group,
            region_names,
        )
    }));
    regions.push(lambda_region("post", &op.form.post, group, region_names));

    GraphOperation {
        semantic_id: None,
        operand_groups,
        regions,
        results: graph_results(group, effect.result()),
        soac_state: Some(P::screma_state(group, op)),
    }
}

fn filter_operation<P: SnapshotPhase>(
    group: &str,
    effect: &wyn_core::egir::types::SideEffect<P>,
    op: &wyn_core::egir::soac::filter::Op<P>,
    region_names: &HashMap<FunctionId, String>,
) -> GraphOperation {
    GraphOperation {
        semantic_id: None,
        operand_groups: vec![GraphOperandGroup {
            role: "inputs".to_string(),
            values: effect
                .operands()
                .iter()
                .take(op.body.inputs.len())
                .copied()
                .map(|operand| graph_reference(group, operand))
                .collect(),
        }],
        regions: vec![
            lambda_region("map", &op.body.map, group, region_names),
            lambda_region("predicate", &op.body.predicate, group, region_names),
        ],
        results: graph_results(group, effect.result()),
        soac_state: Some(P::filter_state(group, op)),
    }
}

fn hist_operation<P: SnapshotPhase>(
    group: &str,
    effect: &wyn_core::egir::types::SideEffect<P>,
    op: &wyn_core::egir::soac::hist::Op<P>,
    region_names: &HashMap<FunctionId, String>,
) -> GraphOperation {
    let mut operand_groups = vec![GraphOperandGroup {
        role: "inputs".to_string(),
        values: effect
            .operands()
            .iter()
            .take(op.inputs.len())
            .copied()
            .map(|operand| graph_reference(group, operand))
            .collect(),
    }];
    let mut regions = vec![lambda_region("bucket", &op.form.bucket, group, region_names)];

    for (index, operation) in op.form.operations.iter().enumerate() {
        operand_groups.push(GraphOperandGroup {
            role: format!("operation[{index}].shape"),
            values: operation.shape.iter().copied().map(|value| value_reference(group, value)).collect(),
        });
        operand_groups.push(GraphOperandGroup {
            role: format!("operation[{index}].race_factor"),
            values: vec![value_reference(group, operation.race_factor)],
        });
        operand_groups.push(GraphOperandGroup {
            role: format!("operation[{index}].destinations"),
            values: operation.destinations.iter().map(|view| view_reference(group, *view)).collect(),
        });
        match &operation.update {
            wyn_core::egir::soac::hist::Update::OrderedOverwrite { .. } => {}
            wyn_core::egir::soac::hist::Update::Reduce { operator, neutral } => {
                operand_groups.push(GraphOperandGroup {
                    role: format!("operation[{index}].neutral"),
                    values: neutral.iter().copied().map(|value| value_reference(group, value)).collect(),
                });
                regions.push(lambda_region(
                    format!("operation[{index}].reduce"),
                    operator,
                    group,
                    region_names,
                ));
            }
            wyn_core::egir::soac::hist::Update::BucketInsert { capacity, .. } => {
                operand_groups.push(GraphOperandGroup {
                    role: format!("operation[{index}].capacity"),
                    values: vec![value_reference(group, *capacity)],
                });
            }
        }
    }

    GraphOperation {
        semantic_id: None,
        operand_groups,
        regions,
        results: graph_results(group, effect.result()),
        soac_state: Some(P::hist_state(group, op)),
    }
}

fn graph_results(
    group: &str,
    result: Option<&wyn_core::egir::types::ResultBinding<wyn_core::types::Type>>,
) -> Vec<GraphResult> {
    result
        .into_iter()
        .flat_map(|result| result.destination_leaves_with_paths())
        .filter_map(|(path, leaf)| {
            let (ty, destination) = leaf.single_destination()?;
            let (destination, references) = match destination {
                ResultDestination::ReturnValue(value) => {
                    ("return_value", vec![value_reference(group, *value)])
                }
                ResultDestination::Place(PlaceDestination::Fixed(place)) => {
                    ("place", vec![place_reference(group, *place)])
                }
                ResultDestination::Place(PlaceDestination::Bounded { storage, length }) => (
                    "bounded_place",
                    vec![place_reference(group, *storage), place_reference(group, *length)],
                ),
            };
            Some(GraphResult {
                path: path.into_vec(),
                ty: wyn_core::diags::format_type(ty),
                destination: destination.to_string(),
                references,
            })
        })
        .collect()
}

fn lambda_region(
    role: impl Into<String>,
    lambda: &Lambda,
    group: &str,
    region_names: &HashMap<FunctionId, String>,
) -> GraphRegion {
    let (symbol, identity, captures) = match lambda.seg_body() {
        Some(body) => (
            Some(
                region_names
                    .get(&body.region())
                    .cloned()
                    .unwrap_or_else(|| format!("region_{:?}", body.region())),
            ),
            false,
            body.captures().iter().copied().map(|capture| graph_reference(group, capture)).collect(),
        ),
        None => (None, true, Vec::new()),
    };
    GraphRegion {
        role: role.into(),
        symbol,
        identity,
        captures,
        parameter_types: lambda.parameter_types.iter().map(wyn_core::diags::format_type).collect(),
        result_types: lambda.result_types.iter().map(wyn_core::diags::format_type).collect(),
    }
}

fn graph_reference(group: &str, operand: OperandRef) -> GraphReference {
    match operand {
        OperandRef::Value(value) => value_reference(group, value),
        OperandRef::View(view) => view_reference(group, view),
        OperandRef::Place(place) => place_reference(group, place),
    }
}

fn place_reference(group: &str, place: wyn_core::egir::types::PlaceId) -> GraphReference {
    GraphReference {
        id: place_node_id(group, place),
        kind: "place".to_string(),
    }
}

fn value_reference(group: &str, value: wyn_core::egir::types::ValueId) -> GraphReference {
    GraphReference {
        id: value_node_id(group, value),
        kind: "value".to_string(),
    }
}

fn view_reference(group: &str, view: wyn_core::egir::types::ViewId) -> GraphReference {
    GraphReference {
        id: value_node_id(group, view.value()),
        kind: "view".to_string(),
    }
}

fn inline_debug(value: &impl std::fmt::Debug) -> String {
    let text = format!("{value:?}");
    text.lines().map(str::trim).collect::<Vec<_>>().join(" ")
}

fn push_edge(snapshot: &mut GraphSnapshot, source: String, target: String, kind: &str) {
    let id = format!("edge:{}", snapshot.edges.len());
    snapshot.edges.push(GraphEdge {
        id,
        source,
        target,
        kind: kind.to_string(),
    });
}

#[cfg(test)]
#[path = "lib_tests.rs"]
mod tests;
