//! Direct TLC to EGraph conversion.
//!
//! Converts a TLC program directly to the acyclic e-graph representation,
//! bypassing the sequential SSA construction of `to_ssa`. Pure operations
//! are hash-consed (giving GVN for free), and the result is elaborated
//! back to `FuncBody` via demand-driven scheduling (giving DCE for free).

/// EGIR directly converted from backend-ready TLC.
#[derive(Debug, Clone, Copy)]
pub enum ConvertedTag {}
pub type Converted = super::program::Program<
    ConvertedTag,
    super::ir::ProgramFamily<
        super::types::Raw,
        super::program::SemanticResourceDecl,
        super::ir::UnrealizedOutputRoute,
        super::program::CoreProgramData,
    >,
    super::program::RewriteGlobal,
>;

use crate::builtins::{catalog, Purity};
use crate::op::BinaryOperator;
use crate::tlc::VarRef;
use crate::{LookupMap, LookupSet};

use super::types::EffectToken;
use crate::ast::{Span, TypeName};
use crate::binding_layout::{
    extract_io_decoration, extract_sampler_binding, extract_storage_access, extract_storage_binding,
    extract_storage_image_binding, extract_storage_image_resource, extract_texture_backing,
    extract_texture_binding, extract_texture_resource, extract_uniform_binding,
};
use crate::flow::{BlockId, ControlHeader};
use crate::interface::{
    self, BindingExposure, EntryInput, EntryInputKind, EntryOutput, EntryOutputDestination,
    EntryOutputKind, EntryParamBinding, EntryParamBindingKind, IoDecoration, PushConstantSlot,
    StorageAccess, TextureSource,
};
use crate::tlc::{
    ArrayExpr as GenericArrayExpr, Def as GenericDef, DefMeta as GenericDefMeta, Lambda as GenericLambda,
    LoopKind as GenericLoopKind, SoacBody as GenericSoacBody, SoacOp as GenericSoacOp, Term as GenericTerm,
    TermKind as GenericTermKind,
};
use crate::types::ExternDecl;
use crate::types::{extract_function_signature, Diet, TypeExt};
use crate::{BindingRef, SymbolId, SymbolTable};
use polytype::Type;
use smallvec::{smallvec, SmallVec};
use thiserror::Error;

use super::program::{
    CompilerResource, CompilerResourceKind, ConstantDef, CoreProgramData, LogicalResourceArenaBuilder,
    LogicalSize, Program, ProgramIdentities, RawEntry, RawFunc, RewriteGlobal, SemanticOpIdSource,
    SemanticResourceDecl, SemanticResourceRef,
};
use super::soac::{filter, hist, screma};
use super::types::*;
use crate::pipeline_descriptor::BufferLen;

type TlcFamily = crate::tlc::input_slice_bounds::InputBounded;
type ClosureData = crate::tlc::data::ExplicitClosurePayload;
type SoacBodyData = crate::tlc::data::ExplicitCapturesPayload;
type TlcProgram = crate::tlc::stage::InputSliceBoundsInferred;
type TlcDef = GenericDef<TlcFamily>;
type DefMeta = GenericDefMeta<crate::tlc::data::EntryInputBounds>;
type Term = GenericTerm<ClosureData, SoacBodyData>;
type TermKind = GenericTermKind<ClosureData, SoacBodyData>;
type Lambda = GenericLambda<ClosureData, SoacBodyData>;
type LoopKind = GenericLoopKind<ClosureData, SoacBodyData>;
type ArrayExpr = GenericArrayExpr<ClosureData, SoacBodyData>;
type SoacBody = GenericSoacBody<ClosureData, SoacBodyData>;
type SoacOp = GenericSoacOp<ClosureData, SoacBodyData>;

// ============================================================================
// Descriptor-set convention
// ============================================================================

/// Descriptor set reserved for compiler-allocated storage. Compute
/// entry-input/output buffers (from SoA tuple splits), multi-stage SOAC
/// intermediates, and graphical-invariant prepass results all live on this
/// set. User-declared `#[uniform(...)]` and `#[storage(...)]` must use a
/// higher set (the parser enforces `set >= 1`). See SPECIFICATION.md
/// "Descriptor Set Layout" for the rationale.
pub const AUTO_STORAGE_SET: u32 = 0;

// ============================================================================
// Error type
// ============================================================================

#[derive(Debug, Error)]
pub enum ConvertError {
    /// Error during EGraph construction.
    #[error("EGraph conversion error: {0}")]
    GraphError(String),
    /// Unsupported TLC construct (todo).
    #[error("Unsupported: {0}")]
    Unsupported(String),
    /// Compiler invariant violated — a downstream pass should have made
    /// this state unreachable. Surfaces as a propagated error rather than
    /// a panic so the caller can label it as an internal compiler error
    /// in user-facing output.
    #[error("internal compiler error: {0}")]
    Internal(String),
    /// A source `#[dispatch(...)]` grid is incompatible with the entry — e.g.
    /// it launches fewer threads than the entry's data-parallel domain has
    /// elements, silently dropping the tail. A user error, not internal.
    #[error("{0}")]
    InvalidDispatch(String),
    /// Two declared descriptor resources claim the same `(set, binding)` with
    /// incompatible descriptor classes or layout-relevant properties.
    #[error("{0}")]
    DescriptorLayout(#[from] super::publish::DescriptorError),
}

/// Untyped downstream diagnostics are compiler-internal by default. User
/// errors use explicit variants at their classification boundary.
impl From<String> for ConvertError {
    fn from(error: String) -> Self {
        Self::Internal(error)
    }
}

/// Look up `sym`'s source name in `symbols`, returning a propagated
/// `ConvertError::Internal` if the symbol isn't in the table. The
/// situation is a compiler-internal invariant violation — every symbol
/// reachable from a `Def` should have been registered during
/// resolution — but propagating instead of panicking lets the caller
/// surface a structured "internal compiler error" message.
fn symbol_name(symbols: &SymbolTable, sym: SymbolId) -> Result<&str, ConvertError> {
    symbols
        .get(sym)
        .map(String::as_str)
        .ok_or_else(|| ConvertError::Internal(format!("symbol {sym:?} not in symbol table")))
}

#[derive(Default)]
struct DefinitionEffects {
    direct_effect: bool,
    calls: LookupSet<SymbolId>,
}

/// Infer callable purity from the complete, post-defunctionalization TLC
/// program before constructing any EGIR bodies. Starting with every locally
/// effect-free definition and removing callers of non-candidates propagates
/// impurity transitively through the call graph.
///
/// Pure calls are currently limited to copy-value ABIs. Array and resource
/// consumers remain anchored because semantic EGIR uses skeleton effects to
/// represent their producer/consumer dependencies before residency is fixed.
fn infer_pure_definitions(program: &TlcProgram) -> LookupSet<SymbolId> {
    let top_level: LookupMap<SymbolId, &TlcDef> = program.defs.iter().map(|def| (def.name, def)).collect();
    let summaries: LookupMap<SymbolId, DefinitionEffects> = program
        .defs
        .iter()
        .filter(|def| matches!(def.meta, DefMeta::Function | DefMeta::LiftedLambda))
        .filter(|def| !matches!(def.body.kind, TermKind::Extern(_)))
        .filter(|def| {
            let (params, result) = extract_function_signature(&def.ty);
            params.iter().all(crate::types::is_copy) && crate::types::is_copy(&result)
        })
        .map(|def| {
            let mut summary = DefinitionEffects::default();
            summarize_definition_effects(&def.body, &top_level, &mut summary);
            (def.name, summary)
        })
        .collect();

    let mut pure = summaries
        .iter()
        .filter_map(|(&definition, summary)| (!summary.direct_effect).then_some(definition))
        .collect::<LookupSet<_>>();
    loop {
        let rejected = pure
            .iter()
            .copied()
            .filter(|definition| summaries[definition].calls.iter().any(|callee| !pure.contains(callee)))
            .collect::<Vec<_>>();
        if rejected.is_empty() {
            return pure;
        }
        for definition in rejected {
            pure.remove(&definition);
        }
    }
}

fn summarize_definition_effects(
    term: &Term,
    top_level: &LookupMap<SymbolId, &TlcDef>,
    summary: &mut DefinitionEffects,
) {
    match &term.kind {
        TermKind::App { func, args } => {
            for arg in args {
                summarize_definition_effects(arg, top_level, summary);
            }
            match &func.kind {
                TermKind::BinOp(_) | TermKind::UnOp(_) => {}
                TermKind::Var(VarRef::Builtin { id, .. }) => {
                    // `storage_index` is surface-pure but EGIR models the
                    // backing-buffer read as an ordered Load.
                    if *id == catalog().known().storage_index
                        || crate::builtins::by_id(*id).raw.purity == Purity::Effectful
                    {
                        summary.direct_effect = true;
                    }
                }
                TermKind::Var(VarRef::Symbol(callee)) if top_level.contains_key(callee) => {
                    summary.calls.insert(*callee);
                }
                // General/indirect application is not representable as a pure
                // EGIR call at this construction boundary.
                _ => summary.direct_effect = true,
            }
        }
        TermKind::Var(VarRef::Symbol(definition)) => {
            // Arity-zero definitions are evaluated by `convert_var`, which
            // inlines their bodies at the reference site.
            if top_level.get(definition).is_some_and(|def| def.arity == 0) {
                summary.calls.insert(*definition);
            }
        }
        TermKind::Index { array, index } => {
            summarize_definition_effects(array, top_level, summary);
            summarize_definition_effects(index, top_level, summary);
            if array.ty.array_variant().is_some_and(crate::types::is_array_variant_view) {
                summary.direct_effect = true;
            }
        }
        // SOACs remain anchored semantic operations during EGIR construction,
        // even when their element function is pure.
        TermKind::Soac(_) | TermKind::Extern(_) | TermKind::Coerce { .. } => {
            summary.direct_effect = true;
        }
        _ => term.for_each_child(&mut |child| summarize_definition_effects(child, top_level, summary)),
    }
}

/// Read-only state shared across every converter built during a single
/// `run` — the top-level def index, the arity-0 name → symbol map, and
/// the symbol table. Acts as a factory: `new_converter` snapshots the
/// caller's current `pure_constants` set into a fresh `Converter`,
/// keeping the per-call `clone()` inside one method.
struct GlobalContext<'a> {
    top_level: &'a LookupMap<SymbolId, &'a TlcDef>,
    symbols: &'a SymbolTable,
    pure_definitions: &'a LookupSet<SymbolId>,
}

struct ConversionArenas {
    identities: ProgramIdentities,
    function_ids: LookupMap<SymbolId, crate::FunctionId>,
    global_ids: LookupMap<SymbolId, crate::GlobalId>,
    entry_ids: LookupMap<SymbolId, crate::EntryId>,
    resources: LogicalResourceArenaBuilder,
}

impl ConversionArenas {
    fn new() -> Self {
        Self {
            identities: ProgramIdentities::new(),
            function_ids: LookupMap::new(),
            global_ids: LookupMap::new(),
            entry_ids: LookupMap::new(),
            resources: LogicalResourceArenaBuilder::default(),
        }
    }
}

impl<'a> GlobalContext<'a> {
    fn new_converter<'b>(
        &self,
        pure_constants: &LookupSet<SymbolId>,
        binding_ids: &'b mut crate::IdSource<u32>,
        effect_ids: &'b mut crate::IdSource<EffectToken>,
        arenas: &'b mut ConversionArenas,
    ) -> Converter<'a, 'b> {
        Converter::new(
            self.top_level,
            self.symbols,
            pure_constants.clone(),
            self.pure_definitions.clone(),
            binding_ids,
            effect_ids,
            arenas,
        )
    }
}

// ============================================================================
// Public entry point
// ============================================================================

/// Convert a TLC program into a raw EGIR program — each function and entry
/// point becomes a per-body `EGraph` + metadata, waiting for the caller to
/// chain the pipeline (`expand_soacs → [materialize →] optimize_skeleton →
/// elaborate`).
pub fn convert_program(
    program: &TlcProgram,
    mut binding_ids: crate::IdSource<u32>,
    mut effect_ids: crate::IdSource<EffectToken>,
) -> Result<Converted, ConvertError> {
    let super::pipeline_seed::PipelineSeed {
        pipeline,
        stage_symbols,
    } = super::pipeline_seed::build(program);
    let top_level: LookupMap<SymbolId, &TlcDef> = program.defs.iter().map(|d| (d.name, d)).collect();
    let symbols = &program.symbols;
    let pure_definitions = infer_pure_definitions(program);

    // Program-level arenas are borrowed by one converter at a time, then
    // handed intact to the semantic program.
    let mut arenas = ConversionArenas::new();
    for def in &program.defs {
        let name = symbol_name(symbols, def.name)?.to_owned();
        match &def.meta {
            DefMeta::EntryPoint(_) => {
                let id = arenas.identities.alloc_entry(name);
                arenas.entry_ids.insert(def.name, id);
            }
            DefMeta::Function | DefMeta::LiftedLambda => {
                let id = arenas.identities.alloc_function(name);
                arenas.function_ids.insert(def.name, id);
            }
        }
    }

    let stage_entries: Vec<Vec<crate::EntryId>> = stage_symbols
        .into_iter()
        .map(|stages| {
            stages
                .into_iter()
                .map(|symbol| {
                    *arenas.entry_ids.get(&symbol).expect("pipeline stage has no allocated entry identity")
                })
                .collect()
        })
        .collect();

    let ctx = GlobalContext {
        top_level: &top_level,
        symbols,
        pure_definitions: &pure_definitions,
    };

    // Phase 1: detect pure constants. We elaborate each arity-0 def's body
    // through the full EGIR pipeline once (using a throwaway chain) to see if
    // it collapses to a purely-constant FuncBody. Constants are hoisted to
    // program scope and referenced by `PureOp::Global`.
    let mut pure_constant_symbols: LookupSet<SymbolId> = LookupSet::new();
    let mut constants = Vec::new();

    for def in &program.defs {
        if def.arity != 0 || !matches!(&def.meta, DefMeta::Function) {
            continue;
        }
        if matches!(&def.body.kind, TermKind::Extern(_)) {
            continue;
        }
        let def_name = symbols.get(def.name).expect("BUG: symbol not in table").clone();

        let mut converter = ctx.new_converter(
            &pure_constant_symbols,
            &mut binding_ids,
            &mut effect_ids,
            &mut arenas,
        );
        if let Ok(result_nid) = converter.convert_term(&def.body) {
            converter.set_return(Some(result_nid));
            let mut graph = converter.into_graph();
            let aliases = super::skel_opt::run_one_body(&mut graph);
            graph.install_aliases(aliases);
            if is_purely_constant_graph(&graph) {
                pure_constant_symbols.insert(def.name);
                let id = arenas.identities.alloc_global(def_name.clone());
                arenas.global_ids.insert(def.name, id);
                constants.push(ConstantDef {
                    id,
                    name: def_name,
                    span: def.body.span,
                    return_ty: def.body.ty.clone(),
                    graph,
                });
                continue;
            }
        }
    }

    // Phase 2: convert functions and entry points into raw EGIR records.
    let mut functions: Vec<RawFunc> = Vec::new();
    let mut externs: Vec<ExternDecl<Type<TypeName>>> = Vec::new();
    let mut entry_points: Vec<RawEntry> = Vec::new();

    for def in &program.defs {
        match &def.meta {
            DefMeta::Function | DefMeta::LiftedLambda => {
                if pure_constant_symbols.contains(&def.name) {
                    continue;
                }
                match convert_function(
                    def,
                    &ctx,
                    &pure_constant_symbols,
                    &mut binding_ids,
                    &mut effect_ids,
                    &mut arenas,
                )? {
                    ConvertedFunc::Extern(f) => externs.push(f),
                    ConvertedFunc::Regular(fe) => functions.push(fe),
                }
            }
            DefMeta::EntryPoint(entry) => {
                let workgroup =
                    pipeline_workgroup_size(&pipeline, &stage_entries, arenas.entry_ids[&def.name]);
                let ep = convert_entry_point(
                    def,
                    &entry.declaration,
                    &entry.data.param_bindings,
                    &ctx,
                    &pure_constant_symbols,
                    workgroup,
                    &entry.data.by_symbol,
                    &mut binding_ids,
                    &mut effect_ids,
                    &mut arenas,
                )?;
                entry_points.push(ep);
            }
        }
    }

    debug_assert!(functions.iter().all(|function| {
        arenas.identities.contains_function(function.region)
            && arenas.identities.function_name(function.region) == function.name
    }));
    debug_assert!(externs.iter().all(|function| {
        arenas.identities.contains_function(function.id)
            && arenas.identities.function_name(function.id) == function.name
    }));
    debug_assert!(entry_points.iter().all(|entry| {
        arenas.identities.contains_entry(entry.id) && arenas.identities.entry_name(entry.id) == entry.name
    }));
    debug_assert!(constants.iter().all(|constant| {
        arenas.identities.contains_global(constant.id)
            && arenas.identities.global_name(constant.id) == constant.name
    }));
    let ConversionArenas {
        identities,
        resources,
        ..
    } = arenas;
    let resources = resources.finish().map_err(|resource| {
        ConvertError::Internal(format!(
            "semantic resource {resource:?} was referenced but never declared"
        ))
    })?;
    let mut converted = Program::from_parts(
        functions,
        externs,
        entry_points,
        constants,
        CoreProgramData {
            pipeline,
            stage_entries,
            resources,
            identities,
        },
        RewriteGlobal {
            binding_ids,
            effect_ids,
            semantic_ids: SemanticOpIdSource::default(),
        },
    );
    super::program::finalize_converted_resources(&mut converted);
    Ok(converted)
}

fn pipeline_workgroup_size(
    pipeline: &crate::pipeline_descriptor::PipelineDescriptor,
    stage_entries: &[Vec<crate::EntryId>],
    entry: crate::EntryId,
) -> (u32, u32, u32) {
    use crate::pipeline_descriptor::Pipeline;
    pipeline
        .pipelines
        .iter()
        .enumerate()
        .find_map(|(pipeline_index, pipeline)| match pipeline {
            Pipeline::Compute(compute) => {
                compute.stages.iter().enumerate().find_map(|(stage_index, stage)| {
                    (stage_entries.get(pipeline_index).and_then(|entries| entries.get(stage_index))
                        == Some(&entry))
                    .then_some(stage.workgroup_size)
                })
            }
            Pipeline::Graphics(_) => None,
        })
        .unwrap_or((64, 1, 1))
}

enum ConvertedFunc {
    Extern(ExternDecl<Type<TypeName>>),
    Regular(RawFunc),
}

// ============================================================================
// Function conversion
// ============================================================================

fn convert_function<'a>(
    def: &TlcDef,
    ctx: &GlobalContext<'a>,
    pure_constants: &LookupSet<SymbolId>,
    binding_ids: &'a mut crate::IdSource<u32>,
    effect_ids: &'a mut crate::IdSource<EffectToken>,
    arenas: &'a mut ConversionArenas,
) -> Result<ConvertedFunc, ConvertError> {
    let symbols = ctx.symbols;
    let def_name = symbol_name(symbols, def.name)?.to_string();
    let function_id = arenas.function_ids[&def.name];

    // Extern functions are bodyless declarations; SSA lowering decides how
    // to represent the imported callable.
    if let TermKind::Extern(linkage_name) = &def.body.kind {
        let (param_types, ret_type) = extract_function_signature(&def.ty);
        let params: Vec<(Type<TypeName>, String)> =
            param_types.into_iter().enumerate().map(|(i, ty)| (ty, format!("arg{}", i))).collect();
        return Ok(ConvertedFunc::Extern(ExternDecl {
            id: function_id,
            name: def_name,
            span: def.body.span,
            linkage_name: linkage_name.clone(),
            params,
            return_ty: ret_type,
        }));
    }

    // Regular functions: extract lambda params and build an EGraph.
    let (inner_body, params) = crate::tlc::extract_lambda_params_ref(&def.body);
    let ret_type = inner_body.ty.clone();
    let param_info: Vec<(Type<TypeName>, String)> = params
        .iter()
        .map(|(sym, ty)| Ok((ty.clone(), symbol_name(symbols, *sym)?.to_string())))
        .collect::<Result<_, ConvertError>>()?;

    let mut converter = ctx.new_converter(pure_constants, binding_ids, effect_ids, arenas);

    for (i, (sym, ty)) in params.iter().enumerate() {
        let nid = converter.graph.add_func_param(i, ty.clone());
        converter.locals.insert(*sym, nid);
    }
    let result = converter.convert_term(inner_body)?;
    converter.set_return(Some(result));

    // A runtime `filter` compacts into a compiler scratch resource, which only
    // a `SemanticEntry` can own as a physicalization requirement. A standalone
    // function has no entry interface to publish that requirement. In practice
    // a function whose
    // result is a runtime filter is inlined into its caller before this pass
    // (see `filter_runtime_in_subroutine_compiles`), so this never fires. If it
    // does, that inlining invariant broke: either restore it, or represent the
    // scratch resource explicitly in the function's semantic ABI.
    if !converter.extra_resource_declarations.is_empty() {
        return Err(ConvertError::GraphError(format!(
            "runtime `filter` in function `{def_name}` reserved a scratch storage buffer, but a \
             standalone function has no descriptor-set interface to host it — the call must be \
             inlined into a compute entry (it was not)"
        )));
    }

    let region = function_id;
    let graph = converter.into_graph();
    Ok(ConvertedFunc::Regular(RawFunc::new(
        region,
        def_name,
        def.body.span,
        None,
        param_info,
        ret_type,
        graph,
    )))
}

/// Translate descriptor sizing metadata into stable logical-resource sizes.
fn logical_size(resources: &mut LogicalResourceArenaBuilder, length: Option<&BufferLen>) -> LogicalSize {
    match length {
        Some(BufferLen::Fixed { bytes }) => LogicalSize::FixedBytes(*bytes),
        Some(BufferLen::LikeInput {
            set,
            binding,
            elem_bytes,
            src_elem_bytes,
        }) => LogicalSize::LikeResource {
            resource: resources.host_id(BindingRef::new(*set, *binding)),
            elem_bytes: *elem_bytes,
            src_elem_bytes: *src_elem_bytes,
        },
        Some(BufferLen::SameAsDispatch { elem_bytes }) => LogicalSize::SameAsDispatch {
            elem_bytes: *elem_bytes,
        },
        None => LogicalSize::Unspecified,
    }
}

fn literal_binding(args: &[Term], intrinsic: &str) -> Result<BindingRef, ConvertError> {
    let parse = |term: &Term, part: &str| match &term.kind {
        TermKind::IntLit(value) => value
            .parse::<u32>()
            .map_err(|_| ConvertError::GraphError(format!("{intrinsic}: {part} not a u32"))),
        _ => Err(ConvertError::GraphError(format!(
            "{intrinsic}: {part} must be int literal"
        ))),
    };
    Ok(BindingRef::new(
        parse(&args[0], "set")?,
        parse(&args[1], "binding")?,
    ))
}

fn entry_resource_declarations(
    inputs: &[EntryInput],
    outputs: &[EntryOutput],
    resources: &mut LogicalResourceArenaBuilder,
) -> Vec<SemanticResourceDecl> {
    let mut declarations = Vec::new();
    let mut declare = |resources: &mut LogicalResourceArenaBuilder,
                       binding: BindingRef,
                       role: interface::StorageRole,
                       elem_ty: Type<TypeName>,
                       size: LogicalSize| {
        let resource = resources.declare_host(binding, elem_ty.clone(), size.clone());
        if !declarations
            .iter()
            .any(|item: &SemanticResourceDecl| item.resource == SemanticResourceRef(resource))
        {
            declarations.push(SemanticResourceDecl {
                resource: SemanticResourceRef(resource),
                role,
                elem_ty,
                size,
            });
        }
    };
    for input in inputs {
        match &input.kind {
            EntryInputKind::Storage {
                exposure: BindingExposure::Host(binding),
                length,
                ..
            } => {
                let size = logical_size(resources, length.as_ref());
                let elem_ty = input.ty.elem_type().cloned().unwrap_or_else(|| input.ty.clone());
                declare(resources, *binding, interface::StorageRole::Input, elem_ty, size);
            }
            EntryInputKind::StorageImage { binding, .. } => {
                resources.declare_host(*binding, input.ty.clone(), LogicalSize::Unspecified);
            }
            _ => {}
        }
    }
    for output in outputs {
        let EntryOutputKind::Storage {
            exposure: BindingExposure::Host(binding),
            length,
        } = &output.kind
        else {
            continue;
        };
        let size = logical_size(resources, length.as_ref());
        let elem_ty = output.ty.elem_type().cloned().unwrap_or_else(|| output.ty.clone());
        declare(resources, *binding, interface::StorageRole::Output, elem_ty, size);
    }
    declarations
}

fn storage_access_from_diet(diet: Option<&Diet>) -> StorageAccess {
    if diet.is_some_and(Diet::is_consuming) {
        StorageAccess::ReadWrite
    } else {
        StorageAccess::ReadOnly
    }
}

fn tuple_field_storage_access(diet: Option<&Diet>, field: usize) -> StorageAccess {
    let consuming = diet.is_some_and(|diet| match diet {
        Diet::Aggregate { unique: true, .. } => true,
        Diet::Aggregate { components, .. } => components.get(field).is_some_and(Diet::is_consuming),
        other => other.is_consuming(),
    });
    if consuming {
        StorageAccess::ReadWrite
    } else {
        StorageAccess::ReadOnly
    }
}

#[allow(clippy::too_many_arguments)]
fn convert_entry_point(
    def: &TlcDef,
    entry: &interface::EntryDecl,
    param_bindings: &[Option<EntryParamBinding>],
    ctx: &GlobalContext,
    pure_constants: &LookupSet<SymbolId>,
    workgroup: (u32, u32, u32),
    input_bounds: &LookupMap<SymbolId, BufferLen>,
    binding_ids: &mut crate::IdSource<u32>,
    effect_ids: &mut crate::IdSource<EffectToken>,
    arenas: &mut ConversionArenas,
) -> Result<RawEntry, ConvertError> {
    use crate::flow::ExecutionModel;

    let entry_id = arenas.entry_ids[&def.name];
    let symbols = ctx.symbols;
    let def_name = symbol_name(symbols, def.name)?;
    let (inner_body, params) = crate::tlc::extract_lambda_params_ref(&def.body);
    let is_compute = entry.entry_kind == interface::EntryKind::Compute;

    // The converted body carries the specialized return representation; use it
    // rather than the parse-time entry declaration.
    let ret_type = inner_body.ty.clone();
    let param_info: Vec<(Type<TypeName>, String)> = params
        .iter()
        .map(|(sym, ty)| Ok((ty.clone(), symbol_name(symbols, *sym)?.to_string())))
        .collect::<Result<_, ConvertError>>()?;

    let mut converter = ctx.new_converter(pure_constants, binding_ids, effect_ids, arenas);

    // Build entry inputs alongside the symbol → NodeId bindings. A compute
    // entry param that's a tuple-of-unsized-arrays gets one storage binding
    // per field (SoA lowered the source `[]T` of tuples into a tuple of
    // `[]T`s, but entry I/O can't carry a tuple handle — each runtime-sized
    // array needs its own buffer). The body still references the original
    // tuple symbol, so we reconstruct it as a `Tuple(views…)` node.
    let mut inputs: Vec<EntryInput> = Vec::with_capacity(params.len());
    // A source parameter can expand into multiple ABI slots (tuple views).
    let mut input_parameter_indices = Vec::with_capacity(params.len());
    let mut pc_offset: u32 = 0;

    // The auto-storage binding layout is dense — same length as `params`,
    // with `None` for non-storage params — so we walk them in lockstep.
    for (i, ((sym, ty), param_binding)) in params.iter().zip(param_bindings.iter()).enumerate() {
        let name = symbol_name(symbols, *sym)?;
        let decoration = entry.params.get(i).and_then(extract_io_decoration);
        let size_hint = entry.params.get(i).and_then(extract_size_hint);
        let uniform_binding = entry.params.get(i).and_then(extract_uniform_binding);
        let attr_storage_binding = entry.params.get(i).and_then(extract_storage_binding);
        let storage_access = entry.params.get(i).and_then(extract_storage_access);
        let texture_binding = entry.params.get(i).and_then(extract_texture_binding);
        let texture_backing = entry.params.get(i).and_then(extract_texture_backing);
        let texture_resource = entry.params.get(i).and_then(extract_texture_resource);
        let sampler_binding = entry.params.get(i).and_then(extract_sampler_binding);
        let storage_image_binding = entry.params.get(i).and_then(extract_storage_image_binding);
        let storage_image_resource = entry.params.get(i).and_then(extract_storage_image_resource);

        // Uniqueness is an ownership-tracking concept that's already been
        // consumed by `apply_ownership`; codegen operates on the stripped
        // shape so `*[N]T` and `[N]T` lower identically.
        let ty = ty.clone();
        let ty = &ty;

        // Always register a FuncParam placeholder so param indexing stays
        // stable; the binding below may override it.
        let fp_nid = converter.graph.add_func_param(i, ty.clone());
        converter.locals.insert(*sym, fp_nid);

        // Tuple-of-unsized-arrays: the layout already decided which
        // (set, binding) goes to each field. Reconstruct the param as a
        // `Tuple(view…)` node so the body's reference resolves.
        if let Some(EntryParamBinding {
            kind: EntryParamBindingKind::TupleOfViews(fields),
            ..
        }) = param_binding
        {
            let field_tys = match ty {
                Type::Constructed(TypeName::Tuple(_), field_tys) => field_tys.as_slice(),
                _ => {
                    return Err(ConvertError::Internal(format!(
                        "tuple-of-views param `{name}` is not a tuple type"
                    )));
                }
            };
            if field_tys.len() != fields.len() {
                return Err(ConvertError::Internal(format!(
                    "tuple-of-views param `{name}`: layout has {} fields, type has {}",
                    fields.len(),
                    field_tys.len(),
                )));
            }
            let mut view_nids: SmallVec<[NodeId; 4]> = SmallVec::new();
            for (field_idx, (field_ty, slot)) in field_tys.iter().zip(fields.iter()).enumerate() {
                inputs.push(EntryInput {
                    name: format!("{}_{}", name, field_idx),
                    ty: crate::types::canonical_storage_buffer_ty(field_ty),
                    size_hint: None,
                    kind: EntryInputKind::Storage {
                        exposure: BindingExposure::Host(slot.binding),
                        access: tuple_field_storage_access(entry.param_diets.get(i), field_idx),
                        length: None,
                    },
                });
                input_parameter_indices.push(i);
                view_nids.push(converter.emit_storage_view(slot.binding, field_ty.clone()));
            }
            let tuple_nid = converter.intern_pure(PureOp::Tuple(view_nids.len()), view_nids, ty.clone());
            converter.locals.insert(*sym, tuple_nid);
            continue;
        }

        let auto_storage_binding = param_binding.as_ref().and_then(|b| match &b.kind {
            EntryParamBindingKind::Single { binding, .. } => Some(*binding),
            EntryParamBindingKind::TupleOfViews(_) => None,
        });
        let storage_binding = auto_storage_binding.or(attr_storage_binding);

        let push_constant = if is_compute
            && storage_binding.is_none()
            && uniform_binding.is_none()
            && texture_binding.is_none()
            && sampler_binding.is_none()
            && storage_image_binding.is_none()
            && !matches!(&decoration, Some(IoDecoration::BuiltIn(_)))
        {
            let size = crate::ssa::layout::type_byte_size(ty).ok_or_else(|| {
                ConvertError::Internal(format!(
                    "push-constant param `{}` has no static byte layout",
                    name
                ))
            })?;
            let offset = pc_offset;
            pc_offset += size;
            Some(PushConstantSlot { offset, size })
        } else {
            None
        };

        if let Some(br) = storage_binding {
            let view_nid = converter.emit_storage_view(br, ty.clone());
            converter.locals.insert(*sym, view_nid);
        }

        let kind = if let Some(binding) = storage_binding {
            EntryInputKind::Storage {
                exposure: BindingExposure::Host(binding),
                access: storage_access
                    .unwrap_or_else(|| storage_access_from_diet(entry.param_diets.get(i))),
                length: None,
            }
        } else if let Some(binding) = uniform_binding {
            EntryInputKind::Uniform { binding }
        } else if let Some(binding) = texture_binding {
            let source = match (texture_backing, texture_resource) {
                (Some(backing), None) => TextureSource::Backing(backing),
                (backing, Some(name)) => TextureSource::Resource { name, backing },
                (None, None) => TextureSource::External,
            };
            EntryInputKind::Texture { binding, source }
        } else if let Some(binding) = sampler_binding {
            EntryInputKind::Sampler { binding }
        } else if let Some((binding, format, access, size)) = storage_image_binding {
            EntryInputKind::StorageImage {
                binding,
                format,
                access,
                size,
                resource: storage_image_resource,
            }
        } else if let Some(slot) = push_constant {
            EntryInputKind::PushConstant { slot }
        } else {
            EntryInputKind::Value { decoration }
        };
        inputs.push(EntryInput {
            name: name.to_string(),
            ty: crate::types::canonical_storage_buffer_ty(ty),
            size_hint,
            kind,
        });
        input_parameter_indices.push(i);
    }

    // The owning TLC entry carries its inferred per-parameter minimums.
    for (input, (sym, _)) in inputs.iter_mut().zip(params.iter()) {
        let EntryInputKind::Storage { length, .. } = &mut input.kind else {
            continue;
        };
        if let Some(len) = input_bounds.get(sym).cloned() {
            *length = Some(len);
        }
    }
    let execution_model = match entry.entry_kind {
        interface::EntryKind::Root => {
            return Err(ConvertError::Internal(
                "unextracted unified root reached EGIR".into(),
            ))
        }
        interface::EntryKind::Vertex => ExecutionModel::Vertex,
        interface::EntryKind::Fragment => ExecutionModel::Fragment,
        interface::EntryKind::Compute => ExecutionModel::Compute {
            local_size: workgroup,
        },
    };

    let is_unit_return = matches!(
        ret_type,
        Type::Constructed(TypeName::Unit | TypeName::SideEffect, _)
    ) || is_storage_image_ty(&ret_type);

    // Convert body. Output assignment (storing the result into the bound
    // storage views / graphics output slots, and retargeting tail
    // Map/Scan SOACs to stream directly into a runtime-sized output) is a
    // separate, uniform pass — `egir::realize_outputs`, run right after
    // this conversion. Here we just leave the body terminating in its
    // single tail value (or `None` for a unit entry).
    let output_arity = entry_output_arity(entry, &ret_type);
    let result_nid = if is_compute && output_arity != 0 {
        converter.convert_compute_outputs(inner_body, output_arity)?
    } else {
        Some(converter.convert_term(inner_body)?)
    };

    let slot_value_tys = converter
        .output_sources
        .iter()
        .map(|sources| sources.first().map(|source| converter.graph.nodes[source.value].ty.clone()))
        .collect::<Vec<_>>();
    let outputs = build_entry_outputs(
        entry,
        &ret_type,
        &slot_value_tys,
        &inputs,
        is_compute,
        converter.binding_ids,
    )?;

    // `convert_compute_outputs` records per-slot sources while preserving the
    // original control flow. Output realization later assigns concrete writers.
    converter.set_return(if is_unit_return { None } else { result_nid });

    // Compute entries publish through routes, not function return values.
    let ret_type = if is_compute || is_storage_image_ty(&ret_type) {
        Type::Constructed(TypeName::Unit, vec![])
    } else {
        ret_type
    };

    let mut resource_declarations =
        entry_resource_declarations(&inputs, &outputs, &mut converter.arenas.resources);
    let Converter {
        graph,
        output_sources: slot_sources,
        extra_resource_declarations,
        ..
    } = converter;
    resource_declarations.extend(extra_resource_declarations);
    let output_count = outputs.len();
    let mut entry = RawEntry::new_with_resources(
        def_name.to_string(),
        entry_id,
        def.body.span,
        execution_model,
        inputs,
        outputs,
        resource_declarations,
        param_info,
        ret_type,
        graph,
    );
    entry.parameter_inputs = vec![Vec::new(); entry.params.len()];
    for (slot, parameter_index) in input_parameter_indices.into_iter().enumerate() {
        entry.parameter_inputs[parameter_index].push(super::program::InputSlotId(slot));
    }
    for (slot, sources) in slot_sources.into_iter().enumerate().take(output_count) {
        entry.outputs[slot]
            .routes
            .extend(sources.into_iter().map(|source| super::ir::UnrealizedOutputRoute { source }));
    }

    Ok(entry)
}

fn is_storage_image_ty(ty: &Type<TypeName>) -> bool {
    matches!(ty, Type::Constructed(TypeName::StorageTexture, _))
}

fn strip_existentials(mut ty: &Type<TypeName>) -> &Type<TypeName> {
    while let Type::Constructed(TypeName::Existential(_), args) = ty {
        let Some(inner) = args.first() else {
            break;
        };
        ty = inner;
    }
    ty
}

fn entry_output_arity(entry: &interface::EntryDecl, ret_type: &Type<TypeName>) -> usize {
    match strip_existentials(ret_type) {
        Type::Constructed(TypeName::Unit | TypeName::SideEffect | TypeName::StorageTexture, _) => 0,
        Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields) => fields.len(),
        _ => usize::from(!entry.outputs.is_empty()),
    }
}

// ============================================================================
// Converter
// ============================================================================

struct Converter<'a, 'b> {
    /// The e-graph being built.
    graph: EGraph<Raw>,
    /// Current skeleton block for side effects and terminators.
    current_block: BlockId,
    /// TLC variable → EGraph node mapping.
    locals: LookupMap<SymbolId, NodeId>,
    /// Top-level definitions.
    top_level: &'a LookupMap<SymbolId, &'a TlcDef>,
    /// Symbol table.
    symbols: &'a SymbolTable,
    /// Cache for inlined constant bodies.
    inlined_constants: LookupMap<SymbolId, NodeId>,
    /// Identities of hoisted pure constants.
    pure_constants: LookupSet<SymbolId>,
    /// User definitions proven pure before EGIR construction.
    pure_definitions: LookupSet<SymbolId>,
    /// Program-wide identity source for effect-chain endpoints.
    effect_ids: &'b mut crate::IdSource<EffectToken>,
    /// Span of the term currently being converted. Threaded through every
    /// pure-node intern and side-effect push so backend errors can blame
    /// the originating source. Pushed/popped in `convert_term`; `None`
    /// only outside any term conversion (e.g. entry-point glue).
    current_span: Option<Span>,
    /// Per-slot `SlotSource { block, value }` records derived directly from a
    /// compute entry's original tail and consumed by `convert_entry_point` to
    /// populate the declared outputs' route lists.
    ///
    /// A slot with one source has `vec![one]`; a slot written from both
    /// arms of an `If` has two. Unit-returning entries leave it empty.
    output_sources: Vec<Vec<crate::egir::program::SlotSource>>,
    /// Module-wide id factory for host-visible auto-storage binding numbers.
    /// Compiler resources never draw from this namespace.
    binding_ids: &'b mut crate::IdSource<u32>,
    /// Compiler-introduced logical resource declarations accumulated during
    /// body conversion (runtime `filter` scratch buffers).
    extra_resource_declarations: Vec<SemanticResourceDecl>,
    /// Program-wide arenas borrowed exclusively for this conversion.
    arenas: &'b mut ConversionArenas,
}

impl<'a, 'b> Converter<'a, 'b> {
    fn new(
        top_level: &'a LookupMap<SymbolId, &'a TlcDef>,
        symbols: &'a SymbolTable,
        pure_constants: LookupSet<SymbolId>,
        pure_definitions: LookupSet<SymbolId>,
        binding_ids: &'b mut crate::IdSource<u32>,
        effect_ids: &'b mut crate::IdSource<EffectToken>,
        arenas: &'b mut ConversionArenas,
    ) -> Self {
        let graph = EGraph::new();
        let entry = graph.skeleton.entry;
        Converter {
            graph,
            current_block: entry,
            locals: LookupMap::new(),
            top_level,
            symbols,
            inlined_constants: LookupMap::new(),
            pure_constants,
            pure_definitions,
            effect_ids,
            current_span: None,
            output_sources: Vec::new(),
            binding_ids,
            extra_resource_declarations: Vec::new(),
            arenas,
        }
    }

    fn function_id(&self, symbol: SymbolId) -> crate::FunctionId {
        self.arenas.function_ids[&symbol]
    }

    /// Intern a pure node, attaching the current term's span (if any).
    /// Use in preference to `self.graph.intern_pure` so spans flow through.
    fn intern_pure(&mut self, op: PureOp, operands: SmallVec<[NodeId; 4]>, ty: Type<TypeName>) -> NodeId {
        if let Some(folded) = self.graph.try_algebraic_fold(&op, &operands, &ty) {
            return folded;
        }
        self.graph.intern_pure(op, operands, ty, self.current_span)
    }

    fn alloc_effect(&mut self) -> EffectToken {
        super::graph_ops::alloc_effect(self.effect_ids)
    }

    /// Set the return terminator on the current block.
    fn set_return(&mut self, result: Option<NodeId>) {
        self.graph.skeleton.blocks[self.current_block].term = SkeletonTerminator::Return(result);
    }

    // -- Entry-point emission helpers (thin delegations to `graph_ops`) --

    fn emit_storage_view(&mut self, binding: BindingRef, view_ty: Type<TypeName>) -> NodeId {
        let elem_ty = view_ty.elem_type().cloned().unwrap_or_else(|| view_ty.clone());
        let resource = self.arenas.resources.declare_host(binding, elem_ty, LogicalSize::Unspecified);
        super::graph_ops::intern_resource_view(&mut self.graph, resource, view_ty, self.current_span)
    }

    fn emit_storage_store(
        &mut self,
        view_nid: NodeId,
        index_nid: NodeId,
        value_nid: NodeId,
        elem_ty: Type<TypeName>,
    ) {
        let span = self.current_span;
        super::graph_ops::emit_storage_store(
            &mut self.graph,
            self.current_block,
            view_nid,
            index_nid,
            value_nid,
            elem_ty,
            self.effect_ids,
            span,
        );
    }

    /// Finish this body after all graph-owned metadata has been attached.
    fn into_graph(self) -> EGraph<Raw> {
        self.graph
    }

    // ========================================================================
    // Term conversion
    // ========================================================================

    fn convert_term(&mut self, term: &Term) -> Result<NodeId, ConvertError> {
        let ty = term.ty.clone();
        let saved_span = self.current_span;
        self.current_span = Some(term.span);
        let result = self.convert_term_kind(term, ty);
        self.current_span = saved_span;
        result
    }

    fn convert_term_kind(&mut self, term: &Term, ty: Type<TypeName>) -> Result<NodeId, ConvertError> {
        match &term.kind {
            // --- Literals ---
            TermKind::IntLit(s) => {
                let op = if matches!(&ty, Type::Constructed(TypeName::UInt(_), _)) {
                    PureOp::Uint(s.clone())
                } else {
                    PureOp::Int(s.clone())
                };
                Ok(self.intern_pure(op, smallvec![], ty))
            }
            TermKind::FloatLit(f) => Ok(self.intern_pure(PureOp::Float(f.to_string()), smallvec![], ty)),
            TermKind::BoolLit(b) => Ok(self.intern_pure(PureOp::Bool(*b), smallvec![], ty)),
            TermKind::UnitLit => Ok(self.intern_pure(PureOp::Unit, smallvec![], ty)),
            TermKind::Coerce { .. } => Err(ConvertError::Unsupported(
                "type coercion (:>) is not yet supported".into(),
            )),

            // --- Variables ---
            TermKind::Var(VarRef::Symbol(sym)) => self.convert_var(*sym, ty),
            // Catalog builtin reference: emit PureOp::Intrinsic directly,
            // bypassing the string-keyed lookup path in `convert_var`.
            TermKind::Var(VarRef::Builtin { id, overload_idx }) => Ok(self.intern_pure(
                PureOp::Intrinsic {
                    id: *id,
                    overload_idx: *overload_idx,
                },
                smallvec![],
                ty,
            )),

            // --- Let bindings (scope only, no instruction) ---
            TermKind::Let {
                name,
                name_ty: _,
                rhs,
                body,
            } => {
                let rhs_nid = self.convert_term(rhs)?;
                self.locals.insert(*name, rhs_nid);
                let result = self.convert_term(body)?;
                self.locals.remove(name);
                Ok(result)
            }

            // --- Extern ---
            TermKind::Extern(_) => Err(ConvertError::GraphError(
                "extern declaration reached expression conversion".into(),
            )),

            // --- If/else (Step 3) ---
            TermKind::If {
                cond,
                then_branch,
                else_branch,
            } => self.convert_if(cond, then_branch, else_branch, ty),

            // --- Application (Step 2 + 4) ---
            TermKind::App { func, args } => self.convert_app(func, args, ty),

            // --- Loops ---
            TermKind::Loop {
                loop_var,
                loop_var_ty,
                init,
                init_bindings,
                kind,
                body,
            } => self.convert_loop(*loop_var, loop_var_ty, init, init_bindings, kind, body, ty),

            // --- SOACs ---
            TermKind::Soac(soac) => self.convert_soac(soac, ty),

            // --- Array expressions ---
            TermKind::ArrayExpr(ae) => self.convert_array_expr(ae, ty),

            // --- Projection / construction operators ---
            TermKind::Tuple(parts) => {
                let operands: SmallVec<[NodeId; 4]> =
                    parts.iter().map(|p| self.convert_term(p)).collect::<Result<_, _>>()?;
                let n = operands.len();
                Ok(self.intern_pure(PureOp::Tuple(n), operands, ty))
            }
            TermKind::TupleProj { tuple, idx } => {
                let base = self.convert_term(tuple)?;
                Ok(self.intern_pure(PureOp::Project { index: *idx as u32 }, smallvec![base], ty))
            }
            TermKind::Index { array, index } => {
                let base = self.convert_term(array)?;
                let idx = self.convert_term(index)?;
                // View-variant arrays index via OpAccessChain into the
                // backing storage buffer — `Materialize + DynamicExtract`
                // (the path the materialize pass would generate for a
                // pure `Index`) tries to spill the view's `{offset,len}`
                // struct to a function-local array, which crashes the
                // SPIR-V backend. Emit `ViewIndex + Load` directly so
                // the side-effect pipeline handles it.
                let arr_ty = self.graph.nodes[base].ty.clone();
                let is_view =
                    arr_ty.array_variant().map(crate::types::is_array_variant_view).unwrap_or(false);
                if is_view {
                    let place_nid = self.intern_pure(PureOp::ViewIndex, smallvec![base, idx], ty.clone());
                    Ok(super::graph_ops::emit_load(
                        &mut self.graph,
                        self.current_block,
                        place_nid,
                        ty.clone(),
                        self.effect_ids,
                        self.current_span,
                    ))
                } else {
                    // Keep the semantic index explicit. EGIR residency planning
                    // decides whether its producer needs a storage handoff.
                    Ok(self.intern_pure(PureOp::Index, smallvec![base, idx], ty))
                }
            }
            TermKind::VecLit(parts) => {
                let operands: SmallVec<[NodeId; 4]> =
                    parts.iter().map(|p| self.convert_term(p)).collect::<Result<_, _>>()?;
                let n = operands.len();
                Ok(self.intern_pure(PureOp::Vector(n), operands, ty))
            }

            // --- Should not appear after defunctionalization ---
            TermKind::Lambda(_) => {
                panic!("ICE: bare Lambda in to_egir (should be lifted)")
            }
            TermKind::Closure(_) => {
                panic!("ICE: bare Closure in to_egir (closure calls should be lowered)")
            }
            TermKind::BinOp(_) | TermKind::UnOp(_) => {
                panic!("ICE: bare operator in to_egir (should be inside App)")
            }
        }
    }

    // ========================================================================
    // Compute output-route collection
    // ========================================================================

    /// Convert one logical output value, recursing through `If` and wrapping
    /// `Let`s so each producing leaf records its own `SlotSource`.
    ///
    /// `If`-shaped values fork at the EGIR level: the current block ends
    /// with `CondBranch`, each arm recursively converts the same route
    /// against its branch, and both arms terminate with
    /// `Branch(merge)` carrying no result args. There's nothing to
    /// merge; output realization later gives both routes the same destination.
    ///
    /// `Let { x = rhs, body }` wrapping an output value (e.g. `let n =
    /// length(xs) in if c then map_using_n else map_using_n`) binds
    /// `rhs` at the current block, then recurses on `body` — the
    /// binding is visible in both `If` arms when the body forks.
    ///
    /// Non-control-flow values are converted normally; a single
    /// `SlotSource { block: self.current_block, value: <converted> }`
    /// is pushed to `slot_sources_accum[slot_index]`.
    fn convert_output_source(&mut self, slot_index: usize, value: &Term) -> Result<(), ConvertError> {
        use crate::flow::ControlHeader;
        match &value.kind {
            TermKind::Let {
                name,
                name_ty: _,
                rhs,
                body,
            } => {
                // Bind `rhs` at the current block (it produces a value;
                // for unit-valued RHS like a side-effect call, the
                // resulting NodeId is just the Unit constant). The
                // binding survives the branch fork in `body`.
                let rhs_nid = self.convert_term(rhs)?;
                self.locals.insert(*name, rhs_nid);
                self.convert_output_source(slot_index, body)
            }
            TermKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let cond_nid = self.convert_term(cond)?;

                let then_block = self.graph.skeleton.create_block();
                let else_block = self.graph.skeleton.create_block();
                let merge_block = self.graph.skeleton.create_block();

                self.graph.skeleton.blocks[self.current_block].control_header =
                    Some(ControlHeader::Selection { merge: merge_block });
                self.graph.skeleton.blocks[self.current_block].term = SkeletonTerminator::CondBranch {
                    cond: cond_nid,
                    then_target: then_block,
                    then_args: vec![],
                    else_target: else_block,
                    else_args: vec![],
                };

                self.current_block = then_block;
                self.convert_output_source(slot_index, then_branch)?;
                self.graph.skeleton.blocks[self.current_block].term = SkeletonTerminator::Branch {
                    target: merge_block,
                    args: vec![],
                };

                self.current_block = else_block;
                self.convert_output_source(slot_index, else_branch)?;
                self.graph.skeleton.blocks[self.current_block].term = SkeletonTerminator::Branch {
                    target: merge_block,
                    args: vec![],
                };

                self.current_block = merge_block;
                Ok(())
            }
            _ => {
                let value_nid = self.convert_term(value)?;
                while self.output_sources.len() <= slot_index {
                    self.output_sources.push(Vec::new());
                }
                self.output_sources[slot_index].push(crate::egir::program::SlotSource {
                    block: self.current_block,
                    value: value_nid,
                });
                Ok(())
            }
        }
    }

    fn convert_compute_outputs(
        &mut self,
        term: &Term,
        output_count: usize,
    ) -> Result<Option<NodeId>, ConvertError> {
        match &term.kind {
            TermKind::Let { name, rhs, body, .. } => {
                let rhs_nid = self.convert_term(rhs)?;
                self.locals.insert(*name, rhs_nid);
                self.convert_compute_outputs(body, output_count)
            }
            TermKind::Tuple(values) if values.len() == output_count => {
                for (slot, value) in values.iter().enumerate() {
                    self.convert_output_source(slot, value)?;
                }
                Ok(None)
            }
            _ if output_count == 1 => {
                self.convert_output_source(0, term)?;
                Ok(None)
            }
            _ => self.convert_term(term).map(Some),
        }
    }

    // ========================================================================
    // Variable resolution
    // ========================================================================

    fn convert_var(&mut self, sym: SymbolId, ty: Type<TypeName>) -> Result<NodeId, ConvertError> {
        if let Some(&nid) = self.locals.get(&sym) {
            return Ok(nid);
        }
        if let Some(&nid) = self.inlined_constants.get(&sym) {
            return Ok(nid);
        }
        if self.pure_constants.contains(&sym) {
            let global = self.arenas.global_ids[&sym];
            return Ok(self.intern_pure(PureOp::Global(global), smallvec![], ty));
        }
        if let Some(def) = self.top_level.get(&sym).filter(|definition| definition.arity == 0).copied() {
            let body = def.body.clone();
            let nid = self.convert_term(&body)?;
            self.inlined_constants.insert(sym, nid);
            return Ok(nid);
        }
        let name = symbol_name(self.symbols, sym)?;

        Err(ConvertError::Unsupported(format!(
            "callable `{name}` used as a first-class value after defunctionalization"
        )))
    }

    // ========================================================================
    // Application
    // ========================================================================

    fn convert_app(
        &mut self,
        func: &Term,
        args: &[Term],
        ty: Type<TypeName>,
    ) -> Result<NodeId, ConvertError> {
        match &func.kind {
            TermKind::BinOp(op) => {
                let lhs = self.convert_term(&args[0])?;
                let rhs = self.convert_term(&args[1])?;
                Ok(self.intern_pure(PureOp::BinOp(op.op.clone()), smallvec![lhs, rhs], ty))
            }
            TermKind::UnOp(op) => {
                let operand = self.convert_term(&args[0])?;
                Ok(self.intern_pure(PureOp::UnaryOp(op.op.clone()), smallvec![operand], ty))
            }
            TermKind::Var(VarRef::Symbol(sym)) => self.convert_named_app(*sym, args, ty),
            TermKind::Var(VarRef::Builtin { id, overload_idx }) => {
                // Catalog-resolved builtin call. Most catalog entries lower
                // to a pure `PureOp::Intrinsic` and the backend dispatches
                // on `catalog.get(id).overloads()[overload_idx].lowering`.
                // A small set of entries (StorageIndex, StorageStore) is
                // side-effectful and must emit a Load/Store side effect at
                // EGIR conversion — dispatch by id against
                // `catalog.known()` so we never reflect on the surface
                // name.
                let known = catalog().known();
                if *id == known.storage_index && args.len() == 3 {
                    self.lower_storage_index(args, ty)
                } else if *id == known.storage_store && args.len() == 4 {
                    self.lower_storage_store(args)
                } else if *id == known.storage_len && args.len() == 2 {
                    let binding = literal_binding(args, "storage_len").ok();
                    if let Some(resource) = binding.map(|binding| self.arenas.resources.host_id(binding)) {
                        Ok(self.intern_pure(
                            PureOp::ResourceLen(SemanticResourceRef(resource)),
                            smallvec![],
                            ty,
                        ))
                    } else {
                        let arg_nids: SmallVec<[NodeId; 4]> =
                            args.iter().map(|a| self.convert_term(a)).collect::<Result<_, _>>()?;
                        Ok(self.intern_pure(
                            PureOp::Intrinsic {
                                id: *id,
                                overload_idx: *overload_idx,
                            },
                            arg_nids,
                            ty,
                        ))
                    }
                } else if *id == known.image_with && args.len() == 3 {
                    self.lower_image_with(args, ty)
                } else if *id == known.image_load && args.len() == 2 {
                    let binding = crate::types::storage_image_buffer(&args[0].ty).ok_or_else(|| {
                        ConvertError::GraphError(
                            "image_load operand has no concrete storage-image binding after monomorphization"
                                .into(),
                        )
                    })?;
                    let resource = SemanticResourceRef(self.arenas.resources.host_id(binding));
                    let coord = self.convert_term(&args[1])?;
                    Ok(self.intern_pure(PureOp::StorageImageLoad(resource), smallvec![coord], ty))
                } else {
                    let arg_nids: SmallVec<[NodeId; 4]> =
                        args.iter().map(|a| self.convert_term(a)).collect::<Result<_, _>>()?;
                    Ok(self.intern_pure(
                        PureOp::Intrinsic {
                            id: *id,
                            overload_idx: *overload_idx,
                        },
                        arg_nids,
                        ty,
                    ))
                }
            }
            _ => {
                // General application: convert func, then call
                let _func_nid = self.convert_term(func)?;
                let _arg_nids: Vec<NodeId> =
                    args.iter().map(|a| self.convert_term(a)).collect::<Result<_, _>>()?;
                // TODO: emit Call side effect
                Err(ConvertError::Unsupported("general application".into()))
            }
        }
    }

    fn convert_named_app(
        &mut self,
        symbol: SymbolId,
        args: &[Term],
        ty: Type<TypeName>,
    ) -> Result<NodeId, ConvertError> {
        if let Some(def) = self.top_level.get(&symbol) {
            if def.arity == args.len() {
                let operands: SmallVec<[NodeId; 4]> =
                    args.iter().map(|argument| self.convert_term(argument)).collect::<Result<_, _>>()?;
                return Ok(self.emit_named_call(symbol, operands, ty));
            }
        }
        let name = symbol_name(self.symbols, symbol)?;
        Err(ConvertError::Unsupported(format!(
            "application of `{name}` with the wrong arity"
        )))
    }

    fn emit_named_call(
        &mut self,
        symbol: SymbolId,
        operands: SmallVec<[NodeId; 4]>,
        ty: Type<TypeName>,
    ) -> NodeId {
        let function = self.function_id(symbol);
        if self.pure_definitions.contains(&symbol) {
            return self.intern_pure(PureOp::Call(function), operands, ty);
        }

        let result = self.graph.alloc_side_effect_result(ty);
        let effect_in = self.alloc_effect();
        let effect_out = self.alloc_effect();
        self.graph.skeleton.blocks[self.current_block].side_effects.push(SideEffect {
            kind: SideEffectKind::Effect(EffectOp::Op {
                tag: crate::op::OpTag::Call(function),
            }),
            operand_nodes: operands,
            result: Some(result),
            effects: Some((effect_in, effect_out)),
            span: self.current_span,
        });
        result
    }

    // ========================================================================
    // Side-effectful intrinsic helpers — called from the Builtin-arm
    // dispatch on `BuiltinLowering::Intrinsic(StorageIndex|StorageStore)`.
    // ========================================================================

    fn lower_storage_index(&mut self, args: &[Term], ty: Type<TypeName>) -> Result<NodeId, ConvertError> {
        let binding = literal_binding(args, "_w_intrinsic_storage_index")?;
        let index_nid = self.convert_term(&args[2])?;
        let view_nid = self.emit_storage_view(binding, ty.clone());
        let place_nid = self.intern_pure(PureOp::ViewIndex, smallvec![view_nid, index_nid], ty.clone());
        Ok(super::graph_ops::emit_load(
            &mut self.graph,
            self.current_block,
            place_nid,
            ty.clone(),
            self.effect_ids,
            self.current_span,
        ))
    }

    fn lower_storage_store(&mut self, args: &[Term]) -> Result<NodeId, ConvertError> {
        let binding = literal_binding(args, "_w_intrinsic_storage_store")?;
        let index_nid = self.convert_term(&args[2])?;
        let value_nid = self.convert_term(&args[3])?;
        let value_ty = args[3].ty.clone();
        let view_nid = self.emit_storage_view(binding, value_ty.clone());
        self.emit_storage_store(view_nid, index_nid, value_nid, value_ty);
        let unit_ty = Type::Constructed(TypeName::Unit, vec![]);
        Ok(self.intern_pure(PureOp::Unit, smallvec![], unit_ty))
    }

    /// Convert `img with [xy] = rgba` into the same image-write side effect as
    /// the backend storage-image store, returning a compile-time-only placeholder for the next
    /// linear image handle. The image handle itself has no runtime payload; the
    /// concrete descriptor binding is carried by `args[0].ty`.
    fn lower_image_with(&mut self, args: &[Term], ty: Type<TypeName>) -> Result<NodeId, ConvertError> {
        let binding = crate::types::storage_image_buffer(&args[0].ty).ok_or_else(|| {
            ConvertError::GraphError(
                "storage-image update operand has no concrete storage-image binding after monomorphization"
                    .into(),
            )
        })?;
        let resource = SemanticResourceRef(self.arenas.resources.host_id(binding));
        let arg_nids: SmallVec<[NodeId; 4]> =
            args[1..].iter().map(|a| self.convert_term(a)).collect::<Result<_, _>>()?;
        let unit_ty = Type::Constructed(TypeName::Unit, vec![]);
        let effect_result = self.graph.alloc_side_effect_result(unit_ty);
        let effect_in = self.alloc_effect();
        let effect_out = self.alloc_effect();
        self.graph.skeleton.blocks[self.current_block].side_effects.push(SideEffect {
            kind: SideEffectKind::Effect(EffectOp::Op {
                tag: crate::op::OpTag::StorageImageStore(resource),
            }),
            operand_nodes: arg_nids,
            result: Some(effect_result),
            effects: Some((effect_in, effect_out)),
            span: self.current_span,
        });
        Ok(self.intern_pure(PureOp::Unit, smallvec![], ty))
    }

    // ========================================================================
    // If/else
    // ========================================================================

    fn convert_if(
        &mut self,
        cond: &Term,
        then_branch: &Term,
        else_branch: &Term,
        ty: Type<TypeName>,
    ) -> Result<NodeId, ConvertError> {
        let cond_nid = self.convert_term(cond)?;

        let then_block = self.graph.skeleton.create_block();
        let else_block = self.graph.skeleton.create_block();
        let merge_block = self.graph.skeleton.create_block();

        let storage_image_result = matches!(ty, Type::Constructed(TypeName::StorageTexture, _));
        let result_nid = if storage_image_result {
            None
        } else {
            let result_nid = self.graph.add_block_param(merge_block, ty.clone());
            Some(result_nid)
        };

        // Selection header for SPIR-V structured control flow.
        self.graph.skeleton.blocks[self.current_block].control_header =
            Some(ControlHeader::Selection { merge: merge_block });

        // Terminate current block with CondBranch.
        self.graph.skeleton.blocks[self.current_block].term = SkeletonTerminator::CondBranch {
            cond: cond_nid,
            then_target: then_block,
            then_args: vec![],
            else_target: else_block,
            else_args: vec![],
        };

        // Then branch.
        self.current_block = then_block;
        let then_result = self.convert_term(then_branch)?;
        self.graph.skeleton.blocks[self.current_block].term = SkeletonTerminator::Branch {
            target: merge_block,
            args: result_nid.map(|_| then_result).into_iter().collect(),
        };

        // Else branch.
        self.current_block = else_block;
        let else_result = self.convert_term(else_branch)?;
        self.graph.skeleton.blocks[self.current_block].term = SkeletonTerminator::Branch {
            target: merge_block,
            args: result_nid.map(|_| else_result).into_iter().collect(),
        };

        // Continue from merge.
        self.current_block = merge_block;
        Ok(result_nid.unwrap_or_else(|| self.intern_pure(PureOp::Unit, smallvec![], ty)))
    }

    // ========================================================================
    // Loops
    // ========================================================================

    fn convert_loop(
        &mut self,
        loop_var: SymbolId,
        loop_var_ty: &Type<TypeName>,
        init: &Term,
        init_bindings: &[(SymbolId, Type<TypeName>, Term)],
        kind: &LoopKind,
        body: &Term,
        _result_ty: Type<TypeName>,
    ) -> Result<NodeId, ConvertError> {
        if matches!(loop_var_ty, Type::Constructed(TypeName::StorageTexture, _)) {
            return match kind {
                LoopKind::While { cond } => self.convert_storage_image_while_loop(
                    loop_var,
                    loop_var_ty,
                    init,
                    init_bindings,
                    cond,
                    body,
                ),
                LoopKind::ForRange { var, var_ty, bound } => self.convert_storage_image_for_range_loop(
                    loop_var,
                    loop_var_ty,
                    init,
                    init_bindings,
                    *var,
                    var_ty,
                    bound,
                    body,
                ),
                LoopKind::For { var, var_ty, iter } => self.convert_storage_image_for_in_loop(
                    loop_var,
                    loop_var_ty,
                    init,
                    init_bindings,
                    *var,
                    var_ty,
                    iter,
                    body,
                ),
            };
        }
        match kind {
            LoopKind::While { cond } => {
                self.convert_while_loop(loop_var, loop_var_ty, init, init_bindings, cond, body)
            }
            LoopKind::ForRange { var, var_ty, bound } => self.convert_for_range_loop(
                loop_var,
                loop_var_ty,
                init,
                init_bindings,
                *var,
                var_ty,
                bound,
                body,
            ),
            LoopKind::For { var, var_ty, iter } => self.convert_for_in_loop(
                loop_var,
                loop_var_ty,
                init,
                init_bindings,
                *var,
                var_ty,
                iter,
                body,
            ),
        }
    }

    fn storage_image_placeholder(&mut self, ty: &Type<TypeName>) -> NodeId {
        self.intern_pure(PureOp::Unit, smallvec![], ty.clone())
    }

    fn bind_storage_image_loop_var(
        &mut self,
        loop_var: SymbolId,
        loop_var_ty: &Type<TypeName>,
        init_bindings: &[(SymbolId, Type<TypeName>, Term)],
    ) -> Result<(), ConvertError> {
        let handle = self.storage_image_placeholder(loop_var_ty);
        self.locals.insert(loop_var, handle);
        for (sym, _ty, expr) in init_bindings {
            let val = self.convert_term(expr)?;
            self.locals.insert(*sym, val);
        }
        Ok(())
    }

    fn unbind_loop_vars(&mut self, loop_var: SymbolId, init_bindings: &[(SymbolId, Type<TypeName>, Term)]) {
        self.locals.remove(&loop_var);
        for (sym, _, _) in init_bindings {
            self.locals.remove(sym);
        }
    }

    fn convert_storage_image_while_loop(
        &mut self,
        loop_var: SymbolId,
        loop_var_ty: &Type<TypeName>,
        init: &Term,
        init_bindings: &[(SymbolId, Type<TypeName>, Term)],
        cond: &Term,
        body: &Term,
    ) -> Result<NodeId, ConvertError> {
        let header = self.graph.skeleton.create_block();
        let body_block = self.graph.skeleton.create_block();
        let exit = self.graph.skeleton.create_block();

        self.graph.skeleton.blocks[header].control_header = Some(ControlHeader::Loop {
            merge: exit,
            continue_block: body_block,
        });

        let _init_nid = self.convert_term(init)?;
        self.graph.skeleton.blocks[self.current_block].term = SkeletonTerminator::Branch {
            target: header,
            args: vec![],
        };

        self.current_block = header;
        self.bind_storage_image_loop_var(loop_var, loop_var_ty, init_bindings)?;
        let cond_nid = self.convert_term(cond)?;
        self.graph.skeleton.blocks[header].term = SkeletonTerminator::CondBranch {
            cond: cond_nid,
            then_target: body_block,
            then_args: vec![],
            else_target: exit,
            else_args: vec![],
        };

        self.current_block = body_block;
        let _new_handle = self.convert_term(body)?;
        self.graph.skeleton.blocks[self.current_block].term = SkeletonTerminator::Branch {
            target: header,
            args: vec![],
        };

        self.current_block = exit;
        self.unbind_loop_vars(loop_var, init_bindings);
        Ok(self.storage_image_placeholder(loop_var_ty))
    }

    fn convert_storage_image_for_range_loop(
        &mut self,
        loop_var: SymbolId,
        loop_var_ty: &Type<TypeName>,
        init: &Term,
        init_bindings: &[(SymbolId, Type<TypeName>, Term)],
        index_var: SymbolId,
        _index_var_ty: &Type<TypeName>,
        bound: &Term,
        body: &Term,
    ) -> Result<NodeId, ConvertError> {
        let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
        let bool_ty = Type::Constructed(TypeName::Bool, vec![]);
        let header = self.graph.skeleton.create_block();
        let body_block = self.graph.skeleton.create_block();
        let exit = self.graph.skeleton.create_block();
        let idx_nid = self.graph.add_block_param(header, i32_ty.clone());

        self.graph.skeleton.blocks[header].control_header = Some(ControlHeader::Loop {
            merge: exit,
            continue_block: body_block,
        });

        let _init_nid = self.convert_term(init)?;
        let bound_nid = self.convert_term(bound)?;
        let zero = self.intern_pure(PureOp::Int("0".into()), smallvec![], i32_ty.clone());
        self.graph.skeleton.blocks[self.current_block].term = SkeletonTerminator::Branch {
            target: header,
            args: vec![zero],
        };

        self.current_block = header;
        self.bind_storage_image_loop_var(loop_var, loop_var_ty, init_bindings)?;
        self.locals.insert(index_var, idx_nid);
        let cond_nid = self.intern_pure(
            PureOp::BinOp(BinaryOperator::Less),
            smallvec![idx_nid, bound_nid],
            bool_ty,
        );
        self.graph.skeleton.blocks[header].term = SkeletonTerminator::CondBranch {
            cond: cond_nid,
            then_target: body_block,
            then_args: vec![],
            else_target: exit,
            else_args: vec![],
        };

        self.current_block = body_block;
        let _new_handle = self.convert_term(body)?;
        let one = self.intern_pure(PureOp::Int("1".into()), smallvec![], i32_ty.clone());
        let next_i = self.intern_pure(
            PureOp::BinOp(BinaryOperator::Add),
            smallvec![idx_nid, one],
            i32_ty,
        );
        self.graph.skeleton.blocks[self.current_block].term = SkeletonTerminator::Branch {
            target: header,
            args: vec![next_i],
        };

        self.current_block = exit;
        self.unbind_loop_vars(loop_var, init_bindings);
        self.locals.remove(&index_var);
        Ok(self.storage_image_placeholder(loop_var_ty))
    }

    fn convert_storage_image_for_in_loop(
        &mut self,
        loop_var: SymbolId,
        loop_var_ty: &Type<TypeName>,
        init: &Term,
        init_bindings: &[(SymbolId, Type<TypeName>, Term)],
        elem_var: SymbolId,
        elem_ty: &Type<TypeName>,
        iter: &Term,
        body: &Term,
    ) -> Result<NodeId, ConvertError> {
        let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
        let bool_ty = Type::Constructed(TypeName::Bool, vec![]);
        let header = self.graph.skeleton.create_block();
        let body_block = self.graph.skeleton.create_block();
        let exit = self.graph.skeleton.create_block();
        let idx_nid = self.graph.add_block_param(header, i32_ty.clone());

        self.graph.skeleton.blocks[header].control_header = Some(ControlHeader::Loop {
            merge: exit,
            continue_block: body_block,
        });

        let _init_nid = self.convert_term(init)?;
        let iter_nid = self.convert_term(iter)?;
        let len_nid = self.intern_pure(
            PureOp::Intrinsic {
                id: catalog().known().length,
                overload_idx: 0,
            },
            smallvec![iter_nid],
            i32_ty.clone(),
        );
        let zero = self.intern_pure(PureOp::Int("0".into()), smallvec![], i32_ty.clone());
        self.graph.skeleton.blocks[self.current_block].term = SkeletonTerminator::Branch {
            target: header,
            args: vec![zero],
        };

        self.current_block = header;
        self.bind_storage_image_loop_var(loop_var, loop_var_ty, init_bindings)?;
        let cond_nid = self.intern_pure(
            PureOp::BinOp(BinaryOperator::Less),
            smallvec![idx_nid, len_nid],
            bool_ty,
        );
        self.graph.skeleton.blocks[header].term = SkeletonTerminator::CondBranch {
            cond: cond_nid,
            then_target: body_block,
            then_args: vec![],
            else_target: exit,
            else_args: vec![],
        };

        self.current_block = body_block;
        let elem_nid = self.intern_pure(PureOp::Index, smallvec![iter_nid, idx_nid], elem_ty.clone());
        self.locals.insert(elem_var, elem_nid);
        let _new_handle = self.convert_term(body)?;
        let one = self.intern_pure(PureOp::Int("1".into()), smallvec![], i32_ty.clone());
        let next_i = self.intern_pure(
            PureOp::BinOp(BinaryOperator::Add),
            smallvec![idx_nid, one],
            i32_ty,
        );
        self.graph.skeleton.blocks[self.current_block].term = SkeletonTerminator::Branch {
            target: header,
            args: vec![next_i],
        };

        self.current_block = exit;
        self.unbind_loop_vars(loop_var, init_bindings);
        self.locals.remove(&elem_var);
        Ok(self.storage_image_placeholder(loop_var_ty))
    }

    fn convert_while_loop(
        &mut self,
        loop_var: SymbolId,
        loop_var_ty: &Type<TypeName>,
        init: &Term,
        init_bindings: &[(SymbolId, Type<TypeName>, Term)],
        cond: &Term,
        body: &Term,
    ) -> Result<NodeId, ConvertError> {
        let acc_ty = loop_var_ty.clone();

        // Create blocks: header, body, exit
        let header = self.graph.skeleton.create_block();
        let body_block = self.graph.skeleton.create_block();
        let exit = self.graph.skeleton.create_block();

        // Header has acc param; exit has result param
        let acc_nid = self.graph.add_block_param(header, acc_ty.clone());
        let result_nid = self.graph.add_block_param(exit, acc_ty.clone());

        // Loop header for SPIR-V
        self.graph.skeleton.blocks[header].control_header = Some(ControlHeader::Loop {
            merge: exit,
            continue_block: body_block,
        });

        // Init → header
        let init_nid = self.convert_term(init)?;
        self.graph.skeleton.blocks[self.current_block].term = SkeletonTerminator::Branch {
            target: header,
            args: vec![init_nid],
        };

        // Header: bind loop_var, process init_bindings, check cond
        self.current_block = header;
        self.locals.insert(loop_var, acc_nid);
        for (sym, _ty, expr) in init_bindings {
            let val = self.convert_term(expr)?;
            self.locals.insert(*sym, val);
        }
        let cond_nid = self.convert_term(cond)?;
        self.graph.skeleton.blocks[header].term = SkeletonTerminator::CondBranch {
            cond: cond_nid,
            then_target: body_block,
            then_args: vec![],
            else_target: exit,
            else_args: vec![acc_nid],
        };

        // Body: convert body, branch back to header
        self.current_block = body_block;
        let new_acc = self.convert_term(body)?;
        self.graph.skeleton.blocks[self.current_block].term = SkeletonTerminator::Branch {
            target: header,
            args: vec![new_acc],
        };

        // Exit
        self.current_block = exit;
        self.locals.remove(&loop_var);
        for (sym, _, _) in init_bindings {
            self.locals.remove(sym);
        }
        Ok(result_nid)
    }

    fn convert_for_range_loop(
        &mut self,
        loop_var: SymbolId,
        loop_var_ty: &Type<TypeName>,
        init: &Term,
        init_bindings: &[(SymbolId, Type<TypeName>, Term)],
        index_var: SymbolId,
        _index_var_ty: &Type<TypeName>,
        bound: &Term,
        body: &Term,
    ) -> Result<NodeId, ConvertError> {
        let acc_ty = loop_var_ty.clone();
        let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
        let bool_ty = Type::Constructed(TypeName::Bool, vec![]);

        // Create blocks
        let header = self.graph.skeleton.create_block();
        let body_block = self.graph.skeleton.create_block();
        let exit = self.graph.skeleton.create_block();

        // Header has (acc, index) params; exit has result param
        let acc_nid = self.graph.add_block_param(header, acc_ty.clone());
        let idx_nid = self.graph.add_block_param(header, i32_ty.clone());
        let result_nid = self.graph.add_block_param(exit, acc_ty.clone());

        self.graph.skeleton.blocks[header].control_header = Some(ControlHeader::Loop {
            merge: exit,
            continue_block: body_block,
        });

        // Init → header with (init, 0)
        let init_nid = self.convert_term(init)?;
        let bound_nid = self.convert_term(bound)?;
        let zero = self.intern_pure(PureOp::Int("0".into()), smallvec![], i32_ty.clone());
        self.graph.skeleton.blocks[self.current_block].term = SkeletonTerminator::Branch {
            target: header,
            args: vec![init_nid, zero],
        };

        // Header: bind vars, check i < bound
        self.current_block = header;
        self.locals.insert(loop_var, acc_nid);
        self.locals.insert(index_var, idx_nid);
        for (sym, _ty, expr) in init_bindings {
            let val = self.convert_term(expr)?;
            self.locals.insert(*sym, val);
        }
        let cond_nid = self.intern_pure(
            PureOp::BinOp(BinaryOperator::Less),
            smallvec![idx_nid, bound_nid],
            bool_ty,
        );
        self.graph.skeleton.blocks[header].term = SkeletonTerminator::CondBranch {
            cond: cond_nid,
            then_target: body_block,
            then_args: vec![],
            else_target: exit,
            else_args: vec![acc_nid],
        };

        // Body: convert body, increment index, branch back
        self.current_block = body_block;
        let new_acc = self.convert_term(body)?;
        let one = self.intern_pure(PureOp::Int("1".into()), smallvec![], i32_ty.clone());
        let next_i = self.intern_pure(
            PureOp::BinOp(BinaryOperator::Add),
            smallvec![idx_nid, one],
            i32_ty,
        );
        self.graph.skeleton.blocks[self.current_block].term = SkeletonTerminator::Branch {
            target: header,
            args: vec![new_acc, next_i],
        };

        // Exit
        self.current_block = exit;
        self.locals.remove(&loop_var);
        self.locals.remove(&index_var);
        for (sym, _, _) in init_bindings {
            self.locals.remove(sym);
        }
        Ok(result_nid)
    }

    fn convert_for_in_loop(
        &mut self,
        loop_var: SymbolId,
        loop_var_ty: &Type<TypeName>,
        init: &Term,
        init_bindings: &[(SymbolId, Type<TypeName>, Term)],
        elem_var: SymbolId,
        elem_ty: &Type<TypeName>,
        iter: &Term,
        body: &Term,
    ) -> Result<NodeId, ConvertError> {
        // For-in is like for-range but indexes into the iterator.
        // TODO: SoA-aware soa_length / soa_index
        let acc_ty = loop_var_ty.clone();
        let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
        let bool_ty = Type::Constructed(TypeName::Bool, vec![]);

        let header = self.graph.skeleton.create_block();
        let body_block = self.graph.skeleton.create_block();
        let exit = self.graph.skeleton.create_block();

        let acc_nid = self.graph.add_block_param(header, acc_ty.clone());
        let idx_nid = self.graph.add_block_param(header, i32_ty.clone());
        let result_nid = self.graph.add_block_param(exit, acc_ty.clone());

        self.graph.skeleton.blocks[header].control_header = Some(ControlHeader::Loop {
            merge: exit,
            continue_block: body_block,
        });

        // Init
        let init_nid = self.convert_term(init)?;
        let iter_nid = self.convert_term(iter)?;

        // Length intrinsic. PureOp::UnaryOp keys by op-name string;
        // the catalog-internal `_w_intrinsic_length` is the agreed
        // string the lowering layer dispatches on.
        let len_nid = self.intern_pure(
            PureOp::Intrinsic {
                id: catalog().known().length,
                overload_idx: 0,
            },
            smallvec![iter_nid],
            i32_ty.clone(),
        );
        let zero = self.intern_pure(PureOp::Int("0".into()), smallvec![], i32_ty.clone());
        self.graph.skeleton.blocks[self.current_block].term = SkeletonTerminator::Branch {
            target: header,
            args: vec![init_nid, zero],
        };

        // Header
        self.current_block = header;
        self.locals.insert(loop_var, acc_nid);
        for (sym, _ty, expr) in init_bindings {
            let val = self.convert_term(expr)?;
            self.locals.insert(*sym, val);
        }
        let cond_nid = self.intern_pure(
            PureOp::BinOp(BinaryOperator::Less),
            smallvec![idx_nid, len_nid],
            bool_ty,
        );
        self.graph.skeleton.blocks[header].term = SkeletonTerminator::CondBranch {
            cond: cond_nid,
            then_target: body_block,
            then_args: vec![],
            else_target: exit,
            else_args: vec![acc_nid],
        };

        // Body: index into iterator, bind elem_var
        self.current_block = body_block;
        let elem_nid = self.intern_pure(PureOp::Index, smallvec![iter_nid, idx_nid], elem_ty.clone());
        self.locals.insert(elem_var, elem_nid);

        let new_acc = self.convert_term(body)?;
        let one = self.intern_pure(PureOp::Int("1".into()), smallvec![], i32_ty.clone());
        let next_i = self.intern_pure(
            PureOp::BinOp(BinaryOperator::Add),
            smallvec![idx_nid, one],
            i32_ty,
        );
        self.graph.skeleton.blocks[self.current_block].term = SkeletonTerminator::Branch {
            target: header,
            args: vec![new_acc, next_i],
        };

        // Exit
        self.current_block = exit;
        self.locals.remove(&loop_var);
        self.locals.remove(&elem_var);
        for (sym, _, _) in init_bindings {
            self.locals.remove(sym);
        }
        Ok(result_nid)
    }

    // ========================================================================
    // SOACs
    // ========================================================================

    fn convert_soac(&mut self, soac: &SoacOp, ty: Type<TypeName>) -> Result<NodeId, ConvertError> {
        match soac {
            SoacOp::Map {
                lam,
                inputs,
                destination,
            } => self.convert_soac_map(lam, inputs, (*destination).into(), ty),
            SoacOp::Reduce { op, ne, input, .. } => self.convert_soac_reduce(op, ne, input, ty),
            SoacOp::Scan {
                op,
                ne,
                input,
                destination,
            } => self.convert_soac_scan(op, ne, input, (*destination).into(), ty),
            SoacOp::Filter {
                pred,
                input,
                destination,
            } => self.convert_soac_filter(pred, input, (*destination).into(), ty),
            SoacOp::Scatter { dest, lam, inputs } => self.convert_soac_scatter(dest, lam, inputs, ty),
            SoacOp::BucketScatter {
                dest,
                lam,
                inputs,
                input_dimensions,
                domain_rank,
            } => self.convert_soac_bucket_scatter(dest, lam, inputs, input_dimensions, *domain_rank, ty),
            SoacOp::ReduceByIndex {
                dest,
                op,
                ne,
                indices,
                values,
            } => self.convert_soac_reduce_by_index(dest, op, ne, indices, values, ty),
        }
    }

    fn lambda_fn_symbol(&self, lam: &Lambda) -> Result<SymbolId, ConvertError> {
        match &lam.body.kind {
            TermKind::Var(VarRef::Symbol(symbol)) => Ok(*symbol),
            _ => Err(ConvertError::GraphError(
                "SOAC lambda body should be a function reference post-defunc".into(),
            )),
        }
    }

    /// Emit a SOAC placeholder as a side effect in the skeleton. Returns the
    /// result NodeId that `soac_expand` will rebind during expansion.
    fn emit_soac(
        &mut self,
        soac: Soac<Raw>,
        operands: SmallVec<[NodeId; 4]>,
        ty: Type<TypeName>,
    ) -> NodeId {
        let span = self.current_span;
        super::graph_ops::emit_pending_soac(
            &mut self.graph,
            self.current_block,
            (),
            soac,
            operands,
            ty,
            self.effect_ids,
            span,
        )
    }

    fn convert_soac_map(
        &mut self,
        sb: &SoacBody,
        inputs: &[ArrayExpr],
        destination: SoacDestination,
        result_ty: Type<TypeName>,
    ) -> Result<NodeId, ConvertError> {
        let f_symbol = self.lambda_fn_symbol(&sb.lam)?;
        let capture_nids: Vec<NodeId> =
            sb.data.captures.iter().map(|(_, _, t)| self.convert_term(t)).collect::<Result<_, _>>()?;
        let input_nids: Vec<NodeId> =
            inputs.iter().map(|ae| self.convert_array_expr_value(ae)).collect::<Result<_, _>>()?;
        let input_arr_types: Vec<Type<TypeName>> =
            inputs.iter().zip(input_nids.iter()).map(|(ae, nid)| self.value_array_type(*nid, ae)).collect();
        let input_elem_types: Vec<Type<TypeName>> = input_arr_types
            .iter()
            .zip(inputs.iter())
            .map(|(ty, ae)| self.value_elem_type(ty, ae))
            .collect();
        let output_elem_ty = if result_ty.is_array() {
            result_ty.elem_type().expect("Array has elem").clone()
        } else if as_soa_tuple(&result_ty).is_some() {
            // After `tlc::soa`, the map's output `[N](A, B)` becomes a
            // SoA tuple `([N]A, [N]B)`. The per-iteration element type
            // is the corresponding tuple-of-elements `(A, B)`.
            soac_element_type(&result_ty)
        } else if !input_elem_types.is_empty() {
            input_elem_types[0].clone()
        } else {
            return Err(ConvertError::GraphError(
                "map: cannot determine output elem type".into(),
            ));
        };

        // Operands carry positional data flow only; captures live on the
        // `SegBody` below.
        let mut operands: SmallVec<[NodeId; 4]> = SmallVec::new();
        operands.extend_from_slice(&input_nids);

        // Emit as a singleton Screma + project field 0. For consuming
        // (`InputBuffer`) map the result aliases the input, so the
        // Project's type must match the input view's type (View
        // variant + buffer) rather than the TLC-default `result_ty`
        // (Composite variant with NoBuffer). Mirrors the same handling
        // in `convert_soac_scan` below — without it the SPIR-V backend
        // panics trying to lower a `Composite[Variable, NoBuffer]`
        // array type that survives because the consumer-side Project
        // takes the TLC logical type even when the runtime tuple
        // carries a View.
        // A non-in-place `map` is shape-preserving — inherit the input's
        // representation when `result_ty` carries an unresolved `Skolem` size
        // (see `shape_preserving_result_ty`); otherwise keep `result_ty`.
        let project_ty = if destination.is_input_buffer() {
            input_arr_types[0].clone()
        } else {
            input_arr_types
                .first()
                .and_then(|inp| shape_preserving_result_ty(inp, &output_elem_ty, &result_ty))
                .unwrap_or_else(|| result_ty.clone())
        };
        let tuple_ty = Type::Constructed(TypeName::Tuple(1), vec![project_ty.clone()]);

        let map_region = self.function_id(f_symbol);
        let screma_nid = self.emit_soac(
            Soac::Screma(screma::Op {
                inputs: input_arr_types.into_iter().map(SoacInputType::array).collect(),
                form: screma::ScremaForm {
                    pre: screma::Lambda::region(
                        SegBody {
                            region: map_region,
                            captures: capture_nids,
                        },
                        input_elem_types,
                        vec![output_elem_ty.clone()],
                    ),
                    scans: Vec::new(),
                    reductions: Vec::new(),
                    post: screma::Lambda::identity(vec![output_elem_ty]),
                },
                result_state: vec![screma::ResultState { destination }],
                state: screma::RawState,
            }),
            operands,
            tuple_ty,
        );
        Ok(self.intern_pure(PureOp::Project { index: 0 }, smallvec![screma_nid], project_ty))
    }

    /// Convert `reduce_by_index` to a histogram with an identity bucket and an
    /// explicit associative read-combine-write operator.
    fn convert_soac_reduce_by_index(
        &mut self,
        dest: &crate::tlc::Place,
        op: &SoacBody,
        ne: &Term,
        indices: &ArrayExpr,
        values: &ArrayExpr,
        result_ty: Type<TypeName>,
    ) -> Result<NodeId, ConvertError> {
        let dest_view = *self.locals.get(&dest.id).ok_or_else(|| {
            ConvertError::GraphError("reduce_by_index destination is not a bound #[storage] view".into())
        })?;
        let operator_region = self.function_id(self.lambda_fn_symbol(&op.lam)?);
        let operator_captures = op
            .data
            .captures
            .iter()
            .map(|(_, _, term)| self.convert_term(term))
            .collect::<Result<Vec<_>, _>>()?;
        let neutral = self.convert_term(ne)?;
        let index_node = self.convert_array_expr_value(indices)?;
        let value_node = self.convert_array_expr_value(values)?;
        let index_array = self.value_array_type(index_node, indices);
        let value_array = self.value_array_type(value_node, values);
        let index_type = crate::types::array_elem(&index_array)
            .cloned()
            .ok_or_else(|| ConvertError::GraphError("reduce_by_index indices are not an array".into()))?;
        let value_type = crate::types::array_elem(&value_array)
            .cloned()
            .ok_or_else(|| ConvertError::GraphError("reduce_by_index values are not an array".into()))?;
        let operator_parameters = op.lam.params.iter().map(|(_, ty)| ty.clone()).collect();
        let operator = screma::Lambda::region(
            SegBody {
                region: operator_region,
                captures: operator_captures,
            },
            operator_parameters,
            vec![dest.elem_ty.clone()],
        );
        let destination_length = self.intern_pure(
            PureOp::Intrinsic {
                id: catalog().known().length,
                overload_idx: 0,
            },
            smallvec![dest_view],
            index_type.clone(),
        );
        let race_factor = self.intern_pure(PureOp::Int("1".into()), smallvec![], index_type.clone());
        let operands = smallvec![index_node, value_node];
        self.emit_soac(
            Soac::Hist(hist::Op {
                inputs: vec![
                    SoacInputType::array(index_array),
                    SoacInputType::array(value_array),
                ],
                form: hist::HistForm {
                    bucket: screma::Lambda::identity(vec![index_type, value_type]),
                    operations: vec![hist::HistOp {
                        emission: hist::Emission::Always,
                        shape: vec![destination_length],
                        race_factor,
                        destinations: vec![dest_view],
                        update: hist::Update::Reduce {
                            operator,
                            neutral: vec![neutral],
                        },
                    }],
                },
                state: hist::RawState,
            }),
            operands,
            result_ty,
        );
        Ok(dest_view)
    }

    /// `scatter(dest, indices, values)` becomes an ordered-overwrite histogram.
    /// The destination is a bound storage view; expansion rebinds the dummy
    /// result after emitting the indexed stores.
    fn convert_soac_scatter(
        &mut self,
        dest: &crate::tlc::Place,
        lam: &SoacBody,
        inputs: &[ArrayExpr],
        result_ty: Type<TypeName>,
    ) -> Result<NodeId, ConvertError> {
        let dest_sym = dest.id;
        let dest_elem_ty = dest.elem_ty.clone();
        let dest_view = *self.locals.get(&dest_sym).ok_or_else(|| {
            ConvertError::GraphError(
                "scatter destination is not a bound #[storage] view (must be a storage param)".into(),
            )
        })?;

        // The envelope `(xs..) -> (index, value)` is a lifted function post-defunc.
        let function = self.lambda_fn_symbol(&lam.lam)?;
        let (index_type, value_type) = match &lam.lam.ret_ty {
            Type::Constructed(TypeName::Tuple(2), args) => (args[0].clone(), args[1].clone()),
            other => {
                return Err(ConvertError::GraphError(format!(
                    "scatter envelope must return a 2-tuple (index, value), got {other:?}"
                )));
            }
        };

        let input_nids: Vec<NodeId> =
            inputs.iter().map(|ae| self.convert_array_expr_value(ae)).collect::<Result<_, _>>()?;
        let input_array_types: Vec<Type<TypeName>> =
            inputs.iter().zip(input_nids.iter()).map(|(ae, nid)| self.value_array_type(*nid, ae)).collect();
        let capture_nids: Vec<NodeId> =
            lam.data.captures.iter().map(|(_, _, t)| self.convert_term(t)).collect::<Result<_, _>>()?;

        let operands: SmallVec<[NodeId; 4]> = input_nids.into_iter().collect();
        let body_region = self.function_id(function);
        let destination_length = self.intern_pure(
            PureOp::Intrinsic {
                id: catalog().known().length,
                overload_idx: 0,
            },
            smallvec![dest_view],
            index_type.clone(),
        );
        let race_factor = self.intern_pure(PureOp::Int("1".into()), smallvec![], index_type.clone());

        self.emit_soac(
            Soac::Hist(hist::Op {
                inputs: input_array_types.into_iter().map(SoacInputType::array).collect(),
                form: hist::HistForm {
                    bucket: screma::Lambda::region(
                        SegBody {
                            region: body_region,
                            captures: capture_nids,
                        },
                        lam.lam.params.iter().map(|(_, ty)| ty.clone()).collect(),
                        vec![index_type, value_type],
                    ),
                    operations: vec![hist::HistOp {
                        emission: hist::Emission::Always,
                        shape: vec![destination_length],
                        race_factor,
                        destinations: vec![dest_view],
                        update: hist::Update::OrderedOverwrite {
                            value_types: vec![dest_elem_ty],
                        },
                    }],
                },
                state: hist::RawState,
            }),
            operands,
            result_ty,
        );
        Ok(dest_view)
    }

    /// Lower ranked capacity-bounded insertion to the canonical histogram
    /// representation. Destination shape supplies bucket count and capacity;
    /// the item rank remains explicit on the SOAC input.
    fn convert_soac_bucket_scatter(
        &mut self,
        dest: &crate::tlc::Place,
        lam: &SoacBody,
        inputs: &[ArrayExpr],
        input_dimensions: &[Vec<u8>],
        domain_rank: u8,
        _result_ty: Type<TypeName>,
    ) -> Result<NodeId, ConvertError> {
        let dest_view = *self.locals.get(&dest.id).ok_or_else(|| {
            ConvertError::GraphError(
                "bucket_scatter destination is not a bound storage view (must be a storage parameter)"
                    .into(),
            )
        })?;
        let dest_view_ty = self.graph.nodes[dest_view].ty.clone();
        let fixed_size = |ty: &Type<TypeName>, dimension: &str| -> Result<u32, ConvertError> {
            match ty.array_size() {
                Some(Type::Constructed(TypeName::Size(size), _)) if *size > 0 => {
                    let size = u32::try_from(*size).map_err(|_| {
                        ConvertError::Unsupported(format!(
                            "bucket_scatter destination {dimension} is too large"
                        ))
                    })?;
                    if size > i32::MAX as u32 {
                        return Err(ConvertError::Unsupported(format!(
                            "bucket_scatter destination {dimension} exceeds the signed indexing limit"
                        )));
                    }
                    Ok(size)
                }
                _ => Err(ConvertError::Unsupported(format!(
                    "bucket_scatter requires a positive fixed destination {dimension}"
                ))),
            }
        };
        let bucket_count = fixed_size(&dest_view_ty, "bucket count")?;
        let row_ty = dest_view_ty.elem_type().ok_or_else(|| {
            ConvertError::GraphError("bucket_scatter destination must have rank two".into())
        })?;
        let capacity = fixed_size(row_ty, "capacity")?;

        let function = self.lambda_fn_symbol(&lam.lam)?;
        let (active_type, key_type, value_type) = match &lam.lam.ret_ty {
            Type::Constructed(TypeName::Tuple(3), fields) => {
                (fields[0].clone(), fields[1].clone(), fields[2].clone())
            }
            other => {
                return Err(ConvertError::GraphError(format!(
                    "bucket_scatter envelope must return (bool, i32, value), got {other:?}"
                )));
            }
        };
        let bool_type = Type::Constructed(TypeName::Bool, vec![]);
        let i32_type = Type::Constructed(TypeName::Int(32), vec![]);
        if active_type != bool_type || key_type != i32_type || value_type != dest.elem_ty {
            return Err(ConvertError::GraphError(format!(
                "bucket_scatter envelope must return (bool, i32, {:?}), got {:?}",
                dest.elem_ty, lam.lam.ret_ty
            )));
        }
        if inputs.len() != input_dimensions.len() {
            return Err(ConvertError::Internal(
                "bucket_scatter input/dimension metadata length mismatch".into(),
            ));
        }
        if domain_rank == 0 {
            return Err(ConvertError::Internal(
                "bucket_scatter domain rank must be positive".into(),
            ));
        }
        for (input, dimensions) in inputs.iter().zip(input_dimensions) {
            if dimensions.is_empty() {
                return Err(ConvertError::GraphError(
                    "bucket_scatter input has no logical dimensions".into(),
                ));
            }
            let mut unique = dimensions.clone();
            unique.sort_unstable();
            unique.dedup();
            if unique.len() != dimensions.len()
                || dimensions.iter().any(|dimension| *dimension >= domain_rank)
            {
                return Err(ConvertError::GraphError(format!(
                    "bucket_scatter input dimensions {dimensions:?} are invalid for rank {domain_rank}"
                )));
            }
            let input_array_type = input.array_type();
            let mut input_type = &input_array_type;
            for _ in dimensions {
                input_type = input_type.elem_type().ok_or_else(|| {
                    ConvertError::GraphError(format!(
                        "bucket_scatter input rank exceeds its array type {:?}",
                        input_array_type
                    ))
                })?;
            }
        }
        let input_nodes = inputs
            .iter()
            .map(|input| self.convert_array_expr_value(input))
            .collect::<Result<Vec<_>, _>>()?;
        let input_arrays = inputs
            .iter()
            .zip(&input_nodes)
            .map(|(input, node)| self.value_array_type(*node, input))
            .collect::<Vec<_>>();
        let captures = lam
            .data
            .captures
            .iter()
            .map(|(_, _, term)| self.convert_term(term))
            .collect::<Result<Vec<_>, _>>()?;

        let u32_type = Type::Constructed(TypeName::UInt(32), vec![]);
        let counts_size = LogicalSize::FixedBytes(u64::from(bucket_count) * 4);
        let counts = self.arenas.resources.allocate_compiler(
            CompilerResource::new(CompilerResourceKind::BucketCounts, None, 0),
            u32_type.clone(),
            counts_size.clone(),
        );
        let overflow_size = LogicalSize::FixedBytes(4);
        let overflow = self.arenas.resources.allocate_compiler(
            CompilerResource::new(CompilerResourceKind::BucketOverflow, None, 0),
            u32_type.clone(),
            overflow_size.clone(),
        );
        self.extra_resource_declarations.push(SemanticResourceDecl {
            resource: SemanticResourceRef(counts),
            role: crate::interface::StorageRole::Output,
            elem_ty: u32_type.clone(),
            size: counts_size,
        });
        self.extra_resource_declarations.push(SemanticResourceDecl {
            resource: SemanticResourceRef(overflow),
            role: crate::interface::StorageRole::Intermediate,
            elem_ty: u32_type.clone(),
            size: overflow_size,
        });
        let counts_type = crate::types::view_array_with_size(
            &u32_type,
            Type::Constructed(TypeName::Size(bucket_count as usize), vec![]),
            Type::Constructed(TypeName::Resource(counts), vec![]),
        );
        let overflow_type = crate::types::view_array_with_size(
            &u32_type,
            Type::Constructed(TypeName::Size(1), vec![]),
            Type::Constructed(TypeName::Resource(overflow), vec![]),
        );
        let counts_view =
            super::graph_ops::intern_resource_view(&mut self.graph, counts, counts_type, self.current_span);
        let overflow_view = super::graph_ops::intern_resource_view(
            &mut self.graph,
            overflow,
            overflow_type,
            self.current_span,
        );
        let bucket_count_node = self.intern_pure(
            PureOp::Int(bucket_count.to_string()),
            smallvec![],
            i32_type.clone(),
        );
        let capacity_node =
            self.intern_pure(PureOp::Int(capacity.to_string()), smallvec![], i32_type.clone());
        let race_factor = self.intern_pure(PureOp::Int("1".into()), smallvec![], i32_type);
        let body_region = self.function_id(function);
        let placeholder_type = Type::Constructed(TypeName::Bool, vec![]);
        let _placeholder = self.emit_soac(
            Soac::Hist(hist::Op {
                inputs: input_arrays
                    .into_iter()
                    .zip(input_dimensions)
                    .map(|(array, dimensions)| SoacInputType::mapped(array, dimensions.clone()))
                    .collect(),
                form: hist::HistForm {
                    bucket: screma::Lambda::region(
                        SegBody {
                            region: body_region,
                            captures,
                        },
                        lam.lam.params.iter().map(|(_, ty)| ty.clone()).collect(),
                        vec![active_type, key_type, value_type.clone()],
                    ),
                    operations: vec![hist::HistOp {
                        emission: hist::Emission::Guarded,
                        shape: vec![bucket_count_node],
                        race_factor,
                        destinations: vec![dest_view],
                        update: hist::Update::BucketInsert {
                            value_types: vec![dest.elem_ty.clone()],
                            counts: counts_view,
                            overflow: overflow_view,
                            capacity: capacity_node,
                        },
                    }],
                },
                state: hist::RawState,
            }),
            input_nodes.into_iter().collect(),
            placeholder_type,
        );

        let zero = super::graph_ops::intern_u32(&mut self.graph, 0, self.current_span);
        let overflow_value = super::graph_ops::emit_view_load(
            &mut self.graph,
            self.current_block,
            overflow_view,
            zero,
            u32_type,
            self.effect_ids,
            self.current_span,
        );
        Ok(self.intern_pure(
            PureOp::Tuple(3),
            smallvec![dest_view, counts_view, overflow_value],
            Type::Constructed(
                TypeName::Tuple(3),
                vec![
                    dest_view_ty,
                    self.graph.nodes[counts_view].ty.clone(),
                    Type::Constructed(TypeName::UInt(32), vec![]),
                ],
            ),
        ))
    }

    fn convert_soac_reduce(
        &mut self,
        op: &SoacBody,
        ne: &Term,
        input: &ArrayExpr,
        result_ty: Type<TypeName>,
    ) -> Result<NodeId, ConvertError> {
        let operator_symbol = self.lambda_fn_symbol(&op.lam)?;
        let capture_nids: Vec<NodeId> =
            op.data.captures.iter().map(|(_, _, t)| self.convert_term(t)).collect::<Result<_, _>>()?;
        let arr_nid = self.convert_array_expr_value(input)?;
        let arr_ty = self.value_array_type(arr_nid, input);
        let init_nid = self.convert_term(ne)?;

        // Emit as Screma { 0 maps, 1 Reduce accumulator } + project field
        // 0. Reduce's `op` is both the step (per-element) and the
        // reduce_op (phase 2 combiner).
        let operands: SmallVec<[NodeId; 4]> = smallvec![arr_nid];
        let tuple_ty = Type::Constructed(TypeName::Tuple(1), vec![result_ty.clone()]);
        let op_region = self.function_id(operator_symbol);
        let screma_nid = self.emit_soac(
            Soac::Screma(screma::Op {
                inputs: vec![SoacInputType::array(arr_ty)],
                form: screma::ScremaForm {
                    pre: screma::Lambda::identity(vec![result_ty.clone()]),
                    scans: Vec::new(),
                    reductions: vec![screma::Reduce {
                        operator: screma::Lambda::region(
                            SegBody {
                                region: op_region,
                                captures: capture_nids,
                            },
                            vec![result_ty.clone(), result_ty.clone()],
                            vec![result_ty.clone()],
                        ),
                        neutral: vec![init_nid],
                        commutative: false,
                    }],
                    post: screma::Lambda::identity(Vec::new()),
                },
                result_state: vec![screma::ResultState {
                    destination: SoacDestination::fresh(),
                }],
                state: screma::RawState,
            }),
            operands,
            tuple_ty,
        );
        Ok(self.intern_pure(PureOp::Project { index: 0 }, smallvec![screma_nid], result_ty))
    }

    fn convert_soac_scan(
        &mut self,
        op: &SoacBody,
        ne: &Term,
        input: &ArrayExpr,
        destination: SoacDestination,
        result_ty: Type<TypeName>,
    ) -> Result<NodeId, ConvertError> {
        let operator_symbol = self.lambda_fn_symbol(&op.lam)?;
        let capture_nids: Vec<NodeId> =
            op.data.captures.iter().map(|(_, _, t)| self.convert_term(t)).collect::<Result<_, _>>()?;
        let arr_nid = self.convert_array_expr_value(input)?;
        let arr_ty = self.value_array_type(arr_nid, input);
        let init_nid = self.convert_term(ne)?;

        let operands: SmallVec<[NodeId; 4]> = smallvec![arr_nid];

        // Emit as Screma { 0 maps, 1 Scan acc } + project field 0. For
        // consuming scan the result aliases the input, so the Project's
        // type must match the input view's type (View variant + buffer)
        // rather than the TLC-default result_ty (Composite variant).
        // Non-consuming scan keeps result_ty; realize_outputs fixes its
        // variant via retarget_array_projection.
        // Scan is shape-preserving: inherit the input's representation when
        // `result_ty` carries an unresolved `Skolem` size, keeping scan's own
        // output element type (the accumulator type = `result_ty`'s element).
        // Mirror of the `convert_soac_map` guard; without it
        // `scan(op, ne, filter(p, xs))` leaks the filter's Skolem size into the
        // backend.
        let project_ty = if destination.is_input_buffer() {
            arr_ty.clone()
        } else {
            result_ty
                .elem_type()
                .and_then(|elem| shape_preserving_result_ty(&arr_ty, elem, &result_ty))
                .unwrap_or_else(|| result_ty.clone())
        };
        let tuple_ty = Type::Constructed(TypeName::Tuple(1), vec![project_ty.clone()]);
        let scan_elem_ty = soac_element_type(&project_ty);
        let op_region = self.function_id(operator_symbol);
        let screma_nid = self.emit_soac(
            Soac::Screma(screma::Op {
                inputs: vec![SoacInputType::array(arr_ty)],
                form: screma::ScremaForm {
                    pre: screma::Lambda::identity(vec![scan_elem_ty.clone()]),
                    scans: vec![screma::Scan {
                        operator: screma::Lambda::region(
                            SegBody {
                                region: op_region,
                                captures: capture_nids,
                            },
                            vec![scan_elem_ty.clone(), scan_elem_ty.clone()],
                            vec![scan_elem_ty.clone()],
                        ),
                        neutral: vec![init_nid],
                    }],
                    reductions: Vec::new(),
                    post: screma::Lambda::identity(vec![scan_elem_ty.clone()]),
                },
                result_state: vec![screma::ResultState { destination }],
                state: screma::RawState,
            }),
            operands,
            tuple_ty,
        );
        Ok(self.intern_pure(PureOp::Project { index: 0 }, smallvec![screma_nid], project_ty))
    }

    fn convert_soac_filter(
        &mut self,
        pred: &SoacBody,
        input: &ArrayExpr,
        destination: SoacDestination,
        _result_ty: Type<TypeName>,
    ) -> Result<NodeId, ConvertError> {
        let predicate_symbol = self.lambda_fn_symbol(&pred.lam)?;
        let capture_nids: Vec<NodeId> =
            pred.data.captures.iter().map(|(_, _, t)| self.convert_term(t)).collect::<Result<_, _>>()?;
        let elem_ty = self.array_expr_elem_type(input);
        let arr_ty = self.array_expr_type(input);
        let arr_nid = self.convert_array_expr_value(input)?;

        let output_elem_ty = elem_ty.clone();
        let pred_body = SegBody {
            region: self.function_id(predicate_symbol),
            captures: capture_nids,
        };

        // `[input]` only — map/pred captures live on their `SegBody`s.
        let operands: SmallVec<[NodeId; 4]> = smallvec![arr_nid];

        // The TLC-level result type is an existential `?k. [k]T`; after
        // `open_existential` its size is a `Skolem(k)`. Two lowerings,
        // keyed by whether the input has a static capacity:
        let size = arr_ty
            .array_size()
            .ok_or_else(|| ConvertError::GraphError("filter: input has no array_size".into()))?
            .clone();

        if let Type::Constructed(TypeName::Size(_), _) = &size {
            // Static-capacity input: result is a function-local
            // `Array[T, Size(N), Bounded]` `{buffer, len}` struct (N is the
            // input's static size, the upper bound).
            let bounded_result_ty = Type::Constructed(
                TypeName::Array,
                vec![
                    output_elem_ty.clone(),
                    Type::Constructed(TypeName::ArrayVariantBounded, vec![]),
                    size.clone(),
                    crate::types::no_buffer(),
                ],
            );
            return Ok(self.emit_soac(
                Soac::Filter(filter::Op {
                    body: filter::Body {
                        inputs: vec![SoacInputType::array(arr_ty)],
                        map: screma::Lambda::identity(vec![output_elem_ty.clone()]),
                        predicate: screma::Lambda::region(
                            pred_body.clone(),
                            vec![output_elem_ty.clone()],
                            vec![Type::Constructed(TypeName::Bool, vec![])],
                        ),
                    },
                    state: filter::RawState {
                        storage: filter::Output::Local {
                            capacity: size,
                            destination,
                        },
                    },
                }),
                operands,
                bounded_result_ty,
            ));
        }

        // Runtime-sized input: compact the kept elements into a reserved
        // scratch storage buffer (capacity = input element count), and yield a
        // runtime-length view over it. The surviving count is the view's `len`
        // operand. A runtime-sized result cannot back a function-local array.
        //
        // The scratch buffer is a compiler resource from birth. A runtime
        // `filter` reaching here in a standalone function still has no entry
        // interface that can own the requirement, so `convert_function`
        // rejects that broken inlining state below.
        let input_binding = crate::types::array_view_buffer(&arr_ty);
        let input_elem_bytes = crate::ssa::layout::storage_elem_stride(&elem_ty).ok_or_else(|| {
            ConvertError::GraphError("filter: element type has no static byte size".into())
        })?;
        // The scratch buffer holds the kept output values (`f(x)` when a map is
        // fused), so it is sized in `output_elem_ty`; the surviving-count bound
        // still comes from the input buffer's element count.
        let output_elem_bytes =
            crate::ssa::layout::storage_elem_stride(&output_elem_ty).ok_or_else(|| {
                ConvertError::GraphError("filter: output element type has no static byte size".into())
            })?;
        // A runtime map producer may still be a Composite here. Reserve the
        // filter resource identity now, but let semantic EGIR resolve its size
        // after producer fusion exposes the actual input resource/domain.
        let scratch_size = input_binding
            .map(|binding| LogicalSize::LikeResource {
                resource: self.arenas.resources.host_id(binding),
                elem_bytes: output_elem_bytes,
                src_elem_bytes: input_elem_bytes,
            })
            .unwrap_or(LogicalSize::Unspecified);
        let scratch_out = self.arenas.resources.allocate_compiler(
            CompilerResource::new(CompilerResourceKind::FilterScratch, None, 0),
            output_elem_ty.clone(),
            scratch_size.clone(),
        );
        self.extra_resource_declarations.push(SemanticResourceDecl {
            resource: SemanticResourceRef(scratch_out),
            role: crate::interface::StorageRole::Output,
            elem_ty: output_elem_ty.clone(),
            size: scratch_size,
        });
        let view_result_ty = crate::types::view_array_of(
            &output_elem_ty,
            Type::Constructed(TypeName::Resource(scratch_out), vec![]),
        );
        Ok(self.emit_soac(
            Soac::Filter(filter::Op {
                body: filter::Body {
                    inputs: vec![SoacInputType::array(arr_ty)],
                    map: screma::Lambda::identity(vec![output_elem_ty.clone()]),
                    predicate: screma::Lambda::region(
                        pred_body,
                        vec![output_elem_ty.clone()],
                        vec![Type::Constructed(TypeName::Bool, vec![])],
                    ),
                },
                state: filter::RawState {
                    storage: filter::Output::Runtime {
                        scratch: super::program::SemanticResourceRef(scratch_out),
                        length: filter::RuntimeLength::ViewOnly,
                    },
                },
                // Residency or output realization upgrades this to `Stored`
                // when the compacted array crosses a scheduling/ABI boundary.
            }),
            operands,
            view_result_ty,
        ))
    }

    // ========================================================================
    // ArrayExpr
    // ========================================================================

    fn convert_array_expr(&mut self, ae: &ArrayExpr, ty: Type<TypeName>) -> Result<NodeId, ConvertError> {
        match ae {
            ArrayExpr::Var(vr, var_ty) => {
                let t = crate::tlc::synthetic_atom_var_term(*vr, var_ty.clone());
                self.convert_term(&t)
            }
            // A `Zip` is the SoA form of a tuple-element array input: it lowers
            // to a `Tuple` of its component arrays (`[N](A,B)` ≡ `([N]A, [N]B)`),
            // the same node a `Tuple`-term input would build. `soa::normalize`
            // flattens `Map`-over-`Zip` ahead of EGIR, but an SoA constant
            // inlined into a non-`Map` consumer (e.g. `reduce`) still arrives as
            // a `Zip` here.
            ArrayExpr::Zip(children) => {
                let operands: SmallVec<[NodeId; 4]> =
                    children.iter().map(|c| self.convert_array_expr_value(c)).collect::<Result<_, _>>()?;
                let component_tys: Vec<Type<TypeName>> =
                    children.iter().map(|c| self.array_expr_type(c)).collect();
                let tuple_ty = Type::Constructed(TypeName::Tuple(component_tys.len()), component_tys);
                let n = operands.len();
                Ok(self.intern_pure(PureOp::Tuple(n), operands, tuple_ty))
            }
            ArrayExpr::Literal(terms) => {
                let operands: SmallVec<[NodeId; 4]> =
                    terms.iter().map(|t| self.convert_term(t)).collect::<Result<_, _>>()?;
                let n = operands.len();
                Ok(self.intern_pure(PureOp::ArrayLit(n), operands, ty))
            }
            ArrayExpr::Range { start, len, step } => {
                let start_nid = self.convert_term(start)?;
                let len_nid = self.convert_term(len)?;
                let mut operands: SmallVec<[NodeId; 4]> = smallvec![start_nid, len_nid];
                let has_step = if let Some(step_term) = step {
                    operands.push(self.convert_term(step_term)?);
                    true
                } else {
                    false
                };
                Ok(self.intern_pure(PureOp::ArrayRange { has_step }, operands, ty))
            }
        }
    }

    fn convert_array_expr_value(&mut self, ae: &ArrayExpr) -> Result<NodeId, ConvertError> {
        let ty = self.array_expr_type(ae);
        self.convert_array_expr(ae, ty)
    }

    fn array_expr_type(&self, ae: &ArrayExpr) -> Type<TypeName> {
        // Strip `*` at the EGIR boundary — uniqueness is a TLC concern;
        // downstream array-shape checks (composite vs view vs virtual) operate
        // on the bare array type. Only `Var` can carry uniqueness; for the other
        // atoms `strip_unique` is a no-op.
        ae.array_type()
    }

    fn array_expr_elem_type(&self, ae: &ArrayExpr) -> Type<TypeName> {
        match ae {
            ArrayExpr::Var(_, ty) => match ty {
                Type::Constructed(TypeName::Array, args) if !args.is_empty() => args[0].clone(),
                // After `tlc::soa`, `[N](A, B)` becomes `([N]A, [N]B)` — an
                // SoA tuple. The per-iteration element type is the
                // corresponding `(A, B)` element tuple, not the tuple
                // itself.
                ty if as_soa_tuple(ty).is_some() => soac_element_type(ty),
                _ => ty.clone(),
            },
            // SoA tuple input: the per-iteration element is the tuple of each
            // component array's element type.
            ArrayExpr::Zip(children) => {
                let elem_tys: Vec<Type<TypeName>> =
                    children.iter().map(|c| self.array_expr_elem_type(c)).collect();
                Type::Constructed(TypeName::Tuple(elem_tys.len()), elem_tys)
            }
            ArrayExpr::Literal(terms) => {
                terms.first().map(|t| t.ty.clone()).unwrap_or(Type::Constructed(TypeName::Unit, vec![]))
            }
            ArrayExpr::Range { start, .. } => start.ty.clone(),
        }
    }

    /// Authoritative array type of an already-converted input value `nid`. The
    /// EGIR node's type reflects representation rewrites the TLC term type
    /// predates — notably a runtime `filter` whose result is a `View` even
    /// though its TLC type is the existential-opened `Composite`. So a SOAC
    /// consumer reads the array shape (variant / buffer) off the node, falling
    /// back to the TLC-derived `array_expr_type` when the node isn't a concrete
    /// array (e.g. an opaque tuple handle). Mirrors how `length` dispatches on
    /// the value type rather than the source type.
    fn value_array_type(&self, nid: NodeId, fallback: &ArrayExpr) -> Type<TypeName> {
        if let Some(node) = self.graph.nodes.get(nid) {
            if matches!(&node.ty, Type::Constructed(TypeName::Array, _)) || as_soa_tuple(&node.ty).is_some()
            {
                return node.ty.clone();
            }
        }
        self.array_expr_type(fallback)
    }

    /// Element type matching `value_array_type`: peel the array / SoA-tuple
    /// element off `arr_ty`, falling back to the TLC-derived element type.
    fn value_elem_type(&self, arr_ty: &Type<TypeName>, fallback: &ArrayExpr) -> Type<TypeName> {
        if let Type::Constructed(TypeName::Array, args) = arr_ty {
            if !args.is_empty() {
                return args[0].clone();
            }
        }
        if as_soa_tuple(arr_ty).is_some() {
            return soac_element_type(arr_ty);
        }
        self.array_expr_elem_type(fallback)
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Check whether every reachable value in a zero-parameter EGIR body is a
/// compile-time constant expression. This is deliberately an EGIR property;
/// no temporary SSA body is constructed for classification.
fn is_purely_constant_graph(graph: &EGraph<Raw>) -> bool {
    let mut memo = LookupMap::new();
    graph.skeleton.blocks.values().all(|block| {
        block.side_effects.is_empty()
            && match &block.term {
                SkeletonTerminator::Return(value) => {
                    value.is_none_or(|node| is_constant_node(graph, node, &mut memo))
                }
                SkeletonTerminator::Branch { args, .. } => {
                    args.iter().copied().all(|node| is_constant_node(graph, node, &mut memo))
                }
                SkeletonTerminator::CondBranch {
                    cond,
                    then_args,
                    else_args,
                    ..
                } => std::iter::once(*cond)
                    .chain(then_args.iter().copied())
                    .chain(else_args.iter().copied())
                    .all(|node| is_constant_node(graph, node, &mut memo)),
                SkeletonTerminator::Unreachable => true,
            }
    })
}

fn is_constant_node(graph: &EGraph<Raw>, mut node: NodeId, memo: &mut LookupMap<NodeId, bool>) -> bool {
    while let Some(replacement) = graph.nodes[node].alias {
        node = replacement;
    }
    if let Some(result) = memo.get(&node) {
        return *result;
    }
    let result = match &graph.nodes[node].kind {
        ENode::Constant(_) => true,
        ENode::Pure { op, operands } => {
            matches!(
                op,
                PureOp::Int(_)
                    | PureOp::Uint(_)
                    | PureOp::Float(_)
                    | PureOp::Bool(_)
                    | PureOp::Unit
                    | PureOp::Tuple(_)
                    | PureOp::Vector(_)
                    | PureOp::Matrix { .. }
                    | PureOp::ArrayLit(_)
                    | PureOp::Global(_)
            ) && operands.iter().copied().all(|operand| is_constant_node(graph, operand, memo))
        }
        ENode::Union { left, right } => {
            is_constant_node(graph, *left, memo) && is_constant_node(graph, *right, memo)
        }
        ENode::FuncParam { .. } | ENode::BlockParam { .. } | ENode::SideEffectResult => false,
    };
    memo.insert(node, result);
    result
}

/// Extract a `#[size_hint(N)]` attribute from a lowered entry parameter.
pub fn extract_size_hint(param: &interface::EntryParamDecl) -> Option<std::num::NonZeroU32> {
    param.attributes.iter().find_map(|attribute| match attribute {
        interface::Attribute::SizeHint(n) => Some(*n),
        _ => None,
    })
}

/// Convert an AST attribute to an IO decoration.
fn convert_to_io_decoration(attr: &interface::ResolvedAttribute) -> Option<IoDecoration> {
    use IoDecoration;
    match attr {
        interface::Attribute::BuiltIn(b) => Some(IoDecoration::BuiltIn(*b)),
        interface::Attribute::VertexSlot(n) | interface::Attribute::Varying(n) => {
            Some(IoDecoration::Location(*n))
        }
        _ => None,
    }
}

/// The render-target resource name of a `#[target(name)]` output attribute.
fn target_of(attr: Option<&interface::ResolvedAttribute>) -> Option<String> {
    match attr {
        Some(interface::Attribute::Target(name)) => Some(name.clone()),
        _ => None,
    }
}

/// Shape-preserving result type for a non-in-place `map`/`scan`.
fn storage_output_binding(attr: Option<&interface::ResolvedAttribute>) -> Option<BindingRef> {
    match attr {
        Some(interface::Attribute::Storage { set, binding, .. }) => Some(BindingRef::new(*set, *binding)),
        _ => None,
    }
}

///
/// When the TLC `result_ty` carries an unresolved existential `Skolem` size —
/// the type of a `filter`-produced input opened by `open_existential`, which the
/// backend can't lower — rebuild the result from the input array's
/// representation (variant / size / buffer) with `output_elem_ty`. `map`/`scan`
/// are shape-preserving, so this is exactly the input's shape with a possibly
/// different element type (e.g. `Bounded[N]`, whose runtime `len` a consuming
/// SOAC needs).
///
/// Returns `None` — caller falls back to `result_ty` — when the size is already
/// concrete or the input isn't a plain array. Never panics, so a malformed input
/// degrades to the prior behavior rather than aborting the compiler.
fn shape_preserving_result_ty(
    input_arr_ty: &Type<TypeName>,
    output_elem_ty: &Type<TypeName>,
    result_ty: &Type<TypeName>,
) -> Option<Type<TypeName>> {
    if !matches!(
        crate::types::array_size(result_ty),
        Some(Type::Constructed(TypeName::Skolem(_), _))
    ) {
        return None;
    }
    Some(crate::types::make_array1(
        output_elem_ty.clone(),
        input_arr_ty.array_variant()?.clone(),
        input_arr_ty.array_size()?.clone(),
        input_arr_ty.array_buffer()?.clone(),
    ))
}

fn build_entry_outputs(
    entry: &interface::EntryDecl,
    ret_type: &Type<TypeName>,
    slot_value_tys: &[Option<Type<TypeName>>],
    inputs: &[EntryInput],
    is_compute: bool,
    binding_ids: &mut crate::IdSource<u32>,
) -> Result<Vec<EntryOutput>, ConvertError> {
    use EntryOutput;
    let logical_ret_type = strip_existentials(ret_type);
    let output_arity = entry_output_arity(entry, ret_type);
    // Pick a `BufferLen` policy for the output binding, in order:
    //
    //   1. Output type carries a compile-time-known `Size(n)` literal
    //      → `Fixed { bytes: n * elem_bytes }`.
    //   2. Output's size variable matches one of the entry's storage
    //      inputs (the type checker has unified them) → `LikeInput`
    //      tracking that input.
    //   3. A runtime array output route is sized from the finalized semantic
    //      dispatch domain → `SameAsDispatch { elem_bytes }`.
    //   4. None — the host falls back to its default sizing or, if it
    //      tried to allocate this buffer, surfaces a clean error.
    //
    // The size info is already in the (post-monomorphize) type — we
    // just read it. No structural rewrites needed for `if/else`
    // branches whose result types have already been unified.
    let length_for =
        |binding: Option<BindingRef>, ty: &Type<TypeName>| -> Result<Option<BufferLen>, ConvertError> {
            if binding.is_none() {
                return Ok(None);
            }
            let Some(elem_ty) = ty.elem_type() else {
                let bytes = crate::ssa::layout::type_byte_size(ty).ok_or_else(|| {
                    ConvertError::Internal(format!("output has no static byte layout: {ty:?}"))
                })?;
                return Ok(Some(BufferLen::Fixed {
                    bytes: u64::from(bytes),
                }));
            };
            let elem_bytes = crate::ssa::layout::storage_elem_stride(elem_ty).ok_or_else(|| {
                ConvertError::Internal(format!("output element has no static byte layout: {elem_ty:?}"))
            })?;
            if let Some(out_size) = crate::types::array_size(ty) {
                // Rule 1: compile-time size literal.
                if let Type::Constructed(TypeName::Size(n), _) = out_size {
                    return Ok(Some(BufferLen::Fixed {
                        bytes: (*n as u64) * elem_bytes as u64,
                    }));
                }
                // Rule 2: size variable shared with an entry input.
                for input in inputs {
                    let EntryInputKind::Storage {
                        exposure: BindingExposure::Host(in_binding),
                        ..
                    } = &input.kind
                    else {
                        continue;
                    };
                    let Some(in_size) = crate::types::array_size(&input.ty) else {
                        continue;
                    };
                    if in_size == out_size {
                        let Some(in_elem_ty) = input.ty.elem_type() else {
                            continue;
                        };
                        let src_elem_bytes = crate::ssa::layout::storage_elem_stride(in_elem_ty)
                            .ok_or_else(|| {
                                ConvertError::Internal(format!(
                                    "input element has no static byte layout: {in_elem_ty:?}"
                                ))
                            })?;
                        return Ok(Some(BufferLen::LikeInput {
                            set: in_binding.set,
                            binding: in_binding.binding,
                            elem_bytes,
                            src_elem_bytes,
                        }));
                    }
                }
            }
            // Rule 3: dynamic arrays without a fixed or matching-input size
            // are sized from the finalized semantic dispatch domain.
            if ty.is_array() {
                return Ok(Some(BufferLen::SameAsDispatch { elem_bytes }));
            }
            Ok(None)
        };
    let mut storage_binding_for = |ty: &Type<TypeName>,
                                   is_compute: bool,
                                   attribute: Option<&interface::ResolvedAttribute>|
     -> Option<BindingRef> {
        if is_compute && !matches!(ty, Type::Constructed(TypeName::Unit, _)) {
            storage_output_binding(attribute)
                .or_else(|| Some(BindingRef::new(AUTO_STORAGE_SET, binding_ids.next_id())))
        } else {
            None
        }
    };

    // Prefer the converted route value's representation-specialized type to
    // the parse-time output declaration. A source entry with no return value
    // has no logical output slot. Returning a synthetic Unit-typed
    // `EntryOutput` here would surface to the SPIR-V backend as an
    // `Output<void>` variable in the entry's interface — malformed and
    // rejected by naga / the Vulkan validation layer.
    if is_storage_image_ty(ret_type) || entry.outputs.iter().any(|output| is_storage_image_ty(&output.ty)) {
        return Ok(vec![]);
    }

    if entry.outputs.is_empty()
        && matches!(
            ret_type,
            Type::Constructed(TypeName::Unit | TypeName::SideEffect, _)
        )
    {
        return Ok(vec![]);
    }

    if entry.outputs.iter().all(|o| o.attribute.is_none()) && output_arity == 1 {
        if !matches!(ret_type, Type::Constructed(TypeName::Unit, _)) {
            let source_ty = slot_value_tys.first().and_then(Option::as_ref).unwrap_or(ret_type);
            let ty = crate::types::canonical_storage_buffer_ty(source_ty);
            let attribute = entry.outputs.first().and_then(|output| output.attribute.as_ref());
            let storage_binding = storage_binding_for(&ty, is_compute, attribute);
            let length = length_for(storage_binding, &ty)?;
            Ok(vec![EntryOutput {
                ty,
                kind: entry_output_kind(storage_binding, length, None, None),
            }])
        } else {
            Ok(vec![])
        }
    } else if let Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), component_types) =
        logical_ret_type
    {
        component_types
            .iter()
            .enumerate()
            .map(|(slot, ty)| {
                let ty = slot_value_tys.get(slot).and_then(Option::as_ref).unwrap_or(ty);
                let ty = crate::types::canonical_storage_buffer_ty(ty);
                let attribute = entry.outputs.get(slot).and_then(|output| output.attribute.as_ref());
                let storage_binding = storage_binding_for(&ty, is_compute, attribute);
                let length = length_for(storage_binding, &ty)?;
                Ok(EntryOutput {
                    ty,
                    kind: entry_output_kind(
                        storage_binding,
                        length,
                        attribute.and_then(convert_to_io_decoration),
                        target_of(attribute),
                    ),
                })
            })
            .collect()
    } else {
        let source_ty = slot_value_tys.first().and_then(Option::as_ref).unwrap_or(ret_type);
        let ty = crate::types::canonical_storage_buffer_ty(source_ty);
        let first_attr = entry.outputs.first().and_then(|o| o.attribute.as_ref());
        let storage_binding = storage_binding_for(&ty, is_compute, first_attr);
        let length = length_for(storage_binding, &ty)?;
        Ok(vec![EntryOutput {
            ty,
            kind: entry_output_kind(
                storage_binding,
                length,
                first_attr.and_then(convert_to_io_decoration),
                target_of(first_attr),
            ),
        }])
    }
}

fn entry_output_kind(
    storage_binding: Option<BindingRef>,
    length: Option<BufferLen>,
    decoration: Option<IoDecoration>,
    target: Option<String>,
) -> EntryOutputKind {
    if let Some(binding) = storage_binding {
        return EntryOutputKind::Storage {
            exposure: BindingExposure::Host(binding),
            length,
        };
    }
    let destination = match (decoration, target) {
        (Some(IoDecoration::BuiltIn(builtin)), None) => EntryOutputDestination::BuiltIn(builtin),
        (Some(IoDecoration::Location(location)), None) => EntryOutputDestination::Location(location),
        (None, Some(target)) => EntryOutputDestination::Target(target),
        (None, None) => EntryOutputDestination::Plain,
        (Some(_), Some(_)) => unreachable!("entry output cannot have both a decoration and target"),
    };
    EntryOutputKind::Value { destination }
}

#[cfg(test)]
#[path = "from_tlc_tests.rs"]
mod from_tlc_tests;

#[cfg(test)]
#[path = "fold_tests.rs"]
mod fold_tests;
