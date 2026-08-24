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
        super::program::NoStorageDeclaration,
        super::ir::RealizedOutputRoute,
        super::program::SemanticProgramData,
    >,
    super::program::RewriteGlobal,
>;

use crate::builtins;
use crate::builtins::{catalog, Purity};
use crate::egir;
use crate::op;
use crate::op::BinaryOperator;
use crate::pipeline_descriptor;
use crate::ssa;
use crate::tlc;
use crate::tlc::VarRef;
use crate::types;
use crate::EntryId;
use crate::FunctionId;
use crate::GlobalId;
use crate::{LookupMap, LookupSet};
use wyn_base::IdSource;

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
    ConstantDef, Func, Program, ProgramIdentities, RawEntry, RewriteGlobal, SemanticOpIdSource,
    SemanticProgramData,
};
use super::soac::{filter, hist, screma};
use super::types::*;
use crate::pipeline_descriptor::BufferLen;

type TlcFamily = tlc::input_slice_bounds::InputBounded;
type ClosureData = tlc::data::ExplicitClosurePayload;
type SoacBodyData = tlc::data::ExplicitCapturesPayload;
type TlcProgram = tlc::stage::InputSliceBoundsInferred;
type TlcDef = GenericDef<TlcFamily>;
type DefMeta = GenericDefMeta<tlc::data::EntryInputBounds>;
type Term = GenericTerm<ClosureData, SoacBodyData>;
type TermKind = GenericTermKind<ClosureData, SoacBodyData>;
type Lambda = GenericLambda<ClosureData, SoacBodyData>;
type LoopKind = GenericLoopKind<ClosureData, SoacBodyData>;

fn contains_storage_view_type(ty: &Type<TypeName>) -> bool {
    types::is_array_variant_view(ty)
        || super::types::as_soa_tuple(ty)
            .is_some_and(|components| components.iter().any(contains_storage_view_type))
}

fn strided_fields_layout(array: &Type<TypeName>, rank: usize) -> Option<ArrayLayout> {
    let components = super::types::as_soa_tuple(array)?;
    let mut offset = 0u32;
    let mut field_offsets = Vec::with_capacity(components.len());
    for component in components {
        let leaf = super::types::soac_leaf_type(component, u8::try_from(rank).ok()?);
        field_offsets.push(offset);
        offset = offset.checked_add(ssa::layout::type_byte_size(&leaf)?)?;
    }
    Some(ArrayLayout::StridedFields {
        element_stride_bytes: offset,
        field_offsets_bytes: field_offsets,
    })
}
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
            params.iter().all(types::is_copy) && types::is_copy(&result)
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
                        || builtins::by_id(*id).raw.purity == Purity::Effectful
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
            if array.ty.array_variant().is_some_and(types::is_array_variant_view) {
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
    callable_boundaries: &'a LookupMap<
        SymbolId,
        (
            Parameters<BindingRef, Type<TypeName>>,
            FunctionResult<Type<TypeName>>,
            CallEffects,
        ),
    >,
}

struct ConversionArenas {
    identities: ProgramIdentities,
    function_ids: LookupMap<SymbolId, FunctionId>,
    global_ids: LookupMap<SymbolId, GlobalId>,
    entry_ids: LookupMap<SymbolId, EntryId>,
}

impl ConversionArenas {
    fn new() -> Self {
        Self {
            identities: ProgramIdentities::new(),
            function_ids: LookupMap::new(),
            global_ids: LookupMap::new(),
            entry_ids: LookupMap::new(),
        }
    }
}

impl<'a> GlobalContext<'a> {
    fn new_converter<'b>(
        &self,
        pure_constants: &LookupSet<SymbolId>,
        binding_ids: &'b mut IdSource<u32>,
        effect_ids: &'b mut IdSource<EffectToken>,
        arenas: &'b mut ConversionArenas,
    ) -> Converter<'a, 'b> {
        Converter::new(
            self.top_level,
            self.symbols,
            pure_constants.clone(),
            self.callable_boundaries,
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
    mut binding_ids: IdSource<u32>,
    mut effect_ids: IdSource<EffectToken>,
) -> Result<Converted, ConvertError> {
    let super::pipeline_seed::PipelineSeed {
        pipeline,
        stage_symbols,
    } = super::pipeline_seed::build(program);
    let top_level: LookupMap<SymbolId, &TlcDef> = program.defs.iter().map(|d| (d.name, d)).collect();
    let symbols = &program.symbols;
    let pure_definitions = infer_pure_definitions(program);
    let callable_boundaries = program
        .defs
        .iter()
        .filter(|definition| matches!(definition.meta, DefMeta::Function | DefMeta::LiftedLambda))
        .map(|definition| {
            let (parameter_types, result_type) = extract_function_signature(&definition.ty);
            let parameters =
                super::types::Parameters::from_ordered(parameter_types.into_iter().enumerate().map(
                    |(index, ty)| callable_parameter::<BindingRef, WynLanguage>(format!("arg{index}"), ty),
                ));
            let result = by_value_function_result::<WynLanguage>(result_type);
            let effects = if pure_definitions.contains(&definition.name) {
                CallEffects::Pure
            } else {
                CallEffects::General
            };
            (definition.name, (parameters, result, effects))
        })
        .collect::<LookupMap<_, _>>();

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
    let stage_entries: Vec<Vec<EntryId>> = stage_symbols
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
        callable_boundaries: &callable_boundaries,
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
            let result_abi = by_value_function_result::<WynLanguage>(def.body.ty.clone());
            let result =
                super::graph_ops::bind_by_value_result(&mut converter.graph, &result_abi, result_nid);
            converter.set_return(Some(result));
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
    let mut functions: Vec<Func<Raw>> = Vec::new();
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
    for function in &mut functions {
        normalize_interface_function_parameters(function);
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
    let ConversionArenas { identities, .. } = arenas;
    Ok(Program::from_parts(
        functions,
        externs,
        entry_points,
        constants,
        SemanticProgramData {
            pipeline,
            stage_entries,
            identities,
        },
        RewriteGlobal {
            binding_ids,
            effect_ids,
            semantic_ids: SemanticOpIdSource::default(),
        },
    ))
}

fn pipeline_workgroup_size(
    pipeline: &pipeline_descriptor::PipelineDescriptor,
    stage_entries: &[Vec<EntryId>],
    entry: EntryId,
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
    Regular(Func<Raw>),
}

// ============================================================================
// Function conversion
// ============================================================================

fn convert_function<'a>(
    def: &TlcDef,
    ctx: &GlobalContext<'a>,
    pure_constants: &LookupSet<SymbolId>,
    binding_ids: &'a mut IdSource<u32>,
    effect_ids: &'a mut IdSource<EffectToken>,
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
    let (inner_body, params) = tlc::extract_lambda_params_ref(&def.body);
    let ret_type = inner_body.ty.clone();
    let param_info = super::types::Parameters::from_ordered(
        params
            .iter()
            .map(|(sym, ty)| {
                Ok(callable_parameter::<BindingRef, WynLanguage>(
                    symbol_name(symbols, *sym)?.to_string(),
                    ty.clone(),
                ))
            })
            .collect::<Result<Vec<_>, ConvertError>>()?,
    );

    let mut converter = ctx.new_converter(pure_constants, binding_ids, effect_ids, arenas);

    for ((sym, _), (parameter_id, parameter)) in params.iter().zip(param_info.iter_with_ids()) {
        let operand = converter.graph.add_parameter(parameter_id, parameter.representation());
        converter.locals.insert(
            *sym,
            operand.value().expect("source function parameters use the value or view channel"),
        );
    }
    let result = converter.convert_term(inner_body)?;
    let result_abi = by_value_function_result::<WynLanguage>(ret_type.clone());
    let result_binding = super::graph_ops::bind_by_value_result(&mut converter.graph, &result_abi, result);
    converter.set_return(Some(result_binding));

    let region = function_id;
    let graph = converter.into_graph();
    Ok(ConvertedFunc::Regular(Func::<Raw>::new(
        region,
        def_name,
        def.body.span,
        None,
        param_info,
        result_abi,
        if ctx.pure_definitions.contains(&def.name) { CallEffects::Pure } else { CallEffects::General },
        graph,
    )))
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
    binding_ids: &mut IdSource<u32>,
    effect_ids: &mut IdSource<EffectToken>,
    arenas: &mut ConversionArenas,
) -> Result<RawEntry, ConvertError> {
    use crate::flow::ExecutionModel;

    let entry_id = arenas.entry_ids[&def.name];
    let symbols = ctx.symbols;
    let def_name = symbol_name(symbols, def.name)?;
    let (inner_body, params) = tlc::extract_lambda_params_ref(&def.body);
    let is_compute = entry.entry_kind == interface::EntryKind::Compute;

    // The converted body carries the specialized return representation; use it
    // rather than the parse-time entry declaration.
    let ret_type = inner_body.ty.clone();
    let param_info = super::types::Parameters::from_ordered(
        params
            .iter()
            .map(|(sym, ty)| {
                Ok(callable_parameter::<BindingRef, WynLanguage>(
                    symbol_name(symbols, *sym)?.to_string(),
                    ty.clone(),
                ))
            })
            .collect::<Result<Vec<_>, ConvertError>>()?,
    );

    let mut converter = ctx.new_converter(pure_constants, binding_ids, effect_ids, arenas);

    // Build entry inputs alongside the symbol → ValueId bindings. A compute
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
        let parameter_id = param_info.id_at_abi_position(i).expect("entry parameter position exists");
        let parameter = param_info.get(parameter_id).unwrap();
        let fp_nid = converter
            .graph
            .add_parameter(parameter_id, parameter.representation())
            .value()
            .expect("source entry parameters use the value or view channel");
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
            let mut view_nids: SmallVec<[ValueId; 4]> = SmallVec::new();
            for (field_idx, (field_ty, slot)) in field_tys.iter().zip(fields.iter()).enumerate() {
                inputs.push(EntryInput {
                    name: format!("{}_{}", name, field_idx),
                    ty: types::canonical_storage_buffer_ty(field_ty),
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
            let size = ssa::layout::type_byte_size(ty).ok_or_else(|| {
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
            ty: types::canonical_storage_buffer_ty(ty),
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
    // A fixed storage parameter carries its complete byte size in the type.
    // Preserve that ABI fact when no stronger slice-derived minimum was
    // inferred. Runtime-sized views still have no static layout and remain
    // host-sized (`None`).
    for input in &mut inputs {
        let EntryInputKind::Storage { length, .. } = &mut input.kind else {
            continue;
        };
        if length.is_none() {
            *length = ssa::layout::type_byte_size(&input.ty).map(|bytes| BufferLen::Fixed {
                bytes: u64::from(bytes),
            });
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

    // Convert the body and retain every control-flow source of each declared
    // output. Concrete stores remain a physicalization concern, but a
    // converted entry already owns its complete interface and output routes.
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

    // Compute entries publish through routes, not function return values.
    let ret_type = if is_compute || is_storage_image_ty(&ret_type) {
        Type::Constructed(TypeName::Unit, vec![])
    } else {
        ret_type
    };
    let result_abi = by_value_function_result::<WynLanguage>(ret_type.clone());
    let result_binding = if is_unit_return || is_compute {
        result_abi.bind(
            |_, _| panic!("unit entry result has no by-value leaves"),
            |_| panic!("raw entry result has no destination parameters"),
        )
    } else {
        super::graph_ops::bind_by_value_result(
            &mut converter.graph,
            &result_abi,
            result_nid.expect("non-unit entry has a result value"),
        )
    };
    converter.set_return(Some(result_binding));

    let Converter {
        graph,
        output_sources: slot_sources,
        ..
    } = converter;
    let output_count = outputs.len();
    let mut entry = RawEntry::new_with_resources(
        def_name.to_string(),
        entry_id,
        def.body.span,
        execution_model,
        inputs,
        outputs,
        Vec::new(),
        param_info,
        result_abi,
        graph,
    );
    for input in &mut entry.inputs {
        input.resource = input
            .storage_binding()
            .or_else(|| input.storage_image_binding().map(|(binding, ..)| binding))
            .or_else(|| interface_type_binding(&input.ty));
    }
    for output in &mut entry.outputs {
        output.resource = output.storage_binding().or_else(|| interface_type_binding(&output.ty));
    }
    entry.parameter_inputs = vec![Vec::new(); entry.params.len()];
    for (slot, parameter_index) in input_parameter_indices.into_iter().enumerate() {
        entry.parameter_inputs[parameter_index].push(super::program::InputSlotId(slot));
    }
    for (slot, sources) in slot_sources.into_iter().enumerate().take(output_count) {
        entry.outputs[slot].routes.extend(sources.into_iter().map(|source| {
            super::ir::RealizedOutputRoute {
                source,
                writers: Vec::new(),
            }
        }));
    }

    complete_entry_outputs(&mut entry)?;

    Ok(entry)
}

/// Finish the entry interface while TLC types, bindings, and raw producers are
/// still available together. Writer provenance is deliberately omitted here:
/// linking producers requires the completed graph and is the first private
/// step of semantic reification.
fn complete_entry_outputs(entry: &mut RawEntry) -> Result<(), ConvertError> {
    if entry.outputs.is_empty() {
        return Ok(());
    }
    if entry.routes().next().is_none() {
        synthesize_output_routes(entry)?;
    }

    for slot in 0..entry.outputs.len() {
        if entry.outputs[slot].routes.is_empty() {
            return Err(ConvertError::Unsupported(format!(
                "entry output #{slot} has no source"
            )));
        }

        let [route] = entry.outputs[slot].routes.as_slice() else {
            continue;
        };
        if let Some(source_resource) =
            super::graph_ops::extract_storage_view_source(&entry.graph, route.source.value)
        {
            let length = entry.outputs[slot].storage_length().cloned();
            entry.outputs[slot].kind = interface::EntryOutputKind::Storage {
                exposure: interface::BindingExposure::Host(source_resource),
                length,
            };
            entry.outputs[slot].resource = Some(source_resource);
        }

        if let Some(length) = runtime_filter_output_length(entry, slot)? {
            *entry.outputs[slot].storage_length_mut().expect("runtime Filter output is storage") = length;
        }
    }
    Ok(())
}

fn synthesize_output_routes(entry: &mut RawEntry) -> Result<(), ConvertError> {
    let Some((return_block, result)) = unique_value_return(&entry.graph) else {
        return Ok(());
    };
    let sources = output_route_sources(&mut entry.graph, &result, entry.outputs.len())
        .map_err(ConvertError::Internal)?;
    for (output, source) in entry.outputs.iter_mut().zip(sources) {
        output.routes.push(super::ir::RealizedOutputRoute {
            source: super::program::SlotSource {
                block: return_block,
                value: source,
            },
            writers: Vec::new(),
        });
    }
    Ok(())
}

fn unique_value_return(graph: &EGraph<Raw>) -> Option<(BlockId, ResultBinding<types::Type>)> {
    let mut returns = graph.skeleton.blocks.iter().filter_map(|(block, body)| {
        let SkeletonTerminator::Return(Some(result)) = &body.term else {
            return None;
        };
        Some((block, result.clone()))
    });
    let result = returns.next();
    assert!(
        returns.next().is_none(),
        "entry body has more than one value-returning terminator"
    );
    result
}

fn output_route_sources(
    graph: &mut EGraph<Raw>,
    result: &ResultBinding<types::Type>,
    output_count: usize,
) -> Result<Vec<ValueId>, String> {
    if output_count == 1 {
        return super::graph_ops::pack_result_values(graph, result).map(|source| vec![source]);
    }
    let fields = result.top_level_fields();
    if fields.len() != output_count {
        return Err(format!(
            "entry result has {} logical fields for {output_count} declared outputs",
            fields.len()
        ));
    }
    fields.iter().map(|field| super::graph_ops::pack_result_values(graph, field)).collect()
}

/// Return `Some(length)` when the route is produced by a runtime Filter.
/// The nested option preserves the distinction between "not a Filter" and a
/// Filter whose input representation cannot express a host allocation policy.
fn runtime_filter_output_length(
    entry: &RawEntry,
    slot: usize,
) -> Result<Option<Option<BufferLen>>, ConvertError> {
    let [route] = entry.outputs[slot].routes.as_slice() else {
        return Ok(None);
    };
    if entry.outputs[slot].resource.is_none() {
        return Ok(None);
    }
    let effect_index = entry.graph.side_effect_index();
    let Some((effect, _, _)) = effect_index.effect_result_field(&entry.graph, route.source.value) else {
        return Ok(None);
    };
    let SideEffectKind::Soac(SoacEffect(_, Soac::Filter(op))) = effect.kind() else {
        return Ok(None);
    };
    let filter::RawOutput::Runtime { .. } = &op.state.output else {
        return Ok(None);
    };
    let input =
        op.body.inputs.first().ok_or_else(|| ConvertError::Internal("Filter has no array input".into()))?;
    let input_element = input.element();
    let output_element = op.body.output_element_type();
    let length = input.array.array_buffer().and_then(|region| {
        let Type::Constructed(TypeName::Buffer(binding), _) = region else {
            return None;
        };
        Some(BufferLen::LikeInput {
            set: binding.set,
            binding: binding.binding,
            elem_bytes: ssa::layout::storage_elem_stride(&output_element)?,
            src_elem_bytes: ssa::layout::storage_elem_stride(&input_element)?,
        })
    });
    Ok(Some(length))
}

fn is_storage_image_ty(ty: &Type<TypeName>) -> bool {
    matches!(ty, Type::Constructed(TypeName::StorageTexture, _))
}

fn interface_type_binding(ty: &Type<TypeName>) -> Option<BindingRef> {
    let Type::Constructed(TypeName::Buffer(binding), _) = ty.array_buffer()? else {
        return None;
    };
    Some(*binding)
}

fn interface_parameter_representation(ty: &Type<TypeName>) -> OperandType<BindingRef, Type<TypeName>> {
    let binding = interface_type_binding(&ty);
    let is_view = ty.array_variant().is_some() && (types::is_array_variant_view(&ty) || binding.is_some());
    let ty = viewify_interface_arrays(ty);
    if is_view {
        OperandType::View(ViewType {
            array: ty,
            region: binding.map(PlaceRegion::Resource).unwrap_or(PlaceRegion::Parametric),
            access: PlaceAccess::ReadOnly,
        })
    } else {
        OperandType::Value(ty)
    }
}

fn normalize_interface_function_parameters(function: &mut Func<Raw>) {
    let parameter_ids = function.params.ids().collect::<Vec<_>>();
    for parameter in parameter_ids {
        let representation =
            interface_parameter_representation(function.params.get(parameter).unwrap().ty());
        if representation == *function.params.get(parameter).unwrap().representation() {
            continue;
        }
        let source = function
            .graph
            .nodes
            .iter()
            .find_map(|(value, definition)| {
                matches!(
                    definition.kind(),
                    ValueKind::FuncParam { parameter: candidate } if *candidate == parameter
                )
                .then_some(value)
            })
            .expect("function parameter has a graph binding");
        super::graph_ops::retype_projection_tree(&mut function.graph, source, representation.ty());
        *function.params.get_mut(parameter).unwrap().representation_mut() = representation;
    }
    function.graph.canonicalize_boundary_operands();
}

fn viewify_interface_arrays(ty: &Type<TypeName>) -> Type<TypeName> {
    if ty.array_variant().is_some()
        && (types::is_array_variant_view(ty) || interface_type_binding(ty).is_some())
    {
        return types::view_array_of(ty, ty.array_buffer().cloned().unwrap_or_else(types::no_buffer));
    }
    match ty {
        Type::Constructed(name, fields)
            if matches!(
                name,
                TypeName::Tuple(_) | TypeName::Record(_) | TypeName::Unit | TypeName::SideEffect
            ) =>
        {
            Type::Constructed(
                name.clone(),
                fields.iter().map(viewify_interface_arrays).collect(),
            )
        }
        _ => ty.clone(),
    }
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
    locals: LookupMap<SymbolId, ValueId>,
    /// Top-level definitions.
    top_level: &'a LookupMap<SymbolId, &'a TlcDef>,
    /// Symbol table.
    symbols: &'a SymbolTable,
    /// Cache for inlined constant bodies.
    inlined_constants: LookupMap<SymbolId, ValueId>,
    /// Identities of hoisted pure constants.
    pure_constants: LookupSet<SymbolId>,
    /// Canonical callable metadata built before any bodies or calls.
    callable_boundaries: &'a LookupMap<
        SymbolId,
        (
            Parameters<BindingRef, Type<TypeName>>,
            FunctionResult<Type<TypeName>>,
            CallEffects,
        ),
    >,
    /// Program-wide identity source for effect-chain endpoints.
    effect_ids: &'b mut IdSource<EffectToken>,
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
    output_sources: Vec<Vec<egir::program::SlotSource>>,
    /// Module-wide id factory for host-visible auto-storage binding numbers.
    /// Compiler resources never draw from this namespace.
    binding_ids: &'b mut IdSource<u32>,
    /// Program-wide arenas borrowed exclusively for this conversion.
    arenas: &'b mut ConversionArenas,
}

impl<'a, 'b> Converter<'a, 'b> {
    fn new(
        top_level: &'a LookupMap<SymbolId, &'a TlcDef>,
        symbols: &'a SymbolTable,
        pure_constants: LookupSet<SymbolId>,
        callable_boundaries: &'a LookupMap<
            SymbolId,
            (
                Parameters<BindingRef, Type<TypeName>>,
                FunctionResult<Type<TypeName>>,
                CallEffects,
            ),
        >,
        binding_ids: &'b mut IdSource<u32>,
        effect_ids: &'b mut IdSource<EffectToken>,
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
            callable_boundaries,
            effect_ids,
            current_span: None,
            output_sources: Vec::new(),
            binding_ids,
            arenas,
        }
    }

    fn function_id(&self, symbol: SymbolId) -> FunctionId {
        self.arenas.function_ids[&symbol]
    }

    /// Intern a pure node, attaching the current term's span (if any).
    /// Use in preference to `self.graph.intern_pure` so spans flow through.
    fn intern_pure(&mut self, op: PureOp, operands: SmallVec<[ValueId; 4]>, ty: Type<TypeName>) -> ValueId {
        self.intern_pure_at(op, operands, ty, self.current_span)
    }

    /// Intern a pure node with an explicit source span. Index-spine lowering
    /// uses this to retain the span of each nested source index while handling
    /// the complete spine in one conversion step.
    fn intern_pure_at(
        &mut self,
        op: PureOp,
        operands: SmallVec<[ValueId; 4]>,
        ty: Type<TypeName>,
        span: Option<Span>,
    ) -> ValueId {
        if let Some(folded) = self.graph.try_algebraic_fold(&op, &operands, &ty) {
            return folded;
        }
        self.graph.intern_pure(op, operands, ty, span)
    }

    fn alloc_effect(&mut self) -> EffectToken {
        super::graph_ops::alloc_effect(self.effect_ids)
    }

    /// Set the return terminator on the current block.
    fn set_return(&mut self, result: Option<ResultBinding<Type<TypeName>>>) {
        self.graph.skeleton.blocks[self.current_block].term = SkeletonTerminator::Return(result);
    }

    // -- Entry-point emission helpers (thin delegations to `graph_ops`) --

    fn emit_storage_view(&mut self, binding: BindingRef, view_ty: Type<TypeName>) -> ValueId {
        super::graph_ops::intern_interface_view(&mut self.graph, binding, view_ty, self.current_span)
    }

    fn emit_storage_store(
        &mut self,
        view_nid: ValueId,
        index_nid: ValueId,
        value_nid: ValueId,
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

    fn convert_term(&mut self, term: &Term) -> Result<ValueId, ConvertError> {
        let ty = term.ty.clone();
        let saved_span = self.current_span;
        self.current_span = Some(term.span);
        let result = self.convert_term_kind(term, ty);
        self.current_span = saved_span;
        result
    }

    /// Convert a directly nested source index expression as one address chain.
    ///
    /// Recursively converting `view[i][j]` would convert `view[i]` first and
    /// therefore emit a `Load` of the complete row before the outer `[j]` is
    /// visible. Keep the nested indices together instead: composite values use
    /// ordinary `Index` nodes, while the first index into a storage view starts
    /// a `ViewIndex` place and every remaining coordinate extends that place
    /// with `PlaceIndex`. Only the final selected value is loaded.
    ///
    /// This applies only to a syntactically direct index spine. If an
    /// intermediate row escapes through a let binding, its standalone index is
    /// still converted normally and materialized as a value.
    fn convert_index_spine(
        &mut self,
        term: &Term,
        final_ty: Type<TypeName>,
    ) -> Result<ValueId, ConvertError> {
        let mut levels = SmallVec::<[(&Term, Type<TypeName>, Span); 4]>::new();
        let mut root = term;
        while let TermKind::Index { array, index } = &root.kind {
            levels.push((index.as_ref(), root.ty.clone(), root.span));
            root = array.as_ref();
        }
        levels.reverse();

        let mut value = self.convert_term(root)?;
        let mut current_ty = self.graph.nodes[value].ty.clone();
        let mut place = None;

        for (index, result_ty, span) in levels {
            let index = self.convert_term(index)?;
            if let Some(parent_place) = place {
                let next_place =
                    self.graph.add_index_place(parent_place, index, result_ty.clone(), Some(span));
                place = Some(next_place);
            } else if current_ty.array_variant().is_some_and(types::is_array_variant_view) {
                let view = self.graph.view_id(value);
                let next_place =
                    self.graph.add_view_index_place(view, index, result_ty.clone(), Some(span));
                place = Some(next_place);
            } else {
                value = self.intern_pure_at(
                    PureOp::Index,
                    smallvec![value, index],
                    result_ty.clone(),
                    Some(span),
                );
            }
            current_ty = result_ty;
        }

        if let Some(place) = place {
            Ok(super::graph_ops::emit_load(
                &mut self.graph,
                self.current_block,
                place,
                final_ty,
                self.effect_ids,
                Some(term.span),
            ))
        } else {
            Ok(value)
        }
    }

    fn convert_term_kind(&mut self, term: &Term, ty: Type<TypeName>) -> Result<ValueId, ConvertError> {
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
                let operands: SmallVec<[ValueId; 4]> =
                    parts.iter().map(|p| self.convert_term(p)).collect::<Result<_, _>>()?;
                let n = operands.len();
                Ok(self.intern_pure(PureOp::Tuple(n), operands, ty))
            }
            TermKind::TupleProj { tuple, idx } => {
                let base = self.convert_term(tuple)?;
                Ok(self.intern_pure(PureOp::Project { index: *idx as u32 }, smallvec![base], ty))
            }
            TermKind::Index { .. } => self.convert_index_spine(term, ty),
            TermKind::VecLit(parts) => {
                let operands: SmallVec<[ValueId; 4]> =
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
    /// merge; both routes retain the same declared interface destination.
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
                // resulting ValueId is just the Unit constant). The
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
                self.output_sources[slot_index].push(egir::program::SlotSource {
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
    ) -> Result<Option<ValueId>, ConvertError> {
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
            _ => {
                let value = self.convert_term(term)?;
                let abi = by_value_function_result::<WynLanguage>(term.ty.clone());
                let binding = super::graph_ops::bind_by_value_result(&mut self.graph, &abi, value);
                let fields = binding.top_level_fields();
                if fields.len() != output_count {
                    return Err(ConvertError::Internal(format!(
                        "compute result has {} logical fields for {output_count} declared outputs",
                        fields.len()
                    )));
                }
                for (slot, field) in fields.iter().enumerate() {
                    let value = super::graph_ops::pack_result_values(&mut self.graph, field)
                        .map_err(ConvertError::GraphError)?;
                    while self.output_sources.len() <= slot {
                        self.output_sources.push(Vec::new());
                    }
                    self.output_sources[slot].push(egir::program::SlotSource {
                        block: self.current_block,
                        value,
                    });
                }
                Ok(None)
            }
        }
    }

    // ========================================================================
    // Variable resolution
    // ========================================================================

    fn convert_var(&mut self, sym: SymbolId, ty: Type<TypeName>) -> Result<ValueId, ConvertError> {
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
    ) -> Result<ValueId, ConvertError> {
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
                    if let Some(binding) = binding {
                        Ok(self.intern_pure(PureOp::ResourceLen(binding), smallvec![], ty))
                    } else {
                        let arg_nids: SmallVec<[ValueId; 4]> =
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
                    let binding = types::storage_image_buffer(&args[0].ty).ok_or_else(|| {
                        ConvertError::GraphError(
                            "image_load operand has no concrete storage-image binding after monomorphization"
                                .into(),
                        )
                    })?;
                    let coord = self.convert_term(&args[1])?;
                    Ok(self.intern_pure(PureOp::StorageImageLoad(binding), smallvec![coord], ty))
                } else if *id == known.slice
                    && args.len() == 3
                    && ty.array_variant().is_some_and(types::is_array_variant_view)
                {
                    let parent = self.convert_term(&args[0])?;
                    let start = self.convert_term(&args[1])?;
                    let end = self.convert_term(&args[2])?;
                    let index_ty = self.graph.nodes[end].ty.clone();
                    let len = self.intern_pure(
                        PureOp::BinOp(BinaryOperator::Subtract),
                        smallvec![end, start],
                        index_ty,
                    );
                    Ok(super::graph_ops::intern_inherited_view(
                        &mut self.graph,
                        parent,
                        start,
                        len,
                        ty,
                        self.current_span,
                    ))
                } else {
                    let arg_nids: SmallVec<[ValueId; 4]> =
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
                let _arg_nids: Vec<ValueId> =
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
    ) -> Result<ValueId, ConvertError> {
        if let Some(def) = self.top_level.get(&symbol) {
            if def.arity == args.len() {
                let operands: SmallVec<[ValueId; 4]> =
                    args.iter().map(|argument| self.convert_term(argument)).collect::<Result<_, _>>()?;
                return self.emit_named_call(symbol, operands, ty);
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
        operands: SmallVec<[ValueId; 4]>,
        _ty: Type<TypeName>,
    ) -> Result<ValueId, ConvertError> {
        let function = self.function_id(symbol);
        let (parameters, result, effects) =
            self.callable_boundaries.get(&symbol).cloned().ok_or_else(|| {
                ConvertError::Internal(format!("missing callable boundary for {symbol:?}"))
            })?;
        let arguments = operands.into_iter().map(|value| self.graph.operand_ref(value)).collect::<Vec<_>>();
        let effect_tokens =
            (!matches!(effects, CallEffects::Pure)).then(|| (self.alloc_effect(), self.alloc_effect()));
        let (_, binding) = self
            .graph
            .emit_call(
                self.current_block,
                function,
                &parameters,
                &result,
                arguments,
                effects,
                effect_tokens,
                self.current_span,
            )
            .map_err(ConvertError::GraphError)?;
        super::graph_ops::pack_result_values(&mut self.graph, &binding).map_err(ConvertError::GraphError)
    }

    // ========================================================================
    // Side-effectful intrinsic helpers — called from the Builtin-arm
    // dispatch on `BuiltinLowering::Intrinsic(StorageIndex|StorageStore)`.
    // ========================================================================

    fn lower_storage_index(&mut self, args: &[Term], ty: Type<TypeName>) -> Result<ValueId, ConvertError> {
        let binding = literal_binding(args, "_w_intrinsic_storage_index")?;
        let index_nid = self.convert_term(&args[2])?;
        let view_nid = self.emit_storage_view(binding, ty.clone());
        let view = self.graph.view_id(view_nid);
        let place_nid = self.graph.add_view_index_place(view, index_nid, ty.clone(), self.current_span);
        Ok(super::graph_ops::emit_load(
            &mut self.graph,
            self.current_block,
            place_nid,
            ty.clone(),
            self.effect_ids,
            self.current_span,
        ))
    }

    fn lower_storage_store(&mut self, args: &[Term]) -> Result<ValueId, ConvertError> {
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
    fn lower_image_with(&mut self, args: &[Term], ty: Type<TypeName>) -> Result<ValueId, ConvertError> {
        let binding = types::storage_image_buffer(&args[0].ty).ok_or_else(|| {
            ConvertError::GraphError(
                "storage-image update operand has no concrete storage-image binding after monomorphization"
                    .into(),
            )
        })?;
        let arg_nids: SmallVec<[ValueId; 4]> =
            args[1..].iter().map(|a| self.convert_term(a)).collect::<Result<_, _>>()?;
        let effect_in = self.alloc_effect();
        let effect_out = self.alloc_effect();
        self.graph.skeleton.blocks[self.current_block].side_effects.push(SideEffect {
            kind: SideEffectKind::Effect(EffectOp::Op {
                tag: op::OpTag::StorageImageStore(binding),
            }),
            operands: arg_nids.into_iter().map(OperandRef::Value).collect(),
            result: None,
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
    ) -> Result<ValueId, ConvertError> {
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
            args: self.graph.admit_flow_values(result_nid.map(|_| then_result)),
        };

        // Else branch.
        self.current_block = else_block;
        let else_result = self.convert_term(else_branch)?;
        self.graph.skeleton.blocks[self.current_block].term = SkeletonTerminator::Branch {
            target: merge_block,
            args: self.graph.admit_flow_values(result_nid.map(|_| else_result)),
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
    ) -> Result<ValueId, ConvertError> {
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

    fn storage_image_placeholder(&mut self, ty: &Type<TypeName>) -> ValueId {
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
    ) -> Result<ValueId, ConvertError> {
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
    ) -> Result<ValueId, ConvertError> {
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
            args: self.graph.admit_flow_values([zero]),
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
            args: self.graph.admit_flow_values([next_i]),
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
    ) -> Result<ValueId, ConvertError> {
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
            args: self.graph.admit_flow_values([zero]),
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
            args: self.graph.admit_flow_values([next_i]),
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
    ) -> Result<ValueId, ConvertError> {
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
            args: self.graph.admit_flow_values([init_nid]),
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
            else_args: self.graph.admit_flow_values([acc_nid]),
        };

        // Body: convert body, branch back to header
        self.current_block = body_block;
        let new_acc = self.convert_term(body)?;
        self.graph.skeleton.blocks[self.current_block].term = SkeletonTerminator::Branch {
            target: header,
            args: self.graph.admit_flow_values([new_acc]),
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
    ) -> Result<ValueId, ConvertError> {
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
            args: self.graph.admit_flow_values([init_nid, zero]),
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
            else_args: self.graph.admit_flow_values([acc_nid]),
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
            args: self.graph.admit_flow_values([new_acc, next_i]),
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
    ) -> Result<ValueId, ConvertError> {
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
            args: self.graph.admit_flow_values([init_nid, zero]),
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
            else_args: self.graph.admit_flow_values([acc_nid]),
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
            args: self.graph.admit_flow_values([new_acc, next_i]),
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

    fn convert_soac(&mut self, soac: &SoacOp, ty: Type<TypeName>) -> Result<ValueId, ConvertError> {
        match soac {
            SoacOp::Map {
                lam,
                inputs,
                destination,
            } => self.convert_soac_map(lam, inputs, *destination, ty),
            SoacOp::Reduce { op, ne, input, .. } => self.convert_soac_reduce(op, ne, input, ty),
            SoacOp::Scan {
                op,
                ne,
                input,
                destination,
            } => self.convert_soac_scan(op, ne, input, *destination, ty),
            SoacOp::Filter {
                pred,
                input,
                destination,
            } => self.convert_soac_filter(pred, input, *destination, ty),
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

    fn emit_routed_soac(
        &mut self,
        soac: Soac<Raw>,
        operands: SmallVec<[ValueId; 4]>,
        result: ResultBinding<Type<TypeName>>,
    ) -> ResultBinding<Type<TypeName>> {
        let span = self.current_span;
        let operands = operands.into_iter().map(|value| self.graph.operand_ref(value)).collect();
        super::graph_ops::emit_pending_soac(
            &mut self.graph,
            self.current_block,
            (),
            soac,
            operands,
            result,
            self.effect_ids,
            span,
        )
    }

    fn emit_soac(
        &mut self,
        soac: Soac<Raw>,
        operands: SmallVec<[ValueId; 4]>,
        ty: Type<TypeName>,
    ) -> ValueId {
        let result = super::graph_ops::alloc_by_value_effect_result(&mut self.graph, ty);
        let result = self.emit_routed_soac(soac, operands, result);
        let materialized = if result.field_count() == 1 {
            result.field(0).expect("a one-field result has field zero")
        } else {
            result
        };
        super::graph_ops::pack_result_values(&mut self.graph, &materialized)
            .expect("a by-value SOAC result can be assembled")
    }

    fn emit_shape_preserving_soac(
        &mut self,
        soac: Soac<Raw>,
        operands: SmallVec<[ValueId; 4]>,
        input: ValueId,
        ownership: SoacOwnership,
        tuple_ty: Type<TypeName>,
        result_ty: Type<TypeName>,
    ) -> Result<ValueId, ConvertError> {
        if ownership == SoacOwnership::UniqueInput {
            if let Some(place) =
                super::graph_ops::addressable_value_place(&mut self.graph, input, &result_ty)
            {
                let result = ResultBinding::product(
                    tuple_ty,
                    [ResultBinding::destination(
                        result_ty,
                        ResultDestination::Place(PlaceDestination::Fixed(place)),
                    )],
                );
                self.emit_routed_soac(soac, operands, result);
                return Ok(input);
            }
        }
        Ok(self.emit_soac(soac, operands, tuple_ty))
    }

    fn convert_soac_map(
        &mut self,
        sb: &SoacBody,
        inputs: &[ArrayExpr],
        ownership: SoacOwnership,
        result_ty: Type<TypeName>,
    ) -> Result<ValueId, ConvertError> {
        let f_symbol = self.lambda_fn_symbol(&sb.lam)?;
        let capture_nids: Vec<ValueId> =
            sb.data.captures.iter().map(|(_, _, t)| self.convert_term(t)).collect::<Result<_, _>>()?;
        let input_nids: Vec<ValueId> =
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
        let mut operands: SmallVec<[ValueId; 4]> = SmallVec::new();
        operands.extend_from_slice(&input_nids);

        // A uniquely owned storage input preserves its addressable
        // representation. Other shape-preserving maps inherit an input shape
        // when the logical result carries an unresolved size.
        let project_ty = if ownership == SoacOwnership::UniqueInput
            && input_arr_types[0].array_variant().is_some_and(types::is_array_variant_view)
        {
            input_arr_types[0].clone()
        } else {
            input_arr_types
                .first()
                .and_then(|inp| shape_preserving_result_ty(inp, &output_elem_ty, &result_ty))
                .unwrap_or_else(|| result_ty.clone())
        };
        let tuple_ty = Type::Constructed(TypeName::Tuple(1), vec![project_ty.clone()]);

        let map_region = self.function_id(f_symbol);
        let soac = Soac::Screma(screma::Op {
            inputs: input_arr_types.into_iter().map(SoacInputType::array).collect(),
            form: screma::ScremaForm {
                pre: screma::Lambda::region(
                    SegBody {
                        region: map_region,
                        captures: capture_nids
                            .into_iter()
                            .map(|value| self.graph.operand_ref(value))
                            .collect(),
                    },
                    input_elem_types,
                    vec![output_elem_ty.clone()],
                ),
                scans: Vec::new(),
                reductions: Vec::new(),
                post: screma::Lambda::identity(vec![output_elem_ty]),
            },
            result_state: vec![screma::ResultState { ownership }],
            state: screma::RawState,
        });
        self.emit_shape_preserving_soac(soac, operands, input_nids[0], ownership, tuple_ty, project_ty)
    }

    /// Convert `reduce_by_index` to a histogram with an identity bucket and an
    /// explicit associative read-combine-write operator.
    fn convert_soac_reduce_by_index(
        &mut self,
        dest: &tlc::Place,
        op: &SoacBody,
        ne: &Term,
        indices: &ArrayExpr,
        values: &ArrayExpr,
        result_ty: Type<TypeName>,
    ) -> Result<ValueId, ConvertError> {
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
        let index_type = types::array_elem(&index_array)
            .cloned()
            .ok_or_else(|| ConvertError::GraphError("reduce_by_index indices are not an array".into()))?;
        let value_type = types::array_elem(&value_array)
            .cloned()
            .ok_or_else(|| ConvertError::GraphError("reduce_by_index values are not an array".into()))?;
        let operator_parameters = op.lam.params.iter().map(|(_, ty)| ty.clone()).collect();
        let operator = screma::Lambda::region(
            SegBody {
                region: operator_region,
                captures: operator_captures
                    .into_iter()
                    .map(|value| self.graph.operand_ref(value))
                    .collect(),
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
                        destinations: vec![self.graph.view_id(dest_view)],
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
        dest: &tlc::Place,
        lam: &SoacBody,
        inputs: &[ArrayExpr],
        result_ty: Type<TypeName>,
    ) -> Result<ValueId, ConvertError> {
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

        let input_nids: Vec<ValueId> =
            inputs.iter().map(|ae| self.convert_array_expr_value(ae)).collect::<Result<_, _>>()?;
        let input_array_types: Vec<Type<TypeName>> =
            inputs.iter().zip(input_nids.iter()).map(|(ae, nid)| self.value_array_type(*nid, ae)).collect();
        let capture_nids: Vec<ValueId> =
            lam.data.captures.iter().map(|(_, _, t)| self.convert_term(t)).collect::<Result<_, _>>()?;

        let operands: SmallVec<[ValueId; 4]> = input_nids.into_iter().collect();
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
                            captures: capture_nids
                                .into_iter()
                                .map(|value| self.graph.operand_ref(value))
                                .collect(),
                        },
                        lam.lam.params.iter().map(|(_, ty)| ty.clone()).collect(),
                        vec![index_type, value_type],
                    ),
                    operations: vec![hist::HistOp {
                        emission: hist::Emission::Always,
                        shape: vec![destination_length],
                        race_factor,
                        destinations: vec![self.graph.view_id(dest_view)],
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
        dest: &tlc::Place,
        lam: &SoacBody,
        inputs: &[ArrayExpr],
        input_dimensions: &[Vec<u8>],
        domain_rank: u8,
        _result_ty: Type<TypeName>,
    ) -> Result<ValueId, ConvertError> {
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
            if super::types::as_soa_tuple(&input_array_type).is_some() {
                continue;
            }
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
        for &input_node in &input_nodes {
            let producers = super::graph_ops::value_producer_closure(&self.graph, [input_node]);
            let materialized_soac = self.graph.skeleton.blocks.values().any(|block| {
                block.side_effects.iter().any(|effect| {
                    effect.result.as_ref().is_some_and(|result| {
                        result.values().iter().any(|value| producers.nodes.contains(value))
                    }) && matches!(effect.kind, SideEffectKind::Soac(_))
                })
            });
            if materialized_soac {
                return Err(ConvertError::Unsupported(
                    "bucket_scatter requires direct ranked producer composition; its generated item array could not be fused without materialization"
                        .into(),
                ));
            }
        }
        let input_arrays = inputs
            .iter()
            .zip(&input_nodes)
            .map(|(input, node)| self.value_array_type(*node, input))
            .collect::<Vec<_>>();
        let input_layouts = input_nodes
            .iter()
            .zip(&input_arrays)
            .zip(input_dimensions)
            .map(|((node, array), dimensions)| {
                let producers = super::graph_ops::value_producer_closure(&self.graph, [*node]);
                let storage_type = producers.nodes.iter().find_map(|producer| {
                    matches!(
                        self.graph.nodes[*producer].kind,
                        ValueKind::Pure {
                            op: PureOp::StorageView(_),
                            ..
                        }
                    )
                    .then(|| self.graph.nodes[*producer].ty.clone())
                });
                if super::types::as_soa_tuple(array).is_some() {
                    if contains_storage_view_type(array) {
                        return strided_fields_layout(array, dimensions.len())
                            .unwrap_or(ArrayLayout::StructureOfArrays);
                    }
                    if let Some(mut leaf) = storage_type {
                        for _ in dimensions {
                            let Some(element) = leaf.elem_type().cloned() else {
                                return ArrayLayout::StructureOfArrays;
                            };
                            leaf = element;
                        }
                        if let Type::Constructed(TypeName::Tuple(_), fields) = leaf {
                            let mut offset = 0u32;
                            let mut field_offsets = Vec::with_capacity(fields.len());
                            for field in &fields {
                                field_offsets.push(offset);
                                let Some(size) = ssa::layout::type_byte_size(field) else {
                                    return ArrayLayout::StructureOfArrays;
                                };
                                let Some(next) = offset.checked_add(size) else {
                                    return ArrayLayout::StructureOfArrays;
                                };
                                offset = next;
                            }
                            return ArrayLayout::StridedFields {
                                element_stride_bytes: offset,
                                field_offsets_bytes: field_offsets,
                            };
                        }
                    }
                    ArrayLayout::StructureOfArrays
                } else if storage_type.is_some() {
                    ArrayLayout::StorageAos
                } else if types::is_virtual_array(array) {
                    ArrayLayout::Generated
                } else {
                    ArrayLayout::Composite
                }
            })
            .collect::<Vec<_>>();
        let captures = lam
            .data
            .captures
            .iter()
            .map(|(_, _, term)| self.convert_term(term))
            .collect::<Result<Vec<_>, _>>()?;

        let u32_type = Type::Constructed(TypeName::UInt(32), vec![]);
        let counts_type = types::sized_array(bucket_count as usize, u32_type.clone());
        let hist_result_type =
            Type::Constructed(TypeName::Tuple(2), vec![counts_type.clone(), u32_type.clone()]);
        let bucket_count_node = self.intern_pure(
            PureOp::Int(bucket_count.to_string()),
            smallvec![],
            i32_type.clone(),
        );
        let capacity_node =
            self.intern_pure(PureOp::Int(capacity.to_string()), smallvec![], i32_type.clone());
        let race_factor = self.intern_pure(PureOp::Int("1".into()), smallvec![], i32_type);
        let body_region = self.function_id(function);
        let hist_result = self.emit_soac(
            Soac::Hist(hist::Op {
                inputs: input_arrays
                    .into_iter()
                    .zip(input_layouts)
                    .zip(input_dimensions)
                    .map(|((array, layout), dimensions)| {
                        SoacInputType::mapped(array, dimensions.clone()).with_layout(layout)
                    })
                    .collect(),
                form: hist::HistForm {
                    bucket: screma::Lambda::region(
                        SegBody {
                            region: body_region,
                            captures: captures
                                .into_iter()
                                .map(|value| self.graph.operand_ref(value))
                                .collect(),
                        },
                        lam.lam.params.iter().map(|(_, ty)| ty.clone()).collect(),
                        vec![active_type, key_type, value_type.clone()],
                    ),
                    operations: vec![hist::HistOp {
                        emission: hist::Emission::Guarded,
                        shape: vec![bucket_count_node],
                        race_factor,
                        destinations: vec![self.graph.view_id(dest_view)],
                        update: hist::Update::BucketInsert {
                            value_types: vec![dest.elem_ty.clone()],
                            results: hist::BucketInsertResults {
                                counts: hist::HistResultId(0),
                                overflow: hist::HistResultId(1),
                            },
                            capacity: capacity_node,
                        },
                    }],
                },
                state: hist::RawState,
            }),
            input_nodes.into_iter().collect(),
            hist_result_type,
        );
        let counts_value = self.intern_pure(
            PureOp::Project { index: 0 },
            smallvec![hist_result],
            counts_type.clone(),
        );
        let overflow_value = self.intern_pure(
            PureOp::Project { index: 1 },
            smallvec![hist_result],
            u32_type.clone(),
        );
        Ok(self.intern_pure(
            PureOp::Tuple(3),
            smallvec![dest_view, counts_value, overflow_value],
            Type::Constructed(TypeName::Tuple(3), vec![dest_view_ty, counts_type, u32_type]),
        ))
    }

    fn convert_soac_reduce(
        &mut self,
        op: &SoacBody,
        ne: &Term,
        input: &ArrayExpr,
        result_ty: Type<TypeName>,
    ) -> Result<ValueId, ConvertError> {
        let operator_symbol = self.lambda_fn_symbol(&op.lam)?;
        let capture_nids: Vec<ValueId> =
            op.data.captures.iter().map(|(_, _, t)| self.convert_term(t)).collect::<Result<_, _>>()?;
        let arr_nid = self.convert_array_expr_value(input)?;
        let arr_ty = self.value_array_type(arr_nid, input);
        let init_nid = self.convert_term(ne)?;

        // Emit as Screma { 0 maps, 1 Reduce accumulator } + project field
        // 0. Reduce's `op` is both the step (per-element) and the
        // reduce_op (phase 2 combiner).
        let operands: SmallVec<[ValueId; 4]> = smallvec![arr_nid];
        let tuple_ty = Type::Constructed(TypeName::Tuple(1), vec![result_ty.clone()]);
        let op_region = self.function_id(operator_symbol);
        Ok(self.emit_soac(
            Soac::Screma(screma::Op {
                inputs: vec![SoacInputType::array(arr_ty)],
                form: screma::ScremaForm {
                    pre: screma::Lambda::identity(vec![result_ty.clone()]),
                    scans: Vec::new(),
                    reductions: vec![screma::Reduce {
                        operator: screma::Lambda::region(
                            SegBody {
                                region: op_region,
                                captures: capture_nids
                                    .into_iter()
                                    .map(|value| self.graph.operand_ref(value))
                                    .collect(),
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
                    ownership: SoacOwnership::Fresh,
                }],
                state: screma::RawState,
            }),
            operands,
            tuple_ty,
        ))
    }

    fn convert_soac_scan(
        &mut self,
        op: &SoacBody,
        ne: &Term,
        input: &ArrayExpr,
        ownership: SoacOwnership,
        result_ty: Type<TypeName>,
    ) -> Result<ValueId, ConvertError> {
        let operator_symbol = self.lambda_fn_symbol(&op.lam)?;
        let capture_nids: Vec<ValueId> =
            op.data.captures.iter().map(|(_, _, t)| self.convert_term(t)).collect::<Result<_, _>>()?;
        let arr_nid = self.convert_array_expr_value(input)?;
        let arr_ty = self.value_array_type(arr_nid, input);
        let init_nid = self.convert_term(ne)?;

        let operands: SmallVec<[ValueId; 4]> = smallvec![arr_nid];

        // A scan preserves the input shape while changing its element to the
        // accumulator type. A consuming scan routes that result to the input
        // destination.
        let project_ty = if ownership == SoacOwnership::UniqueInput
            && arr_ty.array_variant().is_some_and(types::is_array_variant_view)
        {
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
        let soac = Soac::Screma(screma::Op {
            inputs: vec![SoacInputType::array(arr_ty)],
            form: screma::ScremaForm {
                pre: screma::Lambda::identity(vec![scan_elem_ty.clone()]),
                scans: vec![screma::Scan {
                    operator: screma::Lambda::region(
                        SegBody {
                            region: op_region,
                            captures: capture_nids
                                .into_iter()
                                .map(|value| self.graph.operand_ref(value))
                                .collect(),
                        },
                        vec![scan_elem_ty.clone(), scan_elem_ty.clone()],
                        vec![scan_elem_ty.clone()],
                    ),
                    neutral: vec![init_nid],
                }],
                reductions: Vec::new(),
                post: screma::Lambda::identity(vec![scan_elem_ty.clone()]),
            },
            result_state: vec![screma::ResultState { ownership }],
            state: screma::RawState,
        });
        self.emit_shape_preserving_soac(soac, operands, arr_nid, ownership, tuple_ty, project_ty)
    }

    fn convert_soac_filter(
        &mut self,
        pred: &SoacBody,
        input: &ArrayExpr,
        ownership: SoacOwnership,
        result_ty: Type<TypeName>,
    ) -> Result<ValueId, ConvertError> {
        let predicate_symbol = self.lambda_fn_symbol(&pred.lam)?;
        let capture_nids: Vec<ValueId> =
            pred.data.captures.iter().map(|(_, _, t)| self.convert_term(t)).collect::<Result<_, _>>()?;
        let elem_ty = self.array_expr_elem_type(input);
        let arr_ty = self.array_expr_type(input);
        let arr_nid = self.convert_array_expr_value(input)?;

        let output_elem_ty = elem_ty.clone();
        let pred_body = SegBody {
            region: self.function_id(predicate_symbol),
            captures: capture_nids.into_iter().map(|value| self.graph.operand_ref(value)).collect(),
        };

        // `[input]` only — map/pred captures live on their `SegBody`s.
        let operands: SmallVec<[ValueId; 4]> = smallvec![arr_nid];

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
                    types::no_buffer(),
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
                        output: filter::RawOutput::Local {
                            capacity: size,
                            ownership,
                        },
                    },
                }),
                operands,
                bounded_result_ty,
            ));
        }

        // Runtime-sized input: preserve only the semantic result requirement.
        // Fusion may eliminate it; allocation binds backing and stored length
        // only if the result crosses an ABI or scheduling boundary.
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
                    output: filter::RawOutput::Runtime {
                        capacity: filter::RuntimeCapacity::LikeInput {
                            input: filter::FilterInputId(0),
                        },
                    },
                },
            }),
            operands,
            result_ty,
        ))
    }

    // ========================================================================
    // ArrayExpr
    // ========================================================================

    fn convert_array_expr(&mut self, ae: &ArrayExpr, ty: Type<TypeName>) -> Result<ValueId, ConvertError> {
        match ae {
            ArrayExpr::Var(vr, var_ty) => {
                let t = tlc::synthetic_atom_var_term(*vr, var_ty.clone());
                self.convert_term(&t)
            }
            // A `Zip` is the SoA form of a tuple-element array input: it lowers
            // to a `Tuple` of its component arrays (`[N](A,B)` ≡ `([N]A, [N]B)`),
            // the same node a `Tuple`-term input would build. `soa::normalize`
            // flattens `Map`-over-`Zip` ahead of EGIR, but an SoA constant
            // inlined into a non-`Map` consumer (e.g. `reduce`) still arrives as
            // a `Zip` here.
            ArrayExpr::Zip(children) => {
                let operands: SmallVec<[ValueId; 4]> =
                    children.iter().map(|c| self.convert_array_expr_value(c)).collect::<Result<_, _>>()?;
                let component_tys: Vec<Type<TypeName>> =
                    children.iter().map(|c| self.array_expr_type(c)).collect();
                let tuple_ty = Type::Constructed(TypeName::Tuple(component_tys.len()), component_tys);
                let n = operands.len();
                Ok(self.intern_pure(PureOp::Tuple(n), operands, tuple_ty))
            }
            ArrayExpr::Literal(terms) => {
                let operands: SmallVec<[ValueId; 4]> =
                    terms.iter().map(|t| self.convert_term(t)).collect::<Result<_, _>>()?;
                let n = operands.len();
                Ok(self.intern_pure(PureOp::ArrayLit(n), operands, ty))
            }
            ArrayExpr::Range { start, len, step } => {
                let start_nid = self.convert_term(start)?;
                let len_nid = self.convert_term(len)?;
                let mut operands: SmallVec<[ValueId; 4]> = smallvec![start_nid, len_nid];
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

    fn convert_array_expr_value(&mut self, ae: &ArrayExpr) -> Result<ValueId, ConvertError> {
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
    fn value_array_type(&self, nid: ValueId, fallback: &ArrayExpr) -> Type<TypeName> {
        if let Some(node) = self.graph.nodes.get(nid) {
            let ty = strip_existentials(&node.ty);
            if matches!(ty, Type::Constructed(TypeName::Array, _)) || as_soa_tuple(ty).is_some() {
                return ty.clone();
            }
        }
        strip_existentials(&self.array_expr_type(fallback)).clone()
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
        block.side_effects.iter().all(|effect| !graph.effect_requires_ordering(effect))
            && match &block.term {
                SkeletonTerminator::Return(value) => value.as_ref().is_none_or(|result| {
                    result.values().into_iter().all(|node| is_constant_node(graph, node, &mut memo))
                }),
                SkeletonTerminator::Branch { args, .. } => {
                    args.iter().all(|node| is_constant_node(graph, node.value(), &mut memo))
                }
                SkeletonTerminator::CondBranch {
                    cond,
                    then_args,
                    else_args,
                    ..
                } => std::iter::once(*cond)
                    .chain(then_args.iter().map(|value| value.value()))
                    .chain(else_args.iter().map(|value| value.value()))
                    .all(|node| is_constant_node(graph, node, &mut memo)),
                SkeletonTerminator::Unreachable => true,
            }
    })
}

fn is_constant_node(graph: &EGraph<Raw>, mut node: ValueId, memo: &mut LookupMap<ValueId, bool>) -> bool {
    while let Some(replacement) = graph.nodes[node].alias {
        node = replacement;
    }
    if let Some(result) = memo.get(&node) {
        return *result;
    }
    let result = match &graph.nodes[node].kind {
        ValueKind::Constant(_) => true,
        ValueKind::Pure { op, operands } => {
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
        ValueKind::Union { left, right } => {
            is_constant_node(graph, *left, memo) && is_constant_node(graph, *right, memo)
        }
        ValueKind::FuncParam { .. }
        | ValueKind::BlockParam { .. }
        | ValueKind::CallResult { .. }
        | ValueKind::PlaceLength { .. }
        | ValueKind::PlaceView { .. }
        | ValueKind::SideEffectResult => false,
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
        types::array_size(result_ty),
        Some(Type::Constructed(TypeName::Skolem(_), _))
    ) {
        return None;
    }
    Some(types::make_array1(
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
    binding_ids: &mut IdSource<u32>,
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
    let length_for = |binding: Option<BindingRef>,
                      ty: &Type<TypeName>|
     -> Result<Option<BufferLen>, ConvertError> {
        if binding.is_none() {
            return Ok(None);
        }
        let Some(elem_ty) = ty.elem_type() else {
            let bytes = ssa::layout::type_byte_size(ty).ok_or_else(|| {
                ConvertError::Internal(format!("output has no static byte layout: {ty:?}"))
            })?;
            return Ok(Some(BufferLen::Fixed {
                bytes: u64::from(bytes),
            }));
        };
        let elem_bytes = ssa::layout::storage_elem_stride(elem_ty).ok_or_else(|| {
            ConvertError::Internal(format!("output element has no static byte layout: {elem_ty:?}"))
        })?;
        if let Some(out_size) = types::array_size(ty) {
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
                let Some(in_size) = types::array_size(&input.ty) else {
                    continue;
                };
                if in_size == out_size {
                    let Some(in_elem_ty) = input.ty.elem_type() else {
                        continue;
                    };
                    let src_elem_bytes = ssa::layout::storage_elem_stride(in_elem_ty).ok_or_else(|| {
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
            let ty = types::canonical_storage_buffer_ty(source_ty);
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
                let ty = types::canonical_storage_buffer_ty(ty);
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
        let ty = types::canonical_storage_buffer_ty(source_ty);
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
