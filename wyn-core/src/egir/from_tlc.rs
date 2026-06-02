//! Direct TLC to EGraph conversion.
//!
//! Converts a TLC program directly to the acyclic e-graph representation,
//! bypassing the sequential SSA construction of `to_ssa`. Pure operations
//! are hash-consed (giving GVN for free), and the result is elaborated
//! back to `FuncBody` via demand-driven scheduling (giving DCE for free).

use crate::builtins::catalog;
use crate::ssa::types::EntryOutput;
use crate::ssa::types::IoDecoration;
use crate::tlc::SoacBody;
use crate::tlc::VarRef;
use std::collections::{HashMap, HashSet};

use super::publish::PipelineDescriptorPublish;
use super::types::EffectToken;
use crate::ast::{Span, TypeName};
use crate::binding_layout::{
    extract_io_decoration, extract_sampler_binding, extract_storage_binding, extract_texture_binding,
    extract_uniform_binding,
};
use crate::interface;
use crate::interface::{EntryParamBinding, EntryParamBindingKind};
use crate::ssa::framework::BlockId;
use crate::ssa::types::{ControlHeader, FuncBody, Function, InstKind, ValueRef};
use crate::tlc::{
    ArrayExpr, Def as TlcDef, DefMeta, Lambda, LoopKind, Program as TlcProgram, SoacOp, Term, TermKind,
};
use crate::types::TypeExt;
use crate::{BindingRef, SymbolId, SymbolTable};
use polytype::Type;
use smallvec::{SmallVec, smallvec};

use super::program::{EgirEntry, EgirFunc, EgirInner};
use super::types::*;
use crate::pipeline_descriptor::{BufferLen, PipelineDescriptor};

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

#[derive(Debug)]
pub enum ConvertError {
    /// Error during EGraph construction.
    GraphError(String),
    /// Unsupported TLC construct (todo).
    Unsupported(String),
    /// Compiler invariant violated — a downstream pass should have made
    /// this state unreachable. Surfaces as a propagated error rather than
    /// a panic so the caller can label it as an internal compiler error
    /// in user-facing output.
    Internal(String),
}

impl std::fmt::Display for ConvertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConvertError::GraphError(msg) => write!(f, "EGraph conversion error: {}", msg),
            ConvertError::Unsupported(msg) => write!(f, "Unsupported: {}", msg),
            ConvertError::Internal(msg) => write!(f, "internal compiler error: {}", msg),
        }
    }
}

impl std::error::Error for ConvertError {}

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

/// Read-only state shared across every converter built during a single
/// `run` — the top-level def index, the arity-0 name → symbol map, and
/// the symbol table. Acts as a factory: `new_converter` snapshots the
/// caller's current `pure_constants` set into a fresh `Converter`,
/// keeping the per-call `clone()` inside one method.
struct GlobalContext<'a> {
    top_level: &'a HashMap<SymbolId, &'a TlcDef>,
    constants_by_name: &'a HashMap<String, SymbolId>,
    symbols: &'a SymbolTable,
}

impl<'a> GlobalContext<'a> {
    fn new_converter(&self, pure_constants: &HashSet<String>) -> Converter<'a> {
        Converter::new(
            self.top_level,
            self.constants_by_name,
            self.symbols,
            pure_constants.clone(),
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
pub fn run(
    program: &TlcProgram,
    mut pipeline: PipelineDescriptor,
    plans: &HashMap<String, crate::tlc::parallelize::ParallelizationPlan>,
) -> Result<EgirInner, ConvertError> {
    let top_level: HashMap<SymbolId, &TlcDef> = program.defs.iter().map(|d| (d.name, d)).collect();
    let symbols = &program.symbols;

    let constants_by_name = program.value_defs_by_name();

    let ctx = GlobalContext {
        top_level: &top_level,
        constants_by_name: &constants_by_name,
        symbols,
    };

    // Phase 1: detect pure constants. We elaborate each arity-0 def's body
    // through the full EGIR pipeline once (using a throwaway chain) to see if
    // it collapses to a purely-constant FuncBody. Constants are hoisted to
    // program scope and referenced by `PureOp::Global`.
    let mut pure_constant_names: HashSet<String> = HashSet::new();
    let mut constants = Vec::new();

    for def in &program.defs {
        if def.arity != 0 || !matches!(&def.meta, DefMeta::Function) {
            continue;
        }
        if matches!(&def.body.kind, TermKind::Extern(_)) {
            continue;
        }
        let def_name = symbols.get(def.name).expect("BUG: symbol not in table").clone();

        let mut converter = ctx.new_converter(&pure_constant_names);
        if let Ok(result_nid) = converter.convert_term(&def.body) {
            converter.set_return(Some(result_nid));
            if let Some(body) = converter.probe_constant_body(def.body.ty.clone()) {
                if is_purely_constant_body(&body) {
                    pure_constant_names.insert(def_name.clone());
                    constants.push(crate::ssa::types::Constant { name: def_name, body });
                    continue;
                }
            }
        }
    }

    // Phase 2: convert functions and entry points into raw EGIR records.
    let mut functions: Vec<EgirFunc> = Vec::new();
    let mut externs: Vec<Function> = Vec::new();
    let mut entry_points: Vec<EgirEntry> = Vec::new();

    for def in &program.defs {
        match &def.meta {
            DefMeta::Function | DefMeta::LiftedLambda => {
                let def_name = symbols.get(def.name).expect("BUG: symbol not in table");
                if pure_constant_names.contains(def_name) {
                    continue;
                }
                match convert_function(def, &ctx, &pure_constant_names)? {
                    ConvertedFunc::Extern(f) => externs.push(f),
                    ConvertedFunc::Regular(fe) => functions.push(fe),
                }
            }
            DefMeta::EntryPoint(entry) => {
                let workgroup = pipeline.workgroup_size_of(&entry.name);
                // If TLC parallelize reserved a specific output binding
                // for this entry, honor it; otherwise auto-allocate.
                // Map and Scan results are single `EntryOutput` buffers whose
                // binding `build_entry_outputs` may pin (gather pre-passes do).
                // Reduce/Redomap manage their result inside the TLC two-phase
                // plan, so `forced_output` reports `None` for them.
                //
                // Fallback for `--single-stage` / `parallelize_soacs(disable=
                // true)`: no plan exists, but `lift_gathers` still records its
                // intended gather binding as an Output-role
                // `StorageBindingDecl` carrying a `length` (the gather-prepass
                // marker shape). Use that as the forced output so the prepass
                // map writes the gather buffer instead of having its result
                // auto-allocated onto a colliding binding.
                let plan = plans.get(&entry.name);
                let forced_output_binding = plan
                    .and_then(|p| p.bindings.forced_output())
                    .or_else(|| gather_prepass_forced_output(entry));
                // A parallel map/scan writes one output element per thread, so
                // its storage output is dispatch-sized. Gather pre-passes are
                // parallel maps too; treat the lift_gathers fallback above the
                // same way. Reduce/Redomap results are single elements;
                // entries without a plan or fallback keep the host's default
                // sizing.
                let dispatch_sized_outputs = plan.is_some_and(|p| {
                    use crate::tlc::parallelize::ParallelStrategy;
                    matches!(p.strategy, ParallelStrategy::Map | ParallelStrategy::Scan)
                }) || (plan.is_none()
                    && gather_prepass_forced_output(entry).is_some());
                let ep = convert_entry_point(
                    def,
                    entry,
                    &ctx,
                    &pure_constant_names,
                    workgroup,
                    forced_output_binding,
                    dispatch_sized_outputs,
                )?;
                entry_points.push(ep);
            }
        }
    }

    // Publish per-entry state from EGIR into the host-runtime-shared
    // `PipelineDescriptor` (see `egir::publish`):
    //   - implicit bindings the compiler invented to communicate
    //     between auto-generated entry points / phases, plus
    //     descriptor-level binding declarations routed from each
    //     entry's `#[storage]`/`#[uniform]`/etc. attributes
    //   - `#[location(N)]` graphics IO on vertex/fragment entries
    pipeline.publish_implicit_bindings(&entry_points);
    pipeline.publish_graphics_io(&entry_points);

    Ok(EgirInner::new(
        functions,
        externs,
        entry_points,
        constants,
        pipeline,
    ))
}

enum ConvertedFunc {
    Extern(Function),
    Regular(EgirFunc),
}

// ============================================================================
// Function conversion
// ============================================================================

fn convert_function(
    def: &TlcDef,
    ctx: &GlobalContext,
    pure_constants: &HashSet<String>,
) -> Result<ConvertedFunc, ConvertError> {
    let symbols = ctx.symbols;
    let def_name = symbol_name(symbols, def.name)?.to_string();

    // Extern functions: emit a 1-block Unreachable stub directly; no EGIR
    // passes apply to them. They flow through as `Function` records.
    if let TermKind::Extern(linkage_name) = &def.body.kind {
        let (param_types, ret_type) = extract_function_signature(&def.ty);
        let params: Vec<(Type<TypeName>, String)> =
            param_types.into_iter().enumerate().map(|(i, ty)| (ty, format!("arg{}", i))).collect();
        let mut builder = crate::ssa::builder::FuncBuilder::new(params, ret_type);
        builder
            .terminate(crate::ssa::types::Terminator::Unreachable)
            .map_err(|e| ConvertError::GraphError(e.to_string()))?;
        let body = builder.finish().map_err(|e| ConvertError::GraphError(e.to_string()))?;
        return Ok(ConvertedFunc::Extern(Function {
            name: def_name,
            body,
            span: def.body.span,
            linkage_name: Some(linkage_name.clone()),
        }));
    }

    // Regular functions: extract lambda params and build an EGraph.
    let (inner_body, params) = extract_lambda_params(&def.body);
    let ret_type = inner_body.ty.clone();
    let param_info: Vec<(Type<TypeName>, String)> = params
        .iter()
        .map(|(sym, ty)| Ok((ty.clone(), symbol_name(symbols, *sym)?.to_string())))
        .collect::<Result<_, ConvertError>>()?;

    let mut converter = ctx.new_converter(pure_constants);
    for (i, (sym, ty)) in params.iter().enumerate() {
        let nid = converter.graph.add_func_param(i, ty.clone());
        converter.locals.insert(*sym, nid);
    }
    let result = converter.convert_term(inner_body)?;
    converter.set_return(Some(result));

    let (graph, control_headers) = converter.into_graph_parts();
    Ok(ConvertedFunc::Regular(EgirFunc::new(
        def_name,
        def.body.span,
        None,
        param_info,
        ret_type,
        graph,
        control_headers,
    )))
}

/// When `parallelize_soacs` is disabled, `lift_gathers`-emitted prepass
/// entries are the only producers carrying a forced-binding intent —
/// recorded as an Output-role `StorageBindingDecl` with `length: Some(_)`
/// (the gather-buffer marker). Recover that binding so
/// `build_entry_outputs` pins the prepass's map output to it instead of
/// auto-allocating onto a colliding slot.
fn gather_prepass_forced_output(entry: &interface::EntryDecl) -> Option<BindingRef> {
    entry
        .storage_bindings
        .iter()
        .find(|d| matches!(d.role, interface::StorageRole::Output) && d.length.is_some())
        .map(|d| d.binding)
}

fn convert_entry_point(
    def: &TlcDef,
    entry: &interface::EntryDecl,
    ctx: &GlobalContext,
    pure_constants: &HashSet<String>,
    workgroup: (u32, u32, u32),
    forced_output_binding: Option<BindingRef>,
    dispatch_sized_outputs: bool,
) -> Result<EgirEntry, ConvertError> {
    use crate::ssa::types::{EntryInput, ExecutionModel, IoDecoration, PushConstantSlot};

    let symbols = ctx.symbols;
    let def_name = symbol_name(symbols, def.name)?;
    let (inner_body, params) = extract_lambda_params(&def.body);
    let is_compute = matches!(entry.entry_type, interface::Attribute::Compute);

    let ret_type = inner_body.ty.clone();
    let param_info: Vec<(Type<TypeName>, String)> = params
        .iter()
        .map(|(sym, ty)| Ok((ty.clone(), symbol_name(symbols, *sym)?.to_string())))
        .collect::<Result<_, ConvertError>>()?;

    let mut converter = ctx.new_converter(pure_constants);

    // Build entry inputs alongside the symbol → NodeId bindings. A compute
    // entry param that's a tuple-of-unsized-arrays gets one storage binding
    // per field (SoA lowered the source `[]T` of tuples into a tuple of
    // `[]T`s, but entry I/O can't carry a tuple handle — each runtime-sized
    // array needs its own buffer). The body still references the original
    // tuple symbol, so we reconstruct it as a `Tuple(views…)` node.
    let mut inputs: Vec<EntryInput> = Vec::with_capacity(params.len());
    let mut pc_offset: u32 = 0;

    // `entry.param_bindings` is dense — same length as `params`, with
    // `None` for non-storage params. Walking them in lockstep means the
    // layout and the body params can't drift; the layout pass enforces
    // the alignment once at the call site, and consumers just zip.
    debug_assert_eq!(
        entry.param_bindings.len(),
        params.len(),
        "entry.param_bindings length must match body params"
    );

    for (i, ((sym, ty), param_binding)) in params.iter().zip(entry.param_bindings.iter()).enumerate() {
        let name = symbol_name(symbols, *sym)?;
        let decoration = entry.params.get(i).and_then(extract_io_decoration);
        let size_hint = entry.params.get(i).and_then(extract_size_hint);
        let uniform_binding = entry.params.get(i).and_then(extract_uniform_binding);
        let attr_storage_binding = entry.params.get(i).and_then(extract_storage_binding);
        let texture_binding = entry.params.get(i).and_then(extract_texture_binding);
        let sampler_binding = entry.params.get(i).and_then(extract_sampler_binding);

        // Uniqueness is an ownership-tracking concept that's already been
        // consumed by `apply_ownership`; codegen operates on the stripped
        // shape so `*[N]T` and `[N]T` lower identically.
        let ty = crate::types::strip_unique(ty);
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
                    ty: field_ty.clone(),
                    decoration: None,
                    size_hint: None,
                    storage_binding: Some(slot.binding),
                    uniform_binding: None,
                    push_constant: None,
                    texture_binding: None,
                    sampler_binding: None,
                });
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

        inputs.push(EntryInput {
            name: name.to_string(),
            ty: ty.clone(),
            decoration,
            size_hint,
            storage_binding,
            uniform_binding,
            push_constant,
            texture_binding,
            sampler_binding,
        });
    }
    let binding_num: u32 = entry.param_bindings.iter().flatten().map(|b| b.buffer_count()).sum();

    let execution_model = match &entry.entry_type {
        interface::Attribute::Vertex => ExecutionModel::Vertex,
        interface::Attribute::Fragment => ExecutionModel::Fragment,
        interface::Attribute::Compute => ExecutionModel::Compute {
            local_size: workgroup,
        },
        _ => panic!("Invalid entry type attribute: {:?}", entry.entry_type),
    };

    let outputs = build_entry_outputs(
        entry,
        &inner_body.ty,
        is_compute,
        binding_num,
        forced_output_binding,
        dispatch_sized_outputs,
    )?;
    let is_unit_return = matches!(ret_type, Type::Constructed(TypeName::Unit, _));

    // Convert body. Output assignment (storing the result into the bound
    // storage views / graphics output slots, and retargeting tail
    // Map/Scan SOACs to stream directly into a runtime-sized output) is a
    // separate, uniform pass — `egir::assign_outputs`, run right after
    // this conversion. Here we just leave the body terminating in its
    // single tail value (or `None` for a unit entry).
    let result_nid = converter.convert_term(inner_body)?;
    if is_unit_return {
        converter.set_return(None);
    } else {
        converter.set_return(Some(result_nid));
    }

    let (graph, control_headers) = converter.into_graph_parts();
    Ok(EgirEntry::new(
        def_name.to_string(),
        def.body.span,
        execution_model,
        inputs,
        outputs,
        entry.storage_bindings.clone(),
        param_info,
        ret_type,
        graph,
        control_headers,
    ))
}

// ============================================================================
// Converter
// ============================================================================

struct Converter<'a> {
    /// The e-graph being built.
    graph: EGraph,
    /// Current skeleton block for side effects and terminators.
    current_block: BlockId,
    /// TLC variable → EGraph node mapping.
    locals: HashMap<SymbolId, NodeId>,
    /// Top-level definitions.
    top_level: &'a HashMap<SymbolId, &'a TlcDef>,
    /// Arity-0 defs indexed by name.
    constants_by_name: &'a HashMap<String, SymbolId>,
    /// Symbol table.
    symbols: &'a SymbolTable,
    /// Cache for inlined constant bodies.
    inlined_constants: HashMap<String, NodeId>,
    /// Names of hoisted pure constants.
    pure_constants: HashSet<String>,
    /// Control headers for structured control flow (SPIR-V).
    control_headers: HashMap<BlockId, ControlHeader>,
    /// Effect token counter.
    next_effect: u32,
    /// Span of the term currently being converted. Threaded through every
    /// pure-node intern and side-effect push so backend errors can blame
    /// the originating source. Pushed/popped in `convert_term`; `None`
    /// only outside any term conversion (e.g. entry-point glue).
    current_span: Option<Span>,
}

impl<'a> Converter<'a> {
    fn new(
        top_level: &'a HashMap<SymbolId, &'a TlcDef>,
        constants_by_name: &'a HashMap<String, SymbolId>,
        symbols: &'a SymbolTable,
        pure_constants: HashSet<String>,
    ) -> Self {
        let graph = EGraph::new();
        let entry = graph.skeleton.entry;
        Converter {
            graph,
            current_block: entry,
            locals: HashMap::new(),
            top_level,
            constants_by_name,
            symbols,
            inlined_constants: HashMap::new(),
            pure_constants,
            control_headers: HashMap::new(),
            next_effect: 1,
            current_span: None,
        }
    }

    /// Intern a pure node, attaching the current term's span (if any).
    /// Use in preference to `self.graph.intern_pure` so spans flow through.
    fn intern_pure(&mut self, op: PureOp, operands: SmallVec<[NodeId; 4]>, ty: Type<TypeName>) -> NodeId {
        self.graph.intern_pure_with_span(op, operands, ty, self.current_span)
    }

    fn alloc_effect(&mut self) -> EffectToken {
        super::graph_ops::alloc_effect(&mut self.next_effect)
    }

    /// Set the return terminator on the current block.
    fn set_return(&mut self, result: Option<NodeId>) {
        self.graph.skeleton.blocks[self.current_block].term = SkeletonTerminator::Return(result);
    }

    // -- Entry-point emission helpers (thin delegations to `graph_ops`) --

    fn emit_storage_view(&mut self, binding: BindingRef, view_ty: Type<TypeName>) -> NodeId {
        super::graph_ops::intern_storage_view(&mut self.graph, binding, view_ty, self.current_span)
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
            &mut self.next_effect,
            span,
        );
    }

    /// Extract the built EGraph + control_headers, leaving the rest of the
    /// Converter state behind. Used by the top-level `convert_program`
    /// phase to feed a ready-to-chain `EgirFunc` / `EgirEntry`.
    fn into_graph_parts(self) -> (EGraph, HashMap<BlockId, ControlHeader>) {
        (self.graph, self.control_headers)
    }

    /// Run the Converter's built EGraph through a throwaway single-function
    /// pipeline (`expand_soacs → optimize_skeleton → elaborate`) and return
    /// the resulting `FuncBody`. Used by:
    ///   * the pure-constant detection phase of `convert_program` (with
    ///     empty params);
    ///   * the inline tests in `mod tests` below.
    ///
    /// Production (non-test) function and entry-point conversion goes
    /// through `run`, which returns an `EgirInner` the caller wraps via
    /// `EgirRaw` to compose the pipeline explicitly.
    fn elaborate_to_funcbody(
        self,
        params: &[(Type<TypeName>, String)],
        return_ty: Type<TypeName>,
    ) -> Option<FuncBody> {
        let (mut graph, mut control_headers) = self.into_graph_parts();
        super::soac_expand::run_one_body(&mut graph, &mut control_headers, true);
        let aliases = super::skel_opt::run_one_body(&mut graph);
        let skel_domtree = super::domtree::DomTree::build(&super::domtree::SkeletonCfgView {
            skeleton: &graph.skeleton,
        });
        let identity_map: HashMap<BlockId, BlockId> =
            graph.skeleton.blocks.keys().map(|b| (b, b)).collect();
        Some(super::elaborate::run(
            &graph,
            &skel_domtree,
            params,
            return_ty,
            &control_headers,
            &identity_map,
            &aliases,
        ))
    }

    fn probe_constant_body(self, return_ty: Type<TypeName>) -> Option<FuncBody> {
        self.elaborate_to_funcbody(&[], return_ty)
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
            TermKind::Extern(name) => Ok(self.intern_pure(PureOp::Extern(name.clone()), smallvec![], ty)),

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
                let arr_ty = self.graph.types[&base].clone();
                let is_view =
                    arr_ty.array_variant().map(crate::types::is_array_variant_view).unwrap_or(false);
                if is_view {
                    let place_nid = self.intern_pure(PureOp::ViewIndex, smallvec![base, idx], ty.clone());
                    let result_nid = self.graph.alloc_side_effect_result(ty.clone());
                    let effect_in = EffectToken(0);
                    let effect_out = self.alloc_effect();
                    self.graph.skeleton.blocks[self.current_block].side_effects.push(SideEffect {
                        kind: SideEffectKind::Inst(InstKind::Load {
                            place: Default::default(),
                        }),
                        operand_nodes: smallvec![place_nid],
                        result: Some(result_nid),
                        effects: Some((effect_in, effect_out)),
                        span: self.current_span,
                    });
                    Ok(result_nid)
                } else {
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
            TermKind::BinOp(_) | TermKind::UnOp(_) => {
                panic!("ICE: bare operator in to_egir (should be inside App)")
            }
        }
    }

    // ========================================================================
    // Variable resolution
    // ========================================================================

    fn convert_var(&mut self, sym: SymbolId, ty: Type<TypeName>) -> Result<NodeId, ConvertError> {
        // Local binding
        if let Some(&nid) = self.locals.get(&sym) {
            return Ok(nid);
        }

        let name = self.symbols.get(sym).expect("BUG: symbol not in table").clone();

        // Cached constant
        if let Some(&nid) = self.inlined_constants.get(&name) {
            return Ok(nid);
        }

        // Hoisted pure constant → Global reference
        if self.pure_constants.contains(&name) {
            return Ok(self.intern_pure(PureOp::Global(name), smallvec![], ty));
        }

        // Arity-0 constant def → inline its body
        let const_def = self
            .top_level
            .get(&sym)
            .filter(|d| d.arity == 0)
            .or_else(|| self.constants_by_name.get(&name).and_then(|def_sym| self.top_level.get(def_sym)))
            .copied();

        if let Some(def) = const_def {
            let body = def.body.clone();
            let nid = self.convert_term(&body)?;
            self.inlined_constants.insert(name, nid);
            return Ok(nid);
        }

        // Function reference → Global
        Ok(self.intern_pure(PureOp::Global(name), smallvec![], ty))
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
            TermKind::Var(VarRef::Symbol(sym)) => {
                let name = self.symbols.get(*sym).expect("BUG").clone();
                self.convert_named_app(&name, *sym, args, ty)
            }
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
        name: &str,
        sym: SymbolId,
        args: &[Term],
        ty: Type<TypeName>,
    ) -> Result<NodeId, ConvertError> {
        match name {
            _ => {
                // Function call
                if let Some(def) = self.top_level.get(&sym) {
                    if def.arity == args.len() {
                        let arg_nids: SmallVec<[NodeId; 4]> =
                            args.iter().map(|a| self.convert_term(a)).collect::<Result<_, _>>()?;
                        let arg_vrefs: Vec<ValueRef> = (0..arg_nids.len())
                            .map(|_| ValueRef::Ssa(crate::ssa::types::ValueId::default()))
                            .collect();
                        let result_nid = self.graph.alloc_side_effect_result(ty.clone());
                        let effect_in = EffectToken(0);
                        let effect_out = self.alloc_effect();
                        self.graph.skeleton.blocks[self.current_block].side_effects.push(SideEffect {
                            kind: SideEffectKind::Inst(InstKind::Op {
                                tag: crate::op::OpTag::Call(name.to_string()),
                                operands: arg_vrefs,
                            }),
                            operand_nodes: arg_nids,
                            result: Some(result_nid),
                            effects: Some((effect_in, effect_out)),
                            span: self.current_span,
                        });
                        return Ok(result_nid);
                    }
                }
                // Arity-0 constant applied to args? Inline the body.
                let var_nid = self.convert_var(sym, ty.clone())?;
                // If we got a Global, emit a call
                if matches!(self.graph.nodes[var_nid], ENode::Pure { ref op, .. } if matches!(op, PureOp::Global(_)))
                {
                    let arg_nids: SmallVec<[NodeId; 4]> =
                        args.iter().map(|a| self.convert_term(a)).collect::<Result<_, _>>()?;
                    let arg_vrefs: Vec<ValueRef> = (0..arg_nids.len())
                        .map(|_| ValueRef::Ssa(crate::ssa::types::ValueId::default()))
                        .collect();
                    let result_nid = self.graph.alloc_side_effect_result(ty);
                    let effect_in = EffectToken(0);
                    let effect_out = self.alloc_effect();
                    self.graph.skeleton.blocks[self.current_block].side_effects.push(SideEffect {
                        kind: SideEffectKind::Inst(InstKind::Op {
                            tag: crate::op::OpTag::Call(name.to_string()),
                            operands: arg_vrefs,
                        }),
                        operand_nodes: arg_nids,
                        result: Some(result_nid),
                        effects: Some((effect_in, effect_out)),
                        span: self.current_span,
                    });
                    Ok(result_nid)
                } else {
                    Err(ConvertError::Unsupported(format!(
                        "application of non-function: {}",
                        name
                    )))
                }
            }
        }
    }

    // ========================================================================
    // Side-effectful intrinsic helpers — called from the Builtin-arm
    // dispatch on `BuiltinLowering::Intrinsic(StorageIndex|StorageStore)`.
    // ========================================================================

    fn lower_storage_index(&mut self, args: &[Term], ty: Type<TypeName>) -> Result<NodeId, ConvertError> {
        let set = match &args[0].kind {
            TermKind::IntLit(s) => s.parse::<u32>().map_err(|_| {
                ConvertError::GraphError("_w_intrinsic_storage_index: set not a u32".into())
            })?,
            _ => {
                return Err(ConvertError::GraphError(
                    "_w_intrinsic_storage_index: set must be int literal".into(),
                ));
            }
        };
        let binding = match &args[1].kind {
            TermKind::IntLit(s) => s.parse::<u32>().map_err(|_| {
                ConvertError::GraphError("_w_intrinsic_storage_index: binding not a u32".into())
            })?,
            _ => {
                return Err(ConvertError::GraphError(
                    "_w_intrinsic_storage_index: binding must be int literal".into(),
                ));
            }
        };
        let index_nid = self.convert_term(&args[2])?;
        let view_nid = self.emit_storage_view(BindingRef::new(set, binding), ty.clone());
        let place_nid = self.intern_pure(PureOp::ViewIndex, smallvec![view_nid, index_nid], ty.clone());
        let result_nid = self.graph.alloc_side_effect_result(ty.clone());
        let effect_in = EffectToken(0);
        let effect_out = self.alloc_effect();
        self.graph.skeleton.blocks[self.current_block].side_effects.push(SideEffect {
            kind: SideEffectKind::Inst(InstKind::Load {
                place: Default::default(),
            }),
            operand_nodes: smallvec![place_nid],
            result: Some(result_nid),
            effects: Some((effect_in, effect_out)),
            span: self.current_span,
        });
        Ok(result_nid)
    }

    fn lower_storage_store(&mut self, args: &[Term]) -> Result<NodeId, ConvertError> {
        let set = match &args[0].kind {
            TermKind::IntLit(s) => s.parse::<u32>().map_err(|_| {
                ConvertError::GraphError("_w_intrinsic_storage_store: set not a u32".into())
            })?,
            _ => {
                return Err(ConvertError::GraphError(
                    "_w_intrinsic_storage_store: set must be int literal".into(),
                ));
            }
        };
        let binding = match &args[1].kind {
            TermKind::IntLit(s) => s.parse::<u32>().map_err(|_| {
                ConvertError::GraphError("_w_intrinsic_storage_store: binding not a u32".into())
            })?,
            _ => {
                return Err(ConvertError::GraphError(
                    "_w_intrinsic_storage_store: binding must be int literal".into(),
                ));
            }
        };
        let index_nid = self.convert_term(&args[2])?;
        let value_nid = self.convert_term(&args[3])?;
        let value_ty = args[3].ty.clone();
        let view_nid = self.emit_storage_view(BindingRef::new(set, binding), value_ty.clone());
        self.emit_storage_store(view_nid, index_nid, value_nid, value_ty);
        let unit_ty = Type::Constructed(TypeName::Unit, vec![]);
        Ok(self.intern_pure(PureOp::Unit, smallvec![], unit_ty))
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

        let result_nid = self.graph.add_block_param(merge_block, 0, ty.clone());
        self.graph.skeleton.blocks[merge_block].params.push(result_nid);

        // Selection header for SPIR-V structured control flow.
        self.control_headers.insert(
            self.current_block,
            ControlHeader::Selection { merge: merge_block },
        );

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
            args: vec![then_result],
        };

        // Else branch.
        self.current_block = else_block;
        let else_result = self.convert_term(else_branch)?;
        self.graph.skeleton.blocks[self.current_block].term = SkeletonTerminator::Branch {
            target: merge_block,
            args: vec![else_result],
        };

        // Continue from merge.
        self.current_block = merge_block;
        Ok(result_nid)
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
        let acc_nid = self.graph.add_block_param(header, 0, acc_ty.clone());
        self.graph.skeleton.blocks[header].params.push(acc_nid);
        let result_nid = self.graph.add_block_param(exit, 0, acc_ty.clone());
        self.graph.skeleton.blocks[exit].params.push(result_nid);

        // Loop header for SPIR-V
        self.control_headers.insert(
            header,
            ControlHeader::Loop {
                merge: exit,
                continue_block: body_block,
            },
        );

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
        let acc_nid = self.graph.add_block_param(header, 0, acc_ty.clone());
        let idx_nid = self.graph.add_block_param(header, 1, i32_ty.clone());
        self.graph.skeleton.blocks[header].params.push(acc_nid);
        self.graph.skeleton.blocks[header].params.push(idx_nid);
        let result_nid = self.graph.add_block_param(exit, 0, acc_ty.clone());
        self.graph.skeleton.blocks[exit].params.push(result_nid);

        self.control_headers.insert(
            header,
            ControlHeader::Loop {
                merge: exit,
                continue_block: body_block,
            },
        );

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
        let cond_nid = self.intern_pure(PureOp::BinOp("<".into()), smallvec![idx_nid, bound_nid], bool_ty);
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
        let next_i = self.intern_pure(PureOp::BinOp("+".into()), smallvec![idx_nid, one], i32_ty);
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

        let acc_nid = self.graph.add_block_param(header, 0, acc_ty.clone());
        let idx_nid = self.graph.add_block_param(header, 1, i32_ty.clone());
        self.graph.skeleton.blocks[header].params.push(acc_nid);
        self.graph.skeleton.blocks[header].params.push(idx_nid);
        let result_nid = self.graph.add_block_param(exit, 0, acc_ty.clone());
        self.graph.skeleton.blocks[exit].params.push(result_nid);

        self.control_headers.insert(
            header,
            ControlHeader::Loop {
                merge: exit,
                continue_block: body_block,
            },
        );

        // Init
        let init_nid = self.convert_term(init)?;
        let iter_nid = self.convert_term(iter)?;

        // Length intrinsic. PureOp::UnaryOp keys by op-name string;
        // the catalog-internal `_w_intrinsic_length` is the agreed
        // string the lowering layer dispatches on.
        let length_name = crate::builtins::by_id(catalog().known().length).dispatch_name();
        let len_nid = self.intern_pure(
            PureOp::UnaryOp(length_name.into()),
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
        let cond_nid = self.intern_pure(PureOp::BinOp("<".into()), smallvec![idx_nid, len_nid], bool_ty);
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
        let next_i = self.intern_pure(PureOp::BinOp("+".into()), smallvec![idx_nid, one], i32_ty);
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
            } => self.convert_soac_map(lam, inputs, *destination, ty),
            SoacOp::Reduce { op, ne, input, .. } => self.convert_soac_reduce(op, ne, input, ty),
            SoacOp::Redomap {
                op,
                reduce_op,
                ne,
                inputs,
                ..
            } => self.convert_soac_redomap(op, reduce_op, ne, inputs, ty),
            SoacOp::Scan {
                op,
                reduce_op,
                ne,
                input,
                destination,
            } => self.convert_soac_scan(op, reduce_op, ne, input, *destination, ty),
            SoacOp::Filter {
                pred,
                input,
                destination,
            } => self.convert_soac_filter(pred, input, *destination, ty),
            // TODO(scatter): no producer in to_tlc yet (no surface name dispatched here).
            // Variant exists to anchor the place-passing SOAC shape; remove if a wider
            // audit confirms no future use.
            SoacOp::Scatter { .. } => Err(ConvertError::Unsupported("SOAC scatter".into())),
            // TODO(reduce_by_index): parallel path needs atomic-op emission
            // (atomicAdd/atomicMin/etc.) in spirv/wgsl backends — not yet wired.
            // Sequential lowering is straightforward (read-combine-write loop) but
            // also not yet wired. Produced by `to_tlc::transform_soac_reduce_by_index`.
            SoacOp::ReduceByIndex { .. } => Err(ConvertError::Unsupported("SOAC reduce_by_index".into())),
        }
    }

    fn lambda_fn_name(&self, lam: &Lambda) -> Result<String, ConvertError> {
        match &lam.body.kind {
            TermKind::Var(VarRef::Symbol(sym)) => {
                Ok(self.symbols.get(*sym).expect("BUG: symbol not in table").clone())
            }
            _ => Err(ConvertError::GraphError(
                "SOAC lambda body should be a function reference post-defunc".into(),
            )),
        }
    }

    /// Emit a SOAC placeholder as a side effect in the skeleton. Returns the
    /// result NodeId that `soac_expand` will rebind during expansion.
    fn emit_soac(
        &mut self,
        soac: PendingSoac,
        operands: SmallVec<[NodeId; 4]>,
        ty: Type<TypeName>,
    ) -> NodeId {
        let span = self.current_span;
        super::graph_ops::emit_pending_soac(
            &mut self.graph,
            self.current_block,
            soac,
            operands,
            ty,
            &mut self.next_effect,
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
        let f_name = self.lambda_fn_name(&sb.lam)?;
        let capture_nids: Vec<NodeId> =
            sb.captures.iter().map(|(_, _, t)| self.convert_term(t)).collect::<Result<_, _>>()?;
        let input_arr_types: Vec<Type<TypeName>> =
            inputs.iter().map(|ae| self.array_expr_type(ae)).collect();
        let input_nids: Vec<NodeId> =
            inputs.iter().map(|ae| self.convert_array_expr_value(ae)).collect::<Result<_, _>>()?;
        let input_elem_types: Vec<Type<TypeName>> =
            inputs.iter().map(|ae| self.array_expr_elem_type(ae)).collect();
        let output_elem_ty = if result_ty.is_array() {
            result_ty.elem_type().expect("Array has elem").clone()
        } else if super::soac_expand::as_soa_tuple(&result_ty).is_some() {
            // After `tlc::soa`, the map's output `[N](A, B)` becomes a
            // SoA tuple `([N]A, [N]B)`. The per-iteration element type
            // is the corresponding tuple-of-elements `(A, B)`.
            super::soac_expand::soa_element_type(&result_ty)
        } else if !input_elem_types.is_empty() {
            input_elem_types[0].clone()
        } else {
            return Err(ConvertError::GraphError(
                "map: cannot determine output elem type".into(),
            ));
        };

        let mut operands: SmallVec<[NodeId; 4]> = SmallVec::new();
        operands.extend_from_slice(&input_nids);
        operands.extend_from_slice(&capture_nids);

        Ok(self.emit_soac(
            PendingSoac::Map {
                func: f_name,
                input_array_types: input_arr_types,
                input_elem_types,
                output_elem_type: output_elem_ty,
                destination,
            },
            operands,
            result_ty,
        ))
    }

    fn convert_soac_reduce(
        &mut self,
        op: &SoacBody,
        ne: &Term,
        input: &ArrayExpr,
        result_ty: Type<TypeName>,
    ) -> Result<NodeId, ConvertError> {
        let op_name = self.lambda_fn_name(&op.lam)?;
        let capture_nids: Vec<NodeId> =
            op.captures.iter().map(|(_, _, t)| self.convert_term(t)).collect::<Result<_, _>>()?;
        let elem_ty = self.array_expr_elem_type(input);
        let arr_ty = self.array_expr_type(input);
        let arr_nid = self.convert_array_expr_value(input)?;
        let init_nid = self.convert_term(ne)?;

        let mut operands: SmallVec<[NodeId; 4]> = smallvec![arr_nid, init_nid];
        operands.extend(capture_nids.iter().copied());

        Ok(self.emit_soac(
            PendingSoac::Reduce {
                func: op_name,
                input_array_type: arr_ty,
                input_elem_type: elem_ty,
            },
            operands,
            result_ty,
        ))
    }

    fn convert_soac_redomap(
        &mut self,
        op: &SoacBody,
        reduce_op: &SoacBody,
        ne: &Term,
        inputs: &[ArrayExpr],
        result_ty: Type<TypeName>,
    ) -> Result<NodeId, ConvertError> {
        let op_name = self.lambda_fn_name(&op.lam)?;
        let reduce_func_name = self.lambda_fn_name(&reduce_op.lam)?;
        let capture_nids: Vec<NodeId> =
            op.captures.iter().map(|(_, _, t)| self.convert_term(t)).collect::<Result<_, _>>()?;
        let reduce_capture_nids: Vec<NodeId> =
            reduce_op.captures.iter().map(|(_, _, t)| self.convert_term(t)).collect::<Result<_, _>>()?;
        let input_arr_types: Vec<Type<TypeName>> =
            inputs.iter().map(|ae| self.array_expr_type(ae)).collect();
        let input_nids: Vec<NodeId> =
            inputs.iter().map(|ae| self.convert_array_expr_value(ae)).collect::<Result<_, _>>()?;
        let input_elem_types: Vec<Type<TypeName>> =
            inputs.iter().map(|ae| self.array_expr_elem_type(ae)).collect();
        let init_nid = self.convert_term(ne)?;

        let mut operands: SmallVec<[NodeId; 4]> = SmallVec::new();
        operands.extend(input_nids.iter().copied());
        operands.push(init_nid);
        operands.extend(capture_nids.iter().copied());
        operands.extend(reduce_capture_nids.iter().copied());

        Ok(self.emit_soac(
            PendingSoac::Redomap {
                func: op_name,
                reduce_func: reduce_func_name,
                input_array_types: input_arr_types,
                input_elem_types,
            },
            operands,
            result_ty,
        ))
    }

    fn convert_soac_scan(
        &mut self,
        op: &SoacBody,
        reduce_op: &SoacBody,
        ne: &Term,
        input: &ArrayExpr,
        destination: SoacDestination,
        result_ty: Type<TypeName>,
    ) -> Result<NodeId, ConvertError> {
        let op_name = self.lambda_fn_name(&op.lam)?;
        let reduce_name = self.lambda_fn_name(&reduce_op.lam)?;
        let capture_nids: Vec<NodeId> =
            op.captures.iter().map(|(_, _, t)| self.convert_term(t)).collect::<Result<_, _>>()?;
        // Take the elem type from the result, not the input — for a
        // `*[N]T` input the input-side helper returns `Unique<[N]T>`
        // rather than `T` (see convert_soac_map's `output_elem_ty`
        // for the same workaround). Scan's accumulator type equals
        // the output element type.
        let elem_ty = if result_ty.is_array() {
            result_ty.elem_type().expect("Array has elem").clone()
        } else {
            self.array_expr_elem_type(input)
        };
        let arr_ty = self.array_expr_type(input);
        let arr_nid = self.convert_array_expr_value(input)?;
        let init_nid = self.convert_term(ne)?;

        let mut operands: SmallVec<[NodeId; 4]> = smallvec![arr_nid, init_nid];
        operands.extend(capture_nids.iter().copied());

        Ok(self.emit_soac(
            PendingSoac::Scan {
                func: op_name,
                reduce_func: reduce_name,
                input_array_type: arr_ty,
                input_elem_type: elem_ty,
                destination,
            },
            operands,
            result_ty,
        ))
    }

    fn convert_soac_filter(
        &mut self,
        pred: &SoacBody,
        input: &ArrayExpr,
        destination: SoacDestination,
        _result_ty: Type<TypeName>,
    ) -> Result<NodeId, ConvertError> {
        let pred_name = self.lambda_fn_name(&pred.lam)?;
        let capture_nids: Vec<NodeId> =
            pred.captures.iter().map(|(_, _, t)| self.convert_term(t)).collect::<Result<_, _>>()?;
        let elem_ty = self.array_expr_elem_type(input);
        let arr_ty = self.array_expr_type(input);
        let arr_nid = self.convert_array_expr_value(input)?;

        // Filter requires a statically-sized input — N is the upper
        // bound on the output count. The TLC-level result type is an
        // existential `?k. [k]T`; after `open_existential` it becomes
        // `Array[T, Skolem(k), Composite]`. We rewrite the EGIR-level
        // result type to `Array[T, Size(N), Bounded]` so downstream
        // (elaborate, backends) see a concrete Bounded struct with
        // both a buffer and a runtime length.
        let size = arr_ty
            .array_size()
            .ok_or_else(|| ConvertError::GraphError("filter: input has no array_size".into()))?
            .clone();
        if !matches!(&size, Type::Constructed(TypeName::Size(_), _)) {
            return Err(ConvertError::GraphError(format!(
                "filter: input must be statically sized, got size {:?}",
                size
            )));
        }
        let bounded_result_ty = Type::Constructed(
            TypeName::Array,
            vec![
                elem_ty.clone(),
                size.clone(),
                Type::Constructed(TypeName::ArrayVariantBounded, vec![]),
            ],
        );

        let mut operands: SmallVec<[NodeId; 4]> = smallvec![arr_nid];
        operands.extend(capture_nids.iter().copied());

        Ok(self.emit_soac(
            PendingSoac::Filter {
                pred_func: pred_name,
                input_array_type: arr_ty,
                input_elem_type: elem_ty,
                output_capacity_size: size,
                destination,
            },
            operands,
            bounded_result_ty,
        ))
    }

    // ========================================================================
    // ArrayExpr
    // ========================================================================

    fn convert_array_expr(&mut self, ae: &ArrayExpr, ty: Type<TypeName>) -> Result<NodeId, ConvertError> {
        match ae {
            ArrayExpr::Ref(term) => self.convert_term(term),
            ArrayExpr::Zip(_) => panic!("ArrayExpr::Zip should have been eliminated by soa::normalize"),
            ArrayExpr::Soac(op) => self.convert_soac(op, ty),
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
            ArrayExpr::StorageView(crate::tlc::StorageView {
                binding,
                offset,
                len,
                elem_ty,
            }) => {
                let offset_nid = self.convert_term(offset)?;
                let len_nid = self.convert_term(len)?;
                let array_ty = Type::Constructed(
                    TypeName::Array,
                    vec![
                        elem_ty.clone(),
                        Type::Constructed(TypeName::SizePlaceholder, vec![]),
                        Type::Constructed(TypeName::ArrayVariantView, vec![]),
                    ],
                );
                let result_nid = self.graph.alloc_side_effect_result(array_ty);
                let effect_in = EffectToken(0);
                let effect_out = self.alloc_effect();
                self.graph.skeleton.blocks[self.current_block].side_effects.push(SideEffect {
                    kind: SideEffectKind::Inst(InstKind::Op {
                        tag: crate::op::OpTag::StorageView(crate::op::PureViewSource::Storage(*binding)),
                        operands: vec![
                            ValueRef::Ssa(Default::default()),
                            ValueRef::Ssa(Default::default()),
                        ],
                    }),
                    operand_nodes: smallvec![offset_nid, len_nid],
                    result: Some(result_nid),
                    effects: Some((effect_in, effect_out)),
                    span: self.current_span,
                });
                Ok(result_nid)
            }
        }
    }

    fn convert_array_expr_value(&mut self, ae: &ArrayExpr) -> Result<NodeId, ConvertError> {
        match ae {
            ArrayExpr::Ref(term) => self.convert_term(term),
            _ => {
                let ty = self.array_expr_type(ae);
                self.convert_array_expr(ae, ty)
            }
        }
    }

    fn array_expr_type(&self, ae: &ArrayExpr) -> Type<TypeName> {
        match ae {
            // Strip `*` at the EGIR boundary — uniqueness is a TLC
            // concern; downstream array-shape checks (composite vs
            // view vs virtual) operate on the bare array type.
            ArrayExpr::Ref(t) => crate::types::strip_unique(&t.ty),
            ArrayExpr::Zip(_) => unreachable!("Zip eliminated"),
            ArrayExpr::Soac(_) => Type::Constructed(TypeName::Unit, vec![]),
            ArrayExpr::Literal(_) => Type::Constructed(TypeName::Unit, vec![]),
            ArrayExpr::Range { start, .. } => Type::Constructed(
                TypeName::Array,
                vec![
                    start.ty.clone(),
                    Type::Constructed(TypeName::SizePlaceholder, vec![]),
                    Type::Constructed(TypeName::ArrayVariantVirtual, vec![]),
                ],
            ),
            ArrayExpr::StorageView(crate::tlc::StorageView { elem_ty, .. }) => Type::Constructed(
                TypeName::Array,
                vec![
                    elem_ty.clone(),
                    Type::Constructed(TypeName::SizePlaceholder, vec![]),
                    Type::Constructed(TypeName::ArrayVariantView, vec![]),
                ],
            ),
        }
    }

    fn array_expr_elem_type(&self, ae: &ArrayExpr) -> Type<TypeName> {
        match ae {
            ArrayExpr::Ref(t) => match &t.ty {
                Type::Constructed(TypeName::Array, args) if !args.is_empty() => args[0].clone(),
                // After `tlc::soa`, `[N](A, B)` becomes `([N]A, [N]B)` — an
                // SoA tuple. The per-iteration element type is the
                // corresponding `(A, B)` element tuple, not the tuple
                // itself.
                ty if super::soac_expand::as_soa_tuple(ty).is_some() => {
                    super::soac_expand::soa_element_type(ty)
                }
                _ => t.ty.clone(),
            },
            ArrayExpr::Zip(_) => unreachable!("Zip eliminated"),
            ArrayExpr::Soac(_) => Type::Constructed(TypeName::Unit, vec![]),
            ArrayExpr::Literal(terms) => {
                terms.first().map(|t| t.ty.clone()).unwrap_or(Type::Constructed(TypeName::Unit, vec![]))
            }
            ArrayExpr::Range { start, .. } => start.ty.clone(),
            ArrayExpr::StorageView(crate::tlc::StorageView { elem_ty, .. }) => elem_ty.clone(),
        }
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Check whether a FuncBody contains only purely constant instructions.
fn is_purely_constant_body(body: &FuncBody) -> bool {
    use crate::op::OpTag;
    body.inner.insts.values().all(|inst| match &inst.data {
        InstKind::Op { tag, .. } => matches!(
            tag,
            OpTag::Int(_)
                | OpTag::Uint(_)
                | OpTag::Float(_)
                | OpTag::Bool(_)
                | OpTag::Unit
                | OpTag::Tuple(_)
                | OpTag::Vector(_)
                | OpTag::Matrix { .. }
                | OpTag::ArrayLit(_)
                | OpTag::Global(_)
        ),
        _ => false,
    })
}

/// Extract parameter types and return type from an arrow type.
fn extract_function_signature(ty: &Type<TypeName>) -> (Vec<Type<TypeName>>, Type<TypeName>) {
    let mut params = Vec::new();
    let mut current = ty.clone();
    while let Type::Constructed(TypeName::Arrow, ref args) = current {
        if args.len() == 2 {
            params.push(args[0].clone());
            current = args[1].clone();
        } else {
            break;
        }
    }
    (params, current)
}

/// Walk through nested Lambdas to extract parameters and the inner body.
fn extract_lambda_params(term: &Term) -> (&Term, Vec<(SymbolId, Type<TypeName>)>) {
    let mut params = Vec::new();
    let mut current = term;
    while let TermKind::Lambda(lam) = &current.kind {
        params.extend(lam.params.iter().cloned());
        current = &lam.body;
    }
    (current, params)
}

/// Extract a `#[size_hint(N)]` attribute from a pattern.
pub fn extract_size_hint(pattern: &crate::ast::Pattern) -> Option<std::num::NonZeroU32> {
    use crate::ast;
    match &pattern.kind {
        ast::PatternKind::Attributed(attrs, inner) => {
            for attr in attrs {
                if let interface::Attribute::SizeHint(n) = attr {
                    return Some(*n);
                }
            }
            extract_size_hint(inner)
        }
        ast::PatternKind::Typed(inner, _) => extract_size_hint(inner),
        _ => None,
    }
}

/// Convert an AST attribute to an IO decoration.
fn convert_to_io_decoration(attr: &interface::Attribute) -> Option<IoDecoration> {
    use IoDecoration;
    match attr {
        interface::Attribute::BuiltIn(b) => Some(IoDecoration::BuiltIn(*b)),
        interface::Attribute::Location(l) => Some(IoDecoration::Location(*l)),
        _ => None,
    }
}

/// Build entry outputs from an AST `EntryDecl`.
/// For compute shaders, non-unit outputs get sequential storage bindings starting at `binding_start`.
fn build_entry_outputs(
    entry: &interface::EntryDecl,
    ret_type: &Type<TypeName>,
    is_compute: bool,
    binding_start: u32,
    forced_output_binding: Option<BindingRef>,
    dispatch_sized: bool,
) -> Result<Vec<EntryOutput>, ConvertError> {
    use EntryOutput;
    let mut binding_num = binding_start;
    let mut forced_remaining = forced_output_binding;
    // A parallel map/scan writes one output element per dispatched thread, so
    // a storage output's length tracks the dispatch. `None` otherwise (the
    // host falls back to its default sizing). `ty` is the runtime-sized
    // array output type; `elem_bytes` is the byte size of *one element*
    // (i.e. of `ty.elem_type()`), not the whole array.
    let length_for =
        |binding: Option<BindingRef>, ty: &Type<TypeName>| -> Result<Option<BufferLen>, ConvertError> {
            if !dispatch_sized || binding.is_none() {
                return Ok(None);
            }
            let elem_ty = ty.elem_type().ok_or_else(|| {
                ConvertError::Internal(format!("dispatch-sized output type is not an array: {ty:?}"))
            })?;
            let elem_bytes = crate::ssa::layout::type_byte_size(elem_ty).ok_or_else(|| {
                ConvertError::Internal(format!(
                    "dispatch-sized output element has no static byte layout: {elem_ty:?}"
                ))
            })?;
            Ok(Some(BufferLen::SameAsDispatch { elem_bytes }))
        };
    let mut storage_binding_for = |ty: &Type<TypeName>, is_compute: bool| -> Option<BindingRef> {
        if is_compute && !matches!(ty, Type::Constructed(TypeName::Unit, _)) {
            // Honor the planned binding for the first storage output if
            // present; subsequent outputs (tuple-return entries) keep
            // auto-allocating.
            if let Some(b) = forced_remaining.take() {
                return Some(b);
            }
            let b = BindingRef::new(AUTO_STORAGE_SET, binding_num);
            binding_num += 1;
            Some(b)
        } else {
            None
        }
    };

    if entry.outputs.iter().all(|o| o.attribute.is_none()) && entry.outputs.len() == 1 {
        if !matches!(ret_type, Type::Constructed(TypeName::Unit, _)) {
            let storage_binding = storage_binding_for(ret_type, is_compute);
            Ok(vec![EntryOutput {
                ty: ret_type.clone(),
                decoration: None,
                storage_binding,
                length: length_for(storage_binding, ret_type)?,
            }])
        } else {
            Ok(vec![])
        }
    } else if let Type::Constructed(TypeName::Tuple(_), component_types) = ret_type {
        entry
            .outputs
            .iter()
            .zip(component_types.iter())
            .map(|(output, ty)| {
                let storage_binding = storage_binding_for(ty, is_compute);
                Ok(EntryOutput {
                    ty: ty.clone(),
                    decoration: output.attribute.as_ref().and_then(convert_to_io_decoration),
                    storage_binding,
                    length: length_for(storage_binding, ty)?,
                })
            })
            .collect()
    } else {
        let storage_binding = storage_binding_for(ret_type, is_compute);
        Ok(vec![EntryOutput {
            ty: ret_type.clone(),
            decoration: entry
                .outputs
                .first()
                .and_then(|o| o.attribute.as_ref())
                .and_then(convert_to_io_decoration),
            storage_binding,
            length: length_for(storage_binding, ret_type)?,
        }])
    }
}

#[cfg(test)]
#[path = "from_tlc_tests.rs"]
mod from_tlc_tests;
