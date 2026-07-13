//! Direct TLC to EGraph conversion.
//!
//! Converts a TLC program directly to the acyclic e-graph representation,
//! bypassing the sequential SSA construction of `to_ssa`. Pure operations
//! are hash-consed (giving GVN for free), and the result is elaborated
//! back to `FuncBody` via demand-driven scheduling (giving DCE for free).

use crate::builtins::catalog;
use crate::ssa::types::{EntryInput, EntryOutput, IoDecoration};
use crate::tlc::{SoacBody, VarRef};
use crate::{LookupMap, LookupSet};

use super::types::EffectToken;
use crate::ast::{Span, TypeName};
use crate::binding_layout::{
    extract_io_decoration, extract_sampler_binding, extract_storage_access, extract_storage_binding,
    extract_storage_image_binding, extract_storage_image_resource, extract_texture_backing,
    extract_texture_binding, extract_texture_resource, extract_uniform_binding,
};
use crate::interface::{EntryParamBinding, EntryParamBindingKind};
use crate::ssa::framework::BlockId;
use crate::ssa::types::{ControlHeader, FuncBody, Function, InstKind, ValueRef};
use crate::tlc::{
    ArrayExpr, Def as TlcDef, DefMeta, Lambda, LoopKind, Program as TlcProgram, SoacOp, Term, TermKind,
};
use crate::types::{extract_function_signature, TypeExt};
use crate::{interface, BindingRef, SymbolId, SymbolTable};
use polytype::Type;
use smallvec::{smallvec, SmallVec};

use super::program::{EgirFunc, SemanticEntry, SemanticProgram};
use super::publish::PipelineDescriptorPublish;
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
    /// A source `#[dispatch(...)]` grid is incompatible with the entry — e.g.
    /// it launches fewer threads than the entry's data-parallel domain has
    /// elements, silently dropping the tail. A user error, not internal.
    InvalidDispatch(String),
    /// Two declared descriptor resources claim the same `(set, binding)` with
    /// incompatible descriptor classes or layout-relevant properties.
    DescriptorLayout(String),
}

impl std::fmt::Display for ConvertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConvertError::GraphError(msg) => write!(f, "EGraph conversion error: {}", msg),
            ConvertError::Unsupported(msg) => write!(f, "Unsupported: {}", msg),
            ConvertError::Internal(msg) => write!(f, "internal compiler error: {}", msg),
            ConvertError::InvalidDispatch(msg) => write!(f, "{}", msg),
            ConvertError::DescriptorLayout(msg) => write!(f, "{}", msg),
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
    top_level: &'a LookupMap<SymbolId, &'a TlcDef>,
    constants_by_name: &'a LookupMap<String, SymbolId>,
    symbols: &'a SymbolTable,
    region_interner: &'a std::cell::RefCell<crate::egir::program::RegionInterner>,
}

impl<'a> GlobalContext<'a> {
    fn new_converter<'b>(
        &self,
        pure_constants: &LookupSet<String>,
        binding_ids: &'b mut crate::IdSource<u32>,
    ) -> Converter<'a, 'b> {
        Converter::new(
            self.top_level,
            self.constants_by_name,
            self.symbols,
            pure_constants.clone(),
            binding_ids,
            self.region_interner,
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
    pipeline: PipelineDescriptor,
    input_slice_bounds: &crate::tlc::input_slice_bounds::ProgramBounds,
    binding_ids: &mut crate::IdSource<u32>,
) -> Result<SemanticProgram, ConvertError> {
    let top_level: LookupMap<SymbolId, &TlcDef> = program.defs.iter().map(|d| (d.name, d)).collect();
    let symbols = &program.symbols;

    let constants_by_name = program.value_defs_by_name();

    // Region interner shared by every converter, then handed to `SemanticProgram` so
    // the function arena keys agree with the SegBody indices built here.
    let region_interner = std::cell::RefCell::new(crate::egir::program::RegionInterner::default());

    let ctx = GlobalContext {
        top_level: &top_level,
        constants_by_name: &constants_by_name,
        symbols,
        region_interner: &region_interner,
    };

    // Phase 1: detect pure constants. We elaborate each arity-0 def's body
    // through the full EGIR pipeline once (using a throwaway chain) to see if
    // it collapses to a purely-constant FuncBody. Constants are hoisted to
    // program scope and referenced by `PureOp::Global`.
    let mut pure_constant_names: LookupSet<String> = LookupSet::new();
    let mut constants = Vec::new();

    for def in &program.defs {
        if def.arity != 0 || !matches!(&def.meta, DefMeta::Function) {
            continue;
        }
        if matches!(&def.body.kind, TermKind::Extern(_)) {
            continue;
        }
        let def_name = symbols.get(def.name).expect("BUG: symbol not in table").clone();

        let mut converter = ctx.new_converter(&pure_constant_names, binding_ids);
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
    let mut entry_points: Vec<SemanticEntry> = Vec::new();
    let mut prepass_roles = LookupMap::new();

    for def in &program.defs {
        match &def.meta {
            DefMeta::Function | DefMeta::LiftedLambda => {
                let def_name = symbols.get(def.name).expect("BUG: symbol not in table");
                if pure_constant_names.contains(def_name) {
                    continue;
                }
                match convert_function(def, &ctx, &pure_constant_names, binding_ids)? {
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
                // Reduce / reducing Screma manage their result inside the
                // two-phase plan, so `forced_output` reports `None` for them.
                //
                // Fallback for `--single-stage` / `parallelize_soacs(disable=
                // true)`: no plan exists, but `lift_gathers` still records its
                // intended gather binding as an Output-role
                // `StorageBindingDecl` carrying a `length` (the gather-prepass
                // marker shape). Use that as the forced output so the prepass
                // map writes the gather buffer instead of having its result
                // auto-allocated onto a colliding binding.
                let forced_output_binding = entry
                    .storage_bindings
                    .iter()
                    .find(|binding| binding.role == interface::StorageRole::Output)
                    .map(|binding| binding.binding)
                    .or_else(|| gather_prepass_forced_output(entry));
                let entry_name = symbol_name(symbols, def.name).unwrap_or("");
                let entry_input_bounds = input_slice_bounds.get(entry_name);
                let ep = convert_entry_point(
                    def,
                    entry,
                    &ctx,
                    &pure_constant_names,
                    workgroup,
                    forced_output_binding,
                    entry_input_bounds,
                    binding_ids,
                )?;
                match entry.origin {
                    interface::EntryOrigin::Source => {}
                    interface::EntryOrigin::ScalarPrepass => {
                        prepass_roles.insert(ep.name.clone(), super::program::PrepassKind::Scalar);
                    }
                    interface::EntryOrigin::GatherPrepass => {
                        prepass_roles.insert(ep.name.clone(), super::program::PrepassKind::Gather);
                    }
                }
                entry_points.push(ep);
            }
        }
    }

    // Converters are done borrowing the interner; reclaim it for the arena.
    drop(ctx);
    let mut semantic = SemanticProgram::new(
        functions,
        externs,
        entry_points,
        constants,
        pipeline,
        region_interner.into_inner(),
    );
    semantic.prepass_roles = prepass_roles;
    Ok(semantic)
}

enum ConvertedFunc {
    Extern(Function),
    Regular(EgirFunc),
}

// ============================================================================
// Function conversion
// ============================================================================

fn convert_function<'a>(
    def: &TlcDef,
    ctx: &GlobalContext<'a>,
    pure_constants: &LookupSet<String>,
    binding_ids: &'a mut crate::IdSource<u32>,
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
    let (inner_body, params) = crate::tlc::extract_lambda_params_ref(&def.body);
    let ret_type = inner_body.ty.clone();
    let param_info: Vec<(Type<TypeName>, String)> = params
        .iter()
        .map(|(sym, ty)| Ok((ty.clone(), symbol_name(symbols, *sym)?.to_string())))
        .collect::<Result<_, ConvertError>>()?;

    let mut converter = ctx.new_converter(pure_constants, binding_ids);
    for (i, (sym, ty)) in params.iter().enumerate() {
        let nid = converter.graph.add_func_param(i, ty.clone());
        converter.locals.insert(*sym, nid);
    }
    let result = converter.convert_term(inner_body)?;
    converter.set_return(Some(result));

    // A runtime `filter` compacts into a reserved scratch storage buffer, which
    // only an `SemanticEntry` can host (it owns a descriptor set + a binding
    // namespace seeded above its params/outputs). An `EgirFunc` has neither, so
    // a scratch binding accumulated during a function body's conversion has
    // nowhere to be declared or sized — emitting it would silently mis-number
    // the binding and drop its host declaration. In practice a function whose
    // result is a runtime filter is inlined into its caller before this pass
    // (see `filter_runtime_in_subroutine_compiles`), so this never fires. If it
    // does, that inlining invariant broke: either restore it, or thread a
    // caller-reserved scratch binding into the function's signature (like a
    // param) so the buffer has a home.
    if !converter.extra_storage_bindings.is_empty() {
        return Err(ConvertError::GraphError(format!(
            "runtime `filter` in function `{def_name}` reserved a scratch storage buffer, but a \
             standalone function has no descriptor-set interface to host it — the call must be \
             inlined into a compute entry (it was not)"
        )));
    }

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

#[allow(clippy::too_many_arguments)]
fn convert_entry_point(
    def: &TlcDef,
    entry: &interface::EntryDecl,
    ctx: &GlobalContext,
    pure_constants: &LookupSet<String>,
    workgroup: (u32, u32, u32),
    forced_output_binding: Option<BindingRef>,
    input_slice_bounds_for_entry: Option<&LookupMap<SymbolId, BufferLen>>,
    binding_ids: &mut crate::IdSource<u32>,
) -> Result<SemanticEntry, ConvertError> {
    use crate::ssa::types::{EntryInput, ExecutionModel, IoDecoration, PushConstantSlot};

    let symbols = ctx.symbols;
    let def_name = symbol_name(symbols, def.name)?;
    let (inner_body, params) = crate::tlc::extract_lambda_params_ref(&def.body);
    let is_compute = matches!(entry.entry_type, interface::Attribute::Compute);

    // After `normalize_outputs`, `def.ty == def.body.ty` (the body
    // ends in `SideEffect` for normalized compute entries, in its
    // tail's type for graphics). Reading `inner_body.ty` is the right
    // source either way.
    let ret_type = inner_body.ty.clone();
    // For normalized compute entries, the body's `OutputSlotStore.value`
    // terms carry the post-`monomorphize`/`buffer_specialize` per-slot
    // types — pre-walk the body to harvest them so
    // `build_entry_outputs` can use them in preference to the parse-
    // time `entry.outputs[i].ty`.
    let slot_value_tys = collect_output_slot_value_tys(inner_body);
    let param_info: Vec<(Type<TypeName>, String)> = params
        .iter()
        .map(|(sym, ty)| Ok((ty.clone(), symbol_name(symbols, *sym)?.to_string())))
        .collect::<Result<_, ConvertError>>()?;

    let mut converter = ctx.new_converter(pure_constants, binding_ids);

    // Build entry inputs alongside the symbol → NodeId bindings. A compute
    // entry param that's a tuple-of-unsized-arrays gets one storage binding
    // per field (SoA lowered the source `[]T` of tuples into a tuple of
    // `[]T`s, but entry I/O can't carry a tuple handle — each runtime-sized
    // array needs its own buffer). The body still references the original
    // tuple symbol, so we reconstruct it as a `Tuple(views…)` node.
    let mut inputs: Vec<EntryInput> = Vec::with_capacity(params.len());
    let mut pc_offset: u32 = 0;

    // The auto-storage binding layout is dense — same length as `params`,
    // with `None` for non-storage params — so we walk them in lockstep.
    let param_bindings: &[Option<EntryParamBinding>] = &entry.param_bindings;

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
                    decoration: None,
                    size_hint: None,
                    storage_binding: Some(slot.binding),
                    storage_access: None,
                    uniform_binding: None,
                    push_constant: None,
                    texture_binding: None,
                    texture_backing: None,
                    texture_resource: None,
                    storage_image_resource: None,
                    sampler_binding: None,
                    storage_image_binding: None,
                    length: None,
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

        inputs.push(EntryInput {
            name: name.to_string(),
            ty: crate::types::canonical_storage_buffer_ty(ty),
            decoration,
            size_hint,
            storage_binding,
            storage_access,
            uniform_binding,
            push_constant,
            texture_binding,
            texture_backing,
            texture_resource,
            sampler_binding,
            storage_image_binding,
            storage_image_resource,
            length: None,
        });
    }

    // Patch each storage-bound input's `length` from the caller-
    // supplied side-table. `input_slice_bounds_for_entry` was
    // computed by the TLC-level `input_slice_bounds` analyzer (the
    // `TlcInputSliceBoundsInferred` typestate) and is keyed by the
    // entry param's TLC `SymbolId`.
    if let Some(map) = input_slice_bounds_for_entry {
        for (input, (sym, _)) in inputs.iter_mut().zip(params.iter()) {
            if input.storage_binding.is_some() {
                if let Some(len) = map.get(sym).cloned() {
                    input.length = Some(len);
                }
            }
        }
    }
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
        &ret_type,
        &slot_value_tys,
        &inputs,
        is_compute,
        converter.binding_ids,
        forced_output_binding,
    )?;
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
    let result_nid = converter.convert_term(inner_body)?;

    // Slot-collected entries (post-`normalize_outputs`) have their
    // writes recorded as per-slot `SlotSource`s rather than flowing
    // through a single Return value. `egir::realize_outputs` reads
    // those sources directly off `SemanticEntry.output_routes` and
    // retargets each one. The body terminates with `Return(None)` —
    // there's no value to merge through the CFG.
    let was_slot_collected = !converter.slot_sources_accum.is_empty();
    if was_slot_collected {
        converter.set_return(None);
    } else if is_unit_return {
        converter.set_return(None);
    } else {
        converter.set_return(Some(result_nid));
    }

    // `ret_type` is used by downstream code to type the entry's
    // function signature. Post-`normalize_outputs` the body's TLC
    // return type is `SideEffect`; reconstruct from `entry.outputs`
    // so the signature matches the declared output shape.
    let ret_type =
        if was_slot_collected && entry.outputs.iter().any(|output| is_storage_image_ty(&output.ty)) {
            Type::Constructed(TypeName::Unit, vec![])
        } else if was_slot_collected {
            if entry.outputs.len() == 1 {
                // Unwrap a `?k. [k]T` filter output to its runtime array, matching
                // the `EntryOutput.ty` `build_entry_outputs` produced.
                unwrap_existential_array(&entry.outputs[0].ty)
            } else {
                let component_tys: Vec<_> = entry.outputs.iter().map(|o| o.ty.clone()).collect();
                Type::Constructed(TypeName::Tuple(component_tys.len()), component_tys)
            }
        } else if is_storage_image_ty(&ret_type) {
            Type::Constructed(TypeName::Unit, vec![])
        } else {
            ret_type
        };

    let slot_sources = std::mem::take(&mut converter.slot_sources_accum);
    // Gather buffers declared in TLC (`lift_gathers`, on the `EntryDecl`) plus
    // scratch buffers allocated during body conversion (runtime `filter`).
    let mut storage_bindings = entry.storage_bindings.clone();
    storage_bindings.extend(std::mem::take(&mut converter.extra_storage_bindings));
    let (graph, control_headers) = converter.into_graph_parts();
    let output_count = outputs.len();
    let mut entry = SemanticEntry::new(
        def_name.to_string(),
        def.body.span,
        execution_model,
        inputs,
        outputs,
        storage_bindings,
        param_info,
        ret_type,
        graph,
        control_headers,
    );
    entry.output_routes = slot_sources
        .into_iter()
        .enumerate()
        .filter(|(slot, _)| *slot < output_count)
        .flat_map(|(slot, sources)| {
            sources.into_iter().map(move |source| crate::egir::program::OutputRoute {
                source,
                slot: crate::egir::program::OutputSlotId(slot),
                writers: Vec::new(),
            })
        })
        .collect();

    Ok(entry)
}

fn is_storage_image_ty(ty: &Type<TypeName>) -> bool {
    matches!(ty, Type::Constructed(TypeName::StorageTexture, _))
}

// ============================================================================
// Converter
// ============================================================================

struct Converter<'a, 'b> {
    /// The e-graph being built.
    graph: EGraph,
    /// Current skeleton block for side effects and terminators.
    current_block: BlockId,
    /// TLC variable → EGraph node mapping.
    locals: LookupMap<SymbolId, NodeId>,
    /// Top-level definitions.
    top_level: &'a LookupMap<SymbolId, &'a TlcDef>,
    /// Arity-0 defs indexed by name.
    constants_by_name: &'a LookupMap<String, SymbolId>,
    /// Symbol table.
    symbols: &'a SymbolTable,
    /// Cache for inlined constant bodies.
    inlined_constants: LookupMap<String, NodeId>,
    /// Names of hoisted pure constants.
    pure_constants: LookupSet<String>,
    /// Control headers for structured control flow (SPIR-V).
    control_headers: LookupMap<BlockId, ControlHeader>,
    /// Effect token counter.
    next_effect: u32,
    /// Span of the term currently being converted. Threaded through every
    /// pure-node intern and side-effect push so backend errors can blame
    /// the originating source. Pushed/popped in `convert_term`; `None`
    /// only outside any term conversion (e.g. entry-point glue).
    current_span: Option<Span>,
    /// Per-slot list of `SlotSource { block, value }` records collected
    /// from `OutputSlotStore` terms during entry-body conversion. Indexed
    /// by slot index. Populated by `convert_slot_store` (the
    /// `OutputSlotStore` arm of `convert_term_kind` delegates here);
    /// consumed by `convert_entry_point` to populate `SemanticEntry.output_routes`
    /// and decide whether the body terminates with `Return(None)` (when
    /// all outputs were written via slot stores) or a value (legacy
    /// non-normalized entries).
    ///
    /// A slot with one source has `vec![one]`; a slot written from both
    /// arms of an `If` (where both branches independently store to the
    /// same slot) has two. Empty for unit-returning entries that never
    /// went through `normalize_outputs`.
    slot_sources_accum: Vec<Vec<crate::egir::program::SlotSource>>,
    /// Module-wide id factory for auto-storage binding numbers.
    /// `alloc_scratch_binding` draws scratch slots (runtime `filter`
    /// output buffers) from it during body conversion; the enclosing
    /// `convert_entry_point` reborrows it through the converter to
    /// allocate output bindings.
    binding_ids: &'b mut crate::IdSource<u32>,
    /// Compiler-introduced storage-binding declarations accumulated during
    /// body conversion (runtime `filter` scratch buffers). Merged into the
    /// `SemanticEntry.storage_bindings` at construction, where `publish.rs`
    /// surfaces them to the host descriptor as `Intermediate`s.
    extra_storage_bindings: Vec<crate::interface::StorageBindingDecl>,
    /// Program-wide region interner, shared across every converter so SegBody
    /// region indices agree with the final function arena.
    region_interner: &'a std::cell::RefCell<crate::egir::program::RegionInterner>,
}

impl<'a, 'b> Converter<'a, 'b> {
    fn new(
        top_level: &'a LookupMap<SymbolId, &'a TlcDef>,
        constants_by_name: &'a LookupMap<String, SymbolId>,
        symbols: &'a SymbolTable,
        pure_constants: LookupSet<String>,
        binding_ids: &'b mut crate::IdSource<u32>,
        region_interner: &'a std::cell::RefCell<crate::egir::program::RegionInterner>,
    ) -> Self {
        let graph = EGraph::new();
        let entry = graph.skeleton.entry;
        Converter {
            graph,
            current_block: entry,
            locals: LookupMap::new(),
            top_level,
            constants_by_name,
            symbols,
            inlined_constants: LookupMap::new(),
            pure_constants,
            control_headers: LookupMap::new(),
            next_effect: 1,
            current_span: None,
            slot_sources_accum: Vec::new(),
            binding_ids,
            extra_storage_bindings: Vec::new(),
            region_interner,
        }
    }

    /// Resolve a callable name to its region index, interning it on first use.
    fn region(&self, name: impl AsRef<str>) -> RegionId {
        self.region_interner.borrow_mut().intern(name)
    }

    /// Reserve a fresh auto-storage binding for a compiler-introduced scratch
    /// buffer (runtime `filter` output). Draws from the module-wide id factory.
    fn alloc_scratch_binding(&mut self) -> BindingRef {
        BindingRef::new(AUTO_STORAGE_SET, self.binding_ids.next_id())
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
    /// phase to feed a ready-to-chain `EgirFunc` / `SemanticEntry`.
    fn into_graph_parts(self) -> (EGraph, LookupMap<BlockId, ControlHeader>) {
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
    /// through `run`, which returns an `SemanticProgram` the caller wraps via
    /// `EgirRaw` to compose the pipeline explicitly.
    fn elaborate_to_funcbody(
        self,
        params: &[(Type<TypeName>, String)],
        return_ty: Type<TypeName>,
    ) -> Option<FuncBody> {
        let region_interner = self.region_interner;
        let (mut graph, mut control_headers) = self.into_graph_parts();
        super::soac_expand::run_one_body(&mut graph, &mut control_headers, &region_interner.borrow());
        let aliases = super::skel_opt::run_one_body(&mut graph);
        let skel_domtree = super::elaborate::skeleton_domtree(&graph.skeleton);
        let identity_map: LookupMap<BlockId, BlockId> =
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
                        semantic_id: None,
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
                    // NOTE: a runtime-sized Composite array reaching here is an
                    // un-lifted gather (its producer wasn't materialized to a
                    // storage buffer — e.g. a producer behind a helper call:
                    // cross-function gather-lifting is unimplemented). It can't be
                    // lowered as a value and currently panics in the SPIR-V
                    // backend (`polytype_to_spirv`, the unsized-Composite arm).
                    // See the ignored `cross_function_gather_errors_cleanly` test.
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
            // `OutputSlotStore { slot_index, value, .. }`: delegate
            // to `convert_slot_store`, which recurses through `If` and
            // records one `SlotSource` per producing leaf. Returning a
            // unit constant keeps the surrounding `Let` chain's `body`
            // well-typed.
            TermKind::OutputSlotStore {
                slot_index, value, ..
            } => {
                self.convert_slot_store(*slot_index, value)?;
                let unit_ty = Type::Constructed(TypeName::Unit, vec![]);
                Ok(self.intern_pure(PureOp::Unit, smallvec![], unit_ty))
            }
        }
    }

    // ========================================================================
    // Output slot stores (DPS write at the producing site)
    // ========================================================================

    /// Convert one `OutputSlotStore(slot_index, value)`, recursing through
    /// `If` (and `Let` wrapping it) so each producing leaf records its
    /// own `SlotSource` at the block in which it fires.
    ///
    /// `If`-shaped values fork at the EGIR level: the current block ends
    /// with `CondBranch`, each arm recursively converts the same slot
    /// store against its branch, and both arms terminate with
    /// `Branch(merge)` carrying no result args. There's nothing to
    /// merge — each branch's slot write went to the same binding, the
    /// runtime CFG picks which one fires.
    ///
    /// `Let { x = rhs, body }` wrapping an output value (e.g. `let n =
    /// length(xs) in if c then map_using_n else map_using_n`) binds
    /// `rhs` at the current block, then recurses on `body` — the
    /// binding is visible in both `If` arms when the body forks.
    ///
    /// Non-control-flow values are converted normally; a single
    /// `SlotSource { block: self.current_block, value: <converted> }`
    /// is pushed to `slot_sources_accum[slot_index]`.
    fn convert_slot_store(&mut self, slot_index: usize, value: &Term) -> Result<(), ConvertError> {
        use crate::ssa::types::ControlHeader;
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
                self.convert_slot_store(slot_index, body)
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

                self.control_headers.insert(
                    self.current_block,
                    ControlHeader::Selection { merge: merge_block },
                );
                self.graph.skeleton.blocks[self.current_block].term = SkeletonTerminator::CondBranch {
                    cond: cond_nid,
                    then_target: then_block,
                    then_args: vec![],
                    else_target: else_block,
                    else_args: vec![],
                };

                self.current_block = then_block;
                self.convert_slot_store(slot_index, then_branch)?;
                self.graph.skeleton.blocks[self.current_block].term = SkeletonTerminator::Branch {
                    target: merge_block,
                    args: vec![],
                };

                self.current_block = else_block;
                self.convert_slot_store(slot_index, else_branch)?;
                self.graph.skeleton.blocks[self.current_block].term = SkeletonTerminator::Branch {
                    target: merge_block,
                    args: vec![],
                };

                self.current_block = merge_block;
                Ok(())
            }
            _ => {
                let value_nid = self.convert_term(value)?;
                while self.slot_sources_accum.len() <= slot_index {
                    self.slot_sources_accum.push(Vec::new());
                }
                self.slot_sources_accum[slot_index].push(crate::egir::program::SlotSource {
                    block: self.current_block,
                    value: value_nid,
                });
                Ok(())
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
                } else if *id == known.image_with && args.len() == 3 {
                    self.lower_image_with(args, ty)
                } else if *id == known.image_load && args.len() == 2 {
                    let binding = crate::types::storage_image_buffer(&args[0].ty).ok_or_else(|| {
                        ConvertError::GraphError(
                            "image_load operand has no concrete storage-image binding after monomorphization"
                                .into(),
                        )
                    })?;
                    let coord = self.convert_term(&args[1])?;
                    Ok(self.intern_pure(PureOp::StorageImageLoad(binding), smallvec![coord], ty))
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
                            semantic_id: None,
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
                        semantic_id: None,
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
            semantic_id: None,
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
        let arg_nids: SmallVec<[NodeId; 4]> =
            args[1..].iter().map(|a| self.convert_term(a)).collect::<Result<_, _>>()?;
        let arg_vrefs: Vec<ValueRef> =
            (0..arg_nids.len()).map(|_| ValueRef::Ssa(crate::ssa::types::ValueId::default())).collect();
        let unit_ty = Type::Constructed(TypeName::Unit, vec![]);
        let effect_result = self.graph.alloc_side_effect_result(unit_ty);
        let effect_in = EffectToken(0);
        let effect_out = self.alloc_effect();
        self.graph.skeleton.blocks[self.current_block].side_effects.push(SideEffect {
            semantic_id: None,
            kind: SideEffectKind::Inst(InstKind::Op {
                tag: crate::op::OpTag::StorageImageStore(binding),
                operands: arg_vrefs,
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
            let result_nid = self.graph.add_block_param(merge_block, 0, ty.clone());
            self.graph.skeleton.blocks[merge_block].params.push(result_nid);
            Some(result_nid)
        };

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

        self.control_headers.insert(
            header,
            ControlHeader::Loop {
                merge: exit,
                continue_block: body_block,
            },
        );

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
        let idx_nid = self.graph.add_block_param(header, 0, i32_ty.clone());
        self.graph.skeleton.blocks[header].params.push(idx_nid);

        self.control_headers.insert(
            header,
            ControlHeader::Loop {
                merge: exit,
                continue_block: body_block,
            },
        );

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
        let cond_nid = self.intern_pure(PureOp::BinOp("<".into()), smallvec![idx_nid, bound_nid], bool_ty);
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
        let next_i = self.intern_pure(PureOp::BinOp("+".into()), smallvec![idx_nid, one], i32_ty);
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
        let idx_nid = self.graph.add_block_param(header, 0, i32_ty.clone());
        self.graph.skeleton.blocks[header].params.push(idx_nid);

        self.control_headers.insert(
            header,
            ControlHeader::Loop {
                merge: exit,
                continue_block: body_block,
            },
        );

        let _init_nid = self.convert_term(init)?;
        let iter_nid = self.convert_term(iter)?;
        let length_name = crate::builtins::by_id(catalog().known().length).dispatch_name();
        let len_nid = self.intern_pure(
            PureOp::UnaryOp(length_name.into()),
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
        let cond_nid = self.intern_pure(PureOp::BinOp("<".into()), smallvec![idx_nid, len_nid], bool_ty);
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
        let next_i = self.intern_pure(PureOp::BinOp("+".into()), smallvec![idx_nid, one], i32_ty);
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
            SoacOp::Screma {
                lanes,
                accumulators,
                inputs,
            } => {
                // Discriminate single-output vs multi-output by SHAPE, not by the
                // result type: a single-output Screma's sole output may itself be
                // a tuple value, so `Tuple(_)` in `ty` does NOT imply a
                // multi-output Screma. A single-output Screma's `ty` is that
                // output directly; egir re-wraps it to `Tuple(1)+Project`.
                if crate::tlc::is_single_output_screma(lanes, accumulators) {
                    self.convert_soac_screma_single(lanes, accumulators, inputs, ty)
                } else {
                    self.convert_soac_screma(lanes, accumulators, inputs, ty)
                }
            }
            SoacOp::Scan {
                op,
                reduce_op,
                ne,
                input,
                destination,
            } => self.convert_soac_scan(op, reduce_op, ne, input, *destination, ty),
            SoacOp::Filter {
                map_lam,
                pred,
                input,
                destination,
            } => self.convert_soac_filter(map_lam.as_ref(), pred, input, *destination, ty),
            SoacOp::Scatter { dest, lam, inputs } => self.convert_soac_scatter(dest, lam, inputs, ty),
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
    fn emit_soac(&mut self, soac: EgirSoac, operands: SmallVec<[NodeId; 4]>, ty: Type<TypeName>) -> NodeId {
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
        let project_ty = if matches!(destination, SoacDestination::InputBuffer) {
            input_arr_types[0].clone()
        } else {
            input_arr_types
                .first()
                .and_then(|inp| shape_preserving_result_ty(inp, &output_elem_ty, &result_ty))
                .unwrap_or_else(|| result_ty.clone())
        };
        let tuple_ty = Type::Constructed(TypeName::Tuple(1), vec![project_ty.clone()]);
        // Singleton map: its one lane reads every input.
        let map_input_indices = vec![(0..input_arr_types.len()).collect::<Vec<usize>>()];
        let screma_nid = self.emit_soac(
            EgirSoac::Screma {
                map_bodies: vec![SegBody {
                    region: self.region(f_name),
                    captures: capture_nids,
                }],
                accumulators: vec![],
                input_array_types: input_arr_types,
                input_elem_types,
                map_output_elem_types: vec![output_elem_ty],
                map_input_indices,
                map_destinations: vec![destination],
                acc_destinations: vec![],
            },
            operands,
            tuple_ty,
        );
        Ok(self.intern_pure(PureOp::Project { index: 0 }, smallvec![screma_nid], project_ty))
    }

    /// `scatter(dest, indices, values)` → a side-effecting `EgirSoac::Hist`.
    /// `dest` is a `#[storage]` buffer param whose `StorageView` was already
    /// interned at param setup and stored in `self.locals`; the writes target
    /// that view. The result node is discarded (rebound during expansion).
    fn convert_soac_scatter(
        &mut self,
        dest: &crate::tlc::Place,
        lam: &crate::tlc::SoacBody,
        inputs: &[ArrayExpr],
        result_ty: Type<TypeName>,
    ) -> Result<NodeId, ConvertError> {
        let (dest_sym, dest_elem_ty) = match dest {
            crate::tlc::Place::LocalArray { id, elem_ty, .. } => (*id, elem_ty.clone()),
            _ => {
                return Err(ConvertError::Unsupported(
                    "scatter destination must be a storage-buffer param".into(),
                ));
            }
        };
        let dest_view = *self.locals.get(&dest_sym).ok_or_else(|| {
            ConvertError::GraphError(
                "scatter destination is not a bound #[storage] view (must be a storage param)".into(),
            )
        })?;

        // The envelope `(xs..) -> (index, value)` is a lifted function post-defunc.
        let func = self.lambda_fn_name(&lam.lam)?;
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
        let input_elem_types: Vec<Type<TypeName>> = input_array_types
            .iter()
            .zip(inputs.iter())
            .map(|(ty, ae)| self.value_elem_type(ty, ae))
            .collect();
        let capture_nids: Vec<NodeId> =
            lam.captures.iter().map(|(_, _, t)| self.convert_term(t)).collect::<Result<_, _>>()?;

        // `[dest_view, inputs..]` — captures live on the `SegBody`.
        let mut operands: SmallVec<[NodeId; 4]> = smallvec![dest_view];
        operands.extend_from_slice(&input_nids);

        Ok(self.emit_soac(
            EgirSoac::Hist {
                body: SegBody {
                    region: self.region(func),
                    captures: capture_nids,
                },
                input_array_types,
                input_elem_types,
                index_type,
                value_type,
                dest_elem_type: dest_elem_ty,
                update_policy: HistUpdatePolicy::OrderedOverwrite,
                execution: super::types::HistExecution::Raw,
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
        let arr_nid = self.convert_array_expr_value(input)?;
        let arr_ty = self.value_array_type(arr_nid, input);
        let elem_ty = self.value_elem_type(&arr_ty, input);
        let init_nid = self.convert_term(ne)?;

        // Emit as Screma { 0 maps, 1 Reduce accumulator } + project field
        // 0. Reduce's `op` is both the step (per-element) and the
        // reduce_op (phase 2 combiner).
        let operands: SmallVec<[NodeId; 4]> = smallvec![arr_nid, init_nid];
        let tuple_ty = Type::Constructed(TypeName::Tuple(1), vec![result_ty.clone()]);
        let screma_nid = self.emit_soac(
            EgirSoac::Screma {
                map_bodies: vec![],
                accumulators: vec![super::types::ScremaOperator {
                    kind: crate::tlc::ScremaAccumulator::Reduce,
                    step: SegBody {
                        region: self.region(op_name.clone()),
                        captures: capture_nids,
                    },
                    combine: SegBody {
                        region: self.region(op_name),
                        captures: vec![],
                    },
                    input_indices: vec![0],
                }],
                input_array_types: vec![arr_ty],
                input_elem_types: vec![elem_ty],
                map_output_elem_types: vec![],
                map_input_indices: vec![],
                map_destinations: vec![],
                acc_destinations: vec![SoacDestination::Fresh],
            },
            operands,
            tuple_ty,
        );
        Ok(self.intern_pure(PureOp::Project { index: 0 }, smallvec![screma_nid], result_ty))
    }

    /// Lower a single-output `Screma` (one map lane xor one accumulator) whose
    /// TLC `result_ty` is that output directly (no tuple wrapper). Builds the
    /// general `Tuple(1)` Screma via `convert_soac_screma`, then projects field
    /// 0 to recover the bare output — scalar for a single `Reduce`, an array for
    /// a single `Scan` or map lane. (The genuine multi-output path is
    /// `convert_soac_screma` directly.)
    fn convert_soac_screma_single(
        &mut self,
        lanes: &[crate::tlc::ScremaLane],
        accumulators: &[crate::tlc::ScremaAccumulatorSpec],
        inputs: &[ArrayExpr],
        result_ty: Type<TypeName>,
    ) -> Result<NodeId, ConvertError> {
        debug_assert_eq!(lanes.len() + accumulators.len(), 1, "single-output Screma");
        let tuple_ty = Type::Constructed(TypeName::Tuple(1), vec![result_ty.clone()]);
        let screma_nid = self.convert_soac_screma(lanes, accumulators, inputs, tuple_ty)?;
        Ok(self.intern_pure(PureOp::Project { index: 0 }, smallvec![screma_nid], result_ty))
    }

    fn convert_soac_screma(
        &mut self,
        lanes: &[crate::tlc::ScremaLane],
        accumulators: &[crate::tlc::ScremaAccumulatorSpec],
        inputs: &[ArrayExpr],
        result_ty: Type<TypeName>,
    ) -> Result<NodeId, ConvertError> {
        let result_fields = match &result_ty {
            Type::Constructed(TypeName::Tuple(_), fields)
                if fields.len() == lanes.len() + accumulators.len() =>
            {
                fields.clone()
            }
            other => {
                return Err(ConvertError::GraphError(format!(
                    "screma result must be a tuple with {} mapped and {} accumulator fields, got {other:?}",
                    lanes.len(),
                    accumulators.len()
                )));
            }
        };

        let map_funcs: Vec<String> =
            lanes.iter().map(|lane| self.lambda_fn_name(&lane.lam.lam)).collect::<Result<_, _>>()?;
        let map_capture_nids: Vec<Vec<NodeId>> = lanes
            .iter()
            .map(|lane| {
                lane.lam
                    .captures
                    .iter()
                    .map(|(_, _, t)| self.convert_term(t))
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<_, _>>()?;

        let mut pending_accs = Vec::with_capacity(accumulators.len());
        let mut acc_init_nids = Vec::with_capacity(accumulators.len());
        for acc in accumulators {
            let step_func = self.lambda_fn_name(&acc.step_lam.lam)?;
            let reduce_op_func = self.lambda_fn_name(&acc.reduce_op.lam)?;
            let step_caps: Vec<NodeId> = acc
                .step_lam
                .captures
                .iter()
                .map(|(_, _, t)| self.convert_term(t))
                .collect::<Result<_, _>>()?;
            let reduce_op_caps: Vec<NodeId> = acc
                .reduce_op
                .captures
                .iter()
                .map(|(_, _, t)| self.convert_term(t))
                .collect::<Result<_, _>>()?;
            let init_nid = self.convert_term(&acc.ne)?;
            pending_accs.push(super::types::ScremaOperator {
                kind: acc.kind,
                step: SegBody {
                    region: self.region(step_func),
                    captures: step_caps,
                },
                combine: SegBody {
                    region: self.region(reduce_op_func),
                    captures: reduce_op_caps,
                },
                input_indices: vec![],
            });
            acc_init_nids.push(init_nid);
        }

        let input_nids: Vec<NodeId> =
            inputs.iter().map(|ae| self.convert_array_expr_value(ae)).collect::<Result<_, _>>()?;
        let input_arr_types: Vec<Type<TypeName>> =
            inputs.iter().zip(input_nids.iter()).map(|(ae, nid)| self.value_array_type(*nid, ae)).collect();
        let input_elem_types: Vec<Type<TypeName>> = input_arr_types
            .iter()
            .zip(inputs.iter())
            .map(|(ty, ae)| self.value_elem_type(ty, ae))
            .collect();

        let mut map_output_elem_types = Vec::with_capacity(lanes.len());
        for map_idx in 0..lanes.len() {
            let map_array_ty = result_fields[map_idx].clone();
            let elem_ty = if map_array_ty.is_array() {
                map_array_ty.elem_type().expect("Array has elem").clone()
            } else if super::soac_expand::as_soa_tuple(&map_array_ty).is_some() {
                super::soac_expand::soa_element_type(&map_array_ty)
            } else {
                return Err(ConvertError::GraphError(format!(
                    "screma mapped result must be an array or SoA tuple, got {map_array_ty:?}"
                )));
            };
            map_output_elem_types.push(elem_ty);
        }

        // Captures live on each `SegBody`; the operand list is positional
        // data flow only: `[inputs.., init_accs.., output_views..]`. Output
        // views are appended later by output realization / chunking.
        let mut operands: SmallVec<[NodeId; 4]> = SmallVec::new();
        operands.extend(input_nids.iter().copied());
        operands.extend(acc_init_nids.iter().copied());

        Ok(self.emit_soac(
            EgirSoac::Screma {
                map_bodies: map_funcs
                    .into_iter()
                    .zip(map_capture_nids.iter())
                    .map(|(name, captures)| SegBody {
                        region: self.region(name),
                        captures: captures.clone(),
                    })
                    .collect(),
                accumulators: pending_accs,
                input_array_types: input_arr_types,
                input_elem_types,
                map_output_elem_types,
                map_input_indices: lanes.iter().map(|lane| lane.input_indices.clone()).collect(),
                map_destinations: vec![SoacDestination::Fresh; lanes.len()],
                acc_destinations: vec![SoacDestination::Fresh; accumulators.len()],
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
        let arr_nid = self.convert_array_expr_value(input)?;
        let arr_ty = self.value_array_type(arr_nid, input);
        // Input element type can differ from the accumulator type when
        // fusion has folded a `map` producer into the scan combiner
        // (`scan(op, ne, map(g, xs))` ⇒ step takes `(acc: Acc, x: T)`
        // with `T != Acc`).
        let input_elem_ty = self.value_elem_type(&arr_ty, input);
        let init_nid = self.convert_term(ne)?;

        let operands: SmallVec<[NodeId; 4]> = smallvec![arr_nid, init_nid];

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
        let project_ty = if matches!(destination, SoacDestination::InputBuffer) {
            arr_ty.clone()
        } else {
            result_ty
                .elem_type()
                .and_then(|elem| shape_preserving_result_ty(&arr_ty, elem, &result_ty))
                .unwrap_or_else(|| result_ty.clone())
        };
        let tuple_ty = Type::Constructed(TypeName::Tuple(1), vec![project_ty.clone()]);
        let screma_nid = self.emit_soac(
            EgirSoac::Screma {
                map_bodies: vec![],
                accumulators: vec![super::types::ScremaOperator {
                    kind: crate::tlc::ScremaAccumulator::Scan,
                    step: SegBody {
                        region: self.region(op_name),
                        captures: capture_nids,
                    },
                    combine: SegBody {
                        region: self.region(reduce_name),
                        captures: vec![],
                    },
                    input_indices: vec![0],
                }],
                input_array_types: vec![arr_ty],
                input_elem_types: vec![input_elem_ty],
                map_output_elem_types: vec![],
                map_input_indices: vec![],
                map_destinations: vec![],
                acc_destinations: vec![destination],
            },
            operands,
            tuple_ty,
        );
        Ok(self.intern_pure(PureOp::Project { index: 0 }, smallvec![screma_nid], project_ty))
    }

    fn convert_soac_filter(
        &mut self,
        map_lam: Option<&SoacBody>,
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

        // A fused producer map (`filter(p, map(f, xs))`): the loop applies `f` to
        // each input element before the predicate and keeps `f(x)`. The output
        // element type is `f`'s return type; the input element type stays the
        // array's. The fused map's captures live on its `SegBody`.
        let (map_body, output_elem_ty): (Option<SegBody>, Type<TypeName>) = match map_lam {
            Some(f) => {
                let name = self.lambda_fn_name(&f.lam)?;
                let caps: Vec<NodeId> =
                    f.captures.iter().map(|(_, _, t)| self.convert_term(t)).collect::<Result<_, _>>()?;
                (
                    Some(SegBody {
                        region: self.region(name),
                        captures: caps,
                    }),
                    f.lam.ret_ty.clone(),
                )
            }
            None => (None, elem_ty.clone()),
        };
        let pred_body = SegBody {
            region: self.region(pred_name),
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
                EgirSoac::Filter {
                    state: super::types::FilterState::Raw,
                    map_body,
                    output_elem_type: output_elem_ty,
                    pred_body,
                    input_array_type: arr_ty,
                    input_elem_type: elem_ty,
                    output: super::types::FilterOutput::Local {
                        capacity: size,
                        destination,
                    },
                },
                operands,
                bounded_result_ty,
            ));
        }

        // Runtime-sized input: compact the kept elements into a reserved
        // scratch storage buffer (capacity = input element count), and yield a
        // runtime-length view over it. The surviving count is the view's `len`
        // operand. A runtime-sized result cannot back a function-local array.
        //
        // The scratch buffer must live in a descriptor set the host sees and a
        // binding namespace seeded above the entry's params/outputs. Only an
        // `SemanticEntry` provides both; an `EgirFunc` has no `storage_bindings`
        // interface and an unseeded cursor. A runtime `filter` reaching here in
        // a standalone function (one monomorphize/inlining didn't fold into its
        // caller) is caught at `convert_function` — see the guard there.
        let input_binding = crate::types::array_view_buffer(&arr_ty).ok_or_else(|| {
            ConvertError::GraphError(
                "filter: runtime-sized input has no concrete buffer — its size is \
                 not statically known and it is not backed by a storage buffer"
                    .into(),
            )
        })?;
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
        let scratch_out = self.alloc_scratch_binding();
        self.extra_storage_bindings.push(crate::interface::StorageBindingDecl {
            binding: scratch_out,
            role: crate::interface::StorageRole::Output,
            elem_ty: output_elem_ty.clone(),
            // Host derives the element count from the input buffer
            // (`src_elem_bytes`), then sizes the output buffer in output elements
            // (`elem_bytes`); these differ when a fused map changes the element type.
            length: Some(crate::pipeline_descriptor::BufferLen::LikeInput {
                set: input_binding.set,
                binding: input_binding.binding,
                elem_bytes: output_elem_bytes,
                src_elem_bytes: input_elem_bytes,
            }),
        });
        let view_result_ty =
            crate::types::view_array_of(&output_elem_ty, crate::types::buffer_tag(scratch_out));
        Ok(self.emit_soac(
            EgirSoac::Filter {
                state: super::types::FilterState::Raw,
                map_body,
                output_elem_type: output_elem_ty,
                pred_body,
                input_array_type: arr_ty,
                input_elem_type: elem_ty,
                output: super::types::FilterOutput::Runtime {
                    scratch: super::program::SemanticResourceRef::Binding(scratch_out),
                    length: super::types::RuntimeFilterLength::ViewOnly,
                },
                // Set by `realize_outputs` only when this filter is a compute
                // entry's output (it then needs a host-readable length cell).
            },
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
                let mut ids = crate::tlc::TermIdSource::new();
                let t = crate::tlc::atom_var_term(*vr, var_ty.clone(), &mut ids);
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
                        Type::Constructed(TypeName::ArrayVariantView, vec![]),
                        Type::Constructed(TypeName::SizePlaceholder, vec![]),
                        crate::types::buffer_tag(*binding),
                    ],
                );
                let result_nid = self.graph.alloc_side_effect_result(array_ty);
                let effect_in = EffectToken(0);
                let effect_out = self.alloc_effect();
                self.graph.skeleton.blocks[self.current_block].side_effects.push(SideEffect {
                    semantic_id: None,
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
                ty if super::soac_expand::as_soa_tuple(ty).is_some() => {
                    super::soac_expand::soa_element_type(ty)
                }
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
            ArrayExpr::StorageView(crate::tlc::StorageView { elem_ty, .. }) => elem_ty.clone(),
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
        if let Some(ty) = self.graph.types.get(&nid) {
            if matches!(ty, Type::Constructed(TypeName::Array, _))
                || super::soac_expand::as_soa_tuple(ty).is_some()
            {
                return ty.clone();
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
        if super::soac_expand::as_soa_tuple(arr_ty).is_some() {
            return super::soac_expand::soa_element_type(arr_ty);
        }
        self.array_expr_elem_type(fallback)
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
        interface::Attribute::VertexSlot(n) | interface::Attribute::Varying(n) => {
            Some(IoDecoration::Location(*n))
        }
        _ => None,
    }
}

/// The render-target resource name of a `#[target(name)]` output attribute.
fn target_of(attr: Option<&interface::Attribute>) -> Option<String> {
    match attr {
        Some(interface::Attribute::Target(name)) => Some(name.clone()),
        _ => None,
    }
}

/// Build entry outputs from an AST `EntryDecl`.
/// For compute shaders, non-unit outputs get sequential storage bindings starting at `binding_start`.
/// Walk a (post-normalize_outputs) entry body and collect each
/// `OutputSlotStore { slot_index, value, .. }`'s `value.ty` into a
/// dense `Vec<Option<Type>>` indexed by slot. Returns empty if the
/// body contains no `OutputSlotStore` (graphics entries; the
/// multi-output non-Tuple fallthrough that `normalize_outputs` leaves
/// unrewritten). `build_entry_outputs` reads from this in preference
/// to the parse-time `entry.outputs[i].ty`, since the body's term
/// types are current with `monomorphize` / `buffer_specialize`.
fn collect_output_slot_value_tys(body: &crate::tlc::Term) -> Vec<Option<Type<TypeName>>> {
    use crate::tlc::TermKind;
    let mut out: Vec<(usize, Type<TypeName>)> = Vec::new();
    let mut cur = body;
    loop {
        match &cur.kind {
            TermKind::Let { rhs, body, .. } => {
                if let TermKind::OutputSlotStore {
                    slot_index, value, ..
                } = &rhs.kind
                {
                    out.push((*slot_index, value.ty.clone()));
                }
                cur = body;
            }
            _ => break,
        }
    }
    if out.is_empty() {
        return Vec::new();
    }
    let max = out.iter().map(|(i, _)| *i).max().unwrap();
    let mut dense: Vec<Option<Type<TypeName>>> = vec![None; max + 1];
    for (i, ty) in out {
        dense[i] = Some(ty);
    }
    dense
}

/// Unwrap an existential array return `?k. [k]T` to its inner array type
/// `[k]T`. A runtime `filter`'s entry output is declared existential, but the
/// host-visible buffer is a plain runtime `[]T`; the backend lays out a storage
/// buffer from the element type alone, so the inner array (size variable and
/// all) is what `create_storage_buffer` needs. Non-existential types pass
/// through unchanged.
fn unwrap_existential_array(ty: &Type<TypeName>) -> Type<TypeName> {
    match ty {
        Type::Constructed(TypeName::Existential(_), args) if !args.is_empty() => args[0].clone(),
        _ => ty.clone(),
    }
}

/// Shape-preserving result type for a non-in-place `map`/`scan`.
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
    forced_output_binding: Option<BindingRef>,
) -> Result<Vec<EntryOutput>, ConvertError> {
    use EntryOutput;
    let mut forced_remaining = forced_output_binding;
    // Pick a `BufferLen` policy for the output binding, in order:
    //
    //   1. Output type carries a compile-time-known `Size(n)` literal
    //      → `Fixed { bytes: n * elem_bytes }`.
    //   2. Output's size variable matches one of the entry's storage
    //      inputs (the type checker has unified them) → `LikeInput`
    //      tracking that input.
    //   3. The parallelize plan said the entry is dispatch-sized
    //      (a single Map/Scan SOAC at the tail, or a forced gather
    //      prepass) → `SameAsDispatch { elem_bytes }`.
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
                return Ok(None);
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
                    let Some(in_binding) = input.storage_binding else {
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
    let mut storage_binding_for = |ty: &Type<TypeName>, is_compute: bool| -> Option<BindingRef> {
        if is_compute && !matches!(ty, Type::Constructed(TypeName::Unit, _)) {
            // Honor the planned binding for the first storage output if
            // present; subsequent outputs (tuple-return entries) keep
            // auto-allocating.
            if let Some(b) = forced_remaining.take() {
                return Some(b);
            }
            Some(BindingRef::new(AUTO_STORAGE_SET, binding_ids.next_id()))
        } else {
            None
        }
    };

    // Normalized compute entries type their return position as
    // `SideEffect` (the body produces no value; it writes via
    // `OutputSlotStore`). The decoration is on `entry.outputs[i]`; the
    // per-slot ty must come from `slot_value_tys[i]` (the body's
    // `OutputSlotStore.value.ty` post-monomorphize/buffer_specialize),
    // not from `entry.outputs[i].ty` (which is parse-time-frozen and
    // can still carry unresolved Array-variant type variables).
    // Source declared no return type (`entry foo(...) () = …`). For
    // both Unit (compute entries whose tail is a side-effectful builtin
    // like storage-image `with` — `normalize_outputs` leaves the body Unit-
    // typed when `n_outputs == 0`) and SideEffect (a normalised
    // `OutputSlotStore` chain that bottomed out empty), there's no
    // logical output slot to emit. Returning a synthetic Unit-typed
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

    if matches!(ret_type, Type::Constructed(TypeName::SideEffect, _)) {
        let slot_count = slot_value_tys.len().max(entry.outputs.len());
        return (0..slot_count)
            .map(|i| {
                let output = entry.outputs.get(i);
                let raw = slot_value_tys
                    .get(i)
                    .and_then(|t| t.as_ref())
                    .or_else(|| output.map(|output| &output.ty))
                    .ok_or_else(|| {
                        ConvertError::Internal(format!(
                            "normalized output slot {i} has neither a value type nor a declaration"
                        ))
                    })?;
                // Canonicalize for storage layout: strip `Unique<_>` and
                // unwrap any top-level `Existential` (e.g. a runtime
                // `filter` output declared `?k. [k]T`) so
                // `create_storage_buffer` sees the concrete runtime
                // array. See `types::canonical_storage_buffer_ty`.
                let ty = crate::types::canonical_storage_buffer_ty(raw);
                let storage_binding = storage_binding_for(&ty, is_compute);
                let length = length_for(storage_binding, &ty)?;
                Ok(EntryOutput {
                    ty,
                    decoration: output
                        .and_then(|output| output.attribute.as_ref())
                        .and_then(convert_to_io_decoration),
                    target: target_of(output.and_then(|output| output.attribute.as_ref())),
                    storage_binding,
                    length,
                })
            })
            .collect();
    }

    if entry.outputs.iter().all(|o| o.attribute.is_none()) && entry.outputs.len() == 1 {
        if !matches!(ret_type, Type::Constructed(TypeName::Unit, _)) {
            let ty = crate::types::canonical_storage_buffer_ty(ret_type);
            let storage_binding = storage_binding_for(&ty, is_compute);
            let length = length_for(storage_binding, &ty)?;
            Ok(vec![EntryOutput {
                ty,
                decoration: None,
                target: None,
                storage_binding,
                length,
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
                let ty = crate::types::canonical_storage_buffer_ty(ty);
                let storage_binding = storage_binding_for(&ty, is_compute);
                let length = length_for(storage_binding, &ty)?;
                Ok(EntryOutput {
                    ty,
                    decoration: output.attribute.as_ref().and_then(convert_to_io_decoration),
                    target: target_of(output.attribute.as_ref()),
                    storage_binding,
                    length,
                })
            })
            .collect()
    } else {
        let ty = crate::types::canonical_storage_buffer_ty(ret_type);
        let storage_binding = storage_binding_for(&ty, is_compute);
        let length = length_for(storage_binding, &ty)?;
        let first_attr = entry.outputs.first().and_then(|o| o.attribute.as_ref());
        Ok(vec![EntryOutput {
            ty,
            decoration: first_attr.and_then(convert_to_io_decoration),
            target: target_of(first_attr),
            storage_binding,
            length,
        }])
    }
}

#[cfg(test)]
#[path = "from_tlc_tests.rs"]
mod from_tlc_tests;
