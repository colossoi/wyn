//! Direct TLC to EGraph conversion.
//!
//! Converts a TLC program directly to the acyclic e-graph representation,
//! bypassing the sequential SSA construction of `to_ssa`. Pure operations
//! are hash-consed (giving GVN for free), and the result is elaborated
//! back to `FuncBody` via demand-driven scheduling (giving DCE for free).

use std::collections::{HashMap, HashSet};

use crate::ast::TypeName;
use crate::ssa::types::ViewSource;
use crate::ssa::types::{ControlHeader, EffectToken, FuncBody, Function, InstKind, Program, ValueRef};
use crate::tlc::{
    ArrayExpr, Def as TlcDef, DefMeta, Lambda, LoopKind, Program as TlcProgram, SoacOp, Term, TermKind,
};
use crate::types::TypeExt;
use crate::{SymbolId, SymbolTable};
use polytype::Type;
use smallvec::{SmallVec, smallvec};
use wyn_ssa::BlockId;

use super::pipeline::{EgirRaw, EntryEgir, FuncEgir, ProgramEgir, Raw};
use super::types::*;
use crate::ast::Span;
use crate::pipeline_descriptor::PipelineDescriptor;

// ============================================================================
// Error type
// ============================================================================

#[derive(Debug)]
pub enum ConvertError {
    /// Error during EGraph construction.
    GraphError(String),
    /// Unsupported TLC construct (todo).
    Unsupported(String),
}

impl std::fmt::Display for ConvertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConvertError::GraphError(msg) => write!(f, "EGraph conversion error: {}", msg),
            ConvertError::Unsupported(msg) => write!(f, "Unsupported: {}", msg),
        }
    }
}

impl std::error::Error for ConvertError {}

// ============================================================================
// Public entry point
// ============================================================================

/// Convert a TLC program into a raw EGIR program — each function and entry
/// point becomes a per-body `EGraph` + metadata, waiting for the caller to
/// chain the pipeline (`expand_soacs → [materialize →] optimize_skeleton →
/// elaborate`).
pub fn convert_program(
    program: &TlcProgram,
    pipeline: PipelineDescriptor,
) -> Result<EgirRaw, ConvertError> {
    let top_level: HashMap<SymbolId, &TlcDef> = program.defs.iter().map(|d| (d.name, d)).collect();
    let symbols = &program.symbols;

    let constants_by_name: HashMap<String, SymbolId> = program
        .defs
        .iter()
        .filter(|d| d.arity == 0 && matches!(&d.meta, DefMeta::Function))
        .filter_map(|d| symbols.get(d.name).map(|n| (n.clone(), d.name)))
        .collect();

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

        let mut converter = Converter::new(
            &top_level,
            &constants_by_name,
            symbols,
            pure_constant_names.clone(),
        );
        if let Ok(result_nid) = converter.convert_term(&def.body) {
            converter.set_return(Some(result_nid));
            if let Some(body) = converter.probe_constant_body(def.body.ty.clone()) {
                if is_purely_constant_body(&body) {
                    pure_constant_names.insert(def_name.clone());
                    constants.push(crate::ssa::types::Constant {
                        name: def_name,
                        body,
                        result_ty: def.body.ty.clone(),
                    });
                    continue;
                }
            }
        }
    }

    // Phase 2: convert functions and entry points into raw EGIR records.
    let mut functions: Vec<FuncEgir<Raw>> = Vec::new();
    let mut externs: Vec<Function> = Vec::new();
    let mut entry_points: Vec<EntryEgir<Raw>> = Vec::new();

    for def in &program.defs {
        match &def.meta {
            DefMeta::Function => {
                let def_name = symbols.get(def.name).expect("BUG: symbol not in table");
                if pure_constant_names.contains(def_name) {
                    continue;
                }
                match convert_function(def, &top_level, &constants_by_name, symbols, &pure_constant_names)?
                {
                    ConvertedFunc::Extern(f) => externs.push(f),
                    ConvertedFunc::Regular(fe) => functions.push(fe),
                }
            }
            DefMeta::EntryPoint(entry) => {
                let ep = convert_entry_point(
                    def,
                    entry,
                    &top_level,
                    &constants_by_name,
                    symbols,
                    &pure_constant_names,
                )?;
                entry_points.push(ep);
            }
        }
    }

    Ok(ProgramEgir::new(
        functions,
        externs,
        entry_points,
        constants,
        program.uniforms.clone(),
        program.storage.clone(),
        pipeline,
    ))
}

enum ConvertedFunc {
    Extern(Function),
    Regular(FuncEgir<Raw>),
}

// ============================================================================
// Function conversion
// ============================================================================

fn convert_function(
    def: &TlcDef,
    top_level: &HashMap<SymbolId, &TlcDef>,
    constants_by_name: &HashMap<String, SymbolId>,
    symbols: &SymbolTable,
    pure_constants: &HashSet<String>,
) -> Result<ConvertedFunc, ConvertError> {
    let def_name = symbols.get(def.name).expect("BUG").clone();

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
        .map(|(sym, ty)| {
            let name = symbols.get(*sym).unwrap_or(&format!("arg")).clone();
            (ty.clone(), name)
        })
        .collect();

    let mut converter = Converter::new(top_level, constants_by_name, symbols, pure_constants.clone());
    for (i, (sym, ty)) in params.iter().enumerate() {
        let nid = converter.graph.add_func_param(i, ty.clone());
        converter.locals.insert(*sym, nid);
    }
    let result = converter.convert_term(inner_body)?;
    converter.set_return(Some(result));

    let (graph, control_headers) = converter.into_graph_parts();
    Ok(ConvertedFunc::Regular(FuncEgir::new(
        def_name,
        def.body.span,
        None,
        param_info,
        ret_type,
        graph,
        control_headers,
    )))
}

/// Convert an entry point directly via the EGraph Converter — one path, no
/// round-trip through the legacy SSA builder.
fn convert_entry_point(
    def: &TlcDef,
    entry: &crate::ast::EntryDecl,
    top_level: &HashMap<SymbolId, &TlcDef>,
    constants_by_name: &HashMap<String, SymbolId>,
    symbols: &SymbolTable,
    pure_constants: &HashSet<String>,
) -> Result<EntryEgir<Raw>, ConvertError> {
    use crate::ast;
    use crate::ssa::types::{EntryInput, ExecutionModel, IoDecoration};

    let def_name = symbols.get(def.name).expect("BUG: symbol not in table").clone();
    let (inner_body, params) = extract_lambda_params(&def.body);
    let is_compute = matches!(entry.entry_type, ast::Attribute::Compute);

    // Build entry inputs with IO decorations, storage bindings, push constants.
    let mut inputs: Vec<EntryInput> = Vec::with_capacity(params.len());
    let mut param_syms: Vec<SymbolId> = Vec::with_capacity(params.len());
    let mut binding_num: u32 = 0;
    let mut pc_offset: u32 = 0;

    for (i, (sym, ty)) in params.iter().enumerate() {
        let name = symbols.get(*sym).expect("BUG: symbol not in table").clone();
        let decoration = entry.params.get(i).and_then(extract_io_decoration);
        let size_hint = entry.params.get(i).and_then(extract_size_hint);

        let storage_binding = if is_compute && is_unsized_array(ty) {
            let b = (0, binding_num);
            binding_num += 1;
            Some(b)
        } else {
            None
        };

        let push_constant_offset = if is_compute
            && storage_binding.is_none()
            && !matches!(&decoration, Some(IoDecoration::BuiltIn(_)))
        {
            let offset = pc_offset;
            pc_offset += crate::ssa::layout::type_byte_size(ty).unwrap_or(4);
            Some(offset)
        } else {
            None
        };

        param_syms.push(*sym);
        inputs.push(EntryInput {
            name,
            ty: ty.clone(),
            decoration,
            size_hint,
            storage_binding,
            push_constant_offset,
        });
    }

    let ret_type = inner_body.ty.clone();
    let param_info: Vec<(Type<TypeName>, String)> =
        params.iter().map(|(sym, ty)| (ty.clone(), symbols.get(*sym).expect("BUG").clone())).collect();

    let mut converter = Converter::new(top_level, constants_by_name, symbols, pure_constants.clone());

    // Register function params as FuncParam NodeIds.
    for (i, (sym, ty)) in params.iter().enumerate() {
        let nid = converter.graph.add_func_param(i, ty.clone());
        converter.locals.insert(*sym, nid);
    }

    // Wrap storage-buffer inputs in whole-buffer StorageViews so any direct use
    // of the param sees a view rather than a raw buffer handle.
    for (idx, input) in inputs.iter().enumerate() {
        if let Some((set, binding)) = input.storage_binding {
            let view_nid = converter.emit_storage_view(set, binding, input.ty.clone());
            converter.locals.insert(param_syms[idx], view_nid);
        }
    }

    let execution_model = match &entry.entry_type {
        ast::Attribute::Vertex => ExecutionModel::Vertex,
        ast::Attribute::Fragment => ExecutionModel::Fragment,
        ast::Attribute::Compute => ExecutionModel::Compute {
            local_size: (64, 1, 1),
        },
        _ => panic!("Invalid entry type attribute: {:?}", entry.entry_type),
    };

    let outputs = build_entry_outputs(entry, &inner_body.ty, is_compute, binding_num);
    let is_unit_return = matches!(ret_type, Type::Constructed(TypeName::Unit, _));

    // Pre-create the output StorageView for compute+storage-output entries so
    // it's available for the Map→MapInto rewrite on the result node.
    let has_storage_output =
        is_compute && !is_unit_return && outputs.iter().any(|o| o.storage_binding.is_some());
    let compute_output_view = if has_storage_output {
        outputs.first().map(|output| {
            let (set, binding) =
                output.storage_binding.expect("BUG: compute output without storage binding");
            let elem_ty = output.ty.elem_type().cloned().unwrap_or(output.ty.clone());
            converter.emit_storage_view(set, binding, elem_ty)
        })
    } else {
        None
    };

    // Convert body.
    let result_nid = converter.convert_term(inner_body)?;

    // Finalize based on entry-point kind.
    if is_unit_return {
        converter.set_return(None);
    } else if is_compute && !outputs.is_empty() {
        if let Some(output_view_nid) = compute_output_view {
            // Rewrite the Soac::Map/Scan whose result is result_nid to
            // Map/ScanInto so its output goes directly to the view.
            rewrite_map_scan_to_into(&mut converter.graph, result_nid, output_view_nid);
            converter.set_return(None);
        } else {
            // Non-view result: emit storage stores for each output.
            emit_compute_output_stores(&mut converter, result_nid, &outputs);
            converter.set_return(None);
        }
    } else if !is_compute && !is_unit_return && !outputs.is_empty() {
        // Vertex/fragment: OutputPtr + Store per output.
        emit_vertex_fragment_output_stores(&mut converter, result_nid, &outputs);
        converter.set_return(None);
    } else {
        converter.set_return(Some(result_nid));
    }

    let (graph, control_headers) = converter.into_graph_parts();
    Ok(EntryEgir::new(
        def_name,
        def.body.span,
        execution_model,
        inputs,
        outputs,
        param_info,
        ret_type,
        graph,
        control_headers,
    ))
}

/// Rewrite a `Soac::Map` / `Soac::Scan` skeleton side-effect whose result is
/// `target_result` into the corresponding `MapInto` / `ScanInto` variant
/// writing to `output_view`.
fn rewrite_map_scan_to_into(graph: &mut EGraph, target_result: NodeId, output_view: NodeId) {
    for (_bid, block) in graph.skeleton.blocks.iter_mut() {
        for se in &mut block.side_effects {
            if se.result != Some(target_result) {
                continue;
            }
            let kind = se.kind.clone();
            match kind {
                SideEffectKind::Pending(PendingSoac::Map {
                    func,
                    input_array_types,
                    input_elem_types,
                    output_elem_type,
                }) => {
                    se.kind = SideEffectKind::Pending(PendingSoac::MapInto {
                        func,
                        input_array_types,
                        input_elem_types,
                        output_elem_type,
                    });
                    se.operand_nodes.push(output_view);
                }
                SideEffectKind::Pending(PendingSoac::Scan {
                    func,
                    input_array_type,
                    input_elem_type,
                }) => {
                    se.kind = SideEffectKind::Pending(PendingSoac::ScanInto {
                        func,
                        input_array_type,
                        input_elem_type,
                    });
                    se.operand_nodes.push(output_view);
                }
                _ => {}
            }
            return;
        }
    }
}

/// Compute entry with a non-view result: store the result (or its tuple
/// components) into the output storage buffers.
fn emit_compute_output_stores(
    converter: &mut Converter<'_>,
    result_nid: NodeId,
    outputs: &[crate::ssa::types::EntryOutput],
) {
    for (i, output) in outputs.iter().enumerate() {
        let (set, binding) = output.storage_binding.expect("BUG: compute output without storage binding");
        let value_nid = if outputs.len() == 1 {
            result_nid
        } else {
            converter.emit_project(result_nid, i as u32, output.ty.clone())
        };

        let fixed_size = output.ty.array_size().and_then(|s| {
            if let Type::Constructed(TypeName::Size(n), _) = s { Some(*n) } else { None }
        });
        let elem_ty = output.ty.elem_type().cloned();

        if let (Some(n), Some(et)) = (fixed_size, elem_ty) {
            let view_nid = converter.emit_storage_view(set, binding, et.clone());
            for j in 0..n {
                let elem_nid = converter.emit_project(value_nid, j as u32, et.clone());
                let idx_nid = converter.intern_u32(j as u32);
                converter.emit_storage_store(view_nid, idx_nid, elem_nid, et.clone());
            }
        } else {
            let view_nid = converter.emit_storage_view(set, binding, output.ty.clone());
            let idx_zero = converter.intern_u32(0);
            converter.emit_storage_store(view_nid, idx_zero, value_nid, output.ty.clone());
        }
    }
}

/// Vertex/fragment entry: write the result (or tuple components) to
/// `OutputPtr(i)` locations.
fn emit_vertex_fragment_output_stores(
    converter: &mut Converter<'_>,
    result_nid: NodeId,
    outputs: &[crate::ssa::types::EntryOutput],
) {
    if outputs.len() == 1 {
        let ptr_ty = Type::Constructed(
            TypeName::Pointer,
            vec![
                outputs[0].ty.clone(),
                Type::Constructed(TypeName::PointerOutput, vec![]),
            ],
        );
        let ptr_nid = converter.emit_output_ptr(0, ptr_ty);
        converter.emit_store(ptr_nid, result_nid);
    } else {
        for (i, output) in outputs.iter().enumerate() {
            let component = converter.emit_project(result_nid, i as u32, output.ty.clone());
            let ptr_ty = Type::Constructed(
                TypeName::Pointer,
                vec![
                    output.ty.clone(),
                    Type::Constructed(TypeName::PointerOutput, vec![]),
                ],
            );
            let ptr_nid = converter.emit_output_ptr(i, ptr_ty);
            converter.emit_store(ptr_nid, component);
        }
    }
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
        }
    }

    fn alloc_effect(&mut self) -> EffectToken {
        let t = EffectToken(self.next_effect);
        self.next_effect += 1;
        t
    }

    /// Set the return terminator on the current block.
    fn set_return(&mut self, result: Option<NodeId>) {
        self.graph.skeleton.blocks[self.current_block].term = SkeletonTerminator::Return(result);
    }

    // -- Entry-point emission helpers --
    //
    // These build the shader-entry-point glue (storage views, output ptrs,
    // stores) as EGraph nodes + skeleton side-effects. Pure pieces go through
    // intern_pure; memory writes go into the current skeleton block.

    fn intern_u32(&mut self, n: u32) -> NodeId {
        let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
        self.graph.intern_pure(PureOp::Uint(n.to_string()), smallvec![], u32_ty)
    }

    /// Create a pure `StorageView(Storage { set, binding })` node. Builds the
    /// implicit `offset=0` and `len=_w_intrinsic_storage_len(set, binding)`
    /// operands as pure ops.
    fn emit_storage_view(&mut self, set: u32, binding: u32, view_ty: Type<TypeName>) -> NodeId {
        let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
        let set_nid = self.intern_u32(set);
        let binding_nid = self.intern_u32(binding);
        let len_nid = self.graph.intern_pure(
            PureOp::Intrinsic("_w_intrinsic_storage_len".into()),
            smallvec![set_nid, binding_nid],
            u32_ty,
        );
        let zero_nid = self.intern_u32(0);
        self.graph.intern_pure(
            PureOp::StorageView(PureViewSource::Storage { set, binding }),
            smallvec![zero_nid, len_nid],
            view_ty,
        )
    }

    /// Emit a `Store` side-effect in the current block, returning the new effect token.
    fn emit_store(&mut self, ptr_nid: NodeId, value_nid: NodeId) -> EffectToken {
        let effect_in = EffectToken(0); // placeholder; real chain is built by elaborate
        let effect_out = self.alloc_effect();
        let kind = InstKind::Store {
            ptr: ValueRef::Ssa(Default::default()),
            value: ValueRef::Ssa(Default::default()),
        };
        self.graph.skeleton.blocks[self.current_block].side_effects.push(SideEffect {
            kind: SideEffectKind::Inst(kind),
            operand_nodes: smallvec![ptr_nid, value_nid],
            result: None,
            effects: Some((effect_in, effect_out)),
        });
        effect_out
    }

    /// Emit a store through a StorageView at `index`. Pure `StorageViewIndex`
    /// produces the pointer; the Store is effectful.
    fn emit_storage_store(
        &mut self,
        view_nid: NodeId,
        index_nid: NodeId,
        value_nid: NodeId,
        elem_ty: Type<TypeName>,
    ) {
        let ptr_nid =
            self.graph.intern_pure(PureOp::StorageViewIndex, smallvec![view_nid, index_nid], elem_ty);
        let _ = self.emit_store(ptr_nid, value_nid);
    }

    /// Emit a pure `OutputPtr(index)` node.
    fn emit_output_ptr(&mut self, index: usize, ptr_ty: Type<TypeName>) -> NodeId {
        self.graph.intern_pure(PureOp::OutputPtr { index }, smallvec![], ptr_ty)
    }

    /// Emit a pure `Project(base, index)` node.
    fn emit_project(&mut self, base_nid: NodeId, index: u32, ty: Type<TypeName>) -> NodeId {
        self.graph.intern_pure(PureOp::Project { index }, smallvec![base_nid], ty)
    }

    /// Extract the built EGraph + control_headers, leaving the rest of the
    /// Converter state behind. Used by the top-level `convert_program`
    /// phase to feed a ready-to-chain `FuncEgir<Raw>` / `EntryEgir<Raw>`.
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
    /// through `convert_program`, which returns a `ProgramEgir<Raw>` so the
    /// caller can compose the pipeline explicitly.
    fn elaborate_to_funcbody(
        self,
        params: &[(Type<TypeName>, String)],
        return_ty: Type<TypeName>,
    ) -> Option<FuncBody> {
        let (graph, control_headers) = self.into_graph_parts();
        let func = FuncEgir::<Raw>::new(
            "<probe>".to_string(),
            Span::new(0, 0, 0, 0),
            None,
            params.to_vec(),
            return_ty,
            graph,
            control_headers,
        );
        let elaborated = ProgramEgir::single_function(func).expand_soacs().optimize_skeleton().elaborate();
        elaborated.ssa.functions.into_iter().next().map(|f| f.body)
    }

    fn probe_constant_body(self, return_ty: Type<TypeName>) -> Option<FuncBody> {
        self.elaborate_to_funcbody(&[], return_ty)
    }

    // ========================================================================
    // Term conversion
    // ========================================================================

    fn convert_term(&mut self, term: &Term) -> Result<NodeId, ConvertError> {
        let ty = term.ty.clone();

        match &term.kind {
            // --- Literals ---
            TermKind::IntLit(s) => {
                let op = if matches!(&ty, Type::Constructed(TypeName::UInt(_), _)) {
                    PureOp::Uint(s.clone())
                } else {
                    PureOp::Int(s.clone())
                };
                Ok(self.graph.intern_pure(op, smallvec![], ty))
            }
            TermKind::FloatLit(f) => {
                Ok(self.graph.intern_pure(PureOp::Float(f.to_string()), smallvec![], ty))
            }
            TermKind::BoolLit(b) => Ok(self.graph.intern_pure(PureOp::Bool(*b), smallvec![], ty)),
            TermKind::StringLit(s) => {
                Ok(self.graph.intern_pure(PureOp::StringLit(s.clone()), smallvec![], ty))
            }

            // --- Variables ---
            TermKind::Var(sym) => self.convert_var(*sym, ty),

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
            TermKind::Extern(name) => {
                Ok(self.graph.intern_pure(PureOp::Extern(name.clone()), smallvec![], ty))
            }

            // --- Force (pass-through) ---
            TermKind::Force(inner) => self.convert_term(inner),

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

            // --- Should not appear after defunctionalization ---
            TermKind::Lambda(_) => {
                panic!("ICE: bare Lambda in to_egir (should be lifted)")
            }
            TermKind::BinOp(_) | TermKind::UnOp(_) => {
                panic!("ICE: bare operator in to_egir (should be inside App)")
            }
            TermKind::Pack { .. } | TermKind::Unpack { .. } => Err(ConvertError::Unsupported(
                "existentials not yet implemented".into(),
            )),
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
            return Ok(self.graph.intern_pure(PureOp::Global(name), smallvec![], ty));
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
        Ok(self.graph.intern_pure(PureOp::Global(name), smallvec![], ty))
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
                Ok(self.graph.intern_pure(PureOp::BinOp(op.op.clone()), smallvec![lhs, rhs], ty))
            }
            TermKind::UnOp(op) => {
                let operand = self.convert_term(&args[0])?;
                Ok(self.graph.intern_pure(PureOp::UnaryOp(op.op.clone()), smallvec![operand], ty))
            }
            TermKind::Var(sym) => {
                let name = self.symbols.get(*sym).expect("BUG").clone();
                self.convert_named_app(&name, *sym, args, ty)
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
        // Intrinsic patterns
        match name {
            "_w_tuple" => {
                let operands: SmallVec<[NodeId; 4]> =
                    args.iter().map(|a| self.convert_term(a)).collect::<Result<_, _>>()?;
                let n = operands.len();
                Ok(self.graph.intern_pure(PureOp::Tuple(n), operands, ty))
            }
            "_w_vec_lit" => {
                let operands: SmallVec<[NodeId; 4]> =
                    args.iter().map(|a| self.convert_term(a)).collect::<Result<_, _>>()?;
                let n = operands.len();
                Ok(self.graph.intern_pure(PureOp::Vector(n), operands, ty))
            }
            "_w_tuple_proj" => {
                if args.len() != 2 {
                    return Err(ConvertError::GraphError("_w_tuple_proj expects 2 args".into()));
                }
                let base = self.convert_term(&args[0])?;
                let index = match &args[1].kind {
                    TermKind::IntLit(s) => s.parse::<u32>().unwrap_or(0),
                    _ => {
                        return Err(ConvertError::GraphError(
                            "_w_tuple_proj index must be literal".into(),
                        ));
                    }
                };
                Ok(self.graph.intern_pure(PureOp::Project { index }, smallvec![base], ty))
            }
            "_w_index" => {
                if args.len() != 2 {
                    return Err(ConvertError::GraphError("_w_index expects 2 args".into()));
                }
                let base = self.convert_term(&args[0])?;
                let index = self.convert_term(&args[1])?;
                Ok(self.graph.intern_pure(PureOp::Index, smallvec![base, index], ty))
            }
            "_w_array_lit" => {
                let operands: SmallVec<[NodeId; 4]> =
                    args.iter().map(|a| self.convert_term(a)).collect::<Result<_, _>>()?;
                let n = operands.len();
                Ok(self.graph.intern_pure(PureOp::ArrayLit(n), operands, ty))
            }
            "_w_range" => {
                let start = self.convert_term(&args[0])?;
                let len = self.convert_term(&args[1])?;
                Ok(self.graph.intern_pure(
                    PureOp::ArrayRange { has_step: false },
                    smallvec![start, len],
                    ty,
                ))
            }
            // _w_intrinsic_storage_index(set_const, binding_const, index) → load
            // from a storage view. Emitted by buffer_specialize for functions
            // that index directly into a bound buffer.
            "_w_intrinsic_storage_index" if args.len() == 3 => {
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
                let view_nid = self.emit_storage_view(set, binding, ty.clone());
                let ptr_nid = self.graph.intern_pure(
                    PureOp::StorageViewIndex,
                    smallvec![view_nid, index_nid],
                    ty.clone(),
                );
                // Load the element; Load is effectful.
                let result_nid = self.graph.alloc_side_effect_result(ty.clone());
                let effect_in = EffectToken(0);
                let effect_out = self.alloc_effect();
                self.graph.skeleton.blocks[self.current_block].side_effects.push(SideEffect {
                    kind: SideEffectKind::Inst(InstKind::Load {
                        ptr: ValueRef::Ssa(crate::ssa::types::ValueId::default()),
                    }),
                    operand_nodes: smallvec![ptr_nid],
                    result: Some(result_nid),
                    effects: Some((effect_in, effect_out)),
                });
                Ok(result_nid)
            }
            name if name.starts_with("_w_intrinsic_") => {
                // Intrinsic call → side effect
                let arg_nids: SmallVec<[NodeId; 4]> =
                    args.iter().map(|a| self.convert_term(a)).collect::<Result<_, _>>()?;
                let arg_vrefs: Vec<ValueRef> = (0..arg_nids.len())
                    .map(|_| ValueRef::Ssa(crate::ssa::types::ValueId::default()))
                    .collect();
                let result_nid = self.graph.alloc_side_effect_result(ty.clone());
                let effect_in = EffectToken(0);
                let effect_out = self.alloc_effect();
                self.graph.skeleton.blocks[self.current_block].side_effects.push(SideEffect {
                    kind: SideEffectKind::Inst(InstKind::Intrinsic {
                        name: name.to_string(),
                        args: arg_vrefs,
                    }),
                    operand_nodes: arg_nids,
                    result: Some(result_nid),
                    effects: Some((effect_in, effect_out)),
                });
                Ok(result_nid)
            }
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
                            kind: SideEffectKind::Inst(InstKind::Call {
                                func: name.to_string(),
                                args: arg_vrefs,
                            }),
                            operand_nodes: arg_nids,
                            result: Some(result_nid),
                            effects: Some((effect_in, effect_out)),
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
                        kind: SideEffectKind::Inst(InstKind::Call {
                            func: name.to_string(),
                            args: arg_vrefs,
                        }),
                        operand_nodes: arg_nids,
                        result: Some(result_nid),
                        effects: Some((effect_in, effect_out)),
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
        let zero = self.graph.intern_pure(PureOp::Int("0".into()), smallvec![], i32_ty.clone());
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
        let cond_nid =
            self.graph.intern_pure(PureOp::BinOp("<".into()), smallvec![idx_nid, bound_nid], bool_ty);
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
        let one = self.graph.intern_pure(PureOp::Int("1".into()), smallvec![], i32_ty.clone());
        let next_i = self.graph.intern_pure(PureOp::BinOp("+".into()), smallvec![idx_nid, one], i32_ty);
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

        // Length intrinsic
        let len_nid = self.graph.intern_pure(
            PureOp::UnaryOp("_w_intrinsic_length".into()),
            smallvec![iter_nid],
            i32_ty.clone(),
        );
        let zero = self.graph.intern_pure(PureOp::Int("0".into()), smallvec![], i32_ty.clone());
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
        let cond_nid =
            self.graph.intern_pure(PureOp::BinOp("<".into()), smallvec![idx_nid, len_nid], bool_ty);
        self.graph.skeleton.blocks[header].term = SkeletonTerminator::CondBranch {
            cond: cond_nid,
            then_target: body_block,
            then_args: vec![],
            else_target: exit,
            else_args: vec![acc_nid],
        };

        // Body: index into iterator, bind elem_var
        self.current_block = body_block;
        let elem_nid = self.graph.intern_pure(PureOp::Index, smallvec![iter_nid, idx_nid], elem_ty.clone());
        self.locals.insert(elem_var, elem_nid);

        let new_acc = self.convert_term(body)?;
        let one = self.graph.intern_pure(PureOp::Int("1".into()), smallvec![], i32_ty.clone());
        let next_i = self.graph.intern_pure(PureOp::BinOp("+".into()), smallvec![idx_nid, one], i32_ty);
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
            SoacOp::Map { lam, inputs } => self.convert_soac_map(lam, inputs, ty),
            SoacOp::Reduce { op, ne, input, .. } => self.convert_soac_reduce(op, ne, input, ty),
            SoacOp::Redomap {
                op,
                reduce_op,
                ne,
                inputs,
                ..
            } => self.convert_soac_redomap(op, reduce_op, ne, inputs, ty),
            SoacOp::Scan { op, ne, input } => self.convert_soac_scan(op, ne, input, ty),
            SoacOp::Filter { pred, input } => self.convert_soac_filter(pred, input, ty),
            SoacOp::Scatter { .. } => Err(ConvertError::Unsupported("SOAC scatter".into())),
            SoacOp::ReduceByIndex { .. } => Err(ConvertError::Unsupported("SOAC reduce_by_index".into())),
        }
    }

    fn lambda_fn_name(&self, lam: &Lambda) -> Result<String, ConvertError> {
        match &lam.body.kind {
            TermKind::Var(sym) => Ok(self.symbols.get(*sym).expect("BUG: symbol not in table").clone()),
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
        let result_nid = self.graph.alloc_side_effect_result(ty);
        let effect_in = EffectToken(0);
        let effect_out = self.alloc_effect();
        self.graph.skeleton.blocks[self.current_block].side_effects.push(SideEffect {
            kind: SideEffectKind::Pending(soac),
            operand_nodes: operands,
            result: Some(result_nid),
            effects: Some((effect_in, effect_out)),
        });
        result_nid
    }

    fn convert_soac_map(
        &mut self,
        lam: &Lambda,
        inputs: &[ArrayExpr],
        result_ty: Type<TypeName>,
    ) -> Result<NodeId, ConvertError> {
        let f_name = self.lambda_fn_name(lam)?;
        let capture_nids: Vec<NodeId> =
            lam.captures.iter().map(|(_, _, t)| self.convert_term(t)).collect::<Result<_, _>>()?;
        let input_arr_types: Vec<Type<TypeName>> =
            inputs.iter().map(|ae| self.array_expr_type(ae)).collect();
        let input_nids: Vec<NodeId> =
            inputs.iter().map(|ae| self.convert_array_expr_value(ae)).collect::<Result<_, _>>()?;
        let input_elem_types: Vec<Type<TypeName>> =
            inputs.iter().map(|ae| self.array_expr_elem_type(ae)).collect();
        let output_elem_ty = if result_ty.is_array() {
            result_ty.elem_type().expect("Array has elem").clone()
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
            },
            operands,
            result_ty,
        ))
    }

    fn convert_soac_reduce(
        &mut self,
        op: &Lambda,
        ne: &Term,
        input: &ArrayExpr,
        result_ty: Type<TypeName>,
    ) -> Result<NodeId, ConvertError> {
        let op_name = self.lambda_fn_name(op)?;
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
        op: &Lambda,
        reduce_op: &Lambda,
        ne: &Term,
        inputs: &[ArrayExpr],
        result_ty: Type<TypeName>,
    ) -> Result<NodeId, ConvertError> {
        let op_name = self.lambda_fn_name(op)?;
        let reduce_func_name = self.lambda_fn_name(reduce_op)?;
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
        op: &Lambda,
        ne: &Term,
        input: &ArrayExpr,
        result_ty: Type<TypeName>,
    ) -> Result<NodeId, ConvertError> {
        let op_name = self.lambda_fn_name(op)?;
        let capture_nids: Vec<NodeId> =
            op.captures.iter().map(|(_, _, t)| self.convert_term(t)).collect::<Result<_, _>>()?;
        let elem_ty = self.array_expr_elem_type(input);
        let arr_ty = self.array_expr_type(input);
        let arr_nid = self.convert_array_expr_value(input)?;
        let init_nid = self.convert_term(ne)?;

        let mut operands: SmallVec<[NodeId; 4]> = smallvec![arr_nid, init_nid];
        operands.extend(capture_nids.iter().copied());

        Ok(self.emit_soac(
            PendingSoac::Scan {
                func: op_name,
                input_array_type: arr_ty,
                input_elem_type: elem_ty,
            },
            operands,
            result_ty,
        ))
    }

    fn convert_soac_filter(
        &mut self,
        pred: &Lambda,
        input: &ArrayExpr,
        result_ty: Type<TypeName>,
    ) -> Result<NodeId, ConvertError> {
        let pred_name = self.lambda_fn_name(pred)?;
        let capture_nids: Vec<NodeId> =
            pred.captures.iter().map(|(_, _, t)| self.convert_term(t)).collect::<Result<_, _>>()?;
        let arr_nid = self.convert_array_expr_value(input)?;
        let pred_ref = self.graph.intern_pure(
            PureOp::Global(pred_name),
            smallvec![],
            Type::Constructed(TypeName::Unit, vec![]),
        );

        let mut operands: SmallVec<[NodeId; 4]> = smallvec![pred_ref, arr_nid];
        operands.extend(capture_nids.iter().copied());

        // Filter is not a SOAC that soac_expand handles; emit it as a regular
        // effectful Intrinsic side-effect. (Upstream compilation typically
        // lowers Filter before reaching here, but we keep the fallback for
        // completeness.)
        let dummy_vrefs: Vec<ValueRef> =
            (0..operands.len()).map(|_| ValueRef::Ssa(Default::default())).collect();
        let result_nid = self.graph.alloc_side_effect_result(result_ty);
        let effect_in = EffectToken(0);
        let effect_out = self.alloc_effect();
        self.graph.skeleton.blocks[self.current_block].side_effects.push(SideEffect {
            kind: SideEffectKind::Inst(InstKind::Intrinsic {
                name: "_w_intrinsic_filter".into(),
                args: dummy_vrefs,
            }),
            operand_nodes: operands,
            result: Some(result_nid),
            effects: Some((effect_in, effect_out)),
        });
        Ok(result_nid)
    }

    // ========================================================================
    // ArrayExpr
    // ========================================================================

    fn convert_array_expr(&mut self, ae: &ArrayExpr, ty: Type<TypeName>) -> Result<NodeId, ConvertError> {
        match ae {
            ArrayExpr::Ref(term) => self.convert_term(term),
            ArrayExpr::Zip(_) => panic!("ArrayExpr::Zip should have been eliminated by soa::normalize"),
            ArrayExpr::Soac(op) => self.convert_soac(op, ty),
            ArrayExpr::Generate { .. } => Err(ConvertError::Unsupported("ArrayExpr::Generate".into())),
            ArrayExpr::Literal(terms) => {
                let operands: SmallVec<[NodeId; 4]> =
                    terms.iter().map(|t| self.convert_term(t)).collect::<Result<_, _>>()?;
                let n = operands.len();
                Ok(self.graph.intern_pure(PureOp::ArrayLit(n), operands, ty))
            }
            ArrayExpr::Range { start, len } => {
                let start_nid = self.convert_term(start)?;
                let len_nid = self.convert_term(len)?;
                Ok(self.graph.intern_pure(
                    PureOp::ArrayRange { has_step: false },
                    smallvec![start_nid, len_nid],
                    ty,
                ))
            }
            ArrayExpr::StorageBuffer {
                set,
                binding,
                offset,
                len,
                elem_ty,
            } => {
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
                    kind: SideEffectKind::Inst(InstKind::StorageView {
                        source: ViewSource::Storage {
                            set: *set,
                            binding: *binding,
                        },
                        offset: ValueRef::Ssa(Default::default()),
                        len: ValueRef::Ssa(Default::default()),
                    }),
                    operand_nodes: smallvec![offset_nid, len_nid],
                    result: Some(result_nid),
                    effects: Some((effect_in, effect_out)),
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
            ArrayExpr::Ref(t) => t.ty.clone(),
            ArrayExpr::Zip(_) => unreachable!("Zip eliminated"),
            ArrayExpr::Soac(_) => Type::Constructed(TypeName::Unit, vec![]),
            ArrayExpr::Generate { elem_ty, .. } => elem_ty.clone(),
            ArrayExpr::Literal(_) => Type::Constructed(TypeName::Unit, vec![]),
            ArrayExpr::Range { start, .. } => Type::Constructed(
                TypeName::Array,
                vec![
                    start.ty.clone(),
                    Type::Constructed(TypeName::SizePlaceholder, vec![]),
                    Type::Constructed(TypeName::ArrayVariantVirtual, vec![]),
                ],
            ),
            ArrayExpr::StorageBuffer { elem_ty, .. } => Type::Constructed(
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
                _ => t.ty.clone(),
            },
            ArrayExpr::Zip(_) => unreachable!("Zip eliminated"),
            ArrayExpr::Soac(_) => Type::Constructed(TypeName::Unit, vec![]),
            ArrayExpr::Generate { elem_ty, .. } => elem_ty.clone(),
            ArrayExpr::Literal(terms) => {
                terms.first().map(|t| t.ty.clone()).unwrap_or(Type::Constructed(TypeName::Unit, vec![]))
            }
            ArrayExpr::Range { start, .. } => start.ty.clone(),
            ArrayExpr::StorageBuffer { elem_ty, .. } => elem_ty.clone(),
        }
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Check whether a FuncBody contains only purely constant instructions.
fn is_purely_constant_body(body: &FuncBody) -> bool {
    body.inner.insts.values().all(|inst| {
        matches!(
            &inst.data,
            InstKind::Int(_)
                | InstKind::Float(_)
                | InstKind::Bool(_)
                | InstKind::Unit
                | InstKind::String(_)
                | InstKind::Tuple(_)
                | InstKind::Vector(_)
                | InstKind::Matrix(_)
                | InstKind::ArrayLit { .. }
                | InstKind::Global(_)
        )
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

/// Check if a type is an unsized array (runtime-sized storage buffer).
fn is_unsized_array(ty: &Type<TypeName>) -> bool {
    ty.array_size().map(|s| matches!(s, Type::Variable(_))).unwrap_or(false)
}

/// Extract an IO decoration (builtin or location attribute) from a pattern.
fn extract_io_decoration(pattern: &crate::ast::Pattern) -> Option<crate::ssa::types::IoDecoration> {
    use crate::ast;
    use crate::ssa::types::IoDecoration;
    match &pattern.kind {
        ast::PatternKind::Attributed(attrs, inner) => {
            for attr in attrs {
                match attr {
                    ast::Attribute::BuiltIn(builtin) => return Some(IoDecoration::BuiltIn(*builtin)),
                    ast::Attribute::Location(loc) => return Some(IoDecoration::Location(*loc)),
                    _ => {}
                }
            }
            extract_io_decoration(inner)
        }
        ast::PatternKind::Typed(inner, _) => extract_io_decoration(inner),
        _ => None,
    }
}

/// Extract a `#[size_hint(N)]` attribute from a pattern.
fn extract_size_hint(pattern: &crate::ast::Pattern) -> Option<u32> {
    use crate::ast;
    match &pattern.kind {
        ast::PatternKind::Attributed(attrs, inner) => {
            for attr in attrs {
                if let ast::Attribute::SizeHint(n) = attr {
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
fn convert_to_io_decoration(attr: &crate::ast::Attribute) -> Option<crate::ssa::types::IoDecoration> {
    use crate::ast;
    use crate::ssa::types::IoDecoration;
    match attr {
        ast::Attribute::BuiltIn(b) => Some(IoDecoration::BuiltIn(*b)),
        ast::Attribute::Location(l) => Some(IoDecoration::Location(*l)),
        _ => None,
    }
}

/// Build entry outputs from an AST `EntryDecl`.
/// For compute shaders, non-unit outputs get sequential storage bindings starting at `binding_start`.
fn build_entry_outputs(
    entry: &crate::ast::EntryDecl,
    ret_type: &Type<TypeName>,
    is_compute: bool,
    binding_start: u32,
) -> Vec<crate::ssa::types::EntryOutput> {
    use crate::ssa::types::EntryOutput;
    let mut binding_num = binding_start;
    let mut storage_binding_for = |ty: &Type<TypeName>, is_compute: bool| -> Option<(u32, u32)> {
        if is_compute && !matches!(ty, Type::Constructed(TypeName::Unit, _)) {
            let b = (0, binding_num);
            binding_num += 1;
            Some(b)
        } else {
            None
        }
    };

    if entry.outputs.iter().all(|o| o.attribute.is_none()) && entry.outputs.len() == 1 {
        if !matches!(ret_type, Type::Constructed(TypeName::Unit, _)) {
            vec![EntryOutput {
                ty: ret_type.clone(),
                decoration: None,
                storage_binding: storage_binding_for(ret_type, is_compute),
            }]
        } else {
            vec![]
        }
    } else if let Type::Constructed(TypeName::Tuple(_), component_types) = ret_type {
        entry
            .outputs
            .iter()
            .zip(component_types.iter())
            .map(|(output, ty)| EntryOutput {
                ty: ty.clone(),
                decoration: output.attribute.as_ref().and_then(convert_to_io_decoration),
                storage_binding: storage_binding_for(ty, is_compute),
            })
            .collect()
    } else {
        vec![EntryOutput {
            ty: ret_type.clone(),
            decoration: entry
                .outputs
                .first()
                .and_then(|o| o.attribute.as_ref())
                .and_then(convert_to_io_decoration),
            storage_binding: storage_binding_for(ret_type, is_compute),
        }]
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Compile a source string through the full TLC pipeline, then convert
    /// through the full EGIR chain (`from_tlc → expand_soacs → optimize_skeleton
    /// → elaborate`) to a `Program`. No `materialize` — tests don't exercise
    /// SPIR-V-specific dynamic-index rewrites.
    fn compile_via_egir(src: &str) -> Program {
        let mut frontend = crate::cached_frontend();
        let parsed = crate::Compiler::parse(src, &mut frontend.node_counter).expect("Parsing failed");
        let alias_checked = parsed
            .desugar(&mut frontend.node_counter)
            .expect("Desugaring failed")
            .resolve(&mut frontend.module_manager)
            .expect("Name resolution failed")
            .fold_ast_constants()
            .type_check(&mut frontend.module_manager, &mut frontend.schemes)
            .expect("Type checking failed")
            .alias_check()
            .expect("Alias checking failed");

        let tlc = alias_checked
            .to_tlc(&frontend.schemes, &frontend.module_manager)
            .partial_eval()
            .normalize_soacs()
            .fuse_maps()
            .defunctionalize()
            .monomorphize()
            .buffer_specialize()
            .fold_generated_lambdas()
            .inline_small()
            .parallelize_soacs()
            .filter_reachable();

        convert_program(&tlc.tlc, PipelineDescriptor::default())
            .expect("egir::from_tlc conversion failed")
            .expand_soacs()
            .optimize_skeleton()
            .elaborate()
            .ssa
    }

    use crate::ast::Span;
    use crate::tlc::TermIdSource;

    fn i32_ty() -> Type<TypeName> {
        Type::Constructed(TypeName::Int(32), vec![])
    }

    fn mk_term(ty: Type<TypeName>, kind: TermKind) -> Term {
        Term {
            id: TermIdSource::new().next_id(),
            ty,
            span: Span::dummy(),
            kind,
        }
    }

    /// Build a minimal TLC def and convert it via EGraph.
    fn convert_simple_def(body: Term, params: Vec<(SymbolId, Type<TypeName>)>) -> FuncBody {
        let symbols = crate::SymbolTable::new();
        let top_level = HashMap::new();
        let constants_by_name = HashMap::new();
        let pure_constants = HashSet::new();

        let ret_ty = body.ty.clone();
        let param_info: Vec<(Type<TypeName>, String)> =
            params.iter().enumerate().map(|(i, (_, ty))| (ty.clone(), format!("p{}", i))).collect();

        let mut converter = Converter::new(&top_level, &constants_by_name, &symbols, pure_constants);
        for (i, (sym, ty)) in params.iter().enumerate() {
            let nid = converter.graph.add_func_param(i, ty.clone());
            converter.locals.insert(*sym, nid);
        }
        let result = converter.convert_term(&body).expect("conversion failed");
        converter.set_return(Some(result));
        converter.elaborate_to_funcbody(&param_info, ret_ty).expect("elaboration failed")
    }

    #[test]
    fn test_int_literal_roundtrip() {
        let body = mk_term(i32_ty(), TermKind::IntLit("42".into()));
        let func = convert_simple_def(body, vec![]);
        let entry = func.get_block(func.entry_block());
        // Should have one Int instruction.
        assert!(
            entry
                .insts
                .iter()
                .any(|&iid| { matches!(&func.get_inst(iid).data, InstKind::Int(s) if s == "42") })
        );
    }

    #[test]
    fn test_add_roundtrip() {
        let mut symbols = crate::SymbolTable::new();
        let a_sym = symbols.alloc("a".into());
        let b_sym = symbols.alloc("b".into());

        // Build: a + b
        let a_var = mk_term(i32_ty(), TermKind::Var(a_sym));
        let b_var = mk_term(i32_ty(), TermKind::Var(b_sym));
        let add_op = mk_term(
            i32_ty(), // simplified — real type would be arrow
            TermKind::BinOp(crate::ast::BinaryOp { op: "+".into() }),
        );
        let app = mk_term(
            i32_ty(),
            TermKind::App {
                func: Box::new(add_op),
                args: vec![a_var, b_var],
            },
        );

        let top_level = HashMap::new();
        let constants_by_name = HashMap::new();
        let pure_constants = HashSet::new();

        let mut converter = Converter::new(&top_level, &constants_by_name, &symbols, pure_constants);
        let a_nid = converter.graph.add_func_param(0, i32_ty());
        converter.locals.insert(a_sym, a_nid);
        let b_nid = converter.graph.add_func_param(1, i32_ty());
        converter.locals.insert(b_sym, b_nid);

        let result = converter.convert_term(&app).expect("conversion failed");
        converter.set_return(Some(result));

        let params = vec![(i32_ty(), "a".into()), (i32_ty(), "b".into())];
        let func = converter.elaborate_to_funcbody(&params, i32_ty()).expect("elaboration failed");

        let entry = func.get_block(func.entry_block());
        // Should have a BinOp(+) instruction.
        assert!(
            entry.insts.iter().any(|&iid| {
                matches!(&func.get_inst(iid).data, InstKind::BinOp { op, .. } if op == "+")
            })
        );
    }

    #[test]
    fn test_gvn_via_let() {
        // let x = 42 in let y = 42 in (x, y)
        // GVN should deduplicate the two 42 constants into a single node.
        // (A `+` would be constant-folded to `84`, erasing the evidence.)
        use polytype::Type;
        let pair_ty = Type::Constructed(TypeName::Tuple(2), vec![i32_ty(), i32_ty()]);

        let lit42 = mk_term(i32_ty(), TermKind::IntLit("42".into()));
        let lit42b = mk_term(i32_ty(), TermKind::IntLit("42".into()));

        let mut symbols = crate::SymbolTable::new();
        let x_sym = symbols.alloc("x".into());
        let y_sym = symbols.alloc("y".into());
        let tuple_sym = symbols.alloc("_w_tuple".into());

        let tuple_op = mk_term(
            Type::Constructed(
                TypeName::Arrow,
                vec![
                    i32_ty(),
                    Type::Constructed(TypeName::Arrow, vec![i32_ty(), pair_ty.clone()]),
                ],
            ),
            TermKind::Var(tuple_sym),
        );
        let x_ref = mk_term(i32_ty(), TermKind::Var(x_sym));
        let y_ref = mk_term(i32_ty(), TermKind::Var(y_sym));
        let pair_app = mk_term(
            pair_ty.clone(),
            TermKind::App {
                func: Box::new(tuple_op),
                args: vec![x_ref, y_ref],
            },
        );

        let inner_let = mk_term(
            pair_ty.clone(),
            TermKind::Let {
                name: y_sym,
                name_ty: i32_ty(),
                rhs: Box::new(lit42b),
                body: Box::new(pair_app),
            },
        );
        let outer_let = mk_term(
            pair_ty.clone(),
            TermKind::Let {
                name: x_sym,
                name_ty: i32_ty(),
                rhs: Box::new(lit42),
                body: Box::new(inner_let),
            },
        );

        let top_level = HashMap::new();
        let constants_by_name = HashMap::new();
        let pure_constants = HashSet::new();

        let mut converter = Converter::new(&top_level, &constants_by_name, &symbols, pure_constants);
        let result = converter.convert_term(&outer_let).expect("conversion failed");
        converter.set_return(Some(result));

        let func = converter.elaborate_to_funcbody(&[], pair_ty).expect("elaboration failed");

        let entry = func.get_block(func.entry_block());
        // GVN: should have only ONE Int("42") instruction, not two.
        let const_count = entry
            .insts
            .iter()
            .filter(|&&iid| matches!(&func.get_inst(iid).data, InstKind::Int(s) if s == "42"))
            .count();
        assert_eq!(
            const_count, 1,
            "GVN should deduplicate: found {} copies of 42",
            const_count
        );
    }

    #[test]
    fn test_if_else_roundtrip() {
        // if cond then 1 else 0
        let mut symbols = crate::SymbolTable::new();
        let c_sym = symbols.alloc("c".into());
        let bool_ty = Type::Constructed(TypeName::Bool, vec![]);

        let cond = mk_term(bool_ty.clone(), TermKind::Var(c_sym));
        let then_br = mk_term(i32_ty(), TermKind::IntLit("1".into()));
        let else_br = mk_term(i32_ty(), TermKind::IntLit("0".into()));
        let if_term = mk_term(
            i32_ty(),
            TermKind::If {
                cond: Box::new(cond),
                then_branch: Box::new(then_br),
                else_branch: Box::new(else_br),
            },
        );

        let top_level = HashMap::new();
        let constants_by_name = HashMap::new();
        let pure_constants = HashSet::new();

        let mut converter = Converter::new(&top_level, &constants_by_name, &symbols, pure_constants);
        let c_nid = converter.graph.add_func_param(0, bool_ty);
        converter.locals.insert(c_sym, c_nid);

        let result = converter.convert_term(&if_term).expect("conversion failed");
        converter.set_return(Some(result));

        let params = vec![(Type::Constructed(TypeName::Bool, vec![]), "c".into())];
        let func = converter.elaborate_to_funcbody(&params, i32_ty()).expect("elaboration failed");

        // Should have 4 blocks: entry, then, else, merge
        assert_eq!(func.inner.blocks.len(), 4, "if/else should produce 4 blocks");

        // Entry should end with CondBranch
        let entry = func.get_block(func.entry_block());
        assert!(
            matches!(&entry.term, wyn_ssa::Terminator::CondBranch { .. }),
            "Entry should end with CondBranch, got {:?}",
            entry.term
        );
    }

    // ====================================================================
    // Full pipeline integration tests
    // ====================================================================

    #[test]
    fn test_full_pipeline_simple() {
        let program = compile_via_egir(
            r#"
def add(a: i32, b: i32) i32 = a + b

#[vertex]
entry main() #[builtin(position)] vec4f32 =
    let x = add(1, 2) in
    @[f32.i32(x), 0.0, 0.0, 1.0]
"#,
        );
        // 'add' may be inlined by TLC passes — just verify the program is valid
        assert!(!program.entry_points.is_empty(), "Should have entry points");
    }

    #[test]
    fn test_full_pipeline_if_else() {
        let program = compile_via_egir(
            r#"
def pick(c: bool, a: i32, b: i32) i32 = if c then a else b

#[vertex]
entry main() #[builtin(position)] vec4f32 =
    let x = pick(true, 1, 2) in
    @[f32.i32(x), 0.0, 0.0, 1.0]
"#,
        );
        // 'pick' may be inlined — just verify compilation succeeds
        assert!(!program.entry_points.is_empty(), "Should have entry points");
    }

    #[test]
    fn test_full_pipeline_loop() {
        let program = compile_via_egir(
            r#"
def sum_to(n: i32) i32 =
    loop acc = 0 for i < n do acc + i

#[vertex]
entry main() #[builtin(position)] vec4f32 =
    let x = sum_to(10) in
    @[f32.i32(x), 0.0, 0.0, 1.0]
"#,
        );
        // 'sum_to' may be inlined — just verify compilation succeeds
        assert!(!program.entry_points.is_empty(), "Should have entry points");
    }
}
