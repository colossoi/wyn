//! Whole-program EGIR container + per-body records.
//!
//! These are plain (non-generic) structs. State tracking happens at the
//! public API boundary via the `EgirRaw` / `EgirSoacExpanded` /
//! `EgirMaterialized` / `EgirSkelOptimized` newtypes in `crate::lib`, each
//! of which wraps an `EgirInner`.
//!
//! `EgirInner` carries, for each function and entry point, a per-body
//! `EGraph` + control-headers + alias map, plus program-level metadata
//! (constants, uniforms, storage decls, pipeline descriptor, extern stubs).

use crate::LookupMap;

use polytype::Type;

use crate::ast::{Span, TypeName};
use crate::interface;
use crate::pipeline_descriptor::PipelineDescriptor;
use crate::ssa::types::{
    BlockId, Constant, ControlHeader, EntryInput, EntryOutput, ExecutionModel, Function,
};

use super::types::{EGraph, NodeId};

pub struct EgirFunc {
    pub name: String,
    pub span: Span,
    pub linkage_name: Option<String>,
    pub params: Vec<(Type<TypeName>, String)>,
    pub return_ty: Type<TypeName>,
    pub graph: EGraph,
    pub control_headers: LookupMap<BlockId, ControlHeader>,
    pub aliases: LookupMap<NodeId, NodeId>,
}

impl EgirFunc {
    pub fn new(
        name: String,
        span: Span,
        linkage_name: Option<String>,
        params: Vec<(Type<TypeName>, String)>,
        return_ty: Type<TypeName>,
        graph: EGraph,
        control_headers: LookupMap<BlockId, ControlHeader>,
    ) -> Self {
        EgirFunc {
            name,
            span,
            linkage_name,
            params,
            return_ty,
            graph,
            control_headers,
            aliases: LookupMap::new(),
        }
    }
}

/// One write site for an entry output slot: the block in which the
/// store fires and the value produced there. A slot can have multiple
/// sources when different CFG paths each write it (e.g. both arms of
/// an `If` whose result flows into the slot).
///
/// The `block` is load-bearing for any pass that emits side-effect
/// stores at the producer site — retargeting a `Map`'s destination is
/// metadata-only on the node, but emitting `Store` for a scalar
/// requires knowing the block to insert it into.
#[derive(Debug, Clone, Copy)]
pub struct SlotSource {
    pub block: BlockId,
    pub value: NodeId,
}

#[derive(Clone)]
pub struct EgirEntry {
    pub name: String,
    pub span: Span,
    pub execution_model: ExecutionModel,
    pub inputs: Vec<EntryInput>,
    pub outputs: Vec<EntryOutput>,
    pub storage_bindings: Vec<interface::StorageBindingDecl>,
    pub params: Vec<(Type<TypeName>, String)>,
    pub return_ty: Type<TypeName>,
    pub graph: EGraph,
    pub control_headers: LookupMap<BlockId, ControlHeader>,
    pub aliases: LookupMap<NodeId, NodeId>,
    /// Per-slot list of (producing-block, value) pairs. Indexed by
    /// declared output slot. A slot with one source has `vec![one]`;
    /// a slot written from both arms of an `If` has two. Empty for
    /// unit-returning entries. Phase 1 of the DPS migration: populated
    /// only by code added in later phases; today's `from_tlc` leaves
    /// it untouched.
    pub slot_sources: Vec<Vec<SlotSource>>,
}

impl EgirEntry {
    pub fn new(
        name: String,
        span: Span,
        execution_model: ExecutionModel,
        inputs: Vec<EntryInput>,
        outputs: Vec<EntryOutput>,
        storage_bindings: Vec<interface::StorageBindingDecl>,
        params: Vec<(Type<TypeName>, String)>,
        return_ty: Type<TypeName>,
        graph: EGraph,
        control_headers: LookupMap<BlockId, ControlHeader>,
    ) -> Self {
        EgirEntry {
            name,
            span,
            execution_model,
            inputs,
            outputs,
            storage_bindings,
            params,
            return_ty,
            graph,
            control_headers,
            aliases: LookupMap::new(),
            slot_sources: Vec::new(),
        }
    }
}

/// Whole-program EGIR container. Wrapped by the `EgirRaw` /
/// `EgirSoacExpanded` / `EgirMaterialized` / `EgirSkelOptimized` newtypes at
/// the public-API layer (see `crate::lib`).
pub struct EgirInner {
    pub functions: Vec<EgirFunc>,
    /// Extern function stubs. These don't have a body that flows through EGIR;
    /// they're already `Function` records with a 1-block Unreachable body and
    /// pass straight through.
    pub externs: Vec<Function>,
    pub entry_points: Vec<EgirEntry>,
    pub constants: Vec<Constant>,
    pub pipeline: PipelineDescriptor,
}

impl EgirInner {
    pub fn new(
        functions: Vec<EgirFunc>,
        externs: Vec<Function>,
        entry_points: Vec<EgirEntry>,
        constants: Vec<Constant>,
        pipeline: PipelineDescriptor,
    ) -> Self {
        EgirInner {
            functions,
            externs,
            entry_points,
            constants,
            pipeline,
        }
    }

    /// Convenience: build an EGIR program wrapping a single function body.
    /// Used by the probe path in `from_tlc`.
    pub fn single_function(func: EgirFunc) -> Self {
        Self::new(vec![func], vec![], vec![], vec![], PipelineDescriptor::default())
    }
}
