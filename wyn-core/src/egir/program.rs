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

use std::collections::HashMap;

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
    pub control_headers: HashMap<BlockId, ControlHeader>,
    pub aliases: HashMap<NodeId, NodeId>,
}

impl EgirFunc {
    pub fn new(
        name: String,
        span: Span,
        linkage_name: Option<String>,
        params: Vec<(Type<TypeName>, String)>,
        return_ty: Type<TypeName>,
        graph: EGraph,
        control_headers: HashMap<BlockId, ControlHeader>,
    ) -> Self {
        EgirFunc {
            name,
            span,
            linkage_name,
            params,
            return_ty,
            graph,
            control_headers,
            aliases: HashMap::new(),
        }
    }
}

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
    pub control_headers: HashMap<BlockId, ControlHeader>,
    pub aliases: HashMap<NodeId, NodeId>,
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
        control_headers: HashMap<BlockId, ControlHeader>,
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
            aliases: HashMap::new(),
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
    pub uniforms: Vec<interface::UniformDecl>,
    pub storage: Vec<interface::StorageDecl>,
    pub pipeline: PipelineDescriptor,
}

impl EgirInner {
    pub fn new(
        functions: Vec<EgirFunc>,
        externs: Vec<Function>,
        entry_points: Vec<EgirEntry>,
        constants: Vec<Constant>,
        uniforms: Vec<interface::UniformDecl>,
        storage: Vec<interface::StorageDecl>,
        pipeline: PipelineDescriptor,
    ) -> Self {
        EgirInner {
            functions,
            externs,
            entry_points,
            constants,
            uniforms,
            storage,
            pipeline,
        }
    }

    /// Convenience: build an EGIR program wrapping a single function body.
    /// Used by the probe path in `from_tlc`.
    pub fn single_function(func: EgirFunc) -> Self {
        Self::new(
            vec![func],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            PipelineDescriptor::default(),
        )
    }
}
