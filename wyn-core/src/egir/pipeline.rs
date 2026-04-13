//! Typestate for the whole-program EGIR pass pipeline.
//!
//! Each pass is a method on a concrete typestate alias; the generic
//! `ProgramEgir<S>` is internal scaffolding.
//!
//! ```text
//! EgirRaw
//!   .expand_soacs()       → EgirSoacExpanded    (mandatory)
//!   .materialize()        → EgirMaterialized    (optional — SPIR-V only)
//!   .optimize_skeleton()  → EgirSkelOptimized   (mandatory; callable from SoacExpanded or Materialized)
//!   .elaborate()          → ProgramEgirElaborated  (consumes the chain)
//! ```
//!
//! `ProgramEgir<S>` carries, for each function and entry point, a per-body
//! `EGraph` + control-headers + alias map, plus program-level metadata
//! (constants, uniforms, storage decls, pipeline descriptor, extern stubs).
//! A pass transitions the state marker for every per-body record at once.

use std::collections::HashMap;
use std::marker::PhantomData;

use polytype::Type;

use crate::ast::{self, Span, TypeName};
use crate::pipeline_descriptor::PipelineDescriptor;
use crate::ssa::types::{
    BlockId, Constant, ControlHeader, EntryInput, EntryOutput, EntryPoint, ExecutionModel, Function,
    Program,
};

use super::domtree::{DomTree, SkeletonCfgView};
use super::types::{EGraph, NodeId};
use super::{elaborate, skel_opt, soac_expand};

// ---------------------------------------------------------------------------
// State markers
// ---------------------------------------------------------------------------

pub struct Raw;
pub struct SoacExpanded;
pub struct Materialized;
pub struct SkelOptimized;

// ---------------------------------------------------------------------------
// Per-function / per-entry payload (identical shape across states; the marker
// is phantom).
// ---------------------------------------------------------------------------

pub struct FuncEgir<S> {
    pub name: String,
    pub span: Span,
    pub linkage_name: Option<String>,
    pub params: Vec<(Type<TypeName>, String)>,
    pub return_ty: Type<TypeName>,
    pub graph: EGraph,
    pub control_headers: HashMap<BlockId, ControlHeader>,
    pub aliases: HashMap<NodeId, NodeId>,
    _marker: PhantomData<S>,
}

impl FuncEgir<Raw> {
    pub fn new(
        name: String,
        span: Span,
        linkage_name: Option<String>,
        params: Vec<(Type<TypeName>, String)>,
        return_ty: Type<TypeName>,
        graph: EGraph,
        control_headers: HashMap<BlockId, ControlHeader>,
    ) -> Self {
        FuncEgir {
            name,
            span,
            linkage_name,
            params,
            return_ty,
            graph,
            control_headers,
            aliases: HashMap::new(),
            _marker: PhantomData,
        }
    }
}

impl<S> FuncEgir<S> {
    fn transition<S2>(self) -> FuncEgir<S2> {
        FuncEgir {
            name: self.name,
            span: self.span,
            linkage_name: self.linkage_name,
            params: self.params,
            return_ty: self.return_ty,
            graph: self.graph,
            control_headers: self.control_headers,
            aliases: self.aliases,
            _marker: PhantomData,
        }
    }
}

pub struct EntryEgir<S> {
    pub name: String,
    pub span: Span,
    pub execution_model: ExecutionModel,
    pub inputs: Vec<EntryInput>,
    pub outputs: Vec<EntryOutput>,
    pub params: Vec<(Type<TypeName>, String)>,
    pub return_ty: Type<TypeName>,
    pub graph: EGraph,
    pub control_headers: HashMap<BlockId, ControlHeader>,
    pub aliases: HashMap<NodeId, NodeId>,
    _marker: PhantomData<S>,
}

impl EntryEgir<Raw> {
    pub fn new(
        name: String,
        span: Span,
        execution_model: ExecutionModel,
        inputs: Vec<EntryInput>,
        outputs: Vec<EntryOutput>,
        params: Vec<(Type<TypeName>, String)>,
        return_ty: Type<TypeName>,
        graph: EGraph,
        control_headers: HashMap<BlockId, ControlHeader>,
    ) -> Self {
        EntryEgir {
            name,
            span,
            execution_model,
            inputs,
            outputs,
            params,
            return_ty,
            graph,
            control_headers,
            aliases: HashMap::new(),
            _marker: PhantomData,
        }
    }
}

impl<S> EntryEgir<S> {
    fn transition<S2>(self) -> EntryEgir<S2> {
        EntryEgir {
            name: self.name,
            span: self.span,
            execution_model: self.execution_model,
            inputs: self.inputs,
            outputs: self.outputs,
            params: self.params,
            return_ty: self.return_ty,
            graph: self.graph,
            control_headers: self.control_headers,
            aliases: self.aliases,
            _marker: PhantomData,
        }
    }
}

// ---------------------------------------------------------------------------
// ProgramEgir<S> — the top-level typestate container
// ---------------------------------------------------------------------------

pub struct ProgramEgir<S> {
    pub functions: Vec<FuncEgir<S>>,
    /// Extern function stubs. These don't have a body that flows through EGIR;
    /// they're already `Function` records with a 1-block Unreachable body and
    /// pass straight through.
    pub externs: Vec<Function>,
    pub entry_points: Vec<EntryEgir<S>>,
    pub constants: Vec<Constant>,
    pub uniforms: Vec<ast::UniformDecl>,
    pub storage: Vec<ast::StorageDecl>,
    pub pipeline: PipelineDescriptor,
    _marker: PhantomData<S>,
}

pub type EgirRaw = ProgramEgir<Raw>;
pub type EgirSoacExpanded = ProgramEgir<SoacExpanded>;
pub type EgirMaterialized = ProgramEgir<Materialized>;
pub type EgirSkelOptimized = ProgramEgir<SkelOptimized>;

impl ProgramEgir<Raw> {
    pub fn new(
        functions: Vec<FuncEgir<Raw>>,
        externs: Vec<Function>,
        entry_points: Vec<EntryEgir<Raw>>,
        constants: Vec<Constant>,
        uniforms: Vec<ast::UniformDecl>,
        storage: Vec<ast::StorageDecl>,
        pipeline: PipelineDescriptor,
    ) -> Self {
        ProgramEgir {
            functions,
            externs,
            entry_points,
            constants,
            uniforms,
            storage,
            pipeline,
            _marker: PhantomData,
        }
    }

    /// Convenience: build an EGIR program wrapping a single function body and
    /// nothing else. Used by the internal probe path in `from_tlc` and by
    /// `egir::optimize_func` to round-trip a single SSA body through the
    /// full typestate chain.
    pub fn single_function(func: FuncEgir<Raw>) -> Self {
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

    pub fn expand_soacs(self) -> EgirSoacExpanded {
        let functions = self
            .functions
            .into_iter()
            .map(|mut f| {
                soac_expand::expand_soacs(&mut f.graph, &mut f.control_headers);
                f.transition::<SoacExpanded>()
            })
            .collect();
        let entry_points = self
            .entry_points
            .into_iter()
            .map(|mut e| {
                soac_expand::expand_soacs(&mut e.graph, &mut e.control_headers);
                e.transition::<SoacExpanded>()
            })
            .collect();
        ProgramEgir {
            functions,
            externs: self.externs,
            entry_points,
            constants: self.constants,
            uniforms: self.uniforms,
            storage: self.storage,
            pipeline: self.pipeline,
            _marker: PhantomData,
        }
    }
}

impl ProgramEgir<SoacExpanded> {
    pub fn materialize(self) -> EgirMaterialized {
        let functions = self
            .functions
            .into_iter()
            .map(|mut f| {
                super::materialize::run(&mut f.graph);
                f.transition::<Materialized>()
            })
            .collect();
        let entry_points = self
            .entry_points
            .into_iter()
            .map(|mut e| {
                super::materialize::run(&mut e.graph);
                e.transition::<Materialized>()
            })
            .collect();
        ProgramEgir {
            functions,
            externs: self.externs,
            entry_points,
            constants: self.constants,
            uniforms: self.uniforms,
            storage: self.storage,
            pipeline: self.pipeline,
            _marker: PhantomData,
        }
    }

    pub fn optimize_skeleton(self) -> EgirSkelOptimized {
        skel_opt_all(self)
    }
}

impl ProgramEgir<Materialized> {
    pub fn optimize_skeleton(self) -> EgirSkelOptimized {
        skel_opt_all(self)
    }
}

fn skel_opt_all<S>(p: ProgramEgir<S>) -> EgirSkelOptimized {
    let functions = p
        .functions
        .into_iter()
        .map(|mut f| {
            let new_aliases = skel_opt::optimize_skeleton(&mut f.graph);
            f.aliases.extend(new_aliases);
            f.transition::<SkelOptimized>()
        })
        .collect();
    let entry_points = p
        .entry_points
        .into_iter()
        .map(|mut e| {
            let new_aliases = skel_opt::optimize_skeleton(&mut e.graph);
            e.aliases.extend(new_aliases);
            e.transition::<SkelOptimized>()
        })
        .collect();
    ProgramEgir {
        functions,
        externs: p.externs,
        entry_points,
        constants: p.constants,
        uniforms: p.uniforms,
        storage: p.storage,
        pipeline: p.pipeline,
        _marker: PhantomData,
    }
}

// ---------------------------------------------------------------------------
// Elaborate → SSA
// ---------------------------------------------------------------------------

impl ProgramEgir<SkelOptimized> {
    /// Terminal step: lower each per-body e-graph to SSA and assemble the
    /// final `SsaConverted`.
    pub fn elaborate(self) -> crate::SsaConverted {
        let functions: Vec<Function> = self
            .functions
            .into_iter()
            .map(|f| {
                let body = elaborate_one(f.graph, &f.control_headers, &f.aliases, &f.params, f.return_ty);
                Function {
                    name: f.name,
                    body,
                    span: f.span,
                    linkage_name: f.linkage_name,
                }
            })
            .chain(self.externs.into_iter())
            .collect();

        let entry_points: Vec<EntryPoint> = self
            .entry_points
            .into_iter()
            .map(|e| {
                let body = elaborate_one(e.graph, &e.control_headers, &e.aliases, &e.params, e.return_ty);
                EntryPoint {
                    name: e.name,
                    body,
                    execution_model: e.execution_model,
                    inputs: e.inputs,
                    outputs: e.outputs,
                    span: e.span,
                }
            })
            .collect();

        crate::SsaConverted {
            ssa: Program {
                functions,
                entry_points,
                constants: self.constants,
                uniforms: self.uniforms,
                storage: self.storage,
            },
            pipeline: self.pipeline,
        }
    }
}

fn elaborate_one(
    graph: EGraph,
    control_headers: &HashMap<BlockId, ControlHeader>,
    aliases: &HashMap<NodeId, NodeId>,
    params: &[(Type<TypeName>, String)],
    return_ty: Type<TypeName>,
) -> crate::ssa::types::FuncBody {
    let skel_domtree = DomTree::build(&SkeletonCfgView {
        skeleton: &graph.skeleton,
    });
    let identity_map: HashMap<BlockId, BlockId> = graph.skeleton.blocks.keys().map(|b| (b, b)).collect();
    elaborate::elaborate(
        &graph,
        &skel_domtree,
        params,
        return_ty,
        control_headers,
        &identity_map,
        aliases,
    )
}
