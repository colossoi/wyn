//! Pass states own only the data available at that checkpoint.
use super::blocks::{BlockData, BodyData, BufferData, DispatchData, GridData};
use crate::egglog::data::{
    BlockId, BodyId, BufferId, DispatchId, GridId, Ir, OutputData, OutputId, PlacementData, PlacementId,
};
use crate::kernel_graph::PhysicalKernelGraph;
use egglog_engine::ast::Command;
use egglog_engine::EGraph;
use wyn_base::IdArena;

/// A compiler checkpoint. Only passes can construct or change its state.
///
/// Fusion cannot run after expressions have been inserted:
/// ```compile_fail
/// use wyn_core::egglog::{fuse, Expressions, Program};
/// fn invalid(program: Program<Expressions>) { let _ = fuse(program); }
/// ```
/// Expression insertion requires completed fusion:
/// ```compile_fail
/// use wyn_core::egglog::{insert_expressions, Imported, Program};
/// fn invalid(program: Program<Imported>) { let _ = insert_expressions(program); }
/// ```
/// Scheduling requires explicit placement, not just simplification:
/// ```compile_fail
/// use wyn_core::egglog::{schedule, Simplified, Program};
/// fn invalid(program: Program<Simplified>) { let _ = schedule(program, wyn_core::PipelineTopologyPolicy::AllowGenerated); }
/// ```
/// SSA lowering requires the scheduled blocks and resources:
/// ```compile_fail
/// use wyn_core::{egglog::{to_ssa, Placed, Program}, CodegenTarget};
/// fn invalid(program: Program<Placed>) {
///     let _ = to_ssa(&program, CodegenTarget::Wgsl);
/// }
/// ```
#[derive(Clone, Debug)]
pub struct Program<State> {
    pub(super) ir: Ir,
    pub(super) state: State,
}

/// Normalized source IR and the facts needed for fusion.
#[derive(Clone, Debug)]
pub struct Imported {
    pub(super) facts: Vec<Command>,
}
/// Fusion is complete and its decisions have been materialized in the IR.
#[derive(Clone, Debug)]
pub struct Fused;
/// Native expression graph and roots prepared for scalar simplification.
#[derive(Clone)]
pub struct Expressions {
    pub(super) graph: EGraph,
}
/// Extracted scalar expressions after arithmetic EqSat.
#[derive(Clone, Debug)]
pub struct Simplified;
/// Specialized callbacks and explicit evaluation sites for hoisted expressions.
#[derive(Clone, Debug)]
pub struct Placed {
    pub(super) placements: IdArena<PlacementId, PlacementData>,
}
/// Final functions, blocks, resource allocations, and dispatches for SSA lowering.
#[derive(Clone, Debug, Default)]
pub struct Scheduled {
    pub(super) abi: super::abi::Abi,
    pub(super) physical_kernels: PhysicalKernelGraph,
    pub(super) unsupported_host: Option<BlockId>,
    pub(super) materialized: std::collections::BTreeMap<super::OperationId, super::Value>,
    pub(super) execution: super::execution::Execution,
    pub(super) placements: IdArena<PlacementId, PlacementData>,
    pub(super) outputs: IdArena<OutputId, OutputData>,
    pub(super) blocks: IdArena<BlockId, BlockData>,
    pub(super) bodies: IdArena<BodyId, BodyData>,
    pub(super) buffers: IdArena<BufferId, BufferData>,
    pub(super) dispatches: IdArena<DispatchId, DispatchData>,
    pub(super) grids: IdArena<GridId, GridData>,
}

impl<S> std::ops::Deref for Program<S> {
    type Target = Ir;
    fn deref(&self) -> &Ir {
        &self.ir
    }
}
