//! Target-independent dead-operation elimination and compositional fusion.
//! Fusion contracts an owned dependency graph to a fixed point before emitting
//! EGIR. Snapshots are rebuilt after application and dead-operation elimination.

/// Semantic EGIR after dead-operation elimination and fusion reach a fixpoint.
#[derive(Debug, Clone, Copy)]
pub enum SemanticOperationsOptimizedTag {}
pub type SemanticOperationsOptimized = super::program::Program<
    SemanticOperationsOptimizedTag,
    super::ir::ProgramFamily<
        super::types::Semantic,
        super::program::NoStorageDeclaration,
        super::ir::RealizedOutputRoute,
        super::program::SemanticProgramData,
    >,
    super::program::RewriteGlobal,
>;

/// Semantic EGIR after target-independent graph optimization and stage lifting.
#[derive(Debug, Clone, Copy)]
pub enum OptimizedTag {}
pub type Optimized = super::program::Program<
    OptimizedTag,
    super::ir::ProgramFamily<
        super::types::Semantic,
        super::program::NoStorageDeclaration,
        super::ir::RealizedOutputRoute,
        super::program::SemanticProgramData,
    >,
    super::program::RewriteGlobal,
>;

use super::ir::BodySite;
use super::program::SemanticOpId;
use super::reify::Segmented;
use super::soac::screma;
use super::types::{
    EGraph, GraphResource, ResourceAccess, Semantic, SideEffectKind, Soac, SoacEffect, ValueId,
};
use crate::egir::analysis::GraphAnalysis;
use crate::egir::soac::SegmentedMetadata;
use crate::error::CompilerError;
use crate::flow::BlockId;
use crate::LookupMap;

#[cfg(test)]
#[path = "semantic_opt_tests.rs"]
mod semantic_opt_tests;

/// One structural relationship observed while semantic optimization rewrote a
/// single operation or eliminated dead work. A fusion typically has multiple
/// `before` identities and one `after` identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticOptimizationRelation {
    pub before: Vec<SemanticOpId>,
    pub after: Vec<SemanticOpId>,
}

/// Compiler-authored provenance for consumers that need to relate semantic
/// operations across the optimization boundary.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SemanticOptimizationTrace {
    pub relations: Vec<SemanticOptimizationRelation>,
}

/// Eliminate dead segmented operations and fuse legal operations to a single
/// shared fixpoint.
pub fn optimize_semantic_operations(
    program: Segmented,
) -> std::result::Result<SemanticOperationsOptimized, CompilerError> {
    Ok(optimize_semantic_operations_with_trace(program)?.0)
}

/// The fixpoint transition with compiler-authored rewrite provenance.
pub fn optimize_semantic_operations_with_trace(
    program: Segmented,
) -> std::result::Result<(SemanticOperationsOptimized, SemanticOptimizationTrace), CompilerError> {
    let mut trace = SemanticOptimizationTrace::default();
    let mut program = program;

    loop {
        let (rewritten, changed, step_trace) = eliminate_dead_semantic_operations(program);
        program = rewritten;
        trace.extend(step_trace);
        if changed {
            continue;
        }

        let (rewritten, changed, step_trace) = super::fusion::run(program, None)
            .map_err(|error| CompilerError::Internal(error.to_string()))?;
        program = rewritten;
        trace.extend(step_trace);
        if changed {
            continue;
        }
        break;
    }

    Ok((program.retag(), trace))
}

/// Apply one whole-program dead semantic-operation elimination step.
///
/// The returned boolean reports whether the program changed. Callers driving
/// the production fixpoint must rebuild all semantic analyses before invoking
/// another semantic optimization sub-pass.
pub fn eliminate_dead_semantic_operations(
    program: Segmented,
) -> (Segmented, bool, SemanticOptimizationTrace) {
    let Some(patch) = analyze_dead_seg_ops(&program) else {
        return (program, false, SemanticOptimizationTrace::default());
    };
    let mut before = patch
        .bodies
        .iter()
        .flat_map(|(site, blocks)| {
            let graph = program.body_graph(*site).expect("dead-operation patch body");
            blocks.iter().flat_map(move |(block, indices)| {
                indices.iter().filter_map(move |index| {
                    match graph.skeleton.blocks[*block].side_effects[*index].kind() {
                        SideEffectKind::Soac(SoacEffect(id, _)) => Some(*id),
                        _ => None,
                    }
                })
            })
        })
        .collect::<Vec<_>>();
    before.sort_unstable();
    before.dedup();
    let program = apply_dead_seg_ops(program, patch);
    let trace = SemanticOptimizationTrace {
        relations: vec![SemanticOptimizationRelation {
            before,
            after: vec![],
        }],
    };
    (program, true, trace)
}

/// Apply at most one legal semantic fusion rewrite.
///
/// Builds a source snapshot and runs the shared planner with a one-action limit.
pub fn fuse_semantic_operations(
    program: Segmented,
) -> std::result::Result<(Segmented, bool, SemanticOptimizationTrace), CompilerError> {
    super::fusion::run(program, Some(1)).map_err(|error| CompilerError::Internal(error.to_string()))
}

/// Lift values that are uniform at their execution stage, then validate the
/// final semantic dependency graph in debug builds.
pub fn lift_stage_uniform_values(program: SemanticOperationsOptimized) -> Optimized {
    let program: Segmented = program.retag();
    let program = super::stage_lift::lift_stage_uniform_values(program)
        .expect("stage-uniform region lifting must preserve semantic EGIR");

    if cfg!(debug_assertions) {
        if let Err(error) = super::semantic_graph::verify(&program) {
            panic!("semantic optimization produced invalid EGIR: {error}");
        }
    }
    program.retag()
}

/// Cross the semantic optimization boundary without hoisting values into
/// compiler-created stages. Direct WGSL uses this path to preserve authored
/// graphics-stage boundaries.
pub fn preserve_authored_stage_boundaries(program: SemanticOperationsOptimized) -> Optimized {
    let program: Segmented = program.retag();
    if cfg!(debug_assertions) {
        if let Err(error) = super::semantic_graph::verify(&program) {
            panic!("semantic optimization produced invalid EGIR: {error}");
        }
    }
    program.retag()
}

/// Apply the requested pipeline-topology policy at the semantic stage
/// boundary. This keeps the choice between profitable stage lifting and exact
/// authored topology independent of any backend or command-line spelling.
pub fn apply_pipeline_topology_policy(
    program: SemanticOperationsOptimized,
    topology: crate::PipelineTopologyPolicy,
) -> Optimized {
    match topology {
        crate::PipelineTopologyPolicy::AllowGenerated => lift_stage_uniform_values(program),
        crate::PipelineTopologyPolicy::AuthoredOnly => preserve_authored_stage_boundaries(program),
    }
}

impl SemanticOptimizationTrace {
    fn extend(&mut self, mut other: Self) {
        self.relations.append(&mut other.relations);
    }
}

/// Remove SegOps (of any placement) that write no observable resource and whose
/// result is unused. The outer fixpoint re-runs it so producer chains collapse.
type DeadGraphPatch = LookupMap<BlockId, Vec<usize>>;

struct DeadSegOpsPatch {
    bodies: LookupMap<BodySite, DeadGraphPatch>,
}

fn analyze_dead_seg_ops(inner: &Segmented) -> Option<DeadSegOpsPatch> {
    let mut bodies = LookupMap::new();
    for (index, entry) in inner.entry_points.iter().enumerate() {
        let patch = dead_seg_ops_in_graph(
            &entry.graph,
            entry.routes().flat_map(|route| route.referenced_values()),
        );
        if !patch.is_empty() {
            bodies.insert(BodySite::Entry(index), patch);
        }
    }
    for function in &inner.functions {
        let patch = dead_seg_ops_in_graph(&function.graph, []);
        if !patch.is_empty() {
            bodies.insert(BodySite::Function(function.region), patch);
        }
    }
    (!bodies.is_empty()).then_some(DeadSegOpsPatch { bodies })
}

fn apply_dead_seg_ops(inner: Segmented, mut patch: DeadSegOpsPatch) -> Segmented {
    let rebuilt = inner.map_graphs(|site, mut graph| {
        let Some(blocks) = patch.bodies.remove(&site) else {
            return graph;
        };
        for (block, mut effects) in blocks {
            effects.sort_unstable();
            for effect in effects.into_iter().rev() {
                graph.skeleton.blocks[block].side_effects.remove(effect);
            }
        }
        graph
    });
    assert!(
        patch.bodies.is_empty(),
        "dead-SegOp patches targeted bodies absent from the rebuilt program"
    );
    rebuilt
}

fn dead_seg_ops_in_graph<R: GraphResource>(
    graph: &EGraph<Semantic<R>>,
    external_roots: impl IntoIterator<Item = ValueId>,
) -> DeadGraphPatch {
    let observable = |effect: &super::types::SideEffect<Semantic<R>>| match &effect.kind {
        SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) => match op.semantic_state() {
            screma::SemanticState::Segmented(SegmentedMetadata {
                resources,
                output_slots,
                ..
            }) => !output_slots.is_empty() || resources.iter().any(|r| r.access != ResourceAccess::Read),
            screma::SemanticState::Serial => true,
        },
        _ => true,
    };
    let demands = graph.skeleton.blocks.iter().flat_map(|(block, body)| {
        body.side_effects.iter().enumerate().filter_map(move |(index, effect)| {
            observable(effect).then_some(super::types::SideEffectSite { block, index })
        })
    });
    let graph_analysis = GraphAnalysis::new(graph);
    let facts = graph_analysis.slice();
    let slice = facts
        .select_with_demands(
            external_roots,
            demands,
            graph.skeleton.blocks.values().flat_map(|block| block.term.referenced_nodes()),
            [],
            |_| true,
            |_| false,
            &[],
        )
        .expect("DCE requires valid EGIR");
    graph
        .skeleton
        .blocks
        .iter()
        .filter_map(|(block, body)| {
            let dead: Vec<_> = (0..body.side_effects.len())
                .filter(|index| {
                    !slice.operations().contains(&super::types::SideEffectSite { block, index: *index })
                })
                .collect();
            (!dead.is_empty()).then_some((block, dead))
        })
        .collect()
}

pub(super) fn eliminate_dead_seg_ops_in_graph<R: GraphResource>(
    graph: &mut EGraph<Semantic<R>>,
    external_roots: impl IntoIterator<Item = ValueId>,
) -> bool {
    let mut patch = dead_seg_ops_in_graph(graph, external_roots);
    let changed = !patch.is_empty();
    for (block, mut effects) in patch.drain() {
        effects.sort_unstable();
        for effect in effects.into_iter().rev() {
            graph.skeleton.blocks[block].side_effects.remove(effect);
        }
    }
    changed
}
