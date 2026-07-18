//! Classification and construction of per-worker SOAC chunks.

#![cfg_attr(not(test), deny(clippy::expect_used, clippy::unwrap_used))]

use super::*;

#[derive(Clone, Copy)]
pub(super) enum ChunkInputKind {
    StorageOnly,
    StorageOrRange,
}

#[derive(Clone, Copy)]
enum ChunkableView {
    Storage(SemanticResourceRef),
    Range {
        start: NodeId,
        len: NodeId,
        step: Option<NodeId>,
    },
}

impl ChunkableView {
    fn classify(graph: &EGraph, view: NodeId, kind: ChunkInputKind) -> Option<Self> {
        if let Some(resource) = graph_ops::extract_storage_view_source(graph, view) {
            return Some(Self::Storage(resource));
        }
        if matches!(kind, ChunkInputKind::StorageOrRange) {
            if let Some((start, len, step)) = graph_ops::extract_array_range_operands(graph, view) {
                if matches!(
                    graph.types.get(&len),
                    Some(Type::Constructed(TypeName::UInt(32) | TypeName::Int(32), _))
                ) {
                    return Some(Self::Range { start, len, step });
                }
            }
        }
        None
    }

    fn len(self, graph: &mut EGraph) -> NodeId {
        match self {
            Self::Storage(resource) => emit_semantic_resource_len(graph, resource),
            Self::Range { len, .. } => len,
        }
    }

    fn chunk(
        self,
        graph: &mut EGraph,
        view_ty: Type<TypeName>,
        chunk_start: NodeId,
        chunk_len: NodeId,
        context: &str,
    ) -> Result<NodeId, String> {
        match self {
            Self::Storage(resource) => Ok(graph_ops::intern_chunked_resource_view(
                graph,
                resource.0,
                chunk_start,
                chunk_len,
                view_ty,
                None,
            )),
            Self::Range { start, step, .. } => {
                let has_step = step.is_some();
                let start_ty = graph
                    .types
                    .get(&start)
                    .cloned()
                    .ok_or_else(|| format!("phase1 {context}: range start has no type"))?;
                let start_delta = if let Some(step) = step {
                    graph_ops::intern_binop(graph, "*", chunk_start, step, start_ty.clone(), None)
                } else {
                    chunk_start
                };
                let new_start = graph_ops::intern_binop(graph, "+", start, start_delta, start_ty, None);
                let mut operands: smallvec::SmallVec<[NodeId; 4]> = smallvec![new_start, chunk_len];
                if let Some(step) = step {
                    operands.push(step);
                }
                Ok(graph.intern_pure(PureOp::ArrayRange { has_step }, operands, view_ty, None))
            }
        }
    }
}

pub(super) fn can_chunk_view(graph: &EGraph, view: NodeId, kind: ChunkInputKind) -> bool {
    ChunkableView::classify(graph, view, kind).is_some()
}

pub(super) fn can_clone_pure_subgraph(graph: &EGraph, root: NodeId, substitutions: &[NodeId]) -> bool {
    fn visit(
        graph: &EGraph,
        node: NodeId,
        substitutions: &[NodeId],
        seen: &mut std::collections::HashSet<NodeId>,
    ) -> bool {
        if substitutions.contains(&node) || !seen.insert(node) {
            return true;
        }
        match &graph.nodes[node] {
            ENode::Constant(_) => true,
            ENode::Pure { operands, .. } => {
                operands.iter().all(|operand| visit(graph, *operand, substitutions, seen))
            }
            _ => false,
        }
    }

    visit(graph, root, substitutions, &mut std::collections::HashSet::new())
}

pub(super) struct ChunkedSoacInputs {
    pub tid: NodeId,
    pub chunk_start: NodeId,
    pub chunk_len: NodeId,
    pub views: Vec<NodeId>,
}

pub(super) fn chunk_soac_inputs(
    graph: &mut EGraph,
    inputs: &[(NodeId, Type<TypeName>)],
    total_threads: u32,
    kind: ChunkInputKind,
    context: &str,
) -> Result<ChunkedSoacInputs, String> {
    let classified = inputs
        .iter()
        .map(|(view, ty)| {
            ChunkableView::classify(graph, *view, kind)
                .map(|view| (view, ty.clone()))
                .ok_or_else(|| format!("phase1 {context}: input is not a chunkable view"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let first = classified.first().ok_or_else(|| format!("phase1 {context}: no SOAC inputs"))?;
    let input_len = first.0.len(graph);
    let (tid, chunk_start, chunk_len) = emit_chunk_arithmetic(graph, total_threads, input_len)?;
    let views = classified
        .into_iter()
        .map(|(view, ty)| view.chunk(graph, ty, chunk_start, chunk_len, context))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ChunkedSoacInputs {
        tid,
        chunk_start,
        chunk_len,
        views,
    })
}

pub(super) fn chunk_view_like(
    graph: &mut EGraph,
    view: NodeId,
    view_ty: Type<TypeName>,
    chunk_start: NodeId,
    chunk_len: NodeId,
    kind: ChunkInputKind,
    context: &str,
) -> Result<NodeId, String> {
    ChunkableView::classify(graph, view, kind)
        .ok_or_else(|| format!("phase1 {context}: input is not a chunkable view"))?
        .chunk(graph, view_ty, chunk_start, chunk_len, context)
}
