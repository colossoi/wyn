//! Shared chunk arithmetic, resource sizing, and callable construction.

use super::*;
pub(super) fn apply_manifest_resource_sizes(
    entry: &mut crate::egir::program::PlannedEntry,
    resources: &crate::egir::program::LogicalResourceArena,
) {
    for declaration in &mut entry.resource_declarations {
        let resource = declaration.resource.0;
        let logical = &resources[resource];
        declaration.size = logical.size.clone();
    }
}

/// Emit the chunk-arithmetic preamble (`tid`, `chunk_start`,
/// `chunk_len`) as pure nodes in `graph`. Caller supplies the
/// `input_len` NodeId (typed `u32`) — for StorageView inputs that's a
/// `_w_intrinsic_storage_len(set, binding)` call; for Range inputs
/// it's the Range's own `len` operand. Returns
/// `(tid, chunk_start, chunk_len)`.
pub(super) fn emit_chunk_arithmetic(
    graph: &mut crate::egir::types::EGraph,
    total_threads: u32,
    input_len: NodeId,
) -> Result<(NodeId, NodeId, NodeId), String> {
    let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
    // The chunk arithmetic runs in the input's *index* type: storage-view
    // inputs index in u32 (`_w_intrinsic_storage_len`), Range inputs in the
    // range's own element type (typically i32). Computing in u32 and feeding
    // a u32 `chunk_start`/`chunk_len` into an i32 Range produced an
    // `OpCompositeConstruct` whose constituents didn't match the i32
    // `{start, step, len}` struct (spirv-val rejected it). Derive the index
    // type from `input_len` and emit all arithmetic there.
    let index_ty = graph
        .types
        .get(&input_len)
        .cloned()
        .ok_or_else(|| format!("chunk input length {input_len:?} has no type"))?;
    let is_u32 = index_ty == u32_ty;

    // `tid`/`num_workgroups` are u32 intrinsics. The returned `tid` stays u32
    // (callers use it as a `partials[tid]` storage index); the index-typed
    // copies feed the chunk math.
    let tid = graph_ops::intern_intrinsic(
        graph,
        catalog().known().thread_id,
        smallvec![],
        u32_ty.clone(),
        None,
    );
    let nwg = graph_ops::intern_intrinsic(
        graph,
        catalog().known().num_workgroups,
        smallvec![],
        u32_ty.clone(),
        None,
    );
    let tid_idx = cast_u32_to_index(graph, tid, &index_ty)?;
    let nwg_idx = cast_u32_to_index(graph, nwg, &index_ty)?;

    // Runtime total thread count = num_workgroups.x * workgroup width. With a
    // `derived_from_input_length` dispatch (~ceil(n / width) workgroups) this
    // makes chunk_size ≈ 1, so each thread reduces ~one element — a saturating
    // grid rather than a fixed `total_threads`-wide one. `total_threads` is the
    // compile-time per-workgroup width.
    let wg_width = intern_index_lit(graph, total_threads, &index_ty);
    let total = graph_ops::intern_binop(graph, "*", nwg_idx, wg_width, index_ty.clone(), None);
    let one = intern_index_lit(graph, 1, &index_ty);
    let total_minus_one = graph_ops::intern_binop(graph, "-", total, one, index_ty.clone(), None);
    let len_plus = graph_ops::intern_binop(graph, "+", input_len, total_minus_one, index_ty.clone(), None);
    let chunk_size = graph_ops::intern_binop(graph, "/", len_plus, total, index_ty.clone(), None);
    let raw_chunk_start = graph_ops::intern_binop(graph, "*", tid_idx, chunk_size, index_ty.clone(), None);
    let min_name = if is_u32 { "u32.min" } else { "i32.min" };
    let min_op =
        catalog().lookup_by_any_name(min_name).ok_or_else(|| format!("{} not in catalog", min_name))?;
    // Clamp idle workers to the end before subtraction. For n < workers this
    // produces `(start=n,len=0)` instead of underflowing `n-start`.
    let chunk_start = graph_ops::intern_intrinsic(
        graph,
        min_op.id,
        smallvec![raw_chunk_start, input_len],
        index_ty.clone(),
        None,
    );
    let remaining = graph_ops::intern_binop(graph, "-", input_len, chunk_start, index_ty.clone(), None);
    let chunk_len =
        graph_ops::intern_intrinsic(graph, min_op.id, smallvec![chunk_size, remaining], index_ty, None);
    Ok((tid, chunk_start, chunk_len))
}

/// Integer literal `n` typed as `index_ty` (`u32` → `PureOp::Uint`, else
/// `PureOp::Int`).
fn intern_index_lit(graph: &mut crate::egir::types::EGraph, n: u32, index_ty: &Type<TypeName>) -> NodeId {
    let op = match index_ty {
        Type::Constructed(TypeName::UInt(32), _) => crate::egir::types::PureOp::Uint(n.to_string()),
        _ => crate::egir::types::PureOp::Int(n.to_string()),
    };
    graph.intern_pure(op, smallvec![], index_ty.clone(), None)
}

/// Cast a u32 value into `index_ty`: identity for u32, else the per-type
/// bitcast intrinsic (`i32.u32`).
fn cast_u32_to_index(
    graph: &mut crate::egir::types::EGraph,
    v: NodeId,
    index_ty: &Type<TypeName>,
) -> Result<NodeId, String> {
    match index_ty {
        Type::Constructed(TypeName::UInt(32), _) => Ok(v),
        Type::Constructed(TypeName::Int(32), _) => {
            let conv = catalog()
                .lookup_by_any_name("i32.u32")
                .ok_or_else(|| "i32.u32 not in catalog".to_string())?;
            Ok(graph_ops::intern_intrinsic(
                graph,
                conv.id,
                smallvec![v],
                index_ty.clone(),
                None,
            ))
        }
        other => Err(format!("chunk arithmetic: unsupported index type {:?}", other)),
    }
}

pub(super) fn dispatch_worker_logical_size(elem_ty: &Type<TypeName>) -> crate::egir::program::LogicalSize {
    crate::ssa::layout::type_byte_size(elem_ty)
        .map_or(crate::egir::program::LogicalSize::Unspecified, |bytes| {
            crate::egir::program::LogicalSize::SameAsDispatch { elem_bytes: bytes }
        })
}

/// Build a two-argument (`a`, `b`) helper function of type `T -> T -> T` named
/// `name`, whose body is produced by `body(graph, a_nid, b_nid)` and returned.
fn synthesize_binary_fn(
    name: String,
    elem_ty: Type<TypeName>,
    span: crate::ast::Span,
    body: impl FnOnce(&mut EGraph, NodeId, NodeId) -> NodeId,
) -> SemanticFunc {
    let mut graph = EGraph::new();
    let a_nid = graph.add_func_param(0, elem_ty.clone());
    let b_nid = graph.add_func_param(1, elem_ty.clone());
    let result = body(&mut graph, a_nid, b_nid);
    let entry_block = graph.skeleton.entry;
    graph.skeleton.blocks[entry_block].term = SkeletonTerminator::Return(Some(result));
    SemanticFunc::new(
        name,
        span,
        None,
        vec![
            (elem_ty.clone(), "a".to_string()),
            (elem_ty.clone(), "b".to_string()),
        ],
        elem_ty,
        graph,
        LookupMap::new(),
    )
}

/// A two-argument helper whose body is `inner(b, a)` — an arg-swapped wrapper
/// around a `T -> T -> T` combiner.
pub(super) fn synthesize_swap_wrapper(
    wrapper_name: String,
    inner: String,
    elem_ty: Type<TypeName>,
    span: crate::ast::Span,
) -> SemanticFunc {
    let result_ty = elem_ty.clone();
    synthesize_binary_fn(wrapper_name, elem_ty, span, move |graph, a_nid, b_nid| {
        graph.intern_pure(PureOp::Call(inner), smallvec![b_nid, a_nid], result_ty, None)
    })
}

pub(super) fn synthesize_u32_add_function(name: String, span: crate::ast::Span) -> SemanticFunc {
    let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
    let result_ty = u32_ty.clone();
    synthesize_binary_fn(name, u32_ty, span, move |graph, a_nid, b_nid| {
        graph.intern_pure(
            PureOp::BinOp("+".into()),
            smallvec![a_nid, b_nid],
            result_ty,
            None,
        )
    })
}

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
            Self::Storage(resource) => graph_ops::intern_resource_len(graph, resource.0, None),
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
