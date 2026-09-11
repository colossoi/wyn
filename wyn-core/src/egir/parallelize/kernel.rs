//! Shared chunk arithmetic, resource sizing, and callable construction.

use super::*;
use crate::ast;
use crate::egir;
use crate::egir::soac::lambda as lambda_ops;
use crate::egir::types::OperandRef;
use crate::interface;
use crate::op;
use crate::types;
use crate::types::TypeExt;

/// Input-storage declarations needed when captured operator values are cloned
/// into a synthesized phase. Non-cloneable captures and captures of writable
/// resources keep the operation on the serial fallback.
pub(super) fn cloneable_capture_inputs(
    entry: &egir::program::PlannedEntry,
    analysis: &graph_ops::GraphAnalysis<'_, Semantic>,
    captures: &[OperandRef],
) -> Option<Vec<SemanticResourceDecl>> {
    let values = captures.iter().map(|capture| capture.value()).collect::<Option<Vec<_>>>()?;
    if values.iter().any(|capture| !can_clone_pure_subgraph(&entry.graph, *capture, &[])) {
        return None;
    }
    graph_ops::read_storage_resources(analysis, values)
        .into_iter()
        .map(|access| {
            entry
                .resource_declarations
                .iter()
                .find(|declaration| {
                    declaration.resource == access.resource
                        && declaration.role == interface::StorageRole::Input
                })
                .cloned()
        })
        .collect()
}

/// Build a two-argument (`a`, `b`) helper function of type `T -> T -> T` named
/// `name`, whose body is produced by `body(graph, a_nid, b_nid)` and returned.
fn synthesize_binary_fn(
    region: FunctionId,
    name: String,
    elem_ty: Type<TypeName>,
    span: ast::Span,
    body: impl FnOnce(&mut EGraph, ValueId, ValueId) -> ValueId,
) -> ParallelizeResult<Func<Semantic>> {
    let params = lambda_ops::named_parameters(&[elem_ty.clone(), elem_ty.clone()], "arg");
    let mut graph = EGraph::new();
    let arguments = lambda_ops::function_parameters(&mut graph, &params);
    let [a, b] = arguments.as_slice() else {
        return Err("binary helper did not produce its two declared parameters".into());
    };
    let Some(a_nid) = a.value() else {
        return Err("binary helper's first parameter is not passed by value".into());
    };
    let Some(b_nid) = b.value() else {
        return Err("binary helper's second parameter is not passed by value".into());
    };
    let result = body(&mut graph, a_nid, b_nid);
    let entry_block = graph.skeleton.entry;
    Ok(lambda_ops::finish_function(
        graph,
        entry_block,
        region,
        name,
        span,
        params,
        &[elem_ty],
        &[result],
    ))
}

/// A two-argument helper whose body is `inner(b, a)` — an arg-swapped wrapper
/// around a `T -> T -> T` combiner.
pub(super) fn synthesize_swap_wrapper(
    region: FunctionId,
    wrapper_name: String,
    inner: &Func<Semantic>,
    elem_ty: Type<TypeName>,
    capture_types: Vec<Type<TypeName>>,
    span: ast::Span,
) -> ParallelizeResult<Func<Semantic>> {
    let mut parameter_types = vec![elem_ty.clone(), elem_ty.clone()];
    parameter_types.extend(capture_types);
    let params = lambda_ops::named_parameters(&parameter_types, "arg");
    let mut graph = EGraph::new();
    let arguments = lambda_ops::function_parameters(&mut graph, &params);
    let [left, right, captures @ ..] = arguments.as_slice() else {
        return Err("swap wrapper did not produce its two declared value parameters".into());
    };
    let mut inner_arguments = vec![*right, *left];
    inner_arguments.extend_from_slice(captures);
    let entry = graph.skeleton.entry;
    let (_, result) = graph
        .emit_call(
            entry,
            inner.region,
            inner.params(),
            inner.result(),
            inner_arguments,
            inner.effects(),
            None,
            None,
        )
        .map_err(|cause| format!("swap wrapper call does not match the operator boundary: {cause}"))?;
    let result = graph_ops::pack_result_values(&mut graph, &result)?;
    Ok(lambda_ops::finish_function(
        graph,
        entry,
        region,
        wrapper_name,
        span,
        params,
        &[elem_ty],
        &[result],
    ))
}

pub(super) fn synthesize_u32_add_function(
    region: FunctionId,
    name: String,
    span: ast::Span,
) -> ParallelizeResult<Func<Semantic>> {
    let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
    let result_ty = u32_ty.clone();
    synthesize_binary_fn(region, name, u32_ty, span, move |graph, a_nid, b_nid| {
        graph.intern_pure(
            PureOp::BinOp(op::BinaryOperator::Add),
            smallvec![a_nid, b_nid],
            result_ty,
            None,
        )
    })
}

#[derive(Clone, Copy)]
enum ChunkableView {
    Storage(SemanticResourceRef),
    Range {
        start: ValueId,
        len: ValueId,
        step: Option<ValueId>,
    },
    StorageSlice {
        view: ValueId,
        len: ValueId,
    },
}

impl ChunkableView {
    fn classify(graph: &EGraph, view: ValueId) -> Option<Self> {
        if let Some(resource) = graph_ops::extract_storage_view_source(graph, view) {
            return Some(Self::Storage(resource));
        }
        if let ValueKind::Pure {
            op: PureOp::StorageView(op::PureViewSource::Inherited),
            operands,
        } = &graph.nodes[view].kind
        {
            let is_flat_storage_slice = operands.len() == 3
                && graph_ops::extract_storage_view_source(graph, operands[2]).is_some()
                && graph.nodes[view].ty.array_variant().is_some_and(types::is_array_variant_view)
                && graph.nodes[operands[0]].ty == graph.nodes[operands[1]].ty
                && matches!(
                    &graph.nodes[operands[0]].ty,
                    Type::Constructed(TypeName::UInt(32) | TypeName::Int(32), _)
                );
            if is_flat_storage_slice {
                return Some(Self::StorageSlice {
                    view,
                    len: operands[1],
                });
            }
        }
        if let Some((start, len, step)) = graph_ops::extract_array_range_operands(graph, view) {
            if matches!(
                graph.nodes.get(len).map(|node| &node.ty),
                Some(Type::Constructed(TypeName::UInt(32) | TypeName::Int(32), _))
            ) {
                return Some(Self::Range { start, len, step });
            }
        }
        None
    }

    fn len(self, graph: &mut EGraph) -> ValueId {
        match self {
            Self::Storage(resource) => graph_ops::intern_resource_len(graph, resource.0, None),
            Self::Range { len, .. } => len,
            Self::StorageSlice { len, .. } => len,
        }
    }

    fn chunk(
        self,
        graph: &mut EGraph,
        view_ty: Type<TypeName>,
        chunk_start: ValueId,
        chunk_len: ValueId,
        context: &str,
    ) -> Result<ValueId, String> {
        match self {
            Self::Storage(resource) => Ok(graph_ops::intern_chunked_resource_view(
                graph,
                resource.0,
                chunk_start,
                chunk_len,
                view_ty,
                None,
            )),
            Self::StorageSlice { view, len: _ } => Ok(graph_ops::intern_inherited_view(
                graph,
                view,
                chunk_start,
                chunk_len,
                view_ty,
                None,
            )),
            Self::Range { start, step, .. } => {
                let has_step = step.is_some();
                let start_ty = graph
                    .nodes
                    .get(start)
                    .map(|node| node.ty.clone())
                    .ok_or_else(|| format!("phase1 {context}: range start has no type"))?;
                let start_delta = if let Some(step) = step {
                    graph_ops::intern_binop(
                        graph,
                        op::BinaryOperator::Multiply,
                        chunk_start,
                        step,
                        start_ty.clone(),
                        None,
                    )
                } else {
                    chunk_start
                };
                let new_start = graph_ops::intern_binop(
                    graph,
                    op::BinaryOperator::Add,
                    start,
                    start_delta,
                    start_ty,
                    None,
                );
                let mut operands: smallvec::SmallVec<[ValueId; 4]> = smallvec![new_start, chunk_len];
                if let Some(step) = step {
                    operands.push(step);
                }
                Ok(graph.intern_pure(PureOp::ArrayRange { has_step }, operands, view_ty, None))
            }
        }
    }
}

pub(super) fn can_chunk_view(graph: &EGraph, view: ValueId) -> bool {
    ChunkableView::classify(graph, view).is_some()
}

pub(super) fn can_clone_pure_subgraph(graph: &EGraph, root: ValueId, substitutions: &[ValueId]) -> bool {
    fn visit(
        graph: &EGraph,
        node: ValueId,
        substitutions: &[ValueId],
        seen: &mut std::collections::HashSet<ValueId>,
    ) -> bool {
        if substitutions.contains(&node) || !seen.insert(node) {
            return true;
        }
        match &graph.nodes[node].kind {
            ValueKind::Constant(_) => true,
            ValueKind::Pure { operands, .. } => {
                operands.iter().all(|operand| visit(graph, *operand, substitutions, seen))
            }
            _ => false,
        }
    }

    visit(graph, root, substitutions, &mut std::collections::HashSet::new())
}

pub(super) struct ChunkedSoacInputs {
    pub tid: ValueId,
    pub chunk_start: ValueId,
    pub chunk_len: ValueId,
    pub views: Vec<ValueId>,
}

pub(super) fn chunk_soac_inputs(
    graph: &mut EGraph,
    inputs: &[(ValueId, Type<TypeName>)],
    total_threads: u32,
    context: &str,
) -> Result<ChunkedSoacInputs, String> {
    let classified = inputs
        .iter()
        .map(|(view, ty)| {
            ChunkableView::classify(graph, *view)
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
    view: ValueId,
    view_ty: Type<TypeName>,
    chunk_start: ValueId,
    chunk_len: ValueId,
    context: &str,
) -> Result<ValueId, String> {
    ChunkableView::classify(graph, view)
        .ok_or_else(|| format!("phase1 {context}: input is not a chunkable view"))?
        .chunk(graph, view_ty, chunk_start, chunk_len, context)
}
