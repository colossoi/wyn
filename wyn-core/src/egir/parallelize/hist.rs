//! Atomic lowering eligibility for canonical histograms.

use polytype::Type;

use crate::ast::TypeName;
use crate::egir::allocation::ResourcesAllocated;
use crate::egir::program::SemanticOpId;
use crate::egir::soac::hist;
use crate::egir::types::{ENode, NodeId, PureOp, SegSpace, Semantic, SkeletonTerminator};
use crate::op::BinaryOperator;
use crate::ssa::types::{AtomicOp, ConstantValue};

use super::planning::LocatedHist;

/// A histogram proven to be expressible as one native atomic update per
/// operation and input element.
pub(super) struct HistCandidate {
    pub owner: SemanticOpId,
    pub space: SegSpace,
    pub operations: Vec<hist::AtomicUpdate>,
}

pub(super) fn analyze_hist_candidate(
    program: &ResourcesAllocated,
    graph: &crate::egir::types::EGraph<Semantic>,
    located: LocatedHist<'_>,
) -> Option<HistCandidate> {
    let hist::SemanticState::Segmented(space) = &located.op.state else {
        return None;
    };
    let operations = located
        .op
        .form
        .operations
        .iter()
        .map(|operation| analyze_operation(program, graph, operation))
        .collect::<Option<Vec<_>>>()?;
    Some(HistCandidate {
        owner: located.owner,
        space: space.clone(),
        operations,
    })
}

fn analyze_operation(
    program: &ResourcesAllocated,
    graph: &crate::egir::types::EGraph<Semantic>,
    operation: &hist::HistOp,
) -> Option<hist::AtomicUpdate> {
    // The race factor is a contention estimate. Until replicated histograms
    // are available, atomic recipes are selected only for bounded contention;
    // high or dynamic estimates retain the serial fallback.
    let race_factor = constant_i32(graph, operation.race_factor)?;
    if !(1..=32).contains(&race_factor) {
        return None;
    }

    let hist::Update::Reduce { operator, .. } = &operation.update else {
        return None;
    };
    let [value_type] = operator.result_types.as_slice() else {
        return None;
    };
    let signed = match value_type {
        Type::Constructed(TypeName::Int(32), args) if args.is_empty() => true,
        Type::Constructed(TypeName::UInt(32), args) if args.is_empty() => false,
        _ => return None,
    };
    let body = operator.seg_body()?;
    let function = program.region(body.region)?;
    let direct = recognize_direct_atomic(function, signed);
    Some(direct.map_or(hist::AtomicUpdate::CompareExchange, hist::AtomicUpdate::Direct))
}

fn recognize_direct_atomic(
    function: &crate::egir::program::SemanticFunc,
    signed: bool,
) -> Option<AtomicOp> {
    if function.graph.skeleton.blocks.len() != 1
        || function.graph.skeleton.blocks.values().any(|block| !block.side_effects.is_empty())
    {
        return None;
    }
    let SkeletonTerminator::Return(Some(result)) =
        function.graph.skeleton.blocks[function.graph.skeleton.entry].term
    else {
        return None;
    };
    let ENode::Pure { op, operands } = &function.graph.nodes[result].kind else {
        return None;
    };
    if !matches_parameter_pair(&function.graph, operands) {
        return None;
    }
    match op {
        PureOp::BinOp(BinaryOperator::Add) => Some(AtomicOp::Add),
        PureOp::BinOp(BinaryOperator::BitwiseAnd) => Some(AtomicOp::And),
        PureOp::BinOp(BinaryOperator::BitwiseOr) => Some(AtomicOp::Or),
        PureOp::BinOp(BinaryOperator::BitwiseXor) => Some(AtomicOp::Xor),
        PureOp::Intrinsic { id, .. } if *id == crate::builtins::catalog().known().min => {
            Some(if signed { AtomicOp::SignedMin } else { AtomicOp::UnsignedMin })
        }
        PureOp::Intrinsic { id, .. } if *id == crate::builtins::catalog().known().max => {
            Some(if signed { AtomicOp::SignedMax } else { AtomicOp::UnsignedMax })
        }
        _ => None,
    }
}
fn matches_parameter_pair(graph: &crate::egir::types::EGraph<Semantic>, operands: &[NodeId]) -> bool {
    let [left, right] = operands else {
        return false;
    };
    matches!(
        (&graph.nodes[*left].kind, &graph.nodes[*right].kind),
        (ENode::FuncParam { index: 0 }, ENode::FuncParam { index: 1 })
            | (ENode::FuncParam { index: 1 }, ENode::FuncParam { index: 0 })
    )
}

fn constant_i32(graph: &crate::egir::types::EGraph<Semantic>, node: NodeId) -> Option<i32> {
    match &graph.nodes[node].kind {
        ENode::Constant(ConstantValue::I32(value)) => Some(*value),
        ENode::Pure {
            op: PureOp::Int(value),
            operands,
        } if operands.is_empty() => value.parse().ok(),
        _ => None,
    }
}
