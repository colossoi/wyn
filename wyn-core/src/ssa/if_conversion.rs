//! Replace private value-producing diamonds with eager select instructions.
//! Only cheap, total arm work can move to the selection header; existing
//! dominating operands need no speculation proof.

use super::ir::{InstPlacement, Substitutions};
use super::optimize::is_speculatable;
use super::types::{BlockId, ControlHeader, FuncBody, InstId, InstKind, Terminator, ValueId, ValueRef};
use crate::builtins::{
    by_id,
    lowering::{BuiltinLowering, PrimOp},
    select,
};
use crate::op::OpTag;
use crate::types::{bool_type, TypeExt};
use crate::LookupSet;
use wyn_graph::DominatorTree;

const ADDED_WORK_LIMIT: usize = 4;

#[cfg(test)]
#[path = "if_conversion_tests.rs"]
mod tests;

pub(super) fn run(body: &mut FuncBody) {
    convert(body, ADDED_WORK_LIMIT);
}

fn convert(body: &mut FuncBody, budget: usize) {
    let dominators = DominatorTree::build(body.inner.entry, |block, successors| {
        successors.extend(body.inner.blocks[block].term.successors());
    });
    // Children first: a converted inner selection becomes a straight-line arm
    // of its enclosing selection. No new conditional headers are introduced.
    let headers = dominators.preorder().iter().rev().copied().collect::<Vec<_>>();
    for header in headers {
        let Some(candidate) = candidate(body, header, budget) else {
            continue;
        };
        let mut replacements = Substitutions::default();
        for instruction in candidate.instructions {
            body.inner.insts[instruction].placement = InstPlacement::Block(header);
            body.inner.blocks[header].insts.push(instruction);
        }
        for ((param, yes), no) in std::mem::take(&mut body.inner.blocks[candidate.merge].params)
            .into_iter()
            .zip(candidate.yes)
            .zip(candidate.no)
        {
            let value = if yes == no {
                yes
            } else {
                body.inner
                    .append_inst(
                        header,
                        InstKind::select(no, yes, candidate.condition),
                        body.inner.values[param].ty.clone(),
                    )
                    .into()
            };
            replacements.insert(param, value);
        }
        body.inner.blocks[header].term = Terminator::Branch {
            target: candidate.merge,
            args: vec![],
        };
        body.inner.blocks[header].control_header = None;
        body.inner.blocks.retain(|block, _| !candidate.blocks.contains(&block));
        replacements.finish(&mut body.inner);
    }
    super::constant_folding::fold(body);
    super::eliminate_dead_pure_instructions(body);
}

struct Arm {
    blocks: Vec<BlockId>,
    instructions: Vec<InstId>,
    arguments: Vec<ValueRef>,
}

struct Candidate {
    merge: BlockId,
    condition: ValueRef,
    blocks: LookupSet<BlockId>,
    instructions: Vec<InstId>,
    yes: Vec<ValueRef>,
    no: Vec<ValueRef>,
}

fn arm(body: &FuncBody, header: BlockId, mut block: BlockId, merge: BlockId) -> Option<Arm> {
    let mut blocks = Vec::new();
    let mut instructions = Vec::new();
    loop {
        if block == header || block == merge || block == body.inner.entry || blocks.contains(&block) {
            return None;
        }
        let node = &body.inner.blocks[block];
        if !node.params.is_empty() || node.control_header.is_some() {
            return None;
        }
        let Terminator::Branch { target, args } = &node.term else {
            return None;
        };
        blocks.push(block);
        instructions.extend(&node.insts);
        if *target == merge {
            return Some(Arm {
                blocks,
                instructions,
                arguments: args.clone(),
            });
        }
        if !args.is_empty() {
            return None;
        }
        block = *target;
    }
}

fn candidate(body: &FuncBody, header: BlockId, budget: usize) -> Option<Candidate> {
    let node = body.inner.blocks.get(header)?;
    let Some(ControlHeader::Selection { merge }) = node.control_header else {
        return None;
    };
    let Terminator::CondBranch {
        cond,
        then_target,
        then_args,
        else_target,
        else_args,
    } = &node.term
    else {
        return None;
    };
    if !then_args.is_empty() || !else_args.is_empty() || body.value_ref_type(*cond) != bool_type() {
        return None;
    }
    let yes = arm(body, header, *then_target, merge)?;
    let no = arm(body, header, *else_target, merge)?;
    let blocks = yes.blocks.iter().chain(&no.blocks).copied().collect::<LookupSet<_>>();
    if blocks.len() != yes.blocks.len() + no.blocks.len() {
        return None;
    }
    let params = &body.inner.blocks[merge].params;
    if params.len() != yes.arguments.len() || params.len() != no.arguments.len() {
        return None;
    }
    for ((&param, &yes), &no) in params.iter().zip(&yes.arguments).zip(&no.arguments) {
        let ty = body.get_value_type(param);
        if !select::supported_type(ty) || body.value_ref_type(yes) != *ty || body.value_ref_type(no) != *ty
        {
            return None;
        }
    }
    // Preserve shared blocks and structured targets, including loop continues.
    for (id, block) in &body.inner.blocks {
        if id == header || blocks.contains(&id) {
            continue;
        }
        if block.term.successors().iter().any(|b| *b == merge || blocks.contains(b)) {
            return None;
        }
        let protected = match block.control_header {
            Some(ControlHeader::Selection { merge }) => blocks.contains(&merge),
            Some(ControlHeader::Loop {
                merge,
                continue_block,
            }) => blocks.contains(&merge) || blocks.contains(&continue_block),
            None => false,
        };
        if protected {
            return None;
        }
    }
    let dominators = DominatorTree::build(body.inner.entry, |block, successors| {
        successors.extend(body.inner.blocks[block].term.successors());
    });
    let mut available = LookupSet::<ValueId>::new();
    let available_value = |value: ValueRef, available: &LookupSet<ValueId>| match value {
        ValueRef::Const(_) => true,
        ValueRef::Ssa(value) => {
            available.contains(&value)
                || body.inner.block_of_value(value).is_some_and(|block| dominators.dominates(block, header))
        }
    };
    let instructions = yes.instructions.into_iter().chain(no.instructions).collect::<Vec<_>>();
    let mut cost = 0usize;
    for &instruction in &instructions {
        let node = &body.inner.insts[instruction];
        let result = node.result?;
        cost = cost.checked_add(cheap_cost(body, instruction)?)?;
        if cost > budget || !node.data.value_uses().into_iter().all(|v| available_value(v, &available)) {
            return None;
        }
        available.insert(result);
    }
    if !yes.arguments.iter().chain(&no.arguments).all(|&v| available_value(v, &available)) {
        return None;
    }
    Some(Candidate {
        merge,
        condition: *cond,
        blocks,
        instructions,
        yes: yes.arguments,
        no: no.arguments,
    })
}

fn cheap_cost(body: &FuncBody, instruction: InstId) -> Option<usize> {
    let node = &body.inner.insts[instruction];
    if !is_speculatable(&node.data) {
        return None;
    }
    let ty = body.get_value_type(node.result?);
    if !select::supported_type(ty) {
        return None;
    }
    let InstKind::Op { tag, operands } = &node.data else {
        return None;
    };
    if !operands.iter().all(|&value| select::supported_type(&body.value_ref_type(value))) {
        return None;
    }
    match tag {
        OpTag::Int(_) | OpTag::Uint(_) | OpTag::Float(_) | OpTag::Bool(_) | OpTag::Vector(_) => {
            return Some(0)
        }
        OpTag::BinOp(_) | OpTag::UnaryOp(_) => {}
        OpTag::Project { .. } if body.value_ref_type(*operands.first()?).is_vec() => {}
        OpTag::Intrinsic { id, overload_idx } => {
            let BuiltinLowering::PrimOp(prim) = &by_id(*id).overloads().get(*overload_idx)?.lowering else {
                return None;
            };
            if !matches!(
                prim,
                PrimOp::Select
                    | PrimOp::FAdd
                    | PrimOp::FSub
                    | PrimOp::FMul
                    | PrimOp::IAdd
                    | PrimOp::ISub
                    | PrimOp::IMul
                    | PrimOp::FOrdEqual
                    | PrimOp::FOrdNotEqual
                    | PrimOp::FOrdLessThan
                    | PrimOp::FOrdGreaterThan
                    | PrimOp::FOrdLessThanEqual
                    | PrimOp::FOrdGreaterThanEqual
                    | PrimOp::IEqual
                    | PrimOp::INotEqual
                    | PrimOp::SLessThan
                    | PrimOp::ULessThan
                    | PrimOp::SGreaterThan
                    | PrimOp::UGreaterThan
                    | PrimOp::SLessThanEqual
                    | PrimOp::ULessThanEqual
                    | PrimOp::SGreaterThanEqual
                    | PrimOp::UGreaterThanEqual
                    | PrimOp::BitwiseAnd
                    | PrimOp::BitwiseOr
                    | PrimOp::BitwiseXor
                    | PrimOp::Not
                    | PrimOp::Bitcast
                    | PrimOp::SIToFP
                    | PrimOp::UIToFP
                    | PrimOp::SConvert
                    | PrimOp::UConvert
                    | PrimOp::FPConvert
            ) {
                return None;
            }
        }
        _ => return None,
    }
    Some(
        operands
            .iter()
            .map(|&value| body.value_ref_type(value).vec_size().unwrap_or(1))
            .fold(ty.vec_size().unwrap_or(1), usize::max),
    )
}
