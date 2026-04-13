//! Canonicalization: convert SSA FuncBody → EGraph (sea-of-nodes + skeleton).
//!
//! Walks the dominator tree in preorder, lifting pure instructions into the
//! e-graph with hash-consing (giving GVN for free), and leaving effectful
//! instructions in the skeleton CFG.

use crate::ast::TypeName;
use crate::ssa::types::{BlockId, ConstantValue, FuncBody, InstKind, ValueId, ValueRef};
use polytype::Type;
use smallvec::SmallVec;
use std::collections::HashMap;
use wyn_ssa::ValueLike;

use super::domtree::{DomTree, SsaCfgView};
use super::types::*;

/// Convert an SSA FuncBody into an EGraph.
///
/// Returns the EGraph, a DomTree built from the *skeleton* CFG
/// (needed later by the elaboration pass), and the orig→skeleton
/// block map that the elaborator needs to rewire control headers.
pub fn canonicalize(body: &FuncBody) -> (EGraph, DomTree, HashMap<BlockId, BlockId>) {
    let mut graph = EGraph::new();
    let mut val_map: HashMap<ValueId, NodeId> = HashMap::new();

    // --- 1. Compute domtree from original SSA CFG. ---
    let ssa_domtree = DomTree::build(&SsaCfgView { body });

    // --- 2. Mirror block structure into the skeleton. ---
    let mut block_map: HashMap<BlockId, BlockId> = HashMap::new();
    // Entry block already exists in skeleton.
    block_map.insert(body.entry_block(), graph.skeleton.entry);
    // Create skeleton blocks for all non-entry blocks (in domtree preorder
    // to keep a stable ordering).
    for &orig_bid in ssa_domtree.preorder() {
        if orig_bid == body.entry_block() {
            continue;
        }
        let skel_bid = graph.skeleton.create_block();
        block_map.insert(orig_bid, skel_bid);
    }

    // --- 3. Create nodes for function parameters. ---
    for (i, (vid, ty, _name)) in body.params.iter().enumerate() {
        let nid = graph.add_func_param(i, ty.clone());
        val_map.insert(*vid, nid);
    }

    // --- 4. Walk domtree in preorder. ---
    for &orig_bid in ssa_domtree.preorder() {
        let skel_bid = block_map[&orig_bid];
        let block = body.get_block(orig_bid);

        // 4a. Block parameters.
        for (i, &param_vid) in block.params.iter().enumerate() {
            let ty = body.get_value_type(param_vid).clone();
            let nid = graph.add_block_param(skel_bid, i, ty);
            val_map.insert(param_vid, nid);
            graph.skeleton.blocks[skel_bid].params.push(nid);
        }

        // 4b. Instructions.
        for &inst_id in &block.insts {
            let inst = body.get_inst(inst_id);
            let kind = &inst.data;

            let ty = inst
                .result
                .map(|r| body.get_value_type(r).clone())
                .unwrap_or_else(|| Type::Constructed(TypeName::Unit, vec![]));

            // An instruction is lifted into the pure sea only if:
            //   (1) it carries no effect tokens (front-end didn't order it), and
            //   (2) its InstKind has a pure mapping (blacklist: Alloca/Load/Store/Soac).
            let pure_op = if inst.effects.is_none() {
                extract_pure_op(kind, &ty)
            } else {
                None
            };

            if let Some(pure_op) = pure_op {
                let operands = resolve_operands(kind, &val_map, &mut graph);
                let nid = graph.intern_pure(pure_op, operands, ty);
                if let Some(result_vid) = inst.result {
                    val_map.insert(result_vid, nid);
                }
            } else {
                // Effectful instruction → keep in skeleton.
                let operands = resolve_operands(kind, &val_map, &mut graph);
                let result_nid = inst.result.map(|r| {
                    let ty = body.get_value_type(r).clone();
                    graph.alloc_side_effect_result(ty)
                });

                graph.skeleton.blocks[skel_bid].side_effects.push(SideEffect {
                    // Canonicalize sees SSA instructions only — wrap in
                    // SideEffectKind::Inst. There's no `PendingSoac` path
                    // because SSA no longer carries SOACs.
                    kind: super::types::SideEffectKind::Inst(kind.clone()),
                    operand_nodes: operands,
                    result: result_nid,
                    effects: inst.effects,
                });

                if let (Some(result_vid), Some(result_nid)) = (inst.result, result_nid) {
                    val_map.insert(result_vid, result_nid);
                }
            }
        }

        // 4c. Terminator.
        let term = convert_terminator(&block.term, &val_map, &block_map);
        graph.skeleton.blocks[skel_bid].term = term;
    }

    // Build the domtree from the skeleton (for elaboration).
    let skel_domtree = DomTree::build(&super::domtree::SkeletonCfgView {
        skeleton: &graph.skeleton,
    });

    (graph, skel_domtree, block_map)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Resolve all value uses of an instruction to NodeIds.
fn resolve_operands(
    kind: &InstKind,
    val_map: &HashMap<ValueId, NodeId>,
    graph: &mut EGraph,
) -> SmallVec<[NodeId; 4]> {
    kind.value_uses().iter().map(|vr| resolve_value_ref(vr, val_map, graph)).collect()
}

fn resolve_value_ref(vr: &ValueRef, val_map: &HashMap<ValueId, NodeId>, graph: &mut EGraph) -> NodeId {
    match vr {
        ValueRef::Ssa(vid) => *val_map
            .get(vid)
            .unwrap_or_else(|| panic!("ValueId {:?} not in val_map during canonicalization", vid)),
        ValueRef::Const(c) => {
            let ty = const_type(c);
            graph.intern_constant(*c, ty)
        }
    }
}

/// Infer the type of a ConstantValue.
fn const_type(c: &ConstantValue) -> Type<TypeName> {
    match c {
        ConstantValue::I32(_) => Type::Constructed(TypeName::Int(32), vec![]),
        ConstantValue::U32(_) => Type::Constructed(TypeName::UInt(32), vec![]),
        ConstantValue::F32(_) => Type::Constructed(TypeName::Float(32), vec![]),
        ConstantValue::Bool(_) => Type::Constructed(TypeName::Bool, vec![]),
    }
}

/// Convert a wyn-ssa Terminator to a SkeletonTerminator.
fn convert_terminator(
    term: &wyn_ssa::Terminator,
    val_map: &HashMap<ValueId, NodeId>,
    block_map: &HashMap<BlockId, BlockId>,
) -> SkeletonTerminator {
    let mv = |vid: ValueId| -> NodeId {
        *val_map.get(&vid).unwrap_or_else(|| panic!("ValueId {:?} not in val_map (terminator)", vid))
    };
    let mb = |bid: BlockId| -> BlockId {
        *block_map.get(&bid).unwrap_or_else(|| panic!("BlockId {:?} not in block_map (terminator)", bid))
    };

    match term {
        wyn_ssa::Terminator::Return(None) => SkeletonTerminator::Return(None),
        wyn_ssa::Terminator::Return(Some(v)) => SkeletonTerminator::Return(Some(mv(*v))),
        wyn_ssa::Terminator::Branch { target, args } => SkeletonTerminator::Branch {
            target: mb(*target),
            args: args.iter().map(|v| mv(*v)).collect(),
        },
        wyn_ssa::Terminator::CondBranch {
            cond,
            then_target,
            then_args,
            else_target,
            else_args,
        } => SkeletonTerminator::CondBranch {
            cond: mv(*cond),
            then_target: mb(*then_target),
            then_args: then_args.iter().map(|v| mv(*v)).collect(),
            else_target: mb(*else_target),
            else_args: else_args.iter().map(|v| mv(*v)).collect(),
        },
        wyn_ssa::Terminator::Unreachable => SkeletonTerminator::Unreachable,
    }
}
