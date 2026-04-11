//! Scoped elaboration: convert EGraph → FuncBody.
//!
//! Demand-driven placement of pure nodes back into sequential blocks,
//! using a scoped hashmap to get GVN and DCE for free.
//!
//! - **GVN**: two uses of the same canonical node in the same domtree path
//!   hit the `elaborated` cache → reuse the same ValueId.
//! - **DCE**: nodes not reachable from any skeleton root are never demanded,
//!   so they're never emitted.
//! - **No partial redundancy**: scoping ensures we never reuse a value across
//!   unrelated CFG paths.

use crate::ast::TypeName;
use crate::ssa::builder::FuncBuilder;
use crate::ssa::types::{BlockId, ControlHeader, FuncBody, InstKind, ValueId, ValueRef};
use polytype::Type;
use std::collections::HashMap;
use wyn_ssa::BlockId as SkelBlockId;

use super::domtree::DomTree;
use super::extract;
use super::scoped_map::ScopedMap;
use super::types::*;

/// Elaborate an EGraph back into a FuncBody.
pub fn elaborate(
    graph: &EGraph,
    domtree: &DomTree,
    params: &[(Type<TypeName>, String)],
    return_ty: Type<TypeName>,
    control_headers: &HashMap<BlockId, ControlHeader>,
    orig_block_map: &HashMap<BlockId, SkelBlockId>,
) -> FuncBody {
    // Phase 1: cost-based extraction.
    let best = extract::extract(graph);

    // Phase 2: set up elaborator.
    let mut elab = Elaborator {
        graph,
        best,
        domtree,
        elaborated: ScopedMap::new(),
        builder: FuncBuilder::new(params.to_vec(), return_ty),
        block_map: HashMap::new(),
        current_block: None,
    };

    // Map function params: NodeId → ValueId.
    for i in 0..elab.builder.num_params() {
        let vid = elab.builder.get_param(i);
        // Find the FuncParam NodeId for this index.
        for (nid, node) in &graph.nodes {
            if matches!(node, ENode::FuncParam { index } if *index == i) {
                let resolved = elab.resolve(nid);
                elab.elaborated.insert(resolved, vid);
                break;
            }
        }
    }

    // Pre-create all output blocks to match the skeleton.
    for (skel_bid, _) in &graph.skeleton.blocks {
        if skel_bid == graph.skeleton.entry {
            elab.block_map.insert(skel_bid, elab.builder.entry());
        } else {
            let out_bid = elab.builder.create_block();
            elab.block_map.insert(skel_bid, out_bid);
        }
    }

    // Pre-create block params for all blocks.
    for (skel_bid, skel_block) in &graph.skeleton.blocks {
        let out_bid = elab.block_map[&skel_bid];
        for &param_nid in &skel_block.params {
            let ty = graph.types[&param_nid].clone();
            let vid = elab.builder.add_block_param(out_bid, ty);
            let resolved = elab.resolve(param_nid);
            elab.elaborated.insert(resolved, vid);
        }
    }

    // Map control headers from original blocks → output blocks.
    let skel_to_output = &elab.block_map;
    let mut reverse_orig_map: HashMap<SkelBlockId, BlockId> = HashMap::new();
    for (&orig, &skel) in orig_block_map {
        reverse_orig_map.insert(skel, orig);
    }
    for (&skel_bid, &out_bid) in skel_to_output.iter() {
        if let Some(&orig_bid) = reverse_orig_map.get(&skel_bid) {
            if let Some(ch) = control_headers.get(&orig_bid) {
                let mapped = match ch {
                    ControlHeader::Selection { merge } => {
                        let skel_merge = orig_block_map[merge];
                        ControlHeader::Selection {
                            merge: skel_to_output[&skel_merge],
                        }
                    }
                    ControlHeader::Loop {
                        merge,
                        continue_block,
                    } => {
                        let skel_merge = orig_block_map[merge];
                        let skel_cont = orig_block_map[continue_block];
                        ControlHeader::Loop {
                            merge: skel_to_output[&skel_merge],
                            continue_block: skel_to_output[&skel_cont],
                        }
                    }
                };
                elab.builder.set_control_header(out_bid, mapped);
            }
        }
    }

    // Elaborate in domtree preorder.
    let preorder: Vec<SkelBlockId> = domtree.preorder().to_vec();
    elab.elaborate_entry(&preorder);

    elab.builder.finish_unchecked()
}

struct Elaborator<'a> {
    graph: &'a EGraph,
    best: HashMap<NodeId, NodeId>,
    domtree: &'a DomTree,
    elaborated: ScopedMap<NodeId, ValueId>,
    builder: FuncBuilder,
    block_map: HashMap<SkelBlockId, BlockId>,
    current_block: Option<BlockId>,
}

impl<'a> Elaborator<'a> {
    /// Elaborate all blocks in domtree preorder.
    fn elaborate_entry(&mut self, preorder: &[SkelBlockId]) {
        if let Some(&entry) = preorder.first() {
            self.elaborate_subtree(entry);
        }
    }

    fn elaborate_subtree(&mut self, skel_bid: SkelBlockId) {
        self.elaborated.push_scope();
        let out_bid = self.block_map[&skel_bid];
        self.current_block = Some(out_bid);
        self.builder.switch_to_block_unchecked(out_bid);

        let skel_block = self.graph.skeleton.blocks[skel_bid].clone();

        // Elaborate side-effectful instructions.
        for se in &skel_block.side_effects {
            self.elaborate_side_effect(se);
        }

        // Elaborate terminator.
        self.elaborate_terminator(&skel_block.term);

        // Recurse into domtree children.
        let children: Vec<SkelBlockId> = self.domtree.dom_children(skel_bid).to_vec();
        for child in children {
            self.elaborate_subtree(child);
        }

        self.elaborated.pop_scope();
    }

    /// Elaborate a side-effectful instruction.
    fn elaborate_side_effect(&mut self, se: &SideEffect) {
        // Demand-elaborate all operands (this recursively places pure nodes).
        let args: Vec<ValueId> = se.operand_nodes.iter().map(|&nid| self.demand(nid)).collect();

        // Rebuild the InstKind with new ValueIds.
        let kind = rebuild_effectful_inst_kind(&se.kind, &args);

        if let Some(result_nid) = se.result {
            let ty = self.graph.types[&result_nid].clone();
            let effects = se.effects;
            let vid = self
                .builder
                .push_inst_with_effects(kind, ty, effects)
                .expect("elaborate side-effect push failed");
            self.elaborated.insert(result_nid, vid);
        } else {
            let effects = se.effects;
            self.builder
                .push_void_inst_with_effects(kind, effects)
                .expect("elaborate void side-effect push failed");
        }
    }

    /// Demand-driven elaboration: given a NodeId, produce a ValueId.
    fn demand(&mut self, nid: NodeId) -> ValueId {
        let resolved = self.resolve(nid);

        // Check scoped cache.
        if let Some(vid) = self.elaborated.get(&resolved) {
            return vid;
        }

        let node = self.graph.nodes[resolved].clone();

        match &node {
            ENode::Constant(c) => {
                let ty = self.graph.types[&resolved].clone();
                let kind = const_to_inst_kind(c);
                let vid = self.builder.push_inst(kind, ty).expect("elaborate constant push failed");
                self.elaborated.insert(resolved, vid);
                vid
            }
            ENode::Pure { op, operands } => {
                // Recursively demand-elaborate operands.
                let args: Vec<ValueId> = operands.iter().map(|&op_nid| self.demand(op_nid)).collect();

                let ty = self.graph.types[&resolved].clone();
                let kind = pure_to_inst_kind(op, &args);
                let vid = self.builder.push_inst(kind, ty).expect("elaborate pure push failed");
                self.elaborated.insert(resolved, vid);
                vid
            }
            ENode::FuncParam { .. } | ENode::BlockParam { .. } => {
                panic!(
                    "FuncParam/BlockParam {:?} should have been pre-populated in elaborated map",
                    resolved
                );
            }
            ENode::SideEffectResult => {
                panic!(
                    "SideEffectResult {:?} should have been populated during side-effect elaboration",
                    resolved
                );
            }
            ENode::Union { .. } => {
                panic!("Union {:?} should have been resolved by extract", resolved);
            }
        }
    }

    /// Resolve a NodeId through the extraction map.
    fn resolve(&self, nid: NodeId) -> NodeId {
        self.best.get(&nid).copied().unwrap_or(nid)
    }

    /// Elaborate a skeleton terminator.
    fn elaborate_terminator(&mut self, term: &SkeletonTerminator) {
        let t = match term {
            SkeletonTerminator::Return(None) => wyn_ssa::Terminator::Return(None),
            SkeletonTerminator::Return(Some(nid)) => wyn_ssa::Terminator::Return(Some(self.demand(*nid))),
            SkeletonTerminator::Branch { target, args } => {
                let out_args: Vec<ValueId> = args.iter().map(|&nid| self.demand(nid)).collect();
                wyn_ssa::Terminator::Branch {
                    target: self.block_map[target],
                    args: out_args,
                }
            }
            SkeletonTerminator::CondBranch {
                cond,
                then_target,
                then_args,
                else_target,
                else_args,
            } => {
                let cond_vid = self.demand(*cond);
                let ta: Vec<ValueId> = then_args.iter().map(|&nid| self.demand(nid)).collect();
                let ea: Vec<ValueId> = else_args.iter().map(|&nid| self.demand(nid)).collect();
                wyn_ssa::Terminator::CondBranch {
                    cond: cond_vid,
                    then_target: self.block_map[then_target],
                    then_args: ta,
                    else_target: self.block_map[else_target],
                    else_args: ea,
                }
            }
            SkeletonTerminator::Unreachable => wyn_ssa::Terminator::Unreachable,
        };
        let _ = self.builder.terminate(t);
    }
}

// ---------------------------------------------------------------------------
// Conversion: ENode → InstKind
// ---------------------------------------------------------------------------

use crate::ssa::types::ConstantValue;

fn const_to_inst_kind(c: &ConstantValue) -> InstKind {
    match c {
        ConstantValue::I32(v) => InstKind::Int(v.to_string()),
        ConstantValue::U32(v) => InstKind::Int(v.to_string()),
        ConstantValue::F32(bits) => InstKind::Float(f32::from_bits(*bits).to_string()),
        ConstantValue::Bool(v) => InstKind::Bool(*v),
    }
}

fn pure_to_inst_kind(op: &PureOp, args: &[ValueId]) -> InstKind {
    let vr = |i: usize| -> ValueRef { ValueRef::Ssa(args[i]) };
    match op {
        PureOp::Int(s) => InstKind::Int(s.clone()),
        PureOp::Float(s) => InstKind::Float(s.clone()),
        PureOp::Bool(b) => InstKind::Bool(*b),
        PureOp::Unit => InstKind::Unit,
        PureOp::StringLit(s) => InstKind::String(s.clone()),
        PureOp::Global(s) => InstKind::Global(s.clone()),
        PureOp::Extern(s) => InstKind::Extern(s.clone()),
        PureOp::BinOp(op_name) => InstKind::BinOp {
            op: op_name.clone(),
            lhs: vr(0),
            rhs: vr(1),
        },
        PureOp::UnaryOp(op_name) => InstKind::UnaryOp {
            op: op_name.clone(),
            operand: vr(0),
        },
        PureOp::Tuple(n) => InstKind::Tuple((0..*n).map(|i| vr(i)).collect()),
        PureOp::Vector(n) => InstKind::Vector((0..*n).map(|i| vr(i)).collect()),
        PureOp::Matrix { rows, cols } => {
            let mut mat = Vec::with_capacity(*rows);
            let mut idx = 0;
            for _ in 0..*rows {
                let row: Vec<ValueRef> = (0..*cols)
                    .map(|_| {
                        let v = vr(idx);
                        idx += 1;
                        v
                    })
                    .collect();
                mat.push(row);
            }
            InstKind::Matrix(mat)
        }
        PureOp::ArrayLit(n) => InstKind::ArrayLit {
            elements: (0..*n).map(|i| vr(i)).collect(),
        },
        PureOp::ArrayRange { has_step } => InstKind::ArrayRange {
            start: vr(0),
            len: vr(1),
            step: if *has_step { Some(vr(2)) } else { None },
        },
        PureOp::Project { index } => InstKind::Project {
            base: vr(0),
            index: *index,
        },
        PureOp::Index => InstKind::Index {
            base: vr(0),
            index: vr(1),
        },
        PureOp::Materialize => InstKind::Materialize { value: vr(0) },
        PureOp::DynamicExtract => InstKind::DynamicExtract {
            base: vr(0),
            index: vr(1),
        },
    }
}

/// Rebuild an effectful InstKind from the original kind and new operands.
fn rebuild_effectful_inst_kind(original: &InstKind, operands: &[ValueId]) -> InstKind {
    let mut result = original.clone();
    let mut idx = 0;
    result.substitute_values(&mut |vr| {
        *vr = ValueRef::Ssa(operands[idx]);
        idx += 1;
    });
    result
}
