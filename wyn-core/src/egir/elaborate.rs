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
use crate::pipeline_descriptor::PipelineDescriptor;
use crate::ssa::builder::FuncBuilder;
use crate::ssa::framework::BlockId as SkelBlockId;
use crate::ssa::types::{
    BlockId, ControlHeader, EntryPoint, FuncBody, Function, InstKind, PlaceId, Program, ValueId, ValueRef,
};
use crate::LookupMap;
use polytype::Type;
use smallvec::SmallVec;

use super::extract;
use super::loop_analysis::LoopAnalysis;
use super::program::{PhysicalEGraph, PhysicalProgram, PhysicalPureOp, PhysicalSideEffect};
use super::scoped_map::ScopedMap;
use super::types::*;

/// Lower the whole EGIR program to SSA. Each per-body EGraph is
/// elaborated to a `FuncBody`, externs pass through, and the result is
/// assembled into a single `ssa::types::Program`. The pipeline
/// descriptor passes through unchanged.
pub fn run_program(inner: PhysicalProgram) -> (Program, PipelineDescriptor) {
    let functions: Vec<Function> = inner
        .functions
        .into_iter()
        .map(|f| {
            let body = elaborate_one_body(f.graph, &f.control_headers, &f.aliases, &f.params, f.return_ty);
            Function {
                name: f.name,
                body,
                span: f.span,
                linkage_name: f.linkage_name,
            }
        })
        .chain(inner.externs.into_iter())
        .collect();

    let entry_points: Vec<EntryPoint> = inner
        .entry_points
        .into_iter()
        .map(|e| {
            let body = elaborate_one_body(e.graph, &e.control_headers, &e.aliases, &e.params, e.return_ty);
            EntryPoint {
                name: e.name,
                body,
                execution_model: e.execution_model,
                inputs: e.inputs,
                outputs: e.outputs,
                storage_bindings: e.storage_bindings,
                span: e.span,
            }
        })
        .collect();

    let program = Program {
        functions,
        entry_points,
        constants: inner.constants,
    };
    (program, inner.pipeline)
}

fn elaborate_one_body(
    graph: PhysicalEGraph,
    control_headers: &LookupMap<BlockId, ControlHeader>,
    aliases: &LookupMap<NodeId, NodeId>,
    params: &[(Type<TypeName>, String)],
    return_ty: Type<TypeName>,
) -> FuncBody {
    let skel_domtree = skeleton_domtree(&graph.skeleton);
    let identity_map: LookupMap<BlockId, BlockId> = graph.skeleton.blocks.keys().map(|b| (b, b)).collect();
    run(
        &graph,
        &skel_domtree,
        params,
        return_ty,
        control_headers,
        &identity_map,
        aliases,
    )
}

pub(super) fn skeleton_domtree<P: super::types::EgirPhase>(
    skeleton: &Skeleton<P>,
) -> wyn_graph::DominatorTree<SkelBlockId> {
    wyn_graph::DominatorTree::build(skeleton.entry, |block, successors| {
        match &skeleton.blocks[block].term {
            SkeletonTerminator::Return(_) | SkeletonTerminator::Unreachable => {}
            SkeletonTerminator::Branch { target, .. } => successors.push(*target),
            SkeletonTerminator::CondBranch {
                then_target,
                else_target,
                ..
            } => successors.extend([*then_target, *else_target]),
        }
    })
}

/// Elaborate an EGraph back into a FuncBody.
///
/// `aliases` maps block-param NodeIds that were stripped by skeleton
/// rewrites (see `skel_opt`) to their replacement NodeIds. These are
/// merged into the extraction's `best` map so any incidental demand of
/// a stripped param — e.g., via a Pure node hash-consed with the param
/// in its operands — transparently redirects to the replacement.
pub fn run(
    graph: &PhysicalEGraph,
    domtree: &wyn_graph::DominatorTree<SkelBlockId>,
    params: &[(Type<TypeName>, String)],
    return_ty: Type<TypeName>,
    control_headers: &LookupMap<BlockId, ControlHeader>,
    orig_block_map: &LookupMap<BlockId, SkelBlockId>,
    aliases: &LookupMap<NodeId, NodeId>,
) -> FuncBody {
    // Phase 1: cost-based extraction.
    let mut best = extract::extract(graph);
    // Merge skel_opt aliases into `best` preserving transitivity: if X is
    // itself in `best`, follow through so resolve(param) lands on the
    // final representative.
    for (&k, &v) in aliases {
        let target = best.get(&v).copied().unwrap_or(v);
        best.insert(k, target);
    }

    // Loop analysis over the skeleton, used by LICM placement.
    let loop_analysis = LoopAnalysis::build(&graph.skeleton, control_headers);

    // Phase 2: set up elaborator.
    let mut elab = Elaborator {
        graph,
        best,
        domtree,
        loop_analysis: &loop_analysis,
        loop_stack: SmallVec::new(),
        elaborated: ScopedMap::new(),
        elaborated_places: ScopedMap::new(),
        builder: FuncBuilder::new(params.to_vec(), return_ty),
        block_map: LookupMap::new(),
        current_block: None,
        current_skel_block: None,
    };

    // Map function params: NodeId → (ValueId, skel entry block).
    let skel_entry = graph.skeleton.entry;
    for i in 0..elab.builder.num_params() {
        let vid = elab.builder.get_param(i);
        for (nid, node) in &graph.nodes {
            if matches!(node, ENode::FuncParam { index } if *index == i) {
                let resolved = elab.resolve(nid);
                elab.elaborated.insert(resolved, (vid, skel_entry));
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
            elab.elaborated.insert(resolved, (vid, skel_bid));
        }
    }

    // Map control headers from original blocks → output blocks.
    let skel_to_output = &elab.block_map;
    let mut reverse_orig_map: LookupMap<SkelBlockId, BlockId> = LookupMap::new();
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

/// Stack frame for each loop we're currently inside (innermost at top).
struct LoopStackEntry {
    /// Skeleton block that is the loop header.
    header: SkelBlockId,
    /// Skeleton block to hoist loop-invariants into (the header's idom).
    hoist_block: SkelBlockId,
    /// ScopedMap depth at which this loop's body scope was pushed.
    scope_depth: usize,
}

struct Elaborator<'a> {
    graph: &'a PhysicalEGraph,
    best: LookupMap<NodeId, NodeId>,
    domtree: &'a wyn_graph::DominatorTree<SkelBlockId>,
    loop_analysis: &'a LoopAnalysis,
    loop_stack: SmallVec<[LoopStackEntry; 4]>,
    /// NodeId → (ValueId, skeleton block where it was placed) for
    /// value-producing nodes.
    elaborated: ScopedMap<NodeId, (ValueId, SkelBlockId)>,
    /// NodeId → (PlaceId, skeleton block) for place-producing nodes
    /// (`ViewIndex`, `OutputSlot`). Separate from `elaborated` because
    /// places are not interchangeable with values, and because identity
    /// matters: two hashconsed `ViewIndex` nodes still get distinct places
    /// when demanded from unrelated scopes (the `ScopedMap` already
    /// handles that via its scope-depth pop).
    elaborated_places: ScopedMap<NodeId, (PlaceId, SkelBlockId)>,
    builder: FuncBuilder,
    block_map: LookupMap<SkelBlockId, BlockId>,
    current_block: Option<BlockId>,
    current_skel_block: Option<SkelBlockId>,
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
        self.elaborated_places.push_scope();
        let pushed_loop = self.maybe_push_loop(skel_bid);

        let out_bid = self.block_map[&skel_bid];
        self.current_block = Some(out_bid);
        self.current_skel_block = Some(skel_bid);
        self.builder.switch_to_block_unchecked(out_bid);

        let skel_block = self.graph.skeleton.blocks[skel_bid].clone();

        // Elaborate side-effectful instructions.
        for se in &skel_block.side_effects {
            self.elaborate_side_effect(se, skel_bid);
        }

        // Elaborate terminator.
        self.elaborate_terminator(&skel_block.term);

        // Recurse into domtree children. Each child switches to its own block
        // on entry, so we re-set ours before emitting our terminator above, and
        // the child loop here just handles descent.
        let children: Vec<SkelBlockId> = self.domtree.children(skel_bid).to_vec();
        for child in children {
            self.elaborate_subtree(child);
        }

        if pushed_loop {
            self.loop_stack.pop();
        }
        self.elaborated_places.pop_scope();
        self.elaborated.pop_scope();
    }

    /// Push a loop stack frame if `skel_bid` is a loop header.
    fn maybe_push_loop(&mut self, skel_bid: SkelBlockId) -> bool {
        if !self.loop_analysis.is_header(skel_bid) {
            return false;
        }
        let hoist_block = self
            .domtree
            .idom(skel_bid)
            .expect("loop header should have an immediate dominator (the preheader)");
        let scope_depth = self.elaborated.depth();
        self.loop_stack.push(LoopStackEntry {
            header: skel_bid,
            hoist_block,
            scope_depth,
        });
        true
    }

    /// Elaborate a side-effectful instruction. Side effects stay pinned to
    /// their containing skeleton block — only the operands go through
    /// demand() where LICM may move them.
    fn elaborate_side_effect(&mut self, se: &PhysicalSideEffect, skel_bid: SkelBlockId) {
        let inst_kind = match &se.kind {
            super::types::SideEffectKind::Inst(k) => k,
            super::types::SideEffectKind::Soac(p) => {
                panic!("elaborate: unexpanded EgirSoac in skeleton: {:?}", p)
            }
        };

        // Load/Store carry a PlaceId operand in `operand_nodes[0]` rather
        // than a value; handle them explicitly so the place operand stays
        // typed as `PlaceId`, not a ValueId. Alloca produces a PlaceId rather
        // than a ValueId — register it in `elaborated_places` so downstream
        // `PlaceIndex` / `Load` / `Store` consumers resolve it via `demand_place`.
        if let InstKind::Alloca { elem_ty, .. } = inst_kind {
            let result_nid =
                se.result.expect("Alloca side-effect must carry a result NodeId for its place");
            let place = self.builder.new_place(elem_ty.clone());
            let kind = InstKind::Alloca {
                elem_ty: elem_ty.clone(),
                result: place,
            };
            let out_bid = self.block_map[&skel_bid];
            self.builder.func_mut().append_void_inst_with_span(out_bid, kind, se.span);
            let resolved = self.resolve(result_nid);
            self.elaborated_places.insert(resolved, (place, skel_bid));
            return;
        }

        let kind = match inst_kind {
            InstKind::Load { .. } => {
                let place = self.demand_place(se.operand_nodes[0]);
                InstKind::Load { place }
            }
            InstKind::Store { .. } => {
                let place = self.demand_place(se.operand_nodes[0]);
                let value = self.demand(se.operand_nodes[1]);
                InstKind::Store {
                    place,
                    value: ValueRef::Ssa(value),
                }
            }
            _ => {
                let args: Vec<ValueId> = se.operand_nodes.iter().map(|&nid| self.demand(nid)).collect();
                rebuild_effectful_inst_kind(inst_kind, &args)
            }
        };

        if let Some(result_nid) = se.result {
            let ty = self.graph.types[&result_nid].clone();
            let vid = self.emit_at(skel_bid, kind, ty, se.span);
            // Insert under the resolved id so demand_placed's `self.resolve(nid)
            // → get(&resolved)` path finds it. Today extract maps every
            // SideEffectResult to itself, but that's a brittle invariant —
            // this keeps the insert symmetric with every other site.
            let resolved = self.resolve(result_nid);
            self.elaborated.insert(resolved, (vid, skel_bid));
        } else {
            let out_bid = self.block_map[&skel_bid];
            self.builder.func_mut().append_void_inst_with_span(out_bid, kind, se.span);
        }
    }

    /// Demand-driven elaboration: given a NodeId, produce a ValueId.
    fn demand(&mut self, nid: NodeId) -> ValueId {
        self.demand_placed(nid).0
    }

    /// Demand the place defined by `nid`. Only valid for nodes whose
    /// `PureOp` produces a `PlaceId` (`ViewIndex`, `OutputSlot`).
    fn demand_place(&mut self, nid: NodeId) -> PlaceId {
        let resolved = self.resolve(nid);

        if let Some((place, _)) = self.elaborated_places.get(&resolved) {
            return place;
        }

        let node = self.graph.nodes[resolved].clone();
        let ENode::Pure { op, operands } = &node else {
            panic!(
                "demand_place({:?}): expected a place-producing Pure node, got {:?}",
                resolved, node
            );
        };

        let (kind, placed) = match op {
            PureOp::ViewIndex => {
                let arg_placements: Vec<(ValueId, SkelBlockId)> =
                    operands.iter().map(|&op_nid| self.demand_placed(op_nid)).collect();
                let args: Vec<ValueId> = arg_placements.iter().map(|&(v, _)| v).collect();
                let elem_ty = self.graph.types[&resolved].clone();
                let place = self.builder.new_place(elem_ty);
                let kind = InstKind::ViewIndex {
                    view: ValueRef::Ssa(args[0]),
                    index: ValueRef::Ssa(args[1]),
                    result: place,
                };
                let placed = self.choose_placement(&arg_placements);
                (kind, placed)
            }
            PureOp::PlaceIndex => {
                // operands[0] is the parent place (resolved via demand_place),
                // operands[1] is the index value (resolved via demand_placed).
                let parent_place = self.demand_place(operands[0]);
                let (index_val, index_placed) = self.demand_placed(operands[1]);
                let elem_ty = self.graph.types[&resolved].clone();
                let place = self.builder.new_place(elem_ty);
                let kind = InstKind::PlaceIndex {
                    place: parent_place,
                    index: ValueRef::Ssa(index_val),
                    result: place,
                };
                // Place this with the index's placement so it follows the
                // control-flow point where the index becomes available.
                let placed = self.choose_placement(&[(index_val, index_placed)]);
                (kind, placed)
            }
            PureOp::OutputSlot { index } => {
                let elem_ty = self.graph.types[&resolved].clone();
                let place = self.builder.new_place(elem_ty);
                let kind = InstKind::OutputSlot {
                    index: *index,
                    result: place,
                };
                let placed = self.choose_placement(&[]);
                (kind, placed)
            }
            other => panic!(
                "demand_place({:?}): {:?} does not produce a place",
                resolved, other
            ),
        };

        let place = match &kind {
            InstKind::ViewIndex { result, .. }
            | InstKind::PlaceIndex { result, .. }
            | InstKind::OutputSlot { result, .. } => *result,
            _ => unreachable!(),
        };
        let span = self.graph.node_spans.get(&resolved).copied();
        let out_bid = self.block_map[&placed];
        self.builder.func_mut().append_void_inst_with_span(out_bid, kind, span);
        self.elaborated_places.insert(resolved, (place, placed));
        place
    }

    /// Demand a node and return both the ValueId and the skeleton block it
    /// was placed in.
    fn demand_placed(&mut self, nid: NodeId) -> (ValueId, SkelBlockId) {
        let resolved = self.resolve(nid);

        if let Some(entry) = self.elaborated.get(&resolved) {
            return entry;
        }

        let node = self.graph.nodes[resolved].clone();

        match &node {
            ENode::Constant(c) => {
                let ty = self.graph.types[&resolved].clone();
                let kind = const_to_inst_kind(c);
                let placed = self.choose_placement(&[]);
                let span = self.graph.node_spans.get(&resolved).copied();
                let vid = self.emit_at(placed, kind, ty, span);
                self.record_placement(resolved, vid, placed);
                (vid, placed)
            }
            ENode::Pure { op, operands } => {
                if matches!(
                    op,
                    PureOp::ViewIndex | PureOp::PlaceIndex | PureOp::OutputSlot { .. }
                ) {
                    panic!(
                        "demand_placed({:?}): {:?} produces a PlaceId, not a ValueId — \
                         its consumer (Load/Store/etc.) must call demand_place",
                        resolved, op
                    );
                }
                let arg_placements: Vec<(ValueId, SkelBlockId)> =
                    operands.iter().map(|&op_nid| self.demand_placed(op_nid)).collect();
                let args: Vec<ValueId> = arg_placements.iter().map(|&(v, _)| v).collect();

                let ty = self.graph.types[&resolved].clone();
                let kind = pure_to_inst_kind(op, &args);
                let placed = self.choose_placement(&arg_placements);
                let span = self.graph.node_spans.get(&resolved).copied();
                let vid = self.emit_at(placed, kind, ty, span);
                self.record_placement(resolved, vid, placed);
                (vid, placed)
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

    /// Decide where to place a pure node given the skeleton blocks where its
    /// operands live. Walks the loop stack innermost→outermost and hoists
    /// out of every enclosing loop whose body contains none of the operands.
    fn choose_placement(&self, operand_blocks: &[(ValueId, SkelBlockId)]) -> SkelBlockId {
        let current = self.current_skel_block.expect("current skel block unset");
        let mut candidate = current;
        // Active loops are the enclosing loops whose body contains the
        // current block, innermost first.
        let active: SmallVec<[&LoopStackEntry; 4]> = self
            .loop_stack
            .iter()
            .rev()
            .filter(|f| self.loop_analysis.is_in_loop(current, f.header))
            .collect();
        // Walk outward one loop at a time. A pure node can hoist past a
        // loop whose body contains no operand of the node — the operand's
        // placement, by induction, is already at least as outer as the
        // loop's preheader. Stop climbing the moment a loop holds an
        // operand inside it (the node has to stay inside that loop to
        // reference the operand). Textual-scope backends depend on this
        // uniformly for all pure nodes, including nullary ones: the
        // resulting placement is the LCA of the node's uses, which by
        // construction dominates every use site.
        for frame in active.iter() {
            let any_inside =
                operand_blocks.iter().any(|&(_, b)| self.loop_analysis.is_in_loop(b, frame.header));
            if any_inside {
                break;
            }
            candidate = frame.hoist_block;
        }
        candidate
    }

    /// Record an elaborated node. If the placement is at an outer scope (a
    /// loop's hoist_block), insert the binding at that loop's scope_depth so
    /// it remains visible to siblings inside the loop body but scopes out
    /// with the loop frame.
    fn record_placement(&mut self, nid: NodeId, vid: ValueId, placed: SkelBlockId) {
        let current = self.current_skel_block.expect("current skel block unset");
        // Only consider active loop frames (same filter as choose_placement).
        let insert_depth = self
            .loop_stack
            .iter()
            .rev()
            .filter(|f| self.loop_analysis.is_in_loop(current, f.header))
            .find(|f| f.hoist_block == placed)
            .map(|f| f.scope_depth);
        if let Some(d) = insert_depth {
            self.elaborated.insert_at_depth(d, nid, (vid, placed));
        } else {
            self.elaborated.insert(nid, (vid, placed));
        }
    }

    /// Emit an instruction into `target_skel`'s output block, bypassing the
    /// "block already terminated" check. The insts list is stored separately
    /// from the terminator, so appending is still well-formed.
    fn emit_at(
        &mut self,
        target_skel: SkelBlockId,
        kind: InstKind,
        ty: Type<TypeName>,
        span: Option<crate::ast::Span>,
    ) -> ValueId {
        let out_bid = self.block_map[&target_skel];
        self.builder.func_mut().append_inst_with_span(out_bid, kind, ty, span)
    }

    /// Resolve a NodeId through the extraction map.
    fn resolve(&self, nid: NodeId) -> NodeId {
        self.best.get(&nid).copied().unwrap_or(nid)
    }

    /// Elaborate a skeleton terminator.
    fn elaborate_terminator(&mut self, term: &SkeletonTerminator) {
        let t = match term {
            SkeletonTerminator::Return(None) => crate::ssa::framework::Terminator::Return(None),
            SkeletonTerminator::Return(Some(nid)) => {
                crate::ssa::framework::Terminator::Return(Some(self.demand(*nid)))
            }
            SkeletonTerminator::Branch { target, args } => {
                let out_args: Vec<ValueId> = args.iter().map(|&nid| self.demand(nid)).collect();
                crate::ssa::framework::Terminator::Branch {
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
                crate::ssa::framework::Terminator::CondBranch {
                    cond: cond_vid,
                    then_target: self.block_map[then_target],
                    then_args: ta,
                    else_target: self.block_map[else_target],
                    else_args: ea,
                }
            }
            SkeletonTerminator::Unreachable => crate::ssa::framework::Terminator::Unreachable,
        };
        let _ = self.builder.terminate(t);
    }
}

// ---------------------------------------------------------------------------
// Conversion: ENode → InstKind
// ---------------------------------------------------------------------------

use crate::ssa::types::ConstantValue;

fn const_to_inst_kind(c: &ConstantValue) -> InstKind {
    use crate::op::OpTag;
    let tag = match c {
        ConstantValue::I32(v) => OpTag::Int(v.to_string()),
        ConstantValue::U32(v) => OpTag::Uint(v.to_string()),
        ConstantValue::F32(bits) => OpTag::Float(f32::from_bits(*bits).to_string()),
        ConstantValue::Bool(v) => OpTag::Bool(*v),
    };
    InstKind::Op {
        tag,
        operands: vec![],
    }
}

fn pure_to_inst_kind(op: &PhysicalPureOp, args: &[ValueId]) -> InstKind {
    super::types::rebuild_inst_kind(op, args)
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
