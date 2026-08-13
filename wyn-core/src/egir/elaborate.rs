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
use crate::flow::BlockId as SkelBlockId;
use crate::op::OpTag;
use crate::pipeline_descriptor::PipelineDescriptor;
use crate::ssa::builder::FuncBuilder;
use crate::ssa::types::{
    BlockId, Constant, ControlHeader, EntryPoint, FuncBody, Function, InstKind, PlaceId, Program, ValueId,
    ValueRef,
};
use crate::types::{ExternDecl, TypeExt};
use crate::{BindingRef, EntryId, LookupMap, LookupSet, ResourceAccess};
use polytype::Type;
use smallvec::SmallVec;

use super::extract;
use super::loop_analysis::LoopAnalysis;
use super::program::{PhysicalEGraph, PhysicalPureOp, PhysicalSideEffect, PlannedGlobal};
use super::scoped_map::ScopedMap;
use super::types::*;

/// Lower the whole EGIR program to SSA. Each per-body EGraph is
/// elaborated to a `FuncBody`, externs pass through, and the result is
/// assembled into a backend-bound SSA program.
pub fn elaborate(inner: super::resource_erasure::ResourcesErased) -> crate::ssa::stage::Elaborated {
    let super::ir::Program {
        functions,
        externs,
        entry_points,
        constants,
        data,
        global_context,
        state: _,
    } = inner;
    let pipeline = data.pipeline;
    let pipeline_storage_accesses = pipeline_storage_accesses(&pipeline, &data.stage_entries);
    let functions: Vec<Function> = functions
        .into_iter()
        .map(|f| {
            let body = elaborate_one_body(f.graph, &f.params, f.return_ty);
            Function {
                id: f.region,
                name: f.name,
                body,
                span: f.span,
                linkage_name: f.linkage_name,
            }
        })
        .chain(externs.into_iter().map(elaborate_extern))
        .collect();

    let entry_points: Vec<EntryPoint> = entry_points
        .into_iter()
        .map(|e| {
            let body = elaborate_one_body(e.graph, &e.params, e.return_ty);
            let entry_pipeline_accesses = pipeline_storage_accesses.get(&e.id).cloned().unwrap_or_default();
            EntryPoint {
                id: e.id,
                name: e.name,
                body,
                execution_model: e.execution_model,
                inputs: e.inputs.into_iter().map(|input| input.inner).collect(),
                parameter_inputs: e
                    .parameter_inputs
                    .into_iter()
                    .map(|slots| slots.into_iter().map(|slot| slot.0).collect())
                    .collect(),
                outputs: e.outputs.into_iter().map(|output| output.inner).collect(),
                storage_bindings: e.resource_declarations,
                pipeline_storage_accesses: entry_pipeline_accesses,
                span: e.span,
            }
        })
        .collect();

    let constants = constants
        .into_iter()
        .map(|constant| Constant {
            id: constant.id,
            name: constant.name,
            body: elaborate_one_body(constant.graph, &[], constant.return_ty),
        })
        .collect();
    let program = Program::bare(functions, entry_points, constants);
    let PlannedGlobal {
        kernel_plan,
        profile,
        effect_ids: _,
        semantic_ids: _,
    } = global_context;
    program.with_context::<crate::ssa::stage::ElaboratedTag, _>(crate::ssa::context::BackendGlobal {
        pipeline,
        profile,
        kernel_plan,
    })
}

/// The descriptor binding access is a physical pipeline-layout property: all
/// stages in one pipeline share it even when an individual stage only reads or
/// only writes the slot. Preserve that separately from per-entry usage before
/// elaboration consumes the physical program.
fn pipeline_storage_accesses(
    descriptor: &PipelineDescriptor,
    stage_entries: &[Vec<EntryId>],
) -> LookupMap<EntryId, LookupMap<BindingRef, ResourceAccess>> {
    use crate::pipeline_descriptor::{Access, Binding, Pipeline};

    let mut entries = LookupMap::new();
    for (pipeline_index, pipeline) in descriptor.pipelines.iter().enumerate() {
        let bindings: &[Binding] = match pipeline {
            Pipeline::Compute(compute) => &compute.bindings,
            Pipeline::Graphics(graphics) => &graphics.bindings,
        };
        let layout = bindings
            .iter()
            .filter_map(|binding| {
                let Binding::StorageBuffer {
                    set, binding, access, ..
                } = binding
                else {
                    return None;
                };
                let access = match access {
                    Access::ReadOnly => ResourceAccess::Read,
                    Access::WriteOnly => ResourceAccess::Write,
                    Access::ReadWrite => ResourceAccess::ReadWrite,
                };
                Some((BindingRef::new(*set, *binding), access))
            })
            .collect::<LookupMap<_, _>>();
        for &entry_id in stage_entries.get(pipeline_index).into_iter().flatten() {
            let entry = entries.entry(entry_id).or_insert_with(LookupMap::new);
            for (&binding, &access) in &layout {
                entry
                    .entry(binding)
                    .and_modify(|current: &mut ResourceAccess| *current = current.merge(access))
                    .or_insert(access);
            }
        }
    }
    entries
}

fn elaborate_extern(declaration: ExternDecl<Type<TypeName>>) -> Function {
    let body = FuncBuilder::new(declaration.params, declaration.return_ty).finish_unchecked();
    Function {
        id: declaration.id,
        name: declaration.name,
        body,
        span: declaration.span,
        linkage_name: Some(declaration.linkage_name),
    }
}

pub(super) fn elaborate_one_body(
    graph: PhysicalEGraph,
    params: &[(Type<TypeName>, String)],
    return_ty: Type<TypeName>,
) -> FuncBody {
    let skel_domtree = skeleton_domtree(&graph.skeleton);
    elaborate_graph(&graph, &skel_domtree, params, return_ty)
}

pub(super) fn skeleton_domtree<P: super::types::Family>(
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
/// Canonical aliases owned by graph nodes are merged into extraction's `best`
/// map so incidental demands of stripped block parameters transparently
/// redirect to their replacements.
pub fn elaborate_graph(
    graph: &PhysicalEGraph,
    domtree: &wyn_graph::DominatorTree<SkelBlockId>,
    params: &[(Type<TypeName>, String)],
    return_ty: Type<TypeName>,
) -> FuncBody {
    // Phase 1: cost-based extraction.
    let mut best = extract::extract(graph);
    close_extraction_over_aliases(graph, &mut best);

    // Loop analysis over the skeleton, used by LICM placement.
    let loop_analysis = LoopAnalysis::build(&graph.skeleton);

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
            if matches!(&node.kind, ENode::FuncParam { index } if *index == i) {
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
            let ty = graph.nodes[param_nid].ty.clone();
            let vid = elab.builder.add_block_param(out_bid, ty);
            let resolved = elab.resolve(param_nid);
            elab.elaborated.insert(resolved, (vid, skel_bid));
        }
    }

    // Map graph-owned control headers to output blocks.
    let skel_to_output = &elab.block_map;
    for (&skel_bid, &out_bid) in skel_to_output.iter() {
        if let Some(header) = &graph.skeleton.blocks[skel_bid].control_header {
            let mapped = match header {
                ControlHeader::Selection { merge } => ControlHeader::Selection {
                    merge: skel_to_output[merge],
                },
                ControlHeader::Loop {
                    merge,
                    continue_block,
                } => ControlHeader::Loop {
                    merge: skel_to_output[merge],
                    continue_block: skel_to_output[continue_block],
                },
            };
            elab.builder.set_control_header(out_bid, mapped);
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
        let effect = match &se.kind {
            super::types::SideEffectKind::Effect(effect) => effect,
            super::types::SideEffectKind::Soac(SoacEffect(_, p)) => {
                panic!("elaborate: unexpanded EgirSoac in skeleton: {:?}", p)
            }
        };

        // Load/Store carry a PlaceId operand in `operand_nodes[0]` rather
        // than a value; handle them explicitly so the place operand stays
        // typed as `PlaceId`, not a ValueId. Alloca produces a PlaceId rather
        // than a ValueId — register it in `elaborated_places` so downstream
        // `PlaceIndex` / `Load` / `Store` consumers resolve it via `demand_place`.
        if let EffectOp::Alloca { elem_ty } = effect {
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

        let kind = match effect {
            EffectOp::Load => {
                let place = self.demand_place(se.operand_nodes[0]);
                InstKind::Load { place }
            }
            EffectOp::Store => {
                let place = self.demand_place(se.operand_nodes[0]);
                let value = self.demand_ref(se.operand_nodes[1]);
                InstKind::Store { place, value }
            }
            EffectOp::Atomic(op) => {
                let place = self.demand_place(se.operand_nodes[0]);
                let values = se.operand_nodes[1..].iter().map(|&node| self.demand_ref(node)).collect();
                InstKind::Atomic {
                    place,
                    op: *op,
                    values,
                }
            }
            EffectOp::Op { tag } => {
                let operands = se.operand_nodes.iter().map(|&nid| self.demand_ref(nid)).collect();
                InstKind::Op {
                    tag: tag.clone(),
                    operands,
                }
            }
            EffectOp::ControlBarrier => InstKind::ControlBarrier,
            EffectOp::Alloca { .. } => unreachable!("alloca handled above"),
        };

        if let Some(result_nid) = se.result {
            let ty = self.graph.nodes[result_nid].ty.clone();
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

    /// Demand a node as an instruction operand. Canonical 32-bit scalar
    /// constants stay inline instead of being expanded into nullary SSA
    /// instructions; values that need an SSA identity use `demand_placed`.
    fn demand_ref(&mut self, nid: NodeId) -> ValueRef {
        self.demand_ref_placed(nid).0
    }

    /// Operand form plus the block that defines an SSA operand. Constants
    /// have no placement dependency and therefore carry `None`.
    fn demand_ref_placed(&mut self, nid: NodeId) -> (ValueRef, Option<SkelBlockId>) {
        let resolved = self.resolve(nid);
        if let Some(value) = self.inline_constant(resolved) {
            return (ValueRef::Const(value), None);
        }
        let (value, block) = self.demand_placed(resolved);
        (ValueRef::Ssa(value), Some(block))
    }

    /// Convert only constants whose type is fully represented by
    /// `ConstantValue`. Wider/narrower numeric literals retain an SSA result
    /// so their explicit result type is not lost.
    fn inline_constant(&self, nid: NodeId) -> Option<ConstantValue> {
        let node = &self.graph.nodes[nid];
        match (&node.kind, &node.ty) {
            (ENode::Constant(value @ ConstantValue::I32(_)), Type::Constructed(TypeName::Int(32), _))
            | (ENode::Constant(value @ ConstantValue::U32(_)), Type::Constructed(TypeName::UInt(32), _))
            | (ENode::Constant(value @ ConstantValue::F32(_)), Type::Constructed(TypeName::Float(32), _))
            | (ENode::Constant(value @ ConstantValue::Bool(_)), Type::Constructed(TypeName::Bool, _)) => {
                Some(*value)
            }
            (
                ENode::Pure {
                    op: PureOp::Int(value),
                    operands,
                },
                Type::Constructed(TypeName::Int(32), _),
            ) if operands.is_empty() => value.parse().ok().map(ConstantValue::I32),
            (
                ENode::Pure {
                    op: PureOp::Uint(value),
                    operands,
                },
                Type::Constructed(TypeName::UInt(32), _),
            ) if operands.is_empty() => value.parse().ok().map(ConstantValue::U32),
            (
                ENode::Pure {
                    op: PureOp::Float(value),
                    operands,
                },
                Type::Constructed(TypeName::Float(32), _),
            ) if operands.is_empty() => value.parse().ok().map(ConstantValue::from_f32),
            (
                ENode::Pure {
                    op: PureOp::Bool(value),
                    operands,
                },
                Type::Constructed(TypeName::Bool, _),
            ) if operands.is_empty() => Some(ConstantValue::Bool(*value)),
            _ => None,
        }
    }

    /// Demand the place defined by `nid`. Only valid for nodes whose
    /// `PureOp` produces a `PlaceId` (`ViewIndex`, `OutputSlot`).
    fn demand_place(&mut self, nid: NodeId) -> PlaceId {
        let resolved = self.resolve(nid);

        if let Some((place, _)) = self.elaborated_places.get(&resolved) {
            return place;
        }

        let node = self.graph.nodes[resolved].clone();
        let ENode::Pure { op, operands } = &node.kind else {
            panic!(
                "demand_place({:?}): expected a place-producing Pure node, got {:?}",
                resolved, node
            );
        };

        let (kind, placed) = match op {
            PureOp::ViewIndex => {
                let arg_placements: Vec<(ValueRef, Option<SkelBlockId>)> =
                    operands.iter().map(|&op_nid| self.demand_ref_placed(op_nid)).collect();
                let args: Vec<ValueRef> = arg_placements.iter().map(|&(value, _)| value).collect();
                let elem_ty = self.graph.nodes[resolved].ty.clone();
                let place = self.builder.new_place(elem_ty);
                let kind = InstKind::ViewIndex {
                    view: args[0],
                    index: args[1],
                    result: place,
                };
                let operand_blocks: Vec<_> =
                    arg_placements.iter().filter_map(|(_, block)| *block).collect();
                let placed = self.choose_placement(&operand_blocks);
                (kind, placed)
            }
            PureOp::PlaceIndex => {
                // operands[0] is the parent place (resolved via demand_place),
                // operands[1] is the index value (resolved via demand_placed).
                let parent_place = self.demand_place(operands[0]);
                let (index, index_placed) = self.demand_ref_placed(operands[1]);
                let elem_ty = self.graph.nodes[resolved].ty.clone();
                let place = self.builder.new_place(elem_ty);
                let kind = InstKind::PlaceIndex {
                    place: parent_place,
                    index,
                    result: place,
                };
                // Place this with the index's placement so it follows the
                // control-flow point where the index becomes available.
                let placed = self.choose_placement(&index_placed.into_iter().collect::<Vec<_>>());
                (kind, placed)
            }
            PureOp::OutputSlot { index } => {
                let elem_ty = self.graph.nodes[resolved].ty.clone();
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
        let span = self.graph.nodes[resolved].span;
        let out_bid = self.block_map[&placed];
        self.builder.func_mut().append_void_inst_with_span(out_bid, kind, span);
        self.elaborated_places.insert(resolved, (place, placed));
        place
    }

    /// Recover a direct `Index(...Index(view, i)..., j)` spine after
    /// representation specialization has changed its root to a storage view.
    /// Each entry is `(index-result node, coordinate node)` in base-to-leaf
    /// order. Stopping at the first view-typed base preserves any value
    /// operations that precede the storage address chain.
    fn view_index_spine(&self, nid: NodeId) -> Option<(NodeId, SmallVec<[(NodeId, NodeId); 4]>)> {
        let mut steps = SmallVec::<[(NodeId, NodeId); 4]>::new();
        let mut current = self.resolve(nid);
        loop {
            let ENode::Pure {
                op: PureOp::Index,
                operands,
            } = &self.graph.nodes[current].kind
            else {
                return None;
            };
            if operands.len() != 2 {
                return None;
            }
            let base = self.resolve(operands[0]);
            let index = self.resolve(operands[1]);
            steps.push((current, index));
            if self.graph.nodes[base].ty.array_variant().is_some_and(crate::types::is_array_variant_view) {
                steps.reverse();
                return Some((base, steps));
            }
            current = base;
        }
    }

    /// Elaborate a view-rooted index spine as one address calculation followed
    /// by one leaf load. This is the late counterpart of TLC index-spine
    /// lowering: it handles helper parameters that become views only after
    /// call/capture representation reconciliation.
    fn demand_view_index_spine(
        &mut self,
        view_node: NodeId,
        steps: SmallVec<[(NodeId, NodeId); 4]>,
    ) -> (ValueId, SkelBlockId) {
        let result_node = steps.last().expect("view index spine has at least one coordinate").0;
        let (view, view_placed) = self.demand_ref_placed(view_node);
        let index_values = steps
            .iter()
            .map(|(_, index)| self.demand_ref_placed(*index))
            .collect::<SmallVec<[(ValueRef, Option<SkelBlockId>); 4]>>();
        let mut operand_placements = SmallVec::<[SkelBlockId; 4]>::new();
        operand_placements.extend(view_placed);
        operand_placements.extend(index_values.iter().filter_map(|(_, block)| *block));
        let placed = self.choose_placement(&operand_placements);
        let out_bid = self.block_map[&placed];

        let mut place = None;
        for ((step_node, _), (index, _)) in steps.iter().zip(&index_values) {
            let next = self.builder.new_place(self.graph.nodes[*step_node].ty.clone());
            let kind = if let Some(parent) = place {
                InstKind::PlaceIndex {
                    place: parent,
                    index: *index,
                    result: next,
                }
            } else {
                InstKind::ViewIndex {
                    view,
                    index: *index,
                    result: next,
                }
            };
            self.builder.func_mut().append_void_inst_with_span(
                out_bid,
                kind,
                self.graph.nodes[*step_node].span,
            );
            place = Some(next);
        }

        let ty = self.graph.nodes[result_node].ty.clone();
        let span = self.graph.nodes[result_node].span;
        let value = self.emit_at(
            placed,
            InstKind::Load {
                place: place.expect("view index spine produced a leaf place"),
            },
            ty,
            span,
        );
        self.record_placement(result_node, value, placed);
        (value, placed)
    }

    /// Demand a node and return both the ValueId and the skeleton block it
    /// was placed in.
    fn demand_placed(&mut self, nid: NodeId) -> (ValueId, SkelBlockId) {
        let resolved = self.resolve(nid);

        if let Some(entry) = self.elaborated.get(&resolved) {
            return entry;
        }

        let node = self.graph.nodes[resolved].clone();

        match &node.kind {
            ENode::Constant(c) => {
                let ty = self.graph.nodes[resolved].ty.clone();
                let kind = const_to_inst_kind(c);
                let placed = self.choose_placement(&[]);
                let span = self.graph.nodes[resolved].span;
                let vid = self.emit_at(placed, kind, ty, span);
                self.record_placement(resolved, vid, placed);
                (vid, placed)
            }
            ENode::Pure { op, operands } => {
                if matches!(op, PureOp::Index) {
                    if let Some((view, steps)) = self.view_index_spine(resolved) {
                        return self.demand_view_index_spine(view, steps);
                    }
                }
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
                let arg_placements: Vec<(ValueRef, Option<SkelBlockId>)> =
                    operands.iter().map(|&op_nid| self.demand_ref_placed(op_nid)).collect();
                let args: Vec<ValueRef> = arg_placements.iter().map(|&(value, _)| value).collect();

                let ty = self.graph.nodes[resolved].ty.clone();
                let kind = pure_to_inst_kind(op, &args);
                let operand_blocks: Vec<_> =
                    arg_placements.iter().filter_map(|(_, block)| *block).collect();
                let placed = self.choose_placement(&operand_blocks);
                let span = self.graph.nodes[resolved].span;
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
    fn choose_placement(&self, operand_blocks: &[SkelBlockId]) -> SkelBlockId {
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
            if !self.loop_analysis.operands_are_invariant(frame.header, operand_blocks.iter().copied()) {
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
                crate::ssa::framework::Terminator::Return(Some(self.demand_ref(*nid)))
            }
            SkeletonTerminator::Branch { target, args } => {
                let out_args: Vec<ValueRef> = args.iter().map(|&nid| self.demand_ref(nid)).collect();
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
                let cond = self.demand_ref(*cond);
                let ta: Vec<ValueRef> = then_args.iter().map(|&nid| self.demand_ref(nid)).collect();
                let ea: Vec<ValueRef> = else_args.iter().map(|&nid| self.demand_ref(nid)).collect();
                crate::ssa::framework::Terminator::CondBranch {
                    cond,
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

/// Compose extraction winners with CFG aliases to a fixed point. A union may
/// have selected a block parameter before skeleton optimization strips that
/// parameter; forwarding only the parameter itself would leave the union's
/// winner pointing at a definition that is no longer in its block signature.
fn close_extraction_over_aliases<P: Family>(graph: &EGraph<P>, best: &mut LookupMap<NodeId, NodeId>) {
    let extracted = best.clone();
    let resolve = |start| {
        let mut current = start;
        let mut seen = LookupSet::new();
        loop {
            assert!(
                seen.insert(current),
                "cycle while composing extraction and CFG aliases from {start:?}"
            );
            let chosen = extracted.get(&current).copied().unwrap_or(current);
            if chosen != current {
                current = chosen;
                continue;
            }
            let Some(alias) = graph.nodes[current].alias else {
                return current;
            };
            current = alias;
        }
    };
    for node in graph.nodes.keys() {
        best.insert(node, resolve(node));
    }
}

// ---------------------------------------------------------------------------
// Conversion: ENode → InstKind
// ---------------------------------------------------------------------------

use crate::ssa::types::ConstantValue;

fn const_to_inst_kind(c: &ConstantValue) -> InstKind {
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

fn pure_to_inst_kind(op: &PhysicalPureOp, args: &[ValueRef]) -> InstKind {
    if matches!(
        op,
        OpTag::ViewIndex | OpTag::PlaceIndex | OpTag::OutputSlot { .. }
    ) {
        panic!(
            "pure_to_inst_kind: place-producing op {:?} must use elaborate's \
             place-aware path (allocates a fresh PlaceId from FuncBody.places)",
            op
        );
    }
    InstKind::Op {
        tag: op.clone(),
        operands: args.to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ssa::types::{ConstantValue, Terminator};

    #[test]
    fn scalar_literals_stay_inline_in_ssa_operands() {
        let ty = Type::Constructed(TypeName::UInt(32), vec![]);
        let mut graph = EGraph::<Physical>::new();
        let one = graph.intern_pure(PureOp::Uint("1".into()), SmallVec::new(), ty.clone(), None);
        let two = graph.intern_pure(PureOp::Uint("2".into()), SmallVec::new(), ty.clone(), None);
        let sum = graph.intern_pure(
            PureOp::BinOp(crate::op::BinaryOperator::Add),
            smallvec::smallvec![one, two],
            ty.clone(),
            None,
        );
        graph.skeleton.blocks[graph.skeleton.entry].term = SkeletonTerminator::Return(Some(sum));

        let body = elaborate_one_body(graph, &[], ty);
        assert_eq!(
            body.num_insts(),
            1,
            "literal operands must not become SSA instructions"
        );
        let inst = body.inner.insts.values().next().expect("sum instruction");
        assert!(matches!(
            &inst.data,
            InstKind::Op {
                tag: PureOp::BinOp(crate::op::BinaryOperator::Add),
                operands,
            } if operands == &vec![
                ValueRef::Const(ConstantValue::U32(1)),
                ValueRef::Const(ConstantValue::U32(2)),
            ]
        ));
    }

    #[test]
    fn scalar_literals_stay_inline_in_ssa_terminators() {
        let ty = Type::Constructed(TypeName::UInt(32), vec![]);
        let mut graph = EGraph::<Physical>::new();
        let seven = graph.intern_pure(PureOp::Uint("7".into()), SmallVec::new(), ty.clone(), None);
        graph.skeleton.blocks[graph.skeleton.entry].term = SkeletonTerminator::Return(Some(seven));

        let body = elaborate_one_body(graph, &[], ty);
        assert_eq!(
            body.num_insts(),
            0,
            "literal return must not become an SSA instruction"
        );
        assert!(matches!(
            body.inner.blocks[body.inner.entry].term,
            Terminator::Return(Some(ValueRef::Const(ConstantValue::U32(7))))
        ));
    }

    #[test]
    fn scalar_literals_stay_inline_in_ssa_block_arguments() {
        let ty = Type::Constructed(TypeName::UInt(32), vec![]);
        let mut graph = EGraph::<Physical>::new();
        let target = graph.skeleton.create_block();
        let target_param = graph.add_block_param(target, ty.clone());
        let seven = graph.intern_pure(PureOp::Uint("7".into()), SmallVec::new(), ty.clone(), None);
        graph.skeleton.blocks[graph.skeleton.entry].term = SkeletonTerminator::Branch {
            target,
            args: vec![seven],
        };
        graph.skeleton.blocks[target].term = SkeletonTerminator::Return(Some(target_param));

        let body = elaborate_one_body(graph, &[], ty);
        assert_eq!(
            body.num_insts(),
            0,
            "literal block arguments must not become SSA instructions"
        );
        assert!(body.inner.blocks.values().any(|block| matches!(
            &block.term,
            Terminator::Branch { args, .. }
                if args == &[ValueRef::Const(ConstantValue::U32(7))]
        )));
    }

    #[test]
    fn extraction_winners_follow_eliminated_block_parameter_aliases() {
        let ty = Type::Constructed(TypeName::UInt(32), vec![]);
        let mut graph = EGraph::<Physical>::new();
        let merge = graph.skeleton.create_block();
        let selected = graph.add_block_param(merge, ty.clone());
        let replacement = graph.intern_constant(ConstantValue::U32(7), ty.clone());
        let call = graph.intern_pure(
            PureOp::Call(crate::FunctionId::from_index(0)),
            SmallVec::new(),
            ty,
            None,
        );
        graph.subsume_pure_in_place(call, selected);
        graph.nodes[selected].alias = Some(replacement);

        let mut best = extract::extract(&graph);
        close_extraction_over_aliases(&graph, &mut best);

        assert_eq!(best[&call], replacement);
        assert_eq!(best[&selected], replacement);
    }
}
