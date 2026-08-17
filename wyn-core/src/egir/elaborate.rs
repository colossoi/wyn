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
    BlockId, Constant, ControlHeader, EntryPoint, FuncBody, Function, InstKind, PlaceId, Program,
    ValueId as SsaValueId, ValueRef,
};
use crate::types::{ExternDecl, TypeExt};
use crate::{BindingRef, EntryId, LookupMap, LookupSet, ResourceAccess};
use polytype::Type;
use smallvec::SmallVec;

use super::extract;
use super::ir::{
    CallSiteId, FuncParam, OperandType, PlaceId as EgirPlaceId, PlaceOp, ResultBinding, ResultDestination,
};
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
            let body = elaborate_one_body(f.graph, &f.params, f.result.ty().clone());
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
            let body = elaborate_one_body(e.graph, &e.params, e.result.ty().clone());
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
    params: &[FuncParam<BindingRef, Type<TypeName>>],
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
    params: &[FuncParam<BindingRef, Type<TypeName>>],
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
        elaborated_calls: ScopedMap::new(),
        builder: FuncBuilder::new(
            params.iter().map(|parameter| (parameter.ty().clone(), parameter.name().to_owned())).collect(),
            return_ty,
        ),
        block_map: LookupMap::new(),
        current_block: None,
        current_skel_block: None,
    };

    // Map function params: ValueId → (ValueId, skel entry block).
    let skel_entry = graph.skeleton.entry;
    for i in 0..elab.builder.num_params() {
        let vid = elab.builder.get_param(i);
        for (nid, node) in &graph.nodes {
            if matches!(&node.kind, ValueKind::FuncParam { parameter } if parameter.index() == i) {
                let resolved = elab.resolve(nid);
                elab.elaborated.insert(resolved, (vid, skel_entry));
                break;
            }
        }
    }

    for (place, definition) in graph.places() {
        let PlaceOp::Parameter { parameter } = definition.op() else {
            continue;
        };
        let ssa_place =
            elab.builder.new_parameter_place(parameter.index(), definition.ty().pointee.clone());
        elab.elaborated_places.insert(place, (ssa_place, skel_entry));
        debug_assert!(matches!(
            params.get(parameter.index()).map(FuncParam::representation),
            Some(OperandType::Place(_))
        ));
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
            let param_nid = param_nid.value();
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

#[derive(Clone, Copy)]
enum IndexCoordinate {
    Dynamic(ValueId),
    Static(u32),
}

struct Elaborator<'a> {
    graph: &'a PhysicalEGraph,
    best: LookupMap<ValueId, ValueId>,
    domtree: &'a wyn_graph::DominatorTree<SkelBlockId>,
    loop_analysis: &'a LoopAnalysis,
    loop_stack: SmallVec<[LoopStackEntry; 4]>,
    /// EGIR value → (SSA value, skeleton block where it was placed) for
    /// value-producing nodes.
    elaborated: ScopedMap<ValueId, (SsaValueId, SkelBlockId)>,
    /// ValueId → (PlaceId, skeleton block) for place-producing nodes
    /// (`ViewIndex`, `OutputSlot`). Separate from `elaborated` because
    /// places are not interchangeable with values, and because identity
    /// matters: two hashconsed `ViewIndex` nodes still get distinct places
    /// when demanded from unrelated scopes (the `ScopedMap` already
    /// handles that via its scope-depth pop).
    elaborated_places: ScopedMap<EgirPlaceId, (PlaceId, SkelBlockId)>,
    elaborated_calls: ScopedMap<CallSiteId, Option<(SsaValueId, SkelBlockId)>>,
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
        self.elaborated_calls.push_scope();
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
        self.elaborated_calls.pop_scope();
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

        // Load/Store carry a PlaceId operand in `operands[0]` rather
        // than a value; handle them explicitly so the place operand stays
        // typed as `PlaceId`, not a ValueId. Alloca produces a PlaceId rather
        // than a ValueId — register it in `elaborated_places` so downstream
        // `PlaceIndex` / `Load` / `Store` consumers resolve it via `demand_place`.
        if let EffectOp::Alloca { result } = effect {
            let elem_ty = self.graph.place(*result).ty().pointee.clone();
            let place = self.builder.new_place(elem_ty.clone());
            let kind = InstKind::Alloca {
                elem_ty,
                result: place,
            };
            let out_bid = self.block_map[&skel_bid];
            self.builder.func_mut().append_void_inst_with_span(out_bid, kind, se.span);
            self.elaborated_places.insert(*result, (place, skel_bid));
            return;
        }

        if let EffectOp::Call { site } = effect {
            if self.graph.call(*site).effects() == CallEffects::Pure {
                return;
            }
            self.elaborate_call(*site, skel_bid);
            return;
        }

        let kind = match effect {
            EffectOp::Load { place, .. } => {
                let place = self.demand_place(*place);
                InstKind::Load { place }
            }
            EffectOp::Store { place } => {
                let place = self.demand_place(*place);
                let value = self
                    .demand_ref(se.operands[0].value().expect("store value must use the value channel"));
                InstKind::Store { place, value }
            }
            EffectOp::Atomic { place, op } => {
                let place = self.demand_place(*place);
                let values = se.operand_values().map(|node| self.demand_ref(node)).collect();
                InstKind::Atomic {
                    place,
                    op: *op,
                    values,
                }
            }
            EffectOp::Op { tag } => {
                let operands = se.operand_values().map(|nid| self.demand_ref(nid)).collect();
                InstKind::Op {
                    tag: tag.clone().map_call(|call| match call {}),
                    operands,
                }
            }
            EffectOp::ControlBarrier => InstKind::ControlBarrier,
            EffectOp::Alloca { .. } | EffectOp::Call { .. } => unreachable!(),
        };

        if let Some(result_nid) = se.result().and_then(ResultBinding::single_value) {
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

    fn elaborate_call(
        &mut self,
        site: CallSiteId,
        forced_block: SkelBlockId,
    ) -> Option<(SsaValueId, SkelBlockId)> {
        if let Some(result) = self.elaborated_calls.get(&site) {
            return result;
        }
        let call = self.graph.call(site).clone();
        let mut arguments = Vec::with_capacity(call.arguments().len());
        for argument in call.arguments() {
            let value = argument.value().expect(
                "physical SSA calls require destination places to be eliminated before elaboration",
            );
            arguments.push(self.demand_ref_placed(value));
        }
        let placed = if call.effects() == CallEffects::Pure {
            self.choose_placement(&arguments.iter().filter_map(|(_, block)| *block).collect::<Vec<_>>())
        } else {
            forced_block
        };
        let kind = InstKind::Op {
            tag: OpTag::Call(call.callee()),
            operands: arguments.into_iter().map(|(value, _)| value).collect(),
        };
        let values = call.result().values();
        if values.is_empty() {
            let out_bid = self.block_map[&placed];
            self.builder.func_mut().append_void_inst(out_bid, kind);
            self.record_call_placement(site, None, placed);
            return None;
        }

        let root = self.emit_at(placed, kind, call.result().ty().clone(), None);
        self.record_call_placement(site, Some((root, placed)), placed);
        for (value, path, ty) in result_value_paths(call.result()) {
            let mut projected = root;
            for (depth, index) in path.iter().enumerate() {
                let field_ty = result_type_at_path(call.result(), &path[..=depth]);
                projected = self.emit_at(
                    placed,
                    InstKind::Op {
                        tag: OpTag::Project { index: *index },
                        operands: vec![ValueRef::Ssa(projected)],
                    },
                    field_ty,
                    None,
                );
            }
            if path.is_empty() {
                debug_assert_eq!(self.graph.value(value).ty(), &ty);
            }
            self.record_placement(value, projected, placed);
        }
        Some((root, placed))
    }

    /// Demand a node as an instruction operand. Canonical 32-bit scalar
    /// constants stay inline instead of being expanded into nullary SSA
    /// instructions; values that need an SSA identity use `demand_placed`.
    fn demand_ref(&mut self, nid: ValueId) -> ValueRef {
        self.demand_ref_placed(nid).0
    }

    /// Operand form plus the block that defines an SSA operand. Constants
    /// have no placement dependency and therefore carry `None`.
    fn demand_ref_placed(&mut self, nid: ValueId) -> (ValueRef, Option<SkelBlockId>) {
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
    fn inline_constant(&self, nid: ValueId) -> Option<ConstantValue> {
        let node = &self.graph.nodes[nid];
        match (&node.kind, &node.ty) {
            (
                ValueKind::Constant(value @ ConstantValue::I32(_)),
                Type::Constructed(TypeName::Int(32), _),
            )
            | (
                ValueKind::Constant(value @ ConstantValue::U32(_)),
                Type::Constructed(TypeName::UInt(32), _),
            )
            | (
                ValueKind::Constant(value @ ConstantValue::F32(_)),
                Type::Constructed(TypeName::Float(32), _),
            )
            | (ValueKind::Constant(value @ ConstantValue::Bool(_)), Type::Constructed(TypeName::Bool, _)) => {
                Some(*value)
            }
            (
                ValueKind::Pure {
                    op: PureOp::Int(value),
                    operands,
                },
                Type::Constructed(TypeName::Int(32), _),
            ) if operands.is_empty() => value.parse().ok().map(ConstantValue::I32),
            (
                ValueKind::Pure {
                    op: PureOp::Uint(value),
                    operands,
                },
                Type::Constructed(TypeName::UInt(32), _),
            ) if operands.is_empty() => value.parse().ok().map(ConstantValue::U32),
            (
                ValueKind::Pure {
                    op: PureOp::Float(value),
                    operands,
                },
                Type::Constructed(TypeName::Float(32), _),
            ) if operands.is_empty() => value.parse().ok().map(ConstantValue::from_f32),
            (
                ValueKind::Pure {
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
    fn demand_place(&mut self, nid: EgirPlaceId) -> PlaceId {
        self.demand_place_placed(nid).0
    }

    fn demand_place_placed(&mut self, nid: EgirPlaceId) -> (PlaceId, SkelBlockId) {
        if let Some(place) = self.elaborated_places.get(&nid) {
            return place;
        }

        let definition = self.graph.place(nid).clone();
        let elem_ty = definition.ty().pointee.clone();
        let place = self.builder.new_place(elem_ty);
        let (kind, placed) = match definition.op() {
            PlaceOp::Index { base, index } => {
                let (parent, parent_block) = self.demand_place_placed(*base);
                let (index, index_block) = self.demand_ref_placed(*index);
                let operand_blocks = index_block.into_iter().chain([parent_block]).collect::<Vec<_>>();
                (
                    InstKind::PlaceIndex {
                        place: parent,
                        index,
                        result: place,
                    },
                    self.choose_placement(&operand_blocks),
                )
            }
            PlaceOp::ViewIndex { view, index } => {
                let (view, view_block) = self.demand_ref_placed(view.value());
                let (index, index_block) = self.demand_ref_placed(*index);
                let operand_blocks = [view_block, index_block].into_iter().flatten().collect::<Vec<_>>();
                (
                    InstKind::ViewIndex {
                        view,
                        index,
                        result: place,
                    },
                    self.choose_placement(&operand_blocks),
                )
            }
            PlaceOp::OutputSlot { index } => (
                InstKind::OutputSlot {
                    index: *index,
                    result: place,
                },
                self.choose_placement(&[]),
            ),
            PlaceOp::Parameter { .. } | PlaceOp::AllocaResult => {
                panic!("place {nid:?} should have been registered at its definition")
            }
            PlaceOp::View { .. } => {
                panic!("whole-view place {nid:?} must be indexed before SSA elaboration")
            }
            PlaceOp::Slice { .. } => {
                panic!("slice place {nid:?} must be indexed before SSA elaboration")
            }
        };

        let place = match &kind {
            InstKind::ViewIndex { result, .. }
            | InstKind::PlaceIndex { result, .. }
            | InstKind::OutputSlot { result, .. } => *result,
            _ => unreachable!(),
        };
        let span = definition.span();
        let out_bid = self.block_map[&placed];
        self.builder.func_mut().append_void_inst_with_span(out_bid, kind, span);
        self.elaborated_places.insert(nid, (place, placed));
        (place, placed)
    }

    /// Recover a direct `Index(...Index(view, i)..., j)` spine after
    /// representation specialization has changed its root to a storage view.
    /// Each entry is `(index-result node, coordinate node)` in base-to-leaf
    /// order. Stopping at the first view-typed base preserves any value
    /// operations that precede the storage address chain.
    fn view_index_spine(
        &self,
        nid: ValueId,
    ) -> Option<(ValueId, SmallVec<[(ValueId, IndexCoordinate); 4]>)> {
        let mut steps = SmallVec::<[(ValueId, IndexCoordinate); 4]>::new();
        let mut current = self.resolve(nid);
        loop {
            let ValueKind::Pure { op, operands } = &self.graph.nodes[current].kind else {
                return None;
            };
            let (base, coordinate) = match (op, operands.as_slice()) {
                (PureOp::Index, [base, index]) => (*base, IndexCoordinate::Dynamic(*index)),
                (PureOp::Project { index }, [base]) => (*base, IndexCoordinate::Static(*index)),
                _ => return None,
            };
            let base = self.resolve(base);
            steps.push((current, coordinate));
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
        view_node: ValueId,
        steps: SmallVec<[(ValueId, IndexCoordinate); 4]>,
    ) -> (SsaValueId, SkelBlockId) {
        let result_node = steps.last().expect("view index spine has at least one coordinate").0;
        let (view, root_place, root_placed) = match &self.graph.nodes[view_node].kind {
            ValueKind::PlaceView { place } => {
                let (place, placed) = self.demand_place_placed(*place);
                (None, Some(place), placed)
            }
            _ => {
                let (view, placed) = self.demand_ref_placed(view_node);
                (
                    Some(view),
                    None,
                    placed.expect("view handle has an SSA definition"),
                )
            }
        };
        let index_values = steps
            .iter()
            .map(|(_, coordinate)| match coordinate {
                IndexCoordinate::Dynamic(index) => self.demand_ref_placed(*index),
                IndexCoordinate::Static(index) => {
                    (ValueRef::Const(ConstantValue::I32(*index as i32)), None)
                }
            })
            .collect::<SmallVec<[(ValueRef, Option<SkelBlockId>); 4]>>();
        let mut operand_placements = SmallVec::<[SkelBlockId; 4]>::new();
        operand_placements.push(root_placed);
        operand_placements.extend(index_values.iter().filter_map(|(_, block)| *block));
        let placed = self.choose_placement(&operand_placements);
        let out_bid = self.block_map[&placed];

        let mut place = root_place;
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
                    view: view.expect("non-place view has an SSA handle"),
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

    /// Demand an EGIR value and return both the SSA value and the skeleton block it
    /// was placed in.
    fn demand_placed(&mut self, nid: ValueId) -> (SsaValueId, SkelBlockId) {
        let resolved = self.resolve(nid);

        if let Some(entry) = self.elaborated.get(&resolved) {
            return entry;
        }

        let node = self.graph.nodes[resolved].clone();

        match &node.kind {
            ValueKind::Constant(c) => {
                let ty = self.graph.nodes[resolved].ty.clone();
                let kind = const_to_inst_kind(c);
                let placed = self.choose_placement(&[]);
                let span = self.graph.nodes[resolved].span;
                let vid = self.emit_at(placed, kind, ty, span);
                self.record_placement(resolved, vid, placed);
                (vid, placed)
            }
            ValueKind::Pure { op, operands } => {
                if matches!(op, PureOp::Index | PureOp::Project { .. }) {
                    if let Some((view, steps)) = self.view_index_spine(resolved) {
                        return self.demand_view_index_spine(view, steps);
                    }
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
            ValueKind::FuncParam { .. } | ValueKind::BlockParam { .. } => {
                let users = self
                    .graph
                    .nodes
                    .iter()
                    .filter_map(|(user, node)| {
                        node.kind().children().contains(&resolved).then_some((
                            user,
                            node.kind(),
                            node.alias(),
                            node.result_origins(),
                        ))
                    })
                    .collect::<Vec<_>>();
                panic!(
                    "FuncParam/BlockParam {:?} should have been pre-populated in elaborated map; users {users:?}",
                    resolved,
                );
            }
            ValueKind::SideEffectResult => {
                let node_users = self
                    .graph
                    .nodes
                    .iter()
                    .filter_map(|(user, node)| {
                        node.kind().children().contains(&resolved).then_some((user, node.kind()))
                    })
                    .collect::<Vec<_>>();
                let mut boundary_users = Vec::new();
                for (block, contents) in &self.graph.skeleton.blocks {
                    for (index, effect) in contents.side_effects.iter().enumerate() {
                        if self.graph.effect_boundary_value_dependencies(effect).contains(&resolved) {
                            boundary_users.push((block, index, effect.kind()));
                        }
                    }
                }
                panic!(
                    "SideEffectResult {:?} should have been populated during side-effect elaboration; node users {node_users:?}; boundary users {boundary_users:?}",
                    resolved,
                );
            }
            ValueKind::Union { .. } => {
                panic!("Union {:?} should have been resolved by extract", resolved);
            }
            ValueKind::CallResult { call, .. } => {
                self.elaborate_call(*call, self.current_skel_block.expect("current block unset"));
                self.elaborated.get(&resolved).expect("call elaboration did not bind its by-value result")
            }
            ValueKind::PlaceLength { place } => {
                let (place, placed) = self.demand_place_placed(*place);
                let ty = self.graph.nodes[resolved].ty.clone();
                let span = self.graph.nodes[resolved].span;
                let value = self.emit_at(placed, InstKind::Load { place }, ty, span);
                self.record_placement(resolved, value, placed);
                (value, placed)
            }
            ValueKind::PlaceView { .. } => {
                let value_users = self
                    .graph
                    .nodes
                    .iter()
                    .filter_map(|(user, node)| {
                        node.kind().children().contains(&resolved).then_some((user, node.ty(), node.kind()))
                    })
                    .collect::<Vec<_>>();
                let effect_users = self
                    .graph
                    .skeleton
                    .blocks
                    .iter()
                    .flat_map(|(block, body)| {
                        body.side_effects.iter().enumerate().filter_map(move |(index, effect)| {
                            effect
                                .operands()
                                .iter()
                                .any(|operand| operand.value() == Some(resolved))
                                .then_some((
                                    block,
                                    index,
                                    effect.kind(),
                                    effect.operands(),
                                    match effect.kind() {
                                        SideEffectKind::Effect(EffectOp::Store { place }) => Some(*place),
                                        _ => None,
                                    },
                                ))
                        })
                    })
                    .collect::<Vec<_>>();
                let mut reached = LookupSet::from([resolved]);
                loop {
                    let discovered = self
                        .graph
                        .nodes
                        .iter()
                        .filter_map(|(user, node)| {
                            (!reached.contains(&user)
                                && node.kind().children().iter().any(|value| reached.contains(value)))
                            .then_some(user)
                        })
                        .collect::<Vec<_>>();
                    if discovered.is_empty() {
                        break;
                    }
                    reached.extend(discovered);
                }
                let terminator_users = self
                    .graph
                    .skeleton
                    .blocks
                    .iter()
                    .filter_map(|(block, body)| {
                        body.term
                            .referenced_nodes()
                            .into_iter()
                            .any(|value| reached.contains(&value))
                            .then_some((block, &body.term))
                    })
                    .collect::<Vec<_>>();
                let call_users = self
                    .graph
                    .calls()
                    .iter()
                    .filter_map(|(site, call)| {
                        call.arguments()
                            .iter()
                            .any(|argument| argument.value().is_some_and(|value| reached.contains(&value)))
                            .then_some((site, call.callee(), call.arguments()))
                    })
                    .collect::<Vec<_>>();
                let effect_places = effect_users
                    .iter()
                    .filter_map(|(_, _, _, _, place)| place.map(|place| (place, self.graph.place(place))))
                    .collect::<Vec<_>>();
                panic!(
                    "place-backed view {resolved:?} must be consumed through addressing; value users {value_users:?}; effect users {effect_users:?}; effect places {effect_places:?}; call users {call_users:?}; terminator users {terminator_users:?}; length intrinsic {:?}; storage-length intrinsic {:?}",
                    crate::builtins::catalog().known().length,
                    crate::builtins::catalog().known().storage_len,
                )
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
    fn record_placement(&mut self, nid: ValueId, vid: SsaValueId, placed: SkelBlockId) {
        if let Some(depth) = self.placement_scope_depth(placed) {
            self.elaborated.insert_at_depth(depth, nid, (vid, placed));
        } else {
            self.elaborated.insert(nid, (vid, placed));
        }
    }

    fn record_call_placement(
        &mut self,
        site: CallSiteId,
        result: Option<(SsaValueId, SkelBlockId)>,
        placed: SkelBlockId,
    ) {
        if let Some(depth) = self.placement_scope_depth(placed) {
            self.elaborated_calls.insert_at_depth(depth, site, result);
        } else {
            self.elaborated_calls.insert(site, result);
        }
    }

    fn placement_scope_depth(&self, placed: SkelBlockId) -> Option<usize> {
        let current = self.current_skel_block.expect("current skel block unset");
        self.loop_stack
            .iter()
            .rev()
            .filter(|f| self.loop_analysis.is_in_loop(current, f.header))
            .find(|f| f.hoist_block == placed)
            .map(|f| f.scope_depth)
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
    ) -> SsaValueId {
        let out_bid = self.block_map[&target_skel];
        self.builder.func_mut().append_inst_with_span(out_bid, kind, ty, span)
    }

    /// Resolve a ValueId through the extraction map.
    fn resolve(&self, nid: ValueId) -> ValueId {
        self.best.get(&nid).copied().unwrap_or(nid)
    }

    /// Elaborate a skeleton terminator.
    fn elaborate_terminator(&mut self, term: &SkeletonTerminator) {
        let t = match term {
            SkeletonTerminator::Return(None) => crate::ssa::framework::Terminator::Return(None),
            SkeletonTerminator::Return(Some(result)) => {
                crate::ssa::framework::Terminator::Return(self.demand_result_ref(result))
            }
            SkeletonTerminator::Branch { target, args } => {
                let out_args: Vec<ValueRef> =
                    args.iter().map(|&nid| self.demand_ref(nid.value())).collect();
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
                let ta: Vec<ValueRef> = then_args.iter().map(|&nid| self.demand_ref(nid.value())).collect();
                let ea: Vec<ValueRef> = else_args.iter().map(|&nid| self.demand_ref(nid.value())).collect();
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

    fn demand_result_ref(&mut self, result: &ResultBinding<Type<TypeName>>) -> Option<ValueRef> {
        if result.destination_count() == 0 {
            return None;
        }
        if !result.is_product() {
            let mut value = None;
            result.for_each_destination(|_, destination| {
                if let ResultDestination::ReturnValue(result) = destination {
                    value = Some(*result);
                }
            });
            return value.map(|value| self.demand_ref(value));
        }

        let fields = (0..result.field_count())
            .filter_map(|index| result.field(index))
            .map(|field| self.demand_result_ref(&field))
            .collect::<Option<Vec<_>>>()?;
        let placed = self.current_skel_block.expect("current block unset");
        let value = self.emit_at(
            placed,
            InstKind::Op {
                tag: OpTag::Tuple(fields.len()),
                operands: fields,
            },
            result.ty().clone(),
            None,
        );
        Some(ValueRef::Ssa(value))
    }
}

fn result_value_paths(result: &ResultBinding<Type<TypeName>>) -> Vec<(ValueId, Vec<u32>, Type<TypeName>)> {
    fn walk(
        result: &ResultBinding<Type<TypeName>>,
        path: &mut Vec<u32>,
        values: &mut Vec<(ValueId, Vec<u32>, Type<TypeName>)>,
    ) {
        if result.is_product() {
            for index in 0..result.field_count() {
                path.push(index as u32);
                walk(
                    &result.field(index).expect("result field disappeared"),
                    path,
                    values,
                );
                path.pop();
            }
            return;
        }
        result.for_each_destination(|ty, destination| {
            if let ResultDestination::ReturnValue(value) = destination {
                values.push((*value, path.clone(), ty.clone()));
            }
        });
    }

    let mut values = Vec::new();
    walk(result, &mut Vec::new(), &mut values);
    values
}

fn result_type_at_path(result: &ResultBinding<Type<TypeName>>, path: &[u32]) -> Type<TypeName> {
    let mut current = result.clone();
    for index in path {
        current = current.field(*index as usize).expect("result projection path is invalid");
    }
    current.ty().clone()
}

/// Compose extraction winners with CFG aliases to a fixed point. A union may
/// have selected a block parameter before skeleton optimization strips that
/// parameter; forwarding only the parameter itself would leave the union's
/// winner pointing at a definition that is no longer in its block signature.
fn close_extraction_over_aliases<P: Family>(graph: &EGraph<P>, best: &mut LookupMap<ValueId, ValueId>) {
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
// Conversion: ValueKind → InstKind
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
    InstKind::Op {
        tag: op.clone().map_call(|call| match call {}),
        operands: args.to_vec(),
    }
}

#[cfg(test)]
#[path = "elaborate_tests.rs"]
mod tests;
