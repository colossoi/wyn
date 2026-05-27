//! Unit tests for `EntryBuilder` and the reduce-port helpers
//! (`phase1_transform_reduce`, `synthesize_phase2_reduce`). Each test
//! hand-builds a minimal EgirEntry, runs one of the helpers, and
//! asserts the resulting graph shape — independent of TLC / from_tlc.

use polytype::Type;

use super::EntryBuilder;
use crate::ast::TypeName;
use crate::builtins::catalog;

fn f32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Float(32), vec![])
}

fn unsized_arr_view_ty(elem: Type<TypeName>) -> Type<TypeName> {
    // `Array[elem, ?size, ArrayVariantView]` — the storage-view shape
    // a runtime-sized partials buffer takes inside the EGraph.
    Type::Constructed(
        TypeName::Array,
        vec![
            elem,
            Type::Variable(0),
            Type::Constructed(TypeName::ArrayVariantView, vec![]),
        ],
    )
}

/// Construct a phase2-style EgirEntry programmatically and assert its
/// shape (one `PendingSoac::Reduce`, one `Store` to the result binding,
/// `Compute` execution model). `op_add` is a placeholder function name
/// — the test doesn't elaborate, only inspects the synthesized skeleton.
#[test]
fn entry_builder_constructs_phase2_shape() {
    let mut b = EntryBuilder::new_compute("compute_sum_phase2_combine".to_string(), (1, 1, 1));
    b.declare_intermediate_storage(0, 1, f32_ty());
    b.declare_output_storage(0, 2, f32_ty());

    let view = b.emit_storage_view(0, 1, unsized_arr_view_ty(f32_ty()));
    let init = b.emit_constant(crate::ssa::types::ConstantValue::from_f32(0.0), f32_ty());
    let r = b.emit_pending_reduce(
        "op_add".into(),
        view,
        unsized_arr_view_ty(f32_ty()),
        f32_ty(),
        init,
        vec![],
    );
    let zero = b.emit_u32(0);
    b.emit_storage_store(0, 2, zero, r, f32_ty());

    let entry = b.build();
    assert_eq!(entry.name, "compute_sum_phase2_combine");
    assert_eq!(entry.storage_bindings.len(), 2);
    assert!(matches!(
        entry.execution_model,
        crate::ssa::types::ExecutionModel::Compute { .. }
    ));

    // The entry's skeleton should contain exactly one PendingSoac::Reduce
    // (waiting for `soac_expand` to lower it into a serial loop) and one
    // InstKind::Store (the storage_store of the reduced scalar).
    let mut pending_reduces = 0;
    let mut stores = 0;
    for (_, block) in &entry.graph.skeleton.blocks {
        for se in &block.side_effects {
            match &se.kind {
                crate::egir::types::SideEffectKind::Pending(crate::egir::types::PendingSoac::Reduce {
                    ..
                }) => pending_reduces += 1,
                crate::egir::types::SideEffectKind::Inst(crate::ssa::types::InstKind::Store { .. }) => {
                    stores += 1
                }
                _ => {}
            }
        }
    }
    assert_eq!(pending_reduces, 1, "expected exactly one PendingSoac::Reduce");
    assert_eq!(stores, 1, "expected exactly one Store");
    assert!(matches!(
        entry.graph.skeleton.blocks[entry.graph.skeleton.entry].term,
        crate::egir::types::SkeletonTerminator::Return(None)
    ));
    let _ = catalog(); // ensure the catalog is initialised (storage_len is used inside the builder).
}

/// Build a minimal pre-parallelize reduce-shaped EgirEntry, run
/// `phase1_transform_reduce`, and assert the resulting graph matches
/// the phase1 shape: chunked reduce input, store redirected to
/// `partials[tid]`, outputs cleared, partials decl appended.
#[test]
fn phase1_transform_reduce_in_place() {
    use super::EntryBuilder;
    use crate::egir::parallelize::phase1_transform_reduce;
    use crate::egir::types::{PendingSoac, PureOp, PureViewSource, SideEffectKind};

    // Build a fake pre-parallelize reduce entry by hand using the
    // builder. The shape mirrors what `from_tlc::convert_entry_point`
    // would produce for `entry sum(xs: []f32) f32 = reduce(+, 0.0, xs)`.
    let mut b = EntryBuilder::new_compute("sum".to_string(), (64, 1, 1));
    b.declare_intermediate_storage(0, 0, f32_ty()); // input xs
    b.declare_output_storage(0, 1, f32_ty()); // auto-allocated output

    let xs_view = b.emit_storage_view(0, 0, unsized_arr_view_ty(f32_ty()));
    let init = b.emit_constant(crate::ssa::types::ConstantValue::from_f32(0.0), f32_ty());
    let reduce_result = b.emit_pending_reduce(
        "op_add".into(),
        xs_view,
        unsized_arr_view_ty(f32_ty()),
        f32_ty(),
        init,
        vec![],
    );
    let zero = b.emit_u32(0);
    b.emit_storage_store(0, 1, zero, reduce_result, f32_ty());
    let mut entry = b.build();

    // Decorate the entry as if `from_tlc::build_entry_outputs` had run:
    // there's one storage output at (0, 1).
    entry.outputs.push(crate::ssa::types::EntryOutput {
        ty: f32_ty(),
        decoration: None,
        storage_binding: Some((0, 1)),
    });

    // Run the in-place transformation. Partials at (0, 2).
    phase1_transform_reduce(&mut entry, 64, (0, 2)).expect("phase1 transform succeeded");

    // 1. The Reduce side-effect's input should now be a NEW storage view
    //    whose operands are (chunk_start, chunk_len) — not (0, full_len).
    let entry_block = entry.graph.skeleton.entry;
    let reduce_se = entry.graph.skeleton.blocks[entry_block]
        .side_effects
        .iter()
        .find(|se| matches!(se.kind, SideEffectKind::Pending(PendingSoac::Reduce { .. })))
        .expect("Reduce present");
    let new_input = reduce_se.operand_nodes[0];
    let new_input_op = match &entry.graph.nodes[new_input] {
        crate::egir::types::ENode::Pure { op, .. } => op.clone(),
        other => panic!("input view is not a Pure node: {:?}", other),
    };
    match new_input_op {
        PureOp::StorageView(PureViewSource::Storage { set, binding }) => {
            assert_eq!((set, binding), (0, 0), "input view still points at xs");
        }
        other => panic!("input view became non-StorageView: {:?}", other),
    }
    let new_input_operands = match &entry.graph.nodes[new_input] {
        crate::egir::types::ENode::Pure { operands, .. } => operands.clone(),
        _ => unreachable!(),
    };
    assert_eq!(
        new_input_operands.len(),
        2,
        "chunked view has [offset, len] operands"
    );
    // The offset operand should be a BinOp `*` (chunk_start = tid * chunk_size).
    let chunk_start_node = &entry.graph.nodes[new_input_operands[0]];
    assert!(
        matches!(
            chunk_start_node,
            crate::egir::types::ENode::Pure {
                op: PureOp::BinOp(s),
                ..
            } if s == "*"
        ),
        "chunked view's offset operand should be tid * chunk_size: {:?}",
        chunk_start_node
    );

    // 2. The Store side-effect should target partials[tid] now, not
    //    output[0]. Identify by walking from the place operand.
    let store_se = entry.graph.skeleton.blocks[entry_block]
        .side_effects
        .iter()
        .find(|se| {
            matches!(
                &se.kind,
                SideEffectKind::Inst(crate::ssa::types::InstKind::Store { .. })
            )
        })
        .expect("Store present");
    let place = store_se.operand_nodes[0];
    // place = ViewIndex(view, index) — extract the view; should be
    // a StorageView at (0, 2).
    match &entry.graph.nodes[place] {
        crate::egir::types::ENode::Pure {
            op: PureOp::ViewIndex,
            operands,
            ..
        } => {
            let view_op = match &entry.graph.nodes[operands[0]] {
                crate::egir::types::ENode::Pure { op, .. } => op.clone(),
                _ => panic!("place's view operand is not Pure"),
            };
            match view_op {
                PureOp::StorageView(PureViewSource::Storage { set, binding }) => {
                    assert_eq!((set, binding), (0, 2), "Store now targets partials at (0, 2)");
                }
                other => panic!("Store's view is not StorageView(partials): {:?}", other),
            }
        }
        other => panic!("Store place is not a ViewIndex: {:?}", other),
    }

    // 3. outputs[0].storage_binding cleared (phase2 will take it).
    assert_eq!(entry.outputs[0].storage_binding, None);

    // 4. storage_bindings contains the new partials Intermediate.
    assert!(entry.storage_bindings.iter().any(|b| b.set == 0
        && b.binding == 2
        && matches!(b.role, crate::interface::StorageRole::Intermediate)));
}

/// Synthesize a phase2-combine EgirEntry from scratch using
/// `synthesize_phase2_reduce` and assert its workgroup-parallel tree-reduce
/// shape: a `LocalSize(W,1,1)` entry with a workgroup-shared array, barriers,
/// and `op_func` calls — not the old single-threaded `PendingSoac::Reduce`.
#[test]
fn synthesize_phase2_reduce_shape() {
    use crate::egir::parallelize::{PHASE2_WIDTH, synthesize_phase2_reduce};
    use crate::egir::types::{ENode, PendingSoac, PureOp, PureViewSource, SideEffectKind};
    use crate::ssa::types::{ConstantValue, InstKind};

    let entry = synthesize_phase2_reduce(
        "sum",
        "_w_lambda_0".to_string(),
        f32_ty(),
        ConstantValue::from_f32(0.0),
        (0, 1), // partials at (0, 1)
        (0, 2), // result at (0, 2)
    );

    assert_eq!(entry.name, "sum_phase2_combine");
    // Single-workgroup tree reduce: local_size = (W, 1, 1).
    match entry.execution_model {
        crate::ssa::types::ExecutionModel::Compute { local_size } => {
            assert_eq!(local_size, (PHASE2_WIDTH, 1, 1));
        }
        other => panic!("expected Compute execution model, got {:?}", other),
    }

    // Two storage decls: intermediate partials, output result.
    assert_eq!(entry.storage_bindings.len(), 2);
    assert!(entry.storage_bindings.iter().any(|b| {
        b.set == 0 && b.binding == 1 && matches!(b.role, crate::interface::StorageRole::Intermediate)
    }));
    assert!(entry.storage_bindings.iter().any(|b| {
        b.set == 0 && b.binding == 2 && matches!(b.role, crate::interface::StorageRole::Output)
    }));

    // No serial PendingSoac::Reduce anymore — the combine is open-coded.
    let any_se = |pred: &dyn Fn(&crate::egir::types::SideEffect) -> bool| {
        entry.graph.skeleton.blocks.iter().any(|(_, b)| b.side_effects.iter().any(|se| pred(se)))
    };
    assert!(
        !any_se(&|se| matches!(&se.kind, SideEffectKind::Pending(PendingSoac::Reduce { .. }))),
        "tree-reduce phase2 must not emit a serial PendingSoac::Reduce"
    );

    // A workgroup-shared `array<f32, W>` view backs the tree.
    assert!(
        entry.graph.nodes.values().any(|n| matches!(
            n,
            ENode::Pure {
                op: PureOp::StorageView(PureViewSource::Workgroup { count, .. }),
                ..
            } if *count == PHASE2_WIDTH
        )),
        "expected a workgroup view of {} elements",
        PHASE2_WIDTH
    );

    // Two barriers (after the grid-stride store, and once per tree step).
    let barriers = entry
        .graph
        .skeleton
        .blocks
        .iter()
        .flat_map(|(_, b)| b.side_effects.iter())
        .filter(|se| matches!(&se.kind, SideEffectKind::Inst(InstKind::ControlBarrier)))
        .count();
    assert_eq!(barriers, 2, "grid-stride barrier + tree-step barrier");

    // The combiner is invoked via Call(op_func) (grid-stride + tree merge).
    assert!(
        entry.graph.nodes.values().any(|n| matches!(
            n,
            ENode::Pure { op: PureOp::Call(f), .. } if f == "_w_lambda_0"
        )),
        "expected Call to the reduce combiner"
    );

    // local_id drives shared-memory addressing.
    let local_id = crate::builtins::catalog().known().local_id;
    assert!(
        entry.graph.nodes.values().any(|n| matches!(
            n,
            ENode::Pure { op: PureOp::Intrinsic { id, .. }, .. } if *id == local_id
        )),
        "expected a _w_intrinsic_local_id node"
    );

    // The neutral element is still the f32 0.0 constant.
    assert!(
        entry.graph.nodes.values().any(|n| matches!(
            n,
            ENode::Constant(ConstantValue::F32(bits)) if f32::from_bits(*bits) == 0.0
        )),
        "expected the f32 0.0 neutral element"
    );
}
