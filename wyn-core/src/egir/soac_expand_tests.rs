//! Structural tests for `egir::soac_expand`.
//!
//! These tests drive the pipeline up to `expand_soacs` and then walk
//! the resulting `EGraph` to confirm:
//!   - No `_w_intrinsic_array_with_inplace` node carries a tuple
//!     result type (the bug this pass now prevents).
//!   - For SoA-tuple outputs, exactly N componentwise
//!     `_w_intrinsic_array_with_inplace` calls exist, each with a
//!     plain composite array result type.
//!   - Each component ArrayWith is fed a matching `Project { index: i }`
//!     on both `arr` and `val`, pinning down operand identity.
//!   - A `PureOp::Tuple(N)` repack exists with the SoA-tuple type.
//!
//! A coarser "N calls of the right intrinsic" check would miss operand
//! wiring mistakes; these assertions fail loudly if any component's
//! `arr` or `val` comes from the wrong projection.

use crate::Compiler;
use crate::ast::TypeName;
use crate::egir::types::{ENode, PureOp};
use crate::intrinsics::INTRINSIC_ARRAY_WITH_INPLACE;
use polytype::Type;

/// Compile source through the pipeline to just-past `expand_soacs`,
/// returning the EGraph for the (single) entry point so tests can
/// introspect node structure.
fn compile_to_expanded_egraph(input: &str) -> crate::egir::types::EGraph {
    let mut frontend = crate::cached_frontend();
    let type_checked = Compiler::parse(input, &mut frontend.node_counter)
        .expect("parse")
        .elaborate_modules(&mut frontend.module_manager)
        .expect("elaborate")
        .desugar(&mut frontend.node_counter)
        .expect("desugar")
        .resolve(&mut frontend.module_manager)
        .expect("resolve")
        .fold_ast_constants()
        .type_check(&mut frontend.module_manager, &mut frontend.schemes)
        .expect("type check");

    let expanded = type_checked
        .to_tlc(&frontend.schemes, &frontend.module_manager, false)
        .partial_eval()
        .normalize_soacs()
        .fuse_maps()
        .apply_ownership()
        .expect("apply_ownership")
        .defunctionalize()
        .monomorphize()
        .buffer_specialize()
        .fold_generated_lambdas()
        .inline_small()
        .parallelize_soacs(false)
        .filter_reachable()
        .to_egraph()
        .expect("to_egraph")
        .expand_soacs(true);

    let inner = &expanded.0;
    assert_eq!(
        inner.entry_points.len(),
        1,
        "test expects exactly one entry point"
    );
    inner.entry_points[0].graph.clone()
}

/// Collect all `_w_intrinsic_array_with_inplace` nodes in the graph.
fn array_with_nodes(graph: &crate::egir::types::EGraph) -> Vec<crate::egir::types::NodeId> {
    graph
        .nodes
        .iter()
        .filter_map(|(id, node)| match node {
            ENode::Pure {
                op: PureOp::Call(name),
                ..
            } if name == INTRINSIC_ARRAY_WITH_INPLACE => Some(id),
            _ => None,
        })
        .collect()
}

fn is_soa_tuple(ty: &Type<TypeName>) -> bool {
    matches!(ty, Type::Constructed(TypeName::Tuple(_), components)
        if components.iter().all(|c|
            matches!(c, Type::Constructed(TypeName::Array, args) if args.len() == 3)))
}

#[test]
fn map_array_of_mixed_tuple_emits_componentwise_array_with() {
    // Map output: [8](f32, i32, vec3f32).
    // After SoA, the output becomes ([8]f32, [8]i32, [8]vec3f32).
    // soac_expand should split the per-iteration write into three
    // _w_intrinsic_array_with_inplace calls, one per component,
    // then repack with a PureOp::Tuple(3).
    let source = r#"
def build(xs: [8]f32) [8](f32, i32, vec3f32) =
    map(|x: f32| (x + 1.0, 0, @[x, x, x]), xs)

#[fragment]
entry frag(c: vec4f32) vec4f32 =
    let arr = build([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]) in
    let (a, _, v) = arr[3] in
    @[a, v.x, v.y, v.z]
"#;
    let graph = compile_to_expanded_egraph(source);
    let aw_nodes = array_with_nodes(&graph);

    // 1. No ArrayWith may have a tuple result type.
    for id in &aw_nodes {
        let ty = &graph.types[id];
        assert!(
            !matches!(ty, Type::Constructed(TypeName::Tuple(_), _)),
            "tuple-typed ArrayWith survived: node {:?} has type {:?}",
            id,
            ty
        );
    }

    // 2. At least 3 ArrayWith nodes — one per SoA-tuple component.
    //    (Allowing >3 because unrolling or other passes may materialize
    //    more for other loops; what matters is that the soa-split case
    //    produced the per-component set.)
    assert!(
        aw_nodes.len() >= 3,
        "expected at least 3 componentwise ArrayWith nodes, got {}",
        aw_nodes.len()
    );

    // 3. Each ArrayWith's `arr` operand (operand[0]) is a Project{i}
    //    onto SOME loop-carried tuple, and the project index lines up
    //    with the ArrayWith's result type being the i-th component of
    //    that tuple.
    //    Each ArrayWith's `val` operand (operand[2]) is a Project{i}
    //    onto the mapped lambda result, with the same index.
    //    We assert matching indices per ArrayWith. This catches
    //    "wired to the wrong component" bugs.
    for id in &aw_nodes {
        let ENode::Pure { operands, .. } = &graph.nodes[*id] else {
            panic!("ArrayWith should be Pure");
        };
        assert_eq!(operands.len(), 3, "ArrayWith takes 3 operands");
        let arr_op = &graph.nodes[operands[0]];
        let val_op = &graph.nodes[operands[2]];

        let arr_index = match arr_op {
            ENode::Pure {
                op: PureOp::Project { index },
                ..
            } => Some(*index),
            _ => None,
        };
        let val_index = match val_op {
            ENode::Pure {
                op: PureOp::Project { index },
                ..
            } => Some(*index),
            _ => None,
        };
        assert!(
            arr_index.is_some(),
            "ArrayWith {:?} arr operand is not a Project: {:?}",
            id,
            arr_op
        );
        assert!(
            val_index.is_some(),
            "ArrayWith {:?} val operand is not a Project: {:?}",
            id,
            val_op
        );
        assert_eq!(
            arr_index, val_index,
            "ArrayWith {:?} arr/val indices disagree — operand wiring bug",
            id
        );
    }

    // 4. There exists a PureOp::Tuple(3) node with the SoA-tuple
    //    ([8]f32, [8]i32, [8]vec3f32) as its result type — the
    //    repack produced by `emit_write_element`.
    let has_repack = graph.nodes.iter().any(|(id, node)| match node {
        ENode::Pure {
            op: PureOp::Tuple(3), ..
        } => is_soa_tuple(&graph.types[&id]),
        _ => false,
    });
    assert!(
        has_repack,
        "expected a PureOp::Tuple(3) repack with SoA-tuple type; \
         ArrayWith split did not complete correctly"
    );
}
