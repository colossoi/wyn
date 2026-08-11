//! Structural tests for `egir::soac_expand`.
//!
//! These tests drive the pipeline up to `expand_soacs` and then walk
//! the resulting `EGraph` to confirm:
//!   - No `_w_intrinsic_array_with_inplace` node carries a tuple
//!     result type.
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

use crate::ast::TypeName;
use crate::egir::program::{PhysicalEGraph, PhysicalSideEffectKind, ProgramIdentities, SemanticOpId};
use crate::egir::soac::{hist, screma};
use crate::egir::types::{ENode, Family, NodeId, Physical, PureOp, Soac, SoacEffect, SoacInputType};
use polytype::Type;

/// Compile source through the pipeline to just-past `expand_soacs`,
/// returning the EGraph for the (single) entry point so tests can
/// introspect node structure.
fn compile_to_expanded_egraph(input: &str) -> PhysicalEGraph {
    let program = crate::compile_thru_tlc(input).expect("compile_thru_tlc");
    let program = crate::tlc::infer_input_slice_bounds(program);
    let program = crate::to_egraph(program).expect("to_egraph");
    let program = crate::egir::realize_outputs(program).expect("realize_outputs");
    let program = crate::egir::reify_soacs(program);
    let program = crate::egir::optimize_semantics(program);
    let program = crate::egir::plan_logical_resources(program).expect("allocate semantic EGIR");
    let program = crate::egir::plan(program, crate::LoweringProfile::PORTABLE).expect("terminal schedule");
    let program = crate::egir::expand_soacs(program).expect("physical SOAC expansion");
    let inner = &program;
    inner
        .entry_points
        .iter()
        .find(|entry| entry.name.ends_with("__fragment"))
        .expect("test expects an extracted fragment stage")
        .graph
        .clone()
}

/// Collect all `_w_intrinsic_array_with_inplace` nodes in the graph.
fn array_with_nodes<P: Family>(graph: &crate::egir::types::EGraph<P>) -> Vec<crate::egir::types::NodeId> {
    let inplace_id = crate::builtins::catalog().known().array_with_in_place;
    graph
        .nodes
        .iter()
        .filter_map(|(id, node)| match &node.kind {
            ENode::Pure {
                op: PureOp::Intrinsic { id: bid, .. },
                ..
            } if *bid == inplace_id => Some(id),
            _ => None,
        })
        .collect()
}

fn is_soa_tuple(ty: &Type<TypeName>) -> bool {
    matches!(ty, Type::Constructed(TypeName::Tuple(_), components)
        if components.iter().all(|c|
            matches!(c, Type::Constructed(TypeName::Array, args) if args.len() == 4)))
}

fn plain_array_ty(elem: Type<TypeName>) -> Type<TypeName> {
    Type::Constructed(
        TypeName::Array,
        vec![
            elem,
            Type::Constructed(TypeName::ArrayVariantComposite, vec![]),
            Type::Constructed(TypeName::Size(4), vec![]),
            crate::types::no_buffer(),
        ],
    )
}

#[test]
fn scatter_handleability_checks_every_input() {
    let mut identities = ProgramIdentities::default();
    let bucket_region = identities.alloc_function("scatter_bucket".into());
    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
    let f32_ty = Type::Constructed(TypeName::Float(32), vec![]);
    let bad_input_ty = Type::Constructed(TypeName::Tuple(2), vec![i32_ty.clone(), f32_ty.clone()]);
    let kind: PhysicalSideEffectKind = PhysicalSideEffectKind::Soac(SoacEffect(
        SemanticOpId::for_test(0),
        Soac::<Physical>::Hist(hist::Op {
            inputs: vec![
                SoacInputType::array(plain_array_ty(i32_ty.clone())),
                SoacInputType::array(bad_input_ty),
            ],
            form: hist::HistForm {
                bucket: screma::Lambda::region(
                    crate::egir::types::SegBody {
                        region: bucket_region,
                        captures: vec![],
                    },
                    vec![i32_ty.clone(), f32_ty.clone()],
                    vec![i32_ty.clone(), f32_ty.clone()],
                ),
                operations: vec![hist::HistOp {
                    shape: vec![NodeId::from(slotmap::KeyData::from_ffi(1))],
                    race_factor: NodeId::from(slotmap::KeyData::from_ffi(2)),
                    destinations: vec![NodeId::from(slotmap::KeyData::from_ffi(3))],
                    update: hist::Update::OrderedOverwrite {
                        value_types: vec![f32_ty],
                    },
                }],
            },
            state: hist::PhysicalState::Serial,
        }),
    ));

    assert!(
        !super::is_handleable_soac(&kind),
        "scatter expansion reads every input, so an unreadable later input must reject the SOAC"
    );
}

#[test]
fn serial_hist_lowers_multiple_shapes_components_and_one_tuple_reducer_call() {
    use crate::egir::graph_ops;
    use crate::egir::program::ProgramIdentities;
    use crate::egir::types::{EffectOp, SideEffectKind};
    use smallvec::{smallvec, SmallVec};

    let mut graph = PhysicalEGraph::new();
    let block = graph.skeleton.entry;
    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
    let bool_ty = Type::Constructed(TypeName::Bool, vec![]);
    let array_ty = plain_array_ty(i32_ty.clone());
    let zero = graph.intern_pure(PureOp::Int("0".into()), smallvec![], i32_ty.clone(), None);
    let one = graph.intern_pure(PureOp::Int("1".into()), smallvec![], i32_ty.clone(), None);
    let two = graph.intern_pure(PureOp::Int("2".into()), smallvec![], i32_ty.clone(), None);
    let four = graph.intern_pure(PureOp::Int("4".into()), smallvec![], i32_ty.clone(), None);
    let mut input_nodes = SmallVec::<[NodeId; 4]>::new();
    for _ in 0..6 {
        input_nodes.push(graph.intern_pure(
            PureOp::ArrayLit(4),
            smallvec![zero, zero, zero, zero],
            array_ty.clone(),
            None,
        ));
    }
    let destinations = (0..3)
        .map(|binding| {
            graph_ops::intern_storage_view(
                &mut graph,
                crate::BindingRef::new(2, binding),
                i32_ty.clone(),
                None,
            )
        })
        .collect::<Vec<_>>();
    let mut regions = ProgramIdentities::default();
    let reducer_region = regions.alloc_function("hist_tuple_reducer".into());
    let histogram = hist::Op::<Physical> {
        inputs: (0..6).map(|_| SoacInputType::array(array_ty.clone())).collect(),
        form: hist::HistForm {
            bucket: screma::Lambda::identity(vec![i32_ty.clone(); 6]),
            operations: vec![
                hist::HistOp {
                    shape: vec![two, two],
                    race_factor: one,
                    destinations: destinations[..2].to_vec(),
                    update: hist::Update::Reduce {
                        operator: screma::Lambda::region(
                            crate::egir::types::SegBody {
                                region: reducer_region,
                                captures: vec![],
                            },
                            vec![i32_ty.clone(); 4],
                            vec![i32_ty.clone(); 2],
                        ),
                        neutral: vec![zero, zero],
                    },
                },
                hist::HistOp {
                    shape: vec![four],
                    race_factor: one,
                    destinations: destinations[2..].to_vec(),
                    update: hist::Update::OrderedOverwrite {
                        value_types: vec![i32_ty.clone()],
                    },
                },
            ],
        },
        state: hist::PhysicalState::Serial,
    };
    let mut effect_ids = crate::IdSource::new();
    graph_ops::emit_pending_soac(
        &mut graph,
        block,
        SemanticOpId::for_test(0),
        Soac::Hist(histogram),
        input_nodes,
        bool_ty,
        &mut effect_ids,
        None,
    );

    let graph =
        super::run_one_body(graph, &regions, &mut effect_ids).expect("general serial Hist should expand");
    let stores = graph
        .skeleton
        .blocks
        .iter()
        .flat_map(|(_, block)| &block.side_effects)
        .filter(|effect| matches!(effect.kind, SideEffectKind::Effect(EffectOp::Store)))
        .count();
    assert_eq!(
        stores, 12,
        "three destination components are stored across four unrolled iterations"
    );
    assert!(graph.skeleton.blocks.iter().all(|(_, block)| {
        block.side_effects.iter().all(|effect| !matches!(effect.kind, SideEffectKind::Soac(_)))
    }));
    let reducer_calls = graph
        .nodes
        .iter()
        .filter(|(_, node)| {
            matches!(
                &node.kind,
                ENode::Pure {
                    op: PureOp::Call(name),
                    operands,
                } if *name == reducer_region && operands.len() == 4
            )
        })
        .count();
    assert_eq!(
        reducer_calls, 4,
        "the two-component reducer must be invoked once, not once per component, in each iteration"
    );
    assert_eq!(
        stores / reducer_calls,
        3,
        "each iteration stores all three components"
    );
    assert!(
        graph.nodes.iter().any(|(_, node)| {
            matches!(
                &node.kind,
                ENode::Pure {
                    op: PureOp::BinOp(op),
                    ..
                } if *op == crate::op::BinaryOperator::Multiply
            )
        }),
        "rank-2 indices must be flattened row-major"
    );
}
#[test]
fn serial_hist_ignores_out_of_bounds_indices() {
    use crate::egir::graph_ops;
    use crate::egir::program::ProgramIdentities;
    use crate::egir::types::SkeletonTerminator;
    use smallvec::{smallvec, SmallVec};

    let mut graph = PhysicalEGraph::new();
    let block = graph.skeleton.entry;
    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
    let bool_ty = Type::Constructed(TypeName::Bool, vec![]);
    let array_ty = plain_array_ty(i32_ty.clone());
    let negative_one = graph.intern_pure(PureOp::Int("-1".into()), smallvec![], i32_ty.clone(), None);
    let zero = graph.intern_pure(PureOp::Int("0".into()), smallvec![], i32_ty.clone(), None);
    let one = graph.intern_pure(PureOp::Int("1".into()), smallvec![], i32_ty.clone(), None);
    let four = graph.intern_pure(PureOp::Int("4".into()), smallvec![], i32_ty.clone(), None);
    let indices = graph.intern_pure(
        PureOp::ArrayLit(4),
        smallvec![negative_one, zero, four, one],
        array_ty.clone(),
        None,
    );
    let values = graph.intern_pure(
        PureOp::ArrayLit(4),
        smallvec![one, one, one, one],
        array_ty.clone(),
        None,
    );
    let destination =
        graph_ops::intern_storage_view(&mut graph, crate::BindingRef::new(2, 0), i32_ty.clone(), None);
    let histogram = hist::Op::<Physical> {
        inputs: vec![
            SoacInputType::array(array_ty.clone()),
            SoacInputType::array(array_ty.clone()),
        ],
        form: hist::HistForm {
            bucket: screma::Lambda::identity(vec![i32_ty.clone(), i32_ty.clone()]),
            operations: vec![hist::HistOp {
                shape: vec![four],
                race_factor: one,
                destinations: vec![destination],
                update: hist::Update::OrderedOverwrite {
                    value_types: vec![i32_ty.clone()],
                },
            }],
        },
        state: hist::PhysicalState::Serial,
    };
    let mut effect_ids = crate::IdSource::new();
    graph_ops::emit_pending_soac(
        &mut graph,
        block,
        SemanticOpId::for_test(0),
        Soac::Hist(histogram),
        SmallVec::from_vec(vec![indices, values]),
        bool_ty,
        &mut effect_ids,
        None,
    );

    let regions = ProgramIdentities::default();
    let graph =
        super::run_one_body(graph, &regions, &mut effect_ids).expect("serial histogram should expand");
    assert!(
        graph.nodes.iter().any(|(_, node)| {
            matches!(
                &node.kind,
                ENode::Pure {
                    op: PureOp::BinOp(op),
                    ..
                } if *op == crate::op::BinaryOperator::GreaterEqual
            )
        }),
        "serial Hist must reject negative bucket indices"
    );
    assert!(
        graph.skeleton.blocks.iter().any(|(_, block)| {
            matches!(
                block.control_header,
                Some(crate::flow::ControlHeader::Selection { .. })
            ) && matches!(block.term, SkeletonTerminator::CondBranch { .. })
        }),
        "serial Hist must branch around the load/store for invalid indices"
    );
}
#[test]
fn atomic_hist_lowers_multiple_operations_with_bounds_checks() {
    use crate::egir::graph_ops;
    use crate::egir::program::ProgramIdentities;
    use crate::egir::types::{EffectOp, SegExtent, SegSpace, SideEffectKind};
    use crate::ssa::types::AtomicOp;
    use smallvec::{smallvec, SmallVec};

    let mut graph = PhysicalEGraph::new();
    let block = graph.skeleton.entry;
    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
    let bool_ty = Type::Constructed(TypeName::Bool, vec![]);
    let array_ty = plain_array_ty(i32_ty.clone());
    let zero = graph.intern_pure(PureOp::Int("0".into()), smallvec![], i32_ty.clone(), None);
    let one = graph.intern_pure(PureOp::Int("1".into()), smallvec![], i32_ty.clone(), None);
    let two = graph.intern_pure(PureOp::Int("2".into()), smallvec![], i32_ty.clone(), None);
    let four = graph.intern_pure(PureOp::Int("4".into()), smallvec![], i32_ty.clone(), None);
    let mut inputs = SmallVec::<[NodeId; 4]>::new();
    for _ in 0..5 {
        inputs.push(graph.intern_pure(
            PureOp::ArrayLit(4),
            smallvec![zero, zero, zero, zero],
            array_ty.clone(),
            None,
        ));
    }
    let destinations = (0..2)
        .map(|binding| {
            graph_ops::intern_storage_view(
                &mut graph,
                crate::BindingRef::new(2, binding),
                i32_ty.clone(),
                None,
            )
        })
        .collect::<Vec<_>>();
    let mut regions = ProgramIdentities::default();
    let first_reducer = regions.alloc_function("first_reducer".into());
    let second_reducer = regions.alloc_function("second_reducer".into());
    let histogram = hist::Op::<Physical> {
        inputs: (0..5).map(|_| SoacInputType::array(array_ty.clone())).collect(),
        form: hist::HistForm {
            bucket: screma::Lambda::identity(vec![i32_ty.clone(); 5]),
            operations: vec![
                hist::HistOp {
                    shape: vec![two, two],
                    race_factor: one,
                    destinations: vec![destinations[0]],
                    update: hist::Update::Reduce {
                        operator: screma::Lambda::region(
                            crate::egir::types::SegBody {
                                region: first_reducer,
                                captures: vec![],
                            },
                            vec![i32_ty.clone(); 2],
                            vec![i32_ty.clone()],
                        ),
                        neutral: vec![zero],
                    },
                },
                hist::HistOp {
                    shape: vec![four],
                    race_factor: one,
                    destinations: vec![destinations[1]],
                    update: hist::Update::Reduce {
                        operator: screma::Lambda::region(
                            crate::egir::types::SegBody {
                                region: second_reducer,
                                captures: vec![],
                            },
                            vec![i32_ty.clone(); 2],
                            vec![i32_ty.clone()],
                        ),
                        neutral: vec![zero],
                    },
                },
            ],
        },
        state: hist::PhysicalState::Atomic {
            space: SegSpace::from_dims(vec![SegExtent::Fixed(4)]).unwrap(),
            operations: vec![
                hist::AtomicUpdate::Direct(AtomicOp::Add),
                hist::AtomicUpdate::Direct(AtomicOp::Xor),
            ],
        },
    };
    let mut effect_ids = crate::IdSource::new();
    graph_ops::emit_pending_soac(
        &mut graph,
        block,
        SemanticOpId::for_test(0),
        Soac::Hist(histogram),
        inputs,
        bool_ty,
        &mut effect_ids,
        None,
    );

    let graph = super::run_one_body(graph, &regions, &mut effect_ids)
        .expect("multi-operation atomic Hist should expand");
    let atomics = graph
        .skeleton
        .blocks
        .iter()
        .flat_map(|(_, block)| &block.side_effects)
        .filter(|effect| matches!(effect.kind, SideEffectKind::Effect(EffectOp::Atomic(_))))
        .count();
    assert_eq!(atomics, 2, "one atomic update per histogram operation");
    assert!(graph.skeleton.blocks.iter().all(|(_, block)| {
        block.side_effects.iter().all(|effect| {
            !matches!(
                effect.kind,
                SideEffectKind::Effect(EffectOp::Load | EffectOp::Store) | SideEffectKind::Soac(_)
            )
        })
    }));
    assert!(graph.nodes.iter().any(|(_, node)| {
        matches!(
            &node.kind,
            ENode::Pure {
                op: PureOp::BinOp(op),
                ..
            } if *op == crate::op::BinaryOperator::GreaterEqual
        )
    }));
    assert!(graph.nodes.iter().any(|(_, node)| {
        matches!(
            &node.kind,
            ENode::Pure {
                op: PureOp::BinOp(op),
                ..
            } if *op == crate::op::BinaryOperator::Multiply
        )
    }));
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

def fragment_main(fragment: fragment_invocation<vec4f32>) vec4f32 =
    let arr = build([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]) in
    let (a, _, v) = arr[3] in
    @[a, v.x, v.y, v.z]

entry frame(target: render_target<vec4f32>) render_target<vec4f32> =
    let covered = rasterize_triangles(
      direct_draw(3u32, 1u32),
      |vertex| vertex_output(
        if vertex.vertex_index == 0u32 then @[-1.0, -1.0, 0.0, 1.0]
        else if vertex.vertex_index == 1u32 then @[3.0, -1.0, 0.0, 1.0]
        else @[-1.0, 3.0, 0.0, 1.0],
        @[0.0, 0.0, 0.0, 0.0])) in
    shade(target, covered, fragment_main)
"#;
    let graph = compile_to_expanded_egraph(source);
    let aw_nodes = array_with_nodes(&graph);

    // 1. No ArrayWith may have a tuple result type.
    for id in &aw_nodes {
        let ty = &graph.nodes[*id].ty;
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
        let ENode::Pure { operands, .. } = &graph.nodes[*id].kind else {
            panic!("ArrayWith should be Pure");
        };
        assert_eq!(operands.len(), 3, "ArrayWith takes 3 operands");
        let arr_op = &graph.nodes[operands[0]].kind;
        let val_op = &graph.nodes[operands[2]].kind;

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
    let has_repack = graph.nodes.iter().any(|(_, node)| match &node.kind {
        ENode::Pure {
            op: PureOp::Tuple(3), ..
        } => is_soa_tuple(&node.ty),
        _ => false,
    });
    assert!(
        has_repack,
        "expected a PureOp::Tuple(3) repack with SoA-tuple type; \
         ArrayWith split did not complete correctly"
    );
}
