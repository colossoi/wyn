//! Structural tests for `egir::soac_expand`.
//!
//! These tests drive the pipeline up to `expand_soacs` and inspect the
//! resulting graph, including its addressable places and CFG state.

use crate::ast::Span;
use crate::ast::TypeName;
use crate::egir::graph_ops::bind_by_value_result;
use crate::egir::ir::PlaceOp;
use crate::egir::program::{Func, ProgramIdentities, SemanticOpId};
use crate::egir::soac::{hist, screma};
use crate::egir::types::{
    by_value_function_result, callable_parameter, CallEffects, EGraph, EffectOp, Family, Language,
    OperandRef, Physical, PureOp, SideEffectKind, SkeletonTerminator, Soac, SoacEffect, SoacInputType,
    ValueId, ValueKind, ViewId, WynLanguage,
};
use crate::BindingRef;
use polytype::Type;

/// Compile source through the pipeline to just-past `expand_soacs`,
/// returning the EGraph for the (single) entry point so tests can
/// introspect node structure.
fn compile_to_expanded_egraph(input: &str) -> EGraph<Physical> {
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
fn array_with_nodes<P: Family>(graph: &crate::egir::types::EGraph<P>) -> Vec<crate::egir::types::ValueId> {
    let inplace_id = crate::builtins::catalog().known().array_with_in_place;
    graph
        .nodes
        .iter()
        .filter_map(|(id, node)| match &node.kind {
            ValueKind::Pure {
                op: PureOp::Intrinsic { id: bid, .. },
                ..
            } if *bid == inplace_id => Some(id),
            _ => None,
        })
        .collect()
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

fn physical_callable(
    region: crate::FunctionId,
    name: &str,
    parameter_types: Vec<Type<TypeName>>,
    result_types: Vec<Type<TypeName>>,
) -> Func<Physical> {
    let mut graph = EGraph::<Physical>::new();
    let parameters = parameter_types
        .iter()
        .enumerate()
        .map(|(index, ty)| graph.add_test_value_parameter(index, ty.clone()))
        .collect::<Vec<_>>();
    let result_ty = if result_types.len() == 1 {
        result_types[0].clone()
    } else {
        Type::Constructed(TypeName::Tuple(result_types.len()), result_types.clone())
    };
    let result_abi = by_value_function_result::<WynLanguage>(result_ty.clone());
    let result_start = parameters.len() - result_types.len();
    let result = if result_types.len() == 1 {
        parameters[result_start]
    } else {
        graph.intern_pure(
            PureOp::Tuple(result_types.len()),
            parameters[result_start..].iter().copied().collect(),
            result_ty,
            None,
        )
    };
    let binding = bind_by_value_result(&mut graph, &result_abi, result);
    graph.skeleton.blocks[graph.skeleton.entry].term = SkeletonTerminator::Return(Some(binding));
    let params = parameter_types
        .into_iter()
        .enumerate()
        .map(|(index, ty)| callable_parameter::<BindingRef, WynLanguage>(format!("p{index}"), ty))
        .collect();
    Func::<Physical>::new(
        region,
        name.into(),
        Span::dummy(),
        None,
        params,
        result_abi,
        CallEffects::Pure,
        graph,
    )
}

#[test]
fn scatter_handleability_checks_every_input() {
    let mut identities = ProgramIdentities::default();
    let bucket_region = identities.alloc_function("scatter_bucket".into());
    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
    let f32_ty = Type::Constructed(TypeName::Float(32), vec![]);
    let bad_input_ty = Type::Constructed(TypeName::Tuple(2), vec![i32_ty.clone(), f32_ty.clone()]);
    let kind: SideEffectKind<Physical> = SideEffectKind::<Physical>::Soac(SoacEffect(
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
                    emission: hist::Emission::Always,
                    shape: vec![ValueId::from(slotmap::KeyData::from_ffi(1))],
                    race_factor: ValueId::from(slotmap::KeyData::from_ffi(2)),
                    destinations: vec![ViewId::test(ValueId::from(slotmap::KeyData::from_ffi(3)))],
                    update: hist::Update::OrderedOverwrite {
                        value_types: vec![f32_ty],
                    },
                }],
            },
            state: hist::ScheduledState::Serial,
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

    let mut graph = EGraph::<Physical>::new();
    let block = graph.skeleton.entry;
    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
    let bool_ty = Type::Constructed(TypeName::Bool, vec![]);
    let array_ty = plain_array_ty(i32_ty.clone());
    let zero = graph.intern_pure(PureOp::Int("0".into()), smallvec![], i32_ty.clone(), None);
    let one = graph.intern_pure(PureOp::Int("1".into()), smallvec![], i32_ty.clone(), None);
    let two = graph.intern_pure(PureOp::Int("2".into()), smallvec![], i32_ty.clone(), None);
    let four = graph.intern_pure(PureOp::Int("4".into()), smallvec![], i32_ty.clone(), None);
    let mut input_nodes = SmallVec::<[ValueId; 4]>::new();
    for _ in 0..6 {
        input_nodes.push(graph.intern_pure(
            PureOp::ArrayLit(4),
            smallvec![zero, zero, zero, zero],
            array_ty.clone(),
            None,
        ));
    }
    let destination_values = (0..3)
        .map(|binding| {
            graph_ops::intern_storage_view(
                &mut graph,
                crate::BindingRef::new(2, binding),
                i32_ty.clone(),
                None,
            )
        })
        .collect::<Vec<_>>();
    let destinations = destination_values.into_iter().map(|view| graph.view_id(view)).collect::<Vec<_>>();
    let mut regions = ProgramIdentities::default();
    let reducer_region = regions.alloc_function("hist_tuple_reducer".into());
    let histogram = hist::Op::<Physical> {
        inputs: (0..6).map(|_| SoacInputType::array(array_ty.clone())).collect(),
        form: hist::HistForm {
            bucket: screma::Lambda::identity(vec![i32_ty.clone(); 6]),
            operations: vec![
                hist::HistOp {
                    emission: hist::Emission::Always,
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
                    emission: hist::Emission::Always,
                    shape: vec![four],
                    race_factor: one,
                    destinations: destinations[2..].to_vec(),
                    update: hist::Update::OrderedOverwrite {
                        value_types: vec![i32_ty.clone()],
                    },
                },
            ],
        },
        state: hist::ScheduledState::Serial,
    };
    let mut effect_ids = crate::IdSource::new();
    let result = graph_ops::alloc_by_value_effect_result(&mut graph, bool_ty);
    graph_ops::emit_pending_soac(
        &mut graph,
        block,
        SemanticOpId::for_test(0),
        Soac::Hist(histogram),
        input_nodes.into_iter().map(OperandRef::Value).collect(),
        result,
        &mut effect_ids,
        None,
    );

    let reducer = physical_callable(
        reducer_region,
        "hist_tuple_reducer",
        vec![i32_ty.clone(); 4],
        vec![i32_ty.clone(); 2],
    );
    let callables = [(reducer_region, reducer)].into_iter().collect();
    let graph =
        super::run_one_body(graph, &callables, &mut effect_ids).expect("general serial Hist should expand");
    let stores = graph
        .skeleton
        .blocks
        .iter()
        .flat_map(|(_, block)| &block.side_effects)
        .filter(|effect| matches!(effect.kind, SideEffectKind::Effect(EffectOp::Store { .. })))
        .count();
    assert_eq!(
        stores, 12,
        "three destination components are stored across four unrolled iterations"
    );
    assert!(graph.skeleton.blocks.iter().all(|(_, block)| {
        block.side_effects.iter().all(|effect| !matches!(effect.kind, SideEffectKind::Soac(_)))
    }));
    let reducer_calls = graph
        .calls
        .iter()
        .filter(|(_, call)| call.callee() == reducer_region && call.arguments().len() == 4)
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
                ValueKind::Pure {
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
    use crate::egir::types::SkeletonTerminator;
    use smallvec::{smallvec, SmallVec};

    let mut graph = EGraph::<Physical>::new();
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
                emission: hist::Emission::Always,
                shape: vec![four],
                race_factor: one,
                destinations: vec![graph.view_id(destination)],
                update: hist::Update::OrderedOverwrite {
                    value_types: vec![i32_ty.clone()],
                },
            }],
        },
        state: hist::ScheduledState::Serial,
    };
    let mut effect_ids = crate::IdSource::new();
    let result = graph_ops::alloc_by_value_effect_result(&mut graph, bool_ty);
    graph_ops::emit_pending_soac(
        &mut graph,
        block,
        SemanticOpId::for_test(0),
        Soac::Hist(histogram),
        SmallVec::from_vec(vec![OperandRef::Value(indices), OperandRef::Value(values)]),
        result,
        &mut effect_ids,
        None,
    );

    let callables = Default::default();
    let graph =
        super::run_one_body(graph, &callables, &mut effect_ids).expect("serial histogram should expand");
    assert!(
        graph.nodes.iter().any(|(_, node)| {
            matches!(
                &node.kind,
                ValueKind::Pure {
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

    let mut graph = EGraph::<Physical>::new();
    let block = graph.skeleton.entry;
    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
    let bool_ty = Type::Constructed(TypeName::Bool, vec![]);
    let array_ty = plain_array_ty(i32_ty.clone());
    let zero = graph.intern_pure(PureOp::Int("0".into()), smallvec![], i32_ty.clone(), None);
    let one = graph.intern_pure(PureOp::Int("1".into()), smallvec![], i32_ty.clone(), None);
    let two = graph.intern_pure(PureOp::Int("2".into()), smallvec![], i32_ty.clone(), None);
    let four = graph.intern_pure(PureOp::Int("4".into()), smallvec![], i32_ty.clone(), None);
    let mut inputs = SmallVec::<[ValueId; 4]>::new();
    for _ in 0..5 {
        inputs.push(graph.intern_pure(
            PureOp::ArrayLit(4),
            smallvec![zero, zero, zero, zero],
            array_ty.clone(),
            None,
        ));
    }
    let destination_values = (0..2)
        .map(|binding| {
            graph_ops::intern_storage_view(
                &mut graph,
                crate::BindingRef::new(2, binding),
                i32_ty.clone(),
                None,
            )
        })
        .collect::<Vec<_>>();
    let destinations = destination_values.into_iter().map(|view| graph.view_id(view)).collect::<Vec<_>>();
    let mut regions = ProgramIdentities::default();
    let first_reducer = regions.alloc_function("first_reducer".into());
    let second_reducer = regions.alloc_function("second_reducer".into());
    let histogram = hist::Op::<Physical> {
        inputs: (0..5).map(|_| SoacInputType::array(array_ty.clone())).collect(),
        form: hist::HistForm {
            bucket: screma::Lambda::identity(vec![i32_ty.clone(); 5]),
            operations: vec![
                hist::HistOp {
                    emission: hist::Emission::Always,
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
                    emission: hist::Emission::Always,
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
        state: hist::ScheduledState::Atomic {
            space: SegSpace::from_dims(vec![SegExtent::Fixed(4)]).unwrap(),
            operations: vec![
                hist::AtomicUpdate::Direct(AtomicOp::Add),
                hist::AtomicUpdate::Direct(AtomicOp::Xor),
            ],
        },
    };
    let mut effect_ids = crate::IdSource::new();
    let result = graph_ops::alloc_by_value_effect_result(&mut graph, bool_ty);
    graph_ops::emit_pending_soac(
        &mut graph,
        block,
        SemanticOpId::for_test(0),
        Soac::Hist(histogram),
        inputs.into_iter().map(OperandRef::Value).collect(),
        result,
        &mut effect_ids,
        None,
    );

    let first = physical_callable(
        first_reducer,
        "first_reducer",
        vec![i32_ty.clone(); 2],
        vec![i32_ty.clone()],
    );
    let second = physical_callable(
        second_reducer,
        "second_reducer",
        vec![i32_ty.clone(); 2],
        vec![i32_ty.clone()],
    );
    let callables = [(first_reducer, first), (second_reducer, second)].into_iter().collect();
    let graph = super::run_one_body(graph, &callables, &mut effect_ids)
        .expect("multi-operation atomic Hist should expand");
    let atomics = graph
        .skeleton
        .blocks
        .iter()
        .flat_map(|(_, block)| &block.side_effects)
        .filter(|effect| matches!(effect.kind, SideEffectKind::Effect(EffectOp::Atomic { .. })))
        .count();
    assert_eq!(atomics, 2, "one atomic update per histogram operation");
    assert!(graph.skeleton.blocks.iter().all(|(_, block)| {
        block.side_effects.iter().all(|effect| {
            !matches!(
                effect.kind,
                SideEffectKind::Effect(EffectOp::Load { .. } | EffectOp::Store { .. })
                    | SideEffectKind::Soac(_)
            )
        })
    }));
    assert!(graph.nodes.iter().any(|(_, node)| {
        matches!(
            &node.kind,
            ValueKind::Pure {
                op: PureOp::BinOp(op),
                ..
            } if *op == crate::op::BinaryOperator::GreaterEqual
        )
    }));
    assert!(graph.nodes.iter().any(|(_, node)| {
        matches!(
            &node.kind,
            ValueKind::Pure {
                op: PureOp::BinOp(op),
                ..
            } if *op == crate::op::BinaryOperator::Multiply
        )
    }));
}
#[test]
fn map_array_of_mixed_tuple_writes_component_places_without_array_flow() {
    // Map output: [8](f32, i32, vec3f32).
    // After SoA, the output becomes ([8]f32, [8]i32, [8]vec3f32).
    // soac_expand should split the per-iteration write across three
    // addressable component arrays without carrying any array value through
    // the loop CFG.
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
    assert!(array_with_nodes(&graph).is_empty());

    let allocated = graph
        .skeleton
        .blocks
        .iter()
        .flat_map(|(_, block)| &block.side_effects)
        .filter_map(|effect| match effect.kind {
            SideEffectKind::<Physical>::Effect(EffectOp::Alloca { result }) => Some(result),
            _ => None,
        })
        .collect::<std::collections::HashSet<_>>();
    let stored_types = graph
        .skeleton
        .blocks
        .iter()
        .flat_map(|(_, block)| &block.side_effects)
        .filter_map(|effect| match effect.kind {
            SideEffectKind::<Physical>::Effect(EffectOp::Store { place }) => {
                let PlaceOp::Index { base, .. } = graph.place(place).op() else {
                    return None;
                };
                allocated.contains(base).then(|| graph.place(place).ty().pointee.clone())
            }
            _ => None,
        })
        .collect::<Vec<_>>();

    let f32_ty = Type::Constructed(TypeName::Float(32), vec![]);
    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
    let vec3_ty = Type::Constructed(
        TypeName::Vec,
        vec![f32_ty.clone(), Type::Constructed(TypeName::Size(3), vec![])],
    );
    assert!(stored_types.contains(&f32_ty));
    assert!(stored_types.contains(&i32_ty));
    assert!(stored_types.contains(&vec3_ty));

    for (_, block) in &graph.skeleton.blocks {
        for parameter in &block.params {
            assert!(!WynLanguage::is_materialized_aggregate(
                graph.value(parameter.value()).ty()
            ));
        }
    }
}
