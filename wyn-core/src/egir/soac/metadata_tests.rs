use super::*;
use crate::ast::TypeName;
use crate::egir::graph_ops;
use crate::egir::program::{OutputSlotId, ResourceId, SemanticResourceRef};
use crate::egir::slice::ValueProducerPhase;
use crate::egir::soac::{filter, hist, remap, screma};
use crate::egir::types::{
    EGraph, EffectOp, Family, PureOp, Raw, ResourceAccess, SegExtent, SegResourceAccess, SegSpace,
    SideEffect, SideEffectKind, SoacEffect,
};
use crate::ssa::types::ConstantValue;
use crate::{BindingRef, FunctionId, LookupMap, LookupSet};
use polytype::Type;
use smallvec::smallvec;

fn scalar() -> Type<TypeName> {
    Type::Constructed(TypeName::Int(32), vec![])
}

fn node(index: u64) -> ValueId {
    ValueId::from(slotmap::KeyData::from_ffi(index))
}

fn place(index: u64) -> PlaceId {
    PlaceId::from(slotmap::KeyData::from_ffi(index))
}

fn lambda(index: u32, captures: Vec<OperandRef>) -> Lambda {
    Lambda::region(
        SegBody::new(FunctionId::from_index(index), captures),
        vec![],
        vec![],
    )
}

/// Distinct references make omissions visible even when other fields are
/// retained. Include every Hist update family and all three capture channels.
fn histogram(value: impl Fn(u64) -> ValueId, places: [PlaceId; 2]) -> hist::HistForm {
    hist::HistForm {
        bucket: lambda(
            10,
            vec![
                OperandRef::Value(value(1)),
                OperandRef::View(ViewId::test(value(2))),
                OperandRef::Place(places[0]),
            ],
        ),
        operations: vec![
            hist::HistOp {
                emission: hist::Emission::Guarded,
                shape: vec![value(3), value(4)],
                race_factor: value(5),
                destinations: vec![ViewId::test(value(6)), ViewId::test(value(7))],
                update: hist::Update::Reduce {
                    operator: lambda(
                        11,
                        vec![OperandRef::Value(value(8)), OperandRef::Place(places[1])],
                    ),
                    neutral: vec![value(9), value(10)],
                },
            },
            hist::HistOp {
                emission: hist::Emission::Always,
                shape: vec![value(11)],
                race_factor: value(12),
                destinations: vec![ViewId::test(value(13))],
                update: hist::Update::OrderedOverwrite {
                    value_types: vec![scalar()],
                },
            },
            hist::HistOp {
                emission: hist::Emission::Guarded,
                shape: vec![value(14)],
                race_factor: value(15),
                destinations: vec![ViewId::test(value(16))],
                update: hist::Update::BucketInsert {
                    value_types: vec![scalar()],
                    results: hist::BucketInsertResults {
                        counts: hist::HistResultId(0),
                        overflow: hist::HistResultId(1),
                    },
                    capacity: value(17),
                },
            },
        ],
    }
}

#[test]
fn raw_hist_slice_retains_metadata_producers_and_captured_place_dependencies() {
    let mut graph = EGraph::<Raw>::new();
    let block = graph.skeleton.entry;
    let mut values = LookupMap::new();
    for index in 1..=17 {
        let value = if [2, 6, 7, 13, 16].contains(&index) {
            graph_ops::intern_interface_view(&mut graph, BindingRef::new(0, index as u32), scalar(), None)
        } else {
            graph.intern_constant(ConstantValue::I32(index as i32), scalar())
        };
        values.insert(index, value);
    }
    // A neutral's producer must survive the live slice even though neither
    // the neutral nor its input is an ordinary Hist operand or capture.
    let neutral_source = values[&9];
    let neutral = graph.alloc_side_effect_result(scalar());
    let result = graph.value_result(neutral);
    graph.skeleton.blocks[block].side_effects.push(SideEffect {
        kind: SideEffectKind::Effect(EffectOp::Op {
            tag: PureOp::Materialize,
        }),
        operands: smallvec![OperandRef::Value(neutral_source)],
        result: Some(result),
        effects: None,
        span: None,
    });
    values.insert(9, neutral);
    let place_view = graph_ops::intern_interface_view(&mut graph, BindingRef::new(1, 0), scalar(), None);
    let place_index = graph.intern_constant(ConstantValue::I32(99), scalar());
    let captured_place = graph.add_view_index_place(graph.view_id(place_view), place_index, scalar(), None);
    let form = histogram(|index| values[&index], [captured_place; 2]);
    let result = graph.alloc_side_effect_result(Type::Constructed(TypeName::Bool, vec![]));
    let result_binding = graph.value_result(result);
    let effect = SideEffect {
        kind: SideEffectKind::Soac(SoacEffect(
            (),
            Soac::Hist(hist::Op {
                inputs: vec![],
                form,
                state: hist::RawState,
            }),
        )),
        operands: smallvec![],
        result: Some(result_binding),
        effects: None,
        span: None,
    };
    let metadata = Raw::effect_metadata_inputs(&effect).into_iter().collect::<LookupSet<_>>();
    assert_eq!(metadata, values.values().copied().collect());
    graph.skeleton.blocks[block].side_effects.push(effect);
    let unused = graph.intern_constant(ConstantValue::I32(1000), scalar());
    let analysis = graph_ops::GraphAnalysis::new(&graph);
    let live = graph_ops::value_producer_closure(&analysis, [result]);
    for value in values.values().copied().chain([neutral_source, place_view, place_index]) {
        assert!(
            live.values().contains(&value),
            "metadata dependency {value:?} was dropped"
        );
    }
    assert_eq!(
        live.operations().len(),
        2,
        "Hist and the neutral producer must both survive"
    );
    assert!(!live.values().contains(&unused));
}

#[test]
fn hist_remapping_preserves_capture_channels_and_maps_every_metadata_field() {
    let original = histogram(node, [place(1), place(2)]);
    let mut soac = SoacEffect(
        (),
        Soac::<Raw>::Hist(hist::Op {
            inputs: vec![],
            form: original.clone(),
            state: hist::RawState,
        }),
    );
    let nodes = (1..=17).map(|index| (node(index), node(index + 100))).collect::<LookupMap<_, _>>();
    Raw::remap_soac_values(&mut soac, &mut |value| nodes[&value]);
    let Soac::Hist(mapped) = &soac.1 else {
        unreachable!()
    };
    assert_eq!(
        mapped.form.bucket.captures(),
        &[
            OperandRef::Value(node(101)),
            OperandRef::View(ViewId::test(node(102))),
            OperandRef::Place(place(1)),
        ]
    );
    assert_eq!(
        mapped.form.metadata_values(),
        [101, 102, 108, 103, 104, 105, 106, 107, 109, 110, 111, 112, 113, 114, 115, 116, 117].map(node)
    );

    let places = [(place(1), place(101)), (place(2), place(102))].into_iter().collect();
    let remap = remap::Remap::<BindingRef, BindingRef, (), _>::new(&nodes, &places, Ok);
    let mapped = remap.hist_form(original);
    assert_eq!(mapped.bucket.captures()[2], OperandRef::Place(place(101)));
    let [reduce, overwrite, bucket] = mapped.operations.as_slice() else {
        unreachable!()
    };
    assert_eq!(reduce.shape, [node(103), node(104)]);
    assert_eq!(reduce.race_factor, node(105));
    assert_eq!(
        reduce.destinations,
        [ViewId::test(node(106)), ViewId::test(node(107))]
    );
    let hist::Update::Reduce { operator, neutral } = &reduce.update else {
        unreachable!()
    };
    assert_eq!(
        operator.captures(),
        &[OperandRef::Value(node(108)), OperandRef::Place(place(102))]
    );
    assert_eq!(neutral, &[node(109), node(110)]);
    assert_eq!(overwrite.shape, [node(111)]);
    assert_eq!(overwrite.race_factor, node(112));
    assert_eq!(overwrite.destinations, [ViewId::test(node(113))]);
    assert_eq!(bucket.shape, [node(114)]);
    assert_eq!(bucket.race_factor, node(115));
    assert_eq!(bucket.destinations, [ViewId::test(node(116))]);
    let hist::Update::BucketInsert {
        capacity, results, ..
    } = bucket.update
    else {
        unreachable!()
    };
    assert_eq!(capacity, node(117));
    assert_eq!(
        results,
        hist::BucketInsertResults {
            counts: hist::HistResultId(0),
            overflow: hist::HistResultId(1)
        }
    );
}

#[test]
fn screma_and_filter_resource_copy_maps_places_in_every_lambda() {
    let captures = vec![
        OperandRef::Value(node(1)),
        OperandRef::View(ViewId::test(node(2))),
        OperandRef::Place(place(1)),
    ];
    let expected = vec![
        OperandRef::Value(node(101)),
        OperandRef::View(ViewId::test(node(102))),
        OperandRef::Place(place(101)),
    ];
    let form = screma::ScremaForm {
        pre: lambda(1, captures.clone()),
        scans: vec![screma::Scan {
            operator: lambda(2, captures.clone()),
            neutral: vec![node(3)],
        }],
        reductions: vec![screma::Reduce {
            operator: lambda(3, captures.clone()),
            neutral: vec![node(4)],
            commutative: false,
        }],
        post: lambda(4, captures.clone()),
    };
    let nodes = (1..=4).map(|index| (node(index), node(index + 100))).collect();
    let places = [(place(1), place(101))].into_iter().collect();
    let remap = remap::Remap::<BindingRef, BindingRef, (), _>::new(&nodes, &places, Ok);
    let mapped = remap.screma_form(form);
    for lambda in [
        &mapped.pre,
        &mapped.scans[0].operator,
        &mapped.reductions[0].operator,
        &mapped.post,
    ] {
        assert_eq!(lambda.captures(), expected);
    }
    assert_eq!(mapped.scans[0].neutral, [node(103)]);
    assert_eq!(mapped.reductions[0].neutral, [node(104)]);
    let body = remap.filter_body(filter::Body {
        inputs: vec![],
        map: lambda(5, captures.clone()),
        predicate: lambda(6, captures),
    });
    assert_eq!(body.map.captures(), expected);
    assert_eq!(body.predicate.captures(), expected);
}

#[test]
fn region_indexes_keep_form_order_and_skip_identity_lambdas() {
    let mut hist = histogram(node, [place(1), place(2)]);
    hist.bucket = Lambda::identity(vec![]);
    let mut operations = vec![
        Soac::<Raw>::Screma(screma::Op {
            inputs: vec![],
            result_state: vec![],
            state: screma::RawState,
            form: screma::ScremaForm {
                pre: Lambda::identity(vec![]),
                scans: vec![screma::Scan {
                    operator: lambda(1, vec![]),
                    neutral: vec![],
                }],
                reductions: vec![screma::Reduce {
                    operator: lambda(2, vec![]),
                    neutral: vec![],
                    commutative: false,
                }],
                post: lambda(3, vec![]),
            },
        }),
        Soac::Filter(filter::Op {
            body: filter::Body {
                inputs: vec![],
                map: Lambda::identity(vec![]),
                predicate: lambda(4, vec![]),
            },
            state: filter::RawState {
                output: filter::RawOutput::Runtime {
                    capacity: filter::RuntimeCapacity::LikeInput {
                        input: filter::FilterInputId(0),
                    },
                },
            },
        }),
        Soac::Hist(hist::Op {
            inputs: vec![],
            form: hist,
            state: hist::RawState,
        }),
    ];
    for (op, expected) in operations.iter_mut().zip([vec![1, 2, 3], vec![4], vec![11]]) {
        assert_eq!(
            op.seg_bodies().iter().map(|body| body.region()).collect::<Vec<_>>(),
            expected.iter().copied().map(FunctionId::from_index).collect::<Vec<_>>()
        );
        for (index, region) in expected.iter().copied().enumerate() {
            let body = op.seg_body_mut(index).expect("matching region index");
            assert_eq!(body.region(), FunctionId::from_index(region));
            body.region = FunctionId::from_index(region + 100);
        }
        assert!(op.seg_body_mut(expected.len()).is_none());
        assert_eq!(
            op.seg_bodies().iter().map(|body| body.region()).collect::<Vec<_>>(),
            expected.into_iter().map(|region| FunctionId::from_index(region + 100)).collect::<Vec<_>>()
        );
    }
}

#[test]
fn resource_conversion_keeps_space_values_and_resource_identities_distinct() {
    let binding = BindingRef::new(2, 3);
    let resource = SemanticResourceRef(ResourceId::for_test(40));
    let nodes = (1..=4).map(|index| (node(index), node(index + 100))).collect();
    let places = LookupMap::new();
    let mut remap = remap::Remap::new(&nodes, &places, |input: BindingRef| {
        assert_eq!(input, binding);
        Ok::<_, ()>(resource)
    });
    let mapped = remap
        .segment(screma::Segmented {
            space: SegSpace::from_dims(vec![
                SegExtent::Fixed(8),
                SegExtent::HostProvided {
                    node: node(1),
                    inputs: vec![],
                },
                SegExtent::PushConstant {
                    node: node(2),
                    offset: 12,
                },
                SegExtent::Value(node(3)),
                SegExtent::ResourceLength {
                    view: ViewId::test(node(4)),
                    resource: binding,
                    elem_bytes: 4,
                },
            ])
            .unwrap(),
            output_slots: vec![OutputSlotId(5)],
            resources: vec![SegResourceAccess {
                resource: binding,
                access: ResourceAccess::ReadWrite,
            }],
        })
        .unwrap();
    assert_eq!(
        mapped.space.dims(),
        &[
            SegExtent::Fixed(8),
            SegExtent::HostProvided {
                node: node(101),
                inputs: vec![]
            },
            SegExtent::PushConstant {
                node: node(102),
                offset: 12
            },
            SegExtent::Value(node(103)),
            SegExtent::ResourceLength {
                view: ViewId::test(node(104)),
                resource,
                elem_bytes: 4
            },
        ]
    );
    assert_eq!(mapped.output_slots, [OutputSlotId(5)]);
    assert_eq!(mapped.resources[0].resource, resource);
    assert_eq!(mapped.resources[0].access, ResourceAccess::ReadWrite);
}
