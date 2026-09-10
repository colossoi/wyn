use super::*;
type Graph = super::Graph<u8, u8, ()>;
type Builder = super::Builder<u8, u8>;

fn proposal(graph: &Graph, sources: &[GroupId]) -> Proposal<()> {
    let boundary = graph.boundary(sources, &[]).unwrap();
    Proposal {
        sources: sources.to_vec(),
        absorbed_values: vec![],
        results: boundary.outputs.into_iter().map(|port| vec![port]).collect(),
        payload: (),
        accounted_constraints: boundary.constraints,
    }
}

fn finish(builder: Builder) -> Graph {
    let payloads = builder.graph.groups.keys().map(|id| (*id, ())).collect::<Vec<_>>();
    builder.finish(payloads).unwrap()
}

fn chain() -> (Graph, [GroupId; 3]) {
    let mut builder = Builder::new();
    let input = builder.input();
    let a = builder.operation(0, vec![input], 1).unwrap();
    let a_out = builder.outputs(a).unwrap()[0];
    let b = builder.operation(0, vec![a_out, input], 1).unwrap();
    let b_out = builder.outputs(b).unwrap()[0];
    let c = builder.operation(0, vec![b_out], 1).unwrap();
    builder.observe(builder.outputs(c).unwrap()[0]).unwrap();
    (finish(builder), [a, b, c])
}

#[test]
fn successive_contractions_deduplicate_inputs_and_preserve_membership() {
    let (mut graph, [a, b, c]) = chain();
    let first = graph.contract(proposal(&graph, &[a, b])).unwrap();
    assert_eq!(graph.group(first.target).unwrap().inputs().len(), 1);
    assert_eq!(graph.group(a), Err(Error::StaleGroup));
    let second = graph.contract(proposal(&graph, &[first.target, c])).unwrap();
    let plan = graph.finalize().unwrap();
    assert_eq!(plan.groups().count(), 1);
    assert_eq!(plan.groups().next().unwrap().1.members().len(), 3);
    assert_eq!(plan.actions(), [first, second]);
}

#[test]
fn external_result_observers_are_preserved() {
    let mut builder = Builder::new();
    let a = builder.operation(0, vec![], 2).unwrap();
    let ports = builder.outputs(a).unwrap().to_vec();
    builder.observe(ports[0]).unwrap();
    let b = builder.operation(0, vec![ports[1]], 0).unwrap();
    let mut graph = finish(builder);
    let action = graph.contract(proposal(&graph, &[a, b])).unwrap();
    assert_eq!(action.outputs.len(), 1);
    assert_eq!(action.outputs[0].0, ports[0]);
    assert_eq!(
        graph.boundary(&[action.target], &[]).unwrap().outputs,
        graph.group(action.target).unwrap().outputs
    );
}

#[test]
fn rejection_is_atomic_including_identity_allocation() {
    let (mut graph, [a, b, c]) = chain();
    let before = format!("{graph:?}");
    assert_eq!(graph.contract(proposal(&graph, &[a, c])), Err(Error::Cycle));
    assert_eq!(format!("{graph:?}"), before);
    let mut missing = proposal(&graph, &[a, b]);
    missing.results.clear();
    assert_eq!(graph.contract(missing), Err(Error::Routing));
    assert_eq!(format!("{graph:?}"), before);
    let expected = graph.clone().contract(proposal(&graph, &[a, b])).unwrap();
    assert_eq!(graph.contract(proposal(&graph, &[a, b])).unwrap(), expected);
    let after = format!("{graph:?}");
    assert_eq!(
        graph.contract(Proposal {
            sources: vec![a, c],
            ..proposal(&graph, &[expected.target, c])
        }),
        Err(Error::StaleGroup)
    );
    assert_eq!(format!("{graph:?}"), after);
}

#[test]
fn scope_membership_and_ordering_are_checked() {
    let mut builder = Builder::new();
    let a = builder.operation(0, vec![], 0).unwrap();
    let b = builder.operation(1, vec![], 0).unwrap();
    let c = builder.operation(0, vec![], 0).unwrap();
    let edge = builder.order(a, c, OrderingReason::Resource(0)).unwrap();
    assert_eq!(builder.order(a, c, OrderingReason::Resource(0)).unwrap(), edge);
    builder.order(a, c, OrderingReason::Effect).unwrap();
    let mut graph = finish(builder);
    assert_eq!(graph.boundary(&[a, b], &[]), Err(Error::Scope));
    assert_eq!(graph.boundary(&[a, a], &[]), Err(Error::Membership));
    let mut missing = proposal(&graph, &[a, c]);
    missing.accounted_constraints.pop();
    let before = format!("{graph:?}");
    assert_eq!(graph.contract(missing), Err(Error::Ordering));
    assert_eq!(format!("{graph:?}"), before);
    graph.contract(proposal(&graph, &[a, c])).unwrap();
}

#[test]
fn unary_demand_rewrite_preserves_other_escaping_paths() {
    let mut builder = Builder::new();
    let input = builder.input();
    let index = builder.input();
    let a = builder.operation(0, vec![input], 1).unwrap();
    let array = builder.outputs(a).unwrap()[0];
    let point = builder.value(vec![array, index]).unwrap();
    builder.observe(point).unwrap();
    builder.observe(array).unwrap();
    let mut graph = finish(builder);
    let boundary = graph.boundary(&[a], &[point]).unwrap();
    assert_eq!(boundary.outputs, vec![array, point]);
    assert_eq!(boundary.inputs, vec![input, index]);
    let rewrite = Proposal {
        sources: vec![a],
        absorbed_values: vec![point],
        results: vec![vec![point]],
        payload: (),
        accounted_constraints: vec![],
    };
    assert_eq!(graph.rewrite(rewrite), Err(Error::Routing));
}

#[test]
fn exhaustive_four_node_dags_preserve_external_dependencies() {
    // All DAGs in a fixed topological labelling, every pair contraction, and
    // every external-observer subset. Relabellings do not change the quotient
    // criterion; stable source preference is checked separately below.
    for edges in 0u32..64 {
        for observers in 0u32..16 {
            let mut builder = Builder::new();
            let ids = (0..4).map(|_| builder.operation(0, vec![], 1).unwrap()).collect::<Vec<_>>();
            let ports = ids.iter().map(|id| builder.outputs(*id).unwrap()[0]).collect::<Vec<_>>();
            let mut bit = 0;
            let mut original = Vec::new();
            for after in 0..4 {
                let mut inputs = Vec::new();
                for before in 0..after {
                    if edges & (1 << bit) != 0 {
                        inputs.push(ports[before]);
                        original.push((before, after));
                    }
                    bit += 1;
                }
                builder.set_inputs(ids[after], inputs).unwrap();
                if observers & (1 << after) != 0 {
                    builder.observe(ports[after]).unwrap();
                }
            }
            let graph = finish(builder);
            for left in 0..4 {
                for right in left + 1..4 {
                    let mut candidate = graph.clone();
                    let result = candidate.contract(proposal(&graph, &[ids[left], ids[right]]));
                    match result {
                        Ok(action) => {
                            let map =
                                |i: usize| if i == left || i == right { action.target } else { ids[i] };
                            let first = candidate.clone().finalize().unwrap();
                            let second = candidate.finalize().unwrap();
                            let order = first.groups().map(|(id, _)| id).collect::<Vec<_>>();
                            assert_eq!(order, second.groups().map(|(id, _)| id).collect::<Vec<_>>());
                            for (before, after) in &original {
                                if map(*before) != map(*after) {
                                    assert!(
                                        order.iter().position(|id| *id == map(*before)).unwrap()
                                            < order.iter().position(|id| *id == map(*after)).unwrap()
                                    );
                                }
                            }
                            let members = first
                                .groups()
                                .flat_map(|(_, group)| group.members().iter().copied())
                                .collect::<Vec<_>>();
                            assert_eq!(members.len(), 4);
                            assert_eq!(members.into_iter().collect::<SortedSet<_>>().len(), 4);
                            for i in [left, right] {
                                if observers & (1 << i) != 0 {
                                    assert!(action.outputs.iter().any(|(old, _)| *old == ports[i]));
                                }
                            }
                        }
                        Err(Error::Cycle) => assert_eq!(format!("{candidate:?}"), format!("{graph:?}")),
                        Err(error) => panic!("unexpected rejection: {error}"),
                    }
                }
            }
        }
    }
}

#[test]
fn aliased_external_inputs_are_deduplicated_after_contraction() {
    let mut builder = Builder::new();
    let a = builder.operation(0, vec![], 1).unwrap();
    let b = builder.operation(0, vec![], 1).unwrap();
    let a_out = builder.outputs(a).unwrap()[0];
    let b_out = builder.outputs(b).unwrap()[0];
    let consumer = builder.operation(0, vec![a_out, b_out], 1).unwrap();
    builder.observe(builder.outputs(consumer).unwrap()[0]).unwrap();
    let mut graph = finish(builder);
    let mut contraction = proposal(&graph, &[a, b]);
    contraction.results = vec![contraction.results.into_iter().flatten().collect()];
    let action = graph.contract(contraction).unwrap();
    assert_eq!(
        graph.boundary(&[consumer], &[]).unwrap().inputs,
        graph.group(action.target).unwrap().outputs()
    );
    assert!(graph.depends_on(a_out, b_out).unwrap());
}

#[test]
fn payloads_are_complete_and_commit_only_with_accepted_contractions() {
    let mut builder = Builder::new();
    let a = builder.operation(0, vec![], 1).unwrap();
    let a_out = builder.outputs(a).unwrap()[0];
    let b = builder.operation(0, vec![a_out], 1).unwrap();
    let b_out = builder.outputs(b).unwrap()[0];
    let c = builder.operation(0, vec![b_out], 1).unwrap();
    let c_out = builder.outputs(c).unwrap()[0];
    builder.observe(c_out).unwrap();
    assert!(matches!(builder.clone().finish([(a, "a")]), Err(Error::Payload)));
    assert!(matches!(
        builder.clone().finish([(a, "a"), (a, "duplicate"), (b, "b"), (c, "c")]),
        Err(Error::Payload)
    ));
    let mut graph = builder.finish([(a, "a"), (b, "b"), (c, "c")]).unwrap();
    let rejected = Proposal {
        sources: vec![a, c],
        absorbed_values: vec![],
        results: vec![vec![a_out], vec![c_out]],
        accounted_constraints: vec![],
        payload: "rejected",
    };
    let before = format!("{graph:?}");
    assert_eq!(graph.contract(rejected), Err(Error::Cycle));
    assert_eq!(format!("{graph:?}"), before);
    let accepted = graph
        .contract(Proposal {
            sources: vec![a, b],
            absorbed_values: vec![],
            results: vec![vec![b_out]],
            accounted_constraints: vec![],
            payload: "ab",
        })
        .unwrap();
    assert_eq!(graph.group(a), Err(Error::StaleGroup));
    assert_eq!(graph.group(b), Err(Error::StaleGroup));
    assert_eq!(graph.group(accepted.target).unwrap().payload(), &"ab");
    assert_eq!(graph.finalize().unwrap().group(c).unwrap().payload(), &"c");
}

#[test]
fn result_consumers_stop_at_effects_and_follow_composed_aliases() {
    let mut builder = Builder::new();
    let a = builder.operation(0, vec![], 2).unwrap();
    let ports = builder.outputs(a).unwrap().to_vec();
    let projected = builder.value(vec![ports[0]]).unwrap();
    let b = builder.operation(0, vec![projected, projected], 1).unwrap();
    let b_out = builder.outputs(b).unwrap()[0];
    let c = builder.operation(0, vec![b_out, ports[1]], 0).unwrap();
    let mut graph = finish(builder);
    assert_eq!(
        graph.consumers(&[ports[0], ports[0]]).unwrap(),
        SortedSet::from([b])
    );
    assert_eq!(graph.consumers(&ports).unwrap(), SortedSet::from([b, c]));
    let action = graph.contract(proposal(&graph, &[a, b])).unwrap();
    assert_eq!(graph.consumers(&[ports[1], b_out]).unwrap(), SortedSet::from([c]));
    assert_eq!(
        graph.consumers(graph.group(action.target).unwrap().outputs()).unwrap(),
        SortedSet::from([c])
    );
}

#[test]
fn canonical_resource_hazards_preserve_reasons_and_scope() {
    let mut builder = Builder::new();
    let read = builder.operation(0, vec![], 0).unwrap();
    let read_again = builder.operation(0, vec![], 0).unwrap();
    let write = builder.operation(0, vec![], 0).unwrap();
    let read_after = builder.operation(0, vec![], 0).unwrap();
    let other_scope = builder.operation(1, vec![], 0).unwrap();
    for (id, resource, writes) in [
        (read, 1, false),
        (read_again, 1, false),
        (read_again, 2, true),
        (write, 1, true),
        (write, 1, true),
        (read_after, 1, false),
        (other_scope, 1, true),
    ] {
        builder.access(id, resource, writes).unwrap();
    }
    let mut graph = finish(builder);
    let edges =
        graph.constraints.values().map(|edge| (edge.before, edge.after, edge.reason)).collect::<Vec<_>>();
    assert_eq!(
        edges,
        vec![
            (read, write, OrderingReason::Resource(1)),
            (read_again, write, OrderingReason::Resource(1)),
            (write, read_after, OrderingReason::Resource(1))
        ]
    );
    assert!(graph.boundary(&[read, read_again], &[]).unwrap().constraints.is_empty());
    let mut unchecked = proposal(&graph, &[write, read_after]);
    unchecked.accounted_constraints.clear();
    assert_eq!(graph.contract(unchecked), Err(Error::Ordering));
    graph.contract(proposal(&graph, &[write, read_after])).unwrap();
    let plan = graph.finalize().unwrap();
    assert_eq!(plan.changed_scopes().collect::<Vec<_>>(), vec![0]);
    assert_eq!(
        plan.scopes().map(|(scope, _)| scope).collect::<Vec<_>>(),
        vec![0, 1]
    );
    assert!(!plan.changed(other_scope));
}

#[test]
fn final_schedules_and_replacements_resolve_successive_contractions() {
    let (mut graph, [a, b, c]) = chain();
    let b_out = graph.group(b).unwrap().outputs()[0];
    graph.observers.insert(b_out);
    let dropped = graph.group(a).unwrap().outputs()[0];
    let first = graph.contract(proposal(&graph, &[a, b])).unwrap();
    let intermediate = graph.group(first.target).unwrap().outputs()[0];
    let second = graph.contract(proposal(&graph, &[first.target, c])).unwrap();
    let final_port = graph.canonical(b_out).unwrap();
    let plan = graph.finalize().unwrap();
    assert_eq!(plan.changed_scopes().collect::<Vec<_>>(), vec![0]);
    assert_eq!(
        plan.scopes().collect::<Vec<_>>(),
        vec![(0, [second.target].as_slice())]
    );
    let replacements = plan.replacements().collect::<LookupMap<_, _>>();
    assert_eq!(replacements[&b_out], final_port);
    assert_eq!(replacements[&intermediate], final_port);
    assert!(!replacements.contains_key(&dropped));
    assert!(!plan.survives(dropped).unwrap());
    assert!(plan.survives(b_out).unwrap());
}

#[test]
fn result_routes_reject_ambiguous_origins_and_preserve_unary_aliases() {
    let mut builder = Builder::new();
    let a = builder.operation(0, vec![], 1).unwrap();
    let array = builder.outputs(a).unwrap()[0];
    let point = builder.value(vec![array]).unwrap();
    let other_point = builder.value(vec![array]).unwrap();
    builder.observe(point).unwrap();
    builder.observe(other_point).unwrap();
    let mut graph = finish(builder);
    let before = format!("{graph:?}");
    let rewrite = Proposal {
        sources: vec![a],
        absorbed_values: vec![point, other_point],
        results: vec![vec![point, other_point], vec![point]],
        accounted_constraints: vec![],
        payload: (),
    };
    assert_eq!(graph.rewrite(rewrite.clone()), Err(Error::Routing));
    assert_eq!(format!("{graph:?}"), before);
    assert_eq!(
        graph.rewrite(Proposal {
            results: vec![vec![PortId::from(u32::MAX)]],
            ..rewrite.clone()
        }),
        Err(Error::Port)
    );
    assert_eq!(format!("{graph:?}"), before);
    graph
        .rewrite(Proposal {
            results: vec![vec![point, other_point]],
            ..rewrite
        })
        .unwrap();
    let plan = graph.finalize().unwrap();
    assert_eq!(
        plan.canonical(point).unwrap(),
        plan.canonical(other_point).unwrap()
    );
    assert_eq!(plan.replacements().count(), 2);
    assert!(!plan.survives(array).unwrap());
}
