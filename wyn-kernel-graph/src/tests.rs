use super::*;

type Graph = Builder<u32, u32, u32>;
fn leaf(id: u32) -> Fragment {
    Fragment::kernel(id.into(), id as u64)
}
fn graph(stages: u32) -> Graph {
    let mut graph = Graph::default();
    for stage in 0..stages {
        graph.register_stage(stage, None, stage as u64).unwrap();
    }
    graph
}
fn deps(plan: &Plan<u32, u32, u32>, kernel: u32) -> Vec<KernelId> {
    plan.kernel(kernel.into()).unwrap().dependencies().to_vec()
}

#[test]
fn single_sequence_parallel_and_nested_boundaries() {
    let mut graph = graph(2);
    graph.sequence_stages(0, 1).unwrap();
    let branches = Fragment::parallel(vec![
        Fragment::sequence(vec![leaf(1), leaf(2)]).unwrap(),
        Fragment::parallel(vec![leaf(3), leaf(4)]).unwrap(),
    ])
    .unwrap();
    graph.bind_stage(1, Fragment::sequence(vec![branches, leaf(5)]).unwrap()).unwrap();
    graph.bind_stage(0, leaf(0)).unwrap();
    let plan = graph.finalize().unwrap();
    assert_eq!(
        plan.kernel_order(),
        &[0.into(), 1.into(), 2.into(), 3.into(), 4.into(), 5.into()]
    );
    assert_eq!(deps(&plan, 1), [0.into()]);
    assert_eq!(deps(&plan, 3), [0.into()]);
    assert_eq!(deps(&plan, 4), [0.into()]);
    assert_eq!(deps(&plan, 5), [2.into(), 3.into(), 4.into()]);
    assert_eq!(plan.kernel(3.into()).unwrap().stage(), 1);
}

#[test]
fn complete_expansion_does_not_depend_on_construction_or_primary_identity() {
    for primary in 1..=3 {
        let mut plans = Vec::new();
        for reverse in [false, true] {
            let mut graph = graph(3);
            graph.sequence_stages(0, 1).unwrap();
            graph.sequence_stages(1, 2).unwrap();
            let expansion = Fragment::sequence(vec![leaf(1), leaf(2), leaf(3)]).unwrap();
            let mut stages = vec![(0, leaf(0)), (1, expansion), (2, leaf(4))];
            if reverse {
                stages.reverse();
            }
            for (stage, fragment) in stages {
                graph.bind_stage(stage, fragment).unwrap();
            }
            let plan = graph.finalize().unwrap();
            assert_eq!(plan.kernel(primary.into()).unwrap().stage(), 1);
            assert_eq!(deps(&plan, 1), [0.into()]);
            assert_eq!(deps(&plan, 4), [3.into()]);
            plans.push(plan);
        }
        assert_eq!(plans[0], plans[1]);
    }
}

#[test]
fn checked_mutations_leave_committed_topology_unchanged() {
    let mut graph = graph(3);
    graph.sequence_stages(0, 1).unwrap();
    graph.bind_stage(0, leaf(0)).unwrap();
    let baseline = graph.clone();
    for result in [
        graph.sequence_stages(1, 0),
        graph.sequence_stages(0, 0),
        graph.sequence_stages(0, 99),
        graph.bind_stage(1, leaf(0)),
        graph.bind_stage(0, leaf(5)),
        graph.register_stage(0, None, 0),
        graph.add_dependency(Endpoint::Kernel(99.into()), Endpoint::StageEntry(1)),
    ] {
        assert!(result.is_err());
        assert_eq!(graph, baseline);
    }
    assert_eq!(graph.clone().finalize(), Err(Error::UnboundStage));
    graph.bind_stage(1, leaf(1)).unwrap();
    graph.bind_stage(2, leaf(2)).unwrap();
    graph.connect_resource(7, 1.into(), 2.into()).unwrap();
    let baseline = graph.clone();
    assert!(graph.connect_resource(8, 2.into(), 0.into()).is_err());
    assert_eq!(graph, baseline);
}

#[test]
fn explicit_edges_to_nested_boundaries_and_dependency_deduplication() {
    let mut fragment =
        Fragment::parallel(vec![leaf(0), Fragment::parallel(vec![leaf(1), leaf(2)]).unwrap()]).unwrap();
    fragment
        .add_dependency(
            FragmentEndpoint::Completion(vec![0]),
            FragmentEndpoint::Entry(vec![1]),
        )
        .unwrap();
    fragment
        .add_dependency(
            FragmentEndpoint::Kernel(0.into()),
            FragmentEndpoint::Kernel(1.into()),
        )
        .unwrap();
    let baseline = fragment.clone();
    assert!(fragment
        .add_dependency(
            FragmentEndpoint::Kernel(1.into()),
            FragmentEndpoint::Kernel(0.into())
        )
        .is_err());
    assert_eq!(fragment, baseline);
    assert!(fragment
        .add_dependency(
            FragmentEndpoint::Entry(vec![9]),
            FragmentEndpoint::Completion(vec![])
        )
        .is_err());
    assert_eq!(fragment, baseline);
    assert!(Fragment::parallel(vec![leaf(0), leaf(0)]).is_err());
    assert!(Fragment::sequence(vec![]).is_err());
    let mut graph = graph(1);
    graph.bind_stage(0, fragment).unwrap();
    let plan = graph.finalize().unwrap();
    assert_eq!(deps(&plan, 1), [0.into()]);
    assert_eq!(deps(&plan, 2), [0.into()]);
}

#[test]
fn multiple_roots_and_leaves_connect_complete_stages() {
    let mut graph = graph(2);
    graph.sequence_stages(0, 1).unwrap();
    graph.bind_stage(0, Fragment::parallel(vec![leaf(0), leaf(1)]).unwrap()).unwrap();
    graph.bind_stage(1, Fragment::parallel(vec![leaf(2), leaf(3)]).unwrap()).unwrap();
    let plan = graph.finalize().unwrap();
    assert_eq!(deps(&plan, 2), [0.into(), 1.into()]);
    assert_eq!(deps(&plan, 3), [0.into(), 1.into()]);
}

#[test]
fn groups_coalesce_from_edges_and_keep_graphics_separate() {
    let mut graph = Graph::default();
    for group in 0..3 {
        graph.register_group(group, group as u64, group != 0).unwrap();
        graph.register_stage(group, Some(group), group as u64).unwrap();
        graph.bind_stage(group, leaf(group)).unwrap();
    }
    graph.sequence_stages(2, 1).unwrap();
    graph.sequence_stages(1, 0).unwrap();
    let plan = graph.finalize().unwrap();
    assert_eq!(plan.kernel_order(), &[2.into(), 1.into(), 0.into()]);
    assert_eq!(plan.groups()[0].members(), &[1, 2]);
    assert_eq!(plan.groups()[0].kernels(), &[2.into(), 1.into()]);
    assert_eq!(plan.groups()[1].id(), 0);
    assert_eq!(plan.groups()[1].dependencies(), &[1]);
}

#[test]
fn acyclic_kernels_can_have_invalid_publication_groups() {
    let mut graph = Graph::default();
    graph.register_group(0, 0, true).unwrap();
    graph.register_group(1, 1, false).unwrap();
    for (stage, group) in [(0, 0), (1, 1), (2, 0)] {
        graph.register_stage(stage, Some(group), stage as u64).unwrap();
        graph.bind_stage(stage, leaf(stage)).unwrap();
    }
    graph.sequence_stages(0, 1).unwrap();
    graph.sequence_stages(1, 2).unwrap();
    assert_eq!(graph.finalize(), Err(Error::PublicationCycle));
}
