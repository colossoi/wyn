#![allow(clippy::expect_used, clippy::unwrap_used)]

use super::*;
use crate::egir;
use crate::egir::builder::EntryBuilder;
use wyn_base::IdSource;

fn body(name: &str, identities: &mut egir::program::ProgramIdentities) -> PlannedEntry {
    let mut semantic_ids = egir::program::SemanticOpIdSource::default();
    let mut effect_ids = IdSource::new();
    EntryBuilder::new_compute(
        name.to_string(),
        (1, 1, 1),
        identities,
        &mut semantic_ids,
        &mut effect_ids,
    )
    .build()
}

fn spec(name: &str, label: &'static str, identities: &mut egir::program::ProgramIdentities) -> PhaseSpec {
    PhaseSpec::compute(
        body(name, identities),
        KernelDispatch::explicit(KernelDomain::Fixed { x: 1, y: 1, z: 1 }),
        label,
    )
}

fn draft(
    names: &[&str],
    identities: &mut egir::program::ProgramIdentities,
) -> (ScheduleBuilder, Vec<StageId>, BTreeMap<StageId, PlannedEntry>) {
    let mut stages = wyn_staged_ir::StagedIrBuilder::<(), (), ()>::new();
    let mut topology = TopologyBuilder::default();
    let mut metadata = BTreeMap::new();
    let mut bodies = BTreeMap::new();
    let mut kernel_ids = IdSource::new();
    let mut ids = Vec::new();
    for (rank, name) in names.iter().enumerate() {
        let stage = stages.add_stage((), ()).unwrap();
        topology.register_stage(stage, None, rank as u64).unwrap();
        metadata.insert(
            stage,
            StageMetadata {
                primary: kernel_ids.next_id(),
                compute: true,
                source_entry: None,
                dispatch: KernelDispatch::explicit(KernelDomain::Fixed { x: 1, y: 1, z: 1 }),
                output_routes: Vec::new(),
                required_elements: None,
                generated_kind: None,
            },
        );
        bodies.insert(stage, body(name, identities));
        ids.push(stage);
    }
    (
        ScheduleBuilder {
            topology,
            catalog: BTreeMap::new(),
            stages: metadata,
            pipelines: BTreeMap::new(),
            graphics_associations: BTreeMap::new(),
            source_entries: BTreeMap::new(),
            kernel_ids,
        },
        ids,
        bodies,
    )
}

fn install_unchanged(draft: &mut ScheduleBuilder, stage: StageId, body: PlannedEntry) {
    let recipe =
        PreparedRecipe::unchanged(draft.primary_kernel(stage), body, draft.stage_metadata(stage)).unwrap();
    draft.install_stage(stage, recipe).unwrap();
}

fn unchanged_recipe(draft: &ScheduleBuilder, stages: &StagedProgram, stage: StageId) -> PreparedRecipe {
    PreparedRecipe::unchanged(
        draft.primary_kernel(stage),
        PlannedEntry::project(stages.stage(stage).unwrap().body()).unwrap(),
        draft.stage_metadata(stage),
    )
    .unwrap()
}

fn plan(draft: ScheduleBuilder) -> KernelPlan {
    KernelPlan {
        topology: draft.topology.finalize().unwrap(),
        catalog: draft.catalog,
        pipelines: Vec::new(),
        source_entries: draft.source_entries,
    }
}

#[test]
fn body_preparation_retains_creator_supplied_facts() {
    let mut identities = egir::program::ProgramIdentities::default();
    let prepared = spec("kernel", "diagnostic_label", &mut identities).prepare().unwrap();
    assert_eq!(prepared.label, "diagnostic_label");
    assert_eq!(prepared.entry.name, "kernel");
    assert_eq!(
        prepared.dispatch,
        KernelDispatch::explicit(KernelDomain::Fixed { x: 1, y: 1, z: 1 })
    );

    let mut graphics = body("graphics", &mut identities);
    graphics.execution_model = ExecutionModel::Vertex;
    let prepared = PhaseSpec::graphics(
        graphics,
        KernelDispatch::inferred(KernelDomain::Fixed { x: 1, y: 1, z: 1 }),
    )
    .prepare()
    .unwrap();
    assert_eq!(prepared.label, "graphics_passthrough");
    assert_eq!(prepared.entry.name, "graphics");
}

#[test]
fn body_preparation_rejects_compute_graphics_mismatches() {
    let mut identities = egir::program::ProgramIdentities::default();
    assert!(PhaseSpec::graphics(
        body("compute", &mut identities),
        KernelDispatch::inferred(KernelDomain::Fixed { x: 1, y: 1, z: 1 })
    )
    .prepare()
    .is_err());

    let mut graphics = body("graphics", &mut identities);
    graphics.execution_model = ExecutionModel::Fragment;
    assert!(PhaseSpec::compute(
        graphics,
        KernelDispatch::inferred(KernelDomain::Fixed { x: 1, y: 1, z: 1 }),
        "compute"
    )
    .prepare()
    .is_err());
}

#[test]
fn validator_rejects_duplicate_names() {
    let mut identities = egir::program::ProgramIdentities::default();
    let (mut draft, stages, mut bodies) = draft(&["same", "same"], &mut identities);
    for stage in stages {
        install_unchanged(&mut draft, stage, bodies.remove(&stage).unwrap());
    }
    assert!(plan(draft).validate().unwrap_err().contains("duplicate physical entry"));
}

#[test]
fn validator_rejects_distinct_kernels_sharing_a_body_identity() {
    let mut identities = egir::program::ProgramIdentities::default();
    let (mut draft, stages, mut bodies) = draft(&["first", "second"], &mut identities);
    let shared = bodies[&stages[0]].id;
    bodies.get_mut(&stages[1]).unwrap().id = shared;
    for stage in stages {
        install_unchanged(&mut draft, stage, bodies.remove(&stage).unwrap());
    }
    assert!(plan(draft).validate().unwrap_err().contains("owned by multiple kernels"));
}

#[test]
fn prepared_recipes_keep_primary_dispatch_independent_of_boundaries() {
    for position in 0..3 {
        let mut identities = egir::program::ProgramIdentities::default();
        let (mut draft, stages, mut bodies) =
            draft(&["producer", "consumer", "following"], &mut identities);
        draft.topology.sequence_stages(stages[0], stages[1]).unwrap();
        draft.topology.sequence_stages(stages[1], stages[2]).unwrap();
        let primary = draft.primary_kernel(stages[1]);
        draft.stages.get_mut(&stages[1]).unwrap().dispatch =
            KernelDispatch::explicit(KernelDomain::Fixed { x: 1, y: 1, z: 1 });
        let mut ids = vec![draft.allocate_kernel(), draft.allocate_kernel()];
        ids.insert(position, primary);
        let specs = ids
            .iter()
            .enumerate()
            .map(|(index, &id)| {
                (
                    id,
                    PhaseSpec::compute(
                        body(&format!("phase{index}"), &mut identities),
                        KernelDispatch::inferred(KernelDomain::Fixed { x: 7, y: 1, z: 1 }),
                        "phase",
                    ),
                )
            })
            .collect();
        draft.install_stage(stages[1], PreparedRecipe::sequence(specs, primary).unwrap()).unwrap();
        install_unchanged(&mut draft, stages[2], bodies.remove(&stages[2]).unwrap());
        install_unchanged(&mut draft, stages[0], bodies.remove(&stages[0]).unwrap());
        let plan = plan(draft);
        plan.validate().unwrap();
        let graph = PhysicalKernelGraph::from(&plan);
        assert_eq!(graph.kernel(ids[0]).unwrap().dependencies, [KernelId::from(0)]);
        assert_eq!(graph.kernel(KernelId::from(2)).unwrap().dependencies, [ids[2]]);
        assert_eq!(
            plan.catalog[&primary].dispatch,
            KernelDispatch::explicit(KernelDomain::Fixed { x: 1, y: 1, z: 1 })
        );
        for id in ids.iter().filter(|&&id| id != primary) {
            assert_eq!(
                plan.catalog[id].dispatch.domain,
                KernelDomain::Fixed { x: 7, y: 1, z: 1 }
            );
        }
        let expected = graph.kernels().map(|kernel| kernel.entry).collect::<Vec<_>>();
        assert_eq!(
            plan.into_physical_entries().iter().map(|entry| entry.id).collect::<Vec<_>>(),
            expected
        );
    }
}

#[test]
fn projected_components_are_independent_complete_fragments() {
    let mut identities = egir::program::ProgramIdentities::default();
    let (mut draft, stages, mut bodies) = draft(&["root", "components"], &mut identities);
    draft.topology.sequence_stages(stages[0], stages[1]).unwrap();
    let primary = draft.primary_kernel(stages[1]);
    let branch = draft.allocate_kernel();
    let end = draft.allocate_kernel();
    let left = PreparedRecipe::single(primary, spec("left", "map", &mut identities)).unwrap();
    let right = PreparedRecipe::sequence(
        vec![
            (branch, spec("right", "filter_flags", &mut identities)),
            (end, spec("end", "filter_scatter", &mut identities)),
        ],
        end,
    )
    .unwrap();
    draft
        .install_stage(
            stages[1],
            PreparedRecipe::parallel(vec![left, right], primary).unwrap(),
        )
        .unwrap();
    install_unchanged(&mut draft, stages[0], bodies.remove(&stages[0]).unwrap());
    let plan = plan(draft);
    let graph = PhysicalKernelGraph::from(&plan);
    assert_eq!(graph.kernel(primary).unwrap().dependencies, [KernelId::from(0)]);
    assert_eq!(graph.kernel(branch).unwrap().dependencies, [KernelId::from(0)]);
    assert_eq!(graph.kernel(end).unwrap().dependencies, [branch]);
}

#[test]
fn preparation_and_topology_failures_do_not_install_partial_bodies() {
    let mut identities = egir::program::ProgramIdentities::default();
    let (mut draft, stages, mut bodies) = draft(&["first", "second"], &mut identities);
    let topology = draft.topology.clone();
    let primary = draft.primary_kernel(stages[0]);
    let second = draft.allocate_kernel();
    let mut invalid = body("invalid", &mut identities);
    invalid.execution_model = ExecutionModel::Fragment;
    assert!(PreparedRecipe::sequence(
        vec![
            (primary, spec("valid", "compute", &mut identities)),
            (
                second,
                PhaseSpec::compute(
                    invalid,
                    KernelDispatch::inferred(KernelDomain::Fixed { x: 1, y: 1, z: 1 }),
                    "bad"
                )
            )
        ],
        primary
    )
    .is_err());
    assert_eq!(draft.topology, topology);
    assert!(draft.catalog.is_empty());
    install_unchanged(&mut draft, stages[0], bodies.remove(&stages[0]).unwrap());
    let topology = draft.topology.clone();
    let other = draft.primary_kernel(stages[1]);
    let recipe = PreparedRecipe::sequence(
        vec![
            (other, spec("other", "compute", &mut identities)),
            (primary, spec("duplicate", "compute", &mut identities)),
        ],
        other,
    )
    .unwrap();
    assert!(draft.install_stage(stages[1], recipe).is_err());
    assert_eq!(draft.topology, topology);
    assert_eq!(draft.catalog.len(), 1);
    assert!(draft.stages.contains_key(&stages[1]));
}

#[test]
fn physical_graph_checks_body_ownership() {
    let mut identities = egir::program::ProgramIdentities::default();
    let (mut draft, stages, mut bodies) = draft(&["first", "second"], &mut identities);
    draft.topology.sequence_stages(stages[0], stages[1]).unwrap();
    for stage in stages {
        install_unchanged(&mut draft, stage, bodies.remove(&stage).unwrap());
    }
    let plan = plan(draft);
    let graph = PhysicalKernelGraph::from(&plan);
    assert!(graph.validate_entry_ids(graph.kernels().map(|kernel| kernel.entry)).is_ok());
    let first = graph.kernels().next().unwrap().entry;
    assert!(graph.validate_entry_ids([first]).is_err());
    assert!(graph.validate_entry_ids([first, first]).is_err());
}

fn allocated(
    name: &str,
    identities: &mut egir::program::ProgramIdentities,
) -> egir::program::AllocatedEntry {
    body(name, identities).into_inner()
}

fn compute_pipeline(names: &[&str]) -> Pipeline {
    Pipeline::Compute(ComputePipeline {
        bindings: Vec::new(),
        default_total_threads: None,
        stages: names
            .iter()
            .map(|name| ComputeStage {
                entry_point: (*name).into(),
                owner: (*name).into(),
                workgroup_size: (1, 1, 1),
                dispatch_size: DispatchSize::Fixed {
                    x: 1,
                    y: 1,
                    z: 1,
                    explicit: false,
                },
                uses: Default::default(),
            })
            .collect(),
    })
}

fn access(resource: u32, access: crate::ResourceAccess) -> SegResourceAccess<ResourceId> {
    SegResourceAccess {
        resource: ResourceId::for_test(resource),
        access,
    }
}

#[test]
fn generated_flow_connects_both_storage_resources_and_complete_consumers_in_either_order() {
    use crate::egir::program::{ResidentStorage, StageOrigin};
    use crate::ResourceAccess::{Read, Write};
    for reverse in [false, true] {
        let mut identities = egir::program::ProgramIdentities::default();
        let first = allocated("first", &mut identities);
        let second = allocated("second", &mut identities);
        let tail = allocated("tail", &mut identities);
        let associations = vec![vec![first.id, tail.id], vec![second.id]];
        let mut stages = wyn_staged_ir::StagedIrBuilder::new();
        let first = stages.add_stage(StageOrigin::Authored, first).unwrap();
        let second = stages.add_stage(StageOrigin::Authored, second).unwrap();
        let tail = stages.add_stage(StageOrigin::Authored, tail).unwrap();
        let producer = stages
            .add_stage(
                StageOrigin::Generated {
                    kind: GeneratedStageKind::RuntimeArray,
                    space: None,
                },
                allocated("producer", &mut identities),
            )
            .unwrap();
        let flow = stages
            .add_flow(
                producer,
                polytype::Type::Constructed(crate::ast::TypeName::Unit, Vec::new()),
                ResidentStorage {
                    data: ResourceId::for_test(0),
                    length: Some(ResourceId::for_test(1)),
                },
            )
            .unwrap();
        stages.add_consumer(flow, first).unwrap();
        stages.add_consumer(flow, second).unwrap();
        let stages = stages.finish().unwrap();
        let descriptor = PipelineDescriptor {
            pipelines: vec![
                compute_pipeline(&["first", "tail"]),
                compute_pipeline(&["second"]),
            ],
            ..Default::default()
        };
        let mut draft = ScheduleBuilder::from_descriptor(
            &descriptor,
            &associations,
            &LogicalResourceArena::default(),
            &stages,
        )
        .unwrap();
        let first_id = draft.primary_kernel(first);
        let first_start = draft.allocate_kernel();
        let second_id = draft.primary_kernel(second);
        let second_start = draft.allocate_kernel();
        let producer_id = draft.primary_kernel(producer);
        let length_id = draft.allocate_kernel();
        let tail_id = draft.primary_kernel(tail);
        let mut recipes = vec![
            (
                first,
                PreparedRecipe::sequence(
                    vec![
                        (first_start, spec("first_start", "start", &mut identities)),
                        (
                            first_id,
                            spec("first_read", "reader", &mut identities)
                                .with_resources(vec![access(0, Read), access(1, Read)]),
                        ),
                    ],
                    first_id,
                )
                .unwrap(),
            ),
            (
                second,
                PreparedRecipe::sequence(
                    vec![
                        (second_start, spec("second_start", "start", &mut identities)),
                        (
                            second_id,
                            spec("second_read", "reader", &mut identities)
                                .with_resources(vec![access(0, Read)]),
                        ),
                    ],
                    second_id,
                )
                .unwrap(),
            ),
            (
                producer,
                PreparedRecipe::parallel(
                    vec![
                        PreparedRecipe::single(
                            producer_id,
                            spec("write_data", "writer", &mut identities)
                                .with_resources(vec![access(0, Write)]),
                        )
                        .unwrap(),
                        PreparedRecipe::single(
                            length_id,
                            spec("write_length", "writer", &mut identities)
                                .with_resources(vec![access(1, Write)]),
                        )
                        .unwrap(),
                    ],
                    producer_id,
                )
                .unwrap(),
            ),
        ];
        if reverse {
            recipes.reverse();
        }
        for (stage, recipe) in recipes {
            draft.install_stage(stage, recipe).unwrap();
        }
        let recipe = unchanged_recipe(&draft, &stages, tail);
        draft.install_stage(tail, recipe).unwrap();
        let plan = draft.finish(&stages, false).unwrap();
        assert_eq!(
            plan.topology.groups().len(),
            1,
            "a shared producer connects both authored compute pipelines"
        );
        let edges = plan.topology.resource_edges();
        assert!(edges.contains(&(ResourceId::for_test(0), producer_id, first_id)));
        assert!(edges.contains(&(ResourceId::for_test(1), length_id, first_id)));
        assert!(edges.contains(&(ResourceId::for_test(0), producer_id, second_id)));
        for start in [first_start, second_start] {
            let deps = plan.topology.kernel(start).unwrap().dependencies();
            assert!(
                deps.contains(&producer_id) && deps.contains(&length_id),
                "the complete producer precedes every consumer entry"
            );
        }
        assert!(
            plan.topology.kernel(tail_id).unwrap().dependencies().contains(&first_id),
            "authored sequencing survives expansion and coalescing"
        );
        assert!(
            !plan.topology.kernel(second_start).unwrap().dependencies().contains(&first_id),
            "read/read sharing does not order consumers"
        );
        assert_eq!(plan.catalog[&producer_id].source_entry, Some(associations[0][0]));
        assert_eq!(
            plan.phase_ids_in(plan.pipelines[0].id),
            plan.topology.kernel_order()
        );
    }
}

#[test]
fn pruned_consumers_do_not_recreate_prerequisites_and_surviving_reads_require_writers() {
    use crate::egir::program::ResidentStorage;
    for (reads, writes) in [(false, false), (true, false), (true, true)] {
        let mut identities = egir::program::ProgramIdentities::default();
        let consumer = allocated("consumer", &mut identities);
        let association = vec![vec![consumer.id]];
        let mut stages = wyn_staged_ir::StagedIrBuilder::new();
        let consumer = stages.add_stage(StageOrigin::Authored, consumer).unwrap();
        let producer = stages
            .add_stage(
                StageOrigin::Generated {
                    kind: GeneratedStageKind::Scalar,
                    space: None,
                },
                allocated("producer", &mut identities),
            )
            .unwrap();
        let flow = stages
            .add_flow(
                producer,
                polytype::Type::Constructed(crate::ast::TypeName::Unit, Vec::new()),
                ResidentStorage {
                    data: ResourceId::for_test(0),
                    length: Some(ResourceId::for_test(1)),
                },
            )
            .unwrap();
        stages.add_consumer(flow, consumer).unwrap();
        let stages = stages.finish().unwrap();
        let descriptor = PipelineDescriptor {
            pipelines: vec![compute_pipeline(&["consumer"])],
            ..Default::default()
        };
        let mut draft = ScheduleBuilder::from_descriptor(
            &descriptor,
            &association,
            &LogicalResourceArena::default(),
            &stages,
        )
        .unwrap();
        let producer_id = draft.primary_kernel(producer);
        let consumer_id = draft.primary_kernel(consumer);
        let mut consumer_recipe = unchanged_recipe(&draft, &stages, consumer);
        let mut producer_recipe = unchanged_recipe(&draft, &stages, producer);
        // A length-only read is still a surviving resident dependency.
        if reads {
            consumer_recipe
                .bodies
                .get_mut(&consumer_id)
                .unwrap()
                .resources
                .push(access(1, crate::ResourceAccess::Read));
        }
        if writes {
            producer_recipe
                .bodies
                .get_mut(&producer_id)
                .unwrap()
                .resources
                .push(access(1, crate::ResourceAccess::Write));
        }
        draft.install_stage(consumer, consumer_recipe).unwrap();
        draft.install_stage(producer, producer_recipe).unwrap();
        let result = draft.finish(&stages, false);
        if reads && !writes {
            assert!(result.unwrap_err().to_string().contains("does not declare a writer"));
        } else {
            let plan = result.unwrap();
            assert_eq!(plan.topology.resource_edges().is_empty(), !reads);
            assert_eq!(
                plan.topology.kernel(consumer_id).unwrap().dependencies().contains(&producer_id),
                reads
            );
            assert_eq!(plan.topology.groups().len(), if reads { 1 } else { 2 });
        }
    }
}

#[test]
fn generated_work_serving_graphics_gets_a_separate_compute_publication() {
    use crate::egir::program::ResidentStorage;
    use crate::pipeline_descriptor::{GraphicsPipeline, GraphicsStage, ShaderStage};
    let mut identities = egir::program::ProgramIdentities::default();
    let mut vertex = allocated("vertex", &mut identities);
    vertex.execution_model = ExecutionModel::Vertex;
    let mut fragment = allocated("fragment", &mut identities);
    fragment.execution_model = ExecutionModel::Fragment;
    let associations = vec![vec![vertex.id, fragment.id]];
    let mut stages = wyn_staged_ir::StagedIrBuilder::new();
    // Register the fragment first to ensure publication uses the authored
    // graphics associations, independently of the staged DAG's source ranks.
    let fragment = stages.add_stage(StageOrigin::Authored, fragment).unwrap();
    let vertex = stages.add_stage(StageOrigin::Authored, vertex).unwrap();
    let generated = stages
        .add_stage(
            StageOrigin::Generated {
                kind: GeneratedStageKind::Scalar,
                space: None,
            },
            allocated("prepass", &mut identities),
        )
        .unwrap();
    let flow = stages
        .add_flow(
            generated,
            polytype::Type::Constructed(crate::ast::TypeName::Unit, Vec::new()),
            ResidentStorage {
                data: ResourceId::for_test(0),
                length: None,
            },
        )
        .unwrap();
    stages.add_consumer(flow, vertex).unwrap();
    let stages = stages.finish().unwrap();
    let descriptor = PipelineDescriptor {
        pipelines: vec![Pipeline::Graphics(GraphicsPipeline {
            stages: vec![
                GraphicsStage {
                    entry_point: "vertex".into(),
                    owner: "vertex".into(),
                    stage: ShaderStage::Vertex,
                    uses: Default::default(),
                },
                GraphicsStage {
                    entry_point: "fragment".into(),
                    owner: "fragment".into(),
                    stage: ShaderStage::Fragment,
                    uses: Default::default(),
                },
            ],
            invocation: Default::default(),
            bindings: Vec::new(),
            vertex_inputs: Vec::new(),
            fragment_outputs: Vec::new(),
        })],
        ..Default::default()
    };
    let mut draft = ScheduleBuilder::from_descriptor(
        &descriptor,
        &associations,
        &LogicalResourceArena::default(),
        &stages,
    )
    .unwrap();
    let writer = draft.primary_kernel(generated);
    let reader = draft.primary_kernel(vertex);
    for stage in [fragment, vertex, generated] {
        let mut recipe = unchanged_recipe(&draft, &stages, stage);
        if stage == generated {
            recipe.bodies.get_mut(&writer).unwrap().resources.push(access(0, crate::ResourceAccess::Write));
        } else if stage == vertex {
            recipe.bodies.get_mut(&reader).unwrap().resources.push(access(0, crate::ResourceAccess::Read));
        }
        draft.install_stage(stage, recipe).unwrap();
    }
    let plan = draft.finish(&stages, false).unwrap();
    assert!(matches!(plan.pipelines[0].template, Pipeline::Compute(_)));
    assert!(matches!(plan.pipelines[1].template, Pipeline::Graphics(_)));
    assert_eq!(plan.pipelines[1].graphics_entries, associations[0]);
    assert_eq!(plan.topology.groups()[1].dependencies(), &[plan.pipelines[0].id]);
    assert!(plan.topology.kernel(reader).unwrap().dependencies().contains(&writer));
    assert_eq!(plan.catalog[&writer].source_entry, Some(associations[0][0]));
}
