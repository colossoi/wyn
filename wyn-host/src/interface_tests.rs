use super::{
    Access, BackingRef, Binding, BufferUsage, ComputePipeline, ComputeStage, DispatchLoop, DispatchSize,
    FragmentOutput, FrameGraph, FramePass, FramePassKind, FrameResourceExtent, FrameResourceKind,
    GraphicsInvocation, GraphicsPipeline, GraphicsStage, ModuleInterface, Pipeline, ShaderStage,
    StageBindingUses, StorageImageFormat, StorageTextureSize, TextureSampleType, TextureViewDimension,
};
use crate::{Expr, ScalarExpr, ScalarSource};

#[test]
fn selected_loops_stay_contiguous_and_wait_for_outer_dependencies() {
    let dependencies = [vec![], vec![0, 4], vec![1], vec![], vec![3], vec![2]];
    let mut graph = FrameGraph {
        passes: dependencies
            .into_iter()
            .enumerate()
            .map(|(index, depends_on)| FramePass {
                name: format!("stage{index}"),
                kind: FramePassKind::Compute,
                pipeline_index: 0,
                stage_index: index,
                reads: vec![],
                writes: vec![],
                depends_on,
            })
            .collect(),
        ..FrameGraph::default()
    };
    let repeated = DispatchLoop {
        initial_length: ScalarExpr::I32(8),
        pipeline: 0,
        setup: 0,
        body: vec![1],
        completion: 2,
        count: ScalarExpr::I32(8),
        index: ScalarSource::Binding { set: 0, binding: 0 },
        current: ScalarSource::Binding { set: 0, binding: 1 },
        next: ScalarSource::Binding { set: 0, binding: 2 },
    };
    assert_eq!(graph.topological_order().unwrap(), [0, 3, 4, 1, 2, 5]);
    assert_eq!(
        graph.execution_order(&[]).unwrap(),
        graph.topological_order().unwrap()
    );
    assert_eq!(
        graph.execution_order(&[repeated.clone()]).unwrap(),
        [3, 4, 0, 1, 2, 5]
    );
    assert!(graph.execution_order(&[repeated.clone(), repeated.clone()]).is_err());
    let mut missing = repeated.clone();
    missing.body.push(6);
    assert!(graph.execution_order(&[missing]).is_err());
    // The flat graph is acyclic, but this outer pass would have to run inside the loop.
    graph.passes[4].depends_on.push(0);
    assert!(graph.topological_order().is_ok());
    assert!(graph.execution_order(&[repeated]).unwrap_err().contains("across dispatch loop boundaries"));
}

#[test]
fn frame_graph_aliases_storage_texture_views_and_orders_consumers() {
    let width = Expr::TextureDimension {
        source: ScalarSource::Binding { set: 1, binding: 0 },
        axis: 0,
    };
    let height = Expr::TextureDimension {
        source: ScalarSource::Binding { set: 1, binding: 0 },
        axis: 1,
    };
    let mut descriptor = ModuleInterface {
        scalar_tasks: vec![],
        dispatch_loops: vec![],
        pipelines: vec![
            Pipeline::Compute(ComputePipeline {
                bindings: vec![Binding::StorageTexture {
                    set: 1,
                    binding: 0,
                    name: "out_color".to_string(),
                    format: StorageImageFormat::Rgba32Float,
                    access: Access::WriteOnly,
                    size: StorageTextureSize::Fixed {
                        width: 64,
                        height: 32,
                    },
                    resource: None,
                }],
                stages: vec![ComputeStage {
                    dependencies: vec![],
                    entry_point: "paint".to_string(),
                    owner: "paint".to_string(),
                    workgroup_size: (8, 8, 1),
                    dispatch_size: DispatchSize::Computed {
                        elements: width.clone().multiply(height.clone()),
                        groups: [
                            width.ceiling(8).unwrap(),
                            height.ceiling(8).unwrap(),
                            Expr::Integer(1),
                        ],
                    },
                    uses: StageBindingUses {
                        reads: vec![],
                        writes: vec![0],
                    },
                }],
                default_total_threads: None,
            }),
            Pipeline::Graphics(GraphicsPipeline {
                source_operation: None,
                invocation: GraphicsInvocation::default(),
                stages: vec![GraphicsStage {
                    entry_point: "shade".to_string(),
                    owner: "shade".to_string(),
                    stage: ShaderStage::Fragment,
                    uses: StageBindingUses {
                        reads: vec![0],
                        writes: vec![],
                    },
                }],
                bindings: vec![Binding::Texture {
                    set: 2,
                    binding: 0,
                    name: "color_tex".to_string(),
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                    backing: Some(BackingRef { set: 1, binding: 0 }),
                    resource: None,
                }],
                vertex_inputs: vec![],
                fragment_outputs: vec![],
            }),
        ],
        source_results: Vec::new(),
        frame_graph: FrameGraph::default(),
    };

    descriptor.frame_graph =
        FrameGraph::from_selected_pipelines(&descriptor.pipelines, &[((0, 0), (1, 0))]).unwrap();
    let graph = &descriptor.frame_graph;
    assert_eq!(graph.resources.len(), 1);
    assert_eq!(graph.resources[0].kind, FrameResourceKind::StorageTexture);
    assert_eq!(graph.resources[0].first_pass, Some(0));
    assert_eq!(graph.resources[0].last_pass, Some(1));
    assert!(matches!(
        graph.resources[0].extent.as_ref(),
        Some(FrameResourceExtent::StorageTexture { size })
            if *size == (StorageTextureSize::Fixed {
                width: 64,
                height: 32
            })
    ));
    assert_eq!(graph.passes.len(), 2);
    assert_eq!(graph.passes[1].depends_on, vec![0]);

    let program = crate::Program::new(descriptor).unwrap();
    let entry = program.entries.iter().find(|entry| entry.name == "paint").unwrap();
    let [crate::Operation::Dispatch { groups, .. }] = entry.operations.as_slice() else {
        panic!("one texture dispatch");
    };
    assert_eq!(
        groups[0].to_whl().unwrap(),
        "(ceiling (i64 (gpu-texture-dimension resource-0 0 'width)) (i64 8))"
    );
    assert_eq!(
        groups[1].to_whl().unwrap(),
        "(ceiling (i64 (gpu-texture-dimension resource-0 0 'height)) (i64 8))"
    );
    assert_eq!(groups[2], Expr::Integer(1));
}

#[test]
fn frame_graph_fragment_target_write_orders_downstream_reader() {
    // A fragment `#[target(scene_depth)]` write and a later pass that
    // samples a texture named `scene_depth` resolve to one resource, so the
    // reader depends on the fragment that produced it.
    let mut descriptor = ModuleInterface {
        scalar_tasks: vec![],
        dispatch_loops: vec![],
        pipelines: vec![
            Pipeline::Graphics(GraphicsPipeline {
                source_operation: None,
                invocation: GraphicsInvocation::default(),
                stages: vec![GraphicsStage {
                    entry_point: "scene_fragment".to_string(),
                    owner: "scene_fragment".to_string(),
                    stage: ShaderStage::Fragment,
                    uses: StageBindingUses::default(),
                }],
                bindings: vec![],
                vertex_inputs: vec![],
                fragment_outputs: vec![FragmentOutput {
                    location: 0,
                    name: "scene_depth".to_string(),
                }],
            }),
            Pipeline::Compute(ComputePipeline {
                bindings: vec![Binding::Texture {
                    set: 0,
                    binding: 0,
                    name: "scene_depth".to_string(),
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                    backing: None,
                    resource: None,
                }],
                stages: vec![ComputeStage {
                    dependencies: vec![],
                    entry_point: "occ_reduce".to_string(),
                    owner: "occ_reduce".to_string(),
                    workgroup_size: (8, 8, 1),
                    dispatch_size: DispatchSize::Fixed {
                        x: 1,
                        y: 1,
                        z: 1,
                        explicit: false,
                    },
                    uses: StageBindingUses {
                        reads: vec![0],
                        writes: vec![],
                    },
                }],
                default_total_threads: None,
            }),
        ],
        source_results: Vec::new(),
        frame_graph: FrameGraph::default(),
    };

    descriptor.frame_graph =
        FrameGraph::from_selected_pipelines(&descriptor.pipelines, &[((0, 0), (1, 0))]).unwrap();
    let graph = &descriptor.frame_graph;

    // The render target and the sampled read are one resource.
    let depth: Vec<_> = graph.resources.iter().filter(|r| r.name == "scene_depth").collect();
    assert_eq!(depth.len(), 1, "target write and reader must share one resource");
    let depth_index = graph.resources.iter().position(|r| r.name == "scene_depth").unwrap();

    // The fragment pass writes it; the compute pass reads it and depends on
    // the fragment.
    let frag = &graph.passes[0];
    assert_eq!(frag.name, "scene_fragment");
    assert!(frag.writes.iter().any(|a| a.resource == depth_index));
    let reader = &graph.passes[1];
    assert_eq!(reader.name, "occ_reduce");
    assert!(reader.reads.iter().any(|a| a.resource == depth_index));
    assert_eq!(reader.depends_on, vec![0]);
}

/// Two compute passes sharing one storage buffer: `producer` writes it as an
/// entry output, `consumer` reads it as an input. The fixture selects the edge
/// in either declaration order.
fn producer_consumer_descriptor(producer_first: bool) -> ModuleInterface {
    let buffer = |name: &str, access: Access, usage: BufferUsage| Binding::StorageBuffer {
        set: 0,
        binding: 0,
        access,
        usage,
        name: name.to_string(),
        resource: Some("inst".to_string()),
        length: None,
        members: Vec::new(),
    };
    let stage = |entry: &str| ComputeStage {
        dependencies: vec![],
        entry_point: entry.to_string(),
        owner: entry.to_string(),
        workgroup_size: (64, 1, 1),
        dispatch_size: DispatchSize::Fixed {
            x: 1,
            y: 1,
            z: 1,
            explicit: false,
        },
        uses: StageBindingUses {
            reads: vec![0],
            writes: vec![],
        },
    };
    let pipeline = |entry: &str, access, usage| {
        Pipeline::Compute(ComputePipeline {
            bindings: vec![buffer(&format!("{entry}_binding"), access, usage)],
            stages: vec![stage(entry)],
            default_total_threads: None,
        })
    };
    let producer = pipeline("producer", Access::WriteOnly, BufferUsage::Output);
    let consumer = pipeline("consumer", Access::ReadOnly, BufferUsage::Input);
    let pipelines = if producer_first { vec![producer, consumer] } else { vec![consumer, producer] };

    let mut descriptor = ModuleInterface {
        scalar_tasks: vec![],
        dispatch_loops: vec![],
        pipelines,
        source_results: Vec::new(),
        frame_graph: FrameGraph::default(),
    };
    descriptor.frame_graph = FrameGraph::from_selected_pipelines(
        &descriptor.pipelines,
        &[if producer_first { ((0, 0), (1, 0)) } else { ((1, 0), (0, 0)) }],
    )
    .unwrap();
    descriptor
}

#[test]
fn frame_graph_orders_a_consumer_after_its_producer_in_either_declaration_order() {
    for producer_first in [true, false] {
        let descriptor = producer_consumer_descriptor(producer_first);
        let graph = &descriptor.frame_graph;
        let index = |name: &str| graph.passes.iter().position(|pass| pass.name == name).unwrap();
        let (producer, consumer) = (index("producer"), index("consumer"));

        assert!(
            graph.passes[consumer].depends_on.contains(&producer),
            "producer_first={producer_first}: consumer must depend on producer, got {:?}",
            graph.passes[consumer].depends_on
        );

        let order = graph.topological_order().expect("producer/consumer is acyclic");
        let position = |pass: usize| order.iter().position(|&p| p == pass).unwrap();
        assert!(
            position(producer) < position(consumer),
            "producer_first={producer_first}: schedule runs the consumer first: {order:?}"
        );
    }
}

#[test]
fn dispatch_sized_allocations_keep_the_unrounded_expression() {
    let mut descriptor = producer_consumer_descriptor(true);
    let Pipeline::Compute(producer) = &mut descriptor.pipelines[0] else {
        panic!("compute producer");
    };
    let Binding::StorageBuffer { length, .. } = &mut producer.bindings[0] else {
        panic!("output buffer");
    };
    *length = Some(super::BufferLen::SameAsDispatch { elem_bytes: 4 });
    let elements = Expr::Input("rows".into()).multiply(Expr::Input("columns".into()));
    producer.stages[0].dispatch_size = DispatchSize::linear(elements.clone(), 64).unwrap();
    descriptor.frame_graph = descriptor.frame_graph.refresh_resources(&descriptor.pipelines).unwrap();
    let program = crate::Program::new(descriptor).unwrap();
    let entry = program.entries.iter().find(|entry| entry.name == "producer").unwrap();
    let [crate::Allocation::Buffer { bytes, .. }] = entry.allocations.as_slice() else {
        panic!("one output allocation");
    };
    assert_eq!(bytes, &elements.multiply(Expr::Integer(4)));
    assert_eq!(
        entry.scalar_inputs,
        ["columns".into(), "rows".into()].into_iter().collect()
    );
    assert!(DispatchSize::linear(Expr::Integer(1), 0).is_err());
}

#[test]
fn frame_graph_reports_a_selected_cycle() {
    let descriptor = producer_consumer_descriptor(true);
    let graph =
        FrameGraph::from_selected_pipelines(&descriptor.pipelines, &[((0, 0), (1, 0)), ((1, 0), (0, 0))])
            .unwrap();
    assert_eq!(graph.topological_order().unwrap_err().len(), 2);
}

#[test]
fn frame_graph_target_write_merges_with_storage_read_view() {
    // A fragment `#[target(gbuf)]` and a compute `#[view(gbuf,
    // storage_read)]` collapse to one texture-kind resource keyed by name,
    // even though the read binding is a storage texture. The compute depends
    // on the fragment.
    let mut descriptor = ModuleInterface {
        scalar_tasks: vec![],
        dispatch_loops: vec![],
        pipelines: vec![
            Pipeline::Graphics(GraphicsPipeline {
                source_operation: None,
                invocation: GraphicsInvocation::default(),
                stages: vec![GraphicsStage {
                    entry_point: "frag".to_string(),
                    owner: "frag".to_string(),
                    stage: ShaderStage::Fragment,
                    uses: StageBindingUses::default(),
                }],
                bindings: vec![],
                vertex_inputs: vec![],
                fragment_outputs: vec![FragmentOutput {
                    location: 0,
                    name: "gbuf".to_string(),
                }],
            }),
            Pipeline::Compute(ComputePipeline {
                bindings: vec![Binding::StorageTexture {
                    set: 1,
                    binding: 0,
                    name: "g".to_string(),
                    format: StorageImageFormat::R32Float,
                    access: Access::ReadOnly,
                    size: StorageTextureSize::SameAsWindow,
                    resource: Some("gbuf".to_string()),
                }],
                stages: vec![ComputeStage {
                    dependencies: vec![],
                    entry_point: "reduce".to_string(),
                    owner: "reduce".to_string(),
                    workgroup_size: (8, 8, 1),
                    dispatch_size: DispatchSize::Fixed {
                        x: 1,
                        y: 1,
                        z: 1,
                        explicit: false,
                    },
                    uses: StageBindingUses {
                        reads: vec![0],
                        writes: vec![],
                    },
                }],
                default_total_threads: None,
            }),
        ],
        source_results: Vec::new(),
        frame_graph: FrameGraph::default(),
    };

    descriptor.frame_graph =
        FrameGraph::from_selected_pipelines(&descriptor.pipelines, &[((0, 0), (1, 0))]).unwrap();
    let graph = &descriptor.frame_graph;

    let gbuf: Vec<_> = graph.resources.iter().filter(|r| r.name == "gbuf").collect();
    assert_eq!(
        gbuf.len(),
        1,
        "storage read and target write must share one resource"
    );
    assert_eq!(gbuf[0].kind, FrameResourceKind::Texture);
    let idx = graph.resources.iter().position(|r| r.name == "gbuf").unwrap();
    assert!(graph.passes[0].writes.iter().any(|a| a.resource == idx));
    assert!(graph.passes[1].reads.iter().any(|a| a.resource == idx));
    assert_eq!(graph.passes[1].depends_on, vec![0]);
}

#[test]
fn selected_frame_graph_does_not_infer_accesses_or_dependencies() {
    let mut descriptor = producer_consumer_descriptor(false);
    for pipeline in &mut descriptor.pipelines {
        if let Pipeline::Compute(pipeline) = pipeline {
            for stage in &mut pipeline.stages {
                stage.uses = StageBindingUses::default();
            }
        }
    }
    let graph = FrameGraph::from_selected_pipelines(&descriptor.pipelines, &[]).unwrap();
    assert!(graph.passes.iter().all(|pass| pass.depends_on.is_empty()));
    assert!(graph.passes.iter().all(|pass| pass.reads.is_empty() && pass.writes.is_empty()));
    let explicit = FrameGraph::from_selected_pipelines(&descriptor.pipelines, &[((1, 0), (0, 0))]).unwrap();
    assert_eq!(explicit.passes[0].depends_on, vec![1]);
    let refreshed = explicit.refresh_resources(&descriptor.pipelines).unwrap();
    assert_eq!(refreshed.passes[0].depends_on, vec![1]);
    assert!(refreshed.passes[1].depends_on.is_empty());
}
