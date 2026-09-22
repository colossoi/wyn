use super::{
    Access, BackingRef, Binding, BufferUsage, ComputePipeline, ComputeStage, DispatchLen, DispatchSize,
    FragmentOutput, FrameGraph, FrameResourceExtent, FrameResourceKind, GraphicsInvocation,
    GraphicsPipeline, GraphicsStage, ModuleInterface, Pipeline, ShaderStage, StageBindingUses,
    StorageImageFormat, StorageTextureSize, TextureSampleType, TextureViewDimension,
};

#[test]
fn frame_graph_aliases_storage_texture_views_and_orders_consumers() {
    let mut descriptor = ModuleInterface {
        scalar_tasks: vec![],
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
                    entry_point: "paint".to_string(),
                    owner: "paint".to_string(),
                    workgroup_size: (8, 8, 1),
                    dispatch_size: DispatchSize::DerivedFrom {
                        len: DispatchLen::StorageImage { set: 1, binding: 0 },
                        workgroup_size: 8,
                    },
                    uses: StageBindingUses::default(),
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
                    uses: StageBindingUses::default(),
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

    descriptor.rebuild_frame_graph();
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
}

#[test]
fn frame_graph_fragment_target_write_orders_downstream_reader() {
    // A fragment `#[target(scene_depth)]` write and a later pass that
    // samples a texture named `scene_depth` resolve to one resource, so the
    // reader depends on the fragment that produced it.
    let mut descriptor = ModuleInterface {
        scalar_tasks: vec![],
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
                    entry_point: "occ_reduce".to_string(),
                    owner: "occ_reduce".to_string(),
                    workgroup_size: (8, 8, 1),
                    dispatch_size: DispatchSize::Fixed {
                        x: 1,
                        y: 1,
                        z: 1,
                        explicit: false,
                    },
                    uses: StageBindingUses::default(),
                }],
                default_total_threads: None,
            }),
        ],
        source_results: Vec::new(),
        frame_graph: FrameGraph::default(),
    };

    descriptor.rebuild_frame_graph();
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
/// entry output, `consumer` reads it as an input. The edge follows the
/// bindings, so it is the same whichever order the passes are declared in.
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
        entry_point: entry.to_string(),
        owner: entry.to_string(),
        workgroup_size: (64, 1, 1),
        dispatch_size: DispatchSize::Fixed {
            x: 1,
            y: 1,
            z: 1,
            explicit: false,
        },
        uses: StageBindingUses::default(),
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
        pipelines,
        source_results: Vec::new(),
        frame_graph: FrameGraph::default(),
    };
    descriptor.rebuild_frame_graph();
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

/// A consumer that also overwrites, this frame, the state its producer reads
/// has no valid single-frame order. The graph says so instead of emitting an
/// order that runs one of them too early.
#[test]
fn frame_graph_reports_a_producer_consumer_cycle() {
    let inst = |access, usage| Binding::StorageBuffer {
        set: 0,
        binding: 0,
        access,
        usage,
        name: "inst".to_string(),
        resource: None,
        length: None,
        members: Vec::new(),
    };
    let occ = |access| Binding::StorageTexture {
        set: 1,
        binding: 0,
        name: "occ".to_string(),
        format: StorageImageFormat::R32Float,
        access,
        size: StorageTextureSize::SameAsWindow,
        resource: Some("occ".to_string()),
    };
    let stage = |entry: &str| ComputeStage {
        entry_point: entry.to_string(),
        owner: entry.to_string(),
        workgroup_size: (64, 1, 1),
        dispatch_size: DispatchSize::Fixed {
            x: 1,
            y: 1,
            z: 1,
            explicit: false,
        },
        uses: StageBindingUses::default(),
    };
    // `reduce` consumes `inst` and overwrites `occ`; `cull` produces `inst`
    // and reads `occ`. Declared reduce-first, so the hazard sweep also wants
    // `cull` after `reduce`.
    let mut descriptor = ModuleInterface {
        scalar_tasks: vec![],
        pipelines: vec![
            Pipeline::Compute(ComputePipeline {
                bindings: vec![inst(Access::ReadOnly, BufferUsage::Input), occ(Access::WriteOnly)],
                stages: vec![stage("reduce")],
                default_total_threads: None,
            }),
            Pipeline::Compute(ComputePipeline {
                bindings: vec![
                    inst(Access::WriteOnly, BufferUsage::Output),
                    occ(Access::ReadOnly),
                ],
                stages: vec![stage("cull")],
                default_total_threads: None,
            }),
        ],
        source_results: Vec::new(),
        frame_graph: FrameGraph::default(),
    };
    descriptor.rebuild_frame_graph();

    let cycle = descriptor
        .frame_graph
        .topological_order()
        .expect_err("reduce needs cull's `inst`, cull needs the `occ` reduce overwrites");
    assert_eq!(cycle.len(), 2, "both passes lie on the cycle: {cycle:?}");
}

#[test]
fn frame_graph_target_write_merges_with_storage_read_view() {
    // A fragment `#[target(gbuf)]` and a compute `#[view(gbuf,
    // storage_read)]` collapse to one texture-kind resource keyed by name,
    // even though the read binding is a storage texture. The compute depends
    // on the fragment.
    let mut descriptor = ModuleInterface {
        scalar_tasks: vec![],
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
                    entry_point: "reduce".to_string(),
                    owner: "reduce".to_string(),
                    workgroup_size: (8, 8, 1),
                    dispatch_size: DispatchSize::Fixed {
                        x: 1,
                        y: 1,
                        z: 1,
                        explicit: false,
                    },
                    uses: StageBindingUses::default(),
                }],
                default_total_threads: None,
            }),
        ],
        source_results: Vec::new(),
        frame_graph: FrameGraph::default(),
    };

    descriptor.rebuild_frame_graph();
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
