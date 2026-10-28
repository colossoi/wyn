//! Cross-stage contracts on public emitted host/shader artifacts.
#[path = "support/pipeline.rs"]
mod pipeline;
use wyn_core::host::{
    self, Allocation, Binding, BufferLen, Operation, Pipeline, Program, ResultLayout, ResultScalar,
    ShaderFormat,
};
use wyn_core::{lower_ssa_to_spirv, lower_ssa_to_wgsl_with_program, CodegenTarget, PipelineTopologyPolicy};

fn host_outputs(source: &str) -> [Program; 2] {
    let source = pipeline::source(source);
    let ssa = pipeline::lower(&source, CodegenTarget::Portable);
    let spirv = lower_ssa_to_spirv(ssa.clone()).unwrap();
    let wgsl = lower_ssa_to_wgsl_with_program(ssa).unwrap();
    validate(
        naga::front::spv::Frontend::new(spirv.spirv.into_iter(), &Default::default()).parse().unwrap(),
    );
    validate(naga::front::wgsl::parse_str(&wgsl.wgsl).unwrap());
    [spirv.program, wgsl.program]
}
fn validate(module: naga::Module) {
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
}

#[test]
fn unrelated_compute_entries_preserve_graphics_inputs_and_resources() {
    for source in [
        include_str!("../../tests/compiler/library/graphics_entry_context_0.wyn"),
        include_str!("../../tests/compiler/library/graphics_entry_context_1.wyn"),
        include_str!("../../tests/compiler/library/graphics_entry_context_2.wyn"),
        include_str!("../../tests/compiler/library/graphics_entry_context_3.wyn"),
        include_str!("../../tests/compiler/library/graphics_entry_context_4.wyn"),
    ] {
        for (program, format) in
            host_outputs(source).into_iter().zip([ShaderFormat::Spirv, ShaderFormat::Wgsl])
        {
            let entry = program.entries.iter().find(|entry| entry.name == "draw").unwrap();
            let resources = &program.interface.frame_graph.resources;
            let inputs: Vec<_> = entry.inputs.iter().map(|id| resources[id.0].name.as_str()).collect();
            assert_eq!(inputs, ["positions", "colors", "target"], "{format:?}");
            let rust = program.to_rust_wgpu("entry_bindings", format).unwrap();
            let signature = rust.split_once("pub fn encode_draw(").unwrap().1.split_once(") ->").unwrap().0;
            let signature: String = signature.split_whitespace().collect();
            assert_eq!(signature, "context:&mutHostContext,encoder:&mutCommandEncoder,positions:&Buffer,colors:&Buffer,target:&Texture,");
        }
    }
}

#[test]
fn graphics_helper_returns_image_with_inline_array_results() {
    for (program, format) in host_outputs(include_str!(
        "../../tests/compiler/library/graphics_helper_results.wyn"
    ))
    .into_iter()
    .zip([ShaderFormat::Spirv, ShaderFormat::Wgsl])
    {
        for entry in &program.entries {
            assert_eq!(entry.inputs.len(), 1);
            assert_eq!(entry.results.len(), 2);
            assert_eq!(entry.results[1], *entry.inputs.first().unwrap());
            assert_ne!(entry.results[0], *entry.inputs.first().unwrap());
            assert!(entry.allocations.iter().any(|allocation|
                matches!(allocation, Allocation::Buffer { resource, .. } if *resource == entry.results[0])));
            assert_eq!(
                entry.operations.iter().filter(|op| matches!(op, Operation::Draw { .. })).count(),
                1
            );
            assert!(entry.operations.iter().any(|op| matches!(op, Operation::Dispatch { .. })));
            program.to_rust_wgpu("helper_tuple", format).unwrap();
        }
    }
}

#[test]
fn authored_outputs_consume_selected_capacities() {
    for (source, expected) in [
        (
            include_str!("../../tests/compiler/library/direct_capacity_0.wyn"),
            BufferLen::Fixed { bytes: 80 },
        ),
        (
            include_str!("../../tests/compiler/library/direct_capacity_1.wyn"),
            BufferLen::LikeInput {
                set: 0,
                binding: 0,
                elem_bytes: 4,
                src_elem_bytes: 4,
            },
        ),
        (
            include_str!("../../tests/compiler/library/direct_capacity_2.wyn"),
            BufferLen::LikeInput {
                set: 0,
                binding: 1,
                elem_bytes: 4,
                src_elem_bytes: 4,
            },
        ),
        (
            include_str!("../../tests/compiler/library/direct_capacity_3.wyn"),
            BufferLen::SameAsDispatch { elem_bytes: 4 },
        ),
    ] {
        let source = pipeline::source(source);
        let ssa = pipeline::lower_with_options(
            &source,
            CodegenTarget::Wgsl,
            wyn_core::egglog::ScalarOptimization::Basic,
            PipelineTopologyPolicy::AuthoredOnly,
        );
        let output = lower_ssa_to_wgsl_with_program(ssa).unwrap();
        validate(naga::front::wgsl::parse_str(&output.wgsl).unwrap());
        assert!(
            output.program.interface.pipelines.iter().any(|pipeline| {
                let Pipeline::Compute(pipeline) = pipeline else {
                    return false;
                };
                pipeline.bindings.iter().any(|binding| {
                    matches!(binding,
                Binding::StorageBuffer { usage: host::BufferUsage::Output, length: Some(actual), .. }
                if *actual == expected)
                })
            }),
            "expected {expected:?}"
        );
    }
}

#[test]
fn render_target_load_fetches_only_fields_used_by_the_fragment_output() {
    use crate::host::{Binding, Pipeline};
    use std::collections::{HashMap, HashSet};
    use wspirv::dr::Operand;
    use wspirv::spirv::{Decoration, ExecutionModel, Op};

    for (source, expected_names) in [
        (
            include_str!("../../tests/compiler/library/render_target_field_0.wyn"),
            vec!["scene_depth"],
        ),
        (
            include_str!("../../tests/compiler/library/render_target_field_1.wyn"),
            vec!["scene_unused"],
        ),
        (
            include_str!("../../tests/compiler/library/render_target_field_2.wyn"),
            vec!["scene_unused", "scene_depth"],
        ),
        (
            include_str!("../../tests/compiler/library/render_target_field_3.wyn"),
            vec!["scene"],
        ),
    ] {
        let prepared = pipeline::source(source);
        let lowered = lower_ssa_to_spirv(pipeline::lower(&prepared, CodegenTarget::Spirv)).unwrap();
        let bytes: Vec<_> = lowered.spirv.iter().flat_map(|word| word.to_le_bytes()).collect();
        let validated = naga::front::spv::parse_u8_slice(&bytes, &Default::default()).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&validated)
        .unwrap();

        let module = wspirv::dr::load_words(&lowered.spirv).unwrap();
        let entry = module
            .entry_points
            .iter()
            .find(|inst| inst.operands.first() == Some(&Operand::ExecutionModel(ExecutionModel::Fragment)))
            .unwrap();
        let Operand::IdRef(fragment_id) = entry.operands[1] else {
            panic!("fragment function")
        };
        let fragment = module
            .functions
            .iter()
            .find(|f| f.def.as_ref().unwrap().result_id == Some(fragment_id))
            .unwrap();
        let instructions: Vec<_> = fragment.blocks.iter().flat_map(|block| &block.instructions).collect();
        let definitions: HashMap<_, _> =
            instructions.iter().filter_map(|inst| inst.result_id.map(|id| (id, *inst))).collect();
        let fetches: Vec<_> =
            instructions.iter().filter(|inst| inst.class.opcode == Op::ImageFetch).collect();
        assert_eq!(fetches.len(), expected_names.len(), "{source}");

        // Trace each fetch through its image load and decorations to the named
        // attachment in the descriptor; neither result IDs nor binding slots
        // are fixed by the test.
        let decoration = |id, kind| {
            module
                .annotations
                .iter()
                .find_map(|inst| match inst.operands.as_slice() {
                    [Operand::IdRef(target), Operand::Decoration(actual), Operand::LiteralBit32(value)]
                        if inst.class.opcode == Op::Decorate && *target == id && *actual == kind =>
                    {
                        Some(*value)
                    }
                    _ => None,
                })
                .unwrap()
        };
        let mut fetched_names = Vec::new();
        for fetch in &fetches {
            let fetch_id = fetch.result_id.unwrap();
            assert!(
                instructions.iter().any(|inst| inst.operands.contains(&Operand::IdRef(fetch_id))),
                "unused fetch %{fetch_id}"
            );
            let Operand::IdRef(image) = fetch.operands[0] else {
                panic!("fetch image")
            };
            let load = definitions[&image];
            assert_eq!(load.class.opcode, Op::Load);
            let Operand::IdRef(variable) = load.operands[0] else {
                panic!("image variable")
            };
            let set = decoration(variable, Decoration::DescriptorSet);
            let binding = decoration(variable, Decoration::Binding);
            let name = lowered
                .program
                .interface
                .pipelines
                .iter()
                .filter_map(|p| match p {
                    Pipeline::Graphics(p) => Some(&p.bindings),
                    _ => None,
                })
                .flatten()
                .find_map(|b| match b {
                    Binding::Texture {
                        set: s,
                        binding: b,
                        name,
                        ..
                    } if *s == set && *b == binding => Some(name.as_str()),
                    _ => None,
                })
                .unwrap();
            fetched_names.push(name);
        }
        fetched_names.sort_unstable();
        let mut expected_names = expected_names;
        expected_names.sort_unstable();
        assert_eq!(fetched_names, expected_names, "{source}");

        // Every retained fetch must contribute to the fragment's output store.
        let stores: Vec<_> = instructions.iter().filter(|inst| inst.class.opcode == Op::Store).collect();
        assert_eq!(stores.len(), 1);
        let Operand::IdRef(output) = stores[0].operands[1] else {
            panic!("output value")
        };
        let mut pending = vec![output];
        let mut live = HashSet::new();
        while let Some(id) = pending.pop() {
            if live.insert(id) {
                if let Some(inst) = definitions.get(&id) {
                    pending.extend(inst.operands.iter().filter_map(|operand| match operand {
                        Operand::IdRef(id) => Some(*id),
                        _ => None,
                    }));
                }
            }
        }
        assert!(fetches.iter().all(|inst| live.contains(&inst.result_id.unwrap())));
    }
}

#[test]
fn fluid_leaf_bounds_are_computed_outside_the_bfs() {
    stacker::grow(32 * 1024 * 1024, || {
        // Use the simulator's implementation, including its queue sizes and guards.
        let neighbors = include_str!("../../fluid-simulation/packages/spatial/src/neighbors.wyn")
            .replace(
                "module Tree = import \"octree\"",
                &format!(
                    "module Tree = {{ {} }}",
                    include_str!("../../fluid-simulation/packages/spatial/src/octree.wyn")
                ),
            )
            .replace(
                "module Hilbert = import \"hilbert\"",
                &format!(
                    "module Hilbert = {{ {} }}",
                    include_str!("../../fluid-simulation/packages/spatial/src/hilbert.wyn")
                ),
            );
        let source = format!(
            "{neighbors}\n{}",
            include_str!("../../tests/compiler/library/fluid_neighbor_entry.wyn")
        );
        placement_shaders(&source, |module| {
            let mut bounds_depths = Vec::new();
            let mut queues = 0;
            for function in module
                .functions
                .iter()
                .map(|(_, f)| f)
                .chain(module.entry_points.iter().map(|e| &e.function))
            {
                let queue_variables: Vec<_> = function
                    .local_variables
                    .iter()
                    .filter_map(|(id, v)| {
                        matches!(module.types[v.ty].inner,
                        naga::TypeInner::Array { base, .. } if matches!(module.types[base].inner,
                            naga::TypeInner::Vector { size: naga::VectorSize::Bi, scalar }
                                if scalar.kind == naga::ScalarKind::Uint))
                        .then_some(id)
                    })
                    .collect();
                queues += queue_variables.len();
                let whole_queue = |pointer| {
                    matches!(function.expressions[pointer],
                    naga::Expression::LocalVariable(v) if queue_variables.contains(&v))
                };
                assert!(!function.expressions.iter().any(
                    |(_, e)| matches!(e, naga::Expression::Load { pointer } if whole_queue(*pointer))
                ));
                visit_loop_depths(&function.body, 0, &mut |statement, depth| {
                    if let naga::Statement::Store { pointer, .. } = statement {
                        assert!(!whole_queue(*pointer), "BFS must update queue elements in place");
                    }
                    if let naga::Statement::Emit(range) = statement {
                        for value in range.clone() {
                            if matches!(
                                function.expressions[value],
                                naga::Expression::Math {
                                    fun: naga::MathFunction::Max,
                                    ..
                                }
                            ) {
                                bounds_depths.push(depth);
                            }
                        }
                    }
                });
            }
            // Max updates the upper bound inside the bounds loop, never in a
            // second enclosing loop (the BFS). Also reject duplicated bounds scans.
            assert_eq!(bounds_depths, vec![1]);
            assert_eq!(queues, 1);
        });
    });
}

#[test]
fn scalar_results_keep_public_layouts_and_generated_rust_contracts() {
    let source = pipeline::source(include_str!(
        "../../tests/compiler/library/host_scalar_output_descriptors.wyn"
    ));
    let program =
        lower_ssa_to_wgsl_with_program(pipeline::lower(&source, CodegenTarget::Wgsl)).unwrap().program;
    let rust = program.to_rust_wgpu("results.wgsl", ShaderFormat::Wgsl).unwrap();
    for (name, scalar) in [
        ("scalar_i32", ResultScalar::I32),
        ("scalar_u32", ResultScalar::U32),
        ("scalar_f32", ResultScalar::F32),
    ] {
        let result = program.interface.source_results.iter().find(|result| result.name == name).unwrap();
        assert_eq!(result.layout, ResultLayout::Scalar(scalar));
        assert!(rust.contains(&format!("name: {name:?}")));
        assert!(rust.contains(&format!("ResultLayout::Scalar(ResultScalar::{scalar:?})")));
    }
    for text in [
        "-> Result<OutputDescriptor, HostError>",
        "size: 4u64",
        "Buffer::clone(",
        "pub mod output",
    ] {
        assert!(rust.contains(text), "missing {text}");
    }
    for text in [
        "pub fn read_",
        "map_async",
        "from_le_bytes",
        "count_host: &[u8]",
        "read_gpu_word",
    ] {
        assert!(!rust.contains(text), "unexpected {text}");
    }
}

#[test]
fn allocated_results_keep_source_fields_and_rust_parameter_contracts() {
    let source = pipeline::source(include_str!(
        "../../tests/compiler/library/host_array_output_descriptors.wyn"
    ));
    let program =
        lower_ssa_to_wgsl_with_program(pipeline::lower(&source, CodegenTarget::Wgsl)).unwrap().program;
    let rust = program.to_rust_wgpu("results.wgsl", ShaderFormat::Wgsl).unwrap();
    assert!(rust.contains("count: &ParameterBuffer"));
    assert!(!rust.contains("count_host: &[u8]"));
    assert!(!rust.contains("read_gpu_word"));
    for (name, fields) in [
        ("generated_array", &["generated_array"][..]),
        ("generated_tuple", &["result_0", "result_1"][..]),
        ("generated_record", &["ao", "coarse"][..]),
    ] {
        let entry = program.entries.iter().find(|entry| entry.name == name).unwrap();
        assert!(entry.results.iter().all(|id| !entry.inputs.contains(id)));
        assert!(entry.results.iter().all(|id| entry.allocations.iter().any(|a| a.resource() == *id)));
        for field in fields {
            assert!(rust.contains(&format!("name: {field:?}")), "missing {field}");
        }
    }
}

#[test]
fn direct_mode_keeps_collectives_in_the_authored_entry() {
    for source in [
        include_str!("../../tests/compiler/library/direct_collective_0.wyn"),
        include_str!("../../tests/compiler/library/direct_collective_1.wyn"),
        include_str!("../../testfiles/unified_triangle.wyn"),
    ] {
        let prepared = pipeline::source(source);
        let ssa = pipeline::lower_with_options(
            &prepared,
            CodegenTarget::Wgsl,
            wyn_core::egglog::ScalarOptimization::Basic,
            PipelineTopologyPolicy::AuthoredOnly,
        );
        let output = lower_ssa_to_wgsl_with_program(ssa).unwrap();
        for pipeline in &output.program.interface.pipelines {
            if let Pipeline::Compute(p) = pipeline {
                assert_eq!(p.stages.len(), 1);
            }
        }
        let module = naga::front::wgsl::parse_str(&output.wgsl).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }
}

#[test]
fn independent_work_and_successive_loops_reach_host_backends() {
    for (program, format) in host_outputs(include_str!(
        "../../tests/compiler/library/successive_host_loops.wyn"
    ))
    .into_iter()
    .zip([ShaderFormat::Spirv, ShaderFormat::Wgsl])
    {
        assert_eq!(
            program.entries[0].operations.iter().filter(|op| matches!(op, Operation::Loop { .. })).count(),
            2
        );
        program.to_rust_wgpu("loops", format).unwrap();
        program.to_whl("loops", format).unwrap();
    }
}

fn placement_shaders(source: &str, mut inspect: impl FnMut(&naga::Module)) {
    let source = pipeline::source(source);
    for policy in [
        wyn_core::egglog::ScalarOptimization::Basic,
        wyn_core::egglog::ScalarOptimization::Full,
    ] {
        let ssa = pipeline::lower_with_options(
            &source,
            CodegenTarget::Portable,
            policy,
            PipelineTopologyPolicy::AllowGenerated,
        );
        let spirv = lower_ssa_to_spirv(ssa.clone()).unwrap();
        let module =
            naga::front::spv::Frontend::new(spirv.spirv.into_iter(), &Default::default()).parse().unwrap();
        inspect(&module);
        validate(module);
        let wgsl = lower_ssa_to_wgsl_with_program(ssa).unwrap();
        let module = naga::front::wgsl::parse_str(&wgsl.wgsl).unwrap();
        inspect(&module);
        validate(module);
    }
}
fn visit_loop_depths(block: &naga::Block, depth: usize, visit: &mut impl FnMut(&naga::Statement, usize)) {
    for statement in block.iter() {
        visit(statement, depth);
        match statement {
            naga::Statement::Loop { body, continuing, .. } => {
                visit_loop_depths(body, depth + 1, visit);
                visit_loop_depths(continuing, depth + 1, visit);
            }
            naga::Statement::Block(block) => visit_loop_depths(block, depth, visit),
            naga::Statement::If { accept, reject, .. } => {
                visit_loop_depths(accept, depth, visit);
                visit_loop_depths(reject, depth, visit);
            }
            naga::Statement::Switch { cases, .. } => {
                for case in cases {
                    visit_loop_depths(&case.body, depth, visit);
                }
            }
            _ => {}
        }
    }
}
