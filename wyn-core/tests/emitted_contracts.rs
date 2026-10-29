//! Cross-stage contracts on public emitted host/shader artifacts.
#[path = "support/pipeline.rs"]
mod pipeline;
use std::collections::BTreeSet;
use wyn_core::host::{
    self, Allocation, Binding, BufferLen, Expr, Operation, Pipeline, Program, ResultLayout, ResultScalar,
    ShaderFormat,
};
use wyn_core::{lower_ssa_to_spirv, lower_ssa_to_wgsl_with_program, CodegenTarget, PipelineTopologyPolicy};
use wyn_host_interp::Program as WhlProgram;

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

#[test]
fn generated_compute_captures_are_allocated_by_the_authored_host_entry() {
    let source = include_str!("../../testfiles/regressions/graphics_computed_stage_inputs.wyn");
    for (program, format) in host_outputs(source).into_iter().zip([ShaderFormat::Spirv, ShaderFormat::Wgsl])
    {
        let [entry] = program.entries.as_slice() else {
            panic!("one authored entry")
        };
        let resources = &program.interface.frame_graph.resources;
        let inputs: BTreeSet<_> = entry.inputs.iter().map(|id| resources[id.0].name.as_str()).collect();
        assert_eq!(inputs, BTreeSet::from(["xs", "target"]), "{format:?}");
        assert!(!entry.inputs.contains(&entry.results[0]));
        assert!(entry.allocations.iter().any(|a| a.resource() == entry.results[0]));
        WhlProgram::parse(&program.to_whl("staged", format).unwrap()).unwrap();
    }
}

#[test]
fn resolution_allocation_expressions_keep_cpu_parameters() {
    let source = include_str!("../../testfiles/rust_host_resolution_sizes.wyn");
    for (program, format) in host_outputs(source).into_iter().zip([ShaderFormat::Spirv, ShaderFormat::Wgsl])
    {
        let entry = &program.entries[0];
        assert_eq!(entry.results.len(), 4);
        assert_eq!(entry.host_inputs.len(), 1);
        let (&frame, &bytes) = entry.host_inputs.first_key_value().unwrap();
        assert_eq!(program.interface.frame_graph.resources[frame.0].name, "frame");
        assert_eq!(
            bytes, 24,
            "only resolution.x/y are needed at std140 offsets 16/20"
        );
        let whl = program.to_whl("resolution_sizes", format).unwrap();
        assert!(whl.contains(":host-bytes 24"));
        assert!(whl.contains("host-read-scalar"));
        assert!(!whl.contains("gpu-read-scalar"));
        for id in entry.results.iter().take(3) {
            let Allocation::Buffer { bytes, .. } =
                entry.allocations.iter().find(|a| a.resource() == *id).unwrap()
            else {
                panic!("computed output allocation");
            };
            let mut reads = Vec::new();
            bytes.clone().reads_mut(&mut |_, offset| reads.push(*offset));
            assert_eq!(reads, [16, 20]);
        }
        let rust = program.to_rust_wgpu("resolution_sizes", format).unwrap();
        assert!(rust.contains("frame: &ParameterBuffer"), "{rust}");
        assert!(!rust.contains("frame_host: &[u8]"), "{rust}");
        assert!(rust.contains("frame.upload(device, encoder)"), "{rust}");
        assert!(rust.contains("pub fn encode_sizes("), "{rust}");
        for readback in [
            "read_gpu_word",
            "map_async",
            "wait_indefinitely",
            "support::read_f32",
        ] {
            assert!(!rust.contains(readback), "unexpected {readback}: {rust}");
        }
    }
}

#[test]
fn filter_phase_layout_matches_the_selected_shader_storage_access() {
    let source = include_str!("../../testfiles/rust_host_storage_access_mismatch.wyn");
    for (program, format) in host_outputs(source).into_iter().zip([ShaderFormat::Spirv, ShaderFormat::Wgsl])
    {
        let read_only = format == ShaderFormat::Spirv;
        let rust = program.to_rust_wgpu("storage_access", format).unwrap();
        let [Pipeline::Compute(pipeline)] = program.interface.pipelines.as_slice() else {
            panic!("compute pipeline")
        };
        let (consumer, producers) = pipeline.stages.split_last().unwrap();
        let shared = consumer
            .reads
            .iter()
            .find(|binding| {
                !consumer.writes.contains(binding)
                    && producers.iter().any(|stage| stage.writes.contains(binding))
            })
            .unwrap();
        let Binding::StorageBuffer {
            set, binding: slot, ..
        } = pipeline.bindings[*shared]
        else {
            panic!("shared storage buffer")
        };
        let phase = rust
            .split("let compute_")
            .find(|phase| phase.contains(&format!("entry_point: Some({:?})", consumer.entry_point)))
            .unwrap();
        let layout = phase.split("let pipeline =").next().unwrap();
        let binding = layout
            .split("BindGroupLayoutEntry {")
            .find(|entry| entry.trim_start().starts_with(&format!("binding: {slot}u32,")))
            .unwrap();
        assert!(
            binding.contains(&format!("read_only: {read_only}")),
            "{format:?}: {binding}"
        );

        let whl = WhlProgram::parse(&program.to_whl("storage_access", format).unwrap()).unwrap();
        let phase = whl
            .kernels
            .values()
            .find(|kernel| kernel.options.text(":entry").unwrap() == consumer.entry_point)
            .unwrap();
        let abi = phase.options.get(":abi").unwrap().list().unwrap();
        let binding = abi
            .iter()
            .map(|value| value.list().unwrap())
            .find(|binding| {
                binding[1].text().unwrap() == ":storage"
                    && binding[2].u32().unwrap() == set
                    && binding[3].u32().unwrap() == slot
            })
            .unwrap();
        let parameter =
            phase.parameters.iter().find(|parameter| parameter.name == binding[0].text().unwrap()).unwrap();
        assert_eq!(
            parameter.access.as_deref(),
            Some(if read_only { ":read" } else { ":read-write" })
        );
    }
}

#[test]
fn graphics_compute_helpers_capture_computed_records_and_return_both_arrays() {
    let source = include_str!("../../testfiles/graphics_computed_record_capture.wyn");
    for (program, format) in host_outputs(source).into_iter().zip([ShaderFormat::Spirv, ShaderFormat::Wgsl])
    {
        let [entry] = program.entries.as_slice() else {
            panic!("one source entry")
        };
        let resources = &program.interface.frame_graph.resources;
        let inputs: BTreeSet<_> = entry.inputs.iter().map(|id| resources[id.0].name.as_str()).collect();
        assert_eq!(inputs, BTreeSet::from(["ui", "points", "target"]), "{format:?}");
        assert_eq!(entry.results.len(), 3);
        assert_ne!(entry.results[0], entry.results[1]);
        for result in &entry.results[..2] {
            assert!(
                !entry.inputs.contains(result),
                "computed result must not alias an input"
            );
            assert!(entry.allocations.iter().any(
                |allocation| matches!(allocation, Allocation::Buffer { resource, .. } if resource == result)
            ));
        }
        assert!(entry.operations.iter().any(|op| matches!(op, Operation::Dispatch { .. })));
        assert!(entry.operations.iter().any(|op| matches!(op, Operation::Draw { .. })));
        program.to_rust_wgpu("record_capture", format).unwrap();
    }
}

#[test]
fn uniform_sized_maps_launch_from_their_logical_domain() {
    let source = include_str!("../../testfiles/rust_host_runtime_dispatch.wyn");
    for (program, format) in host_outputs(source).into_iter().zip([ShaderFormat::Spirv, ShaderFormat::Wgsl])
    {
        let [entry] = program.entries.as_slice() else {
            panic!("one host entry");
        };
        let pixels = entry.results[0];
        assert!(
            entry.allocations.iter().any(|a| a.resource() == pixels),
            "host computes and allocates the output capacity"
        );
        let [Operation::Dispatch { groups, .. }, Operation::Draw { .. }] = entry.operations.as_slice()
        else {
            panic!("one map followed by its consuming draw");
        };
        let mut domain = groups[0].clone();
        let mut sources = Vec::new();
        domain.reads_mut(&mut |source, _| sources.push(source.clone()));
        assert!(!sources.is_empty(), "launch reads the source dimensions");
        assert!(
            !sources.contains(&host::ScalarSource::Resource(pixels)),
            "allocation capacity must not determine the logical launch domain"
        );
        assert_eq!(&groups[1..], &[Expr::Integer(1), Expr::Integer(1)]);
        let rust = program.to_rust_wgpu("runtime_dispatch", format).unwrap();
        let call = rust.split_once("pub fn host_reproduce(").unwrap().1;
        assert!(call.contains("_bytes"), "{call}");
        assert!(call.contains("ceiling("), "{call}");
        let whl = program.to_whl("runtime_dispatch", format).unwrap();
        assert!(whl.contains(&groups[0].to_whl().unwrap()), "{whl}");
    }
}

#[test]
fn rematerialized_helpers_and_dispatch_extents_publish_their_inputs() {
    for source in [
        include_str!("../../testfiles/scalar_setup.wyn"),
        include_str!("../../testfiles/rust_host_filter.wyn"),
    ] {
        let prepared = pipeline::source(source);
        let ssa = pipeline::lower(&prepared, CodegenTarget::Spirv);
        let program = lower_ssa_to_spirv(ssa).unwrap().program;
        let Pipeline::Compute(pipeline) = &program.interface.pipelines[0] else {
            panic!("compute pipeline");
        };
        let push =
            pipeline.bindings.iter().position(|b| matches!(b, Binding::PushConstant { .. })).unwrap();
        assert!(pipeline.stages[0].uses.reads.contains(&push), "{source}");
    }
}

#[test]
fn graphics_root_composes_multiple_computations_and_source_results() {
    let source = include_str!("../../testfiles/rust_host_frame_composition.wyn");
    for (program, format) in host_outputs(source).into_iter().zip([ShaderFormat::Spirv, ShaderFormat::Wgsl])
    {
        let [entry] = program.entries.as_slice() else {
            panic!(
                "one source root, got {:?}",
                program.entries.iter().map(|e| &e.name).collect::<Vec<_>>()
            );
        };
        assert_eq!(entry.name, "reproduce");
        assert_eq!(entry.inputs.len(), 2);
        assert_eq!(entry.results.len(), 3);
        assert_eq!(
            entry.operations.iter().filter(|op| matches!(op, Operation::Dispatch { .. })).count(),
            2
        );
        let draw = entry.operations.iter().position(|op| matches!(op, Operation::Draw { .. })).unwrap();
        for &id in &entry.results[..2] {
            assert!(!entry.inputs.contains(&id));
            assert!(entry
                .allocations
                .iter()
                .any(|a| matches!(a, Allocation::Buffer { resource, .. } if *resource == id)));
            assert!(entry.operations.iter().any(|op| {
                let Operation::Dispatch { pipeline, stage, .. } = op else {
                    return false;
                };
                let Pipeline::Compute(compute) = &program.interface.pipelines[*pipeline] else {
                    return false;
                };
                compute.stages[*stage]
                    .uses
                    .writes
                    .iter()
                    .any(|&b| program.binding_resource(*pipeline, b).unwrap() == id)
            }));
        }
        let Operation::Draw { pipeline } = entry.operations[draw] else {
            unreachable!()
        };
        assert!(program
            .parameter_indices(pipeline, None)
            .iter()
            .any(|&b| program.binding_resource(pipeline, b).unwrap() == entry.results[0]));
        assert!(entry.operations[..draw].iter().any(|op| {
            let Operation::Dispatch { pipeline, stage, .. } = op else {
                return false;
            };
            let Pipeline::Compute(compute) = &program.interface.pipelines[*pipeline] else {
                return false;
            };
            compute.stages[*stage]
                .uses
                .writes
                .iter()
                .any(|&b| program.binding_resource(*pipeline, b).unwrap() == entry.results[0])
        }));
        let results = &program.interface.source_results;
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.entry == "reproduce"));
        assert_eq!(
            results.iter().map(|r| (r.result, r.name.as_str())).collect::<Vec<_>>(),
            [(0, "result_0"), (1, "result_1")]
        );
        let rust = program.to_rust_wgpu("composition", format).unwrap();
        assert_eq!(rust.matches("pub fn host_").count(), 1);
        assert_eq!(rust.matches("pass.dispatch_workgroups(").count(), 2);
        assert_eq!(rust.matches("OutputResource::Buffer {").count(), 2);
        assert_eq!(rust.matches("OutputResource::Texture(Texture::clone(").count(), 1);
    }
}

#[test]
fn indirect_draw_waits_for_both_count_epilogue_and_compacted_vertices() {
    let source = include_str!("../../testfiles/rust_host_indirect_epilogue.wyn");
    for (program, _format) in
        host_outputs(source).into_iter().zip([ShaderFormat::Spirv, ShaderFormat::Wgsl])
    {
        let graph = &program.interface.frame_graph;
        assert_eq!(graph.passes.len(), 2, "one compaction dispatch and one draw");
        let [indirect] = graph.indirect_draws.as_slice() else {
            panic!("one indirect draw")
        };
        let writer = graph
            .passes
            .iter()
            .position(|pass| pass.writes.iter().any(|w| w.resource == indirect.buffer_resource))
            .unwrap();
        let compact = graph.passes.iter().position(|pass| pass.name.ends_with("compact")).unwrap();
        assert_eq!(
            writer, compact,
            "compaction also publishes the indirect draw command"
        );
        let draw = &graph.passes[indirect.draw_pass];
        assert!(draw.depends_on.contains(&writer));
        assert!(draw.depends_on.contains(&compact));
        assert!(graph.passes[compact]
            .writes
            .iter()
            .any(|w| draw.reads.iter().any(|r| r.resource == w.resource)));
    }
}

fn wgsl_output(source: &str) -> wyn_core::LoweredWgsl {
    let source = pipeline::source(source);
    let output = lower_ssa_to_wgsl_with_program(pipeline::lower(&source, CodegenTarget::Wgsl)).unwrap();
    validate(naga::front::wgsl::parse_str(&output.wgsl).unwrap());
    output
}

#[test]
fn tuple_projections_keep_result_capacity_on_both_backends() {
    for program in host_outputs(include_str!(
        "../../testfiles/regressions/fusion_shared_tuple.wyn"
    )) {
        let outputs: Vec<_> = program
            .interface
            .pipelines
            .iter()
            .flat_map(|p| match p {
                Pipeline::Compute(p) => &p.bindings,
                _ => panic!("compute pipeline"),
            })
            .filter_map(|b| match b {
                Binding::StorageBuffer {
                    usage: host::BufferUsage::Output,
                    length,
                    ..
                } => Some(length),
                _ => None,
            })
            .collect();
        assert_eq!(outputs.len(), 2);
        assert!(outputs.iter().all(|n| matches!(n, Some(BufferLen::Fixed { bytes: 64 }))));
    }
}

#[test]
fn graphics_stages_preserve_shader_interfaces_and_draw_metadata() {
    use host::{Pipeline, ShaderStage};
    let source = include_str!("../../testfiles/unified_triangle.wyn");
    let output = wgsl_output(source);
    let module = naga::front::wgsl::parse_str(&output.wgsl).unwrap();
    assert_eq!(module.entry_points.len(), 2);
    assert!(module.entry_points.iter().any(|e| e.stage == naga::ShaderStage::Vertex));
    assert!(module.entry_points.iter().any(|e| e.stage == naga::ShaderStage::Fragment));
    let [Pipeline::Graphics(graphics)] = output.program.interface.pipelines.as_slice() else {
        panic!("one graphics pipeline");
    };
    assert_eq!(graphics.stages.len(), 2);
    assert_eq!(graphics.source_operation, Some(0));
    assert!(graphics.stages.iter().any(|s| matches!(s.stage, ShaderStage::Vertex)));
    assert!(graphics.stages.iter().any(|s| matches!(s.stage, ShaderStage::Fragment)));
    assert!(!graphics.fragment_outputs.is_empty());
    assert!(output.program.interface.source_results.is_empty());
}

#[test]
fn shared_load_helper_publishes_both_results() {
    let output = wgsl_output(include_str!(
        "../../tests/compiler/library/shared_load_helper.wyn"
    ));
    assert_eq!(output.program.interface.source_results.len(), 2);
}

#[test]
fn grouped_entries_keep_storage_and_dispatch_contracts() {
    let output = wgsl_output(include_str!(
        "../../tests/compiler/library/host_stage_contracts.wyn"
    ));
    let program = &output.program;
    for entry in &program.entries {
        let pipeline = entry
            .operations
            .iter()
            .find_map(|op| match op {
                Operation::Dispatch { pipeline, .. } => Some(*pipeline),
                _ => None,
            })
            .expect("compute dispatch");
        let Pipeline::Compute(p) = &program.interface.pipelines[pipeline] else {
            panic!("compute")
        };
        let results: Vec<_> =
            program.interface.source_results.iter().filter(|r| r.entry == entry.name).collect();
        match entry.name.as_str() {
            "consuming_map" | "consuming_scan" => {
                let scan = entry.name == "consuming_scan";
                let (set, binding) = p
                    .bindings
                    .iter()
                    .find_map(|b| match b {
                        Binding::StorageBuffer {
                            set,
                            binding,
                            usage: host::BufferUsage::Input,
                            access: host::Access::ReadWrite,
                            ..
                        } => Some((*set, *binding)),
                        _ => None,
                    })
                    .expect("read-write input buffer");
                assert_eq!((results[0].set, results[0].binding), (set, binding));
                assert_eq!(p.bindings.len(), if scan { 5 } else { 2 });
                assert_eq!(p.stages.len(), if scan { 3 } else { 1 });
            }
            "observed_scatter" | "captured_scatter" => {
                assert_eq!(p.bindings.len(), 4);
                assert_eq!(results.len(), 2);
                assert_ne!(results[0].binding, results[1].binding);
            }
            "buffer_domain" | "scalar_domain" => assert!(matches!(
                p.stages[0].dispatch_size,
                host::DispatchSize::Computed {
                    elements: host::Expr::BufferLength { .. } | host::Expr::Scalar(_),
                    ..
                }
            )),
            "partial_whole" => assert_eq!(p.stages.len(), 1),
            "partial_slice" => assert_eq!(p.stages.len(), 2),
            "scalar_loop" => {
                assert_eq!(p.stages.len(), 2);
                assert_eq!(p.stages[0].workgroup_size, (1, 1, 1));
                assert_eq!(p.stages[1].workgroup_size, (64, 1, 1));
            }
            "large_fixed" => {
                assert_eq!(p.stages.len(), 1);
                assert_eq!(
                    p.stages[0].dispatch_size,
                    host::DispatchSize::Fixed {
                        x: 65_535,
                        y: 1,
                        z: 1,
                        explicit: true
                    }
                );
            }
            name => panic!("unexpected entry {name}"),
        }
    }
    assert_eq!(program.entries.len(), 10);
}

#[test]
fn push_constant_record_padding_precedes_the_next_parameter() {
    for source in [
        include_str!("../../tests/compiler/library/push_tail_vec3f32.wyn"),
        include_str!("../../tests/compiler/library/push_tail_vec4f32.wyn"),
    ] {
        let source = pipeline::source(source);
        let output = lower_ssa_to_spirv(pipeline::lower(&source, CodegenTarget::Spirv)).unwrap();
        let module =
            naga::front::spv::Frontend::new(output.spirv.into_iter(), &Default::default()).parse().unwrap();
        let (_, block) = module
            .global_variables
            .iter()
            .find(|(_, v)| v.space == naga::AddressSpace::PushConstant)
            .unwrap();
        let naga::TypeInner::Struct { members, .. } = &module.types[block.ty].inner else {
            panic!("push constant struct")
        };
        assert_eq!(members[1].offset, 16);
        validate(module);
    }
}

#[path = "support/source_contracts.rs"]
mod source_contracts;
