//! Source-to-artifact integration contracts that need structured assertions.
use super::pipeline;
use wyn_core::{CodegenTarget, PipelineTopologyPolicy};

fn compile_thru_ssa(source: &str) -> wyn_core::error::Result<wyn_core::ssa::stage::Elaborated> {
    let source = pipeline::source(source);
    Ok(pipeline::lower_with_options(
        &source,
        CodegenTarget::Portable,
        wyn_core::egglog::ScalarOptimization::Full,
        PipelineTopologyPolicy::AllowGenerated,
    ))
}
fn compile_to_spirv_output(source: &str) -> wyn_core::error::Result<wyn_core::Lowered> {
    wyn_core::lower_ssa_to_spirv(compile_thru_ssa(source)?)
}

mod source_pipeline {
    use super::*;
    use naga::{Expression, Statement};
    use wyn_core::{lower_ssa_to_spirv, lower_ssa_to_wgsl};
    fn placement_shaders(source: &str, inspect: impl FnMut(&naga::Module)) {
        super::super::placement_shaders(source, inspect);
    }
    fn shaders(source: &str) -> naga::Module {
        let program = compile_thru_ssa(source).unwrap_or_else(|error| panic!("{error}\n{source}"));
        let wgsl = lower_ssa_to_wgsl(program.clone()).unwrap_or_else(|error| panic!("{error}\n{source}"));
        let module = naga::front::wgsl::parse_str(&wgsl)
            .unwrap_or_else(|error| panic!("{}\n{wgsl}", error.emit_to_string(&wgsl)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|error| panic!("{error:?}\n{wgsl}"));
        let binary = lower_ssa_to_spirv(program).unwrap_or_else(|error| panic!("{error}\n{source}"));
        wspirv::dr::load_words(&binary.spirv).expect("valid SPIR-V encoding");
        module
    }
    fn storage_loads(function: &naga::Function, global: naga::Handle<naga::GlobalVariable>) -> usize {
        function
            .expressions
            .iter()
            .filter(|(_, expression)| {
                let Expression::Load { pointer } = expression else {
                    return false;
                };
                let mut pointer = *pointer;
                loop {
                    match function.expressions[pointer] {
                        Expression::Access { base, .. } | Expression::AccessIndex { base, .. } => {
                            pointer = base
                        }
                        Expression::GlobalVariable(value) => return value == global,
                        _ => return false,
                    }
                }
            })
            .count()
    }
    fn guarded_loops(block: &naga::Block, depth: usize) -> Vec<usize> {
        block
            .iter()
            .flat_map(|statement| match statement {
                Statement::If { accept, reject, .. } => {
                    let mut loops = guarded_loops(accept, depth + 1);
                    loops.extend(guarded_loops(reject, depth + 1));
                    loops
                }
                Statement::Loop { body, .. } => {
                    let mut loops = vec![depth];
                    loops.extend(guarded_loops(body, depth));
                    loops
                }
                Statement::Block(block) => guarded_loops(block, depth),
                _ => Vec::new(),
            })
            .collect()
    }
    fn visit_loop_depths(block: &naga::Block, depth: usize, visit: &mut impl FnMut(&Statement, usize)) {
        for statement in block.iter() {
            visit(statement, depth);
            match statement {
                Statement::Loop { body, continuing, .. } => {
                    visit_loop_depths(body, depth + 1, visit);
                    visit_loop_depths(continuing, depth + 1, visit);
                }
                Statement::If { accept, reject, .. } => {
                    visit_loop_depths(accept, depth, visit);
                    visit_loop_depths(reject, depth, visit);
                }
                Statement::Block(block) => visit_loop_depths(block, depth, visit),
                Statement::Switch { cases, .. } => {
                    for case in cases {
                        visit_loop_depths(&case.body, depth, visit);
                    }
                }
                _ => {}
            }
        }
    }
    #[test]
    fn fused_collectives_share_producer_elements() {
        let module = shaders(include_str!(
            "../../../tests/compiler/library/fused_collectives_share_producer_elements.wyn"
        ));
        let input = module
            .global_variables
            .iter()
            .find(|(_, v)| v.binding.as_ref().is_some_and(|b| b.group == 0 && b.binding == 0))
            .unwrap()
            .0;
        let partials = module.entry_points.iter().find(|e| e.name.ends_with("_partials")).unwrap();
        assert_eq!(
            storage_loads(&partials.function, input),
            1,
            "one input load feeds both reductions"
        );
    }
    #[test]
    fn radix_scratch_initializer_has_no_array_allocation() {
        let source = format!(
            "{}\n entry main(xs: []i32) []i32 = radix_sort_step(xs, |i:i32,x:i32| (x >> i) & 1, 0)",
            include_str!("../../../pkg/sort/src/radix_sort.wyn")
        );
        let compiled =
            wyn_core::lower_ssa_to_wgsl_with_program(compile_thru_ssa(&source).unwrap()).unwrap();
        let entry = &compiled.program.entries[0];
        let arrays = entry.allocations.iter().filter(|allocation| matches!(allocation,
        wyn_core::host::Allocation::Buffer { bytes, .. } if !matches!(bytes, wyn_core::host::Expr::Integer(_))
    )).count();
        assert_eq!(
            arrays, 2,
            "only scan prefixes and sorted output scale with input length\n{}\n{:?}",
            compiled.wgsl, entry.allocations
        );
    }
    #[test]
    fn radix_host_loop_dispatches_three_phases_per_pass() {
        let source = format!(
            "{}\n entry main(xs: []i32) []i32 = radix_sort(32, i32.get_bit, xs)",
            include_str!("../../../pkg/sort/src/radix_sort.wyn")
        );
        let compiled =
            wyn_core::lower_ssa_to_wgsl_with_program(compile_thru_ssa(&source).unwrap()).unwrap();
        let [region] = compiled.program.interface.dispatch_loops.as_slice() else {
            panic!("expected one host loop");
        };
        assert_eq!(region.body.len(), 3);
    }
    #[test]
    fn record_update_resource_fields_resolve_before_shader_emission() {
        placement_shaders(
        include_str!("../../../tests/compiler/library/record_update_resource_fields_resolve_before_shader_emission.wyn"),
        |_| {},
    );
    }
    #[test]
    fn selected_record_expansion_is_used_as_a_helper_argument() {
        // The redundant differences survive early TLC simplification and keep
        // repack above the early inlining threshold; egglog reduces them to zero.
        let module = shaders(include_str!(
            "../../../tests/compiler/library/selected_record_expansion_is_used_as_a_helper_argument.wyn"
        ));
        assert!(
            module.functions.iter().all(|(_, function)| {
                !function.name.as_deref().is_some_and(|name| name.contains("repack"))
            }),
            "the expanded record must not leave a callable repacking helper"
        );
    }
    #[test]
    fn shared_branch_producers_keep_one_guarded_loop() {
        let source = include_str!("../../../testfiles/regressions/shared_branch_producer.wyn");
        for (consumer, loops) in [
            ("if y > 1.0 then color else color * 0.8", 1),
            (
                "if y > 1.0 then (if y > 2.0 then color else @[0.0,0.0,0.0,0.0])
             else (if y < 0.0 then color * 0.8 else @[0.0,0.0,0.0,0.0])",
                1,
            ),
        ] {
            let source = source.replace("if y > 1.0 then color else color * 0.8", consumer);
            let module = shaders(&source);
            let fragment = &module
                .entry_points
                .iter()
                .find(|e| e.stage == naga::ShaderStage::Fragment)
                .unwrap()
                .function;
            let depths = guarded_loops(&fragment.body, 0);
            assert_eq!(depths.len(), loops, "{source}");
            assert!(
                depths.iter().all(|&depth| depth > 0),
                "producer escaped its x guard: {source}"
            );
        }
    }
    #[test]
    fn enclosing_producer_is_shared_across_nested_consumer_guards() {
        let source = include_str!("../../../testfiles/regressions/shared_branch_producer.wyn").replace(
            "if y > 1.0 then color else color * 0.8",
            "if y > 1.0 then color else if y < 0.0 then @[0.0,0.0,0.0,0.0] else color * 0.8",
        );
        let module = shaders(&source);
        let fragment =
            &module.entry_points.iter().find(|e| e.stage == naga::ShaderStage::Fragment).unwrap().function;
        let depths = guarded_loops(&fragment.body, 0);
        assert_eq!(depths.len(), 1);
        assert!(depths[0] > 0, "producer must retain its own guard");
    }
    #[test]
    fn resource_records_passed_to_helpers_reach_both_backends() {
        placement_shaders(
        include_str!("../../../tests/compiler/library/enclosing_producer_is_shared_across_nested_consumer_guards.wyn"),
        |_| {},
    );
    }
    #[test]
    fn scans_and_reductions_use_one_shared_bank_per_phase() {
        let reduce = shaders(include_str!(
            "../../../tests/compiler/library/resource_records_passed_to_helpers_reach_both_backends.wyn"
        ));
        let scan = shaders(include_str!(
            "../../../tests/compiler/library/scans_and_reductions_use_one_shared_bank_per_phase.wyn"
        ));
        let shared = |module: &naga::Module| {
            module.global_variables.iter().filter(|(_, v)| v.space == naga::AddressSpace::WorkGroup).count()
        };
        assert!(shared(&reduce) > 0);
        assert_eq!(
            shared(&reduce),
            shared(&scan),
            "scans reuse their bank after a read barrier"
        );
    }
    #[test]
    fn local_map_captures_are_computed_before_the_element_loop() {
        // Keep these traversals above the small-loop expansion limit.
        let module = shaders(include_str!(
            "../../../tests/compiler/library/local_map_captures_are_computed_before_the_element_loop.wyn"
        ));
        let mut loops = 0;
        for (_, function) in module.functions.iter() {
            visit_loop_depths(&function.body, 0, &mut |statement, depth| {
                if matches!(statement, Statement::Loop { .. }) {
                    loops += 1;
                    assert_eq!(depth, 0, "captured scan must not repeat for every output element");
                }
            });
        }
        for entry in &module.entry_points {
            visit_loop_depths(&entry.function.body, 0, &mut |statement, depth| {
                if matches!(statement, Statement::Loop { .. }) && depth > 0 {
                    loops += 1;
                    assert_eq!(
                        depth, 1,
                        "inlined captured scan must not repeat for every output element"
                    );
                }
            });
        }
        assert!(loops >= 2, "fixture must retain the scan and output traversal");
    }
    #[test]
    fn batched_traversal_has_shared_state_and_no_partial_lane_barriers() {
        fn inspect(block: &naga::Block, conditional: bool) -> usize {
            let mut barriers = 0;
            for statement in block.iter() {
                match statement {
                    Statement::ControlBarrier(_) => {
                        assert!(!conditional, "traversal barrier inside a partial-lane branch");
                        barriers += 1;
                    }
                    Statement::If { accept, reject, .. } => {
                        barriers += inspect(accept, true) + inspect(reject, true);
                    }
                    Statement::Loop { body, continuing, .. } => {
                        barriers += inspect(body, conditional) + inspect(continuing, conditional);
                    }
                    Statement::Block(block) => barriers += inspect(block, conditional),
                    _ => {}
                }
            }
            barriers
        }
        let module = shaders(include_str!(
            "../../../fluid-simulation/test/batched_traversal.wyn"
        ));
        assert!(module.global_variables.iter().any(|(_, v)| v.space == naga::AddressSpace::WorkGroup));
        let mut barriers = 0;
        for function in
            module.functions.iter().map(|(_, f)| f).chain(module.entry_points.iter().map(|e| &e.function))
        {
            barriers += inspect(&function.body, false);
        }
        assert!(barriers > 2, "traversal must execute cooperative collectives");
        assert_eq!(
            module.entry_points.len(),
            1,
            "traversal should stay in one kernel"
        );
        assert!(
            module.entry_points[0].workgroup_size[0] > 1,
            "queries need a parallel workgroup"
        );
    }
    #[test]
    #[ignore = "nested scheduling follow-up: ranked buckets inside map fail with an unbound TLC symbol during source import"]
    fn ranked_bucket_scatter_inside_map_reaches_both_backends() {
        shaders(include_str!("../../../tests/compiler/library/batched_traversal_has_shared_state_and_no_partial_lane_barriers.wyn"));
    }
    #[test]
    fn independent_scalar_results_do_not_share_a_host_resource() {
        let source = include_str!("../../../testfiles/scalar_epilogues.wyn");
        let output = wyn_core::lower_ssa_to_wgsl_with_program(compile_thru_ssa(source).unwrap()).unwrap();
        let mut resources = std::collections::BTreeMap::new();
        for pipeline in &output.program.interface.pipelines {
            let wyn_core::host::Pipeline::Compute(pipeline) = pipeline else {
                continue;
            };
            for binding in &pipeline.bindings {
                if let wyn_core::host::Binding::StorageBuffer {
                    set,
                    binding,
                    resource: Some(name),
                    ..
                } = binding
                {
                    assert!(!name.is_empty());
                    if let Some(previous) = resources.insert(name, (*set, *binding)) {
                        assert_eq!(
                            previous,
                            (*set, *binding),
                            "distinct buffers must not acquire one host identity"
                        );
                    }
                }
            }
        }
    }
    #[test]
    fn runtime_array_metadata_shares_storage_and_fits_binding_limits() {
        for (source, limit, inputs) in [
            (
                include_str!("../../../testfiles/regressions/runtime_array_three_outputs.wyn"),
                8,
                1,
            ),
            (
                include_str!("../../../testfiles/regressions/runtime_array_four_inputs.wyn"),
                6,
                4,
            ),
        ] {
            let output =
                wyn_core::lower_ssa_to_wgsl_with_program(compile_thru_ssa(source).unwrap()).unwrap();
            let [wyn_core::host::Pipeline::Compute(pipeline)] =
                output.program.interface.pipelines.as_slice()
            else {
                panic!("compute pipeline");
            };
            for stage in &pipeline.stages {
                let storage_bindings = pipeline
                    .bindings
                    .iter()
                    .enumerate()
                    .filter(|(index, binding)| {
                        matches!(binding, wyn_core::host::Binding::StorageBuffer { .. })
                            && stage.uses.access(*index).is_some()
                    })
                    .count();
                assert!(
                    storage_bindings <= limit,
                    "binding limit exceeded: {storage_bindings} > {limit}"
                );
            }
            let lengths: Vec<_> = output
                .program
                .interface
                .scalar_tasks
                .iter()
                .filter(|task| matches!(task.value, wyn_core::host::ScalarExpr::BufferLength { .. }))
                .collect();
            assert!(
                lengths.len() >= inputs,
                "each dynamic input needs its logical length"
            );
            for task in &lengths {
                assert_eq!(task.destination, lengths[0].destination);
                assert_eq!(task.offset % 4, 0);
            }
        }
    }
    #[test]
    fn camera_field_math_is_shared_across_indexing() {
        for (ty, angle) in [("{angle:f32}", "camera.angle"), ("vec2f32", "camera.x")] {
            // Keep the calculation per-element so host capture cannot remove it
            // from the shader whose sharing this regression checks.
            let source = format!(
                "entry main(xs:[]i32,camera:{ty}) []f32 = map(|x|
             let q=f32.sin({angle}+f32(x))
             let values=[x,x+1] in f32(values[i32(f32.abs(q))%2])+q,xs)"
            );
            let module = shaders(&source);
            let sines = module
                .functions
                .iter()
                .map(|(_, f)| f)
                .chain(module.entry_points.iter().map(|e| &e.function))
                .flat_map(|f| f.expressions.iter())
                .filter(|(_, expression)| {
                    matches!(
                        expression,
                        Expression::Math {
                            fun: naga::MathFunction::Sin,
                            ..
                        }
                    )
                })
                .count();
            assert_eq!(sines, 1, "{source}");
        }
    }
}

mod source_host {
    use super::*;
    use wyn_core::host::Pipeline;
    use wyn_core::{host, LoweredWgsl};
    fn pipeline(source: &str) -> LoweredWgsl {
        wyn_core::lower_ssa_to_wgsl_with_program(compile_thru_ssa(source).unwrap()).unwrap()
    }
    fn compile(source: &str) -> naga::Module {
        let source = pipeline::source(source);
        let ssa = pipeline::lower_with_options(
            &source,
            CodegenTarget::Wgsl,
            wyn_core::egglog::ScalarOptimization::Full,
            PipelineTopologyPolicy::AllowGenerated,
        );
        let text = wyn_core::lower_ssa_to_wgsl(ssa).unwrap();
        let module = naga::front::wgsl::parse_str(&text).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        module
    }
    #[test]
    fn loop_local_scratch_annotation_is_consumed_before_codegen() {
        let source = include_str!("../../../tests/compiler/library/loop_local_scratch_annotation_is_consumed_before_codegen_additional.wyn");
        let program = compile_thru_ssa(source).unwrap();
        wyn_core::lower_ssa_to_spirv(program.clone()).unwrap();
        wyn_core::lower_ssa_to_wgsl(program).unwrap();
    }
    #[test]
    fn loop_local_zip_preserves_nested_input_elements() {
        let source = include_str!(
            "../../../tests/compiler/library/loop_local_scratch_annotation_is_consumed_before_codegen.wyn"
        );
        let program = compile_thru_ssa(source).unwrap();
        wyn_core::lower_ssa_to_spirv(program.clone()).unwrap();
        wyn_core::lower_ssa_to_wgsl(program).unwrap();
    }
    #[test]
    fn scalar_output_epilogues_lower_without_a_finish_entry() {
        let module = compile(include_str!("../../../testfiles/scalar_epilogues.wyn"));
        assert_eq!(module.entry_points.len(), 3);
        assert!(module.entry_points.iter().all(|entry| !entry.name.ends_with("finish")));
    }
    #[test]
    fn filter_count_draw_record_is_published_by_compaction() {
        let module = compile(include_str!(
            "../../../tests/compiler/library/scalar_output_epilogues_lower_without_a_finish_entry.wyn"
        ));
        assert_eq!(module.entry_points.len(), 1);
    }
    #[test]
    fn filter_post_map_record_outputs_reach_wgsl() {
        let module = compile(include_str!("../../../testfiles/rust_host_filter_post.wyn"));
        let entries: Vec<_> = module.entry_points.iter().map(|entry| entry.name.as_str()).collect();
        assert!(entries.iter().any(|name| name.ends_with("compact")));
        assert_eq!(module.entry_points[0].workgroup_size, [64, 1, 1]);
    }
    #[test]
    fn filter_element_read_epilogue_lowers_in_a_separate_dispatch() {
        let module = compile(include_str!(
            "../../../tests/compiler/library/filter_post_map_record_outputs_reach_wgsl.wyn"
        ));
        assert!(module.entry_points.len() >= 2);
    }
    #[test]
    fn runtime_input_lengths_and_output_allocations_share_the_published_bindings() {
        use host::{Binding, BufferLen, Pipeline};
        let output = pipeline(include_str!("../../../tests/compiler/library/filter_element_read_epilogue_lowers_in_a_separate_dispatch.wyn"));
        let module = naga::front::wgsl::parse_str(&output.wgsl).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        assert!(!module.entry_points.iter().any(|e| e
            .function
            .expressions
            .iter()
            .any(|(_, x)| matches!(x, naga::Expression::ArrayLength(_)))));
        assert!(
            output
                .program
                .interface
                .scalar_tasks
                .iter()
                .any(|task| matches!(task.value, host::ScalarExpr::BufferLength { stride: 4, .. })),
            "logical input length is supplied by the host"
        );
        let [Pipeline::Compute(p)] = output.program.interface.pipelines.as_slice() else {
            panic!("compute pipeline")
        };
        assert_eq!(p.stages.len(), 1);
        assert_eq!(p.stages[0].owner, "main");
        let result = &output.program.interface.source_results[0];
        assert!(p.bindings.iter().any(|b| matches!(b, Binding::StorageBuffer { set, binding, length: Some(BufferLen::LikeInput { set: 0, binding: 0, elem_bytes: 4, src_elem_bytes: 4 }), .. } if (*set, *binding) == (result.set, result.binding))));
        assert!(p.bindings.iter().any(
        |b| matches!(b, Binding::StorageBuffer { members, .. } if members.iter().any(|m| m.name == "bias"))
    ));
        assert!(p.bindings.iter().all(|b| !matches!(b, Binding::PushConstant { .. })));
    }
    #[test]
    fn reduction_publication_has_scratch_writers_readers_and_a_source_result() {
        use host::{Binding, BufferLen, Pipeline};
        let output = pipeline(include_str!("../../../tests/compiler/library/runtime_input_lengths_and_output_allocations_share_the_published_bindings.wyn"));
        let [Pipeline::Compute(p)] = output.program.interface.pipelines.as_slice() else {
            panic!("compute pipeline")
        };
        assert_eq!(p.stages.len(), 3, "chunks, combine, scalar result publication");
        let host::DispatchSize::Fixed { x, y, z, .. } = p.stages[0].dispatch_size else {
            panic!("collective chunk grid must be bounded");
        };
        let bytes = u64::from(x) * u64::from(y) * u64::from(z) * 4;
        let scratch = p
            .bindings
            .iter()
            .position(|b| {
                matches!(
                    b,
                    Binding::StorageBuffer {
                        length: Some(BufferLen::Fixed { bytes: capacity }),
                        ..
                    } if *capacity == bytes
                )
            })
            .unwrap();
        assert!(p.stages[0].writes.contains(&scratch));
        assert!(p.stages[1].reads.contains(&scratch));
        assert!(!p.stages[1].writes.contains(&scratch));
        assert_eq!(output.program.interface.source_results.len(), 1);
        assert_eq!(output.program.interface.frame_graph.passes.len(), 3);
        assert!(output.program.interface.frame_graph.topological_order().is_ok());
    }
    #[test]
    fn scalar_results_after_collectives_are_executed_and_published() {
        let output = pipeline(include_str!("../../../tests/compiler/library/reduction_publication_has_scratch_writers_readers_and_a_source_result.wyn"));
        assert_eq!(output.program.interface.source_results.len(), 1);
        let module = naga::front::wgsl::parse_str(&output.wgsl).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        assert_eq!(module.entry_points.len(), 3);
    }
    #[test]
    fn computed_graphics_captures_share_the_producer_binding() {
        use host::Binding;
        let output = pipeline(include_str!("../../../testfiles/playground/conway.wyn"));
        let [result] = output.program.interface.source_results.as_slice() else {
            panic!("one computed board");
        };
        assert!(output.program.interface.pipelines.iter().any(|p| {
            let Pipeline::Graphics(p) = p else { return false };
            p.bindings.iter().any(|b| {
                matches!(b, Binding::StorageBuffer { set, binding, .. }
            if (*set, *binding) == (result.set, result.binding))
            })
        }));
    }
    #[test]
    fn host_sized_outputs_publish_uniform_dependencies_and_storage_stride() {
        use host::{Binding, BufferLen, Expr, ScalarSource};
        let source = include_str!("../../../testfiles/regressions/uniform_output_size.wyn")
            .replace("{ resolution: vec3f32 }", "{ padding: f32, resolution: vec3f32 }")
            .replace("([]f32,", "([]vec3f32,")
            .replace(
                "target_load(rendered, @[i % width, i / width], 0u32)",
                "@[f32(i), 0.0, 0.0]",
            );
        let output = pipeline(&source);
        let result = &output.program.interface.source_results[0];
        let Pipeline::Compute(pipeline) = &output.program.interface.pipelines[result.pipeline_index] else {
            panic!("compute")
        };
        let bytes = pipeline
            .bindings
            .iter()
            .find_map(|binding| match binding {
                Binding::StorageBuffer {
                    set,
                    binding,
                    length: Some(BufferLen::Computed { bytes }),
                    ..
                } if (*set, *binding) == (result.set, result.binding) => Some(bytes),
                _ => None,
            })
            .expect("generated output allocation");
        let Expr::Multiply(_, stride) = bytes else {
            panic!("physical stride")
        };
        assert_eq!(**stride, Expr::Integer(16));
        let mut reads = Vec::new();
        bytes.clone().reads_mut(&mut |source, offset| {
            assert!(matches!(source, ScalarSource::Binding { .. }));
            reads.push(*offset);
        });
        // Project the two needed components before emitting parameter reads. Keep
        // std140 offsets and do not touch padding or the unused resolution.z.
        assert_eq!(reads, [16, 20]);
        let whl = output.program.to_whl("uniform.wgsl", host::ShaderFormat::Wgsl).unwrap();
        assert!(whl.contains("wyn-f32-to-i32"));
    }
}

mod source_host_output {
    use super::*;
    use std::collections::BTreeSet;
    use wyn_core::host::{Allocation, Binding, FrameGraph, Operation, Pipeline, Program, ShaderFormat};
    use wyn_core::lower_ssa_to_wgsl_with_program;
    fn compile(source: &str) -> Program {
        lower_ssa_to_wgsl_with_program(compile_thru_ssa(source).unwrap()).unwrap().program
    }
    fn assert_dependency(graph: &FrameGraph, before: usize, after: usize) {
        let mut pending = graph.passes[after].depends_on.clone();
        let mut visited = BTreeSet::new();
        while let Some(index) = pending.pop() {
            if index == before {
                return;
            }
            if visited.insert(index) {
                pending.extend(&graph.passes[index].depends_on);
            }
        }
        panic!(
            "{} must depend on {}",
            graph.passes[after].name, graph.passes[before].name
        );
    }
    fn check_whl(source: &str) {
        let mut depth = 0i32;
        let mut string = false;
        let mut escaped = false;
        let mut comment = false;
        for c in source.chars() {
            if comment {
                if c == '\n' {
                    comment = false;
                }
                continue;
            }
            if string {
                if escaped {
                    escaped = false;
                } else if c == '\\' {
                    escaped = true;
                } else if c == '"' {
                    string = false;
                }
                continue;
            }
            match c {
                ';' => comment = true,
                '"' => string = true,
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    assert!(depth >= 0, "unexpected closing parenthesis in {source}");
                }
                _ => {}
            }
        }
        assert!(!string && depth == 0, "unbalanced WHL: {source}");
    }
    #[test]
    fn runtime_array_joins_wait_for_producers_in_branch_arms() {
        for source in [
            include_str!("../../../tests/compiler/library/runtime_array_joins_wait_for_producers_in_branch_arms_additional.wyn"),
            include_str!("../../../tests/compiler/library/runtime_array_joins_wait_for_producers_in_branch_arms_additional.wyn"),
        ] {
            let program = compile(source);
            let graph = &program.interface.frame_graph;
            let join = graph.passes.iter().position(|pass| pass.name == "main_view").unwrap();
            let producers: Vec<_> = graph
                .passes
                .iter()
                .enumerate()
                .filter(|(_, pass)| pass.name.starts_with("main_compute"))
                .map(|(i, _)| i)
                .collect();
            assert!(!producers.is_empty());
            for producer in producers {
                assert_dependency(graph, producer, join);
            }
        }
    }
    #[test]
    fn device_computed_trip_counts_are_published_before_the_loop() {
        let program = compile(include_str!(
            "../../../tests/compiler/library/runtime_array_joins_wait_for_producers_in_branch_arms.wyn"
        ));
        let whl = program.to_whl("device_count", ShaderFormat::Wgsl).unwrap();
        assert!(whl.contains("gpu-read-scalar"), "{whl}");
        assert_eq!(
            program.entries[0].operations.iter().filter(|op| matches!(op, Operation::Loop { .. })).count(),
            1
        );
    }
    #[test]
    fn finish_dispatch_waits_for_all_reduction_results() {
        let program = compile(include_str!(
            "../../../tests/compiler/library/device_computed_trip_counts_are_published_before_the_loop.wyn"
        ));
        let dispatches: Vec<_> = program.entries[0]
            .operations
            .iter()
            .filter_map(|operation| {
                let Operation::Dispatch { pipeline, stage, .. } = operation else {
                    return None;
                };
                let Pipeline::Compute(compute) = &program.interface.pipelines[*pipeline] else {
                    panic!("compute pipeline");
                };
                Some(compute.stages[*stage].entry_point.as_str())
            })
            .collect();
        let combine = dispatches.iter().position(|name| name.ends_with("_combine")).unwrap();
        let finish = dispatches.iter().position(|name| name.ends_with("_finish")).unwrap();
        assert!(combine < finish, "{dispatches:?}");
    }
    #[test]
    fn host_program_preserves_compute_phases_and_returns_source_results() {
        let program = compile(include_str!(
            "../../../tests/compiler/library/finish_dispatch_waits_for_all_reduction_results.wyn"
        ));
        let [entry] = program.entries.as_slice() else {
            panic!("one host entry");
        };
        let stages = program
            .interface
            .pipelines
            .iter()
            .map(|p| match p {
                Pipeline::Compute(c) => c.stages.len(),
                Pipeline::Graphics(_) => 0,
            })
            .sum::<usize>();
        assert_eq!(
            entry.operations.iter().filter(|op| matches!(op, Operation::Dispatch { .. })).count(),
            stages
        );
        assert_eq!(entry.results.len(), 1);
        assert!(entry
            .operations
            .iter()
            .all(|o| matches!(o, Operation::Dispatch { .. } | Operation::Scalar { .. })));
        let whl = program.to_whl("scan.wgsl", ShaderFormat::Wgsl).unwrap();
        check_whl(&whl);
        assert_eq!(whl.matches("(gpu-dispatch ").count(), stages);
        assert!(whl.contains("(gpu-alloc "));
        assert!(whl.contains("(gpu-free "));
        assert!(whl.contains(":source-name \"main\""));
    }
    #[test]
    fn rust_recording_api_batches_scan_passes_and_scratch_clears() {
        let program = compile(include_str!("../../../tests/compiler/library/host_program_preserves_compute_phases_and_returns_source_results.wyn"));
        let rust = program.to_rust_wgpu("scan.wgsl", ShaderFormat::Wgsl).unwrap();
        let (wrapper, recording) = rust.split_once("pub fn encode_main(").unwrap();
        assert_eq!(wrapper.matches("queue.submit(").count(), 1);
        assert_eq!(wrapper.matches("create_command_encoder(").count(), 1);
        assert!(!recording.contains("queue.submit("));
        assert!(!recording.contains("create_command_encoder("));
        assert!(recording.contains("encoder.clear_buffer("));
        assert_eq!(recording.matches("pass.dispatch_workgroups(").count(), 3);
    }
    #[test]
    fn integer_capacity_expressions_reach_both_hosts() {
        let program = compile(include_str!(
            "../../../tests/compiler/library/rust_recording_api_batches_scan_passes_and_scratch_clears.wyn"
        ));
        let [entry] = program.entries.as_slice() else {
            panic!("one host entry");
        };
        assert!(entry.allocations.iter().any(
        |a| matches!(a,Allocation::Buffer{bytes,..} if bytes.to_whl().unwrap().contains("host-read-scalar"))
    ));
        let whl = program.to_whl("sizes.wgsl", ShaderFormat::Wgsl).unwrap();
        check_whl(&whl);
        assert!(whl.contains("(wyn-i32-add "), "{whl}");
        assert!(whl.contains("(wyn-i32-mul "), "{whl}");
        assert!(!whl.contains("4294967296"));
        let rust = program.to_rust_wgpu("sizes.wgsl", ShaderFormat::Wgsl).unwrap();
        assert!(rust.contains(".wrapping_add("));
        assert!(rust.contains(".wrapping_mul("));
        assert!(rust.contains("n: &ParameterBuffer"));
        let compact: String = rust.split_whitespace().collect();
        assert!(!rust.contains("n_host: &[u8]"));
        assert!(compact.contains("n.upload(device,encoder)"));
        assert!(compact.contains("support::scalar_bytes(resource_"));
        assert!(!rust.contains("read_host_scalar"));
        assert!(!rust.contains("read_gpu_word"));
        assert!(rust.contains("pub fn encode_main("));
    }
    #[test]
    fn published_shader_names_identify_source_phases_and_stay_consistent() {
        let source = include_str!(
            "../../../tests/compiler/library/integer_capacity_expressions_reach_both_hosts.wyn"
        );
        let ssa = compile_thru_ssa(source).unwrap();
        let names: BTreeSet<_> = ssa.entry_points.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(names.len(), ssa.entry_points.len());
        assert!(names.iter().all(|n| n.starts_with("totals_")), "{names:?}");
        assert!(names.contains("totals_partials"));
        assert!(names.contains("totals_combine"));
        let compiled = lower_ssa_to_wgsl_with_program(ssa).unwrap();
        let module = naga::front::wgsl::parse_str(&compiled.wgsl).unwrap();
        let shader_names: BTreeSet<_> = module.entry_points.iter().map(|e| e.name.as_str()).collect();
        for pipeline in &compiled.program.interface.pipelines {
            let Pipeline::Compute(pipeline) = pipeline else {
                panic!("compute pipeline")
            };
            for stage in &pipeline.stages {
                assert!(shader_names.contains(stage.entry_point.as_str()));
            }
        }
        let resource_names: Vec<_> =
            compiled.program.interface.frame_graph.resources.iter().map(|r| r.name.as_str()).collect();
        assert!(resource_names.contains(&"totals_result_0"), "{resource_names:?}");
        assert!(resource_names.contains(&"totals_result_1"));
        assert!(resource_names.iter().any(|n| n.starts_with("totals_scratch")));
        let whl = compiled.program.to_whl("totals.wgsl", ShaderFormat::Wgsl).unwrap();
        let rust = compiled.program.to_rust_wgpu("totals.wgsl", ShaderFormat::Wgsl).unwrap();
        for artifact in [&compiled.wgsl, &whl, &rust] {
            for prefix in ["egg_kernel", "egg_resource", "egg_helper"] {
                assert!(!artifact.contains(prefix), "unexpected generated name {prefix}");
            }
        }
    }
    #[test]
    fn rust_allocates_graphics_intermediates_from_host_scalar_expressions() {
        let program = compile(
        include_str!("../../../tests/compiler/library/published_shader_names_identify_source_phases_and_stay_consistent.wyn"),
    );
        let entry = &program.entries[0];
        assert!(program.interface.source_results.is_empty());
        assert!(entry.allocations.iter().any(|a| matches!(a, Allocation::Buffer { .. })));
        let rust = program.to_rust_wgpu("scratch.wgsl", ShaderFormat::Wgsl).unwrap();
        assert!(rust.contains("count: &ParameterBuffer"));
        assert!(entry.inputs.iter().all(|id| {
            program.interface.frame_graph.resources[id.0].bindings.iter().all(|binding| {
                !matches!(
                    program.bindings(binding.pipeline_index)[binding.binding_index],
                    Binding::StorageBuffer {
                        usage: wyn_core::host::BufferUsage::Output
                            | wyn_core::host::BufferUsage::Intermediate,
                        ..
                    }
                )
            })
        }));
        assert!(!rust.contains("count_host: &[u8]"));
        assert!(!rust.contains("read_gpu_word"));
    }
    #[test]
    fn graphics_capture_uses_the_produced_buffer_after_its_writer() {
        let program = compile(include_str!("../../../testfiles/playground/conway.wyn"));
        let [entry] = program.entries.as_slice() else {
            panic!("one graphics host entry");
        };
        let result = entry.results[0];
        assert!(
            !entry.inputs.contains(&result),
            "computed capture is allocated by the host function"
        );
        let draw = entry.operations.iter().position(|op| matches!(op, Operation::Draw { .. })).unwrap();
        let producer = entry
            .operations
            .iter()
            .position(|op| {
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
                    .any(|&binding| program.binding_resource(*pipeline, binding).unwrap() == result)
            })
            .unwrap();
        assert!(producer < draw, "graphics waits for its captured buffer");
        let whl = program.to_whl("conway.wgsl", ShaderFormat::Wgsl).unwrap();
        check_whl(&whl);
        assert!(whl.contains("define-gpu-graphics"));
        assert!(whl.contains("gpu-draw"));
        let rust = program.to_rust_wgpu("conway.wgsl", ShaderFormat::Wgsl).unwrap();
        assert!(rust.contains("create_render_pipeline"));
        assert!(rust.contains("begin_render_pass"));
        assert!(rust.contains("OutputResource::Texture(Texture::clone("));
        assert!(!rust.contains("read_buffers"));
        for pipeline in &program.interface.pipelines {
            if let Pipeline::Graphics(graphics) = pipeline {
                for stage in &graphics.stages {
                    assert!(
                        stage.entry_point.starts_with(&format!("{}_", stage.owner)),
                        "{}",
                        stage.entry_point
                    );
                }
            }
        }
    }
    #[test]
    fn graphics_aliases_preserve_host_buffer_identity() {
        // The exhaustive spellings stop at stage extraction. These two in-process
        // compilations check the final allocation and scheduling contract as well.
        for (prefix, input, computed) in [
            ("", "values", false),
            (
                "let produced = map(|v| v + @[1.0, 0.0, 0.0, 0.0], values)",
                "produced",
                true,
            ),
        ] {
            let program = compile(&format!(
            "entry reproduce(values: []vec4f32, target: render_target<vec4f32>) render_target<vec4f32> =
               {prefix}
               let first = ({input}, target) let (unpacked, _) = first
               let second = {{ data = (unpacked, values), tag = 2i32 }}
               let alias = second.data.0
               let fragments = rasterize_triangles(direct_draw(3u32, 1u32),
                 |_, _, _| vertex_output(alias[0], ())) in
               shade(target, fragments, |_, _, _, _, _| alias[0])"
        ));
            let [entry] = program.entries.as_slice() else {
                panic!("one source entry")
            };
            assert_eq!(entry.inputs.len(), 2, "generated buffers must remain internal");
            assert_eq!(entry.results.len(), 1, "only the render target is returned");
            assert!(entry.inputs.contains(&entry.results[0]), "borrowed render target");
            // Computed captures allocate element storage and an explicit input length.
            assert_eq!(entry.allocations.len(), 2 * usize::from(computed));
            let captured = if computed {
                let Allocation::Buffer { resource, .. } = entry.allocations[0] else {
                    panic!("computed capture needs one buffer")
                };
                assert!(!entry.inputs.contains(&resource));
                resource
            } else {
                assert!(entry.operations.iter().all(|op| matches!(op, Operation::Draw { .. })));
                *entry.inputs.iter().find(|id| **id != entry.results[0]).unwrap()
            };
            let draws = entry
                .operations
                .iter()
                .enumerate()
                .filter_map(|(index, op)| match op {
                    Operation::Draw { pipeline } => Some((index, *pipeline)),
                    _ => None,
                })
                .collect::<Vec<_>>();
            let [(draw, pipeline)] = draws.as_slice() else {
                panic!("one draw")
            };
            let Pipeline::Graphics(graphics) = &program.interface.pipelines[*pipeline] else {
                panic!("graphics pipeline")
            };
            assert_eq!(graphics.stages.len(), 2);
            for stage in &graphics.stages {
                let buffers = stage
                    .uses
                    .reads
                    .iter()
                    .filter(|&&binding| matches!(graphics.bindings[binding], Binding::StorageBuffer { .. }))
                    .map(|&binding| program.binding_resource(*pipeline, binding).unwrap())
                    .collect::<Vec<_>>();
                assert_eq!(
                    buffers,
                    [captured],
                    "{:?} capture, computed={computed}",
                    stage.stage
                );
            }
            if computed {
                assert!(
                    entry.operations[..*draw].iter().any(|op| {
                        let Operation::Dispatch { pipeline, stage, .. } = op else {
                            return false;
                        };
                        let Pipeline::Compute(compute) = &program.interface.pipelines[*pipeline] else {
                            return false;
                        };
                        compute.stages[*stage].uses.writes.iter().any(|&binding| {
                            program.binding_resource(*pipeline, binding).unwrap() == captured
                        })
                    }),
                    "the captured buffer must be produced before the draw"
                );
            }
        }
    }
    #[test]
    fn texture_readers_finish_before_the_target_is_overwritten() {
        let program = compile(include_str!(
            "../../../tests/compiler/library/graphics_aliases_preserve_host_buffer_identity.wyn"
        ));
        let entry = &program.entries[0];
        let [sampled, target] = entry.results.as_slice() else {
            panic!("sampled values and updated target")
        };
        let graph = &program.interface.frame_graph;
        let draw = |operation| {
            graph
                .passes
                .iter()
                .position(|pass| {
                    matches!(
                        &program.interface.pipelines[pass.pipeline_index],
                        Pipeline::Graphics(graphics) if graphics.source_operation == Some(operation)
                    )
                })
                .unwrap()
        };
        let reader = graph
            .passes
            .iter()
            .position(|pass| pass.writes.iter().any(|access| access.resource == sampled.0))
            .unwrap();
        assert!(graph.passes[reader].reads.iter().any(|access| access.resource == target.0));
        for writer in [draw(0), draw(1)] {
            assert!(graph.passes[writer].writes.iter().any(|access| access.resource == target.0));
        }
        assert_dependency(graph, draw(0), reader);
        assert_dependency(graph, reader, draw(1));
    }
}

mod source_spirv {
    #[test]
    fn fixed_array_output_constructs_only_the_indexed_value() {
        use std::collections::HashSet;
        use wspirv::spirv::Op;

        let module = wspirv::dr::load_words(
            compile_to_spirv(
                "entry repro(xs: []i32) ([]i32, [1]i32) =
              let n = xs[0] in (map(|x| x + n, xs), [n])",
            )
            .unwrap(),
        )
        .unwrap();
        let arrays: HashSet<_> = module
            .types_global_values
            .iter()
            .filter(|inst| inst.class.opcode == Op::TypeArray)
            .filter_map(|inst| inst.result_id)
            .collect();
        let constructors = module
            .functions
            .iter()
            .flat_map(|function| &function.blocks)
            .flat_map(|block| &block.instructions)
            .filter(|inst| {
                inst.class.opcode == Op::CompositeConstruct
                    && inst.result_type.is_some_and(|ty| arrays.contains(&ty))
            })
            .count();
        assert_eq!(
            constructors, 0,
            "a single-element output needs no array constructor"
        );
    }
    use super::*;
    fn compile_to_spirv(source: &str) -> wyn_core::error::Result<Vec<u32>> {
        Ok(compile_to_spirv_output(source)?.spirv)
    }

    fn disasm(words: &[u32]) -> String {
        use wspirv::binary::Disassemble;
        wspirv::dr::load_words(words).expect("backend emitted valid SPIR-V").disassemble()
    }
    #[test]
    fn fragment_helper_result_is_evaluated_once_for_multiple_projections() {
        use wspirv::spirv::Op;

        let module = wspirv::dr::load_words(
        compile_to_spirv(
            include_str!("../../../tests/compiler/library/texture_readers_finish_before_the_target_is_overwritten.wyn"),
        )
        .unwrap(),
    )
    .unwrap();
        let calls = module
            .functions
            .iter()
            .flat_map(|function| &function.blocks)
            .flat_map(|block| &block.instructions)
            .filter(|inst| inst.class.opcode == Op::FunctionCall)
            .count();
        assert!(calls <= 1, "the fragment must not duplicate its helper");
        let predicates =
            module.all_inst_iter().filter(|inst| inst.class.opcode == Op::FOrdLessThan).count();
        assert_eq!(
            predicates, 1,
            "the helper's predicate must be evaluated once, even when inlined"
        );
    }
    #[test]
    fn vertex_conditional_is_evaluated_once_for_nested_varyings() {
        use wspirv::dr::Operand;
        use wspirv::spirv::{ExecutionModel, Op};

        let module = wspirv::dr::load_words(
        compile_to_spirv(
            include_str!("../../../tests/compiler/library/fragment_helper_result_is_evaluated_once_for_multiple_projections.wyn"),
        )
        .unwrap(),
    )
    .unwrap();
        let entry = module
            .entry_points
            .iter()
            .find(|inst| inst.operands.first() == Some(&Operand::ExecutionModel(ExecutionModel::Vertex)))
            .unwrap();
        let Operand::IdRef(vertex_id) = entry.operands[1] else {
            panic!("vertex function")
        };
        let vertex = module
            .functions
            .iter()
            .find(|function| function.def.as_ref().unwrap().result_id == Some(vertex_id))
            .unwrap();
        let predicates = vertex
            .blocks
            .iter()
            .flat_map(|block| &block.instructions)
            .filter(|inst| inst.class.opcode == Op::FOrdLessThan)
            .count();
        assert_eq!(
            predicates, 1,
            "varying projections must share the predicate even when selecting lanes separately"
        );
    }
    #[test]
    fn entry_point_interfaces_are_unique_when_storage_input_and_output_alias() {
        use wspirv::dr::Operand;
        use wyn_core::LookupSet;

        let spirv = compile_to_spirv(include_str!(
            "../../../tests/compiler/library/vertex_conditional_is_evaluated_once_for_nested_varyings.wyn"
        ))
        .expect("record-array graphics program should compile");

        let module = wspirv::dr::load_words(&spirv).expect("backend emitted parseable SPIR-V");
        for entry in &module.entry_points {
            let interface_ids = entry.operands.iter().skip(3).filter_map(|operand| match operand {
                Operand::IdRef(id) => Some(*id),
                _ => None,
            });
            let mut seen = LookupSet::new();
            for interface_id in interface_ids {
                assert!(
                    seen.insert(interface_id),
                    "OpEntryPoint contains duplicate interface %{interface_id}"
                );
            }
        }
    }
    #[test]
    fn scatter_into_storage_buffer_lowers() {
        let spirv = compile_to_spirv(
        include_str!("../../../tests/compiler/library/entry_point_interfaces_are_unique_when_storage_input_and_output_alias.wyn"),
    )
    .expect("scatter rasterizer must lower to SPIR-V");
        assert_eq!(spirv[0], 0x07230203, "SPIR-V magic number");
        // Parse instruction boundaries: an SSA id whose low bits equal OpStore's
        // opcode is not a store. The five particles are processed by a loop.
        let module = wspirv::dr::load_words(&spirv).unwrap();
        let stores: Vec<_> =
            module.all_inst_iter().filter(|inst| inst.class.opcode == spirv::Op::Store).collect();
        assert_eq!(
            stores.len(),
            1,
            "scatter emits one store in its runtime loop\n{}",
            disasm(&spirv)
        );
        let wspirv::dr::Operand::IdRef(pointer) = stores[0].operands[0] else {
            panic!("store pointer")
        };
        let address = module.all_inst_iter().find(|inst| inst.result_id == Some(pointer)).unwrap();
        assert_eq!(address.class.opcode, spirv::Op::AccessChain);
        let pointer_type =
            module.types_global_values.iter().find(|inst| inst.result_id == address.result_type).unwrap();
        assert_eq!(
            pointer_type.operands[0],
            wspirv::dr::Operand::StorageClass(spirv::StorageClass::StorageBuffer)
        );
        assert!(module.all_inst_iter().any(|inst| inst.class.opcode == spirv::Op::LoopMerge));
    }
}

mod source_wgsl {
    #[test]
    fn wgsl_composite_constant_inlines_transitive_constant_dependencies() {
        let wgsl = compile_to_wgsl(
            r#"
type word64 = (u32, u32)

def WORDS: [2]word64 = [
  (0x11223344u32, 0x55667788u32),
  (0x99aabbccu32, 0xddeeff00u32)
]

def COPIED_WORDS: [2]word64 = [WORDS[0], WORDS[1]]

def select_word(words: [2]word64, index: i32) word64 = words[index]

entry composite_constant(index: u32) word64 =
  select_word(COPIED_WORDS, i32.u32(index & 1u32))
"#,
        )
        .expect("compile a composite constant with a constant dependency");

        validate_wgsl(&wgsl);
        assert!(
            !wgsl.contains("w_WORDS"),
            "transitive constants must be expanded before WGSL lowering:\n{wgsl}"
        );
    }
    #[test]
    fn wgsl_testfile_reduce_compute() {
        // Exercises the function-scope hoist of SSA inst-result `var`
        // declarations. Without hoisting, a storage-view offset declared
        // inside the reduction loop would be out-of-scope at the post-loop
        // write site, which WGSL's textual block scoping rejects.
        validate_testfile_wgsl("testfiles/reduce_compute.wyn");
    }
    fn validate_testfile_wgsl(path: &str) {
        let source =
            std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(path))
                .unwrap();
        let wgsl = stacker::grow(16 * 1024 * 1024, || compile_to_wgsl(&source)).unwrap();
        validate_wgsl(&wgsl);
    }

    use super::*;
    use wyn_core::LookupSet;
    fn compile_to_wgsl(source: &str) -> wyn_core::error::Result<String> {
        wyn_core::lower_ssa_to_wgsl(compile_thru_ssa(source)?)
    }

    fn validate_wgsl(source: &str) {
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|e| panic!("naga parse failed:\n{}\n\n--- source ---\n{}", e, source));
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator
            .validate(&module)
            .unwrap_or_else(|e| panic!("naga validation failed:\n{:?}\n\n--- source ---\n{}", e, source));
    }
    fn stores_binding(module: &naga::Module, group: u32, binding: u32) -> bool {
        fn root(
            function: &naga::Function,
            mut expression: naga::Handle<naga::Expression>,
        ) -> Option<naga::Handle<naga::GlobalVariable>> {
            loop {
                match function.expressions[expression] {
                    naga::Expression::GlobalVariable(global) => return Some(global),
                    naga::Expression::Access { base, .. } | naga::Expression::AccessIndex { base, .. } => {
                        expression = base
                    }
                    _ => return None,
                }
            }
        }
        fn stores(
            module: &naga::Module,
            function: &naga::Function,
            block: &naga::Block,
            group: u32,
            binding: u32,
        ) -> bool {
            block.iter().any(|statement| match statement {
                naga::Statement::Store { pointer, .. } => root(function, *pointer).is_some_and(|global| {
                    module.global_variables[global]
                        .binding
                        .as_ref()
                        .is_some_and(|b| b.group == group && b.binding == binding)
                }),
                naga::Statement::Block(block) => stores(module, function, block, group, binding),
                naga::Statement::If { accept, reject, .. } => {
                    stores(module, function, accept, group, binding)
                        || stores(module, function, reject, group, binding)
                }
                naga::Statement::Loop { body, continuing, .. } => {
                    stores(module, function, body, group, binding)
                        || stores(module, function, continuing, group, binding)
                }
                naga::Statement::Switch { cases, .. } => {
                    cases.iter().any(|case| stores(module, function, &case.body, group, binding))
                }
                _ => false,
            })
        }
        module
            .functions
            .iter()
            .map(|(_, f)| f)
            .chain(module.entry_points.iter().map(|entry| &entry.function))
            .any(|f| stores(module, f, &f.body, group, binding))
    }
    #[test]
    fn wgsl_testfile_array_call_demo() {
        // Exercises `_w_intrinsic_slice` view-to-view lowering. `data[0..4]`
        // with `data: []f32` remains a storage-backed view and is passed to
        // a view-specialized user function.
        validate_testfile_wgsl("testfiles/array_call_demo.wyn");
    }
    #[test]
    fn wgsl_testfile_pc_echo_test() {
        // Exercises push-constant-backed compute inputs — broadcast scalars
        // and small arrays — routed through a synthesized storage-read
        // block (WGSL uniform alignment would reject the array stride).
        validate_testfile_wgsl("testfiles/pc_echo_test.wyn");
    }
    #[test]
    fn wgsl_testfile_creation() {
        validate_testfile_wgsl("testfiles/playground/creation.wyn");
    }
    #[test]
    fn wgsl_testfile_lava() {
        validate_testfile_wgsl("testfiles/playground/lava.wyn");
    }
    #[test]
    fn wgsl_testfile_seascape() {
        validate_testfile_wgsl("testfiles/playground/seascape.wyn");
    }
    #[test]
    fn wgsl_testfile_raytrace() {
        validate_testfile_wgsl("testfiles/playground/raytrace.wyn");
    }
    #[test]
    fn wgsl_testfile_mandelbulb() {
        validate_testfile_wgsl("testfiles/playground/mandelbulb.wyn");
    }
    #[test]
    fn wgsl_testfile_da_rasterizer() {
        validate_testfile_wgsl("testfiles/playground/da_rasterizer.wyn");
    }
    #[test]
    fn wgsl_testfile_loopingspline() {
        validate_testfile_wgsl("testfiles/playground/loopingspline.wyn");
    }
    #[test]
    fn wgsl_compiler_assigned_scalar_capture_emits_uniform_binding() {
        let wgsl = compile_to_wgsl(include_str!(
            "../../../tests/compiler/library/wgsl_compiler_assigned_scalar_capture_emits_uniform_binding.wyn"
        ))
        .expect("compile");
        validate_wgsl(&wgsl);
        assert!(
            wgsl.contains("var<uniform>"),
            "expected a compiler-assigned uniform capture, got:\n{wgsl}"
        );
    }
    #[test]
    fn wgsl_compute_multi_output_runtime_sized_arrays() {
        // A compute entry returning a tuple of >1 runtime-sized array: each
        // field's producing `map` must stream into its own bound output storage
        // view.
        let wgsl = compile_to_wgsl(include_str!(
            "../../../tests/compiler/library/wgsl_compute_multi_output_runtime_sized_arrays_additional.wyn"
        ))
        .expect("compile");
        validate_wgsl(&wgsl);

        // Two distinct output buffers, each written directly by its own map.
        assert!(
            wgsl.contains("_buf_0_1[") && wgsl.contains("] = "),
            "expected a direct write to output buffer 1:\n{wgsl}"
        );
        assert!(
            wgsl.contains("_buf_0_2[") && wgsl.contains("] = "),
            "expected a direct write to output buffer 2:\n{wgsl}"
        );
    }
    #[test]
    fn wgsl_gather_computed_array() {
        // A randomly-indexed computed array is materialized into its own
        // storage buffer; the consumer reads it by index. The WGSL backend must
        // emit the gather buffer as a module-scope `var<storage>` and validate
        // (naga) end-to-end. Three buffers: input `bh` (0), consumer output (1),
        // gather intermediate (2).
        let wgsl = compile_to_wgsl(include_str!(
            "../../../tests/compiler/library/wgsl_compute_multi_output_runtime_sized_arrays.wyn"
        ))
        .expect("gather program must lower to WGSL");

        validate_wgsl(&wgsl);

        assert!(
            wgsl.contains("@group(0) @binding(2)") && wgsl.contains("var<storage"),
            "the gather buffer must be declared as a storage binding:\n{wgsl}"
        );
        // The producer and consumer share one physical pipeline, so the consumer
        // indexes the same read_write global selected by that pipeline layout.
        assert!(
            wgsl.contains("@group(0) @binding(2) var<storage, read_write> _buf_0_2:")
                && wgsl.contains("_buf_0_2["),
            "consumer must read the gather buffer by index:\n{wgsl}"
        );
    }
    #[test]
    fn wgsl_scatter_into_storage_buffer() {
        let source = include_str!("../../../tests/compiler/library/wgsl_gather_computed_array.wyn");
        let wgsl = compile_to_wgsl(source).expect("compile to WGSL");
        assert!(
            wgsl.contains("@group(0) @binding(1) var<storage, read_write>"),
            "scatter destination must be read_write:\n{wgsl}"
        );
        assert!(
            wgsl.contains("_buf_0_1["),
            "scatter must emit indexed stores into the framebuffer:\n{wgsl}"
        );
    }
    #[test]
    fn wgsl_map_over_unique_storage_view_updates_backing_buffer() {
        let wgsl = compile_to_wgsl(include_str!(
            "../../../tests/compiler/library/wgsl_map_over_unique_storage_view_updates_backing_buffer_additional.wyn"
        ))
        .expect("map/scatter over a unique storage view must lower");

        validate_wgsl(&wgsl);
        let module = naga::front::wgsl::parse_str(&wgsl).unwrap();
        assert!(
            stores_binding(&module, 0, 0),
            "updates must store into the input binding: {wgsl}"
        );
        assert!(
            !stores_binding(&module, 0, 1),
            "the additional binding contains read-only length metadata"
        );
        assert_eq!(
            module
                .global_variables
                .iter()
                .filter(|(_, v)| v.binding.as_ref().is_some_and(|b| b.group == 0))
                .count(),
            2,
            "array storage and length metadata"
        );
    }
    #[test]
    fn wgsl_dynamic_index_of_captured_fixed_array_values() {
        // Captured arrays pass through callable parameters. Check that their
        // dynamic indexing and the generated helpers remain valid WGSL.
        let wgsl = std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            compile_to_wgsl(
                include_str!("../../../tests/compiler/library/wgsl_map_over_unique_storage_view_updates_backing_buffer.wyn"),
            )
        })
        .expect("spawn fixed-array regression compiler thread")
        .join()
        .expect("fixed-array regression compiler thread panicked")
        .expect("compile dynamic indexing of captured fixed arrays");

        validate_wgsl(&wgsl);
        assert!(
            wgsl.contains("_pc0.w_roots") && wgsl.contains("array<u32, 2>"),
            "the captured parameter must reach the indexing helpers:\n{wgsl}"
        );
        assert!(
            !wgsl.lines().any(|line| {
                (line.trim_start().starts_with("var ") || line.trim_start().starts_with("let "))
                    && line.contains("array<u32, 2>")
                    && line.contains(" = _pc0.w_roots[")
            }),
            "dynamic indexing must not copy an aggregate row out of the captured parameter:\n{wgsl}"
        );
    }
    #[test]
    fn wgsl_distinctness_indexes_512_element_array_without_inner_copies() {
        let wgsl = std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(|| {
                compile_to_wgsl(include_str!(
                    "../../../tests/compiler/library/wgsl_dynamic_index_of_captured_fixed_array_values.wyn"
                ))
            })
            .expect("spawn distinctness regression compiler thread")
            .join()
            .expect("distinctness regression compiler thread panicked")
            .expect("compile 512-element distinctness check");

        validate_wgsl(&wgsl);
        let module = naga::front::wgsl::parse_str(&wgsl).unwrap();
        let is_row = |ty| matches!(module.types[ty].inner,naga::TypeInner::Array {size:naga::ArraySize::Constant(n),..} if n.get()==512);
        let function = &module.entry_points[0].function;
        let rows: LookupSet<_> = function
            .local_variables
            .iter()
            .filter(|(_, local)| is_row(local.ty))
            .map(|(id, _)| id)
            .collect();
        let is_row_pointer = |expression| {
            matches!(function.expressions[expression],
        naga::Expression::LocalVariable(local) if rows.contains(&local))
        };
        let accesses = function.expressions.iter().filter(|(_, expression)|
        matches!(expression, naga::Expression::Access {base,..} if is_row_pointer(*base))).count();
        assert!(
            accesses >= 2,
            "both distinctness indices must address stable array storage"
        );
        let mut blocks = vec![(&function.body, 0)];
        while let Some((block, depth)) = blocks.pop() {
            for statement in block {
                match statement {
                    naga::Statement::Store { pointer, .. } if is_row_pointer(*pointer) => {
                        assert!(
                            depth <= 1,
                            "array snapshot repeated inside the distinctness loops"
                        );
                    }
                    naga::Statement::Block(block) => blocks.push((block, depth)),
                    naga::Statement::If { accept, reject, .. } => {
                        blocks.extend([(accept, depth), (reject, depth)]);
                    }
                    naga::Statement::Loop { body, continuing, .. } => {
                        blocks.extend([(body, depth + 1), (continuing, depth + 1)]);
                    }
                    naga::Statement::Switch { cases, .. } => {
                        blocks.extend(cases.iter().map(|case| (&case.body, depth)));
                    }
                    _ => {}
                }
            }
        }
    }
    #[test]
    fn wgsl_structural_capture_emits_uniform_struct_and_validates() {
        let wgsl = compile_to_wgsl(
        include_str!("../../../tests/compiler/library/wgsl_distinctness_indexes_512_element_array_without_inner_copies.wyn"),
    )
    .expect("compile");
        validate_wgsl(&wgsl);
        assert!(wgsl.contains("struct T0"));
        assert!(
            wgsl.contains("var<uniform>"),
            "expected a compiler-assigned structural uniform capture, got:\n{wgsl}"
        );
    }
}

mod constant_math {
    use super::*;
    use wspirv::dr;
    use wyn_core::ssa::types::{ConstantValue, InstKind, ValueRef};
    use wyn_core::{
        op::{BinaryOperator, OpTag},
        ssa,
    };
    const FIXTURE: &str = include_str!("../../../testfiles/constant_shader_math.wyn");

    #[test]
    fn constant_shader_math_folds_after_inlining_and_preserves_dynamic_depth_order() {
        let program = compile_thru_ssa(FIXTURE).unwrap();
        let placed = ssa::place_floating(ssa::optimize(program.clone())).unwrap();
        let prepared = ssa::prepare_spirv(ssa::filter_reachable(placed)).unwrap();
        let entry = prepared.entry_points.iter().find(|e| e.name == "constant_math_compute").unwrap();
        let function = &entry.body.inner;
        let depth = ValueRef::Const(ConstantValue::from_f32(1.0 / (0.1f32 - 1000.0)));
        let first = function
            .insts
            .values()
            .find_map(|node| match &node.data {
                InstKind::Op {
                    tag: OpTag::BinOp(BinaryOperator::Multiply),
                    operands,
                } if operands.get(1) == Some(&ValueRef::Const(ConstantValue::from_f32(1000.0))) => {
                    node.result
                }
                _ => None,
            })
            .expect("p.z * far must retain its own rounding step");
        assert!(function.insts.values().any(|node| matches!(&node.data,
        InstKind::Op { tag: OpTag::BinOp(BinaryOperator::Multiply), operands }
        if operands == &[first.into(), depth])));

        let lowered = wyn_core::lower_ssa_to_spirv(program.clone()).unwrap();
        let module = dr::load_words(&lowered.spirv).unwrap();
        let compute = entry_function(&module, "constant_math_compute");
        let count = |op| instructions(compute).filter(|i| i.class.opcode == op).count();
        assert_eq!(
            count(spirv::Op::FDiv),
            2,
            "only dynamic aspect/projection divisions remain"
        );
        assert_eq!(count(spirv::Op::FSub), 0);
        assert!(!instructions(compute).any(|i| matches!(
            i.operands.get(1),
            Some(dr::Operand::LiteralExtInstInteger(11 | 15 | 46))
        )));

        let wgsl = wyn_core::lower_ssa_to_wgsl(program).unwrap();
        assert!(!wgsl.contains("radians("));
        assert!(!wgsl.contains("tan("));
        assert!(!wgsl.contains("mix("));
        validate(naga::front::wgsl::parse_str(&wgsl).unwrap());
        let bytes = lowered.spirv.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<_>>();
        validate(naga::front::spv::parse_u8_slice(&bytes, &Default::default()).unwrap());
    }

    #[test]
    fn unused_shader_inputs_keep_their_interfaces_without_runtime_setup() {
        let program = compile_thru_ssa(FIXTURE).unwrap();
        // Interface declarations survive even when the shader does not read them.
        for name in ["unused_inputs_vertex", "unused_inputs_fragment"] {
            let entry = program.entry_points.iter().find(|e| e.name == name).unwrap();
            assert_eq!(entry.inputs.len(), 4);
            assert_eq!(
                entry.inputs.iter().filter(|i| i.storage_binding().is_some()).count(),
                2
            );
            assert_eq!(
                entry.inputs.iter().filter(|i| i.uniform_binding().is_some()).count(),
                1
            );
        }
        let lowered = wyn_core::lower_ssa_to_spirv(program).unwrap();
        let module = dr::load_words(&lowered.spirv).unwrap();
        for (name, lengths, loads) in [("unused_inputs_vertex", 1, 2), ("unused_inputs_fragment", 0, 1)] {
            let function = entry_function(&module, name);
            assert_eq!(
                instructions(function).filter(|i| i.class.opcode == spirv::Op::ArrayLength).count(),
                lengths
            );
            // Vertex index + point, or fragment color. Neither shader reads config.
            assert_eq!(
                instructions(function).filter(|i| i.class.opcode == spirv::Op::Load).count(),
                loads
            );
        }
    }

    fn entry_function<'a>(module: &'a dr::Module, name: &str) -> &'a dr::Function {
        let entry = module
            .entry_points
            .iter()
            .find(|e| matches!(e.operands.get(2), Some(dr::Operand::LiteralString(n)) if n == name))
            .unwrap();
        let dr::Operand::IdRef(id) = entry.operands[1] else {
            panic!("entry id")
        };
        module.functions.iter().find(|f| f.def.as_ref().and_then(|i| i.result_id) == Some(id)).unwrap()
    }

    fn instructions(function: &dr::Function) -> impl Iterator<Item = &dr::Instruction> {
        function.blocks.iter().flat_map(|b| &b.instructions)
    }

    fn validate(module: naga::Module) {
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }
}
