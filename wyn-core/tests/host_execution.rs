//! Execute emitted WHL with a tracing backend; no GPU work is simulated.
#[path = "support/pipeline.rs"]
mod pipeline;
#[path = "../../wyn-host-interp/src/test_backend.rs"]
mod test_backend;
use test_backend::Trace;
use wyn_core::{lower_ssa_to_spirv, lower_ssa_to_wgsl_with_program, CodegenTarget};
use wyn_host::{Operation, ShaderFormat};
use wyn_host_interp::{Backend, Error, Number, Options, Program, Result, Value};

fn generated(source: &str) -> Program {
    let source = pipeline::source(source);
    let output = lower_ssa_to_wgsl_with_program(pipeline::lower(&source, CodegenTarget::Wgsl)).unwrap();
    Program::parse(&output.program.to_whl("test", ShaderFormat::Wgsl).unwrap()).unwrap()
}

#[test]
fn uniform_sized_launches_allocate_capacity_and_clamp_the_grid() {
    #[derive(Default)]
    struct LaunchTrace {
        trace: Trace,
        allocation: Option<(u64, u64)>,
    }
    impl Backend for LaunchTrace {
        fn call(&mut self, program: &Program, name: &str, args: &[Value]) -> Result<Value> {
            match name {
                "gpu-alloc" => {
                    let buffer = self.trace.input(vec![]);
                    assert!(self.allocation.is_none(), "one map output allocation");
                    self.allocation = Some((buffer.handle()?, args[0].u64()?));
                    Ok(buffer)
                }
                "gpu-buffer-size" => {
                    let (buffer, bytes) = self.allocation.unwrap();
                    assert_eq!(args[0].handle()?, buffer);
                    Ok(Value::Number(Number::U64(bytes)))
                }
                "gpu-texture-view" => Ok(args[0].clone()),
                "gpu-draw" => Ok(Value::Nil),
                _ => self.trace.call(program, name, args),
            }
        }
    }
    let source = include_str!("../../testfiles/rust_host_runtime_dispatch.wyn");
    let source = pipeline::source(source);
    let ssa = pipeline::lower(&source, CodegenTarget::Portable);
    for format in [ShaderFormat::Spirv, ShaderFormat::Wgsl] {
        let ssa = ssa.clone();
        let program = match format {
            ShaderFormat::Spirv => lower_ssa_to_spirv(ssa).unwrap().program,
            ShaderFormat::Wgsl => lower_ssa_to_wgsl_with_program(ssa).unwrap().program,
        };
        let program = Program::parse(&program.to_whl("runtime_dispatch", format).unwrap()).unwrap();
        let kernels = program.kernels.values().collect::<Vec<_>>();
        let [kernel] = kernels.as_slice() else {
            panic!("one parallel map kernel");
        };
        let dimensions = kernel.options.get(":workgroup-size").unwrap().list().unwrap();
        let width = u64::from(dimensions[0].u32().unwrap());
        for elements in [
            0,
            width,
            width + 1,
            width * 1_000,
            width * 16_000,
            65_535 * width + 1,
        ] {
            let groups = elements.div_ceil(width).min(65_535) as u32;
            let mut backend = LaunchTrace::default();
            let arguments: Vec<_> = program
                .entry("reproduce")
                .unwrap()
                .parameters
                .iter()
                .map(|parameter| {
                    if parameter.source_name() == "frame" {
                        backend.trace.input(
                            [elements as f32, 1.0, 0.0, 0.0]
                                .into_iter()
                                .flat_map(f32::to_le_bytes)
                                .collect(),
                        )
                    } else {
                        backend.trace.input(vec![])
                    }
                })
                .collect();
            assert_eq!(arguments.len(), 2, "only source inputs are caller-supplied");
            program.run("reproduce", &arguments, &mut backend).unwrap();
            assert_eq!(backend.allocation.unwrap().1, elements.max(1) * 16);
            let [(_, actual)] = backend.trace.dispatches.as_slice() else {
                panic!("one parallel map");
            };
            assert_eq!(actual, &[groups, 1, 1], "{format:?}, elements={elements}");
        }
    }
}

#[test]
fn counted_loop_results_survive_releasing_temporary_buffers() {
    let program = generated(
        "entry counted(limit:i32) ([4]i32,[4]i32,i32) =
           loop (a,b,total) = ([1,2,3,4],[10,20,30,40],0i32) for i < limit do
             (map(|x|x+i,b),map(|x|x+1,a),total+reduce((+),0i32,a))",
    );
    let entry = program.entry("counted").unwrap();
    for limit in [0_i32, 1, 3] {
        let mut backend = Trace::default();
        let inputs = entry
            .parameters
            .iter()
            .map(|_| backend.input(limit.to_le_bytes().to_vec()))
            .collect::<Vec<_>>();
        let result = program.run("counted", &inputs, &mut backend).unwrap();
        let results = result.list().unwrap();
        assert_eq!(results.len(), 3);
        for (result, size) in results.iter().zip([16, 16, 4]) {
            assert_eq!(backend.buffers[&result.handle().unwrap()].len(), size);
        }
    }
}

#[test]
fn branch_output_shape_is_read_before_temporary_storage_is_freed() {
    for source in [
        include_str!("../../testfiles/regressions/triangulation_runtime_array_branch.wyn"),
        include_str!("../../testfiles/regressions/triangulation_runtime_array_zip_map.wyn"),
    ] {
        let source = source.replacen("length(xs)>10", "xs[0]>10", 1);
        let program = generated(&source);
        let mut backend = Trace::default();
        let input = backend.input(vec![0; 48]);
        let result = program.run("main", &[input], &mut backend).unwrap();
        // Trace leaves the GPU predicate false: the selected result retains the input shape.
        assert_eq!(backend.buffers[&result.handle().unwrap()].len(), 48);
        let locations = backend.device_reads.iter().copied().collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            locations.len(),
            1,
            "only the GPU branch predicate requires readback"
        );
        assert!(locations.iter().all(|(buffer, _)| backend.freed.contains(buffer)));
    }
}

#[test]
fn cpu_known_branch_shapes_need_no_device_readback() {
    let program = generated(include_str!(
        "../../testfiles/regressions/triangulation_runtime_array_branch.wyn"
    ));
    for length in [0, 3, 12] {
        let mut backend = Trace::default();
        let input = backend.input(vec![0; length * 4]);
        let result = program.run("main", &[input], &mut backend).unwrap();
        assert_eq!(
            backend.buffers[&result.handle().unwrap()].len(),
            if length > 10 { 0 } else { length * 4 }
        );
        assert!(backend.device_reads.is_empty());
    }
}

#[test]
fn branch_selects_the_buffer_and_length_from_the_same_arm() {
    let source = pipeline::source(
        "entry choose(xs: []i32, ys: []i32) []i32 =
           if length(xs) > 2 then map(|x| x + 1, xs)
           else map(|y| y * 2, ys)",
    );
    let ssa = pipeline::lower(&source, CodegenTarget::Portable);
    for format in [ShaderFormat::Spirv, ShaderFormat::Wgsl] {
        let compiled = match format {
            ShaderFormat::Spirv => lower_ssa_to_spirv(ssa.clone()).unwrap().program,
            ShaderFormat::Wgsl => lower_ssa_to_wgsl_with_program(ssa.clone()).unwrap().program,
        };
        let entry = &compiled.entries[0];
        let joins: Vec<_> = entry
            .operations
            .iter()
            .filter_map(|op| match op {
                Operation::JoinBuffer { target, .. } => Some(*target),
                _ => None,
            })
            .collect();
        assert_eq!(entry.results, joins);
        assert!(entry.allocations().all(|allocation| !joins.contains(&allocation.resource())));
        compiled.to_rust_wgpu("branch", format).unwrap();
        let program = Program::parse(&compiled.to_whl("branch", format).unwrap()).unwrap();
        for (xs, ys, expected) in [(1, 5, 5), (3, 5, 3), (1, 0, 0)] {
            let mut backend = Trace::default();
            let inputs = program
                .entry("choose")
                .unwrap()
                .parameters
                .iter()
                .map(|param| {
                    let count = match param.source_name() {
                        "xs" => xs,
                        "ys" => ys,
                        name => panic!("unexpected input {name}"),
                    };
                    backend.input(vec![0; count * 4])
                })
                .collect::<Vec<_>>();
            let result = program.run("choose", &inputs, &mut backend).unwrap();
            assert_eq!(backend.buffers[&result.handle().unwrap()].len(), expected * 4);
            assert!(backend.device_reads.is_empty());
        }
    }
}

#[test]
fn host_loop_sizes_its_initial_storage_from_the_selected_branch_buffer() {
    let source = pipeline::source(
        "entry branch_loop(xs: []i32, ys: []i32, n: i32) []i32 =
           let a = if length(xs) > 2 then map(|x| x+1,xs) else map(|y| y*2,ys) in
           loop a = a for k < n do
             map(|i| if i < length(a) then a[i]+1 else k,iota(length(a)+1))",
    );
    let ssa = pipeline::lower(&source, CodegenTarget::Wgsl);
    let compiled = lower_ssa_to_wgsl_with_program(ssa).unwrap().program;
    let target = compiled.entries[0]
        .operations
        .iter()
        .find_map(|op| match op {
            Operation::JoinBuffer { target, .. } => Some(target.0),
            _ => None,
        })
        .expect("host branch selects a buffer");
    let rust = compiled.to_rust_wgpu("branch_loop", ShaderFormat::Wgsl).unwrap();
    let capacity = format!("let resource_{target}_bytes = resource_{target}.size();");
    let allocation = format!("signed_size(resource_{target}_bytes)");
    assert!(rust.contains(&capacity), "missing joined buffer capacity");
    assert!(rust.find(&capacity).unwrap() < rust.find(&allocation).unwrap());
}

#[test]
fn a_later_dispatch_consumes_the_selected_branch_buffer() {
    let source = pipeline::source(
        "entry consume(xs: []i32, ys: []i32) []i32 =
           let chosen = if length(xs) > 2 then map(|x| x + 1, xs)
                        else map(|y| y * 2, ys) in
           map(|value| value + 3, chosen)",
    );
    let ssa = pipeline::lower(&source, CodegenTarget::Wgsl);
    let compiled = lower_ssa_to_wgsl_with_program(ssa).unwrap().program;
    let operations = &compiled.entries[0].operations;
    let (join, target) = operations
        .iter()
        .enumerate()
        .find_map(|(index, op)| match op {
            Operation::JoinBuffer { target, .. } => Some((index, *target)),
            _ => None,
        })
        .expect("selected array backing");
    let consumer = compiled
        .interface
        .frame_graph
        .passes
        .iter()
        .find(|pass| pass.reads.iter().any(|read| read.resource == target.0))
        .expect("dispatch reading the selected array");
    assert!(operations[join + 1..].iter().any(|operation| matches!(operation,
        Operation::Dispatch { pipeline, stage, .. }
            if (*pipeline, *stage) == (consumer.pipeline_index, consumer.stage_index)
    )));
}

#[test]
fn returned_array_views_preserve_each_live_range_without_shape_readback() {
    let program = generated(include_str!(
        "../../testfiles/regressions/runtime_array_three_outputs.wyn"
    ));
    for length in [0usize, 3, 12] {
        let mut backend = Trace::default();
        let input = backend.input(vec![0; length * 4]);
        let result = program.run("main", &[input.clone()], &mut backend).unwrap();
        let result = result.list().unwrap();
        assert_eq!(result.len(), 3);
        let selected = if length > 10 { 0 } else { length };
        for (result, elements) in result.iter().zip([selected, selected, selected.saturating_sub(1)]) {
            assert_eq!(backend.buffers[&result.handle().unwrap()].len(), elements * 4);
        }
        assert_eq!(backend.buffers[&input.handle().unwrap()].len(), length * 4);
        assert!(backend.device_reads.is_empty());
    }
}

#[test]
fn copying_a_returned_slice_keeps_its_original_output_and_input_alive() {
    let program = generated(include_str!(
        "../../testfiles/regressions/triangulation_runtime_array_return_alias.wyn"
    ));
    let mut backend = Trace::default();
    let input = backend.input(vec![0; 12]);
    let result = program.run("main", &[input.clone()], &mut backend).unwrap();
    let result = result.list().unwrap();
    assert_eq!(result.len(), 2);
    assert_ne!(result[1].handle().unwrap(), result[0].handle().unwrap());
    assert_eq!(backend.buffers[&result[0].handle().unwrap()].len(), 12);
    assert_eq!(backend.buffers[&result[1].handle().unwrap()].len(), 8);
    assert!(!backend.freed.contains(&input.handle().unwrap()));
    assert!(backend.device_reads.is_empty());
}

#[test]
fn rectangular_results_keep_their_source_layout_and_byte_range() {
    for (expression, result_type, row_bytes) in [
        ("map(|x|map(|j|x+j,iota(3)),xs)", "[137][3]i32", 12),
        (
            "map(|x|map(|j|map(|k|x+j+k,iota(5)),iota(3)),xs)",
            "[137][3][5]i32",
            60,
        ),
    ] {
        let source = pipeline::source(&format!("entry main(xs:[137]i32) {result_type} = {expression}"));
        let ssa = pipeline::lower(&source, CodegenTarget::Portable);
        for format in [ShaderFormat::Spirv, ShaderFormat::Wgsl] {
            let host = match format {
                ShaderFormat::Spirv => lower_ssa_to_spirv(ssa.clone()).unwrap().program,
                ShaderFormat::Wgsl => lower_ssa_to_wgsl_with_program(ssa.clone()).unwrap().program,
            };
            let wyn_host::ResultLayout::Array {
                length,
                stride,
                element,
                ..
            } = &host.interface.source_results[0].layout
            else {
                panic!("array result layout");
            };
            assert_eq!(*length, Some(137));
            assert_eq!(*stride, row_bytes);
            let wyn_host::ResultLayout::Sequence { count, stride, .. } = element.as_ref() else {
                panic!("logical row layout");
            };
            assert_eq!(*count, 3);
            assert_eq!(*stride, row_bytes / 3);
            host.to_rust_wgpu("test", format).unwrap();
            let program = Program::parse(&host.to_whl("test", format).unwrap()).unwrap();
            let mut backend = Trace::default();
            let input = backend.input(vec![0; 137 * 4]);
            let result = program.run("main", &[input], &mut backend).unwrap();
            assert_eq!(
                backend.buffers[&result.handle().unwrap()].len(),
                137 * row_bytes as usize
            );
        }
    }
}

#[test]
fn runtime_array_loop_output_keeps_input_length() {
    let source = include_str!("../../testfiles/regressions/triangulation_runtime_array_loop.wyn");
    let source = pipeline::source(source);
    let ssa = pipeline::lower(&source, CodegenTarget::Portable);
    for format in [ShaderFormat::Spirv, ShaderFormat::Wgsl] {
        let ssa = ssa.clone();
        let host = match format {
            ShaderFormat::Spirv => lower_ssa_to_spirv(ssa).unwrap().program,
            ShaderFormat::Wgsl => lower_ssa_to_wgsl_with_program(ssa).unwrap().program,
        };
        let program = Program::parse(&host.to_whl("loop", format).unwrap()).unwrap();
        for length in [0, 1, 3, 65] {
            let mut backend = Trace::default();
            let input = backend.input(vec![0; length * 4]);
            let result = program.run("main", &[input], &mut backend).unwrap();
            assert_eq!(backend.buffers[&result.handle().unwrap()].len(), length * 4);
        }
    }
}

#[test]
fn computed_dispatch_extents_preserve_parameters_wrapping_and_grid_limits() {
    let source = include_str!("../../tests/compiler/library/computed_dispatch_extents_preserve_parameters_wrapping_and_grid_limits.wyn");
    let source = pipeline::source(source);
    let ssa = pipeline::lower(&source, CodegenTarget::Portable);
    for format in [ShaderFormat::Spirv, ShaderFormat::Wgsl] {
        let ssa = ssa.clone();
        let host = match format {
            ShaderFormat::Spirv => lower_ssa_to_spirv(ssa).unwrap().program,
            ShaderFormat::Wgsl => lower_ssa_to_wgsl_with_program(ssa).unwrap().program,
        };
        assert!(
            host.entries[0].allocations().next().is_none(),
            "scatter reuses its destination: {:?}",
            host.entries[0].allocations().collect::<Vec<_>>()
        );
        host.to_rust_wgpu("computed_dispatch", format).unwrap();
        let program = Program::parse(&host.to_whl("computed_dispatch", format).unwrap()).unwrap();
        for (rows, columns, groups) in [
            (0i32, 4i32, 0),
            (8, 8, 1),
            (5, 13, 2),
            (-1, 4, 0),
            (i32::MAX, 2, 0),
            (65_536, 64, 65_535),
        ] {
            let mut backend = Trace::default();
            let arguments: Vec<_> = program
                .entry("main")
                .unwrap()
                .parameters
                .iter()
                .map(|parameter| {
                    backend.input(match parameter.source_name() {
                        "dest" => vec![0; 128 * 4],
                        name => {
                            assert_eq!(parameter.host_bytes().unwrap(), Some(8), "{name}");
                            rows.to_le_bytes().into_iter().chain(columns.to_le_bytes()).collect()
                        }
                    })
                })
                .collect();
            program.run("main", &arguments, &mut backend).unwrap();
            let [(_, actual)] = backend.dispatches.as_slice() else {
                panic!("one scatter dispatch: {:?}", backend.dispatches);
            };
            assert_eq!(actual, &[groups, 1, 1], "{format:?}, shape=({rows},{columns})");
            assert!(backend.device_reads.is_empty(), "parameters stay host-readable");
        }
    }
}

#[test]
fn uses_input_buffer_capacity_and_preserves_returned_aliases() {
    let program = generated(include_str!(
        "../../tests/compiler/library/runtime_capacity_alias.wyn"
    ));
    let mut backend = Trace::default();
    let input = backend.input(vec![0; 524]);
    let result = program.run("main", &[input], &mut backend).unwrap();
    assert_eq!(backend.buffers[&result.handle().unwrap()].len(), 524);
    let input = backend.input(vec![0; 16]);
    assert_eq!(
        program.run("echo", &[input.clone()], &mut backend).unwrap(),
        input
    );
    let bytes: Vec<_> = (1i32..=4).flat_map(i32::to_le_bytes).collect();
    let input = backend.input(bytes.clone());
    let result = program.run("prefixes", &[input.clone()], &mut backend).unwrap();
    let results = result.list().unwrap();
    assert_eq!(results[0], input);
    assert_eq!(backend.buffers[&results[1].handle().unwrap()], bytes[..8]);
    assert!(backend.buffers.contains_key(&input.handle().unwrap()));
}

#[test]
fn multiple_callable_arguments_keep_distinct_captured_values() {
    let source = include_str!(
        "../../tests/compiler/library/multiple_callable_arguments_keep_distinct_captured_values.wyn"
    );
    let prepared = pipeline::source(source);
    let output = lower_ssa_to_wgsl_with_program(pipeline::lower(&prepared, CodegenTarget::Wgsl)).unwrap();
    assert!(output.program.entries[0]
        .operations
        .iter()
        .all(|op| matches!(op, Operation::Scalar { .. } | Operation::Allocate(_))));
    assert!(!output
        .program
        .to_rust_wgpu("scalar", ShaderFormat::Wgsl)
        .unwrap()
        .contains("dispatch_workgroups"));
    let program = Program::parse(&output.program.to_whl("scalar", ShaderFormat::Wgsl).unwrap()).unwrap();
    for (x, expected) in [(-3i32, 4), (5, 55)] {
        let arguments = [x, 7, 11].into_iter().flat_map(i32::to_le_bytes).collect();
        let mut backend = Trace::default();
        let input = backend.input(arguments);
        let result = program.run("main", &[input], &mut backend).unwrap();
        assert!(backend.dispatches.is_empty());
        let bytes = backend.buffers[&result.handle().unwrap()].clone();
        assert_eq!(i32::from_le_bytes(bytes.try_into().unwrap()), expected);
    }
}

#[test]
fn allocated_array_loop_output_keeps_logical_length() {
    let source = include_str!("../../testfiles/regressions/triangulation_allocated_array_loop.wyn");
    let program = generated(source);
    for length in [0i32, 1, 3, 65] {
        let mut backend = Trace::default();
        let inputs = program
            .entry("main")
            .unwrap()
            .parameters
            .iter()
            .map(|_| backend.input(length.to_le_bytes().to_vec()))
            .collect::<Vec<_>>();
        let result = program.run("main", &inputs, &mut backend).unwrap();
        assert_eq!(
            backend.buffers[&result.handle().unwrap()].len(),
            length as usize * 4
        );
    }
}

#[test]
fn interprets_emitted_size_calculation_before_allocating_and_dispatching() {
    let program = generated("entry main(n:i32) []i32 = iota(n*2+3)");
    let mut backend = Trace::default();
    let input = backend.input(7i32.to_le_bytes().to_vec());
    let result = program.run("main", &[input], &mut backend).unwrap();
    assert_eq!(backend.buffers[&result.handle().unwrap()].len(), 68);
    assert!(!backend.host_reads.is_empty());
    assert!(backend.device_reads.is_empty());
    assert_eq!(
        program.entry("main").unwrap().parameters[0].host_bytes().unwrap(),
        Some(4)
    );
    assert!(!backend.dispatches.is_empty());
}

#[test]
fn sliced_range_count_arithmetic_reaches_host_allocations() {
    let program = generated("entry main(n:i32, xs:[]i32) []i32 = iota(length(xs[(n+1)..(n*2+3)])*2+1)");
    let entry = program.entry("main").unwrap();
    assert_eq!(entry.parameters.len(), 2);
    for n in [0i32, 7, 33] {
        let mut backend = Trace::default();
        let arguments: Vec<_> = entry
            .parameters
            .iter()
            .map(|parameter| {
                backend.input(if parameter.source_name() == "xs" {
                    vec![0; 512]
                } else {
                    n.to_le_bytes().to_vec()
                })
            })
            .collect();
        let result = program.run("main", &arguments, &mut backend).unwrap();
        assert_eq!(
            backend.buffers[&result.handle().unwrap()].len(),
            (2 * n as usize + 5) * 4
        );
        assert!(!backend.dispatches.is_empty());
    }
}

#[test]
fn empty_maps_launch_no_work_despite_nonzero_storage_capacity() {
    let program = generated("entry main(xs:[]i32) []i32=map(|x|x+1,xs)");
    let mut backend = Trace::default();
    let input = backend.input(vec![]);
    let result = program.run("main", &[input], &mut backend).unwrap();
    assert!(backend.buffers[&result.handle().unwrap()].is_empty());
    assert!(backend.dispatches.iter().all(|(_, groups)| groups.contains(&0)));
}

#[test]
fn segmented_collectives_launch_per_row_without_row_scratch_allocations() {
    let source = "entry main(xs:[][513]i32) [][513]i32=map(|row|scan((+),0,row),xs)";
    let program = generated(source);
    for rows in [0, 1, 3, 129] {
        let mut backend = Trace::default();
        let input = backend.input(vec![0; rows * 513 * 4]);
        let result = program.run("main", &[input], &mut backend).unwrap();
        assert!(backend.buffers[&result.handle().unwrap()].len() >= rows * 513 * 4);
        assert_eq!(backend.dispatches.len(), 1);
        assert_eq!(backend.dispatches[0].1, [rows as u32, 1, 1]);
        assert_eq!(
            backend.buffers.len(),
            2,
            "only the input and final result are allocated"
        );
    }
}

#[test]
fn reduction_dispatches_parallel_stages_and_publishes_scalar_result_on_gpu() {
    let program = generated("entry sum(xs:[]i32) i32 = reduce(|a:i32,b:i32|a+b,0,xs)");
    let mut backend = Trace::default();
    let input = backend.input(vec![0; 137 * 4]);
    let result = program.run("sum", &[input], &mut backend).unwrap();
    assert_eq!(backend.dispatches.len(), 2);
    assert!(backend.scalar_writes.is_empty());
    assert!(!backend.freed.is_empty());
    assert_eq!(backend.buffers[&result.handle().unwrap()].len(), 4);
}

#[test]
fn reads_graphics_declarations_from_compiler_output() {
    let program = generated(include_str!("../../testfiles/texture_sample.wyn"));
    assert_eq!(program.graphics.len(), 1);
    let entry = program.entry("texture_sample_demo").unwrap();
    assert!(entry.parameters.iter().any(|p| p.source_name() == "tex" && p.kind == ":texture"));
    assert!(entry.results.iter().any(|p| p.source_name() == "screen"));
}

#[test]
fn generated_draw_buffers_supply_their_logical_counts() {
    #[derive(Default)]
    struct DrawTrace {
        trace: Trace,
        counts: Vec<u32>,
    }
    impl Backend for DrawTrace {
        fn call(&mut self, program: &Program, name: &str, args: &[Value]) -> Result<Value> {
            match name {
                "gpu-texture-view" => Ok(args[0].clone()),
                "gpu-draw" => {
                    let options = Options::parse(
                        &args[1..],
                        &[
                            ":args",
                            ":vertices",
                            ":colors",
                            ":depth",
                            ":viewport",
                            ":scissor",
                            ":draw",
                        ],
                    )?;
                    self.counts.push(options.get(":draw")?.list()?[3].u32()?);
                    Ok(Value::Nil)
                }
                _ => self.trace.call(program, name, args),
            }
        }
    }
    for (parameters, indices, expected_parameters, gpu_count) in [
        ("", "0u32..<3u32", 1, false),
        ("n:i32,", "0u32..<u32.i32(n)", 2, false),
        ("xs:[]u32,", "0u32..<reduce((+),0u32,xs)", 2, true),
    ] {
        let source = format!(
            "entry main({parameters}screen:render_target<vec4f32>) render_target<vec4f32> =
            let indices={indices} in
            let fragments=rasterize_triangles(indexed_draw(indices,1u32),
                |vi,_,_|vertex_output(@[f32.u32(vi),0.0,0.0,1.0],())) in
            shade(screen,fragments,|_,_,_,_,_|@[1.0,0.0,0.0,1.0])"
        );
        let source = pipeline::source(&source);
        let ssa = pipeline::lower(&source, CodegenTarget::Portable);
        for format in [ShaderFormat::Spirv, ShaderFormat::Wgsl] {
            let host = match format {
                ShaderFormat::Spirv => lower_ssa_to_spirv(ssa.clone()).unwrap().program,
                ShaderFormat::Wgsl => lower_ssa_to_wgsl_with_program(ssa.clone()).unwrap().program,
            };
            host.to_rust_wgpu("test", format).unwrap();
            let program = Program::parse(&host.to_whl("test", format).unwrap()).unwrap();
            let entry = program.entry("main").unwrap();
            assert_eq!(entry.parameters.len(), expected_parameters, "{indices}");
            for n in [0i32, 3, 17] {
                let mut backend = DrawTrace::default();
                let arguments = entry
                    .parameters
                    .iter()
                    .map(|parameter| {
                        let bytes = if parameter.kind == ":texture" {
                            vec![]
                        } else if parameter.source_name() == "xs" {
                            vec![0; 12]
                        } else {
                            n.to_le_bytes().to_vec()
                        };
                        backend.trace.input(bytes)
                    })
                    .collect::<Vec<_>>();
                program.run("main", &arguments, &mut backend).unwrap();
                assert_eq!(
                    backend.counts,
                    [if gpu_count {
                        0
                    } else if parameters.is_empty() {
                        3
                    } else {
                        n as u32
                    }]
                );
                assert_eq!(backend.trace.device_reads.is_empty(), !gpu_count);
            }
        }
    }
}

#[test]
fn spirv_dispatch_includes_its_scalar_push_constants() {
    let compiled =
        lower_ssa_to_spirv(source_ssa("entry main(n:i32) []i32 = iota(n*2+3)").unwrap()).unwrap();
    let source = compiled.program.to_whl("test.spv", ShaderFormat::Spirv).unwrap();
    let program = Program::parse(&source).unwrap();
    assert!(
        program.kernels.values().all(|kernel| kernel.parameters.iter().any(|p| p.kind == ":host-buffer")),
        "{source}"
    );
    let mut backend = Trace::default();
    let input = backend.input(7i32.to_le_bytes().to_vec());
    let result = program.run("main", &[input], &mut backend).unwrap();
    assert_eq!(backend.buffers[&result.handle().unwrap()].len(), 68);
}

#[test]
fn emitted_dispatch_loop_runs_setup_condition_and_iteration_writes() {
    let program = generated(
        "entry main(xs:[]i32) []i32 =
         loop xs=xs for i<2 do map(|x|x+i,xs)",
    );
    for count in [0, 4] {
        let mut backend = Trace::default();
        let input = backend.input(vec![0; count * 4]);
        program.run("main", &[input], &mut backend).unwrap();
        // Shape metadata also writes scalars; identify the loop counter by its second value.
        let counter =
            backend.scalar_writes.iter().find(|(_, bytes)| i32::from_le_bytes(*bytes) == 1).unwrap().0;
        let iterations: Vec<_> = backend
            .scalar_writes
            .iter()
            .filter(|(buffer, _)| *buffer == counter)
            .map(|(_, bytes)| i32::from_le_bytes(*bytes))
            .collect();
        assert_eq!(iterations, [0, 1]);
    }
}

#[test]
fn bounded_device_array_returns_its_live_range() {
    let program = generated(include_str!(
        "../../testfiles/regressions/narrowing_loop_capacity.wyn"
    ));
    let mut backend = Trace::default();
    let inputs = program
        .entry("main")
        .unwrap()
        .parameters
        .iter()
        .map(|parameter| backend.input(vec![0; parameter.minimum_bytes().unwrap().unwrap() as usize]))
        .collect::<Vec<_>>();
    let result = program.run("main", &inputs, &mut backend).unwrap();
    // The GPU test backend leaves the search result zero, despite capacity for 32 elements.
    assert!(backend.buffers[&result.handle().unwrap()].is_empty());
    assert_eq!(backend.device_reads.len(), 1, "{:?}", backend.device_reads);
}

#[test]
fn cpu_launch_sizes_preserve_division_errors_for_empty_maps() {
    let program = generated("entry main(n:i32) []i32=map(|i|i,iota(10/n))");
    let mut backend = Trace::default();
    let input = backend.input(0i32.to_le_bytes().to_vec());
    assert!(program.run("main", &[input], &mut backend).is_err());
    assert!(backend.dispatches.is_empty());
    let mut backend = Trace::default();
    let input = backend.input((-2i32).to_le_bytes().to_vec());
    let output = program.run("main", &[input], &mut backend).unwrap();
    assert!(backend.buffers[&output.handle().unwrap()].is_empty());
    assert!(backend.dispatches.iter().all(|(_, groups)| groups.contains(&0)));
}

#[test]
fn readbacks_are_shared_until_a_writer_and_unselected_reads_remain_lazy() {
    use wyn_host::{
        Access, Binding, BufferLen, BufferUsage, ComputePipeline, ComputeStage, DispatchSize, Expr,
        FrameGraph, ModuleInterface, Pipeline, ScalarExpr, ScalarSource, ScalarType, StageBindingUses,
    };
    let source = ScalarSource::Binding { set: 0, binding: 0 };
    let count = ScalarExpr::Read {
        source: source.clone(),
        offset: 0,
        ty: ScalarType::U32,
    };
    let mut bindings = vec![Binding::StorageBuffer {
        set: 0,
        binding: 0,
        name: "count".into(),
        access: Access::ReadWrite,
        usage: BufferUsage::Intermediate,
        resource: None,
        members: vec![],
        length: Some(BufferLen::Fixed { bytes: 4 }),
    }];
    for binding in 1..=2 {
        bindings.push(Binding::StorageBuffer {
            set: 0,
            binding,
            name: format!("data{binding}"),
            access: Access::WriteOnly,
            usage: BufferUsage::Intermediate,
            resource: None,
            members: vec![],
            length: Some(BufferLen::Computed {
                bytes: Expr::Scalar(count.clone()).multiply(Expr::Integer(4)),
            }),
        });
    }
    let mut stages = Vec::new();
    for stage in 0..4 {
        let writes_count = stage % 2 == 0;
        stages.push(ComputeStage {
            entry_point: format!("stage{stage}"),
            owner: "main".into(),
            dependencies: vec![],
            workgroup_size: (1, 1, 1),
            dispatch_size: if writes_count {
                DispatchSize::Fixed {
                    x: 1,
                    y: 1,
                    z: 1,
                    explicit: false,
                }
            } else {
                DispatchSize::Computed {
                    elements: Expr::Scalar(count.clone()),
                    groups: [
                        Expr::Scalar(count.clone()),
                        Expr::Integer(1),
                        Expr::Scalar(ScalarExpr::If {
                            condition: Box::new(ScalarExpr::Bool(false)),
                            yes: Box::new(ScalarExpr::Read {
                                source: source.clone(),
                                offset: 4,
                                ty: ScalarType::U32,
                            }),
                            no: Box::new(ScalarExpr::U32(1)),
                        }),
                    ],
                }
            },
            uses: StageBindingUses {
                reads: if writes_count { vec![] } else { vec![0] },
                writes: vec![if writes_count { 0 } else { (stage + 1) / 2 }],
            },
        });
    }
    let mut interface = ModuleInterface {
        pipelines: vec![Pipeline::Compute(ComputePipeline {
            bindings,
            stages,
            default_total_threads: None,
        })],
        ..ModuleInterface::default()
    };
    interface.frame_graph = FrameGraph::from_selected_pipelines(
        &interface.pipelines,
        &[((0, 0), (0, 1)), ((0, 1), (0, 2)), ((0, 2), (0, 3))],
    )
    .unwrap();
    let host = wyn_host::Program::new(interface).unwrap();
    let whl = host.to_whl("test", ShaderFormat::Wgsl).unwrap();
    let rebuilt = wyn_host::Program::new(host.interface.clone()).unwrap();
    assert_eq!(whl, rebuilt.to_whl("test", ShaderFormat::Wgsl).unwrap());
    assert_eq!(
        host.to_rust_wgpu("test", ShaderFormat::Wgsl).unwrap(),
        rebuilt.to_rust_wgpu("test", ShaderFormat::Wgsl).unwrap()
    );
    let program = Program::parse(&whl).unwrap();
    #[derive(Default)]
    struct ChangingCount {
        trace: Trace,
        allocations: Vec<u64>,
    }
    impl Backend for ChangingCount {
        fn call(&mut self, program: &Program, name: &str, args: &[Value]) -> Result<Value> {
            if name == "gpu-alloc" {
                self.allocations.push(args[0].u64()?);
            }
            if name == "gpu-dispatch" {
                let stage = args[0].text()?;
                if stage == "kernel-0-0" || stage == "kernel-0-2" {
                    let options = Options::parse(&args[1..], &[":groups", ":args"])?;
                    let buffer = options.get(":args")?.list()?[0].handle()?;
                    self.trace.buffers.insert(
                        buffer,
                        if stage == "kernel-0-0" { 1u32 } else { 2u32 }.to_le_bytes().to_vec(),
                    );
                }
            }
            self.trace.call(program, name, args)
        }
    }
    let mut backend = ChangingCount::default();
    program.run("main", &[], &mut backend).unwrap();
    assert_eq!(backend.allocations, [4, 4, 8]);
    assert_eq!(
        backend.trace.device_reads.len(),
        2,
        "one read per count generation"
    );
    assert_eq!(backend.trace.device_reads[0], backend.trace.device_reads[1]);
    assert_eq!(backend.trace.dispatches[1].1, [1, 1, 1]);
    assert_eq!(backend.trace.dispatches[3].1, [2, 1, 1]);
}

#[test]
fn shared_map_capture_is_computed_on_host() {
    let compiled = lower_ssa_to_wgsl_with_program(
        source_ssa("entry main(xs:[]i32,bias:i32) []i32=map(|x:i32|x+bias*bias,xs)").unwrap(),
    )
    .unwrap();
    let whl = compiled.program.to_whl("test.wgsl", ShaderFormat::Wgsl).unwrap();
    assert!(whl.contains("(wyn-i32-mul "), "{whl}");
    assert!(whl.find("gpu-write-scalar").unwrap() < whl.find("(gpu-dispatch ").unwrap());
    assert!(!compiled.wgsl.contains("_pc0.w_bias"), "{}", compiled.wgsl);
    let program = Program::parse(&whl).unwrap();
    let mut backend = Trace::default();
    let entry = program.entry("main").unwrap();
    let arguments: Vec<_> = entry
        .parameters
        .iter()
        .map(|p| {
            backend.input(if p.source_name() == "xs" { vec![0; 16] } else { 7i32.to_le_bytes().to_vec() })
        })
        .collect();
    program.run("main", &arguments, &mut backend).unwrap();
    assert!(!backend.dispatches.is_empty());
    assert!(backend.scalar_writes.iter().any(|(_, word)| i32::from_le_bytes(*word) == 49));
}

#[test]
fn spirv_host_captures_keep_push_constant_inputs() {
    let compiled = lower_ssa_to_spirv(
        source_ssa("entry main(xs:[]i32,bias:i32) []i32=map(|x:i32|x+bias*bias,xs)").unwrap(),
    )
    .unwrap();
    let whl = compiled.program.to_whl("test.spv", ShaderFormat::Spirv).unwrap();
    let program = Program::parse(&whl).unwrap();
    let mut backend = Trace::default();
    let arguments: Vec<_> = program
        .entry("main")
        .unwrap()
        .parameters
        .iter()
        .map(|p| {
            backend.input(if p.source_name() == "xs" { vec![0; 16] } else { 7i32.to_le_bytes().to_vec() })
        })
        .collect();
    program.run("main", &arguments, &mut backend).unwrap();
    assert!(backend.scalar_writes.iter().any(|(_, word)| i32::from_le_bytes(*word) == 49));
}

#[test]
fn gpu_scalar_capture_stays_between_reduction_and_map_without_readback() {
    let compiled = lower_ssa_to_wgsl_with_program(source_ssa(
        "entry main(xs:[137]i32) []i32=let total=reduce(|a:i32,b:i32|a+b,0,xs) in map(|x:i32|x+total*total,xs)"
    ).unwrap()).unwrap();
    let host = &compiled.program;
    let operations = &host.entries[0].operations;
    assert!(operations.iter().all(|op| matches!(op, Operation::Dispatch { .. } | Operation::Allocate(_))));
    let whl = host.to_whl("test.wgsl", ShaderFormat::Wgsl).unwrap();
    assert!(!whl.contains("gpu-read-scalar"), "{whl}");
    let rust = host.to_rust_wgpu("test.wgsl", ShaderFormat::Wgsl).unwrap();
    assert!(rust.contains("pub fn encode_main("), "{rust}");
    assert!(!rust.contains("read_gpu_word"), "{rust}");
}

#[test]
fn cpu_available_operations_can_cross_materialized_scalar_boundaries() {
    let compiled = source_ssa(
        "entry main(xs:[]i32,n:i32) []i32 =
        let a=loop acc=0 for i<n do acc+i in
        let b=loop acc=a for i<n do acc+i*2 in map(|x:i32|x+b,xs)",
    )
    .unwrap();
    let host = wyn_host::Program::new(compiled.global_context.pipeline).unwrap();
    let whl = host.to_whl("test.spv", ShaderFormat::Spirv).unwrap();
    let rust = host.to_rust_wgpu("test.spv", ShaderFormat::Spirv).unwrap();
    assert!(!rust.contains("read_gpu_word"), "{rust}");
    assert_eq!(
        host.entries[0].operations.iter().filter(|op| matches!(op, Operation::Dispatch { .. })).count(),
        1
    );
    let program = Program::parse(&whl).unwrap();
    let mut backend = Trace::default();
    let args = program
        .entry("main")
        .unwrap()
        .parameters
        .iter()
        .map(|p| {
            backend.input(if p.source_name() == "xs" { vec![0; 12] } else { 4i32.to_le_bytes().to_vec() })
        })
        .collect::<Vec<_>>();
    program.run("main", &args, &mut backend).unwrap();
    assert!(backend.scalar_writes.iter().any(|(_, word)| i32::from_le_bytes(*word) == 18));
}

#[test]
fn scalar_branches_and_loops_execute_on_the_cpu() {
    for (source, input, expected) in [
        (
            "entry main(n:i32) i32=if n>0 then n-1 else n+1",
            i32::MAX,
            i32::MAX - 1,
        ),
        ("entry main(n:i32) i32=n*n", i32::MAX, 1),
        ("entry main(n:i32) i32=loop acc=0 for i<n do acc+i", 4, 6),
        ("entry main(n:i32) i32=loop acc=0 while acc<n do acc+1", 4, 4),
        ("entry main(n:i32) i32=if n>0 then 10/n else 7", 0, 7),
        ("entry main(n:i32) i32=i32.get_bit(31,n)", i32::MIN, 1),
        ("entry main(n:i32) i32=i32.bool(n>0)", 4, 1),
        ("entry main(n:i32) i32=i32.u32(u32.i32(n))", -3, -3),
    ] {
        let program = generated(source);
        let mut backend = Trace::default();
        let input = backend.input(input.to_le_bytes().to_vec());
        let result = program.run("main", &[input], &mut backend).unwrap();
        assert_eq!(
            backend.buffers[&result.handle().unwrap()],
            expected.to_le_bytes(),
            "{source}"
        );
        assert!(backend.dispatches.is_empty(), "{source}");
        assert!(backend.device_reads.is_empty(), "{source}");
    }
}

fn source_ssa(source: &str) -> wyn_core::error::Result<wyn_core::ssa::stage::Elaborated> {
    let source = pipeline::source(source);
    Ok(pipeline::lower_with_options(
        &source,
        CodegenTarget::Portable,
        wyn_core::egglog::ScalarOptimization::Full,
        wyn_core::PipelineTopologyPolicy::AllowGenerated,
    ))
}

#[test]
fn authored_outputs_consume_selected_capacities() {
    for (source, input_sizes, scalar_count, expected) in [
        (
            include_str!("../../tests/compiler/library/direct_capacity_0.wyn"),
            vec![80],
            None,
            80,
        ),
        (
            include_str!("../../tests/compiler/library/direct_capacity_1.wyn"),
            vec![68],
            None,
            68,
        ),
        (
            include_str!("../../tests/compiler/library/direct_capacity_1.wyn"),
            vec![0],
            None,
            0,
        ),
        (
            include_str!("../../tests/compiler/library/direct_capacity_2.wyn"),
            vec![160, 68],
            None,
            68,
        ),
        (
            include_str!("../../tests/compiler/library/direct_capacity_3.wyn"),
            vec![],
            Some(17i32),
            68,
        ),
        (
            include_str!("../../tests/compiler/library/direct_capacity_3.wyn"),
            vec![],
            Some(0i32),
            0,
        ),
    ] {
        let source = pipeline::source(source);
        let ssa = pipeline::lower_with_options(
            &source,
            CodegenTarget::Wgsl,
            wyn_core::egglog::ScalarOptimization::Basic,
            wyn_core::PipelineTopologyPolicy::AuthoredOnly,
        );
        let output = lower_ssa_to_wgsl_with_program(ssa).unwrap();
        let shader = naga::front::wgsl::parse_str(&output.wgsl).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&shader)
        .unwrap();
        let program =
            Program::parse(&output.program.to_whl("capacity", ShaderFormat::Wgsl).unwrap()).unwrap();
        let mut backend = Trace::default();
        let inputs: Vec<_> = program
            .entry("main")
            .unwrap()
            .parameters
            .iter()
            .map(|parameter| {
                let bytes = match parameter.source_name() {
                    "xs" => vec![0; input_sizes[0]],
                    "ys" => vec![0; input_sizes[1]],
                    _ => scalar_count
                        .map(|count| count.to_le_bytes().to_vec())
                        .unwrap_or_else(|| vec![0; input_sizes[0]]),
                };
                backend.input(bytes)
            })
            .collect();
        let result = program.run("main", &inputs, &mut backend).unwrap();
        assert_eq!(backend.buffers[&result.handle().unwrap()].len(), expected);
    }
}

#[test]
fn dynamic_domains_launch_from_the_source_extent() {
    let source = pipeline::source(
        "entry buffer_domain(xs:[]i32) []i32=map(|x|x+1,xs)
        entry scalar_domain(n:i32) []i32=map(|i|i+1,iota(n))",
    );
    let output = lower_ssa_to_wgsl_with_program(pipeline::lower(&source, CodegenTarget::Wgsl)).unwrap();
    let program = Program::parse(&output.program.to_whl("domains", ShaderFormat::Wgsl).unwrap()).unwrap();
    for entry in &output.program.entries {
        let Operation::Dispatch { pipeline, stage, .. } =
            entry.operations.iter().find(|op| matches!(op, Operation::Dispatch { .. })).unwrap()
        else {
            unreachable!()
        };
        let wyn_host::Pipeline::Compute(compute) = &output.program.interface.pipelines[*pipeline] else {
            unreachable!()
        };
        let width = compute.stages[*stage].workgroup_size.0;
        for n in [0u32, 1, width, width + 1, width * 3 + 1] {
            let mut backend = Trace::default();
            let bytes = if entry.name == "scalar_domain" {
                (n as i32).to_le_bytes().to_vec()
            } else {
                vec![0; n as usize * 4]
            };
            let input = backend.input(bytes);
            let result = program.run(&entry.name, &[input], &mut backend).unwrap();
            assert_eq!(backend.buffers[&result.handle().unwrap()].len(), n as usize * 4);
            assert_eq!(backend.dispatches.len(), 1);
            assert_eq!(backend.dispatches[0].1, [n.div_ceil(width), 1, 1]);
        }
    }
}
