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
        for (elements, groups) in [
            (0, 0),
            (64, 1),
            (65, 2),
            (64_000, 1_000),
            (1_024_000, 16_000),
            (65_535 * 64 + 1, 65_535),
        ] {
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
fn branch_output_shape_is_read_before_temporary_storage_is_freed() {
    for source in [
        include_str!("../../testfiles/regressions/triangulation_runtime_array_branch.wyn"),
        include_str!("../../testfiles/regressions/triangulation_runtime_array_zip_map.wyn"),
    ] {
        let program = generated(source);
        let mut backend = Trace::default();
        let input = backend.input(vec![0; 48]);
        let result = program.run("main", &[input], &mut backend).unwrap();
        // Trace leaves GPU-written words zero: exercise the empty logical result.
        assert!(backend.buffers[&result.handle().unwrap()].is_empty());
        assert_eq!(backend.device_reads.len(), 1);
        assert!(backend.freed.contains(&backend.device_reads[0].0));
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
            host.entries[0].allocations.is_empty(),
            "scatter reuses its destination"
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
}

#[test]
fn multiple_callable_arguments_keep_distinct_captured_values() {
    let source = include_str!(
        "../../tests/compiler/library/multiple_callable_arguments_keep_distinct_captured_values.wyn"
    );
    let prepared = pipeline::source(source);
    let output = lower_ssa_to_wgsl_with_program(pipeline::lower(&prepared, CodegenTarget::Wgsl)).unwrap();
    assert!(output.program.entries[0].operations.iter().all(|op| matches!(op, Operation::Scalar { .. })));
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
    let program = generated("entry sum(xs:[137]i32) i32 = reduce(|a:i32,b:i32|a+b,0,xs)");
    let mut backend = Trace::default();
    let input = backend.input(vec![0; 137 * 4]);
    let result = program.run("sum", &[input], &mut backend).unwrap();
    assert_eq!(backend.dispatches.len(), 3);
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
    assert_eq!(backend.device_reads.len(), 1);
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
    assert!(operations.iter().all(|op| matches!(op, Operation::Dispatch { .. })));
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

fn source_ssa(source: &str) -> wyn_core::error::Result<wyn_core::ssa::stage::Elaborated> {
    let source = pipeline::source(source);
    Ok(pipeline::lower_with_options(
        &source,
        CodegenTarget::Portable,
        wyn_core::egglog::ScalarOptimization::Full,
        wyn_core::PipelineTopologyPolicy::AllowGenerated,
    ))
}
