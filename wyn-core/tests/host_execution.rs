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
