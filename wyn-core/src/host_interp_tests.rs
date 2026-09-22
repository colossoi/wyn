use crate::{compile_thru_ssa, lower_ssa_to_spirv, lower_ssa_to_wgsl_with_program};
use wyn_host::ShaderFormat;
use wyn_host_interp::{Backend, Error, Number, Options, Program, Result, Value};

#[path = "../../wyn-host-interp/src/test_backend.rs"]
mod test_backend;
use test_backend::Trace;

fn generated(source: &str) -> Program {
    let compiled = lower_ssa_to_wgsl_with_program(compile_thru_ssa(source).unwrap()).unwrap();
    Program::parse(&compiled.program.to_whl("test.wgsl", ShaderFormat::Wgsl).unwrap()).unwrap()
}

#[test]
fn interprets_emitted_size_calculation_before_allocating_and_dispatching() {
    let program = generated("entry main(n:i32) []i32 = iota(n*2+3)");
    let mut backend = Trace::default();
    let input = backend.input(7i32.to_le_bytes().to_vec());
    let result = program.run("main", &[input], &mut backend).unwrap();
    assert_eq!(backend.buffers[&result.handle().unwrap()].len(), 68);
    assert!(!backend.dispatches.is_empty());
}

#[test]
fn uses_input_buffer_capacity_and_preserves_returned_aliases() {
    let program = generated("entry main(xs:[]i32) []i32 = map(|x:i32|x+1,xs)");
    let mut backend = Trace::default();
    let input = backend.input(vec![0; 524]);
    let result = program.run("main", &[input], &mut backend).unwrap();
    assert_eq!(backend.buffers[&result.handle().unwrap()].len(), 524);
    let program = generated("entry echo(xs:[]i32) []i32 = xs");
    let input = backend.input(vec![0; 16]);
    assert_eq!(
        program.run("echo", &[input.clone()], &mut backend).unwrap(),
        input
    );
}

#[test]
fn uniform_sized_launches_scale_with_capacity_and_clamp_the_grid() {
    struct LaunchTrace {
        trace: Trace,
        bytes: u64,
    }
    impl Backend for LaunchTrace {
        fn call(&mut self, program: &Program, name: &str, args: &[Value]) -> Result<Value> {
            match name {
                "gpu-buffer-size" => {
                    assert_eq!(args[0], Value::Resource(2), "map output capacity");
                    Ok(Value::Number(Number::U64(self.bytes)))
                }
                "gpu-texture-view" => Ok(args[0].clone()),
                "gpu-draw" => Ok(Value::Nil),
                _ => self.trace.call(program, name, args),
            }
        }
    }
    let source = include_str!("../../testfiles/rust_host_runtime_dispatch.wyn");
    for format in [ShaderFormat::Spirv, ShaderFormat::Wgsl] {
        let ssa = compile_thru_ssa(source).unwrap();
        let program = match format {
            ShaderFormat::Spirv => lower_ssa_to_spirv(ssa).unwrap().program,
            ShaderFormat::Wgsl => lower_ssa_to_wgsl_with_program(ssa).unwrap().program,
        };
        let program = Program::parse(&program.to_whl("runtime_dispatch", format).unwrap()).unwrap();
        for (elements, groups) in [
            (0, 1),
            (64, 1),
            (65, 2),
            (320 * 200, 1_000),
            (1280 * 800, 16_000),
            (65_535 * 64 + 1, 65_535),
        ] {
            let mut backend = LaunchTrace {
                trace: Trace::default(),
                bytes: elements * 16,
            };
            // Uniform contents are deliberately unavailable: launching from
            // capacity must not require scalar readback or extra host inputs.
            program
                .run(
                    "reproduce",
                    &[Value::Resource(1), Value::Resource(2), Value::Resource(3)],
                    &mut backend,
                )
                .unwrap();
            let [(_, actual)] = backend.trace.dispatches.as_slice() else {
                panic!("one parallel map");
            };
            assert_eq!(actual, &[groups, 1, 1], "{format:?}, capacity={elements}");
        }
    }
}

#[test]
fn reduction_dispatches_parallel_stages_and_copies_scalar_result_on_host() {
    let program = generated("entry sum(xs:[137]i32) i32 = reduce(|a:i32,b:i32|a+b,0,xs)");
    let mut backend = Trace::default();
    let input = backend.input(vec![0; 137 * 4]);
    let result = program.run("sum", &[input], &mut backend).unwrap();
    assert_eq!(backend.dispatches.len(), 2);
    assert_eq!(backend.scalar_writes.len(), 1);
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
        lower_ssa_to_spirv(compile_thru_ssa("entry main(n:i32) []i32 = iota(n*2+3)").unwrap()).unwrap();
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
