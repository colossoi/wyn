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
