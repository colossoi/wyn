use crate::host::{self, Operation, ShaderFormat};
use crate::{compile_thru_ssa, lower_ssa_to_spirv, lower_ssa_to_wgsl_with_program};
use wyn_host_interp::{Backend, Error, Number, Options, Program, Result, Value};

#[path = "../../wyn-host-interp/src/test_backend.rs"]
mod test_backend;
use test_backend::Trace;

fn scalar(source: &str, arguments: &[Vec<u8>]) -> Vec<u8> {
    let compiled = lower_ssa_to_wgsl_with_program(compile_thru_ssa(source).unwrap()).unwrap();
    let whl = compiled.program.to_whl("test.wgsl", ShaderFormat::Wgsl).unwrap();
    assert!(
        compiled.program.entries[0].operations.iter().all(|op| matches!(op, Operation::Scalar { .. })),
        "{whl}"
    );
    let rust = compiled.program.to_rust_wgpu("test.wgsl", ShaderFormat::Wgsl).unwrap();
    assert!(!rust.contains("dispatch_workgroups"), "{rust}");
    let program = Program::parse(&whl).unwrap();
    let mut backend = Trace::default();
    let arguments: Vec<_> = arguments.iter().map(|b| backend.input(b.clone())).collect();
    let result = program.run("main", &arguments, &mut backend).unwrap();
    assert!(backend.dispatches.is_empty());
    backend.buffers[&result.handle().unwrap()].clone()
}

#[test]
fn sequential_integer_arithmetic_runs_on_host() {
    let bytes = scalar("entry main(n:i32) i32=n*n+1", &[7i32.to_le_bytes().to_vec()]);
    assert_eq!(i32::from_le_bytes(bytes.try_into().unwrap()), 50);
}

#[test]
fn sequential_float_intrinsics_run_on_host() {
    let bytes = scalar(
        "entry main(x:f32) f32=f32.sin(x)+x*x",
        &[0.5f32.to_le_bytes().to_vec()],
    );
    assert_eq!(f32::from_le_bytes(bytes.try_into().unwrap()), 0.5f32.sin() + 0.25);
}

#[test]
fn host_branches_do_not_evaluate_untaken_division() {
    let bytes = scalar(
        "entry main(n:i32) i32=if n==0 then 7 else 12/n",
        &[0i32.to_le_bytes().to_vec()],
    );
    assert_eq!(i32::from_le_bytes(bytes.try_into().unwrap()), 7);
}

#[test]
fn host_counted_loop_preserves_accumulator_and_zero_trips() {
    for n in [0i32, 5] {
        let bytes = scalar(
            "entry main(n:i32) i32=loop acc=0 for i<n do acc+i*i",
            &[n.to_le_bytes().to_vec()],
        );
        assert_eq!(
            i32::from_le_bytes(bytes.try_into().unwrap()),
            (0..n).map(|i| i * i).sum::<i32>()
        );
    }
}

#[test]
fn shared_map_capture_is_computed_on_host() {
    let compiled = lower_ssa_to_wgsl_with_program(
        compile_thru_ssa("entry main(xs:[]i32,bias:i32) []i32=map(|x:i32|x+bias*bias,xs)").unwrap(),
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
fn host_while_loop_and_nested_scalar_calls_preserve_scope() {
    let bytes = scalar(
        "def twice(x:i32) i32=x+x entry main(n:i32) i32=loop acc=0 while acc<n do twice(acc)+1",
        &[10i32.to_le_bytes().to_vec()],
    );
    assert_eq!(i32::from_le_bytes(bytes.try_into().unwrap()), 15);
}

#[test]
fn host_float_results_preserve_infinity_and_nan() {
    for (input, nan) in [(0.0f32, false), (-1.0f32, true)] {
        let bytes = scalar(
            "entry main(x:f32) f32=if x==0.0 then 1.0/x else f32.sqrt(x)",
            &[input.to_le_bytes().to_vec()],
        );
        let value = f32::from_le_bytes(bytes.try_into().unwrap());
        assert_eq!(value.is_nan(), nan);
        assert!(value.is_nan() || value.is_infinite());
    }
}

#[test]
fn host_comparison_preserves_ordered_float_inequality() {
    let bytes = scalar(
        "entry main(x:f32) i32=if x!=0.0 then 1 else 0",
        &[f32::NAN.to_le_bytes().to_vec()],
    );
    assert_eq!(i32::from_le_bytes(bytes.try_into().unwrap()), 0);
}

#[test]
fn host_calls_bind_each_invocation_separately() {
    let bytes = scalar(
        "def square(n:i32) i32=n*n entry main(n:i32) i32=square(n)+square(n+1)",
        &[3i32.to_le_bytes().to_vec()],
    );
    assert_eq!(i32::from_le_bytes(bytes.try_into().unwrap()), 25);
}

#[test]
fn host_reads_packed_parameters_and_preserves_unsigned_comparisons() {
    let bytes = scalar(
        "entry main(a:u32,b:u32) u32=if a>b then a-b else b-a",
        &[u32::MAX.to_le_bytes().into_iter().chain(1u32.to_le_bytes()).collect()],
    );
    assert_eq!(u32::from_le_bytes(bytes.try_into().unwrap()), u32::MAX - 1);
}

#[test]
fn host_eager_select_arms_preserve_native_arithmetic_at_boundaries() {
    for (expression, input, expected) in [
        ("if n>0 then n-1 else n+1", i32::MAX, i32::MAX - 1),
        ("if n>0 then n-1 else n*n", i32::MAX, i32::MAX - 1),
        ("if n<0 then n+1 else n-1", i32::MIN, i32::MIN + 1),
        ("n+1", i32::MAX, i32::MIN),
        ("n*n", i32::MAX, 1),
    ] {
        let bytes = scalar(
            &format!("entry main(n:i32) i32={expression}"),
            &[input.to_le_bytes().to_vec()],
        );
        assert_eq!(i32::from_le_bytes(bytes.try_into().unwrap()), expected);
    }
    for (expression, input, expected) in [
        ("if n>0.0 then n else n+n", f32::MAX, f32::MAX),
        ("n+n", f32::MAX, f32::INFINITY),
        ("n*n", f32::INFINITY, f32::INFINITY),
    ] {
        let bytes = scalar(
            &format!("entry main(n:f32) f32={expression}"),
            &[input.to_le_bytes().to_vec()],
        );
        assert_eq!(f32::from_le_bytes(bytes.try_into().unwrap()), expected);
    }
    let compiled = lower_ssa_to_wgsl_with_program(
        compile_thru_ssa("entry main(n:i32) i32=if n>0 then n-1 else n*n").unwrap(),
    )
    .unwrap();
    let rust = compiled.program.to_rust_wgpu("test.wgsl", ShaderFormat::Wgsl).unwrap();
    assert!(rust.contains("wrapping_mul"), "{rust}");
    assert!(rust.contains("wrapping_sub"), "{rust}");
}

#[test]
fn spirv_host_captures_keep_push_constant_inputs() {
    let compiled = lower_ssa_to_spirv(
        compile_thru_ssa("entry main(xs:[]i32,bias:i32) []i32=map(|x:i32|x+bias*bias,xs)").unwrap(),
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
    let compiled = lower_ssa_to_wgsl_with_program(compile_thru_ssa(
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
    let compiled = compile_thru_ssa(
        "entry main(xs:[]i32,n:i32) []i32 =
        let a=loop acc=0 for i<n do acc+i in
        let b=loop acc=a for i<n do acc+i*2 in map(|x:i32|x+b,xs)",
    )
    .unwrap();
    let host = host::Program::new(compiled.global_context.pipeline).unwrap();
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
fn host_boolean_inputs_and_outputs_use_scalar_words() {
    let bytes = scalar("entry main(n:i32) bool=n>2", &[3i32.to_le_bytes().to_vec()]);
    assert_eq!(u32::from_le_bytes(bytes.try_into().unwrap()), 1);
    let bytes = scalar(
        "entry main(enabled:bool,n:i32) i32=if enabled then n*n else n",
        &[1u32.to_le_bytes().into_iter().chain(3i32.to_le_bytes()).collect()],
    );
    assert_eq!(i32::from_le_bytes(bytes.try_into().unwrap()), 9);
}

#[test]
fn external_calls_keep_their_scalar_stage_on_device() {
    let compiled = compile_thru_ssa(
        "#[linked(\"observe\")] extern observe(n:i32) i32
         def f(n:i32) i32=observe(n)+1
         entry main(n:i32) i32=f(n)",
    )
    .unwrap();
    let program = host::Program::new(compiled.global_context.pipeline).unwrap();
    assert!(program.entries[0].operations.iter().any(|op| matches!(op, Operation::Dispatch { .. })));
}
