use crate::host::arithmetic::{
    add, ceiling, dimension, floor, modulo, multiply, signed_size, size, subtract,
};
use crate::host::{Allocation, Operation, Pipeline, Program, ShaderFormat};
use crate::{compile_thru_ssa, lower_ssa_to_wgsl_with_program};

fn compile(source: &str) -> Program {
    lower_ssa_to_wgsl_with_program(compile_thru_ssa(source).unwrap()).unwrap().program
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
fn host_program_preserves_compute_phases_and_returns_source_results() {
    let program = compile("entry main(xs: []i32) []i32 = scan(|a:i32,b:i32|a+b,0,xs)");
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
    assert_eq!(entry.operations.len(), stages);
    assert_eq!(entry.results.len(), 1);
    assert!(entry.operations.iter().all(|o| matches!(o, Operation::Dispatch { .. })));
    let whl = program.to_whl("scan.wgsl", ShaderFormat::Wgsl).unwrap();
    check_whl(&whl);
    assert_eq!(whl.matches("(gpu-dispatch ").count(), stages);
    assert!(whl.contains("(gpu-alloc "));
    assert!(whl.contains("(gpu-free "));
    assert!(whl.contains(":source-name \"main\""));
}

#[test]
fn both_host_outputs_include_buffer_capacity_arithmetic() {
    let program = compile("entry main(xs: []i32) []i32 = map(|x:i32|x+1,xs)");
    let whl = program.to_whl("map.wgsl", ShaderFormat::Wgsl).unwrap();
    check_whl(&whl);
    assert!(whl.contains("(gpu-buffer-size "));
    assert!(whl.contains("(floor "));
    let rust = program.to_rust_wgpu("map.wgsl", ShaderFormat::Wgsl).unwrap();
    assert!(rust.contains("create_compute_pipeline"));
    assert!(rust.contains("dispatch_workgroups"));
    assert!(rust.contains("floor("));
    assert!(rust.contains("i64"));
    assert!(rust.contains("u64"));
    for dependency in ["BigInt", "num_bigint", "num_integer", "num_traits"] {
        assert!(!rust.contains(dependency), "unexpected dependency: {dependency}");
    }
    assert!(!rust.contains("todo!"));
}

#[test]
fn integer_capacity_expressions_reach_both_hosts() {
    let program = compile("entry main(n:i32) []i32 = iota(n*2+3)");
    let [entry] = program.entries.as_slice() else {
        panic!("one host entry");
    };
    assert!(entry
        .allocations
        .iter()
        .any(|a| matches!(a,Allocation::Buffer{bytes,..} if bytes.to_whl().contains("gpu-read-scalar"))));
    let whl = program.to_whl("sizes.wgsl", ShaderFormat::Wgsl).unwrap();
    check_whl(&whl);
    assert!(whl.contains("(i32-add "), "{whl}");
    assert!(whl.contains("(i32-mul "), "{whl}");
    assert!(!whl.contains("4294967296"));
    let rust = program.to_rust_wgpu("sizes.wgsl", ShaderFormat::Wgsl).unwrap();
    assert!(rust.contains(".wrapping_add("));
    assert!(rust.contains(".wrapping_mul("));
    assert!(rust.contains("read_scalar(device"));
}

#[test]
fn fixed_width_host_arithmetic_checks_overflow_and_keeps_large_byte_sizes() {
    let bytes = multiply(i64::from(u32::MAX), 16).unwrap();
    assert_eq!(size(bytes).unwrap(), 68_719_476_720);
    assert_eq!(signed_size(size(bytes).unwrap()).unwrap(), bytes);
    assert!(dimension(bytes).is_err());
    assert!(size(-1).is_err());
    assert!(signed_size(u64::MAX).is_err());
    assert!(add(i64::MAX, 1).is_err());
    assert!(subtract(i64::MIN, 1).is_err());
    assert!(multiply(i64::MAX, 2).is_err());
    assert_eq!(dimension(i64::from(u32::MAX)).unwrap(), u32::MAX);
}

#[test]
fn fixed_width_division_rounds_without_overflowing_intermediates() {
    for (a, b, down, up, remainder) in [
        (5, 2, 2, 3, 1),
        (-5, 2, -3, -2, 1),
        (5, -2, -3, -2, -1),
        (-5, -2, 2, 3, -1),
    ] {
        assert_eq!(floor(a, b).unwrap(), down);
        assert_eq!(ceiling(a, b).unwrap(), up);
        assert_eq!(modulo(a, b).unwrap(), remainder);
    }
    assert_eq!(ceiling(i64::MAX, 2).unwrap(), (i64::MAX / 2) + 1);
    assert_eq!(modulo(i64::MIN, i64::MAX).unwrap(), i64::MAX - 1);
    for (a, b) in [(1, 0), (i64::MIN, -1)] {
        assert!(floor(a, b).is_err());
        assert!(ceiling(a, b).is_err());
        assert!(modulo(a, b).is_err());
    }
}

#[test]
fn graphics_capture_uses_the_produced_buffer_after_its_writer() {
    let program = compile(include_str!("../../testfiles/playground/conway.wyn"));
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
}

#[test]
fn whl_paths_escape_lisp_strings_and_rust_requires_wgsl() {
    let program = compile("entry main() i32 = 1");
    let whl = program.to_whl("a\\b\"c.wgsl", ShaderFormat::Wgsl).unwrap();
    check_whl(&whl);
    assert!(whl.contains("a\\\\b\\\"c.wgsl"));
    assert!(program.to_rust_wgpu("shader.spv", ShaderFormat::Spirv).is_err());
}
