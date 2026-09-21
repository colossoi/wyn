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
    assert!(rust.contains("BigInt"));
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
    assert!(whl.contains("(mod "), "{whl}");
    assert!(whl.contains("4294967296"));
    let rust = program.to_rust_wgpu("sizes.wgsl", ShaderFormat::Wgsl).unwrap();
    assert!(rust.contains("modulo("));
    assert!(rust.contains("read_scalar(device"));
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
