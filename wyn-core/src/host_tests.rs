use crate::host::arithmetic::{
    add, ceiling, dimension, floor, modulo, multiply, signed_size, size, subtract,
};
use crate::host::readback;
use crate::host::{Allocation, Operation, Pipeline, Program, ResultLayout, ResultScalar, ShaderFormat};
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
    assert!(rust.contains("support::read_i32(device"));
    assert!(!rust.contains("read_host_scalar"));
}

#[test]
fn named_result_readers_preserve_native_scalar_types() {
    for (ty, literal, scalar) in [
        ("i32", "-7", ResultScalar::I32),
        ("u32", "4294967295u32", ResultScalar::U32),
        ("f32", "2.5", ResultScalar::F32),
    ] {
        let program = compile(&format!("entry frobnicator() {ty} = {literal}"));
        assert_eq!(program.interface.source_results[0].name, "frobnicator");
        assert_eq!(
            program.interface.source_results[0].layout,
            ResultLayout::Scalar(scalar)
        );
        let rust = program.to_rust_wgpu("frobnicator.wgsl", ShaderFormat::Wgsl).unwrap();
        assert!(rust.contains("pub fn read_frobnicator("), "{rust}");
        assert!(rust.contains(&format!("-> Result<{ty}, HostError>")), "{rust}");
        assert!(rust.contains("-> Result<FrobnicatorOutput, HostError>"));
        assert!(rust.contains("pub use support::HostError;"));
        assert!(!rust.contains("pub mod support"));
        assert!(!rust.contains("enum Resource"));
        assert!(!rust.contains("read_scalar"));
    }
}

#[test]
fn record_results_are_read_together_with_authored_field_names() {
    let program = compile("entry main() {frobnicator:i32, gain:f32} = {frobnicator=7, gain=2.5}");
    let results = &program.interface.source_results;
    assert_eq!(
        results.iter().map(|r| r.name.as_str()).collect::<Vec<_>>(),
        ["frobnicator", "gain"]
    );
    let rust = program.to_rust_wgpu("record.wgsl", ShaderFormat::Wgsl).unwrap();
    assert!(rust.contains("pub fn read_main("), "{rust}");
    assert!(!rust.contains("pub fn read_frobnicator("));
    assert!(!rust.contains("pub fn read_gain("));
    assert_eq!(rust.matches("support::read_buffers(").count(), 1);
    assert!(rust.contains("pub frobnicator: i32"));
    assert!(rust.contains("pub gain: f32"));
    let whl = program.to_whl("record.wgsl", ShaderFormat::Wgsl).unwrap();
    assert!(whl.contains(":source-name \"frobnicator\""));
    assert!(whl.contains(":value-layout :i32"));
    check_whl(&whl);
}

#[test]
fn array_readers_use_element_ranges_and_padded_vector_stride() {
    let program =
        compile("entry frobnicator(xs: []vec3f32) []vec3f32 = map(|x:vec3f32| x + @[1.0,2.0,3.0],xs)");
    let ResultLayout::Array { element, stride, .. } = &program.interface.source_results[0].layout else {
        panic!("array layout");
    };
    assert_eq!(*stride, 16);
    assert!(matches!(
        element.as_ref(),
        ResultLayout::Sequence {
            count: 3,
            stride: 4,
            ..
        }
    ));
    let rust = program.to_rust_wgpu("vectors.wgsl", ShaderFormat::Wgsl).unwrap();
    assert!(rust.contains("Result<Vec<[f32; 3usize]>, HostError>"), "{rust}");
    assert!(rust.contains("frobnicator_elements: std::ops::Range<u32>"));
    let compact: String = rust.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(compact.contains("element_range(frobnicator_elements.clone(),16u32,output.frobnicator.size()"));
}

#[test]
fn tuple_results_use_one_reader_and_one_gpu_round_trip() {
    let program = compile("entry frobnicator() (i32,u32,f32) = (-7,4294967295u32,2.5)");
    let rust = program.to_rust_wgpu("tuple.wgsl", ShaderFormat::Wgsl).unwrap();
    assert!(rust.contains("-> Result<(i32, u32, f32), HostError>"), "{rust}");
    assert_eq!(rust.matches("pub fn read_frobnicator(").count(), 1);
    assert_eq!(rust.matches("support::read_buffers(").count(), 1);
    assert_eq!(rust.matches("queue.submit(Some(encoder.finish()))").count(), 2); // dispatch plus batched readback
    assert_eq!(rust.matches(".map_async(").count(), 1);
    assert_eq!(rust.matches(".poll(").count(), 1);
    assert!(!rust.contains("pub fn read_result_"));
}

#[test]
fn single_field_records_remain_records_in_the_result_api() {
    let program = compile("entry frobnicator() {value:i32} = {value=7}");
    let rust = program.to_rust_wgpu("single.wgsl", ShaderFormat::Wgsl).unwrap();
    assert!(rust.contains("pub struct Frobnicator {"), "{rust}");
    assert!(rust.contains("pub value: i32"));
    assert!(rust.contains("-> Result<Frobnicator, HostError>"));
}

#[test]
fn batch_readback_packs_exact_spans_into_one_aligned_allocation() {
    let (copies, size) =
        readback::copy_ranges(&[(2..6, 8), (12..12, 16), (16..28, 32), (0..4, 4)]).unwrap();
    assert_eq!(size, 24);
    assert_eq!(copies[0].source, 0..8);
    assert_eq!(copies[0].mapped, 2..6);
    assert_eq!(copies[1].mapped, 8..8);
    assert!(copies[1].source.is_empty());
    assert_eq!(copies[2].staging_offset, 8);
    assert_eq!(copies[2].source, 16..28);
    assert_eq!(copies[2].mapped, 8..20);
    assert_eq!(copies[3].mapped, 20..24);
    assert_eq!(readback::copy_ranges(&[(0..0, 0), (4..4, 4)]).unwrap().1, 0);
    assert!(readback::copy_ranges(&[(3..2, 8)]).is_err());
    assert!(readback::copy_ranges(&[(0..3, 3)]).is_err());
    assert!(readback::copy_ranges(&[(0..u64::MAX, u64::MAX)]).is_err());
    assert!(readback::copy_ranges(&[(0..u64::MAX - 3, u64::MAX), (0..4, 4)]).is_err());
    let high = u64::MAX - 7;
    assert_eq!(
        readback::copy_ranges(&[(0..8, 8), (high..high + 4, u64::MAX)]).unwrap().0[1].mapped,
        8..12
    );
}

#[test]
fn result_decoding_preserves_signed_unsigned_and_float_values() {
    let mut data = Vec::new();
    data.extend((-7i32).to_le_bytes());
    data.extend(u32::MAX.to_le_bytes());
    data.extend(2.5f32.to_le_bytes());
    assert_eq!(i32::from_le_bytes(readback::bytes(&data, 0).unwrap()), -7);
    assert_eq!(u32::from_le_bytes(readback::bytes(&data, 4).unwrap()), u32::MAX);
    assert_eq!(f32::from_le_bytes(readback::bytes(&data, 8).unwrap()), 2.5);
    assert!(readback::bytes::<4>(&data, 10).is_err());
    assert!(readback::bytes::<4>(&data, u64::MAX).is_err());
    assert!(readback::at(u64::MAX, 1).is_err());
}

#[test]
fn result_array_decoding_excludes_padding_and_unused_capacity() {
    let data: Vec<_> = [1.0f32, 2.0, 3.0, 99.0, 4.0, 5.0, 6.0, 99.0, 7.0, 8.0, 9.0, 99.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let range = readback::element_range(1..2, 16, data.len() as u64).unwrap();
    let values = readback::array(
        &data[range.start as usize..range.end as usize],
        1,
        16,
        |bytes, offset| {
            Ok([
                f32::from_le_bytes(readback::bytes(bytes, offset)?),
                f32::from_le_bytes(readback::bytes(bytes, readback::at(offset, 4)?)?),
                f32::from_le_bytes(readback::bytes(bytes, readback::at(offset, 8)?)?),
            ])
        },
    )
    .unwrap();
    assert_eq!(values, [[4.0, 5.0, 6.0]]);
    assert!(
        readback::array::<u32>(&[], 0, 4, |_, _| panic!("empty array is not decoded")).unwrap().is_empty()
    );
    assert!(readback::element_range(2..1, 4, 16).is_err());
    assert!(readback::element_range(0..5, 4, 16).is_err());
    assert!(readback::element_range(0..1, 0, 16).is_err());
    assert_eq!(
        readback::element_range(u32::MAX - 1..u32::MAX, 16, u64::MAX).unwrap().end,
        u64::from(u32::MAX) * 16
    );
    assert!(readback::array::<u32>(&[0; 3], 1, 4, |_, _| panic!("short buffer is rejected")).is_err());
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
