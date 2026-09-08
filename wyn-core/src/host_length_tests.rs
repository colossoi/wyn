use crate::pipeline_descriptor::{Binding, BufferLen, BufferUsage, DispatchSize, Pipeline};
const SOURCE: &str = include_str!("../../testfiles/regressions/uniform_output_size.wyn");

fn output_length(source: &str, serial: bool) -> BufferLen {
    let lowered =
        if serial { crate::compile_thru_spirv_serial(source) } else { crate::compile_thru_spirv(source) }
            .expect("compile reproducer");
    let compute = lowered
        .pipeline
        .pipelines
        .iter()
        .find_map(|p| match p {
            Pipeline::Compute(p) => Some(p),
            _ => None,
        })
        .unwrap();
    if source.contains("i32(frame.resolution.x)") {
        assert!(
            compute
                .stages
                .iter()
                .all(|s| matches!(s.dispatch_size, DispatchSize::Fixed { x: 1, y: 1, z: 1, .. })),
            "allocation must not require parallelizing the physical stage"
        );
    }
    let len = compute
        .bindings
        .iter()
        .find_map(|b| match b {
            Binding::StorageBuffer {
                usage: BufferUsage::Output,
                length,
                ..
            } => length.clone(),
            _ => None,
        })
        .expect("output length");
    // Exercise the actual public descriptor round trip.
    serde_json::from_str(&serde_json::to_string(&len).unwrap()).unwrap()
}

#[test]
fn uniform_output_capacity_is_independent_of_dispatch() {
    for serial in [false, true] {
        let length = output_length(SOURCE, serial);
        assert!(matches!(length, BufferLen::HostExpression { elem_bytes: 4, .. }));
        for (w, h) in [
            (64.0f32, 32.0f32),
            (17.9, 9.8),
            (1.0, 1.0),
            (0.0, 32.0),
            (128.0, 65.0),
        ] {
            assert_eq!(
                length
                    .resolve_host_bytes(&|set, binding, offset| match (set, binding, offset) {
                        (0, 0, 0) => Some(w.to_bits()),
                        (0, 0, 4) => Some(h.to_bits()),
                        _ => None,
                    })
                    .unwrap(),
                4 * (w as i32 as u64) * (h as i32 as u64)
            );
        }
        assert!(length.resolve_host_bytes(&|_, _, _| None).is_err());
        for (w, h) in [
            (-1.0f32, 2.0f32),
            (f32::NAN, 2.0),
            (f32::INFINITY, 2.0),
            (65536.0, 65536.0),
        ] {
            assert!(length
                .resolve_host_bytes(&|_, _, offset| Some(if offset == 0 {
                    w.to_bits()
                } else {
                    h.to_bits()
                }))
                .is_err());
        }
    }
}

#[test]
fn fixed_output_capacity_control() {
    let source =
        SOURCE.replace("i32(frame.resolution.x)", "64i32").replace("i32(frame.resolution.y)", "32i32");
    let lowered = crate::compile_thru_spirv(&source).unwrap();
    let compute = lowered
        .pipeline
        .pipelines
        .iter()
        .find_map(|p| match p {
            Pipeline::Compute(p) => Some(p),
            _ => None,
        })
        .unwrap();
    assert!(
        compute.stages.iter().any(|s| matches!(
            s.dispatch_size,
            DispatchSize::DerivedFrom {
                len: crate::pipeline_descriptor::DispatchLen::Fixed { count: 2048 },
                ..
            }
        )),
        "constant control retains its fixed logical domain"
    );
    let output = compute
        .bindings
        .iter()
        .find_map(|b| match b {
            Binding::StorageBuffer {
                usage: BufferUsage::Output,
                length,
                ..
            } => length.as_ref(),
            _ => None,
        })
        .unwrap();
    assert_eq!(
        output.dispatch_elem_bytes().map(|stride| u64::from(stride) * 2048),
        Some(8192)
    );
}

#[test]
fn uniform_output_capacity_uses_abi_offsets_and_arithmetic() {
    let source =
        SOURCE.replace("{ resolution: vec3f32 }", "{ padding: f32, resolution: vec3f32 }").replace(
            "width * height",
            "((width + 7i32) / 8i32) * ((height + 7i32) / 8i32)",
        );
    let length = output_length(&source, false);
    assert_eq!(
        length
            .resolve_host_bytes(&|_, _, offset| match offset {
                16 => Some(65.0f32.to_bits()),
                20 => Some(33.0f32.to_bits()),
                _ => None,
            })
            .unwrap(),
        4 * 9 * 5
    );
}

#[test]
fn unsupported_output_length_has_an_allocation_diagnostic() {
    let source = SOURCE.replace("width * height", "i32(target_load(rendered, @[0i32,0i32],0u32))");
    let error =
        crate::compile_thru_spirv(&source).err().expect("unsupported length diagnostic").to_string();
    assert!(
        error.contains("logical length") && error.contains("host uniforms"),
        "{error}"
    );
}

#[test]
fn uniform_output_capacity_respects_storage_stride_and_wgsl() {
    let source = SOURCE.replace("([]f32,", "([]vec3f32,").replace(
        "target_load(rendered, @[i % width, i / width], 0u32)",
        "@[f32(i), 0.0, 0.0]",
    );
    let length = output_length(&source, false);
    assert!(matches!(length, BufferLen::HostExpression { elem_bytes: 16, .. }));
    assert_eq!(
        length
            .resolve_host_bytes(&|_, _, offset| Some(if offset == 0 {
                17.0f32.to_bits()
            } else {
                9.0f32.to_bits()
            }))
            .unwrap(),
        16 * 17 * 9
    );
    let lowered = crate::lower_ssa_to_wgsl_with_pipeline(crate::compile_thru_ssa(SOURCE).unwrap()).unwrap();
    assert!(lowered.pipeline.pipelines.iter().any(|p| match p {
        Pipeline::Compute(p) => p.bindings.iter().any(|b| matches!(
            b,
            Binding::StorageBuffer {
                length: Some(BufferLen::HostExpression { elem_bytes: 4, .. }),
                ..
            }
        )),
        _ => false,
    }));
}
