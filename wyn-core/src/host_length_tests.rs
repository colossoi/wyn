use crate::pipeline_descriptor::{
    Binding, BufferLen, BufferUsage, DispatchSize, HostSizeInput, HostSizeScalar, Pipeline,
};

const SOURCE: &str = include_str!("../../testfiles/regressions/uniform_output_size.wyn");

fn output_length(source: &str, serial: bool) -> BufferLen {
    let lowered =
        if serial { crate::compile_thru_spirv_serial(source) } else { crate::compile_thru_spirv(source) }
            .expect("compile reproducer");
    let compute = lowered
        .pipeline
        .pipelines
        .iter()
        .find_map(|pipeline| match pipeline {
            Pipeline::Compute(pipeline) => Some(pipeline),
            _ => None,
        })
        .unwrap();
    if source.contains("i32(frame.resolution.x)") {
        assert!(
            compute
                .stages
                .iter()
                .all(|stage| matches!(stage.dispatch_size, DispatchSize::Fixed { x: 1, y: 1, z: 1, .. })),
            "host-provided capacity must not require parallelizing the physical stage"
        );
    }
    let length = compute
        .bindings
        .iter()
        .find_map(|binding| match binding {
            Binding::StorageBuffer {
                usage: BufferUsage::Output,
                length,
                ..
            } => length.clone(),
            _ => None,
        })
        .expect("output length");
    serde_json::from_str(&serde_json::to_string(&length).unwrap()).unwrap()
}

fn assert_frame_inputs(inputs: &[HostSizeInput], x_offset: u32, y_offset: u32) {
    assert_eq!(
        inputs,
        [
            HostSizeInput::Uniform {
                name: "frame_resolution_x".into(),
                set: 0,
                binding: 0,
                offset: x_offset,
                scalar: HostSizeScalar::F32,
            },
            HostSizeInput::Uniform {
                name: "frame_resolution_y".into(),
                set: 0,
                binding: 0,
                offset: y_offset,
                scalar: HostSizeScalar::F32,
            },
        ]
    );
}

#[test]
fn uniform_output_capacity_is_host_provided_and_independent_of_dispatch() {
    for serial in [false, true] {
        let BufferLen::HostProvided { inputs, elem_bytes } = output_length(SOURCE, serial) else {
            panic!("uniform-derived output must have host-provided capacity");
        };
        assert_eq!(elem_bytes, 4);
        assert_frame_inputs(&inputs, 0, 4);
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
        .find_map(|pipeline| match pipeline {
            Pipeline::Compute(pipeline) => Some(pipeline),
            _ => None,
        })
        .unwrap();
    assert!(
        compute.stages.iter().any(|stage| matches!(
            stage.dispatch_size,
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
        .find_map(|binding| match binding {
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
fn host_provided_capacity_reports_abi_inputs_for_arbitrary_calculation() {
    let source = SOURCE
        .replace("{ resolution: vec3f32 }", "{ padding: f32, resolution: vec3f32 }")
        .replace("width * height", "if width > height then width else height");
    let BufferLen::HostProvided { inputs, elem_bytes } = output_length(&source, false) else {
        panic!("conditional output length must have host-provided capacity");
    };
    assert_eq!(elem_bytes, 4);
    assert_frame_inputs(&inputs, 16, 20);
}

#[test]
fn gpu_derived_output_length_is_host_provided_without_a_formula() {
    let source = SOURCE.replace("width * height", "i32(target_load(rendered, @[0i32,0i32],0u32))");
    let BufferLen::HostProvided { inputs, elem_bytes } = output_length(&source, false) else {
        panic!("GPU-derived output length must have host-provided capacity");
    };
    assert_eq!(elem_bytes, 4);
    assert!(inputs.is_empty());
}

#[test]
fn host_provided_capacity_respects_storage_stride_and_wgsl() {
    let source = SOURCE.replace("([]f32,", "([]vec3f32,").replace(
        "target_load(rendered, @[i % width, i / width], 0u32)",
        "@[f32(i), 0.0, 0.0]",
    );
    let BufferLen::HostProvided { inputs, elem_bytes } = output_length(&source, false) else {
        panic!("uniform-derived output must have host-provided capacity");
    };
    assert_eq!(elem_bytes, 16);
    assert_frame_inputs(&inputs, 0, 4);

    let lowered = crate::lower_ssa_to_wgsl_with_pipeline(crate::compile_thru_ssa(SOURCE).unwrap()).unwrap();
    assert!(lowered.pipeline.pipelines.iter().any(|pipeline| match pipeline {
        Pipeline::Compute(pipeline) => pipeline.bindings.iter().any(|binding| matches!(
            binding,
            Binding::StorageBuffer {
                length: Some(BufferLen::HostProvided { elem_bytes: 4, .. }),
                ..
            }
        )),
        _ => false,
    }));
}
