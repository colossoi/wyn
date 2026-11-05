#[path = "support/pipeline.rs"]
mod pipeline;
use wyn_core::host::{Allocation, Operation, Pipeline, ShaderFormat};
use wyn_core::{lower_ssa_to_spirv, lower_ssa_to_wgsl_with_program, CodegenTarget};

#[test]
fn tiny_while_loops_have_no_host_readback_cycle() {
    let source = pipeline::source(
        "entry small(xs:[4]i32) ([4]i32,i32) =
         loop (ys,i)=(xs,0) while i<128 do (map(|x|x+1,ys),i+1)
         entry nested(xs:[4]i32) ([4]i32,i32) =
         loop (ys,i)=(xs,0) while i<128 do
           let next=loop zs=ys for j<2 do map(|x|x+1,zs) in (next,i+1)",
    );
    let ssa = pipeline::lower(&source, CodegenTarget::Portable);
    for format in [ShaderFormat::Spirv, ShaderFormat::Wgsl] {
        let program = match format {
            ShaderFormat::Spirv => lower_ssa_to_spirv(ssa.clone()).unwrap().program,
            ShaderFormat::Wgsl => lower_ssa_to_wgsl_with_program(ssa.clone()).unwrap().program,
        };
        assert!(program.interface.dispatch_loops.is_empty());
        let whl = program.to_whl("small", format).unwrap();
        wyn_host_interp::Program::parse(&whl).unwrap();
        program.to_rust_wgpu("small", format).unwrap();
    }
}

#[test]
fn dispatch_loops_share_product_carries_and_control_across_backends() {
    use wyn_core::host::DispatchLoopControl;
    let source = pipeline::source(include_str!(
        "../../tests/compiler/library/dispatch_loop_products.wyn"
    ));
    let ssa = pipeline::lower(&source, CodegenTarget::Portable);
    for format in [ShaderFormat::Spirv, ShaderFormat::Wgsl] {
        let program = match format {
            ShaderFormat::Spirv => lower_ssa_to_spirv(ssa.clone()).unwrap().program,
            ShaderFormat::Wgsl => lower_ssa_to_wgsl_with_program(ssa.clone()).unwrap().program,
        };
        let loops = &program.interface.dispatch_loops;
        assert!(loops
            .iter()
            .any(|loop_| matches!(loop_.control, DispatchLoopControl::While { .. })
                && loop_.carried.len() == 3));
        assert!(loops.iter().any(
            |loop_| matches!(loop_.control, DispatchLoopControl::Counted { .. })
                && loop_.carried.len() == 3
        ));
        let nested = program.entries.iter().find(|entry| entry.name == "nested_runtime").unwrap();
        let Some(Operation::Loop { region, body, .. }) =
            nested.operations.iter().find(|operation| matches!(operation, Operation::Loop { .. }))
        else {
            panic!("runtime outer loop must remain distributed");
        };
        assert!(matches!(
            loops[*region].control,
            DispatchLoopControl::While { .. }
        ));
        let Some(Operation::Loop { region, .. }) =
            body.iter().find(|operation| matches!(operation, Operation::Loop { .. }))
        else {
            panic!("runtime inner loop must remain distributed inside the outer loop");
        };
        assert!(matches!(
            loops[*region].control,
            DispatchLoopControl::Counted { .. }
        ));
        for loop_ in loops {
            if matches!(loop_.control, DispatchLoopControl::While { .. }) {
                assert!(!loop_.test.is_empty());
            }
            assert!(!loop_.body.is_empty());
            let Pipeline::Compute(pipeline) = &program.interface.pipelines[loop_.pipeline] else {
                panic!("compute loop");
            };
            assert!(loop_.body.iter().any(|&stage| pipeline.stages[stage].workgroup_size.0 > 1));
        }
        let whl = program.to_whl("loop-products", format).unwrap();
        wyn_host_interp::Program::parse(&whl).unwrap();
        program.to_rust_wgpu("loop_products", format).unwrap();
    }
}

#[test]
fn tuple_loop_outputs_and_scalar_bounds_reach_host_backends() {
    for source in [
        include_str!("../../tests/compiler/library/local_tuple_collectives.wyn"),
        include_str!("../../tests/compiler/library/nested_tuple_loop_state.wyn"),
    ] {
        let nested = source.contains("entry updated(");
        let source = pipeline::source(source);
        let ssa = pipeline::lower(&source, CodegenTarget::Portable);
        for format in [ShaderFormat::Spirv, ShaderFormat::Wgsl] {
            let ssa = ssa.clone();
            let program = match format {
                ShaderFormat::Spirv => lower_ssa_to_spirv(ssa).unwrap().program,
                ShaderFormat::Wgsl => lower_ssa_to_wgsl_with_program(ssa).unwrap().program,
            };
            program.to_rust_wgpu("tuple_loops", format).unwrap();
            if nested {
                use wyn_core::host::{Binding, BufferLen, ResultLayout, ResultScalar};
                let descriptor = &program.interface;
                assert_eq!(descriptor.source_results.len(), 4);
                for result in &descriptor.source_results {
                    let Pipeline::Compute(p) = &descriptor.pipelines[result.pipeline_index] else {
                        panic!("compute result");
                    };
                    let ResultLayout::Array {
                        element,
                        stride,
                        length: Some(2),
                    } = &result.layout
                    else {
                        panic!("fixed array result: {:?}", result.layout);
                    };
                    if result.result == 0 {
                        assert_eq!(*stride, 4);
                        assert_eq!(**element, ResultLayout::Scalar(ResultScalar::I32));
                    } else {
                        assert_eq!(*stride, 8);
                        let ResultLayout::Tuple { fields, size: 8 } = &**element else {
                            panic!("tuple element: {element:?}");
                        };
                        assert_eq!(fields.len(), 2);
                        assert_eq!(fields[0].offset, 0);
                        assert_eq!(fields[1].offset, 4);
                        assert_eq!(fields[1].layout, ResultLayout::Scalar(ResultScalar::Bool));
                    }
                    assert!(p.bindings.iter().any(|b| matches!(b,
                Binding::StorageBuffer { set, binding, length: Some(BufferLen::Fixed { bytes }), .. }
                    if (*set, *binding) == (result.set, result.binding) && *bytes == u64::from(*stride) * 2
            )));
                }
            }
            for entry in &program.entries {
                for (position, operation) in entry.operations.iter().enumerate() {
                    let Operation::Loop { region, pipeline, .. } = operation else {
                        continue;
                    };
                    let completion = program.interface.dispatch_loops[*region].completion;
                    let Pipeline::Compute(compute) = &program.interface.pipelines[*pipeline] else {
                        panic!("compute loop");
                    };
                    if !compute.stages[completion].uses.writes.is_empty() {
                        assert!(entry.operations[position + 1..].iter().any(|operation|
                            matches!(operation, Operation::Dispatch { pipeline: p, stage, .. }
                                if p == pipeline && *stage == completion)), "{} output copy", entry.name);
                    }
                }
            }
        }
    }
}

#[test]
fn rust_context_keeps_pipeline_creation_out_of_entry_calls() {
    for source in [
        include_str!("../../tests/compiler/library/rust_host_storage_access_mismatch.wyn"),
        include_str!("../../tests/compiler/library/rust_host_frame_composition.wyn"),
    ] {
        let source = pipeline::source(source);
        let ssa = pipeline::lower(&source, CodegenTarget::Portable);
        for format in [ShaderFormat::Spirv, ShaderFormat::Wgsl] {
            let ssa = ssa.clone();
            let program = match format {
                ShaderFormat::Spirv => lower_ssa_to_spirv(ssa).unwrap().program,
                ShaderFormat::Wgsl => lower_ssa_to_wgsl_with_program(ssa).unwrap().program,
            };
            let rust = program.to_rust_wgpu("shader", format).unwrap();
            let (context, call) = rust.split_once("pub fn host_reproduce(").unwrap();
            assert!(context.contains("pub struct HostContext"));
            assert_eq!(context.matches("create_shader_module").count(), 1);
            for creation in [
                "create_shader_module",
                "create_compute_pipeline",
                "create_render_pipeline",
                "create_pipeline_layout",
            ] {
                assert!(
                    !call.contains(creation),
                    "{format:?}: {creation} inside entry call"
                );
            }
            let entry = &program.entries[0];
            let scratch = entry.allocations.iter().filter(|allocation| {
                matches!(allocation, Allocation::Buffer { resource, .. } if !entry.results.contains(resource))
            }).count();
            assert_eq!(call.matches("support::scratch_buffer(").count(), scratch);
            assert_eq!(call.matches("encoder.clear_buffer(").count(), scratch);
            let returned_buffers = entry.allocations.iter().filter(|allocation| {
                matches!(allocation, Allocation::Buffer { resource, .. } if entry.results.contains(resource))
            }).count();
            assert_eq!(call.matches("device.create_buffer(").count(), returned_buffers);
        }
    }
}
