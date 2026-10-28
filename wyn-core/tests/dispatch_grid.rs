#[path = "support/pipeline.rs"]
mod pipeline;
use wyn_core::host::Pipeline;
use wyn_core::{host, lower_ssa_to_wgsl_with_program, CodegenTarget};

#[test]
fn explicit_grids_preserve_all_axes_in_the_shader_and_descriptor() {
    use host::DispatchSize;
    let mut source = pipeline::source(include_str!("../../tests/compiler/library/dispatch_grid.wyn"));
    for (x, y, z) in [(1, 1, 1), (2, 3, 4), (257, 1, 1), (3, 7, 17)] {
        for definition in &mut source.defs {
            if let wyn_core::tlc::DefMeta::EntryPoint(entry) = &mut definition.meta {
                entry.declaration.compute_dispatch =
                    Some(wyn_core::interface::ComputeDispatchGrid { x, y, z });
            }
        }
        let output = lower_ssa_to_wgsl_with_program(pipeline::lower(&source, CodegenTarget::Wgsl)).unwrap();
        assert_eq!(output.program.interface.pipelines.len(), 3);
        for pipeline in &output.program.interface.pipelines {
            let Pipeline::Compute(p) = pipeline else {
                panic!("compute")
            };
            assert_eq!(
                p.stages[0].dispatch_size,
                DispatchSize::Fixed {
                    x,
                    y,
                    z,
                    explicit: true
                }
            );
            for stage in &p.stages[1..] {
                let (x, y, z) = if stage.entry_point.ends_with("offsets") { (x, y, z) } else { (1, 1, 1) };
                assert_eq!(
                    stage.dispatch_size,
                    DispatchSize::Fixed {
                        x,
                        y,
                        z,
                        explicit: true
                    }
                );
            }
            if p.stages.len() > 1 {
                let carries: Vec<_> =
                    p.stages[0].writes.iter().filter(|slot| p.stages[1].reads.contains(slot)).collect();
                assert!(!carries.is_empty());
                for &slot in carries {
                    assert!(matches!(&p.bindings[slot], host::Binding::StorageBuffer {
                        length: Some(host::BufferLen::Fixed { bytes }), ..
                    } if *bytes == u64::from(x) * u64::from(y) * u64::from(z) * 4));
                }
            }
        }
        let module = naga::front::wgsl::parse_str(&output.wgsl).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        if y > 1 {
            assert!(output.wgsl.contains(".y") && output.wgsl.contains(".z"));
        }
    }
}
