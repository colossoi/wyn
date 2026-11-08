//! Shader validation and host extraction for compiler-packed result fields.
#[path = "support/pipeline.rs"]
mod pipeline;
#[path = "../../wyn-host-interp/src/test_backend.rs"]
mod test_backend;
use test_backend::Trace;
use wyn_core::host::{Binding, ShaderFormat};
use wyn_core::{lower_ssa_to_spirv, lower_ssa_to_wgsl_with_program, CodegenTarget};
use wyn_host_interp::{Backend, Error, Number, Options, Program, Result, Value};

// Supply a known packed dispatch result. This checks host byte-range extraction
// and lifetime, independently of GPU execution and of the shader arithmetic.
#[derive(Default)]
struct PackedTrace(Trace, Option<u64>, Vec<u64>);
impl Backend for PackedTrace {
    fn call(&mut self, program: &Program, name: &str, args: &[Value]) -> Result<Value> {
        match name {
            "gpu-alloc" => {
                let value = self.0.call(program, name, args)?;
                self.2.push(args[0].u64()?);
                if args[0].u64()? == 120 {
                    self.1 = Some(value.handle()?);
                }
                Ok(value)
            }
            "gpu-dispatch" => {
                let output = self.1.unwrap();
                let expected: Vec<_> = (1..=30).flat_map(|i| (i as f32).to_le_bytes()).collect();
                assert_eq!(self.0.buffers[&output].len(), expected.len());
                self.0.buffers.insert(output, expected);
                Ok(Value::Nil)
            }
            _ => self.0.call(program, name, args),
        }
    }
}
fn validate(module: naga::Module) {
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
}
#[test]
fn wide_results_pack_and_preserve_individual_values() {
    let source = pipeline::source(include_str!("../../tests/compiler/library/storage_packing.wyn"));
    let ssa = pipeline::lower(&source, CodegenTarget::Portable);
    let spirv = lower_ssa_to_spirv(ssa.clone()).unwrap();
    let repeated = lower_ssa_to_wgsl_with_program(ssa.clone()).unwrap();
    let wgsl = lower_ssa_to_wgsl_with_program(ssa).unwrap();
    assert_eq!(repeated.wgsl, wgsl.wgsl);
    assert_eq!(
        repeated.program.to_whl("packed", ShaderFormat::Wgsl).unwrap(),
        wgsl.program.to_whl("packed", ShaderFormat::Wgsl).unwrap()
    );
    validate(
        naga::front::spv::Frontend::new(spirv.spirv.into_iter(), &Default::default()).parse().unwrap(),
    );
    validate(naga::front::wgsl::parse_str(&wgsl.wgsl).unwrap());
    for (host, format) in [
        (spirv.program, ShaderFormat::Spirv),
        (wgsl.program, ShaderFormat::Wgsl),
    ] {
        let fields: Vec<_> = host.interface.source_results.iter().filter(|r| r.entry == "repro").collect();
        assert_eq!(fields.len(), 30);
        for (i, field) in fields.iter().enumerate() {
            assert_eq!(field.result, i);
            assert_eq!(field.packed_range, Some((i as u32 * 4, 4)));
            assert_eq!((field.set, field.binding), (fields[0].set, fields[0].binding));
        }
        assert!(host
            .interface
            .source_results
            .iter()
            .filter(|r| r.entry == "small")
            .all(|r| r.packed_range.is_none()));
        let mixed: Vec<_> = host.interface.source_results.iter().filter(|r| r.entry == "mixed").collect();
        assert_eq!(mixed.len(), 9);
        for (i, field) in mixed.iter().enumerate() {
            let (offset, bytes) = field.packed_range.expect("mixed fields must pack");
            let stride = if i % 3 == 2 { 16 } else { 4 };
            assert_eq!(bytes, stride);
            assert_eq!(offset, (i / 3) as u32 * stride);
            assert_eq!(field.binding, mixed[i % 3].binding);
        }
        let aliases = host.entries.iter().find(|e| e.name == "aliases").unwrap();
        assert_eq!(aliases.results[8], aliases.results[9]);
        assert!(aliases.inputs.contains(&aliases.results[8]));
        let alias_fields: Vec<_> =
            host.interface.source_results.iter().filter(|r| r.entry == "aliases").collect();
        assert!(alias_fields[..8].iter().all(|r| r.packed_range.is_some()));
        assert!(alias_fields[8..].iter().all(|r| r.packed_range.is_none()));
        let staged: Vec<_> = host.interface.source_results.iter().filter(|r| r.entry == "staged").collect();
        assert_eq!(staged.len(), 8);
        for field in &staged {
            if let Some((offset, bytes)) = field.packed_range {
                assert_eq!(bytes, 4);
                assert_eq!(offset % 4, 0);
            }
        }
        let flags: Vec<_> = host.interface.source_results.iter().filter(|r| r.entry == "flags").collect();
        assert_eq!(flags.len(), 8);
        assert!(flags.iter().all(|r| r.packed_range.is_some()));
        let carried: Vec<_> =
            host.interface.source_results.iter().filter(|r| r.entry == "carried").collect();
        assert_eq!(carried.len(), 9);
        assert!(carried[0].packed_range.is_none());
        assert!(carried[1..].iter().all(|r| r.packed_range.is_some()));
        for repeated in &host.interface.dispatch_loops {
            let mut resources = std::collections::BTreeSet::new();
            for (current, next) in &repeated.carried {
                for source in [current, next] {
                    if let wyn_core::host::ScalarSource::Binding { set, binding } = source {
                        assert!(resources
                            .insert(host.slot_resource(repeated.pipeline, *set, *binding).unwrap()));
                    }
                }
            }
        }
        let p = fields[0].pipeline_index;
        assert!(
            host.bindings(p).iter().filter(|b| matches!(b, Binding::StorageBuffer { .. })).count() <= 3
        );
        let whl = host.to_whl("packed", format).unwrap();
        let program = Program::parse(&whl).unwrap();
        let mut backend = PackedTrace::default();
        let arguments = program
            .entry("repro")
            .unwrap()
            .parameters
            .iter()
            .map(|_| backend.0.input(vec![0; 120]))
            .collect::<Vec<_>>();
        let result = program.run("repro", &arguments, &mut backend).unwrap();
        let values = result.list().unwrap();
        assert_eq!(values.len(), 30);
        let mut handles = std::collections::BTreeSet::new();
        for (i, value) in values.iter().enumerate() {
            let handle = value.handle().unwrap();
            assert!(handles.insert(handle), "result handles must remain independent");
            assert_eq!(backend.0.buffers[&handle], ((i + 1) as f32).to_le_bytes());
        }
        assert_eq!(
            backend.0.freed.iter().filter(|&&id| Some(id) == backend.1).count(),
            1,
            "packed temporary must be released once"
        );
        assert_eq!(
            backend.2.iter().filter(|&&bytes| bytes == 120).count(),
            1,
            "one packed allocation"
        );
        assert_eq!(
            backend.2.iter().filter(|&&bytes| bytes == 4).count(),
            30,
            "WHL returns retain independent buffers"
        );
        assert!(
            backend.2.iter().all(|bytes| [4, 12, 120].contains(bytes)),
            "only result storage and the existing length metadata are allocated"
        );
        assert!(backend.0.device_reads.is_empty());
        let rust = host.to_rust_wgpu("packed", format).unwrap();
        assert!(rust.contains("offset: 116"));
    }
}

// Keep the conditional graphics case separate from the byte-repeat fixture:
// its contract is shared allocation identity and producer/consumer ordering.
#[test]
fn packed_settings_cross_compute_and_graphics_without_host_copies() {
    let source = pipeline::source(include_str!(
        "../../tests/compiler/library/storage_packing_handoff.wyn"
    ));
    let ssa = pipeline::lower(&source, CodegenTarget::Portable);
    let spirv = lower_ssa_to_spirv(ssa.clone()).unwrap();
    let wgsl = lower_ssa_to_wgsl_with_program(ssa).unwrap();
    validate(
        naga::front::spv::Frontend::new(spirv.spirv.into_iter(), &Default::default()).parse().unwrap(),
    );
    validate(naga::front::wgsl::parse_str(&wgsl.wgsl).unwrap());
    for (host, format) in [
        (spirv.program, ShaderFormat::Spirv),
        (wgsl.program, ShaderFormat::Wgsl),
    ] {
        let handoff_pack = host.interface.frame_graph.resources.iter().find(|resource| {
            matches!(resource.extent, Some(wyn_core::host::FrameResourceExtent::StorageBuffer {
                length: Some(wyn_core::host::BufferLen::Fixed { bytes: 48 })
            })) && resource.bindings.iter().any(|binding| {
                matches!(&host.interface.pipelines[binding.pipeline_index], wyn_core::host::Pipeline::Compute(p) if p.stages.iter().any(|s| s.owner == "handoff"))
            }) && resource.bindings.iter().any(|binding| {
                matches!(&host.interface.pipelines[binding.pipeline_index], wyn_core::host::Pipeline::Graphics(p) if p.stages.iter().any(|s| s.owner == "handoff"))
            })
        }).expect("packed settings must remain shared between producer and graphics consumer");
        assert_eq!(handoff_pack.bindings.len(), 2);
        let graphics = host
            .interface
            .pipelines
            .iter()
            .find_map(|p| match p {
                wyn_core::host::Pipeline::Graphics(p) if p.stages.iter().any(|s| s.owner == "handoff") => {
                    Some(p)
                }
                _ => None,
            })
            .unwrap();
        assert_eq!(
            graphics.bindings.iter().filter(|b| matches!(b, Binding::StorageBuffer { .. })).count(),
            3,
            "input, vertices, and a single settings allocation"
        );
        assert!(host.to_whl("packed", format).unwrap().contains("handoff"));

        assert!(!host.to_whl("packed", format).unwrap().contains("(gpu-copy "));
        let bytes: Vec<_> = host.entries[0]
            .allocations
            .iter()
            .filter_map(|a| match a {
                wyn_core::host::Allocation::Buffer {
                    bytes: wyn_core::host::Expr::Integer(n),
                    ..
                } => Some(*n),
                _ => None,
            })
            .collect();
        assert_eq!(
            bytes.iter().sum::<i64>(),
            108,
            "one 48-byte settings pack, 48-byte geometry, and 12-byte length metadata"
        );
    }
}
