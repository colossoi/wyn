use super::*;
use wyn_pipeline_descriptor::{BufferLen, UniformMember};

#[test]
fn spirv_passthrough_is_only_enabled_for_vulkan() {
    assert!(supports_spirv_passthrough(Backend::Vulkan));
    assert!(!supports_spirv_passthrough(Backend::Metal));
    assert!(!supports_spirv_passthrough(Backend::Dx12));
    assert!(!supports_spirv_passthrough(Backend::Gl));
    assert!(!supports_spirv_passthrough(Backend::BrowserWebGpu));
}

#[test]
fn packs_wgsl_parameter_blocks_at_descriptor_offsets() {
    let bindings = vec![Binding::StorageBuffer {
        set: 1,
        binding: 0,
        access: Access::ReadOnly,
        usage: BufferUsage::Input,
        name: "params".to_string(),
        resource: None,
        length: Some(BufferLen::Fixed { bytes: 32 }),
        members: vec![
            UniformMember {
                name: "count".to_string(),
                offset: 0,
                size: 4,
            },
            UniformMember {
                name: "direction".to_string(),
                offset: 16,
                size: 12,
            },
        ],
    }];
    let values = vec![
        PushConstantSpec {
            name: "count".to_string(),
            offset: 0,
            data: 130u32.to_le_bytes().to_vec(),
        },
        PushConstantSpec {
            name: "direction".to_string(),
            offset: 0,
            data: [1.0f32, 2.0, 3.0].into_iter().flat_map(f32::to_le_bytes).collect(),
        },
    ];

    let blocks = build_parameter_block_bytes(&bindings, &values, false).unwrap();
    let block = &blocks[&(1, 0)];
    assert_eq!(&block[0..4], &130u32.to_le_bytes());
    assert!(block[4..16].iter().all(|byte| *byte == 0));
    assert_eq!(&block[16..28], values[1].data.as_slice());
    assert!(block[28..32].iter().all(|byte| *byte == 0));

    let dispatch = DispatchSize::DerivedFrom {
        len: DispatchLen::StorageBuffer {
            set: 1,
            binding: 0,
            offset: 0,
        },
        workgroup_size: 64,
    };
    assert_eq!(
        resolve_dispatch_size_with_parameters(&dispatch, &StorageBuffers::new(), &[], &blocks),
        (3, 1, 1)
    );
}
