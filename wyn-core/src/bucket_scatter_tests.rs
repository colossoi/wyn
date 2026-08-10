const SOURCE: &str = r#"
#[compute]
entry bucket(
    #[storage(set=2, binding=0, access=write)] dest: *[32]u32,
    #[storage(set=2, binding=1, access=read)] keys: []i32,
    #[storage(set=2, binding=2, access=read)] values: []u32
) ([32]u32, [4]u32, u32) =
    bucket_scatter(dest, 4i32, 8i32, keys, values)
"#;

const GENERATED_SOURCE: &str = r#"
#[compute]
entry bucket_generated(
    #[storage(set=2, binding=0, access=write)] dest: *[32]u32
) ([32]u32, [4]u32, u32) =
  let domain = iota(16)
  let keys = map(|i: i32| i % 4, domain)
  let values = map(|i: i32| u32.i32(i), domain) in
  bucket_scatter(dest, 4i32, 8i32, keys, values)
"#;

#[test]
fn bucket_scatter_typechecks_and_reaches_tlc() {
    crate::compile_thru_tlc(SOURCE).expect("bucket_scatter should pass the functional frontend");
}

#[test]
fn bucket_scatter_emits_three_valid_wgsl_stages_with_internal_atomics() {
    use crate::pipeline_descriptor::{Binding, BufferLen, BufferUsage, Pipeline};

    let ssa = crate::compile_thru_ssa(SOURCE).expect("bucket_scatter should lower to SSA");
    let lowered = crate::lower_ssa_to_wgsl_with_pipeline(ssa).expect("bucket_scatter should lower to WGSL");
    let wgsl = lowered.wgsl;

    assert!(wgsl.contains("fn bucket_bucket_init("));
    assert!(wgsl.contains("fn bucket_bucket_insert("));
    assert!(wgsl.contains("fn bucket("));
    assert!(wgsl.contains("atomicAdd("));
    assert!(wgsl.contains("atomicStore("));
    assert!(wgsl.contains("atomicLoad("));

    let module = naga::front::wgsl::parse_str(&wgsl)
        .unwrap_or_else(|error| panic!("Naga rejected bucket_scatter WGSL: {error}\n{wgsl}"));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("Naga should validate bucket_scatter WGSL");

    let Pipeline::Compute(compute) = lowered.pipeline.pipelines.first().expect("one pipeline") else {
        panic!("bucket_scatter should produce a compute pipeline");
    };
    assert_eq!(
        compute.stages.iter().map(|stage| stage.entry_point.as_str()).collect::<Vec<_>>(),
        ["bucket_bucket_init", "bucket_bucket_insert", "bucket"]
    );
    let binding_index = |name: &str| {
        compute
            .bindings
            .iter()
            .position(
                |binding| matches!(binding, Binding::StorageBuffer { name: actual, .. } if actual == name),
            )
            .unwrap_or_else(|| panic!("missing bucket_scatter binding `{name}`"))
    };
    let dest = binding_index("dest");
    let keys = binding_index("keys");
    let values = binding_index("values");
    let counts = binding_index("bucket_counts");
    let overflow_out = binding_index("bucket_output_1");
    let overflow_cell = compute
        .bindings
        .iter()
        .position(|binding| {
            matches!(
                binding,
                Binding::StorageBuffer {
                    usage: BufferUsage::Intermediate,
                    length: Some(BufferLen::Fixed { bytes: 4 }),
                    ..
                }
            )
        })
        .expect("bucket_scatter should publish its internal overflow cell");
    assert!(matches!(
        &compute.bindings[counts],
        Binding::StorageBuffer {
            usage: BufferUsage::Output,
            length: Some(BufferLen::Fixed { bytes: 16 }),
            ..
        }
    ));
    assert!(matches!(
        &compute.bindings[overflow_out],
        Binding::StorageBuffer {
            usage: BufferUsage::Output,
            length: Some(BufferLen::Fixed { bytes: 4 }),
            ..
        }
    ));
    assert!(compute.stages[0].reads.is_empty());
    assert_eq!(compute.stages[0].writes, [counts, overflow_cell]);
    for index in [keys, values, counts] {
        assert!(compute.stages[1].reads.contains(&index));
    }
    for index in [dest, counts, overflow_cell] {
        assert!(compute.stages[1].writes.contains(&index));
    }
    assert_eq!(compute.stages[2].reads, [overflow_cell]);
    assert_eq!(compute.stages[2].writes, [overflow_out]);
}

#[test]
fn bucket_scatter_emits_valid_spirv_atomics() {
    use wspirv::binary::parse_words;
    use wspirv::dr::Loader;
    use wspirv::spirv::Op;

    let lowered = crate::compile_thru_spirv(SOURCE).expect("bucket_scatter should lower to SPIR-V");
    let bytes = lowered.spirv.iter().flat_map(|word| word.to_le_bytes()).collect::<Vec<_>>();
    let mut loader = Loader::new();
    parse_words(&lowered.spirv, &mut loader).expect("bucket_scatter SPIR-V should parse");
    let module = loader.module();
    let ops = module
        .functions
        .iter()
        .flat_map(|function| &function.blocks)
        .flat_map(|block| &block.instructions)
        .map(|instruction| instruction.class.opcode)
        .collect::<Vec<_>>();
    assert!(ops.contains(&Op::AtomicIAdd));
    assert!(ops.contains(&Op::AtomicStore));
    assert!(ops.contains(&Op::AtomicLoad));

    let naga_module = naga::front::spv::parse_u8_slice(
        &bytes,
        &naga::front::spv::Options {
            strict_capabilities: false,
            ..Default::default()
        },
    )
    .expect("Naga should parse bucket_scatter SPIR-V");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&naga_module)
    .expect("Naga should validate bucket_scatter SPIR-V");
}

#[test]
fn bucket_scatter_accepts_map_produced_keys_and_values() {
    use crate::pipeline_descriptor::{Binding, Pipeline};

    let ssa = crate::compile_thru_ssa(GENERATED_SOURCE)
        .expect("map-produced bucket_scatter inputs should lower to SSA");
    let lowered = crate::lower_ssa_to_wgsl_with_pipeline(ssa)
        .expect("map-produced bucket_scatter inputs should lower to WGSL");
    let wgsl = lowered.wgsl;
    assert!(wgsl.contains("fn bucket_generated_bucket_insert("));
    assert!(wgsl.contains(" % "), "the key-producing map must survive fusion");
    assert!(
        wgsl.contains("bitcast<u32>"),
        "the value-producing map must survive fusion"
    );

    let module = naga::front::wgsl::parse_str(&wgsl).unwrap_or_else(|error| {
        panic!("Naga rejected generated-input bucket_scatter WGSL: {error}\n{wgsl}")
    });
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("Naga should validate generated-input bucket_scatter WGSL");

    let Pipeline::Compute(compute) = lowered.pipeline.pipelines.first().expect("one pipeline") else {
        panic!("bucket_scatter should produce a compute pipeline");
    };
    assert!(compute.bindings.iter().all(|binding| {
        !matches!(
            binding,
            Binding::StorageBuffer { name, .. } if name == "keys" || name == "values"
        )
    }));

    crate::compile_thru_spirv(GENERATED_SOURCE)
        .expect("map-produced bucket_scatter inputs should lower to valid SPIR-V");
}

#[test]
fn bucket_scatter_rejects_runtime_bucket_count_until_dynamic_resource_sizing_exists() {
    let source = r#"
#[compute]
entry bucket(
    bucket_count: i32,
    #[storage(set=2, binding=0, access=write)] dest: *[32]u32,
    #[storage(set=2, binding=1, access=read)] keys: []i32,
    #[storage(set=2, binding=2, access=read)] values: []u32
) ([32]u32, [4]u32, u32) =
    bucket_scatter(dest, bucket_count, 8i32, keys, values)
"#;
    let error = crate::compile_thru_ssa(source).expect_err("runtime bucket_count should be rejected");
    assert!(
        error.to_string().contains("positive compile-time bucket_count"),
        "unexpected diagnostic: {error}"
    );
}
