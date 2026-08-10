const SOURCE: &str = r#"
#[compute]
entry bucket(
    #[storage(set=2, binding=0, access=write)] dest: *[4][8]u32
) ([4][8]u32, [4]u32, u32) =
    let domain = iota(16)
    let items = map(|i: i32| (i % 4, u32.i32(i)), domain) in
    bucket_scatter_1d(dest, items)
"#;

const GENERATED_SOURCE: &str = r#"
#[compute]
entry bucket_generated(
    #[storage(set=2, binding=0, access=write)] dest: *[4][8]u32
) ([4][8]u32, [4]u32, u32) =
  let domain = iota(16)
  let items = map(|i: i32| (i % 4, u32.i32(i)), domain) in
  bucket_scatter_1d(dest, items)
"#;

fn ranked_source(rank: u8) -> String {
    let dimensions: &[usize] = match rank {
        2 => &[2, 8],
        3 => &[2, 2, 4],
        4 => &[2, 2, 2, 2],
        _ => panic!("ranked bucket_scatter test only covers ranks 2 through 4"),
    };
    let shape = dimensions.iter().map(|size| format!("[{size}]")).collect::<String>();
    fn literal(dimensions: &[usize]) -> String {
        let Some((&size, rest)) = dimensions.split_first() else {
            return "(0i32, 7u32)".to_string();
        };
        let item = literal(rest);
        format!("[{}]", vec![item; size].join(", "))
    }
    let items = literal(dimensions);
    format!(
        r#"
#[compute]
entry bucket_{rank}d(
    #[storage(set=2, binding=0, access=write)] dest: *[4][8]u32
) ([4][8]u32, [4]u32, u32) =
    let items: {shape}(i32, u32) = {items} in
    bucket_scatter_{rank}d(dest, items)
"#
    )
}

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
    assert!(compute.stages[1].reads.contains(&counts));
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
fn bucket_scatter_accepts_map_produced_items() {
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
            Binding::StorageBuffer { name, .. } if name == "items"
        )
    }));

    crate::compile_thru_spirv(GENERATED_SOURCE)
        .expect("map-produced bucket_scatter inputs should lower to valid SPIR-V");
}

#[test]
fn bucket_scatter_ranked_forms_emit_valid_wgsl_and_spirv() {
    for rank in 2..=4 {
        let source = ranked_source(rank);
        let ssa = crate::compile_thru_ssa(&source)
            .unwrap_or_else(|error| panic!("bucket_scatter_{rank}d should lower to SSA: {error}"));
        let lowered = crate::lower_ssa_to_wgsl_with_pipeline(ssa)
            .unwrap_or_else(|error| panic!("bucket_scatter_{rank}d should lower to WGSL: {error}"));
        let module = naga::front::wgsl::parse_str(&lowered.wgsl).unwrap_or_else(|error| {
            panic!(
                "Naga rejected bucket_scatter_{rank}d WGSL: {error}\n{}",
                lowered.wgsl
            )
        });
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|error| {
            panic!(
                "Naga rejected bucket_scatter_{rank}d: {error:?}\n{}",
                lowered.wgsl
            )
        });

        let spirv = crate::compile_thru_spirv(&source)
            .unwrap_or_else(|error| panic!("bucket_scatter_{rank}d should lower to SPIR-V: {error}"));
        let bytes = spirv.spirv.iter().flat_map(|word| word.to_le_bytes()).collect::<Vec<_>>();
        let module = naga::front::spv::parse_u8_slice(
            &bytes,
            &naga::front::spv::Options {
                strict_capabilities: false,
                ..Default::default()
            },
        )
        .unwrap_or_else(|error| panic!("Naga could not parse bucket_scatter_{rank}d SPIR-V: {error}"));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|error| panic!("Naga rejected bucket_scatter_{rank}d SPIR-V: {error}"));
    }
}

#[test]
fn bucket_scatter_rejects_a_non_rank_two_destination() {
    let source = r#"
#[compute]
entry bucket(
    #[storage(set=2, binding=0, access=write)] dest: *[32]u32,
    #[storage(set=2, binding=1, access=read)] items: [16](i32, u32)
) ([32]u32, [4]u32, u32) =
    bucket_scatter_1d(dest, items)
"#;
    let error = crate::compile_thru_ssa(source).expect_err("rank-one destination should be rejected");
    assert!(
        error.to_string().contains("type mismatch"),
        "unexpected diagnostic: {error}"
    );
}
