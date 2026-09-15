use super::{
    assert_naga_accepts_spirv, assert_scalar_prefix_emits_valid_wgsl, spirv_entry_interface_has_binding,
};
use crate::op::OpTag;
use crate::pipeline_descriptor::{Binding, BufferLen, Pipeline};
use crate::ssa::types::{ControlHeader, InstKind};
use crate::{compile_thru_spirv, compile_thru_ssa};

#[test]
fn camera_scalar_prepass_does_not_create_a_second_copy_dispatch() {
    let source = include_str!("../../testfiles/tinyporto_camera_prepass.wyn");
    let lowered = compile_thru_spirv(source).expect("camera prepass compiles");
    let compute_stages =
        lowered
            .pipeline
            .pipelines
            .iter()
            .filter_map(|pipeline| {
                if let Pipeline::Compute(compute) = pipeline {
                    Some(&compute.stages)
                } else {
                    None
                }
            })
            .flatten()
            .collect::<Vec<_>>();
    assert_eq!(
        compute_stages.len(),
        1,
        "camera construction needs only one singleton producer"
    );
    assert_naga_accepts_spirv(&lowered.spirv);
    assert_scalar_prefix_emits_valid_wgsl(source);
}

#[test]
fn unused_array_loop_state_does_not_allocate_or_copy() {
    let source = include_str!("../../testfiles/tinyporto_dead_array_state.wyn");
    let program = compile_thru_ssa(source).expect("scalar loop compiles");
    let arrays = program
        .entry_points
        .iter()
        .flat_map(|entry| entry.body.inner.insts.values())
        .filter(|instruction| matches!(instruction.data, InstKind::Alloca { .. }))
        .count();
    assert_eq!(
        arrays, 0,
        "the unused array state must disappear before physical allocation"
    );
    assert_naga_accepts_spirv(&compile_thru_spirv(source).unwrap().spirv);
    assert_scalar_prefix_emits_valid_wgsl(source);
}

#[test]
fn shared_capture_handoff_serves_later_serial_helper_without_dead_copies() {
    let source = include_str!("../../testfiles/tinyporto_capture_handoff.wyn");
    let program = compile_thru_ssa(source).expect("capture handoff compiles");
    assert_eq!(program.entry_points.len(), 4, "one producer and three consumers");
    let consumers: Vec<_> =
        program.entry_points.iter().filter(|entry| !entry.name.contains("prepass")).collect();
    assert_eq!(consumers.len(), 3);
    for entry in &consumers {
        assert!(
            !entry
                .body
                .inner
                .blocks
                .values()
                .any(|block| matches!(block.control_header, Some(ControlHeader::Loop { .. }))),
            "{} must consume the shared fold",
            entry.name
        );
    }
    for entry in &consumers[1..] {
        assert!(
            !entry.body.inner.insts.values().any(|inst| matches!(inst.data, InstKind::Alloca { .. })),
            "{} must not copy the unused array",
            entry.name
        );
    }
    assert_naga_accepts_spirv(&compile_thru_spirv(source).unwrap().spirv);
    assert_scalar_prefix_emits_valid_wgsl(source);
}

#[test]
fn array_state_splitting_preserves_declared_field_types() {
    compile_thru_ssa(include_str!("../../testfiles/miner.wyn"))
        .expect("tuple fields preserve declared array types across reduction joins");
}

#[test]
fn loop_arrays_survive_when_returned_or_read_by_live_scalar_state() {
    let base = include_str!("../../testfiles/tinyporto_dead_array_state.wyn");
    for source in [
        base.replace("[1]i32", "[4]i32").replace("[state]", "last"),
        base.replace("state + events[k]", "state + last[k % 4]"),
        base.replace("for k < 32", "for k < 0").replace("[state]", "[last[0]]"),
    ] {
        let lowered = compile_thru_spirv(&source).expect("live array loop compiles");
        assert_naga_accepts_spirv(&lowered.spirv);
        assert_scalar_prefix_emits_valid_wgsl(&source);
    }
}

#[test]
fn repeated_pure_clamp_calls_share_only_identical_arguments_in_dominating_scopes() {
    for (expression, expected) in [
        ("clampi(x, 0, 10) + clampi(x, 0, 10)", 1),
        ("clampi(x, 0, 10) + clampi(x, 0, 11)", 2),
        ("if x > 5 then clampi(x, 0, 10) else clampi(x, 0, 10)", 2),
        ("if clampi(x, 0, 10) > 5 then clampi(x, 0, 10) else 0", 1),
    ] {
        let source = format!(
            "def clampi(x: i32, lo: i32, hi: i32) i32 =\n\
               if x < lo then lo else if x > hi then hi else x\n\
             entry repro(xs: []i32) []i32 = map(|x| {expression}, xs)"
        );
        let program = compile_thru_ssa(&source).expect("clamp case compiles");
        let helper = program.functions.iter().find(|function| function.name == "clampi").unwrap();
        let calls = program
            .functions
            .iter()
            .map(|function| &function.body)
            .chain(program.entry_points.iter().map(|entry| &entry.body))
            .flat_map(|body| body.inner.insts.values())
            .filter(|instruction| {
                matches!(instruction.data, InstKind::Op { tag: OpTag::Call(callee), .. } if callee == helper.id)
            })
            .count();
        assert_eq!(calls, expected, "{expression}");
        assert_naga_accepts_spirv(&compile_thru_spirv(&source).unwrap().spirv);
    }
}

#[test]
fn filter_scatter_declares_the_count_buffer_it_reads() {
    let lowered = compile_thru_spirv("entry repro(xs: []i32) []i32 = filter(|i| xs[i] > 0, iota(4096))")
        .expect("filter compiles");
    let compute = lowered
        .pipeline
        .pipelines
        .iter()
        .find_map(
            |pipeline| {
                if let Pipeline::Compute(compute) = pipeline {
                    Some(compute)
                } else {
                    None
                }
            },
        )
        .unwrap();
    let (index, count) = compute
        .bindings
        .iter()
        .enumerate()
        .find(|(_, binding)| {
            matches!(
                binding,
                Binding::StorageBuffer {
                    length: Some(BufferLen::Fixed { bytes: 4 }),
                    ..
                }
            )
        })
        .expect("filter count allocation");
    let scatter = compute.stages.iter().find(|stage| stage.entry_point == "repro").unwrap();
    assert!(
        scatter.reads.contains(&index),
        "the scatter must depend on the count producer"
    );
    assert!(
        !scatter.writes.contains(&index),
        "the scatter only reads the count"
    );
    let Binding::StorageBuffer { set, binding, .. } = count else {
        unreachable!()
    };
    assert!(spirv_entry_interface_has_binding(
        &lowered.spirv,
        "repro",
        *set,
        *binding
    ));
    assert_naga_accepts_spirv(&lowered.spirv);
}

#[test]
fn cooperative_filter_scan_validates_empty_partial_and_multiple_tiles() {
    for length in [0, 1, 63, 64, 65, 255, 256, 257, 4096, 39592] {
        let source = format!("entry repro(xs: []i32) []i32 = filter(|i| xs[i] > 0, iota({length}))");
        assert_naga_accepts_spirv(&compile_thru_spirv(&source).expect("filter compiles").spirv);
        assert_scalar_prefix_emits_valid_wgsl(&source);
    }
    assert_scalar_prefix_emits_valid_wgsl("entry repro(xs: []i32) []i32 = filter(|x| x > 0, xs)");
}
