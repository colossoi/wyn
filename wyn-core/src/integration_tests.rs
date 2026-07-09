#![cfg(test)]
//! Integration tests for the full compilation pipeline.
//!
//! These tests verify that source code compiles correctly through all stages:
//! parse → desugar → resolve → type_check → alias_check → TLC → monomorphize → SSA
//!
//! All tests include entry points to ensure monomorphization can find reachable code.

use crate::ssa::types::Program;
use crate::tlc::extract_lambda_params;
use crate::tlc::TermKind;
use crate::tlc::VarRef;
use crate::Compiler;
use crate::SymbolTable;

/// Run source through the pipeline up to SSA.
fn compile_to_ssa(input: &str) -> Program {
    crate::compile_thru_ssa(input).expect("compile to SSA").ssa
}

/// Helper to check that code fails type checking (for testing error cases).
fn should_fail_type_check(input: &str) -> bool {
    crate::compile_thru_frontend(input).is_err()
}

/// Helper to compile up through TLC fusion (stops before defunctionalization).
/// Off-milestone stop — drives the typestate API directly so the same
/// `module_manager` covers both `type_check` and `to_tlc`.
fn compile_to_fused_tlc(input: &str) -> crate::tlc::Program {
    crate::test_pipeline::compile_to_tlc_program(input)
}

#[test]
fn semantic_segops_survive_optimization_and_logical_allocation() {
    use crate::egir::types::{EgirSoac, SegExtent, SegOpKind, SideEffectKind};

    let tlc = crate::compile_thru_tlc(
        r#"
#[compute]
entry sum(xs: []i32) i32 = reduce(|a: i32, b: i32| a + b, 0, xs)
"#,
    )
    .expect("TLC");
    let allocated = tlc.infer_input_slice_bounds().to_egraph().expect("semantic EGIR allocation");
    crate::egir::parallelize::verify_semantic(&allocated.inner).expect("complete semantic EGIR");
    let seg = allocated
        .inner
        .entry_points
        .iter()
        .flat_map(|entry| entry.graph.skeleton.blocks.iter().flat_map(|(_, block)| &block.side_effects))
        .find_map(|effect| match &effect.kind {
            SideEffectKind::Soac(EgirSoac::Seg { space, kind, .. }) => Some((space, kind)),
            _ => None,
        })
        .expect("SegRed remains present before target lowering");
    assert!(matches!(seg.1, SegOpKind::SegRed { .. }));
    assert!(matches!(
        seg.0.dims.as_slice(),
        [SegExtent::ResourceLength { .. }]
    ));
    assert!(
        allocated.inner.resources.len() >= 2,
        "input and output resources are planned logically"
    );
    assert!(allocated.semantic_ir().contains("SegRed"));
    assert!(allocated.semantic_ir().contains("ResourceLength"));

    // Milestone 5: the reduce's partial buffer is now an authoritative logical
    // resource (drawn at the allocation boundary), and the SegRed op records the
    // reserved `ResourceId` for terminal lowering to consume.
    use crate::egir::program::{CompilerResourceKind, ResourceOrigin};
    let partials = allocated
        .inner
        .resources
        .iter()
        .filter(|resource| {
            matches!(
                resource.origin,
                ResourceOrigin::Compiler(ref compiler)
                    if compiler.kind == CompilerResourceKind::ReducePartial
            )
        })
        .count();
    assert_eq!(partials, 1, "one reduce partial reserved as a logical resource");
    let records_scratch = allocated
        .inner
        .entry_points
        .iter()
        .flat_map(|entry| entry.graph.skeleton.blocks.iter().flat_map(|(_, block)| &block.side_effects))
        .any(|effect| {
            matches!(
                &effect.kind,
                SideEffectKind::Soac(EgirSoac::Seg {
                    kind: SegOpKind::SegRed { .. },
                    scratch_resources,
                    ..
                }) if scratch_resources.len() == 1
            )
        });
    assert!(records_scratch, "SegRed records its reserved partial ResourceId");
}

/// Milestone-5 horizontal fusion: the four same-space reductions of
/// `[sum, product, min, max]` — separate lane-local reductions over one input,
/// writing fields of one aggregate output — merge into a single
/// four-accumulator SegOp instead of four one-accumulator ops.
#[test]
fn same_space_reductions_fuse_into_one_multi_accumulator_op() {
    use crate::egir::types::{EgirSoac, SegOpKind, SideEffectKind};

    let tlc = crate::compile_thru_tlc(
        r#"
def N: i32 = 256
#[compute]
entry e() [4]f32 =
    let xs = map(|i: i32| f32.i32(i), 0i32 ..< N) in
    [f32.sum(xs), f32.product(xs), f32.minimum(xs), f32.maximum(xs)]
"#,
    )
    .expect("TLC");
    let allocated = tlc.infer_input_slice_bounds().to_egraph().expect("semantic EGIR allocation");
    let operator_counts: Vec<usize> = allocated
        .inner
        .entry_points
        .iter()
        .flat_map(|entry| entry.graph.skeleton.blocks.iter().flat_map(|(_, block)| &block.side_effects))
        .filter_map(|effect| match &effect.kind {
            SideEffectKind::Soac(EgirSoac::Seg {
                kind:
                    SegOpKind::SegRed { operators }
                    | SegOpKind::SegScan { operators }
                    | SegOpKind::SegComposite { operators },
                ..
            }) => Some(operators.len()),
            _ => None,
        })
        .collect();
    assert_eq!(
        operator_counts,
        vec![4],
        "the four same-space reductions fuse into one four-accumulator op"
    );
    let remaining_maps = allocated
        .inner
        .entry_points
        .iter()
        .flat_map(|entry| entry.graph.skeleton.blocks.iter().flat_map(|(_, block)| &block.side_effects))
        .filter(|effect| {
            matches!(
                effect.kind,
                SideEffectKind::Soac(EgirSoac::Seg {
                    kind: SegOpKind::SegMap,
                    ..
                })
            )
        })
        .count();
    assert_eq!(remaining_maps, 0, "the single-consumer map is vertically fused");
    assert_eq!(
        allocated
            .inner
            .functions
            .iter()
            .filter(|function| function.name.contains("_vertical_step_"))
            .count(),
        4,
        "one composed step region per accumulator"
    );
    let operators = allocated
        .inner
        .entry_points
        .iter()
        .flat_map(|entry| entry.graph.skeleton.blocks.iter().flat_map(|(_, block)| &block.side_effects))
        .find_map(|effect| match &effect.kind {
            SideEffectKind::Soac(EgirSoac::Seg {
                kind: SegOpKind::SegRed { operators },
                ..
            }) => Some(operators),
            _ => None,
        })
        .expect("fused SegRed");
    for operator in operators {
        assert_eq!(operator.input_indices.len(), 1);
        assert_eq!(
            allocated.inner.regions[&operator.step.region].params.len(),
            2,
            "composed step receives accumulator plus only its routed input"
        );
    }
}

#[test]
fn horizontal_fusion_does_not_cross_an_intervening_effect_token() {
    use crate::egir::types::{EgirSoac, SegOpKind, SideEffectKind};
    let allocated = crate::compile_thru_tlc(
        r#"
#[compute]
entry e() [3]i32 =
    let xs = 0i32 ..< 8 in
    let ys = 0i32 ..< 4 in
    [
      reduce(|a: i32, b: i32| a + b, 0, xs),
      reduce(|a: i32, b: i32| a + b, 0, ys),
      reduce(|a: i32, b: i32| if a > b then a else b, -2147483648, xs)
    ]
"#,
    )
    .expect("TLC")
    .infer_input_slice_bounds()
    .to_egraph()
    .expect("semantic EGIR");
    let operator_counts: Vec<_> = allocated
        .inner
        .entry_points
        .iter()
        .flat_map(|entry| entry.graph.skeleton.blocks.iter().flat_map(|(_, block)| &block.side_effects))
        .filter_map(|effect| match &effect.kind {
            SideEffectKind::Soac(EgirSoac::Seg {
                kind: SegOpKind::SegRed { operators },
                ..
            }) => Some(operators.len()),
            _ => None,
        })
        .collect();
    assert_eq!(
        operator_counts,
        [1, 1, 1],
        "equal-space first/third reductions may not leapfrog the middle effect"
    );
}

#[test]
fn fused_accumulators_preserve_distinct_input_columns() {
    use crate::egir::types::{EgirSoac, SegOpKind, SideEffectKind};
    let source = r#"
#[compute]
entry e() [2]i32 =
  let xs = map(|i: i32| i + 1, 0i32 ..< 8) in
  let ys = map(|i: i32| i * 2, 0i32 ..< 8) in
  [reduce(|a: i32, b: i32| a + b, 0, xs),
   reduce(|a: i32, b: i32| a + b, 0, ys)]
"#;
    let allocated = crate::compile_thru_tlc(source)
        .expect("TLC")
        .infer_input_slice_bounds()
        .to_egraph()
        .expect("semantic EGIR");
    let operators = allocated
        .inner
        .entry_points
        .iter()
        .flat_map(|entry| entry.graph.skeleton.blocks.iter().flat_map(|(_, block)| &block.side_effects))
        .find_map(|effect| match &effect.kind {
            SideEffectKind::Soac(EgirSoac::Seg {
                kind: SegOpKind::SegRed { operators },
                ..
            }) if operators.len() == 2 => Some(operators),
            _ => None,
        })
        .expect("two-accumulator SegRed");
    assert_ne!(
        operators[0].input_indices, operators[1].input_indices,
        "each accumulator keeps its own source column"
    );
    allocated
        .lower_to_ssa(crate::LoweringProfile::PORTABLE)
        .expect("distinct routed inputs lower to SSA")
        .lower()
        .expect("distinct routed inputs lower to SPIR-V");
    let base: Vec<i32> = (0..8).collect();
    let xs = crate::egir::semantic_exec::map(&base, |value| value + 1);
    let ys = crate::egir::semantic_exec::map(&base, |value| value * 2);
    assert_eq!(
        [
            crate::egir::semantic_exec::reduce(&xs, 0, |a, b| a + b),
            crate::egir::semantic_exec::reduce(&ys, 0, |a, b| a + b),
        ],
        [36, 56]
    );
}

#[test]
fn logical_manifest_covers_scan_and_filter_scratch() {
    use crate::egir::program::{CompilerResourceKind, ResourceOrigin};
    use crate::egir::types::{EgirSoac, SegOpKind, SideEffectKind};

    let scan = crate::compile_thru_tlc(
        "#[compute] entry prefix(xs: []i32) []i32 = scan(|a: i32, b: i32| a + b, 0, xs)",
    )
    .expect("scan TLC")
    .infer_input_slice_bounds()
    .to_egraph()
    .expect("scan allocation");
    let scan_kinds: std::collections::HashSet<_> = scan
        .logical_resources()
        .iter()
        .filter_map(|resource| match &resource.origin {
            ResourceOrigin::Compiler(compiler) => Some(compiler.kind),
            _ => None,
        })
        .collect();
    assert!(scan_kinds.contains(&CompilerResourceKind::ScanBlockSums));
    assert!(scan_kinds.contains(&CompilerResourceKind::ScanBlockOffsets));
    let scan_resources: Vec<_> = scan
        .inner
        .entry_points
        .iter()
        .flat_map(|entry| entry.graph.skeleton.blocks.iter().flat_map(|(_, block)| &block.side_effects))
        .find_map(|effect| match &effect.kind {
            SideEffectKind::Soac(EgirSoac::Seg {
                kind: SegOpKind::SegScan { .. },
                scratch_resources,
                resources,
                ..
            }) => Some((scratch_resources.clone(), resources.clone())),
            _ => None,
        })
        .map(|(ids, accesses)| {
            assert_eq!(ids.len(), 2);
            ids.into_iter()
                .map(|id| {
                    let binding = scan.inner.binding_of(id);
                    assert!(accesses.iter().any(|access| access.binding == binding));
                    binding
                })
                .collect()
        })
        .expect("semantic SegScan");
    assert_eq!(scan_resources.len(), 2);

    let filter = crate::compile_thru_tlc(
        "#[compute] entry evens(xs: []i32) []i32 = filter(|x: i32| x % 2 == 0, xs)",
    )
    .expect("filter TLC")
    .infer_input_slice_bounds()
    .to_egraph()
    .expect("filter allocation");
    let filter_kinds: std::collections::HashSet<_> = filter
        .logical_resources()
        .iter()
        .filter_map(|resource| match &resource.origin {
            ResourceOrigin::Compiler(compiler) => Some(compiler.kind),
            _ => None,
        })
        .collect();
    assert!(filter_kinds.contains(&CompilerResourceKind::FilterScratch));
    assert!(filter_kinds.contains(&CompilerResourceKind::FilterLenCell));
    assert!(filter_kinds.contains(&CompilerResourceKind::FilterFlags));
    assert!(filter_kinds.contains(&CompilerResourceKind::FilterOffsets));
    assert!(filter_kinds.contains(&CompilerResourceKind::FilterScanBlockSums));
    assert!(filter_kinds.contains(&CompilerResourceKind::FilterScanBlockOffsets));

    let scalar_handoff = crate::compile_thru_tlc(
        r#"
#[compute]
entry add_sum(xs: []i32) []i32 =
  let total = reduce(|a: i32, b: i32| a + b, 0, xs) in
  map(|x: i32| x + total, xs)
"#,
    )
    .expect("scalar handoff TLC")
    .infer_input_slice_bounds()
    .to_egraph()
    .expect("scalar handoff allocation");
    assert!(scalar_handoff.logical_resources().iter().any(|resource| {
        matches!(
            &resource.origin,
            ResourceOrigin::Compiler(compiler)
                if compiler.kind == CompilerResourceKind::ScalarHandoff
        )
    }));
}

#[test]
fn runtime_filter_lowers_to_flag_scan_scatter_pipeline() {
    use crate::builtins::catalog;
    use crate::egir::parallelize::schedule::KernelDomain;
    use crate::op::OpTag;
    use crate::ssa::types::InstKind;

    let r4 = r#"
#[compute]
entry r(xs: []u32) ?k. [k]u32 = filter(|x| x < 100u32, xs)
"#;
    let converted = crate::compile_thru_ssa(r4).expect("runtime filter reaches SSA");
    let phases: Vec<_> = converted.kernel_schedule.phases().collect();
    assert_eq!(phases.len(), 5);
    assert_eq!(phases[0].entry_point, "r_filter_flags");
    assert_eq!(phases[1].entry_point, "r_filter_scan");
    assert_eq!(phases[2].entry_point, "r_filter_scan_phase2_scan_sums");
    assert_eq!(phases[3].entry_point, "r_filter_scan_phase3_add_offsets");
    assert_eq!(phases[4].entry_point, "r");
    assert!(matches!(phases[0].domain, KernelDomain::Elements(_)));
    // The scan runs a fixed worker grid so each worker scans a large chunk;
    // dispatching it per-element (Elements) would collapse the chunk to 1 and
    // leave phase-2 an O(len) serial scan.
    assert!(matches!(
        phases[1].domain,
        KernelDomain::Fixed {
            x: crate::egir::parallelize::FILTER_SCAN_GROUPS,
            y: 1,
            z: 1
        }
    ));
    assert_eq!(
        phases[1].workgroup_size,
        (crate::egir::parallelize::REDUCE_PHASE1_WIDTH, 1, 1)
    );
    assert!(matches!(
        phases[2].domain,
        KernelDomain::Fixed { x: 1, y: 1, z: 1 }
    ));
    assert!(matches!(phases[3].domain, KernelDomain::Elements(_)));
    assert!(matches!(phases[4].domain, KernelDomain::Elements(_)));

    let thread_id = catalog().known().thread_id;
    for name in [
        "r_filter_flags",
        "r_filter_scan",
        "r_filter_scan_phase3_add_offsets",
        "r",
    ] {
        let entry = converted
            .ssa
            .entry_points
            .iter()
            .find(|entry| entry.name == name)
            .unwrap_or_else(|| panic!("missing filter phase {name}"));
        assert!(entry.body.inner.blocks.iter().any(|(_, block)| {
            block.insts.iter().any(|&instruction| {
                matches!(
                    &entry.body.get_inst(instruction).data,
                    InstKind::Op { tag: OpTag::Intrinsic { id, .. }, .. } if *id == thread_id
                )
            })
        }));
    }
    crate::compile_thru_spirv(r4).expect("three-stage filter emits SPIR-V");

    let r5 = r#"
#[compute]
entry r(xs: []u32) (?k. [k]u32, [1]u32) =
  let v = filter(|x| x < 100u32, xs)
  let n = length(v) in
  (v, [u32(n)])
"#;
    let lowered = crate::compile_thru_spirv(r5).expect("filter count path emits SPIR-V");
    let compute = lowered
        .pipeline
        .pipelines
        .iter()
        .find_map(|pipeline| match pipeline {
            crate::pipeline_descriptor::Pipeline::Compute(compute) => Some(compute),
            _ => None,
        })
        .expect("filter compute pipeline");
    assert_eq!(compute.stages.len(), 5);
    assert_eq!(compute.stages[1].entry_point, "r_filter_scan");
    assert_eq!(compute.stages[2].entry_point, "r_filter_scan_phase2_scan_sums");
    assert_eq!(compute.stages[3].entry_point, "r_filter_scan_phase3_add_offsets");
}

#[test]
fn widened_filter_output_uses_output_element_size() {
    use crate::pipeline_descriptor::{BufferLen, BufferUsage, Pipeline};

    let lowered = crate::compile_thru_spirv(
        r#"
#[compute]
entry r(bidx: []u32) ?k. [k]vec4f32 =
  let cand = map(|s| let i = i32(s) in @[f32(i), 0.0, f32(i), 1.0], bidx) in
  filter(|c| c.z > 0.0, cand)
"#,
    )
    .expect("widening map-filter compiles");
    let output_length = lowered.pipeline.pipelines.iter().find_map(|pipeline| match pipeline {
        Pipeline::Compute(compute) => compute.bindings.iter().find_map(|binding| match binding {
            crate::pipeline_descriptor::Binding::StorageBuffer {
                usage: BufferUsage::Output,
                length,
                ..
            } => length.as_ref(),
            _ => None,
        }),
        _ => None,
    });
    assert_eq!(
        output_length,
        Some(&BufferLen::LikeInput {
            set: 0,
            binding: 0,
            elem_bytes: 16,
            src_elem_bytes: 4,
        })
    );
}

/// Characterizes which multi-consumer array shapes are resolved before the EGIR
/// semantic layer and which survive as a genuine multi-consumer producer (a
/// SegMap result with >=2 value consumers in the semantic dependency DAG).
///
/// Most shapes are subsumed upstream: same-domain sibling consumers fuse (TLC
/// `fuse_maps` + EGIR horizontal fusion), and point reads / nested captures are
/// materialized. `reduce_then_map` — an array consumed by a reduce *and* by a
/// later map that depends on the reduce's scalar result — cannot fuse (a true
/// producer→consumer dependency forces the map after the reduce). Phase M
/// therefore turns that producer into one logical storage prepass, and both
/// consumers read the same buffer. This test pins that no live multi-consumer
/// SegMap remains and that the shared prepass replaces local `Materialize`s.
#[test]
fn multi_consumer_producer_survival_is_characterized() {
    use crate::egir::program::SemanticDependencyKind;
    use crate::egir::types::{EgirSoac, SegOpKind, SideEffectKind};
    use std::collections::HashMap;

    fn multi_consumer_producers(src: &str) -> usize {
        let tlc = crate::compile_thru_tlc(src).expect("multi-consumer program should reach TLC");
        let allocated = tlc
            .infer_input_slice_bounds()
            .to_egraph()
            .expect("multi-consumer program should reach semantic EGIR");
        let seg_maps: std::collections::HashSet<_> = allocated
            .inner
            .entry_points
            .iter()
            .flat_map(|entry| {
                entry.graph.skeleton.blocks.iter().flat_map(move |(_, block)| {
                    block.side_effects.iter().filter_map(move |effect| match &effect.kind {
                        SideEffectKind::Soac(EgirSoac::Seg {
                            kind: SegOpKind::SegMap,
                            ..
                        }) => effect.result.map(|result| crate::egir::program::SemanticOpId {
                            scope: entry.name.clone(),
                            result,
                        }),
                        _ => None,
                    })
                })
            })
            .collect();
        let mut consumers: HashMap<_, usize> = HashMap::new();
        for dep in &allocated.inner.semantic_dependencies {
            if matches!(dep.kind, SemanticDependencyKind::Value) && seg_maps.contains(&dep.producer) {
                *consumers.entry(dep.producer.clone()).or_default() += 1;
            }
        }
        consumers.values().filter(|count| **count >= 2).count()
    }

    // ys read by two elementwise maps (combined via zip).
    let two_maps = r#"
def N: i32 = 8
#[compute]
entry e() [8]i32 =
    let ys = map(|i: i32| i + 1, 0i32 ..< N) in
    let a = map(|y: i32| y * 2, ys) in
    let b = map(|y: i32| y + 100, ys) in
    map(|p: (i32, i32)| p.0 + p.1, zip(a, b))
"#;
    // ys read by a full reduce and then by a map (different consumer kinds).
    let reduce_then_map = r#"
def N: i32 = 8
#[compute]
entry e() [8]i32 =
    let ys = map(|i: i32| i + 1, 0i32 ..< N) in
    let s = reduce(|a: i32, b: i32| a + b, 0, ys) in
    map(|y: i32| y + s, ys)
"#;
    // ys read by two reductions with different operators (sum and max).
    let two_reduces = r#"
def N: i32 = 8
#[compute]
entry e() i32 =
    let ys = map(|i: i32| i * i, 0i32 ..< N) in
    reduce(|a: i32, b: i32| a + b, 0, ys) * reduce(|a: i32, b: i32| if a > b then a else b, 0, ys)
"#;
    // ys read by a full reduce and by a point index.
    let reduce_and_index = r#"
def N: i32 = 8
#[compute]
entry e() i32 =
    let ys = map(|i: i32| i * i, 0i32 ..< N) in
    reduce(|a: i32, b: i32| a + b, 0, ys) + ys[0]
"#;
    // ys reduced inside a map over a *different* domain — read once per outer
    // iteration; the classic "materialize once, reuse" case.
    let reduce_in_nested_map = r#"
def N: i32 = 8
#[compute]
entry e() [4]i32 =
    let ys = map(|i: i32| i + 1, 0i32 ..< N) in
    map(|j: i32| reduce(|a: i32, b: i32| a + b, 0, ys) + j, 0i32 ..< 4)
"#;

    let survivors: Vec<(&str, usize)> = [
        ("two_maps", two_maps),
        ("reduce_then_map", reduce_then_map),
        ("two_reduces", two_reduces),
        ("reduce_and_index", reduce_and_index),
        ("reduce_in_nested_map", reduce_in_nested_map),
    ]
    .into_iter()
    .map(|(name, src)| (name, multi_consumer_producers(src)))
    .collect();

    // Every shape is now either fused or represented by one shared logical
    // materialization; no multi-consumer SegMap remains after allocation.
    assert_eq!(
        survivors,
        vec![
            ("two_maps", 0),
            ("reduce_then_map", 0),
            ("two_reduces", 0),
            ("reduce_and_index", 0),
            ("reduce_in_nested_map", 0),
        ],
        "multi-consumer subsumption boundary moved — Phase M scope changed"
    );

    let allocated = crate::compile_thru_tlc(reduce_then_map)
        .expect("TLC")
        .infer_input_slice_bounds()
        .to_egraph()
        .expect("allocated EGIR");
    use crate::egir::program::{CompilerResourceKind, ResourceOrigin};
    let shared: Vec<_> = allocated
        .logical_resources()
        .iter()
        .filter(|resource| {
            matches!(
                &resource.origin,
                ResourceOrigin::Compiler(compiler)
                    if compiler.kind == CompilerResourceKind::MultiConsumerArray
            )
        })
        .collect();
    assert_eq!(
        shared.len(),
        1,
        "the surviving producer owns one shared logical buffer"
    );
    let shared_binding = shared[0].legacy_binding;
    let lowered =
        allocated.lower_to_ssa(crate::LoweringProfile::PORTABLE).expect("shared materialization lowers");
    let mir = crate::ssa::print::format_program(&lowered.ssa);
    assert_eq!(
        mir.matches("materialize ").count(),
        0,
        "consumers read the shared storage prepass rather than copying a composite per consumer"
    );
    let stages: Vec<_> = lowered.kernel_schedule.phases().map(|phase| phase.entry_point.as_str()).collect();
    assert_eq!(stages, ["e_materialize_shared", "e"]);
    let phases: Vec<_> = lowered.kernel_schedule.phases().collect();
    assert!(phases[0].resources.iter().any(|resource| {
        resource.binding == shared_binding
            && resource.access == crate::egir::parallelize::schedule::ResourceAccess::Write
    }));
    assert!(phases[1]
        .resources
        .iter()
        .any(|resource| resource.binding == shared_binding && resource.access.reads()));
    assert!(phases[1].dependencies.contains(&0));
    let second = crate::compile_thru_tlc(reduce_then_map)
        .expect("repeat TLC")
        .infer_input_slice_bounds()
        .to_egraph()
        .expect("repeat allocation")
        .lower_to_ssa(crate::LoweringProfile::PORTABLE)
        .expect("repeat lowering");
    assert_eq!(
        serde_json::to_string(&lowered.pipeline).unwrap(),
        serde_json::to_string(&second.pipeline).unwrap(),
        "shared materialization descriptor is deterministic"
    );
    let ys: Vec<i32> = (0..8).map(|value| value + 1).collect();
    let sum = crate::egir::semantic_exec::reduce(&ys, 0, |a, b| a + b);
    assert_eq!(
        crate::egir::semantic_exec::map(&ys, |value| value + sum),
        [37, 38, 39, 40, 41, 42, 43, 44]
    );

    let single = crate::compile_thru_tlc(reduce_then_map)
        .expect("single-stage TLC")
        .infer_input_slice_bounds()
        .to_egraph()
        .expect("single-stage allocation")
        .lower_to_ssa(crate::LoweringProfile::new(
            crate::CodegenTarget::Portable,
            crate::SchedulePolicy::SingleStage,
        ))
        .expect("single-stage shared materialization");
    let single_phases: Vec<_> = single.kernel_schedule.phases().collect();
    assert_eq!(
        single_phases.len(),
        2,
        "required materialization plus source entry"
    );
    assert!(matches!(
        single_phases[0].domain,
        crate::egir::parallelize::schedule::KernelDomain::Fixed { x: 1, y: 1, z: 1 }
    ));

    let wgsl = crate::compile_thru_tlc(reduce_then_map)
        .expect("WGSL TLC")
        .infer_input_slice_bounds()
        .to_egraph()
        .expect("WGSL allocation")
        .lower_to_ssa(crate::LoweringProfile::new(
            crate::CodegenTarget::Wgsl,
            crate::SchedulePolicy::Parallel,
        ))
        .expect("WGSL-targeted semantic lowering")
        .lower_wgsl()
        .expect("shared materialization lowers to WGSL");
    assert!(wgsl.contains("e_materialize_shared"));
}

#[test]
fn tuple_outputs_with_independent_map_chains_lower_after_semantic_fusion() {
    crate::compile_thru_spirv(
        r#"
def N: i32 = 8
def f(x: i32) i32 = x + 1
def g(x: i32) i32 = x * 2
def h(x: i32) i32 = x - 3
def k(x: i32) i32 = x * 5
#[compute]
entry e() ([8]i32, [8]i32) =
  (map(f, map(g, 0i32 ..< N)), map(h, map(k, 0i32 ..< N)))
"#,
    )
    .expect("two tuple fields with independent map chains lower to SPIR-V");
}

#[test]
fn terminal_schedule_and_descriptor_are_atomic_and_deterministic() {
    let source = r#"
#[compute]
entry sum(xs: []i32) i32 = reduce(|a: i32, b: i32| a + b, 0, xs)
"#;
    let allocated = crate::compile_thru_tlc(source)
        .expect("TLC")
        .infer_input_slice_bounds()
        .to_egraph()
        .expect("semantic allocation");
    for pipeline in &allocated.inner.pipeline.pipelines {
        if let crate::pipeline_descriptor::Pipeline::Compute(compute) = pipeline {
            assert!(
                compute.bindings.is_empty(),
                "bindings publish only at terminal lowering"
            );
        }
    }
    let first = allocated.lower_to_ssa(crate::LoweringProfile::PORTABLE).expect("terminal lowering");
    let phases: Vec<_> = first.kernel_schedule.phases().collect();
    assert!(phases.len() >= 2, "parallel reduction owns at least two phases");
    assert!(phases.iter().skip(1).any(|phase| !phase.dependencies.is_empty()));
    assert!(phases.iter().all(|phase| !phase.resources.is_empty()));

    let second = crate::compile_thru_ssa(source).expect("second lowering");
    assert_eq!(
        serde_json::to_string(&first.pipeline).unwrap(),
        serde_json::to_string(&second.pipeline).unwrap(),
        "descriptor publication is deterministic"
    );
}

#[test]
fn single_stage_is_a_terminal_schedule_policy() {
    let source = r#"
#[compute]
entry sum(xs: []i32) i32 = reduce(|a: i32, b: i32| a + b, 0, xs)
"#;
    let tlc = crate::compile_thru_tlc(source).expect("TLC");
    let allocated = tlc.infer_input_slice_bounds().to_egraph().expect("semantic allocation");
    assert!(allocated.semantic_ir().contains("SegRed"));
    let lowered = allocated
        .lower_to_ssa(crate::LoweringProfile::new(
            crate::CodegenTarget::Portable,
            crate::SchedulePolicy::SingleStage,
        ))
        .expect("single-stage terminal lowering");
    assert_eq!(lowered.kernel_schedule.phases().count(), 1);
    assert!(!lowered.ssa.entry_points.iter().any(|entry| entry.name.contains("phase2")));
}

#[test]
fn target_profiles_are_selected_before_ssa_lowering() {
    let portable = crate::compile_thru_ssa("#[compute] entry e(xs: []i32) []i32 = map(|x: i32| x + 1, xs)")
        .expect("portable SSA");
    assert_eq!(portable.profile.target, crate::CodegenTarget::Portable);

    let spirv = crate::compile_thru_spirv("#[compute] entry e(xs: []i32) []i32 = map(|x: i32| x + 1, xs)")
        .expect("SPIR-V-targeted lowering");
    assert!(!spirv.spirv.is_empty());
}

#[test]
fn terminal_scan_helpers_are_complete_region_arena_members() {
    let tlc = crate::compile_thru_tlc(
        "#[compute] entry prefix(xs: []i32) []i32 = scan(|a: i32, b: i32| a + b, 0, xs)",
    )
    .expect("TLC");
    let mut allocated = tlc.infer_input_slice_bounds().to_egraph().expect("semantic EGIR");
    crate::egir::target_lowering::schedule(
        &mut allocated.inner,
        &mut allocated.binding_ids,
        crate::LoweringProfile::PORTABLE,
    )
    .expect("terminal schedule");
    let helper = allocated
        .inner
        .functions
        .iter()
        .find(|function| function.name.ends_with("_scan_op_swap"))
        .expect("scan swap helper");
    let region = allocated.inner.region_interner.get(&helper.name).expect("helper region id");
    assert!(allocated.inner.regions.contains_key(&region));
}

/// Assert that a compute `reduce`-over-`map`-of-range `src` parallelizes and
/// that phase1's per-thread loop trip-count transitively depends on
/// `thread_id` — i.e. each thread reduces only its *chunk* of the range.
///
/// The bug: phase1 looped the full `0..n` per invocation (quadratic — every
/// thread redoes the whole reduction; `thread_id` was used only as the
/// partials output slot, never to bound the loop). With the bug the
/// trip-count is the raw input length `n` (thread-independent), so
/// `thread_id` is unreachable from the loop condition and this fails.
fn assert_phase1_loop_depends_on_thread_id(src: &str) {
    use crate::builtins::catalog;
    use crate::op::OpTag;
    use crate::ssa::types::{ControlHeader, FuncBody, InstKind, Terminator, ValueId};
    use std::collections::{HashMap, HashSet, VecDeque};

    let program = compile_to_ssa(src);
    let thread_id_builtin = catalog().known().thread_id;

    // The phase1 entry is the parallelized worker — the one that reads
    // `thread_id` (the per-thread partials slot). phase2 is single-threaded.
    // (The EGIR reduce path mutates the original entry in place, so phase1
    // keeps the source entry name rather than gaining a `_phase1` suffix.)
    let has_thread_id = |body: &FuncBody| -> bool {
        body.inner.blocks.iter().any(|(_, block)| {
            block.insts.iter().any(|&i| {
                matches!(&body.get_inst(i).data,
                    InstKind::Op { tag: OpTag::Intrinsic { id, .. }, .. } if *id == thread_id_builtin)
            })
        })
    };
    let phase1 = program.entry_points.iter().find(|e| has_thread_id(&e.body)).unwrap_or_else(|| {
        panic!(
            "expected a parallelized phase1 entry (one using thread_id); entries: {:?}",
            program.entry_points.iter().map(|e| e.name.clone()).collect::<Vec<_>>()
        )
    });
    let body = &phase1.body;

    // The two-phase reduce must have a phase2 that combines the partials into
    // the result — otherwise the partials are written but never reduced (an
    // incomplete program; the descriptor would reference a phantom entry).
    assert!(
        program.entry_points.iter().any(|e| e.name.contains("phase2") || e.name.contains("combine")),
        "missing phase2 combine entry — partials are never reduced to a result; entries: {:?}",
        program.entry_points.iter().map(|e| e.name.clone()).collect::<Vec<_>>()
    );

    // Map each SSA result to its operand values; locate the `thread_id` result.
    let mut def: HashMap<ValueId, Vec<ValueId>> = HashMap::new();
    let mut thread_id_val: Option<ValueId> = None;
    for (_bid, block) in &body.inner.blocks {
        for &inst_id in &block.insts {
            let inst = body.get_inst(inst_id);
            let Some(result) = inst.result else { continue };
            def.insert(result, inst.data.ssa_uses());
            if let InstKind::Op {
                tag: OpTag::Intrinsic { id, .. },
                ..
            } = &inst.data
            {
                if *id == thread_id_builtin {
                    thread_id_val = Some(result);
                }
            }
        }
    }
    let thread_id_val = thread_id_val.expect("phase1 must compute thread_id");

    // Loop-header condition value(s).
    let cond_vals: Vec<ValueId> = body
        .inner
        .blocks
        .iter()
        .filter(|(bid, _)| matches!(body.control_headers.get(bid), Some(ControlHeader::Loop { .. })))
        .filter_map(|(_, block)| match &block.term {
            Terminator::CondBranch { cond, .. } => Some(*cond),
            _ => None,
        })
        .collect();
    assert!(!cond_vals.is_empty(), "phase1 must contain a loop");

    // Is `thread_id` reachable from a loop condition via def→operand edges?
    let reaches_tid = |start: ValueId| -> bool {
        let mut seen = HashSet::new();
        let mut q = VecDeque::from([start]);
        while let Some(v) = q.pop_front() {
            if v == thread_id_val {
                return true;
            }
            if seen.insert(v) {
                if let Some(ops) = def.get(&v) {
                    q.extend(ops.iter().copied());
                }
            }
        }
        false
    };
    assert!(
        cond_vals.iter().any(|&c| reaches_tid(c)),
        "phase1's loop trip-count is independent of thread_id — every thread \
         reduces the full input (quadratic) instead of a per-thread chunk"
    );

    // End-to-end: the parallelized program (including phase2 reducing the
    // partials and storing the result) must still reach SPIR-V.
    compile_to_spirv(src).expect("parallelized reduce-over-range must compile to SPIR-V");
}

/// Baseline: a scalar reduce over a mapped range chunks phase1 correctly.
#[test]
fn compute_scalar_reduce_over_range_chunks_phase1() {
    assert_phase1_loop_depends_on_thread_id(
        r#"
#[compute]
entry mn(n: u32) u32 =
  let cands = map(|i: u32| i * 2654435761u32, 0u32..<n) in
  reduce(|a: u32, b: u32| if a < b then a else b, 4294967295u32, cands)
"#,
    );
}

/// The miner's shape: a reduce whose element is an AoS `(scalar, array)`
/// tuple. Routed (like scalars) through the EGIR Screma reduce
/// chunking — phase 1 chunks the range and phase 2 combines the
/// partials.
#[test]
fn compute_soa_tuple_reduce_over_range_chunks_phase1() {
    assert_phase1_loop_depends_on_thread_id(
        r#"
#[compute]
entry mn(n: u32) (u32, [4]u32) =
  let cands = map(|i: u32| (i, [i, i, i, i]), 0u32..<n) in
  reduce(
    |a: (u32, [4]u32), b: (u32, [4]u32)| if a.0 < b.0 then a else b,
    (4294967295u32, [0u32, 0u32, 0u32, 0u32]),
    cands)
"#,
    );
}

/// A reduce whose element is a tuple is decomposed across one output buffer per
/// field. Phase 1 must store the whole accumulator tuple to its partials buffer,
/// and phase 2 must write *every* output field from the combined result.
/// Regression: the lowering kept only the first field's store (writing a scalar
/// into the tuple-typed partials → invalid SPIR-V) and dropped the array field's
/// output entirely. `compile_to_spirv` alone doesn't catch it — the malformed
/// store only fails `spirv-val` (see `testfiles/miner.wyn`); this asserts the
/// descriptor invariant that no output buffer is left unwritten.
#[test]
fn tuple_reduce_writes_every_output_field() {
    use crate::pipeline_descriptor::{Binding, BufferUsage, Pipeline};
    let lowered = crate::compile_thru_spirv(
        r#"
#[compute]
entry mn(n: u32) (u32, [4]u32) =
  let cands = map(|i: u32| (i, [i, i, i, i]), 0u32..<n) in
  reduce(
    |a: (u32, [4]u32), b: (u32, [4]u32)| if a.0 < b.0 then a else b,
    (4294967295u32, [0u32, 0u32, 0u32, 0u32]),
    cands)
"#,
    )
    .expect("tuple-element reduce compiles");
    let cp = lowered
        .pipeline
        .pipelines
        .iter()
        .find_map(|p| match p {
            Pipeline::Compute(c) => Some(c),
            _ => None,
        })
        .expect("one compute pipeline");
    assert!(
        cp.stages.iter().any(|s| s.entry_point.contains("phase2_combine")),
        "a parallel reduce splits into phase 1 + phase2_combine"
    );
    let output_indices: Vec<usize> = cp
        .bindings
        .iter()
        .enumerate()
        .filter_map(|(i, b)| {
            matches!(
                b,
                Binding::StorageBuffer {
                    usage: BufferUsage::Output,
                    ..
                }
            )
            .then_some(i)
        })
        .collect();
    assert_eq!(
        output_indices.len(),
        2,
        "the tuple `(u32, [4]u32)` yields two output buffers"
    );
    for idx in output_indices {
        assert!(
            cp.stages.iter().any(|s| s.writes.contains(&idx)),
            "output buffer #{idx} must be written by some stage; \
             stage writes = {:?}",
            cp.stages.iter().map(|s| &s.writes).collect::<Vec<_>>()
        );
    }
}

/// The reduce phase2 is a workgroup-parallel tree reduce: a `LocalSize(W,1,1)`
/// `*_phase2_combine` entry that uses `local_id`, a workgroup-shared array, and
/// `ControlBarrier`s — not a single-threaded combine loop.
#[test]
fn phase2_reduce_is_workgroup_parallel_tree() {
    use crate::builtins::catalog;
    use crate::op::{OpTag, PureViewSource};
    use crate::ssa::types::{ExecutionModel, InstKind};

    let program = compile_to_ssa(
        r#"
#[compute]
entry sum(xs: []f32) f32 =
  reduce(|a: f32, b: f32| a + b, 0.0, xs)
"#,
    );

    let phase2 = program
        .entry_points
        .iter()
        .find(|e| e.name.ends_with("_phase2_combine"))
        .expect("a *_phase2_combine entry");

    match &phase2.execution_model {
        ExecutionModel::Compute { local_size } => {
            assert_eq!(local_size.0, 256);
            assert_eq!((local_size.1, local_size.2), (1, 1));
        }
        other => panic!("phase2 not compute: {:?}", other),
    }

    let body = &phase2.body;
    let insts = || body.inner.blocks.iter().flat_map(|(_, b)| b.insts.iter().map(|&i| body.get_inst(i)));

    let barriers = insts().filter(|n| matches!(n.data, InstKind::ControlBarrier)).count();
    assert_eq!(barriers, 2, "grid-stride + tree-step barriers");

    assert!(
        insts().any(|n| matches!(
            &n.data,
            InstKind::Op {
                tag: OpTag::StorageView(PureViewSource::Workgroup { .. }),
                ..
            }
        )),
        "phase2 must use a workgroup-shared array"
    );

    let local_id = catalog().known().local_id;
    assert!(
        insts().any(|n| matches!(
            &n.data,
            InstKind::Op { tag: OpTag::Intrinsic { id, .. }, .. } if *id == local_id
        )),
        "phase2 must read local_id"
    );
}

#[test]
fn parallel_reduce_descriptor_wires_partials_and_original_output() {
    use crate::pipeline_descriptor::{Binding, BufferLen, BufferUsage, Pipeline};
    let lowered = crate::compile_thru_spirv(
        r#"
#[compute]
entry sum(xs: []f32) f32 = reduce(|a: f32, b: f32| a + b, 0.0, xs)
"#,
    )
    .expect("parallel reduction compiles");
    let compute = lowered
        .pipeline
        .pipelines
        .iter()
        .find_map(|pipeline| match pipeline {
            Pipeline::Compute(compute) if compute.stages.iter().any(|stage| stage.entry_point == "sum") => {
                Some(compute)
            }
            _ => None,
        })
        .expect("sum compute pipeline");
    assert_eq!(compute.stages.len(), 2, "phase 1 plus phase 2");
    let partial_index = compute
        .bindings
        .iter()
        .position(|binding| {
            matches!(
                binding,
                Binding::StorageBuffer {
                    usage: BufferUsage::Intermediate,
                    length: Some(BufferLen::SameAsDispatch { .. }),
                    ..
                }
            )
        })
        .expect("dispatch-sized partial buffer is published");
    let output_index = compute
        .bindings
        .iter()
        .position(|binding| {
            matches!(
                binding,
                Binding::StorageBuffer {
                    usage: BufferUsage::Output,
                    name,
                    ..
                } if name == "sum_output"
            )
        })
        .expect("original host output remains published");
    assert!(compute.stages[0].writes.contains(&partial_index));
    assert!(compute.stages[1].reads.contains(&partial_index));
    assert!(compute.stages[1].writes.contains(&output_index));
}

#[test]
fn mapped_reduce_with_phase1_capture_stays_parallel() {
    use crate::pipeline_descriptor::Pipeline;
    let lowered = crate::compile_thru_spirv(
        r#"
#[compute]
entry scaled_sum(xs: []i32, scale: i32) i32 =
  reduce(|a: i32, b: i32| a + b, 0, map(|x: i32| x * scale, xs))
"#,
    )
    .expect("capturing mapped reduction compiles");
    let stages = lowered.pipeline.pipelines.iter().find_map(|pipeline| match pipeline {
        Pipeline::Compute(compute)
            if compute.stages.iter().any(|stage| stage.entry_point == "scaled_sum") =>
        {
            Some(&compute.stages)
        }
        _ => None,
    });
    assert_eq!(stages.expect("scaled_sum pipeline").len(), 2);
}

/// Output sizing (review finding #2): `build_entry_outputs` now sizes a runtime
/// output to the dispatch domain (`SameAsDispatch`) per *output type*
/// (`ty.is_array()`) instead of a per-*entry* `dispatch_sized` flag. A reduction
/// returns a scalar, so its output buffer must NOT be dispatch-sized — that rule
/// is only for one-element-per-thread map/scan arrays. (No source construct
/// currently yields a reduction whose result is a runtime-sized array; if one is
/// added, this is where its sizing must be pinned.)
#[test]
fn reduce_scalar_output_is_not_dispatch_sized() {
    use crate::pipeline_descriptor::{Binding, BufferLen, BufferUsage, Pipeline};
    let lowered = crate::compile_thru_spirv(
        r#"
#[compute]
entry total(xs: []i32) i32 = reduce(|a: i32, b: i32| a + b, 0, xs)
"#,
    )
    .expect("scalar reduction compiles");
    let compute = lowered
        .pipeline
        .pipelines
        .iter()
        .find_map(|pipeline| match pipeline {
            Pipeline::Compute(compute)
                if compute.stages.iter().any(|stage| stage.entry_point == "total") =>
            {
                Some(compute)
            }
            _ => None,
        })
        .expect("total compute pipeline");
    let output_lengths: Vec<_> = compute
        .bindings
        .iter()
        .filter_map(|binding| match binding {
            Binding::StorageBuffer {
                usage: BufferUsage::Output,
                length,
                ..
            } => Some(length.clone()),
            _ => None,
        })
        .collect();
    assert!(!output_lengths.is_empty(), "reduce entry has an output buffer");
    for length in output_lengths {
        assert!(
            !matches!(length, Some(BufferLen::SameAsDispatch { .. })),
            "scalar reduction output must not be dispatch-sized, got {length:?}"
        );
    }
}

#[test]
fn parallel_scan_descriptor_wires_three_phases_and_scratch() {
    use crate::pipeline_descriptor::{Binding, BufferLen, BufferUsage, Pipeline};
    let lowered = crate::compile_thru_spirv(
        r#"
#[compute]
entry prefix(xs: []i32) []i32 = scan(|a: i32, b: i32| a + b, 0, xs)
"#,
    )
    .expect("parallel scan compiles");
    let compute = lowered
        .pipeline
        .pipelines
        .iter()
        .find_map(|pipeline| match pipeline {
            Pipeline::Compute(compute)
                if compute.stages.iter().any(|stage| stage.entry_point == "prefix") =>
            {
                Some(compute)
            }
            _ => None,
        })
        .expect("prefix compute pipeline");
    assert_eq!(
        compute.stages.len(),
        3,
        "chunk scan, exclusive block scan, offset application"
    );
    let scratch: Vec<usize> = compute
        .bindings
        .iter()
        .enumerate()
        .filter_map(|(index, binding)| {
            matches!(
                binding,
                Binding::StorageBuffer {
                    usage: BufferUsage::Intermediate,
                    length: Some(BufferLen::SameAsDispatch { .. }),
                    ..
                }
            )
            .then_some(index)
        })
        .collect();
    assert_eq!(scratch.len(), 2, "block sums and exclusive block offsets");
    for index in scratch {
        assert!(compute.stages.iter().any(|stage| stage.writes.contains(&index)));
        assert!(compute.stages.iter().any(|stage| stage.reads.contains(&index)));
    }
}

#[test]
fn mapped_scan_with_phase1_capture_stays_parallel() {
    use crate::pipeline_descriptor::Pipeline;
    let lowered = crate::compile_thru_spirv(
        r#"
#[compute]
entry scaled_prefix(xs: []i32, scale: i32) []i32 =
  scan(|a: i32, b: i32| a + b, 0, map(|x: i32| x * scale, xs))
"#,
    )
    .expect("capturing mapped scan compiles");
    let stages = lowered.pipeline.pipelines.iter().find_map(|pipeline| match pipeline {
        Pipeline::Compute(compute)
            if compute.stages.iter().any(|stage| stage.entry_point == "scaled_prefix") =>
        {
            Some(&compute.stages)
        }
        _ => None,
    });
    assert_eq!(stages.expect("scaled_prefix pipeline").len(), 3);
}

#[test]
fn range_map_dispatch_uses_range_length() {
    use crate::pipeline_descriptor::{DispatchLen, DispatchSize, Pipeline};
    let lowered = crate::compile_thru_spirv(
        r#"
#[compute]
entry generated() []i32 = map(|i: i32| i + 1, 0i32..<2048)
"#,
    )
    .expect("range map compiles");
    let stage = lowered
        .pipeline
        .pipelines
        .iter()
        .find_map(|pipeline| match pipeline {
            Pipeline::Compute(compute) => {
                compute.stages.iter().find(|stage| stage.entry_point == "generated")
            }
            _ => None,
        })
        .expect("generated stage");
    assert_eq!(
        stage.dispatch_size,
        DispatchSize::DerivedFrom {
            len: DispatchLen::Fixed { count: 2048 },
            workgroup_size: 64,
        }
    );
}

#[test]
fn scalar_prepass_and_consumer_share_one_scheduled_pipeline() {
    use crate::pipeline_descriptor::Pipeline;
    let lowered = crate::compile_thru_spirv(
        r#"
#[compute]
entry add_sum(xs: []i32) []i32 =
  let total = reduce(|a: i32, b: i32| a + b, 0, xs) in
  map(|x: i32| x + total, xs)
"#,
    )
    .expect("scalar prepass feeding a map compiles");

    let matching: Vec<_> = lowered
        .pipeline
        .pipelines
        .iter()
        .filter_map(|pipeline| match pipeline {
            Pipeline::Compute(compute)
                if compute.stages.iter().any(|stage| {
                    stage.entry_point == "add_sum" || stage.entry_point.starts_with("add_sum_prepass_")
                }) =>
            {
                Some(compute)
            }
            _ => None,
        })
        .collect();
    assert_eq!(
        matching.len(),
        1,
        "compiler producer and consumer must share one binding table"
    );
    let stages = &matching[0].stages;
    assert_eq!(
        stages.len(),
        3,
        "two reduction phases followed by the map consumer"
    );
    assert!(stages[0].entry_point.starts_with("add_sum_prepass_"));
    assert!(stages[1].entry_point.contains("phase2"));
    assert_eq!(stages[2].entry_point, "add_sum");
    let handoff = stages[1]
        .writes
        .iter()
        .copied()
        .find(|binding| stages[2].reads.contains(binding))
        .expect("phase 2 result feeds the map consumer");
    assert!(
        !stages[0].writes.contains(&handoff),
        "phase 1 writes partials, not the final scalar handoff"
    );
}

// =============================================================================
// Phase 2 regression: unbound `Var(Symbol(sym))` references through TLC passes
// =============================================================================

/// Walk a term and assert that every `TermKind::Var(VarRef::Symbol(sym))`
/// references a sym that is either:
/// - bound by an enclosing Let / Lambda param / Loop var / SOAC element
///   parameter, or
/// - a top-level def name in `top_level`.
///
/// On violation, panics with the offending sym, its symbol-table name,
/// and the pipeline stage name.
fn assert_no_unbound_var_refs(program: &crate::tlc::Program, stage: &str) {
    use crate::tlc::{ArrayExpr, Lambda, LoopKind, SoacOp, Term, TermKind};
    use crate::SymbolId;
    use std::collections::HashSet;

    fn walk(term: &Term, bound: &HashSet<SymbolId>, symbols: &SymbolTable, stage: &str, def_name: &str) {
        match &term.kind {
            TermKind::Var(VarRef::Symbol(sym)) => {
                assert!(
                    bound.contains(sym),
                    "[{stage}] def `{def_name}`: unbound Var(sym{:?}) name={:?}",
                    sym.0,
                    symbols.get(*sym)
                );
            }
            TermKind::Var(VarRef::Builtin { .. })
            | TermKind::BinOp(_)
            | TermKind::UnOp(_)
            | TermKind::IntLit(_)
            | TermKind::FloatLit(_)
            | TermKind::BoolLit(_)
            | TermKind::UnitLit
            | TermKind::Extern(_) => {}
            TermKind::Coerce { inner, .. } => walk(inner, bound, symbols, stage, def_name),
            TermKind::App { func, args } => {
                walk(func, bound, symbols, stage, def_name);
                for a in args {
                    walk(a, bound, symbols, stage, def_name);
                }
            }
            TermKind::Lambda(Lambda { params, body, .. }) => {
                let mut inner = bound.clone();
                for (p, _) in params {
                    inner.insert(*p);
                }
                walk(body, &inner, symbols, stage, def_name);
            }
            TermKind::Let { name, rhs, body, .. } => {
                walk(rhs, bound, symbols, stage, def_name);
                let mut inner = bound.clone();
                inner.insert(*name);
                walk(body, &inner, symbols, stage, def_name);
            }
            TermKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                walk(cond, bound, symbols, stage, def_name);
                walk(then_branch, bound, symbols, stage, def_name);
                walk(else_branch, bound, symbols, stage, def_name);
            }
            TermKind::Loop {
                loop_var,
                init,
                init_bindings,
                kind,
                body,
                ..
            } => {
                walk(init, bound, symbols, stage, def_name);
                for (_, _, e) in init_bindings {
                    walk(e, bound, symbols, stage, def_name);
                }
                match kind {
                    LoopKind::For { iter, .. } => walk(iter, bound, symbols, stage, def_name),
                    LoopKind::ForRange { bound: bnd, .. } => {
                        walk(bnd, bound, symbols, stage, def_name);
                    }
                    LoopKind::While { cond } => walk(cond, bound, symbols, stage, def_name),
                }
                let mut inner = bound.clone();
                inner.insert(*loop_var);
                if let LoopKind::For { var, .. } | LoopKind::ForRange { var, .. } = kind {
                    inner.insert(*var);
                }
                for (n, _, _) in init_bindings {
                    inner.insert(*n);
                }
                walk(body, &inner, symbols, stage, def_name);
            }
            TermKind::Soac(soac) => walk_soac(soac, bound, symbols, stage, def_name),
            TermKind::ArrayExpr(ae) => walk_array_expr(ae, bound, symbols, stage, def_name),

            TermKind::Tuple(parts) | TermKind::VecLit(parts) => {
                for p in parts {
                    walk(p, bound, symbols, stage, def_name);
                }
            }
            TermKind::TupleProj { tuple, .. } => walk(tuple, bound, symbols, stage, def_name),
            TermKind::Index { array, index } => {
                walk(array, bound, symbols, stage, def_name);
                walk(index, bound, symbols, stage, def_name);
            }
            TermKind::OutputSlotStore { value, .. } => walk(value, bound, symbols, stage, def_name),
        }
    }

    fn walk_lambda(
        lam: &crate::tlc::Lambda,
        bound: &HashSet<SymbolId>,
        symbols: &SymbolTable,
        stage: &str,
        def_name: &str,
    ) {
        let mut inner = bound.clone();
        for (p, _) in &lam.params {
            inner.insert(*p);
        }
        walk(&lam.body, &inner, symbols, stage, def_name);
    }

    fn walk_soac(
        soac: &SoacOp,
        bound: &HashSet<SymbolId>,
        symbols: &SymbolTable,
        stage: &str,
        def_name: &str,
    ) {
        match soac {
            SoacOp::Map { lam, inputs, .. } => {
                for i in inputs {
                    walk_array_expr(i, bound, symbols, stage, def_name);
                }
                walk_lambda(&lam.lam, bound, symbols, stage, def_name);
            }
            SoacOp::Reduce { op, ne, input, .. } => {
                walk(ne, bound, symbols, stage, def_name);
                walk_array_expr(input, bound, symbols, stage, def_name);
                walk_lambda(&op.lam, bound, symbols, stage, def_name);
            }
            SoacOp::Scan { op, ne, input, .. } => {
                walk(ne, bound, symbols, stage, def_name);
                walk_array_expr(input, bound, symbols, stage, def_name);
                walk_lambda(&op.lam, bound, symbols, stage, def_name);
            }
            SoacOp::Screma {
                lanes,
                accumulators,
                inputs,
            } => {
                for acc in accumulators {
                    walk(&acc.ne, bound, symbols, stage, def_name);
                }
                for i in inputs {
                    walk_array_expr(i, bound, symbols, stage, def_name);
                }
                for lane in lanes {
                    walk_lambda(&lane.lam.lam, bound, symbols, stage, def_name);
                }
                for acc in accumulators {
                    walk_lambda(&acc.step_lam.lam, bound, symbols, stage, def_name);
                    walk_lambda(&acc.reduce_op.lam, bound, symbols, stage, def_name);
                }
            }
            SoacOp::Filter { pred, input, .. } => {
                walk_array_expr(input, bound, symbols, stage, def_name);
                walk_lambda(&pred.lam, bound, symbols, stage, def_name);
            }
            SoacOp::Scatter { lam, inputs, .. } => {
                for i in inputs {
                    walk_array_expr(i, bound, symbols, stage, def_name);
                }
                walk_lambda(&lam.lam, bound, symbols, stage, def_name);
            }
            SoacOp::ReduceByIndex {
                op,
                ne,
                indices,
                values,
                ..
            } => {
                walk(ne, bound, symbols, stage, def_name);
                walk_array_expr(indices, bound, symbols, stage, def_name);
                walk_array_expr(values, bound, symbols, stage, def_name);
                walk_lambda(&op.lam, bound, symbols, stage, def_name);
            }
        }
    }

    fn walk_array_expr(
        ae: &ArrayExpr,
        bound: &HashSet<SymbolId>,
        symbols: &SymbolTable,
        stage: &str,
        def_name: &str,
    ) {
        match ae {
            ArrayExpr::Var(vr, ty) => {
                let mut ids = crate::tlc::TermIdSource::new();
                let t = crate::tlc::atom_var_term(*vr, ty.clone(), &mut ids);
                walk(&t, bound, symbols, stage, def_name);
            }
            ArrayExpr::Zip(arrs) => {
                for a in arrs {
                    walk_array_expr(a, bound, symbols, stage, def_name);
                }
            }
            ArrayExpr::Literal(elems) => {
                for e in elems {
                    walk(e, bound, symbols, stage, def_name);
                }
            }
            ArrayExpr::Range { start, len, step } => {
                walk(start, bound, symbols, stage, def_name);
                walk(len, bound, symbols, stage, def_name);
                if let Some(s) = step {
                    walk(s, bound, symbols, stage, def_name);
                }
            }
            ArrayExpr::StorageView(crate::tlc::StorageView { offset, len, .. }) => {
                walk(offset, bound, symbols, stage, def_name);
                walk(len, bound, symbols, stage, def_name);
            }
        }
    }

    // `bound` = everything the TLC name-resolver considers a top-level
    // symbol, not just things with bodies. `def_syms` holds the
    // pre-allocated SymbolId for every prelude/user/sig top-level name.
    let mut top_level: std::collections::HashSet<SymbolId> = std::collections::HashSet::new();
    for sym in program.def_syms.values() {
        top_level.insert(*sym);
    }
    // Catch any defs whose own SymbolId isn't already in `def_syms` —
    // shouldn't happen by construction, but assert via union.
    for d in &program.defs {
        top_level.insert(d.name);
    }
    for def in &program.defs {
        let def_name = program.symbols.get(def.name).cloned().unwrap_or_default();
        walk(&def.body, &top_level, &program.symbols, stage, &def_name);
    }
}

/// Regression: under Phase 2, `let x = arr in body[x]` was leaving a
/// free `Var(Symbol)` in the post-`partial_eval` TLC because `apply`'s
/// `Var(Builtin)` and catch-all arms cloned the original term instead
/// of residualizing through the eval'd args. The fix lives in
/// `tlc::partial_eval::residualize_call`. This test compiles the
/// minimal repro through the full canonical TLC pipeline and asserts
/// no `Var(Symbol(sym))` references an unbound symbol.
#[test]
fn let_binding_substitution_survives_partial_eval() {
    let source = r#"
#[fragment]
entry frag() #[target(screen)] vec4f32 =
    let range = [1, 2, 3, 4] in
    @[f32.i32(range[0]), 0.0, 0.0, 1.0]
"#;
    let tlc = crate::compile_thru_tlc(source).expect("compile_thru_tlc");
    assert_no_unbound_var_refs(&tlc.tlc, "compile_thru_tlc");
}

// =============================================================================
// SOAC Fusion Integration Tests
// =============================================================================

#[test]
fn consuming_map_compiles_end_to_end() {
    // `*[N]T` map whose input is dead-after: the ownership pass
    // sets `destination = SoacDestination::InputBuffer`, EGIR
    // conversion threads it through, and `soac_expand` emits the
    // in-place loop. Compiling end-to-end through SSA exercises
    // every layer.
    let _ssa = compile_to_ssa(
        r#"
def f(a: *[8]i32) [8]i32 = map(|x: i32| x + 1, a)
"#,
    );
}

/// Count `_w_intrinsic_uninit` calls across the entire SSA program
/// (functions + entry points). `Fresh` Map destinations introduce
/// one per allocation; the `InputBuffer` destination should
/// introduce zero. Aggregating across all bodies sidesteps
/// inlining choices that move the map's body between functions.
fn count_uninit_in_program(ssa: &Program) -> usize {
    let mut count = 0;
    let bodies = ssa
        .functions
        .iter()
        .map(|f| &f.body.inner.insts)
        .chain(ssa.entry_points.iter().map(|e| &e.body.inner.insts));
    for insts in bodies {
        for (_id, inst) in insts {
            if let crate::ssa::types::InstKind::Op { tag, .. } = &inst.data {
                match tag {
                    crate::op::OpTag::Call(f) => {
                        if f == "_w_intrinsic_uninit" {
                            count += 1;
                        }
                    }
                    crate::op::OpTag::Intrinsic { id, .. } => {
                        let name = crate::builtins::by_id(*id).raw.surface_name;
                        if name == "_w_intrinsic_uninit" {
                            count += 1;
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    count
}

#[test]
fn consuming_map_skips_fresh_allocation() {
    // The underlying ownership-pass invariant: when a SOAC's input
    // is unique-and-dead-after, the loop's carried buffer reuses
    // the input — no `_w_intrinsic_uninit`. When the input is
    // still live after the SOAC (an alias), the SOAC falls back to
    // a fresh allocation.
    //
    // The previous shape (consuming `*[N]T` vs borrowing `[N]T`
    // helper) is now collapsed by `force_inline_soac_helpers`: both
    // helper variants get inlined upstream and ownership sees the
    // exact same caller-level shape in either case. We express the
    // same invariant by varying *use-after-SOAC* on a let-bound
    // array literal: the fresh literal is consumable iff it has no
    // remaining uses past the map.
    let dead_after_ssa = compile_to_ssa(
        r#"
#[fragment]
entry frag(c: vec4f32) #[target(screen)] vec4f32 =
    let xs = [1, 2, 3, 4, 5, 6, 7, 8] in
    let r = map(|x: i32| x + 1, xs) in
    @[f32.i32(r[0]), f32.i32(r[1]), 0.0, 0.0]
"#,
    );
    assert_eq!(
        count_uninit_in_program(&dead_after_ssa),
        0,
        "map over a unique-and-dead input should write back in place, no fresh buffer",
    );

    let aliased_ssa = compile_to_ssa(
        r#"
#[fragment]
entry frag(c: vec4f32) #[target(screen)] vec4f32 =
    let xs = [1, 2, 3, 4, 5, 6, 7, 8] in
    let r = map(|x: i32| x + 1, xs) in
    -- `xs` aliased past the map → map must allocate a fresh buffer
    @[f32.i32(r[0]), f32.i32(xs[0]), 0.0, 0.0]
"#,
    );
    assert!(
        count_uninit_in_program(&aliased_ssa) >= 1,
        "map over an input that's still aliased after the map should allocate a fresh buffer",
    );
}

#[test]
fn consuming_scan_compiles_end_to_end() {
    // Parallel of `consuming_map_compiles_end_to_end` for Scan: `*[N]T`
    // input that's dead-after; ownership sets
    // `destination = SoacDestination::InputBuffer`, EGIR threads it through,
    // and `soac_expand` runs the destination-passing loop. Pre-S4 this hit a
    // panic.
    let _ssa = compile_to_ssa(
        r#"
def cumsum(a: *[8]i32) [8]i32 = scan(|acc: i32, x: i32| acc + x, 0, a)
"#,
    );
}

#[test]
fn consuming_scan_skips_fresh_allocation() {
    // Sibling of `consuming_map_skips_fresh_allocation` for Scan.
    // Same invariant — unique-and-dead input → in-place writeback;
    // aliased-after input → fresh allocation — expressed at the
    // use-after-SOAC level rather than via a helper boundary
    // (which `force_inline_soac_helpers` would collapse).
    let dead_after_ssa = compile_to_ssa(
        r#"
#[fragment]
entry frag(c: vec4f32) #[target(screen)] vec4f32 =
    let xs = [1, 2, 3, 4, 5, 6, 7, 8] in
    let r = scan(|acc: i32, x: i32| acc + x, 0, xs) in
    @[f32.i32(r[0]), f32.i32(r[1]), 0.0, 0.0]
"#,
    );
    assert_eq!(
        count_uninit_in_program(&dead_after_ssa),
        0,
        "scan over a unique-and-dead input should write back in place, no fresh buffer",
    );

    let aliased_ssa = compile_to_ssa(
        r#"
#[fragment]
entry frag(c: vec4f32) #[target(screen)] vec4f32 =
    let xs = [1, 2, 3, 4, 5, 6, 7, 8] in
    let r = scan(|acc: i32, x: i32| acc + x, 0, xs) in
    @[f32.i32(r[0]), f32.i32(xs[0]), 0.0, 0.0]
"#,
    );
    assert!(
        count_uninit_in_program(&aliased_ssa) >= 1,
        "scan over an input that's still aliased after the scan should allocate a fresh buffer",
    );
}

#[test]
fn consuming_scan_compute_entry_compiles_to_spirv() {
    // Compute entry with `*[]T` param. Exercises the Scan-DPS path
    // end-to-end through SPIR-V emission. Regression guards:
    //
    // 1. Type-checker: `*[]T` on a compute-entry param must constrain
    //    the array variant to `View`, not default to `Composite`.
    //    Otherwise `polytype_to_spirv` panics with "Composite variant
    //    unsized arrays not supported".
    //
    // 2. SPIR-V backend: views threaded through loop block params
    //    (`%phi = phi(orig_view, array_with_inplace_result)`) must
    //    keep their buffer provenance. `view_buffer_id` is propagated
    //    across branch edges and through `array_with_inplace`, so
    //    `ViewIndex` can resolve the backing buffer without
    //    extracting buffer_id from a runtime struct field.
    let spv = compile_to_spirv(
        r#"
#[compute]
entry scan_inplace(a: *[]i32) []i32 =
  scan(|acc: i32, x: i32| acc + x, 0, a)
"#,
    )
    .expect("compute scan_inplace should compile end-to-end");
    assert!(!spv.is_empty(), "compute scan_inplace emitted empty SPIR-V");
}

#[test]
fn parallel_scan_emits_swap_wrapper_with_swapped_args() {
    // Phase 3 of parallel scan reads `off = block_offsets[tid]` and
    // applies `op(off, elem)` to each element of `output[chunk]`. Map's
    // body-call convention is `func(elem, ...captures)`, which would
    // give `op(elem, off)` — sound for commutative ops, silently wrong
    // for non-commutative ones. EGIR plumbs around this by synthesizing
    // a swap-args wrapper function `\(a, b) -> op(b, a)` and routing
    // phase 3 through it; this test pins that wiring in SSA.
    let ssa = compile_to_ssa(
        r#"
#[compute]
entry parallel_scan(a: []i32) []i32 = scan(|acc: i32, x: i32| acc + x, 0, a)
"#,
    );

    let wrapper = ssa
        .functions
        .iter()
        .find(|f| f.name.ends_with("_scan_op_swap"))
        .expect("parallel scan should synthesize a swap wrapper EgirFunc");

    assert_eq!(
        wrapper.body.params.len(),
        2,
        "swap wrapper must take exactly two params"
    );
    let a_id = wrapper.body.params[0].0;
    let b_id = wrapper.body.params[1].0;

    let call = wrapper
        .body
        .inner
        .insts
        .values()
        .find_map(|inst| match &inst.data {
            crate::ssa::types::InstKind::Op {
                tag: crate::op::OpTag::Call(name),
                operands,
            } => Some((name.clone(), operands.clone())),
            _ => None,
        })
        .expect("swap wrapper body must contain a Call");

    assert!(
        !call.0.ends_with("_scan_op_swap"),
        "swap wrapper should call the underlying op, not itself: got {}",
        call.0
    );
    let operands: Vec<_> = call.1.iter().map(|v| v.as_ssa()).collect();
    assert_eq!(
        operands,
        vec![Some(b_id), Some(a_id)],
        "swap wrapper must call inner(b, a), not inner(a, b); got operands {:?} vs params [a={:?}, b={:?}]",
        operands,
        a_id,
        b_id,
    );
}

#[test]
fn consuming_filter_compiles_end_to_end() {
    // `*[N]T` filter whose input is dead-after: ownership sets
    // `destination = SoacDestination::InputBuffer`, EGIR threads it
    // through, and `build_filter_loop` carries the input array as the
    // destination buffer.
    let _ssa = compile_to_ssa(
        r#"
def keep_pos(a: *[8]i32) ?k.[k]i32 = filter(|x: i32| x > 0, a)
"#,
    );
}

#[test]
fn consuming_filter_skips_fresh_allocation() {
    // Filter's static-capacity lowering targets a function-local `Alloca`
    // and writes surviving elements through `PlaceIndex`, so neither
    // variant emits `_w_intrinsic_uninit`. The consuming case seeds the
    // alloca with the input array (an init `Store`); the borrowing case
    // skips the init store. Both compile and validate.
    let consuming_ssa = compile_to_ssa(
        r#"
def keep_pos(a: *[8]i32) ?k.[k]i32 = filter(|x: i32| x > 0, a)

#[fragment]
entry frag(c: vec4f32) vec4f32 =
    let r = keep_pos([1, -2, 3, -4, 5, -6, 7, -8]) in
    @[f32.i32(r[0]), 0.0, 0.0, 1.0]
"#,
    );
    assert_eq!(
        count_uninit_in_program(&consuming_ssa),
        0,
        "filter lowering should not emit `_w_intrinsic_uninit`",
    );

    let borrowing_ssa = compile_to_ssa(
        r#"
def keep_pos(a: [8]i32) ?k.[k]i32 = filter(|x: i32| x > 0, a)

#[fragment]
entry frag(c: vec4f32) vec4f32 =
    let r = keep_pos([1, -2, 3, -4, 5, -6, 7, -8]) in
    @[f32.i32(r[0]), 0.0, 0.0, 1.0]
"#,
    );
    assert_eq!(
        count_uninit_in_program(&borrowing_ssa),
        0,
        "filter lowering should not emit `_w_intrinsic_uninit`",
    );
}

/// Multiset of `(category, identifier)` pairs across every instruction
/// in `ssa.functions` + `ssa.entry_points`. Used by structural-equivalence
/// tests that need to compare two SSA programs while ignoring value-id
/// renumbering, block-ordering, and other low-level details.
fn inst_signature_multiset(ssa: &Program) -> std::collections::BTreeMap<String, usize> {
    use crate::op::OpTag;
    use crate::ssa::types::InstKind;
    use std::collections::BTreeMap;

    let signature = |kind: &InstKind| -> String {
        match kind {
            InstKind::Alloca { .. } => "Alloca".to_string(),
            InstKind::Load { .. } => "Load".to_string(),
            InstKind::Store { .. } => "Store".to_string(),
            InstKind::ViewIndex { .. } => "ViewIndex".to_string(),
            InstKind::PlaceIndex { .. } => "PlaceIndex".to_string(),
            InstKind::OutputSlot { .. } => "OutputSlot".to_string(),
            InstKind::ControlBarrier => "ControlBarrier".to_string(),
            InstKind::Op { tag, .. } => format!(
                "Op:{}",
                match tag {
                    OpTag::Call(name) => format!("Call({})", name),
                    OpTag::Intrinsic { id, .. } => {
                        let name = crate::builtins::by_id(*id).raw.surface_name;
                        format!("Intrinsic({})", name)
                    }
                    OpTag::BinOp(s) => format!("BinOp({})", s),
                    OpTag::UnaryOp(s) => format!("UnaryOp({})", s),
                    // Literal values intentionally NOT included in the
                    // signature — a constant-folding refactor shouldn't
                    // make the test flake. Variant name alone is the
                    // structural signal.
                    OpTag::Int(_) => "Int".to_string(),
                    OpTag::Uint(_) => "Uint".to_string(),
                    OpTag::Float(_) => "Float".to_string(),
                    OpTag::Bool(_) => "Bool".to_string(),
                    OpTag::Unit => "Unit".to_string(),
                    OpTag::Global(_) => "Global".to_string(),
                    OpTag::Extern(_) => "Extern".to_string(),
                    OpTag::Tuple(_) => "Tuple".to_string(),
                    OpTag::Vector(_) => "Vector".to_string(),
                    OpTag::Matrix { .. } => "Matrix".to_string(),
                    OpTag::ArrayLit(_) => "ArrayLit".to_string(),
                    OpTag::ArrayRange { .. } => "ArrayRange".to_string(),
                    OpTag::Project { .. } => "Project".to_string(),
                    OpTag::Index => "Index".to_string(),
                    OpTag::Materialize => "Materialize".to_string(),
                    OpTag::DynamicExtract => "DynamicExtract".to_string(),
                    OpTag::StorageView(_) => "StorageView".to_string(),
                    OpTag::StorageViewLen => "StorageViewLen".to_string(),
                    OpTag::StorageImageLoad(_) => "StorageImageLoad".to_string(),
                    OpTag::StorageImageStore(_) => "StorageImageStore".to_string(),
                    OpTag::ViewIndex => "ViewIndex(pure)".to_string(),
                    OpTag::PlaceIndex => "PlaceIndex(pure)".to_string(),
                    OpTag::OutputSlot { .. } => "OutputSlot(pure)".to_string(),
                }
            ),
        }
    };

    let mut out: BTreeMap<String, usize> = BTreeMap::new();
    let bodies = ssa
        .functions
        .iter()
        .map(|f| &f.body.inner.insts)
        .chain(ssa.entry_points.iter().map(|e| &e.body.inner.insts));
    for insts in bodies {
        for (_id, inst) in insts {
            *out.entry(signature(&inst.data)).or_insert(0) += 1;
        }
    }
    out
}

#[test]
fn filter_length_is_runtime_count_not_static_capacity() {
    // The bug this test guards against: if the bounded filter result's
    // `len` field is fed the input array's static capacity (e.g. 4)
    // instead of the runtime write-cursor count, `length(r)` silently
    // returns the capacity. Indexed reads `r[0]` / `r[1]` still
    // produce the first two kept elements (those slots were written),
    // so a smoke test would pass — but anything iterating
    // `0..length(r)` would read garbage past the real count.
    //
    // The previous shape (consuming `*[N]T` vs borrowing `[N]T` helper)
    // collapsed under `force_inline_soac_helpers`: both helper
    // variants get force-inlined before ownership runs, so the helper
    // boundary the test depended on disappears. We test the same
    // invariant directly on a static-literal filter in the entry
    // body: the lowered SSA must contain a `_w_intrinsic_length`
    // intrinsic call against the filter result — proving the length
    // is *computed* from the bounded wrapper's runtime `len` field,
    // not short-circuited to the literal capacity.
    let ssa = compile_to_ssa(
        r#"
#[fragment]
entry frag(c: vec4f32) #[target(screen)] vec4f32 =
    let r = filter(|x: i32| x > 0, [1, -2, 3, -4]) in
    @[f32.i32(length(r)), f32.i32(r[0]), f32.i32(r[1]), 1.0]
"#,
    );

    let tags = inst_signature_multiset(&ssa);
    let length_calls = tags.get("Op:Intrinsic(length)").copied().unwrap_or(0);
    assert!(
        length_calls >= 1,
        "filter result's length must reach the SSA as a `length` intrinsic call \
         (proving the bounded wrapper's runtime `len` field is being read), \
         not be short-circuited to the static capacity. \
         Got tag multiset: {:?}",
        tags,
    );
}

#[test]
fn test_map_reduce_fusion_end_to_end() {
    let source = r#"
def globalArr: [4]f32 = [10.0, 20.0, 30.0, 40.0]

def myMap(ro: f32, rd: f32) [4]f32 =
  map(|x: f32| x + ro + rd, globalArr)

def myReduce(hits: [4]f32) f32 =
  reduce(|acc: f32, x: f32| if acc < x then acc else x, 999.0, hits)

#[fragment]
entry fragment_main() #[target(screen)] vec4f32 =
  let hits = myMap(1.0, 2.0) in
  let closest = myReduce(hits) in
  @[closest, 0.0, 0.0, 1.0]
"#;

    let tlc = compile_to_fused_tlc(source);

    // After fusion, check that myMap's body is no longer a standalone map
    // or that some def contains a fused reduce
    let _my_map_has_map = tlc.defs.iter().any(|def| {
        let name = tlc.symbols.get(def.name).cloned().unwrap_or_default();
        if name != "myMap" {
            return false;
        }
        let (_, body) = extract_lambda_params(&def.body);
        has_soac_kind(&body, "Map")
    });

    let _any_has_reduce = tlc.defs.iter().any(|def| {
        let (_, body) = extract_lambda_params(&def.body);
        has_soac_kind(&body, "Reduce")
    });

    // Check fragment_main: does it contain a fused Reduce?
    let fragment_main = tlc
        .defs
        .iter()
        .find(|def| tlc.symbols.get(def.name).map(|s| s.as_str()) == Some("fragment_main"))
        .expect("fragment_main not found");

    let (_, frag_body) = extract_lambda_params(&fragment_main.body);
    // A fused `map → reduce` lowers to a single-accumulator `Screma`.
    let frag_has_screma = has_soac_kind(&frag_body, "Screma");
    let frag_has_reduce = has_soac_kind(&frag_body, "Reduce");
    let frag_has_map = has_soac_kind(&frag_body, "Map");

    eprintln!("fragment_main has Screma: {}", frag_has_screma);
    eprintln!("fragment_main has Reduce: {}", frag_has_reduce);
    eprintln!("fragment_main has Map: {}", frag_has_map);
    eprintln!(
        "fragment_main body: {:?}",
        std::mem::discriminant(&frag_body.kind)
    );

    // Print the Let chain structure
    fn print_term(term: &crate::tlc::Term, syms: &SymbolTable, depth: usize) {
        let indent = "  ".repeat(depth);
        match &term.kind {
            TermKind::Let { name, rhs, body, .. } => {
                let n = syms.get(*name).cloned().unwrap_or_else(|| format!("{:?}", name));
                eprintln!("{indent}let {n} = ...");
                print_term(rhs, syms, depth + 1);
                print_term(body, syms, depth);
            }
            TermKind::Soac(soac) => {
                eprintln!("{indent}SOAC {:?}", std::mem::discriminant(soac));
            }
            TermKind::App { func, args } => {
                eprintln!("{indent}App:");
                print_term(func, syms, depth + 1);
                for a in args {
                    print_term(a, syms, depth + 1);
                }
            }
            TermKind::Var(VarRef::Symbol(s)) => {
                let n = syms.get(*s).cloned().unwrap_or_else(|| format!("{:?}", s));
                eprintln!("{indent}Var({n})");
            }
            other => {
                eprintln!("{indent}{:?}", std::mem::discriminant(other));
            }
        }
    }
    print_term(&frag_body, &tlc.symbols, 0);

    // The fusion should have replaced the let chain with a fused SOAC
    // or at minimum the fragment_main should contain a Reduce
    assert!(
        frag_has_screma || frag_has_reduce,
        "Expected fragment_main to contain a fused Screma or Reduce after interprocedural fusion"
    );
}

fn has_soac_kind(term: &crate::tlc::Term, kind: &str) -> bool {
    use crate::tlc::{SoacOp, TermKind};
    match &term.kind {
        TermKind::Soac(SoacOp::Map { .. }) if kind == "Map" => true,
        TermKind::Soac(SoacOp::Reduce { .. }) if kind == "Reduce" => true,
        TermKind::Soac(SoacOp::Screma { .. }) if kind == "Screma" => true,
        TermKind::Soac(SoacOp::Filter { .. }) if kind == "Filter" => true,
        TermKind::Let { rhs, body, .. } => has_soac_kind(rhs, kind) || has_soac_kind(body, kind),
        TermKind::Lambda(lam) => has_soac_kind(&lam.body, kind),
        TermKind::App { func, args } => {
            has_soac_kind(func, kind) || args.iter().any(|a| has_soac_kind(a, kind))
        }
        TermKind::Tuple(parts) | TermKind::VecLit(parts) => parts.iter().any(|p| has_soac_kind(p, kind)),
        TermKind::TupleProj { tuple, .. } => has_soac_kind(tuple, kind),
        TermKind::Index { array, index } => has_soac_kind(array, kind) || has_soac_kind(index, kind),
        TermKind::If {
            cond,
            then_branch,
            else_branch,
        } => {
            has_soac_kind(cond, kind)
                || has_soac_kind(then_branch, kind)
                || has_soac_kind(else_branch, kind)
        }
        _ => false,
    }
}

#[test]
fn test_screma_fusion_end_to_end() {
    let source = r#"
#[compute]
entry gen(xs: []i32) ([]i32, [1]i32) =
  let b = map(|x: i32| x + 1, xs) in
  let c = map(|y: i32| y * 2, b) in
  let d = reduce(|acc: i32, z: i32| acc + z, 0, b) in
  (c, [d])
"#;

    let tlc = compile_to_fused_tlc(source);
    let gen = tlc
        .defs
        .iter()
        .find(|def| tlc.symbols.get(def.name).map(|s| s.as_str()) == Some("gen"))
        .expect("gen not found");

    let (_, body) = extract_lambda_params(&gen.body);
    assert!(
        has_soac_kind(&body, "Screma"),
        "expected shared map producer feeding map+reduce to fuse to Screma"
    );

    compile_to_spirv(source).expect("Screma-fused map+reduce should lower to SPIR-V");
}

#[test]
fn test_screma_scan_fusion_end_to_end() {
    let source = r#"
#[compute]
entry gen(xs: []i32) ([]i32, []i32) =
  let b = map(|x: i32| x + 1, xs) in
  let c = map(|y: i32| y * 2, b) in
  let d = scan(|acc: i32, z: i32| acc + z, 0, b) in
  (c, d)
"#;

    let tlc = compile_to_fused_tlc(source);
    let gen = tlc
        .defs
        .iter()
        .find(|def| tlc.symbols.get(def.name).map(|s| s.as_str()) == Some("gen"))
        .expect("gen not found");

    let (_, body) = extract_lambda_params(&gen.body);
    assert!(
        has_soac_kind(&body, "Screma"),
        "expected shared map producer feeding map+scan to fuse to Screma"
    );

    compile_to_spirv(source).expect("Screma-fused map+scan should lower to SPIR-V");
}

#[test]
fn test_screma_multi_output_fusion_end_to_end() {
    let source = r#"
#[compute]
entry gen(xs: []i32) ([]i32, []i32, [1]i32, []i32) =
  let b = map(|x: i32| x + 1, xs) in
  let c = map(|y: i32| y * 2, b) in
  let d = reduce(|acc: i32, z: i32| acc + z, 0, b) in
  let e = map(|w: i32| w - 3, b) in
  let f = scan(|acc: i32, q: i32| acc + q, 0, b) in
  (c, e, [d], f)
"#;

    let tlc = compile_to_fused_tlc(source);
    let gen = tlc
        .defs
        .iter()
        .find(|def| tlc.symbols.get(def.name).map(|s| s.as_str()) == Some("gen"))
        .expect("gen not found");

    let (_, body) = extract_lambda_params(&gen.body);
    assert!(
        has_soac_kind(&body, "Screma"),
        "expected shared map producer feeding multiple maps/reduce/scan to fuse to Screma"
    );
    fn screma_shape(term: &crate::tlc::Term) -> Option<(usize, usize, usize, usize)> {
        use crate::tlc::{ScremaAccumulator, SoacOp, TermKind};
        match &term.kind {
            TermKind::Soac(SoacOp::Screma {
                lanes, accumulators, ..
            }) => Some((
                lanes.len(),
                accumulators.len(),
                accumulators.iter().filter(|acc| acc.kind == ScremaAccumulator::Reduce).count(),
                accumulators.iter().filter(|acc| acc.kind == ScremaAccumulator::Scan).count(),
            )),
            TermKind::Let { rhs, body, .. } => screma_shape(rhs).or_else(|| screma_shape(body)),
            TermKind::Lambda(lam) => screma_shape(&lam.body),
            TermKind::Tuple(parts) | TermKind::VecLit(parts) => parts.iter().find_map(screma_shape),
            TermKind::TupleProj { tuple, .. } => screma_shape(tuple),
            _ => None,
        }
    }
    assert_eq!(screma_shape(&body), Some((2, 2, 1, 1)));

    compile_to_spirv(source).expect("multi-output Screma should lower to SPIR-V");
}

// =============================================================================
// Multi-output compute entries
// =============================================================================

/// Regression: a compute entry returning a tuple of >1 runtime-sized
/// array used to panic in EGIR lowering — the SOAC→OutputView rewrite
/// that streams a runtime-sized result into its bound storage view only
/// fired for the single-output case. For a tuple result, each field
/// reached `emit_compute_output_stores` as a plain value and hit the
/// `is_unsized_array` guard (`from_tlc.rs` "must rewrite to OutputView
/// upstream"). Each tuple field's producing Map/Scan must be retargeted
/// to its own output view.
#[test]
fn test_multi_output_compute_runtime_sized_arrays() {
    let _ssa = compile_to_ssa(
        r#"
#[compute]
entry gen(src: []f32) ([]f32, []f32) =
    (map(|x: f32| x * 2.0, src), map(|x: f32| x * 3.0, src))
"#,
    );
    // Compilation success (no panic) is the test.
}

/// A multidomain split where two map outputs share an input domain. TLC fuses
/// the equal-domain pair into one multi-lane SegMap that writes *two* output
/// slots; the split must keep that fused side-effect — and both its output
/// bindings — together in one kernel, not strand it as "shared" while dropping
/// its bindings from `outputs`. Regression: a computed-fixed output plus a
/// shared-domain map pair plus enough distinct-domain maps to reach five
/// outputs used to lower to a stage whose output binding wasn't allocated
/// ("Storage buffer not found").
#[test]
fn multidomain_split_with_shared_domain_map_pair_compiles() {
    use crate::pipeline_descriptor::Pipeline;
    let lowered = crate::compile_thru_spirv(
        r#"
#[compute]
entry r(a: []u32, b: []u32, c: []u32, st: []f32)
  ([2]f32, []u32, []u32, []u32, []u32) =
  let g = st[0] in
  let o0 = [g, g] in
  let m1 = map(|s: u32| s + 1u32, a) in
  let m2 = map(|s: u32| s + 2u32, b) in
  let m3 = map(|s: u32| s + 3u32, c) in
  let m4 = map(|s: u32| s + 4u32, c) in
  (o0, m1, m2, m3, m4)
"#,
    )
    .expect("fixed + distinct maps + shared-domain map pair compiles");
    let compute = lowered
        .pipeline
        .pipelines
        .iter()
        .find_map(|p| match p {
            Pipeline::Compute(c) => Some(c),
            _ => None,
        })
        .expect("one compute pipeline");
    // Five outputs, but m3 and m4 share domain `c` and fuse into one kernel:
    // o0 (fixed) + m1 + m2 + (m3,m4) = four stages.
    assert_eq!(
        compute.stages.len(),
        4,
        "the shared-domain map pair fuses into one stage; stages = {:?}",
        compute.stages.iter().map(|s| &s.entry_point).collect::<Vec<_>>()
    );
}

/// Same-domain sibling maps should fuse even after defunctionalization has
/// attached lexical captures to their bodies. The loop bodies exercise the
/// light/GTAO shape where per-element code runs a local sequential loop while
/// reading scalars computed outside the lambda.
#[test]
fn captured_loop_bodied_sibling_maps_fuse_to_one_stage() {
    use crate::pipeline_descriptor::Pipeline;

    let lowered = crate::compile_thru_spirv(
        r#"
#[compute]
entry geom(ids: []u32, params: []f32) ([]f32, []f32) =
  let scale = params[0] in
  let bias = params[1] in
  let geom_pos = map(|id: u32|
    let base = f32.u32(id) * scale in
    loop acc = base for k < 4 do
      acc + bias * f32.i32(k)
    , ids) in
  let geom_nrm = map(|id: u32|
    let base = f32.u32(id) + bias in
    loop acc = base for k < 4 do
      acc - scale * f32.i32(k)
    , ids) in
  (geom_pos, geom_nrm)
"#,
    )
    .expect("captured loop-bodied sibling maps should compile and fuse");

    let compute = lowered
        .pipeline
        .pipelines
        .iter()
        .find_map(|p| match p {
            Pipeline::Compute(c) => Some(c),
            _ => None,
        })
        .expect("one compute pipeline");
    assert_eq!(
        compute.stages.len(),
        1,
        "captured same-domain maps should lower as one multi-output stage; stages = {:?}",
        compute.stages.iter().map(|s| &s.entry_point).collect::<Vec<_>>()
    );
}

/// Fusing two captured sibling maps must keep each lane's own body — not
/// cross-wire both outputs to one lane's captures. The two lanes carry
/// distinctive constants (`1000.0`, `7.0`); if either is missing from the
/// single fused kernel, a lane collapsed onto the other. Value-level guard
/// beyond the stage-count check above.
#[test]
fn fused_sibling_maps_keep_each_lanes_own_body() {
    let source = r#"
#[compute]
entry two(ids: []u32, params: []f32) ([]f32, []f32) =
  let a = params[0] in
  let b = params[1] in
  let lo = map(|id: u32| f32.u32(id) * 1000.0 + a, ids) in
  let hi = map(|id: u32| f32.u32(id) * 7.0 + b, ids) in
  (lo, hi)
"#;

    let wgsl = crate::compile_thru_tlc(source)
        .expect("TLC")
        .infer_input_slice_bounds()
        .to_egraph()
        .expect("EGIR")
        .lower_to_ssa(crate::LoweringProfile::new(
            crate::CodegenTarget::Wgsl,
            crate::SchedulePolicy::Parallel,
        ))
        .expect("SSA")
        .lower_wgsl()
        .expect("WGSL lowering");

    assert!(
        wgsl.contains("1000.0"),
        "first lane's body (× 1000.0) was lost when fusing:\n{wgsl}"
    );
    assert!(
        wgsl.contains("7.0"),
        "second lane's body (× 7.0) was lost when fusing:\n{wgsl}"
    );

    let module = naga::front::wgsl::parse_str(&wgsl)
        .unwrap_or_else(|e| panic!("Naga rejected fused WGSL: {e:?}\n{wgsl}"));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("Naga validation rejected fused WGSL: {e:?}\n{wgsl}"));
}

/// An in-place map that prepares a buffer consumed by a later `scatter` is an
/// internal producer, not an entry output. It must stay wired into the scatter
/// pipeline — not reify into an independent output `SegMap` stage — otherwise
/// the scatter's view resolution is left with a bogus placeholder and the
/// backend emits an invalid `OpCompositeExtract` from a non-aggregate.
/// Regression distilled from `testfiles/playground/particles.wyn`.
#[test]
fn inplace_clear_feeding_scatter_stays_wired() {
    use crate::pipeline_descriptor::Pipeline;
    let src = r#"
#[compute]
entry sim(#[storage(set=2, binding=1, access=write)] fb: *[]u32, pos: []u32) []u32 =
  let cleared = map(|_p: u32| 0u32, fb) in
  let idxs = map(|p: u32| i32.u32(p), pos) in
  let vals = map(|_p: u32| 1u32, pos) in
  let _ = scatter(cleared, idxs, vals) in
  map(|p: u32| p + 1u32, pos)
"#;
    let lowered = crate::compile_thru_spirv(src).expect("in-place clear + scatter compiles");
    // The framebuffer clear is internal to the scatter, so it must not appear
    // as its own `_dispatch_` output stage.
    let stage_names: Vec<&str> = lowered
        .pipeline
        .pipelines
        .iter()
        .flat_map(|p| match p {
            Pipeline::Compute(c) => c.stages.iter().map(|s| s.entry_point.as_str()).collect(),
            Pipeline::Graphics(_) => Vec::new(),
        })
        .collect();
    assert!(
        !stage_names.iter().any(|n| n.contains("_dispatch_")),
        "the in-place clear must not split into an independent output stage; stages = {stage_names:?}"
    );
}

/// When a multi-output entry splits across distinct output domains, a shared
/// effectful side-effect (a `scatter` and the in-place clear / producer maps
/// feeding it) must run in exactly one of the split kernels — not be duplicated
/// into every clone, which would apply the scatter once per dispatch. The
/// scatter and its serial producers lower to loops; the other domain's map
/// kernel is loop-free, so exactly one compute entry contains a loop. Producers
/// orphaned in the non-host clone are dead-code pruned, so that clone stays
/// loop-free.
#[test]
fn multidomain_split_runs_shared_scatter_in_one_kernel() {
    use crate::pipeline_descriptor::Pipeline;
    use crate::ssa::types::ControlHeader;
    let src = r#"
#[compute]
entry r(a: []u32, b: []u32, #[storage(set=2, binding=0, access=write)] fb: *[]u32, pos: []u32)
  ([]u32, []u32) =
  let cleared = map(|_p: u32| 0u32, fb) in
  let idxs = map(|p: u32| i32.u32(p), pos) in
  let vals = map(|_p: u32| 1u32, pos) in
  let _ = scatter(cleared, idxs, vals) in
  (map(|x: u32| x + 1u32, a), map(|y: u32| y + 2u32, b))
"#;

    // Two distinct output domains → two map kernels.
    let lowered = crate::compile_thru_spirv(src).expect("multidomain split + shared scatter compiles");
    let compute = lowered
        .pipeline
        .pipelines
        .iter()
        .find_map(|p| match p {
            Pipeline::Compute(c) => Some(c),
            _ => None,
        })
        .expect("one compute pipeline");
    assert_eq!(
        compute.stages.len(),
        2,
        "the two map outputs split into two stages; stages = {:?}",
        compute.stages.iter().map(|s| &s.entry_point).collect::<Vec<_>>()
    );

    // The shared scatter (a serial loop) plus its serial producers live in
    // exactly one kernel; the other map kernel is loop-free after pruning.
    let program = compile_to_ssa(src);
    let entries_with_loops = program
        .entry_points
        .iter()
        .filter(|e| e.body.control_headers.values().any(|h| matches!(h, ControlHeader::Loop { .. })))
        .count();
    assert_eq!(
        entries_with_loops, 1,
        "the shared scatter must run in exactly one split kernel, not be duplicated; \
         entries with loops = {entries_with_loops}"
    );
}

/// A compute entry returning a tuple of pointwise maps over *different*
/// runtime-sized inputs splits into one parallel stage per output slot, each
/// dispatched over its own input's length. The two slots have independent
/// domains, so they become independent dispatches over their own inputs.
#[test]
fn multidomain_maps_split_into_per_domain_stages() {
    use crate::pipeline_descriptor::{DispatchLen, DispatchSize, Pipeline};
    let lowered = crate::compile_thru_spirv(
        r#"
#[compute]
entry two(a: []f32, b: []f32) ([]f32, []f32) =
    (map(|x: f32| x + 1.0, a), map(|x: f32| x + 2.0, b))
"#,
    )
    .expect("two compiles");

    let computes: Vec<_> = lowered
        .pipeline
        .pipelines
        .iter()
        .filter_map(|p| match p {
            Pipeline::Compute(c) => Some(c),
            _ => None,
        })
        .collect();
    assert_eq!(computes.len(), 1, "one compute pipeline backs entry two");
    let stages = &computes[0].stages;
    assert_eq!(
        stages.len(),
        2,
        "two output slots → two parallel stages, not one serial kernel"
    );

    let dispatch_binding = |i: usize| match &stages[i].dispatch_size {
        DispatchSize::DerivedFrom {
            len: DispatchLen::InputBinding { binding, .. },
            ..
        } => Some(*binding),
        _ => None,
    };
    let mut domains: Vec<u32> = (0..2).filter_map(dispatch_binding).collect();
    domains.sort();
    assert_eq!(
        domains,
        vec![0, 1],
        "the two stages dispatch over their own inputs (bindings 0 and 1), not a shared grid"
    );
}

/// Sibling maps over *different* buffers that share one size var
/// (`<[n]>(xs, ys)`) fuse into a single parallel kernel: both lanes read their
/// own input at the same `tid` under one guard and write both outputs. This is
/// equal-domain fusion — the buffers differ but the domain `n` is shared.
#[test]
fn equal_domain_sibling_maps_fuse_to_one_stage() {
    use crate::pipeline_descriptor::{DispatchSize, Pipeline};
    let lowered = crate::compile_thru_spirv(
        r#"
#[compute]
entry eqn<[n]>(xs: [n]f32, ys: [n]f32) ([n]f32, [n]f32) =
    (map(|x: f32| x + 1.0, xs), map(|y: f32| y + 2.0, ys))
"#,
    )
    .expect("eqn compiles");
    let compute = lowered
        .pipeline
        .pipelines
        .iter()
        .find_map(|p| match p {
            Pipeline::Compute(c) => Some(c),
            _ => None,
        })
        .expect("one compute pipeline");
    assert_eq!(compute.stages.len(), 1, "equal-domain slots fuse into one stage");
    assert!(
        matches!(compute.stages[0].dispatch_size, DispatchSize::DerivedFrom { .. }),
        "the fused stage dispatches over the shared runtime length"
    );
}

/// When a split entry's maps capture a storage entry param (`table`), every
/// synthesized stage declares that buffer in its `reads`. The host binds each
/// stage's inputs from that list, so a captured buffer missing from `reads`
/// would be read unbound. Pins that captures — not just SOAC inputs — reach the
/// descriptor.
#[test]
fn split_stage_reads_include_captured_storage_param() {
    use crate::pipeline_descriptor::{Binding, Pipeline};
    let lowered = crate::compile_thru_spirv(
        r#"
#[compute]
entry cap(a: []f32, b: []f32, table: []f32) ([]f32, []f32) =
    (map(|x: f32| x + table[0], a), map(|y: f32| y + table[0], b))
"#,
    )
    .expect("cap compiles");
    let compute = lowered
        .pipeline
        .pipelines
        .iter()
        .find_map(|p| match p {
            Pipeline::Compute(c) => Some(c),
            _ => None,
        })
        .expect("one compute pipeline");
    assert_eq!(
        compute.stages.len(),
        2,
        "a and b are distinct domains → two stages"
    );

    // `table` is the third input param, bound at (set 0, binding 2).
    let table_idx = compute
        .bindings
        .iter()
        .position(|b| {
            matches!(
                b,
                Binding::StorageBuffer {
                    set: 0,
                    binding: 2,
                    ..
                }
            )
        })
        .expect("captured `table` buffer is declared in the pipeline bindings");
    for (i, stage) in compute.stages.iter().enumerate() {
        assert!(
            stage.reads.contains(&table_idx),
            "stage {i} captures `table`, so it must read that buffer; reads = {:?}",
            stage.reads
        );
    }
}

/// Same-symbol sibling maps returned as a direct tuple (`(map(f, xs),
/// map(g, xs))`) share one domain (the same input `xs`) and fuse into a
/// single parallel compute stage that writes both outputs from one `tid`
/// grid, dispatched over `xs`'s length.
#[test]
fn same_symbol_sibling_maps_fuse_to_one_stage() {
    use crate::pipeline_descriptor::{DispatchLen, DispatchSize, Pipeline};
    let lowered = crate::compile_thru_spirv(
        r#"
#[compute]
entry same(xs: []f32) ([]f32, []f32) =
    (map(|x: f32| x + 1.0, xs), map(|x: f32| x + 2.0, xs))
"#,
    )
    .expect("same compiles");
    let compute = lowered
        .pipeline
        .pipelines
        .iter()
        .find_map(|p| match p {
            Pipeline::Compute(c) => Some(c),
            _ => None,
        })
        .expect("one compute pipeline");
    assert_eq!(compute.stages.len(), 1, "same-domain slots fuse into one stage");
    assert!(
        matches!(
            compute.stages[0].dispatch_size,
            DispatchSize::DerivedFrom {
                len: DispatchLen::InputBinding { binding: 0, .. },
                ..
            }
        ),
        "the fused stage dispatches over xs (binding 0)"
    );
}

// =============================================================================
// Basic Expressions
// =============================================================================

#[test]
fn test_basic_expressions() {
    // Tests: functions, let bindings, if expressions, binary/unary ops
    let _ssa = compile_to_ssa(
        r#"
def add(x: i32, y: i32) i32 = x + y

def with_let(a: i32, b: i32) i32 =
    let x = a in
    let y = b in
    x + y

def with_if(x: bool) i32 = if x then 1 else 0

def with_ops(x: i32, y: i32) i32 = x * y + x / y - (-x)

#[vertex]
entry vertex_main() #[builtin(position)] vec4f32 =
    let b = add(1, 2) in
    let c = with_let(3, 4) in
    let d = with_if(true) in
    let e = with_ops(5, 6) in
    @[f32.i32(b + c + d + e), 0.0, 0.0, 1.0]
"#,
    );

    // Compilation success is the test (partial eval may inline simple functions)
}

// =============================================================================
// Data Structures
// =============================================================================

#[test]
fn test_data_structures() {
    // Tests: arrays, tuples, records, tuple patterns
    let _ssa = compile_to_ssa(
        r#"
def arr = [1, 2, 3]

def record = {x = 1, y = 2}

def tuple_destruct: i32 =
    let (a, b) = (1, 2) in a + b

def nested_tuple: i32 =
    let ((a, b), c) = ((1, 2), 3) in a + b + c

def array_index(arr: [4]i32, i: i32) i32 = arr[i]

#[vertex]
entry vertex_main() #[builtin(position)] vec4f32 =
    let a = arr[0] in
    let b = record.x in
    let c = tuple_destruct in
    let d = nested_tuple in
    let e = array_index([1, 2, 3, 4], 0) in
    @[f32.i32(a + b + c + d + e), 0.0, 0.0, 1.0]
"#,
    );
    // Compilation success is the test
}

// =============================================================================
// Tuple Positional Access
// =============================================================================

#[test]
fn test_tuple_positional_access() {
    // Tests: .0, .1 on tuples, chained access, in expressions
    let _ssa = compile_to_ssa(
        r#"
def first(t: (i32, f32)) i32 = t.0

def second(t: (i32, f32)) f32 = t.1

def sum_pair(t: (i32, i32)) i32 = t.0 + t.1

def nested(t: ((i32, i32), f32)) i32 =
    let inner = t.0 in inner.0 + inner.1

#[vertex]
entry vertex_main() #[builtin(position)] vec4f32 =
    let t = (42, 3.14) in
    let a = first(t) in
    let b = sum_pair((1, 2)) in
    let c = nested(((10, 20), 1.0)) in
    @[f32.i32(a + b + c), 0.0, 0.0, 1.0]
"#,
    );
}

// =============================================================================
// Loops
// =============================================================================

#[test]
fn test_loops() {
    // Tests: while loops, for-range loops, for-in loops
    let _ssa = compile_to_ssa(
        r#"
def while_loop: i32 =
    loop x = 0 while x < 10 do x + 1

def for_range_loop: i32 =
    loop acc = 0 for i < 10 do acc + i

def for_in_loop(arr: [5]i32) i32 =
    loop acc = 0 for x in arr do acc + x

#[vertex]
entry vertex_main() #[builtin(position)] vec4f32 =
    let a = while_loop in
    let b = for_range_loop in
    let c = for_in_loop([1, 2, 3, 4, 5]) in
    @[f32.i32(a + b + c), 0.0, 0.0, 1.0]
"#,
    );
    // Compilation success is the test
}

// =============================================================================
// Lambdas and Closures
// =============================================================================

#[test]
fn test_lambdas_and_closures() {
    // Tests: lambdas with captures, nested lambdas, direct calls, tuple params
    let _ssa = compile_to_ssa(
        r#"
def with_capture(y: i32) i32 =
    let f = |x: i32| x + y in
    f(10)

def nested_lambda(x: i32) i32 =
    let outer = |a: i32|
        let inner = |b: i32| a + b + x in
        inner(a)
    in
    outer(5)

def tuple_param_lambda: i32 =
    let add = |(x, y): (i32, i32)| x + y in
    add((1, 2))

def direct_call: i32 =
    let inc = |x: i32| x + 1 in
    inc(5)

#[vertex]
entry vertex_main() #[builtin(position)] vec4f32 =
    let b = with_capture(10) in
    let c = nested_lambda(100) in
    let d = tuple_param_lambda in
    let e = direct_call in
    @[f32.i32(b + c + d + e), 0.0, 0.0, 1.0]
"#,
    );
    // Compilation success is the test
}

// =============================================================================
// Higher-Order Functions (map, reduce, filter)
// =============================================================================

#[test]
fn test_higher_order_functions() {
    // Tests: map, reduce, filter with lambdas and named functions
    let _ssa = compile_to_ssa(
        r#"
def double(x: i32) i32 = x * 2

def map_named(arr: [4]i32) [4]i32 = map(double, arr)

def map_lambda(arr: [4]i32) [4]i32 = map(|x: i32| x + 1, arr)

def map_with_capture(arr: [4]i32, offset: i32) [4]i32 =
    map(|x: i32| x + offset, arr)

def reduce_sum(arr: [4]f32) f32 =
    reduce(|acc: f32, x: f32| acc + x, 0.0, arr)

def reduce_tuple(hits: [4](f32, i32)) (f32, i32) =
    reduce(|(t1, m1): (f32, i32), (t2, m2): (f32, i32)|
             if t1 < t2 then (t1, m1) else (t2, m2),
           (1000.0, 0),
           hits)

def is_positive(x: i32) bool = x > 0

def filter_positive(arr: [5]i32) ?k. [k]i32 =
    filter(is_positive, arr)

def filter_lambda(arr: [4]i32) ?k. [k]i32 =
    filter(|x: i32| x % 2 == 0, arr)

#[vertex]
entry vertex_main() #[builtin(position)] vec4f32 =
    let a = map_named([1, 2, 3, 4]) in
    let b = map_lambda([1, 2, 3, 4]) in
    let c = map_with_capture([1, 2, 3, 4], 10) in
    let d = reduce_sum([1.0, 2.0, 3.0, 4.0]) in
    let (t, _) = reduce_tuple([(1.0, 0), (2.0, 1), (0.5, 2), (3.0, 3)]) in
    let e = filter_positive([1, -2, 3, -4, 5]) in
    let f = filter_lambda([1, 2, 3, 4]) in
    @[d + t, f32.i32(a[0] + b[0] + c[0] + length(e) + length(f)), 0.0, 1.0]
"#,
    );
    // Compilation success is the test
}

// =============================================================================
// Defunctionalization Scenarios
// =============================================================================

#[test]
fn test_defunctionalization() {
    // Tests various defunctionalization scenarios
    let _ssa = compile_to_ssa(
        r#"
def different_captures(x: i32, y: i32, arr: [4]i32) ([4]i32, [4]i32) =
    let result1 = map(|e: i32| e + x, arr) in
    let result2 = map(|e: i32| e * y, arr) in
    (result1, result2)

def nested_capture(x: i32, arr: [4]i32) [4]i32 =
    let outer = |y: i32|
        let inner = |z: i32| x + y + z in
        inner(y)
    in
    map(outer, arr)

def reused_lambda(x: i32, arr1: [4]i32, arr2: [4]i32) ([4]i32, [4]i32) =
    let adder = |e: i32| e + x in
    let result1 = map(adder, arr1) in
    let result2 = map(adder, arr2) in
    (result1, result2)

def hof_chain(scale: i32, offset: i32, arr: [4]i32) i32 =
    let scaled = map(|x: i32| x * scale, arr) in
    let shifted = map(|x: i32| x + offset, scaled) in
    reduce(|a: i32, b: i32| a + b, 0, shifted)

#[vertex]
entry vertex_main() #[builtin(position)] vec4f32 =
    let (a, b) = different_captures(1, 2, [1, 2, 3, 4]) in
    let c = nested_capture(10, [1, 2, 3, 4]) in
    let (d, e) = reused_lambda(5, [1, 2, 3, 4], [5, 6, 7, 8]) in
    let f = hof_chain(2, 10, [1, 2, 3, 4]) in
    @[f32.i32(a[0] + b[0] + c[0] + d[0] + e[0] + f), 0.0, 0.0, 1.0]
"#,
    );
    // Compilation success is the test
}

// =============================================================================
// Type Checking Errors
// =============================================================================

#[test]
fn test_type_errors() {
    // These fail during type checking, before monomorphization, so no entry points needed

    // Arrays of functions are not permitted
    assert!(
        should_fail_type_check(
            r#"
def test: [2](i32 -> i32) =
    [|x: i32| x + 1, |x: i32| x * 2]
"#
        ),
        "Should reject arrays of functions"
    );

    // Function from if expression
    assert!(
        should_fail_type_check(
            r#"
def choose(b: bool) (i32 -> i32) =
    if b then |x: i32| x + 1 else |x: i32| x * 2
"#
        ),
        "Should reject function returned from if expression"
    );

    // Loop parameter cannot be a function
    assert!(
        should_fail_type_check(
            r#"
def test: (i32 -> i32) =
    loop f = |x: i32| x while false do f
"#
        ),
        "Should reject function as loop parameter"
    );
}

/// Companion to `test_spirv_loop_carrying_map_over_iota`, with the
/// loop initialized from a composite ARRAY LITERAL rather than a
/// `map(…, iota(…))` call. Before the fix these had asymmetric
/// outcomes: literal-init produced a Composite variant, map-init
/// produced a Virtual variant, and the loop back-edge unification
/// failed with "Loop body type must match loop variable type:
/// Failure(Virtual, Composite)". Now that `map`'s output variant is
/// pinned to Composite, both styles compile.
#[test]
fn test_spirv_loop_carrying_literal_init() {
    let source = r#"
def f(seed: f32) [4]f32 =
    let init: [4]f32 = [seed, seed, seed, seed] in
    let (_, out) =
        loop (i, arr) = (0, init) while i < 2 do
            let arr' = map(|j: i32| arr[j] + 1.0, iota(4))
            in (i + 1, arr')
    in out

#[compute]
entry main(x: []f32) [4]f32 = f(x[0])
"#;
    compile_to_spirv(source).expect(
        "loop back-edge carrying a literal-init array across a map(…, iota(…)) body \
         should compile; both init and body variants are Composite",
    );
}

/// Regression: `lift_graphical_invariant_soacs` used to look only at the
/// direct free vars of a candidate SOAC for entry-param refs. A
/// fragment-shader-local `let uv = fragCoord.x` introduces `uv` as a
/// fresh symbol that's *not* an entry param but transitively depends on
/// one. A reduce (plain or fused map→reduce) whose body reads `uv` would then be wrongly
/// classified as graphical-invariant and hoisted into a compute prepass
/// that references `@uv` as an unbound global — SPIR-V codegen panics
/// with `Unknown global: uv`. The check needs to follow let bindings
/// transitively.
#[test]
fn test_no_overhoist_fused_reduce_through_let_bound_dependency() {
    let source = r#"
def cands: [12]i32 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

#[fragment]
entry fragment_main(#[builtin(position)] fragCoord: vec4f32)
  #[target(screen)] vec4f32 =
  let uv = fragCoord.x in
  let glows = map(|i: i32| uv + f32.i32(i), cands) in
  let total = reduce(|a: f32, b: f32| a + b, 0.0, glows) in
  @[total, 0.0, 0.0, 1.0]
"#;
    compile_to_spirv(source).expect(
        "a fused reduce whose body reads a let-bound local that transitively \
         depends on an entry param must remain in the fragment shader; \
         the lift pass must not classify it as graphical-invariant",
    );
}

/// A graphical-invariant reduction may capture a lexical scalar introduced
/// before the reduction. The pre-pass must carry that definition into its own
/// scope before defunctionalization attaches the composed map/reduce capture.
#[test]
fn test_graphical_fused_reduce_carries_local_scalar_into_prepass() {
    let source = r#"
def globalData: [12]f32 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]

#[vertex]
entry vertex_main(#[builtin(vertex_index)] vid: i32)
  #[builtin(position)] vec4f32 =
  let x = 2.0 in
  let total = reduce(
    |acc: f32, value: f32| acc + value,
    0.0,
    map(|value: f32| value * x, globalData)
  ) in
  @[total, 0.0, 0.0, 1.0]
"#;

    crate::compile_thru_spirv(source).expect(
        "a fused graphical reduce must carry its invariant local scalar into \
         the generated pre-pass instead of emitting an unresolved global",
    );
}

/// Capture classification is by SymbolId, not spelling: the parameter named
/// `lightDir` shadows a top-level constant and must still be captured when its
/// map is fused into a graphical reduction.
#[test]
fn test_graphical_fused_reduce_captures_shadowing_local() {
    let source = r#"
def lightDir: vec3f32 = @[0.5, 0.5, -0.5]
def globalData: [12]f32 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]

def shade(lightDir: vec3f32) f32 =
  reduce(
    |acc: f32, value: f32| acc + value,
    0.0,
    map(|value: f32| value * lightDir.x, globalData)
  )

#[vertex]
entry vertex_main(#[builtin(vertex_index)] vid: i32)
  #[builtin(position)] vec4f32 =
  let total = shade(lightDir) in
  @[total, 0.0, 0.0, 1.0]
"#;

    compile_to_spirv(source).expect(
        "a captured parameter that shadows a top-level constant must not be \
         mistaken for that global during closure conversion",
    );
}

/// Companion to the over-hoist test above: a reduce whose only
/// non-constant dependency is a `#[uniform]` param IS graphical-invariant
/// (a uniform is constant across invocations), so it must lift into a
/// compute pre-pass. Because `#[uniform]` is entry-param-only and the
/// lift's taint set treats every entry param as per-invocation, the lift
/// has to explicitly exempt uniform params — otherwise it silently stops
/// firing for the common uniform-driven case. And since the pre-pass is a
/// separate entry, it must re-declare the uniform as its own `#[uniform]`
/// param; without that, codegen panics with `Unknown global: iTime`.
#[test]
fn test_uniform_driven_reduce_lifts_into_prepass() {
    let source = r#"
#[vertex]
entry vertex_main(#[builtin(vertex_index)] vid: i32)
  #[builtin(position)] vec4f32 =
  @[-1.0, -1.0, 0.0, 1.0]

#[fragment]
entry fragment_main(
  #[uniform(set=1, binding=0)] iTime: f32,
  #[builtin(position)] fragCoord: vec4f32
) #[target(screen)] vec4f32 =
  let samples = map(|i: i32| f32.cos(iTime + f32.i32(i)), 0..<64) in
  let breath = reduce(|a: f32, b: f32| a + b, 0.0, samples) in
  @[breath, 0.0, 0.0, 1.0]
"#;

    // The lift must fire: the uniform-driven reduce becomes a compute
    // pre-pass (multi-staged into phase1 + phase2 compute entries). Before
    // the uniform exemption there were no compute entries at all.
    let ssa = compile_to_ssa_with_modules(source);
    let compute_entries: Vec<&str> = ssa
        .entry_points
        .iter()
        .filter(|e| {
            matches!(
                e.execution_model,
                crate::ssa::types::ExecutionModel::Compute { .. }
            )
        })
        .map(|e| e.name.as_str())
        .collect();
    assert!(
        compute_entries.iter().any(|n| n.contains("prepass")),
        "uniform-driven reduce should lift into a compute pre-pass; \
         compute entries were {:?}",
        compute_entries
    );

    // ...and the lifted result must still compile: the pre-pass re-declares
    // iTime as its own uniform, so codegen resolves it (no `Unknown global`).
    compile_to_spirv(source).expect(
        "a fragment reduce depending only on a uniform must lift into a \
         compute pre-pass that re-declares the uniform and compiles to SPIR-V",
    );
}

/// Uniform discovery must include pulled local definitions, not only direct
/// free variables of the reduction. Here the reduction captures `scale`, and
/// only `scale`'s definition references the entry uniform.
#[test]
fn test_uniform_reached_through_local_prepass_dependency_is_redeclared() {
    let source = r#"
def samples: [12]f32 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]

#[fragment]
entry fragment_main(#[uniform(set=1, binding=0)] iTime: f32)
  #[target(screen)] vec4f32 =
  let scale = iTime * 2.0 in
  let total = reduce(
    |acc: f32, value: f32| acc + value,
    0.0,
    map(|value: f32| value * scale, samples)
  ) in
  @[total, 0.0, 0.0, 1.0]
"#;

    compile_to_spirv(source).expect(
        "a uniform used through a pulled local definition must be declared on \
         the generated pre-pass",
    );
}

// =============================================================================
// Materialization Optimization
// =============================================================================

#[test]
fn test_materialization_optimization() {
    // Tests that materialization hoisting works correctly
    let _ssa = compile_to_ssa(
        r#"
def identity(arr: [3]i32) [3]i32 = arr

def no_redundant_complex(arr: [3]i32, i: i32) i32 =
    if true then (identity(arr))[i] else (identity(arr))[i]

def no_materialize_tuple(x: i32) i32 =
    let pair = (x, x + 1) in
    let (a, b) = pair in
    a + b

def no_materialize_loop_tuple(arr: [10]i32) i32 =
    let (sum, _) = loop (acc, i) = (0, 0) while i < 10 do
        (acc + arr[i], i + 1)
    in sum

#[vertex]
entry vertex_main() #[builtin(position)] vec4f32 =
    let a = no_redundant_complex([1, 2, 3], 0) in
    let b = no_materialize_tuple(5) in
    let c = no_materialize_loop_tuple([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) in
    @[f32.i32(a + b + c), 0.0, 0.0, 1.0]
"#,
    );
    // Compilation success is the test
}

// =============================================================================
// Math Functions and Conversions
// =============================================================================

#[test]
fn test_math_and_conversions() {
    // Tests: f32 conversions, math operations, qualified names
    let _ssa = compile_to_ssa(
        r#"
def conversions(x: i32, y: i64) f32 =
    let f1 = f32.i32(x) in
    let f2 = f32.i64(y) in
    f1 + f2

def math_ops(x: f32) f32 =
    let a = f32.sin(x) in
    let b = f32.cos(x) in
    let c = f32.sqrt(a) in
    let d = f32.exp(b) in
    let e = f32.log(c) in
    let f = d ** 2.0f32 in
    let g = f32.sinh(x) in
    let h = f32.asinh(g) in
    let i = f32.atan2(x, a) in
    f32.fma(f, e, i)

def vector_length(v: vec2f32) f32 =
    f32.sqrt(v.x * v.x + v.y * v.y)

#[vertex]
entry vertex_main() #[builtin(position)] vec4f32 =
    let a = conversions(1, 2i64) in
    let b = math_ops(1.0) in
    let c = vector_length(@[3.0, 4.0]) in
    @[a + b + c, 0.0, 0.0, 1.0]
"#,
    );
    // Compilation success is the test
}

// =============================================================================
// Matrix Operations
// =============================================================================

#[test]
fn test_matrix_operations() {
    // Tests: mul overloads (mat*mat, mat*vec, vec*mat)
    let _ssa = compile_to_ssa(
        r#"
def test_mul(m1: mat4f32, m2: mat4f32, v: vec4f32) vec4f32 =
    let mat_result = mul(m1, m2) in
    let vec_result1 = mul(mat_result, v) in
    let vec_result2 = mul(v, m1) in
    vec_result1

#[vertex]
entry vertex_main() #[builtin(position)] vec4f32 =
    let m = @[[1.0, 0.0, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 1.0]] in
    let v = @[1.0, 2.0, 3.0, 1.0] in
    test_mul(m, m, v)
"#,
    );
    // Compilation success is the test
}

// =============================================================================
// Complex Shader Integration
// =============================================================================

#[test]
fn test_complex_shader() {
    // Full shader with uniforms, matrices, map, multiple functions
    let _ssa = compile_to_ssa(
        r#"
def verts: [3]vec4f32 =
    [@[-1.0, -1.0, 0.0, 1.0],
     @[3.0, -1.0, 0.0, 1.0],
     @[-1.0, 3.0, 0.0, 1.0]]

#[vertex]
entry vertex_main(#[builtin(vertex_index)] vertex_id: i32) #[builtin(position)] vec4f32 =
    verts[vertex_id]

def translation(p: vec3f32) mat4f32 =
    @[[1.0f32, 0.0f32, 0.0f32, p.x],
      [0.0f32, 1.0f32, 0.0f32, p.y],
      [0.0f32, 0.0f32, 1.0f32, p.z],
      [0.0f32, 0.0f32, 0.0f32, 1.0f32]]

def rotation_y(angle: f32) mat4f32 =
    let s = f32.sin(angle) in
    let c = f32.cos(angle) in
    @[[c, 0.0f32, s, 0.0f32],
      [0.0f32, 1.0f32, 0.0f32, 0.0f32],
      [0.0 - s, 0.0f32, c, 0.0f32],
      [0.0f32, 0.0f32, 0.0f32, 1.0f32]]

def cube_corners: [8]vec3f32 =
    [@[-1.0, -1.0, 1.0], @[-1.0, 1.0, 1.0],
     @[1.0, 1.0, 1.0], @[1.0, -1.0, 1.0],
     @[-1.0, -1.0, -1.0], @[-1.0, 1.0, -1.0],
     @[1.0, 1.0, -1.0], @[1.0, -1.0, -1.0]]

def main_image(res: vec2f32, time: f32, fragCoord: vec2f32) vec4f32 =
    let cam = translation(@[0.0, 0.0, 10.0]) in
    let rot = rotation_y(time) in
    let mat = rot * cam in
    let v4s = map(|v: vec3f32| @[v.x, v.y, v.z, 1.0] * mat, cube_corners) in
    v4s[0]

#[fragment]
entry fragment_main(#[uniform(set=1, binding=0)] iResolution: vec2f32, #[uniform(set=1, binding=1)] iTime: f32, #[builtin(frag_coord)] pos: vec4f32) #[target(screen)] vec4f32 =
    main_image(@[iResolution.x, iResolution.y], iTime, @[pos.x, pos.y])
"#,
    );
    // Compilation success is the test
}

// =============================================================================
// Full Pipeline to SPIR-V
// =============================================================================

#[test]
fn test_function_call_with_array_arg() {
    // Test calling a function with an array literal argument
    let source = r#"
def sum_first_two(arr: [4]i32) i32 =
    arr[0] + arr[1]

#[vertex]
entry vertex_main() #[builtin(position)] vec4f32 =
    let result = sum_first_two([1, 2, 3, 4]) in
    @[f32.i32(result), 0.0, 0.0, 1.0]
"#;

    let result = crate::compile_thru_spirv(source);

    assert!(result.is_ok(), "SPIR-V compilation failed: {:?}", result.err());
}

#[test]
fn test_compute_shader_with_storage_slice() {
    // Test compute shader with storage buffer slice
    let source = r#"
def sum_first_two(arr: [4]i32) i32 =
    arr[0] + arr[1]

#[compute]
entry compute_main(data: []i32) i32 =
    let from_storage = sum_first_two(data[0..4]) in
    let from_literal = sum_first_two([1, 2, 3, 4]) in
    from_storage + from_literal
"#;

    let result = crate::compile_thru_spirv(source);

    assert!(result.is_ok(), "SPIR-V compilation failed: {:?}", result.err());
}

#[test]
fn test_full_pipeline_to_spirv() {
    // Verify the full pipeline compiles successfully to SPIR-V
    let source = r#"
def compute(x: f32, y: f32) f32 =
    let a = f32.sin(x) in
    let b = f32.cos(y) in
    a + b

#[fragment]
entry fragment_main(#[uniform(set=1, binding=0)] iTime: f32, #[builtin(position)] pos: vec4f32) #[target(screen)] vec4f32 =
    let s = compute(pos.x, pos.y) in
    @[s + iTime, 0.0, 0.0, 1.0]
"#;

    let result = crate::compile_thru_spirv(source);

    assert!(result.is_ok(), "SPIR-V compilation failed: {:?}", result.err());
}

/// Spec §x binop y: `f32 ** i32` (float base, integer exponent) must
/// type-check, lower to valid SPIR-V, and route through `OpConvertSToF`
/// before `GLSL Pow`. Use exponent `9` to skip the EGIR fold's
/// `2..8` constant-power-to-mul-chain rewrite, forcing the
/// backend-conversion path.
#[test]
fn pow_float_base_int_exp_lowers_via_convert_then_pow() {
    use wspirv::binary::parse_words;
    use wspirv::dr::Loader;
    use wspirv::spirv::Op;

    let spirv = compile_to_spirv(
        "\
#[compute]
entry e(xs: []f32) []f32 = map(|x: f32| x ** 9, xs)
",
    )
    .expect("f32 ** i32 (exp=9) compiles to SPIR-V");

    let mut loader = Loader::new();
    parse_words(&spirv, &mut loader).expect("parse spirv");
    let module = loader.module();

    let mut converts = 0;
    let mut pows = 0;
    for func in &module.functions {
        for block in &func.blocks {
            for inst in &block.instructions {
                match inst.class.opcode {
                    Op::ConvertSToF => converts += 1,
                    Op::ExtInst => {
                        // GLSL.std.450 Pow = opcode 26 (operand index 1).
                        if let Some(wspirv::dr::Operand::LiteralExtInstInteger(26)) = inst.operands.get(1) {
                            pows += 1;
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    assert!(
        converts >= 1,
        "expected at least one OpConvertSToF to coerce i32 exponent, found {converts}"
    );
    assert!(pows >= 1, "expected at least one GLSL Pow ext-inst, found {pows}");
}

#[test]
fn mul_all_three_overloads_compile_to_spirv() {
    // `mul` has three overloads with three different `PrimOp`s
    // (MatrixTimesMatrix / MatrixTimesVector / VectorTimesMatrix).
    // `tlc::specialize` rewrites every `mul(a, b)` call into
    // `BinOp("*")(a, b)`; the BinOp lowering then picks the right
    // SPIR-V op based on operand shapes. This pins the wiring end-to-
    // end: a single shader exercising all three call shapes must
    // compile through to valid SPIR-V.
    let source = r#"
def m1 = @[[1.0f32, 0.0f32], [0.0f32, 1.0f32]]
def m2 = @[[2.0f32, 0.0f32], [0.0f32, 2.0f32]]

#[fragment]
entry fragment_main(#[builtin(position)] pos: vec4f32) #[target(screen)] vec4f32 =
    let a: mat2f32 = m1 in
    let b: mat2f32 = m2 in
    let v: vec2f32 = @[pos.x, pos.y] in
    let mm: mat2f32 = mul(a, b) in
    let mv: vec2f32 = mul(mm, v) in
    let vm: vec2f32 = mul(v, mm) in
    @[mv.x, mv.y, vm.x, vm.y]
"#;
    let result = crate::compile_thru_spirv(source);
    assert!(
        result.is_ok(),
        "all three mul overloads should compile to SPIR-V: {:?}",
        result.err()
    );
}

// =============================================================================
// Array Variant Monomorphization
// =============================================================================

#[test]
fn test_array_variant_monomorphization() {
    // Slicing a storage view with constant bounds stays a View; the call below
    // specializes sum_first_two separately from the array-literal Composite
    // call site instead of materializing the slice.
    let ssa = compile_to_ssa(
        r#"
def sum_first_two(arr: [4]i32) i32 =
    arr[0] + arr[1]

#[compute]
entry compute_main(data: []i32) i32 =
    let from_storage = sum_first_two(data[0..4]) in
    let from_literal = sum_first_two([1, 2, 3, 4]) in
    from_storage + from_literal
"#,
    );

    // Collect all sum_first_two variants (including buffer-specialized)
    let sum_versions: Vec<_> =
        ssa.functions.iter().filter(|f| f.name.starts_with("sum_first_two")).collect();

    eprintln!("sum_first_two SSA functions:");
    for f in &sum_versions {
        eprintln!("  {}", f.name);
        // Show param types
        for (val, ty, name) in &f.body.params {
            eprintln!("    param {} ({:?}) :: {:?}", name, val, ty);
        }
        // Show all instructions that involve indexing or storage views
        for inst in f.body.inner.insts.values() {
            match &inst.data {
                crate::ssa::types::InstKind::Op {
                    tag: crate::op::OpTag::Index,
                    ..
                } => {
                    eprintln!("    inst {:?}: Index", inst.result);
                }
                crate::ssa::types::InstKind::Op {
                    tag: crate::op::OpTag::StorageView(_),
                    ..
                } => {
                    eprintln!("    inst {:?}: StorageView", inst.result);
                }
                crate::ssa::types::InstKind::ViewIndex { .. } => {
                    eprintln!("    inst {:?}: ViewIndex", inst.result);
                }
                _ => {}
            }
        }
    }

    // After TLC-level inlining and DCE, sum_first_two may be fully inlined
    // at all call sites and eliminated. The important thing is that the program
    // compiles successfully to SSA with both View and Composite call shapes.
    // (The function may or may not survive depending on inlining thresholds.)
}

// =============================================================================
// SPIR-V Block Param / Phi Node Tests
// =============================================================================

/// Compile source all the way through SPIR-V and return Ok/Err.
fn compile_to_spirv(input: &str) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    Ok(crate::compile_thru_spirv(input)?.spirv)
}

/// Single-stage equivalent of `compile_to_spirv` — disables
/// `parallelize_soacs`, matching the CLI's `--single-stage` mode.
fn compile_to_spirv_single_stage(input: &str) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    Ok(crate::compile_thru_spirv_single_stage(input)?.spirv)
}

/// Helper: compile source through SSA. Same as `compile_to_ssa`; kept as
/// a separate name because some legacy tests distinguished module-bearing
/// programs from non-module ones — `compile_thru_frontend` handles both.
fn compile_to_ssa_with_modules(input: &str) -> Program {
    crate::compile_thru_ssa(input).expect("compile to SSA").ssa
}

// =========================================================================
// Backend gaps (aspirational, #[ignore]d)
//
// Each test asserts the *desired* code-gen outcome for a construct the SPIR-V
// backend currently can't handle; `#[ignore]`d so the suite stays green.
// Surfaced while building the lib/ statistics generators. Drop the `#[ignore]`
// when the gap is closed.
// =========================================================================

/// Returning a runtime-sized `[]f32` from a helper and reading one *constant*
/// slot. `g` inlines to `map(|i| f32.i32(i), 0..<256)`, and `static_index_fusion`
/// rewrites `map(f, src)[3]` → `let i = src[3] in f32.i32(i)` — a virtual-array
/// access, materializing nothing rather than a whole runtime-sized buffer.
#[test]
fn returning_runtime_sized_array_from_fn_lowers() {
    let source = r#"
def g(n: i32) []f32 = map(|i: i32| f32.i32(i), 0i32 ..< n)
#[compute]
entry e() [1]f32 = [g(256)[3]]
"#;
    compile_to_spirv(source).expect("returning a runtime-sized array should lower to SPIR-V");
}

/// The runtime counterpart of the static fusion above: a *runtime* index into a
/// nested runtime-sized producer (`g(256)[j]`). With no fused form (fusion is
/// literal-index only), `lift_gathers`'s post-materialize pass floats the nested
/// producer to an entry-level `let` (inlining its internal `let n = 256` so the
/// map is self-contained) and materializes it to a gather buffer; the runtime
/// index then reads the buffer. Distinct from the static case, which never
/// materializes.
#[test]
fn runtime_index_into_nested_producer_lowers() {
    let source = r#"
def g(n: i32) []f32 = map(|i: i32| f32.i32(i), 0i32 ..< n)
#[compute]
entry e(j: i32) [1]f32 = [g(256)[j]]
"#;
    compile_to_spirv(source)
        .expect("a runtime index into a nested runtime-sized producer should materialize + lower");
}

/// Gap: a runtime-sized array with *two or more* consumers panics the backend
/// with "Composite variant unsized arrays not supported" (`spirv/mod.rs`).
/// A single-consumer runtime-sized array fuses into its consumer and never
/// materializes (`f32.sum(map(…, 0..<n))` lowers fine), but binding it and
/// reading it more than once forces materialization of an unsized Composite
/// array, which the type lowering rejects. This is what blocks the lib/ `Stats`
/// gatherer, whose sample array feeds `sum`, a deviation `map`, `minimum`, and
/// `maximum`. Distinct from `returning_runtime_sized_array_from_fn_lowers`,
/// which is about *returning* such an array.
#[test]
fn runtime_sized_array_with_multiple_consumers_lowers() {
    let source = r#"
def g(n: i32) f32 =
    let xs = map(|i: i32| f32.i32(i), 0i32 ..< n) in
    f32.sum(xs) + f32.maximum(xs)
#[compute]
entry e() f32 = g(256)
"#;
    compile_to_spirv(source)
        .expect("a runtime-sized array read by multiple consumers should lower to SPIR-V");
}

/// Regression for the named-callee HOF gap. The outer `fbm2(perlin2, …)`
/// is specialized cleanly by the existing main loop, but `fbm2`'s body
/// also closes over `noise` via a lifted SOAC lambda (closure_convert
/// adds the function-typed capture as a parameter on the lifted def).
/// The main loop's substitution only rewrites direct `noise(…)` references
/// in `fbm2`'s surface body, not the lifted def's signature — so the
/// captured `noise` still survives as an arrow-typed parameter and the
/// verifier rejects it. Fixed by adding a cascade closure-specialization
/// step in `hof_specialize::run` that walks every reachable def, finds
/// `SoacBody`s whose captures include `(_, arrow_ty, Var(known_callable))`,
/// clones the lifted def with the callable substituted into the body,
/// and drops the callable param from its signature. Lets `lib/noise.wyn`
/// collapse its four `fbm_<kind>` defs into one generic `fbm2`.
#[test]
fn function_typed_param_with_named_callee_specializes() {
    let source = r#"
def perlin2(k: u32, p: vec2f32) f32 = f32.u32(k) + p.x
def fbm2(noise: u32 -> vec2f32 -> f32, k: u32, p: vec2f32, n: i32) f32 =
  reduce(|a: f32, b: f32| a + b, 0.0f32,
    map(|i: i32| noise(k, p) * f32.i32(i), 0i32 ..< n))
def fbm_perlin(k: u32, p: vec2f32, n: i32) f32 = fbm2(perlin2, k, p, n)
#[compute]
entry e() f32 = fbm_perlin(1u32, @[0.0f32, 0.0f32], 4i32)
"#;
    compile_to_spirv(source)
        .expect("a named def passed as a function-typed argument should be specialized away");
}

/// A local lambda closing over an *enclosing function's parameter*, applied
/// more than once, must lower. `partial_eval` inlines the enclosing function
/// and residualizes the lambda (it can't beta-reduce a value used twice); the
/// captured param has to be substituted into that residual body, otherwise it
/// survives as a free var and SPIR-V gen fails with `Unknown global: k`. Fixed
/// by substituting env-bound captures into residual lambda bodies in
/// `partial_eval::eval`. Surfaced writing the perlin-noise-fields playground,
/// where `let fb = |q| fhnoise.fbm_perlin(k, q, …)` is called five times.
#[test]
fn local_lambda_capturing_param_applied_twice_lowers() {
    let source = r#"
def f(k: u32, p: f32) f32 =
  let g = |q: f32| q + f32.u32(k) in
  g(p) + g(p + 1.0f32)

#[compute]
entry e() f32 = f(3u32, 1.0f32)
"#;
    compile_to_spirv(source).expect(
        "a local lambda closing over an enclosing fn parameter must lower even \
         when applied more than once",
    );
}

/// The reified operator members on the builtin numeric modules — `i32.(+)`,
/// `f32.(<)`, `u32.(<<)`, etc. — resolve as real module functions and lower.
/// The `(op)` form is the function reification of the primitive infix BinOp;
/// its body is the BinOp itself, so it lowers through the same path as `a + b`.
#[test]
fn reified_numeric_operators_lower_to_spirv() {
    let source = r#"
#[compute]
entry arith() i32 =
    i32.(+)(i32.(*)(i32.(-)(10i32, 3i32), 4i32), i32.(%)(9i32, 5i32))
#[compute]
entry bits() u32 = u32.(>>)(u32.(<<)(u32.(^)(u32.(&)(255u32, 15u32), 8u32), 2u32), 1u32)
#[compute]
entry cmp(x: f32, y: f32) i32 =
    let a = if f32.(==)(x, y) then 1i32 else 0i32 in
    let b = if f32.(!=)(x, y) then 2i32 else 0i32 in
    let c = if f32.(<)(x, y) then 4i32 else 0i32 in
    let d = if f32.(>)(x, y) then 8i32 else 0i32 in
    let e = if f32.(<=)(x, y) then 16i32 else 0i32 in
    let f = if f32.(>=)(x, y) then 32i32 else 0i32 in
    a + b + c + d + e + f
"#;
    compile_to_spirv(source).expect("reified numeric operator members should lower to SPIR-V");
}

/// The payoff of reifying operators into real module members: an operator can
/// be passed as a first-class value to a higher-order function (Wyn forbids
/// partial application, but operator members are saturated function references,
/// which it does support). Before reification `i32.(+)` had no referent.
#[test]
fn reified_operator_passed_as_first_class_value() {
    let source = r#"
#[compute]
entry sum(xs: [16]i32) i32 = reduce(i32.(+), 0i32, xs)
"#;
    compile_to_spirv(source).expect("a reified operator member should be passable to a HOF");
}

/// A top-level `def map` shadowing the `map` SOAC is a normal call, not a SOAC.
/// SOAC identity is decided by the frontend resolver (structurally, respecting
/// shadowing) rather than re-derived by string match in TLC — so the user's
/// one-argument `map` type-checks and lowers instead of panicking as the
/// two-argument SOAC. `reduce` is exercised alongside to confirm the genuine
/// SOACs still resolve when not shadowed.
#[test]
fn user_def_shadowing_soac_is_a_normal_call() {
    let source = r#"
def map(x: i32) i32 = x + 1
#[compute]
entry e(xs: [8]i32) i32 = reduce(i32.(+), map(0i32), xs)
"#;
    compile_to_spirv(source).expect("a user def shadowing a SOAC name should lower as a normal call");
}

/// Regression: a user `def map(x: i32) i32 = x` at file scope must
/// not break prelude `unzip`'s `map(|...|, xys)` call. Both `unzip`
/// and the user reference `map` by surface name, but `name_resolution`
/// structurally tags the prelude reference as `Soac(Map)` while the
/// user reference is left bare. Post env-split, prelude bodies are
/// checked under `LookupContext::Prelude`, which never sees user
/// file-scope; the structural Soac tag routes directly to
/// `globals.builtins["map"]` so the SOAC scheme resolves regardless
/// of what the user did at file scope.
#[test]
fn user_def_shadowing_map_does_not_break_prelude_unzip() {
    let source = r#"
def map(x: i32) i32 = x + 1
#[compute]
entry e(xs: [4](i32, i32)) i32 =
    let (xs0, xs1) = unzip(xs) in
    reduce(i32.(+), 0i32, xs0) + reduce(i32.(+), 0i32, xs1)
"#;
    compile_to_spirv(source)
        .expect("user `def map` must not interfere with prelude unzip's internal `map` call");
}

/// Companion to `aspiration_user_module_body_sees_file_scope_shadow_of_soac`
/// — the env-split makes user file-scope visible inside user module
/// bodies, so a `def map(x: i32) [4]i32` shadows the SOAC `map` for a
/// module's `map(xs)` call too. Pre-env-split, the module body saw
/// only its own siblings + catalog and the call resolved as the SOAC.
/// This test pins the new shadowing behaviour with a multi-line
/// transitive `def map` body so an inline-small pass can't collapse
/// the call before TLC observes it.
#[test]
fn user_def_shadowing_map_reaches_into_user_module_body() {
    let source = r#"
def map(xs: [4]i32) i32 = xs[0] + xs[1] + xs[2] + xs[3]
module m = {
  def first_four_sum(xs: [4]i32) i32 = map(xs)
}
#[compute]
entry e(xs: [4]i32) i32 = m.first_four_sum(xs)
"#;
    compile_to_spirv(source).expect(
        "after env-split, a user module body's `map(xs)` resolves to the user `def map` \
         that shadows the SOAC at file scope",
    );
}

/// Regression for the env-split. After the conversion landed, user
/// module bodies see user file-scope shadows of SOAC names at the
/// surface level — `name_resolution` seeds user file-scope into each
/// user-defined module's `module_scope`, so a bare `map(xs[0])` inside
/// `m.first_doubled` resolves to the user `def map`, not the SOAC.
#[test]
fn aspiration_user_module_body_sees_file_scope_shadow_of_soac() {
    let source = r#"
def map(x: i32) i32 = x * 2
module m = {
  def first_doubled(xs: [4]i32) i32 = map(xs[0])
}
#[compute]
entry e(xs: [4]i32) i32 = m.first_doubled(xs)
"#;
    compile_to_spirv(source).expect(
        "user `def map(x: i32)` at file scope should shadow the SOAC `map` inside a \
         user module body (env-split aspiration)",
    );
}

/// The `numeric` whole-array reductions `sum`/`product`/`minimum`/`maximum` are
/// implemented (for the float modules) as `reduce` over the per-type operator
/// and its neutral, so they lower to real SPIR-V reduction loops.
#[test]
fn numeric_array_reductions_lower_to_spirv() {
    let source = r#"
def N: i32 = 256
#[compute]
entry e() [4]f32 =
    let xs = map(|i: i32| f32.i32(i), 0i32 ..< N) in
    [f32.sum(xs), f32.product(xs), f32.minimum(xs), f32.maximum(xs)]
"#;
    compile_to_spirv(source).expect("numeric array-reductions should lower to SPIR-V");
}

/// The statistics-gatherer shape that motivated the reductions: reduce a sample
/// stream to `[count, mean, variance, stddev, min, max]` using `f32.sum`,
/// `f32.minimum`, `f32.maximum`. This is the lib/ `Stats` summarize body.
#[test]
fn statistics_gatherer_lowers() {
    let source = r#"
def N: i32 = 256
#[compute]
entry summarize() [6]f32 =
    let xs = map(|i: i32| f32.i32(i), 0i32 ..< N) in
    let n = f32.i32(N) in
    let mean = f32.sum(xs) / n in
    let sq = map(|v: f32| (v - mean) * (v - mean), xs) in
    let variance = f32.sum(sq) / n in
    [n, mean, variance, f32.sqrt(variance), f32.minimum(xs), f32.maximum(xs)]
"#;
    compile_to_spirv(source).expect("the statistics gatherer should lower to SPIR-V");
}

/// Regression for an elaborate-pass bug: `emit_storage_store` interns the
/// output's `view_index` access chain, so both arms of an `if`-then-`else`
/// writing the same output slot share one hashconsed `ViewIndex` node.
/// `demand_place`'s cache (`elaborated_places`) wasn't scope-pushed per
/// subtree, so the access-chain instruction landed in whichever arm
/// demanded it first; the sibling arm's store then referenced a place
/// defined in a non-dominating block ("place … has no pointer"). Fix:
/// scope `elaborated_places` alongside `elaborated` in `elaborate_subtree`
/// so per-arm cache entries pop with the arm, and the second arm re-emits
/// its own access chain. The bitwise `&` here matters only because the
/// arithmetic version constant-folds through `partial_eval`; see
/// `branch_with_let_terminal_into_output_slot_lowers` for the
/// fold-resistant parameter-driven repro.
#[test]
fn bitwise_in_deep_let_chain_feeding_if_lowers() {
    let source = r#"
#[compute]
entry t() i32 =
    let s = 2i32 + 3i32 in
    let p = s * 4i32 in
    let x = p & 12i32 in
    if x < 100i32 then x else 0i32
"#;
    compile_to_spirv(source)
        .expect("bitwise result threaded through a deep let-chain into an if should lower");
}

/// Minimal repro for the same `elaborate_subtree` place-cache bug. The
/// runtime parameter `n` keeps the `if` branch live (`partial_eval` can't
/// fold it), so both arms emit `OutputSlotStore` against the same output
/// slot's `view_index`. Pre-fix this panicked at SPIR-V emission with
/// "place … has no pointer".
#[test]
fn branch_with_let_terminal_into_output_slot_lowers() {
    let source = r#"
#[compute]
entry t(n: i32) i32 =
    let x = n + 1i32 in
    if x < 100i32 then x else 0i32
"#;
    compile_to_spirv(source).expect("both arms of an if writing the same output slot should lower");
}

/// Verify that nested if/else chains compile to SPIR-V.
#[test]
fn test_spirv_nested_if_else_block_params() {
    let source = r#"
def choose(a: f32, b: f32, c: f32, sel1: i32, sel2: i32) f32 =
    let x = if sel1 == 0 then a
            else if sel1 == 1 then b
            else c in
    let y = if sel2 == 0 then a
            else if sel2 == 1 then c
            else b in
    x + y

#[fragment]
entry fragment_main(#[builtin(position)] pos: vec4f32) #[target(screen)] vec4f32 =
    let r = choose(pos.x, pos.y, pos.z, 1, 2) in
    @[r, 0.0, 0.0, 1.0]
"#;
    compile_to_spirv(source).expect("Nested if/else should compile to SPIR-V");
}

/// Verify many conditional branches producing block params compile to SPIR-V.
#[test]
fn test_spirv_many_conditional_block_params() {
    let source = r#"
def process(a: f32, b: f32, c: f32, d: f32, flag: i32) (f32, f32, f32, f32) =
    let x = if flag == 0 then a + b else a - b in
    let y = if flag == 1 then b + c else b * c in
    let z = if flag == 2 then c + d else c - d in
    let w = if flag == 0 then d * a else d + a in
    (x, y, z, w)

def combine(t: (f32, f32, f32, f32)) f32 =
    let (a, b, c, d) = t in
    a + b + c + d

#[fragment]
entry fragment_main(#[builtin(position)] pos: vec4f32) #[target(screen)] vec4f32 =
    let result = process(pos.x, pos.y, pos.z, pos.w, 1) in
    let s = combine(result) in
    @[s, 0.0, 0.0, 1.0]
"#;
    compile_to_spirv(source).expect("Many conditionals should compile to SPIR-V");
}

/// Verify maps over small arrays with nested conditionals compile to SPIR-V.
#[test]
fn test_spirv_map_with_nested_conditionals() {
    let source = r#"
def selectValue(x: f32, flag: i32) f32 =
    if flag == 0 then x * 2.0
    else if flag == 1 then x + 1.0
    else x - 1.0

#[compute]
entry compute_main(data: [8]f32) [8]f32 =
    map(|x| selectValue(x, 1), data)
"#;
    compile_to_spirv(source).expect("Map with nested conditionals should compile to SPIR-V");
}

/// Verify multiple maps followed by a reduce compile to SPIR-V.
#[test]
fn test_spirv_multiple_maps_and_reduce() {
    let source = r#"
#[compute]
entry compute_main(data: [8](f32, f32)) [8]f32 =
    let first = map(|t| let (a, _) = t in a, data) in
    let second = map(|t| let (_, b) = t in b, data) in
    let combined = map(|(a, b): (f32, f32)| a + b, zip(first, second)) in
    let total = reduce(|a: f32, b: f32| a + b, 0.0, combined) in
    map(|x| x + total, combined)
"#;
    compile_to_spirv(source).expect("Multiple maps + reduce should compile to SPIR-V");
}

/// Verify conditional array element selection compiles to SPIR-V
/// (the finalOrigins/finalDirs pattern from raytrace.wyn).
#[test]
fn test_spirv_conditional_array_construction() {
    let source = r#"
def build(a: [4]f32, b: [4]f32, flags: [4]i32) [4]f32 =
    [
        if flags[0] == 1 then b[0] else a[0],
        if flags[1] == 1 then b[1] else a[1],
        if flags[2] == 1 then b[2] else a[2],
        if flags[3] == 1 then b[3] else a[3]
    ]

#[compute]
entry compute_main(data: [4]f32) [4]f32 =
    let doubled = map(|x| x * 2.0, data) in
    let flags = [1, 0, 1, 0] in
    build(data, doubled, flags)
"#;
    compile_to_spirv(source).expect("Conditional array construction should compile to SPIR-V");
}

/// Regression: mapping a lambda whose return type is a mixed
/// scalar/vector tuple. The SoA transform rewrites the output
/// `[N](f32, i32, vec3f32)` into a tuple-of-arrays before EGIR
/// conversion; `egir::soac_expand` must split the per-iteration
/// ArrayWith into per-component ArrayWith calls + a Tuple repack.
/// Without the split, SPIR-V lowering's runtime-index path hit a
/// cache miss and failed with "element type not found".
#[test]
fn test_spirv_map_array_of_mixed_tuple() {
    let source = r#"
def build(xs: [8]f32) [8](f32, i32, vec3f32) =
    map(|x: f32| (x + 1.0, 0, @[x, x, x]), xs)

#[fragment]
entry frag(c: vec4f32) vec4f32 =
    let arr = build([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]) in
    let (a, _, v) = arr[3] in
    @[a, v.x, v.y, v.z]
"#;
    compile_to_spirv(source).expect("map over [N](f32, i32, vec3f32) should compile to SPIR-V");
}

/// Regression: nested SoA. The element type itself contains a
/// composite array, so the SoA transform produces a tuple of
/// arrays whose components are themselves arrays — exercising
/// `emit_write_element`'s recursion through `soa_element_type`.
#[test]
fn test_spirv_map_array_of_nested_tuple() {
    let source = r#"
def build(xs: [4]f32) [4](f32, [3]f32) =
    map(|x: f32| (x + 1.0, [x, x, x]), xs)

#[fragment]
entry frag(c: vec4f32) vec4f32 =
    let arr = build([0.0, 1.0, 2.0, 3.0]) in
    let (a, inner) = arr[2] in
    @[a, inner[0], inner[1], inner[2]]
"#;
    compile_to_spirv(source).expect("map over [N](f32, [M]f32) should compile to SPIR-V");
}

/// Regression: a loop that carries an array whose next-iteration
/// value comes from `map(…, iota(N))`.
///
/// Before the fix, `map`'s type scheme claimed the output variant
/// was the same as the input's. Since `iota` returns a Virtual
/// array, `map`'s output was typed Virtual — but `egir::soac_expand`
/// actually materializes the result via `_w_intrinsic_uninit` +
/// `_w_intrinsic_array_with_inplace`, which is a Composite
/// representation. That type/representation mismatch surfaced as
/// a SPIR-V "ArrayWith: element type not found" cache miss when
/// the loop back-edge carried what SSA thought was a Virtual
/// array. Fix: pin `map` / `scan` / `filter` output variant to
/// Composite in the type scheme, matching the runtime
/// representation.
#[test]
fn test_spirv_loop_carrying_map_over_iota() {
    let source = r#"
def f(seed: f32) [4]f32 =
  let init: [4]f32 = map(|j: i32| seed + f32.i32(j), iota(4)) in
  let (_, out) =
    loop (i, arr) = (0, init) while i < 2 do
      let arr' = map(|j: i32| arr[j] + 1.0, iota(4))
      in (i + 1, arr')
  in out

#[compute]
entry main(x: []f32) [4]f32 = f(x[0])
"#;
    compile_to_spirv(source).expect(
        "loop carrying `map(..., iota(N))` should compile; currently fails with \
         ArrayWith cache miss because the back-edge array is Virtual variant",
    );
}

/// Test the specific raytrace.wyn file compiles to SPIR-V.
#[test]
fn test_spirv_raytrace() {
    let source = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../testfiles/playground/raytrace.wyn"
    ))
    .expect("Could not read testfiles/playground/raytrace.wyn");
    compile_to_spirv(&source).expect("raytrace.wyn should compile to SPIR-V");
}

/// Regression: if/else before interprocedural map+reduce fusion caused
/// UnterminatedBlock in soac_lower. The if/else creates a dead Unreachable
/// block in SSA, which soac_lower's rebuild would pre-create but finish()
/// rejected because Unreachable doubles as the "unterminated" sentinel.
#[test]
fn test_interproc_fusion_if_before_fused_reduce() {
    let source = r#"
def maxDist: f32 = 100.0
def globalData: [12]f32 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]

def producer(x: f32) [12]f32 =
  map(|a: f32| a * x, globalData)

def consumer(arr: [12]f32) f32 =
  reduce(|acc: f32, x: f32| if acc < x then acc else x, maxDist, arr)

def scene(x: f32, y: f32) f32 =
  let ground = if y > 0.0 then y else maxDist in
  let hits = producer(x) in
  let closest = consumer(hits) in
  closest + ground

#[vertex]
entry vertex_main(#[builtin(vertex_index)] vid: i32) #[builtin(position)] vec4f32 =
  let r = scene(1.0, 0.5) in
  @[r, 0.0, 0.0, 1.0]
"#;
    compile_to_spirv(source).expect("if-before-interproc-fusion should compile");
}

/// Verify raytrace.wyn compiles through SSA to SPIR-V without errors.
/// This exercises the RPO block emission and incremental array literal
/// lowering that were needed for complex cross-block value references.
/// (test_spirv_raytrace covers this; this test verifies the SSA is well-formed
/// enough that compile_to_ssa_with_modules succeeds.)
#[test]
fn test_ssa_raytrace_well_formed() {
    let source = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../testfiles/playground/raytrace.wyn"
    ))
    .expect("Could not read testfiles/playground/raytrace.wyn");

    let ssa = compile_to_ssa_with_modules(&source);

    // SOAC-bearing helpers such as `trace` are intentionally force-inlined
    // before SSA and then removed by DCE. Verify the durable contract instead:
    // both graphical entry points survived and SSA construction completed.
    assert!(
        ssa.entry_points.iter().any(|entry| entry.name == "vertex_main"),
        "vertex_main should be in SSA output"
    );
    assert!(
        ssa.entry_points.iter().any(|entry| entry.name == "fragment_main"),
        "fragment_main should be in SSA output"
    );
}

// =============================================================================
// Constant Inlining
// =============================================================================

/// Constants that reference other constants should be fully inlined.
#[test]
fn test_constant_referencing_constant() {
    let source = r#"
def PI: f32 = 3.14159265
def TAU: f32 = PI * 2.0
def QUARTER_TAU: f32 = TAU / 4.0

#[fragment]
entry frag(#[builtin(position)] pos: vec4f32) #[target(screen)] vec4f32 =
  @[QUARTER_TAU, PI, TAU, 1.0]
"#;
    compile_to_spirv(source).expect("constants referencing constants should compile");
}

// `test_soa_eliminates_extraction_maps` was deleted: it inspected post-SSA for
// `InstKind::Soac(Soac::Map { .. })`, but SOACs no longer exist as SSA
// instructions — they're expanded inside EGIR (`egir::soac_expand`) before
// elaboration produces any SSA. The underlying optimization concern (extraction
// maps after a zip-map should fold to tuple projections) is now an EGIR-level
// question and would need a different shape of test.

/// Pipeline that includes TLC inline_small (now always part of the mainline).
fn compile_to_ssa_with_inline_small(input: &str) -> Program {
    compile_to_ssa(input)
}

#[test]
fn test_constant_inlining_global_ref() {
    // Minimal repro: a constant def used by a function, going through inline_small.
    // This should NOT produce an unresolved Global("PI") in SSA.
    let ssa = compile_to_ssa_with_inline_small(
        r#"
def PI: f32 = 3.141592

def use_pi(x: f32) f32 = x * PI

#[fragment]
entry fragment_main(#[builtin(position)] pos: vec4f32) #[target(screen)] vec4f32 =
    let r = use_pi(pos.x) in
    @[r, 0.0, 0.0, 1.0]
"#,
    );

    // Dump what we got.
    eprintln!("{}", crate::ssa::print::format_program(&ssa));

    // Check that no Global("PI") instruction exists — it should have been inlined.
    for func in &ssa.functions {
        for (_id, inst) in &func.body.inner.insts {
            if let crate::ssa::types::InstKind::Op {
                tag: crate::op::OpTag::Global(name),
                ..
            } = &inst.data
            {
                assert_ne!(
                    name, "PI",
                    "Global @PI should have been inlined, but survived in function '{}'",
                    func.name
                );
            }
        }
    }
    for ep in &ssa.entry_points {
        for (_id, inst) in &ep.body.inner.insts {
            if let crate::ssa::types::InstKind::Op {
                tag: crate::op::OpTag::Global(name),
                ..
            } = &inst.data
            {
                assert_ne!(
                    name, "PI",
                    "Global @PI should have been inlined, but survived in entry '{}'",
                    ep.name
                );
            }
        }
    }
}

// ============================================================================
// `--fill-holes`: type-hole default fill
// ============================================================================

/// Compile through TLC with `fill_holes = true`; return the resulting
/// `TlcTransformed` so tests can inspect `fill_hole_errors` and the
/// program shape.
fn compile_tlc_with_fill_holes(input: &str) -> crate::TlcTransformed {
    let (mut node_counter, mut module_manager) = crate::cached_compiler_init();
    let parsed = Compiler::parse(input, &mut node_counter).expect("parse");
    let type_checked = parsed
        .resolve(&mut module_manager)
        .expect("resolve")
        .fold_ast_constants()
        .type_check(&mut module_manager)
        .expect("type_check");
    type_checked.to_tlc(&module_manager, true)
}

#[test]
fn fill_holes_numeric_scalars_compile_clean() {
    // Scalar holes (i32 / f32 / bool) default to 0 / 0.0 / false and
    // compile through with no fill-hole errors.
    for src in ["def x: i32 = ???", "def y: f32 = ???", "def z: bool = ???"] {
        let tlc = compile_tlc_with_fill_holes(src);
        assert!(
            tlc.0.fill_hole_errors.is_empty(),
            "scalar hole in `{}` should fill cleanly: {:?}",
            src,
            tlc.0.fill_hole_errors
        );
    }
}

#[test]
fn fill_holes_vec_compiles_clean() {
    let tlc = compile_tlc_with_fill_holes("def v: vec3f32 = ???");
    assert!(
        tlc.0.fill_hole_errors.is_empty(),
        "vec3 hole should fill cleanly: {:?}",
        tlc.0.fill_hole_errors
    );
}

#[test]
fn fill_holes_rejects_function_type() {
    let tlc = compile_tlc_with_fill_holes("def f: i32 -> i32 = ???");
    assert!(
        !tlc.0.fill_hole_errors.is_empty(),
        "function-typed hole should surface a fill-hole error"
    );
    let msg = format!("{:?}", tlc.0.fill_hole_errors[0]);
    assert!(
        msg.contains("function value") || msg.contains("Arrow"),
        "error should mention function type: {}",
        msg
    );
}

#[test]
fn fill_holes_respects_inferred_type_from_context() {
    // Hole's type is inferred from the enclosing context (array
    // element type here). Default-fill fires at the inferred type.
    let tlc = compile_tlc_with_fill_holes("def arr: [3]i32 = [1i32, ???, 3i32]");
    assert!(
        tlc.0.fill_hole_errors.is_empty(),
        "hole in i32 array should fill as i32 cleanly: {:?}",
        tlc.0.fill_hole_errors
    );
}

// =============================================================================
// Known-failing tests for higher-order-function defunctionalization gaps
// =============================================================================
//
// Both currently fail. Keep them as guards: when the underlying bugs in
// defunctionalization / SPIR-V lowering are fixed, these tests start
// passing and the cleanup will be obvious.

/// Bug 1: A closure with captured free variables, passed to a user-
/// defined HOF, produces SPIR-V that fails `spirv-val` with
/// `OpFunctionCall Function <id>'s parameter count does not match the
/// argument count`. Defunc/mono specializes both the HOF and the
/// closure body correctly (each gets the captured-env params it
/// needs), but the call-sites for the function-typed parameter inside
/// the HOF body still pass only the original arguments — the captured
/// env never gets threaded through.
///
/// Minimal repro: a 2-arg HOF that calls its f-arg twice, with a
/// closure capturing two scalars from the entry's params.
#[test]
fn hof_closure_with_captures_lowers_to_valid_spirv() {
    let src = r#"
def apply2(f: f32 -> f32, x0: f32, x1: f32) f32 = f(x0) + f(x1)

#[compute]
entry test(a: f32, b: f32) f32 =
  let g = |y: f32| y * y + a + b in
  apply2(g, 1.0f32, 2.0f32)
"#;
    let spirv = compile_to_spirv(src).expect("compile");
    assert_spirv_call_arities_match(&spirv);
}

/// Bug 2: An inline lambda with no captures, called from inside a
/// `map` body, panics in SPIR-V lowering with "BUG: Unknown type
/// reached lowering: Ignored". The HOF passed the lambda survives
/// type checking but the elaborated function's signature carries an
/// `Ignored` placeholder where the lambda's argument type should be —
/// presumably defunctionalization or the elaborator is leaving the
/// inner call's type unresolved.
///
/// Minimal repro: a HOF taking `f32 -> f32`, called inside a `map`
/// over `[]f32`, with a closure-free inline lambda.
#[test]
fn hof_no_capture_lambda_in_map_body_lowers_without_panic() {
    let src = r#"
def apply2(f: f32 -> f32, x0: f32, x1: f32) f32 = f(x0) + f(x1)

#[compute]
entry test(in_arr: []f32) []f32 =
  map(|x: f32| apply2(|y: f32| y * y, x, x + 1.0f32), in_arr)
"#;
    // catch_unwind because the bug surfaces as a panic in
    // spirv/mod.rs (not a Result::Err).
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| compile_to_spirv(src)));
    let bytes = result.expect("compilation panicked — see Bug 2 docstring").expect("compile returned Err");
    assert_spirv_call_arities_match(&bytes);
}

// =============================================================================
// Sum-type lowering (Phase C) integration tests
// =============================================================================

/// Build a sum value with one constructor and select on it. Exercises
/// constructor-expression → flattened-tuple lowering and match →
/// tag-checked if-chain lowering, end-to-end through SPIR-V.
#[test]
fn sum_type_lowering_compiles_to_spirv() {
    let src = r#"
def pick(v: #left(f32) | #right(f32)) f32 =
    match v
    case #left(x) -> x + 1.0f32
    case #right(y) -> y * 2.0f32

#[fragment]
entry main() #[target(screen)] vec4f32 =
    let a = pick(#left(0.5f32)) in
    let b = pick(#right(0.25f32)) in
    @[a, b, 0.0f32, 1.0f32]
"#;
    let spirv = compile_to_spirv(src).expect("sum-type program should compile to SPIR-V");
    assert_spirv_call_arities_match(&spirv);
}

/// Multi-payload constructor with mixed arities: `#point(f32, f32)`
/// and a nullary `#origin`. Verifies that the flattened-no-sharing
/// layout zero-fills dead slots in the nullary case.
#[test]
fn sum_type_multi_payload_compiles_to_spirv() {
    let src = r#"
def length_sq(p: #point(f32, f32) | #origin) f32 =
    match p
    case #point(x, y) -> x * x + y * y
    case #origin -> 0.0f32

#[fragment]
entry main() #[target(screen)] vec4f32 =
    let a = length_sq(#point(3.0f32, 4.0f32)) in
    let b = length_sq(#origin) in
    @[a, b, 0.0f32, 1.0f32]
"#;
    let spirv = compile_to_spirv(src).expect("multi-payload sum should compile to SPIR-V");
    assert_spirv_call_arities_match(&spirv);
}

// =============================================================================
// Swizzle-with lowering (Phase C) integration tests
// =============================================================================

/// Plain `=` swizzle update: write `e` into v.yz, leaving v.x intact.
#[test]
fn swizzle_with_plain_assign_compiles_to_spirv() {
    let src = r#"
def update(v: vec3f32, e: vec2f32) vec3f32 = v with .yz = e

#[fragment]
entry main() #[target(screen)] vec4f32 =
    let v = update(@[1.0f32, 2.0f32, 3.0f32], @[20.0f32, 30.0f32]) in
    @[v.x, v.y, v.z, 1.0f32]
"#;
    let spirv = compile_to_spirv(src).expect("plain swizzle-with should compile to SPIR-V");
    assert_spirv_call_arities_match(&spirv);
}

/// Compound `*=` swizzle update: vec2 × mat2 multiply, written into
/// v.yz. Exercises the binary-op path inside transform_vec_with.
#[test]
fn swizzle_with_compound_mul_compiles_to_spirv() {
    let src = r#"
def update(v: vec3f32, m: mat2f32) vec3f32 = v with .yz *= m

#[fragment]
entry main() #[target(screen)] vec4f32 =
    let m: mat2f32 = @[[1.0f32, 0.0f32], [0.0f32, 1.0f32]] in
    let v = update(@[1.0f32, 2.0f32, 3.0f32], m) in
    @[v.x, v.y, v.z, 1.0f32]
"#;
    let spirv = compile_to_spirv(src).expect("compound swizzle-with should compile to SPIR-V");
    assert_spirv_call_arities_match(&spirv);
}

/// Four chained `with .swizzle *= mat2` rotations on a direction vector.
#[test]
fn swizzle_with_chained_rotations_compiles_to_spirv() {
    let src = r#"
def rot(a: f32) mat2f32 =
    let c = f32.cos(a) in
    let s = f32.sin(a) in
    @[[c, s], [0.0f32 - s, c]]

def transform(dir0: vec3f32, mx: f32, my: f32) vec3f32 =
    let d1 = dir0 with .yz *= rot(my) in
    let d2 = d1 with .xz *= rot(mx) in
    let d3 = d2 with .yz *= rot(my) in
    d3 with .xz *= rot(mx)

#[fragment]
entry main() #[target(screen)] vec4f32 =
    let d = transform(@[0.0f32, 0.0f32, 1.0f32], 0.5f32, 0.3f32) in
    @[d.x, d.y, d.z, 1.0f32]
"#;
    let spirv = compile_to_spirv(src).expect("chained swizzle-with rotations should compile");
    assert_spirv_call_arities_match(&spirv);
}

/// Walk every `OpAccessChain` into a `StorageBuffer`-class `OpVariable`
/// and assert the access-chain's result pointer points at the same
/// element type the variable's runtime array carries. spirv-val rejects
/// a mismatch with `OpAccessChain result type ... does not match the
/// type that results from indexing into the base`; the in-process check
/// catches the same shape so tests don't need `spirv-val` on $PATH.
fn assert_spirv_storage_access_chain_pointee_types_match(spirv_words: &[u32]) {
    use std::collections::HashMap;
    use wspirv::binary::parse_words;
    use wspirv::dr::{Loader, Operand};
    use wspirv::spirv::{Op, StorageClass};

    let mut loader = Loader::new();
    parse_words(spirv_words, &mut loader).expect("parse spirv");
    let module = loader.module();

    // First pass: index every type-defining instruction by its result id.
    let mut types: HashMap<u32, &wspirv::dr::Instruction> = HashMap::new();
    for inst in module.types_global_values.iter() {
        if let Some(id) = inst.result_id {
            types.insert(id, inst);
        }
    }

    // Helper: drill `OpTypePointer Class %inner` → `inner`.
    let ptr_pointee = |type_id: u32| -> Option<u32> {
        let inst = types.get(&type_id)?;
        if inst.class.opcode != Op::TypePointer {
            return None;
        }
        // Operands: [0] StorageClass, [1] inner type IdRef.
        match inst.operands.get(1) {
            Some(Operand::IdRef(id)) => Some(*id),
            _ => None,
        }
    };

    // Helper: drill `OpTypeStruct %member0 %member1 ...` → first member.
    // For wyn's storage buffer blocks this is the `OpTypeRuntimeArray`.
    let struct_first_member = |type_id: u32| -> Option<u32> {
        let inst = types.get(&type_id)?;
        if inst.class.opcode != Op::TypeStruct {
            return None;
        }
        match inst.operands.first() {
            Some(Operand::IdRef(id)) => Some(*id),
            _ => None,
        }
    };

    // Helper: drill `OpTypeRuntimeArray %elem` → `elem`. (Also accepts
    // OpTypeArray.)
    let array_elem = |type_id: u32| -> Option<u32> {
        let inst = types.get(&type_id)?;
        if !matches!(inst.class.opcode, Op::TypeRuntimeArray | Op::TypeArray) {
            return None;
        }
        match inst.operands.first() {
            Some(Operand::IdRef(id)) => Some(*id),
            _ => None,
        }
    };

    // Collect each StorageBuffer-class OpVariable's element type.
    let mut storage_var_elem: HashMap<u32, u32> = HashMap::new();
    for inst in module.types_global_values.iter() {
        if inst.class.opcode != Op::Variable {
            continue;
        }
        let class = match inst.operands.first() {
            Some(Operand::StorageClass(c)) => *c,
            _ => continue,
        };
        if class != StorageClass::StorageBuffer {
            continue;
        }
        let var_id = match inst.result_id {
            Some(id) => id,
            None => continue,
        };
        // Variable's result_type is `OpTypePointer StorageBuffer %struct`.
        let var_ptr_ty = match inst.result_type {
            Some(id) => id,
            None => continue,
        };
        let struct_ty = match ptr_pointee(var_ptr_ty) {
            Some(id) => id,
            None => continue,
        };
        let runtime_arr = match struct_first_member(struct_ty) {
            Some(id) => id,
            None => continue,
        };
        let elem = match array_elem(runtime_arr) {
            Some(id) => id,
            None => continue,
        };
        storage_var_elem.insert(var_id, elem);
    }

    // Walk every function body for OpAccessChain into such a variable.
    for func in &module.functions {
        for block in &func.blocks {
            for inst in &block.instructions {
                if inst.class.opcode != Op::AccessChain {
                    continue;
                }
                // Operands: [0] base IdRef, [1..] index IdRefs.
                let base = match inst.operands.first() {
                    Some(Operand::IdRef(id)) => *id,
                    _ => continue,
                };
                let Some(expected_elem) = storage_var_elem.get(&base).copied() else {
                    continue;
                };
                let result_ptr_ty = inst.result_type.expect("OpAccessChain has result type");
                let actual_pointee =
                    ptr_pointee(result_ptr_ty).expect("OpAccessChain result type must be OpTypePointer");
                assert_eq!(
                    actual_pointee, expected_elem,
                    "OpAccessChain into StorageBuffer var %{base}: result pointer pointee \
                     %{actual_pointee} does not match the variable's array element type \
                     %{expected_elem} (chain result_id %{:?})",
                    inst.result_id
                );
            }
        }
    }
}

/// Walk every `OpFunctionCall` in a SPIR-V module and assert each
/// call's argument count matches the called function's declared
/// parameter count. The arity-mismatch class of bug above produces
/// SPIR-V that round-trips through rspirv but fails this invariant.
fn assert_spirv_call_arities_match(spirv_words: &[u32]) {
    use wspirv::binary::parse_words;
    use wspirv::dr::{Loader, Operand};
    use wspirv::spirv::Op;

    let mut loader = Loader::new();
    parse_words(spirv_words, &mut loader).expect("parse spirv");
    let module = loader.module();

    // Map function id → declared parameter count.
    let mut arities: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    for func in &module.functions {
        if let Some(def) = &func.def {
            if let Some(Operand::IdRef(_)) = def.result_id.map(Operand::IdRef) {}
            if let Some(id) = def.result_id {
                arities.insert(id, func.parameters.len());
            }
        }
    }

    // Walk every block's instructions for OpFunctionCall.
    for func in &module.functions {
        for block in &func.blocks {
            for inst in &block.instructions {
                if inst.class.opcode == Op::FunctionCall {
                    // Operands: [0] callee IdRef, [1..] argument IdRefs.
                    let callee = match inst.operands.first() {
                        Some(Operand::IdRef(id)) => *id,
                        _ => continue,
                    };
                    let arg_count = inst.operands.len() - 1;
                    let expected = match arities.get(&callee) {
                        Some(n) => *n,
                        None => continue, // external call (e.g. GlslExt)
                    };
                    assert_eq!(
                        arg_count, expected,
                        "OpFunctionCall to function %{} passes {} args but the function declares {} parameters",
                        callee, arg_count, expected
                    );
                }
            }
        }
    }
}

// =============================================================================
// EGIR-side Map parallelization regression tests
// =============================================================================

/// A compute entry whose body is `map(f, xs)` should emit a kernel that
/// loads `gl_GlobalInvocationID` (lane-indexed access) — not a serial
/// driver loop over `0..N`.
#[test]
fn compute_map_loads_global_invocation_id() {
    use wspirv::binary::parse_words;
    use wspirv::dr::{Loader, Operand};
    use wspirv::spirv::Op;
    let src = r#"
#[compute]
entry sq(xs: []f32) []f32 = map(|x: f32| x * x, xs)
"#;
    let spirv = compile_to_spirv(src).expect("map compute compiles");

    let mut loader = Loader::new();
    parse_words(&spirv, &mut loader).expect("parse spirv");
    let module = loader.module();

    // Find the gl_GlobalInvocationID input variable's id from the
    // EntryPoint interface (3rd-and-later operands of OpEntryPoint).
    let entry = module.entry_points.iter().find(|i| {
        if let Some(Operand::LiteralString(name)) = i.operands.get(2) {
            name == "sq"
        } else {
            false
        }
    });
    let entry = entry.expect("entry sq present");
    let func_id = match entry.operands.get(1) {
        Some(Operand::IdRef(id)) => *id,
        _ => panic!("entry has function id"),
    };

    // gl_GlobalInvocationID is the Input variable decorated with
    // BuiltIn GlobalInvocationId.
    // OpDecorate operand layout: [target_id, Decoration, *literals].
    // For BuiltIn the literal at operand 2 is the BuiltIn kind.
    let gid_var = module
        .annotations
        .iter()
        .find(|inst| {
            inst.class.opcode == Op::Decorate
                && matches!(
                    inst.operands.get(2),
                    Some(Operand::BuiltIn(wspirv::spirv::BuiltIn::GlobalInvocationId))
                )
        })
        .and_then(|inst| match inst.operands.first() {
            Some(Operand::IdRef(id)) => Some(*id),
            _ => None,
        })
        .expect("gl_GlobalInvocationID decoration present");

    // The entry function body must contain an OpLoad whose pointer is
    // the gl_GlobalInvocationID variable.
    let func = module
        .functions
        .iter()
        .find(|f| f.def.as_ref().and_then(|d| d.result_id) == Some(func_id))
        .expect("entry function present");
    let loads_gid = func.blocks.iter().any(|b| {
        b.instructions.iter().any(|inst| {
            inst.class.opcode == Op::Load
                && matches!(inst.operands.first(), Some(Operand::IdRef(id)) if *id == gid_var)
        })
    });
    assert!(
        loads_gid,
        "compute Map entry must OpLoad gl_GlobalInvocationID; got serial loop instead"
    );
}

/// True iff the named compute entry's function body `OpLoad`s
/// `gl_GlobalInvocationID` — i.e. it lowered to a lane-indexed parallel
/// kernel rather than a serial driver loop. Returns false if the entry
/// or the GID builtin isn't present.
fn entry_loads_global_invocation_id(spirv: &[u32], entry_name: &str) -> bool {
    use wspirv::binary::parse_words;
    use wspirv::dr::{Loader, Operand};
    use wspirv::spirv::Op;

    let mut loader = Loader::new();
    parse_words(spirv, &mut loader).expect("parse spirv");
    let module = loader.module();

    let Some(entry) = module
        .entry_points
        .iter()
        .find(|i| matches!(i.operands.get(2), Some(Operand::LiteralString(name)) if name == entry_name))
    else {
        return false;
    };
    let Some(Operand::IdRef(func_id)) = entry.operands.get(1).cloned() else {
        return false;
    };
    let Some(gid_var) = module
        .annotations
        .iter()
        .find(|inst| {
            inst.class.opcode == Op::Decorate
                && matches!(
                    inst.operands.get(2),
                    Some(Operand::BuiltIn(wspirv::spirv::BuiltIn::GlobalInvocationId))
                )
        })
        .and_then(|inst| match inst.operands.first() {
            Some(Operand::IdRef(id)) => Some(*id),
            _ => None,
        })
    else {
        return false;
    };
    let Some(func) =
        module.functions.iter().find(|f| f.def.as_ref().and_then(|d| d.result_id) == Some(func_id))
    else {
        return false;
    };
    func.blocks.iter().any(|b| {
        b.instructions.iter().any(|inst| {
            inst.class.opcode == Op::Load
                && matches!(inst.operands.first(), Some(Operand::IdRef(id)) if *id == gid_var)
        })
    })
}

/// A fixed-size output ahead of a streamed `map` must not force the entry
/// serial — the map still lowers to a GID-indexed kernel. Regression: the
/// planner used to treat only output slot 0 as the parallelizable tail, so
/// `([2]u32, []u32)` (fixed first) lowered serial while `([]u32, [2]u32)`
/// sharded.
#[test]
fn fixed_output_before_streamed_map_still_shards() {
    let spirv = compile_to_spirv(
        "#[compute]\nentry r(a: []u32) ([2]u32, []u32) = ([7u32, 9u32], map(|x| x + 1u32, a))\n",
    )
    .expect("fixed-then-map compute compiles");
    assert!(
        entry_loads_global_invocation_id(&spirv, "r"),
        "a map after a fixed output must still shard (load gl_GlobalInvocationID)"
    );
}

/// A fixed slot's direct local alias is still a fixed producer; its surface
/// `Var` form must not prevent sibling maps from supplying the domain.
#[test]
fn let_bound_literal_fixed_output_with_multidomain_maps_shards() {
    let spirv = compile_to_spirv(
        r#"
#[compute]
entry r(a: []u32, b: []u32) ([2]u32, []u32, []u32) =
  let o0 = [1u32, 2u32] in
  (o0, map(|x| x + 1u32, a), map(|y| y + 2u32, b))
"#,
    )
    .expect("let-bound fixed output + multidomain maps compiles");
    assert!(entry_loads_global_invocation_id(&spirv, "r_dispatch_1"));
    assert!(entry_loads_global_invocation_id(&spirv, "r_dispatch_2"));
}

/// Output-slot analysis must classify a let-bound map by its producer,
/// even when a fixed output occupies the first slot.
#[test]
fn fixed_output_before_let_bound_map_still_shards() {
    let spirv = compile_to_spirv(
        r#"
#[compute]
entry r(a: []u32) ([2]u32, []u32) =
  let m = map(|x| x + 1u32, a) in
  ([1u32, 2u32], m)
"#,
    )
    .expect("fixed output + let-bound map compiles");
    assert!(entry_loads_global_invocation_id(&spirv, "r"));
}

/// Canonical resolved slots must also reach the per-domain stage splitter;
/// otherwise admission succeeds but planning sees the original alias syntax.
#[test]
fn fixed_output_with_let_bound_multidomain_maps_shards() {
    let spirv = compile_to_spirv(
        r#"
#[compute]
entry r(a: []u32, b: []u32) ([2]u32, []u32, []u32) =
  let ma = map(|x| x + 1u32, a) in
  let mb = map(|y| y + 2u32, b) in
  ([1u32, 2u32], ma, mb)
"#,
    )
    .expect("fixed output + let-bound multidomain maps compiles");
    assert!(entry_loads_global_invocation_id(&spirv, "r_dispatch_1"));
    assert!(entry_loads_global_invocation_id(&spirv, "r_dispatch_2"));
}

/// Let-bound maps with nested scalar lets and captured storage inputs must use
/// the same resolved-slot path as syntactically inline maps.
#[test]
fn let_bound_complex_same_domain_maps_shard() {
    let spirv = compile_to_spirv(
        r#"
#[compute]
entry r(tidx: []u32, src: []vec4f32, st: []f32) ([2]f32, []vec4f32, []vec4f32) =
  let g = st[0]
  let o0 = [g, g]
  let m1 = map(|s| let i = i32(s) in let it = src[i % 4] in @[it.x, 0.0, it.y, it.z], tidx)
  let m2 = map(|s| let i = i32(s) in let it = src[i % 4] in @[0.0, 1.0, 0.0, it.w], tidx) in
  (o0, m1, m2)
"#,
    )
    .expect("let-bound complex same-domain maps compile");
    assert!(entry_loads_global_invocation_id(&spirv, "r"));
}

/// A fixed-size output alongside several *different-domain* maps: the fixed
/// slot becomes its own 1×1×1 constant-write stage while each map keeps its
/// own GID-indexed per-domain dispatch. Regression: any non-map slot used to
/// drag the whole multidomain entry onto the serial fallback.
#[test]
fn fixed_output_with_multidomain_maps_shards() {
    let spirv = compile_to_spirv(
        "#[compute]\nentry r(a: []u32, b: []u32) ([2]u32, []u32, []u32) = \
         ([7u32, 9u32], map(|x| x + 1u32, a), map(|y| y + 2u32, b))\n",
    )
    .expect("fixed + multidomain maps compiles");
    assert!(
        entry_loads_global_invocation_id(&spirv, "r_dispatch_1"),
        "first map dispatch must shard"
    );
    assert!(
        entry_loads_global_invocation_id(&spirv, "r_dispatch_2"),
        "second map dispatch must shard"
    );
}

/// A fixed output derived through a pure prefix let must carry that lexical
/// dependency into its 1x1x1 stage without serializing sibling map domains.
#[test]
fn fixed_output_from_storage_scalar_with_multidomain_maps_shards() {
    let spirv = compile_to_spirv(
        r#"
#[compute]
entry r(a: []u32, b: []u32, st: []u32) ([2]u32, []u32, []u32) =
  let g = st[0] in
  ([g, g + 1u32], map(|x| x + 1u32, a), map(|y| y + 2u32, b))
"#,
    )
    .expect("captured fixed output + multidomain maps compiles");
    assert!(entry_loads_global_invocation_id(&spirv, "r_dispatch_1"));
    assert!(entry_loads_global_invocation_id(&spirv, "r_dispatch_2"));
}

/// A pure prefix value may be reproduced by more than one synthesized stage:
/// here the fixed output and the first map both depend on `g`.
#[test]
fn fixed_output_and_map_share_storage_scalar_and_multidomain_maps_shard() {
    let spirv = compile_to_spirv(
        r#"
#[compute]
entry r(a: []u32, b: []u32, st: []u32) ([2]u32, []u32, []u32) =
  let g = st[0] in
  ([g, g + 1u32], map(|x| x + g, a), map(|y| y + 2u32, b))
"#,
    )
    .expect("shared captured scalar + multidomain maps compiles");
    assert!(entry_loads_global_invocation_id(&spirv, "r_dispatch_1"));
    assert!(entry_loads_global_invocation_id(&spirv, "r_dispatch_2"));
}

/// A compute entry whose body is `map(f, xs)` should not contain an
/// `OpLoopMerge` — the parallel kernel is a single guarded scalar
/// branch. Inner function loops (e.g. raymarch) are not affected.
#[test]
fn compute_map_has_no_full_serial_loop() {
    use wspirv::binary::parse_words;
    use wspirv::dr::{Loader, Operand};
    use wspirv::spirv::Op;
    let src = r#"
#[compute]
entry sq(xs: []f32) []f32 = map(|x: f32| x * x, xs)
"#;
    let spirv = compile_to_spirv(src).expect("map compute compiles");

    let mut loader = Loader::new();
    parse_words(&spirv, &mut loader).expect("parse spirv");
    let module = loader.module();

    let entry = module
        .entry_points
        .iter()
        .find(|i| matches!(i.operands.get(2), Some(Operand::LiteralString(n)) if n == "sq"));
    let func_id = match entry.and_then(|i| i.operands.get(1)) {
        Some(Operand::IdRef(id)) => *id,
        _ => panic!("entry sq not found"),
    };
    let func = module
        .functions
        .iter()
        .find(|f| f.def.as_ref().and_then(|d| d.result_id) == Some(func_id))
        .expect("entry function present");
    let has_loop_merge =
        func.blocks.iter().any(|b| b.instructions.iter().any(|inst| inst.class.opcode == Op::LoopMerge));
    assert!(
        !has_loop_merge,
        "compute Map entry must NOT contain OpLoopMerge — the parallel kernel is loop-free"
    );
}

fn assert_compute_entry_reads_thread_id(src: &str, entry_name: &str) {
    use crate::builtins::catalog;
    use crate::op::OpTag;
    use crate::ssa::types::InstKind;

    let program = compile_to_ssa(src);
    let thread_id_builtin = catalog().known().thread_id;
    let entry = program
        .entry_points
        .iter()
        .find(|entry| entry.name == entry_name)
        .unwrap_or_else(|| panic!("entry {entry_name} present"));
    let reads_thread_id = entry.body.inner.blocks.iter().any(|(_, block)| {
        block.insts.iter().any(|&inst_id| {
            matches!(
                &entry.body.get_inst(inst_id).data,
                InstKind::Op {
                    tag: OpTag::Intrinsic { id, .. },
                    ..
                } if *id == thread_id_builtin
            )
        })
    });
    assert!(
        reads_thread_id,
        "entry {entry_name} should use thread_id for pointwise parallelization"
    );
}

fn assert_compute_entry_has_no_ssa_loops(src: &str, entry_name: &str) {
    use crate::ssa::types::ControlHeader;

    let program = compile_to_ssa(src);
    let entry = program
        .entry_points
        .iter()
        .find(|entry| entry.name == entry_name)
        .unwrap_or_else(|| panic!("entry {entry_name} present"));
    assert!(
        entry.body.control_headers.values().all(|header| !matches!(header, ControlHeader::Loop { .. })),
        "entry {entry_name} should be a loop-free guarded lane kernel"
    );
}

#[test]
fn compute_pointwise_screma_from_horizontal_maps_is_parallel() {
    use crate::builtins::catalog;
    use crate::op::OpTag;
    use crate::ssa::types::{ControlHeader, InstKind};
    let src = r#"
#[compute]
entry pair(xs: []f32) ([]f32, []f32) =
  let a = map(|x: f32| x * x, xs) in
  let b = map(|x: f32| x + 1.0, xs) in
  (a, b)
"#;
    let fused = compile_to_fused_tlc(src);
    let pair = fused
        .defs
        .iter()
        .find(|def| fused.symbols.get(def.name).map(|s| s.as_str()) == Some("pair"))
        .expect("pair not found");
    let (_, body) = extract_lambda_params(&pair.body);
    assert!(
        has_soac_kind(&body, "Screma"),
        "sibling maps should fuse to pointwise Screma before parallelization"
    );

    let program = compile_to_ssa(src);
    let thread_id_builtin = catalog().known().thread_id;
    let pair = program.entry_points.iter().find(|entry| entry.name == "pair").expect("entry pair present");
    let loads_thread_id = pair.body.inner.blocks.iter().any(|(_, block)| {
        block.insts.iter().any(|&inst_id| {
            matches!(
                &pair.body.get_inst(inst_id).data,
                InstKind::Op {
                    tag: OpTag::Intrinsic { id, .. },
                    ..
                } if *id == thread_id_builtin
            )
        })
    });
    assert!(loads_thread_id, "pointwise Screma entry must read thread_id");
    assert!(
        pair.body.control_headers.values().all(|header| !matches!(header, ControlHeader::Loop { .. })),
        "pointwise Screma entry must be the loop-free guarded lane kernel"
    );

    let spirv = compile_to_spirv(src).expect("pointwise Screma compute compiles");
    assert!(!spirv.is_empty(), "pointwise Screma should lower to SPIR-V");
}

/// Compile `source` through the full *parallelized* pipeline (matching the
/// production driver, which always parallelizes compute) and return the
/// lowered SPIR-V + pipeline descriptor.
fn compile_parallel(source: &str) -> crate::Lowered {
    crate::compile_thru_spirv(source).expect("compile_thru_spirv")
}

/// Full storage-buffer descriptors of a compute pipeline.
fn compute_storage_buffers(
    pipeline: &crate::pipeline_descriptor::PipelineDescriptor,
    entry: &str,
) -> Vec<crate::pipeline_descriptor::Binding> {
    use crate::pipeline_descriptor::{Binding, Pipeline};
    pipeline
        .pipelines
        .iter()
        .find_map(|p| match p {
            Pipeline::Compute(cp) if cp.stages.iter().any(|s| s.entry_point == entry) => Some(cp),
            _ => None,
        })
        .unwrap_or_else(|| panic!("no compute pipeline named {entry}"))
        .bindings
        .iter()
        .filter(|b| matches!(b, Binding::StorageBuffer { .. }))
        .cloned()
        .collect()
}

/// Gathering a *computed* array (a `map` result) at a runtime index used to
/// panic in SPIR-V emission ("Composite variant unsized arrays not
/// supported"). `lift_gathers` splits the producer `map` into its own
/// `gen_gather_0` compute stage writing a storage buffer, and rewrites the
/// consumer's `counts[i]` into a load from that buffer. This pins the
/// end-to-end wiring: both stages agree on the gather buffer's binding, it's a
/// compiler-managed Intermediate (not host I/O), it doesn't collide with the
/// consumer's own input/output, and it carries a `LikeInput` sizing policy so
/// the host allocates it from `bh`'s length (a `map` preserves element count;
/// `[]vec4f32` → `[]i32` is 4 of 16 bytes per element).
#[test]
fn gather_computed_array_materializes_to_shared_intermediate() {
    use crate::pipeline_descriptor::{Access, Binding, BufferLen, BufferUsage};
    let src = "\
#[compute]
entry gen(bh: []vec4f32) []i32 =
  let counts = map(|h:vec4f32| 4 + 5*(if h.x>4.0 then 3 else 1), bh) in
  map(|i:i32| counts[i % 256], iota(6144))
";
    let lowered = compile_parallel(src);
    assert!(!lowered.spirv.is_empty(), "lowering produced no SPIR-V");

    let gather_entry = lowered
        .pipeline
        .pipelines
        .iter()
        .find_map(|p| match p {
            crate::pipeline_descriptor::Pipeline::Compute(cp) => {
                cp.stages.iter().find(|s| s.entry_point.contains("_gather_")).map(|s| s.entry_point.clone())
            }
            _ => None,
        })
        .expect("a gather pre-pass compute stage must exist");

    let gather_bufs = compute_storage_buffers(&lowered.pipeline, &gather_entry);
    let consumer_bufs = compute_storage_buffers(&lowered.pipeline, "gen");

    // The pre-pass writes exactly one Intermediate (the gather buffer), sized
    // LikeInput of `bh` (binding 0): one i32 (4B) per vec4f32 (16B) element.
    let producer_intermediates: Vec<&Binding> = gather_bufs
        .iter()
        .filter(|b| {
            matches!(
                b,
                Binding::StorageBuffer {
                    usage: BufferUsage::Intermediate,
                    ..
                }
            )
        })
        .collect();
    assert_eq!(
        producer_intermediates.len(),
        1,
        "pre-pass writes one gather intermediate: {gather_bufs:?}"
    );
    let Binding::StorageBuffer {
        binding: gather_binding,
        access,
        length,
        ..
    } = producer_intermediates[0]
    else {
        unreachable!()
    };
    // Producer and consumer are phases of one scheduled pipeline and share
    // one binding table. The intermediate therefore carries their combined
    // access; stage-level `reads`/`writes` below retain the precise direction.
    assert_eq!(
        *access,
        Access::ReadWrite,
        "gather phases share a read/write binding"
    );
    assert_eq!(
        length.as_ref(),
        Some(&BufferLen::LikeInput {
            set: 0,
            binding: 0,
            elem_bytes: 4,
            src_elem_bytes: 16,
        }),
        "gather buffer must be sized from its input array's element count"
    );

    let shared_pipeline = lowered
        .pipeline
        .pipelines
        .iter()
        .find_map(|pipeline| match pipeline {
            crate::pipeline_descriptor::Pipeline::Compute(compute)
                if compute.stages.iter().any(|stage| stage.entry_point == gather_entry) =>
            {
                Some(compute)
            }
            _ => None,
        })
        .expect("gather pipeline");
    let gather_index = shared_pipeline
        .bindings
        .iter()
        .position(|binding| {
            matches!(binding, Binding::StorageBuffer { binding, .. } if binding == gather_binding)
        })
        .expect("gather binding index");
    assert!(shared_pipeline
        .stages
        .iter()
        .find(|stage| stage.entry_point == gather_entry)
        .expect("gather producer stage")
        .writes
        .contains(&gather_index));
    assert!(shared_pipeline
        .stages
        .iter()
        .find(|stage| stage.entry_point == "gen")
        .expect("gather consumer stage")
        .reads
        .contains(&gather_index));

    // The consumer reads that same binding as an Intermediate. Access is
    // `ReadWrite` because of the module-level promotion described above —
    // not because the consumer itself writes (it doesn't).
    let reads_gather = consumer_bufs.iter().any(|b| {
        matches!(b, Binding::StorageBuffer { binding, usage: BufferUsage::Intermediate, access: Access::ReadWrite, .. } if binding == gather_binding)
    });
    assert!(
        reads_gather,
        "consumer must read the gather buffer (binding {gather_binding}) as ReadWrite intermediate: {consumer_bufs:?}"
    );

    // The consumer's own output goes to a different binding — no collision.
    let consumer_outputs: Vec<u32> = consumer_bufs
        .iter()
        .filter_map(|b| match b {
            Binding::StorageBuffer {
                binding,
                usage: BufferUsage::Output,
                ..
            } => Some(*binding),
            _ => None,
        })
        .collect();
    assert_eq!(
        consumer_outputs.len(),
        1,
        "consumer writes one output: {consumer_bufs:?}"
    );
    assert_ne!(
        consumer_outputs[0], *gather_binding,
        "consumer output must not collide with the gather buffer"
    );
}

// ---------------------------------------------------------------------------
// Multi-consumer gather regression
// ---------------------------------------------------------------------------
//
// `lift_gathers` handles a computed array `counts = map(...)` consumed by a
// single downstream SOAC/gather. Whenever `counts` has *two or more* downstream
// consumers (e.g. a `reduce` plus a `scan`, or a `scan` plus a direct
// `counts[i % N]` gather in the same lambda), the lift currently leaves the
// in-register Composite array in place and the SPIR-V backend panics at
// `spirv/mod.rs:374` ("Composite variant unsized arrays not supported"). The
// single-consumer controls below pin the working baseline; the
// `#[ignore]`-marked tests capture the multi-consumer bug — run with
// `cargo test -- --ignored` to verify the panic and remove the `#[ignore]`
// once the lift threads the same intermediate buffer to every consumer.

/// Control: a single `scan` consumer of a computed `counts` map lifts cleanly.
#[test]
fn single_consumer_scan_compiles() {
    compile_to_spirv(
        "\
#[compute]
entry gen(xs: []i32) []i32 =
  let counts  = map(|x: i32| x * 2, xs) in
  let offsets = scan(|a: i32, b: i32| a + b, 0, counts) in
  map(|i: i32| offsets[i % 8], iota(64))
",
    )
    .expect("single-consumer scan-over-map must lift + compile");
}

/// A scalar bound in an outer SOAC lambda and captured by a *nested* SOAC
/// lambda must be threaded as a capture, not left as a free global. This is
/// the N-body inner-force-sum shape — `map(|i| … reduce(+, …, map(|j| f(xs[j],
/// xs[i]), …)))` — where the inner `map` over `j` closes over `pi = xs[i]`.
#[test]
fn nested_soac_captures_outer_scalar() {
    compile_to_spirv(
        "\
#[compute]
entry t(xs: []f32) []f32 =
  map(|i: i32|
        let pi = xs[i] in
        reduce(|a: f32, b: f32| a + b, 0.0,
               map(|j: i32| xs[j] - pi, 0i32 ..< 4)),
      0i32 ..< 4)
",
    )
    .expect("nested SOAC capturing an outer scalar must compile");
}

// ---- `filter` SOAC composition (gaps surfaced exploring the N-body port) ----
//
// `filter(pred, xs)` needs a statically-sized `xs` and returns the existential
// `?k. [k]T`, which a `let` opens before use. These pin which compositions of
// that result the compiler accepts. (None of these can be *executed* in unit
// tests — there's no GPU adapter here — so they assert compilation only.)

/// The supported shape: fixed-size input, existential result opened by `let`,
/// consumed by `length`. Compiles end to end.
#[test]
fn filter_in_subroutine_length_compiles() {
    compile_to_spirv(
        "\
def evens(arr: [8]i32) ?k. [k]i32 = filter(|x: i32| x % 2i32 == 0i32, arr)

#[compute]
entry filt_count() i32 =
  let e = evens([1i32, 2i32, 3i32, 4i32, 5i32, 6i32, 7i32, 8i32]) in
  length(e)
",
    )
    .expect("filter in a subroutine, opened by `let`, consumed by `length` must compile");
}

/// Runtime-sized `filter` consumed by `length`: the input is an entry-param
/// view (`[]i32`), so `filter` compacts kept elements into a reserved scratch
/// storage buffer and yields a runtime-length view; `length` reads the view's
/// `len` operand (the surviving count).
#[test]
fn filter_runtime_length_compiles() {
    compile_to_spirv(
        "\
#[compute]
entry filt_count(xs: []i32) i32 =
  let e = filter(|x: i32| x % 2i32 == 0i32, xs) in
  length(e)
",
    )
    .expect("runtime-sized filter consumed by length must compile");
}

/// A runtime-sized `filter` inside a **subroutine** that the entry calls. This
/// is the safety net for the scratch-binding home: `filter` compacts into a
/// reserved storage buffer, and only a compute *entry* owns a descriptor set +
/// binding namespace to host it (an `EgirFunc` does not — see the guard in
/// `from_tlc::convert_function`). This compiles because `evens` is **inlined**
/// into `filt_count` before EGIR conversion, so `convert_soac_filter` runs in
/// the entry's converter and the scratch buffer lands at a non-colliding entry
/// binding.
///
/// IF THIS TEST STARTS FAILING with "runtime `filter` in function `evens`
/// reserved a scratch storage buffer …": the inlining invariant broke — a
/// function whose result is a runtime filter survived to EGIR as a standalone
/// `EgirFunc`. The scratch buffer then has no descriptor-set home. To fix,
/// either (a) restore inlining of filter-returning functions before `from_tlc`,
/// or (b) thread a caller-reserved scratch binding into the function's
/// signature (like an extra param / interface entry) so the buffer is declared
/// and sized on a real descriptor set. Do NOT relax the `convert_function`
/// guard to emit anyway — that mis-numbers the binding and silently drops its
/// host declaration (wrong-buffer codegen).
#[test]
fn filter_runtime_in_subroutine_compiles() {
    compile_to_spirv(
        "\
def evens(arr: []i32) ?k. [k]i32 = filter(|x: i32| x % 2i32 == 0i32, arr)

#[compute]
entry filt_count(xs: []i32) i32 =
  let e = evens(xs) in
  length(e)
",
    )
    .expect("runtime filter in an (inlined) subroutine must compile");
}

/// Summing a filtered runtime-sized array — `reduce(+, 0, filter(p, xs))` over
/// an entry-param view. `filter` yields a runtime-length scratch view; `reduce`
/// consumes it like any reduce-over-view. (Was a gap: the static-literal form
/// left an unexpanded `EgirSoac` panicking at `elaborate.rs`.)
#[test]
fn filter_into_reduce_compiles() {
    compile_to_spirv(
        "\
#[compute]
entry filt_reduce(xs: []i32) i32 =
  let kept = filter(|x: i32| x > 4i32, xs) in
  reduce(|a: i32, b: i32| a + b, 0i32, kept)
",
    )
    .expect("summing a filtered runtime array (filter → reduce) must compile");
}

#[test]
fn filter_runtime_scalar_consumers_fuse_to_screma_and_compile() {
    let source = "\
#[compute]
entry filt_stats(xs: []i32) (i32, i32) =
  let kept = filter(|x: i32| x > 4i32, xs) in
  (length(kept), reduce(|a: i32, b: i32| a + b, 0i32, kept))
";

    let tlc = compile_to_fused_tlc(source);
    let entry = tlc
        .defs
        .iter()
        .find(|def| tlc.symbols.get(def.name).map(|s| s.as_str()) == Some("filt_stats"))
        .expect("filt_stats not found");
    let (_, body) = extract_lambda_params(&entry.body);
    assert!(
        has_soac_kind(&body, "Screma"),
        "filter feeding scalar consumers should fuse to a Screma"
    );
    assert!(
        !has_soac_kind(&body, "Filter"),
        "scalar filter consumers should not materialize the filtered array"
    );

    compile_to_spirv(source).expect("runtime filter feeding length+reduce scalar outputs must compile");
}

/// Companion working form for the aspiration below: let-bind first, then
/// pass to a helper that consumes a plain `[]i32`. The `let` opens
/// `filter`'s existential into a skolem-sized `[k]i32`, and
/// `rep_specialize` handles the `Abstract`-variant abstract array
/// crossing the call boundary into `total`. Pins that this shape stays
/// working — the inline-existential fix should not regress it.
#[test]
fn filter_into_reduce_let_bound_crosses_call_boundary() {
    compile_to_spirv(
        "\
def total(ys: []i32) i32 = reduce(|a: i32, b: i32| a + b, 0i32, ys)
#[compute]
entry filt_reduce(xs: []i32) i32 =
  let kept = filter(|x: i32| x > 4i32, xs) in
  total(kept)
",
    )
    .expect("let-bound filter result crossing a call boundary into a helper that takes a plain array must compile");
}

/// Regression: `reduce(_, _, filter(...))` with the filter result used inline
/// as an argument (no `let` to bind it first) compiles to the same program
/// as the let-bound form above. Used to fail with
///
///   Function argument type mismatch at argument 3:
///   expected Array[i32, ?, ?, ?],
///   got ?k. [Array[i32, abstract, k, no_region]]
///
/// because existential elimination only fired at `let` binders, not at
/// general use sites. Fixed in `unify_apply_arg` by mirroring the let-
/// binder's `open_existential` call at each function-argument unification
/// site, gated on "expected param is not itself existential" so existential-
/// typed values can still flow through unchanged when that's the param's
/// declared type. Surfaced minimizing the type error in
/// `testfiles/playground/particles3.wyn`'s `align`.
#[test]
fn filter_into_reduce_inline_arg_opens_existential() {
    compile_to_spirv(
        "\
#[compute]
entry filt_reduce(xs: []i32) i32 =
  reduce(|a: i32, b: i32| a + b, 0i32, filter(|x: i32| x > 4i32, xs))
",
    )
    .expect("filter result used inline as `reduce`'s array arg unifies like the let-bound form");
}

/// Regression: a multi-letter swizzle on a non-trivial expression
/// must not duplicate the expression. `reduce(...).xy` previously
/// desugared to `@[reduce(...).0, reduce(...).1]` (clone-per-letter)
/// and downstream saw two independent `Soac(Reduce)` producers, so
/// the compiled output ran the reduce twice. Surfaced in
/// `particles3.wyn`: `center`, `align`, and `separate` each took a
/// `.xy` (or `.zw`) of an aggregate reduce, and SPIR-V emitted
/// duplicated reduce loops + duplicate `Length`/`Normalize` ops.
/// Fixed by let-binding the projection base before splitting it into
/// per-letter `TupleProj`s when the base isn't already a `Var` /
/// literal.
#[test]
fn swizzle_on_nontrivial_base_does_not_duplicate_producer() {
    use crate::tlc::{SoacOp, Term, TermKind};
    fn count_reduces(t: &Term) -> usize {
        let mut n = 0;
        if matches!(&t.kind, TermKind::Soac(SoacOp::Reduce { .. })) {
            n += 1;
        }
        t.for_each_child(&mut |c| n += count_reduces(c));
        n
    }
    // Each `def` returns a swizzle of a reduce result. With the fix
    // there's one physical reduce per def (let-bound, then projected);
    // without the fix each `.xy` would emit two independent reduces.
    let tlc = crate::compile_thru_tlc(
        "\
def sum2<[n]>(xs: [n]vec4f32) vec2f32 =
  reduce(|a: vec4f32, b: vec4f32| a + b, @[0.0f32, 0.0f32, 0.0f32, 0.0f32], xs).xy
#[compute]
entry e(xs: [8]vec4f32) vec2f32 = sum2(xs)
",
    )
    .expect("compile_thru_tlc");
    let total: usize = tlc.tlc.defs.iter().map(|d| count_reduces(&d.body)).sum();
    assert_eq!(
        total, 1,
        "`reduce(...).xy` should compile to one physical reduce, not one per swizzle slot — \
         found {total} `Soac(Reduce)` terms across all defs"
    );
}

/// True iff the pipeline for `entry` is a multi-stage compute (the two-phase
/// shape a parallelized scalar reduction lowers to: chunk + combine). Used to
/// confirm the masked fused-reduce fusion fired — a *serial* filter→reduce would be
/// a single-stage `Compute` instead.
fn is_two_phase_compute(pipeline: &crate::pipeline_descriptor::PipelineDescriptor, entry: &str) -> bool {
    use crate::pipeline_descriptor::Pipeline;
    pipeline.pipelines.iter().any(|p| match p {
        Pipeline::Compute(mc) => mc.stages.len() >= 2 && mc.stages.iter().any(|s| s.entry_point == entry),
        _ => false,
    })
}

/// `reduce(op, ne, filter(p, xs))` fuses into a masked single-accumulator Screma — no compacted
/// intermediate array — and parallelizes as a two-phase reduce. Pins that the
/// fusion fired (not the serial scratch-view filter path).
#[test]
fn filter_into_reduce_fuses_to_parallel_screma() {
    let lowered = crate::compile_thru_spirv(
        "\
#[compute]
entry filt_reduce(xs: []i32) i32 =
  let kept = filter(|x: i32| x > 4i32, xs) in
  reduce(|a: i32, b: i32| a + b, 0i32, kept)
",
    )
    .expect("filter→reduce compiles");
    assert!(
        is_two_phase_compute(&lowered.pipeline, "filt_reduce"),
        "reduce(filter(..)) must fuse to filtered Screma (two-phase compute), not a serial filter",
    );
}

/// The masked fused-reduce fusion must fire even when `filter` and `reduce` live in
/// **different functions** — TLC fusion sees through the `evens` call via
/// function summaries (no inlining needed at fusion time).
#[test]
fn filter_into_reduce_fuses_across_functions() {
    let lowered = crate::compile_thru_spirv(
        "\
def evens(xs: []i32) ?k. [k]i32 = filter(|x: i32| x % 2i32 == 0i32, xs)

#[compute]
entry filt_reduce(xs: []i32) i32 =
  let kept = evens(xs) in
  reduce(|a: i32, b: i32| a + b, 0i32, kept)
",
    )
    .expect("cross-function filter→reduce compiles");
    assert!(
        is_two_phase_compute(&lowered.pipeline, "filt_reduce"),
        "cross-function reduce(evens(xs)) must fuse to filtered Screma via function summaries",
    );
}

/// Every compute entry point generated for a program, across all pipelines and
/// their stages (the source entries plus any lifted `_gather_` pre-passes).
/// Lets a test assert how many GPU dispatches one source entry expands to.
fn compute_entry_points(pipeline: &crate::pipeline_descriptor::PipelineDescriptor) -> Vec<String> {
    use crate::pipeline_descriptor::Pipeline;
    pipeline
        .pipelines
        .iter()
        .flat_map(|p| match p {
            Pipeline::Compute(cp) => cp.stages.iter().map(|s| s.entry_point.clone()).collect::<Vec<_>>(),
            Pipeline::Graphics(_) => Vec::new(),
        })
        .collect()
}

/// A `map` feeding a `filter` in one entry should compact in a single coherent
/// pipeline — exactly as `filter` alone does, where the gather is an internal
/// stage of *one* pipeline. Today it instead splits into TWO compute pipelines
/// (`pick` plus a `pick_gather_0` pre-pass) whose intermediate buffers don't
/// even share a name — `pick` reads `pick_gather_b1` while `pick_gather_0`
/// writes `pick_gather_0_gather_b1` — and with nothing in the descriptor
/// recording that the gather must run first. A host runtime can neither wire the
/// gather's output into the filter's input nor order the two dispatches.
#[test]
fn map_into_filter_is_one_wired_pipeline() {
    let lowered = compile_parallel(
        "\
open f32
#[compute]
entry pick(xs: []u32) ?k. [k]u32 =
  let ys = map(|x| x + 1u32, xs) in
  filter(|y| y < 100u32, ys)
",
    );
    assert_eq!(
        lowered.pipeline.pipelines.len(),
        1,
        "map→filter should compact in one pipeline (filter alone does); got entry points {:?}",
        compute_entry_points(&lowered.pipeline),
    );
}

/// A capturing producer map folded into a filter: `map(|x| x + bound, xs)`
/// captures the runtime value `bound`, so the fused filter's `map_lam` carries a
/// capture. That capture must survive closure conversion's free-variable
/// analysis, ownership/liveness (it is read inside the fused map), and the filter
/// lowering (where it becomes an extra operand of the per-element map call,
/// carried by the map body's explicit capture list. Still one coherent pipeline.
#[test]
fn capturing_map_into_filter_is_one_pipeline() {
    let lowered = compile_parallel(
        "\
open f32
#[compute]
entry pick(xs: []u32) ?k. [k]u32 =
  let bound = xs[0] in
  let ys = map(|x| x + bound, xs) in
  filter(|y| y < 100u32, ys)
",
    );
    assert_eq!(
        lowered.pipeline.pipelines.len(),
        1,
        "capturing map→filter should still compact to one pipeline; got {:?}",
        compute_entry_points(&lowered.pipeline),
    );
}

/// Inlining the `map` directly into `filter` (instead of let-binding it first)
/// compiles to the same thing as the let-bound form, and the map *changes the
/// element type* (`u32` → `vec4f32`), exercising the filter lowering's distinct
/// input vs. output element types: the buffer/view are sized in the output type
/// while elements are read in the input type.
#[test]
fn inlined_filter_over_map_compiles() {
    compile_parallel(
        "\
open f32
#[compute]
entry cmptest(idx: []u32) ?k. [k]vec4f32 =
  filter(|c| c.x < 100.0, map(|s| @[f32(i32(s)), 0.0, 0.0, 0.0], idx))
",
    );
}

/// An entry returning *both* a filtered array and a value derived from its
/// `length`, with the existential over the WHOLE tuple: `?k. ([k]u32, [1]u32)`.
/// The `?k.` packs over the tuple — `normalize_outputs` must see through the
/// existential wrapper and count the tuple's two outputs, not treat the whole
/// `?k.(…)` as a single output. (Form A.)
#[test]
fn filter_array_and_length_existential_over_tuple_compiles() {
    compile_parallel(
        "\
open f32
#[compute]
entry both(xs: []u32) ?k. ([k]u32, [1]u32) =
  let v = filter(|x| x < 100u32, xs) in
  let n = length(v) in
  (v, [u32(n)])
",
    );
}

/// Same body as Form A, but the existential is on just the first tuple
/// component: `(?k. [k]u32, [1]u32)`. The per-component existential must unify
/// with the filter result's skolem-pinned size, so the entry still type-checks
/// and lowers. (Form B.)
#[test]
fn filter_array_and_length_per_component_existential_compiles() {
    compile_parallel(
        "\
open f32
#[compute]
entry both(xs: []u32) (?k. [k]u32, [1]u32) =
  let v = filter(|x| x < 100u32, xs) in
  let n = length(v) in
  (v, [u32(n)])
",
    );
}

/// Two filters with *independent* runtime lengths, returned as a tuple under
/// stacked existentials — `?k. ?j. ([k]u32, [j]u32)`. Exercises both halves of
/// the existential-over-tuple handling: the output-count see-through peels the
/// stacked `?k. ?j.` wrappers to the tuple (two output slots), and the return
/// check packs each quantifier to its *own* fresh witness — so the two distinct
/// skolem lengths are kept distinct, never conflated into one.
#[test]
fn two_filters_distinct_existential_lengths_compile() {
    compile_parallel(
        "\
open f32
#[compute]
entry both2(xs: []u32) ?k. ?j. ([k]u32, [j]u32) =
  let v = filter(|x| x < 100u32, xs) in
  let w = filter(|x| x > 5u32, xs) in
  (v, w)
",
    );
}

/// A `map → filter → map → reduce` chain (the `separation`-style shape: a
/// producer map feeds a filter, whose result feeds another map then a reduce)
/// must collapse to a single masked `Screma`. The trailing map fuses into the
/// reduce (a reducing `Screma`); the filter then folds into that Screma's step
/// — preserving its pure combiner — and the leading map folds in too, leaving
/// no materialized intermediate array. Before reducing-`Screma`s exposed
/// `Reduction` semantics this stalled at three separate loops.
#[test]
fn map_filter_map_reduce_collapses_to_one_screma() {
    let tlc = compile_to_fused_tlc(
        "\
#[compute]
entry e(#[storage(set=2, binding=0, access=read)] xs: []f32,
        #[storage(set=2, binding=1, access=write)] out: *[]f32) []f32 =
  let p = xs[0..512] in
  map(|x: f32|
        let ys = map(|v: f32| v - x, p) in
        let zs = filter(|y: f32| y < 10.0, ys) in
        let ws = map(|z: f32| z * 2.0, zs) in
        reduce(|a: f32, b: f32| a + b, 0.0, ws),
      p)
",
    );
    use crate::tlc::SoacOp;
    fn count_soac(term: &crate::tlc::Term, pred: &dyn Fn(&SoacOp) -> bool) -> usize {
        let mut n = if let TermKind::Soac(s) = &term.kind { usize::from(pred(s)) } else { 0 };
        term.for_each_child(&mut |c| n += count_soac(c, pred));
        n
    }
    // Count across the whole reachable program, not just entry `e`'s body:
    // defunctionalize (now after fusion) lifts the outer map's operator lambda
    // — which carries the fused Screma — into a separate top-level def, so the
    // Screma no longer lives inside `e.body`. A real fusion regression would
    // leave a surviving Filter/Reduce somewhere in the reachable set, so the
    // program-wide filter/reduce counts still catch it.
    let count_all = |pred: &dyn Fn(&SoacOp) -> bool| -> usize {
        tlc.defs.iter().map(|d| count_soac(&d.body, pred)).sum()
    };
    let filters = count_all(&|s| matches!(s, SoacOp::Filter { .. }));
    let reduces = count_all(&|s| matches!(s, SoacOp::Reduce { .. }));
    let scremas = count_all(&|s| matches!(s, SoacOp::Screma { .. }));
    assert_eq!(
        filters, 0,
        "the filter must fold into the reducing Screma, not survive as a Filter SOAC"
    );
    assert_eq!(reduces, 0, "the reduce must fold into the Screma");
    assert_eq!(
        scremas, 1,
        "the fused map→filter→map→reduce should collapse to a single Screma"
    );
}

/// Cross-function auto-parallelization: a `scan` factored into a helper that
/// `inline_small` will NOT fold (its operator has control flow, so the
/// size/control-flow gate skips it) still parallelizes — `materialize_entry_soacs`
/// exposes it at the entry boundary so `parallelize` produces the same
/// multi-phase pipeline as the in-entry form. (`inline_small` skipping the
/// helper is what makes this exercise the new pass specifically.)
#[test]
fn cross_function_scan_parallelizes() {
    let lowered = crate::compile_thru_spirv(
        "\
def stencil(xs: []i32) []i32 = scan(|a: i32, b: i32| if a > b then a else b, 0i32, xs)
#[compute]
entry e(xs: []i32) []i32 = stencil(xs)
",
    )
    .expect("cross-function scan compiles");
    assert!(
        is_two_phase_compute(&lowered.pipeline, "e"),
        "a scan factored into a (non-inlinable) helper must still parallelize cross-function",
    );
}

/// Guard for the runtime-sized-index clean-rejection (above): a *statically
/// sized* composite array must still index fine. The clean-reject keys on
/// runtime (unsized) Composite size, so a `[N]T` local indexed at runtime
/// lowers as before, not rejected.
#[test]
fn sized_composite_array_runtime_index_still_lowers() {
    let source = r#"
#[compute]
entry e(i: i32) i32 =
    let m: [4]i32 = [10, 20, 30, 40] in
    m[i]
"#;
    compile_to_spirv(source).expect("runtime index into a statically-sized array should lower");
}

/// Invariant, end to end: a SOAC helper called *per element* inside a `map`
/// lambda must NOT be hoisted and parallelized — the inner reduce stays a
/// serial per-thread loop. The entry parallelizes as a single-stage
/// lane-indexed map, not a multi-phase reduce pipeline.
#[test]
fn per_element_helper_soac_stays_serial() {
    let lowered = crate::compile_thru_spirv(
        "\
def rsum(x: i32) i32 = reduce(|a: i32, b: i32| a + b, 0i32, [x, x, x])
#[compute]
entry e(xs: []i32) []i32 = map(|x: i32| rsum(x), xs)
",
    )
    .expect("per-element helper compiles");
    assert!(
        !is_two_phase_compute(&lowered.pipeline, "e"),
        "a per-element helper reduce must stay serial, not become a parallel reduce pipeline",
    );
}

/// Returning a filtered runtime-sized array from a compute entry. The filter
/// compacts directly into the user-visible output buffer (sized to the input's
/// element count), and its surviving count is written to a paired `len` cell
/// the host reads back. `realize_outputs::retarget_filter_output` wires this.
#[test]
fn filter_result_as_compute_output_compiles() {
    compile_to_spirv(
        "\
#[compute]
entry filt_out(xs: []i32) ?k. [k]i32 =
  filter(|x: i32| x % 2i32 == 0i32, xs)
",
    )
    .expect("returning a filtered array from a compute entry must compile");
}

/// Pins the filter→output host ABI (the paired length buffer). The `filt_out`
/// pipeline must expose: the input, a host-readable **Output** data buffer sized
/// `LikeInput` of the input (capacity n), and a compiler-managed **Intermediate**
/// length cell sized `Fixed { bytes: 4 }` (one u32) holding the surviving count,
/// plus compiler-managed u32 work buffers for the parallel prefix scan.
#[test]
fn filter_output_descriptor_has_paired_length_buffer() {
    use crate::pipeline_descriptor::{Binding, BufferLen, BufferUsage};
    let src = "\
#[compute]
entry filt_out(xs: []i32) ?k. [k]i32 =
  filter(|x: i32| x % 2i32 == 0i32, xs)
";
    let lowered = crate::compile_thru_spirv(src).expect("filter→output compiles");
    let bufs = compute_storage_buffers(&lowered.pipeline, "filt_out");

    let output = bufs
        .iter()
        .find(|b| {
            matches!(
                b,
                Binding::StorageBuffer {
                    usage: BufferUsage::Output,
                    ..
                }
            )
        })
        .expect("filter→output has a host-readable Output buffer");
    let Binding::StorageBuffer { length: out_len, .. } = output else {
        unreachable!()
    };
    assert!(
        matches!(out_len, Some(BufferLen::LikeInput { .. })),
        "output data buffer is sized to the input element count (capacity n): {output:?}",
    );

    let intermediates: Vec<&Binding> = bufs
        .iter()
        .filter(|b| {
            matches!(
                b,
                Binding::StorageBuffer {
                    usage: BufferUsage::Intermediate,
                    ..
                }
            )
        })
        .collect();
    assert_eq!(
        intermediates.len(),
        5,
        "length cell plus filter flags, offsets, and scan block scratch: {bufs:?}",
    );
    assert!(intermediates.iter().any(|binding| matches!(
        binding,
        Binding::StorageBuffer {
            length: Some(BufferLen::Fixed { bytes: 4 }),
            ..
        }
    )));
    assert_eq!(
        intermediates
            .iter()
            .filter(|binding| matches!(
                binding,
                Binding::StorageBuffer {
                    length: Some(BufferLen::LikeInput { .. }),
                    ..
                }
            ))
            .count(),
        2,
        "flags and offsets are input-sized u32 buffers"
    );
    assert_eq!(
        intermediates
            .iter()
            .filter(|binding| matches!(
                binding,
                Binding::StorageBuffer {
                    length: Some(BufferLen::Fixed { bytes: 1024 }),
                    ..
                }
            ))
            .count(),
        2,
        "scan block sums and block offsets have a fixed length (FILTER_SCAN_GROUPS \
         * REDUCE_PHASE1_WIDTH = 256 u32s = 1024 bytes), bounding the serial phase-2"
    );
}

/// Control: a single `reduce` consumer of a computed `counts` map lifts cleanly.
#[test]
fn single_consumer_reduce_compiles() {
    compile_to_spirv(
        "\
#[compute]
entry gen(xs: []i32) []i32 =
  let counts = map(|x: i32| x * 2, xs) in
  let total  = reduce(|a: i32, b: i32| a + b, 0, counts) in
  map(|i: i32| total, iota(64))
",
    )
    .expect("single-consumer reduce-over-map must lift + compile");
}

/// When `counts` is consumed by **both** a `reduce` and a `scan` (and a
/// downstream gather then reads the scan result), `lift_gathers` materializes
/// `counts` into one shared gather buffer that both downstream SOACs read from.
#[test]
fn multi_consumer_scan_plus_reduce_lifts() {
    compile_to_spirv(
        "\
#[compute]
entry gen(xs: []i32) []i32 =
  let counts  = map(|x: i32| x * 2, xs) in
  let total   = reduce(|a: i32, b: i32| a + b, 0, counts) in
  let offsets = scan(|a: i32, b: i32| a + b, 0, counts) in
  map(|i: i32| offsets[i % 8] + total, iota(64))
",
    )
    .expect("multi-consumer (reduce + scan over the same counts) should lift + compile");
}

/// `counts` consumed by both `scan(counts)` and a direct random gather
/// `counts[i % 8]`. The scan's input is a SOAC edge in the producer graph;
/// the `counts[i % 8]` reference inside the outer map's lambda body is *not*
/// a SOAC edge. With the use-count fix in `producer_graph` counting every
/// `Var(counts)` reference (not just SOAC edges), fusion sees `counts` as
/// multi-use and declines to fuse + drop the let, so `lift_gathers` handles
/// it as a normal multi-consumer let-bound producer.
#[test]
fn multi_consumer_scan_plus_gather_lifts() {
    compile_to_spirv(
        "\
#[compute]
entry gen(xs: []i32) []i32 =
  let counts  = map(|x: i32| x * 2, xs) in
  let offsets = scan(|a: i32, b: i32| a + b, 0, counts) in
  map(|i: i32| offsets[i % 8] + counts[i % 8], iota(64))
",
    )
    .expect("multi-consumer (scan + direct gather of the same counts) should lift + compile");
}

/// Bisected min-repro: a fused scan whose op-lambda calls a user helper
/// function (`box_count`) and whose result is randomly indexed
/// (`offsets[nb - 1]`). Prior to the fix, `lift_gathers::free_symbol_vars`
/// passed empty `known_defs` to `collect_free_vars`, so the top-level def
/// reference to `box_count` appeared as a "free var" — failing the
/// entry-param predicate and declining the lift. The scan then stayed in
/// `gen`'s body where `parallelize::make_scan_plan` fell back to the serial
/// pipeline, and EGIR's `is_handleable_soac` rejected the resulting
/// `Scan { destination: Fresh, input: View }` combination → unexpanded
/// `EgirSoac` panic at `egir/elaborate.rs:314`. With the fix, top-level
/// defs are filtered out of the predicate, the scan lifts into a gather
/// pre-pass, and the rest of the pipeline handles it normally.
#[test]
fn fused_scan_helper_call_then_indexed_read_compiles() {
    compile_to_spirv(
        "\
def win_count(hw: f32) i32 =
  let span = 2.0 * hw - 1.0 in
  let fit  = i32.f32(floor(span / 2.4)) in
  if fit < 0 then 0 else if fit > 3 then 3 else fit

def box_count(hw: f32) i32 = 8 + 5 * win_count(hw)

#[compute]
entry gen(bh: []vec4f32, #[uniform(set=1,binding=0)] nb: i32) [1]i32 =
  let counts  = map(|h: vec4f32| box_count(h.x), bh) in
  let offsets = scan(|a: i32, b: i32| a + b, 0, counts) in
  [if nb <= 0 then 0 else offsets[nb - 1]]
",
    )
    .expect("fused-scan-of-helper-mapping with indexed scan read should compile");
}

/// Bisected min-repro: a multi-output entry returns a scan result AS one
/// output and also reads that scan by a (constant) index for a second
/// output. Previously panicked at `spirv/mod.rs:375` ("Composite variant
/// unsized arrays not supported") because `realize_outputs` retargeted the
/// scan to `OutputView` for slot 0 but slot 1's `[offsets[0]]` still
/// demanded the (now-vanished) in-register Composite. Fixed in
/// `realize_outputs::rewrite_other_index_consumers_to_loads`: detect the
/// sibling Index consumer, synthesize a `ViewIndex + Load` against slot
/// 0's output view (both backends declare output bindings as read-write),
/// alias the Index NodeId to the load result. Slot 0's binding doubles as
/// the shared buffer.
#[test]
fn multi_output_returns_scan_and_reads_it_by_index() {
    compile_to_spirv(
        "\
#[compute]
entry gen(xs: []i32) ([]i32, [1]i32) =
  let offsets = scan(|a: i32, b: i32| a + b, 0, xs) in
  (offsets, [offsets[0]])
",
    )
    .expect("multi-output (scan + indexed read of same scan) should compile");
}

/// Dynamic-index variant of the above — slot 1 reads `offsets[k]` where
/// `k` is a uniform, exercising the path where the rewrite passes the
/// dynamic index NodeId straight through `emit_view_load`.
#[test]
fn multi_output_returns_scan_and_reads_it_by_dynamic_index() {
    compile_to_spirv(
        "\
#[compute]
entry gen(xs: []i32, #[uniform(set=1,binding=0)] k: i32) ([]i32, [1]i32) =
  let offsets = scan(|a: i32, b: i32| a + b, 0, xs) in
  (offsets, [offsets[k]])
",
    )
    .expect("multi-output with dynamic-index sibling read of same scan should compile");
}

/// Map producer variant — slot 0 retargets a Map (not a Scan); slot 1
/// reads it by index. Same mechanism, different SOAC kind.
#[test]
fn multi_output_returns_map_and_reads_it_by_index() {
    compile_to_spirv(
        "\
#[compute]
entry gen(xs: []i32) ([]i32, [1]i32) =
  let doubled = map(|x: i32| x * 2, xs) in
  (doubled, [doubled[0]])
",
    )
    .expect("multi-output (map + indexed read of same map) should compile");
}

/// Returning the same scan in two output slots — `(offsets, offsets)` —
/// can't be served by retargeting the scan twice. Slot 0 retargets to
/// its view; slot 1, also a runtime-sized array, can't retarget the
/// (already-retargeted) SOAC, so it falls through to the existing
/// "runtime-sized array not produced by a retargetable map/scan"
/// `ConvertError::Unsupported`. Pinned here to confirm we surface a
/// clean diagnostic rather than panicking.
#[test]
fn multi_output_returns_scan_in_two_slots_is_rejected() {
    let result = crate::compile_thru_spirv(
        "\
#[compute]
entry gen(xs: []i32) ([]i32, []i32) =
  let offsets = scan(|a: i32, b: i32| a + b, 0, xs) in
  (offsets, offsets)
",
    );
    let err = match result {
        Ok(_) => panic!("expected returning the same scan in two output slots to fail"),
        Err(e) => e,
    };
    let msg = format!("{err}");
    assert!(
        msg.contains("runtime-sized array"),
        "expected runtime-sized-array diagnostic, got: {msg}"
    );
}

/// Under `--single-stage` mode, a vec4-emitting map that gathers from a
/// derived (map/scan-produced) array must still produce well-formed
/// SPIR-V. The bug was: `lift_gathers` flagged the producer's gather
/// buffer via an Output-role `StorageBindingDecl` on the prepass entry,
/// but `from_tlc::convert` only consulted the parallelize `plans` for
/// `forced_output_binding`. With `parallelize_soacs(disable=true)`
/// `plans` is empty, so `build_entry_outputs` auto-allocated the
/// prepass's output at the next free binding — colliding with the
/// consumer's vec4 output and emitting an `int` store into a `vec4`
/// buffer. spirv-val flagged the OpAccessChain type mismatch.
#[test]
fn single_stage_vec4_map_gather_from_derived_array_repro() {
    let spirv = compile_to_spirv_single_stage(
        "\
#[compute]
entry gen(xs: []i32) []vec4f32 =
  let cs = map(|x: i32| x * 2, xs) in
  map(|i: i32| @[f32.i32(cs[i]), 0.0, 0.0, 1.0], iota(8))
",
    )
    .expect("single-stage vec4-map gathering derived array compiles");
    assert_spirv_storage_access_chain_pointee_types_match(&spirv);
}

/// The descriptor for a compiler-allocated `lift_gathers` intermediate
/// must agree with the SPIR-V module's per-binding writability. SPIR-V's
/// `NonWritable` decoration is module-level, so when a sibling entry in
/// the same module writes the gather buffer, the consumer pipeline's
/// `OpVariable` has no `NonWritable` and Naga reports the storage as
/// `read_write`. Previously the descriptor reported `read_only` based on
/// the consumer's `StorageBindingDecl.role` alone, causing wgpu to fail
/// pipeline creation with `Storage class Storage{LOAD} doesn't match
/// shader Storage{LOAD | STORE}`. The fix promotes any intermediate
/// binding whose `(set, binding)` is also an entry-output target to
/// `Access::ReadWrite`.
#[test]
fn intermediate_buffer_descriptor_access_repro() {
    use crate::pipeline_descriptor::{Access, Binding, BufferUsage, Pipeline};
    use std::collections::HashSet;

    let lowered = crate::compile_thru_spirv(
        "\
#[compute]
entry gen(xs: []i32) ([]i32, [1]i32) =
  let counts  = map(|x: i32| x * 2, xs) in
  let offsets = scan(|a: i32, b: i32| a + b, 0, counts) in
  (map(|i: i32| offsets[i % 8], iota(64)),
   [offsets[7]])
",
    )
    .expect("compile to SPIR-V");

    // Collect every (set, binding) any pipeline writes to anywhere in
    // the module. A read-only declaration of one of these in a sibling
    // pipeline is a descriptor↔shader mismatch.
    let written: HashSet<(u32, u32)> = lowered
        .pipeline
        .pipelines
        .iter()
        .flat_map(|p| match p {
            Pipeline::Compute(cp) => cp.bindings.iter().collect::<Vec<_>>(),
            Pipeline::Graphics(gp) => gp.bindings.iter().collect::<Vec<_>>(),
        })
        .filter_map(|b| match b {
            Binding::StorageBuffer {
                set, binding, access, ..
            } if matches!(access, Access::WriteOnly | Access::ReadWrite) => Some((*set, *binding)),
            _ => None,
        })
        .collect();

    let mut violations: Vec<String> = Vec::new();
    for p in &lowered.pipeline.pipelines {
        let (label, bindings): (String, &[Binding]) = match p {
            Pipeline::Compute(cp) => {
                let names: Vec<&str> = cp.stages.iter().map(|s| s.entry_point.as_str()).collect();
                (format!("compute[{}]", names.join(",")), &cp.bindings)
            }
            Pipeline::Graphics(gp) => {
                let names: Vec<&str> = gp.stages.iter().map(|s| s.entry_point.as_str()).collect();
                (format!("graphics[{}]", names.join(",")), &gp.bindings)
            }
        };
        for b in bindings {
            if let Binding::StorageBuffer {
                set,
                binding,
                access: Access::ReadOnly,
                usage: BufferUsage::Intermediate,
                name,
                ..
            } = b
            {
                if written.contains(&(*set, *binding)) {
                    violations.push(format!(
                        "pipeline {label}: intermediate binding {set}.{binding} ({name}) \
                         declared read_only but written by another entry in the module"
                    ));
                }
            }
        }
    }
    assert!(
        violations.is_empty(),
        "descriptor↔shader access mismatch on lift_gathers intermediates:\n  {}",
        violations.join("\n  ")
    );
}

/// A `scan` producer gathers the same way a `map` does: it's lifted into its
/// own pre-pass (here a multi-stage parallel scan) writing the gather buffer,
/// which the consumer reads via `storage_index`. The forced-output binding is
/// honored uniformly across SOAC kinds, so the scan's final output lands on
/// the buffer the consumer reads, and the scan's own intermediates (block
/// sums/offsets) sit above it without collision.
#[test]
fn gather_scan_producer_materializes_to_shared_intermediate() {
    use crate::pipeline_descriptor::{Access, Binding, BufferLen, BufferUsage, Pipeline};
    let src = "\
#[compute]
entry g(xs: []i32) []i32 =
  let o = scan(|a:i32,b:i32| a+b, 0, xs) in
  map(|i:i32| o[i % 256], iota(6144))
";
    let lowered = compile_parallel(src);
    assert!(!lowered.spirv.is_empty(), "lowering produced no SPIR-V");

    // The consumer reads the gather buffer as an Intermediate sized
    // LikeInput of `xs` (scan preserves element count and type: i32 → i32).
    // Access is `ReadWrite`: `publish` promotes any binding written by a
    // sibling entry in the module so the descriptor matches the SPIR-V's
    // module-level access.
    let consumer_bufs = compute_storage_buffers(&lowered.pipeline, "g");
    let gather = consumer_bufs
        .iter()
        .find_map(|b| match b {
            Binding::StorageBuffer {
                binding,
                usage: BufferUsage::Intermediate,
                access: Access::ReadWrite,
                length: Some(len),
                ..
            } => Some((*binding, len.clone())),
            _ => None,
        })
        .expect("consumer must read a sized gather intermediate");
    assert_eq!(
        gather.1,
        BufferLen::LikeInput {
            set: 0,
            binding: 0,
            elem_bytes: 4,
            src_elem_bytes: 4,
        }
    );

    // The gather pre-pass is a multi-stage parallel scan that writes that same
    // binding as its result, with its block-sum/offset intermediates above it.
    let scan_pipeline = lowered
        .pipeline
        .pipelines
        .iter()
        .find_map(|p| match p {
            Pipeline::Compute(mc) if mc.stages.iter().any(|s| s.entry_point.contains("_gather_")) => {
                Some(mc)
            }
            _ => None,
        })
        .expect("scan gather pre-pass must be a multi_compute pipeline");
    let writes_gather = scan_pipeline
        .bindings
        .iter()
        .any(|b| matches!(b, Binding::StorageBuffer { binding, .. } if *binding == gather.0));
    assert!(
        writes_gather,
        "scan pre-pass must write the gather buffer (binding {})",
        gather.0
    );
    // No other binding in the scan pipeline collides with the gather output.
    let dup = scan_pipeline
        .bindings
        .iter()
        .filter(|b| matches!(b, Binding::StorageBuffer { binding, .. } if *binding == gather.0))
        .count();
    assert_eq!(
        dup, 1,
        "exactly one scan binding is the gather buffer: {:?}",
        scan_pipeline.bindings
    );
}

/// A helper's size variables are generalized, so one caller equating two of
/// its params' lengths does not equate them for every other caller.
///
/// `go2` passes `xc` for both of `slice_b`'s params, forcing `size(a) ==
/// size(b)` *at that call*. `render` — dead, never called — applies both
/// helpers to the same pair, which would chain `slice_b`'s size equation onto
/// `slice_ab` and from there onto `go`. If sizes were monomorphic per
/// declaration, `go`'s two outputs would both size `LikeInput` on binding 0
/// and its two maps would fuse into one dispatch.
#[test]
fn helper_size_equation_does_not_leak_across_call_sites() {
    use crate::pipeline_descriptor::{Binding, BufferLen, BufferUsage};
    let src = "\
open f32

def slice_ab(a: []u32, b: []u32) ([]f32, []f32) =
  (map(|s| f32(s), a),
   map(|s| f32(s) * 2.0, b))

def slice_b(a: []u32, b: []u32) []f32 =
  map(|s| f32(s) * 3.0 + f32(a[0]), b)

#[compute]
entry go(xa: []u32, xb: []u32) ([]f32, []f32) =
  slice_ab(xa, xb)

#[compute]
entry go2(xc: []u32) []f32 =
  slice_b(xc, xc)

def render(a: []u32, b: []u32) ([]f32, []f32, []f32) =
  let (p, q) = slice_ab(a, b)
  let r = slice_b(a, b) in
  (p, q, r)
";
    let lowered = compile_parallel(src);
    let outputs: Vec<(u32, BufferLen)> = compute_storage_buffers(&lowered.pipeline, "go")
        .iter()
        .filter_map(|b| match b {
            Binding::StorageBuffer {
                binding,
                usage: BufferUsage::Output,
                length: Some(len),
                ..
            } => Some((*binding, len.clone())),
            _ => None,
        })
        .collect();
    assert_eq!(
        outputs.len(),
        2,
        "`go` returns two runtime-sized arrays: {outputs:?}"
    );

    // Output 0 is a map over `xa` (binding 0); output 1 over `xb` (binding 1).
    let src_binding = |len: &BufferLen| match len {
        BufferLen::LikeInput { binding, .. } => Some(*binding),
        _ => None,
    };
    assert_eq!(
        src_binding(&outputs[0].1),
        Some(0),
        "first output sizes like `xa`"
    );
    assert_eq!(
        src_binding(&outputs[1].1),
        Some(1),
        "second output sizes like `xb`"
    );
}

/// A record whose fields are storage-buffer views, bound outside a map and
/// indexed inside that map's body, keeps a concrete `Region(set, binding)` on
/// each field. Passing the record to a helper that maps over its fields and
/// returns a record of the results is what drives those field types view-ward.
///
/// Without this, `sc.pts` reaches `lower_index` as
/// `Array[f32, View, ?size, ?region]` and SPIR-V lowering has no buffer to
/// build an `OpAccessChain` into.
#[test]
fn record_of_views_indexed_in_map_body_compiles() {
    let src = "\
open f32

type world = { pts: []f32, its: []f32 }

def update(w: world, pdom: []u32, idom: []u32) world =
  { pts = map(|s| w.pts[i32(s)] * 2.0, pdom),
    its = map(|s| w.its[i32(s)] + 1.0, idom) }

#[compute]
entry go(dom: []u32, pdom: []u32, idom: []u32, pts_in: []f32, its_in: []f32)
  ([]f32, []f32, []f32) =
  let w  = { pts = pts_in, its = its_in }
  let w2 = update(w, pdom, idom)
  let sc = { pts = w.pts, its = w.its } in
  (w2.pts, w2.its,
   map(|s| sc.pts[0] + sc.its[0] + f32(s), dom))
";
    crate::compile_thru_spirv(src).expect("record of views indexed inside a map must compile");
}

/// The equal-domain fuser rewrites sibling output maps into one `Screma`. That
/// `Screma` reads whatever the lanes captured, so it must sit *below* the `let`s
/// that bind those captures — not above the whole store chain.
///
/// A let-bound scalar is enough to expose it: hoisted above `let k`, the fused
/// body's reference to `k` resolves to `PureOp::Global("k")`, and SPIR-V
/// lowering reports `Unknown global: k`.
#[test]
fn fused_maps_are_placed_below_the_bindings_they_capture() {
    let src = "\
open f32
#[compute]
entry go(dom: []u32, pts_in: []f32) ([]f32, []f32) =
  let k = pts_in[0] in
  (map(|s| pts_in[i32(s)] * 2.0, dom),
   map(|s| k + f32(s), dom))
";
    let lowered = compile_parallel(src);

    // Still one fused stage — the fix moves the Screma, it does not disable fusion.
    let stages: Vec<&str> = lowered
        .pipeline
        .pipelines
        .iter()
        .find_map(|p| match p {
            crate::pipeline_descriptor::Pipeline::Compute(cp) => Some(cp),
            _ => None,
        })
        .expect("a compute pipeline")
        .stages
        .iter()
        .map(|s| s.entry_point.as_str())
        .collect();
    assert_eq!(stages, vec!["go"], "both maps fuse into one stage");
}

/// Two maps over the same domain fuse into one stage. A record of views bound
/// outside them must be carried into the fused stage; otherwise the field
/// projection in the second map's body resolves to a global the fused stage
/// never declared.
#[test]
fn record_of_views_survives_stage_fusion() {
    let src = "\
open f32
type painted = { pts: []f32 }
def sdf(sc: painted, x: f32) f32 = sc.pts[0] + x

#[compute]
entry go(dom: []u32, pts_in: []f32) ([]f32, []f32) =
  let sc = { pts = pts_in } in
  (map(|s| pts_in[i32(s)] * 2.0, dom),
   map(|s| sdf(sc, f32(s)), dom))
";
    crate::compile_thru_spirv(src).expect("record of views must survive map fusion into one stage");
}

/// Multiple gathers of the *same* computed array coalesce to one buffer: the
/// lift keys on the let-bound symbol and rewrites every `arr[..]` use to the
/// same storage binding, so a single pre-pass materializes `arr` once no
/// matter how many times (or in how many consumer maps) it's indexed.
#[test]
fn gather_same_array_coalesces_to_one_buffer() {
    use crate::pipeline_descriptor::{Binding, BufferUsage, Pipeline};
    let src = "\
#[compute]
entry gen(bh: []i32) []i32 =
  let arr = map(|x:i32| x + 1, bh) in
  map(|i:i32| arr[i % 256] + arr[(i + 1) % 256], iota(6144))
";
    let lowered = compile_parallel(src);
    assert!(!lowered.spirv.is_empty());

    // Exactly one gather pre-pass, despite two `arr[..]` uses.
    let gather_prepasses = lowered
        .pipeline
        .pipelines
        .iter()
        .filter(|p| {
            matches!(p, Pipeline::Compute(cp) if cp.stages.iter().any(|s| s.entry_point.contains("_gather_")))
        })
        .count();
    assert_eq!(
        gather_prepasses, 1,
        "two uses of one computed array must share one gather pre-pass"
    );

    // The consumer references exactly one gather intermediate.
    let consumer_intermediates = compute_storage_buffers(&lowered.pipeline, "gen")
        .into_iter()
        .filter(|b| {
            matches!(
                b,
                Binding::StorageBuffer {
                    usage: BufferUsage::Intermediate,
                    ..
                }
            )
        })
        .count();
    assert_eq!(
        consumer_intermediates, 1,
        "both gathers must read the same intermediate buffer"
    );
}

// =============================================================================
// Stage 0: multi-dimensional fixed-size composite locals
// =============================================================================
//
// `[N][M]T` composite literal declared as a local in a compute entry and
// indexed both with constants (→ OpCompositeExtract in SPIR-V) and with
// runtime values (→ OpAccessChain). Tests both the type-system + lowering
// path (already implemented across `parser → checker → tlc → egir → ssa →
// spirv/wgsl`) and the const-fold path that produces a single nested
// `OpConstantComposite`.

#[test]
fn multidim_composite_local_const_and_runtime_index() {
    use crate::ssa::types::InstKind;

    let src = r#"
        #[compute]
        entry pick_const() i32 =
            let m: [3][2]i32 = [[1, 2], [3, 4], [5, 6]] in
            m[1][0]

        #[compute]
        entry pick_runtime(i: i32, j: i32) i32 =
            let m: [3][2]i32 = [[1, 2], [3, 4], [5, 6]] in
            m[i][j]
    "#;

    let program = compile_to_ssa(src);

    // Both entries survive — neither got DCE'd or fused away.
    let entry_names: Vec<&str> = program.entry_points.iter().map(|e| e.name.as_str()).collect();
    assert!(
        entry_names.contains(&"pick_const") && entry_names.contains(&"pick_runtime"),
        "expected both entries in SSA; got {entry_names:?}",
    );

    // The const-index entry should produce zero `Op::Index` insts in its
    // body — folding reduces `m[1][0]` to the literal scalar `3`. The
    // runtime-index entry MUST still carry `Op::Index` operations, since
    // `i` and `j` are entry parameters and can't be folded away.
    let body_dyn_extracts = |ep_name: &str| -> usize {
        let ep = program.entry_points.iter().find(|e| e.name == ep_name).unwrap();
        ep.body
            .inner
            .insts
            .iter()
            .filter(|(_, inst)| {
                matches!(
                    &inst.data,
                    InstKind::Op {
                        tag: crate::op::OpTag::DynamicExtract,
                        ..
                    }
                )
            })
            .count()
    };
    assert_eq!(
        body_dyn_extracts("pick_const"),
        0,
        "const-index `m[1][0]` should fold; saw DynamicExtract ops remaining"
    );
    assert_eq!(
        body_dyn_extracts("pick_runtime"),
        2,
        "runtime-index `m[i][j]` should emit two chained DynamicExtract ops"
    );

    // End-to-end smoke: the SPIR-V backend accepts the program.
    let _ = crate::compile_thru_spirv(src).expect("compile_thru_spirv should succeed");
}

// Stage 2: runtime-outer / fixed-inner storage view. Pins the descriptor
// shape — the per-element byte count must reflect the full inner sub-array
// size (`[4]u32` → 16 B), not the innermost scalar (4 B), because the
// dispatch length is `byte_size / elem_bytes` and the buffer holds one
// `[4]u32` per dispatched thread.
#[test]
fn multidim_view_inner_fixed_carries_subarray_elem_bytes() {
    use crate::pipeline_descriptor::{BufferLen, DispatchLen, DispatchSize, Pipeline};
    let src = r#"
        #[compute]
        entry row_sums(buf: []([4]u32)) []u32 =
            map(|row: [4]u32| row[0] + row[1] + row[2] + row[3], buf)
    "#;
    let lowered = crate::compile_thru_spirv(src).expect("compile_thru_spirv");
    let Pipeline::Compute(cp) = lowered.pipeline.pipelines.first().expect("one pipeline") else {
        panic!("expected single-compute pipeline");
    };
    let stage = cp.stages.first().expect("one stage");
    match &stage.dispatch_size {
        DispatchSize::DerivedFrom { len, .. } => match len {
            DispatchLen::InputBinding {
                set,
                binding,
                elem_bytes,
            } => {
                assert_eq!(*set, 0);
                assert_eq!(*binding, 0);
                assert_eq!(
                    *elem_bytes, 16,
                    "buf: []([4]u32) — each iterated element is [4]u32 (16 bytes), not 4"
                );
            }
            other => panic!("expected InputBinding dispatch length, got {other:?}"),
        },
        other => panic!("expected DerivedFrom dispatch size, got {other:?}"),
    }
    // The output `[]u32`'s size variable matches `buf`'s (the
    // type checker unified them — `map(f, buf): [n]u32` for the same
    // `n` as `buf: [n][4]u32`), so the length-policy inference
    // emits `LikeInput` rather than the looser `SameAsDispatch`.
    // Both resolve to the same allocated byte size when the dispatch
    // is itself derived from `buf` (as it is here), but `LikeInput`
    // names the source binding explicitly.
    let output_len = cp.bindings.iter().find_map(|b| match b {
        crate::pipeline_descriptor::Binding::StorageBuffer { name, length, .. }
            if name == "row_sums_output" =>
        {
            length.clone()
        }
        _ => None,
    });
    match output_len {
        Some(BufferLen::LikeInput {
            set,
            binding,
            elem_bytes,
            src_elem_bytes,
        }) => {
            assert_eq!(set, 0);
            assert_eq!(binding, 0);
            assert_eq!(elem_bytes, 4);
            assert_eq!(src_elem_bytes, 16);
        }
        other => panic!(
            "output should be LikeInput {{set:0, binding:0, elem_bytes:4, src_elem_bytes:16}}, got {other:?}"
        ),
    }
}

/// `If`-over-two-retargetable-maps with a runtime-sized output:
/// previously rejected by `realize_outputs::lower_slot` because the
/// merge-block param wasn't a Map/Scan node. After the DPS migration,
/// each branch's `OutputSlotStore` records its own `SlotSource` at
/// its block; `realize_outputs` retargets both Maps into the same
/// output view. Runtime CFG ensures only one fires per execution
/// path.
#[test]
fn compute_if_over_two_maps_compiles_runtime_sized() {
    use crate::pipeline_descriptor::{BufferLen, Pipeline};
    let src = r#"
        #[compute]
        entry tick(#[storage(set=2, binding=0, access=read)] prev: []vec2f32,
                   #[uniform(set=1, binding=1)] iTime: f32) []vec2f32 =
          if iTime == 0.0
            then map(|p: vec2f32| @[1.0f32, 1.0f32], prev)
            else map(|p: vec2f32| @[p.x + 1.0f32, p.y + 1.0f32], prev)
    "#;
    let lowered = crate::compile_thru_spirv(src).expect("compile_thru_spirv");
    let Pipeline::Compute(cp) = lowered.pipeline.pipelines.first().expect("one pipeline") else {
        panic!("expected single-compute pipeline");
    };
    // Output's size variable matches `prev`'s — the length-inference
    // rule emits `LikeInput` rather than `SameAsDispatch`.
    let output_len = cp.bindings.iter().find_map(|b| match b {
        crate::pipeline_descriptor::Binding::StorageBuffer { name, length, .. }
            if name == "tick_output" =>
        {
            length.clone()
        }
        _ => None,
    });
    match output_len {
        Some(BufferLen::LikeInput {
            set,
            binding,
            elem_bytes,
            ..
        }) => {
            assert_eq!(set, 2);
            assert_eq!(binding, 0);
            assert_eq!(elem_bytes, 8); // vec2f32
        }
        other => panic!("output should be LikeInput, got {other:?}"),
    }
}

#[test]
fn compute_if_over_two_maps_becomes_parallel_pointwise_map() {
    use crate::tlc::{SoacOp, TermKind};

    let src = r#"
        #[compute]
        entry tick(#[storage(set=2, binding=0, access=read)] prev: []vec2f32,
                   #[uniform(set=1, binding=1)] iTime: f32) []vec2f32 =
          if iTime == 0.0
            then map(|p: vec2f32| @[1.0f32, 1.0f32], prev)
            else map(|p: vec2f32| @[p.x + 1.0f32, p.y + 1.0f32], prev)
    "#;

    // Inspect at the pre-defunctionalize stage: `if_over_producer` normalizes
    // here, and later passes obscure it (`normalize_outputs` wraps the result
    // in an OutputSlotStore, `defunctionalize` lifts the Map operator to a ref).
    let fused = crate::test_pipeline::compile_thru_expose_producers(src);
    let tick = fused
        .defs
        .iter()
        .find(|def| fused.symbols.get(def.name).map(|s| s.as_str()) == Some("tick"))
        .expect("tick not found");
    let (_, body) = extract_lambda_params(&tick.body);
    let mut tail = &body;
    while let TermKind::Let { body, .. } = &tail.kind {
        tail = body;
    }
    let TermKind::Soac(SoacOp::Map { lam, .. }) = &tail.kind else {
        panic!("if-over-maps should normalize to one Map, got {:?}", tail.kind);
    };
    assert!(
        matches!(&lam.lam.body.kind, TermKind::If { .. }),
        "the fused Map lambda should contain the original condition"
    );

    assert_compute_entry_reads_thread_id(src, "tick");
    assert_compute_entry_has_no_ssa_loops(src, "tick");
}

#[test]
fn compute_if_over_range_and_let_wrapped_slice_map_parallelizes() {
    let src = r#"
        def N: i32 = 8
        #[compute]
        entry tick(#[storage(set=2, binding=0, access=read)] prev_pos: []vec4f32,
                   #[uniform(set=1, binding=1)] iTime: f32) []vec4f32 =
          if iTime < 0.1 then
            map(|i:i32| @[f32.i32(i), 0.0, 0.0, 0.0], 0i32..<N)
          else
            let prev_pos = prev_pos[0..N] in
            map(
              |upd:vec4f32| @[upd.x + 1.0, upd.y, upd.z, upd.w],
              map(|elem:vec4f32| @[elem.x, elem.y, elem.z, elem.w], prev_pos))
    "#;

    // Pre-defunctionalize: see `if_over_producer`'s normalized Map before
    // `normalize_outputs`/`defunctionalize` obscure it.
    let fused = crate::test_pipeline::compile_thru_expose_producers(src);
    let tick = fused
        .defs
        .iter()
        .find(|def| fused.symbols.get(def.name).map(|s| s.as_str()) == Some("tick"))
        .expect("tick not found");
    let (_, body) = extract_lambda_params(&tick.body);
    assert!(
        has_soac_kind(&body, "Map"),
        "let-wrapped branch maps over equal N domains should normalize to a pointwise Map"
    );

    assert_compute_entry_reads_thread_id(src, "tick");
}

#[test]
fn compute_if_over_different_runtime_sources_stays_branching() {
    use crate::tlc::TermKind;

    let src = r#"
        #[compute]
        entry pick(xs: []f32, ys: []f32, flag: bool) []f32 =
          if flag
            then map(|x: f32| x + 1.0, xs)
            else map(|y: f32| y * 2.0, ys)
    "#;

    // Pre-defunctionalize: maps over distinct domains must stay a branching
    // `If` here (`if_over_producer` only merges branches over one domain).
    let fused = crate::test_pipeline::compile_thru_expose_producers(src);
    let pick = fused
        .defs
        .iter()
        .find(|def| fused.symbols.get(def.name).map(|s| s.as_str()) == Some("pick"))
        .expect("pick not found");
    let (_, body) = extract_lambda_params(&pick.body);
    assert!(
        matches!(&body.kind, TermKind::If { .. }),
        "maps over unrelated runtime-sized inputs must not be collapsed into one output-length choice"
    );
}

/// Nested `If` over retargetable maps. The `convert_slot_store`
/// recursion handles arbitrary nesting; each leaf records its own
/// `SlotSource` against the branch's block. After realization the
/// slot has three sources (one per leaf), all retargeting into the
/// same `OutputView`.
#[test]
fn compute_nested_if_over_three_maps_compiles_runtime_sized() {
    let src = r#"
        #[compute]
        entry tick(#[storage(set=2, binding=0, access=read)] prev: []vec2f32,
                   #[uniform(set=1, binding=1)] iTime: f32) []vec2f32 =
          if iTime < 0.0
            then map(|p: vec2f32| @[0.0f32, 0.0f32], prev)
            else if iTime == 0.0
              then map(|p: vec2f32| @[1.0f32, 1.0f32], prev)
              else map(|p: vec2f32| @[p.x + 1.0f32, p.y + 1.0f32], prev)
    "#;
    crate::compile_thru_spirv(src).expect("nested If over three maps must compile");
}

/// `Let`-wrapped `If` whose body's branches read the let-bound value.
/// `convert_slot_store` recognises the `Let` and binds the RHS at the
/// current block before recursing into the body — so the binding
/// survives the branch fork.
#[test]
fn compute_let_wrapped_if_over_two_maps_compiles_runtime_sized() {
    let src = r#"
        #[compute]
        entry tick<[n]>(#[storage(set=2, binding=0, access=read)] prev: [n]vec2f32,
                        #[uniform(set=1, binding=1)] iTime: f32) [n]vec2f32 =
          let nudge: f32 = iTime * 0.1f32 in
          if iTime == 0.0
            then map(|p: vec2f32| @[nudge, nudge], prev)
            else map(|p: vec2f32| @[p.x + nudge, p.y + nudge], prev)
    "#;
    crate::compile_thru_spirv(src).expect("Let-wrapped If over two maps must compile");
}

/// The user's original case 1 (fixed-size output): both branches map
/// over different sources (`0..<N` vs `prev_pos`) but the output is
/// `[Size(2)]vec2f32` because `N = 2` is a literal. Lands in the
/// fixed-aggregate path of `dispatch::compute_slot_source`. This
/// always worked; the test pins it as regression protection alongside
/// the runtime-sized variants the DPS migration unblocked.
#[test]
fn compute_if_over_two_maps_compiles_fixed_size_different_sources() {
    let src = r#"
        def N: i32 = 2
        #[compute]
        entry tick(#[storage(set=2, binding=0, access=read)] prev_pos: []vec2f32,
                   #[uniform(set=1, binding=1)] iTime: f32) []vec2f32 =
          if iTime == 0.0 then
            map(|i:i32| if i == 0 then @[2.0, 2.0] else @[15.0, 5.0], 0i32..<N)
          else
            map(|pos:vec2f32| @[pos.x + 1.0, pos.y + 1.0], prev_pos)
    "#;
    crate::compile_thru_spirv(src).expect("fixed-size If-over-maps (different sources) must compile");
}

/// The user's original case 2 (fixed-size output): both branches map
/// over the *same* range source. Output is still `[Size(2)]vec2f32`.
/// Same code path as case 1 — the source-difference doesn't matter for
/// fixed-size aggregates.
#[test]
fn compute_if_over_two_maps_compiles_fixed_size_same_source() {
    let src = r#"
        def N: i32 = 2
        #[compute]
        entry tick(#[storage(set=2, binding=0, access=read)] prev_pos: []vec2f32,
                   #[uniform(set=1, binding=1)] iTime: f32) []vec2f32 =
          if iTime == 0.0 then
            map(|i:i32| if i == 0 then @[2.0, 2.0] else @[15.0, 5.0], 0i32..<N)
          else
            map(|i:i32| @[f32.i32(i), f32.i32(i)], 0i32..<N)
    "#;
    crate::compile_thru_spirv(src).expect("fixed-size If-over-maps (same source) must compile");
}

/// Multi-output entry whose Tuple components each contain an `If`.
/// `normalize_outputs` decomposes the Tuple into per-slot
/// `OutputSlotStore`s; each then enters the `If`-fork recursion in
/// `convert_slot_store`. Both slots end up multi-source, each
/// retargeting into its own `OutputView`.
#[test]
fn compute_multi_output_tuple_of_ifs_compiles() {
    let src = r#"
        #[compute]
        entry tick<[n]>(#[storage(set=2, binding=0, access=read)] prev_pos: [n]vec2f32,
                        #[uniform(set=1, binding=1)] iTime: f32) ([n]vec2f32, [n]f32) =
          (if iTime == 0.0
             then map(|p: vec2f32| @[0.0f32, 0.0f32], prev_pos)
             else map(|p: vec2f32| @[p.x + 1.0f32, p.y + 1.0f32], prev_pos),
           if iTime == 0.0
             then map(|p: vec2f32| 0.0f32, prev_pos)
             else map(|p: vec2f32| p.x * p.x + p.y * p.y, prev_pos))
    "#;
    crate::compile_thru_spirv(src).expect("multi-output tuple of Ifs must compile");
}

/// Assert the StorageBuffer variable decorated `(set, binding)` is the base
/// of at least one `OpAccessChain` — i.e. actually read/written, not merely
/// declared. Regression guard for view-array provenance loss, where a lifted
/// lambda's reads went to the (wrong) output descriptor and the input buffer
/// was declared but never accessed.
fn assert_storage_descriptor_is_accessed(spirv_words: &[u32], set: u32, binding: u32) {
    use std::collections::HashMap;
    use wspirv::binary::parse_words;
    use wspirv::dr::{Loader, Operand};
    use wspirv::spirv::{Decoration, Op};

    let mut loader = Loader::new();
    parse_words(spirv_words, &mut loader).expect("parse spirv");
    let module = loader.module();

    let mut sets: HashMap<u32, u32> = HashMap::new();
    let mut binds: HashMap<u32, u32> = HashMap::new();
    for inst in &module.annotations {
        if inst.class.opcode != Op::Decorate {
            continue;
        }
        let target = match inst.operands.first() {
            Some(Operand::IdRef(id)) => *id,
            _ => continue,
        };
        match (inst.operands.get(1), inst.operands.get(2)) {
            (Some(Operand::Decoration(Decoration::DescriptorSet)), Some(Operand::LiteralBit32(n))) => {
                sets.insert(target, *n);
            }
            (Some(Operand::Decoration(Decoration::Binding)), Some(Operand::LiteralBit32(n))) => {
                binds.insert(target, *n);
            }
            _ => {}
        }
    }

    let target_var = sets
        .iter()
        .find(|(id, s)| **s == set && binds.get(id) == Some(&binding))
        .map(|(id, _)| *id)
        .unwrap_or_else(|| panic!("no StorageBuffer variable decorated (set={set}, binding={binding})"));

    let accessed = module.functions.iter().any(|f| {
        f.blocks.iter().any(|b| {
            b.instructions.iter().any(|inst| {
                inst.class.opcode == Op::AccessChain
                    && matches!(inst.operands.first(), Some(Operand::IdRef(base)) if *base == target_var)
            })
        })
    });

    assert!(
        accessed,
        "descriptor (set={set}, binding={binding}) is declared but never reached by an \
         OpAccessChain — view-array provenance was lost (reads went to the wrong buffer)"
    );
}

// View-array slice provenance through a SOAC capture: `xs[0..3]` is fine at
// top level but used to fail inside a `map` lambda body with
// "slice_to_composite: no buffer provenance". After buffer-specialize learned
// the slice→composite case, the lifted lambda's reads must come from `xs`'s
// own descriptor (set=2, binding=0), not the compiler-allocated output buffer.
#[test]
fn slice_view_inside_map_lambda_compiles_to_spirv() {
    let src = r#"
        def gather3(arr: [3]f32) f32 = arr[0] + arr[1] + arr[2]

        #[compute]
        entry tick(#[storage(set=2, binding=0, access=read)] xs: []f32) []f32 =
          map(|_:i32| gather3(xs[0..3]), 0i32..<3)
    "#;
    let lowered = crate::compile_thru_spirv(src)
        .expect("view-array slice inside a map lambda must preserve buffer provenance");
    assert_storage_descriptor_is_accessed(&lowered.spirv, 2, 0);
}

// ---- Buffer-provenance guards ------------------------------------------------
//
// These pin the *correct* descriptor each view read resolves to, so a later
// refactor of buffer_specialize (e.g. unifying `rewrite_term` and
// `rewrite_specialized_body`) can't silently mis-route a read to the wrong
// buffer. `cargo test` green alone does NOT prove this: a wrong-buffer read
// still passes spirv-val as long as the (wrong) descriptor is declared — which
// is exactly the historical bug. Each guard asserts via rspirv that the
// expected `(set, binding)` is actually the base of an `OpAccessChain`.

/// Indexing a *captured* view inside a `map` lambda (→ lifted lambda, the
/// `rewrite_specialized_body` Index arm) must read from the captured buffer.
#[test]
fn view_index_in_map_lambda_reads_own_buffer() {
    let src = r#"
        #[compute]
        entry tick(#[storage(set=2, binding=0, access=read)] xs: []f32) []f32 =
          map(|i: i32| xs[i] + xs[0], 0i32..<4)
    "#;
    let lowered = crate::compile_thru_spirv(src).expect("captured-view index compiles");
    assert_storage_descriptor_is_accessed(&lowered.spirv, 2, 0);
}

/// Two captured views at distinct `(set, binding)` must each be read from
/// their own descriptor — catches a unification that swaps or collapses
/// buffer provenance.
#[test]
fn two_view_captures_read_distinct_buffers() {
    let src = r#"
        #[compute]
        entry tick(
          #[storage(set=2, binding=0, access=read)] xs: []f32,
          #[storage(set=2, binding=1, access=read)] ys: []f32
        ) []f32 =
          map(|i: i32| xs[i] + ys[0], 0i32..<4)
    "#;
    let lowered = crate::compile_thru_spirv(src).expect("two captured views compile");
    assert_storage_descriptor_is_accessed(&lowered.spirv, 2, 0);
    assert_storage_descriptor_is_accessed(&lowered.spirv, 2, 1);
}

/// A captured view passed to a user function that itself indexes it (→
/// recursive per-buffer specialization) must read from the captured buffer.
#[test]
fn view_through_nested_fn_specialization_reads_own_buffer() {
    let src = r#"
        def firstx(zs: []f32) f32 = zs[0]

        #[compute]
        entry tick(#[storage(set=2, binding=0, access=read)] xs: []f32) []f32 =
          map(|_: i32| firstx(xs), 0i32..<4)
    "#;
    let lowered = crate::compile_thru_spirv(src).expect("nested view specialization compiles");
    assert_storage_descriptor_is_accessed(&lowered.spirv, 2, 0);
}

/// A view used directly as a `map` *input* (→ the entry walker
/// `rewrite_term` / SOAC-input path, not a capture) must read from its buffer.
#[test]
fn view_as_map_input_reads_own_buffer() {
    let src = r#"
        #[compute]
        entry tick(#[storage(set=2, binding=0, access=read)] xs: []f32) []f32 =
          map(|x: f32| x * 2.0, xs)
    "#;
    let lowered = crate::compile_thru_spirv(src).expect("view-as-map-input compiles");
    assert_storage_descriptor_is_accessed(&lowered.spirv, 2, 0);
}

/// Assert that some `OpArrayLength` queries the runtime-array length of the
/// `(set, binding)` descriptor — the lowering of `length(view)`. Distinct from
/// `assert_storage_descriptor_is_accessed`: a length query is an `OpArrayLength`
/// on the buffer struct, not an `OpAccessChain` into it.
fn assert_array_length_queried_on_descriptor(spirv_words: &[u32], set: u32, binding: u32) {
    use std::collections::HashMap;
    use wspirv::binary::parse_words;
    use wspirv::dr::{Loader, Operand};
    use wspirv::spirv::{Decoration, Op};

    let mut loader = Loader::new();
    parse_words(spirv_words, &mut loader).expect("parse spirv");
    let module = loader.module();

    let mut sets: HashMap<u32, u32> = HashMap::new();
    let mut binds: HashMap<u32, u32> = HashMap::new();
    for inst in &module.annotations {
        if inst.class.opcode != Op::Decorate {
            continue;
        }
        let target = match inst.operands.first() {
            Some(Operand::IdRef(id)) => *id,
            _ => continue,
        };
        match (inst.operands.get(1), inst.operands.get(2)) {
            (Some(Operand::Decoration(Decoration::DescriptorSet)), Some(Operand::LiteralBit32(n))) => {
                sets.insert(target, *n);
            }
            (Some(Operand::Decoration(Decoration::Binding)), Some(Operand::LiteralBit32(n))) => {
                binds.insert(target, *n);
            }
            _ => {}
        }
    }

    let target_var = sets
        .iter()
        .find(|(id, s)| **s == set && binds.get(id) == Some(&binding))
        .map(|(id, _)| *id)
        .unwrap_or_else(|| panic!("no StorageBuffer variable decorated (set={set}, binding={binding})"));

    let queried = module.functions.iter().any(|f| {
        f.blocks.iter().any(|b| {
            b.instructions.iter().any(|inst| {
                inst.class.opcode == Op::ArrayLength
                    && matches!(inst.operands.first(), Some(Operand::IdRef(s)) if *s == target_var)
            })
        })
    });

    assert!(
        queried,
        "descriptor (set={set}, binding={binding}) is declared but its length is never \
         queried by an OpArrayLength — length(view) provenance was lost"
    );
}

/// `length(view)` on an entry param must query *its* descriptor. The binding is
/// baked into `_w_storage_len(set, binding)` as constants (not the side-map), so
/// this also pins that the right `(set, binding)` reaches the `OpArrayLength`.
#[test]
fn entry_length_queries_own_buffer() {
    let src = r#"
        #[compute]
        entry tick(#[storage(set=2, binding=0, access=read)] xs: []f32) []f32 =
          map(|i: i32| xs[i] * f32.i32(length(xs)), 0i32..<4)
    "#;
    let lowered = crate::compile_thru_spirv(src).expect("entry length(view) compiles");
    assert_array_length_queried_on_descriptor(&lowered.spirv, 2, 0);
    // and the indexed read still hits the same descriptor
    assert_storage_descriptor_is_accessed(&lowered.spirv, 2, 0);
}

/// A `scan` over a view threads that view through loop-carried block params
/// (scan-DPS phase 1). This is exactly the path `propagate_view_provenance`
/// keeps correct today; the guard pins that the loop-carried read still resolves
/// to the input descriptor, so the Tier-2 deletion of that propagation (binding
/// in the type) can't silently re-route it.
#[test]
fn scan_over_view_reads_own_buffer() {
    let src = r#"
        #[compute]
        entry tick(#[storage(set=2, binding=0, access=read)] xs: []f32) []f32 =
          scan(|a: f32, b: f32| a + b, 0.0, xs)
    "#;
    let lowered = crate::compile_thru_spirv(src).expect("scan over a view compiles");
    assert_storage_descriptor_is_accessed(&lowered.spirv, 2, 0);
}

/// Merging two views at *distinct* descriptors — `if c then xs else ys` — has
/// no single static binding, so it must not compile. Type inference unifies the
/// two branches' region variables into one; `pin_entry_regions` then tries to
/// pin that one variable to both `Region(2,0)` and `Region(2,1)`, detects the
/// conflict, and rejects it — rather than silently reading the wrong buffer.
#[test]
fn merge_of_distinct_buffers_is_a_type_error() {
    let src = r#"
        #[compute]
        entry tick(
          #[storage(set=2, binding=0, access=read)] xs: []f32,
          #[storage(set=2, binding=1, access=read)] ys: []f32,
          #[uniform(set=1, binding=0)] c: u32
        ) []f32 =
          map(|i: i32| (if c > 0u32 then xs else ys)[i], 0i32..<4)
    "#;
    let err = crate::compile_thru_spirv(src)
        .err()
        .expect("merging xs and ys (distinct descriptors) must not compile");
    let msg = format!("{err}");
    assert!(
        msg.contains("region") || msg.contains("binding") || msg.contains("descriptor"),
        "expected a region/binding-mismatch type error, got: {msg}"
    );
}

// ---- Constructor-style type conversions `T(value)` ----
//
// The `i32(x)` form dispatches via the existing per-type catalog
// entries (`i32.f32`, etc.); the `vec2i32(v)` form additionally
// desugars at `to_tlc` time into a `VecLit` of componentwise scalar
// conversion calls. These tests pin the end-to-end pipeline.

#[test]
fn ctor_scalar_constructor_compiles_to_spirv() {
    let src = r#"
        #[compute]
        entry tick(#[storage(set=2, binding=0, access=read)] xs: []f32,
                   #[uniform(set=1, binding=0)] n: u32) []i32 =
          map(|x: f32| i32(x), xs)
    "#;
    crate::compile_thru_spirv(src).expect("i32(f32) constructor must compile to SPIR-V");
}

#[test]
fn ctor_scalar_constructor_matches_legacy_dot_form() {
    let new = r#"
        #[compute]
        entry tick(#[storage(set=2, binding=0, access=read)] xs: []f32,
                   #[uniform(set=1, binding=0)] n: u32) []i32 =
          map(|x: f32| i32(x), xs)
    "#;
    let legacy = r#"
        #[compute]
        entry tick(#[storage(set=2, binding=0, access=read)] xs: []f32,
                   #[uniform(set=1, binding=0)] n: u32) []i32 =
          map(|x: f32| i32.f32(x), xs)
    "#;
    // Both must compile; backward-compat sanity for the legacy form.
    crate::compile_thru_spirv(new).expect("new T(value) form must compile");
    crate::compile_thru_spirv(legacy).expect("legacy T.source(value) form must still compile");
}

#[test]
fn ctor_vec2_constructor_compiles_to_spirv() {
    let src = r#"
        #[compute]
        entry tick(#[storage(set=2, binding=0, access=read)] xs: []vec2f32,
                   #[uniform(set=1, binding=0)] n: u32) []vec2i32 =
          map(|v: vec2f32| vec2i32(v), xs)
    "#;
    crate::compile_thru_spirv(src).expect("vec2i32(vec2f32) must compile to SPIR-V");
}

#[test]
fn ctor_vec3_and_vec4_constructors_compile_to_spirv() {
    let v3 = r#"
        #[compute]
        entry tick(#[storage(set=2, binding=0, access=read)] xs: []vec3i32,
                   #[uniform(set=1, binding=0)] n: u32) []vec3f32 =
          map(|v: vec3i32| vec3f32(v), xs)
    "#;
    let v4 = r#"
        #[compute]
        entry tick(#[storage(set=2, binding=0, access=read)] xs: []vec4u32,
                   #[uniform(set=1, binding=0)] n: u32) []vec4f32 =
          map(|v: vec4u32| vec4f32(v), xs)
    "#;
    crate::compile_thru_spirv(v3).expect("vec3f32(vec3i32) must compile");
    crate::compile_thru_spirv(v4).expect("vec4f32(vec4u32) must compile");
}

// ---- ArrayVariantAbstract — `filter` → size-polymorphic consumer ----
//
// `filter`'s return scheme is now `?k. Array[a, Abstract, k, no_region]`
// (was `Composite`). The producer's EGIR lowering picks Bounded for
// static-capacity inputs and View for runtime-sized ones; the consumer
// can be a size-polymorphic helper that gets specialized against the
// `Abstract` representation in TLC and resolved at the producer edge in
// EGIR. The backend-boundary verifier (`egir::verify_no_abstract`)
// rejects any residual `Array[_, Abstract, _, _]`.
//
// These pin the canonical patterns; for fusion-shape and runtime-length
// coverage see the older `filter_into_reduce_*` tests above.

#[test]
fn filter_into_user_size_poly_helper_compiles() {
    let src = r#"
def sum<[n]>(xs: [n]f32) f32 = reduce(|a: f32, b: f32| a + b, 0.0, xs)

#[compute]
entry tick(#[storage(set=2, binding=0, access=read)] xs: []f32) f32 =
  let kept = filter(|x: f32| x > 0.0, xs) in
  sum(kept)
"#;
    crate::compile_thru_spirv(src)
        .expect("`filter` piped through a user-defined size-poly helper must compile to SPIR-V");
}

#[test]
fn filter_into_user_size_poly_helper_static_capacity() {
    // Static-capacity input exercises the Bounded producer path
    // (filter result is `{buffer: [8]f32, len: u32}`).
    let src = r#"
def sum<[n]>(xs: [n]f32) f32 = reduce(|a: f32, b: f32| a + b, 0.0, xs)

#[compute]
entry tick() f32 =
  let kept = filter(|x: f32| x > 0.0, [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0]) in
  sum(kept)
"#;
    crate::compile_thru_spirv(src)
        .expect("static-capacity filter piped through a size-poly helper must compile");
}

/// Regression: a `filter -> map -> reduce` chain whose reduced result
/// feeds a vector op + swizzle, inside a helper called from a compute
/// `map`. Distilled from the `separation` boids force in
/// `testfiles/playground/particles.wyn`. Two defects had to be fixed:
///   1. the intermediate `map`'s result carried the existential
///      `Composite` + `Skolem` size opened from the `filter`, which the
///      backend can't lower (panic: "invalid size argument: Skolem").
///      `convert_soac_map` now inherits the input's shape (`from_tlc.rs`).
///   2. the chain didn't fuse: the swizzle desugars the `reduce` under a
///      binop (`let r = reduce(..) * k`), and `normalize` lifted it only
///      into a *nested* let, invisible to the top-level-only fusion
///      driver — so the `filter` was materialized and hit "ArrayWith on
///      an unsized scratch". `normalize` now flattens nested lets so the
///      reduce joins the top-level chain and `map->reduce` /
///      `filter->reduce` collapse it to a masked fused reduce.
#[test]
fn filter_map_reduce_vecop_swizzle_in_helper_compiles() {
    let src = r#"
def f(arr: []vec4f32) vec2f32 =
  let selected = filter(|d| d.x < 1.0, arr) in
  let contributions = map(|d| d * 2.0, selected) in
  (reduce(|a, b| a + b, @[0.0, 0.0, 0.0, 0.0], contributions) * 0.1).xy

#[compute]
entry e(#[storage(set=2, binding=0, access=read)] arr0: []vec4f32) []vec2f32 =
  let arr = arr0[0..512] in
  map(|p: vec4f32| f(arr), arr)
"#;
    crate::compile_thru_spirv(src)
        .expect("filter -> map -> reduce -> vec-op -> swizzle in a non-inlined helper must compile");
}

// ---- Missing fusion combinations (open gaps) --------------------------------
//
// `tlc::array_semantics::can_fuse` returns a buildable `FusionRecipe` for 5
// producer->consumer pairs: Map->Map, Map->Reduce, Map->Scan, Map->Scatter,
// Filter->Reduce. The combinations below are NOT fused but DO compile
// (materialized intermediates). Fusing them is a perf TODO, not a correctness
// gap; the Filter-producer cases lower via `array_with` on the Bounded variant.

/// Filter -> Map (a "filtered map"). `map(g, filter(p, a))` used directly.
/// Compiles: the filter is materialized as a Bounded result and the map runs
/// over it (unfused). Fusing it into a single compact pass (CompactMap/mapMaybe)
/// is a perf TODO; correctness comes from `array_with` supporting the Bounded
/// variant (the struct's [N]T buffer member).
#[test]
fn filter_into_map_compiles() {
    let src = r#"
#[compute]
entry e(#[storage(set=2, binding=0, access=read)] a: []f32,
        #[storage(set=2, binding=1, access=write)] o: *[]f32) () =
  let m = map(|x: f32| x * 2.0, filter(|x: f32| x > 0.0, a[0..256])) in
  let _ = scatter(o, [0i32], [m[0]]) in ()
"#;
    crate::compile_thru_spirv(src).expect("Filter->Map should compile");
}

/// Range -> Reduce, e.g. `reduce(op, ne, lo..<hi)`. The iota is NOT
/// materialized: a `Range` lowers to a Virtual array `{start, step, len}` and the
/// reduce reads each element as `start + i*step` arithmetic inside its own loop
/// (see `egir/soac_expand.rs` `is_virtual_source`). So this is already optimally
/// fused at the backend level — no fusion-engine Range builder is needed. We
/// assert exactly ONE loop in the MIR: a materialized-then-reduced range would
/// emit two (one to fill the buffer, one to fold it).
#[test]
fn range_into_reduce_is_virtual_single_loop() {
    let src = r#"
#[compute]
entry e(#[storage(set=2, binding=0, access=write)] o: *[]i32) () =
  let s = reduce(|a: i32, b: i32| a + b, 0i32, 0i32 ..< 256) in
  let _ = scatter(o, [0i32], [s]) in ()
"#;
    let ssa = crate::compile_thru_ssa(src).expect("Range->Reduce should compile");
    let mir = crate::ssa::print::format_program(&ssa.ssa);
    let loops = mir.matches("loop merge").count();
    assert_eq!(
        loops, 1,
        "Range->Reduce should fuse to a single virtual-source loop (no materialized iota); \
         found {loops} loops in MIR:\n{mir}"
    );
}

/// Range -> Scan, e.g. `scan(op, ne, lo..<hi)`. Like Range->Reduce, the iota
/// stays Virtual (`start + i*step` read on the fly), so it compiles without
/// materializing a backing buffer.
#[test]
fn range_into_scan_compiles() {
    let src = r#"
#[compute]
entry e(#[storage(set=2, binding=0, access=write)] o: *[]i32) () =
  let s = scan(|a: i32, b: i32| a + b, 0i32, 0i32 ..< 256) in
  let _ = scatter(o, [0i32], [s[0]]) in ()
"#;
    crate::compile_thru_spirv(src).expect("Range->Scan should compile");
}

/// Filter -> Scan, e.g. `scan(op, ne, filter(p, a))`. Compiles: two fixes
/// combine — the `convert_soac_scan` shape-preserving `project_ty` guard (stops
/// the filter's `Skolem` size leaking into the scan) and `array_with` supporting
/// the Bounded variant (so the filter compaction lowers). The filter is
/// materialized as a compact result and scanned (unfused, which is the correct
/// semantics — a compact scan, not a masked scan over the original).
#[test]
fn filter_into_scan_compiles() {
    let src = r#"
#[compute]
entry e(#[storage(set=2, binding=0, access=read)] a: []f32,
        #[storage(set=2, binding=1, access=write)] o: *[]f32) () =
  let s = scan(|x: f32, y: f32| x + y, 0.0, filter(|x: f32| x > 0.0, a[0..256])) in
  let _ = scatter(o, [0i32], [s[0]]) in ()
"#;
    crate::compile_thru_spirv(src).expect("Filter->Scan should compile");
}

/// Scan -> Map, e.g. `map(g, scan(op, ne, a))`. Compiles (unfused). Fusing it
/// (post-compose the map onto the scan's output) is a perf TODO.
#[test]
fn scan_into_map_compiles() {
    let src = r#"
#[compute]
entry e(#[storage(set=2, binding=0, access=read)] a: []f32,
        #[storage(set=2, binding=1, access=write)] o: *[]f32) () =
  let m = map(|x: f32| x + 1.0, scan(|x: f32, y: f32| x + y, 0.0, a[0..256])) in
  let _ = scatter(o, [0i32], [m[0]]) in ()
"#;
    crate::compile_thru_spirv(src).expect("Scan->Map should compile");
}

/// Regression: an entry returning a tuple where the second output is a
/// fixed-size literal that *indexes into a scan result* used to silently
/// drop the second output's binding from the descriptor JSON. Root
/// cause: `lift_gathers::lift_entry` read `out_count` off `def.ty`'s
/// return slot, but `normalize_outputs` rewrites that to `SideEffect`
/// (the body's tail is an `OutputSlotStore` chain). The old
/// `storage_output_count(SideEffect)` undercounted, so the gather
/// intermediate landed on the binding slot the second output expected.
/// Fix reads `decl.outputs.len()` directly.
#[test]
fn entry_tuple_output_with_scan_indexed_literal_keeps_both_bindings() {
    use crate::pipeline_descriptor::{BufferUsage, Pipeline};
    let lowered = crate::compile_thru_spirv(
        "\
#[compute]
entry gen(xs: []i32, #[uniform(set=1,binding=0)] n: i32) ([]vec4f32, [5]i32) =
  let offsets = scan(|a:i32,b:i32| a+b, 0, xs) in
  (map(|i:i32| @[f32.i32(i),0.0,0.0,1.0], iota(64)),
   [36, offsets[n - 1], 0, 0, 0])
",
    )
    .expect("scan-into-tuple-literal must compile");
    let gen_pipeline = lowered
        .pipeline
        .pipelines
        .iter()
        .find_map(|p| match p {
            Pipeline::Compute(c) if c.stages.iter().any(|s| s.entry_point == "gen") => Some(c),
            _ => None,
        })
        .expect("compute pipeline `gen` present");
    let output_names: Vec<&str> = gen_pipeline
        .bindings
        .iter()
        .filter_map(|b| match b {
            crate::pipeline_descriptor::Binding::StorageBuffer {
                usage: BufferUsage::Output,
                name,
                ..
            } => Some(name.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        output_names.contains(&"gen_output_0") && output_names.contains(&"gen_output_1"),
        "both gen_output_0 and gen_output_1 must be present as outputs in the descriptor; got {output_names:?}"
    );
}

/// Infix bitwise/shift operators must lower to the matching SPIR-V ops.
/// `^` → OpBitwiseXor and `<<` → OpShiftLeftLogical; the operands are
/// unsigned so this also pins the (UInt, _) arm of `lower_binop`.
#[test]
fn bitwise_shift_ops_lower_to_spirv() {
    use wspirv::binary::parse_words;
    use wspirv::dr::Loader;
    use wspirv::spirv::Op;

    let spirv = compile_to_spirv(
        "\
#[compute]
entry e(xs: []u32) []u32 = map(|x: u32| (x ^ 5u32) << 1u32, xs)
",
    )
    .expect("infix bitwise/shift compiles to SPIR-V");

    let mut loader = Loader::new();
    parse_words(&spirv, &mut loader).expect("parse spirv");
    let module = loader.module();

    let (mut xors, mut shls) = (0, 0);
    for func in &module.functions {
        for block in &func.blocks {
            for inst in &block.instructions {
                match inst.class.opcode {
                    Op::BitwiseXor => xors += 1,
                    Op::ShiftLeftLogical => shls += 1,
                    _ => {}
                }
            }
        }
    }
    assert!(xors >= 1, "expected at least one OpBitwiseXor, found {xors}");
    assert!(
        shls >= 1,
        "expected at least one OpShiftLeftLogical, found {shls}"
    );
}

/// A function whose body uses bitwise/shift operators with a reused let-local
/// miscompiles when it is inlined BOTH into a captured value hoisted before a
/// SOAC and into the SOAC's lambda: the local leaks as "Unknown global: w"
/// during SPIR-V generation.
///
/// Bisected trigger (all three required):
///   1. bitwise/shift body with a reused let-local: `let w = .. in (w >> _) ^ w`
///   2. the fn called to produce a *captured* value: `let k = f(7u32) in ..`
///   3. the fn *also* called inside the SOAC lambda
/// An arithmetic-only body, a literal (non-call) `k`, or calling `f` only
/// inside the lambda each compile fine. Surfaced by the lib/rng.wyn PCG hash
/// (`pcg` has `let w = .. in (w >> 22) ^ w`, used for the hoisted key and
/// inside the per-element map).
#[test]
fn bitwise_fn_inlined_both_captured_and_in_soac_lambda_lowers() {
    compile_to_spirv(
        "\
def f(v: u32) u32 = let w = v ^ 1u32 in (w >> 1u32) ^ w
#[compute]
entry e() []f32 =
  let k = f(7u32) in
  map(|i: i32| f32.u32(f(k + u32.i32(i))), 0i32 ..< 4)
",
    )
    .expect("bitwise fn inlined both as captured value and in SOAC lambda must lower to SPIR-V");
}

/// `inner` captures `x` transitively through the intermediate lambda `outer`.
/// partial_eval inlines the constant call `nested_lambda(100)`, dissolving the
/// inner `let outer = <lambda>`; `apply_var` must apply the call through that
/// env-bound lambda, otherwise `outer` is left dangling and closure conversion
/// mis-threads its capture (`ArityMismatch`). Regression for the env-bound
/// lambda case of dissolved-let residualization.
#[test]
fn nested_transitive_capture_through_inlined_lambda_lowers() {
    let _ = compile_to_ssa(
        "\
def nested_lambda(x: i32) i32 =
  let outer = |a: i32|
    let inner = |b: i32| a + b + x in
    inner(a)
  in
  outer(5)
#[vertex]
entry v() #[builtin(position)] vec4f32 = @[f32.i32(nested_lambda(100)), 0.0, 0.0, 1.0]
",
    );
}

/// partial_eval folds integer arithmetic; a u32 multiply like `C * K`
/// overflows u32 (and its i128-free product would overflow i64), so the fold
/// must wrap mod 2^32 rather than emit an out-of-range literal
/// ("Invalid u32"). Surfaced by lib/rng.wyn's PCG hash.
#[test]
fn folded_u32_arithmetic_wraps_to_width() {
    compile_to_spirv(
        "\
def C: u32 = 2654435769u32
#[compute]
entry e() []u32 = map(|i: i32| C * 747796405u32 + 2891336453u32, 0i32 ..< 4)
",
    )
    .expect("folded overflowing u32 arithmetic must wrap, not error");
}

/// A deep chain of `let (x0, x1) = mix(x0, x1, ..)` — each step uses the
/// previous result twice — inlined by partial_eval must not duplicate the
/// residual at every use site. Doing so is exponential in the chain depth
/// (the term doubles per step) and at shallower depth also drops a binding
/// ("Unknown global: x0"). partial_eval keeps non-trivial residual `let`s
/// shared instead. Surfaced by lib/rng.wyn's Threefry `block`.
#[test]
fn deep_tuple_let_chain_keeps_sharing() {
    compile_to_spirv(
        "\
module type RA = {
  type key
  sig at(k: key, p: u32) u32
}
module g : RA = {
  type key = (u32, u32)
  def mix(a: u32, b: u32, r: u32) (u32, u32) = let y = a + b in (y, (b << r) ^ y)
  def block(c0: u32, c1: u32, k0: u32, k1: u32) (u32, u32) =
    let (x0, x1) = mix(c0 + k0, c1 + k1, 13u32) in
    let (x0, x1) = mix(x0, x1, 15u32) in
    let (x0, x1) = mix(x0, x1, 26u32) in
    let (x0, x1) = mix(x0, x1, 6u32) in
    let (x0, x1) = mix(x0, x1, 17u32) in
    let (x0, x1) = mix(x0, x1, 29u32) in
    let (x0, x1) = mix(x0, x1, 16u32) in
    let (x0, x1) = mix(x0, x1, 24u32) in
    (x0 + k0, x1 + k1)
  def at(k: key, p: u32) u32 = let (k0, k1) = k in let (r0, _) = block(p, 0u32, k0, k1) in r0
}
#[compute]
entry e() []u32 = map(|i: i32| g.at((0x9e3779b9u32, 0x243f6a88u32), u32.i32(i)), 0i32 ..< 4)
",
    )
    .expect("deep tuple-destructure let-chain must lower without blowup or dangling vars");
}

// =========================================================================
// Captured compiler panics — minimal reproducers
//
// Each test here pins a Wyn-source shape that currently panics during
// compilation. Tests use `#[should_panic]` so the suite stays green while
// the fixture stays committed; drop `#[should_panic]` when the panic is
// fixed to make the test a passing regression.
// =========================================================================

/// `T(value)` where `value` is already of type `T` errors with
/// "Partial application not allowed: result is function type T -> T"
/// instead of resolving as the identity conversion.
///
/// Root cause: `try_resolve_constructor_call` builds its candidate set
/// from `lookup_by_surface_prefix(T)`, which returns every `T.*`
/// catalog entry — operators (`T.+ : T -> T -> T`), unary intrinsics
/// (`T.abs : T -> T`), as well as conversions (`T.<source> : source -> T`).
/// Overload resolution picks a 2-arg operator (catalog insertion order
/// puts operators before conversions); applying it to one arg yields
/// a function-typed result that `ensure_not_partial` rejects.
///
/// Fix direction: filter `lookup_by_surface_prefix` results to only
/// the conversion entries (those whose suffix is a primitive type
/// name) before handing them to the overload resolver. Or carry a
/// `is_conversion` marker on the catalog entry.
#[test]
fn constructor_form_same_type_conversion_is_identity() {
    crate::compile_thru_frontend("def f(x: i32) i32 = i32(x)")
        .expect("i32(x) where x: i32 should resolve as the identity conversion");
}

/// Two compute entries in one module: both entries' outputs land on
/// distinct `(set 0, binding N)` slots even when their input shapes
/// are identical. Single shared `IdSource<u32>` across all
/// compiler-allocated set-0 bindings guarantees no two `OpVariable`s
/// share a slot.
#[test]
fn two_compute_entries_do_not_collide_on_auto_bindings() {
    let lowered = crate::compile_thru_spirv(
        r#"
#[compute]
entry a(xs: []u32) []vec4f32 = map(|x| @[f32.u32(x), 0.0, 0.0, 0.0], xs)
#[compute]
entry b(xs: []u32) []f32 = map(|x| f32.u32(x), xs)
"#,
    )
    .expect("two compute entries with same input shape should compile to one valid SPIR-V module");
    // The SPIR-V should pass spirv-val. Smoke-check that the byte
    // count looks plausible; the real assertion is that the call
    // above returned Ok and the descriptor doesn't put two
    // differently-typed buffers on the same (set, binding).
    assert!(!lowered.spirv.is_empty());
}

/// Two compute entries whose inputs auto-allocate to the same `(set 0,
/// binding 0)` slot with DIFFERENT element types must each get a fresh
/// slot — sharing one `OpVariable` between a `[]u32` and a `[]f32`
/// trips `spirv-val: OpAccessChain result type '%float' does not match
/// indexing into base '%uint'`. Fix: `pin_entry_regions` owns a single
/// `IdSource<u32>` across all entries so input bindings can't alias.
#[test]
fn two_compute_entries_with_differently_typed_inputs_do_not_alias() {
    let lowered = crate::compile_thru_spirv(
        r#"
#[compute]
entry a(xs: []u32) []u32 = map(|x| x, xs)
#[compute]
entry b(ys: []f32) []f32 = map(|y| y, ys)
"#,
    )
    .expect("heterogeneously-typed compute-entry inputs must compile to one valid SPIR-V module");
    assert!(!lowered.spirv.is_empty());
}

/// Two entries binding the *same* explicit `#[storage(set, binding)]`
/// slot to buffers with different element types (`[]f32` vs `[]vec4f32`)
/// must be rejected at compile time. The compiler coalesces same-slot
/// storage into one module-global whose type is the first declaration's;
/// the other entry then indexes it as the wrong element type, producing
/// `spirv-val: OpAccessChain result type ... does not match indexing into
/// base ...`. Reaching SPIR-V at all is the bug — the type checker must
/// reject the conflicting interface first.
#[test]
fn conflicting_explicit_storage_binding_across_entries_is_rejected() {
    let result = crate::compile_thru_spirv(
        r#"
#[compute]
entry ent_a(idx: []u32, #[storage(set=1, binding=0, access=read)] buf: []f32) []f32 =
  map(|s| buf[i32.u32(s)], idx)
#[compute]
entry ent_b(idx: []u32, #[storage(set=1, binding=0, access=read)] buf: []vec4f32) []f32 =
  map(|s| buf[i32.u32(s)].x, idx)
"#,
    );
    let msg = match result {
        Ok(_) => panic!("a (set, binding) reused with a conflicting element type must be a compile error"),
        Err(e) => e.to_string(),
    };
    assert!(
        msg.contains("set=1, binding=0") && msg.contains("must use the same element type"),
        "expected a buffer element-type conflict diagnostic, got: {msg}"
    );
}

/// The dual of the rejection test: two entries may legitimately share an
/// explicit `#[storage(set, binding)]` slot when they agree on the
/// element type — that's how a pipeline wires one buffer into several
/// stages. This must compile to one valid module, not trip the
/// conflict check on the differing array-length variables.
#[test]
fn matching_explicit_storage_binding_across_entries_compiles() {
    let lowered = crate::compile_thru_spirv(
        r#"
#[compute]
entry ent_a(idx: []u32, #[storage(set=1, binding=0, access=read)] buf: []f32) []f32 =
  map(|s| buf[i32.u32(s)], idx)
#[compute]
entry ent_b(idx: []u32, #[storage(set=1, binding=0, access=read)] buf: []f32) []f32 =
  map(|s| buf[i32.u32(s)] * 2.0, idx)
"#,
    )
    .expect("entries that agree on a shared (set, binding) element type must compile");
    assert!(!lowered.spirv.is_empty());
}

/// A compute entry writing a `storage_image` and a fragment entry sampling a
/// `texture2d` at the *same* `(set, binding)` is valid multi-pass aliasing —
/// the runtime binds one image to both, written then sampled. Different
/// resource kinds get separate per-entry SPIR-V variables, so this is not a
/// conflict (the narrow guard only compares same-kind buffers).
#[test]
fn cross_kind_image_aliasing_compiles() {
    let lowered = crate::compile_thru_spirv(
        r#"
#[compute]
entry paint(#[storage_image(set=0, binding=0, format=rgba8unorm, access=write_only)] img: *storage_image,
            #[builtin(global_invocation_id)] gid: vec3u32) () =
  img with [@[i32.u32(gid.x), i32.u32(gid.y)]] = @[1.0, 0.0, 0.0, 1.0]
#[vertex]
entry vertex_main(#[builtin(vertex_index)] vid: i32) #[builtin(position)] vec4f32 =
  let verts = [@[-1.0, -1.0, 0.0, 1.0], @[3.0, -1.0, 0.0, 1.0], @[-1.0, 3.0, 0.0, 1.0]] in
  verts[vid]
#[fragment]
entry fragment_main(#[builtin(position)] pos: vec4f32,
                    #[texture(set=0, binding=0)] tex: texture2d,
                    #[sampler(set=0, binding=1)] samp: sampler) #[target(screen)] vec4f32 =
  texture_sample(tex, samp, @[pos.x / 1024.0, pos.y / 1024.0], 0.0)
"#,
    )
    .expect("cross-kind image aliasing at one (set, binding) must compile");
    assert!(!lowered.spirv.is_empty());
}

/// A named `resource` viewed `storage_write` by a compute entry and `sampled`
/// by a fragment entry compiles to one valid module — the resource/view form
/// of the compute-write / fragment-sample handoff. The resource omits `layout`,
/// so the compiler auto-assigns its binding.
#[test]
fn resource_view_ping_pong_compiles() {
    let lowered = crate::compile_thru_spirv(
        r#"
resource color: image2d {
  format = rgba8unorm
  size   = 1024x1024
  usages = [storage_write, sampled]
}

#[compute]
entry paint(#[view(color, storage_write)] img: *storage_image,
            #[builtin(global_invocation_id)] gid: vec3u32) () =
  img with [@[i32.u32(gid.x), i32.u32(gid.y)]] = @[1.0, 0.0, 0.0, 1.0]
#[vertex]
entry vertex_main(#[builtin(vertex_index)] vid: i32) #[builtin(position)] vec4f32 =
  let verts = [@[-1.0, -1.0, 0.0, 1.0], @[3.0, -1.0, 0.0, 1.0], @[-1.0, 3.0, 0.0, 1.0]] in
  verts[vid]
#[fragment]
entry show(#[builtin(position)] pos: vec4f32,
           #[view(color, sampled)] tex: texture2d,
           #[sampler(set=0, binding=1)] samp: sampler) #[target(screen)] vec4f32 =
  texture_sample(tex, samp, @[pos.x / 1024.0, pos.y / 1024.0], 0.0)
"#,
    )
    .expect("a resource viewed write+sampled across entries must compile");
    assert!(!lowered.spirv.is_empty());
}

#[test]
fn loop_bodied_map_uses_storage_image_globals_not_function_parameters() {
    use std::collections::{HashMap, HashSet};
    use wspirv::binary::parse_words;
    use wspirv::dr::{Loader, Operand};
    use wspirv::spirv::Op;

    let lowered = crate::compile_thru_spirv(
        r#"
#[compute]
entry r(xs: []u32,
        #[storage_image(set=1, binding=0, format=r32float, access=read_only)] src: storage_image) []u32 =
  map(|s|
        let x = i32(s) in
        let (_, total) = loop (j, total) = (0, 0.0) while j < 2 do
          let px = image_load(src, @[x, j]) in
          (j + 1, total + px.x) in
        if total > 0.0 then 1u32 else 0u32,
      xs)
"#,
    )
    .expect("loop-bodied map with storage image compiles");

    assert_no_runtime_storage_image_handles(&lowered.spirv);

    let mut loader = Loader::new();
    parse_words(&lowered.spirv, &mut loader).expect("parse generated SPIR-V");
    let module = loader.module();
    let parameters: HashSet<_> = module
        .functions
        .iter()
        .flat_map(|function| function.parameters.iter().filter_map(|parameter| parameter.result_id))
        .collect();
    let loads: HashMap<_, _> = module
        .functions
        .iter()
        .flat_map(|function| &function.blocks)
        .flat_map(|block| &block.instructions)
        .filter_map(|instruction| {
            if instruction.class.opcode != Op::Load {
                return None;
            }
            match (instruction.result_id, instruction.operands.first()) {
                (Some(result), Some(Operand::IdRef(pointer))) => Some((result, *pointer)),
                _ => None,
            }
        })
        .collect();
    let globals: HashSet<_> = module
        .types_global_values
        .iter()
        .filter(|instruction| instruction.class.opcode == Op::Variable)
        .filter_map(|instruction| instruction.result_id)
        .collect();
    let global_bindings: HashMap<_, _> = module
        .annotations
        .iter()
        .filter_map(|instruction| match instruction.operands.as_slice() {
            [Operand::IdRef(variable), Operand::Decoration(wspirv::spirv::Decoration::Binding), Operand::LiteralBit32(binding)] => {
                Some((*variable, *binding))
            }
            _ => None,
        })
        .collect();
    let mut image_ops = 0;
    for instruction in module
        .functions
        .iter()
        .flat_map(|function| &function.blocks)
        .flat_map(|block| &block.instructions)
        .filter(|instruction| matches!(instruction.class.opcode, Op::ImageRead | Op::ImageWrite))
    {
        image_ops += 1;
        let Some(Operand::IdRef(image)) = instruction.operands.first() else {
            panic!("image operation has no image operand")
        };
        assert!(
            !parameters.contains(image),
            "image operation targets OpFunctionParameter"
        );
        let global = loads.get(image).expect("image operand is loaded from its descriptor global");
        assert!(
            globals.contains(global),
            "image load pointer is not a module-scope OpVariable"
        );
        assert_eq!(instruction.class.opcode, Op::ImageRead);
        assert_eq!(global_bindings.get(global), Some(&0));
    }
    assert!(image_ops >= 1, "expected image_load");
}

/// A storage image shared across entries with mixed access — read in one entry,
/// written in another — collapses to a single module global. Its access
/// decoration must be the union of every view, so a read+write image carries
/// neither `NonReadable` nor `NonWritable`. With the reader entry first (as
/// here), the pre-fix behaviour decorated the global `NonWritable` and naga
/// rejected the writer's `OpImageWrite`; `assert_no_runtime_storage_image_handles`
/// runs naga validation, so it fails if the decoration regresses.
#[test]
fn storage_image_shared_read_and_write_across_entries_validates() {
    let source = r#"
#[compute]
entry reader(xs: []u32,
             #[storage_image(set=1, binding=0, format=r32float, access=read_only)] img: storage_image) []f32 =
  map(|s| let p = image_load(img, @[i32(s), 0]) in p.x, xs)
#[compute]
entry writer(#[builtin(global_invocation_id)] gid: vec3u32,
             #[storage_image(set=1, binding=0, format=r32float, access=write_only)] img: *storage_image) () =
  img with [@[i32.u32(gid.x), 0]] = @[1.0, 1.0, 1.0, 1.0]
"#;
    let lowered = crate::compile_thru_spirv(source).expect("shared read+write storage image compiles");
    assert_no_runtime_storage_image_handles(&lowered.spirv);
}

/// Minimal `history = 1` feedback program shared by the two KNOWN-BUG tests
/// below: one compute entry writing a resource's current frame and sampling
/// its `previous` view.
const HISTORY_FEEDBACK_SOURCE: &str = r#"
resource acc: image2d {
  format  = rgba32float
  size    = 64x64
  usages  = [storage_write, sampled]
  history = 1
}
#[compute]
entry step(#[view(acc, storage_write)] out_acc: *storage_image,
           #[view(acc, sampled, previous)] prev_acc: texture2d,
           #[builtin(global_invocation_id)] gid: vec3u32) () =
  let xy = @[i32(gid.x), i32(gid.y)] in
  let prev = texture_load(prev_acc, xy, 0) in
  out_acc with [xy] = prev + @[1.0, 0.0, 0.0, 0.0]
"#;

/// The `FeedbackPair` a `previous` view of a `history` resource records must
/// reach the published descriptor — it's what makes the runtime ping-pong the
/// two textures. Regression test: `tlc::parallelize` once seeded the
/// per-entry pipelines with an empty `feedback` instead of the entry's pairs.
#[test]
fn history_resource_feedback_pair_reaches_descriptor() {
    use crate::pipeline_descriptor::Pipeline;

    let lowered = crate::compile_thru_ssa(HISTORY_FEEDBACK_SOURCE).expect("history feedback compiles");
    let has_feedback = lowered.pipeline.pipelines.iter().any(|pipeline| match pipeline {
        Pipeline::Compute(compute) => !compute.feedback.is_empty(),
        Pipeline::Graphics(graphics) => !graphics.feedback.is_empty(),
    });
    assert!(
        has_feedback,
        "no pipeline carries the previous-view feedback pair"
    );
}

/// Resource/view metadata should publish an executable frame graph: storage
/// texture allocations carry their size, feedback views are marked as
/// previous-frame reads, and pass lifetimes come from descriptor resources.
#[test]
fn resource_dependencies_publish_frame_graph() {
    use crate::pipeline_descriptor::{
        FrameAccessRole, FrameHistoryRoleKind, FrameResourceExtent, FrameResourceKind, StorageTextureSize,
    };

    let lowered = crate::compile_thru_ssa(HISTORY_FEEDBACK_SOURCE).expect("history feedback compiles");
    let graph = &lowered.pipeline.frame_graph;
    assert!(!graph.passes.is_empty(), "frame graph has no passes");
    assert!(!graph.resources.is_empty(), "frame graph has no resources");
    assert_eq!(graph.feedback.len(), 1, "history pair should reach frame graph");

    let step = graph.passes.iter().find(|pass| pass.name == "step").expect("step pass reaches frame graph");
    assert!(
        step.reads.iter().any(|access| access.role == FrameAccessRole::Previous),
        "previous view should be a previous-frame read: {:?}",
        step.reads
    );
    assert!(!step.writes.is_empty(), "storage image write should be visible");

    let resource_index =
        graph.feedback[0].write_resource.expect("feedback write should resolve to a frame resource");
    let resource = &graph.resources[resource_index];
    // Views of one `resource` collapse to a single texture-kind frame resource
    // keyed by name; the storage allocation's size still rides on `extent`.
    assert_eq!(resource.kind, FrameResourceKind::Texture);
    assert_eq!(resource.first_pass, Some(0));
    assert_eq!(resource.last_pass, Some(0));
    assert!(matches!(
        resource.extent.as_ref(),
        Some(FrameResourceExtent::StorageTexture { size })
            if *size == (StorageTextureSize::Fixed {
                width: 64,
                height: 64
            })
    ));
    // The current-frame write and the previous-frame read are distinct
    // ping-pong buffers → distinct resources, linked by the feedback pair: the
    // write resource carries `WriteCurrent`, the read resource `ReadPrevious`.
    assert!(resource.history.iter().any(|role| role.role == FrameHistoryRoleKind::WriteCurrent));
    let read_index =
        graph.feedback[0].read_resource.expect("feedback read should resolve to a frame resource");
    let read_resource = &graph.resources[read_index];
    assert!(read_resource.history.iter().any(|role| role.role == FrameHistoryRoleKind::ReadPrevious));
}

/// Descriptor bindings must be slot-unique — a duplicated (set, binding)
/// fails wgpu bind-group-layout creation ("Conflicting binding"). Regression
/// test: the publication pass's claimed-slot snapshot skipped the texture /
/// sampler / storage-texture kinds, so a second pass re-pushed exactly those.
#[test]
fn history_resource_bindings_are_not_duplicated() {
    use crate::pipeline_descriptor::Pipeline;

    let lowered = crate::compile_thru_ssa(HISTORY_FEEDBACK_SOURCE).expect("history feedback compiles");
    for pipeline in &lowered.pipeline.pipelines {
        let Pipeline::Compute(compute) = pipeline else {
            continue;
        };
        let mut seen = std::collections::HashSet::new();
        for binding in &compute.bindings {
            let key = serde_json::to_string(binding).expect("serialize binding");
            assert!(
                seen.insert(key),
                "duplicate descriptor binding published: {binding:?}"
            );
        }
    }
}

/// A compute entry that writes a `#[storage_image]` (and has no SOAC-derived
/// domain) dispatches one thread per texel: `DerivedFrom { len: StorageImage }`,
/// which the host resolves from the bound texture's extent. An incidental
/// storage-buffer input (mountains' keyboard buffer) doesn't opt out — the
/// image is the domain. Regression test: the skeletal `Fixed {1,1,1}` survived
/// scheduling for serial per-texel entries (fresh mountains.wyn compiles
/// disagreed with its committed descriptor), so a window-sized pass only ever
/// ran one workgroup.
#[test]
fn storage_image_entry_dispatch_derives_from_image() {
    use crate::pipeline_descriptor::{DispatchLen, DispatchSize, Pipeline};

    // The mountains buffer_a shape: per-texel image pass that also reads a
    // raw storage buffer.
    let with_buffer_input = r#"
resource acc: image2d {
  format  = rgba32float
  size    = 64x64
  usages  = [storage_write, sampled]
  history = 1
}
#[compute]
entry step(#[view(acc, storage_write)] out_acc: *storage_image,
           #[view(acc, sampled, previous)] prev_acc: texture2d,
           #[storage(set=2, binding=0, access=read)] keyboard: []u32,
           #[builtin(global_invocation_id)] gid: vec3u32) () =
  let xy = @[i32(gid.x), i32(gid.y)] in
  let prev = texture_load(prev_acc, xy, 0) in
  let bump = if keyboard[0] != 0u32 then 1.0 else 0.0 in
  out_acc with [xy] = prev + @[bump, 0.0, 0.0, 0.0]
"#;
    for source in [HISTORY_FEEDBACK_SOURCE, with_buffer_input] {
        let lowered = crate::compile_thru_ssa(source).expect("per-texel image pass compiles");
        let stage = lowered
            .pipeline
            .pipelines
            .iter()
            .find_map(|pipeline| match pipeline {
                Pipeline::Compute(compute) => {
                    compute.stages.iter().find(|stage| stage.entry_point == "step")
                }
                _ => None,
            })
            .expect("step compute stage");
        assert!(
            matches!(
                &stage.dispatch_size,
                DispatchSize::DerivedFrom {
                    len: DispatchLen::StorageImage { .. },
                    ..
                }
            ),
            "per-texel entry should dispatch from its storage image, got {:?}",
            stage.dispatch_size
        );
    }
}

#[test]
fn explicit_compute_dispatch_grid_overrides_image_domain_inference() {
    use crate::pipeline_descriptor::{DispatchSize, Pipeline};

    let lowered = crate::compile_thru_ssa(
        r#"
resource color: image2d {
  format = rgba8unorm
  size   = 64x64
  usages = [storage_write]
}

#[compute]
#[dispatch(4, 8)]
entry paint(#[view(color, storage_write)] img: *storage_image,
            #[builtin(global_invocation_id)] gid: vec3u32) () =
  let xy = @[i32.u32(gid.x), i32.u32(gid.y)] in
  img with [xy] = @[1.0, 0.0, 0.0, 1.0]
"#,
    )
    .expect("explicit-dispatch storage image pass compiles");

    let stage = lowered
        .pipeline
        .pipelines
        .iter()
        .find_map(|pipeline| match pipeline {
            Pipeline::Compute(compute) => compute.stages.iter().find(|stage| stage.entry_point == "paint"),
            _ => None,
        })
        .expect("paint compute stage");
    assert_eq!(
        stage.dispatch_size,
        DispatchSize::Fixed {
            x: 4,
            y: 8,
            z: 1,
            explicit: true
        },
        "source-authored #[dispatch] should be an explicit launch domain"
    );
}

/// A source `#[dispatch(1,1,1)]` is the one grid value that collides with the
/// unspecified-default placeholder, yet it must still win over storage-image
/// domain inference — the explicit bit, not the value, decides.
#[test]
fn explicit_dispatch_one_one_one_survives_image_domain_inference() {
    use crate::pipeline_descriptor::{DispatchSize, Pipeline};

    let lowered = crate::compile_thru_ssa(
        r#"
resource color: image2d {
  format = rgba8unorm
  size   = 64x64
  usages = [storage_write]
}

#[compute]
#[dispatch(1, 1, 1)]
entry paint(#[view(color, storage_write)] img: *storage_image) () =
  img with [@[0, 0]] = @[1.0, 0.0, 0.0, 1.0]
"#,
    )
    .expect("explicit 1x1x1 storage image pass compiles");

    let stage = lowered
        .pipeline
        .pipelines
        .iter()
        .find_map(|pipeline| match pipeline {
            Pipeline::Compute(compute) => compute.stages.iter().find(|stage| stage.entry_point == "paint"),
            _ => None,
        })
        .expect("paint compute stage");
    assert_eq!(
        stage.dispatch_size,
        DispatchSize::Fixed {
            x: 1,
            y: 1,
            z: 1,
            explicit: true
        },
        "explicit #[dispatch(1,1,1)] must stay a single invocation, not upgrade \
         to the storage image's per-texel grid, got {:?}",
        stage.dispatch_size
    );
}

/// A `#[dispatch]` grid that provably under-covers a statically-sized
/// data-parallel domain is rejected rather than silently dropping the tail.
#[test]
fn undercovering_dispatch_grid_is_rejected() {
    // iota(100) is a fixed 100-element domain; #[dispatch(1,1,1)] at the
    // default 64-wide workgroup launches only 64 threads, dropping 36.
    let result = crate::compile_thru_ssa(
        r#"
#[compute]
#[dispatch(1, 1, 1)]
entry e() []i32 = map(|x| x * 2, iota(100))
"#,
    );
    let msg = match result {
        Ok(_) => panic!("under-covering #[dispatch] must be rejected"),
        Err(e) => e.to_string(),
    };
    assert!(
        msg.contains("would be dropped") && msg.contains("#[dispatch"),
        "diagnostic should explain the dropped elements, got: {msg}"
    );

    // A grid that covers the domain (2 * 64 = 128 >= 100) compiles.
    crate::compile_thru_ssa(
        r#"
#[compute]
#[dispatch(2, 1, 1)]
entry e() []i32 = map(|x| x * 2, iota(100))
"#,
    )
    .expect("a covering #[dispatch] grid compiles");
}

/// The former storage-image write function is no longer a surface builtin.
#[test]
fn legacy_image_store_is_not_user_visible() {
    let result = crate::compile_thru_spirv(
        r#"
#[compute]
entry r(xs: []u32,
        #[storage_image(set=1, binding=0, format=r32float, access=write_only)] img: storage_image) []u32 =
  map(|s|
        let i = i32(s)
        let _ = image_store(img, @[i, 0], @[1.0, 1.0, 1.0, 1.0]) in
        0u32,
      xs)
"#,
    );
    let msg = match result {
        Ok(_) => panic!("legacy image_store must not remain user-visible"),
        Err(e) => e.to_string(),
    };
    assert!(
        msg.contains("image_store"),
        "diagnostic should mention the removed builtin name, got: {msg}"
    );
}

#[test]
fn storage_image_with_tail_lowers_without_runtime_handle() {
    let source = r#"
#[compute]
entry r(#[storage_image(set=1, binding=0, format=r32float, access=write_only)] img: *storage_image,
        #[builtin(global_invocation_id)] gid: vec3u32) () =
  let xy = @[i32.u32(gid.x), i32.u32(gid.y)] in
  img with [xy] = @[1.0, 0.0, 0.0, 1.0]
"#;
    let lowered = crate::compile_thru_spirv(source).expect("linear image update compiles");
    assert_no_runtime_storage_image_handles(&lowered.spirv);

    let wgsl = crate::compile_thru_tlc(source)
        .expect("WGSL TLC")
        .infer_input_slice_bounds()
        .to_egraph()
        .expect("WGSL EGIR")
        .lower_to_ssa(crate::LoweringProfile::new(
            crate::CodegenTarget::Wgsl,
            crate::SchedulePolicy::Parallel,
        ))
        .expect("WGSL SSA")
        .lower_wgsl()
        .expect("WGSL lowering");
    assert!(
        wgsl.contains("textureStore("),
        "linear image update did not lower to textureStore:\n{wgsl}"
    );
}

#[test]
fn storage_image_with_explicit_entry_handle_return_erases_to_unit() {
    let source = r#"
#[compute]
entry r(#[storage_image(set=1, binding=0, format=r32float, access=write_only)] img: *storage_image,
        #[builtin(global_invocation_id)] gid: vec3u32) *storage_image =
  let xy = @[i32.u32(gid.x), i32.u32(gid.y)] in
  img with [xy] = @[1.0, 0.0, 0.0, 1.0]
"#;
    let lowered = crate::compile_thru_spirv(source)
        .expect("explicit storage-image handle entry return should erase to unit");
    assert_no_runtime_storage_image_handles(&lowered.spirv);
}

#[test]
fn storage_image_with_if_tail_has_no_void_phi() {
    use wspirv::binary::parse_words;
    use wspirv::dr::Loader;
    use wspirv::spirv::Op;

    let source = r#"
#[compute]
entry r(#[storage_image(set=1, binding=0, format=r32float, access=write_only)] img: *storage_image,
        #[builtin(global_invocation_id)] gid: vec3u32) () =
  let xy = @[i32.u32(gid.x), i32.u32(gid.y)] in
  if gid.x < 32u32 then
    img with [xy] = @[0.0, 0.0, 0.0, 1.0]
  else
    img with [xy] = @[1.0, 1.0, 1.0, 1.0]
"#;
    let lowered = crate::compile_thru_spirv(source).expect("linear image-update if compiles");
    assert_no_runtime_storage_image_handles(&lowered.spirv);

    let mut loader = Loader::new();
    parse_words(&lowered.spirv, &mut loader).expect("parse generated SPIR-V");
    let module = loader.module();
    let void_ty = module
        .types_global_values
        .iter()
        .find(|instruction| instruction.class.opcode == Op::TypeVoid)
        .and_then(|instruction| instruction.result_id);
    for function in &module.functions {
        for block in &function.blocks {
            for instruction in &block.instructions {
                if instruction.class.opcode == Op::Phi {
                    assert!(
                        instruction.result_type != void_ty,
                        "linear image-update if materialized an OpPhi %void"
                    );
                }
            }
        }
    }
}

#[test]
fn storage_image_with_loop_tail_lowers_without_runtime_handle() {
    let source = r#"
#[compute]
entry r(#[storage_image(set=1, binding=0, format=r32float, access=write_only)] img: *storage_image,
        #[builtin(global_invocation_id)] gid: vec3u32) () =
  let xy = @[i32.u32(gid.x), i32.u32(gid.y)] in
  loop img = img for i < 2 do
    img with [xy] = @[f32.i32(i), 0.0, 0.0, 1.0]
"#;
    let lowered = crate::compile_thru_spirv(source).expect("linear image-update loop compiles");
    assert_no_runtime_storage_image_handles(&lowered.spirv);

    let wgsl = crate::compile_thru_tlc(source)
        .expect("WGSL TLC")
        .infer_input_slice_bounds()
        .to_egraph()
        .expect("WGSL EGIR")
        .lower_to_ssa(crate::LoweringProfile::new(
            crate::CodegenTarget::Wgsl,
            crate::SchedulePolicy::Parallel,
        ))
        .expect("WGSL SSA")
        .lower_wgsl()
        .expect("WGSL lowering");
    assert!(
        wgsl.contains("textureStore("),
        "linear image-update loop did not lower to textureStore:\n{wgsl}"
    );
    assert!(
        !wgsl.lines().any(|line| line.starts_with("fn ") && line.contains("texture_storage_2d")),
        "storage image survived in a WGSL helper signature:\n{wgsl}"
    );
    let module = naga::front::wgsl::parse_str(&wgsl)
        .unwrap_or_else(|error| panic!("Naga rejected generated WGSL: {error:?}\n{wgsl}"));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|error| panic!("Naga validation rejected generated WGSL: {error:?}\n{wgsl}"));
}

#[test]
fn user_helper_specializes_storage_image_without_runtime_handle() {
    let lowered = crate::compile_thru_spirv(
        r#"
def read_pixel(img: storage_image, x: i32) f32 =
  let pixel = image_load(img, @[x, 0]) in pixel.x

#[compute]
entry r(xs: []u32,
        #[storage_image(set=1, binding=0, format=r32float, access=read_only)] src: storage_image) []u32 =
  map(|s| if read_pixel(src, i32(s)) > 0.0 then 1u32 else 0u32, xs)
"#,
    )
    .expect("storage-image helper specializes by binding");
    assert_no_runtime_storage_image_handles(&lowered.spirv);
}

fn assert_no_runtime_storage_image_handles(words: &[u32]) {
    use std::collections::{HashMap, HashSet};
    use wspirv::binary::parse_words;
    use wspirv::dr::{Loader, Operand};
    use wspirv::spirv::Op;

    let mut loader = Loader::new();
    parse_words(words, &mut loader).expect("parse generated SPIR-V");
    let module = loader.module();
    let image_types: HashSet<_> = module
        .types_global_values
        .iter()
        .filter(|instruction| instruction.class.opcode == Op::TypeImage)
        .filter_map(|instruction| instruction.result_id)
        .collect();

    for function_type in
        module.types_global_values.iter().filter(|instruction| instruction.class.opcode == Op::TypeFunction)
    {
        for operand in &function_type.operands {
            if let Operand::IdRef(ty) = operand {
                assert!(
                    !image_types.contains(ty),
                    "OpTypeFunction contains a storage-image return or parameter type"
                );
            }
        }
    }

    let mut value_types: HashMap<_, _> = HashMap::new();
    for instruction in &module.types_global_values {
        if let (Some(id), Some(ty)) = (instruction.result_id, instruction.result_type) {
            value_types.insert(id, ty);
        }
    }
    for function in &module.functions {
        for parameter in &function.parameters {
            if let (Some(id), Some(ty)) = (parameter.result_id, parameter.result_type) {
                assert!(
                    !image_types.contains(&ty),
                    "storage image survived as OpFunctionParameter"
                );
                value_types.insert(id, ty);
            }
        }
        for instruction in function.blocks.iter().flat_map(|block| &block.instructions) {
            if let (Some(id), Some(ty)) = (instruction.result_id, instruction.result_type) {
                value_types.insert(id, ty);
            }
        }
    }

    for call in module
        .functions
        .iter()
        .flat_map(|function| &function.blocks)
        .flat_map(|block| &block.instructions)
        .filter(|instruction| instruction.class.opcode == Op::FunctionCall)
    {
        for operand in call.operands.iter().skip(1) {
            if let Operand::IdRef(argument) = operand {
                assert!(
                    value_types.get(argument).is_none_or(|ty| !image_types.contains(ty)),
                    "OpFunctionCall passes a storage-image argument"
                );
            }
        }
    }

    let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let naga_module = naga::front::spv::parse_u8_slice(&bytes, &naga::front::spv::Options::default())
        .unwrap_or_else(|error| panic!("Naga rejected generated SPIR-V: {error:?}"));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&naga_module)
    .unwrap_or_else(|error| panic!("Naga validation rejected generated SPIR-V: {error:?}"));
}

/// A `#[view(...)]` naming a resource that doesn't exist is a compile error.
#[test]
fn view_of_unknown_resource_is_rejected() {
    let result = crate::compile_thru_spirv(
        r#"
#[compute]
entry paint(#[view(nope, storage_write)] img: *storage_image,
            #[builtin(global_invocation_id)] gid: vec3u32) () =
  img with [@[i32.u32(gid.x), i32.u32(gid.y)]] = @[1.0, 0.0, 0.0, 1.0]
"#,
    );
    let msg = match result {
        Ok(_) => panic!("a view of an undeclared resource must be a compile error"),
        Err(e) => e.to_string(),
    };
    assert!(
        msg.contains("unknown resource") && msg.contains("nope"),
        "got: {msg}"
    );
}

/// A raw `#[storage_image]` and `#[texture]` reusing one `(set, binding)` with
/// different image types is invalid: it emits two descriptor variables of
/// different types at one slot, which the runtime can't bind. The `resource`
/// form makes this unrepresentable, but hand-written raw bindings can still
/// express it; a future resolution-time check should reject it with an
/// incompatible-descriptor-kind error.
///
/// `#[ignore]`d: the collision check is not implemented yet — the module
/// currently compiles (and `cross_kind_image_aliasing_compiles` documents
/// that). Un-ignore once the check lands.
#[test]
#[ignore = "collision check not implemented: raw storage_image+texture at one (set,binding) still compiles"]
fn raw_cross_kind_same_binding_is_rejected() {
    let result = crate::compile_thru_spirv(
        r#"
#[compute]
entry paint(#[storage_image(set=0, binding=0, format=rgba8unorm, access=write_only)] img: *storage_image,
            #[builtin(global_invocation_id)] gid: vec3u32) () =
  img with [@[i32.u32(gid.x), i32.u32(gid.y)]] = @[1.0, 0.0, 0.0, 1.0]
#[vertex]
entry vertex_main(#[builtin(vertex_index)] vid: i32) #[builtin(position)] vec4f32 =
  let verts = [@[-1.0, -1.0, 0.0, 1.0], @[3.0, -1.0, 0.0, 1.0], @[-1.0, 3.0, 0.0, 1.0]] in
  verts[vid]
#[fragment]
entry fragment_main(#[builtin(position)] pos: vec4f32,
                    #[texture(set=0, binding=0)] tex: texture2d,
                    #[sampler(set=0, binding=1)] samp: sampler) #[target(screen)] vec4f32 =
  texture_sample(tex, samp, @[pos.x / 1024.0, pos.y / 1024.0], 0.0)
"#,
    );
    let msg = match result {
        Ok(_) => panic!("storage_image + texture at one (set, binding) must be rejected"),
        Err(e) => e.to_string(),
    };
    assert!(
        msg.contains("set") && msg.contains("binding"),
        "expected an incompatible-descriptor-kind error, got: {msg}"
    );
}

/// An array-of-tuples entry input (`pts: [](f32, f32)`) is split by the
/// SoA transform into a tuple-of-arrays whose component arrays share one
/// outer length. Two maps over those components are therefore the same
/// size class, so they must stay horizontally fused: one parallel kernel
/// driving both maps as lanes over the shared length, not two separate
/// dispatches. This is the SoA case that the size-class scheduling must
/// keep together.
///
/// `#[ignore]`d: array-of-tuples *entry inputs* do not lower today — the
/// input lowers to a storage buffer whose element type stays
/// `Tuple(2, [Array, Array])`, which has no static size. Un-ignore once
/// the SoA-input lowering and per-size-class scheduling land.
#[test]
#[ignore = "array-of-tuples entry input does not lower (element type Tuple(2) has no static size); blocks the SoA same-size-class case"]
fn soa_array_of_tuples_components_stay_one_size_class() {
    let lowered = crate::compile_thru_spirv(
        r#"
#[compute]
entry main(pts: [](f32, f32)) ([]f32, []f32) =
  (map(|p| p.0 + 1.0, pts), map(|p| p.1 + 2.0, pts))
"#,
    )
    .expect("array-of-tuples entry with two component maps must compile to one valid module");
    assert!(!lowered.spirv.is_empty());
}

/// A global `def` whose initializer contains a *function call*
/// referencing other globals used to error at SPIR-V emission with
/// "Unknown global: ELEV" when the synthesized global was then used
/// inside another function. Contrast: a global initialized by plain
/// constant arithmetic (e.g. `def g4 = base * 3.0`) compiled because
/// the entire chain got constant-folded — the failure shape only
/// surfaced when a function-call initializer prevented folding and
/// the SPIR-V backend had to actually emit the consumer body. The
/// fix: forward-declare and lower `program.constants` as zero-arg
/// functions, mirroring the existing handling of `program.functions`
/// (the WGSL backend handled them already; the SPIR-V backend did
/// not).
#[test]
fn function_call_initialized_global_compiles() {
    crate::compile_thru_spirv(
        r#"
def DIST: f32 = 5.0
def ELEV: f32 = 0.7
def rotm(a: f32) mat3f32 =
  @[[f32.cos(a), 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, f32.cos(a)]]
def eye: vec3f32 = rotm(ELEV) * @[0.0, 0.0, DIST]
def use_eye(p: vec3f32) vec3f32 = p + eye
#[fragment]
entry f() #[target(screen)] vec4f32 =
  let q = use_eye(@[1.0, 2.0, 3.0]) in @[q.x, q.y, q.z, 1.0]
"#,
    )
    .expect("global whose initializer calls a function should lower like any other global");
}

/// `filter` allocates a scratch storage binding (`filt_gather_b<n>`)
/// that the same compute stage writes into via the SOAC expansion.
/// `egir::from_tlc::convert_soac_filter` declares that scratch with
/// `role: Output` so the descriptor reports it as write-capable —
/// previously the role was `Intermediate`, which fell through to
/// `Access::ReadOnly` in `publish.rs` and produced an "unwritten
/// read-only intermediate" the host couldn't safely bind without
/// zero-initing.
#[test]
fn filter_scratch_binding_is_not_read_only() {
    use crate::pipeline_descriptor::{Access, Binding, BufferUsage};
    let lowered = compile_parallel(
        r#"
def keep(x: u32) bool = x != 0u32
#[compute]
entry filt(xs: []u32) ([]u32, [1]u32) =
  let ys = filter(keep, xs) in
  (ys, [u32.i32(length(ys))])
"#,
    );
    let bufs = compute_storage_buffers(&lowered.pipeline, "filt");
    let intermediates: Vec<&Binding> = bufs
        .iter()
        .filter(|b| {
            matches!(
                b,
                Binding::StorageBuffer {
                    usage: BufferUsage::Intermediate,
                    ..
                }
            )
        })
        .collect();
    assert!(
        !intermediates.is_empty(),
        "filter pipeline should declare at least one scratch intermediate: {bufs:?}"
    );
    for b in intermediates {
        if let Binding::StorageBuffer { access, name, .. } = b {
            assert!(
                !matches!(access, Access::ReadOnly),
                "scratch intermediate `{name}` is host-allocated and shader-written; \
                 must not surface as read_only: {b:?}"
            );
        }
    }
}

/// In-place write to a readwrite storage buffer that's then returned
/// by the entry: a shape SPIR-V can't yet lay out directly. Two
/// pieces keep the failure graceful instead of a `create_storage_buffer`
/// panic:
///   1. `types::canonical_storage_buffer_ty` strips `Unique<_>` and
///      top-level `Existential<_>` at the EGIR `EntryOutput`
///      construction sites, so `*[]T` reaches the backend as a
///      concrete runtime array.
///   2. `spirv::verify_buffer_layouts` rejects any storage-bound
///      type whose post-`array_elem` shape has no static size as a
///      structured error before backend emission — the tripwire for
///      any construction site that bypasses #1.
///
/// What this test pins: compilation surfaces the actionable
/// `realize_outputs` diagnostic ("runtime-sized array … wrap the
/// producer in a `map`") for this source shape. If we ever teach
/// the compiler to compile `*[]T with [i] = v` returns directly,
/// flip the assertion to expect clean success.
#[test]
fn inplace_write_to_returned_readwrite_storage_errors_gracefully() {
    let result = crate::compile_thru_spirv(
        r#"
#[compute]
entry tick(#[storage(set=2, binding=0, access=readwrite)] buf: *[]u32) *[]u32 =
  buf with [0] = 42u32
"#,
    );
    let err = match result {
        Ok(_) => panic!("compilation should still surface a graceful unsupported-shape error"),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("runtime-sized") && msg.contains("map"),
        "expected actionable 'wrap in map' diagnostic, got: {msg}"
    );
}

/// A consuming `*[]T` map's `Project` and the carried buffer it
/// drives must both carry the input view's type, not the TLC-
/// default `Composite[Variable, NoRegion]`. Otherwise the SPIR-V
/// backend tries to lower a Composite array with a runtime size
/// (the input view's runtime length) and panics. Wired in
/// `egir::from_tlc::convert_soac_map` (`InputBuffer`-aware project
/// type, mirroring `convert_soac_scan`) and in `egir::soac_expand`
/// (`emit_write_element` takes the post-decision carried type).
/// Structural records lower to `OpTypeStruct` in SPIR-V (via the
/// alias to `draw_args`). Member offsets get added when the record is
/// the leaf of a runtime-sized storage buffer, mirroring the tuple
/// path. Two shapes exercised: returning a record from a fragment
/// entry, and scattering records into a `*[]point` storage buffer.
#[test]
fn structural_record_lowers_through_spirv() {
    crate::compile_thru_spirv(
        r#"
type draw_args = {x: f32, y: f32, z: f32, w: f32}
#[fragment]
entry frag(#[uniform(set=1, binding=0)] iTime: f32) #[target(screen)] draw_args =
  {x = iTime, y = 0.0, z = 0.0, w = 1.0}
"#,
    )
    .expect("fragment returning a structural record should lower");

    crate::compile_thru_spirv(
        r#"
type point = {x: f32, y: f32}
#[compute]
entry e(#[storage(set=2, binding=1, access=write)] o: *[]point) () =
  let _ = scatter(o, [0i32], [{x = 1.0, y = 2.0}]) in ()
"#,
    )
    .expect("compute scatter into *[]record should lower");
}

/// A record-of-runtime-arrays alias (`world`) passed as a function PARAM.
/// The array fields' variant/region slots must be region-polymorphic across
/// the call boundary (the alias body's placeholders freshen per use), so
/// `world` unifies with a `{ points = view, items = view }` argument.
/// (PR7 repro pr7_2a.)
#[test]
fn record_of_arrays_param_across_boundary_compiles() {
    crate::compile_thru_spirv(
        r#"
open f32
type world = { points: []vec2f32, items: []vec4f32 }

def use_world(w: world, dom: []u32) ([]vec2f32, []vec4f32) =
  let p = map(|i| let j = i32(i) in w.points[j] + @[1.0, 1.0], dom)
  let it = map(|i| let j = i32(i) in w.items[j] * @[2.0, 2.0, 2.0, 2.0], dom) in
  (p, it)

#[compute]
entry step(dom: []u32, points_in: []vec2f32, items_in: []vec4f32)
  ([]vec2f32, []vec4f32) =
  use_world({ points = points_in, items = items_in }, dom)
"#,
    )
    .expect("record-of-arrays as a function param should compile");
}

/// Construct a record-of-runtime-arrays from `map` outputs and RETURN it.
/// The declared `world` return must unify with the body's concrete
/// `composite`/`no_region` map-result arrays. (PR7 repro pr7_2b.)
#[test]
fn record_of_arrays_construct_and_return_compiles() {
    crate::compile_thru_spirv(
        r#"
open f32
type world = { points: []vec2f32, items: []vec4f32 }

def make_world(dom: []u32) world =
  let p = map(|i| @[f32(i), f32(i)], dom)
  let it = map(|i| @[f32(i), 0.0, 0.0, 1.0], dom) in
  { points = p, items = it }

#[compute]
entry step(dom: []u32) ([]vec2f32, []vec4f32) =
  let w = make_world(dom) in
  (w.points, w.items)
"#,
    )
    .expect("constructing and returning a record-of-arrays should compile");
}

/// A `map` output (`occ`) that is BOTH fed to another map (`occ[j%4]`) AND
/// returned. `occ` must be materialized to storage rather than left an
/// in-register runtime-sized Composite array. Currently panics in SPIR-V
/// lowering ("Composite variant unsized arrays not supported"); goes green
/// when the `lift_gathers` output-position materialization lands. (PR7 repro
/// pr7_3c — the #2 showstopper.)
#[test]
fn map_output_fed_and_returned_compiles() {
    crate::compile_thru_spirv(
        r#"
#[compute]
entry frame(occ_dom: []u32, sett_dom: []u32) ([]u32, []u32) =
  let occ = map(|i| i + 7u32, occ_dom)
  let setts = map(|i| let j = i32(i) in i + occ[j % 4], sett_dom) in
  (occ, setts)
"#,
    )
    .expect("a map output both consumed and returned should compile");
}

/// Control for pr7_3c: the same producer→consumer dataflow, but only the
/// dependent array (`setts`) is returned — `occ` is consumed solely via
/// dynamic index, so `lift_gathers` materializes it and this compiles today.
/// Bounds the pr7_3c trigger to the returned-AND-fed shape. (PR7 repro pr7_3b.)
#[test]
fn map_output_fed_but_only_dependent_returned_compiles() {
    crate::compile_thru_spirv(
        r#"
def build(occ_dom: []u32, sett_dom: []u32) []u32 =
  let occ = map(|i| i + 7u32, occ_dom)
  let setts = map(|i| let j = i32(i) in i + occ[j % 4], sett_dom) in
  setts

#[compute]
entry frame(occ_dom: []u32, sett_dom: []u32) []u32 =
  build(occ_dom, sett_dom)
"#,
    )
    .expect("consuming a map output internally (not returned) should compile");
}

/// pr7_3e: a `map` output carried in a RECORD FIELD (`w.points`), then both
/// read by a downstream map (through the whole-record capture) AND returned.
/// Output realization retargets the producer `p` to the output view and the
/// record is built holding that view (`tuple(view)`), but the drift is not
/// propagated: the capturing lambda's parameter keeps its stale
/// `Record([Composite])` type, and its internal `w.points` projection lowers a
/// runtime-sized Composite array — panicking at `types_lowering.rs:139`.
/// Byte-identical dataflow with plain arrays (pr7_3c) already compiles; the
/// record indirection is the whole delta. Central to the PR9 `world` value.
#[test]
fn map_output_in_record_field_fed_and_returned_compiles() {
    crate::compile_thru_spirv(
        r#"
open f32

type world = { points: []vec2f32 }

def build_geom(w: world, tdom: []u32) []vec4f32 =
  map(|i| let j = i32(i) in @[w.points[j % 8].x, 0.0, w.points[j % 8].y, 1.0], tdom)

#[compute]
entry step(pdom: []u32, tdom: []u32, points_in: []vec2f32)
  ([]vec2f32, []vec4f32) =
  let p = map(|i| let j = i32(i) in points_in[j] + @[1.0, 0.0], pdom)
  let w = { points = p }
  let geom = build_geom(w, tdom) in
  (w.points, geom)
"#,
    )
    .expect("a map output in a record field, both fed and returned, should compile");
}

#[test]
fn clear_then_scatter_on_consuming_write_storage_compiles() {
    crate::compile_thru_spirv(
        r#"
#[compute]
entry e(#[storage(set=2, binding=1, access=write)] fb: *[]vec4f32) () =
  let cleared = map(|_p:vec4f32| @[0.0, 0.0, 0.0, 1.0], fb) in
  let _ = scatter(cleared, [0i32, 1i32], [@[1.0,1.0,1.0,1.0], @[1.0,1.0,1.0,1.0]]) in ()
"#,
    )
    .expect("clear-then-scatter on consuming `*[]T` write storage should compile end-to-end");
}

/// A compute entry may both *return* a Screma result and *consume*
/// it as a downstream side-effect's array input — here `new_pos` is
/// the entry's output and also the per-element input the scatter
/// envelope reads. The rewrite in
/// `egir::realize_outputs::dispatch::rewrite_sibling_index_consumers`
/// retargets `new_pos` to the entry-output view and substitutes
/// `source → view` in the scatter's input-region operand.
#[test]
fn compute_entry_returns_screma_result_and_scatters_through_it() {
    crate::compile_thru_spirv(
        r#"
def N:i32 = 8
def RES:i32 = 8

#[compute]
entry sim(#[storage(set=2, binding=0, access=read)] prev: []vec4f32,
          #[storage(set=2, binding=1, access=write)] fb: []vec4f32) []vec4f32 =
  let new_pos = map(|x:vec4f32| @[x.x + 1.0, x.y + 1.0, x.z, x.w], prev) in
  let idxs = map(|p:vec4f32| i32.f32(p.y) * RES + i32.f32(p.x), new_pos) in
  let vals = map(|p:vec4f32| @[1.0, 1.0, 1.0, 1.0], new_pos) in
  let _ = scatter(fb, idxs, vals) in
  new_pos
"#,
    )
    .expect("returning a Screma result while a downstream scatter consumes it should compile");
}

// ============================================================================
// Module-system spec-gap tests (ignored).
//
// These tests pin the behavior that `SPECIFICATION.md` describes but the
// current compiler does NOT yet implement. They are `#[ignore]`d so the
// suite stays green; running `cargo test -- --ignored` reveals the gap.
//
// When the implementation catches up:
//   1. Un-ignore the relevant test(s).
//   2. Remove the matching "Implementation discrepancy" callout in
//      `SPECIFICATION.md` (search for "DISCREPANCY:" — there are two
//      callouts, one in §Declaration Modifiers and one in §Referencing
//      Other Files).
// ============================================================================

/// SPEC (`SPECIFICATION.md`, "Declaration Modifiers"):
///   `local dec` binds the names defined by `dec` in the current scope
///   but hides them from users of the enclosing module.
///
/// CURRENT IMPL: no `local` keyword exists. Parser errors with
///   "Expected declaration, got Identifier(\"local\")".
///
/// IMPLEMENTATION OPTIONS (smallest → largest scope):
///   A. Parser stub. Add `Local` keyword in `lexer/mod.rs`, a
///      `Declaration::Local(Box<Declaration>)` AST variant, and parse it
///      in `parser.rs`. Treat as equivalent to non-local everywhere else.
///      Reserves the keyword and unblocks libraries that want to write
///      `local open` for future-compat; does not yet hide anything.
///   B. Filter at the user-module boundary. In
///      `module_manager::elaborate_module_body`, drop `Local(...)` decls
///      when building the exported `items` list. Hides locals from users
///      of `module foo = { ... }` bodies.
///   C. Filter at the file-import boundary. `import "lib.wyn"` is
///      currently literal-inlining (`resolve_imports::run`), so by the
///      time `resolve_opens` runs there's no remaining "this came from
///      lib.wyn" boundary. Two paths: (i) run `resolve_opens` per-file
///      before inlining and strip `Local(...)` from the inlined result,
///      or (ii) inject begin/end-file scope markers around inlined
///      decls and teach the resolver to pop opens at end markers.
#[test]
#[ignore = "SPEC: `local <dec>` / `local open` not implemented; see DISCREPANCY in SPECIFICATION.md"]
fn local_open_parses_per_spec() {
    let src = r#"
        local open f32
        def f (x: f32) f32 = clamp(x, 0.0f32, 1.0f32)
    "#;
    compile_to_ssa(src);
}

/// SPEC (`SPECIFICATION.md`, "Referencing Other Files"):
///   Qualified imports: `module M = import "file"` creates a module
///   whose members are the file's top-level non-local decls, accessed
///   as `M.foo`.
///
/// CURRENT IMPL: the parser accepts `module M = import "..."` (because
/// `parse_module_expression` handles the `Import` form), but
/// `module_manager::elaborate_module_body` returns
/// "Unsupported module expression type" — it has no case for
/// `ModuleExpression::Import`.
///
/// IMPLEMENTATION OPTIONS:
///   A. In `elaborate_module_body`, when seeing
///      `ModuleExpression::Import(path)`, resolve `path` to a file,
///      parse it, and recursively elaborate its top-level non-local
///      decls as if they were the body of a synthetic struct. Requires
///      filesystem access at elaboration time — the manager would need
///      a `base_dir` thread-through (which `resolve_imports::run`
///      already does for its case).
///   B. Desugar at parse time: rewrite `module M = import "path"` into
///      `module M = { <inlined parsed decls> }` in a pass that runs
///      after `resolve_imports` but before `elaborate_modules`.
#[test]
#[ignore = "SPEC: `module M = import \"...\"` not implemented; see DISCREPANCY in SPECIFICATION.md"]
fn qualified_module_import_per_spec() {
    // For a self-contained test, we'd usually point at a real file via
    // `import`. Here we just exercise the elaboration path — when this
    // form is supported, the test should be expanded to write a temp
    // file and reference it.
    let src = r#"
        module M = import "nonexistent_for_now"
        def use_it: f32 = M.something
    "#;
    compile_to_ssa(src);
}

/// SPEC (`SPECIFICATION.md`, "Referencing Other Files"):
///   A plain `import "file"` is equivalent to `local open import "file"`
///   — it pulls in another file's exports without re-exporting them.
///
/// CURRENT IMPL: `resolve_imports::run` literally inlines the imported
/// file's decls into the importer's top-level declaration list. Every
/// non-local decl AND every `open` from the imported file becomes a
/// top-level decl in the importer's program. This is the opposite of
/// "without re-exporting them" — closer to the spec's
/// `open import "file"` semantics (re-export).
///
/// The dedupe fix in [`crate::resolve_opens`] for two identical
/// `open M` entries papers over the most-visible symptom of this
/// mismatch (importer + library both doing `open f32` no longer
/// ambiguates), but does not restore the spec's hiding semantics.
///
/// IMPLEMENTATION OPTIONS: see the C-variants documented on
/// [`local_open_parses_per_spec`] — same machinery.
#[test]
#[ignore = "SPEC: plain `import \"file\"` should not re-export; see DISCREPANCY in SPECIFICATION.md"]
fn bare_import_does_not_reexport_per_spec() {
    // Sketch only — exercising this properly needs a real on-disk
    // import. The intended assertion: after `import "lib"`, a name
    // that `lib` opened (e.g. `f32.clamp` brought in by lib's
    // `open f32`) is NOT visible bare in the importer; the importer
    // must do its own `open f32` to see it.
    let src = r#"
        import "lib_that_opens_f32"
        def f (x: f32) f32 = clamp(x, 0.0f32, 1.0f32)
    "#;
    compile_to_ssa(src);
}

/// SPEC (`SPECIFICATION.md`, "Declaration Modifiers"):
///   `local dec` is general — `dec` can be any declaration form,
///   including `import "file"`. `local import "file"` is the explicit
///   spelling of the same thing plain `import "file"` is *already*
///   defined to mean (`local open import "file"`); writing it
///   explicitly should still parse.
///
/// More generally: the parser should accept `local` in front of every
/// declaration kind — `local def`, `local type`, `local module`,
/// `local open`, `local import`, etc. — and each should hide its
/// bound name(s) from users of the enclosing module while keeping
/// them visible to siblings.
///
/// CURRENT IMPL: no `local` keyword; all of these are parse errors.
///
/// IMPLEMENTATION OPTIONS: see `local_open_parses_per_spec`. The
/// parser-stub option (A) covers every `local <dec>` form uniformly
/// by wrapping the inner decl in `Declaration::Local(Box<_>)`.
#[test]
#[ignore = "SPEC: `local <dec>` (including `local import`) not implemented; see DISCREPANCY in SPECIFICATION.md"]
fn local_import_parses_per_spec() {
    let src = r#"
        local import "lib_that_opens_f32"
        def f (x: f32) f32 = clamp(x, 0.0f32, 1.0f32)
    "#;
    compile_to_ssa(src);
}

/// `f32.from_bits` / `f32.to_bits` are per-type members whose schemes
/// come from the prelude `float` signature but whose lowering must be
/// published in the builtin catalog under the member names — the module
/// defs are sig-only to the backend, like the other per-type
/// conversions. Surfaced by `prelude/math.wyn`'s `fastmath.sqrt`
/// (exponent-halving bit trick): without catalog entries the call
/// survives to SPIR-V lowering and fails with "Unknown function:
/// f32.to_bits".
#[test]
fn f32_bit_reinterpret_members_lower_through_spirv() {
    crate::compile_thru_spirv(
        r#"
def fsqrt(x: f32) f32 =
  f32.from_bits(0x1fbd1df5u32 + (f32.to_bits(x) >> 1u32))

#[compute]
entry e() [1]f32 = [fsqrt(4.0f32)]
"#,
    )
    .expect("f32.from_bits/to_bits should lower to OpBitcast");
}

/// A `map` compute entry whose lambda runs a loop containing an
/// `image_load` (storage image) and THEN does a `texture_load` (sampled
/// texture) panics during EGIR elaboration:
///
///   elaborate.rs: "FuncParam/BlockParam NodeId(..) should have been
///   pre-populated in elaborated map"
///
/// Both operations and their order are load-bearing: texture_load
/// before the loop, or an image_load in place of the texture_load,
/// compiles fine. This is the light-pass shape for driving lib/gtao.wyn
/// from the map/iota idiom (loop over shadow taps, then sample the AO
/// result).
#[test]
fn texture_load_after_image_load_loop_in_map_lambda() {
    crate::compile_thru_spirv(
        r#"
open f32
resource src: image2d { format = r32float  size = window  usages = [storage_read, storage_write] }
resource tex: image2d { format = r32float  size = window  usages = [storage_write, sampled] }

#[compute]
entry g6(pxl: []u32,
         #[view(src, storage_read)]  s: storage_image,
         #[view(tex, sampled)]       tx: texture2d)
  []u32 =
  map(|t|
    let i = i32(t)  let x = i % 1280  let y = i / 1280
    let acc =
      loop acc = 0.0 for k < 8 do
        let v = image_load(s, @[x, y]).x in
        acc + v
    let base = texture_load(tx, @[x, y], 0).x in
    if acc + base > 0.0 then 1u32 else 0u32
    , pxl)
"#,
    )
    .expect("texture_load after an image_load loop inside a map lambda should lower");
}

/// A runtime filter whose host entry also binds a storage image schedules the
/// filter scan over a fixed worker grid. The filter stages are clones of the
/// host entry, so they inherit its storage-image input; the per-texel dispatch
/// upgrade must not seize their domains and wire the scan to the unrelated
/// image extent. Because the scan phase is scheduled `Explicit(Fixed)`, no
/// domain inference can touch it.
#[test]
fn filter_scan_domain_ignores_storage_image_entry_binding() {
    use crate::egir::parallelize::schedule::KernelDomain;
    use crate::pipeline_descriptor::{DispatchSize, Pipeline};

    let src = r#"
resource occ: image2d { format = r32float  size = 256x256  usages = [storage_read, storage_write] }

#[compute]
entry cull(xs: []u32,
           #[view(occ, storage_read)] od: storage_image) ?k. [k]u32 =
  filter(|x|
    let d = image_load(od, @[i32(x % 256u32), i32(x / 256u32)]).x in
    d < 0.5
    , xs)
"#;
    let converted = crate::compile_thru_ssa(src).expect("image-reading filter reaches SSA");
    let scan = converted
        .kernel_schedule
        .phases()
        .find(|phase| phase.entry_point == "cull_filter_scan")
        .expect("scan phase scheduled");
    assert!(
        matches!(
            scan.domain,
            KernelDomain::Fixed {
                x: crate::egir::parallelize::FILTER_SCAN_GROUPS,
                y: 1,
                z: 1
            }
        ),
        "scan domain must be the fixed worker grid, not the image extent, got {:?}",
        scan.domain
    );
    assert_eq!(
        scan.workgroup_size,
        (crate::egir::parallelize::REDUCE_PHASE1_WIDTH, 1, 1)
    );

    let lowered = crate::compile_thru_spirv(src).expect("image-reading filter emits SPIR-V");
    let stage = lowered
        .pipeline
        .pipelines
        .iter()
        .find_map(|pipeline| match pipeline {
            Pipeline::Compute(compute) => {
                compute.stages.iter().find(|stage| stage.entry_point == "cull_filter_scan")
            }
            _ => None,
        })
        .expect("scan stage published");
    assert!(
        matches!(stage.dispatch_size, DispatchSize::Fixed { x: 4, y: 1, z: 1, .. }),
        "scan dispatch must publish the fixed worker grid, got {:?}",
        stage.dispatch_size
    );
}

/// A record-typed `#[uniform]` param must reach SSA as ONE EntryInput
/// carrying the Record type and the uniform binding — no TLC pass may
/// flatten it into per-field params (the SPIR-V backend builds the
/// uniform block from exactly this input).
#[test]
fn record_uniform_reaches_ssa_as_single_input() {
    let lowered = crate::compile_thru_ssa(
        r#"
type block = { radius: f32, tint: vec2f32, center: vec2f32 }
#[compute]
entry e(xs: []u32, #[uniform(set=1, binding=0)] c: block) []u32 =
  map(|x| x + u32(c.radius + c.tint.x + c.center.y), xs)
"#,
    )
    .expect("record uniform reaches SSA");
    let entry = lowered.ssa.entry_points.iter().find(|e| e.name == "e").expect("entry `e`");
    let c: Vec<_> = entry.inputs.iter().filter(|i| i.name == "c").collect();
    assert_eq!(c.len(), 1, "record uniform must stay one input, got {:?}", c);
    assert!(
        c[0].uniform_binding.is_some(),
        "input `c` must carry the uniform binding"
    );
    assert!(
        matches!(
            &c[0].ty,
            polytype::Type::Constructed(crate::ast::TypeName::Record(_), _)
        ),
        "input `c` must keep its Record type, got {:?}",
        c[0].ty
    );
}

/// The record uniform lowers to a Block-decorated struct whose members
/// carry std140 Offsets, and the Uniform-class variable points at that
/// laid-out struct (never at the shared plain record struct).
#[test]
fn record_uniform_emits_std140_offset_decorated_block() {
    use std::collections::HashMap;
    use wspirv::binary::parse_words;
    use wspirv::dr::{Loader, Operand};
    use wspirv::spirv::{Decoration, Op, StorageClass};

    let lowered = crate::compile_thru_spirv(
        r#"
type block = { radius: f32, tint: vec2f32, center: vec2f32 }
#[compute]
entry e(xs: []u32, #[uniform(set=1, binding=0)] c: block) []u32 =
  map(|x| x + u32(c.radius + c.tint.x + c.center.y), xs)
"#,
    )
    .expect("record uniform emits SPIR-V");

    let mut loader = Loader::new();
    parse_words(&lowered.spirv, &mut loader).expect("parse spirv");
    let module = loader.module();

    // Member offsets per struct id, and the set of Block-decorated ids.
    let mut offsets: HashMap<u32, Vec<(u32, u32)>> = HashMap::new();
    let mut blocks: Vec<u32> = Vec::new();
    for inst in &module.annotations {
        match inst.class.opcode {
            Op::MemberDecorate => {
                let id = inst.operands[0].unwrap_id_ref();
                let member = inst.operands[1].unwrap_literal_bit32();
                if inst.operands[2] == Operand::Decoration(Decoration::Offset) {
                    let off = inst.operands[3].unwrap_literal_bit32();
                    offsets.entry(id).or_default().push((member, off));
                }
            }
            Op::Decorate if inst.operands[1] == Operand::Decoration(Decoration::Block) => {
                blocks.push(inst.operands[0].unwrap_id_ref());
            }
            _ => {}
        }
    }

    // A Block struct with the record's std140 offsets exists...
    let block_id = blocks
        .iter()
        .copied()
        .find(|id| {
            let mut o = offsets.get(id).cloned().unwrap_or_default();
            o.sort_unstable();
            o == vec![(0, 0), (1, 8), (2, 16)]
        })
        .expect("a Block struct with member offsets [0, 8, 16] must exist");

    // ...and a Uniform-storage variable points at it (via its pointer type).
    let pointee_of: HashMap<u32, u32> = module
        .types_global_values
        .iter()
        .filter(|i| i.class.opcode == Op::TypePointer)
        .filter_map(|i| Some((i.result_id?, i.operands[1].unwrap_id_ref())))
        .collect();
    let uniform_points_at_block = module.types_global_values.iter().any(|i| {
        i.class.opcode == Op::Variable
            && i.operands.first() == Some(&Operand::StorageClass(StorageClass::Uniform))
            && i.result_type.and_then(|t| pointee_of.get(&t)).copied() == Some(block_id)
    });
    assert!(
        uniform_points_at_block,
        "a Uniform-class OpVariable must point at the laid-out block struct"
    );
}

/// Two stages sharing the same record uniform (set, binding) compile —
/// the interface-block cache prevents double Offset decoration and the
/// cross-entry consistency check accepts the matching types.
#[test]
fn record_uniform_shared_across_stages_compiles() {
    crate::compile_thru_spirv(
        r#"
type block = { radius: f32, tint: vec2f32 }

#[compute]
entry step(xs: []u32, #[uniform(set=1, binding=0)] c: block) []u32 =
  map(|x| x + u32(c.radius), xs)

#[vertex]
entry vertex_main(#[builtin(vertex_index)] vid: i32) #[builtin(position)] vec4f32 =
  let verts = [@[-1.0, -1.0, 0.0, 1.0],
               @[3.0, -1.0, 0.0, 1.0],
               @[-1.0, 3.0, 0.0, 1.0]] in
  verts[vid]

#[fragment]
entry fragment_main(#[builtin(position)] pos: vec4f32,
                    #[uniform(set=1, binding=0)] c: block)
  #[target(screen)] vec4f32 =
  @[c.tint.x, c.tint.y, c.radius, 1.0]
"#,
    )
    .expect("stages sharing a record uniform block should lower");
}

/// Storage-buffer struct elements get std430 member offsets and an
/// aligned ArrayStride. Regression: the offsets used to be a tight
/// unaligned sum ({f32, vec2f32} → [0, 4], stride 12), which is
/// std430-nonconformant AND disagrees with naga's layout of the same
/// struct on the WGSL path — silent data corruption for records with
/// vector members.
#[test]
fn storage_record_elements_get_std430_offsets_and_stride() {
    use std::collections::HashMap;
    use wspirv::binary::parse_words;
    use wspirv::dr::{Loader, Operand};
    use wspirv::spirv::{Decoration, Op};

    let lowered = crate::compile_thru_spirv(
        r#"
type point = { w: f32, uv: vec2f32 }
#[compute]
entry e(#[storage(set=2, binding=1, access=write)] o: *[]point) () =
  let _ = scatter(o, [0i32], [{w = 1.0, uv = @[2.0, 3.0]}]) in ()
"#,
    )
    .expect("record storage element lowers");

    let mut loader = Loader::new();
    parse_words(&lowered.spirv, &mut loader).expect("parse spirv");
    let module = loader.module();

    let mut offsets: HashMap<u32, Vec<(u32, u32)>> = HashMap::new();
    let mut strides: Vec<u32> = Vec::new();
    for inst in &module.annotations {
        match inst.class.opcode {
            Op::MemberDecorate if inst.operands[2] == Operand::Decoration(Decoration::Offset) => {
                offsets.entry(inst.operands[0].unwrap_id_ref()).or_default().push((
                    inst.operands[1].unwrap_literal_bit32(),
                    inst.operands[3].unwrap_literal_bit32(),
                ));
            }
            Op::Decorate if inst.operands[1] == Operand::Decoration(Decoration::ArrayStride) => {
                strides.push(inst.operands[2].unwrap_literal_bit32());
            }
            _ => {}
        }
    }

    assert!(
        offsets.values().any(|o| {
            let mut o = o.clone();
            o.sort_unstable();
            o == vec![(0, 0), (1, 8)]
        }),
        "the point struct must carry std430 offsets [0, 8], got {offsets:?}"
    );
    assert!(
        strides.contains(&16),
        "the runtime array of point must have ArrayStride 16, got {strides:?}"
    );
}

/// The descriptor publishes a uniform block's std140 size and member
/// layout: record fields under their source names, tuples as `f0..`,
/// bare scalars/vectors as a single member at offset 0 — the same
/// host contract push constants have.
#[test]
fn uniform_bindings_publish_size_and_members() {
    use crate::pipeline_descriptor::{Binding, Pipeline};

    let lowered = crate::compile_thru_spirv(
        r#"
type block = { radius: f32, tint: vec2f32, center: vec2f32 }
#[compute]
entry e(xs: []u32,
        #[uniform(set=1, binding=0)] c: block,
        #[uniform(set=1, binding=1)] pair: (f32, vec2f32),
        #[uniform(set=1, binding=2)] t: f32) []u32 =
  map(|x| x + u32(c.radius + c.tint.x + c.center.y + pair.0 + pair.1.x + t), xs)
"#,
    )
    .expect("uniform shapes compile");

    let uniforms: Vec<_> = lowered
        .pipeline
        .pipelines
        .iter()
        .flat_map(|p| match p {
            Pipeline::Compute(cp) => cp.bindings.iter(),
            Pipeline::Graphics(gp) => gp.bindings.iter(),
        })
        .filter_map(|b| match b {
            Binding::Uniform {
                binding,
                name,
                size,
                members,
                ..
            } => Some((*binding, name.clone(), *size, members.clone())),
            _ => None,
        })
        .collect();

    let (_, _, size, members) = uniforms.iter().find(|(b, ..)| *b == 0).expect("record uniform");
    assert_eq!(*size, 32);
    let shape: Vec<_> = members.iter().map(|m| (m.name.as_str(), m.offset, m.size)).collect();
    assert_eq!(shape, vec![("radius", 0, 4), ("tint", 8, 8), ("center", 16, 8)]);

    let (_, _, size, members) = uniforms.iter().find(|(b, ..)| *b == 1).expect("tuple uniform");
    assert_eq!(*size, 16);
    let shape: Vec<_> = members.iter().map(|m| (m.name.as_str(), m.offset, m.size)).collect();
    assert_eq!(shape, vec![("f0", 0, 4), ("f1", 8, 8)]);

    let (_, _, size, members) = uniforms.iter().find(|(b, ..)| *b == 2).expect("scalar uniform");
    assert_eq!(*size, 16);
    assert_eq!(members.len(), 1);
    assert_eq!((members[0].offset, members[0].size), (0, 4));
}
