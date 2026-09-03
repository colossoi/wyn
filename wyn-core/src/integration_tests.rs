#![cfg(test)]
//! Integration tests for the full compilation pipeline.
//!
//! These tests verify that source code compiles correctly through all stages:
//! parse → desugar → resolve → type_check → alias_check → TLC → monomorphize → SSA
//!
//! All tests include entry points to ensure monomorphization can find reachable code.

use crate::ast;
use crate::ast_type_holes;
use crate::builtins;
use crate::compile_thru_frontend;
use crate::compile_thru_frontend_with_options;
use crate::compile_thru_spirv;
use crate::compile_thru_spirv_serial;
use crate::compile_thru_ssa;
use crate::compile_thru_tlc;
use crate::egir;
use crate::egir::soac::screma;
use crate::error;
use crate::interface;
use crate::lower_egir_to_ssa;
use crate::lower_ssa_to_spirv;
use crate::lower_ssa_to_wgsl;
use crate::lower_ssa_to_wgsl_with_pipeline;
use crate::op;
use crate::pipeline_descriptor;
use crate::ssa;
use crate::ssa::types::Program;
use crate::test_pipeline;
use crate::tlc;
use crate::tlc::extract_lambda_params;
use crate::tlc::VarRef;
use crate::to_egraph;
use crate::CodegenTarget;
use crate::CompilerOptions;
use crate::Lowered;
use crate::LoweringProfile;
use crate::PipelineTopologyPolicy;
use crate::ResourceAccess;
use crate::SchedulePolicy;
use crate::SymbolTable;

#[test]
fn graphics_vocabulary_is_absent_without_opt_in() {
    let error = compile_thru_frontend_with_options(
        "entry main() i32 = direct_draw(3u32, 1u32)",
        CompilerOptions::default(),
    )
    .expect_err("graphics builtin must not resolve without opt-in");
    assert!(
        error.to_string().contains("Undefined variable 'direct_draw'"),
        "unexpected diagnostic: {error}"
    );
}

#[test]
fn disabled_graphics_names_can_be_defined_by_user_code() {
    compile_thru_frontend_with_options(
        r#"
type render_target = i32
def direct_draw(x: render_target) render_target = x
entry main() render_target = direct_draw(7)
"#,
        CompilerOptions::default(),
    )
    .expect("disabled graphics spellings should remain ordinary user names");
}

#[test]
fn graphics_builtin_identity_can_still_be_shadowed_when_enabled() {
    compile_thru_frontend_with_options(
        r#"
def direct_draw(x: i32) i32 = x
entry main() i32 = direct_draw(7)
"#,
        CompilerOptions { graphics: true },
    )
    .expect("a user definition should shadow the enabled graphics builtin");
}

fn plan_direct(
    source: &str,
    target: CodegenTarget,
) -> Result<egir::parallelize::Planned, Box<dyn std::error::Error>> {
    let program = compile_thru_tlc(source)?;
    let program = tlc::infer_input_slice_bounds(program);
    let program = to_egraph(program)?;
    let program = egir::reify_soacs(program);
    let program = egir::optimize_semantic_operations(program)?;
    let topology = PipelineTopologyPolicy::AuthoredOnly;
    let program = egir::apply_pipeline_topology_policy(program, topology);
    let program = egir::plan_logical_resources_with_policy(program, topology)?;
    Ok(egir::plan(
        program,
        LoweringProfile::with_topology(target, SchedulePolicy::Serial, topology),
    )?)
}

fn run_with_large_stack(test: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(test)
        .expect("spawn test thread")
        .join()
        .expect("test thread panicked");
}

#[test]
fn playground_da_rasterizer_preserves_aliased_output_producers() {
    run_with_large_stack(|| {
        let source = format!(
            "{}\n{}",
            include_str!("../../scripts/playground_image_header.wyn"),
            include_str!("../../testfiles/playground/da_rasterizer.wyn"),
        );
        let ssa = compile_thru_ssa(&source)
            .expect("indexed fusion must retain producers observed through aliased output routes");
        lower_ssa_to_wgsl(ssa).expect("da_rasterizer lowers to WGSL");
    });
}

#[test]
fn playground_nested_fragment_loops_preserve_side_effect_producers() {
    run_with_large_stack(|| {
        let source = include_str!("../../testfiles/regressions/playground_nested_fragment_loops.wyn");
        let ssa = lower_semantic_egir(
            compile_to_semantic_egir(source),
            LoweringProfile::new(CodegenTarget::Wgsl, SchedulePolicy::Parallel),
        );
        lower_ssa_to_wgsl(ssa).expect("nested fragment loops lower to WGSL");
    });
}

#[test]
fn playground_nested_loop_helper_binds_all_wgsl_values() {
    run_with_large_stack(|| {
        let source = include_str!("../../testfiles/regressions/playground_nested_loop_helper.wyn");
        let ssa = lower_semantic_egir(
            compile_to_semantic_egir(source),
            LoweringProfile::new(CodegenTarget::Wgsl, SchedulePolicy::Parallel),
        );
        lower_ssa_to_wgsl(ssa).expect("nested loop helper binds every referenced WGSL value");
    });
}

#[test]
fn direct_backends_emit_only_the_requested_graphics_stages() {
    run_with_large_stack(|| {
        let source = include_str!("../../testfiles/unified_triangle.wyn");
        for target in [CodegenTarget::Spirv, CodegenTarget::Wgsl] {
            let planned = plan_direct(source, target).expect("direct graphics plan");
            let ssa = lower_egir_to_ssa(planned).expect("direct graphics SSA");
            assert_eq!(ssa.entry_points.len(), 2, "one vertex and one fragment entry");
            assert!(
                ssa.entry_points.iter().all(|entry| !entry.execution_model.is_compute()),
                "direct graphics output must not contain compute prepasses"
            );
            assert!(
                ssa.global_context
                    .pipeline
                    .pipelines
                    .iter()
                    .all(|pipeline| { matches!(pipeline, pipeline_descriptor::Pipeline::Graphics(_)) }),
                "direct graphics output must publish only the requested graphics pipeline"
            );
            match target {
                CodegenTarget::Spirv => {
                    let lowered = lower_ssa_to_spirv(ssa).expect("direct SPIR-V lowering");
                    assert_naga_accepts_spirv(&lowered.spirv);
                }
                CodegenTarget::Wgsl => {
                    let wgsl = lower_ssa_to_wgsl(ssa).expect("direct WGSL lowering");
                    assert_eq!(wgsl.matches("@vertex").count(), 1);
                    assert_eq!(wgsl.matches("@fragment").count(), 1);
                    assert!(!wgsl.contains("@compute"));
                }
                CodegenTarget::Portable => unreachable!("test selects concrete backends"),
            }
        }
    });
}

#[test]
fn direct_output_keeps_fragment_local_reduce_in_the_authored_stage() {
    run_with_large_stack(|| {
        let source = format!(
            "{}\n{}",
            include_str!("../../scripts/playground_image_header.wyn"),
            include_str!("../../testfiles/playground/ripples.wyn"),
        );
        let planned = plan_direct(&source, CodegenTarget::Wgsl)
            .expect("fragment-local reduction should remain serial in direct WGSL");
        let ssa = lower_egir_to_ssa(planned).expect("direct graphics SSA");
        assert_eq!(ssa.entry_points.len(), 2, "one vertex and one fragment entry");
        assert!(
            ssa.entry_points.iter().all(|entry| !entry.execution_model.is_compute()),
            "direct graphics output must not contain a reduction prepass"
        );
        let wgsl = lower_ssa_to_wgsl(ssa).expect("direct WGSL lowering");
        assert_eq!(wgsl.matches("@vertex").count(), 1);
        assert_eq!(wgsl.matches("@fragment").count(), 1);
        assert!(!wgsl.contains("@compute"));
    });
}

fn assert_direct_computed_result_descriptor(descriptor: &pipeline_descriptor::PipelineDescriptor) {
    use pipeline_descriptor::{Binding, BufferUsage, FramePassKind, Pipeline, ShaderStage};

    assert_eq!(
        descriptor.pipelines.len(),
        2,
        "one compute and one graphics pipeline"
    );
    let Pipeline::Compute(compute) = &descriptor.pipelines[0] else {
        panic!("the authored result producer must be the first pipeline")
    };
    assert_eq!(
        compute.stages.len(),
        1,
        "the authored compute pipeline has one stage"
    );
    assert_eq!(compute.stages[0].entry_point, "frame");
    assert!(
        !compute.stages[0].entry_point.contains("prepass_scalar"),
        "direct lowering must not publish a scalar prepass"
    );

    let result = descriptor
        .source_results
        .iter()
        .find(|result| result.entry == "frame" && result.result == 0)
        .expect("computed array keeps its authored result binding");
    assert_eq!(result.pipeline_index, 0);
    let result_binding = compute
        .bindings
        .iter()
        .position(|binding| {
            matches!(binding, Binding::StorageBuffer { set, binding, .. }
                if (*set, *binding) == (result.set, result.binding))
        })
        .expect("compute pipeline publishes the authored result binding");
    assert!(
        compute.stages[0].writes.contains(&result_binding),
        "compute stage writes the authored result binding"
    );

    let Pipeline::Graphics(graphics) = &descriptor.pipelines[1] else {
        panic!("the authored graphics pipeline must follow the compute pipeline")
    };
    let graphics_binding = graphics
        .bindings
        .iter()
        .position(|binding| {
            matches!(binding, Binding::StorageBuffer { set, binding, .. }
                if (*set, *binding) == (result.set, result.binding))
        })
        .expect("graphics pipeline reads the authored result binding");
    let fragment = graphics
        .stages
        .iter()
        .find(|stage| matches!(stage.stage, ShaderStage::Fragment))
        .expect("fragment stage");
    assert!(fragment.reads.contains(&graphics_binding));

    assert!(descriptor.pipelines.iter().all(|pipeline| {
        let bindings = match pipeline {
            Pipeline::Compute(pipeline) => &pipeline.bindings,
            Pipeline::Graphics(pipeline) => &pipeline.bindings,
        };
        bindings.iter().all(|binding| {
            !matches!(
                binding,
                Binding::StorageBuffer {
                    usage: BufferUsage::Intermediate,
                    ..
                }
            )
        })
    }));

    let compute_pass = descriptor
        .frame_graph
        .passes
        .iter()
        .position(|pass| pass.pipeline_index == 0 && pass.kind == FramePassKind::Compute)
        .expect("compute pass");
    let fragment_pass = descriptor
        .frame_graph
        .passes
        .iter()
        .position(|pass| pass.pipeline_index == 1 && pass.kind == FramePassKind::Fragment)
        .expect("fragment pass");
    assert!(
        descriptor.frame_graph.passes[fragment_pass].depends_on.contains(&compute_pass),
        "fragment pass must depend on the authored compute result"
    );
}

#[test]
fn direct_output_shares_an_authored_computed_result_with_fragment_shading() {
    run_with_large_stack(|| {
        let source = r#"
def vertex_main(vertex: vertex_invocation) vertex<vec2f32> =
  vertex_output(
    if vertex.vertex_index == 0u32 then @[-1.0, -1.0, 0.0, 1.0]
    else if vertex.vertex_index == 1u32 then @[3.0, -1.0, 0.0, 1.0]
    else @[-1.0, 3.0, 0.0, 1.0],
    @[0.0, 0.0])

entry frame(xs: []f32,
            screen: render_target<vec4f32>)
    ([]f32, render_target<vec4f32>) =
  let mapped = map(|x: f32| x + 1.0, xs) in
  let raster = rasterize_triangles(direct_draw(3u32, 1u32), vertex_main) in
  let screen1 = shade(screen, raster,
    |fragment| @[mapped[0], fragment.value.x, 0.0, 1.0]) in
  (mapped, screen1)
"#;

        let spirv = lower_ssa_to_spirv(
            lower_egir_to_ssa(plan_direct(source, CodegenTarget::Spirv).expect("direct SPIR-V plan"))
                .expect("direct SPIR-V SSA"),
        )
        .expect("direct SPIR-V lowering");
        assert_naga_accepts_spirv(&spirv.spirv);
        assert_direct_computed_result_descriptor(&spirv.pipeline);

        let wgsl = lower_ssa_to_wgsl_with_pipeline(
            lower_egir_to_ssa(plan_direct(source, CodegenTarget::Wgsl).expect("direct WGSL plan"))
                .expect("direct WGSL SSA"),
        )
        .expect("direct WGSL lowering");
        assert_eq!(wgsl.wgsl.matches("@compute").count(), 1);
        assert_eq!(wgsl.wgsl.matches("@vertex").count(), 1);
        assert_eq!(wgsl.wgsl.matches("@fragment").count(), 1);
        assert_direct_computed_result_descriptor(&wgsl.pipeline);
    });
}

/// Run source through the pipeline up to SSA.
fn compile_to_ssa(input: &str) -> ssa::stage::Elaborated {
    compile_thru_ssa(input).expect("compile to SSA")
}

/// Helper to check that code fails type checking (for testing error cases).
fn should_fail_type_check(input: &str) -> bool {
    compile_thru_frontend(input).is_err()
}

fn compile_to_segmented_egir(input: &str) -> egir::reify::Segmented {
    let program = compile_thru_tlc(input).expect("compile through TLC");
    let program = tlc::infer_input_slice_bounds(program);
    let program = to_egraph(program).expect("convert to raw semantic EGIR");
    egir::reify_soacs(program)
}

/// Helper to compile through semantic EGIR optimization and allocation.
/// Off-milestone stop — drives the typestate API directly so the same
/// Semantic module coverage spans both `type_check` and `to_tlc`.
fn compile_to_semantic_egir(input: &str) -> egir::ResourcesAllocated {
    let program = egir::optimize_semantic_operations(compile_to_segmented_egir(input))
        .expect("semantic EGIR optimization failed");
    let program = egir::lift_stage_uniform_values(program);
    egir::plan_logical_resources(program).expect("allocate semantic EGIR resources")
}

fn lower_semantic_egir(
    allocated: egir::ResourcesAllocated,
    profile: LoweringProfile,
) -> ssa::stage::Elaborated {
    let program = egir::plan(allocated, profile).expect("plan semantic EGIR");
    lower_egir_to_ssa(program).expect("lower planned EGIR to SSA")
}

fn allocated_entries(
    allocated: &egir::ResourcesAllocated,
) -> impl Iterator<Item = &egir::program::AllocatedEntry> {
    allocated.data.stages.stages().map(|(_, stage)| stage.body())
}

#[derive(Debug, Default, PartialEq, Eq)]
struct SemanticSoacStats {
    filters: usize,
    hists: usize,
    seg_maps: usize,
    seg_reds: usize,
    seg_scans: usize,
    mixed_scremas: usize,
    map_bodies: usize,
    reduce_operators: usize,
    scan_operators: usize,
}

fn semantic_soac_stats(allocated: &egir::ResourcesAllocated) -> SemanticSoacStats {
    use crate::egir::types::{EGraph, SideEffectKind, Soac, SoacEffect};

    fn visit(
        graph: &EGraph<egir::types::Semantic<egir::program::SemanticResourceRef>>,
        stats: &mut SemanticSoacStats,
    ) {
        for effect in graph.skeleton.blocks.iter().flat_map(|(_, block)| &block.side_effects) {
            let SideEffectKind::Soac(SoacEffect(_, soac)) = &effect.kind else {
                continue;
            };
            match soac {
                Soac::Filter(_) => stats.filters += 1,
                Soac::Hist(_) => stats.hists += 1,
                Soac::Screma(op) => {
                    stats.map_bodies +=
                        usize::from(!op.form.pre.is_identity()) + usize::from(!op.form.post.is_identity());
                    if op.is_map() {
                        stats.seg_maps += 1;
                    } else if op.is_reduce() {
                        stats.seg_reds += 1;
                        stats.reduce_operators += op.form.reductions.len();
                    } else if !op.form.scans.is_empty() && op.form.reductions.is_empty() {
                        stats.seg_scans += 1;
                        stats.scan_operators += op.form.scans.len();
                    } else {
                        stats.mixed_scremas += 1;
                        stats.reduce_operators += op.form.reductions.len();
                        stats.scan_operators += op.form.scans.len();
                    }
                }
            }
        }
    }

    let mut stats = SemanticSoacStats::default();
    for function in &allocated.functions {
        visit(&function.graph, &mut stats);
    }
    for (_, stage) in allocated.data.stages.stages() {
        visit(&stage.body().graph, &mut stats);
    }
    stats
}

fn segmented_entry_maps(program: &egir::reify::Segmented) -> Vec<&screma::Op<egir::types::Semantic>> {
    use crate::egir::types::{SideEffectKind, Soac, SoacEffect};

    program
        .entry_points
        .iter()
        .flat_map(|entry| entry.graph.skeleton.blocks.iter().flat_map(|(_, block)| &block.side_effects))
        .filter_map(|effect| {
            let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &effect.kind else {
                return None;
            };
            op.is_map().then_some(op)
        })
        .collect()
}

fn segmented_entry_map_output_fields(program: &egir::reify::Segmented) -> Vec<usize> {
    use crate::egir::types::{SideEffectKind, Soac, SoacEffect};

    let entry = program.entry_points.first().expect("semantic test entry");
    let results = entry.graph.side_effect_index();
    entry
        .routes()
        .map(|route| {
            results
                .effect_result_field(&entry.graph, route.source.value)
                .filter(|(effect, _, _)| {
                    matches!(&effect.kind, SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) if op.is_map())
                })
                .map(|(_, _, field)| field)
                .expect("entry output route does not select a map result field")
        })
        .collect()
}

// These semantic-EGIR tests are the behavioral successors to the deleted
// `tlc/fusion_tests.rs` suite. They assert the optimized operation graph rather
// than TLC syntax, so the checks survive representation changes while still
// pinning fusion legality, input routing, and escape behavior.

#[test]
fn egir_vertical_fusion_collapses_three_map_chain() {
    let source = r#"
entry chain(xs: []i32) []i32 =
  let a = map(|x: i32| x + 1, xs) in
  let b = map(|x: i32| x * 2, a) in
  map(|x: i32| x - 3, b)
"#;
    let stats = semantic_soac_stats(&compile_to_semantic_egir(source));
    assert_eq!(stats.seg_maps, 1, "three vertically fused maps become one SegMap");
    assert_eq!(
        stats.map_bodies, 1,
        "composition must not leave parallel map lanes"
    );
}

#[test]
fn reified_map_chain_does_not_materialize_singleton_result_bundles() {
    use crate::egir::types::{PureOp, ValueKind};

    let source = r#"
entry chain(xs: []i32) []i32 =
  let a = map(|x: i32| x + 1, xs) in
  let b = map(|x: i32| x * 2, a) in
  map(|x: i32| x - 3, b)
"#;
    let program = compile_to_segmented_egir(source);
    assert_eq!(
        segmented_entry_maps(&program).len(),
        3,
        "the reification checkpoint retains all three maps before fusion"
    );

    let entry = program.entry_points.first().expect("map-chain entry");
    let singleton_tuples = entry
        .graph
        .nodes
        .values()
        .filter(|node| {
            matches!(
                node.kind(),
                ValueKind::Pure {
                    op: PureOp::Tuple(1),
                    ..
                }
            )
        })
        .count();
    assert_eq!(
        singleton_tuples, 0,
        "one-field SOAC result bundles must remain binding metadata, not pure tuple values"
    );
}

#[test]
fn egir_vertical_fusion_preserves_multi_input_producer_sources() {
    use crate::egir::types::{ResourceAccess, SideEffectKind, Soac, SoacEffect};

    let allocated = compile_to_semantic_egir(
        r#"
entry zipped<[n]>(xs: [n]i32, ys: [n]i32) [n]i32 =
  let pairs = zip(xs, ys) in
  let sums = map(|p: (i32, i32)| p.0 + p.1, pairs) in
  map(|x: i32| x * 2, sums)
"#,
    );
    let maps: Vec<_> = allocated_entries(&allocated)
        .flat_map(|entry| entry.graph.skeleton.blocks.iter().flat_map(|(_, block)| &block.side_effects))
        .filter_map(|effect| {
            let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &effect.kind else {
                return None;
            };
            if !op.is_map() {
                return None;
            }
            let screma::SemanticState::Segmented { resources, .. } = op.semantic_state() else {
                return None;
            };
            Some((
                op.inputs.len(),
                resources.iter().filter(|resource| resource.access == ResourceAccess::Read).count(),
            ))
        })
        .collect();
    assert_eq!(maps.len(), 1, "the two maps should compose into one SegMap");
    assert_eq!(
        maps[0].1, 2,
        "both zip source resources must reach the composed region"
    );
}

#[test]
fn egir_vertical_fusion_composes_one_slot_of_multi_input_consumer() {
    use crate::egir::types::{SideEffectKind, Soac, SoacEffect};

    let allocated = compile_to_semantic_egir(
        r#"
entry mixed() [4]i32 =
  let produced = map(|x: i32| x + 1, [1, 2, 3, 4]) in
  map(|p: (i32, i32)| p.0 + p.1, zip(produced, [10, 20, 30, 40]))
"#,
    );
    let maps: Vec<_> = allocated_entries(&allocated)
        .flat_map(|entry| entry.graph.skeleton.blocks.iter().flat_map(|(_, block)| &block.side_effects))
        .filter_map(|effect| {
            let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &effect.kind else {
                return None;
            };
            if !op.is_map() {
                return None;
            }
            Some((
                op.inputs.len(),
                op.form.pre.parameter_types.len(),
                op.form.post.result_types.len(),
            ))
        })
        .collect();
    assert_eq!(maps.len(), 1, "the producer should compose into the zip consumer");
    assert_eq!(
        maps[0].0, maps[0].1,
        "the canonical pre-lambda has one parameter per array input"
    );
    assert_eq!(maps[0].2, 1, "the fused consumer keeps one output lane");
}

#[test]
fn egir_vertical_fusion_routes_distinct_producer_results() {
    let source = r#"
entry paired<[n]>(xs: [n]i32) [n]i32 =
  let pair = map(|x: i32| (x + 1, x * 2), xs) in
  let (plus, times) = unzip(pair) in
  map(|values: (i32, i32)| values.0 + values.1, zip(plus, times))
"#;

    let stats = semantic_soac_stats(&compile_to_semantic_egir(source));
    assert_eq!(
        stats.seg_maps, 1,
        "both producer fields route into one composed map"
    );
    assert_eq!(stats.mixed_scremas, 0);
    compile_to_spirv(source).expect("multi-result vertical fusion lowers to SPIR-V");
}
#[test]
fn egir_horizontal_fusion_deduplicates_shared_multi_input_vector() {
    use crate::egir::types::{SideEffectKind, Soac, SoacEffect};

    let allocated = compile_to_semantic_egir(
        r#"
entry siblings<[n]>(xs: [n]i32, ys: [n]i32) ([n]i32, [n]i32) =
  let pairs = zip(xs, ys) in
  let sums = map(|p: (i32, i32)| p.0 + p.1, pairs) in
  let diffs = map(|p: (i32, i32)| p.0 - p.1, pairs) in
  (sums, diffs)
"#,
    );
    let fused = allocated_entries(&allocated)
        .flat_map(|entry| entry.graph.skeleton.blocks.iter().flat_map(|(_, block)| &block.side_effects))
        .find_map(|effect| {
            let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &effect.kind else {
                return None;
            };
            if !op.is_map() || op.form.post.result_types.len() != 2 {
                return None;
            }
            Some((op.inputs.len(), op.form.pre.parameter_types.len()))
        })
        .expect("one two-lane SegMap");
    assert_eq!(fused.0, 1, "the shared zipped input must not be duplicated");
    assert_eq!(fused.1, 1);
}

#[test]
fn egir_horizontal_fusion_preserves_shared_input_semantics() {
    use crate::egir::semantic_exec::{execute_map_screma, Value};

    let source = r#"
entry siblings(xs: []i32) ([]i32, []i32) =
  let plus = map(|x: i32| x + 1, xs) in
  let times = map(|x: i32| x * 2, xs) in
  (plus, times)
"#;
    let before = compile_to_segmented_egir(source);
    let after = egir::optimize_semantic_operations(compile_to_segmented_egir(source))
        .expect("semantic EGIR optimization failed");
    let after: egir::reify::Segmented = egir::lift_stage_uniform_values(after).retag();
    let input = (0..8).map(Value::Int).collect::<Vec<_>>();

    let before_maps = segmented_entry_maps(&before);
    assert_eq!(before_maps.len(), 2, "unoptimized graph retains both siblings");
    let plus = execute_map_screma(&before, before_maps[0], &[input.clone()]).unwrap();
    let times = execute_map_screma(&before, before_maps[1], &[input.clone()]).unwrap();
    let expected = vec![plus[0].clone(), times[0].clone()];

    let after_maps = segmented_entry_maps(&after);
    assert_eq!(after_maps.len(), 1, "shared-input siblings fuse horizontally");
    let actual_fields = execute_map_screma(&after, after_maps[0], &[input]).unwrap();
    let actual = segmented_entry_map_output_fields(&after)
        .into_iter()
        .map(|field| actual_fields[field].clone())
        .collect::<Vec<_>>();
    assert_eq!(
        actual, expected,
        "horizontal fusion must preserve both result lanes"
    );
}

#[test]
fn egir_vertical_fusion_preserves_escaping_producer_output() {
    let source = r#"
entry both(xs: []i32) ([]i32, []i32) =
  let produced = map(|x: i32| x + 1, xs) in
  let consumed = map(|x: i32| x * 2, produced) in
  (produced, consumed)
"#;
    let stats = semantic_soac_stats(&compile_to_semantic_egir(source));
    assert_eq!(stats.seg_maps, 1, "the fused map retains both observable outputs");
    compile_to_spirv(source).expect("output-preserving vertical fusion lowers to SPIR-V");
}

#[test]
fn egir_vertical_fusion_absorbs_multiple_consumers() {
    let source = r#"
entry shared(xs: []i32) ([]i32, []i32) =
  let produced = map(|x: i32| x + 1, xs) in
  let left = map(|x: i32| x * 2, produced) in
  let right = map(|x: i32| x - 3, produced) in
  (left, right)
"#;
    let stats = semantic_soac_stats(&compile_to_semantic_egir(source));
    assert_eq!(
        stats.seg_maps, 1,
        "one Screma serves both consumers without materializing the shared producer"
    );
    compile_to_spirv(source).expect("multi-consumer vertical fusion lowers to SPIR-V");
}

#[test]
fn egir_vertical_fusion_preserves_fanout_semantics() {
    use crate::egir::semantic_exec::{execute_map_screma, Value};

    let source = r#"
entry shared(xs: []i32) ([]i32, []i32) =
  let produced = map(|x: i32| x + 1, xs) in
  let left = map(|x: i32| x * 2, produced) in
  let right = map(|x: i32| x - 3, produced) in
  (left, right)
"#;
    let before = compile_to_segmented_egir(source);
    let after = egir::optimize_semantic_operations(compile_to_segmented_egir(source))
        .expect("semantic EGIR optimization failed");
    let after: egir::reify::Segmented = egir::lift_stage_uniform_values(after).retag();
    let input = (0..8).map(Value::Int).collect::<Vec<_>>();

    let before_maps = segmented_entry_maps(&before);
    assert_eq!(before_maps.len(), 3, "unoptimized graph retains the fan-out");
    let produced = execute_map_screma(&before, before_maps[0], &[input.clone()]).unwrap();
    let left = execute_map_screma(&before, before_maps[1], &[produced[0].clone()]).unwrap();
    let right = execute_map_screma(&before, before_maps[2], &[produced[0].clone()]).unwrap();
    let expected = vec![left[0].clone(), right[0].clone()];

    let after_maps = segmented_entry_maps(&after);
    assert_eq!(after_maps.len(), 1, "fan-out normalizes to one Screma");
    let actual_fields = execute_map_screma(&after, after_maps[0], &[input]).unwrap();
    let actual = segmented_entry_map_output_fields(&after)
        .into_iter()
        .map(|field| actual_fields[field].clone())
        .collect::<Vec<_>>();
    assert_eq!(actual, expected, "fan-out fusion must preserve both consumers");
}
#[test]
fn egir_vertical_fusion_pushes_slice_onto_producer_inputs() {
    let source = r#"
entry sliced(xs: []i32) [4]i32 =
  let produced = map(|x: i32| x + 1, xs) in
  map(|x: i32| x * 2, produced[2..6])
"#;
    let stats = semantic_soac_stats(&compile_to_semantic_egir(source));
    assert_eq!(
        stats.seg_maps, 1,
        "a sliced producer should execute as one transformed Screma"
    );
    compile_to_spirv(source).expect("slice-transform vertical fusion lowers to SPIR-V");
}

#[test]
fn egir_vertical_fusion_preserves_slice_semantics() {
    use crate::egir::semantic_exec::{execute_map_screma, Value};

    let source = r#"
entry sliced(xs: []i32) [4]i32 =
  let produced = map(|x: i32| x + 1, xs) in
  map(|x: i32| x * 2, produced[2..6])
"#;
    let before = compile_to_segmented_egir(source);
    let after = egir::optimize_semantic_operations(compile_to_segmented_egir(source))
        .expect("semantic EGIR optimization failed");
    let after: egir::reify::Segmented = egir::lift_stage_uniform_values(after).retag();

    let input = (0..8).map(Value::Int).collect::<Vec<_>>();
    let before_maps = segmented_entry_maps(&before);
    assert_eq!(
        before_maps.len(),
        2,
        "unoptimized graph retains producer and consumer"
    );
    let produced = execute_map_screma(&before, before_maps[0], &[input.clone()]).unwrap();
    let sliced = produced[0][2..6].to_vec();
    let expected = execute_map_screma(&before, before_maps[1], &[sliced.clone()]).unwrap();

    let after_maps = segmented_entry_maps(&after);
    assert_eq!(after_maps.len(), 1, "optimization composes the two maps");
    let actual = execute_map_screma(&after, after_maps[0], &[input[2..6].to_vec()]).unwrap();
    assert_eq!(
        actual, expected,
        "fused lambda must preserve concrete lane values"
    );
}
#[test]
fn egir_vertical_fusion_composes_nested_slices() {
    let source = r#"
entry sliced_twice(xs: []i32) [3]i32 =
  let produced = map(|x: i32| x + 1, xs) in
  map(|x: i32| x * 2, produced[1..7][2..5])
"#;
    let stats = semantic_soac_stats(&compile_to_semantic_egir(source));
    assert_eq!(
        stats.seg_maps, 1,
        "nested slices should remain one ordered input-transform chain"
    );
    compile_to_spirv(source).expect("nested slice-transform fusion lowers to SPIR-V");
}

#[test]
fn egir_vertical_fusion_slices_every_producer_input() {
    let source = r#"
entry sliced_zip(xs: []i32, ys: []i32) [4]i32 =
  let produced = map(|pair: (i32, i32)| pair.0 + pair.1, zip(xs, ys)) in
  map(|x: i32| x * 2, produced[2..6])
"#;
    let stats = semantic_soac_stats(&compile_to_semantic_egir(source));
    assert_eq!(
        stats.seg_maps, 1,
        "one slice transform must be pushed onto every producer input"
    );
    compile_to_spirv(source).expect("multi-input slice-transform fusion lowers to SPIR-V");
}

#[test]
fn egir_vertical_fusion_declines_incompatible_slice_routes() {
    let source = r#"
entry differently_sliced(xs: [8]i32) [4]i32 =
  let produced = map(|x: i32| x + 1, xs) in
  map(|pair: (i32, i32)| pair.0 + pair.1, zip(produced[0..4], produced[2..6]))
"#;
    let stats = semantic_soac_stats(&compile_to_semantic_egir(source));
    assert_eq!(
        stats.seg_maps, 2,
        "different producer indices cannot share one transformed producer invocation"
    );
    compile_to_spirv(source).expect("incompatible slice routes remain safely materialized");
}

#[test]
fn egir_vertical_fusion_does_not_shrink_retained_producer_output() {
    let source = r#"
entry retained(xs: []i32) ([]i32, [4]i32) =
  let produced = map(|x: i32| x + 1, xs) in
  let sliced = map(|x: i32| x * 2, produced[2..6]) in
  (produced, sliced)
"#;
    let stats = semantic_soac_stats(&compile_to_semantic_egir(source));
    assert_eq!(
        stats.seg_maps, 2,
        "a transformed fusion must not replace an escaping full-domain producer"
    );
    compile_to_spirv(source).expect("retained producer and sliced consumer lower independently");
}

#[test]
fn egir_indexed_fusion_scalarizes_one_static_demand() {
    let source = r#"
entry one() [1]i32 =
  let produced = map(|x: i32| x + 1, 0i32 ..< 8) in
  [produced[3]]
"#;
    let stats = semantic_soac_stats(&compile_to_semantic_egir(source));
    assert_eq!(
        stats.seg_maps, 0,
        "one static demand should remove the array producer"
    );
    compile_to_spirv(source).expect("scalarized static demand should lower");
}

#[test]
fn egir_indexed_fusion_rewrites_direct_output_route() {
    let source = r#"
entry one() i32 =
  let produced = map(|x: i32| x + 1, 0i32 ..< 8) in
  produced[3]
"#;
    let stats = semantic_soac_stats(&compile_to_semantic_egir(source));
    assert_eq!(
        stats.seg_maps, 0,
        "a direct scalar output should not keep the array producer"
    );
    compile_to_spirv(source).expect("scalarized direct output should lower");
}

#[test]
fn egir_indexed_fusion_lowers_two_static_demands() {
    let source = r#"
entry two() [1]i32 =
  let produced = map(|x: i32| x + 1, 0i32 ..< 8) in
  [produced[2] + produced[3]]
"#;
    compile_to_spirv(source).expect("two static demands should lower");
}

#[test]
fn egir_indexed_fusion_keeps_producer_that_is_also_returned() {
    let source = r#"
entry both() ([8]i32, [1]i32) =
  let produced = map(|x: i32| x + 1, 0i32 ..< 8) in
  (produced, [produced[3]])
"#;
    let stats = semantic_soac_stats(&compile_to_semantic_egir(source));
    assert_eq!(
        stats.seg_maps, 1,
        "a directly returned producer must remain materialized"
    );
    compile_to_spirv(source).expect("returned producer with a static demand should lower");
}
#[test]
fn egir_indexed_fusion_respects_fixed_domain_profitability() {
    let source = r#"
entry many(i: i32, j: i32, k: i32) [1]i32 =
  let produced = map(|x: i32| x + 1, [10, 20]) in
  [produced[i % 2] + produced[j % 2] + produced[k % 2]]
"#;
    let stats = semantic_soac_stats(&compile_to_semantic_egir(source));
    assert_eq!(
        stats.seg_maps, 1,
        "three point evaluations should not replace a two-element materialization"
    );
}

#[test]
fn egir_indexed_fusion_bounds_unknown_domain_duplication() {
    let source = r#"
entry many(xs: []i32, i: i32, j: i32, k: i32) [1]i32 =
  let produced = map(|x: i32| x + 1, xs) in
  [produced[i] + produced[j] + produced[k]]
"#;
    let stats = semantic_soac_stats(&compile_to_semantic_egir(source));
    assert_eq!(
        stats.seg_maps, 1,
        "unknown-size maps should not be copied across unbounded point demands"
    );
}
#[test]
fn egir_filter_length_only_becomes_count_reduction() {
    let source = r#"
entry count(xs: []i32) i32 =
  let kept = filter(|x: i32| x > 0, xs) in
  length(kept)
"#;
    let stats = semantic_soac_stats(&compile_to_semantic_egir(source));
    assert_eq!(stats.filters, 0);
    assert_eq!(stats.seg_reds, 1);
    assert_eq!(
        stats.reduce_operators, 1,
        "length-only fusion needs one count operator"
    );
}

#[test]
fn egir_filter_fusion_reuses_count_for_multiple_reductions() {
    use crate::egir::types::{SideEffectKind, Soac, SoacEffect};

    let source = r#"
entry stats(xs: []i32) [4]i32 =
  let kept = filter(|x: i32| x > 0, xs) in
  let n1 = length(kept) in
  let total = reduce(|a: i32, x: i32| a + x, 0, kept) in
  let n2 = length(kept) in
  let maximum = reduce(|a: i32, x: i32| if a > x then a else x, -2147483648, kept) in
  [n1, total, n2, maximum]
"#;
    let allocated = compile_to_semantic_egir(source);
    let stats = semantic_soac_stats(&allocated);
    assert_eq!(stats.filters, 0, "the non-escaping filter should disappear");
    assert_eq!(
        stats.seg_reds, 1,
        "the reductions and count should share one SegRed"
    );
    assert_eq!(stats.reduce_operators, 3, "two reductions plus one shared count");

    let op = allocated_entries(&allocated)
        .flat_map(|entry| entry.graph.skeleton.blocks.iter().flat_map(|(_, block)| &block.side_effects))
        .find_map(|effect| {
            let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &effect.kind else {
                return None;
            };
            (op.is_reduce() && op.form.reductions.len() == 3).then_some(op)
        })
        .expect("three-operator filtered SegRed");
    let pre_region = op.form.pre.seg_body().expect("masked pre-lambda").region;
    let pre_name = allocated.data.core.identities.function_name(pre_region);
    assert!(
        pre_name.contains("filter_pre"),
        "masking belongs in the pre-lambda: {pre_name}"
    );
    let operator_names = op
        .form
        .reductions
        .iter()
        .map(|reduction| {
            allocated.data.core.identities.function_name(reduction.operator.seg_body().unwrap().region)
        })
        .collect::<Vec<_>>();
    assert!(
        operator_names[2].contains("filter_count_combine"),
        "the shared count reducer stays last: {operator_names:?}"
    );

    compile_to_spirv(source).expect("multi-consumer filtered reduction should lower");
}

#[test]
fn egir_filter_fusion_is_blocked_when_filtered_array_escapes() {
    let source = r#"
entry both(xs: []i32) ?k. ([k]i32, i32) =
  let kept = filter(|x: i32| x > 0, xs) in
  (kept, length(kept))
"#;
    let stats = semantic_soac_stats(&compile_to_semantic_egir(source));
    assert_eq!(
        stats.filters, 1,
        "the returned filtered array still needs compaction"
    );
    assert_eq!(
        stats.seg_reds, 0,
        "its length cannot be detached into a masked reduction"
    );
}

#[test]
fn egir_map_filter_envelope_fuses_producer_into_escaping_filter() {
    use crate::egir::soac::filter;
    use crate::egir::types::{SideEffectKind, Soac, SoacEffect};

    let source = r#"
entry pick(xs: []i32) ?k. [k]i32 =
  let shifted = map(|x: i32| x + 1, xs) in
  filter(|x: i32| x > 0, shifted)
"#;
    let allocated = compile_to_semantic_egir(source);
    let stats = semantic_soac_stats(&allocated);
    assert_eq!(stats.seg_maps, 0, "the producer map should not materialize");
    assert_eq!(stats.filters, 1, "the escaping filter remains the envelope");
    let has_map_body = allocated_entries(&allocated)
        .flat_map(|entry| entry.graph.skeleton.blocks.iter().flat_map(|(_, block)| &block.side_effects))
        .any(|effect| match &effect.kind {
            SideEffectKind::Soac(SoacEffect(
                _,
                Soac::Filter(filter::Op {
                    body: filter::Body { map, .. },
                    ..
                }),
            )) => !map.is_identity(),
            _ => false,
        });
    assert!(
        has_map_body,
        "the filter must carry the producer's callable region"
    );
}

#[test]
fn egir_reduce_by_index_is_a_general_histogram_and_lowers() {
    use crate::egir::soac::hist;
    use crate::egir::types::{SideEffectKind, Soac, SoacEffect};

    let source = r#"
entry accumulate(indices: []i32,
                 values: []i32,
                 bias: i32,
                 dest: *[]i32) () =
  let updated = reduce_by_index(dest, |a: i32, b: i32| a + b + bias, -bias, indices, values) in
  let _ = scatter(updated, [0i32], [bias]) in
  ()
"#;
    let allocated = compile_to_semantic_egir(source);
    let histogram = allocated_entries(&allocated)
        .flat_map(|entry| entry.graph.skeleton.blocks.iter().flat_map(|(_, block)| &block.side_effects))
        .find_map(|effect| match &effect.kind {
            SideEffectKind::Soac(SoacEffect(_, Soac::Hist(op))) => Some(op),
            _ => None,
        })
        .expect("semantic reduce_by_index histogram");
    assert!(histogram.form.bucket.is_identity());
    let hist::Update::Reduce { operator, .. } = &histogram.form.operations[0].update else {
        panic!("reduce_by_index must retain its reducer")
    };
    assert_eq!(operator.parameter_types.len(), 2);
    assert_eq!(operator.result_types.len(), 1);
    assert_eq!(
        operator.seg_body().expect("reduce_by_index reducer region").captures.len(),
        1,
        "captured reducer inputs remain explicit in the histogram operator",
    );
    compile_to_spirv(source).expect("general histogram should lower through read-combine-write");
}

#[test]
fn egir_map_into_reducing_histogram_fuses() {
    use crate::egir::soac::hist;
    use crate::egir::types::{SideEffectKind, Soac, SoacEffect};

    let source = r#"
entry accumulate(indices: []i32,
                 values: []i32,
                 dest: *[]i32) () =
  let doubled = map(|value: i32| value * 2, values) in
  let _ = reduce_by_index(dest, |a: i32, b: i32| a + b, 0, indices, doubled) in
  ()
"#;
    let allocated = compile_to_semantic_egir(source);
    let stats = semantic_soac_stats(&allocated);
    assert_eq!(
        stats.seg_maps, 0,
        "the producer map should fold into the bucket lambda"
    );
    let histogram = allocated_entries(&allocated)
        .flat_map(|entry| entry.graph.skeleton.blocks.iter().flat_map(|(_, block)| &block.side_effects))
        .find_map(|effect| match &effect.kind {
            SideEffectKind::Soac(SoacEffect(_, Soac::Hist(op))) => Some(op),
            _ => None,
        })
        .expect("fused reducing histogram");
    assert!(!histogram.form.bucket.is_identity());
    assert!(matches!(
        histogram.form.operations[0].update,
        hist::Update::Reduce { .. }
    ));
    let planned = egir::plan(compile_to_semantic_egir(source), LoweringProfile::PORTABLE)
        .expect("plan direct-atomic histogram");
    assert_eq!(
        planned.physical_kernels().phases().map(|phase| phase.label.as_str()).collect::<Vec<_>>(),
        ["hist_atomic"]
    );

    let spirv = compile_to_spirv(source).expect("map-reduce_by_index fusion should lower");
    let mut loader = wspirv::dr::Loader::new();
    wspirv::binary::parse_words(&spirv, &mut loader).expect("parse atomic histogram SPIR-V");
    assert!(
        loader.module().functions.iter().any(|function| {
            function.blocks.iter().any(|block| {
                block.instructions.iter().any(|inst| inst.class.opcode == wspirv::spirv::Op::AtomicIAdd)
            })
        }),
        "integer-add histogram must emit OpAtomicIAdd"
    );

    let wgsl = lower_ssa_to_wgsl(lower_semantic_egir(
        compile_to_semantic_egir(source),
        LoweringProfile::new(CodegenTarget::Wgsl, SchedulePolicy::Parallel),
    ))
    .expect("atomic histogram lowers to WGSL");
    assert!(
        wgsl.contains("array<atomic<i32>>"),
        "atomic storage declaration:\n{wgsl}"
    );
    assert!(wgsl.contains("atomicAdd("), "atomic histogram update:\n{wgsl}");
}
#[test]
fn captured_integer_histogram_reducer_uses_compare_exchange_loop() {
    let source = r#"
entry accumulate(indices: []i32,
                 values: []i32,
                 bias: i32,
                 dest: *[]i32) () =
  let _ = reduce_by_index(dest, |a: i32, b: i32| a + b + bias, -bias, indices, values) in
  ()
"#;
    let planned = egir::plan(compile_to_semantic_egir(source), LoweringProfile::PORTABLE)
        .expect("plan compare-exchange histogram");
    assert_eq!(
        planned.physical_kernels().phases().map(|phase| phase.label.as_str()).collect::<Vec<_>>(),
        ["hist_atomic"]
    );

    let spirv = compile_to_spirv(source).expect("CAS histogram compiles to SPIR-V");
    let mut loader = wspirv::dr::Loader::new();
    wspirv::binary::parse_words(&spirv, &mut loader).expect("parse CAS histogram SPIR-V");
    assert!(
        loader.module().functions.iter().any(|function| {
            function.blocks.iter().any(|block| {
                block
                    .instructions
                    .iter()
                    .any(|inst| inst.class.opcode == wspirv::spirv::Op::AtomicCompareExchange)
            })
        }),
        "general scalar reducer must emit OpAtomicCompareExchange"
    );

    let wgsl = lower_ssa_to_wgsl(lower_semantic_egir(
        compile_to_semantic_egir(source),
        LoweringProfile::new(CodegenTarget::Wgsl, SchedulePolicy::Parallel),
    ))
    .expect("CAS histogram lowers to WGSL");
    assert!(
        wgsl.contains("atomicCompareExchangeWeak("),
        "CAS histogram update:\n{wgsl}"
    );
    assert!(wgsl.contains(".exchanged"), "weak-CAS retry condition:\n{wgsl}");
}
#[test]
fn high_race_factor_histogram_keeps_serial_fallback_without_replication() {
    use crate::egir::types::{PureOp, SideEffectKind, Soac, SoacEffect};
    use smallvec::smallvec;

    let source = r#"
entry accumulate(indices: []i32,
                 values: []i32,
                 dest: *[]i32) () =
  let _ = reduce_by_index(dest, |a: i32, b: i32| a + b, 0, indices, values) in
  ()
"#;
    let mut allocated = compile_to_semantic_egir(source);
    let (stage, race_factor) = allocated
        .data
        .stages
        .stages()
        .find_map(|(stage, staged)| {
            let entry = staged.body();
            entry.graph.skeleton.blocks.iter().flat_map(|(_, block)| &block.side_effects).find_map(
                |effect| match &effect.kind {
                    SideEffectKind::Soac(SoacEffect(_, Soac::Hist(op))) => {
                        Some((stage, op.form.operations[0].race_factor))
                    }
                    _ => None,
                },
            )
        })
        .expect("semantic histogram race factor");
    allocated.data.stages.stage_body_mut(stage).unwrap().graph.replace_pure_node(
        race_factor,
        PureOp::Int("33".into()),
        smallvec![],
    );

    let planned = egir::plan(allocated, LoweringProfile::PORTABLE).expect("plan high-contention histogram");
    assert_eq!(
        planned.physical_kernels().phases().map(|phase| phase.label.as_str()).collect::<Vec<_>>(),
        ["serial_compute"],
        "without replica storage, a high race-factor hint must not select the direct-atomic recipe"
    );
}

#[test]
fn ranked_bucket_scatter_fuses_generated_items_and_tiles_rank_three() {
    use crate::pipeline_descriptor::{Binding, BufferLen, BufferUsage, DispatchSize, Pipeline};

    let source = r#"
entry collision_shape_3d(dest: *[64][64]u32) ([64][64]u32, [64]u32, u32) =
  let items: [4096][658][2016](i32, u32) =
    map(|bucket_y: i32|
      map(|bucket_x: i32|
        map(|pair: i32|
          ((bucket_y + bucket_x + pair) % 64,
           u32(bucket_y + bucket_x + pair)),
          iota(2016)),
        iota(658)),
      iota(4096))
  in bucket_scatter_3d(dest, items)
"#;

    let wgsl = lower_ssa_to_wgsl(lower_semantic_egir(
        compile_to_semantic_egir(source),
        LoweringProfile::new(CodegenTarget::Wgsl, SchedulePolicy::Parallel),
    ))
    .expect("ranked bucket scatter lowers to WGSL");
    assert!(
        wgsl.contains("_wgsl_gid.y"),
        "ranked insertion must use dispatch y:\n{wgsl}"
    );
    assert!(
        wgsl.contains("_wgsl_gid.z"),
        "ranked insertion must use dispatch z:\n{wgsl}"
    );
    assert_eq!(
        wgsl.matches("@compute").count(),
        3,
        "init, insert, and finish stages:\n{wgsl}"
    );
    let module = naga::front::wgsl::parse_str(&wgsl)
        .unwrap_or_else(|error| panic!("Naga rejected bucket WGSL: {error:?}\n{wgsl}"));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|error| panic!("Naga validation rejected bucket WGSL: {error:?}\n{wgsl}"));

    let lowered = compile_thru_spirv(source).expect("ranked bucket scatter emits SPIR-V");
    let compute = lowered
        .pipeline
        .pipelines
        .iter()
        .find_map(|pipeline| match pipeline {
            Pipeline::Compute(compute)
                if compute
                    .stages
                    .iter()
                    .any(|stage| stage.entry_point == "collision_shape_3d_bucket_insert") =>
            {
                Some(compute)
            }
            _ => None,
        })
        .expect("bucket compute pipeline");
    assert_eq!(compute.stages.len(), 3, "bucket scatter has exactly three stages");
    let output_lengths = compute
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
        .collect::<Vec<_>>();
    assert_eq!(
        output_lengths,
        [
            Some(BufferLen::Fixed { bytes: 256 }),
            Some(BufferLen::Fixed { bytes: 4 }),
        ],
        "counts and scalar overflow outputs must both publish allocation sizes"
    );
    let materialized_candidate_bytes = 4096u64 * 658 * 2016 * 8;
    assert!(
        compute.bindings.iter().all(|binding| {
            !matches!(
                binding,
                Binding::StorageBuffer {
                    length: Some(BufferLen::Fixed { bytes }),
                    ..
                } if *bytes == materialized_candidate_bytes
            )
        }),
        "generated collision candidates must not receive a materialized storage binding"
    );
    let insert = compute
        .stages
        .iter()
        .find(|stage| stage.entry_point == "collision_shape_3d_bucket_insert")
        .expect("bucket insertion stage");
    assert_eq!(
        insert.dispatch_size,
        DispatchSize::Fixed {
            x: 32,
            y: 658,
            z: 4096,
            explicit: false,
        }
    );
}

#[test]
fn bucket_scatter_accepts_named_constant_destination_dimensions() {
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            let source = r#"
def BUCKETS = 1i32
def CAPACITY = 4i32

entry scatter_with_named_shape(
    dest: *[BUCKETS][CAPACITY]u32
) ([BUCKETS][CAPACITY]u32, [BUCKETS]u32, u32) =
  let items = map(|i: i32| (0i32, u32.i32(i)), iota(CAPACITY)) in
  bucket_scatter_1d(dest, items)
"#;

            let wgsl = lower_ssa_to_wgsl(lower_semantic_egir(
                compile_to_semantic_egir(source),
                LoweringProfile::new(CodegenTarget::Wgsl, SchedulePolicy::Parallel),
            ))
            .expect("named bucket dimensions lower to WGSL");
            let module = naga::front::wgsl::parse_str(&wgsl).unwrap_or_else(|error| {
                panic!("Naga rejected named-dimension bucket WGSL: {error:?}\n{wgsl}")
            });
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap_or_else(|error| {
                panic!("Naga validation rejected named-dimension bucket WGSL: {error:?}\n{wgsl}")
            });
        })
        .expect("spawn named-dimension bucket regression")
        .join()
        .expect("named-dimension bucket regression panicked");
}

#[test]
fn wgsl_lowering_retains_the_pipeline_descriptor() {
    let source = r#"
entry descriptor_for_wgsl(dest: *[2][4]u32) ([2][4]u32, [2]u32, u32) =
  bucket_scatter_1d(dest, [(0, 10u32), (1, 20u32)])
"#;

    let lowered = lower_ssa_to_wgsl_with_pipeline(lower_semantic_egir(
        compile_to_semantic_egir(source),
        LoweringProfile::new(CodegenTarget::Wgsl, SchedulePolicy::Parallel),
    ))
    .expect("bucket scatter lowers to WGSL with its descriptor");

    assert!(lowered.wgsl.contains("@compute"));
    assert!(
        !lowered.pipeline.pipelines.is_empty(),
        "WGSL output must retain the planned runtime pipeline"
    );
}

#[test]
fn wgsl_descriptor_publishes_scalar_inputs_as_the_emitted_storage_block() {
    use crate::pipeline_descriptor::{Access, Binding, BufferLen, BufferUsage, Pipeline};

    let source = r#"
entry scalar_parameters(index: u32) u32 = index + 1u32
"#;
    let lowered = lower_ssa_to_wgsl_with_pipeline(lower_semantic_egir(
        compile_to_semantic_egir(source),
        LoweringProfile::new(CodegenTarget::Wgsl, SchedulePolicy::Parallel),
    ))
    .expect("scalar input lowers to WGSL with a WebGPU descriptor");

    assert!(
        lowered.wgsl.contains("@group(1) @binding(0) var<storage, read> _pc0"),
        "WGSL must declare the parameter block at the published slot:\n{}",
        lowered.wgsl
    );
    let Pipeline::Compute(compute) = &lowered.pipeline.pipelines[0] else {
        panic!("scalar entry must publish a compute pipeline");
    };
    assert!(
        compute.bindings.iter().all(|binding| !matches!(binding, Binding::PushConstant { .. })),
        "WebGPU descriptors cannot expose push constants"
    );
    let parameter = compute
        .bindings
        .iter()
        .find(|binding| {
            matches!(
                binding,
                Binding::StorageBuffer {
                    set: 1,
                    binding: 0,
                    ..
                }
            )
        })
        .expect("descriptor must publish the WGSL parameter block");
    let Binding::StorageBuffer {
        access,
        usage,
        name,
        length,
        members,
        ..
    } = parameter
    else {
        unreachable!()
    };
    assert_eq!((access, usage), (&Access::ReadOnly, &BufferUsage::Input));
    assert_eq!(name, "index");
    assert_eq!(*length, Some(BufferLen::Fixed { bytes: 4 }));
    assert_eq!(members.len(), 1);
    assert_eq!(
        (members[0].name.as_str(), members[0].offset, members[0].size),
        ("index", 0, 4)
    );
    assert_eq!(compute.stages[0].reads, vec![1]);

    let spirv = compile_thru_spirv(source).expect("SPIR-V still accepts the scalar entry");
    let Pipeline::Compute(compute) = &spirv.pipeline.pipelines[0] else {
        panic!("scalar entry must publish a compute pipeline");
    };
    assert!(
        compute
            .bindings
            .iter()
            .any(|binding| matches!(binding, Binding::PushConstant { offset: 0, size: 4, name } if name == "index")),
        "SPIR-V must retain its native push-constant ABI"
    );
}

#[test]
fn wgsl_descriptor_reads_dynamic_dispatch_length_from_the_parameter_block() {
    use crate::pipeline_descriptor::{Binding, DispatchLen, DispatchSize, Pipeline};

    let source = r#"
entry dynamic_parameter(n: u32) []u32 =
  map(|i: u32| i + 1u32, 0u32..<n)
"#;
    let lowered = lower_ssa_to_wgsl_with_pipeline(lower_semantic_egir(
        compile_to_semantic_egir(source),
        LoweringProfile::new(CodegenTarget::Wgsl, SchedulePolicy::Parallel),
    ))
    .expect("dynamic scalar input lowers to a WebGPU descriptor");
    let Pipeline::Compute(compute) = &lowered.pipeline.pipelines[0] else {
        panic!("map entry must publish a compute pipeline");
    };
    assert!(compute.bindings.iter().all(|binding| !matches!(binding, Binding::PushConstant { .. })));
    assert!(compute.stages.iter().any(|stage| {
        matches!(
            stage.dispatch_size,
            DispatchSize::DerivedFrom {
                len: DispatchLen::StorageBuffer {
                    set: 1,
                    binding: 0,
                    offset: 0
                },
                ..
            }
        )
    }));
}

#[test]
fn ranked_bucket_scatter_named_helper_reads_fixed_storage_array() {
    use crate::pipeline_descriptor::{Binding, BufferLen, Pipeline};

    let source = r#"
def item_from_source(
    source: [64][64]u32,
    row: i32,
    column: i32
) (i32, u32) =
  ((row + column) % 4, source[row][column])

entry named_storage_helper(
    dest: *[4][8]u32,
    source: [64][64]u32
) ([4][8]u32, [4]u32, u32) =
  let items: [2][2](i32, u32) =
    map(|row: i32|
      map(|column: i32|
        item_from_source(source, row, column),
        iota(2)),
      iota(2))
  in bucket_scatter_2d(dest, items)
"#;

    let wgsl = lower_ssa_to_wgsl(lower_semantic_egir(
        compile_to_semantic_egir(source),
        LoweringProfile::new(CodegenTarget::Wgsl, SchedulePolicy::Parallel),
    ))
    .expect("named helper over a fixed storage array lowers to WGSL");

    let module = naga::front::wgsl::parse_str(&wgsl)
        .unwrap_or_else(|error| panic!("Naga rejected named-helper WGSL: {error:?}\n{wgsl}"));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|error| panic!("Naga validation rejected named-helper WGSL: {error:?}\n{wgsl}"));

    let lowered = compile_thru_spirv(source).expect("named helper emits a pipeline descriptor");
    let Pipeline::Compute(compute) = lowered.pipeline.pipelines.first().expect("compute pipeline") else {
        panic!("named helper must publish a compute pipeline")
    };
    let mut lengths = compute
        .bindings
        .iter()
        .filter_map(|binding| match binding {
            Binding::StorageBuffer { name, length, .. } => Some((name.as_str(), length.clone())),
            _ => None,
        })
        .collect::<Vec<_>>();
    lengths.sort_by_key(|(_, length)| match length {
        Some(BufferLen::Fixed { bytes }) => *bytes,
        _ => u64::MAX,
    });
    assert_eq!(
        lengths,
        [
            (
                "named_storage_helper_output_2",
                Some(BufferLen::Fixed { bytes: 4 })
            ),
            (
                "named_storage_helper_output_1",
                Some(BufferLen::Fixed { bytes: 16 })
            ),
            ("dest", Some(BufferLen::Fixed { bytes: 128 })),
            ("source", Some(BufferLen::Fixed { bytes: 16_384 })),
        ]
    );
}

#[test]
fn ranked_bucket_scatter_reads_bound_rank_two_aos_items() {
    let source = r#"
entry collision_shape_2d_bound(
    dest: *[64][64]u32,
    items: [64][32](i32, u32)
) ([64][64]u32, [64]u32, u32) =
  bucket_scatter_2d(dest, items)
"#;

    let wgsl = lower_ssa_to_wgsl(lower_semantic_egir(
        compile_to_semantic_egir(source),
        LoweringProfile::new(CodegenTarget::Wgsl, SchedulePolicy::Parallel),
    ))
    .expect("bound ranked bucket scatter lowers to WGSL");
    assert!(
        wgsl.contains("@binding(1) var<storage, read> _buf_0_1: array<array<T"),
        "bound item storage must retain its array-of-struct layout:\n{wgsl}"
    );
    let module = naga::front::wgsl::parse_str(&wgsl)
        .unwrap_or_else(|error| panic!("Naga rejected bound bucket WGSL: {error:?}\n{wgsl}"));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|error| panic!("Naga rejected bound bucket layout: {error:?}\n{wgsl}"));
    compile_thru_spirv(source).expect("bound ranked bucket scatter emits SPIR-V");

    let serial = egir::plan(
        compile_to_semantic_egir(source),
        LoweringProfile::new(CodegenTarget::Wgsl, SchedulePolicy::Serial),
    );
    let Err(error) = serial else {
        panic!("serial bucket scatter must report its required pipeline")
    };
    assert!(
        error.to_string().contains("requires its init/insert/finish pipeline"),
        "unexpected serial-scheduling diagnostic: {error}"
    );
}

#[test]
fn ranked_bucket_scatter_records_layout_independently_of_logical_rank() {
    use crate::egir::types::{ArrayLayout, SideEffectKind, Soac, SoacEffect};

    fn bucket_layouts(source: &str) -> Vec<ArrayLayout> {
        let program = compile_to_semantic_egir(source);
        let layouts = allocated_entries(&program)
            .flat_map(|entry| entry.graph.skeleton.blocks.iter().flat_map(|(_, block)| &block.side_effects))
            .find_map(|effect| match &effect.kind {
                SideEffectKind::Soac(SoacEffect(_, Soac::Hist(operation))) => {
                    Some(operation.inputs.iter().map(|input| input.layout.clone()).collect())
                }
                _ => None,
            })
            .expect("bucket histogram");
        layouts
    }

    let bound = r#"
entry bound_layout(dest: *[4][8]u32, items: [64][32](i32, u32))
    ([4][8]u32, [4]u32, u32) =
  bucket_scatter_2d(dest, items)
"#;
    assert_eq!(bucket_layouts(bound), [ArrayLayout::StorageAos]);

    let generated = r#"
entry generated_layout(dest: *[4][8]u32, offset: u32)
    ([4][8]u32, [4]u32, u32) =
  let items: [2][2](i32, u32) =
    map(|row: i32|
      map(|column: i32| ((row + column) % 4, u32(row + column) + offset), iota(2)),
      iota(2))
  in bucket_scatter_2d(dest, items)
"#;
    assert_eq!(
        bucket_layouts(generated),
        [ArrayLayout::Generated, ArrayLayout::Generated]
    );
    compile_thru_spirv(generated).expect("generated layout with a mixed scalar capture emits SPIR-V");

    let literal = r#"
entry literal_layout(dest: *[4][8]u32) ([4][8]u32, [4]u32, u32) =
  let items: [2][2](i32, u32) =
    [[(0, 10u32), (1, 11u32)], [(2, 12u32), (3, 13u32)]]
  in bucket_scatter_2d(dest, items)
"#;
    assert_eq!(bucket_layouts(literal), [ArrayLayout::StructureOfArrays]);
    compile_thru_spirv(literal).expect("literal SoA layout emits SPIR-V");
    let wgsl = lower_ssa_to_wgsl(lower_semantic_egir(
        compile_to_semantic_egir(literal),
        LoweringProfile::new(CodegenTarget::Wgsl, SchedulePolicy::Parallel),
    ))
    .expect("literal SoA layout emits WGSL");
    let module = naga::front::wgsl::parse_str(&wgsl)
        .unwrap_or_else(|error| panic!("Naga rejected literal bucket WGSL: {error:?}\n{wgsl}"));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|error| panic!("Naga validation rejected literal bucket WGSL: {error:?}\n{wgsl}"));
}

#[test]
fn ranked_bucket_scatter_accepts_rank_one_and_rank_four_generated_domains() {
    let source = r#"
entry bucket_rank_1(dest: *[4][8]u32) ([4][8]u32, [4]u32, u32) =
  let items = map(|i: i32| (i % 5 - 1, u32(i)), iota(16)) in
  bucket_scatter_1d(dest, items)

entry bucket_rank_4(dest: *[4][8]u32) ([4][8]u32, [4]u32, u32) =
  let items: [2][2][2][2](i32, u32) =
    map(|a: i32|
      map(|b: i32|
        map(|c: i32|
          map(|d: i32| ((a + b + c + d) % 4, u32(a + b + c + d)), iota(2)),
          iota(2)),
        iota(2)),
      iota(2))
  in bucket_scatter_4d(dest, items)
"#;

    compile_thru_spirv(source).expect("rank-one and rank-four bucket domains emit SPIR-V");
}

#[test]
fn ranked_bucket_scatter_regroups_logical_dimensions_to_fit_dispatch_limits() {
    use crate::pipeline_descriptor::{DispatchSize, Pipeline};

    let source = r#"
entry regrouped_bucket_domain(dest: *[4][8]u32) ([4][8]u32, [4]u32, u32) =
  let items: [256][256][2][2](i32, u32) =
    map(|a: i32|
      map(|b: i32|
        map(|c: i32|
          map(|d: i32| ((a + b + c + d) % 4, u32(a + b + c + d)), iota(2)),
          iota(2)),
        iota(256)),
      iota(256))
  in bucket_scatter_4d(dest, items)
"#;

    let lowered = compile_thru_spirv(source).expect("regrouped bucket domain emits SPIR-V");
    let insert = lowered
        .pipeline
        .pipelines
        .iter()
        .find_map(|pipeline| match pipeline {
            Pipeline::Compute(compute) => compute
                .stages
                .iter()
                .find(|stage| stage.entry_point == "regrouped_bucket_domain_bucket_insert"),
            _ => None,
        })
        .expect("bucket insertion stage");
    assert_eq!(
        insert.dispatch_size,
        DispatchSize::Fixed {
            x: 1,
            y: 512,
            z: 256,
            explicit: false,
        },
        "the invalid fixed z prefix [256][256] must be repartitioned"
    );
}

#[test]
fn ranked_bucket_scatter_grid_strides_an_oversized_axis() {
    use crate::pipeline_descriptor::{DispatchSize, Pipeline};

    let source = r#"
entry strided_bucket_domain(dest: *[4][8]u32) ([4][8]u32, [4]u32, u32) =
  let items = map(|i: i32| (i % 4, u32(i)), iota(4194241)) in
  bucket_scatter_1d(dest, items)
"#;

    let wgsl = lower_ssa_to_wgsl(lower_semantic_egir(
        compile_to_semantic_egir(source),
        LoweringProfile::new(CodegenTarget::Wgsl, SchedulePolicy::Parallel),
    ))
    .expect("grid-stride bucket scatter lowers to WGSL");
    assert!(
        wgsl.contains("loop {"),
        "oversized insertion must contain a grid-stride loop:\n{wgsl}"
    );
    let module = naga::front::wgsl::parse_str(&wgsl)
        .unwrap_or_else(|error| panic!("Naga rejected grid-stride bucket WGSL: {error:?}\n{wgsl}"));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|error| panic!("Naga validation rejected grid-stride bucket WGSL: {error:?}\n{wgsl}"));

    let lowered = compile_thru_spirv(source).expect("grid-stride bucket domain emits SPIR-V");
    let insert = lowered
        .pipeline
        .pipelines
        .iter()
        .find_map(|pipeline| match pipeline {
            Pipeline::Compute(compute) => compute
                .stages
                .iter()
                .find(|stage| stage.entry_point == "strided_bucket_domain_bucket_insert"),
            _ => None,
        })
        .expect("bucket insertion stage");
    assert_eq!(
        insert.dispatch_size,
        DispatchSize::Fixed {
            x: 65_535,
            y: 1,
            z: 1,
            explicit: false,
        }
    );
}

#[test]
fn guarded_bucket_scatter_semantics_discard_count_and_overflow_correctly() {
    use crate::egir::semantic_exec::{execute_bucket_hist, Value};
    use crate::egir::types::{SideEffectKind, Soac, SoacEffect};

    let source = r#"
entry bucket_semantics(dest: *[2][2]i32) ([2][2]i32, [2]u32, u32) =
  let keys: [4]i32 = [-1, 0, 1, 2]
  let values: [3]i32 = [10, 11, 12]
  let items: [4][3](i32, i32) =
    map(|key: i32| map(|value: i32| (key, value), values), keys)
  in bucket_scatter_2d(dest, items)
"#;
    let program = compile_to_segmented_egir(source);
    let operation = program
        .entry_points
        .iter()
        .flat_map(|entry| entry.graph.skeleton.blocks.iter().flat_map(|(_, block)| &block.side_effects))
        .find_map(|effect| match &effect.kind {
            SideEffectKind::Soac(SoacEffect(_, Soac::Hist(operation))) => Some(operation),
            _ => None,
        })
        .expect("guarded bucket histogram");
    assert_eq!(
        operation.inputs.iter().map(|input| input.dimensions.clone()).collect::<Vec<_>>(),
        [vec![0], vec![1]],
        "the semantic oracle must exercise the fused ranked coordinate mapping"
    );
    let values = vec![Value::Int(10), Value::Int(11), Value::Int(12)];
    let empty = || vec![vec![Value::Int(0); 2]; 2];

    let discarded = execute_bucket_hist(
        &program,
        operation,
        &[4, 3],
        &[vec![Value::Int(-1); 4], values.clone()],
        empty(),
    )
    .expect("execute negative-key bucket emissions");
    assert_eq!(discarded.counts, [0, 0]);
    assert!(
        !discarded.overflow,
        "inactive negative-key leaves must not set overflow"
    );
    assert_eq!(discarded.buckets, empty());

    let invalid = execute_bucket_hist(
        &program,
        operation,
        &[4, 3],
        &[vec![Value::Int(2); 4], values.clone()],
        empty(),
    )
    .expect("execute invalid-key bucket emissions");
    assert_eq!(invalid.counts, [0, 0]);
    assert!(
        invalid.overflow,
        "active nonnegative invalid keys must set overflow"
    );

    let full = execute_bucket_hist(
        &program,
        operation,
        &[4, 3],
        &[
            vec![Value::Int(0), Value::Int(1), Value::Int(0), Value::Int(1)],
            values,
        ],
        empty(),
    )
    .expect("execute capacity-limited bucket emissions");
    assert_eq!(full.counts, [6, 6], "counts include leaves beyond capacity");
    assert!(full.overflow, "capacity overflow must be reported");
    assert_eq!(
        full.buckets,
        [
            vec![Value::Int(10), Value::Int(11)],
            vec![Value::Int(10), Value::Int(11)],
        ]
    );
}

#[test]
fn bucket_scatter_rejects_uncomposed_generated_item_storage() {
    let source = r#"
entry dependent_rows(dest: *[4][8]i32) ([4][8]i32, [4]u32, u32) =
  let items: [4][6](i32, i32) =
    map(|bucket: i32|
      let shifted = map(|pair: i32| pair + bucket, iota(6)) in
      map(|value: i32| (value % 4, value), shifted),
      iota(4))
  in bucket_scatter_2d(dest, items)
"#;
    let Err(error) = compile_thru_spirv(source) else {
        panic!("a non-composable generated candidate array must not be materialized")
    };
    assert!(
        error.to_string().contains("requires direct ranked producer composition"),
        "unexpected bucket producer diagnostic: {error}"
    );
}

#[test]
fn multiple_bucket_scatter_operations_share_one_entry_pipeline() {
    let source = r#"
entry two_buckets(
    dest1: *[2][4]u32,
    dest2: *[2][4]u32
) (([2][4]u32, [2]u32, u32), ([2][4]u32, [2]u32, u32)) =
  let first = bucket_scatter_1d(dest1, [(0, 10u32), (1, 11u32)])
  let second = bucket_scatter_1d(dest2, [(0, 20u32), (1, 21u32)])
  in (first, second)
"#;

    compile_thru_spirv(source).expect("multiple destination-passed bucket scatters compile in one entry");
}

#[test]
fn egir_map_scatter_envelope_fuses_and_deduplicates_both_producers() {
    use crate::egir::types::{SideEffectKind, Soac, SoacEffect};

    let source = r#"
entry write(xs: []i32, dest: *[]i32) () =
  let indices = map(|x: i32| x, xs) in
  let values = map(|x: i32| x * 2, xs) in
  let _ = scatter(dest, indices, values) in
  ()
"#;
    let allocated = compile_to_semantic_egir(source);
    let stats = semantic_soac_stats(&allocated);
    assert_eq!(
        stats.seg_maps, 0,
        "both map producers should compose into scatter"
    );
    assert_eq!(stats.hists, 1);
    let input_count = allocated_entries(&allocated)
        .flat_map(|entry| entry.graph.skeleton.blocks.iter().flat_map(|(_, block)| &block.side_effects))
        .find_map(|effect| match &effect.kind {
            SideEffectKind::Soac(SoacEffect(_, Soac::Hist(op))) => Some(op.inputs.len()),
            _ => None,
        })
        .expect("fused SegHist");
    assert_eq!(
        input_count, 1,
        "both scatter channels share the same source array"
    );
    compile_to_spirv(source).expect("map-map-scatter envelope should lower");
}

#[test]
fn semantic_segops_survive_optimization_and_logical_allocation() {
    use crate::egir::types::{SegExtent, SideEffectKind, Soac, SoacEffect};

    let allocated = compile_to_semantic_egir(
        r#"
entry sum(xs: []i32) i32 = reduce(|a: i32, b: i32| a + b, 0, xs)
"#,
    );
    let seg = allocated
        .data
        .stages
        .stages()
        .map(|(_, stage)| stage.body())
        .flat_map(|entry| entry.graph.skeleton.blocks.iter().flat_map(|(_, block)| &block.side_effects))
        .find_map(|effect| {
            let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &effect.kind else {
                return None;
            };
            let screma::SemanticState::Segmented { space, .. } = op.semantic_state() else {
                return None;
            };
            Some((space, op.is_reduce()))
        })
        .expect("SegRed remains present before target lowering");
    assert!(seg.1);
    assert!(matches!(seg.0.dims(), [SegExtent::ResourceLength { .. }]));
    assert!(
        allocated.data.core.resources.len() >= 2,
        "input and output resources are planned logically"
    );
    assert!(allocated.data.stages.external_inputs().any(|input| matches!(
        allocated.data.core.resources[input.storage().data].origin(),
        egir::program::ResourceOrigin::Host { .. }
    )));
    assert!(allocated.data.stages.flows().any(|(_, flow)| {
        flow.is_published()
            && matches!(
                allocated.data.core.resources[flow.storage().data].origin(),
                egir::program::ResourceOrigin::Host { .. }
            )
    }));

    assert!(allocated.semantic_ir().contains("ResourceLength"));

    // Residency allocation is target independent: the semantic operation does
    // not reserve a phase-local partial buffer yet.
    use crate::egir::program::{CompilerResourceKind, ResourceOrigin};
    let partials = allocated
        .data
        .core
        .resources
        .iter()
        .filter(|resource| {
            matches!(
                resource.origin(),
                ResourceOrigin::Compiler { resource: compiler, .. }
                    if compiler.kind == CompilerResourceKind::ReducePartial
            )
        })
        .count();
    assert_eq!(partials, 0, "pre-target allocation has no reduce scratch");
    let planned = egir::plan(allocated, LoweringProfile::PORTABLE).expect("plan parallel reduction");
    assert!(planned.logical_resources().iter().any(|resource| matches!(
        resource.origin(),
        ResourceOrigin::Compiler { resource: compiler, .. }
            if compiler.kind == CompilerResourceKind::ReducePartial && compiler.owner.is_some()
    )));
}

/// Milestone-5 horizontal fusion: the four same-space reductions of
/// `[sum, product, min, max]` — separate lane-local reductions over one input,
/// writing fields of one aggregate output — merge into a single
/// four-accumulator SegOp instead of four one-accumulator ops.
#[test]
fn same_space_reductions_fuse_into_one_multi_accumulator_op() {
    use crate::egir::types::{SideEffectKind, Soac, SoacEffect};

    let allocated = compile_to_semantic_egir(
        r#"
def N: i32 = 256
entry e() [4]f32 =
    let xs = map(|i: i32| f32.i32(i), 0i32 ..< N) in
    [f32.sum(xs), f32.product(xs), f32.minimum(xs), f32.maximum(xs)]
"#,
    );
    let operator_counts: Vec<usize> = allocated_entries(&allocated)
        .flat_map(|entry| entry.graph.skeleton.blocks.iter().flat_map(|(_, block)| &block.side_effects))
        .filter_map(|effect| {
            let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &effect.kind else {
                return None;
            };
            (!op.is_map()).then(|| op.form.scans.len() + op.form.reductions.len())
        })
        .collect();
    assert_eq!(
        operator_counts,
        vec![4],
        "the four same-space reductions fuse into one four-accumulator op"
    );
    let remaining_maps = allocated_entries(&allocated)
        .flat_map(|entry| entry.graph.skeleton.blocks.iter().flat_map(|(_, block)| &block.side_effects))
        .filter(|effect| {
            matches!(&effect.kind, SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) if op.is_map())
        })
        .count();
    assert_eq!(remaining_maps, 0, "the single-consumer map is vertically fused");
    let (pre, operators) = allocated_entries(&allocated)
        .flat_map(|entry| entry.graph.skeleton.blocks.iter().flat_map(|(_, block)| &block.side_effects))
        .find_map(|effect| {
            let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &effect.kind else {
                return None;
            };
            op.is_reduce().then_some((&op.form.pre, op.form.reductions.as_slice()))
        })
        .expect("fused SegRed");
    assert!(
        !pre.is_identity(),
        "all elementwise producer work belongs in the fused Screma pre-lambda"
    );
    for operator in operators {
        assert_eq!(
            allocated.region(operator.operator.seg_body().unwrap().region).unwrap().params.len(),
            2,
            "composed step receives accumulator plus only its routed input"
        );
    }
}

#[test]
fn horizontal_fusion_does_not_cross_an_intervening_effect_token() {
    use crate::egir::types::{SideEffectKind, Soac, SoacEffect};
    let allocated = compile_to_semantic_egir(
        r#"
entry e() [3]i32 =
    let xs = 0i32 ..< 8 in
    let ys = 0i32 ..< 4 in
    [
      reduce(|a: i32, b: i32| a + b, 0, xs),
      reduce(|a: i32, b: i32| a + b, 0, ys),
      reduce(|a: i32, b: i32| if a > b then a else b, -2147483648, xs)
    ]
"#,
    );
    let operator_counts: Vec<_> = allocated_entries(&allocated)
        .flat_map(|entry| entry.graph.skeleton.blocks.iter().flat_map(|(_, block)| &block.side_effects))
        .filter_map(|effect| {
            let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &effect.kind else {
                return None;
            };
            op.is_reduce().then(|| op.form.reductions.len())
        })
        .collect();
    assert_eq!(
        operator_counts,
        [1, 1, 1],
        "equal-space first/third reductions may not leapfrog the middle effect"
    );
}

#[test]
fn fused_accumulators_preserve_distinct_composed_steps_on_shared_input() {
    use crate::egir::types::{SideEffectKind, Soac, SoacEffect};
    let source = r#"
entry e() [2]i32 =
  let xs = map(|i: i32| i + 1, 0i32 ..< 8) in
  let ys = map(|i: i32| i * 2, 0i32 ..< 8) in
  [reduce(|a: i32, b: i32| a + b, 0, xs),
   reduce(|a: i32, b: i32| a + b, 0, ys)]
"#;
    let allocated = compile_to_semantic_egir(source);
    let operators = allocated_entries(&allocated)
        .flat_map(|entry| entry.graph.skeleton.blocks.iter().flat_map(|(_, block)| &block.side_effects))
        .find_map(|effect| {
            let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &effect.kind else {
                return None;
            };
            if !op.is_reduce() {
                return None;
            }
            (op.form.reductions.len() == 2).then_some(op.form.reductions.as_slice())
        })
        .expect("two-accumulator SegRed");
    assert_ne!(
        operators[0].operator.seg_body().unwrap().region,
        operators[1].operator.seg_body().unwrap().region,
        "deduplicated inputs may share a column, but composed map bodies must remain distinct"
    );
    lower_ssa_to_spirv(lower_semantic_egir(allocated, LoweringProfile::PORTABLE))
        .expect("distinct composed steps lower to SPIR-V");
    let base: Vec<i32> = (0..8).collect();
    let xs = egir::semantic_exec::map(&base, |value| value + 1);
    let ys = egir::semantic_exec::map(&base, |value| value * 2);
    assert_eq!(
        [
            egir::semantic_exec::reduce(&xs, 0, |a, b| a + b),
            egir::semantic_exec::reduce(&ys, 0, |a, b| a + b),
        ],
        [36, 56]
    );
}

#[test]
fn target_planning_owns_parallel_work_scratch() {
    use crate::egir::program::{CompilerResourceKind, ResourceOrigin};

    let kinds = |resources: &[egir::program::LogicalResource]| {
        resources
            .iter()
            .filter_map(|resource| match resource.origin() {
                ResourceOrigin::Compiler {
                    resource: compiler, ..
                } => Some(compiler.kind),
                ResourceOrigin::Host { .. } => None,
            })
            .collect::<std::collections::HashSet<_>>()
    };

    let scan =
        compile_to_semantic_egir(" entry prefix(xs: []i32) []i32 = scan(|a: i32, b: i32| a + b, 0, xs)");
    let scan_kinds = kinds(scan.logical_resources());
    assert!(!scan_kinds.contains(&CompilerResourceKind::ScanBlockSums));
    assert!(!scan_kinds.contains(&CompilerResourceKind::ScanBlockOffsets));
    let scan = egir::plan(scan, LoweringProfile::PORTABLE).expect("plan parallel scan");
    let scan_kinds = kinds(scan.logical_resources());
    assert!(scan_kinds.contains(&CompilerResourceKind::ScanBlockSums));
    assert!(scan_kinds.contains(&CompilerResourceKind::ScanBlockOffsets));
    let scan_resource_count = scan
        .logical_resources()
        .iter()
        .filter_map(|resource| match resource.origin() {
            ResourceOrigin::Compiler {
                resource: compiler, ..
            } if matches!(
                compiler.kind,
                CompilerResourceKind::ScanBlockSums | CompilerResourceKind::ScanBlockOffsets
            ) =>
            {
                Some(())
            }
            _ => None,
        })
        .count();
    assert_eq!(scan_resource_count, 2);

    let filter =
        compile_to_semantic_egir(" entry evens(xs: []i32) []i32 = filter(|x: i32| x % 2 == 0, xs)");
    let host_abi = filter
        .logical_resources()
        .iter()
        .filter_map(|resource| resource.host_binding())
        .collect::<Vec<_>>();
    let filter_kinds = kinds(filter.logical_resources());
    assert!(
        filter
            .logical_resources()
            .iter()
            .filter(|resource| matches!(resource.origin(), ResourceOrigin::Host { .. }))
            .count()
            >= 2,
        "the input and returned filter capacity remain host ABI resources"
    );
    assert!(filter_kinds.contains(&CompilerResourceKind::FilterLenCell));
    assert!(!filter_kinds.contains(&CompilerResourceKind::FilterFlags));
    assert!(!filter_kinds.contains(&CompilerResourceKind::FilterOffsets));
    assert!(!filter_kinds.contains(&CompilerResourceKind::FilterScanBlockSums));
    assert!(!filter_kinds.contains(&CompilerResourceKind::FilterScanBlockOffsets));
    let filter = egir::plan(filter, LoweringProfile::PORTABLE).expect("plan parallel filter");
    assert_eq!(
        filter
            .logical_resources()
            .iter()
            .filter_map(|resource| resource.host_binding())
            .collect::<Vec<_>>(),
        host_abi,
        "target scratch allocation must not change host ABI bindings"
    );
    let filter_kinds = kinds(filter.logical_resources());
    assert!(filter_kinds.contains(&CompilerResourceKind::FilterFlags));
    assert!(filter_kinds.contains(&CompilerResourceKind::FilterOffsets));
    assert!(filter_kinds.contains(&CompilerResourceKind::FilterScanBlockSums));
    assert!(filter_kinds.contains(&CompilerResourceKind::FilterScanBlockOffsets));

    let scalar_handoff = compile_to_semantic_egir(
        r#"
entry add_sum(xs: []i32) []i32 =
  let total = reduce(|a: i32, b: i32| a + b, 0, xs) in
  map(|x: i32| x + total, xs)
"#,
    );
    assert!(scalar_handoff.logical_resources().iter().any(|resource| {
        matches!(
            resource.origin(),
            ResourceOrigin::Compiler { resource: compiler, .. }
                if compiler.kind == CompilerResourceKind::ScalarHandoff
        )
    }));

    let single = egir::plan(
        compile_to_semantic_egir(" entry sum(xs: []i32) i32 = reduce(|a: i32, b: i32| a + b, 0, xs)"),
        LoweringProfile::new(CodegenTarget::Portable, SchedulePolicy::Serial),
    )
    .expect("plan sequential reduction");
    assert!(
        !kinds(single.logical_resources()).contains(&CompilerResourceKind::ReducePartial),
        "serial planning must not reserve parallel partial buffers"
    );

    let fallback = compile_to_semantic_egir(
        " entry sum_from(xs: []i32, z: i32) i32 = reduce(|a: i32, b: i32| a + b, z, xs)",
    );
    let fallback =
        egir::plan(fallback, LoweringProfile::PORTABLE).expect("plan reduction with a runtime neutral");
    assert!(
        !kinds(fallback.logical_resources()).contains(&CompilerResourceKind::ReducePartial),
        "a reduction rejected before mutation must not retain speculative scratch"
    );
    assert_eq!(
        fallback.physical_kernels().phases().map(|phase| phase.label.as_str()).collect::<Vec<_>>(),
        ["serial_compute"],
        "an unsupported parallel recipe must preserve serial execution"
    );
}

#[test]
fn selected_recipes_allocate_exact_ordered_scratch() {
    use crate::egir::program::{CompilerResourceKind as Kind, LogicalSize, ResourceOrigin};

    let planned_scratch = |resources: &[egir::program::LogicalResource]| {
        resources
            .iter()
            .filter_map(|resource| match resource.origin() {
                ResourceOrigin::Compiler {
                    resource: compiler, ..
                } if matches!(
                    compiler.kind,
                    Kind::ReducePartial
                        | Kind::ScanBlockSums
                        | Kind::ScanBlockOffsets
                        | Kind::FilterFlags
                        | Kind::FilterOffsets
                        | Kind::FilterScanBlockSums
                        | Kind::FilterScanBlockOffsets
                ) =>
                {
                    Some((
                        compiler.kind,
                        compiler.owner,
                        compiler.slot,
                        resource.size().expect("compiler scratch has a concrete size").clone(),
                    ))
                }
                _ => None,
            })
            .collect::<Vec<_>>()
    };

    let scalar_reduce = egir::plan(
        compile_to_semantic_egir(" entry sum(xs: []i32) i32 = reduce(|a: i32, b: i32| a + b, 0, xs)"),
        LoweringProfile::PORTABLE,
    )
    .expect("plan scalar reduction");
    let scratch = planned_scratch(scalar_reduce.logical_resources());
    assert_eq!(scratch.len(), 1);
    assert_eq!(scratch[0].0, Kind::ReducePartial);
    assert!(scratch[0].1.is_some());
    assert_eq!(scratch[0].2, 0);
    assert_eq!(scratch[0].3, LogicalSize::SameAsDispatch { elem_bytes: 4 });

    let multi_reduce = compile_to_semantic_egir(
        r#"
entry sums() (i32, i32) =
  let xs = map(|i: i32| i + 1, 0i32 ..< 8) in
  let ys = map(|i: i32| i * 2, 0i32 ..< 8) in
  (reduce(|a: i32, b: i32| a + b, 0, xs),
   reduce(|a: i32, b: i32| a + b, 0, ys))
"#,
    );
    let multi_reduce =
        egir::plan(multi_reduce, LoweringProfile::PORTABLE).expect("plan multi-accumulator reduction");
    let scratch = planned_scratch(multi_reduce.logical_resources());
    assert_eq!(scratch.len(), 2);
    assert_eq!(
        multi_reduce.physical_kernels().phases().map(|phase| phase.label.as_str()).collect::<Vec<_>>(),
        ["reduce_phase1", "reduce_combine", "reduce_combine"]
    );
    let owner = scratch[0].1.expect("scratch has an operation owner");
    assert_eq!(
        scratch
            .iter()
            .map(|(kind, candidate, slot, size)| (*kind, *candidate, *slot, size.clone()))
            .collect::<Vec<_>>(),
        [
            (
                Kind::ReducePartial,
                Some(owner),
                0,
                LogicalSize::SameAsDispatch { elem_bytes: 4 },
            ),
            (
                Kind::ReducePartial,
                Some(owner),
                1,
                LogicalSize::SameAsDispatch { elem_bytes: 4 },
            ),
        ]
    );

    let scan = egir::plan(
        compile_to_semantic_egir(" entry prefix(xs: []i32) []i32 = scan(|a: i32, b: i32| a + b, 0, xs)"),
        LoweringProfile::PORTABLE,
    )
    .expect("plan scan");
    let scratch = planned_scratch(scan.logical_resources());
    let owner = scratch[0].1.expect("scratch has an operation owner");
    assert_eq!(
        scratch,
        [
            (
                Kind::ScanBlockSums,
                Some(owner),
                0,
                LogicalSize::SameAsDispatch { elem_bytes: 4 },
            ),
            (
                Kind::ScanBlockOffsets,
                Some(owner),
                1,
                LogicalSize::SameAsDispatch { elem_bytes: 4 },
            ),
        ]
    );

    let filter = egir::plan(
        compile_to_semantic_egir(" entry evens(xs: []i32) []i32 = filter(|x: i32| x % 2 == 0, xs)"),
        LoweringProfile::PORTABLE,
    )
    .expect("plan runtime filter");
    let scratch = planned_scratch(filter.logical_resources());
    assert_eq!(scratch.len(), 4);
    let owner = scratch[0].1.expect("scratch has an operation owner");
    assert_eq!(
        scratch.iter().map(|item| (item.0, item.1, item.2)).collect::<Vec<_>>(),
        [
            (Kind::FilterFlags, Some(owner), 0),
            (Kind::FilterOffsets, Some(owner), 1),
            (Kind::FilterScanBlockSums, Some(owner), 2),
            (Kind::FilterScanBlockOffsets, Some(owner), 3),
        ]
    );
    assert!(scratch[..2].iter().all(|item| matches!(
        item.3,
        LogicalSize::LikeResource {
            elem_bytes: 4,
            src_elem_bytes: 4,
            ..
        }
    )));
    assert!(scratch[2..].iter().all(|item| item.3 == LogicalSize::FixedBytes(4 * 64 * 4)));
}

#[test]
fn parallel_reduce_and_scan_recipe_shapes_are_stable() {
    use crate::egir::parallelize::KernelDomain;

    let reduce = compile_thru_ssa(" entry sum(xs: []i32) i32 = reduce(|a: i32, b: i32| a + b, 0, xs)")
        .expect("parallel reduction reaches SSA");
    let phases = reduce.global_context.physical_kernels.phases().collect::<Vec<_>>();
    assert_eq!(
        phases.iter().map(|phase| phase.label.as_str()).collect::<Vec<_>>(),
        ["reduce_phase1", "reduce_combine"]
    );
    assert_eq!(phases[0].workgroup_size, (64, 1, 1));
    assert_eq!(phases[1].workgroup_size, (256, 1, 1));
    assert!(matches!(
        phases[1].domain,
        KernelDomain::Fixed { x: 1, y: 1, z: 1 }
    ));

    let scan = compile_thru_ssa(" entry prefix(xs: []i32) []i32 = scan(|a: i32, b: i32| a + b, 0, xs)")
        .expect("parallel scan reaches SSA");
    let phases = scan.global_context.physical_kernels.phases().collect::<Vec<_>>();
    assert_eq!(
        phases.iter().map(|phase| phase.label.as_str()).collect::<Vec<_>>(),
        ["scan_phase1", "scan_block", "scan_apply_offsets"]
    );
    assert_eq!(
        phases.iter().map(|phase| phase.workgroup_size).collect::<Vec<_>>(),
        [(64, 1, 1), (1, 1, 1), (64, 1, 1)]
    );
    assert!(matches!(
        phases[1].domain,
        KernelDomain::Fixed { x: 1, y: 1, z: 1 }
    ));
}

#[test]
fn chunked_recipes_accept_empty_small_uneven_and_unsigned_ranges() {
    let cases = [
        (
            " entry empty() i32 = reduce(|a: i32, b: i32| a + b, 0, 0i32 ..< 0)",
            "reduce_phase1",
        ),
        (
            " entry small() []i32 = scan(|a: i32, b: i32| a + b, 0, 0i32 ..< 7)",
            "scan_phase1",
        ),
        (
            " entry uneven() i32 = reduce(|a: i32, b: i32| a + b, 0, 0i32 ..< 70)",
            "reduce_phase1",
        ),
        (
            " entry unsigned() u32 = reduce(|a: u32, b: u32| a + b, 0u32, 0u32 ..< 70u32)",
            "reduce_phase1",
        ),
    ];

    for (source, expected) in cases {
        let lowered = compile_thru_ssa(source).expect("edge-domain recipe reaches SSA");
        assert_eq!(
            lowered.global_context.physical_kernels.phases().next().map(|phase| phase.label.as_str()),
            Some(expected),
            "edge-domain shape must retain its selected parallel recipe"
        );
    }
}

#[test]
fn associative_noncommutative_reduce_and_scan_keep_parallel_ordered_recipes() {
    // Dihedral-group composition encoded as `rotation + 3 * reflected`.
    // It is associative with identity 0, but reflections and rotations do not
    // commute, making it a useful ordering characterization in one scalar.
    let compose = |left: &i32, right: &i32| {
        let (left_rotation, left_reflected) = (left % 3, left / 3);
        let (right_rotation, right_reflected) = (right % 3, right / 3);
        let rotation = if left_reflected == 0 {
            (left_rotation + right_rotation) % 3
        } else {
            (left_rotation + 3 - right_rotation) % 3
        };
        rotation + 3 * ((left_reflected + right_reflected) % 2)
    };
    let values = [3, 1];
    let forward = egir::semantic_exec::reduce(&values, 0, compose);
    let mut reversed = values;
    reversed.reverse();
    assert_ne!(
        forward,
        egir::semantic_exec::reduce(&reversed, 0, compose),
        "the characterization operator must actually be non-commutative"
    );

    let reduce = compile_thru_ssa(
        r#"
entry compose_all(xs: []i32) i32 =
  reduce(
    |left: i32, right: i32|
      let left_rotation = left % 3
      let left_reflected = left / 3
      let right_rotation = right % 3
      let right_reflected = right / 3
      let rotation = if left_reflected == 0
        then (left_rotation + right_rotation) % 3
        else (left_rotation + 3 - right_rotation) % 3 in
      rotation + 3 * ((left_reflected + right_reflected) % 2),
    0,
    xs)
"#,
    )
    .expect("ordered tuple reduction reaches SSA");
    assert_eq!(
        reduce
            .global_context
            .physical_kernels
            .phases()
            .map(|phase| phase.label.as_str())
            .collect::<Vec<_>>(),
        ["reduce_phase1", "reduce_combine"]
    );

    let scan = compile_thru_ssa(
        r#"
entry compose_prefix(xs: []i32) []i32 =
  scan(
    |left: i32, right: i32|
      let left_rotation = left % 3
      let left_reflected = left / 3
      let right_rotation = right % 3
      let right_reflected = right / 3
      let rotation = if left_reflected == 0
        then (left_rotation + right_rotation) % 3
        else (left_rotation + 3 - right_rotation) % 3 in
      rotation + 3 * ((left_reflected + right_reflected) % 2),
    0,
    xs)
"#,
    )
    .expect("ordered tuple scan reaches SSA");
    assert_eq!(
        scan.global_context.physical_kernels.phases().map(|phase| phase.label.as_str()).collect::<Vec<_>>(),
        ["scan_phase1", "scan_block", "scan_apply_offsets"]
    );
}

#[test]
fn runtime_filter_lowers_to_flag_scan_scatter_pipeline() {
    use crate::builtins::catalog;
    use crate::egir::parallelize::KernelDomain;
    use crate::op::OpTag;
    use crate::ssa::types::InstKind;

    let r4 = r#"
entry r(xs: []u32) ?k. [k]u32 = filter(|x| x < 100u32, xs)
"#;
    let converted = compile_thru_ssa(r4).expect("runtime filter reaches SSA");
    let phases: Vec<_> = converted.global_context.physical_kernels.phases().collect();
    assert_eq!(phases.len(), 5);
    assert_eq!(phases[0].entry_point, "r_filter_flags");
    assert_eq!(phases[1].entry_point, "r_filter_scan");
    assert_eq!(phases[2].entry_point, "r_filter_scan_phase2_scan_sums");
    assert_eq!(phases[3].entry_point, "r_filter_scan_phase3_add_offsets");
    assert_eq!(phases[4].entry_point, "r");
    assert!(matches!(phases[0].domain, KernelDomain::ResourceElements { .. }));
    // Scan phases 1 and 3 run the same fixed worker grid: phase 1 records one
    // block sum per worker, and phase 3 uses that same worker id to load and
    // apply the block's exclusive offset. Dispatching either phase per element
    // would give the two phases different chunk ownership.
    assert!(matches!(
        phases[1].domain,
        KernelDomain::Fixed {
            x: egir::parallelize::tests::FILTER_SCAN_GROUPS,
            y: 1,
            z: 1
        }
    ));
    assert_eq!(
        phases[1].workgroup_size,
        (egir::parallelize::tests::REDUCE_PHASE1_WIDTH, 1, 1)
    );
    assert!(matches!(
        phases[2].domain,
        KernelDomain::Fixed { x: 1, y: 1, z: 1 }
    ));
    assert_eq!(phases[3].domain, phases[1].domain);
    assert_eq!(phases[3].workgroup_size, phases[1].workgroup_size);
    assert!(matches!(phases[4].domain, KernelDomain::ResourceElements { .. }));

    let thread_id = catalog().known().thread_id;
    for name in [
        "r_filter_flags",
        "r_filter_scan",
        "r_filter_scan_phase3_add_offsets",
        "r",
    ] {
        let entry = converted
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
    compile_thru_spirv(r4).expect("three-stage filter emits SPIR-V");

    let r5 = r#"
entry r(xs: []u32) (?k. [k]u32, [1]u32) =
  let v = filter(|x| x < 100u32, xs)
  let n = length(v) in
  (v, [u32(n)])
"#;
    let lowered = compile_thru_spirv(r5).expect("filter count path emits SPIR-V");
    let compute = lowered
        .pipeline
        .pipelines
        .iter()
        .find_map(|pipeline| match pipeline {
            pipeline_descriptor::Pipeline::Compute(compute) => Some(compute),
            _ => None,
        })
        .expect("filter compute pipeline");
    assert_eq!(compute.stages.len(), 5);
    assert_eq!(compute.stages[1].entry_point, "r_filter_scan");
    assert_eq!(compute.stages[2].entry_point, "r_filter_scan_phase2_scan_sums");
    assert_eq!(compute.stages[3].entry_point, "r_filter_scan_phase3_add_offsets");
    assert_eq!(compute.stages[3].dispatch_size, compute.stages[1].dispatch_size);
    assert_eq!(compute.stages[3].workgroup_size, compute.stages[1].workgroup_size);
}

#[test]
fn mixed_map_filter_outputs_keep_complete_phase_family() {
    use crate::egir::parallelize::OutputRouteProjection;
    use crate::egir::program::OutputSlotId;

    let source = r#"
entry mixed() ([]i32, []i32) =
  let mapped = map(|i| i, iota(1))
  let compacted = filter(|i| true, iota(1)) in
  (mapped, compacted)
"#;
    let converted = compile_thru_ssa(source).expect("mixed map/filter reaches SSA");
    let phases = converted.global_context.physical_kernels.phases().collect::<Vec<_>>();
    assert_eq!(
        phases.iter().map(|phase| phase.label.as_str()).collect::<Vec<_>>(),
        [
            "serial_compute",
            "filter_flags",
            "filter_scan",
            "filter_combine",
            "filter_apply_offsets",
            "filter_scatter",
        ]
    );
    assert_eq!(
        phases[0].output_routes,
        [OutputRouteProjection {
            semantic_slot: OutputSlotId(0),
            physical_slot: OutputSlotId(0),
        }]
    );
    assert_eq!(
        phases[5].output_routes,
        [OutputRouteProjection {
            semantic_slot: OutputSlotId(1),
            physical_slot: OutputSlotId(0),
        }]
    );
    assert!(!compile_thru_spirv(source).expect("mixed map/filter emits SPIR-V").spirv.is_empty());
}

fn assert_naga_accepts_spirv(words: &[u32]) {
    let bytes = words.iter().flat_map(|word| word.to_le_bytes()).collect::<Vec<_>>();
    let module = naga::front::spv::parse_u8_slice(
        &bytes,
        &naga::front::spv::Options {
            strict_capabilities: false,
            ..Default::default()
        },
    )
    .unwrap_or_else(|error| panic!("Naga rejected generated SPIR-V: {error:?}"));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|error| panic!("Naga validation rejected generated SPIR-V: {error:?}"));
}

fn spirv_has_builtin(words: &[u32], builtin: spirv::BuiltIn) -> bool {
    let mut index = 5usize;
    while index < words.len() {
        let instruction = words[index];
        let word_count = (instruction >> 16) as usize;
        let opcode = instruction & 0xffff;
        if opcode == 71
            && word_count >= 4
            && words[index + 2] == spirv::Decoration::BuiltIn as u32
            && words[index + 3] == builtin as u32
        {
            return true;
        }
        if word_count == 0 {
            break;
        }
        index += word_count;
    }
    false
}

#[test]
fn filter_over_iota_emits_well_typed_length_and_index_operations() {
    let lowered = compile_thru_spirv(
        r#"
entry compact_i32() []i32 =
  filter(|i| i % 2 == 0, iota(128))
"#,
    )
    .expect("filter over iota emits SPIR-V");
    assert_naga_accepts_spirv(&lowered.spirv);
}

#[test]
fn filter_iota_scan_apply_offsets_reuses_phase1_worker_grid() {
    use crate::pipeline_descriptor::{DispatchSize, Pipeline};

    let lowered = compile_thru_spirv(
        r#"
entry compact_iota() []i32 =
  filter(|i| i % 2 == 0, iota(39592))
"#,
    )
    .expect("large filter over iota emits SPIR-V");
    let compute = lowered
        .pipeline
        .pipelines
        .iter()
        .find_map(|pipeline| match pipeline {
            Pipeline::Compute(compute) => Some(compute),
            _ => None,
        })
        .expect("filter compute pipeline");
    let phase1 = compute
        .stages
        .iter()
        .find(|stage| stage.entry_point == "compact_iota_filter_scan")
        .expect("filter scan phase 1");
    let phase3 = compute
        .stages
        .iter()
        .find(|stage| stage.entry_point == "compact_iota_filter_scan_phase3_add_offsets")
        .expect("filter scan phase 3");

    assert_eq!(phase3.workgroup_size, phase1.workgroup_size);
    assert_eq!(phase3.dispatch_size, phase1.dispatch_size);
    assert!(matches!(
        phase1.dispatch_size,
        DispatchSize::Fixed {
            x: egir::parallelize::tests::FILTER_SCAN_GROUPS,
            y: 1,
            z: 1,
            explicit: true,
        }
    ));
}

#[test]
fn map_over_filtered_array_emits_well_typed_dynamic_extent() {
    let lowered = compile_thru_spirv(
        r#"
entry compact_then_map() ([]i32, [1]u32) =
  let visible_indices = filter(|i| i % 2 == 0, iota(128))
  let live_props = map(|i| i + 1, visible_indices) in
  (live_props, [u32(length(visible_indices))])
"#,
    )
    .expect("filter survivors feed a map");
    assert_naga_accepts_spirv(&lowered.spirv);
}

#[test]
fn filter_then_map_publishes_runtime_array_handoff() {
    use crate::pipeline_descriptor::{Binding, BufferLen, BufferUsage, DispatchLen, DispatchSize};

    let lowered = compile_thru_spirv(
        r#"
entry filter_then_map() []i32 =
  let kept = filter(|i| i % 2 == 0, iota(4096)) in
  map(|i| i + 1, kept)
"#,
    )
    .expect("filter survivors feed a separately scheduled map");
    let pipeline = scalar_prelude_pipeline(&lowered, "filter_then_map");
    let consumer_index = pipeline
        .stages
        .iter()
        .position(|stage| stage.entry_point == "filter_then_map")
        .expect("the public map stage is published");
    let consumer = &pipeline.stages[consumer_index];
    assert!(consumer_index > 0, "the runtime-array producer precedes the map");

    let handoffs = pipeline
        .bindings
        .iter()
        .enumerate()
        .filter(|(binding, descriptor)| {
            consumer.reads.contains(binding)
                && matches!(
                    descriptor,
                    Binding::StorageBuffer {
                        usage: BufferUsage::Intermediate,
                        ..
                    }
                )
                && pipeline.stages[..consumer_index].iter().any(|stage| stage.writes.contains(binding))
        })
        .collect::<Vec<_>>();
    assert_eq!(
        handoffs.len(),
        2,
        "the producer passes one data buffer and one logical-length cell to the map"
    );
    let data = handoffs
        .iter()
        .find(|(_, descriptor)| {
            matches!(
                descriptor,
                Binding::StorageBuffer {
                    length: Some(BufferLen::Fixed { bytes: 16_384 }),
                    ..
                }
            )
        })
        .expect("the runtime-array data buffer has input capacity");
    assert!(handoffs.iter().any(|(_, descriptor)| {
        matches!(
            descriptor,
            Binding::StorageBuffer {
                length: Some(BufferLen::Fixed { bytes: 4 }),
                ..
            }
        )
    }));
    let Binding::StorageBuffer {
        set: data_set,
        binding: data_binding,
        ..
    } = data.1
    else {
        unreachable!()
    };
    assert!(matches!(
        consumer.dispatch_size,
        DispatchSize::DerivedFrom {
            len: DispatchLen::InputBinding { set, binding, elem_bytes: 4 },
            ..
        } if set == *data_set && binding == *data_binding
    ));
    assert!(
        !spirv_entry_reaches_loop(&lowered.spirv, &consumer.entry_point),
        "the map entry must not replay the filter's serial loop"
    );
}

#[test]
fn filter_after_serial_prefix_detaches_its_parallel_producer() {
    let lowered = compile_thru_spirv(
        r#"
entry filter_after_serial_prefix(xs: []i32) ([1]i32, []i32, i32) =
  let prefix =
    loop acc = 0 for k < 4 do
      if xs[k] > 0 then acc + xs[k] else acc
  let kept = filter(|i| i % 2 == 0, iota(4096))
  let mapped = map(|i| i + 1, kept) in
  ([prefix], mapped, length(kept))
"#,
    )
    .expect("a continuation-block filter is materialized before its map consumer");
    let pipeline = scalar_prelude_pipeline(&lowered, "filter_after_serial_prefix");
    let consumer_index = pipeline
        .stages
        .iter()
        .position(|stage| stage.entry_point == "filter_after_serial_prefix")
        .expect("the public consumer stage is published");
    let producer_stages = &pipeline.stages[..consumer_index];
    let flags = producer_stages
        .iter()
        .enumerate()
        .find_map(|(index, stage)| {
            let output_feeds_later_phase = stage.writes.iter().any(|binding| {
                producer_stages[index + 1..].iter().any(|later| later.reads.contains(binding))
            });
            (stage.reads.is_empty() && output_feeds_later_phase).then_some(stage)
        })
        .expect("the filter flags phase precedes the remaining producer phases");
    assert!(
        !spirv_entry_reaches_loop(&lowered.spirv, &flags.entry_point),
        "the independent serial prefix must not be copied into the filter flags phase"
    );
}

#[test]
fn fixed_output_serial_prefix_is_not_cloned_into_parallel_output_stage() {
    use crate::pipeline_descriptor::{Binding, BufferUsage};

    let lowered = compile_thru_spirv(
        r#"
entry filter_after_serial_prefix(xs: []i32) ([1]i32, []i32, i32) =
  let prefix =
    loop acc = 0 for k < 4 do
      if xs[k] > 0 then acc + xs[k] else acc
  let kept = filter(|i| i % 2 == 0, iota(4096))
  let mapped = map(|i| i + 1, kept) in
  ([prefix], mapped, length(kept))
"#,
    )
    .expect("serial prefix and filtered map compile");
    let pipeline = scalar_prelude_pipeline(&lowered, "filter_after_serial_prefix");
    let output_bindings = pipeline
        .bindings
        .iter()
        .enumerate()
        .filter_map(|(index, binding)| {
            matches!(
                binding,
                Binding::StorageBuffer {
                    usage: BufferUsage::Output,
                    ..
                }
            )
            .then_some(index)
        })
        .collect::<std::collections::HashSet<_>>();
    let input_bindings = pipeline
        .bindings
        .iter()
        .enumerate()
        .filter_map(|(index, binding)| {
            matches!(
                binding,
                Binding::StorageBuffer {
                    usage: BufferUsage::Input,
                    ..
                }
            )
            .then_some(index)
        })
        .collect::<std::collections::HashSet<_>>();
    let prefix_stages = pipeline
        .stages
        .iter()
        .filter(|stage| stage.reads.iter().any(|binding| input_bindings.contains(binding)))
        .collect::<Vec<_>>();
    assert_eq!(
        prefix_stages.len(),
        1,
        "the independent prefix input must be read by exactly one stage"
    );
    assert!(
        is_singleton_stage(prefix_stages[0]),
        "the stage computing the fixed-output prefix must execute once"
    );
    assert!(
        spirv_entry_reaches_loop(&lowered.spirv, &prefix_stages[0].entry_point),
        "the singleton fixed-output stage retains the prefix loop"
    );
    let parallel_output_stages = pipeline
        .stages
        .iter()
        .filter(|stage| {
            !is_singleton_stage(stage)
                && stage.writes.iter().any(|binding| output_bindings.contains(binding))
        })
        .collect::<Vec<_>>();
    assert!(
        !parallel_output_stages.is_empty(),
        "the mapped output must retain a parallel writer"
    );
    for stage in parallel_output_stages {
        assert!(
            !spirv_entry_reaches_loop(&lowered.spirv, &stage.entry_point),
            "parallel output stage `{}` replays the independent serial prefix",
            stage.entry_point
        );
    }
}

#[test]
fn widened_filter_output_uses_output_element_size() {
    use crate::pipeline_descriptor::{BufferLen, BufferUsage, Pipeline};

    let lowered = compile_thru_spirv(
        r#"
entry r(bidx: []u32) ?k. [k]vec4f32 =
  let cand = map(|s| let i = i32(s) in @[f32(i), 0.0, f32(i), 1.0], bidx) in
  filter(|c| c.z > 0.0, cand)
"#,
    )
    .expect("widening map-filter compiles");
    let output_length = lowered.pipeline.pipelines.iter().find_map(|pipeline| match pipeline {
        Pipeline::Compute(compute) => compute.bindings.iter().find_map(|binding| match binding {
            pipeline_descriptor::Binding::StorageBuffer {
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
/// Most shapes are subsumed upstream: same-domain sibling consumers fuse in
/// semantic EGIR, and point reads / nested captures are
/// materialized. `reduce_then_map` — an array consumed by a reduce *and* by a
/// later map that depends on the reduce's scalar result — cannot fuse (a true
/// producer→consumer dependency forces the map after the reduce). Phase M
/// therefore turns that producer into one logical storage prepass, and both
/// consumers read the same buffer. This test pins that no live multi-consumer
/// SegMap remains and that the shared prepass replaces local `Materialize`s.
#[test]
fn multi_consumer_producer_survival_is_characterized() {
    use crate::egir::semantic_graph::SemanticDependencyKind;
    use crate::egir::types::{SideEffectKind, Soac, SoacEffect};
    use std::collections::HashMap;

    fn multi_consumer_producers(src: &str) -> usize {
        let allocated = compile_to_semantic_egir(src);
        let seg_maps: std::collections::HashSet<_> = allocated_entries(&allocated)
            .flat_map(|entry| {
                entry.graph.skeleton.blocks.iter().flat_map(move |(_, block)| {
                    block.side_effects.iter().filter_map(move |effect| match &effect.kind {
                        SideEffectKind::Soac(SoacEffect(id, Soac::Screma(op))) if op.is_map() => Some(*id),
                        _ => None,
                    })
                })
            })
            .collect();
        let mut consumers: HashMap<_, usize> = HashMap::new();
        for dep in allocated_entries(&allocated)
            .flat_map(|entry| egir::semantic_graph::graph_dependencies(&entry.graph))
        {
            if matches!(dep.kind, SemanticDependencyKind::Value) && seg_maps.contains(&dep.producer) {
                *consumers.entry(dep.producer).or_default() += 1;
            }
        }
        consumers.values().filter(|count| **count >= 2).count()
    }

    // ys read by two elementwise maps (combined via zip).
    let two_maps = r#"
def N: i32 = 8
entry e() [8]i32 =
    let ys = map(|i: i32| i + 1, 0i32 ..< N) in
    let a = map(|y: i32| y * 2, ys) in
    let b = map(|y: i32| y + 100, ys) in
    map(|p: (i32, i32)| p.0 + p.1, zip(a, b))
"#;
    // ys read by a full reduce and then by a map (different consumer kinds).
    let reduce_then_map = r#"
def N: i32 = 8
entry e() [8]i32 =
    let ys = map(|i: i32| i + 1, 0i32 ..< N) in
    let s = reduce(|a: i32, b: i32| a + b, 0, ys) in
    map(|y: i32| y + s, ys)
"#;
    // ys read by two reductions with different operators (sum and max).
    let two_reduces = r#"
def N: i32 = 8
entry e() i32 =
    let ys = map(|i: i32| i * i, 0i32 ..< N) in
    reduce(|a: i32, b: i32| a + b, 0, ys) * reduce(|a: i32, b: i32| if a > b then a else b, 0, ys)
"#;
    // ys read by a full reduce and by a point index.
    let reduce_and_index = r#"
def N: i32 = 8
entry e() i32 =
    let ys = map(|i: i32| i * i, 0i32 ..< N) in
    reduce(|a: i32, b: i32| a + b, 0, ys) + ys[0]
"#;
    // ys reduced inside a map over a *different* domain — read once per outer
    // iteration; the classic "materialize once, reuse" case.
    let reduce_in_nested_map = r#"
def N: i32 = 8
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

    let allocated = compile_to_semantic_egir(reduce_then_map);
    use crate::egir::program::{CompilerResourceKind, ResourceOrigin};
    let shared: Vec<_> = allocated
        .logical_resources()
        .iter()
        .filter(|resource| {
            matches!(
                resource.origin(),
                ResourceOrigin::Compiler { resource: compiler, .. }
                    if compiler.kind == CompilerResourceKind::MultiConsumerArray
            )
        })
        .collect();
    assert_eq!(
        shared.len(),
        1,
        "the surviving producer owns one shared logical buffer"
    );
    let shared_resource = shared[0].id();
    let shared_stages = allocated
        .data
        .stages
        .stages()
        .filter(|(_, stage)| {
            stage.origin().generated_kind() == Some(egir::program::GeneratedStageKind::SharedArray)
        })
        .collect::<Vec<_>>();
    assert_eq!(
        shared_stages.len(),
        1,
        "shared producer is represented by one typed stage"
    );
    let (producer_stage, producer) = shared_stages[0];
    let producer_name = producer.body().name.clone();
    assert!(
        producer_name.starts_with("e_materialize_shared_"),
        "generated stage retains a diagnostic role suffix"
    );
    let ResourceOrigin::Compiler { .. } = shared[0].origin() else {
        unreachable!("shared resource must be compiler-owned")
    };
    let flow = allocated
        .data
        .stages
        .flows()
        .find_map(|(_, flow)| (flow.storage().data == shared_resource).then_some(flow))
        .expect("shared resource has a staged flow");
    assert_eq!(flow.producer(), producer_stage);
    let consumer = flow
        .consumers()
        .iter()
        .copied()
        .find(|consumer| {
            allocated
                .data
                .stages
                .stage(*consumer)
                .is_some_and(|stage| matches!(stage.origin(), egir::program::StageOrigin::Authored))
        })
        .expect("shared array remains an input of the source entry");
    assert_eq!(allocated.data.stages.stage(consumer).unwrap().body().name, "e");
    let lowered = lower_semantic_egir(allocated, LoweringProfile::PORTABLE);
    let mir = ssa::print::format_program(&lowered);
    assert_eq!(
        mir.matches("materialize ").count(),
        0,
        "consumers read the shared storage prepass rather than copying a composite per consumer"
    );
    let stages: Vec<_> =
        lowered.global_context.physical_kernels.phases().map(|phase| phase.entry_point.as_str()).collect();
    assert_eq!(stages.first(), Some(&producer_name.as_str()));
    assert_eq!(stages.last(), Some(&"e"));
    assert!(stages.iter().any(|stage| stage.contains("prepass_scalar")));
    let phases: Vec<_> = lowered.global_context.physical_kernels.phases().collect();
    assert!(phases[0].resources.iter().any(|resource| {
        resource.resource == shared_resource && resource.access == ResourceAccess::Write
    }));
    assert!(phases
        .last()
        .unwrap()
        .resources
        .iter()
        .any(|resource| resource.resource == shared_resource && resource.access.reads()));
    assert!(phases.last().unwrap().dependencies.contains(&phases[0].id));
    let second = lower_semantic_egir(
        compile_to_semantic_egir(reduce_then_map),
        LoweringProfile::PORTABLE,
    );
    assert_eq!(
        serde_json::to_string(&lowered.global_context.pipeline).unwrap(),
        serde_json::to_string(&second.global_context.pipeline).unwrap(),
        "shared materialization descriptor is deterministic"
    );
    let ys: Vec<i32> = (0..8).map(|value| value + 1).collect();
    let sum = egir::semantic_exec::reduce(&ys, 0, |a, b| a + b);
    assert_eq!(
        egir::semantic_exec::map(&ys, |value| value + sum),
        [37, 38, 39, 40, 41, 42, 43, 44]
    );

    let single = lower_semantic_egir(
        compile_to_semantic_egir(reduce_then_map),
        LoweringProfile::new(CodegenTarget::Portable, SchedulePolicy::Serial),
    );
    let single_phases: Vec<_> = single.global_context.physical_kernels.phases().collect();
    assert_eq!(
        single_phases.len(),
        3,
        "shared array, serial scalar reduction, and source entry"
    );
    assert!(matches!(
        single_phases[0].domain,
        egir::parallelize::KernelDomain::Fixed { x: 1, y: 1, z: 1 }
    ));

    let wgsl = lower_ssa_to_wgsl(lower_semantic_egir(
        compile_to_semantic_egir(reduce_then_map),
        LoweringProfile::new(CodegenTarget::Wgsl, SchedulePolicy::Parallel),
    ))
    .expect("shared materialization lowers to WGSL");
    assert!(wgsl.contains("e_materialize_shared"));
}

#[test]
fn tuple_outputs_with_independent_map_chains_lower_after_semantic_fusion() {
    compile_thru_spirv(
        r#"
def N: i32 = 8
def f(x: i32) i32 = x + 1
def g(x: i32) i32 = x * 2
def h(x: i32) i32 = x - 3
def k(x: i32) i32 = x * 5
entry e() ([8]i32, [8]i32) =
  (map(f, map(g, 0i32 ..< N)), map(h, map(k, 0i32 ..< N)))
"#,
    )
    .expect("two tuple fields with independent map chains lower to SPIR-V");
}

#[test]
fn terminal_schedule_and_descriptor_are_atomic_and_deterministic() {
    let source = r#"
entry sum(xs: []i32) i32 = reduce(|a: i32, b: i32| a + b, 0, xs)
"#;
    let allocated = compile_to_semantic_egir(source);
    for pipeline in &allocated.data.core.pipeline.pipelines {
        if let pipeline_descriptor::Pipeline::Compute(compute) = pipeline {
            assert!(
                compute.bindings.is_empty(),
                "bindings publish only at terminal lowering"
            );
        }
    }
    let first = lower_semantic_egir(allocated, LoweringProfile::PORTABLE);
    let phases: Vec<_> = first.global_context.physical_kernels.phases().collect();
    assert!(phases.len() >= 2, "parallel reduction owns at least two phases");
    assert!(phases.iter().skip(1).any(|phase| !phase.dependencies.is_empty()));
    assert!(phases.iter().all(|phase| !phase.resources.is_empty()));

    let second = compile_thru_ssa(source).expect("second lowering");
    assert_eq!(
        serde_json::to_string(&first.global_context.pipeline).unwrap(),
        serde_json::to_string(&second.global_context.pipeline).unwrap(),
        "descriptor publication is deterministic"
    );
}

#[test]
fn serial_is_a_terminal_schedule_policy() {
    let source = r#"
entry sum(xs: []i32) i32 = reduce(|a: i32, b: i32| a + b, 0, xs)
"#;
    let allocated = compile_to_semantic_egir(source);

    let lowered = lower_semantic_egir(
        allocated,
        LoweringProfile::new(CodegenTarget::Portable, SchedulePolicy::Serial),
    );
    assert_eq!(lowered.global_context.physical_kernels.phases().count(), 1);
    assert!(!lowered.entry_points.iter().any(|entry| entry.name.contains("phase2")));
}
#[test]
fn unified_root_array_program_compiles_through_spirv() {
    let lowered = compile_thru_spirv("entry frame(xs: []i32) []i32 = map(|x: i32| x + 1, xs)")
        .expect("attribute-free root entry lowers through the array planner");
    assert_naga_accepts_spirv(&lowered.spirv);
}
#[test]
fn unified_root_is_classified_before_egir() {
    let program = compile_thru_tlc("entry frame(xs: []i32) []i32 = map(|x: i32| x + 1, xs)")
        .expect("root reaches TLC");
    let entry_kind = program.defs.iter().find_map(|definition| {
        let tlc::DefMeta::EntryPoint(entry) = &definition.meta else {
            return None;
        };
        Some(entry.declaration.entry_kind)
    });
    assert_eq!(entry_kind, Some(interface::EntryKind::Compute));
}

#[test]
fn unified_root_graphics_program_reaches_tlc() {
    let program = compile_thru_tlc(
        r#"
entry triangle(target: render_target<vec4f32>) render_target<vec4f32> =
  let covered = rasterize_triangles(
    direct_draw(3u32, 1u32),
    |vertex| vertex_output(
      @[f32(vertex.vertex_index), 0.0, 0.0, 1.0],
      @[1.0, 0.0, 0.0, 1.0])) in
  shade(target, covered, |fragment| fragment.value)
"#,
    )
    .expect("unified graphics entry reaches the stage-extraction boundary");
    assert_eq!(program.defs.len(), 2);
    assert!(program.defs.iter().all(|definition| matches!(definition.meta, tlc::DefMeta::EntryPoint(_))));
    let groups = program
        .defs
        .iter()
        .filter_map(|definition| {
            let tlc::DefMeta::EntryPoint(entry) = &definition.meta else {
                return None;
            };
            entry.declaration.graphics_group.as_ref()
        })
        .collect::<Vec<_>>();
    assert_eq!(groups.len(), 2);
    assert_eq!(groups[0], groups[1]);
}

#[test]
fn unified_root_graphics_program_compiles_through_spirv() {
    let lowered = compile_thru_spirv(
        r#"
entry triangle(target: render_target<vec4f32>) render_target<vec4f32> =
  let covered = rasterize_lines(
    direct_draw_from(5u32, 2u32, 7u32, 3u32),
    |vertex| vertex_output(
      @[f32(vertex.vertex_index), 0.0, 0.0, 1.0],
      @[1.0, 0.0, 0.0, 1.0])) in
  shade(target, covered, |fragment| fragment.value)
"#,
    )
    .expect("unified graphics entry lowers through stage extraction");
    assert_naga_accepts_spirv(&lowered.spirv);
    assert_eq!(lowered.pipeline.pipelines.len(), 1);
    let pipeline_descriptor::Pipeline::Graphics(graphics) = &lowered.pipeline.pipelines[0] else {
        panic!("unified rasterization must publish a graphics pipeline");
    };
    assert_eq!(graphics.stages.len(), 2);
    assert!(matches!(
        graphics.stages[0].stage,
        pipeline_descriptor::ShaderStage::Vertex
    ));
    assert!(matches!(
        graphics.stages[1].stage,
        pipeline_descriptor::ShaderStage::Fragment
    ));
    assert_eq!(
        graphics.invocation.topology,
        pipeline_descriptor::PrimitiveTopology::LineList
    );
    assert_eq!(
        graphics.invocation.draw,
        pipeline_descriptor::DrawCall::Direct {
            vertex_count: 5,
            instance_count: 2,
            first_vertex: 7,
            first_instance: 3,
        }
    );
}
#[test]
fn unified_root_normalizes_ordinary_value_bindings_before_planning() {
    let lowered = compile_thru_spirv(
        r#"
entry triangle(target: render_target<vec4f32>) render_target<vec4f32> =
  let vertex_count = 3u32 in
  let draw = direct_draw(vertex_count, 1u32) in
  let covered = rasterize_triangles(
    draw,
    |vertex| vertex_output(
      @[f32(vertex.vertex_index), 0.0, 0.0, 1.0],
      @[1.0, 0.0, 0.0, 1.0])) in
  shade(target, covered, |fragment| fragment.value)
"#,
    )
    .expect("ordinary value bindings do not change the root operation plan");
    assert_naga_accepts_spirv(&lowered.spirv);

    let pipeline_descriptor::Pipeline::Graphics(graphics) = &lowered.pipeline.pipelines[0] else {
        panic!("graphics pipeline")
    };
    assert_eq!(
        graphics.invocation.draw,
        pipeline_descriptor::DrawCall::Direct {
            vertex_count: 3,
            instance_count: 1,
            first_vertex: 0,
            first_instance: 0,
        }
    );
}

#[test]
fn unified_fragment_helper_can_load_render_target_packed_in_record() {
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            let lowered = compile_thru_spirv(
                r#"
def read_scene(scene: render_target<f32>) f32 =
  target_load(scene, @[0i32, 0i32], 0u32)

entry render_target_record_helper(scene: render_target<f32>,
                                  screen: render_target<f32>)
    render_target<f32> =
  let raster = rasterize_triangles(
    direct_draw(3u32, 1u32),
    |_| vertex_output(@[0.0, 0.0, 0.0, 1.0], ()))
  let value = { scene = scene } in
  shade(screen, raster,
    |_| read_scene(value.scene))
"#,
            )
            .expect("a render target retains its identity through a record projection and helper");
            assert_naga_accepts_spirv(&lowered.spirv);
        })
        .expect("spawn record-packed render-target regression")
        .join()
        .expect("record-packed render-target regression panicked");
}

#[test]
fn unified_fragment_discards_unused_record_containing_render_target() {
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            let lowered = compile_thru_spirv(
                r#"
entry unused_render_target_record(
    scene: render_target<f32>, output: render_target<f32>)
    render_target<f32> =
  let raster = rasterize_triangles(
    direct_draw(3u32, 1u32),
    |_| vertex_output(@[0.0, 0.0, 0.0, 1.0], ())) in
  shade(output, raster,
    |_| let unused = { scene = scene } in 1.0)
"#,
            )
            .expect("an unused record must not retain a render-target stage capture");
            assert_naga_accepts_spirv(&lowered.spirv);
            let descriptor = serde_json::to_string(&lowered.pipeline).expect("serialize pipeline");
            assert!(
                !descriptor.contains("scene"),
                "the dead record must not synthesize a scene texture interface: {descriptor}"
            );
        })
        .expect("spawn unused render-target-record regression")
        .join()
        .expect("unused render-target-record regression panicked");
}

#[test]
fn unified_root_accepts_named_u32_constants_in_direct_draw() {
    let program = compile_thru_tlc(
        r#"
def WALL_VERTEX_COUNT: u32 = 36u32
def PROP_WALLS: i32 = 2632

entry walls(target: render_target<vec4f32>) render_target<vec4f32> =
  let covered = rasterize_triangles(
    direct_draw(WALL_VERTEX_COUNT, u32(PROP_WALLS)),
    |vertex| vertex_output(
      @[f32(vertex.vertex_index), 0.0, 0.0, 1.0],
      @[1.0, 0.0, 0.0, 1.0])) in
  shade(target, covered, |fragment| fragment.value)
"#,
    )
    .expect("named and cast integer constants are exposed before graphics stage extraction");
    let graphics = program
        .defs
        .iter()
        .find_map(|definition| {
            let tlc::DefMeta::EntryPoint(entry) = &definition.meta else {
                return None;
            };
            entry.declaration.graphics_group.as_ref()
        })
        .expect("graphics stage group");
    assert_eq!(
        graphics.invocation.draw,
        pipeline_descriptor::DrawCall::Direct {
            vertex_count: 36,
            instance_count: 2632,
            first_vertex: 0,
            first_instance: 0,
        }
    );
}

#[test]
fn unified_root_flattens_nested_record_compute_output() {
    let lowered = compile_thru_spirv(
        r#"
def fullscreen(vertex: vertex_invocation) vertex<()> =
  let x = if vertex.vertex_index == 2u32 then 3.0 else -1.0
  let y = if vertex.vertex_index == 1u32 then 3.0 else -1.0 in
  vertex_output(@[x, y, 0.0, 1.0], ())

def white(fragment: fragment_invocation<()>) vec4f32 =
  @[1.0, 1.0, 1.0, 1.0]

entry nested_record_output(values: []f32, target: render_target<vec4f32>)
  ([]f32, render_target<vec4f32>) =
  let prepared = {
    world = {
      values = map(|x| x + 1.0, values),
    },
  }
  let raster = rasterize_triangles(direct_draw(3u32, 1u32), fullscreen)
  let target2 = shade(target, raster, white) in
  (prepared.world.values, target2)
"#,
    )
    .expect("nested record compute outputs flatten to their storage leaves");
    assert_naga_accepts_spirv(&lowered.spirv);
}

#[test]
fn unified_root_array_result_can_feed_vertex_callback() {
    let lowered = compile_thru_spirv(
        r#"
entry frame(points: []vec2f32,
            target: render_target<vec4f32>)
    ([]vec2f32, render_target<vec4f32>) =
  let updated = map(|p: vec2f32| p * 0.5, points) in
  let covered = rasterize_triangles(
    direct_draw(3u32, 1u32),
    |vertex|
      let p = updated[i32(vertex.vertex_index)] in
      vertex_output(@[p.x, p.y, 0.0, 1.0], p)) in
  let target' = shade(
    target,
    covered,
    |fragment| @[fragment.value.x,
                  fragment.value.y,
                  0.0,
                  1.0]) in
  (updated, target')
"#,
    )
    .expect("an ordinary array result feeds a stage callback in one root");
    assert_naga_accepts_spirv(&lowered.spirv);
    assert_eq!(lowered.pipeline.pipelines.len(), 2);
    let pipeline_descriptor::Pipeline::Compute(compute) = &lowered.pipeline.pipelines[0] else {
        panic!("array prefix must publish a compute pipeline");
    };
    let pipeline_descriptor::Pipeline::Graphics(graphics) = &lowered.pipeline.pipelines[1] else {
        panic!("rasterization must publish a graphics pipeline");
    };
    let named_frame_output = |binding: &pipeline_descriptor::Binding| {
        matches!(
            binding,
            pipeline_descriptor::Binding::StorageBuffer { name, .. }
                if name == "frame_output"
        )
    };
    let compute_output = compute.bindings.iter().any(named_frame_output);
    let graphics_input = graphics.bindings.iter().any(named_frame_output);
    assert!(
        compute_output && graphics_input,
        "intermediate binding was not shared: {:#?}",
        lowered.pipeline
    );
    assert!(lowered.pipeline.frame_graph.passes.iter().any(|pass| !pass.depends_on.is_empty()));
}

#[test]
fn unified_invocation_fields_compile_through_spirv() {
    let lowered = compile_thru_spirv(
        r#"
entry fields(target: render_target<vec4f32>) render_target<vec4f32> =
  let covered = rasterize_triangles(
    direct_draw_from(3u32, 1u32, 0u32, 0u32),
    |vertex|
      let index = vertex.vertex_index + vertex.instance_index + vertex.draw_index in
      let x = if index == 0u32 then -1.0 else if index == 1u32 then 3.0 else -1.0 in
      let y = if index == 0u32 then -1.0 else if index == 1u32 then -1.0 else 3.0 in
      vertex_output(@[x, y, 0.0, 1.0], @[x, y])) in
  shade(target, covered,
    |fragment|
      let sample = f32(fragment.sample_index) in
      let primitive = f32(fragment.primitive_index) in
      let face = if fragment.front_facing then 1.0 else 0.0 in
      @[fragment.value.x + fragment.position.x * 0.0,
        fragment.value.y, primitive + sample, face])
"#,
    )
    .expect("all unified invocation fields lower through SPIR-V");
    assert_naga_accepts_spirv(&lowered.spirv);
}

/// Explicit derivatives require uniform fragment control flow. A source
/// branch controlled by an invocation-varying value must therefore be
/// diagnosed rather than lowered to an `OpFwidth` with undefined results.
#[test]
#[ignore = "derivative uniformity analysis is not implemented yet"]
fn nonuniform_fragment_derivative_is_rejected() {
    let error = match compile_thru_spirv(
        r#"
open f32

def divergent_color(fragment: fragment_invocation<f32>) vec4f32 =
  let x = fragment.position.x in
  let width = if x > 0.5 then fwidth(x) else 0.0 in
  @[width, 0.0, 0.0, 1.0]

entry divergent_fwidth(target: render_target<vec4f32>) render_target<vec4f32> =
  let covered = rasterize_triangles(
    direct_draw(3u32, 1u32),
    |vertex|
      let x = if vertex.vertex_index == 1u32 then 3.0 else -1.0 in
      let y = if vertex.vertex_index == 2u32 then 3.0 else -1.0 in
      vertex_output(@[x, y, 0.0, 1.0], 0.0)) in
  shade(target, covered, divergent_color)
"#,
    ) {
        Ok(_) => panic!("an invocation-varying derivative call must be rejected"),
        Err(error) => error,
    };

    let diagnostic = error.to_string();
    assert!(
        diagnostic.contains("derivative") && diagnostic.contains("uniform"),
        "unexpected diagnostic: {diagnostic}",
    );
}

#[test]
fn unified_graphics_captures_use_compiler_assigned_interfaces() {
    let lowered = compile_thru_spirv(
        r#"
entry triangle(points: []vec2f32, scale: f32,
               target: render_target<vec4f32>) render_target<vec4f32> =
  let covered = rasterize_triangles(
    direct_draw(3u32, 1u32),
    |vertex|
      let p = points[i32(vertex.vertex_index)] in
      vertex_output(
        @[p.x * scale, p.y * scale, 0.0, 1.0],
        @[1.0, 0.0, 0.0, 1.0])) in
  shade(target, covered,
    |fragment| @[fragment.value.x * scale,
                  fragment.value.y,
                  fragment.value.z,
                  fragment.value.w])
"#,
    )
    .expect("unified callback captures receive compiler-assigned interfaces");
    assert_naga_accepts_spirv(&lowered.spirv);
    let pipeline_descriptor::Pipeline::Graphics(graphics) = &lowered.pipeline.pipelines[0] else {
        panic!("unified rasterization must publish a graphics pipeline");
    };
    assert!(graphics.bindings.iter().any(|binding| matches!(
        binding,
        pipeline_descriptor::Binding::StorageBuffer {
            set: 0,
            binding: 0,
            name,
            ..
        } if name.contains("points")
    )));
    assert!(graphics.bindings.iter().any(|binding| matches!(
        binding,
        pipeline_descriptor::Binding::Uniform { name, .. } if name.contains("scale")
    )));
}

#[test]
fn unified_graphics_supports_structured_varyings_and_sampled_resources() {
    let lowered = compile_thru_spirv(
        r#"
type varying = { uv: vec2f32, tint: vec4f32 }

entry textured(tex: texture2d, sampling: sampler,
               target: render_target<vec4f32>) render_target<vec4f32> =
  let covered = rasterize_triangles(
    direct_draw(3u32, 1u32),
    |vertex|
      let index = vertex.vertex_index in
      let position =
        if index == 0u32 then @[-0.5, -0.5, 0.0, 1.0]
        else if index == 1u32 then @[0.5, -0.5, 0.0, 1.0]
        else @[0.0, 0.5, 0.0, 1.0] in
      vertex_output(position,
        { uv = position.xy + @[0.5, 0.5],
          tint = @[1.0, 0.8, 0.6, 1.0] })) in
  shade(target, covered,
    |fragment|
      texture_sample(tex, sampling, fragment.value.uv, 0.0)
        * fragment.value.tint)
"#,
    )
    .expect("structured varyings and sampled resources lower through unified callbacks");
    assert_naga_accepts_spirv(&lowered.spirv);
    let pipeline_descriptor::Pipeline::Graphics(graphics) = &lowered.pipeline.pipelines[0] else {
        panic!("unified rasterization must publish a graphics pipeline");
    };
    assert!(graphics
        .bindings
        .iter()
        .any(|binding| matches!(binding, pipeline_descriptor::Binding::Texture { binding: 0, .. })));
    assert!(graphics
        .bindings
        .iter()
        .any(|binding| matches!(binding, pipeline_descriptor::Binding::Sampler { binding: 1, .. })));
}

#[test]
fn unified_graphics_callbacks_may_call_named_helpers() {
    let lowered = compile_thru_spirv(
        r#"
def make_vertex(vertex: vertex_invocation) vertex<vec4f32> =
  vertex_output(
    if vertex.vertex_index == 0u32 then @[-0.5, -0.5, 0.0, 1.0]
    else if vertex.vertex_index == 1u32 then @[0.5, -0.5, 0.0, 1.0]
    else @[0.0, 0.5, 0.0, 1.0],
    @[0.2, 0.6, 1.0, 1.0])

def make_fragment(fragment: fragment_invocation<vec4f32>) vec4f32 =
  fragment.value

entry triangle(target: render_target<vec4f32>) render_target<vec4f32> =
  let covered = rasterize_triangles(direct_draw(3u32, 1u32), make_vertex) in
  shade(target, covered, make_fragment)
"#,
    )
    .expect("named helpers inherit their callback stage context");
    assert_naga_accepts_spirv(&lowered.spirv);
}

#[test]
fn unified_root_supports_successive_draws_into_one_target() {
    let lowered = compile_thru_spirv(
        r#"
def triangle_vertex(offset: f32, vertex: vertex_invocation) vertex<vec4f32> =
  let x = if vertex.vertex_index == 0u32 then -0.5
          else if vertex.vertex_index == 1u32 then 0.5
          else 0.0 in
  let y = if vertex.vertex_index == 2u32 then 0.5 else -0.5 in
  vertex_output(@[x + offset, y, 0.0, 1.0], @[1.0, 1.0, 1.0, 1.0])

entry layered(target: render_target<vec4f32>) render_target<vec4f32> =
  let background = rasterize_triangles(
    direct_draw(3u32, 1u32), triangle_vertex(-0.25, _)) in
  let target1 = shade(
    target, background,
    |fragment| fragment.value * @[0.2, 0.4, 0.8, 1.0]) in
  let foreground = rasterize_triangles(
    direct_draw(3u32, 1u32), triangle_vertex(0.25, _)) in
  shade(
    target1, foreground,
    |fragment| fragment.value * @[0.9, 0.3, 0.1, 1.0])
"#,
    )
    .expect("successive unified draws lower through stage extraction");
    assert_naga_accepts_spirv(&lowered.spirv);

    let graphics = lowered
        .pipeline
        .pipelines
        .iter()
        .filter_map(|pipeline| match pipeline {
            pipeline_descriptor::Pipeline::Graphics(graphics) => Some(graphics),
            pipeline_descriptor::Pipeline::Compute(_) => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(graphics.len(), 2, "one graphics pipeline per draw");
    assert!(graphics
        .iter()
        .all(|pipeline| pipeline.fragment_outputs.iter().any(|output| output.name == "target")));

    let fragment_passes = lowered
        .pipeline
        .frame_graph
        .passes
        .iter()
        .filter(|pass| pass.kind == pipeline_descriptor::FramePassKind::Fragment)
        .collect::<Vec<_>>();
    assert_eq!(fragment_passes.len(), 2);
    assert!(
        !fragment_passes[1].depends_on.is_empty(),
        "the second target update must follow the first"
    );
}

#[test]
fn unified_root_can_transfer_raster_through_an_ordinary_helper() {
    let source = r#"
def pass_raster<V>(covered: raster<V>) raster<V> = covered

entry frame(target: render_target<vec4f32>) render_target<vec4f32> =
  let covered = rasterize_triangles(
    direct_draw(3u32, 1u32),
    |vertex| vertex_output(@[0.0, 0.0, 0.0, 1.0], @[1.0, 1.0, 1.0, 1.0])) in
  shade(target, pass_raster(covered), |fragment| fragment.value)
"#;
    compile_to_spirv(source).expect("an ordinary helper may transfer a raster value");
}

#[test]
fn unified_compute_stage_inlines_nested_render_target_helpers() {
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            let source = r#"
def read_depth(target: render_target<f32>) f32 =
  target_load(target, @[0i32, 0i32], 0u32)

def keep(target: render_target<f32>) bool =
  read_depth(target) > 0.0

entry render_target_filter_helper(source: render_target<f32>) render_target<f32> =
  let visible =
    let kept = filter(|i| keep(source), iota(1)) in
    { instances = kept }
  let raster = rasterize_triangles(
    direct_draw(3u32, 1u32),
    |_| vertex_output(
      @[f32(length(visible.instances)) * 0.0, 0.0, 0.0, 1.0], ())) in
  shade(source, raster, |_| 1.0)
"#;
            compile_thru_spirv(source)
                .expect("nested helpers must expose render-target reads before compute-stage extraction");
        })
        .expect("spawn nested render-target helper regression")
        .join()
        .expect("nested render-target helper regression panicked");
}

#[test]
fn unified_root_rejects_reusing_consumed_raster() {
    let source = r#"
entry frame(target: render_target<vec4f32>) render_target<vec4f32> =
  let covered = rasterize_triangles(
    direct_draw(3u32, 1u32),
    |vertex| vertex_output(@[0.0, 0.0, 0.0, 1.0], @[1.0, 1.0, 1.0, 1.0])) in
  let first = shade(target, covered, |fragment| fragment.value) in
  shade(first, covered, |fragment| fragment.value)
"#;
    let error = compile_thru_tlc(source).expect_err("a raster value is consumed by shade");
    assert!(
        error.to_string().contains("use of moved value `covered`"),
        "unexpected diagnostic: {error}"
    );
}

#[test]
fn unified_root_rejects_reusing_consumed_render_target() {
    let source = r#"
entry frame(target: render_target<vec4f32>) render_target<vec4f32> =
  let first_raster = rasterize_triangles(
    direct_draw(3u32, 1u32),
    |vertex| vertex_output(@[0.0, 0.0, 0.0, 1.0], @[1.0, 1.0, 1.0, 1.0])) in
  let first = shade(target, first_raster, |fragment| fragment.value) in
  let second_raster = rasterize_triangles(
    direct_draw(3u32, 1u32),
    |vertex| vertex_output(@[0.0, 0.0, 0.0, 1.0], @[1.0, 1.0, 1.0, 1.0])) in
  let second = shade(target, second_raster, |fragment| fragment.value) in
  second
"#;
    let error = compile_thru_tlc(source).expect_err("shade consumes its render target");
    assert!(
        error.to_string().contains("use of moved value `target`"),
        "unexpected diagnostic: {error}"
    );
}

#[test]
fn unified_root_rejects_reading_target_from_its_own_fragment_callback() {
    let source = r#"
entry frame(target: render_target<vec4f32>) render_target<vec4f32> =
  let covered = rasterize_triangles(
    direct_draw(3u32, 1u32),
    |vertex| vertex_output(@[0.0, 0.0, 0.0, 1.0], @[1.0, 1.0, 1.0, 1.0])) in
  shade(target, covered,
    |fragment| target_load(target, @[0, 0], fragment.sample_index))
"#;
    let error =
        compile_thru_tlc(source).expect_err("a fragment callback must not read the target being consumed");
    assert!(
        error
            .to_string()
            .contains("fragment callback reads render target `target` while `shade` consumes it"),
        "unexpected diagnostic: {error}"
    );
}

#[test]
fn unified_root_rejects_discarded_raster() {
    let source = r#"
entry frame(target: render_target<vec4f32>) render_target<vec4f32> =
  let covered = rasterize_triangles(
    direct_draw(3u32, 1u32),
    |vertex| vertex_output(@[0.0, 0.0, 0.0, 1.0], @[1.0, 1.0, 1.0, 1.0])) in
  target
"#;
    let error = compile_thru_tlc(source).expect_err("a raster value must not be discarded");
    assert!(
        error.to_string().contains("raster value `covered` must be consumed exactly once"),
        "unexpected diagnostic: {error}"
    );
}

#[test]
fn unified_root_orders_graphics_compute_graphics_operations() {
    let lowered = compile_thru_spirv(
        r#"
entry postprocess(values: []vec2i32,
                  scene: render_target<vec4f32>,
                  surface: render_target<vec4f32>)
    ([]f32, render_target<vec4f32>, render_target<vec4f32>) =
  let scene_raster = rasterize_triangles(
    direct_draw(3u32, 1u32),
    |vertex| vertex_output(
      if vertex.vertex_index == 0u32 then @[-1.0, -1.0, 0.0, 1.0]
      else if vertex.vertex_index == 1u32 then @[3.0, -1.0, 0.0, 1.0]
      else @[-1.0, 3.0, 0.0, 1.0],
      1.0)) in
  let scene1 = shade(
    scene, scene_raster,
    |fragment| @[fragment.value, 0.0, 0.0, 1.0]) in
  let adjusted = map(
    |coord: vec2i32| target_load(scene1, coord, 0u32).x * 0.5,
    values) in
  let resolve_raster = rasterize_triangles(
    direct_draw(3u32, 1u32),
    |vertex| vertex_output(
      if vertex.vertex_index == 0u32 then @[-1.0, -1.0, 0.0, 1.0]
      else if vertex.vertex_index == 1u32 then @[3.0, -1.0, 0.0, 1.0]
      else @[-1.0, 3.0, 0.0, 1.0],
      ())) in
  let surface1 = shade(
    surface, resolve_raster,
    |fragment|
      let gain = adjusted[i32(fragment.position.x) % length(adjusted)] in
      @[gain, gain, gain, 1.0]) in
  (adjusted, scene1, surface1)
"#,
    )
    .expect("a unified root may alternate graphics and ordinary computation");
    assert_naga_accepts_spirv(&lowered.spirv);

    assert_eq!(lowered.pipeline.pipelines.len(), 3);
    assert!(matches!(
        lowered.pipeline.pipelines[0],
        pipeline_descriptor::Pipeline::Graphics(_)
    ));
    assert!(matches!(
        lowered.pipeline.pipelines[1],
        pipeline_descriptor::Pipeline::Compute(_)
    ));
    let pipeline_descriptor::Pipeline::Graphics(resolve) = &lowered.pipeline.pipelines[2] else {
        panic!("the final operation must be the resolve graphics pipeline")
    };
    let compute_names = match &lowered.pipeline.pipelines[1] {
        pipeline_descriptor::Pipeline::Compute(compute) => compute
            .bindings
            .iter()
            .filter_map(|binding| match binding {
                pipeline_descriptor::Binding::StorageBuffer { name, .. } => Some(name),
                _ => None,
            })
            .collect::<Vec<_>>(),
        _ => unreachable!(),
    };
    assert!(resolve.bindings.iter().any(|binding| {
        matches!(
            binding,
            pipeline_descriptor::Binding::StorageBuffer { name, .. }
                if compute_names.contains(&name)
        )
    }));

    let passes = &lowered.pipeline.frame_graph.passes;
    let scene_fragment = passes
        .iter()
        .position(|pass| {
            pass.pipeline_index == 0 && pass.kind == pipeline_descriptor::FramePassKind::Fragment
        })
        .expect("scene fragment pass");
    let compute =
        passes.iter().position(|pass| pass.pipeline_index == 1).expect("intermediate compute pass");
    let resolve_fragment = passes
        .iter()
        .position(|pass| {
            pass.pipeline_index == 2 && pass.kind == pipeline_descriptor::FramePassKind::Fragment
        })
        .expect("resolve fragment pass");
    assert!(passes[compute].depends_on.contains(&scene_fragment));
    assert!(passes[resolve_fragment].depends_on.contains(&compute));
}

#[test]
fn unified_root_samples_a_prior_render_target_in_a_later_pass() {
    let lowered = compile_thru_spirv(
        r#"
entry resolve(sampling: sampler,
              scene: render_target<vec4f32>,
              surface: render_target<vec4f32>)
    (render_target<vec4f32>, render_target<vec4f32>) =
  let geometry = rasterize_triangles(
    direct_draw(3u32, 1u32),
    |vertex| vertex_output(
      if vertex.vertex_index == 0u32 then @[-1.0, -1.0, 0.0, 1.0]
      else if vertex.vertex_index == 1u32 then @[3.0, -1.0, 0.0, 1.0]
      else @[-1.0, 3.0, 0.0, 1.0],
      @[0.25, 0.5, 0.75, 1.0])) in
  let scene1 = shade(scene, geometry, |fragment| fragment.value) in
  let fullscreen = rasterize_triangles(
    direct_draw(3u32, 1u32),
    |vertex| vertex_output(
      if vertex.vertex_index == 0u32 then @[-1.0, -1.0, 0.0, 1.0]
      else if vertex.vertex_index == 1u32 then @[3.0, -1.0, 0.0, 1.0]
      else @[-1.0, 3.0, 0.0, 1.0],
      ())) in
  let surface1 = shade(
    surface,
    fullscreen,
    |fragment| target_sample(
      scene1,
      sampling,
      fragment.position.xy / @[640.0, 480.0])) in
  (scene1, surface1)
"#,
    )
    .expect("a later graphics pass may filter a prior render target");
    assert_naga_accepts_spirv(&lowered.spirv);

    assert_eq!(lowered.pipeline.pipelines.len(), 2);
    let pipeline_descriptor::Pipeline::Graphics(resolve) = &lowered.pipeline.pipelines[1] else {
        panic!("resolve graphics pipeline")
    };
    assert!(resolve.bindings.iter().any(|binding| matches!(
        binding,
        pipeline_descriptor::Binding::Texture {
            resource: Some(resource),
            ..
        } if resource == "scene"
    )));
    assert!(resolve.bindings.iter().any(|binding| matches!(
        binding,
        pipeline_descriptor::Binding::Sampler { name, .. } if name == "sampling"
    )));

    let passes = &lowered.pipeline.frame_graph.passes;
    let scene_fragment = passes
        .iter()
        .position(|pass| {
            pass.pipeline_index == 0 && pass.kind == pipeline_descriptor::FramePassKind::Fragment
        })
        .expect("scene fragment pass");
    let resolve_fragment = passes
        .iter()
        .position(|pass| {
            pass.pipeline_index == 1 && pass.kind == pipeline_descriptor::FramePassKind::Fragment
        })
        .expect("resolve fragment pass");
    assert!(passes[resolve_fragment].depends_on.contains(&scene_fragment));
}

#[test]
fn unified_root_supports_structured_render_targets() {
    let lowered = compile_thru_spirv(
        r#"
type gbuffer = {
  albedo: vec4f32,
  normal: vec4f32,
  depth: f32
}

entry deferred(coords: []vec2i32,
               scene: render_target<gbuffer>,
               surface: render_target<vec4f32>)
    ([]f32, render_target<gbuffer>, render_target<vec4f32>) =
  let geometry = rasterize_triangles(
    direct_draw(3u32, 1u32),
    |vertex| vertex_output(
      if vertex.vertex_index == 0u32 then @[-1.0, -1.0, 0.0, 1.0]
      else if vertex.vertex_index == 1u32 then @[3.0, -1.0, 0.0, 1.0]
      else @[-1.0, 3.0, 0.0, 1.0],
      0.5)) in
  let scene1 = shade(
    scene, geometry,
    |fragment| {
      albedo = @[fragment.value, 0.2, 0.1, 1.0],
      normal = @[0.0, 0.0, 1.0, 0.0],
      depth = fragment.position.z
    }) in
  let depths = map(
    |coord: vec2i32| target_load(scene1, coord, 0u32).depth,
    coords) in
  let fullscreen = rasterize_triangles(
    direct_draw(3u32, 1u32),
    |vertex| vertex_output(
      if vertex.vertex_index == 0u32 then @[-1.0, -1.0, 0.0, 1.0]
      else if vertex.vertex_index == 1u32 then @[3.0, -1.0, 0.0, 1.0]
      else @[-1.0, 3.0, 0.0, 1.0],
      ())) in
  let surface1 = shade(
    surface, fullscreen,
    |fragment|
      let depth = depths[i32(fragment.position.x) % length(depths)] in
      @[depth, depth, depth, 1.0]) in
  (depths, scene1, surface1)
"#,
    )
    .expect("structured render targets flow between graphics and compute");
    assert_naga_accepts_spirv(&lowered.spirv);

    let pipeline_descriptor::Pipeline::Graphics(geometry) = &lowered.pipeline.pipelines[0] else {
        panic!("geometry pipeline")
    };
    assert_eq!(
        geometry.fragment_outputs.iter().map(|output| output.name.as_str()).collect::<Vec<_>>(),
        vec!["scene_albedo", "scene_normal", "scene_depth"]
    );

    let pipeline_descriptor::Pipeline::Compute(depths) = &lowered.pipeline.pipelines[1] else {
        panic!("depth processing pipeline")
    };
    let sampled_resources = depths
        .bindings
        .iter()
        .filter_map(|binding| match binding {
            pipeline_descriptor::Binding::Texture {
                resource: Some(resource),
                ..
            } => Some(resource.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        sampled_resources,
        vec!["scene_albedo", "scene_normal", "scene_depth"]
    );
}

#[test]
fn fragment_output_helpers_can_destructure_the_predeclared_sum() {
    let lowered = compile_thru_spirv(
        r#"
def normalize_output(output: fragment_output<vec4f32>) fragment_output<vec4f32> =
  match output
  case #color(value) -> #color(value)
  case #depth(value, depth) -> #depth(value, depth)
  case #discard -> #discard

def fragment_stage(fragment: fragment_invocation<vec4f32>) fragment_output<vec4f32> =
  normalize_output(
    if fragment.front_facing
    then
      let depth = 0.25 in
      #depth(fragment.value, depth)
    else #discard)

entry helper(target: render_target<vec4f32>) render_target<vec4f32> =
  let covered = rasterize_triangles(
    direct_draw(3u32, 1u32),
    |vertex| vertex_output(
      if vertex.vertex_index == 0u32 then @[-0.5, -0.5, 0.0, 1.0]
      else if vertex.vertex_index == 1u32 then @[0.5, -0.5, 0.0, 1.0]
      else @[0.0, 0.5, 0.0, 1.0],
      @[1.0, 0.5, 0.25, 1.0])) in
  shade(target, covered, fragment_stage)
"#,
    )
    .expect("fragment-output helpers can construct and match their predeclared sum");
    assert_naga_accepts_spirv(&lowered.spirv);
}

#[test]
fn direct_tuple_color_is_not_mistaken_for_fragment_output() {
    let lowered = compile_thru_spirv(
        r#"
entry tuple_color(target: render_target<(u32, vec4f32, vec4f32, f32)>)
    render_target<(u32, vec4f32, vec4f32, f32)> =
  let covered = rasterize_triangles(
    direct_draw(3u32, 1u32),
    |vertex| vertex_output(
      if vertex.vertex_index == 0u32 then @[-0.5, -0.5, 0.0, 1.0]
      else if vertex.vertex_index == 1u32 then @[0.5, -0.5, 0.0, 1.0]
      else @[0.0, 0.5, 0.0, 1.0],
      (7u32, @[1.0, 0.0, 0.0, 1.0], @[0.0, 1.0, 0.0, 1.0], 0.5))) in
  shade(target, covered, |fragment| fragment.value)
"#,
    )
    .expect("a direct tuple color remains an ordinary color result");
    assert_naga_accepts_spirv(&lowered.spirv);

    let pipeline_descriptor::Pipeline::Graphics(graphics) = &lowered.pipeline.pipelines[0] else {
        panic!("graphics pipeline")
    };
    assert_eq!(graphics.fragment_outputs.len(), 4);
    assert!(!spirv_has_builtin(&lowered.spirv, spirv::BuiltIn::FragDepth));
    assert!(!spirv_has_builtin(&lowered.spirv, spirv::BuiltIn::SampleMask));
}

#[test]
fn unified_root_supports_explicit_depth_and_conditional_discard() {
    let source = r#"
entry cutout(target: render_target<vec4f32>) render_target<vec4f32> =
  let covered = rasterize_triangles(
    direct_draw(3u32, 1u32),
    |vertex| vertex_output(
      if vertex.vertex_index == 0u32 then @[-0.5, -0.5, 0.0, 1.0]
      else if vertex.vertex_index == 1u32 then @[0.5, -0.5, 0.0, 1.0]
      else @[0.0, 0.5, 0.0, 1.0],
      @[1.0, 0.5, 0.25, 1.0])) in
  shade_with(
    { depth_test = #less,
      depth_write = true,
      blend = #replace,
      color_write = true },
    target,
    covered,
    |fragment|
      if fragment.front_facing
      then #depth(fragment.value, 0.25)
      else #discard)
"#;
    let lowered = compile_thru_spirv(source)
        .expect("fragment_output supports explicit depth and conditional discard");
    assert_naga_accepts_spirv(&lowered.spirv);

    let pipeline_descriptor::Pipeline::Graphics(graphics) = &lowered.pipeline.pipelines[0] else {
        panic!("graphics pipeline")
    };
    assert_eq!(
        graphics.fragment_outputs.iter().map(|output| output.name.as_str()).collect::<Vec<_>>(),
        vec!["target"]
    );

    assert!(spirv_has_builtin(&lowered.spirv, spirv::BuiltIn::FragDepth));
    assert!(spirv_has_builtin(&lowered.spirv, spirv::BuiltIn::SampleMask));

    let wgsl = lower_ssa_to_wgsl(compile_thru_ssa(source).expect("fragment_output lowers to portable SSA"))
        .expect("fragment_output lowers to WGSL");
    assert!(wgsl.contains("@builtin(frag_depth)"), "{wgsl}");
    assert!(wgsl.contains("@builtin(sample_mask)"), "{wgsl}");
    let module = naga::front::wgsl::parse_str(&wgsl)
        .unwrap_or_else(|error| panic!("Naga rejected fragment-output WGSL: {error:?}\n{wgsl}"));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|error| panic!("Naga validation rejected fragment-output WGSL: {error:?}\n{wgsl}"));
}
#[test]
fn unified_root_accepts_explicit_fragment_state() {
    let lowered = compile_thru_spirv(
        r#"
entry depth_tested(target: render_target<vec4f32>) render_target<vec4f32> =
  let covered = rasterize_triangles(
    direct_draw(3u32, 1u32),
    |vertex| vertex_output(
      if vertex.vertex_index == 0u32 then @[-0.5, -0.5, 0.0, 1.0]
      else if vertex.vertex_index == 1u32 then @[0.5, -0.5, 0.0, 1.0]
      else @[0.0, 0.5, 0.0, 1.0],
      @[1.0, 1.0, 1.0, 1.0])) in
  shade_with(
    { depth_test = #less_equal,
      depth_write = true,
      blend = #replace,
      color_write = true },
    target, covered, |fragment| fragment.value)
"#,
    )
    .expect("shade_with accepts the specified structural fragment state");
    assert_naga_accepts_spirv(&lowered.spirv);

    let pipeline_descriptor::Pipeline::Graphics(graphics) = &lowered.pipeline.pipelines[0] else {
        panic!("graphics pipeline")
    };
    assert_eq!(
        graphics.invocation.fragment_state.depth_test,
        pipeline_descriptor::DepthTest::LessEqual
    );
    assert!(graphics.invocation.fragment_state.depth_write);
    assert_eq!(
        graphics.invocation.fragment_state.blend,
        pipeline_descriptor::BlendMode::Replace
    );
    assert!(graphics.invocation.fragment_state.color_write);
}

#[test]
fn unified_root_accepts_named_explicit_fragment_state() {
    let lowered = compile_thru_spirv(
        r#"
def opaque_depth: fragment_state = {
  depth_test = #less_equal,
  depth_write = true,
  blend = #replace,
  color_write = true,
}

entry reproduce(target: render_target<vec4f32>) render_target<vec4f32> =
  let covered = rasterize_triangles(
    direct_draw(3u32, 1u32),
    |vertex| vertex_output(
      if vertex.vertex_index == 0u32 then @[-0.5, -0.5, 0.0, 1.0]
      else if vertex.vertex_index == 1u32 then @[0.5, -0.5, 0.0, 1.0]
      else @[0.0, 0.5, 0.0, 1.0],
      @[1.0, 0.5, 0.25, 1.0])) in
  shade_with(opaque_depth, target, covered, |fragment| fragment.value)
"#,
    )
    .expect("a named fragment_state supplies context to its sum fields");
    assert_naga_accepts_spirv(&lowered.spirv);

    let pipeline_descriptor::Pipeline::Graphics(graphics) = &lowered.pipeline.pipelines[0] else {
        panic!("graphics pipeline")
    };
    assert_eq!(
        graphics.invocation.fragment_state.depth_test,
        pipeline_descriptor::DepthTest::LessEqual
    );
    assert!(graphics.invocation.fragment_state.depth_write);
    assert_eq!(
        graphics.invocation.fragment_state.blend,
        pipeline_descriptor::BlendMode::Replace
    );
    assert!(graphics.invocation.fragment_state.color_write);
}

#[test]
fn unified_root_flattens_structured_compute_results() {
    let lowered = compile_thru_spirv(
        r#"
entry prepare_and_draw<[n]>(values: [n]vec4f32,
                            target: render_target<vec4f32>)
    ({ positions: [n]vec4f32, colors: [n]vec4f32 },
     render_target<vec4f32>) =
  let prepared = {
    positions = map(|value: vec4f32| value, values),
    colors = map(|value: vec4f32| value * 0.5, values)
  } in
  let covered = rasterize_triangles(
    direct_draw(3u32, 1u32),
    |vertex|
      let i = i32(vertex.vertex_index) in
      vertex_output(prepared.positions[i], prepared.colors[i])) in
  let target1 = shade(target, covered, |fragment| fragment.value) in
  (prepared, target1)
"#,
    )
    .expect("records of compute arrays are flattened into stage resources");
    assert_naga_accepts_spirv(&lowered.spirv);

    assert_eq!(lowered.pipeline.pipelines.len(), 2);
    let pipeline_descriptor::Pipeline::Compute(compute) = &lowered.pipeline.pipelines[0] else {
        panic!("preparation compute pipeline")
    };
    let outputs = compute
        .bindings
        .iter()
        .filter_map(|binding| match binding {
            pipeline_descriptor::Binding::StorageBuffer {
                usage: pipeline_descriptor::BufferUsage::Output,
                name,
                ..
            } => Some(name.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(outputs.len(), 2);

    let pipeline_descriptor::Pipeline::Graphics(graphics) = &lowered.pipeline.pipelines[1] else {
        panic!("draw graphics pipeline")
    };
    for output in outputs {
        assert!(graphics.bindings.iter().any(|binding| {
            matches!(binding, pipeline_descriptor::Binding::StorageBuffer { name, .. } if name == output)
        }));
    }
}

#[test]
fn map_can_return_a_dynamic_array_of_records() {
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            let lowered = compile_thru_spirv(
                r#"
type pair = { x: i32, y: i32 }

entry structured_map_result() []pair =
  map(|i| { x = i, y = i }, iota(1))
"#,
            )
            .expect("a mapped record result must match the lambda's physical result boundary");
            assert_naga_accepts_spirv(&lowered.spirv);
        })
        .expect("spawn structured map regression")
        .join()
        .expect("structured map regression panicked");
}

#[test]
fn unified_root_uses_computed_indirect_draw_command() {
    let lowered = compile_thru_spirv(
        r#"
type draw_command = {
  vertex_count: u32,
  instance_count: u32,
  first_vertex: u32,
  first_instance: u32
}

type prepared = { instances: []vec4f32, commands: [1]draw_command }

entry compact_and_draw(values: []vec4f32,
                       target: render_target<vec4f32>)
    render_target<vec4f32> =
  let prepared =
    let live = filter(|value: vec4f32| value.w > 0.0, values) in
    {
      instances = live,
      commands = [{
        vertex_count = 3u32,
        instance_count = u32(length(live)),
        first_vertex = 0u32,
        first_instance = 0u32
      }]
    } in
  let covered = rasterize_triangles(
    indirect_draw(prepared.commands[0]),
    |vertex|
      let p = prepared.instances[i32(vertex.instance_index)] in
      let x = if vertex.vertex_index == 0u32 then -0.5
              else if vertex.vertex_index == 1u32 then 0.5
              else 0.0 in
      let y = if vertex.vertex_index == 2u32 then 0.5 else -0.5 in
      vertex_output(@[p.x + x, p.y + y, 0.0, 1.0], p)) in
  shade(target, covered, |fragment| fragment.value)
"#,
    )
    .expect("a computed command buffer can drive an indirect draw");
    assert_naga_accepts_spirv(&lowered.spirv);

    assert_eq!(lowered.pipeline.pipelines.len(), 2);
    let pipeline_descriptor::Pipeline::Graphics(graphics) = &lowered.pipeline.pipelines[1] else {
        panic!("indirect graphics pipeline")
    };
    let pipeline_descriptor::DrawCall::Indirect { commands, offset, .. } = &graphics.invocation.draw else {
        panic!("draw must be indirect")
    };
    assert_eq!(*offset, 0);
    let resource = commands.resource.as_ref().expect("computed draw resource");
    assert!(
        lowered.pipeline.frame_graph.indirect_draws.iter().any(|dependency| {
            lowered.pipeline.frame_graph.resources[dependency.buffer_resource].name == *resource
        })
    );
}

#[test]
fn unified_root_publishes_indexed_draws() {
    let lowered = compile_thru_spirv(
        r#"
entry indexed(indices: [3]u32,
              target: render_target<vec4f32>) render_target<vec4f32> =
  let covered = rasterize_triangles(
    indexed_draw(indices, 2u32),
    |vertex| vertex_output(@[0.0, 0.0, 0.0, 1.0], @[1.0, 1.0, 1.0, 1.0])) in
  shade(target, covered, |fragment| fragment.value)
"#,
    )
    .expect("indexed_draw is accepted by unified roots");
    assert_naga_accepts_spirv(&lowered.spirv);

    let pipeline_descriptor::Pipeline::Graphics(graphics) = &lowered.pipeline.pipelines[0] else {
        panic!("graphics pipeline")
    };
    let pipeline_descriptor::DrawCall::Indexed {
        indices,
        index_count,
        instance_count,
        ..
    } = &graphics.invocation.draw
    else {
        panic!("draw must be indexed")
    };
    assert_eq!(*index_count, pipeline_descriptor::DrawCount::Fixed(3));
    assert_eq!(*instance_count, 2);
    let index_resource = indices.resource.as_ref().unwrap_or(&indices.name);
    let vertex_pass = lowered
        .pipeline
        .frame_graph
        .passes
        .iter()
        .find(|pass| pass.kind == pipeline_descriptor::FramePassKind::Vertex)
        .expect("vertex pass");
    assert!(vertex_pass
        .reads
        .iter()
        .any(|read| { lowered.pipeline.frame_graph.resources[read.resource].name == *index_resource }));
}

#[test]
fn unified_root_publishes_indexed_and_plural_indirect_draws() {
    let lowered = compile_thru_spirv(
        r#"
type draw_command = {
  vertex_count: u32,
  instance_count: u32,
  first_vertex: u32,
  first_instance: u32
}

type indexed_draw_command = {
  index_count: u32,
  instance_count: u32,
  first_index: u32,
  vertex_offset: i32,
  first_instance: u32
}

entry draw_many(indices: [3]u32,
                target: render_target<vec4f32>) render_target<vec4f32> =
  let commands = [
    { vertex_count = 3u32, instance_count = 1u32,
      first_vertex = 0u32, first_instance = 0u32 },
    { vertex_count = 3u32, instance_count = 2u32,
      first_vertex = 0u32, first_instance = 0u32 }
  ] in
  let indexed_commands = [{
    index_count = 3u32, instance_count = 1u32,
    first_index = 0u32, vertex_offset = 0i32, first_instance = 0u32
  }] in
  let many = rasterize_triangles(
    indirect_draws(commands),
    |vertex| vertex_output(@[0.0, 0.0, 0.0, 1.0], @[1.0, 0.0, 0.0, 1.0])) in
  let target1 = shade(target, many, |fragment| fragment.value) in
  let indexed = rasterize_triangles(
    indexed_indirect_draws(indices, indexed_commands),
    |vertex| vertex_output(@[0.0, 0.0, 0.0, 1.0], @[0.0, 1.0, 0.0, 1.0])) in
  shade(target1, indexed, |fragment| fragment.value)
"#,
    )
    .expect("plural indirect draw forms are accepted");
    assert_naga_accepts_spirv(&lowered.spirv);

    let graphics = lowered
        .pipeline
        .pipelines
        .iter()
        .filter_map(|pipeline| match pipeline {
            pipeline_descriptor::Pipeline::Graphics(graphics) => Some(graphics),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(graphics.len(), 2);
    assert!(matches!(
        graphics[0].invocation.draw,
        pipeline_descriptor::DrawCall::Indirect {
            draw_count: pipeline_descriptor::DrawCount::Fixed(2),
            ..
        }
    ));
    assert!(matches!(
        graphics[1].invocation.draw,
        pipeline_descriptor::DrawCall::IndexedIndirect {
            draw_count: pipeline_descriptor::DrawCount::Fixed(1),
            ..
        }
    ));
}
#[test]
fn unified_root_publishes_offset_and_singular_indexed_draws() {
    let lowered = compile_thru_spirv(
        r#"
type indexed_draw_command = {
  index_count: u32,
  instance_count: u32,
  first_index: u32,
  vertex_offset: i32,
  first_instance: u32
}

entry indexed_forms(indices: [4]u16,
                    target: render_target<vec4f32>) render_target<vec4f32> =
  let commands = [{
    index_count = 3u32, instance_count = 2u32,
    first_index = 1u32, vertex_offset = -1i32, first_instance = 4u32
  }] in
  let direct = rasterize_triangles(
    indexed_draw_from(indices, 3u32, 2u32, 1u32, -1i32, 4u32),
    |vertex| vertex_output(@[0.0, 0.0, 0.0, 1.0], @[1.0, 0.0, 0.0, 1.0])) in
  let target1 = shade(target, direct, |fragment| fragment.value) in
  let indirect = rasterize_triangles(
    indexed_indirect_draw(indices, commands[0]),
    |vertex| vertex_output(@[0.0, 0.0, 0.0, 1.0], @[0.0, 1.0, 0.0, 1.0])) in
  shade(target1, indirect, |fragment| fragment.value)
"#,
    )
    .expect("offset and singular indexed draw forms are accepted");
    assert_naga_accepts_spirv(&lowered.spirv);

    let graphics = lowered
        .pipeline
        .pipelines
        .iter()
        .filter_map(|pipeline| match pipeline {
            pipeline_descriptor::Pipeline::Graphics(graphics) => Some(graphics),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(graphics.len(), 2);
    assert!(matches!(
        graphics[0].invocation.draw,
        pipeline_descriptor::DrawCall::Indexed {
            index_format: pipeline_descriptor::IndexFormat::Uint16,
            index_count: pipeline_descriptor::DrawCount::Fixed(3),
            vertex_offset: -1,
            ..
        }
    ));
    assert!(matches!(
        graphics[1].invocation.draw,
        pipeline_descriptor::DrawCall::IndexedIndirect {
            index_format: pipeline_descriptor::IndexFormat::Uint16,
            draw_count: pipeline_descriptor::DrawCount::Fixed(1),
            ..
        }
    ));
}

#[test]
fn unified_root_preserves_dynamic_indirect_draw_count() {
    let lowered = compile_thru_spirv(
        r#"
type draw_command = {
  vertex_count: u32,
  instance_count: u32,
  first_vertex: u32,
  first_instance: u32
}

entry draw_dynamic(commands: []draw_command,
                   target: render_target<vec4f32>) render_target<vec4f32> =
  let covered = rasterize_triangles(
    indirect_draws(commands),
    |vertex| vertex_output(@[0.0, 0.0, 0.0, 1.0], @[1.0, 1.0, 1.0, 1.0])) in
  shade(target, covered, |fragment| fragment.value)
"#,
    )
    .expect("a runtime-sized command array remains runtime-sized in the descriptor");
    assert_naga_accepts_spirv(&lowered.spirv);

    let pipeline_descriptor::Pipeline::Graphics(graphics) = &lowered.pipeline.pipelines[0] else {
        panic!("graphics pipeline")
    };
    assert!(matches!(
        graphics.invocation.draw,
        pipeline_descriptor::DrawCall::Indirect {
            draw_count: pipeline_descriptor::DrawCount::BufferLength,
            ..
        }
    ));
}
#[test]
fn unified_root_publishes_explicit_raster_state() {
    let lowered = compile_thru_spirv(
        r#"
entry clipped(target: render_target<vec4f32>) render_target<vec4f32> =
  let covered = rasterize_triangles_with(
    {
      viewport = #custom({
        origin = @[10.0, 20.0],
        extent = @[640.0, 480.0],
        depth = @[0.25, 0.75]
      }),
      scissor = #custom({ origin = @[4, 8], extent = @[320u32, 240u32] }),
      front_face = #clockwise,
      cull = #back,
      fill = #line
    },
    direct_draw(3u32, 1u32),
    |vertex| vertex_output(@[0.0, 0.0, 0.0, 1.0], @[1.0, 1.0, 1.0, 1.0])) in
  shade(target, covered, |fragment| fragment.value)
"#,
    )
    .expect("rasterize_*_with accepts the specified raster state");
    assert_naga_accepts_spirv(&lowered.spirv);

    let pipeline_descriptor::Pipeline::Graphics(graphics) = &lowered.pipeline.pipelines[0] else {
        panic!("graphics pipeline")
    };
    assert_eq!(
        graphics.invocation.raster_state,
        pipeline_descriptor::RasterState {
            viewport: pipeline_descriptor::Viewport::Custom {
                origin: [10.0, 20.0],
                extent: [640.0, 480.0],
                depth: [0.25, 0.75],
            },
            scissor: pipeline_descriptor::Scissor::Custom {
                origin: [4, 8],
                extent: [320, 240],
            },
            front_face: pipeline_descriptor::FrontFace::Clockwise,
            cull: pipeline_descriptor::CullMode::Back,
            fill: pipeline_descriptor::FillMode::Line,
        }
    );
}
#[test]
fn target_profiles_are_selected_before_ssa_lowering() {
    let portable =
        compile_thru_ssa(" entry e(xs: []i32) []i32 = map(|x: i32| x + 1, xs)").expect("portable SSA");
    assert_eq!(portable.global_context.profile.target, CodegenTarget::Portable);

    let spirv = compile_thru_spirv(" entry e(xs: []i32) []i32 = map(|x: i32| x + 1, xs)")
        .expect("SPIR-V-targeted lowering");
    assert!(!spirv.spirv.is_empty());
}

#[test]
fn terminal_scan_helpers_are_complete_region_arena_members() {
    let source = " entry prefix(xs: []i32) []i32 = scan(|a: i32, b: i32| a + b, 0, xs)";
    let allocated = compile_to_semantic_egir(source);
    assert!(
        !allocated.functions.iter().any(|function| function.name.ends_with("_scan_op_swap")),
        "planner-generated scan helper leaked into semantic EGIR"
    );
    let planned_callables =
        egir::parallelize::tests::planned_callable_names(compile_to_semantic_egir(source))
            .expect("parallel schedule");
    assert!(
        planned_callables.iter().any(|name| name.ends_with("_scan_op_swap")),
        "scan helper must be owned by the kernel plan"
    );
    let physical = egir::plan(allocated, LoweringProfile::PORTABLE).expect("terminal schedule");
    let helper = physical
        .functions
        .iter()
        .find(|function| function.name.ends_with("_scan_op_swap"))
        .expect("scan swap helper");
    assert!(physical.contains_region(helper.region));
}

/// Assert that a compute `reduce`-over-`map`-of-range `src` parallelizes and
/// that phase1's per-thread loop trip-count transitively depends on
/// `thread_id` — i.e. each thread reduces only its *chunk* of the range.
///
/// The trip count must therefore depend on `thread_id`; a raw,
/// thread-independent input length would make every invocation reduce the
/// entire range.
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
        .filter(|(_, block)| matches!(block.control_header, Some(ControlHeader::Loop { .. })))
        .filter_map(|(_, block)| match &block.term {
            Terminator::CondBranch { cond, .. } => cond.as_ssa(),
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
/// The descriptor must therefore contain a writer for every output buffer;
/// `spirv-val` additionally checks that the partials store has tuple type.
#[test]
fn tuple_reduce_writes_every_output_field() {
    use crate::pipeline_descriptor::{Binding, BufferUsage, Pipeline};
    let lowered = compile_thru_spirv(
        r#"
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
    let lowered = compile_thru_spirv(
        r#"
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
    let lowered = compile_thru_spirv(
        r#"
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

#[test]
fn captured_reduce_operator_stays_parallel() {
    use crate::pipeline_descriptor::Pipeline;
    let lowered = compile_thru_spirv(
        r#"
entry captured_reduce(xs: []i32, modes: []i32) i32 =
  reduce(
    |a: i32, b: i32| if modes[0] > 0 then a + b else a * b,
    1,
    xs)
"#,
    )
    .expect("capturing reduction operator compiles");
    let stages = lowered.pipeline.pipelines.iter().find_map(|pipeline| match pipeline {
        Pipeline::Compute(compute)
            if compute.stages.iter().any(|stage| stage.entry_point == "captured_reduce") =>
        {
            Some(&compute.stages)
        }
        _ => None,
    });
    assert_eq!(stages.expect("captured_reduce pipeline").len(), 2);
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
    let lowered = compile_thru_spirv(
        r#"
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
    let lowered = compile_thru_spirv(
        r#"
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
    let lowered = compile_thru_spirv(
        r#"
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
fn captured_scan_operator_stays_parallel() {
    use crate::pipeline_descriptor::Pipeline;
    let lowered = compile_thru_spirv(
        r#"
entry captured_scan(xs: []i32, modes: []i32) []i32 =
  scan(
    |a: i32, b: i32| if modes[0] > 0 then a + b else a * b,
    1,
    xs)
"#,
    )
    .expect("capturing scan operator compiles");
    let stages = lowered.pipeline.pipelines.iter().find_map(|pipeline| match pipeline {
        Pipeline::Compute(compute)
            if compute.stages.iter().any(|stage| stage.entry_point == "captured_scan") =>
        {
            Some(&compute.stages)
        }
        _ => None,
    });
    assert_eq!(stages.expect("captured_scan pipeline").len(), 3);
}
#[test]
fn tuple_element_scan_stays_parallel() {
    use crate::egir::types::{SideEffectKind, Soac, SoacEffect};

    let source = r#"
entry tuple_prefixes(xs: [8](i32, i32)) [8](i32, i32) =
  scan(
    |(sum1, max1): (i32, i32), (sum2, max2): (i32, i32)|
      (sum1 + sum2, if max1 > max2 then max1 else max2),
    (0, -2147483648),
    xs)
"#;
    let allocated = compile_to_semantic_egir(source);
    let scan = allocated_entries(&allocated)
        .flat_map(|entry| entry.graph.skeleton.blocks.iter().flat_map(|(_, block)| &block.side_effects))
        .find_map(|effect| {
            let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &effect.kind else {
                return None;
            };
            (!op.form.scans.is_empty() && op.form.reductions.is_empty()).then_some(&op.form.scans)
        })
        .expect("tuple scan remains canonical");
    assert_eq!(scan.len(), 1);
    assert_eq!(scan[0].neutral.len(), 1);
    compile_thru_spirv(source).expect("tuple-element scan compiles through the parallel recipe");
}
#[test]
fn range_map_dispatch_uses_range_length() {
    use crate::pipeline_descriptor::{DispatchLen, DispatchSize, Pipeline};
    let lowered = compile_thru_spirv(
        r#"
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

/// A map over a fixed `iota` keeps its iteration domain when the array is
/// returned directly from a helper. The helper-return boundary must preserve
/// the 1024-element dispatch instead of falling back to `Fixed { 1, 1, 1 }`.
#[test]
fn iota_map_returned_from_helper_keeps_dispatch_domain() {
    use crate::pipeline_descriptor::{DispatchLen, DispatchSize, Pipeline};
    let source = r#"
def f(n: i32) []f32 =
  map(|i| f32(i + n), iota(1024))

entry gen(events: []vec4f32) []f32 =
  f(0)
"#;
    let lowered = compile_thru_spirv(source).expect("iota map returned from a helper compiles");
    let stage = lowered
        .pipeline
        .pipelines
        .iter()
        .find_map(|pipeline| match pipeline {
            Pipeline::Compute(compute) => compute.stages.iter().find(|stage| stage.entry_point == "gen"),
            _ => None,
        })
        .expect("gen stage");
    assert_eq!(
        stage.dispatch_size,
        DispatchSize::DerivedFrom {
            len: DispatchLen::Fixed { count: 1024 },
            workgroup_size: 64,
        }
    );
}

#[test]
fn scalar_prepass_and_consumer_share_one_scheduled_pipeline() {
    use crate::pipeline_descriptor::Pipeline;
    let lowered = compile_thru_spirv(
        r#"
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
                    stage.entry_point == "add_sum" || stage.entry_point.contains("add_sum_prepass_")
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
        "two reduction phases followed by the map consumer: {stages:?}"
    );
    assert!(stages[0].entry_point.contains("add_sum_prepass_"));
    assert!(stages[1].entry_point.contains("phase2"));
    assert_eq!(stages[2].entry_point, "add_sum");
    assert!(
        stages.iter().all(|stage| stage.owner == "add_sum"),
        "all phases in the generated pipeline retain the authored owner"
    );
    assert!(
        stages[..2].iter().all(|stage| stage.entry_point.starts_with("add_sum_")),
        "generated phase names retain the authored-entry prefix"
    );
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

const SCALAR_PRELUDE_FOLD: &str = r#"
def fold_events(events: []u32) u32 =
  loop state = 0u32 for k < 32 do
    (state ^ events[k]) * 1664525u32 + 1013904223u32
"#;

fn scalar_prelude_pipeline<'a>(
    lowered: &'a Lowered,
    source_entry: &str,
) -> &'a pipeline_descriptor::ComputePipeline {
    use crate::pipeline_descriptor::Pipeline;
    let mut pipelines = lowered.pipeline.pipelines.iter().filter_map(|pipeline| match pipeline {
        Pipeline::Compute(compute)
            if compute.stages.iter().any(|stage| stage.entry_point == source_entry) =>
        {
            Some(compute)
        }
        _ => None,
    });
    let pipeline = pipelines.next().expect("source entry has a compute pipeline");
    assert!(
        pipelines.next().is_none(),
        "source entry must publish one coherent compute pipeline"
    );
    pipeline
}

fn is_singleton_stage(stage: &pipeline_descriptor::ComputeStage) -> bool {
    use crate::pipeline_descriptor::DispatchSize;
    stage.workgroup_size == (1, 1, 1)
        && matches!(stage.dispatch_size, DispatchSize::Fixed { x: 1, y: 1, z: 1, .. })
}

fn spirv_entry_reaches_loop(spirv: &[u32], entry_name: &str) -> bool {
    use std::collections::{HashMap, HashSet};
    use wspirv::binary::parse_words;
    use wspirv::dr::{Loader, Operand};
    use wspirv::spirv::Op;

    let mut loader = Loader::new();
    parse_words(spirv, &mut loader).expect("parse generated SPIR-V");
    let module = loader.module();
    let Some(entry) = module
        .entry_points
        .iter()
        .find(|instruction| {
            matches!(instruction.operands.get(2), Some(Operand::LiteralString(name)) if name == entry_name)
        })
    else {
        return false;
    };
    let Some(Operand::IdRef(entry_function)) = entry.operands.get(1) else {
        return false;
    };
    let mut calls = HashMap::<u32, Vec<u32>>::new();
    let mut loops = HashSet::new();
    for function in &module.functions {
        let Some(function_id) = function.def.as_ref().and_then(|definition| definition.result_id) else {
            continue;
        };
        for instruction in function.blocks.iter().flat_map(|block| &block.instructions) {
            if instruction.class.opcode == Op::LoopMerge {
                loops.insert(function_id);
            }
            if instruction.class.opcode == Op::FunctionCall {
                if let Some(Operand::IdRef(callee)) = instruction.operands.first() {
                    calls.entry(function_id).or_default().push(*callee);
                }
            }
        }
    }
    let mut pending = vec![*entry_function];
    let mut reachable = HashSet::new();
    while let Some(function) = pending.pop() {
        if !reachable.insert(function) {
            continue;
        }
        pending.extend(calls.get(&function).into_iter().flatten().copied());
    }
    reachable.iter().any(|function| loops.contains(function))
}

fn spirv_entry_interface_has_binding(spirv: &[u32], entry_name: &str, set: u32, binding: u32) -> bool {
    use std::collections::HashMap;
    use wspirv::binary::parse_words;
    use wspirv::dr::{Loader, Operand};
    use wspirv::spirv::{Decoration, Op};

    let mut loader = Loader::new();
    parse_words(spirv, &mut loader).expect("parse generated SPIR-V");
    let module = loader.module();
    let mut sets = HashMap::new();
    let mut bindings = HashMap::new();
    for instruction in &module.annotations {
        if instruction.class.opcode != Op::Decorate {
            continue;
        }
        let Some(Operand::IdRef(target)) = instruction.operands.first() else {
            continue;
        };
        match (instruction.operands.get(1), instruction.operands.get(2)) {
            (Some(Operand::Decoration(Decoration::DescriptorSet)), Some(Operand::LiteralBit32(value))) => {
                sets.insert(*target, *value);
            }
            (Some(Operand::Decoration(Decoration::Binding)), Some(Operand::LiteralBit32(value))) => {
                bindings.insert(*target, *value);
            }
            _ => {}
        }
    }
    let Some(entry) = module.entry_points.iter().find(|instruction| {
        matches!(instruction.operands.get(2), Some(Operand::LiteralString(name)) if name == entry_name)
    }) else {
        return false;
    };
    entry.operands.iter().skip(3).any(|operand| {
        let Operand::IdRef(variable) = operand else {
            return false;
        };
        sets.get(variable) == Some(&set) && bindings.get(variable) == Some(&binding)
    })
}

fn spirv_entry_storage_binding_is_writable(
    spirv: &[u32],
    entry_name: &str,
    set: u32,
    binding: u32,
) -> Option<bool> {
    use std::collections::HashMap;
    use std::collections::HashSet;
    use wspirv::binary::parse_words;
    use wspirv::dr::{Loader, Operand};
    use wspirv::spirv::{Decoration, Op};

    let mut loader = Loader::new();
    parse_words(spirv, &mut loader).expect("parse generated SPIR-V");
    let module = loader.module();
    let storage_variables = module
        .types_global_values
        .iter()
        .filter_map(|instruction| {
            (instruction.class.opcode == Op::Variable
                && matches!(
                    instruction.operands.first(),
                    Some(Operand::StorageClass(wspirv::spirv::StorageClass::StorageBuffer))
                ))
            .then_some(instruction.result_id?)
        })
        .collect::<HashSet<_>>();
    let mut sets = HashMap::new();
    let mut bindings = HashMap::new();
    let mut nonwritable = HashSet::new();
    for instruction in &module.annotations {
        if instruction.class.opcode != Op::Decorate {
            continue;
        }
        let Some(Operand::IdRef(target)) = instruction.operands.first() else {
            continue;
        };
        match (instruction.operands.get(1), instruction.operands.get(2)) {
            (Some(Operand::Decoration(Decoration::DescriptorSet)), Some(Operand::LiteralBit32(value))) => {
                sets.insert(*target, *value);
            }
            (Some(Operand::Decoration(Decoration::Binding)), Some(Operand::LiteralBit32(value))) => {
                bindings.insert(*target, *value);
            }
            (Some(Operand::Decoration(Decoration::NonWritable)), None) => {
                nonwritable.insert(*target);
            }
            _ => {}
        }
    }
    let entry = module.entry_points.iter().find(|instruction| {
        matches!(instruction.operands.get(2), Some(Operand::LiteralString(name)) if name == entry_name)
    })?;
    entry.operands.iter().skip(3).find_map(|operand| {
        let Operand::IdRef(variable) = operand else {
            return None;
        };
        (storage_variables.contains(variable)
            && sets.get(variable) == Some(&set)
            && bindings.get(variable) == Some(&binding))
        .then(|| !nonwritable.contains(variable))
    })
}

fn assert_expensive_scalar_prefix_pipeline(lowered: &Lowered, source_entry: &str) {
    use crate::pipeline_descriptor::{Binding, DispatchLen, DispatchSize};

    let pipeline = scalar_prelude_pipeline(lowered, source_entry);
    let stages = &pipeline.stages;
    assert_eq!(stages.len(), 3, "one singleton plus two map stages");
    let singleton = stages
        .iter()
        .find(|stage| is_singleton_stage(stage))
        .expect("expensive source runs in a singleton stage");
    let maps = stages.iter().filter(|stage| !is_singleton_stage(stage)).collect::<Vec<_>>();
    assert_eq!(maps.len(), 2);
    assert!(maps.iter().all(|stage| matches!(
        stage.dispatch_size,
        DispatchSize::DerivedFrom {
            len: DispatchLen::InputBinding { .. },
            ..
        }
    )));
    assert_ne!(
        maps[0].dispatch_size, maps[1].dispatch_size,
        "maps retain their independent input dispatch domains"
    );
    assert_eq!(
        singleton
            .writes
            .iter()
            .filter(|binding| maps.iter().all(|map| map.reads.contains(binding)))
            .count(),
        1,
        "one singleton-written scalar feeds both maps"
    );
    let (events, events_set, events_binding) = pipeline
        .bindings
        .iter()
        .enumerate()
        .find_map(|(index, binding)| match binding {
            Binding::StorageBuffer {
                set, binding, name, ..
            } if name == "events" => Some((index, *set, *binding)),
            _ => None,
        })
        .expect("the source events input remains published");
    assert!(
        singleton.reads.contains(&events),
        "the singleton reads the producer-only input"
    );
    assert!(
        maps.iter().all(|map| !map.reads.contains(&events)),
        "materialized consumers do not retain producer-only resource reads"
    );
    assert!(
        maps.iter().all(|map| {
            !spirv_entry_interface_has_binding(&lowered.spirv, &map.entry_point, events_set, events_binding)
        }),
        "materialized consumers do not retain the producer-only SPIR-V interface binding"
    );
    assert!(
        spirv_entry_reaches_loop(&lowered.spirv, &singleton.entry_point),
        "the expensive loop is reachable from the singleton"
    );
    assert!(
        maps.iter().all(|map| !spirv_entry_reaches_loop(&lowered.spirv, &map.entry_point)),
        "the expensive loop is not reachable from either map stage"
    );
}

#[test]
fn expensive_scalar_source_is_one_singleton_feeding_two_map_domains() {
    let source = format!(
        "{SCALAR_PRELUDE_FOLD}\n\
         \n\
         entry serial_prefix_before_maps(xs: []u32, ys: []u32, events: []u32) ([]u32, []u32) =\n\
           let state = fold_events(events)\n\
           let out_x = map(|x| x + state, xs)\n\
           let out_y = map(|y| y ^ state, ys) in\n\
           (out_x, out_y)\n"
    );
    let lowered = compile_thru_spirv(&source).expect("expensive scalar source compiles");
    assert_expensive_scalar_prefix_pipeline(&lowered, "serial_prefix_before_maps");
}

#[test]
fn direct_loop_scalar_prefix_uses_the_general_residency_policy() {
    let lowered = compile_thru_spirv(
        r#"
entry direct_loop_prefix(xs: []u32, ys: []u32, events: []u32) ([]u32, []u32) =
  let state =
    loop state = 0u32 for k < 32 do
      (state ^ events[k]) * 1664525u32 + 1013904223u32
  let out_x = map(|x| x + state, xs)
  let out_y = map(|y| y ^ state, ys) in
  (out_x, out_y)
"#,
    )
    .expect("direct structured loop prefix compiles");
    assert_expensive_scalar_prefix_pipeline(&lowered, "direct_loop_prefix");
}

#[test]
fn composite_serial_prefix_is_one_singleton_feeding_two_map_domains() {
    use crate::pipeline_descriptor::{Binding, DispatchLen, DispatchSize};

    let lowered = compile_thru_spirv(
        r#"
def serial_prefix(events: []i32) (i32, [1]i32) =
  loop (sum, last) = (0, [0]) for k < 32 do
    (sum + events[k], [events[k]])

entry serial_prefix_composite_two_maps(events: []i32) ([]i32, []i32) =
  let (sum, last) = serial_prefix(events) in
  (map(|i| i + sum + last[0], iota(1024)),
   map(|i| i + sum, iota(128)))
"#,
    )
    .expect("composite serial prefix compiles");
    let pipeline = scalar_prelude_pipeline(&lowered, "serial_prefix_composite_two_maps");
    assert_eq!(pipeline.stages.len(), 3, "one singleton plus two map stages");
    let singleton = pipeline
        .stages
        .iter()
        .find(|stage| is_singleton_stage(stage))
        .expect("the composite prefix runs in a singleton stage");
    let maps = pipeline.stages.iter().filter(|stage| !is_singleton_stage(stage)).collect::<Vec<_>>();
    assert_eq!(maps.len(), 2);
    assert!(maps.iter().any(|stage| matches!(
        stage.dispatch_size,
        DispatchSize::DerivedFrom {
            len: DispatchLen::Fixed { count: 1024 },
            ..
        }
    )));
    assert!(maps.iter().any(|stage| matches!(
        stage.dispatch_size,
        DispatchSize::DerivedFrom {
            len: DispatchLen::Fixed { count: 128 },
            ..
        }
    )));
    assert_eq!(
        singleton
            .writes
            .iter()
            .filter(|binding| maps.iter().all(|map| map.reads.contains(binding)))
            .count(),
        1,
        "one materialized composite feeds both maps"
    );
    let events = pipeline
        .bindings
        .iter()
        .position(|binding| matches!(binding, Binding::StorageBuffer { name, .. } if name == "events"))
        .expect("the source events input remains published");
    assert!(singleton.reads.contains(&events));
    assert!(
        maps.iter().all(|map| !map.reads.contains(&events)),
        "parallel consumers do not retain the prefix input"
    );
    assert!(spirv_entry_reaches_loop(&lowered.spirv, &singleton.entry_point));
    assert!(
        maps.iter().all(|map| !spirv_entry_reaches_loop(&lowered.spirv, &map.entry_point)),
        "the serial fold is not cloned into either parallel map"
    );
}

#[test]
fn composite_serial_prefix_is_shared_by_fixed_outputs_and_parallel_maps() {
    use crate::pipeline_descriptor::{Binding, DispatchLen, DispatchSize};

    let lowered = compile_thru_spirv(
        r#"
def serial_prefix(events: []i32) (i32, [1]i32) =
  loop (sum, last) = (0, [0]) for k < 32 do
    (sum + events[k], [events[k]])

entry serial_prefix_mixed_consumers(events: []i32)
  ([1]i32, []i32, []i32, [1]i32) =
  let (sum, last) = serial_prefix(events) in
  ([sum],
   map(|i| i + sum + last[0], iota(1024)),
   map(|i| i + sum, iota(128)),
   last)
"#,
    )
    .expect("mixed composite-prefix consumers compile");
    let pipeline = scalar_prelude_pipeline(&lowered, "serial_prefix_mixed_consumers");
    let events = pipeline
        .bindings
        .iter()
        .position(|binding| matches!(binding, Binding::StorageBuffer { name, .. } if name == "events"))
        .expect("the source events input remains published");
    let prefix_stages =
        pipeline.stages.iter().filter(|stage| stage.reads.contains(&events)).collect::<Vec<_>>();
    assert_eq!(prefix_stages.len(), 1, "the serial input is read by one producer");
    let prefix = prefix_stages[0];
    assert!(
        is_singleton_stage(prefix),
        "the serial prefix producer executes once"
    );
    assert!(spirv_entry_reaches_loop(&lowered.spirv, &prefix.entry_point));

    let consumers =
        pipeline.stages.iter().filter(|stage| stage.entry_point != prefix.entry_point).collect::<Vec<_>>();
    assert!(
        consumers.iter().all(|stage| !spirv_entry_reaches_loop(&lowered.spirv, &stage.entry_point)),
        "neither fixed-output writers nor parallel maps replay the prefix loop"
    );
    assert_eq!(
        prefix
            .writes
            .iter()
            .filter(|binding| consumers.iter().all(|stage| stage.reads.contains(binding)))
            .count(),
        1,
        "one composite handoff feeds every fixed and parallel consumer"
    );
    let maps = consumers.iter().filter(|stage| !is_singleton_stage(stage)).copied().collect::<Vec<_>>();
    assert_eq!(maps.len(), 2);
    assert!(maps.iter().any(|stage| matches!(
        stage.dispatch_size,
        DispatchSize::DerivedFrom {
            len: DispatchLen::Fixed { count: 1024 },
            ..
        }
    )));
    assert!(maps.iter().any(|stage| matches!(
        stage.dispatch_size,
        DispatchSize::DerivedFrom {
            len: DispatchLen::Fixed { count: 128 },
            ..
        }
    )));
}

#[test]
fn mixed_fixed_parallel_output_preserves_independent_load_and_serial_prepass() {
    use crate::pipeline_descriptor::{Binding, DispatchLen, DispatchSize};

    let lowered = compile_thru_spirv(
        r#"
entry mixed_fixed_parallel_prefix_ice(events: []i32) ([1]i32, []i32) =
  let sum = loop total = 0 for k < 1 do total + events[k] in
  ([events[0]], map(|i| i + sum, iota(1)))
"#,
    )
    .expect("independent fixed output and serial-prefix map compile");
    let pipeline = scalar_prelude_pipeline(&lowered, "mixed_fixed_parallel_prefix_ice");
    assert_eq!(pipeline.stages.len(), 3, "fixed output, serial prepass, and map");
    let events = pipeline
        .bindings
        .iter()
        .position(|binding| matches!(binding, Binding::StorageBuffer { name, .. } if name == "events"))
        .expect("the source events input remains published");
    let loop_stages = pipeline
        .stages
        .iter()
        .filter(|stage| spirv_entry_reaches_loop(&lowered.spirv, &stage.entry_point))
        .collect::<Vec<_>>();
    assert_eq!(
        loop_stages.len(),
        1,
        "the serial producer remains present exactly once"
    );
    let prepass = loop_stages[0];
    assert!(is_singleton_stage(prepass));
    assert!(prepass.reads.contains(&events));

    let map = pipeline.stages.iter().find(|stage| !is_singleton_stage(stage)).expect("parallel map stage");
    assert!(matches!(
        map.dispatch_size,
        DispatchSize::DerivedFrom {
            len: DispatchLen::Fixed { count: 1 },
            ..
        }
    ));
    assert!(
        !map.reads.contains(&events),
        "the map reads the handoff, not the prefix input"
    );
    assert_eq!(
        prepass.writes.iter().filter(|binding| map.reads.contains(binding)).count(),
        1,
        "the serial prepass dominates the map through one handoff"
    );

    let fixed = pipeline
        .stages
        .iter()
        .find(|stage| stage.entry_point != prepass.entry_point && is_singleton_stage(stage))
        .expect("independent fixed-output stage");
    assert!(
        fixed.reads.contains(&events),
        "the fixed output retains its own input load"
    );
    assert!(!spirv_entry_reaches_loop(&lowered.spirv, &fixed.entry_point));
}

#[test]
fn conditional_state_prefix_publishes_all_live_outs_to_mixed_consumers() {
    use crate::pipeline_descriptor::{Binding, DispatchLen, DispatchSize};

    let lowered = compile_thru_spirv(
        r#"
type ui_state = { tool: i32 }
type stroke_head = { count: i32 }
type state_update = { ui: ui_state, head: stroke_head, emit_count: i32 }

def fold_ui(ui_in: []i32, events: []i32) i32 =
  loop tool = ui_in[0] for k < 2 do tool + events[k]

def next_state(ui_in: []i32, head_in: []i32, events: []i32)
  (state_update, [2]i32) =
  let tool = fold_ui(ui_in, events)
  let (count, emitted) = loop (n, out) = (head_in[0], [0, 0]) for k < 2 do
    if events[k] > 0 then
      if n < 2 then (n + 1, out with [n] = events[k]) else (n, out)
    else (n, out) in
  ({ ui = { tool = tool }, head = { count = count }, emit_count = count }, emitted)

def update_points(points_in: []i32, update: state_update, emitted: [2]i32)
  []i32 =
  map(|i| points_in[i] + update.emit_count + emitted[i % 2], iota(1024))

def update_items(items_in: []i32, update: state_update) []i32 =
  map(|i| items_in[i] + update.emit_count, iota(128))

entry mixed_fixed_parallel_prefix_ice(ui_in: []i32, points_in: []i32,
                                      items_in: []i32, head_in: []i32,
                                      events: []i32)
  ([1]i32, []i32, []i32, [1]i32) =
  let (update, emitted) = next_state(ui_in, head_in, events) in
  ([update.ui.tool], update_points(points_in, update, emitted),
   update_items(items_in, update), [update.head.count])
"#,
    )
    .expect("conditional state prefix compiles");
    let pipeline = scalar_prelude_pipeline(&lowered, "mixed_fixed_parallel_prefix_ice");
    let prefix_inputs = pipeline
        .bindings
        .iter()
        .enumerate()
        .filter_map(|(index, binding)| {
            matches!(binding, Binding::StorageBuffer { name, .. }
                if name == "ui_in" || name == "head_in" || name == "events")
            .then_some(index)
        })
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(prefix_inputs.len(), 3);
    let loop_stages = pipeline
        .stages
        .iter()
        .filter(|stage| spirv_entry_reaches_loop(&lowered.spirv, &stage.entry_point))
        .collect::<Vec<_>>();
    assert_eq!(
        loop_stages.len(),
        1,
        "both ordered folds remain in one producer stage"
    );
    let prepass = loop_stages[0];
    assert!(is_singleton_stage(prepass));
    assert!(prefix_inputs.iter().all(|binding| prepass.reads.contains(binding)));

    let consumers =
        pipeline.stages.iter().filter(|stage| stage.entry_point != prepass.entry_point).collect::<Vec<_>>();
    assert!(
        consumers.iter().all(|stage| prefix_inputs.iter().all(|binding| !stage.reads.contains(binding))),
        "fixed and parallel consumers read handoffs instead of rebuilding the state prefix"
    );
    assert!(consumers.iter().all(|stage| !spirv_entry_reaches_loop(&lowered.spirv, &stage.entry_point)));
    let maps = consumers.iter().filter(|stage| !is_singleton_stage(stage)).copied().collect::<Vec<_>>();
    assert_eq!(maps.len(), 2);
    assert!(maps.iter().any(|stage| matches!(
        stage.dispatch_size,
        DispatchSize::DerivedFrom {
            len: DispatchLen::Fixed { count: 1024 },
            ..
        }
    )));
    assert!(maps.iter().any(|stage| matches!(
        stage.dispatch_size,
        DispatchSize::DerivedFrom {
            len: DispatchLen::Fixed { count: 128 },
            ..
        }
    )));
    assert_eq!(
        prepass
            .writes
            .iter()
            .filter(|binding| maps.iter().all(|stage| stage.reads.contains(binding)))
            .count(),
        2,
        "the stroke tuple and earlier tool result both cross the prepass boundary"
    );
}

#[test]
fn conditional_scalar_prefix_uses_the_general_residency_policy() {
    let source = format!(
        "{SCALAR_PRELUDE_FOLD}\n\
         \n\
         entry conditional_prefix(xs: []u32, ys: []u32, events: []u32) ([]u32, []u32) =\n\
           let state = if events[0] == 0u32\n\
                       then fold_events(events)\n\
                       else fold_events(events) ^ 1u32\n\
           let out_x = map(|x| x + state, xs)\n\
           let out_y = map(|y| y ^ state, ys) in\n\
           (out_x, out_y)\n"
    );
    let lowered = compile_thru_spirv(&source).expect("conditional structured prefix compiles");
    assert_expensive_scalar_prefix_pipeline(&lowered, "conditional_prefix");
}

#[test]
fn expensive_scalar_source_is_profitable_for_one_map() {
    let source = format!(
        "{SCALAR_PRELUDE_FOLD}\n\
         \n\
         entry serial_prefix_one_map(xs: []u32, events: []u32) []u32 =\n\
           let state = fold_events(events) in\n\
           map(|x| x + state, xs)\n"
    );
    let lowered = compile_thru_spirv(&source).expect("single-map scalar source compiles");
    let stages = &scalar_prelude_pipeline(&lowered, "serial_prefix_one_map").stages;
    assert_eq!(stages.len(), 2, "one singleton and one map stage");
    assert_eq!(stages.iter().filter(|stage| is_singleton_stage(stage)).count(), 1);
}

fn assert_scalar_prefix_emits_valid_wgsl(source: &str) {
    let wgsl = lower_ssa_to_wgsl(lower_semantic_egir(
        compile_to_semantic_egir(source),
        LoweringProfile::new(CodegenTarget::Wgsl, SchedulePolicy::Parallel),
    ))
    .expect("scalar prepass lowers to WGSL");
    let module = naga::front::wgsl::parse_str(&wgsl)
        .unwrap_or_else(|error| panic!("Naga rejected generated WGSL: {error:?}\n{wgsl}"));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|error| panic!("Naga validation failed: {error:?}\n{wgsl}"));
}

#[test]
fn expensive_scalar_source_emits_valid_wgsl() {
    let source = format!(
        "{SCALAR_PRELUDE_FOLD}\n\
         \n\
         entry serial_prefix_wgsl(xs: []u32, events: []u32) []u32 =\n\
           let state = fold_events(events) in\n\
           map(|x| x + state, xs)\n"
    );
    assert_scalar_prefix_emits_valid_wgsl(&source);
}

#[test]
fn structured_scalar_prefix_emits_valid_wgsl() {
    assert_scalar_prefix_emits_valid_wgsl(
        r#"
entry direct_loop_prefix_wgsl(xs: []u32, events: []u32) []u32 =
  let state =
    loop state = 0u32 for k < 32 do
      (state ^ events[k]) * 1664525u32 + 1013904223u32 in
  map(|x| x + state, xs)
"#,
    );
}

#[test]
fn cheap_scalar_source_stays_cloned_into_two_maps() {
    let lowered = compile_thru_spirv(
        r#"
entry cheap_prefix(xs: []u32, ys: []u32, factor: u32) ([]u32, []u32) =
  let state = factor * 3u32
  let out_x = map(|x| x + state, xs)
  let out_y = map(|y| y ^ state, ys) in
  (out_x, out_y)
"#,
    )
    .expect("cheap scalar source compiles");
    let stages = &scalar_prelude_pipeline(&lowered, "cheap_prefix").stages;
    assert_eq!(stages.len(), 2, "cheap multiplication must not create a prepass");
    assert!(stages.iter().all(|stage| !is_singleton_stage(stage)));
    assert!(
        stages.iter().enumerate().all(|(writer_index, writer)| {
            writer.writes.iter().all(|binding| {
                stages.iter().enumerate().all(|(reader_index, reader)| {
                    writer_index == reader_index || !reader.reads.contains(binding)
                })
            })
        }),
        "cheap duplication must not create an inter-stage scalar binding"
    );
}

#[test]
fn scalar_prepass_flow_is_explicit_in_resource_manifest() {
    use crate::egir::program::{CompilerResourceKind, ResourceOrigin};

    let allocated = compile_to_semantic_egir(
        r#"
entry add_sum(xs: []i32) []i32 =
  let total = reduce(|a: i32, b: i32| a + b, 0, xs) in
  map(|x: i32| x + total, xs)
"#,
    );
    let resource = allocated
        .logical_resources()
        .iter()
        .find(|resource| match resource.origin() {
            ResourceOrigin::Compiler {
                resource: compiler, ..
            } if compiler.kind == CompilerResourceKind::ScalarHandoff => true,
            _ => false,
        })
        .expect("scalar handoff resource");
    let flow = allocated
        .data
        .stages
        .flows()
        .find_map(|(_, flow)| (flow.storage().data == resource.id()).then_some(flow))
        .expect("scalar handoff has an explicit staged flow");
    let producer = allocated.data.stages.stage(flow.producer()).expect("staged producer");
    assert_eq!(
        producer.origin().generated_kind(),
        Some(egir::program::GeneratedStageKind::Scalar)
    );
    assert!(producer.body().name.contains("prepass_scalar"));
    assert_eq!(flow.consumers().len(), 1);
    let consumer = allocated.data.stages.stage(flow.consumers()[0]).expect("staged consumer");
    assert_eq!(consumer.body().name, "add_sum");
}

#[test]
fn scalar_prepass_precedes_every_phase_of_an_expanded_filter_consumer() {
    use crate::pipeline_descriptor::Pipeline;

    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            let lowered = compile_thru_spirv(
                r#"
def clampi(v: i32, hi: i32) i32 = if v > hi then hi else v

def tile(f: vec2f32, front: bool) (f32, i32, i32, bool) =
  if !front then (0.0, 0i32, 0i32, false)
  else (0.0, 0i32, i32(f.y), true)

def history_visible(f: vec2f32, history: render_target<f32>) bool =
  let (_, _, _, infront) = tile(f, true) in
  if !infront then false
  else
    let sampled =
      loop m = 0.0 for k < 1 do
        let x = clampi(k, (i32(f.x) + 1) - 1) in
        0.0
    in sampled > 0.0

def inner(a: i32, b: i32) (i32, i32) = (1i32, b)
def outer(v: i32) (i32, i32) = inner(0i32, v)

def other_visible(i: i32) bool =
  let (_, v) = outer(i) in v > 0

def keep(f: vec2f32, history: render_target<f32>, i: i32) bool =
  if i < 1 then history_visible(f, history) else other_visible(i)

entry scheduler_resource_cycle(f: vec2f32, history: render_target<f32>)
    ([]i32, render_target<f32>) =
  let visible =
    let kept = filter(|i| keep(f, history, i), iota(2)) in
    { values = map(|i| 0i32, kept) }
  let raster = rasterize_triangles(
    direct_draw(3u32, 1u32),
    |_| vertex_output(@[0.0, 0.0, 0.0, 1.0], ()))
  let history' = shade(history, raster, |_| 1.0)
  in (visible.values, history')
"#,
            )
            .expect("a nested scalar prepass must be inserted before filter flags and scan phases");
            let stages = lowered
                .pipeline
                .pipelines
                .iter()
                .find_map(|pipeline| match pipeline {
                    Pipeline::Compute(compute)
                        if compute
                            .stages
                            .iter()
                            .any(|stage| stage.entry_point.contains("_filter_flags")) =>
                    {
                        Some(&compute.stages)
                    }
                    _ => None,
                })
                .expect("expanded filter pipeline");
            let prepass = stages
                .iter()
                .position(|stage| stage.entry_point.contains("_prepass_scalar_"))
                .expect("nested scalar prepass");
            let flags = stages
                .iter()
                .position(|stage| stage.entry_point.contains("_filter_flags"))
                .expect("filter flags phase");
            assert!(
                prepass < flags,
                "the scalar prepass must dominate the whole filter phase chain: {stages:?}"
            );
        })
        .expect("spawn nested scalar-prepass scheduling regression")
        .join()
        .expect("nested scalar-prepass scheduling regression panicked");
}

// =============================================================================
// Bound-symbol verification through TLC passes
// =============================================================================

/// Walk a term and assert that every `TermKind::Var(VarRef::Symbol(sym))`
/// references a sym that is either:
/// - bound by an enclosing Let / Lambda param / Loop var / SOAC element
///   parameter, or
/// - a top-level def name in `top_level`.
///
/// On violation, panics with the offending sym, its symbol-table name,
/// and the pipeline stage name.
fn assert_no_unbound_var_refs(program: &tlc::stage::Reachable, stage: &str) {
    use crate::tlc::data::{ExplicitCapturesPayload, ExplicitClosurePayload};
    use crate::tlc::{ArrayExpr, Lambda, LoopKind, SoacOp, Term, TermKind};
    use crate::SymbolId;
    use std::collections::HashSet;

    fn walk(
        term: &Term<ExplicitClosurePayload, ExplicitCapturesPayload>,
        bound: &HashSet<SymbolId>,
        symbols: &SymbolTable,
        stage: &str,
        def_name: &str,
    ) {
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
            TermKind::Closure(closure) => {
                for capture in &closure.captures {
                    walk(capture, bound, symbols, stage, def_name);
                }
            }
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
        }
    }

    fn walk_lambda(
        lam: &Lambda<ExplicitClosurePayload, ExplicitCapturesPayload>,
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
        soac: &SoacOp<ExplicitClosurePayload, ExplicitCapturesPayload>,
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
            SoacOp::BucketScatter { lam, inputs, .. } => {
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
        ae: &ArrayExpr<ExplicitClosurePayload, ExplicitCapturesPayload>,
        bound: &HashSet<SymbolId>,
        symbols: &SymbolTable,
        stage: &str,
        def_name: &str,
    ) {
        match ae {
            ArrayExpr::Var(vr, ty) => {
                let t = tlc::synthetic_atom_var_term(*vr, ty.clone());
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
        }
    }

    // Top-level TLC references retain their exact definition SymbolId. The
    // reachable checkpoint stores only retained definitions, so their IDs are
    // the complete top-level binding set at this stage.
    let top_level = program.defs.iter().map(|def| def.name).collect::<HashSet<_>>();
    for def in &program.defs {
        let def_name = program.symbols.get(def.name).cloned().unwrap_or_default();
        walk(&def.body, &top_level, &program.symbols, stage, &def_name);
    }
}

/// `tlc::partial_eval::residualize_call` must substitute `let x = arr` through
/// `body[x]`. Compile through the canonical TLC pipeline and assert that every
/// `Var(Symbol(sym))` remains bound.
#[test]
fn let_binding_substitution_survives_partial_eval() {
    let source = r#"

entry frag() vec4f32 =
    let range = [1, 2, 3, 4] in
    @[f32.i32(range[0]), 0.0, 0.0, 1.0]
"#;
    let tlc = compile_thru_tlc(source).expect("compile_thru_tlc");
    assert_no_unbound_var_refs(&tlc, "compile_thru_tlc");
}

// =============================================================================
// SOAC Fusion Integration Tests
// =============================================================================

#[test]
fn consuming_map_compiles_end_to_end() {
    // `*[N]T` map whose input is dead-after: TLC ownership grants a
    // `UniqueInput` capability, EGIR resolves it to `InputBuffer` from the
    // final use graph, and `soac_expand` emits the in-place loop. Compiling
    // end-to-end through SSA exercises every layer.
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
fn count_uninit_in_program<Tag, GlobalContext>(ssa: &Program<Tag, GlobalContext>) -> usize {
    let mut count = 0;
    let bodies = ssa
        .functions
        .iter()
        .map(|f| &f.body.inner.insts)
        .chain(ssa.entry_points.iter().map(|e| &e.body.inner.insts));
    for insts in bodies {
        for (_id, inst) in insts {
            if let ssa::types::InstKind::Op { tag, .. } = &inst.data {
                match tag {
                    op::OpTag::Intrinsic { id, .. } if *id == builtins::catalog().known().uninit => {
                        count += 1;
                    }
                    _ => {}
                }
            }
        }
    }
    count
}

#[test]
fn consuming_scan_compiles_end_to_end() {
    // Parallel of `consuming_map_compiles_end_to_end` for Scan: `*[N]T`
    // input that's dead-after; ownership grants `UniqueInput`, EGIR resolves
    // it to `InputBuffer`, and `soac_expand` runs the destination-passing
    // loop.
    let _ssa = compile_to_ssa(
        r#"
def cumsum(a: *[8]i32) [8]i32 = scan(|acc: i32, x: i32| acc + x, 0, a)
"#,
    );
}

#[test]
fn scan_destinations_do_not_reintroduce_uninitialized_aggregate_values() {
    let dead_after_ssa = compile_to_ssa(
        r#"

entry frag(c: vec4f32) vec4f32 =
    let xs = [1, 2, 3, 4, 5, 6, 7, 8] in
    let r = scan(|acc: i32, x: i32| acc + x, 0, xs) in
    @[f32.i32(r[0]), f32.i32(r[1]), 0.0, 0.0]
"#,
    );
    assert_eq!(
        count_uninit_in_program(&dead_after_ssa),
        0,
        "destination-directed scan lowering must not construct an uninitialized aggregate value",
    );

    let aliased_ssa = compile_to_ssa(
        r#"

entry frag(c: vec4f32) vec4f32 =
    let xs = [1, 2, 3, 4, 5, 6, 7, 8] in
    let r = scan(|acc: i32, x: i32| acc + x, 0, xs) in
    let j = i32.f32(c.x) % 8 in
    @[f32.i32(r[j]), f32.i32(xs[j]), 0.0, 0.0]
"#,
    );
    assert_eq!(
        count_uninit_in_program(&aliased_ssa),
        0,
        "a Fresh scan destination must be an addressable place",
    );
}

#[test]
fn consuming_scan_compute_entry_compiles_to_spirv() {
    // Compute entry with `*[]T` param. Exercises the Scan-DPS path
    // end-to-end through SPIR-V emission. Required invariants:
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
entry parallel_scan(a: []i32) []i32 = scan(|acc: i32, x: i32| acc + x, 0, a)
"#,
    );

    let wrapper = ssa
        .functions
        .iter()
        .find(|f| f.name.ends_with("_scan_op_swap"))
        .expect("parallel scan should synthesize a swap wrapper Func<Semantic>");

    assert_eq!(
        wrapper.body.params().len(),
        2,
        "swap wrapper must take exactly two params"
    );
    let a_id = wrapper.body.param(0).unwrap().0;
    let b_id = wrapper.body.param(1).unwrap().0;

    let call = wrapper
        .body
        .inner
        .insts
        .values()
        .find_map(|inst| match &inst.data {
            ssa::types::InstKind::Op {
                tag: op::OpTag::Call(name),
                operands,
            } => Some((name.clone(), operands.clone())),
            _ => None,
        })
        .expect("swap wrapper body must contain a Call");

    assert_ne!(
        call.0, wrapper.id,
        "swap wrapper should call the underlying operator, not itself"
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
    // `*[N]T` filter whose input is dead-after: ownership grants
    // `UniqueInput`, EGIR resolves it to `InputBuffer`, and
    // `build_filter_loop` carries the input array as the destination buffer.
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
fn inst_signature_multiset<Tag, GlobalContext>(
    ssa: &Program<Tag, GlobalContext>,
) -> std::collections::BTreeMap<String, usize> {
    use crate::op::OpTag;
    use crate::ssa::types::InstKind;
    use std::collections::BTreeMap;

    let signature = |kind: &InstKind| -> String {
        match kind {
            InstKind::Alloca { .. } => "Alloca".to_string(),
            InstKind::Load { .. } => "Load".to_string(),
            InstKind::Store { .. } => "Store".to_string(),
            InstKind::Atomic { op, .. } => format!("Atomic({op:?})"),
            InstKind::ViewIndex { .. } => "ViewIndex".to_string(),
            InstKind::PlaceIndex { .. } => "PlaceIndex".to_string(),
            InstKind::OutputSlot { .. } => "OutputSlot".to_string(),
            InstKind::ControlBarrier => "ControlBarrier".to_string(),
            InstKind::Op { tag, .. } => format!(
                "Op:{}",
                match tag {
                    OpTag::Call(name) => format!("Call({})", name),
                    OpTag::Intrinsic { id, .. } => {
                        let name = builtins::by_id(*id).raw.surface_name;
                        format!("Intrinsic({})", name)
                    }
                    OpTag::BinOp(op) => format!("BinOp({})", op.symbol()),
                    OpTag::UnaryOp(op) => format!("UnaryOp({})", op.symbol()),
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
                    OpTag::Tuple(_) => "Tuple".to_string(),
                    OpTag::Vector(_) => "Vector".to_string(),
                    OpTag::Matrix { .. } => "Matrix".to_string(),
                    OpTag::ArrayLit(_) => "ArrayLit".to_string(),
                    OpTag::ArrayRange { .. } => "ArrayRange".to_string(),
                    OpTag::Project { .. } => "Project".to_string(),
                    OpTag::Index => "Index".to_string(),
                    OpTag::Materialize => "Materialize".to_string(),
                    OpTag::AddressableConstant(_) => "AddressableConstant".to_string(),
                    OpTag::DynamicExtract => "DynamicExtract".to_string(),
                    OpTag::StorageView(_) => "StorageView".to_string(),
                    OpTag::ResourceLen(_) => "ResourceLen".to_string(),
                    OpTag::StorageViewLen => "StorageViewLen".to_string(),
                    OpTag::StorageImageLoad(_) => "StorageImageLoad".to_string(),
                    OpTag::StorageImageStore(_) => "StorageImageStore".to_string(),
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
    // A bounded filter's `len` field holds the runtime write-cursor count, not
    // its static capacity. On a static-literal filter, the lowered SSA must
    // contain a `_w_intrinsic_length`
    // intrinsic call against the filter result — proving the length
    // is *computed* from the bounded wrapper's runtime `len` field,
    // not short-circuited to the literal capacity.
    let ssa = compile_to_ssa(
        r#"

entry frag(c: vec4f32) vec4f32 =
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
fn test_graphics_map_reduce_end_to_end() {
    let source = r#"
def globalArr: [4]f32 = [10.0, 20.0, 30.0, 40.0]

def myMap(ro: f32, rd: f32) [4]f32 =
  map(|x: f32| x + ro + rd, globalArr)

def myReduce(hits: [4]f32) f32 =
  reduce(|acc: f32, x: f32| if acc < x then acc else x, 999.0, hits)


entry fragment_main() vec4f32 =
  let hits = myMap(1.0, 2.0) in
  let closest = myReduce(hits) in
  @[closest, 0.0, 0.0, 1.0]
"#;

    compile_to_spirv(source).expect("fragment map+reduce should lower to SPIR-V");
}

fn has_soac_kind(term: &tlc::Term, kind: &str) -> bool {
    use crate::tlc::{SoacOp, TermKind};
    match &term.kind {
        TermKind::Soac(SoacOp::Map { .. }) if kind == "Map" => true,
        TermKind::Soac(SoacOp::Reduce { .. }) if kind == "Reduce" => true,
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
entry gen(xs: []i32) ([]i32, [1]i32) =
  let b = map(|x: i32| x + 1, xs) in
  let c = map(|y: i32| y * 2, b) in
  let d = reduce(|acc: i32, z: i32| acc + z, 0, b) in
  (c, [d])
"#;

    let stats = semantic_soac_stats(&compile_to_semantic_egir(source));
    assert_eq!(
        stats.seg_reds, 1,
        "map producer and map+reduce consumers should fuse"
    );
    assert_eq!(stats.seg_maps, 0, "the shared producer should not materialize");

    compile_to_spirv(source).expect("Screma-fused map+reduce should lower to SPIR-V");
}

#[test]
fn test_screma_scan_fusion_end_to_end() {
    let source = r#"
entry gen(xs: []i32) ([]i32, []i32) =
  let b = map(|x: i32| x + 1, xs) in
  let c = map(|y: i32| y * 2, b) in
  let d = scan(|acc: i32, z: i32| acc + z, 0, b) in
  (c, d)
"#;

    let stats = semantic_soac_stats(&compile_to_semantic_egir(source));
    assert_eq!(
        stats.seg_scans, 1,
        "map producer and map+scan consumers should fuse"
    );
    assert_eq!(stats.seg_maps, 0, "the shared producer should not materialize");

    compile_to_spirv(source).expect("Screma-fused map+scan should lower to SPIR-V");
}

#[test]
fn test_screma_multi_output_fusion_end_to_end() {
    let source = r#"
entry gen(xs: []i32) ([]i32, []i32, [1]i32, []i32) =
  let b = map(|x: i32| x + 1, xs) in
  let c = map(|y: i32| y * 2, b) in
  let d = reduce(|acc: i32, z: i32| acc + z, 0, b) in
  let e = map(|w: i32| w - 3, b) in
  let f = scan(|acc: i32, q: i32| acc + q, 0, b) in
  (c, e, [d], f)
"#;

    let stats = semantic_soac_stats(&compile_to_semantic_egir(source));
    assert_eq!(
        stats.seg_reds, 0,
        "the reduction is represented by the mixed canonical Screma"
    );
    assert_eq!(
        stats.seg_scans, 0,
        "the scan is represented by the mixed canonical Screma"
    );
    assert_eq!(
        stats.mixed_scremas, 1,
        "scan and reduction siblings share one canonical Screma"
    );
    assert_eq!(stats.reduce_operators, 1);
    assert_eq!(stats.scan_operators, 1);

    compile_to_spirv(source).expect("multi-output Screma should lower to SPIR-V");
}

// =============================================================================
// Multi-output compute entries
// =============================================================================

/// For a compute entry returning multiple runtime-sized arrays, each tuple
/// field's producing Map or Scan must be retargeted to its own output view
/// before `emit_compute_output_stores`.
#[test]
fn test_multi_output_compute_runtime_sized_arrays() {
    let _ssa = compile_to_ssa(
        r#"
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
/// its bindings from `outputs`. This case combines a computed-fixed output, a
/// shared-domain map pair, and enough distinct domains to reach five outputs.
#[test]
fn multidomain_split_with_shared_domain_map_pair_compiles() {
    use crate::pipeline_descriptor::Pipeline;
    let lowered = compile_thru_spirv(
        r#"
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

    let lowered = compile_thru_spirv(
        r#"
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
entry two(ids: []u32, params: []f32) ([]f32, []f32) =
  let a = params[0] in
  let b = params[1] in
  let lo = map(|id: u32| f32.u32(id) * 1000.0 + a, ids) in
  let hi = map(|id: u32| f32.u32(id) * 7.0 + b, ids) in
  (lo, hi)
"#;

    let wgsl = lower_ssa_to_wgsl(lower_semantic_egir(
        compile_to_semantic_egir(source),
        LoweringProfile::new(CodegenTarget::Wgsl, SchedulePolicy::Parallel),
    ))
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
#[test]
fn inplace_clear_feeding_scatter_stays_wired() {
    use crate::pipeline_descriptor::Pipeline;
    let src = r#"
entry sim(fb: *[]u32, pos: []u32) []u32 =
  let cleared = map(|_p: u32| 0u32, fb) in
  let idxs = map(|p: u32| i32.u32(p), pos) in
  let vals = map(|_p: u32| 1u32, pos) in
  let _ = scatter(cleared, idxs, vals) in
  map(|p: u32| p + 1u32, pos)
"#;
    let lowered = compile_thru_spirv(src).expect("in-place clear + scatter compiles");
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
entry r(a: []u32, b: []u32, fb: *[]u32, pos: []u32)
  ([]u32, []u32) =
  let cleared = map(|_p: u32| 0u32, fb) in
  let idxs = map(|p: u32| i32.u32(p), pos) in
  let vals = map(|_p: u32| 1u32, pos) in
  let _ = scatter(cleared, idxs, vals) in
  (map(|x: u32| x + 1u32, a), map(|y: u32| y + 2u32, b))
"#;

    // Two distinct output domains → two map kernels.
    let lowered = compile_thru_spirv(src).expect("multidomain split + shared scatter compiles");
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
        .filter(|entry| {
            entry
                .body
                .inner
                .blocks
                .values()
                .any(|block| matches!(block.control_header, Some(ControlHeader::Loop { .. })))
        })
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
    let source = r#"
entry two(a: []f32, b: []f32) ([]f32, []f32) =
    (map(|x: f32| x + 1.0, a), map(|x: f32| x + 2.0, b))
"#;
    let lowered = compile_thru_spirv(source).expect("two compiles");

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

/// Splitting one compute entry into multiple fixed-domain map stages must keep
/// the source storage input on every projected entry that reads it. Otherwise
/// the descriptor still publishes the input as read-only while SPIR-V omits
/// `NonWritable`, causing Naga/wgpu to infer a read-write shader binding and
/// reject the pipeline layout.
#[test]
fn multidomain_input_storage_keeps_nonwritable_decoration() {
    use crate::pipeline_descriptor::{Access, Binding, BufferUsage, Pipeline};

    let lowered = compile_thru_spirv(
        r#"
entry gen(data: []f32) ([]f32, []f32) =
  (map(|i| data[i] + 1.0, iota(1024)),
   map(|i| data[i] * 2.0, iota(128)))
"#,
    )
    .expect("multidomain input repro compiles");
    let compute = lowered
        .pipeline
        .pipelines
        .iter()
        .find_map(|pipeline| match pipeline {
            Pipeline::Compute(compute) if compute.stages.iter().any(|stage| stage.entry_point == "gen") => {
                Some(compute)
            }
            _ => None,
        })
        .expect("gen compute pipeline");
    assert_eq!(
        compute.stages.len(),
        2,
        "the two iota domains must split into two stages"
    );
    let (data_index, data_set, data_binding) = compute
        .bindings
        .iter()
        .enumerate()
        .find_map(|(index, binding)| match binding {
            Binding::StorageBuffer {
                set,
                binding,
                access: Access::ReadOnly,
                usage: BufferUsage::Input,
                name,
                ..
            } if name == "data" => Some((index, *set, *binding)),
            _ => None,
        })
        .expect("data is published as a read-only input");

    let readers =
        compute.stages.iter().filter(|stage| stage.reads.contains(&data_index)).collect::<Vec<_>>();
    assert_eq!(readers.len(), 2, "both projected entries read `data`");
    for stage in readers {
        assert_eq!(
            spirv_entry_storage_binding_is_writable(
                &lowered.spirv,
                &stage.entry_point,
                data_set,
                data_binding,
            ),
            Some(false),
            "entry `{}` must use the pipeline's read-only variable",
            stage.entry_point
        );
    }
}

/// Futhark-style horizontal fusion requires siblings to share an input array.
/// A common size variable establishes compatible dispatch domains, but does
/// not justify coupling otherwise independent kernels.
#[test]
fn equal_domain_independent_sibling_maps_remain_separate() {
    use crate::pipeline_descriptor::{DispatchSize, Pipeline};
    let lowered = compile_thru_spirv(
        r#"
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
    assert_eq!(
        compute.stages.len(),
        2,
        "equal extents alone must not couple independent kernels"
    );
    assert!(
        compute.stages.iter().all(|stage| matches!(stage.dispatch_size, DispatchSize::DerivedFrom { .. })),
        "both stages dispatch over their runtime input lengths"
    );
}

/// When split maps capture a scalar produced by loading `table`, every stage
/// still publishes the transitive storage read. The load is a value-producing
/// effect outside the maps, so a pure-node-only walk would leave the source
/// buffer unbound.
#[test]
fn split_stage_reads_include_storage_behind_scalar_producer() {
    use crate::pipeline_descriptor::{Binding, DispatchLen, DispatchSize, Pipeline};
    let lowered = compile_thru_spirv(
        r#"
entry cap(a: []f32, b: []f32, table: []f32) ([]f32, []f32) =
    let scalar = table[0] in
    (map(|x: f32| x + scalar, a), map(|y: f32| y + scalar, b))
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

    let dispatch_binding = |stage: &pipeline_descriptor::ComputeStage| {
        let DispatchSize::DerivedFrom {
            len: DispatchLen::InputBinding { set, binding, .. },
            ..
        } = stage.dispatch_size
        else {
            panic!("map stage must retain its input-buffer dispatch domain")
        };
        compute
            .bindings
            .iter()
            .position(|candidate| {
                matches!(candidate, Binding::StorageBuffer {
                    set: candidate_set,
                    binding: candidate_binding,
                    ..
                } if *candidate_set == set && *candidate_binding == binding)
            })
            .expect("dispatch input is published in the pipeline binding table")
    };
    let domains = compute.stages.iter().map(dispatch_binding).collect::<Vec<_>>();
    assert_ne!(
        domains[0], domains[1],
        "the maps retain their distinct input domains"
    );
    for (stage, domain) in compute.stages.iter().zip(&domains) {
        assert!(
            stage.reads.contains(domain),
            "each map stage reads the input that supplies its dispatch domain"
        );
    }

    let shared_reads = compute.stages[0]
        .reads
        .iter()
        .copied()
        .filter(|binding| compute.stages[1].reads.contains(binding))
        .collect::<Vec<_>>();
    assert_eq!(
        shared_reads.len(),
        1,
        "the scalar source must be the one shared read of both map stages; reads = {:?}",
        compute.stages.iter().map(|stage| &stage.reads).collect::<Vec<_>>()
    );
    assert!(
        matches!(compute.bindings[shared_reads[0]], Binding::StorageBuffer { .. }),
        "the shared scalar producer must retain its storage-buffer source"
    );
    for domain in domains {
        assert_ne!(
            shared_reads[0], domain,
            "the transitive scalar source is separate from each map's dispatch input"
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
    let lowered = compile_thru_spirv(
        r#"
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


entry vertex_main() vec4f32 =
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


entry vertex_main() vec4f32 =
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


entry vertex_main() vec4f32 =
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


entry vertex_main() vec4f32 =
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


entry vertex_main() vec4f32 =
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


entry vertex_main() vec4f32 =
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


entry vertex_main() vec4f32 =
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
/// `map(…, iota(…))` call. Both initializers must produce the Composite
/// representation carried around the loop back edge.
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

entry main(x: []f32) [4]f32 = f(x[0])
"#;
    compile_to_spirv(source).expect(
        "loop back-edge carrying a literal-init array across a map(…, iota(…)) body \
         should compile; both init and body variants are Composite",
    );
}

/// `lift_graphical_invariant_soacs` must follow let bindings transitively when
/// looking for entry-parameter dependencies. A
/// fragment-shader-local `let uv = fragCoord.x` introduces `uv` as a
/// fresh symbol that's *not* an entry param but transitively depends on
/// one. A reduce (plain or fused map→reduce) whose body reads `uv` would then be wrongly
/// classified as graphical-invariant or hoisted into a compute prepass.
#[test]
fn test_no_overhoist_fused_reduce_through_let_bound_dependency() {
    let source = r#"
def cands: [12]i32 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


entry fragment_main(fragCoord: vec4f32)
  vec4f32 =
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


entry vertex_main(vid: i32)
  vec4f32 =
  let x = 2.0 in
  let total = reduce(
    |acc: f32, value: f32| acc + value,
    0.0,
    map(|value: f32| value * x, globalData)
  ) in
  @[total, 0.0, 0.0, 1.0]
"#;

    compile_thru_spirv(source).expect(
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


entry vertex_main(vid: i32)
  vec4f32 =
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
fn test_uniform_reached_through_local_prepass_dependency_is_redeclared() {
    let source = r#"
def samples: [12]f32 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]


entry fragment_main(iTime: f32)
  vec4f32 =
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


entry vertex_main() vec4f32 =
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


entry vertex_main() vec4f32 =
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


entry vertex_main() vec4f32 =
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


entry vertex_main(vertex_id: i32) vec4f32 =
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


entry fragment_main(iResolution: vec2f32, iTime: f32, pos: vec4f32) vec4f32 =
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


entry vertex_main() vec4f32 =
    let result = sum_first_two([1, 2, 3, 4]) in
    @[f32.i32(result), 0.0, 0.0, 1.0]
"#;

    let result = compile_thru_spirv(source);

    assert!(result.is_ok(), "SPIR-V compilation failed: {:?}", result.err());
}

#[test]
fn test_compute_shader_with_storage_slice() {
    // Test compute shader with storage buffer slice
    let source = r#"
def sum_first_two(arr: [4]i32) i32 =
    arr[0] + arr[1]

entry compute_main(data: []i32) i32 =
    let from_storage = sum_first_two(data[0..4]) in
    let from_literal = sum_first_two([1, 2, 3, 4]) in
    from_storage + from_literal
"#;

    let result = compile_thru_spirv(source);

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


entry fragment_main(iTime: f32, pos: vec4f32) vec4f32 =
    let s = compute(pos.x, pos.y) in
    @[s + iTime, 0.0, 0.0, 1.0]
"#;

    let result = compile_thru_spirv(source);

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
    // Monomorphization's intrinsic-specialization step rewrites every
    // `mul(a, b)` call into `BinOp("*")(a, b)`; BinOp lowering picks the right
    // SPIR-V op based on operand shapes. This pins the wiring end-to-
    // end: a single shader exercising all three call shapes must
    // compile through to valid SPIR-V.
    let source = r#"
def m1 = @[[1.0f32, 0.0f32], [0.0f32, 1.0f32]]
def m2 = @[[2.0f32, 0.0f32], [0.0f32, 2.0f32]]


entry fragment_main(pos: vec4f32) vec4f32 =
    let a: mat2f32 = m1 in
    let b: mat2f32 = m2 in
    let v: vec2f32 = @[pos.x, pos.y] in
    let mm: mat2f32 = mul(a, b) in
    let mv: vec2f32 = mul(mm, v) in
    let vm: vec2f32 = mul(v, mm) in
    @[mv.x, mv.y, vm.x, vm.y]
"#;
    let result = compile_thru_spirv(source);
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
        for (val, ty, name) in f.body.params() {
            eprintln!("    param {} ({:?}) :: {:?}", name, val, ty);
        }
        // Show all instructions that involve indexing or storage views
        for inst in f.body.inner.insts.values() {
            match &inst.data {
                ssa::types::InstKind::Op {
                    tag: op::OpTag::Index,
                    ..
                } => {
                    eprintln!("    inst {:?}: Index", inst.result);
                }
                ssa::types::InstKind::Op {
                    tag: op::OpTag::StorageView(_),
                    ..
                } => {
                    eprintln!("    inst {:?}: StorageView", inst.result);
                }
                ssa::types::InstKind::ViewIndex { .. } => {
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
    Ok(compile_thru_spirv(input)?.spirv)
}

/// Single-stage equivalent of `compile_to_spirv` — disables
/// `parallelize_soacs`, exercising the internal serial schedule policy.
fn compile_to_spirv_serial(input: &str) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    Ok(compile_thru_spirv_serial(input)?.spirv)
}

/// Compile module-bearing source through SSA using the shared frontend.
fn compile_to_ssa_with_modules(input: &str) -> ssa::stage::Elaborated {
    compile_thru_ssa(input).expect("compile to SSA")
}

// =========================================================================
// Backend gaps (aspirational, #[ignore]d)
//
// Each test asserts the *desired* code-gen outcome for a construct the SPIR-V
// backend currently can't handle; `#[ignore]`d so the suite stays green.
// Surfaced while building the statistics generators. Drop the `#[ignore]`
// when the gap is closed.
// =========================================================================

/// Returning a runtime-sized `[]f32` from a helper and reading one *constant*
/// slot. `g` inlines to `map(|i| f32.i32(i), 0..<256)`, and EGIR indexed-demand
/// scalarization rewrites `map(f, src)[3]` to a virtual-array
/// access, materializing nothing rather than a whole runtime-sized buffer.
#[test]
fn returning_runtime_sized_array_from_fn_lowers() {
    let source = r#"
def g(n: i32) []f32 = map(|i: i32| f32.i32(i), 0i32 ..< n)
entry e() [1]f32 = [g(256)[3]]
"#;
    compile_to_spirv(source).expect("returning a runtime-sized array should lower to SPIR-V");
}

/// A runtime point demand is as legal to scalarize as a literal one when its
/// index is independent of the producer. The complete map pre-lambda is invoked
/// at `j`, so no runtime-sized intermediate array or gather handoff is needed.
#[test]
fn runtime_index_into_nested_producer_scalarizes() {
    let source = r#"
def g(n: i32) []f32 = map(|i: i32| f32.i32(i), 0i32 ..< n)
entry e(j: i32) [1]f32 = [g(256)[j]]
"#;
    let stats = semantic_soac_stats(&compile_to_semantic_egir(source));
    assert_eq!(
        stats.seg_maps, 0,
        "a producer-independent runtime point demand should not materialize its map"
    );
    compile_to_spirv(source).expect("a scalarized runtime point demand should lower");
}

/// Gap: a runtime-sized array with *two or more* consumers panics the backend
/// with "Composite variant unsized arrays not supported" (`spirv/mod.rs`).
/// A single-consumer runtime-sized array fuses into its consumer and never
/// materializes (`f32.sum(map(…, 0..<n))` lowers fine), but binding it and
/// reading it more than once forces materialization of an unsized Composite
/// array, which the type lowering rejects. This is what blocks the `Stats`
/// gatherer, whose sample array feeds `sum`, a deviation `map`, `minimum`, and
/// `maximum`. Distinct from `returning_runtime_sized_array_from_fn_lowers`,
/// which is about *returning* such an array.
#[test]
fn runtime_sized_array_with_multiple_consumers_lowers() {
    let source = r#"
def g(n: i32) f32 =
    let xs = map(|i: i32| f32.i32(i), 0i32 ..< n) in
    f32.sum(xs) + f32.maximum(xs)
entry e() f32 = g(256)
"#;
    compile_to_spirv(source)
        .expect("a runtime-sized array read by multiple consumers should lower to SPIR-V");
}

/// Named-callee HOF specialization must also rewrite lifted SOAC closures.
/// `fbm2` closes over `noise` through a function-typed capture on its lifted
/// definition. The cascade in `hof_specialize::specialize_higher_order_functions` walks reachable defs, finds
/// `SoacBody`s whose captures include `(_, arrow_ty, Var(known_callable))`,
/// clones the lifted def with the callable substituted into the body,
/// and drops the callable param from its signature. Lets procedural noise
/// collapse its four `fbm_<kind>` defs into one generic `fbm2`.
#[test]
fn function_typed_param_with_named_callee_specializes() {
    let source = r#"
def perlin2(k: u32, p: vec2f32) f32 = f32.u32(k) + p.x
def fbm2(noise: u32 -> vec2f32 -> f32, k: u32, p: vec2f32, n: i32) f32 =
  reduce(|a: f32, b: f32| a + b, 0.0f32,
    map(|i: i32| noise(k, p) * f32.i32(i), 0i32 ..< n))
def fbm_perlin(k: u32, p: vec2f32, n: i32) f32 = fbm2(perlin2, k, p, n)
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
entry arith() i32 =
    i32.(+)(i32.(*)(i32.(-)(10i32, 3i32), 4i32), i32.(%)(9i32, 5i32))
entry bits() u32 = u32.(>>)(u32.(<<)(u32.(^)(u32.(&)(255u32, 15u32), 8u32), 2u32), 1u32)
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
/// which it does support).
#[test]
fn reified_operator_passed_as_first_class_value() {
    let source = r#"
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
entry e(xs: [8]i32) i32 = reduce(i32.(+), map(0i32), xs)
"#;
    compile_to_spirv(source).expect("a user def shadowing a SOAC name should lower as a normal call");
}

/// A user `def map(x: i32) i32 = x` at file scope must not break prelude
/// `unzip`'s `map(|...|, xys)` call. Both `unzip`
/// and the user reference `map` by surface name, but `name_resolution`
/// structurally tags the prelude reference as `Soac(Map)` while the
/// user reference is left bare. Prelude bodies are checked under
/// `LookupContext::Prelude`, which never sees user
/// file-scope; the structural Soac tag routes directly to
/// `globals.builtins["map"]` so the SOAC scheme resolves regardless
/// of what the user did at file scope.
#[test]
fn user_def_shadowing_map_does_not_break_prelude_unzip() {
    let source = r#"
def map(x: i32) i32 = x + 1
entry e(xs: [4](i32, i32)) i32 =
    let (xs0, xs1) = unzip(xs) in
    reduce(i32.(+), 0i32, xs0) + reduce(i32.(+), 0i32, xs1)
"#;
    compile_to_spirv(source)
        .expect("user `def map` must not interfere with prelude unzip's internal `map` call");
}

/// User file scope is visible inside user module bodies, so a
/// `def map(x: i32) [4]i32` shadows the SOAC `map` for a module's
/// `map(xs)` call. This test pins that shadowing with a multi-line
/// transitive `def map` body so an inline-small pass can't collapse
/// the call before TLC observes it.
#[test]
fn user_def_shadowing_map_reaches_into_user_module_body() {
    let source = r#"
def map(xs: [4]i32) i32 = xs[0] + xs[1] + xs[2] + xs[3]
module m = {
  def first_four_sum(xs: [4]i32) i32 = map(xs)
}
entry e(xs: [4]i32) i32 = m.first_four_sum(xs)
"#;
    compile_to_spirv(source).expect(
        "after env-split, a user module body's `map(xs)` resolves to the user `def map` \
         that shadows the SOAC at file scope",
    );
}

/// User module bodies see file-scope shadows of SOAC names at the
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
entry e() [4]f32 =
    let xs = map(|i: i32| f32.i32(i), 0i32 ..< N) in
    [f32.sum(xs), f32.product(xs), f32.minimum(xs), f32.maximum(xs)]
"#;
    compile_to_spirv(source).expect("numeric array-reductions should lower to SPIR-V");
}

/// The statistics-gatherer shape that motivated the reductions: reduce a sample
/// stream to `[count, mean, variance, stddev, min, max]` using `f32.sum`,
/// `f32.minimum`, `f32.maximum`. This is the `Stats` summarize body.
#[test]
fn statistics_gatherer_lowers() {
    let source = r#"
def N: i32 = 256
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

/// `emit_storage_store` interns an output's `view_index` access chain, so both
/// arms of an `if`-then-`else`
/// writing the same output slot share one hashconsed `ViewIndex` node.
/// `elaborated_places` must be scoped per subtree so each arm emits an
/// access chain in a dominating block. The bitwise `&` here matters only because the
/// arithmetic version constant-folds through `partial_eval`; see
/// `branch_with_let_terminal_into_output_slot_lowers` for the
/// fold-resistant parameter-driven repro.
#[test]
fn bitwise_in_deep_let_chain_feeding_if_lowers() {
    let source = r#"
entry t() i32 =
    let s = 2i32 + 3i32 in
    let p = s * 4i32 in
    let x = p & 12i32 in
    if x < 100i32 then x else 0i32
"#;
    compile_to_spirv(source)
        .expect("bitwise result threaded through a deep let-chain into an if should lower");
}

/// The runtime parameter `n` keeps the `if` branch live, so both arms route to
/// the same output slot's `view_index`. Each arm must elaborate that place in
/// a dominating scope.
#[test]
fn branch_with_let_terminal_into_output_slot_lowers() {
    let source = r#"
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


entry fragment_main(pos: vec4f32) vec4f32 =
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


entry fragment_main(pos: vec4f32) vec4f32 =
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

entry compute_main(data: [8]f32) [8]f32 =
    map(|x| selectValue(x, 1), data)
"#;
    compile_to_spirv(source).expect("Map with nested conditionals should compile to SPIR-V");
}

/// Verify multiple maps followed by a reduce compile to SPIR-V.
#[test]
fn test_spirv_multiple_maps_and_reduce() {
    let source = r#"
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

entry compute_main(data: [4]f32) [4]f32 =
    let doubled = map(|x| x * 2.0, data) in
    let flags = [1, 0, 1, 0] in
    build(data, doubled, flags)
"#;
    compile_to_spirv(source).expect("Conditional array construction should compile to SPIR-V");
}

/// Mapping a lambda whose return type is a mixed scalar/vector tuple. The SoA
/// transform rewrites the output
/// `[N](f32, i32, vec3f32)` into a tuple-of-arrays before EGIR
/// conversion; `egir::soac_expand` must split the per-iteration
/// ArrayWith into per-component ArrayWith calls + a Tuple repack.
/// The split supplies the element types needed by SPIR-V runtime indexing.
#[test]
fn test_spirv_map_array_of_mixed_tuple() {
    let source = r#"
def build(xs: [8]f32) [8](f32, i32, vec3f32) =
    map(|x: f32| (x + 1.0, 0, @[x, x, x]), xs)

def fragment_main(fragment: fragment_invocation<vec4f32>) vec4f32 =
    let arr = build([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]) in
    let (a, _, v) = arr[3] in
    @[a, v.x, v.y, v.z]

entry frame(target: render_target<vec4f32>) render_target<vec4f32> =
    let covered = rasterize_triangles(
      direct_draw(3u32, 1u32),
      |vertex| vertex_output(
        if vertex.vertex_index == 0u32 then @[-1.0, -1.0, 0.0, 1.0]
        else if vertex.vertex_index == 1u32 then @[3.0, -1.0, 0.0, 1.0]
        else @[-1.0, 3.0, 0.0, 1.0],
        @[0.0, 0.0, 0.0, 0.0])) in
    shade(target, covered, fragment_main)
"#;
    compile_to_spirv(source).expect("map over [N](f32, i32, vec3f32) should compile to SPIR-V");
}

/// Nested SoA where the element type itself contains a composite array. The
/// transform produces a tuple of
/// arrays whose components are themselves arrays — exercising
/// `emit_write_element`'s recursion through `soa_element_type`.
#[test]
fn test_spirv_map_array_of_nested_tuple() {
    let source = r#"
def build(xs: [4]f32) [4](f32, [3]f32) =
    map(|x: f32| (x + 1.0, [x, x, x]), xs)

def fragment_main(fragment: fragment_invocation<vec4f32>) vec4f32 =
    let arr = build([0.0, 1.0, 2.0, 3.0]) in
    let (a, inner) = arr[2] in
    @[a, inner[0], inner[1], inner[2]]

entry frame(target: render_target<vec4f32>) render_target<vec4f32> =
    let covered = rasterize_triangles(
      direct_draw(3u32, 1u32),
      |vertex| vertex_output(
        if vertex.vertex_index == 0u32 then @[-1.0, -1.0, 0.0, 1.0]
        else if vertex.vertex_index == 1u32 then @[3.0, -1.0, 0.0, 1.0]
        else @[-1.0, 3.0, 0.0, 1.0],
        @[0.0, 0.0, 0.0, 0.0])) in
    shade(target, covered, fragment_main)
"#;
    compile_to_spirv(source).expect("map over [N](f32, [M]f32) should compile to SPIR-V");
}

/// A loop carries an array whose next-iteration value comes from
/// `map(…, iota(N))`. Although `iota` is Virtual, `egir::soac_expand`
/// materializes the map result through `_w_intrinsic_uninit` and
/// `_w_intrinsic_array_with_inplace`, so the carried result must have
/// Composite representation.
/// The `map`, `scan`, and `filter` schemes pin their materialized outputs to
/// Composite to match that runtime representation.
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

entry main(x: []f32) [4]f32 = f(x[0])
"#;
    compile_to_spirv(source).expect(
        "loop carrying `map(..., iota(N))` should compile; currently fails with \
         ArrayWith cache miss because the back-edge array is Virtual variant",
    );
}

/// Indexing a materialized array produced inside a loop is a memory read from
/// a mutable local place.  It must not be treated as an invariant pure
/// expression and hoisted into the loop preheader, before the inner loop has
/// populated the array.
#[test]
fn loop_local_materialized_array_load_stays_after_initialization() {
    use crate::ssa::types::InstKind;

    let ssa = compile_to_ssa(
        r#"
entry nested_array_load(seed: f32) f32 =
  let (_, total) =
    loop (j, total) = (0, 0.0) while j < 2 do
      let (_, values) =
        loop (i, values) = (0, [0.0, 0.0, 0.0]) while i < 3 do
          (i + 1, values with [i] = seed + f32.i32(i))
      in (j + 1, total + values[0])
  in total
"#,
    );

    let entry = ssa
        .entry_points
        .iter()
        .find(|entry| entry.name == "nested_array_load")
        .expect("missing nested_array_load entry point");
    let preheader = entry.body.entry_block();
    let preheader_loads = entry.body.inner.blocks[preheader]
        .insts
        .iter()
        .copied()
        .filter(|inst| matches!(entry.body.inner.insts[*inst].data, InstKind::Load { .. }))
        .collect::<Vec<_>>();

    assert!(
        preheader_loads.is_empty(),
        "loads from the inner loop's materialized array were hoisted before its initialization: \
         {preheader_loads:?}",
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

/// Interprocedural map+reduce fusion must preserve a dead Unreachable block
/// introduced by a preceding if/else without treating it as an unterminated
/// block during reconstruction.
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


entry vertex_main(vid: i32) vec4f32 =
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
    let source = format!(
        "{}\n{}",
        include_str!("../../scripts/playground_image_header.wyn"),
        include_str!("../../testfiles/playground/raytrace.wyn"),
    );

    let ssa = compile_to_ssa_with_modules(&source);

    // SOAC-bearing helpers such as `trace` are intentionally force-inlined
    // before SSA and then removed by DCE. Verify the durable contract instead:
    // both extracted graphical stages survived and SSA construction completed.
    assert!(
        ssa.entry_points.iter().any(|entry| entry.name == "_w_stage_image__vertex"),
        "the extracted raytrace vertex stage should be in SSA output"
    );
    assert!(
        ssa.entry_points.iter().any(|entry| entry.name == "_w_stage_image__fragment"),
        "the extracted raytrace fragment stage should be in SSA output"
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


entry frag(pos: vec4f32) vec4f32 =
  @[QUARTER_TAU, PI, TAU, 1.0]
"#;
    compile_to_spirv(source).expect("constants referencing constants should compile");
}

#[test]
fn test_constant_inlining_global_ref() {
    // Minimal repro: a constant def used by a function, going through inline_small.
    // This should NOT produce an unresolved Global("PI") in SSA.
    let ssa = compile_to_ssa(
        r#"
def PI: f32 = 3.141592

def use_pi(x: f32) f32 = x * PI


entry fragment_main(pos: vec4f32) vec4f32 =
    let r = use_pi(pos.x) in
    @[r, 0.0, 0.0, 1.0]
"#,
    );

    // Dump what we got.
    eprintln!("{}", ssa::print::format_program(&ssa));

    // Check that no Global("PI") instruction exists — it should have been inlined.
    for func in &ssa.functions {
        for (_id, inst) in &func.body.inner.insts {
            if let ssa::types::InstKind::Op {
                tag: op::OpTag::Global(_),
                ..
            } = &inst.data
            {
                panic!("a global reference survived in function '{}'", func.name);
            }
        }
    }
    for ep in &ssa.entry_points {
        for (_id, inst) in &ep.body.inner.insts {
            if let ssa::types::InstKind::Op {
                tag: op::OpTag::Global(_),
                ..
            } = &inst.data
            {
                panic!("a global reference survived in entry '{}'", ep.name);
            }
        }
    }
}

// ============================================================================
// `--fill-holes`: type-hole default fill
// ============================================================================

fn compile_tlc_with_fill_holes(input: &str) -> error::Result<tlc::stage::Transformed> {
    let typed = compile_thru_frontend(input)?;
    let filled = ast_type_holes::fill_type_holes(typed)?;
    tlc::lower_from_ast(filled)
}

#[test]
fn fill_holes_numeric_scalars_compile_clean() {
    // Scalar holes (i32 / f32 / bool) default to 0 / 0.0 / false and
    // compile through with no fill-hole errors.
    for src in ["def x: i32 = ???", "def y: f32 = ???", "def z: bool = ???"] {
        compile_tlc_with_fill_holes(src)
            .unwrap_or_else(|error| panic!("scalar hole in `{src}` should fill cleanly: {error}"));
    }
}

#[test]
fn fill_holes_vec_compiles_clean() {
    compile_tlc_with_fill_holes("def v: vec3f32 = ???")
        .unwrap_or_else(|error| panic!("vec3 hole should fill cleanly: {error}"));
}

#[test]
fn fill_holes_rejects_function_type() {
    let error = compile_tlc_with_fill_holes("def f: i32 -> i32 = ???")
        .expect_err("function-typed hole should surface a fill-hole error");
    let msg = error.to_string();
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
    compile_tlc_with_fill_holes("def arr: [3]i32 = [1i32, ???, 3i32]")
        .unwrap_or_else(|error| panic!("hole in i32 array should fill cleanly: {error}"));
}

// =============================================================================
// Higher-order-function defunctionalization
// =============================================================================

/// A two-argument HOF that calls its function twice must thread the closure's
/// captured environment through every specialized call site.
#[test]
fn hof_closure_with_captures_lowers_to_valid_spirv() {
    let src = r#"
def apply2(f: f32 -> f32, x0: f32, x1: f32) f32 = f(x0) + f(x1)

entry test(a: f32, b: f32) f32 =
  let g = |y: f32| y * y + a + b in
  apply2(g, 1.0f32, 2.0f32)
"#;
    let spirv = compile_to_spirv(src).expect("compile");
    assert_spirv_call_arities_match(&spirv);
}

/// A closure-free inline lambda passed to a HOF inside a map body retains its
/// concrete argument type through defunctionalization and SPIR-V lowering.
#[test]
fn hof_no_capture_lambda_in_map_body_lowers_without_panic() {
    let src = r#"
def apply2(f: f32 -> f32, x0: f32, x1: f32) f32 = f(x0) + f(x1)

entry test(in_arr: []f32) []f32 =
  map(|x: f32| apply2(|y: f32| y * y, x, x + 1.0f32), in_arr)
"#;
    // catch_unwind because the bug surfaces as a panic in
    // spirv/mod.rs (not a Result::Err).
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| compile_to_spirv(src)));
    let bytes = result.expect("compilation panicked").expect("compile returned Err");
    assert_spirv_call_arities_match(&bytes);
}

/// A fixed-size fallback built with `replicate` must stay an array producer,
/// not become a function-typed capture on the lifted outer map. The latter
/// used to leave `_w_lambda_1` higher-order and panic in the post-DCE HOF
/// verifier. The large storage captures mirror proof recovery in Equihash.
#[test]
fn nested_map_with_large_storage_captures_and_replicate_lowers() {
    let src = r#"
def recover_one(root: [2]u32,
                r0: [262144][24]u32, r1: [262144][24]u32,
                r2: [262144][24]u32, r3: [262144][24]u32,
                r4: [262144][24]u32, r5: [262144][24]u32,
                r6: [262144][24]u32, r7: [262144][24]u32) (u32, [512]u32) =
  let leaves = map(|i: i32|
    root[i % 2i32]
    + r0[0][0] + r1[0][0] + r2[0][0] + r3[0][0]
    + r4[0][0] + r5[0][0] + r6[0][0] + r7[0][0],
    iota(512))
  in (1u32, leaves)

entry recover_shape(roots: [1][1024][2]u32,
                    count: [1]u32,
                    r0: [262144][24]u32, r1: [262144][24]u32,
                    r2: [262144][24]u32, r3: [262144][24]u32,
                    r4: [262144][24]u32, r5: [262144][24]u32,
                    r6: [262144][24]u32, r7: [262144][24]u32)
                    ([1024]u32, [1024][512]u32) =
  unzip(map(|i: i32|
    if i < i32.u32(count[0])
    then recover_one(roots[0][i], r0, r1, r2, r3, r4, r5, r6, r7)
    else (0u32, replicate(512, 0u32)),
    iota(1024)))
"#;
    compile_to_spirv(src).expect("proof-recovery map shape should compile without a HOF panic");
}

// =============================================================================
// Sum-type lowering integration tests
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


entry main() vec4f32 =
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


entry main() vec4f32 =
    let a = length_sq(#point(3.0f32, 4.0f32)) in
    let b = length_sq(#origin) in
    @[a, b, 0.0f32, 1.0f32]
"#;
    let spirv = compile_to_spirv(src).expect("multi-payload sum should compile to SPIR-V");
    assert_spirv_call_arities_match(&spirv);
}

// =============================================================================
// Swizzle-with lowering integration tests
// =============================================================================

/// Plain `=` swizzle update: write `e` into v.yz, leaving v.x intact.
#[test]
fn swizzle_with_plain_assign_compiles_to_spirv() {
    let src = r#"
def update(v: vec3f32, e: vec2f32) vec3f32 = v with .yz = e


entry main() vec4f32 =
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


entry main() vec4f32 =
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


entry main() vec4f32 =
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
/// parameter count.
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
// EGIR-side Map parallelization
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
/// serial; the map still lowers to a GID-indexed kernel regardless of whether
/// the streamed slot appears before or after the fixed slot.
fn assert_fixed_output_and_streamed_map_partition(source: &str, output_count: usize) {
    use crate::pipeline_descriptor::{Binding, BufferUsage, DispatchSize, Pipeline};

    let lowered = compile_thru_spirv(source).expect("fixed output and streamed map compile");
    let compute = lowered
        .pipeline
        .pipelines
        .iter()
        .find_map(|pipeline| match pipeline {
            Pipeline::Compute(compute) => Some(compute),
            Pipeline::Graphics(_) => None,
        })
        .expect("one compute pipeline");
    assert_eq!(
        compute.stages.len(),
        2,
        "the singleton output and streamed map require separate execution domains"
    );
    let singleton = compute
        .stages
        .iter()
        .find(|stage| is_singleton_stage(stage))
        .expect("the fixed output has a singleton writer");
    let parallel = compute
        .stages
        .iter()
        .find(|stage| matches!(stage.dispatch_size, DispatchSize::DerivedFrom { .. }))
        .expect("the map retains its streamed dispatch domain");
    assert!(
        !entry_loads_global_invocation_id(&lowered.spirv, &singleton.entry_point),
        "the singleton output writer is invocation-independent"
    );
    assert!(
        entry_loads_global_invocation_id(&lowered.spirv, &parallel.entry_point),
        "the map stage shards with GlobalInvocationID"
    );

    let output_bindings = compute
        .bindings
        .iter()
        .enumerate()
        .filter_map(|(index, binding)| {
            matches!(
                binding,
                Binding::StorageBuffer {
                    usage: BufferUsage::Output,
                    ..
                }
            )
            .then_some(index)
        })
        .collect::<Vec<_>>();
    assert_eq!(output_bindings.len(), output_count);
    for binding in output_bindings {
        assert_eq!(
            compute.stages.iter().filter(|stage| stage.writes.contains(&binding)).count(),
            1,
            "each output binding has exactly one execution-domain owner"
        );
    }
}

#[test]
fn fixed_output_before_streamed_map_still_shards() {
    assert_fixed_output_and_streamed_map_partition(
        "\nentry r(a: []u32) ([2]u32, []u32) = ([7u32, 9u32], map(|x| x + 1u32, a))\n",
        2,
    );
}

/// A fixed slot's direct local alias is still a fixed producer; its surface
/// `Var` form must not prevent sibling maps from supplying the domain.
#[test]
fn let_bound_literal_fixed_output_with_multidomain_maps_shards() {
    let spirv = compile_to_spirv(
        r#"
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
    assert_fixed_output_and_streamed_map_partition(
        r#"
entry r(a: []u32) ([2]u32, []u32) =
  let m = map(|x| x + 1u32, a) in
  ([1u32, 2u32], m)
"#,
        2,
    );
}

/// Canonical resolved slots must also reach the per-domain stage splitter;
/// otherwise admission succeeds but planning sees the original alias syntax.
#[test]
fn fixed_output_with_let_bound_multidomain_maps_shards() {
    let spirv = compile_to_spirv(
        r#"
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
    assert_fixed_output_and_streamed_map_partition(
        r#"
entry r(tidx: []u32, src: []vec4f32, st: []f32) ([2]f32, []vec4f32, []vec4f32) =
  let g = st[0]
  let o0 = [g, g]
  let m1 = map(|s| let i = i32(s) in let it = src[i % 4] in @[it.x, 0.0, it.y, it.z], tidx)
  let m2 = map(|s| let i = i32(s) in let it = src[i % 4] in @[0.0, 1.0, 0.0, it.w], tidx) in
  (o0, m1, m2)
"#,
        3,
    );
}

/// A fixed-size output alongside several *different-domain* maps: the fixed
/// slot becomes its own 1×1×1 constant-write stage while each map keeps its
/// own GID-indexed per-domain dispatch.
#[test]
fn fixed_output_with_multidomain_maps_shards() {
    let spirv = compile_to_spirv(
        "\nentry r(a: []u32, b: []u32) ([2]u32, []u32, []u32) = \
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
        entry
            .body
            .inner
            .blocks
            .values()
            .all(|block| !matches!(block.control_header, Some(ControlHeader::Loop { .. }))),
        "entry {entry_name} should be a loop-free guarded lane kernel"
    );
}

#[test]
fn compute_pointwise_screma_from_horizontal_maps_is_parallel() {
    use crate::builtins::catalog;
    use crate::op::OpTag;
    use crate::ssa::types::{ControlHeader, InstKind};
    let src = r#"
entry pair(xs: []f32) ([]f32, []f32) =
  let a = map(|x: f32| x * x, xs) in
  let b = map(|x: f32| x + 1.0, xs) in
  (a, b)
"#;
    let stats = semantic_soac_stats(&compile_to_semantic_egir(src));
    assert_eq!(stats.seg_maps, 1, "equal-domain sibling maps should co-schedule");
    assert_eq!(
        stats.map_bodies, 1,
        "the sibling bodies are composed into one canonical pre-lambda"
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
        pair.body
            .inner
            .blocks
            .values()
            .all(|block| !matches!(block.control_header, Some(ControlHeader::Loop { .. }))),
        "pointwise Screma entry must be the loop-free guarded lane kernel"
    );

    let spirv = compile_to_spirv(src).expect("pointwise Screma compute compiles");
    assert!(!spirv.is_empty(), "pointwise Screma should lower to SPIR-V");
}

/// Compile `source` through the full *parallelized* pipeline (matching the
/// production driver, which always parallelizes compute) and return the
/// lowered SPIR-V + pipeline descriptor.
fn compile_parallel(source: &str) -> Lowered {
    compile_thru_spirv(source).expect("compile_thru_spirv")
}

/// Full storage-buffer descriptors of a compute pipeline.
fn compute_storage_buffers(
    pipeline: &pipeline_descriptor::PipelineDescriptor,
    entry: &str,
) -> Vec<pipeline_descriptor::Binding> {
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

/// Gathering a computed `map` result at a runtime index requires residency
/// planning to split the producer into its own
/// materialization stage writing a storage buffer, and rewrites the
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
            pipeline_descriptor::Pipeline::Compute(cp) => {
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
            pipeline_descriptor::Pipeline::Compute(compute)
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

    // The producer and consumer share this pipeline's binding table, so its
    // layout access is their union even though stage uses retain direction.
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
// Multi-consumer gather
// ---------------------------------------------------------------------------
//
// EGIR residency planning handles a computed array `counts = map(...)` shared
// by one or more downstream SOAC/gather consumers. The controls below pin both
// single-consumer and shared-resource cases.

/// Control: a single `scan` consumer of a computed `counts` map lifts cleanly.
#[test]
fn single_consumer_scan_compiles() {
    compile_to_spirv(
        "\
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
/// binding namespace to host it (an `Func<Semantic>` does not — see the guard in
/// `from_tlc::convert_function`). This compiles because `evens` is **inlined**
/// into `filt_count` before EGIR conversion, so `convert_soac_filter` runs in
/// the entry's converter and the scratch buffer lands at a non-colliding entry
/// binding.
///
/// IF THIS TEST STARTS FAILING with "runtime `filter` in function `evens`
/// reserved a scratch storage buffer …": the inlining invariant broke — a
/// function whose result is a runtime filter survived to EGIR as a standalone
/// `Func<Semantic>`. The scratch buffer then has no descriptor-set home. To fix,
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

entry filt_count(xs: []i32) i32 =
  let e = evens(xs) in
  length(e)
",
    )
    .expect("runtime filter in an (inlined) subroutine must compile");
}

/// Summing a filtered runtime-sized array — `reduce(+, 0, filter(p, xs))` over
/// an entry-param view. `filter` yields a runtime-length scratch view; `reduce`
/// consumes it like any reduce-over-view.
#[test]
fn filter_into_reduce_compiles() {
    compile_to_spirv(
        "\
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
entry filt_stats(xs: []i32) (i32, i32) =
  let kept = filter(|x: i32| x > 4i32, xs) in
  (length(kept), reduce(|a: i32, b: i32| a + b, 0i32, kept))
";

    let stats = semantic_soac_stats(&compile_to_semantic_egir(source));
    assert_eq!(
        stats.filters, 0,
        "scalar consumers should eliminate filter materialization"
    );
    assert_eq!(
        stats.seg_reds, 1,
        "length and reduce should share a masked SegRed"
    );
    assert_eq!(stats.reduce_operators, 2, "count and sum both remain observable");

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
entry filt_reduce(xs: []i32) i32 =
  let kept = filter(|x: i32| x > 4i32, xs) in
  total(kept)
",
    )
    .expect("let-bound filter result crossing a call boundary into a helper that takes a plain array must compile");
}

/// `reduce(_, _, filter(...))` with the filter result used inline as an
/// argument compiles to the same program as the let-bound form above.
/// `unify_apply_arg` opens the existential at ordinary argument sites unless
/// the expected parameter is itself existential, so existential values can
/// still flow through unchanged when that is the parameter's
/// declared type. Surfaced minimizing the type error in
/// `testfiles/playground/particles3.wyn`'s `align`.
#[test]
fn filter_into_reduce_inline_arg_opens_existential() {
    compile_to_spirv(
        "\
entry filt_reduce(xs: []i32) i32 =
  reduce(|a: i32, b: i32| a + b, 0i32, filter(|x: i32| x > 4i32, xs))
",
    )
    .expect("filter result used inline as `reduce`'s array arg unifies like the let-bound form");
}

/// A multi-letter swizzle on a non-trivial expression must not duplicate its
/// producer. The lowering let-binds a non-atomic projection base before
/// splitting it into per-letter `TupleProj`s.
#[test]
fn swizzle_on_nontrivial_base_does_not_duplicate_producer() {
    use crate::tlc::{Payload, SoacOp, Term, TermKind};
    fn count_reduces<C: Payload, S: Payload>(t: &Term<C, S>) -> usize {
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
    let tlc = compile_thru_tlc(
        "\
def sum2<[n]>(xs: [n]vec4f32) vec2f32 =
  reduce(|a: vec4f32, b: vec4f32| a + b, @[0.0f32, 0.0f32, 0.0f32, 0.0f32], xs).xy
entry e(xs: [8]vec4f32) vec2f32 = sum2(xs)
",
    )
    .expect("compile_thru_tlc");
    let total: usize = tlc.defs.iter().map(|d| count_reduces(&d.body)).sum();
    assert_eq!(
        total, 1,
        "`reduce(...).xy` should compile to one physical reduce, not one per swizzle slot — \
         found {total} `Soac(Reduce)` terms across all defs"
    );
}

/// True iff the pipeline for `entry` is a multi-stage compute (the two-phase
/// shape of a parallelized scalar reduction: chunk + combine). This
/// distinguishes masked fused reduction from a serial schedule
/// filter→reduce.
fn is_two_phase_compute(pipeline: &pipeline_descriptor::PipelineDescriptor, entry: &str) -> bool {
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
    let lowered = compile_thru_spirv(
        "\
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

/// The masked reduction fusion must fire even when `filter` and `reduce` were
/// written in different functions; force-inlining exposes the semantic edge
/// before EGIR optimization.
#[test]
fn filter_into_reduce_fuses_across_functions() {
    let lowered = compile_thru_spirv(
        "\
def evens(xs: []i32) ?k. [k]i32 = filter(|x: i32| x % 2i32 == 0i32, xs)

entry filt_reduce(xs: []i32) i32 =
  let kept = evens(xs) in
  reduce(|a: i32, b: i32| a + b, 0i32, kept)
",
    )
    .expect("cross-function filter→reduce compiles");
    assert!(
        is_two_phase_compute(&lowered.pipeline, "filt_reduce"),
        "cross-function reduce(evens(xs)) must fuse after helper inlining",
    );
}

/// Every compute entry point generated for a program, across all pipelines and
/// their stages (the source entries plus any lifted `_gather_` pre-passes).
/// Lets a test assert how many GPU dispatches one source entry expands to.
fn compute_entry_points(pipeline: &pipeline_descriptor::PipelineDescriptor) -> Vec<String> {
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
/// captures the runtime value `bound`, so the fused EGIR filter body carries a
/// capture. That capture must survive closure conversion's free-variable
/// analysis, ownership/liveness (it is read inside the fused map), and the filter
/// lowering (where it becomes an extra operand of the per-element map call,
/// carried by the map body's explicit capture list. Still one coherent pipeline.
#[test]
fn capturing_map_into_filter_is_one_pipeline() {
    let lowered = compile_parallel(
        "\
open f32
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
entry cmptest(idx: []u32) ?k. [k]vec4f32 =
  filter(|c| c.x < 100.0, map(|s| @[f32(i32(s)), 0.0, 0.0, 0.0], idx))
",
    );
}

/// An entry returning *both* a filtered array and a value derived from its
/// `length`, with the existential over the WHOLE tuple: `?k. ([k]u32, [1]u32)`.
/// The `?k.` packs over the tuple — EGIR route construction must see through
/// the existential wrapper and count the tuple's two outputs, not treat the
/// whole `?k.(…)` as a single output. (Form A.)
#[test]
fn filter_array_and_length_existential_over_tuple_compiles() {
    compile_parallel(
        "\
open f32
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
    let source = "\
entry e(xs: []f32,
        out: *[]f32) []f32 =
  let p = xs[0..512] in
  map(|x: f32|
        let ys = map(|v: f32| v - x, p) in
        let zs = filter(|y: f32| y < 10.0, ys) in
        let ws = map(|z: f32| z * 2.0, zs) in
        reduce(|a: f32, b: f32| a + b, 0.0, ws),
      p)
";
    let stats = semantic_soac_stats(&compile_to_semantic_egir(source));
    assert_eq!(stats.filters, 0, "the filter must fold into the masked SegRed");
    assert_eq!(stats.seg_reds, 1, "the nested chain should contain one SegRed");
    compile_to_spirv(source).expect("map→filter→map→reduce should lower to SPIR-V");
}

/// Cross-function auto-parallelization: a `scan` factored into a helper that
/// `inline_small` will NOT fold (its operator has control flow, so the
/// size/control-flow gate skips it) still parallelizes: force-inlining SOAC
/// helpers exposes it before semantic conversion, and EGIR produces the same
/// multi-phase pipeline as the in-entry form.
#[test]
fn cross_function_scan_parallelizes() {
    let lowered = compile_thru_spirv(
        "\
def stencil(xs: []i32) []i32 = scan(|a: i32, b: i32| if a > b then a else b, 0i32, xs)
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
entry e(i: i32) i32 =
    let m: [4]i32 = [10, 20, 30, 40] in
    m[i]
"#;
    compile_to_spirv(source).expect("runtime index into a statically-sized array should lower");
}

/// Invariant, end to end: a SOAC helper called *per element* inside a `map`
/// lambda must NOT be hoisted and parallelized — the inner reduce stays a
/// serial per-thread loop. The entry parallelizes as one scheduled stage
/// lane-indexed map, not a multi-phase reduce pipeline.
#[test]
fn per_element_helper_soac_stays_serial() {
    let lowered = compile_thru_spirv(
        "\
def rsum(x: i32) i32 = reduce(|a: i32, b: i32| a + b, 0i32, [x, x, x])
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
/// the host reads back. Reification links the output publication and logical
/// resource planning binds its representation.
#[test]
fn filter_result_as_compute_output_compiles() {
    compile_to_spirv(
        "\
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
entry filt_out(xs: []i32) ?k. [k]i32 =
  filter(|x: i32| x % 2i32 == 0i32, xs)
";
    let lowered = compile_thru_spirv(src).expect("filter→output compiles");
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
entry gen(xs: []i32) []i32 =
  let counts = map(|x: i32| x * 2, xs) in
  let total  = reduce(|a: i32, b: i32| a + b, 0, counts) in
  map(|i: i32| total, iota(64))
",
    )
    .expect("single-consumer reduce-over-map must lift + compile");
}

/// When `counts` is consumed by both a `reduce` and a `scan`, EGIR
/// materializes it into one shared buffer that both downstream SOACs read.
#[test]
fn multi_consumer_scan_plus_reduce_lifts() {
    use crate::egir::program::{CompilerResourceKind, ResourceOrigin};

    let source = "\
entry gen(xs: []i32) []i32 =
  let counts  = map(|x: i32| x * 2, xs) in
  let total   = reduce(|a: i32, b: i32| a + b, 0, counts) in
  let offsets = scan(|a: i32, b: i32| a + b, 0, counts) in
  map(|i: i32| offsets[i % 8] + total, iota(64))
";
    let allocated = compile_to_semantic_egir(source);
    let handoff_kinds = allocated
        .data
        .core
        .resources
        .iter()
        .filter_map(|resource| match resource.origin() {
            ResourceOrigin::Compiler {
                resource: compiler, ..
            } => Some(compiler.kind),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        handoff_kinds.iter().filter(|kind| **kind == CompilerResourceKind::MultiConsumerArray).count(),
        1,
        "one shared producer has one canonical handoff"
    );
    assert!(
        !handoff_kinds.contains(&CompilerResourceKind::GatherHandoff),
        "a multi-consumer producer is not a single-consumer gather"
    );

    let lowered = lower_semantic_egir(allocated, LoweringProfile::PORTABLE);
    lower_ssa_to_spirv(lowered)
        .expect("multi-consumer (reduce + scan over the same counts) should lift + compile");
}
/// `counts` consumed by both `scan(counts)` and a direct random gather
/// `counts[i % 8]`. The scan's input is a SOAC edge in the producer graph;
/// the `counts[i % 8]` reference inside the outer map's lambda body is *not*
/// a SOAC edge. With the use-count fix in `producer_graph` counting every
/// `Var(counts)` reference (not just SOAC edges), fusion sees `counts` as
/// multi-use and declines to fuse + drop the let, so EGIR residency planning
/// handles it as a shared producer.
#[test]
fn multi_consumer_scan_plus_gather_lifts() {
    compile_to_spirv(
        "\
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
/// (`offsets[nb - 1]`). This pins that top-level helper references are not
/// mistaken for materialization captures and the scan can be scheduled as a
/// gather-producing materialization.
#[test]
fn fused_scan_helper_call_then_indexed_read_compiles() {
    compile_to_spirv(
        "\
def win_count(hw: f32) i32 =
  let span = 2.0 * hw - 1.0 in
  let fit  = i32.f32(floor(span / 2.4)) in
  if fit < 0 then 0 else if fit > 3 then 3 else fit

def box_count(hw: f32) i32 = 8 + 5 * win_count(hw)

entry gen(bh: []vec4f32, nb: i32) [1]i32 =
  let counts  = map(|h: vec4f32| box_count(h.x), bh) in
  let offsets = scan(|a: i32, b: i32| a + b, 0, counts) in
  [if nb <= 0 then 0 else offsets[nb - 1]]
",
    )
    .expect("fused-scan-of-helper-mapping with indexed scan read should compile");
}

/// A multi-output entry returns a scan as one output and reads it by a
/// constant index for another. Once slot 0 retargets the scan to an
/// `OutputView`, `rewrite_other_index_consumers_to_loads` rewrites the sibling
/// index to `ViewIndex + Load` against that same read-write binding.
#[test]
fn multi_output_returns_scan_and_reads_it_by_index() {
    compile_to_spirv(
        "\
entry gen(xs: []i32) ([]i32, [1]i32) =
  let offsets = scan(|a: i32, b: i32| a + b, 0, xs) in
  (offsets, [offsets[0]])
",
    )
    .expect("multi-output (scan + indexed read of same scan) should compile");
}

/// Dynamic-index variant of the above — slot 1 reads `offsets[k]` where
/// `k` is a uniform, exercising the path where the rewrite passes the
/// dynamic index ValueId straight through `emit_view_load`.
#[test]
fn multi_output_returns_scan_and_reads_it_by_dynamic_index() {
    compile_to_spirv(
        "\
entry gen(xs: []i32, k: i32) ([]i32, [1]i32) =
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
entry gen(xs: []i32) ([]i32, [1]i32) =
  let doubled = map(|x: i32| x * 2, xs) in
  (doubled, [doubled[0]])
",
    )
    .expect("multi-output (map + indexed read of same map) should compile");
}

#[test]
fn multi_output_returns_scan_in_two_slots() {
    compile_thru_spirv(
        "\
entry gen(xs: []i32) ([]i32, []i32) =
  let offsets = scan(|a: i32, b: i32| a + b, 0, xs) in
  (offsets, offsets)
",
    )
    .expect("one scan result can be published through two explicit destinations");
}

/// Under serial scheduling, a vec4-emitting map that gathers from a
/// derived (map/scan-produced) array must still produce well-formed
/// SPIR-V. The materialization resource and consumer output must remain
/// distinct even when target scheduling selects serial recipes.
#[test]
fn serial_vec4_map_gather_from_derived_array_repro() {
    let spirv = compile_to_spirv_serial(
        "\
entry gen(xs: []i32) []vec4f32 =
  let cs = map(|x: i32| x * 2, xs) in
  map(|i: i32| @[f32.i32(cs[i]), 0.0, 0.0, 1.0], iota(8))
",
    )
    .expect("serial vec4-map gathering derived array compiles");
    assert_spirv_storage_access_chain_pointee_types_match(&spirv);
}

/// A pipeline layout's storage access must be exactly the union of its stage
/// uses. Accesses in unrelated pipelines do not promote this layout.
#[test]
fn intermediate_buffer_descriptor_access_repro() {
    use crate::pipeline_descriptor::{Access, Binding, Pipeline};

    let lowered = compile_thru_spirv(
        "\
entry gen(xs: []i32) ([]i32, [1]i32) =
  let counts  = map(|x: i32| x * 2, xs) in
  let offsets = scan(|a: i32, b: i32| a + b, 0, counts) in
  (map(|i: i32| offsets[i % 8], iota(64)),
   [offsets[7]])
",
    )
    .expect("compile to SPIR-V");

    for p in &lowered.pipeline.pipelines {
        let (bindings, stages): (&[Binding], Vec<_>) = match p {
            Pipeline::Compute(cp) => (&cp.bindings, cp.stages.iter().map(|stage| &stage.uses).collect()),
            Pipeline::Graphics(gp) => (&gp.bindings, gp.stages.iter().map(|stage| &stage.uses).collect()),
        };
        for (index, binding) in bindings.iter().enumerate() {
            let Binding::StorageBuffer { access, .. } = binding else {
                continue;
            };
            let reads = stages.iter().any(|stage| stage.reads.contains(&index));
            let writes = stages.iter().any(|stage| stage.writes.contains(&index));
            let expected = match (reads, writes) {
                (true, true) => Access::ReadWrite,
                (true, false) => Access::ReadOnly,
                (false, true) => Access::WriteOnly,
                (false, false) => continue,
            };
            assert_eq!(
                *access, expected,
                "pipeline storage layout disagrees with stage uses"
            );
        }
    }
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
entry g(xs: []i32) []i32 =
  let o = scan(|a:i32,b:i32| a+b, 0, xs) in
  map(|i:i32| o[i % 256], iota(6144))
";
    let lowered = compile_parallel(src);
    assert!(!lowered.spirv.is_empty(), "lowering produced no SPIR-V");

    // The consumer reads the gather buffer as an Intermediate sized
    // LikeInput of `xs` (scan preserves element count and type: i32 → i32).
    // Producer and consumer stages share one compute-pipeline layout, whose
    // access is the union of the stage-local write and read.
    let consumer_bufs = compute_storage_buffers(&lowered.pipeline, "g");
    let gather = consumer_bufs
        .iter()
        .find_map(|b| match b {
            Binding::StorageBuffer {
                set,
                binding,
                usage: BufferUsage::Intermediate,
                access: Access::ReadWrite,
                length: Some(len),
                ..
            } => Some((*set, *binding, len.clone())),
            _ => None,
        })
        .expect("consumer must read a sized gather intermediate");
    assert_eq!(
        gather.2,
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
        .any(|b| matches!(b, Binding::StorageBuffer { set, binding, .. } if (*set, *binding) == (gather.0, gather.1)));
    assert!(
        writes_gather,
        "scan pre-pass must write the gather buffer (binding {})",
        gather.1
    );
    // No other binding in the scan pipeline collides with the gather output.
    let dup = scan_pipeline
        .bindings
        .iter()
        .filter(|b| matches!(b, Binding::StorageBuffer { set, binding, .. } if (*set, *binding) == (gather.0, gather.1)))
        .count();
    assert_eq!(
        dup, 1,
        "exactly one scan binding is the gather buffer: {:?}",
        scan_pipeline.bindings
    );

    // Every entry point sharing this physical pipeline must declare the
    // pipeline-layout access, not its narrower stage-local use. In particular,
    // the scan result reader must not select a second `NonWritable` variable
    // for a slot whose descriptor layout is read_write.
    let gather_index = scan_pipeline
        .bindings
        .iter()
        .position(|binding| {
            matches!(binding, Binding::StorageBuffer { set, binding, .. } if (*set, *binding) == (gather.0, gather.1))
        })
        .expect("scan gather binding index");
    let users = scan_pipeline
        .stages
        .iter()
        .filter(|stage| stage.reads.contains(&gather_index) || stage.writes.contains(&gather_index))
        .collect::<Vec<_>>();
    assert!(
        users
            .iter()
            .any(|stage| stage.reads.contains(&gather_index) && !stage.writes.contains(&gather_index)),
        "the regression requires a read-only stage use"
    );
    for stage in users {
        assert_eq!(
            spirv_entry_storage_binding_is_writable(&lowered.spirv, &stage.entry_point, gather.0, gather.1),
            Some(true),
            "entry `{}` must use the pipeline's read_write variable",
            stage.entry_point
        );
    }

    let converted = lower_semantic_egir(
        compile_to_semantic_egir(src),
        LoweringProfile::new(CodegenTarget::Wgsl, SchedulePolicy::Parallel),
    );
    let wgsl_slot = converted
        .global_context
        .pipeline
        .pipelines
        .iter()
        .find_map(|pipeline| match pipeline {
            Pipeline::Compute(compute) => {
                compute.bindings.iter().enumerate().find_map(|(index, binding)| {
                    let Binding::StorageBuffer {
                        set,
                        binding,
                        access: Access::ReadWrite,
                        ..
                    } = binding
                    else {
                        return None;
                    };
                    compute
                        .stages
                        .iter()
                        .any(|stage| stage.reads.contains(&index) && !stage.writes.contains(&index))
                        .then_some((*set, *binding))
                })
            }
            Pipeline::Graphics(_) => None,
        })
        .expect("WGSL scan pipeline has a read-only use of a read_write layout slot");
    let wgsl = lower_ssa_to_wgsl(converted).expect("scan gather lowers to WGSL");
    assert!(
        wgsl.contains(&format!(
            "@group({}) @binding({}) var<storage, read_write>",
            wgsl_slot.0, wgsl_slot.1
        )),
        "WGSL must declare the shared scan slot with pipeline-union access"
    );
    let module = naga::front::wgsl::parse_str(&wgsl)
        .unwrap_or_else(|error| panic!("Naga rejected generated WGSL: {error:?}\n{wgsl}"));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|error| panic!("Naga validation failed: {error:?}\n{wgsl}"));
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

entry go(xa: []u32, xb: []u32) ([]f32, []f32) =
  slice_ab(xa, xb)

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
/// indexed inside that map's body, keeps a concrete `Buffer(set, binding)` on
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

entry go(dom: []u32, pdom: []u32, idom: []u32, pts_in: []f32, its_in: []f32)
  ([]f32, []f32, []f32) =
  let w  = { pts = pts_in, its = its_in }
  let w2 = update(w, pdom, idom)
  let sc = { pts = w.pts, its = w.its } in
  (w2.pts, w2.its,
   map(|s| sc.pts[0] + sc.its[0] + f32(s), dom))
";
    compile_thru_spirv(src).expect("record of views indexed inside a map must compile");
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
            pipeline_descriptor::Pipeline::Compute(cp) => Some(cp),
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

entry go(dom: []u32, pts_in: []f32) ([]f32, []f32) =
  let sc = { pts = pts_in } in
  (map(|s| pts_in[i32(s)] * 2.0, dom),
   map(|s| sdf(sc, f32(s)), dom))
";
    compile_thru_spirv(src).expect("record of views must survive map fusion into one stage");
}

/// A fragment entry's storage-buffer parameters carry a concrete
/// `Buffer(set, binding)` in their type, exactly as a compute entry's do. A
/// helper that indexes them through a record is the shape that exposes it:
/// the helper's buffer variable is generalized and instantiated per call, so
/// the entry parameter is the only place a concrete region can be pinned.
///
/// Without that pin, `scene_sdf` reaches SPIR-V lowering holding
/// `Array[vec2f32, View, ?size, ?region]` and there is no buffer to build an
/// `OpAccessChain` into.
#[test]
fn fragment_storage_buffer_params_pin_a_buffer_region() {
    let src = "\
open f32

type scene = {
  points: []vec2f32,
  items: []vec4f32,
}

def scene_sdf(sc: scene, x: f32) f32 =
  loop acc = x for i < 4 do
    let p = sc.points[i % 2]
    let it = sc.items[i % 2] in
    acc + p.x + p.y + it.x


entry resolve_like(fc: vec4f32,
                   points: []vec2f32,
                   items: []vec4f32)
  vec4f32 =
  let sc = { points = points, items = items }
  let v = scene_sdf(sc, fc.x) in
  @[v, v, v, 1.0]
";
    compile_thru_spirv(src).expect("fragment storage-buffer reads must pin a buffer region");
}

/// A unique `*storage_image` handle threaded through both arms of an `if` is
/// consumed once, not once per arm. Each arm yields the updated handle — one
/// arm writes, the other passes it through — so the alias checker must treat
/// the branch as a single use of `small`.
#[test]
fn gather_same_array_coalesces_to_one_buffer() {
    use crate::pipeline_descriptor::{Binding, BufferUsage, Pipeline};
    let src = "\
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
                entry pick_const() i32 =
            let m: [3][2]i32 = [[1, 2], [3, 4], [5, 6]] in
            m[1][0]

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
                        tag: op::OpTag::DynamicExtract,
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
    let _ = compile_thru_spirv(src).expect("compile_thru_spirv should succeed");
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
                entry row_sums(buf: []([4]u32)) []u32 =
            map(|row: [4]u32| row[0] + row[1] + row[2] + row[3], buf)
    "#;
    let lowered = compile_thru_spirv(src).expect("compile_thru_spirv");
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
        pipeline_descriptor::Binding::StorageBuffer { name, length, .. } if name == "row_sums_output" => {
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
/// TLC-to-EGIR conversion records each branch's `SlotSource` at its block;
/// both routes retain the same output slot. Runtime CFG ensures only one fires per execution
/// path.
#[test]
fn compute_if_over_two_maps_compiles_runtime_sized() {
    use crate::pipeline_descriptor::{BufferLen, Pipeline};
    let src = r#"
                entry tick(prev: []vec2f32,
                   iTime: f32) []vec2f32 =
          if iTime == 0.0
            then map(|p: vec2f32| @[1.0f32, 1.0f32], prev)
            else map(|p: vec2f32| @[p.x + 1.0f32, p.y + 1.0f32], prev)
    "#;
    let lowered = compile_thru_spirv(src).expect("compile_thru_spirv");
    let Pipeline::Compute(cp) = lowered.pipeline.pipelines.first().expect("one pipeline") else {
        panic!("expected single-compute pipeline");
    };
    let output_slot = cp.bindings.iter().find_map(|binding| match binding {
        pipeline_descriptor::Binding::StorageBuffer {
            set, binding, name, ..
        } if name == "tick_output" => Some((*set, *binding)),
        _ => None,
    });
    let source_result = lowered.pipeline.source_results.as_slice();
    assert_eq!(source_result.len(), 1);
    assert_eq!(source_result[0].entry, "tick");
    assert_eq!(source_result[0].result, 0);
    assert_eq!(source_result[0].pipeline_index, 0);
    assert_eq!(
        Some((source_result[0].set, source_result[0].binding)),
        output_slot
    );
    // Output's size variable matches `prev`'s — the length-inference
    // rule emits `LikeInput` rather than `SameAsDispatch`.
    let output_len = cp.bindings.iter().find_map(|b| match b {
        pipeline_descriptor::Binding::StorageBuffer { name, length, .. } if name == "tick_output" => {
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
            assert_eq!(set, 0);
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
                entry tick(prev: []vec2f32,
                   iTime: f32) []vec2f32 =
          if iTime == 0.0
            then map(|p: vec2f32| @[1.0f32, 1.0f32], prev)
            else map(|p: vec2f32| @[p.x + 1.0f32, p.y + 1.0f32], prev)
    "#;

    // Inspect at the pre-defunctionalize stage: `if_over_producer` normalizes
    // here, before defunctionalization lifts the Map operator to a ref.
    let fused = test_pipeline::compile_thru_expose_producers(src);
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
                entry tick(prev_pos: []vec4f32,
                   iTime: f32) []vec4f32 =
          if iTime < 0.1 then
            map(|i:i32| @[f32.i32(i), 0.0, 0.0, 0.0], 0i32..<N)
          else
            let prev_pos = prev_pos[0..N] in
            map(
              |upd:vec4f32| @[upd.x + 1.0, upd.y, upd.z, upd.w],
              map(|elem:vec4f32| @[elem.x, elem.y, elem.z, elem.w], prev_pos))
    "#;

    // Pre-defunctionalize: see `if_over_producer`'s normalized Map before
    // defunctionalization obscures it.
    let fused = test_pipeline::compile_thru_expose_producers(src);
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
                entry pick(xs: []f32, ys: []f32, flag: bool) []f32 =
          if flag
            then map(|x: f32| x + 1.0, xs)
            else map(|y: f32| y * 2.0, ys)
    "#;

    // Pre-defunctionalize: maps over distinct domains must stay a branching
    // `If` here (`if_over_producer` only merges branches over one domain).
    let fused = test_pipeline::compile_thru_expose_producers(src);
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
                entry tick(prev: []vec2f32,
                   iTime: f32) []vec2f32 =
          if iTime < 0.0
            then map(|p: vec2f32| @[0.0f32, 0.0f32], prev)
            else if iTime == 0.0
              then map(|p: vec2f32| @[1.0f32, 1.0f32], prev)
              else map(|p: vec2f32| @[p.x + 1.0f32, p.y + 1.0f32], prev)
    "#;
    compile_thru_spirv(src).expect("nested If over three maps must compile");
}

/// `Let`-wrapped `If` whose body's branches read the let-bound value.
/// `convert_slot_store` recognises the `Let` and binds the RHS at the
/// current block before recursing into the body — so the binding
/// survives the branch fork.
#[test]
fn compute_let_wrapped_if_over_two_maps_compiles_runtime_sized() {
    let src = r#"
                entry tick<[n]>(prev: [n]vec2f32,
                        iTime: f32) [n]vec2f32 =
          let nudge: f32 = iTime * 0.1f32 in
          if iTime == 0.0
            then map(|p: vec2f32| @[nudge, nudge], prev)
            else map(|p: vec2f32| @[p.x + nudge, p.y + nudge], prev)
    "#;
    compile_thru_spirv(src).expect("Let-wrapped If over two maps must compile");
}

/// Fixed-size output: both branches map
/// over different sources (`0..<N` vs `prev_pos`) but the output is
/// `[Size(2)]vec2f32` because `N = 2` is a literal. Lands in the
/// fixed-aggregate path of `dispatch::compute_slot_source`.
#[test]
fn compute_if_over_two_maps_compiles_fixed_size_different_sources() {
    let src = r#"
        def N: i32 = 2
                entry tick(prev_pos: [2]vec2f32,
                   iTime: f32) [2]vec2f32 =
          if iTime == 0.0 then
            map(|i:i32| if i == 0 then @[2.0, 2.0] else @[15.0, 5.0], 0i32..<N)
          else
            map(|pos:vec2f32| @[pos.x + 1.0, pos.y + 1.0], prev_pos)
    "#;
    compile_thru_spirv(src).expect("fixed-size If-over-maps (different sources) must compile");
}

/// The user's original case 2 (fixed-size output): both branches map
/// over the *same* range source. Output is still `[Size(2)]vec2f32`.
/// Same code path as case 1 — the source-difference doesn't matter for
/// fixed-size aggregates.
#[test]
fn compute_if_over_two_maps_compiles_fixed_size_same_source() {
    let src = r#"
        def N: i32 = 2
                entry tick(prev_pos: [2]vec2f32,
                   iTime: f32) [2]vec2f32 =
          if iTime == 0.0 then
            map(|i:i32| if i == 0 then @[2.0, 2.0] else @[15.0, 5.0], 0i32..<N)
          else
            map(|i:i32| @[f32.i32(i), f32.i32(i)], 0i32..<N)
    "#;
    compile_thru_spirv(src).expect("fixed-size If-over-maps (same source) must compile");
}

/// Multi-output entry whose Tuple components each contain an `If`.
/// TLC-to-EGIR conversion decomposes the tuple into per-slot routes and follows
/// each `If` fork. Both slots end up multi-source, each
/// retargeting into its own `OutputView`.
#[test]
fn compute_multi_output_tuple_of_ifs_compiles() {
    let src = r#"
                entry tick<[n]>(prev_pos: [n]vec2f32,
                        iTime: f32) ([n]vec2f32, [n]f32) =
          (if iTime == 0.0
             then map(|p: vec2f32| @[0.0f32, 0.0f32], prev_pos)
             else map(|p: vec2f32| @[p.x + 1.0f32, p.y + 1.0f32], prev_pos),
           if iTime == 0.0
             then map(|p: vec2f32| 0.0f32, prev_pos)
             else map(|p: vec2f32| p.x * p.x + p.y * p.y, prev_pos))
    "#;
    compile_thru_spirv(src).expect("multi-output tuple of Ifs must compile");
}

/// Assert the StorageBuffer variable decorated `(set, binding)` is the base
/// of at least one `OpAccessChain` — i.e. actually read/written, not merely
/// declared. A lifted lambda must retain the input view's descriptor
/// provenance rather than reading from an output descriptor.
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

    let target_vars = sets
        .iter()
        .filter(|(id, s)| **s == set && binds.get(id) == Some(&binding))
        .map(|(id, _)| *id)
        .collect::<std::collections::HashSet<_>>();
    assert!(
        !target_vars.is_empty(),
        "no StorageBuffer variable decorated (set={set}, binding={binding})"
    );

    let accessed = module.functions.iter().any(|f| {
        f.blocks.iter().any(|b| {
            b.instructions.iter().any(|inst| {
                inst.class.opcode == Op::AccessChain
                    && matches!(inst.operands.first(), Some(Operand::IdRef(base)) if target_vars.contains(base))
            })
        })
    });

    assert!(
        accessed,
        "descriptor (set={set}, binding={binding}) is declared but never reached by an \
         OpAccessChain — view-array provenance was lost (reads went to the wrong buffer)"
    );
}

// View-array slice provenance through a SOAC capture: the lifted lambda's
// `xs[0..3]` reads must come from `xs`'s compiler-assigned descriptor, not
// the compiler-allocated output buffer.
#[test]
fn slice_view_inside_map_lambda_compiles_to_spirv() {
    let src = r#"
        def gather3(arr: [3]f32) f32 = arr[0] + arr[1] + arr[2]

                entry tick(xs: []f32) []f32 =
          map(|_:i32| gather3(xs[0..3]), 0i32..<3)
    "#;
    let lowered = compile_thru_spirv(src)
        .expect("view-array slice inside a map lambda must preserve buffer provenance");
    assert_storage_descriptor_is_accessed(&lowered.spirv, 0, 0);
}

// ---- Buffer-provenance guards ------------------------------------------------
//
// These assert the exact descriptor each view read resolves to. A wrong-buffer
// read can still pass spirv-val when that descriptor is declared, so each guard
// checks via rspirv that the expected `(set, binding)` is the base of an
// `OpAccessChain`.

/// Indexing a *captured* view inside a `map` lambda (→ lifted lambda, the
/// `rewrite_specialized_body` Index arm) must read from the captured buffer.
#[test]
fn view_index_in_map_lambda_reads_own_buffer() {
    let src = r#"
                entry tick(xs: []f32) []f32 =
          map(|i: i32| xs[i] + xs[0], 0i32..<4)
    "#;
    let lowered = compile_thru_spirv(src).expect("captured-view index compiles");
    assert_storage_descriptor_is_accessed(&lowered.spirv, 0, 0);
}

/// Two captured views at distinct `(set, binding)` must each be read from
/// their own descriptor — catches a unification that swaps or collapses
/// buffer provenance.
#[test]
fn two_view_captures_read_distinct_buffers() {
    let src = r#"
                entry tick(
          xs: []f32,
          ys: []f32
        ) []f32 =
          map(|i: i32| xs[i] + ys[0], 0i32..<4)
    "#;
    let lowered = compile_thru_spirv(src).expect("two captured views compile");
    assert_storage_descriptor_is_accessed(&lowered.spirv, 0, 0);
    assert_storage_descriptor_is_accessed(&lowered.spirv, 0, 1);
}

/// A captured view passed to a user function that itself indexes it (→
/// recursive per-buffer specialization) must read from the captured buffer.
#[test]
fn view_through_nested_fn_specialization_reads_own_buffer() {
    let src = r#"
        def firstx(zs: []f32) f32 = zs[0]

                entry tick(xs: []f32) []f32 =
          map(|_: i32| firstx(xs), 0i32..<4)
    "#;
    let lowered = compile_thru_spirv(src).expect("nested view specialization compiles");
    assert_storage_descriptor_is_accessed(&lowered.spirv, 0, 0);
}

/// A fixed storage array must remain addressable across lifted-map capture
/// boundaries and the final ordinary call into `leaf`.
#[test]
fn fixed_view_through_two_named_helpers_emits_valid_wgsl() {
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            let source = r#"
def leaf(table: [64][64]u32, index: i32) u32 = table[0][index]
def expand(seed: [2]u32, table: [64][64]u32) [4]u32 =
  map(|i: i32| seed[i / 2i32] + leaf(table, i), iota(4))
entry nested_view(
    roots: [4][2]u32,
    table: [64][64]u32
) [4][4]u32 =
  map(|i: i32| expand(roots[i], table), iota(4))
"#;

            let ssa = lower_semantic_egir(
                compile_to_semantic_egir(source),
                LoweringProfile::new(CodegenTarget::Wgsl, SchedulePolicy::Parallel),
            );
            let leaf =
                ssa.functions.iter().find(|function| function.name.contains("leaf")).unwrap_or_else(|| {
                    panic!(
                        "leaf reaches elaborated SSA before definition reachability; functions: {:?}",
                        ssa.functions.iter().map(|function| &function.name).collect::<Vec<_>>()
                    )
                });
            let table = leaf.body.parameter_place(0).expect("leaf table parameter remains addressable");
            assert!(
                matches!(
                    leaf.body.place_elem_ty(table),
                    polytype::Type::Constructed(ast::TypeName::Array, _)
                ),
                "leaf table parameter must address array storage"
            );

            let reachable = ssa::filter_reachable(ssa.clone());
            assert!(
                reachable.functions.iter().all(|function| !function.name.contains("leaf")),
                "the fully inlined leaf helper must be removed before backend lowering"
            );

            let wgsl = lower_ssa_to_wgsl(ssa).expect("nested fixed storage view lowers to WGSL");
            assert!(
                !wgsl.lines().any(|line| line.starts_with("fn w_leaf")),
                "WGSL must not emit the fully inlined leaf helper:\n{wgsl}"
            );
            let module = naga::front::wgsl::parse_str(&wgsl)
                .unwrap_or_else(|error| panic!("Naga rejected nested-view WGSL: {error:?}\n{wgsl}"));
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap_or_else(|error| panic!("Naga validation rejected nested-view WGSL: {error:?}\n{wgsl}"));
        })
        .expect("spawn nested-view WGSL regression")
        .join()
        .expect("nested-view WGSL regression panicked");
}

/// A directly nested index into a ranked storage view must remain an address
/// chain through the selected leaf. Loading after the first coordinate copies
/// the complete row into function-local storage before selecting one scalar.
#[test]
fn ranked_storage_index_loads_only_the_selected_leaf() {
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            let source = r#"
entry main(roots: [1][1024]u32) [1]u32 =
  [roots[0][7i32]]
"#;

            let ssa = lower_semantic_egir(
                compile_to_semantic_egir(source),
                LoweringProfile::new(CodegenTarget::Wgsl, SchedulePolicy::Parallel),
            );
            let entry = ssa.entry_points.iter().find(|entry| entry.name == "main").expect("main entry");
            let mut view_places = Vec::new();
            let mut place_indices = Vec::new();
            let mut load_places = Vec::new();
            let mut array_loads = 0;
            for inst in entry.body.inner.insts.values() {
                match &inst.data {
                    ssa::types::InstKind::ViewIndex { result, .. } => view_places.push(*result),
                    ssa::types::InstKind::PlaceIndex { place, result, .. } => {
                        place_indices.push((*result, *place));
                    }
                    ssa::types::InstKind::Load { place } => {
                        load_places.push(*place);
                        let result = inst.result.expect("Load has a result");
                        if matches!(
                            entry.body.get_value_type(result),
                            polytype::Type::Constructed(ast::TypeName::Array, _)
                        ) {
                            array_loads += 1;
                        }
                    }
                    _ => {}
                }
            }
            assert_eq!(load_places.len(), 1, "ranked read performs one final leaf Load");
            assert_eq!(array_loads, 0, "ranked read must not load an intermediate row");
            let (_, parent_place) = place_indices
                .iter()
                .find(|(result, _)| *result == load_places[0])
                .expect("the leaf Load reads through a PlaceIndex");
            assert!(
                view_places.contains(parent_place),
                "the leaf PlaceIndex must extend the input ViewIndex"
            );

            let wgsl = lower_ssa_to_wgsl(ssa).expect("ranked storage index lowers to WGSL");
            assert!(
                !wgsl.lines().any(|line| line.contains("var ") && line.contains(": array<u32, 1024>")),
                "WGSL must not materialize the complete row in a local variable:\n{wgsl}"
            );
            assert!(
                wgsl.lines().any(|line| line.contains(": u32 = _buf_0_0[") && line.contains("][")),
                "WGSL scalar load must index both storage coordinates directly:\n{wgsl}"
            );

            let module = naga::front::wgsl::parse_str(&wgsl)
                .unwrap_or_else(|error| panic!("Naga rejected ranked-index WGSL: {error:?}\n{wgsl}"));
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap_or_else(|error| {
                panic!("Naga validation rejected ranked-index WGSL: {error:?}\n{wgsl}")
            });
        })
        .expect("spawn ranked-storage-index regression")
        .join()
        .expect("ranked-storage-index regression panicked");
}

/// A mapped helper's ranked storage access must remain an address chain rooted
/// at its place parameter; otherwise the dynamic leaf coordinate materializes
/// the complete row.
#[test]
fn mapped_helper_ranked_storage_index_loads_only_the_selected_leaf() {
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            let source = r#"
def leaf(rows: [1][1024]u32, index: i32) u32 =
  rows[0][index]

entry main(roots: [1][1024]u32) [4]u32 =
  map(|index: i32| leaf(roots, index), iota(4))
"#;

            let ssa = lower_semantic_egir(
                compile_to_semantic_egir(source),
                LoweringProfile::new(CodegenTarget::Wgsl, SchedulePolicy::Parallel),
            );
            let helper = ssa
                .functions
                .iter()
                .find(|function| {
                    function
                        .body
                        .inner
                        .insts
                        .values()
                        .any(|inst| matches!(inst.data, ssa::types::InstKind::PlaceIndex { .. }))
                })
                .expect("mapped helper contains the recovered ranked place chain");
            let mut view_places = Vec::new();
            let mut place_indices = Vec::new();
            let mut load_places = Vec::new();
            let mut array_loads = 0;
            for inst in helper.body.inner.insts.values() {
                match &inst.data {
                    ssa::types::InstKind::ViewIndex { result, .. } => view_places.push(*result),
                    ssa::types::InstKind::PlaceIndex { place, result, .. } => {
                        place_indices.push((*result, *place));
                    }
                    ssa::types::InstKind::Load { place } => {
                        load_places.push(*place);
                        let result = inst.result.expect("Load has a result");
                        if matches!(
                            helper.body.get_value_type(result),
                            polytype::Type::Constructed(ast::TypeName::Array, _)
                        ) {
                            array_loads += 1;
                        }
                    }
                    _ => {}
                }
            }
            assert_eq!(load_places.len(), 1, "mapped helper performs one final leaf Load");
            assert_eq!(array_loads, 0, "mapped helper must not load an intermediate row");
            let mut root = load_places[0];
            while let Some((_, parent)) = place_indices.iter().find(|(result, _)| *result == root) {
                root = *parent;
            }
            assert!(
                (0..helper.body.params().len())
                    .any(|index| helper.body.parameter_place(index) == Some(root))
                    || view_places.contains(&root),
                "mapped helper's indexed address chain must have an addressable boundary root"
            );

            let wgsl = lower_ssa_to_wgsl(ssa).expect("mapped ranked storage index lowers to WGSL");
            assert!(
                !wgsl.lines().any(|line| line.contains("var ") && line.contains(": array<u32, 1024>")),
                "mapped-helper WGSL must not materialize the complete row:\n{wgsl}"
            );
            assert!(
                wgsl.lines().any(|line| line.contains(": u32 = _buf_0_0[") && line.contains("][")),
                "mapped-helper WGSL scalar load must index both storage coordinates directly:\n{wgsl}"
            );
            let module = naga::front::wgsl::parse_str(&wgsl)
                .unwrap_or_else(|error| panic!("Naga rejected mapped-helper WGSL: {error:?}\n{wgsl}"));
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap_or_else(|error| {
                panic!("Naga validation rejected mapped-helper WGSL: {error:?}\n{wgsl}")
            });

            compile_thru_spirv(source)
                .expect("mapped helper's storage-rooted PlaceIndex chain lowers to valid SPIR-V");
        })
        .expect("spawn mapped-helper ranked-index regression")
        .join()
        .expect("mapped-helper ranked-index regression panicked");
}

/// A map lambda that captures a small fixed array used to receive the complete
/// array by value and then materialize another local copy for its dynamic
/// index. The physical inliner must remove that per-lane call boundary.
#[test]
fn mapped_helper_inlines_fixed_array_capture_before_dynamic_extract() {
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            let source = r#"
def expand_2(values: [2]u32) [4]u32 =
  map(|i: i32| values[i / 2i32], iota(4))

entry main(root: [2]u32) [4]u32 =
  expand_2(root)
"#;

            let ssa = lower_semantic_egir(
                compile_to_semantic_egir(source),
                LoweringProfile::new(CodegenTarget::Wgsl, SchedulePolicy::Parallel),
            );
            let entry = ssa.entry_points.iter().find(|entry| entry.name == "main").expect("main entry");
            assert!(
                entry.body.inner.insts.values().all(|inst| !matches!(
                    inst.data,
                    ssa::types::InstKind::Op {
                        tag: op::OpTag::Call(_),
                        ..
                    }
                )),
                "fixed-array capture must be propagated into the mapped entry instead of passed by value"
            );
            assert_eq!(
                entry
                    .body
                    .inner
                    .insts
                    .values()
                    .filter(|inst| matches!(
                        inst.data,
                        ssa::types::InstKind::Op {
                            tag: op::OpTag::Materialize,
                            ..
                        }
                    ))
                    .count(),
                0,
                "the entry parameter place should make dynamic extraction copy-free"
            );

            let wgsl = lower_ssa_to_wgsl(ssa).expect("fixed-array capture lowers to WGSL");
            let main = wgsl.split("fn main(").nth(1).expect("WGSL main body");
            assert!(
                !main.contains("w_Uw_Ulambda_U0("),
                "WGSL main must not pass the fixed array through a per-lane helper:\n{wgsl}"
            );
        })
        .expect("spawn fixed-array capture regression")
        .join()
        .expect("fixed-array capture regression panicked");
}

/// A storage-view call is conservatively effect-classified before partial
/// inlining. It must not prevent propagation of a fixed-array capture through
/// the surrounding map helper.
#[test]
fn mapped_helper_inlines_fixed_array_capture_alongside_storage_view() {
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            let source = r#"
def lookup(table: [1024][2]u32, reference: u32, child: i32) u32 =
  table[i32.u32(reference)][child]

def expand_2(root: [2]u32, table: [1024][2]u32) [4]u32 =
  map(|i: i32| lookup(table, root[i / 2i32], i % 2i32), iota(4))

entry main(root: [2]u32, table: [1024][2]u32) [4]u32 =
  expand_2(root, table)
"#;

            let ssa = lower_semantic_egir(
                compile_to_semantic_egir(source),
                LoweringProfile::new(CodegenTarget::Wgsl, SchedulePolicy::Parallel),
            );
            let entry = ssa.entry_points.iter().find(|entry| entry.name == "main").expect("main entry");
            assert!(
                entry.body.inner.insts.values().all(|inst| !matches!(
                    inst.data,
                    ssa::types::InstKind::Op {
                        tag: op::OpTag::Call(_),
                        ..
                    }
                )),
                "the mapped entry must not pass the fixed array through a helper call"
            );
            assert_eq!(
                entry
                    .body
                    .inner
                    .insts
                    .values()
                    .filter(|inst| matches!(
                        inst.data,
                        ssa::types::InstKind::Op {
                            tag: op::OpTag::Materialize,
                            ..
                        }
                    ))
                    .count(),
                0,
                "the root parameter place should make dynamic indexing copy-free"
            );

            let wgsl = lower_ssa_to_wgsl(ssa).expect("mixed capture lowers to WGSL");
            let main = wgsl.split("fn main(").nth(1).expect("WGSL main body");
            assert!(
                !main.contains("w_Uw_Ulambda_U0("),
                "WGSL main must not pass root by value through the map helper:\n{wgsl}"
            );
            assert!(
                main.lines().any(|line| line.contains("_buf_0_0[") && line.contains("][")),
                "WGSL main must preserve direct ranked storage indexing after inlining:\n{wgsl}"
            );
        })
        .expect("spawn mixed fixed-array/storage-view capture regression")
        .join()
        .expect("mixed fixed-array/storage-view capture regression panicked");
}

/// A selection inside the element helper used to block fixed-array capture
/// propagation. The mapped owner then called a helper that copied `root` by
/// value in both branches. Structured inlining must splice the selection into
/// the owner loop and let placement keep its one addressable root copy in the
/// loop preheader. Exercise both the minimal reproducer and an Equihash-sized
/// fixed array so profitability does not accidentally scale with type width.
#[test]
fn mapped_helper_inlines_fixed_array_capture_through_selection_cfg() {
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            for width in [2, 256] {
                let source = format!(
                    r#"
def expand(root: [{width}]u32, table: [1024][2]u32) [{width}]u32 =
  map(|i: i32|
    let reference = root[i]
    in if i == 0i32
       then table[i32.u32(reference)][0]
       else table[i32.u32(reference)][1],
    iota({width}))

entry main(roots: [1]([{width}]u32), table: [1024][2]u32) [1]([{width}]u32) =
  map(|i: i32| expand(roots[i], table), iota(1))
"#
                );

                let ssa = lower_semantic_egir(
                    compile_to_semantic_egir(&source),
                    LoweringProfile::new(
                        CodegenTarget::Wgsl,
                        SchedulePolicy::Parallel,
                    ),
                );
                let owner = ssa
                    .functions
                    .iter()
                    .find(|function| function.name.contains("lambda_1"))
                    .unwrap_or_else(|| {
                        panic!(
                            "mapped owner missing for width {width}; functions: {:?}",
                            ssa.functions.iter().map(|function| &function.name).collect::<Vec<_>>()
                        )
                    });
                assert!(
                    owner.body.inner.insts.values().all(|inst| !matches!(
                        inst.data,
                        ssa::types::InstKind::Op {
                            tag: op::OpTag::Call(_),
                            ..
                        }
                    )),
                    "width-{width} owner must not pass root through a per-element helper"
                );
                let materializes = owner
                    .body
                    .inner
                    .insts
                    .iter()
                    .filter_map(|(id, inst)| {
                        matches!(
                            inst.data,
                            ssa::types::InstKind::Op {
                                tag: op::OpTag::Materialize,
                                ..
                            }
                        )
                        .then_some(id)
                    })
                    .collect::<Vec<_>>();
                assert!(
                    materializes.is_empty(),
                    "width-{width} owner must preserve addressable captures without aggregate materialization"
                );

                let wgsl = lower_ssa_to_wgsl(ssa)
                    .unwrap_or_else(|error| panic!("width-{width} capture lowers to WGSL: {error}"));
                assert!(
                    !wgsl.contains("fn w_Uw_Ulambda_U0("),
                    "width-{width} WGSL must not emit the inlined per-element helper:\n{wgsl}"
                );
                let module = naga::front::wgsl::parse_str(&wgsl).unwrap_or_else(|error| {
                    panic!("Naga rejected width-{width} capture WGSL: {error:?}\n{wgsl}")
                });
                naga::valid::Validator::new(
                    naga::valid::ValidationFlags::all(),
                    naga::valid::Capabilities::all(),
                )
                .validate(&module)
                .unwrap_or_else(|error| {
                    panic!("Naga validation rejected width-{width} capture WGSL: {error:?}\n{wgsl}")
                });
            }
        })
        .expect("spawn structured fixed-array capture regression")
        .join()
        .expect("structured fixed-array capture regression panicked");
}

/// `unzip(map(...))` becomes two projected map consumers. Fusion must fold
/// projections through the inlined producer tuples instead of forwarding the
/// complete aggregate through nested wrapper tuples.
#[test]
fn unzip_map_fixed_array_item_folds_aggregate_projection_carriers() {
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            let source = r#"
entry main(input: [1]u32) ([1]u32, [1]([2]u32)) =
  unzip(map(|i: i32| (input[i], [input[i], input[i] + 1u32]), iota(1)))
"#;

            let ssa = lower_semantic_egir(
                compile_to_semantic_egir(source),
                LoweringProfile::new(CodegenTarget::Wgsl, SchedulePolicy::Parallel),
            );
            let helper = ssa
                .functions
                .iter()
                .find(|function| function.name.contains("main_vertical_pre_1"))
                .expect("second unzip projection fusion helper");
            let projects = helper
                .body
                .inner
                .insts
                .values()
                .filter(|inst| {
                    matches!(
                        inst.data,
                        ssa::types::InstKind::Op {
                            tag: op::OpTag::Project { .. },
                            ..
                        }
                    )
                })
                .count();
            let tuples = helper
                .body
                .inner
                .insts
                .values()
                .filter(|inst| {
                    matches!(
                        inst.data,
                        ssa::types::InstKind::Op {
                            tag: op::OpTag::Tuple(_),
                            ..
                        }
                    )
                })
                .count();
            assert_eq!(
                projects, 0,
                "fused unzip helper must not project through aggregate carriers"
            );
            assert_eq!(
                tuples, 0,
                "fused unzip helper writes both destination fields directly"
            );

            let entry = ssa.entry_points.iter().find(|entry| entry.name == "main").expect("main entry");
            assert_eq!(
                entry
                    .body
                    .inner
                    .insts
                    .values()
                    .filter(|inst| matches!(
                        inst.data,
                        ssa::types::InstKind::Op {
                            tag: op::OpTag::Tuple(_),
                            ..
                        }
                    ))
                    .count(),
                0,
                "entry stores projected unzip components directly without repacking the helper result"
            );

            let wgsl = lower_ssa_to_wgsl(ssa).expect("simplified unzip/map aggregate lowers to WGSL");
            let main = wgsl.split("fn main(").nth(1).expect("WGSL main body");
            assert!(
                !main.lines().any(|line| line.contains(": T") && line.contains(" = T")),
                "WGSL entry must not construct the final aggregate carrier:\n{wgsl}"
            );
        })
        .expect("spawn unzip/map aggregate-projection regression")
        .join()
        .expect("unzip/map aggregate-projection regression panicked");
}

/// Physical inlining can consume every call to synthesized map/projection
/// helpers. Final SSA reachability must then remove the orphan definitions so
/// WGSL does not validate or emit bodies that no entry can call.
#[test]
fn ssa_reachability_removes_fully_inlined_unzip_map_helpers() {
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            let source = r#"
entry main(input: [1]u32) ([1]u32, [1]u32) =
  unzip(map(|i: i32| (input[i], input[i] + 1u32), iota(1)))
"#;

            let ssa = lower_semantic_egir(
                compile_to_semantic_egir(source),
                LoweringProfile::new(CodegenTarget::Wgsl, SchedulePolicy::Parallel),
            );
            assert!(
                !ssa.functions.is_empty(),
                "reproducer must reach SSA with helpers orphaned by physical inlining"
            );

            let reachable = ssa::filter_reachable(ssa.clone());
            assert!(
                reachable.functions.is_empty(),
                "the fully inlined entry must not retain any callable definitions"
            );

            let wgsl = lower_ssa_to_wgsl(ssa).expect("reachable unzip/map SSA lowers to WGSL");
            assert_eq!(
                wgsl.lines().filter(|line| line.starts_with("fn ")).count(),
                1,
                "WGSL must contain only the live entry function:\n{wgsl}"
            );
            assert!(
                !wgsl.contains("vertical_Upre") && !wgsl.contains("lambda"),
                "WGSL must omit orphan map, projection, and vertical-pre helpers:\n{wgsl}"
            );
        })
        .expect("spawn SSA reachability regression")
        .join()
        .expect("SSA reachability regression panicked");
}

/// A view used directly as a `map` *input* (→ the entry walker
/// `rewrite_term` / SOAC-input path, not a capture) must read from its buffer.
#[test]
fn view_as_map_input_reads_own_buffer() {
    let src = r#"
                entry tick(xs: []f32) []f32 =
          map(|x: f32| x * 2.0, xs)
    "#;
    let lowered = compile_thru_spirv(src).expect("view-as-map-input compiles");
    assert_storage_descriptor_is_accessed(&lowered.spirv, 0, 0);
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

    let target_vars = sets
        .iter()
        .filter(|(id, s)| **s == set && binds.get(id) == Some(&binding))
        .map(|(id, _)| *id)
        .collect::<std::collections::HashSet<_>>();
    assert!(
        !target_vars.is_empty(),
        "no StorageBuffer variable decorated (set={set}, binding={binding})"
    );

    let queried = module.functions.iter().any(|f| {
        f.blocks.iter().any(|b| {
            b.instructions.iter().any(|inst| {
                inst.class.opcode == Op::ArrayLength
                    && matches!(inst.operands.first(), Some(Operand::IdRef(s)) if target_vars.contains(s))
            })
        })
    });

    assert!(
        queried,
        "descriptor (set={set}, binding={binding}) is declared but its length is never \
         queried by an OpArrayLength — length(view) provenance was lost"
    );
}

/// An `OpArrayLength` already produces `u32`; consuming it as `u32` must not
/// insert an identity `OpBitcast %uint`. A cast to a genuinely different type
/// remains legal.
fn assert_array_lengths_have_no_identity_bitcasts(spirv_words: &[u32]) {
    use std::collections::{HashMap, HashSet};
    use wspirv::binary::parse_words;
    use wspirv::dr::{Loader, Operand};
    use wspirv::spirv::Op;

    let mut loader = Loader::new();
    parse_words(spirv_words, &mut loader).expect("parse spirv");
    let module = loader.module();
    let instructions = module
        .functions
        .iter()
        .flat_map(|function| function.blocks.iter().flat_map(|block| block.instructions.iter()));

    let mut result_types = HashMap::new();
    let mut array_lengths = HashSet::new();
    let instructions: Vec<_> = instructions.collect();
    for inst in &instructions {
        if let (Some(result_id), Some(result_type)) = (inst.result_id, inst.result_type) {
            result_types.insert(result_id, result_type);
            if inst.class.opcode == Op::ArrayLength {
                array_lengths.insert(result_id);
            }
        }
    }

    for inst in instructions {
        if inst.class.opcode != Op::Bitcast {
            continue;
        }
        let Some(Operand::IdRef(source)) = inst.operands.first() else {
            continue;
        };
        if array_lengths.contains(source) && result_types.get(source) == inst.result_type.as_ref() {
            panic!("OpArrayLength result %{source} is followed by a redundant identity OpBitcast");
        }
    }
}

/// `length(view)` on an entry param must query *its* descriptor. The binding is
/// baked into `_w_storage_len(set, binding)` as constants (not the side-map), so
/// this also pins that the right `(set, binding)` reaches the `OpArrayLength`.
#[test]
fn entry_length_queries_own_buffer() {
    let src = r#"
                entry tick(xs: []f32) []f32 =
          map(|i: i32| xs[i] * f32.i32(length(xs)), 0i32..<4)
    "#;
    let lowered = compile_thru_spirv(src).expect("entry length(view) compiles");
    assert_array_length_queried_on_descriptor(&lowered.spirv, 0, 0);
    assert_array_lengths_have_no_identity_bitcasts(&lowered.spirv);
    // and the indexed read still hits the same descriptor
    assert_storage_descriptor_is_accessed(&lowered.spirv, 0, 0);
}

/// A `scan` over a view threads that view through loop-carried block params
/// (scan-DPS phase 1). This is exactly the path `propagate_view_provenance`
/// keeps correct today; the guard pins that the loop-carried read still resolves
/// to the input descriptor, so the Tier-2 deletion of that propagation (binding
/// in the type) can't silently re-route it.
#[test]
fn scan_over_view_reads_own_buffer() {
    let src = r#"
                entry tick(xs: []f32) []f32 =
          scan(|a: f32, b: f32| a + b, 0.0, xs)
    "#;
    let lowered = compile_thru_spirv(src).expect("scan over a view compiles");
    assert_storage_descriptor_is_accessed(&lowered.spirv, 0, 0);
}

/// Merging two views at *distinct* descriptors — `if c then xs else ys` — has
/// no single static binding, so it must not compile. Type inference unifies the
/// two branches' buffer variables into one; `pin_entry_buffers` then tries to
/// pin that one variable to both `Buffer(2,0)` and `Buffer(2,1)`, detects the
/// conflict, and rejects it — rather than silently reading the wrong buffer.
#[test]
fn merge_of_distinct_buffers_is_a_type_error() {
    let src = r#"
                entry tick(
          xs: []f32,
          ys: []f32,
          c: u32
        ) []f32 =
          map(|i: i32| (if c > 0u32 then xs else ys)[i], 0i32..<4)
    "#;
    let err =
        compile_thru_spirv(src).err().expect("merging xs and ys (distinct descriptors) must not compile");
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
                entry tick(xs: []f32,
                   n: u32) []i32 =
          map(|x: f32| i32(x), xs)
    "#;
    compile_thru_spirv(src).expect("i32(f32) constructor must compile to SPIR-V");
}

#[test]
fn ctor_scalar_constructor_matches_legacy_dot_form() {
    let new = r#"
                entry tick(xs: []f32,
                   n: u32) []i32 =
          map(|x: f32| i32(x), xs)
    "#;
    let legacy = r#"
                entry tick(xs: []f32,
                   n: u32) []i32 =
          map(|x: f32| i32.f32(x), xs)
    "#;
    // Constructor and dot-form compatibility syntax lower identically.
    compile_thru_spirv(new).expect("new T(value) form must compile");
    compile_thru_spirv(legacy).expect("legacy T.source(value) form must still compile");
}

#[test]
fn ctor_vec2_constructor_compiles_to_spirv() {
    let src = r#"
                entry tick(xs: []vec2f32,
                   n: u32) []vec2i32 =
          map(|v: vec2f32| vec2i32(v), xs)
    "#;
    compile_thru_spirv(src).expect("vec2i32(vec2f32) must compile to SPIR-V");
}

#[test]
fn ctor_vec3_and_vec4_constructors_compile_to_spirv() {
    let v3 = r#"
                entry tick(xs: []vec3i32,
                   n: u32) []vec3f32 =
          map(|v: vec3i32| vec3f32(v), xs)
    "#;
    let v4 = r#"
                entry tick(xs: []vec4u32,
                   n: u32) []vec4f32 =
          map(|v: vec4u32| vec4f32(v), xs)
    "#;
    compile_thru_spirv(v3).expect("vec3f32(vec3i32) must compile");
    compile_thru_spirv(v4).expect("vec4f32(vec4u32) must compile");
}

// ---- ArrayVariantAbstract — `filter` → size-polymorphic consumer ----
//
// `filter` returns `?k. Array[a, Abstract, k, no_buffer]`. The producer's EGIR
// lowering picks Bounded for
// static-capacity inputs and View for runtime-sized ones; the consumer
// can be a size-polymorphic helper that gets specialized against the
// `Abstract` representation in TLC and resolved at the producer edge in
// EGIR. The backend-boundary verifier (`egir::verify_no_abstract`)
// rejects any residual `Array[_, Abstract, _, _]`.
//
// These pin the canonical patterns; `filter_into_reduce_*` covers fusion shape
// and runtime length.

#[test]
fn filter_into_user_size_poly_helper_compiles() {
    let src = r#"
def sum<[n]>(xs: [n]f32) f32 = reduce(|a: f32, b: f32| a + b, 0.0, xs)

entry tick(xs: []f32) f32 =
  let kept = filter(|x: f32| x > 0.0, xs) in
  sum(kept)
"#;
    compile_thru_spirv(src)
        .expect("`filter` piped through a user-defined size-poly helper must compile to SPIR-V");
}

#[test]
fn filter_into_user_size_poly_helper_static_capacity() {
    // Static-capacity input exercises the Bounded producer path
    // (filter result is `{buffer: [8]f32, len: u32}`).
    let src = r#"
def sum<[n]>(xs: [n]f32) f32 = reduce(|a: f32, b: f32| a + b, 0.0, xs)

entry tick() f32 =
  let kept = filter(|x: f32| x > 0.0, [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0]) in
  sum(kept)
"#;
    compile_thru_spirv(src).expect("static-capacity filter piped through a size-poly helper must compile");
}

/// A `filter -> map -> reduce` chain feeds a vector operation and swizzle
/// inside a helper called from a compute map. `convert_soac_map` inherits the
/// existential input shape, and normalization flattens nested lets so the
/// fusion driver sees the complete chain.
///      reduce joins the top-level chain and `map->reduce` /
///      `filter->reduce` collapse it to a masked fused reduce.
#[test]
fn filter_map_reduce_vecop_swizzle_in_helper_compiles() {
    let src = r#"
def f(arr: []vec4f32) vec2f32 =
  let selected = filter(|d| d.x < 1.0, arr) in
  let contributions = map(|d| d * 2.0, selected) in
  (reduce(|a, b| a + b, @[0.0, 0.0, 0.0, 0.0], contributions) * 0.1).xy

entry e(arr0: []vec4f32) []vec2f32 =
  let arr = arr0[0..512] in
  map(|p: vec4f32| f(arr), arr)
"#;
    compile_thru_spirv(src)
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
entry e(a: []f32,
        o: *[]f32) () =
  let m = map(|x: f32| x * 2.0, filter(|x: f32| x > 0.0, a[0..256])) in
  let _ = scatter(o, [0i32], [m[0]]) in ()
"#;
    compile_thru_spirv(src).expect("Filter->Map should compile");
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
entry e(o: *[]i32) () =
  let s = reduce(|a: i32, b: i32| a + b, 0i32, 0i32 ..< 256) in
  let _ = scatter(o, [0i32], [s]) in ()
"#;
    let ssa = compile_thru_ssa(src).expect("Range->Reduce should compile");
    let mir = ssa::print::format_program(&ssa);
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
entry e(o: *[]i32) () =
  let s = scan(|a: i32, b: i32| a + b, 0i32, 0i32 ..< 256) in
  let _ = scatter(o, [0i32], [s[0]]) in ()
"#;
    compile_thru_spirv(src).expect("Range->Scan should compile");
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
entry e(a: []f32,
        o: *[]f32) () =
  let s = scan(|x: f32, y: f32| x + y, 0.0, filter(|x: f32| x > 0.0, a[0..256])) in
  let _ = scatter(o, [0i32], [s[0]]) in ()
"#;
    compile_thru_spirv(src).expect("Filter->Scan should compile");
}

/// Scan -> Map is represented as one canonical Screma whose post-map consumes
/// the inclusive scan value. A flat post region runs after
/// the parallel scan has applied global block offsets.
#[test]
fn scan_into_map_compiles() {
    let cases = [
        (
            "scan-map",
            r#"
entry e(a: []f32) []f32 =
  map(|x: f32| x + 1.0, scan(|x: f32, y: f32| x + y, 0.0, a))
"#,
            true,
            1,
            1,
        ),
        (
            "map-scan-map",
            r#"
entry e(a: []f32) []f32 =
  map(
    |x: f32| x + 1.0,
    scan(|x: f32, y: f32| x + y, 0.0, map(|x: f32| x * 2.0, a)))
"#,
            true,
            2,
            1,
        ),
        (
            "type-changing-scan-map",
            r#"
entry e(a: []f32) []vec2f32 =
  map(|x: f32| @[x, x + 1.0], scan(|x: f32, y: f32| x + y, 0.0, a))
"#,
            true,
            1,
            1,
        ),
        (
            "sliced-scan-map",
            r#"
entry e(a: []f32) []f32 =
  map(|x: f32| x + 1.0, scan(|x: f32, y: f32| x + y, 0.0, a[0..256]))
"#,
            true,
            1,
            1,
        ),
        (
            "multi-output-scan-map",
            r#"
entry e(a: []f32) ([]f32, []f32) =
  let prefixes = scan(|x: f32, y: f32| x + y, 0.0, a) in
  (map(|x: f32| x + 1.0, prefixes), map(|x: f32| x * 2.0, prefixes))
"#,
            true,
            1,
            2,
        ),
        (
            "nested-sliced-scan-map",
            r#"
entry e(a: []f32) []f32 =
  map(|x: f32| x + 1.0, scan(|x: f32, y: f32| x + y, 0.0, a[0..512][0..256]))
"#,
            false,
            1,
            1,
        ),
    ];

    for (label, src, parallel, expected_lambdas, expected_outputs) in cases {
        let allocated = compile_to_semantic_egir(src);
        let stats = semantic_soac_stats(&allocated);
        assert_eq!(stats.seg_scans, 1, "{label}: scan and post-map share one Screma");
        assert_eq!(stats.seg_maps, 0, "{label}: no map may remain materialized");
        assert_eq!(
            stats.map_bodies, expected_lambdas,
            "{label}: canonical pre/post lambda count"
        );
        let has_post_scan = allocated_entries(&allocated)
            .flat_map(|entry| entry.graph.skeleton.blocks.iter())
            .flat_map(|(_, block)| &block.side_effects)
            .any(|effect| {
                matches!(
                    &effect.kind,
                    egir::types::SideEffectKind::Soac(egir::types::SoacEffect(
                        _,
                        egir::types::Soac::Screma(op)
                    )) if !op.form.scans.is_empty() && op.form.reductions.is_empty() && !op.form.post.is_identity() && op.form.post.result_types.len() == expected_outputs
                )
            });
        assert!(
            has_post_scan,
            "{label}: scan result must route through the post-map"
        );
        let planned = egir::plan(compile_to_semantic_egir(src), LoweringProfile::PORTABLE)
            .unwrap_or_else(|error| panic!("{label}: parallel plan: {error}"));
        let expected_phases: &[&str] = if parallel {
            &["scan_phase1", "scan_block", "scan_apply_offsets"]
        } else {
            &["serial_compute"]
        };
        assert_eq!(
            planned.physical_kernels().phases().map(|phase| phase.label.as_str()).collect::<Vec<_>>(),
            expected_phases,
            "{label}: recipe selection"
        );
        assert_eq!(
            planned
                .logical_resources()
                .iter()
                .filter(|resource| matches!(
                    resource.origin(),
                    egir::program::ResourceOrigin::Compiler { resource: compiler, .. }
                        if compiler.kind == egir::program::CompilerResourceKind::ScanPrefixes
                ))
                .count(),
            usize::from(parallel),
            "{label}: prefix handoff ownership"
        );
        compile_thru_spirv(src).unwrap_or_else(|error| panic!("{label}: SPIR-V: {error}"));
        lower_ssa_to_wgsl(lower_semantic_egir(
            compile_to_semantic_egir(src),
            LoweringProfile::new(CodegenTarget::Wgsl, SchedulePolicy::Parallel),
        ))
        .unwrap_or_else(|error| panic!("{label}: WGSL: {error}"));
    }
}

#[test]
fn independent_scans_use_one_parallel_product_recipe() {
    let source = r#"
entry paired_prefixes(xs: []i32) ([]i32, []i32) =
  (scan(|a: i32, b: i32| a + b, 0, xs),
   scan(|a: i32, b: i32| if a > b then a else b, 0, xs))
"#;

    let stats = semantic_soac_stats(&compile_to_semantic_egir(source));
    assert_eq!(stats.seg_scans, 1, "independent scans share one Screma");
    assert_eq!(stats.scan_operators, 2);
    let planned = egir::plan(compile_to_semantic_egir(source), LoweringProfile::PORTABLE)
        .expect("plan independent scans");
    assert_eq!(
        planned.physical_kernels().phases().map(|phase| phase.label.as_str()).collect::<Vec<_>>(),
        ["scan_phase1", "scan_block", "scan_apply_offsets"]
    );
    compile_thru_spirv(source).expect("parallel product scan compiles to SPIR-V");
}
#[test]
fn scan_fuses_with_independent_consumer_collective() {
    let source = r#"
entry e(xs: []i32) ([]i32, [1]i32) =
  let prefixes = scan(|a: i32, b: i32| a + b, 0, xs) in
  let mapped = map(|x: i32| x * 2, prefixes) in
  let total = reduce(|a: i32, b: i32| a + b, 0, xs) in
  (mapped, [total])
"#;

    let stats = semantic_soac_stats(&compile_to_semantic_egir(source));
    assert_eq!(
        stats.mixed_scremas, 1,
        "independent scan and reduction share one Screma"
    );
    assert_eq!(stats.scan_operators, 1);
    assert_eq!(stats.reduce_operators, 1);
    assert_eq!(stats.seg_maps + stats.seg_scans + stats.seg_reds, 0);
    let planned = egir::plan(compile_to_semantic_egir(source), LoweringProfile::PORTABLE)
        .expect("plan mixed scan/reduction");
    assert_eq!(
        planned.physical_kernels().phases().map(|phase| phase.label.as_str()).collect::<Vec<_>>(),
        ["scan_phase1", "scan_block", "scan_apply_offsets"]
    );
    compile_to_spirv(source).expect("middle-barrier-normalized Screma lowers to SPIR-V");
}

#[test]
fn multiple_scans_and_reductions_share_one_parallel_product_recipe() {
    let source = r#"
entry collective_product(xs: []i32, modes: []i32) ([2]i32, []i32, []i32) =
  let total = reduce(
    |a: i32, b: i32| if modes[0] > 0 then a + b else a * b,
    1,
    xs) in
  let maximum = reduce(|a: i32, b: i32| if a > b then a else b, -2147483648, xs) in
  let totals = scan(|a: i32, b: i32| a + b, 0, xs) in
  let maxima = scan(|a: i32, b: i32| if a > b then a else b, -2147483648, xs) in
  ([total, maximum], totals, maxima)
"#;

    let stats = semantic_soac_stats(&compile_to_semantic_egir(source));
    assert_eq!(stats.mixed_scremas, 1);
    assert_eq!(stats.scan_operators, 2);
    assert_eq!(stats.reduce_operators, 2);
    let planned = egir::plan(compile_to_semantic_egir(source), LoweringProfile::PORTABLE)
        .expect("plan collective product");
    assert_eq!(
        planned.physical_kernels().phases().map(|phase| phase.label.as_str()).collect::<Vec<_>>(),
        ["scan_phase1", "scan_block", "scan_apply_offsets"]
    );
    compile_thru_spirv(source).expect("collective product compiles to SPIR-V");
}
#[test]
fn dependent_scan_into_reduce_keeps_two_collective_barriers() {
    let source = r#"
entry e(xs: [8]i32) [1]i32 =
  let prefixes = scan(|a: i32, b: i32| a + b, 0, xs) in
  let total = reduce(|a: i32, b: i32| a + b, 0, prefixes) in
  [total]
"#;

    let stats = semantic_soac_stats(&compile_to_semantic_egir(source));
    assert_eq!(stats.seg_scans, 1, "the producer scan barrier remains");
    assert_eq!(stats.seg_reds, 1, "the dependent reduction barrier remains");
    assert_eq!(stats.mixed_scremas, 0);
    compile_to_spirv(source).expect("unfused dependent barriers lower to SPIR-V");
}
/// An entry returning a scan and a fixed-size literal that indexes it must
/// derive both output routes before allocating the scan materialization
/// resource.
#[test]
fn entry_tuple_output_with_scan_indexed_literal_keeps_both_bindings() {
    use crate::pipeline_descriptor::{BufferUsage, Pipeline};
    let lowered = compile_thru_spirv(
        "\
entry gen(xs: []i32, n: i32) ([]vec4f32, [5]i32) =
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
            pipeline_descriptor::Binding::StorageBuffer {
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
/// inside the lambda each compile fine. Surfaced by a PCG hash
/// (`pcg` has `let w = .. in (w >> 22) ^ w`, used for the hoisted key and
/// inside the per-element map).
#[test]
fn bitwise_fn_inlined_both_captured_and_in_soac_lambda_lowers() {
    compile_to_spirv(
        "\
def f(v: u32) u32 = let w = v ^ 1u32 in (w >> 1u32) ^ w
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
/// mis-threads its capture (`ArityMismatch`). This covers env-bound lambdas
/// during dissolved-let residualization.
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

entry v() vec4f32 = @[f32.i32(nested_lambda(100)), 0.0, 0.0, 1.0]
",
    );
}

/// partial_eval folds integer arithmetic; a u32 multiply like `C * K`
/// overflows u32 (and its i128-free product would overflow i64), so the fold
/// must wrap mod 2^32 rather than emit an out-of-range literal
/// ("Invalid u32"). Surfaced by a PCG hash.
#[test]
fn folded_u32_arithmetic_wraps_to_width() {
    compile_to_spirv(
        "\
def C: u32 = 2654435769u32
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
/// shared instead. Surfaced by a Threefry `block`.
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
entry e() []u32 = map(|i: i32| g.at((0x9e3779b9u32, 0x243f6a88u32), u32.i32(i)), 0i32 ..< 4)
",
    )
    .expect("deep tuple-destructure let-chain must lower without blowup or dangling vars");
}

// =========================================================================
// Unsupported compiler shapes — minimal reproducers
//
// Each test here pins a Wyn-source shape that currently panics during
// compilation. Tests use `#[should_panic]` so the suite stays green while
// the fixture stays committed; drop `#[should_panic]` when the panic is
// supported.
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
    compile_thru_frontend("def f(x: i32) i32 = i32(x)")
        .expect("i32(x) where x: i32 should resolve as the identity conversion");
}

/// Two compute entries in one module: both entries' outputs land on
/// distinct `(set 0, binding N)` slots even when their input shapes
/// are identical. Single shared `IdSource<u32>` across all
/// compiler-allocated set-0 bindings guarantees no two `OpVariable`s
/// share a slot.
#[test]
fn two_compute_entries_do_not_collide_on_auto_bindings() {
    let lowered = compile_thru_spirv(
        r#"
entry a(xs: []u32) []vec4f32 = map(|x| @[f32.u32(x), 0.0, 0.0, 0.0], xs)
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
/// indexing into base '%uint'`. `pin_entry_buffers` therefore owns a single
/// `IdSource<u32>` across all entries.
#[test]
fn two_compute_entries_with_differently_typed_inputs_do_not_alias() {
    let lowered = compile_thru_spirv(
        r#"
entry a(xs: []u32) []u32 = map(|x| x, xs)
entry b(ys: []f32) []f32 = map(|y| y, ys)
"#,
    )
    .expect("heterogeneously-typed compute-entry inputs must compile to one valid SPIR-V module");
    assert!(!lowered.spirv.is_empty());
}

/// Two entries binding the *same* explicit ``
/// slot to buffers with different element types (`[]f32` vs `[]vec4f32`)
/// must be rejected at compile time. The compiler coalesces same-slot
/// storage into one module-global whose type is the first declaration's;
/// the other entry then indexes it as the wrong element type, producing
/// `spirv-val: OpAccessChain result type ... does not match indexing into
/// base ...`. Reaching SPIR-V at all is the bug — the type checker must
/// reject the conflicting interface first.
#[test]
fn matching_explicit_storage_binding_across_entries_compiles() {
    let lowered = compile_thru_spirv(
        r#"
entry ent_a(idx: []u32, buf: []f32) []f32 =
  map(|s| buf[i32.u32(s)], idx)
entry ent_b(idx: []u32, buf: []f32) []f32 =
  map(|s| buf[i32.u32(s)] * 2.0, idx)
"#,
    )
    .expect("entries that agree on a shared (set, binding) element type must compile");
    assert!(!lowered.spirv.is_empty());
}

/// A raw compute `storage_image` and fragment `texture2d` cannot occupy the
/// same descriptor slot. Use a named `resource` instead, which gives the
/// sampled view a distinct texture descriptor with a backing reference.
#[test]
fn image_store_is_not_user_visible() {
    let result = compile_thru_spirv(
        r#"
entry r(xs: []u32,
        img: storage_image) []u32 =
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
#[ignore = "array-of-tuples entry input does not lower (element type Tuple(2) has no static size); blocks the SoA same-size-class case"]
fn soa_array_of_tuples_components_stay_one_size_class() {
    let lowered = compile_thru_spirv(
        r#"
entry main(pts: [](f32, f32)) ([]f32, []f32) =
  (map(|p| p.0 + 1.0, pts), map(|p| p.1 + 2.0, pts))
"#,
    )
    .expect("array-of-tuples entry with two component maps must compile to one valid module");
    assert!(!lowered.spirv.is_empty());
}

/// A global definition whose initializer calls a function may reference other
/// globals. The SPIR-V backend forward-declares and lowers
/// `program.constants` as zero-argument functions, matching
/// `program.functions`.
#[test]
fn function_call_initialized_global_compiles() {
    compile_thru_spirv(
        r#"
def DIST: f32 = 5.0
def ELEV: f32 = 0.7
def rotm(a: f32) mat3f32 =
  @[[f32.cos(a), 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, f32.cos(a)]]
def eye: vec3f32 = rotm(ELEV) * @[0.0, 0.0, DIST]
def use_eye(p: vec3f32) vec3f32 = p + eye

entry f() vec4f32 =
  let q = use_eye(@[1.0, 2.0, 3.0]) in @[q.x, q.y, q.z, 1.0]
"#,
    )
    .expect("global whose initializer calls a function should lower like any other global");
}

/// `filter` allocates a scratch storage binding (`filt_gather_b<n>`)
/// that the same compute stage writes into via the SOAC expansion.
/// `egir::from_tlc::convert_soac_filter` declares that scratch with
/// `role: Output` so `publish.rs` reports it as write-capable rather than a
/// read-only intermediate.
#[test]
fn filter_scratch_binding_is_not_read_only() {
    use crate::pipeline_descriptor::{Access, Binding, BufferUsage};
    let lowered = compile_parallel(
        r#"
def keep(x: u32) bool = x != 0u32
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
/// buffer-layout diagnostic ("runtime-sized array … wrap the
/// producer in a `map`") for this source shape. If we ever teach
/// the compiler to compile `*[]T with [i] = v` returns directly,
/// flip the assertion to expect clean success.
#[test]
fn inplace_write_to_returned_readwrite_storage_errors_gracefully() {
    let result = compile_thru_spirv(
        r#"
entry tick(buf: *[]u32) *[]u32 =
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
/// default `Composite[Variable, NoBuffer]`. Otherwise the SPIR-V
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
    compile_thru_spirv(
        r#"
type draw_args = {x: f32, y: f32, z: f32, w: f32}

entry frag(iTime: f32) draw_args =
  {x = iTime, y = 0.0, z = 0.0, w = 1.0}
"#,
    )
    .expect("fragment returning a structural record should lower");

    compile_thru_spirv(
        r#"
type point = {x: f32, y: f32}
entry e(o: *[]point) () =
  let _ = scatter(o, [0i32], [{x = 1.0, y = 2.0}]) in ()
"#,
    )
    .expect("compute scatter into *[]record should lower");
}

/// A record-of-runtime-arrays alias (`world`) passed as a function PARAM.
/// The array fields' variant/buffer slots must be buffer-polymorphic across
/// the call boundary (the alias body's placeholders freshen per use), so
/// `world` unifies with a `{ points = view, items = view }` argument.
#[test]
fn record_of_arrays_param_across_boundary_compiles() {
    compile_thru_spirv(
        r#"
open f32
type world = { points: []vec2f32, items: []vec4f32 }

def use_world(w: world, dom: []u32) ([]vec2f32, []vec4f32) =
  let p = map(|i| let j = i32(i) in w.points[j] + @[1.0, 1.0], dom)
  let it = map(|i| let j = i32(i) in w.items[j] * @[2.0, 2.0, 2.0, 2.0], dom) in
  (p, it)

entry step(dom: []u32, points_in: []vec2f32, items_in: []vec4f32)
  ([]vec2f32, []vec4f32) =
  use_world({ points = points_in, items = items_in }, dom)
"#,
    )
    .expect("record-of-arrays as a function param should compile");
}

#[test]
fn physical_planning_finalizes_internal_and_extern_callable_abis() {
    use egir::types::{OperandType, PlaceAccess, ResultDestination, ValueKind};

    let planned = egir::plan(
        compile_to_semantic_egir(
            r#"
def countdown(n: i32) i32 =
  if n <= 0 then 0 else countdown(n - 1)
def fixed_loop(xs: [4]i32, n: i32) i32 =
  if n <= 0 then xs[0] else fixed_loop(xs, n - 1)

entry run(xs: [4]i32, n: i32) [2]i32 =
  [countdown(n), fixed_loop(xs, n)]
"#,
        ),
        LoweringProfile::PORTABLE,
    )
    .expect("plan scalar, fixed-array, and recursive call boundaries");

    let countdown =
        planned.functions.iter().find(|function| function.name.contains("countdown")).unwrap_or_else(
            || {
                panic!(
                    "countdown boundary not found among {:?}",
                    planned.functions.iter().map(|function| &function.name).collect::<Vec<_>>()
                )
            },
        );
    assert!(matches!(
        countdown.params().iter().next().unwrap().representation(),
        OperandType::Value(_)
    ));
    assert!(matches!(
        countdown.result().single_destination().unwrap().1,
        ResultDestination::ReturnValue(_)
    ));

    let fixed = planned.functions.iter().find(|function| function.name.contains("fixed_loop")).unwrap();
    assert!(matches!(
        fixed.params().iter().next().unwrap().representation(),
        OperandType::Place(place) if place.access == PlaceAccess::ReadOnly
    ));

    let recursive = countdown
        .graph
        .calls()
        .values()
        .find(|call| call.callee() == countdown.region)
        .expect("recursive call remains in countdown");
    assert_eq!(
        recursive.argument_bindings().iter().map(|argument| argument.parameter()).collect::<Vec<_>>(),
        countdown.params().ids().collect::<Vec<_>>()
    );
    assert_eq!(
        recursive
            .result()
            .destination_leaves_with_paths()
            .iter()
            .map(|(path, _)| path.as_ref())
            .collect::<Vec<_>>(),
        countdown
            .result()
            .destination_leaves_with_paths()
            .iter()
            .map(|(path, _)| path.as_ref())
            .collect::<Vec<_>>()
    );
    for function in &planned.functions {
        let declared = function.params().ids().collect::<std::collections::HashSet<_>>();
        assert!(function.graph.values().values().all(|node| {
            !matches!(node.kind(), ValueKind::FuncParam { parameter } if !declared.contains(parameter))
        }));
    }

    let extern_planned = egir::plan(
        compile_to_semantic_egir(
            r#"
#[linked("keep_abi")]
extern keep_abi(xs: [4]i32) [4]i32
entry call_extern(xs: [4]i32) [4]i32 = keep_abi(xs)
"#,
        ),
        LoweringProfile::PORTABLE,
    )
    .expect("plan an explicitly declared extern ABI");
    let declaration = &extern_planned.externs[0];
    assert_eq!(declaration.params.len(), 1);
    let call = extern_planned
        .entry_points
        .iter()
        .flat_map(|entry| entry.graph.calls().values())
        .find(|call| call.callee() == declaration.id)
        .expect("entry calls the extern");
    assert_eq!(call.arguments().len(), 1);
    assert!(matches!(
        call.arguments().next().unwrap(),
        egir::types::OperandRef::Value(_)
    ));
    assert!(matches!(
        call.result().single_destination().unwrap().1,
        ResultDestination::ReturnValue(_)
    ));
}

/// Construct a record-of-runtime-arrays from `map` outputs and RETURN it.
/// The declared `world` return must unify with the body's concrete
/// `composite`/`no_buffer` map-result arrays.
#[test]
fn record_of_arrays_construct_and_return_compiles() {
    compile_thru_spirv(
        r#"
open f32
type world = { points: []vec2f32, items: []vec4f32 }

def make_world(dom: []u32) world =
  let p = map(|i| @[f32(i), f32(i)], dom)
  let it = map(|i| @[f32(i), 0.0, 0.0, 1.0], dom) in
  { points = p, items = it }

entry step(dom: []u32) ([]vec2f32, []vec4f32) =
  let w = make_world(dom) in
  (w.points, w.items)
"#,
    )
    .expect("constructing and returning a record-of-arrays should compile");
}

/// A `map` output (`occ`) that is BOTH fed to another map (`occ[j%4]`) AND
/// returned. `occ` must be materialized to storage rather than left an
/// in-register runtime-sized Composite array.
#[test]
fn map_output_fed_and_returned_compiles() {
    compile_thru_spirv(
        r#"
entry frame(occ_dom: []u32, sett_dom: []u32) ([]u32, []u32) =
  let occ = map(|i| i + 7u32, occ_dom)
  let setts = map(|i| let j = i32(i) in i + occ[j % 4], sett_dom) in
  (occ, setts)
"#,
    )
    .expect("a map output both consumed and returned should compile");
}

/// The same producer→consumer dataflow with only the dependent array returned.
/// `occ` is consumed solely by dynamic index, so EGIR residency planning
/// materializes it.
#[test]
fn map_output_fed_but_only_dependent_returned_compiles() {
    compile_thru_spirv(
        r#"
def build(occ_dom: []u32, sett_dom: []u32) []u32 =
  let occ = map(|i| i + 7u32, occ_dom)
  let setts = map(|i| let j = i32(i) in i + occ[j % 4], sett_dom) in
  setts

entry frame(occ_dom: []u32, sett_dom: []u32) []u32 =
  build(occ_dom, sett_dom)
"#,
    )
    .expect("consuming a map output internally (not returned) should compile");
}

/// A `map` output carried in a record field (`w.points`), then both read by a
/// downstream map through the whole-record capture and returned.
/// Output realization retargets the producer `p` to the output view and the
/// record holds that view (`tuple(view)`); the capturing lambda's parameter
/// and internal `w.points` projection must receive the same representation.
#[test]
fn map_output_in_record_field_fed_and_returned_compiles() {
    compile_thru_spirv(
        r#"
open f32

type world = { points: []vec2f32 }

def build_geom(w: world, tdom: []u32) []vec4f32 =
  map(|i| let j = i32(i) in @[w.points[j % 8].x, 0.0, w.points[j % 8].y, 1.0], tdom)

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
    compile_thru_spirv(
        r#"
entry e(fb: *[]vec4f32) () =
  let cleared = map(|_p:vec4f32| @[0.0, 0.0, 0.0, 1.0], fb) in
  let _ = scatter(cleared, [0i32, 1i32], [@[1.0,1.0,1.0,1.0], @[1.0,1.0,1.0,1.0]]) in ()
"#,
    )
    .expect("clear-then-scatter on consuming `*[]T` write storage should compile end-to-end");
}

/// A compute entry may both *return* a Screma result and *consume*
/// it as a downstream side-effect's array input — here `new_pos` is
/// the entry's output and also the per-element input the scatter
/// envelope reads. Output routing preserves `new_pos` as the publication
/// source while physicalization binds its concrete destination.
#[test]
fn compute_entry_returns_screma_result_and_scatters_through_it() {
    compile_thru_spirv(
        r#"
def N:i32 = 8
def RES:i32 = 8

entry sim(prev: []vec4f32,
          fb: []vec4f32) []vec4f32 =
  let new_pos = map(|x:vec4f32| @[x.x + 1.0, x.y + 1.0, x.z, x.w], prev) in
  let idxs = map(|p:vec4f32| i32.f32(p.y) * RES + i32.f32(p.x), new_pos) in
  let vals = map(|p:vec4f32| @[1.0, 1.0, 1.0, 1.0], new_pos) in
  let _ = scatter(fb, idxs, vals) in
  new_pos
"#,
    )
    .expect("returning a Screma result while a downstream scatter consumes it should compile");
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
    compile_thru_spirv(
        r#"
def fsqrt(x: f32) f32 =
  f32.from_bits(0x1fbd1df5u32 + (f32.to_bits(x) >> 1u32))

entry e() [1]f32 = [fsqrt(4.0f32)]
"#,
    )
    .expect("f32.from_bits/to_bits should lower to OpBitcast");
}

/// A `map` compute entry whose lambda runs a loop containing an
/// `image_load` (storage image) and THEN does a `texture_load` (sampled
/// texture) panics during EGIR elaboration:
///
///   elaborate.rs: "FuncParam/BlockParam ValueId(..) should have been
///   pre-populated in elaborated map"
///
/// Both operations and their order are load-bearing: texture_load
/// before the loop, or an image_load in place of the texture_load,
/// compiles fine. This is the light-pass shape for driving GTAO
/// from the map/iota idiom (loop over shadow taps, then sample the AO
/// result).
#[test]
fn record_uniform_shared_across_stages_compiles() {
    compile_thru_spirv(
        r#"
type block = { radius: f32, tint: vec2f32 }

entry step(xs: []u32, c: block) []u32 =
  map(|x| x + u32(c.radius), xs)


entry vertex_main(vid: i32) vec4f32 =
  let verts = [@[-1.0, -1.0, 0.0, 1.0],
               @[3.0, -1.0, 0.0, 1.0],
               @[-1.0, 3.0, 0.0, 1.0]] in
  verts[vid]


entry fragment_main(pos: vec4f32,
                    c: block)
  vec4f32 =
  @[c.tint.x, c.tint.y, c.radius, 1.0]
"#,
    )
    .expect("stages sharing a record uniform block should lower");
}

/// Matrices stored in a runtime-array buffer need both the outer array stride
/// and the matrix column layout on the containing block member. Scalar
/// prepasses use exactly this representation when they hand a lifted camera
/// matrix back to a graphics stage.
#[test]
fn storage_matrix_elements_publish_std430_matrix_layout() {
    use wspirv::binary::parse_words;
    use wspirv::dr::{Loader, Operand};
    use wspirv::spirv::{Decoration, Op};

    let lowered = compile_thru_spirv(
        r#"
entry copy_matrix(
    input: []mat3f32,
    output: *[]mat3f32) () =
  let _ = scatter(output, [0i32], [input[0]]) in ()
"#,
    )
    .expect("matrix storage elements lower");

    let mut loader = Loader::new();
    parse_words(&lowered.spirv, &mut loader).expect("parse spirv");
    let module = loader.module();

    let mut matrix_stride_members = Vec::new();
    let mut column_major_members = Vec::new();
    let mut array_strides = Vec::new();
    for inst in &module.annotations {
        match inst.class.opcode {
            Op::MemberDecorate if inst.operands[2] == Operand::Decoration(Decoration::MatrixStride) => {
                matrix_stride_members.push((
                    inst.operands[0].unwrap_id_ref(),
                    inst.operands[1].unwrap_literal_bit32(),
                    inst.operands[3].unwrap_literal_bit32(),
                ));
            }
            Op::MemberDecorate if inst.operands[2] == Operand::Decoration(Decoration::ColMajor) => {
                column_major_members.push((
                    inst.operands[0].unwrap_id_ref(),
                    inst.operands[1].unwrap_literal_bit32(),
                ));
            }
            Op::Decorate if inst.operands[1] == Operand::Decoration(Decoration::ArrayStride) => {
                array_strides.push(inst.operands[2].unwrap_literal_bit32());
            }
            _ => {}
        }
    }

    assert!(
        matrix_stride_members
            .iter()
            .any(|&(block, member, stride)| stride == 16
                && column_major_members.contains(&(block, member))),
        "the matrix block member needs MatrixStride 16 and ColMajor; got {matrix_stride_members:?}"
    );
    assert!(
        array_strides.contains(&48),
        "a std430 runtime array of mat3f32 needs ArrayStride 48; got {array_strides:?}"
    );
}

/// Storage-buffer struct elements get std430 member offsets and an
/// aligned ArrayStride. For `{f32, vec2f32}`, std430 requires aligned member
/// offsets and stride matching naga's WGSL layout.
#[test]
fn storage_record_elements_get_std430_offsets_and_stride() {
    use std::collections::HashMap;
    use wspirv::binary::parse_words;
    use wspirv::dr::{Loader, Operand};
    use wspirv::spirv::{Decoration, Op};

    let lowered = compile_thru_spirv(
        r#"
type point = { w: f32, uv: vec2f32 }
entry e(o: *[]point) () =
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
fn mixed_variance_world_to_clip_call_exposes_camera_work_to_licm() {
    use crate::flow::ControlHeader;
    use crate::op::OpTag;
    use crate::ssa::types::InstKind;

    let source = r#"
def FOV: f32 = 20.0
def CLIP_NEAR: f32 = 0.1
def CLIP_FAR: f32 = 1000.0
type orbit = { target: vec3f32, az: f32, elev: f32, dist: f32 }
type frame_globals = {
  resolution: vec3f32,
  mods: u32,
  cam_target: vec3f32,
  cam_az: f32,
  cam_elev: f32,
  cam_dist: f32,
  time: f32,
}
def cam(f: frame_globals) orbit =
  { target = f.cam_target, az = f.cam_az, elev = f.cam_elev, dist = f.cam_dist }
def rotation(angle: vec2f32) mat3f32 =
  let c = @[f32.cos(angle.x), f32.cos(angle.y)]
  let s = @[f32.sin(angle.x), f32.sin(angle.y)] in
  @[[c.y,       0.0,       0.0 - s.y],
    [s.y * s.x, c.x,       c.y * s.x],
    [s.y * c.x, 0.0 - s.x, c.y * c.x]]
def cam_eye(o: orbit) vec3f32 =
  o.target + rotation(@[o.elev, o.az]) * @[0.0, 0.0, o.dist]
def perspective(fovy_deg: f32, aspect: f32, near: f32, far: f32) mat4f32 =
  let f = 1.0 / f32.tan(f32.radians(fovy_deg) * 0.5)
  let nf = 1.0 / (near - far) in
  @[[f / aspect, 0.0, 0.0,         0.0],
    [0.0,        f,   0.0,         0.0],
    [0.0,        0.0, far * nf,    0.0 - 1.0],
    [0.0,        0.0, far * near * nf, 0.0]]
def look_at(eye: vec3f32, center: vec3f32, up: vec3f32) mat4f32 =
  let f = normalize(center - eye)
  let s = normalize(cross(f, up))
  let u = cross(s, f) in
  @[[s.x,                u.x,                0.0 - f.x,     0.0],
    [s.y,                u.y,                0.0 - f.y,     0.0],
    [s.z,                u.z,                0.0 - f.z,     0.0],
    [0.0 - dot(s, eye),  0.0 - dot(u, eye),  dot(f, eye),   1.0]]
def world_to_clip(p: vec3f32, resolution: vec3f32, o: orbit) vec4f32 =
  let aspect = resolution.x / resolution.y
  let eye = cam_eye(o)
  let vm = look_at(eye, o.target, @[0.0, 1.0, 0.0])
  let proj = perspective(FOV, aspect, CLIP_NEAR, CLIP_FAR) in
  proj * (vm * @[p.x, p.y, p.z, 1.0])
def project_twenty_samples(base: vec3f32, resolution: vec3f32, o: orbit) vec4f32 =
  loop total = @[0.0, 0.0, 0.0, 0.0] for k < 20 do
    let fk = f32(k)
    let p = base + @[fk * 0.05, fk * 0.01, fk * 0.03] in
    total + world_to_clip(p, resolution, o)
entry world_to_clip_loop_invariant(
    points: []vec4f32,
    frame: frame_globals) []vec4f32 =
  let o = cam(frame) in
  map(|i| project_twenty_samples(points[i].xyz, frame.resolution, o), iota(1024))
"#;

    let converted = compile_thru_ssa(source).expect("camera LICM repro compiles to SSA");
    let function_id = |name: &str| {
        converted
            .functions
            .iter()
            .find(|function| function.name == name)
            .map(|function| function.id)
            .unwrap_or_else(|| panic!("missing SSA function `{name}`"))
    };
    let world_to_clip = function_id("world_to_clip");
    let camera_calls = [
        (function_id("rotation"), "rotation"),
        (function_id("look_at"), "look_at"),
        (function_id("perspective"), "perspective"),
    ];
    let project_id = function_id("project_twenty_samples");
    let project = converted
        .functions
        .iter()
        .find(|function| function.id == project_id)
        .expect("project helper remains as an SSA function");
    let (loop_header, continue_block) = project
        .body
        .inner
        .blocks
        .iter()
        .find_map(|(header, block)| match &block.control_header {
            Some(ControlHeader::Loop { continue_block, .. }) => Some((header, *continue_block)),
            Some(ControlHeader::Selection { .. }) | None => None,
        })
        .expect("project helper contains its source loop");
    let preheader = project.body.entry_block();
    assert_ne!(loop_header, preheader);

    let calls = project
        .body
        .inner
        .insts
        .values()
        .filter_map(|inst| match &inst.data {
            InstKind::Op {
                tag: OpTag::Call(function),
                ..
            } => Some((*function, inst.parent)),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert!(
        calls.iter().all(|(function, _)| *function != world_to_clip),
        "mixed call should be expanded, got {calls:?}"
    );
    for (camera_call, camera_name) in camera_calls {
        assert_eq!(
            calls.iter().filter(|(function, _)| *function == camera_call).copied().collect::<Vec<_>>(),
            vec![(camera_call, preheader)],
            "`{camera_name}` should execute once in the loop preheader; calls: {calls:?}"
        );
    }
    assert!(
        project.body.inner.blocks[continue_block].insts.iter().all(|inst| !matches!(
            project.body.inner.insts[*inst].data,
            InstKind::Op {
                tag: OpTag::Call(_),
                ..
            }
        )),
        "the loop body should contain no residual function calls"
    );

    lower_ssa_to_spirv(converted).expect("optimized camera repro lowers to valid SPIR-V");
}
