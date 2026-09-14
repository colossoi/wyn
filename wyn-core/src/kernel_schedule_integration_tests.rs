use super::*;
use crate::pipeline_descriptor::Pipeline;

fn assert_publication_agreement(program: &egir::parallelize::Planned) {
    program.validate_kernel_bodies().unwrap();
    let graph = program.physical_kernels();
    let kernels = graph.kernels().collect::<Vec<_>>();
    assert_eq!(
        graph.topological_kernel_ids(),
        kernels.iter().map(|kernel| kernel.id).collect::<Vec<_>>()
    );
    assert_eq!(
        program.entry_points.iter().map(|body| body.id).collect::<Vec<_>>(),
        kernels.iter().map(|kernel| kernel.entry).collect::<Vec<_>>()
    );
    for (index, pipeline) in program.data.pipeline.pipelines.iter().enumerate() {
        let names = match pipeline {
            Pipeline::Compute(compute) => {
                compute.stages.iter().map(|stage| stage.entry_point.as_str()).collect::<Vec<_>>()
            }
            Pipeline::Graphics(graphics) => {
                graphics.stages.iter().map(|stage| stage.entry_point.as_str()).collect::<Vec<_>>()
            }
        };
        let associated = &program.data.stage_entries[index];
        assert_eq!(names.len(), associated.len());
        for (name, entry) in names.iter().zip(associated) {
            assert!(kernels.iter().any(|kernel| kernel.entry == *entry && kernel.entry_point == *name));
        }
        if matches!(pipeline, Pipeline::Compute(_)) {
            let graph_order = kernels
                .iter()
                .filter(|kernel| associated.contains(&kernel.entry))
                .map(|kernel| kernel.entry_point.as_str())
                .collect::<Vec<_>>();
            assert_eq!(
                names, graph_order,
                "descriptor compute stages use the finalized kernel order"
            );
        }
    }
    for (position, kernel) in kernels.iter().enumerate() {
        for dependency in &kernel.dependencies {
            assert!(kernels[..position].iter().any(|prior| prior.id == *dependency));
        }
    }
}

#[test]
fn finalized_order_owns_recipe_bodies_and_descriptor_stages() {
    for source in [
        "entry sum(xs: []i32) i32 = reduce(|a: i32, b: i32| a + b, 0, xs)",
        "entry prefix(xs: []i32) []i32 = scan(|a: i32, b: i32| a + b, 0, xs)",
        "entry mixed() ([]i32, []i32) = (map(|i| i, iota(1)), filter(|i| true, iota(1)))",
        "entry buckets(dest: *[4][8]u32, items: [](i32, u32)) ([4][8]u32, [4]u32, u32) = bucket_scatter_1d(dest, items)",
    ] {
        let planned = plan_residency(compile_to_residency(source), LoweringProfile::PORTABLE).unwrap();
        assert_publication_agreement(&planned);
        let rebuilt = plan_residency(compile_to_residency(source), LoweringProfile::PORTABLE).unwrap();
        assert_eq!(planned.physical_kernels().topological_kernel_ids(), rebuilt.physical_kernels().topological_kernel_ids());
        assert_eq!(serde_json::to_value(&planned.data.pipeline).unwrap(), serde_json::to_value(&rebuilt.data.pipeline).unwrap());
    }
}

#[test]
fn serial_recipes_preserve_unsplit_outputs_without_allocating_parallel_scratch() {
    use crate::egir::parallelize::{
        allocate_recipe_scratch, build_kernel_schedule, physicalize_kernel_schedule,
    };
    let allocated = compile_to_residency(
        "entry mixed() ([]i32, []i32) = (map(|i| i, iota(1)), filter(|i| true, iota(1)))",
    );
    let stage_entries =
        allocated.data.stages.stage_records().map(|(_, stage)| stage.body().id).collect::<Vec<_>>();
    let resource_count = allocated.data.core.resources.len();
    let profile = LoweringProfile::new(CodegenTarget::Portable, SchedulePolicy::Serial);
    let analyzed = egir::finalize_staged_ir(allocated, profile).unwrap();
    let allocated = allocate_recipe_scratch(analyzed).unwrap();
    assert_eq!(allocated.data.core.resources.len(), resource_count);
    let planned = physicalize_kernel_schedule(build_kernel_schedule(allocated).unwrap()).unwrap();
    let kernels = planned.physical_kernels().kernels().collect::<Vec<_>>();
    assert_eq!(kernels.len(), stage_entries.len());
    assert!(kernels.iter().all(|kernel| stage_entries.contains(&kernel.entry)));
    assert_eq!(
        kernels.iter().map(|kernel| kernel.output_routes.len()).sum::<usize>(),
        2
    );
    assert_publication_agreement(&planned);
}

#[test]
fn mixed_projected_outputs_have_no_invented_cross_branch_dependency() {
    let planned = plan_residency(
        compile_to_residency(
            "entry mixed() ([]i32, []i32) = (map(|i| i, iota(1)), filter(|i| true, iota(1)))",
        ),
        LoweringProfile::PORTABLE,
    )
    .unwrap();
    let graph = planned.physical_kernels();
    let mapped = graph.kernels().find(|kernel| kernel.label == "serial_compute").unwrap();
    let flags = graph.kernels().find(|kernel| kernel.label == "filter_flags").unwrap();
    assert!(mapped.dependencies.is_empty());
    assert!(flags.dependencies.is_empty());
    assert_publication_agreement(&planned);
}

#[test]
fn graphics_publication_preserves_authored_associations_and_invocation() {
    let source = r#"
entry frame(points: []vec2f32, target: render_target<vec4f32>) ([]vec2f32, render_target<vec4f32>) =
  let updated = map(|p: vec2f32| p * 0.5, points) in
  let covered = rasterize_triangles(
    direct_draw(3u32, 1u32),
    |vertex_index, instance_index, draw_index|
      let p = updated[i32(vertex_index)] in
      vertex_output(@[p.x, p.y, 0.0, 1.0], p)) in
  let target' = shade(target, covered, |value, position, front, primitive, sample| @[value.x, value.y, 0.0, 1.0]) in
  (updated, target')
"#;
    let allocated = compile_to_residency(source);
    let authored_graphics = allocated
        .data
        .core
        .pipeline
        .pipelines
        .iter()
        .zip(&allocated.data.core.stage_entries)
        .filter_map(|(pipeline, entries)| {
            matches!(pipeline, Pipeline::Graphics(_)).then_some(entries.clone())
        })
        .collect::<Vec<_>>();
    let invocation = allocated
        .data
        .core
        .pipeline
        .pipelines
        .iter()
        .find_map(|pipeline| match pipeline {
            Pipeline::Graphics(graphics) => Some(graphics.invocation.clone()),
            _ => None,
        })
        .unwrap();
    let planned = plan_residency(allocated, LoweringProfile::PORTABLE).unwrap();
    assert_publication_agreement(&planned);
    let pipelines = &planned.data.pipeline.pipelines;
    let graphics_index =
        pipelines.iter().position(|pipeline| matches!(pipeline, Pipeline::Graphics(_))).unwrap();
    assert!(graphics_index > 0);
    assert!(pipelines[..graphics_index].iter().any(|pipeline| matches!(pipeline, Pipeline::Compute(_))));
    assert_eq!(planned.data.stage_entries[graphics_index], authored_graphics[0]);
    let Pipeline::Graphics(graphics) = &pipelines[graphics_index] else {
        unreachable!()
    };
    assert_eq!(graphics.invocation, invocation);
}

#[test]
fn explicit_single_workgroup_dispatch_is_preserved_and_coverage_is_checked() {
    use crate::egir::parallelize::KernelDomain;
    use crate::pipeline_descriptor::DispatchSize;
    for count in [1, 1024] {
        let source = format!("entry mapped() []i32 = map(|i: i32| i + 1, iota({count}))");
        let mut allocated = compile_to_residency(&source);
        let Pipeline::Compute(compute) = &mut allocated.data.core.pipeline.pipelines[0] else {
            unreachable!()
        };
        compute.stages[0].dispatch_size = DispatchSize::Fixed {
            x: 1,
            y: 1,
            z: 1,
            explicit: true,
        };
        let result = plan_residency(allocated, LoweringProfile::PORTABLE);
        if count == 1 {
            let planned = result.unwrap();
            assert_eq!(
                planned.physical_kernels().kernels().next().unwrap().domain,
                KernelDomain::Fixed { x: 1, y: 1, z: 1 }
            );
            let Pipeline::Compute(compute) = &planned.data.pipeline.pipelines[0] else {
                unreachable!()
            };
            assert_eq!(
                compute.stages[0].dispatch_size,
                DispatchSize::Fixed {
                    x: 1,
                    y: 1,
                    z: 1,
                    explicit: true
                }
            );
        } else {
            let error = match result {
                Err(error) => error,
                Ok(_) => panic!("insufficient explicit grid must fail before publication"),
            };
            assert!(
                matches!(error, egir::from_tlc::ConvertError::InvalidDispatch(_)),
                "{error}"
            );
        }
    }
}
