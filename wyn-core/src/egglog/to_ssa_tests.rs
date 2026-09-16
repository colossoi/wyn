use crate::egglog::{
    convert_program, insert_expressions, optimize, optimize_expressions, schedule, to_ssa,
};
use crate::{compile_thru_tlc, lower_ssa_to_wgsl, tlc};

fn compile(source: &str) -> naga::Module {
    let tlc = tlc::infer_input_slice_bounds(compile_thru_tlc(source).unwrap());
    let program = schedule(
        optimize_expressions(
            insert_expressions(optimize(convert_program(&tlc).unwrap()).unwrap()).unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    let ssa = to_ssa(&program.data, crate::CodegenTarget::Wgsl).unwrap();
    let source = lower_ssa_to_wgsl(ssa).unwrap();
    let module = naga::front::wgsl::parse_str(&source)
        .unwrap_or_else(|e| panic!("{}\n{source}", e.emit_to_string(&source)));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("{e:?}\n{source}"));
    module
}

#[test]
fn fused_map_reduce_reaches_the_existing_wgsl_backend() {
    compile("entry main(xs: []f32) f32 = reduce(|a: f32, b: f32| a + b, 0.0, map(|x: f32| x * 2.0, xs))");
}

#[test]
fn fused_indexed_updates_reach_wgsl() {
    compile(
        "entry main(dest: *[3]i32, xs: [5]i32) [3]i32 =
        reduce_by_index(dest, |a:i32,b:i32|a+b, 0,
            map(|x:i32|x-1,xs), map(|x:i32|x*3,xs))",
    );
}

#[test]
fn filtered_reduction_and_shared_count_reach_wgsl() {
    compile(
        "entry main(xs: []i32) (i32,i32) =
        let kept=filter(|x:i32|x>0,xs) in
        (length(kept),reduce(|a:i32,b:i32|a+b,0,kept))",
    );
}

#[test]
fn sliced_fused_map_reaches_wgsl() {
    compile(
        "entry main(xs: []i32) [4]i32 =
        let a=map(|x:i32|x+1,xs) in map(|x:i32|x*2,a[2..6])",
    );
}

#[test]
fn scalar_loop_reaches_the_existing_wgsl_backend() {
    compile("entry main(n: i32) i32 = loop acc = 0 for i < n do acc + i");
}

#[test]
fn scan_and_filter_emit_all_scheduled_kernels() {
    let scan = compile("entry main(xs: []i32) []i32 = scan(|a: i32, b: i32| a + b, 0, xs)");
    let filter = compile("entry main(xs: []i32) ?k. [k]i32 = filter(|x: i32| x % 3 == 1, xs)");
    assert_eq!(scan.entry_points.len(), 3);
    assert_eq!(filter.entry_points.len(), 4);
}

#[test]
fn tuple_collectives_and_runtime_captures_reach_wgsl() {
    compile("entry main(xs: []i32, bias: i32) (i32, i32) = reduce(|a: (i32, i32), b: (i32, i32)| (a.0 + b.0, a.1 + b.1), (0, 0), map(|x: i32| (x + bias, 1), xs))");
}

#[test]
fn nested_device_collectives_reach_wgsl() {
    compile("entry main(xs: []i32) []i32 = map(|x: i32| reduce(|a: i32, b: i32| a + b, 0, map(|y: i32| y + x, iota(5))), xs)");
}

#[test]
fn array_and_while_loops_reach_wgsl() {
    compile(
        "entry main(xs: [4]i32, n: i32) (i32, i32) =
        let a = loop acc = 0 for x in xs do acc + x in
        let (b, _) = loop (acc, i) = (0, 0) while i < n do (acc + i, i + 1) in (a, b)",
    );
}

#[test]
fn conditional_expressions_inside_device_loops_reach_wgsl() {
    compile("entry main(xs: []i32) []i32 = map(|x: i32| loop acc = 0 for i < x do if i % 2 == 0 then acc + i else acc, xs)");
}

#[test]
fn host_control_requires_a_runtime_cfg_instead_of_a_false_static_descriptor() {
    let source =
        "entry main(xs: [4]i32, n: i32) [4]i32 = loop acc = xs for k < n do map(|x: i32| x + k, acc)";
    let tlc = tlc::infer_input_slice_bounds(compile_thru_tlc(source).unwrap());
    let program = schedule(
        optimize_expressions(
            insert_expressions(optimize(convert_program(&tlc).unwrap()).unwrap()).unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    let error = to_ssa(&program.data, crate::CodegenTarget::Wgsl).err().unwrap();
    assert!(error.to_string().contains("conditional or repeated host dispatches"));
}

#[test]
fn existing_spirv_backend_also_accepts_the_handoff() {
    let tlc = tlc::infer_input_slice_bounds(
        compile_thru_tlc("entry main(xs: []i32) []i32 = map(|x: i32| x * 2, xs)").unwrap(),
    );
    let program = schedule(
        optimize_expressions(
            insert_expressions(optimize(convert_program(&tlc).unwrap()).unwrap()).unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    let ssa = to_ssa(&program.data, crate::CodegenTarget::Spirv).unwrap();
    let output = crate::lower_ssa_to_spirv(ssa).unwrap();
    let module = wspirv::dr::load_words(output.spirv).unwrap();
    assert_eq!(module.entry_points.len(), 1);
}

#[test]
fn vector_input_and_runtime_gather_after_scan_reach_wgsl() {
    compile(include_str!("../../../testfiles/gather_scan_chain.wyn"));
}

fn pipeline(source: &str) -> crate::LoweredWgsl {
    let tlc = tlc::infer_input_slice_bounds(compile_thru_tlc(source).unwrap());
    let program = schedule(
        optimize_expressions(
            insert_expressions(optimize(convert_program(&tlc).unwrap()).unwrap()).unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    crate::lower_ssa_to_wgsl_with_pipeline(to_ssa(&program.data, crate::CodegenTarget::Wgsl).unwrap())
        .unwrap()
}

#[test]
fn runtime_input_lengths_and_output_allocations_share_the_published_bindings() {
    use crate::pipeline_descriptor::{Binding, BufferLen, Pipeline};
    let output = pipeline("entry main(xs: []i32, bias: i32) []i32 = map(|x:i32|x+bias, xs)");
    let module = naga::front::wgsl::parse_str(&output.wgsl).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    assert!(module.entry_points.iter().any(|e| e
        .function
        .expressions
        .iter()
        .any(|(_, x)| matches!(x, naga::Expression::ArrayLength(_)))));
    let [Pipeline::Compute(p)] = output.pipeline.pipelines.as_slice() else {
        panic!("compute pipeline")
    };
    assert_eq!(p.stages.len(), 1);
    assert_eq!(p.stages[0].owner, "main");
    let result = &output.pipeline.source_results[0];
    assert!(p.bindings.iter().any(|b| matches!(b, Binding::StorageBuffer { set, binding, length: Some(BufferLen::LikeInput { set: 0, binding: 0, elem_bytes: 4, src_elem_bytes: 4 }), .. } if (*set, *binding) == (result.set, result.binding))));
    assert!(p.bindings.iter().any(
        |b| matches!(b, Binding::StorageBuffer { members, .. } if members.iter().any(|m| m.name == "bias"))
    ));
    assert!(p.bindings.iter().all(|b| !matches!(b, Binding::PushConstant { .. })));
}

#[test]
fn reduction_publication_has_scratch_writers_readers_and_a_source_result() {
    use crate::pipeline_descriptor::{Binding, BufferLen, Pipeline};
    let output = pipeline("entry main(xs: [137]i32) i32 = reduce(|a:i32,b:i32|a+b,0,xs)");
    let [Pipeline::Compute(p)] = output.pipeline.pipelines.as_slice() else {
        panic!("compute pipeline")
    };
    assert_eq!(p.stages.len(), 3, "chunks, combine, scalar result publication");
    let scratch = p
        .bindings
        .iter()
        .position(|b| {
            matches!(
                b,
                Binding::StorageBuffer {
                    length: Some(BufferLen::Fixed { bytes: 12 }),
                    ..
                }
            )
        })
        .unwrap();
    assert!(p.stages[0].writes.contains(&scratch));
    assert!(p.stages[1].reads.contains(&scratch));
    assert!(!p.stages[1].writes.contains(&scratch));
    assert_eq!(output.pipeline.source_results.len(), 1);
    assert_eq!(output.pipeline.frame_graph.passes.len(), 3);
    assert!(output.pipeline.frame_graph.topological_order().is_ok());
}

#[test]
fn independent_entries_keep_their_own_dispatches_and_results() {
    let output = pipeline("entry first(xs:[5]i32) [5]i32 = map(|x:i32|x+1,xs)\nentry second(xs:[9]i32) [9]i32 = map(|x:i32|x*2,xs)");
    assert_eq!(output.pipeline.pipelines.len(), 2);
    assert_eq!(output.pipeline.source_results.len(), 2);
    assert_ne!(
        output.pipeline.source_results[0].pipeline_index,
        output.pipeline.source_results[1].pipeline_index
    );
    assert_ne!(
        output.pipeline.source_results[0].binding,
        output.pipeline.source_results[1].binding
    );
}

#[test]
fn scalar_results_after_collectives_are_executed_and_published() {
    let output = pipeline("entry main(xs:[]i32) i32 = reduce(|a:i32,b:i32|a+b,0,xs) * 3 + 1");
    assert_eq!(output.pipeline.source_results.len(), 1);
    let module = naga::front::wgsl::parse_str(&output.wgsl).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    assert_eq!(module.entry_points.len(), 3);
}

#[test]
fn in_place_results_keep_the_host_input_binding_and_upload_role() {
    use crate::pipeline_descriptor::{Access, Binding, BufferUsage, Pipeline};
    let output = pipeline("entry main(dest:*[3]i32,xs:[5]i32) [3]i32 = reduce_by_index(dest,|a:i32,b:i32|a+b,0,map(|x:i32|x%3,xs),xs)");
    let result = &output.pipeline.source_results[0];
    assert_eq!((result.set, result.binding), (0, 0));
    let Pipeline::Compute(p) = &output.pipeline.pipelines[0] else {
        panic!("compute pipeline")
    };
    assert!(p.bindings.iter().any(|b| matches!(
        b,
        Binding::StorageBuffer {
            set: 0,
            binding: 0,
            usage: BufferUsage::Input,
            access: Access::ReadWrite,
            ..
        }
    )));
}

#[test]
fn tuple_of_views_uses_the_tlc_component_bindings() {
    let source = "entry main(xs: ([]i32,[]i32)) ([]i32,[]i32) = xs";
    let tlc = tlc::infer_input_slice_bounds(compile_thru_tlc(source).unwrap());
    let imported = convert_program(&tlc).unwrap();
    let expected: Vec<_> = imported
        .data
        .entry_params
        .values()
        .filter_map(|p| p.binding.as_ref())
        .flat_map(|p| match &p.kind {
            crate::interface::EntryParamBindingKind::Single { binding, .. } => vec![binding.binding],
            crate::interface::EntryParamBindingKind::TupleOfViews(fields) => {
                fields.iter().map(|f| f.binding.binding).collect()
            }
        })
        .collect();
    let output = pipeline(source);
    assert_eq!(output.pipeline.source_results.len(), 2);
    assert_eq!(
        output.pipeline.source_results.iter().map(|o| o.binding).collect::<Vec<_>>(),
        expected
    );
    let module = naga::front::wgsl::parse_str(&output.wgsl).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
}
