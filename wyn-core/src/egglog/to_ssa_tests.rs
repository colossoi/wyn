use crate::egglog::{from_tlc, fuse, optimize, place, schedule, to_ssa};
use crate::host::Pipeline;
use crate::interface::ComputeDispatchGrid;
use crate::tlc::infer_input_slice_bounds;
use crate::tlc::DefMeta;
use crate::PipelineTopologyPolicy;
use crate::{
    compile_thru_spirv, compile_thru_tlc, host, lower_ssa_to_spirv, lower_ssa_to_wgsl,
    lower_ssa_to_wgsl_with_program, CodegenTarget, LoweredWgsl,
};

use naga::front::spv;
use naga::valid::{Capabilities, ValidationFlags, Validator};

#[test]
fn zipped_runtime_loop_retains_its_domain() {
    let source = "entry main(xs:[]i32) []i32 =
      let pairs=zip(xs,iota(length(xs))) in
      let count=if length(xs)==0 then 0 else 3 in
      let ys=loop ys=pairs for i<count do map(|(x,j)| (x+j,j),ys) in
      map(|(x,j)| x,ys)";
    compile(source);
}

#[test]
fn missing_selected_abi_facts_are_errors() {
    for (table, diagnostic) in [
        ("SelectedLaunch", "missing selected SelectedLaunch"),
        ("RootWorkgroup", "missing selected RootWorkgroup"),
        ("ParameterAbi", "missing selected ParameterAbi"),
        ("AbiStorage", "has no binding"),
    ] {
        let tlc = infer_input_slice_bounds(
            compile_thru_tlc("entry main(xs:[]i32,k:i32) []i32=map(|x|x+k,xs)").unwrap(),
        );
        let mut program = optimize(
            schedule(
                place(
                    fuse(from_tlc(&tlc).unwrap()).unwrap(),
                    PipelineTopologyPolicy::AllowGenerated,
                )
                .unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
        program.graph.clear_function(table).unwrap();
        let Err(error) = to_ssa(program, CodegenTarget::Wgsl) else {
            panic!("missing {table} was silently accepted");
        };
        assert!(error.to_string().contains(diagnostic), "{table}: {error}");
    }
}

#[test]
fn invalid_selected_workgroup_is_an_error() {
    use egglog_engine::Write;
    let tlc =
        infer_input_slice_bounds(compile_thru_tlc("entry main(xs:[4]i32) [4]i32=map(|x|x+1,xs)").unwrap());
    let mut program = optimize(
        schedule(
            place(
                fuse(from_tlc(&tlc).unwrap()).unwrap(),
                PipelineTopologyPolicy::AllowGenerated,
            )
            .unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    let mut roots = Vec::new();
    program.graph.function_entries("RootWorkgroup", |row| roots.push(row.inputs[0])).unwrap();
    program.graph.clear_function("RootWorkgroup").unwrap();
    program
        .graph
        .update(|mut sink| {
            let grid = sink.add("FixedGrid", (0i64, 1i64, 1i64))?;
            for root in roots {
                sink.set("RootWorkgroup", root, grid)?;
            }
            Ok(())
        })
        .unwrap();
    let Err(error) = to_ssa(program, CodegenTarget::Wgsl) else {
        panic!("zero-width workgroup was silently accepted");
    };
    assert!(error.to_string().contains("grid x must be positive"), "{error}");
}

#[test]
fn partial_maps_stream_only_over_the_whole_domain() {
    for (source, count) in [
        ("entry main(xs:[8]i32) [8]i32=map(|x|x+1,map(|x|12/x,xs))", 1),
        (
            "entry main(xs:[8]i32) []i32=let ys=map(|x|12/x,xs) in map(|x|x+1,ys[0..4])",
            2,
        ),
    ] {
        let output = pipeline(source);
        let [Pipeline::Compute(pipeline)] = output.program.interface.pipelines.as_slice() else {
            panic!("one compute pipeline");
        };
        assert_eq!(pipeline.stages.len(), count, "{source}");
    }
}

fn compile(source: &str) -> naga::Module {
    let ssa = crate::compile_thru_ssa_for_target(source, CodegenTarget::Wgsl)
        .unwrap_or_else(|error| panic!("{error}\n{source}"));
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
fn loop_local_scratch_annotation_is_consumed_before_codegen() {
    let source = "entry main(k:i32) [4]i32 =
        loop xs=[1,2,3,4] for i<k do
        scatter((#[scratch] map(|x|x,xs)),[3,2,1,0],xs)";
    let program = crate::compile_thru_ssa(source).unwrap();
    crate::lower_ssa_to_spirv(program.clone()).unwrap();
    crate::lower_ssa_to_wgsl(program).unwrap();
}

#[test]
fn loop_local_zip_preserves_nested_input_elements() {
    let source = "entry main(xs:[5]i32) [5]i32 =
        loop xs=xs for i<2 do
        let bins=map(|x|x & 3,xs) in
        let offsets=scan(|(a,b,c,d),(e,f,g,h)|(a+e,b+f,c+g,d+h),
            (0,0,0,0),map(|x|(1,1,1,1),xs)) in
        map(|(bin,(a,b,c,d))|i32(bin)+a+b+c+d,zip(bins,offsets))";
    let program = crate::compile_thru_ssa(source).unwrap();
    crate::lower_ssa_to_spirv(program.clone()).unwrap();
    crate::lower_ssa_to_wgsl(program).unwrap();
}

#[test]
fn nested_tuple_loop_state_preserves_component_arrays() {
    use host::{Binding, BufferLen, ResultLayout, ResultScalar};
    let source = include_str!("../../../testfiles/regressions/nested_tuple_loop_state.wyn");
    compile(source);
    for descriptor in [
        pipeline(source).program.interface,
        crate::compile_thru_spirv(source).unwrap().program.interface,
    ] {
        assert_eq!(descriptor.source_results.len(), 4);
        for result in &descriptor.source_results {
            let Pipeline::Compute(p) = &descriptor.pipelines[result.pipeline_index] else {
                panic!("compute result");
            };
            let ResultLayout::Array {
                element,
                stride,
                length: Some(2),
            } = &result.layout
            else {
                panic!("fixed array result: {:?}", result.layout);
            };
            if result.result == 0 {
                assert_eq!(*stride, 4);
                assert_eq!(**element, ResultLayout::Scalar(ResultScalar::I32));
            } else {
                assert_eq!(*stride, 8);
                let ResultLayout::Tuple { fields, size: 8 } = &**element else {
                    panic!("tuple element: {element:?}");
                };
                assert_eq!(fields.len(), 2);
                assert_eq!(fields[0].offset, 0);
                assert_eq!(fields[1].offset, 4);
                assert_eq!(fields[1].layout, ResultLayout::Scalar(ResultScalar::Bool));
            }
            assert!(p.bindings.iter().any(|b| matches!(b,
                Binding::StorageBuffer { set, binding, length: Some(BufferLen::Fixed { bytes }), .. }
                    if (*set, *binding) == (result.set, result.binding) && *bytes == u64::from(*stride) * 2
            )));
        }
    }
}

#[test]
fn loop_local_tuple_collectives_preserve_component_arrays() {
    let source = include_str!("../../../testfiles/regressions/local_tuple_collectives.wyn");
    compile(source);
    crate::compile_thru_spirv(source).unwrap();
}

#[test]
fn length_only_filter_view_lowers_without_its_discarded_element_buffer() {
    let source = "entry main(xs:[]i32) i32=length(filter(|x:i32|x>0,xs))";
    compile(source);
    crate::compile_thru_spirv(source).unwrap();
}

#[test]
fn length_queries_lower_without_materializing_array_elements() {
    compile(include_str!("../../../testfiles/length_metadata.wyn"));
}

#[test]
fn scalar_output_epilogues_lower_without_a_finish_entry() {
    let module = compile(include_str!("../../../testfiles/scalar_epilogues.wyn"));
    assert_eq!(module.entry_points.len(), 3);
    assert!(module.entry_points.iter().all(|entry| !entry.name.ends_with("finish")));
}

#[test]
fn filter_count_draw_record_is_published_by_compaction() {
    let module = compile("def draw(n:i32) (u32,u32,u32,u32) = (18u32,u32(n),0u32,0u32)
entry main(xs:[]i32) ([]i32,(u32,u32,u32,u32)) = let ys=filter(|x:i32|x>0,xs) in (map(|x:i32|x+1,ys),draw(length(ys)))");
    assert_eq!(module.entry_points.len(), 1);
}

#[test]
fn filter_post_map_record_outputs_reach_wgsl() {
    let module = compile(include_str!("../../../testfiles/rust_host_filter_post.wyn"));
    let entries: Vec<_> = module.entry_points.iter().map(|entry| entry.name.as_str()).collect();
    assert!(entries.iter().any(|name| name.ends_with("compact")));
    assert_eq!(module.entry_points[0].workgroup_size, [64, 1, 1]);
}

#[test]
fn filter_element_read_epilogue_lowers_in_a_separate_dispatch() {
    let module=compile("entry main(xs:[]i32) ([]i32,[2]i32) = let ys=filter(|x:i32|x>0,xs) in (ys,[length(ys),if length(ys)>0 then ys[length(ys)-1] else -1])");
    assert!(module.entry_points.len() >= 2);
}

#[test]
fn literal_and_runtime_nested_array_indices_reach_wgsl() {
    compile(include_str!("../../../testfiles/composite_2d_local.wyn"));
}

#[test]
fn mixed_array_scalar_stage_outputs_reach_wgsl() {
    compile(include_str!(
        "../../../testfiles/regressions/stage_shared_mixed_tuple.wyn"
    ));
}

#[test]
fn conditional_fixed_array_outputs_reach_wgsl() {
    for array in ["[n]", "if flag then [n] else [n+1]"] {
        compile(&format!(
            "entry repro(xs:[]i32,flag:bool) ([]i32,[1]i32)=let n=xs[0] in (map(|x|x+n,xs),{array})"
        ));
    }
}

#[test]
fn nested_mountain_shader_reaches_valid_wgsl() {
    std::thread::Builder::new()
        .stack_size(32 * 1024 * 1024)
        .spawn(|| {
            compile(include_str!("../../../testfiles/playground/mountains.wyn"));
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn ranked_bucket_scatter_writes_through_nested_storage() {
    compile(
        "entry main(dest: *[2][2]i32) ([2][2]i32, [2]u32, u32) =
        bucket_scatter_2d(dest,
            [[(-1, 9), (0, 10), (0, 11)], [(0, 12), (1, 20), (2, 9)]])",
    );
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
fn integer_indexed_reductions_select_atomic_and_compare_exchange_kernels() {
    for (operator, primitive) in [
        ("a+b", "atomicAdd"),
        ("a&b", "atomicAnd"),
        ("a|b", "atomicOr"),
        ("a^b", "atomicXor"),
        ("max(a,b)", "atomicCompareExchangeWeak"),
    ] {
        let source = format!(
            "entry main(dest:*[3]i32, xs:[137]i32) [3]i32 =
            reduce_by_index(dest, |a:i32,b:i32| {operator}, 0, map(|x:i32|x%3,xs), xs)"
        );
        compile(&source);
        let output = pipeline(&source);
        assert!(output.wgsl.contains(primitive), "{primitive}: {}", output.wgsl);
        let Pipeline::Compute(p) = &output.program.interface.pipelines[0] else {
            panic!("compute");
        };
        assert_eq!(p.stages[0].workgroup_size, (64, 1, 1));
    }
}

#[test]
fn cooperative_recipes_emit_barriers_and_workgroup_storage() {
    for source in [
        "entry main(xs:[]i32) i32 = reduce(|a:i32,b:i32|a+b,0,xs)",
        "entry main(xs:[]i32) []i32 = filter(|x:i32|x%3==0,xs)",
    ] {
        compile(source);
        let output = pipeline(source);
        assert!(output.wgsl.contains("workgroupBarrier"));
        assert!(output.wgsl.contains("var<workgroup>"));
    }
}

#[test]
fn scalar_domain_collectives_publish_scratch_and_capacity_dependencies() {
    for source in [
        include_str!("../../../testfiles/reduce_map_iota.wyn"),
        include_str!("../../../testfiles/tinyporto_filter_scan.wyn"),
    ] {
        compile(source);
        let output = pipeline(source);
        let whl = output.program.to_whl("collectives.wgsl", host::ShaderFormat::Wgsl).unwrap();
        wyn_host_interp::Program::parse(&whl).unwrap();
        assert!(output.program.entries.iter().any(|entry| !entry.allocations.is_empty()));
    }
}

#[test]
fn external_functions_retain_their_declared_signatures() {
    let tlc = infer_input_slice_bounds(
        compile_thru_tlc(
            "#[linked(\"foreign_add\")] extern add(a:i32,b:i32) i32\nentry main(x:i32) i32 = add(x, 3)",
        )
        .unwrap(),
    );
    let program = optimize(
        schedule(
            place(
                fuse(from_tlc(&tlc).unwrap()).unwrap(),
                PipelineTopologyPolicy::AllowGenerated,
            )
            .unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    let ssa = to_ssa(program, CodegenTarget::Spirv).unwrap();
    let external = ssa.functions.iter().find(|f| f.linkage_name.as_deref() == Some("foreign_add")).unwrap();
    assert_eq!(external.body.params().len(), 2);
    assert_eq!(external.body.return_ty, crate::types::i32());
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
fn invocation_local_filtered_and_mixed_collectives_reach_both_backends() {
    let source = include_str!("../../../testfiles/rust_host_collectives.wyn");
    compile(source);
    let spirv = compile_thru_spirv(source).unwrap();
    let bytes: Vec<_> = spirv.spirv.iter().flat_map(|word| word.to_le_bytes()).collect();
    let module = spv::parse_u8_slice(&bytes, &spv::Options::default()).unwrap();
    Validator::new(ValidationFlags::all(), Capabilities::all()).validate(&module).unwrap();
}

#[test]
fn mapped_filter_result_keeps_its_runtime_sized_view() {
    compile(include_str!("../../../testfiles/filter_then_map.wyn"));
}

#[test]
fn signed_and_unsigned_integer_power_reach_wgsl() {
    compile(include_str!("../../../testfiles/int_pow.wyn"));
}

#[test]
fn invocation_local_filter_keeps_a_bounded_array_length() {
    compile(include_str!("../../../testfiles/filter_demo.wyn"));
}

#[test]
fn computed_array_outputs_have_storage_and_a_writer() {
    use host::{Binding, BufferLen};
    for source in [
        include_str!("../../../testfiles/array_param_view_multi.wyn"),
        include_str!("../../../testfiles/array_param_view_slice.wyn"),
    ] {
        compile(source);
        let output = pipeline(source);
        let [result] = output.program.interface.source_results.as_slice() else {
            panic!("one array result");
        };
        let Pipeline::Compute(p) = &output.program.interface.pipelines[result.pipeline_index] else {
            panic!("compute result");
        };
        assert!(p.bindings.iter().any(|b| matches!(b, Binding::StorageBuffer {
            set, binding, length: Some(BufferLen::Fixed { bytes: 4 }), ..
        } if (*set, *binding) == (result.set, result.binding))));
    }
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
fn scalar_loop_result_is_stored_before_parallel_consumers() {
    use Pipeline;
    let source = "entry main(xs: []i32, n: i32) []i32 =
        let bias = loop acc = 0 for i < n do acc + i in
        map(|x: i32| x + bias, xs)";
    compile(source);
    let output = pipeline(source);
    let [Pipeline::Compute(p)] = output.program.interface.pipelines.as_slice() else {
        panic!("compute pipeline");
    };
    assert_eq!(p.stages.len(), 2);
    assert_eq!(p.stages[0].workgroup_size, (1, 1, 1));
    assert_eq!(p.stages[1].workgroup_size, (64, 1, 1));
}

#[test]
fn scan_and_filter_emit_all_scheduled_kernels() {
    let scan = compile("entry main(xs: []i32) []i32 = scan(|a: i32, b: i32| a + b, 0, xs)");
    let filter = compile("entry main(xs: []i32) ?k. [k]i32 = filter(|x: i32| x % 3 == 1, xs)");
    assert_eq!(scan.entry_points.len(), 3);
    assert_eq!(filter.entry_points.len(), 1);
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
fn branching_loop_tests_and_header_array_reads_reach_wgsl() {
    compile("entry main(xs: [4]i32, limit: i32) i32 = loop acc = 0 while (if acc < limit then xs[acc % 4] > 0 else false) do acc + 1");
    compile("entry main(xs: [4]i32) i32 = loop acc = 0 for i < xs[0] do acc + i");
}

#[test]
fn runtime_control_keeps_collectives_inside_device_execution() {
    for source in [
        "entry main(xs: [4]i32, n: i32) [4]i32 = loop acc = xs for k < n do map(|x: i32| x + k, acc)",
        "entry main(xs: [4]i32, n: i32) i32 = loop acc = 0 for k < n do acc + reduce(|a:i32,b:i32|a+b, 0, map(|x:i32|x+k,xs))",
        "entry main(xs: [4]i32, yes: bool) [4]i32 = if yes then map(|x:i32|x+1,xs) else map(|x:i32|x-1,xs)",
    ] {
        compile(source);
    }
}

#[test]
fn scatter_local_destinations_reach_wgsl() {
    for source in [
        include_str!("../../../testfiles/regressions/scatter_radix_bit.wyn"),
        "entry main(n: i32) [4]i32 = loop acc = [7, 7, 7, 7] for k < n do scatter(acc, [k], [k+10])",
        "entry main() [4]i32 = scatter(replicate(4, 7i32), [2i32, 0i32], [30i32, 10i32])",
        "entry main(xs: []i32) []i32 = scatter(replicate(length(xs), 7i32), [2i32, 0i32], [30i32, 10i32])",
        "entry main() [4]i32 = scatter([7i32, 7i32, 7i32, 7i32], [2i32, 0i32], [30i32, 10i32])",
        "entry main() [4]i32 = spread(4, 7i32, [2i32, 0i32], [30i32, 10i32])",
    ] {
        compile(source);
    }
}

#[test]
fn stored_fixed_array_updates_materialize_owned_contents() {
    for source in [
        "entry main() [258]i32 =
            ((replicate(258,-1i32) with [0] = 0) with [256] = 1) with [257] = 0",
        "entry main(n:i32) [258]i32 =
            let empty = replicate(258,-1i32) in
            if n == 0 then (empty with [256] = 0) with [257] = 0 else
            let (result,count) = loop (result,count) = (empty,0i32)
                while count < n do (result with [count] = count,count+1) in
            (result with [256] = count) with [257] = 0",
    ] {
        compile(source);
        let output = compile_thru_spirv(source).unwrap();
        let bytes: Vec<_> = output.spirv.iter().flat_map(|word| word.to_le_bytes()).collect();
        let module = spv::parse_u8_slice(&bytes, &spv::Options::default()).unwrap();
        Validator::new(ValidationFlags::all(), Capabilities::all()).validate(&module).unwrap();
    }
}

#[test]
fn array_updates_preserve_loop_state_and_output_capacity() {
    use crate::host::interface::{Binding, BufferLen};
    for source in [
        include_str!("../../../testfiles/regressions/array_update_return.wyn"),
        include_str!("../../../testfiles/regressions/array_update_loop.wyn"),
        include_str!("../../../testfiles/regressions/array_update_branch.wyn"),
        include_str!("../../../testfiles/regressions/array_update_nested_queue.wyn"),
        "entry main(n: i32) [4]i32 =
            let (_, output) = loop (i, output) = (0i32, replicate(4, -1i32))
                while i < n do (i+1, output with [i % 4] = i) in output",
    ] {
        compile(source);
        for descriptor in [
            pipeline(source).program.interface,
            crate::compile_thru_spirv(source).unwrap().program.interface,
        ] {
            let [result] = descriptor.source_results.as_slice() else {
                panic!("one array result");
            };
            let Pipeline::Compute(p) = &descriptor.pipelines[result.pipeline_index] else {
                panic!("compute result");
            };
            assert!(
                p.bindings.iter().any(|b| matches!(b, Binding::StorageBuffer {
            set, binding, length: Some(BufferLen::Fixed { bytes: 16 }), ..
        } if (*set, *binding) == (result.set, result.binding))),
                "{source}\n{:?}",
                p.bindings
            );
        }
    }
}

#[test]
fn existing_spirv_backend_also_accepts_the_handoff() {
    for source in [
        include_str!("../../../testfiles/regressions/stage_shared_mixed_tuple.wyn"),
        include_str!("../../../testfiles/regressions/array_update_return.wyn"),
        include_str!("../../../testfiles/regressions/array_update_loop.wyn"),
        include_str!("../../../testfiles/regressions/scatter_radix_bit.wyn"),
        "entry main(n: i32) [4]i32 = loop acc = [7, 7, 7, 7] for k < n do scatter(acc, [k], [k+10])",
        "entry main() [4]i32 = scatter(replicate(4, 7i32), [2i32, 0i32], [30i32, 10i32])",
        "entry main(xs: []i32) []i32 = scatter(replicate(length(xs), 7i32), [2i32, 0i32], [30i32, 10i32])",
        "entry main() [4]i32 = scatter([7i32, 7i32, 7i32, 7i32], [2i32, 0i32], [30i32, 10i32])",
        "entry main() [4]i32 = spread(4, 7i32, [2i32, 0i32], [30i32, 10i32])",
        "entry main(xs: []i32) []i32 = map(|x: i32| x * 2, xs)",
        "entry main(xs: []i32) [2]i32 = [xs[0], xs[1]]",
        include_str!("../../../testfiles/filter_captures_runtime_array.wyn"),
        include_str!("../../../testfiles/reduce_over_map.wyn"),
        include_str!("../../../testfiles/scan_compute.wyn"),
        include_str!("../../../testfiles/filter_then_map.wyn"),
        "entry main(xs: [4]i32, n: i32) [4]i32 = loop acc = xs for k < n do map(|x: i32| x + k, acc)",
    ] {
        let tlc = infer_input_slice_bounds(compile_thru_tlc(source).unwrap());
        let program = optimize(
            schedule(
                place(
                    fuse(from_tlc(&tlc).unwrap()).unwrap(),
                    PipelineTopologyPolicy::AllowGenerated,
                )
                .unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
        let ssa = to_ssa(program, CodegenTarget::Spirv).unwrap();
        let output = lower_ssa_to_spirv(ssa).unwrap();
        let bytes: Vec<_> = output.spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
        let module = naga::front::spv::parse_u8_slice(&bytes, &Default::default()).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|error| panic!("{error:?}\n{source}"));
    }
}

#[test]
fn vector_input_and_runtime_gather_after_scan_reach_wgsl() {
    compile(include_str!("../../../testfiles/gather_scan_chain.wyn"));
}

#[test]
fn graphics_stages_preserve_shader_interfaces_and_draw_metadata() {
    use host::{Pipeline, ShaderStage};
    let source = include_str!("../../../testfiles/unified_triangle.wyn");
    let module = compile(source);
    assert_eq!(module.entry_points.len(), 2);
    assert!(module.entry_points.iter().any(|e| e.stage == naga::ShaderStage::Vertex));
    assert!(module.entry_points.iter().any(|e| e.stage == naga::ShaderStage::Fragment));
    let output = pipeline(source);
    let [Pipeline::Graphics(graphics)] = output.program.interface.pipelines.as_slice() else {
        panic!("one graphics pipeline");
    };
    assert_eq!(graphics.stages.len(), 2);
    assert_eq!(graphics.source_operation, Some(0));
    assert!(graphics.stages.iter().any(|s| matches!(s.stage, ShaderStage::Vertex)));
    assert!(graphics.stages.iter().any(|s| matches!(s.stage, ShaderStage::Fragment)));
    assert!(!graphics.fragment_outputs.is_empty());
    assert!(output.program.interface.source_results.is_empty());
}

fn pipeline(source: &str) -> LoweredWgsl {
    pipeline_with_grid(source, None)
}

fn pipeline_with_grid(source: &str, grid: Option<(u32, u32, u32)>) -> LoweredWgsl {
    let mut tlc = infer_input_slice_bounds(compile_thru_tlc(source).unwrap());
    for definition in &mut tlc.defs {
        if let DefMeta::EntryPoint(entry) = &mut definition.meta {
            entry.declaration.compute_dispatch = grid.map(|(x, y, z)| ComputeDispatchGrid { x, y, z });
        }
    }
    let program = optimize(
        schedule(
            place(
                fuse(from_tlc(&tlc).unwrap()).unwrap(),
                PipelineTopologyPolicy::AllowGenerated,
            )
            .unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    lower_ssa_to_wgsl_with_program(to_ssa(program, CodegenTarget::Wgsl).unwrap()).unwrap()
}

#[test]
fn runtime_input_lengths_and_output_allocations_share_the_published_bindings() {
    use host::{Binding, BufferLen, Pipeline};
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
    let [Pipeline::Compute(p)] = output.program.interface.pipelines.as_slice() else {
        panic!("compute pipeline")
    };
    assert_eq!(p.stages.len(), 1);
    assert_eq!(p.stages[0].owner, "main");
    let result = &output.program.interface.source_results[0];
    assert!(p.bindings.iter().any(|b| matches!(b, Binding::StorageBuffer { set, binding, length: Some(BufferLen::LikeInput { set: 0, binding: 0, elem_bytes: 4, src_elem_bytes: 4 }), .. } if (*set, *binding) == (result.set, result.binding))));
    assert!(p.bindings.iter().any(
        |b| matches!(b, Binding::StorageBuffer { members, .. } if members.iter().any(|m| m.name == "bias"))
    ));
    assert!(p.bindings.iter().all(|b| !matches!(b, Binding::PushConstant { .. })));
}

#[test]
fn input_length_uses_do_not_require_a_separate_dispatch() {
    let output = pipeline(
        "entry main(xs: []i32, index:i32) []i32 =
         let a=xs[index] in map(|i:i32|i+a, iota(length(xs)))",
    );
    let [Pipeline::Compute(p)] = output.program.interface.pipelines.as_slice() else {
        panic!("compute pipeline")
    };
    assert_eq!(p.stages.len(), 1, "immutable read and length stay inside the map");
}

#[test]
fn scalar_projection_setup_is_evaluated_inside_its_map() {
    let output = pipeline(include_str!("../../../testfiles/scalar_setup.wyn"));
    let [Pipeline::Compute(p)] = output.program.interface.pipelines.as_slice() else {
        panic!("compute pipeline")
    };
    assert_eq!(p.stages.len(), 1);
}

#[test]
fn reads_before_consuming_updates_remain_materialized() {
    let output = pipeline(
        "entry main(xs:*[]i32, index:i32) []i32 =
         let old=xs[index] in map(|x:i32|x+old, xs)",
    );
    let [Pipeline::Compute(p)] = output.program.interface.pipelines.as_slice() else {
        panic!("compute pipeline")
    };
    assert_eq!(
        p.stages.len(),
        2,
        "save the old value before overwriting its buffer"
    );
}

#[test]
fn reduction_publication_has_scratch_writers_readers_and_a_source_result() {
    use host::{Binding, BufferLen, Pipeline};
    let output = pipeline("entry main(xs: [137]i32) i32 = reduce(|a:i32,b:i32|a+b,0,xs)");
    let [Pipeline::Compute(p)] = output.program.interface.pipelines.as_slice() else {
        panic!("compute pipeline")
    };
    assert_eq!(p.stages.len(), 3, "chunks, combine, scalar result publication");
    let host::DispatchSize::Fixed { x, y, z, .. } = p.stages[0].dispatch_size else {
        panic!("collective chunk grid must be bounded");
    };
    let bytes = u64::from(x) * u64::from(y) * u64::from(z) * 4;
    let scratch = p
        .bindings
        .iter()
        .position(|b| {
            matches!(
                b,
                Binding::StorageBuffer {
                    length: Some(BufferLen::Fixed { bytes: capacity }),
                    ..
                } if *capacity == bytes
            )
        })
        .unwrap();
    assert!(p.stages[0].writes.contains(&scratch));
    assert!(p.stages[1].reads.contains(&scratch));
    assert!(!p.stages[1].writes.contains(&scratch));
    assert_eq!(output.program.interface.source_results.len(), 1);
    assert_eq!(output.program.interface.frame_graph.passes.len(), 3);
    assert!(output.program.interface.frame_graph.topological_order().is_ok());
}

#[test]
fn independent_entries_keep_their_own_dispatches_and_results() {
    let output = pipeline("entry first(xs:[5]i32) [5]i32 = map(|x:i32|x+1,xs)\nentry second(xs:[9]i32) [9]i32 = map(|x:i32|x*2,xs)");
    assert_eq!(output.program.interface.pipelines.len(), 2);
    assert_eq!(output.program.interface.source_results.len(), 2);
    assert_ne!(
        output.program.interface.source_results[0].pipeline_index,
        output.program.interface.source_results[1].pipeline_index
    );
    assert_ne!(
        output.program.interface.source_results[0].binding,
        output.program.interface.source_results[1].binding
    );
}

#[test]
fn scalar_results_after_collectives_are_executed_and_published() {
    let output = pipeline("entry main(xs:[]i32) i32 = reduce(|a:i32,b:i32|a+b,0,xs) * 3 + 1");
    assert_eq!(output.program.interface.source_results.len(), 1);
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
    use host::{Access, Binding, BufferUsage, Pipeline};
    let output = pipeline("entry main(dest:*[3]i32,xs:[5]i32) [3]i32 = reduce_by_index(dest,|a:i32,b:i32|a+b,0,map(|x:i32|x%3,xs),xs)");
    let result = &output.program.interface.source_results[0];
    assert_eq!((result.set, result.binding), (0, 0));
    let Pipeline::Compute(p) = &output.program.interface.pipelines[0] else {
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
fn consuming_maps_and_scans_publish_the_input_as_their_read_write_result() {
    use host::{Access, Binding, BufferUsage};
    for (body, buffers, stages) in [
        ("map(|x:i32|x+7,xs)", 1, 1),
        ("scan(|a:i32,b:i32|a+b,0,xs)", 4, 3),
    ] {
        let output = pipeline(&format!("entry main(xs:*[]i32) []i32 = {body}"));
        assert_eq!(output.program.interface.source_results[0].binding, 0);
        let Pipeline::Compute(p) = &output.program.interface.pipelines[0] else {
            panic!("compute")
        };
        assert_eq!(p.bindings.len(), buffers);
        assert_eq!(p.stages.len(), stages);
        assert!(matches!(
            p.bindings[0],
            Binding::StorageBuffer {
                usage: BufferUsage::Input,
                access: Access::ReadWrite,
                ..
            }
        ));
        let module = naga::front::wgsl::parse_str(&output.wgsl).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }
}

#[test]
fn fused_maps_publish_reused_nonprimary_inputs() {
    let output = pipeline(
        "entry main(xs:*[4]i32,ys:*[4]i32) ([4]i32,[4]i32) = (map(|x:i32|x+1,ys),map(|x:i32|x*2,xs))",
    );
    assert_eq!(
        output.program.interface.source_results.iter().map(|r| r.binding).collect::<Vec<_>>(),
        [1, 0]
    );
    let Pipeline::Compute(p) = &output.program.interface.pipelines[0] else {
        panic!("compute")
    };
    assert_eq!(p.bindings.len(), 2);
    assert!(p.stages.iter().all(|stage| stage.workgroup_size == (64, 1, 1)));
    let module = naga::front::wgsl::parse_str(&output.wgsl).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
}

#[test]
fn tuple_of_views_uses_the_tlc_component_bindings() {
    let source = "entry main(xs: ([]i32,[]i32)) ([]i32,[]i32) = xs";
    let expected = vec![0, 1];
    let output = pipeline(source);
    assert_eq!(output.program.interface.source_results.len(), 2);
    assert_eq!(
        output.program.interface.source_results.iter().map(|o| o.binding).collect::<Vec<_>>(),
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

#[test]
fn computed_graphics_captures_share_the_producer_binding() {
    use host::Binding;
    let output = pipeline(include_str!("../../../testfiles/playground/conway.wyn"));
    let [result] = output.program.interface.source_results.as_slice() else {
        panic!("one computed board");
    };
    assert!(output.program.interface.pipelines.iter().any(|p| {
        let Pipeline::Graphics(p) = p else { return false };
        p.bindings.iter().any(|b| {
            matches!(b, Binding::StorageBuffer { set, binding, .. }
            if (*set, *binding) == (result.set, result.binding))
        })
    }));
}

#[test]
fn host_sized_outputs_publish_uniform_dependencies_and_storage_stride() {
    use host::{Binding, BufferLen, Expr, ScalarSource};
    let source = include_str!("../../../testfiles/regressions/uniform_output_size.wyn")
        .replace("{ resolution: vec3f32 }", "{ padding: f32, resolution: vec3f32 }")
        .replace("([]f32,", "([]vec3f32,")
        .replace(
            "target_load(rendered, @[i % width, i / width], 0u32)",
            "@[f32(i), 0.0, 0.0]",
        );
    let output = pipeline(&source);
    let result = &output.program.interface.source_results[0];
    let Pipeline::Compute(pipeline) = &output.program.interface.pipelines[result.pipeline_index] else {
        panic!("compute")
    };
    let bytes = pipeline
        .bindings
        .iter()
        .find_map(|binding| match binding {
            Binding::StorageBuffer {
                set,
                binding,
                length: Some(BufferLen::Computed { bytes }),
                ..
            } if (*set, *binding) == (result.set, result.binding) => Some(bytes),
            _ => None,
        })
        .expect("generated output allocation");
    let Expr::Multiply(_, stride) = bytes else {
        panic!("physical stride")
    };
    assert_eq!(**stride, Expr::Integer(16));
    let mut reads = Vec::new();
    bytes.clone().reads_mut(&mut |source, offset| {
        assert!(matches!(source, ScalarSource::Binding { .. }));
        reads.push(*offset);
    });
    // Project the two needed components before emitting parameter reads. Keep
    // std140 offsets and do not touch padding or the unused resolution.z.
    assert_eq!(reads, [16, 20]);
    let whl = output.program.to_whl("uniform.wgsl", host::ShaderFormat::Wgsl).unwrap();
    assert!(whl.contains("wyn-f32-to-i32"));
}

#[test]
fn shared_helper_reuses_its_emitted_body_and_storage_requirements() {
    let output=pipeline("def load(xs:[]i32) i32=xs[0] entry first(xs:[]i32) i32=load(xs) entry second(ys:[]i32) i32=load(ys)");
    assert_eq!(output.program.interface.source_results.len(), 2);
    compile("def load(xs:[]i32) i32=xs[0] entry first(xs:[]i32) i32=load(xs) entry second(ys:[]i32) i32=load(ys)");
}

#[test]
fn runtime_launches_use_buffer_and_scalar_domains() {
    use host::{DispatchLen, DispatchSize};
    for source in [
        "entry main(xs: []i32) []i32 = map(|x:i32|x+1,xs)",
        "entry main(n: i32) []i32 = map(|i|i+1,iota(n))",
    ] {
        let output = pipeline(source);
        let [Pipeline::Compute(p)] = output.program.interface.pipelines.as_slice() else {
            panic!("compute")
        };
        assert!(matches!(
            p.stages[0].dispatch_size,
            DispatchSize::DerivedFrom {
                len: DispatchLen::InputBinding { .. } | DispatchLen::StorageBuffer { .. },
                workgroup_size: 64
            }
        ));
    }
}

#[test]
fn unsigned_ranges_keep_their_extent_in_the_element_representation() {
    let source = "entry main(n:u32) []u32 = map(|i:u32|i+1u32, 0u32..<n)";
    compile(source);
    let tlc = infer_input_slice_bounds(compile_thru_tlc(source).unwrap());
    let program = optimize(
        schedule(
            place(
                fuse(from_tlc(&tlc).unwrap()).unwrap(),
                PipelineTopologyPolicy::AllowGenerated,
            )
            .unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    let output = lower_ssa_to_spirv(to_ssa(program, CodegenTarget::Spirv).unwrap()).unwrap();
    let bytes: Vec<_> = output.spirv.iter().flat_map(|word| word.to_le_bytes()).collect();
    let module = naga::front::spv::parse_u8_slice(&bytes, &Default::default()).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
}

#[test]
fn runtime_collective_scratch_matches_its_bounded_chunk_grid() {
    use host::{Binding, BufferLen, DispatchSize};
    let output = pipeline("entry main(xs: []i32) []i32 = scan(|a:i32,b:i32|a+b,0,xs)");
    let [Pipeline::Compute(p)] = output.program.interface.pipelines.as_slice() else {
        panic!("compute")
    };
    let DispatchSize::Fixed { x, y, z, .. } = p.stages[0].dispatch_size else {
        panic!("collective chunk grid must be bounded");
    };
    let bytes = u64::from(x) * u64::from(y) * u64::from(z) * 4;
    assert_eq!(
        p.bindings
            .iter()
            .filter(|binding| matches!(binding,
                Binding::StorageBuffer { length: Some(BufferLen::Fixed { bytes: capacity }), .. }
                    if *capacity == bytes
            ))
            .count(),
        2,
        "one partial and one carry per chunk"
    );
}

#[test]
fn bounded_filter_output_does_not_use_its_packed_backing_as_a_length() {
    use host::{Binding, BufferLen};
    let output = pipeline(include_str!("../../../testfiles/filter_then_map.wyn"));
    let result = &output.program.interface.source_results[0];
    let Pipeline::Compute(p) = &output.program.interface.pipelines[result.pipeline_index] else {
        panic!("compute")
    };
    assert!(p.bindings.iter().any(|b| matches!(b, Binding::StorageBuffer {
        set, binding, length: Some(BufferLen::Fixed { bytes: 16384 }), ..
    } if (*set, *binding) == (result.set, result.binding))));
}

#[test]
fn explicit_grids_preserve_all_axes_in_the_shader_and_descriptor() {
    use host::DispatchSize;
    for source in [
        "entry main(xs:[4096]i32) [4096]i32 = map(|x:i32|x+1,xs)",
        "entry main(xs:[4096]i32) i32 = reduce(|a:i32,b:i32|a+b,0,xs)",
        "entry main(xs:[4096]i32) [4096]i32 = scan(|a:i32,b:i32|a+b,0,xs)",
    ] {
        for (x, y, z) in [(1, 1, 1), (2, 3, 4), (257, 1, 1), (3, 7, 17)] {
            let output = pipeline_with_grid(source, Some((x, y, z)));
            let [Pipeline::Compute(p)] = output.program.interface.pipelines.as_slice() else {
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
}

#[test]
fn direct_mode_keeps_collectives_in_the_authored_entry() {
    for source in [
        "entry main(xs:[137]i32) i32 = reduce(|a:i32,b:i32|a+b,0,xs)",
        "entry main(xs:[137]i32) [137]i32 = map(|x:i32|x+1,xs)",
        include_str!("../../../testfiles/unified_triangle.wyn"),
    ] {
        let tlc = infer_input_slice_bounds(compile_thru_tlc(source).unwrap());
        let scheduled = optimize(
            schedule(
                place(
                    fuse(from_tlc(&tlc).unwrap()).unwrap(),
                    PipelineTopologyPolicy::AuthoredOnly,
                )
                .unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
        let output =
            lower_ssa_to_wgsl_with_program(to_ssa(scheduled, CodegenTarget::Wgsl).unwrap()).unwrap();
        for pipeline in &output.program.interface.pipelines {
            if let Pipeline::Compute(p) = pipeline {
                assert_eq!(p.stages.len(), 1);
            }
        }
        let module = naga::front::wgsl::parse_str(&output.wgsl).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }
}

#[test]
fn authored_outputs_consume_selected_capacities() {
    use host::{Binding, BufferLen};
    for (source, expected) in [
        (
            "entry main(xs:[5]vec3f32) [5]vec3f32=xs",
            BufferLen::Fixed { bytes: 80 },
        ),
        (
            "entry main(xs:[]i32) []i32=xs",
            BufferLen::LikeInput {
                set: 0,
                binding: 0,
                elem_bytes: 4,
                src_elem_bytes: 4,
            },
        ),
        (
            "entry main(xs:[]vec3f32,ys:[]i32) []i32=ys",
            BufferLen::LikeInput {
                set: 0,
                binding: 1,
                elem_bytes: 4,
                src_elem_bytes: 4,
            },
        ),
        (
            "entry main(n:i32) []i32=iota(n)",
            BufferLen::SameAsDispatch { elem_bytes: 4 },
        ),
    ] {
        let tlc = infer_input_slice_bounds(compile_thru_tlc(source).unwrap());
        let program = optimize(
            schedule(
                place(
                    fuse(from_tlc(&tlc).unwrap()).unwrap(),
                    PipelineTopologyPolicy::AuthoredOnly,
                )
                .unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
        let output = lower_ssa_to_wgsl_with_program(to_ssa(program, CodegenTarget::Wgsl).unwrap()).unwrap();
        assert!(
            output.program.interface.pipelines.iter().any(|pipeline| {
                let Pipeline::Compute(pipeline) = pipeline else {
                    return false;
                };
                pipeline.bindings.iter().any(|binding| {
                    matches!(binding,
                Binding::StorageBuffer { usage: host::BufferUsage::Output, length: Some(actual), .. }
                if *actual == expected)
                })
            }),
            "{source}: expected {expected:?}"
        );
    }
}

#[test]
fn missing_authored_output_capacity_is_an_error() {
    let tlc = infer_input_slice_bounds(compile_thru_tlc("entry main() i32=42").unwrap());
    let mut program = optimize(
        schedule(
            place(
                fuse(from_tlc(&tlc).unwrap()).unwrap(),
                PipelineTopologyPolicy::AuthoredOnly,
            )
            .unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    program.graph.clear_function("SelectedOutputCapacity").unwrap();
    let Err(error) = to_ssa(program, CodegenTarget::Wgsl) else {
        panic!("missing selected capacity was silently accepted");
    };
    assert!(
        error.to_string().contains("missing selected SelectedOutputCapacity"),
        "{error}"
    );
}

#[test]
fn materialized_tuple_projections_reach_both_backends() {
    let source = include_str!("../../../testfiles/regressions/fusion_shared_tuple.wyn");
    compile(source);
    let output = crate::compile_thru_spirv(source).unwrap();
    let outputs = output
        .program
        .interface
        .pipelines
        .iter()
        .flat_map(|pipeline| match pipeline {
            Pipeline::Compute(compute) => compute.bindings.iter(),
            _ => panic!("expected a compute pipeline"),
        })
        .filter_map(|binding| match binding {
            crate::host::interface::Binding::StorageBuffer {
                usage: crate::host::interface::BufferUsage::Output,
                length,
                ..
            } => Some(length),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(outputs.len(), 2);
    assert!(outputs.iter().all(|length| matches!(
        length,
        Some(crate::host::interface::BufferLen::Fixed { bytes: 64 })
    )));
    let bytes: Vec<_> = output.spirv.iter().flat_map(|word| word.to_le_bytes()).collect();
    let module = naga::front::spv::parse_u8_slice(&bytes, &Default::default()).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
}

#[test]
fn aggregate_forwarding_preserves_record_subsets_and_loop_state() {
    let source = include_str!("../../../testfiles/regressions/aggregate_forwarding.wyn");
    compile(source);
    let output = crate::compile_thru_spirv(source).unwrap();
    let bytes: Vec<_> = output.spirv.iter().flat_map(|word| word.to_le_bytes()).collect();
    let module = naga::front::spv::parse_u8_slice(&bytes, &Default::default()).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    let module = wspirv::dr::load_words(&output.spirv).unwrap();
    let definitions: std::collections::HashMap<_, _> = module
        .all_inst_iter()
        .filter_map(|instruction| instruction.result_id.map(|id| (id, instruction)))
        .collect();
    for function in &module.functions {
        for block in &function.blocks {
            let mut seen = std::collections::HashSet::new();
            for instruction in &block.instructions {
                use wspirv::{dr::Operand, spirv::Op};
                if matches!(
                    instruction.class.opcode,
                    Op::CompositeConstruct | Op::CompositeExtract
                ) {
                    assert!(
                        seen.insert((
                            instruction.class.opcode,
                            instruction.result_type,
                            &instruction.operands
                        )),
                        "duplicate aggregate operation"
                    );
                }
                if instruction.class.opcode == Op::CompositeExtract {
                    let Operand::IdRef(source) = instruction.operands[0] else {
                        panic!("missing source")
                    };
                    assert_ne!(definitions[&source].class.opcode, Op::CompositeConstruct);
                }
            }
        }
    }
}

#[test]
fn materialized_boolean_reduction_reaches_both_backends() {
    for condition in ["xs[0] > 0 || any", "any || xs[0] > 0", "any", "!any"] {
        let source = format!(
            "entry repro(xs: []i32) []i32 =
              let any = reduce(|a,b| a || b,false,map(|x| x != 0,xs)) in
              [if {condition} then 1i32 else 0i32]"
        );
        compile(&source);
        let output = crate::compile_thru_spirv(&source).unwrap();
        let bytes: Vec<_> = output.spirv.iter().flat_map(|word| word.to_le_bytes()).collect();
        let module = naga::front::spv::parse_u8_slice(&bytes, &Default::default()).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }
}

#[test]
fn shared_fused_helpers_reach_valid_wgsl() {
    compile(include_str!(
        "../../../testfiles/regressions/shared_fusion_helper.wyn"
    ));
}

#[test]
fn runtime_index_into_shared_array_helper_reaches_wgsl() {
    compile(
        "def g(n: i32) []f32 = map(|i: i32| f32.i32(i),0i32..<n)\nentry e(j: i32) [1]f32 = [g(256)[j]]",
    );
}

#[test]
fn workgroup_storage_is_local_to_each_entry() {
    let source = "entry ints(xs:[]i32) i32 = reduce(|a:i32,b:i32|a+b,0,xs)
                  entry floats(xs:[]f32) f32 = reduce(|a:f32,b:f32|a+b,0.0,xs)";
    compile(source);
    let output = compile_thru_spirv(source).unwrap();
    let bytes: Vec<_> = output.spirv.iter().flat_map(|word| word.to_le_bytes()).collect();
    let module = spv::parse_u8_slice(&bytes, &Default::default()).unwrap();
    Validator::new(ValidationFlags::all(), Capabilities::all()).validate(&module).unwrap();
}

#[test]
fn large_fixed_domains_cap_generated_dispatches() {
    let source = "entry main(xs:[25000000]i32) [25000000]i32 = map(|x:i32|x+1,xs)";
    let output = pipeline(source);
    let [Pipeline::Compute(p)] = output.program.interface.pipelines.as_slice() else {
        panic!("one compute pipeline")
    };
    assert_eq!(p.stages.len(), 1);
    assert_eq!(
        p.stages[0].dispatch_size,
        host::DispatchSize::Fixed {
            x: 65_535,
            y: 1,
            z: 1,
            explicit: true,
        }
    );
    compile(source);
}

#[test]
fn explicit_grids_do_not_replicate_single_workgroup_compaction() {
    let output = pipeline_with_grid(
        "entry main(xs:[]i32) []i32 = filter(|x:i32|x>0,xs)",
        Some((2, 3, 4)),
    );
    let [Pipeline::Compute(p)] = output.program.interface.pipelines.as_slice() else {
        panic!("compute");
    };
    assert_eq!(p.stages.len(), 1);
    assert_eq!(
        p.stages[0].dispatch_size,
        host::DispatchSize::Fixed {
            x: 1,
            y: 1,
            z: 1,
            explicit: true
        }
    );
    let module = naga::front::wgsl::parse_str(&output.wgsl).unwrap();
    Validator::new(ValidationFlags::all(), Capabilities::all()).validate(&module).unwrap();
}

#[test]
fn collective_phase_accesses_exclude_unused_inputs_and_producer_captures() {
    let output =
        pipeline("entry main(xs:[]i32,unused:[]i32,bias:i32) i32=reduce((+),0,map(|x|x+bias*bias,xs))");
    let [Pipeline::Compute(p)] = output.program.interface.pipelines.as_slice() else {
        panic!("compute pipeline");
    };
    let unused = p
        .bindings
        .iter()
        .position(
            |binding| matches!(binding, host::Binding::StorageBuffer { name, .. } if name == "unused"),
        )
        .unwrap();
    assert!(p.stages.iter().all(|stage| stage.uses.access(unused).is_none()));
    let combine = p.stages.iter().find(|stage| stage.entry_point.ends_with("combine")).unwrap();
    assert!(output.program.interface.scalar_tasks.iter().all(|task| task.stage != combine.entry_point));
    assert!(!output.program.interface.scalar_tasks.is_empty());
}
