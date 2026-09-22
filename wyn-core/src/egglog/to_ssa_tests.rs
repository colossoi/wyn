use crate::egglog::{
    from_tlc, fuse, insert_expressions, schedule, simplify_and_place, to_ssa, Fused, OperationKind,
    Program, Value,
};
use crate::host::Pipeline;
use crate::interface::EntryParamBindingKind;
use crate::tlc::infer_input_slice_bounds;
use crate::PipelineTopologyPolicy;
use crate::{
    compile_thru_tlc, host, lower_ssa_to_spirv, lower_ssa_to_wgsl, lower_ssa_to_wgsl_with_program,
    CodegenTarget, LoweredWgsl,
};

fn compile(source: &str) -> naga::Module {
    let tlc = infer_input_slice_bounds(compile_thru_tlc(source).unwrap());
    let program = schedule(
        simplify_and_place(insert_expressions(fuse(from_tlc(&tlc).unwrap()).unwrap()).unwrap()).unwrap(),
        PipelineTopologyPolicy::AllowGenerated,
    )
    .unwrap();
    let ssa = to_ssa(&program, CodegenTarget::Wgsl).unwrap_or_else(|error| panic!("{error}\n{source}"));
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
fn length_only_filter_view_lowers_without_its_discarded_element_buffer() {
    let tlc = infer_input_slice_bounds(
        compile_thru_tlc("entry main(xs:[]i32) i32=length(filter(|x:i32|x>0,xs))").unwrap(),
    );
    // Isolate scheduling: normal fusion replaces this filter with a count reduction.
    let program = Program {
        ir: from_tlc(&tlc).unwrap().ir,
        state: Fused,
    };
    let program = schedule(
        simplify_and_place(insert_expressions(program).unwrap()).unwrap(),
        PipelineTopologyPolicy::AllowGenerated,
    )
    .unwrap();
    let (&filter, _) =
        program.operations.iter().find(|(_, op)| matches!(op.kind, OperationKind::Filter { .. })).unwrap();
    assert!(
        matches!(&program.state.materialized[&filter], Value::Primitive("slice", args) if matches!(args[0], Value::Discarded))
    );
    let source = lower_ssa_to_wgsl(to_ssa(&program, CodegenTarget::Wgsl).unwrap()).unwrap();
    let module = naga::front::wgsl::parse_str(&source).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    lower_ssa_to_spirv(to_ssa(&program, CodegenTarget::Spirv).unwrap()).unwrap();
}

#[test]
fn length_queries_lower_without_materializing_array_elements() {
    compile(include_str!("../../../testfiles/length_metadata.wyn"));
}

#[test]
fn filter_post_map_record_outputs_lower_in_three_filter_phases() {
    let module = compile(include_str!("../../../testfiles/rust_host_filter_post.wyn"));
    let entries: Vec<_> = module.entry_points.iter().map(|entry| entry.name.as_str()).collect();
    assert_eq!(
        entries,
        [
            "post_mapped_local_offsets",
            "post_mapped_offsets",
            "post_mapped_compact",
            "post_mapped_finish"
        ]
    );
}

pub(super) fn assert_ssa_dominance<Tag>(
    phase: &str,
    program: &crate::ssa::Program<Tag, crate::ssa::context::BackendGlobal>,
) {
    for body in program
        .functions
        .iter()
        .map(|function| &function.body)
        .chain(program.entry_points.iter().map(|entry| &entry.body))
    {
        let dominators = wyn_graph::DominatorTree::build(body.inner.entry, |block, successors| {
            successors.extend(body.inner.blocks[block].term.successors())
        });
        for node in body.inner.insts.values() {
            let Some(parent) = node.placement.block() else {
                panic!("{phase}: floating instruction remained after placement")
            };
            if !dominators.is_reachable(parent) {
                continue;
            }
            for value in node.data.ssa_uses() {
                let Some(producer) = body.inner.block_of_value(value) else {
                    panic!("{phase}: operand {value:?} remained floating")
                };
                assert!(
                    dominators.dominates(producer, parent),
                    "{phase}: {value:?} in {producer:?} used by {parent:?}; producer {:?}; consumer {node:?}",
                    body.inner.inst_of_value(value).map(|instruction| &body.inner.insts[instruction])
                );
            }
        }
    }
}

#[test]
fn literal_and_runtime_nested_array_indices_reach_wgsl() {
    compile(include_str!("../../../testfiles/composite_2d_local.wyn"));
}

#[test]
fn fixed_output_lengths_do_not_emit_element_reads_or_array_constructions() {
    use crate::op::OpTag;
    use crate::ssa::types::InstKind;

    for (array, expected) in [("[n]", 1), ("if flag then [n] else [n + 1]", 2)] {
        let program = crate::compile_thru_ssa(&format!(
            "entry repro(xs: []i32, flag: bool) ([]i32, [1]i32) =
              let n = xs[0] in (map(|x| x + n, xs), {array})",
        ))
        .unwrap();
        // Inspect the emitted SSA before optimization, placement, or dead-code
        // elimination can conceal unnecessary work in the output copy kernel.
        let output = program
            .entry_points
            .iter()
            .find(|entry| {
                entry.body.inner.insts.values().any(|node| {
                    matches!(
                        node.data,
                        InstKind::Op {
                            tag: OpTag::ArrayLit(_),
                            ..
                        }
                    )
                })
            })
            .unwrap();
        let instructions = &output.body.inner.insts;
        assert_eq!(
            instructions
                .values()
                .filter(|node| {
                    matches!(
                        node.data,
                        InstKind::Op {
                            tag: OpTag::ArrayLit(_),
                            ..
                        }
                    )
                })
                .count(),
            expected,
            "only the indexed output value needs array constructions: {array}"
        );
        let calls = instructions
            .values()
            .filter(|node| {
                matches!(
                    node.data,
                    InstKind::Op {
                        tag: OpTag::Call(_),
                        ..
                    }
                )
            })
            .count();
        assert_eq!(
            instructions.values().filter(|node| matches!(node.data, InstKind::Load { .. })).count() + calls,
            expected,
            "length queries must not reload the captured element: {array}"
        );
    }
}

#[test]
fn nested_mountain_shader_preserves_ssa_dominance() {
    std::thread::Builder::new()
        .stack_size(32 * 1024 * 1024)
        .spawn(|| {
            let source = include_str!("../../../testfiles/playground/mountains.wyn");
            let tlc = infer_input_slice_bounds(compile_thru_tlc(source).unwrap());
            let program = schedule(
                simplify_and_place(insert_expressions(fuse(from_tlc(&tlc).unwrap()).unwrap()).unwrap())
                    .unwrap(),
                PipelineTopologyPolicy::AllowGenerated,
            )
            .unwrap();
            let ssa = to_ssa(&program, CodegenTarget::Wgsl).unwrap();
            assert_ssa_dominance("before", &ssa);
            lower_ssa_to_wgsl(ssa.clone()).unwrap();
            let placed = crate::ssa::place_floating(crate::ssa::optimize(ssa)).unwrap();
            assert_ssa_dominance("after", &placed);
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
        for p in &output.program.interface.pipelines {
            let Pipeline::Compute(p) = p else { continue };
            for binding in &p.bindings {
                if let host::Binding::StorageBuffer {
                    length: Some(host::BufferLen::HostProvided { inputs, .. }),
                    ..
                } = binding
                {
                    assert!(!inputs.is_empty());
                    assert!(inputs.iter().all(|i| matches!(i, host::HostSizeInput::Uniform { .. })));
                }
            }
        }
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
    let program = schedule(
        simplify_and_place(insert_expressions(fuse(from_tlc(&tlc).unwrap()).unwrap()).unwrap()).unwrap(),
        PipelineTopologyPolicy::AllowGenerated,
    )
    .unwrap();
    let ssa = to_ssa(&program, CodegenTarget::Spirv).unwrap();
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
    assert_eq!(filter.entry_points.len(), 3);
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
fn existing_spirv_backend_also_accepts_the_handoff() {
    for source in [
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
        let program = schedule(
            simplify_and_place(insert_expressions(fuse(from_tlc(&tlc).unwrap()).unwrap()).unwrap())
                .unwrap(),
            PipelineTopologyPolicy::AllowGenerated,
        )
        .unwrap();
        let ssa = to_ssa(&program, CodegenTarget::Spirv).unwrap();
        let output = lower_ssa_to_spirv(ssa).unwrap();
        let bytes: Vec<_> = output.spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
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
    let tlc = infer_input_slice_bounds(compile_thru_tlc(source).unwrap());
    let program = schedule(
        simplify_and_place(insert_expressions(fuse(from_tlc(&tlc).unwrap()).unwrap()).unwrap()).unwrap(),
        PipelineTopologyPolicy::AllowGenerated,
    )
    .unwrap();
    lower_ssa_to_wgsl_with_program(to_ssa(&program, CodegenTarget::Wgsl).unwrap()).unwrap()
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
fn reduction_publication_has_scratch_writers_readers_and_a_source_result() {
    use host::{Binding, BufferLen, Pipeline};
    let output = pipeline("entry main(xs: [137]i32) i32 = reduce(|a:i32,b:i32|a+b,0,xs)");
    let [Pipeline::Compute(p)] = output.program.interface.pipelines.as_slice() else {
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
    assert_eq!(p.stages.len(), 1);
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
    let tlc = infer_input_slice_bounds(compile_thru_tlc(source).unwrap());
    let imported = from_tlc(&tlc).unwrap();
    let expected: Vec<_> = imported
        .ir
        .entry_params
        .values()
        .filter_map(|p| p.binding.as_ref())
        .flat_map(|p| match &p.kind {
            EntryParamBindingKind::Single { binding, .. } => vec![binding.binding],
            EntryParamBindingKind::TupleOfViews(fields) => {
                fields.iter().map(|f| f.binding.binding).collect()
            }
        })
        .collect();
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
    use host::{Binding, BufferLen, HostSizeScalar};
    let source = include_str!("../../../testfiles/regressions/uniform_output_size.wyn")
        .replace("{ resolution: vec3f32 }", "{ padding: f32, resolution: vec3f32 }")
        .replace("([]f32,", "([]vec3f32,")
        .replace(
            "target_load(rendered, @[i % width, i / width], 0u32)",
            "@[f32(i), 0.0, 0.0]",
        );
    let output = pipeline(&source);
    let lengths: Vec<_> = output
        .program
        .interface
        .pipelines
        .iter()
        .filter_map(|p| {
            let Pipeline::Compute(p) = p else { return None };
            p.bindings.iter().find_map(|b| match b {
                Binding::StorageBuffer {
                    length: Some(BufferLen::HostProvided { inputs, elem_bytes }),
                    ..
                } => Some((inputs, *elem_bytes)),
                _ => None,
            })
        })
        .collect();
    assert_eq!(lengths.len(), 1);
    let (inputs, stride) = lengths[0];
    assert_eq!(stride, 16);
    assert_eq!(
        inputs
            .iter()
            .map(|i| {
                let crate::host::HostSizeInput::Uniform {
                    name, offset, scalar, ..
                } = i
                else {
                    panic!("uniform input");
                };
                (name.as_str(), *offset, *scalar)
            })
            .collect::<Vec<_>>(),
        [
            ("frame_resolution_x", 16, HostSizeScalar::F32),
            ("frame_resolution_y", 20, HostSizeScalar::F32)
        ]
    );
}

#[test]
fn shared_helper_reuses_its_emitted_body_and_storage_requirements() {
    use crate::egglog::blocks::{BufferData, Function, FunctionKind, Storage, Value};
    use crate::egglog::{Program, Scheduled};
    use crate::interface::{StorageBindingDecl, StorageRole};
    let mut data = Program {
        ir: Default::default(),
        state: Scheduled::default(),
    };
    let buffer = data.state.buffers.alloc(BufferData {
        name: "shared".into(),
        length: Value::Int(1),
        element: crate::types::i32(),
        storage: Storage::Device,
    });
    data.state.abi.bindings.insert(
        buffer,
        StorageBindingDecl {
            binding: crate::BindingRef::new(0, 0),
            elem_ty: crate::types::i32(),
            role: StorageRole::Input,
            logical_resource: None,
            length: None,
        },
    );
    let root = Function {
        name: "load".into(),
        kind: FunctionKind::Device,
        results: 1,
        blocks: vec![],
    }
    .insert(vec![], &mut data.state.blocks, &mut data.state.bodies);
    let crate::egglog::Exit::Return(returns) = data.state.blocks[root].exit else {
        panic!("return")
    };
    data.state.bodies[returns].results.push(Value::op("index", [Value::Buffer(buffer), Value::Int(0)]));
    let mut compiler = super::Compiler {
        host: Default::default(),
        origins: Default::default(),
        placements: Default::default(),
        data: &data,
        functions: vec![],
        externs: Default::default(),
        specializations: Default::default(),
        active: Default::default(),
        used: Default::default(),
    };
    let first = compiler.function(root, vec![]).unwrap();
    assert_eq!(
        compiler.functions[first.0 as usize].body.return_ty,
        crate::types::i32()
    );
    assert!(compiler.used.contains(&buffer));
    compiler.used.clear();
    assert_eq!(compiler.function(root, vec![]).unwrap(), first);
    assert_eq!(compiler.functions.len(), 1);
    assert!(
        compiler.used.contains(&buffer),
        "cached helper still needs its storage in each shader"
    );
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
    let program = schedule(
        simplify_and_place(insert_expressions(fuse(from_tlc(&tlc).unwrap()).unwrap()).unwrap()).unwrap(),
        PipelineTopologyPolicy::AllowGenerated,
    )
    .unwrap();
    let output = lower_ssa_to_spirv(to_ssa(&program, CodegenTarget::Spirv).unwrap()).unwrap();
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
fn runtime_collective_scratch_has_an_input_capacity_and_chunk_grid() {
    use host::{Binding, BufferLen, DispatchLen, DispatchSize};
    let output = pipeline("entry main(xs: []i32) []i32 = scan(|a:i32,b:i32|a+b,0,xs)");
    let [Pipeline::Compute(p)] = output.program.interface.pipelines.as_slice() else {
        panic!("compute")
    };
    assert!(matches!(
        p.stages[0].dispatch_size,
        DispatchSize::DerivedFrom {
            len: DispatchLen::InputBinding { elem_bytes: 4, .. },
            workgroup_size: 4096
        }
    ));
    assert!(p.bindings.iter().all(|b| !matches!(
        b,
        Binding::StorageBuffer {
            length: Some(BufferLen::HostProvided { .. }),
            ..
        }
    )));
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
    use crate::interface::ComputeDispatchGrid;
    use host::DispatchSize;
    for source in [
        "entry main(xs:[4096]i32) [4096]i32 = map(|x:i32|x+1,xs)",
        "entry main(xs:[4096]i32) i32 = reduce(|a:i32,b:i32|a+b,0,xs)",
    ] {
        for (x, y, z) in [(1, 1, 1), (2, 3, 4)] {
            let tlc = infer_input_slice_bounds(compile_thru_tlc(source).unwrap());
            let mut imported = from_tlc(&tlc).unwrap();
            for (_, entry) in &mut imported.ir.entries {
                entry.declaration.compute_dispatch = Some(ComputeDispatchGrid { x, y, z });
            }
            let scheduled = schedule(
                simplify_and_place(insert_expressions(fuse(imported).unwrap()).unwrap()).unwrap(),
                PipelineTopologyPolicy::AllowGenerated,
            )
            .unwrap();
            let output =
                lower_ssa_to_wgsl_with_program(to_ssa(&scheduled, CodegenTarget::Wgsl).unwrap()).unwrap();
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
                assert_eq!(
                    stage.dispatch_size,
                    DispatchSize::Fixed {
                        x: 1,
                        y: 1,
                        z: 1,
                        explicit: true
                    }
                );
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
        let scheduled = schedule(
            simplify_and_place(insert_expressions(fuse(from_tlc(&tlc).unwrap()).unwrap()).unwrap())
                .unwrap(),
            PipelineTopologyPolicy::AuthoredOnly,
        )
        .unwrap();
        assert!(scheduled.state.dispatches.is_empty());
        let output =
            lower_ssa_to_wgsl_with_program(to_ssa(&scheduled, CodegenTarget::Wgsl).unwrap()).unwrap();
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
