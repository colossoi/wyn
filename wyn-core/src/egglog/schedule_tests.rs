use super::super::{
    from_tlc, fuse, insert_expressions, simplify_and_place, Exit, FunctionKind, Instruction, Program,
    Storage,
};
use super::schedule;
use super::validation::validate;
use crate::compile_thru_tlc;
use crate::egglog::data::OperationKind;
use crate::egglog::Scheduled;
use crate::tlc::infer_input_slice_bounds;
use crate::PipelineTopologyPolicy;
use exec::{run, Value};

#[path = "schedule_test_exec.rs"]
mod exec;

fn compile(source: &str) -> Program<Scheduled> {
    let tlc = infer_input_slice_bounds(compile_thru_tlc(source).unwrap());
    let result = schedule(
        simplify_and_place(insert_expressions(fuse(from_tlc(&tlc).unwrap()).unwrap()).unwrap()).unwrap(),
        PipelineTopologyPolicy::AllowGenerated,
    )
    .unwrap();
    result
}

fn kernel_count(data: &Program<Scheduled>) -> usize {
    data.state
        .blocks
        .values()
        .filter(|b| b.interface.as_ref().is_some_and(|f| matches!(f.kind, FunctionKind::Kernel(_))))
        .count()
}

#[test]
fn consuming_fused_maps_reuse_the_input_without_allocating() {
    let result = compile("entry main(xs:*[]i32) []i32 = let a=map(|x:i32|x+7,xs) in map(|x:i32|x*2,a)");
    assert_eq!(kernel_count(&result), 1);
    assert!(result.state.buffers.values().all(|b| b.storage != Storage::Device));
    assert_eq!(
        result.state.buffers.len(),
        1,
        "only the input resource needs a buffer record"
    );
    for n in [0, 1, 63, 64, 65, 137] {
        let input = Value::array(0..n);
        let output = run(&result, vec![input.clone()]);
        let expected: Vec<_> = (0..n).map(|x| (x + 7) * 2).collect();
        assert_eq!(output[0].ints(), expected);
        assert_eq!(input.ints(), expected);
    }
}

#[test]
fn fused_maps_remap_reuse_slots_and_preserve_return_order() {
    for (returns, order) in [("(a,c)", [0, 1]), ("(c,a)", [1, 0])] {
        let result = compile(&format!("entry main(xs:*[4]i32,ys:*[4]i32) ([4]i32,[4]i32) = let a=map(|x:i32|x+1,xs) in let b=map(|x:i32|x*2,ys) in let c=map(|x:i32|x+7,b) in {returns}"));
        assert_eq!(kernel_count(&result), 1);
        assert!(result.state.buffers.values().all(|b| b.storage != Storage::Device));
        assert_eq!(
            result.state.buffers.len(),
            2,
            "reused outputs must not create buffer records"
        );
        let xs = Value::array(0..4);
        let ys = Value::array(10..14);
        let output = run(&result, vec![xs.clone(), ys.clone()]);
        let expected = [vec![1, 2, 3, 4], vec![27, 29, 31, 33]];
        assert_eq!(xs.ints(), expected[0]);
        assert_eq!(ys.ints(), expected[1]);
        assert_eq!(
            output,
            [Value::Tuple(
                order.into_iter().map(|i| Value::array(expected[i].iter().copied())).collect()
            )]
        );
    }
}

#[test]
fn retained_fused_outputs_have_distinct_storage() {
    let result = compile("entry main(xs:*[4]i32,ys:*[4]i32) ([4]i32,[4]i32,[4]i32) = let a=map(|x:i32|x+1,xs) in let b=map(|x:i32|x*2,ys) in let c=map(|x:i32|x+7,b) in (c,a,b)");
    assert_eq!(kernel_count(&result), 1);
    assert_eq!(
        result.state.buffers.values().filter(|b| b.storage == Storage::Device).count(),
        1
    );
    let output = run(&result, vec![Value::array(0..4), Value::array(10..14)]);
    assert_eq!(
        output,
        [Value::Tuple(vec![
            Value::array([27, 29, 31, 33]),
            Value::array([1, 2, 3, 4]),
            Value::array([20, 22, 24, 26])
        ])]
    );
}

#[test]
fn map_reuses_a_compiler_owned_array_after_its_reduction_finishes() {
    let result = compile("entry main(xs: []i32) []i32 = let a=map(|x:i32|x*2,xs) in let s=reduce(|x:i32,y:i32|x+y,0,a) in map(|x:i32|x+s,a)");
    assert_eq!(kernel_count(&result), 3);
    assert_eq!(
        result.state.buffers.values().filter(|b| b.storage == Storage::Device).count(),
        3
    );
    for n in [0, 1, 63, 64, 65, 137] {
        let input = Value::array(0..n);
        let output = run(&result, vec![input.clone()]);
        let sum = n * (n - 1);
        assert_eq!(output[0].ints(), (0..n).map(|x| x * 2 + sum).collect::<Vec<_>>());
        assert_eq!(
            input.ints(),
            (0..n).collect::<Vec<_>>(),
            "borrowed input must survive"
        );
    }
}

#[test]
fn fused_map_chain_becomes_one_guarded_kernel_with_explicit_captures() {
    let result = compile("entry main(xs: []i32, bias: i32) []i32 = let a = map(|x: i32| x + bias, xs) in map(|x: i32| x * 2, a)");
    assert_eq!(kernel_count(&result), 1);
    assert_eq!(result.state.dispatches.len(), 1);
    for n in [0, 1, 63, 64, 65, 137] {
        let input: Vec<_> = (0..n).collect();
        let output = run(&result, vec![Value::array(input.iter().copied()), Value::Int(7)]);
        assert_eq!(
            output[0].ints(),
            input.iter().map(|x| (x + 7) * 2).collect::<Vec<_>>()
        );
    }
}

#[test]
fn reduction_handles_empty_tail_chunks_and_tuple_accumulators() {
    let result = compile("entry main(xs: []i32) (i32, i32) = reduce(|a: (i32, i32), b: (i32, i32)| (a.0 + b.0, a.1 + b.1), (0, 0), map(|x: i32| (x, 1), xs))");
    assert_eq!(kernel_count(&result), 2);
    assert!(matches!(
        result.state.physical_kernels.kernels().next().unwrap().domain,
        crate::kernel_graph::KernelDomain::ChunkedElements { chunk_size: 64, .. }
    ));
    for n in [0, 1, 63, 64, 65, 137, 4097] {
        let output = run(&result, vec![Value::array(0..n)]);
        assert_eq!(
            output[0],
            Value::Tuple(vec![Value::Int(n * (n - 1) / 2), Value::Int(n)])
        );
    }
}

#[test]
fn scan_and_filter_have_global_dispatch_boundaries_and_correct_results() {
    let scan = compile("entry main(xs: []i32) []i32 = scan(|a: i32, b: i32| a + b, 0, xs)");
    let filter = compile("entry main(xs: []i32) ?k. [k]i32 = filter(|x: i32| x % 3 == 1, xs)");
    assert_eq!(kernel_count(&scan), 3);
    assert_eq!(kernel_count(&filter), 4);
    for (result, names) in [
        (&scan, &["chunks", "combine", "offsets"][..]),
        (&filter, &["flags", "local_offsets", "offsets", "compact"][..]),
    ] {
        let stages: std::collections::BTreeMap<_, _> = result
            .state
            .dispatches
            .iter()
            .map(|(&id, d)| {
                (
                    result.state.blocks[d.kernel].interface.as_ref().unwrap().name.as_str(),
                    id,
                )
            })
            .collect();
        for pair in names.windows(2) {
            assert!(result.state.dispatches[stages[pair[1]]].dependencies.contains(&stages[pair[0]]));
        }
    }
    for n in [0, 1, 63, 64, 65, 137] {
        let output = run(&scan, vec![Value::array(0..n)]);
        assert_eq!(
            output[0].ints(),
            (0..n).map(|i| i * (i + 1) / 2).collect::<Vec<_>>()
        );
        let output = run(&filter, vec![Value::array(0..n)]);
        assert_eq!(
            output[0].ints(),
            (0..n).filter(|i| i % 3 == 1).collect::<Vec<_>>()
        );
    }
}

#[test]
fn conditional_dispatches_execute_only_the_selected_arm() {
    // Different collective shapes prevent TLC's if-over-map normalization.
    let result = compile("entry main(xs: []i32, flag: bool) []i32 = if flag then scan(|a: i32, b: i32| a + b, 0, xs) else map(|x: i32| x * 3, xs)");
    assert_eq!(kernel_count(&result), 4);
    assert_eq!(
        result.state.dispatches.values().filter(|d| d.dependencies.is_empty()).count(),
        2
    );
    for flag in [false, true] {
        let output = run(&result, vec![Value::array(0..5), Value::Bool(flag)]);
        assert_eq!(
            output[0].ints(),
            (0..5).map(|x| if flag { x * (x + 1) / 2 } else { x * 3 }).collect::<Vec<_>>()
        );
    }
}

#[test]
fn device_loop_reexecutes_local_collectives_and_handles_zero_iterations() {
    let result = compile(
        "entry main(xs: [4]i32, n: i32) [4]i32 = loop acc = xs for k < n do map(|x: i32| x + k, acc)",
    );
    assert_eq!(kernel_count(&result), 1);
    assert_eq!(result.state.dispatches.len(), 1);
    for n in [0, 1, 4] {
        let output = run(&result, vec![Value::array(1..5), Value::Int(n)]);
        assert_eq!(
            output[0].ints(),
            (1..5).map(|x| x + n * (n - 1) / 2).collect::<Vec<_>>()
        );
    }
}

#[test]
fn scalar_only_array_counted_and_while_loops_execute_accumulator_updates() {
    let result = compile(
        "entry main(xs: [4]i32, n: i32) (i32, i32, i32) =
        let a = loop acc = 0 for x in xs do acc + x in
        let b = loop acc = 0 for i < n do acc + i in
        let (c, _) = loop (acc, i) = (0, 0) while i < n do (acc + i, i + 1) in
        (a, b, c)",
    );
    assert_eq!(kernel_count(&result), 0);
    for n in [0, 1, 4] {
        let output = run(&result, vec![Value::array(1..5), Value::Int(n)]);
        let sum = n * (n - 1) / 2;
        assert_eq!(
            output,
            [Value::Tuple(vec![
                Value::Int(10),
                Value::Int(sum),
                Value::Int(sum)
            ])]
        );
    }
}

#[test]
fn noncommutative_associative_reduction_preserves_chunk_order() {
    let result = compile("entry main(xs: []i32) (i32, i32) = reduce(|a: (i32, i32), b: (i32, i32)| (a.0 * b.0 % 97, (a.1 * b.0 + b.1) % 97), (1, 0), map(|x: i32| (x % 3 + 1, x), xs))");
    for n in [0, 65, 137] {
        let expected = (0..n).fold((1, 0), |(a, b), x| {
            ((a * (x % 3 + 1)) % 97, (b * (x % 3 + 1) + x) % 97)
        });
        let result = run(&result, vec![Value::array(0..n)]);
        assert_eq!(
            result[0],
            Value::Tuple(vec![Value::Int(expected.0), Value::Int(expected.1)])
        );
    }
}

#[test]
fn ordered_writes_and_ranked_bucket_overflow_lower_to_ordinary_memory_ops() {
    let scatter =
        compile("entry main(dest: *[3]i32) [3]i32 = scatter(dest, [0, 0, -1, 3, 2], [1, 2, 9, 9, 7])");
    let output = run(&scatter, vec![Value::array([10, 20, 30])]);
    assert_eq!(output[0].ints(), [2, 20, 7]);
    let buckets = compile("entry main(dest: *[2][2]i32) ([2][2]i32, [2]u32, u32) = bucket_scatter_2d(dest, [[(-1, 9), (0, 10), (0, 11)], [(0, 12), (1, 20), (2, 9)]])");
    let output = run(
        &buckets,
        vec![Value::arrays(vec![Value::array([0, 0]), Value::array([0, 0])])],
    );
    let Value::Tuple(fields) = &output[0] else {
        panic!("bucket outputs");
    };
    assert_eq!(
        fields[0],
        Value::arrays(vec![Value::array([10, 11]), Value::array([20, 0])])
    );
    assert_eq!(fields[1].ints(), [3, 1]);
    assert_eq!(fields[2], Value::Int(1));
}

#[test]
fn nested_array_work_uses_device_loops_without_device_launches() {
    let result = compile("entry main(xs: []i32) []i32 = map(|x: i32| reduce(|a: i32, b: i32| a + b, 0, map(|y: i32| y + x, iota(5))), xs)");
    assert_eq!(kernel_count(&result), 1);
    let output = run(&result, vec![Value::array(0..7)]);
    assert_eq!(output[0].ints(), (0..7).map(|x| 10 + x * 5).collect::<Vec<_>>());
    for block in result.state.blocks.values() {
        let kind = &result.state.blocks[block.function].interface.as_ref().unwrap().kind;
        if matches!(kind, FunctionKind::Kernel(_) | FunctionKind::Device) {
            assert!(!result.state.bodies[block.body]
                .instructions
                .iter()
                .any(|i| matches!(i, Instruction::Dispatch(_))));
        }
    }
}

#[test]
fn indexed_updates_preserve_collisions_existing_bins_and_invalid_index_guards() {
    let result = compile("entry main(dest: *[3]i32, indices: [5]i32, values: [5]i32) [3]i32 = reduce_by_index(dest, |a: i32, b: i32| a + b, 0, indices, values)");
    assert_eq!(kernel_count(&result), 1);
    let output = run(
        &result,
        vec![
            Value::array([10, 20, 30]),
            Value::array([0, 0, -1, 3, 2]),
            Value::array([1, 2, 100, 200, 7]),
        ],
    );
    assert_eq!(output[0].ints(), [13, 20, 37]);
}

#[test]
fn scheduling_replaces_all_soacs_with_blocks() {
    let result =
        compile("entry main(xs: []i32) []i32 = scan(|a: i32, b: i32| a + b, 0, map(|x: i32| x * 2, xs))");
    assert_eq!(result.state.dispatches.len(), 3);
    for body in result.state.bodies.values() {
        for instruction in &body.instructions {
            if let Instruction::Evaluate(op) = instruction {
                assert!(matches!(
                    result.ir.operations[*op].kind,
                    OperationKind::Call { .. } | OperationKind::EvalGlobal(_) | OperationKind::Index { .. }
                ));
            }
        }
    }
}

#[test]
fn verifier_rejects_incorrect_block_arguments_and_cyclic_dispatches() {
    let mut result = compile("entry main(xs: []i32) i32 = reduce(|a: i32, b: i32| a + b, 0, xs)");
    let first = result.state.dispatches.ids().next().unwrap();
    result.state.dispatches[first].dependencies.insert(first);
    assert!(validate(&result, PipelineTopologyPolicy::AllowGenerated)
        .unwrap_err()
        .to_string()
        .contains("cyclic"));
    result.state.dispatches[first].dependencies.remove(&first);
    let edge = result
        .state
        .blocks
        .values()
        .find_map(|b| if let Exit::Jump(edge) = &b.exit { Some(edge.clone()) } else { None })
        .unwrap();
    result.state.bodies[edge.arguments].results.clear();
    assert!(validate(&result, PipelineTopologyPolicy::AllowGenerated)
        .unwrap_err()
        .to_string()
        .contains("arity"));
}

#[test]
fn grid_stride_kernels_cover_more_work_than_the_launched_invocations() {
    for source in [
        "entry main(xs: []i32) []i32 = map(|x: i32| x + 1, xs)",
        "entry main(xs: []i32) []i32 = scan(|a: i32, b: i32| a + b, 0, xs)",
        "entry main(xs: []i32) ?k. [k]i32 = filter(|x: i32| x % 2 == 0, xs)",
    ] {
        let mut result = compile(source);
        for id in result.state.grids.ids().collect::<Vec<_>>() {
            result.state.grids[id].groups[0] = super::Value::Int(1);
        }
        let output = run(&result, vec![Value::array(0..137)]);
        let expected: Vec<_> = if source.contains("map(") {
            (0..137).map(|x| x + 1).collect()
        } else if source.contains("scan(") {
            (0..137).map(|x| x * (x + 1) / 2).collect()
        } else {
            (0..137).filter(|x| x % 2 == 0).collect()
        };
        assert_eq!(output[0].ints(), expected);
    }
}

#[test]
fn nested_source_loops_keep_distinct_accumulators_and_header_bindings() {
    let result = compile("entry main(xs: [4]i32, n: i32) [4]i32 = let (ys, _) = loop (acc, i) = (xs, 0) while i < n do (loop inner = acc for j < 3 do map(|x: i32| x + i + j, inner), i + 1) in ys");
    for n in [0, 1, 3] {
        let output = run(&result, vec![Value::array(0..4), Value::Int(n)]);
        let increment = (0..n).map(|i| 3 * i + 3).sum::<i64>();
        assert_eq!(
            output[0].ints(),
            (0..4).map(|x| x + increment).collect::<Vec<_>>()
        );
    }
}

#[test]
fn final_roots_exclude_dead_functions_and_pure_array_work() {
    let result = compile("def unused(xs: []i32) []i32 = scan(|a: i32, b: i32| a + b, 0, xs) entry main(xs: []i32) i32 = let _ = filter(|x: i32| x > 0, xs) in 7");
    assert_eq!(kernel_count(&result), 0);
    assert_eq!(run(&result, vec![Value::array(0..5)]), [Value::Int(7)]);
    let functions: Vec<_> = result.state.blocks.values().filter_map(|b| b.interface.as_ref()).collect();
    assert_eq!(functions.len(), 1);
}

#[test]
fn producer_buffers_are_shared_by_identity_with_consumer_dispatches() {
    let result =
        compile("entry main(xs: []i32) i32 = reduce(|a: i32, b: i32| a + b, 0, scan(|a: i32, b: i32| a + b, 0, xs))");
    assert_eq!(result.state.dispatches.len(), 5);
    let (&producer_id, producer) = result
        .state
        .dispatches
        .iter()
        .find(|(_, d)| result.state.blocks[d.kernel].interface.as_ref().unwrap().name == "offsets")
        .unwrap();
    let consumer = result
        .state
        .dispatches
        .values()
        .find(|d| {
            result.state.blocks[d.kernel].interface.as_ref().unwrap().name == "chunks"
                && d.dependencies.contains(&producer_id)
        })
        .unwrap();
    assert!(!producer.writes.is_empty());
    assert!(producer.writes.is_subset(&consumer.reads));
    let output = run(&result, vec![Value::array(0..65)]);
    assert_eq!(output, [Value::Int(64 * 65 * 66 / 6)]);
}

#[test]
fn result_copy_waits_for_its_entry_launches_and_publishes_storage() {
    let result = compile("entry first(xs:[137]i32) i32 = reduce(|a:i32,b:i32|a+b,0,xs) entry second(xs:[5]i32) i32 = reduce(|a:i32,b:i32|a*b,1,xs)");
    let kernels: Vec<_> = result.state.physical_kernels.kernels().collect();
    for output in result.state.outputs.values() {
        assert!(output.copy);
        let owner = crate::EntryId::from(output.entry.as_u32());
        let mut owned =
            kernels.iter().filter(|k| k.source_entry == Some(owner)).copied().collect::<Vec<_>>();
        let copy = owned.pop().unwrap();
        assert_eq!(copy.dependencies, owned.iter().map(|k| k.id).collect::<Vec<_>>());
        let resource = crate::ResourceId::from_egglog_buffer(output.buffer.unwrap().as_u32());
        assert!(copy.resources.iter().any(|r| r.resource == resource
            && matches!(
                r.access,
                crate::ResourceAccess::Write | crate::ResourceAccess::ReadWrite
            )));
        assert!(owned.iter().all(|k| k.resources.iter().all(|r| r.resource != resource)));
    }
}
