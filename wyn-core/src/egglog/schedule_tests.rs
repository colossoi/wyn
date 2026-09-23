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
fn input_and_slice_lengths_are_evaluated_by_their_consumers() {
    for source in [
        "entry main(xs:[]i32) []i32 = let n=length(xs) in map(|i|xs[i]+n,iota(n))",
        "entry main(xs:[]i32) []i32 = let ys=xs[1..length(xs)] in
         let n=length(ys) in map(|i|ys[i]+n,iota(n))",
    ] {
        let result = compile(source);
        assert_eq!(kernel_count(&result), 1);
        for n in [1, 2, 65] {
            let start = i64::from(source.contains("let ys="));
            assert_eq!(
                run(&result, vec![Value::array(0..n)]),
                [Value::array((start..n).map(|x| x + n - start))]
            );
        }
    }
    let empty = compile("entry main(xs:[]i32) []i32 = map(|i|i+length(xs),iota(length(xs)))");
    assert_eq!(kernel_count(&empty), 1);
    assert_eq!(run(&empty, vec![Value::array([])]), [Value::array([])]);
}

#[test]
fn filter_length_consumers_read_the_live_count_without_an_extra_dispatch() {
    let result = compile(
        "entry main(xs:[]i32) []i32 = let ys=filter(|x:i32|x>0,xs) in
         map(|i|ys[i]+length(ys),iota(length(ys)))",
    );
    assert_eq!(kernel_count(&result), 4);
    for xs in [vec![], vec![-1, 0], vec![-1, 3, 0, 7], vec![2; 65]] {
        let kept: Vec<_> = xs.iter().copied().filter(|&x| x > 0).collect();
        assert_eq!(
            run(&result, vec![Value::array(xs)]),
            [Value::array(kept.iter().map(|&x| x + kept.len() as i64))]
        );
    }
}

#[test]
fn mutable_input_length_does_not_capture_element_contents() {
    let result = compile(
        "entry main(xs:*[3]i32) [3]i32 = let n=length(xs) in
         let ys=scatter(xs,[0],[100]) in map(|x:i32|x+n,ys)",
    );
    assert_eq!(kernel_count(&result), 2);
    assert_eq!(
        run(&result, vec![Value::array([5, 2, 9])]),
        [Value::array([103, 5, 12])]
    );
}

#[test]
fn mapped_array_length_does_not_require_its_discarded_elements() {
    for (view, start) in [("ys", 0), ("ys[0..length(ys)]", 0), ("ys[1..length(ys)]", 1)] {
        let result = compile(&format!(
            "entry main(xs:[]i32) []i32 = let ys=map(|x:i32|x+1,xs) in
             map(|i|i,iota(length({view})))",
        ));
        for n in [0, 1, 65].into_iter().filter(|&n| n >= start) {
            assert_eq!(
                run(&result, vec![Value::array(0..n)]),
                [Value::array(0..n - start)]
            );
        }
    }
}

#[test]
fn immutable_input_scalar_loads_are_read_by_their_consumers() {
    let result = compile("entry main(xs: []i32) []i32 = let first = xs[0] in map(|x:i32|x+first, xs)");
    assert_eq!(kernel_count(&result), 1);
    assert_eq!(
        run(&result, vec![Value::array([5, 2, 9])]),
        [Value::array([10, 7, 14])]
    );
}

#[test]
fn dynamic_indices_and_small_helpers_rematerialize_in_consumers() {
    let result = compile(
        "def adjust(x:i32) i32=x*x+1
        entry main(xs:[]i32,k:i32) []i32 =
        let bias=adjust(xs[k]) in map(|x:i32|x+bias,xs)",
    );
    assert_eq!(kernel_count(&result), 1);
    assert_eq!(
        run(&result, vec![Value::array([5, 2, 9]), Value::Int(1)]),
        [Value::array([10, 7, 14])]
    );
}

#[test]
fn consecutive_gpu_scalar_loops_share_one_kernel_and_keep_intermediates_local() {
    let result = compile(
        "entry main(xs:[]i32,n:i32) []i32 =
        let a=loop acc=xs[0] for i<n do acc+i in
        let b=loop acc=a for i<n do acc+i*2 in map(|x:i32|x+b,xs)",
    );
    assert_eq!(kernel_count(&result), 2);
    let group = result.state.execution.groups.values().find(|group| group.len() == 2).unwrap();
    assert!(!result.state.materialized.contains_key(&group[0]));
    assert!(result.state.materialized.contains_key(&group[1]));
    assert_eq!(
        run(&result, vec![Value::array([5, 2, 9]), Value::Int(4)]),
        [Value::array([28, 25, 32])]
    );
    assert_eq!(
        run(&result, vec![Value::array([5, 2, 9]), Value::Int(0)]),
        [Value::array([10, 7, 14])]
    );
}

#[test]
fn scalar_epilogues_join_across_independent_maps_and_publish_once() {
    let result = compile(include_str!("../../../testfiles/scalar_epilogues.wyn"));
    assert_eq!(kernel_count(&result), 3);
    assert_eq!(result.state.abi.roots.len(), 3, "no finish kernel");
    assert!(result.state.execution.groups.values().any(|group| group.len() == 2));
    for n in [0, 1, 7] {
        let a = 5 + n * (n - 1) / 2;
        let b = a + n * (n - 1);
        for xs in [vec![], vec![3], vec![-2; 65]] {
            assert_eq!(
                run(
                    &result,
                    vec![Value::array(xs.clone()), Value::array([5]), Value::Int(n)]
                ),
                [Value::Tuple(vec![
                    Value::array(xs.iter().map(|x| x + a)),
                    Value::array((0..3).map(|i| i * a)),
                    Value::array([a, b]),
                ])]
            );
        }
    }
}

#[test]
fn scalar_join_waits_for_intervening_map_results() {
    let result = compile(
        "entry main(xs:[]i32,n:i32) ([]i32,[1]i32) =
         let a=loop acc=xs[0] for i<n do acc+i in
         let ys=map(|x:i32|x+a,xs) in
         let b=loop acc=ys[0] for i<n do acc+i*2 in (ys,[b])",
    );
    assert_eq!(kernel_count(&result), 3);
    assert!(!result.state.execution.groups.values().any(|group| group.len() > 1));
    assert_eq!(
        run(&result, vec![Value::array([5, 2, 9]), Value::Int(4)]),
        [Value::Tuple(vec![Value::array([16, 13, 20]), Value::array([28])])]
    );
}

#[test]
fn scalar_join_preserves_intervening_mutation_and_old_state_reads() {
    let result = compile(
        "entry main(xs:*[3]i32,n:i32) ([3]i32,[2]i32) =
         let a=loop acc=xs[0] for i<n do acc+i in
         let ys=scatter(xs,[0],[100]) in
         let b=loop acc=ys[0] for i<n do acc+i*2 in (ys,[a,b])",
    );
    assert_eq!(kernel_count(&result), 3);
    assert_eq!(
        run(&result, vec![Value::array([5, 2, 9]), Value::Int(4)]),
        [Value::Tuple(vec![
            Value::array([100, 2, 9]),
            Value::array([11, 112])
        ])]
    );
}

#[test]
fn mutable_scalar_snapshot_stays_before_the_update() {
    let result = compile(
        "entry main(xs:*[3]i32) [3]i32 =
        let old=xs[0] in let updated=scatter(xs,[0],[100]) in
        map(|x:i32|x+old,updated)",
    );
    assert_eq!(kernel_count(&result), 3);
    assert_eq!(
        run(&result, vec![Value::array([5, 2, 9])]),
        [Value::array([105, 7, 14])]
    );
}

#[test]
fn mutable_alias_through_control_preserves_the_snapshot() {
    let result = compile(
        "entry main(xs:*[3]i32,flag:bool) [3]i32 =
        let view=if flag then xs[0..2] else xs[1..3] in let old=view[0] in
        let updated=scatter(xs,[0],[100]) in map(|x:i32|x+old,updated)",
    );
    for (flag, expected) in [(true, [105, 7, 14]), (false, [102, 4, 11])] {
        assert_eq!(
            run(&result, vec![Value::array([5, 2, 9]), Value::Bool(flag)]),
            [Value::array(expected)]
        );
    }
}

#[test]
fn literal_tuple_reduction_has_a_fixed_scratch_capacity() {
    let source = "def min_pair(hits: [4](i32, i32)) (i32, i32) =
         reduce(|(a, ai): (i32, i32), (b, bi): (i32, i32)|
                  if a < b then (a, ai) else (b, bi),
                (1000, 0), hits)
         def hits: [4](i32, i32) = [(4, 0), (2, 1), (1, 2), (3, 3)]
         entry main() (i32, i32) = min_pair(hits)";
    let tlc = infer_input_slice_bounds(compile_thru_tlc(source).unwrap());
    let program = insert_expressions(fuse(from_tlc(&tlc).unwrap()).unwrap()).unwrap();
    let program = super::super::place(super::super::simplify(program, false).unwrap()).unwrap();
    let result = schedule(program, PipelineTopologyPolicy::AllowGenerated).unwrap();
    assert_eq!(kernel_count(&result), 2);
    assert!(result
        .state
        .abi
        .bindings
        .values()
        .all(|binding| matches!(binding.length, Some(crate::host::BufferLen::Fixed { .. }))));
    assert_eq!(
        run(&result, vec![]),
        [Value::Tuple(vec![Value::Int(1), Value::Int(2)])]
    );
}

#[test]
fn scatter_local_destination_preserves_initial_values() {
    for source in [
        "entry main() [4]i32 = scatter(replicate(4, 7i32), [2i32, 0i32], [30i32, 10i32])",
        "entry main() [4]i32 = let dest = [7i32, 7i32, 7i32, 7i32] in scatter(dest, [2i32, 0i32], [30i32, 10i32])",
        "entry main() [4]i32 = spread(4, 7i32, [2i32, 0i32], [30i32, 10i32])",
    ] {
        let result = compile(source);
        assert_eq!(run(&result, vec![])[0].ints(), [10, 7, 30, 7], "{source}");
    }
}

#[test]
fn scatter_local_runtime_destination() {
    let result = compile(
        "entry main(xs: []i32) []i32 = scatter(replicate(length(xs), 7i32), [2i32, 0i32], [30i32, 10i32])",
    );
    assert_eq!(
        run(&result, vec![Value::array(0..5)])[0].ints(),
        [10, 7, 30, 7, 7]
    );
    assert_eq!(run(&result, vec![Value::array(0..1)])[0].ints(), [10]);
    assert!(run(&result, vec![Value::array(0..0)])[0].ints().is_empty());
}

#[test]
fn scatter_local_bounds_empty_updates_and_consumers() {
    for (source, expected) in [
        ("entry main() [4]i32 = scatter(0i32..<4, [-1, 4, 1, 1], [90, 90, 8, 9])", vec![0, 9, 2, 3]),
        ("entry main() [4]i32 = scatter([7, 8, 9, 10], 0i32..<0, 0i32..<0)", vec![7, 8, 9, 10]),
        ("entry main() [4]i32 = let a = scatter([7, 7, 7, 7], [1], [10]) in map(|x| x+1, scatter(a, [2], [20]))", vec![8, 11, 21, 8]),
    ] {
        assert_eq!(run(&compile(source), vec![])[0].ints(), expected, "{source}");
    }
}

#[test]
fn scatter_local_copy_does_not_change_a_value_parameter() {
    let result = compile("entry main(xs: [4]i32) ([4]i32, [4]i32) = (scatter(xs, [1], [99]), xs)");
    let input = Value::array([1, 2, 3, 4]);
    let output = run(&result, vec![input.clone()]);
    let Value::Tuple(fields) = &output[0] else {
        panic!("tuple result")
    };
    assert_eq!(fields[0].ints(), [1, 99, 3, 4]);
    assert_eq!(fields[1].ints(), [1, 2, 3, 4]);
    assert_eq!(input.ints(), [1, 2, 3, 4]);
}

#[test]
fn scatter_local_copy_preserves_its_producer_and_can_be_discarded() {
    let result = compile("entry main() ([4]i32, [4]i32) = let initial = replicate(4, 7i32) in (initial, scatter(initial, [1], [99]))");
    let output = run(&result, vec![]);
    let Value::Tuple(fields) = &output[0] else {
        panic!("tuple result")
    };
    assert_eq!(fields[0].ints(), [7, 7, 7, 7]);
    assert_eq!(fields[1].ints(), [7, 99, 7, 7]);
    let discarded = compile("entry main() i32 = let _ = scatter([1, 2], [0], [9]) in 42");
    assert_eq!(run(&discarded, vec![]), [Value::Int(42)]);
}

#[test]
fn scatter_local_inside_runtime_loop() {
    let result = compile(
        "entry main(n: i32) [4]i32 = loop acc = [7, 7, 7, 7] for k < n do scatter(acc, [k], [k+10])",
    );
    assert_eq!(run(&result, vec![Value::Int(3)])[0].ints(), [10, 11, 12, 7]);
}

#[test]
fn scatter_local_radix_partition_is_stable_across_scan_chunks() {
    let result = compile(
        r#"
entry main(keys: []i32) []i32 =
  let ids = 0i32..<length(keys) in
  let zero = map(|key| if key % 2 == 0 then 1 else 0, keys) in
  let prefix = scan(|a,b| a+b, 0i32, zero) in
  let total = prefix[length(keys)-1] in
  let destinations = map(|i| if zero[i] == 1 then prefix[i]-1 else total+i-prefix[i], ids) in
  scatter(replicate(length(keys), 0i32), destinations, ids)
"#,
    );
    for count in [4, 65, 130] {
        let keys: Vec<i64> = (0..count).map(|i| (i * 7 + 3) % 11).collect();
        let expected: Vec<i64> = (0..count)
            .filter(|&i| keys[i as usize] % 2 == 0)
            .chain((0..count).filter(|&i| keys[i as usize] % 2 != 0))
            .collect();
        assert_eq!(run(&result, vec![Value::array(keys)])[0].ints(), expected);
    }
}

#[test]
fn map_after_scatter_does_not_wait_for_its_own_length_read() {
    let result = compile(
        "entry update(positions: []i32, fb: *[]i32) *[]i32 =
         let pts = positions[0..5] in
         let indices = map(|p: i32| p, pts) in
         let values = map(|p: i32| 1, pts) in
         let updated = scatter(fb, indices, values) in
         map(|x: i32| x, updated)",
    );
    let input = Value::array([1, 3, 0, 4, 2]);
    let framebuffer = Value::array(0..10);
    let output = run(&result, vec![input, framebuffer.clone()]);
    assert_eq!(output[0].ints(), [1, 1, 1, 1, 1, 5, 6, 7, 8, 9]);
    assert_eq!(framebuffer.ints(), output[0].ints());
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
    let dispatch = result.state.dispatches.values().next().unwrap();
    assert_eq!(dispatch.reads.len(), 1);
    assert_eq!(
        dispatch.reads, dispatch.writes,
        "reuse must preserve both access flags"
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
fn readonly_callback_loops_use_parallel_invocations() {
    let result = compile(
        "entry main(xs:[]i32) []i32 =
        map(|i:i32| loop acc=0 for j<3 do acc+xs[i], iota(length(xs)))",
    );
    assert!(result.state.blocks.values().any(|block| block
        .interface
        .as_ref()
        .is_some_and(|f| matches!(f.kind, FunctionKind::Kernel([64, 1, 1])))));
    assert_eq!(
        run(&result, vec![Value::array(0..137)])[0].ints(),
        (0..137).map(|i| i * 3).collect::<Vec<_>>()
    );
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
    assert_eq!(kernel_count(&filter), 3);
    for (result, names) in [
        (&scan, &["chunks", "combine", "offsets"][..]),
        (&filter, &["local_offsets", "offsets", "compact"][..]),
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
fn filter_post_map_runs_only_for_survivors_in_the_compact_phase() {
    let result = compile("entry main(xs:[]i32) []i32 = map(|x:i32|120/x,filter(|x:i32|x!=0,xs))");
    assert_eq!(kernel_count(&result), 3);
    assert_eq!(
        result.state.buffers.values().filter(|b| b.storage == Storage::Device).count(),
        6
    );
    for n in [0, 1, 63, 64, 65, 137] {
        for pattern in 0..4 {
            let xs: Vec<_> = (0..n)
                .map(|i| match pattern {
                    0 => i % 5 + 1,
                    1 => 0,
                    2 => {
                        if i % 2 == 0 {
                            2
                        } else {
                            0
                        }
                    }
                    _ => {
                        if i % 17 == 1 {
                            3
                        } else {
                            0
                        }
                    }
                })
                .collect();
            let expected = Value::array(xs.iter().copied().filter(|&x| x != 0).map(|x| 120 / x));
            assert_eq!(
                run(&result, vec![Value::array(xs)]),
                [expected],
                "n={n}, pattern={pattern}"
            );
        }
    }
}

#[test]
fn filter_post_map_composes_chains_and_captured_type_changing_outputs() {
    let result = compile(
        "entry main(xs:*[]i32,bias:i32) [](i32,i32) =
         let ys=filter(|x:i32|x>0,map(|x:i32|x-1,xs)) in
         let zs=map(|x:i32|x+bias,ys) in map(|x:i32|(x,x*2),zs)",
    );
    assert_eq!(kernel_count(&result), 3);
    let output = run(&result, vec![Value::array([0, 2, 5, 1, 9]), Value::Int(7)]);
    for (i, x) in [8, 11, 15].into_iter().enumerate() {
        assert_eq!(
            output[0].at(i),
            Value::Tuple(vec![Value::Int(x), Value::Int(x * 2)])
        );
    }
}

#[test]
fn filter_post_map_record_output_keeps_live_count_and_capacity_distinct() {
    let result = compile(include_str!("../../../testfiles/rust_host_filter_post.wyn"));
    assert_eq!(kernel_count(&result), 3);
    for xs in [
        vec![],
        vec![-1; 65],
        vec![1; 65],
        (0..137).map(|i| i % 3 - 1).collect(),
    ] {
        let n = xs.len() as i64;
        let values: Vec<_> = xs
            .iter()
            .enumerate()
            .filter(|(_, x)| **x > 0)
            .map(|(i, _)| {
                let value = i as i64 + 1;
                Value::Tuple(vec![Value::Int(value), Value::Int(value * 3 + n)])
            })
            .collect();
        let output = run(&result, vec![Value::array(xs), Value::Int(n)]);
        let Value::Tuple(fields) = &output[0] else {
            panic!("expected count and record array");
        };
        assert_eq!(fields[0], Value::Int(values.len() as i64));
        for (i, value) in values.into_iter().enumerate() {
            assert_eq!(fields[1].at(i), value);
        }
    }
}

#[test]
fn filter_post_map_executes_in_nested_serial_filters() {
    let result = compile(
        "entry main(xs:[]i32) []i32 = map(|x:i32|
         let kept=filter(|y:i32|y!=0,[-1,0,1,2]) in
         let mapped=map(|y:i32|x+120/y,kept) in
         loop total=0 for y in mapped do total+y, xs)",
    );
    assert_eq!(kernel_count(&result), 1);
    assert_eq!(
        run(&result, vec![Value::array([0, 1, 5])]),
        [Value::array([60, 63, 75])]
    );
}

#[test]
fn consuming_scans_reuse_the_input_only_for_final_prefixes() {
    for (body, multiplier, scratch) in [
        ("scan(|a:i32,b:i32|a+b,0,xs)", 1, 3),
        // Fusion carries one mapped component across the scan barrier.
        ("map(|x:i32|x*2,scan(|a:i32,b:i32|a+b,0,xs))", 2, 4),
    ] {
        let result = compile(&format!("entry main(xs:*[]i32) []i32 = {body}"));
        assert_eq!(kernel_count(&result), 3);
        assert_eq!(
            result.state.buffers.values().filter(|b| b.storage == Storage::Device).count(),
            scratch,
            "only collective scratch needs fresh storage"
        );
        for n in [0, 1, 63, 64, 65, 137] {
            let input = Value::array(0..n);
            let output = run(&result, vec![input.clone()]);
            let expected: Vec<_> = (0..n).map(|i| multiplier * i * (i + 1) / 2).collect();
            assert_eq!(output[0].ints(), expected);
            assert_eq!(input.ints(), expected);
        }
    }
}

#[test]
fn dependent_scans_share_owned_storage_without_overwriting_borrowed_input() {
    let result = compile(
        "entry main(xs:[]i32) []i32 = let a=scan(|x:i32,y:i32|x+y,0,xs) in scan(|x:i32,y:i32|x+y,0,a)",
    );
    assert_eq!(kernel_count(&result), 6);
    assert_eq!(
        result.state.buffers.values().filter(|b| b.storage == Storage::Device).count(),
        7
    );
    for n in [0, 1, 63, 64, 65, 137] {
        let input = Value::array(0..n);
        let output = run(&result, vec![input.clone()]);
        assert_eq!(
            output[0].ints(),
            (0..n).map(|i| i * (i + 1) * (i + 2) / 6).collect::<Vec<_>>()
        );
        assert_eq!(input.ints(), (0..n).collect::<Vec<_>>());
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
fn ordered_scatter_and_parallel_ranked_buckets_preserve_results_and_overflow() {
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
    let first = fields[0].at(0).ints();
    assert_eq!(first.len(), 2);
    assert_ne!(first[0], first[1]);
    assert!(first.iter().all(|x| (10..=12).contains(x)));
    assert_eq!(fields[0].at(1).ints(), [20, 0]);
    assert_eq!(buckets.state.dispatches.len(), 2);
    assert_eq!(fields[1].ints(), [3, 1]);
    assert_eq!(fields[2], Value::Int(1));
}

#[test]
fn atomic_indexed_reductions_preserve_existing_bins_and_collisions() {
    for operator in ["a+b", "max(a,b)"] {
        let program = compile(&format!(
            "entry main(dest:*[3]i32, xs:[7]i32) [3]i32 =
            reduce_by_index(dest, |a:i32,b:i32|{operator}, 0, [-1,0,0,1,2,3,0], xs)"
        ));
        let output = run(
            &program,
            vec![
                Value::array([10, 20, 30]),
                Value::array([100, 2, 3, 40, 50, 100, 7]),
            ],
        );
        assert_eq!(
            output[0].ints(),
            if operator == "a+b" { vec![22, 60, 80] } else { vec![10, 40, 50] }
        );
    }
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
