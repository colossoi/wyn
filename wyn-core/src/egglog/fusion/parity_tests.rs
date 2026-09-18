use super::super::{fuse, Fused, Program};
use crate::compile_thru_tlc;
use crate::egglog::dependencies::analyze;
use crate::egglog::{from_tlc, insert_expressions, schedule, simplify_and_place, Imported, Ir, Scheduled};
use crate::tlc::infer_input_slice_bounds;
use crate::PipelineTopologyPolicy;
use exec::{run, Value};

#[allow(dead_code)]
#[path = "../schedule_test_exec.rs"]
mod exec;

fn imported(source: &str) -> Program<Imported> {
    from_tlc(&infer_input_slice_bounds(compile_thru_tlc(source).unwrap())).unwrap()
}
fn count(data: &Ir) -> usize {
    let root = data.definitions[data.entries.values().next().unwrap().definition].body;
    analyze(data).live.iter().filter(|&&id| data.operations[id].region == root).count()
}
fn check(source: &str, operations: usize) {
    let input = imported(source);
    let fused = fuse(input.clone()).unwrap();
    assert_eq!(
        count(&fused.ir),
        operations,
        "{source}\n{:#?}",
        fused.ir.operations
    );
    // Deliberately bypass fusion only for the unfused execution oracle.
    let input = Program {
        ir: input.ir,
        state: Fused,
    };
    let original = schedule(
        simplify_and_place(insert_expressions(input).unwrap()).unwrap(),
        PipelineTopologyPolicy::AllowGenerated,
    )
    .unwrap();
    let fused = schedule(
        simplify_and_place(insert_expressions(fused).unwrap()).unwrap(),
        PipelineTopologyPolicy::AllowGenerated,
    )
    .unwrap();
    for n in [0, 1, 4, 63, 64, 65, 137] {
        let args = || vec![Value::array((0..n).map(|i| i % 11 - 3))];
        assert_eq!(run(&fused, args()), run(&original, args()), "n={n}: {source}");
    }
}

#[test]
fn horizontal_maps_reductions_and_scans_preserve_all_outputs() {
    check("entry main(xs: []i32) ([]i32,i32,[]i32) = (map(|x:i32|x+1,xs),reduce(|a:i32,b:i32|a+b,0,xs),scan(|a:i32,b:i32|a+b,0,xs))",1);
}
#[test]
fn shared_producer_outputs_and_multiple_consumers_survive() {
    check("entry main(xs: []i32) ([]i32,[]i32,[]i32) = let a=map(|x:i32|x+1,xs) in (a,map(|x:i32|x*2,a),map(|x:i32|x-3,a))",1);
}
#[test]
fn zipped_inputs_and_reduction_bearing_producers_compose() {
    check("entry main(xs: []i32) ([]i32,i32) = let a=map(|x:i32|x+1,xs) in let other=map(|x:i32|x*3,xs) in let b=reduce(|a:i32,b:i32|a+b,0,xs) in (map(|(x,y):(i32,i32)|x*y,zip(a,other)),b)",1);
}

#[test]
fn scan_followed_by_map_preserves_the_barrier() {
    check(
        "entry main(xs: []i32) []i32 = map(|x:i32|x*2,scan(|a:i32,b:i32|a+b,0,xs))",
        1,
    );
}
#[test]
fn dependent_scan_then_reduction_keeps_two_operations() {
    check(
        "entry main(xs: []i32) i32 = reduce(|a:i32,b:i32|a+b,0,scan(|a:i32,b:i32|a+b,0,xs))",
        2,
    );
}
#[test]
fn conditional_projection_crosses_only_the_independent_scan_barrier() {
    // Conditional lambda results may be projected across independent barriers.
    check(
        "entry main(xs: []i32) ([]i32,i32) =
        let prefixes=scan(|a:i32,b:i32|a+b,0,xs) in
        let values=map(|x:i32|x+1,xs) in
        let paired=map(|(prefix,x):(i32,i32)|(if x<0 then 0-x else x,prefix*2),zip(prefixes,values)) in
        let (magnitudes,doubled)=unzip(paired) in
        (doubled,reduce(|a:i32,b:i32|a+b,0,magnitudes))",
        1,
    );
}

#[test]
fn maps_compose_into_filter_without_losing_the_mapped_elements() {
    check(
        "entry main(xs: []i32) ?k. [k]i32 = filter(|x:i32|x%3==1,map(|x:i32|x*2+7,xs))",
        1,
    );
}
#[test]
fn map_filter_map_reduce_becomes_one_masked_collective() {
    check("entry main(xs: []i32) i32 = reduce(|a:i32,b:i32|a+b,0,map(|x:i32|x*5,filter(|x:i32|x%2==0,map(|x:i32|x+1,xs))))",1);
}
#[test]
fn returned_filter_array_prevents_masking_away_compaction() {
    check("entry main(xs: []i32) (?k. [k]i32,i32) = let kept=filter(|x:i32|x>0,xs) in (kept,reduce(|a:i32,b:i32|a+b,0,kept))",2);
}

fn check_args(source: &str, operations: usize, args: impl Fn() -> Vec<Value>) -> Program<Scheduled> {
    let input = imported(source);
    let fused = fuse(input.clone()).unwrap();
    assert_eq!(
        count(&fused.ir),
        operations,
        "{source}\n{:#?}",
        fused.ir.operations
    );
    // Deliberately bypass fusion only for the unfused execution oracle.
    let input = Program {
        ir: input.ir,
        state: Fused,
    };
    let original = schedule(
        simplify_and_place(insert_expressions(input).unwrap()).unwrap(),
        PipelineTopologyPolicy::AllowGenerated,
    )
    .unwrap();
    let fused = schedule(
        simplify_and_place(insert_expressions(fused).unwrap()).unwrap(),
        PipelineTopologyPolicy::AllowGenerated,
    )
    .unwrap();
    assert_eq!(run(&fused, args()), run(&original, args()), "{source}");
    fused
}

#[test]
fn maps_into_reduce_by_index_preserve_collisions_and_initial_bins() {
    for (keys, values) in [
        ("map(|x:i32|x-1, indices)", "values"),
        ("indices", "map(|x:i32|x*3+1, values)"),
        ("map(|x:i32|x-1, indices)", "map(|x:i32|x*3+1, values)"),
    ] {
        check_args(
            &format!(
                "entry main(dest: *[3]i32, indices: [5]i32, values: [5]i32) [3]i32 =
            reduce_by_index(dest, |a:i32,b:i32|a+b, 0, {keys}, {values})"
            ),
            1,
            || {
                vec![
                    Value::array([10, 20, 30]),
                    Value::array([1, 1, -1, 4, 2]),
                    Value::array([2, 3, 7, 9, 11]),
                ]
            },
        );
    }
}

#[test]
fn maps_into_scatter_preserve_last_write_and_invalid_index_guards() {
    check_args(
        "entry main(dest: *[3]i32, xs: [5]i32) [3]i32 =
        scatter(dest, map(|x:i32|x-1, xs), map(|x:i32|x*7, xs))",
        1,
        || vec![Value::array([10, 20, 30]), Value::array([1, 1, -1, 4, 2])],
    );
}

#[test]
fn map_into_bucket_scatter_preserves_counts_and_overflow() {
    check_args(
        "entry main(dest: *[2][2]i32, xs: [6]i32) ([2][2]i32,[2]u32,u32) =
        bucket_scatter_1d(dest, map(|x:i32|(x, x*3), xs))",
        1,
        || {
            vec![
                Value::arrays(vec![Value::array([0, 0]), Value::array([0, 0])]),
                Value::array([0, 0, -1, 0, 1, 2]),
            ]
        },
    );
}

#[test]
fn indexed_map_demands_preserve_tuple_elements_and_captures() {
    check_args(
        "entry main(xs: [4]i32, bias:i32) (i32,i32) =
        let pairs=map(|x:i32|(x+bias,[x,x*2,x*3]), xs) in
        let p=pairs[2] in (p.0,p.1[1])",
        3,
        || vec![Value::array([1, 3, 7, 11]), Value::Int(5)],
    );
}

#[test]
fn filter_length_becomes_a_count_reduction() {
    check("entry main(xs: []i32) i32 = length(filter(|x:i32|x>0,xs))", 1);
}

#[test]
fn filtered_reductions_share_a_count_for_multiple_length_observers() {
    check(
        "entry main(xs: []i32) (i32,i32,i32,i32) =
        let kept=filter(|x:i32|x>0,xs) in
        let n=length(kept) in
        let total=reduce(|a:i32,b:i32|a+b,0,kept) in
        let m=length(kept) in
        let largest=reduce(|a:i32,b:i32|if a>b then a else b,-2147483648,kept) in
        (n,total,m,largest)",
        1,
    );
}

#[test]
fn sliced_map_inputs_compose_without_materializing_the_full_producer() {
    for slice in ["produced[2..6]", "produced[1..7][2..6]"] {
        check_args(
            &format!(
                "entry main(xs: []i32, ys: []i32) [4]i32 =
            let produced=map(|(x,y):(i32,i32)|x+y,zip(xs,ys)) in
            map(|x:i32|x*2,{slice})"
            ),
            1,
            || vec![Value::array(0..8), Value::array(10..18)],
        );
    }
}

#[test]
fn incompatible_slices_keep_the_producer_materialized() {
    check_args(
        "entry main(xs: [8]i32) [4]i32 =
        let a=map(|x:i32|x+1,xs) in
        map(|(x,y):(i32,i32)|x+y,zip(a[0..4],a[2..6]))",
        2,
        || vec![Value::array(0..8)],
    );
}
