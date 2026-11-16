#[path = "support/pipeline.rs"]
mod pipeline;
use wyn_core::{lower_ssa_to_spirv, lower_ssa_to_wgsl_with_program, CodegenTarget};

fn check(source: &str) {
    let program = pipeline::source(source);
    let ssa = pipeline::lower(&program, CodegenTarget::Portable);
    lower_ssa_to_spirv(ssa.clone()).unwrap();
    lower_ssa_to_wgsl_with_program(ssa).unwrap();
}

#[test]
fn conditional_tuple_maps_reach_both_backends() {
    check(
        "entry main(xs: [4]i32, ys: [4]i32, flag: bool) [4](i32,i32) =
             if flag then map(|x|(x,x+1),xs) else map(|y|(y,y+2),ys)",
    );
}

#[test]
fn conditional_zipped_maps_need_no_parameter_repair() {
    check(
        "entry main(xs: [4]i32, ys: [4]i32, flag: bool) [4]i32 =
             if flag then map(|(x,y)|x+y,zip(xs,ys))
             else map(|(y,x)|y-x,zip(ys,xs))",
    );
}

#[test]
fn conditional_maps_with_unproven_domains_stay_separate() {
    check(
        "entry main(xs: []i32, ys: []i32, flag: bool) []i32 =
             if flag then map(|x|x+1,xs) else map(|y|y+2,ys)",
    );
}

#[test]
fn conditional_maps_keep_branch_local_prefixes_and_captures() {
    check(
        "entry main(xs: [4]i32, ys: [4]i32, flag: bool, bias: i32) [4]i32 =
             if flag then let offset = bias + 1 in map(|x| x + offset, xs)
             else let offset = bias + 2 in map(|y| y + offset, ys)",
    );
}
