//! Indexed and ranked producers cross the public pipeline without TLC hoisting.
#[path = "support/pipeline.rs"]
mod pipeline;
use wyn_core::{lower_ssa_to_spirv, lower_ssa_to_wgsl_with_program, CodegenTarget};

#[test]
fn runtime_indices_and_ranked_consumers_reach_both_backends() {
    let source = "
entry ranked(dest:*[4][8]u32) ([4][8]u32,[4]u32,u32)=
  let items=map(|bucket|map(|pair|((bucket+pair)%4,u32(bucket+pair)),iota(6)),iota(4))
  in bucket_scatter_2d(dest,items)
entry scoped(dest:*[4][8]i32) ([4][8]i32,[4]u32,u32)=
  let items:[4][6](i32,i32)=map(|bucket|let offset=bucket*2 in map(|pair|((offset+pair)%4,offset+pair),iota(6)),iota(4))
  in bucket_scatter_2d(dest,items)
entry transitive(dest:*[4][8]i32) ([4][8]i32,[4]u32,u32)=
  let items:[4][6](i32,i32)=map(|bucket|let offset=bucket*2 let shifted=offset+1 in map(|pair|((shifted+pair)%4,shifted+pair),iota(6)),iota(4))
  in bucket_scatter_2d(dest,items)
entry retained(dest:*[4][8]i32) (i32,[4]u32,u32)=
  let items=map(|bucket|map(|pair|((bucket+pair)%4,bucket+pair),iota(6)),iota(4))
  let (values,counts,overflow)=bucket_scatter_2d(dest,items)
  in (reduce((+),0,map(|row|reduce((+),0,map(|p|p.1,row)),items)),counts,overflow)
def producer(n:i32) []f32=map(|i|f32.i32(i),0i32..<n)
entry nested_capture(n:i32,j:i32) [2][3]f32=map(|a|map(|b|producer(n)[j]+f32.i32(a+b),iota(3)),iota(2))
entry indexed(j:i32) [1]f32=[producer(256)[j]]
entry scatter_index(dest:*[8]f32,j:i32) [8]f32=
  scatter(dest,map(|p|i32.f32(producer(8)[j]),iota(4)),[0.0,1.0,2.0,3.0])
";
    let program = pipeline::source(source);
    let ssa = pipeline::lower(&program, CodegenTarget::Portable);
    lower_ssa_to_spirv(ssa.clone()).unwrap();
    lower_ssa_to_wgsl_with_program(ssa).unwrap();
}
