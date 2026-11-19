//! Ordinary maps reject nonuniform nested results before shader lowering.
#[path = "support/pipeline.rs"]
mod pipeline;
use wyn_core::{lower_ssa_to_spirv, lower_ssa_to_wgsl_with_program, CodegenTarget};
use wyn_host::ShaderFormat;

#[test]
fn retained_prefixes_and_packed_outputs_publish_distinct_lengths() {
    let source = pipeline::source("entry main() ([300]i32,[]i32)=let prefix=scan((+),0,iota(300)) let packed=filter(|x|x>100,prefix) in (prefix,packed)");
    let ssa = pipeline::lower(&source, CodegenTarget::Portable);
    lower_ssa_to_spirv(ssa.clone()).unwrap();
    let output = lower_ssa_to_wgsl_with_program(ssa).unwrap();
    assert_eq!(output.program.interface.source_results.len(), 2);
    output.program.to_whl("retained", ShaderFormat::Wgsl).unwrap();
}

#[test]
fn maps_reject_nonuniform_result_shapes() {
    let source = "
entry ragged() [][]i32=map(|n|map(|i|i+n,iota(n)),[0,1,3,7])
entry ragged_transpose() [][]i32=transpose(map(|n|iota(n),[0,1,3,7]))
entry nested() [][][]i32=map(|n|map(|j|map(|k|100*n+10*j+k,iota(j)),iota(n)),[0,1,3])
entry mixed() [][]i32=map(|n|if n>3 then filter(|i|i>1,iota(n)) else filter(|i|true,scan((+),0,iota(n))),[0,1,3,7])
entry packed() [][]i32=map(|n|filter(|i|i>1,iota(n)),[0,1,3,7])
entry prefixes() [][]i32=map(|n|scan((+),0,iota(n)),[0,1,3,7])
entry retained() ([][]i32,[][]i32)=let rows=map(|n|map(|i|i+n,iota(n)),[0,1,3,7]) in (rows,rows[1..3])
entry sliced() [][]i32=let rows=map(|n|map(|i|i+n,iota(n)),[0,1,3,7]) in rows[1..3]
entry copied() [][]i32=let rows=map(|n|map(|i|i+n,iota(n)),[0,1,3,7]) in map(|row|row,rows)
entry consumed() [4]i32=let rows=map(|n|map(|i|i+n,iota(n)),[0,1,3,7]) in map(|row|reduce((+),0,row),rows)
entry choice() [][]i32=map(|n|if n>1 then map(|i|i,iota(n)) else map(|i|10+i,iota(n+1)),[0,1,3,4])
entry empty_rows() [][]i32=map(|n|map(|i|i+n,iota(n)),[0,0,0])
entry deep(xs:[]i32,n:i32,m:i32) [][][]i32=map(|x|if x>0 then [iota(n)] else [iota(m)],xs)
entry captured_branch(xs:[]i32,b:bool) [][]i32=map(|x|if b then iota(x) else iota(3),xs)
";
    for source in source.lines().filter(|line| line.starts_with("entry ")) {
        let program = pipeline::source(source);
        let imported = wyn_core::egglog::from_tlc(&program).unwrap();
        let error = wyn_core::egglog::fuse(imported).err().expect("irregular map must be rejected");
        assert!(
            error.to_string().contains("map callback results must have the same array shape"),
            "{source}: {error}"
        );
    }
}
