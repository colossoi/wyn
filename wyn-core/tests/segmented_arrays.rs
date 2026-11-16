//! Nonuniform nested results retain their offsets through both public backends.
#[path = "support/pipeline.rs"]
mod pipeline;
use wyn_core::{lower_ssa_to_spirv, lower_ssa_to_wgsl_with_program, CodegenTarget};
use wyn_host::{Operation, ResultShape, ShaderFormat};

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
fn segmented_shapes_allocate_after_their_counts_and_publish_every_level() {
    let source = "
entry ragged() [][]i32=map(|n|map(|i|i+n,iota(n)),[0,1,3,7])
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
";
    let program = pipeline::source(source);
    let ssa = pipeline::lower(&program, CodegenTarget::Portable);
    lower_ssa_to_spirv(ssa.clone()).unwrap();
    let emitted = lower_ssa_to_wgsl_with_program(ssa).unwrap();
    let host = &emitted.program;
    for result in host.interface.source_results.iter().filter(|r| r.entry != "consumed") {
        let ResultShape::Segmented { offsets, .. } = &result.shape else {
            panic!("nonuniform shape lost for {}", result.entry);
        };
        assert_eq!(offsets.len(), if result.entry == "nested" { 2 } else { 1 });
    }
    for entry in host.entries.iter().filter(|entry| entry.name != "consumed") {
        let first_dispatch =
            entry.operations.iter().position(|op| matches!(op, Operation::Dispatch { .. })).unwrap();
        assert!(
            entry.operations[first_dispatch + 1..].iter().any(|op| matches!(op, Operation::Allocate(_))),
            "{} must allocate its segmented result after producing counts",
            entry.name
        );
        if entry.name == "ragged" {
            let allocation = entry
                .operations
                .iter()
                .enumerate()
                .skip(first_dispatch + 1)
                .find(|(_, op)| matches!(op, Operation::Allocate(_)))
                .unwrap()
                .0;
            assert!(
                entry.operations[..allocation]
                    .iter()
                    .filter(|op| matches!(op, Operation::Dispatch { .. }))
                    .count()
                    >= 2,
                "counting ragged rows must wait for the dispatch that writes their descriptors"
            );
        }
    }
    host.to_whl("segmented", ShaderFormat::Wgsl).unwrap();
    host.to_rust_wgpu("segmented", ShaderFormat::Wgsl).unwrap();
}
