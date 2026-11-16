use crate::compile_thru_tlc;
use crate::egglog::{from_tlc, fuse};

// Exported intermediate reads are checked in memory_ir_retained_read.wyn.
#[test]
fn rewrites_stay_with_their_program_revision() {
    let source = compile_thru_tlc("entry main(xs:[8]i32) [8]i32=map(|x|x*2,map(|x|x+1,xs))").unwrap();
    let program = from_tlc(&source).unwrap();
    let mut original = program.graph.clone();
    crate::egglog::memory::validate(&mut original).unwrap();
    let root = crate::egglog::dataflow::edit::revision(&original).unwrap();
    let fused = fuse(program).unwrap();
    assert_eq!(crate::egglog::dataflow::edit::revision(&original).unwrap(), root);
    let after = crate::egglog::dataflow::edit::revision(&fused.graph).unwrap();
    assert_ne!(after, root);
}
