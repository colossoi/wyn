use crate::compile_thru_tlc;
use crate::egglog::{from_tlc, fuse, query::Query};

// Exported intermediate reads are checked in memory_ir_retained_read.wyn.
#[test]
fn rewrites_stay_with_their_program_revision() {
    let source = compile_thru_tlc("entry main(xs:[8]i32) [8]i32=map(|x|x*2,map(|x|x+1,xs))").unwrap();
    let program = from_tlc(&source).unwrap();
    let mut original = program.graph.clone();
    crate::egglog::memory::validate(&mut original).unwrap();
    let root = Query(&original).row("CurrentProgram", |_| true).unwrap().unwrap()[0];
    let fused = fuse(program).unwrap();
    assert_eq!(
        Query(&original).row("CurrentProgram", |_| true).unwrap().unwrap()[0],
        root
    );
    let after = Query(&fused.graph).row("CurrentProgram", |_| true).unwrap().unwrap()[0];
    assert_ne!(after, root);
}
