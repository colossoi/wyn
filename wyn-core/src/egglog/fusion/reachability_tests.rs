use crate::compile_thru_tlc;
use crate::egglog::{from_tlc, fuse, query::Query};

#[test]
fn contractions_preserve_intervening_reads() {
    let source = compile_thru_tlc(
        "entry main(xs:[8]i32,j:i32) ([8]i32,i32) = let ys=map(|x|x+1,xs) in (map(|x|x*2,ys),ys[j])",
    )
    .unwrap();
    let program = fuse(from_tlc(&source).unwrap()).unwrap();
    let query = Query(&program.graph);
    let root = query.row("CurrentProgram", |_| true).unwrap().unwrap()[0];
    let root = query.constructor("ProgramRoot", (root,)).unwrap().unwrap();
    let mut loads = 0;
    query
        .for_each("ContainsExp", |row| {
            if row[0] == root && query.enode("Load", row[1])?.is_some() {
                loads += 1;
            }
            Ok(())
        })
        .unwrap();
    assert!(
        loads > 0,
        "the exported intermediate read cannot disappear with its producer"
    );
}

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
