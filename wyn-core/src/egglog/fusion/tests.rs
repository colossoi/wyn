use crate::compile_thru_tlc;
use crate::egglog::{from_tlc, fuse, query::Query};

#[test]
fn scan_barriers_survive_intermediate_map_contraction() {
    let source =
        compile_thru_tlc("entry main(xs:[64]i32) i32 = reduce((+),0i32,map(|x|x+1,scan((+),0i32,xs)))")
            .unwrap();
    let program = fuse(from_tlc(&source).unwrap()).unwrap();
    let query = Query(&program.graph);
    let program = query.row("CurrentProgram", |_| true).unwrap().unwrap()[0];
    let root = query.constructor("ProgramRoot", (program,)).unwrap().unwrap();
    let mut scans = 0;
    let mut totals = 0;
    query
        .for_each("ContainsExp", |row| {
            if row[0] != root {
                return Ok(());
            }
            if let Some(parallel) = query.enode("Parallel", row[1])? {
                let body = query.fields("ParallelBlock", parallel[2])?;
                let mut scan = false;
                let mut total = false;
                for clause in query.vector(body[1])? {
                    if let Some(fold) = query.enode("Fold", clause)? {
                        total |= query.enode("Total", fold[0])?.is_some();
                        scan |= query.enode("InclusiveScan", fold[0])?.is_some();
                    }
                }
                assert!(
                    !(scan && total),
                    "a dependent total must wait for completed scan prefixes"
                );
                scans += usize::from(scan);
                totals += usize::from(total);
            }
            Ok(())
        })
        .unwrap();
    assert_eq!((scans, totals), (1, 1));
}

#[test]
fn free_names_preserve_large_shared_fanout() {
    let mut graph = super::new_graph().unwrap();
    let mut statements = String::new();
    for i in 0..256 {
        statements.push_str(&format!("(Let (StatementTag {i}) (vec-of (BindValue (ValueId {}) (ScalarInfo (ScalarElement (I32))))) (vec-of) (PrimOp \"+\" (vec-of (Var (ValueId 0)) (LiteralBits (I32) {i}))))\n", i+1));
    }
    graph
        .parse_and_run_program(
            None,
            &format!(
                r#"
        (let body (Block (vec-of {statements}) (vec-of)))
        (run-schedule (saturate memory-names))
        (check (set-contains (NamesBody body) (ValueNameRef (ValueId 0))))
        (check (= (set-length (BindersBody body)) 256))
    "#
            ),
        )
        .unwrap();
}

// Cross-entry canonicalization is exercised through both backends by
// tests/compiler/cases/independent_fusion_*.wyn.
