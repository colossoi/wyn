//! Array descriptors use the same lexical scope checks as instructions.
use super::query::Query;
use super::{from_tlc, fuse, optimize_for_test, place, prepare_host, schedule};
use crate::{compile_thru_tlc, PipelineTopologyPolicy};

#[test]
fn planned_array_descriptors_are_consistent() {
    let source = r#"
entry case_0(xs: []i32) []i32 =
  if length(xs)>10 then replicate(0,0i32) else xs

entry case_1(xs: []i32) []i32 =
  loop out = xs for i < length(xs) do out with [i] = xs[i] + 1

entry case_2(xs: []i32) []i32 =
  let out = if length(xs)>10 then replicate(0,0i32) else xs
  let start = if length(out)>0 then 1 else 0
  in out[start..length(out)] with [0] = 99i32
"#;
    let tlc = compile_thru_tlc(source).unwrap();
    let program = optimize_for_test(
        schedule(
            place(
                fuse(from_tlc(&tlc).unwrap()).unwrap(),
                PipelineTopologyPolicy::AllowGenerated,
            )
            .unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    let prepared = prepare_host(program).unwrap();
    let query = Query(&prepared.program.graph);
    let root = query.row("CurrentProgram", |_| true).unwrap().unwrap()[0];
    let root = query.constructor("ProgramRoot", (root,)).unwrap().unwrap();
    let mut arrays = 0;
    query
        .for_each("ContainsValueInfo", |row| {
            if row[0] == root {
                if let Some(info) = query.enode("ArrayInfo", row[1])? {
                    let location = query.fields("ArrayIn", info[1])?;
                    let layout = query.fields("Layout", location[1])?;
                    assert!(!query.vector(layout[1])?.is_empty());
                    arrays += 1;
                }
                assert!(query.enode("ArrayValue", row[1])?.is_none());
            }
            Ok(())
        })
        .unwrap();
    assert!(arrays > 0);
}
