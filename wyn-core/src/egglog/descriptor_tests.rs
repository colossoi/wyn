//! Unit checks for the array descriptors produced by planning.
use super::facts::Facts;
use super::query::Query;
use super::{
    from_tlc, fuse, optimize_for_test, output_error as error, place, prepare_host, schedule, OptimizeError,
};
use crate::types::TypeExt;
use crate::{compile_thru_tlc, LookupSet, PipelineTopologyPolicy};
use egglog_engine::Value;

fn validate(facts: &Facts<'_, '_>) -> Result<(), OptimizeError> {
    let query = Query(&facts.program.graph);
    let mut controls = LookupSet::default();
    query.for_function("SourceLoop", |row, info| {
        controls.insert(row[0]);
        let header =
            facts.enode("LoopInfo", info).ok_or_else(|| error("loop descriptor context missing"))?[0];
        controls.insert(facts.loop_state(header).ok_or_else(|| error("loop state descriptor missing"))?);
        Ok(())
    })?;
    query.for_function("SourceBranch", |row, _| {
        controls.insert(row[0]);
        Ok(())
    })?;
    query.for_function("ArrayStorage", |row, storage| {
        let source = row[0];
        let fields = facts
            .record("ArrayView", "ViewInfo", (source,))
            .ok_or_else(|| error(format!("array {source:?} has no complete descriptor")))?;
        if fields[0] != storage {
            return Err(error("array descriptor disagrees with planned storage"));
        }
        let mut seen = LookupSet::default();
        extent(facts, fields[1], &controls, &mut seen)?;
        extent(facts, fields[2], &controls, &mut seen)?;
        Ok(())
    })?;
    query.for_function("ValueRead", |row, access| {
        if let Some(fields) = facts.enode("ReadView", access) {
            if facts
                .lookup("SourceType", (row[0],))
                .and_then(|ty| facts.lookup("SourceArraySize", (ty,)))
                .is_some()
            {
                if facts.lookup("ArrayStorage", (row[0],)) != Some(fields[0]) {
                    return Err(error("array read disagrees with planned storage"));
                }
            }
        }
        Ok(())
    })?;
    Ok(())
}

fn extent(
    facts: &Facts<'_, '_>,
    value: Value,
    controls: &LookupSet<Value>,
    seen: &mut LookupSet<Value>,
) -> Result<(), OptimizeError> {
    if !seen.insert(value) {
        return Ok(());
    }
    let (name, fields) = facts.extent(value).ok_or_else(|| error("invalid array descriptor extent"))?;
    match name {
        "Fixed" | "Stored" => Ok(()),
        "Sum" | "Product" | "Difference" => {
            extent(facts, fields[0], controls, seen)?;
            extent(facts, fields[1], controls, seen)
        }
        "Length" | "Scalar" => {
            let source = fields[0];
            let ty =
                facts.source_type(source).ok_or_else(|| error("array descriptor scalar has no type"))?;
            if name == "Scalar" {
                if ty.is_array() || facts.lookup("ExprContextOwner", (source,)).is_none() {
                    return Err(error("array descriptor scalar has no lexical context"));
                }
                for constructor in ["SourceShape", "SourceOffset"] {
                    if let Some(parent) = facts.enode(constructor, source) {
                        if !controls.contains(&parent[0]) {
                            return Err(error("array descriptor context field has no control-flow binder"));
                        }
                    }
                }
            }
            Ok(())
        }
        _ => Err(error("unsupported array descriptor extent")),
    }
}

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
    validate(&Facts {
        program: &prepared.program,
    })
    .unwrap();
}
