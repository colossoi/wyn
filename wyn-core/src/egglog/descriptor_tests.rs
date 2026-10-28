//! Unit checks for the array descriptors produced by planning.
use super::facts::Facts;
use super::query::Query;
use super::{
    from_tlc, fuse, optimize_for_test, output_error as error, place, prepare_host, schedule, OptimizeError,
};
use crate::types::TypeExt;
use crate::{compile_thru_tlc, LookupSet, PipelineTopologyPolicy};
use egglog_engine::{Value, Write};

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
    for source in [
        include_str!("../../../testfiles/regressions/triangulation_runtime_array_branch.wyn"),
        include_str!("../../../testfiles/regressions/triangulation_runtime_array_loop.wyn"),
        include_str!("../../../testfiles/regressions/runtime_array_updated_slice.wyn"),
    ] {
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
}

#[test]
fn incomplete_descriptors_are_detected() {
    let tlc = compile_thru_tlc(include_str!(
        "../../../testfiles/regressions/triangulation_runtime_array_branch.wyn"
    ))
    .unwrap();
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
    let mut prepared = prepare_host(program).unwrap();
    validate(&Facts {
        program: &prepared.program,
    })
    .unwrap();
    for (missing, expected) in [
        ("ArrayView", "no complete descriptor"),
        ("ArrayStorage", "array read disagrees with planned storage"),
        ("SourceType", "descriptor scalar has no type"),
        ("ExprContextOwner", "descriptor scalar has no lexical context"),
        ("SourceBranch", "context field has no control-flow binder"),
    ] {
        let query = Query(&prepared.program.graph);
        let output = query.row("ReturnArray", |_| true).unwrap().unwrap()[1];
        let view = query.record("ArrayView", "ViewInfo", (output,)).unwrap().unwrap();
        let scalar = query.enode("Scalar", view[2]).unwrap().unwrap()[0];
        let value =
            if matches!(missing, "ArrayView" | "ArrayStorage" | "SourceBranch") { output } else { scalar };
        let original = query.required(missing, (value,)).unwrap();
        prepared.program.graph.update(|mut sink| sink.remove(missing, (value,))).unwrap();
        let error = validate(&Facts {
            program: &prepared.program,
        })
        .expect_err("invalid descriptor accepted")
        .to_string();
        assert!(error.contains(expected), "{missing}: {error}");
        prepared.program.graph.update(|mut sink| sink.set(missing, (value,), original)).unwrap();
        validate(&Facts {
            program: &prepared.program,
        })
        .unwrap();
    }
}
