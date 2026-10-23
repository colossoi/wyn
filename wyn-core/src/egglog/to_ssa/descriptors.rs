//! Check the descriptor contract before allocating bindings or emitting SSA.
use super::{error, Facts, OptimizeError};
use crate::egglog::query::Query;
use crate::types::TypeExt;
use crate::LookupSet;
use egglog_engine::Value;

pub(super) fn validate(facts: &Facts<'_, '_>) -> Result<(), OptimizeError> {
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
