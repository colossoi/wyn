//! Apply scalar-only substitutions without discarding unaffected proofs.
use super::remove_leaves;
use crate::egglog::dataflow::{cse, edit, ir::Code};
use crate::egglog::graph::{Graph, Origin};
use crate::egglog::scalar::ScalarInputs;
use crate::egglog::{analysis, output_error, query::Query, timing, OptimizeError};
use crate::{LookupMap, LookupSet};
use egglog_engine::{Value, Write};

/// Resolve structural identities before creating dataflow and effect proofs.
pub(super) fn before_analysis(graph: &mut Graph, roots: Vec<Value>) -> Result<(), OptimizeError> {
    let _timing = timing::span("Rust structural value forwarding");
    let replacements: Vec<_> = cse::find(graph, roots)?.into_iter().collect();
    if replacements.is_empty() {
        return Ok(());
    }
    let mut removed = Vec::new();
    for &(old, _) in &replacements {
        let Some(Origin::Result(operation, 0)) = graph.origin(old) else {
            return Err(output_error("forwarded expression has no defining operation"));
        };
        removed.push(operation);
    }
    edit::replace(graph, &replacements)?;
    edit::remove(graph, &removed)
}

pub(super) fn run(graph: &mut Graph) -> Result<bool, OptimizeError> {
    let query = Query(graph);
    let mut candidates = LookupMap::new();
    let mut pending = Vec::new();
    query.for_each("ForwardCandidate", |row| {
        if row[0] != row[1] {
            candidates.entry(row[0]).or_insert(row[1]);
        }
        pending.push((row[0], row[1]));
        Ok(())
    })?;
    if candidates.is_empty() {
        return Ok(false);
    }
    let _timing = timing::span("Rust value forwarding");
    let candidates = candidates.into_iter().collect::<Vec<_>>();
    let replacements = edit::resolve_replacements(&candidates)?;
    let mut removed = Vec::new();
    for &(old, new) in &replacements {
        let Some(origin) = query.live_origin(old)? else {
            return Err(output_error("forwarded operand has no live memory definition"));
        };
        if query.live_origin(new)?.is_none() {
            return Err(output_error("forwarding target has no live memory definition"));
        }
        if let Origin::Result(operation, 0) = origin {
            if graph.output_count(operation) == 1 {
                removed.push(operation);
            }
        }
    }
    let local_inputs = if removed.len() == replacements.len() {
        scalar_inputs(query, &replacements, &removed)?
    } else {
        None
    };
    graph.update(|mut sink| {
        for edge in pending {
            sink.remove("ForwardCandidate", edge)?;
        }
        Ok(())
    })?;
    if let Some(inputs) = local_inputs {
        let _timing = timing::span("Rust local scalar forwarding");
        for input in inputs {
            input.apply(graph)?;
        }
        if !remove_leaves::apply(graph, &removed, scalar_uses)? {
            return Err(output_error(
                "scalar forwarding changed whether its definitions could be removed",
            ));
        }
        return Ok(true);
    }
    edit::replace(graph, &replacements)?;
    if !removed.is_empty() {
        edit::remove(graph, &removed)?;
        return Ok(true);
    }
    analysis::ensure(graph)?;
    Ok(false)
}

fn scalar_inputs(
    query: Query<'_>,
    replacements: &[(Value, Value)],
    removed: &[Value],
) -> Result<Option<Vec<ScalarInputs>>, OptimizeError> {
    let graph = query.0;
    let mut updates = LookupMap::<Value, Vec<(usize, Value)>>::new();
    let structure = graph.structure().read();
    for (&(old, new), &producer) in replacements.iter().zip(removed) {
        let Some(name) = query.value_name(old)? else {
            return Ok(None);
        };
        let Some(primitive) = query.lookup("ValuePrimitive", old)? else {
            return Ok(None);
        };
        if !matches!(graph.code(producer)?, Code::Scalar(_) | Code::Constant { .. })
            || !query.flag("PureOperation", producer)?
            || query.contains("LaneDependentOperation", producer)?
            || query.lookup("ValuePrimitive", new)? != Some(primitive)
            || structure.argument_uses.contains_key(&old)
            || structure.yields.uses.contains_key(&old)
            || structure.collective.uses.contains_key(&old)
            || structure.collective.uses.contains_key(&name)
            || structure.certificates.uses.contains_key(&name)
            || query.constructor("IndexValue", name)?.is_some()
        {
            return Ok(None);
        }
        for &operand in structure.inputs.values.get(&producer).into_iter().flatten() {
            if query.enode("AccNameRef", operand)?.is_some() {
                return Ok(None);
            }
        }
        for &(user, slot) in structure.inputs.uses.get(&old).into_iter().flatten() {
            if !matches!(graph.code(user)?, Code::Scalar(_))
                || !query.contains("LiveOperation", user)?
                || !query.flag("PureOperation", user)?
            {
                return Ok(None);
            }
            let Ok(slot) = usize::try_from(slot) else {
                return Ok(None);
            };
            updates.entry(user).or_default().push((slot, new));
        }
    }
    drop(structure);
    let mut inputs = Vec::new();
    for (user, updates) in updates {
        let Some(change) = ScalarInputs::read(query, user, updates)? else {
            return Ok(None);
        };
        inputs.push(change);
    }
    Ok(Some(inputs))
}

fn scalar_uses(query: Query<'_>, operation: Value) -> Result<Vec<Value>, OptimizeError> {
    let mut uses: LookupSet<_> = query.0.inputs(operation).into_iter().collect();
    for certificate in query.0.operation_certificates(operation) {
        let Some(name) = query.constructor("ValueNameRef", certificate)? else {
            return Err(output_error("scalar certificate has no name"));
        };
        uses.insert(name);
    }
    Ok(uses.into_iter().collect())
}
