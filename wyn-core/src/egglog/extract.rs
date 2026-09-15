//! Apply one fusion decision to the complete program retained in the sidecar.

use super::data::{AssociatedData, OperationId, OperationKind, RegionId, SoacBody};
use super::optimize::OptimizeError;
use crate::types::SoacOwnership;
use egglog_engine::{ast::Literal, EGraph, Term, TermDag};

/// Choose deterministically, then refresh the summary before the next decision.
/// Egglog owns legality; this step materializes composition of opaque bodies.
pub(super) fn fuse_one(graph: &EGraph, data: &mut AssociatedData) -> Result<bool, OptimizeError> {
    let (rows, _, dag) = graph.function_to_dag("FusionCandidate", usize::MAX, false)?;
    let mut choices = Vec::new();
    for row in rows {
        let args = app(&dag, row, "FusionCandidate", 3)?;
        let region: RegionId = key(&dag, args[0], "RegionId")?;
        let producer: OperationId = key(&dag, args[1], "OperationId")?;
        let consumer: OperationId = key(&dag, args[2], "OperationId")?;
        choices.push((region, producer, consumer));
    }
    choices.sort();
    let Some((region, producer, consumer)) = choices.into_iter().next() else {
        return Ok(false);
    };
    let Some(scope) = data.regions.get(region) else {
        return Err(invalid("fusion region has no sidecar record"));
    };
    if producer == consumer || !scope.members.contains(&producer) || !scope.members.contains(&consumer) {
        return Err(invalid("fusion candidate does not belong to the selected graph"));
    }
    let Some(source) = data.operations.get(producer) else {
        return Err(invalid("fusion producer has no sidecar record"));
    };
    let OperationKind::Screma {
        form: producer_form,
        inputs: producer_inputs,
        ..
    } = &source.kind
    else {
        return Err(invalid("fusion producer is not a Screma"));
    };
    let first = producer_form.pre.clone();
    let inputs = producer_inputs.clone();
    let Some(record) = data.operations.get_mut(consumer) else {
        return Err(invalid("fusion consumer has no sidecar record"));
    };
    let OperationKind::Screma {
        form,
        inputs: consumer_inputs,
        ownership,
    } = &mut record.kind
    else {
        return Err(invalid("fusion consumer is not a Screma"));
    };
    form.pre = match (first, form.pre.clone()) {
        (SoacBody::Identity(_), then) => then,
        (first, SoacBody::Identity(_)) => first,
        (first, then) => SoacBody::Compose {
            first: Box::new(first),
            then: Box::new(then),
        },
    };
    *consumer_inputs = inputs;
    ownership.fill(SoacOwnership::Fresh);
    // Keep old records for provenance. Reachability omits the absorbed producer.
    Ok(true)
}

fn invalid(message: &str) -> OptimizeError {
    OptimizeError::Extraction(message.into())
}

fn app<'a>(
    dag: &'a TermDag,
    node: usize,
    expected: &str,
    arity: usize,
) -> Result<&'a [usize], OptimizeError> {
    match dag.get(node) {
        Term::App(name, args) if name == expected && args.len() == arity => Ok(args),
        _ => Err(invalid(&format!(
            "expected {expected}/{arity} in fusion decision"
        ))),
    }
}

fn key<Id: From<u32>>(dag: &TermDag, node: usize, name: &str) -> Result<Id, OptimizeError> {
    let args = app(dag, node, name, 1)?;
    let Term::Lit(Literal::Int(value)) = dag.get(args[0]) else {
        return Err(invalid("expected an integer sidecar ID"));
    };
    let value = u32::try_from(*value).map_err(|_| invalid("sidecar ID is out of range"))?;
    Ok(Id::from(value))
}
