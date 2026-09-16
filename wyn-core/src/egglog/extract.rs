//! Apply one fusion decision to the complete program retained in the sidecar.

use super::data::{AssociatedData, OperationId, RegionId};
use super::optimize::OptimizeError;
use egglog_engine::{ast::Literal, EGraph, Term, TermDag};

/// Choose deterministically, then refresh the summary before the next decision.
/// Egglog owns legality; this step materializes composition of opaque bodies.
pub(super) fn fuse_one(graph: &EGraph, data: &mut AssociatedData) -> Result<bool, OptimizeError> {
    let (rows, _, dag) = graph.function_to_dag("FusionCandidate", usize::MAX, false)?;
    let mut choices = Vec::new();
    for row in rows {
        let args = app(&dag, row, "FusionCandidate", 4)?;
        let Term::Lit(Literal::Int(family)) = dag.get(args[0]) else {
            return Err(invalid("fusion family"));
        };
        let region: RegionId = key(&dag, args[1], "RegionId")?;
        let producer: OperationId = key(&dag, args[2], "OperationId")?;
        let consumer: OperationId = key(&dag, args[3], "OperationId")?;
        choices.push((*family, region, producer, consumer));
    }
    choices.sort();
    let Some((family, region, producer, consumer)) = choices.into_iter().next() else {
        return Ok(false);
    };
    let Some(scope) = data.regions.get(region) else {
        return Err(invalid("fusion region has no sidecar record"));
    };
    if (producer == consumer && family != 4 && family != 5)
        || !scope.members.contains(&producer)
        || !scope.members.contains(&consumer)
    {
        return Err(invalid("fusion candidate does not belong to the selected graph"));
    }
    let summary = super::snapshot::fusion(data);
    let retained = summary.observed.contains(&producer)
        || summary.uses.iter().any(|&(p, c, _)| p == producer && c != consumer);
    match family {
        0|1=>super::fusion::scremas(data,producer,consumer,family==1,retained),
        2=>super::fusion::envelope(data,producer,consumer),
        3|5=>super::fusion::masked(data,producer,consumer),
        4=>super::fusion::indexed(data,producer),
        _=>None,
    }
        .ok_or_else(||invalid(&format!("fusion body composition disagrees with its legality facts: family {family}, {producer:?} -> {consumer:?}")))?;
    Ok(true)
}

fn invalid(message: &str) -> OptimizeError {
    OptimizeError::Extraction(message.into())
}

pub(super) fn app<'a>(
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

pub(super) fn key<Id: From<u32>>(dag: &TermDag, node: usize, name: &str) -> Result<Id, OptimizeError> {
    let args = app(dag, node, name, 1)?;
    let Term::Lit(Literal::Int(value)) = dag.get(args[0]) else {
        return Err(invalid("expected an integer sidecar ID"));
    };
    let value = u32::try_from(*value).map_err(|_| invalid("sidecar ID is out of range"))?;
    Ok(Id::from(value))
}
