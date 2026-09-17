//! Read completed egglog decisions. This module never inspects or mutates bodies.
use crate::egglog::data::{OperationId, RegionId};
use crate::egglog::term::{app, key};
use crate::egglog::OptimizeError;
use egglog_engine::ast::Literal;
use egglog_engine::{EGraph, Term, TermDag};
use std::collections::{BTreeMap, BTreeSet};

pub(super) struct Step {
    pub(super) family: i64,
    pub(super) region: RegionId,
    pub(super) producer: OperationId,
    pub(super) consumer: OperationId,
    pub(super) retained: bool,
    pub(super) lengths: BTreeSet<OperationId>,
    pub(super) demands: BTreeSet<OperationId>,
}

pub(super) fn read(graph: &EGraph) -> Result<Vec<Step>, OptimizeError> {
    fn integer(dag: &TermDag, value: usize) -> Result<i64, OptimizeError> {
        match dag.get(value) {
            Term::Lit(Literal::Int(n)) => Ok(*n),
            _ => Err(OptimizeError::Extraction("expected fusion step number".into())),
        }
    }
    let mut steps = BTreeMap::new();
    let (rows, _, dag) = graph.function_to_dag("FusionStep", usize::MAX, false)?;
    for row in rows {
        let args = app(&dag, row, "FusionStep", 5)?;
        steps.insert(
            integer(&dag, args[0])?,
            Step {
                family: integer(&dag, args[1])?,
                region: key(&dag, args[2], "RegionId")?,
                producer: key(&dag, args[3], "OperationId")?,
                consumer: key(&dag, args[4], "OperationId")?,
                retained: false,
                lengths: BTreeSet::new(),
                demands: BTreeSet::new(),
            },
        );
    }
    let (rows, _, dag) = graph.function_to_dag("StepUse", usize::MAX, false)?;
    for row in rows {
        let args = app(&dag, row, "StepUse", 3)?;
        let n = integer(&dag, args[0])?;
        let consumer = key(&dag, args[1], "OperationId")?;
        let Term::App(name, _) = dag.get(args[2]) else {
            return Err(OptimizeError::Extraction("expected operand role".into()));
        };
        let Some(step) = steps.get_mut(&n) else {
            return Err(OptimizeError::Extraction("use without fusion step".into()));
        };
        match name.as_str() {
            "Length" => {
                step.lengths.insert(consumer);
            }
            "Input" | "Capture" | "Neutral" | "Argument" => {}
            _ => return Err(OptimizeError::Extraction("invalid operand role".into())),
        }
        step.retained |= consumer != step.consumer;
        step.demands.insert(consumer);
    }
    let (rows, _, dag) = graph.function_to_dag("StepObserved", usize::MAX, false)?;
    for row in rows {
        let args = app(&dag, row, "StepObserved", 1)?;
        let n = integer(&dag, args[0])?;
        let Some(step) = steps.get_mut(&n) else {
            return Err(OptimizeError::Extraction("observer without fusion step".into()));
        };
        step.retained = true;
    }
    Ok(steps.into_values().collect())
}
