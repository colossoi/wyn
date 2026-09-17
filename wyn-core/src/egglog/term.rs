//! Decode constructor applications and sidecar IDs from egglog terms.
use super::OptimizeError;
use egglog_engine::ast::Literal;
use egglog_engine::{Term, TermDag};

pub(super) fn app<'a>(
    dag: &'a TermDag,
    node: usize,
    expected: &str,
    arity: usize,
) -> Result<&'a [usize], OptimizeError> {
    match dag.get(node) {
        Term::App(name, args) if name == expected && args.len() == arity => Ok(args),
        _ => Err(OptimizeError::Extraction(format!(
            "expected {expected}/{arity} in egglog term"
        ))),
    }
}

pub(super) fn key<Id: From<u32>>(dag: &TermDag, node: usize, name: &str) -> Result<Id, OptimizeError> {
    let args = app(dag, node, name, 1)?;
    let Term::Lit(Literal::Int(value)) = dag.get(args[0]) else {
        return Err(OptimizeError::Extraction("expected an integer sidecar ID".into()));
    };
    let value = u32::try_from(*value)
        .map_err(|_| OptimizeError::Extraction("sidecar ID is out of range".into()))?;
    Ok(Id::from(value))
}
