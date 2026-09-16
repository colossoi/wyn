//! Decode constructor applications and sidecar IDs from egglog terms.
use super::OptimizeError;
use egglog_engine::ast::{Command, Parser};
use egglog_engine::{ast::Literal, Term, TermDag};

pub(super) fn parse(filename: &str, source: &str) -> Result<Vec<Command>, OptimizeError> {
    Parser::default()
        .get_program_from_string(Some(filename.into()), source)
        .map_err(|e| OptimizeError::Output(e.to_string()))
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
        _ => Err(invalid(&format!("expected {expected}/{arity} in egglog term"))),
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
