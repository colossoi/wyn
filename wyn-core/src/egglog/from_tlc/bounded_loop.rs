use super::Term;
use crate::op::BinaryOperator;
use crate::tlc::{TermKind, VarRef};
use crate::types::{self, Type};
use crate::{LookupMap, SymbolId};

pub(super) fn maximum(
    accumulator: SymbolId,
    fields: &[(SymbolId, Type, Term)],
    initial: &Term,
    condition: &Term,
    step: &Term,
) -> Option<i64> {
    let mut definitions = LookupMap::default();
    let initial = result(initial, &mut definitions);
    let initial_definitions = definitions;
    let mut definitions = LookupMap::default();
    let step = result(step, &mut definitions);
    let step_definitions = definitions;
    let mut conditions = LookupMap::default();
    let condition = result(condition, &mut conditions);
    let (counter, stop) = guard(condition, &conditions)?;
    let (start, next) = if counter == accumulator {
        (initial, step)
    } else {
        let (_, _, field) = fields.iter().find(|(name, _, _)| *name == counter)?;
        let TermKind::TupleProj { tuple, idx } = &field.kind else {
            return None;
        };
        if symbol(tuple) != Some(accumulator) {
            return None;
        }
        let TermKind::Tuple(initial_fields) = &initial.kind else {
            return None;
        };
        let TermKind::Tuple(next_fields) = &step.kind else {
            return None;
        };
        (initial_fields.get(*idx)?, next_fields.get(*idx)?)
    };
    let start = literal(result(start, &mut initial_definitions.clone()))?;
    let next = result(next, &mut step_definitions.clone());
    let (lhs, rhs) = binary(next, BinaryOperator::Add)?;
    let increment = |value: &Term| literal(result(value, &mut step_definitions.clone()));
    if !(symbol(lhs) == Some(counter) && increment(rhs) == Some(1)
        || symbol(rhs) == Some(counter) && increment(lhs) == Some(1))
    {
        return None;
    }
    // A signed counter with a strict upper bound reaches that bound before i32 overflow.
    Some((i64::from(stop) - i64::from(start)).max(0))
}

fn symbol(term: &Term) -> Option<SymbolId> {
    match term.kind {
        TermKind::Var(VarRef::Symbol(name)) => Some(name),
        _ => None,
    }
}

fn literal(term: &Term) -> Option<i32> {
    if term.ty != types::i32() {
        return None;
    }
    match &term.kind {
        TermKind::IntLit(text) => text.parse().ok(),
        _ => None,
    }
}

fn result<'a>(mut term: &'a Term, definitions: &mut LookupMap<SymbolId, &'a Term>) -> &'a Term {
    loop {
        match &term.kind {
            TermKind::Let { name, rhs, body, .. } => {
                definitions.insert(*name, rhs);
                term = body;
            }
            TermKind::Var(VarRef::Symbol(name)) => match definitions.get(name) {
                Some(value) => term = value,
                None => return term,
            },
            _ => return term,
        }
    }
}

fn binary(term: &Term, expected: BinaryOperator) -> Option<(&Term, &Term)> {
    let TermKind::App { func, args } = &term.kind else {
        return None;
    };
    let TermKind::BinOp(op) = &func.kind else {
        return None;
    };
    let [lhs, rhs] = args.as_slice() else {
        return None;
    };
    (op.op == expected).then_some((lhs, rhs))
}

fn guard<'a>(condition: &'a Term, definitions: &LookupMap<SymbolId, &'a Term>) -> Option<(SymbolId, i32)> {
    let mut definitions = definitions.clone();
    let condition = result(condition, &mut definitions);
    if let Some((lhs, rhs)) = binary(condition, BinaryOperator::Less) {
        let counter = result(lhs, &mut definitions);
        return (counter.ty == types::i32())
            .then(|| Some((symbol(counter)?, literal(result(rhs, &mut definitions))?)))
            .flatten();
    }
    if let Some((lhs, rhs)) = binary(condition, BinaryOperator::LogicalAnd) {
        return guard(lhs, &definitions).or_else(|| guard(rhs, &definitions));
    }
    if let TermKind::If {
        cond,
        then_branch,
        else_branch,
    } = &condition.kind
    {
        if matches!(
            result(else_branch, &mut definitions.clone()).kind,
            TermKind::BoolLit(false)
        ) {
            return guard(cond, &definitions).or_else(|| guard(then_branch, &definitions));
        }
    }
    None
}
