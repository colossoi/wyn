//! Read-only path predicates over native edges, with query-local traversal state.
use crate::egglog::{parse_program, OptimizeError};
use egglog_engine::ast::Span;
use egglog_engine::constraint::{SimpleTypeConstraint, TypeConstraint};
use egglog_engine::prelude::BaseSort;
use egglog_engine::sort::{BoolSort, I64Sort};
use egglog_engine::{ArcSort, Core, EGraph, Primitive, Read, ReadPrim, ReadState, Value, Write};
use std::collections::HashSet;

pub(super) fn register(graph: &mut EGraph) -> Result<(), OptimizeError> {
    let Some(group) = graph.get_sort_by_name("FusionGroup").cloned() else {
        return Err(OptimizeError::Output("fusion group sort missing".into()));
    };
    for intermediate in [false, true] {
        graph.add_read_primitive(
            Path {
                group: group.clone(),
                intermediate,
            },
            None,
        );
    }
    Ok(())
}

#[derive(Clone)]
struct Path {
    group: ArcSort,
    intermediate: bool,
}

impl Primitive for Path {
    fn name(&self) -> &str {
        if self.intermediate {
            "wyn-order-intermediate"
        } else {
            "wyn-group-path"
        }
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        SimpleTypeConstraint::new(
            self.name(),
            vec![
                I64Sort.to_arcsort(),
                self.group.clone(),
                self.group.clone(),
                BoolSort.to_arcsort(),
            ],
            span.clone(),
        )
        .into_box()
    }
}

impl ReadPrim for Path {
    fn apply<'a, 'db>(&self, state: ReadState<'a, 'db>, args: &[Value]) -> Option<Value> {
        // The round is part of the query key: contractions change native edges.
        let &[_, from, to] = args else {
            unreachable!("invalid path predicate arity");
        };
        let table = if self.intermediate { "GroupBefore" } else { "GroupEdge" };
        let mut pending = vec![(from, false)];
        let mut visited = HashSet::new();
        while let Some((node, external)) = pending.pop() {
            if !visited.insert((node, external)) {
                continue;
            }
            if node == to && (!self.intermediate || external) {
                return Some(state.base_to_value(true));
            }
            if let Err(error) = state.constructor_enodes(table, |row| {
                if row.children[0] == node {
                    let next = row.children[1];
                    pending.push((next, external || (next != from && next != to)));
                }
            }) {
                panic!("invalid native path relation {table}: {error}");
            }
        }
        Some(state.base_to_value(false))
    }
}

pub(super) fn run(graph: &mut EGraph) -> Result<(), OptimizeError> {
    graph.update(|mut sink| sink.add("PlanningRound", 0i64))?;
    let dependencies = parse_program(
        "fusion dependencies",
        "(run-schedule (saturate fusion-dependencies))",
    )?;
    let choose = parse_program(
        "fusion candidates",
        "(run-schedule (saturate fusion-candidates) (saturate fusion-select) fusion-choose)",
    )?;
    let commit = parse_program("fusion contraction", "(run-schedule (saturate fusion-record) (saturate fusion-commit) fusion-clear-candidates fusion-union fusion-clear)")?;
    loop {
        graph.run_program(dependencies.clone())?;
        graph.run_program(choose.clone())?;
        let mut chosen = false;
        graph.constructor_enodes("Chosen", |_| chosen = true)?;
        if !chosen {
            break;
        }
        graph.run_program(commit.clone())?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "reachability_tests.rs"]
mod tests;
