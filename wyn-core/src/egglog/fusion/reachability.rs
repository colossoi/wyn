//! Read-only paths over round-local adjacency, with query-local traversal state.
use crate::egglog::{parse_program, timing, OptimizeError};
use egglog_engine::ast::Span;
use egglog_engine::constraint::{SimpleTypeConstraint, TypeConstraint};
use egglog_engine::prelude::BaseSort;
use egglog_engine::sort::{BoolSort, VecContainer};
use egglog_engine::{ArcSort, Core, EGraph, Primitive, PurePrim, PureState, RawValues, Value, Write};
use std::collections::{BTreeMap, HashSet};

pub(super) fn register(graph: &mut EGraph) -> Result<(), OptimizeError> {
    let Some(group) = graph.get_sort_by_name("FusionGroup").cloned() else {
        return Err(OptimizeError::Output("fusion group sort missing".into()));
    };
    let Some(adjacency) = graph.get_sort_by_name("FusionPathGraph").cloned() else {
        return Err(OptimizeError::Output("fusion adjacency sort missing".into()));
    };
    for intermediate in [false, true] {
        graph.add_pure_primitive(
            Path {
                group: group.clone(),
                adjacency: adjacency.clone(),
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
    adjacency: ArcSort,
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
                self.adjacency.clone(),
                self.group.clone(),
                self.group.clone(),
                BoolSort.to_arcsort(),
            ],
            span.clone(),
        )
        .into_box()
    }
}

impl PurePrim for Path {
    fn apply<'a, 'db>(&self, state: PureState<'a, 'db>, args: &[Value]) -> Option<Value> {
        let &[adjacency, from, to] = args else {
            unreachable!("invalid path predicate arity");
        };
        let Some(edges) = state.value_to_container::<VecContainer>(adjacency) else {
            panic!("invalid fusion path adjacency");
        };
        let mut pending = vec![(from, false)];
        let mut visited = HashSet::new();
        while let Some((node, external)) = pending.pop() {
            if !visited.insert((node, external)) {
                continue;
            }
            if node == to && (!self.intermediate || external) {
                return Some(state.base_to_value(true));
            }
            // Rows are sorted by their source group, followed by its neighbors.
            if let Ok(index) = edges.data.binary_search_by_key(&node, |&row| {
                let Some(row) = state.value_to_container::<VecContainer>(row) else {
                    panic!("invalid fusion path row");
                };
                row.data[0]
            }) {
                let Some(neighbors) = state.value_to_container::<VecContainer>(edges.data[index]) else {
                    panic!("invalid fusion path neighbors");
                };
                for &next in &neighbors.data[1..] {
                    pending.push((next, external || (next != from && next != to)));
                }
            }
        }
        Some(state.base_to_value(false))
    }
}

fn clear_snapshot(graph: &mut EGraph) -> Result<(), OptimizeError> {
    for table in ["FusionPathEdges", "FusionPathOrder"] {
        graph.clear_function(table)?;
    }
    Ok(())
}

fn snapshot(graph: &mut EGraph) -> Result<(), OptimizeError> {
    clear_snapshot(graph)?;
    for (source, target) in [
        ("GroupEdge", "FusionPathEdges"),
        ("GroupBefore", "FusionPathOrder"),
    ] {
        let mut edges: BTreeMap<Value, Vec<Value>> = BTreeMap::new();
        graph.constructor_enodes(source, |row| {
            edges.entry(row.children[0]).or_insert_with(|| vec![row.children[0]]).push(row.children[1]);
        })?;
        graph.update(|mut sink| {
            let data = edges
                .into_values()
                .map(|data| {
                    sink.container_to_value(VecContainer {
                        data,
                        do_rebuild: false,
                    })
                })
                .collect();
            // The snapshot is discarded before any group can be merged.
            let adjacency = sink.container_to_value(VecContainer {
                data,
                do_rebuild: false,
            });
            sink.set(target, RawValues(Vec::new()), adjacency)?;
            Ok(())
        })?;
    }
    Ok(())
}

pub(super) fn run(graph: &mut EGraph) -> Result<(), OptimizeError> {
    graph.update(|mut sink| sink.add("PlanningRound", 0i64))?;
    let dependencies = parse_program(
        "fusion dependencies",
        "(run-schedule (saturate fusion-dependencies))",
    )?;
    let choose = parse_program(
        "fusion candidates",
        "(run-schedule (saturate fusion-counts) (saturate fusion-candidates) (saturate fusion-select) fusion-choose)",
    )?;
    let commit = parse_program("fusion contraction", "(run-schedule (saturate fusion-record) (saturate fusion-commit) fusion-clear-candidates fusion-union fusion-clear)")?;
    loop {
        timing::time("egglog fusion / dependencies", || {
            graph.run_program(dependencies.clone())
        })?;
        timing::time("egglog fusion / adjacency", || snapshot(graph))?;
        timing::time("egglog fusion / choose", || graph.run_program(choose.clone()))?;
        clear_snapshot(graph)?;
        let mut chosen = false;
        graph.constructor_enodes("Chosen", |_| chosen = true)?;
        if !chosen {
            break;
        }
        timing::time("egglog fusion / commit", || graph.run_program(commit.clone()))?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "reachability_tests.rs"]
mod tests;
