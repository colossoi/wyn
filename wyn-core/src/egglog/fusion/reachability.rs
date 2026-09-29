//! Candidate-local legality on the rebuilt quotient graph. Never retain a
//! transitive closure or sets of intermediate operations in the e-graph.
use crate::egglog::{parse_program, OptimizeError};
use egglog_engine::{EGraph, Value, Write};
use std::collections::HashMap;

#[derive(Default)]
struct Graph {
    ids: HashMap<Value, usize>,
    forward: Vec<Vec<usize>>,
    reverse: Vec<Vec<usize>>,
}
impl Graph {
    fn node(&mut self, value: Value) -> usize {
        let next = self.ids.len();
        *self.ids.entry(value).or_insert_with(|| {
            self.forward.push(Vec::new());
            self.reverse.push(Vec::new());
            next
        })
    }
    fn edge(&mut self, a: Value, b: Value) {
        let a = self.node(a);
        let b = self.node(b);
        if a != b {
            self.forward[a].push(b);
            self.reverse[b].push(a);
        }
    }
    fn read(graph: &EGraph, name: &str) -> Result<Self, OptimizeError> {
        let mut out = Self::default();
        graph.constructor_enodes(name, |row| out.edge(row.children[0], row.children[1]))?;
        Ok(out)
    }
    fn visit(edges: &[Vec<usize>], start: usize) -> Vec<bool> {
        let mut seen = vec![false; edges.len()];
        let mut todo = vec![start];
        while let Some(n) = todo.pop() {
            if std::mem::replace(&mut seen[n], true) {
                continue;
            }
            todo.extend(edges[n].iter().copied());
        }
        seen
    }
    fn reachable(&self, a: Value, b: Value) -> bool {
        let (Some(&a), Some(&b)) = (self.ids.get(&a), self.ids.get(&b)) else {
            return false;
        };
        Self::visit(&self.forward, a)[b]
    }
    fn between(&self, a: Value, b: Value) -> bool {
        let (Some(&a), Some(&b)) = (self.ids.get(&a), self.ids.get(&b)) else {
            return false;
        };
        let from_a = Self::visit(&self.forward, a);
        if !from_a[b] {
            return false;
        }
        let to_b = Self::visit(&self.reverse, b);
        // Endpoints are permitted to have self edges after contraction. An
        // external node on *any* path is still a barrier, including in cycles.
        from_a.iter().zip(to_b).enumerate().any(|(i, (&f, r))| i != a && i != b && f && r)
    }
}

fn refresh(graph: &mut EGraph) -> Result<(), OptimizeError> {
    let order = Graph::read(graph, "GroupBefore")?;
    let data = Graph::read(graph, "GroupEdge")?;
    let mut pairs = Vec::new();
    graph.constructor_enodes("RelevantPair", |row| {
        if row.children[0] != row.children[1] {
            pairs.push((row.children[0], row.children[1]));
        }
    })?;
    for table in ["OrderClear", "OrderBlocked", "DataIndependent"] {
        graph.clear_function(table)?;
    }
    graph.update(|mut sink| -> Result<(), egglog_engine::Error> {
        for &(p, c) in &pairs {
            sink.add(
                if order.between(p, c) { "OrderBlocked" } else { "OrderClear" },
                (p, c),
            )?;
            if !data.reachable(p, c) && !data.reachable(c, p) {
                sink.add("DataIndependent", (p, c))?;
            }
        }
        Ok(())
    })?;
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
        "(run-schedule (saturate fusion-candidates) (saturate fusion-select) fusion-choose)",
    )?;
    let commit = parse_program("fusion contraction", "(run-schedule (saturate fusion-record) (saturate fusion-commit) fusion-clear-candidates fusion-union fusion-clear)")?;
    loop {
        graph.run_program(dependencies.clone())?;
        refresh(graph)?;
        graph.run_program(choose.clone())?;
        let mut chosen = false;
        graph.constructor_enodes("Chosen", |_| chosen = true)?;
        if !chosen {
            break;
        }
        graph.run_program(commit.clone())?;
    }
    graph.run_program(parse_program(
        "fusion indexed schedule",
        include_str!("schedule.egg"),
    )?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use egglog_engine::Core;

    #[test]
    fn candidate_checks_match_closure_for_every_four_node_graph() {
        let database = EGraph::default();
        let ids: Vec<Value> = database.read(|r| (0..4i64).map(|n| r.base_to_value(n)).collect());
        let edges: Vec<_> =
            (0..4).flat_map(|a| (0..4).filter(move |&b| a != b).map(move |b| (a, b))).collect();
        // Includes cycles, diamonds, disconnected nodes, and alternate paths.
        for mask in 0..(1 << edges.len()) {
            let mut graph = Graph::default();
            let mut closure = [[false; 4]; 4];
            for (i, &(a, b)) in edges.iter().enumerate() {
                if mask & (1 << i) != 0 {
                    graph.edge(ids[a], ids[b]);
                    closure[a][b] = true;
                }
            }
            for k in 0..4 {
                for a in 0..4 {
                    for b in 0..4 {
                        closure[a][b] |= closure[a][k] && closure[k][b];
                    }
                }
            }
            for a in 0..4 {
                for b in 0..4 {
                    if a == b {
                        continue;
                    }
                    assert_eq!(graph.reachable(ids[a], ids[b]), closure[a][b]);
                    assert_eq!(
                        graph.between(ids[a], ids[b]),
                        (0..4).any(|m| m != a && m != b && closure[a][m] && closure[m][b]),
                        "mask={mask}, pair=({a},{b})"
                    );
                }
            }
        }
    }

    #[test]
    fn contraction_recomputes_order_and_keeps_effect_paths() {
        let mut graph = super::super::new_graph().unwrap();
        graph.parse_and_run_program(None, include_str!("fusion.egg")).unwrap();
        graph
            .parse_and_run_program(
                None,
                r#"
            (let a (Group (OperationId 0)))
            (let b (Group (OperationId 1)))
            (let c (Group (OperationId 2)))
            (GroupEdge a b)
            (GroupBefore b c)
            (RelevantPair a c)
            (run-schedule (saturate fusion-dependencies))
        "#,
            )
            .unwrap();
        refresh(&mut graph).unwrap();
        graph.parse_and_run_program(None, "(check (OrderBlocked a c) (DataIndependent a c)) (union a b) (run-schedule (saturate fusion-dependencies))").unwrap();
        refresh(&mut graph).unwrap();
        graph
            .parse_and_run_program(
                None,
                "(check (OrderClear a c) (DataIndependent a c)) (fail (check (OrderBlocked a c)))",
            )
            .unwrap();
    }
}
