use super::*;
use crate::LookupMap;
use std::collections::BTreeSet;
use wyn_graph::topo_sort_by_dependencies;

const SCHEMA: &str = r#"
    (datatype OperationKey (OperationId i64))
    (datatype RegionKey (RegionId i64))
    (relation PlannedSourcePosition (OperationKey RegionKey i64))
    (relation OperationDependency (OperationKey OperationKey))
    (relation EffectInput (i64 OperationKey))
    (relation EffectWait (OperationKey i64))
    (relation OperationOrderPosition (OperationKey RegionKey i64))
    (relation Consecutive (OperationKey OperationKey))
    (relation OrderPending (OperationKey RegionKey i64))
    (ruleset order-import) (ruleset order-select) (ruleset order-emit)
"#;

type Output = (BTreeSet<(i64, i64, i64)>, BTreeSet<(i64, i64)>);

fn graph(facts: &str) -> EGraph {
    let mut graph = EGraph::default();
    graph.parse_and_run_program(None, SCHEMA).unwrap();
    graph.parse_and_run_program(None, facts).unwrap();
    graph
}

fn output(graph: &EGraph) -> Output {
    let query = Query(graph);
    let id = |name, value| graph.value_to_base::<i64>(query.enode(name, value).unwrap().unwrap()[0]);
    let mut positions = BTreeSet::new();
    query
        .for_each("OperationOrderPosition", |row| {
            positions.insert((
                id("RegionId", row[1]),
                id("OperationId", row[0]),
                graph.value_to_base::<i64>(row[2]),
            ));
            Ok(())
        })
        .unwrap();
    let mut consecutive = BTreeSet::new();
    query
        .for_each("Consecutive", |row| {
            consecutive.insert((id("OperationId", row[0]), id("OperationId", row[1])));
            Ok(())
        })
        .unwrap();
    (positions, consecutive)
}

fn compare(facts: &str, cycle: bool) -> Output {
    let mut reference = graph(facts);
    assert_eq!(reference_order(&mut reference).is_err(), cycle);
    let mut native = graph(facts);
    native
        .parse_and_run_program(
            None,
            include_str!("dispatch.egg")
                .split("; Select a dependency order within each invocation")
                .nth(1)
                .unwrap()
                .split_once("\n")
                .unwrap()
                .1,
        )
        .unwrap();
    let result = order(&mut native);
    assert_eq!(result.is_err(), cycle);
    if cycle {
        assert!(result.unwrap_err().to_string().contains("cycle in the fused execution graph"));
    } else {
        assert_eq!(output(&native), output(&reference));
    }
    output(&native)
}

#[test]
fn order_preserves_ready_priority_effects_and_region_boundaries() {
    let actual = compare(
        r#"
        (PlannedSourcePosition (OperationId 0) (RegionId 0) 0)
        (PlannedSourcePosition (OperationId 1) (RegionId 0) 1)
        (PlannedSourcePosition (OperationId 2) (RegionId 0) 2)
        (PlannedSourcePosition (OperationId 3) (RegionId 0) 3)
        (PlannedSourcePosition (OperationId 4) (RegionId 1) 0)
        (OperationDependency (OperationId 0) (OperationId 1))
        (OperationDependency (OperationId 0) (OperationId 4))
        (OperationDependency (OperationId 1) (OperationId 4))
        (OperationDependency (OperationId 4) (OperationId 0))
        (EffectInput 10 (OperationId 2))
        (EffectInput 11 (OperationId 1))
        (EffectWait (OperationId 1) 10)
        (EffectWait (OperationId 0) 11)
    "#,
        false,
    );
    assert_eq!(
        actual.0,
        [(0, 0, 2), (0, 1, 1), (0, 2, 0), (0, 3, 3), (1, 4, 0)].into_iter().collect()
    );
    assert_eq!(actual.1, [(2, 1), (1, 0), (0, 3)].into_iter().collect());
}

#[test]
fn order_matches_reference_for_generated_dags() {
    compare("", false);
    for seed in 0..8 {
        let mut facts = String::new();
        for region in 0..2 {
            for node in 0..12 {
                let op = region * 12 + node;
                let position = (node * 5 + seed) % 12;
                facts +=
                    &format!("(PlannedSourcePosition (OperationId {op}) (RegionId {region}) {position})\n");
                for before in 0..node {
                    let dependency = region * 12 + before;
                    if (node * 7 + before * 3 + seed) % 5 == 0 {
                        facts += &format!(
                            "(OperationDependency (OperationId {op}) (OperationId {dependency}))\n"
                        );
                    }
                    if (node * 3 + before * 7 + seed) % 11 == 0 {
                        let gate = op * 12 + before;
                        facts += &format!("(EffectInput {gate} (OperationId {dependency})) (EffectWait (OperationId {op}) {gate})\n");
                    }
                }
            }
        }
        compare(&facts, false);
    }
}

#[test]
fn order_rejects_dependency_and_effect_cycles() {
    let positions = "(PlannedSourcePosition (OperationId 0) (RegionId 0) 0)
                     (PlannedSourcePosition (OperationId 1) (RegionId 0) 1)";
    for cycle in [
        "(OperationDependency (OperationId 0) (OperationId 0))",
        "(OperationDependency (OperationId 0) (OperationId 1)) (OperationDependency (OperationId 1) (OperationId 0))",
        "(OperationDependency (OperationId 0) (OperationId 1)) (EffectInput 0 (OperationId 0)) (EffectWait (OperationId 1) 0)",
    ] {
        compare(&format!("{positions} {cycle}"), true);
    }
}

fn reference_order(graph: &mut EGraph) -> Result<(), OptimizeError> {
    let query = Query(graph);
    let mut regions = LookupMap::<_, Vec<_>>::default();
    query.for_each("PlannedSourcePosition", |row| {
        regions.entry(row[1]).or_default().push((graph.value_to_base::<i64>(row[2]), row[0]));
        Ok(())
    })?;
    let mut dependencies = LookupMap::<_, Vec<_>>::default();
    query.for_each("OperationDependency", |row| {
        dependencies.entry(row[0]).or_default().push(row[1]);
        Ok(())
    })?;
    let mut effects = LookupMap::<_, Vec<_>>::default();
    query.for_each("EffectInput", |row| {
        effects.entry(row[0]).or_default().push(row[1]);
        Ok(())
    })?;
    query.for_each("EffectWait", |row| {
        dependencies.entry(row[0]).or_default().extend(effects.get(&row[1]).into_iter().flatten());
        Ok(())
    })?;
    let mut orders = Vec::new();
    for (region, mut positions) in regions {
        positions.sort();
        // The supplied universe confines dependency/effect edges to this region.
        let order = topo_sort_by_dependencies(positions.into_iter().map(|(_, op)| op), |op, out| {
            out.extend(dependencies.get(&op).into_iter().flatten().copied());
        })
        .map_err(|_| OptimizeError::Extraction("cycle in the fused execution graph".into()))?;
        orders.push((region, order));
    }
    graph.update(|mut sink| {
        for (region, order) in &orders {
            for (position, &op) in order.iter().enumerate() {
                sink.add("OperationOrderPosition", (op, *region, position as i64))?;
            }
            for pair in order.windows(2) {
                sink.add("Consecutive", (pair[0], pair[1]))?;
            }
        }
        Ok::<_, egglog_engine::Error>(())
    })?;
    Ok(())
}
