use super::*;
use std::collections::BTreeSet;

const SCHEMA: &str = r#"
    (datatype OperationKey (OperationId i64))
    (datatype RegionKey (RegionId i64))
    (relation PlannedSourcePosition (OperationKey RegionKey i64))
    (relation OperationDependency (OperationKey OperationKey))
    (relation EffectInput (i64 OperationKey))
    (relation EffectWait (OperationKey i64))
    (relation OperationOrderPosition (OperationKey RegionKey i64))
    (relation Consecutive (OperationKey OperationKey))
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
    reference.parse_and_run_program(None, include_str!("order_reference.egg")).unwrap();
    let mut pending = false;
    Query(&reference)
        .for_each("OrderPending", |_| {
            pending = true;
            Ok(())
        })
        .unwrap();
    assert_eq!(pending, cycle);
    let mut native = graph(facts);
    let result = apply(&mut native);
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
