use egglog_engine::EGraph;

fn settle(graph: &mut EGraph, round: i64) {
    graph.parse_and_run_program(None, &format!(
        "(PlanningRound {round}) (RelevantPair a c) (RelevantPair c a) (RelevantPair a d) (run-schedule (saturate fusion-paths))"
    )).unwrap();
}
fn check(graph: &mut EGraph, round: i64, facts: &str) {
    graph.parse_and_run_program(None, &facts.replace("ROUND", &round.to_string())).unwrap();
}

#[test]
fn native_paths_distinguish_direct_edges_barriers_and_contractions() {
    let mut graph = super::new_graph().unwrap();
    graph
        .parse_and_run_program(
            None,
            r#"
        (let a (Group (OperationId 0))) (let b (Group (OperationId 1)))
        (let c (Group (OperationId 2))) (let d (Group (OperationId 3)))
        (GroupEdge a c) (GroupBefore a b) (GroupBefore b c)
    "#,
        )
        .unwrap();
    settle(&mut graph, 0);
    check(
        &mut graph,
        0,
        r#"
        (check (= (OrderIntermediate ROUND a c) true))
        (check (= (GroupPath ROUND a c) true))
        (check (= (GroupPath ROUND c a) false))
        (check (= (GroupPath ROUND a d) false))
    "#,
    );
    graph.parse_and_run_program(None, "(delete (PlanningRound 0)) (union a b)").unwrap();
    settle(&mut graph, 1);
    check(&mut graph, 1, "(check (= (OrderIntermediate ROUND a c) false))");
    graph
        .parse_and_run_program(
            None,
            "(delete (PlanningRound 1)) (GroupBefore c d) (GroupBefore d c)",
        )
        .unwrap();
    settle(&mut graph, 2);
    check(
        &mut graph,
        2,
        r#"
        (check (= (OrderIntermediate ROUND a c) true))
        (check (= (OrderIntermediate ROUND c a) false))
    "#,
    );
}

#[test]
fn native_paths_follow_contractions_and_stay_with_their_graph() {
    let mut graph = super::new_graph().unwrap();
    graph
        .parse_and_run_program(
            None,
            r#"
        (let a (Group (OperationId 0))) (let b (Group (OperationId 1)))
        (let c (Group (OperationId 2))) (let d (Group (OperationId 3)))
        (GroupEdge a b) (GroupBefore a b) (GroupBefore b c)
    "#,
        )
        .unwrap();
    settle(&mut graph, 0);
    let original = "(check (= (GroupPath ROUND a c) false)) (check (= (OrderIntermediate ROUND a c) true))";
    check(&mut graph, 0, original);
    let mut other = graph.clone();
    other.parse_and_run_program(None, "(delete (PlanningRound 0)) (GroupEdge b c) (union a b)").unwrap();
    settle(&mut other, 1);
    check(
        &mut other,
        1,
        "(check (= (GroupPath ROUND a c) true)) (check (= (OrderIntermediate ROUND a c) false))",
    );
    check(&mut graph, 0, original);
    other
        .parse_and_run_program(
            None,
            "(delete (PlanningRound 1)) (GroupBefore c d) (GroupBefore d c)",
        )
        .unwrap();
    settle(&mut other, 2);
    check(&mut other, 2, "(check (= (OrderIntermediate ROUND a c) true))");
}
