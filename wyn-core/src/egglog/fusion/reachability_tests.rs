#[test]
fn native_paths_distinguish_direct_edges_barriers_and_contractions() {
    let mut graph = super::super::new_graph().unwrap();
    graph
        .parse_and_run_program(
            None,
            r#"
        (let a (Group (OperationId 0)))
        (let b (Group (OperationId 1)))
        (let c (Group (OperationId 2)))
        (let d (Group (OperationId 3)))
        (GroupEdge a c)
        (GroupBefore a b)
        (GroupBefore b c)
    "#,
        )
        .unwrap();
    super::snapshot(&mut graph).unwrap();
    graph
        .parse_and_run_program(
            None,
            r#"
        (check (= (wyn-order-intermediate (FusionPathOrder) a c) true))
        (check (= (wyn-group-path (FusionPathEdges) a c) true))
        (check (= (wyn-group-path (FusionPathEdges) c a) false))
        (check (= (wyn-group-path (FusionPathEdges) a d) false))
    "#,
        )
        .unwrap();
    super::clear_snapshot(&mut graph).unwrap();
    graph.parse_and_run_program(None, "(union a b)").unwrap();
    super::snapshot(&mut graph).unwrap();
    graph
        .parse_and_run_program(
            None,
            r#"
        (check (= (wyn-order-intermediate (FusionPathOrder) a c) false))
    "#,
        )
        .unwrap();
    super::clear_snapshot(&mut graph).unwrap();
    graph.parse_and_run_program(None, "(GroupBefore c d) (GroupBefore d c)").unwrap();
    super::snapshot(&mut graph).unwrap();
    graph
        .parse_and_run_program(
            None,
            r#"
        (check (= (wyn-order-intermediate (FusionPathOrder) a c) true))
        (check (= (wyn-order-intermediate (FusionPathOrder) c a) false))
    "#,
        )
        .unwrap();
}

#[test]
fn round_snapshots_follow_contractions_and_stay_with_their_graph() {
    let mut graph = super::super::new_graph().unwrap();
    graph
        .parse_and_run_program(
            None,
            r#"
        (let a (Group (OperationId 0)))
        (let b (Group (OperationId 1)))
        (let c (Group (OperationId 2)))
        (let d (Group (OperationId 3)))
        (GroupEdge a b)
        (GroupBefore a b)
        (GroupBefore b c)
    "#,
        )
        .unwrap();
    super::snapshot(&mut graph).unwrap();
    graph
        .parse_and_run_program(
            None,
            "(check (= (wyn-group-path (FusionPathEdges) a c) false))
             (check (= (wyn-order-intermediate (FusionPathOrder) a c) true))",
        )
        .unwrap();

    let mut other = graph.clone();
    super::clear_snapshot(&mut other).unwrap();
    other.parse_and_run_program(None, "(GroupEdge b c) (union a b)").unwrap();
    super::snapshot(&mut other).unwrap();
    other
        .parse_and_run_program(
            None,
            "(check (= (wyn-group-path (FusionPathEdges) a c) true))
             (check (= (wyn-order-intermediate (FusionPathOrder) a c) false))",
        )
        .unwrap();
    graph
        .parse_and_run_program(
            None,
            "(check (= (wyn-group-path (FusionPathEdges) a c) false))
             (check (= (wyn-order-intermediate (FusionPathOrder) a c) true))",
        )
        .unwrap();

    super::clear_snapshot(&mut other).unwrap();
    other.parse_and_run_program(None, "(GroupBefore c d) (GroupBefore d c)").unwrap();
    super::snapshot(&mut other).unwrap();
    other
        .parse_and_run_program(
            None,
            "(check (= (wyn-order-intermediate (FusionPathOrder) a c) true))",
        )
        .unwrap();
}
