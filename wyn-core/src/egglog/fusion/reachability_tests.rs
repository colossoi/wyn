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
        (check (= (wyn-order-intermediate 0 a c) true))
        (check (= (wyn-group-path 0 a c) true))
        (check (= (wyn-group-path 0 c a) false))
        (check (= (wyn-group-path 0 a d) false))
        (union a b)
        (check (= (wyn-order-intermediate 1 a c) false))
        (GroupBefore c d)
        (GroupBefore d c)
        (check (= (wyn-order-intermediate 2 a c) true))
        (check (= (wyn-order-intermediate 2 c a) false))
    "#,
        )
        .unwrap();
}
