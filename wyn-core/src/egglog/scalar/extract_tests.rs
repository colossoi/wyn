use super::*;

#[test]
fn trimming_unused_identities_preserves_extraction_and_nested_source_arguments() {
    let mut graph = EGraph::default();
    graph
        .parse_and_run_program(
            None,
            concat!(
                include_str!("../ids.egg"),
                r#"
                (datatype TypeKey (TypeId i64))
                (datatype FusionPlan (FusionSource OperationKey))
                (sort SourceArguments (Vec SourceValue))
                (constructor SourceInvocation (RegionKey SourceValue SourceArguments) SourceValue)
                "#,
                include_str!("schema.egg"),
                r#"
                (let $ctx (ExprContextFunction (RegionId 0)))
                (let $ty (TypeId 0))
                (let $used (SourceTerm (RegionId 0) 1))
                (let $argument (SourceTerm (RegionId 0) 2))
                (let $call (SourceInvocation (RegionId 0) (SourceGlobal 1) (vec-of $argument)))
                (ExprChoice $ctx $ty (ExprLiteral $ctx $ty "true")
                  (ExprLeaf $ctx $ty $used) (ExprLeaf $ctx $ty $call) (SourceFormal (RegionId 0) 3))
                (SourceTerm (RegionId 99) 99)
                (SourceFormal (RegionId 99) 99)
                (SourceGlobal 99)
                "#,
            ),
        )
        .unwrap();
    let mut root = None;
    graph.constructor_enodes("ExprChoice", |row| root = Some(row.eclass)).unwrap();
    let root = root.unwrap();
    let extract = |graph: &EGraph| {
        let extractor = Extractor::compute_costs_from_rootsorts(
            Some(vec![graph.get_sort_by_name("Expr").unwrap().clone()]),
            graph,
            Model {
                prefer_inlining: false,
            },
        );
        let mut dag = TermDag::default();
        let (_, term) = extractor.extract_best(graph, &mut dag, root).unwrap();
        dag.to_string(term)
    };
    let before = extract(&graph);
    trim_metadata(&mut graph).unwrap();
    assert_eq!(extract(&graph), before);
    let mut retained = 0;
    let mut removed = 0;
    for name in ["SourceTerm", "SourceFormal", "SourceGlobal"] {
        graph
            .constructor_enodes(name, |row| {
                if row.subsumed {
                    removed += 1;
                } else {
                    retained += 1;
                }
            })
            .unwrap();
    }
    assert_eq!(removed, 3, "only the unreferenced identity rows are trimmed");
    assert_eq!(
        retained, 4,
        "keep source, nested argument, callable, and branch identity"
    );
}
