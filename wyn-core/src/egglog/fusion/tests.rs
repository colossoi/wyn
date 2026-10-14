#[test]
fn source_use_sets_survive_group_initialization_and_merging() {
    let source = crate::compile_thru_tlc("entry main(x:i32) i32 = x").unwrap();
    let (mut graph, _) = crate::egglog::source::import(&source).unwrap();
    crate::egglog::analysis::run(&mut graph).unwrap();
    let mut facts = String::new();
    for id in 1000..1258 {
        facts.push_str(&format!(
            "(SourceOperation (OperationId {id}) (RegionId 1000) \"control\")\n\
             (set (SourceInput (OperationId {id}) 0) (SourceGlobal 1000))\n"
        ));
    }
    facts.push_str("(set (SourceExtent (SourceGlobal 1000)) (Fixed 64))\n");
    for consumer in 1002..1258 {
        let producer = 1000 + consumer % 2;
        facts.push_str(&format!(
            "(SourceDepends (OperationId {consumer}) (OperationId {producer}) (Input))\n\
             (SourceDepends (OperationId {consumer}) (OperationId {producer}) (LengthUse))\n"
        ));
    }
    graph.parse_and_run_program(None, &facts).unwrap();
    graph
        .parse_and_run_program(
            None,
            r#"
            (run-schedule (saturate source-fusion))
            (check (= (set-length (GroupUsesOf (Group (OperationId 1000)))) 256))
            (check (= (set-length (GroupLengths (Group (OperationId 1000)))) 128))
            (check (= (set-length (GroupUsesOf (Group (OperationId 1002)))) 0))
            (union (Group (OperationId 1000)) (Group (OperationId 1001)))
            (check (= (set-length (GroupUsesOf (Group (OperationId 1000)))) 512))
            (check (= (set-length (GroupLengths (Group (OperationId 1000)))) 256))
            (check (set-contains (GroupUsesOf (Group (OperationId 1000)))
                                 (UseAt (OperationId 1003) (Input))))
            "#,
        )
        .unwrap();
}
