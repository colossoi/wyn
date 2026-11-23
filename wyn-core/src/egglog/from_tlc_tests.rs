use super::import;
use crate::compile_thru_tlc;
use crate::egglog::dataflow::ir::Code;
use crate::egglog::graph::Graph;
use crate::egglog::query::Query;
use crate::tlc::{self, DefMeta, TermKind};
use egglog_engine::Value;

fn function_body(graph: &Graph) -> Vec<Value> {
    let query = Query(graph);
    let entry = query.first("EntryOrder").unwrap().unwrap()[0];
    query.operations(query.required("FunctionRoot", entry).unwrap()).unwrap()
}

#[test]
fn evaluation_edges_stay_direct_and_borrowed_values_do_not_repeat_effects() {
    for depth in [16, 64, 256] {
        let mut program = compile_thru_tlc("entry main(x: i32) i32 = x + 1").unwrap();
        let ids: Vec<_> = (0..depth).map(|_| program.next_term_id()).collect();
        let entry = program.defs.iter_mut().find(|def| matches!(def.meta, DefMeta::EntryPoint(_))).unwrap();
        let TermKind::Lambda(lambda) = &mut entry.body.kind else {
            panic!("entry lambda")
        };
        for &id in &ids {
            *lambda.body = tlc::Term {
                id,
                ty: lambda.body.ty.clone(),
                span: lambda.body.span,
                kind: TermKind::Coerce {
                    inner: lambda.body.clone(),
                    target_ty: lambda.body.ty.clone(),
                },
            };
        }
        let (mut graph, _) = import(&program).unwrap();
        crate::egglog::analysis::run(&mut graph).unwrap();
        let body = function_body(&graph);
        let statements = body;
        assert!(
            statements.len() <= 3,
            "{depth} coercions must not repeat scalar evaluation"
        );
        let mut additions = 0;
        for statement in statements {
            if let Code::Scalar(payload) = graph.code(statement).unwrap() {
                additions += usize::from(payload.spelling() == "+");
            }
        }
        assert_eq!(additions, 1);
    }
}

#[test]
fn indexed_write_effects_distinguish_owned_arrays_from_views() {
    for (ty, readonly) in [("[4]i32", true), ("[]i32", false)] {
        let source = format!("entry main(xs:{ty}) {ty} = scatter(xs,[0i32],[7i32])");
        let program = compile_thru_tlc(&source).unwrap();
        let (mut graph, _) = import(&program).unwrap();
        crate::egglog::analysis::run(&mut graph).unwrap();
        let query = Query(&graph);
        let body = function_body(&graph);
        let mut copies = 0;
        for statement in body {
            copies += usize::from(query.contains("MaterializedCopyOperation", statement).unwrap());
        }
        assert_eq!(
            copies > 0,
            readonly,
            "owned input is copied before writes; views preserve storage identity"
        );
    }
}

#[test]
fn repeated_free_references_keep_dependencies_in_sibling_and_nested_regions() {
    for (ty, producer, yes, no, step, kind) in [
        (
            "i32",
            "reduce((+),0,xs)",
            "alias+alias+produced",
            "produced+produced",
            "acc+produced+produced",
            "reduce",
        ),
        (
            "[]i32",
            "map(|x|x+1,xs)",
            "map(|(x,y)|x+y,zip(alias,produced))",
            "map(|(x,y)|x-y,zip(produced,produced))",
            "map(|(x,y)|x+y,zip(acc,produced))",
            "map",
        ),
    ] {
        let source = format!(
            "entry main(xs:[]i32,n:i32,choose:bool,other:bool) {ty} =
             let produced={producer} in
             if choose then let alias=produced in if other then {yes} else {no}
             else loop acc=produced for i<n do {step}"
        );
        let program = compile_thru_tlc(&source).unwrap();
        let (mut graph, _) = import(&program).unwrap();
        crate::egglog::analysis::run(&mut graph).unwrap();
        let query = Query(&graph);
        let body = function_body(&graph);
        let statements = body;
        let producer = *statements
            .iter()
            .find(|&&operation| query.contains("ParallelOperation", operation).unwrap())
            .expect("outer producer");
        let names = query.sequence("Output", producer).unwrap();
        assert!(!names.is_empty(), "{kind} producer has explicit result bindings");
        let mut captures = 0;
        query
            .for_each("LiveOperation", |row| {
                if query.contains("BranchOperation", row[0])?
                    || query.contains("CountedLoopOperation", row[0])?
                {
                    captures += usize::from(
                        names.iter().any(|&name| query.contains("OperationUse", (row[0], name)).unwrap()),
                    );
                }
                Ok(())
            })
            .unwrap();
        assert!(captures >= 2, "both control paths retain the producer dependency");
    }
}

#[test]
fn scatter_reads_include_branch_dependencies_without_turning_them_into_operand_uses() {
    let program = compile_thru_tlc(
        "entry main(xs:[]i32,ys:[]i32,choose:bool) []i32 =
         let selected=if choose then xs[0] else ys[0] in scatter(xs,[0],[selected])",
    )
    .unwrap();
    let (mut graph, _) = import(&program).unwrap();
    crate::egglog::analysis::run(&mut graph).unwrap();
    let query = Query(&graph);
    let body = function_body(&graph);
    let mut branch_reads = 0;
    for statement in body {
        if query.contains("BranchOperation", statement).unwrap() {
            for edge in 0_i64..2 {
                let arm = query.required("ControlRegion", (statement, edge)).unwrap();
                assert!(query.any("RegionCapture", |row| row[0] == arm).unwrap());
                branch_reads += 1;
            }
        }
    }
    assert_eq!(branch_reads, 2);
}

// Product projection is checked in memory_ir_product_projection.wyn.
#[test]
fn runtime_maps_preserve_collection_until_shape_analysis() {
    for source in [
        "entry main(xs:[]i32,n:i32) [][]i32=map(|x|map(|y|x+y,iota(n)),xs)",
        "entry main(xs:[]i32,n:i32,m:i32) [][][]i32=map(|x|map(|y|map(|z|x+y+z,iota(m)),iota(n)),xs)",
        "entry main(xs:[]i32,n:i32,b:bool) [][]i32=if b then map(|x|map(|y|x+y,iota(n)),xs) else map(|x|map(|y|x-y,iota(n)),xs)",
    ] {
        let program = compile_thru_tlc(source).unwrap();
        let (mut graph, _) = import(&program).unwrap();
        crate::egglog::analysis::run(&mut graph).unwrap();
        let query = Query(&graph);
        let mut loops = 0;
        let mut maps = 0;
        query.for_each("LiveOperation", |row| {
            loops += usize::from(query.contains("CountedLoopOperation", row[0])? || query.contains("WhileLoopOperation",row[0])?);
            maps += usize::from(query.contains("ParallelOperation", row[0])?);
            Ok(())
        }).unwrap();
        assert_eq!(loops, 0, "uniform shape must not require a growable loop: {source}");
        assert!(maps >= 2, "nested maps must survive import: {source}");
    }
}
