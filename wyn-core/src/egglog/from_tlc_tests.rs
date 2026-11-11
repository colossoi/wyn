use super::import;
use crate::egglog::query::Query;
use crate::tlc::{self, DefMeta, TermKind};
use crate::{compile_thru_tlc, types};
use egglog_engine::{EGraph, Value};

fn function_body(graph: &EGraph) -> Vec<Value> {
    let query = Query(graph);
    let root = query.row("CurrentProgram", |_| true).unwrap().unwrap()[0];
    let program = query.fields("MemoryProgram", root).unwrap();
    let function = query.vector(program[0]).unwrap()[0];
    query.fields("Block", query.fields("DefineFunction", function).unwrap()[3]).unwrap()
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
        let statements = Query(&graph).vector(body[0]).unwrap();
        assert!(
            statements.len() <= 3,
            "{depth} coercions must not repeat scalar evaluation"
        );
        let mut additions = 0;
        for statement in statements {
            let fields = Query(&graph).fields("Let", statement).unwrap();
            if let Some(op) = Query(&graph).enode("PrimOp", fields[3]).unwrap() {
                if graph.value_to_base::<egglog_engine::sort::S>(op[0]).as_str() == "+" {
                    additions += 1;
                }
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
        for statement in query.vector(body[0]).unwrap() {
            let statement = query.fields("Let", statement).unwrap();
            if query.enode("Copy", statement[3]).unwrap().is_some() {
                copies += 1;
            }
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
        let statements = query.vector(body[0]).unwrap();
        let producer = statements
            .iter()
            .find_map(|&statement| {
                let statement = query.fields("Let", statement).unwrap();
                query.enode("Parallel", statement[3]).unwrap().map(|_| statement[1])
            })
            .expect("outer producer");
        let bindings = query.required("BindersPattern", (producer,)).unwrap();
        let names = graph.value_to_container::<egglog_engine::sort::SetContainer>(bindings).unwrap();
        assert!(
            !names.data.is_empty(),
            "{kind} producer has explicit result bindings"
        );
        let mut captures = 0;
        query
            .for_each("CheckStatement", |row| {
                if query.enode("If", row[3])?.is_some() || query.enode("Loop", row[3])?.is_some() {
                    let uses = query.required("NamesExp", (row[3],))?;
                    let uses = graph.value_to_container::<egglog_engine::sort::SetContainer>(uses).unwrap();
                    if names.data.iter().any(|name| uses.data.contains(name)) {
                        captures += 1;
                    }
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
    for statement in query.vector(body[0]).unwrap() {
        let statement = query.fields("Let", statement).unwrap();
        if let Some(branch) = query.enode("If", statement[3]).unwrap() {
            for arm in [branch[2], branch[3]] {
                let names = query.required("NamesBody", (arm,)).unwrap();
                let names = graph.value_to_container::<egglog_engine::sort::SetContainer>(names).unwrap();
                assert!(!names.data.is_empty());
                branch_reads += 1;
            }
        }
    }
    assert_eq!(branch_reads, 2);
}

#[test]
fn source_import_resolves_coerced_tuple_fields_in_their_visible_scope() {
    let mut program = compile_thru_tlc("entry main(x: i32) i32 = x + 1").unwrap();
    let [literal_id, tuple_id, coerced_id, projected_id] = std::array::from_fn(|_| program.next_term_id());
    let entry = program.defs.iter_mut().find(|def| matches!(def.meta, DefMeta::EntryPoint(_))).unwrap();
    let TermKind::Lambda(lambda) = &mut entry.body.kind else {
        panic!("entry lambda")
    };
    let scalar = (*lambda.body).clone();
    let ty = scalar.ty.clone();
    let span = scalar.span;
    let tuple_ty = types::Type::Constructed(types::TypeName::Tuple(2), vec![ty.clone(), ty.clone()]);
    let other = tlc::Term {
        id: literal_id,
        ty: ty.clone(),
        span,
        kind: TermKind::IntLit("7".into()),
    };
    let tuple = tlc::Term {
        id: tuple_id,
        ty: tuple_ty.clone(),
        span,
        kind: TermKind::Tuple(vec![other, scalar]),
    };
    let coerced = tlc::Term {
        id: coerced_id,
        ty: tuple_ty.clone(),
        span,
        kind: TermKind::Coerce {
            inner: Box::new(tuple),
            target_ty: tuple_ty,
        },
    };
    let projected = tlc::Term {
        id: projected_id,
        ty,
        span,
        kind: TermKind::TupleProj {
            tuple: Box::new(coerced),
            idx: 1,
        },
    };
    *lambda.body = projected;
    let (mut graph, _) = import(&program).unwrap();
    crate::egglog::analysis::run(&mut graph).unwrap();
    let query = Query(&graph);
    let body = function_body(&graph);
    let results = query.vector(body[1]).unwrap();
    let result = query.fields("Yield", results[0]).unwrap();
    let atom = query.fields("ValueOperand", result[1]).unwrap()[0];
    let name = query.fields("Var", atom).unwrap()[0];
    let statements = query.vector(body[0]).unwrap();
    let producer = statements
        .into_iter()
        .find(|&statement| {
            let statement = query.fields("Let", statement).unwrap();
            query
                .vector(statement[1])
                .unwrap()
                .into_iter()
                .any(|binding| query.fields("BindValue", binding).unwrap()[0] == name)
        })
        .unwrap();
    let expression = query.fields("Let", producer).unwrap()[3];
    assert!(
        query.enode("PrimOp", expression).unwrap().is_some(),
        "projection must return the computed field"
    );
}

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
        query.for_each("CheckStatement", |row| {
            loops += usize::from(query.enode("Loop", row[3])?.is_some());
            maps += usize::from(query.enode("Parallel", row[3])?.is_some());
            Ok(())
        }).unwrap();
        assert_eq!(loops, 0, "uniform shape must not require a growable loop: {source}");
        assert!(maps >= 2, "nested maps must survive import: {source}");
    }
}
