use super::import;
use crate::egglog::query::Query;
use crate::tlc::{self, DefMeta, TermKind};
use crate::{compile_thru_tlc, types};
use egglog_engine::sort::S;

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
        let (graph, identities) = import(&program).unwrap();
        let query = Query(&graph);
        let entry = query.row("SourceOriginalEntry", |_| true).unwrap().unwrap()[1];
        let mut producer = None;
        query
            .for_function("SourceOperation", |keys, info| {
                let fields = query.enode("OperationInfo", info)?.unwrap();
                if fields[0] == entry && graph.value_to_base::<S>(fields[1]).as_str() == kind {
                    assert!(producer.replace(keys[0]).is_none());
                }
                Ok(())
            })
            .unwrap();
        let producer = producer.expect("outer producer");
        let mut nested = Vec::new();
        for (table, constructor) in [("SourceBranch", "BranchInfo"), ("SourceLoop", "LoopInfo")] {
            query
                .for_function(table, |_, info| {
                    let fields = query.enode(constructor, info)?.unwrap();
                    nested.extend_from_slice(&fields[fields.len() - 2..]);
                    Ok(())
                })
                .unwrap();
        }
        // Lifted helpers depend on their own parameters, not this entry's producer.
        nested.retain(|&region| {
            let mut root = region;
            while let Some(parent) = identities.scopes[&root].0 {
                root = parent;
            }
            root == entry
        });
        assert!(!nested.is_empty(), "fixture must retain nested execution regions");
        for region in nested {
            assert!(
                query.contains("SourceRegionDependency", (region, producer)).unwrap(),
                "{source}: missing dependency in {region:?}"
            );
        }
        assert!(
            !query.contains("SourceRegionDependency", (entry, producer)).unwrap(),
            "a reference must stop at the producer's own region"
        );
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
    graph.parse_and_run_program(None, "(run-schedule (saturate source-summary))").unwrap();
    let query = Query(&graph);
    let entry = query.row("SourceOriginalEntry", |_| true).unwrap().unwrap()[1];
    let mut scatter = None;
    query
        .for_function("SourceOperation", |keys, info| {
            let fields = query.enode("OperationInfo", info)?.unwrap();
            if graph.value_to_base::<S>(fields[1]).as_str() == "scatter" {
                assert!(scatter.replace(keys[0]).is_none());
            }
            Ok(())
        })
        .unwrap();
    let scatter = scatter.unwrap();
    let mut branch = None;
    query
        .for_function("SourceBranch", |keys, _| {
            assert!(branch.replace(keys[0]).is_none());
            Ok(())
        })
        .unwrap();
    let branch = branch.unwrap();
    for index in [0i64, 1] {
        let input = query.required("SourceParameter", (entry, index)).unwrap();
        assert!(query.contains("ScatterReadValue", (scatter, input)).unwrap());
        assert!(query.contains("SourceNestedUse", (branch, input)).unwrap());
        assert!(!query.contains("SourceDirectUse", (branch, input)).unwrap());
    }
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
    let scalar_id = scalar.id;
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
    let (graph, identities) = import(&program).unwrap();
    let query = Query(&graph);
    let (&(scope, _), &projected) =
        identities.occurrences.iter().find(|((_, id), _)| *id == projected_id).unwrap();
    let scalar = identities.occurrences[&(scope, scalar_id)];
    assert_eq!(query.required("SourceAlias", (projected,)).unwrap(), scalar);
}
