use crate::compile_thru_tlc;
use crate::egglog::dataflow::{
    edit,
    ir::{self, Code, Name, OperationId},
};
use crate::egglog::payload::Payload;
use crate::egglog::{analysis, from_tlc, fuse, memory, query::Query, test_graph, OptimizeError};
use crate::op::BinaryOperator;
use egglog_engine::{EGraph, RawValues, Value, Write};

fn change_lane_index(
    graph: &mut EGraph,
    traversal: Value,
    delta: Option<Value>,
    commute: bool,
) -> Result<(), OptimizeError> {
    let query = Query(graph);
    let lane = query.required("ControlRegion", (traversal, 0_i64))?;
    let operations = query.operations(lane)?;
    let position = operations.iter().position(|&op| query.contains("IndexOperation", op).unwrap()).unwrap();
    let load = operations[position];
    let index = query.required("Input", (load, 1_i64))?;
    let info = query.required("ValueType", query.name(index)?)?;
    let index = Name::Value(query.name(index)?);
    let delta = delta.map(|name| query.name(name)).transpose()?.map(Name::Value);
    graph.update(|mut sink| {
        Ok((|| -> Result<_, OptimizeError> {
            let mut inserted = Vec::new();
            let mut insert = |code, inputs: &[Name]| -> Result<Name, OptimizeError> {
                let slot = inserted.len() as i64;
                let key = sink.add("DerivedOperation", (load, 99_i64, slot))?;
                let result = sink.add("DerivedValue", (load, 99_i64, slot))?;
                let operation = ir::operation_at(&mut sink, OperationId(key), code, inputs)?;
                sink.set("ValueType", result, info)?;
                let output = Name::Value(result);
                ir::outputs(&mut sink, operation, &[output])?;
                inserted.push(key);
                Ok(output)
            };
            let add = Code::Scalar(Payload::Binary(BinaryOperator::Add));
            let new_index = if let Some(delta) = delta {
                let operands = if commute { [delta, index] } else { [index, delta] };
                let sum = insert(add, &operands)?;
                insert(
                    Code::Scalar(Payload::Binary(BinaryOperator::Subtract)),
                    &[sum, delta],
                )?
            } else {
                let one = insert(
                    Code::Constant {
                        ty: crate::types::i32(),
                        bits: 1,
                    },
                    &[],
                )?;
                insert(add, &[index, one])?
            };
            for (slot, key) in inserted.iter().enumerate() {
                sink.set("OperationRegion", *key, lane)?;
                sink.set("OperationOrder", *key, (position + slot) as i64)?;
            }
            for (slot, key) in operations[position..].iter().enumerate() {
                sink.remove("OperationOrder", *key)?;
                sink.set("OperationOrder", *key, (position + inserted.len() + slot) as i64)?;
            }
            sink.remove("Input", (load, 1_i64))?;
            let new_index = new_index.fact(&mut sink)?;
            sink.set("Input", (load, 1_i64), new_index)?;
            Ok(())
        })())
    })??;
    edit::changed(graph)
}

#[test]
fn branch_sinking_moves_single_arm_producers_and_shares_overlapping_demands() -> Result<(), OptimizeError> {
    let mut graph = test_graph::import(
        "entry one(b:bool,x:i32,n:i32) i32 =
            let p=loop s=x for i<n do s*2+i in if b then p+1 else x
         entry both(b:bool,x:i32,n:i32) i32 =
            let p=loop s=x for i<n do s*2+i in if b then p+1 else p-1
         entry separate(b:bool,c:bool,x:i32,n:i32) (i32,i32) =
            let p=loop s=x for i<n do s*2+i
            let y=if b then p+1 else x
            let z=if c then p-1 else x
            in (y,z)",
    )?;
    memory::simplify(&mut graph, false)?;
    let query = Query(&graph);
    let mut loops = Vec::new();
    query.for_each("CountedLoopOperation", |row| {
        if query.contains("LiveOperation", row[0])? {
            loops.push((row[0], query.required("OperationRegion", row[0])?));
        }
        Ok(())
    })?;
    assert_eq!(loops.len(), 3);
    let analyzed = query.required("DataflowRevision", RawValues(vec![]))?;
    assert!(super::parallel::branches(&mut graph)?);
    let query = Query(&graph);
    assert_eq!(query.required("DataflowRevision", RawValues(vec![]))?, analyzed);
    let mut moved = 0;
    for (op, parent) in loops {
        let region = query.required("OperationRegion", op)?;
        if region != parent {
            assert!(query.row("ControlRegion", |row| row[2] == region)?.is_some());
            moved += 1;
        }
    }
    assert_eq!(moved, 2);
    let mut copies = 0;
    query.for_each("CountedLoopOperation", |row| {
        copies += usize::from(query.enode("ClonedOperation", row[0])?.is_some());
        Ok(())
    })?;
    assert_eq!(copies, 0);
    memory::validate(&mut graph)
}

#[test]
fn motion_preserves_array_storage_inside_repeated_regions() -> Result<(), OptimizeError> {
    for (program, allocation_kind, lifted) in [
        (
            "entry main(n:i32) [2]i32=map(|i|[n,n+1][i],iota(2))",
            "ArrayLiteralOperation",
            true,
        ),
        (
            "entry main(n:i32) [2][2]i32=map(|i|[n,n+1],iota(2))",
            "ArrayLiteralOperation",
            false,
        ),
        (
            "entry main(xs:[]i32) i32=reduce(|a,b|
            let first=scratch(2) with [0]=a
            let both=first with [1]=b in both[0]+both[1],0i32,xs)",
            "ScratchAllocation",
            false,
        ),
    ] {
        let mut graph = test_graph::import(program)?;
        memory::simplify(&mut graph, false)?;
        let root = test_graph::root(&graph);
        let query = Query(&graph);
        let allocation = query.first(allocation_kind)?.unwrap()[0];
        let owner = query.required("OperationRegion", allocation)?;
        assert_ne!(owner, root);
        if allocation_kind == "ScratchAllocation" {
            assert!(query.row("ResultReducer", |row| row[2] == owner)?.is_some());
        }
        super::parallel::hoist(&mut graph)?;
        assert_eq!(
            Query(&graph).required("OperationRegion", allocation)?,
            if lifted { root } else { owner },
            "{allocation_kind} in {program}"
        );
        memory::validate(&mut graph)?;
    }
    Ok(())
}

#[test]
fn cross_region_motion_moves_a_capture_chain_in_one_edit() -> Result<(), OptimizeError> {
    let mut graph = test_graph::import(
        "entry main(xs:[8]i32,n:i32,b:bool) [8]i32=map(|x|
            if b then let k=n*n let j=k+n in x+j else x,xs)",
    )?;
    memory::simplify(&mut graph, false)?;
    let root = test_graph::root(&graph);
    let query = Query(&graph);
    let mut multiply = None;
    query.for_each("LiveOperation", |row| {
        if ir::code(query.0, query.required("OperationPayload", row[0])?)
            == Code::Scalar(Payload::Binary(BinaryOperator::Multiply))
        {
            multiply = Some(row[0]);
        }
        Ok(())
    })?;
    let multiply = multiply.unwrap();
    assert_ne!(query.required("OperationRegion", multiply)?, root);
    let demand_work = |graph: &mut EGraph| -> Result<(), OptimizeError> {
        graph.update(|mut sink| sink.add("RegionWorkDemand", root))?;
        graph.parse_and_run_program(None, "(run-schedule (saturate graph-work))")?;
        Ok(())
    };
    demand_work(&mut graph)?;
    let revision = edit::revision(&graph)?;
    assert!(super::parallel::hoist(&mut graph)?);
    assert_eq!(edit::revision(&graph)?, revision + 1);
    let query = Query(&graph);
    assert_eq!(query.required("OperationRegion", multiply)?, root);
    assert!(query.operations(root)?.into_iter().any(|op| {
        query.lookup("Input", (op, 0_i64)).unwrap() == query.lookup("Output", (multiply, 0_i64)).unwrap()
            && ir::code(query.0, query.required("OperationPayload", op).unwrap())
                == Code::Scalar(Payload::Binary(BinaryOperator::Add))
    }));
    // The repaired scope and work summaries must agree with a fresh solve.
    let facts = |graph: &EGraph| -> Result<_, OptimizeError> {
        let mut facts = crate::LookupSet::new();
        for table in [
            "DefinitionRegion",
            "RegionCapture",
            "OperationUse",
            "OperationDependency",
            "RegionWork",
        ] {
            Query(graph).for_each(table, |row| {
                facts.insert((table, row.to_vec()));
                Ok(())
            })?;
        }
        Ok(facts)
    };
    let repaired = facts(&graph)?;
    memory::validate(&mut graph)?;
    crate::egglog::planning::analyze(&mut graph)?;
    demand_work(&mut graph)?;
    assert_eq!(repaired, facts(&graph)?);
    Ok(())
}

#[test]
fn moving_into_empty_regions_preserves_operation_order() {
    let mut graph = test_graph::fixture(
        &[(1, Code::Product), (2, Code::Product)],
        r#"
        (set (OperationRegion (OperationId 1)) (RegionId 0))
        (set (OperationOrder (OperationId 1)) 0)
        (set (OperationRegion (OperationId 2)) (RegionId 0))
        (set (OperationOrder (OperationId 2)) 1)
        (Region (RegionId 1))
    "#,
    )
    .unwrap();
    let query = Query(&graph);
    let first = query.constructor("OperationId", 1_i64).unwrap().unwrap();
    let second = query.constructor("OperationId", 2_i64).unwrap().unwrap();
    let destination = query.constructor("RegionId", 1_i64).unwrap().unwrap();
    let source = query.constructor("RegionId", 0_i64).unwrap().unwrap();
    edit::move_into(&mut graph, first, destination, None).unwrap();
    edit::move_into(&mut graph, second, destination, None).unwrap();
    assert_eq!(Query(&graph).operations(destination).unwrap(), [first, second]);
    assert!(Query(&graph).operations(source).unwrap().is_empty());
    edit::move_before(&mut graph, second, first).unwrap();
    assert_eq!(Query(&graph).operations(destination).unwrap(), [second, first]);
}

#[test]
fn merging_matching_branches_preserves_certificates_and_path_order() -> Result<(), OptimizeError> {
    let mut graph = test_graph::import(
        "entry main(b:bool,x:i32) (i32,i32,i32) =
            let a=if b then x+x else x*x
            let c=if b then x*x*x else x-x*x
            let d=if b then x-a*a else x+a
            in (a,c,d)",
    )?;
    memory::simplify(&mut graph, false)?;
    let root = test_graph::root(&graph);
    let query = Query(&graph);
    let mut branches = Vec::new();
    for op in query.operations(root)? {
        if query.contains("BranchOperation", op)? {
            branches.push(op);
        }
    }
    let [first, second, dependent] = branches[..] else {
        panic!("three authored branches")
    };
    let condition = query.name(query.required("Input", (first, 0_i64))?)?;
    let mut arms = Vec::new();
    for edge in [0_i64, 1] {
        let a = query.required("ControlRegion", (first, edge))?;
        let b = query.required("ControlRegion", (second, edge))?;
        let offset = analysis::count(query, "RegionYieldCount", a)?;
        let operations = [query.operations(a)?, query.operations(b)?].concat();
        arms.push((a, b, offset, operations));
    }
    graph.update(|mut sink| {
        sink.add("OperationCertificate", (second, condition))?;
        for &(_, arm, _, _) in &arms {
            sink.add("YieldCertificate", (arm, 0_i64, condition))?;
        }
        Ok(())
    })?;
    let revision = edit::revision(&graph)?;
    assert!(super::parallel::branches(&mut graph)?);
    assert_eq!(edit::revision(&graph)?, revision + 1);
    let query = Query(&graph);
    assert_eq!(analysis::count(query, "OutputCount", first)?, 2);
    assert!(query.contains("OperationCertificate", (first, condition))?);
    assert!(!query.contains("Operation", second)?);
    assert!(query.contains("Operation", dependent)?);
    for (arm, _, slot, operations) in arms {
        assert_eq!(query.operations(arm)?, operations);
        assert!(query.contains("YieldCertificate", (arm, slot, condition))?);
    }
    memory::validate(&mut graph)
}

#[test]
fn compact_length_uses_a_count_without_a_collection() {
    for (source, has_prefix) in [
        ("entry main(xs:[]i32) i32=length(filter(|x|x>0,xs))", false),
        (
            "entry main(xs:[]i32) ([]i32,i32)=(scan((+),0,xs),length(filter(|x|x>0,xs)))",
            true,
        ),
    ] {
        let source = compile_thru_tlc(source).unwrap();
        let program = fuse(from_tlc(&source).unwrap()).unwrap();
        let query = Query(&program.graph);
        let mut totals = 0;
        let mut packed = 0;
        let mut prefixes = 0;
        query
            .for_function("OutputRole", |_, role| {
                totals += usize::from(query.enode("TotalOutput", role)?.is_some());
                packed += usize::from(query.enode("CompactedOutput", role)?.is_some());
                prefixes += usize::from(query.enode("PrefixOutput", role)?.is_some());
                Ok(())
            })
            .unwrap();
        assert_eq!((totals, packed, prefixes), (1, 0, usize::from(has_prefix)));
        if has_prefix {
            let mut mixed = false;
            query
                .for_each("LiveOperation", |row| {
                    if query.contains("ParallelOperation", row[0])?
                        && analysis::count(query, "OutputCount", row[0])? == 2
                    {
                        mixed = true;
                    }
                    Ok(())
                })
                .unwrap();
            assert!(mixed, "count conversion must preserve the retained scan output");
        }
    }
}

#[test]
fn count_compaction_batches_independent_lengths_and_preserves_observed_arrays() -> Result<(), OptimizeError>
{
    let mut graph = test_graph::import(
        "entry main(xs:[]i32,ys:[]i32) (i32,i32,[]i32,i32) =
            let a=filter(|x|x>0,xs)
            let b=filter(|y|y<0,ys)
            let c=filter(|x|x<7,xs)
            in (length(a),length(b),c,length(c))",
    )?;
    memory::simplify(&mut graph, false)?;
    let revision = edit::revision(&graph)?;
    assert!(super::compaction::run(&mut graph)?);
    assert_eq!(edit::revision(&graph)?, revision + 1);
    let query = Query(&graph);
    let mut roles = (0, 0);
    query.for_function("OutputRole", |_, role| {
        roles.0 += usize::from(query.enode("TotalOutput", role)?.is_some());
        roles.1 += usize::from(query.enode("CompactedOutput", role)?.is_some());
        Ok(())
    })?;
    assert_eq!(roles, (2, 1));
    memory::validate(&mut graph)
}

#[test]
fn guarded_compaction_streams_a_map_into_a_reduce_in_one_batch() -> Result<(), OptimizeError> {
    let source = ["commuted", "ordered", "shifted"]
        .map(|entry| {
            format!(
                "entry {entry}(xs:[]i32,zero:i32,d:i32) ([]i32,i32) =
            let kept=filter(|x|x>0,xs)
            let mapped=map(|x|x+1,kept)
            in (mapped,reduce((+),zero,mapped))"
            )
        })
        .join("\n");
    let mut graph = test_graph::import(&source)?;
    memory::simplify(&mut graph, false)?;
    let parallels = |query: Query<'_>, root| -> Result<Vec<Value>, OptimizeError> {
        let mut traversals = Vec::new();
        for op in query.operations(root)? {
            if query.contains("ParallelOperation", op)? {
                traversals.push(op);
            }
        }
        Ok(traversals)
    };
    let query = Query(&graph);
    let mut roots = Vec::new();
    query.for_function("EntryOrder", |keys, order| {
        roots.push((
            query.0.value_to_base::<i64>(order),
            query.required("FunctionRoot", keys[0])?,
        ));
        Ok(())
    })?;
    roots.sort();
    for (index, &(_, root)) in roots.iter().enumerate() {
        let query = Query(&graph);
        let traversals = parallels(query, root)?;
        assert_eq!(traversals.len(), 3);
        let delta = if index < 2 { Some(query.required("RegionParameter", (root, 2_i64))?) } else { None };
        change_lane_index(&mut graph, traversals[1], delta, index == 0)?;
    }
    memory::simplify(&mut graph, false)?;
    let query = Query(&graph);
    let analyzed = query.required("DataflowRevision", RawValues(vec![]))?;
    assert!(super::parallel::compact(&mut graph)?);
    let query = Query(&graph);
    assert_eq!(query.required("DataflowRevision", RawValues(vec![]))?, analyzed);
    for ((_, root), expected) in roots.into_iter().zip([1, 1, 3]) {
        assert_eq!(parallels(query, root)?.len(), expected);
    }
    assert!(query.first("BranchOperation")?.is_some());
    memory::validate(&mut graph)
}

#[test]
fn compact_consumers_preserve_an_unrelated_scan_and_guard_partial_callbacks() {
    for (source, reduction) in [
        ("entry main(xs:[]i32) ([]i32,[]i32)=(scan((+),0,xs),map(|x|x+1,filter(|x|x>0,xs)))", false),
        ("entry main(xs:[]i32) ([]i32,i32)=(scan((+),0,xs),reduce((+),0,map(|x|100/x,filter(|x|x>0,xs))))", true),
    ] {
        let source = compile_thru_tlc(source).unwrap();
        let program = fuse(from_tlc(&source).unwrap()).unwrap();
        let query = Query(&program.graph);
        let root = test_graph::root(&program.graph);
        let parallels: Vec<_> = query.operations(root).unwrap().into_iter()
            .filter(|&operation| query.contains("ParallelOperation", operation).unwrap()).collect();
        assert_eq!(parallels.len(), 1);
        let mut roles = (0, 0, 0);
        query.for_function("OutputRole", |keys, role| {
            if keys[0] == parallels[0] {
                roles.0 += usize::from(query.enode("PrefixOutput", role)?.is_some());
                roles.1 += usize::from(query.enode("CompactedOutput", role)?.is_some());
                roles.2 += usize::from(query.enode("TotalOutput", role)?.is_some());
            }
            Ok(())
        }).unwrap();
        assert_eq!(roles, (1, usize::from(!reduction), usize::from(reduction)));
        if reduction {
            let mut guarded = 0;
            query.for_each("LiveOperation", |row| {
                if ir::code(&program.graph, query.required("OperationPayload", row[0])?)
                    == Code::Scalar(Payload::Binary(BinaryOperator::Divide)) {
                    let region = query.required("OperationRegion", row[0])?;
                    let owner = query.row("RegionOwnedBy", |row| row[0] == region)?.unwrap()[1];
                    assert!(query.contains("BranchOperation", owner)?);
                    guarded += 1;
                }
                Ok(())
            }).unwrap();
            assert_eq!(guarded, 1);
        }
    }
}

#[test]
fn scan_barriers_survive_intermediate_map_contraction() {
    let source =
        compile_thru_tlc("entry main(xs:[64]i32) i32 = reduce((+),0i32,map(|x|x+1,scan((+),0i32,xs)))")
            .unwrap();
    let program = fuse(from_tlc(&source).unwrap()).unwrap();
    let query = Query(&program.graph);
    let mut scans = 0;
    let mut totals = 0;
    query
        .for_each("LiveOperation", |row| {
            if query.contains("ParallelOperation", row[0])? {
                let mut scan = false;
                let mut total = false;
                for slot in 0..analysis::count(query, "ResultCount", row[0])? {
                    let kind = query.required("ResultKindOf", (row[0], slot))?;
                    if let Some(fold) = query.enode("Folded", kind)? {
                        total |= query.enode("Total", fold[0])?.is_some();
                        scan |= query.enode("InclusiveScan", fold[0])?.is_some();
                    }
                }
                assert!(
                    !(scan && total),
                    "a dependent total must wait for completed scan prefixes"
                );
                scans += usize::from(scan);
                totals += usize::from(total);
            }
            Ok(())
        })
        .unwrap();
    assert_eq!((scans, totals), (1, 1));
}

#[test]
fn one_fusion_publication_contracts_independent_bodies_and_pointwise_chains() -> Result<(), OptimizeError> {
    let mut graph = test_graph::import(
        "entry chain(xs:[8]i32,d:i32) ([8]i32,[8]i32,[8]i32) =
            let a = map(|x| x+1,xs)
            let b = map(|x| x*2,a)
            in (a,b,map(|x| x-3,b))
         entry pair(xs:[8]i32,d:i32) ([8]i32,[8]i32) =
            let a = map(|x| x-1,xs)
            in (a,map(|x| x*3,a))
         entry shifted(xs:[8]i32) ([8]i32,[8]i32) =
            let a = map(|x| x-1,xs)
            in (a,map(|x| x*3,a))
         entry capture(xs:[8]i32, n:i32) ([8]i32,[8]i32,[8]i32) =
            let a = map(|x| x+1,xs)
            let k = n*n
            let b = map(|x| x*2,a)
            in (a,b,map(|x| x+k,b))
         entry disjoint(xs:[8]i32, ys:[16]i32) ([8]i32,[16]i32) =
            let a = map(|x| x+1,xs)
            let b = map(|x| x*2,a)
            let c = map(|x| x-1,ys)
            in (b,map(|x| x*3,c))
         entry final_value(xs:[8]i32) [8]i32 =
            let a = map(|x| x+1,xs)
            let b = map(|x| x*2,a)
            in map(|x| x-3,b)",
    )?;
    memory::simplify(&mut graph, false)?;
    let query = Query(&graph);
    let mut roots = Vec::new();
    query.for_function("EntryOrder", |keys, order| {
        roots.push((
            query.0.value_to_base::<i64>(order),
            query.required("FunctionRoot", keys[0])?,
        ));
        Ok(())
    })?;
    roots.sort();
    for (_, root) in roots.iter().take(3).copied() {
        let query = Query(&graph);
        let consumer = query
            .operations(root)?
            .into_iter()
            .filter(|&op| query.contains("ParallelOperation", op).unwrap())
            .nth(1)
            .unwrap();
        let delta = query.lookup("RegionParameter", (root, 1_i64))?;
        let commute = root == roots[0].1;
        change_lane_index(&mut graph, consumer, delta, commute)?;
    }
    memory::simplify(&mut graph, false)?;
    assert!(super::parallel::run(&mut graph)?);
    let mut counts = Vec::new();
    for (_, region) in roots {
        let query = Query(&graph);
        let mut outputs = Vec::new();
        for op in query.operations(region)? {
            if query.contains("ParallelOperation", op)? {
                outputs.push(analysis::count(query, "OutputCount", op)?);
            }
        }
        outputs.sort_unstable();
        counts.push(outputs);
    }
    counts.sort_unstable();
    assert_eq!(
        counts,
        [vec![1], vec![1, 1], vec![1, 1], vec![1, 2], vec![2], vec![3]]
    );
    memory::validate(&mut graph)
}

#[test]
fn sibling_motion_moves_a_preparation_chain_in_one_edit() -> Result<(), OptimizeError> {
    let mut graph = test_graph::import(
        "entry main(xs:[8]i32, ys:[16]i32, n:i32) ([8]i32,[16]i32) =
            let a = map(|x| x+1,xs)
            let k = n*n
            let j = k+n
            in (a,map(|x| x+j,ys))",
    )?;
    memory::simplify(&mut graph, false)?;
    let root = test_graph::root(&graph);
    let revision = edit::revision(&graph)?;
    let arithmetic_positions = |query: Query<'_>| -> Result<Vec<usize>, OptimizeError> {
        let mut positions = Vec::new();
        for (position, operation) in query.operations(root)?.into_iter().enumerate() {
            let code = ir::code(query.0, query.required("OperationPayload", operation)?);
            if matches!(
                code,
                Code::Scalar(Payload::Binary(BinaryOperator::Multiply | BinaryOperator::Add))
            ) {
                positions.push(position);
            }
        }
        Ok(positions)
    };
    let before = arithmetic_positions(Query(&graph))?;
    assert_eq!(before.len(), 2);
    assert!(super::parallel::move_siblings(&mut graph)?);
    assert_eq!(edit::revision(&graph)?, revision + 1);
    let after = arithmetic_positions(Query(&graph))?;
    assert!(after.iter().zip(before).all(|(after, before)| *after < before));
    let query = Query(&graph);
    let order = query.operations(root)?;
    assert_eq!(query.vector(query.required("RegionOperations", root)?)?, order);
    for pair in order.windows(2) {
        assert!(query.contains("NextOperation", (pair[0], pair[1]))?);
    }
    memory::validate(&mut graph)
}

#[test]
fn free_names_preserve_large_shared_fanout() {
    let mut source = String::from("entry main(x:i32) (");
    source.push_str(&vec!["i32"; 256].join(","));
    source.push_str(")=(");
    source.push_str(&(0..256).map(|i| format!("x+{i}")).collect::<Vec<_>>().join(","));
    source.push(')');
    let tlc = compile_thru_tlc(&source).unwrap();
    let mut program = from_tlc(&tlc).unwrap();
    analysis::run(&mut program.graph).unwrap();
    let query = Query(&program.graph);
    let entry = query.first("EntryOrder").unwrap().unwrap()[0];
    let root = query.required("FunctionRoot", entry).unwrap();
    let input = query.required("RegionParameter", (root, 0_i64)).unwrap();
    let mut uses = 0;
    query
        .for_each("DirectOperationUse", |row| {
            uses += usize::from(row[1] == input);
            Ok(())
        })
        .unwrap();
    assert!(
        uses >= 256,
        "shared input must remain a dependency of every operation"
    );
}

#[test]
fn fusion_across_scalar_work_preserves_dependencies_and_traps() {
    for source in [
        "entry main(xs:[]i32) (i32,[]i32)=let ys=filter(|x|x>0,xs) let n=length(ys) in (n,map(|x|x+n,xs))",
        "entry main(xs:[]i32,n:i32) ([]i32,i32,[]i32)=let a=map(|x|x+1,xs) let guard=100/n let b=map(|x|x*2,xs) in (a,guard,b)",
    ] {
        let source = compile_thru_tlc(source).unwrap();
        let program = fuse(from_tlc(&source).unwrap()).unwrap();
        let query = Query(&program.graph);
        let parallels = query.operations(test_graph::root(&program.graph)).unwrap().into_iter()
            .filter(|&operation| query.contains("ParallelOperation", operation).unwrap()).count();
        assert_eq!(parallels, 2, "an unavailable input or intervening trap prevents fusion");
    }
}

#[test]
fn scatter_fuses_across_copy_and_repeated_shape_calculations() {
    let source = compile_thru_tlc(include_str!("../../../../tests/compiler/library/computed_dispatch_extents_preserve_parameters_wrapping_and_grid_limits.wyn")).unwrap();
    let program = fuse(from_tlc(&source).unwrap()).unwrap();
    let query = Query(&program.graph);
    let mut parallels = 0;
    query
        .for_each("LiveOperation", |row| {
            parallels += usize::from(query.contains("ParallelOperation", row[0])?);
            Ok(())
        })
        .unwrap();
    assert_eq!(
        parallels, 1,
        "the mapped contribution is computed in the scatter lane"
    );
}

#[test]
fn adjacent_accumulators_keep_distinct_capabilities_in_one_batch() -> Result<(), OptimizeError> {
    let mut graph = test_graph::import(
        "entry main(xs:[]i32) ([]i32,[]i32,[]i32) =
            let a=replicate(length(xs),0i32)
            let b=replicate(length(xs),1i32)
            let c=replicate(length(xs),2i32)
            in (scatter(a,xs,xs),scatter(b,xs,xs),scatter(c,xs,xs))",
    )?;
    memory::simplify(&mut graph, false)?;
    if super::parallel::run(&mut graph)? {
        memory::simplify(&mut graph, false)?;
    }
    if super::parallel::move_siblings(&mut graph)? {
        memory::simplify(&mut graph, false)?;
    }
    let root = test_graph::root(&graph);
    let query = Query(&graph);
    let accumulators = |query: Query<'_>| -> Result<Vec<_>, OptimizeError> {
        let mut found = Vec::new();
        for op in query.operations(root)? {
            if query.contains("AccumulatorOperation", op)? {
                found.push(analysis::count(query, "AccumulatorGroupCount", op)?);
            }
        }
        Ok(found)
    };
    assert_eq!(accumulators(query)?, [1, 1, 1]);
    let analyzed = query.required("DataflowRevision", RawValues(vec![]))?;
    assert!(super::parallel::accumulators(&mut graph, false)?);
    let query = Query(&graph);
    assert_eq!(query.required("DataflowRevision", RawValues(vec![]))?, analyzed);
    assert_eq!(accumulators(query)?, [3]);
    memory::validate(&mut graph)
}

#[test]
fn accumulator_sinking_batches_independent_producers() -> Result<(), OptimizeError> {
    let mut graph = test_graph::import(
        "entry first(dst:*[]i32,xs:[]i32) []i32 =
            let ys=map(|x|x+1,xs) in scatter(dst,iota(length(xs)),ys)
         entry second(dst:*[]i32,xs:[]i32) []i32 =
            let ys=map(|x|x*2,xs) in scatter(dst,iota(length(xs)),ys)",
    )?;
    memory::simplify(&mut graph, false)?;
    let query = Query(&graph);
    let mut regions = Vec::new();
    query.for_function("EntryOrder", |key, _| {
        regions.push(query.required("FunctionRoot", key[0])?);
        Ok(())
    })?;
    let analyzed = query.required("DataflowRevision", RawValues(vec![]))?;
    assert!(super::parallel::accumulators(&mut graph, true)?);
    let query = Query(&graph);
    assert_eq!(query.required("DataflowRevision", RawValues(vec![]))?, analyzed);
    for region in regions {
        let operations = query.operations(region)?;
        assert!(operations.iter().all(|&op| !query.contains("ParallelOperation", op).unwrap()));
        let acc =
            operations.into_iter().find(|&op| query.contains("AccumulatorOperation", op).unwrap()).unwrap();
        let body = query.required("ControlRegion", (acc, 0_i64))?;
        let traversals = query
            .operations(body)?
            .into_iter()
            .filter(|&op| query.contains("ParallelOperation", op).unwrap())
            .count();
        assert_eq!(
            traversals, 2,
            "external reads and writes retain separate traversals"
        );
    }
    memory::validate(&mut graph)
}

#[test]
fn accumulator_sinking_stabilizes_when_domains_differ() {
    let source = compile_thru_tlc("entry main(dest:*[]i32,n:i32,m:i32) []i32 = let values=map(|i|i+1,iota(n)) in scatter(dest,iota(m),values)").unwrap();
    let program = fuse(from_tlc(&source).unwrap()).unwrap();
    let query = Query(&program.graph);
    let mut parallels = 0;
    query
        .for_each("LiveOperation", |row| {
            parallels += usize::from(query.contains("ParallelOperation", row[0])?);
            Ok(())
        })
        .unwrap();
    assert_eq!(parallels, 2, "unequal domains must retain separate traversals");
}
