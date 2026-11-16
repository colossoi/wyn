use crate::egglog::dataflow::edit;
use crate::egglog::dataflow::ir::Code;
use crate::egglog::query::Query;
use crate::egglog::{analysis, planning, test_graph, OptimizeError};
use crate::types;
use egglog_engine::{EGraph, RawValues, Write};

#[test]
fn keyed_positions_keep_multi_result_occurrences_distinct() -> Result<(), OptimizeError> {
    let graph = test_graph::fixture(
        &[
            (
                0,
                Code::Constant {
                    ty: types::i32(),
                    bits: 7,
                },
            ),
            (
                1,
                Code::Constant {
                    ty: types::i32(),
                    bits: 7,
                },
            ),
        ],
        r#"
        (set (OutputCount (OperationId 0)) 2)
        (set (Output (OperationId 0) 0) (ValueNameRef (ValueId 9000)))
        (set (Output (OperationId 0) 1) (ValueNameRef (ValueId 9001)))
        (set (OutputCount (OperationId 1)) 1)
        (set (Output (OperationId 1) 0) (ValueNameRef (ValueId 9002)))
    "#,
    )?;
    let query = Query(&graph);
    let a = query.constructor("OperationId", 0_i64)?.unwrap();
    let b = query.constructor("OperationId", 1_i64)?.unwrap();
    assert_ne!(a, b);
    assert_eq!(
        query.required("OperationPayload", a)?,
        query.required("OperationPayload", b)?
    );
    assert_eq!(query.sequence("Output", a)?.len(), 2);
    assert_eq!(query.sequence("Output", b)?.len(), 1);
    assert_ne!(
        query.required("Output", (a, 0_i64))?,
        query.required("Output", (a, 1_i64))?
    );
    assert!(query.lookup("Output", (a, -1_i64))?.is_none());
    assert!(query.lookup("Output", (a, 2_i64))?.is_none());
    Ok(())
}

#[test]
fn unused_values_follow_complete_uses_and_the_current_program() -> Result<(), OptimizeError> {
    let mut graph = test_graph::import("entry main(x:i32) i32=x+7")?;
    super::validate(&mut graph)?;
    let function = test_graph::entry(&graph);
    let root = test_graph::root(&graph);
    let outputs = Query(&graph).sequence("RegionYield", root)?;
    super::simplify(&mut graph, false)?;
    assert_eq!(Query(&graph).sequence("RegionYield", root)?, outputs);
    assert!(!Query(&graph).operations(root)?.is_empty());
    graph.update(|mut sink| {
        sink.remove("RegionYield", (root, 0_i64))?;
        sink.remove("RegionYieldCount", root)?;
        sink.set("RegionYieldCount", root, 0_i64)?;
        sink.remove("RegionReturnType", (root, 0_i64))?;
        sink.remove("FunctionResultCount", function)?;
        sink.set("FunctionResultCount", function, 0_i64)?;
        sink.remove("FunctionResultType", (function, 0_i64))?;
        Ok(())
    })?;
    super::simplify(&mut graph, false)?;
    assert!(Query(&graph).operations(root)?.is_empty());
    Ok(())
}

#[test]
fn substitution_preserves_static_types_and_renames_nested_indices() -> Result<(), OptimizeError> {
    let mut graph = test_graph::import("entry main(xs:[][4]i32,start:i32,other:i32) [][4]i32=xs[start..]")?;
    analysis::run(&mut graph)?;
    let root = test_graph::root(&graph);
    let query = Query(&graph);
    let start = query.required("RegionParameter", (root, 1_i64))?;
    let other = query.required("RegionParameter", (root, 2_i64))?;
    let view = query.first("ViewOperation")?.unwrap()[0];
    let output = query.name(query.required("Output", (view, 0_i64))?)?;
    let info = query.required("ValueType", output)?;
    edit::replace(&mut graph, &[(start, other)])?;
    analysis::run(&mut graph)?;
    let query = Query(&graph);
    assert_eq!(query.required("ValueType", output)?, info);
    assert!(query.contains("DirectOperationUse", (view, other))?);
    assert!(!query.contains("DirectOperationUse", (view, start))?);
    Ok(())
}

#[test]
fn applied_rewrites_do_not_replay_when_another_pass_revisits_the_same_body() -> Result<(), OptimizeError> {
    let mut graph = test_graph::import("entry main(x:i32,y:i32,z:i32) i32=x+1")?;
    analysis::run(&mut graph)?;
    let root = test_graph::root(&graph);
    let query = Query(&graph);
    let x = query.required("RegionParameter", (root, 0_i64))?;
    let y = query.required("RegionParameter", (root, 1_i64))?;
    let z = query.required("RegionParameter", (root, 2_i64))?;
    edit::replace(&mut graph, &[(x, y)])?;
    analysis::run(&mut graph)?;
    edit::replace(&mut graph, &[(y, z)])?;
    analysis::run(&mut graph)?;
    let revision = edit::revision(&graph)?;
    graph.parse_and_run_program(
        None,
        "(run-schedule (saturate graph-candidates graph-needed graph-apply))",
    )?;
    assert_eq!(edit::revision(&graph)?, revision);
    let operation = Query(&graph).first("Operation")?.unwrap()[0];
    assert!(!Query(&graph).contains("DirectOperationUse", (operation, x))?);
    assert!(!Query(&graph).contains("DirectOperationUse", (operation, y))?);
    Ok(())
}

#[test]
fn reduction_combiner_cannot_capture_a_lane_index() -> Result<(), OptimizeError> {
    let mut graph = test_graph::import("entry main(xs:[4]i32) i32=reduce((+),0,xs)")?;
    analysis::run(&mut graph)?;
    let query = Query(&graph);
    let parallel = query.first("ParallelOperation")?.unwrap()[0];
    let domain = query.required("OperationDomain", parallel)?;
    let index = query.required("DomainAxis", (domain, 0_i64))?;
    let reducer = query.required("ResultReducer", (parallel, 0_i64))?;
    graph.update(|mut sink| {
        let value = sink.add("ValueNameRef", index)?;
        sink.remove("RegionYield", (reducer, 0_i64))?;
        sink.set("RegionYield", (reducer, 0_i64), value)
    })?;
    assert!(super::validate(&mut graph).is_err());
    Ok(())
}

fn fusion_case(pointwise: bool) -> Result<EGraph, OptimizeError> {
    let consumer = if pointwise { "map(|x|x+2,a)" } else { "map(|x|a[0],xs)" };
    let source =
        format!("entry main(xs:[4]i32) ([4]i32,[4]i32)=let a=map(|x|x+1,xs) in let b={consumer} in (a,b)");
    test_graph::import(&source)
}

#[test]
fn fusion_forwards_pointwise_reads_and_retains_both_results() -> Result<(), OptimizeError> {
    let mut graph = fusion_case(true)?;
    super::simplify(&mut graph, false)?;
    crate::egglog::fusion::run(&mut graph)?;
    super::validate(&mut graph)?;
    let query = Query(&graph);
    let mut results = 0;
    let mut parallels = 0;
    query.for_each("LiveOperation", |row| {
        if query.contains("ParallelOperation", row[0])? {
            parallels += 1;
            results += analysis::count(query, "OutputCount", row[0])?;
        }
        Ok(())
    })?;
    assert_eq!(results, 2);
    assert_eq!(parallels, 1);
    let mut reads = 0;
    query.for_each("LiveOperation", |row| {
        if query.contains("IndexOperation", row[0])? {
            let source = query.required("Input", (row[0], 0_i64))?;
            query.for_function("Output", |keys, output| {
                reads += usize::from(output == source && query.contains("ParallelOperation", keys[0])?);
                Ok(())
            })?;
        }
        Ok(())
    })?;
    assert_eq!(reads, 0, "pointwise loads of unfinished arrays are redirected");
    Ok(())
}

#[test]
fn fusion_preserves_nonpointwise_dependencies() -> Result<(), OptimizeError> {
    let mut graph = fusion_case(false)?;
    super::simplify(&mut graph, false)?;
    crate::egglog::fusion::run(&mut graph)?;
    let query = Query(&graph);
    let mut parallels = 0;
    query.for_each("LiveOperation", |row| {
        parallels += usize::from(query.contains("ParallelOperation", row[0])?);
        Ok(())
    })?;
    assert_eq!(parallels, 2);
    Ok(())
}

#[test]
fn source_import_builds_closed_scalar_control_and_parallel_bodies() -> Result<(), OptimizeError> {
    for text in [
        "entry main(x:i32) i32 = x + 1",
        "entry main(x:i32) i32 = if x > 0 then x + 1 else x - 1",
        "entry main(xs:[64]i32) [64]i32 = map(|x|x+1,xs)",
        "entry main(xs:[64]i32) i32 = reduce((+),0i32,map(|x|x+1,xs))",
    ] {
        let source = crate::compile_thru_tlc(text).unwrap();
        let (mut graph, _) = super::super::from_tlc::import(&source)?;
        super::validate(&mut graph)?;
        super::super::fusion::run(&mut graph)?;
        super::validate(&mut graph)?;
        assert!(Query(&graph).first("InvalidDataflow")?.is_none());
    }
    Ok(())
}

#[test]
fn logical_import_preserves_storage_products_and_rectangular_rows() -> Result<(), OptimizeError> {
    for text in [
        "entry main(xs:*[4](i32,i32)) (i32,i32) = xs[1]",
        "entry main(xs:*[4](i32,i32)) [4]i32 = map(|p|p.0+p.1,xs)",
        "entry main(xs:*[4](i32,i32)) [4](i32,i32) = xs with [1] = (3,4)",
        "entry main(xs:*[2][3](i32,i32)) (i32,i32) = xs[1][2]",
        "entry main() [2][2]i32 = [[1,2],[3,4]]",
    ] {
        let source = crate::compile_thru_tlc(text).unwrap();
        let (mut graph, _) = super::super::from_tlc::import(&source)
            .map_err(|error| crate::egglog::output_error(format!("{text}: {error}")))?;
        super::validate(&mut graph)?;
        assert!(Query(&graph).any("ValueType", |row| Query(&graph)
            .enode("LogicalValue", *row.last().unwrap())
            .unwrap()
            .is_some())?);
    }
    Ok(())
}

#[test]
fn execution_planning_preserves_logical_array_types() -> Result<(), OptimizeError> {
    for text in [
        "#[linked(\"helper_impl\")] extern helper(xs:[4]i32) [4]i32\nentry main(xs:[4]i32) [4]i32 = helper(xs)",
        "entry main(xs:[64]i32) [64]i32 = map(|x|x*2,map(|x|x+1,xs))",
        "entry main(xs:[4]i32,flag:bool) []i32 = if flag then xs[1..3] else xs[0..2]",
        "entry main(xs:[4]i32,n:i32) [4]i32 = loop a = xs for i < n do a with [0] = i",
        "entry main(xs:[64]i32) []i32 = filter(|x|x>0,xs)",
        "entry main(dest:[3]i32,xs:[4]i32) [3]i32 = reduce_by_index(dest,(+),0i32,[0,1,1,2],xs)",
    ] {
        let source = crate::compile_thru_tlc(text).unwrap();
        let (mut graph, _) = super::super::from_tlc::import(&source)?;
        super::validate(&mut graph)?;
        super::super::fusion::run(&mut graph)?;
        crate::egglog::planning::place(&mut graph, crate::PipelineTopologyPolicy::AllowGenerated, crate::egglog::cost::CostModel::default())?;
        crate::egglog::planning::schedule(&mut graph)?;
        assert!(Query(&graph).any("ValueType",|row| Query(&graph).enode("LogicalValue",*row.last().unwrap()).unwrap().is_some())?);
    }
    Ok(())
}

#[test]
fn different_parameter_names_do_not_prove_disjoint_memory() -> Result<(), OptimizeError> {
    let mut graph = test_graph::fixture(
        &[],
        r#"
        (let array (LogicalValue (ArrayType (StorageArray) (vec-of (FixedExtent 4)) (ElementValue (ScalarElement (I32))))))
        (set (ValueType (ValueId 0)) array) (set (ValueType (ValueId 1)) array)
        (set (RegionParameter (RegionId 0) 0) (ValueNameRef (ValueId 0)))
        (set (RegionParameter (RegionId 0) 1) (ValueNameRef (ValueId 1)))
        (set (Input (OperationId 0) 0) (ValueNameRef (ValueId 0))) (WritesInput (OperationId 0) 0)
        (set (Input (OperationId 1) 0) (ValueNameRef (ValueId 1))) (ReadsInput (OperationId 1) 0)
        (EffectCompare (OperationId 0) (OperationId 1))
        (run-schedule (saturate dataflow-resources) (saturate dataflow-effects))
        (check (OperationsConflict (OperationId 0) (OperationId 1)))
    "#,
    )?;
    graph.parse_and_run_program(None,"(check (ResourceRoot (ValueNameRef (ValueId 0)) (ExternalMemory))) (check (ResourceRoot (ValueNameRef (ValueId 1)) (ExternalMemory)))")?;
    Ok(())
}

#[test]
fn consumption_follows_product_components_and_excludes_shape_witnesses() -> Result<(), OptimizeError> {
    let mut graph = test_graph::import(
        r#"
        #[linked("consume_pair_impl")] extern consume_pair(p: (*[4]i32, [4]i32)) i32
        entry main(x:*[4]i32,y:*[4]i32) i32=consume_pair((x,y))
    "#,
    )?;
    super::validate(&mut graph)?;
    graph.parse_and_run_program(
        None,
        r#"
        (check (External function) (= (ExternalParameterCount function) 1)
            (= (ExternalParameterDiet function 0) (ProductDiet (vec-of (Consume) (Observe)))))
        (check (CallOperation operation) (= (InputCount operation) 1)
            (= (InputDiet operation 0) (ProductDiet (vec-of (Consume) (Observe)))))
    "#,
    )?;
    Ok(())
}

#[test]
fn read_only_indexing_keeps_its_guard_without_a_domain_bounds_proof() -> Result<(), OptimizeError> {
    for text in [
        "entry main(xs:[]i32,i:i32,b:bool) i32=if b then xs[i] else 0",
        "entry main(xs:[]i32,i:i32,n:i32) []i32=map(|j|xs[i]+j,iota(n))",
    ] {
        let source = crate::compile_thru_tlc(text).unwrap();
        let program = crate::egglog::fuse(crate::egglog::from_tlc(&source)?)?;
        let query = Query(&program.graph);
        let root = test_graph::root(&program.graph);
        let mut guarded = 0;
        query.for_each("LiveOperation", |row| {
            if query.contains("IndexOperation", row[0])? && query.contains("OperationTrap", row[0])? {
                assert_ne!(query.required("OperationRegion", row[0])?, root);
                guarded += 1;
            }
            Ok(())
        })?;
        assert!(guarded > 0);
    }
    Ok(())
}

#[test]
fn allocation_bounds_do_not_assume_integer_sums_cannot_overflow() -> Result<(), OptimizeError> {
    for limit in [1, 2147483647] {
        let text = format!("entry main(b:bool) i32=reduce((+),0,map(|i|if b then {limit} else 0,iota(8)))");
        let mut graph = test_graph::import(&text)?;
        super::simplify(&mut graph, false)?;
        crate::egglog::fusion::run(&mut graph)?;
        planning::analyze(&mut graph)?;
        let query = Query(&graph);
        let output = query.name(query.required("RegionYield", (test_graph::root(&graph), 0_i64))?)?;
        let index = graph.update(|mut sink| sink.add("IndexValue", output))?;
        let bound =
            Query(&graph).lookup("IndexUpper", index)?.map(|value| graph.value_to_base::<i64>(value));
        assert_eq!(bound, if limit == 1 { Some(8) } else { None });
    }
    Ok(())
}

#[test]
fn range_lengths_clamp_negative_counts_before_shape_planning() -> Result<(), OptimizeError> {
    let mut graph = test_graph::import("entry main() i32=length(iota(-3))")?;
    super::simplify(&mut graph, false)?;
    let query = Query(&graph);
    let result = query.required("RegionYield", (test_graph::root(&graph), 0_i64))?;
    assert_eq!(
        graph.value_to_base::<i64>(query.required("ConstantValue", result)?),
        0
    );

    let mut graph = test_graph::import("entry main(n:i32) i32=length(iota(n))")?;
    super::simplify(&mut graph, false)?;
    let query = Query(&graph);
    let calculation = query.first("IndexCalculationOperation")?.unwrap()[0];
    let extent = query.required("OperationIndex", (calculation, 0_i64))?;
    let max = query.fields("IndexMax", extent)?;
    assert_eq!(
        graph.value_to_base::<i64>(query.fields("IndexConstant", max[0])?[0]),
        0
    );
    Ok(())
}

#[test]
fn consumed_tokens_cannot_be_reused() -> Result<(), OptimizeError> {
    let mut graph = test_graph::fixture(
        &[(0, Code::Use), (1, Code::Use)],
        r#"
        (let info (AccumulatorInfo (vec-of (IndexConstant 4)) (vec-of (ScalarElement (I32)))))
        (set (AccumulatorType (AccumulatorId 0)) info) (set (AccumulatorType (AccumulatorId 1)) info) (set (AccumulatorType (AccumulatorId 2)) info)
        (Region (RegionId 0)) (FlowScope (RegionId 0))
        (set (RegionParameterCount (RegionId 0)) 1) (set (RegionParameter (RegionId 0) 0) (AccNameRef (AccumulatorId 0)))
        (set (ParameterDiet (RegionId 0) 0) (Consume)) (set (RegionYieldCount (RegionId 0)) 0)
        (set (InputCount (OperationId 0)) 1) (set (Input (OperationId 0) 0) (AccNameRef (AccumulatorId 0)))
        (set (OutputCount (OperationId 0)) 1) (set (Output (OperationId 0) 0) (AccNameRef (AccumulatorId 1)))
        (set (InputCount (OperationId 1)) 1) (set (Input (OperationId 1) 0) (AccNameRef (AccumulatorId 0)))
        (set (OutputCount (OperationId 1)) 1) (set (Output (OperationId 1) 0) (AccNameRef (AccumulatorId 2)))
        (set (OperationRegion (OperationId 0)) (RegionId 0)) (set (OperationOrder (OperationId 0)) 0)
        (set (OperationRegion (OperationId 1)) (RegionId 0)) (set (OperationOrder (OperationId 1)) 1)
        (Function (FunctionId 0 "invalid")) (set (FunctionRoot (FunctionId 0 "invalid")) (RegionId 0))
        (set (FunctionResultCount (FunctionId 0 "invalid")) 0) (set (EntryOrder (FunctionId 0 "invalid")) 0)
    "#,
    )?;
    let error = super::validate(&mut graph).expect_err("consumed capability reuse");
    assert!(error.to_string().contains("consumed"), "{error}");
    Ok(())
}

#[test]
fn validation_checks_live_functions_and_ignores_retired_programs() -> Result<(), OptimizeError> {
    let mut graph = test_graph::import("entry main(x:i32) i32=x")?;
    let root = test_graph::root(&graph);
    let function = test_graph::entry(&graph);
    graph.update(|mut sink| {
        let missing = sink.add("ValueId", 999_i64)?;
        let name = sink.add("ValueNameRef", missing)?;
        let dead_root = sink.add("RegionId", 999_i64)?;
        sink.set("RegionYieldCount", dead_root, 1_i64)?;
        let dead = sink.add("FunctionId", (999_i64, "dead"))?;
        let dead_root = sink.add("RegionId", 999_i64)?;
        sink.add("Function", dead)?;
        sink.set("FunctionRoot", dead, dead_root)?;
        sink.set("RegionYield", (dead_root, 0_i64), name)?;
        Ok(())
    })?;
    super::validate(&mut graph)?;
    let original = Query(&graph).required("RegionParameter", (root, 0_i64))?;
    graph.update(|mut sink| {
        sink.remove("RegionParameterCount", root)?;
        sink.set("RegionParameterCount", root, 2_i64)?;
        sink.set("RegionParameter", (root, 1_i64), original)?;
        let observe = sink.add("Observe", RawValues(Vec::new()))?;
        sink.set("ParameterDiet", (root, 1_i64), observe)?;
        Ok(())
    })?;
    let error = super::validate(&mut graph).expect_err("duplicate binders");
    assert!(error.to_string().contains("more than one definition"), "{error}");
    graph.update(|mut sink| {
        sink.remove("RegionParameter", (root, 1_i64))?;
        sink.remove("ParameterDiet", (root, 1_i64))?;
        sink.remove("RegionParameterCount", root)?;
        sink.set("RegionParameterCount", root, 1_i64)?;
        sink.remove("EntryOrder", function)?;
        Ok(())
    })?;
    super::validate(&mut graph)?;
    Ok(())
}

#[test]
fn accumulator_name_filter_preserves_only_capabilities() -> Result<(), OptimizeError> {
    let graph = test_graph::fixture(
        &[],
        r#"
        (function FixtureName (i64) Name :no-merge)
        (set (FixtureName 0) (ValueNameRef (ValueId 0))) (set (FixtureName 1) (AccNameRef (AccumulatorId 2)))
        (set (FixtureName 2) (ValueNameRef (ValueId 3))) (set (FixtureName 3) (AccNameRef (AccumulatorId 4)))
    "#,
    )?;
    let query = Query(&graph);
    let mut tokens = 0;
    query.for_function("FixtureName", |_, name| {
        tokens += usize::from(query.value_name(name)?.is_none());
        Ok(())
    })?;
    assert_eq!(tokens, 2);
    Ok(())
}

#[test]
fn invariant_auditing_is_enabled_only_in_debug_builds() -> Result<(), OptimizeError> {
    let mut graph = test_graph::import("entry main(x:i32) i32=x")?;
    let root = test_graph::root(&graph);
    graph.update(|mut sink| {
        let value = sink.add("ValueId", 999_i64)?;
        let name = sink.add("ValueNameRef", value)?;
        sink.remove("RegionYield", (root, 0_i64))?;
        sink.set("RegionYield", (root, 0_i64), name)
    })?;
    assert_eq!(super::validate(&mut graph).is_err(), cfg!(debug_assertions));
    Ok(())
}

#[test]
fn borrowed_vector_indexing_preserves_bounds_and_sort_overloads() -> Result<(), OptimizeError> {
    let mut graph = test_graph::fixture(
        &[],
        r#"
        (sort FixtureNumbers (Vec i64)) (sort FixtureNames (Vec ValueName))
        (let numbers (vec-of 7 19)) (let names (vec-of (ValueId 3) (ValueId 8)))
        (check (= (vec-get numbers 0) 7)) (check (= (vec-get numbers 1) 19))
        (check (= (vec-get names 1) (ValueId 8)))
        (fail (check (= (vec-get numbers -1) 7))) (fail (check (= (vec-get numbers 2) 7)))
        (let empty (vec-pop (vec-of 7))) (fail (check (= (vec-get empty 0) 7)))
        (relation IndexedNumber (i64)) (ruleset vector-read)
        (rule ((= value (vec-get numbers 1))) ((IndexedNumber value)) :ruleset vector-read)
        (run-schedule vector-read) (check (IndexedNumber 19))
    "#,
    )?;
    graph.parse_and_run_program(None, "(check (IndexedNumber 19))")?;
    Ok(())
}

#[test]
fn compact_bindings_preserve_clause_offsets_and_liveness() -> Result<(), OptimizeError> {
    let mut graph = test_graph::fixture(&[], include_str!("tests/compact_bindings.egg"))?;
    graph.parse_and_run_program(None,"(run-schedule (saturate graph-needed) (saturate graph-result-select)) (check (DeadResultClause (OperationId 0) 1)) (fail (check (DeadResultClause (OperationId 1) 0))) (fail (check (DeadResultClause (OperationId 2) 0)))")?;
    Ok(())
}

#[test]
fn collection_extents_preserve_domains_and_clause_offsets() -> Result<(), OptimizeError> {
    let mut graph = test_graph::fixture(&[], include_str!("tests/collection_extents.egg"))?;
    graph.parse_and_run_program(None,"(run-schedule (saturate graph-sizes)) (check (= (LogicalOuterCount (ValueId 11)) (IndexConstant 5))) (check (= (LogicalOuterCount (ValueId 12)) (IndexConstant 5))) (check (= (LogicalOuterCount (ValueId 13)) (IndexConstant 7))) (fail (check (= count (LogicalOuterCount (ValueId 10))))) (fail (check (= count (LogicalOuterCount (ValueId 14)))))")?;
    Ok(())
}

#[test]
fn local_rewrites_batch_functions_without_combining_competing_edits() -> Result<(), OptimizeError> {
    let mut graph = test_graph::import(
        "entry first(x:i32) i32=x+1 entry second(y:i32) i32=y+2 entry untouched(z:i32) i32=z",
    )?;
    analysis::run(&mut graph)?;
    let query = Query(&graph);
    let mut operations = Vec::new();
    query.for_each("LiveOperation", |row| {
        if query.lookup("ScalarPayload", row[0])?.is_some() {
            operations.push(row[0]);
        }
        Ok(())
    })?;
    assert_eq!(operations.len(), 2);
    for (operation, bits) in operations.into_iter().zip([11, 22]) {
        edit::set_code(
            &mut graph,
            operation,
            Code::Constant {
                ty: types::i32(),
                bits,
            },
        )?;
        edit::inputs(&mut graph, operation, &[])?;
    }
    super::simplify(&mut graph, false)?;
    let query = Query(&graph);
    let mut entries = 0;
    query.for_function("EntryOrder", |_, _| {
        entries += 1;
        Ok(())
    })?;
    assert_eq!(entries, 3);
    Ok(())
}
