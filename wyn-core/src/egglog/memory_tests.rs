use super::load;
use crate::egglog::OptimizeError;
use egglog_engine::EGraph;

#[test]
fn applied_rewrites_do_not_replay_when_another_pass_revisits_the_same_body() -> Result<(), OptimizeError> {
    let mut graph = EGraph::default();
    load(&mut graph)?;
    graph.parse_and_run_program(
        None,
        r#"
        (let scalar (ScalarInfo (ScalarElement (I32))))
        (let first (Let (StatementTag 0) (vec-of (BindValue (ValueId 0) scalar)) (vec-of)
            (Use (ValueOperand (LiteralBits (I32) 1)))))
        (let second (Let (StatementTag 0) (vec-of (BindValue (ValueId 0) scalar)) (vec-of)
            (Use (ValueOperand (LiteralBits (I32) 2)))))
        (let name (FunctionId 0 "rewrite"))
        (let a (MemoryProgram (vec-of (DefineFunction name (vec-of) (vec-of)
            (Block (vec-of first) (vec-of)))) (vec-of) (vec-of)))
        (let b (MemoryProgram (vec-of (DefineFunction name (vec-of) (vec-of)
            (Block (vec-of second) (vec-of)))) (vec-of) (vec-of)))
        (let forward (Rename (map-empty) (map-empty) (map-empty) (map-empty)
            (map-insert (map-empty) (StatementTag 0) (vec-of second))))
        (let backward (Rename (map-empty) (map-empty) (map-empty) (map-empty)
            (map-insert (map-empty) (StatementTag 0) (vec-of first))))
        (set (PendingRewrite "forward" a) (set-of forward))
        (set (PendingRewrite "backward" b) (set-of backward))
        (RewriteDemand "forward" a) (RewriteDemand "backward" b)
        (CurrentProgram a)
        (run-schedule (saturate memory-rewrite memory-rename))
        (check (CurrentProgram a))
        (fail (check (RewriteDemand "forward" a)))
        (fail (check (RewriteDemand "backward" b)))
        (let third (Let (StatementTag 0) (vec-of (BindValue (ValueId 0) scalar)) (vec-of)
            (Use (ValueOperand (LiteralBits (I32) 3)))))
        (let c (MemoryProgram (vec-of (DefineFunction name (vec-of) (vec-of)
            (Block (vec-of third) (vec-of)))) (vec-of) (vec-of)))
        (let alternative (Rename (map-empty) (map-empty) (map-empty) (map-empty)
            (map-insert (map-empty) (StatementTag 0) (vec-of third))))
        (set (PendingRewrite "fusion" a) (set-of forward))
        (set (PendingRewrite "simplify" a) (set-of alternative))
        (RewriteDemand "fusion" a) (RewriteDemand "simplify" a)
        (run-schedule (saturate memory-rewrite-fusion memory-rename))
        (check (CurrentProgram b))
        (fail (check (CurrentProgram c)))
        (check (RewriteDemand "simplify" a))
        (delete (CurrentProgram b)) (CurrentProgram a)
        (run-schedule (saturate memory-rewrite-simplify memory-rename))
        (check (CurrentProgram c))
        (fail (check (CurrentProgram b)))
    "#,
    )?;
    Ok(())
}

#[test]
fn reduction_combiner_cannot_capture_a_lane_index() -> Result<(), OptimizeError> {
    let mut graph = EGraph::default();
    load(&mut graph)?;
    graph.parse_and_run_program(None, r#"
        (let function (FunctionId 0 "reduce"))
        (let lane (ValueId 0))
        (let combine (FunctionBody (vec-of) (vec-of (ReturnScalar (ScalarElement (I64))))
            (Block (vec-of) (vec-of (Yield (vec-of) (ValueOperand (Var lane)))))))
        (let definition (DefineFunction function (vec-of) (vec-of)
            (Block (vec-of (Let (StatementTag 0) (vec-of) (vec-of)
                (Parallel (Unscheduled) (IterationSpace (ValueId 1) (vec-of (DenseAxis lane (IndexConstant 4))))
                    (ParallelBlock (vec-of)
                        (vec-of (Fold (Total) (Reduction (Ordered) (vec-of (LiteralBits (I64) 0)) combine)
                            (vec-of (Yield (vec-of) (ValueOperand (Var lane)))))))))) (vec-of))))
        (CheckFunction definition)
        (run-schedule (saturate memory-names) (saturate memory-scopes))
        (check (InvalidScope (ReducerScope (NestedScope (FunctionScope definition) (StatementTag 0) 0) 0)
            "operand or annotation outside its scope, or accumulator already consumed"))
    "#)?;
    Ok(())
}

#[test]
fn fusion_forwards_pointwise_reads_and_retains_both_results() -> Result<(), OptimizeError> {
    let mut graph = fusion_case("(Var j)")?;
    super::validate(&mut graph)?;
    super::super::fusion::run(&mut graph)?;
    super::validate(&mut graph)?;
    graph.parse_and_run_program(
        None,
        r#"
        (let forwarded (Let (StatementTag 11) (vec-of (BindValue y scalar)) (vec-of)
            (Use (ValueOperand (Var x)))))
        (let combined (Let (StatementTag 0) (vec-of output-a output-b) (vec-of)
            (Parallel (Unscheduled) da
                (ParallelBlock (vec-of producer-value forwarded)
                    (vec-of (Collect (vec-of (Yield (vec-of) (ValueOperand (Var x)))))
                            (Collect (vec-of (Yield (vec-of) (ValueOperand (Var y))))))))))
        (let expected (DefineFunction name (vec-of) signature (Block (vec-of combined) results)))
        (check (CurrentProgram (MemoryProgram (vec-of expected) (vec-of) (vec-of))))
        (check (= (ScopeAt (FunctionScope expected) 1) (set-of (ValueNameRef a) (ValueNameRef b))))
    "#,
    )?;
    Ok(())
}

#[test]
fn fusion_preserves_nonpointwise_dependencies() -> Result<(), OptimizeError> {
    let mut graph = fusion_case("(LiteralBits (I32) 0)")?;
    super::super::fusion::run(&mut graph)?;
    graph.parse_and_run_program(None, "(check (CurrentProgram original))")?;
    Ok(())
}

fn fusion_case(index: &str) -> Result<EGraph, OptimizeError> {
    let mut graph = EGraph::default();
    load(&mut graph)?;
    graph.parse_and_run_program(None, &format!(r#"
        (let name (FunctionId 0 "pointwise"))
        (let i (ValueId 0)) (let j (ValueId 1))
        (let x (ValueId 2)) (let y (ValueId 3))
        (let a (ValueId 4)) (let b (ValueId 5))
        (let scalar (ScalarInfo (ScalarElement (I32))))
        (let array (ArrayType (ValueArray) (vec-of (FixedExtent 4)) (ElementValue (ScalarElement (I32)))))
        (let output-a (BindValue a (LogicalValue array)))
        (let output-b (BindValue b (LogicalValue array)))
        (let da (IterationSpace (ValueId 6) (vec-of (DenseAxis i (IndexConstant 4)))))
        (let db (IterationSpace (ValueId 7) (vec-of (DenseAxis j (IndexConstant 4)))))
        (let producer-value (Let (StatementTag 10) (vec-of (BindValue x scalar)) (vec-of)
            (Use (ValueOperand (Var i)))))
        (let first (Let (StatementTag 0) (vec-of output-a) (vec-of)
            (Parallel (Unscheduled) da (ParallelBlock (vec-of producer-value)
                (vec-of (Collect (vec-of (Yield (vec-of) (ValueOperand (Var x))))))))))
        (let second (Let (StatementTag 1) (vec-of output-b) (vec-of)
            (Parallel (Unscheduled) db (ParallelBlock
                (vec-of (Let (StatementTag 11) (vec-of (BindValue y scalar)) (vec-of)
                    (Index a (vec-of {index}))))
                (vec-of (Collect (vec-of (Yield (vec-of) (ValueOperand (Var y))))))))))
        (let signature (vec-of (LogicalReturn array) (LogicalReturn array)))
        (let results (vec-of (Yield (vec-of) (ValueOperand (Var a))) (Yield (vec-of) (ValueOperand (Var b)))))
        (let definition (DefineFunction name (vec-of) signature (Block (vec-of first second) results)))
        (let original (MemoryProgram (vec-of definition) (vec-of) (vec-of)))
        (CurrentProgram original)
    "#))?;
    Ok(graph)
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
        graph.parse_and_run_program(None, "(run-schedule (saturate memory-names) (saturate memory-scopes))\n(fail (check (InvalidScope context reason)))")?;
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
        graph.parse_and_run_program(
            None,
            r#"
            (check (CurrentProgram p) (ProgramFunction p f)
                (ContainsLogicalType (FunctionRoot f) (ArrayType ownership extents element)))
        "#,
        )?;
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
        crate::egglog::planning::schedule(&mut graph, crate::egglog::cost::CostModel::default())
            .map_err(|error| crate::egglog::output_error(format!("{text}: {error}")))?;
        graph.parse_and_run_program(None, r#"
            (check (CurrentProgram p) (ProgramFunction p f)
                (ContainsLogicalType (FunctionRoot f) (ArrayType ownership extents element)))
        "#)?;
    }
    Ok(())
}

#[test]
fn different_parameter_names_do_not_prove_disjoint_memory() -> Result<(), OptimizeError> {
    let mut graph = EGraph::default();
    load(&mut graph)?;
    graph.parse_and_run_program(
        None,
        r#"
        (let array (LogicalValue (ArrayType (StorageArray) (vec-of (FixedExtent 4))
            (ElementValue (ScalarElement (I32))))))
        (let function (DefineFunction (FunctionId 0 "aliases")
            (vec-of (Parameter (BindValue (ValueId 0) array) (Observe))
                    (Parameter (BindValue (ValueId 1) array) (Observe)))
            (vec-of) (Block (vec-of) (vec-of))))
        (CurrentProgram (MemoryProgram (vec-of function) (vec-of) (vec-of)))
    "#,
    )?;
    super::validate(&mut graph)?;
    graph.parse_and_run_program(
        None,
        r#"
        (run-schedule (saturate memory-effects))
        (CompareEffects (ArrayWrites function (ValueId 0)) (ArrayReads function (ValueId 1)))
        (run-schedule (saturate memory-effects))
        (check (= (Commutes (ArrayWrites function (ValueId 0)) (ArrayReads function (ValueId 1))) false))
    "#,
    )?;
    Ok(())
}

#[test]
fn consumption_follows_product_components_and_excludes_shape_witnesses() -> Result<(), OptimizeError> {
    let source = crate::compile_thru_tlc(
        r#"
        #[linked("consume_pair_impl")]
        extern consume_pair(p: (*[4]i32, [4]i32)) i32
        entry main(x: *[4]i32, y: *[4]i32) i32 = consume_pair((x,y))
    "#,
    )
    .unwrap();
    let (mut graph, _) = super::super::from_tlc::import(&source)?;
    super::validate(&mut graph)?;
    graph.parse_and_run_program(
        None,
        r#"
        (check (CurrentProgram program) (ProgramExternal program
          (DeclareExternal name params returns behavior accesses))
          (= (vec-length params) 1)
          (= (vec-get params 0) (Parameter (BindValue array (LogicalValue (ProductType fields)))
              (ProductDiet (vec-of (Consume) (Observe))))))
        (check (CurrentProgram program) (ContainsExp (ProgramRoot program) (Call name args returns))
          (= (vec-length args) 1)
          (= (vec-get args 0) (Pass array (ProductDiet (vec-of (Consume) (Observe))))))
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
        let mut program = crate::egglog::fuse(crate::egglog::from_tlc(&source)?)?;
        program.graph.parse_and_run_program(
            None,
            r#"
            (check (CurrentProgram p) (ProgramFunction p f)
                (ContainsExp (FunctionRoot f) expression) (= expression (Index array indices))
                (= effects (EffectsExp f expression)) (set-contains effects (OrderedEffect)))
            (fail (check (CurrentProgram p) (ProgramFunction p f)
                (= f (DefineFunction name params returns (Block statements results)))
                (ContainsEntry (ProgramRoot p) (EntryPoint name identity))
                (WalkStms statements i) (< i (vec-length statements))
                (= (vec-get statements i) (Let id pattern certs (Index array indices)))))
        "#,
        )?;
    }
    Ok(())
}

#[test]
fn allocation_bounds_do_not_assume_integer_sums_cannot_overflow() -> Result<(), OptimizeError> {
    for limit in [1, 2147483647] {
        let mut graph = EGraph::default();
        load(&mut graph)?;
        graph.parse_and_run_program(None, &format!(r#"
            (let i32 (ScalarElement (I32)))
            (let combine (FunctionBody
                (vec-of (Parameter (BindValue (ValueId 0) (ScalarInfo i32)) (Observe))
                        (Parameter (BindValue (ValueId 1) (ScalarInfo i32)) (Observe)))
                (vec-of (ReturnScalar i32))
                (Block (vec-of (Let (StatementTag 0) (vec-of (BindValue (ValueId 2) (ScalarInfo i32)))
                    (vec-of) (PrimOp "+" (vec-of (Var (ValueId 0)) (Var (ValueId 1))))))
                    (vec-of (Yield (vec-of) (ValueOperand (Var (ValueId 2))))))))
            (let function (DefineFunction (FunctionId 0 "bounded")
                (vec-of (Parameter (BindValue (ValueId 10) (ScalarInfo (ScalarElement (Bool)))) (Observe)))
                (vec-of (ReturnScalar i32))
                (Block (vec-of (Let (StatementTag 2) (vec-of (BindValue (ValueId 4) (ScalarInfo i32)))
                    (vec-of) (Parallel (Unscheduled)
                        (IterationSpace (ValueId 6) (vec-of (DenseAxis (ValueId 5) (IndexConstant 8))))
                        (ParallelBlock (vec-of (Let (StatementTag 1)
                            (vec-of (BindValue (ValueId 3) (ScalarInfo i32))) (vec-of)
                            (PrimOp "select" (vec-of (LiteralBits (I32) 0) (LiteralBits (I32) {limit}) (Var (ValueId 10))))))
                            (vec-of (Fold (Total) (Reduction (Commutative) (vec-of (LiteralBits (I32) 0)) combine)
                                (vec-of (Yield (vec-of) (ValueOperand (Var (ValueId 3)))))))))))
                    (vec-of (Yield (vec-of) (ValueOperand (Var (ValueId 4))))))))
            (CurrentProgram (MemoryProgram (vec-of function) (vec-of) (vec-of)))
            (run-schedule (saturate fusion-index memory-names) (saturate memory-scopes)
                (saturate schedule-cost))
        "#))?;
        let assertion = if limit == 1 {
            "(check (= (IndexUpper function (IndexValue (ValueId 4))) 8))"
        } else {
            "(fail (check (= upper (IndexUpper function (IndexValue (ValueId 4))))))"
        };
        graph.parse_and_run_program(None, assertion)?;
    }
    Ok(())
}

#[test]
fn consumed_tokens_cannot_be_reused() -> Result<(), OptimizeError> {
    for (parameters, statements, message) in [
        (
            "(Parameter (BindAccumulator (AccumulatorId 0) (AccumulatorInfo (vec-of (IndexConstant 4)) (vec-of (ScalarElement (I32))))) (Consume))",
            "(Let (StatementTag 0) (vec-of (BindAccumulator (AccumulatorId 1) (AccumulatorInfo (vec-of (IndexConstant 4)) (vec-of (ScalarElement (I32)))))) (vec-of) (Use (AccOperand (AccumulatorId 0))))
             (Let (StatementTag 1) (vec-of (BindAccumulator (AccumulatorId 2) (AccumulatorInfo (vec-of (IndexConstant 4)) (vec-of (ScalarElement (I32)))))) (vec-of) (Use (AccOperand (AccumulatorId 0))))",
            "scope",
        ),
    ] {
        let mut graph = EGraph::default();
        load(&mut graph)?;
        graph.parse_and_run_program(None, &format!(
            "(CurrentProgram (MemoryProgram (vec-of
                (DefineFunction (FunctionId 0 \"invalid\") (vec-of {parameters}) (vec-of)
                    (Block (vec-of {statements}) (vec-of)))) (vec-of) (vec-of)))"
        ))?;
        let error = super::validate(&mut graph).expect_err("invalid memory statements must be rejected");
        assert!(error.to_string().contains(message), "{error}");
    }
    Ok(())
}

#[test]
fn validation_checks_live_functions_and_ignores_retired_programs() -> Result<(), OptimizeError> {
    let mut graph = EGraph::default();
    load(&mut graph)?;
    graph.parse_and_run_program(None, r#"
        (let scalar (ScalarInfo (ScalarElement (I32))))
        (let parameter (Parameter (BindValue (ValueId 0) scalar) (Observe)))
        (let signature (vec-of (ReturnScalar (ScalarElement (I32)))))
        (let result (vec-of (Yield (vec-of) (ValueOperand (Var (ValueId 0))))))
        (let good (DefineFunction (FunctionId 0 "good") (vec-of parameter) signature (Block (vec-of) result)))
        (let missing (DefineFunction (FunctionId 1 "missing") (vec-of) signature (Block (vec-of) result)))
        (let duplicate (DefineFunction (FunctionId 2 "duplicate") (vec-of parameter parameter) signature (Block (vec-of) result)))
        (let good-program (MemoryProgram (vec-of good) (vec-of) (vec-of)))
        (let missing-program (MemoryProgram (vec-of good missing) (vec-of) (vec-of)))
        (let duplicate-program (MemoryProgram (vec-of good duplicate) (vec-of) (vec-of)))
        (CurrentProgram missing-program)
    "#)?;
    let error = super::validate(&mut graph).expect_err("every live function must be checked");
    assert!(error.to_string().contains("declared type"), "{error}");
    graph.parse_and_run_program(
        None,
        "(delete (CurrentProgram missing-program)) (CurrentProgram good-program)",
    )?;
    super::validate(&mut graph)?;
    graph.parse_and_run_program(
        None,
        "(delete (CurrentProgram good-program)) (CurrentProgram duplicate-program)",
    )?;
    let error = super::validate(&mut graph).expect_err("duplicate live binders must be rejected");
    assert!(error.to_string().contains("duplicate binders"), "{error}");
    Ok(())
}

#[test]
fn accumulator_name_filter_preserves_only_capabilities() -> Result<(), OptimizeError> {
    let mut graph = EGraph::default();
    load(&mut graph)?;
    graph.parse_and_run_program(
        None,
        r#"
        (let names (set-of (ValueNameRef (ValueId 0)) (BlockNameRef (MemoryId 1))
            (AccNameRef (AccumulatorId 2)) (ValueNameRef (ValueId 3))
            (AccNameRef (AccumulatorId 4))))
        (AccumulatorNamesDemand names)
        (AccumulatorNamesDemand (set-empty))
        (run-schedule (saturate memory-scopes))
        (check (= (AccumulatorNames names)
            (set-of (AccNameRef (AccumulatorId 2)) (AccNameRef (AccumulatorId 4)))))
        (check (= (AccumulatorNames (set-empty)) (set-empty)))
    "#,
    )?;
    Ok(())
}
