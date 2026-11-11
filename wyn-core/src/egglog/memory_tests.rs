use super::load;
use crate::egglog::OptimizeError;
use egglog_engine::EGraph;

#[test]
fn renaming_covers_memory_annotations_and_nested_bodies() -> Result<(), OptimizeError> {
    let mut graph = EGraph::default();
    load(&mut graph)?;
    graph.parse_and_run_program(
        None,
        r#"
        (let size (ValueId 0))
        (let renamed-size (ValueId 10))
        (let storage (MemoryId 0))
        (let renamed-storage (MemoryId 10))
        (let input (ValueId 2))
        (let output (ValueId 1))
        (let renamed-output (ValueId 11))
        (let layout (Layout (IndexValue size) (vec-of (Dim (IndexConstant 1) (IndexValue size)))))
        (let binding (BindValue output (ArrayInfo (ScalarElement (I32)) (ArrayIn storage layout))))
        (let body (Block
            (vec-of (Let (StatementTag 0) (vec-of binding) (vec-of) (Copy input)))
            (vec-of (Yield (vec-of) (ValueOperand (Var output))))))
        (let env (Rename
            (map-insert (map-insert (map-empty) size renamed-size) output renamed-output)
            (map-insert (map-empty) storage renamed-storage)
            (map-empty) (map-insert (map-empty) (StatementTag 0) (StatementTag 10)) (map-empty)))
        (RenameBodyDemand env body)
        (run-schedule (saturate memory-rename memory-names))
        (check (= (RenameBody env body)
            (Block
                (vec-of (Let (StatementTag 10)
                    (vec-of (BindValue renamed-output (ArrayInfo (ScalarElement (I32))
                        (ArrayIn renamed-storage (Layout (IndexValue renamed-size)
                            (vec-of (Dim (IndexConstant 1) (IndexValue renamed-size))))))))
                    (vec-of) (Copy input)))
                (vec-of (Yield (vec-of) (ValueOperand (Var renamed-output)))))))
        (check (set-contains (NamesBody body) (ValueNameRef size)))
        (check (set-contains (NamesBody body) (BlockNameRef storage)))
        (check (set-contains (BindersBody body) (ValueNameRef output)))
        (check (set-not-contains (BindersBody body) (BlockNameRef storage)))
    "#,
    )?;
    Ok(())
}

#[test]
fn statement_replacement_supports_splicing_deletion_and_retained_ids() -> Result<(), OptimizeError> {
    let mut graph = EGraph::default();
    load(&mut graph)?;
    graph.parse_and_run_program(None, r#"
        (let first (Let (StatementTag 0) (vec-of) (vec-of) (Use (ValueOperand (LiteralBits (UnitType) 0)))))
        (let removed (Let (StatementTag 1) (vec-of) (vec-of) (Use (ValueOperand (LiteralBits (UnitType) 0)))))
        (let inserted (Let (StatementTag 0) (vec-of) (vec-of) (Barrier (WorkgroupFence))))
        (let extra (Let (StatementTag 2) (vec-of) (vec-of) (Barrier (WorkgroupFence))))
        (let body (Block (vec-of first removed) (vec-of)))
        (let env (Rename (map-empty) (map-empty) (map-empty) (map-empty)
            (map-insert (map-insert (map-empty) (StatementTag 0) (vec-of inserted extra)) (StatementTag 1) (vec-of))))
        (RenameBodyDemand env body)
        (run-schedule (saturate memory-rename))
        (check (= (RenameBody env body) (Block (vec-of inserted extra) (vec-of))))
    "#)?;
    Ok(())
}

#[test]
fn scope_check_rejects_annotation_capture_from_a_sibling_branch() -> Result<(), OptimizeError> {
    let mut graph = EGraph::default();
    load(&mut graph)?;
    graph.parse_and_run_program(
        None,
        r#"
        (let function (FunctionId 0 "capture"))
        (let storage (MemoryId 0))
        (let hidden (ValueId 0))
        (let output (ValueId 1))
        (let left (Block
            (vec-of (Let (StatementTag 0) (vec-of (BindValue hidden (ScalarInfo (ScalarElement (I64)))))
                (vec-of) (Use (ValueOperand (LiteralBits (I64) 4)))))
            (vec-of)))
        (let right (Block
            (vec-of (Let (StatementTag 1)
                (vec-of (BindValue output (ArrayInfo (ScalarElement (I32))
                    (ArrayIn storage (Layout (IndexConstant 0)
                        (vec-of (Dim (IndexValue hidden) (IndexConstant 4))))))))
                (vec-of) (GenerateArray (Uninitialized (ScalarElement (I32)) (vec-of (IndexConstant 4))))))
            (vec-of)))
        (let definition (DefineFunction function
            (vec-of (Parameter (BindBlock storage (IndexConstant 16)) (Observe)))
            (vec-of) (Block
                (vec-of (Let (StatementTag 2) (vec-of) (vec-of)
                    (If (LiteralBits (Bool) 1) (vec-of) left right))) (vec-of))))
        (CheckFunction definition)
        (run-schedule (saturate memory-names) (saturate memory-scopes))
        (check (InvalidScope (NestedScope (FunctionScope definition) (StatementTag 2) 1)
            "operand or annotation outside its scope, or accumulator already consumed"))
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
fn fusion_forwards_pointwise_loads_and_retains_both_array_bindings() -> Result<(), OptimizeError> {
    let mut graph = fusion_case("(IndexValue j)", 1)?;
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
        (let expected (DefineFunction name (vec-of) signature
            (Block (vec-of alloc-a alloc-b combined) results)))
        (check (CurrentProgram (MemoryProgram (vec-of expected) (vec-of) (vec-of))))
        (CheckFunction expected)
        (run-schedule (saturate memory-names) (saturate memory-scopes))
        (check (= (ScopeAt (FunctionScope expected) 3)
            (set-of (BlockNameRef (MemoryId 0)) (BlockNameRef (MemoryId 1))
                    (ValueNameRef a) (ValueNameRef b))))
    "#,
    )?;
    Ok(())
}

#[test]
fn fusion_preserves_nonpointwise_dependencies_and_aliased_writes() -> Result<(), OptimizeError> {
    for (index, destination) in [("(IndexConstant 0)", 1), ("(IndexValue j)", 0)] {
        let mut graph = fusion_case(index, destination)?;
        super::super::fusion::run(&mut graph)?;
        graph.parse_and_run_program(None, "(check (CurrentProgram original))")?;
    }
    Ok(())
}

fn fusion_case(index: &str, destination: i64) -> Result<EGraph, OptimizeError> {
    let mut graph = EGraph::default();
    load(&mut graph)?;
    graph.parse_and_run_program(None, &format!(r#"
        (let name (FunctionId 0 "pointwise"))
        (let i (ValueId 0)) (let j (ValueId 1))
        (let x (ValueId 2)) (let y (ValueId 3))
        (let a (ValueId 4)) (let b (ValueId 5))
        (let scalar (ScalarInfo (ScalarElement (I32))))
        (let layout (Layout (IndexConstant 0) (vec-of (Dim (IndexConstant 1) (IndexConstant 4)))))
        (let output-a (BindValue a (ArrayInfo (ScalarElement (I32)) (ArrayIn (MemoryId 0) layout))))
        (let output-b (BindValue b (ArrayInfo (ScalarElement (I32)) (ArrayIn (MemoryId {destination}) layout))))
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
                    (Load a (vec-of {index}))))
                (vec-of (Collect (vec-of (Yield (vec-of) (ValueOperand (Var y))))))))))
        (let alloc-a (Let (StatementTag 20) (vec-of (BindBlock (MemoryId 0) (IndexConstant 16)))
            (vec-of) (Alloc (DeviceSpace) (IndexConstant 16))))
        (let alloc-b (Let (StatementTag 21) (vec-of (BindBlock (MemoryId 1) (IndexConstant 16)))
            (vec-of) (Alloc (DeviceSpace) (IndexConstant 16))))
        (let signature (vec-of
            (ReturnMemory (FreeIndex (IndexConstant 16))) (ReturnMemory (FreeIndex (IndexConstant 16)))
            (ReturnArray (ScalarElement (I32)) (ResultBlock 0)
                (ExtendedLayout (FreeIndex (IndexConstant 0)) (vec-of (ExtendedDimension (FreeIndex (IndexConstant 1)) (FreeIndex (IndexConstant 4))))))
            (ReturnArray (ScalarElement (I32)) (ResultBlock {destination})
                (ExtendedLayout (FreeIndex (IndexConstant 0)) (vec-of (ExtendedDimension (FreeIndex (IndexConstant 1)) (FreeIndex (IndexConstant 4))))))))
        (let results (vec-of
            (Yield (vec-of) (BlockOperand (MemoryId 0)))
            (Yield (vec-of) (BlockOperand (MemoryId 1))) (Yield (vec-of) (ValueOperand (Var a))) (Yield (vec-of) (ValueOperand (Var b)))))
        (let definition (DefineFunction name (vec-of) signature (Block (vec-of alloc-a alloc-b first second) results)))
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
fn source_import_normalizes_accumulator_operations() -> Result<(), OptimizeError> {
    for text in [
        "entry main(xs:[4]i32) [4]i32 = scatter(xs,[0,1],[7,8])",
        "entry main(xs:[64]i32) []i32 = filter(|x|x>0,xs)",
        "entry main(dest:[3]i32,xs:[4]i32) [3]i32 = reduce_by_index(dest,(+),0i32,[0,1,1,2],xs)",
        "entry main(dest:*[2][2]i32,items:*[2][3](i32,i32)) ([2][2]i32,[2]u32,u32) = bucket_scatter_2d(dest,items)",
    ] {
        let source = crate::compile_thru_tlc(text).unwrap();
        let (mut graph, _) = super::super::from_tlc::import(&source)
            .map_err(|error| crate::egglog::output_error(format!("{text}: {error}")))?;
        super::validate(&mut graph)
            .map_err(|error| crate::egglog::output_error(format!("{text}: {error}")))?;
        graph.parse_and_run_program(None, "(check (CurrentProgram program) (ProgramFunction program function)\n(ContainsExp (FunctionRoot function) (WithAcc inputs body)))")?;
    }
    Ok(())
}

#[test]
fn logical_array_fusion_is_not_separated_by_allocations() -> Result<(), OptimizeError> {
    let source =
        crate::compile_thru_tlc("entry main(xs:[64]i32) [64]i32 = map(|x|x*2,map(|x|x+1,xs))").unwrap();
    let (mut graph, _) = super::super::from_tlc::import(&source)?;
    super::validate(&mut graph)?;
    graph.parse_and_run_program(
        None,
        r#"
        (fail (check (CurrentProgram p) (ProgramFunction p f)
            (ContainsExp (FunctionRoot f) (Alloc space bytes))))
        (check (CurrentProgram p) (ProgramFunction p f)
            (= f (DefineFunction (FunctionId id "main") params returns (Block statements results)))
            (= (vec-length statements) 2))
    "#,
    )?;
    super::super::fusion::run(&mut graph)?;
    super::validate(&mut graph)?;
    graph.parse_and_run_program(
        None,
        r#"
        (check (CurrentProgram p) (ProgramFunction p f)
            (= f (DefineFunction (FunctionId id "main") params returns (Block statements results)))
            (= (vec-length statements) 1)
            (= (vec-get statements 0) (Let tag pattern certs (Parallel level domain body))))
    "#,
    )?;
    Ok(())
}

#[test]
fn component_normalization_handles_storage_products_and_rectangular_rows() -> Result<(), OptimizeError> {
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
            (fail (check (CurrentProgram p) (ProgramFunction p f)
                (ContainsExp (FunctionRoot f) (Alloc space bytes))))
        "#,
        )?;
    }
    Ok(())
}

#[test]
fn allocation_elaborates_fused_arrays_and_control_results() -> Result<(), OptimizeError> {
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
            (fail (check (CurrentProgram p) (ProgramFunction p f)
                (ContainsValueInfo (FunctionRoot f) (ArrayValue element shape))))
            (fail (check (CurrentProgram p) (ProgramFunction p f)
                (ContainsReturnInfo (FunctionRoot f) (ReturnArrayValue element shape))))
            (check (CurrentProgram p) (ProgramFunction p f)
                (ContainsValueInfo (FunctionRoot f) (ArrayInfo element storage)))
        "#)?;
    }
    Ok(())
}

#[test]
fn different_parameter_names_do_not_prove_disjoint_memory() -> Result<(), OptimizeError> {
    let mut graph = fusion_case("(IndexValue j)", 1)?;
    graph.parse_and_run_program(
        None,
        r#"
        (delete (CurrentProgram original))
        (let borrowed (DefineFunction name
            (vec-of (Parameter (BindBlock (MemoryId 0) (IndexConstant 16)) (Observe))
                    (Parameter (BindBlock (MemoryId 1) (IndexConstant 16)) (Observe)))
            signature (Block (vec-of first second) results)))
        (let borrowed-program (MemoryProgram (vec-of borrowed) (vec-of) (vec-of)))
        (CurrentProgram borrowed-program)
    "#,
    )?;
    super::super::fusion::run(&mut graph)?;
    graph.parse_and_run_program(None, "(check (CurrentProgram borrowed-program))")?;
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
          (= (vec-get params 0) (Parameter size (Observe)))
          (= (vec-get params 1) (Parameter (BindValue array (ArrayValue element shape)) (Consume)))
          (= (vec-get params 2) (Parameter other-size (Observe)))
          (= (vec-get params 3) (Parameter other-array (Observe))))
        (check (CurrentProgram program) (ContainsExp (ProgramRoot program) (Call name args returns))
          (= (vec-get args 0) (Pass size (Observe)))
          (= (vec-get args 1) (Pass array (Consume)))
          (= (vec-get args 2) (Pass other-size (Observe)))
          (= (vec-get args 3) (Pass other-array (Observe))))
    "#,
    )?;
    Ok(())
}

#[test]
fn helper_inlining_clones_whole_regions_and_preserves_scope() -> Result<(), OptimizeError> {
    let mut graph = EGraph::default();
    load(&mut graph)?;
    graph.parse_and_run_program(None, r#"
        (let element (ScalarElement (I32)))
        (let array (ArrayValue element (vec-of (IndexConstant 4))))
        (let scalar (ScalarInfo element))
        (let signature (vec-of (ReturnArrayValue element (vec-of (FreeIndex (IndexConstant 4))))))
        (let helper-name (FunctionId 0 "transform"))
        (let main-name (FunctionId 1 "main"))
        (let helper (DefineFunction helper-name
          (vec-of (Parameter (BindValue (ValueId 0) array) (Observe))
                  (Parameter (BindValue (ValueId 1) scalar) (Observe))) signature
          (Block (vec-of (Let (StatementTag 2) (vec-of (BindValue (ValueId 6) array)) (vec-of)
            (Parallel (Unscheduled) (IterationSpace (ValueId 2) (vec-of (DenseAxis (ValueId 3) (IndexConstant 4))))
              (ParallelBlock (vec-of
                (Let (StatementTag 0) (vec-of (BindValue (ValueId 4) scalar)) (vec-of)
                  (Load (ValueId 0) (vec-of (IndexValue (ValueId 3)))))
                (Let (StatementTag 1) (vec-of (BindValue (ValueId 5) scalar)) (vec-of)
                  (PrimOp "+" (vec-of (Var (ValueId 4)) (Var (ValueId 1))))))
                (vec-of (Collect (vec-of (Yield (vec-of) (ValueOperand (Var (ValueId 5)))))))))))
            (vec-of (Yield (vec-of) (ValueOperand (Var (ValueId 6))))))))
        (let main (DefineFunction main-name (vec-of (Parameter (BindValue (ValueId 10) array) (Observe))) signature
          (Block (vec-of
            (Let (StatementTag 3) (vec-of (BindValue (ValueId 11) array)) (vec-of)
              (Call helper-name (vec-of (Pass (ValueOperand (Var (ValueId 10))) (Observe))
                                       (Pass (ValueOperand (LiteralBits (I32) 1)) (Observe))) signature))
            (Let (StatementTag 4) (vec-of (BindValue (ValueId 12) array)) (vec-of)
              (Call helper-name (vec-of (Pass (ValueOperand (Var (ValueId 11))) (Observe))
                                       (Pass (ValueOperand (LiteralBits (I32) 2)) (Observe))) signature)))
            (vec-of (Yield (vec-of) (ValueOperand (Var (ValueId 12))))))))
        (set (PrimitiveEffects "+" (ScalarElement (I32))) (set-empty))
        (CurrentProgram (MemoryProgram (vec-of helper main) (vec-of) (vec-of (EntryPoint main-name 1))))
    "#)?;
    super::inline(&mut graph)?;
    graph.parse_and_run_program(None, r#"
        (fail (check (CurrentProgram program) (ContainsEntry (ProgramRoot program) (EntryPoint name identity))
            (ProgramFunction program function) (= function (DefineFunction name params returns body))
            (ContainsExp (FunctionRoot function) (Call callee args signature))))
        (check (CurrentProgram program) (ContainsStm (ProgramRoot program)
            (Let (CallStatement call original) pattern certs (Parallel execution domain body))))
    "#)?;
    super::simplify(&mut graph, true)?;
    super::super::fusion::run(&mut graph)?;
    super::simplify(&mut graph, true)?;
    super::validate(&mut graph)?;
    Ok(())
}

#[test]
fn invariant_region_motion_preserves_dependencies_and_traps() -> Result<(), OptimizeError> {
    let source =
        crate::compile_thru_tlc("entry main(xs:[4]i32) [4]i32 = map(|p|reduce((+),0i32,xs)+p,xs)").unwrap();
    let mut program = crate::egglog::fuse(crate::egglog::from_tlc(&source)?)?;
    program.graph.parse_and_run_program(None, r#"
        (check (CurrentProgram program) (ProgramFunction program function)
          (ContainsEntry (ProgramRoot program) (EntryPoint name identity))
          (= function (DefineFunction name params returns (Block statements results)))
          (WalkStms statements i) (< i (vec-length statements))
          (= (vec-get statements i) (Let id pattern certs (Parallel execution domain (ParallelBlock body output))))
          (WalkParallelResults output j) (< j (vec-length output))
          (= (vec-get output j) (Fold kind reducer values)))
    "#)?;
    let source = crate::compile_thru_tlc("entry main(xs:[]i32,d:i32) []i32 = map(|x|1/d+x,xs)").unwrap();
    let mut program = crate::egglog::fuse(crate::egglog::from_tlc(&source)?)?;
    program.graph.parse_and_run_program(
        None,
        r#"
        (fail (check (CurrentProgram program) (ProgramFunction program function)
          (ContainsEntry (ProgramRoot program) (EntryPoint name identity))
          (= function (DefineFunction name params returns (Block statements results)))
          (WalkStms statements i) (< i (vec-length statements))
          (= (vec-get statements i) (Let id pattern certs (PrimOp "/" arguments)))))
        (check (CurrentProgram program) (ContainsExp (ProgramRoot program) (PrimOp "/" arguments)))
    "#,
    )?;
    Ok(())
}

#[test]
fn layout_transforms_alias_strided_views_and_copy_noncontiguous_reshapes() -> Result<(), OptimizeError> {
    let mut graph = EGraph::default();
    load(&mut graph)?;
    graph.parse_and_run_program(None, r#"
        (let element (ScalarElement (I32)))
        (let block (MemoryId 0))
        (let matrix (Layout (IndexConstant 0) (vec-of
          (Dim (IndexConstant 3) (IndexConstant 2)) (Dim (IndexConstant 1) (IndexConstant 3)))))
        (let name (FunctionId 0 "layouts"))
        (let function (DefineFunction name
          (vec-of (Parameter (BindBlock block (IndexConstant 24)) (Observe))
            (Parameter (BindValue (ValueId 0) (ArrayInfo element (ArrayIn block matrix))) (Observe)))
          (vec-of)
          (Block (vec-of
            (Let (StatementTag 0) (vec-of (BindValue (ValueId 1) (ArrayValue element (vec-of (IndexConstant 3) (IndexConstant 2))))) (vec-of)
              (Permute (ValueId 0) (vec-of 1 0)))
            (Let (StatementTag 1) (vec-of (BindValue (ValueId 2) (ArrayValue element (vec-of (IndexConstant 1) (IndexConstant 3) (IndexConstant 2))))) (vec-of)
              (Reshape (ValueId 1) (vec-of (IndexConstant 1) (IndexConstant 3) (IndexConstant 2))))
            (Let (StatementTag 2) (vec-of (BindValue (ValueId 3) (ArrayValue element (vec-of (IndexConstant 6))))) (vec-of)
              (Reshape (ValueId 1) (vec-of (IndexConstant 6))))
            (Let (StatementTag 3) (vec-of (BindValue (ValueId 4) (ArrayValue element (vec-of (IndexConstant 6))))) (vec-of)
              (Reshape (ValueId 0) (vec-of (IndexConstant 6))))) (vec-of))))
        (CurrentProgram (MemoryProgram (vec-of function) (vec-of) (vec-of)))
        (set (FunctionHost function) false)
    "#)?;
    super::allocate(&mut graph)?;
    graph.parse_and_run_program(None, r#"
        (check (= (ArrayStorage function (ValueId 1)) (ArrayIn block
          (Layout (IndexConstant 0) (vec-of (Dim (IndexConstant 1) (IndexConstant 3)) (Dim (IndexConstant 3) (IndexConstant 2)))))))
        (check (= (ArrayStorage function (ValueId 2)) (ArrayIn block
          (Layout (IndexConstant 0) (vec-of (Dim (IndexConstant 0) (IndexConstant 1))
            (Dim (IndexConstant 1) (IndexConstant 3)) (Dim (IndexConstant 3) (IndexConstant 2)))))))
        (check (= (ArrayStorage function (ValueId 4)) (ArrayIn block layout)))
        (check (CurrentProgram program) (ContainsStm (ProgramRoot program)
          (Let (StatementTag 2) pattern certificates (Copy (ValueId 1)))))
        (check (CurrentProgram program) (ContainsStm (ProgramRoot program)
          (Let (AllocationTag (StatementTag 2) 0) pattern certificates (Alloc space bytes))))
        (fail (check (CurrentProgram program) (ContainsStm (ProgramRoot program)
          (Let (AllocationTag (StatementTag 1) 0) pattern certificates (Alloc space bytes)))))
    "#)?;
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
                (ContainsExp (FunctionRoot f) (Load array indices))
                (= (BoundedLoad f (Load array indices)) false))
            (fail (check (CurrentProgram p) (ProgramFunction p f)
                (= f (DefineFunction name params returns (Block statements results)))
                (ContainsEntry (ProgramRoot p) (EntryPoint name identity))
                (WalkStms statements i) (< i (vec-length statements))
                (= (vec-get statements i) (Let id pattern certs (Load array indices)))))
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
                (saturate schedule-cost dispatch-shapes allocation-bounds))
        "#))?;
        let assertion = if limit == 1 {
            "(check (= (IntegerUpper (Var (ValueId 4))) 8))"
        } else {
            "(fail (check (= upper (IntegerUpper (Var (ValueId 4))))))"
        };
        graph.parse_and_run_program(None, assertion)?;
    }
    Ok(())
}

#[test]
fn stores_cannot_bind_values_and_consumed_tokens_cannot_be_reused() -> Result<(), OptimizeError> {
    for (parameters, statements, message) in [
        (
            "(Parameter (BindValue (ValueId 0) (ArrayValue (ScalarElement (I32)) (vec-of (IndexConstant 4)))) (Observe))",
            "(Let (StatementTag 0) (vec-of (BindValue (ValueId 1) (ScalarInfo (ScalarElement (I32))))) (vec-of)
                (Store (ProvenInBounds) (ValueId 0) (vec-of (At (IndexConstant 0))) (LiteralBits (I32) 1)))",
            "memory stores do not bind results",
        ),
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
