use crate::egglog::{fusion, memory};

#[test]
fn constant_index_arithmetic_does_not_prove_wrapping_counts() {
    let mut graph = fusion::new_graph().unwrap();
    graph
        .parse_and_run_program(
            None,
            r#"
        (let function (DefineFunction (FunctionId 0 "counts") (vec-of) (vec-of) (Block (vec-of) (vec-of))))
        (let difference (IndexSub (IndexConstant 3) (IndexConstant 1)))
        (let sum (IndexAdd (IndexConstant 2147483647) (IndexConstant 1)))
        (let product (IndexMul (IndexConstant 65536) (IndexConstant 65536)))
        (ContainsIndexExp (FunctionRoot function) difference)
        (ContainsIndexExp (FunctionRoot function) sum)
        (ContainsIndexExp (FunctionRoot function) product)
        (run-schedule (saturate memory-names) (saturate schedule-cost))
        (check (= (StaticIndex function difference) 2))
        (fail (check (= count (StaticIndex function sum))))
        (fail (check (= count (StaticIndex function product))))
    "#,
        )
        .unwrap();
}

#[test]
fn logical_values_and_storage_reads_have_distinct_effects() {
    let mut graph = fusion::new_graph().unwrap();
    graph.parse_and_run_program(None, r#"
        (let function (DefineFunction (FunctionId 0 "effects") (vec-of) (vec-of) (Block (vec-of) (vec-of))))
        (let value (ValueId 0)) (let storage (ValueId 1))
        (ContainsValueName (FunctionRoot function) value)
        (ContainsValueName (FunctionRoot function) storage)
        (ContainsBinding (FunctionRoot function) (BindValue value (LogicalValue (ArrayType (ValueArray) (vec-of (FixedExtent 4)) (ElementValue (ScalarElement (I32)))))))
        (ContainsBinding (FunctionRoot function) (BindValue storage (LogicalValue (ArrayType (StorageArray) (vec-of (FixedExtent 4)) (ElementValue (ScalarElement (I32)))))))
        (CompareEffects (set-of (OrderedEffect)) (set-of (SharedRead)))
        (CompareEffects (set-of (OrderedEffect)) (set-of (SharedWrite)))
        (CompareEffects (set-of (OrderedEffect)) (set-of (OrderedEffect)))
        (run-schedule (saturate memory-effects))
        (check (= (ArrayReads function value) (set-empty)))
        (check (= (ArrayReads function storage) (set-of (SharedRead))))
        (check (= (Commutes (set-of (OrderedEffect)) (set-of (SharedRead))) true))
        (check (= (Commutes (set-of (OrderedEffect)) (set-of (SharedWrite))) false))
        (check (= (Commutes (set-of (OrderedEffect)) (set-of (OrderedEffect))) false))
    "#).unwrap();
}

#[test]
fn bounded_work_counts_every_loop_iteration() {
    let mut graph = fusion::new_graph().unwrap();
    graph.parse_and_run_program(None, r#"
        (let one (LiteralBits (I32) 1))
        (let inner (Block (vec-of (Let (StatementTag 0)
            (vec-of (BindValue (ValueId 0) (ScalarInfo (ScalarElement (I32)))))
            (vec-of) (Use (ValueOperand one)))) (vec-of)))
        (let loop (Loop (vec-of) (Counted (ValueId 1) (IndexConstant 0) (IndexConstant 40) (IndexConstant 1)) (vec-of) inner))
        (let function (DefineFunction (FunctionId 0 "work") (vec-of) (vec-of)
            (Block (vec-of (Let (StatementTag 1) (vec-of) (vec-of) loop)) (vec-of))))
        (CurrentProgram (MemoryProgram (vec-of function) (vec-of) (vec-of)))
    "#).unwrap();
    memory::validate(&mut graph).unwrap();
    graph
        .parse_and_run_program(
            None,
            r#"
        (run-schedule (saturate schedule-cost))
        (check (= (BoundedWork function loop) (bigint 40)))
    "#,
        )
        .unwrap();
}
