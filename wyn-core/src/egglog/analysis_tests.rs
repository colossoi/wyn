use crate::egglog::{fusion, memory};

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
