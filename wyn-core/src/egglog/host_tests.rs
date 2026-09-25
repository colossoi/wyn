use super::*;
use crate::egglog::{insert_expressions, place, schedule, select_tests, simplify};

#[test]
fn select_host_extraction_evaluates_both_arms_before_choosing() {
    let program =
        select_tests::eager_program("entry choose(c:bool,x:i32,y:i32) i32=if c then x/y else y/x");
    let program = schedule(
        place(simplify(insert_expressions(program).unwrap(), false).unwrap()).unwrap(),
        crate::PipelineTopologyPolicy::AllowGenerated,
    )
    .unwrap();
    let id = program
        .expressions
        .iter()
        .find_map(|(&id, _)| program.ir.conditional_value(id).map(|_| id))
        .unwrap();
    let schedules = analyze(&program).schedules(&program).unwrap();
    let mut lower = Lower::new(&program, &schedules);
    for (&p, _) in &program.parameters {
        lower.parameters.insert(p, ScalarExpr::Local(format!("input-{}", p.as_u32())));
    }
    let value = lower.expression_body(id).unwrap();
    assert!(
        matches!(value, ScalarExpr::Apply { ref op, ty: ScalarType::I32, ref args }
        if op == "select" && args.len() == 3)
    );
    assert_eq!(
        lower
            .bindings
            .iter()
            .filter(|(_, v)| matches!(v, ScalarExpr::Apply { op, .. } if op == "div"))
            .count(),
        2
    );
}
