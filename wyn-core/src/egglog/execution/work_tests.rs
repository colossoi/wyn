use super::*;
use crate::egglog::dependencies::analyze;
use crate::egglog::{from_tlc, fuse, insert_expressions, simplify};

fn cost(source: &str) -> u8 {
    let tlc = crate::tlc::infer_input_slice_bounds(crate::compile_thru_tlc(source).unwrap());
    let data = crate::egglog::place(
        simplify(
            insert_expressions(fuse(from_tlc(&tlc).unwrap()).unwrap()).unwrap(),
            false,
        )
        .unwrap(),
    )
    .unwrap()
    .ir;
    let summary = analyze(&data);
    let root = data.definitions[data.entries.values().next().unwrap().definition].body;
    Costs::new(&data, &summary.live).count([Node::Region(root)], WORK_BUDGET)
}

#[test]
fn select_work_includes_both_operands_and_shared_work_once() {
    let program = crate::egglog::select_tests::eager_program(
        "entry choose(c:bool,x:i32) i32=if c then x*x+1 else x*x+2",
    );
    let data =
        crate::egglog::place(simplify(insert_expressions(program).unwrap(), false).unwrap()).unwrap().ir;
    let summary = analyze(&data);
    let root = data.definitions[data.entries.values().next().unwrap().definition].body;
    // One multiply, two additions, and one eager select.
    assert_eq!(
        Costs::new(&data, &summary.live).count([Node::Region(root)], WORK_BUDGET),
        4
    );
}

#[test]
fn shared_arithmetic_counts_once_and_distinct_work_saturates() {
    let mut body = "x".to_owned();
    for _ in 0..7 {
        body = format!("let y={body} in y*y+1");
    }
    assert_eq!(cost(&format!("entry main(x:i32) i32={body}")), 14);
    let fields =
        (0..24).map(|i| format!("(x+{})*(x+{})", i * 2 + 1, i * 2 + 2)).collect::<Vec<_>>().join(",");
    assert_eq!(
        cost(&format!("entry main(x:i32) [24]i32=[{fields}]")),
        OVER_BUDGET
    );
}

#[test]
fn folded_constants_contribute_no_arithmetic_work() {
    assert_eq!(cost("entry main(x:i32) i32=x+(2*3+4)"), 1);
}

#[test]
fn requesting_only_selected_costs_preserves_shared_callee_estimates() {
    let tlc = crate::tlc::infer_input_slice_bounds(
        crate::compile_thru_tlc(
            "def setup(x:i32) i32=(x*x+1)*(x+2)
         entry main(xs:[]i32) []i32=map(|x:i32|setup(x)+setup(x+1)+xs[0],xs)",
        )
        .unwrap(),
    );
    let data = crate::egglog::place(
        simplify(
            insert_expressions(fuse(from_tlc(&tlc).unwrap()).unwrap()).unwrap(),
            false,
        )
        .unwrap(),
    )
    .unwrap()
    .ir;
    let summary = analyze(&data);
    assert!(summary.live.len() > 1);
    let all = estimate(&data, &summary.live, summary.live.iter().copied());
    for &op in &summary.live {
        let selected = estimate(&data, &summary.live, [op]);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[&op], all[&op]);
    }
}

#[test]
fn separate_helper_invocations_do_not_share_argument_bindings() {
    let mut body = "x".to_owned();
    for _ in 0..7 {
        body = format!("let y={body} in y*y+1");
    }
    let work = cost(&format!(
        "def setup(x:i32) i32={body}
         entry main(a:i32,b:i32) (i32,i32)=(setup(a),setup(b))"
    ));
    assert!(
        (28..=30).contains(&work),
        "both 14-operation bodies must be counted: {work}"
    );
}

#[test]
fn choices_count_eager_shared_work_once_and_lazy_arms_separately() {
    // Safe arithmetic becomes an eager select; its shared multiply counts once.
    assert_eq!(cost("entry main(x:i32,c:bool) i32=if c then x*x+1 else x*x+2"), 4);
    // Guarded division retains two lazy arms, each with its own work estimate.
    assert_eq!(
        cost("entry main(x:i32,d:i32,c:bool) i32=if c then x/d+1 else x/d+2"),
        5
    );
}

#[test]
fn deeply_shared_choices_stop_at_the_budget_without_expanding_all_paths() {
    use crate::egglog::data::intern_expr;
    let tlc = crate::tlc::infer_input_slice_bounds(
        crate::compile_thru_tlc("entry main(x:i32,c:bool) i32=x+1").unwrap(),
    );
    let mut data = simplify(
        insert_expressions(fuse(from_tlc(&tlc).unwrap()).unwrap()).unwrap(),
        false,
    )
    .unwrap()
    .ir;
    let root = data.definitions[data.entries.values().next().unwrap().definition].body;
    let mut value = data.regions[root].results[0];
    let ExprKind::PureApp { function, .. } = data.expressions[value].kind else {
        panic!("addition")
    };
    let ty = data.expressions[value].ty;
    let parameter = data.regions[root].parameters[1];
    let condition_ty = data.parameters[parameter].ty;
    let condition = intern_expr(&mut data, condition_ty, ExprKind::Parameter(parameter));
    let one = intern_expr(&mut data, ty, ExprKind::Int("1".into()));
    let two = intern_expr(&mut data, ty, ExprKind::Int("2".into()));
    for _ in 0..80 {
        let yes = intern_expr(
            &mut data,
            ty,
            ExprKind::PureApp {
                function,
                args: vec![value, one],
            },
        );
        let no = intern_expr(
            &mut data,
            ty,
            ExprKind::PureApp {
                function,
                args: vec![value, two],
            },
        );
        value = intern_expr(
            &mut data,
            ty,
            ExprKind::If {
                condition,
                then_value: yes,
                else_value: no,
            },
        );
    }
    data.regions[root].results = vec![value];
    let summary = analyze(&data);
    assert_eq!(
        Costs::new(&data, &summary.live).count([Node::Region(root)], WORK_BUDGET),
        OVER_BUDGET
    );
}

#[test]
fn stored_loop_result_and_array_length_do_not_repeat_producer_work() {
    let source = "entry main(xs:[]i32,n:i32) []i32 =
        let a=loop acc=xs[0] for i<n do acc+i in map(|x:i32|x+a+length(xs),xs)";
    let tlc = crate::tlc::infer_input_slice_bounds(crate::compile_thru_tlc(source).unwrap());
    let program = crate::egglog::place(
        simplify(
            insert_expressions(fuse(from_tlc(&tlc).unwrap()).unwrap()).unwrap(),
            false,
        )
        .unwrap(),
    )
    .unwrap();
    let length = program
        .ir
        .operations
        .iter()
        .find_map(|(&op, data)| length_source(&program.ir, &data.kind).map(|_| op))
        .unwrap();
    let summary = analyze(&program.ir);
    assert_eq!(estimate(&program.ir, &summary.live, [length])[&length], 1);
    let mut costs = Costs::new(&program.ir, &summary.live);
    let result = program
        .ir
        .expressions
        .iter()
        .find_map(|(&e, data)| match data.kind {
            ExprKind::OperationResult(op)
                if matches!(program.ir.operations[op].kind, OperationKind::Loop { .. }) =>
            {
                Some(e)
            }
            _ => None,
        })
        .unwrap();
    assert_eq!(costs.count([Node::Expr(result)], WORK_BUDGET), 0);
}

#[test]
fn length_counts_metadata_arithmetic_without_evaluating_elements() {
    assert_eq!(cost("entry main(xs:[]i32,x:i32) i32=length(xs[0..x*x+x])"), 3);
    assert_eq!(cost("entry main(xs:[]i32,x:i32) i32=length(xs[x*x..x*x+x])"), 3);
    assert_eq!(cost("entry main(x:i32) i32=length([x*x+x,x*x*x])"), 1);
    assert_eq!(cost("entry main(x:i32) i32=length(iota(x*x+x))"), 3);
    let mut bounds: Vec<_> = (0..24).map(|i| format!("(x+{})*(x+{})", i * 2 + 1, i * 2 + 2)).collect();
    while bounds.len() > 1 {
        bounds = bounds
            .chunks(2)
            .map(|xs| if xs.len() == 2 { format!("({}+{})", xs[0], xs[1]) } else { xs[0].clone() })
            .collect();
    }
    let bounds = &bounds[0];
    assert_eq!(
        cost(&format!(
            "entry main(xs:[]i32,x:i32) i32=length(xs[0..({bounds})])"
        )),
        OVER_BUDGET
    );
}
