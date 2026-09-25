use super::*;
use crate::egglog::data::{Evaluation, OperationKind};
use crate::egglog::select_tests::make_eager;
use crate::egglog::{self, Fused, Program};
use crate::tlc;
use crate::{
    compile_thru_tlc, lower_ssa_to_spirv, lower_ssa_to_wgsl, CodegenTarget, PipelineTopologyPolicy,
};

fn input(source: &str) -> Program<Fused> {
    let tlc = tlc::infer_input_slice_bounds(compile_thru_tlc(source).unwrap());
    egglog::fuse(egglog::from_tlc(&tlc).unwrap()).unwrap()
}

fn simplified(source: &str, algebra: bool) -> Program<egglog::Simplified> {
    egglog::simplify(egglog::insert_expressions(input(source)).unwrap(), algebra).unwrap()
}

fn result(data: &Ir) -> ExprId {
    let entry = data.entries.values().next().unwrap();
    data.regions[data.definitions[entry.definition].body].results[0]
}

fn shaders(program: Program<egglog::Simplified>) -> (String, wspirv::dr::Module) {
    let program = egglog::schedule(
        egglog::place(program).unwrap(),
        PipelineTopologyPolicy::AllowGenerated,
    )
    .unwrap();
    let ssa = egglog::to_ssa(&program, CodegenTarget::Portable).unwrap();
    let wgsl = lower_ssa_to_wgsl(ssa.clone()).unwrap();
    let module = naga::front::wgsl::parse_str(&wgsl).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    let words = lower_ssa_to_spirv(ssa).unwrap().spirv;
    (wgsl, wspirv::dr::load_words(words).unwrap())
}

#[test]
fn early_select_forms_before_scalar_optimization_and_preserves_unsafe_arms() {
    for (value, eager) in [
        ("if c then x else y", true),
        ("if c then x+1 else y+2", true),
        ("if c then x/y else 0", false),
        ("if c then (x/y)+1 else 0", false),
    ] {
        let source = format!("entry choose(c:bool,x:i32,y:i32) i32={value}");
        let program = egglog::insert_expressions(input(&source)).unwrap();
        assert_eq!(
            program.ir.conditional_value(result(&program.ir)).unwrap().1,
            if eager { Evaluation::Eager } else { Evaluation::Lazy },
            "{source}"
        );
        shaders(egglog::simplify(program, true).unwrap());
    }
}

#[test]
fn early_select_recognition_handles_many_equivalent_operand_terms() {
    let mut program =
        egglog::insert_expressions(input("entry choose(c:bool,a:i32,b:i32) i32=if c then a else b"))
            .unwrap();
    let data = &mut program.ir;
    let id = result(data);
    let (choice, _) = data.conditional_value(id).unwrap();
    let ExprKind::PureApp { function, .. } = data.expressions[id].kind else {
        panic!("select")
    };
    let mut graph = program.state.graph;
    fold::register(&mut graph);
    graph.parse_and_run_program(None, include_str!("../arithmetic.egg")).unwrap();
    graph.parse_and_run_program(None, include_str!("select.egg")).unwrap();
    graph.update(|sink| fold::facts(data, sink)).unwrap();
    // Recognition must depend on operand e-classes, not combinations of their
    // equivalent terms. Synthetic opaque terms avoid unrelated arithmetic rules.
    let mut alternatives = String::new();
    for (operand, id) in [choice.no, choice.yes, choice.condition].into_iter().enumerate() {
        let ty = data.expressions[id].ty.as_u32();
        for n in 0..256 {
            alternatives.push_str(&format!(
                "(union (SourceExpression {}) (Typed (TypeId {ty}) (Global (SymbolId {}))))\n",
                id.as_u32(),
                operand * 256 + n,
            ));
        }
    }
    graph.parse_and_run_program(None, &alternatives).unwrap();
    graph.parse_and_run_program(None, "(run-schedule (saturate (run arithmetic)))").unwrap();
    graph
        .parse_and_run_program(
            None,
            &format!(
                "(check (ValueSelect (SourceExpression {}) (TypeId {}) (SourceExpression {}) (SourceExpression {}) (SourceExpression {}) (SourceExpression {})))",
                id.as_u32(), data.expressions[id].ty.as_u32(), function.as_u32(),
                choice.no.as_u32(), choice.yes.as_u32(), choice.condition.as_u32(),
            ),
        )
        .unwrap();
}

#[test]
fn early_select_correlated_integer_choices_eliminate_both_selections() {
    for algebra in [false, true] {
        for ty in ["i32", "u32"] {
            let source = format!("entry choose(c:bool,a:{ty},b:{ty}) {ty}=\nlet x=if c then a else b\nlet y=if c then b else a\nin x+y");
            let program = simplified(&source, algebra);
            let id = result(&program.ir);
            let ExprKind::PureApp { function, args } = &program.ir.expressions[id].kind else {
                panic!("sum")
            };
            assert_eq!(
                program.ir.expressions[*function].kind,
                ExprKind::BinOp("+".into())
            );
            assert!(args
                .iter()
                .all(|&arg| matches!(program.ir.expressions[arg].kind, ExprKind::Parameter(_))));
            let (wgsl, spirv) = shaders(program);
            assert!(!wgsl.contains("select("), "{wgsl}");
            assert!(!spirv
                .all_inst_iter()
                .any(|i| matches!(i.class.opcode, spirv::Op::Select | spirv::Op::SelectionMerge)));
        }
    }
}

#[test]
fn early_select_impossible_comparisons_remove_guarded_loops() {
    for algebra in [false, true] {
        for comparison in ["n == 0", "0 == n", "n < 0", "n >= 9"] {
            let source = format!("entry choose(c:bool,x:f32) f32=\nlet n=if c then 4 else 8\nin if {comparison} then loop v=x for i<4 do f32.sin(v) else x");
            let program = simplified(&source, algebra);
            assert!(program.ir.operations.values().any(|op| matches!(&op.kind, OperationKind::If { condition, .. } if program.ir.expressions[*condition].kind == ExprKind::Bool(false))));
            let (wgsl, spirv) = shaders(program);
            assert!(!wgsl.contains("sin("), "{wgsl}");
            assert!(!spirv
                .all_inst_iter()
                .any(|i| matches!(i.class.opcode, spirv::Op::SelectionMerge | spirv::Op::LoopMerge)));
        }
    }
}

#[test]
fn early_select_boolean_and_repeated_predicate_identities() {
    for (expression, expected) in [
        ("if c then true else false", "c"),
        ("if c then false else true", "not"),
    ] {
        let program = simplified(&format!("entry choose(c:bool) bool={expression}"), false);
        let kind = &program.ir.expressions[result(&program.ir)].kind;
        assert!(match expected {
            "c" => matches!(kind, ExprKind::Parameter(_)),
            _ =>
                matches!(kind, ExprKind::PureApp { function, .. } if program.ir.expressions[*function].kind == ExprKind::UnOp("!".into())),
        });
    }
    for expression in [
        "if c then (if c then x else y) else y",
        "if c then x else (if c then x else y)",
    ] {
        let program = simplified(
            &format!("entry choose(c:bool,x:i32,y:i32) i32={expression}"),
            false,
        );
        let (choice, mode) = program.ir.conditional_value(result(&program.ir)).unwrap();
        assert_eq!(mode, Evaluation::Eager);
        assert!(program.ir.conditional_value(choice.yes).is_none());
        assert!(program.ir.conditional_value(choice.no).is_none());
        shaders(program);
    }
}

#[test]
fn early_select_normalizes_negation_and_pushes_vector_projection() {
    let program = simplified("entry choose(c:bool,a:i32,b:i32) i32=\nlet x=if c then a else b\nlet y=if !c then a else b\nin x+y", false);
    let (wgsl, _) = shaders(program);
    assert!(!wgsl.contains("select("), "{wgsl}");
    let program = simplified(
        "entry choose(c:bool,x:f32,y:f32,z:f32) f32=\nlet v=if c then @[x,z] else @[y,z]\nin v.x",
        false,
    );
    let (choice, _) = program.ir.conditional_value(result(&program.ir)).unwrap();
    assert!(matches!(
        program.ir.expressions[choice.yes].kind,
        ExprKind::Parameter(_)
    ));
    assert!(matches!(
        program.ir.expressions[choice.no].kind,
        ExprKind::Parameter(_)
    ));
    let (wgsl, _) = shaders(program);
    assert!(!wgsl.contains("vec2<f32>"), "{wgsl}");
}

#[test]
fn early_select_does_not_erase_eager_partial_operands_or_condition_evaluation() {
    for source in [
        "entry choose(c:bool,x:i32,y:i32) i32=if c then x/y else y/x",
        "entry choose(c:bool,x:i32,y:i32) i32=if x/y>0 then x else y",
    ] {
        let mut program = input(source);
        make_eager(&mut program.ir);
        let id = result(&program.ir);
        let (choice, _) = program.ir.conditional_value(id).unwrap();
        // Force a tempting rewrite without letting source simplification erase it first.
        let ExprKind::PureApp { args, .. } = &mut program.ir.expressions[id].kind else {
            panic!("select")
        };
        args[0] = choice.yes;
        let program = egglog::simplify(egglog::insert_expressions(program).unwrap(), true).unwrap();
        assert!(
            matches!(
                program.ir.conditional_value(result(&program.ir)),
                Some((_, Evaluation::Eager))
            ),
            "{source}"
        );
    }
}

#[test]
fn early_select_respects_work_limits_and_opaque_execution_boundaries() {
    let program = egglog::insert_expressions(input(
        "entry choose(c:bool,x:u32,y:u32) u32=if c then x<<y else 0u32",
    ))
    .unwrap();
    assert_eq!(
        program.ir.conditional_value(result(&program.ir)).unwrap().1,
        Evaluation::Lazy
    );
    shaders(egglog::simplify(program, true).unwrap());
    for (expression, expected) in [
        ("if c then x+x+x+x+x else y", Some(Evaluation::Eager)),
        ("if c then x+x+x+x+x+x else y", Some(Evaluation::Lazy)),
    ] {
        let program = egglog::insert_expressions(input(&format!(
            "entry choose(c:bool,x:i32,y:i32) i32={expression}"
        )))
        .unwrap();
        assert_eq!(
            program.ir.conditional_value(result(&program.ir)).map(|(_, mode)| mode),
            expected
        );
    }
    let program =
        egglog::insert_expressions(input("entry choose(c:bool,xs:[]i32) i32=if c then xs[0] else 0"))
            .unwrap();
    assert!(matches!(
        program.ir.expressions[result(&program.ir)].kind,
        ExprKind::OperationResult(_)
    ));
    let (wgsl, spirv) = shaders(egglog::simplify(program, true).unwrap());
    assert!(wgsl.contains("if "), "{wgsl}");
    assert!(spirv.all_inst_iter().any(|i| i.class.opcode == spirv::Op::SelectionMerge));
    for expression in ["f32.sqrt(x)", "f32.sin(x)"] {
        let program = egglog::insert_expressions(input(&format!(
            "entry choose(c:bool,x:f32) f32=if c then {expression} else 0.0"
        )))
        .unwrap();
        assert_eq!(
            program.ir.conditional_value(result(&program.ir)).unwrap().1,
            Evaluation::Lazy
        );
    }
}

#[test]
fn early_select_keeps_live_comparisons_and_float_correlations() {
    let program = simplified(
        "entry choose(c:bool,x:f32) f32=\nlet n=if c then 4 else 8\nin if n == 4 then f32.sin(x) else x",
        true,
    );
    let (wgsl, _) = shaders(program);
    assert!(wgsl.contains("sin("), "{wgsl}");
    let program = simplified(
        "entry choose(c:bool,a:f32,b:f32) f32=(if c then a else b)+(if c then b else a)",
        true,
    );
    let (wgsl, spirv) = shaders(program);
    assert!(wgsl.contains("select("), "{wgsl}");
    assert!(spirv.all_inst_iter().any(|i| i.class.opcode == spirv::Op::Select));
}
