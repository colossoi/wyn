use super::super::data::intern_type;
use super::{intern_expr, Expressions, Placed, Program};
use crate::builtins::catalog;
use crate::egglog::data::{
    BuiltinData, ExprId, ExprKind, Ir, OperationKind, PlacementSite, RegionId, SoacBody, TypeId,
};
use crate::egglog::{
    from_tlc, fuse, insert_expressions, schedule, to_ssa, Fused, OptimizeError, Scheduled,
};
use crate::op::{BinaryOperator, OpTag};
use crate::ssa::types::InstKind;
use crate::tlc::infer_input_slice_bounds;
use crate::types::{function, Type, TypeName};
use crate::{compile_thru_tlc, lower_ssa_to_wgsl, CodegenTarget};
use exec::{run, Value};
use std::collections::BTreeSet;

#[path = "schedule_test_exec.rs"]
mod exec;

fn input(source: &str) -> Program<Expressions> {
    let tlc = infer_input_slice_bounds(compile_thru_tlc(source).unwrap());
    insert_expressions(fuse(from_tlc(&tlc).unwrap()).unwrap()).unwrap()
}
fn simplify_and_place(program: Program<Expressions>) -> Result<Program<Placed>, OptimizeError> {
    // These tests construct expressions directly after import.
    let program = insert_expressions(Program {
        ir: program.ir,
        state: Fused,
    })?;
    super::super::simplify_and_place(program)
}

fn compile(source: &str) -> Program<Placed> {
    simplify_and_place(input(source)).unwrap()
}
fn root(data: &Ir) -> RegionId {
    data.definitions[data.entries.values().next().unwrap().definition].body
}
fn parameter(data: &mut Ir, index: usize) -> ExprId {
    let p = data.regions[root(data)].parameters[index];
    intern_expr(data, data.parameters[p].ty, ExprKind::Parameter(p))
}
fn integer(data: &mut Ir, ty: TypeId, n: i64) -> ExprId {
    intern_expr(data, ty, ExprKind::Int(n.to_string()))
}
fn binary(data: &mut Ir, op: &str, a: ExprId, b: ExprId) -> ExprId {
    let t = data.expressions[a].ty;
    let fty = intern_type(
        data,
        function(
            data.types[t].ty.clone(),
            function(
                data.types[data.expressions[b].ty].ty.clone(),
                data.types[t].ty.clone(),
            ),
        ),
    );
    let f = intern_expr(data, fty, ExprKind::BinOp(op.into()));
    intern_expr(
        data,
        t,
        ExprKind::PureApp {
            function: f,
            args: vec![a, b],
        },
    )
}
fn output(data: &mut Ir, e: ExprId) {
    let r = root(data);
    data.regions[r].results = vec![e];
}
fn result(data: &Ir) -> ExprId {
    data.regions[root(data)].results[0]
}
fn wgsl(data: &Program<Scheduled>) {
    let ssa = to_ssa(data, CodegenTarget::Wgsl).unwrap();
    let source = lower_ssa_to_wgsl(ssa).unwrap();
    let module = naga::front::wgsl::parse_str(&source)
        .unwrap_or_else(|e| panic!("{}\n{source}", e.emit_to_string(&source)));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("{e:?}\n{source}"));
}

#[test]
fn eqsat_factors_and_folds_constants_inside_a_new_alternative() {
    let mut c = input("entry main(x:i32) i32 = x");
    let d = &mut c.ir;
    let x = parameter(d, 0);
    let t = d.expressions[x].ty;
    let two = integer(d, t, 2);
    let three = integer(d, t, 3);
    let a = binary(d, "*", x, two);
    let b = binary(d, "*", three, x);
    let sum = binary(d, "+", a, b);
    output(d, sum);
    let c = simplify_and_place(c).unwrap();
    let ExprKind::PureApp { function, args } = &c.ir.expressions[result(&c.ir)].kind else {
        panic!("expected product")
    };
    assert_eq!(c.ir.expressions[*function].kind, ExprKind::BinOp("*".into()));
    assert!(args.iter().any(|&a| c.ir.expressions[a].kind == ExprKind::Int("5".into())));
    let c = schedule(c).unwrap();
    for x in [-91, 0, 123] {
        assert_eq!(run(&c, vec![Value::Int(x)]), vec![Value::Int(x * 5)]);
    }
    wgsl(&c);
}

#[test]
fn algebra_is_optional_while_folding_and_strength_reduction_remain_enabled() {
    for algebra in [false, true] {
        let c = super::simplify(input("entry main(x:i32) i32=x*2+x*3"), algebra).unwrap();
        let ExprKind::PureApp { function, .. } = c.ir.expressions[result(&c.ir)].kind else {
            panic!("arithmetic result");
        };
        assert_eq!(
            c.ir.expressions[function].kind,
            ExprKind::BinOp(if algebra { "*" } else { "+" }.into())
        );
        let c = super::simplify(input("entry main(x:i32) i32=x ** (1+2)"), algebra).unwrap();
        let c = schedule(super::place(c).unwrap()).unwrap();
        assert_eq!(run(&c, vec![Value::Int(-3)]), vec![Value::Int(-27)]);
    }
}

#[test]
fn constant_folding_wraps_at_the_declared_width_and_keeps_partial_operations() {
    for (source, a, b, expected) in [
        ("entry main(x:i32) i32=x", i32::MAX as i64, 1, i32::MIN as i64),
        ("entry main(x:u32) u32=x", u32::MAX as i64, 1, 0),
        ("entry main(x:i16) i16=x", i16::MAX as i64, 1, i16::MIN as i64),
    ] {
        let mut c = input(source);
        let t = c.ir.expressions[result(&c.ir)].ty;
        let a = integer(&mut c.ir, t, a);
        let b = integer(&mut c.ir, t, b);
        let e = binary(&mut c.ir, "+", a, b);
        output(&mut c.ir, e);
        let c = simplify_and_place(c).unwrap();
        assert_eq!(
            c.ir.expressions[result(&c.ir)].kind,
            ExprKind::Int(expected.to_string())
        );
    }
    for (op, rhs) in [("/", 0), ("%", 0), ("<<", 32)] {
        let mut c = input("entry main(x:i32) i32=x");
        let t = c.ir.expressions[result(&c.ir)].ty;
        let a = integer(&mut c.ir, t, 7);
        let b = integer(&mut c.ir, t, rhs);
        let e = binary(&mut c.ir, op, a, b);
        output(&mut c.ir, e);
        let c = simplify_and_place(c).unwrap();
        assert!(matches!(
            c.ir.expressions[result(&c.ir)].kind,
            ExprKind::PureApp { .. }
        ));
    }
}

#[test]
fn float_folding_does_not_turn_invalid_ring_laws_into_equalities() {
    let mut c = input("entry main(x:f32) f32=x");
    let x = parameter(&mut c.ir, 0);
    let t = c.ir.expressions[x].ty;
    let zero = intern_expr(&mut c.ir, t, ExprKind::FloatBits(0));
    let e = binary(&mut c.ir, "*", x, zero);
    output(&mut c.ir, e);
    let c = simplify_and_place(c).unwrap();
    assert!(matches!(
        c.ir.expressions[result(&c.ir)].kind,
        ExprKind::PureApp { .. }
    ));
    for (a, b) in [(f32::INFINITY, 0.0f32), (-0.0, 0.0), (1.0, 3.0)] {
        let mut c = input("entry main(x:f32) f32=x");
        let t = c.ir.expressions[result(&c.ir)].ty;
        let av = intern_expr(&mut c.ir, t, ExprKind::FloatBits(a.to_bits()));
        let bv = intern_expr(&mut c.ir, t, ExprKind::FloatBits(b.to_bits()));
        let e = binary(&mut c.ir, "+", av, bv);
        output(&mut c.ir, e);
        let c = simplify_and_place(c).unwrap();
        assert_eq!(
            c.ir.expressions[result(&c.ir)].kind,
            ExprKind::FloatBits((a + b).to_bits())
        );
    }
}

#[test]
fn constant_power_chains_reach_wgsl_for_integer_and_float_scalars() {
    for ty in ["i32", "u32", "f32"] {
        for exponent in 2..=8 {
            let suffix = if ty == "u32" { "u32" } else { "" };
            let c = compile(&format!("entry main(x:{ty}) {ty} = x ** {exponent}{suffix}"));
            let c = schedule(c).unwrap();
            if ty != "f32" {
                for x in [0_i64, 2, 3] {
                    assert_eq!(run(&c, vec![Value::Int(x)]), vec![Value::Int(x.pow(exponent))]);
                }
            }
            let ssa = to_ssa(&c, CodegenTarget::Wgsl).unwrap();
            let instructions = &ssa.entry_points[0].body.inner.insts;
            assert!(!instructions.values().any(|i| matches!(
                i.data,
                InstKind::Op {
                    tag: OpTag::BinOp(BinaryOperator::Power),
                    ..
                }
            )));
            let multiplies = instructions
                .values()
                .filter(|i| {
                    matches!(
                        i.data,
                        InstKind::Op {
                            tag: OpTag::BinOp(BinaryOperator::Multiply),
                            ..
                        }
                    )
                })
                .count();
            assert!(
                (1..exponent as usize).contains(&multiplies),
                "{ty} ** {exponent}: {multiplies}"
            );
            wgsl(&c);
        }
    }
}

#[test]
fn constant_power_chains_fold_new_products_with_typed_arithmetic() {
    for (ty, base, exponent, expected) in [
        ("i32", "-3", "7", ExprKind::Int("-2187".into())),
        ("i16", "32767i16", "2i16", ExprKind::Int("1".into())),
        ("u32", "4294967295u32", "2u32", ExprKind::Int("1".into())),
        ("f32", "-2.0", "3.0", ExprKind::FloatBits((-8.0_f32).to_bits())),
        ("f32", "-0.0", "3", ExprKind::FloatBits((-0.0_f32).to_bits())),
    ] {
        let c = compile(&format!("entry main() {ty} = ({base}) ** {exponent}"));
        assert_eq!(c.ir.expressions[result(&c.ir)].kind, expected);
    }
    let c = compile("entry main(x:i32) i32 = x ** (1 + 2)");
    let c = schedule(c).unwrap();
    assert_eq!(run(&c, vec![Value::Int(-3)]), vec![Value::Int(-27)]);
}

#[test]
fn constant_power_chains_do_not_duplicate_an_expensive_base() {
    let c = compile("entry main(x:f32) f32 = (x+1.5+2.5+3.5+4.5+5.5+6.5+7.5+8.5+9.5+10.5) ** 5");
    let c = schedule(c).unwrap();
    let ssa = to_ssa(&c, CodegenTarget::Wgsl).unwrap();
    let count = |op| {
        ssa.entry_points[0]
            .body
            .inner
            .insts
            .values()
            .filter(|i| {
                matches!(
                    i.data, InstKind::Op { tag: OpTag::BinOp(actual), .. } if actual == op
                )
            })
            .count()
    };
    assert_eq!(count(BinaryOperator::Power), 0);
    assert_eq!(count(BinaryOperator::Add), 10);
    assert_eq!(count(BinaryOperator::Multiply), 4);
    wgsl(&c);
}

#[test]
fn constant_power_chains_leave_other_exponents_and_vectors_alone() {
    for ty in ["i32", "u32", "f32"] {
        for exponent in ["0", "1", "9", "17", "y"] {
            let suffix = if ty == "u32" && exponent != "y" { "u32" } else { "" };
            let c = compile(&format!(
                "entry main(x:{ty}, y:{ty}) {ty} = x ** {exponent}{suffix}"
            ));
            let ExprKind::PureApp { function, .. } = c.ir.expressions[result(&c.ir)].kind else {
                panic!("expected residual power");
            };
            assert_eq!(c.ir.expressions[function].kind, ExprKind::BinOp("**".into()));
        }
    }
    for exponent in [-1.0_f32, 2.5, f32::INFINITY, f32::NAN] {
        let mut c = input("entry main(x:f32) f32=x");
        let x = parameter(&mut c.ir, 0);
        let ty = c.ir.expressions[x].ty;
        let n = intern_expr(&mut c.ir, ty, ExprKind::FloatBits(exponent.to_bits()));
        let power = binary(&mut c.ir, "**", x, n);
        output(&mut c.ir, power);
        let c = simplify_and_place(c).unwrap();
        assert_eq!(result(&c.ir), power);
    }
    let mut c = input("entry main(x:vec3f32) vec3f32=x");
    let x = parameter(&mut c.ir, 0);
    let scalar = intern_type(&mut c.ir, Type::Constructed(TypeName::Float(32), vec![]));
    let n = intern_expr(&mut c.ir, scalar, ExprKind::FloatBits(2.0_f32.to_bits()));
    let power = binary(&mut c.ir, "**", x, n);
    output(&mut c.ir, power);
    let c = simplify_and_place(c).unwrap();
    assert_eq!(result(&c.ir), power);
}

#[test]
fn explicit_loop_hoists_invariants_but_keeps_iteration_and_accumulator_dependencies() {
    let c = compile("entry main(n:i32, bias:i32) i32 = loop acc=0 for i<n do acc + (bias * bias) + i");
    assert!(c.state.placements.values().any(|p| matches!(p.before, PlacementSite::Operation(op) if matches!(c.ir.operations[op].kind, OperationKind::Loop { .. }))));
    let c = schedule(c).unwrap();
    for n in [0, 1, 7] {
        assert_eq!(
            run(&c, vec![Value::Int(n), Value::Int(3)]),
            vec![Value::Int(n * 9 + n * (n - 1) / 2)]
        );
    }
    wgsl(&c);
}

#[test]
fn nested_loop_invariants_refresh_on_each_outer_iteration() {
    let c = compile(
        "entry main(n:i32) i32 = loop total=0 for i<n do total + (loop acc=0 for j<3 do acc + i*i + j)",
    );
    let c = schedule(c).unwrap();
    for n in [0, 1, 5] {
        assert_eq!(
            run(&c, vec![Value::Int(n)]),
            vec![Value::Int((0..n).map(|i| 3 * i * i + 3).sum())]
        );
    }
    wgsl(&c);
}

#[test]
fn common_if_and_zero_trip_safety() {
    let c = compile("entry main(flag:bool, x:i32) i32 = if flag then x*x+1 else x*x+2");
    assert!(c.state.placements.values().any(|p| matches!(p.before, PlacementSite::Expression(_))));
    let c = schedule(c).unwrap();
    assert_eq!(
        run(&c, vec![Value::Bool(true), Value::Int(7)]),
        vec![Value::Int(50)]
    );
    assert_eq!(
        run(&c, vec![Value::Bool(false), Value::Int(7)]),
        vec![Value::Int(51)]
    );
    let ssa = to_ssa(&c, CodegenTarget::Wgsl).unwrap();
    assert_eq!(
        ssa.entry_points[0]
            .body
            .inner
            .insts
            .values()
            .filter(|i| matches!(
                i.data,
                InstKind::Op {
                    tag: OpTag::BinOp(BinaryOperator::Multiply),
                    ..
                }
            ))
            .count(),
        1,
        "common work must be emitted once before the selection"
    );
    wgsl(&c);
    let c = compile("entry main(n:i32, d:i32) i32 = loop acc=0 for i<n do acc + 12/d");
    assert!(c.state.placements.is_empty(), "division must not be speculated");
    let c = schedule(c).unwrap();
    assert_eq!(run(&c, vec![Value::Int(0), Value::Int(0)]), vec![Value::Int(0)]);
}

#[test]
fn while_header_reuses_syntax_without_reusing_the_previous_iterations_value() {
    let c = compile("entry main(n:i32) i32 = let (_,value)=loop (i,total)=(0,0) while i<n do (i+1,total+(loop v=0 for j<3 do v+i*i+j)) in value");
    assert!(!c.state.placements.is_empty());
    let c = schedule(c).unwrap();
    for n in [0, 1, 3] {
        assert_eq!(
            run(&c, vec![Value::Int(n)]),
            vec![Value::Int((0..n).map(|i| 3 * i * i + 3).sum())]
        );
    }
    wgsl(&c);
}

#[test]
fn soac_capture_computations_move_out_and_refresh_between_iterations() {
    let c = compile("entry main(xs:[4]i32, n:i32) [4]i32 = loop acc=xs for i<n do map(|x:i32|x+i*i,acc)");
    assert!(c.state.placements.values().any(|p| matches!(p.before, PlacementSite::Operation(op) if matches!(c.ir.operations[op].kind, OperationKind::Screma { .. }))));
    let c = schedule(c).unwrap();
    for n in [0, 1, 5] {
        let add: i64 = (0..n).map(|i| i * i).sum();
        assert_eq!(
            run(&c, vec![Value::array(1..5), Value::Int(n)])[0].ints(),
            (1..5).map(|x| x + add).collect::<Vec<_>>()
        );
    }
    wgsl(&c);
}

#[test]
fn nested_soac_capture_hoisting_preserves_element_dependence() {
    let c = compile("entry main(xs:[]i32, bias:i32) []i32 = map(|x:i32|reduce(|a:i32,b:i32|a+b,0,map(|y:i32|y+x*x+bias*bias,iota(3))),xs)");
    let c = schedule(c).unwrap();
    assert_eq!(
        run(&c, vec![Value::array(0..5), Value::Int(2)])[0].ints(),
        (0..5).map(|x| 3 + 3 * x * x + 12).collect::<Vec<_>>()
    );
    wgsl(&c);
}

#[test]
fn capture_bounds_stop_at_the_loop_binding_that_varies() {
    let c = compile("entry main(xs:[4]i32, n:i32, bias:i32) [4]i32 = loop acc=xs for i<n do map(|x:i32|x+i*i+bias*bias,acc)");
    let d = &c;
    let bias = d.regions[root(d)].parameters[2];
    let outer =
        d.operations.iter().find(|(_, op)| matches!(op.kind, OperationKind::Loop { .. })).unwrap().0;
    let outside: Vec<_> =
        d.state.placements.values().filter(|p| p.before == PlacementSite::Operation(*outer)).collect();
    assert!(
        !outside.is_empty(),
        "capture-only work should leave both the map and loop"
    );
    for p in outside {
        let ExprKind::PureApp { args, .. } = &d.expressions[p.expression].kind else {
            panic!("product")
        };
        assert!(
            args.iter().all(|&e| d.expressions[e].kind == ExprKind::Parameter(bias)),
            "iteration-dependent work cannot cross the outer loop"
        );
    }
    assert!(
        d.state.placements.values().any(|p| matches!(p.before, PlacementSite::Operation(op)
        if matches!(d.operations[op].kind, OperationKind::Screma { .. })))
    );
    let c = schedule(c).unwrap();
    for n in [0, 1, 5] {
        let add: i64 = (0..n).map(|i| i * i + 9).sum();
        assert_eq!(
            run(&c, vec![Value::array(1..5), Value::Int(n), Value::Int(3)])[0].ints(),
            (1..5).map(|x| x + add).collect::<Vec<_>>()
        );
    }
    wgsl(&c);
}

#[test]
fn operation_result_bounds_preserve_order_and_branch_scope() {
    for source in [
        "entry main(xs:[]i32, n:i32) i32 = loop total=0 for i<n do let v=xs[i] in total+(loop acc=0 for j<3 do acc+v*v+j)",
        "entry main(xs:[]i32, n:i32) i32 = if n>0 then let v=xs[0] in (loop acc=0 for j<3 do acc+v*v+j) else 0",
    ] {
        let c = compile(source);
        let read = c.ir.operations.iter().find(|(_, op)| matches!(op.kind, OperationKind::Index { .. })).unwrap().0;
        assert!(!c.state.placements.is_empty(), "reuse the value after the read, before the inner loop");
        for p in c.state.placements.values() {
            let PlacementSite::Operation(op) = p.before else { panic!("loop placement") };
            assert_eq!(c.ir.operations[op].region, c.ir.operations[*read].region,
                "a binding cannot escape its defining loop/branch");
        }
        let c = schedule(c).unwrap();
        assert_eq!(run(&c, vec![Value::array([]), Value::Int(0)]), vec![Value::Int(0)]);
        assert_eq!(run(&c, vec![Value::array([4]), Value::Int(1)]), vec![Value::Int(51)]);
        wgsl(&c);
    }
}

#[test]
fn shared_callback_bounds_translate_captures_for_each_invocation() {
    let mut c = input("entry main(xs:[2]i32, ys:[3]i32, a:i32, b:i32) ([2]i32,[3]i32) = (map(|x:i32|x+a*a,xs),map(|x:i32|x+b*b,ys))");
    let ops: Vec<_> =
        c.ir.operations
            .iter()
            .filter_map(|(&id, op)| matches!(op.kind, OperationKind::Screma { .. }).then_some(id))
            .collect();
    assert_eq!(ops.len(), 2);
    let OperationKind::Screma { form, .. } = &c.ir.operations[ops[0]].kind else {
        panic!("map")
    };
    let SoacBody::Apply { region: shared, .. } = form.pre else {
        panic!("callback")
    };
    // Identical code, but separate actual arguments; sharing syntax must not
    // equate the two invocations' capture values.
    let OperationKind::Screma { form, .. } = &mut c.ir.operations[ops[1]].kind else {
        panic!("map")
    };
    let SoacBody::Apply { region, .. } = &mut form.pre else {
        panic!("callback")
    };
    *region = shared;
    let c = simplify_and_place(c).unwrap();
    assert!(c.state.placements.values().any(|p| p.before == PlacementSite::Operation(ops[0])));
    assert!(c.state.placements.values().any(|p| p.before == PlacementSite::Operation(ops[1])));
    let regions: BTreeSet<_> = ops
        .iter()
        .map(|&op| {
            let OperationKind::Screma { form, .. } = &c.ir.operations[op].kind else {
                panic!("map")
            };
            let SoacBody::Apply { region, .. } = form.pre else {
                panic!("callback")
            };
            region
        })
        .collect();
    assert_eq!(
        regions.len(),
        1,
        "share specialized code, while preserving different actual captures"
    );
    let c = schedule(c).unwrap();
    let values = run(
        &c,
        vec![
            Value::array([1, 2]),
            Value::array([3, 4, 5]),
            Value::Int(2),
            Value::Int(5),
        ],
    );
    let Value::Tuple(arrays) = &values[0] else {
        panic!("two outputs")
    };
    assert_eq!(arrays[0].ints(), vec![5, 6]);
    assert_eq!(arrays[1].ints(), vec![28, 29, 30]);
    wgsl(&c);
}

#[test]
fn structured_if_hoists_shared_work_out_of_both_loop_bodies() {
    let c = compile("entry main(flag:bool, x:i32) i32 = if flag then (loop a=0 for i<2 do a+x*x) else (loop a=0 for i<3 do a+x*x)");
    assert!(c.state.placements.values().any(|p| matches!(p.before, PlacementSite::Operation(op) if matches!(c.ir.operations[op].kind, OperationKind::If { .. }))));
    let c = schedule(c).unwrap();
    for flag in [false, true] {
        assert_eq!(
            run(&c, vec![Value::Bool(flag), Value::Int(7)]),
            vec![Value::Int(if flag { 98 } else { 147 })]
        );
    }
    wgsl(&c);
}

#[test]
fn read_occurrences_and_partial_branch_expressions_are_not_speculated() {
    let c = compile("entry main(xs:[]i32, n:i32) i32 = loop a=0 for i<n do a+xs[0]*xs[0]");
    assert!(c.state.placements.is_empty());
    let c = schedule(c).unwrap();
    assert_eq!(
        run(&c, vec![Value::array([]), Value::Int(0)]),
        vec![Value::Int(0)]
    );
    assert_eq!(
        run(&c, vec![Value::array([4]), Value::Int(3)]),
        vec![Value::Int(48)]
    );
    let c = compile("entry main(flag:bool, d:i32) i32 = if flag then 100/d+1 else 100/d+2");
    assert!(c.state.placements.is_empty());
}

#[test]
fn tuple_captures_can_be_hoisted_without_flattening_element_arguments() {
    let c = compile("entry main(xs:[]i32, pair:(i32,i32)) []i32 = map(|x:i32|x+pair.0*pair.1,xs)");
    assert!(!c.state.placements.is_empty());
    let c = schedule(c).unwrap();
    let actual = run(
        &c,
        vec![
            Value::arrays(vec![Value::Int(1), Value::Int(2)]),
            Value::Tuple(vec![Value::Int(3), Value::Int(5)]),
        ],
    );
    assert_eq!(actual[0].ints(), vec![16, 17]);
    wgsl(&c);
}

#[test]
fn cancellation_removes_the_parameter_from_the_result() {
    let mut c = input("entry main(x:i32) i32=x");
    let x = parameter(&mut c.ir, 0);
    let e = binary(&mut c.ir, "-", x, x);
    output(&mut c.ir, e);
    let c = simplify_and_place(c).unwrap();
    assert_eq!(c.ir.expressions[result(&c.ir)].kind, ExprKind::Int("0".into()));
}

#[test]
fn inverse_bitcasts_preserve_binding_identity_and_nan_payloads() {
    fn builtin(data: &mut Ir, name: &str, arg: ExprId, ty: TypeId) -> ExprId {
        let builtin = catalog().lookup_by_any_name(name).unwrap();
        let b = data.builtins.alloc(BuiltinData {
            builtin: builtin.id,
            overload_idx: 0,
        });
        let ft = intern_type(
            data,
            function(
                data.types[data.expressions[arg].ty].ty.clone(),
                data.types[ty].ty.clone(),
            ),
        );
        let f = intern_expr(data, ft, ExprKind::Builtin(b));
        intern_expr(
            data,
            ty,
            ExprKind::PureApp {
                function: f,
                args: vec![arg],
            },
        )
    }
    let mut c = input("entry main(x:u32) u32=x");
    let x = parameter(&mut c.ir, 0);
    let u = c.ir.expressions[x].ty;
    let i = intern_type(&mut c.ir, Type::Constructed(TypeName::Int(32), vec![]));
    let a = builtin(&mut c.ir, "i32.u32", x, i);
    let b = builtin(&mut c.ir, "u32.i32", a, u);
    output(&mut c.ir, b);
    let c = simplify_and_place(c).unwrap();
    assert_eq!(result(&c.ir), x);

    let mut c = input("entry main(x:f32) f32=x");
    let f = c.ir.expressions[result(&c.ir)].ty;
    let u = intern_type(&mut c.ir, Type::Constructed(TypeName::UInt(32), vec![]));
    let value = intern_expr(&mut c.ir, f, ExprKind::FloatBits(0x7fc01234));
    // Constant folding must preserve the bit pattern of a NaN.
    let app = builtin(&mut c.ir, "f32.to_bits", value, u);
    output(&mut c.ir, app);
    let c = simplify_and_place(c).unwrap();
    let folded = result(&c.ir);
    assert_eq!(
        c.ir.expressions[folded].kind,
        ExprKind::Int(0x7fc01234u32.to_string())
    );
}
#[test]
fn constant_folding_reaches_a_fixed_point_beyond_thirty_two_dependent_operations() {
    let mut c = input("entry main(x:f32) f32=x");
    let t = c.ir.expressions[result(&c.ir)].ty;
    let mut value = intern_expr(&mut c.ir, t, ExprKind::FloatBits(2.0f32.powi(60).to_bits()));
    let two = intern_expr(&mut c.ir, t, ExprKind::FloatBits(2.0f32.to_bits()));
    for _ in 0..48 {
        value = binary(&mut c.ir, "/", value, two);
    }
    output(&mut c.ir, value);
    let c = simplify_and_place(c).unwrap();
    assert_eq!(
        c.ir.expressions[result(&c.ir)].kind,
        ExprKind::FloatBits(4096.0f32.to_bits())
    );
}
