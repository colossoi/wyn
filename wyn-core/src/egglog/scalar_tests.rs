use super::*;
use crate::egglog::{convert_program, insert_expressions, optimize, schedule, to_ssa};
use crate::{compile_thru_tlc, tlc, types};

#[path = "schedule_test_exec.rs"]
mod exec;
use exec::{run, Value};

fn input(source: &str) -> Converted {
    let tlc = tlc::infer_input_slice_bounds(compile_thru_tlc(source).unwrap());
    insert_expressions(optimize(convert_program(&tlc).unwrap()).unwrap()).unwrap()
}
fn compile(source: &str) -> Converted {
    optimize_expressions(input(source)).unwrap()
}
fn root(data: &AssociatedData) -> RegionId {
    data.definitions[data.entries.values().next().unwrap().definition].body
}
fn parameter(data: &mut AssociatedData, index: usize) -> ExprId {
    let p = data.regions[root(data)].parameters[index];
    expr(data, data.parameters[p].ty, ExprKind::Parameter(p))
}
fn integer(data: &mut AssociatedData, ty: TypeId, n: i64) -> ExprId {
    expr(data, ty, ExprKind::Int(n.to_string()))
}
fn binary(data: &mut AssociatedData, op: &str, a: ExprId, b: ExprId) -> ExprId {
    let t = data.expressions[a].ty;
    let fty = super::super::fusion::ty(
        data,
        types::function(
            data.types[t].ty.clone(),
            types::function(
                data.types[data.expressions[b].ty].ty.clone(),
                data.types[t].ty.clone(),
            ),
        ),
    );
    let f = expr(data, fty, ExprKind::BinOp(op.into()));
    expr(
        data,
        t,
        ExprKind::PureApp {
            function: f,
            args: vec![a, b],
        },
    )
}
fn output(data: &mut AssociatedData, e: ExprId) {
    let r = root(data);
    data.regions[r].results = vec![e];
}
fn result(data: &AssociatedData) -> ExprId {
    data.regions[root(data)].results[0]
}
fn wgsl(data: &AssociatedData) {
    let ssa = to_ssa(data, crate::CodegenTarget::Wgsl).unwrap();
    let source = crate::lower_ssa_to_wgsl(ssa).unwrap();
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
    let d = &mut c.data;
    let x = parameter(d, 0);
    let t = d.expressions[x].ty;
    let two = integer(d, t, 2);
    let three = integer(d, t, 3);
    let a = binary(d, "*", x, two);
    let b = binary(d, "*", three, x);
    let sum = binary(d, "+", a, b);
    output(d, sum);
    let c = optimize_expressions(c).unwrap();
    let ExprKind::PureApp { function, args } = &c.data.expressions[result(&c.data)].kind else {
        panic!("expected product")
    };
    assert_eq!(c.data.expressions[*function].kind, ExprKind::BinOp("*".into()));
    assert!(args.iter().any(|&a| c.data.expressions[a].kind == ExprKind::Int("5".into())));
    let c = schedule(c).unwrap();
    for x in [-91, 0, 123] {
        assert_eq!(run(&c.data, vec![Value::Int(x)]), vec![Value::Int(x * 5)]);
    }
    wgsl(&c.data);
}

#[test]
fn constant_folding_wraps_at_the_declared_width_and_keeps_partial_operations() {
    for (source, a, b, expected) in [
        ("entry main(x:i32) i32=x", i32::MAX as i64, 1, i32::MIN as i64),
        ("entry main(x:u32) u32=x", u32::MAX as i64, 1, 0),
        ("entry main(x:i16) i16=x", i16::MAX as i64, 1, i16::MIN as i64),
    ] {
        let mut c = input(source);
        let t = c.data.expressions[result(&c.data)].ty;
        let a = integer(&mut c.data, t, a);
        let b = integer(&mut c.data, t, b);
        let e = binary(&mut c.data, "+", a, b);
        output(&mut c.data, e);
        let c = optimize_expressions(c).unwrap();
        assert_eq!(
            c.data.expressions[result(&c.data)].kind,
            ExprKind::Int(expected.to_string())
        );
    }
    for (op, rhs) in [("/", 0), ("%", 0), ("<<", 32)] {
        let mut c = input("entry main(x:i32) i32=x");
        let t = c.data.expressions[result(&c.data)].ty;
        let a = integer(&mut c.data, t, 7);
        let b = integer(&mut c.data, t, rhs);
        let e = binary(&mut c.data, op, a, b);
        output(&mut c.data, e);
        let c = optimize_expressions(c).unwrap();
        assert!(matches!(
            c.data.expressions[result(&c.data)].kind,
            ExprKind::PureApp { .. }
        ));
    }
}

#[test]
fn float_folding_does_not_turn_invalid_ring_laws_into_equalities() {
    let mut c = input("entry main(x:f32) f32=x");
    let x = parameter(&mut c.data, 0);
    let t = c.data.expressions[x].ty;
    let zero = expr(&mut c.data, t, ExprKind::FloatBits(0));
    let e = binary(&mut c.data, "*", x, zero);
    output(&mut c.data, e);
    let c = optimize_expressions(c).unwrap();
    assert!(matches!(
        c.data.expressions[result(&c.data)].kind,
        ExprKind::PureApp { .. }
    ));
    for (a, b) in [(f32::INFINITY, 0.0f32), (-0.0, 0.0), (1.0, 3.0)] {
        let mut c = input("entry main(x:f32) f32=x");
        let t = c.data.expressions[result(&c.data)].ty;
        let av = expr(&mut c.data, t, ExprKind::FloatBits(a.to_bits()));
        let bv = expr(&mut c.data, t, ExprKind::FloatBits(b.to_bits()));
        let e = binary(&mut c.data, "+", av, bv);
        output(&mut c.data, e);
        let c = optimize_expressions(c).unwrap();
        assert_eq!(
            c.data.expressions[result(&c.data)].kind,
            ExprKind::FloatBits((a + b).to_bits())
        );
    }
}

#[test]
fn explicit_loop_hoists_invariants_but_keeps_iteration_and_accumulator_dependencies() {
    let c = compile("entry main(n:i32, bias:i32) i32 = loop acc=0 for i<n do acc + (bias * bias) + i");
    assert!(c.data.placements.values().any(|p| matches!(p.before, PlacementSite::Operation(op) if matches!(c.data.operations[op].kind, OperationKind::Loop { .. }))));
    let c = schedule(c).unwrap();
    for n in [0, 1, 7] {
        assert_eq!(
            run(&c.data, vec![Value::Int(n), Value::Int(3)]),
            vec![Value::Int(n * 9 + n * (n - 1) / 2)]
        );
    }
    wgsl(&c.data);
}

#[test]
fn nested_loop_invariants_refresh_on_each_outer_iteration() {
    let c = compile(
        "entry main(n:i32) i32 = loop total=0 for i<n do total + (loop acc=0 for j<3 do acc + i*i + j)",
    );
    let c = schedule(c).unwrap();
    for n in [0, 1, 5] {
        assert_eq!(
            run(&c.data, vec![Value::Int(n)]),
            vec![Value::Int((0..n).map(|i| 3 * i * i + 3).sum())]
        );
    }
    wgsl(&c.data);
}

#[test]
fn common_if_and_zero_trip_safety() {
    let c = compile("entry main(flag:bool, x:i32) i32 = if flag then x*x+1 else x*x+2");
    assert!(c.data.placements.values().any(|p| matches!(p.before, PlacementSite::Expression(_))));
    let c = schedule(c).unwrap();
    assert_eq!(
        run(&c.data, vec![Value::Bool(true), Value::Int(7)]),
        vec![Value::Int(50)]
    );
    assert_eq!(
        run(&c.data, vec![Value::Bool(false), Value::Int(7)]),
        vec![Value::Int(51)]
    );
    let ssa = to_ssa(&c.data, crate::CodegenTarget::Wgsl).unwrap();
    assert_eq!(
        ssa.entry_points[0]
            .body
            .inner
            .insts
            .values()
            .filter(|i| matches!(
                i.data,
                crate::ssa::types::InstKind::Op {
                    tag: crate::op::OpTag::BinOp(crate::op::BinaryOperator::Multiply),
                    ..
                }
            ))
            .count(),
        1,
        "common work must be emitted once before the selection"
    );
    wgsl(&c.data);
    let c = compile("entry main(n:i32, d:i32) i32 = loop acc=0 for i<n do acc + 12/d");
    assert!(c.data.placements.is_empty(), "division must not be speculated");
    let c = schedule(c).unwrap();
    assert_eq!(
        run(&c.data, vec![Value::Int(0), Value::Int(0)]),
        vec![Value::Int(0)]
    );
}

#[test]
fn soac_capture_computations_move_out_and_refresh_between_launches() {
    let c = compile("entry main(xs:[4]i32, n:i32) [4]i32 = loop acc=xs for i<n do map(|x:i32|x+i*i,acc)");
    assert!(c.data.placements.values().any(|p| matches!(p.before, PlacementSite::Operation(op) if matches!(c.data.operations[op].kind, OperationKind::Screma { .. }))));
    let c = schedule(c).unwrap();
    for n in [0, 1, 5] {
        let add: i64 = (0..n).map(|i| i * i).sum();
        assert_eq!(
            run(&c.data, vec![Value::array(1..5), Value::Int(n)])[0].ints(),
            (1..5).map(|x| x + add).collect::<Vec<_>>()
        );
    }
    wgsl(&c.data);
}

#[test]
fn nested_soac_capture_hoisting_preserves_element_dependence() {
    let c = compile("entry main(xs:[]i32, bias:i32) []i32 = map(|x:i32|reduce(|a:i32,b:i32|a+b,0,map(|y:i32|y+x*x+bias*bias,iota(3))),xs)");
    let c = schedule(c).unwrap();
    assert_eq!(
        run(&c.data, vec![Value::array(0..5), Value::Int(2)])[0].ints(),
        (0..5).map(|x| 3 + 3 * x * x + 12).collect::<Vec<_>>()
    );
    wgsl(&c.data);
}

#[test]
fn structured_if_hoists_shared_work_out_of_both_loop_bodies() {
    let c = compile("entry main(flag:bool, x:i32) i32 = if flag then (loop a=0 for i<2 do a+x*x) else (loop a=0 for i<3 do a+x*x)");
    assert!(c.data.placements.values().any(|p| matches!(p.before, PlacementSite::Operation(op) if matches!(c.data.operations[op].kind, OperationKind::If { .. }))));
    let c = schedule(c).unwrap();
    for flag in [false, true] {
        assert_eq!(
            run(&c.data, vec![Value::Bool(flag), Value::Int(7)]),
            vec![Value::Int(if flag { 98 } else { 147 })]
        );
    }
    wgsl(&c.data);
}

#[test]
fn read_occurrences_and_partial_branch_expressions_are_not_speculated() {
    let c = compile("entry main(xs:[]i32, n:i32) i32 = loop a=0 for i<n do a+xs[0]*xs[0]");
    assert!(c.data.placements.is_empty());
    let c = schedule(c).unwrap();
    assert_eq!(
        run(&c.data, vec![Value::array([]), Value::Int(0)]),
        vec![Value::Int(0)]
    );
    assert_eq!(
        run(&c.data, vec![Value::array([4]), Value::Int(3)]),
        vec![Value::Int(48)]
    );
    let c = compile("entry main(flag:bool, d:i32) i32 = if flag then 100/d+1 else 100/d+2");
    assert!(c.data.placements.is_empty());
}

#[test]
fn tuple_captures_can_be_hoisted_without_flattening_element_arguments() {
    let c = compile("entry main(xs:[]i32, pair:(i32,i32)) []i32 = map(|x:i32|x+pair.0*pair.1,xs)");
    assert!(!c.data.placements.is_empty());
    let c = schedule(c).unwrap();
    let actual = run(
        &c.data,
        vec![
            Value::arrays(vec![Value::Int(1), Value::Int(2)]),
            Value::Tuple(vec![Value::Int(3), Value::Int(5)]),
        ],
    );
    assert_eq!(actual[0].ints(), vec![16, 17]);
    wgsl(&c.data);
}

#[test]
fn cancellation_drops_dependencies_in_the_final_fact_base() {
    let mut c = input("entry main(x:i32) i32=x");
    let x = parameter(&mut c.data, 0);
    let e = binary(&mut c.data, "-", x, x);
    output(&mut c.data, e);
    let c = optimize_expressions(c).unwrap();
    assert_eq!(
        c.data.expressions[result(&c.data)].kind,
        ExprKind::Int("0".into())
    );
    let mut graph = EGraph::default();
    graph.run_program(c.program).unwrap();
    graph.parse_and_run_program(None, "(check (RegionResult r 0 e) (= (Dependencies e) (set-empty))) (fail (check (RegionResult r 0 e) (ExprParameter e p)))").unwrap();
}

#[test]
fn inverse_bitcasts_preserve_binding_identity_and_nan_payloads() {
    fn builtin(data: &mut AssociatedData, name: &str, arg: ExprId, ty: TypeId) -> ExprId {
        let builtin = crate::builtins::catalog().lookup_by_any_name(name).unwrap();
        let b = data.builtins.alloc(BuiltinData {
            builtin: builtin.id,
            overload_idx: 0,
        });
        let ft = super::super::fusion::ty(
            data,
            types::function(
                data.types[data.expressions[arg].ty].ty.clone(),
                data.types[ty].ty.clone(),
            ),
        );
        let f = expr(data, ft, ExprKind::Builtin(b));
        expr(
            data,
            ty,
            ExprKind::PureApp {
                function: f,
                args: vec![arg],
            },
        )
    }
    let mut c = input("entry main(x:u32) u32=x");
    let x = parameter(&mut c.data, 0);
    let u = c.data.expressions[x].ty;
    let i = super::super::fusion::ty(
        &mut c.data,
        types::Type::Constructed(types::TypeName::Int(32), vec![]),
    );
    let a = builtin(&mut c.data, "i32.u32", x, i);
    let b = builtin(&mut c.data, "u32.i32", a, u);
    output(&mut c.data, b);
    let c = optimize_expressions(c).unwrap();
    assert_eq!(result(&c.data), x);

    let mut c = input("entry main(x:f32) f32=x");
    let f = c.data.expressions[result(&c.data)].ty;
    let u = super::super::fusion::ty(
        &mut c.data,
        types::Type::Constructed(types::TypeName::UInt(32), vec![]),
    );
    let value = expr(&mut c.data, f, ExprKind::FloatBits(0x7fc01234));
    // Exercise the constant evaluator directly: a bitcast must not canonicalize NaNs.
    let app = builtin(&mut c.data, "f32.to_bits", value, u);
    let folded = fold::evaluate(&mut c.data, app).unwrap();
    assert_eq!(
        c.data.expressions[folded].kind,
        ExprKind::Int(0x7fc01234u32.to_string())
    );
}
