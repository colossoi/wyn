use super::*;
use crate::ssa::builder::FuncBuilder;
use crate::ssa::eliminate_dead_pure_instructions;
use crate::ssa::types::{ConstantValue, Terminator};
use crate::types::{i32, sized_array};
use crate::{BindingRef, FunctionId};

fn binary(op: BinaryOperator, a: ValueRef, b: ValueRef) -> InstKind {
    InstKind::Op {
        tag: OpTag::BinOp(op),
        operands: vec![a, b],
    }
}

#[test]
fn representation_lowering_makes_view_updates_explicit_without_selecting_local_reuse() {
    let composite = sized_array(4, i32());
    let view = crate::types::view_array_with_size(
        &i32(),
        composite.array_size().unwrap().clone(),
        composite.array_buffer().unwrap().clone(),
    );
    for (ty, expected) in [
        (composite, crate::builtins::catalog().known().array_with),
        (view, crate::builtins::catalog().known().array_with_in_place),
    ] {
        let mut builder = FuncBuilder::new(vec![(ty.clone(), "xs".into())], ty.clone());
        let source = builder.get_param(0);
        let result = builder
            .push_inst(
                InstKind::Op {
                    tag: OpTag::Intrinsic {
                        id: crate::builtins::catalog().known().array_with,
                        overload_idx: 0,
                    },
                    operands: vec![
                        source.into(),
                        ValueRef::Const(ConstantValue::I32(0)),
                        ValueRef::Const(ConstantValue::I32(7)),
                    ],
                },
                ty,
            )
            .unwrap();
        builder.terminate(Terminator::Return(Some(result.into()))).unwrap();
        let mut body = builder.finish().unwrap();
        prepare_values(&mut body);
        let ValueDef::Inst { inst } = body.inner.values[result].def else {
            panic!("instruction result")
        };
        assert!(matches!(body.inner.insts[inst].data,
            InstKind::Op { tag: OpTag::Intrinsic { id, .. }, .. } if id == expected));
    }
}

#[test]
fn division_and_remainder_used_by_a_condition_are_reused_in_both_arms() {
    for source in [
        r#"entry repro(indices: []i32, width: i32) []i32 =
      map(|i|
        let q = i / width
        let r = i % width in
        if q < r then q - r else q + r,
        indices)"#,
        r#"entry repro(indices: []i32, width: i32) []i32 =
      map(|i|
        loop total = 0 for j < 3 do
          let q = (i + j) / width
          let r = (i + j) % width in
          total + (if q < r then q - r else q + r),
        indices)"#,
    ] {
        assert_division_sites(source, 1);
    }
}

#[test]
fn division_and_remainder_are_reused_from_enclosing_loop_scopes() {
    for source in [
        r#"entry repro(indices: []i32, width: i32) []i32 =
  map(|i|
    let pixel = @[i % width, i / width] in
    loop total = pixel.x + pixel.y for k < 2 do total + pixel.x + pixel.y,
    indices)"#,
        // The producer is two lexical loop scopes outside the repeated use.
        r#"entry repro(indices: []i32, width: i32) []i32 =
  map(|i|
    let pixel = @[i % width, i / width] in
    loop total = pixel.x + pixel.y for j < 2 do
      loop inner = total for k < 2 do inner + pixel.x + pixel.y,
    indices)"#,
        // An outer iteration's result is available throughout its inner loop.
        r#"entry repro(indices: []i32, width: i32) []i32 =
  map(|i|
    loop total = 0 for j < 2 do
      let pixel = @[(i + j) % width, (i + j) / width] in
      loop inner = total + pixel.x + pixel.y for k < 2 do
        inner + pixel.x + pixel.y,
    indices)"#,
    ] {
        assert_division_sites(source, 1);
    }
}

#[test]
fn loop_header_division_and_remainder_are_reused_after_the_loop() {
    // The header dominates the merge. WGSL declares these results outside
    // the loop while evaluating them at their original header positions.
    assert_division_sites(
        r#"entry repro(indices: []i32, width: i32) []i32 =
  map(|i|
    let total = loop total = 0 while total < i / width + i % width do total + 1 in
    total + i / width + i % width,
    indices)"#,
        1,
    );
}

#[test]
fn division_and_remainder_in_a_possibly_empty_loop_stay_inside_it() {
    let placed = assert_division_sites(
        r#"entry repro(indices: []i32, width: i32) []i32 =
  map(|i| loop total = 0 for k < i do total + i / width + i % width, indices)"#,
        1,
    );
    for body in placed.functions.iter().map(|f| &f.body).chain(placed.entry_points.iter().map(|e| &e.body))
    {
        let scopes = LoopScopes::analyze(&body.inner);
        for node in body.inner.insts.values() {
            if matches!(
                node.data,
                InstKind::Op {
                    tag: OpTag::BinOp(BinaryOperator::Divide | BinaryOperator::Remainder),
                    ..
                }
            ) {
                assert!(scopes.scope(node.placement.block().unwrap()).is_some());
            }
        }
    }
}

fn assert_division_sites(source: &str, expected: usize) -> crate::ssa::stage::Placed {
    let ssa = crate::compile_thru_ssa(source).unwrap();
    let placed = live_instructions(ssa.clone());
    for operator in [BinaryOperator::Divide, BinaryOperator::Remainder] {
        let sites = placed
            .functions
            .iter()
            .map(|f| &f.body)
            .chain(placed.entry_points.iter().map(|e| &e.body))
            .flat_map(|body| body.inner.insts.values())
            .filter(
                |node| matches!(node.data, InstKind::Op { tag: OpTag::BinOp(op), .. } if op == operator),
            )
            .count();
        assert_eq!(
            sites,
            expected,
            "unexpected {operator:?} count before backend lowering\n{source}\n{}",
            crate::ssa::print::format_program(&placed)
        );
    }
    let wgsl = crate::lower_ssa_to_wgsl(ssa.clone()).unwrap();
    let module = naga::front::wgsl::parse_str(&wgsl).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    let spirv = crate::lower_ssa_to_spirv(ssa).unwrap().spirv;
    let module = wspirv::dr::load_words(&spirv).unwrap();
    for opcode in [wspirv::spirv::Op::SDiv, wspirv::spirv::Op::SRem] {
        assert_eq!(
            module.all_inst_iter().filter(|inst| inst.class.opcode == opcode).count(),
            expected,
            "unexpected {opcode:?} count"
        );
    }
    let bytes = spirv.iter().flat_map(|word| word.to_le_bytes()).collect::<Vec<_>>();
    let module = naga::front::spv::parse_u8_slice(&bytes, &Default::default()).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    placed
}

#[test]
fn dynamic_array_materialization_is_shared_outside_the_loop() {
    let mut body = FuncBuilder::new(vec![(sized_array(128, i32()), "xs".into())], i32()).finish_unchecked();
    let entry = body.inner.entry;
    let array = body.inner.params[0];
    let loop_body = body.inner.create_block();
    let i = body.inner.add_block_param(loop_body, i32());
    body.inner.blocks[entry].term = Terminator::Branch {
        target: loop_body,
        args: vec![ValueRef::Const(ConstantValue::I32(0))],
    };
    let mut results = vec![];
    for _ in 0..2 {
        results.push(body.inner.append_inst(
            loop_body,
            InstKind::Op {
                tag: OpTag::Index,
                operands: vec![array.into(), i.into()],
            },
            i32(),
        ));
    }
    let sum = body.inner.append_inst(
        loop_body,
        binary(BinaryOperator::Add, results[0].into(), results[1].into()),
        i32(),
    );
    body.inner.blocks[loop_body].term = Terminator::Branch {
        target: loop_body,
        args: vec![sum.into()],
    };
    prepare_values(&mut body);
    crate::ssa::ir::schedule_floating(&mut body.inner).unwrap();
    let materialized: Vec<_> = body
        .inner
        .insts
        .values()
        .filter(|n| {
            matches!(
                n.data,
                InstKind::Op {
                    tag: OpTag::Materialize,
                    ..
                }
            )
        })
        .collect();
    assert_eq!(materialized.len(), 1);
    assert_eq!(materialized[0].placement.block(), Some(entry));
    assert_eq!(
        body.inner
            .insts
            .values()
            .filter(|n| matches!(
                n.data,
                InstKind::Op {
                    tag: OpTag::DynamicExtract,
                    ..
                }
            ))
            .count(),
        2
    );
}

fn intrinsic(name: &str) -> OpTag<BindingRef, FunctionId> {
    OpTag::Intrinsic {
        id: crate::builtins::catalog().lookup_by_any_name(name).unwrap().id,
        overload_idx: 0,
    }
}

fn assert_partial_intrinsic_sites(source: &str, name: &str, expected: usize) -> crate::ssa::stage::Placed {
    let ssa = crate::compile_thru_ssa(source).unwrap();
    let placed = live_instructions(ssa.clone());
    let tag = intrinsic(name);
    let sites = placed
        .functions
        .iter()
        .map(|f| &f.body)
        .chain(placed.entry_points.iter().map(|e| &e.body))
        .flat_map(|body| body.inner.insts.values())
        .filter(|node| matches!(&node.data, InstKind::Op { tag: actual, .. } if *actual == tag))
        .count();
    assert_eq!(
        sites,
        expected,
        "unexpected {name} count before backend lowering\n{source}\n{}",
        crate::ssa::print::format_program(&placed)
    );
    let wgsl = crate::lower_ssa_to_wgsl(ssa.clone()).unwrap();
    let module = naga::front::wgsl::parse_str(&wgsl).unwrap();
    validate_shader(&module);
    let spirv = crate::lower_ssa_to_spirv(ssa).unwrap().spirv;
    let bytes = spirv.iter().flat_map(|word| word.to_le_bytes()).collect::<Vec<_>>();
    let module = naga::front::spv::parse_u8_slice(&bytes, &Default::default()).unwrap();
    validate_shader(&module);
    placed
}

fn validate_shader(module: &naga::Module) {
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(module)
    .unwrap();
}

fn live_instructions(ssa: crate::ssa::stage::Elaborated) -> crate::ssa::stage::Placed {
    let mut placed = crate::ssa::place_floating(optimize(ssa)).unwrap();
    // Backend preparation drops unused pure instructions. Count live producers
    // without introducing any sharing pass outside the production pipeline.
    for body in placed
        .functions
        .iter_mut()
        .map(|f| &mut f.body)
        .chain(placed.entry_points.iter_mut().map(|e| &mut e.body))
    {
        eliminate_dead_pure_instructions(body);
    }
    placed
}

#[test]
fn branch_clamp_reuses_normalization_and_dependent_projection() {
    let source = r#"
def clampi(v: i32, lo: i32, hi: i32) i32 =
  if v < lo then lo else if v > hi then hi else v
def pixel(p: vec3f32) vec2i32 =
  let n = normalize(p)
  let projected = n.xy / n.z in
  @[clampi(i32(projected.x), 0, 63), clampi(i32(projected.y), 0, 63)]
entry repro(points: []vec3f32) []vec2i32 = map(pixel, points)
"#;
    let placed = assert_partial_intrinsic_sites(source, "normalize", 1);
    assert_eq!(
        placed
            .functions
            .iter()
            .map(|f| &f.body)
            .chain(placed.entry_points.iter().map(|e| &e.body))
            .flat_map(|body| body.inner.insts.values())
            .filter(|node| {
                matches!(
                    node.data,
                    InstKind::Op {
                        tag: OpTag::BinOp(BinaryOperator::Divide),
                        ..
                    }
                )
            })
            .count(),
        1,
        "the projection must share the retained normalization"
    );
    let spirv = crate::compile_thru_spirv(source).unwrap().spirv;
    let module = wspirv::dr::load_words(&spirv).unwrap();
    assert_eq!(
        module.all_inst_iter().filter(|inst| inst.class.opcode == wspirv::spirv::Op::FDiv).count(),
        1
    );
    assert_eq!(
        module
            .all_inst_iter()
            .filter(|inst| {
                inst.class.opcode == wspirv::spirv::Op::ExtInst
                    && inst.operands.get(1) == Some(&wspirv::dr::Operand::LiteralExtInstInteger(69))
            })
            .count(),
        1
    );
}

#[test]
fn partial_intrinsic_reuse_respects_dominance_and_loop_carried_operands() {
    for (source, expected) in [
        (
            r#"entry repro(xs: []f32) []f32 = map(|x|
          let root = f32.sqrt(x) in
          loop total = root for j < 2 do
            loop inner = total for k < 2 do inner + f32.sqrt(x), xs)"#,
            1,
        ),
        // A loop header dominates its merge, so its result can be reused.
        (
            r#"entry repro(xs: []f32) []f32 = map(|x|
          let total = loop total = 0.0 while total < f32.sqrt(x) do total + 1.0 in
          total + f32.sqrt(x), xs)"#,
            1,
        ),
        // The second sqrt's operand changes on every iteration.
        (
            r#"entry repro(xs: []f32) []f32 = map(|x|
          loop total = f32.sqrt(x) for k < 2 do f32.sqrt(total), xs)"#,
            2,
        ),
    ] {
        assert_partial_intrinsic_sites(source, "f32.sqrt", expected);
    }
    let placed = assert_partial_intrinsic_sites(
        r#"entry repro(xs: []f32, count: i32) []f32 =
      map(|x| loop total = 0.0 for k < count do total + f32.sqrt(x), xs)"#,
        "f32.sqrt",
        1,
    );
    let sqrt = intrinsic("f32.sqrt");
    for body in placed.functions.iter().map(|f| &f.body).chain(placed.entry_points.iter().map(|e| &e.body))
    {
        let scopes = LoopScopes::analyze(&body.inner);
        for node in body.inner.insts.values() {
            if matches!(&node.data, InstKind::Op { tag, .. } if *tag == sqrt) {
                assert!(
                    scopes.scope(node.placement.block().unwrap()).is_some(),
                    "a possibly empty loop must keep its sqrt guarded"
                );
            }
        }
    }
}

#[test]
fn dead_instruction_worklist_removes_long_chains() {
    let mut body = FuncBuilder::new(vec![(i32(), "x".into())], i32()).finish_unchecked();
    let x = body.inner.params[0].into();
    let mut value = x;
    for _ in 0..4096 {
        value =
            body.inner.append_inst(body.inner.entry, binary(BinaryOperator::Add, value, x), i32()).into();
    }
    body.inner.blocks[body.inner.entry].term = Terminator::Return(Some(x));
    eliminate_dead_pure_instructions(&mut body);
    assert_eq!(body.num_insts(), 0);
}

#[test]
fn division_reuse_distinguishes_loop_carried_operands() {
    assert_division_sites(
        "entry repro(xs:[]i32,width:i32) []i32 = map(|x|
         loop total=x/width+x%width for k<2 do total/width+total%width,xs)",
        2,
    );
}

#[test]
fn partial_math_is_not_reused_between_independent_guards() {
    for name in ["f32.sqrt", "f32.log"] {
        assert_partial_intrinsic_sites(
            &format!(
                "entry repro(xs:[]f32,first:bool,second:bool) []f32 = map(|x|
                let a=if first then {name}(x) else 0.0 in
                let b=if second then {name}(x) else 0.0 in a+b,xs)"
            ),
            name,
            2,
        );
    }
}

#[test]
fn partial_intrinsics_reuse_guarded_producers_in_nested_branches() {
    for name in ["f32.sqrt", "f32.log"] {
        assert_partial_intrinsic_sites(
            &format!(
                "entry repro(xs:[]f32,outer:bool,inner:bool) []f32 = map(|x|
                if outer then let value={name}(x) in
                    value+(if inner then {name}(x)+1.0 else {name}(x)-1.0)
                else x,xs)"
            ),
            name,
            1,
        );
    }
}
