use super::*;
use crate::ssa::builder::FuncBuilder;
use crate::ssa::eliminate_dead_pure_instructions;
use crate::ssa::types::{ConstantValue, Terminator};
use crate::types::{bool_type, i32, sized_array};

fn binary(op: BinaryOperator, a: ValueRef, b: ValueRef) -> InstKind {
    InstKind::Op {
        tag: OpTag::BinOp(op),
        operands: vec![a, b],
    }
}

#[test]
fn common_branch_work_moves_to_its_operand_dominator_but_division_stays_guarded() {
    let mut body = FuncBuilder::new(
        vec![(i32(), "x".into()), (bool_type(), "condition".into())],
        i32(),
    )
    .finish_unchecked();
    let entry = body.inner.entry;
    let x = body.inner.params[0].into();
    let condition = body.inner.params[1].into();
    let left = body.inner.create_block();
    let right = body.inner.create_block();
    body.inner.blocks[entry].term = Terminator::CondBranch {
        cond: condition,
        then_target: left,
        then_args: vec![],
        else_target: right,
        else_args: vec![],
    };
    for block in [left, right] {
        let doubled = body.inner.append_inst(block, binary(BinaryOperator::Add, x, x), i32());
        let divided =
            body.inner.append_inst(block, binary(BinaryOperator::Divide, doubled.into(), x), i32());
        body.inner.blocks[block].term = Terminator::Return(Some(divided.into()));
    }
    float_pure_values(&mut body);
    crate::ssa::ir::schedule_floating(&mut body.inner).unwrap();
    assert_eq!(body.inner.blocks[entry].insts.len(), 1);
    for block in [left, right] {
        assert_eq!(body.inner.blocks[block].insts.len(), 1);
        assert!(matches!(
            body.inner.insts[body.inner.blocks[block].insts[0]].data,
            InstKind::Op {
                tag: OpTag::BinOp(BinaryOperator::Divide),
                ..
            }
        ));
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
        assert_branch_division_reuse(source);
    }
}

fn assert_branch_division_reuse(source: &str) {
    let ssa = crate::compile_thru_ssa(source).unwrap();
    let placed = crate::ssa::place_floating(optimize(ssa.clone())).unwrap();
    for operator in [BinaryOperator::Divide, BinaryOperator::Remainder] {
        let sites = placed
            .functions
            .iter()
            .flat_map(|f| f.body.inner.insts.values())
            .filter(
                |node| matches!(node.data, InstKind::Op { tag: OpTag::BinOp(op), .. } if op == operator),
            )
            .count();
        assert_eq!(sites, 1, "{operator:?} must be shared before backend lowering");
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
            1,
            "{opcode:?} must be emitted once"
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
}

#[test]
fn partial_arithmetic_is_reused_locally_but_not_from_a_sibling_or_into_a_merge() {
    for operator in [
        BinaryOperator::Divide,
        BinaryOperator::Remainder,
        BinaryOperator::FloorDivide,
        BinaryOperator::FloorRemainder,
        BinaryOperator::Power,
    ] {
        let mut body = FuncBuilder::new(
            vec![
                (i32(), "x".into()),
                (i32(), "width".into()),
                (bool_type(), "condition".into()),
            ],
            i32(),
        )
        .finish_unchecked();
        let entry = body.inner.entry;
        // Allocate the merge first to make allocation order differ from dominance order.
        let merge = body.inner.create_block();
        let joined = body.inner.add_block_param(merge, i32());
        let left = body.inner.create_block();
        let right = body.inner.create_block();
        let x = body.inner.params[0].into();
        let width = body.inner.params[1].into();
        body.inner.blocks[entry].term = Terminator::CondBranch {
            cond: body.inner.params[2].into(),
            then_target: left,
            then_args: vec![],
            else_target: right,
            else_args: vec![],
        };
        for block in [left, right, merge] {
            let a = body.inner.append_inst(block, binary(operator, x, width), i32());
            let b = body.inner.append_inst(block, binary(operator, x, width), i32());
            let sum = body.inner.append_inst(block, binary(BinaryOperator::Add, a.into(), b.into()), i32());
            body.inner.blocks[block].term = if block == merge {
                let result = body.inner.append_inst(
                    block,
                    binary(BinaryOperator::Add, joined.into(), sum.into()),
                    i32(),
                );
                Terminator::Return(Some(result.into()))
            } else {
                Terminator::Branch {
                    target: merge,
                    args: vec![sum.into()],
                }
            };
        }
        float_pure_values(&mut body);
        crate::ssa::ir::schedule_floating(&mut body.inner).unwrap();
        assert!(body.inner.blocks[entry].insts.is_empty());
        for block in [left, right, merge] {
            assert_eq!(
                body.inner.blocks[block].insts.iter().filter(|&&id| {
                    matches!(body.inner.insts[id].data, InstKind::Op { tag: OpTag::BinOp(op), .. } if op == operator)
                }).count(),
                1,
                "{operator:?}: each arm and the merge must retain its own computation"
            );
        }
    }
}

#[test]
fn division_reuse_distinguishes_loop_carried_operands() {
    let mut body =
        FuncBuilder::new(vec![(i32(), "x".into()), (i32(), "width".into())], i32()).finish_unchecked();
    let entry = body.inner.entry;
    let loop_body = body.inner.create_block();
    let carried = body.inner.add_block_param(loop_body, i32());
    let width = body.inner.params[1].into();
    let initial = body.inner.append_inst(
        entry,
        binary(BinaryOperator::Divide, body.inner.params[0].into(), width),
        i32(),
    );
    body.inner.blocks[entry].term = Terminator::Branch {
        target: loop_body,
        args: vec![initial.into()],
    };
    let a = body.inner.append_inst(
        loop_body,
        binary(BinaryOperator::Divide, carried.into(), width),
        i32(),
    );
    let b = body.inner.append_inst(
        loop_body,
        binary(BinaryOperator::Divide, carried.into(), width),
        i32(),
    );
    let next = body.inner.append_inst(loop_body, binary(BinaryOperator::Add, a.into(), b.into()), i32());
    body.inner.blocks[loop_body].term = Terminator::Branch {
        target: loop_body,
        args: vec![next.into()],
    };
    float_pure_values(&mut body);
    crate::ssa::ir::schedule_floating(&mut body.inner).unwrap();
    let divisions: Vec<_> = body
        .inner
        .insts
        .values()
        .filter(|node| {
            matches!(
                node.data,
                InstKind::Op {
                    tag: OpTag::BinOp(BinaryOperator::Divide),
                    ..
                }
            )
        })
        .collect();
    assert_eq!(divisions.len(), 2);
    assert!(divisions.iter().any(|node| node.placement.block() == Some(entry)));
    assert!(divisions.iter().any(|node| node.placement.block() == Some(loop_body)));
    let InstKind::Op { operands, .. } = &body.inner.insts[body.inner.inst_of_value(next).unwrap()].data
    else {
        panic!("sum")
    };
    assert_eq!(*operands, vec![a.into(), a.into()]);
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
    float_pure_values(&mut body);
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

#[test]
fn helper_substitution_does_not_confuse_distinct_value_arenas() {
    let mut helper =
        FuncBuilder::new(vec![(i32(), "a".into()), (i32(), "b".into())], i32()).finish_unchecked();
    let sum = helper.inner.append_inst(
        helper.inner.entry,
        binary(
            BinaryOperator::Subtract,
            helper.inner.params[0].into(),
            helper.inner.params[1].into(),
        ),
        i32(),
    );
    helper.inner.blocks[helper.inner.entry].term = Terminator::Return(Some(sum.into()));
    let mut caller =
        FuncBuilder::new(vec![(i32(), "x".into()), (i32(), "y".into())], i32()).finish_unchecked();
    let result = caller.inner.append_inst(
        caller.inner.entry,
        InstKind::Op {
            tag: OpTag::Call(FunctionId::from(0)),
            operands: vec![caller.inner.params[1].into(), caller.inner.params[0].into()],
        },
        i32(),
    );
    caller.inner.blocks[caller.inner.entry].term = Terminator::Return(Some(result.into()));
    inline_small_helpers(&mut caller, |_| Some(&helper));
    let [id] = caller.inner.blocks[caller.inner.entry].insts.as_slice() else {
        panic!("single inlined instruction")
    };
    let InstKind::Op {
        tag: OpTag::BinOp(BinaryOperator::Subtract),
        operands,
    } = &caller.inner.insts[*id].data
    else {
        panic!("subtraction")
    };
    assert_eq!(
        *operands,
        vec![caller.inner.params[1].into(), caller.inner.params[0].into()]
    );
}

fn intrinsic(name: &str) -> OpTag<BindingRef, FunctionId> {
    OpTag::Intrinsic {
        id: crate::builtins::catalog().lookup_by_any_name(name).unwrap().id,
        overload_idx: 0,
    }
}

#[test]
fn dominance_reuse_preserves_loads_calls_and_context_dependent_intrinsics() {
    let float = Type::Constructed(TypeName::Float(32), vec![]);
    let mut builder = FuncBuilder::new(vec![(float.clone(), "x".into())], float.clone());
    let x = builder.get_param(0).into();
    let place = builder.new_place(float.clone());
    builder
        .push_void_inst(InstKind::Alloca {
            elem_ty: float.clone(),
            result: place,
        })
        .unwrap();
    builder.push_void_inst(InstKind::Store { place, value: x }).unwrap();
    let child = builder.create_block();
    let tags = [OpTag::Call(FunctionId::from(99)), intrinsic("f32.d_fdx")];
    for block in [builder.entry(), child] {
        builder.switch_to_block_unchecked(block);
        let loaded = builder.push_inst(InstKind::Load { place }, float.clone()).unwrap();
        let divided =
            builder.push_inst(binary(BinaryOperator::Divide, loaded.into(), x), float.clone()).unwrap();
        for tag in &tags {
            builder
                .push_inst(
                    InstKind::Op {
                        tag: tag.clone(),
                        operands: vec![x],
                    },
                    float.clone(),
                )
                .unwrap();
        }
        if block == child {
            builder.terminate(Terminator::Return(Some(divided.into()))).unwrap();
        } else {
            // The child's load must observe this write, not reuse the entry load.
            builder
                .push_void_inst(InstKind::Store {
                    place,
                    value: divided.into(),
                })
                .unwrap();
            builder
                .terminate(Terminator::Branch {
                    target: child,
                    args: vec![],
                })
                .unwrap();
        }
    }
    let mut body = builder.finish().unwrap();
    float_pure_values(&mut body);
    crate::ssa::ir::schedule_floating(&mut body.inner).unwrap();
    assert_eq!(
        body.inner.insts.values().filter(|node| matches!(node.data, InstKind::Load { .. })).count(),
        2
    );
    for tag in tags.into_iter().chain([OpTag::BinOp(BinaryOperator::Divide)]) {
        assert_eq!(
            body.inner
                .insts
                .values()
                .filter(|node| { matches!(&node.data, InstKind::Op { tag: actual, .. } if *actual == tag) })
                .count(),
            2,
            "{tag:?}: dominance alone does not prove equal values"
        );
    }
}

#[test]
fn inlining_preserves_guards_on_partial_math_and_context_dependent_intrinsics() {
    let float = Type::Constructed(TypeName::Float(32), vec![]);
    for tag in [
        OpTag::BinOp(BinaryOperator::Divide),
        intrinsic("f32.sqrt"),
        intrinsic("f32.log"),
        intrinsic("f32.d_fdx"),
        OpTag::Call(FunctionId::from(99)),
    ] {
        let mut helper =
            FuncBuilder::new(vec![(float.clone(), "x".into())], float.clone()).finish_unchecked();
        let x = helper.inner.params[0].into();
        let sum = helper.inner.append_inst(
            helper.inner.entry,
            binary(BinaryOperator::Add, x, x),
            float.clone(),
        );
        let operands = if matches!(tag, OpTag::BinOp(_)) { vec![sum.into(), x] } else { vec![sum.into()] };
        let value = helper.inner.append_inst(
            helper.inner.entry,
            InstKind::Op {
                tag: tag.clone(),
                operands,
            },
            float.clone(),
        );
        helper.inner.blocks[helper.inner.entry].term = Terminator::Return(Some(value.into()));
        // Exercise the production order: simplify callees before their callers.
        float_pure_values(&mut helper);
        let mut caller = FuncBuilder::new(
            vec![(float.clone(), "x".into()), (bool_type(), "condition".into())],
            float.clone(),
        )
        .finish_unchecked();
        let entry = caller.inner.entry;
        let taken = caller.inner.create_block();
        let skipped = caller.inner.create_block();
        caller.inner.blocks[entry].term = Terminator::CondBranch {
            cond: caller.inner.params[1].into(),
            then_target: taken,
            then_args: vec![],
            else_target: skipped,
            else_args: vec![],
        };
        let value = caller.inner.append_inst(
            taken,
            InstKind::Op {
                tag: OpTag::Call(FunctionId::from(0)),
                operands: vec![caller.inner.params[0].into()],
            },
            float.clone(),
        );
        caller.inner.blocks[taken].term = Terminator::Return(Some(value.into()));
        caller.inner.blocks[skipped].term =
            Terminator::Return(Some(ValueRef::Const(ConstantValue::from_f32(0.0))));
        inline_small_helpers(&mut caller, |_| Some(&helper));
        float_pure_values(&mut caller);
        crate::ssa::ir::schedule_floating(&mut caller.inner).unwrap();
        assert_eq!(
            caller.num_insts(),
            2,
            "{tag:?}: helper must be expanded exactly once"
        );
        assert!(
            caller.inner.insts.values().any(|node| {
                matches!(&node.data, InstKind::Op { tag: actual, .. } if *actual == tag)
                    && node.placement.block() == Some(taken)
            }),
            "{tag:?}: inlining must preserve the original guard"
        );
        assert!(caller.inner.blocks[skipped].insts.is_empty());
    }
}

#[test]
fn rotate_helper_is_inlined_and_trig_is_placed_before_the_march_loop() {
    let source = r#"
def rotate(p: vec2f32, angle: f32) vec2f32 =
  let c = f32.cos(angle)
  let s = f32.sin(angle) in
  @[c * p.x - s * p.y, s * p.x + c * p.y]
def march(p: vec2f32, angle: f32) vec2f32 =
  loop total = @[0.0, 0.0] for k < 20 do
    total + rotate(p + @[f32(k) * 0.06, 0.0], angle)
entry repro(points: []vec2f32, angle: f32) []vec2f32 =
  map(|p| march(p, angle), points)
"#;
    let ssa = crate::compile_thru_ssa(source).unwrap();
    let sin = intrinsic("f32.sin");
    let cos = intrinsic("f32.cos");
    let rotate = ssa
        .functions
        .iter()
        .find(|f| {
            f.body
                .inner
                .insts
                .values()
                .any(|node| matches!(&node.data, InstKind::Op { tag, .. } if *tag == sin))
        })
        .unwrap()
        .id;
    let march = ssa
        .functions
        .iter()
        .find(|f| {
            f.body.inner.insts.values().any(
                |node| matches!(&node.data, InstKind::Op { tag: OpTag::Call(id), .. } if *id == rotate),
            )
        })
        .unwrap()
        .id;
    let placed = crate::ssa::place_floating(optimize(ssa.clone())).unwrap();
    let body = &placed.functions.iter().find(|f| f.id == march).unwrap().body;
    for expected in [sin, cos] {
        let nodes = body
            .inner
            .insts
            .values()
            .filter(|node| matches!(&node.data, InstKind::Op { tag, .. } if *tag == expected))
            .collect::<Vec<_>>();
        assert_eq!(nodes.len(), 1, "{expected:?} must be computed once");
        assert_eq!(nodes[0].placement.block(), Some(body.inner.entry));
    }
    assert!(!body
        .inner
        .insts
        .values()
        .any(|node| { matches!(node.data, InstKind::Op { tag: OpTag::Call(id), .. } if id == rotate) }));
    let wgsl = crate::lower_ssa_to_wgsl(ssa.clone()).unwrap();
    let module = naga::front::wgsl::parse_str(&wgsl).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    let spirv = crate::lower_ssa_to_spirv(ssa).unwrap();
    let bytes = spirv.spirv.iter().flat_map(|word| word.to_le_bytes()).collect::<Vec<_>>();
    let module = naga::front::spv::parse_u8_slice(&bytes, &Default::default()).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
}

#[test]
fn camera_helper_above_old_inline_limit_is_placed_before_the_march_loop() {
    let source = r#"
def camera_term(angle: f32) f32 =
  let a = f32.sin(f32.sin(f32.sin(f32.sin(angle))))
  let b = f32.sin(f32.sin(f32.sin(f32.sin(a))))
  let c = f32.sin(f32.sin(f32.sin(f32.sin(b))))
  let d = f32.sin(f32.sin(f32.sin(f32.sin(c)))) in
  f32.cos(d)
def march(angle: f32) f32 =
  loop total = 0.0 for k < 20 do
    total + camera_term(angle) * f32(k)
entry repro(angles: []f32) []f32 = map(march, angles)
"#;
    let ssa = crate::compile_thru_ssa(source).unwrap();
    let sin = intrinsic("f32.sin");
    let cos = intrinsic("f32.cos");
    let camera = ssa
        .functions
        .iter()
        .find(|f| {
            f.body
                .inner
                .insts
                .values()
                .any(|node| matches!(&node.data, InstKind::Op { tag, .. } if *tag == sin))
        })
        .unwrap();
    assert_eq!(
        camera.body.num_insts(),
        17,
        "must exceed the old late-inline limit"
    );
    let march = ssa
        .functions
        .iter()
        .find(|f| {
            f.body.inner.insts.values().any(
                |node| matches!(&node.data, InstKind::Op { tag: OpTag::Call(id), .. } if *id == camera.id),
            )
        })
        .unwrap()
        .id;
    let placed = crate::ssa::place_floating(optimize(ssa.clone())).unwrap();
    let body = &placed.functions.iter().find(|f| f.id == march).unwrap().body;
    for (expected, count) in [(sin, 16), (cos, 1)] {
        let nodes = body
            .inner
            .insts
            .values()
            .filter(|node| matches!(&node.data, InstKind::Op { tag, .. } if *tag == expected))
            .collect::<Vec<_>>();
        assert_eq!(nodes.len(), count);
        assert!(
            nodes.iter().all(|node| node.placement.block() == Some(body.inner.entry)),
            "{expected:?} must be computed before the march loop"
        );
    }
    assert!(!body
        .inner
        .insts
        .values()
        .any(|node| matches!(node.data, InstKind::Op { tag: OpTag::Call(id), .. } if id == camera.id)));
    let wgsl = crate::lower_ssa_to_wgsl(ssa.clone()).unwrap();
    let module = naga::front::wgsl::parse_str(&wgsl).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    let spirv = crate::lower_ssa_to_spirv(ssa).unwrap();
    let bytes = spirv.spirv.iter().flat_map(|word| word.to_le_bytes()).collect::<Vec<_>>();
    let module = naga::front::spv::parse_u8_slice(&bytes, &Default::default()).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
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
fn newly_exposed_integer_constants_wrap_at_the_declared_width() {
    use crate::types::Type;
    let ty = Type::Constructed(crate::types::TypeName::UInt(32), vec![]);
    let mut body = FuncBuilder::new(vec![], ty.clone()).finish_unchecked();
    let value = body.inner.append_inst(
        body.inner.entry,
        binary(
            BinaryOperator::Multiply,
            ValueRef::Const(ConstantValue::U32(2654435769)),
            ValueRef::Const(ConstantValue::U32(747796405)),
        ),
        ty,
    );
    body.inner.blocks[body.inner.entry].term = Terminator::Return(Some(value.into()));
    float_pure_values(&mut body);
    crate::ssa::ir::schedule_floating(&mut body.inner).unwrap();
    assert!(
        matches!(body.inner.blocks[body.inner.entry].term, Terminator::Return(Some(ValueRef::Const(ConstantValue::U32(n)))) if n == 2654435769u32.wrapping_mul(747796405))
    );
}
