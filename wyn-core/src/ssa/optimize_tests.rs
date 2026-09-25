use super::*;
use crate::ssa::builder::FuncBuilder;
use crate::ssa::eliminate_dead_pure_instructions;
use crate::ssa::types::{ConstantValue, Terminator};
use crate::types::{bool_type, i32, sized_array};
use crate::{BindingRef, FunctionId};

fn binary(op: BinaryOperator, a: ValueRef, b: ValueRef) -> InstKind {
    InstKind::Op {
        tag: OpTag::BinOp(op),
        operands: vec![a, b],
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
    for function in &placed.functions {
        let scopes = LoopScopes::analyze(&function.body.inner);
        for node in function.body.inner.insts.values() {
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
    let mut ssa = crate::compile_thru_ssa(source).unwrap();
    // Exercise the retained SSA pass explicitly; it is outside the production pipeline.
    for body in ssa
        .functions
        .iter_mut()
        .map(|f| &mut f.body)
        .chain(ssa.entry_points.iter_mut().map(|e| &mut e.body))
    {
        reuse_dominating_expressions(body);
    }
    let placed = crate::ssa::place_floating(optimize(ssa.clone())).unwrap();
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
            sites, expected,
            "unexpected {operator:?} count before backend lowering"
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
fn partial_math_is_reused_locally_but_not_from_a_sibling_or_into_a_merge() {
    let binary_cases = [
        BinaryOperator::Divide,
        BinaryOperator::Remainder,
        BinaryOperator::FloorDivide,
        BinaryOperator::FloorRemainder,
        BinaryOperator::Power,
    ]
    .into_iter()
    .map(|operator| (OpTag::BinOp(operator), i32()));
    let intrinsic_cases = [
        (intrinsic("f32.sqrt"), crate::types::f32()),
        (intrinsic("f32.log"), crate::types::f32()),
        (intrinsic("normalize"), crate::types::vec(3, crate::types::f32())),
    ];
    for (tag, ty) in binary_cases.chain(intrinsic_cases) {
        let mut body = FuncBuilder::new(
            vec![
                (ty.clone(), "x".into()),
                (ty.clone(), "width".into()),
                (bool_type(), "condition".into()),
            ],
            ty.clone(),
        )
        .finish_unchecked();
        let entry = body.inner.entry;
        // Allocate the merge first to make allocation order differ from dominance order.
        let merge = body.inner.create_block();
        let joined = body.inner.add_block_param(merge, ty.clone());
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
        let data = InstKind::Op {
            tag: tag.clone(),
            operands: if matches!(tag, OpTag::BinOp(_)) { vec![x, width] } else { vec![x] },
        };
        for block in [left, right, merge] {
            let a = body.inner.append_inst(block, data.clone(), ty.clone());
            let b = body.inner.append_inst(block, data.clone(), ty.clone());
            let sum =
                body.inner.append_inst(block, binary(BinaryOperator::Add, a.into(), b.into()), ty.clone());
            body.inner.blocks[block].term = if block == merge {
                let result = body.inner.append_inst(
                    block,
                    binary(BinaryOperator::Add, joined.into(), sum.into()),
                    ty.clone(),
                );
                Terminator::Return(Some(result.into()))
            } else {
                Terminator::Branch {
                    target: merge,
                    args: vec![sum.into()],
                }
            };
        }
        reuse_dominating_expressions(&mut body);
        crate::ssa::ir::schedule_floating(&mut body.inner).unwrap();
        assert!(body.inner.blocks[entry].insts.is_empty());
        for block in [left, right, merge] {
            assert_eq!(
                body.inner.blocks[block].insts.iter().filter(|&&id| {
                    matches!(&body.inner.insts[id].data, InstKind::Op { tag: actual, .. } if *actual == tag)
                }).count(),
                1,
                "{tag:?}: each arm and the merge must retain its own computation"
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
    reuse_dominating_expressions(&mut body);
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
    let mut ssa = crate::compile_thru_ssa(source).unwrap();
    // Exercise the retained SSA pass explicitly; it is outside the production pipeline.
    for body in ssa
        .functions
        .iter_mut()
        .map(|f| &mut f.body)
        .chain(ssa.entry_points.iter_mut().map(|e| &mut e.body))
    {
        reuse_dominating_expressions(body);
    }
    let placed = crate::ssa::place_floating(optimize(ssa.clone())).unwrap();
    let tag = intrinsic(name);
    let sites = placed
        .functions
        .iter()
        .map(|f| &f.body)
        .chain(placed.entry_points.iter().map(|e| &e.body))
        .flat_map(|body| body.inner.insts.values())
        .filter(|node| matches!(&node.data, InstKind::Op { tag: actual, .. } if *actual == tag))
        .count();
    assert_eq!(sites, expected, "unexpected {name} count before backend lowering");
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

#[test]
fn intrinsic_reuse_excludes_pointer_and_context_operations() {
    use crate::builtins::lowering::{BuiltinLowering, PrimOp};
    // GLSL Modf/Frexp have output pointers; InterpolateAt* reads fragment context.
    for ext in [35, 51, 76, 77, 78, u32::MAX] {
        assert!(!BuiltinLowering::PrimOp(PrimOp::GlslExt(ext)).is_reusable());
    }
    for name in [
        "_w_intrinsic_uninit",
        "_w_intrinsic_storage_index",
        "f32.d_fdx",
        "f32.d_fdy",
        "f32.fwidth",
    ] {
        assert!(
            !reusable(&InstKind::Op {
                tag: intrinsic(name),
                operands: vec![]
            }),
            "{name}"
        );
    }
    assert!(!BuiltinLowering::LinkedSpirv("unknown").is_reusable());
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
fn partial_intrinsics_reuse_guarded_producers_in_nested_branches() {
    for (name, ty) in [
        ("f32.sqrt", crate::types::f32()),
        ("f32.log", crate::types::f32()),
        ("normalize", crate::types::vec(3, crate::types::f32())),
    ] {
        let tag = intrinsic(name);
        let mut body = FuncBuilder::new(
            vec![(ty.clone(), "x".into()), (bool_type(), "condition".into())],
            ty.clone(),
        )
        .finish_unchecked();
        let entry = body.inner.entry;
        let guarded = body.inner.create_block();
        let skipped = body.inner.create_block();
        let left = body.inner.create_block();
        let right = body.inner.create_block();
        let x = body.inner.params[0].into();
        let cond = body.inner.params[1].into();
        body.inner.blocks[entry].term = Terminator::CondBranch {
            cond,
            then_target: guarded,
            then_args: vec![],
            else_target: skipped,
            else_args: vec![],
        };
        body.inner.blocks[guarded].term = Terminator::CondBranch {
            cond,
            then_target: left,
            then_args: vec![],
            else_target: right,
            else_args: vec![],
        };
        body.inner.blocks[skipped].term = Terminator::Return(Some(x));
        let data = InstKind::Op {
            tag: tag.clone(),
            operands: vec![x],
        };
        let producer = body.inner.append_inst(guarded, data.clone(), ty.clone());
        for block in [left, right] {
            let a = body.inner.append_inst(block, data.clone(), ty.clone());
            let b = body.inner.append_inst(block, data.clone(), ty.clone());
            let sum =
                body.inner.append_inst(block, binary(BinaryOperator::Add, a.into(), b.into()), ty.clone());
            body.inner.blocks[block].term = Terminator::Return(Some(sum.into()));
        }
        reuse_dominating_expressions(&mut body);
        crate::ssa::ir::schedule_floating(&mut body.inner).unwrap();
        let sites = body
            .inner
            .insts
            .values()
            .filter(|node| matches!(&node.data, InstKind::Op { tag: actual, .. } if *actual == tag))
            .collect::<Vec<_>>();
        assert_eq!(sites.len(), 1, "{name}: reuse the guarded producer");
        assert_eq!(sites[0].result, Some(producer));
        assert_eq!(sites[0].placement.block(), Some(guarded));
        assert!(body.inner.blocks[entry].insts.is_empty());
        assert!(body.inner.blocks[skipped].insts.is_empty());
        let sum = body
            .inner
            .insts
            .values()
            .find(|node| {
                matches!(
                    node.data,
                    InstKind::Op {
                        tag: OpTag::BinOp(BinaryOperator::Add),
                        ..
                    }
                )
            })
            .unwrap();
        assert!(
            matches!(&sum.data, InstKind::Op { operands, .. } if *operands == vec![producer.into(), producer.into()])
        );
    }
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
    for function in &placed.functions {
        let scopes = LoopScopes::analyze(&function.body.inner);
        for node in function.body.inner.insts.values() {
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
    let tags = [
        OpTag::Call(FunctionId::from(99)),
        intrinsic("f32.d_fdx"),
        intrinsic("f32.d_fdy"),
        intrinsic("f32.fwidth"),
    ];
    for block in [builder.entry(), child] {
        builder.switch_to_block_unchecked(block);
        let loaded = builder.push_inst(InstKind::Load { place }, float.clone()).unwrap();
        let divided =
            builder.push_inst(binary(BinaryOperator::Divide, loaded.into(), x), float.clone()).unwrap();
        // Immutable operands can share across a store; separate loads cannot.
        for operand in [x, loaded.into()] {
            builder
                .push_inst(
                    InstKind::Op {
                        tag: intrinsic("f32.sqrt"),
                        operands: vec![operand],
                    },
                    float.clone(),
                )
                .unwrap();
        }
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
    reuse_dominating_expressions(&mut body);
    crate::ssa::ir::schedule_floating(&mut body.inner).unwrap();
    assert_eq!(
        body.inner.insts.values().filter(|node| matches!(node.data, InstKind::Load { .. })).count(),
        2
    );
    let sqrt = intrinsic("f32.sqrt");
    assert_eq!(
        body.inner
            .insts
            .values()
            .filter(|node| { matches!(&node.data, InstKind::Op { tag, .. } if *tag == sqrt) })
            .count(),
        3,
        "share sqrt(x), but retain sqrt of each mutable load"
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
