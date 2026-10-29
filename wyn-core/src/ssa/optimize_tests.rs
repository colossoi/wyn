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
        let mut builder = FuncBuilder::new(vec![ty.clone()], ty.clone());
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
    assert_case_operations(&[
        "entry main(i:i32,width:i32) i32=let q=i/width let r=i%width in if q<r then q-r else q+r",
        "entry main(i:i32,width:i32) i32=loop total=0 for j<3 do let q=(i+j)/width let r=(i+j)%width in total+(if q<r then q-r else q+r)",
    ], &[OpTag::BinOp(BinaryOperator::Divide), OpTag::BinOp(BinaryOperator::Remainder)], 1);
}

#[test]
fn division_and_remainder_are_reused_from_enclosing_loop_scopes() {
    let placed = live_instructions(
        crate::compile_thru_ssa(
            r#"
entry case_0(i:i32,width:i32) i32 =
    let pixel = @[i % width, i / width] in
    loop total = pixel.x + pixel.y for k < 2 do total + pixel.x + pixel.y
entry case_1(i:i32,width:i32) i32 =
    let pixel = @[i % width, i / width] in
    loop total = pixel.x + pixel.y for j < 2 do
      loop inner = total for k < 2 do inner + pixel.x + pixel.y
entry case_2(i:i32,width:i32) i32 =
    loop total = 0 for j < 2 do
      let pixel = @[(i + j) % width, (i + j) / width] in
      loop inner = total + pixel.x + pixel.y for k < 2 do
        inner + pixel.x + pixel.y
"#,
        )
        .unwrap(),
    );
    for index in 0..3 {
        let bodies = reachable_bodies(&placed, &format!("case_{index}"));
        for op in [BinaryOperator::Divide, BinaryOperator::Remainder] {
            let sites = bodies.iter().flat_map(|body| body.inner.insts.values()).filter(|node| matches!(&node.data, InstKind::Op { tag: OpTag::BinOp(actual), .. } if *actual == op)).count();
            assert_eq!(sites, 1, "case {index}: {op:?}");
        }
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
    let mut body = FuncBuilder::new(vec![sized_array(128, i32())], i32()).finish_unchecked();
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
    placed
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
}

#[test]
fn partial_intrinsic_reuse_respects_dominance_and_loop_carried_operands() {
    let placed = live_instructions(
        crate::compile_thru_ssa(
            r#"
entry case_0(x:f32) f32 =
          let root = f32.sqrt(x) in
          loop total = root for j < 2 do
            loop inner = total for k < 2 do inner + f32.sqrt(x)
entry case_1(x:f32) f32 =
          let total = loop total = 0.0 while total < f32.sqrt(x) do total + 1.0 in
          total + f32.sqrt(x)
entry case_2(x:f32) f32 =
          loop total = f32.sqrt(x) for k < 2 do f32.sqrt(total)
entry case_3(x:f32,count:i32) f32 =  loop total = 0.0 for k < count do total + f32.sqrt(x)
"#,
        )
        .unwrap(),
    );
    let sqrt = intrinsic("f32.sqrt");
    for (index, expected) in [1, 1, 2, 1].into_iter().enumerate() {
        let bodies = reachable_bodies(&placed, &format!("case_{index}"));
        let sites = bodies
            .iter()
            .flat_map(|body| body.inner.insts.values())
            .filter(|node| matches!(&node.data, InstKind::Op { tag, .. } if *tag == sqrt))
            .count();
        assert_eq!(sites, expected, "case {index}");
        if index == 3 {
            for body in bodies {
                let scopes = LoopScopes::analyze(&body.inner);
                for node in body.inner.insts.values() {
                    if matches!(&node.data, InstKind::Op { tag, .. } if *tag == sqrt) {
                        assert!(
                            scopes.scope(node.placement.block().unwrap()).is_some(),
                            "possibly empty loop must guard sqrt"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn dead_instruction_worklist_removes_long_chains() {
    let mut body = FuncBuilder::new(vec![i32()], i32()).finish_unchecked();
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
    assert_partial_cases(
        "let a=if first then OP(x) else 0.0 in let b=if second then OP(x) else 0.0 in a+b",
        2,
    );
}

#[test]
fn partial_intrinsics_reuse_guarded_producers_in_nested_branches() {
    assert_partial_cases(
        "if first then let value=OP(x) in value+(if second then OP(x)+1.0 else OP(x)-1.0) else x",
        1,
    );
}

// Inspect each case independently even when its producer is retained in a helper.
fn reachable_bodies<'a>(
    program: &'a crate::ssa::stage::Placed,
    entry: &str,
) -> Vec<&'a crate::ssa::types::FuncBody> {
    let mut bodies: Vec<_> = program
        .entry_points
        .iter()
        .filter(|e| e.name == entry || e.name.starts_with(&format!("{entry}_")))
        .map(|e| &e.body)
        .collect();
    assert!(!bodies.is_empty(), "missing entry {entry}");
    let mut seen = crate::LookupSet::default();
    let mut index = 0;
    while index < bodies.len() {
        let mut callees = Vec::new();
        for node in bodies[index].inner.insts.values() {
            if let InstKind::Op {
                tag: OpTag::Call(id), ..
            } = &node.data
            {
                callees.push(*id);
            }
        }
        for id in callees {
            if seen.insert(id) {
                bodies.push(&program.functions.iter().find(|f| f.id == id).unwrap().body);
            }
        }
        index += 1;
    }
    bodies
}

fn assert_case_operations(sources: &[&str], tags: &[OpTag<BindingRef, FunctionId>], expected: usize) {
    let source = sources
        .iter()
        .enumerate()
        .map(|(i, source)| source.replacen("entry main(", &format!("entry case_{i}("), 1))
        .collect::<Vec<_>>()
        .join("\n");
    let placed = live_instructions(crate::compile_thru_ssa(&source).unwrap());
    for index in 0..sources.len() {
        let bodies = reachable_bodies(&placed, &format!("case_{index}"));
        assert!(!bodies.is_empty());
        for tag in tags {
            let count = bodies
                .iter()
                .flat_map(|body| body.inner.insts.values())
                .filter(|node| matches!(&node.data, InstKind::Op { tag: actual, .. } if actual == tag))
                .count();
            assert_eq!(count, expected, "case {index}: {tag:?}");
        }
    }
}

fn assert_partial_cases(expression: &str, expected: usize) {
    let names = ["f32.sqrt", "f32.log"];
    let source = names
        .iter()
        .enumerate()
        .map(|(index, name)| {
            format!(
                "entry case_{index}(x:f32,first:bool,second:bool) f32={}",
                expression.replace("OP", name)
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let placed = live_instructions(crate::compile_thru_ssa(&source).unwrap());
    for (index, name) in names.into_iter().enumerate() {
        let tag = intrinsic(name);
        let count = reachable_bodies(&placed, &format!("case_{index}"))
            .iter()
            .flat_map(|body| body.inner.insts.values())
            .filter(|node| matches!(&node.data, InstKind::Op { tag: actual, .. } if *actual == tag))
            .count();
        assert_eq!(count, expected, "{name}");
    }
}
