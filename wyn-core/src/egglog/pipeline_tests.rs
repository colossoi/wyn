//! Regression inputs exercise the public pipeline and backend-visible output.
use crate::compile_thru_ssa;
use crate::egglog::query::Query;
use crate::egglog::{prepare_interfaces, scalar, ScalarOptimization};

#[test]
fn size_hint_survives_egglog_optimization() {
    use super::{from_tlc, fuse, optimize_for_test, place, schedule};
    use crate::interface::Attribute;
    use crate::tlc::DefMeta;
    use crate::PipelineTopologyPolicy;

    let source = "entry keep(#[size_hint(1024)] data:[]i32) []i32=data";
    let tlc = crate::compile_thru_tlc(source).unwrap();
    let program = prepare_interfaces(
        schedule(
            place(
                optimize_for_test(fuse(from_tlc(&tlc).unwrap()).unwrap()).unwrap(),
                PipelineTopologyPolicy::AllowGenerated,
            )
            .unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    let mut entries = 0;
    for definition in &program.source.defs {
        if let DefMeta::EntryPoint(entry) = &definition.meta {
            entries += 1;
            let data = entry.declaration.params.iter().find(|param| param.name == "data").unwrap();
            assert!(data
                .attributes
                .contains(&Attribute::SizeHint(std::num::NonZeroU32::new(1024).unwrap())));
        }
    }
    assert!(entries > 0);
}

// Scratch element liveness is checked in memory_ir_dead_elements.wyn.
#[test]
fn host_preparation_preserves_the_authoritative_memory_program() {
    let tlc =
        crate::compile_thru_tlc("entry main(xs:[]i32,bias:i32) []i32=map(|x|x+bias*bias,xs)").unwrap();
    let program = interfaces_for_test(&tlc);
    let before = super::dataflow::edit::revision(&program.graph).unwrap();
    let program = super::prepare_host(program).unwrap();
    let after = super::dataflow::edit::revision(&program.program.graph).unwrap();
    assert_eq!(before, after, "host readout must not rebuild executable bodies");
    let ssa = super::to_ssa(program, crate::CodegenTarget::Portable).unwrap();
    assert!(ssa.entry_points.iter().any(|entry| entry.inputs.iter().any(|input| input.name == "bias")));
}

// Broad policy/backend coverage lives in tests/compiler/cases/scalar_policy_*.wyn.
#[test]
fn scalar_policy_controls_optional_templates() {
    let tlc =
        crate::compile_thru_tlc("def helper(x:i32) i32=x+1 entry main(x:i32) i32=helper(x)*2").unwrap();
    for policy in [super::ScalarOptimization::Basic, super::ScalarOptimization::Full] {
        let mut scalars = super::fuse(super::from_tlc(&tlc).unwrap()).unwrap().graph;
        super::scalar::run(&mut scalars, policy).unwrap();
        let query = Query(&scalars);
        let mut calls = 0;
        query
            .for_each("LiveOperation", |row| {
                calls += usize::from(query.contains("CallOperation", row[0])?);
                Ok(())
            })
            .unwrap();
        if policy == super::ScalarOptimization::Full {
            assert_eq!(calls, 0);
        }
    }
}

#[test]
fn scalar_entry_lowers_to_compute_ssa() {
    let module = placed_source("entry scalar(x:i32) i32=x+1");
    assert_eq!(module.entry_points.len(), 1);
    assert!(module.entry_points[0].execution_model.is_compute());
}

#[test]
fn batched_scalar_extraction_keeps_entry_contexts_distinct() {
    let module = placed_source(
        "entry first(x:u32) u32=x+11u32
         entry second(x:u32) u32=x+29u32",
    );
    for (name, expected, other) in [("first", 11, 29), ("second", 29, 11)] {
        let entry = module
            .entry_points
            .iter()
            .find(|entry| entry.name.contains(name))
            .expect("each authored entry must reach the backend");
        let literals: Vec<_> = entry
            .body
            .inner
            .insts
            .values()
            .flat_map(|instruction| {
                let mut values = instruction.data.value_uses();
                if let crate::ssa::types::InstKind::Op {
                    tag: crate::op::OpTag::Uint(value),
                    ..
                } = &instruction.data
                {
                    values.push(crate::ssa::types::ValueRef::Const(
                        crate::ssa::types::ConstantValue::U32(value.parse().unwrap()),
                    ));
                }
                values
            })
            .filter_map(|value| match value {
                crate::ssa::types::ValueRef::Const(crate::ssa::types::ConstantValue::U32(value)) => {
                    Some(value)
                }
                _ => None,
            })
            .collect();
        assert!(
            literals.contains(&expected),
            "{name} lost its own arithmetic operand"
        );
        assert!(
            !literals.contains(&other),
            "{name} acquired another context's operand"
        );
    }
}

#[test]
fn storage_input_reads_preserve_the_public_interface() {
    let module = placed_source("entry first(xs:[]i32) i32=xs[0]");
    assert!(
        module.entry_points.iter().any(|entry| !entry.storage_bindings.is_empty()
            || entry.inputs.iter().any(|input| input.storage_binding().is_some()))
    );
}

#[test]
fn fused_maps_preserve_captured_inputs() {
    let program = placed_source("entry result(xs:[]i32,n:i32) []i32=map(|x:i32|x*2,map(|x:i32|x+n,xs))");
    assert_eq!(program.entry_points.len(), 1, "fused maps remain one stage");
    assert!(
        program.entry_points[0].inputs.iter().any(|input| input.name == "n"),
        "capture must reach the fused interface"
    );
}

// Target-independent preparation is shared; each backend still lowers and validates separately.
fn interfaces_for_test(
    tlc: &crate::tlc::stage::OwnershipApplied,
) -> super::Program<'_, super::InterfacesPrepared> {
    let program = super::from_tlc(tlc).unwrap();
    let program = super::fuse(program).unwrap();
    let program = super::optimize(program).unwrap();
    let program = super::place(program, crate::PipelineTopologyPolicy::AllowGenerated).unwrap();
    let program = super::schedule(program).unwrap();
    prepare_interfaces(program).unwrap()
}

// Inspect both emitted shader formats under the CLI's default and -O policies.

// Loop capacity is checked through both backends in memory_ir_loop_capacity.wyn.
#[test]
fn helper_substitution_keeps_argument_contexts_and_shared_products() {
    use crate::op::{BinaryOperator, OpTag};
    use crate::ssa::types::InstKind;
    let program = compile_thru_ssa(
        "def shared(x:u32,y:u32) u32 =
        let a=x*y+x let b=a*a+a let c=b*b+b let d=c*c+c let e=d*d+d in
        if y==0u32 then x else e
        entry one(x:u32,y:u32) u32=shared(x,y)
        entry two(x:u32,y:u32,u:u32,v:u32) (u32,u32)=(shared(x,y),shared(u,v))",
    )
    .unwrap();
    let program =
        crate::ssa::filter_reachable(crate::ssa::place_floating(crate::ssa::optimize(program)).unwrap());
    assert!(program.functions.is_empty(), "shared arithmetic must inline");
    for (name, expected) in [("one", 5), ("two", 10)] {
        let entry = program.entry_points.iter().find(|entry| entry.name.contains(name)).unwrap();
        let products = entry
            .body
            .inner
            .insts
            .values()
            .filter(|instruction| {
                matches!(
                    instruction.data,
                    InstKind::Op {
                        tag: OpTag::BinOp(BinaryOperator::Multiply),
                        ..
                    }
                )
            })
            .count();
        assert_eq!(
            products, expected,
            "{name}: substitutions must preserve sharing and argument identity"
        );
    }
}

#[test]
fn partial_helpers_inline_without_escaping_their_guard() {
    use crate::op::{BinaryOperator, OpTag};
    use crate::ssa::types::InstKind;
    use crate::structured::{structurize, Node};
    let program = placed_source(
        "def divide(x:u32,y:u32) u32=if y==0u32 then x else x/y
        def shift(x:u32,y:u32) u32=if y==0u32 then x else x<<y
        entry guarded_div(x:u32,y:u32) u32=divide(x,y)
        entry guarded_shift(x:u32,y:u32) u32=shift(x,y)
        entry choose(c:bool,x:i32,y:i32) i32=if c then x/y else 0",
    );
    assert!(program.functions.is_empty());
    for entry in &program.entry_points {
        let partial = |node: &Node| match node {
            Node::Inst(id) => matches!(
                entry.body.inner.insts[*id].data,
                InstKind::Op {
                    tag: OpTag::BinOp(BinaryOperator::Divide | BinaryOperator::ShiftLeft),
                    ..
                }
            ),
            _ => false,
        };
        let tree = structurize(&entry.body);
        assert!(
            !tree.root.nodes.iter().any(&partial),
            "{}: unguarded partial operation",
            entry.name
        );
        assert!(
            tree.root.nodes.iter().any(|node| match node {
                Node::If {
                    then_body, else_body, ..
                } => then_body.nodes.iter().any(&partial) ^ else_body.nodes.iter().any(&partial),
                _ => false,
            }),
            "{}: expected the partial operation in exactly one branch",
            entry.name
        );
    }
}

fn placed_source(source: &str) -> crate::ssa::stage::Reachable {
    crate::ssa::filter_reachable(
        crate::ssa::place_floating(crate::ssa::optimize(compile_thru_ssa(source).unwrap())).unwrap(),
    )
}

#[test]
fn scalar_admission_uses_ir_literal_edits() {
    let tlc = crate::compile_thru_tlc("entry main(x:i32) i32=x+1").unwrap();
    let mut program = interfaces_for_test(&tlc);
    let operation = Query(&program.graph).first("ScalarPayload").unwrap().unwrap()[0];
    super::dataflow::edit::set_code(
        &mut program.graph,
        operation,
        super::dataflow::ir::Code::Constant {
            ty: crate::types::i32(),
            bits: 7,
        },
    )
    .unwrap();
    super::dataflow::edit::inputs(&mut program.graph, operation, &[]).unwrap();
    scalar::run(&mut program.graph, ScalarOptimization::Basic).unwrap();
    let root = super::test_graph::root(&program.graph);
    let query = Query(&program.graph);
    let result = query.required("RegionYield", (root, 0_i64)).unwrap();
    assert_eq!(
        program.graph.value_to_base::<i64>(query.required("ConstantValue", result).unwrap()),
        7
    );
}

// Parallel emission contracts live in memory_ir_parallel_*.wyn CLI fixtures.
