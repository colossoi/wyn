use super::retain_reachable;
use crate::ssa;

use crate::ast::{Span, TypeName};
use crate::flow::ExecutionModel;
use crate::op::OpTag;
use crate::ssa::builder::FuncBuilder;
use crate::ssa::types::{EntryPoint, Function, InstKind, Program, Terminator};
use crate::{EntryId, FunctionId, LookupMap};
use polytype::Type;

fn unit_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Unit, vec![])
}

fn body(references: &[FunctionId]) -> ssa::types::FuncBody {
    let mut builder = FuncBuilder::new(vec![], unit_ty());
    for reference in references {
        builder
            .push_inst(
                InstKind::Op {
                    tag: OpTag::Call(*reference),
                    operands: vec![],
                },
                unit_ty(),
            )
            .unwrap();
    }
    builder.terminate(Terminator::Return(None)).unwrap();
    builder.finish().unwrap()
}

fn function(id: FunctionId, name: &str, references: &[FunctionId]) -> Function {
    Function {
        id,
        name: name.into(),
        body: body(references),
        span: Span::generated(),
        linkage_name: None,
    }
}

fn entry(references: &[FunctionId]) -> EntryPoint {
    EntryPoint {
        id: EntryId::from_index(0),
        name: "main".into(),
        body: body(references),
        execution_model: ExecutionModel::Compute {
            local_size: (1, 1, 1),
        },
        inputs: vec![],
        parameter_inputs: vec![],
        outputs: vec![],
        storage_bindings: vec![],
        stage_descriptor_storage_accesses: LookupMap::new(),
        pipeline_storage_accesses: LookupMap::new(),
        span: Span::generated(),
    }
}

#[test]
fn follows_calls_to_a_fixpoint_while_preserving_order() {
    let dead = FunctionId::from_index(0);
    let leaf = FunctionId::from_index(1);
    let root = FunctionId::from_index(2);
    let other = FunctionId::from_index(3);
    let program = Program::bare(
        vec![
            function(dead, "dead", &[]),
            // Mutual recursion also terminates at the visited set.
            function(leaf, "leaf", &[root, other]),
            function(root, "root", &[leaf]),
            function(other, "other", &[]),
        ],
        vec![entry(&[root])],
    );
    let reachable = retain_reachable(program);
    assert_eq!(
        reachable.functions.iter().map(|function| function.name.as_str()).collect::<Vec<_>>(),
        ["leaf", "root", "other"]
    );
}

#[test]
fn ignores_definition_references_in_disconnected_blocks() {
    let dead = FunctionId::from_index(0);
    let mut builder = FuncBuilder::new(vec![], unit_ty());
    builder.terminate(Terminator::Return(None)).unwrap();
    let disconnected = builder.create_block();
    builder.switch_to_block(disconnected).unwrap();
    builder
        .push_inst(
            InstKind::Op {
                tag: OpTag::Call(dead),
                operands: vec![],
            },
            unit_ty(),
        )
        .unwrap();
    builder.terminate(Terminator::Return(None)).unwrap();

    let mut main = entry(&[]);
    main.body = builder.finish().unwrap();
    let program = Program::bare(vec![function(dead, "dead", &[])], vec![main]);

    let reachable = retain_reachable(program);
    assert!(reachable.functions.is_empty());
}
