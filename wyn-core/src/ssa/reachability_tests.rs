use super::{retain_reachable, Definition};

use crate::ast::{Span, TypeName};
use crate::flow::ExecutionModel;
use crate::op::OpTag;
use crate::ssa::builder::FuncBuilder;
use crate::ssa::types::{Constant, EntryPoint, Function, InstKind, Program, Terminator};
use crate::{EntryId, FunctionId, GlobalId, LookupMap};
use polytype::Type;

fn unit_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Unit, vec![])
}

fn body(references: &[Definition]) -> crate::ssa::types::FuncBody {
    let mut builder = FuncBuilder::new(vec![], unit_ty());
    for reference in references {
        let tag = match reference {
            Definition::Function(id) => OpTag::Call(*id),
            Definition::Constant(id) => OpTag::Global(*id),
        };
        builder
            .push_inst(
                InstKind::Op {
                    tag,
                    operands: vec![],
                },
                unit_ty(),
            )
            .unwrap();
    }
    builder.terminate(Terminator::Return(None)).unwrap();
    builder.finish().unwrap()
}

fn function(id: FunctionId, name: &str, references: &[Definition]) -> Function {
    Function {
        id,
        name: name.into(),
        body: body(references),
        span: Span::dummy(),
        linkage_name: None,
    }
}

fn constant(id: GlobalId, name: &str, references: &[Definition]) -> Constant {
    Constant {
        id,
        name: name.into(),
        body: body(references),
    }
}

fn entry(references: &[Definition]) -> EntryPoint {
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
        pipeline_storage_accesses: LookupMap::new(),
        span: Span::dummy(),
    }
}

#[test]
fn follows_calls_and_globals_to_a_fixpoint_while_preserving_order() {
    let dead = FunctionId::from_index(0);
    let leaf = FunctionId::from_index(1);
    let root = FunctionId::from_index(2);
    let from_constant = FunctionId::from_index(3);
    let dead_constant = GlobalId::from(0);
    let nested_constant = GlobalId::from(1);
    let live_constant = GlobalId::from(2);

    let program = Program::bare(
        vec![
            function(dead, "dead", &[]),
            // Mutual recursion also terminates at the visited set.
            function(leaf, "leaf", &[Definition::Function(root)]),
            function(root, "root", &[Definition::Function(leaf)]),
            function(from_constant, "from_constant", &[]),
        ],
        vec![entry(&[
            Definition::Function(root),
            Definition::Constant(live_constant),
        ])],
        vec![
            constant(dead_constant, "dead_constant", &[Definition::Function(dead)]),
            constant(nested_constant, "nested_constant", &[Definition::Function(leaf)]),
            constant(
                live_constant,
                "live_constant",
                &[
                    Definition::Function(from_constant),
                    Definition::Constant(nested_constant),
                ],
            ),
        ],
    );

    let reachable = retain_reachable(program);
    assert_eq!(
        reachable.functions.iter().map(|function| function.name.as_str()).collect::<Vec<_>>(),
        ["leaf", "root", "from_constant"]
    );
    assert_eq!(
        reachable.constants.iter().map(|constant| constant.name.as_str()).collect::<Vec<_>>(),
        ["nested_constant", "live_constant"]
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
    let program = Program::bare(vec![function(dead, "dead", &[])], vec![main], vec![]);

    let reachable = retain_reachable(program);
    assert!(reachable.functions.is_empty());
}
