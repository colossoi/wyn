use std::convert::Infallible;

use super::super::{Decl, ExprKind, Expression, Header, Node, NodeId, Program, SourceTree, Span};
use crate::resolve_imports::{self, ImportsResolvedFamily};
use crate::test_pipeline;
use crate::types::Diet;
use crate::{Compiler, CompilerOptions};

#[derive(Debug)]
enum Before {}

#[derive(Debug)]
enum After {}

fn unit(id: u32) -> Expression {
    Node {
        h: Header {
            id: NodeId(id),
            span: Span::generated(),
        },
        kind: ExprKind::Unit,
    }
}

#[test]
fn program_rebuild_preserves_the_shared_node_allocator() {
    let compiler = Compiler::new(CompilerOptions::default()).expect("compiler should initialize");
    let modules = test_pipeline::load_test_modules("", compiler);
    let root = modules.graph.root();
    let program = resolve_imports::resolve_imports(modules).expect("imports should resolve");
    let program = program.map_global_context::<Before, _>(|_| 42usize);
    let next_node_id = program.node_ids.peek_id();

    let rebuilt: Program<After, ImportsResolvedFamily, String> = program
        .try_rebuild(|declarations, global_context, node_ids| {
            assert_eq!(node_ids.next_id(), next_node_id);
            Ok::<_, Infallible>((declarations, global_context.to_string()))
        })
        .unwrap();

    assert_eq!(rebuilt.global_context, "42");
    assert_eq!(rebuilt.node_ids.peek_id(), NodeId(next_node_id.0 + 1));
    assert_eq!(rebuilt.source_graph().source(root), Some(""));
}

#[test]
fn declaration_rebuild_carries_signature_fields() {
    let span = Span::generated();
    let definition = Decl::<u8, SourceTree> {
        data: 7,
        name: "f".to_owned(),
        name_span: span,
        size_params: vec!["n".to_owned()],
        type_params: vec!["t".to_owned()],
        params: Vec::new(),
        ty: None,
        body: unit(3),
        param_diets: vec![Diet::Leaf(true)],
        return_diet: Diet::observing(),
    };

    let rebuilt: Decl<String, SourceTree> = definition
        .try_rebuild(
            |data, name, name_span| {
                assert_eq!(name, "f");
                assert_eq!(name_span, span);
                Ok::<_, Infallible>(data.to_string())
            },
            |params, body| Ok((params, body)),
        )
        .unwrap();

    assert_eq!(rebuilt.data, "7");
    assert_eq!(rebuilt.name, "f");
    assert_eq!(rebuilt.name_span, span);
    assert_eq!(rebuilt.size_params, ["n"]);
    assert_eq!(rebuilt.type_params, ["t"]);
    assert_eq!(rebuilt.param_diets, [Diet::Leaf(true)]);
    assert_eq!(rebuilt.return_diet, Diet::observing());
    assert_eq!(rebuilt.body.h.id, NodeId(3));
}
