use super::*;

fn i32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Int(32), vec![])
}

fn u32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::UInt(32), vec![])
}

fn effect(result: NodeId) -> SideEffect {
    SideEffect {
        kind: SideEffectKind::Effect(EffectOp::ControlBarrier),
        operand_nodes: SmallVec::new(),
        result: Some(result),
        effects: None,
        span: None,
    }
}

#[derive(Clone, Debug)]
struct TestPhase;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum TestConst {
    FortyTwo,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct TestLanguage;

impl Language for TestLanguage {
    type Const = TestConst;
    type Ty = String;
}

impl EgirPhase for TestPhase {
    type Resource = ();
    type ResourceDecl = u16;
    type SoacId = ();
    type MapState = ();
    type ReduceState = ();
    type ScanState = ();
    type CompositeState = ();
    type FilterState = ();
    type HistState = ();
}

#[test]
fn graph_accepts_non_wyn_payloads() {
    let mut graph = super::super::ir::EGraph::<Semantic, TestLanguage>::new();
    let node = graph.intern_pure(PureOp::Unit, SmallVec::new(), "unit".to_string(), None);
    let constant = graph.intern_constant(TestConst::FortyTwo, "number".to_string());

    assert_eq!(graph.types[&node], "unit");
    assert!(matches!(
        graph.nodes[constant],
        super::super::ir::ENode::Constant(TestConst::FortyTwo)
    ));
}

#[test]
fn entry_and_program_accept_non_wyn_resource_metadata() {
    let graph = super::super::ir::EGraph::<TestPhase, TestLanguage>::new();
    let entry = super::super::ir::Entry::<TestPhase, TestLanguage>::new_with_resources(
        "custom".to_string(),
        crate::ast::Span::new(0, 0, 0, 0),
        crate::flow::ExecutionModel::Compute {
            local_size: (1, 1, 1),
        },
        vec![],
        vec![],
        vec![7],
        vec![],
        "unit".to_string(),
        graph,
        crate::LookupMap::new(),
    );
    assert_eq!(entry.resource_declarations, [7]);

    let program = super::super::ir::Program::<TestPhase, TestLanguage>::new(
        vec![],
        vec![],
        vec![entry],
        vec![],
        crate::pipeline_descriptor::PipelineDescriptor::default(),
        super::super::ir::RegionInterner::default(),
    );
    assert_eq!(program.entry_points[0].resource_declarations, [7]);
}

#[test]
fn indexes_results_across_skeleton_blocks() {
    let mut graph: EGraph = EGraph::new();
    let unit = Type::Constructed(TypeName::Unit, vec![]);
    let first = graph.alloc_side_effect_result(unit.clone());
    let second = graph.alloc_side_effect_result(unit);
    let entry = graph.skeleton.entry;
    let other = graph.skeleton.create_block();
    graph.skeleton.blocks[entry].side_effects.push(effect(first));
    graph.skeleton.blocks[other].side_effects.push(effect(second));

    let index = graph.side_effect_index();
    assert_eq!(
        index.site(first),
        Some(SideEffectSite {
            block: entry,
            index: 0
        })
    );
    assert_eq!(
        index.site(second),
        Some(SideEffectSite {
            block: other,
            index: 0
        })
    );
    assert_eq!(
        index.effect(&graph, second).and_then(|effect| effect.result),
        Some(second)
    );
}

#[test]
fn replace_all_references_does_not_leave_stale_hash_cons_key() {
    let mut graph = EGraph::new();
    let int = i32_ty();
    let a = graph.intern_pure(PureOp::Int("1".into()), smallvec::smallvec![], int.clone(), None);
    let b = graph.intern_pure(PureOp::Int("2".into()), smallvec::smallvec![], int.clone(), None);
    let old_call = graph.intern_pure(
        PureOp::Call("__test_hash_cons".into()),
        smallvec::smallvec![a, b],
        int.clone(),
        None,
    );

    crate::egir::graph_ops::replace_all_references(&mut graph, b, a);

    let reinterned_old_call = graph.intern_pure(
        PureOp::Call("__test_hash_cons".into()),
        smallvec::smallvec![a, b],
        int,
        None,
    );

    assert_ne!(old_call, reinterned_old_call);
    assert!(graph.verify_hash_cons().is_ok());
}

#[test]
fn retype_node_does_not_leave_stale_hash_cons_key() {
    let mut graph = EGraph::<Semantic>::new();
    let int = i32_ty();
    let uint = u32_ty();
    let arg = graph.intern_pure(PureOp::Int("1".into()), smallvec::smallvec![], int.clone(), None);
    let old_call = graph.intern_pure(
        PureOp::Call("__test_hash_cons_retype".into()),
        smallvec::smallvec![arg],
        int.clone(),
        None,
    );

    graph.retype_node(old_call, uint);

    let reinterned_old_call = graph.intern_pure(
        PureOp::Call("__test_hash_cons_retype".into()),
        smallvec::smallvec![arg],
        int,
        None,
    );

    assert_ne!(old_call, reinterned_old_call);
    assert!(graph.verify_hash_cons().is_ok());
}
