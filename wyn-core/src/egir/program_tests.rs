//! Tests for the region arena / interner invariants that the opaque
//! `RegionId` representation depends on.

use super::*;
use crate::ast::{Span, TypeName};
use crate::egir::types::{EGraph, RegionId};
use crate::flow::ExecutionModel;
use crate::pipeline_descriptor::PipelineDescriptor;
use polytype::Type;

fn unit_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Unit, vec![])
}

fn empty_func(name: &str) -> SemanticFunc {
    SemanticFunc::new(
        name.to_string(),
        Span::dummy(),
        None,
        vec![],
        unit_ty(),
        EGraph::new(),
        LookupMap::new(),
    )
}

fn empty_entry(name: &str) -> SemanticEntry {
    SemanticEntry::new_with_resources(
        name.to_string(),
        Span::dummy(),
        ExecutionModel::Compute {
            local_size: (1, 1, 1),
        },
        vec![],
        vec![],
        vec![],
        vec![],
        unit_ty(),
        EGraph::new(),
        LookupMap::new(),
    )
}

fn allocated_program(size: LogicalSize) -> AllocatedProgram {
    let binding = crate::BindingRef::new(0, 7);
    let mut program = SemanticProgram::new(
        vec![],
        vec![],
        vec![empty_entry("main")],
        vec![],
        PipelineDescriptor::default(),
        RegionInterner::default(),
    );
    let resource = program.resources.allocate(ResourceOrigin::Host(binding), unit_ty(), size);
    let resource_size = program.resources[resource].size.clone();
    program.entry_points[0].resource_declarations.push(SemanticResourceDecl {
        resource: SemanticResourceRef(resource),
        role: crate::interface::StorageRole::Input,
        elem_ty: unit_ty(),
        size: resource_size,
    });
    AllocatedProgram {
        semantic: program,
        materializations: crate::IdArena::new(),
    }
}

#[test]
fn logical_allocation_introduces_the_allocated_sidecar() {
    let binding = crate::BindingRef::new(2, 3);
    let mut semantic = SemanticProgram::new(
        vec![],
        vec![],
        vec![],
        vec![],
        PipelineDescriptor::default(),
        RegionInterner::default(),
    );
    semantic.resources.allocate(ResourceOrigin::Host(binding), unit_ty(), LogicalSize::Unspecified);

    let mut effect_ids = crate::IdSource::new();
    let allocated = plan_logical_resources(semantic, &mut effect_ids).expect("logical resource planning");

    assert!(allocated.materializations.is_empty());
    assert_eq!(allocated.resources.len(), 1);
    assert_eq!(allocated.resources[0].host_binding(), Some(binding));
    assert!(allocated.semantic_dependencies.is_empty());
}

#[test]
fn semantic_entry_identity_is_stable_and_reused_by_flow_endpoints() {
    let mut program = SemanticProgram::new(
        vec![],
        vec![],
        vec![empty_entry("first"), empty_entry("second")],
        vec![],
        PipelineDescriptor::default(),
        RegionInterner::default(),
    );

    let before = program.entry_ids().collect::<Vec<_>>();
    program.entry_points[0].name = "renamed".into();
    let after = program.entry_ids().collect::<Vec<_>>();
    assert_ne!(before[0], before[1]);
    assert_eq!(
        after, before,
        "entry optimization must not remint semantic identity"
    );
    let allocated = AllocatedProgram {
        semantic: program,
        materializations: crate::IdArena::new(),
    };
    let entries = allocated
        .entries_with_endpoints()
        .map(|(endpoint, entry)| {
            let CompilerFlowEndpoint::Entry(id) = endpoint else {
                unreachable!("program has no materializations")
            };
            (id, entry.name.as_str())
        })
        .collect::<Vec<_>>();
    assert_eq!(entries, vec![(before[0], "renamed"), (before[1], "second")]);
}

#[test]
fn allocated_resource_verifier_accepts_resource_only_program() {
    let program = allocated_program(LogicalSize::Unspecified);
    verify_allocated_resources(&program).expect("resource-normalized program");
}

#[test]
fn semantic_resource_ref_has_no_binding_constructor() {
    let mut resources = LogicalResourceArena::default();
    let resource = resources.allocate(
        ResourceOrigin::Host(crate::BindingRef::new(0, 0)),
        unit_ty(),
        LogicalSize::Unspecified,
    );
    let reference = SemanticResourceRef(resource);
    assert_eq!(reference.0, resource);
}

#[test]
fn logical_resource_arena_owns_dense_identity_assignment() {
    let mut resources = LogicalResourceArena::default();
    let first = resources.allocate(
        ResourceOrigin::Host(crate::BindingRef::new(0, 1)),
        unit_ty(),
        LogicalSize::Unspecified,
    );
    let second = resources.allocate(
        ResourceOrigin::Compiler(CompilerResource::new(
            CompilerResourceKind::ReducePartial,
            Some(SemanticOpId::for_test(7)),
            0,
        )),
        unit_ty(),
        LogicalSize::FixedBytes(4),
    );

    assert_eq!(first.index(), 0);
    assert_eq!(second.index(), 1);
    assert_eq!(resources[first].id(), first);
    assert_eq!(resources[second].id(), second);
}

#[test]
fn physicalization_rebuilds_resource_nodes_as_binding_nodes() {
    let binding = crate::BindingRef::new(3, 5);
    let mut resources = LogicalResourceArena::default();
    let resource = resources.allocate(
        ResourceOrigin::Host(binding),
        Type::Constructed(TypeName::UInt(32), vec![]),
        LogicalSize::Unspecified,
    );
    let table = PhysicalResourceTable::allocate(&resources, &mut crate::IdSource::new());
    let mut graph = EGraph::new();
    let view = crate::egir::graph_ops::intern_resource_view(
        &mut graph,
        resource,
        Type::Constructed(TypeName::UInt(32), vec![]),
        None,
    );

    let (physical, node_map, _) =
        physicalize_graph_resources(graph, &table).expect("resource graph should physicalize");
    let mapped_view = node_map[&view];
    assert!(matches!(
        &physical.nodes[mapped_view],
        crate::egir::types::ENode::Pure {
            op: crate::egir::types::PureOp::StorageView(crate::op::PureViewSource::Storage(found)),
            ..
        } if *found == binding
    ));
    assert!(physical.nodes.values().all(|node| !matches!(
        node,
        crate::egir::types::ENode::Pure {
            op: crate::egir::types::PureOp::ResourceLen(_),
            ..
        }
    )));
}

#[test]
fn allocated_resource_verifier_rejects_missing_size_source() {
    let program = allocated_program(LogicalSize::LikeResource {
        resource: ResourceId::for_test(1),
        elem_bytes: 4,
        src_elem_bytes: 4,
    });
    let error = verify_allocated_resources(&program).expect_err("missing size source must be rejected");
    assert!(error.contains("missing source"), "{error}");
}

#[test]
fn interner_assigns_dense_indices_and_deduplicates_names() {
    let mut interner = RegionInterner::default();
    let foo = interner.intern("foo");
    let bar = interner.intern("bar");
    assert_eq!(
        foo,
        interner.intern("foo"),
        "same name re-interns to the same index"
    );
    assert_ne!(foo, bar);
    assert_eq!(foo.index(), 0);
    assert_eq!(bar.index(), 1);
    assert_eq!(interner.resolve(foo), "foo");
    assert_eq!(interner.resolve(bar), "bar");
    assert_eq!(interner.get("foo"), Some(foo));
    assert_eq!(interner.get("absent"), None);
}

/// Correctness risk #1 — index agreement. A `SegBody` built during TLC→EGIR
/// conversion interns its callee *before* `SemanticProgram::new` walks the function
/// list, and the callee may appear later in that list than other functions.
/// The arena entry for the callee must still land on the index the `SegBody`
/// already holds.
#[test]
fn segbody_index_resolves_to_its_arena_function() {
    // `op` is referenced (interned) first, as a SegBody operator region would
    // be, even though it is the *second* function in the program.
    let mut interner = RegionInterner::default();
    let op_id: RegionId = interner.intern("op");

    let inner = SemanticProgram::new(
        vec![empty_func("main"), empty_func("op")],
        vec![],
        vec![],
        vec![],
        PipelineDescriptor::default(),
        interner,
    );

    // The pre-interned index is preserved and points at the right body.
    assert_eq!(inner.region_interner.get("op"), Some(op_id));
    assert!(inner.contains_region(op_id), "callee region must be in the arena");
    assert_eq!(inner.region_name(op_id), "op");
    assert_eq!(inner.region(op_id).unwrap().name, "op");
    // `main` was assigned its index by `SemanticProgram::new`, after `op`.
    let main_id = inner.region_interner.get("main").expect("main interned");
    assert_ne!(main_id, op_id);
    assert_eq!(inner.region_name(main_id), "main");
}

/// A synthesized region's index is its interned name, so a caller that reserved
/// the index before building the body — as vertical fusion does, to reference it
/// from the `SegBody` it is composing — gets that same index back from
/// `define_region`, and the body lands under it.
#[test]
fn define_region_records_the_body_under_the_reserved_index() {
    let mut inner = SemanticProgram::new(
        vec![empty_func("main")],
        vec![],
        vec![],
        vec![],
        PipelineDescriptor::default(),
        RegionInterner::default(),
    );

    let reserved = inner.intern_region("composed");
    assert!(
        !inner.contains_region(reserved),
        "reserving an index must not invent a body"
    );

    let defined = inner.define_region(empty_func("composed"));

    assert_eq!(defined, reserved, "define_region honors the reserved index");
    assert_eq!(inner.region(defined).unwrap().name, "composed");
    assert!(
        inner.functions.iter().any(|function| function.name == "composed"),
        "the region is callable"
    );
}
