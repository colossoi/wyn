//! Tests for stable program identity arenas and resource planning invariants.

use super::*;
use crate::ast::{Span, TypeName};
use crate::egir::allocation::{
    entries_with_endpoints, plan_logical_resources, verify_allocated_resources, CompilerFlowEndpoint,
    ResourcesAllocated,
};
use crate::egir::types::{EGraph, RegionId};
use crate::flow::ExecutionModel;
use crate::pipeline_descriptor::PipelineDescriptor;
use polytype::Type;

fn unit_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Unit, vec![])
}

fn empty_func(id: RegionId, name: &str) -> SemanticFunc {
    SemanticFunc::new(
        id,
        name.to_string(),
        Span::dummy(),
        None,
        vec![],
        unit_ty(),
        EGraph::new(),
    )
}

fn empty_entry(id: crate::EntryId, name: &str) -> SemanticEntry {
    SemanticEntry::new_with_resources(
        name.to_string(),
        id,
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
    )
}

fn into_allocated(program: crate::egir::reify::Segmented) -> ResourcesAllocated {
    let Program {
        functions,
        externs,
        entry_points,
        constants,
        data,
        global_context,
        state: _,
    } = program;
    Program::from_parts(
        functions,
        externs,
        entry_points,
        constants,
        AllocatedProgramData {
            core: data,
            materializations: crate::IdArena::new(),
        },
        global_context,
    )
}

fn allocated_program(size: LogicalSize) -> ResourcesAllocated {
    let binding = crate::BindingRef::new(0, 7);
    let mut identities = ProgramIdentities::default();
    let main = identities.alloc_entry("main".into());
    let mut program = semantic_program_for_test(
        vec![],
        vec![],
        vec![empty_entry(main, "main")],
        vec![],
        PipelineDescriptor::default(),
        identities,
    );
    let resource = program.data.resources.allocate(ResourceOrigin::host(binding), unit_ty(), size);
    let resource_size = program.data.resources[resource].size.clone();
    program.entry_points[0].resource_declarations.push(SemanticResourceDecl {
        resource: SemanticResourceRef(resource),
        role: crate::interface::StorageRole::Input,
        elem_ty: unit_ty(),
        size: resource_size,
    });
    into_allocated(program)
}

#[test]
fn logical_allocation_introduces_the_allocated_sidecar() {
    let binding = crate::BindingRef::new(2, 3);
    let mut semantic = semantic_program_for_test(
        vec![],
        vec![],
        vec![],
        vec![],
        PipelineDescriptor::default(),
        ProgramIdentities::default(),
    );
    semantic.data.resources.allocate(ResourceOrigin::host(binding), unit_ty(), LogicalSize::Unspecified);

    let allocated = plan_logical_resources(semantic.retag()).expect("logical resource planning");

    assert!(allocated.data.materializations.is_empty());
    assert_eq!(allocated.data.core.resources.len(), 1);
    assert_eq!(allocated.data.core.resources[0].host_binding(), Some(binding));
}

#[test]
fn semantic_entry_identity_is_stable_and_reused_by_flow_endpoints() {
    let mut identities = ProgramIdentities::default();
    let first = identities.alloc_entry("first".into());
    let second = identities.alloc_entry("second".into());
    let mut program = semantic_program_for_test(
        vec![],
        vec![],
        vec![empty_entry(first, "first"), empty_entry(second, "second")],
        vec![],
        PipelineDescriptor::default(),
        identities,
    );

    let before = program.entry_points.iter().map(|entry| entry.id).collect::<Vec<_>>();
    program.entry_points[0].name = "renamed".into();
    let after = program.entry_points.iter().map(|entry| entry.id).collect::<Vec<_>>();
    assert_ne!(before[0], before[1]);
    assert_eq!(
        after, before,
        "entry optimization must not remint semantic identity"
    );
    let allocated = into_allocated(program);
    let entries = entries_with_endpoints(&allocated)
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
        ResourceOrigin::host(crate::BindingRef::new(0, 0)),
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
        ResourceOrigin::host(crate::BindingRef::new(0, 1)),
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
        ResourceOrigin::host(binding),
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
        &physical.nodes[mapped_view].kind,
        crate::egir::types::ValueKind::Pure {
            op: crate::egir::types::PureOp::StorageView(crate::op::PureViewSource::Storage(found)),
            ..
        } if *found == binding
    ));
    assert!(physical.nodes.values().all(|node| !matches!(
        &node.kind,
        crate::egir::types::ValueKind::Pure {
            op: crate::egir::types::PureOp::ResourceLen(_),
            ..
        }
    )));
}

#[test]
fn compiler_binding_allocation_avoids_non_resource_descriptor_slots() {
    let mut resources = LogicalResourceArena::default();
    let first = resources.allocate(
        ResourceOrigin::Compiler(CompilerResource::new(
            CompilerResourceKind::ScalarHandoff,
            None,
            0,
        )),
        unit_ty(),
        LogicalSize::FixedBytes(4),
    );
    let second = resources.allocate(
        ResourceOrigin::Compiler(CompilerResource::new(
            CompilerResourceKind::ScalarHandoff,
            None,
            1,
        )),
        unit_ty(),
        LogicalSize::FixedBytes(4),
    );
    let mut ids = crate::IdSource::new();
    let table = PhysicalResourceTable::allocate_avoiding(
        &resources,
        &mut ids,
        [crate::BindingRef::new(0, 0), crate::BindingRef::new(0, 2)],
    );

    assert_eq!(table.binding(first), crate::BindingRef::new(0, 1));
    assert_eq!(table.binding(second), crate::BindingRef::new(0, 3));
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
fn function_identity_arena_allocates_distinct_stable_ids() {
    let mut identities = ProgramIdentities::default();
    let foo = identities.alloc_function("foo".into());
    let bar = identities.alloc_function("bar".into());
    let second_foo = identities.alloc_function("foo".into());

    assert_ne!(foo, bar);
    assert_ne!(foo, second_foo, "names are metadata, not identity keys");
    assert_eq!(identities.function_name(foo), "foo");
    assert_eq!(identities.function_name(bar), "bar");
    assert_eq!(identities.function_name(second_foo), "foo");
}

#[test]
fn segbody_identity_selects_its_exact_function() {
    let mut identities = ProgramIdentities::default();
    let op_id = identities.alloc_function("op".into());
    let main_id = identities.alloc_function("main".into());

    let inner = semantic_program_for_test(
        vec![empty_func(main_id, "main"), empty_func(op_id, "op")],
        vec![],
        vec![],
        vec![],
        PipelineDescriptor::default(),
        identities,
    );

    assert!(inner.contains_region(op_id));
    assert_eq!(inner.region_name(op_id), "op");
    assert_eq!(inner.region(op_id).unwrap().name, "op");
    assert_ne!(main_id, op_id);
    assert_eq!(inner.region_name(main_id), "main");
}

#[test]
fn reserved_function_identity_is_retained_by_synthesized_body() {
    let mut identities = ProgramIdentities::default();
    let main_id = identities.alloc_function("main".into());
    let mut inner = semantic_program_for_test(
        vec![empty_func(main_id, "main")],
        vec![],
        vec![],
        vec![],
        PipelineDescriptor::default(),
        identities,
    );

    let reserved = inner.data.identities.alloc_function("composed".into());
    assert!(!inner.contains_region(reserved));

    let composed = empty_func(reserved, "composed");
    inner = inner.extend_functions([composed]);

    assert_eq!(inner.region(reserved).unwrap().name, "composed");
    assert!(inner.functions.iter().any(|function| function.region == reserved));
}
/// Materialization IDs make `{source}_{role}_{id}` unique among generated
/// materializations, but an authored entry can still use that exact spelling.
/// Replace this marker with a real collision test when entry-name allocation
/// either reserves a compiler namespace or checks all authored/generated names.
#[test]
#[ignore = "generated materialization names can still collide with authored entry names"]
fn generated_materialization_names_cannot_collide_with_authored_entries() {
    assert!(false);
}
