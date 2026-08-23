//! Tests for stable program identity arenas and resource planning invariants.

use super::*;
use crate::ast::{Span, TypeName};
use crate::egir;
use crate::egir::allocation::{plan_logical_resources, verify_allocated_resources, ResourcesAllocated};
use crate::egir::types::{by_value_function_result, CallEffects, EGraph, Parameters, WynLanguage};
use crate::flow::ExecutionModel;
use crate::interface;
use crate::op;
use crate::pipeline_descriptor::{BufferLen, PipelineDescriptor};
use crate::BindingRef;
use crate::EntryId;
use crate::IdSource;
use polytype::Type;

fn unit_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Unit, vec![])
}

fn empty_func(id: FunctionId, name: &str) -> Func {
    Func::<Semantic>::new(
        id,
        name.to_string(),
        Span::dummy(),
        None,
        Parameters::new(),
        by_value_function_result::<WynLanguage>(unit_ty()),
        CallEffects::Pure,
        EGraph::new(),
    )
}

fn empty_entry(id: EntryId, name: &str) -> Entry {
    Entry::<Semantic>::new_with_resources(
        name.to_string(),
        id,
        Span::dummy(),
        ExecutionModel::Compute {
            local_size: (1, 1, 1),
        },
        vec![],
        vec![],
        vec![],
        Parameters::new(),
        by_value_function_result::<WynLanguage>(unit_ty()),
        EGraph::new(),
    )
}

fn storage_input(
    name: &str,
    binding: BindingRef,
    elem_ty: Type<TypeName>,
    length: Option<BufferLen>,
) -> egir::ir::EntryInput<BindingRef, WynLanguage> {
    egir::ir::EntryInput {
        inner: interface::EntryInput {
            name: name.into(),
            ty: elem_ty,
            size_hint: None,
            kind: interface::EntryInputKind::Storage {
                exposure: interface::BindingExposure::Host(binding),
                access: interface::StorageAccess::ReadOnly,
                length,
            },
        },
        resource: Some(binding),
    }
}

fn into_allocated(program: egir::reify::Segmented) -> ResourcesAllocated {
    plan_logical_resources(program.retag()).expect("allocate test program")
}

fn allocated_program(size: LogicalSize) -> ResourcesAllocated {
    let binding = BindingRef::new(0, 7);
    let mut identities = ProgramIdentities::default();
    let main = identities.alloc_entry("main".into());
    let program = semantic_program_for_test(
        vec![],
        vec![],
        vec![empty_entry(main, "main")],
        vec![],
        PipelineDescriptor::default(),
        identities,
    );
    let mut program = into_allocated(program);
    let resource = program.data.core.resources.allocate(ResourceOrigin::host(binding), unit_ty(), size);
    let stage = program.data.stages.stages().next().expect("allocated stage").0;
    program.data.stages.stage_body_mut(stage).expect("allocated entry").resource_declarations.push(
        SemanticResourceDecl {
            resource: SemanticResourceRef(resource),
            role: interface::StorageRole::Input,
        },
    );
    program
}

#[test]
fn logical_allocation_introduces_the_allocated_sidecar() {
    let binding = BindingRef::new(2, 3);
    let mut entry = empty_entry(EntryId::from_index(0), "main");
    entry.inputs.push(egir::ir::EntryInput {
        inner: interface::EntryInput {
            name: "input".into(),
            ty: unit_ty(),
            size_hint: None,
            kind: interface::EntryInputKind::Storage {
                exposure: interface::BindingExposure::Host(binding),
                access: interface::StorageAccess::ReadOnly,
                length: None,
            },
        },
        resource: Some(binding),
    });
    let semantic = semantic_program_for_test(
        vec![],
        vec![],
        vec![entry],
        vec![],
        PipelineDescriptor::default(),
        ProgramIdentities::default(),
    );
    let allocated = plan_logical_resources(semantic.retag()).expect("logical resource planning");

    assert_eq!(allocated.data.stages.stages().len(), 1);
    assert_eq!(allocated.data.stages.flows().len(), 0);
    assert_eq!(allocated.data.core.resources.len(), 1);
    assert_eq!(allocated.data.core.resources[0].host_binding(), Some(binding));
}

#[test]
fn host_size_policy_can_reference_a_later_interface_binding() {
    let target_binding = BindingRef::new(1, 0);
    let source_binding = BindingRef::new(1, 1);
    let mut entry = empty_entry(EntryId::from_index(0), "main");
    entry.inputs.push(storage_input(
        "target",
        target_binding,
        unit_ty(),
        Some(BufferLen::LikeInput {
            set: source_binding.set,
            binding: source_binding.binding,
            elem_bytes: 4,
            src_elem_bytes: 8,
        }),
    ));
    entry.inputs.push(storage_input(
        "source",
        source_binding,
        unit_ty(),
        Some(BufferLen::Fixed { bytes: 32 }),
    ));
    let semantic = semantic_program_for_test(
        vec![],
        vec![],
        vec![entry],
        vec![],
        PipelineDescriptor::default(),
        ProgramIdentities::default(),
    );

    let allocated = plan_logical_resources(semantic.retag()).expect("forward LikeInput reference");
    let target = allocated.data.core.resources.host_resource(target_binding).unwrap();
    let source = allocated.data.core.resources.host_resource(source_binding).unwrap();
    assert_eq!(
        allocated.data.core.resources[target].size,
        LogicalSize::LikeResource {
            resource: source,
            elem_bytes: 4,
            src_elem_bytes: 8,
        }
    );
}

#[test]
fn host_size_policy_rejects_a_reference_to_an_unreserved_binding() {
    let target_binding = BindingRef::new(1, 0);
    let missing_binding = BindingRef::new(1, 9);
    let mut entry = empty_entry(EntryId::from_index(0), "main");
    entry.inputs.push(storage_input(
        "target",
        target_binding,
        unit_ty(),
        Some(BufferLen::LikeInput {
            set: missing_binding.set,
            binding: missing_binding.binding,
            elem_bytes: 4,
            src_elem_bytes: 4,
        }),
    ));
    let semantic = semantic_program_for_test(
        vec![],
        vec![],
        vec![entry],
        vec![],
        PipelineDescriptor::default(),
        ProgramIdentities::default(),
    );

    let error = plan_logical_resources(semantic.retag()).expect_err("missing size source must fail");
    assert!(error.to_string().contains("is not declared"), "{error}");
}

#[test]
fn repeated_compatible_host_declarations_share_an_identity() {
    let binding = BindingRef::new(1, 0);
    let mut entry = empty_entry(EntryId::from_index(0), "main");
    entry.inputs.push(storage_input(
        "first",
        binding,
        unit_ty(),
        Some(BufferLen::Fixed { bytes: 16 }),
    ));
    entry.inputs.push(storage_input(
        "second",
        binding,
        unit_ty(),
        Some(BufferLen::Fixed { bytes: 16 }),
    ));
    let semantic = semantic_program_for_test(
        vec![],
        vec![],
        vec![entry],
        vec![],
        PipelineDescriptor::default(),
        ProgramIdentities::default(),
    );

    let allocated = plan_logical_resources(semantic.retag()).expect("compatible declarations");
    assert_eq!(allocated.data.core.resources.len(), 1);
    assert_eq!(allocated.data.core.resources[0].size, LogicalSize::FixedBytes(16));
}

#[test]
fn conflicting_host_element_types_are_rejected() {
    let binding = BindingRef::new(1, 0);
    let mut entry = empty_entry(EntryId::from_index(0), "main");
    entry.inputs.push(storage_input(
        "first",
        binding,
        Type::Constructed(TypeName::UInt(32), vec![]),
        None,
    ));
    entry.inputs.push(storage_input(
        "second",
        binding,
        Type::Constructed(TypeName::Int(32), vec![]),
        None,
    ));
    let semantic = semantic_program_for_test(
        vec![],
        vec![],
        vec![entry],
        vec![],
        PipelineDescriptor::default(),
        ProgramIdentities::default(),
    );

    let error = plan_logical_resources(semantic.retag()).expect_err("element types must conflict");
    assert!(error.to_string().contains("conflicting element types"), "{error}");
}

#[test]
fn conflicting_host_size_policies_are_rejected() {
    let binding = BindingRef::new(1, 0);
    let mut entry = empty_entry(EntryId::from_index(0), "main");
    entry.inputs.push(storage_input(
        "first",
        binding,
        unit_ty(),
        Some(BufferLen::Fixed { bytes: 16 }),
    ));
    entry.inputs.push(storage_input(
        "second",
        binding,
        unit_ty(),
        Some(BufferLen::Fixed { bytes: 32 }),
    ));
    let semantic = semantic_program_for_test(
        vec![],
        vec![],
        vec![entry],
        vec![],
        PipelineDescriptor::default(),
        ProgramIdentities::default(),
    );

    let error = plan_logical_resources(semantic.retag()).expect_err("size policies must conflict");
    assert!(error.to_string().contains("conflicting size policies"), "{error}");
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
    let entries = allocated
        .data
        .stages
        .stages()
        .map(|(_, stage)| (stage.body().id, stage.body().name.as_str()))
        .collect::<Vec<_>>();
    assert_eq!(entries, vec![(before[0], "renamed"), (before[1], "second")]);
}

#[test]
fn allocated_resource_verifier_accepts_resource_only_program() {
    let program = allocated_program(LogicalSize::Unspecified);
    verify_allocated_resources(&program).expect("resource-normalized program");
}

#[test]
fn entry_publication_reads_type_and_size_from_resource_arena() {
    let mut program = allocated_program(LogicalSize::FixedBytes(12));
    let resource = program.data.core.resources.allocate(
        ResourceOrigin::Compiler(CompilerResource::new(
            CompilerResourceKind::ScalarHandoff,
            None,
            0,
        )),
        unit_ty(),
        LogicalSize::FixedBytes(12),
    );
    let stage = program.data.stages.stages().next().expect("allocated stage").0;
    program.data.stages.stage_body_mut(stage).unwrap().resource_declarations[0].resource =
        SemanticResourceRef(resource);
    let physical = PhysicalResourceTable::allocate(&program.data.core.resources, &mut IdSource::new());

    let publication = program
        .data
        .stages
        .stage(stage)
        .unwrap()
        .body()
        .publication(&physical)
        .expect("publish allocated entry");
    let [binding] = publication.storage_bindings.as_slice() else {
        panic!("expected one compiler-owned storage declaration")
    };
    assert_eq!(binding.elem_ty, unit_ty());
    assert_eq!(
        binding.length,
        Some(crate::pipeline_descriptor::BufferLen::Fixed { bytes: 12 })
    );
}

#[test]
fn semantic_resource_ref_has_no_binding_constructor() {
    let mut resources = LogicalResourceArena::default();
    let resource = resources.allocate(
        ResourceOrigin::host(BindingRef::new(0, 0)),
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
        ResourceOrigin::host(BindingRef::new(0, 1)),
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
    let binding = BindingRef::new(3, 5);
    let mut resources = LogicalResourceArena::default();
    let resource = resources.allocate(
        ResourceOrigin::host(binding),
        Type::Constructed(TypeName::UInt(32), vec![]),
        LogicalSize::Unspecified,
    );
    let table = PhysicalResourceTable::allocate(&resources, &mut IdSource::new());
    let mut graph = EGraph::new();
    let view = egir::graph_ops::intern_resource_view(
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
        egir::types::ValueKind::Pure {
            op: egir::types::PureOp::StorageView(op::PureViewSource::Storage(found)),
            ..
        } if *found == binding
    ));
    assert!(physical.nodes.values().all(|node| !matches!(
        &node.kind,
        egir::types::ValueKind::Pure {
            op: egir::types::PureOp::ResourceLen(_),
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
    let mut ids = IdSource::new();
    let table = PhysicalResourceTable::allocate_avoiding(
        &resources,
        &mut ids,
        [BindingRef::new(0, 0), BindingRef::new(0, 2)],
    );

    assert_eq!(table.binding(first), BindingRef::new(0, 1));
    assert_eq!(table.binding(second), BindingRef::new(0, 3));
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
    assert_eq!(inner.data.identities.function_name(op_id), "op");
    assert_eq!(inner.region(op_id).unwrap().name, "op");
    assert_ne!(main_id, op_id);
    assert_eq!(inner.data.identities.function_name(main_id), "main");
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
