//! Tests for the region arena / interner invariants that the opaque
//! `RegionId` representation depends on.

use super::*;
use crate::ast::{Span, TypeName};
use crate::egir::types::{EGraph, RegionId};
use crate::pipeline_descriptor::PipelineDescriptor;
use crate::ssa::types::ExecutionModel;
use polytype::Type;

fn unit_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Unit, vec![])
}

fn empty_func(name: &str) -> EgirFunc {
    EgirFunc::new(
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

fn allocated_program(reference: SemanticResourceRef, size: LogicalSize) -> SemanticProgram {
    let binding = crate::BindingRef::new(0, 7);
    let mut entry = empty_entry("main");
    entry.resource_declarations.push(SemanticResourceDecl {
        resource: reference,
        role: crate::interface::StorageRole::Input,
        elem_ty: unit_ty(),
        size: size.clone(),
    });
    let mut program = SemanticProgram::new(
        vec![],
        vec![],
        vec![entry],
        vec![],
        PipelineDescriptor::default(),
        RegionInterner::default(),
    );
    program.resources.push(LogicalResource {
        id: ResourceId(0),
        origin: ResourceOrigin::Host(binding),
        elem_ty: unit_ty(),
        size,
    });
    program
}

#[test]
fn allocated_resource_verifier_accepts_resource_only_program() {
    let program = allocated_program(SemanticResourceRef(ResourceId(0)), LogicalSize::Unspecified);
    verify_allocated_resources(&program).expect("resource-normalized program");
}

#[test]
fn semantic_resource_ref_has_no_binding_constructor() {
    let reference = SemanticResourceRef(ResourceId(0));
    assert_eq!(reference.resource(), Some(ResourceId(0)));
}

#[test]
fn allocated_resource_verifier_rejects_missing_size_source() {
    let program = allocated_program(
        SemanticResourceRef(ResourceId(0)),
        LogicalSize::LikeResource {
            resource: ResourceId(1),
            elem_bytes: 4,
            src_elem_bytes: 4,
        },
    );
    let error = verify_allocated_resources(&program).expect_err("missing size source must be rejected");
    assert!(error.contains("missing source"), "{error}");
}

#[test]
fn scalar_handoff_classification_uses_typed_prepass_role_not_name() {
    let typed_binding = crate::BindingRef::new(0, 10);
    let misleading_binding = crate::BindingRef::new(0, 11);
    let declaration = |resource, role| SemanticResourceDecl {
        resource: SemanticResourceRef(resource),
        role,
        elem_ty: unit_ty(),
        size: LogicalSize::Unspecified,
    };

    let mut typed = empty_entry("renamed_without_magic_marker");
    typed.resource_declarations.push(declaration(ResourceId(0), crate::interface::StorageRole::Output));

    let mut misleading = empty_entry("user_prepass_name");
    misleading
        .resource_declarations
        .push(declaration(ResourceId(1), crate::interface::StorageRole::Output));

    let mut consumer = empty_entry("consumer");
    consumer
        .resource_declarations
        .push(declaration(ResourceId(0), crate::interface::StorageRole::Input));
    consumer
        .resource_declarations
        .push(declaration(ResourceId(1), crate::interface::StorageRole::Input));

    let mut inner = SemanticProgram::new(
        vec![],
        vec![],
        vec![typed, misleading, consumer],
        vec![],
        PipelineDescriptor::default(),
        RegionInterner::default(),
    );
    inner.resources = vec![
        LogicalResource {
            id: ResourceId(0),
            origin: ResourceOrigin::Host(typed_binding),
            elem_ty: unit_ty(),
            size: LogicalSize::Unspecified,
        },
        LogicalResource {
            id: ResourceId(1),
            origin: ResourceOrigin::Host(misleading_binding),
            elem_ty: unit_ty(),
            size: LogicalSize::Unspecified,
        },
    ];
    inner.prepass_roles.insert("renamed_without_magic_marker".into(), PrepassKind::Scalar);
    plan_logical_resources(&mut inner);
    let typed_resource = inner.prepasses[0].body.resource_declarations[0].resource.resource().unwrap();
    let misleading_resource = inner
        .entry_points
        .iter()
        .find(|entry| entry.name == "user_prepass_name")
        .unwrap()
        .resource_declarations[0]
        .resource
        .resource()
        .unwrap();
    assert!(matches!(
        &inner.resources[typed_resource.0 as usize].origin,
        ResourceOrigin::Compiler(CompilerResource {
            kind: CompilerResourceKind::ScalarHandoff,
            ..
        })
    ));
    assert!(matches!(
        inner.resources[misleading_resource.0 as usize].origin,
        ResourceOrigin::Host(_)
    ));
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
    assert_eq!(interner.name(foo), "foo");
    assert_eq!(interner.name(bar), "bar");
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
    assert!(
        inner.regions.contains_key(&op_id),
        "callee region must be in the arena"
    );
    assert_eq!(inner.region_name(op_id), "op");
    assert_eq!(inner.regions.get(&op_id).unwrap().name, "op");
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
        !inner.regions.contains_key(&reserved),
        "reserving an index must not invent a body"
    );

    let defined = inner.define_region(empty_func("composed"));

    assert_eq!(defined, reserved, "define_region honors the reserved index");
    assert_eq!(inner.regions.get(&defined).unwrap().name, "composed");
    assert!(
        inner.functions.iter().any(|function| function.name == "composed"),
        "the region is callable"
    );
}
