#![allow(clippy::expect_used, clippy::unwrap_used)]

use super::*;
use crate::egir::program::{LogicalResourceArena, ResourceOrigin, SemanticEntryId};

fn resource(
    resources: &mut LogicalResourceArena,
    owner: u32,
    kind: CompilerResourceKind,
    slot: usize,
) -> crate::ResourceId {
    resources.allocate(
        ResourceOrigin::Compiler(CompilerResource::new(
            kind,
            Some(SemanticOpId::for_test(owner)),
            slot,
        )),
        Type::Constructed(TypeName::UInt(32), vec![]),
        LogicalSize::FixedBytes(4),
    )
}

#[test]
fn resource_arena_interns_compiler_ownership_keys() {
    let mut resources = LogicalResourceArena::default();
    let slot_one = resource(&mut resources, 7, CompilerResourceKind::ReducePartial, 1);
    let slot_zero = resource(&mut resources, 7, CompilerResourceKind::ReducePartial, 0);
    let duplicate = resource(&mut resources, 7, CompilerResourceKind::ReducePartial, 0);
    let scan = resource(&mut resources, 8, CompilerResourceKind::ScanBlockSums, 0);

    assert_eq!(duplicate, slot_zero);
    assert_eq!(resources.len(), 3);
    assert_eq!(
        resources.compiler_resource(SemanticOpId::for_test(7), CompilerResourceKind::ReducePartial, 0,),
        Some(slot_zero)
    );
    assert_eq!(
        resources.compiler_resource(SemanticOpId::for_test(7), CompilerResourceKind::ReducePartial, 1,),
        Some(slot_one)
    );
    assert_eq!(
        resources.compiler_resource(SemanticOpId::for_test(8), CompilerResourceKind::ScanBlockSums, 0,),
        Some(scan)
    );
    assert_eq!(
        resources.compiler_resource(SemanticOpId::for_test(9), CompilerResourceKind::ScanBlockSums, 0,),
        None
    );
}

#[test]
fn serial_recipe_index_carries_no_parallel_recipe_state() {
    let mut serial = RecipeIndex::serial();
    assert!(serial
        .take_endpoint(CompilerFlowEndpoint::Entry(SemanticEntryId::for_test(0)))
        .expect("serial endpoint lookup")
        .is_none());
}
