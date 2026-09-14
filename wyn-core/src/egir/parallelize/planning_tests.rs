#![allow(clippy::expect_used, clippy::unwrap_used)]

use super::*;
use crate::egir::program::LogicalResourceArena;
use crate::ResourceId;

fn resource(
    resources: &mut LogicalResourceArena,
    owner: u32,
    kind: CompilerResourceKind,
    slot: usize,
) -> ResourceId {
    resources.allocate_compiler(
        CompilerResource::new(kind, Some(SemanticOpId::for_test(owner)), slot),
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
fn scratch_binding_rejects_conflicting_ownership() {
    let draft = crate::integration_tests::compile_to_residency(
        "entry sum(xs: []i32) i32 = reduce(|a, b| a + b, 0, xs)",
    );
    let mut planned = crate::egir::finalize_staged_ir(draft, crate::LoweringProfile::PORTABLE).unwrap();
    let stage = planned.data.topology.stages().next().unwrap().0;
    let Recipe::Reduce(recipe) = &mut planned.data.topology.stage_body_mut(stage).unwrap().primary.recipe
    else {
        panic!("reduction recipe");
    };
    let ScratchRef::Allocate(mut request) = recipe.accumulators[0].partials.clone() else {
        panic!("scratch requirement");
    };
    request.size = LogicalSize::FixedBytes(7);
    recipe.accumulators.push(super::super::reduce::ReductionAccumulator {
        capture_inputs: Vec::new(),
        partials: ScratchRef::Allocate(request),
    });
    assert!(allocate_scratch(planned).unwrap_err().to_string().contains("conflicting scratch"));
}
