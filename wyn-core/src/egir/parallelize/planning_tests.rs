#![allow(clippy::expect_used, clippy::unwrap_used)]

use super::*;
use crate::egir::program::{LogicalResource, SemanticEntryId};

fn resource(id: u32, owner: u32, kind: CompilerResourceKind, slot: usize) -> LogicalResource {
    LogicalResource {
        id: crate::ResourceId(id),
        origin: ResourceOrigin::Compiler(CompilerResource::new(kind, Some(SemanticOpId(owner)), slot)),
        elem_ty: Type::Constructed(TypeName::UInt(32), vec![]),
        size: LogicalSize::FixedBytes(4),
    }
}

#[test]
fn resource_index_checks_density_and_exact_cardinality() {
    let sparse = [resource(1, 7, CompilerResourceKind::ReducePartial, 0)];
    assert!(ResourceIndex::new(&sparse).is_err());

    let missing_slot = [resource(0, 7, CompilerResourceKind::ReducePartial, 1)];
    let missing_slot = ResourceIndex::new(&missing_slot).expect("dense manifest");
    assert!(missing_slot
        .ordered_slots(SemanticOpId(7), CompilerResourceKind::ReducePartial, 0, 1)
        .is_err());

    let duplicate_slot = [
        resource(0, 7, CompilerResourceKind::ReducePartial, 0),
        resource(1, 7, CompilerResourceKind::ReducePartial, 0),
    ];
    assert!(ResourceIndex::new(&duplicate_slot).is_err());

    let resources = [
        resource(0, 7, CompilerResourceKind::ReducePartial, 1),
        resource(1, 7, CompilerResourceKind::ReducePartial, 0),
        resource(2, 8, CompilerResourceKind::ScanBlockSums, 0),
    ];
    let index = ResourceIndex::new(&resources).expect("dense test manifest");
    let ordered = index.owned(SemanticOpId(7), CompilerResourceKind::ReducePartial);
    assert_eq!(
        ordered.iter().map(|resource| resource.id.0).collect::<Vec<_>>(),
        [1, 0]
    );
    assert!(index.ordered_slots(SemanticOpId(7), CompilerResourceKind::ReducePartial, 0, 2).is_ok());
    assert!(index.exactly_one(SemanticOpId(7), CompilerResourceKind::ReducePartial).is_err());
    assert_eq!(
        index
            .exactly_one(SemanticOpId(8), CompilerResourceKind::ScanBlockSums)
            .expect("one scan-sum resource")
            .id,
        crate::ResourceId(2)
    );
    assert!(index.exactly_one_at(SemanticOpId(8), CompilerResourceKind::ScanBlockSums, 0).is_ok());
    assert!(index.exactly_one(SemanticOpId(9), CompilerResourceKind::ScanBlockSums).is_err());
    assert!(index.exactly_one(SemanticOpId(7), CompilerResourceKind::ScanBlockSums).is_err());
    assert!(index
        .optional(SemanticOpId(9), CompilerResourceKind::ScanBlockSums)
        .expect("missing optional resource is valid")
        .is_none());
    assert!(index.optional(SemanticOpId(7), CompilerResourceKind::ReducePartial).is_err());
}

#[test]
fn sequential_candidate_index_carries_no_parallel_recipe_state() {
    let mut sequential = CandidateIndex::sequential();
    assert!(sequential
        .take_endpoint(CompilerFlowEndpoint::Entry(SemanticEntryId(0)))
        .expect("sequential endpoint lookup")
        .is_none());
}
