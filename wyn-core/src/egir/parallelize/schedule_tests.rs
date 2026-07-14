use super::*;
use crate::egir::builder::EntryBuilder;
use crate::egir::program::MaterializationId;

fn body(name: &str) -> PlannedEntry {
    EntryBuilder::new_compute(name.to_string(), (1, 1, 1)).build()
}

fn phase(id: u32, name: &str, kind: KernelKind) -> KernelPhase {
    phase_from_body(
        KernelId(id),
        None,
        None,
        body(name),
        DomainSelection::Explicit(KernelDomain::Fixed { x: 1, y: 1, z: 1 }),
        kind,
    )
    .unwrap()
}

fn plan(phases: Vec<KernelPhase>) -> KernelPlan {
    KernelPlan {
        unpublished: phases,
        ..KernelPlan::default()
    }
}

#[test]
fn every_kernel_kind_closes_over_a_complete_planned_entry() {
    let kinds = [
        KernelKind::GraphicsPassthrough,
        KernelKind::SerialCompute,
        KernelKind::OutputDomainProjection,
        KernelKind::SharedArrayMaterialization,
        KernelKind::ScalarPrepass,
        KernelKind::GatherPrepass,
        KernelKind::FilterFlags,
        KernelKind::FilterScan,
        KernelKind::FilterCombine,
        KernelKind::FilterScatter,
        KernelKind::ReducePhase1,
        KernelKind::ReduceCombine,
        KernelKind::ScanPhase1,
        KernelKind::ScanBlock,
        KernelKind::ScanApplyOffsets,
    ];
    for (index, kind) in kinds.into_iter().enumerate() {
        let recipe = KernelRecipe::close(kind, body(&format!("kernel_{index}")));
        assert_eq!(recipe.kind(), kind);
        assert_eq!(recipe.entry().name, format!("kernel_{index}"));
    }
}

#[test]
fn validator_rejects_duplicate_names_and_dependency_cycles() {
    let error = plan(vec![
        phase(0, "same", KernelKind::SerialCompute),
        phase(1, "same", KernelKind::SerialCompute),
    ])
    .validate()
    .unwrap_err();
    assert!(error.contains("duplicate physical entry"));

    let mut first = phase(0, "first", KernelKind::SerialCompute);
    let mut second = phase(1, "second", KernelKind::SerialCompute);
    first.dependencies.push(second.id);
    second.dependencies.push(first.id);
    assert!(plan(vec![first, second]).validate().unwrap_err().contains("cycle"));
}

#[test]
fn validator_rejects_incomplete_phase_families() {
    let mut head = phase(0, "reduce", KernelKind::ReducePhase1);
    head.flow_source = Some(CompilerFlowEndpoint::Materialization(MaterializationId(0)));
    let error = plan(vec![head]).validate_program(&[], &[], &PipelineDescriptor::default()).unwrap_err();
    assert!(error.contains("incomplete phase family"));
}
