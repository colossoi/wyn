use super::*;
use crate::egir::builder::EntryBuilder;
use crate::egir::program::MaterializationId;

fn body(name: &str) -> PlannedEntry {
    let mut effect_ids = crate::IdSource::new();
    EntryBuilder::new_compute(name.to_string(), (1, 1, 1), &mut effect_ids).build()
}

fn phase(id: u32, name: &str, kind: KernelKind) -> KernelPhase {
    let spec = PhaseSpec::new(
        body(name),
        DomainSelection::Explicit(KernelDomain::Fixed { x: 1, y: 1, z: 1 }),
        kind,
    );
    phase_from_body(KernelId(id), None, None, spec).unwrap()
}

fn plan(phases: Vec<KernelPhase>) -> KernelPlan {
    let next_kernel_id = phases.iter().map(|phase| phase.id.0).max().map_or(0, |id| id + 1);
    KernelPlan {
        unpublished: phases,
        next_kernel_id,
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
        let recipe = KernelRecipe::close(KernelRecipeSpec::new(body(&format!("kernel_{index}")), kind));
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

#[test]
fn mutation_handles_survive_entry_point_changes_and_chain_insertions() {
    let root = KernelId(7);
    let mut plan = plan(vec![phase(root.0, "seeded", KernelKind::SerialCompute)]);

    assert_eq!(
        plan.commit_kernel(
            root,
            KernelRecipeSpec::new(body("renamed"), KernelKind::SerialCompute),
        ),
        Ok(root)
    );
    plan.set_output_projection(root, Vec::new()).unwrap();

    let child = plan
        .add_phase_after(
            root,
            PhaseSpec::new(
                body("child"),
                DomainSelection::Explicit(KernelDomain::Fixed { x: 1, y: 1, z: 1 }),
                KernelKind::ReduceCombine,
            ),
        )
        .unwrap();
    let grandchild = plan
        .add_phase_after(
            child,
            PhaseSpec::new(
                body("grandchild"),
                DomainSelection::Explicit(KernelDomain::Fixed { x: 1, y: 1, z: 1 }),
                KernelKind::ReduceCombine,
            ),
        )
        .unwrap();

    let phases = plan.phases().collect::<Vec<_>>();
    assert_eq!(
        phases.iter().map(|phase| phase.entry_point.as_str()).collect::<Vec<_>>(),
        ["renamed", "child", "grandchild",]
    );
    assert_eq!(phases[1].id, child);
    assert_eq!(phases[1].dependencies, vec![root]);
    assert_eq!(phases[2].id, grandchild);
    assert_eq!(phases[2].dependencies, vec![child]);
}

#[test]
fn mutation_apis_report_unknown_kernel_ids() {
    let mut plan = plan(vec![phase(0, "seeded", KernelKind::SerialCompute)]);
    let unknown = KernelId(99);

    assert_eq!(
        plan.commit_kernel(
            unknown,
            KernelRecipeSpec::new(body("seeded"), KernelKind::SerialCompute),
        ),
        Err(KernelMutationError::UnknownKernel(unknown))
    );
    assert_eq!(
        plan.set_output_projection(unknown, Vec::new()),
        Err(KernelMutationError::UnknownKernel(unknown))
    );
}
