#![allow(clippy::expect_used, clippy::unwrap_used)]

use super::*;
use crate::egir::builder::EntryBuilder;
use crate::egir::program::MaterializationId;

fn body(name: &str) -> PlannedEntry {
    let mut effect_ids = crate::IdSource::new();
    EntryBuilder::new_compute(name.to_string(), (1, 1, 1), &mut effect_ids).build()
}

fn phase(name: &str, kind: ComputeKernelKind) -> KernelPhase {
    let spec = PhaseSpec::compute(
        body(name),
        DomainSelection::Explicit(KernelDomain::Fixed { x: 1, y: 1, z: 1 }),
        kind,
    );
    phase_from_body(None, None, spec).unwrap()
}

fn plan(phases: Vec<KernelPhase>) -> KernelPlan {
    let mut arena = crate::IdArena::new();
    let unpublished = phases
        .into_iter()
        .map(|phase| arena.alloc(PlacedPhase(PhaseListId::Unpublished, phase)))
        .collect::<Vec<_>>();
    KernelPlan {
        phases: arena,
        unpublished,
        ..KernelPlan::default()
    }
}

#[test]
fn every_kernel_kind_closes_over_a_complete_planned_entry() {
    let kinds = [
        (KernelKind::SerialCompute, ComputeKernelKind::Serial),
        (
            KernelKind::SharedArrayMaterialization,
            ComputeKernelKind::SharedArrayMaterialization,
        ),
        (KernelKind::ScalarPrepass, ComputeKernelKind::ScalarPrepass),
        (KernelKind::GatherPrepass, ComputeKernelKind::GatherPrepass),
        (KernelKind::FilterScan, ComputeKernelKind::FilterScan),
        (KernelKind::FilterCombine, ComputeKernelKind::FilterCombine),
        (KernelKind::ReducePhase1, ComputeKernelKind::ReducePhase1),
        (KernelKind::ReduceCombine, ComputeKernelKind::ReduceCombine),
        (KernelKind::ScanPhase1, ComputeKernelKind::ScanPhase1),
        (KernelKind::ScanBlock, ComputeKernelKind::ScanBlock),
        (KernelKind::ScanApplyOffsets, ComputeKernelKind::ScanApplyOffsets),
    ];
    for (index, (kind, internal_kind)) in kinds.into_iter().enumerate() {
        let entry = body(&format!("kernel_{index}"));
        let recipe =
            KernelRecipe::close(KernelRecipeSpec::compute(entry, internal_kind)).expect("close recipe");
        assert_eq!(recipe.kind(), kind);
        assert_eq!(recipe.entry().name, format!("kernel_{index}"));
    }

    let mut graphics = body("graphics");
    graphics.execution_model = ExecutionModel::Vertex;
    let recipe = KernelRecipe::close(KernelRecipeSpec::graphics(graphics)).expect("close graphics");
    assert_eq!(recipe.kind(), KernelKind::GraphicsPassthrough);
}

#[test]
fn closed_recipe_type_rejects_compute_graphics_mismatches() {
    assert!(KernelRecipe::close(KernelRecipeSpec::graphics(body("compute"))).is_err());

    let mut graphics = body("graphics");
    graphics.execution_model = ExecutionModel::Fragment;
    assert!(KernelRecipe::close(KernelRecipeSpec::compute(graphics, ComputeKernelKind::Serial)).is_err());
}

#[test]
fn validator_rejects_duplicate_names_and_dependency_cycles() {
    let error = plan(vec![
        phase("same", ComputeKernelKind::Serial),
        phase("same", ComputeKernelKind::Serial),
    ])
    .validate()
    .unwrap_err();
    assert!(error.contains("duplicate physical entry"));

    let mut first = phase("first", ComputeKernelKind::Serial);
    let mut second = phase("second", ComputeKernelKind::Serial);
    first.dependencies.push(KernelId(1));
    second.dependencies.push(KernelId(0));
    assert!(plan(vec![first, second]).validate().unwrap_err().contains("cycle"));
}

#[test]
fn checked_dependency_insertion_preserves_the_dag() {
    let first = KernelId(0);
    let second = KernelId(1);
    let mut plan = plan(vec![
        phase("first", ComputeKernelKind::Serial),
        phase("second", ComputeKernelKind::Serial),
    ]);

    plan.add_dependency(first, second).expect("first dependency");
    assert_eq!(
        plan.add_dependency(second, first),
        Err(KernelMutationError::DependencyCycle {
            reader: second,
            writer: first,
        })
    );
    assert!(plan.phase(second).dependencies.is_empty());
}

#[test]
fn validator_rejects_incomplete_phase_families() {
    let mut head = phase("reduce", ComputeKernelKind::ReducePhase1);
    head.flow_source = Some(CompilerFlowEndpoint::Materialization(MaterializationId(0)));
    let resources = LogicalResourceArena::default();
    let error = plan(vec![head]).validate_program(&resources, &PipelineDescriptor::default()).unwrap_err();
    assert!(error.contains("incomplete phase family"));
}

#[test]
fn mutation_handles_survive_entry_point_changes_and_chain_insertions() {
    let root = KernelId(0);
    let mut plan = plan(vec![phase("seeded", ComputeKernelKind::Serial)]);

    plan.install_chain(
        root,
        KernelChainSpec::new(KernelRecipeSpec::compute(
            body("renamed"),
            ComputeKernelKind::Serial,
        ))
        .with_after(vec![
            PhaseSpec::compute(
                body("child"),
                DomainSelection::Explicit(KernelDomain::Fixed { x: 1, y: 1, z: 1 }),
                ComputeKernelKind::ReduceCombine,
            ),
            PhaseSpec::compute(
                body("grandchild"),
                DomainSelection::Explicit(KernelDomain::Fixed { x: 1, y: 1, z: 1 }),
                ComputeKernelKind::ReduceCombine,
            ),
        ]),
    )
    .unwrap();
    plan.set_output_projection(root, Vec::new()).unwrap();

    let phases = plan.phases().collect::<Vec<_>>();
    assert_eq!(
        phases.iter().map(|phase| phase.entry_point()).collect::<Vec<_>>(),
        ["renamed", "child", "grandchild",]
    );
    let child = KernelId(1);
    assert_eq!(phases[1].dependencies, vec![root]);
    assert_eq!(phases[2].dependencies, vec![child]);
}

#[test]
fn mutation_apis_report_unknown_kernel_ids() {
    let mut plan = plan(vec![phase("seeded", ComputeKernelKind::Serial)]);
    let unknown = KernelId(99);

    assert_eq!(
        plan.commit_kernel(
            unknown,
            KernelRecipeSpec::compute(body("seeded"), ComputeKernelKind::Serial),
        ),
        Err(KernelMutationError::UnknownKernel(unknown))
    );
    assert_eq!(
        plan.set_output_projection(unknown, Vec::new()),
        Err(KernelMutationError::UnknownKernel(unknown))
    );
}
