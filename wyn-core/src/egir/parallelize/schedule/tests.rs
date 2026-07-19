#![allow(clippy::expect_used, clippy::unwrap_used)]

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
        let mut entry = body(&format!("kernel_{index}"));
        if kind == KernelKind::GraphicsPassthrough {
            entry.execution_model = ExecutionModel::Vertex;
        }
        let recipe = KernelRecipe::close(KernelRecipeSpec::new(entry, kind)).expect("close recipe");
        assert_eq!(recipe.kind(), kind);
        assert_eq!(recipe.entry().name, format!("kernel_{index}"));
    }
}

#[test]
fn closed_recipe_type_rejects_compute_graphics_mismatches() {
    assert!(KernelRecipe::close(KernelRecipeSpec::new(
        body("compute"),
        KernelKind::GraphicsPassthrough,
    ))
    .is_err());

    let mut graphics = body("graphics");
    graphics.execution_model = ExecutionModel::Fragment;
    assert!(KernelRecipe::close(KernelRecipeSpec::new(graphics, KernelKind::SerialCompute,)).is_err());
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
fn checked_dependency_insertion_preserves_the_dag() {
    let first = KernelId(0);
    let second = KernelId(1);
    let mut plan = plan(vec![
        phase(first.0, "first", KernelKind::SerialCompute),
        phase(second.0, "second", KernelKind::SerialCompute),
    ]);

    plan.add_dependency(first, second).expect("first dependency");
    assert_eq!(
        plan.add_dependency(second, first),
        Err(KernelMutationError::DependencyCycle {
            reader: second,
            writer: first,
        })
    );
    assert!(plan.phase(second).expect("second phase").dependencies.is_empty());
}

#[test]
fn validator_rejects_incomplete_phase_families() {
    let mut head = phase(0, "reduce", KernelKind::ReducePhase1);
    head.flow_source = Some(CompilerFlowEndpoint::Materialization(MaterializationId(0)));
    let resources = LogicalResourceArena::default();
    let error = plan(vec![head]).validate_program(&resources, &PipelineDescriptor::default()).unwrap_err();
    assert!(error.contains("incomplete phase family"));
}

#[test]
fn mutation_handles_survive_entry_point_changes_and_chain_insertions() {
    let root = KernelId(7);
    let mut plan = plan(vec![phase(root.0, "seeded", KernelKind::SerialCompute)]);

    plan.install_chain(
        root,
        KernelChainSpec::new(KernelRecipeSpec::new(body("renamed"), KernelKind::SerialCompute)).with_after(
            vec![
                PhaseSpec::new(
                    body("child"),
                    DomainSelection::Explicit(KernelDomain::Fixed { x: 1, y: 1, z: 1 }),
                    KernelKind::ReduceCombine,
                ),
                PhaseSpec::new(
                    body("grandchild"),
                    DomainSelection::Explicit(KernelDomain::Fixed { x: 1, y: 1, z: 1 }),
                    KernelKind::ReduceCombine,
                ),
            ],
        ),
    )
    .unwrap();
    plan.set_output_projection(root, Vec::new()).unwrap();

    let phases = plan.phases().collect::<Vec<_>>();
    assert_eq!(
        phases.iter().map(|phase| phase.entry_point()).collect::<Vec<_>>(),
        ["renamed", "child", "grandchild",]
    );
    let child = phases[1].id;
    let grandchild = phases[2].id;
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
