#![allow(clippy::expect_used, clippy::unwrap_used)]

use super::*;
use crate::egir::builder::EntryBuilder;

fn body(name: &str) -> PlannedEntry {
    let mut effect_ids = crate::IdSource::new();
    EntryBuilder::new_compute(name.to_string(), (1, 1, 1), &mut effect_ids).build()
}

fn phase(name: &str, label: &'static str) -> KernelPhase {
    let spec = NewPhaseSpec::compute(
        body(name),
        PhaseDomain::explicit(KernelDomain::Fixed { x: 1, y: 1, z: 1 }),
        label,
    );
    phase_from_body(None, None, spec).unwrap()
}

fn plan(phases: Vec<KernelPhase>) -> KernelPlan {
    let mut arena = crate::IdArena::new();
    let unpublished = phases.into_iter().map(|phase| arena.alloc(phase)).collect::<Vec<_>>();
    KernelPlan {
        phases: arena,
        unpublished,
        ..KernelPlan::default()
    }
}

#[test]
fn body_preparation_retains_diagnostic_labels() {
    let (label, entry, _) =
        prepare_body(KernelBodySpec::compute(body("kernel"), "diagnostic_label")).unwrap();
    assert_eq!(label, "diagnostic_label");
    assert_eq!(entry.name, "kernel");

    let mut graphics = body("graphics");
    graphics.execution_model = ExecutionModel::Vertex;
    let (label, entry, _) = prepare_body(KernelBodySpec::graphics(graphics)).unwrap();
    assert_eq!(label, "graphics_passthrough");
    assert_eq!(entry.name, "graphics");
}

#[test]
fn body_preparation_rejects_compute_graphics_mismatches() {
    assert!(prepare_body(KernelBodySpec::graphics(body("compute"))).is_err());

    let mut graphics = body("graphics");
    graphics.execution_model = ExecutionModel::Fragment;
    assert!(prepare_body(KernelBodySpec::compute(graphics, "compute")).is_err());
}

#[test]
fn validator_rejects_duplicate_names_and_dependency_cycles() {
    let error = plan(vec![
        phase("same", "serial_compute"),
        phase("same", "serial_compute"),
    ])
    .validate()
    .unwrap_err();
    assert!(error.contains("duplicate physical entry"));

    let mut first = phase("first", "serial_compute");
    let mut second = phase("second", "serial_compute");
    first.dependencies.push(KernelId(1));
    second.dependencies.push(KernelId(0));
    assert!(plan(vec![first, second]).validate().unwrap_err().contains("cycle"));
}

#[test]
fn checked_dependency_insertion_preserves_the_dag() {
    let first = KernelId(0);
    let second = KernelId(1);
    let mut plan = plan(vec![
        phase("first", "serial_compute"),
        phase("second", "serial_compute"),
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
fn mutation_handles_survive_entry_point_changes_and_chain_insertions() {
    let root = KernelId(0);
    let mut plan = plan(vec![phase("seeded", "serial_compute")]);

    plan.install_chain(
        root,
        Vec::new(),
        KernelBodySpec::compute(body("renamed"), "serial_compute"),
        vec![
            NewPhaseSpec::compute(
                body("child"),
                PhaseDomain::explicit(KernelDomain::Fixed { x: 1, y: 1, z: 1 }),
                "reduce_combine",
            ),
            NewPhaseSpec::compute(
                body("grandchild"),
                PhaseDomain::explicit(KernelDomain::Fixed { x: 1, y: 1, z: 1 }),
                "reduce_combine",
            ),
        ],
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
    let mut plan = plan(vec![phase("seeded", "serial_compute")]);
    let unknown = KernelId(99);

    assert_eq!(
        plan.commit_kernel(unknown, KernelBodySpec::compute(body("seeded"), "serial_compute"),),
        Err(KernelMutationError::UnknownKernel(unknown))
    );
    assert_eq!(
        plan.set_output_projection(unknown, Vec::new()),
        Err(KernelMutationError::UnknownKernel(unknown))
    );
}
