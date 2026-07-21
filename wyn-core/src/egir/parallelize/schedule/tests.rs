#![allow(clippy::expect_used, clippy::unwrap_used)]

use super::*;
use crate::egir::builder::EntryBuilder;

fn body(name: &str) -> PlannedEntry {
    let mut effect_ids = crate::IdSource::new();
    EntryBuilder::new_compute(name.to_string(), (1, 1, 1), &mut effect_ids).build()
}

fn spec(name: &str, label: &'static str) -> PhaseSpec {
    PhaseSpec::compute(
        body(name),
        KernelDispatch::explicit(KernelDomain::Fixed { x: 1, y: 1, z: 1 }),
        label,
    )
}

fn phase(name: &str, label: &'static str, order: usize) -> KernelPhase {
    phase_from_body(
        None,
        None,
        PhasePlacement {
            group: PhaseGroup::Unpublished,
            order,
        },
        spec(name, label),
    )
    .unwrap()
}

fn plan(mut phases: Vec<KernelPhase>) -> KernelPlan {
    for (order, phase) in phases.iter_mut().enumerate() {
        phase.placement = PhasePlacement {
            group: PhaseGroup::Unpublished,
            order,
        };
    }
    KernelPlan {
        phases,
        ..KernelPlan::default()
    }
}

#[test]
fn body_preparation_retains_creator_supplied_facts() {
    let prepared = spec("kernel", "diagnostic_label").prepare().unwrap();
    assert_eq!(prepared.label, "diagnostic_label");
    assert_eq!(prepared.entry.name, "kernel");
    assert_eq!(
        prepared.dispatch,
        KernelDispatch::explicit(KernelDomain::Fixed { x: 1, y: 1, z: 1 })
    );

    let mut graphics = body("graphics");
    graphics.execution_model = ExecutionModel::Vertex;
    let prepared = PhaseSpec::graphics(
        graphics,
        KernelDispatch::inferred(KernelDomain::Fixed { x: 1, y: 1, z: 1 }),
    )
    .prepare()
    .unwrap();
    assert_eq!(prepared.label, "graphics_passthrough");
    assert_eq!(prepared.entry.name, "graphics");
}

#[test]
fn body_preparation_rejects_compute_graphics_mismatches() {
    assert!(PhaseSpec::graphics(
        body("compute"),
        KernelDispatch::inferred(KernelDomain::Fixed { x: 1, y: 1, z: 1 })
    )
    .prepare()
    .is_err());

    let mut graphics = body("graphics");
    graphics.execution_model = ExecutionModel::Fragment;
    assert!(PhaseSpec::compute(
        graphics,
        KernelDispatch::inferred(KernelDomain::Fixed { x: 1, y: 1, z: 1 }),
        "compute"
    )
    .prepare()
    .is_err());
}

#[test]
fn validator_rejects_duplicate_names_and_dependency_cycles() {
    let error = plan(vec![
        phase("same", "serial_compute", 0),
        phase("same", "serial_compute", 1),
    ])
    .validate()
    .unwrap_err();
    assert!(error.contains("duplicate physical entry"));

    let mut first = phase("first", "serial_compute", 0);
    let mut second = phase("second", "serial_compute", 1);
    first.dependencies.push(KernelId::for_test(1));
    second.dependencies.push(KernelId::for_test(0));
    assert!(plan(vec![first, second]).validate().unwrap_err().contains("cycle"));
}

#[test]
fn validator_rejects_duplicate_and_gapped_placements() {
    let mut duplicate = plan(vec![
        phase("first", "serial_compute", 0),
        phase("second", "serial_compute", 1),
    ]);
    duplicate.phases[1].placement.order = 0;
    assert!(duplicate.validate().unwrap_err().contains("duplicates placement"));

    let mut gapped = plan(vec![
        phase("first", "serial_compute", 0),
        phase("second", "serial_compute", 1),
    ]);
    gapped.phases[1].placement.order = 2;
    assert!(gapped.validate().unwrap_err().contains("has a gap"));
}

#[test]
fn checked_dependency_insertion_preserves_the_dag() {
    let first = KernelId::for_test(0);
    let second = KernelId::for_test(1);
    let mut plan = plan(vec![
        phase("first", "serial_compute", 0),
        phase("second", "serial_compute", 1),
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
fn dense_handles_survive_entry_changes_and_chain_insertions() {
    let root = KernelId::for_test(0);
    let mut plan = plan(vec![phase("seeded", "serial_compute", 0)]);

    plan.replace_chain(
        root,
        Vec::new(),
        spec("renamed", "serial_compute").with_output_projection(Some(Vec::new())),
        vec![
            spec("child", "reduce_combine"),
            spec("grandchild", "reduce_combine"),
        ],
    )
    .unwrap();

    let phases = plan.phases().collect::<Vec<_>>();
    assert_eq!(
        phases.iter().map(|phase| phase.entry_point()).collect::<Vec<_>>(),
        ["renamed", "child", "grandchild"]
    );
    assert_eq!(phases[1].dependencies, vec![root]);
    assert_eq!(phases[2].dependencies, vec![KernelId::for_test(1)]);
    assert_eq!(
        plan.phases_with_ids().map(|(id, _)| id).collect::<Vec<_>>(),
        [
            KernelId::for_test(0),
            KernelId::for_test(1),
            KernelId::for_test(2)
        ]
    );
}
