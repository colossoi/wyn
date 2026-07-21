use super::*;

impl KernelPlan {
    pub(super) fn validate(&self) -> Result<(), String> {
        let phases = self.phases_with_ids().collect::<Vec<_>>();
        let mut names = HashSet::new();
        let mut placements = HashSet::new();
        let mut group_orders = HashMap::<PhaseGroup, Vec<usize>>::new();
        for (id, phase) in &phases {
            if !names.insert(phase.entry_point()) {
                return Err(format!("duplicate physical entry `{}`", phase.entry_point()));
            }
            if phase.dependencies.iter().any(|dependency| dependency == id) {
                return Err(format!("kernel `{}` depends on itself", phase.entry_point()));
            }
            if !placements.insert((phase.placement.group, phase.placement.order)) {
                return Err(format!(
                    "kernel `{}` duplicates placement {:?}/{}",
                    phase.entry_point(),
                    phase.placement.group,
                    phase.placement.order
                ));
            }
            group_orders.entry(phase.placement.group).or_default().push(phase.placement.order);
        }
        for (group, mut orders) in group_orders {
            orders.sort_unstable();
            if orders.iter().copied().ne(0..orders.len()) {
                return Err(format!("phase placement {group:?} has a gap"));
            }
        }
        validate_acyclic(&phases)?;
        Ok(())
    }
}

fn validate_acyclic(phases: &[(KernelId, &KernelPhase)]) -> Result<(), String> {
    let mut emitted = HashSet::new();
    while emitted.len() < phases.len() {
        let ready = phases.iter().find(|(id, phase)| {
            !emitted.contains(id)
                && phase.dependencies.iter().all(|dependency| emitted.contains(dependency))
        });
        let Some(phase) = ready else {
            return Err("kernel dependency graph contains a cycle".into());
        };
        emitted.insert(phase.0);
    }
    Ok(())
}
