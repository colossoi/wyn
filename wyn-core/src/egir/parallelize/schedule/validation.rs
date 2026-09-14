use super::*;

impl KernelPlan {
    pub(super) fn validate(&self) -> Result<(), String> {
        let ids = self.topology.kernel_order().iter().copied().collect::<HashSet<_>>();
        if ids.len() != self.catalog.len() || self.catalog.keys().any(|id| !ids.contains(id)) {
            return Err(
                "finalized topology and prepared body catalog have different kernel ownership".into(),
            );
        }
        let mut names = HashSet::new();
        let mut entries = HashSet::new();
        for phase in self.phases() {
            if !names.insert(phase.entry_point()) {
                return Err(format!("duplicate physical entry {:?}", phase.entry_point()));
            }
            if !entries.insert(phase.entry.id) {
                return Err(format!(
                    "physical entry {:?} is owned by multiple kernels",
                    phase.entry.id
                ));
            }
        }
        validate_routes(
            self.catalog.iter().map(|(&id, body)| (id, body)),
            &self.source_entries,
        )?;
        Ok(())
    }
}
