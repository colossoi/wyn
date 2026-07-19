use super::*;

impl KernelPlan {
    pub(super) fn validate(&self) -> Result<(), String> {
        let phases = self.phases().collect::<Vec<_>>();
        let mut ids = HashSet::new();
        let mut names = HashSet::new();
        for phase in &phases {
            if !ids.insert(phase.id) {
                return Err(format!("duplicate kernel id {:?}", phase.id));
            }
            if !names.insert(phase.entry_point()) {
                return Err(format!("duplicate physical entry `{}`", phase.entry_point()));
            }
            if phase.dependencies.iter().any(|dependency| *dependency == phase.id) {
                return Err(format!("kernel `{}` depends on itself", phase.entry_point()));
            }
            validate_entry(phase.recipe.entry())?;
        }
        for phase in &phases {
            for dependency in &phase.dependencies {
                if !ids.contains(dependency) {
                    return Err(format!(
                        "kernel `{}` depends on missing kernel {:?}",
                        phase.entry_point(),
                        dependency
                    ));
                }
            }
        }
        validate_acyclic(&phases)?;
        let mut callable_names = HashSet::new();
        for function in &self.generated_callables {
            if !callable_names.insert(function.name.as_str()) {
                return Err(format!("generated callable `{}` is duplicated", function.name));
            }
            if self.region_interner.get(&function.name).is_none() {
                return Err(format!("generated callable `{}` has no region id", function.name));
            }
        }
        Ok(())
    }

    pub(super) fn validate_program(
        &self,
        resources: &[LogicalResource],
        descriptor: &PipelineDescriptor,
    ) -> Result<(), String> {
        let resource_ids = resources.iter().map(|resource| resource.id).collect::<HashSet<_>>();
        if resource_ids.len() != resources.len()
            || resources.iter().enumerate().any(|(index, resource)| resource.id.0 as usize != index)
        {
            return Err("logical resource arena is not dense and unique".into());
        }
        let host_bindings = resources.iter().filter_map(LogicalResource::host_binding).collect::<Vec<_>>();
        if host_bindings.iter().copied().collect::<HashSet<_>>().len() != host_bindings.len() {
            return Err("host resources contain a binding collision".into());
        }

        let mut output_owners = HashMap::new();
        let mut primary = HashMap::<SemanticEntryId, usize>::new();
        let mut projected = HashMap::<SemanticEntryId, HashSet<OutputSlotId>>::new();
        let mut family = HashMap::<CompilerFlowEndpoint, HashSet<KernelKind>>::new();
        for phase in self.phases() {
            let entry = phase.recipe.entry();
            let resources = phase.resources();
            for scheduled in &resources {
                if !resource_ids.contains(&scheduled.resource) {
                    return Err(format!(
                        "kernel `{}` references missing resource {:?}",
                        phase.entry_point(),
                        scheduled.resource
                    ));
                }
            }
            for actual in planned_resources(entry) {
                let Some(scheduled) = resources.iter().find(|item| item.resource == actual.resource) else {
                    return Err(format!(
                        "kernel `{}` omits resource {:?}",
                        phase.entry_point(),
                        actual.resource
                    ));
                };
                if actual.access.reads() && !scheduled.access.reads()
                    || actual.access.writes() && !scheduled.access.writes()
                {
                    return Err(format!(
                        "kernel `{}` understates access to {:?}",
                        phase.entry_point(),
                        actual.resource
                    ));
                }
            }
            if let Some(flow) = phase.flow_source {
                family.entry(flow).or_default().insert(phase.recipe.kind());
            }
            let projection = phase.abi_projection();
            let Some(source) = projection.source_entry else {
                continue;
            };
            let abi = self
                .entry_schedules
                .get(&source)
                .ok_or_else(|| format!("kernel `{}` has invalid source id", phase.entry_point()))?;
            if phase.id == abi.primary {
                *primary.entry(source).or_default() += 1;
                if phase.entry_point() != abi.name {
                    return Err(format!(
                        "semantic entry `{}` has primary kernel `{}`",
                        abi.name,
                        phase.entry_point()
                    ));
                }
            }
            if projection.inputs.iter().any(|slot| slot.0 >= abi.input_count) {
                return Err(format!(
                    "kernel `{}` has an invalid input projection",
                    phase.entry_point()
                ));
            }
            let mut physical_slots = HashSet::new();
            for route in &projection.output_routes {
                if route.semantic_slot.0 >= abi.output_count
                    || route.physical_slot.0 >= entry.outputs.len()
                    || !physical_slots.insert(route.physical_slot)
                {
                    return Err(format!(
                        "kernel `{}` has an invalid output projection",
                        phase.entry_point()
                    ));
                }
                if abi.output_owners.get(&route.semantic_slot) != Some(&phase.id) {
                    return Err(format!(
                        "kernel `{}` does not own semantic output {:?}/{}",
                        phase.entry_point(),
                        source,
                        route.semantic_slot.0
                    ));
                }
                if let Some(owner) = output_owners.insert((source, route.semantic_slot), phase.id) {
                    return Err(format!(
                        "semantic output {:?}/{} is owned by {:?} and {:?}",
                        source, route.semantic_slot.0, owner, phase.id
                    ));
                }
                projected.entry(source).or_default().insert(route.semantic_slot);
            }
        }
        for (&source, abi) in &self.entry_schedules {
            if primary.get(&source).copied().unwrap_or(0) != 1 {
                return Err(format!(
                    "semantic entry `{}` has no unique primary kernel",
                    abi.name
                ));
            }
            let routed = projected.get(&source).cloned().unwrap_or_default();
            if !abi.output_owners.keys().all(|output| routed.contains(output)) {
                return Err(format!("semantic entry `{}` has unpublished outputs", abi.name));
            }
        }
        validate_families(&family)?;
        validate_resource_flows(self, resources)?;
        for pipeline in &descriptor.pipelines {
            if let Pipeline::Compute(compute) = pipeline {
                for stage in &compute.stages {
                    if !self.contains_entry(&stage.entry_point) {
                        return Err(format!("source stage `{}` is not planned", stage.entry_point));
                    }
                }
            }
        }
        Ok(())
    }
}

fn validate_entry<P: crate::egir::types::EgirPhase>(entry: &PlannedEntry<P>) -> Result<(), String> {
    let effects = entry
        .graph
        .skeleton
        .blocks
        .values()
        .flat_map(|block| &block.side_effects)
        .flat_map(|effect| effect.effects.into_iter().flat_map(|(left, right)| [left, right]))
        .collect::<HashSet<_>>();
    for route in &entry.output_routes {
        if route.slot.0 >= entry.outputs.len()
            || !entry.graph.skeleton.blocks.contains_key(route.source.block)
            || !entry.graph.nodes.contains_key(route.source.value)
        {
            return Err(format!("entry `{}` has an invalid output route", entry.name));
        }
        for writer in &route.writers {
            let valid = match writer {
                crate::egir::program::OutputWriter::Value(value) => entry.graph.nodes.contains_key(*value),
                crate::egir::program::OutputWriter::Effect(effect) => effects.contains(effect),
            };
            if !valid {
                return Err(format!("entry `{}` has an invalid output writer", entry.name));
            }
        }
    }
    if entry.aliases.iter().any(|(left, right)| {
        !entry.graph.nodes.contains_key(*left) || !entry.graph.nodes.contains_key(*right)
    }) {
        return Err(format!("entry `{}` has an invalid alias", entry.name));
    }
    Ok(())
}

fn validate_acyclic(phases: &[&KernelPhase]) -> Result<(), String> {
    if super::topologically_order_phases(phases.iter().map(|phase| (*phase).clone()).collect()).is_err() {
        return Err("kernel dependency graph contains a cycle".into());
    }
    Ok(())
}

fn validate_families(families: &HashMap<CompilerFlowEndpoint, HashSet<KernelKind>>) -> Result<(), String> {
    for (flow, kinds) in families {
        let requires = |head, tail: &[KernelKind]| {
            !kinds.contains(&head) || tail.iter().all(|kind| kinds.contains(kind))
        };
        if !requires(
            KernelKind::FilterFlags,
            &[
                KernelKind::FilterScan,
                KernelKind::FilterCombine,
                KernelKind::FilterScatter,
            ],
        ) || !requires(KernelKind::ReducePhase1, &[KernelKind::ReduceCombine])
            || !requires(
                KernelKind::ScanPhase1,
                &[KernelKind::ScanBlock, KernelKind::ScanApplyOffsets],
            )
        {
            return Err(format!("compiler flow {flow:?} has an incomplete phase family"));
        }
    }
    Ok(())
}

fn validate_resource_flows(plan: &KernelPlan, resources: &[LogicalResource]) -> Result<(), String> {
    for resource in resources {
        let ResourceOrigin::Compiler(compiler) = &resource.origin else {
            continue;
        };
        let Some(flow) = &compiler.flow else {
            continue;
        };
        let writers = plan
            .flow_resource_phases(flow.producer, resource.id, true)
            .map(|phase| phase.id)
            .collect::<HashSet<_>>();
        if writers.is_empty() {
            return Err(format!("resource {:?} has no producer", resource.id));
        }
        for consumer in &flow.consumers {
            for reader in plan.flow_resource_phases(*consumer, resource.id, false) {
                if !reader.dependencies.iter().any(|dependency| writers.contains(dependency)) {
                    return Err(format!(
                        "kernel `{}` reads {:?} without a producer dependency",
                        reader.entry_point(),
                        resource.id
                    ));
                }
            }
        }
    }
    Ok(())
}
