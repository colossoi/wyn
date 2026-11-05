//! Transient summaries accumulated by the TLC walk. These contain no executable
//! expressions. Only operation/region summaries enter structural fixed points.
use super::{Import, Scope};
use crate::egglog::OptimizeError;
use crate::tlc::value_graph::Source;
use crate::tlc::TermId;
use crate::BindingRef;
use crate::{LookupMap, LookupSet};
use egglog_engine::{RawValues, Value, Write};

#[derive(Clone)]
pub(super) struct Summary {
    pub device: bool,
    pub pure: bool,
    pub readonly: bool,
    pub duplicate: bool,
    pub work: i64,
    pub reads: LookupSet<BindingRef>,
    pub dependencies: LookupSet<Value>,
    pub calls: LookupSet<Value>,
    pub regions: LookupSet<Value>,
    pub device_calls: LookupSet<Value>,
    pub device_regions: LookupSet<Value>,
}

impl Default for Summary {
    fn default() -> Self {
        Self {
            device: false,
            pure: true,
            readonly: true,
            duplicate: true,
            work: 0,
            reads: LookupSet::default(),
            dependencies: LookupSet::default(),
            calls: LookupSet::default(),
            regions: LookupSet::default(),
            device_calls: LookupSet::default(),
            device_regions: LookupSet::default(),
        }
    }
}

impl Summary {
    pub(super) fn use_value(&mut self, other: &Self) {
        self.device |= other.device;
        self.dependencies.extend(&other.dependencies);
        self.device_calls.extend(other.calls.iter().chain(&other.device_calls));
        self.device_regions.extend(other.regions.iter().chain(&other.device_regions));
    }

    pub(super) fn evaluate(&mut self, other: &Self) {
        self.use_value(other);
        self.pure &= other.pure;
        self.readonly &= other.readonly;
        self.duplicate &= other.duplicate;
        // Consumers cap inline queries; placement needs the unsaturated sum.
        self.work = self.work.saturating_add(other.work);
        self.reads.extend(&other.reads);
        self.calls.extend(&other.calls);
        self.regions.extend(&other.regions);
    }
}

#[derive(Default)]
pub(super) struct Summaries {
    pub values: LookupMap<Value, Summary>,
    pub pending_uses: Vec<(Value, Value)>,
    pub operations: LookupMap<Value, Value>,
    pub inputs: LookupSet<(Value, Value)>,
    pub captures: LookupSet<(Value, Value)>,
    pub regions_with_operations: LookupSet<Value>,
    pub aliases: LookupSet<Value>,
    pub lengths: LookupMap<Value, Value>,
    pub owners: LookupMap<Value, Value>,
    pub free: LookupMap<Value, LookupSet<Value>>,
}

impl Import<'_, '_, '_, '_> {
    pub(super) fn use_value(&mut self, owner: Value, value: Value) -> Result<(), OptimizeError> {
        self.sink.add("SourceDirectUse", (owner, value))?;
        self.summaries.pending_uses.push((owner, value));
        Ok(())
    }

    /// Classify uses after operation inputs and captures are known.
    /// Scalar dependencies remain direct edges rather than transitive sets.
    pub(super) fn finish_uses(&mut self) -> Result<(), OptimizeError> {
        for &(value, child) in &self.summaries.pending_uses {
            if self.summaries.aliases.contains(&value) {
                continue;
            }
            if let Some(&operation) = self.summaries.operations.get(&value) {
                if self.summaries.inputs.contains(&(operation, child))
                    && !self.summaries.captures.contains(&(operation, child))
                {
                    continue;
                }
                self.sink.add("SourceOperand", (operation, child))?;
            } else {
                self.sink.add("SourceComputedUse", (value, child))?;
            }
        }
        Ok(())
    }

    pub(super) fn use_summary(&mut self, owner: Value, value: Value, evaluate: bool) {
        let child = self.summaries.values.get(&value).cloned().unwrap_or_default();
        let summary = self.summaries.values.entry(owner).or_default();
        if evaluate {
            summary.evaluate(&child);
        } else {
            summary.use_value(&child);
        }
    }

    pub(super) fn free_reference(&mut self, target: Value, mut scope: Value) -> Result<(), OptimizeError> {
        let owner = self.summaries.owners.get(&target).copied();
        while Some(scope) != owner {
            // Referenced dependencies are settled; the first visit covered all ancestors.
            if !self.summaries.free.entry(scope).or_default().insert(target) {
                break;
            }
            if let Some(summary) = self.summaries.values.get(&target) {
                for &before in &summary.dependencies {
                    self.sink.add("SourceRegionDependency", (scope, before))?;
                }
            }
            let Some((Some(parent), _)) = self.identities.scopes.get(&scope) else {
                break;
            };
            scope = *parent;
        }
        Ok(())
    }

    pub(super) fn alias_summary(&mut self, value: Value, target: Value) -> Result<(), OptimizeError> {
        self.summaries.aliases.insert(value);
        self.sink.set("SourceAlias", value, target)?;
        Ok(())
    }

    pub(super) fn project_source(
        &mut self,
        value: Value,
        tuple: Value,
        term: TermId,
        index: usize,
        scope: Value,
    ) -> Result<(), OptimizeError> {
        let Some(field) = self.graph_source(term, &[index], scope)? else {
            let info = self.sink.add("ProjectionInfo", (tuple, index as i64))?;
            self.sink.set("SourceProjection", value, info)?;
            return Ok(());
        };
        self.alias_summary(value, field)?;
        let dependencies =
            self.summaries.values.get(&field).map(|s| s.dependencies.clone()).unwrap_or_default();
        self.summaries.values.entry(value).or_default().dependencies = dependencies;
        Ok(())
    }

    pub(super) fn graph_source(
        &mut self,
        term: TermId,
        fields: &[usize],
        mut scope: Value,
    ) -> Result<Option<Value>, OptimizeError> {
        let Some(source) = self.values.projected_source(Source::Term(term), fields) else {
            return Ok(None);
        };
        Ok(Some(match source {
            Source::Input(symbol) => self.resolve(symbol)?,
            Source::Term(term) => loop {
                if let Some(&value) = self.identities.occurrences.get(&(scope, term)) {
                    break value;
                }
                let Some((Some(parent), _)) = self.identities.scopes.get(&scope) else {
                    return Err(OptimizeError::Output(
                        "projected TLC value has no visible source occurrence".into(),
                    ));
                };
                scope = *parent;
            },
        }))
    }

    pub(super) fn enter_summary(
        &mut self,
        owner: Value,
        child: Value,
        parent: &mut Scope,
    ) -> Result<(), OptimizeError> {
        self.summaries.values.entry(owner).or_default().regions.insert(child);
        parent.summary.regions.insert(child);
        if let Some(free) = self.summaries.free.get(&child).cloned() {
            for value in free {
                self.sink.add("SourceNestedUse", (owner, value))?;
                self.summaries.pending_uses.push((owner, value));
                self.use_summary(owner, value, false);
            }
        }
        Ok(())
    }

    pub(super) fn dependencies(
        &mut self,
        operation: Value,
        value: Value,
        role: &str,
    ) -> Result<(), OptimizeError> {
        if let Some(summary) = self.summaries.values.get(&value) {
            let role = self.sink.add(role, RawValues(vec![]))?;
            for &before in &summary.dependencies {
                if before != operation {
                    self.sink.add("SourceDepends", (operation, before, role))?;
                }
            }
        }
        Ok(())
    }

    pub(super) fn finish_summary(&mut self, value: Value, scope: &mut Scope) -> Result<(), OptimizeError> {
        self.summaries.owners.insert(value, scope.key);
        let summary = self.summaries.values.get(&value).cloned().unwrap_or_default();
        self.sink.set("SourceReadOnly", value, summary.readonly)?;
        // Region-local properties were accumulated from each node once. Device
        // requirements additionally follow references to already evaluated values.
        scope.summary.device |= summary.device;
        scope.summary.device_calls.extend(&summary.device_calls);
        scope.summary.device_regions.extend(&summary.device_regions);
        if let Some(&operation) = self.summaries.operations.get(&value) {
            if let Some(&array) = self.summaries.lengths.get(&value) {
                self.dependencies(operation, array, "LengthUse")?;
            } else {
                self.dependencies(operation, value, "Other")?;
            }
            self.write_summary(value, &summary, false)?;
            let result = self.summaries.values.entry(value).or_default();
            result.dependencies.clear();
            result.dependencies.insert(operation);
            self.sink.add("ImportedPosition", (operation, scope.position))?;
            if let Some(previous) = scope.previous_operation.replace(operation) {
                self.sink.add("SourceNextOperation", (previous, operation))?;
            }
        }
        Ok(())
    }

    pub(super) fn finish_region_summary(&mut self, scope: &Scope) -> Result<(), OptimizeError> {
        self.write_summary(scope.key, &scope.summary, true)
    }

    fn write_summary(&mut self, key: Value, summary: &Summary, region: bool) -> Result<(), OptimizeError> {
        let names = if region {
            [
                "SourceRegionRequiresDevice",
                "SourceRegionSpeculatable",
                "SourceRegionReadOnly",
                "SourceRegionDuplicable",
                "SourceRegionWork",
                "SourceRegionCalls",
                "SourceRegionEnters",
                "SourceRegionDeviceCalls",
                "SourceRegionDeviceEnters",
            ]
        } else {
            [
                "SourceRequiresDevice",
                "SourceSpeculatable",
                "SourceReadOnly",
                "SourceDuplicable",
                "SourceWork",
                "SourceSummaryCalls",
                "SourceSummaryEnters",
                "SourceSummaryDeviceCalls",
                "SourceSummaryDeviceEnters",
            ]
        };
        for (name, flag) in
            names[..4].iter().zip([summary.device, summary.pure, summary.readonly, summary.duplicate])
        {
            self.sink.set(name, key, flag)?;
        }
        self.sink.set(names[4], key, summary.work)?;
        for binding in &summary.reads {
            self.sink.add(
                if region { "SourceRegionReadsBinding" } else { "SourceReadsBinding" },
                (key, i64::from(binding.set), i64::from(binding.binding)),
            )?;
        }
        for (name, values) in names[5..].iter().zip([
            &summary.calls,
            &summary.regions,
            &summary.device_calls,
            &summary.device_regions,
        ]) {
            for &value in values {
                self.sink.add(name, (key, value))?;
            }
        }
        Ok(())
    }
}
