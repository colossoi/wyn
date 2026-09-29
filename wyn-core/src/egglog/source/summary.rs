//! Transient summaries accumulated by the TLC walk. These contain no executable
//! expressions. Only operation/region summaries enter structural fixed points.
use super::{Import, Scope};
use crate::egglog::OptimizeError;
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
        self.work = (self.work + other.work).min(65);
        self.reads.extend(&other.reads);
        self.calls.extend(&other.calls);
        self.regions.extend(&other.regions);
    }
}

#[derive(Default)]
pub(super) struct Summaries {
    pub values: LookupMap<Value, Summary>,
    pub uses: LookupMap<Value, LookupSet<Value>>,
    pub reads: Vec<(Value, Value)>,
    pub operations: LookupMap<Value, Value>,
    pub inputs: LookupSet<(Value, Value)>,
    pub captures: LookupSet<(Value, Value)>,
    pub regions_with_operations: LookupSet<Value>,
    pub canonical: LookupMap<Value, Value>,
    pub lengths: LookupMap<Value, Value>,
    pub fields: LookupMap<(Value, i64), Value>,
    pub owners: LookupMap<Value, Value>,
    pub parents: LookupMap<Value, Value>,
    pub free: LookupMap<Value, LookupSet<Value>>,
}

impl Import<'_, '_, '_, '_> {
    pub(super) fn use_value(&mut self, owner: Value, value: Value) {
        self.summaries.uses.entry(owner).or_default().insert(value);
    }

    /// Publish operation operands once; scalar dependencies retain their shared
    /// DAG rather than expanding a transitive operand set at every source node.
    pub(super) fn finish_uses(&mut self) -> Result<(), OptimizeError> {
        for (&value, children) in &self.summaries.uses {
            if self.summaries.canonical.contains_key(&value) {
                continue;
            }
            if let Some(&operation) = self.summaries.operations.get(&value) {
                for &child in children {
                    if self.summaries.inputs.contains(&(operation, child))
                        && !self.summaries.captures.contains(&(operation, child))
                    {
                        continue;
                    }
                    self.sink.add("SourceOperand", (operation, child))?;
                }
            } else {
                for &child in children {
                    self.sink.add("SourceComputedUse", (value, child))?;
                }
            }
        }
        // Read footprints are source-walk summaries, not an egglog traversal
        // through every scalar expression for each collective.
        for &(operation, root) in &self.summaries.reads {
            let mut pending = vec![root];
            let mut seen = LookupSet::default();
            while let Some(value) = pending.pop() {
                if !seen.insert(value) {
                    continue;
                }
                self.sink.add("SourceReadCandidate", (operation, value))?;
                pending.extend(self.summaries.uses.get(&value).into_iter().flatten().copied());
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
            self.summaries.free.entry(scope).or_default().insert(target);
            if let Some(summary) = self.summaries.values.get(&target) {
                for &before in &summary.dependencies {
                    self.sink.add("SourceRegionDependency", (scope, before))?;
                }
            }
            let Some(&parent) = self.summaries.parents.get(&scope) else {
                break;
            };
            scope = parent;
        }
        Ok(())
    }

    pub(super) fn alias_summary(&mut self, value: Value, target: Value) -> Result<(), OptimizeError> {
        let target = self.summaries.canonical.get(&target).copied().unwrap_or(target);
        self.summaries.canonical.insert(value, target);
        self.sink.add("SourceAlias", (value, target))?;
        Ok(())
    }

    pub(super) fn projection_summary(
        &mut self,
        value: Value,
        tuple: Value,
        index: i64,
    ) -> Result<bool, OptimizeError> {
        let tuple = self.summaries.canonical.get(&tuple).copied().unwrap_or(tuple);
        if let Some(&field) = self.summaries.fields.get(&(tuple, index)) {
            self.alias_summary(value, field)?;
            let dependencies =
                self.summaries.values.get(&field).map(|s| s.dependencies.clone()).unwrap_or_default();
            self.summaries.values.entry(value).or_default().dependencies = dependencies;
            return Ok(true);
        }
        Ok(false)
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
                self.use_value(owner, value);
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
        }
        Ok(())
    }

    pub(super) fn finish_region_summary(&mut self, scope: &Scope) -> Result<(), OptimizeError> {
        self.write_summary(scope.key, &scope.summary, true)
    }

    fn write_summary(&mut self, key: Value, summary: &Summary, region: bool) -> Result<(), OptimizeError> {
        let names = if region {
            [
                "SourceRegionDevice",
                "SourceRegionPure",
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
                "SourceDevice",
                "SourcePure",
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
