//! Local flags and evaluation roots accumulated by the TLC walk.
//! The shared evaluation graph propagates calls, region entries and buffer reads.
use super::{Import, Scope};
use crate::egglog::OptimizeError;
use crate::{LookupMap, LookupSet};
use egglog_engine::{RawValues, Value, Write};

#[derive(Clone)]
pub(super) struct Summary {
    pub device: bool,
    pub pure: bool,
    pub readonly: bool,
    pub duplicate: bool,
    pub work: i64,
    pub evaluations: LookupSet<Value>,
}

impl Default for Summary {
    fn default() -> Self {
        Self {
            device: false,
            pure: true,
            readonly: true,
            duplicate: true,
            work: 0,
            evaluations: LookupSet::default(),
        }
    }
}

impl Summary {
    pub(super) fn use_value(&mut self, other: &Self) {
        self.device |= other.device;
    }

    pub(super) fn evaluate(&mut self, other: &Self) {
        self.use_value(other);
        self.pure &= other.pure;
        self.readonly &= other.readonly;
        self.duplicate &= other.duplicate;
        // Consumers cap inline queries; placement needs the unsaturated sum.
        self.work = self.work.saturating_add(other.work);
        self.evaluations.extend(&other.evaluations);
    }
}

#[derive(Default)]
pub(super) struct Summaries {
    pub values: LookupMap<Value, Summary>,
    pub operations: LookupMap<Value, Value>,
    pub regions_with_operations: LookupSet<Value>,
    pub lengths: LookupMap<Value, Value>,
    pub owners: LookupMap<Value, Value>,
}

impl Import<'_, '_, '_, '_> {
    pub(super) fn use_summary(&mut self, owner: Value, value: Value, evaluate: bool) {
        let child = self.summaries.values.get(&value).cloned().unwrap_or_default();
        let summary = self.summaries.values.entry(owner).or_default();
        if evaluate {
            summary.evaluate(&child);
        } else {
            summary.use_value(&child);
        }
    }

    pub(super) fn free_reference(&mut self, target: Value, scope: Value) -> Result<(), OptimizeError> {
        let unowned = self.sink.add("RegionId", -1i64)?;
        let owner = self.summaries.owners.get(&target).copied().unwrap_or(unowned);
        self.sink.add("SourceLexicalUse", (scope, target, owner))?;
        Ok(())
    }

    pub(super) fn finish_summary(&mut self, value: Value, scope: &mut Scope) -> Result<(), OptimizeError> {
        self.summaries.owners.insert(value, scope.key);
        let summary = self.summaries.values.get(&value).cloned().unwrap_or_default();
        self.sink.set("SourceReadOnly", value, summary.readonly)?;
        // Region-local properties were accumulated from each node once. Device
        // requirements additionally follow references to already evaluated values.
        scope.summary.device |= summary.device;
        if let Some(&operation) = self.summaries.operations.get(&value) {
            let role = if self.summaries.lengths.contains_key(&value) { "LengthUse" } else { "Other" };
            let role = self.sink.add(role, RawValues(vec![]))?;
            self.sink.set("SourceOperandRole", operation, role)?;
            self.write_summary(value, &summary, false)?;
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
            ]
        } else {
            [
                "SourceRequiresDevice",
                "SourceSpeculatable",
                "SourceReadOnly",
                "SourceDuplicable",
                "SourceWork",
            ]
        };
        for (name, flag) in
            names[..4].iter().zip([summary.device, summary.pure, summary.readonly, summary.duplicate])
        {
            self.sink.set(name, key, flag)?;
        }
        self.sink.set(names[4], key, summary.work)?;
        for &value in &summary.evaluations {
            self.sink.add(
                if region { "FlowRegionEvaluation" } else { "FlowEvaluation" },
                (key, value),
            )?;
        }
        Ok(())
    }
}
