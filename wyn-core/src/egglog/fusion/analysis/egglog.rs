//! Native insertion for the fusion fact sink. All writes share one update batch.
use super::{InputSite, Kind, Operation, Role, Sink};
use crate::egglog::data::{Array, ExprId, OperationId, RegionId, TypeId};
use egglog_engine::sort::{SetContainer, VecContainer};
use egglog_engine::{Core, Error, FullState, IntoValues, RawValues, Value, Write};

pub(in crate::egglog) struct Egglog<'a, 'db> {
    state: FullState<'a, 'db>,
    independent: Value,
}
impl<'a, 'db> Egglog<'a, 'db> {
    pub(in crate::egglog) fn new(mut state: FullState<'a, 'db>) -> Result<Self, Error> {
        let independent = state.add("NoDependency", RawValues(vec![]))?;
        Ok(Self { state, independent })
    }
    fn operation_key(&mut self, id: OperationId) -> Result<Value, Error> {
        self.state.add("OperationId", i64::from(id.as_u32()))
    }
    fn region_key(&mut self, id: RegionId) -> Result<Value, Error> {
        self.state.add("RegionId", i64::from(id.as_u32()))
    }
    fn group(&mut self, id: OperationId) -> Result<Value, Error> {
        let id = self.operation_key(id)?;
        self.state.add("Group", id)
    }
    fn set(
        &mut self,
        name: &str,
        key: impl IntoValues,
        values: impl IntoIterator<Item = Value>,
        do_rebuild: bool,
    ) -> Result<(), Error> {
        let values = self.state.register_container(SetContainer {
            data: values.into_iter().collect(),
            do_rebuild,
        });
        self.state.set(name, key, values)
    }
    fn dependencies(&mut self, values: &[Value]) -> Result<Value, Error> {
        let mut tail = self.state.add("NoDependencies", RawValues(vec![]))?;
        for &value in values.iter().rev() {
            tail = self.state.add("MoreDependencies", (value, tail))?;
        }
        Ok(tail)
    }
    fn domain_key(&mut self, array: &Array) -> Result<Value, Error> {
        match array {
            Array::Value(v) => self.state.add("DomainValue", i64::from(v.as_u32())),
            Array::Literal(xs) => {
                let data = xs.iter().map(|v| self.state.base_to_value(i64::from(v.as_u32()))).collect();
                let values = self.state.register_container(VecContainer {
                    data,
                    do_rebuild: false,
                });
                self.state.add("DomainLiteral", values)
            }
            Array::Range { start, len, step } => self.state.add(
                "DomainRange",
                (
                    i64::from(start.as_u32()),
                    i64::from(len.as_u32()),
                    step.map_or(-1, |v| i64::from(v.as_u32())),
                ),
            ),
            Array::Zip(xs) => {
                let data = xs.iter().map(|a| self.domain_key(a)).collect::<Result<_, _>>()?;
                let values = self.state.register_container(VecContainer {
                    data,
                    do_rebuild: true,
                });
                self.state.add("DomainZip", values)
            }
        }
    }
    fn indexed_use(
        &mut self,
        name: &str,
        producer: OperationId,
        consumer: OperationId,
    ) -> Result<(), Error> {
        let producer = self.group(producer)?;
        let consumer = self.operation_key(consumer)?;
        let role = self.state.add("Argument", RawValues(vec![]))?;
        let site = self.state.add("At", (consumer, role))?;
        self.set(name, producer, [site], true)
    }
}
impl Sink for Egglog<'_, '_> {
    type Dependency = Value;
    fn scalar_region(&mut self, region: RegionId, operations: &[OperationId]) -> Result<(), Error> {
        let region = self.region_key(region)?;
        let members = operations.iter().map(|&op| self.operation_key(op)).collect::<Result<Vec<_>, _>>()?;
        self.set("ScalarRegionMembers", region, members.iter().copied(), true)?;
        self.set("PureRegionMembers", region, [], true)?;
        self.set("ReadOnlyRegionMembers", region, [], true)?;
        for op in members {
            self.state.add("ScalarMember", (region, op))?;
        }
        Ok(())
    }
    fn scalar_operation(&mut self, operation: OperationId, regions: &[RegionId]) -> Result<(), Error> {
        let operation = self.operation_key(operation)?;
        let members = regions.iter().map(|&r| self.region_key(r)).collect::<Result<Vec<_>, _>>()?;
        self.set("ScalarCallees", operation, members.iter().copied(), true)?;
        self.set("PureCallees", operation, [], true)?;
        self.set("ReadOnlyCallees", operation, [], true)?;
        for region in members {
            self.state.add("ScalarCall", (operation, region))?;
        }
        Ok(())
    }
    fn scalar_read(&mut self, operation: OperationId) -> Result<(), Error> {
        let operation = self.operation_key(operation)?;
        self.state.add("ScalarRead", operation)?;
        Ok(())
    }
    fn operation(&mut self, id: OperationId, region: RegionId, fact: Operation) -> Result<(), Error> {
        let op = self.operation_key(id)?;
        let g = self.state.add("Group", op)?;
        let p = self.state.add("Source", op)?;
        let region = self.region_key(region)?;
        let domain = self.state.add("SourceDomain", op)?;
        self.state.add("Current", (g, p))?;
        self.state.add("Member", (g, op))?;
        self.state.set("Owner", p, op)?;
        self.state.set("Scope", p, region)?;
        self.state.set("IterationDomain", p, domain)?;
        for name in [
            "Members",
            "DirectInputs",
            "SelfUses",
            "GroupUsesOf",
            "GroupLengths",
            "ValidLengths",
            "LengthTypes",
            "GroupInputs",
            "InternalInputs",
            "IndexedUses",
            "IndexAddressUses",
        ] {
            self.set(name, g, [], name != "LengthTypes")?;
        }
        match fact.kind {
            Kind::Screma => {
                self.state.add("ScremaPlan", p)?;
            }
            Kind::Filter => {
                self.state.add("FilterPlan", p)?;
            }
            Kind::Element | Kind::Other => {}
        }
        for (name, yes) in [
            ("SafePlan", fact.safe),
            ("MovablePlan", fact.movable),
            ("ElementPlan", fact.element_consumer),
        ] {
            if yes {
                self.state.add(name, p)?;
            }
        }
        self.state.set("DemandLimit", p, fact.demand_limit as i64)?;
        self.state.set("ArrayOutputs", p, fact.arrays as i64)?;
        for (name, yes) in [
            ("HasScans", fact.scans > 0),
            ("HasReductions", fact.reductions > 0),
            ("HasArrayOutputs", fact.arrays > 0),
            ("PrePure", fact.pre_projectable),
            ("PostPure", fact.post_projectable),
            ("PredicatePure", fact.predicate_projectable),
        ] {
            self.state.set(name, p, yes)?;
        }
        Ok(())
    }
    fn usage(
        &mut self,
        producer: OperationId,
        consumer: OperationId,
        role: Role,
        internal: bool,
    ) -> Result<(), Error> {
        let role = self.state.add(
            match role {
                Role::Input => "Input",
                Role::Other => "Other",
                Role::Argument => "Argument",
                Role::Length => "Length",
            },
            RawValues(vec![]),
        )?;
        let producer = self.group(producer)?;
        let op = self.operation_key(consumer)?;
        let site = if internal {
            let consumer = self.group(consumer)?;
            self.state.add("To", (consumer, role))?
        } else {
            self.state.add("Outside", (op, role))?
        };
        self.state.add("GroupUseFact", (producer, site))?;
        self.state.add("OriginalUseFact", (producer, op, role))?;
        Ok(())
    }
    fn observed(&mut self, producer: OperationId) -> Result<(), Error> {
        let producer = self.group(producer)?;
        let site = self.state.add("Returned", RawValues(vec![]))?;
        self.state.add("GroupUseFact", (producer, site))?;
        Ok(())
    }
    fn dependency(&mut self, producer: OperationId, consumer: OperationId) -> Result<(), Error> {
        let producer = self.group(producer)?;
        let consumer = self.group(consumer)?;
        self.state.add("GroupEdge", (producer, consumer))?;
        Ok(())
    }
    fn effect_member(&mut self, gate: usize, operation: OperationId) -> Result<(), Error> {
        let operation = self.group(operation)?;
        self.state.add("EffectMember", (gate as i64, operation))?;
        Ok(())
    }
    fn effect_wait(&mut self, operation: OperationId, gate: usize) -> Result<(), Error> {
        let operation = self.group(operation)?;
        self.state.add("EffectWait", (operation, gate as i64))?;
        Ok(())
    }
    fn input(&mut self, operation: OperationId, site: InputSite) -> Result<(), Error> {
        let operation = self.group(operation)?;
        let site = match site {
            InputSite::Direct(op) => {
                let op = self.operation_key(op)?;
                self.state.add("Direct", op)?
            }
            InputSite::Indirect(op, token) => {
                let op = self.operation_key(op)?;
                self.state.add("Indirect", (op, token as i64))?
            }
            InputSite::External(token) => self.state.add("External", token as i64)?,
        };
        self.state.add("InputFact", (operation, site))?;
        Ok(())
    }
    fn domain(&mut self, operation: OperationId, array: &Array, fixed: Option<u64>) -> Result<(), Error> {
        let operation = self.operation_key(operation)?;
        let source = self.state.add("SourceDomain", operation)?;
        let key = if let Some(n) = fixed {
            self.state.add("FixedDomain", n as i64)?
        } else {
            self.domain_key(array)?
        };
        self.state.union(source, key)
    }
    fn domain_source(&mut self, operation: OperationId, producer: OperationId) -> Result<(), Error> {
        let operation = self.operation_key(operation)?;
        let producer = self.operation_key(producer)?;
        let a = self.state.add("SourceDomain", operation)?;
        let b = self.state.add("SourceDomain", producer)?;
        self.state.union(a, b)
    }
    fn read_resource(&mut self, operation: OperationId, resource: ExprId) -> Result<(), Error> {
        let operation = self.group(operation)?;
        self.state.add("ReadResource", (operation, i64::from(resource.as_u32())))?;
        Ok(())
    }
    fn write_resource(&mut self, operation: OperationId, resource: ExprId) -> Result<(), Error> {
        let operation = self.group(operation)?;
        self.state.add("WriteResource", (operation, i64::from(resource.as_u32())))?;
        Ok(())
    }
    fn stream(
        &mut self,
        producer: OperationId,
        consumer: OperationId,
        slices: &[(ExprId, ExprId)],
    ) -> Result<(), Error> {
        let producer = self.group(producer)?;
        let consumer = self.group(consumer)?;
        let mut chain = self.state.add("WholeInput", RawValues(vec![]))?;
        for &(a, b) in slices.iter().rev() {
            chain = self.state.add(
                "InputSlice",
                (i64::from(a.as_u32()), i64::from(b.as_u32()), chain),
            )?;
        }
        self.set("StreamSlices", (producer, consumer), [chain], true)
    }
    fn blocked_stream(&mut self, producer: OperationId, consumer: OperationId) -> Result<(), Error> {
        let producer = self.group(producer)?;
        let consumer = self.group(consumer)?;
        self.state.set("StreamBlocked", (producer, consumer), true)
    }
    fn length(
        &mut self,
        producer: OperationId,
        consumer: OperationId,
        ty: TypeId,
        direct: bool,
    ) -> Result<(), Error> {
        let producer = self.group(producer)?;
        if direct {
            let consumer = self.operation_key(consumer)?;
            let role = self.state.add("Length", RawValues(vec![]))?;
            let site = self.state.add("At", (consumer, role))?;
            self.set("ValidLengths", producer, [site], true)?;
        }
        let ty = self.state.base_to_value(i64::from(ty.as_u32()));
        self.set("LengthTypes", producer, [ty], false)
    }
    fn indexed(&mut self, producer: OperationId, consumer: OperationId) -> Result<(), Error> {
        self.indexed_use("IndexedUses", producer, consumer)
    }
    fn index_address(&mut self, producer: OperationId, consumer: OperationId) -> Result<(), Error> {
        self.indexed_use("IndexAddressUses", producer, consumer)
    }
    fn independent(&mut self) -> Value {
        self.independent
    }
    fn scan(&mut self, operation: OperationId) -> Result<Value, Error> {
        let operation = self.operation_key(operation)?;
        self.state.add("ScanDependency", operation)
    }
    fn output(&mut self, operation: OperationId, slot: usize) -> Result<Value, Error> {
        let operation = self.operation_key(operation)?;
        self.state.add("OutputDependency", (operation, slot as i64))
    }
    fn tuple(&mut self, fields: &[Value]) -> Result<Value, Error> {
        if fields.iter().all(|v| *v == self.independent) {
            return Ok(self.independent);
        }
        let fields = self.dependencies(fields)?;
        self.state.add("TupleDependency", fields)
    }
    fn field(&mut self, value: Value, index: usize) -> Result<Value, Error> {
        if value == self.independent {
            return Ok(value);
        }
        self.state.add("FieldDependency", (value, index as i64))
    }
    fn all(&mut self, values: &[Value]) -> Result<Value, Error> {
        if values.iter().all(|v| *v == self.independent) {
            return Ok(self.independent);
        }
        let values = self.dependencies(values)?;
        self.state.add("AllDependencies", values)
    }
    fn choice(&mut self, condition: Value, yes: Value, no: Value) -> Result<Value, Error> {
        if [condition, yes, no].iter().all(|v| *v == self.independent) {
            return Ok(self.independent);
        }
        self.state.add("ChoiceDependency", (condition, yes, no))
    }
    fn output_dependency(
        &mut self,
        operation: OperationId,
        slot: usize,
        value: Value,
    ) -> Result<(), Error> {
        let output = self.output(operation, slot)?;
        self.state.union(output, value)
    }
    fn collective_dependency(&mut self, operation: OperationId, value: Value) -> Result<(), Error> {
        let operation = self.operation_key(operation)?;
        self.state.add("CollectiveDependency", (operation, value))?;
        Ok(())
    }
}

#[cfg(test)]
#[path = "egglog_tests.rs"]
mod tests;
