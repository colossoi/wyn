//! Egglog encoding for the backend-independent fusion fact sink.
use super::{InputSite, Kind, Operation, Role, Sink};
use crate::egglog::data::{Array, ExprId, OperationId, RegionId, TypeId};
use crate::egglog::SCHEMA;

pub(in crate::egglog) struct Egglog {
    pub(in crate::egglog) text: String,
    next_dependency: usize,
}
impl Egglog {
    pub(in crate::egglog) fn new() -> Self {
        Self {
            text: format!("{}\n(let $dep-0_ (NoDependency))\n", SCHEMA),
            next_dependency: 1,
        }
    }
    fn emit(&mut self, text: std::fmt::Arguments<'_>) {
        self.text.push_str(&format!("{text}\n"));
    }
    fn node(&mut self, text: std::fmt::Arguments<'_>) -> usize {
        let id = self.next_dependency;
        self.next_dependency += 1;
        self.emit(format_args!("(let $dep-{id}_ {text})"));
        id
    }
}
fn group(id: OperationId) -> String {
    format!("(Group {})", id.egglog())
}
fn source(id: OperationId) -> String {
    format!("(Source {})", id.egglog())
}
fn dependencies(values: &[usize]) -> String {
    values.iter().rev().fold("(NoDependencies)".to_owned(), |tail, id| {
        format!("(MoreDependencies $dep-{id}_ {tail})")
    })
}
fn domain(array: &Array) -> String {
    match array {
        Array::Value(v) => format!("(DomainValue {})", v.as_u32()),
        Array::Literal(xs) => format!(
            "(DomainLiteral (vec-of {}))",
            xs.iter().map(|v| v.as_u32().to_string()).collect::<Vec<_>>().join(" ")
        ),
        Array::Range { start, len, step } => format!(
            "(DomainRange {} {} {})",
            start.as_u32(),
            len.as_u32(),
            step.map_or(-1, |v| v.as_u32() as i64)
        ),
        Array::Zip(xs) => format!(
            "(DomainZip (vec-of {}))",
            xs.iter().map(domain).collect::<Vec<_>>().join(" ")
        ),
    }
}
impl Sink for Egglog {
    type Dependency = usize;
    fn operation(&mut self, id: OperationId, region: RegionId, fact: Operation) {
        let (g, p) = (group(id), source(id));
        self.emit(format_args!("(Current {g} {p}) (Member {g} {}) (set (Owner {p}) {}) (set (Scope {p}) {}) (set (IterationDomain {p}) (SourceDomain {}))", id.egglog(), id.egglog(), region.egglog(), id.egglog()));
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
            self.emit(format_args!("(set ({name} {g}) (set-empty))"));
        }
        match fact.kind {
            Kind::Screma => self.emit(format_args!("(ScremaPlan {p})")),
            Kind::Filter => self.emit(format_args!("(FilterPlan {p})")),
            Kind::Element | Kind::Other => {}
        }
        for (name, yes) in [
            ("SafePlan", fact.safe),
            ("MovablePlan", fact.movable),
            ("ElementPlan", fact.element_consumer),
        ] {
            if yes {
                self.emit(format_args!("({name} {p})"));
            }
        }
        self.emit(format_args!("(set (DemandLimit {p}) {})", fact.demand_limit));
        for (name, yes) in [
            ("HasScans", fact.scans > 0),
            ("HasReductions", fact.reductions > 0),
            ("HasArrayOutputs", fact.arrays > 0),
            ("PrePure", fact.pre_projectable),
            ("PostPure", fact.post_projectable),
            ("PredicatePure", fact.predicate_projectable),
        ] {
            self.emit(format_args!("(set ({name} {p}) {yes})"));
        }
    }
    fn usage(&mut self, producer: OperationId, consumer: OperationId, role: Role, internal: bool) {
        let role = match role {
            Role::Input => "Input",
            Role::Other => "Other",
            Role::Argument => "Argument",
            Role::Length => "Length",
        };
        let site = if internal {
            format!("(To {} ({role}))", group(consumer))
        } else {
            format!("(Outside {} ({role}))", consumer.egglog())
        };
        self.emit(format_args!(
            "(GroupUseFact {} {site}) (OriginalUseFact {} {} ({role}))",
            group(producer),
            group(producer),
            consumer.egglog()
        ));
    }
    fn observed(&mut self, producer: OperationId) {
        self.emit(format_args!("(GroupUseFact {} (Returned))", group(producer)));
    }
    fn dependency(&mut self, producer: OperationId, consumer: OperationId) {
        self.emit(format_args!(
            "(GroupEdge {} {})",
            group(producer),
            group(consumer)
        ));
    }
    fn effect_member(&mut self, gate: usize, operation: OperationId) {
        self.emit(format_args!("(EffectMember {gate} {})", group(operation)));
    }
    fn effect_wait(&mut self, operation: OperationId, gate: usize) {
        self.emit(format_args!("(EffectWait {} {gate})", group(operation)));
    }
    fn input(&mut self, operation: OperationId, site: InputSite) {
        let site = match site {
            InputSite::Direct(op) => format!("(Direct {})", op.egglog()),
            InputSite::Indirect(op, token) => format!("(Indirect {} {token})", op.egglog()),
            InputSite::External(token) => format!("(External {token})"),
        };
        self.emit(format_args!("(InputFact {} {site})", group(operation)));
    }
    fn domain(&mut self, operation: OperationId, array: &Array, fixed: Option<u64>) {
        let key = fixed.map_or_else(|| domain(array), |n| format!("(FixedDomain {n})"));
        self.emit(format_args!(
            "(union (SourceDomain {}) {key})",
            operation.egglog()
        ));
    }
    fn domain_source(&mut self, operation: OperationId, producer: OperationId) {
        self.emit(format_args!(
            "(union (SourceDomain {}) (SourceDomain {}))",
            operation.egglog(),
            producer.egglog()
        ));
    }
    fn read_resource(&mut self, operation: OperationId, resource: ExprId) {
        self.emit(format_args!(
            "(ReadResource {} {})",
            group(operation),
            resource.as_u32()
        ));
    }
    fn write_resource(&mut self, operation: OperationId, resource: ExprId) {
        self.emit(format_args!(
            "(WriteResource {} {})",
            group(operation),
            resource.as_u32()
        ));
    }
    fn stream(&mut self, producer: OperationId, consumer: OperationId, slices: &[(ExprId, ExprId)]) {
        let chain = slices.iter().rev().fold("(WholeInput)".to_owned(), |tail, (a, b)| {
            format!("(InputSlice {} {} {tail})", a.as_u32(), b.as_u32())
        });
        self.emit(format_args!(
            "(GroupStream {} {}) (set (StreamSlices {} {}) (set-of {chain}))",
            group(producer),
            group(consumer),
            group(producer),
            group(consumer)
        ));
    }
    fn blocked_stream(&mut self, producer: OperationId, consumer: OperationId) {
        self.emit(format_args!(
            "(set (StreamBlocked {} {}) true)",
            group(producer),
            group(consumer)
        ));
    }
    fn length(&mut self, producer: OperationId, consumer: OperationId, ty: TypeId, direct: bool) {
        if direct {
            self.emit(format_args!(
                "(set (ValidLengths {}) (set-of (At {} (Length))))",
                group(producer),
                consumer.egglog()
            ));
        }
        self.emit(format_args!(
            "(set (LengthTypes {}) (set-of {}))",
            group(producer),
            ty.as_u32()
        ));
    }
    fn indexed(&mut self, producer: OperationId, consumer: OperationId) {
        self.emit(format_args!(
            "(set (IndexedUses {}) (set-of (At {} (Argument))))",
            group(producer),
            consumer.egglog()
        ));
    }
    fn index_address(&mut self, producer: OperationId, consumer: OperationId) {
        self.emit(format_args!(
            "(set (IndexAddressUses {}) (set-of (At {} (Argument))))",
            group(producer),
            consumer.egglog()
        ));
    }
    fn independent(&mut self) -> usize {
        0
    }
    fn scan(&mut self, operation: OperationId) -> usize {
        self.node(format_args!("(ScanDependency {})", operation.egglog()))
    }
    fn output(&mut self, operation: OperationId, slot: usize) -> usize {
        self.node(format_args!("(OutputDependency {} {slot})", operation.egglog()))
    }
    fn tuple(&mut self, fields: &[usize]) -> usize {
        if fields.iter().all(|v| *v == 0) {
            return 0;
        }
        self.node(format_args!("(TupleDependency {})", dependencies(fields)))
    }
    fn field(&mut self, value: usize, index: usize) -> usize {
        if value == 0 {
            return 0;
        }
        self.node(format_args!("(FieldDependency $dep-{value}_ {index})"))
    }
    fn all(&mut self, values: &[usize]) -> usize {
        if values.iter().all(|v| *v == 0) {
            return 0;
        }
        self.node(format_args!("(AllDependencies {})", dependencies(values)))
    }
    fn choice(&mut self, condition: usize, yes: usize, no: usize) -> usize {
        if condition == 0 && yes == 0 && no == 0 {
            return 0;
        }
        self.node(format_args!(
            "(ChoiceDependency $dep-{condition}_ $dep-{yes}_ $dep-{no}_)"
        ))
    }
    fn output_dependency(&mut self, operation: OperationId, slot: usize, value: usize) {
        self.emit(format_args!(
            "(union (OutputDependency {} {slot}) $dep-{value}_)",
            operation.egglog()
        ));
    }
    fn collective_dependency(&mut self, operation: OperationId, value: usize) {
        self.emit(format_args!(
            "(CollectiveDependency {} $dep-{value}_)",
            operation.egglog()
        ));
    }
}

#[cfg(test)]
#[path = "egglog_tests.rs"]
mod tests;
