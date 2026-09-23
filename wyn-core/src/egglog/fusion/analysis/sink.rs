//! Typed import boundary. Implementations decide where facts are stored.
use super::Role;
use crate::egglog::data::{Array, ExprId, OperationId, RegionId, TypeId};
use egglog_engine::Error;

pub(super) enum Kind {
    Other,
    Screma,
    Filter,
    Element,
}

/// One operation's local properties, passed directly to the sink.
pub(in crate::egglog) struct Operation {
    pub(super) kind: Kind,
    pub(super) scans: usize,
    pub(super) reductions: usize,
    pub(super) arrays: usize,
    pub(super) pre_projectable: bool,
    pub(super) post_projectable: bool,
    pub(super) predicate_projectable: bool,
    pub(super) safe: bool,
    pub(super) movable: bool,
    pub(super) element_consumer: bool,
    pub(super) demand_limit: usize,
}

pub(in crate::egglog) enum InputSite {
    Direct(OperationId),
    Indirect(OperationId, u64),
    External(u64),
}

pub(in crate::egglog) trait Sink {
    /// A sink-owned handle; analysis never stores the dependency graph itself.
    type Dependency: Copy;
    fn operation(&mut self, id: OperationId, region: RegionId, fact: Operation) -> Result<(), Error>;
    fn scalar_region(&mut self, region: RegionId, operations: &[OperationId]) -> Result<(), Error>;
    fn scalar_operation(&mut self, operation: OperationId, regions: &[RegionId]) -> Result<(), Error>;
    fn scalar_read(&mut self, operation: OperationId) -> Result<(), Error>;
    fn usage(
        &mut self,
        producer: OperationId,
        consumer: OperationId,
        role: Role,
        internal: bool,
    ) -> Result<(), Error>;
    fn observed(&mut self, producer: OperationId) -> Result<(), Error>;
    fn dependency(&mut self, producer: OperationId, consumer: OperationId) -> Result<(), Error>;
    fn effect_member(&mut self, gate: usize, operation: OperationId) -> Result<(), Error>;
    fn effect_wait(&mut self, operation: OperationId, gate: usize) -> Result<(), Error>;
    fn input(&mut self, operation: OperationId, site: InputSite) -> Result<(), Error>;
    fn domain(&mut self, operation: OperationId, array: &Array, fixed: Option<u64>) -> Result<(), Error>;
    fn domain_source(&mut self, operation: OperationId, producer: OperationId) -> Result<(), Error>;
    fn read_resource(&mut self, operation: OperationId, resource: ExprId) -> Result<(), Error>;
    fn write_resource(&mut self, operation: OperationId, resource: ExprId) -> Result<(), Error>;
    fn stream(
        &mut self,
        producer: OperationId,
        consumer: OperationId,
        slices: &[(ExprId, ExprId)],
    ) -> Result<(), Error>;
    fn blocked_stream(&mut self, producer: OperationId, consumer: OperationId) -> Result<(), Error>;
    fn length(
        &mut self,
        producer: OperationId,
        consumer: OperationId,
        ty: TypeId,
        direct: bool,
    ) -> Result<(), Error>;
    fn indexed(&mut self, producer: OperationId, consumer: OperationId) -> Result<(), Error>;
    fn index_address(&mut self, producer: OperationId, consumer: OperationId) -> Result<(), Error>;

    // Dependency structure only: no scalar instructions, literals, or arithmetic.
    /// No dependence on a scan result; this can still be a runtime value.
    fn independent(&mut self) -> Self::Dependency;
    fn scan(&mut self, operation: OperationId) -> Result<Self::Dependency, Error>;
    fn output(&mut self, operation: OperationId, slot: usize) -> Result<Self::Dependency, Error>;
    fn tuple(&mut self, fields: &[Self::Dependency]) -> Result<Self::Dependency, Error>;
    fn field(&mut self, value: Self::Dependency, index: usize) -> Result<Self::Dependency, Error>;
    fn all(&mut self, values: &[Self::Dependency]) -> Result<Self::Dependency, Error>;
    fn choice(
        &mut self,
        condition: Self::Dependency,
        yes: Self::Dependency,
        no: Self::Dependency,
    ) -> Result<Self::Dependency, Error>;
    fn output_dependency(
        &mut self,
        operation: OperationId,
        slot: usize,
        value: Self::Dependency,
    ) -> Result<(), Error>;
    fn collective_dependency(
        &mut self,
        operation: OperationId,
        value: Self::Dependency,
    ) -> Result<(), Error>;
}
