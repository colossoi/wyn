//! Structural memory dataflow between TLC and lowering. A graph traversal reads
//! operand references directly. Consumers that interpret an operation also read
//! its definition (for example, the scalar opcode or a parallel result clause).

pub mod call_graph;

use crate::ast::Span;
use crate::types::{Diet, Type};
use slotmap::{new_key_type, SlotMap};

new_key_type! {
    /// Identity of a callable definition. All aliases of that callable share this ID.
    pub struct FunctionId;
    pub struct RegionId;
    pub struct OperationId;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Function {
    Defined(RegionId),
    External,
}

/// Identifies a region parameter or operation result by its origin and position.
/// Equal references denote the same value in a given execution of that origin.
/// A loop body binds its parameters anew on each iteration.
///
/// Import a plain let-binding by associating the new name with the existing
/// reference. In `let x = a in let y = x in (a, y)`, both tuple operands carry
/// the reference for `a`. For an array view, this identifies the view. Memory
/// effects determine which writes a subsequent read can observe.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ValueRef {
    Parameter(RegionId, usize),
    Result(OperationId, usize),
}

#[derive(Clone, Debug)]
pub struct Parameter {
    pub ty: Type,
    pub diet: Diet,
}

#[derive(Clone, Debug, Default)]
pub struct Region {
    pub parameters: Vec<Parameter>,
    /// Executable occurrences in evaluation order.
    pub operations: Vec<OperationId>,
    pub yields: Vec<ValueRef>,
}

/// A nested control, lane, reducer or combiner body. Its owning operation
/// defines the role of each body and binds its incoming arguments in order.
#[derive(Clone, Debug)]
pub struct RegionEdge {
    pub region: RegionId,
    pub arguments: Vec<ValueRef>,
}

/// A direct call carries the ID resolved at import, including through let bindings.
#[derive(Clone, Debug)]
pub enum Definition<D> {
    Call(FunctionId),
    Other(D),
}

/// Value and region dependencies belong in the structural fields so graph
/// passes can follow them without interpreting the operation definition.
#[derive(Clone, Debug)]
pub struct Operation<D> {
    pub definition: Definition<D>,
    /// Ordered operand occurrences. For `a - a` both inputs carry the same reference.
    pub inputs: Vec<ValueRef>,
    pub result_types: Vec<Type>,
    pub bodies: Vec<RegionEdge>,
    pub span: Span,
}

/// Function, region and operation IDs belong to this program.
/// Each attached region is owned by one function or one operation.
#[derive(Clone, Debug)]
pub struct MemoryIr<D> {
    pub functions: SlotMap<FunctionId, Function>,
    pub entries: Vec<FunctionId>,
    pub regions: SlotMap<RegionId, Region>,
    pub operations: SlotMap<OperationId, Operation<D>>,
}

impl<D> Default for MemoryIr<D> {
    fn default() -> Self {
        Self {
            functions: SlotMap::with_key(),
            entries: Vec::new(),
            regions: SlotMap::with_key(),
            operations: SlotMap::with_key(),
        }
    }
}

impl<D> MemoryIr<D> {
    /// Append an occurrence to its owning region and return its identity.
    ///
    /// # Panics
    /// Panics if `region` is not a live key in this program.
    pub fn append(&mut self, region: RegionId, operation: Operation<D>) -> OperationId {
        let body = &mut self.regions[region];
        let id = self.operations.insert(operation);
        body.operations.push(id);
        id
    }
}
