//! Structural memory dataflow between TLC and lowering. A graph traversal reads
//! operand references directly. Consumers that interpret an operation also read
//! its definition (for example, the scalar opcode or a parallel result clause).

pub mod call_graph;

use crate::ast::Span;
use crate::types::{Diet, Type};
use crate::LookupMap;
use slotmap::{new_key_type, SlotMap};
use wyn_graph::{ReplacementError, ReplacementForest};

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
pub struct Parameter<T = Type, A = Diet> {
    pub ty: T,
    pub diet: A,
}

#[derive(Clone, Debug)]
pub struct Region<T = Type, A = Diet> {
    pub parameters: Vec<Parameter<T, A>>,
    /// Executable occurrences in evaluation order.
    pub operations: Vec<OperationId>,
    pub yields: Vec<ValueRef>,
}

impl<T, A> Default for Region<T, A> {
    fn default() -> Self {
        Self {
            parameters: Vec::new(),
            operations: Vec::new(),
            yields: Vec::new(),
        }
    }
}

/// A nested control, lane, reducer or combiner body. Its owning operation
/// defines the role of each body and binds its incoming arguments in order.
#[derive(Clone, Debug)]
pub struct RegionEdge {
    pub role: BodyRole,
    pub region: RegionId,
    pub arguments: Vec<ValueRef>,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum BodyRole {
    Control(usize),
    Reducer(usize),
    Combiner(usize),
}

#[derive(Clone, Copy, Debug)]
pub enum UseSite {
    Input(OperationId, usize),
    Argument(OperationId, BodyRole, usize),
    Yield(RegionId, usize),
}

pub struct ForwardedCopies {
    pub values: LookupMap<ValueRef, ValueRef>,
    pub uses: Vec<(UseSite, ValueRef)>,
}

/// A direct call carries the ID resolved at import, including through let bindings.
#[derive(Clone, Debug)]
pub enum Definition<D> {
    Call(FunctionId),
    /// Forwards its single operand. Evaluating this occurrence has no effect.
    Copy,
    Other(D),
}

/// Value and region dependencies belong in the structural fields so graph
/// passes can follow them without interpreting the operation definition.
#[derive(Clone, Debug)]
pub struct Operation<D, T = Type> {
    pub definition: Definition<D>,
    /// Ordered operand occurrences. For `a - a` both inputs carry the same reference.
    pub inputs: Vec<ValueRef>,
    pub result_types: Vec<T>,
    pub bodies: Vec<RegionEdge>,
    pub span: Span,
}

/// Function, region and operation IDs belong to this program.
/// Each attached region is owned by one function or one operation.
#[derive(Clone, Debug)]
pub struct MemoryIr<D, T = Type, A = Diet> {
    pub functions: SlotMap<FunctionId, Function>,
    pub entries: Vec<FunctionId>,
    pub regions: SlotMap<RegionId, Region<T, A>>,
    pub operations: SlotMap<OperationId, Operation<D, T>>,
}

impl<D, T, A> Default for MemoryIr<D, T, A> {
    fn default() -> Self {
        Self {
            functions: SlotMap::with_key(),
            entries: Vec::new(),
            regions: SlotMap::with_key(),
            operations: SlotMap::with_key(),
        }
    }
}

impl<D, T, A> MemoryIr<D, T, A> {
    /// Append an occurrence to its owning region and return its identity.
    ///
    /// # Panics
    /// Panics if `region` is not a live key in this program.
    pub fn append(&mut self, region: RegionId, operation: Operation<D, T>) -> OperationId {
        let body = &mut self.regions[region];
        let id = self.operations.insert(operation);
        body.operations.push(id);
        id
    }

    /// Forward a batch of pure copies, including captures, yields and loop carries.
    /// Returns the terminal substitutions for metadata that refers to the same values.
    pub fn forward_copies(&mut self) -> Result<ForwardedCopies, ForwardError> {
        let mut replacements = ReplacementForest::new();
        for (id, operation) in &self.operations {
            if matches!(operation.definition, Definition::Copy) {
                let ([input], [_]) = (&operation.inputs[..], &operation.result_types[..]) else {
                    return Err(ForwardError::CopySignature(id));
                };
                replacements.replace(ValueRef::Result(id, 0), *input)?;
            }
        }
        let values = replacements.into_map();
        let mut uses = Vec::new();
        if values.is_empty() {
            return Ok(ForwardedCopies { values, uses });
        }
        let mut redirect = |site, value: &mut ValueRef| {
            if let Some(&source) = values.get(value) {
                *value = source;
                uses.push((site, source));
            }
        };
        self.operations.retain(|id, op| {
            if matches!(op.definition, Definition::Copy) {
                return false;
            }
            for (i, value) in op.inputs.iter_mut().enumerate() {
                redirect(UseSite::Input(id, i), value);
            }
            for edge in &mut op.bodies {
                for (i, value) in edge.arguments.iter_mut().enumerate() {
                    redirect(UseSite::Argument(id, edge.role, i), value);
                }
            }
            true
        });
        for (id, region) in &mut self.regions {
            for (i, value) in region.yields.iter_mut().enumerate() {
                redirect(UseSite::Yield(id, i), value);
            }
            region.operations.retain(|id| self.operations.contains_key(*id));
        }
        Ok(ForwardedCopies { values, uses })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ForwardError {
    #[error("copy {0:?} requires one operand and one result")]
    CopySignature(OperationId),
    #[error(transparent)]
    Replacement(#[from] ReplacementError<ValueRef>),
}
