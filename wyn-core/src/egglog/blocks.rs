//! Control-flow scaffold. Scalar instructions are opaque payloads outside egglog.
use crate::egglog::data::{
    Array, BlockId, BodyId, BufferId, DispatchId, EntryId, ExprId, GridId, OperationId, ParameterId,
};
use crate::pipeline_descriptor::DispatchSize;
use crate::types::Type;
use std::collections::BTreeSet;
use wyn_base::IdArena;

#[derive(Clone, Debug)]
pub struct BlockData {
    /// The entry block owning this block. Functions are adornments of entries.
    pub function: BlockId,
    /// Structured control boundaries supplied by block construction.
    pub control: Option<Control>,
    pub interface: Option<Function>,
    pub parameters: Vec<String>,
    pub body: BodyId,
    pub exit: Exit,
}

#[derive(Clone, Copy, Debug)]
pub enum Control {
    Selection {
        merge: BlockId,
    },
    Loop {
        merge: BlockId,
        continuing: BlockId,
    },
}

#[derive(Clone, Debug)]
pub struct Function {
    pub name: String,
    pub kind: FunctionKind,
    pub results: usize,
    pub blocks: Vec<BlockId>,
}

#[derive(Clone, Debug)]
pub enum FunctionKind {
    Entry(EntryId),
    Host,
    Device,
    Kernel([u32; 3]),
}

#[derive(Clone, Debug)]
pub struct Edge {
    pub target: BlockId,
    /// An opaque tuple of arguments, evaluated in the source block.
    pub arguments: BodyId,
}

#[derive(Clone, Debug)]
pub enum Exit {
    Return(BodyId),
    Jump(Edge),
    Branch {
        condition: BodyId,
        yes: Edge,
        no: Edge,
    },
}

#[derive(Clone, Debug, Default)]
pub struct BodyData {
    pub instructions: Vec<Instruction>,
    pub results: Vec<Value>,
}

/// Generated scalar instructions remain opaque. Source expressions
/// resolve through the parameter/result bindings of their enclosing invocation.
/// Array loads, stores, length and dimension operate on logical arrays, including
/// TLC's tuple-of-component-arrays representation; physical layout is deferred.
#[derive(Clone, Debug)]
pub enum Value {
    Local(String),
    Int(u32),
    Source(ExprId),
    Array(Array),
    Buffer(BufferId),
    Tuple(Vec<Value>),
    Field(Box<Value>, usize),
    Primitive(&'static str, Vec<Value>),
}

#[derive(Clone, Debug)]
pub enum Instruction {
    BindParameter(ParameterId, Value),
    BindExpression(ExprId, Value),
    BindResult(OperationId, Value),
    /// An ordinary source call, global evaluation, or index operation only.
    Evaluate(OperationId),
    Call {
        function: BlockId,
        arguments: Vec<Value>,
        results: Vec<String>,
    },
    Load {
        result: String,
        buffer: Value,
        index: Value,
    },
    Store {
        buffer: Value,
        index: Value,
        value: Value,
    },
    Allocate(BufferId),
    /// Execute this dispatch site and make its writes visible before continuing
    /// host control flow. A loop can execute a site more than once.
    Dispatch(DispatchId),
}

#[derive(Clone, Debug)]
pub struct GridData {
    /// Workgroup counts, not invocation counts. Scalar formulas stay opaque.
    pub groups: [Value; 3],
}

#[derive(Clone, Debug)]
pub struct DispatchData {
    /// Source entry and logical phase supplied by the relational plan.
    pub owner: EntryId,
    pub kernel: BlockId,
    pub grid: GridId,
    pub size: DispatchSize,
    pub dependencies: BTreeSet<DispatchId>,
    /// Physical readout of the plan's Access facts.
    pub reads: BTreeSet<BufferId>,
    pub writes: BTreeSet<BufferId>,
    /// Explicit scalar environment passed from the containing host scope.
    pub captures: Vec<ExprId>,
}

#[derive(Clone, Debug)]
pub struct BufferData {
    pub name: String,
    pub length: Value,
    /// Logical element type; physical packing and binding numbers are deferred.
    pub element: Type,
    pub storage: Storage,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Storage {
    Device,
    /// Invocation-local storage (including values returned by a device helper).
    Function,
    /// A source view, possibly selected by a host branch or call. Its aliasing
    /// and physical binding remain described by the source sidecar expression.
    External(ExprId),
    /// An unused result slot. It has no allocation or shader binding.
    Discarded,
}

impl Value {
    pub(super) fn local(name: &str) -> Self {
        Self::Local(name.into())
    }
    pub(super) fn field(self, index: usize) -> Self {
        Self::Field(Box::new(self), index)
    }
    pub(super) fn op(name: &'static str, args: impl IntoIterator<Item = Self>) -> Self {
        Self::Primitive(name, args.into_iter().collect())
    }
}

impl Function {
    pub(super) fn insert(
        mut self,
        parameters: Vec<String>,
        blocks: &mut IdArena<BlockId, BlockData>,
        bodies: &mut IdArena<BodyId, BodyData>,
    ) -> BlockId {
        let id = blocks.alloc_id();
        self.blocks.push(id);
        let body = bodies.alloc(BodyData::default());
        let returns = bodies.alloc(BodyData::default());
        blocks.insert(
            id,
            BlockData {
                function: id,
                control: None,
                interface: Some(self),
                parameters,
                body,
                exit: Exit::Return(returns),
            },
        );
        id
    }
}
