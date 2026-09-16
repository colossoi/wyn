//! Complete program structure retained outside the egglog fusion graph.

use super::blocks::{BlockData, BodyData, BufferData, DispatchData, GridData};
use crate::{ast, builtins, interface, pipeline_descriptor, types};
use std::collections::BTreeSet;
use wyn_base::IdArena;
use wyn_module_graph::PackageId;

macro_rules! ids {
    ($($name:ident),* $(,)?) => {$(
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name(u32);

        impl From<u32> for $name {
            fn from(value: u32) -> Self { Self(value) }
        }

        impl $name {
            pub const fn as_u32(self) -> u32 { self.0 }

        }
    )*};
}

ids! {
    ProgramId, SymbolId, TypeId, ExprId, OperationId, RegionId, ParameterId,
    OriginId, DefinitionId, EntryId, EntryParamId, InputBoundId, BuiltinId,
    ExternId, BucketShapeId, BlockId, BodyId, BufferId, DispatchId, GridId,
}

impl RegionId {
    /// Render this scope's opaque identity in the fusion graph.
    pub fn egglog(self) -> String {
        format!("(RegionId {})", self.0)
    }
}

impl OperationId {
    /// Render this execution's opaque identity in the fusion graph.
    pub fn egglog(self) -> String {
        format!("(OperationId {})", self.0)
    }
}

/// Every collection of associated data is an arena of structs. Lookup maps
/// used during conversion are temporary and do not escape into this sidecar.
#[derive(Clone, Debug, Default)]
pub struct AssociatedData {
    pub blocks: IdArena<BlockId, BlockData>,
    pub bodies: IdArena<BodyId, BodyData>,
    pub buffers: IdArena<BufferId, BufferData>,
    pub dispatches: IdArena<DispatchId, DispatchData>,
    pub grids: IdArena<GridId, GridData>,
    pub programs: IdArena<ProgramId, ProgramData>,
    pub symbols: IdArena<SymbolId, SymbolData>,
    pub types: IdArena<TypeId, TypeData>,
    pub expressions: IdArena<ExprId, ExprData>,
    pub operations: IdArena<OperationId, OperationData>,
    pub regions: IdArena<RegionId, RegionData>,
    pub parameters: IdArena<ParameterId, ParameterData>,
    pub origins: IdArena<OriginId, OriginData>,
    pub definitions: IdArena<DefinitionId, DefinitionData>,
    pub entries: IdArena<EntryId, EntryData>,
    pub entry_params: IdArena<EntryParamId, EntryParamData>,
    pub input_bounds: IdArena<InputBoundId, InputBoundData>,
    pub builtins: IdArena<BuiltinId, BuiltinData>,
    pub externs: IdArena<ExternId, ExternData>,
    pub bucket_shapes: IdArena<BucketShapeId, BucketShapeData>,
}

#[derive(Clone, Debug)]
pub struct ProgramData {
    /// First unused compiler-allocated storage binding at the TLC boundary.
    pub next_auto_storage_binding: u32,
}

#[derive(Clone, Debug)]
pub struct SymbolData {
    pub source: crate::SymbolId,
    pub name: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TypeData {
    pub ty: types::Type,
}

/// Sidecar values are interned by content, independently of diagnostic metadata.
/// This shares computation syntax, not runtime values across invocations or
/// loop iterations. Parameters and operation results carry binding identities.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ExprData {
    pub ty: TypeId,
    pub kind: ExprKind,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ExprKind {
    Global(SymbolId),
    Parameter(ParameterId),
    Builtin(BuiltinId),
    BinOp(String),
    UnOp(String),
    /// Only statically known scalar operators/catalog builtins use PureApp.
    PureApp {
        function: ExprId,
        args: Vec<ExprId>,
    },
    Lambda(RegionId),
    Closure {
        code: SymbolId,
        param_count: usize,
        captures: Vec<ExprId>,
    },
    Int(String),
    FloatBits(u32),
    Bool(bool),
    Unit,
    Coerce(ExprId),
    If {
        condition: ExprId,
        then_value: ExprId,
        else_value: ExprId,
    },
    Array(Array),
    Tuple(Vec<ExprId>),
    Project {
        tuple: ExprId,
        index: usize,
    },
    Vector(Vec<ExprId>),
    Extern(ExternId),
    /// A value dependency on one particular execution.
    OperationResult(OperationId),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Array {
    Value(ExprId),
    Zip(Vec<Array>),
    Literal(Vec<ExprId>),
    Range {
        start: ExprId,
        len: ExprId,
        step: Option<ExprId>,
    },
}

/// Diagnostic provenance attaches to values, without tracking TLC node IDs.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct OriginData {
    pub span: ast::Span,
    pub expression: ExprId,
    pub definition: DefinitionId,
}

#[derive(Clone, Debug)]
pub struct ParameterData {
    pub symbol: SymbolId,
    pub ty: TypeId,
    pub region: RegionId,
}

/// A lexical execution scope with unordered operation membership. Dependencies
/// and effect constraints determine execution order, never iteration over members.
/// Nested regions execute only when invoked.
#[derive(Clone, Debug)]
pub struct RegionData {
    pub definition: DefinitionId,
    pub parent: Option<RegionId>,
    pub parameters: Vec<ParameterId>,
    pub members: BTreeSet<OperationId>,
    pub results: Vec<ExprId>,
}

#[derive(Clone, Debug)]
pub struct OperationData {
    pub region: RegionId,
    /// Original TLC position, used only to orient conservative effect edges.
    /// Independent, movable operations need not execute in this order.
    pub source_position: usize,
    pub ty: TypeId,
    pub span: ast::Span,
    pub kind: OperationKind,
}

#[derive(Clone, Debug)]
pub enum OperationKind {
    Call {
        function: ExprId,
        args: Vec<ExprId>,
    },
    EvalGlobal(SymbolId),
    If {
        condition: ExprId,
        then_region: RegionId,
        else_region: RegionId,
    },
    /// The header binds the accumulator and (for counted loops) iteration
    /// parameter. While headers return the condition. Counted headers run
    /// their destructuring bindings only on actual iterations. The body
    /// returns the next accumulator and can reference header-local values.
    Loop {
        init: ExprId,
        header: RegionId,
        kind: LoopKind,
        body: RegionId,
    },
    Index {
        array: ExprId,
        index: ExprId,
    },
    Screma {
        form: ScremaForm,
        inputs: Vec<Array>,
        ownership: Vec<types::SoacOwnership>,
    },
    Filter {
        /// Element transformation before predicate evaluation and compaction.
        map: SoacBody,
        body: SoacBody,
        inputs: Vec<Array>,
        ownership: types::SoacOwnership,
    },
    Scatter {
        destination: Place,
        body: SoacBody,
        inputs: Vec<Array>,
    },
    BucketScatter {
        destination: Place,
        body: SoacBody,
        inputs: Vec<Array>,
        shape: BucketShapeId,
    },
    ReduceByIndex {
        destination: Place,
        /// Maps input elements to two logical results: destination index, value.
        map: SoacBody,
        body: SoacBody,
        neutral: ExprId,
        inputs: Vec<Array>,
    },
}

#[derive(Clone, Debug)]
pub enum LoopKind {
    For(ExprId),
    ForRange(ExprId),
    While,
}

#[derive(Clone, Debug)]
pub struct Place {
    pub value: ExprId,
    pub elem_ty: TypeId,
}

#[derive(Clone, Debug)]
pub enum SoacBody {
    /// Select whole logical arguments, without flattening tuple-valued elements.
    Route {
        parameters: Vec<TypeId>,
        indices: Vec<usize>,
    },
    /// Both bodies receive the same arguments; results are left then right.
    Parallel {
        left: Box<SoacBody>,
        right: Box<SoacBody>,
    },
    /// Apply `first` to the inputs, then pass its logical results to `then`.
    /// Each body retains its own capture arguments.
    Compose {
        first: Box<SoacBody>,
        then: Box<SoacBody>,
    },
    /// Apply a parameterized region to logical inputs followed by captures.
    /// Naming/export metadata, when present, belongs to DefinitionData.
    Apply {
        region: RegionId,
        parameters: Vec<TypeId>,
        results: Vec<TypeId>,
        captures: Vec<ExprId>,
    },
    Identity(Vec<TypeId>),
}

#[derive(Clone, Debug)]
pub struct Scan {
    pub operator: SoacBody,
    pub neutral: Vec<ExprId>,
}

#[derive(Clone, Debug)]
pub struct Reduction {
    pub operator: SoacBody,
    pub neutral: Vec<ExprId>,
    pub commutative: bool,
}

/// pre returns scan inputs, reduction inputs, then mapped values. post receives
/// scan results and mapped values. Final results are reductions, then post arrays.
#[derive(Clone, Debug)]
pub struct ScremaForm {
    pub pre: SoacBody,
    pub scans: Vec<Scan>,
    pub reductions: Vec<Reduction>,
    pub post: SoacBody,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DefinitionKind {
    Function,
    LiftedLambda,
    Entry(EntryId),
}

#[derive(Clone, Debug)]
pub struct DefinitionData {
    pub symbol: SymbolId,
    pub package: Option<PackageId>,
    pub ty: TypeId,
    pub body: RegionId,
    pub kind: DefinitionKind,
    pub arity: usize,
    pub param_diets: Vec<types::Diet>,
    pub return_diet: types::Diet,
}

#[derive(Clone, Debug)]
pub struct EntryData {
    pub definition: DefinitionId,
    pub declaration: interface::EntryDecl,
}

/// Includes unbound parameters, preserving the original parameter positions.
#[derive(Clone, Debug)]
pub struct EntryParamData {
    pub entry: EntryId,
    pub position: usize,
    pub binding: Option<interface::EntryParamBinding>,
}

#[derive(Clone, Debug)]
pub struct InputBoundData {
    pub entry: EntryId,
    pub symbol: SymbolId,
    pub length: pipeline_descriptor::BufferLen,
}

#[derive(Clone, Debug)]
pub struct BuiltinData {
    pub builtin: builtins::BuiltinId,
    pub overload_idx: usize,
}

#[derive(Clone, Debug)]
pub struct ExternData {
    pub linkage_name: String,
}

#[derive(Clone, Debug)]
pub struct BucketShapeData {
    pub input_dimensions: Vec<Vec<u8>>,
    pub domain_rank: u8,
}
