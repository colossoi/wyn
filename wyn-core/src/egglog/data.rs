//! Complete program structure retained outside the egglog fusion graph.
use crate::ast::Span;
use crate::builtins::{catalog, select::Selection};
use crate::host::BufferLen;
use crate::interface::{EntryDecl, EntryParamBinding};
use crate::types::{Diet, Type};
use std::collections::{BTreeMap, BTreeSet};
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
    ExternId, BucketShapeId, BlockId, BodyId, BufferId, DispatchId, GridId, PlacementId, OutputId,
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

/// Shared source structure, interned expressions, types, and metadata.
/// Pass states retain these arenas through scheduling and SSA lowering.
#[derive(Clone, Debug, Default)]
pub struct Ir {
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

/// Evaluate once at this control site, each time the site is reached. This is
/// deliberately separate from a globally interned expression's identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlacementData {
    pub before: PlacementSite,
    pub expression: ExprId,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum PlacementSite {
    Operation(OperationId),
    Expression(ExprId),
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
    pub ty: Type,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Evaluation {
    Lazy,
    Eager,
}

impl Ir {
    /// Value-choice facts shared by lazy conditionals and eager select. The
    /// evaluation mode must be respected by control-flow and motion analyses.
    pub(crate) fn conditional_value(&self, expression: ExprId) -> Option<(Selection<ExprId>, Evaluation)> {
        match &self.expressions[expression].kind {
            ExprKind::If {
                condition,
                then_value,
                else_value,
            } => Some((
                Selection {
                    condition: *condition,
                    yes: *then_value,
                    no: *else_value,
                },
                Evaluation::Lazy,
            )),
            ExprKind::PureApp { function, args } => {
                let ExprKind::Builtin(id) = self.expressions[*function].kind else {
                    return None;
                };
                let builtin = &self.builtins[id];
                if builtin.builtin != catalog().known().select || builtin.overload_idx != 0 {
                    return None;
                }
                Some((Selection::from_operands(args)?, Evaluation::Eager))
            }
            _ => None,
        }
    }
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
    pub span: Span,
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
    pub span: Span,
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
        /// Per-result permission to reuse an input slot; scheduling proves safety.
        reuse_inputs: Vec<Option<usize>>,
    },
    Filter {
        /// Element transformation before predicate evaluation and compaction.
        map: SoacBody,
        body: SoacBody,
        /// Element transformation evaluated only for selected elements.
        post: SoacBody,
        inputs: Vec<Array>,
        reuse_input: Option<usize>,
    },
    Scatter {
        destination: Place,
        /// Copy a value destination into fresh writable result storage.
        /// External storage destinations retain their in-place semantics.
        initialize: bool,
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

impl OperationKind {
    /// Scalar uses consume these producers' stored results rather than
    /// repeating their loop or collective computation.
    pub(super) fn has_stored_result(&self) -> bool {
        matches!(
            self,
            Self::Loop { .. }
                | Self::Screma { .. }
                | Self::Filter { .. }
                | Self::Scatter { .. }
                | Self::BucketScatter { .. }
                | Self::ReduceByIndex { .. }
        )
    }

    /// Resolve a direct scalar invocation without changing its argument bindings.
    pub(super) fn called_region(
        &self,
        data: &Ir,
        definitions: &BTreeMap<SymbolId, RegionId>,
    ) -> Option<RegionId> {
        match self {
            Self::Call { function, .. } => match data.expressions[*function].kind {
                ExprKind::Lambda(r) => Some(r),
                ExprKind::Global(s) | ExprKind::Closure { code: s, .. } => definitions.get(&s).copied(),
                _ => None,
            },
            Self::EvalGlobal(s) => definitions.get(s).copied(),
            _ => None,
        }
    }
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
    pub param_diets: Vec<Diet>,
    pub return_diet: Diet,
}

#[derive(Clone, Debug)]
pub struct EntryData {
    pub definition: DefinitionId,
    pub declaration: EntryDecl,
}

#[derive(Clone, Debug)]
pub struct OutputData {
    pub entry: EntryId,
    pub index: usize,
    pub expression: ExprId,
    /// Filled by the plan's OutputBacking relation.
    pub buffer: Option<BufferId>,
    /// Copy this expression into its backing in the entry's final kernel.
    pub copy: bool,
}

/// Includes unbound parameters, preserving the original parameter positions.
#[derive(Clone, Debug)]
pub struct EntryParamData {
    pub entry: EntryId,
    pub position: usize,
    pub binding: Option<EntryParamBinding>,
}

#[derive(Clone, Debug)]
pub struct InputBoundData {
    pub entry: EntryId,
    pub symbol: SymbolId,
    pub length: BufferLen,
}

#[derive(Clone, Debug)]
pub struct BuiltinData {
    pub builtin: crate::builtins::BuiltinId,
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

pub(super) fn intern_type(data: &mut Ir, value: Type) -> TypeId {
    if let Some((&id, _)) = data.types.iter().find(|(_, t)| t.ty == value) {
        return id;
    }
    data.types.alloc(TypeData { ty: value })
}
pub(super) fn intern_expr(data: &mut Ir, ty: TypeId, kind: ExprKind) -> ExprId {
    let value = ExprData { ty, kind };
    if let Some((&id, _)) = data.expressions.iter().find(|(_, e)| **e == value) {
        return id;
    }
    data.expressions.alloc(value)
}
pub(super) fn body_signature(body: &SoacBody) -> (Vec<TypeId>, Vec<TypeId>) {
    match body {
        SoacBody::Apply {
            parameters, results, ..
        } => (parameters.clone(), results.clone()),
        SoacBody::Identity(ts) => (ts.clone(), ts.clone()),
        SoacBody::Route { parameters, indices } => (
            parameters.clone(),
            indices.iter().map(|&i| parameters[i]).collect(),
        ),
        SoacBody::Compose { first, then } => (body_signature(first).0, body_signature(then).1),
        SoacBody::Parallel { left, right } => {
            let (p, mut r) = body_signature(left);
            r.extend(body_signature(right).1);
            (p, r)
        }
    }
}
pub(in crate::egglog) fn length_source(data: &Ir, kind: &OperationKind) -> Option<ExprId> {
    let OperationKind::Call { function, args } = kind else {
        return None;
    };
    let [array] = args.as_slice() else {
        return None;
    };
    let mut function = *function;
    while let ExprKind::Coerce(inner) = data.expressions[function].kind {
        function = inner;
    }
    let ExprKind::Builtin(id) = data.expressions[function].kind else {
        return None;
    };
    (data.builtins[id].builtin == catalog().known().length).then_some(*array)
}

/// Resolve structural aliases without rewriting the sidecar.
pub(super) fn value_source(data: &Ir, mut value: ExprId) -> ExprId {
    loop {
        match data.expressions[value].kind {
            ExprKind::Coerce(inner) => value = inner,
            ExprKind::Project { tuple, index } => {
                let tuple = value_source(data, tuple);
                let ExprKind::Tuple(fields) = &data.expressions[tuple].kind else {
                    return value;
                };
                value = fields[index];
            }
            _ => return value,
        }
    }
}

/// Recognize the shared slice builtin through type coercions.
pub(super) fn is_slice(data: &Ir, mut function: ExprId) -> bool {
    while let ExprKind::Coerce(inner) = data.expressions[function].kind {
        function = inner;
    }
    matches!(data.expressions[function].kind, ExprKind::Builtin(id)
        if data.builtins[id].builtin == catalog().known().slice)
}
