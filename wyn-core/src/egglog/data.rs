//! Typed, append-only metadata for one imported program.

use crate::{ast, builtins, interface, pipeline_descriptor, tlc, types};
use wyn_base::IdArena;
use wyn_module_graph::PackageId;

macro_rules! ids {
    ($($name:ident => $constructor:literal),* $(,)?) => {$(
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name(u32);

        impl From<u32> for $name {
            fn from(value: u32) -> Self { Self(value) }
        }

        impl $name {
            pub const fn as_u32(self) -> u32 { self.0 }

            /// Render this sidecar ID as a typed egglog value.
            pub fn egglog(self) -> String {
                format!(concat!("(", $constructor, " {})"), self.0)
            }
        }
    )*};
}

ids! {
    ProgramId => "ProgramId",
    SymbolId => "SymbolId",
    TypeId => "TypeId",
    TermId => "TermId",
    DefinitionId => "DefinitionId",
    EntryId => "EntryId",
    EntryParamId => "EntryParamId",
    InputBoundId => "InputBoundId",
    BuiltinId => "BuiltinId",
    ExternId => "ExternId",
    BucketShapeId => "BucketShapeId",
}

impl TermId {
    /// The global egglog binding containing this source occurrence's expression.
    pub fn binding_name(self) -> String {
        format!("term-{}", self.0)
    }
}

/// Every collection of associated data is an arena of structs. Lookup maps
/// used during conversion are temporary and do not escape into this sidecar.
#[derive(Clone, Debug, Default)]
pub struct AssociatedData {
    pub programs: IdArena<ProgramId, ProgramData>,
    pub symbols: IdArena<SymbolId, SymbolData>,
    pub types: IdArena<TypeId, TypeData>,
    pub terms: IdArena<TermId, TermData>,
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

#[derive(Clone, Debug)]
pub struct TypeData {
    pub ty: types::Type,
}

/// Provenance is a relation, not part of pure expression identity. Multiple
/// source occurrences may therefore point to one egglog equivalence class.
#[derive(Clone, Debug)]
pub struct TermData {
    pub source: tlc::TermId,
    pub span: ast::Span,
    pub ty: TypeId,
    pub definition: DefinitionId,
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
    pub body: TermId,
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
