//! Shader interface metadata: entry points and resource declarations.
//!
//! These types describe *what a compiled shader exposes to the host* —
//! entry points, their inputs/outputs, and the storage/uniform buffers
//! they bind. They're created during parsing but flow through every
//! downstream pass (TLC → EGIR → SSA → SPIR-V) as metadata, so they
//! don't belong in `ast` (which is meant for the syntactic AST proper).
//!
//! Later compiler passes (notably `parallelize`) may *add* to this
//! metadata — e.g. declaring an intermediate `partials` buffer on a
//! phase entry point. Having a single typed home for these declarations
//! gives backends one source of truth for the entry interface.

use crate::ast::{Expression, Pattern, Span};
use crate::types::Type;

// ---------------------------------------------------------------------------
// Shader-stage / parameter attributes
// ---------------------------------------------------------------------------

/// Attribute that can decorate an entry point, parameter, or type.
///
/// Covers shader-stage markers (`Vertex`/`Fragment`/`Compute`), resource
/// bindings (`Storage`/`Uniform`), IO decorations (`BuiltIn`/`Location`),
/// and compiler hints (`SizeHint`/`Linked`).
#[derive(Debug, Clone, PartialEq)]
pub enum Attribute {
    BuiltIn(spirv::BuiltIn),
    Location(u32),
    Vertex,
    Fragment,
    /// Compute shader entry point. Workgroup size is determined by the compiler.
    Compute,
    Uniform {
        set: u32,
        binding: u32,
    },
    Storage {
        set: u32,
        binding: u32,
        layout: StorageLayout,
        access: StorageAccess,
    },
    /// Hint for the expected size of a dynamic array (in elements).
    /// Used for parallelization decisions. Ignored on non-arrays or
    /// statically sized arrays. `NonZeroU32` encodes that
    /// `#[size_hint(0)]` is meaningless and rejected by the parser.
    SizeHint(std::num::NonZeroU32),
    /// Linked SPIR-V function - the string is the linkage name for spirv-link
    Linked(String),
}

impl Attribute {
    pub fn is_vertex(&self) -> bool {
        matches!(self, Attribute::Vertex)
    }
    pub fn is_fragment(&self) -> bool {
        matches!(self, Attribute::Fragment)
    }
    pub fn is_compute(&self) -> bool {
        matches!(self, Attribute::Compute)
    }
}

pub trait AttrExt {
    fn has<F: Fn(&Attribute) -> bool>(&self, pred: F) -> bool;
    fn first_builtin(&self) -> Option<spirv::BuiltIn>;
    fn first_location(&self) -> Option<u32>;
    fn has_uniform(&self) -> bool;
    fn has_storage(&self) -> bool;
}

impl AttrExt for [Attribute] {
    fn has<F: Fn(&Attribute) -> bool>(&self, pred: F) -> bool {
        self.iter().any(pred)
    }
    fn first_builtin(&self) -> Option<spirv::BuiltIn> {
        self.iter().find_map(|a| if let Attribute::BuiltIn(b) = a { Some(*b) } else { None })
    }
    fn first_location(&self) -> Option<u32> {
        self.iter().find_map(|a| if let Attribute::Location(l) = a { Some(*l) } else { None })
    }
    fn has_uniform(&self) -> bool {
        self.has(|a| matches!(a, Attribute::Uniform { .. }))
    }
    fn has_storage(&self) -> bool {
        self.has(|a| matches!(a, Attribute::Storage { .. }))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AttributedType {
    pub attributes: Vec<Attribute>,
    pub ty: Type,
}

// ---------------------------------------------------------------------------
// Entry point declarations
// ---------------------------------------------------------------------------

/// Output field for entry point declarations.
#[derive(Debug, Clone, PartialEq)]
pub struct EntryOutput {
    pub ty: Type,
    pub attribute: Option<Attribute>,
}

/// One auto-allocated storage-buffer binding slot for a compute entry
/// param. Tuple-of-views entry params expand to one slot per field.
#[derive(Debug, Clone, PartialEq)]
pub struct EntryBindingSlot {
    pub set: u32,
    pub binding: u32,
    /// The body-level symbol the slot is allocated for.
    pub param_sym: crate::SymbolId,
    /// For tuple-of-views entry params: which tuple field. `None` for
    /// plain view-array params.
    pub tuple_field: Option<usize>,
    /// Element type stored at each index of the buffer.
    pub elem_ty: Type,
}

/// Entry point declaration (vertex/fragment/compute shader).
#[derive(Debug, Clone, PartialEq)]
pub struct EntryDecl {
    pub entry_type: Attribute, // Attribute::Vertex, Attribute::Fragment, or Attribute::Compute
    pub name: String,
    pub name_span: Span,
    pub size_params: Vec<String>,  // Size type parameters: <[n], [m]>
    pub type_params: Vec<String>,  // Regular type parameters: <T, U>
    pub params: Vec<Pattern>,      // Input parameters as patterns
    pub outputs: Vec<EntryOutput>, // Output fields with optional attributes
    /// Compiler-introduced storage bindings (e.g. parallelize's partials/result
    /// intermediates). Parsers leave this empty; later passes fill it in so
    /// downstream stages have one source of truth for the entry interface.
    pub storage_bindings: Vec<StorageBindingDecl>,
    /// Auto-allocated bindings for this entry's view-typed params,
    /// computed once and consulted by every pass that cares (buffer
    /// specialization, parallelize sizing, EGIR conversion). Empty
    /// for non-compute entries and for entries before the populate
    /// pass has run.
    pub param_bindings: Vec<EntryBindingSlot>,
    pub body: Expression,
}

// ---------------------------------------------------------------------------
// Resource declarations (uniforms / storage buffers)
// ---------------------------------------------------------------------------

/// Memory layout for storage buffers.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum StorageLayout {
    #[default]
    Std430, // Default for SSBOs, tightly packed
    Std140, // More relaxed alignment, compatible with UBOs
}

/// Access mode for storage buffers.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum StorageAccess {
    ReadOnly,
    WriteOnly,
    #[default]
    ReadWrite,
}

/// Role a storage buffer plays in a compute entry point's interface.
///
/// Distinguishes user-visible reads/writes from compiler-introduced
/// intermediate buffers (e.g. the `partials` buffer threaded between the
/// two phases of a parallelized reduce). Intermediates are not source-level
/// outputs — conflating them with `EntryOutput` would muddle semantics.
#[derive(Debug, Clone, PartialEq)]
pub enum StorageRole {
    /// Entry reads from this buffer.
    Input,
    /// Entry writes the user-visible result to this buffer.
    Output,
    /// Compiler-introduced pipeline-staging buffer (read or written).
    Intermediate,
}

/// A storage-buffer binding the entry point touches, declared as first-class
/// interface metadata. Populated by compiler passes that introduce
/// bindings the source program didn't (e.g. `parallelize`).
#[derive(Debug, Clone, PartialEq)]
pub struct StorageBindingDecl {
    pub set: u32,
    pub binding: u32,
    pub role: StorageRole,
    /// The element type stored at each index of the buffer.
    pub elem_ty: Type,
}
