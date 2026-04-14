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
//!
//! For transitional compatibility, `ast` re-exports these names so
//! existing `interface::Attribute` / `interface::EntryDecl` call sites keep working.

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
    /// Used for parallelization decisions. Ignored on non-arrays or statically sized arrays.
    SizeHint(u32),
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
    pub body: Expression,
}

// ---------------------------------------------------------------------------
// Resource declarations (uniforms / storage buffers)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct UniformDecl {
    pub name: String,
    pub ty: Type,     // Uniforms always have an explicit type
    pub set: u32,     // Descriptor set number (default 0)
    pub binding: u32, // Explicit binding number (required)
}

#[derive(Debug, Clone, PartialEq)]
pub struct StorageDecl {
    pub name: String,
    pub ty: Type,     // Storage buffers have an explicit type (usually runtime-sized array)
    pub set: u32,     // Descriptor set number
    pub binding: u32, // Binding number within the set
    pub layout: StorageLayout, // Memory layout (std430, std140)
    pub access: StorageAccess, // Access mode (read, write, readwrite)
}

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
