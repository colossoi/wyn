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
use crate::{BindingRef, SymbolId};

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
    /// A 2D float texture resource bound at (set, binding). Carried on an
    /// entry-point param of type `texture2d`.
    Texture {
        set: u32,
        binding: u32,
    },
    /// A sampler resource bound at (set, binding). Carried on an
    /// entry-point param of type `sampler`.
    Sampler {
        set: u32,
        binding: u32,
    },
    /// A 2D storage image resource bound at (set, binding). Carried on
    /// an entry-point param of type `storage_image`. The `format` pins
    /// the on-GPU pixel format; the `access` declares whether the
    /// shader writes, reads, or both; the `size` picks the host-side
    /// resolution policy (defaults to `SameAsWindow`). The same
    /// `(set, binding)` may be declared as `Texture { ... }` in a
    /// sibling pipeline — the host allocates one wgpu texture and
    /// binds it via two views.
    StorageImage {
        set: u32,
        binding: u32,
        format: crate::pipeline_descriptor::StorageImageFormat,
        access: StorageAccess,
        size: crate::pipeline_descriptor::StorageTextureSize,
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
    fn has_texture(&self) -> bool;
    fn has_sampler(&self) -> bool;
    fn has_storage_image(&self) -> bool;
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
    fn has_texture(&self) -> bool {
        self.has(|a| matches!(a, Attribute::Texture { .. }))
    }
    fn has_sampler(&self) -> bool {
        self.has(|a| matches!(a, Attribute::Sampler { .. }))
    }
    fn has_storage_image(&self) -> bool {
        self.has(|a| matches!(a, Attribute::StorageImage { .. }))
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

/// Auto-allocated storage-buffer binding(s) for a single compute-entry
/// param. Each param produces exactly one record; the `kind`
/// discriminates plain `[]T` view params (one buffer) from
/// tuple-of-views params (one buffer per tuple field).
#[derive(Debug, Clone, PartialEq)]
pub struct EntryParamBinding {
    /// The body-level symbol this binding describes.
    pub param_sym: SymbolId,
    pub kind: EntryParamBindingKind,
}

/// What shape of storage allocation a param received. `elem_bytes`
/// is recorded at binding-allocation time alongside `elem_ty` so
/// downstream consumers (dispatch sizing, buffer length) never
/// re-derive it from the type.
#[derive(Debug, Clone, PartialEq)]
pub enum EntryParamBindingKind {
    /// Plain `[]T` view param: one storage buffer at `binding`.
    Single {
        binding: BindingRef,
        elem_ty: Type,
        elem_bytes: u32,
    },
    /// Tuple-of-views param: one storage buffer per tuple field,
    /// indexed by field position.
    TupleOfViews(Vec<TupleFieldBinding>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct TupleFieldBinding {
    pub binding: BindingRef,
    pub elem_ty: Type,
    pub elem_bytes: u32,
}

impl EntryParamBinding {
    /// Number of distinct `(set, binding)` slots this param consumed.
    pub fn buffer_count(&self) -> u32 {
        match &self.kind {
            EntryParamBindingKind::Single { .. } => 1,
            EntryParamBindingKind::TupleOfViews(fields) => fields.len() as u32,
        }
    }

    /// Highest binding number this param consumed. For `Single`, the
    /// only binding; for `TupleOfViews`, the last field's (the slots
    /// are allocated sequentially).
    pub fn max_binding(&self) -> BindingRef {
        match &self.kind {
            EntryParamBindingKind::Single { binding, .. } => *binding,
            EntryParamBindingKind::TupleOfViews(fields) => {
                fields.last().expect("tuple-of-views with zero fields").binding
            }
        }
    }

    /// First `(binding, elem_ty, elem_bytes)` for this param —
    /// the only buffer for `Single`, the field-0 buffer for
    /// `TupleOfViews`. Callers that need to size a dispatch from the
    /// param's outer length use this; tuple fields share the outer
    /// length by construction.
    pub fn first_buffer(&self) -> (BindingRef, &Type, u32) {
        match &self.kind {
            EntryParamBindingKind::Single {
                binding,
                elem_ty,
                elem_bytes,
            } => (*binding, elem_ty, *elem_bytes),
            EntryParamBindingKind::TupleOfViews(fields) => {
                let f = fields.first().expect("tuple-of-views with zero fields");
                (f.binding, &f.elem_ty, f.elem_bytes)
            }
        }
    }
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
    /// Per-body-param auto-storage binding. Same length as the lambda
    /// parameter list (one slot per body param, in declaration order);
    /// `None` for params that don't take a storage buffer (scalars,
    /// uniforms, push-constant routed values, builtins, etc.). Indexing
    /// in lockstep with body params is the contract — consumers should
    /// `zip` rather than rebuild a sym→binding map.
    pub param_bindings: Vec<Option<EntryParamBinding>>,
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
    pub binding: BindingRef,
    pub role: StorageRole,
    /// The element type stored at each index of the buffer.
    pub elem_ty: Type,
    /// Sizing policy for a compiler-managed buffer whose length isn't a
    /// host-supplied input (e.g. a gather intermediate). `None` for ordinary
    /// inputs/outputs, which the runtime sizes from host data or dispatch.
    pub length: Option<crate::pipeline_descriptor::BufferLen>,
}
