pub mod ast;
pub mod ast_renumber;
pub mod binding_layout;
pub mod builtins;
pub mod diags;
pub mod error;
pub mod flow;
pub mod interface;
pub mod lexer;
pub mod module_manager;
pub mod name_resolution;
pub mod op;
pub mod parser;
pub mod pattern;
pub mod scope;
pub mod ssa;
pub mod types;

// Re-export type_checker from its new location for backwards compatibility
pub use types::checker as type_checker;

pub mod ast_const_fold;
pub mod lowering_common;
pub mod name_registry;
pub mod tlc;

pub mod egir;
pub use egir::program::ResourceId;
/// Re-export of the pipeline descriptor format. Lives in its own
/// crate so host runtimes (e.g. `extra/viz`) can deserialize the
/// JSON without pulling in the whole compiler.
pub use wyn_pipeline_descriptor as pipeline_descriptor;
pub mod resolve_imports;
pub mod resolve_opens;
pub mod resolve_placeholders;
pub mod resolve_resources;
pub mod spirv;
pub mod structured;
pub mod wgsl;

#[cfg(test)]
mod integration_tests;

#[cfg(test)]
mod test_pipeline;

#[cfg(test)]
mod slice_range_tests;

use std::hash::Hash;
use std::marker::PhantomData;

use egir::from_tlc::ConvertError;
use egir::program::{AllocatedProgram, RawProgram, SemanticProgram};

use ast::{NodeCounter, NodeId};
use error::Result;
use polytype::TypeScheme;

// =============================================================================
// Collection aliases
// =============================================================================

/// Use for maps whose iteration order affects program output (binding
/// allocation, code emission order, etc.). Insertion order is stable
/// across compiles; `HashMap`'s randomized hasher is not.
pub type StableMap<K, V> = indexmap::IndexMap<K, V>;

/// Set companion to [`StableMap`]: insertion-order iteration.
pub type StableSet<T> = indexmap::IndexSet<T>;

/// Use for sets whose iteration follows the values' [`Ord`] ordering.
pub type SortedSet<T> = std::collections::BTreeSet<T>;

/// Use for maps consulted only via `get`/`contains_key`. Iteration
/// order doesn't escape into observable output, so `HashMap`'s
/// per-process random hash is fine — and we get the slightly faster
/// lookups in exchange.
pub type LookupMap<K, V> = std::collections::HashMap<K, V>;

/// Set companion to [`LookupMap`].
pub type LookupSet<T> = std::collections::HashSet<T>;

// =============================================================================
// Generic ID allocation
// =============================================================================

/// Generic counter for generating unique IDs.
///
/// The ID type must implement `From<u32>` to convert the raw counter value.
#[derive(Debug, Clone)]
pub struct IdSource<Id> {
    next_id: u32,
    _phantom: PhantomData<Id>,
}

impl<Id: From<u32>> IdSource<Id> {
    pub fn new() -> Self {
        IdSource {
            next_id: 0,
            _phantom: PhantomData,
        }
    }

    pub fn next_id(&mut self) -> Id {
        let raw = self.next_id;
        self.next_id = self.next_id.checked_add(1).expect("compiler ID space exhausted");
        let id = Id::from(raw);
        id
    }

    /// Read the next id without consuming it. Useful for "would-allocate"
    /// dry-runs: peek, attempt, commit via `next_id()` only on success.
    pub fn peek_id(&self) -> Id {
        Id::from(self.next_id)
    }
}

impl<Id: From<u32>> Default for IdSource<Id> {
    fn default() -> Self {
        Self::new()
    }
}

/// Append-only bidirectional interner for values with compiler-assigned IDs.
///
/// Equal values share one ID. The arena provides ID-to-value resolution while
/// the lookup map provides value-to-ID lookup and deduplication.
#[derive(Debug, Clone)]
pub struct Interner<Id, T> {
    arena: IdArena<Id, T>,
    by_value: LookupMap<T, Id>,
}

impl<Id, T> Interner<Id, T>
where
    Id: From<u32> + Copy + Eq + Hash,
{
    pub fn new() -> Self {
        Self {
            arena: IdArena::new(),
            by_value: LookupMap::new(),
        }
    }

    pub fn intern<Q>(&mut self, value: &Q) -> Id
    where
        T: std::borrow::Borrow<Q> + Clone + Eq + Hash,
        Q: Eq + Hash + ToOwned<Owned = T> + ?Sized,
    {
        if let Some(id) = self.by_value.get(value) {
            return *id;
        }
        let value = value.to_owned();
        let id = self.arena.alloc(value.clone());
        self.by_value.insert(value, id);
        id
    }

    pub fn get<Q>(&self, value: &Q) -> Option<Id>
    where
        T: std::borrow::Borrow<Q> + Eq + Hash,
        Q: Eq + Hash + ?Sized,
    {
        self.by_value.get(value).copied()
    }

    pub fn resolve(&self, id: Id) -> &T {
        self.arena.get(id).expect("interner received an ID it did not allocate")
    }

    pub fn resolve_cloned(&self, ids: impl IntoIterator<Item = Id>) -> Vec<T>
    where
        T: Clone,
    {
        ids.into_iter().map(|id| self.resolve(id).clone()).collect()
    }

    pub fn len(&self) -> usize {
        self.arena.len()
    }

    pub fn is_empty(&self) -> bool {
        self.arena.is_empty()
    }
}

impl<Id, T> Default for Interner<Id, T>
where
    Id: From<u32> + Copy + Eq + Hash,
{
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Symbol Table for TLC
// =============================================================================

/// Unique identifier for a symbol (variable, function, parameter).
/// After AST → TLC conversion, all variable references use SymbolIds
/// instead of strings, eliminating name resolution from later passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SymbolId(pub u32);

impl From<u32> for SymbolId {
    fn from(v: u32) -> Self {
        SymbolId(v)
    }
}

impl std::fmt::Display for SymbolId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "sym{}", self.0)
    }
}

/// Symbol table: maps SymbolId to original name (for errors/debugging).
pub type SymbolTable = IdArena<SymbolId, String>;

// =============================================================================
// Storage binding reference
// =============================================================================

/// A `(descriptor set, binding)` pair naming a host-runtime storage /
/// uniform / texture / sampler resource. Semantic EGIR graphs use
/// `ResourceId`; this pair remains the host ABI constraint and becomes the
/// resource identity again only in physical EGIR and SSA. Deliberately no
/// `Default` impl —
/// `BindingRef { set: 0, binding: 0 }` is a meaningful binding, and a
/// default value would silently mask construction bugs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BindingRef {
    pub set: u32,
    pub binding: u32,
}

impl BindingRef {
    pub const fn new(set: u32, binding: u32) -> Self {
        BindingRef { set, binding }
    }
}

impl std::fmt::Display for BindingRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "set={},binding={}", self.set, self.binding)
    }
}

/// Conservative read/write access to a compiler-internal resource.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ResourceAccess {
    Read,
    Write,
    ReadWrite,
}

impl ResourceAccess {
    pub fn merge(self, other: Self) -> Self {
        if self == other {
            self
        } else {
            Self::ReadWrite
        }
    }

    pub fn reads(self) -> bool {
        matches!(self, Self::Read | Self::ReadWrite)
    }

    pub fn writes(self) -> bool {
        matches!(self, Self::Write | Self::ReadWrite)
    }
}

/// Look up `sym`'s source name in `symbols`, or panic with a uniform
/// "internal compiler bug" message. Use this when downstream code
/// structurally requires that every `SymbolId` it sees was registered
/// by an earlier pass — the panic is the structural assertion, not a
/// placeholder. Call sites in `Result`-returning paths should prefer
/// `egir::from_tlc::symbol_name` (which propagates the same condition
/// as `ConvertError::Internal`).
pub fn symbol_name_or_bug(symbols: &SymbolTable, sym: SymbolId) -> &str {
    symbols.get(sym).map(String::as_str).unwrap_or_else(|| {
        panic!("BUG: symbol {sym:?} not in symbol table — registration invariant violated")
    })
}

/// Run a by-value transform over every element of a collection, in place.
/// Each element is owned while it passes through `f`, so `f` can move,
/// destructure, and rebuild it freely — the idiom for tree-restructuring
/// passes whose walkers are `Term -> Term`. The collection is empty only
/// for the duration of the call (`mem::take` leaves its `Default`).
pub fn map_in_place<C, T, F>(collection: &mut C, f: F)
where
    C: Default + Extend<T> + IntoIterator<Item = T>,
    F: FnMut(T) -> T,
{
    let items = std::mem::take(collection);
    collection.extend(items.into_iter().map(f));
}

/// Arena that allocates IDs and stores associated items.
///
/// Combines ID generation with storage, ensuring each item gets a unique ID.
/// Uses StableMap for deterministic iteration order (insertion order).
#[derive(Debug, Clone)]
pub struct IdArena<Id, T> {
    source: IdSource<Id>,
    items: StableMap<Id, T>,
}

impl<Id: From<u32> + Copy + Eq + Hash, T> IdArena<Id, T> {
    pub fn new() -> Self {
        IdArena {
            source: IdSource::new(),
            items: StableMap::new(),
        }
    }

    /// Allocate a new ID and store the item.
    pub fn alloc(&mut self, item: T) -> Id {
        let id = self.source.next_id();
        self.items.insert(id, item);
        id
    }

    /// Allocate a new ID without storing anything yet.
    /// Use `insert` later to store the item.
    pub fn alloc_id(&mut self) -> Id {
        self.source.next_id()
    }

    /// Insert an item with a pre-allocated ID.
    /// Panics if the ID is already in use.
    pub fn insert(&mut self, id: Id, item: T) {
        let old = self.items.insert(id, item);
        assert!(old.is_none(), "IdArena::insert called with duplicate ID");
    }

    /// Get an item by ID.
    pub fn get(&self, id: Id) -> Option<&T> {
        self.items.get(&id)
    }

    /// Get a mutable reference to an item by ID.
    pub fn get_mut(&mut self, id: Id) -> Option<&mut T> {
        self.items.get_mut(&id)
    }

    /// Iterate over all (id, item) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&Id, &T)> {
        self.items.iter()
    }

    /// Iterate over the IDs that currently name stored items.
    pub fn ids(&self) -> impl Iterator<Item = Id> + '_ {
        self.items.keys().copied()
    }

    /// Iterate over all items (without IDs).
    pub fn values(&self) -> impl Iterator<Item = &T> {
        self.items.values()
    }

    /// Number of items in the arena.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Check if the arena is empty.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

impl<Id: From<u32> + Copy + Eq + Hash, T> Default for IdArena<Id, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Id: From<u32> + Copy + Eq + Hash, T> std::ops::Index<Id> for IdArena<Id, T> {
    type Output = T;

    fn index(&self, id: Id) -> &Self::Output {
        &self.items[&id]
    }
}

impl<Id: From<u32> + Copy + Eq + Hash, T> std::ops::IndexMut<Id> for IdArena<Id, T> {
    fn index_mut(&mut self, id: Id) -> &mut Self::Output {
        &mut self.items[&id]
    }
}

impl<Id: From<u32> + Copy + Eq + Hash, T> IntoIterator for IdArena<Id, T> {
    type Item = (Id, T);
    type IntoIter = indexmap::map::IntoIter<Id, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.into_iter()
    }
}

impl<'a, Id: From<u32> + Copy + Eq + Hash, T> IntoIterator for &'a IdArena<Id, T> {
    type Item = (&'a Id, &'a T);
    type IntoIter = indexmap::map::Iter<'a, Id, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.iter()
    }
}

impl<'a, Id: From<u32> + Copy + Eq + Hash, T> IntoIterator for &'a mut IdArena<Id, T> {
    type Item = (&'a Id, &'a mut T);
    type IntoIter = indexmap::map::IterMut<'a, Id, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.iter_mut()
    }
}

// Re-export key types for the public API
pub use ast::TypeName;
pub use polytype::Context as PolytypeContext;
pub type TypeTable = LookupMap<NodeId, TypeScheme<TypeName>>;

// =============================================================================
// Typestate Compiler Pipeline
// =============================================================================
//
// The compiler uses a typestate pattern where each struct represents a stage.
// Methods consume `self` and return the next stage, enforcing valid ordering.
//
//   let (mut node_counter, mut module_manager) = init_compiler();
//   let parsed = Compiler::parse(source, &mut node_counter)?;
//
// FrontEnd (AST) stages:
//     parsed.resolve_imports(...)                  -> Parsed (CLI only)
//       .elaborate_modules(...)                    -> Parsed
//       .resolve(&module_manager)                  -> Resolved
//       .fold_ast_constants()                      -> AstConstFoldedEarly
//       .type_check(&mut module_manager)           -> TypeChecked
//
// TLC stages (typed AST → semantic input):
//       .to_tlc(&module_manager, fill_holes)       -> TlcTransformed
//       .pin_entry_buffers()                       -> TlcBuffersPinned
//       .validate_ownership()                      -> TlcOwnershipValidated
//       .partial_eval()                            -> TlcPartialEvaled
//       .normalize_soacs()                         -> TlcSoaNormalized
//       .monomorphize()                            -> TlcMonomorphized
//       .rep_specialize()                          -> TlcRepSpecialized
//       .inline_small()                            -> TlcSmallInlined
//       .force_inline_soac_helpers()               -> TlcSoacHelpersInlined
//       .renormalize_inlined_soa()                 -> TlcInlinedSoaNormalized
//       .canonicalize_conditional_producers()      -> TlcConditionalProducersCanonicalized
//       .normalize_soacs_to_anf()                  -> TlcSoacsAnfNormalized
//       .float_runtime_index_nested_producers()    -> TlcRuntimeIndexProducersFloated
//       .defunctionalize()                         -> TlcDefunctionalized
//       .fold_generated_lambdas()                  -> TlcGeneratedLambdasFolded
//       .apply_ownership()                         -> TlcOwnershipApplied
//       .filter_reachable()                        -> TlcReachable
//       .infer_input_slice_bounds()                -> TlcInputSliceBoundsInferred
//       .to_egraph()                               -> EgirRaw
//
// EGIR stages:
//       .realize_outputs()                         -> EgirOutputsRealized
//       .segment()                                 -> EgirSegmented
//       .optimize()                                -> EgirOptimized
//       .allocate()?                               -> EgirAllocated
//       .plan(profile)                             -> EgirPlanned
//       .lower_to_ssa()                            -> SsaConverted
//
// Backend:
//       .lower() | .lower_wgsl()
//
// Tests should prefer the `compile_thru_*` helpers below, which subsume
// the chain up to a milestone and centralize updates as new passes land.

/// Build a fresh `(NodeCounter, ModuleManager)` pair. The node counter is
/// shared between user code parsing and prelude loading so all NodeIds
/// stay unique. The module manager comes pre-loaded with the parsed
/// prelude.
pub fn init_compiler() -> Result<(NodeCounter, module_manager::ModuleManager)> {
    let mut node_counter = NodeCounter::new();
    let prelude = module_manager::ModuleManager::create_prelude(&mut node_counter)?;
    let module_manager = module_manager::ModuleManager::from_prelude(prelude);
    Ok((node_counter, module_manager))
}

/// Build a `(NodeCounter, ModuleManager)` pair from an already-elaborated
/// prelude. Faster than `init_compiler()` when callers can amortize the
/// prelude across multiple compiles.
pub fn init_compiler_from_prelude(
    prelude: module_manager::PreElaboratedPrelude,
    node_counter: NodeCounter,
) -> (NodeCounter, module_manager::ModuleManager) {
    let module_manager = module_manager::ModuleManager::from_prelude(prelude);
    (node_counter, module_manager)
}

// =============================================================================
// Compiler entry point
// =============================================================================

/// Entry point for the compiler. Use `Compiler::parse()` to start the pipeline.
pub struct Compiler;

impl Compiler {
    /// Parse source code into an AST using the provided node counter.
    /// This ensures NodeIds don't collide with prelude modules.
    pub fn parse(source: &str, node_counter: &mut NodeCounter) -> Result<Parsed> {
        let tokens = lexer::tokenize(source).map_err(|e| err_parse!("{}", e))?;
        let mut parser = parser::Parser::new(tokens, node_counter);
        let ast = parser.parse()?;
        Ok(Parsed(FrontInner { ast }))
    }
}

// =============================================================================
// FrontEnd stages (AST-based)
// =============================================================================

/// Shared payload for the front-end AST states (`Parsed`, `Desugared`,
/// `Resolved`, `AstConstFoldedEarly`). Each state is a newtype wrapping this
/// inner; transitions within the group just unwrap / re-wrap.
struct FrontInner {
    ast: ast::Program,
}

/// Source has been parsed into an AST
pub struct Parsed(FrontInner);

impl Parsed {
    /// Resolve `import "path"` declarations against the filesystem,
    /// loading each referenced file relative to `base_dir`, parsing
    /// it, recursively resolving its own imports, and prepending its
    /// declarations into this program. The `Import` nodes themselves
    /// are removed.
    ///
    /// Cycle / re-import safety: a file is loaded at most once per
    /// compilation (keyed by canonical path). Diamond imports work
    /// fine; cycles are silently broken (only the first encounter
    /// loads decls).
    ///
    /// Path resolution: `import "foo"` looks for `<base_dir>/foo.wyn`.
    /// Imports inside `foo.wyn` resolve relative to `foo.wyn`'s
    /// directory.
    pub fn resolve_imports(
        mut self,
        base_dir: &std::path::Path,
        node_counter: &mut NodeCounter,
    ) -> Result<Self> {
        self.0.ast.declarations = resolve_imports::run(self.0.ast.declarations, base_dir, node_counter)?;
        Ok(self)
    }

    /// Elaborate inline module declarations from the parsed program.
    /// This registers modules with the module_manager so they're available during resolution,
    /// then removes the Module declarations from the AST (they've been copied to module_manager).
    /// Should be called before desugar() if the program contains module definitions.
    pub fn elaborate_modules(
        mut self,
        module_manager: &mut module_manager::ModuleManager,
        node_counter: &mut ast::NodeCounter,
    ) -> Result<Self> {
        module_manager.elaborate_modules(&self.0.ast, node_counter)?;
        // Remove Module and ModuleTypeBind declarations - they've been elaborated
        self.0.ast.declarations.retain(|decl| {
            !matches!(
                decl,
                ast::Declaration::Module(_) | ast::Declaration::ModuleTypeBind(_)
            )
        });
        Ok(self)
    }

    /// Resolve names: rewrite FieldAccess -> QualifiedName and load modules
    pub fn resolve(mut self, module_manager: &module_manager::ModuleManager) -> Result<Resolved> {
        name_resolution::run(&mut self.0.ast, module_manager)?;
        resolve_resources::run(&mut self.0.ast)?;
        Ok(Resolved(self.0))
    }
}

/// Names have been resolved
pub struct Resolved(FrontInner);

impl Resolved {
    /// Fold AST-level integer constants (required before type checking)
    pub fn fold_ast_constants(mut self) -> AstConstFoldedEarly {
        ast_const_fold::run(&mut self.0.ast);
        AstConstFoldedEarly(self.0)
    }
}

/// AST integer constants have been folded (before type checking)
pub struct AstConstFoldedEarly(FrontInner);

impl AstConstFoldedEarly {
    /// Type check the program
    pub fn type_check(mut self, module_manager: &mut module_manager::ModuleManager) -> Result<TypeChecked> {
        let out = types::run::run(&mut self.0.ast, module_manager)?;
        Ok(TypeChecked {
            ast: self.0.ast,
            type_table: out.type_table,
            warnings: out.warnings,
            checker_builtins: out.builtin_names,
            schemes: out.schemes,
            name_resolution: out.name_resolution,
        })
    }
}

/// Program has been type checked
pub struct TypeChecked {
    pub ast: ast::Program,
    pub type_table: TypeTable,
    pub warnings: Vec<type_checker::TypeWarning>,
    pub checker_builtins: Vec<String>,
    pub schemes: LookupMap<String, TypeScheme<TypeName>>,
    pub name_resolution: name_resolution::NameResolution,
}

impl TypeChecked {
    /// Print warnings to stderr (convenience method)
    pub fn print_warnings(&self) {
        for warning in &self.warnings {
            eprintln!(
                "Warning: {} at {:?}",
                warning.message(&|t| types::format_type(t)),
                warning.span()
            );
        }
    }

    /// Reject programs that still contain `???` type holes. Each hole
    /// is listed with its inferred type and source location; the
    /// caller (the driver) maps `CompilerError::TypeHole` to a
    /// distinct exit code so tooling can tell hole-errors apart from
    /// generic compilation failures. Holes are a development aid —
    /// they're not expected to reach the backend, so this is the
    /// right place to bail.
    pub fn reject_type_holes(self) -> Result<Self> {
        use std::fmt::Write;
        let holes: Vec<_> = self
            .warnings
            .iter()
            .filter_map(|w| match w {
                type_checker::TypeWarning::TypeHoleFilled { inferred_type, span } => {
                    Some((inferred_type, span))
                }
            })
            .collect();
        if holes.is_empty() {
            return Ok(self);
        }
        let mut msg = String::from("type hole(s) in program:\n");
        for (ty, span) in &holes {
            let _ = writeln!(
                &mut msg,
                "  at {}:{} — inferred `{}`",
                span.start_line,
                span.start_col,
                types::format_type(ty),
            );
        }
        Err(err_type_hole!("{}", msg.trim_end()))
    }

    /// Transform AST to TLC. `module_manager` provides access to prelude
    /// declarations. `fill_holes` makes `???` type-hole expressions lower
    /// to a default value of the inferred type rather than panicking. The
    /// driver should call `reject_type_holes` first when
    /// `fill_holes = false`; unfillable hole types under
    /// `fill_holes = true` land in the returned program's
    /// `fill_hole_errors`.
    pub fn to_tlc(
        self,
        module_manager: &module_manager::ModuleManager,
        fill_holes: bool,
    ) -> TlcTransformed {
        let out = tlc::run::run(
            &self.ast,
            self.type_table,
            &self.schemes,
            &self.checker_builtins,
            &self.name_resolution,
            module_manager,
            fill_holes,
        );
        TlcTransformed(TlcEarlyInner {
            tlc: out.program,
            type_table: out.type_table,
            known_defs: out.known_defs,
            schemes: out.schemes,
            fill_hole_errors: out.fill_hole_errors,
            auto_storage_binding_ids: IdSource::new(),
        })
    }
}

// =============================================================================
// TLC-based pipeline stages
// =============================================================================

/// Shared payload for the TLC early states (`TlcTransformed`,
/// `TlcPartialEvaled`, `TlcSoaNormalized`, `TlcSoacsAnfNormalized`). Wrapped by each
/// newtype; inner fields are crate-pub so the transition impls can move
/// them across group boundaries by value.
pub struct TlcEarlyInner {
    pub tlc: tlc::Program,
    pub type_table: TypeTable,
    /// Built-in names that should not be captured as free variables
    pub known_defs: LookupSet<String>,
    /// Type schemes for functions (for monomorphization)
    pub schemes: LookupMap<SymbolId, types::TypeScheme>,
    /// Errors surfaced while default-filling `???` type holes with
    /// `--fill-holes`. Empty unless `to_tlc` was called with
    /// `fill_holes = true` and some hole had a type that couldn't
    /// be defaulted. The driver checks this before proceeding to
    /// later TLC passes and turns a non-empty list into a
    /// `CompilerError::TypeHole`.
    pub fill_hole_errors: Vec<error::CompilerError>,
    /// Module-wide id factory for compiler-auto-allocated storage
    /// bindings (the ones not pinned by a user `#[storage(set, binding)]`
    /// attribute). `pin_entry_buffers` draws inputs from it; `egir::from_tlc`
    /// draws outputs and scratch slots from the same factory. Sharing
    /// one counter across passes guarantees no two `OpVariable`s land
    /// on the same `(set, binding)` slot regardless of declared type.
    pub auto_storage_binding_ids: IdSource<u32>,
}

/// AST has been transformed to TLC
pub struct TlcTransformed(pub TlcEarlyInner);

impl std::ops::Deref for TlcTransformed {
    type Target = TlcEarlyInner;
    fn deref(&self) -> &TlcEarlyInner {
        &self.0
    }
}

impl TlcTransformed {
    /// Pin each compute entry's storage-param buffer region into its type:
    /// substitute the param's buffer variable → `Buffer(set, binding)` so a
    /// view's descriptor is a statically-known property of its type, and every
    /// downstream view inherits its region by unification. This is the first
    /// TLC pass; the resulting `TlcBuffersPinned` typestate is what `partial_eval`
    /// consumes, so the rest of the pipeline cannot run without it.
    pub fn pin_entry_buffers(self) -> Result<TlcBuffersPinned> {
        let mut inner = self.0;
        tlc::pin_entry_buffers::run(&mut inner.tlc, &mut inner.auto_storage_binding_ids)?;
        Ok(TlcBuffersPinned(inner))
    }
}

/// TLC after entry-param buffer regions are pinned into their types.
pub struct TlcBuffersPinned(pub TlcEarlyInner);

impl std::ops::Deref for TlcBuffersPinned {
    type Target = TlcEarlyInner;
    fn deref(&self) -> &TlcEarlyInner {
        &self.0
    }
}

impl TlcBuffersPinned {
    /// Validate source-level consumption before any simplification or
    /// inlining can erase the call boundary that carries the `*T`
    /// contract. `partial_eval` is defined only on the resulting
    /// typestate, so no pipeline reaches the optimizer unvalidated.
    pub fn validate_ownership(self) -> Result<TlcOwnershipValidated> {
        tlc::ownership::check(&self.0.tlc)?;
        Ok(TlcOwnershipValidated(self.0))
    }
}

/// TLC whose source-level consumption has been validated.
pub struct TlcOwnershipValidated(pub TlcEarlyInner);

impl std::ops::Deref for TlcOwnershipValidated {
    type Target = TlcEarlyInner;
    fn deref(&self) -> &TlcEarlyInner {
        &self.0
    }
}

impl TlcOwnershipValidated {
    /// Constant folding and algebraic simplifications.
    pub fn partial_eval(self) -> TlcPartialEvaled {
        let mut inner = self.0;
        tlc::partial_eval::PartialEvaluator::partial_eval(&mut inner.tlc);
        TlcPartialEvaled(inner)
    }
}

#[cfg(test)]
pub(crate) fn optimize_tlc_for_test(state: TlcOwnershipValidated) -> TlcReachable {
    optimize_tlc_for_test_thru_soac_normalization(state)
        .float_runtime_index_nested_producers()
        .defunctionalize()
        .fold_generated_lambdas()
        .apply_ownership()
        .expect("apply_ownership")
        .filter_reachable()
}

#[cfg(test)]
pub(crate) fn optimize_tlc_for_test_thru_soac_normalization(
    state: TlcOwnershipValidated,
) -> TlcSoacsAnfNormalized {
    state
        .partial_eval()
        .normalize_soacs()
        .monomorphize()
        .rep_specialize()
        .inline_small()
        .force_inline_soac_helpers()
        .renormalize_inlined_soa()
        .canonicalize_conditional_producers()
        .normalize_soacs_to_anf()
}

/// TLC after partial evaluation
pub struct TlcPartialEvaled(pub TlcEarlyInner);

impl std::ops::Deref for TlcPartialEvaled {
    type Target = TlcEarlyInner;
    fn deref(&self) -> &TlcEarlyInner {
        &self.0
    }
}

impl TlcPartialEvaled {
    /// SoA transform + SOAC normalization: rewrite array-of-tuple types,
    /// flatten Map+Zip into multi-input Map, and convert standalone Zip to tuple.
    pub fn normalize_soacs(self) -> TlcSoaNormalized {
        let mut inner = self.0;
        tlc::soa::run(&mut inner.tlc);
        TlcSoaNormalized(inner)
    }
}

/// TLC after SoA normalization (arrays never contain tuples, zips eliminated)
pub struct TlcSoaNormalized(pub TlcEarlyInner);

impl std::ops::Deref for TlcSoaNormalized {
    type Target = TlcEarlyInner;
    fn deref(&self) -> &TlcEarlyInner {
        &self.0
    }
}

/// TLC after force-inlining of every user function whose body contains a SOAC.
/// This runs after monomorphization so helpers are concrete, exposing
/// producer/consumer edges to semantic EGIR without cross-call summaries.
pub struct TlcSoacHelpersInlined(pub TlcEarlyInner);

impl std::ops::Deref for TlcSoacHelpersInlined {
    type Target = TlcEarlyInner;
    fn deref(&self) -> &TlcEarlyInner {
        &self.0
    }
}

impl TlcSoacHelpersInlined {
    /// Re-run SoA normalization after inlining exposed tuple/zip/map structure.
    pub fn renormalize_inlined_soa(self) -> TlcInlinedSoaNormalized {
        let mut inner = self.0;
        tlc::soa::run(&mut inner.tlc);
        TlcInlinedSoaNormalized(inner)
    }
}

/// TLC after post-inlining SoA normalization.
pub struct TlcInlinedSoaNormalized(pub TlcEarlyInner);

impl std::ops::Deref for TlcInlinedSoaNormalized {
    type Target = TlcEarlyInner;
    fn deref(&self) -> &TlcEarlyInner {
        &self.0
    }
}

impl TlcInlinedSoaNormalized {
    /// Turn eligible array-valued conditionals into one pointwise producer.
    pub fn canonicalize_conditional_producers(self) -> TlcConditionalProducersCanonicalized {
        let mut inner = self.0;
        tlc::if_over_producer::run(&mut inner.tlc);
        TlcConditionalProducersCanonicalized(inner)
    }
}

/// TLC after array-valued conditional producer canonicalization.
pub struct TlcConditionalProducersCanonicalized(pub TlcEarlyInner);

impl std::ops::Deref for TlcConditionalProducersCanonicalized {
    type Target = TlcEarlyInner;
    fn deref(&self) -> &TlcEarlyInner {
        &self.0
    }
}

impl TlcConditionalProducersCanonicalized {
    /// Lift nested SOAC expressions onto flat let chains so TLC-to-EGIR
    /// conversion records every semantic producer/consumer edge explicitly.
    pub fn normalize_soacs_to_anf(self) -> TlcSoacsAnfNormalized {
        let mut inner = self.0;
        tlc::soac_anf::run(&mut inner.tlc);
        TlcSoacsAnfNormalized(inner)
    }
}

/// TLC with every SOAC explicit in a flat let chain, ready for EGIR.
pub struct TlcSoacsAnfNormalized(pub TlcEarlyInner);

impl std::ops::Deref for TlcSoacsAnfNormalized {
    type Target = TlcEarlyInner;
    fn deref(&self) -> &TlcEarlyInner {
        &self.0
    }
}

impl TlcSoacsAnfNormalized {
    /// Expose runtime-indexed nested producers without assigning resources or
    /// physical entries. Static demand and producer/consumer fusion are EGIR
    /// graph rewrites, where use edges and resource effects are explicit.
    pub fn float_runtime_index_nested_producers(self) -> TlcRuntimeIndexProducersFloated {
        let mut inner = self.0;
        tlc::runtime_index_producers::run(&mut inner.tlc);
        TlcRuntimeIndexProducersFloated(inner)
    }
}

/// TLC after ownership-driven rewrites. All `with` calls have
/// settled on either the functional or in-place intrinsic; eligible
/// SOACs carry a uniqueness-only input candidate for EGIR to resolve.
pub struct TlcOwnershipApplied(pub TlcEarlyInner);

impl std::ops::Deref for TlcOwnershipApplied {
    type Target = TlcEarlyInner;
    fn deref(&self) -> &TlcEarlyInner {
        &self.0
    }
}

impl TlcGeneratedLambdasFolded {
    /// Ownership/liveness analysis and ownership-driven source rewrites.
    /// Physical residency and output routing are deferred to EGIR.
    pub fn apply_ownership(self) -> Result<TlcOwnershipApplied> {
        let mut inner = self.0;
        tlc::ownership::apply_ownership(&mut inner.tlc)?;
        Ok(TlcOwnershipApplied(inner))
    }
}

/// TLC after runtime-indexed producers have been exposed without assigning
/// resources or physical entries.
pub struct TlcRuntimeIndexProducersFloated(pub TlcEarlyInner);

impl std::ops::Deref for TlcRuntimeIndexProducersFloated {
    type Target = TlcEarlyInner;
    fn deref(&self) -> &TlcEarlyInner {
        &self.0
    }
}

impl TlcRuntimeIndexProducersFloated {
    /// Closure-convert, specialize HOFs, and thread explicit captures. EGIR
    /// performs subsequent residency and physical-entry decisions.
    pub fn defunctionalize(self) -> TlcDefunctionalized {
        let mut inner = self.0;
        let closure_info = tlc::closure_convert::run(&mut inner.tlc, &inner.known_defs);
        tlc::hof_specialize::run(&mut inner.tlc, &closure_info);
        tlc::closure_calls_lower::run(&mut inner.tlc, &closure_info);
        TlcDefunctionalized(inner)
    }
}

impl TlcOwnershipApplied {
    /// Eliminate unreachable defs before semantic EGIR conversion.
    pub fn filter_reachable(self) -> TlcReachable {
        let mut inner = self.0;
        tlc::inline::run_reachable(&mut inner.tlc);
        TlcReachable(inner)
    }
}

/// TLC with all lambdas defunctionalized (lifted + SOAC captures flattened).
/// Carries the full early bag so `schemes` is available to `monomorphize`.
pub struct TlcDefunctionalized(pub TlcEarlyInner);

impl std::ops::Deref for TlcDefunctionalized {
    type Target = TlcEarlyInner;
    fn deref(&self) -> &TlcEarlyInner {
        &self.0
    }
}

impl TlcSoaNormalized {
    /// Specialize polymorphic intrinsics and monomorphize user functions
    /// (region-specialized). After this no `Type::Variable` remains.
    pub fn monomorphize(self) -> TlcMonomorphized {
        let mut inner = self.0;
        tlc::specialize::run(&mut inner.tlc);
        tlc::monomorphize::run(&mut inner.tlc, &inner.schemes);
        TlcMonomorphized(inner)
    }
}

/// TLC with all functions monomorphized (no type variables remain)
pub struct TlcMonomorphized(pub TlcEarlyInner);

impl std::ops::Deref for TlcMonomorphized {
    type Target = TlcEarlyInner;
    fn deref(&self) -> &TlcEarlyInner {
        &self.0
    }
}

impl TlcMonomorphized {
    /// Representation-specialize call edges whose array representation is known
    /// only at the producer site, before helper inlining and EGIR conversion.
    pub fn rep_specialize(self) -> TlcRepSpecialized {
        let mut inner = self.0;
        tlc::rep_specialize::run(&mut inner.tlc);
        TlcRepSpecialized(inner)
    }
}

/// TLC after inlining compiler-generated lambda defs and DCE
pub struct TlcGeneratedLambdasFolded(pub TlcEarlyInner);

impl std::ops::Deref for TlcGeneratedLambdasFolded {
    type Target = TlcEarlyInner;
    fn deref(&self) -> &TlcEarlyInner {
        &self.0
    }
}

impl TlcRepSpecialized {
    /// Inline small user functions and constants at their call/reference sites.
    pub fn inline_small(self) -> TlcSmallInlined {
        let mut inner = self.0;
        tlc::inline::run_small(&mut inner.tlc);
        TlcSmallInlined(inner)
    }
}

/// TLC after small function and constant inlining
pub struct TlcSmallInlined(pub TlcEarlyInner);

impl std::ops::Deref for TlcSmallInlined {
    type Target = TlcEarlyInner;
    fn deref(&self) -> &TlcEarlyInner {
        &self.0
    }
}

impl TlcSmallInlined {
    /// Force-inline every user function whose body contains a SOAC. Helpers are
    /// already monomorphic, so EGIR receives explicit producer/consumer edges.
    pub fn force_inline_soac_helpers(self) -> TlcSoacHelpersInlined {
        let mut inner = self.0;
        tlc::inline::run_force_soac_helpers(&mut inner.tlc);
        TlcSoacHelpersInlined(inner)
    }
}

/// TLC after representation specialization (Phase 2 of array-variant-
/// abstract). Every `App` whose `Var(Symbol(callee))` arg at an
/// `Abstract`-typed position had a producer-known concrete variant now
/// invokes a specialized clone of the callee with that concrete
/// variant baked in. Runs before semantic EGIR conversion so every call edge
/// has a concrete representation.
pub struct TlcRepSpecialized(pub TlcEarlyInner);

impl std::ops::Deref for TlcRepSpecialized {
    type Target = TlcEarlyInner;
    fn deref(&self) -> &TlcEarlyInner {
        &self.0
    }
}

impl TlcDefunctionalized {
    /// Inline compiler-generated `_w_lambda_*` defs back at their call sites,
    /// then DCE. Runs after defunctionalize (which produces those lambdas).
    pub fn fold_generated_lambdas(self) -> TlcGeneratedLambdasFolded {
        let mut inner = self.0;
        tlc::inline::run_large(&mut inner.tlc);
        TlcGeneratedLambdasFolded(inner)
    }
}

/// TLC after dead code elimination, ready for semantic EGIR conversion.
pub struct TlcReachable(pub TlcEarlyInner);

impl std::ops::Deref for TlcReachable {
    type Target = TlcEarlyInner;
    fn deref(&self) -> &TlcEarlyInner {
        &self.0
    }
}

impl TlcReachable {
    /// Run `tlc::input_slice_bounds` over every entry, returning a state
    /// that carries the inferred per-entry minimum-required input
    /// buffer lengths alongside the program. `TlcReachable.to_egraph()`
    /// is intentionally absent: the only mainline path off `TlcReachable`
    /// is through this analyzer, which keeps the descriptor's input
    /// `length` fields populated for every storage-bound input the body
    /// slices.
    pub fn infer_input_slice_bounds(self) -> TlcInputSliceBoundsInferred {
        let input_lens = tlc::input_slice_bounds::compute_for_program(&self.0.tlc);
        TlcInputSliceBoundsInferred {
            inner: self.0,
            input_lens,
        }
    }
}

/// TLC after `input_slice_bounds` inference. Same program as
/// `TlcReachable`; carries a per-entry-name → per-`SymbolId` map of
/// minimum-required input buffer lengths that flows into
/// `EntryInput.length` at `to_egraph` time.
pub struct TlcInputSliceBoundsInferred {
    pub inner: TlcEarlyInner,
    pub input_lens: tlc::input_slice_bounds::ProgramBounds,
}

impl std::ops::Deref for TlcInputSliceBoundsInferred {
    type Target = TlcEarlyInner;
    fn deref(&self) -> &TlcEarlyInner {
        &self.inner
    }
}

impl TlcInputSliceBoundsInferred {
    /// Convert TLC to raw semantic EGIR. Every later EGIR pass has its own
    /// consuming typestate transition.
    pub fn to_egraph(self) -> std::result::Result<EgirRaw, ConvertError> {
        let TlcEarlyInner {
            tlc,
            mut auto_storage_binding_ids,
            ..
        } = self.inner;
        let mut effect_ids = IdSource::new();
        let inner = egir::from_tlc::run(
            &tlc,
            &self.input_lens,
            &mut auto_storage_binding_ids,
            &mut effect_ids,
        )?;
        Ok(EgirRaw {
            inner,
            binding_ids: auto_storage_binding_ids,
            effect_ids,
        })
    }
}

// =============================================================================
// EGIR typestate chain
//
// Six newtypes over the semantic and physical programs defined in
// `egir::program`. Transitions consume `self` and re-wrap the inner into the
// next newtype.
// Pass modules in `egir::*` are called per-body from inside the transitions
// and are unaware of the newtype wrapping.
// =============================================================================

/// Target capability profile selected before semantic EGIR is lowered to SSA.
/// `Portable` deliberately uses the common SPIR-V/WGSL capability subset and
/// is retained for tools and tests that want to inspect one shared SSA module.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CodegenTarget {
    Portable,
    Spirv,
    Wgsl,
}

/// Whether semantic segmented operations may expand into multiple host
/// dispatches.  Single-stage mode still constructs semantic SegOps; only the
/// terminal scheduling decision changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SchedulePolicy {
    Serial,
    Parallel,
}

/// Target and scheduling policy for the semantic-EGIR-to-SSA boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoweringProfile {
    pub target: CodegenTarget,
    pub schedule: SchedulePolicy,
}

impl LoweringProfile {
    pub const PORTABLE: Self = Self {
        target: CodegenTarget::Portable,
        schedule: SchedulePolicy::Parallel,
    };

    pub const fn new(target: CodegenTarget, schedule: SchedulePolicy) -> Self {
        Self { target, schedule }
    }
}

/// Raw EGIR program, directly produced by TLC → EGIR conversion.
pub struct EgirRaw {
    inner: RawProgram,
    binding_ids: IdSource<u32>,
    effect_ids: IdSource<egir::types::EffectToken>,
}

/// EGIR after entry-output realization. Every declared output has its
/// writes materialised as side effects against the bound storage view
/// (compute) or `OutputSlot` place (graphics). The body's `Return`
/// terminator carries no value. Non-output runtime arrays can still await
/// EGIR residency planning. See `egir::realize_outputs`.
pub struct EgirOutputsRealized {
    inner: RawProgram,
    binding_ids: IdSource<u32>,
    effect_ids: IdSource<egir::types::EffectToken>,
}

/// EGIR after all reachable SOACs have been reified as semantic
/// `SegMap`, `SegRed`, or `SegScan` operations. No dispatch schedule or scratch
/// storage has been chosen yet.
pub struct EgirSegmented {
    inner: SemanticProgram,
    binding_ids: IdSource<u32>,
    effect_ids: IdSource<egir::types::EffectToken>,
}

/// Semantic EGIR after graph-level optimization. SegOps remain intact.
pub struct EgirOptimized {
    inner: SemanticProgram,
    binding_ids: IdSource<u32>,
    effect_ids: IdSource<egir::types::EffectToken>,
}

/// Semantic EGIR after logical resource planning. Physical scratch bindings,
/// phase kernels, and descriptor publication are intentionally deferred until
/// `plan` receives a target profile.
pub struct EgirAllocated {
    inner: AllocatedProgram,
    binding_ids: IdSource<u32>,
    effect_ids: IdSource<egir::types::EffectToken>,
}

/// Physical EGIR together with a graph-free summary of the schedule that
/// produced it. Construction is private to `EgirAllocated::plan`, so callers
/// cannot bypass schedule validation.
pub struct EgirPlanned {
    physical: egir::program::PhysicalProgram,
    kernel_plan: egir::parallelize::KernelPlanSummary,
    profile: LoweringProfile,
    effect_ids: IdSource<egir::types::EffectToken>,
}

impl EgirRaw {
    /// Realize every entry's outputs into side-effect writes. For
    /// compute entries: walk `SemanticEntry.output_routes` and emit a DPS
    /// write per route against the slot's `OutputView` (Map/Scan
    /// retarget, fixed-aggregate element stores, scalar `Store` at
    /// index 0). For graphics entries: classify the body's
    /// `Return(Some(value))` and emit one store per declared
    /// `OutputSlot { index }`. Runs before segmentation so the
    /// SOAC→OutputView rewrite precedes semantic segmented-op wrapping;
    /// allocation later plans any remaining residency requirements.
    pub fn realize_outputs(self) -> std::result::Result<EgirOutputsRealized, ConvertError> {
        let EgirRaw {
            mut inner,
            binding_ids,
            mut effect_ids,
        } = self;
        egir::realize_outputs::run(&mut inner, &mut effect_ids)?;
        Ok(EgirOutputsRealized {
            inner,
            binding_ids,
            effect_ids,
        })
    }
}

impl EgirOutputsRealized {
    /// Reify every reachable SOAC as a semantic segmented operation and choose
    /// kernel versus lane-local placement from EGIR value/effect context.
    pub fn segment(self) -> EgirSegmented {
        let EgirOutputsRealized {
            inner,
            binding_ids,
            effect_ids,
        } = self;
        let inner = egir::reify::run(inner);
        if cfg!(debug_assertions) {
            egir::semantic_graph::verify(&inner).expect("invalid semantic EGIR");
        }
        EgirSegmented {
            inner,
            binding_ids,
            effect_ids,
        }
    }
}

impl EgirSegmented {
    /// Preserve semantic SegOps while running graph-level fusion, indexed
    /// demand scalarization, and dead-operation elimination. The typestate is
    /// load-bearing because resource planning and scheduling are not permitted
    /// before this boundary.
    pub fn optimize(self) -> EgirOptimized {
        let EgirSegmented {
            mut inner,
            binding_ids,
            effect_ids,
        } = self;
        egir::semantic_opt::run(&mut inner);
        EgirOptimized {
            inner,
            binding_ids,
            effect_ids,
        }
    }
}

impl EgirOptimized {
    /// Plan logical resources without choosing physical descriptor bindings.
    /// The carried allocator lets terminal lowering allocate scratch
    /// transactionally without mutating upstream TLC state.
    pub fn allocate(self) -> std::result::Result<EgirAllocated, ConvertError> {
        let EgirOptimized {
            inner,
            binding_ids,
            mut effect_ids,
        } = self;
        if cfg!(debug_assertions) {
            egir::semantic_graph::verify(&inner).expect("invalid optimized semantic EGIR");
        }
        let inner = egir::program::plan_logical_resources(inner, &mut effect_ids)?;
        Ok(EgirAllocated {
            inner,
            binding_ids,
            effect_ids,
        })
    }
}

impl EgirAllocated {
    /// Human-readable semantic IR including concrete spaces, region captures,
    /// output routing, and logical resource accesses.
    pub fn semantic_ir(&self) -> String {
        egir::semantic_graph::summary(&self.inner)
    }

    /// The pre-target residency manifest. It contains semantic output storage,
    /// materializations, filter capacity/length cells, and compiler flows, but
    /// deliberately excludes reduce/scan/filter algorithm work buffers.
    pub fn logical_resources(&self) -> &[egir::program::LogicalResource] {
        &self.inner.resources
    }

    pub fn semantic_dependencies(&self) -> &[egir::program::SemanticDependency] {
        &self.inner.semantic_dependencies
    }

    /// Select and validate a target-specific kernel plan, then construct the
    /// physical EGIR entries described by that plan.
    pub fn plan(self, profile: LoweringProfile) -> std::result::Result<EgirPlanned, ConvertError> {
        let mut binding_ids = self.binding_ids;
        let mut effect_ids = self.effect_ids;
        let (physical, kernel_plan) =
            egir::parallelize::plan(self.inner, &mut binding_ids, &mut effect_ids, profile)?;
        Ok(EgirPlanned {
            physical,
            kernel_plan,
            profile,
            effect_ids,
        })
    }
}

impl EgirPlanned {
    /// Final logical resource manifest after target recipe selection has added
    /// only the algorithm-specific work buffers required by the validated
    /// target recipes.
    pub fn logical_resources(&self) -> &[egir::program::LogicalResource] {
        self.physical.logical_resources()
    }

    pub fn kernel_plan(&self) -> &egir::parallelize::KernelPlanSummary {
        &self.kernel_plan
    }

    /// Expand and elaborate the validated physical plan into SSA.
    pub fn lower_to_ssa(mut self) -> std::result::Result<SsaConverted, ConvertError> {
        let plan = self.kernel_plan;
        egir::soac_expand::run(&mut self.physical, &mut self.effect_ids)?;
        egir::materialize::run(&mut self.physical);
        egir::partial_inline::run(&mut self.physical).map_err(ConvertError::Internal)?;
        egir::rewrite::run(&mut self.physical);
        egir::skel_opt::run(&mut self.physical);
        egir::resource_erasure::run(&mut self.physical)?;
        let (ssa, pipeline) = egir::elaborate::run_program(self.physical);
        Ok(SsaConverted {
            ssa,
            pipeline,
            profile: self.profile,
            kernel_plan: plan,
        })
    }
}

/// TLC has been converted directly to SSA (via EGIR).
pub struct SsaConverted {
    pub ssa: ssa::types::Program,
    pub pipeline: pipeline_descriptor::PipelineDescriptor,
    pub profile: LoweringProfile,
    pub kernel_plan: egir::parallelize::KernelPlanSummary,
}

impl SsaConverted {
    /// Lower SSA to SPIR-V. Materialization (`Index` → `Materialize` +
    /// `DynamicExtract`) was already done by the EGIR pipeline if `to_egir`
    /// was called with `EgirOpts::for_spirv()`.
    pub fn lower(self) -> error::Result<Lowered> {
        if self.profile.target == CodegenTarget::Wgsl {
            return Err(crate::err_spirv!(
                "SSA was scheduled for WGSL and cannot be lowered as SPIR-V"
            ));
        }
        egir::verify_no_abstract::run(&self.ssa)?;
        spirv::verify_buffer_layouts::run(&self.ssa)?;
        let spirv = spirv::lower_ssa_program(&self.ssa)?;
        Ok(Lowered {
            spirv,
            pipeline: self.pipeline,
        })
    }

    pub fn lower_wgsl(self) -> error::Result<String> {
        if self.profile.target == CodegenTarget::Spirv {
            return Err(crate::err_spirv!(
                "SSA was scheduled for SPIR-V and cannot be lowered as WGSL"
            ));
        }
        egir::verify_no_abstract::run(&self.ssa)?;
        wgsl::lower(&self.ssa)
    }
}

/// Final SPIR-V output
pub struct Lowered {
    pub spirv: Vec<u32>,
    pub pipeline: pipeline_descriptor::PipelineDescriptor,
}

// =============================================================================
// Test utilities - cached prelude for faster test execution
// =============================================================================

#[cfg(test)]
use std::sync::OnceLock;

/// Cached prelude data AND the node counter state after parsing it
#[cfg(test)]
static PRELUDE_CACHE: OnceLock<(module_manager::PreElaboratedPrelude, NodeCounter)> = OnceLock::new();

/// Get the cached prelude and a cloned node counter (test-only)
/// This avoids re-parsing prelude files for each test, providing ~10x speedup
#[cfg(test)]
fn get_prelude_cache() -> (&'static module_manager::PreElaboratedPrelude, NodeCounter) {
    let (prelude, counter) = PRELUDE_CACHE.get_or_init(|| {
        let mut nc = NodeCounter::new();
        let prelude =
            module_manager::ModuleManager::create_prelude(&mut nc).expect("Failed to create prelude cache");
        (prelude, nc)
    });
    (prelude, counter.clone())
}

/// Create a ModuleManager and NodeCounter using the cached prelude (test-only)
#[cfg(test)]
pub fn cached_module_manager() -> (module_manager::ModuleManager, NodeCounter) {
    let (prelude, node_counter) = get_prelude_cache();
    (
        module_manager::ModuleManager::from_prelude(prelude.clone()),
        node_counter,
    )
}

/// Build a `(NodeCounter, ModuleManager)` pair using the cached prelude (test-only).
#[cfg(test)]
pub fn cached_compiler_init() -> (NodeCounter, module_manager::ModuleManager) {
    let (prelude, node_counter) = get_prelude_cache();
    init_compiler_from_prelude(prelude.clone(), node_counter)
}

// =============================================================================
// Test-only milestone helpers
// =============================================================================
//
// `compile_thru_*` helpers run the pipeline up to a milestone and return
// just the milestone value. Each subsumes the previous one:
//
//   compile_thru_frontend  →  TypeChecked          (AST passes done)
//   compile_thru_tlc       →  TlcReachable         (TLC pipeline done)
//   compile_thru_ssa       →  SsaConverted         (EGIR + elaborate done)
//   compile_thru_spirv     →  Lowered              (final SPIR-V binary)
//
// These exist so test files don't have to enumerate every pass — when a
// new pass lands, only the helper that owns its milestone needs updating.
// Tests that need an off-milestone stop call the typestate methods directly.

/// Run AST passes through type checking. Uses the cached prelude.
#[cfg(test)]
pub fn compile_thru_frontend(source: &str) -> error::Result<TypeChecked> {
    let (mut node_counter, mut module_manager) = cached_compiler_init();
    Compiler::parse(source, &mut node_counter)?
        .elaborate_modules(&mut module_manager, &mut node_counter)?
        .resolve(&module_manager)?
        .fold_ast_constants()
        .type_check(&mut module_manager)
}

/// Run the canonical TLC optimization pipeline (no physical scheduling or
/// hole-filling) through `filter_reachable`.
#[cfg(test)]
pub fn compile_thru_tlc(source: &str) -> error::Result<TlcReachable> {
    let (mut node_counter, mut module_manager) = cached_compiler_init();
    let type_checked = Compiler::parse(source, &mut node_counter)?
        .elaborate_modules(&mut module_manager, &mut node_counter)?
        .resolve(&module_manager)?
        .fold_ast_constants()
        .type_check(&mut module_manager)?;
    let state = type_checked.to_tlc(&module_manager, false).pin_entry_buffers()?.validate_ownership()?;
    Ok(optimize_tlc_for_test(state))
}

/// Internal: run all the way through EGIR + elaborate to SSA from a
/// pre-built `TlcReachable`. Both `compile_thru_ssa` and
/// `compile_thru_spirv_single_stage` build the SSA the same way; only
/// the downstream scheduling profile differs.
#[cfg(test)]
fn ssa_from_reachable(
    tlc: TlcReachable,
    profile: LoweringProfile,
) -> std::result::Result<SsaConverted, Box<dyn std::error::Error>> {
    let raw = tlc.infer_input_slice_bounds().to_egraph()?;
    let allocated = raw.realize_outputs()?.segment().optimize().allocate()?;
    let planned = allocated.plan(profile)?;
    Ok(planned.lower_to_ssa()?)
}

/// Run all the way through EGIR + elaborate to SSA. Materialize is enabled
/// (matches the SPIR-V backend's requirements). Returns the boxed
/// `Result<_, dyn Error>` so callers see both compiler errors and EGIR
/// conversion errors uniformly.
#[cfg(test)]
pub fn compile_thru_ssa(source: &str) -> std::result::Result<SsaConverted, Box<dyn std::error::Error>> {
    ssa_from_reachable(compile_thru_tlc(source)?, LoweringProfile::PORTABLE)
}

/// Run the full pipeline to a final SPIR-V binary.
#[cfg(test)]
pub fn compile_thru_spirv(source: &str) -> std::result::Result<Lowered, Box<dyn std::error::Error>> {
    Ok(ssa_from_reachable(
        compile_thru_tlc(source)?,
        LoweringProfile::new(CodegenTarget::Spirv, SchedulePolicy::Parallel),
    )?
    .lower()?)
}

/// Single-stage equivalent of `compile_thru_spirv`: matches the CLI's
/// `--single-stage` terminal scheduling policy.
#[cfg(test)]
pub fn compile_thru_spirv_single_stage(
    source: &str,
) -> std::result::Result<Lowered, Box<dyn std::error::Error>> {
    Ok(ssa_from_reachable(
        compile_thru_tlc(source)?,
        LoweringProfile::new(CodegenTarget::Spirv, SchedulePolicy::Serial),
    )?
    .lower()?)
}
