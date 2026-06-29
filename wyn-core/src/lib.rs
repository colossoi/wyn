pub mod ast;
pub mod ast_renumber;
pub mod binding_layout;
pub mod builtins;
pub mod diags;
pub mod error;
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
use egir::program::EgirInner;
use egir::publish::PipelineDescriptorPublish;

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
        let id = Id::from(self.next_id);
        self.next_id += 1;
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
/// uniform / texture / sampler resource. The one type carrying that pair
/// through every layer (TLC `ArrayExpr`, SSA / EGIR `ViewSource`,
/// `EntryInput`, `StorageBindingDecl`, `EntryParamBindingKind`, …),
/// rather than parallel `(u32, u32)` fields. Deliberately no `Default` impl —
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
//     parsed.resolve(&module_manager)              -> Resolved
//       .fold_ast_constants()                      -> AstConstFoldedEarly
//       .type_check(&mut module_manager)           -> TypeChecked
//
// TLC stages (AST → SSA):
//       .to_tlc(&module_manager, fill_holes)       -> TlcTransformed
//       .pin_entry_regions()                       -> TlcTransformed
//       .partial_eval()                            -> TlcPartialEvaled
//       .normalize_soacs()                         -> TlcSoaNormalized
//       .fuse_maps()                               -> TlcFused
//       .apply_ownership()                         -> TlcOwnershipApplied
//       .normalize_outputs()                       -> TlcOutputsNormalized
//       .expose_entry_producer_helpers()           -> TlcEntryProducersExposed
//       .fuse_static_indices()                     -> TlcStaticIndexFused
//       .float_runtime_index_nested_producers()    -> TlcRuntimeIndexProducersFloated
//       .plan_execute_gather_residency()           -> TlcGathersLifted
//       .hoist_scalar_prepasses(disable)            -> TlcScalarPrepassesHoisted
//       .defunctionalize()                         -> TlcDefunctionalized
//       .monomorphize()                            -> TlcMonomorphized
//       .fold_generated_lambdas()                  -> TlcGeneratedLambdasFolded
//       .inline_small()                            -> TlcSmallInlined
//       .rep_specialize()                          -> TlcRepSpecialized
//       .parallelize_soacs(disable)                -> TlcParallelized
//       .filter_reachable()                        -> TlcReachable
//       .to_egraph()                               -> EgirRaw
//
// EGIR stages:
//       .realize_outputs()                         -> EgirOutputsRealized
//       .segment(recognitions)                     -> EgirSegmented
//       .schedule(binding_ids)                     -> EgirScheduled
//       .expand_soacs()                            -> EgirSoacExpanded
//       [.materialize()]                           -> EgirMaterialized
//       .optimize_skeleton()                       -> EgirSkelOptimized
//       .elaborate()                               -> SsaConverted
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
pub fn init_compiler() -> (NodeCounter, module_manager::ModuleManager) {
    let mut node_counter = NodeCounter::new();
    let module_manager = match module_manager::ModuleManager::create_prelude(&mut node_counter) {
        Ok(prelude) => module_manager::ModuleManager::from_prelude(prelude),
        Err(e) => {
            eprintln!("ERROR creating prelude: {:?}", e);
            module_manager::ModuleManager::new_empty()
        }
    };
    (node_counter, module_manager)
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
/// `TlcPartialEvaled`, `TlcSoaNormalized`, `TlcFused`). Wrapped by each
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
    /// attribute). `pin_entry_regions` draws inputs from it; `egir::from_tlc`
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
    /// substitute the param's region variable → `Region(set, binding)` so a
    /// view's descriptor is a statically-known property of its type, and every
    /// downstream view inherits its region by unification. This is the first
    /// TLC pass; the resulting `TlcRegionsPinned` typestate is what `partial_eval`
    /// consumes, so the rest of the pipeline cannot run without it.
    pub fn pin_entry_regions(self) -> Result<TlcRegionsPinned> {
        let mut inner = self.0;
        tlc::pin_entry_regions::run(&mut inner.tlc, &mut inner.auto_storage_binding_ids)?;
        Ok(TlcRegionsPinned(inner))
    }
}

/// TLC after entry-param buffer regions are pinned into their types.
pub struct TlcRegionsPinned(pub TlcEarlyInner);

impl std::ops::Deref for TlcRegionsPinned {
    type Target = TlcEarlyInner;
    fn deref(&self) -> &TlcEarlyInner {
        &self.0
    }
}

impl TlcRegionsPinned {
    /// Constant folding and algebraic simplifications.
    pub fn partial_eval(self) -> TlcPartialEvaled {
        let mut inner = self.0;
        inner.tlc = tlc::partial_eval::PartialEvaluator::partial_eval(inner.tlc);
        TlcPartialEvaled(inner)
    }
}

#[cfg(test)]
impl TlcRegionsPinned {
    /// Test-only: run the canonical TLC optimization pipeline to
    /// `TlcReachable` without the per-stage `time(...)` wrappers the
    /// driver uses. Runs on the already-region-pinned state, so the
    /// fallible `pin_entry_regions` step (and its merge-error) is the
    /// caller's to propagate. The bool is forwarded as `parallelize_soacs(
    /// disable=bool)`: `false` ⇒ parallelization enabled (default driver
    /// mode, what most tests want); `true` ⇒ disabled (the `--single-stage`
    /// flag).
    pub fn optimize_for_test(self, disable_parallelize: bool) -> TlcReachable {
        self.optimize_for_test_thru_expose_producers()
            .fuse_static_indices()
            .float_runtime_index_nested_producers()
            .plan_execute_gather_residency()
            .hoist_scalar_prepasses(disable_parallelize)
            .defunctionalize()
            .fold_generated_lambdas()
            .apply_ownership()
            .expect("apply_ownership")
            .normalize_outputs()
            .expect("normalize_outputs")
            .parallelize_soacs(disable_parallelize)
            .expect("parallelize_soacs")
            .filter_reachable()
    }

    /// The shared `optimize_for_test` prefix up to and including
    /// `expose_entry_producer_helpers` — the boundary just before the residency
    /// cluster (`fuse_static_indices` → `float_runtime_index_nested_producers`
    /// → `plan_execute_gather_residency`). Per-pass tests in that cluster start
    /// here (chaining a couple more public stage methods as needed) so each pass
    /// gets its real pre-pass input rather than re-running the whole pipeline.
    /// A reorder of the prefix still only touches this method.
    pub fn optimize_for_test_thru_expose_producers(self) -> TlcEntryProducersExposed {
        self.partial_eval()
            .normalize_soacs()
            .monomorphize()
            .rep_specialize()
            .inline_small()
            .force_inline_soac_helpers()
            .canonicalize_producers()
            .fuse_maps()
            .expose_entry_producer_helpers()
    }
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
        inner.tlc = tlc::soa::run(inner.tlc);
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

impl TlcGathersLifted {
    /// Outline scalar reductions as compute pre-passes while their lexical
    /// dependencies and lambda captures are still intact.
    /// `disable` keeps single-stage compilation free of generated pre-passes.
    pub fn hoist_scalar_prepasses(self, disable: bool) -> TlcScalarPrepassesHoisted {
        let mut inner = self.0;
        if !disable {
            inner.tlc =
                tlc::parallelize::hoist_scalar_prepasses(inner.tlc, &mut inner.auto_storage_binding_ids);
        }
        TlcScalarPrepassesHoisted(inner)
    }
}

/// TLC after scalar reductions have either been outlined before closure
/// conversion or deliberately left inline in single-stage mode.
pub struct TlcScalarPrepassesHoisted(pub TlcEarlyInner);

impl std::ops::Deref for TlcScalarPrepassesHoisted {
    type Target = TlcEarlyInner;
    fn deref(&self) -> &TlcEarlyInner {
        &self.0
    }
}

impl TlcScalarPrepassesHoisted {
    /// Closure-converts + specializes HOFs + threads captures, lifting SOAC
    /// operators (including the ones fusion just composed) to function
    /// references. SOAC envelopes stay inline (not lowered to loops).
    pub fn defunctionalize(self) -> TlcDefunctionalized {
        let mut inner = self.0;
        let (cc, closure_info) = tlc::closure_convert::run(inner.tlc, &inner.known_defs);
        let hof_free = tlc::hof_specialize::run(cc, &closure_info);
        inner.tlc = tlc::closure_calls_lower::run(hof_free, &closure_info);
        TlcDefunctionalized(inner)
    }
}

/// TLC after force-inlining of every user function whose body contains a SOAC.
/// In the experimental order this runs *post-monomorphize* (helpers are already
/// concrete), so it opens producer/consumer boundaries for intraprocedural
/// fusion without any cross-call summary reasoning.
pub struct TlcSoacHelpersInlined(pub TlcEarlyInner);

impl std::ops::Deref for TlcSoacHelpersInlined {
    type Target = TlcEarlyInner;
    fn deref(&self) -> &TlcEarlyInner {
        &self.0
    }
}

impl TlcSoacHelpersInlined {
    /// Re-run SoA normalization (inlining exposed new tuple/zip/map structure)
    /// then canonicalize producers (`if_over_producer`) so fusion sees clean
    /// top-of-let-chain SOACs.
    pub fn canonicalize_producers(self) -> TlcProducerCanonicalized {
        let mut inner = self.0;
        inner.tlc = tlc::soa::run(inner.tlc);
        inner.tlc = tlc::if_over_producer::run(inner.tlc);
        TlcProducerCanonicalized(inner)
    }
}

/// TLC after producer canonicalization, ready to fuse.
pub struct TlcProducerCanonicalized(pub TlcEarlyInner);

impl std::ops::Deref for TlcProducerCanonicalized {
    type Target = TlcEarlyInner;
    fn deref(&self) -> &TlcEarlyInner {
        &self.0
    }
}

impl TlcProducerCanonicalized {
    /// Fuse consecutive SOAC operations to eliminate intermediate arrays, then
    /// DCE. With monomorphize + inlining already done, every producer/consumer
    /// pair is intraprocedural — no interprocedural summary path is needed.
    pub fn fuse_maps(self) -> TlcFused {
        let mut inner = self.0;
        inner.tlc = tlc::fusion::run(inner.tlc);
        inner.tlc = tlc::if_over_producer::run(inner.tlc);
        inner.tlc = tlc::inline::run_reachable(inner.tlc);
        TlcFused(inner)
    }
}

/// TLC after map fusion
pub struct TlcFused(pub TlcEarlyInner);

impl std::ops::Deref for TlcFused {
    type Target = TlcEarlyInner;
    fn deref(&self) -> &TlcEarlyInner {
        &self.0
    }
}

impl TlcFused {
    /// Entry-boundary producer exposure (normalization, not residency).
    /// Inline helper calls whose result is an array producer while the caller's
    /// indexed uses are still local in the entry body.
    pub fn expose_entry_producer_helpers(self) -> TlcEntryProducersExposed {
        let mut inner = self.0;
        inner.tlc = tlc::materialize_entry_soacs::run(inner.tlc);
        TlcEntryProducersExposed(inner)
    }
}

/// TLC after ownership-driven rewrites. All `with` calls have
/// settled on either the functional or in-place intrinsic; eligible
/// SOACs (Phase C+) carry the consuming-input flag.
pub struct TlcOwnershipApplied(pub TlcEarlyInner);

impl std::ops::Deref for TlcOwnershipApplied {
    type Target = TlcEarlyInner;
    fn deref(&self) -> &TlcEarlyInner {
        &self.0
    }
}

impl TlcOutputsNormalized {
    /// Parallelize SOACs in compute entry points at the TLC level (terminal
    /// TLC pass in the experimental order). `disable` makes it a near no-op.
    pub fn parallelize_soacs(self, disable: bool) -> Result<TlcParallelized> {
        let mut inner = self.0;
        inner.tlc = tlc::if_over_producer::run(inner.tlc);
        let result = tlc::parallelize::run(inner.tlc, disable, &mut inner.auto_storage_binding_ids)?;
        Ok(TlcParallelized(TlcPipelineInner {
            tlc: result.program,
            pipeline: result.pipeline,
            type_table: inner.type_table,
            recognitions: result.recognitions,
            input_names: result.input_names,
            auto_storage_binding_ids: inner.auto_storage_binding_ids,
        }))
    }
}

/// TLC with compute entries normalised to unit-producing bodies built
/// from `TermKind::OutputSlotStore` per declared output.
pub struct TlcOutputsNormalized(pub TlcEarlyInner);

impl std::ops::Deref for TlcOutputsNormalized {
    type Target = TlcEarlyInner;
    fn deref(&self) -> &TlcEarlyInner {
        &self.0
    }
}

impl TlcGeneratedLambdasFolded {
    /// Ownership/liveness analysis + ownership-driven rewrites. Runs after
    /// gather residency and before output normalization, so the liveness walk
    /// never encounters an `OutputSlotStore` (which `normalize_outputs`
    /// introduces in the next step).
    pub fn apply_ownership(self) -> Result<TlcOwnershipApplied> {
        let mut inner = self.0;
        inner.tlc = tlc::ownership::apply_ownership(inner.tlc)?;
        Ok(TlcOwnershipApplied(inner))
    }
}

/// TLC after entry-boundary producer helpers have been exposed.
pub struct TlcEntryProducersExposed(pub TlcEarlyInner);

impl std::ops::Deref for TlcEntryProducersExposed {
    type Target = TlcEarlyInner;
    fn deref(&self) -> &TlcEarlyInner {
        &self.0
    }
}

impl TlcEntryProducersExposed {
    /// Fuse constant-index reads of directly nested elementwise producers
    /// before the gather residency pass can materialize them.
    pub fn fuse_static_indices(self) -> TlcStaticIndexFused {
        let mut inner = self.0;
        inner.tlc = tlc::static_index_fusion::run(inner.tlc);
        TlcStaticIndexFused(inner)
    }
}

/// TLC after static-index producer reads have been scalar-fused.
pub struct TlcStaticIndexFused(pub TlcEarlyInner);

impl std::ops::Deref for TlcStaticIndexFused {
    type Target = TlcEarlyInner;
    fn deref(&self) -> &TlcEarlyInner {
        &self.0
    }
}

impl TlcStaticIndexFused {
    /// Float runtime-indexed nested producers into local let-bound producers so
    /// gather residency can rewrite the indexed uses before defunctionalization.
    pub fn float_runtime_index_nested_producers(self) -> TlcRuntimeIndexProducersFloated {
        let mut inner = self.0;
        inner.tlc = tlc::runtime_index_producers::run(inner.tlc);
        TlcRuntimeIndexProducersFloated(inner)
    }
}

/// TLC after runtime-indexed nested producers have been let-bound.
pub struct TlcRuntimeIndexProducersFloated(pub TlcEarlyInner);

impl std::ops::Deref for TlcRuntimeIndexProducersFloated {
    type Target = TlcEarlyInner;
    fn deref(&self) -> &TlcEarlyInner {
        &self.0
    }
}

impl TlcRuntimeIndexProducersFloated {
    /// Plan and execute gather residency while producer lets and indexed uses
    /// are still in the same pre-defunctionalization term.
    pub fn plan_execute_gather_residency(self) -> TlcGathersLifted {
        let mut inner = self.0;
        inner.tlc = tlc::lift_gathers::run(inner.tlc, &mut inner.auto_storage_binding_ids);
        TlcGathersLifted(inner)
    }
}

/// TLC after randomly-indexed computed arrays have been lifted into gather
/// pre-pass entries (computed-array gathers now read from storage buffers).
pub struct TlcGathersLifted(pub TlcEarlyInner);

impl std::ops::Deref for TlcGathersLifted {
    type Target = TlcEarlyInner;
    fn deref(&self) -> &TlcEarlyInner {
        &self.0
    }
}

impl TlcOwnershipApplied {
    /// Normalise compute-entry bodies into per-slot `OutputSlotStore` writes.
    /// Runs after ownership, so `apply_ownership`'s liveness analysis sees the
    /// pre-slot-store form.
    pub fn normalize_outputs(self) -> Result<TlcOutputsNormalized> {
        let mut inner = self.0;
        inner.tlc = tlc::normalize_outputs::run(inner.tlc)
            .map_err(|e| crate::error::CompilerError::NormalizeOutputsError(format!("{e}"), None))?;
        Ok(TlcOutputsNormalized(inner))
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
        let specialized = tlc::specialize::run(inner.tlc);
        inner.tlc = tlc::monomorphize::run(specialized, &inner.schemes);
        TlcMonomorphized(inner)
    }
}

/// Shared payload for TLC states that carry just `{tlc, type_table}`
/// (`TlcMonomorphized`, `TlcGeneratedLambdasFolded`, `TlcSmallInlined`).
pub struct TlcLateInner {
    pub tlc: tlc::Program,
    pub type_table: TypeTable,
    pub auto_storage_binding_ids: IdSource<u32>,
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
    /// only at the producer site. Moved early (pre-fusion) in the experimental
    /// order so fusion sees concrete reps.
    pub fn rep_specialize(self) -> TlcRepSpecialized {
        let mut inner = self.0;
        inner.tlc = tlc::rep_specialize::run(inner.tlc);
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
        inner.tlc = tlc::inline::run_small(inner.tlc);
        TlcSmallInlined(inner)
    }
}

impl TlcGeneratedLambdasFolded {
    /// Eliminate unreachable defs (dead code elimination at TLC level).
    pub fn filter_reachable(self) -> TlcReachable {
        let inner = self.0;
        let tlc = tlc::inline::run_reachable(inner.tlc);
        TlcReachable(TlcPipelineInner {
            tlc,
            pipeline: pipeline_descriptor::PipelineDescriptor::default(),
            type_table: inner.type_table,
            recognitions: LookupMap::new(),
            input_names: LookupMap::new(),
            auto_storage_binding_ids: inner.auto_storage_binding_ids,
        })
    }

    /// Build the raw EGIR program. Callers chain the pipeline
    /// (`expand_soacs → materialize → optimize_skeleton → elaborate`)
    /// explicitly.
    pub fn to_egraph(mut self) -> std::result::Result<EgirScheduled, ConvertError> {
        let empty = LookupMap::new();
        let input_lens = tlc::input_slice_bounds::compute_for_program(&self.0.tlc);
        let inner = egir::from_tlc::run(
            &self.0.tlc,
            pipeline_descriptor::PipelineDescriptor::default(),
            &empty,
            &input_lens,
            &mut self.0.auto_storage_binding_ids,
        )?;
        let realized = EgirRaw(inner).realize_outputs()?;
        Ok(realized.segment(&empty).schedule(&mut self.0.auto_storage_binding_ids))
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
    /// Force-inline every user function whose body contains a SOAC. In the
    /// experimental order this runs post-monomorphize, so helpers are concrete
    /// and inlining opens every producer/consumer boundary for fusion.
    pub fn force_inline_soac_helpers(self) -> TlcSoacHelpersInlined {
        let mut inner = self.0;
        inner.tlc = tlc::inline::run_force_soac_helpers(inner.tlc);
        TlcSoacHelpersInlined(inner)
    }

    /// Direct shortcut: parallelize without the rep-specialize step.
    /// Off-mainline test paths and any caller that's certain its
    /// program doesn't ferry filter results across non-inlined call
    /// boundaries can use this; the verifier catches anything that
    /// slipped through. The canonical path goes through
    /// `.rep_specialize().parallelize_soacs(...)`.
    pub fn parallelize_soacs(self, disable: bool) -> Result<TlcParallelized> {
        let mut inner = self.0;
        inner.tlc = tlc::if_over_producer::run(inner.tlc);
        let result = tlc::parallelize::run(inner.tlc, disable, &mut inner.auto_storage_binding_ids)?;
        Ok(TlcParallelized(TlcPipelineInner {
            tlc: result.program,
            pipeline: result.pipeline,
            type_table: inner.type_table,
            recognitions: result.recognitions,
            input_names: result.input_names,
            auto_storage_binding_ids: inner.auto_storage_binding_ids,
        }))
    }

    /// Eliminate unreachable defs (dead code elimination at TLC level).
    pub fn filter_reachable(self) -> TlcReachable {
        let inner = self.0;
        let tlc = tlc::inline::run_reachable(inner.tlc);
        TlcReachable(TlcPipelineInner {
            tlc,
            pipeline: pipeline_descriptor::PipelineDescriptor::default(),
            type_table: inner.type_table,
            recognitions: LookupMap::new(),
            input_names: LookupMap::new(),
            auto_storage_binding_ids: inner.auto_storage_binding_ids,
        })
    }
}

/// Shared payload for TLC states carrying `{tlc, pipeline, type_table}`
/// (`TlcParallelized`, `TlcReachable`).
pub struct TlcPipelineInner {
    pub tlc: tlc::Program,
    pub pipeline: pipeline_descriptor::PipelineDescriptor,
    pub type_table: TypeTable,
    pub auto_storage_binding_ids: IdSource<u32>,
    /// Per-entry parallel recognition for EGIR to lower. Keyed by entry
    /// surface name; present only for parallelizable compute entries.
    pub recognitions: LookupMap<String, tlc::parallelize::EntryRecognition>,
    /// Source-parameter name for each storage `(set, binding)`, captured
    /// before parallelization replaced the original compute entries.
    /// `to_egraph` applies these to the finalized descriptor's input
    /// bindings, which the parallel path otherwise names positionally.
    pub input_names: LookupMap<(u32, u32), String>,
}

/// TLC after SOAC parallelization
pub struct TlcParallelized(pub TlcPipelineInner);

impl std::ops::Deref for TlcParallelized {
    type Target = TlcPipelineInner;
    fn deref(&self) -> &TlcPipelineInner {
        &self.0
    }
}

impl TlcParallelized {
    /// Eliminate unreachable defs (dead code elimination at TLC level).
    /// Drops the original `Abstract`-param defs whose call sites all
    /// got rewritten to specialized siblings by `rep_specialize`.
    pub fn filter_reachable(self) -> TlcReachable {
        let mut inner = self.0;
        inner.tlc = tlc::inline::run_reachable(inner.tlc);
        TlcReachable(inner)
    }

    pub fn to_egraph(self) -> std::result::Result<EgirScheduled, ConvertError> {
        let TlcPipelineInner {
            tlc,
            pipeline,
            recognitions,
            input_names,
            mut auto_storage_binding_ids,
            ..
        } = self.0;
        let input_lens = tlc::input_slice_bounds::compute_for_program(&tlc);
        let mut inner = egir::from_tlc::run(
            &tlc,
            pipeline,
            &recognitions,
            &input_lens,
            &mut auto_storage_binding_ids,
        )?;
        inner.pipeline.relabel_input_storage_names(&input_names);
        let realized = EgirRaw(inner).realize_outputs()?;
        Ok(realized.segment(&recognitions).schedule(&mut auto_storage_binding_ids))
    }
}

/// TLC after representation specialization (Phase 2 of array-variant-
/// abstract). Every `App` whose `Var(Symbol(callee))` arg at an
/// `Abstract`-typed position had a producer-known concrete variant now
/// invokes a specialized clone of the callee with that concrete
/// variant baked in. Lives between `materialize_entry_soacs` and
/// `parallelize_soacs` so the parallelizer sees concrete reps on every
/// call edge.
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
        inner.tlc = tlc::inline::run_large(inner.tlc);
        TlcGeneratedLambdasFolded(inner)
    }
}

/// TLC after dead code elimination. Always carries a `PipelineDescriptor`
/// (empty for non-parallelized paths).
pub struct TlcReachable(pub TlcPipelineInner);

impl std::ops::Deref for TlcReachable {
    type Target = TlcPipelineInner;
    fn deref(&self) -> &TlcPipelineInner {
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
    pub inner: TlcPipelineInner,
    pub input_lens: tlc::input_slice_bounds::ProgramBounds,
}

impl std::ops::Deref for TlcInputSliceBoundsInferred {
    type Target = TlcPipelineInner;
    fn deref(&self) -> &TlcPipelineInner {
        &self.inner
    }
}

impl TlcInputSliceBoundsInferred {
    pub fn to_egraph(self) -> std::result::Result<EgirScheduled, ConvertError> {
        let TlcPipelineInner {
            tlc,
            pipeline,
            recognitions,
            input_names,
            mut auto_storage_binding_ids,
            ..
        } = self.inner;
        let mut inner = egir::from_tlc::run(
            &tlc,
            pipeline,
            &recognitions,
            &self.input_lens,
            &mut auto_storage_binding_ids,
        )?;
        inner.pipeline.relabel_input_storage_names(&input_names);
        let realized = EgirRaw(inner).realize_outputs()?;
        Ok(realized.segment(&recognitions).schedule(&mut auto_storage_binding_ids))
    }
}

// =============================================================================
// EGIR typestate chain
//
// Four newtypes over a shared `EgirInner` (defined in `egir::program`).
// Transitions consume `self` and re-wrap the inner into the next newtype.
// Pass modules in `egir::*` are called per-body from inside the transitions
// and are unaware of the newtype wrapping.
// =============================================================================

/// Raw EGIR program, directly produced by TLC → EGIR conversion.
pub struct EgirRaw(EgirInner);

/// EGIR after entry-output realization. Every declared output has its
/// writes materialised as side effects against the bound storage view
/// (compute) or `OutputSlot` place (graphics). The body's `Return`
/// terminator carries no value. See `egir::realize_outputs`.
pub struct EgirOutputsRealized(EgirInner);

/// EGIR after recognized entry-tail SOACs have been reified as semantic
/// `SegMap`, `SegRed`, or `SegScan` operations. No dispatch schedule or scratch
/// storage has been chosen yet.
pub struct EgirSegmented(EgirInner);

/// EGIR after segmented operations have been assigned concrete kernel phases.
/// Pointwise SegMaps remain for lane expansion; SegReds have become phase
/// entries; unsupported SegScans have been restored to their serial fallback.
pub struct EgirScheduled(EgirInner);

/// EGIR after SOAC lowering: every `PendingSoac::{Map, Scan, Reduce, …}` in
/// the skeleton has been expanded to explicit loops / unrolled code.
pub struct EgirSoacExpanded(EgirInner);

/// EGIR after materialization: dynamic `Index` into non-materialized composite
/// values has been rewritten to `Materialize` + `DynamicExtract`. SPIR-V only.
pub struct EgirMaterialized(EgirInner);

/// EGIR after skeleton-CFG optimizations (LICM, dead-block elim, etc).
pub struct EgirSkelOptimized(EgirInner);

impl EgirRaw {
    /// Realize every entry's outputs into side-effect writes. For
    /// compute entries: walk `EgirEntry.slot_sources` and emit a DPS
    /// write per source against the slot's `OutputView` (Map/Scan
    /// retarget, fixed-aggregate element stores, scalar `Store` at
    /// index 0). For graphics entries: classify the body's
    /// `Return(Some(value))` and emit one store per declared
    /// `OutputSlot { index }`. After this pass the
    /// `realize_outputs::verify` invariant holds: no runtime-sized
    /// Composite array is reachable from any entry output. Runs
    /// before `parallelize` so the SOAC→OutputView rewrite precedes
    /// SOAC wrapping.
    pub fn realize_outputs(self) -> std::result::Result<EgirOutputsRealized, ConvertError> {
        let EgirRaw(mut inner) = self;
        egir::realize_outputs::run(&mut inner)?;
        Ok(EgirOutputsRealized(inner))
    }
}

impl EgirOutputsRealized {
    /// EGIR-side SOAC parallelization. Consumes `plans` from TLC analysis
    /// and tags each planned compute entry's tail SOAC for lane-indexed
    /// lowering downstream. Always called before `expand_soacs` — see the
    /// SOAC Parallelization Boundary section in the README.
    pub fn segment(
        self,
        recognitions: &LookupMap<String, tlc::parallelize::EntryRecognition>,
    ) -> EgirSegmented {
        let EgirOutputsRealized(mut inner) = self;
        egir::parallelize::reify(&mut inner, recognitions);
        EgirSegmented(inner)
    }
}

impl EgirSegmented {
    /// Choose concrete kernel phases for each semantic SegOp and allocate any
    /// scratch bindings required by that schedule.
    pub fn schedule(mut self, binding_ids: &mut IdSource<u32>) -> EgirScheduled {
        egir::parallelize::lower(&mut self.0, binding_ids);
        egir::publish::finalize_compute_io(&mut self.0.pipeline, &self.0.entry_points);
        EgirScheduled(self.0)
    }
}

impl EgirScheduled {
    pub fn expand_soacs(self) -> EgirSoacExpanded {
        let EgirScheduled(mut inner) = self;
        egir::soac_expand::run(&mut inner);
        EgirSoacExpanded(inner)
    }
}

impl EgirSoacExpanded {
    pub fn materialize(self) -> EgirMaterialized {
        let EgirSoacExpanded(mut inner) = self;
        egir::materialize::run(&mut inner);
        EgirMaterialized(inner)
    }

    pub fn optimize_skeleton(self) -> EgirSkelOptimized {
        let EgirSoacExpanded(mut inner) = self;
        egir::skel_opt::run(&mut inner);
        EgirSkelOptimized(inner)
    }
}

impl EgirMaterialized {
    pub fn optimize_skeleton(self) -> EgirSkelOptimized {
        let EgirMaterialized(mut inner) = self;
        egir::skel_opt::run(&mut inner);
        EgirSkelOptimized(inner)
    }
}

impl EgirSkelOptimized {
    /// Terminal step: lower each per-body e-graph to SSA and assemble the
    /// final `SsaConverted`.
    pub fn elaborate(self) -> SsaConverted {
        let EgirSkelOptimized(inner) = self;
        let (ssa, pipeline) = egir::elaborate::run_program(inner);
        SsaConverted { ssa, pipeline }
    }
}

/// TLC has been converted directly to SSA (via EGIR).
pub struct SsaConverted {
    pub ssa: ssa::types::Program,
    pub pipeline: pipeline_descriptor::PipelineDescriptor,
}

impl SsaConverted {
    /// Lower SSA to SPIR-V. Materialization (`Index` → `Materialize` +
    /// `DynamicExtract`) was already done by the EGIR pipeline if `to_egir`
    /// was called with `EgirOpts::for_spirv()`.
    pub fn lower(self) -> error::Result<Lowered> {
        egir::verify_no_abstract::run(&self.ssa)?;
        spirv::verify_buffer_layouts::run(&self.ssa)?;
        let spirv = spirv::lower_ssa_program(&self.ssa)?;
        Ok(Lowered {
            spirv,
            pipeline: self.pipeline,
        })
    }

    pub fn lower_wgsl(self) -> error::Result<String> {
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

/// Internal: run frontend + TLC pipeline with explicit
/// `disable_parallelize`, returning `TlcReachable`. Parameterized so the
/// canonical (`compile_thru_tlc`) and single-stage (`compile_thru_*
/// _single_stage`) variants share the same code path.
#[cfg(test)]
fn compile_thru_tlc_with(source: &str, disable_parallelize: bool) -> error::Result<TlcReachable> {
    let (mut node_counter, mut module_manager) = cached_compiler_init();
    let type_checked = Compiler::parse(source, &mut node_counter)?
        .elaborate_modules(&mut module_manager, &mut node_counter)?
        .resolve(&module_manager)?
        .fold_ast_constants()
        .type_check(&mut module_manager)?;
    Ok(type_checked
        .to_tlc(&module_manager, false)
        .pin_entry_regions()?
        .optimize_for_test(disable_parallelize))
}

/// Run the canonical TLC optimization pipeline (no compute parallelization,
/// no hole-filling) through `filter_reachable`.
#[cfg(test)]
pub fn compile_thru_tlc(source: &str) -> error::Result<TlcReachable> {
    compile_thru_tlc_with(source, false)
}

/// Internal: run all the way through EGIR + elaborate to SSA from a
/// pre-built `TlcReachable`. Both `compile_thru_ssa` and
/// `compile_thru_spirv_single_stage` build the SSA the same way; only
/// the upstream parallelize flag differs.
#[cfg(test)]
fn ssa_from_reachable(tlc: TlcReachable) -> std::result::Result<SsaConverted, Box<dyn std::error::Error>> {
    let raw = tlc.infer_input_slice_bounds().to_egraph()?;
    Ok(raw.expand_soacs().materialize().optimize_skeleton().elaborate())
}

/// Run all the way through EGIR + elaborate to SSA. Materialize is enabled
/// (matches the SPIR-V backend's requirements). Returns the boxed
/// `Result<_, dyn Error>` so callers see both compiler errors and EGIR
/// conversion errors uniformly.
#[cfg(test)]
pub fn compile_thru_ssa(source: &str) -> std::result::Result<SsaConverted, Box<dyn std::error::Error>> {
    ssa_from_reachable(compile_thru_tlc(source)?)
}

/// Run the full pipeline to a final SPIR-V binary.
#[cfg(test)]
pub fn compile_thru_spirv(source: &str) -> std::result::Result<Lowered, Box<dyn std::error::Error>> {
    Ok(compile_thru_ssa(source)?.lower()?)
}

/// Single-stage equivalent of `compile_thru_spirv`: matches the CLI's
/// `--single-stage` mode by disabling `parallelize_soacs`. Used by tests
/// that exercise compiler paths the default driver-equivalent doesn't
/// reach.
#[cfg(test)]
pub fn compile_thru_spirv_single_stage(
    source: &str,
) -> std::result::Result<Lowered, Box<dyn std::error::Error>> {
    Ok(ssa_from_reachable(compile_thru_tlc_with(source, true)?)?.lower()?)
}
