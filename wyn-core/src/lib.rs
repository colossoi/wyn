pub mod ast;
pub mod ast_renumber;
pub mod ast_type_holes;
pub mod binding_layout;
pub mod builtins;
pub mod diags;
pub mod elaborate_modules;
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
// The compiler uses typestate to enforce valid pass ordering. Every phase is
// driven by named functions that consume one generic program stage and return
// the next.
//
//   let (node_ids, module_manager) = init_compiler();
//
// FrontEnd (AST) stages:
//     let program = parser::parse(source, node_ids, module_manager)?;
//     let program = resolve_imports::resolve_imports(program, ...)?;
//     let program = elaborate_modules::elaborate_modules(program)?;
//     let program = name_resolution::resolve_names(program);
//     let program = resolve_resources::resolve_resources(program)?;
//     let program = ast_const_fold::fold_constants(program);
//     let program = resolve_placeholders::resolve_type_placeholders(program, ...);
//     let program = resolve_opens::resolve_opens(program, ...)?;
//     let program = types::run::type_check(program, ...)?;
//     let program = ast_type_holes::reject_type_holes(program)?;
//
// TLC stages (typed AST → semantic input):
//       tlc::lower_from_ast(program)    -> tlc::stage::Transformed
//       tlc::pin_entry_buffers(...)      -> tlc::stage::BuffersPinned
//       tlc::validate_ownership(...)     -> tlc::stage::OwnershipValidated
//       tlc::partial_eval(...)           -> tlc::stage::PartialEvaled
//       tlc::normalize_soacs(...)        -> tlc::stage::SoaNormalized
//       tlc::monomorphize(...)           -> tlc::stage::Monomorphized
//       tlc::rep_specialize(...)         -> tlc::stage::RepSpecialized
//       tlc::inline_small(...)           -> tlc::stage::SmallInlined
//       tlc::force_inline_soac_helpers(...)
//                                      -> tlc::stage::SoacHelpersInlined
//       tlc::renormalize_inlined_soa(...)
//                                      -> tlc::stage::InlinedSoaNormalized
//       tlc::canonicalize_conditional_producers(...)
//                                      -> tlc::stage::ConditionalProducersCanonicalized
//       tlc::normalize_soacs_to_anf(...) -> tlc::stage::SoacsAnfNormalized
//       tlc::float_runtime_index_nested_producers(...)
//                                      -> tlc::stage::RuntimeIndexProducersFloated
//       tlc::defunctionalize(...)        -> tlc::stage::Defunctionalized
//       tlc::fold_generated_lambdas(...) -> tlc::stage::GeneratedLambdasFolded
//       tlc::apply_ownership(...)        -> tlc::stage::OwnershipApplied
//       tlc::filter_reachable(...)       -> tlc::stage::Reachable
//       tlc::infer_input_slice_bounds(...)
//                                      -> tlc::stage::InputSliceBoundsInferred
//       to_egraph(...)                  -> egir::from_tlc::Converted
//
// EGIR stages:
//       egir::realize_outputs(...)       -> OutputsRealized
//       egir::reify_soacs(...)           -> Segmented
//       egir::optimize_semantics(...)     -> Optimized
//       egir::plan_logical_resources(...) -> ResourcesAllocated
//       egir::plan(..., profile)          -> Planned
//       lower_egir_to_ssa(...)            -> ssa::stage::Elaborated
//
// Backend:
//       lower_ssa_to_spirv(program) | lower_ssa_to_wgsl(program)
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
// TLC-based pipeline stages
// =============================================================================

#[cfg(test)]
pub(crate) fn optimize_tlc_for_test(program: tlc::stage::OwnershipValidated) -> tlc::stage::Reachable {
    let program = optimize_tlc_for_test_thru_soac_normalization(program);
    let program = tlc::float_runtime_index_nested_producers(program);
    let program = tlc::defunctionalize(program);
    let program = tlc::fold_generated_lambdas(program);
    let program = tlc::apply_ownership(program);
    tlc::filter_reachable(program)
}

#[cfg(test)]
pub(crate) fn optimize_tlc_for_test_thru_soac_normalization(
    program: tlc::stage::OwnershipValidated,
) -> tlc::stage::SoacsAnfNormalized {
    let program = tlc::partial_eval(program);
    let program = tlc::normalize_soacs(program);
    let program = tlc::monomorphize(program);
    let program = tlc::rep_specialize(program);
    let program = tlc::inline_small(program);
    let program = tlc::force_inline_soac_helpers(program);
    let program = tlc::renormalize_inlined_soa(program);
    let program = tlc::canonicalize_conditional_producers(program);
    tlc::normalize_soacs_to_anf(program)
}

/// Convert fully analyzed TLC into raw semantic EGIR.
pub fn to_egraph(
    program: tlc::stage::InputSliceBoundsInferred,
) -> std::result::Result<egir::from_tlc::Converted, ConvertError> {
    let binding_ids = program.global_context.auto_storage_binding_ids.clone();
    egir::from_tlc::run(&program, binding_ids, IdSource::new())
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

/// Run the physical EGIR passes and construct backend-bound SSA.
pub fn lower_egir_to_ssa(
    program: egir::parallelize::Planned,
) -> std::result::Result<ssa::stage::Elaborated, ConvertError> {
    let program = egir::expand_soacs(program)?;
    let program = egir::materialize_dynamic_extracts(program);
    let program = egir::partially_inline_calls(program)?;
    let program = egir::rewrite(program);
    let program = egir::optimize_skeleton(program);
    let program = egir::erase_resources(program)?;
    Ok(egir::elaborate(program))
}

/// Validate and lower elaborated SSA to SPIR-V.
pub fn lower_ssa_to_spirv(program: ssa::stage::Elaborated) -> error::Result<Lowered> {
    let program = ssa::prepare_spirv(program)?;
    let spirv = spirv::lower_ssa_program(&program)?;
    Ok(Lowered {
        spirv,
        pipeline: program.global_context.pipeline,
    })
}

/// Validate and lower elaborated SSA to WGSL.
pub fn lower_ssa_to_wgsl(program: ssa::stage::Elaborated) -> error::Result<String> {
    Ok(lower_ssa_to_wgsl_with_pipeline(program)?.wgsl)
}

/// Validate and lower elaborated SSA to WGSL while retaining the pipeline
/// descriptor required by host runtimes.
pub fn lower_ssa_to_wgsl_with_pipeline(program: ssa::stage::Elaborated) -> error::Result<LoweredWgsl> {
    let program = ssa::prepare_wgsl(program)?;
    let wgsl = wgsl::lower(&program)?;
    Ok(LoweredWgsl {
        wgsl,
        pipeline: program.global_context.pipeline,
    })
}

/// Final SPIR-V output
pub struct Lowered {
    pub spirv: Vec<u32>,
    pub pipeline: pipeline_descriptor::PipelineDescriptor,
}

/// Final WGSL output and its host-side execution descriptor.
pub struct LoweredWgsl {
    pub wgsl: String,
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
//   compile_thru_frontend  →  types::run::TypeChecked
//   compile_thru_tlc       →  tlc::stage::Reachable
//   compile_thru_ssa       →  ssa::Elaborated
//   compile_thru_spirv     →  Lowered              (final SPIR-V binary)
//
// These exist so test files don't have to enumerate every pass — when a
// new pass lands, only the helper that owns its milestone needs updating.
// Tests that need an off-milestone stop call the pass functions directly.

/// Run AST passes through type checking. Uses the cached prelude.
#[cfg(test)]
pub fn compile_thru_frontend(source: &str) -> error::Result<types::run::TypeChecked> {
    let (node_ids, module_manager) = cached_compiler_init();
    let program = parser::parse(source, node_ids, module_manager)?;
    let program = resolve_imports::resolve_imports(program, std::path::Path::new("."))?;
    let program = elaborate_modules::elaborate_modules(program)?;
    let program = name_resolution::resolve_names(program);
    let program = resolve_resources::resolve_resources(program)?;
    let program = ast_const_fold::fold_constants(program);
    let program = resolve_placeholders::resolve_type_placeholders(program);
    let program = resolve_opens::resolve_opens(program)?;
    types::run::type_check(program)
}

/// Run the canonical TLC optimization pipeline (no physical scheduling or
/// hole-filling) through `filter_reachable`.
#[cfg(test)]
pub fn compile_thru_tlc(source: &str) -> error::Result<tlc::stage::Reachable> {
    let type_checked = compile_thru_frontend(source)?;
    let program = ast_type_holes::reject_type_holes(type_checked)?;
    let program = tlc::lower_from_ast(program);
    let program = tlc::pin_entry_buffers(program)?;
    let program = tlc::validate_ownership(program)?;
    Ok(optimize_tlc_for_test(program))
}

/// Internal: run all the way through EGIR + elaborate to SSA from a
/// pre-built `tlc::stage::Reachable`. Both `compile_thru_ssa` and
/// `compile_thru_spirv_single_stage` build the SSA the same way; only
/// the downstream scheduling profile differs.
#[cfg(test)]
fn ssa_from_reachable(
    program: tlc::stage::Reachable,
    profile: LoweringProfile,
) -> std::result::Result<ssa::stage::Elaborated, Box<dyn std::error::Error>> {
    let program = tlc::infer_input_slice_bounds(program);
    let program = to_egraph(program)?;
    let program = egir::realize_outputs(program)?;
    let program = egir::reify_soacs(program);
    let program = egir::optimize_semantics(program);
    let program = egir::plan_logical_resources(program)?;
    let program = egir::plan(program, profile)?;
    Ok(lower_egir_to_ssa(program)?)
}

/// Run all the way through EGIR + elaborate to SSA. Materialize is enabled
/// (matches the SPIR-V backend's requirements). Returns the boxed
/// `Result<_, dyn Error>` so callers see both compiler errors and EGIR
/// conversion errors uniformly.
#[cfg(test)]
pub fn compile_thru_ssa(
    source: &str,
) -> std::result::Result<ssa::stage::Elaborated, Box<dyn std::error::Error>> {
    ssa_from_reachable(compile_thru_tlc(source)?, LoweringProfile::PORTABLE)
}

/// Run the full pipeline to a final SPIR-V binary.
#[cfg(test)]
pub fn compile_thru_spirv(source: &str) -> std::result::Result<Lowered, Box<dyn std::error::Error>> {
    Ok(lower_ssa_to_spirv(ssa_from_reachable(
        compile_thru_tlc(source)?,
        LoweringProfile::new(CodegenTarget::Spirv, SchedulePolicy::Parallel),
    )?)?)
}

/// Single-stage equivalent of `compile_thru_spirv`: matches the CLI's
/// `--single-stage` terminal scheduling policy.
#[cfg(test)]
pub fn compile_thru_spirv_single_stage(
    source: &str,
) -> std::result::Result<Lowered, Box<dyn std::error::Error>> {
    Ok(lower_ssa_to_spirv(ssa_from_reachable(
        compile_thru_tlc(source)?,
        LoweringProfile::new(CodegenTarget::Spirv, SchedulePolicy::Serial),
    )?)?)
}
