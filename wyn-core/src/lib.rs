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
mod scalar_eval;
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

use egir::from_tlc::ConvertError;
use wyn_base::{IdArena, IdSource};

use ast::NodeCounter;
use error::Result;

// =============================================================================
// Collection aliases
// =============================================================================

/// Use for maps whose iteration order affects program output (binding
/// allocation, code emission order, etc.). Insertion order is stable
/// across compiles; `HashMap`'s randomized hasher is not.
pub type StableMap<K, V> = indexmap::IndexMap<K, V>;

/// Use for sets whose iteration follows the values' [`Ord`] ordering.
pub type SortedSet<T> = std::collections::BTreeSet<T>;

/// Use for maps consulted only via `get`/`contains_key`. Iteration
/// order doesn't escape into observable output, so `HashMap`'s
/// per-process random hash is fine — and we get the slightly faster
/// lookups in exchange.
pub type LookupMap<K, V> = std::collections::HashMap<K, V>;

/// Set companion to [`LookupMap`].
pub type LookupSet<T> = std::collections::HashSet<T>;

/// Stable compiler-internal identity of a callable body.
///
/// Allocated once while TLC is converted to EGIR and carried unchanged through
/// EGIR, SSA, and backend lowering. Source-level callables use `SymbolId`
/// through TLC; the conversion between the two realms is structural. Human-readable
/// function names and extern linkage symbols are metadata only.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FunctionId(u32);

impl From<u32> for FunctionId {
    fn from(index: u32) -> Self {
        Self(index)
    }
}

impl FunctionId {
    #[cfg(test)]
    pub(crate) const fn from_index(index: u32) -> Self {
        Self(index)
    }
}

impl std::fmt::Display for FunctionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "fn#{}", self.0)
    }
}

/// Stable compiler-internal identity of a program-level constant.
///
/// Constants have their own realm because their source semantics are values;
/// a backend may choose to implement one as a zero-argument function without
/// turning it into a function identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GlobalId(u32);

impl From<u32> for GlobalId {
    fn from(index: u32) -> Self {
        Self(index)
    }
}

impl std::fmt::Display for GlobalId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "global#{}", self.0)
    }
}

/// Stable compiler-internal identity of an entry point.
///
/// The entry's source/emitted name remains host-facing ABI metadata. Internal
/// pipeline and scheduling edges use this ID after the descriptor has been
/// resolved once at the frontend boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EntryId(u32);

impl From<u32> for EntryId {
    fn from(index: u32) -> Self {
        Self(index)
    }
}

impl EntryId {
    #[cfg(test)]
    pub(crate) const fn from_index(index: usize) -> Self {
        Self(index as u32)
    }

    pub(crate) const fn index(self) -> usize {
        self.0 as usize
    }
}

impl std::fmt::Display for EntryId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "entry#{}", self.0)
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
/// uniform / texture / sampler resource. Pre-allocation semantic EGIR keeps
/// this interface identity directly; logical-resource allocation replaces it
/// with `ResourceId`. Deliberately no
/// `Default` impl —
/// `BindingRef { set: 0, binding: 0 }` is a meaningful binding, and a
/// default value would silently mask construction bugs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
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

// Re-export key types for the public API
pub use ast::TypeName;
pub use polytype::Context as PolytypeContext;
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
//       egir::reify_soacs(...)           -> Segmented
//       egir::optimize_semantic_operations(...)
//                                      -> SemanticOperationsOptimized
//       egir::lift_stage_uniform_values(...)
//                                      -> Optimized
//       egir::plan_logical_resources(...) -> ResourcesAllocated
//       egir::plan(..., profile)          -> Planned
//       lower_egir_to_ssa(...)            -> ssa::stage::Elaborated
//
// Backend:
//       ssa::filter_reachable(...)         -> ssa::stage::Reachable
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
    egir::from_tlc::convert_program(&program, binding_ids, IdSource::new())
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
    let program = egir::partially_inline_calls(program)?;
    let program = egir::materialize_dynamic_extracts(program);
    let program = egir::rewrite(program);
    let program = egir::optimize_skeleton(program);
    let program = egir::erase_resources(program)?;
    Ok(egir::elaborate(program))
}

/// Validate and lower elaborated SSA to SPIR-V.
pub fn lower_ssa_to_spirv(program: ssa::stage::Elaborated) -> error::Result<Lowered> {
    let program = ssa::filter_reachable(program);
    let program = ssa::prepare_spirv(program)?;
    let spirv = spirv::lower_ssa_program(&program)?;
    Ok(Lowered {
        spirv,
        pipeline: program.global_context.pipeline,
    })
}

/// Validate and lower elaborated SSA to WGSL.
pub fn lower_ssa_to_wgsl(program: ssa::stage::Elaborated) -> error::Result<String> {
    lower_ssa_to_wgsl_with_options(program, wgsl::WgslOptions::default())
}

/// Validate and lower elaborated SSA to WGSL using an explicit backend
/// legalization policy.
pub fn lower_ssa_to_wgsl_with_options(
    program: ssa::stage::Elaborated,
    options: wgsl::WgslOptions,
) -> error::Result<String> {
    Ok(lower_ssa_to_wgsl_with_pipeline_and_options(program, options)?.wgsl)
}

/// Validate and lower elaborated SSA to WGSL while retaining its runtime
/// pipeline descriptor.
pub fn lower_ssa_to_wgsl_with_pipeline(program: ssa::stage::Elaborated) -> error::Result<LoweredWgsl> {
    lower_ssa_to_wgsl_with_pipeline_and_options(program, wgsl::WgslOptions::default())
}

/// Validate and lower elaborated SSA to WGSL while retaining its runtime
/// pipeline descriptor and using an explicit backend legalization policy.
pub fn lower_ssa_to_wgsl_with_pipeline_and_options(
    program: ssa::stage::Elaborated,
    options: wgsl::WgslOptions,
) -> error::Result<LoweredWgsl> {
    let program = ssa::filter_reachable(program);
    let program = ssa::prepare_wgsl(program)?;
    let lowered = wgsl::ssa_lowering::lower_with_abi(&program, options)?;
    let mut pipeline = program.global_context.pipeline;
    adapt_pipeline_descriptor_for_wgsl(&mut pipeline, &lowered.parameter_blocks)?;
    Ok(LoweredWgsl {
        wgsl: lowered.source,
        pipeline,
    })
}

/// Rewrite target-neutral push-constant contracts to the read-only storage
/// blocks actually declared by the WGSL backend. SPIR-V lowering deliberately
/// bypasses this adaptation and retains native push constants.
fn adapt_pipeline_descriptor_for_wgsl(
    descriptor: &mut pipeline_descriptor::PipelineDescriptor,
    parameter_blocks: &[wgsl::ssa_lowering::ParameterBlock],
) -> error::Result<()> {
    use pipeline_descriptor::{
        Access, Binding, BufferLen, BufferUsage, DispatchLen, DispatchSize, Pipeline,
    };

    for pipeline in &mut descriptor.pipelines {
        let Pipeline::Compute(compute) = pipeline else {
            continue;
        };
        let blocks = parameter_blocks
            .iter()
            .filter(|block| compute.stages.iter().any(|stage| stage.entry_point == block.entry_point))
            .collect::<Vec<_>>();

        let push_constants = compute
            .bindings
            .iter()
            .filter_map(|binding| match binding {
                Binding::PushConstant { offset, size, name } => Some((*offset, *size, name.clone())),
                _ => None,
            })
            .collect::<Vec<_>>();
        for (offset, size, name) in &push_constants {
            if !blocks.iter().any(|block| {
                block.members.iter().any(|member| {
                    member.push_constant_offset == *offset && member.name == *name && member.size == *size
                })
            }) {
                return Err(err_wgsl!(
                    "pipeline push constant '{}' at offset {} has no WGSL storage parameter",
                    name,
                    offset
                ));
            }
        }

        let old_bindings = std::mem::take(&mut compute.bindings);
        let mut old_to_new = vec![None; old_bindings.len()];
        for (old_index, binding) in old_bindings.into_iter().enumerate() {
            if matches!(binding, Binding::PushConstant { .. }) {
                continue;
            }
            old_to_new[old_index] = Some(compute.bindings.len());
            compute.bindings.push(binding);
        }

        let mut block_binding_indices = Vec::with_capacity(blocks.len());
        for block in &blocks {
            let binding_index = compute.bindings.len();
            let name = if let [member] = block.members.as_slice() {
                member.name.clone()
            } else {
                format!("{}_parameters", block.entry_point)
            };
            compute.bindings.push(Binding::StorageBuffer {
                set: block.set,
                binding: block.binding,
                access: Access::ReadOnly,
                usage: BufferUsage::Input,
                name,
                resource: None,
                length: Some(BufferLen::Fixed {
                    bytes: u64::from(block.size),
                }),
                members: block
                    .members
                    .iter()
                    .map(|member| pipeline_descriptor::UniformMember {
                        name: member.name.clone(),
                        offset: member.offset,
                        size: member.size,
                    })
                    .collect(),
            });
            block_binding_indices.push(binding_index);
        }

        for stage in &mut compute.stages {
            stage.reads = stage
                .reads
                .iter()
                .filter_map(|&old_index| old_to_new.get(old_index).copied().flatten())
                .collect();
            stage.writes = stage
                .writes
                .iter()
                .filter_map(|&old_index| old_to_new.get(old_index).copied().flatten())
                .collect();

            for (block_index, block) in blocks.iter().enumerate() {
                if block.entry_point == stage.entry_point {
                    let binding_index = block_binding_indices[block_index];
                    if !stage.reads.contains(&binding_index) {
                        stage.reads.push(binding_index);
                    }
                }
            }

            if let DispatchSize::DerivedFrom { len, .. } = &mut stage.dispatch_size {
                let DispatchLen::PushConstant { offset } = *len else {
                    continue;
                };
                let Some((block, member)) = blocks.iter().find_map(|block| {
                    (block.entry_point == stage.entry_point)
                        .then(|| block.members.iter().find(|member| member.push_constant_offset == offset))
                        .flatten()
                        .map(|member| (*block, member))
                }) else {
                    return Err(err_wgsl!(
                        "entry '{}': dynamic dispatch push constant at offset {} has no WGSL storage parameter",
                        stage.entry_point,
                        offset
                    ));
                };
                *len = DispatchLen::StorageBuffer {
                    set: block.set,
                    binding: block.binding,
                    offset: member.offset,
                };
            }
        }
    }

    if descriptor.pipelines.iter().any(|pipeline| match pipeline {
        Pipeline::Compute(compute) => {
            compute.bindings.iter().any(|binding| matches!(binding, Binding::PushConstant { .. }))
        }
        Pipeline::Graphics(graphics) => {
            graphics.bindings.iter().any(|binding| matches!(binding, Binding::PushConstant { .. }))
        }
    }) {
        return Err(err_wgsl!(
            "WGSL pipeline descriptor still contains a push constant with no WebGPU binding"
        ));
    }

    descriptor.rebuild_frame_graph();
    Ok(())
}

/// Final SPIR-V output
pub struct Lowered {
    pub spirv: Vec<u32>,
    pub pipeline: pipeline_descriptor::PipelineDescriptor,
}

/// Final WGSL output and the runtime contract for dispatching it.
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
    let program = tlc::lower_from_ast(program)?;
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
    let program = egir::reify_soacs(program);
    let program = egir::optimize_semantic_operations(program);
    let program = egir::lift_stage_uniform_values(program);
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
