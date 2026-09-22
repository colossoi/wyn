pub mod ast;
mod ast_const_fold;
pub mod ast_renumber;
pub mod ast_type_holes;
pub mod binding_layout;
pub mod builtins;
pub mod diags;
mod elaborate_modules;
pub mod error;
pub mod flow;
mod frontend;
pub mod interface;
pub mod kernel_graph;
pub mod lexer;
mod name_resolution;
pub mod op;
mod parser;
pub mod pattern;
mod resolve_imports;
mod resolve_opens;
mod resolve_placeholders;
mod resolve_resources;
mod scalar_eval;
pub mod scope;
pub mod semantic_modules;
pub mod ssa;
pub mod types;

pub use error::{CompilationFailure, LoadModulesError};
pub use frontend::{initialize_frontend, ParsedModules};

pub mod lowering_common;
pub mod name_registry;
pub mod tlc;

pub mod egglog;
/// Portable host programs and shader interface metadata.
pub use wyn_host as host;
pub mod spirv;
pub mod structured;
pub mod wgsl;

#[cfg(test)]
mod test_pipeline;

#[cfg(test)]
mod slice_range_tests;

use wyn_base::IdArena;

use ast::NodeCounter;
use host::ScalarSource;
use std::collections::BTreeMap;
// =============================================================================
// Collection aliases
// =============================================================================

pub use wyn_base::{LookupMap, LookupSet, SortedSet, StableMap};

/// Stable compiler-internal identity of a callable body.
///
/// Allocated at SSA construction and carried unchanged through backend lowering.
/// Source-level callables use `SymbolId` through TLC; the conversion between the
/// two realms is structural. Human-readable
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
}

impl std::fmt::Display for EntryId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "entry#{}", self.0)
    }
}

pub use wyn_base::Interner;

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
/// uniform / texture / sampler resource. Source interfaces keep
/// this binding identity; scheduled resources also carry `ResourceId` values.
/// Deliberately no
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

/// Target-independent identity of a semantic storage resource. Identities are
/// assigned by the selected compiler route. Callers can observe its dense
/// index but cannot manufacture one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ResourceId(u32);

impl ResourceId {
    /// A finalized egglog resource uses its sidecar buffer arena identity.
    pub(crate) const fn from_egglog_buffer(index: u32) -> Self {
        Self(index)
    }

    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// Access to one resource in a compiler route's resource namespace.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceUse<R> {
    pub resource: R,
    pub access: ResourceAccess,
}

impl<R: Copy + Ord> ResourceUse<R> {
    /// Combine accesses to the same resource in ascending resource order.
    pub fn merge(a: &[Self], b: &[Self]) -> Vec<Self> {
        let mut merged: BTreeMap<R, ResourceAccess> = BTreeMap::new();
        for resource in a.iter().chain(b) {
            merged
                .entry(resource.resource)
                .and_modify(|access| *access = access.merge(resource.access))
                .or_insert(resource.access);
        }
        merged.into_iter().map(|(resource, access)| Self { resource, access }).collect()
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
/// placeholder. Fallible passes can instead report a missing symbol as a
/// structured compiler error.
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
//   let modules = ParsedModules::load(plan, options)?;
//   let program = modules.type_check()?;
//     let program = ast_type_holes::reject_type_holes(program)?;
//
// TLC stages (typed AST → semantic input):
//       tlc::lower_from_ast(program)    -> tlc::stage::Transformed
//       tlc::validate_ownership(...)     -> tlc::stage::OwnershipValidated
//       tlc::partial_eval(...)           -> tlc::stage::PartialEvaled
//       tlc::extract_stages(...)          -> tlc::stage::StagesExtracted
//       tlc::pin_entry_buffers(...)       -> tlc::stage::BuffersPinned
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
// Egglog stages:
//       egglog::from_tlc(&program)       -> Imported
//       egglog::fuse(...)                -> Fused
//       egglog::insert_expressions(...)  -> Expressions
//       egglog::simplify(..., algebra)   -> Simplified
//       egglog::place(...)               -> Placed
//       egglog::schedule(..., topology)  -> Scheduled
//       egglog::to_ssa(&program, target) -> ssa::stage::Elaborated
//
// Backend:
//       ssa::filter_reachable(...)         -> ssa::stage::Reachable
//       lower_ssa_to_spirv(program) | lower_ssa_to_wgsl(program)
//
// Tests should prefer the `compile_thru_*` helpers below, which subsume
// the chain up to a milestone and centralize updates as new passes land.

/// Source-language features enabled for one compilation.
///
/// Graphics vocabulary is opt-in. When disabled, graphics-specific type
/// and value spellings are not reserved and resolve exactly like any other
/// user-defined or undefined identifier.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CompilerOptions {
    pub graphics: bool,
}

// =============================================================================
// TLC-based pipeline stages
// =============================================================================

#[cfg(test)]
pub(crate) fn optimize_tlc_for_test(
    program: tlc::stage::OwnershipValidated,
) -> error::Result<tlc::stage::Reachable> {
    let program = optimize_tlc_for_test_thru_soac_normalization(program)?;
    let program = tlc::float_runtime_index_nested_producers(program);
    let program = tlc::defunctionalize(program);
    let program = tlc::fold_generated_lambdas(program);
    let program = tlc::apply_ownership(program);
    Ok(tlc::filter_reachable(program))
}

#[cfg(test)]
pub(crate) fn optimize_tlc_for_test_thru_soac_normalization(
    program: tlc::stage::OwnershipValidated,
) -> error::Result<tlc::stage::SoacsAnfNormalized> {
    let program = tlc::partial_eval(program);
    let program = tlc::extract_stages(program)?;
    let program = tlc::pin_entry_buffers(program)?;
    let program = tlc::normalize_soacs(program);
    let program = tlc::monomorphize(program)?;
    let program = tlc::rep_specialize(program);
    let program = tlc::inline_small(program);
    let program = tlc::force_inline_soac_helpers(program);
    let program = tlc::renormalize_inlined_soa(program);
    let program = tlc::canonicalize_conditional_producers(program);
    Ok(tlc::normalize_soacs_to_anf(program))
}

/// Target capabilities selected before scheduled blocks are lowered to SSA.
/// `Portable` deliberately uses the common SPIR-V/WGSL capability subset and
/// is retained for tools and tests that want to inspect one shared SSA module.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CodegenTarget {
    Portable,
    Spirv,
    Wgsl,
}

/// Whether lowering may introduce pipeline structure beyond the stages and
/// resources authored by the source program.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PipelineTopologyPolicy {
    AllowGenerated,
    AuthoredOnly,
}

/// Validate and lower elaborated SSA to SPIR-V.
pub fn lower_ssa_to_spirv(program: ssa::stage::Elaborated) -> error::Result<Lowered> {
    let program = ssa::optimize(program);
    let program = ssa::place_floating(program)?;
    let program = ssa::filter_reachable(program);
    let program = ssa::prepare_spirv(program)?;
    let spirv = spirv::lower_ssa_program(&program)?;
    Ok(Lowered {
        spirv,
        program: host::Program::new(program.global_context.pipeline)?,
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
    Ok(lower_ssa_to_wgsl_with_program_and_options(program, options)?.wgsl)
}

/// Validate and lower elaborated SSA to WGSL while retaining its runtime
/// host program.
pub fn lower_ssa_to_wgsl_with_program(program: ssa::stage::Elaborated) -> error::Result<LoweredWgsl> {
    lower_ssa_to_wgsl_with_program_and_options(program, wgsl::WgslOptions::default())
}

/// Validate and lower elaborated SSA to WGSL while retaining its runtime
/// host program and using an explicit backend legalization policy.
pub fn lower_ssa_to_wgsl_with_program_and_options(
    program: ssa::stage::Elaborated,
    options: wgsl::WgslOptions,
) -> error::Result<LoweredWgsl> {
    let program = ssa::optimize(program);
    let program = ssa::place_floating(program)?;
    let program = ssa::filter_reachable(program);
    let program = ssa::prepare_wgsl(program)?;
    let lowered = wgsl::ssa_lowering::lower_with_abi(&program, options)?;
    let mut pipeline = program.global_context.pipeline;
    adapt_host_interface_for_wgsl(&mut pipeline, &lowered.parameter_blocks)?;
    Ok(LoweredWgsl {
        wgsl: lowered.source,
        program: host::Program::new(pipeline)?,
    })
}

/// Rewrite target-neutral push-constant contracts to the read-only storage
/// blocks actually declared by the WGSL backend. SPIR-V lowering deliberately
/// bypasses this adaptation and retains native push constants.
fn adapt_host_interface_for_wgsl(
    descriptor: &mut host::ModuleInterface,
    parameter_blocks: &[wgsl::ssa_lowering::ParameterBlock],
) -> error::Result<()> {
    use host::{
        Access, Binding, BufferLen, BufferUsage, DispatchLen, DispatchSize, HostSizeInput, Pipeline,
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
                // Parameter blocks belong to one entry, even when two entries
                // both have a single parameter with the same source name.
                resource: Some(format!("{}_parameters", block.entry_point)),
                length: Some(BufferLen::Fixed {
                    bytes: u64::from(block.size),
                }),
                members: block
                    .members
                    .iter()
                    .map(|member| host::UniformMember {
                        name: member.name.clone(),
                        offset: member.offset,
                        size: member.size,
                    })
                    .collect(),
            });
            block_binding_indices.push(binding_index);
        }

        for binding in &mut compute.bindings {
            let Binding::StorageBuffer {
                length: Some(length), ..
            } = binding
            else {
                continue;
            };
            for input in length.inputs_mut() {
                let HostSizeInput::PushConstant {
                    name,
                    push_constant_offset,
                    scalar,
                } = input
                else {
                    continue;
                };
                let Some((block, member)) = blocks.iter().find_map(|block| {
                    block
                        .members
                        .iter()
                        .find(|m| m.name == *name && m.push_constant_offset == *push_constant_offset)
                        .map(|m| (*block, m))
                }) else {
                    return Err(err_wgsl!("host size input '{}' has no WGSL parameter", name));
                };
                *input = HostSizeInput::Uniform {
                    name: name.clone(),
                    set: block.set,
                    binding: block.binding,
                    offset: member.offset,
                    scalar: *scalar,
                };
            }
        }

        for task in &mut descriptor.scalar_tasks {
            if !compute.stages.iter().any(|stage| stage.entry_point == task.stage) {
                continue;
            }
            let mut missing = None;
            task.value.reads_mut(&mut |source, offset| {
                let ScalarSource::PushConstant { name, offset: base } = source else {
                    return;
                };
                let found = blocks.iter().find_map(|block| {
                    block
                        .members
                        .iter()
                        .find(|member| member.name == *name && member.push_constant_offset == *base)
                        .map(|member| (*block, member))
                });
                if let Some((block, member)) = found {
                    *offset += member.offset;
                    *source = ScalarSource::Binding {
                        set: block.set,
                        binding: block.binding,
                    };
                } else {
                    missing = Some(name.clone());
                }
            });
            if let Some(name) = missing {
                return Err(err_wgsl!("host scalar '{}' has no WGSL parameter", name));
            }
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
                let Some((block, storage_offset)) = blocks.iter().find_map(|block| {
                    (block.entry_point == stage.entry_point)
                        .then(|| {
                            block.members.iter().find_map(|member| {
                                let relative = offset.checked_sub(member.push_constant_offset)?;
                                let end = relative.checked_add(4)?;
                                (end <= member.size)
                                    .then(|| {
                                        member.offset.checked_add(relative).map(|offset| (*block, offset))
                                    })
                                    .flatten()
                            })
                        })
                        .flatten()
                }) else {
                    return Err(err_wgsl!(
                        "entry '{}': dynamic dispatch push constant at offset {} has no containing WGSL storage parameter",
                        stage.entry_point,
                        offset
                    ));
                };
                *len = DispatchLen::StorageBuffer {
                    set: block.set,
                    binding: block.binding,
                    offset: storage_offset,
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
            "WGSL host program still contains a push constant with no WebGPU binding"
        ));
    }

    descriptor.rebuild_frame_graph();
    Ok(())
}

/// Final SPIR-V output
pub struct Lowered {
    pub spirv: Vec<u32>,
    pub program: host::Program,
}

/// Final WGSL output and the runtime contract for dispatching it.
pub struct LoweredWgsl {
    pub wgsl: String,
    pub program: host::Program,
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
    compile_thru_frontend_with_options(source, CompilerOptions { graphics: true })
}

#[cfg(test)]
pub fn compile_thru_frontend_with_options(
    source: &str,
    options: CompilerOptions,
) -> error::Result<types::run::TypeChecked> {
    let modules = test_pipeline::try_load_test_modules(source, options)?;
    modules.type_check().map_err(error::CompilationFailure::into_error)
}

/// Run the canonical TLC optimization pipeline (no physical scheduling or
/// hole-filling) through `filter_reachable`.
#[cfg(test)]
pub fn compile_thru_tlc(source: &str) -> error::Result<tlc::stage::Reachable> {
    let type_checked = compile_thru_frontend(source)?;
    let program = ast_type_holes::reject_type_holes(type_checked)?;
    let program = tlc::lower_from_ast(program)?;
    let program = tlc::validate_ownership(program)?;
    optimize_tlc_for_test(program)
}

/// Run the egglog pipeline to backend-bound SSA.
#[cfg(test)]
pub fn compile_thru_ssa(
    source: &str,
) -> std::result::Result<ssa::stage::Elaborated, Box<dyn std::error::Error>> {
    compile_thru_ssa_for_target(source, CodegenTarget::Portable)
}

#[cfg(test)]
fn compile_thru_ssa_for_target(
    source: &str,
    target: CodegenTarget,
) -> std::result::Result<ssa::stage::Elaborated, Box<dyn std::error::Error>> {
    let program = tlc::infer_input_slice_bounds(compile_thru_tlc(source)?);
    let program = egglog::from_tlc(&program)?;
    let program = egglog::fuse(program)?;
    let program = egglog::insert_expressions(program)?;
    let program = egglog::simplify(program, false)?;
    let program = egglog::place(program)?;
    let program = egglog::schedule(program, PipelineTopologyPolicy::AllowGenerated)?;
    Ok(egglog::to_ssa(&program, target)?)
}

/// Run the full pipeline to a final SPIR-V binary.
#[cfg(test)]
pub fn compile_thru_spirv(source: &str) -> std::result::Result<Lowered, Box<dyn std::error::Error>> {
    Ok(lower_ssa_to_spirv(compile_thru_ssa_for_target(
        source,
        CodegenTarget::Spirv,
    )?)?)
}

#[cfg(test)]
mod literal_expansion_tests;

#[cfg(test)]
mod host_tests;

#[cfg(test)]
mod host_interp_tests;
#[cfg(test)]
mod host_scalar_tests;
