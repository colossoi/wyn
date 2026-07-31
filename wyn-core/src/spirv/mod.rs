//! SPIR-V code generation backend.
//!
//! This module contains the lowering pass from SSA to SPIR-V.

// `builder::TypeId` / `builder::ConstId` / etc. path literals
// throughout this module reach the typed wrapper that lives in the
// `wyn-spirv` crate (renamed to `wspirv` in our `Cargo.toml`).
use wspirv as builder;
mod entry;
mod lower;
mod lower_builtin;
mod lower_index;
mod lower_ops;
#[cfg(test)]
mod lowering_tests;
mod pow;
mod storage;
mod types_lowering;
pub mod verify_buffer_layouts;
use crate::builtins::catalog;
use crate::{LookupMap, LookupSet};

use crate::ast::{Span, TypeName};
use crate::builtins::lowering::{BuiltinLowering, PrimOp};
use crate::error::Result;
use crate::interface::IoDecoration;
use crate::ssa::layout::{buffer_array_strides, std430_alignment};
use crate::ssa::storage_function_variants::{FunctionEmissionId, StorageFunctionVariants};
use crate::ssa::types::{
    BlockId, ConstantValue, ControlHeader, EntryPoint, ExecutionModel, FuncBody, Function, InstKind,
    Terminator, ValueId, ValueRef, WynInstNode,
};
use crate::types::TypeExt;
use crate::{bail_spirv, bail_spirv_at, err_spirv, err_spirv_at, types, BindingRef};
use polytype::Type as PolyType;
use wspirv::binary::Assemble;
use wspirv::dr::{InsertPoint, Operand};
use wspirv::spirv::{self, StorageClass};

// =============================================================================
// Constructor - SPIR-V Builder Wrapper
// =============================================================================

/// Constructor wraps wspirv::Builder with an ergonomic API that handles:
/// - Automatic variable hoisting to function entry block
/// Cache key for interface block types (push constants, storage buffers, uniforms).
/// These are distinct from plain struct types even when member types match.
#[derive(Clone, Hash, PartialEq, Eq)]
struct InterfaceBlockKey {
    kind: InterfaceBlockKind,
    /// Member types + offsets + optional array strides
    members: Vec<(spirv::Word, u32)>, // (type, offset)
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
enum InterfaceBlockKind {
    PushConstant,
    /// A record-typed `#[uniform]` block: the record's fields are the
    /// block's members, laid out std140.
    Uniform,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct StorageBufferUse {
    binding: BindingRef,
    writable: bool,
}

/// - Block management with implicit branch from variables block to code
/// - Value and type caching
struct Constructor {
    builder: builder::SpirvBuilder,

    // Well-known scalar type ids. `SpirvBuilder` owns the canonical
    // copy and dedups; these mirrors are populated at init so call
    // sites can read them without borrowing the builder.
    void_type: spirv::Word,
    bool_type: spirv::Word,
    i32_type: spirv::Word,
    u32_type: spirv::Word,
    f32_type: spirv::Word,

    // GLSL extended instruction set
    glsl_ext_inst_id: spirv::Word,

    // Top-level polytype → SPIR-V memoization (subsumes type + constant dedup for wyn types)
    polytype_cache: LookupMap<PolyType<TypeName>, spirv::Word>,

    // Interface-block + nested-array lookups stay here for now —
    // they're entangled with the compiler's `PolyType` walks
    // (interface members need `apply_buffer_array_strides`, which is
    // PolyType-driven). The simpler block wrappers and structural
    // type caches live on `SpirvBuilder`.
    interface_block_cache: LookupMap<InterfaceBlockKey, spirv::Word>,

    // Entry point interface tracking
    entry_point_interfaces: LookupMap<crate::EntryId, Vec<spirv::Word>>,

    /// Access-qualified storage-buffer globals. The same descriptor slot can
    /// be writable in a compute prepass and read-only in a graphics entry.
    storage_buffers: LookupMap<StorageBufferUse, (spirv::Word, spirv::Word, spirv::Word)>,
    current_storage_accesses: LookupMap<BindingRef, crate::ResourceAccess>,
    /// Per-entry bindings keyed by the SSA parameter they initialize.
    /// Names are emitted/debug metadata only; they are never identity here.
    env: LookupMap<ValueId, spirv::Word>,
    current_functions: LookupMap<crate::FunctionId, spirv::Word>,
    emitted_functions: LookupMap<FunctionEmissionId, spirv::Word>,
    entry_functions: LookupMap<crate::EntryId, spirv::Word>,
    globals: LookupMap<crate::GlobalId, spirv::Word>,

    /// Storage-image globals: (set, binding) -> (image `OpVariable`, image type).
    /// Predeclared from entry resource metadata before function bodies are
    /// lowered. Binding-qualified image operations load the global from here;
    /// no opaque image handle enters a runtime function signature.
    storage_images: LookupMap<BindingRef, (spirv::Word, spirv::Word)>,

    /// GlobalInvocationId variable for compute shaders (set during entry point setup)
    global_invocation_id: Option<spirv::Word>,

    /// LocalInvocationId variable for compute shaders (set during entry point setup)
    local_invocation_id: Option<spirv::Word>,

    /// NumWorkgroups variable for compute shaders (set during entry point setup)
    num_workgroups: Option<spirv::Word>,

    /// Shared push constant variable (at most one per SPIR-V module)
    push_constant_var: Option<spirv::Word>,

    /// Imported SPIR-V functions keyed by compiler-internal callable identity.
    linked_functions: LookupMap<crate::FunctionId, spirv::Word>,
    /// Imported functions indexed by their explicit external ABI linkage symbol.
    linked_functions_by_linkage: LookupMap<String, spirv::Word>,

    /// Compiler-generated integer-pow helpers (see `spirv::pow`), keyed
    /// by `signed`. Emitted once per module after function forward
    /// declarations; `PrimOp::IntPow` lowers to `OpFunctionCall` against
    /// the cached id.
    int_pow_functions: LookupMap<bool, spirv::Word>,

    /// Output variables for the current entry point being lowered.
    /// Set during entry point setup, cleared at end. Used by OutputPtr lowering.
    current_entry_outputs: Vec<spirv::Word>,

    /// buffer_id → (buffer_var, elem_spirv_type). The buffer_id is recovered
    /// from a view's type via `array_view_buffer` → `get_or_assign_buffer_id`.
    buffer_vars: Vec<(spirv::Word, spirv::Word)>,
    /// (set, binding) → buffer_id, for deduplication in get_or_assign_buffer_id.
    buffer_id_map: LookupMap<StorageBufferUse, u32>,
    /// Workgroup-shared arrays: id → (workgroup `OpVariable`, element type).
    /// Created in `lower_ssa_entry_point` by pre-scanning the body for
    /// `StorageView(Workgroup{id, count})` ops, so the var exists (and is in
    /// the entry interface, required by SPIR-V ≥1.4) before `ViewIndex` chains
    /// into it.
    workgroup_vars: LookupMap<u32, (spirv::Word, spirv::Word)>,
}

impl Constructor {
    fn new() -> Self {
        let builder = builder::SpirvBuilder::new();
        let void_type = *builder.void_type();
        let bool_type = *builder.bool_type();
        let i32_type = *builder.i32_type();
        let u32_type = *builder.u32_type();
        let f32_type = *builder.f32_type();
        let glsl_ext_inst_id = builder.glsl_ext_inst_id();

        Constructor {
            builder,
            void_type,
            bool_type,
            i32_type,
            u32_type,
            f32_type,
            env: LookupMap::new(),
            glsl_ext_inst_id,
            polytype_cache: LookupMap::new(),
            interface_block_cache: LookupMap::new(),
            entry_point_interfaces: LookupMap::new(),
            storage_buffers: LookupMap::new(),
            current_storage_accesses: LookupMap::new(),
            current_functions: LookupMap::new(),
            emitted_functions: LookupMap::new(),
            entry_functions: LookupMap::new(),
            globals: LookupMap::new(),
            storage_images: LookupMap::new(),
            global_invocation_id: None,
            local_invocation_id: None,
            num_workgroups: None,
            push_constant_var: None,
            linked_functions: LookupMap::new(),
            linked_functions_by_linkage: LookupMap::new(),
            int_pow_functions: LookupMap::new(),
            current_entry_outputs: Vec::new(),
            buffer_vars: Vec::new(),
            workgroup_vars: LookupMap::new(),
            buffer_id_map: LookupMap::new(),
        }
    }

    fn select_storage_accesses(&mut self, accesses: &LookupMap<BindingRef, crate::ResourceAccess>) {
        self.current_storage_accesses.clone_from(accesses);
    }

    fn storage_use(&self, binding: BindingRef) -> StorageBufferUse {
        StorageBufferUse {
            binding,
            writable: self.current_storage_accesses.get(&binding).is_none_or(|access| access.writes()),
        }
    }

    fn storage_buffer(&self, binding: BindingRef) -> Option<(spirv::Word, spirv::Word, spirv::Word)> {
        self.storage_buffers.get(&self.storage_use(binding)).copied()
    }

    fn select_functions(&mut self, functions: &LookupMap<crate::FunctionId, FunctionEmissionId>) {
        let externs = self.linked_functions.clone();
        self.current_functions = externs;
        for (&id, &emission) in functions {
            // A storage-variant plan includes every callable, including
            // linked externs. Those already have their structural SPIR-V id
            // in `linked_functions`; only locally emitted functions belong in
            // `emitted_functions`.
            if self.linked_functions.contains_key(&id) {
                continue;
            }
            let function =
                self.emitted_functions.get(&emission).copied().unwrap_or_else(|| {
                    panic!("local function {id:?} has no reserved emission {emission:?}")
                });
            self.current_functions.insert(id, function);
        }
    }
    /// Reserve a SPIR-V id for a function whose body is emitted later.
    fn reserve_function(&mut self) -> spirv::Word {
        *self.builder.reserve_function()
    }

    /// Forward-declare a linked (extern) function with Import linkage.
    /// Creates a function stub with no body that will be resolved by spirv-link.
    fn forward_declare_linked_function(
        &mut self,
        linkage_name: &str,
        param_types: &[spirv::Word],
        return_type: spirv::Word,
    ) -> spirv::Word {
        let param_types_typed: Vec<builder::TypeId> =
            param_types.iter().map(|&w| builder::TypeId::new(w)).collect();
        *self.builder.forward_declare_linked_function(
            linkage_name,
            &param_types_typed,
            builder::TypeId::new(return_type),
        )
    }

    /// Begin a new function. Returns `(func_id, param_ids, first_code_block)`.
    fn begin_function(
        &mut self,
        reserved: Option<spirv::Word>,
        param_types: &[spirv::Word],
        return_type: spirv::Word,
    ) -> Result<(spirv::Word, Vec<spirv::Word>, spirv::Word)> {
        let param_types_typed: Vec<builder::TypeId> =
            param_types.iter().map(|&w| builder::TypeId::new(w)).collect();
        let (func_id, param_ids, code_block) = self.builder.begin_function(
            reserved.map(builder::FuncId::new),
            &param_types_typed,
            builder::TypeId::new(return_type),
        )?;
        Ok((*func_id, param_ids, *code_block))
    }

    /// End the current function and clear per-entry-point name bindings.
    fn end_function(&mut self) -> Result<()> {
        self.builder.end_function()?;
        self.env.clear();
        Ok(())
    }

    /// Declare a variable in the function's variables block
    fn declare_variable(&mut self, _name: &str, value_type: spirv::Word) -> Result<spirv::Word> {
        Ok(*self.builder.declare_variable(builder::TypeId::new(value_type))?)
    }

    /// Load the storage-image global selected by the operand's pinned region.
    /// This supports multiple images and keeps image operations on module-scope
    /// variables even inside captured loop/SOAC bodies.
    fn load_storage_image(&mut self, binding: BindingRef) -> Result<spirv::Word> {
        let &(var, img_type) = self.storage_images.get(&binding).ok_or_else(|| {
            err_spirv!(
                "storage image binding(set={}, binding={}) has no declared global",
                binding.set,
                binding.binding
            )
        })?;
        Ok(self.builder.load(img_type, None, var, None, [])?)
    }

    /// Get or create an i32 constant
    // LowerCtx exposes raw SPIR-V words at this boundary; these delegators use
    // the typed builder internally and unwrap its `ConstId` result.
    fn const_i32(&mut self, value: i32) -> spirv::Word {
        *self.builder.const_i32(value)
    }
    fn const_u32(&mut self, value: u32) -> spirv::Word {
        *self.builder.const_u32(value)
    }
    fn const_f32(&mut self, value: f32) -> spirv::Word {
        *self.builder.const_f32(value)
    }
    fn const_bool(&mut self, value: bool) -> spirv::Word {
        *self.builder.const_bool(value)
    }

    /// Get the literal i32 value from a constant id created via the
    /// builder's `const_i32`. Thin delegator — the builder owns the
    /// reverse-lookup table.
    fn get_const_i32_value(&self, id: spirv::Word) -> Option<i32> {
        self.builder.get_const_i32_value(builder::ConstId::new(id))
    }

    /// Get the literal u32 value from a constant id created via the
    /// builder's `const_u32`. Thin delegator.
    fn get_const_u32_value(&self, id: spirv::Word) -> Option<u32> {
        self.builder.get_const_u32_value(builder::ConstId::new(id))
    }

    /// Get the element type of an array type. Thin delegator over
    /// `SpirvBuilder::array_element_type`, surfacing a structured
    /// error for missing entries (callers couldn't continue without
    /// the elem id).
    fn get_array_element_type(&self, array_type: spirv::Word) -> Result<spirv::Word> {
        self.builder
            .array_element_type(builder::TypeId::new(array_type))
            .map(|t| *t)
            .ok_or_else(|| crate::err_spirv!("Array element type not found for type ID: {}", array_type))
    }

    /// Thin delegator over `SpirvBuilder::composite_or_construct`.
    fn composite_or_constant(
        &mut self,
        result_type: spirv::Word,
        elem_ids: Vec<spirv::Word>,
    ) -> Result<spirv::Word> {
        Ok(self.builder.composite_or_construct(builder::TypeId::new(result_type), elem_ids)?)
    }
}

// =============================================================================
// SSA to SPIR-V Lowering
// =============================================================================

/// Lower a constant definition body to module-level SPIR-V constants.
///
/// Walks instructions in order, emitting OpConstant/OpConstantComposite.
/// Lower an SSA function body to SPIR-V.
///
/// This creates a SPIR-V function from the SSA representation:
/// - SSA blocks become SPIR-V blocks
/// - Block parameters become OpPhi nodes
/// - Terminators become branch instructions
/// Map a `StorageImageFormat` from the descriptor to the matching
/// SPIR-V `ImageFormat` literal used in `OpTypeImage`. Kept in lock-step
/// with the wgpu side: every format we emit here must also be allocated
/// by the host with the matching `wgpu::TextureFormat`.
fn storage_image_format_to_spirv(f: crate::pipeline_descriptor::StorageImageFormat) -> spirv::ImageFormat {
    use crate::pipeline_descriptor::StorageImageFormat as F;
    match f {
        F::Rgba8Unorm => spirv::ImageFormat::Rgba8,
        F::Rgba16Float => spirv::ImageFormat::Rgba16f,
        F::Rgba32Float => spirv::ImageFormat::Rgba32f,
        F::R32Float => spirv::ImageFormat::R32f,
    }
}

// =============================================================================
// SSA Program Lowering (new direct path)
// =============================================================================

/// Lower an SSA program directly to SPIR-V.
///
/// This is the new direct path: TLC → SSA → SPIR-V, bypassing MIR.
pub fn lower_ssa_program(program: &crate::ssa::stage::SpirvReady) -> Result<Vec<u32>> {
    // Use a thread with larger stack size for complex shaders
    const STACK_SIZE: usize = 16 * 1024 * 1024; // 16MB

    let program_clone = program.clone();

    let handle = std::thread::Builder::new()
        .stack_size(STACK_SIZE)
        .spawn(move || lower_ssa_program_impl(&program_clone))
        .expect("Failed to spawn lowering thread");

    match handle.join() {
        Ok(result) => result,
        Err(payload) => {
            // Preserve the worker thread's panic message so callers
            // (`#[should_panic(expected = ...)]` tests, CLI users) see
            // the real diagnostic rather than `Any { .. }`.
            let msg = payload
                .downcast_ref::<String>()
                .map(|s| s.as_str())
                .or_else(|| payload.downcast_ref::<&'static str>().copied())
                .unwrap_or("<non-string panic payload>");
            panic!("Lowering thread panicked: {}", msg);
        }
    }
}

fn lower_ssa_program_impl(program: &crate::ssa::stage::SpirvReady) -> Result<Vec<u32>> {
    let mut constructor = Constructor::new();
    let function_variants = StorageFunctionVariants::new(program);

    // Collect entry point info for later
    let mut entry_info: Vec<(
        crate::EntryId,
        String,
        spirv::ExecutionModel,
        Option<(u32, u32, u32)>,
    )> = Vec::new();

    // Forward-declare all functions first (so they can call each other in any order)
    for emission in function_variants.emissions() {
        let func = program
            .functions
            .iter()
            .find(|func| func.id == emission.function)
            .expect("function emission refers to missing FunctionId");
        if func.linkage_name.is_some() {
            continue;
        }
        let id = constructor.reserve_function();
        constructor.emitted_functions.insert(emission.id, id);
    }

    // Forward-declare program-level constants too. Each is a zero-arg
    // function whose body returns the folded literal; consumer bodies
    // reach them via `InstKind::Global(name)`, which looks up
    // `Constructor.functions`. Without this step the SPIR-V emit
    // fails with "Unknown global: <name>" whenever a non-constant
    // initializer (function call etc.) references a hoisted pure
    // constant.
    for constant in &program.constants {
        let id = constructor.reserve_function();
        constructor.globals.insert(constant.id, id);
    }

    // Forward-declare extern (linked) functions with Import linkage
    for func in &program.functions {
        if let Some(linkage_name) = &func.linkage_name {
            let body = &func.body;
            let param_types: Vec<spirv::Word> =
                body.params().map(|(_, ty, _)| constructor.polytype_to_spirv(ty)).collect();
            let return_type = constructor.polytype_to_spirv(&body.return_ty);
            let func_id =
                constructor.forward_declare_linked_function(linkage_name, &param_types, return_type);
            constructor.linked_functions.insert(func.id, func_id);
            constructor.linked_functions_by_linkage.insert(linkage_name.clone(), func_id);
            constructor.current_functions.insert(func.id, func_id);
        }
    }

    // Emit compiler-generated helpers. Integer `**` lowers to an
    // OpFunctionCall against one of these (see `spirv::pow`); emitting
    // both signedness variants unconditionally is ~60 instructions of
    // module overhead and drivers DCE them when unused.
    pow::emit_int_pow_helpers(&mut constructor)?;

    // Pre-create storage buffers for all entry point bindings so that
    // buffer-specialized functions (which reference set/binding directly) can
    // resolve them during lowering, even though they're lowered before entry points.
    for entry in &program.entry_points {
        let accesses = entry.shader_storage_accesses();
        for input in &entry.inputs {
            if let Some(br) = input.storage_binding() {
                constructor.create_storage_buffer(&input.ty, br.set, br.binding, true);
                if !accesses[&br].writes() {
                    constructor.create_storage_buffer(&input.ty, br.set, br.binding, false);
                }
            }
        }
        for output in &entry.outputs {
            if let Some(br) = output.storage_binding() {
                constructor.create_storage_buffer(&output.ty, br.set, br.binding, true);
            }
        }
    }

    // Also pre-create buffers from each entry's `storage_bindings` — the
    // typed list of compiler-introduced bindings (e.g. parallelize's
    // partials/result intermediates) that aren't user-visible outputs.
    for entry in &program.entry_points {
        let accesses = entry.shader_storage_accesses();
        for sb in &entry.storage_bindings {
            constructor.create_storage_buffer(&sb.elem_ty, sb.binding.set, sb.binding.binding, true);
            if !accesses[&sb.binding].writes() {
                constructor.create_storage_buffer(&sb.elem_ty, sb.binding.set, sb.binding.binding, false);
            }
        }
    }

    // Pre-create storage-image globals for all entry bindings so that image
    // ops inside SOAC-body functions (lowered before entry points) reference
    // the module-scope variable rather than an OpFunctionParameter.
    //
    // A binding shared across entries with mixed access (read in one, written in
    // another — e.g. a compute pass writes an image a later pass reads)
    // collapses to one global, so its `NonReadable`/`NonWritable` decoration
    // must be the *union* of every view's access. Compute the union first (map
    // values, so iteration order doesn't matter), then create in deterministic
    // entry/input order (`create_storage_image` is idempotent).
    let mut image_access: LookupMap<BindingRef, crate::interface::StorageAccess> = LookupMap::new();
    for entry in &program.entry_points {
        for input in &entry.inputs {
            if let Some((br, _format, access, _size)) = input.storage_image_binding() {
                image_access.entry(br).and_modify(|acc| *acc = acc.merge(access)).or_insert(access);
            }
        }
    }
    for entry in &program.entry_points {
        for input in &entry.inputs {
            if let Some((br, format, _access, _size)) = input.storage_image_binding() {
                constructor.create_storage_image(br, format, image_access[&br]);
            }
        }
    }

    // Now lower all function bodies.
    for emission in function_variants.emissions() {
        let func = program
            .functions
            .iter()
            .find(|func| func.id == emission.function)
            .expect("function emission refers to missing FunctionId");
        if func.linkage_name.is_some() {
            // Extern functions have no local body; the Import-linkage
            // declaration emitted above is the full handling, and
            // `InstKind::Extern` resolves them at call sites via
            // `constructor.linked_functions`.
            continue;
        }

        constructor
            .select_storage_accesses(&function_variants.accesses_for(program, emission.entry_context));
        constructor.select_functions(function_variants.emissions_for_context(emission.entry_context));
        lower_ssa_function(&mut constructor, func, emission.id)?;
    }

    // Lower program-level constants as zero-arg functions. Their
    // forward-declared IDs are already in `Constructor.functions`
    // (the loop above ran before any body lowering); now emit the
    // body so calls to `Global(name)` from other functions resolve.
    constructor.select_storage_accesses(&function_variants.accesses_for(program, None));
    constructor.select_functions(function_variants.emissions_for_context(None));
    for constant in &program.constants {
        let return_type = constructor.polytype_to_spirv(&constant.body.return_ty);
        let (_, param_ids, first_code_block) =
            constructor.begin_function(Some(constructor.globals[&constant.id]), &[], return_type)?;
        lower::LowerCtx::new(
            &mut constructor,
            &constant.body,
            false,
            Span::new(0, 0, 0, 0),
            param_ids,
            first_code_block,
        )
        .lower()
        .map_err(|e| err_spirv!("in constant '{}': {}", constant.name, e))?;
        constructor.end_function()?;
    }

    // Lower each entry under its own storage-access map. When entries use the
    // same slot with different access, they reference distinct module globals
    // and storage-dependent helper variants selected above.
    for entry in &program.entry_points {
        let (spirv_model, local_size) = match &entry.execution_model {
            ExecutionModel::Vertex => (spirv::ExecutionModel::Vertex, None),
            ExecutionModel::Fragment => (spirv::ExecutionModel::Fragment, None),
            ExecutionModel::Compute { local_size } => (spirv::ExecutionModel::GLCompute, Some(*local_size)),
        };

        entry_info.push((entry.id, entry.name.clone(), spirv_model, local_size));
        constructor.select_storage_accesses(&entry.shader_storage_accesses());
        constructor.select_functions(function_variants.emissions_for_entry(entry.id));
        entry::lower_ssa_entry_point(&mut constructor, entry)?;
    }

    // Hoisted constant `Private` globals must be listed in each entry's
    // interface (SPIR-V ≥1.4 lists every referenced global). They are
    // shared module-wide and may be reached through helper functions
    // called by several entries, so list them in every entry — a superset
    // (one an entry doesn't reach) is valid and avoids call-graph tracking.
    // Collected once here, after all bodies are lowered.
    let private_global_ids: Vec<spirv::Word> = constructor.builder.private_globals().map(|v| *v).collect();

    // Emit entry point declarations
    for (entry_id, name, model, local_size) in &entry_info {
        if let Some(&func_id) = constructor.entry_functions.get(entry_id) {
            let mut interfaces =
                constructor.entry_point_interfaces.get(entry_id).cloned().unwrap_or_default();
            for &var_id in &private_global_ids {
                if !interfaces.contains(&var_id) {
                    interfaces.push(var_id);
                }
            }

            // Add storage buffer variables that this entry point declares
            // (via its inputs/outputs). Don't add ALL storage vars — other
            // entry points may have buffers this one doesn't reference.
            if let Some(entry) = program.entry_points.iter().find(|e| e.id == *entry_id) {
                constructor.select_storage_accesses(&entry.shader_storage_accesses());
                for input in &entry.inputs {
                    if let Some(br) = input.storage_binding() {
                        if let Some((var_id, _, _)) = constructor.storage_buffer(br) {
                            if !interfaces.contains(&var_id) {
                                interfaces.push(var_id);
                            }
                        }
                    }
                }
                for output in &entry.outputs {
                    if let Some(br) = output.storage_binding() {
                        if let Some((var_id, _, _)) = constructor.storage_buffer(br) {
                            if !interfaces.contains(&var_id) {
                                interfaces.push(var_id);
                            }
                        }
                    }
                }
                // Also include compiler-introduced storage bindings from
                // the entry's typed `storage_bindings` list (e.g.
                // parallelize's partials/result intermediates).
                for sb in &entry.storage_bindings {
                    if let Some((var_id, _, _)) = constructor.storage_buffer(sb.binding) {
                        if !interfaces.contains(&var_id) {
                            interfaces.push(var_id);
                        }
                    }
                }
            }
            constructor.builder.entry_point(*model, func_id, name, interfaces);

            // Add execution modes
            match model {
                spirv::ExecutionModel::Fragment => {
                    constructor.builder.execution_mode(func_id, spirv::ExecutionMode::OriginUpperLeft, []);
                }
                spirv::ExecutionModel::GLCompute => {
                    if let Some((x, y, z)) = local_size {
                        constructor.builder.execution_mode(
                            func_id,
                            spirv::ExecutionMode::LocalSize,
                            [*x, *y, *z],
                        );
                    }
                }
                _ => {}
            }
        }
    }

    Ok(constructor.builder.into_module().assemble())
}

/// Lower an SSA function to SPIR-V.
fn lower_ssa_function(
    constructor: &mut Constructor,
    func: &Function,
    emission: FunctionEmissionId,
) -> Result<()> {
    let body = &func.body;

    // Convert function parameter types to their SPIR-V representations.
    let param_types: Vec<spirv::Word> =
        body.params().map(|(_, ty, _)| constructor.polytype_to_spirv(ty)).collect();

    let return_type = constructor.polytype_to_spirv(&body.return_ty);

    let (_, param_ids, first_code_block) = constructor.begin_function(
        Some(constructor.emitted_functions[&emission]),
        &param_types,
        return_type,
    )?;
    lower::LowerCtx::new(constructor, body, false, func.span, param_ids, first_code_block)
        .lower()
        .map_err(|e| err_spirv!("in function '{}': {}", func.name, e))?;
    constructor.end_function()?;

    Ok(())
}
