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
use crate::ssa::storage_function_variants::StorageFunctionVariants;
use crate::ssa::types::{
    BlockId, ConstantValue, ControlHeader, EntryPoint, ExecutionModel, FuncBody, Function, InstKind,
    Program, Terminator, ValueId, ValueRef, WynInstNode,
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

    // Per-entry-point name → loaded-value lookup. Populated by
    // entry-point I/O setup (push constants / uniforms / locations
    // load through here so the SSA body can fetch the value by the
    // input's declared name).
    env: LookupMap<String, spirv::Word>,

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
    entry_point_interfaces: LookupMap<String, Vec<spirv::Word>>,

    /// Access-qualified storage-buffer globals. The same descriptor slot can
    /// be writable in a compute prepass and read-only in a graphics entry.
    storage_buffers: LookupMap<StorageBufferUse, (spirv::Word, spirv::Word, spirv::Word)>,
    current_storage_accesses: LookupMap<BindingRef, crate::ResourceAccess>,
    current_function_names: LookupMap<String, String>,

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

    /// Linked SPIR-V functions: linkage_name -> function_id
    linked_functions: LookupMap<String, spirv::Word>,

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
            current_function_names: LookupMap::new(),
            storage_images: LookupMap::new(),
            global_invocation_id: None,
            local_invocation_id: None,
            num_workgroups: None,
            push_constant_var: None,
            linked_functions: LookupMap::new(),
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

    fn select_function_names(&mut self, names: &LookupMap<String, String>) {
        self.current_function_names.clone_from(names);
    }

    fn emitted_function_name<'a>(&'a self, source_name: &'a str) -> &'a str {
        self.current_function_names.get(source_name).map(String::as_str).unwrap_or(source_name)
    }

    /// Forward-declare a function (reserve ID without emitting body).
    /// This allows functions to call each other regardless of order.
    fn forward_declare_function(
        &mut self,
        name: &str,
        _param_types: &[spirv::Word],
        _return_type: spirv::Word,
    ) -> spirv::Word {
        *self.builder.forward_declare_function(name)
    }

    /// Forward-declare a linked (extern) function with Import linkage.
    /// Creates a function stub with no body that will be resolved by spirv-link.
    fn forward_declare_linked_function(
        &mut self,
        name: &str,
        linkage_name: &str,
        param_types: &[spirv::Word],
        return_type: spirv::Word,
    ) -> spirv::Word {
        let param_types_typed: Vec<builder::TypeId> =
            param_types.iter().map(|&w| builder::TypeId::new(w)).collect();
        *self.builder.forward_declare_linked_function(
            name,
            linkage_name,
            &param_types_typed,
            builder::TypeId::new(return_type),
        )
    }

    /// Begin a new function. Returns `(func_id, param_ids, first_code_block)`.
    fn begin_function(
        &mut self,
        name: &str,
        _param_names: &[&str],
        param_types: &[spirv::Word],
        return_type: spirv::Word,
    ) -> Result<(spirv::Word, Vec<spirv::Word>, spirv::Word)> {
        let param_types_typed: Vec<builder::TypeId> =
            param_types.iter().map(|&w| builder::TypeId::new(w)).collect();
        let (func_id, param_ids, code_block) =
            self.builder.begin_function(name, &param_types_typed, builder::TypeId::new(return_type))?;
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
    // Thin delegators over `SpirvBuilder`'s typed constant
    // emitters. The wrapper-returned `ConstId` is `*`-deref'd back to
    // `spirv::Word` here so existing untyped call sites in
    // `LowerCtx` and `pow.rs` keep working. As call sites migrate to
    // typed `Id<K>`s the delegators will be removed.
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
/// Combine two views' accesses to the same storage image. Reads and writes
/// accumulate independently: a binding read by one view and written by another
/// is genuinely `ReadWrite`, so the shared global carries neither `NonReadable`
/// nor `NonWritable`.
fn union_storage_access(
    a: crate::interface::StorageAccess,
    b: crate::interface::StorageAccess,
) -> crate::interface::StorageAccess {
    use crate::interface::StorageAccess::{ReadOnly, ReadWrite, WriteOnly};
    let reads = |x| matches!(x, ReadOnly | ReadWrite);
    let writes = |x| matches!(x, WriteOnly | ReadWrite);
    match (reads(a) || reads(b), writes(a) || writes(b)) {
        (true, true) => ReadWrite,
        (false, true) => WriteOnly,
        // A view always reads or writes, so `(false, false)` is unreachable;
        // treat read-only as the conservative default for that impossible case.
        _ => ReadOnly,
    }
}

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
pub fn lower_ssa_program(program: &Program) -> Result<Vec<u32>> {
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

fn lower_ssa_program_impl(program: &Program) -> Result<Vec<u32>> {
    let mut constructor = Constructor::new();
    let function_variants = StorageFunctionVariants::new(program);

    // Collect entry point info for later
    let mut entry_info: Vec<(String, spirv::ExecutionModel, Option<(u32, u32, u32)>)> = Vec::new();

    // Forward-declare all functions first (so they can call each other in any order)
    for emission in function_variants.emissions() {
        let func = &program.functions[emission.function];
        if func.linkage_name.is_some() {
            continue;
        }
        let body = &func.body;
        let param_types: Vec<spirv::Word> =
            body.params.iter().map(|(_, ty, _)| constructor.polytype_to_spirv(ty)).collect();
        let return_type = constructor.polytype_to_spirv(&body.return_ty);
        constructor.forward_declare_function(&emission.name, &param_types, return_type);
    }

    // Forward-declare program-level constants too. Each is a zero-arg
    // function whose body returns the folded literal; consumer bodies
    // reach them via `InstKind::Global(name)`, which looks up
    // `Constructor.functions`. Without this step the SPIR-V emit
    // fails with "Unknown global: <name>" whenever a non-constant
    // initializer (function call etc.) references a hoisted pure
    // constant.
    for constant in &program.constants {
        let return_type = constructor.polytype_to_spirv(&constant.body.return_ty);
        constructor.forward_declare_function(&constant.name, &[], return_type);
    }

    // Forward-declare extern (linked) functions with Import linkage
    for func in &program.functions {
        if let Some(linkage_name) = &func.linkage_name {
            let body = &func.body;
            let param_types: Vec<spirv::Word> =
                body.params.iter().map(|(_, ty, _)| constructor.polytype_to_spirv(ty)).collect();
            let return_type = constructor.polytype_to_spirv(&body.return_ty);
            let func_id = constructor.forward_declare_linked_function(
                &func.name,
                linkage_name,
                &param_types,
                return_type,
            );
            constructor.linked_functions.insert(func.name.clone(), func_id);
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
                image_access
                    .entry(br)
                    .and_modify(|acc| *acc = union_storage_access(*acc, access))
                    .or_insert(access);
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
        let func = &program.functions[emission.function];
        if func.linkage_name.is_some() {
            // Extern functions have no local body; the Import-linkage
            // declaration emitted above is the full handling, and
            // `InstKind::Extern` resolves them at call sites via
            // `constructor.linked_functions`.
            continue;
        }

        constructor
            .select_storage_accesses(&function_variants.accesses_for(program, emission.entry_context));
        let names = emission
            .entry_context
            .map(|entry| function_variants.names_for_entry(entry))
            .cloned()
            .unwrap_or_default();
        constructor.select_function_names(&names);
        lower_ssa_function(&mut constructor, func, &emission.name)?;
    }

    // Lower program-level constants as zero-arg functions. Their
    // forward-declared IDs are already in `Constructor.functions`
    // (the loop above ran before any body lowering); now emit the
    // body so calls to `Global(name)` from other functions resolve.
    constructor.select_storage_accesses(&function_variants.accesses_for(program, None));
    constructor.select_function_names(&LookupMap::new());
    for constant in &program.constants {
        let return_type = constructor.polytype_to_spirv(&constant.body.return_ty);
        let (_, param_ids, first_code_block) =
            constructor.begin_function(&constant.name, &[], &[], return_type)?;
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
    for (entry_index, entry) in program.entry_points.iter().enumerate() {
        let (spirv_model, local_size) = match &entry.execution_model {
            ExecutionModel::Vertex => (spirv::ExecutionModel::Vertex, None),
            ExecutionModel::Fragment => (spirv::ExecutionModel::Fragment, None),
            ExecutionModel::Compute { local_size } => (spirv::ExecutionModel::GLCompute, Some(*local_size)),
        };

        entry_info.push((entry.name.clone(), spirv_model, local_size));
        constructor.select_storage_accesses(&entry.shader_storage_accesses());
        constructor.select_function_names(function_variants.names_for_entry(entry_index));
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
    for (name, model, local_size) in &entry_info {
        if let Some(func_id) = constructor.builder.get_function(name) {
            let func_id = *func_id;
            let mut interfaces = constructor.entry_point_interfaces.get(name).cloned().unwrap_or_default();
            for &var_id in &private_global_ids {
                if !interfaces.contains(&var_id) {
                    interfaces.push(var_id);
                }
            }

            // Add storage buffer variables that this entry point declares
            // (via its inputs/outputs). Don't add ALL storage vars — other
            // entry points may have buffers this one doesn't reference.
            if let Some(entry) = program.entry_points.iter().find(|e| e.name == *name) {
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
fn lower_ssa_function(constructor: &mut Constructor, func: &Function, emitted_name: &str) -> Result<()> {
    let body = &func.body;

    // Extract parameter types and names, converting types to SPIR-V
    let param_names: Vec<&str> = body.params.iter().map(|(_, _, name)| name.as_str()).collect();
    let param_types: Vec<spirv::Word> =
        body.params.iter().map(|(_, ty, _)| constructor.polytype_to_spirv(ty)).collect();

    let return_type = constructor.polytype_to_spirv(&body.return_ty);

    let (_, param_ids, first_code_block) =
        constructor.begin_function(emitted_name, &param_names, &param_types, return_type)?;
    lower::LowerCtx::new(constructor, body, false, func.span, param_ids, first_code_block)
        .lower()
        .map_err(|e| err_spirv!("in function '{}': {}", func.name, e))?;
    constructor.end_function()?;

    Ok(())
}
