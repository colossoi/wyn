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
use std::collections::{HashMap, HashSet};

use crate::ast::Span;
use crate::ast::TypeName;
use crate::builtins::lowering::{BuiltinLowering, PrimOp};
use crate::error::Result;
use crate::ssa::layout::{buffer_array_strides, std430_alignment, type_byte_size};
use crate::ssa::types::{
    BlockId, ConstantValue, ControlHeader, FuncBody, InstKind, Terminator, ValueId, ValueRef, WynInstNode,
};
use crate::ssa::types::{EntryPoint, ExecutionModel, Function, IoDecoration, Program};
use crate::types;
use crate::types::TypeExt;
use crate::BindingRef;
use crate::{bail_spirv, bail_spirv_at, err_spirv, err_spirv_at};
use polytype::Type as PolyType;
use wspirv::binary::Assemble;
use wspirv::dr::{InsertPoint, Operand};
use wspirv::spirv::{self, Capability, StorageClass};

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

    // Current function state
    current_block: Option<spirv::Word>,
    variables_block: Option<spirv::Word>, // Block for OpVariable declarations
    first_code_block: Option<spirv::Word>, // First block with actual code

    // Environment: name -> value ID
    env: HashMap<String, spirv::Word>,
    // Function parameter SPIR-V IDs in declaration order (positional mapping)
    param_ids: Vec<spirv::Word>,

    // Function map: name -> function ID
    functions: HashMap<String, spirv::Word>,

    // GLSL extended instruction set
    glsl_ext_inst_id: spirv::Word,

    // Top-level polytype → SPIR-V memoization (subsumes type + constant dedup for wyn types)
    polytype_cache: HashMap<PolyType<TypeName>, spirv::Word>,

    // Interface-block + nested-array lookups stay here for now —
    // they're entangled with the compiler's `PolyType` walks
    // (interface members need `apply_buffer_array_strides`, which is
    // PolyType-driven). The simpler block wrappers and structural
    // type caches live on `SpirvBuilder`.
    interface_block_cache: HashMap<InterfaceBlockKey, spirv::Word>,

    // Entry point interface tracking
    entry_point_interfaces: HashMap<String, Vec<spirv::Word>>,

    /// Storage buffers for compute shaders: (set, binding) -> (buffer_var, elem_type_id, buffer_ptr_type)
    storage_buffers: HashMap<BindingRef, (spirv::Word, spirv::Word, spirv::Word)>,

    /// GlobalInvocationId variable for compute shaders (set during entry point setup)
    global_invocation_id: Option<spirv::Word>,

    /// LocalInvocationId variable for compute shaders (set during entry point setup)
    local_invocation_id: Option<spirv::Word>,

    /// NumWorkgroups variable for compute shaders (set during entry point setup)
    num_workgroups: Option<spirv::Word>,

    /// Shared push constant variable (at most one per SPIR-V module)
    push_constant_var: Option<spirv::Word>,

    /// Linked SPIR-V functions: linkage_name -> function_id
    linked_functions: HashMap<String, spirv::Word>,

    /// Compiler-generated integer-pow helpers (see `spirv::pow`), keyed
    /// by `signed`. Emitted once per module after function forward
    /// declarations; `PrimOp::IntPow` lowers to `OpFunctionCall` against
    /// the cached id.
    int_pow_functions: HashMap<bool, spirv::Word>,

    /// Output variables for the current entry point being lowered.
    /// Set during entry point setup, cleared at end. Used by OutputPtr lowering.
    current_entry_outputs: Vec<spirv::Word>,

    /// buffer_id → (buffer_var, elem_spirv_type). The buffer_id is recovered
    /// from a view's type via `array_view_region` → `get_or_assign_buffer_id`.
    buffer_vars: Vec<(spirv::Word, spirv::Word)>,
    /// (set, binding) → buffer_id, for deduplication in get_or_assign_buffer_id.
    buffer_id_map: HashMap<BindingRef, u32>,
    /// Workgroup-shared arrays: id → (workgroup `OpVariable`, element type).
    /// Created in `lower_ssa_entry_point` by pre-scanning the body for
    /// `StorageView(Workgroup{id, count})` ops, so the var exists (and is in
    /// the entry interface, required by SPIR-V ≥1.4) before `ViewIndex` chains
    /// into it.
    workgroup_vars: HashMap<u32, (spirv::Word, spirv::Word)>,

    /// Used by `polytype_to_spirv` when emitting `StorageTexture` for
    /// a function signature; entry-var emission still overrides per-
    /// binding. Set once in `lower_ssa_program_impl`.
    storage_image_default_format: Option<crate::pipeline_descriptor::StorageImageFormat>,
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
            current_block: None,
            variables_block: None,
            first_code_block: None,
            env: HashMap::new(),
            param_ids: Vec::new(),
            functions: HashMap::new(),
            glsl_ext_inst_id,
            polytype_cache: HashMap::new(),
            interface_block_cache: HashMap::new(),
            entry_point_interfaces: HashMap::new(),
            storage_buffers: HashMap::new(),
            global_invocation_id: None,
            local_invocation_id: None,
            num_workgroups: None,
            push_constant_var: None,
            linked_functions: HashMap::new(),
            int_pow_functions: HashMap::new(),
            current_entry_outputs: Vec::new(),
            buffer_vars: Vec::new(),
            workgroup_vars: HashMap::new(),
            buffer_id_map: HashMap::new(),
            storage_image_default_format: None,
        }
    }

    /// Forward-declare a function (reserve ID without emitting body).
    /// This allows functions to call each other regardless of order.
    fn forward_declare_function(
        &mut self,
        name: &str,
        _param_types: &[spirv::Word],
        _return_type: spirv::Word,
    ) -> spirv::Word {
        // Check if already declared
        if let Some(&id) = self.functions.get(name) {
            return id;
        }

        // Reserve an ID for the function
        let func_id = self.builder.id();

        // Store the mapping (we'll use this ID when we actually define the function)
        self.functions.insert(name.to_string(), func_id);

        func_id
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
        // Add Linkage capability
        self.builder.capability(Capability::Linkage);

        // Create function type
        let func_type = self.builder.type_function(return_type, param_types.to_vec());

        // Declare function with no body (Import linkage)
        let func_id = self
            .builder
            .begin_function(return_type, None, spirv::FunctionControl::NONE, func_type)
            .expect("BUG: failed to begin linked function");

        // Add parameters (required by SPIR-V)
        for &param_ty in param_types {
            self.builder.function_parameter(param_ty).expect("BUG: failed to add function parameter");
        }
        self.builder.end_function().expect("BUG: failed to end linked function");

        // Decorate with Import linkage
        self.builder.decorate(
            func_id,
            spirv::Decoration::LinkageAttributes,
            [
                Operand::LiteralString(linkage_name.to_string()),
                Operand::LinkageType(spirv::LinkageType::Import),
            ],
        );

        // Register in functions map so Call instructions can find it
        self.functions.insert(name.to_string(), func_id);

        func_id
    }

    /// Begin a new function
    fn begin_function(
        &mut self,
        name: &str,
        param_names: &[&str],
        param_types: &[spirv::Word],
        return_type: spirv::Word,
    ) -> Result<spirv::Word> {
        let func_type = self.builder.type_function(return_type, param_types.to_vec());

        // Check if this function was forward-declared
        let func_id = if let Some(&pre_id) = self.functions.get(name) {
            // Use the pre-allocated ID
            self.builder.begin_function(
                return_type,
                Some(pre_id),
                spirv::FunctionControl::NONE,
                func_type,
            )?
        } else {
            let id =
                self.builder.begin_function(return_type, None, spirv::FunctionControl::NONE, func_type)?;
            self.functions.insert(name.to_string(), id);
            id
        };

        // Create function parameters
        self.param_ids.clear();
        for (i, &param_name) in param_names.iter().enumerate() {
            let param_id = self.builder.function_parameter(param_types[i])?;
            self.env.insert(param_name.to_string(), param_id);
            self.param_ids.push(param_id);
        }

        // Create two blocks: one for variables, one for code
        let vars_block_id = self.builder.id();
        let code_block_id = self.builder.id();
        self.variables_block = Some(vars_block_id);
        self.first_code_block = Some(code_block_id);

        // Begin variables block (leave it open - no terminator yet)
        self.builder.begin_block(Some(vars_block_id))?;

        // Deselect current block so we can begin a new one
        self.builder.select_block(None)?;

        // Begin code block - this is where we'll emit code
        self.builder.begin_block(Some(code_block_id))?;
        self.current_block = Some(code_block_id);

        Ok(func_id)
    }

    /// End the current function
    fn end_function(&mut self) -> Result<()> {
        // Terminate the variables block with a branch to the code block
        if let (Some(vars_block), Some(code_block)) = (self.variables_block, self.first_code_block) {
            // Find the variables block index and select it
            let func = self.builder.module_ref().functions.last().expect("No function");
            let vars_idx = func
                .blocks
                .iter()
                .position(|b| b.label.as_ref().map(|l| l.result_id) == Some(Some(vars_block)));

            if let Some(idx) = vars_idx {
                self.builder.select_block(Some(idx))?;
                self.builder.branch(code_block)?;
            }
        }

        self.builder.end_function()?;

        // Clear function state
        self.current_block = None;
        self.variables_block = None;
        self.first_code_block = None;
        self.env.clear();

        Ok(())
    }

    /// Declare a variable in the function's variables block
    fn declare_variable(&mut self, _name: &str, value_type: spirv::Word) -> Result<spirv::Word> {
        let ptr_type = self.get_or_create_ptr_type(StorageClass::Function, value_type);

        // Save current block
        let current_idx = self.builder.selected_block();

        // Find and select the variables block
        let vars_block = self.variables_block.expect("declare_variable called outside function");
        let func = self.builder.module_ref().functions.last().expect("No function");
        let vars_idx = func
            .blocks
            .iter()
            .position(|b| b.label.as_ref().map(|l| l.result_id) == Some(Some(vars_block)))
            .expect("Variables block not found");

        self.builder.select_block(Some(vars_idx))?;

        // Emit the variable
        let var_id = self.builder.variable(ptr_type, None, StorageClass::Function, None);

        // Restore current block
        self.builder.select_block(current_idx)?;

        Ok(var_id)
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

    /// Begin a block (must be called before emitting instructions into it)
    fn begin_block(&mut self, block_id: spirv::Word) -> Result<()> {
        self.builder.begin_block(Some(block_id))?;
        self.current_block = Some(block_id);
        // Clear extract cache since values from previous blocks may not dominate this block
        Ok(())
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

fn lower_ssa_body(constructor: &mut Constructor, body: &FuncBody, func_span: Span) -> Result<spirv::Word> {
    let mut ctx = lower::LowerCtx::new(constructor, body, false, func_span);
    ctx.lower()
}

/// Lower an SSA function body for an entry point.
///
/// Entry points are void functions — OpReturnValue is invalid.
/// SSA for entry points should use OutputPtr+Store then ReturnUnit;
/// Return(value) will produce an error.
fn lower_ssa_body_for_entry(
    constructor: &mut Constructor,
    body: &FuncBody,
    func_span: Span,
) -> Result<spirv::Word> {
    let mut ctx = lower::LowerCtx::new(constructor, body, true, func_span);
    ctx.lower()
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

    // Pin the program-wide `storage_image` format used by
    // `polytype_to_spirv` for function-signature `StorageTexture`
    // params. v1 requires a uniform format across entries; mixed
    // formats need per-function monomorphization.
    let mut formats: Vec<crate::pipeline_descriptor::StorageImageFormat> = Vec::new();
    for entry in &program.entry_points {
        for input in &entry.inputs {
            if let Some((_, fmt, _, _)) = input.storage_image_binding {
                if !formats.contains(&fmt) {
                    formats.push(fmt);
                }
            }
        }
    }
    if formats.len() > 1 {
        return Err(crate::err_spirv!(
            "spirv backend: multiple storage_image formats in one program ({:?}) — \
             only uniform-format programs are supported until per-function \
             monomorphization over format lands",
            formats
        ));
    }
    constructor.storage_image_default_format = formats.into_iter().next();

    // Collect entry point info for later
    let mut entry_info: Vec<(String, spirv::ExecutionModel, Option<(u32, u32, u32)>)> = Vec::new();

    // Forward-declare all functions first (so they can call each other in any order)
    for func in &program.functions {
        if func.linkage_name.is_some() {
            continue;
        }
        let body = &func.body;
        let param_types: Vec<spirv::Word> =
            body.params.iter().map(|(_, ty, _)| constructor.polytype_to_spirv(ty)).collect();
        let return_type = constructor.polytype_to_spirv(&body.return_ty);
        constructor.forward_declare_function(&func.name, &param_types, return_type);
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
        for input in &entry.inputs {
            if let Some(br) = input.storage_binding {
                if !constructor.storage_buffers.contains_key(&br) {
                    constructor.create_storage_buffer(&input.ty, br.set, br.binding);
                }
            }
        }
        for output in &entry.outputs {
            if let Some(br) = output.storage_binding {
                if !constructor.storage_buffers.contains_key(&br) {
                    constructor.create_storage_buffer(&output.ty, br.set, br.binding);
                }
            }
        }
    }

    // Also pre-create buffers from each entry's `storage_bindings` — the
    // typed list of compiler-introduced bindings (e.g. parallelize's
    // partials/result intermediates) that aren't user-visible outputs.
    for entry in &program.entry_points {
        for sb in &entry.storage_bindings {
            if !constructor.storage_buffers.contains_key(&sb.binding) {
                constructor.create_storage_buffer(&sb.elem_ty, sb.binding.set, sb.binding.binding);
            }
        }
    }

    // Now lower all function bodies
    for func in &program.functions {
        if func.linkage_name.is_some() {
            // Extern functions have no local body; the Import-linkage
            // declaration emitted above is the full handling, and
            // `InstKind::Extern` resolves them at call sites via
            // `constructor.linked_functions`.
            continue;
        }

        lower_ssa_function(&mut constructor, func)?;
    }

    // Lower program-level constants as zero-arg functions. Their
    // forward-declared IDs are already in `Constructor.functions`
    // (the loop above ran before any body lowering); now emit the
    // body so calls to `Global(name)` from other functions resolve.
    for constant in &program.constants {
        let return_type = constructor.polytype_to_spirv(&constant.body.return_ty);
        constructor.begin_function(&constant.name, &[], &[], return_type)?;
        lower_ssa_body(&mut constructor, &constant.body, Span::new(0, 0, 0, 0))
            .map_err(|e| err_spirv!("in constant '{}': {}", constant.name, e))?;
        constructor.end_function()?;
    }

    // Lower all entry points
    // Collect all bindings that ANY entry point writes to: storage outputs,
    // plus `#[storage(access=write/readwrite)]` inputs written in place (e.g. a
    // `scatter` destination). These must not be marked `NonWritable` in any
    // entry — including entries that only read the binding — so the shader's
    // declared access stays consistent across the module and matches the
    // descriptor's promoted `ReadWrite` (mirrors `egir::publish` written_bindings).
    let written_bindings: HashSet<BindingRef> = program
        .entry_points
        .iter()
        .flat_map(|e| e.outputs.iter().filter_map(|o| o.storage_binding))
        .chain(program.entry_points.iter().flat_map(|e| {
            e.inputs.iter().filter_map(|i| {
                let br = i.storage_binding?;
                match i.storage_access {
                    Some(
                        crate::interface::StorageAccess::WriteOnly
                        | crate::interface::StorageAccess::ReadWrite,
                    ) => Some(br),
                    _ => None,
                }
            })
        }))
        .collect();

    for entry in &program.entry_points {
        let (spirv_model, local_size) = match &entry.execution_model {
            ExecutionModel::Vertex => (spirv::ExecutionModel::Vertex, None),
            ExecutionModel::Fragment => (spirv::ExecutionModel::Fragment, None),
            ExecutionModel::Compute { local_size } => (spirv::ExecutionModel::GLCompute, Some(*local_size)),
        };

        entry_info.push((entry.name.clone(), spirv_model, local_size));
        entry::lower_ssa_entry_point(&mut constructor, entry, &written_bindings)?;
    }

    // Emit entry point declarations
    for (name, model, local_size) in &entry_info {
        if let Some(&func_id) = constructor.functions.get(name) {
            let mut interfaces = constructor.entry_point_interfaces.get(name).cloned().unwrap_or_default();

            // Add storage buffer variables that this entry point declares
            // (via its inputs/outputs). Don't add ALL storage vars — other
            // entry points may have buffers this one doesn't reference.
            if let Some(entry) = program.entry_points.iter().find(|e| e.name == *name) {
                for input in &entry.inputs {
                    if let Some(br) = input.storage_binding {
                        if let Some(&(var_id, _, _)) = constructor.storage_buffers.get(&br) {
                            if !interfaces.contains(&var_id) {
                                interfaces.push(var_id);
                            }
                        }
                    }
                }
                for output in &entry.outputs {
                    if let Some(br) = output.storage_binding {
                        if let Some(&(var_id, _, _)) = constructor.storage_buffers.get(&br) {
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
                    if let Some(&(var_id, _, _)) = constructor.storage_buffers.get(&sb.binding) {
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
fn lower_ssa_function(constructor: &mut Constructor, func: &Function) -> Result<()> {
    let body = &func.body;

    // Extract parameter types and names, converting types to SPIR-V
    let param_names: Vec<&str> = body.params.iter().map(|(_, _, name)| name.as_str()).collect();
    let param_types: Vec<spirv::Word> =
        body.params.iter().map(|(_, ty, _)| constructor.polytype_to_spirv(ty)).collect();

    let return_type = constructor.polytype_to_spirv(&body.return_ty);

    constructor.begin_function(&func.name, &param_names, &param_types, return_type)?;
    lower_ssa_body(constructor, body, func.span)
        .map_err(|e| err_spirv!("in function '{}': {}", func.name, e))?;
    constructor.end_function()?;

    Ok(())
}
