//! SPIR-V Lowering
//!
//! This module converts MIR (from flattening) directly to SPIR-V.
//! It uses a Constructor wrapper that handles variable hoisting automatically.
//! Dependencies are lowered on-demand using ensure_lowered pattern.

use crate::ast::TypeName;
use crate::error::Result;
use crate::impl_source::ImplSource;
use crate::lowering_common::is_empty_closure_type;
// Note: detect_simple_compute_map removed - soac_parallelize transforms the MIR instead
use crate::mir::to_ssa::convert_body;
use crate::mir::{
    self, ArrayBacking, Body, Def, ExecutionModel, Expr, ExprId, LambdaId, LambdaInfo, LocalId, Program,
};
use crate::pipeline::{self, Pipeline};

/// Tracks SPIR-V descriptor set and binding numbers for storage buffers.
/// This is an internal implementation detail for SPIR-V lowering.
/// Address space is tracked in types (Storage/Function), but we still need
/// to assign specific descriptor bindings during SPIR-V lowering.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemBinding {
    Storage {
        set: u32,
        binding: u32,
    },
}
use crate::types;
use crate::{IdArena, bail_spirv, err_spirv};
use polytype::Type as PolyType;
use rspirv::binary::Assemble;
use rspirv::dr::Builder;
use rspirv::dr::Operand;
use rspirv::spirv::{self, AddressingModel, Capability, MemoryModel, StorageClass};
use std::collections::HashMap;

/// Tracks the lowering state of each definition
#[derive(Clone, Copy, PartialEq, Eq)]
enum LowerState {
    NotStarted,
    InProgress,
    Done,
}
/// Entry point information for SPIR-V emission
struct EntryPointInfo {
    name: String,
    model: spirv::ExecutionModel,
    /// Local size for compute shaders (x, y, z)
    local_size: Option<(u32, u32, u32)>,
}

/// Context for on-demand lowering of MIR to SPIR-V
struct LowerCtx<'a> {
    /// The MIR program being lowered
    program: &'a Program,
    /// Map from definition name to its index in program.defs
    def_index: HashMap<String, usize>,
    /// Lowering state of each definition
    state: HashMap<String, LowerState>,
    /// The SPIR-V builder
    constructor: Constructor,
    /// Entry points to emit
    entry_points: Vec<EntryPointInfo>,
}

/// Constructor wraps rspirv::Builder with an ergonomic API that handles:
/// - Automatic variable hoisting to function entry block
/// - Block management with implicit branch from variables block to code
/// - Value and type caching
pub(crate) struct Constructor {
    pub(crate) builder: Builder,

    // Type caching
    pub(crate) void_type: spirv::Word,
    pub(crate) bool_type: spirv::Word,
    pub(crate) i32_type: spirv::Word,
    pub(crate) u32_type: spirv::Word,
    pub(crate) f32_type: spirv::Word,

    // Constant caching
    int_const_cache: HashMap<i32, spirv::Word>,
    pub(crate) int_const_reverse: HashMap<spirv::Word, i32>, // reverse lookup: ID -> value
    uint_const_cache: HashMap<u32, spirv::Word>,
    uint_const_reverse: HashMap<spirv::Word, u32>, // reverse lookup: ID -> value
    float_const_cache: HashMap<u32, spirv::Word>,  // bits as u32
    bool_const_cache: HashMap<bool, spirv::Word>,

    // Current function state
    pub(crate) current_block: Option<spirv::Word>,
    variables_block: Option<spirv::Word>, // Block for OpVariable declarations
    first_code_block: Option<spirv::Word>, // First block with actual code

    // Environment: name -> value ID
    pub(crate) env: HashMap<String, spirv::Word>,

    // Function map: name -> function ID
    pub(crate) functions: HashMap<String, spirv::Word>,

    // GLSL extended instruction set
    pub(crate) glsl_ext_inst_id: spirv::Word,

    // Type cache: avoid recreating same types
    vec_type_cache: HashMap<(spirv::Word, u32), spirv::Word>,
    struct_type_cache: HashMap<Vec<spirv::Word>, spirv::Word>,
    ptr_type_cache: HashMap<(spirv::StorageClass, spirv::Word), spirv::Word>,
    runtime_array_cache: HashMap<(spirv::Word, u32), spirv::Word>, // (elem_type, stride) -> decorated type
    buffer_block_cache: HashMap<spirv::Word, spirv::Word>, // runtime_array_type -> Block-decorated struct

    // Entry point interface tracking
    pub(crate) entry_point_interfaces: HashMap<String, Vec<spirv::Word>>,
    current_is_entry_point: bool,
    current_output_vars: Vec<spirv::Word>,
    current_input_vars: Vec<(spirv::Word, String, spirv::Word)>, // (var_id, param_name, type_id)
    current_used_globals: Vec<spirv::Word>, // Global constants accessed in current entry point

    // Global constants: name -> constant_id (SPIR-V OpConstant)
    pub(crate) global_constants: HashMap<String, spirv::Word>,
    pub(crate) uniform_variables: HashMap<String, spirv::Word>,
    pub(crate) uniform_types: HashMap<String, spirv::Word>, // uniform name -> SPIR-V type ID
    uniform_load_cache: HashMap<String, spirv::Word>,       // cached OpLoad results per function
    extract_cache: HashMap<(spirv::Word, u32), spirv::Word>, // CSE for OpCompositeExtract

    /// Lambda registry: LambdaId -> LambdaInfo
    lambda_registry: IdArena<LambdaId, LambdaInfo>,

    // Builtin function registry
    pub(crate) impl_source: ImplSource,

    /// In-place optimization: ExprIds of operations where input array can be reused

    /// Storage buffers for compute shaders: (set, binding) -> (buffer_var, elem_type_id, buffer_ptr_type)
    pub(crate) storage_buffers: HashMap<(u32, u32), (spirv::Word, spirv::Word, spirv::Word)>,

    /// Compute shader parameters: maps param name to (set, binding) for storage buffer lookup
    compute_params: HashMap<String, (u32, u32)>,

    /// GlobalInvocationId variable for compute shaders (set during entry point setup)
    pub(crate) global_invocation_id: Option<spirv::Word>,

    /// Linked SPIR-V functions: linkage_name -> function_id
    pub(crate) linked_functions: HashMap<String, spirv::Word>,

    /// Whether we need linkage capability
    needs_linkage: bool,
}

impl Constructor {
    pub(crate) fn new() -> Self {
        let mut builder = Builder::new();
        builder.set_version(1, 5);
        builder.capability(Capability::Shader);
        builder.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

        let void_type = builder.type_void();
        let bool_type = builder.type_bool();
        let i32_type = builder.type_int(32, 1);
        let u32_type = builder.type_int(32, 0);
        let f32_type = builder.type_float(32);
        let glsl_ext_inst_id = builder.ext_inst_import("GLSL.std.450");

        Constructor {
            builder,
            void_type,
            bool_type,
            i32_type,
            u32_type,
            f32_type,
            int_const_cache: HashMap::new(),
            int_const_reverse: HashMap::new(),
            uint_const_cache: HashMap::new(),
            uint_const_reverse: HashMap::new(),
            float_const_cache: HashMap::new(),
            bool_const_cache: HashMap::new(),
            current_block: None,
            variables_block: None,
            first_code_block: None,
            env: HashMap::new(),
            functions: HashMap::new(),
            glsl_ext_inst_id,
            vec_type_cache: HashMap::new(),
            struct_type_cache: HashMap::new(),
            ptr_type_cache: HashMap::new(),
            runtime_array_cache: HashMap::new(),
            buffer_block_cache: HashMap::new(),
            entry_point_interfaces: HashMap::new(),
            current_is_entry_point: false,
            current_output_vars: Vec::new(),
            current_input_vars: Vec::new(),
            current_used_globals: Vec::new(),
            global_constants: HashMap::new(),
            uniform_variables: HashMap::new(),
            uniform_types: HashMap::new(),
            uniform_load_cache: HashMap::new(),
            extract_cache: HashMap::new(),
            lambda_registry: IdArena::new(),
            impl_source: ImplSource::default(),
            storage_buffers: HashMap::new(),
            compute_params: HashMap::new(),
            global_invocation_id: None,
            linked_functions: HashMap::new(),
            needs_linkage: false,
        }
    }

    /// Get or create a pointer type
    pub(crate) fn get_or_create_ptr_type(
        &mut self,
        storage_class: spirv::StorageClass,
        pointee_id: spirv::Word,
    ) -> spirv::Word {
        let key = (storage_class, pointee_id);
        if let Some(&ty) = self.ptr_type_cache.get(&key) {
            return ty;
        }
        let ty = self.builder.type_pointer(None, storage_class, pointee_id);
        self.ptr_type_cache.insert(key, ty);
        ty
    }

    /// Convert a polytype Type to a SPIR-V type ID
    pub(crate) fn polytype_to_spirv(&mut self, ty: &PolyType<TypeName>) -> spirv::Word {
        match ty {
            PolyType::Variable(id) => {
                panic!("BUG: Unresolved type variable Variable({}) reached lowering.", id);
            }
            PolyType::Constructed(name, args) => {
                // Assert that no UserVar or SizeVar reaches lowering
                match name {
                    TypeName::UserVar(v) => {
                        panic!("BUG: UserVar('{}') reached lowering.", v);
                    }
                    TypeName::SizeVar(v) => {
                        panic!("BUG: SizeVar('{}') reached lowering.", v);
                    }
                    _ => {}
                }

                match name {
                    TypeName::Int(32) => self.i32_type,
                    TypeName::Float(32) => self.f32_type,
                    TypeName::Int(bits) => self.builder.type_int(*bits as u32, 1),
                    TypeName::UInt(bits) => self.builder.type_int(*bits as u32, 0),
                    TypeName::Float(bits) => self.builder.type_float(*bits as u32),
                    TypeName::Str(s) if *s == "bool" => self.bool_type,
                    TypeName::Unit => {
                        // Unit type - use void type
                        // Unit values are never actually constructed since they can only be assigned to _
                        self.void_type
                    }
                    TypeName::Tuple(_) => {
                        // Empty tuples should not reach lowering:
                        // - Unit values are bound to _ (not stored)
                        // - Empty closures are handled specially in map (dummy i32 passed directly)
                        if args.is_empty() {
                            panic!(
                                "BUG: Empty tuple type reached lowering. Empty tuples/unit values should be \
                                handled at call sites (let _ = ..., map with empty closures, etc.)"
                            );
                        }
                        // Non-empty tuple becomes struct
                        let field_types: Vec<spirv::Word> =
                            args.iter().map(|a| self.polytype_to_spirv(a)).collect();
                        self.get_or_create_struct_type(field_types)
                    }
                    TypeName::Array => {
                        // Array[elem, variant, size]
                        assert!(args.len() == 3);
                        let elem_type = self.polytype_to_spirv(&args[0]);
                        let variant = &args[1];
                        let size = &args[2];

                        // Dispatch on variant first - View arrays are always {buffer_ptr, offset, len} structs
                        if let PolyType::Constructed(TypeName::ArrayVariantView, _) = variant {
                            // View variant: struct { buffer_ptr, offset, len }
                            // buffer_ptr points to the StorageBuffer block struct containing the runtime array
                            let stride = crate::mir::layout::type_byte_size(&args[0])
                                .expect("View element type must have computable size");
                            let runtime_array_type =
                                self.get_or_create_runtime_array_type(elem_type, stride);
                            let buffer_struct_type =
                                self.get_or_create_buffer_block_type(runtime_array_type);
                            let buffer_ptr_type = self.get_or_create_ptr_type(
                                spirv::StorageClass::StorageBuffer,
                                buffer_struct_type,
                            );
                            self.get_or_create_struct_type(vec![
                                buffer_ptr_type,
                                self.u32_type,
                                self.u32_type,
                            ])
                        } else if let PolyType::Constructed(TypeName::ArrayVariantVirtual, _) = variant {
                            // Virtual variant: struct { start, step, len } for range representation
                            self.get_or_create_struct_type(vec![
                                self.i32_type,
                                self.i32_type,
                                self.i32_type,
                            ])
                        } else {
                            // Composite variant (or placeholder): sized array value
                            match size {
                                PolyType::Constructed(TypeName::Size(n), _) => {
                                    // Fixed-size array
                                    let size_const = self.const_i32(*n as i32);
                                    self.builder.type_array(elem_type, size_const)
                                }
                                PolyType::Constructed(TypeName::SizePlaceholder, _) => {
                                    panic!("SizePlaceholder should be resolved before SPIR-V lowering");
                                }
                                PolyType::Variable(_) => {
                                    // Unsized composite array - not supported
                                    panic!("BUG: Composite variant unsized arrays not supported: {:?}", ty);
                                }
                                _ => {
                                    panic!(
                                        "BUG: Array type has invalid size argument: {:?}. This should have been resolved during type checking.",
                                        size
                                    );
                                }
                            }
                        }
                    }
                    TypeName::Vec => {
                        // Vec type with args: args[0] is size, args[1] is element type
                        if args.len() < 2 {
                            panic!(
                                "BUG: Vec type requires 2 arguments (size, element_type), got {}.",
                                args.len()
                            );
                        }
                        let size = match &args[0] {
                            PolyType::Constructed(TypeName::Size(n), _) => *n as u32,
                            _ => {
                                panic!("BUG: Vec type has invalid size argument: {:?}.", args[0]);
                            }
                        };
                        let elem_type = self.polytype_to_spirv(&args[1]);
                        self.get_or_create_vec_type(elem_type, size)
                    }
                    TypeName::Mat => {
                        // Mat type with args: args[0] is cols, args[1] is rows, args[2] is element type
                        if args.len() < 3 {
                            panic!(
                                "BUG: Mat type requires 3 arguments (cols, rows, element_type), got {}.",
                                args.len()
                            );
                        }
                        let cols = match &args[0] {
                            PolyType::Constructed(TypeName::Size(n), _) => *n as u32,
                            _ => {
                                panic!("BUG: Mat type has invalid cols argument: {:?}.", args[0]);
                            }
                        };
                        let rows = match &args[1] {
                            PolyType::Constructed(TypeName::Size(n), _) => *n as u32,
                            _ => {
                                panic!("BUG: Mat type has invalid rows argument: {:?}.", args[1]);
                            }
                        };
                        let elem_type = self.polytype_to_spirv(&args[2]);
                        let col_vec_type = self.get_or_create_vec_type(elem_type, rows);
                        self.builder.type_matrix(col_vec_type, cols)
                    }
                    TypeName::Record(_fields) => {
                        panic!("should never get here")
                    }
                    TypeName::Pointer => {
                        // Pointer type: args[0] is pointee type
                        if args.is_empty() {
                            panic!("BUG: Pointer type requires a pointee type argument.");
                        }
                        let pointee_type = self.polytype_to_spirv(&args[0]);
                        self.builder.type_pointer(None, StorageClass::Function, pointee_type)
                    }
                    TypeName::Unique => {
                        // Unique type wrapper: strip and convert underlying type
                        // Unique is only used for alias checking, has no runtime representation
                        if args.is_empty() {
                            panic!("BUG: Unique type requires an underlying type argument.");
                        }
                        self.polytype_to_spirv(&args[0])
                    }
                    TypeName::Existential(_) => {
                        // Existential type: unwrap and convert the inner type (in args[0])
                        // The size variable is runtime-determined, handled by Slice representation
                        let inner = &args[0];
                        self.polytype_to_spirv(inner)
                    }
                    TypeName::Arrow => {
                        // Arrow types (function types) come from closures that have been defunctionalized.
                        // Since closures are represented as (captures_tuple, lambda_name), the actual
                        // runtime value is just the captures. The Arrow type is a phantom type used
                        // for type checking only. Map to unit type since it has no runtime representation.
                        self.void_type
                    }
                    TypeName::ArrayVariantComposite | TypeName::ArrayVariantView => {
                        // Address space markers are used within Array types but shouldn't appear
                        // as standalone types requiring SPIR-V representation.
                        panic!(
                            "BUG: Address space marker {:?} reached polytype_to_spirv as standalone type. \
                            This should only appear as part of Array[elem, addrspace, size]. Full type: {:?}",
                            name, ty
                        );
                    }
                    TypeName::AddressPlaceholder | TypeName::SizePlaceholder => {
                        panic!("Placeholders should be resolved before SPIR-V lowering");
                    }
                    _ => {
                        panic!(
                            "BUG: Unknown type reached lowering: {:?}. This should have been caught during type checking.",
                            name
                        )
                    }
                }
            }
        }
    }

    /// Get or create a vector type
    pub(crate) fn get_or_create_vec_type(&mut self, elem_type: spirv::Word, size: u32) -> spirv::Word {
        let key = (elem_type, size);
        if let Some(&ty) = self.vec_type_cache.get(&key) {
            return ty;
        }
        let ty = self.builder.type_vector(elem_type, size);
        self.vec_type_cache.insert(key, ty);
        ty
    }

    /// Get or create a struct type
    pub(crate) fn get_or_create_struct_type(&mut self, field_types: Vec<spirv::Word>) -> spirv::Word {
        if let Some(&ty) = self.struct_type_cache.get(&field_types) {
            return ty;
        }
        let ty = self.builder.type_struct(field_types.clone());
        self.struct_type_cache.insert(field_types, ty);
        ty
    }

    /// Get or create a runtime array type with ArrayStride decoration
    fn get_or_create_runtime_array_type(&mut self, elem_type: spirv::Word, stride: u32) -> spirv::Word {
        let key = (elem_type, stride);
        if let Some(&ty) = self.runtime_array_cache.get(&key) {
            return ty;
        }
        let ty = self.builder.type_runtime_array(elem_type);
        self.builder.decorate(
            ty,
            spirv::Decoration::ArrayStride,
            [Operand::LiteralBit32(stride)],
        );
        self.runtime_array_cache.insert(key, ty);
        ty
    }

    /// Get or create a Block-decorated struct wrapping a runtime array (for storage buffers)
    fn get_or_create_buffer_block_type(&mut self, runtime_array_type: spirv::Word) -> spirv::Word {
        if let Some(&ty) = self.buffer_block_cache.get(&runtime_array_type) {
            return ty;
        }
        let ty = self.builder.type_struct([runtime_array_type]);
        self.builder.decorate(ty, spirv::Decoration::Block, []);
        self.builder.member_decorate(ty, 0, spirv::Decoration::Offset, [Operand::LiteralBit32(0)]);
        self.buffer_block_cache.insert(runtime_array_type, ty);
        ty
    }

    /// Create a Block-decorated struct type for a uniform buffer.
    /// Returns the struct type ID. Each uniform gets its own unique struct
    /// (not cached) since Block structs shouldn't be shared.
    pub(crate) fn create_uniform_block_type(&mut self, value_type: spirv::Word) -> spirv::Word {
        let block_struct = self.builder.type_struct(vec![value_type]);

        // Decorate as Block (required for UBO in Vulkan)
        self.builder.decorate(block_struct, spirv::Decoration::Block, []);

        // Decorate member 0 with Offset 0
        self.builder.member_decorate(
            block_struct,
            0,
            spirv::Decoration::Offset,
            [Operand::LiteralBit32(0)],
        );

        block_struct
    }

    /// Begin a new function
    pub(crate) fn begin_function(
        &mut self,
        name: &str,
        param_names: &[&str],
        param_types: &[spirv::Word],
        return_type: spirv::Word,
    ) -> Result<spirv::Word> {
        let func_type = self.builder.type_function(return_type, param_types.to_vec());
        let func_id =
            self.builder.begin_function(return_type, None, spirv::FunctionControl::NONE, func_type)?;

        self.functions.insert(name.to_string(), func_id);

        // Create function parameters
        for (i, &param_name) in param_names.iter().enumerate() {
            let param_id = self.builder.function_parameter(param_types[i])?;
            self.env.insert(param_name.to_string(), param_id);
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
        self.uniform_load_cache.clear();
        self.extract_cache.clear();

        Ok(())
    }

    /// Declare a variable in the function's variables block
    pub(crate) fn declare_variable(&mut self, _name: &str, value_type: spirv::Word) -> Result<spirv::Word> {
        let ptr_type = self.builder.type_pointer(None, StorageClass::Function, value_type);

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
    pub(crate) fn const_i32(&mut self, value: i32) -> spirv::Word {
        if let Some(&id) = self.int_const_cache.get(&value) {
            return id;
        }
        let id = self.builder.constant_bit32(self.i32_type, value as u32);
        self.int_const_cache.insert(value, id);
        self.int_const_reverse.insert(id, value);
        id
    }

    /// Get the literal i32 value from a constant ID (reverse lookup)
    pub(crate) fn get_const_i32_value(&self, id: spirv::Word) -> Option<i32> {
        self.int_const_reverse.get(&id).copied()
    }

    /// Get or create a u32 constant
    pub(crate) fn const_u32(&mut self, value: u32) -> spirv::Word {
        if let Some(&id) = self.uint_const_cache.get(&value) {
            return id;
        }
        let id = self.builder.constant_bit32(self.u32_type, value);
        self.uint_const_cache.insert(value, id);
        self.uint_const_reverse.insert(id, value);
        id
    }

    /// Get the literal u32 value from a constant ID (reverse lookup)
    pub(crate) fn get_const_u32_value(&self, id: spirv::Word) -> Option<u32> {
        self.uint_const_reverse.get(&id).copied()
    }

    /// Get or create an f32 constant
    pub(crate) fn const_f32(&mut self, value: f32) -> spirv::Word {
        let bits = value.to_bits();
        if let Some(&id) = self.float_const_cache.get(&bits) {
            return id;
        }
        let id = self.builder.constant_bit32(self.f32_type, bits);
        self.float_const_cache.insert(bits, id);
        id
    }

    /// Get or create a bool constant
    pub(crate) fn const_bool(&mut self, value: bool) -> spirv::Word {
        if let Some(&id) = self.bool_const_cache.get(&value) {
            return id;
        }
        let id = if value {
            self.builder.constant_true(self.bool_type)
        } else {
            self.builder.constant_false(self.bool_type)
        };
        self.bool_const_cache.insert(value, id);
        id
    }

    /// Begin a block (must be called before emitting instructions into it)
    pub(crate) fn begin_block(&mut self, block_id: spirv::Word) -> Result<()> {
        self.builder.begin_block(Some(block_id))?;
        self.current_block = Some(block_id);
        // Clear extract cache since values from previous blocks may not dominate this block
        self.extract_cache.clear();
        Ok(())
    }
}

impl<'a> LowerCtx<'a> {
    fn new(program: &'a Program) -> Self {
        let mut constructor = Constructor::new();
        constructor.lambda_registry = program.lambda_registry.clone();

        // Build index from name to def position
        let mut def_index = HashMap::new();
        let mut entry_points = Vec::new();

        for (i, def) in program.defs.iter().enumerate() {
            let name = match def {
                Def::Function { name, .. } => name.clone(),
                Def::EntryPoint {
                    name,
                    execution_model,
                    ..
                } => {
                    // Collect entry points
                    match execution_model {
                        mir::ExecutionModel::Vertex => {
                            entry_points.push(EntryPointInfo {
                                name: name.clone(),
                                model: spirv::ExecutionModel::Vertex,
                                local_size: None,
                            });
                        }
                        mir::ExecutionModel::Fragment => {
                            entry_points.push(EntryPointInfo {
                                name: name.clone(),
                                model: spirv::ExecutionModel::Fragment,
                                local_size: None,
                            });
                        }
                        mir::ExecutionModel::Compute { local_size } => {
                            entry_points.push(EntryPointInfo {
                                name: name.clone(),
                                model: spirv::ExecutionModel::GLCompute,
                                local_size: Some(*local_size),
                            });
                        }
                    }
                    name.clone()
                }
                Def::Constant { name, .. } => name.clone(),
                Def::Uniform { name, .. } => name.clone(),
                Def::Storage { name, .. } => name.clone(),
            };
            def_index.insert(name, i);
        }

        LowerCtx {
            program,
            def_index,
            state: HashMap::new(),
            constructor,
            entry_points,
        }
    }

    /// Ensure a definition is lowered, recursively lowering dependencies first
    fn ensure_lowered(&mut self, name: &str) -> Result<()> {
        match self.state.get(name).copied().unwrap_or(LowerState::NotStarted) {
            LowerState::Done => return Ok(()),
            LowerState::InProgress => {
                bail_spirv!("Recursive definition detected: {}", name);
            }
            LowerState::NotStarted => { /* proceed */ }
        }

        // Look up the definition
        let def_idx = match self.def_index.get(name) {
            Some(&idx) => idx,
            None => return Ok(()), // Not a user def (might be a builtin)
        };

        self.state.insert(name.to_string(), LowerState::InProgress);

        let def = &self.program.defs[def_idx];
        self.lower_def(def)?;

        self.state.insert(name.to_string(), LowerState::Done);
        Ok(())
    }

    /// Lower a single definition
    fn lower_def(&mut self, def: &Def) -> Result<()> {
        match def {
            Def::Function {
                name,
                params,
                ret_type,
                body,
                dps_output,
                ..
            } => {
                // Check if this is an extern function declaration
                if let Expr::Extern(linkage_name) = body.get_expr(body.root) {
                    // This is an extern function - create an imported function declaration
                    lower_extern_function(
                        &mut self.constructor,
                        name,
                        params,
                        ret_type,
                        body,
                        linkage_name,
                    )?;
                } else {
                    // First, ensure all dependencies are lowered
                    self.ensure_deps_lowered(body)?;

                    // Regular function (entry points are now Def::EntryPoint)
                    // DPS functions have dps_output set and return Unit
                    lower_regular_function(
                        &mut self.constructor,
                        name,
                        params,
                        ret_type,
                        body,
                        *dps_output,
                    )?;
                }
            }
            Def::EntryPoint {
                name,
                execution_model: ExecutionModel::Compute { .. },
                outputs,
                body,
                ..
            } => {
                // Build pipeline from compute entry point for buffer setup
                let Some(pipeline) = pipeline::build_pipeline(def) else {
                    bail_spirv!("Compute shader '{}' could not be converted to pipeline", name);
                };

                // Ensure dependencies are lowered
                self.ensure_deps_lowered(body)?;

                // Lower the compute shader - the MIR should already be transformed by soac_parallelize
                lower_compute_entry_point(&mut self.constructor, &pipeline, outputs, body)?;
            }
            Def::EntryPoint {
                name,
                inputs,
                outputs,
                body,
                ..
            } => {
                // Vertex/Fragment entry points
                // First, ensure all dependencies are lowered
                self.ensure_deps_lowered(body)?;

                lower_entry_point_from_def(&mut self.constructor, name, inputs, outputs, body)?;
            }
            Def::Constant { name, body, .. } => {
                // First, ensure all dependencies are lowered
                self.ensure_deps_lowered(body)?;

                // Evaluate the constant expression at compile time
                // lower_const_expr validates that the expression is a compile-time constant
                let const_id = lower_const_expr(&mut self.constructor, body, body.root)?;

                // Store constant ID for lookup
                self.constructor.global_constants.insert(name.clone(), const_id);
            }
            Def::Uniform {
                name,
                ty,
                set,
                binding,
                ..
            } => {
                // Create a SPIR-V uniform variable wrapped in a Block-decorated struct
                // (required for Vulkan uniform buffer compatibility)
                let value_type = self.constructor.polytype_to_spirv(ty);
                let block_type = self.constructor.create_uniform_block_type(value_type);
                let ptr_type =
                    self.constructor.get_or_create_ptr_type(spirv::StorageClass::Uniform, block_type);
                let var_id =
                    self.constructor.builder.variable(ptr_type, None, spirv::StorageClass::Uniform, None);

                // Decorate with descriptor set and binding
                self.constructor.builder.decorate(
                    var_id,
                    spirv::Decoration::DescriptorSet,
                    [Operand::LiteralBit32(*set)],
                );
                self.constructor.builder.decorate(
                    var_id,
                    spirv::Decoration::Binding,
                    [Operand::LiteralBit32(*binding)],
                );

                // Store uniform variable ID and the inner value type for lookup
                self.constructor.uniform_variables.insert(name.clone(), var_id);
                self.constructor.uniform_types.insert(name.clone(), value_type);
            }
            Def::Storage {
                name,
                ty,
                set,
                binding,
                ..
            } => {
                // Create a SPIR-V storage buffer variable
                // TODO: This works correctly but reuses `uniform_variables` map for storage buffers.
                // A cleaner implementation might:
                // 1. Rename to `descriptor_variables` (more accurate)
                // 2. Or have separate `uniform_variables` and `storage_variables` maps
                let storage_type = self.constructor.polytype_to_spirv(ty);
                let ptr_type = self
                    .constructor
                    .get_or_create_ptr_type(spirv::StorageClass::StorageBuffer, storage_type);
                let var_id = self.constructor.builder.variable(
                    ptr_type,
                    None,
                    spirv::StorageClass::StorageBuffer,
                    None,
                );

                // Decorate with descriptor set and binding
                self.constructor.builder.decorate(
                    var_id,
                    spirv::Decoration::DescriptorSet,
                    [Operand::LiteralBit32(*set)],
                );
                self.constructor.builder.decorate(
                    var_id,
                    spirv::Decoration::Binding,
                    [Operand::LiteralBit32(*binding)],
                );

                // Store storage buffer variable ID for lookup
                self.constructor.uniform_variables.insert(name.clone(), var_id);
                self.constructor.uniform_types.insert(name.clone(), storage_type);
            }
        }
        Ok(())
    }

    /// Walk a body and ensure all referenced definitions are lowered
    fn ensure_deps_lowered(&mut self, body: &Body) -> Result<()> {
        // With arena-based MIR, we can simply walk all expressions in the body
        // since they're stored flat in a Vec
        for expr in body.iter_exprs() {
            match expr {
                Expr::Global(name) => {
                    self.ensure_lowered(name)?;
                }
                Expr::Call { func, .. } => {
                    self.ensure_lowered(func)?;
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Run the lowering, starting from entry points
    fn run(mut self) -> Result<Vec<u32>> {
        // Lower all entry points (and their dependencies, including uniforms)
        let entry_names: Vec<String> = self.entry_points.iter().map(|ep| ep.name.clone()).collect();
        for name in entry_names {
            self.ensure_lowered(&name)?;
        }

        // Emit entry points with interface variables
        for ep in &self.entry_points {
            if let Some(&func_id) = self.constructor.functions.get(&ep.name) {
                let mut interfaces =
                    self.constructor.entry_point_interfaces.get(&ep.name).cloned().unwrap_or_default();
                // Global constants are now true OpConstants (not variables), so they don't need
                // to be added to the interface list
                // Add all uniform variables to the interface
                for &uniform_var in self.constructor.uniform_variables.values() {
                    if !interfaces.contains(&uniform_var) {
                        interfaces.push(uniform_var);
                    }
                }
                self.constructor.builder.entry_point(ep.model, func_id, &ep.name, interfaces);

                // Add execution modes
                match ep.model {
                    spirv::ExecutionModel::Fragment => {
                        self.constructor.builder.execution_mode(
                            func_id,
                            spirv::ExecutionMode::OriginUpperLeft,
                            [],
                        );
                    }
                    spirv::ExecutionModel::GLCompute => {
                        // Compute shaders need VariablePointersStorageBuffer for buffer access
                        self.constructor.builder.capability(Capability::VariablePointersStorageBuffer);
                        if let Some((x, y, z)) = ep.local_size {
                            self.constructor.builder.execution_mode(
                                func_id,
                                spirv::ExecutionMode::LocalSize,
                                [x, y, z],
                            );
                        }
                    }
                    _ => {}
                }
            }
        }

        Ok(self.constructor.builder.module().assemble())
    }
}

/// Lower a MIR program to SPIR-V
pub fn lower(program: &mir::Program) -> Result<Vec<u32>> {
    // Use a thread with larger stack size to handle deeply nested expressions
    // Default Rust stack is 2MB on macOS which is too small for complex shaders
    const STACK_SIZE: usize = 16 * 1024 * 1024; // 16MB

    // Clone program since we need 'static lifetime for thread
    let program_clone = program.clone();

    let handle = std::thread::Builder::new()
        .stack_size(STACK_SIZE)
        .spawn(move || {
            let ctx = LowerCtx::new(&program_clone);
            ctx.run()
        })
        .expect("Failed to spawn lowering thread");

    handle.join().expect("Lowering thread panicked")
}

fn lower_regular_function(
    constructor: &mut Constructor,
    name: &str,
    params: &[LocalId],
    ret_type: &PolyType<TypeName>,
    body: &Body,
    _dps_output: Option<LocalId>,
) -> Result<()> {
    // Check if first parameter is an empty closure (lambda with no captures)
    // If so, skip it - don't include in SPIR-V function signature
    let skip_first_param = if let Some(&first_param_id) = params.first() {
        is_empty_closure_type(&body.get_local(first_param_id).ty)
    } else {
        false
    };

    let params_to_lower = if skip_first_param { &params[1..] } else { params };

    let param_names: Vec<&str> = params_to_lower.iter().map(|&p| body.get_local(p).name.as_str()).collect();
    let param_types: Vec<spirv::Word> =
        params_to_lower.iter().map(|&p| constructor.polytype_to_spirv(&body.get_local(p).ty)).collect();
    let return_type = constructor.polytype_to_spirv(ret_type);

    // Convert to SSA form
    let param_names_owned: Vec<String> = param_names.iter().map(|s| s.to_string()).collect();
    let ssa_body = convert_body(body, &param_names_owned, ret_type.clone())
        .map_err(|e| err_spirv!("SSA conversion failed: {}", e))?;

    // Begin SPIR-V function and lower SSA body
    constructor.begin_function(name, &param_names, &param_types, return_type)?;
    super::ssa_lowering::lower_ssa_body(constructor, &ssa_body)?;
    constructor.end_function()?;
    Ok(())
}

/// Lower an extern function declaration to a SPIR-V import.
/// This creates a function declaration with Import linkage that will be resolved by spirv-link.
fn lower_extern_function(
    constructor: &mut Constructor,
    name: &str,
    params: &[LocalId],
    ret_type: &PolyType<TypeName>,
    body: &Body,
    linkage_name: &str,
) -> Result<()> {
    // Add Linkage capability if not already added
    if !constructor.needs_linkage {
        constructor.builder.capability(Capability::Linkage);
        constructor.needs_linkage = true;
    }

    // Skip empty closure parameters (same logic as regular functions)
    let skip_first_param = if let Some(&first_param_id) = params.first() {
        is_empty_closure_type(&body.get_local(first_param_id).ty)
    } else {
        false
    };

    let params_to_lower = if skip_first_param { &params[1..] } else { params };

    // Build parameter types
    let param_types: Vec<spirv::Word> =
        params_to_lower.iter().map(|&p| constructor.polytype_to_spirv(&body.get_local(p).ty)).collect();
    let return_type = constructor.polytype_to_spirv(ret_type);

    // Create function type
    let func_type = constructor.builder.type_function(return_type, param_types.iter().copied());

    // Create the function declaration with no body
    // Use Pure function control to indicate no side effects
    let func_id =
        constructor.builder.begin_function(return_type, None, spirv::FunctionControl::NONE, func_type)?;

    // Add function parameters (required even for imported functions)
    for &param_id in params_to_lower {
        let param_ty = constructor.polytype_to_spirv(&body.get_local(param_id).ty);
        constructor.builder.function_parameter(param_ty)?;
    }

    // End the function (no body for imports)
    constructor.builder.end_function()?;

    // Decorate with Import linkage
    constructor.builder.decorate(
        func_id,
        spirv::Decoration::LinkageAttributes,
        [
            Operand::LiteralString(linkage_name.to_string()),
            Operand::LinkageType(spirv::LinkageType::Import),
        ],
    );

    // Store the function ID for later reference
    constructor.functions.insert(name.to_string(), func_id);
    constructor.linked_functions.insert(linkage_name.to_string(), func_id);

    Ok(())
}

/// Lower an entry point from Def::EntryPoint structure (new format with EntryInput/EntryOutput)
fn lower_entry_point_from_def(
    constructor: &mut Constructor,
    name: &str,
    inputs: &[mir::EntryInput],
    outputs: &[mir::EntryOutput],
    body: &Body,
) -> Result<()> {
    constructor.current_is_entry_point = true;
    constructor.current_output_vars.clear();
    constructor.current_input_vars.clear();

    let mut interface_vars = Vec::new();

    // Create Input variables for parameters
    for input in inputs.iter() {
        let input_type_id = constructor.polytype_to_spirv(&input.ty);
        let ptr_type_id = constructor.get_or_create_ptr_type(StorageClass::Input, input_type_id);
        let var_id = constructor.builder.variable(ptr_type_id, None, StorageClass::Input, None);

        // Add decorations from IoDecoration
        if let Some(decoration) = &input.decoration {
            match decoration {
                mir::IoDecoration::Location(loc) => {
                    constructor.builder.decorate(
                        var_id,
                        spirv::Decoration::Location,
                        [rspirv::dr::Operand::LiteralBit32(*loc)],
                    );
                }
                mir::IoDecoration::BuiltIn(builtin) => {
                    constructor.builder.decorate(
                        var_id,
                        spirv::Decoration::BuiltIn,
                        [rspirv::dr::Operand::BuiltIn(*builtin)],
                    );
                }
            }
        }

        interface_vars.push(var_id);
        constructor.current_input_vars.push((var_id, input.name.clone(), input_type_id));
    }

    // Create Output variables for return values (skip Unit types - they have no output)
    for output in outputs.iter() {
        // Skip Unit type outputs - compute shaders returning () shouldn't have output variables
        if matches!(&output.ty, PolyType::Constructed(TypeName::Unit, _)) {
            continue;
        }

        let output_type_id = constructor.polytype_to_spirv(&output.ty);
        let ptr_type_id = constructor.get_or_create_ptr_type(StorageClass::Output, output_type_id);
        let var_id = constructor.builder.variable(ptr_type_id, None, StorageClass::Output, None);

        // Add decorations from IoDecoration
        if let Some(decoration) = &output.decoration {
            match decoration {
                mir::IoDecoration::Location(loc) => {
                    constructor.builder.decorate(
                        var_id,
                        spirv::Decoration::Location,
                        [rspirv::dr::Operand::LiteralBit32(*loc)],
                    );
                }
                mir::IoDecoration::BuiltIn(builtin) => {
                    constructor.builder.decorate(
                        var_id,
                        spirv::Decoration::BuiltIn,
                        [rspirv::dr::Operand::BuiltIn(*builtin)],
                    );
                }
            }
        }

        interface_vars.push(var_id);
        constructor.current_output_vars.push(var_id);
    }

    // Store interface variables for entry point declaration
    constructor.entry_point_interfaces.insert(name.to_string(), interface_vars);

    // Create void(void) function for entry point
    let func_type = constructor.builder.type_function(constructor.void_type, vec![]);
    let func_id = constructor.builder.begin_function(
        constructor.void_type,
        None,
        spirv::FunctionControl::NONE,
        func_type,
    )?;
    constructor.functions.insert(name.to_string(), func_id);

    // Create two blocks: one for variables, one for code (same pattern as regular functions)
    let vars_block_id = constructor.builder.id();
    let code_block_id = constructor.builder.id();
    constructor.variables_block = Some(vars_block_id);
    constructor.first_code_block = Some(code_block_id);

    // Begin variables block (leave it open - no terminator yet)
    constructor.builder.begin_block(Some(vars_block_id))?;

    // Deselect current block so we can begin a new one
    constructor.builder.select_block(None)?;

    // Begin code block - this is where we'll emit code
    constructor.builder.begin_block(Some(code_block_id))?;
    constructor.current_block = Some(code_block_id);

    // Load input variables into environment
    for (var_id, param_name, type_id) in constructor.current_input_vars.clone() {
        let loaded = constructor.builder.load(type_id, None, var_id, None, [])?;
        constructor.env.insert(param_name, loaded);
    }

    // Convert to SSA and lower
    let param_names: Vec<String> = inputs.iter().map(|i| i.name.clone()).collect();
    let return_ty = if outputs.len() == 1 {
        outputs[0].ty.clone()
    } else if outputs.is_empty() {
        PolyType::Constructed(TypeName::Unit, vec![])
    } else {
        // Multiple outputs - tuple type
        PolyType::Constructed(
            TypeName::Tuple(outputs.len()),
            outputs.iter().map(|o| o.ty.clone()).collect(),
        )
    };
    let ssa_body = convert_body(body, &param_names, return_ty)
        .map_err(|e| err_spirv!("SSA conversion failed: {}", e))?;
    let result = super::ssa_lowering::lower_ssa_body_for_entry(constructor, &ssa_body)?;

    // Store result to output variables
    if outputs.len() > 1 {
        // Multiple outputs - extract components from tuple result
        for (i, &output_var) in constructor.current_output_vars.clone().iter().enumerate() {
            let comp_type_id = constructor.polytype_to_spirv(&outputs[i].ty);
            let component =
                constructor.builder.composite_extract(comp_type_id, None, result, [i as u32])?;
            constructor.builder.store(output_var, component, None, [])?;
        }
    } else if let Some(&output_var) = constructor.current_output_vars.first() {
        // Single output
        constructor.builder.store(output_var, result, None, [])?;
    }

    // Return void
    constructor.builder.ret()?;

    // Terminate the variables block with a branch to the code block
    if let (Some(vars_block), Some(code_block)) =
        (constructor.variables_block, constructor.first_code_block)
    {
        // Find the variables block index and select it
        let func = constructor.builder.module_ref().functions.last().expect("No function");
        let vars_idx = func
            .blocks
            .iter()
            .position(|b| b.label.as_ref().map(|l| l.result_id) == Some(Some(vars_block)));

        if let Some(idx) = vars_idx {
            constructor.builder.select_block(Some(idx))?;
            constructor.builder.branch(code_block)?;
        }
    }

    constructor.builder.end_function()?;

    // Clean up
    constructor.current_is_entry_point = false;
    constructor.current_used_globals.clear();
    constructor.variables_block = None;
    constructor.first_code_block = None;
    constructor.env.clear();

    Ok(())
}

/// Lower a compute entry point.
///
/// The MIR is expected to have been transformed by soac_parallelize to use
/// __builtin_thread_id for parallelization. This function:
/// 1. Sets up storage buffers from the pipeline
/// 2. Creates GlobalInvocationId input (for __builtin_thread_id intrinsic)
/// 3. Lowers the body with regular expression lowering
/// 4. Writes the result to the output buffer at thread_id index
fn lower_compute_entry_point(
    constructor: &mut Constructor,
    pipeline: &Pipeline,
    outputs: &[mir::EntryOutput],
    body: &Body,
) -> Result<()> {
    // Set up the compute entry point
    constructor.current_is_entry_point = true;
    constructor.current_output_vars.clear();
    constructor.current_input_vars.clear();

    let mut interface_vars = Vec::new();

    // Create storage buffers from pipeline buffer blocks
    for buffer in &pipeline.buffers {
        let field = &buffer.fields[0]; // Single runtime-sized field for now
        let elem_type_id = constructor.polytype_to_spirv(&field.ty);

        // Calculate proper array stride from element type
        let stride = crate::mir::layout::type_byte_size(&field.ty)
            .expect("Storage buffer element type must have computable size");

        let runtime_array_type = constructor.get_or_create_runtime_array_type(elem_type_id, stride);
        let buffer_struct_type = constructor.get_or_create_buffer_block_type(runtime_array_type);

        let buffer_ptr_type =
            constructor.get_or_create_ptr_type(spirv::StorageClass::StorageBuffer, buffer_struct_type);
        let buffer_var =
            constructor.builder.variable(buffer_ptr_type, None, spirv::StorageClass::StorageBuffer, None);
        constructor.builder.decorate(
            buffer_var,
            spirv::Decoration::DescriptorSet,
            [Operand::LiteralBit32(buffer.set)],
        );
        constructor.builder.decorate(
            buffer_var,
            spirv::Decoration::Binding,
            [Operand::LiteralBit32(buffer.binding)],
        );
        interface_vars.push(buffer_var);

        // Register buffer for use when lowering StorageSlice accesses
        constructor.storage_buffers.insert(
            (buffer.set, buffer.binding),
            (buffer_var, elem_type_id, buffer_ptr_type),
        );

        // Also register by name so captured variables can be resolved
        constructor.uniform_variables.insert(buffer.name.clone(), buffer_var);
        constructor.uniform_types.insert(buffer.name.clone(), runtime_array_type);

        // Register as compute parameter for Local variable resolution
        constructor.compute_params.insert(buffer.name.clone(), (buffer.set, buffer.binding));
    }

    // Create GlobalInvocationId input variable
    let uvec3_type = constructor.get_or_create_vec_type(constructor.u32_type, 3);
    let gid_ptr_type = constructor.get_or_create_ptr_type(spirv::StorageClass::Input, uvec3_type);
    let global_id_var = constructor.builder.variable(gid_ptr_type, None, spirv::StorageClass::Input, None);
    constructor.builder.decorate(
        global_id_var,
        spirv::Decoration::BuiltIn,
        [Operand::BuiltIn(spirv::BuiltIn::GlobalInvocationId)],
    );
    interface_vars.push(global_id_var);

    // Store GlobalInvocationId var for __builtin_thread_id intrinsic
    constructor.global_invocation_id = Some(global_id_var);

    // Store interface variables for entry point declaration
    constructor.entry_point_interfaces.insert(pipeline.name.clone(), interface_vars);

    // Create void(void) function for entry point
    let func_type = constructor.builder.type_function(constructor.void_type, vec![]);
    let func_id = constructor.builder.begin_function(
        constructor.void_type,
        None,
        spirv::FunctionControl::NONE,
        func_type,
    )?;
    constructor.functions.insert(pipeline.name.clone(), func_id);

    // Create blocks
    let vars_block_id = constructor.builder.id();
    let code_block_id = constructor.builder.id();
    constructor.variables_block = Some(vars_block_id);
    constructor.first_code_block = Some(code_block_id);

    constructor.builder.begin_block(Some(vars_block_id))?;
    constructor.builder.select_block(None)?;
    constructor.builder.begin_block(Some(code_block_id))?;
    constructor.current_block = Some(code_block_id);

    // Convert to SSA and lower - the MIR has been transformed by soac_parallelize
    // to use __builtin_thread_id() for indexing
    let return_ty = if outputs.is_empty() {
        PolyType::Constructed(TypeName::Unit, vec![])
    } else {
        outputs[0].ty.clone()
    };
    let ssa_body =
        convert_body(body, &[], return_ty).map_err(|e| err_spirv!("SSA conversion failed: {}", e))?;
    let _result = super::ssa_lowering::lower_ssa_body_for_entry(constructor, &ssa_body)?;

    // Check if the body returns Unit - if so, map_into already wrote to output
    let body_ty = body.get_type(body.root);
    let body_returns_unit = matches!(body_ty, PolyType::Constructed(TypeName::Unit, _));

    if !body_returns_unit {
        // Body returns a value that needs to be written to output buffer
        let output_elem_type_id = if let Some(output) = outputs.first() {
            constructor.polytype_to_spirv(&output.ty)
        } else {
            // No output - just return
            constructor.builder.ret()?;
            finalize_compute_entry(constructor)?;
            return Ok(());
        };

        // Get thread_id for output write - load GlobalInvocationId.x
        let gid = constructor.builder.load(uvec3_type, None, global_id_var, None, [])?;
        let thread_id = constructor.builder.composite_extract(constructor.u32_type, None, gid, [0])?;

        // Find output buffer (last buffer in the pipeline)
        let num_buffers = pipeline.buffers.len() as u32;
        let output_buffer_binding = if num_buffers > 1 { num_buffers - 1 } else { 0 };

        let (output_buffer_var, output_mem) =
            if let Some(&(out_var, _, _)) = constructor.storage_buffers.get(&(0, output_buffer_binding)) {
                (
                    out_var,
                    MemBinding::Storage {
                        set: 0,
                        binding: output_buffer_binding,
                    },
                )
            } else if let Some(&(out_var, _, _)) = constructor.storage_buffers.get(&(0, 0)) {
                // Fallback to first buffer for in-place operations
                (out_var, MemBinding::Storage { set: 0, binding: 0 })
            } else {
                bail_spirv!("No output buffer available for compute shader")
            };

        // Write result to output buffer at thread_id index
        write_array_element(
            constructor,
            Some(output_mem),
            output_buffer_var,
            thread_id,
            _result,
            output_elem_type_id,
            constructor.void_type,
        )?;
    }
    // If body returns Unit, map_into already wrote to output - nothing more to do

    // Return
    constructor.builder.ret()?;

    finalize_compute_entry(constructor)?;

    Ok(())
}

/// Write a value to an array element (storage buffer write).
fn write_array_element(
    constructor: &mut Constructor,
    mem: Option<MemBinding>,
    _array_val: spirv::Word,
    index: spirv::Word,
    value: spirv::Word,
    elem_type: spirv::Word,
    _result_type: spirv::Word,
) -> Result<spirv::Word> {
    match mem {
        Some(MemBinding::Storage { set, binding }) => {
            let &(buffer_var, _, _) =
                constructor.storage_buffers.get(&(set, binding)).ok_or_else(|| {
                    err_spirv!("Storage buffer not found for set={}, binding={}", set, binding)
                })?;

            let elem_ptr_type =
                constructor.get_or_create_ptr_type(spirv::StorageClass::StorageBuffer, elem_type);
            let zero = constructor.const_u32(0);
            let elem_ptr =
                constructor.builder.access_chain(elem_ptr_type, None, buffer_var, [zero, index])?;
            constructor.builder.store(elem_ptr, value, None, [])?;
            Ok(constructor.const_i32(0)) // Return dummy value
        }
        None => {
            bail_spirv!("write_array_element: expected storage binding")
        }
    }
}

/// Finalize a compute entry point (terminate blocks, end function, cleanup).
fn finalize_compute_entry(constructor: &mut Constructor) -> Result<()> {
    // Terminate variables block
    if let (Some(vars_block), Some(code_block)) =
        (constructor.variables_block, constructor.first_code_block)
    {
        let func = constructor.builder.module_ref().functions.last().expect("No function");
        let vars_idx = func
            .blocks
            .iter()
            .position(|b| b.label.as_ref().map(|l| l.result_id) == Some(Some(vars_block)));

        if let Some(idx) = vars_idx {
            constructor.builder.select_block(Some(idx))?;
            constructor.builder.branch(code_block)?;
        }
    }

    constructor.builder.end_function()?;

    // Clean up
    constructor.current_is_entry_point = false;
    constructor.current_used_globals.clear();
    constructor.variables_block = None;
    constructor.first_code_block = None;
    constructor.env.clear();
    constructor.storage_buffers.clear();
    constructor.compute_params.clear();
    constructor.global_invocation_id = None;

    Ok(())
}

/// Lower a constant expression at compile time.
/// This is used for global constants which must be compile-time evaluable.
/// Currently supports: literals, references to other constants, and basic arithmetic.
fn lower_const_expr(constructor: &mut Constructor, body: &Body, expr_id: ExprId) -> Result<spirv::Word> {
    let ty = body.get_type(expr_id);
    match body.get_expr(expr_id) {
        Expr::Int(n) => match ty {
            types::Type::Constructed(TypeName::UInt(32), _) => {
                // Parse as i32 first to handle wrapped negative values (e.g., 0x80000000u32
                // is stored as -2147483648 in the AST because IntLiteral uses i32),
                // then reinterpret the bits as u32
                let val: u32 = n
                    .parse::<u32>()
                    .or_else(|_| n.parse::<i32>().map(|v| v as u32))
                    .map_err(|_| err_spirv!("Invalid u32 literal: {}", n))?;
                Ok(constructor.const_u32(val))
            }
            _ => {
                let val: i32 = n.parse().map_err(|_| err_spirv!("Invalid integer literal: {}", n))?;
                Ok(constructor.const_i32(val))
            }
        },
        Expr::Float(f) => {
            let val: f32 = f.parse().map_err(|_| err_spirv!("Invalid float literal: {}", f))?;
            Ok(constructor.const_f32(val))
        }
        Expr::Bool(b) => Ok(constructor.const_bool(*b)),
        Expr::Unit => Ok(constructor.const_i32(0)),
        Expr::Tuple(elems) => {
            let elem_ids: Result<Vec<_>> =
                elems.iter().map(|&id| lower_const_expr(constructor, body, id)).collect();
            let elem_ids = elem_ids?;
            let struct_type = constructor.polytype_to_spirv(ty);
            Ok(constructor.builder.constant_composite(struct_type, elem_ids))
        }
        Expr::Array { backing, .. } => match backing {
            ArrayBacking::Literal(elems) => {
                let elem_ids: Result<Vec<_>> =
                    elems.iter().map(|&id| lower_const_expr(constructor, body, id)).collect();
                let elem_ids = elem_ids?;
                let array_type = constructor.polytype_to_spirv(ty);
                Ok(constructor.builder.constant_composite(array_type, elem_ids))
            }
            _ => {
                bail_spirv!("Only literal arrays can be lowered as constants")
            }
        },
        Expr::Vector(elems) => {
            let elem_ids: Result<Vec<_>> =
                elems.iter().map(|&id| lower_const_expr(constructor, body, id)).collect();
            let elem_ids = elem_ids?;
            let array_type = constructor.polytype_to_spirv(ty);
            Ok(constructor.builder.constant_composite(array_type, elem_ids))
        }
        Expr::Matrix(rows) => {
            // Lower each row as a constant vector, then construct the constant matrix
            let row_ids: Vec<spirv::Word> = rows
                .iter()
                .map(|row| {
                    let elem_ids: Vec<spirv::Word> = row
                        .iter()
                        .map(|&e| lower_const_expr(constructor, body, e))
                        .collect::<Result<Vec<_>>>()?;
                    // Get element type from first element
                    let elem_ty = body.get_type(row[0]);
                    let elem_spirv_type = constructor.polytype_to_spirv(elem_ty);
                    let row_type = constructor.get_or_create_vec_type(elem_spirv_type, row.len() as u32);
                    Ok(constructor.builder.constant_composite(row_type, elem_ids))
                })
                .collect::<Result<Vec<_>>>()?;

            // Get the matrix type
            let result_type = constructor.polytype_to_spirv(ty);

            // Construct the constant matrix from row vectors
            Ok(constructor.builder.constant_composite(result_type, row_ids))
        }
        Expr::Global(name) => {
            // Reference to another global constant
            if let Some(&const_id) = constructor.global_constants.get(name) {
                return Ok(const_id);
            }
            // Uniforms cannot be used in constant expressions
            if constructor.uniform_variables.contains_key(name) {
                bail_spirv!(
                    "Uniform variable '{}' cannot be used in constant expressions",
                    name
                );
            }
            Err(err_spirv!(
                "Global constant references undefined constant '{}'",
                name
            ))
        }
        Expr::BinOp { op, .. } => {
            // For now, we don't support constant folding of binary ops
            Err(err_spirv!(
                "Global constants must be literals (found binary operation '{}'). \
                 Constant folding not yet implemented.",
                op
            ))
        }
        Expr::UnaryOp { op, .. } => Err(err_spirv!(
            "Global constants must be literals (found unary operation '{}')",
            op
        )),
        Expr::Attributed { expr, .. } => {
            // Attributes are metadata, just lower the inner expression
            lower_const_expr(constructor, body, *expr)
        }
        _ => Err(err_spirv!(
            "Global constants must be literals or compile-time foldable expressions"
        )),
    }
}
