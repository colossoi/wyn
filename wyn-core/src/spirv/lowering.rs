//! SPIR-V Lowering
//!
//! This module converts MIR (from flattening) directly to SPIR-V.
//! It uses a Constructor wrapper that handles variable hoisting automatically.
//! Dependencies are lowered on-demand using ensure_lowered pattern.

use crate::alias_checker::InPlaceInfo;
use crate::ast::TypeName;
use crate::error::Result;
use crate::impl_source::{BuiltinImpl, ImplSource, PrimOp};
use crate::lowering_common::is_empty_closure_type;
// Note: detect_simple_compute_map removed - soac_parallelize transforms the MIR instead
use crate::mir::{
    self, ArrayBacking, Body, Def, ExecutionModel, Expr, ExprId, LambdaId, LambdaInfo, LocalId, LoopKind,
    Program,
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
use rspirv::dr::Operand;
use rspirv::dr::{Builder, InsertPoint};
use rspirv::spirv::{self, AddressingModel, Capability, MemoryModel, StorageClass};
use std::collections::{HashMap, HashSet};

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
struct Constructor {
    builder: Builder,

    // Type caching
    void_type: spirv::Word,
    bool_type: spirv::Word,
    i32_type: spirv::Word,
    u32_type: spirv::Word,
    f32_type: spirv::Word,

    // Constant caching
    int_const_cache: HashMap<i32, spirv::Word>,
    uint_const_cache: HashMap<u32, spirv::Word>,
    uint_const_reverse: HashMap<spirv::Word, u32>, // reverse lookup: ID -> value
    float_const_cache: HashMap<u32, spirv::Word>,  // bits as u32
    bool_const_cache: HashMap<bool, spirv::Word>,

    // Current function state
    current_block: Option<spirv::Word>,
    variables_block: Option<spirv::Word>, // Block for OpVariable declarations
    first_code_block: Option<spirv::Word>, // First block with actual code

    // Environment: name -> value ID
    env: HashMap<String, spirv::Word>,

    // Function map: name -> function ID
    functions: HashMap<String, spirv::Word>,

    // GLSL extended instruction set
    glsl_ext_inst_id: spirv::Word,

    // Type cache: avoid recreating same types
    vec_type_cache: HashMap<(spirv::Word, u32), spirv::Word>,
    struct_type_cache: HashMap<Vec<spirv::Word>, spirv::Word>,
    ptr_type_cache: HashMap<(spirv::StorageClass, spirv::Word), spirv::Word>,
    runtime_array_cache: HashMap<(spirv::Word, u32), spirv::Word>, // (elem_type, stride) -> decorated type
    buffer_block_cache: HashMap<spirv::Word, spirv::Word>, // runtime_array_type -> Block-decorated struct

    // Entry point interface tracking
    entry_point_interfaces: HashMap<String, Vec<spirv::Word>>,
    current_is_entry_point: bool,
    current_output_vars: Vec<spirv::Word>,
    current_input_vars: Vec<(spirv::Word, String, spirv::Word)>, // (var_id, param_name, type_id)
    current_used_globals: Vec<spirv::Word>, // Global constants accessed in current entry point

    // Global constants: name -> constant_id (SPIR-V OpConstant)
    global_constants: HashMap<String, spirv::Word>,
    uniform_variables: HashMap<String, spirv::Word>,
    uniform_types: HashMap<String, spirv::Word>, // uniform name -> SPIR-V type ID
    uniform_load_cache: HashMap<String, spirv::Word>, // cached OpLoad results per function
    extract_cache: HashMap<(spirv::Word, u32), spirv::Word>, // CSE for OpCompositeExtract

    /// Lambda registry: LambdaId -> LambdaInfo
    lambda_registry: IdArena<LambdaId, LambdaInfo>,

    // Builtin function registry
    impl_source: ImplSource,

    /// In-place optimization: ExprIds of operations where input array can be reused
    inplace_nodes: HashSet<crate::mir::ExprId>,

    /// Storage buffers for compute shaders: (set, binding) -> (buffer_var, elem_type_id)
    storage_buffers: HashMap<(u32, u32), (spirv::Word, spirv::Word)>,

    /// Compute shader parameters: maps param name to (set, binding) for storage buffer lookup
    compute_params: HashMap<String, (u32, u32)>,

    /// GlobalInvocationId variable for compute shaders (set during entry point setup)
    global_invocation_id: Option<spirv::Word>,

    /// Linked SPIR-V functions: linkage_name -> function_id
    linked_functions: HashMap<String, spirv::Word>,

    /// Whether we need linkage capability
    needs_linkage: bool,
}

impl Constructor {
    fn new() -> Self {
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
            inplace_nodes: HashSet::new(),
            storage_buffers: HashMap::new(),
            compute_params: HashMap::new(),
            global_invocation_id: None,
            linked_functions: HashMap::new(),
            needs_linkage: false,
        }
    }

    /// Get or create a pointer type
    fn get_or_create_ptr_type(
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

    /// Get or declare a linked (imported) external function
    /// Creates an OpFunction with Import linkage decoration
    fn get_or_declare_linked_function(
        &mut self,
        linkage_name: &str,
        arg_types: &[spirv::Word],
        return_type: spirv::Word,
    ) -> Result<spirv::Word> {
        // Return cached function if already declared
        if let Some(&func_id) = self.linked_functions.get(linkage_name) {
            return Ok(func_id);
        }

        // Enable linkage capability if not already done
        if !self.needs_linkage {
            self.builder.capability(Capability::Linkage);
            self.needs_linkage = true;
        }

        // Create function type
        let func_type = self.builder.type_function(return_type, arg_types.to_vec());

        // Declare the function (empty body - will be linked)
        let func_id = self.builder.begin_function(
            return_type,
            None,
            spirv::FunctionControl::NONE,
            func_type,
        )?;

        // Add Import linkage decoration
        self.builder.decorate(
            func_id,
            spirv::Decoration::LinkageAttributes,
            [
                Operand::LiteralString(linkage_name.to_string()),
                Operand::LinkageType(spirv::LinkageType::Import),
            ],
        );

        // End the empty function body
        self.builder.end_function()?;

        // Cache and return
        self.linked_functions.insert(linkage_name.to_string(), func_id);
        Ok(func_id)
    }

    /// Convert a polytype Type to a SPIR-V type ID
    fn ast_type_to_spirv(&mut self, ty: &PolyType<TypeName>) -> spirv::Word {
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
                            args.iter().map(|a| self.ast_type_to_spirv(a)).collect();
                        self.get_or_create_struct_type(field_types)
                    }
                    TypeName::Array => {
                        // Array[elem, addrspace, size]
                        assert!(args.len() == 3);
                        let elem_type = self.ast_type_to_spirv(&args[0]);
                        let addrspace = &args[1];
                        let size = &args[2];

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
                                // Unsized array (size is unresolved type variable) - representation depends on address space
                                if let PolyType::Constructed(TypeName::AddressStorage, _) = addrspace {
                                    // Storage array: struct { pointer, length }
                                    let ptr_type = self.get_or_create_ptr_type(
                                        spirv::StorageClass::StorageBuffer,
                                        elem_type,
                                    );
                                    self.get_or_create_struct_type(vec![ptr_type, self.i32_type])
                                } else if let PolyType::Constructed(TypeName::AddressFunction, _) =
                                    addrspace
                                {
                                    // Function-local unsized array
                                    panic!(
                                        "BUG: Function address space unsized arrays not yet implemented: {:?}",
                                        ty
                                    );
                                } else if let PolyType::Variable(_) = addrspace {
                                    // Unresolved address space - treat as storage for now
                                    let ptr_type = self.get_or_create_ptr_type(
                                        spirv::StorageClass::StorageBuffer,
                                        elem_type,
                                    );
                                    self.get_or_create_struct_type(vec![ptr_type, self.i32_type])
                                } else {
                                    // Unknown address space
                                    panic!("BUG: Unsized array has unexpected address space: {:?}", ty);
                                }
                            }
                            _ => {
                                panic!(
                                    "BUG: Array type has invalid size argument: {:?}. This should have been resolved during type checking.",
                                    size
                                );
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
                        let elem_type = self.ast_type_to_spirv(&args[1]);
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
                        let elem_type = self.ast_type_to_spirv(&args[2]);
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
                        let pointee_type = self.ast_type_to_spirv(&args[0]);
                        self.builder.type_pointer(None, StorageClass::Function, pointee_type)
                    }
                    TypeName::Unique => {
                        // Unique type wrapper: strip and convert underlying type
                        // Unique is only used for alias checking, has no runtime representation
                        if args.is_empty() {
                            panic!("BUG: Unique type requires an underlying type argument.");
                        }
                        self.ast_type_to_spirv(&args[0])
                    }
                    TypeName::Existential(_) => {
                        // Existential type: unwrap and convert the inner type (in args[0])
                        // The size variable is runtime-determined, handled by Slice representation
                        let inner = &args[0];
                        self.ast_type_to_spirv(inner)
                    }
                    TypeName::Arrow => {
                        // Arrow types (function types) come from closures that have been defunctionalized.
                        // Since closures are represented as (captures_tuple, lambda_name), the actual
                        // runtime value is just the captures. The Arrow type is a phantom type used
                        // for type checking only. Map to unit type since it has no runtime representation.
                        self.void_type
                    }
                    TypeName::AddressFunction | TypeName::AddressStorage => {
                        // Address space markers are used within Array types but shouldn't appear
                        // as standalone types requiring SPIR-V representation.
                        panic!(
                            "BUG: Address space marker {:?} reached ast_type_to_spirv as standalone type. \
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
    fn get_or_create_vec_type(&mut self, elem_type: spirv::Word, size: u32) -> spirv::Word {
        let key = (elem_type, size);
        if let Some(&ty) = self.vec_type_cache.get(&key) {
            return ty;
        }
        let ty = self.builder.type_vector(elem_type, size);
        self.vec_type_cache.insert(key, ty);
        ty
    }

    /// Get or create a struct type
    fn get_or_create_struct_type(&mut self, field_types: Vec<spirv::Word>) -> spirv::Word {
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
    fn create_uniform_block_type(&mut self, value_type: spirv::Word) -> spirv::Word {
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
    fn begin_function(
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
    fn declare_variable(&mut self, _name: &str, value_type: spirv::Word) -> Result<spirv::Word> {
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
    fn const_i32(&mut self, value: i32) -> spirv::Word {
        if let Some(&id) = self.int_const_cache.get(&value) {
            return id;
        }
        let id = self.builder.constant_bit32(self.i32_type, value as u32);
        self.int_const_cache.insert(value, id);
        id
    }

    /// Get or create a u32 constant
    fn const_u32(&mut self, value: u32) -> spirv::Word {
        if let Some(&id) = self.uint_const_cache.get(&value) {
            return id;
        }
        let id = self.builder.constant_bit32(self.u32_type, value);
        self.uint_const_cache.insert(value, id);
        self.uint_const_reverse.insert(id, value);
        id
    }

    /// Get the literal u32 value from a constant ID (reverse lookup)
    fn get_const_u32_value(&self, id: spirv::Word) -> Option<u32> {
        self.uint_const_reverse.get(&id).copied()
    }

    /// Get or create an f32 constant
    fn const_f32(&mut self, value: f32) -> spirv::Word {
        let bits = value.to_bits();
        if let Some(&id) = self.float_const_cache.get(&bits) {
            return id;
        }
        let id = self.builder.constant_bit32(self.f32_type, bits);
        self.float_const_cache.insert(bits, id);
        id
    }

    /// Get or create a bool constant
    fn const_bool(&mut self, value: bool) -> spirv::Word {
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

    /// CSE-cached composite extract - reuses result if same (source, index) was extracted before
    fn composite_extract_cached(
        &mut self,
        result_type: spirv::Word,
        composite: spirv::Word,
        index: u32,
    ) -> Result<spirv::Word> {
        let key = (composite, index);
        if let Some(&id) = self.extract_cache.get(&key) {
            return Ok(id);
        }
        let id = self.builder.composite_extract(result_type, None, composite, [index])?;
        self.extract_cache.insert(key, id);
        Ok(id)
    }

    /// Begin a block (must be called before emitting instructions into it)
    fn begin_block(&mut self, block_id: spirv::Word) -> Result<()> {
        self.builder.begin_block(Some(block_id))?;
        self.current_block = Some(block_id);
        // Clear extract cache since values from previous blocks may not dominate this block
        self.extract_cache.clear();
        Ok(())
    }

    /// Emit a conditional branch with selection merge
    fn branch_conditional(
        &mut self,
        cond: spirv::Word,
        true_block: spirv::Word,
        false_block: spirv::Word,
        merge_block: spirv::Word,
    ) -> Result<()> {
        self.builder.selection_merge(merge_block, spirv::SelectionControl::NONE)?;
        self.builder.branch_conditional(cond, true_block, false_block, [])?;
        Ok(())
    }

    /// Get array type
    fn type_array(&mut self, elem_type: spirv::Word, length: u32) -> spirv::Word {
        let length_id = self.const_i32(length as i32);
        self.builder.type_array(elem_type, length_id)
    }
}

impl<'a> LowerCtx<'a> {
    fn new(program: &'a Program, inplace_info: &InPlaceInfo) -> Self {
        let mut constructor = Constructor::new();
        constructor.lambda_registry = program.lambda_registry.clone();
        constructor.inplace_nodes = inplace_info.can_reuse_input.clone();

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
                ..
            } => {
                // First, ensure all dependencies are lowered
                self.ensure_deps_lowered(body)?;

                // Regular function (entry points are now Def::EntryPoint)
                lower_regular_function(&mut self.constructor, name, params, ret_type, body)?;
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
                let value_type = self.constructor.ast_type_to_spirv(ty);
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
                let storage_type = self.constructor.ast_type_to_spirv(ty);
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
pub fn lower(program: &mir::Program, inplace_info: &InPlaceInfo) -> Result<Vec<u32>> {
    // Use a thread with larger stack size to handle deeply nested expressions
    // Default Rust stack is 2MB on macOS which is too small for complex shaders
    const STACK_SIZE: usize = 16 * 1024 * 1024; // 16MB

    // Clone program and inplace info since we need 'static lifetime for thread
    let program_clone = program.clone();
    let inplace_info_clone = inplace_info.clone();

    let handle = std::thread::Builder::new()
        .stack_size(STACK_SIZE)
        .spawn(move || {
            let ctx = LowerCtx::new(&program_clone, &inplace_info_clone);
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
        params_to_lower.iter().map(|&p| constructor.ast_type_to_spirv(&body.get_local(p).ty)).collect();
    let return_type = constructor.ast_type_to_spirv(ret_type);
    constructor.begin_function(name, &param_names, &param_types, return_type)?;

    let result = lower_expr(constructor, body, body.root)?;

    // Use ret() for void functions, ret_value() for functions that return a value
    if matches!(ret_type, PolyType::Constructed(TypeName::Unit, _)) {
        constructor.builder.ret()?;
    } else {
        constructor.builder.ret_value(result)?;
    }

    constructor.end_function()?;
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
        let input_type_id = constructor.ast_type_to_spirv(&input.ty);
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

        let output_type_id = constructor.ast_type_to_spirv(&output.ty);
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

    // Lower the body
    let result = lower_expr(constructor, body, body.root)?;

    // Store result to output variables
    if outputs.len() > 1 {
        // Multiple outputs - extract components from tuple result
        for (i, &output_var) in constructor.current_output_vars.clone().iter().enumerate() {
            let comp_type_id = constructor.ast_type_to_spirv(&outputs[i].ty);
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
        let elem_type_id = constructor.ast_type_to_spirv(&field.ty);

        // Calculate proper array stride from element type
        let stride = crate::mir::layout::type_byte_size(&field.ty).unwrap_or(4);

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
        constructor.storage_buffers.insert((buffer.set, buffer.binding), (buffer_var, elem_type_id));

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

    // Lower the body - the MIR has been transformed by soac_parallelize to use
    // __builtin_thread_id() for indexing, so regular expression lowering works
    let _result = lower_expr(constructor, body, body.root)?;

    // Check if the body returns Unit - if so, map_into already wrote to output
    let body_ty = body.get_type(body.root);
    let body_returns_unit = matches!(body_ty, PolyType::Constructed(TypeName::Unit, _));

    if !body_returns_unit {
        // Body returns a value that needs to be written to output buffer
        let output_elem_type_id = if let Some(output) = outputs.first() {
            constructor.ast_type_to_spirv(&output.ty)
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
            if let Some((out_var, _)) = constructor.storage_buffers.get(&(0, output_buffer_binding)) {
                (
                    *out_var,
                    MemBinding::Storage {
                        set: 0,
                        binding: output_buffer_binding,
                    },
                )
            } else if let Some((out_var, _)) = constructor.storage_buffers.get(&(0, 0)) {
                // Fallback to first buffer for in-place operations
                (*out_var, MemBinding::Storage { set: 0, binding: 0 })
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
            let struct_type = constructor.ast_type_to_spirv(ty);
            Ok(constructor.builder.constant_composite(struct_type, elem_ids))
        }
        Expr::Array { backing, .. } => match backing {
            ArrayBacking::Literal(elems) => {
                let elem_ids: Result<Vec<_>> =
                    elems.iter().map(|&id| lower_const_expr(constructor, body, id)).collect();
                let elem_ids = elem_ids?;
                let array_type = constructor.ast_type_to_spirv(ty);
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
            let array_type = constructor.ast_type_to_spirv(ty);
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
                    let elem_spirv_type = constructor.ast_type_to_spirv(elem_ty);
                    let row_type = constructor.get_or_create_vec_type(elem_spirv_type, row.len() as u32);
                    Ok(constructor.builder.constant_composite(row_type, elem_ids))
                })
                .collect::<Result<Vec<_>>>()?;

            // Get the matrix type
            let result_type = constructor.ast_type_to_spirv(ty);

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

fn lower_expr(constructor: &mut Constructor, body: &Body, expr_id: ExprId) -> Result<spirv::Word> {
    let expr_ty = body.get_type(expr_id);

    // Debug: check for address space types being used as expression types
    if let PolyType::Constructed(TypeName::AddressFunction | TypeName::AddressStorage, _) = expr_ty {
        panic!(
            "BUG: Expression {:?} has address space type {:?}. Expression: {:?}",
            expr_id,
            expr_ty,
            body.get_expr(expr_id)
        );
    }

    match body.get_expr(expr_id) {
        Expr::Int(s) => match expr_ty {
            PolyType::Constructed(TypeName::UInt(32), _) => {
                let val: u32 = s.parse().map_err(|_| err_spirv!("Invalid u32 literal: {}", s))?;
                Ok(constructor.const_u32(val))
            }
            _ => {
                let val: i32 = s.parse().map_err(|_| err_spirv!("Invalid integer literal: {}", s))?;
                Ok(constructor.const_i32(val))
            }
        },

        Expr::Float(s) => {
            let val: f32 = s.parse().map_err(|_| err_spirv!("Invalid float literal: {}", s))?;
            Ok(constructor.const_f32(val))
        }

        Expr::Bool(b) => Ok(constructor.const_bool(*b)),

        Expr::Unit => Ok(constructor.const_i32(0)),

        Expr::String(s) => {
            // Strings are not really supported in SPIR-V, return a placeholder
            Err(err_spirv!("String literals not supported in SPIR-V: {}", s))
        }

        Expr::Local(local_id) => {
            // Look up the local variable name and find it in the environment
            let name = &body.get_local(*local_id).name;
            if let Some(&id) = constructor.env.get(name) {
                return Ok(id);
            }

            // Check if this is a compute shader parameter (storage buffer)
            if let Some(&(set, binding)) = constructor.compute_params.get(name) {
                // This is a storage buffer parameter - return a StorageSlice-like reference
                // The actual indexing will be handled by lower_index_intrinsic
                if let Some(&(buffer_var, _elem_type_id)) = constructor.storage_buffers.get(&(set, binding))
                {
                    // Return the buffer variable - indexing operations will use storage_buffers to access it
                    return Ok(buffer_var);
                }
            }

            let env_keys: Vec<_> = constructor.env.keys().collect();
            Err(err_spirv!(
                "Undefined local variable: {} (local_id={:?}, env has: {:?})",
                name,
                local_id,
                env_keys
            ))
        }

        Expr::Global(name) => {
            // First check global constants (these are now OpConstants, not variables)
            if let Some(&const_id) = constructor.global_constants.get(name) {
                return Ok(const_id);
            }
            // Check if it's a uniform variable
            if let Some(&var_id) = constructor.uniform_variables.get(name) {
                // Check cache first to avoid redundant OpLoads
                if let Some(&cached_id) = constructor.uniform_load_cache.get(name) {
                    return Ok(cached_id);
                }
                // Uniform is wrapped in a Block struct - need OpAccessChain to get member 0
                let value_type_id = constructor
                    .uniform_types
                    .get(name)
                    .copied()
                    .ok_or_else(|| err_spirv!("Could not find type for uniform variable: {}", name))?;

                // Create pointer type for the member access
                let member_ptr_type =
                    constructor.builder.type_pointer(None, spirv::StorageClass::Uniform, value_type_id);

                // Access member 0 of the Block struct
                let zero = constructor.const_i32(0);
                let member_ptr = constructor.builder.access_chain(member_ptr_type, None, var_id, [zero])?;

                // Load the value
                let load_id = constructor.builder.load(value_type_id, None, member_ptr, None, [])?;
                constructor.uniform_load_cache.insert(name.to_string(), load_id);
                return Ok(load_id);
            }
            Err(err_spirv!("Undefined global: {}", name))
        }

        Expr::BinOp { op, lhs, rhs } => {
            let lhs_id = lower_expr(constructor, body, *lhs)?;
            let rhs_id = lower_expr(constructor, body, *rhs)?;
            let lhs_ty = body.get_type(*lhs);
            let same_out_type = constructor.ast_type_to_spirv(lhs_ty);
            let bool_type = constructor.bool_type;

            use PolyType::*;
            use TypeName::*;
            match (op.as_str(), lhs_ty) {
                // Float operations
                ("+", Constructed(Float(_), _)) => {
                    Ok(constructor.builder.f_add(same_out_type, None, lhs_id, rhs_id)?)
                }
                ("-", Constructed(Float(_), _)) => {
                    Ok(constructor.builder.f_sub(same_out_type, None, lhs_id, rhs_id)?)
                }
                ("*", Constructed(Float(_), _)) => {
                    Ok(constructor.builder.f_mul(same_out_type, None, lhs_id, rhs_id)?)
                }
                ("/", Constructed(Float(_), _)) => {
                    Ok(constructor.builder.f_div(same_out_type, None, lhs_id, rhs_id)?)
                }
                ("%", Constructed(Float(_), _)) => {
                    Ok(constructor.builder.f_rem(same_out_type, None, lhs_id, rhs_id)?)
                }
                ("==", Constructed(Float(_), _)) => {
                    Ok(constructor.builder.f_ord_equal(bool_type, None, lhs_id, rhs_id)?)
                }
                ("!=", Constructed(Float(_), _)) => {
                    Ok(constructor.builder.f_ord_not_equal(bool_type, None, lhs_id, rhs_id)?)
                }
                ("<", Constructed(Float(_), _)) => {
                    Ok(constructor.builder.f_ord_less_than(bool_type, None, lhs_id, rhs_id)?)
                }
                ("<=", Constructed(Float(_), _)) => {
                    Ok(constructor.builder.f_ord_less_than_equal(bool_type, None, lhs_id, rhs_id)?)
                }
                (">", Constructed(Float(_), _)) => {
                    Ok(constructor.builder.f_ord_greater_than(bool_type, None, lhs_id, rhs_id)?)
                }
                (">=", Constructed(Float(_), _)) => {
                    Ok(constructor.builder.f_ord_greater_than_equal(bool_type, None, lhs_id, rhs_id)?)
                }
                ("**", Constructed(Float(_), _)) => {
                    // Power operator - use GLSL pow extended instruction
                    let glsl_id = constructor.glsl_ext_inst_id;
                    Ok(constructor.builder.ext_inst(
                        same_out_type,
                        None,
                        glsl_id,
                        26, // Pow
                        vec![Operand::IdRef(lhs_id), Operand::IdRef(rhs_id)],
                    )?)
                }

                // Unsigned integer operations
                ("/", Constructed(UInt(_), _)) => {
                    Ok(constructor.builder.u_div(same_out_type, None, lhs_id, rhs_id)?)
                }
                ("%", Constructed(UInt(_), _)) => {
                    Ok(constructor.builder.u_mod(same_out_type, None, lhs_id, rhs_id)?)
                }
                ("<", Constructed(UInt(_), _)) => {
                    Ok(constructor.builder.u_less_than(bool_type, None, lhs_id, rhs_id)?)
                }
                ("<=", Constructed(UInt(_), _)) => {
                    Ok(constructor.builder.u_less_than_equal(bool_type, None, lhs_id, rhs_id)?)
                }
                (">", Constructed(UInt(_), _)) => {
                    Ok(constructor.builder.u_greater_than(bool_type, None, lhs_id, rhs_id)?)
                }
                (">=", Constructed(UInt(_), _)) => {
                    Ok(constructor.builder.u_greater_than_equal(bool_type, None, lhs_id, rhs_id)?)
                }

                // Signed integer operations (and fallback for +, -, *, ==, != which are the same for signed/unsigned)
                ("+", _) => Ok(constructor.builder.i_add(same_out_type, None, lhs_id, rhs_id)?),
                ("-", _) => Ok(constructor.builder.i_sub(same_out_type, None, lhs_id, rhs_id)?),
                ("*", _) => Ok(constructor.builder.i_mul(same_out_type, None, lhs_id, rhs_id)?),
                ("/", _) => Ok(constructor.builder.s_div(same_out_type, None, lhs_id, rhs_id)?),
                ("%", _) => Ok(constructor.builder.s_mod(same_out_type, None, lhs_id, rhs_id)?),
                ("==", _) => Ok(constructor.builder.i_equal(bool_type, None, lhs_id, rhs_id)?),
                ("!=", _) => Ok(constructor.builder.i_not_equal(bool_type, None, lhs_id, rhs_id)?),
                ("<", _) => Ok(constructor.builder.s_less_than(bool_type, None, lhs_id, rhs_id)?),
                ("<=", _) => Ok(constructor.builder.s_less_than_equal(bool_type, None, lhs_id, rhs_id)?),
                (">", _) => Ok(constructor.builder.s_greater_than(bool_type, None, lhs_id, rhs_id)?),
                (">=", _) => {
                    Ok(constructor.builder.s_greater_than_equal(bool_type, None, lhs_id, rhs_id)?)
                }

                // Logical operations (boolean)
                ("&&", _) => Ok(constructor.builder.logical_and(bool_type, None, lhs_id, rhs_id)?),
                ("||", _) => Ok(constructor.builder.logical_or(bool_type, None, lhs_id, rhs_id)?),

                _ => Err(err_spirv!("Unknown binary op: {}", op)),
            }
        }

        Expr::UnaryOp { op, operand } => {
            let operand_id = lower_expr(constructor, body, *operand)?;
            let operand_ty = body.get_type(*operand);
            let same_type = constructor.ast_type_to_spirv(operand_ty);

            use PolyType::*;
            use TypeName::*;
            match (op.as_str(), operand_ty) {
                ("-", Constructed(Float(_), _)) => {
                    Ok(constructor.builder.f_negate(same_type, None, operand_id)?)
                }
                ("-", Constructed(UInt(bits), _)) => {
                    Err(err_spirv!("Cannot negate unsigned integer type u{}", bits))
                }
                ("-", _) => Ok(constructor.builder.s_negate(same_type, None, operand_id)?),
                ("!", _) => Ok(constructor.builder.logical_not(constructor.bool_type, None, operand_id)?),
                _ => Err(err_spirv!("Unknown unary op: {}", op)),
            }
        }

        Expr::If { cond, then_, else_ } => {
            let cond_id = lower_expr(constructor, body, *cond)?;

            // Get the result type from the expression
            let result_type = constructor.ast_type_to_spirv(expr_ty);

            // Create blocks
            let then_block_id = constructor.builder.id();
            let else_block_id = constructor.builder.id();
            let merge_block_id = constructor.builder.id();

            // Branch based on condition
            constructor.branch_conditional(cond_id, then_block_id, else_block_id, merge_block_id)?;

            // Then block
            constructor.begin_block(then_block_id)?;
            let then_result = lower_expr(constructor, body, *then_)?;
            let then_exit_block = constructor.current_block.unwrap();

            constructor.builder.branch(merge_block_id)?;

            // Else block
            constructor.begin_block(else_block_id)?;
            let else_result = lower_expr(constructor, body, *else_)?;
            let else_exit_block = constructor.current_block.unwrap();
            constructor.builder.branch(merge_block_id)?;

            // Merge block with phi
            constructor.begin_block(merge_block_id)?;

            // If result is unit type, no phi needed - unit can only be assigned to _
            if matches!(expr_ty, PolyType::Constructed(TypeName::Unit, _)) {
                // Return a dummy value - it will never be used since unit can only bind to _
                Ok(constructor.const_i32(0))
            } else {
                let incoming = vec![(then_result, then_exit_block), (else_result, else_exit_block)];
                let result = constructor.builder.phi(result_type, None, incoming)?;
                Ok(result)
            }
        }

        Expr::Let {
            local,
            rhs,
            body: let_body,
        } => {
            let name = &body.get_local(*local).name;
            // If binding to _, evaluate value for side effects but don't store it
            if name == "_" {
                let _ = lower_expr(constructor, body, *rhs)?;
                lower_expr(constructor, body, *let_body)
            } else {
                let value_id = lower_expr(constructor, body, *rhs)?;
                constructor.env.insert(name.clone(), value_id);
                let result = lower_expr(constructor, body, *let_body)?;
                constructor.env.remove(name);
                Ok(result)
            }
        }

        Expr::Loop {
            loop_var,
            init,
            init_bindings,
            kind,
            body: loop_body,
        } => {
            // Create blocks for loop structure
            let header_block_id = constructor.builder.id();
            let body_block_id = constructor.builder.id();
            let continue_block_id = constructor.builder.id();
            let merge_block_id = constructor.builder.id();

            // Evaluate the init expression for loop_var
            let init_val = lower_expr(constructor, body, *init)?;
            let init_ty = body.get_type(*init);
            let loop_var_type = constructor.ast_type_to_spirv(init_ty);

            // For For loops, lower the iterable and get its length in pre-header
            let for_loop_preheader = if let LoopKind::For { iter, .. } = kind {
                let iter_val = lower_expr(constructor, body, *iter)?;
                let length = lower_length_intrinsic(constructor, body, *iter, iter_val)?;
                Some((iter_val, length))
            } else {
                None
            };

            let pre_header_block = constructor.current_block.unwrap();

            // Check for unit type loop accumulator - this is an error
            // Loops must accumulate a value, not just perform side effects
            if matches!(init_ty, PolyType::Constructed(TypeName::Unit, _)) {
                bail_spirv!(
                    "Loop accumulator cannot be unit type (). \
                     Loops must return a value - use an accumulator pattern like: \
                     loop (acc1, acc2) = (init1, init2) for i < n do (new_acc1, new_acc2)"
                );
            }

            // Branch to header
            constructor.builder.branch(header_block_id)?;

            // Header block - we'll add phi node later
            constructor.begin_block(header_block_id)?;
            let header_block_idx = constructor.builder.selected_block().expect("No block selected");

            // Allocate phi ID for loop_var
            let loop_var_name = body.get_local(*loop_var).name.clone();
            let loop_var_phi_id = constructor.builder.id();
            constructor.env.insert(loop_var_name.clone(), loop_var_phi_id);

            // For ForRange and For loops, create a phi for the iteration index
            let iter_var_phi = match kind {
                LoopKind::ForRange { var, .. } => {
                    let var_name = body.get_local(*var).name.clone();
                    let iter_phi_id = constructor.builder.id();
                    constructor.env.insert(var_name.clone(), iter_phi_id);
                    Some((var_name, iter_phi_id))
                }
                LoopKind::For { .. } => {
                    // For-in loops need an internal index variable
                    let iter_phi_id = constructor.builder.id();
                    // Use a generated name that won't conflict
                    let var_name = "_for_idx".to_string();
                    constructor.env.insert(var_name.clone(), iter_phi_id);
                    Some((var_name, iter_phi_id))
                }
                _ => None,
            };

            // Evaluate init_bindings to bind user variables from loop_var
            // These extractions reference loop_var which is now bound to the phi
            for (local_id, binding_expr_id) in init_bindings.iter() {
                let val = lower_expr(constructor, body, *binding_expr_id)?;
                let binding_name = body.get_local(*local_id).name.clone();
                constructor.env.insert(binding_name, val);
            }

            // For For loops, we need to bind the element variable to arr[index] in the header
            // This happens before init_bindings are evaluated
            let for_loop_iter_val = if let LoopKind::For { var, iter } = kind {
                // Lower the iterable (done once, before loop header)
                // Actually we need to lower it before the loop starts...
                // For now, lower it here in header and it will be used each iteration
                let iter_val = lower_expr(constructor, body, *iter)?;

                // Get the index variable
                let idx_id = *constructor
                    .env
                    .get("_for_idx")
                    .ok_or_else(|| err_spirv!("For loop index not found"))?;

                // Index into the array: arr[idx]
                // Array[elem, addrspace, size] - elem is at args[0]
                let iter_ty = body.get_type(*iter);
                let elem_ty = match iter_ty {
                    PolyType::Constructed(TypeName::Array, args) => {
                        assert!(args.len() == 3);
                        &args[0]
                    }
                    _ => bail_spirv!("For-in loop over non-array type: {:?}", iter_ty),
                };
                let elem_spirv_ty = constructor.ast_type_to_spirv(elem_ty);

                // For unsized arrays, we need to handle differently
                let elem_val = match iter_ty {
                    PolyType::Constructed(TypeName::Array, args) => {
                        assert!(args.len() == 3);
                        // Check if unsized (type variable means runtime size)
                        if matches!(&args[2], PolyType::Variable(_)) {
                            // Unsized array (slice) - this shouldn't happen in for-in directly
                            bail_spirv!("For-in loop over unsized array not supported: {:?}", iter_ty);
                        } else {
                            // Sized array: direct vector extract
                            constructor.builder.vector_extract_dynamic(
                                elem_spirv_ty,
                                None,
                                iter_val,
                                idx_id,
                            )?
                        }
                    }
                    _ => bail_spirv!("For-in loop over non-array type: {:?}", iter_ty),
                };

                // Bind the element to the loop variable
                let var_name = body.get_local(*var).name.clone();
                constructor.env.insert(var_name.clone(), elem_val);

                Some((iter_val, var_name))
            } else {
                None
            };

            // Generate condition based on loop kind
            let cond_id = match kind {
                LoopKind::While { cond } => lower_expr(constructor, body, *cond)?,
                LoopKind::ForRange { var, bound } => {
                    let bound_id = lower_expr(constructor, body, *bound)?;
                    let var_name = &body.get_local(*var).name;
                    let var_id = *constructor
                        .env
                        .get(var_name)
                        .ok_or_else(|| err_spirv!("Loop variable {} not found", var_name))?;
                    constructor.builder.s_less_than(constructor.bool_type, None, var_id, bound_id)?
                }
                LoopKind::For { .. } => {
                    // Use pre-computed length from pre-header
                    let (_, length) = for_loop_preheader
                        .ok_or_else(|| err_spirv!("For loop preheader not initialized"))?;
                    let idx_id = *constructor
                        .env
                        .get("_for_idx")
                        .ok_or_else(|| err_spirv!("For loop index not found"))?;
                    constructor.builder.s_less_than(constructor.bool_type, None, idx_id, length)?
                }
            };

            // Loop merge and conditional branch
            constructor.builder.loop_merge(
                merge_block_id,
                continue_block_id,
                spirv::LoopControl::NONE,
                [],
            )?;
            constructor.builder.branch_conditional(cond_id, body_block_id, merge_block_id, [])?;

            // Body block
            constructor.begin_block(body_block_id)?;
            let body_result = lower_expr(constructor, body, *loop_body)?;
            constructor.builder.branch(continue_block_id)?;

            // Continue block - body_result is the new value for loop_var
            constructor.begin_block(continue_block_id)?;

            // For ForRange loops, increment the iteration variable
            let iter_next_val = if let Some((ref var_name, _)) = iter_var_phi {
                let var_id = *constructor.env.get(var_name).unwrap();
                let one = constructor.const_i32(1);
                let next_val = constructor.builder.i_add(constructor.i32_type, None, var_id, one)?;
                Some(next_val)
            } else {
                None
            };

            constructor.builder.branch(header_block_id)?;

            // Now go back and insert phi node at the beginning of header block
            constructor.builder.select_block(Some(header_block_idx))?;
            let incoming = vec![(init_val, pre_header_block), (body_result, continue_block_id)];
            constructor.builder.insert_phi(
                InsertPoint::Begin,
                loop_var_type,
                Some(loop_var_phi_id),
                incoming,
            )?;

            // Insert phi for iteration variable if ForRange
            if let Some((_, iter_phi_id)) = iter_var_phi {
                let zero = constructor.const_i32(0);
                let iter_next = iter_next_val.unwrap();
                let iter_incoming = vec![(zero, pre_header_block), (iter_next, continue_block_id)];
                constructor.builder.insert_phi(
                    InsertPoint::Begin,
                    constructor.i32_type,
                    Some(iter_phi_id),
                    iter_incoming,
                )?;
            }

            // Deselect block before continuing
            constructor.builder.select_block(None)?;

            // Continue to merge block
            constructor.begin_block(merge_block_id)?;

            // Clean up environment
            constructor.env.remove(&loop_var_name);
            for (local_id, _) in init_bindings.iter() {
                let binding_name = &body.get_local(*local_id).name;
                constructor.env.remove(binding_name);
            }
            if let Some((ref var_name, _)) = iter_var_phi {
                constructor.env.remove(var_name);
            }
            // Clean up For loop element variable
            if let Some((_, ref elem_var_name)) = for_loop_iter_val {
                constructor.env.remove(elem_var_name);
            }

            // Return the loop_var phi value as loop result
            Ok(loop_var_phi_id)
        }

        Expr::Call { func, args } => {
            // Get the result type from the expression
            let result_type = constructor.ast_type_to_spirv(expr_ty);

            // Special case for _w_array_with - check for in-place optimization
            if func == "_w_array_with" {
                if args.len() != 3 {
                    bail_spirv!("_w_array_with requires 3 args (array, index, value)");
                }

                // Lower arguments
                let arr_id = lower_expr(constructor, body, args[0])?;
                let idx_id = lower_expr(constructor, body, args[1])?;
                let val_id = lower_expr(constructor, body, args[2])?;

                // Check if we can do in-place update
                let can_inplace = constructor.inplace_nodes.contains(&expr_id);

                if can_inplace {
                    // In-place optimization: use OpCompositeInsert
                    // This creates a new SSA value but signals to the optimizer
                    // that the input array can potentially be reused
                    return Ok(constructor.builder.composite_insert(
                        result_type,
                        None,
                        val_id,
                        arr_id,
                        [idx_id],
                    )?);
                } else {
                    // Non-in-place: use copy-modify-load pattern
                    let arr_var = constructor.declare_variable("_w_array_with_tmp", result_type)?;
                    constructor.builder.store(arr_var, arr_id, None, [])?;

                    // Get pointer to element and store new value
                    let arg2_ty = body.get_type(args[2]);
                    let elem_type = constructor.ast_type_to_spirv(arg2_ty);
                    let elem_ptr_type =
                        constructor.builder.type_pointer(None, StorageClass::Function, elem_type);
                    let elem_ptr =
                        constructor.builder.access_chain(elem_ptr_type, None, arr_var, [idx_id])?;
                    constructor.builder.store(elem_ptr, val_id, None, [])?;

                    // Load and return the updated array
                    return Ok(constructor.builder.load(result_type, None, arr_var, None, [])?);
                }
            }

            // For all other calls, lower arguments normally
            let arg_ids: Vec<spirv::Word> =
                args.iter().map(|&a| lower_expr(constructor, body, a)).collect::<Result<Vec<_>>>()?;

            // Collect argument types for linked function declarations
            let arg_types: Vec<spirv::Word> = args
                .iter()
                .map(|&a| constructor.ast_type_to_spirv(body.get_type(a)))
                .collect();

            // Check for builtin vector constructors
            match func.as_str() {
                "vec2" | "vec3" | "vec4" => {
                    // Use the result type which should be the proper vector type
                    Ok(constructor.builder.composite_construct(result_type, None, arg_ids)?)
                }
                _ => {
                    // Check if it's a builtin function
                    if let Some(builtin_impl) = constructor.impl_source.get(func) {
                        match builtin_impl {
                            BuiltinImpl::PrimOp(spirv_op) => {
                                // Handle core SPIR-V operations
                                match spirv_op {
                                    PrimOp::GlslExt(ext_op) => {
                                        // Call GLSL extended instruction
                                        let glsl_id = constructor.glsl_ext_inst_id;
                                        let operands: Vec<Operand> =
                                            arg_ids.iter().map(|&id| Operand::IdRef(id)).collect();
                                        Ok(constructor.builder.ext_inst(
                                            result_type,
                                            None,
                                            glsl_id,
                                            *ext_op,
                                            operands,
                                        )?)
                                    }
                                    PrimOp::Dot => {
                                        if arg_ids.len() != 2 {
                                            bail_spirv!("dot requires 2 args");
                                        }
                                        Ok(constructor.builder.dot(
                                            result_type,
                                            None,
                                            arg_ids[0],
                                            arg_ids[1],
                                        )?)
                                    }
                                    PrimOp::MatrixTimesMatrix => {
                                        if arg_ids.len() != 2 {
                                            bail_spirv!("matrix × matrix requires 2 args");
                                        }
                                        Ok(constructor.builder.matrix_times_matrix(
                                            result_type,
                                            None,
                                            arg_ids[0],
                                            arg_ids[1],
                                        )?)
                                    }
                                    PrimOp::MatrixTimesVector => {
                                        if arg_ids.len() != 2 {
                                            bail_spirv!("matrix × vector requires 2 args");
                                        }
                                        Ok(constructor.builder.matrix_times_vector(
                                            result_type,
                                            None,
                                            arg_ids[0],
                                            arg_ids[1],
                                        )?)
                                    }
                                    PrimOp::VectorTimesMatrix => {
                                        if arg_ids.len() != 2 {
                                            bail_spirv!("vector × matrix requires 2 args");
                                        }
                                        Ok(constructor.builder.vector_times_matrix(
                                            result_type,
                                            None,
                                            arg_ids[0],
                                            arg_ids[1],
                                        )?)
                                    }
                                    // Type conversions
                                    PrimOp::FPToSI => {
                                        if arg_ids.len() != 1 {
                                            bail_spirv!("FPToSI requires 1 arg");
                                        }
                                        Ok(constructor.builder.convert_f_to_s(
                                            result_type,
                                            None,
                                            arg_ids[0],
                                        )?)
                                    }
                                    PrimOp::FPToUI => {
                                        if arg_ids.len() != 1 {
                                            bail_spirv!("FPToUI requires 1 arg");
                                        }
                                        Ok(constructor.builder.convert_f_to_u(
                                            result_type,
                                            None,
                                            arg_ids[0],
                                        )?)
                                    }
                                    PrimOp::SIToFP => {
                                        if arg_ids.len() != 1 {
                                            bail_spirv!("SIToFP requires 1 arg");
                                        }
                                        Ok(constructor.builder.convert_s_to_f(
                                            result_type,
                                            None,
                                            arg_ids[0],
                                        )?)
                                    }
                                    PrimOp::UIToFP => {
                                        if arg_ids.len() != 1 {
                                            bail_spirv!("UIToFP requires 1 arg");
                                        }
                                        Ok(constructor.builder.convert_u_to_f(
                                            result_type,
                                            None,
                                            arg_ids[0],
                                        )?)
                                    }
                                    PrimOp::FPConvert => {
                                        if arg_ids.len() != 1 {
                                            bail_spirv!("FPConvert requires 1 arg");
                                        }
                                        Ok(constructor.builder.f_convert(result_type, None, arg_ids[0])?)
                                    }
                                    PrimOp::SConvert => {
                                        if arg_ids.len() != 1 {
                                            bail_spirv!("SConvert requires 1 arg");
                                        }
                                        Ok(constructor.builder.s_convert(result_type, None, arg_ids[0])?)
                                    }
                                    PrimOp::UConvert => {
                                        if arg_ids.len() != 1 {
                                            bail_spirv!("UConvert requires 1 arg");
                                        }
                                        Ok(constructor.builder.u_convert(result_type, None, arg_ids[0])?)
                                    }
                                    PrimOp::Bitcast => {
                                        if arg_ids.len() != 1 {
                                            bail_spirv!("Bitcast requires 1 arg");
                                        }
                                        Ok(constructor.builder.bitcast(result_type, None, arg_ids[0])?)
                                    }
                                    _ => {
                                        bail_spirv!("Unsupported PrimOp for: {}", func)
                                    }
                                }
                            }
                            BuiltinImpl::Intrinsic(custom_impl) => {
                                use crate::impl_source::Intrinsic;
                                match custom_impl {
                                    Intrinsic::Placeholder if func == "length" => {
                                        if args.len() != 1 {
                                            bail_spirv!("length expects exactly 1 argument");
                                        }
                                        lower_length_intrinsic(constructor, body, args[0], arg_ids[0])
                                    }
                                    Intrinsic::Placeholder => {
                                        // Other placeholder intrinsics should have been desugared
                                        bail_spirv!(
                                            "Placeholder intrinsic '{}' should have been desugared before lowering",
                                            func
                                        )
                                    }
                                    Intrinsic::Uninit => {
                                        // Return an undefined value of the result type
                                        Ok(constructor.builder.undef(result_type, None))
                                    }
                                    Intrinsic::ArrayWith => {
                                        // array_with arr idx val: functional update, returns new array
                                        if arg_ids.len() != 3 {
                                            bail_spirv!("array_with expects exactly 3 arguments");
                                        }
                                        let arr_id = arg_ids[0];
                                        let idx_id = arg_ids[1];
                                        let val_id = arg_ids[2];

                                        // Store array in a variable, update element, load back
                                        let arr_type = result_type;
                                        let arr_var =
                                            constructor.declare_variable("_w_array_with_tmp", arr_type)?;
                                        constructor.builder.store(arr_var, arr_id, None, [])?;

                                        // Get pointer to element and store new value
                                        let arg2_ty = body.get_type(args[2]);
                                        let elem_type = constructor.ast_type_to_spirv(arg2_ty);
                                        let elem_ptr_type = constructor.builder.type_pointer(
                                            None,
                                            StorageClass::Function,
                                            elem_type,
                                        );
                                        let elem_ptr = constructor.builder.access_chain(
                                            elem_ptr_type,
                                            None,
                                            arr_var,
                                            [idx_id],
                                        )?;
                                        constructor.builder.store(elem_ptr, val_id, None, [])?;

                                        // Load and return the updated array
                                        Ok(constructor.builder.load(arr_type, None, arr_var, None, [])?)
                                    }
                                }
                            }
                            BuiltinImpl::LinkedSpirv(linkage_name) => {
                                let linkage_name = linkage_name.clone();
                                let func_id = constructor.get_or_declare_linked_function(
                                    &linkage_name,
                                    &arg_types,
                                    result_type,
                                )?;
                                Ok(constructor.builder.function_call(result_type, None, func_id, arg_ids)?)
                            }
                        }
                    } else {
                        // Look up user-defined function
                        let func_id = *constructor
                            .functions
                            .get(func)
                            .ok_or_else(|| err_spirv!("Unknown function: {}", func))?;
                        Ok(constructor.builder.function_call(result_type, None, func_id, arg_ids)?)
                    }
                }
            }
        }

        Expr::Intrinsic { name, args } => {
            // Get the result type from the expression
            let result_type = constructor.ast_type_to_spirv(expr_ty);

            match name.as_str() {
                "tuple_access" => {
                    if args.len() != 2 {
                        bail_spirv!("tuple_access requires 2 args");
                    }
                    // Second arg should be a constant index - extract it from the literal
                    let index = match body.get_expr(args[1]) {
                        Expr::Int(s) => {
                            s.parse::<u32>().unwrap_or_else(|e| {
                                panic!("BUG: tuple_access index '{}' failed to parse as u32: {}. Type checking should ensure valid indices.", s, e)
                            })
                        }
                        _ => {
                            panic!("BUG: tuple_access requires a constant integer literal as second argument. Type checking should ensure this.")
                        }
                    };

                    let arg0_ty = body.get_type(args[0]);
                    let composite_id = if types::is_pointer(arg0_ty) {
                        // It's a pointer, load the value
                        let ptr = lower_expr(constructor, body, args[0])?;
                        let pointee_ty = types::pointee(arg0_ty).expect("Pointer type should have pointee");
                        let value_type = constructor.ast_type_to_spirv(pointee_ty);
                        constructor.builder.load(value_type, None, ptr, None, [])?
                    } else {
                        lower_expr(constructor, body, args[0])?
                    };

                    constructor.composite_extract_cached(result_type, composite_id, index)
                }
                "index" => {
                    if args.len() != 2 {
                        bail_spirv!("index requires 2 args");
                    }
                    lower_index_intrinsic(constructor, body, args[0], args[1], result_type)
                }
                "assert" => {
                    // Assertions are no-ops in release, return body
                    if args.len() >= 2 {
                        lower_expr(constructor, body, args[1])
                    } else {
                        Err(err_spirv!(
                            "BUG: assert intrinsic requires at least 2 arguments (condition, body), got {}",
                            args.len()
                        ))
                    }
                }
                "_w_index" => {
                    if args.len() != 2 {
                        bail_spirv!("_w_index requires 2 args");
                    }
                    lower_index_intrinsic(constructor, body, args[0], args[1], result_type)
                }
                "_w_tuple_proj" => {
                    // _w_tuple_proj tuple index - extracts field from a tuple/record
                    if args.len() != 2 {
                        bail_spirv!("_w_tuple_proj requires 2 args, got {}", args.len());
                    }

                    // Second arg should be a constant index
                    let index = match body.get_expr(args[1]) {
                        Expr::Int(s) => s.parse::<u32>().unwrap_or_else(|e| {
                            panic!("BUG: _w_tuple_proj index '{}' failed to parse as u32: {}", s, e)
                        }),
                        _ => {
                            panic!(
                                "BUG: _w_tuple_proj requires a constant integer literal as second argument"
                            )
                        }
                    };

                    let arg0_ty = body.get_type(args[0]);
                    let composite_id = if types::is_pointer(arg0_ty) {
                        let ptr = lower_expr(constructor, body, args[0])?;
                        let pointee_ty = types::pointee(arg0_ty).expect("Pointer type should have pointee");
                        let value_type = constructor.ast_type_to_spirv(pointee_ty);
                        constructor.builder.load(value_type, None, ptr, None, [])?
                    } else {
                        lower_expr(constructor, body, args[0])?
                    };

                    constructor.composite_extract_cached(result_type, composite_id, index)
                }
                // SOAC (Second-Order Array Combinator) intrinsics
                "_w_intrinsic_map" | "map" => lower_map(constructor, body, args, expr_ty, result_type),
                "_w_intrinsic_inplace_map" => {
                    lower_inplace_map(constructor, body, args, expr_ty, result_type)
                }
                "_w_intrinsic_zip" => lower_zip(constructor, body, args, result_type),
                "_w_intrinsic_map_into" => lower_map_into(constructor, body, args),
                "_w_intrinsic_reduce" => lower_reduce(constructor, body, args, result_type),
                "_w_intrinsic_scan" => lower_scan(constructor, body, args, result_type),
                "_w_intrinsic_filter" => lower_filter(constructor, body, args, result_type),
                "_w_intrinsic_scatter" => lower_scatter(constructor, body, args, result_type),
                "_w_intrinsic_hist_1d" => lower_hist_1d(constructor, body, args, result_type),
                "_w_intrinsic_length" => lower_length(constructor, body, args),
                "_w_intrinsic_replicate" => lower_replicate(constructor, body, args, expr_ty, result_type),
                "_w_intrinsic_rotr32" => lower_rotr32(constructor, body, args),
                "_w_intrinsic_sha256_block" => lower_sha256_block(constructor, body, args, result_type),
                "_w_intrinsic_sha256_block_with_state" => {
                    lower_sha256_block_with_state(constructor, body, args, result_type)
                }
                "__builtin_thread_id" => {
                    // Get thread index from GlobalInvocationId.x
                    let gid_var = constructor
                        .global_invocation_id
                        .ok_or_else(|| err_spirv!("__builtin_thread_id used outside of compute shader"))?;
                    let uvec3_type = constructor.get_or_create_vec_type(constructor.u32_type, 3);
                    let gid = constructor.builder.load(uvec3_type, None, gid_var, None, [])?;
                    let thread_id_u32 =
                        constructor.builder.composite_extract(constructor.u32_type, None, gid, [0])?;
                    // Bitcast u32 to i32 for indexing
                    Ok(constructor.builder.bitcast(constructor.i32_type, None, thread_id_u32)?)
                }
                _ => Err(err_spirv!("Unknown intrinsic: {}", name)),
            }
        }

        Expr::Attributed { expr, .. } => {
            // Attributes are metadata, just lower the inner expression
            lower_expr(constructor, body, *expr)
        }

        Expr::Materialize(inner) => {
            // Evaluate the inner expression to get its value
            let value_id = lower_expr(constructor, body, *inner)?;
            let inner_ty = body.get_type(*inner);
            let value_type_id = constructor.ast_type_to_spirv(inner_ty);

            // Declare a function-local variable to hold the value
            let var_id = constructor.declare_variable("_mat", value_type_id)?;

            // Store the value into the variable
            constructor.builder.store(var_id, value_id, None, [])?;

            // Return the variable pointer (for use with OpAccessChain)
            Ok(var_id)
        }

        Expr::Tuple(elems) => {
            // Lower all element expressions
            let elem_ids: Vec<spirv::Word> =
                elems.iter().map(|&e| lower_expr(constructor, body, e)).collect::<Result<Vec<_>>>()?;

            // Get the tuple type
            let result_type = constructor.ast_type_to_spirv(expr_ty);

            // Construct the composite
            Ok(constructor.builder.composite_construct(result_type, None, elem_ids)?)
        }

        Expr::Array { backing, size } => lower_array_expr(constructor, body, backing, *size, expr_ty),

        Expr::Vector(elems) => {
            // Lower all element expressions
            let elem_ids: Vec<spirv::Word> =
                elems.iter().map(|&e| lower_expr(constructor, body, e)).collect::<Result<Vec<_>>>()?;

            // Get the vector type
            let result_type = constructor.ast_type_to_spirv(expr_ty);

            // Construct the composite
            Ok(constructor.builder.composite_construct(result_type, None, elem_ids)?)
        }

        Expr::Matrix(rows) => {
            // Lower each row as a vector, then construct the matrix
            let row_ids: Vec<spirv::Word> = rows
                .iter()
                .map(|row| {
                    let elem_ids: Vec<spirv::Word> = row
                        .iter()
                        .map(|&e| lower_expr(constructor, body, e))
                        .collect::<Result<Vec<_>>>()?;
                    // Get element type from first element
                    let elem_ty = body.get_type(row[0]);
                    let elem_spirv_type = constructor.ast_type_to_spirv(elem_ty);
                    let row_type = constructor.get_or_create_vec_type(elem_spirv_type, row.len() as u32);
                    Ok(constructor.builder.composite_construct(row_type, None, elem_ids)?)
                })
                .collect::<Result<Vec<_>>>()?;

            // Get the matrix type
            let result_type = constructor.ast_type_to_spirv(expr_ty);

            // Construct the matrix from row vectors
            Ok(constructor.builder.composite_construct(result_type, None, row_ids)?)
        }

        // --- Memory operations ---
        Expr::Load { .. } | Expr::Store { .. } => {
            // TODO(Phase 5): Implement Load/Store lowering with address space support
            Err(err_spirv!(
                "Load/Store expressions not yet implemented in SPIR-V lowering"
            ))
        }
    }
}

/// Lower an array expression with unified backing representation.
fn lower_array_expr(
    constructor: &mut Constructor,
    body: &Body,
    backing: &ArrayBacking,
    size: ExprId,
    expr_ty: &PolyType<TypeName>,
) -> Result<spirv::Word> {
    match backing {
        ArrayBacking::Literal(elems) => {
            // Lower all element expressions
            let elem_ids: Vec<spirv::Word> =
                elems.iter().map(|&e| lower_expr(constructor, body, e)).collect::<Result<Vec<_>>>()?;

            // Get the array type
            let result_type = constructor.ast_type_to_spirv(expr_ty);

            // Construct the composite
            Ok(constructor.builder.composite_construct(result_type, None, elem_ids)?)
        }

        ArrayBacking::Range { start, step, kind } => {
            // Extract constant values from range components
            let start_val = try_extract_const_int(body, *start)
                .ok_or_else(|| err_spirv!("Range start must be a compile-time constant integer"))?;
            let size_val = try_extract_const_int(body, size)
                .ok_or_else(|| err_spirv!("Range size must be a compile-time constant integer"))?;

            // Calculate stride
            let stride = match step {
                Some(s) => {
                    let step_val = try_extract_const_int(body, *s)
                        .ok_or_else(|| err_spirv!("Range step must be a compile-time constant integer"))?;
                    step_val - start_val
                }
                None => 1,
            };
            if stride == 0 {
                bail_spirv!("Range stride cannot be zero");
            }

            // Determine count from size and kind
            let count = match kind {
                crate::mir::RangeKind::ExclusiveLt => size_val as i32,
                _ => size_val as i32,
            };

            if count <= 0 {
                bail_spirv!("Range produces empty or negative-count array");
            }

            // Generate the array elements
            let elem_ids: Vec<spirv::Word> =
                (0..count).map(|i| constructor.const_i32(start_val + i * stride)).collect();

            // Get element type (i32) and construct array type
            let elem_type = constructor.i32_type;
            let array_type = constructor.type_array(elem_type, count as u32);

            // Construct the composite array
            Ok(constructor.builder.composite_construct(array_type, None, elem_ids)?)
        }

        ArrayBacking::IndexFn { .. } => {
            // Index functions should be materialized before lowering
            bail_spirv!("IndexFn arrays should be materialized before SPIR-V lowering")
        }

        ArrayBacking::View { base, offset } => {
            // View: creates a view into base[offset..offset+size]
            let result_ty = expr_ty;
            let base_ty = body.get_type(*base);

            // Extract element type and static size from the result type
            let (elem_ty, static_size) = match result_ty {
                PolyType::Constructed(TypeName::Array, type_args) if type_args.len() == 3 => {
                    let size_from_type = match &type_args[2] {
                        PolyType::Constructed(TypeName::Size(s), _) => Some(*s),
                        PolyType::Variable(_) => {
                            // Unsized (type variable) - try to get from size expression
                            try_extract_const_int(body, size).map(|s| s as usize)
                        }
                        _ => None,
                    };
                    (type_args[0].clone(), size_from_type)
                }
                _ => {
                    return Err(err_spirv!("View result must be Array type, got {:?}", result_ty));
                }
            };

            if let Some(sz) = static_size {
                let offset_is_zero = matches!(body.get_expr(*offset), Expr::Int(s) if s == "0");
                let elem_spirv_type = constructor.ast_type_to_spirv(&elem_ty);

                if offset_is_zero {
                    // Optimization: if offset is 0, we can use compile-time indices
                    let base_val = lower_expr(constructor, body, *base)?;
                    let mut elements = Vec::with_capacity(sz);

                    for i in 0..sz {
                        let elem = constructor.builder.composite_extract(
                            elem_spirv_type,
                            None,
                            base_val,
                            [i as u32],
                        )?;
                        elements.push(elem);
                    }

                    let result_array_type = constructor.type_array(elem_spirv_type, sz as u32);
                    Ok(constructor.builder.composite_construct(result_array_type, None, elements)?)
                } else {
                    // For non-zero offset, use array indexing with runtime computation
                    let base_val = lower_expr(constructor, body, *base)?;
                    let offset_val = lower_expr(constructor, body, *offset)?;
                    let base_spirv_ty = constructor.ast_type_to_spirv(base_ty);

                    let ptr_type = constructor.builder.type_pointer(
                        None,
                        spirv::StorageClass::Function,
                        base_spirv_ty,
                    );
                    let var =
                        constructor.builder.variable(ptr_type, None, spirv::StorageClass::Function, None);
                    constructor.builder.store(var, base_val, None, [])?;

                    let i32_type = constructor.i32_type;
                    let elem_ptr_type = constructor.builder.type_pointer(
                        None,
                        spirv::StorageClass::Function,
                        elem_spirv_type,
                    );
                    let mut elements = Vec::with_capacity(sz);

                    for i in 0..sz {
                        let i_const = constructor.const_i32(i as i32);
                        let idx = constructor.builder.i_add(i32_type, None, offset_val, i_const)?;
                        let ptr = constructor.builder.access_chain(elem_ptr_type, None, var, [idx])?;
                        let elem = constructor.builder.load(elem_spirv_type, None, ptr, None, [])?;
                        elements.push(elem);
                    }

                    let result_array_type = constructor.type_array(elem_spirv_type, sz as u32);
                    Ok(constructor.builder.composite_construct(result_array_type, None, elements)?)
                }
            } else {
                Err(err_spirv!(
                    "View with dynamic size not yet supported, result type: {:?}",
                    result_ty
                ))
            }
        }

        ArrayBacking::Owned { data } => {
            // Owned: the data is already a complete array, just return it
            lower_expr(constructor, body, *data)
        }

        ArrayBacking::Storage { name, offset } => {
            // Storage: creates a (pointer, length) pair for a storage buffer slice
            let (set, binding) = constructor
                .compute_params
                .get(name)
                .ok_or_else(|| err_spirv!("Unknown storage buffer: {}", name))?;
            let (set, binding) = (*set, *binding);

            let (buffer_var, elem_type_id) = constructor
                .storage_buffers
                .get(&(set, binding))
                .ok_or_else(|| err_spirv!("Storage buffer not registered: {}:{}", set, binding))?;
            let (buffer_var, elem_type_id) = (*buffer_var, *elem_type_id);

            let offset_val = lower_expr(constructor, body, *offset)?;
            let len_val = lower_expr(constructor, body, size)?;

            let elem_ptr_type =
                constructor.get_or_create_ptr_type(spirv::StorageClass::StorageBuffer, elem_type_id);
            let zero = constructor.const_u32(0);
            let ptr =
                constructor.builder.access_chain(elem_ptr_type, None, buffer_var, [zero, offset_val])?;

            let slice_struct_type =
                constructor.get_or_create_struct_type(vec![elem_ptr_type, constructor.i32_type]);

            Ok(constructor.builder.composite_construct(slice_struct_type, None, [ptr, len_val])?)
        }
    }
}

/// Read a single element from an array at a given index, dispatching on the backing type.
///
/// This is the core primitive for virtual array access:
/// - For Range: computes `start + index * step` (virtual - no memory access)
/// - For IndexFn: calls the function with the index
/// - For Literal/Owned/View/Storage: performs actual memory access
///
/// Returns the SPIR-V ID of the loaded element value.
fn read_elem(
    constructor: &mut Constructor,
    body: &Body,
    array_expr_id: ExprId,
    backing: &ArrayBacking,
    index: spirv::Word,
    elem_type: spirv::Word,
) -> Result<spirv::Word> {
    match backing {
        ArrayBacking::Literal(elems) => {
            // For literal arrays, we need to check if index is a constant
            if let Some(literal_idx) = constructor.get_const_u32_value(index) {
                // Constant index: use composite extract directly
                if (literal_idx as usize) < elems.len() {
                    // Lower the element expression at that index
                    lower_expr(constructor, body, elems[literal_idx as usize])
                } else {
                    bail_spirv!(
                        "Array index {} out of bounds for literal array of size {}",
                        literal_idx,
                        elems.len()
                    )
                }
            } else {
                // Dynamic index: must store array in variable and use access chain
                let array_val = lower_array_expr(
                    constructor,
                    body,
                    backing,
                    // Need to get size expr - use a dummy since we already have the elements
                    body.root, // This is a hack; we really should pass size through
                    body.get_type(array_expr_id),
                )?;
                let array_type = constructor.ast_type_to_spirv(body.get_type(array_expr_id));
                let array_var = constructor.declare_variable("_w_lit_arr_tmp", array_type)?;
                constructor.builder.store(array_var, array_val, None, [])?;

                let elem_ptr_type =
                    constructor.builder.type_pointer(None, spirv::StorageClass::Function, elem_type);
                let elem_ptr = constructor.builder.access_chain(elem_ptr_type, None, array_var, [index])?;
                Ok(constructor.builder.load(elem_type, None, elem_ptr, None, [])?)
            }
        }

        ArrayBacking::Range { start, step, .. } => {
            // Virtual: compute start + index * step (no memory access!)
            let i32_type = constructor.i32_type;
            let start_val = lower_expr(constructor, body, *start)?;

            if let Some(step_expr) = step {
                // step is actually (start + stride), so stride = step - start
                let step_val = lower_expr(constructor, body, *step_expr)?;
                let stride = constructor.builder.i_sub(i32_type, None, step_val, start_val)?;
                let offset = constructor.builder.i_mul(i32_type, None, index, stride)?;
                Ok(constructor.builder.i_add(i32_type, None, start_val, offset)?)
            } else {
                // No step means stride of 1
                Ok(constructor.builder.i_add(i32_type, None, start_val, index)?)
            }
        }

        ArrayBacking::IndexFn { index_fn } => {
            // Virtual: call the index function with the index
            // The index function should be a closure: i32 -> T
            let (func_name, capture_vals) = extract_closure_info(constructor, body, *index_fn)?;
            let func_id = *constructor
                .functions
                .get(&func_name)
                .ok_or_else(|| err_spirv!("IndexFn function not found: {}", func_name))?;

            let mut call_args = capture_vals;
            call_args.push(index);
            Ok(constructor.builder.function_call(elem_type, None, func_id, call_args)?)
        }

        ArrayBacking::View { base, offset } => {
            // View: adjust index by offset and read from base
            let i32_type = constructor.i32_type;
            let offset_val = lower_expr(constructor, body, *offset)?;
            let adjusted_index = constructor.builder.i_add(i32_type, None, offset_val, index)?;

            // Get base array's backing and recurse
            let base_expr = body.get_expr(*base);
            match base_expr {
                Expr::Array {
                    backing: base_backing,
                    ..
                } => read_elem(constructor, body, *base, base_backing, adjusted_index, elem_type),
                _ => {
                    // Base is a value expression, not an Array with backing
                    // Fall back to storing and indexing
                    let base_val = lower_expr(constructor, body, *base)?;
                    let base_ty = body.get_type(*base);
                    let base_type = constructor.ast_type_to_spirv(base_ty);
                    let base_var = constructor.declare_variable("_w_view_base_tmp", base_type)?;
                    constructor.builder.store(base_var, base_val, None, [])?;

                    let elem_ptr_type =
                        constructor.builder.type_pointer(None, spirv::StorageClass::Function, elem_type);
                    let elem_ptr = constructor.builder.access_chain(
                        elem_ptr_type,
                        None,
                        base_var,
                        [adjusted_index],
                    )?;
                    Ok(constructor.builder.load(elem_type, None, elem_ptr, None, [])?)
                }
            }
        }

        ArrayBacking::Owned { data } => {
            // Owned: the data is a complete array, index into it
            let data_val = lower_expr(constructor, body, *data)?;
            let data_ty = body.get_type(*data);
            let data_type = constructor.ast_type_to_spirv(data_ty);
            let data_var = constructor.declare_variable("_w_owned_tmp", data_type)?;
            constructor.builder.store(data_var, data_val, None, [])?;

            let elem_ptr_type =
                constructor.builder.type_pointer(None, spirv::StorageClass::Function, elem_type);
            let elem_ptr = constructor.builder.access_chain(elem_ptr_type, None, data_var, [index])?;
            Ok(constructor.builder.load(elem_type, None, elem_ptr, None, [])?)
        }

        ArrayBacking::Storage { name, offset } => {
            // Storage: load from external buffer at offset + index
            let (set, binding) = constructor
                .compute_params
                .get(name)
                .ok_or_else(|| err_spirv!("Unknown storage buffer: {}", name))?;
            let (set, binding) = (*set, *binding);

            let (buffer_var, _) = constructor
                .storage_buffers
                .get(&(set, binding))
                .ok_or_else(|| err_spirv!("Storage buffer not registered: {}:{}", set, binding))?;
            let buffer_var = *buffer_var;

            let i32_type = constructor.i32_type;
            let offset_val = lower_expr(constructor, body, *offset)?;
            let adjusted_index = constructor.builder.i_add(i32_type, None, offset_val, index)?;

            let elem_ptr_type =
                constructor.get_or_create_ptr_type(spirv::StorageClass::StorageBuffer, elem_type);
            let zero = constructor.const_u32(0);
            let elem_ptr = constructor.builder.access_chain(
                elem_ptr_type,
                None,
                buffer_var,
                [zero, adjusted_index],
            )?;
            Ok(constructor.builder.load(elem_type, None, elem_ptr, None, [])?)
        }
    }
}

/// Lower the `length` intrinsic for arrays and slices.
/// For static arrays, returns the compile-time size constant.
/// For unsized arrays, extracts the dynamic length field.
fn lower_length_intrinsic(
    constructor: &mut Constructor,
    body: &Body,
    arg_expr_id: ExprId,
    arg_lowered: spirv::Word,
) -> Result<spirv::Word> {
    let arg_ty = body.get_type(arg_expr_id);
    match arg_ty {
        PolyType::Constructed(TypeName::Array, type_args) => {
            // Array[elem, addrspace, size] - size is at index 2
            assert!(type_args.len() == 3);
            let size_arg = &type_args[2];

            match size_arg {
                PolyType::Constructed(TypeName::Size(n), _) => {
                    // Static size: return constant
                    Ok(constructor.const_i32(*n as i32))
                }
                PolyType::Variable(_) => {
                    // Unsized array (type variable): extract size from array expression
                    let array_expr = body.get_expr(arg_expr_id);
                    let i32_type = constructor.i32_type;
                    match array_expr {
                        Expr::Array { size, .. } => lower_expr(constructor, body, *size),
                        _ => {
                            // Array value from a variable - extract from struct (len at index 1)
                            Ok(constructor.builder.composite_extract(i32_type, None, arg_lowered, [1])?)
                        }
                    }
                }
                _ => bail_spirv!("Cannot determine array size for length: {:?}", size_arg),
            }
        }
        _ => bail_spirv!("length called on non-array type: {:?}", arg_ty),
    }
}

/// Lower the `index` intrinsic for arrays.
/// Handles both sized arrays (direct indexing) and unsized arrays (slice semantics).
fn lower_index_intrinsic(
    constructor: &mut Constructor,
    body: &Body,
    array_expr_id: ExprId,
    index_expr_id: ExprId,
    result_type: spirv::Word,
) -> Result<spirv::Word> {
    let arg0_ty = body.get_type(array_expr_id);
    let index_val = lower_expr(constructor, body, index_expr_id)?;

    match arg0_ty {
        PolyType::Constructed(TypeName::Array, type_args) => {
            assert!(type_args.len() == 3);
            // Check if this is an unsized array (slice-like) - size is a type variable
            let is_unsized = matches!(&type_args[2], PolyType::Variable(_));

            if is_unsized {
                // Unsized array - handle slice semantics based on array backing
                let array_expr = body.get_expr(array_expr_id);
                match array_expr {
                    Expr::Array {
                        backing: ArrayBacking::Owned { data },
                        ..
                    } => {
                        // Owned: data is already a complete array, index into it
                        lower_index_intrinsic(constructor, body, *data, index_expr_id, result_type)
                    }
                    Expr::Array {
                        backing: ArrayBacking::View { base, offset },
                        ..
                    } => {
                        // View: adjust index by offset and access base array
                        let offset_val = lower_expr(constructor, body, *offset)?;
                        let i32_type = constructor.i32_type;
                        let adjusted_index =
                            constructor.builder.i_add(i32_type, None, offset_val, index_val)?;

                        let base_ty = body.get_type(*base);
                        let base_var = if types::is_pointer(base_ty) {
                            lower_expr(constructor, body, *base)?
                        } else {
                            let base_val = lower_expr(constructor, body, *base)?;
                            let base_type = constructor.ast_type_to_spirv(base_ty);
                            let base_var = constructor.declare_variable("_w_slice_base_tmp", base_type)?;
                            constructor.builder.store(base_var, base_val, None, [])?;
                            base_var
                        };

                        let elem_ptr_type =
                            constructor.builder.type_pointer(None, StorageClass::Function, result_type);
                        let elem_ptr = constructor.builder.access_chain(
                            elem_ptr_type,
                            None,
                            base_var,
                            [adjusted_index],
                        )?;
                        Ok(constructor.builder.load(result_type, None, elem_ptr, None, [])?)
                    }
                    Expr::Array {
                        backing: ArrayBacking::Storage { name, offset },
                        ..
                    } => {
                        // Storage: access external buffer by name
                        let (set, binding) = constructor
                            .compute_params
                            .get(name)
                            .ok_or_else(|| err_spirv!("Unknown storage buffer: {}", name))?;
                        let (set, binding) = (*set, *binding);

                        let (buffer_var, _elem_type_id) =
                            constructor.storage_buffers.get(&(set, binding)).ok_or_else(|| {
                                err_spirv!("Storage buffer not registered: {}:{}", set, binding)
                            })?;
                        let buffer_var = *buffer_var;

                        let offset_val = lower_expr(constructor, body, *offset)?;
                        let i32_type = constructor.i32_type;
                        let adjusted_index =
                            constructor.builder.i_add(i32_type, None, offset_val, index_val)?;

                        let elem_ptr_type = constructor
                            .get_or_create_ptr_type(spirv::StorageClass::StorageBuffer, result_type);
                        let zero = constructor.const_u32(0);
                        let elem_ptr = constructor.builder.access_chain(
                            elem_ptr_type,
                            None,
                            buffer_var,
                            [zero, adjusted_index],
                        )?;
                        Ok(constructor.builder.load(result_type, None, elem_ptr, None, [])?)
                    }
                    _ => {
                        // Array value from a variable or function call
                        let slice_val = lower_expr(constructor, body, array_expr_id)?;

                        // Check if this is a storage array (Array[elem, Storage, ?size])
                        if types::is_bound_slice_access(arg0_ty) {
                            // Storage slice: {ptr, len} struct
                            let ptr_type = constructor
                                .get_or_create_ptr_type(spirv::StorageClass::StorageBuffer, result_type);
                            let ptr =
                                constructor.builder.composite_extract(ptr_type, None, slice_val, [0])?;
                            let elem_ptr =
                                constructor.builder.ptr_access_chain(ptr_type, None, ptr, index_val, [])?;
                            return Ok(constructor.builder.load(result_type, None, elem_ptr, None, [])?);
                        }

                        // Owned array layout: extract data field and index into it
                        bail_spirv!("Unsupported unsized array indexing: {:?}", arg0_ty)
                    }
                }
            } else {
                // Sized array - check for constant index optimization
                let array_val = lower_expr(constructor, body, array_expr_id)?;

                // If index is a compile-time constant, use OpCompositeExtract
                if let Some(literal_idx) = constructor.get_const_u32_value(index_val) {
                    Ok(constructor.builder.composite_extract(
                        result_type,
                        None,
                        array_val,
                        [literal_idx],
                    )?)
                } else {
                    // Runtime index - must materialize to local variable
                    let array_type = constructor.ast_type_to_spirv(arg0_ty);
                    let array_var = constructor.declare_variable("_w_index_tmp", array_type)?;
                    constructor.builder.store(array_var, array_val, None, [])?;

                    let elem_ptr_type =
                        constructor.builder.type_pointer(None, StorageClass::Function, result_type);
                    let elem_ptr =
                        constructor.builder.access_chain(elem_ptr_type, None, array_var, [index_val])?;
                    Ok(constructor.builder.load(result_type, None, elem_ptr, None, [])?)
                }
            }
        }
        PolyType::Constructed(TypeName::Pointer, _) => {
            // Pointer indexing
            let ptr = lower_expr(constructor, body, array_expr_id)?;
            let elem_ptr_type = constructor.builder.type_pointer(None, StorageClass::Function, result_type);
            let elem_ptr = constructor.builder.access_chain(elem_ptr_type, None, ptr, [index_val])?;
            Ok(constructor.builder.load(result_type, None, elem_ptr, None, [])?)
        }
        _ => bail_spirv!("index called on non-array type: {:?}", arg0_ty),
    }
}

/// Try to extract a compile-time constant integer from an expression.
fn try_extract_const_int(body: &Body, expr_id: ExprId) -> Option<i32> {
    match body.get_expr(expr_id) {
        Expr::Int(s) => s.parse().ok(),
        _ => None,
    }
}

// =============================================================================
// SOAC (Second-Order Array Combinator) lowering helpers
// =============================================================================

/// Extract closure information from flattened SOAC args.
/// Args format: [func_ref, other_args..., captures...]
/// For map: [func_ref, array, captures...]
/// Returns (function_name, capture_values) where capture_values is a Vec of lowered captures.
fn extract_flattened_closure_info(
    constructor: &mut Constructor,
    body: &Body,
    args: &[ExprId],
) -> Result<(String, Vec<spirv::Word>)> {
    // args[0] should be a Global reference to the function name
    let func_name = match body.get_expr(args[0]) {
        Expr::Global(name) => name.clone(),
        other => bail_spirv!("Expected function reference (Global) in SOAC, got {:?}", other),
    };

    // For map: args[1] is array, args[2..] are captures
    // Lower captures (skip func_ref at 0 and array at 1)
    let capture_vals: Vec<spirv::Word> =
        args[2..].iter().map(|&c| lower_expr(constructor, body, c)).collect::<Result<Vec<_>>>()?;

    Ok((func_name, capture_vals))
}

/// Extract function information from a SOAC operator argument.
/// Returns (function_name, capture_values) where capture_values is a Vec of individual captures.
fn extract_closure_info(
    constructor: &mut Constructor,
    body: &Body,
    arg_expr_id: ExprId,
) -> Result<(String, Vec<spirv::Word>)> {
    extract_closure_info_inner(constructor, body, arg_expr_id, 0)
}

fn extract_closure_info_inner(
    constructor: &mut Constructor,
    body: &Body,
    arg_expr_id: ExprId,
    depth: usize,
) -> Result<(String, Vec<spirv::Word>)> {
    // Prevent infinite recursion
    if depth > 10 {
        bail_spirv!("Too deep recursion in extract_closure_info");
    }

    match body.get_expr(arg_expr_id) {
        Expr::Global(name) => {
            // Named function reference - no captures
            Ok((name.clone(), vec![]))
        }
        Expr::Call { func, args } => {
            // Partial application to a lifted lambda: (_lambda_N arg1 arg2 ...)
            // The args become captures for the resulting closure
            let capture_vals: Vec<spirv::Word> =
                args.iter().map(|&a| lower_expr(constructor, body, a)).collect::<Result<Vec<_>>>()?;
            Ok((func.clone(), capture_vals))
        }
        Expr::Let {
            rhs, body: let_body, ..
        } => {
            // Let binding - look at the body (the continuation)
            // But first check the rhs in case it's the function
            if let Ok(result) = extract_closure_info_inner(constructor, body, *rhs, depth + 1) {
                return Ok(result);
            }
            extract_closure_info_inner(constructor, body, *let_body, depth + 1)
        }
        Expr::Local(local_id) => {
            // Local reference - try to find what it was assigned to
            // Look through the expression tree to find the Let that binds this local
            if let Some(rhs_id) = find_local_binding(body, *local_id, arg_expr_id) {
                extract_closure_info_inner(constructor, body, rhs_id, depth + 1)
            } else {
                bail_spirv!("Cannot find binding for local {:?}", local_id);
            }
        }
        other => {
            bail_spirv!("Expected function reference (Global), got {:?}", other);
        }
    }
}

/// Find the RHS expression that binds a local variable by searching the expression tree.
fn find_local_binding(body: &Body, target_local: LocalId, _start_expr: ExprId) -> Option<ExprId> {
    // Search through all expressions to find the Let that binds this local
    // This is a simple linear search - could be optimized with a pre-computed map
    for expr in body.exprs.iter() {
        if let Expr::Let { local, rhs, .. } = expr {
            if *local == target_local {
                return Some(*rhs);
            }
        }
    }
    None
}

/// Extract array size and element type from an array type.
fn extract_array_info(
    constructor: &mut Constructor,
    ty: &PolyType<TypeName>,
) -> Result<(u32, spirv::Word)> {
    match ty {
        PolyType::Constructed(TypeName::Array, type_args) => {
            assert!(type_args.len() == 3);
            let size = match &type_args[2] {
                PolyType::Constructed(TypeName::Size(n), _) => *n as u32,
                _ => bail_spirv!("Invalid array size type"),
            };
            let elem_type = constructor.ast_type_to_spirv(&type_args[0]);
            Ok((size, elem_type))
        }
        _ => bail_spirv!("Expected array type, got {:?}", ty),
    }
}

/// Read an element from an array, handling both value arrays and storage-backed arrays.
///
/// For value arrays (`mem: None`): uses OpCompositeExtract with constant index
/// For storage arrays (`mem: Some(Storage{..})`): uses OpAccessChain + OpLoad with runtime index
fn read_array_element(
    constructor: &mut Constructor,
    mem: Option<MemBinding>,
    array_val: spirv::Word,
    index: spirv::Word, // SPIR-V ID of the index (constant or runtime)
    elem_type: spirv::Word,
) -> Result<spirv::Word> {
    match mem {
        Some(MemBinding::Storage { set, binding }) => {
            let (buffer_var, _) = constructor.storage_buffers.get(&(set, binding)).ok_or_else(|| {
                err_spirv!("Storage buffer not found for set={}, binding={}", set, binding)
            })?;
            let buffer_var = *buffer_var;

            let elem_ptr_type =
                constructor.get_or_create_ptr_type(spirv::StorageClass::StorageBuffer, elem_type);
            let zero = constructor.const_u32(0);
            let elem_ptr =
                constructor.builder.access_chain(elem_ptr_type, None, buffer_var, [zero, index])?;
            Ok(constructor.builder.load(elem_type, None, elem_ptr, None, [])?)
        }
        None => {
            // For value arrays, we need a literal u32 index for OpCompositeExtract
            let literal_idx = constructor
                .get_const_u32_value(index)
                .ok_or_else(|| err_spirv!("Value array access requires constant index"))?;
            Ok(constructor.builder.composite_extract(elem_type, None, array_val, [literal_idx])?)
        }
    }
}

/// Write an element to an array, handling both value arrays and storage-backed arrays.
///
/// For value arrays (`mem: None`): uses OpCompositeInsert, returns new array value
/// For storage arrays (`mem: Some(Storage{..})`): uses OpAccessChain + OpStore, returns original array_val
fn write_array_element(
    constructor: &mut Constructor,
    mem: Option<MemBinding>,
    array_val: spirv::Word,
    index: spirv::Word, // SPIR-V ID of the index (constant or runtime)
    value: spirv::Word,
    elem_type: spirv::Word,
    result_type: spirv::Word,
) -> Result<spirv::Word> {
    match mem {
        Some(MemBinding::Storage { set, binding }) => {
            let (buffer_var, _) = constructor.storage_buffers.get(&(set, binding)).ok_or_else(|| {
                err_spirv!("Storage buffer not found for set={}, binding={}", set, binding)
            })?;
            let buffer_var = *buffer_var;

            let elem_ptr_type =
                constructor.get_or_create_ptr_type(spirv::StorageClass::StorageBuffer, elem_type);
            let zero = constructor.const_u32(0);
            let elem_ptr =
                constructor.builder.access_chain(elem_ptr_type, None, buffer_var, [zero, index])?;
            constructor.builder.store(elem_ptr, value, None, [])?;
            Ok(array_val) // Storage writes are side-effects, return original value
        }
        None => {
            // For value arrays, we need a literal u32 index for OpCompositeInsert
            let literal_idx = constructor
                .get_const_u32_value(index)
                .ok_or_else(|| err_spirv!("Value array write requires constant index"))?;
            Ok(constructor.builder.composite_insert(result_type, None, value, array_val, [literal_idx])?)
        }
    }
}

/// Core in-place map over storage array elements.
///
/// This is the general-purpose lowering for in-place map, used by both:
/// - `lower_inplace_map` (passes constant indices 0..array_size)
/// - `lower_compute_entry_point` (passes single runtime thread_id)
///
/// The caller provides the indices to iterate over, allowing the same
/// lowering logic to handle both compile-time and runtime iteration.
fn lower_inplace_map_core(
    constructor: &mut Constructor,
    mem: MemBinding,
    array_val: spirv::Word,
    elem_type: spirv::Word,
    map_func_id: spirv::Word,
    capture_vals: &[spirv::Word],
    output_elem_type: spirv::Word,
    indices: &[spirv::Word],
) -> Result<()> {
    for &index in indices {
        // Read element
        let input_elem = read_array_element(constructor, Some(mem), array_val, index, elem_type)?;

        // Apply function: captures first, then input element
        let mut call_args = capture_vals.to_vec();
        call_args.push(input_elem);
        let result_elem =
            constructor.builder.function_call(output_elem_type, None, map_func_id, call_args)?;

        // Write result back
        write_array_element(
            constructor,
            Some(mem),
            array_val,
            index,
            result_elem,
            output_elem_type,
            constructor.void_type,
        )?;
    }
    Ok(())
}

/// Lower `_w_intrinsic_map`: map f [a,b,c] = [f(a), f(b), f(c)]
///
/// For static-sized arrays: unrolls the loop at compile time.
/// For dynamic-sized Views (e.g., in compute shaders): generates a runtime loop
/// that reads from input and writes to output storage buffers.
fn lower_map(
    constructor: &mut Constructor,
    body: &Body,
    args: &[ExprId],
    expr_ty: &PolyType<TypeName>,
    result_type: spirv::Word,
) -> Result<spirv::Word> {
    if args.len() < 2 {
        bail_spirv!(
            "_w_intrinsic_map requires at least 2 args (function, array), got {}",
            args.len()
        );
    }

    // Args format: [func_ref, array, captures...]
    // - args[0] is Global(func_name) or Closure
    // - args[1] is the array
    // - args[2..] are captures (may be empty)
    let (func_name, capture_vals) = extract_flattened_closure_info(constructor, body, args)?;

    let arr_expr_id = args[1];
    let arr_ty = body.get_type(arr_expr_id);

    // Try to get static size; if not available, check for dynamic View
    let static_size = match arr_ty {
        PolyType::Constructed(TypeName::Array, type_args) if type_args.len() == 3 => match &type_args[2] {
            PolyType::Constructed(TypeName::Size(n), _) => Some(*n as u32),
            _ => None,
        },
        _ => None,
    };

    // Get element types
    let (elem_type, output_elem_type) = match arr_ty {
        PolyType::Constructed(TypeName::Array, type_args) if type_args.len() == 3 => {
            let elem = constructor.ast_type_to_spirv(&type_args[0]);
            let out_elem = match expr_ty {
                PolyType::Constructed(TypeName::Array, out_args) if out_args.len() == 3 => {
                    constructor.ast_type_to_spirv(&out_args[0])
                }
                _ => elem,
            };
            (elem, out_elem)
        }
        _ => bail_spirv!("map input must be array type, got {:?}", arr_ty),
    };

    let map_func_id = *constructor
        .functions
        .get(&func_name)
        .ok_or_else(|| err_spirv!("Map function not found: {}", func_name))?;

    // Check if this is a dynamic-sized View into storage
    // The View expression is passed directly (not through a local binding)
    let arr_expr = body.get_expr(arr_expr_id);
    let is_dynamic_storage_view = static_size.is_none()
        && matches!(
            arr_expr,
            Expr::Array {
                backing: ArrayBacking::View { .. },
                ..
            }
        );

    if is_dynamic_storage_view {
        // Dynamic-sized View: generate a runtime loop
        return lower_map_dynamic(
            constructor,
            body,
            arr_expr_id,
            map_func_id,
            &capture_vals,
            elem_type,
            output_elem_type,
        );
    }

    let array_size = static_size.ok_or_else(|| {
        err_spirv!(
            "map requires static array size or View into storage, got {:?}",
            arr_ty
        )
    })?;

    // Check if input array has a virtual backing (Range, IndexFn)
    // If so, use read_elem directly to avoid materialization
    let use_read_elem = matches!(
        arr_expr,
        Expr::Array {
            backing: ArrayBacking::Range { .. } | ArrayBacking::IndexFn { .. },
            ..
        }
    );

    // Build output array elements
    let mut result_elements = Vec::with_capacity(array_size as usize);

    if use_read_elem {
        // Virtual array: use read_elem to compute elements on-the-fly
        if let Expr::Array { backing, .. } = arr_expr {
            for i in 0..array_size {
                let idx = constructor.const_i32(i as i32);
                let input_elem = read_elem(constructor, body, arr_expr_id, backing, idx, elem_type)?;

                // Order: element first, then captures (captures are trailing params after lifting)
                let mut call_args = vec![input_elem];
                call_args.extend(capture_vals.clone());
                let result_elem =
                    constructor.builder.function_call(output_elem_type, None, map_func_id, call_args)?;
                result_elements.push(result_elem);
            }
        }
    } else {
        // Materialized array: lower it first, then read elements
        let arr_val = lower_expr(constructor, body, arr_expr_id)?;
        let mem: Option<MemBinding> = None;

        for i in 0..array_size {
            let idx = constructor.const_u32(i);
            let input_elem = read_array_element(constructor, mem, arr_val, idx, elem_type)?;

            // Order: element first, then captures (captures are trailing params after lifting)
            let mut call_args = vec![input_elem];
            call_args.extend(capture_vals.clone());
            let result_elem =
                constructor.builder.function_call(output_elem_type, None, map_func_id, call_args)?;
            result_elements.push(result_elem);
        }
    }

    Ok(constructor.builder.composite_construct(result_type, None, result_elements)?)
}

/// Lower map over a dynamic-sized View using a runtime loop.
///
/// Uses a variable-based loop instead of phi nodes for simplicity.
/// Generates SPIR-V structured control flow that reads from input,
/// applies the function, and writes results back in-place.
fn lower_map_dynamic(
    constructor: &mut Constructor,
    body: &Body,
    arr_expr_id: ExprId,
    map_func_id: spirv::Word,
    capture_vals: &[spirv::Word],
    elem_type: spirv::Word,
    output_elem_type: spirv::Word,
) -> Result<spirv::Word> {
    let arr_expr = body.get_expr(arr_expr_id);

    // Extract View info
    let (base_expr_id, offset_expr_id, size_expr_id) = match arr_expr {
        Expr::Array {
            backing: ArrayBacking::View { base, offset },
            size,
        } => (*base, *offset, *size),
        _ => bail_spirv!("lower_map_dynamic requires View array"),
    };

    // Get the base storage buffer info
    let base_expr = body.get_expr(base_expr_id);
    let (storage_name, base_offset_expr_id) = match base_expr {
        Expr::Array {
            backing: ArrayBacking::Storage { name, offset },
            ..
        } => (name.clone(), *offset),
        _ => bail_spirv!("lower_map_dynamic requires View into Storage array"),
    };

    // Get storage buffer binding
    let (set, binding) = constructor
        .compute_params
        .get(&storage_name)
        .ok_or_else(|| err_spirv!("Unknown storage buffer: {}", storage_name))?;
    let (buffer_var, buffer_elem_type) = constructor
        .storage_buffers
        .get(&(*set, *binding))
        .ok_or_else(|| err_spirv!("Storage buffer not found for set={}, binding={}", set, binding))?;
    let buffer_var = *buffer_var;
    let buffer_elem_type = *buffer_elem_type;

    // Lower the offset and size expressions
    let base_offset_val = lower_expr(constructor, body, base_offset_expr_id)?;
    let view_offset_val = lower_expr(constructor, body, offset_expr_id)?;
    let size_val = lower_expr(constructor, body, size_expr_id)?;

    // Compute the combined offset: base_offset + view_offset
    let i32_type = constructor.i32_type;
    let combined_offset = constructor.builder.i_add(i32_type, None, base_offset_val, view_offset_val)?;

    // Create loop index variable (in the function's variables block)
    let loop_idx_var = constructor.declare_variable("__map_idx", i32_type)?;

    // Initialize loop index to 0
    let zero = constructor.const_i32(0);
    let one = constructor.const_i32(1);
    constructor.builder.store(loop_idx_var, zero, None, [])?;

    // Create basic blocks
    let header_label = constructor.builder.id();
    let body_label = constructor.builder.id();
    let continue_label = constructor.builder.id();
    let merge_label = constructor.builder.id();

    // Branch to header
    constructor.builder.branch(header_label)?;

    // Header block with loop merge
    constructor.builder.begin_block(Some(header_label))?;
    // Load current index and check if i < size (before loop_merge per SPIR-V rules)
    let i_val = constructor.builder.load(i32_type, None, loop_idx_var, None, [])?;
    let bool_type = constructor.bool_type;
    let cond = constructor.builder.s_less_than(bool_type, None, i_val, size_val)?;
    constructor.builder.loop_merge(merge_label, continue_label, spirv::LoopControl::NONE, [])?;
    constructor.builder.branch_conditional(cond, body_label, merge_label, [])?;

    // Body block: read, apply function, write
    constructor.builder.begin_block(Some(body_label))?;

    // Load index again (needed because we're in a new block)
    let i_val_body = constructor.builder.load(i32_type, None, loop_idx_var, None, [])?;

    // Compute actual index: combined_offset + i
    let actual_idx = constructor.builder.i_add(i32_type, None, combined_offset, i_val_body)?;

    // Read from storage buffer
    let elem_ptr_type =
        constructor.get_or_create_ptr_type(spirv::StorageClass::StorageBuffer, buffer_elem_type);
    let zero_u32 = constructor.const_u32(0);
    // Cast i32 index to u32 for array indexing
    let actual_idx_u32 = constructor.builder.bitcast(constructor.u32_type, None, actual_idx)?;
    let elem_ptr =
        constructor.builder.access_chain(elem_ptr_type, None, buffer_var, [zero_u32, actual_idx_u32])?;
    let input_elem = constructor.builder.load(elem_type, None, elem_ptr, None, [])?;

    // Call the map function
    let mut call_args = vec![input_elem];
    call_args.extend(capture_vals.iter().cloned());
    let result_elem = constructor.builder.function_call(output_elem_type, None, map_func_id, call_args)?;

    // Write result back to same location (in-place)
    constructor.builder.store(elem_ptr, result_elem, None, [])?;

    // Branch to continue
    constructor.builder.branch(continue_label)?;

    // Continue block: increment index and loop back
    constructor.builder.begin_block(Some(continue_label))?;
    let i_val_cont = constructor.builder.load(i32_type, None, loop_idx_var, None, [])?;
    let i_next = constructor.builder.i_add(i32_type, None, i_val_cont, one)?;
    constructor.builder.store(loop_idx_var, i_next, None, [])?;
    constructor.builder.branch(header_label)?;

    // Merge block
    constructor.builder.begin_block(Some(merge_label))?;

    // Return a dummy value (unit-like) - the actual output was written to the storage buffer
    // Return 0 since we can't easily create a unit constant
    Ok(zero)
}

/// Lower `_w_intrinsic_inplace_map`: in-place variant of map
///
/// For storage-backed arrays: reads/writes in-place to same buffer (side-effect)
/// For value arrays: builds new array (same as regular map)
fn lower_inplace_map(
    constructor: &mut Constructor,
    body: &Body,
    args: &[ExprId],
    expr_ty: &PolyType<TypeName>,
    result_type: spirv::Word,
) -> Result<spirv::Word> {
    if args.len() != 2 {
        bail_spirv!(
            "_w_intrinsic_inplace_map requires 2 args (function, array), got {}",
            args.len()
        );
    }

    // TODO(Phase 5): Address space is now tracked in types (Slice[elem, Storage/Function])
    // For now, use None for mem binding since we don't have LocalDecl.mem anymore
    let mem: Option<MemBinding> = None;

    let (func_name, capture_vals) = extract_closure_info(constructor, body, args[0])?;

    // Lower the input array
    let arr_val = lower_expr(constructor, body, args[1])?;
    let arr_ty = body.get_type(args[1]);
    let (array_size, elem_type) = extract_array_info(constructor, arr_ty)?;

    // Get result element type from the expression type
    let output_elem_type = match expr_ty {
        PolyType::Constructed(TypeName::Array, type_args) => {
            assert!(type_args.len() == 3);
            constructor.ast_type_to_spirv(&type_args[0])
        }
        _ => bail_spirv!("inplace_map result must be array type, got {:?}", expr_ty),
    };

    let map_func_id = *constructor
        .functions
        .get(&func_name)
        .ok_or_else(|| err_spirv!("Map function not found: {}", func_name))?;

    match mem {
        Some(mem_binding) => {
            // Storage-backed array: build index list and delegate to core
            let indices: Vec<_> = (0..array_size).map(|i| constructor.const_u32(i)).collect();
            lower_inplace_map_core(
                constructor,
                mem_binding,
                arr_val,
                elem_type,
                map_func_id,
                &capture_vals,
                output_elem_type,
                &indices,
            )?;
            // Return unit for storage ops (side-effect only)
            Ok(constructor.const_i32(0))
        }
        None => {
            // Value array: build new array (same as regular map)
            let mut result_elements = Vec::with_capacity(array_size as usize);
            for i in 0..array_size {
                let input_elem = constructor.builder.composite_extract(elem_type, None, arr_val, [i])?;
                // Order: element first, then captures (captures are trailing params after lifting)
                let mut call_args = vec![input_elem];
                call_args.extend(capture_vals.clone());
                let result_elem =
                    constructor.builder.function_call(output_elem_type, None, map_func_id, call_args)?;
                result_elements.push(result_elem);
            }
            Ok(constructor.builder.composite_construct(result_type, None, result_elements)?)
        }
    }
}

/// Lower `_w_intrinsic_zip`: zip [a,b,c] [x,y,z] = [(a,x), (b,y), (c,z)]
fn lower_zip(
    constructor: &mut Constructor,
    body: &Body,
    args: &[ExprId],
    result_type: spirv::Word,
) -> Result<spirv::Word> {
    if args.len() != 2 {
        bail_spirv!("_w_intrinsic_zip requires 2 args, got {}", args.len());
    }

    let arr1_val = lower_expr(constructor, body, args[0])?;
    let arr2_val = lower_expr(constructor, body, args[1])?;

    let arr1_ty = body.get_type(args[0]);
    let arr2_ty = body.get_type(args[1]);

    let (size1, elem1_type) = extract_array_info(constructor, arr1_ty)?;
    let (size2, elem2_type) = extract_array_info(constructor, arr2_ty)?;

    if size1 != size2 {
        bail_spirv!("zip arrays must have same size, got {} and {}", size1, size2);
    }

    // Get the pair type for elements
    let pair_type = constructor.get_or_create_struct_type(vec![elem1_type, elem2_type]);

    // Build result array of pairs
    let mut result_elements = Vec::with_capacity(size1 as usize);
    for i in 0..size1 {
        let elem1 = constructor.builder.composite_extract(elem1_type, None, arr1_val, [i])?;
        let elem2 = constructor.builder.composite_extract(elem2_type, None, arr2_val, [i])?;
        let pair = constructor.builder.composite_construct(pair_type, None, vec![elem1, elem2])?;
        result_elements.push(pair);
    }

    Ok(constructor.builder.composite_construct(result_type, None, result_elements)?)
}

/// Lower `_w_intrinsic_map_into`: map f over input and write to output buffer.
///
/// Args: [func, input_array, output_storage_array, offset]
/// Generates a loop that reads from input, applies f, writes to output[offset + i].
fn lower_map_into(constructor: &mut Constructor, body: &Body, args: &[ExprId]) -> Result<spirv::Word> {
    if args.len() != 4 {
        bail_spirv!("_w_intrinsic_map_into requires 4 args, got {}", args.len());
    }

    // Extract function info - for map_into, args[0] is the function reference
    // Unlike regular map, map_into doesn't include captures in its args
    let func_name = match body.get_expr(args[0]) {
        Expr::Global(name) => name.clone(),
        other => bail_spirv!(
            "Expected function reference (Global) in map_into, got {:?}",
            other
        ),
    };
    let map_func_id = *constructor
        .functions
        .get(&func_name)
        .ok_or_else(|| err_spirv!("Map function not found: {}", func_name))?;

    let offset_val = lower_expr(constructor, body, args[3])?;

    // Get input array type info
    let input_ty = body.get_type(args[1]);
    let (input_elem_type, output_elem_type) = match input_ty {
        PolyType::Constructed(TypeName::Array, type_args) if type_args.len() == 3 => {
            let in_elem = constructor.ast_type_to_spirv(&type_args[0]);
            // Get output element type from output array type
            let output_ty = body.get_type(args[2]);
            let out_elem = match output_ty {
                PolyType::Constructed(TypeName::Array, out_args) if out_args.len() == 3 => {
                    constructor.ast_type_to_spirv(&out_args[0])
                }
                _ => in_elem, // Same type if not specified
            };
            (in_elem, out_elem)
        }
        _ => bail_spirv!("_w_intrinsic_map_into input must be array type"),
    };

    // Get output storage buffer info
    let output_expr = body.get_expr(args[2]);
    let output_storage_name = match output_expr {
        Expr::Array {
            backing: ArrayBacking::Storage { name, .. },
            ..
        } => name.clone(),
        _ => bail_spirv!("_w_intrinsic_map_into output must be storage array"),
    };

    let (out_set, out_binding) = constructor
        .compute_params
        .get(&output_storage_name)
        .ok_or_else(|| err_spirv!("Unknown output storage buffer: {}", output_storage_name))?;
    let (out_buffer_var, out_buffer_elem_type) =
        constructor.storage_buffers.get(&(*out_set, *out_binding)).ok_or_else(|| {
            err_spirv!(
                "Output storage buffer not found for set={}, binding={}",
                out_set,
                out_binding
            )
        })?;
    let out_buffer_var = *out_buffer_var;
    let out_elem_ptr_type =
        constructor.get_or_create_ptr_type(spirv::StorageClass::StorageBuffer, *out_buffer_elem_type);

    // Get input - must be a View into storage for dynamic sizes
    let input_expr = body.get_expr(args[1]);
    let (input_base, input_offset_expr, input_size_expr) = match input_expr {
        Expr::Array {
            backing: ArrayBacking::View { base, offset },
            size,
        } => (*base, *offset, *size),
        _ => bail_spirv!("_w_intrinsic_map_into input must be a View"),
    };

    // Get the input storage buffer info
    let input_base_expr = body.get_expr(input_base);
    let input_storage_name = match input_base_expr {
        Expr::Array {
            backing: ArrayBacking::Storage { name, .. },
            ..
        } => name.clone(),
        _ => bail_spirv!("_w_intrinsic_map_into input View base must be Storage"),
    };

    let (in_set, in_binding) = constructor
        .compute_params
        .get(&input_storage_name)
        .ok_or_else(|| err_spirv!("Unknown input storage buffer: {}", input_storage_name))?;
    let (in_buffer_var, in_buffer_elem_type) =
        constructor.storage_buffers.get(&(*in_set, *in_binding)).ok_or_else(|| {
            err_spirv!(
                "Input storage buffer not found for set={}, binding={}",
                in_set,
                in_binding
            )
        })?;
    let in_buffer_var = *in_buffer_var;
    let in_elem_ptr_type =
        constructor.get_or_create_ptr_type(spirv::StorageClass::StorageBuffer, *in_buffer_elem_type);

    // Get input length (from the View's size field) and offset
    let input_len = lower_expr(constructor, body, input_size_expr)?;
    let input_offset_val = lower_expr(constructor, body, input_offset_expr)?;

    // Create loop index variable (in the function's variables block)
    let i32_type = constructor.i32_type;
    let loop_idx_var = constructor.declare_variable("__map_into_idx", i32_type)?;

    let zero = constructor.const_i32(0);
    let zero_u32 = constructor.const_u32(0);
    let one = constructor.const_i32(1);
    constructor.builder.store(loop_idx_var, zero, None, [])?;

    // Create basic blocks
    let header_label = constructor.builder.id();
    let body_label = constructor.builder.id();
    let continue_label = constructor.builder.id();
    let merge_label = constructor.builder.id();

    // Branch to header
    constructor.builder.branch(header_label)?;

    // Header block
    constructor.builder.begin_block(Some(header_label))?;
    // Compute condition before loop_merge (SPIR-V requires loop_merge immediately before branch)
    let i_val = constructor.builder.load(i32_type, None, loop_idx_var, None, [])?;
    let bool_type = constructor.bool_type;
    let cond = constructor.builder.s_less_than(bool_type, None, i_val, input_len)?;
    constructor.builder.loop_merge(merge_label, continue_label, spirv::LoopControl::NONE, [])?;
    constructor.builder.branch_conditional(cond, body_label, merge_label, [])?;

    // Body block: read from input, apply f, write to output
    constructor.builder.begin_block(Some(body_label))?;

    let i_val_body = constructor.builder.load(i32_type, None, loop_idx_var, None, [])?;

    // Read from input: input_buffer[input_offset + i]
    let in_idx = constructor.builder.i_add(i32_type, None, input_offset_val, i_val_body)?;
    let in_idx_u32 = constructor.builder.bitcast(constructor.u32_type, None, in_idx)?;
    let in_elem_ptr =
        constructor.builder.access_chain(in_elem_ptr_type, None, in_buffer_var, [zero_u32, in_idx_u32])?;
    let input_elem = constructor.builder.load(input_elem_type, None, in_elem_ptr, None, [])?;

    // Apply function: result = f(input_elem)
    // Note: map_into doesn't support captures currently - the lambda must be a simple function
    let result_elem =
        constructor.builder.function_call(output_elem_type, None, map_func_id, [input_elem])?;

    // Write to output: output_buffer[offset + i]
    let out_idx = constructor.builder.i_add(i32_type, None, offset_val, i_val_body)?;
    let out_idx_u32 = constructor.builder.bitcast(constructor.u32_type, None, out_idx)?;
    let out_elem_ptr = constructor.builder.access_chain(
        out_elem_ptr_type,
        None,
        out_buffer_var,
        [zero_u32, out_idx_u32],
    )?;
    constructor.builder.store(out_elem_ptr, result_elem, None, [])?;

    // Branch to continue
    constructor.builder.branch(continue_label)?;

    // Continue block: increment index
    constructor.builder.begin_block(Some(continue_label))?;
    let i_val_cont = constructor.builder.load(i32_type, None, loop_idx_var, None, [])?;
    let i_next = constructor.builder.i_add(i32_type, None, i_val_cont, one)?;
    constructor.builder.store(loop_idx_var, i_next, None, [])?;
    constructor.builder.branch(header_label)?;

    // Merge block
    constructor.builder.begin_block(Some(merge_label))?;

    // Return unit (void-like constant)
    Ok(constructor.const_i32(0))
}

/// Lower `_w_intrinsic_length`: length [a,b,c] = 3
fn lower_length(constructor: &mut Constructor, body: &Body, args: &[ExprId]) -> Result<spirv::Word> {
    if args.len() != 1 {
        bail_spirv!("_w_intrinsic_length requires 1 arg, got {}", args.len());
    }

    let arr_ty = body.get_type(args[0]);

    // Check if the array has a static size in its type
    let static_size = match arr_ty {
        PolyType::Constructed(TypeName::Array, type_args) if type_args.len() == 3 => match &type_args[2] {
            PolyType::Constructed(TypeName::Size(n), _) => Some(*n as i32),
            _ => None,
        },
        _ => None,
    };

    if let Some(size) = static_size {
        // Static size - return as constant
        Ok(constructor.const_i32(size))
    } else {
        // Dynamic size - check if this is a storage-backed array
        let arr_expr = body.get_expr(args[0]);
        match arr_expr {
            Expr::Array {
                backing: ArrayBacking::Storage { name, .. },
                ..
            } => {
                // Use OpArrayLength to get the runtime size
                let (set, binding) = constructor
                    .compute_params
                    .get(name)
                    .ok_or_else(|| err_spirv!("Unknown storage buffer: {}", name))?;
                let (buffer_var, _) =
                    constructor.storage_buffers.get(&(*set, *binding)).ok_or_else(|| {
                        err_spirv!("Storage buffer not found for set={}, binding={}", set, binding)
                    })?;
                // OpArrayLength returns u32, we need i32
                let u32_type = constructor.builder.type_int(32, 0);
                let len_u32 = constructor.builder.array_length(
                    u32_type,
                    None,
                    *buffer_var,
                    0, // Member index (the runtime array is at member 0)
                )?;
                // Convert to i32
                Ok(constructor.builder.bitcast(constructor.i32_type, None, len_u32)?)
            }
            _ => {
                bail_spirv!(
                    "Cannot get length of dynamically-sized non-storage array: {:?}",
                    arr_expr
                )
            }
        }
    }
}

/// Lower `_w_intrinsic_replicate`: replicate 3 x = [x, x, x]
fn lower_replicate(
    constructor: &mut Constructor,
    body: &Body,
    args: &[ExprId],
    expr_ty: &PolyType<TypeName>,
    result_type: spirv::Word,
) -> Result<spirv::Word> {
    if args.len() != 2 {
        bail_spirv!(
            "_w_intrinsic_replicate requires 2 args (size, value), got {}",
            args.len()
        );
    }

    // Get size from the result array type
    let size = match expr_ty {
        PolyType::Constructed(TypeName::Array, type_args) => {
            assert!(type_args.len() == 3);
            match &type_args[2] {
                PolyType::Constructed(TypeName::Size(n), _) => *n as u32,
                _ => bail_spirv!("replicate size must be a literal, got {:?}", type_args[2]),
            }
        }
        _ => bail_spirv!("replicate result must be array type, got {:?}", expr_ty),
    };

    let value = lower_expr(constructor, body, args[1])?;

    // Build array with repeated value
    let elements: Vec<_> = (0..size).map(|_| value).collect();
    Ok(constructor.builder.composite_construct(result_type, None, elements)?)
}

/// Lower `reduce` SOAC: reduce op ne [a,b,c] = op(op(op(ne,a),b),c)
/// Args format: [op_ref, ne, array, captures...]
fn lower_reduce(
    constructor: &mut Constructor,
    body: &Body,
    args: &[ExprId],
    result_type: spirv::Word,
) -> Result<spirv::Word> {
    if args.len() < 3 {
        bail_spirv!(
            "reduce requires at least 3 args (op, ne, array), got {}",
            args.len()
        );
    }

    // Args: [op_ref, ne, array, captures...]
    let func_name = match body.get_expr(args[0]) {
        Expr::Global(name) => name.clone(),
        other => bail_spirv!("Expected function reference (Global) in reduce, got {:?}", other),
    };
    let capture_vals: Vec<spirv::Word> =
        args[3..].iter().map(|&c| lower_expr(constructor, body, c)).collect::<Result<Vec<_>>>()?;

    let neutral_val = lower_expr(constructor, body, args[1])?;
    let array_val = lower_expr(constructor, body, args[2])?;

    let arg2_ty = body.get_type(args[2]);
    let (array_size, elem_type) = extract_array_info(constructor, arg2_ty)?;

    let op_func_id = *constructor
        .functions
        .get(&func_name)
        .ok_or_else(|| err_spirv!("Operator function not found: {}", func_name))?;

    // Sequential reduction
    let mut acc = neutral_val;
    for i in 0..array_size {
        let elem = constructor.builder.composite_extract(elem_type, None, array_val, [i])?;
        // Order: acc and elem first, then captures (captures are trailing params after lifting)
        let mut call_args = vec![acc, elem];
        call_args.extend(capture_vals.clone());
        acc = constructor.builder.function_call(result_type, None, op_func_id, call_args)?;
    }

    Ok(acc)
}

/// Lower `scan` SOAC (inclusive): scan op ne [a,b,c] = [a, op(a,b), op(op(a,b),c)]
/// Args format: [op_ref, ne, array, captures...]
fn lower_scan(
    constructor: &mut Constructor,
    body: &Body,
    args: &[ExprId],
    result_type: spirv::Word,
) -> Result<spirv::Word> {
    if args.len() < 3 {
        bail_spirv!(
            "scan requires at least 3 args (op, ne, array), got {}",
            args.len()
        );
    }

    // Args: [op_ref, ne, array, captures...]
    let func_name = match body.get_expr(args[0]) {
        Expr::Global(name) => name.clone(),
        other => bail_spirv!("Expected function reference (Global) in scan, got {:?}", other),
    };
    let capture_vals: Vec<spirv::Word> =
        args[3..].iter().map(|&c| lower_expr(constructor, body, c)).collect::<Result<Vec<_>>>()?;

    let _neutral_val = lower_expr(constructor, body, args[1])?; // Used for empty arrays
    let array_val = lower_expr(constructor, body, args[2])?;

    let arg2_ty = body.get_type(args[2]);
    let (array_size, elem_type) = extract_array_info(constructor, arg2_ty)?;

    let op_func_id = *constructor
        .functions
        .get(&func_name)
        .ok_or_else(|| err_spirv!("Operator function not found: {}", func_name))?;

    // Build result array: inclusive scan
    let mut result_elements = Vec::with_capacity(array_size as usize);

    if array_size > 0 {
        // First element is just array[0]
        let first_elem = constructor.builder.composite_extract(elem_type, None, array_val, [0])?;
        result_elements.push(first_elem);

        // Subsequent elements: acc = op(acc, array[i])
        let mut acc = first_elem;
        for i in 1..array_size {
            let elem = constructor.builder.composite_extract(elem_type, None, array_val, [i])?;
            // Order: acc and elem first, then captures (captures are trailing params after lifting)
            let mut call_args = vec![acc, elem];
            call_args.extend(capture_vals.clone());
            acc = constructor.builder.function_call(elem_type, None, op_func_id, call_args)?;
            result_elements.push(acc);
        }
    }

    Ok(constructor.builder.composite_construct(result_type, None, result_elements)?)
}

/// Lower `filter` SOAC: filter pred [a,b,c] = elements where pred is true
/// Args format: [pred_ref, array, captures...]
fn lower_filter(
    constructor: &mut Constructor,
    body: &Body,
    args: &[ExprId],
    result_type: spirv::Word,
) -> Result<spirv::Word> {
    if args.len() < 2 {
        bail_spirv!(
            "filter requires at least 2 args (pred, array), got {}",
            args.len()
        );
    }

    // Args: [pred_ref, array, captures...]
    let func_name = match body.get_expr(args[0]) {
        Expr::Global(name) => name.clone(),
        other => bail_spirv!("Expected function reference (Global) in filter, got {:?}", other),
    };
    let capture_vals: Vec<spirv::Word> =
        args[2..].iter().map(|&c| lower_expr(constructor, body, c)).collect::<Result<Vec<_>>>()?;

    let array_val = lower_expr(constructor, body, args[1])?;

    let arg1_ty = body.get_type(args[1]);
    let (array_size, elem_type) = extract_array_info(constructor, arg1_ty)?;

    let pred_func_id = *constructor
        .functions
        .get(&func_name)
        .ok_or_else(|| err_spirv!("Predicate function not found: {}", func_name))?;

    // Build cumulative count array (how many elements pass up to each index)
    let mut cumulative_counts = Vec::with_capacity(array_size as usize);
    let mut count_exprs = Vec::with_capacity(array_size as usize);

    for i in 0..array_size {
        let elem = constructor.builder.composite_extract(elem_type, None, array_val, [i])?;
        // Order: element first, then captures (captures are trailing params after lifting)
        let mut call_args = vec![elem];
        call_args.extend(capture_vals.clone());
        let pred_result =
            constructor.builder.function_call(constructor.bool_type, None, pred_func_id, call_args)?;

        // Convert bool to i32 (1 if true, 0 if false)
        let one = constructor.const_i32(1);
        let zero = constructor.const_i32(0);
        let count_inc = constructor.builder.select(constructor.i32_type, None, pred_result, one, zero)?;

        // Cumulative count
        let cumulative = if i == 0 {
            count_inc
        } else {
            constructor.builder.i_add(
                constructor.i32_type,
                None,
                cumulative_counts[i as usize - 1],
                count_inc,
            )?
        };
        cumulative_counts.push(cumulative);
        count_exprs.push((pred_result, elem));
    }

    // Build output array using select chains
    let mut output_elements = Vec::with_capacity(array_size as usize);
    let undef_elem = constructor.builder.undef(elem_type, None);

    for out_idx in 0..array_size {
        let out_idx_const = constructor.const_i32(out_idx as i32 + 1);
        let mut result_elem = undef_elem;

        for in_idx in (0..array_size).rev() {
            let (pred_result, in_elem) = count_exprs[in_idx as usize];
            let cum_count = cumulative_counts[in_idx as usize];

            // Check if this input element should go to this output slot
            let matches_slot =
                constructor.builder.i_equal(constructor.bool_type, None, cum_count, out_idx_const)?;
            let should_use =
                constructor.builder.logical_and(constructor.bool_type, None, pred_result, matches_slot)?;
            result_elem = constructor.builder.select(elem_type, None, should_use, in_elem, result_elem)?;
        }
        output_elements.push(result_elem);
    }

    // Create output array
    let cap_const = constructor.const_i32(array_size as i32);
    let data_array_type = constructor.builder.type_array(elem_type, cap_const);
    let data_array = constructor.builder.composite_construct(data_array_type, None, output_elements)?;

    // Final count is the last cumulative count
    let final_count =
        if array_size > 0 { cumulative_counts[array_size as usize - 1] } else { constructor.const_i32(0) };

    // Create OwnedSlice struct: { len: i32, data: [cap]elem }
    Ok(constructor.builder.composite_construct(result_type, None, vec![final_count, data_array])?)
}

/// Lower `scatter` SOAC: scatter dest indices values -> updated dest
fn lower_scatter(
    constructor: &mut Constructor,
    body: &Body,
    args: &[ExprId],
    result_type: spirv::Word,
) -> Result<spirv::Word> {
    if args.len() != 3 {
        bail_spirv!(
            "scatter requires 3 args (dest, indices, values), got {}",
            args.len()
        );
    }

    let dest_val = lower_expr(constructor, body, args[0])?;
    let indices_val = lower_expr(constructor, body, args[1])?;
    let values_val = lower_expr(constructor, body, args[2])?;

    let dest_ty = body.get_type(args[0]);
    let indices_ty = body.get_type(args[1]);

    let (dest_size, elem_type) = extract_array_info(constructor, dest_ty)?;
    let (scatter_size, _) = extract_array_info(constructor, indices_ty)?;

    // For each destination index, check if any scatter index points to it
    // This is O(n*m) but necessary for static unrolling
    let mut result_elements = Vec::with_capacity(dest_size as usize);

    for dest_idx in 0..dest_size {
        let dest_idx_const = constructor.const_i32(dest_idx as i32);
        let orig_elem = constructor.builder.composite_extract(elem_type, None, dest_val, [dest_idx])?;

        // Check each scatter operation (in reverse order so later writes win)
        let mut result_elem = orig_elem;
        for scatter_idx in (0..scatter_size).rev() {
            let index = constructor.builder.composite_extract(
                constructor.i32_type,
                None,
                indices_val,
                [scatter_idx],
            )?;
            let value =
                constructor.builder.composite_extract(elem_type, None, values_val, [scatter_idx])?;

            let matches =
                constructor.builder.i_equal(constructor.bool_type, None, index, dest_idx_const)?;
            result_elem = constructor.builder.select(elem_type, None, matches, value, result_elem)?;
        }
        result_elements.push(result_elem);
    }

    Ok(constructor.builder.composite_construct(result_type, None, result_elements)?)
}

/// Lower `hist_1d` SOAC: hist_1d dest op ne indices values
/// For each destination index, accumulate values using op where indices match.
fn lower_hist_1d(
    constructor: &mut Constructor,
    body: &Body,
    args: &[ExprId],
    result_type: spirv::Word,
) -> Result<spirv::Word> {
    if args.len() != 5 {
        bail_spirv!(
            "hist_1d requires 5 args (dest, op, ne, indices, values), got {}",
            args.len()
        );
    }

    let dest_val = lower_expr(constructor, body, args[0])?;
    let (func_name, capture_vals) = extract_closure_info(constructor, body, args[1])?;
    let _neutral_val = lower_expr(constructor, body, args[2])?; // Used for initialization, but we use dest values
    let indices_val = lower_expr(constructor, body, args[3])?;
    let values_val = lower_expr(constructor, body, args[4])?;

    let dest_ty = body.get_type(args[0]);
    let indices_ty = body.get_type(args[3]);

    let (dest_size, elem_type) = extract_array_info(constructor, dest_ty)?;
    let (scatter_size, _) = extract_array_info(constructor, indices_ty)?;

    let op_func_id = *constructor
        .functions
        .get(&func_name)
        .ok_or_else(|| err_spirv!("Operator function not found: {}", func_name))?;

    // For each destination index, accumulate matching values using the operator
    let mut result_elements = Vec::with_capacity(dest_size as usize);

    for dest_idx in 0..dest_size {
        let dest_idx_const = constructor.const_i32(dest_idx as i32);
        let orig_elem = constructor.builder.composite_extract(elem_type, None, dest_val, [dest_idx])?;

        // Accumulate values where indices match this destination index
        let mut acc = orig_elem;
        for scatter_idx in 0..scatter_size {
            let index = constructor.builder.composite_extract(
                constructor.i32_type,
                None,
                indices_val,
                [scatter_idx],
            )?;
            let value =
                constructor.builder.composite_extract(elem_type, None, values_val, [scatter_idx])?;

            // Check if this index matches the current destination
            let matches =
                constructor.builder.i_equal(constructor.bool_type, None, index, dest_idx_const)?;

            // Apply operator: new_acc = op(acc, value)
            // Order: acc and value first, then captures (captures are trailing params after lifting)
            let mut call_args = vec![acc, value];
            call_args.extend(capture_vals.clone());
            let combined = constructor.builder.function_call(elem_type, None, op_func_id, call_args)?;

            // Select: if matches then combined else acc
            acc = constructor.builder.select(elem_type, None, matches, combined, acc)?;
        }
        result_elements.push(acc);
    }

    Ok(constructor.builder.composite_construct(result_type, None, result_elements)?)
}

// ============================================================================
// SHA256 intrinsics
// ============================================================================

/// SHA256 K constants (first 32 bits of fractional parts of cube roots of first 64 primes)
const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// SHA256 initial hash values (first 32 bits of fractional parts of square roots of first 8 primes)
const SHA256_IV: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// Lower `_w_intrinsic_rotr32`: right rotate a u32 by n bits
fn lower_rotr32(constructor: &mut Constructor, body: &Body, args: &[ExprId]) -> Result<spirv::Word> {
    if args.len() != 2 {
        bail_spirv!("_w_intrinsic_rotr32 requires 2 args (x, n), got {}", args.len());
    }

    let x = lower_expr(constructor, body, args[0])?;
    let n = lower_expr(constructor, body, args[1])?;

    // rotr(x, n) = (x >> n) | (x << (32 - n))
    let const_32 = constructor.builder.constant_bit32(constructor.u32_type, 32);
    let complement = constructor.builder.i_sub(constructor.u32_type, None, const_32, n)?;
    let right_shift = constructor.builder.shift_right_logical(constructor.u32_type, None, x, n)?;
    let left_shift = constructor.builder.shift_left_logical(constructor.u32_type, None, x, complement)?;
    Ok(constructor.builder.bitwise_or(constructor.u32_type, None, right_shift, left_shift)?)
}

/// Lower `_w_intrinsic_sha256_block`: SHA256 compression with standard IV
fn lower_sha256_block(
    constructor: &mut Constructor,
    body: &Body,
    args: &[ExprId],
    result_type: spirv::Word,
) -> Result<spirv::Word> {
    if args.len() != 1 {
        bail_spirv!(
            "_w_intrinsic_sha256_block requires 1 arg (block), got {}",
            args.len()
        );
    }

    let block = lower_expr(constructor, body, args[0])?;

    // Build initial state from SHA256_IV constants
    let mut state_words = Vec::with_capacity(8);
    for &iv in &SHA256_IV {
        state_words.push(constructor.builder.constant_bit32(constructor.u32_type, iv));
    }
    let array_8_u32 = result_type;
    let state = constructor.builder.composite_construct(array_8_u32, None, state_words)?;

    // Call the core compression function
    lower_sha256_compress(constructor, state, block, result_type)
}

/// Lower `_w_intrinsic_sha256_block_with_state`: SHA256 compression with custom state
fn lower_sha256_block_with_state(
    constructor: &mut Constructor,
    body: &Body,
    args: &[ExprId],
    result_type: spirv::Word,
) -> Result<spirv::Word> {
    if args.len() != 2 {
        bail_spirv!(
            "_w_intrinsic_sha256_block_with_state requires 2 args (state, block), got {}",
            args.len()
        );
    }

    let state = lower_expr(constructor, body, args[0])?;
    let block = lower_expr(constructor, body, args[1])?;

    lower_sha256_compress(constructor, state, block, result_type)
}

/// Core SHA256 compression function
/// Takes state [8]u32 and block [16]u32, returns new state [8]u32
fn lower_sha256_compress(
    constructor: &mut Constructor,
    state: spirv::Word,
    block: spirv::Word,
    result_type: spirv::Word,
) -> Result<spirv::Word> {
    let u32_ty = constructor.u32_type;

    // Helper closure for right rotation
    let rotr = |c: &mut Constructor, x: spirv::Word, n: u32| -> Result<spirv::Word> {
        let n_const = c.builder.constant_bit32(u32_ty, n);
        let const_32 = c.builder.constant_bit32(u32_ty, 32);
        let complement = c.builder.i_sub(u32_ty, None, const_32, n_const)?;
        let right = c.builder.shift_right_logical(u32_ty, None, x, n_const)?;
        let left = c.builder.shift_left_logical(u32_ty, None, x, complement)?;
        Ok(c.builder.bitwise_or(u32_ty, None, right, left)?)
    };

    // Helper closure for right shift
    let shr = |c: &mut Constructor, x: spirv::Word, n: u32| -> Result<spirv::Word> {
        let n_const = c.builder.constant_bit32(u32_ty, n);
        Ok(c.builder.shift_right_logical(u32_ty, None, x, n_const)?)
    };

    // Extract initial hash values (a, b, c, d, e, f, g, h)
    let mut h = Vec::with_capacity(8);
    for i in 0..8 {
        h.push(constructor.builder.composite_extract(u32_ty, None, state, [i as u32])?);
    }

    // Extract message block words
    let mut w = Vec::with_capacity(64);
    for i in 0..16 {
        w.push(constructor.builder.composite_extract(u32_ty, None, block, [i as u32])?);
    }

    // Expand message schedule (W[16..63])
    for i in 16..64 {
        // sigma0(x) = rotr(x,7) ^ rotr(x,18) ^ shr(x,3)
        let w_i_15 = w[i - 15];
        let s0_r7 = rotr(constructor, w_i_15, 7)?;
        let s0_r18 = rotr(constructor, w_i_15, 18)?;
        let s0_s3 = shr(constructor, w_i_15, 3)?;
        let s0_tmp = constructor.builder.bitwise_xor(u32_ty, None, s0_r7, s0_r18)?;
        let s0 = constructor.builder.bitwise_xor(u32_ty, None, s0_tmp, s0_s3)?;

        // sigma1(x) = rotr(x,17) ^ rotr(x,19) ^ shr(x,10)
        let w_i_2 = w[i - 2];
        let s1_r17 = rotr(constructor, w_i_2, 17)?;
        let s1_r19 = rotr(constructor, w_i_2, 19)?;
        let s1_s10 = shr(constructor, w_i_2, 10)?;
        let s1_tmp = constructor.builder.bitwise_xor(u32_ty, None, s1_r17, s1_r19)?;
        let s1 = constructor.builder.bitwise_xor(u32_ty, None, s1_tmp, s1_s10)?;

        // W[i] = W[i-16] + s0 + W[i-7] + s1
        let w_i_16 = w[i - 16];
        let w_i_7 = w[i - 7];
        let tmp1 = constructor.builder.i_add(u32_ty, None, w_i_16, s0)?;
        let tmp2 = constructor.builder.i_add(u32_ty, None, tmp1, w_i_7)?;
        w.push(constructor.builder.i_add(u32_ty, None, tmp2, s1)?);
    }

    // Working variables
    let mut a = h[0];
    let mut b = h[1];
    let mut c = h[2];
    let mut d = h[3];
    let mut e = h[4];
    let mut f = h[5];
    let mut g = h[6];
    let mut hh = h[7];

    // 64 rounds
    for i in 0..64 {
        // Sigma1(e) = rotr(e,6) ^ rotr(e,11) ^ rotr(e,25)
        let sig1_r6 = rotr(constructor, e, 6)?;
        let sig1_r11 = rotr(constructor, e, 11)?;
        let sig1_r25 = rotr(constructor, e, 25)?;
        let sig1_tmp = constructor.builder.bitwise_xor(u32_ty, None, sig1_r6, sig1_r11)?;
        let sig1 = constructor.builder.bitwise_xor(u32_ty, None, sig1_tmp, sig1_r25)?;

        // Ch(e,f,g) = (e & f) ^ (~e & g)
        let e_and_f = constructor.builder.bitwise_and(u32_ty, None, e, f)?;
        let not_e = constructor.builder.not(u32_ty, None, e)?;
        let not_e_and_g = constructor.builder.bitwise_and(u32_ty, None, not_e, g)?;
        let ch = constructor.builder.bitwise_xor(u32_ty, None, e_and_f, not_e_and_g)?;

        // T1 = h + Sigma1(e) + Ch(e,f,g) + K[i] + W[i]
        let k_i = constructor.builder.constant_bit32(u32_ty, SHA256_K[i]);
        let t1_tmp1 = constructor.builder.i_add(u32_ty, None, hh, sig1)?;
        let t1_tmp2 = constructor.builder.i_add(u32_ty, None, t1_tmp1, ch)?;
        let t1_tmp3 = constructor.builder.i_add(u32_ty, None, t1_tmp2, k_i)?;
        let t1 = constructor.builder.i_add(u32_ty, None, t1_tmp3, w[i])?;

        // Sigma0(a) = rotr(a,2) ^ rotr(a,13) ^ rotr(a,22)
        let sig0_r2 = rotr(constructor, a, 2)?;
        let sig0_r13 = rotr(constructor, a, 13)?;
        let sig0_r22 = rotr(constructor, a, 22)?;
        let sig0_tmp = constructor.builder.bitwise_xor(u32_ty, None, sig0_r2, sig0_r13)?;
        let sig0 = constructor.builder.bitwise_xor(u32_ty, None, sig0_tmp, sig0_r22)?;

        // Maj(a,b,c) = (a & b) ^ (a & c) ^ (b & c)
        let a_and_b = constructor.builder.bitwise_and(u32_ty, None, a, b)?;
        let a_and_c = constructor.builder.bitwise_and(u32_ty, None, a, c)?;
        let b_and_c = constructor.builder.bitwise_and(u32_ty, None, b, c)?;
        let maj_tmp = constructor.builder.bitwise_xor(u32_ty, None, a_and_b, a_and_c)?;
        let maj = constructor.builder.bitwise_xor(u32_ty, None, maj_tmp, b_and_c)?;

        // T2 = Sigma0(a) + Maj(a,b,c)
        let t2 = constructor.builder.i_add(u32_ty, None, sig0, maj)?;

        // Update working variables
        hh = g;
        g = f;
        f = e;
        e = constructor.builder.i_add(u32_ty, None, d, t1)?;
        d = c;
        c = b;
        b = a;
        a = constructor.builder.i_add(u32_ty, None, t1, t2)?;
    }

    // Add working variables to original hash
    let h0_new = constructor.builder.i_add(u32_ty, None, h[0], a)?;
    let h1_new = constructor.builder.i_add(u32_ty, None, h[1], b)?;
    let h2_new = constructor.builder.i_add(u32_ty, None, h[2], c)?;
    let h3_new = constructor.builder.i_add(u32_ty, None, h[3], d)?;
    let h4_new = constructor.builder.i_add(u32_ty, None, h[4], e)?;
    let h5_new = constructor.builder.i_add(u32_ty, None, h[5], f)?;
    let h6_new = constructor.builder.i_add(u32_ty, None, h[6], g)?;
    let h7_new = constructor.builder.i_add(u32_ty, None, h[7], hh)?;

    // Construct result array
    Ok(constructor.builder.composite_construct(
        result_type,
        None,
        vec![h0_new, h1_new, h2_new, h3_new, h4_new, h5_new, h6_new, h7_new],
    )?)
}
