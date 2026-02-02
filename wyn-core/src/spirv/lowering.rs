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

    /// Storage buffers for compute shaders: (set, binding) -> (buffer_var, elem_type_id, buffer_ptr_type)
    storage_buffers: HashMap<(u32, u32), (spirv::Word, spirv::Word, spirv::Word)>,

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
        let func_id =
            self.builder.begin_function(return_type, None, spirv::FunctionControl::NONE, func_type)?;

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
                        // Array[elem, variant, size]
                        assert!(args.len() == 3);
                        let elem_type = self.ast_type_to_spirv(&args[0]);
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
                    TypeName::ArrayVariantComposite | TypeName::ArrayVariantView => {
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
    dps_output: Option<LocalId>,
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

    // Use ret() for void functions (including DPS functions), ret_value() for functions that return a value
    // DPS functions have dps_output set and already wrote to the output buffer, so they just return
    if matches!(ret_type, PolyType::Constructed(TypeName::Unit, _)) || dps_output.is_some() {
        constructor.builder.ret()?;
    } else {
        constructor.builder.ret_value(result)?;
    }

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
        params_to_lower.iter().map(|&p| constructor.ast_type_to_spirv(&body.get_local(p).ty)).collect();
    let return_type = constructor.ast_type_to_spirv(ret_type);

    // Create function type
    let func_type = constructor.builder.type_function(return_type, param_types.iter().copied());

    // Create the function declaration with no body
    // Use Pure function control to indicate no side effects
    let func_id =
        constructor.builder.begin_function(return_type, None, spirv::FunctionControl::NONE, func_type)?;

    // Add function parameters (required even for imported functions)
    for &param_id in params_to_lower {
        let param_ty = constructor.ast_type_to_spirv(&body.get_local(param_id).ty);
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

/// Get the buffer pointer type for a view expression based on its MIR type.
/// The MIR type is Array<ElemType, view, Size> - we extract ElemType and derive the pointer type.
fn get_view_buffer_ptr_type(
    constructor: &mut Constructor,
    view_ty: &PolyType<TypeName>,
) -> Result<spirv::Word> {
    match view_ty {
        PolyType::Constructed(TypeName::Array, args) if args.len() >= 1 => {
            let elem_ty = &args[0];
            let elem_type_id = constructor.ast_type_to_spirv(elem_ty);

            // Calculate stride from element type
            let stride = crate::mir::layout::type_byte_size(elem_ty).ok_or_else(|| {
                err_spirv!(
                    "get_view_buffer_ptr_type: cannot compute stride for {:?}",
                    elem_ty
                )
            })?;

            // Build buffer struct type: struct { ElemType data[]; }
            let runtime_array_type = constructor.get_or_create_runtime_array_type(elem_type_id, stride);
            let buffer_struct_type = constructor.get_or_create_buffer_block_type(runtime_array_type);

            // Get pointer to buffer struct
            Ok(constructor.get_or_create_ptr_type(spirv::StorageClass::StorageBuffer, buffer_struct_type))
        }
        _ => bail_spirv!("get_view_buffer_ptr_type: expected Array type, got {:?}", view_ty),
    }
}

/// Try to get buffer_ptr_type for an expression by checking if it's a compute parameter.
/// Returns Some(buffer_ptr_type) if the expression refers to a storage buffer via compute_params,
/// None if we can't resolve it (in which case caller should use MIR type).
fn try_get_buffer_ptr_type_from_storage(
    constructor: &Constructor,
    body: &Body,
    expr_id: ExprId,
) -> Option<spirv::Word> {
    // Check if expression is a Local that refers to a compute param
    if let Expr::Local(local_id) = body.get_expr(expr_id) {
        let name = &body.get_local(*local_id).name;
        if let Some(&(set, binding)) = constructor.compute_params.get(name) {
            if let Some(&(_, _, buffer_ptr_type)) = constructor.storage_buffers.get(&(set, binding)) {
                return Some(buffer_ptr_type);
            }
        }
    }
    None
}

/// Try to get the storage buffer (set, binding) for an expression.
/// Returns Some((set, binding)) if the expression refers to a storage buffer via compute_params.
fn get_storage_buffer_binding(
    constructor: &Constructor,
    body: &Body,
    expr_id: ExprId,
) -> Option<(u32, u32)> {
    if let Expr::Local(local_id) = body.get_expr(expr_id) {
        let name = &body.get_local(*local_id).name;
        if let Some(&(set, binding)) = constructor.compute_params.get(name) {
            return Some((set, binding));
        }
    }
    None
}

fn lower_expr(constructor: &mut Constructor, body: &Body, expr_id: ExprId) -> Result<spirv::Word> {
    let expr_ty = body.get_type(expr_id);

    // Debug: check for address space types being used as expression types
    if let PolyType::Constructed(TypeName::ArrayVariantComposite | TypeName::ArrayVariantView, _) = expr_ty
    {
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
                // This is a storage buffer parameter - construct a view struct {buffer_ptr, offset=0, len}
                if let Some(&(buffer_var, _, buffer_ptr_type)) =
                    constructor.storage_buffers.get(&(set, binding))
                {
                    // Get array length via OpArrayLength
                    let array_len =
                        constructor.builder.array_length(constructor.u32_type, None, buffer_var, 0)?;
                    let zero = constructor.const_u32(0);

                    // Construct view struct {buffer_ptr, offset, len}
                    let view_struct_type = constructor.get_or_create_struct_type(vec![
                        buffer_ptr_type,
                        constructor.u32_type,
                        constructor.u32_type,
                    ]);
                    return Ok(constructor.builder.composite_construct(
                        view_struct_type,
                        None,
                        [buffer_var, zero, array_len],
                    )?);
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

        Expr::Extern(linkage_name) => {
            // Look up the externally-linked function by its linkage name
            if let Some(&func_id) = constructor.linked_functions.get(linkage_name) {
                return Ok(func_id);
            }
            Err(err_spirv!(
                "Undefined external function with linkage name: {}",
                linkage_name
            ))
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
            let arg_types: Vec<spirv::Word> =
                args.iter().map(|&a| constructor.ast_type_to_spirv(body.get_type(a))).collect();

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
                                Ok(constructor.builder.function_call(
                                    result_type,
                                    None,
                                    func_id,
                                    arg_ids,
                                )?)
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
                "_w_storage_ptr" => {
                    // Get pointer to first element of a storage buffer
                    // Args: (set: i32, binding: i32)
                    if args.len() != 2 {
                        bail_spirv!(
                            "_w_storage_ptr requires 2 args (set, binding), got {}",
                            args.len()
                        );
                    }
                    // Extract set and binding from integer literals
                    let set = match body.get_expr(args[0]) {
                        Expr::Int(s) => s
                            .parse::<u32>()
                            .map_err(|_| err_spirv!("_w_storage_ptr: set must be integer literal"))?,
                        _ => bail_spirv!("_w_storage_ptr: set must be integer literal"),
                    };
                    let binding = match body.get_expr(args[1]) {
                        Expr::Int(s) => s
                            .parse::<u32>()
                            .map_err(|_| err_spirv!("_w_storage_ptr: binding must be integer literal"))?,
                        _ => bail_spirv!("_w_storage_ptr: binding must be integer literal"),
                    };
                    if let Some(&(buffer_var, elem_type_id, _)) =
                        constructor.storage_buffers.get(&(set, binding))
                    {
                        let elem_ptr_type = constructor
                            .get_or_create_ptr_type(spirv::StorageClass::StorageBuffer, elem_type_id);
                        let zero = constructor.const_i32(0);
                        // OpAccessChain to get pointer to first element: buffer[0][0]
                        Ok(constructor.builder.access_chain(
                            elem_ptr_type,
                            None,
                            buffer_var,
                            [zero, zero],
                        )?)
                    } else {
                        bail_spirv!(
                            "_w_storage_ptr: no storage buffer at set={}, binding={}",
                            set,
                            binding
                        )
                    }
                }
                "_w_storage_len" => {
                    // Get runtime length of a storage buffer via OpArrayLength
                    // Args: (set: i32, binding: i32)
                    if args.len() != 2 {
                        bail_spirv!(
                            "_w_storage_len requires 2 args (set, binding), got {}",
                            args.len()
                        );
                    }
                    // Extract set and binding from integer literals
                    let set = match body.get_expr(args[0]) {
                        Expr::Int(s) => s
                            .parse::<u32>()
                            .map_err(|_| err_spirv!("_w_storage_len: set must be integer literal"))?,
                        _ => bail_spirv!("_w_storage_len: set must be integer literal"),
                    };
                    let binding = match body.get_expr(args[1]) {
                        Expr::Int(s) => s
                            .parse::<u32>()
                            .map_err(|_| err_spirv!("_w_storage_len: binding must be integer literal"))?,
                        _ => bail_spirv!("_w_storage_len: binding must be integer literal"),
                    };
                    if let Some(&(buffer_var, _, _)) = constructor.storage_buffers.get(&(set, binding)) {
                        // OpArrayLength: get length of runtime array at member index 0
                        // SPIR-V requires OpArrayLength to return u32
                        Ok(constructor.builder.array_length(constructor.u32_type, None, buffer_var, 0)?)
                    } else {
                        bail_spirv!(
                            "_w_storage_len: no storage buffer at set={}, binding={}",
                            set,
                            binding
                        )
                    }
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

        // --- Storage view operations ---
        // StorageView is a buffer slice descriptor: {ptr, offset, len}
        // - ptr: pointer to the buffer struct (from storage_buffers)
        // - offset: element offset from start of buffer (u32)
        // - len: number of elements in this view (u32)
        // The buffer pointer type is derived from the expression's MIR type.
        Expr::StorageView {
            set,
            binding,
            offset,
            len,
        } => {
            // Look up buffer from storage_buffers
            let &(buffer_var, _, buffer_ptr_type) = constructor
                .storage_buffers
                .get(&(*set, *binding))
                .ok_or_else(|| err_spirv!("StorageView: no buffer at set={}, binding={}", set, binding))?;

            // Lower offset and length expressions
            let offset_val = lower_expr(constructor, body, *offset)?;
            let len_val = lower_expr(constructor, body, *len)?;

            // Cast offset to u32 if needed (it may be i32)
            let offset_u32 = constructor.builder.bitcast(constructor.u32_type, None, offset_val)?;

            // View struct type: {buffer_ptr_type, u32, u32}
            let view_struct_type = constructor.get_or_create_struct_type(vec![
                buffer_ptr_type,
                constructor.u32_type,
                constructor.u32_type,
            ]);

            Ok(constructor.builder.composite_construct(
                view_struct_type,
                None,
                [buffer_var, offset_u32, len_val],
            )?)
        }

        Expr::SliceStorageView { view, start, len } => {
            // Slice creates a new view with same buffer ptr, adjusted offset/len
            // Get buffer pointer type from the view being sliced (not the result type,
            // as MIR types may be incorrect for output buffers)
            let view_ty = body.get_type(*view);
            let buffer_ptr_type = get_view_buffer_ptr_type(constructor, view_ty)?;

            let view_val = lower_expr(constructor, body, *view)?;
            let start_val = lower_expr(constructor, body, *start)?;
            let len_val = lower_expr(constructor, body, *len)?;

            // Extract fields from view struct {ptr, offset, len}
            let ptr_val = constructor.builder.composite_extract(buffer_ptr_type, None, view_val, [0])?;
            let old_offset =
                constructor.builder.composite_extract(constructor.u32_type, None, view_val, [1])?;

            // Cast start to u32 if needed
            let start_u32 = constructor.builder.bitcast(constructor.u32_type, None, start_val)?;

            // new_offset = old_offset + start
            let new_offset =
                constructor.builder.i_add(constructor.u32_type, None, old_offset, start_u32)?;

            // Cast len to u32 if needed
            let len_u32 = constructor.builder.bitcast(constructor.u32_type, None, len_val)?;

            // Create new view struct
            let view_struct_type = constructor.get_or_create_struct_type(vec![
                buffer_ptr_type,
                constructor.u32_type,
                constructor.u32_type,
            ]);
            Ok(constructor.builder.composite_construct(
                view_struct_type,
                None,
                [ptr_val, new_offset, len_u32],
            )?)
        }

        Expr::StorageViewIndex { view, index } => {
            // Get pointer to element: OpAccessChain buffer_ptr[0][offset + index]
            // Get buffer pointer type from the view's MIR type
            let view_ty = body.get_type(*view);
            let buffer_ptr_type = get_view_buffer_ptr_type(constructor, view_ty)?;

            let view_val = lower_expr(constructor, body, *view)?;
            let index_val = lower_expr(constructor, body, *index)?;

            // Extract ptr, offset from view struct {ptr, offset, len}
            let buffer_ptr = constructor.builder.composite_extract(buffer_ptr_type, None, view_val, [0])?;
            let offset_val =
                constructor.builder.composite_extract(constructor.u32_type, None, view_val, [1])?;

            // Cast index to u32 if needed (for add)
            let index_u32 = constructor.builder.bitcast(constructor.u32_type, None, index_val)?;

            // Compute final index = offset + index
            let final_index =
                constructor.builder.i_add(constructor.u32_type, None, offset_val, index_u32)?;

            // Get element pointer type from result type
            let elem_spirv_type = constructor.ast_type_to_spirv(expr_ty);
            let elem_ptr_type =
                constructor.get_or_create_ptr_type(spirv::StorageClass::StorageBuffer, elem_spirv_type);

            // OpAccessChain buffer_ptr[0][final_index]
            let zero = constructor.const_u32(0);
            Ok(constructor.builder.access_chain(elem_ptr_type, None, buffer_ptr, [zero, final_index])?)
        }

        Expr::StorageViewLen { view } => {
            // Extract length from view struct {ptr, offset, len} - field 2
            let view_val = lower_expr(constructor, body, *view)?;
            Ok(constructor.builder.composite_extract(constructor.u32_type, None, view_val, [2])?)
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
            let variant = &type_args[1];

            // Dispatch on array variant
            if let PolyType::Constructed(TypeName::ArrayVariantView, _) = variant {
                // View variant: {ptr, len} struct - extract ptr and use PtrAccessChain
                lower_view_index(constructor, body, array_expr_id, index_val, result_type)
            } else if let PolyType::Constructed(TypeName::ArrayVariantVirtual, _) = variant {
                // Virtual variant: {start, step, len} - compute start + index * step
                lower_virtual_index(constructor, body, array_expr_id, index_val, result_type)
            } else {
                // Composite variant: SPIR-V array value
                lower_composite_index(constructor, body, array_expr_id, index_val, result_type, arg0_ty)
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

/// Lower indexing into a View array ({buffer_ptr, offset, len} struct).
fn lower_view_index(
    constructor: &mut Constructor,
    body: &Body,
    array_expr_id: ExprId,
    index_val: spirv::Word,
    result_type: spirv::Word,
) -> Result<spirv::Word> {
    // Get buffer pointer type from the view's MIR type
    let view_ty = body.get_type(array_expr_id);
    let buffer_ptr_type = get_view_buffer_ptr_type(constructor, view_ty)?;

    // View has {buffer_ptr, offset, len}
    let view_val = lower_expr(constructor, body, array_expr_id)?;

    // Extract buffer_ptr and offset
    let buffer_ptr = constructor.builder.composite_extract(buffer_ptr_type, None, view_val, [0])?;
    let offset_val = constructor.builder.composite_extract(constructor.u32_type, None, view_val, [1])?;

    // Cast index to u32 if needed (index_val may be i32)
    let index_u32 = constructor.builder.bitcast(constructor.u32_type, None, index_val)?;

    // Compute final index = offset + index
    let final_index = constructor.builder.i_add(constructor.u32_type, None, offset_val, index_u32)?;

    // Get element pointer type
    let elem_ptr_type = constructor.get_or_create_ptr_type(spirv::StorageClass::StorageBuffer, result_type);

    // OpAccessChain buffer_ptr[0][final_index]
    let zero = constructor.const_u32(0);
    let elem_ptr =
        constructor.builder.access_chain(elem_ptr_type, None, buffer_ptr, [zero, final_index])?;
    Ok(constructor.builder.load(result_type, None, elem_ptr, None, [])?)
}

/// Lower indexing into a Virtual array ({start, step, len} struct).
fn lower_virtual_index(
    constructor: &mut Constructor,
    body: &Body,
    array_expr_id: ExprId,
    index_val: spirv::Word,
    result_type: spirv::Word,
) -> Result<spirv::Word> {
    let array_expr = body.get_expr(array_expr_id);

    // Check if we can optimize based on the backing
    match array_expr {
        Expr::Array {
            backing: ArrayBacking::Range { start, step, .. },
            ..
        } => {
            // Direct range - compute start + index * step
            let start_val = lower_expr(constructor, body, *start)?;
            let i32_type = constructor.i32_type;

            if let Some(step_expr) = step {
                let step_val = lower_expr(constructor, body, *step_expr)?;
                let offset = constructor.builder.i_mul(i32_type, None, index_val, step_val)?;
                Ok(constructor.builder.i_add(i32_type, None, start_val, offset)?)
            } else {
                // Step is 1
                Ok(constructor.builder.i_add(i32_type, None, start_val, index_val)?)
            }
        }
        _ => {
            // General case: array value is a {start, step, len} struct
            let range_val = lower_expr(constructor, body, array_expr_id)?;
            let i32_type = constructor.i32_type;
            let start = constructor.builder.composite_extract(i32_type, None, range_val, [0])?;
            let step = constructor.builder.composite_extract(i32_type, None, range_val, [1])?;
            let offset = constructor.builder.i_mul(i32_type, None, index_val, step)?;
            Ok(constructor.builder.i_add(result_type, None, start, offset)?)
        }
    }
}

/// Lower indexing into a Composite array (SPIR-V array value).
fn lower_composite_index(
    constructor: &mut Constructor,
    body: &Body,
    array_expr_id: ExprId,
    index_val: spirv::Word,
    result_type: spirv::Word,
    array_type: &PolyType<TypeName>,
) -> Result<spirv::Word> {
    let array_val = lower_expr(constructor, body, array_expr_id)?;

    // If index is a compile-time constant, use OpCompositeExtract
    if let Some(literal_idx) = constructor.get_const_u32_value(index_val) {
        Ok(constructor.builder.composite_extract(result_type, None, array_val, [literal_idx])?)
    } else {
        // Runtime index - must materialize to local variable
        let spirv_array_type = constructor.ast_type_to_spirv(array_type);
        let array_var = constructor.declare_variable("_w_index_tmp", spirv_array_type)?;
        constructor.builder.store(array_var, array_val, None, [])?;

        let elem_ptr_type = constructor.builder.type_pointer(None, StorageClass::Function, result_type);
        let elem_ptr = constructor.builder.access_chain(elem_ptr_type, None, array_var, [index_val])?;
        Ok(constructor.builder.load(result_type, None, elem_ptr, None, [])?)
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
            // Partial application to a lifted lambda: (_w_lambda_N arg1 arg2 ...)
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
            let &(buffer_var, _, _) =
                constructor.storage_buffers.get(&(set, binding)).ok_or_else(|| {
                    err_spirv!("Storage buffer not found for set={}, binding={}", set, binding)
                })?;

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

    // Check if this is a dynamic-sized storage array (View, parameter, etc.)
    // Storage arrays always have their length available at runtime via fat pointer.
    let arr_expr = body.get_expr(arr_expr_id);
    let is_dynamic_storage = static_size.is_none()
        && matches!(
            arr_ty,
            PolyType::Constructed(TypeName::Array, args)
                if args.len() == 3 && matches!(&args[1], PolyType::Constructed(TypeName::ArrayVariantView, _))
        );

    if is_dynamic_storage {
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

    // Check if input array has a virtual backing (Range)
    // If so, use read_elem directly to avoid materialization
    let use_read_elem = matches!(
        arr_expr,
        Expr::Array {
            backing: ArrayBacking::Range { .. },
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
///
/// Handles two cases:
/// 1. Explicit View into Storage structure (fast path with direct buffer access)
/// 2. Any View-typed expression including function parameters (general path using {ptr, len} struct)
fn lower_map_dynamic(
    constructor: &mut Constructor,
    body: &Body,
    arr_expr_id: ExprId,
    map_func_id: spirv::Word,
    capture_vals: &[spirv::Word],
    elem_type: spirv::Word,
    output_elem_type: spirv::Word,
) -> Result<spirv::Word> {
    let i32_type = constructor.i32_type;
    let u32_type = constructor.u32_type;

    // Get buffer pointer type from MIR array type
    let arr_ty = body.get_type(arr_expr_id);
    let buffer_ptr_type = get_view_buffer_ptr_type(constructor, arr_ty)?;

    // Lower array expression to get {buffer_ptr, offset, len} struct
    let view_val = lower_expr(constructor, body, arr_expr_id)?;
    let buffer_ptr = constructor.builder.composite_extract(buffer_ptr_type, None, view_val, [0])?;
    let base_offset = constructor.builder.composite_extract(u32_type, None, view_val, [1])?;
    let size_u32 = constructor.builder.composite_extract(u32_type, None, view_val, [2])?;
    let size_val = constructor.builder.bitcast(i32_type, None, size_u32)?;

    // Element pointer type for OpAccessChain
    let elem_ptr_type = constructor.get_or_create_ptr_type(spirv::StorageClass::StorageBuffer, elem_type);

    // Create loop index variable
    let loop_idx_var = constructor.declare_variable("__map_idx", i32_type)?;

    // Initialize loop index to 0
    let zero_i32 = constructor.const_i32(0);
    let zero_u32 = constructor.const_u32(0);
    let one = constructor.const_i32(1);
    constructor.builder.store(loop_idx_var, zero_i32, None, [])?;

    // Create basic blocks
    let header_label = constructor.builder.id();
    let body_label = constructor.builder.id();
    let continue_label = constructor.builder.id();
    let merge_label = constructor.builder.id();

    // Branch to header
    constructor.builder.branch(header_label)?;

    // Header block with loop merge
    constructor.builder.begin_block(Some(header_label))?;
    let i_val = constructor.builder.load(i32_type, None, loop_idx_var, None, [])?;
    let bool_type = constructor.bool_type;
    let cond = constructor.builder.s_less_than(bool_type, None, i_val, size_val)?;
    constructor.builder.loop_merge(merge_label, continue_label, spirv::LoopControl::NONE, [])?;
    constructor.builder.branch_conditional(cond, body_label, merge_label, [])?;

    // Body block: read, apply function, write
    constructor.builder.begin_block(Some(body_label))?;

    let i_val_body = constructor.builder.load(i32_type, None, loop_idx_var, None, [])?;
    let i_u32 = constructor.builder.bitcast(u32_type, None, i_val_body)?;

    // Access element at buffer_ptr[0][base_offset + i]
    let actual_idx = constructor.builder.i_add(u32_type, None, base_offset, i_u32)?;
    let elem_ptr =
        constructor.builder.access_chain(elem_ptr_type, None, buffer_ptr, [zero_u32, actual_idx])?;
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

    // Return the View struct {buffer_ptr, offset, len} - since map is in-place, it's the same view
    let view_struct_type = constructor.get_or_create_struct_type(vec![buffer_ptr_type, u32_type, u32_type]);
    let result_view = constructor.builder.composite_construct(
        view_struct_type,
        None,
        [buffer_ptr, base_offset, size_u32],
    )?;
    Ok(result_view)
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

    let i32_type = constructor.i32_type;
    let u32_type = constructor.u32_type;

    // Extract function info
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

    let write_offset_val = lower_expr(constructor, body, args[3])?;

    // Get type info
    let input_ty = body.get_type(args[1]);
    let output_ty = body.get_type(args[2]);

    // Get input element type from MIR
    let input_elem_type = match input_ty {
        PolyType::Constructed(TypeName::Array, type_args) if type_args.len() >= 1 => {
            constructor.ast_type_to_spirv(&type_args[0])
        }
        _ => bail_spirv!("_w_intrinsic_map_into input must be array type"),
    };

    // Get output element type - prefer from storage buffer (MIR type may be incorrect)
    let output_elem_type =
        if let Some((set, binding)) = get_storage_buffer_binding(constructor, body, args[2]) {
            // Found storage buffer binding - use its registered element type
            if let Some(&(_, elem_type_id, _)) = constructor.storage_buffers.get(&(set, binding)) {
                elem_type_id
            } else {
                // Fall back to MIR type
                match output_ty {
                    PolyType::Constructed(TypeName::Array, out_args) if !out_args.is_empty() => {
                        constructor.ast_type_to_spirv(&out_args[0])
                    }
                    _ => input_elem_type,
                }
            }
        } else {
            // No storage buffer - use MIR type
            match output_ty {
                PolyType::Constructed(TypeName::Array, out_args) if !out_args.is_empty() => {
                    constructor.ast_type_to_spirv(&out_args[0])
                }
                _ => input_elem_type,
            }
        };

    // Get buffer pointer types
    // For input, use MIR type
    let in_buffer_ptr_type = get_view_buffer_ptr_type(constructor, input_ty)?;
    // For output, try to get from storage buffer (MIR type may be incorrect for output)
    let out_buffer_ptr_type = match try_get_buffer_ptr_type_from_storage(constructor, body, args[2]) {
        Some(ptr_type) => ptr_type,
        None => get_view_buffer_ptr_type(constructor, output_ty)?,
    };

    // Lower input View and extract {buffer_ptr, offset, len}
    let input_view = lower_expr(constructor, body, args[1])?;
    let in_buffer_ptr = constructor.builder.composite_extract(in_buffer_ptr_type, None, input_view, [0])?;
    let in_base_offset = constructor.builder.composite_extract(u32_type, None, input_view, [1])?;
    let input_len_u32 = constructor.builder.composite_extract(u32_type, None, input_view, [2])?;
    let input_len = constructor.builder.bitcast(i32_type, None, input_len_u32)?;

    // Lower output View and extract {buffer_ptr, offset, len}
    let output_view = lower_expr(constructor, body, args[2])?;
    let out_buffer_ptr =
        constructor.builder.composite_extract(out_buffer_ptr_type, None, output_view, [0])?;
    let out_base_offset = constructor.builder.composite_extract(u32_type, None, output_view, [1])?;

    // Element pointer types for OpAccessChain
    let in_elem_ptr_type =
        constructor.get_or_create_ptr_type(spirv::StorageClass::StorageBuffer, input_elem_type);
    let out_elem_ptr_type =
        constructor.get_or_create_ptr_type(spirv::StorageClass::StorageBuffer, output_elem_type);

    // Create loop index variable
    let loop_idx_var = constructor.declare_variable("__map_into_idx", i32_type)?;

    let zero_i32 = constructor.const_i32(0);
    let zero_u32 = constructor.const_u32(0);
    let one = constructor.const_i32(1);
    constructor.builder.store(loop_idx_var, zero_i32, None, [])?;

    // Create basic blocks
    let header_label = constructor.builder.id();
    let body_label = constructor.builder.id();
    let continue_label = constructor.builder.id();
    let merge_label = constructor.builder.id();

    // Branch to header
    constructor.builder.branch(header_label)?;

    // Header block
    constructor.builder.begin_block(Some(header_label))?;
    let i_val = constructor.builder.load(i32_type, None, loop_idx_var, None, [])?;
    let bool_type = constructor.bool_type;
    let cond = constructor.builder.s_less_than(bool_type, None, i_val, input_len)?;
    constructor.builder.loop_merge(merge_label, continue_label, spirv::LoopControl::NONE, [])?;
    constructor.builder.branch_conditional(cond, body_label, merge_label, [])?;

    // Body block: read from input, apply f, write to output
    constructor.builder.begin_block(Some(body_label))?;

    let i_val_body = constructor.builder.load(i32_type, None, loop_idx_var, None, [])?;
    let i_u32 = constructor.builder.bitcast(u32_type, None, i_val_body)?;

    // Read from input: buffer_ptr[0][in_base_offset + i]
    let in_actual_idx = constructor.builder.i_add(u32_type, None, in_base_offset, i_u32)?;
    let in_elem_ptr = constructor.builder.access_chain(
        in_elem_ptr_type,
        None,
        in_buffer_ptr,
        [zero_u32, in_actual_idx],
    )?;
    let input_elem = constructor.builder.load(input_elem_type, None, in_elem_ptr, None, [])?;

    // Apply function: result = f(input_elem)
    let result_elem =
        constructor.builder.function_call(output_elem_type, None, map_func_id, [input_elem])?;

    // Write to output: buffer_ptr[0][out_base_offset + write_offset + i]
    let write_offset_u32 = constructor.builder.bitcast(u32_type, None, write_offset_val)?;
    let out_idx_partial = constructor.builder.i_add(u32_type, None, out_base_offset, write_offset_u32)?;
    let out_actual_idx = constructor.builder.i_add(u32_type, None, out_idx_partial, i_u32)?;
    let out_elem_ptr = constructor.builder.access_chain(
        out_elem_ptr_type,
        None,
        out_buffer_ptr,
        [zero_u32, out_actual_idx],
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

    // Return unit
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
        // Dynamic size - lower array to get View {ptr, len} and extract length
        // View struct is {ptr, u32_len} - extract and cast to i32
        let view_val = lower_expr(constructor, body, args[0])?;
        let len_u32 = constructor.builder.composite_extract(constructor.u32_type, None, view_val, [1])?;
        let len = constructor.builder.bitcast(constructor.i32_type, None, len_u32)?;
        Ok(len)
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
