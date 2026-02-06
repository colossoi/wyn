//! SPIR-V code generation backend.
//!
//! This module contains the lowering pass from SSA to SPIR-V.

#[cfg(test)]
mod lowering_tests;
use std::collections::{HashMap, HashSet};

use crate::ast::TypeName;
use crate::error::Result;
use crate::impl_source::{BuiltinImpl, ImplSource, PrimOp};
use crate::mir::layout::{buffer_array_strides, type_byte_size};
use crate::mir::ssa::{Block, BlockId, ControlHeader, FuncBody, Inst, InstKind, Terminator, ValueId};
use crate::tlc::to_ssa::{ExecutionModel, IoDecoration, SsaEntryPoint, SsaFunction, SsaProgram};
use crate::types;
use crate::{bail_spirv, err_spirv};
use polytype::Type as PolyType;
use rspirv::binary::Assemble;
use rspirv::dr::{Builder, InsertPoint, Operand};
use rspirv::spirv::{self, AddressingModel, Capability, MemoryModel, StorageClass};

// =============================================================================
// Constructor - SPIR-V Builder Wrapper
// =============================================================================

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
    int_const_reverse: HashMap<spirv::Word, i32>, // reverse lookup: ID -> value
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
    array_elem_cache: HashMap<spirv::Word, spirv::Word>,   // array_type -> element_type

    // Entry point interface tracking
    entry_point_interfaces: HashMap<String, Vec<spirv::Word>>,

    // Global constants: name -> constant_id (SPIR-V OpConstant)
    global_constants: HashMap<String, spirv::Word>,
    uniform_variables: HashMap<String, spirv::Word>,
    uniform_types: HashMap<String, spirv::Word>, // uniform name -> SPIR-V type ID
    uniform_load_cache: HashMap<String, spirv::Word>, // cached OpLoad results per function

    // Storage buffer name maps (for program.storage declarations)
    storage_variables: HashMap<String, spirv::Word>, // name -> var_id
    storage_elem_types: HashMap<String, spirv::Word>, // name -> element SPIR-V type
    extract_cache: HashMap<(spirv::Word, u32), spirv::Word>, // CSE for OpCompositeExtract

    // Builtin function registry
    impl_source: ImplSource,

    /// Storage buffers for compute shaders: (set, binding) -> (buffer_var, elem_type_id, buffer_ptr_type)
    storage_buffers: HashMap<(u32, u32), (spirv::Word, spirv::Word, spirv::Word)>,

    /// GlobalInvocationId variable for compute shaders (set during entry point setup)
    global_invocation_id: Option<spirv::Word>,

    /// Linked SPIR-V functions: linkage_name -> function_id
    linked_functions: HashMap<String, spirv::Word>,

    /// Output variables for the current entry point being lowered.
    /// Set during entry point setup, cleared at end. Used by OutputPtr lowering.
    current_entry_outputs: Vec<spirv::Word>,

    /// Tracks which SPIR-V types already have ArrayStride decorations for buffer layout.
    buffer_stride_decorated: HashSet<spirv::Word>,
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
            array_elem_cache: HashMap::new(),
            entry_point_interfaces: HashMap::new(),
            global_constants: HashMap::new(),
            uniform_variables: HashMap::new(),
            uniform_types: HashMap::new(),
            uniform_load_cache: HashMap::new(),
            storage_variables: HashMap::new(),
            storage_elem_types: HashMap::new(),
            extract_cache: HashMap::new(),
            impl_source: ImplSource::default(),
            storage_buffers: HashMap::new(),
            global_invocation_id: None,
            linked_functions: HashMap::new(),
            current_entry_outputs: Vec::new(),
            buffer_stride_decorated: HashSet::new(),
        }
    }

    /// Resolve a pointer address-space type to a SPIR-V StorageClass.
    fn resolve_storage_class(addrspace: &PolyType<TypeName>) -> StorageClass {
        match addrspace {
            PolyType::Constructed(TypeName::PointerFunction, _) => StorageClass::Function,
            PolyType::Constructed(TypeName::PointerInput, _) => StorageClass::Input,
            PolyType::Constructed(TypeName::PointerOutput, _) => StorageClass::Output,
            PolyType::Constructed(TypeName::PointerStorage, _) => StorageClass::StorageBuffer,
            _ => StorageClass::Function,
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

    /// Convert a polytype Type to a SPIR-V type ID
    fn polytype_to_spirv(&mut self, ty: &PolyType<TypeName>) -> spirv::Word {
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
                            self.apply_buffer_array_strides(elem_type, &args[0]);
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
                                    let arr_type = self.builder.type_array(elem_type, size_const);
                                    // Cache element type for later lookup
                                    self.array_elem_cache.insert(arr_type, elem_type);
                                    arr_type
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
                        // Pointer type: args[0] is pointee type, args[1] is address space
                        if args.is_empty() {
                            panic!("BUG: Pointer type requires a pointee type argument.");
                        }
                        let pointee_type = self.polytype_to_spirv(&args[0]);
                        let sc = args
                            .get(1)
                            .map(Constructor::resolve_storage_class)
                            .unwrap_or(StorageClass::Function);
                        self.get_or_create_ptr_type(sc, pointee_type)
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
                    TypeName::ArrayVariantComposite
                    | TypeName::ArrayVariantView
                    | TypeName::PointerFunction
                    | TypeName::PointerInput
                    | TypeName::PointerOutput
                    | TypeName::PointerStorage => {
                        // Address space markers are used within Array/Pointer types but shouldn't appear
                        // as standalone types requiring SPIR-V representation.
                        panic!(
                            "BUG: Address space marker {:?} reached polytype_to_spirv as standalone type. \
                            This should only appear as part of Array[elem, addrspace, size] or Pointer[pointee, addrspace]. Full type: {:?}",
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

    /// Apply ArrayStride decorations for all nested fixed-size arrays in a type
    /// used inside a storage buffer. Uses layout::buffer_array_strides() for the
    /// stride values and walks nested arrays via array_elem_cache for SPIR-V IDs.
    /// Skips types that have already been decorated.
    fn apply_buffer_array_strides(&mut self, spirv_type: spirv::Word, poly_type: &PolyType<TypeName>) {
        let strides = buffer_array_strides(poly_type);
        if strides.is_empty() {
            return;
        }
        let mut current = spirv_type;
        for stride in strides {
            if !self.buffer_stride_decorated.insert(current) {
                break; // already decorated — nested types are too
            }
            self.builder.decorate(
                current,
                spirv::Decoration::ArrayStride,
                [Operand::LiteralBit32(stride)],
            );
            if let Some(&inner) = self.array_elem_cache.get(&current) {
                current = inner;
            } else {
                break;
            }
        }
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

    /// Create a storage buffer variable for compute shaders.
    /// Returns the variable ID. Also registers it in storage_buffers for later lookup.
    fn create_storage_buffer(
        &mut self,
        array_ty: &PolyType<TypeName>,
        set: u32,
        binding: u32,
    ) -> spirv::Word {
        // Extract element type from array type
        let elem_ty = match array_ty {
            PolyType::Constructed(TypeName::Array, args) if !args.is_empty() => args[0].clone(),
            _ => array_ty.clone(),
        };
        let elem_spirv = self.polytype_to_spirv(&elem_ty);

        // Calculate stride from element type size
        let stride = type_byte_size(&elem_ty).expect("storage buffer element type must have known size");

        // Ensure nested array types have ArrayStride for buffer layout
        self.apply_buffer_array_strides(elem_spirv, &elem_ty);

        // Create runtime array type (cached to avoid duplicate decorations)
        let runtime_array = self.get_or_create_runtime_array_type(elem_spirv, stride);

        // Create block struct (cached)
        let block_struct = self.get_or_create_buffer_block_type(runtime_array);

        let ptr_type = self.get_or_create_ptr_type(spirv::StorageClass::StorageBuffer, block_struct);
        let var_id = self.builder.variable(ptr_type, None, spirv::StorageClass::StorageBuffer, None);

        self.builder.decorate(
            var_id,
            spirv::Decoration::DescriptorSet,
            [Operand::LiteralBit32(set)],
        );
        self.builder.decorate(
            var_id,
            spirv::Decoration::Binding,
            [Operand::LiteralBit32(binding)],
        );

        // Store for later lookup (ptr_type used for StorageView struct construction)
        self.storage_buffers.insert((set, binding), (var_id, block_struct, ptr_type));

        var_id
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
    fn const_i32(&mut self, value: i32) -> spirv::Word {
        if let Some(&id) = self.int_const_cache.get(&value) {
            return id;
        }
        let id = self.builder.constant_bit32(self.i32_type, value as u32);
        self.int_const_cache.insert(value, id);
        self.int_const_reverse.insert(id, value);
        id
    }

    /// Get the literal i32 value from a constant ID (reverse lookup)
    fn get_const_i32_value(&self, id: spirv::Word) -> Option<i32> {
        self.int_const_reverse.get(&id).copied()
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

    /// Get the element type of an array type
    fn get_array_element_type(&self, array_type: spirv::Word) -> Result<spirv::Word> {
        self.array_elem_cache
            .get(&array_type)
            .copied()
            .ok_or_else(|| crate::err_spirv!("Array element type not found for type ID: {}", array_type))
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

    /// Begin a block (must be called before emitting instructions into it)
    fn begin_block(&mut self, block_id: spirv::Word) -> Result<()> {
        self.builder.begin_block(Some(block_id))?;
        self.current_block = Some(block_id);
        // Clear extract cache since values from previous blocks may not dominate this block
        self.extract_cache.clear();
        Ok(())
    }
}

// =============================================================================
// SSA to SPIR-V Lowering
// =============================================================================

/// Lower an SSA function body to SPIR-V.
///
/// This creates a SPIR-V function from the SSA representation:
/// - SSA blocks become SPIR-V blocks
/// - Block parameters become OpPhi nodes
/// - Terminators become branch instructions
fn lower_ssa_body(constructor: &mut Constructor, body: &FuncBody) -> Result<spirv::Word> {
    let mut ctx = SsaLowerCtx::new(constructor, body, false);
    ctx.lower()
}

/// Lower an SSA function body for an entry point.
///
/// Entry points are void functions — OpReturnValue is invalid.
/// SSA for entry points should use OutputPtr+Store then ReturnUnit;
/// Return(value) will produce an error.
fn lower_ssa_body_for_entry(constructor: &mut Constructor, body: &FuncBody) -> Result<spirv::Word> {
    let mut ctx = SsaLowerCtx::new(constructor, body, true);
    ctx.lower()
}

/// Context for lowering SSA to SPIR-V.
struct SsaLowerCtx<'a, 'b> {
    constructor: &'a mut Constructor,
    body: &'b FuncBody,
    /// True when lowering an entry point (void function — OpReturnValue is invalid).
    is_entry_point: bool,
    /// Map from SSA ValueId to SPIR-V Word.
    value_map: HashMap<ValueId, spirv::Word>,
    /// Map from SSA BlockId to SPIR-V block label.
    block_map: HashMap<BlockId, spirv::Word>,
    /// Map from block to its SPIR-V block index (for phi insertion).
    block_indices: HashMap<BlockId, usize>,
    /// Phi node info: (target_block, param_idx, value, source_block)
    /// Collected during terminator lowering, inserted after all blocks processed.
    phi_inputs: Vec<(BlockId, usize, spirv::Word, spirv::Word)>,
}

impl<'a, 'b> SsaLowerCtx<'a, 'b> {
    fn new(constructor: &'a mut Constructor, body: &'b FuncBody, is_entry_point: bool) -> Self {
        SsaLowerCtx {
            constructor,
            body,
            is_entry_point,
            value_map: HashMap::new(),
            block_map: HashMap::new(),
            block_indices: HashMap::new(),
            phi_inputs: Vec::new(),
        }
    }

    fn lower(&mut self) -> Result<spirv::Word> {
        // Map function parameters to their SPIR-V values
        for (value_id, _, name) in &self.body.params {
            if let Some(&spirv_id) = self.constructor.env.get(name) {
                self.value_map.insert(*value_id, spirv_id);
            }
        }

        // Create all SPIR-V blocks first
        for (block_idx, _) in self.body.blocks.iter().enumerate() {
            let block_id = BlockId(block_idx as u32);
            if block_idx == 0 {
                // Entry block is already created by begin_function
                let current = self.constructor.current_block.unwrap();
                self.block_map.insert(block_id, current);
            } else {
                let spirv_block = self.constructor.builder.id();
                self.block_map.insert(block_id, spirv_block);
            }
        }

        // Lower each block
        for (block_idx, block) in self.body.blocks.iter().enumerate() {
            let block_id = BlockId(block_idx as u32);

            // Start block (skip entry which is already started)
            if block_idx != 0 {
                let spirv_block = self.block_map[&block_id];
                self.constructor.begin_block(spirv_block)?;
            }

            // Record block index for phi insertion
            if let Some(idx) = self.constructor.builder.selected_block() {
                self.block_indices.insert(block_id, idx);
            }

            // Allocate phi IDs for block parameters (but don't insert yet)
            for (param_idx, param) in block.params.iter().enumerate() {
                let param_ty = self.constructor.polytype_to_spirv(&param.ty);
                let phi_id = self.constructor.builder.id();
                self.value_map.insert(param.value, phi_id);

                // Store type info for later phi insertion
                // (We'll need this when we actually insert the phi)
                let _ = (param_idx, param_ty); // Used below
            }

            // Lower instructions
            for &inst_id in &block.insts {
                let inst = self.body.get_inst(inst_id);
                self.lower_inst(inst)?;
            }

            // Lower terminator
            if let Some(ref term) = block.terminator {
                self.lower_terminator(block_id, block, term)?;
            }
        }

        // Insert phi nodes for all block parameters
        self.insert_phi_nodes()?;

        // Return placeholder - actual return handled by terminators in SSA
        Ok(self.constructor.const_i32(0))
    }

    fn lower_inst(&mut self, inst: &Inst) -> Result<()> {
        let result_ty = self.constructor.polytype_to_spirv(&inst.result_ty);

        let spirv_result = match &inst.kind {
            InstKind::Int(s) => match &inst.result_ty {
                PolyType::Constructed(TypeName::UInt(32), _) => {
                    let val: u32 = s.parse().map_err(|_| err_spirv!("Invalid u32: {}", s))?;
                    self.constructor.const_u32(val)
                }
                _ => {
                    let val: i32 = s.parse().map_err(|_| err_spirv!("Invalid i32: {}", s))?;
                    self.constructor.const_i32(val)
                }
            },

            InstKind::Float(s) => {
                let val: f32 = s.parse().map_err(|_| err_spirv!("Invalid f32: {}", s))?;
                self.constructor.const_f32(val)
            }

            InstKind::Bool(b) => self.constructor.const_bool(*b),

            InstKind::Unit => {
                unreachable!(
                    "InstKind::Unit should never reach SPIR-V codegen; unit values are not materializable"
                )
            }

            InstKind::String(s) => {
                bail_spirv!("String literals not supported in SPIR-V: {}", s)
            }

            InstKind::BinOp { op, lhs, rhs } => {
                let lhs_id = self.get_value(*lhs)?;
                let rhs_id = self.get_value(*rhs)?;
                let lhs_ty = self.body.get_value_type(*lhs);
                self.lower_binop(op, lhs_id, rhs_id, lhs_ty, result_ty)?
            }

            InstKind::UnaryOp { op, operand } => {
                let operand_id = self.get_value(*operand)?;
                let operand_ty = self.body.get_value_type(*operand);
                self.lower_unaryop(op, operand_id, operand_ty, result_ty)?
            }

            InstKind::Tuple(elems) => {
                let elem_ids: Vec<_> = elems.iter().map(|&v| self.get_value(v)).collect::<Result<_>>()?;
                self.constructor.builder.composite_construct(result_ty, None, elem_ids)?
            }

            InstKind::ArrayLit { elements } => {
                let elem_ids: Vec<_> =
                    elements.iter().map(|&v| self.get_value(v)).collect::<Result<_>>()?;
                self.constructor.builder.composite_construct(result_ty, None, elem_ids)?
            }

            InstKind::ArrayRange { start, len, step } => {
                // Virtual array represented as {start, step, len} struct
                // This matches the layout expected by lower_virtual_index
                let start_id = self.get_value(*start)?;
                let len_id = self.get_value(*len)?;
                let step_id = match step {
                    Some(s) => self.get_value(*s)?,
                    None => self.constructor.const_i32(1), // default step = 1
                };

                // Construct the struct: {start, step, len}
                self.constructor.builder.composite_construct(
                    result_ty,
                    None,
                    vec![start_id, step_id, len_id],
                )?
            }

            InstKind::Vector(elems) => {
                let elem_ids: Vec<_> = elems.iter().map(|&v| self.get_value(v)).collect::<Result<_>>()?;
                self.constructor.builder.composite_construct(result_ty, None, elem_ids)?
            }

            InstKind::Matrix(rows) => {
                // Matrix is constructed as an array of vectors (columns)
                // For now, flatten and construct
                let all_elems: Vec<_> =
                    rows.iter().flatten().map(|&v| self.get_value(v)).collect::<Result<_>>()?;
                self.constructor.builder.composite_construct(result_ty, None, all_elems)?
            }

            InstKind::Project { base, index } => {
                let base_ty = self.body.get_value_type(*base);
                let base_id = self.get_value(*base)?;

                // If base is a pointer, load it first
                let composite_id = if types::is_pointer(base_ty) {
                    let pointee_ty = types::pointee(base_ty).expect("Pointer should have pointee");
                    let value_type = self.constructor.polytype_to_spirv(pointee_ty);
                    self.constructor.builder.load(value_type, None, base_id, None, [])?
                } else {
                    base_id
                };

                self.constructor.builder.composite_extract(result_ty, None, composite_id, [*index])?
            }

            InstKind::Index { base, index } => self.lower_index(*base, *index, result_ty)?,

            InstKind::Call { func, args } => {
                let arg_ids: Vec<_> = args.iter().map(|&v| self.get_value(v)).collect::<Result<_>>()?;

                // Check if it's a builtin function first
                if let Some(builtin_impl) = self.constructor.impl_source.get(func) {
                    if func.contains("array_with") {}
                    self.lower_builtin_call(builtin_impl.clone(), &arg_ids, result_ty)?
                } else if let Some(&func_id) = self.constructor.functions.get(func) {
                    // User-defined function
                    self.constructor.builder.function_call(result_ty, None, func_id, arg_ids)?
                } else {
                    bail_spirv!("Unknown function: {}", func)
                }
            }

            InstKind::Global(name) => {
                if let Some(&const_id) = self.constructor.global_constants.get(name) {
                    const_id
                } else if let Some(&var_id) = self.constructor.uniform_variables.get(name) {
                    // Load uniform value
                    let value_type = self
                        .constructor
                        .uniform_types
                        .get(name)
                        .copied()
                        .ok_or_else(|| err_spirv!("Unknown uniform type: {}", name))?;
                    let member_ptr_type = self.constructor.builder.type_pointer(
                        None,
                        spirv::StorageClass::Uniform,
                        value_type,
                    );
                    let zero = self.constructor.const_i32(0);
                    let member_ptr =
                        self.constructor.builder.access_chain(member_ptr_type, None, var_id, [zero])?;
                    self.constructor.builder.load(value_type, None, member_ptr, None, [])?
                } else if self.constructor.storage_variables.contains_key(name) {
                    bail_spirv!(
                        "Direct global access to storage buffer '{}' is invalid; use array indexing",
                        name
                    )
                } else if let Some(&func_id) = self.constructor.functions.get(name) {
                    // Global constant function - call it with no args to get the value.
                    // This handles `def verts: [3]vec4f32 = [...]` referenced as just `verts`.
                    self.constructor.builder.function_call(result_ty, None, func_id, [])?
                } else {
                    bail_spirv!("Unknown global: {}", name)
                }
            }

            InstKind::Extern(linkage_name) => self
                .constructor
                .linked_functions
                .get(linkage_name)
                .copied()
                .ok_or_else(|| err_spirv!("Unknown extern: {}", linkage_name))?,

            InstKind::Intrinsic { name, args } => {
                let arg_ids: Vec<_> = args.iter().map(|&v| self.get_value(v)).collect::<Result<_>>()?;
                self.lower_intrinsic(name, args, &arg_ids, result_ty)?
            }

            // Effectful operations - for now, just handle the simple cases
            InstKind::Alloca { elem_ty, .. } => {
                let elem_spirv_ty = self.constructor.polytype_to_spirv(elem_ty);
                self.constructor.declare_variable("_alloca", elem_spirv_ty)?
            }

            InstKind::Load { ptr, .. } => {
                let ptr_id = self.get_value(*ptr)?;
                self.constructor.builder.load(result_ty, None, ptr_id, None, [])?
            }

            InstKind::Store { ptr, value, .. } => {
                let ptr_id = self.get_value(*ptr)?;
                let val_id = self.get_value(*value)?;
                self.constructor.builder.store(ptr_id, val_id, None, [])?;
                // Store doesn't produce a value, but we return dummy
                self.constructor.const_i32(0)
            }

            InstKind::StorageView {
                set,
                binding,
                offset,
                len,
            } => {
                let offset_id = self.get_value(*offset)?;
                let len_id = self.get_value(*len)?;

                if let Some(&(buffer_var, _, buffer_ptr_type)) =
                    self.constructor.storage_buffers.get(&(*set, *binding))
                {
                    // Use u32 for offset/len (indices and lengths are non-negative)
                    let view_struct_type = self.constructor.get_or_create_struct_type(vec![
                        buffer_ptr_type,
                        self.constructor.u32_type,
                        self.constructor.u32_type,
                    ]);
                    self.constructor.builder.composite_construct(
                        view_struct_type,
                        None,
                        [buffer_var, offset_id, len_id],
                    )?
                } else {
                    bail_spirv!("Unknown storage buffer: set={}, binding={}", set, binding)
                }
            }

            InstKind::StorageViewIndex { view, index } => {
                let view_id = self.get_value(*view)?;
                let index_id = self.get_value(*index)?;

                // Derive the buffer pointer type from the view's SSA type.
                // The view's type is an array type (e.g., []i32). From that we can compute:
                // elem_type -> runtime_array -> buffer_block -> ptr_to_buffer_block
                let view_ty = self.body.get_value_type(*view);
                let elem_ty = match view_ty {
                    PolyType::Constructed(TypeName::Array, args) if !args.is_empty() => &args[0],
                    _ => bail_spirv!("StorageViewIndex: view must have array type, got {:?}", view_ty),
                };
                let elem_spirv = self.constructor.polytype_to_spirv(elem_ty);
                let stride = type_byte_size(elem_ty)
                    .ok_or_else(|| err_spirv!("StorageViewIndex: element type must have known size"))?;
                let runtime_array = self.constructor.get_or_create_runtime_array_type(elem_spirv, stride);
                let block_struct = self.constructor.get_or_create_buffer_block_type(runtime_array);
                let buffer_ptr_type = self
                    .constructor
                    .get_or_create_ptr_type(spirv::StorageClass::StorageBuffer, block_struct);

                // Extract buffer_ptr and offset from view struct
                let buffer_ptr =
                    self.constructor.builder.composite_extract(buffer_ptr_type, None, view_id, [0u32])?;
                let base_offset = self.constructor.builder.composite_extract(
                    self.constructor.u32_type,
                    None,
                    view_id,
                    [1u32],
                )?;

                // Compute actual index: base_offset + index
                let actual_index = self.constructor.builder.i_add(
                    self.constructor.u32_type,
                    None,
                    base_offset,
                    index_id,
                )?;

                // Access chain into runtime array - returns a pointer to the element
                let zero = self.constructor.const_i32(0);
                let elem_ptr_type =
                    self.constructor.get_or_create_ptr_type(spirv::StorageClass::StorageBuffer, result_ty);
                self.constructor.builder.access_chain(
                    elem_ptr_type,
                    None,
                    buffer_ptr,
                    [zero, actual_index],
                )?
            }

            InstKind::StorageViewLen { view } => {
                let view_id = self.get_value(*view)?;
                // Extract len from view struct (index 2)
                self.constructor.builder.composite_extract(result_ty, None, view_id, [2u32])?
            }

            InstKind::OutputPtr { index } => {
                // Return the output variable pointer for this index
                if *index < self.constructor.current_entry_outputs.len() {
                    self.constructor.current_entry_outputs[*index]
                } else {
                    bail_spirv!(
                        "Output index {} out of bounds (have {} outputs)",
                        index,
                        self.constructor.current_entry_outputs.len()
                    )
                }
            }
        };

        // Map the result
        if let Some(result_value) = inst.result {
            self.value_map.insert(result_value, spirv_result);
        }

        Ok(())
    }

    fn lower_terminator(&mut self, _block_id: BlockId, block: &Block, term: &Terminator) -> Result<()> {
        let current_block = self.constructor.current_block.unwrap();

        match term {
            Terminator::Branch { target, args } => {
                // Record phi inputs for target block parameters
                for (param_idx, &arg) in args.iter().enumerate() {
                    let arg_id = self.get_value(arg)?;
                    self.phi_inputs.push((*target, param_idx, arg_id, current_block));
                }

                let target_label = self.block_map[target];
                self.constructor.builder.branch(target_label)?;
            }

            Terminator::CondBranch {
                cond,
                then_target,
                then_args,
                else_target,
                else_args,
            } => {
                let cond_id = self.get_value(*cond)?;
                let then_label = self.block_map[then_target];
                let else_label = self.block_map[else_target];

                // Record phi inputs for both targets
                for (param_idx, &arg) in then_args.iter().enumerate() {
                    let arg_id = self.get_value(arg)?;
                    self.phi_inputs.push((*then_target, param_idx, arg_id, current_block));
                }
                for (param_idx, &arg) in else_args.iter().enumerate() {
                    let arg_id = self.get_value(arg)?;
                    self.phi_inputs.push((*else_target, param_idx, arg_id, current_block));
                }

                // Emit structured control flow merge instructions if this is a header block
                if let Some(ref control) = block.control {
                    match control {
                        ControlHeader::Loop {
                            merge,
                            continue_block,
                        } => {
                            let merge_label = self.block_map[merge];
                            let continue_label = self.block_map[continue_block];
                            self.constructor.builder.loop_merge(
                                merge_label,
                                continue_label,
                                spirv::LoopControl::NONE,
                                [],
                            )?;
                        }
                        ControlHeader::Selection { merge } => {
                            let merge_label = self.block_map[merge];
                            self.constructor
                                .builder
                                .selection_merge(merge_label, spirv::SelectionControl::NONE)?;
                        }
                    }
                }

                self.constructor.builder.branch_conditional(cond_id, then_label, else_label, [])?;
            }

            Terminator::Return(value) => {
                if self.is_entry_point {
                    bail_spirv!(
                        "Return(value) in entry point body — entry points are void functions \
                         and must use OutputPtr+Store then ReturnUnit"
                    );
                }
                let value_id = self.get_value(*value)?;
                self.constructor.builder.ret_value(value_id)?;
            }

            Terminator::ReturnUnit => {
                // Always emit ret() - with OutputPtr+Store, entry points handle
                // outputs explicitly in SSA, so no special case needed here.
                self.constructor.builder.ret()?;
            }

            Terminator::Unreachable => {
                self.constructor.builder.unreachable()?;
            }
        }

        Ok(())
    }

    fn insert_phi_nodes(&mut self) -> Result<()> {
        // Group phi inputs by (target_block, param_idx)
        let mut phi_map: HashMap<(BlockId, usize), Vec<(spirv::Word, spirv::Word)>> = HashMap::new();

        for (target_block, param_idx, value, source_block) in &self.phi_inputs {
            phi_map.entry((*target_block, *param_idx)).or_default().push((*value, *source_block));
        }

        // Insert phi nodes
        for ((block_id, param_idx), incoming) in phi_map {
            let block = &self.body.blocks[block_id.index()];
            let param = &block.params[param_idx];
            let param_ty = self.constructor.polytype_to_spirv(&param.ty);

            // Get the pre-allocated phi ID
            let phi_id = self.value_map[&param.value];

            // Get block index for insertion
            if let Some(&block_idx) = self.block_indices.get(&block_id) {
                self.constructor.builder.select_block(Some(block_idx))?;
                self.constructor.builder.insert_phi(
                    InsertPoint::Begin,
                    param_ty,
                    Some(phi_id),
                    incoming,
                )?;
                self.constructor.builder.select_block(None)?;
            }
        }

        Ok(())
    }

    fn get_value(&self, value: ValueId) -> Result<spirv::Word> {
        self.value_map.get(&value).copied().ok_or_else(|| err_spirv!("Unknown SSA value: {:?}", value))
    }

    fn lower_binop(
        &mut self,
        op: &str,
        lhs: spirv::Word,
        rhs: spirv::Word,
        lhs_ty: &PolyType<TypeName>,
        result_ty: spirv::Word,
    ) -> Result<spirv::Word> {
        use PolyType::*;
        use TypeName::*;

        let bool_type = self.constructor.bool_type;

        match (op, lhs_ty) {
            // Float operations
            ("+", Constructed(Float(_), _)) => {
                Ok(self.constructor.builder.f_add(result_ty, None, lhs, rhs)?)
            }
            ("-", Constructed(Float(_), _)) => {
                Ok(self.constructor.builder.f_sub(result_ty, None, lhs, rhs)?)
            }
            ("*", Constructed(Float(_), _)) => {
                Ok(self.constructor.builder.f_mul(result_ty, None, lhs, rhs)?)
            }
            ("/", Constructed(Float(_), _)) => {
                Ok(self.constructor.builder.f_div(result_ty, None, lhs, rhs)?)
            }
            ("%", Constructed(Float(_), _)) => {
                Ok(self.constructor.builder.f_rem(result_ty, None, lhs, rhs)?)
            }
            ("<", Constructed(Float(_), _)) => {
                Ok(self.constructor.builder.f_ord_less_than(bool_type, None, lhs, rhs)?)
            }
            ("<=", Constructed(Float(_), _)) => {
                Ok(self.constructor.builder.f_ord_less_than_equal(bool_type, None, lhs, rhs)?)
            }
            (">", Constructed(Float(_), _)) => {
                Ok(self.constructor.builder.f_ord_greater_than(bool_type, None, lhs, rhs)?)
            }
            (">=", Constructed(Float(_), _)) => {
                Ok(self.constructor.builder.f_ord_greater_than_equal(bool_type, None, lhs, rhs)?)
            }
            ("==", Constructed(Float(_), _)) => {
                Ok(self.constructor.builder.f_ord_equal(bool_type, None, lhs, rhs)?)
            }
            ("!=", Constructed(Float(_), _)) => {
                Ok(self.constructor.builder.f_ord_not_equal(bool_type, None, lhs, rhs)?)
            }
            ("**", Constructed(Float(_), _)) => {
                // Power operator using GLSL pow (opcode 26)
                let glsl = self.constructor.glsl_ext_inst_id;
                let operands = vec![Operand::IdRef(lhs), Operand::IdRef(rhs)];
                Ok(self.constructor.builder.ext_inst(result_ty, None, glsl, 26, operands)?)
            }

            // Integer operations (signed)
            ("+", Constructed(Int(_), _)) => {
                Ok(self.constructor.builder.i_add(result_ty, None, lhs, rhs)?)
            }
            ("-", Constructed(Int(_), _)) => {
                Ok(self.constructor.builder.i_sub(result_ty, None, lhs, rhs)?)
            }
            ("*", Constructed(Int(_), _)) => {
                Ok(self.constructor.builder.i_mul(result_ty, None, lhs, rhs)?)
            }
            ("/", Constructed(Int(_), _)) => {
                Ok(self.constructor.builder.s_div(result_ty, None, lhs, rhs)?)
            }
            ("%", Constructed(Int(_), _)) => {
                Ok(self.constructor.builder.s_rem(result_ty, None, lhs, rhs)?)
            }
            ("<", Constructed(Int(_), _)) => {
                Ok(self.constructor.builder.s_less_than(bool_type, None, lhs, rhs)?)
            }
            ("<=", Constructed(Int(_), _)) => {
                Ok(self.constructor.builder.s_less_than_equal(bool_type, None, lhs, rhs)?)
            }
            (">", Constructed(Int(_), _)) => {
                Ok(self.constructor.builder.s_greater_than(bool_type, None, lhs, rhs)?)
            }
            (">=", Constructed(Int(_), _)) => {
                Ok(self.constructor.builder.s_greater_than_equal(bool_type, None, lhs, rhs)?)
            }
            ("==", Constructed(Int(_), _)) => {
                Ok(self.constructor.builder.i_equal(bool_type, None, lhs, rhs)?)
            }
            ("!=", Constructed(Int(_), _)) => {
                Ok(self.constructor.builder.i_not_equal(bool_type, None, lhs, rhs)?)
            }

            // Unsigned integer operations
            ("+", Constructed(UInt(_), _)) => {
                Ok(self.constructor.builder.i_add(result_ty, None, lhs, rhs)?)
            }
            ("-", Constructed(UInt(_), _)) => {
                Ok(self.constructor.builder.i_sub(result_ty, None, lhs, rhs)?)
            }
            ("*", Constructed(UInt(_), _)) => {
                Ok(self.constructor.builder.i_mul(result_ty, None, lhs, rhs)?)
            }
            ("/", Constructed(UInt(_), _)) => {
                Ok(self.constructor.builder.u_div(result_ty, None, lhs, rhs)?)
            }
            ("%", Constructed(UInt(_), _)) => {
                Ok(self.constructor.builder.u_mod(result_ty, None, lhs, rhs)?)
            }
            ("<", Constructed(UInt(_), _)) => {
                Ok(self.constructor.builder.u_less_than(bool_type, None, lhs, rhs)?)
            }
            ("<=", Constructed(UInt(_), _)) => {
                Ok(self.constructor.builder.u_less_than_equal(bool_type, None, lhs, rhs)?)
            }
            (">", Constructed(UInt(_), _)) => {
                Ok(self.constructor.builder.u_greater_than(bool_type, None, lhs, rhs)?)
            }
            (">=", Constructed(UInt(_), _)) => {
                Ok(self.constructor.builder.u_greater_than_equal(bool_type, None, lhs, rhs)?)
            }
            ("==", Constructed(UInt(_), _)) => {
                Ok(self.constructor.builder.i_equal(bool_type, None, lhs, rhs)?)
            }
            ("!=", Constructed(UInt(_), _)) => {
                Ok(self.constructor.builder.i_not_equal(bool_type, None, lhs, rhs)?)
            }

            // Boolean operations
            ("&&", Constructed(Str("bool"), _)) => {
                Ok(self.constructor.builder.logical_and(bool_type, None, lhs, rhs)?)
            }
            ("||", Constructed(Str("bool"), _)) => {
                Ok(self.constructor.builder.logical_or(bool_type, None, lhs, rhs)?)
            }
            ("==", Constructed(Str("bool"), _)) => {
                Ok(self.constructor.builder.logical_equal(bool_type, None, lhs, rhs)?)
            }
            ("!=", Constructed(Str("bool"), _)) => {
                Ok(self.constructor.builder.logical_not_equal(bool_type, None, lhs, rhs)?)
            }

            _ => bail_spirv!("Unsupported binary operation: {} on {:?}", op, lhs_ty),
        }
    }

    fn lower_unaryop(
        &mut self,
        op: &str,
        operand: spirv::Word,
        operand_ty: &PolyType<TypeName>,
        result_ty: spirv::Word,
    ) -> Result<spirv::Word> {
        use PolyType::*;
        use TypeName::*;

        match (op, operand_ty) {
            ("-", Constructed(Float(_), _)) => {
                Ok(self.constructor.builder.f_negate(result_ty, None, operand)?)
            }
            ("-", Constructed(Int(_), _)) => {
                Ok(self.constructor.builder.s_negate(result_ty, None, operand)?)
            }
            ("!", Constructed(Str("bool"), _)) => {
                Ok(self.constructor.builder.logical_not(result_ty, None, operand)?)
            }
            _ => bail_spirv!("Unsupported unary operation: {} on {:?}", op, operand_ty),
        }
    }

    fn lower_intrinsic(
        &mut self,
        name: &str,
        ssa_args: &[ValueId],
        args: &[spirv::Word],
        result_ty: spirv::Word,
    ) -> Result<spirv::Word> {
        // Common GLSL intrinsics
        let glsl = self.constructor.glsl_ext_inst_id;

        // Convert args to Operands for ext_inst
        let operands: Vec<Operand> = args.iter().map(|&id| Operand::IdRef(id)).collect();

        match name {
            "sin" => Ok(self.constructor.builder.ext_inst(result_ty, None, glsl, 13, operands)?),
            "cos" => Ok(self.constructor.builder.ext_inst(result_ty, None, glsl, 14, operands)?),
            "tan" => Ok(self.constructor.builder.ext_inst(result_ty, None, glsl, 15, operands)?),
            "sqrt" => Ok(self.constructor.builder.ext_inst(result_ty, None, glsl, 31, operands)?),
            "abs" => Ok(self.constructor.builder.ext_inst(result_ty, None, glsl, 4, operands)?),
            "floor" => Ok(self.constructor.builder.ext_inst(result_ty, None, glsl, 8, operands)?),
            "ceil" => Ok(self.constructor.builder.ext_inst(result_ty, None, glsl, 9, operands)?),
            "min" => Ok(self.constructor.builder.ext_inst(result_ty, None, glsl, 37, operands)?),
            "max" => Ok(self.constructor.builder.ext_inst(result_ty, None, glsl, 40, operands)?),
            "clamp" => Ok(self.constructor.builder.ext_inst(result_ty, None, glsl, 43, operands)?),
            "mix" => Ok(self.constructor.builder.ext_inst(result_ty, None, glsl, 46, operands)?),
            "pow" => Ok(self.constructor.builder.ext_inst(result_ty, None, glsl, 26, operands)?),
            "exp" => Ok(self.constructor.builder.ext_inst(result_ty, None, glsl, 27, operands)?),
            "log" => Ok(self.constructor.builder.ext_inst(result_ty, None, glsl, 28, operands)?),
            "dot" => {
                if args.len() != 2 {
                    bail_spirv!("dot requires 2 arguments");
                }
                Ok(self.constructor.builder.dot(result_ty, None, args[0], args[1])?)
            }
            "normalize" => Ok(self.constructor.builder.ext_inst(result_ty, None, glsl, 69, operands)?),
            "length" => Ok(self.constructor.builder.ext_inst(result_ty, None, glsl, 66, operands)?),
            "cross" => Ok(self.constructor.builder.ext_inst(result_ty, None, glsl, 68, operands)?),
            "reflect" => Ok(self.constructor.builder.ext_inst(result_ty, None, glsl, 71, operands)?),

            "_w_intrinsic_length" => {
                if args.len() != 1 {
                    bail_spirv!("_w_intrinsic_length requires 1 argument");
                }

                let arr_ty = self.body.get_value_type(ssa_args[0]);
                match arr_ty {
                    PolyType::Constructed(TypeName::Array, arr_args) if arr_args.len() >= 2 => {
                        match &arr_args[1] {
                            // View: struct {buffer_ptr, offset, len} — len at index 2
                            PolyType::Constructed(TypeName::ArrayVariantView, _) => Ok(self
                                .constructor
                                .builder
                                .composite_extract(result_ty, None, args[0], [2u32])?),
                            // Virtual (range): struct {start, step, len} — len at index 2
                            PolyType::Constructed(TypeName::ArrayVariantVirtual, _) => Ok(self
                                .constructor
                                .builder
                                .composite_extract(result_ty, None, args[0], [2u32])?),
                            // Composite: sized SPIR-V array — length is known from the type
                            PolyType::Constructed(TypeName::ArrayVariantComposite, _) => {
                                match &arr_args[2] {
                                    PolyType::Constructed(TypeName::Size(n), _) => {
                                        Ok(self.constructor.const_i32(*n as i32))
                                    }
                                    _ => {
                                        bail_spirv!("_w_intrinsic_length: composite array has unknown size")
                                    }
                                }
                            }
                            _ => {
                                bail_spirv!("_w_intrinsic_length: unknown array variant: {:?}", arr_args[1])
                            }
                        }
                    }
                    _ => bail_spirv!("_w_intrinsic_length: expected array type, got {:?}", arr_ty),
                }
            }

            "_w_slice" => {
                // Slice an array: _w_slice(arr, start, end) -> new array or view
                if args.len() != 3 {
                    bail_spirv!("_w_slice requires 3 arguments (arr, start, end)");
                }
                let arr = args[0];
                let start_id = args[1];
                let end_id = args[2];

                // Check if input is a storage view by examining SSA type
                let arr_ty = self.body.get_value_type(ssa_args[0]);
                let is_view = matches!(
                    arr_ty,
                    PolyType::Constructed(TypeName::Array, arr_args)
                        if arr_args.len() >= 2 && matches!(&arr_args[1], PolyType::Constructed(TypeName::ArrayVariantView, _))
                );

                if is_view {
                    // Slicing a view produces a new view with adjusted offset/len
                    // View struct is {buffer_ptr, offset, len}
                    let elem_ty = match arr_ty {
                        PolyType::Constructed(TypeName::Array, arr_args) => &arr_args[0],
                        _ => unreachable!(),
                    };

                    // Get buffer_ptr type for the view struct
                    let elem_spirv = self.constructor.polytype_to_spirv(elem_ty);
                    let stride = type_byte_size(elem_ty)
                        .ok_or_else(|| err_spirv!("_w_slice: view element must have known size"))?;
                    let runtime_array =
                        self.constructor.get_or_create_runtime_array_type(elem_spirv, stride);
                    let block_struct = self.constructor.get_or_create_buffer_block_type(runtime_array);
                    let buffer_ptr_type = self
                        .constructor
                        .get_or_create_ptr_type(spirv::StorageClass::StorageBuffer, block_struct);

                    // Extract components from input view
                    let buffer_ptr =
                        self.constructor.builder.composite_extract(buffer_ptr_type, None, arr, [0u32])?;
                    let base_offset = self.constructor.builder.composite_extract(
                        self.constructor.u32_type,
                        None,
                        arr,
                        [1u32],
                    )?;

                    // Compute new offset = base_offset + start
                    let new_offset = self.constructor.builder.i_add(
                        self.constructor.u32_type,
                        None,
                        base_offset,
                        start_id,
                    )?;

                    // Compute new len = end - start
                    let new_len = self.constructor.builder.i_sub(
                        self.constructor.u32_type,
                        None,
                        end_id,
                        start_id,
                    )?;

                    // Construct new view struct
                    let view_struct_type = self.constructor.get_or_create_struct_type(vec![
                        buffer_ptr_type,
                        self.constructor.u32_type,
                        self.constructor.u32_type,
                    ]);
                    Ok(self.constructor.builder.composite_construct(
                        view_struct_type,
                        None,
                        [buffer_ptr, new_offset, new_len],
                    )?)
                } else {
                    // Value array: extract elements and construct new array
                    let start =
                        self.constructor.get_const_i32_value(start_id).ok_or_else(|| {
                            err_spirv!("_w_slice: start must be a constant for value arrays")
                        })? as u32;
                    let end =
                        self.constructor.get_const_i32_value(end_id).ok_or_else(|| {
                            err_spirv!("_w_slice: end must be a constant for value arrays")
                        })? as u32;

                    if end <= start {
                        bail_spirv!("_w_slice: end ({}) must be greater than start ({})", end, start);
                    }

                    let elem_type = self.constructor.get_array_element_type(result_ty)?;
                    let mut elements = Vec::with_capacity((end - start) as usize);
                    for i in start..end {
                        let elem = self.constructor.builder.composite_extract(elem_type, None, arr, [i])?;
                        elements.push(elem);
                    }

                    Ok(self.constructor.builder.composite_construct(result_ty, None, elements)?)
                }
            }

            "_w_slice_storage_view" => {
                // Slicing storage views is not yet implemented
                bail_spirv!("_w_slice_storage_view is not yet implemented");
            }

            "__builtin_thread_id" => {
                // Load GlobalInvocationId.x as the thread ID
                let gid_var = self
                    .constructor
                    .global_invocation_id
                    .ok_or_else(|| err_spirv!("GlobalInvocationId not set for compute shader"))?;
                let uvec3_type = self.constructor.get_or_create_vec_type(self.constructor.u32_type, 3);
                let gid = self.constructor.builder.load(uvec3_type, None, gid_var, None, [])?;
                // Extract x component (flattened thread ID)
                let thread_id_u32 = self.constructor.builder.composite_extract(
                    self.constructor.u32_type,
                    None,
                    gid,
                    [0],
                )?;
                Ok(thread_id_u32)
            }

            "_w_storage_len" => {
                // Get the length of a storage buffer via OpArrayLength
                // Args: [set_id, binding_id] (as u32 constants)
                if args.len() != 2 {
                    bail_spirv!("_w_storage_len requires 2 arguments (set, binding)");
                }
                let set = self
                    .constructor
                    .uint_const_reverse
                    .get(&args[0])
                    .copied()
                    .ok_or_else(|| err_spirv!("_w_storage_len: set must be a u32 constant"))?;
                let binding = self
                    .constructor
                    .uint_const_reverse
                    .get(&args[1])
                    .copied()
                    .ok_or_else(|| err_spirv!("_w_storage_len: binding must be a u32 constant"))?;

                let &(buffer_var, _, _) =
                    self.constructor.storage_buffers.get(&(set, binding)).ok_or_else(|| {
                        err_spirv!("Storage buffer not found for set={}, binding={}", set, binding)
                    })?;

                // OpArrayLength returns u32 length of a runtime array in a struct
                // The struct is at index 0 (the buffer block), the array is member 0
                let len_u32 = self.constructor.builder.array_length(
                    self.constructor.u32_type,
                    None,
                    buffer_var,
                    0, // Member index of the runtime array in the struct
                )?;

                Ok(len_u32)
            }

            _ => bail_spirv!("Unknown intrinsic: {}", name),
        }
    }

    /// Lower an index operation, dispatching based on the array variant.
    fn lower_index(
        &mut self,
        base: ValueId,
        index: ValueId,
        result_ty: spirv::Word,
    ) -> Result<spirv::Word> {
        let base_ty = self.body.get_value_type(base);
        let base_id = self.get_value(base)?;
        let index_id = self.get_value(index)?;

        // Dispatch based on the base type
        match base_ty {
            PolyType::Constructed(TypeName::Pointer, ptr_args) => {
                // Pointer indexing: access_chain + load
                let sc = ptr_args
                    .get(1)
                    .map(Constructor::resolve_storage_class)
                    .unwrap_or(StorageClass::Function);
                let elem_ptr_type = self.constructor.get_or_create_ptr_type(sc, result_ty);
                let elem_ptr =
                    self.constructor.builder.access_chain(elem_ptr_type, None, base_id, [index_id])?;
                Ok(self.constructor.builder.load(result_ty, None, elem_ptr, None, [])?)
            }

            PolyType::Constructed(TypeName::Array, type_args) if type_args.len() == 3 => {
                let variant = &type_args[1];

                if types::is_array_variant_view(variant) {
                    // View variant: {ptr, offset, len} struct
                    self.lower_view_index(base_id, index_id, result_ty, &type_args[0])
                } else if types::is_array_variant_virtual(variant) {
                    // Virtual variant: {start, step, len} - computed array
                    self.lower_virtual_index(base_id, index_id, result_ty)
                } else {
                    // Composite variant: SPIR-V array value
                    self.lower_composite_index(base_id, index_id, result_ty, base_ty)
                }
            }

            // Vec types - use vector_extract_dynamic
            PolyType::Constructed(TypeName::Vec, _) => {
                Ok(self.constructor.builder.vector_extract_dynamic(result_ty, None, base_id, index_id)?)
            }

            _ => bail_spirv!("Index called on non-array/non-pointer type: {:?}", base_ty),
        }
    }

    /// Lower indexing into a View array ({buffer_ptr, offset, len} struct).
    fn lower_view_index(
        &mut self,
        view_id: spirv::Word,
        index_id: spirv::Word,
        result_ty: spirv::Word,
        elem_ty: &PolyType<TypeName>,
    ) -> Result<spirv::Word> {
        // View struct is {buffer_ptr, offset, len} where buffer_ptr points to buffer block.
        // Derive buffer_ptr_type from elem_ty (same logic as _w_slice).
        let elem_spirv = self.constructor.polytype_to_spirv(elem_ty);
        let stride = type_byte_size(elem_ty)
            .ok_or_else(|| err_spirv!("lower_view_index: element type must have known size"))?;
        let runtime_array = self.constructor.get_or_create_runtime_array_type(elem_spirv, stride);
        let block_struct = self.constructor.get_or_create_buffer_block_type(runtime_array);
        let buffer_ptr_type =
            self.constructor.get_or_create_ptr_type(spirv::StorageClass::StorageBuffer, block_struct);

        // Extract buffer_ptr (field 0)
        let buffer_ptr = self.constructor.builder.composite_extract(buffer_ptr_type, None, view_id, [0])?;

        // Extract offset (field 1)
        let offset_val =
            self.constructor.builder.composite_extract(self.constructor.u32_type, None, view_id, [1])?;

        // Index may be i32 from the language; reinterpret as u32 for offset arithmetic.
        // OpBitcast is the correct SPIR-V instruction for same-width integer reinterpretation.
        // Negative indices are a semantic error caught earlier in the pipeline.
        let index_u32 = self.constructor.builder.bitcast(self.constructor.u32_type, None, index_id)?;

        // Compute final index = offset + index
        let final_index =
            self.constructor.builder.i_add(self.constructor.u32_type, None, offset_val, index_u32)?;

        // Get element pointer type for access chain result
        let elem_ptr_type =
            self.constructor.get_or_create_ptr_type(spirv::StorageClass::StorageBuffer, result_ty);

        // OpAccessChain buffer_ptr[0][final_index] - [0] indexes into runtime array member
        let zero = self.constructor.const_u32(0);
        let elem_ptr =
            self.constructor.builder.access_chain(elem_ptr_type, None, buffer_ptr, [zero, final_index])?;
        Ok(self.constructor.builder.load(result_ty, None, elem_ptr, None, [])?)
    }

    /// Lower indexing into a Virtual array ({start, step, len} struct).
    fn lower_virtual_index(
        &mut self,
        range_id: spirv::Word,
        index_id: spirv::Word,
        result_ty: spirv::Word,
    ) -> Result<spirv::Word> {
        // Virtual array is {start, step, len}
        // Result = start + index * step
        let i32_type = self.constructor.i32_type;
        let start = self.constructor.builder.composite_extract(i32_type, None, range_id, [0])?;
        let step = self.constructor.builder.composite_extract(i32_type, None, range_id, [1])?;
        let offset = self.constructor.builder.i_mul(i32_type, None, index_id, step)?;
        Ok(self.constructor.builder.i_add(result_ty, None, start, offset)?)
    }

    /// Lower indexing into a Composite array (SPIR-V array value).
    fn lower_composite_index(
        &mut self,
        array_id: spirv::Word,
        index_id: spirv::Word,
        result_ty: spirv::Word,
        array_ty: &PolyType<TypeName>,
    ) -> Result<spirv::Word> {
        // If index is a compile-time constant, use OpCompositeExtract
        if let Some(literal_idx) = self.constructor.get_const_u32_value(index_id) {
            Ok(self.constructor.builder.composite_extract(result_ty, None, array_id, [literal_idx])?)
        } else {
            // Runtime index - must materialize to local variable
            let spirv_array_type = self.constructor.polytype_to_spirv(array_ty);
            let array_var = self.constructor.declare_variable("_w_index_tmp", spirv_array_type)?;
            self.constructor.builder.store(array_var, array_id, None, [])?;

            let elem_ptr_type =
                self.constructor.builder.type_pointer(None, StorageClass::Function, result_ty);
            let elem_ptr =
                self.constructor.builder.access_chain(elem_ptr_type, None, array_var, [index_id])?;
            Ok(self.constructor.builder.load(result_ty, None, elem_ptr, None, [])?)
        }
    }

    fn lower_builtin_call(
        &mut self,
        builtin: BuiltinImpl,
        arg_ids: &[spirv::Word],
        result_ty: spirv::Word,
    ) -> Result<spirv::Word> {
        match builtin {
            BuiltinImpl::PrimOp(prim_op) => self.lower_primop(prim_op, arg_ids, result_ty),
            BuiltinImpl::LinkedSpirv(linkage_name) => {
                let func_id = self
                    .constructor
                    .linked_functions
                    .get(&linkage_name)
                    .copied()
                    .ok_or_else(|| err_spirv!("Unknown linked function: {}", linkage_name))?;
                Ok(self.constructor.builder.function_call(result_ty, None, func_id, arg_ids.to_vec())?)
            }
            BuiltinImpl::Intrinsic(intrinsic) => {
                use crate::impl_source::Intrinsic;
                match intrinsic {
                    Intrinsic::Placeholder => {
                        bail_spirv!("Placeholder intrinsic should not reach lowering")
                    }
                    Intrinsic::Uninit => {
                        // Uninitialized value - return undef
                        Ok(self.constructor.builder.undef(result_ty, None))
                    }
                    Intrinsic::ArrayWith => {
                        // _w_array_with(array, index, value) - functional array update
                        if arg_ids.len() != 3 {
                            bail_spirv!("ArrayWith requires 3 arguments");
                        }
                        // Create copy of array with element at index replaced
                        let arr = arg_ids[0];
                        let idx = arg_ids[1];
                        let val = arg_ids[2];

                        // Try to get literal index for compile-time known indices
                        if let Some(literal_idx) = self.constructor.get_const_i32_value(idx) {
                            Ok(self.constructor.builder.composite_insert(
                                result_ty,
                                None,
                                val,
                                arr,
                                [literal_idx as u32],
                            )?)
                        } else {
                            // Runtime index - need to use copy-modify pattern via local variable
                            let arr_var =
                                self.constructor.declare_variable("_array_with_tmp", result_ty)?;
                            self.constructor.builder.store(arr_var, arr, None, [])?;

                            // Get element type from array type.
                            // This will fail for virtual arrays (ranges) which can't be modified.
                            let elem_ty = self.constructor.get_array_element_type(result_ty)
                                .map_err(|_| crate::err_spirv!(
                                    "ArrayWith requires a concrete array type, not a virtual array (range). \
                                     Map over ranges produces virtual results; consider using a fixed-size array instead."
                                ))?;
                            let elem_ptr_ty = self.constructor.builder.type_pointer(
                                None,
                                spirv::StorageClass::Function,
                                elem_ty,
                            );
                            let elem_ptr =
                                self.constructor.builder.access_chain(elem_ptr_ty, None, arr_var, [idx])?;
                            self.constructor.builder.store(elem_ptr, val, None, [])?;
                            Ok(self.constructor.builder.load(result_ty, None, arr_var, None, [])?)
                        }
                    }
                }
            }
        }
    }

    fn lower_primop(
        &mut self,
        prim_op: PrimOp,
        arg_ids: &[spirv::Word],
        result_ty: spirv::Word,
    ) -> Result<spirv::Word> {
        let glsl = self.constructor.glsl_ext_inst_id;
        let operands: Vec<Operand> = arg_ids.iter().map(|&id| Operand::IdRef(id)).collect();

        match prim_op {
            PrimOp::GlslExt(ext_op) => {
                Ok(self.constructor.builder.ext_inst(result_ty, None, glsl, ext_op, operands)?)
            }
            PrimOp::Dot => {
                if arg_ids.len() != 2 {
                    bail_spirv!("dot requires 2 args");
                }
                Ok(self.constructor.builder.dot(result_ty, None, arg_ids[0], arg_ids[1])?)
            }
            PrimOp::MatrixTimesMatrix => {
                if arg_ids.len() != 2 {
                    bail_spirv!("matrix × matrix requires 2 args");
                }
                Ok(
                    self.constructor
                        .builder
                        .matrix_times_matrix(result_ty, None, arg_ids[0], arg_ids[1])?,
                )
            }
            PrimOp::MatrixTimesVector => {
                if arg_ids.len() != 2 {
                    bail_spirv!("matrix × vector requires 2 args");
                }
                Ok(
                    self.constructor
                        .builder
                        .matrix_times_vector(result_ty, None, arg_ids[0], arg_ids[1])?,
                )
            }
            PrimOp::VectorTimesMatrix => {
                if arg_ids.len() != 2 {
                    bail_spirv!("vector × matrix requires 2 args");
                }
                Ok(
                    self.constructor
                        .builder
                        .vector_times_matrix(result_ty, None, arg_ids[0], arg_ids[1])?,
                )
            }
            PrimOp::FPToSI => {
                if arg_ids.len() != 1 {
                    bail_spirv!("FPToSI requires 1 arg");
                }
                Ok(self.constructor.builder.convert_f_to_s(result_ty, None, arg_ids[0])?)
            }
            PrimOp::FPToUI => {
                if arg_ids.len() != 1 {
                    bail_spirv!("FPToUI requires 1 arg");
                }
                Ok(self.constructor.builder.convert_f_to_u(result_ty, None, arg_ids[0])?)
            }
            PrimOp::SIToFP => {
                if arg_ids.len() != 1 {
                    bail_spirv!("SIToFP requires 1 arg");
                }
                Ok(self.constructor.builder.convert_s_to_f(result_ty, None, arg_ids[0])?)
            }
            PrimOp::UIToFP => {
                if arg_ids.len() != 1 {
                    bail_spirv!("UIToFP requires 1 arg");
                }
                Ok(self.constructor.builder.convert_u_to_f(result_ty, None, arg_ids[0])?)
            }
            PrimOp::Bitcast => {
                if arg_ids.len() != 1 {
                    bail_spirv!("Bitcast requires 1 arg");
                }
                Ok(self.constructor.builder.bitcast(result_ty, None, arg_ids[0])?)
            }
            // Additional arithmetic ops
            PrimOp::FAdd | PrimOp::FSub | PrimOp::FMul | PrimOp::FDiv | PrimOp::FRem | PrimOp::FMod => {
                if arg_ids.len() != 2 {
                    bail_spirv!("Float binary op requires 2 args");
                }
                match prim_op {
                    PrimOp::FAdd => {
                        Ok(self.constructor.builder.f_add(result_ty, None, arg_ids[0], arg_ids[1])?)
                    }
                    PrimOp::FSub => {
                        Ok(self.constructor.builder.f_sub(result_ty, None, arg_ids[0], arg_ids[1])?)
                    }
                    PrimOp::FMul => {
                        Ok(self.constructor.builder.f_mul(result_ty, None, arg_ids[0], arg_ids[1])?)
                    }
                    PrimOp::FDiv => {
                        Ok(self.constructor.builder.f_div(result_ty, None, arg_ids[0], arg_ids[1])?)
                    }
                    PrimOp::FRem => {
                        Ok(self.constructor.builder.f_rem(result_ty, None, arg_ids[0], arg_ids[1])?)
                    }
                    PrimOp::FMod => {
                        Ok(self.constructor.builder.f_mod(result_ty, None, arg_ids[0], arg_ids[1])?)
                    }
                    _ => unreachable!(),
                }
            }
            PrimOp::IAdd
            | PrimOp::ISub
            | PrimOp::IMul
            | PrimOp::SDiv
            | PrimOp::UDiv
            | PrimOp::SRem
            | PrimOp::SMod
            | PrimOp::UMod => {
                if arg_ids.len() != 2 {
                    bail_spirv!("Integer binary op requires 2 args");
                }
                match prim_op {
                    PrimOp::IAdd => {
                        Ok(self.constructor.builder.i_add(result_ty, None, arg_ids[0], arg_ids[1])?)
                    }
                    PrimOp::ISub => {
                        Ok(self.constructor.builder.i_sub(result_ty, None, arg_ids[0], arg_ids[1])?)
                    }
                    PrimOp::IMul => {
                        Ok(self.constructor.builder.i_mul(result_ty, None, arg_ids[0], arg_ids[1])?)
                    }
                    PrimOp::SDiv => {
                        Ok(self.constructor.builder.s_div(result_ty, None, arg_ids[0], arg_ids[1])?)
                    }
                    PrimOp::UDiv => {
                        Ok(self.constructor.builder.u_div(result_ty, None, arg_ids[0], arg_ids[1])?)
                    }
                    PrimOp::SRem => {
                        Ok(self.constructor.builder.s_rem(result_ty, None, arg_ids[0], arg_ids[1])?)
                    }
                    PrimOp::SMod => {
                        Ok(self.constructor.builder.s_mod(result_ty, None, arg_ids[0], arg_ids[1])?)
                    }
                    PrimOp::UMod => {
                        Ok(self.constructor.builder.u_mod(result_ty, None, arg_ids[0], arg_ids[1])?)
                    }
                    _ => unreachable!(),
                }
            }
            // Comparison ops
            PrimOp::FOrdEqual
            | PrimOp::FOrdNotEqual
            | PrimOp::FOrdLessThan
            | PrimOp::FOrdGreaterThan
            | PrimOp::FOrdLessThanEqual
            | PrimOp::FOrdGreaterThanEqual => {
                if arg_ids.len() != 2 {
                    bail_spirv!("Float comparison requires 2 args");
                }
                match prim_op {
                    PrimOp::FOrdEqual => Ok(self
                        .constructor
                        .builder
                        .f_ord_equal(result_ty, None, arg_ids[0], arg_ids[1])?),
                    PrimOp::FOrdNotEqual => Ok(self
                        .constructor
                        .builder
                        .f_ord_not_equal(result_ty, None, arg_ids[0], arg_ids[1])?),
                    PrimOp::FOrdLessThan => Ok(self
                        .constructor
                        .builder
                        .f_ord_less_than(result_ty, None, arg_ids[0], arg_ids[1])?),
                    PrimOp::FOrdGreaterThan => Ok(self
                        .constructor
                        .builder
                        .f_ord_greater_than(result_ty, None, arg_ids[0], arg_ids[1])?),
                    PrimOp::FOrdLessThanEqual => Ok(self
                        .constructor
                        .builder
                        .f_ord_less_than_equal(result_ty, None, arg_ids[0], arg_ids[1])?),
                    PrimOp::FOrdGreaterThanEqual => Ok(self
                        .constructor
                        .builder
                        .f_ord_greater_than_equal(result_ty, None, arg_ids[0], arg_ids[1])?),
                    _ => unreachable!(),
                }
            }
            PrimOp::IEqual
            | PrimOp::INotEqual
            | PrimOp::SLessThan
            | PrimOp::ULessThan
            | PrimOp::SGreaterThan
            | PrimOp::UGreaterThan
            | PrimOp::SLessThanEqual
            | PrimOp::ULessThanEqual
            | PrimOp::SGreaterThanEqual
            | PrimOp::UGreaterThanEqual => {
                if arg_ids.len() != 2 {
                    bail_spirv!("Integer comparison requires 2 args");
                }
                match prim_op {
                    PrimOp::IEqual => {
                        Ok(self.constructor.builder.i_equal(result_ty, None, arg_ids[0], arg_ids[1])?)
                    }
                    PrimOp::INotEqual => Ok(self
                        .constructor
                        .builder
                        .i_not_equal(result_ty, None, arg_ids[0], arg_ids[1])?),
                    PrimOp::SLessThan => Ok(self
                        .constructor
                        .builder
                        .s_less_than(result_ty, None, arg_ids[0], arg_ids[1])?),
                    PrimOp::ULessThan => Ok(self
                        .constructor
                        .builder
                        .u_less_than(result_ty, None, arg_ids[0], arg_ids[1])?),
                    PrimOp::SGreaterThan => Ok(self
                        .constructor
                        .builder
                        .s_greater_than(result_ty, None, arg_ids[0], arg_ids[1])?),
                    PrimOp::UGreaterThan => Ok(self
                        .constructor
                        .builder
                        .u_greater_than(result_ty, None, arg_ids[0], arg_ids[1])?),
                    PrimOp::SLessThanEqual => Ok(self
                        .constructor
                        .builder
                        .s_less_than_equal(result_ty, None, arg_ids[0], arg_ids[1])?),
                    PrimOp::ULessThanEqual => Ok(self
                        .constructor
                        .builder
                        .u_less_than_equal(result_ty, None, arg_ids[0], arg_ids[1])?),
                    PrimOp::SGreaterThanEqual => Ok(self
                        .constructor
                        .builder
                        .s_greater_than_equal(result_ty, None, arg_ids[0], arg_ids[1])?),
                    PrimOp::UGreaterThanEqual => Ok(self
                        .constructor
                        .builder
                        .u_greater_than_equal(result_ty, None, arg_ids[0], arg_ids[1])?),
                    _ => unreachable!(),
                }
            }
            // Bitwise ops
            PrimOp::BitwiseAnd | PrimOp::BitwiseOr | PrimOp::BitwiseXor => {
                if arg_ids.len() != 2 {
                    bail_spirv!("Bitwise binary op requires 2 args");
                }
                match prim_op {
                    PrimOp::BitwiseAnd => Ok(self
                        .constructor
                        .builder
                        .bitwise_and(result_ty, None, arg_ids[0], arg_ids[1])?),
                    PrimOp::BitwiseOr => {
                        Ok(self.constructor.builder.bitwise_or(result_ty, None, arg_ids[0], arg_ids[1])?)
                    }
                    PrimOp::BitwiseXor => Ok(self
                        .constructor
                        .builder
                        .bitwise_xor(result_ty, None, arg_ids[0], arg_ids[1])?),
                    _ => unreachable!(),
                }
            }
            PrimOp::Not => {
                if arg_ids.len() != 1 {
                    bail_spirv!("Not requires 1 arg");
                }
                Ok(self.constructor.builder.not(result_ty, None, arg_ids[0])?)
            }
            PrimOp::ShiftLeftLogical | PrimOp::ShiftRightArithmetic | PrimOp::ShiftRightLogical => {
                if arg_ids.len() != 2 {
                    bail_spirv!("Shift op requires 2 args");
                }
                match prim_op {
                    PrimOp::ShiftLeftLogical => Ok(self
                        .constructor
                        .builder
                        .shift_left_logical(result_ty, None, arg_ids[0], arg_ids[1])?),
                    PrimOp::ShiftRightArithmetic => Ok(self
                        .constructor
                        .builder
                        .shift_right_arithmetic(result_ty, None, arg_ids[0], arg_ids[1])?),
                    PrimOp::ShiftRightLogical => Ok(self
                        .constructor
                        .builder
                        .shift_right_logical(result_ty, None, arg_ids[0], arg_ids[1])?),
                    _ => unreachable!(),
                }
            }
            // Additional type conversions
            PrimOp::FPConvert => {
                if arg_ids.len() != 1 {
                    bail_spirv!("FPConvert requires 1 arg");
                }
                Ok(self.constructor.builder.f_convert(result_ty, None, arg_ids[0])?)
            }
            PrimOp::SConvert => {
                if arg_ids.len() != 1 {
                    bail_spirv!("SConvert requires 1 arg");
                }
                Ok(self.constructor.builder.s_convert(result_ty, None, arg_ids[0])?)
            }
            PrimOp::UConvert => {
                if arg_ids.len() != 1 {
                    bail_spirv!("UConvert requires 1 arg");
                }
                Ok(self.constructor.builder.u_convert(result_ty, None, arg_ids[0])?)
            }
            // Additional matrix/vector ops
            PrimOp::OuterProduct => {
                if arg_ids.len() != 2 {
                    bail_spirv!("OuterProduct requires 2 args");
                }
                Ok(self.constructor.builder.outer_product(result_ty, None, arg_ids[0], arg_ids[1])?)
            }
            PrimOp::VectorTimesScalar => {
                if arg_ids.len() != 2 {
                    bail_spirv!("VectorTimesScalar requires 2 args");
                }
                Ok(
                    self.constructor
                        .builder
                        .vector_times_scalar(result_ty, None, arg_ids[0], arg_ids[1])?,
                )
            }
            PrimOp::MatrixTimesScalar => {
                if arg_ids.len() != 2 {
                    bail_spirv!("MatrixTimesScalar requires 2 args");
                }
                Ok(
                    self.constructor
                        .builder
                        .matrix_times_scalar(result_ty, None, arg_ids[0], arg_ids[1])?,
                )
            }
        }
    }
}

// =============================================================================
// SSA Program Lowering (new direct path)
// =============================================================================

/// Lower an SSA program directly to SPIR-V.
///
/// This is the new direct path: TLC → SSA → SPIR-V, bypassing MIR.
pub fn lower_ssa_program(program: &SsaProgram) -> Result<Vec<u32>> {
    // Use a thread with larger stack size for complex shaders
    const STACK_SIZE: usize = 16 * 1024 * 1024; // 16MB

    let program_clone = program.clone();

    let handle = std::thread::Builder::new()
        .stack_size(STACK_SIZE)
        .spawn(move || lower_ssa_program_impl(&program_clone))
        .expect("Failed to spawn lowering thread");

    handle.join().expect("Lowering thread panicked")
}

fn lower_ssa_program_impl(program: &SsaProgram) -> Result<Vec<u32>> {
    let mut constructor = Constructor::new();

    // Collect entry point info for later
    let mut entry_info: Vec<(String, spirv::ExecutionModel, Option<(u32, u32, u32)>)> = Vec::new();

    // Lower uniforms
    for uniform in &program.uniforms {
        let value_type = constructor.polytype_to_spirv(&uniform.ty);
        let block_type = constructor.create_uniform_block_type(value_type);
        let ptr_type = constructor.get_or_create_ptr_type(spirv::StorageClass::Uniform, block_type);
        let var_id = constructor.builder.variable(ptr_type, None, spirv::StorageClass::Uniform, None);

        constructor.builder.decorate(
            var_id,
            spirv::Decoration::DescriptorSet,
            [Operand::LiteralBit32(uniform.set)],
        );
        constructor.builder.decorate(
            var_id,
            spirv::Decoration::Binding,
            [Operand::LiteralBit32(uniform.binding)],
        );

        constructor.uniform_variables.insert(uniform.name.clone(), var_id);
        constructor.uniform_types.insert(uniform.name.clone(), value_type);
    }

    // Lower storage buffers (using proper Block-decorated runtime array wrapping)
    for storage in &program.storage {
        let var_id = constructor.create_storage_buffer(&storage.ty, storage.set, storage.binding);
        let elem_ty = match &storage.ty {
            PolyType::Constructed(TypeName::Array, args) if !args.is_empty() => &args[0],
            _ => &storage.ty,
        };
        let elem_spirv = constructor.polytype_to_spirv(elem_ty);
        constructor.storage_variables.insert(storage.name.clone(), var_id);
        constructor.storage_elem_types.insert(storage.name.clone(), elem_spirv);
    }

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

    // Now lower all function bodies
    for func in &program.functions {
        if func.linkage_name.is_some() {
            // Extern function - skip for now (handled at call sites)
            continue;
        }

        lower_ssa_function(&mut constructor, func)?;
    }

    // Lower all entry points
    for entry in &program.entry_points {
        let (spirv_model, local_size) = match &entry.execution_model {
            ExecutionModel::Vertex => (spirv::ExecutionModel::Vertex, None),
            ExecutionModel::Fragment => (spirv::ExecutionModel::Fragment, None),
            ExecutionModel::Compute { local_size } => (spirv::ExecutionModel::GLCompute, Some(*local_size)),
        };

        entry_info.push((entry.name.clone(), spirv_model, local_size));
        lower_ssa_entry_point(&mut constructor, entry)?;
    }

    // Emit entry point declarations
    for (name, model, local_size) in &entry_info {
        if let Some(&func_id) = constructor.functions.get(name) {
            let mut interfaces = constructor.entry_point_interfaces.get(name).cloned().unwrap_or_default();

            // Add all uniform variables to the interface
            for &uniform_var in constructor.uniform_variables.values() {
                if !interfaces.contains(&uniform_var) {
                    interfaces.push(uniform_var);
                }
            }

            // Add all storage buffer variables to the interface
            for &storage_var in constructor.storage_variables.values() {
                if !interfaces.contains(&storage_var) {
                    interfaces.push(storage_var);
                }
            }

            constructor.builder.entry_point(*model, func_id, name, interfaces);

            // Add execution modes
            match model {
                spirv::ExecutionModel::Fragment => {
                    constructor.builder.execution_mode(func_id, spirv::ExecutionMode::OriginUpperLeft, []);
                }
                spirv::ExecutionModel::GLCompute => {
                    constructor.builder.capability(Capability::VariablePointersStorageBuffer);
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

    Ok(constructor.builder.module().assemble())
}

/// Lower an SSA function to SPIR-V.
fn lower_ssa_function(constructor: &mut Constructor, func: &SsaFunction) -> Result<()> {
    let body = &func.body;

    // Extract parameter types and names, converting types to SPIR-V
    let param_names: Vec<&str> = body.params.iter().map(|(_, _, name)| name.as_str()).collect();
    let param_types: Vec<spirv::Word> =
        body.params.iter().map(|(_, ty, _)| constructor.polytype_to_spirv(ty)).collect();

    let return_type = constructor.polytype_to_spirv(&body.return_ty);

    constructor.begin_function(&func.name, &param_names, &param_types, return_type)?;
    lower_ssa_body(constructor, body)?;
    constructor.end_function()?;

    Ok(())
}

/// Lower an SSA entry point to SPIR-V.
fn lower_ssa_entry_point(constructor: &mut Constructor, entry: &SsaEntryPoint) -> Result<()> {
    let body = &entry.body;
    let is_compute = matches!(entry.execution_model, ExecutionModel::Compute { .. });

    // Create I/O variables for entry point
    let mut interfaces = Vec::new();

    // For compute shaders, automatically create GlobalInvocationId if not already present
    if is_compute && constructor.global_invocation_id.is_none() {
        let uvec3_type = constructor.get_or_create_vec_type(constructor.u32_type, 3);
        let ptr_type = constructor.get_or_create_ptr_type(spirv::StorageClass::Input, uvec3_type);
        let gid_var = constructor.builder.variable(ptr_type, None, spirv::StorageClass::Input, None);
        constructor.builder.decorate(
            gid_var,
            spirv::Decoration::BuiltIn,
            [Operand::BuiltIn(spirv::BuiltIn::GlobalInvocationId)],
        );
        constructor.global_invocation_id = Some(gid_var);
        interfaces.push(gid_var);
    }

    // Handle inputs
    let mut location = 0u32;
    for input in &entry.inputs {
        let input_type = constructor.polytype_to_spirv(&input.ty);

        if let Some(IoDecoration::BuiltIn(builtin)) = &input.decoration {
            // Built-in input
            let ptr_type = constructor.get_or_create_ptr_type(spirv::StorageClass::Input, input_type);
            let var_id = constructor.builder.variable(ptr_type, None, spirv::StorageClass::Input, None);
            constructor.builder.decorate(var_id, spirv::Decoration::BuiltIn, [Operand::BuiltIn(*builtin)]);
            constructor.env.insert(input.name.clone(), var_id);
            interfaces.push(var_id);

            // Track GlobalInvocationId for compute shaders
            if *builtin == spirv::BuiltIn::GlobalInvocationId {
                constructor.global_invocation_id = Some(var_id);
            }
        } else if let Some((set, binding)) = input.storage_binding {
            let var_id = constructor.create_storage_buffer(&input.ty, set, binding);
            interfaces.push(var_id);
        } else {
            // Regular input with location
            let loc = input
                .decoration
                .as_ref()
                .and_then(|d| match d {
                    IoDecoration::Location(l) => Some(*l),
                    _ => None,
                })
                .unwrap_or_else(|| {
                    let l = location;
                    location += 1;
                    l
                });

            let ptr_type = constructor.get_or_create_ptr_type(spirv::StorageClass::Input, input_type);
            let var_id = constructor.builder.variable(ptr_type, None, spirv::StorageClass::Input, None);
            constructor.builder.decorate(var_id, spirv::Decoration::Location, [Operand::LiteralBit32(loc)]);
            constructor.env.insert(input.name.clone(), var_id);
            interfaces.push(var_id);
        }
    }

    // Handle outputs
    let mut output_vars = Vec::new();
    let mut output_location = 0u32;
    for output in &entry.outputs {
        if let Some((set, binding)) = output.storage_binding {
            let var_id = constructor.create_storage_buffer(&output.ty, set, binding);
            interfaces.push(var_id);
            // Don't add to output_vars - storage buffers are accessed differently
        } else if let Some(IoDecoration::BuiltIn(builtin)) = &output.decoration {
            let output_type = constructor.polytype_to_spirv(&output.ty);
            let ptr_type = constructor.get_or_create_ptr_type(spirv::StorageClass::Output, output_type);
            let var_id = constructor.builder.variable(ptr_type, None, spirv::StorageClass::Output, None);
            constructor.builder.decorate(var_id, spirv::Decoration::BuiltIn, [Operand::BuiltIn(*builtin)]);
            output_vars.push(var_id);
            interfaces.push(var_id);
        } else {
            let output_type = constructor.polytype_to_spirv(&output.ty);
            let loc = output
                .decoration
                .as_ref()
                .and_then(|d| match d {
                    IoDecoration::Location(l) => Some(*l),
                    _ => None,
                })
                .unwrap_or_else(|| {
                    let l = output_location;
                    output_location += 1;
                    l
                });

            let ptr_type = constructor.get_or_create_ptr_type(spirv::StorageClass::Output, output_type);
            let var_id = constructor.builder.variable(ptr_type, None, spirv::StorageClass::Output, None);
            constructor.builder.decorate(var_id, spirv::Decoration::Location, [Operand::LiteralBit32(loc)]);
            output_vars.push(var_id);
            interfaces.push(var_id);
        }
    }

    // Store interfaces for entry point declaration
    constructor.entry_point_interfaces.insert(entry.name.clone(), interfaces);

    // Set output variables for OutputPtr lowering
    constructor.current_entry_outputs = output_vars;

    // Begin void function for entry point (no parameters - I/O is via variables)
    let void_type = constructor.void_type;
    let param_names: Vec<&str> = Vec::new();
    let param_types: Vec<spirv::Word> = Vec::new();
    constructor.begin_function(&entry.name, &param_names, &param_types, void_type)?;

    // Load input values from their pointer variables.
    // Entry point inputs are SPIR-V Input variables (pointers), but the SSA body
    // expects loaded values. Load them now and update env with the loaded values.
    for input in &entry.inputs {
        // Skip storage buffers - they use different access patterns
        if input.storage_binding.is_some() {
            continue;
        }
        let input_type = constructor.polytype_to_spirv(&input.ty);
        if let Some(&var_id) = constructor.env.get(&input.name) {
            let loaded = constructor.builder.load(input_type, None, var_id, None, [])?;
            constructor.env.insert(input.name.clone(), loaded);
        }
    }

    // Lower the body (stores to outputs are now explicit in SSA)
    // ReturnUnit blocks now emit ret() directly, so no extra return needed here.
    let _result = lower_ssa_body_for_entry(constructor, body)?;

    constructor.end_function()?;

    // Clear output variables
    constructor.current_entry_outputs.clear();

    Ok(())
}
