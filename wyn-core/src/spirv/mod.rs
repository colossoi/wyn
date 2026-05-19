//! SPIR-V code generation backend.
//!
//! This module contains the lowering pass from SSA to SPIR-V.

#[cfg(test)]
mod lowering_tests;
use crate::builtins::catalog;
use std::collections::{HashMap, HashSet};

use crate::ast::Span;
use crate::ast::TypeName;
use crate::builtins::lowering::{BuiltinLowering, PrimOp};
use crate::error::Result;
use crate::ssa::layout::{buffer_array_strides, type_byte_size};
use crate::ssa::types::{
    BlockId, ConstantValue, ControlHeader, FuncBody, InstKind, Terminator, ValueId, ValueRef, WynInstNode,
};
use crate::ssa::types::{EntryPoint, ExecutionModel, Function, IoDecoration, Program};
use crate::types;
use crate::types::TypeExt;
use crate::{bail_spirv, bail_spirv_at, err_spirv, err_spirv_at};
use polytype::Type as PolyType;
use rspirv::binary::Assemble;
use rspirv::dr::{Builder, InsertPoint, Operand};
use rspirv::spirv::{self, AddressingModel, Capability, MemoryModel, StorageClass};

// =============================================================================
// Constructor - SPIR-V Builder Wrapper
// =============================================================================

/// Constructor wraps rspirv::Builder with an ergonomic API that handles:
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
    // Function parameter SPIR-V IDs in declaration order (positional mapping)
    param_ids: Vec<spirv::Word>,

    // Function map: name -> function ID
    functions: HashMap<String, spirv::Word>,

    // GLSL extended instruction set
    glsl_ext_inst_id: spirv::Word,

    // Top-level polytype → SPIR-V memoization (subsumes type + constant dedup for wyn types)
    polytype_cache: HashMap<PolyType<TypeName>, spirv::Word>,

    // Composite constant dedup: (result_type, constituents) → constant_id
    composite_const_cache: HashMap<(spirv::Word, Vec<spirv::Word>), spirv::Word>,

    // Type cache: avoid recreating same types
    vec_type_cache: HashMap<(spirv::Word, u32), spirv::Word>,
    struct_type_cache: HashMap<Vec<spirv::Word>, spirv::Word>,
    ptr_type_cache: HashMap<(spirv::StorageClass, spirv::Word), spirv::Word>,
    runtime_array_cache: HashMap<(spirv::Word, u32), spirv::Word>, // (elem_type, stride) -> decorated type
    buffer_block_cache: HashMap<spirv::Word, spirv::Word>, // runtime_array_type -> Block-decorated struct
    interface_block_cache: HashMap<InterfaceBlockKey, spirv::Word>,
    array_elem_cache: HashMap<spirv::Word, spirv::Word>, // array_type -> element_type

    // Entry point interface tracking
    entry_point_interfaces: HashMap<String, Vec<spirv::Word>>,

    extract_cache: HashMap<(spirv::Word, u32), spirv::Word>, // CSE for OpCompositeExtract
    null_const_cache: HashMap<spirv::Word, spirv::Word>,     // type -> OpConstantNull id

    /// Storage buffers for compute shaders: (set, binding) -> (buffer_var, elem_type_id, buffer_ptr_type)
    storage_buffers: HashMap<(u32, u32), (spirv::Word, spirv::Word, spirv::Word)>,

    /// GlobalInvocationId variable for compute shaders (set during entry point setup)
    global_invocation_id: Option<spirv::Word>,

    /// Shared push constant variable (at most one per SPIR-V module)
    push_constant_var: Option<spirv::Word>,

    /// Linked SPIR-V functions: linkage_name -> function_id
    linked_functions: HashMap<String, spirv::Word>,

    /// Output variables for the current entry point being lowered.
    /// Set during entry point setup, cleared at end. Used by OutputPtr lowering.
    current_entry_outputs: Vec<spirv::Word>,

    /// Tracks which SPIR-V types already have ArrayStride decorations for buffer layout.
    buffer_stride_decorated: HashSet<spirv::Word>,

    /// buffer_id → (buffer_var, elem_spirv_type). Indexed by the
    /// compile-time buffer_id stored on the view's SSA value via
    /// `view_buffer_id`.
    buffer_vars: Vec<(spirv::Word, spirv::Word)>,
    /// (set, binding) → buffer_id, for deduplication in get_or_assign_buffer_id.
    buffer_id_map: HashMap<(u32, u32), u32>,

    /// IDs of all module-level constants (OpConstant, OpConstantTrue/False, OpConstantComposite, OpConstantNull).
    /// Used to decide whether a composite can be emitted as OpConstantComposite.
    constant_ids: HashSet<spirv::Word>,
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
            param_ids: Vec::new(),
            functions: HashMap::new(),
            glsl_ext_inst_id,
            polytype_cache: HashMap::new(),
            composite_const_cache: HashMap::new(),
            vec_type_cache: HashMap::new(),
            struct_type_cache: HashMap::new(),
            ptr_type_cache: HashMap::new(),
            runtime_array_cache: HashMap::new(),
            buffer_block_cache: HashMap::new(),
            interface_block_cache: HashMap::new(),
            array_elem_cache: HashMap::new(),
            entry_point_interfaces: HashMap::new(),
            extract_cache: HashMap::new(),
            null_const_cache: HashMap::new(),
            storage_buffers: HashMap::new(),
            global_invocation_id: None,
            push_constant_var: None,
            linked_functions: HashMap::new(),
            current_entry_outputs: Vec::new(),
            buffer_stride_decorated: HashSet::new(),
            buffer_vars: Vec::new(),
            buffer_id_map: HashMap::new(),
            constant_ids: HashSet::new(),
        }
    }

    /// Get or assign a sequential buffer_id for a (set, binding) pair.
    /// Also registers the buffer_var in buffer_vars for later lookup.
    fn get_or_assign_buffer_id(&mut self, set: u32, binding: u32) -> u32 {
        if let Some(&id) = self.buffer_id_map.get(&(set, binding)) {
            return id;
        }
        let id = self.buffer_vars.len() as u32;
        let &(buffer_var, elem_ty, _) = self
            .storage_buffers
            .get(&(set, binding))
            .expect("get_or_assign_buffer_id: storage buffer must exist");
        self.buffer_vars.push((buffer_var, elem_ty));
        self.buffer_id_map.insert((set, binding), id);
        id
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
        if let Some(&cached) = self.polytype_cache.get(ty) {
            return cached;
        }
        let result = self.polytype_to_spirv_uncached(ty);
        self.polytype_cache.insert(ty.clone(), result);
        result
    }

    fn polytype_to_spirv_uncached(&mut self, ty: &PolyType<TypeName>) -> spirv::Word {
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
                    TypeName::Bool => self.bool_type,
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
                        // Array[elem, size, variant]
                        let elem = ty.elem_type().expect("Array has elem");
                        let elem_type = self.polytype_to_spirv(elem);
                        let size = ty.array_size().expect("Array has size");
                        let variant = ty.array_variant().expect("Array has variant");

                        // Dispatch on variant first - View arrays are always {offset, len} structs
                        if let PolyType::Constructed(TypeName::ArrayVariantView, _) = variant {
                            // View variant: struct { offset: u32, len: u32 }. The
                            // backing storage buffer is identified by compile-time
                            // `view_buffer_id` provenance on the SSA value, not a
                            // runtime field — so the provenance survives phis and
                            // view-preserving intrinsics where reverse-mapping a
                            // runtime constant can't recover it.
                            self.get_or_create_struct_type(vec![self.u32_type, self.u32_type])
                        } else if let PolyType::Constructed(TypeName::ArrayVariantVirtual, _) = variant {
                            // Virtual variant: struct { start, step, len } for range representation
                            // Use the element type so u32 ranges get {u32, u32, u32}.
                            self.get_or_create_struct_type(vec![elem_type, elem_type, elem_type])
                        } else if let PolyType::Constructed(TypeName::ArrayVariantBounded, _) = variant {
                            // Bounded variant: struct { buffer: [N]T, len: i32 } —
                            // function-local fixed-capacity buffer plus a runtime count.
                            // The buffer member is a Composite [N]T (sized SPIR-V array).
                            // The len field is i32 to match the language's `length()`
                            // result type and the index type expected by `array_with`.
                            let n = match size {
                                PolyType::Constructed(TypeName::Size(n), _) => *n as u32,
                                _ => panic!("BUG: Bounded array requires Size(N) capacity, got {:?}", size),
                            };
                            let size_const = self.const_u32(n);
                            let buf_type = self.builder.type_array(elem_type, size_const);
                            self.array_elem_cache.insert(buf_type, elem_type);
                            self.get_or_create_struct_type(vec![buf_type, self.i32_type])
                        } else {
                            // Composite variant (or placeholder): sized array value
                            match size {
                                PolyType::Constructed(TypeName::Size(n), _) => {
                                    // Fixed-size array (use unsigned int for array size per SPIR-V convention)
                                    let size_const = self.const_u32(*n as u32);
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
                        // Vec[elem, Size(n)]
                        let elem = ty.elem_type().expect("Vec has elem");
                        let elem_type = self.polytype_to_spirv(elem);
                        let size = ty.vec_size().expect("Vec has concrete size") as u32;
                        self.get_or_create_vec_type(elem_type, size)
                    }
                    TypeName::Mat => {
                        // Mat[elem, Size(cols), Size(rows)]
                        let elem = ty.elem_type().expect("Mat has elem");
                        let elem_type = self.polytype_to_spirv(elem);
                        let cols = ty.mat_cols().expect("Mat has concrete cols") as u32;
                        let rows = ty.mat_rows().expect("Mat has concrete rows") as u32;
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

    /// Create a decorated interface block struct type.
    /// Atomically creates the OpTypeStruct AND all required decorations.
    /// Cached by kind + layout so identical blocks share one ID, but
    /// never share with plain tuple structs.
    fn create_interface_block_type(
        &mut self,
        kind: InterfaceBlockKind,
        member_types: &[spirv::Word],
        member_offsets: &[u32],
        member_poly_types: &[&PolyType<TypeName>],
    ) -> spirv::Word {
        let key = InterfaceBlockKey {
            kind,
            members: member_types.iter().zip(member_offsets.iter()).map(|(&t, &o)| (t, o)).collect(),
        };
        if let Some(&ty) = self.interface_block_cache.get(&key) {
            return ty;
        }

        // Create a fresh struct — do NOT go through get_or_create_struct_type
        // to avoid sharing IDs with plain tuple structs.
        let ty = self.builder.type_struct(member_types.iter().copied());

        // Decorate as Block
        self.builder.decorate(ty, spirv::Decoration::Block, []);

        // Apply member offsets
        for (i, &offset) in member_offsets.iter().enumerate() {
            self.builder.member_decorate(
                ty,
                i as u32,
                spirv::Decoration::Offset,
                [Operand::LiteralBit32(offset)],
            );
        }

        // Apply ArrayStride for array members
        for (i, poly_ty) in member_poly_types.iter().enumerate() {
            self.apply_buffer_array_strides(member_types[i], poly_ty);
        }

        self.interface_block_cache.insert(key, ty);
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

    /// Create a storage buffer variable for compute shaders.
    /// Returns the variable ID. Also registers it in storage_buffers for later lookup.
    /// Idempotent: returns existing variable if already created for this (set, binding).
    fn create_storage_buffer(
        &mut self,
        array_ty: &PolyType<TypeName>,
        set: u32,
        binding: u32,
    ) -> spirv::Word {
        // Return existing if already created
        if let Some(&(var_id, _, _)) = self.storage_buffers.get(&(set, binding)) {
            return var_id;
        }
        // Extract element type from array type
        let elem_ty = array_ty
            .elem_type()
            .filter(|_| array_ty.is_array())
            .cloned()
            .unwrap_or_else(|| array_ty.clone());
        let elem_spirv = self.polytype_to_spirv(&elem_ty);

        // Calculate stride from element type size
        let stride = type_byte_size(&elem_ty).expect("storage buffer element type must have known size");

        // Ensure nested array types have ArrayStride for buffer layout
        self.apply_buffer_array_strides(elem_spirv, &elem_ty);

        // If the element type is a tuple/struct, add member offset decorations
        // for the buffer layout. We add them to the elem type directly since
        // it will be used inside a runtime array in a storage buffer.
        if let PolyType::Constructed(TypeName::Tuple(_), args) = &elem_ty {
            if self.buffer_stride_decorated.insert(elem_spirv) {
                let mut offset = 0u32;
                for (i, field_ty) in args.iter().enumerate() {
                    self.builder.member_decorate(
                        elem_spirv,
                        i as u32,
                        spirv::Decoration::Offset,
                        [Operand::LiteralBit32(offset)],
                    );
                    offset += type_byte_size(field_ty)
                        .unwrap_or_else(|| panic!("tuple field {:?} has unknown byte size", field_ty));
                }
            }
        }

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
        self.constant_ids.insert(id);
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
        self.constant_ids.insert(id);
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
        self.constant_ids.insert(id);
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
        self.constant_ids.insert(id);
        id
    }

    /// Check whether a SPIR-V ID is a module-level constant.
    fn is_constant(&self, id: spirv::Word) -> bool {
        self.constant_ids.contains(&id)
    }

    /// Emit OpConstantComposite if all elements are constants, otherwise OpCompositeConstruct.
    fn composite_or_constant(
        &mut self,
        result_type: spirv::Word,
        elem_ids: Vec<spirv::Word>,
    ) -> Result<spirv::Word> {
        if elem_ids.iter().all(|&id| self.is_constant(id)) {
            let key = (result_type, elem_ids.clone());
            if let Some(&cached) = self.composite_const_cache.get(&key) {
                return Ok(cached);
            }
            let id = self.builder.constant_composite(result_type, elem_ids);
            self.constant_ids.insert(id);
            self.composite_const_cache.insert(key, id);
            Ok(id)
        } else {
            Ok(self.builder.composite_construct(result_type, None, elem_ids)?)
        }
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

/// Lower a constant definition body to module-level SPIR-V constants.
///
/// Walks instructions in order, emitting OpConstant/OpConstantComposite.
/// Lower an SSA function body to SPIR-V.
///
/// This creates a SPIR-V function from the SSA representation:
/// - SSA blocks become SPIR-V blocks
/// - Block parameters become OpPhi nodes
/// - Terminators become branch instructions
fn lower_ssa_body(constructor: &mut Constructor, body: &FuncBody, func_span: Span) -> Result<spirv::Word> {
    let mut ctx = LowerCtx::new(constructor, body, false, func_span);
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
    let mut ctx = LowerCtx::new(constructor, body, true, func_span);
    ctx.lower()
}

/// Context for lowering SSA to SPIR-V.
struct LowerCtx<'a, 'b> {
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
    /// Map from a `StorageView` result ValueId to its buffer_id (index into
    /// `buffer_vars`). Tracks the handle value's provenance so `ViewIndex`
    /// can pick the right storage buffer for its `OpAccessChain`.
    view_buffer_id: HashMap<ValueId, u32>,
    /// Map from a `PlaceId` to the SPIR-V pointer word that addresses it.
    /// Populated by place-producing instructions (`OutputSlot`,
    /// `ViewIndex`, `Alloca`) and read by `Load` / `Store`.
    place_ptr_id: HashMap<crate::ssa::types::PlaceId, spirv::Word>,
    /// Span of the instruction currently being lowered (set by `lower_inst`).
    /// Consumed via `blame_span()` so backend errors blame the source line of
    /// the originating expression.
    current_span: Option<Span>,
    /// Function-level span fallback when an instruction has no span.
    func_span: Span,
}

impl<'a, 'b> LowerCtx<'a, 'b> {
    fn new(
        constructor: &'a mut Constructor,
        body: &'b FuncBody,
        is_entry_point: bool,
        func_span: Span,
    ) -> Self {
        LowerCtx {
            constructor,
            body,
            is_entry_point,
            value_map: HashMap::new(),
            block_map: HashMap::new(),
            block_indices: HashMap::new(),
            phi_inputs: Vec::new(),
            view_buffer_id: HashMap::new(),
            place_ptr_id: HashMap::new(),
            current_span: None,
            func_span,
        }
    }

    /// SPIR-V pointer word for a `PlaceId` — set by the defining instruction
    /// (`OutputSlot`, `ViewIndex`, `Alloca`), consumed by `Load` / `Store`.
    fn place_ptr(&self, place: crate::ssa::types::PlaceId) -> Result<spirv::Word> {
        self.place_ptr_id.get(&place).copied().ok_or_else(|| {
            err_spirv_at!(
                self.blame_span(),
                "SPIR-V: place {:?} has no pointer — its defining instruction \
                 was not lowered (or ran after a consumer)",
                place
            )
        })
    }

    /// Source span used to blame an instruction's lowering errors. Falls back
    /// to the function span when the instruction has no span of its own.
    fn blame_span(&self) -> Span {
        self.current_span.unwrap_or(self.func_span)
    }

    fn lower(&mut self) -> Result<spirv::Word> {
        // Map function parameters to their SPIR-V values.
        // For regular functions, use positional mapping (param_ids) to avoid
        // name collisions when two params share a string name.
        // For entry points (no param_ids), fall back to name-based env lookup.
        if self.constructor.param_ids.len() == self.body.params.len()
            && !self.constructor.param_ids.is_empty()
        {
            for (i, (value_id, _, _)) in self.body.params.iter().enumerate() {
                self.value_map.insert(*value_id, self.constructor.param_ids[i]);
            }
        } else {
            for (value_id, _, name) in &self.body.params {
                if let Some(&spirv_id) = self.constructor.env.get(name) {
                    self.value_map.insert(*value_id, spirv_id);
                }
            }
        }

        // Create all SPIR-V blocks and pre-allocate phi IDs for all block params.
        // Phi IDs must be allocated up front because SSA values from one block's
        // params may be referenced by instructions in other blocks (e.g., an if/else
        // result used in an array literal that spans multiple merge blocks).
        let entry_block = self.body.entry_block();
        for (block_id, block) in &self.body.inner.blocks {
            if block.insts.is_empty() && matches!(block.term, Terminator::Unreachable) {
                continue;
            }
            if block_id == entry_block {
                let current = self.constructor.current_block.unwrap();
                self.block_map.insert(block_id, current);
            } else {
                let spirv_block = self.constructor.builder.id();
                self.block_map.insert(block_id, spirv_block);
            }

            for &param in &block.params {
                let phi_id = self.constructor.builder.id();
                self.value_map.insert(param, phi_id);
            }
        }

        let rpo = self.compute_rpo();

        for &block_id in &rpo {
            let block = &self.body.inner.blocks[block_id];

            if block_id != entry_block {
                let spirv_block = self.block_map[&block_id];
                self.constructor.begin_block(spirv_block)?;
            }

            // Record block index for phi insertion
            if let Some(idx) = self.constructor.builder.selected_block() {
                self.block_indices.insert(block_id, idx);
            }

            // Lower instructions
            for &inst_id in &block.insts {
                let inst = self.body.get_inst(inst_id);
                self.lower_inst(inst).map_err(|e| err_spirv!("Block({:?}): {}", block_id, e))?;
            }

            // Lower terminator
            self.lower_terminator(block_id, block, &block.term)?;
        }

        // Insert phi nodes for all block parameters
        self.insert_phi_nodes()?;

        // Return placeholder - actual return handled by terminators in SSA
        Ok(self.constructor.const_i32(0))
    }

    /// Compute a structured block ordering for SPIR-V emission.
    ///
    /// SPIR-V requires that all blocks in a loop/selection construct appear
    /// between the header and the merge block. A plain RPO can violate this
    /// (e.g. placing a loop's merge before its continue block). This traversal
    /// defers merge blocks until after all construct-interior blocks are visited.
    fn compute_rpo(&self) -> Vec<BlockId> {
        let mut visited: HashSet<BlockId> = HashSet::new();
        let mut order = Vec::with_capacity(self.body.inner.blocks.len());

        fn visit(body: &FuncBody, bid: BlockId, visited: &mut HashSet<BlockId>, order: &mut Vec<BlockId>) {
            if visited.contains(&bid) {
                return;
            }
            let block = &body.inner.blocks[bid];
            if block.insts.is_empty() && matches!(block.term, Terminator::Unreachable) {
                return;
            }
            visited.insert(bid);
            order.push(bid);

            let merge_bid = body.control_headers.get(&bid).map(|ctrl| match ctrl {
                ControlHeader::Loop { merge, .. } => *merge,
                ControlHeader::Selection { merge } => *merge,
            });

            match &block.term {
                Terminator::Branch { target, .. } => {
                    if Some(*target) != merge_bid {
                        visit(body, *target, visited, order);
                    }
                }
                Terminator::CondBranch {
                    then_target,
                    else_target,
                    ..
                } => {
                    if Some(*then_target) != merge_bid {
                        visit(body, *then_target, visited, order);
                    }
                    if Some(*else_target) != merge_bid {
                        visit(body, *else_target, visited, order);
                    }
                }
                _ => {}
            }

            if let Some(m) = merge_bid {
                visit(body, m, visited, order);
            }
        }

        visit(self.body, self.body.entry_block(), &mut visited, &mut order);
        order
    }

    fn lower_inst(&mut self, inst: &WynInstNode) -> Result<()> {
        let ssa_result_ty = inst.result.map(|r| self.body.inner.value_type(r).clone());
        let result_ty = ssa_result_ty.as_ref().map(|t| self.constructor.polytype_to_spirv(t)).unwrap_or(0);
        self.current_span = inst.span;

        let spirv_result = match &inst.data {
            InstKind::Op { tag, operands } => match tag {
                crate::op::OpTag::Int(s) | crate::op::OpTag::Uint(s) => match ssa_result_ty.as_ref() {
                    Some(PolyType::Constructed(TypeName::UInt(32), _)) => {
                        let val: u32 = s
                            .parse()
                            .map_err(|_| err_spirv_at!(self.blame_span(), "Invalid u32: {}", s))?;
                        self.constructor.const_u32(val)
                    }
                    _ => {
                        let val: i32 = s
                            .parse()
                            .map_err(|_| err_spirv_at!(self.blame_span(), "Invalid i32: {}", s))?;
                        self.constructor.const_i32(val)
                    }
                },

                crate::op::OpTag::Float(s) => {
                    let val: f32 =
                        s.parse().map_err(|_| err_spirv_at!(self.blame_span(), "Invalid f32: {}", s))?;
                    self.constructor.const_f32(val)
                }

                crate::op::OpTag::Bool(b) => self.constructor.const_bool(*b),

                crate::op::OpTag::Unit => {
                    unreachable!(
                        "OpTag::Unit should never reach SPIR-V codegen; unit values are not materializable"
                    )
                }

                crate::op::OpTag::BinOp(op) => {
                    let lhs = operands[0];
                    let rhs = operands[1];
                    let lhs_id = self.get_value_ref(lhs)?;
                    let rhs_id = self.get_value_ref(rhs)?;
                    let lhs_ty = self.get_value_type_ref(lhs);
                    let rhs_ty = self.get_value_type_ref(rhs);
                    self.lower_binop(op, lhs_id, rhs_id, &lhs_ty, &rhs_ty, result_ty)?
                }

                crate::op::OpTag::UnaryOp(op) => {
                    let operand = operands[0];
                    let operand_id = self.get_value_ref(operand)?;
                    let operand_ty = self.get_value_type_ref(operand);
                    self.lower_unaryop(op, operand_id, &operand_ty, result_ty)?
                }

                crate::op::OpTag::Tuple(_) => {
                    let elem_ids: Vec<_> =
                        operands.iter().map(|v| self.get_value_ref(*v)).collect::<Result<_>>()?;
                    self.constructor.composite_or_constant(result_ty, elem_ids)?
                }

                crate::op::OpTag::ArrayLit(_) => {
                    let elem_ids: Vec<_> =
                        operands.iter().map(|v| self.get_value_ref(*v)).collect::<Result<_>>()?;
                    self.constructor.composite_or_constant(result_ty, elem_ids)?
                }

                crate::op::OpTag::ArrayRange { has_step } => {
                    // Virtual array represented as {start, step, len} struct
                    // This matches the layout expected by lower_virtual_index
                    let start_id = self.get_value_ref(operands[0])?;
                    let len_id = self.get_value_ref(operands[1])?;
                    let step_id = if *has_step {
                        self.get_value_ref(operands[2])?
                    } else {
                        // Default step = 1, matching the element type of the range.
                        let elem_ty = ssa_result_ty.as_ref().and_then(|t| t.elem_type());
                        if matches!(elem_ty, Some(PolyType::Constructed(TypeName::UInt(_), _))) {
                            self.constructor.const_u32(1)
                        } else {
                            self.constructor.const_i32(1)
                        }
                    };

                    // Construct the struct: {start, step, len}
                    self.constructor.builder.composite_construct(
                        result_ty,
                        None,
                        vec![start_id, step_id, len_id],
                    )?
                }

                crate::op::OpTag::Vector(_) => {
                    let elem_ids: Vec<_> =
                        operands.iter().map(|v| self.get_value_ref(*v)).collect::<Result<_>>()?;
                    self.constructor.composite_or_constant(result_ty, elem_ids)?
                }

                crate::op::OpTag::Matrix { .. } => {
                    // Matrix is constructed as an array of vectors (columns)
                    // For now, flatten and construct
                    let all_elems: Vec<_> =
                        operands.iter().map(|v| self.get_value_ref(*v)).collect::<Result<_>>()?;
                    self.constructor.composite_or_constant(result_ty, all_elems)?
                }

                crate::op::OpTag::Project { index } => {
                    let base = operands[0];
                    let base_ty = self.get_value_type_ref(base);
                    let base_id = self.get_value_ref(base)?;

                    // If base is a pointer, load it first
                    let composite_id = if types::is_pointer(&base_ty) {
                        let pointee_ty = types::pointee(&base_ty).expect("Pointer should have pointee");
                        let value_type = self.constructor.polytype_to_spirv(pointee_ty);
                        self.constructor.builder.load(value_type, None, base_id, None, [])?
                    } else {
                        base_id
                    };

                    self.constructor.builder.composite_extract(result_ty, None, composite_id, [*index])?
                }

                crate::op::OpTag::Index => self.lower_index(operands[0], operands[1], result_ty)?,

                crate::op::OpTag::Call(func) => {
                    let args: Vec<ValueRef> = operands.clone();
                    let arg_ids: Vec<_> =
                        args.iter().map(|v| self.get_value_ref(*v)).collect::<Result<_>>()?;
                    // Catalog-named calls (e.g. per-type ops like `f32.clamp`
                    // emitted by `specialize.rs`) still arrive here as Call
                    // because their func is a SymbolId allocated for the
                    // specialized name. Compiler-internal intrinsics
                    // (`_w_intrinsic_*`) now go through `OpTag::Intrinsic`.
                    if let Some(def) = catalog().lookup_by_any_name(func) {
                        let builtin_impl = &def.overloads()[0].lowering;
                        self.lower_builtin_call(
                            def.id,
                            builtin_impl,
                            func,
                            &args,
                            &arg_ids,
                            result_ty,
                            inst,
                        )?
                    } else if let Some(&func_id) = self.constructor.functions.get(func) {
                        self.constructor.builder.function_call(result_ty, None, func_id, arg_ids)?
                    } else {
                        bail_spirv_at!(self.blame_span(), "Unknown function: {}", func)
                    }
                }

                crate::op::OpTag::Global(name) => {
                    if let Some(&func_id) = self.constructor.functions.get(name) {
                        // Global constant function - call it with no args to get the value.
                        // This handles `def verts: [3]vec4f32 = [...]` referenced as just `verts`.
                        self.constructor.builder.function_call(result_ty, None, func_id, [])?
                    } else {
                        bail_spirv_at!(self.blame_span(), "Unknown global: {}", name)
                    }
                }

                crate::op::OpTag::Extern(linkage_name) => {
                    self.constructor.linked_functions.get(linkage_name).copied().ok_or_else(|| {
                        err_spirv_at!(self.blame_span(), "Unknown extern: {}", linkage_name)
                    })?
                }

                crate::op::OpTag::Intrinsic { id, overload_idx } => {
                    let args: Vec<ValueRef> = operands.clone();
                    let arg_ids: Vec<_> =
                        args.iter().map(|v| self.get_value_ref(*v)).collect::<Result<_>>()?;
                    let def = crate::builtins::by_id(*id);
                    let lowering = &def.overloads()[*overload_idx].lowering;
                    // Variants with a structural arm in `lower_builtin_call`
                    // dispatch via the BuiltinLowering value or the entry
                    // id; the rest still fall through to the name-keyed
                    // `lower_intrinsic` until they're promoted.
                    let known = catalog().known();
                    let typed_dispatch = matches!(
                        lowering,
                        BuiltinLowering::PrimOp(_)
                            | BuiltinLowering::LinkedSpirv(_)
                            | BuiltinLowering::ExtInstSplat { .. }
                    ) || (matches!(lowering, BuiltinLowering::ByBuiltinId)
                        && (*id == known.slice
                            || *id == known.storage_len
                            || *id == known.thread_id
                            || *id == known.length
                            || *id == known.uninit
                            || *id == known.array_with
                            || *id == known.array_with_in_place));
                    if typed_dispatch {
                        self.lower_builtin_call(
                            *id,
                            lowering,
                            def.dispatch_name(),
                            &args,
                            &arg_ids,
                            result_ty,
                            inst,
                        )?
                    } else {
                        bail_spirv!(
                            "OpTag::Intrinsic with no SPIR-V backend dispatch: '{}' \
                             (id={:?}, lowering={:?}). HOF / SOAC intrinsics should be \
                             lowered at EGIR; everything else needs an arm in \
                             lower_builtin_call and an entry in the typed_dispatch list.",
                            def.dispatch_name(),
                            id,
                            lowering
                        )
                    }
                }

                crate::op::OpTag::StorageView(src) => {
                    let offset = operands[0];
                    let len = operands[1];
                    let offset_id = self.get_value_ref(offset)?;
                    let len_id = self.get_value_ref(len)?;

                    match src {
                        crate::op::PureViewSource::Storage { set, binding } => {
                            if self.constructor.storage_buffers.contains_key(&(*set, *binding)) {
                                let buffer_id = self.constructor.get_or_assign_buffer_id(*set, *binding);
                                if let Some(result) = inst.result {
                                    self.view_buffer_id.insert(result, buffer_id);
                                }
                                let u32_ty = self.constructor.u32_type;
                                let view_struct_type =
                                    self.constructor.get_or_create_struct_type(vec![u32_ty, u32_ty]);
                                self.constructor.builder.composite_construct(
                                    view_struct_type,
                                    None,
                                    [offset_id, len_id],
                                )?
                            } else {
                                bail_spirv_at!(
                                    self.blame_span(),
                                    "Unknown storage buffer: set={}, binding={}",
                                    set,
                                    binding
                                )
                            }
                        }
                        crate::op::PureViewSource::Inherited => {
                            let parent =
                                operands[2].as_ssa().expect("StorageView Inherited parent must be SSA");
                            if let (Some(result), Some(&parent_buf_id)) =
                                (inst.result, self.view_buffer_id.get(&parent))
                            {
                                self.view_buffer_id.insert(result, parent_buf_id);
                            }
                            let parent_id = self.get_value(parent)?;
                            let u32_ty = self.constructor.u32_type;
                            let view_struct_type =
                                self.constructor.get_or_create_struct_type(vec![u32_ty, u32_ty]);

                            // Extract parent_offset (field 0) from parent view
                            let parent_offset =
                                self.constructor.builder.composite_extract(u32_ty, None, parent_id, [0])?;

                            // new_offset = parent_offset + offset
                            let new_offset =
                                self.constructor.builder.i_add(u32_ty, None, parent_offset, offset_id)?;

                            self.constructor.builder.composite_construct(
                                view_struct_type,
                                None,
                                [new_offset, len_id],
                            )?
                        }
                    }
                }

                crate::op::OpTag::StorageViewLen => {
                    let view = operands[0];
                    let view_id = self.get_value_ref(view)?;
                    // Extract len from view struct (field 1 in {offset, len})
                    self.constructor.builder.composite_extract(result_ty, None, view_id, [1u32])?
                }

                crate::op::OpTag::Materialize => {
                    let value = operands[0];
                    let value_id = self.get_value_ref(value)?;
                    let value_ty = self.get_value_type_ref(value);
                    let spirv_type = self.constructor.polytype_to_spirv(&value_ty);
                    let var = self.constructor.declare_variable("_materialize", spirv_type)?;
                    self.constructor.builder.store(var, value_id, None, [])?;
                    var
                }

                crate::op::OpTag::DynamicExtract => {
                    let base = operands[0];
                    let index = operands[1];
                    let base_var = self.get_value_ref(base)?;
                    let index_id = self.get_value_ref(index)?;
                    let base_ty = self.get_value_type_ref(base);
                    // For a Bounded base, the underlying value is a struct
                    // `{buffer: [N]T, len: i32}`, so the access chain has to
                    // index member 0 first (constant) before the dynamic
                    // index reaches the array element. Other variants
                    // (Composite/View/Virtual) chain directly to the
                    // element.
                    let elem_ptr_type = self.constructor.builder.type_pointer(
                        None,
                        spirv::StorageClass::Function,
                        result_ty,
                    );
                    let elem_ptr = if matches!(
                        base_ty.array_variant(),
                        Some(PolyType::Constructed(TypeName::ArrayVariantBounded, _))
                    ) {
                        let zero = self.constructor.const_u32(0);
                        self.constructor.builder.access_chain(
                            elem_ptr_type,
                            None,
                            base_var,
                            [zero, index_id],
                        )?
                    } else {
                        self.constructor.builder.access_chain(elem_ptr_type, None, base_var, [index_id])?
                    };
                    self.constructor.builder.load(result_ty, None, elem_ptr, None, [])?
                }

                crate::op::OpTag::ViewIndex | crate::op::OpTag::OutputSlot { .. } => {
                    unreachable!("OpTag::{:?} is EGIR-only and must not reach SSA backend", tag)
                }
            },

            InstKind::Alloca { elem_ty, result } => {
                let elem_spirv_ty = self.constructor.polytype_to_spirv(elem_ty);
                let ptr = self.constructor.declare_variable("_alloca", elem_spirv_ty)?;
                self.place_ptr_id.insert(*result, ptr);
                // Void instruction — no value result; return a harmless dummy.
                self.constructor.const_i32(0)
            }

            InstKind::Load { place } => {
                let ptr_id = self.place_ptr(*place)?;
                self.constructor.builder.load(result_ty, None, ptr_id, None, [])?
            }

            InstKind::Store { place, value } => {
                let ptr_id = self.place_ptr(*place)?;
                let val_id = self.get_value_ref(*value)?;
                self.constructor.builder.store(ptr_id, val_id, None, [])?;
                // Store doesn't produce a value, but we return dummy
                self.constructor.const_i32(0)
            }

            InstKind::ViewIndex { view, index, result } => {
                let view_id = self.get_value_ref(*view)?;
                let index_id = self.get_value_ref(*index)?;
                let u32_ty = self.constructor.u32_type;

                // Extract offset (field 0) from view struct {offset, len}.
                // Buffer-var provenance comes from `view_buffer_id`.
                let base_offset =
                    self.constructor.builder.composite_extract(u32_ty, None, view_id, [0u32])?;

                let view_ssa = view.as_ssa().ok_or_else(|| {
                    err_spirv_at!(self.blame_span(), "ViewIndex view operand must be SSA")
                })?;
                let buf_id = self.view_buffer_id.get(&view_ssa).copied().ok_or_else(|| {
                    err_spirv_at!(
                        self.blame_span(),
                        "ViewIndex: no buffer provenance for value {:?}",
                        view_ssa
                    )
                })?;
                let (buffer_var, _elem_spirv_ty) =
                    self.constructor.buffer_vars.get(buf_id as usize).copied().ok_or_else(|| {
                        err_spirv_at!(self.blame_span(), "view_buffer_id: unknown buffer_id {}", buf_id)
                    })?;

                let actual_index = self.constructor.builder.i_add(u32_ty, None, base_offset, index_id)?;
                let zero = self.constructor.const_i32(0);
                // Infer element SPIR-V type from the place's elem_ty — the
                // place's type is what `Load` will return / `Store` writes.
                let place_elem = self.body.place_elem_ty(*result).clone();
                let elem_ty_id = self.constructor.polytype_to_spirv(&place_elem);
                let elem_ptr_type =
                    self.constructor.get_or_create_ptr_type(spirv::StorageClass::StorageBuffer, elem_ty_id);
                let ptr = self.constructor.builder.access_chain(
                    elem_ptr_type,
                    None,
                    buffer_var,
                    [zero, actual_index],
                )?;
                self.place_ptr_id.insert(*result, ptr);
                // Void instruction.
                self.constructor.const_i32(0)
            }

            InstKind::OutputSlot { index, result } => {
                // Each output was wired up in `lower_ssa_entry_point`; bind
                // the place to its output variable pointer.
                if *index >= self.constructor.current_entry_outputs.len() {
                    bail_spirv_at!(
                        self.blame_span(),
                        "Output index {} out of bounds (have {} outputs)",
                        index,
                        self.constructor.current_entry_outputs.len()
                    );
                }
                let ptr = self.constructor.current_entry_outputs[*index];
                self.place_ptr_id.insert(*result, ptr);
                // Void instruction.
                self.constructor.const_i32(0)
            }
        };

        if let Some(result_value) = inst.result {
            self.value_map.insert(result_value, spirv_result);
        }

        Ok(())
    }

    fn lower_terminator(
        &mut self,
        _block_id: BlockId,
        _block: &crate::ssa::framework::BasicBlock,
        term: &Terminator,
    ) -> Result<()> {
        let current_block = self.constructor.current_block.unwrap();

        match term {
            Terminator::Branch { target, args } => {
                for (param_idx, &arg) in args.iter().enumerate() {
                    let arg_id = self.get_value(arg)?;
                    self.phi_inputs.push((*target, param_idx, arg_id, current_block));
                    self.propagate_view_provenance(*target, param_idx, arg);
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

                for (param_idx, &arg) in then_args.iter().enumerate() {
                    let arg_id = self.get_value(arg)?;
                    self.phi_inputs.push((*then_target, param_idx, arg_id, current_block));
                    self.propagate_view_provenance(*then_target, param_idx, arg);
                }
                for (param_idx, &arg) in else_args.iter().enumerate() {
                    let arg_id = self.get_value(arg)?;
                    self.phi_inputs.push((*else_target, param_idx, arg_id, current_block));
                    self.propagate_view_provenance(*else_target, param_idx, arg);
                }

                // Emit structured control flow merge instructions if this is a header block
                if let Some(control) = self.body.control_headers.get(&_block_id) {
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

            Terminator::Return(Some(value)) => {
                if self.is_entry_point {
                    bail_spirv!(
                        "Return(value) in entry point body — entry points are void functions \
                         and must use OutputPtr+Store then ReturnUnit"
                    );
                }
                let value_id = self.get_value(*value)?;
                self.constructor.builder.ret_value(value_id)?;
            }

            Terminator::Return(None) => {
                self.constructor.builder.ret()?;
            }

            Terminator::Unreachable => {
                self.constructor.builder.unreachable()?;
            }
        }

        Ok(())
    }

    /// Propagate `view_buffer_id` along a branch edge: if `arg` is a view
    /// value with known buffer provenance, attach the same provenance to the
    /// target block's parameter at `param_idx`. Required for views threaded
    /// through loop block params (e.g. scan-DPS phase 1), where the consumer
    /// (`ViewIndex`) sees the block-param ValueId rather than the original
    /// `StorageView` result.
    fn propagate_view_provenance(&mut self, target: BlockId, param_idx: usize, arg: ValueId) {
        let Some(&buf_id) = self.view_buffer_id.get(&arg) else {
            return;
        };
        let target_block = &self.body.inner.blocks[target];
        let Some(&target_param) = target_block.params.get(param_idx) else {
            return;
        };
        if let Some(&existing) = self.view_buffer_id.get(&target_param) {
            debug_assert_eq!(
                existing, buf_id,
                "block param view provenance must agree across incoming edges"
            );
        } else {
            self.view_buffer_id.insert(target_param, buf_id);
        }
    }

    fn insert_phi_nodes(&mut self) -> Result<()> {
        // Group phi inputs by (target_block, param_idx)
        let mut phi_map: HashMap<(BlockId, usize), Vec<(spirv::Word, spirv::Word)>> = HashMap::new();

        for (target_block, param_idx, value, source_block) in &self.phi_inputs {
            phi_map.entry((*target_block, *param_idx)).or_default().push((*value, *source_block));
        }

        // Insert phi nodes
        for ((block_id, param_idx), incoming) in phi_map {
            let block = &self.body.inner.blocks[block_id];
            let param = block.params[param_idx];
            let param_ty = self.constructor.polytype_to_spirv(self.body.inner.value_type(param));

            let phi_id = self.value_map[&param];

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
        self.value_map.get(&value).copied().ok_or_else(|| {
            // Build diagnostic info to help debug SSA/lowering issues
            let producer_block = self
                .body
                .inner
                .insts
                .values()
                .find(|i| i.result == Some(value))
                .map(|i| format!("produced in Block({:?})", i.parent));
            let block_param = self.body.inner.blocks.iter().find_map(|(bid, b)| {
                b.params.contains(&value).then(|| format!("block param of Block({:?})", bid))
            });
            let origin = producer_block.or(block_param).unwrap_or_else(|| "not found in body".to_string());
            err_spirv!("Unknown SSA value: {:?} ({})", value, origin)
        })
    }

    fn get_value_ref(&mut self, vr: ValueRef) -> Result<spirv::Word> {
        match vr {
            ValueRef::Ssa(id) => self.get_value(id),
            ValueRef::Const(c) => match c {
                ConstantValue::I32(v) => Ok(self.constructor.const_i32(v)),
                ConstantValue::U32(v) => Ok(self.constructor.const_u32(v)),
                ConstantValue::F32(bits) => Ok(self.constructor.const_f32(f32::from_bits(bits))),
                ConstantValue::Bool(b) => Ok(self.constructor.const_bool(b)),
            },
        }
    }

    fn get_value_type_ref(&self, vr: ValueRef) -> PolyType<TypeName> {
        match vr {
            ValueRef::Ssa(id) => self.body.get_value_type(id).clone(),
            ValueRef::Const(c) => match c {
                ConstantValue::I32(_) => PolyType::Constructed(TypeName::Int(32), vec![]),
                ConstantValue::U32(_) => PolyType::Constructed(TypeName::UInt(32), vec![]),
                ConstantValue::F32(_) => PolyType::Constructed(TypeName::Float(32), vec![]),
                ConstantValue::Bool(_) => PolyType::Constructed(TypeName::Bool, vec![]),
            },
        }
    }

    fn lower_binop(
        &mut self,
        op: &str,
        lhs: spirv::Word,
        rhs: spirv::Word,
        lhs_ty: &PolyType<TypeName>,
        rhs_ty: &PolyType<TypeName>,
        result_ty: spirv::Word,
    ) -> Result<spirv::Word> {
        use PolyType::*;
        use TypeName::*;

        let bool_type = self.constructor.bool_type;

        match (op, lhs_ty, rhs_ty) {
            // Scalar-left mixed-type ops (must precede scalar catch-alls)
            ("*", Constructed(Float(_), _), Constructed(Vec, _)) => {
                Ok(self.constructor.builder.vector_times_scalar(result_ty, None, rhs, lhs)?)
            }
            ("*", Constructed(Float(_), _), Constructed(Mat, _)) => {
                Ok(self.constructor.builder.matrix_times_scalar(result_ty, None, rhs, lhs)?)
            }
            ("+" | "-" | "/" | "%", Constructed(Float(_) | Int(_) | UInt(_), _), Constructed(Vec, _)) => {
                let splat = self.splat_scalar(lhs, rhs_ty, result_ty)?;
                self.lower_binop(op, splat, rhs, rhs_ty, rhs_ty, result_ty)
            }

            // Float operations
            ("+", Constructed(Float(_), _), _) => {
                Ok(self.constructor.builder.f_add(result_ty, None, lhs, rhs)?)
            }
            ("-", Constructed(Float(_), _), _) => {
                Ok(self.constructor.builder.f_sub(result_ty, None, lhs, rhs)?)
            }
            ("*", Constructed(Float(_), _), _) => {
                Ok(self.constructor.builder.f_mul(result_ty, None, lhs, rhs)?)
            }
            ("/", Constructed(Float(_), _), _) => {
                Ok(self.constructor.builder.f_div(result_ty, None, lhs, rhs)?)
            }
            ("%", Constructed(Float(_), _), _) => {
                Ok(self.constructor.builder.f_rem(result_ty, None, lhs, rhs)?)
            }
            ("<", Constructed(Float(_), _), _) => {
                Ok(self.constructor.builder.f_ord_less_than(bool_type, None, lhs, rhs)?)
            }
            ("<=", Constructed(Float(_), _), _) => {
                Ok(self.constructor.builder.f_ord_less_than_equal(bool_type, None, lhs, rhs)?)
            }
            (">", Constructed(Float(_), _), _) => {
                Ok(self.constructor.builder.f_ord_greater_than(bool_type, None, lhs, rhs)?)
            }
            (">=", Constructed(Float(_), _), _) => {
                Ok(self.constructor.builder.f_ord_greater_than_equal(bool_type, None, lhs, rhs)?)
            }
            ("==", Constructed(Float(_), _), _) => {
                Ok(self.constructor.builder.f_ord_equal(bool_type, None, lhs, rhs)?)
            }
            ("!=", Constructed(Float(_), _), _) => {
                Ok(self.constructor.builder.f_ord_not_equal(bool_type, None, lhs, rhs)?)
            }
            ("**", Constructed(Float(_), _), _) => {
                // Power operator using GLSL pow (opcode 26)
                let glsl = self.constructor.glsl_ext_inst_id;
                let operands = vec![Operand::IdRef(lhs), Operand::IdRef(rhs)];
                Ok(self.constructor.builder.ext_inst(result_ty, None, glsl, 26, operands)?)
            }

            // Integer operations (signed)
            ("+", Constructed(Int(_), _), _) => {
                Ok(self.constructor.builder.i_add(result_ty, None, lhs, rhs)?)
            }
            ("-", Constructed(Int(_), _), _) => {
                Ok(self.constructor.builder.i_sub(result_ty, None, lhs, rhs)?)
            }
            ("*", Constructed(Int(_), _), _) => {
                Ok(self.constructor.builder.i_mul(result_ty, None, lhs, rhs)?)
            }
            ("/", Constructed(Int(_), _), _) => {
                Ok(self.constructor.builder.s_div(result_ty, None, lhs, rhs)?)
            }
            ("%", Constructed(Int(_), _), _) => {
                Ok(self.constructor.builder.s_rem(result_ty, None, lhs, rhs)?)
            }
            ("<", Constructed(Int(_), _), _) => {
                Ok(self.constructor.builder.s_less_than(bool_type, None, lhs, rhs)?)
            }
            ("<=", Constructed(Int(_), _), _) => {
                Ok(self.constructor.builder.s_less_than_equal(bool_type, None, lhs, rhs)?)
            }
            (">", Constructed(Int(_), _), _) => {
                Ok(self.constructor.builder.s_greater_than(bool_type, None, lhs, rhs)?)
            }
            (">=", Constructed(Int(_), _), _) => {
                Ok(self.constructor.builder.s_greater_than_equal(bool_type, None, lhs, rhs)?)
            }
            ("==", Constructed(Int(_), _), _) => {
                Ok(self.constructor.builder.i_equal(bool_type, None, lhs, rhs)?)
            }
            ("!=", Constructed(Int(_), _), _) => {
                Ok(self.constructor.builder.i_not_equal(bool_type, None, lhs, rhs)?)
            }

            // Unsigned integer operations
            ("+", Constructed(UInt(_), _), _) => {
                Ok(self.constructor.builder.i_add(result_ty, None, lhs, rhs)?)
            }
            ("-", Constructed(UInt(_), _), _) => {
                Ok(self.constructor.builder.i_sub(result_ty, None, lhs, rhs)?)
            }
            ("*", Constructed(UInt(_), _), _) => {
                Ok(self.constructor.builder.i_mul(result_ty, None, lhs, rhs)?)
            }
            ("/", Constructed(UInt(_), _), _) => {
                Ok(self.constructor.builder.u_div(result_ty, None, lhs, rhs)?)
            }
            ("%", Constructed(UInt(_), _), _) => {
                Ok(self.constructor.builder.u_mod(result_ty, None, lhs, rhs)?)
            }
            ("<", Constructed(UInt(_), _), _) => {
                Ok(self.constructor.builder.u_less_than(bool_type, None, lhs, rhs)?)
            }
            ("<=", Constructed(UInt(_), _), _) => {
                Ok(self.constructor.builder.u_less_than_equal(bool_type, None, lhs, rhs)?)
            }
            (">", Constructed(UInt(_), _), _) => {
                Ok(self.constructor.builder.u_greater_than(bool_type, None, lhs, rhs)?)
            }
            (">=", Constructed(UInt(_), _), _) => {
                Ok(self.constructor.builder.u_greater_than_equal(bool_type, None, lhs, rhs)?)
            }
            ("==", Constructed(UInt(_), _), _) => {
                Ok(self.constructor.builder.i_equal(bool_type, None, lhs, rhs)?)
            }
            ("!=", Constructed(UInt(_), _), _) => {
                Ok(self.constructor.builder.i_not_equal(bool_type, None, lhs, rhs)?)
            }

            // Boolean operations
            ("&&", Constructed(Bool, _), _) => {
                Ok(self.constructor.builder.logical_and(bool_type, None, lhs, rhs)?)
            }
            ("||", Constructed(Bool, _), _) => {
                Ok(self.constructor.builder.logical_or(bool_type, None, lhs, rhs)?)
            }
            ("==", Constructed(Bool, _), _) => {
                Ok(self.constructor.builder.logical_equal(bool_type, None, lhs, rhs)?)
            }
            ("!=", Constructed(Bool, _), _) => {
                Ok(self.constructor.builder.logical_not_equal(bool_type, None, lhs, rhs)?)
            }

            // Mixed-type multiplication: mat*mat, mat*vec, vec*mat, vec*scalar, mat*scalar
            ("*", Constructed(Mat, _), Constructed(Mat, _)) => {
                Ok(self.constructor.builder.matrix_times_matrix(result_ty, None, lhs, rhs)?)
            }
            ("*", Constructed(Mat, _), Constructed(Vec, _)) => {
                Ok(self.constructor.builder.matrix_times_vector(result_ty, None, lhs, rhs)?)
            }
            ("*", Constructed(Vec, _), Constructed(Mat, _)) => {
                Ok(self.constructor.builder.vector_times_matrix(result_ty, None, lhs, rhs)?)
            }
            ("*", Constructed(Vec, _), Constructed(Float(_), _)) => {
                Ok(self.constructor.builder.vector_times_scalar(result_ty, None, lhs, rhs)?)
            }
            ("*", Constructed(Mat, _), Constructed(Float(_), _)) => {
                Ok(self.constructor.builder.matrix_times_scalar(result_ty, None, lhs, rhs)?)
            }

            // Vector operations: dispatch based on element type
            (_, Constructed(Vec, _), _) => {
                // If rhs is scalar (not vec/mat), splat it to match lhs vec
                let rhs = if matches!(rhs_ty, Constructed(Float(_) | Int(_) | UInt(_), _)) {
                    self.splat_scalar(rhs, lhs_ty, result_ty)?
                } else {
                    rhs
                };

                let elem_ty = lhs_ty
                    .elem_type()
                    .ok_or_else(|| crate::err_spirv!("Vec type missing element type: {:?}", lhs_ty))?;
                match (op, elem_ty) {
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
                    _ => bail_spirv!(
                        "Unsupported vector binary operation: {} on element {:?}",
                        op,
                        elem_ty
                    ),
                }
            }

            _ => bail_spirv!("Unsupported binary operation: {} on {:?}", op, lhs_ty),
        }
    }

    /// Splat a scalar SPIR-V value into a vector matching `vec_ty`.
    fn splat_scalar(
        &mut self,
        scalar: spirv::Word,
        vec_ty: &PolyType<TypeName>,
        vec_spirv_ty: spirv::Word,
    ) -> Result<spirv::Word> {
        let n = vec_ty.vec_size().ok_or_else(|| {
            crate::err_spirv!("Cannot splat: vec type has no concrete size: {:?}", vec_ty)
        })?;
        let components = vec![scalar; n];
        Ok(self.constructor.builder.composite_construct(vec_spirv_ty, None, components)?)
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
            ("!", Constructed(Bool, _)) => {
                Ok(self.constructor.builder.logical_not(result_ty, None, operand)?)
            }
            // Vector unary operations
            ("-", Constructed(Vec, _)) => {
                let elem_ty = operand_ty
                    .elem_type()
                    .ok_or_else(|| crate::err_spirv!("Vec type missing element type: {:?}", operand_ty))?;
                match elem_ty {
                    Constructed(Float(_), _) => {
                        Ok(self.constructor.builder.f_negate(result_ty, None, operand)?)
                    }
                    Constructed(Int(_), _) => {
                        Ok(self.constructor.builder.s_negate(result_ty, None, operand)?)
                    }
                    _ => bail_spirv!(
                        "Unsupported vector unary operation: {} on element {:?}",
                        op,
                        elem_ty
                    ),
                }
            }
            _ => bail_spirv!("Unsupported unary operation: {} on {:?}", op, operand_ty),
        }
    }

    /// Slice a storage view, materializing into a composite array.
    /// Loads each element from the buffer via AccessChain+Load.
    fn slice_view_to_composite(
        &mut self,
        _view_id: spirv::Word,
        buffer_var: spirv::Word,
        base_offset: spirv::Word,
        start_id: spirv::Word,
        end_id: spirv::Word,
        elem_ty: &PolyType<TypeName>,
        result_ty: spirv::Word,
    ) -> Result<spirv::Word> {
        let start = self
            .constructor
            .get_const_i32_value(start_id)
            .ok_or_else(|| err_spirv!("slice_view_to_composite: start must be a constant"))?
            as u32;
        let end = self
            .constructor
            .get_const_i32_value(end_id)
            .ok_or_else(|| err_spirv!("slice_view_to_composite: end must be a constant"))?
            as u32;

        let elem_spirv = self.constructor.polytype_to_spirv(elem_ty);
        let elem_ptr_type =
            self.constructor.get_or_create_ptr_type(spirv::StorageClass::StorageBuffer, elem_spirv);
        let zero = self.constructor.const_i32(0);
        let mut elements = Vec::with_capacity((end - start) as usize);
        for i in start..end {
            let idx_const = self.constructor.const_u32(i);
            let actual_index =
                self.constructor.builder.i_add(self.constructor.u32_type, None, base_offset, idx_const)?;
            let elem_ptr = self.constructor.builder.access_chain(
                elem_ptr_type,
                None,
                buffer_var,
                [zero, actual_index],
            )?;
            let elem = self.constructor.builder.load(elem_spirv, None, elem_ptr, None, [])?;
            elements.push(elem);
        }
        Ok(self.constructor.builder.composite_construct(result_ty, None, elements)?)
    }

    /// Slice a storage view, producing a new handle-based view with adjusted offset/len.
    fn slice_view_to_view(
        &mut self,
        _view_id: spirv::Word,
        base_offset: spirv::Word,
        start_id: spirv::Word,
        end_id: spirv::Word,
    ) -> Result<spirv::Word> {
        let u32_ty = self.constructor.u32_type;
        let new_offset = self.constructor.builder.i_add(u32_ty, None, base_offset, start_id)?;
        let new_len = self.constructor.builder.i_sub(u32_ty, None, end_id, start_id)?;
        let view_struct_type = self.constructor.get_or_create_struct_type(vec![u32_ty, u32_ty]);
        Ok(self.constructor.builder.composite_construct(view_struct_type, None, [new_offset, new_len])?)
    }

    /// Slice a value (composite) array by extracting elements and constructing a new array.
    fn slice_composite(
        &mut self,
        arr: spirv::Word,
        start_id: spirv::Word,
        end_id: spirv::Word,
        result_ty: spirv::Word,
    ) -> Result<spirv::Word> {
        let start =
            self.constructor
                .get_const_i32_value(start_id)
                .ok_or_else(|| err_spirv!("slice_composite: start must be a constant"))? as u32;
        let end = self
            .constructor
            .get_const_i32_value(end_id)
            .ok_or_else(|| err_spirv!("slice_composite: end must be a constant"))? as u32;
        if end <= start {
            bail_spirv!(
                "slice_composite: end ({}) must be greater than start ({})",
                end,
                start
            );
        }
        let elem_type = self.constructor.get_array_element_type(result_ty)?;
        let mut elements = Vec::with_capacity((end - start) as usize);
        for i in start..end {
            let elem = self.constructor.builder.composite_extract(elem_type, None, arr, [i])?;
            elements.push(elem);
        }
        Ok(self.constructor.builder.composite_construct(result_ty, None, elements)?)
    }

    /// Lower an index operation, dispatching based on the array variant.
    fn lower_index(
        &mut self,
        base: ValueRef,
        index: ValueRef,
        result_ty: spirv::Word,
    ) -> Result<spirv::Word> {
        let base_ty = self.get_value_type_ref(base);
        let base_id = self.get_value_ref(base)?;
        let index_id = self.get_value_ref(index)?;

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

            _ if base_ty.is_array() => {
                let variant = base_ty.array_variant().expect("Array has variant");
                let elem = base_ty.elem_type().expect("Array has elem");

                if types::is_array_variant_view(variant) {
                    // View variant: {offset, len} struct; backing buffer
                    // recovered from `view_buffer_id` provenance.
                    self.lower_view_index(
                        base.as_ssa().expect("view base must be SSA"),
                        base_id,
                        index_id,
                        result_ty,
                        elem,
                    )
                } else if types::is_array_variant_virtual(variant) {
                    // Virtual variant: {start, step, len} - computed array
                    self.lower_virtual_index(base_id, index_id, result_ty)
                } else if types::is_array_variant_bounded(variant) {
                    // Bounded variant: {buffer: [N]T, len: u32} struct.
                    // Extract the buffer (member 0), then index it as a Composite.
                    let n = match base_ty.array_size().expect("Array has size") {
                        PolyType::Constructed(TypeName::Size(n), _) => *n as u32,
                        _ => bail_spirv!("Bounded array must have Size(N) capacity"),
                    };
                    let size_const = self.constructor.const_u32(n);
                    let buf_ty = self.constructor.builder.type_array(result_ty, size_const);
                    self.constructor.array_elem_cache.insert(buf_ty, result_ty);
                    let buf_id =
                        self.constructor.builder.composite_extract(buf_ty, None, base_id, [0u32])?;
                    if let Some(const_idx) = self.try_resolve_const_index(index) {
                        Ok(self.constructor.builder.composite_extract(
                            result_ty,
                            None,
                            buf_id,
                            [const_idx],
                        )?)
                    } else {
                        // Synthesize a composite [N]T base type for the helper.
                        let composite_ty = PolyType::Constructed(
                            TypeName::Array,
                            vec![
                                elem.clone(),
                                base_ty.array_size().expect("Array has size").clone(),
                                PolyType::Constructed(TypeName::ArrayVariantComposite, vec![]),
                            ],
                        );
                        self.lower_composite_index(buf_id, index_id, result_ty, &composite_ty)
                    }
                } else {
                    // Composite variant: SPIR-V array value
                    // Check for compile-time constant index for OpCompositeExtract
                    if let Some(const_idx) = self.try_resolve_const_index(index) {
                        Ok(self.constructor.builder.composite_extract(
                            result_ty,
                            None,
                            base_id,
                            [const_idx],
                        )?)
                    } else {
                        self.lower_composite_index(base_id, index_id, result_ty, &base_ty)
                    }
                }
            }

            // Vec types - use vector_extract_dynamic
            PolyType::Constructed(TypeName::Vec, _) => {
                Ok(self.constructor.builder.vector_extract_dynamic(result_ty, None, base_id, index_id)?)
            }

            _ => bail_spirv!("Index called on non-array/non-pointer type: {:?}", base_ty),
        }
    }

    /// Lower indexing into a View array ({buffer_id, offset, len} handle-based struct).
    /// Try to resolve a ValueRef to a constant u32 index.
    /// Handles both inline ValueRef::Const and SSA instructions that produce constants.
    fn try_resolve_const_index(&self, vr: ValueRef) -> Option<u32> {
        match vr {
            ValueRef::Const(ConstantValue::U32(i)) => Some(i),
            ValueRef::Const(ConstantValue::I32(i)) => Some(i as u32),
            ValueRef::Ssa(id) => {
                let inst_id = match self.body.inner.values.get(id)?.def {
                    crate::ssa::framework::ValueDef::Inst { inst } => inst,
                    _ => return None,
                };
                match &self.body.inner.insts.get(inst_id)?.data {
                    InstKind::Op {
                        tag: crate::op::OpTag::Int(s) | crate::op::OpTag::Uint(s),
                        ..
                    } => s.parse::<u32>().ok().or_else(|| s.parse::<i32>().ok().map(|i| i as u32)),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn lower_view_index(
        &mut self,
        view_ssa: ValueId,
        view_id: spirv::Word,
        index_id: spirv::Word,
        result_ty: spirv::Word,
        elem_ty: &PolyType<TypeName>,
    ) -> Result<spirv::Word> {
        let u32_ty = self.constructor.u32_type;

        // Buffer-var lookup goes through compile-time provenance, not runtime
        // struct extraction — field 0 of the view is redundant scaffolding.
        let buf_id = self.view_buffer_id.get(&view_ssa).copied().ok_or_else(|| {
            err_spirv_at!(
                self.blame_span(),
                "lower_view_index: no buffer provenance for value {:?}",
                view_ssa
            )
        })?;
        let (buffer_var, _) =
            self.constructor.buffer_vars.get(buf_id as usize).copied().ok_or_else(|| {
                err_spirv_at!(self.blame_span(), "view_buffer_id: unknown buffer_id {}", buf_id)
            })?;
        let offset_val = self.constructor.builder.composite_extract(u32_ty, None, view_id, [0u32])?;

        // TODO: The view struct stores {u32, u32, u32} but the language uses i32 for
        // indices everywhere. This bitcast papers over the mismatch. The view struct
        // fields should be i32 to match the rest of the language, eliminating this cast
        // and the u32/i32 inconsistency throughout the SOAC lowering pipeline.
        let index_u32 = self.constructor.builder.bitcast(self.constructor.u32_type, None, index_id)?;

        // Compute final index = offset + index
        let final_index =
            self.constructor.builder.i_add(self.constructor.u32_type, None, offset_val, index_u32)?;

        // Access chain directly on the buffer variable
        let elem_spirv = self.constructor.polytype_to_spirv(elem_ty);
        let elem_ptr_type =
            self.constructor.get_or_create_ptr_type(spirv::StorageClass::StorageBuffer, elem_spirv);
        let zero = self.constructor.const_u32(0);
        let elem_ptr =
            self.constructor.builder.access_chain(elem_ptr_type, None, buffer_var, [zero, final_index])?;
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
        id: crate::builtins::BuiltinId,
        builtin: &BuiltinLowering,
        dispatch_name: &str,
        value_refs: &[ValueRef],
        arg_ids: &[spirv::Word],
        result_ty: spirv::Word,
        inst: &WynInstNode,
    ) -> Result<spirv::Word> {
        match builtin {
            BuiltinLowering::PrimOp(prim_op) => self.lower_primop(prim_op, arg_ids, result_ty),
            BuiltinLowering::LinkedSpirv(linkage_name) => {
                let func_id = self
                    .constructor
                    .linked_functions
                    .get(*linkage_name)
                    .copied()
                    .ok_or_else(|| err_spirv!("Unknown linked function: {}", linkage_name))?;
                Ok(self.constructor.builder.function_call(result_ty, None, func_id, arg_ids.to_vec())?)
            }
            BuiltinLowering::NotLowered => {
                bail_spirv!(
                    "NotLowered builtin '{}' reached backend dispatch — \
                     promote it to `BuiltinLowering::ByBuiltinId` (or another typed variant)",
                    dispatch_name
                )
            }
            BuiltinLowering::ExtInstSplat { ext, splat_args } => {
                // GLSL.std.450 ext-inst with operand splatting.
                // Splat each scalar at the named positions to vec
                // width before emitting `OpExtInst` — required
                // because the instruction expects every operand
                // to match the result type.
                let mut operands: Vec<Operand> = arg_ids.iter().map(|&id| Operand::IdRef(id)).collect();
                let result_ssa_ty = inst.result.map(|r| self.body.inner.value_type(r).clone());
                let result_is_vec = result_ssa_ty.as_ref().is_some_and(|t| t.is_vec());
                if result_is_vec {
                    let result_ssa_ty = result_ssa_ty.as_ref().unwrap();
                    for &pos in *splat_args {
                        if self.get_value_type_ref(value_refs[pos]).is_scalar() {
                            let splatted = self.splat_scalar(arg_ids[pos], result_ssa_ty, result_ty)?;
                            operands[pos] = Operand::IdRef(splatted);
                        }
                    }
                }
                let glsl = self.constructor.glsl_ext_inst_id;
                Ok(self.constructor.builder.ext_inst(result_ty, None, glsl, *ext, operands)?)
            }
            BuiltinLowering::ByBuiltinId => {
                let known = catalog().known();
                if id == known.uninit {
                    // Zero-initialized value (OpConstantNull), cached by type.
                    if let Some(&cached) = self.constructor.null_const_cache.get(&result_ty) {
                        Ok(cached)
                    } else {
                        let null_id = self.constructor.builder.constant_null(result_ty);
                        self.constructor.constant_ids.insert(null_id);
                        self.constructor.null_const_cache.insert(result_ty, null_id);
                        Ok(null_id)
                    }
                } else if id == known.array_with || id == known.array_with_in_place {
                    // _w_array_with(array, index, value) - array update.
                    // Same SPIR-V lowering for both flavors today — SPIR-V can
                    // already express OpCompositeInsert (literal idx) or a
                    // local-buffer round-trip (dynamic idx). An in-place
                    // optimization for the dynamic case is left as future work.
                    if arg_ids.len() != 3 {
                        bail_spirv!("ArrayWith requires 3 arguments");
                    }
                    let arr = arg_ids[0];
                    let idx = arg_ids[1];
                    let val = arg_ids[2];

                    // View-variant arrays write back into the backing storage
                    // buffer via OpAccessChain+OpStore. The "result" view is
                    // structurally the same as the input; carry the input's
                    // SPIR-V word and propagate buffer provenance to the
                    // result ValueId so downstream `ViewIndex` consumers
                    // resolve it.
                    let arr_ty = self.get_value_type_ref(value_refs[0]);
                    let is_view = arr_ty
                        .array_variant()
                        .map(|v| matches!(v, PolyType::Constructed(TypeName::ArrayVariantView, _)))
                        .unwrap_or(false);
                    if is_view {
                        let view_ssa = value_refs[0]
                            .as_ssa()
                            .ok_or_else(|| err_spirv!("array_with on view must take SSA view value"))?;
                        let buf_id = self.view_buffer_id.get(&view_ssa).copied().ok_or_else(|| {
                            err_spirv!("array_with view: no buffer provenance for {:?}", view_ssa)
                        })?;
                        let (buffer_var, _) =
                            self.constructor.buffer_vars.get(buf_id as usize).copied().ok_or_else(
                                || err_spirv!("array_with view: unknown buffer_id {}", buf_id),
                            )?;
                        let u32_ty = self.constructor.u32_type;
                        let base_offset =
                            self.constructor.builder.composite_extract(u32_ty, None, arr, [0u32])?;
                        let idx_u32 = self.constructor.builder.bitcast(u32_ty, None, idx)?;
                        let final_index =
                            self.constructor.builder.i_add(u32_ty, None, base_offset, idx_u32)?;
                        let elem_ty = arr_ty.elem_type().expect("View has elem").clone();
                        let elem_spirv = self.constructor.polytype_to_spirv(&elem_ty);
                        let elem_ptr_type = self
                            .constructor
                            .get_or_create_ptr_type(spirv::StorageClass::StorageBuffer, elem_spirv);
                        let zero = self.constructor.const_u32(0);
                        let elem_ptr = self.constructor.builder.access_chain(
                            elem_ptr_type,
                            None,
                            buffer_var,
                            [zero, final_index],
                        )?;
                        self.constructor.builder.store(elem_ptr, val, None, [])?;
                        if let Some(result_ssa) = inst.result {
                            self.view_buffer_id.insert(result_ssa, buf_id);
                        }
                        return Ok(arr);
                    }

                    let literal_idx = match value_refs.get(1).and_then(|vr| vr.as_const()) {
                        Some(ConstantValue::I32(v)) => Some(v as i32),
                        Some(ConstantValue::U32(v)) => Some(v as i32),
                        _ => self.constructor.get_const_i32_value(idx),
                    };
                    if let Some(literal_idx) = literal_idx {
                        Ok(self.constructor.builder.composite_insert(
                            result_ty,
                            None,
                            val,
                            arr,
                            [literal_idx as u32],
                        )?)
                    } else {
                        let arr_var = self.constructor.declare_variable("_array_with_tmp", result_ty)?;
                        self.constructor.builder.store(arr_var, arr, None, [])?;
                        let elem_ty = self.constructor.get_array_element_type(result_ty).map_err(|_| {
                            crate::err_spirv!(
                                "ArrayWith: element type not found for array type ID {}. \
                                 Unsized or view arrays may not support indexed writes.",
                                result_ty
                            )
                        })?;
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
                } else if id == known.length {
                    if arg_ids.len() != 1 {
                        bail_spirv!("length requires 1 argument");
                    }
                    let arr_ty = self.get_value_type_ref(value_refs[0]);
                    let variant = arr_ty
                        .array_variant()
                        .ok_or_else(|| err_spirv!("length: expected array type, got {:?}", arr_ty))?;
                    match variant {
                        // View: struct {buffer_ptr, offset, len} — len is u32 in
                        // the struct but the SSA result type is i32. Extract as
                        // u32 then bitcast.
                        // TODO: view struct should use i32 to match language conventions.
                        PolyType::Constructed(TypeName::ArrayVariantView, _) => {
                            let u32_ty = self.constructor.u32_type;
                            // View struct is {offset, len}; len is field 1.
                            let len_u32 = self.constructor.builder.composite_extract(
                                u32_ty,
                                None,
                                arg_ids[0],
                                [1u32],
                            )?;
                            Ok(self.constructor.builder.bitcast(result_ty, None, len_u32)?)
                        }
                        // Virtual (range): struct {start, step, len} — len field
                        // type matches element type (may be u32), but SSA result
                        // is i32. Extract with the actual field type, then
                        // bitcast if needed.
                        PolyType::Constructed(TypeName::ArrayVariantVirtual, _) => {
                            let elem_spirv = self
                                .constructor
                                .polytype_to_spirv(arr_ty.elem_type().expect("virtual array has elem"));
                            if elem_spirv == result_ty {
                                Ok(self.constructor.builder.composite_extract(
                                    result_ty,
                                    None,
                                    arg_ids[0],
                                    [2u32],
                                )?)
                            } else {
                                let len_raw = self.constructor.builder.composite_extract(
                                    elem_spirv,
                                    None,
                                    arg_ids[0],
                                    [2u32],
                                )?;
                                Ok(self.constructor.builder.bitcast(result_ty, None, len_raw)?)
                            }
                        }
                        // Composite: sized SPIR-V array — length is known from the type.
                        PolyType::Constructed(TypeName::ArrayVariantComposite, _) => {
                            match arr_ty.array_size().expect("Array has size") {
                                PolyType::Constructed(TypeName::Size(n), _) => {
                                    Ok(self.constructor.const_i32(*n as i32))
                                }
                                _ => bail_spirv!("length: composite array has unknown size"),
                            }
                        }
                        // Bounded: struct {buffer, len} — extract member 1 (the
                        // runtime count). The `len` field is already i32, matching
                        // SSA's length() result type.
                        PolyType::Constructed(TypeName::ArrayVariantBounded, _) => Ok(self
                            .constructor
                            .builder
                            .composite_extract(result_ty, None, arg_ids[0], [1u32])?),
                        _ => bail_spirv!("length: unknown array variant: {:?}", variant),
                    }
                } else if id == known.slice {
                    if arg_ids.len() != 3 {
                        bail_spirv!("_w_slice requires 3 arguments (arr, start, end)");
                    }
                    let arr = arg_ids[0];
                    let start_id = arg_ids[1];
                    let end_id = arg_ids[2];

                    let arr_ty = self.get_value_type_ref(value_refs[0]);
                    let is_view = arr_ty
                        .array_variant()
                        .map(|v| matches!(v, PolyType::Constructed(TypeName::ArrayVariantView, _)))
                        .unwrap_or(false);

                    if is_view {
                        let elem_ty = arr_ty.elem_type().expect("Array has elem").clone();
                        let u32_ty = self.constructor.u32_type;
                        let base_offset =
                            self.constructor.builder.composite_extract(u32_ty, None, arr, [0u32])?;
                        let result_is_composite = inst
                            .result
                            .map(|v| self.body.get_value_type(v))
                            .map(|t| {
                                t.array_variant()
                                    .map(|v| types::is_array_variant_composite(v))
                                    .unwrap_or(false)
                            })
                            .unwrap_or(false);

                        if result_is_composite {
                            let view_ssa = value_refs[0]
                                .as_ssa()
                                .ok_or_else(|| err_spirv!("slice_to_composite view operand must be SSA"))?;
                            let buf_id = self.view_buffer_id.get(&view_ssa).copied().ok_or_else(|| {
                                err_spirv!(
                                    "slice_to_composite: no buffer provenance for value {:?}",
                                    view_ssa
                                )
                            })?;
                            let (buffer_var, _) =
                                self.constructor.buffer_vars.get(buf_id as usize).copied().ok_or_else(
                                    || err_spirv!("slice_to_composite: unknown buffer_id {}", buf_id),
                                )?;
                            self.slice_view_to_composite(
                                arr,
                                buffer_var,
                                base_offset,
                                start_id,
                                end_id,
                                &elem_ty,
                                result_ty,
                            )
                        } else {
                            // View-to-view slice: provenance carries through
                            // to the result; emit the runtime struct without
                            // a buffer_id field.
                            let view_ssa = value_refs[0]
                                .as_ssa()
                                .ok_or_else(|| err_spirv!("slice_view_to_view view operand must be SSA"))?;
                            let buf_id = self.view_buffer_id.get(&view_ssa).copied().ok_or_else(|| {
                                err_spirv!(
                                    "slice_view_to_view: no buffer provenance for value {:?}",
                                    view_ssa
                                )
                            })?;
                            let result_word =
                                self.slice_view_to_view(arr, base_offset, start_id, end_id)?;
                            if let Some(result_ssa) = inst.result {
                                self.view_buffer_id.insert(result_ssa, buf_id);
                            }
                            Ok(result_word)
                        }
                    } else {
                        self.slice_composite(arr, start_id, end_id, result_ty)
                    }
                } else if id == known.storage_len {
                    if arg_ids.len() != 2 {
                        bail_spirv!("_w_storage_len requires 2 arguments (set, binding)");
                    }
                    let set = match value_refs[0].as_const() {
                        Some(ConstantValue::U32(v)) => v,
                        _ => self
                            .constructor
                            .uint_const_reverse
                            .get(&arg_ids[0])
                            .copied()
                            .ok_or_else(|| err_spirv!("_w_storage_len: set must be a u32 constant"))?,
                    };
                    let binding =
                        match value_refs[1].as_const() {
                            Some(ConstantValue::U32(v)) => v,
                            _ => self.constructor.uint_const_reverse.get(&arg_ids[1]).copied().ok_or_else(
                                || err_spirv!("_w_storage_len: binding must be a u32 constant"),
                            )?,
                        };
                    let &(buffer_var, _, _) =
                        self.constructor.storage_buffers.get(&(set, binding)).ok_or_else(|| {
                            err_spirv!("Storage buffer not found for set={}, binding={}", set, binding)
                        })?;
                    let len_u32 = self.constructor.builder.array_length(
                        self.constructor.u32_type,
                        None,
                        buffer_var,
                        0,
                    )?;
                    Ok(self.constructor.builder.bitcast(result_ty, None, len_u32)?)
                } else if id == known.thread_id {
                    let gid_var = self
                        .constructor
                        .global_invocation_id
                        .ok_or_else(|| err_spirv!("GlobalInvocationId not set for compute shader"))?;
                    let uvec3_type = self.constructor.get_or_create_vec_type(self.constructor.u32_type, 3);
                    let gid = self.constructor.builder.load(uvec3_type, None, gid_var, None, [])?;
                    Ok(self.constructor.builder.composite_extract(
                        self.constructor.u32_type,
                        None,
                        gid,
                        [0],
                    )?)
                } else if id == known.storage_index || id == known.storage_store {
                    bail_spirv!(
                        "{} reached backend dispatch — should be lowered to \
                         an InstKind::Load/Store side effect during EGIR conversion",
                        dispatch_name
                    )
                } else {
                    bail_spirv!(
                        "ByBuiltinId dispatch: unknown builtin id={:?} dispatch_name={:?}",
                        id,
                        dispatch_name
                    )
                }
            }
        }
    }

    fn lower_primop(
        &mut self,
        prim_op: &PrimOp,
        arg_ids: &[spirv::Word],
        result_ty: spirv::Word,
    ) -> Result<spirv::Word> {
        let glsl = self.constructor.glsl_ext_inst_id;
        let operands: Vec<Operand> = arg_ids.iter().map(|&id| Operand::IdRef(id)).collect();

        match prim_op {
            PrimOp::GlslExt(ext_op) => {
                Ok(self.constructor.builder.ext_inst(result_ty, None, glsl, *ext_op, operands)?)
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
            PrimOp::IsNan => {
                if arg_ids.len() != 1 {
                    bail_spirv!("isnan requires 1 arg");
                }
                Ok(self.constructor.builder.is_nan(result_ty, None, arg_ids[0])?)
            }
            PrimOp::IsInf => {
                if arg_ids.len() != 1 {
                    bail_spirv!("isinf requires 1 arg");
                }
                Ok(self.constructor.builder.is_inf(result_ty, None, arg_ids[0])?)
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
pub fn lower_ssa_program(program: &Program) -> Result<Vec<u32>> {
    // Use a thread with larger stack size for complex shaders
    const STACK_SIZE: usize = 16 * 1024 * 1024; // 16MB

    let program_clone = program.clone();

    let handle = std::thread::Builder::new()
        .stack_size(STACK_SIZE)
        .spawn(move || lower_ssa_program_impl(&program_clone))
        .expect("Failed to spawn lowering thread");

    handle.join().expect("Lowering thread panicked")
}

fn lower_ssa_program_impl(program: &Program) -> Result<Vec<u32>> {
    let mut constructor = Constructor::new();

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

    // Pre-create storage buffers for all entry point bindings so that
    // buffer-specialized functions (which reference set/binding directly) can
    // resolve them during lowering, even though they're lowered before entry points.
    for entry in &program.entry_points {
        for input in &entry.inputs {
            if let Some((set, binding)) = input.storage_binding {
                if !constructor.storage_buffers.contains_key(&(set, binding)) {
                    constructor.create_storage_buffer(&input.ty, set, binding);
                }
            }
        }
        for output in &entry.outputs {
            if let Some((set, binding)) = output.storage_binding {
                if !constructor.storage_buffers.contains_key(&(set, binding)) {
                    constructor.create_storage_buffer(&output.ty, set, binding);
                }
            }
        }
    }

    // Also pre-create buffers from each entry's `storage_bindings` — the
    // typed list of compiler-introduced bindings (e.g. parallelize's
    // partials/result intermediates) that aren't user-visible outputs.
    for entry in &program.entry_points {
        for sb in &entry.storage_bindings {
            if !constructor.storage_buffers.contains_key(&(sb.set, sb.binding)) {
                constructor.create_storage_buffer(&sb.elem_ty, sb.set, sb.binding);
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

    // Lower all entry points
    // Collect all bindings that ANY entry point writes to (outputs).
    // These must not be marked NonWritable even when read by another entry point.
    let written_bindings: std::collections::HashSet<(u32, u32)> = program
        .entry_points
        .iter()
        .flat_map(|e| e.outputs.iter().filter_map(|o| o.storage_binding))
        .collect();

    for entry in &program.entry_points {
        let (spirv_model, local_size) = match &entry.execution_model {
            ExecutionModel::Vertex => (spirv::ExecutionModel::Vertex, None),
            ExecutionModel::Fragment => (spirv::ExecutionModel::Fragment, None),
            ExecutionModel::Compute { local_size } => (spirv::ExecutionModel::GLCompute, Some(*local_size)),
        };

        entry_info.push((entry.name.clone(), spirv_model, local_size));
        lower_ssa_entry_point(&mut constructor, entry, &written_bindings)?;
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
                    if let Some((set, binding)) = input.storage_binding {
                        if let Some(&(var_id, _, _)) = constructor.storage_buffers.get(&(set, binding)) {
                            if !interfaces.contains(&var_id) {
                                interfaces.push(var_id);
                            }
                        }
                    }
                }
                for output in &entry.outputs {
                    if let Some((set, binding)) = output.storage_binding {
                        if let Some(&(var_id, _, _)) = constructor.storage_buffers.get(&(set, binding)) {
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
                    if let Some(&(var_id, _, _)) = constructor.storage_buffers.get(&(sb.set, sb.binding)) {
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

    Ok(constructor.builder.module().assemble())
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

/// Lower an SSA entry point to SPIR-V.
fn lower_ssa_entry_point(
    constructor: &mut Constructor,
    entry: &EntryPoint,
    written_bindings: &std::collections::HashSet<(u32, u32)>,
) -> Result<()> {
    let body = &entry.body;
    let is_compute = matches!(entry.execution_model, ExecutionModel::Compute { .. });

    // Create I/O variables for entry point
    let mut interfaces = Vec::new();

    // For compute shaders, ensure GlobalInvocationId is created and listed as interface
    if is_compute {
        if let Some(gid_var) = constructor.global_invocation_id {
            // Already created by a previous entry point — just add to this entry's interface
            interfaces.push(gid_var);
        } else {
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
    }

    // Create push constant block for compute shader broadcast inputs
    let pc_inputs: Vec<(usize, u32)> = entry
        .inputs
        .iter()
        .enumerate()
        .filter_map(|(i, inp)| inp.push_constant_offset.map(|off| (i, off)))
        .collect();
    let pc_var = if !pc_inputs.is_empty() {
        // Build member types for push constant block
        let member_types: Vec<spirv::Word> =
            pc_inputs.iter().map(|&(i, _)| constructor.polytype_to_spirv(&entry.inputs[i].ty)).collect();
        let member_offsets: Vec<u32> = pc_inputs.iter().map(|&(_, off)| off).collect();
        let member_poly_types: Vec<&PolyType<TypeName>> =
            pc_inputs.iter().map(|&(i, _)| &entry.inputs[i].ty).collect();

        let pc_struct = constructor.create_interface_block_type(
            InterfaceBlockKind::PushConstant,
            &member_types,
            &member_offsets,
            &member_poly_types,
        );

        let pc_ptr_type = constructor.get_or_create_ptr_type(spirv::StorageClass::PushConstant, pc_struct);
        // Reuse the same push constant variable across entry points in the same module.
        // SPIR-V allows at most one PushConstant variable per module.
        let var_id = if let Some(existing) = constructor.push_constant_var {
            existing
        } else {
            let var_id =
                constructor.builder.variable(pc_ptr_type, None, spirv::StorageClass::PushConstant, None);
            constructor.push_constant_var = Some(var_id);
            var_id
        };
        interfaces.push(var_id);
        Some(var_id)
    } else {
        None
    };

    // Handle inputs
    let mut location = 0u32;
    // Uniform-bound inputs need their access-chain + load deferred until
    // after `begin_function`. Each entry here: (input.name, var_id, value_type).
    let mut uniform_loads: Vec<(String, spirv::Word, spirv::Word)> = Vec::new();
    for input in &entry.inputs {
        // Push constant inputs are handled separately above
        if input.push_constant_offset.is_some() {
            continue;
        }

        let input_type = constructor.polytype_to_spirv(&input.ty);

        if let Some(IoDecoration::BuiltIn(builtin)) = &input.decoration {
            // WGSL's `@builtin(position)` is stage-aware (vertex-out vs
            // fragment-in), so the Wyn frontend lets either `position` or
            // `frag_coord` parse to `BuiltIn::Position`/`BuiltIn::FragCoord`
            // and trusts the backend to do the right thing for the stage.
            // SPIR-V's builtins are stage-specific: fragment inputs must
            // decorate as FragCoord, never Position (drivers silently
            // zero a Position-decorated fragment input).
            let stage_builtin = match (&entry.execution_model, builtin) {
                (ExecutionModel::Fragment, spirv::BuiltIn::Position) => spirv::BuiltIn::FragCoord,
                _ => *builtin,
            };
            // Built-in input
            let ptr_type = constructor.get_or_create_ptr_type(spirv::StorageClass::Input, input_type);
            let var_id = constructor.builder.variable(ptr_type, None, spirv::StorageClass::Input, None);
            constructor.builder.decorate(
                var_id,
                spirv::Decoration::BuiltIn,
                [Operand::BuiltIn(stage_builtin)],
            );
            constructor.env.insert(input.name.clone(), var_id);
            interfaces.push(var_id);

            // Track GlobalInvocationId for compute shaders
            if stage_builtin == spirv::BuiltIn::GlobalInvocationId {
                constructor.global_invocation_id = Some(var_id);
            }
        } else if let Some((set, binding)) = input.uniform_binding {
            // `#[uniform(set, binding)]` → Block-decorated struct in
            // Uniform storage class. Per Vulkan rules the struct must
            // wrap the value; member 0 carries the actual data.
            let block_struct = constructor.builder.type_struct(vec![input_type]);
            constructor.builder.decorate(block_struct, spirv::Decoration::Block, []);
            constructor.builder.member_decorate(
                block_struct,
                0,
                spirv::Decoration::Offset,
                [Operand::LiteralBit32(0)],
            );
            let ptr_type = constructor.get_or_create_ptr_type(spirv::StorageClass::Uniform, block_struct);
            let var_id = constructor.builder.variable(ptr_type, None, spirv::StorageClass::Uniform, None);
            constructor.builder.decorate(
                var_id,
                spirv::Decoration::DescriptorSet,
                [Operand::LiteralBit32(set)],
            );
            constructor.builder.decorate(
                var_id,
                spirv::Decoration::Binding,
                [Operand::LiteralBit32(binding)],
            );
            interfaces.push(var_id);
            uniform_loads.push((input.name.clone(), var_id, input_type));
        } else if let Some((set, binding)) = input.storage_binding {
            let var_id = constructor.create_storage_buffer(&input.ty, set, binding);
            // Mark input storage buffers as non-writable ONLY if no other
            // entry point writes to the same binding. In multi-entry modules
            // (e.g., reduce phase1 + phase2), the partials buffer is written
            // by phase1 and read by phase2 — it must stay writable.
            if !written_bindings.contains(&(set, binding)) {
                constructor.builder.decorate(
                    var_id,
                    spirv::Decoration::NonWritable,
                    std::iter::empty::<Operand>(),
                );
            }
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

    // Load push constant members via AccessChain from the push constant variable.
    if let Some(pc_var_id) = pc_var {
        for (member_idx, &(input_idx, _offset)) in pc_inputs.iter().enumerate() {
            let input = &entry.inputs[input_idx];
            let member_type = constructor.polytype_to_spirv(&input.ty);
            let member_ptr_type =
                constructor.get_or_create_ptr_type(spirv::StorageClass::PushConstant, member_type);
            let idx_const = constructor.const_u32(member_idx as u32);
            let access_chain =
                constructor.builder.access_chain(member_ptr_type, None, pc_var_id, [idx_const])?;
            let loaded = constructor.builder.load(member_type, None, access_chain, None, [])?;
            constructor.env.insert(input.name.clone(), loaded);
        }
    }

    // Load uniform members: each `#[uniform]` param is an OpVariable
    // pointing at a Block-decorated `{value_type}` struct in Uniform
    // storage. The body references the value, so AccessChain to
    // member 0 + Load and put the loaded value in env.
    for (name, var_id, value_type) in &uniform_loads {
        let member_ptr_type = constructor.get_or_create_ptr_type(spirv::StorageClass::Uniform, *value_type);
        let zero = constructor.const_i32(0);
        let access_chain = constructor.builder.access_chain(member_ptr_type, None, *var_id, [zero])?;
        let loaded = constructor.builder.load(*value_type, None, access_chain, None, [])?;
        constructor.env.insert(name.clone(), loaded);
    }

    // Load input values from their pointer variables.
    // Entry point inputs are SPIR-V Input variables (pointers), but the SSA body
    // expects loaded values. Load them now and update env with the loaded values.
    for input in &entry.inputs {
        // Skip storage buffers, push constants, and uniforms — each
        // uses a different access pattern handled above.
        if input.storage_binding.is_some()
            || input.push_constant_offset.is_some()
            || input.uniform_binding.is_some()
        {
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
    let _result = lower_ssa_body_for_entry(constructor, body, entry.span)?;

    constructor.end_function()?;

    // Clear output variables
    constructor.current_entry_outputs.clear();

    Ok(())
}
