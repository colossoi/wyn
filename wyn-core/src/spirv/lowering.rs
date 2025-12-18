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
use crate::mir::{self, Body, Def, Expr, ExprId, LambdaId, LambdaInfo, LocalId, LoopKind, Program};
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
    float_const_cache: HashMap<u32, spirv::Word>, // bits as u32
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
    mat_type_cache: HashMap<(spirv::Word, u32, u32), spirv::Word>, // (elem_type, rows, cols)
    struct_type_cache: HashMap<Vec<spirv::Word>, spirv::Word>,
    ptr_type_cache: HashMap<(spirv::StorageClass, spirv::Word), spirv::Word>,

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

    /// Lambda registry: LambdaId -> LambdaInfo
    lambda_registry: IdArena<LambdaId, LambdaInfo>,

    // Builtin function registry
    impl_source: ImplSource,

    //// In-place optimization: NodeIds of operations where input array can be reused
    inplace_nodes: HashSet<crate::ast::NodeId>,
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
            float_const_cache: HashMap::new(),
            bool_const_cache: HashMap::new(),
            current_block: None,
            variables_block: None,
            first_code_block: None,
            env: HashMap::new(),
            functions: HashMap::new(),
            glsl_ext_inst_id,
            vec_type_cache: HashMap::new(),
            mat_type_cache: HashMap::new(),
            struct_type_cache: HashMap::new(),
            ptr_type_cache: HashMap::new(),
            entry_point_interfaces: HashMap::new(),
            current_is_entry_point: false,
            current_output_vars: Vec::new(),
            current_input_vars: Vec::new(),
            current_used_globals: Vec::new(),
            global_constants: HashMap::new(),
            uniform_variables: HashMap::new(),
            uniform_types: HashMap::new(),
            uniform_load_cache: HashMap::new(),
            lambda_registry: IdArena::new(),
            impl_source: ImplSource::default(),
            inplace_nodes: HashSet::new(),
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
                        // Array type: args[0] is size, args[1] is element type
                        if args.len() < 2 {
                            panic!(
                                "BUG: Array type requires 2 arguments (size, element_type), got {}.",
                                args.len()
                            );
                        }
                        // Get element type from args[1]
                        let elem_type = self.ast_type_to_spirv(&args[1]);

                        // Extract size from args[0] - may be concrete Size(n) or Unsized for runtime arrays
                        match &args[0] {
                            PolyType::Constructed(TypeName::Size(n), _) => {
                                // Fixed-size array
                                let size_const = self.const_i32(*n as i32);
                                self.builder.type_array(elem_type, size_const)
                            }
                            PolyType::Constructed(TypeName::Unsized, _) => {
                                // Runtime array (unsized) - used for storage buffers
                                self.builder.type_runtime_array(elem_type)
                            }
                            _ => {
                                panic!(
                                    "BUG: Array type has invalid size argument: {:?}. This should have been resolved during type checking. \
                                     This typically happens when array size inference fails to constrain a size variable to a concrete value.",
                                    args[0]
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

    /// Get or create a matrix type
    /// SPIR-V matrices are column-major: mat<rows x cols> has `cols` column vectors of size `rows`
    fn get_or_create_mat_type(&mut self, elem_type: spirv::Word, rows: u32, cols: u32) -> spirv::Word {
        let key = (elem_type, rows, cols);
        if let Some(&ty) = self.mat_type_cache.get(&key) {
            return ty;
        }
        // Matrix column type is a vector with `rows` elements
        let col_type = self.get_or_create_vec_type(elem_type, rows);
        // Matrix has `cols` columns
        let ty = self.builder.type_matrix(col_type, cols);
        self.mat_type_cache.insert(key, ty);
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
    #[allow(dead_code)]
    fn const_u32(&mut self, value: u32) -> spirv::Word {
        self.builder.constant_bit32(self.u32_type, value)
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
                inputs,
                outputs,
                body,
                ..
            } => {
                // First, ensure all dependencies are lowered
                self.ensure_deps_lowered(body)?;

                lower_entry_point_from_def(&mut self.constructor, name, inputs, outputs, body)?;
            }
            Def::Constant { name, body, .. } => {
                // First, ensure all dependencies are lowered
                self.ensure_deps_lowered(body)?;

                // Evaluate the constant expression at compile time
                // TODO: Validate that body is a literal or compile-time foldable expression
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
                // Create a SPIR-V uniform variable
                let uniform_type = self.constructor.ast_type_to_spirv(ty);
                let ptr_type =
                    self.constructor.get_or_create_ptr_type(spirv::StorageClass::Uniform, uniform_type);
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

                // Store uniform variable ID and type for lookup
                self.constructor.uniform_variables.insert(name.clone(), var_id);
                self.constructor.uniform_types.insert(name.clone(), uniform_type);
            }
            Def::Storage {
                name,
                ty,
                set,
                binding,
                ..
            } => {
                // Create a SPIR-V storage buffer variable
                // TODO: Implement proper storage buffer lowering
                // For now, just register it similarly to uniforms
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
                Expr::Closure { lambda_name, .. } => {
                    self.ensure_lowered(lambda_name)?;
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

/// Lower a constant expression at compile time.
/// This is used for global constants which must be compile-time evaluable.
/// Currently supports: literals, references to other constants, and basic arithmetic.
fn lower_const_expr(constructor: &mut Constructor, body: &Body, expr_id: ExprId) -> Result<spirv::Word> {
    let ty = body.get_type(expr_id);
    match body.get_expr(expr_id) {
        Expr::Int(n) => {
            let val: i32 = n.parse().map_err(|_| err_spirv!("Invalid integer literal: {}", n))?;
            Ok(constructor.const_i32(val))
        }
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
        Expr::Array(elems) | Expr::Vector(elems) => {
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
        _ => Err(err_spirv!(
            "Global constants must be literals or compile-time foldable expressions"
        )),
    }
}

fn lower_expr(constructor: &mut Constructor, body: &Body, expr_id: ExprId) -> Result<spirv::Word> {
    let expr_ty = body.get_type(expr_id);
    let expr_node_id = body.get_node_id(expr_id);

    match body.get_expr(expr_id) {
        Expr::Int(s) => {
            let val: i32 = s.parse().map_err(|_| err_spirv!("Invalid integer literal: {}", s))?;
            Ok(constructor.const_i32(val))
        }

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
            Err(err_spirv!("Undefined local variable: {}", name))
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
                // Load from the uniform variable and cache the result
                let value_type_id = constructor
                    .uniform_types
                    .get(name)
                    .copied()
                    .ok_or_else(|| err_spirv!("Could not find type for uniform variable: {}", name))?;
                let load_id = constructor.builder.load(value_type_id, None, var_id, None, [])?;
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

            // For ForRange loops, also create a phi for the iteration variable
            let iter_var_phi = if let LoopKind::ForRange { var, .. } = kind {
                let var_name = body.get_local(*var).name.clone();
                let iter_phi_id = constructor.builder.id();
                constructor.env.insert(var_name.clone(), iter_phi_id);
                Some((var_name, iter_phi_id))
            } else {
                None
            };

            // Evaluate init_bindings to bind user variables from loop_var
            // These extractions reference loop_var which is now bound to the phi
            for (local_id, binding_expr_id) in init_bindings.iter() {
                let val = lower_expr(constructor, body, *binding_expr_id)?;
                let binding_name = body.get_local(*local_id).name.clone();
                constructor.env.insert(binding_name, val);
            }

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
                    bail_spirv!("For-in loops not yet implemented");
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

            // Return the loop_var phi value as loop result
            Ok(loop_var_phi_id)
        }

        Expr::Call { func, args } => {
            // Get the result type from the expression
            let result_type = constructor.ast_type_to_spirv(expr_ty);

            // Special case for map - extract lambda name from closure before lowering
            if func == "map" {
                // map closure array -> array
                // args[0] is closure tuple (_w_lambda_name, captures)
                // args[1] is input array
                if args.len() != 2 {
                    bail_spirv!("map requires 2 args (closure, array), got {}", args.len());
                }

                // Extract lambda/function name and captures from first argument
                // Can be either a Closure expression or a Global (named function reference)
                let (func_name, closure_val, is_empty_closure) = match body.get_expr(args[0]) {
                    Expr::Closure {
                        lambda_name,
                        captures,
                    } => {
                        let is_empty = is_empty_closure_type(body.get_type(*captures));
                        // For empty closures, use dummy i32(0) instead of lowering
                        // This avoids creating empty SPIR-V structs
                        let closure_val = if is_empty {
                            constructor.const_i32(0)
                        } else {
                            lower_expr(constructor, body, args[0])?
                        };
                        (lambda_name.clone(), closure_val, is_empty)
                    }
                    Expr::Global(name) => {
                        // Named function reference - treat like empty closure
                        (name.clone(), constructor.const_i32(0), true)
                    }
                    other => {
                        bail_spirv!(
                            "map callback must be a closure or function reference, got {:?}",
                            other
                        );
                    }
                };
                let lambda_name = func_name;
                let array_val = lower_expr(constructor, body, args[1])?;

                // Get input array element type from args[1]
                let arg1_ty = body.get_type(args[1]);
                let input_elem_type = match arg1_ty {
                    PolyType::Constructed(TypeName::Array, type_args) if type_args.len() == 2 => {
                        constructor.ast_type_to_spirv(&type_args[1])
                    }
                    _ => bail_spirv!("map input must be array type"),
                };

                // Get output array info from result type
                let (array_size, output_elem_mir_type) = match expr_ty {
                    PolyType::Constructed(TypeName::Array, type_args) if type_args.len() == 2 => {
                        let size = match &type_args[0] {
                            PolyType::Constructed(TypeName::Size(n), _) => *n as u32,
                            _ => bail_spirv!("Invalid array size type"),
                        };
                        (size, &type_args[1])
                    }
                    _ => bail_spirv!("map result must be array type"),
                };

                let output_elem_type = constructor.ast_type_to_spirv(output_elem_mir_type);

                // Look up the lambda function by name
                let lambda_func_id = *constructor
                    .functions
                    .get(&lambda_name)
                    .ok_or_else(|| err_spirv!("Lambda function not found: {}", lambda_name))?;

                // Check if we can do in-place update:
                // 1. This map call was marked as having a dead-after input array
                // 2. Element types match (f : T -> T)
                let can_inplace = constructor.inplace_nodes.contains(&expr_node_id)
                    && input_elem_type == output_elem_type;

                if can_inplace {
                    // In-place optimization: use OpCompositeInsert to update array in place
                    // This allows the SPIR-V optimizer to reuse the input array memory
                    let mut result = array_val;
                    for i in 0..array_size {
                        let input_elem =
                            constructor.builder.composite_extract(input_elem_type, None, array_val, [i])?;
                        let call_args =
                            if is_empty_closure { vec![input_elem] } else { vec![closure_val, input_elem] };
                        let result_elem = constructor.builder.function_call(
                            output_elem_type,
                            None,
                            lambda_func_id,
                            call_args,
                        )?;
                        // Insert the new element into the result array
                        result = constructor.builder.composite_insert(
                            result_type,
                            None,
                            result_elem,
                            result,
                            [i],
                        )?;
                    }
                    return Ok(result);
                } else {
                    // Build result array by calling lambda for each element
                    let mut result_elements = Vec::new();
                    for i in 0..array_size {
                        // Extract element from input array (using input element type)
                        let input_elem =
                            constructor.builder.composite_extract(input_elem_type, None, array_val, [i])?;

                        // Call lambda: for empty closures, only pass element; otherwise pass both
                        let call_args =
                            if is_empty_closure { vec![input_elem] } else { vec![closure_val, input_elem] };
                        let result_elem = constructor.builder.function_call(
                            output_elem_type,
                            None,
                            lambda_func_id,
                            call_args,
                        )?;
                        result_elements.push(result_elem);
                    }

                    // Construct result array
                    return Ok(constructor.builder.composite_construct(
                        result_type,
                        None,
                        result_elements,
                    )?);
                }
            }

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
                let can_inplace = constructor.inplace_nodes.contains(&expr_node_id);

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
                                        // Array length: extract size from array type
                                        if args.len() != 1 {
                                            bail_spirv!("length expects exactly 1 argument");
                                        }
                                        let arg0_ty = body.get_type(args[0]);
                                        if let PolyType::Constructed(TypeName::Array, type_args) = arg0_ty {
                                            match type_args.get(0) {
                                                Some(PolyType::Constructed(TypeName::Size(n), _)) => {
                                                    Ok(constructor.const_i32(*n as i32))
                                                }
                                                _ => bail_spirv!(
                                                    "Cannot determine compile-time array size for length: {:?}",
                                                    type_args.get(0)
                                                ),
                                            }
                                        } else {
                                            bail_spirv!("length called on non-array type: {:?}", arg0_ty)
                                        }
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
                                    Intrinsic::Replicate => {
                                        // replicate n val: create array of n copies of val
                                        if arg_ids.len() != 2 {
                                            bail_spirv!("replicate expects exactly 2 arguments");
                                        }
                                        // Extract array size from result type
                                        if let PolyType::Constructed(TypeName::Array, type_args) = expr_ty {
                                            if let Some(PolyType::Constructed(TypeName::Size(n), _)) =
                                                type_args.get(0)
                                            {
                                                // Build array by repeating the value
                                                let val_id = arg_ids[1]; // second arg is the value
                                                let elem_ids: Vec<_> = (0..*n).map(|_| val_id).collect();
                                                Ok(constructor.builder.composite_construct(
                                                    result_type,
                                                    None,
                                                    elem_ids,
                                                )?)
                                            } else {
                                                bail_spirv!(
                                                    "replicate: cannot determine array size at compile time"
                                                )
                                            }
                                        } else {
                                            bail_spirv!("replicate: result type is not an array")
                                        }
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
                                    Intrinsic::BitcastI32ToU32 => {
                                        // _w_bitcast_i32_to_u32(i) -> u32
                                        // Reinterpret i32 bits as u32
                                        let value_id = arg_ids[0];
                                        Ok(constructor.builder.bitcast(
                                            constructor.u32_type,
                                            None,
                                            value_id,
                                        )?)
                                    }
                                }
                            }
                            BuiltinImpl::CoreFn(core_fn_name) => {
                                // Library-level builtins implemented as normal functions in prelude
                                // Look up the function and call it
                                let func_id = *constructor
                                    .functions
                                    .get(core_fn_name)
                                    .ok_or_else(|| err_spirv!("CoreFn not found: {}", core_fn_name))?;

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

                    Ok(constructor.builder.composite_extract(result_type, None, composite_id, [index])?)
                }
                "index" => {
                    if args.len() != 2 {
                        bail_spirv!("index requires 2 args");
                    }
                    // Array indexing with OpAccessChain + OpLoad
                    let index_val = lower_expr(constructor, body, args[1])?;

                    let arg0_ty = body.get_type(args[0]);
                    let array_var = if types::is_pointer(arg0_ty) {
                        // It's a pointer, use it directly
                        lower_expr(constructor, body, args[0])?
                    } else {
                        // Need to store the value in a variable to get a pointer
                        let array_val = lower_expr(constructor, body, args[0])?;
                        let array_type = constructor.ast_type_to_spirv(arg0_ty);
                        let array_var = constructor.declare_variable("_w_index_tmp", array_type)?;
                        constructor.builder.store(array_var, array_val, None, [])?;
                        array_var
                    };

                    // Use OpAccessChain to get pointer to element
                    let elem_ptr_type =
                        constructor.builder.type_pointer(None, StorageClass::Function, result_type);
                    let elem_ptr =
                        constructor.builder.access_chain(elem_ptr_type, None, array_var, [index_val])?;

                    // Load the element
                    Ok(constructor.builder.load(result_type, None, elem_ptr, None, [])?)
                }
                "assert" => {
                    // Assertions are no-ops in release, return body
                    if args.len() >= 2 {
                        lower_expr(constructor, body, args[1])
                    } else {
                        Ok(constructor.const_i32(0))
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

        Expr::Closure { captures, .. } => {
            // The captures field points to a Tuple or Unit expression.
            // Just lower it directly - Tuple handles struct construction with proper
            // type caching, and Unit produces const_i32(0) for empty closures.
            lower_expr(constructor, body, *captures)
        }

        Expr::Range {
            start,
            step,
            end,
            kind,
        } => {
            // Extract constant values from range components
            let start_val = try_extract_const_int(body, *start)
                .ok_or_else(|| err_spirv!("Range start must be a compile-time constant integer"))?;
            let end_val = try_extract_const_int(body, *end)
                .ok_or_else(|| err_spirv!("Range end must be a compile-time constant integer"))?;

            // Calculate stride:
            // - If step is None, stride = 1
            // - If step is Some(s), it's the second element, so stride = s - start
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

            // Calculate count based on kind
            let count = match kind {
                crate::mir::RangeKind::Inclusive | crate::mir::RangeKind::Exclusive => {
                    // .. and ... inclusive
                    if stride > 0 {
                        ((end_val - start_val) / stride) + 1
                    } else {
                        ((start_val - end_val) / (-stride)) + 1
                    }
                }
                crate::mir::RangeKind::ExclusiveLt => {
                    // ..< exclusive end (positive direction)
                    if stride <= 0 {
                        bail_spirv!("Range ..< requires positive stride");
                    }
                    (end_val - start_val + stride - 1) / stride
                }
                crate::mir::RangeKind::ExclusiveGt => {
                    // ..> exclusive end (negative direction)
                    if stride >= 0 {
                        bail_spirv!("Range ..> requires negative stride");
                    }
                    (start_val - end_val + (-stride) - 1) / (-stride)
                }
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

        Expr::Tuple(elems) => {
            // Lower all element expressions
            let elem_ids: Vec<spirv::Word> =
                elems.iter().map(|&e| lower_expr(constructor, body, e)).collect::<Result<Vec<_>>>()?;

            // Get the tuple type
            let result_type = constructor.ast_type_to_spirv(expr_ty);

            // Construct the composite
            Ok(constructor.builder.composite_construct(result_type, None, elem_ids)?)
        }

        Expr::Array(elems) | Expr::Vector(elems) => {
            // Lower all element expressions
            let elem_ids: Vec<spirv::Word> =
                elems.iter().map(|&e| lower_expr(constructor, body, e)).collect::<Result<Vec<_>>>()?;

            // Get the array/vector type
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
    }
}

/// Try to extract a compile-time constant integer from an expression.
fn try_extract_const_int(body: &Body, expr_id: ExprId) -> Option<i32> {
    match body.get_expr(expr_id) {
        Expr::Int(s) => s.parse().ok(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compile_to_spirv(source: &str) -> Result<Vec<u32>> {
        // Use the typestate API to ensure proper compilation pipeline
        let (module_manager, mut node_counter) = crate::cached_module_manager();
        let parsed = crate::Compiler::parse(source, &mut node_counter).expect("Parsing failed");
        let (flattened, _backend) = parsed
            .desugar(&mut node_counter)
            .expect("Desugaring failed")
            .resolve(&module_manager)
            .expect("Name resolution failed")
            .fold_ast_constants()
            .type_check(&module_manager)
            .expect("Type checking failed")
            .alias_check()
            .expect("Alias checking failed")
            .flatten(&module_manager)
            .expect("Flattening failed");

        let inplace_info = crate::alias_checker::analyze_inplace(&flattened.mir);
        lower(&flattened.mir, &inplace_info)
    }

    #[test]
    fn test_simple_constant() {
        let spirv = compile_to_spirv("def x() = 42").unwrap();
        assert!(!spirv.is_empty());
        // SPIR-V magic number
        assert_eq!(spirv[0], 0x07230203);
    }

    #[test]
    fn test_simple_function() {
        let spirv = compile_to_spirv("def add(x, y) = x + y").unwrap();
        assert!(!spirv.is_empty());
        assert_eq!(spirv[0], 0x07230203);
    }

    #[test]
    fn test_let_binding() {
        let spirv = compile_to_spirv("def f() = let x = 1 in x + 2").unwrap();
        assert!(!spirv.is_empty());
        assert_eq!(spirv[0], 0x07230203);
    }

    #[test]
    fn test_arithmetic() {
        let spirv = compile_to_spirv("def f(x, y) = x * y + x / y - 1").unwrap();
        assert!(!spirv.is_empty());
        assert_eq!(spirv[0], 0x07230203);
    }

    #[test]
    fn test_nested_let() {
        let spirv = compile_to_spirv("def f() = let a = 1 in let b = 2 in a + b").unwrap();
        assert!(!spirv.is_empty());
        assert_eq!(spirv[0], 0x07230203);
    }

    #[test]
    fn test_if_expression() {
        let spirv = compile_to_spirv("def f(x) = if x == 0 then 1 else 2").unwrap();
        assert!(!spirv.is_empty());
        assert_eq!(spirv[0], 0x07230203);
    }

    #[test]
    fn test_comparisons() {
        let spirv = compile_to_spirv("def f(x, y) = if x < y then 1 else if x > y then 2 else 0").unwrap();
        assert!(!spirv.is_empty());
        assert_eq!(spirv[0], 0x07230203);
    }

    #[test]
    fn test_tuple_literal() {
        let spirv = compile_to_spirv("def f() = (1, 2, 3)").unwrap();
        assert!(!spirv.is_empty());
        assert_eq!(spirv[0], 0x07230203);
    }

    #[test]
    fn test_array_literal() {
        let spirv = compile_to_spirv("def f() = [1, 2, 3]").unwrap();
        assert!(!spirv.is_empty());
        assert_eq!(spirv[0], 0x07230203);
    }

    #[test]
    fn test_unary_negation() {
        let spirv = compile_to_spirv("def f(x) = -x").unwrap();
        assert!(!spirv.is_empty());
        assert_eq!(spirv[0], 0x07230203);
    }

    #[test]
    fn test_record_field_access() {
        let spirv = compile_to_spirv(
            r#"
def get_x(r:{x:i32, y:i32}) -> i32 = r.x
"#,
        )
        .unwrap();
        assert!(!spirv.is_empty());
        assert_eq!(spirv[0], 0x07230203);
    }

    #[test]
    fn test_closure_capture_access() {
        // This test uses tuple_access intrinsic for closure field access
        let spirv = compile_to_spirv(
            r#"
def test(x:i32) -> i32 =
    let f = |y:i32| x + y in
    f(10)
"#,
        )
        .unwrap();
        assert!(!spirv.is_empty());
        assert_eq!(spirv[0], 0x07230203);
    }

    #[test]
    fn test_polymorphic_dot2() {
        // Test polymorphic function with type parameters that need proper instantiation
        // This reproduces the primitives.wyn issue where Vec type has unresolved size variable
        let spirv = compile_to_spirv(
            r#"
def dot2<E, T>(v: T) -> E = dot(v, v)

def test_dot2_vec3(v: vec3f32) -> f32 = dot2(v)
"#,
        )
        .unwrap();
        assert!(!spirv.is_empty());
        assert_eq!(spirv[0], 0x07230203);
    }

    #[test]
    fn test_polymorphic_dot2_in_expression() {
        // Test dot2 used in a more complex expression like in primitives.wyn
        // sdCappedTorus: f32.sqrt(dot2(p) + ra*ra - 2.0*ra*k) - rb
        let spirv = compile_to_spirv(
            r#"
def dot2<E, T>(v: T) -> E = dot(v, v)

def sdCappedTorus(p: vec3f32, ra: f32, rb: f32, k: f32) -> f32 =
  f32.sqrt(dot2(p) + ra*ra - 2.0*ra*k) - rb
"#,
        )
        .unwrap();
        assert!(!spirv.is_empty());
        assert_eq!(spirv[0], 0x07230203);
    }

    #[test]
    fn test_polymorphic_dot2_vec2_and_vec3() {
        // Test dot2 with both vec2 and vec3 in same program (like primitives.wyn)
        let spirv = compile_to_spirv(
            r#"
def dot2<E, T>(v: T) -> E = dot(v, v)

def test_vec3(v: vec3f32) -> f32 = dot2(v)
def test_vec2(v: vec2f32) -> f32 = dot2(v)
def test_both(v3: vec3f32, v2: vec2f32) -> f32 = dot2(v3) + dot2(v2)
"#,
        )
        .unwrap();
        assert!(!spirv.is_empty());
        assert_eq!(spirv[0], 0x07230203);
    }
}
