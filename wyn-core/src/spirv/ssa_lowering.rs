//! SSA to SPIR-V lowering.
//!
//! This module converts SSA FuncBody to SPIR-V instructions.
//! The explicit CFG in SSA form maps naturally to SPIR-V blocks.

use std::collections::HashMap;

use crate::ast::TypeName;
use crate::error::Result;
use crate::impl_source::{BuiltinImpl, PrimOp};
use crate::mir::ssa::{BlockId, FuncBody, Inst, InstKind, Terminator, ValueId};
use crate::types;
use crate::{bail_spirv, err_spirv};
use polytype::Type as PolyType;
use rspirv::dr::{InsertPoint, Operand};
use rspirv::spirv;
use spirv::StorageClass;

use super::lowering::Constructor;

/// Lower an SSA function body to SPIR-V.
///
/// This creates a SPIR-V function from the SSA representation:
/// - SSA blocks become SPIR-V blocks
/// - Block parameters become OpPhi nodes
/// - Terminators become branch instructions
pub(crate) fn lower_ssa_body(constructor: &mut Constructor, body: &FuncBody) -> Result<spirv::Word> {
    let mut ctx = SsaLowerCtx::new(constructor, body, false);
    ctx.lower()
}

/// Lower an SSA function body for an entry point.
///
/// Entry points are void functions that don't emit OpReturn/OpReturnValue.
/// Instead, this returns the computed result value which the caller stores
/// to output variables before emitting OpReturn.
pub(crate) fn lower_ssa_body_for_entry(
    constructor: &mut Constructor,
    body: &FuncBody,
) -> Result<spirv::Word> {
    let mut ctx = SsaLowerCtx::new(constructor, body, true);
    ctx.lower()
}

/// Context for lowering SSA to SPIR-V.
struct SsaLowerCtx<'a, 'b> {
    constructor: &'a mut Constructor,
    body: &'b FuncBody,
    /// Map from SSA ValueId to SPIR-V Word.
    value_map: HashMap<ValueId, spirv::Word>,
    /// Map from SSA BlockId to SPIR-V block label.
    block_map: HashMap<BlockId, spirv::Word>,
    /// Map from block to its SPIR-V block index (for phi insertion).
    block_indices: HashMap<BlockId, usize>,
    /// Phi node info: (target_block, param_idx, value, source_block)
    /// Collected during terminator lowering, inserted after all blocks processed.
    phi_inputs: Vec<(BlockId, usize, spirv::Word, spirv::Word)>,
    /// Whether this is an entry point (don't emit OpReturn)
    is_entry_point: bool,
    /// The return value (captured for entry points)
    return_value: Option<spirv::Word>,
}

impl<'a, 'b> SsaLowerCtx<'a, 'b> {
    fn new(constructor: &'a mut Constructor, body: &'b FuncBody, is_entry_point: bool) -> Self {
        SsaLowerCtx {
            constructor,
            body,
            value_map: HashMap::new(),
            block_map: HashMap::new(),
            block_indices: HashMap::new(),
            phi_inputs: Vec::new(),
            is_entry_point,
            return_value: None,
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
                let param_ty = self.constructor.ast_type_to_spirv(&param.ty);
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
                self.lower_terminator(block_id, term)?;
            }
        }

        // Insert phi nodes for all block parameters
        self.insert_phi_nodes()?;

        // For entry points, re-select the last block so caller can emit stores/ret
        // (insert_phi_nodes leaves no block selected)
        if self.is_entry_point && !self.body.blocks.is_empty() {
            let last_block_id = BlockId((self.body.blocks.len() - 1) as u32);
            if let Some(&block_idx) = self.block_indices.get(&last_block_id) {
                self.constructor.builder.select_block(Some(block_idx))?;
            }
        }

        // For entry points, return the captured return value
        // For regular functions, the return is handled by the terminator
        if self.is_entry_point {
            Ok(self.return_value.unwrap_or_else(|| self.constructor.const_i32(0)))
        } else {
            Ok(self.constructor.const_i32(0)) // Placeholder - actual return in terminator
        }
    }

    fn lower_inst(&mut self, inst: &Inst) -> Result<()> {
        let result_ty = self.constructor.ast_type_to_spirv(&inst.result_ty);

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

            InstKind::Unit => self.constructor.const_i32(0),

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

            InstKind::ArrayRange { .. } => {
                // ArrayRange is typically handled by expanding into actual values
                // For now, return an error - this should be lowered earlier
                bail_spirv!("ArrayRange should be expanded before SPIR-V lowering")
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
                    let value_type = self.constructor.ast_type_to_spirv(pointee_ty);
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
                } else if let Some(&func_id) = self.constructor.functions.get(name) {
                    // Function reference (including lambdas)
                    func_id
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
                self.lower_intrinsic(name, &arg_ids, result_ty)?
            }

            // Effectful operations - for now, just handle the simple cases
            InstKind::Alloca { elem_ty, .. } => {
                let elem_spirv_ty = self.constructor.ast_type_to_spirv(elem_ty);
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

                // Extract buffer_ptr and offset from view struct
                let buffer_ptr_type = result_ty; // The result is a pointer
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

                // Access chain into runtime array
                let zero = self.constructor.const_i32(0);
                self.constructor.builder.access_chain(result_ty, None, buffer_ptr, [zero, actual_index])?
            }

            InstKind::StorageViewLen { view } => {
                let view_id = self.get_value(*view)?;
                // Extract len from view struct (index 2)
                self.constructor.builder.composite_extract(result_ty, None, view_id, [2u32])?
            }
        };

        // Map the result
        if let Some(result_value) = inst.result {
            self.value_map.insert(result_value, spirv_result);
        }

        Ok(())
    }

    fn lower_terminator(&mut self, _block_id: BlockId, term: &Terminator) -> Result<()> {
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

                // For structured control flow, we need merge block
                // Find the merge point (common successor)
                // For now, emit without selection merge (works for simple cases)
                self.constructor.builder.branch_conditional(cond_id, then_label, else_label, [])?;
            }

            Terminator::Return(value) => {
                let value_id = self.get_value(*value)?;
                if self.is_entry_point {
                    // For entry points, capture the value but don't emit return
                    self.return_value = Some(value_id);
                } else {
                    self.constructor.builder.ret_value(value_id)?;
                }
            }

            Terminator::ReturnUnit => {
                if !self.is_entry_point {
                    self.constructor.builder.ret()?;
                }
                // For entry points, no return instruction needed
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
            let param_ty = self.constructor.ast_type_to_spirv(&param.ty);

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

            "_w_length" => {
                // Array length - extract from struct or use constant
                if args.len() != 1 {
                    bail_spirv!("_w_length requires 1 argument");
                }
                // For view arrays, length is at index 1 of the struct {ptr, len}
                // For now, try extracting index 1
                Ok(self.constructor.builder.composite_extract(result_ty, None, args[0], [1u32])?)
            }

            "_w_slice_storage_view" => {
                // Slice a storage view - for now, return the input (simplified)
                if args.len() != 3 {
                    bail_spirv!("_w_slice_storage_view requires 3 arguments");
                }
                // TODO: Proper slice implementation
                Ok(args[0])
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
                let thread_id_u32 =
                    self.constructor.builder.composite_extract(self.constructor.u32_type, None, gid, [0])?;
                // Convert to i32 (result_ty should be i32)
                Ok(self.constructor.builder.bitcast(result_ty, None, thread_id_u32)?)
            }

            "_w_storage_len" => {
                // Get the length of a storage buffer via OpArrayLength
                // Args: [set_id, binding_id] (as i32 constants that were lowered to SPIR-V)
                if args.len() != 2 {
                    bail_spirv!("_w_storage_len requires 2 arguments (set, binding)");
                }
                // The args are SPIR-V IDs of constants. We need to extract their values.
                // For now, since we're generating these from known integer literals in
                // soac_parallelize, we can look them up in the constructor's constant cache.
                let set = self
                    .constructor
                    .int_const_reverse
                    .get(&args[0])
                    .copied()
                    .ok_or_else(|| err_spirv!("_w_storage_len: set must be a constant"))?
                    as u32;
                let binding = self
                    .constructor
                    .int_const_reverse
                    .get(&args[1])
                    .copied()
                    .ok_or_else(|| err_spirv!("_w_storage_len: binding must be a constant"))?
                    as u32;

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

                // Convert to i32 if needed
                Ok(self.constructor.builder.bitcast(result_ty, None, len_u32)?)
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
            PolyType::Constructed(TypeName::Pointer, _) => {
                // Pointer indexing: access_chain + load
                let elem_ptr_type =
                    self.constructor.builder.type_pointer(None, StorageClass::Function, result_ty);
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
        _elem_ty: &PolyType<TypeName>,
    ) -> Result<spirv::Word> {
        // View has {buffer_ptr, offset, len}
        // Extract buffer_ptr (field 0) - it's a pointer to StorageBuffer
        let buffer_ptr_type =
            self.constructor.get_or_create_ptr_type(spirv::StorageClass::StorageBuffer, result_ty);
        let buffer_ptr = self.constructor.builder.composite_extract(buffer_ptr_type, None, view_id, [0])?;

        // Extract offset (field 1)
        let offset_val =
            self.constructor.builder.composite_extract(self.constructor.u32_type, None, view_id, [1])?;

        // Cast index to u32 if needed (index may be i32)
        let index_u32 = self.constructor.builder.bitcast(self.constructor.u32_type, None, index_id)?;

        // Compute final index = offset + index
        let final_index =
            self.constructor.builder.i_add(self.constructor.u32_type, None, offset_val, index_u32)?;

        // Get element pointer type
        let elem_ptr_type =
            self.constructor.get_or_create_ptr_type(spirv::StorageClass::StorageBuffer, result_ty);

        // OpAccessChain buffer_ptr[0][final_index]
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
            let spirv_array_type = self.constructor.ast_type_to_spirv(array_ty);
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

                            // Get element pointer type
                            let elem_ptr_ty = self.constructor.builder.type_pointer(
                                None,
                                spirv::StorageClass::Function,
                                result_ty, // This should be element type, simplified here
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
            | PrimOp::SMod => {
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
