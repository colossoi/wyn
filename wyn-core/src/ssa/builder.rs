//! SSA function builder.
//!
//! Wraps wyn-ssa's generic FuncBuilder with wyn-specific convenience methods
//! for constructing InstKind instructions and managing ControlHeaders.

use crate::ast::TypeName;
use polytype::Type;

use super::types::{
    BlockId, ControlHeader, EffectToken, FuncBody, InstId, InstKind, Terminator, ValueId, ViewSource,
};

/// Error during function building.
pub type BuilderError = wyn_ssa::BuilderError;

/// Builder for constructing SSA functions.
pub struct FuncBuilder {
    inner: wyn_ssa::FuncBuilder<InstKind, EffectToken, Type<TypeName>>,
    control_headers: std::collections::HashMap<BlockId, ControlHeader>,
    params: Vec<(ValueId, Type<TypeName>, String)>,
    return_ty: Type<TypeName>,
    next_effect: u32,
    dps_output: Option<ValueId>,
}

impl FuncBuilder {
    /// Create a new function builder with the given parameters and return type.
    pub fn new(params: Vec<(Type<TypeName>, String)>, return_ty: Type<TypeName>) -> Self {
        let mut inner = wyn_ssa::FuncBuilder::new();
        let _entry = inner.entry();

        let mut func_params = Vec::new();
        for (i, (ty, name)) in params.into_iter().enumerate() {
            let value = inner.func_mut().add_function_param(i, ty.clone());
            func_params.push((value, ty, name));
        }

        FuncBuilder {
            inner,
            control_headers: std::collections::HashMap::new(),
            params: func_params,
            return_ty,
            next_effect: 1,
            dps_output: None,
        }
    }

    /// Get the value for a function parameter by index.
    pub fn get_param(&self, index: usize) -> ValueId {
        self.params[index].0
    }

    /// Get the number of function parameters.
    pub fn num_params(&self) -> usize {
        self.params.len()
    }

    /// The entry effect token (always EffectToken(0)).
    pub fn entry_effect(&self) -> EffectToken {
        EffectToken(0)
    }

    /// Allocate a new effect token.
    pub fn alloc_effect(&mut self) -> EffectToken {
        let token = EffectToken(self.next_effect);
        self.next_effect += 1;
        token
    }

    /// Set the DPS output parameter.
    pub fn set_dps_output(&mut self, value: ValueId) {
        self.dps_output = Some(value);
    }

    /// Get the entry block ID.
    pub fn entry(&self) -> BlockId {
        self.inner.entry()
    }

    /// Create a new basic block.
    pub fn create_block(&mut self) -> BlockId {
        self.inner.create_block()
    }

    /// Create a new basic block with parameters.
    pub fn create_block_with_params(
        &mut self,
        param_types: Vec<Type<TypeName>>,
    ) -> (BlockId, Vec<ValueId>) {
        self.inner.create_block_with_params(param_types)
    }

    /// Add a parameter to an existing block.
    pub fn add_block_param(&mut self, block: BlockId, ty: Type<TypeName>) -> ValueId {
        self.inner.add_block_param(block, ty)
    }

    /// Create a new block with named parameters (names are discarded).
    pub fn create_block_with_named_params(
        &mut self,
        params: Vec<(Type<TypeName>, String)>,
    ) -> (BlockId, Vec<ValueId>) {
        let types: Vec<Type<TypeName>> = params.into_iter().map(|(ty, _)| ty).collect();
        self.inner.create_block_with_params(types)
    }

    /// Switch to building in the specified block.
    pub fn switch_to_block(&mut self, block: BlockId) -> Result<(), BuilderError> {
        self.inner.switch_to_block(block)
    }

    /// Switch to a block without checking if the previous block is terminated.
    pub fn switch_to_block_unchecked(&mut self, block: BlockId) {
        self.inner.switch_to_block_unchecked(block)
    }

    /// Get the current block, if any.
    pub fn current_block(&self) -> Option<BlockId> {
        self.inner.current_block()
    }

    /// Check if the current block is terminated.
    pub fn is_current_terminated(&self) -> bool {
        self.current_block()
            .map(|b| !matches!(self.inner.func().blocks[b].term, Terminator::Unreachable))
            .unwrap_or(false)
    }

    /// Push an instruction that produces a value.
    pub fn push_inst(&mut self, kind: InstKind, ty: Type<TypeName>) -> Result<ValueId, BuilderError> {
        self.inner.push_inst(kind, ty, None)
    }

    /// Push an instruction with explicit effects.
    pub fn push_inst_with_effects(
        &mut self,
        kind: InstKind,
        ty: Type<TypeName>,
        effects: Option<(EffectToken, EffectToken)>,
    ) -> Result<ValueId, BuilderError> {
        self.inner.push_inst(kind, ty, effects)
    }

    /// Push an instruction that produces no value (e.g., Store).
    pub fn push_void_inst(&mut self, kind: InstKind) -> Result<InstId, BuilderError> {
        self.inner.push_void_inst(kind, None)
    }

    /// Push a void instruction with explicit effects.
    pub fn push_void_inst_with_effects(
        &mut self,
        kind: InstKind,
        effects: Option<(EffectToken, EffectToken)>,
    ) -> Result<InstId, BuilderError> {
        self.inner.push_void_inst(kind, effects)
    }

    /// Set the terminator for the current block.
    pub fn terminate(&mut self, term: Terminator) -> Result<(), BuilderError> {
        self.inner.terminate(term)
    }

    /// Finish building and return the function body.
    pub fn finish(self) -> Result<FuncBody, BuilderError> {
        let func = self.inner.finish()?;
        Ok(FuncBody {
            inner: func,
            control_headers: self.control_headers,
            params: self.params,
            return_ty: self.return_ty,
            next_effect: self.next_effect,
            dps_output: self.dps_output,
        })
    }

    /// Finish without checking termination (for testing).
    pub fn finish_unchecked(self) -> FuncBody {
        FuncBody {
            inner: self.inner.finish_unchecked(),
            control_headers: self.control_headers,
            params: self.params,
            return_ty: self.return_ty,
            next_effect: self.next_effect,
            dps_output: self.dps_output,
        }
    }

    /// Access the underlying function (read-only).
    pub fn func(&self) -> &wyn_ssa::Function<InstKind, EffectToken, Type<TypeName>> {
        self.inner.func()
    }

    // =========================================================================
    // Convenience methods for common instructions
    // =========================================================================

    pub fn push_int(&mut self, value: &str, ty: Type<TypeName>) -> Result<ValueId, BuilderError> {
        self.push_inst(InstKind::Int(value.to_string()), ty)
    }

    pub fn push_float(&mut self, value: &str, ty: Type<TypeName>) -> Result<ValueId, BuilderError> {
        self.push_inst(InstKind::Float(value.to_string()), ty)
    }

    pub fn push_bool(&mut self, value: bool) -> Result<ValueId, BuilderError> {
        let ty = Type::Constructed(TypeName::Bool, vec![]);
        self.push_inst(InstKind::Bool(value), ty)
    }

    pub fn push_binop(
        &mut self,
        op: &str,
        lhs: ValueId,
        rhs: ValueId,
        ty: Type<TypeName>,
    ) -> Result<ValueId, BuilderError> {
        self.push_inst(
            InstKind::BinOp {
                op: op.to_string(),
                lhs,
                rhs,
            },
            ty,
        )
    }

    pub fn push_unary(
        &mut self,
        op: &str,
        operand: ValueId,
        ty: Type<TypeName>,
    ) -> Result<ValueId, BuilderError> {
        self.push_inst(
            InstKind::UnaryOp {
                op: op.to_string(),
                operand,
            },
            ty,
        )
    }

    pub fn push_call(
        &mut self,
        func: &str,
        args: Vec<ValueId>,
        ty: Type<TypeName>,
    ) -> Result<ValueId, BuilderError> {
        self.push_inst(
            InstKind::Call {
                func: func.to_string(),
                args,
            },
            ty,
        )
    }

    pub fn push_intrinsic(
        &mut self,
        name: &str,
        args: Vec<ValueId>,
        ty: Type<TypeName>,
    ) -> Result<ValueId, BuilderError> {
        self.push_inst(
            InstKind::Intrinsic {
                name: name.to_string(),
                args,
            },
            ty,
        )
    }

    pub fn push_tuple(
        &mut self,
        elements: Vec<ValueId>,
        ty: Type<TypeName>,
    ) -> Result<ValueId, BuilderError> {
        self.push_inst(InstKind::Tuple(elements), ty)
    }

    pub fn push_project(
        &mut self,
        base: ValueId,
        index: u32,
        ty: Type<TypeName>,
    ) -> Result<ValueId, BuilderError> {
        self.push_inst(InstKind::Project { base, index }, ty)
    }

    pub fn push_index(
        &mut self,
        base: ValueId,
        index: ValueId,
        ty: Type<TypeName>,
    ) -> Result<ValueId, BuilderError> {
        self.push_inst(InstKind::Index { base, index }, ty)
    }

    pub fn push_array_lit(
        &mut self,
        elements: Vec<ValueId>,
        ty: Type<TypeName>,
    ) -> Result<ValueId, BuilderError> {
        self.push_inst(InstKind::ArrayLit { elements }, ty)
    }

    pub fn push_global(&mut self, name: &str, ty: Type<TypeName>) -> Result<ValueId, BuilderError> {
        self.push_inst(InstKind::Global(name.to_string()), ty)
    }

    pub fn push_output_ptr(&mut self, index: usize, ty: Type<TypeName>) -> Result<ValueId, BuilderError> {
        self.push_inst(InstKind::OutputPtr { index }, ty)
    }

    /// Push a load instruction. Returns the loaded value.
    pub fn push_load(
        &mut self,
        ptr: ValueId,
        result_ty: Type<TypeName>,
        effect_in: EffectToken,
    ) -> Result<ValueId, BuilderError> {
        let effect_out = self.alloc_effect();
        let block = self.current_block().ok_or(BuilderError::NoCurrentBlock)?;
        if !matches!(self.inner.func().blocks[block].term, Terminator::Unreachable) {
            return Err(BuilderError::BlockAlreadyTerminated(block));
        }
        Ok(self.inner.func_mut().append_inst(
            block,
            InstKind::Load { ptr },
            result_ty,
            Some((effect_in, effect_out)),
        ))
    }

    /// Push a store instruction. Returns the output effect token.
    pub fn push_store(
        &mut self,
        ptr: ValueId,
        value: ValueId,
        effect_in: EffectToken,
    ) -> Result<EffectToken, BuilderError> {
        let effect_out = self.alloc_effect();
        let block = self.current_block().ok_or(BuilderError::NoCurrentBlock)?;
        if !matches!(self.inner.func().blocks[block].term, Terminator::Unreachable) {
            return Err(BuilderError::BlockAlreadyTerminated(block));
        }
        self.inner.func_mut().append_void_inst(
            block,
            InstKind::Store { ptr, value },
            Some((effect_in, effect_out)),
        );
        Ok(effect_out)
    }

    // =========================================================================
    // Storage Buffer Helpers
    // =========================================================================

    pub fn emit_storage_view(
        &mut self,
        set: u32,
        binding: u32,
        view_ty: Type<TypeName>,
    ) -> Result<ValueId, BuilderError> {
        let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
        let set_val = self.push_int(&set.to_string(), u32_ty.clone())?;
        let binding_val = self.push_int(&binding.to_string(), u32_ty.clone())?;
        let storage_len = self.push_intrinsic(
            "_w_intrinsic_storage_len",
            vec![set_val, binding_val],
            u32_ty.clone(),
        )?;
        let zero = self.push_int("0", u32_ty)?;
        self.push_inst(
            InstKind::StorageView {
                source: ViewSource::Storage { set, binding },
                offset: zero,
                len: storage_len,
            },
            view_ty,
        )
    }

    pub fn emit_inherited_view(
        &mut self,
        parent: ValueId,
        offset: ValueId,
        len: ValueId,
        view_ty: Type<TypeName>,
    ) -> Result<ValueId, BuilderError> {
        self.push_inst(
            InstKind::StorageView {
                source: ViewSource::Inherited { parent },
                offset,
                len,
            },
            view_ty,
        )
    }

    pub fn emit_storage_store(
        &mut self,
        view: ValueId,
        index: ValueId,
        value: ValueId,
        elem_ty: Type<TypeName>,
        effect_in: EffectToken,
    ) -> Result<EffectToken, BuilderError> {
        let ptr = self.push_inst(InstKind::StorageViewIndex { view, index }, elem_ty)?;
        self.push_store(ptr, value, effect_in)
    }

    // =========================================================================
    // Control Flow Pattern Helpers
    // =========================================================================

    pub fn create_if_then_else(&mut self, result_ty: Type<TypeName>) -> IfThenElseBlocks {
        let then_block = self.create_block();
        let else_block = self.create_block();
        let (merge_block, merge_params) = self.create_block_with_params(vec![result_ty]);
        let result = merge_params[0];

        IfThenElseBlocks {
            then_block,
            else_block,
            merge_block,
            result,
        }
    }

    pub fn create_while_loop(&mut self, acc_ty: Type<TypeName>) -> WhileLoopBlocks {
        let (header, header_params) = self.create_block_with_params(vec![acc_ty.clone()]);
        let acc = header_params[0];
        let body = self.create_block();
        let (exit, exit_params) = self.create_block_with_params(vec![acc_ty]);
        let result = exit_params[0];

        self.control_headers.insert(
            header,
            ControlHeader::Loop {
                merge: exit,
                continue_block: body,
            },
        );

        WhileLoopBlocks {
            header,
            acc,
            body,
            exit,
            result,
        }
    }

    pub fn create_for_range_loop(&mut self, acc_ty: Type<TypeName>) -> ForRangeLoopBlocks {
        let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
        let (header, header_params) = self.create_block_with_params(vec![acc_ty.clone(), i32_ty]);
        let acc = header_params[0];
        let index = header_params[1];
        let body = self.create_block();
        let (exit, exit_params) = self.create_block_with_params(vec![acc_ty]);
        let result = exit_params[0];

        self.control_headers.insert(
            header,
            ControlHeader::Loop {
                merge: exit,
                continue_block: body,
            },
        );

        ForRangeLoopBlocks {
            header,
            acc,
            index,
            body,
            exit,
            result,
        }
    }

    pub fn mark_selection_header(&mut self, merge: BlockId) -> Result<(), BuilderError> {
        let block_id = self.current_block().ok_or(BuilderError::NoCurrentBlock)?;
        self.control_headers.insert(block_id, ControlHeader::Selection { merge });
        Ok(())
    }

    pub fn mark_loop_header(
        &mut self,
        header: BlockId,
        merge: BlockId,
        continue_block: BlockId,
    ) -> Result<(), BuilderError> {
        self.control_headers.insert(
            header,
            ControlHeader::Loop {
                merge,
                continue_block,
            },
        );
        Ok(())
    }

    pub fn set_control_header(&mut self, block: BlockId, control: ControlHeader) {
        self.control_headers.insert(block, control);
    }
}

/// Block structure for an if-then-else expression.
#[derive(Debug, Clone)]
pub struct IfThenElseBlocks {
    pub then_block: BlockId,
    pub else_block: BlockId,
    pub merge_block: BlockId,
    pub result: ValueId,
}

/// Block structure for a while loop.
#[derive(Debug, Clone)]
pub struct WhileLoopBlocks {
    pub header: BlockId,
    pub acc: ValueId,
    pub body: BlockId,
    pub exit: BlockId,
    pub result: ValueId,
}

/// Block structure for a for-range loop.
#[derive(Debug, Clone)]
pub struct ForRangeLoopBlocks {
    pub header: BlockId,
    pub acc: ValueId,
    pub index: ValueId,
    pub body: BlockId,
    pub exit: BlockId,
    pub result: ValueId,
}
