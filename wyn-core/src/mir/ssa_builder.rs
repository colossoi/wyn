//! SSA function builder.
//!
//! Provides a safe API for constructing SSA functions, ensuring:
//! - Blocks are properly terminated before switching
//! - Values are allocated before use
//! - The entry block is set up correctly

use crate::ast::{NodeId, Span, TypeName};
use polytype::Type;

use super::ssa::{
    Block, BlockId, BlockParam, ControlHeader, EffectToken, FuncBody, Inst, InstId, InstKind, Terminator,
    ValueId,
};

/// Error during function building.
#[derive(Debug, Clone)]
pub enum BuilderError {
    /// Tried to add instructions without selecting a block.
    NoCurrentBlock,
    /// Current block was not terminated before finishing.
    UnterminatedBlock(BlockId),
    /// Block was terminated but instructions were added after.
    BlockAlreadyTerminated(BlockId),
    /// Function has no blocks.
    NoBlocks,
}

impl std::fmt::Display for BuilderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuilderError::NoCurrentBlock => write!(f, "No current block selected"),
            BuilderError::UnterminatedBlock(id) => write!(f, "Block {} has no terminator", id),
            BuilderError::BlockAlreadyTerminated(id) => {
                write!(f, "Block {} already terminated", id)
            }
            BuilderError::NoBlocks => write!(f, "Function has no blocks"),
        }
    }
}

impl std::error::Error for BuilderError {}

/// Builder for constructing SSA functions.
///
/// # Example
///
/// ```ignore
/// let mut builder = FuncBuilder::new(
///     vec![(Type::Constructed(TypeName::Int(32), vec![]), "x".to_string())],
///     Type::Constructed(TypeName::Int(32), vec![]),
/// );
///
/// // Entry block is current by default
/// let x = builder.get_param(0);
/// let one = builder.push_int("1", i32_ty.clone())?;
/// let result = builder.push_binop("+", x, one, i32_ty.clone())?;
/// builder.terminate(Terminator::Return(result))?;
///
/// let body = builder.finish()?;
/// ```
pub struct FuncBuilder {
    body: FuncBody,
    current_block: Option<BlockId>,
}

impl FuncBuilder {
    /// Create a new function builder.
    ///
    /// The entry block (BlockId(0)) is created automatically and set as current.
    pub fn new(params: Vec<(Type<TypeName>, String)>, return_ty: Type<TypeName>) -> Self {
        FuncBuilder {
            body: FuncBody::new(params, return_ty),
            current_block: Some(BlockId::ENTRY),
        }
    }

    /// Get the value for a function parameter by index.
    pub fn get_param(&self, index: usize) -> ValueId {
        self.body.params[index].0
    }

    /// Get the number of function parameters.
    pub fn num_params(&self) -> usize {
        self.body.params.len()
    }

    /// Get the entry effect token.
    pub fn entry_effect(&self) -> EffectToken {
        self.body.entry_effect
    }

    /// Allocate a new value with the given type.
    pub fn alloc_value(&mut self, ty: Type<TypeName>) -> ValueId {
        self.body.alloc_value(ty)
    }

    /// Allocate a new effect token.
    pub fn alloc_effect(&mut self) -> EffectToken {
        self.body.alloc_effect()
    }

    /// Create a new basic block with no parameters.
    pub fn create_block(&mut self) -> BlockId {
        let id = BlockId(self.body.blocks.len() as u32);
        self.body.blocks.push(Block::new());
        id
    }

    /// Create a new basic block with parameters.
    ///
    /// Returns the block ID and the values defined by the parameters.
    pub fn create_block_with_params(
        &mut self,
        param_types: Vec<Type<TypeName>>,
    ) -> (BlockId, Vec<ValueId>) {
        let id = BlockId(self.body.blocks.len() as u32);

        let mut params = Vec::new();
        let mut values = Vec::new();

        for ty in param_types {
            let value = self.body.alloc_value(ty.clone());
            params.push(BlockParam {
                value,
                ty,
                name: None,
            });
            values.push(value);
        }

        self.body.blocks.push(Block::with_params(params));
        (id, values)
    }

    /// Create a new block with named parameters.
    pub fn create_block_with_named_params(
        &mut self,
        params: Vec<(Type<TypeName>, String)>,
    ) -> (BlockId, Vec<ValueId>) {
        let id = BlockId(self.body.blocks.len() as u32);

        let mut block_params = Vec::new();
        let mut values = Vec::new();

        for (ty, name) in params {
            let value = self.body.alloc_value(ty.clone());
            block_params.push(BlockParam {
                value,
                ty,
                name: Some(name),
            });
            values.push(value);
        }

        self.body.blocks.push(Block::with_params(block_params));
        (id, values)
    }

    /// Switch to building in the specified block.
    ///
    /// The previous block must be terminated (have a terminator set).
    pub fn switch_to_block(&mut self, block: BlockId) -> Result<(), BuilderError> {
        // Check if previous block is terminated (if there was one)
        if let Some(prev) = self.current_block {
            if self.body.get_block(prev).terminator.is_none() {
                return Err(BuilderError::UnterminatedBlock(prev));
            }
        }
        self.current_block = Some(block);
        Ok(())
    }

    /// Switch to a block without checking if the previous block is terminated.
    ///
    /// Use this when you need to build blocks out of order.
    pub fn switch_to_block_unchecked(&mut self, block: BlockId) {
        self.current_block = Some(block);
    }

    /// Get the current block, if any.
    pub fn current_block(&self) -> Option<BlockId> {
        self.current_block
    }

    /// Check if the current block is terminated.
    pub fn is_current_terminated(&self) -> bool {
        self.current_block.map(|b| self.body.get_block(b).terminator.is_some()).unwrap_or(false)
    }

    /// Push an instruction to the current block.
    ///
    /// Returns the instruction ID and optionally the result value.
    fn push_inst_internal(
        &mut self,
        kind: InstKind,
        result_ty: Type<TypeName>,
        span: Span,
        node_id: NodeId,
        has_result: bool,
    ) -> Result<(InstId, Option<ValueId>), BuilderError> {
        let block_id = self.current_block.ok_or(BuilderError::NoCurrentBlock)?;

        // Check block isn't already terminated
        if self.body.get_block(block_id).terminator.is_some() {
            return Err(BuilderError::BlockAlreadyTerminated(block_id));
        }

        let result = if has_result { Some(self.body.alloc_value(result_ty.clone())) } else { None };

        let inst = Inst {
            result,
            result_ty,
            kind,
            span,
            node_id,
        };

        let inst_id = InstId(self.body.insts.len() as u32);
        self.body.insts.push(inst);
        self.body.get_block_mut(block_id).insts.push(inst_id);

        Ok((inst_id, result))
    }

    /// Push an instruction that produces a value.
    pub fn push_inst(
        &mut self,
        kind: InstKind,
        ty: Type<TypeName>,
        span: Span,
        node_id: NodeId,
    ) -> Result<ValueId, BuilderError> {
        let (_, result) = self.push_inst_internal(kind, ty, span, node_id, true)?;
        Ok(result.unwrap())
    }

    /// Push an instruction that produces no value (e.g., Store).
    pub fn push_void_inst(
        &mut self,
        kind: InstKind,
        span: Span,
        node_id: NodeId,
    ) -> Result<InstId, BuilderError> {
        let unit_ty = Type::Constructed(TypeName::Unit, vec![]);
        let (inst_id, _) = self.push_inst_internal(kind, unit_ty, span, node_id, false)?;
        Ok(inst_id)
    }

    /// Set the terminator for the current block.
    pub fn terminate(&mut self, term: Terminator) -> Result<(), BuilderError> {
        let block_id = self.current_block.ok_or(BuilderError::NoCurrentBlock)?;

        if self.body.get_block(block_id).terminator.is_some() {
            return Err(BuilderError::BlockAlreadyTerminated(block_id));
        }

        self.body.get_block_mut(block_id).terminator = Some(term);
        Ok(())
    }

    /// Finish building and return the function body.
    ///
    /// All blocks must be terminated.
    pub fn finish(self) -> Result<FuncBody, BuilderError> {
        if self.body.blocks.is_empty() {
            return Err(BuilderError::NoBlocks);
        }

        // Check all blocks are terminated
        for (i, block) in self.body.blocks.iter().enumerate() {
            if block.terminator.is_none() {
                return Err(BuilderError::UnterminatedBlock(BlockId(i as u32)));
            }
        }

        Ok(self.body)
    }

    /// Finish without checking termination (for testing).
    pub fn finish_unchecked(self) -> FuncBody {
        self.body
    }

    // =========================================================================
    // Convenience methods for common instructions
    // =========================================================================

    /// Push an integer literal.
    pub fn push_int(
        &mut self,
        value: &str,
        ty: Type<TypeName>,
        span: Span,
        node_id: NodeId,
    ) -> Result<ValueId, BuilderError> {
        self.push_inst(InstKind::Int(value.to_string()), ty, span, node_id)
    }

    /// Push a float literal.
    pub fn push_float(
        &mut self,
        value: &str,
        ty: Type<TypeName>,
        span: Span,
        node_id: NodeId,
    ) -> Result<ValueId, BuilderError> {
        self.push_inst(InstKind::Float(value.to_string()), ty, span, node_id)
    }

    /// Push a boolean literal.
    pub fn push_bool(&mut self, value: bool, span: Span, node_id: NodeId) -> Result<ValueId, BuilderError> {
        let ty = Type::Constructed(TypeName::Str("bool"), vec![]);
        self.push_inst(InstKind::Bool(value), ty, span, node_id)
    }

    /// Push a binary operation.
    pub fn push_binop(
        &mut self,
        op: &str,
        lhs: ValueId,
        rhs: ValueId,
        ty: Type<TypeName>,
        span: Span,
        node_id: NodeId,
    ) -> Result<ValueId, BuilderError> {
        self.push_inst(
            InstKind::BinOp {
                op: op.to_string(),
                lhs,
                rhs,
            },
            ty,
            span,
            node_id,
        )
    }

    /// Push a unary operation.
    pub fn push_unary(
        &mut self,
        op: &str,
        operand: ValueId,
        ty: Type<TypeName>,
        span: Span,
        node_id: NodeId,
    ) -> Result<ValueId, BuilderError> {
        self.push_inst(
            InstKind::UnaryOp {
                op: op.to_string(),
                operand,
            },
            ty,
            span,
            node_id,
        )
    }

    /// Push a function call.
    pub fn push_call(
        &mut self,
        func: &str,
        args: Vec<ValueId>,
        ty: Type<TypeName>,
        span: Span,
        node_id: NodeId,
    ) -> Result<ValueId, BuilderError> {
        self.push_inst(
            InstKind::Call {
                func: func.to_string(),
                args,
            },
            ty,
            span,
            node_id,
        )
    }

    /// Push an intrinsic call.
    pub fn push_intrinsic(
        &mut self,
        name: &str,
        args: Vec<ValueId>,
        ty: Type<TypeName>,
        span: Span,
        node_id: NodeId,
    ) -> Result<ValueId, BuilderError> {
        self.push_inst(
            InstKind::Intrinsic {
                name: name.to_string(),
                args,
            },
            ty,
            span,
            node_id,
        )
    }

    /// Push a tuple construction.
    pub fn push_tuple(
        &mut self,
        elements: Vec<ValueId>,
        ty: Type<TypeName>,
        span: Span,
        node_id: NodeId,
    ) -> Result<ValueId, BuilderError> {
        self.push_inst(InstKind::Tuple(elements), ty, span, node_id)
    }

    /// Push a tuple/struct projection.
    pub fn push_project(
        &mut self,
        base: ValueId,
        index: u32,
        ty: Type<TypeName>,
        span: Span,
        node_id: NodeId,
    ) -> Result<ValueId, BuilderError> {
        self.push_inst(InstKind::Project { base, index }, ty, span, node_id)
    }

    /// Push an array/vector index.
    pub fn push_index(
        &mut self,
        base: ValueId,
        index: ValueId,
        ty: Type<TypeName>,
        span: Span,
        node_id: NodeId,
    ) -> Result<ValueId, BuilderError> {
        self.push_inst(InstKind::Index { base, index }, ty, span, node_id)
    }

    /// Push a global reference.
    pub fn push_global(
        &mut self,
        name: &str,
        ty: Type<TypeName>,
        span: Span,
        node_id: NodeId,
    ) -> Result<ValueId, BuilderError> {
        self.push_inst(InstKind::Global(name.to_string()), ty, span, node_id)
    }

    /// Push an output pointer reference (for entry points).
    /// Returns a pointer to the output variable at the given index.
    pub fn push_output_ptr(
        &mut self,
        index: usize,
        ty: Type<TypeName>,
        span: Span,
        node_id: NodeId,
    ) -> Result<ValueId, BuilderError> {
        self.push_inst(InstKind::OutputPtr { index }, ty, span, node_id)
    }

    /// Push a store instruction.
    /// Returns the output effect token.
    pub fn push_store(
        &mut self,
        ptr: ValueId,
        value: ValueId,
        effect_in: EffectToken,
        span: Span,
        node_id: NodeId,
    ) -> Result<EffectToken, BuilderError> {
        let effect_out = self.alloc_effect();
        let unit_ty = Type::Constructed(TypeName::Unit, vec![]);
        self.push_inst(
            InstKind::Store {
                ptr,
                value,
                effect_in,
                effect_out,
            },
            unit_ty,
            span,
            node_id,
        )?;
        Ok(effect_out)
    }

    // =========================================================================
    // Control Flow Pattern Helpers
    // =========================================================================
    //
    // These helpers create the block structure for common control flow patterns.
    // They return structs containing the created blocks and values, allowing
    // the caller to fill in the block bodies.
    //
    // ## If-Then-Else Pattern
    //
    // ```text
    // entry:
    //     %cond = ...
    //     br_if %cond, then_block(), else_block()
    //
    // then_block:
    //     %then_result = ...
    //     br merge(%then_result)
    //
    // else_block:
    //     %else_result = ...
    //     br merge(%else_result)
    //
    // merge(%result: T):
    //     // use %result - this is the if expression's value
    // ```
    //
    // ## While Loop Pattern
    //
    // ```text
    // entry:
    //     %init = ...
    //     br header(%init)
    //
    // header(%acc: T):
    //     %cond = ...
    //     br_if %cond, body(), exit(%acc)
    //
    // body:
    //     %new_acc = ...
    //     br header(%new_acc)
    //
    // exit(%final: T):
    //     // %final is the loop result
    // ```
    //
    // ## For-Range Loop Pattern
    //
    // ```text
    // entry:
    //     %init = ...
    //     %bound = ...
    //     br header(%init, 0)
    //
    // header(%acc: T, %i: i32):
    //     %cond = lt %i, %bound
    //     br_if %cond, body(), exit(%acc)
    //
    // body:
    //     %new_acc = ... // use %i here
    //     %next_i = add %i, 1
    //     br header(%new_acc, %next_i)
    //
    // exit(%final: T):
    //     // %final is the loop result
    // ```

    /// Create the block structure for an if-then-else expression.
    ///
    /// This creates three blocks (then, else, merge) and returns them along with
    /// the merge block's result value. The caller is responsible for:
    /// 1. Emitting the condition in the current block
    /// 2. Terminating the current block with a CondBranch to then/else
    /// 3. Filling in the then block and branching to merge with then_result
    /// 4. Filling in the else block and branching to merge with else_result
    /// 5. Continuing from the merge block
    ///
    /// # Example
    ///
    /// ```ignore
    /// let if_blocks = builder.create_if_then_else(result_ty.clone());
    ///
    /// // Emit condition and branch
    /// let cond = builder.push_binop(...)?;
    /// builder.terminate(Terminator::CondBranch {
    ///     cond,
    ///     then_target: if_blocks.then_block,
    ///     then_args: vec![],
    ///     else_target: if_blocks.else_block,
    ///     else_args: vec![],
    /// })?;
    ///
    /// // Then block
    /// builder.switch_to_block(if_blocks.then_block)?;
    /// let then_result = ...;
    /// builder.terminate(Terminator::Branch {
    ///     target: if_blocks.merge_block,
    ///     args: vec![then_result],
    /// })?;
    ///
    /// // Else block
    /// builder.switch_to_block(if_blocks.else_block)?;
    /// let else_result = ...;
    /// builder.terminate(Terminator::Branch {
    ///     target: if_blocks.merge_block,
    ///     args: vec![else_result],
    /// })?;
    ///
    /// // Continue from merge
    /// builder.switch_to_block(if_blocks.merge_block)?;
    /// // if_blocks.result contains the selected value
    /// ```
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

    /// Create the block structure for a while loop.
    ///
    /// This creates header, body, and exit blocks. The header has a parameter
    /// for the accumulator, and the exit block has a parameter for the final result.
    ///
    /// The caller is responsible for:
    /// 1. Emitting the initial accumulator value in the current block
    /// 2. Branching to header with the initial value
    /// 3. In header: emitting the condition and branching to body/exit
    /// 4. In body: computing new accumulator and branching back to header
    /// 5. Continuing from exit with the final result
    ///
    /// # Example
    ///
    /// ```ignore
    /// // loop acc = init while cond do body
    /// let loop_blocks = builder.create_while_loop(acc_ty.clone());
    ///
    /// // Entry: branch to header with initial value
    /// let init = ...;
    /// builder.terminate(Terminator::Branch {
    ///     target: loop_blocks.header,
    ///     args: vec![init],
    /// })?;
    ///
    /// // Header: check condition
    /// builder.switch_to_block(loop_blocks.header)?;
    /// let acc = loop_blocks.acc;  // Current accumulator value
    /// let cond = ...;  // Condition using acc
    /// builder.terminate(Terminator::CondBranch {
    ///     cond,
    ///     then_target: loop_blocks.body,
    ///     then_args: vec![],
    ///     else_target: loop_blocks.exit,
    ///     else_args: vec![acc],
    /// })?;
    ///
    /// // Body: compute new accumulator
    /// builder.switch_to_block(loop_blocks.body)?;
    /// let new_acc = ...;
    /// builder.terminate(Terminator::Branch {
    ///     target: loop_blocks.header,
    ///     args: vec![new_acc],
    /// })?;
    ///
    /// // Exit: use final result
    /// builder.switch_to_block(loop_blocks.exit)?;
    /// // loop_blocks.result contains the final accumulator value
    /// ```
    pub fn create_while_loop(&mut self, acc_ty: Type<TypeName>) -> WhileLoopBlocks {
        let (header, header_params) = self.create_block_with_params(vec![acc_ty.clone()]);
        let acc = header_params[0];

        let body = self.create_block();

        let (exit, exit_params) = self.create_block_with_params(vec![acc_ty]);
        let result = exit_params[0];

        // Mark header as a loop header for SPIR-V structured control flow.
        // body is the continue block (it branches back to header).
        // exit is the merge block (where the loop exits to).
        self.body.blocks[header.index()].control = Some(ControlHeader::Loop {
            merge: exit,
            continue_block: body,
        });

        WhileLoopBlocks {
            header,
            acc,
            body,
            exit,
            result,
        }
    }

    /// Create the block structure for a for-range loop (for i < bound).
    ///
    /// This creates header, body, and exit blocks. The header has parameters
    /// for both the accumulator and the loop index.
    ///
    /// The caller is responsible for:
    /// 1. Emitting the initial accumulator and bound in the current block
    /// 2. Branching to header with (init, 0)
    /// 3. In header: comparing i < bound and branching to body/exit
    /// 4. In body: computing new accumulator and incrementing i
    /// 5. Continuing from exit with the final result
    ///
    /// # Example
    ///
    /// ```ignore
    /// // loop acc = init for i < bound do body
    /// let loop_blocks = builder.create_for_range_loop(acc_ty.clone());
    ///
    /// // Entry: compute init and bound, branch to header
    /// let init = ...;
    /// let bound = ...;
    /// let zero = builder.push_int("0", i32_ty, ...)?;
    /// builder.terminate(Terminator::Branch {
    ///     target: loop_blocks.header,
    ///     args: vec![init, zero],
    /// })?;
    ///
    /// // Header: check i < bound
    /// builder.switch_to_block(loop_blocks.header)?;
    /// let cond = builder.push_binop("<", loop_blocks.index, bound, bool_ty, ...)?;
    /// builder.terminate(Terminator::CondBranch {
    ///     cond,
    ///     then_target: loop_blocks.body,
    ///     then_args: vec![],
    ///     else_target: loop_blocks.exit,
    ///     else_args: vec![loop_blocks.acc],
    /// })?;
    ///
    /// // Body: compute new acc, increment i
    /// builder.switch_to_block(loop_blocks.body)?;
    /// let new_acc = ...;  // Can use loop_blocks.index here
    /// let one = builder.push_int("1", i32_ty, ...)?;
    /// let next_i = builder.push_binop("+", loop_blocks.index, one, i32_ty, ...)?;
    /// builder.terminate(Terminator::Branch {
    ///     target: loop_blocks.header,
    ///     args: vec![new_acc, next_i],
    /// })?;
    ///
    /// // Exit: use final result
    /// builder.switch_to_block(loop_blocks.exit)?;
    /// ```
    pub fn create_for_range_loop(&mut self, acc_ty: Type<TypeName>) -> ForRangeLoopBlocks {
        let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);

        let (header, header_params) = self.create_block_with_params(vec![acc_ty.clone(), i32_ty]);
        let acc = header_params[0];
        let index = header_params[1];

        let body = self.create_block();

        let (exit, exit_params) = self.create_block_with_params(vec![acc_ty]);
        let result = exit_params[0];

        // Mark header as a loop header for SPIR-V structured control flow.
        self.body.blocks[header.index()].control = Some(ControlHeader::Loop {
            merge: exit,
            continue_block: body,
        });

        ForRangeLoopBlocks {
            header,
            acc,
            index,
            body,
            exit,
            result,
        }
    }

    /// Mark the current block as a selection header (for if-then-else).
    ///
    /// Call this before emitting a CondBranch terminator for an if-then-else.
    /// The merge block is where both branches reconverge.
    pub fn mark_selection_header(&mut self, merge: BlockId) -> Result<(), BuilderError> {
        let block_id = self.current_block.ok_or(BuilderError::NoCurrentBlock)?;
        self.body.blocks[block_id.index()].control = Some(ControlHeader::Selection { merge });
        Ok(())
    }
}

/// Block structure for an if-then-else expression.
#[derive(Debug, Clone)]
pub struct IfThenElseBlocks {
    /// The then branch block (no parameters).
    pub then_block: BlockId,
    /// The else branch block (no parameters).
    pub else_block: BlockId,
    /// The merge block (has one parameter for the result).
    pub merge_block: BlockId,
    /// The result value (merge block's parameter).
    pub result: ValueId,
}

/// Block structure for a while loop.
#[derive(Debug, Clone)]
pub struct WhileLoopBlocks {
    /// Loop header block (has accumulator parameter).
    pub header: BlockId,
    /// The accumulator value (header's parameter).
    pub acc: ValueId,
    /// Loop body block (no parameters).
    pub body: BlockId,
    /// Loop exit block (has result parameter).
    pub exit: BlockId,
    /// The final result value (exit's parameter).
    pub result: ValueId,
}

/// Block structure for a for-range loop.
#[derive(Debug, Clone)]
pub struct ForRangeLoopBlocks {
    /// Loop header block (has accumulator and index parameters).
    pub header: BlockId,
    /// The accumulator value (header's first parameter).
    pub acc: ValueId,
    /// The loop index value (header's second parameter).
    pub index: ValueId,
    /// Loop body block (no parameters).
    pub body: BlockId,
    /// Loop exit block (has result parameter).
    pub exit: BlockId,
    /// The final result value (exit's parameter).
    pub result: ValueId,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::NodeId;

    fn dummy_span() -> Span {
        Span {
            start_line: 1,
            start_col: 1,
            end_line: 1,
            end_col: 1,
        }
    }

    fn dummy_node() -> NodeId {
        NodeId(0)
    }

    fn i32_ty() -> Type<TypeName> {
        Type::Constructed(TypeName::Int(32), vec![])
    }

    #[test]
    fn test_simple_function() {
        // Build: fn add(x: i32, y: i32) -> i32 { x + y }
        let mut builder = FuncBuilder::new(
            vec![(i32_ty(), "x".to_string()), (i32_ty(), "y".to_string())],
            i32_ty(),
        );

        let x = builder.get_param(0);
        let y = builder.get_param(1);

        let result = builder.push_binop("+", x, y, i32_ty(), dummy_span(), dummy_node()).unwrap();

        builder.terminate(Terminator::Return(result)).unwrap();

        let body = builder.finish().unwrap();

        assert_eq!(body.params.len(), 2);
        assert_eq!(body.num_blocks(), 1);
        assert_eq!(body.num_insts(), 1);
        assert_eq!(body.num_values(), 3); // x, y, result
    }

    #[test]
    fn test_conditional() {
        // Build: fn max(x: i32, y: i32) -> i32 { if x > y then x else y }
        let mut builder = FuncBuilder::new(
            vec![(i32_ty(), "x".to_string()), (i32_ty(), "y".to_string())],
            i32_ty(),
        );

        let x = builder.get_param(0);
        let y = builder.get_param(1);

        // Create blocks
        let then_block = builder.create_block();
        let else_block = builder.create_block();
        let (merge_block, merge_params) = builder.create_block_with_params(vec![i32_ty()]);
        let result = merge_params[0];

        // Entry: compare and branch
        let cond = builder
            .push_binop(
                ">",
                x,
                y,
                Type::Constructed(TypeName::Str("bool"), vec![]),
                dummy_span(),
                dummy_node(),
            )
            .unwrap();
        builder
            .terminate(Terminator::CondBranch {
                cond,
                then_target: then_block,
                then_args: vec![],
                else_target: else_block,
                else_args: vec![],
            })
            .unwrap();

        // Then block: return x
        builder.switch_to_block(then_block).unwrap();
        builder
            .terminate(Terminator::Branch {
                target: merge_block,
                args: vec![x],
            })
            .unwrap();

        // Else block: return y
        builder.switch_to_block(else_block).unwrap();
        builder
            .terminate(Terminator::Branch {
                target: merge_block,
                args: vec![y],
            })
            .unwrap();

        // Merge block: return result
        builder.switch_to_block(merge_block).unwrap();
        builder.terminate(Terminator::Return(result)).unwrap();

        let body = builder.finish().unwrap();

        assert_eq!(body.num_blocks(), 4); // entry, then, else, merge
        assert_eq!(body.get_block(merge_block).params.len(), 1);
    }

    #[test]
    fn test_unterminated_block_error() {
        let mut builder = FuncBuilder::new(vec![], Type::Constructed(TypeName::Unit, vec![]));

        let block2 = builder.create_block();

        // Try to switch without terminating entry block
        let result = builder.switch_to_block(block2);
        assert!(matches!(result, Err(BuilderError::UnterminatedBlock(_))));
    }

    #[test]
    fn test_finish_unterminated_error() {
        let builder = FuncBuilder::new(vec![], Type::Constructed(TypeName::Unit, vec![]));

        // Entry block not terminated
        let result = builder.finish();
        assert!(matches!(result, Err(BuilderError::UnterminatedBlock(_))));
    }

    // =========================================================================
    // Control Flow Pattern Tests
    // =========================================================================

    fn bool_ty() -> Type<TypeName> {
        Type::Constructed(TypeName::Str("bool"), vec![])
    }

    #[test]
    fn test_if_then_else_pattern() {
        // Build: fn abs(x: i32) -> i32 { if x < 0 then -x else x }
        let mut builder = FuncBuilder::new(vec![(i32_ty(), "x".to_string())], i32_ty());

        let x = builder.get_param(0);

        // Create the if-then-else structure
        let if_blocks = builder.create_if_then_else(i32_ty());

        // Entry: compare x < 0 and branch
        let zero = builder.push_int("0", i32_ty(), dummy_span(), dummy_node()).unwrap();
        let cond = builder.push_binop("<", x, zero, bool_ty(), dummy_span(), dummy_node()).unwrap();
        builder
            .terminate(Terminator::CondBranch {
                cond,
                then_target: if_blocks.then_block,
                then_args: vec![],
                else_target: if_blocks.else_block,
                else_args: vec![],
            })
            .unwrap();

        // Then block: compute -x and branch to merge
        builder.switch_to_block(if_blocks.then_block).unwrap();
        let neg_x = builder.push_unary("-", x, i32_ty(), dummy_span(), dummy_node()).unwrap();
        builder
            .terminate(Terminator::Branch {
                target: if_blocks.merge_block,
                args: vec![neg_x],
            })
            .unwrap();

        // Else block: use x and branch to merge
        builder.switch_to_block(if_blocks.else_block).unwrap();
        builder
            .terminate(Terminator::Branch {
                target: if_blocks.merge_block,
                args: vec![x],
            })
            .unwrap();

        // Merge block: return result
        builder.switch_to_block(if_blocks.merge_block).unwrap();
        builder.terminate(Terminator::Return(if_blocks.result)).unwrap();

        let body = builder.finish().unwrap();

        // Verify structure
        assert_eq!(body.num_blocks(), 4); // entry, then, else, merge
        assert_eq!(body.get_block(if_blocks.merge_block).params.len(), 1);

        // Verify with verifier
        let verify_result = crate::mir::ssa_verify::verify_func(&body);
        assert!(
            verify_result.is_ok(),
            "Verification failed: {:?}",
            verify_result.err()
        );
    }

    #[test]
    fn test_while_loop_pattern() {
        // Build: fn sum_to_n(n: i32) -> i32 { loop acc = 0 while acc < n do acc + 1 }
        // (This is a silly example - it's really "count to n" - but demonstrates the pattern)
        let mut builder = FuncBuilder::new(vec![(i32_ty(), "n".to_string())], i32_ty());

        let n = builder.get_param(0);

        // Create the while loop structure
        let loop_blocks = builder.create_while_loop(i32_ty());

        // Entry: branch to header with initial value 0
        let zero = builder.push_int("0", i32_ty(), dummy_span(), dummy_node()).unwrap();
        builder
            .terminate(Terminator::Branch {
                target: loop_blocks.header,
                args: vec![zero],
            })
            .unwrap();

        // Header: check acc < n
        builder.switch_to_block(loop_blocks.header).unwrap();
        let cond =
            builder.push_binop("<", loop_blocks.acc, n, bool_ty(), dummy_span(), dummy_node()).unwrap();
        builder
            .terminate(Terminator::CondBranch {
                cond,
                then_target: loop_blocks.body,
                then_args: vec![],
                else_target: loop_blocks.exit,
                else_args: vec![loop_blocks.acc],
            })
            .unwrap();

        // Body: compute acc + 1
        builder.switch_to_block(loop_blocks.body).unwrap();
        let one = builder.push_int("1", i32_ty(), dummy_span(), dummy_node()).unwrap();
        let new_acc =
            builder.push_binop("+", loop_blocks.acc, one, i32_ty(), dummy_span(), dummy_node()).unwrap();
        builder
            .terminate(Terminator::Branch {
                target: loop_blocks.header,
                args: vec![new_acc],
            })
            .unwrap();

        // Exit: return result
        builder.switch_to_block(loop_blocks.exit).unwrap();
        builder.terminate(Terminator::Return(loop_blocks.result)).unwrap();

        let body = builder.finish().unwrap();

        // Verify structure
        assert_eq!(body.num_blocks(), 4); // entry, header, body, exit
        assert_eq!(body.get_block(loop_blocks.header).params.len(), 1); // acc
        assert_eq!(body.get_block(loop_blocks.exit).params.len(), 1); // result

        // Verify with verifier
        let verify_result = crate::mir::ssa_verify::verify_func(&body);
        assert!(
            verify_result.is_ok(),
            "Verification failed: {:?}",
            verify_result.err()
        );
    }

    #[test]
    fn test_for_range_loop_pattern() {
        // Build: fn sum_range(n: i32) -> i32 { loop acc = 0 for i < n do acc + i }
        let mut builder = FuncBuilder::new(vec![(i32_ty(), "n".to_string())], i32_ty());

        let n = builder.get_param(0);

        // Create the for-range loop structure
        let loop_blocks = builder.create_for_range_loop(i32_ty());

        // Entry: branch to header with (0, 0)
        let zero = builder.push_int("0", i32_ty(), dummy_span(), dummy_node()).unwrap();
        let zero2 = builder.push_int("0", i32_ty(), dummy_span(), dummy_node()).unwrap();
        builder
            .terminate(Terminator::Branch {
                target: loop_blocks.header,
                args: vec![zero, zero2],
            })
            .unwrap();

        // Header: check i < n
        builder.switch_to_block(loop_blocks.header).unwrap();
        let cond =
            builder.push_binop("<", loop_blocks.index, n, bool_ty(), dummy_span(), dummy_node()).unwrap();
        builder
            .terminate(Terminator::CondBranch {
                cond,
                then_target: loop_blocks.body,
                then_args: vec![],
                else_target: loop_blocks.exit,
                else_args: vec![loop_blocks.acc],
            })
            .unwrap();

        // Body: compute acc + i, increment i
        builder.switch_to_block(loop_blocks.body).unwrap();
        let new_acc = builder
            .push_binop(
                "+",
                loop_blocks.acc,
                loop_blocks.index,
                i32_ty(),
                dummy_span(),
                dummy_node(),
            )
            .unwrap();
        let one = builder.push_int("1", i32_ty(), dummy_span(), dummy_node()).unwrap();
        let next_i =
            builder.push_binop("+", loop_blocks.index, one, i32_ty(), dummy_span(), dummy_node()).unwrap();
        builder
            .terminate(Terminator::Branch {
                target: loop_blocks.header,
                args: vec![new_acc, next_i],
            })
            .unwrap();

        // Exit: return result
        builder.switch_to_block(loop_blocks.exit).unwrap();
        builder.terminate(Terminator::Return(loop_blocks.result)).unwrap();

        let body = builder.finish().unwrap();

        // Verify structure
        assert_eq!(body.num_blocks(), 4); // entry, header, body, exit
        assert_eq!(body.get_block(loop_blocks.header).params.len(), 2); // acc, i
        assert_eq!(body.get_block(loop_blocks.exit).params.len(), 1); // result

        // Verify with verifier
        let verify_result = crate::mir::ssa_verify::verify_func(&body);
        assert!(
            verify_result.is_ok(),
            "Verification failed: {:?}",
            verify_result.err()
        );
    }

    #[test]
    fn test_nested_if_in_loop() {
        // Build a more complex example: loop with if inside
        // fn count_positive(arr_len: i32, threshold: i32) -> i32 {
        //     loop count = 0 for i < arr_len do
        //         if i > threshold then count + 1 else count
        // }
        let mut builder = FuncBuilder::new(
            vec![
                (i32_ty(), "arr_len".to_string()),
                (i32_ty(), "threshold".to_string()),
            ],
            i32_ty(),
        );

        let arr_len = builder.get_param(0);
        let threshold = builder.get_param(1);

        // Create for-range loop
        let loop_blocks = builder.create_for_range_loop(i32_ty());

        // Entry: branch to header with (0, 0)
        let zero = builder.push_int("0", i32_ty(), dummy_span(), dummy_node()).unwrap();
        let zero2 = builder.push_int("0", i32_ty(), dummy_span(), dummy_node()).unwrap();
        builder
            .terminate(Terminator::Branch {
                target: loop_blocks.header,
                args: vec![zero, zero2],
            })
            .unwrap();

        // Header: check i < arr_len
        builder.switch_to_block(loop_blocks.header).unwrap();
        let loop_cond = builder
            .push_binop(
                "<",
                loop_blocks.index,
                arr_len,
                bool_ty(),
                dummy_span(),
                dummy_node(),
            )
            .unwrap();

        // Create if-then-else for the body (we need to create it while in header to get proper block IDs)
        let if_blocks = builder.create_if_then_else(i32_ty());

        builder
            .terminate(Terminator::CondBranch {
                cond: loop_cond,
                then_target: loop_blocks.body,
                then_args: vec![],
                else_target: loop_blocks.exit,
                else_args: vec![loop_blocks.acc],
            })
            .unwrap();

        // Body: if i > threshold then count + 1 else count
        builder.switch_to_block(loop_blocks.body).unwrap();
        let if_cond = builder
            .push_binop(
                ">",
                loop_blocks.index,
                threshold,
                bool_ty(),
                dummy_span(),
                dummy_node(),
            )
            .unwrap();
        builder
            .terminate(Terminator::CondBranch {
                cond: if_cond,
                then_target: if_blocks.then_block,
                then_args: vec![],
                else_target: if_blocks.else_block,
                else_args: vec![],
            })
            .unwrap();

        // If-then: count + 1
        builder.switch_to_block(if_blocks.then_block).unwrap();
        let one = builder.push_int("1", i32_ty(), dummy_span(), dummy_node()).unwrap();
        let count_plus_one =
            builder.push_binop("+", loop_blocks.acc, one, i32_ty(), dummy_span(), dummy_node()).unwrap();
        builder
            .terminate(Terminator::Branch {
                target: if_blocks.merge_block,
                args: vec![count_plus_one],
            })
            .unwrap();

        // If-else: count (unchanged)
        builder.switch_to_block(if_blocks.else_block).unwrap();
        builder
            .terminate(Terminator::Branch {
                target: if_blocks.merge_block,
                args: vec![loop_blocks.acc],
            })
            .unwrap();

        // If-merge: branch back to loop header with new_acc and next_i
        builder.switch_to_block(if_blocks.merge_block).unwrap();
        let one2 = builder.push_int("1", i32_ty(), dummy_span(), dummy_node()).unwrap();
        let next_i =
            builder.push_binop("+", loop_blocks.index, one2, i32_ty(), dummy_span(), dummy_node()).unwrap();
        builder
            .terminate(Terminator::Branch {
                target: loop_blocks.header,
                args: vec![if_blocks.result, next_i],
            })
            .unwrap();

        // Loop exit: return result
        builder.switch_to_block(loop_blocks.exit).unwrap();
        builder.terminate(Terminator::Return(loop_blocks.result)).unwrap();

        let body = builder.finish().unwrap();

        // Verify structure: entry, header, body, if_then, if_else, if_merge, exit = 7 blocks
        assert_eq!(body.num_blocks(), 7);

        // Verify with verifier
        let verify_result = crate::mir::ssa_verify::verify_func(&body);
        assert!(
            verify_result.is_ok(),
            "Verification failed: {:?}",
            verify_result.err()
        );
    }
}
