//! Generic SSA storage and structural transformations.
//!
//! This module knows about blocks, values, instructions, dominance, cloning,
//! and substitution. It does not decide which Wyn operations are pure, safe to
//! move, profitable to inline, foldable, or in need of materialization.

pub use crate::flow::BlockId;
pub use slotmap::Key;
use slotmap::{new_key_type, SlotMap};
use std::fmt::Debug;
use std::hash::Hash;

use crate::ast::Span;
use crate::flow::{ControlHeader, Terminator as FlowTerminator};

mod control_flow;
mod dce;
mod rewrite;
mod schedule;
mod trivial_params;
mod uses;
pub(crate) use control_flow::fold_constant_selections;
pub(crate) use dce::eliminate_dead_values;
pub(crate) use rewrite::Substitutions;
pub use rewrite::VisitValues;
pub(crate) use schedule::{schedule_floating, LoopScopes};
pub(crate) use trivial_params::eliminate_single_input_params;
pub use uses::{UseSite, ValueUses};

new_key_type! {
    pub struct InstId;
    pub struct ValueId;
    pub struct PlaceId;
}

/// A compile-time scalar carried directly by an SSA operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConstantValue {
    I32(i32),
    U32(u32),
    /// IEEE-754 bits, preserving equality and hashing for every payload.
    F32(u32),
    Bool(bool),
}

impl ConstantValue {
    pub fn from_f32(value: f32) -> Self {
        Self::F32(value.to_bits())
    }
}

/// An SSA instruction result or an inline scalar constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValueRef {
    Ssa(ValueId),
    Const(ConstantValue),
}

impl ValueRef {
    pub fn as_ssa(self) -> Option<ValueId> {
        match self {
            Self::Ssa(id) => Some(id),
            Self::Const(_) => None,
        }
    }

    pub fn as_const(self) -> Option<ConstantValue> {
        match self {
            Self::Const(value) => Some(value),
            Self::Ssa(_) => None,
        }
    }

    pub fn map_ssa(self, map: impl Fn(ValueId) -> ValueId) -> Self {
        match self {
            Self::Ssa(id) => Self::Ssa(map(id)),
            Self::Const(value) => Self::Const(value),
        }
    }
}

impl From<ValueId> for ValueRef {
    fn from(id: ValueId) -> Self {
        Self::Ssa(id)
    }
}

#[derive(Clone, Debug)]
pub struct Function<I, T> {
    pub entry: BlockId,
    /// Function parameters in signature order.
    pub params: Vec<ValueId>,
    pub blocks: SlotMap<BlockId, BasicBlock>,
    pub insts: SlotMap<InstId, InstNode<I>>,
    pub values: SlotMap<ValueId, ValueInfo<T>>,
}

#[derive(Clone, Debug)]
pub struct BasicBlock {
    pub params: Vec<ValueId>,
    pub insts: Vec<InstId>,
    pub term: Terminator,
    /// Structured-control metadata intrinsically owned by this block.
    pub control_header: Option<ControlHeader>,
}

#[derive(Clone, Debug)]
pub struct InstNode<I> {
    pub data: I,
    pub result: Option<ValueId>,
    pub placement: InstPlacement,
    /// Source span of the user expression that produced this instruction,
    /// or `None` for synthesized instructions (block-param phis, builder
    /// scratch). Used by backends to blame errors back to source.
    pub span: Option<Span>,
}

/// Whether an instruction has been assigned to a control-flow block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InstPlacement {
    Floating,
    Block(BlockId),
}

impl InstPlacement {
    pub fn block(self) -> Option<BlockId> {
        match self {
            Self::Floating => None,
            Self::Block(block) => Some(block),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ValueInfo<T> {
    pub def: ValueDef,
    pub ty: T,
    /// Source/debug name for function and named block parameters.
    pub name: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueDef {
    /// Block parameter (appears in block's param list, lowered to OpPhi in SPIR-V).
    Param {
        block: BlockId,
        index: usize,
    },
    /// Function parameter (NOT a block param, lowered to OpFunctionParameter in SPIR-V).
    FunctionParam {
        index: usize,
    },
    /// Produced by an instruction.
    Inst {
        inst: InstId,
    },
}

pub type Terminator = FlowTerminator<ValueRef, ValueRef, ValueRef>;

impl<I, T> Function<I, T> {
    pub fn new() -> Self {
        let mut blocks = SlotMap::with_key();
        let entry = blocks.insert(BasicBlock {
            params: Vec::new(),
            insts: Vec::new(),
            term: Terminator::Unreachable,
            control_header: None,
        });
        Self {
            entry,
            params: Vec::new(),
            blocks,
            insts: SlotMap::with_key(),
            values: SlotMap::with_key(),
        }
    }

    pub fn create_block(&mut self) -> BlockId {
        self.blocks.insert(BasicBlock {
            params: Vec::new(),
            insts: Vec::new(),
            term: Terminator::Unreachable,
            control_header: None,
        })
    }

    pub fn add_function_param(&mut self, ty: T, name: String) -> ValueId {
        let index = self.params.len();
        let value = self.values.insert(ValueInfo {
            def: ValueDef::FunctionParam { index },
            ty,
            name: Some(name),
        });
        self.params.push(value);
        value
    }

    pub fn add_block_param(&mut self, block: BlockId, ty: T) -> ValueId {
        self.add_named_block_param(block, ty, None)
    }

    pub fn add_named_block_param(&mut self, block: BlockId, ty: T, name: Option<String>) -> ValueId {
        let index = self.blocks[block].params.len();
        let value = self.values.insert(ValueInfo {
            def: ValueDef::Param { block, index },
            ty,
            name,
        });
        self.blocks[block].params.push(value);
        value
    }

    pub fn value_type(&self, v: ValueId) -> &T {
        &self.values[v].ty
    }

    pub fn block_of_value(&self, v: ValueId) -> Option<BlockId> {
        match self.values[v].def {
            ValueDef::Param { block, .. } => Some(block),
            ValueDef::FunctionParam { .. } => Some(self.entry),
            ValueDef::Inst { inst } => self.insts[inst].placement.block(),
        }
    }

    pub fn inst_of_value(&self, v: ValueId) -> Option<InstId> {
        match self.values[v].def {
            ValueDef::Param { .. } | ValueDef::FunctionParam { .. } => None,
            ValueDef::Inst { inst } => Some(inst),
        }
    }

    pub fn append_inst(&mut self, block: BlockId, data: I, ty: T) -> ValueId {
        self.append_inst_with_span(block, data, ty, None)
    }

    pub fn append_inst_with_span(&mut self, block: BlockId, data: I, ty: T, span: Option<Span>) -> ValueId {
        let (inst, value) = self.insert_value_inst(data, ty, InstPlacement::Block(block), span);
        self.blocks[block].insts.push(inst);
        value
    }

    /// Create a value-producing instruction without assigning it to a block.
    pub(crate) fn append_floating_inst(&mut self, data: I, ty: T) -> ValueId {
        let (_, value) = self.insert_value_inst(data, ty, InstPlacement::Floating, None);
        value
    }

    fn insert_value_inst(
        &mut self,
        data: I,
        ty: T,
        placement: InstPlacement,
        span: Option<Span>,
    ) -> (InstId, ValueId) {
        let values = &mut self.values;
        let inst = self.insts.insert_with_key(|inst| {
            let value = values.insert(ValueInfo {
                def: ValueDef::Inst { inst },
                ty,
                name: None,
            });
            InstNode {
                data,
                result: Some(value),
                placement,
                span,
            }
        });
        let Some(value) = self.insts[inst].result else {
            unreachable!("value-producing instruction {inst:?} has no result")
        };
        (inst, value)
    }

    pub fn append_void_inst(&mut self, block: BlockId, data: I) -> InstId {
        self.append_void_inst_with_span(block, data, None)
    }

    pub fn append_void_inst_with_span(&mut self, block: BlockId, data: I, span: Option<Span>) -> InstId {
        let inst = self.insts.insert(InstNode {
            data,
            result: None,
            placement: InstPlacement::Block(block),
            span,
        });
        self.blocks[block].insts.push(inst);
        inst
    }
}

// =============================================================================
// FuncBuilder
// =============================================================================

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BuilderError {
    NoCurrentBlock,
    BlockAlreadyTerminated(BlockId),
    UnterminatedBlock(BlockId),
}

impl std::fmt::Display for BuilderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuilderError::NoCurrentBlock => write!(f, "No current block selected"),
            BuilderError::BlockAlreadyTerminated(id) => {
                write!(f, "Block {:?} already terminated", id)
            }
            BuilderError::UnterminatedBlock(id) => write!(f, "Block {:?} has no terminator", id),
        }
    }
}

impl std::error::Error for BuilderError {}

#[derive(Clone, Debug)]
pub struct FuncBuilder<I, T> {
    func: Function<I, T>,
    current_block: Option<BlockId>,
}

impl<I, T: Clone + Debug> FuncBuilder<I, T> {
    pub fn new() -> Self {
        let func = Function::new();
        let entry = func.entry;
        Self {
            func,
            current_block: Some(entry),
        }
    }

    pub fn entry(&self) -> BlockId {
        self.func.entry
    }

    pub fn current_block(&self) -> Option<BlockId> {
        self.current_block
    }

    pub fn func(&self) -> &Function<I, T> {
        &self.func
    }

    pub fn func_mut(&mut self) -> &mut Function<I, T> {
        &mut self.func
    }

    pub fn create_block(&mut self) -> BlockId {
        self.func.create_block()
    }

    pub fn create_block_with_params(&mut self, types: Vec<T>) -> (BlockId, Vec<ValueId>) {
        let block = self.func.create_block();
        let params: Vec<ValueId> =
            types.into_iter().map(|ty| self.func.add_block_param(block, ty)).collect();
        (block, params)
    }

    pub fn add_block_param(&mut self, block: BlockId, ty: T) -> ValueId {
        self.func.add_block_param(block, ty)
    }

    pub fn switch_to_block(&mut self, block: BlockId) -> Result<(), BuilderError> {
        if let Some(current) = self.current_block {
            if matches!(self.func.blocks[current].term, Terminator::Unreachable) {
                return Err(BuilderError::UnterminatedBlock(current));
            }
        }
        self.current_block = Some(block);
        Ok(())
    }

    pub fn switch_to_block_unchecked(&mut self, block: BlockId) {
        self.current_block = Some(block);
    }

    pub fn push_inst(&mut self, data: I, ty: T) -> Result<ValueId, BuilderError> {
        self.push_inst_with_span(data, ty, None)
    }

    pub fn push_inst_with_span(
        &mut self,
        data: I,
        ty: T,
        span: Option<Span>,
    ) -> Result<ValueId, BuilderError> {
        let block = self.current_block.ok_or(BuilderError::NoCurrentBlock)?;
        if !matches!(self.func.blocks[block].term, Terminator::Unreachable) {
            return Err(BuilderError::BlockAlreadyTerminated(block));
        }
        Ok(self.func.append_inst_with_span(block, data, ty, span))
    }

    pub fn push_void_inst(&mut self, data: I) -> Result<InstId, BuilderError> {
        self.push_void_inst_with_span(data, None)
    }

    pub fn push_void_inst_with_span(
        &mut self,
        data: I,
        span: Option<Span>,
    ) -> Result<InstId, BuilderError> {
        let block = self.current_block.ok_or(BuilderError::NoCurrentBlock)?;
        if !matches!(self.func.blocks[block].term, Terminator::Unreachable) {
            return Err(BuilderError::BlockAlreadyTerminated(block));
        }
        Ok(self.func.append_void_inst_with_span(block, data, span))
    }

    pub fn terminate(&mut self, term: Terminator) -> Result<(), BuilderError> {
        let block = self.current_block.ok_or(BuilderError::NoCurrentBlock)?;
        if !matches!(self.func.blocks[block].term, Terminator::Unreachable) {
            return Err(BuilderError::BlockAlreadyTerminated(block));
        }
        self.func.blocks[block].term = term;
        Ok(())
    }

    pub fn finish(self) -> Result<Function<I, T>, BuilderError> {
        for (bid, block) in &self.func.blocks {
            if matches!(block.term, Terminator::Unreachable) && bid != self.func.entry {
                return Err(BuilderError::UnterminatedBlock(bid));
            }
        }
        Ok(self.func)
    }

    /// Finish without checking termination.
    pub fn finish_unchecked(self) -> Function<I, T> {
        self.func
    }
}
