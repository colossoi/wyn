pub use crate::flow::BlockId;
use crate::LookupMap;
pub use slotmap::Key;
use slotmap::{new_key_type, SlotMap};
use smallvec::SmallVec;
use std::fmt::Debug;
use std::hash::Hash;

use crate::ast::Span;

new_key_type! {
    pub struct InstId;
    pub struct ValueId;
    pub struct PlaceId;
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
    pub control_header: Option<crate::flow::ControlHeader>,
}

#[derive(Clone, Debug)]
pub struct InstNode<I> {
    pub data: I,
    pub result: Option<ValueId>,
    pub parent: BlockId,
    /// Source span of the user expression that produced this instruction,
    /// or `None` for synthesized instructions (block-param phis, builder
    /// scratch). Used by backends to blame errors back to source.
    pub span: Option<Span>,
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

pub type Terminator = crate::flow::Terminator<ValueId>;

pub type Successors = SmallVec<[BlockId; 2]>;

impl<I, T: Clone + Debug> Function<I, T> {
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

    pub fn value_def(&self, v: ValueId) -> ValueDef {
        self.values[v].def
    }

    pub fn value_type(&self, v: ValueId) -> &T {
        &self.values[v].ty
    }

    pub fn block_of_value(&self, v: ValueId) -> BlockId {
        match self.values[v].def {
            ValueDef::Param { block, .. } => block,
            ValueDef::FunctionParam { .. } => self.entry,
            ValueDef::Inst { inst } => self.insts[inst].parent,
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
        let value = self.values.insert(ValueInfo {
            def: ValueDef::Param { block, index: 0 },
            ty,
            name: None,
        });
        let inst = self.insts.insert(InstNode {
            data,
            result: Some(value),
            parent: block,
            span,
        });
        self.values[value].def = ValueDef::Inst { inst };
        self.blocks[block].insts.push(inst);
        value
    }

    pub fn append_void_inst(&mut self, block: BlockId, data: I) -> InstId {
        self.append_void_inst_with_span(block, data, None)
    }

    pub fn append_void_inst_with_span(&mut self, block: BlockId, data: I, span: Option<Span>) -> InstId {
        let inst = self.insts.insert(InstNode {
            data,
            result: None,
            parent: block,
            span,
        });
        self.blocks[block].insts.push(inst);
        inst
    }

    pub fn insert_inst_at_index(&mut self, block: BlockId, index: usize, data: I, ty: T) -> ValueId {
        self.insert_inst_at_index_with_span(block, index, data, ty, None)
    }

    pub fn insert_inst_at_index_with_span(
        &mut self,
        block: BlockId,
        index: usize,
        data: I,
        ty: T,
        span: Option<Span>,
    ) -> ValueId {
        let value = self.values.insert(ValueInfo {
            def: ValueDef::Param { block, index: 0 },
            ty,
            name: None,
        });
        let inst = self.insts.insert(InstNode {
            data,
            result: Some(value),
            parent: block,
            span,
        });
        self.values[value].def = ValueDef::Inst { inst };
        self.blocks[block].insts.insert(index, inst);
        value
    }

    pub fn predecessors(&self) -> LookupMap<BlockId, Vec<BlockId>> {
        let mut preds: LookupMap<BlockId, Vec<BlockId>> =
            self.blocks.keys().map(|b| (b, Vec::new())).collect();

        for (bid, block) in &self.blocks {
            for succ in block.term.successors() {
                preds.entry(succ).or_default().push(bid);
            }
        }
        preds
    }

    pub fn block_order_index_map(&self) -> LookupMap<InstId, usize> {
        let mut out = LookupMap::new();
        for (_bid, block) in &self.blocks {
            for (idx, &inst) in block.insts.iter().enumerate() {
                out.insert(inst, idx);
            }
        }
        out
    }

    pub fn remove_inst(&mut self, inst: InstId) -> bool {
        let Some(node) = self.insts.remove(inst) else {
            return false;
        };
        let block = node.parent;
        if let Some(pos) = self.blocks[block].insts.iter().position(|&x| x == inst) {
            self.blocks[block].insts.remove(pos);
        }
        if let Some(result) = node.result {
            self.values.remove(result);
        }
        true
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
