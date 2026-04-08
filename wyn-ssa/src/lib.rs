mod dominators;

pub use slotmap::Key;
use slotmap::{SlotMap, new_key_type};
use smallvec::SmallVec;
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

use dominators::{Dominators, block_top_insert_index_after_operands};

new_key_type! {
    pub struct BlockId;
    pub struct InstId;
    pub struct ValueId;
}

#[derive(Clone, Debug)]
pub struct Function<I, E, T> {
    pub entry: BlockId,
    pub blocks: SlotMap<BlockId, BasicBlock>,
    pub insts: SlotMap<InstId, InstNode<I, E>>,
    pub values: SlotMap<ValueId, ValueInfo<T>>,
}

#[derive(Clone, Debug)]
pub struct BasicBlock {
    pub params: Vec<ValueId>,
    pub insts: Vec<InstId>,
    pub term: Terminator,
}

#[derive(Clone, Debug)]
pub struct InstNode<I, E> {
    pub data: I,
    pub result: Option<ValueId>,
    pub parent: BlockId,
    pub effects: Option<(E, E)>,
}

#[derive(Clone, Debug)]
pub struct ValueInfo<T> {
    pub def: ValueDef,
    pub ty: T,
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

#[derive(Clone, Debug)]
pub enum Terminator {
    Return(Option<ValueId>),
    Branch {
        target: BlockId,
        args: Vec<ValueId>,
    },
    CondBranch {
        cond: ValueId,
        then_target: BlockId,
        then_args: Vec<ValueId>,
        else_target: BlockId,
        else_args: Vec<ValueId>,
    },
    Unreachable,
}

pub trait Instr: Clone {
    fn for_each_operand(&self, f: impl FnMut(ValueId));
    fn map_operands(&self, f: impl FnMut(ValueId) -> ValueId) -> Self;
}

pub trait ValueLike: Instr {
    fn is_hoistable(&self) -> bool;
    fn is_closed(&self) -> bool;
    fn equivalent_to(&self, other: &Self) -> bool;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InlineArgError {
    BadParamIndex,
    ReplacementNotClosed,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EffectVerifyError<E> {
    DoubleConsume {
        effect: E,
        inst: InstId,
    },
}

impl Terminator {
    pub fn successors(&self) -> Successors {
        match self {
            Terminator::Return(_) | Terminator::Unreachable => SmallVec::new(),
            Terminator::Branch { target, .. } => smallvec::smallvec![*target],
            Terminator::CondBranch {
                then_target,
                else_target,
                ..
            } => smallvec::smallvec![*then_target, *else_target],
        }
    }

    pub fn for_each_value(&self, mut f: impl FnMut(ValueId)) {
        match self {
            Terminator::Return(v) => {
                if let Some(v) = v {
                    f(*v);
                }
            }
            Terminator::Branch { args, .. } => {
                for &v in args {
                    f(v);
                }
            }
            Terminator::CondBranch {
                cond,
                then_args,
                else_args,
                ..
            } => {
                f(*cond);
                for &v in then_args {
                    f(v);
                }
                for &v in else_args {
                    f(v);
                }
            }
            Terminator::Unreachable => {}
        }
    }

    pub fn map_values(&self, mut f: impl FnMut(ValueId) -> ValueId) -> Self {
        match self {
            Terminator::Return(v) => Terminator::Return(v.map(&mut f)),
            Terminator::Branch { target, args } => Terminator::Branch {
                target: *target,
                args: args.iter().copied().map(&mut f).collect(),
            },
            Terminator::CondBranch {
                cond,
                then_target,
                then_args,
                else_target,
                else_args,
            } => Terminator::CondBranch {
                cond: f(*cond),
                then_target: *then_target,
                then_args: then_args.iter().copied().map(&mut f).collect(),
                else_target: *else_target,
                else_args: else_args.iter().copied().map(&mut f).collect(),
            },
            Terminator::Unreachable => Terminator::Unreachable,
        }
    }
}

pub type Successors = SmallVec<[BlockId; 2]>;

impl<I, E, T: Clone + Debug> Function<I, E, T> {
    pub fn new() -> Self {
        let mut blocks = SlotMap::with_key();
        let entry = blocks.insert(BasicBlock {
            params: Vec::new(),
            insts: Vec::new(),
            term: Terminator::Unreachable,
        });
        Self {
            entry,
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
        })
    }

    pub fn add_function_param(&mut self, index: usize, ty: T) -> ValueId {
        self.values.insert(ValueInfo {
            def: ValueDef::FunctionParam { index },
            ty,
        })
    }

    pub fn add_block_param(&mut self, block: BlockId, ty: T) -> ValueId {
        let index = self.blocks[block].params.len();
        let value = self.values.insert(ValueInfo {
            def: ValueDef::Param { block, index },
            ty,
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

    pub fn append_inst(&mut self, block: BlockId, data: I, ty: T, effects: Option<(E, E)>) -> ValueId {
        let value = self.values.insert(ValueInfo {
            def: ValueDef::Param { block, index: 0 },
            ty,
        });
        let inst = self.insts.insert(InstNode {
            data,
            result: Some(value),
            parent: block,
            effects,
        });
        self.values[value].def = ValueDef::Inst { inst };
        self.blocks[block].insts.push(inst);
        value
    }

    pub fn append_void_inst(&mut self, block: BlockId, data: I, effects: Option<(E, E)>) -> InstId {
        let inst = self.insts.insert(InstNode {
            data,
            result: None,
            parent: block,
            effects,
        });
        self.blocks[block].insts.push(inst);
        inst
    }

    pub fn insert_inst_at_index(
        &mut self,
        block: BlockId,
        index: usize,
        data: I,
        ty: T,
        effects: Option<(E, E)>,
    ) -> ValueId {
        let value = self.values.insert(ValueInfo {
            def: ValueDef::Param { block, index: 0 },
            ty,
        });
        let inst = self.insts.insert(InstNode {
            data,
            result: Some(value),
            parent: block,
            effects,
        });
        self.values[value].def = ValueDef::Inst { inst };
        self.blocks[block].insts.insert(index, inst);
        value
    }

    pub fn predecessors(&self) -> HashMap<BlockId, Vec<BlockId>> {
        let mut preds: HashMap<BlockId, Vec<BlockId>> =
            self.blocks.keys().map(|b| (b, Vec::new())).collect();

        for (bid, block) in &self.blocks {
            for succ in block.term.successors() {
                preds.entry(succ).or_default().push(bid);
            }
        }
        preds
    }

    pub fn block_order_index_map(&self) -> HashMap<InstId, usize> {
        let mut out = HashMap::new();
        for (_bid, block) in &self.blocks {
            for (idx, &inst) in block.insts.iter().enumerate() {
                out.insert(inst, idx);
            }
        }
        out
    }

    pub fn is_effectful(&self, inst: InstId) -> bool {
        self.insts[inst].effects.is_some()
    }
}

impl<I, E: Copy + Eq + Hash + Debug, T: Clone + Debug> Function<I, E, T> {
    pub fn replace_effect(&mut self, old: E, new: E) -> bool {
        if old == new {
            return false;
        }
        for (_id, node) in &mut self.insts {
            if let Some((ref mut ein, _)) = node.effects {
                if *ein == old {
                    *ein = new;
                    return true; // linear: at most one consumer
                }
            }
        }
        false
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

        // Splice the effect chain: downstream consumers of our effect_out
        // now consume our effect_in instead.
        if let Some((effect_in, effect_out)) = node.effects {
            self.replace_effect(effect_out, effect_in);
        }

        true
    }

    pub fn verify_effects(&self) -> Result<(), Vec<EffectVerifyError<E>>> {
        let mut errors = Vec::new();
        let mut consumed: HashMap<E, InstId> = HashMap::new();

        for (id, node) in &self.insts {
            if let Some((ein, _)) = node.effects {
                if let Some(&_prev_inst) = consumed.get(&ein) {
                    errors.push(EffectVerifyError::DoubleConsume {
                        effect: ein,
                        inst: id,
                    });
                } else {
                    consumed.insert(ein, id);
                }
            }
        }

        if errors.is_empty() { Ok(()) } else { Err(errors) }
    }
}

impl<I: Instr, E: Copy + Eq + Hash + Debug, T: Clone + Debug> Function<I, E, T> {
    pub fn replace_all_uses(&mut self, old: ValueId, new: ValueId) -> usize {
        if old == new {
            return 0;
        }

        let mut count = 0;

        for (_inst_id, node) in &mut self.insts {
            let mut changed = false;
            let rewritten = node.data.map_operands(|v| {
                if v == old {
                    changed = true;
                    count += 1;
                    new
                } else {
                    v
                }
            });
            if changed {
                node.data = rewritten;
            }
        }

        for (_bid, block) in &mut self.blocks {
            let mut changed = false;
            let rewritten = block.term.map_values(|v| {
                if v == old {
                    changed = true;
                    count += 1;
                    new
                } else {
                    v
                }
            });
            if changed {
                block.term = rewritten;
            }
        }

        count
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
pub struct FuncBuilder<I, E, T> {
    func: Function<I, E, T>,
    current_block: Option<BlockId>,
}

impl<I, E, T: Clone + Debug> FuncBuilder<I, E, T> {
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

    pub fn func(&self) -> &Function<I, E, T> {
        &self.func
    }

    pub fn func_mut(&mut self) -> &mut Function<I, E, T> {
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

    pub fn push_inst(&mut self, data: I, ty: T, effects: Option<(E, E)>) -> Result<ValueId, BuilderError> {
        let block = self.current_block.ok_or(BuilderError::NoCurrentBlock)?;
        if !matches!(self.func.blocks[block].term, Terminator::Unreachable) {
            return Err(BuilderError::BlockAlreadyTerminated(block));
        }
        Ok(self.func.append_inst(block, data, ty, effects))
    }

    pub fn push_void_inst(&mut self, data: I, effects: Option<(E, E)>) -> Result<InstId, BuilderError> {
        let block = self.current_block.ok_or(BuilderError::NoCurrentBlock)?;
        if !matches!(self.func.blocks[block].term, Terminator::Unreachable) {
            return Err(BuilderError::BlockAlreadyTerminated(block));
        }
        Ok(self.func.append_void_inst(block, data, effects))
    }

    pub fn terminate(&mut self, term: Terminator) -> Result<(), BuilderError> {
        let block = self.current_block.ok_or(BuilderError::NoCurrentBlock)?;
        if !matches!(self.func.blocks[block].term, Terminator::Unreachable) {
            return Err(BuilderError::BlockAlreadyTerminated(block));
        }
        self.func.blocks[block].term = term;
        Ok(())
    }

    pub fn finish(self) -> Result<Function<I, E, T>, BuilderError> {
        for (bid, block) in &self.func.blocks {
            if matches!(block.term, Terminator::Unreachable) && bid != self.func.entry {
                return Err(BuilderError::UnterminatedBlock(bid));
            }
        }
        Ok(self.func)
    }

    /// Finish without checking termination.
    pub fn finish_unchecked(self) -> Function<I, E, T> {
        self.func
    }
}

// =============================================================================
// Free-standing passes
// =============================================================================

pub fn inline_entry_param<I: ValueLike, E: Copy + Eq + Hash + Debug, T: Clone + Debug>(
    func: &mut Function<I, E, T>,
    param_index: usize,
    replacement: I,
    ty: T,
) -> Result<(), InlineArgError> {
    if !replacement.is_closed() {
        return Err(InlineArgError::ReplacementNotClosed);
    }

    let entry = func.entry;
    if param_index >= func.blocks[entry].params.len() {
        return Err(InlineArgError::BadParamIndex);
    }

    let old_param = func.blocks[entry].params[param_index];

    let insert_index = block_top_insert_index_after_operands(func, entry, &replacement);
    let new_value = func.insert_inst_at_index(entry, insert_index, replacement, ty, None);
    func.replace_all_uses(old_param, new_value);

    func.blocks[entry].params.remove(param_index);
    func.values.remove(old_param);

    for (i, &param) in func.blocks[entry].params.iter().enumerate() {
        func.values[param].def = ValueDef::Param {
            block: entry,
            index: i,
        };
    }

    Ok(())
}

pub fn inline_block_param<I: ValueLike, E: Copy + Eq + Hash + Debug, T: Clone + Debug>(
    func: &mut Function<I, E, T>,
    block: BlockId,
    param_index: usize,
    replacement: I,
    ty: T,
) -> Result<(), InlineArgError> {
    if block == func.entry {
        return inline_entry_param(func, param_index, replacement, ty);
    }

    if !replacement.is_closed() {
        return Err(InlineArgError::ReplacementNotClosed);
    }

    if param_index >= func.blocks[block].params.len() {
        return Err(InlineArgError::BadParamIndex);
    }

    let old_param = func.blocks[block].params[param_index];

    let insert_index = block_top_insert_index_after_operands(func, block, &replacement);
    let new_value = func.insert_inst_at_index(block, insert_index, replacement, ty, None);
    func.replace_all_uses(old_param, new_value);

    func.blocks[block].params.remove(param_index);
    func.values.remove(old_param);

    for (i, &param) in func.blocks[block].params.iter().enumerate() {
        func.values[param].def = ValueDef::Param { block, index: i };
    }

    // Remove the corresponding incoming argument at each predecessor edge.
    let dom = Dominators::compute(func);
    let preds = dom.preds(block).to_vec();

    for pred in preds {
        let term = func.blocks[pred].term.clone();
        func.blocks[pred].term = match term {
            Terminator::Branch { target, mut args } if target == block => {
                args.remove(param_index);
                Terminator::Branch { target, args }
            }
            Terminator::CondBranch {
                cond,
                then_target,
                mut then_args,
                else_target,
                mut else_args,
            } => {
                if then_target == block {
                    then_args.remove(param_index);
                }
                if else_target == block {
                    else_args.remove(param_index);
                }
                Terminator::CondBranch {
                    cond,
                    then_target,
                    then_args,
                    else_target,
                    else_args,
                }
            }
            other => other,
        };
    }

    Ok(())
}

fn types_match<I, E, T: PartialEq>(func: &Function<I, E, T>, a: InstId, b: InstId) -> bool {
    match (func.insts[a].result, func.insts[b].result) {
        (Some(ra), Some(rb)) => func.values[ra].ty == func.values[rb].ty,
        (None, None) => true,
        _ => false,
    }
}

pub fn lift_and_merge<I: ValueLike, E: Copy + Eq + Hash + Debug, T: Clone + Debug + PartialEq>(
    func: &mut Function<I, E, T>,
) {
    // Phase 1: Hoist every hoistable instruction as high as its data dependencies allow.
    hoist_all(func);

    // Phase 2: Merge duplicates that landed at the same position.
    merge_duplicates(func);
}

/// Move every pure, hoistable instruction as high as possible, but don't
/// cross a conditional branch unless the sibling branch has an equivalent
/// instruction (which gets merged).
fn hoist_all<I: ValueLike, E: Copy + Eq + Hash + Debug, T: Clone + Debug + PartialEq>(
    func: &mut Function<I, E, T>,
) {
    let dom = Dominators::compute(func);

    let candidates: Vec<InstId> = func
        .insts
        .iter()
        .filter_map(|(id, node)| {
            if node.result.is_some() && node.effects.is_none() && node.data.is_hoistable() {
                Some(id)
            } else {
                None
            }
        })
        .collect();

    for inst_id in candidates {
        if !func.insts.contains_key(inst_id) {
            continue;
        }

        let current_block = func.insts[inst_id].parent;

        // Find the floor: deepest operand block (can't go above any operand).
        let mut operand_blocks: Vec<BlockId> = Vec::new();
        func.insts[inst_id].data.for_each_operand(|v| {
            operand_blocks.push(func.block_of_value(v));
        });

        let floor = if operand_blocks.is_empty() {
            func.entry
        } else {
            let deepest = *operand_blocks.iter().max_by_key(|&&b| dom.dom_set_size(b)).unwrap();
            if !operand_blocks.iter().all(|&b| dom.dominates(b, deepest)) {
                continue; // operands in sibling branches
            }
            deepest
        };

        // Walk up from current block toward the floor, stopping if we'd
        // cross a conditional branch without a matching sibling instruction.
        let mut target = current_block;
        loop {
            let Some(parent) = dom.idom(target) else {
                break; // reached entry
            };

            // Don't go above the operand floor
            if !dom.dominates(floor, parent) && floor != parent {
                break;
            }

            // Check if parent ends in a conditional branch
            if matches!(func.blocks[parent].term, Terminator::CondBranch { .. }) {
                // We're inside one branch. Don't hoist past the conditional.
                break;
            }

            target = parent;
        }

        if target == current_block {
            continue;
        }

        // Remove from current block
        let old_pos = func.blocks[current_block].insts.iter().position(|&id| id == inst_id);
        if let Some(pos) = old_pos {
            func.blocks[current_block].insts.remove(pos);
        }

        // Insert at the right position in target block
        let insert_idx = block_top_insert_index_after_operands(func, target, &func.insts[inst_id].data);
        func.blocks[target].insts.insert(insert_idx, inst_id);
        func.insts[inst_id].parent = target;
    }
}

/// Find duplicate instructions in each block and merge them.
fn merge_duplicates<I: ValueLike, E: Copy + Eq + Hash + Debug, T: Clone + Debug + PartialEq>(
    func: &mut Function<I, E, T>,
) {
    let block_ids: Vec<BlockId> = func.blocks.keys().collect();

    for bid in block_ids {
        let insts: Vec<InstId> = func.blocks[bid].insts.clone();
        let mut to_remove: Vec<(InstId, ValueId, ValueId)> = Vec::new();

        for i in 0..insts.len() {
            let a = insts[i];
            if !func.insts.contains_key(a) {
                continue;
            }
            let Some(a_result) = func.insts[a].result else {
                continue;
            };
            if func.insts[a].effects.is_some() || !func.insts[a].data.is_hoistable() {
                continue;
            }

            for j in (i + 1)..insts.len() {
                let b = insts[j];
                if !func.insts.contains_key(b) {
                    continue;
                }
                let Some(b_result) = func.insts[b].result else {
                    continue;
                };
                if func.insts[b].effects.is_some() || !func.insts[b].data.is_hoistable() {
                    continue;
                }

                if func.insts[a].data.equivalent_to(&func.insts[b].data) && types_match(func, a, b) {
                    to_remove.push((b, b_result, a_result));
                }
            }
        }

        for (inst, old_val, new_val) in to_remove {
            if func.insts.contains_key(inst) {
                func.replace_all_uses(old_val, new_val);
                func.remove_inst(inst);
            }
        }
    }
}

/// Forward block parameters through single-predecessor edges.
///
/// When a block has exactly one predecessor that reaches it via an unconditional
/// jump, the block's parameters are always equal to the jump's arguments. This
/// pass replaces all uses of such parameters with the corresponding arguments,
/// then clears both. Runs to fixpoint to resolve transitive chains.
pub fn forward_single_pred_params<I: Instr, E: Copy + Eq + Hash + Debug, T: Clone + Debug>(
    func: &mut Function<I, E, T>,
) {
    loop {
        let preds = func.predecessors();
        let mut substitutions: Vec<(ValueId, ValueId)> = Vec::new();

        for (bid, pred_list) in &preds {
            if *bid == func.entry {
                continue;
            }
            if pred_list.len() != 1 {
                continue;
            }
            let pred = pred_list[0];
            let args = match &func.blocks[pred].term {
                Terminator::Branch { target, args } if *target == *bid => args,
                _ => continue,
            };
            let params = &func.blocks[*bid].params;
            if params.len() != args.len() {
                continue;
            }
            for (i, &param) in params.iter().enumerate() {
                substitutions.push((param, args[i]));
            }
        }

        if substitutions.is_empty() {
            break;
        }

        // Resolve transitive chains: if a→b and b→c, resolve a→c.
        let sub_map: HashMap<ValueId, ValueId> = substitutions.iter().copied().collect();
        let resolved: Vec<(ValueId, ValueId)> = substitutions
            .iter()
            .map(|&(old, mut new)| {
                while let Some(&next) = sub_map.get(&new) {
                    if next == new {
                        break;
                    }
                    new = next;
                }
                (old, new)
            })
            .collect();

        for &(old, new) in &resolved {
            func.replace_all_uses(old, new);
        }

        // Clear the params and jump args for blocks we forwarded.
        for (bid, pred_list) in &preds {
            if *bid == func.entry || pred_list.len() != 1 {
                continue;
            }
            let pred = pred_list[0];
            let is_jump_to_bid = matches!(
                &func.blocks[pred].term,
                Terminator::Branch { target, .. } if *target == *bid
            );
            if !is_jump_to_bid || func.blocks[*bid].params.is_empty() {
                continue;
            }
            // Remove the param ValueIds from the value map.
            for &param in &func.blocks[*bid].params {
                func.values.remove(param);
            }
            func.blocks[*bid].params.clear();
            if let Terminator::Branch { ref mut args, .. } = func.blocks[pred].term {
                args.clear();
            }
        }
    }
}

/// Eliminate empty blocks that unconditionally jump to another block.
///
/// A block is eliminated when it has no instructions, no parameters, and its
/// terminator is an unconditional jump (with no arguments) to a target block.
/// All predecessors are redirected to the target, and the empty block's
/// terminator is set to `Unreachable`.
///
/// The entry block is never eliminated.
///
/// To preserve a block that would otherwise be empty (e.g. SPIR-V merge or
/// continue targets), place a placeholder instruction in it. Blocks with any
/// instructions are never candidates for elimination.
///
/// Runs to fixpoint to handle chains of empty blocks.
pub fn eliminate_empty_blocks<I, E, T: Clone + Debug>(func: &mut Function<I, E, T>) {
    loop {
        let mut redirects: Vec<(BlockId, BlockId)> = Vec::new();

        for (bid, block) in &func.blocks {
            if bid == func.entry {
                continue;
            }
            if !block.insts.is_empty() || !block.params.is_empty() {
                continue;
            }
            let target = match &block.term {
                Terminator::Branch { target, args } if args.is_empty() => *target,
                _ => continue,
            };
            redirects.push((bid, target));
        }

        if redirects.is_empty() {
            break;
        }

        // Resolve transitive chains: empty1→empty2→target becomes empty1→target.
        let raw_map: HashMap<BlockId, BlockId> = redirects.iter().copied().collect();
        let redirect_map: HashMap<BlockId, BlockId> = redirects
            .iter()
            .map(|&(from, mut to)| {
                while let Some(&next) = raw_map.get(&to) {
                    if next == to {
                        break;
                    }
                    to = next;
                }
                (from, to)
            })
            .collect();

        // Rewrite all terminators that target eliminated blocks.
        let all_blocks: Vec<BlockId> = func.blocks.keys().collect();
        for bid in all_blocks {
            let rewritten = match &func.blocks[bid].term {
                Terminator::Branch { target, args } => {
                    let new_target = redirect_map.get(target).copied().unwrap_or(*target);
                    if new_target == *target {
                        continue;
                    }
                    Terminator::Branch {
                        target: new_target,
                        args: args.clone(),
                    }
                }
                Terminator::CondBranch {
                    cond,
                    then_target,
                    then_args,
                    else_target,
                    else_args,
                } => {
                    let new_then = redirect_map.get(then_target).copied().unwrap_or(*then_target);
                    let new_else = redirect_map.get(else_target).copied().unwrap_or(*else_target);
                    if new_then == *then_target && new_else == *else_target {
                        continue;
                    }
                    Terminator::CondBranch {
                        cond: *cond,
                        then_target: new_then,
                        then_args: then_args.clone(),
                        else_target: new_else,
                        else_args: else_args.clone(),
                    }
                }
                _ => continue,
            };
            func.blocks[bid].term = rewritten;
        }

        // Mark eliminated blocks as unreachable.
        for (bid, _) in &redirects {
            func.blocks[*bid].term = Terminator::Unreachable;
        }
    }
}

// =============================================================================
// Block splitting
// =============================================================================

/// Result of splitting a block at an instruction.
pub struct SplitResult {
    /// The new continuation block that holds instructions after the split point.
    pub cont_block: BlockId,
}

/// Split a block at the given instruction, removing that instruction.
///
/// After splitting:
/// - `block` retains instructions before `inst` and gets `Unreachable` terminator
///   (caller must set the real terminator)
/// - A new continuation block is created with:
///   - A block parameter for the removed instruction's result (if any),
///     with all uses of the result rewritten to this parameter
///   - The instructions that were after `inst`
///   - The original block's terminator
///
/// Returns the continuation block ID. If the removed instruction had a result,
/// the continuation block's first (and only) parameter is the replacement value.
pub fn split_block_at<I: Instr, E: Copy + Eq + Hash + Debug, T: Clone + Debug>(
    func: &mut Function<I, E, T>,
    inst: InstId,
) -> SplitResult {
    let block = func.insts[inst].parent;
    let inst_list = &func.blocks[block].insts;
    let pos = inst_list
        .iter()
        .position(|&id| id == inst)
        .expect("split_block_at: inst not found in its parent block");

    let after_insts: Vec<InstId> = inst_list[pos + 1..].to_vec();
    let original_term = func.blocks[block].term.clone();

    // Truncate the block to just before the split instruction.
    func.blocks[block].insts.truncate(pos);
    func.blocks[block].term = Terminator::Unreachable;

    // Create continuation block with the moved instructions and original terminator.
    let cont_block = func.create_block();
    for &inst_id in &after_insts {
        func.insts[inst_id].parent = cont_block;
    }
    func.blocks[cont_block].insts = after_insts;
    func.blocks[cont_block].term = original_term;

    // If the instruction produced a result, add a block param for it.
    // replace_all_uses now sees the continuation block's terminator too.
    if let Some(result_val) = func.insts[inst].result {
        let ty = func.values[result_val].ty.clone();
        let cont_param = func.add_block_param(cont_block, ty);
        func.replace_all_uses(result_val, cont_param);
        func.values.remove(result_val);
    }

    // Remove the instruction.
    func.insts.remove(inst);

    SplitResult { cont_block }
}

#[cfg(test)]
mod tests;
