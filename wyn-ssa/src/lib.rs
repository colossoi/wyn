mod dominators;

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
    Param {
        block: BlockId,
        index: usize,
    },
    Inst {
        inst: InstId,
    },
}

#[derive(Clone, Debug)]
pub enum Terminator {
    Return(Option<ValueId>),
    Jump {
        target: BlockId,
        args: Vec<ValueId>,
    },
    Branch {
        cond: ValueId,
        then_block: BlockId,
        then_args: Vec<ValueId>,
        else_block: BlockId,
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
            Terminator::Jump { target, .. } => smallvec::smallvec![*target],
            Terminator::Branch {
                then_block,
                else_block,
                ..
            } => smallvec::smallvec![*then_block, *else_block],
        }
    }

    pub fn for_each_value(&self, mut f: impl FnMut(ValueId)) {
        match self {
            Terminator::Return(v) => {
                if let Some(v) = v {
                    f(*v);
                }
            }
            Terminator::Jump { args, .. } => {
                for &v in args {
                    f(v);
                }
            }
            Terminator::Branch {
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
            Terminator::Jump { target, args } => Terminator::Jump {
                target: *target,
                args: args.iter().copied().map(&mut f).collect(),
            },
            Terminator::Branch {
                cond,
                then_block,
                then_args,
                else_block,
                else_args,
            } => Terminator::Branch {
                cond: f(*cond),
                then_block: *then_block,
                then_args: then_args.iter().copied().map(&mut f).collect(),
                else_block: *else_block,
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
        let block = self.blocks.insert(BasicBlock {
            params: Vec::new(),
            insts: Vec::new(),
            term: Terminator::Unreachable,
        });
        // First user block: entry jumps to it automatically.
        if let Terminator::Unreachable = self.blocks[self.entry].term {
            self.blocks[self.entry].term = Terminator::Jump {
                target: block,
                args: vec![],
            };
        }
        block
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
            ValueDef::Inst { inst } => self.insts[inst].parent,
        }
    }

    pub fn inst_of_value(&self, v: ValueId) -> Option<InstId> {
        match self.values[v].def {
            ValueDef::Param { .. } => None,
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
            Terminator::Jump { target, mut args } if target == block => {
                args.remove(param_index);
                Terminator::Jump { target, args }
            }
            Terminator::Branch {
                cond,
                then_block,
                mut then_args,
                else_block,
                mut else_args,
            } => {
                if then_block == block {
                    then_args.remove(param_index);
                }
                if else_block == block {
                    else_args.remove(param_index);
                }
                Terminator::Branch {
                    cond,
                    then_block,
                    then_args,
                    else_block,
                    else_args,
                }
            }
            other => other,
        };
    }

    Ok(())
}

pub fn lift_and_merge<I: ValueLike, E: Copy + Eq + Hash + Debug, T: Clone + Debug>(
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

    let mut classes: Vec<Vec<InstId>> = Vec::new();

    'outer: for inst in candidates {
        if !func.insts.contains_key(inst) {
            continue;
        }
        for class in &mut classes {
            let rep = class[0];
            if !func.insts.contains_key(rep) {
                continue;
            }
            if func.insts[inst].data.equivalent_to(&func.insts[rep].data) {
                class.push(inst);
                continue 'outer;
            }
        }
        classes.push(vec![inst]);
    }

    for class in classes {
        let live: Vec<InstId> = class.into_iter().filter(|id| func.insts.contains_key(*id)).collect();

        if live.len() < 2 {
            continue;
        }

        let target_block = dom
            .nearest_common_dominator_many(live.iter().map(|&id| func.insts[id].parent))
            .unwrap_or(func.entry);

        let rep = live[0];
        let rep_data = func.insts[rep].data.clone();
        let Some(rep_result) = func.insts[rep].result else {
            continue;
        };
        let rep_ty = func.values[rep_result].ty.clone();

        let insert_index = block_top_insert_index_after_operands(func, target_block, &rep_data);

        let needs_new_canonical = {
            let block_insts = &func.blocks[target_block].insts;
            let existing_at_spot = block_insts.get(insert_index).copied();
            match existing_at_spot {
                Some(inst) => !(func.insts[inst].data.equivalent_to(&rep_data)),
                None => true,
            }
        };

        let canonical_value = if needs_new_canonical {
            func.insert_inst_at_index(target_block, insert_index, rep_data, rep_ty, None)
        } else {
            let Some(v) = func.insts[func.blocks[target_block].insts[insert_index]].result else {
                continue;
            };
            v
        };

        for inst in live {
            if !func.insts.contains_key(inst) {
                continue;
            }
            let Some(old_value) = func.insts[inst].result else {
                continue;
            };
            if old_value != canonical_value {
                func.replace_all_uses(old_value, canonical_value);
                func.remove_inst(inst);
            }
        }
    }
}

#[cfg(test)]
mod tests;
