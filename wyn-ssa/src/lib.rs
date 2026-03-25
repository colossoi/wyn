mod dominators;

use slotmap::{SlotMap, new_key_type};
use smallvec::SmallVec;
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

use dominators::{block_top_insert_index_after_operands, Dominators};

new_key_type! {
    pub struct BlockId;
    pub struct InstId;
    pub struct ValueId;
}

#[derive(Clone, Debug)]
pub struct Function<I, E> {
    pub entry: BlockId,
    pub blocks: SlotMap<BlockId, BasicBlock>,
    pub insts: SlotMap<InstId, InstNode<I, E>>,
    pub values: SlotMap<ValueId, ValueDef>,
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
    pub result: ValueId,
    pub parent: BlockId,
    pub effects: Option<(E, E)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueDef {
    Param { block: BlockId, index: usize },
    Inst { inst: InstId },
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
    DoubleConsume { effect: E, inst: InstId },
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

impl<I, E> Function<I, E> {
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

    pub fn add_block_param(&mut self, block: BlockId) -> ValueId {
        let index = self.blocks[block].params.len();
        let value = self.values.insert(ValueDef::Param { block, index });
        self.blocks[block].params.push(value);
        value
    }

    pub fn value_def(&self, v: ValueId) -> ValueDef {
        self.values[v]
    }

    pub fn block_of_value(&self, v: ValueId) -> BlockId {
        match self.values[v] {
            ValueDef::Param { block, .. } => block,
            ValueDef::Inst { inst } => self.insts[inst].parent,
        }
    }

    pub fn inst_of_value(&self, v: ValueId) -> Option<InstId> {
        match self.values[v] {
            ValueDef::Param { .. } => None,
            ValueDef::Inst { inst } => Some(inst),
        }
    }

    pub fn append_inst(
        &mut self,
        block: BlockId,
        data: I,
        effects: Option<(E, E)>,
    ) -> ValueId {
        let value = self.values.insert(ValueDef::Param { block, index: 0 });
        let inst = self.insts.insert(InstNode {
            data,
            result: value,
            parent: block,
            effects,
        });
        self.values[value] = ValueDef::Inst { inst };
        self.blocks[block].insts.push(inst);
        value
    }

    pub fn insert_inst_at_index(
        &mut self,
        block: BlockId,
        index: usize,
        data: I,
        effects: Option<(E, E)>,
    ) -> ValueId {
        let value = self.values.insert(ValueDef::Param { block, index: 0 });
        let inst = self.insts.insert(InstNode {
            data,
            result: value,
            parent: block,
            effects,
        });
        self.values[value] = ValueDef::Inst { inst };
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

impl<I, E: Copy + Eq + Hash + Debug> Function<I, E> {
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
        self.values.remove(node.result);

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
                    errors.push(EffectVerifyError::DoubleConsume { effect: ein, inst: id });
                } else {
                    consumed.insert(ein, id);
                }
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

impl<I: Instr, E: Copy + Eq + Hash + Debug> Function<I, E> {
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

pub fn inline_entry_param<I: ValueLike, E: Copy + Eq + Hash + Debug>(
    func: &mut Function<I, E>,
    param_index: usize,
    replacement: I,
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
    let new_value = func.insert_inst_at_index(entry, insert_index, replacement, None);
    func.replace_all_uses(old_param, new_value);

    func.blocks[entry].params.remove(param_index);
    func.values.remove(old_param);

    for (i, &param) in func.blocks[entry].params.iter().enumerate() {
        func.values[param] = ValueDef::Param {
            block: entry,
            index: i,
        };
    }

    Ok(())
}

pub fn inline_block_param<I: ValueLike, E: Copy + Eq + Hash + Debug>(
    func: &mut Function<I, E>,
    block: BlockId,
    param_index: usize,
    replacement: I,
) -> Result<(), InlineArgError> {
    if block == func.entry {
        return inline_entry_param(func, param_index, replacement);
    }

    if !replacement.is_closed() {
        return Err(InlineArgError::ReplacementNotClosed);
    }

    if param_index >= func.blocks[block].params.len() {
        return Err(InlineArgError::BadParamIndex);
    }

    let old_param = func.blocks[block].params[param_index];

    let insert_index = block_top_insert_index_after_operands(func, block, &replacement);
    let new_value = func.insert_inst_at_index(block, insert_index, replacement, None);
    func.replace_all_uses(old_param, new_value);

    func.blocks[block].params.remove(param_index);
    func.values.remove(old_param);

    for (i, &param) in func.blocks[block].params.iter().enumerate() {
        func.values[param] = ValueDef::Param { block, index: i };
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

pub fn lift_and_merge<I: ValueLike, E: Copy + Eq + Hash + Debug>(
    func: &mut Function<I, E>,
) {
    let dom = Dominators::compute(func);

    let candidates: Vec<InstId> = func
        .insts
        .iter()
        .filter_map(|(id, node)| {
            if node.effects.is_none() && node.data.is_hoistable() {
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
        let live: Vec<InstId> = class
            .into_iter()
            .filter(|id| func.insts.contains_key(*id))
            .collect();

        if live.len() < 2 {
            continue;
        }

        let target_block = dom
            .nearest_common_dominator_many(live.iter().map(|&id| func.insts[id].parent))
            .unwrap_or(func.entry);

        let rep_data = func.insts[live[0]].data.clone();

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
            func.insert_inst_at_index(target_block, insert_index, rep_data, None)
        } else {
            func.insts[func.blocks[target_block].insts[insert_index]].result
        };

        for inst in live {
            if !func.insts.contains_key(inst) {
                continue;
            }
            let old_value = func.insts[inst].result;
            if old_value != canonical_value {
                func.replace_all_uses(old_value, canonical_value);
                func.remove_inst(inst);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    struct TestEffect(u32);

    #[derive(Clone, Debug, PartialEq, Eq)]
    enum TestInst {
        Int(i64),
        Add(ValueId, ValueId),
        Pair(ValueId, ValueId),
        SideEffect(ValueId),
    }

    impl Instr for TestInst {
        fn for_each_operand(&self, mut f: impl FnMut(ValueId)) {
            match self {
                TestInst::Int(_) => {}
                TestInst::Add(a, b) | TestInst::Pair(a, b) => {
                    f(*a);
                    f(*b);
                }
                TestInst::SideEffect(v) => f(*v),
            }
        }

        fn map_operands(&self, mut f: impl FnMut(ValueId) -> ValueId) -> Self {
            match self {
                TestInst::Int(n) => TestInst::Int(*n),
                TestInst::Add(a, b) => TestInst::Add(f(*a), f(*b)),
                TestInst::Pair(a, b) => TestInst::Pair(f(*a), f(*b)),
                TestInst::SideEffect(v) => TestInst::SideEffect(f(*v)),
            }
        }
    }

    impl ValueLike for TestInst {
        fn is_hoistable(&self) -> bool {
            true
        }

        fn is_closed(&self) -> bool {
            matches!(self, TestInst::Int(_))
        }

        fn equivalent_to(&self, other: &Self) -> bool {
            self == other
        }
    }

    type TestFunc = Function<TestInst, TestEffect>;

    #[test]
    fn inline_entry_param_replaces_uses_and_removes_param() {
        let mut f = TestFunc::new();
        let p0 = f.add_block_param(f.entry);

        let pair = f.append_inst(f.entry, TestInst::Pair(p0, p0), None);
        f.blocks[f.entry].term = Terminator::Return(Some(pair));

        inline_entry_param(&mut f, 0, TestInst::Int(42)).unwrap();
        assert!(f.blocks[f.entry].params.is_empty());

        let mut saw_pair = false;
        for (_id, node) in &f.insts {
            if let TestInst::Pair(a, b) = node.data {
                saw_pair = true;
                let ia = f.inst_of_value(a).unwrap();
                let ib = f.inst_of_value(b).unwrap();
                assert_eq!(f.insts[ia].data, TestInst::Int(42));
                assert_eq!(f.insts[ib].data, TestInst::Int(42));
            }
        }
        assert!(saw_pair);
    }

    #[test]
    fn lift_and_merge_merges_duplicate_closed_values() {
        let mut f = TestFunc::new();

        let c1 = f.append_inst(f.entry, TestInst::Int(7), None);
        let c2 = f.append_inst(f.entry, TestInst::Int(7), None);
        let pair = f.append_inst(f.entry, TestInst::Pair(c1, c2), None);
        f.blocks[f.entry].term = Terminator::Return(Some(pair));

        let before = f.insts.len();
        lift_and_merge(&mut f);
        let after = f.insts.len();

        assert!(after < before);

        let mut pair_operands = None;
        for (_id, node) in &f.insts {
            if let TestInst::Pair(a, b) = node.data {
                pair_operands = Some((a, b));
            }
        }

        let (a, b) = pair_operands.unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn lift_and_merge_merges_duplicate_non_closed_values() {
        let mut f = TestFunc::new();
        let p0 = f.add_block_param(f.entry);
        let p1 = f.add_block_param(f.entry);

        let x1 = f.append_inst(f.entry, TestInst::Add(p0, p1), None);
        let x2 = f.append_inst(f.entry, TestInst::Add(p0, p1), None);
        let pair = f.append_inst(f.entry, TestInst::Pair(x1, x2), None);
        f.blocks[f.entry].term = Terminator::Return(Some(pair));

        let before = f.insts.len();
        lift_and_merge(&mut f);
        assert!(f.insts.len() < before);

        let mut pair_operands = None;
        for (_id, node) in &f.insts {
            if let TestInst::Pair(a, b) = node.data {
                pair_operands = Some((a, b));
            }
        }

        let (a, b) = pair_operands.unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn inline_non_entry_block_param_updates_predecessor_edge_args() {
        let mut f = TestFunc::new();
        let cond = f.add_block_param(f.entry);

        let join = f.create_block();
        let jp = f.add_block_param(join);

        let t = f.create_block();
        let e = f.create_block();

        let c1 = f.append_inst(t, TestInst::Int(1), None);
        f.blocks[t].term = Terminator::Jump {
            target: join,
            args: vec![c1],
        };

        let c2 = f.append_inst(e, TestInst::Int(1), None);
        f.blocks[e].term = Terminator::Jump {
            target: join,
            args: vec![c2],
        };

        f.blocks[f.entry].term = Terminator::Branch {
            cond,
            then_block: t,
            then_args: vec![],
            else_block: e,
            else_args: vec![],
        };

        let out = f.append_inst(join, TestInst::Pair(jp, jp), None);
        f.blocks[join].term = Terminator::Return(Some(out));

        inline_block_param(&mut f, join, 0, TestInst::Int(9)).unwrap();

        match &f.blocks[t].term {
            Terminator::Jump { target, args } => {
                assert_eq!(*target, join);
                assert!(args.is_empty());
            }
            _ => panic!("expected jump"),
        }

        match &f.blocks[e].term {
            Terminator::Jump { target, args } => {
                assert_eq!(*target, join);
                assert!(args.is_empty());
            }
            _ => panic!("expected jump"),
        }
    }

    #[test]
    fn effectful_inst_creation_and_query() {
        let mut f = TestFunc::new();
        let p0 = f.add_block_param(f.entry);

        let pure_val = f.append_inst(f.entry, TestInst::Int(1), None);
        let eff_val = f.append_inst(
            f.entry,
            TestInst::SideEffect(p0),
            Some((TestEffect(0), TestEffect(1))),
        );

        assert!(!f.is_effectful(f.inst_of_value(pure_val).unwrap()));
        assert!(f.is_effectful(f.inst_of_value(eff_val).unwrap()));

        let eff_inst = f.inst_of_value(eff_val).unwrap();
        assert_eq!(f.insts[eff_inst].effects, Some((TestEffect(0), TestEffect(1))));
    }

    #[test]
    fn remove_effectful_inst_splices_chain() {
        let mut f = TestFunc::new();
        let p0 = f.add_block_param(f.entry);

        // Chain: e0 -> inst_a -> e1 -> inst_b -> e2
        f.append_inst(f.entry, TestInst::SideEffect(p0), Some((TestEffect(0), TestEffect(1))));
        let b = f.append_inst(f.entry, TestInst::SideEffect(p0), Some((TestEffect(1), TestEffect(2))));

        // Remove inst_a (consumes e0, produces e1)
        // inst_b should now consume e0 instead of e1
        let a_inst = f.blocks[f.entry].insts[0];
        f.remove_inst(a_inst);

        let b_inst = f.inst_of_value(b).unwrap();
        assert_eq!(f.insts[b_inst].effects, Some((TestEffect(0), TestEffect(2))));
    }

    #[test]
    fn replace_effect_works() {
        let mut f = TestFunc::new();
        let p0 = f.add_block_param(f.entry);

        let v = f.append_inst(f.entry, TestInst::SideEffect(p0), Some((TestEffect(5), TestEffect(6))));

        assert!(f.replace_effect(TestEffect(5), TestEffect(99)));

        let inst = f.inst_of_value(v).unwrap();
        assert_eq!(f.insts[inst].effects, Some((TestEffect(99), TestEffect(6))));
    }

    #[test]
    fn verify_effects_ok_for_valid_chain() {
        let mut f = TestFunc::new();
        let p0 = f.add_block_param(f.entry);

        f.append_inst(f.entry, TestInst::SideEffect(p0), Some((TestEffect(0), TestEffect(1))));
        f.append_inst(f.entry, TestInst::SideEffect(p0), Some((TestEffect(1), TestEffect(2))));
        f.append_inst(f.entry, TestInst::Int(42), None);

        assert!(f.verify_effects().is_ok());
    }

    #[test]
    fn verify_effects_catches_double_consume() {
        let mut f = TestFunc::new();
        let p0 = f.add_block_param(f.entry);

        // Both instructions consume TestEffect(0) — invalid
        f.append_inst(f.entry, TestInst::SideEffect(p0), Some((TestEffect(0), TestEffect(1))));
        f.append_inst(f.entry, TestInst::SideEffect(p0), Some((TestEffect(0), TestEffect(2))));

        let errs = f.verify_effects().unwrap_err();
        assert!(errs.iter().any(|e| matches!(e, EffectVerifyError::DoubleConsume { effect: TestEffect(0), .. })));
    }

    #[test]
    fn lift_and_merge_skips_effectful_instructions() {
        let mut f = TestFunc::new();
        let p0 = f.add_block_param(f.entry);

        // Two identical effectful instructions — should NOT be merged
        let a = f.append_inst(f.entry, TestInst::SideEffect(p0), Some((TestEffect(0), TestEffect(1))));
        let b = f.append_inst(f.entry, TestInst::SideEffect(p0), Some((TestEffect(1), TestEffect(2))));
        let pair = f.append_inst(f.entry, TestInst::Pair(a, b), None);
        f.blocks[f.entry].term = Terminator::Return(Some(pair));

        lift_and_merge(&mut f);

        // Both instructions should still exist
        assert!(f.inst_of_value(a).is_some());
        assert!(f.inst_of_value(b).is_some());
    }
}
