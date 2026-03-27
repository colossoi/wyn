use std::collections::{HashMap, HashSet};

use std::fmt::Debug;

use crate::{BlockId, Function, Instr};

pub(crate) struct Dominators {
    entry: BlockId,
    doms: HashMap<BlockId, HashSet<BlockId>>,
    preds: HashMap<BlockId, Vec<BlockId>>,
}

impl Dominators {
    pub(crate) fn compute<I, E, T: Clone + Debug>(func: &Function<I, E, T>) -> Self {
        let blocks: Vec<BlockId> = func.blocks.keys().collect();
        let all: HashSet<BlockId> = blocks.iter().copied().collect();
        let preds = func.predecessors();

        let mut doms: HashMap<BlockId, HashSet<BlockId>> = HashMap::new();
        for &b in &blocks {
            if b == func.entry {
                doms.insert(b, [b].into_iter().collect());
            } else {
                doms.insert(b, all.clone());
            }
        }

        loop {
            let mut changed = false;

            for &b in &blocks {
                if b == func.entry {
                    continue;
                }

                let pred_list = preds.get(&b).cloned().unwrap_or_default();
                let new_set: HashSet<BlockId> = if pred_list.is_empty() {
                    [b].into_iter().collect()
                } else {
                    let mut it = pred_list.into_iter();
                    let first = it.next().unwrap();
                    let mut acc = doms[&first].clone();
                    for p in it {
                        acc = acc.intersection(&doms[&p]).copied().collect::<HashSet<_>>();
                    }
                    acc.insert(b);
                    acc
                };

                if new_set != doms[&b] {
                    doms.insert(b, new_set);
                    changed = true;
                }
            }

            if !changed {
                break;
            }
        }

        Self {
            entry: func.entry,
            doms,
            preds,
        }
    }

    pub(crate) fn nearest_common_dominator_many(
        &self,
        mut blocks: impl Iterator<Item = BlockId>,
    ) -> Option<BlockId> {
        let first = blocks.next()?;
        Some(blocks.fold(first, |acc, b| self.nearest_common_dominator(acc, b)))
    }

    pub(crate) fn preds(&self, block: BlockId) -> &[BlockId] {
        self.preds.get(&block).map(Vec::as_slice).unwrap_or(&[])
    }

    fn nearest_common_dominator(&self, a: BlockId, b: BlockId) -> BlockId {
        if a == b {
            return a;
        }
        let da = &self.doms[&a];
        let db = &self.doms[&b];
        let common: Vec<BlockId> = da.intersection(db).copied().collect();

        common.into_iter().max_by_key(|&cand| self.doms[&cand].len()).unwrap_or(self.entry)
    }
}

pub(crate) fn block_top_insert_index_after_operands<I: Instr, E, T: Clone + Debug>(
    func: &Function<I, E, T>,
    target_block: BlockId,
    data: &I,
) -> usize {
    let order = func.block_order_index_map();
    let mut max_idx_plus_one = 0usize;

    data.for_each_operand(|v| {
        if let Some(inst) = func.inst_of_value(v) {
            if func.insts[inst].parent == target_block {
                let idx = order[&inst] + 1;
                if idx > max_idx_plus_one {
                    max_idx_plus_one = idx;
                }
            }
        }
    });

    max_idx_plus_one
}
