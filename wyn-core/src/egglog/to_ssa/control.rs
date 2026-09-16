use super::{error, ControlHeader, FuncBody, OptimizeError, Terminator};
use crate::flow::BlockId;
use std::collections::{BTreeMap, BTreeSet};

/// Supply the existing backend with selection merges. Reachability alone is
/// insufficient inside a loop: both arms can reach each other on a later iteration.
/// The nearest common postdominator is where both arms must reconverge.
pub(super) fn annotate_selections(body: &mut FuncBody) -> Result<(), OptimizeError> {
    let all: BTreeSet<_> = body.inner.blocks.keys().collect();
    let successors: BTreeMap<BlockId, Vec<BlockId>> = body
        .inner
        .blocks
        .iter()
        .map(|(id, b)| {
            let next = match b.term {
                Terminator::Branch { target, .. } => vec![target],
                Terminator::CondBranch {
                    then_target,
                    else_target,
                    ..
                } => vec![then_target, else_target],
                _ => vec![],
            };
            (id, next)
        })
        .collect();
    let mut post: BTreeMap<_, _> = successors
        .iter()
        .map(|(&id, next)| {
            (
                id,
                if next.is_empty() { BTreeSet::from([id]) } else { all.clone() },
            )
        })
        .collect();
    loop {
        let mut changed = false;
        for (&id, next) in &successors {
            let Some(first) = next.first() else {
                continue;
            };
            let mut common = post[first].clone();
            for successor in &next[1..] {
                common.retain(|b| post[successor].contains(b));
            }
            common.insert(id);
            if common != post[&id] {
                post.insert(id, common);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    for (id, block) in body.inner.blocks.iter_mut() {
        if block.control_header.is_some() || !matches!(block.term, Terminator::CondBranch { .. }) {
            continue;
        }
        let strict: BTreeSet<_> = post[&id].iter().copied().filter(|&b| b != id).collect();
        let Some(merge) = strict.iter().copied().find(|candidate| {
            strict.iter().all(|other| other == candidate || !post[other].contains(candidate))
        }) else {
            return Err(error("TODO: selection with no common merge"));
        };
        block.control_header = Some(ControlHeader::Selection { merge });
    }
    Ok(())
}
