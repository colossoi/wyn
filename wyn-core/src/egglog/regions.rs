//! Lexical containment indexed once, without storing every ancestor pair.
use crate::egglog::data::{Ir, RegionId};
use crate::LookupMap;
use std::collections::BTreeMap;
use wyn_graph::{forest_intervals, DfsInterval};

pub(super) struct Regions {
    pub children: BTreeMap<RegionId, Vec<RegionId>>,
    intervals: LookupMap<RegionId, DfsInterval>,
}

impl Regions {
    pub fn new(data: &Ir) -> Self {
        let mut children: BTreeMap<_, Vec<_>> = BTreeMap::new();
        let mut roots = vec![];
        for (&id, r) in &data.regions {
            if let Some(parent) = r.parent {
                children.entry(parent).or_default().push(id);
            } else {
                roots.push(id);
            }
        }
        let intervals = forest_intervals(roots, |r, out| {
            out.extend(children.get(&r).into_iter().flatten().copied());
        });
        Self { children, intervals }
    }

    pub fn contains(&self, parent: RegionId, child: RegionId) -> bool {
        match (self.intervals.get(&parent), self.intervals.get(&child)) {
            (Some(parent), Some(child)) => parent.contains(child.start),
            _ => false,
        }
    }
}
