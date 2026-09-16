//! Binding barriers on the structured control tree, before block lowering.
//! DFS intervals encode dominance, not execution counts or operation-ID order.
use super::*;

pub(super) use wyn_graph::DfsInterval as Interval;

pub(super) struct Bounds {
    pub regions: BTreeMap<RegionId, Interval>,
    pub operations: BTreeMap<OperationId, (usize, Interval)>,
}

#[derive(Default)]
struct Points {
    children: Vec<Vec<usize>>,
    regions: BTreeMap<RegionId, usize>,
    operations: BTreeMap<OperationId, (usize, usize)>,
}

impl Points {
    fn child(&mut self, parent: usize) -> usize {
        let point = self.children.len();
        self.children.push(vec![]);
        self.children[parent].push(point);
        point
    }

    fn region(
        &mut self,
        data: &AssociatedData,
        id: RegionId,
        parent: usize,
        schedules: &BTreeMap<RegionId, Vec<OperationId>>,
    ) -> Result<usize, OptimizeError> {
        if self.regions.contains_key(&id) {
            return Err(error("structured region has multiple control parents"));
        }
        let mut point = self.child(parent);
        self.regions.insert(id, point);
        for &op in schedules.get(&id).into_iter().flatten() {
            match data.operations[op].kind {
                OperationKind::If {
                    then_region,
                    else_region,
                    ..
                } => {
                    self.region(data, then_region, point, schedules)?;
                    self.region(data, else_region, point, schedules)?;
                }
                OperationKind::Loop { header, body, .. } => {
                    // Header bindings dominate the body, but neither header nor
                    // body bindings become available before/after the whole loop.
                    let header_end = self.region(data, header, point, schedules)?;
                    self.region(data, body, header_end, schedules)?;
                }
                _ => {}
            }
            // A result dominates the rest of this sequence. Branch-local results
            // are siblings of this continuation, so they cannot escape the branch.
            let after = self.child(point);
            self.operations.insert(op, (point, after));
            point = after;
        }
        Ok(point)
    }
}

pub(super) fn analyze(
    data: &AssociatedData,
    schedules: &BTreeMap<RegionId, Vec<OperationId>>,
) -> Result<Bounds, OptimizeError> {
    let mut children = BTreeSet::new();
    for &op in schedules.values().flatten() {
        children.extend(data.operations[op].kind.structured_regions());
    }
    let mut points = Points {
        children: vec![vec![]],
        ..Points::default()
    };
    // Each callable is a separate root. Lexical parent metadata alone does not
    // make one invocation's parameters available in another callable.
    for id in data.regions.ids().filter(|id| !children.contains(id)) {
        points.region(data, id, 0, schedules)?;
    }
    if data.regions.ids().any(|id| !points.regions.contains_key(&id)) {
        return Err(error("cyclic structured control"));
    }
    let bounds = wyn_graph::forest_intervals([0], |p, out| out.extend(&points.children[p]));
    Ok(Bounds {
        regions: points.regions.into_iter().map(|(r, p)| (r, bounds[&p])).collect(),
        operations: points
            .operations
            .into_iter()
            .map(|(op, (before, after))| (op, (bounds[&before].start, bounds[&after])))
            .collect(),
    })
}
