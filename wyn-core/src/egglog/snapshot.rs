//! Summarize the sidecar for fusion without exporting scalar syntax to egglog.

use super::data::{
    Array, AssociatedData, ExprId, ExprKind, LoopKind, OperationId, OperationKind, RegionId, ScremaForm,
    SoacBody,
};
use super::optimize::OptimizeError;
use crate::{types, LookupMap};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Role {
    Input,
    Capture,
    Neutral,
    Argument,
    Length,
}

impl Role {
    pub(super) fn egglog(self) -> &'static str {
        match self {
            Self::Input => "(Input)",
            Self::Capture => "(Capture)",
            Self::Neutral => "(Neutral)",
            Self::Argument => "(Argument)",
            Self::Length => "(Length)",
        }
    }
}

/// Stop at operation results. Nested regions are visited separately so dead
/// work inside a lambda or branch cannot keep an enclosing producer alive.
#[derive(Clone, Default)]
struct References {
    operations: BTreeSet<OperationId>,
    regions: BTreeSet<RegionId>,
}

impl References {
    fn extend(&mut self, other: &Self) {
        self.operations.extend(&other.operations);
        self.regions.extend(&other.regions);
    }
    fn dependencies(&self, external: &BTreeMap<RegionId, BTreeSet<OperationId>>) -> BTreeSet<OperationId> {
        let mut result = self.operations.clone();
        for region in &self.regions {
            result.extend(&external[region]);
        }
        result
    }
}

#[derive(Default)]
struct Operands {
    values: BTreeMap<Role, References>,
    regions: BTreeSet<RegionId>,
}

impl Operands {
    fn all(&self) -> References {
        let mut result = References {
            regions: self.regions.clone(),
            ..References::default()
        };
        for refs in self.values.values() {
            result.extend(refs);
        }
        result
    }
}

pub(super) struct Snapshot {
    pub live: BTreeSet<OperationId>,
    pub dependencies: BTreeSet<(OperationId, OperationId)>, // consumer, producer
    pub effects: BTreeSet<(OperationId, OperationId)>,      // before, after
    pub uses: BTreeSet<(OperationId, OperationId, Role)>,   // producer, consumer, role
    pub observed: BTreeSet<OperationId>,
    pub movable: BTreeSet<OperationId>,
    pub discardable: BTreeSet<OperationId>,
    pub safe_regions: BTreeSet<RegionId>,
}

pub(super) fn analyze(data: &AssociatedData) -> Snapshot {
    let mut visitor = Visitor {
        data,
        expressions: LookupMap::new(),
    };
    let members: BTreeSet<_> = data.regions.values().flat_map(|r| r.members.iter().copied()).collect();
    let operands: BTreeMap<_, _> =
        members.iter().map(|&id| (id, visitor.operation(&data.operations[id].kind))).collect();
    let results: BTreeMap<_, _> =
        data.regions.iter().map(|(&id, r)| (id, visitor.expressions(&r.results))).collect();

    // A body is movable only once every operation in it has been proved movable.
    // Unknown calls, control flow, and memory-sensitive operations remain barriers.
    let mut safe_regions = BTreeSet::new();
    let mut movable = BTreeSet::new();
    let mut discardable = BTreeSet::new();
    loop {
        let before = (safe_regions.len(), movable.len(), discardable.len());
        for (&id, r) in &data.regions {
            if r.members.is_subset(&movable) {
                safe_regions.insert(id);
            }
        }
        for &id in &members {
            match &data.operations[id].kind {
                kind if super::fusion::length_source(data, kind).is_some() => {
                    movable.insert(id);
                    discardable.insert(id);
                }
                OperationKind::Index { array, .. }
                    if types::is_copy(&data.types[data.expressions[*array].ty].ty)
                        || fresh_array(data, *array, &movable) =>
                {
                    movable.insert(id);
                    discardable.insert(id);
                }
                OperationKind::Screma { form, ownership, .. } if safe_form(form, &safe_regions) => {
                    discardable.insert(id);
                    if ownership.iter().all(|o| *o == types::SoacOwnership::Fresh) {
                        movable.insert(id);
                    }
                }
                OperationKind::Filter {
                    map, body, ownership, ..
                } if safe_body(body, &safe_regions) && safe_body(map, &safe_regions) => {
                    discardable.insert(id);
                    if *ownership == types::SoacOwnership::Fresh {
                        movable.insert(id);
                    }
                }
                _ => {}
            }
        }
        if before == (safe_regions.len(), movable.len(), discardable.len()) {
            break;
        }
    }

    // Summarize lexical captures through arbitrarily nested regions. These
    // conservative edges constrain motion; they are not liveness roots.
    let mut ancestors = BTreeSet::new();
    for (&id, region) in &data.regions {
        let mut parent = region.parent;
        while let Some(p) = parent {
            if !ancestors.insert((id, p)) {
                break;
            }
            parent = data.regions[p].parent;
        }
    }
    let mut external: BTreeMap<RegionId, BTreeSet<OperationId>> =
        data.regions.ids().map(|id| (id, BTreeSet::new())).collect();
    loop {
        let mut changed = false;
        for (&id, region) in &data.regions {
            let mut deps = results[&id].dependencies(&external);
            for op in &region.members {
                deps.extend(operands[op].all().dependencies(&external));
            }
            deps.retain(|op| {
                members.contains(op) && ancestors.contains(&(id, data.operations[*op].region))
            });
            let known = external.entry(id).or_default();
            let before = known.len();
            known.extend(deps);
            changed |= known.len() != before;
        }
        if !changed {
            break;
        }
    }

    // Work backward from each callable's outputs and required effects. The
    // printer later chooses callables from entries; unused arena records do not
    // activate nested regions. Effect-order edges never supply demand.
    let mut active: BTreeSet<_> = data.definitions.values().map(|def| def.body).collect();
    let mut live = BTreeSet::new();
    loop {
        let before = (active.len(), live.len());
        for id in active.clone() {
            live.extend(&results[&id].operations);
            active.extend(&results[&id].regions);
            live.extend(data.regions[id].members.difference(&discardable));
        }
        live.retain(|id| members.contains(id));
        for id in live.clone() {
            let refs = operands[&id].all();
            live.extend(refs.operations);
            active.extend(refs.regions);
        }
        if before == (active.len(), live.len()) {
            break;
        }
    }
    live.retain(|id| members.contains(id));
    let mut dependencies = BTreeSet::new();
    let mut uses = BTreeSet::new();
    for &consumer in &live {
        let args = &operands[&consumer];
        for (&role, refs) in &args.values {
            for producer in refs.dependencies(&external) {
                dependencies.insert((consumer, producer));
                uses.insert((producer, consumer, role));
            }
        }
        for child in &args.regions {
            for &producer in &external[child] {
                dependencies.insert((consumer, producer));
                uses.insert((producer, consumer, Role::Capture));
            }
        }
    }
    let mut observed = BTreeSet::new();
    for region in active {
        observed.extend(results[&region].dependencies(&external));
    }
    let mut effects = BTreeSet::new();
    for region in data.regions.values() {
        for &before in region.members.intersection(&live) {
            for &after in region.members.intersection(&live) {
                if data.operations[before].source_position < data.operations[after].source_position
                    && (!movable.contains(&before) || !movable.contains(&after))
                {
                    effects.insert((before, after));
                }
            }
        }
    }
    Snapshot {
        live,
        dependencies,
        effects,
        uses,
        observed,
        movable,
        discardable,
        safe_regions,
    }
}

// Fresh SOAC output storage is immutable until an ordered consumer writes it.
// Its imported array type may still be abstract, so Copy alone is insufficient.
fn fresh_array(data: &AssociatedData, value: super::ExprId, movable: &BTreeSet<OperationId>) -> bool {
    match data.expressions[value].kind {
        ExprKind::Project { tuple, .. } | ExprKind::Coerce(tuple) => fresh_array(data, tuple, movable),
        ExprKind::OperationResult(op) => {
            movable.contains(&op)
                && matches!(
                    data.operations[op].kind,
                    OperationKind::Screma { .. } | OperationKind::Filter { .. }
                )
        }
        _ => false,
    }
}

impl Snapshot {
    /// Schedule only the backward slice. Cross-region dependencies are supplied
    /// by the enclosing execution; branch and loop bodies keep their own work.
    pub(super) fn schedules(
        &self,
        data: &AssociatedData,
    ) -> Result<BTreeMap<RegionId, Vec<OperationId>>, OptimizeError> {
        let mut regions = BTreeMap::<RegionId, BTreeSet<OperationId>>::new();
        for &id in &self.live {
            regions.entry(data.operations[id].region).or_default().insert(id);
        }
        let edges: BTreeSet<_> =
            self.dependencies.iter().map(|&(c, p)| (p, c)).chain(self.effects.iter().copied()).collect();
        let mut result = BTreeMap::new();
        for (region, members) in regions {
            let mut indegree: BTreeMap<_, usize> = members.iter().map(|&op| (op, 0)).collect();
            let mut successors = BTreeMap::<OperationId, Vec<OperationId>>::new();
            for &(before, after) in &edges {
                if members.contains(&before) && members.contains(&after) {
                    if let Some(count) = indegree.get_mut(&after) {
                        *count += 1;
                    }
                    successors.entry(before).or_default().push(after);
                }
            }
            let mut ready: BTreeSet<_> =
                indegree.iter().filter_map(|(&op, &n)| (n == 0).then_some(op)).collect();
            let mut ordered = Vec::new();
            while let Some(op) = ready.pop_first() {
                ordered.push(op);
                for &next in successors.get(&op).into_iter().flatten() {
                    if let Some(count) = indegree.get_mut(&next) {
                        *count -= 1;
                        if *count == 0 {
                            ready.insert(next);
                        }
                    }
                }
            }
            if ordered.len() != members.len() {
                return Err(OptimizeError::Extraction(
                    "cycle in the selected execution graph".into(),
                ));
            }
            result.insert(region, ordered);
        }
        Ok(result)
    }
}

pub(super) fn safe_body(body: &SoacBody, regions: &BTreeSet<RegionId>) -> bool {
    match body {
        SoacBody::Apply { region, .. } => regions.contains(region),
        SoacBody::Compose { first, then } => safe_body(first, regions) && safe_body(then, regions),
        SoacBody::Parallel { left, right } => safe_body(left, regions) && safe_body(right, regions),
        SoacBody::Identity(_) | SoacBody::Route { .. } => true,
    }
}

fn safe_form(form: &ScremaForm, regions: &BTreeSet<RegionId>) -> bool {
    safe_body(&form.pre, regions)
        && safe_body(&form.post, regions)
        && form.scans.iter().all(|scan| safe_body(&scan.operator, regions))
        && form.reductions.iter().all(|reduction| safe_body(&reduction.operator, regions))
}

struct Visitor<'a> {
    data: &'a AssociatedData,
    expressions: LookupMap<ExprId, References>,
}

impl Visitor<'_> {
    fn expressions(&mut self, ids: &[ExprId]) -> References {
        let mut refs = References::default();
        for &id in ids {
            refs.extend(&self.expression(id));
        }
        refs
    }
    fn expression(&mut self, id: ExprId) -> References {
        if let Some(refs) = self.expressions.get(&id) {
            return refs.clone();
        }
        let refs = match &self.data.expressions[id].kind {
            ExprKind::OperationResult(op) => References {
                operations: BTreeSet::from([*op]),
                ..References::default()
            },
            ExprKind::Lambda(region) => References {
                regions: BTreeSet::from([*region]),
                ..References::default()
            },
            ExprKind::PureApp { function, args } => {
                let mut refs = self.expression(*function);
                refs.extend(&self.expressions(args));
                refs
            }
            ExprKind::Closure { captures, .. } => self.expressions(captures),
            ExprKind::Tuple(values) | ExprKind::Vector(values) => self.expressions(values),
            ExprKind::Coerce(inner) | ExprKind::Project { tuple: inner, .. } => self.expression(*inner),
            ExprKind::If {
                condition,
                then_value,
                else_value,
            } => self.expressions(&[*condition, *then_value, *else_value]),
            ExprKind::Array(array) => self.array(array),
            ExprKind::Global(_)
            | ExprKind::Parameter(_)
            | ExprKind::Builtin(_)
            | ExprKind::BinOp(_)
            | ExprKind::UnOp(_)
            | ExprKind::Int(_)
            | ExprKind::FloatBits(_)
            | ExprKind::Bool(_)
            | ExprKind::Unit
            | ExprKind::Extern(_) => References::default(),
        };
        self.expressions.insert(id, refs.clone());
        refs
    }
    fn array(&mut self, array: &Array) -> References {
        match array {
            Array::Value(value) => self.expression(*value),
            Array::Literal(values) => self.expressions(values),
            Array::Zip(arrays) => {
                let mut refs = References::default();
                for array in arrays {
                    refs.extend(&self.array(array));
                }
                refs
            }
            Array::Range { start, len, step } => {
                let mut refs = self.expressions(&[*start, *len]);
                if let Some(step) = step {
                    refs.extend(&self.expression(*step));
                }
                refs
            }
        }
    }
    fn values(&mut self, out: &mut Operands, role: Role, ids: &[ExprId]) {
        out.values.entry(role).or_default().extend(&self.expressions(ids));
    }
    fn inputs(&mut self, out: &mut Operands, arrays: &[Array]) {
        for array in arrays {
            out.values.entry(Role::Input).or_default().extend(&self.array(array));
        }
    }
    fn body(&mut self, out: &mut Operands, body: &SoacBody) {
        match body {
            SoacBody::Apply { region, captures, .. } => {
                out.regions.insert(*region);
                self.values(out, Role::Capture, captures);
            }
            SoacBody::Compose { first, then } => {
                self.body(out, first);
                self.body(out, then);
            }
            SoacBody::Parallel { left, right } => {
                self.body(out, left);
                self.body(out, right);
            }
            SoacBody::Identity(_) | SoacBody::Route { .. } => {}
        }
    }
    fn operation(&mut self, kind: &OperationKind) -> Operands {
        let mut out = Operands::default();
        match kind {
            OperationKind::Call { function, args } => {
                self.values(&mut out, Role::Argument, &[*function]);
                let role = if super::fusion::length_source(self.data, kind).is_some() {
                    Role::Length
                } else {
                    Role::Argument
                };
                self.values(&mut out, role, args);
            }
            OperationKind::EvalGlobal(_) => {}
            OperationKind::If {
                condition,
                then_region,
                else_region,
            } => {
                self.values(&mut out, Role::Argument, &[*condition]);
                out.regions.extend([*then_region, *else_region]);
            }
            OperationKind::Loop {
                init,
                header,
                kind,
                body,
            } => {
                self.values(&mut out, Role::Argument, &[*init]);
                if let LoopKind::For(value) | LoopKind::ForRange(value) = kind {
                    self.values(&mut out, Role::Argument, &[*value]);
                }
                out.regions.extend([*header, *body]);
            }
            OperationKind::Index { array, index } => {
                self.values(&mut out, Role::Argument, &[*array, *index])
            }
            OperationKind::Screma { form, inputs, .. } => {
                self.inputs(&mut out, inputs);
                self.body(&mut out, &form.pre);
                self.body(&mut out, &form.post);
                for scan in &form.scans {
                    self.body(&mut out, &scan.operator);
                    self.values(&mut out, Role::Neutral, &scan.neutral);
                }
                for reduction in &form.reductions {
                    self.body(&mut out, &reduction.operator);
                    self.values(&mut out, Role::Neutral, &reduction.neutral);
                }
            }
            OperationKind::Filter {
                map, body, inputs, ..
            } => {
                self.body(&mut out, map);
                self.body(&mut out, body);
                self.inputs(&mut out, inputs);
            }
            OperationKind::Scatter {
                destination,
                body,
                inputs,
            }
            | OperationKind::BucketScatter {
                destination,
                body,
                inputs,
                ..
            } => {
                self.values(&mut out, Role::Argument, &[destination.value]);
                self.body(&mut out, body);
                self.inputs(&mut out, inputs);
            }
            OperationKind::ReduceByIndex {
                destination,
                map,
                body,
                neutral,
                inputs,
            } => {
                self.values(&mut out, Role::Argument, &[destination.value]);
                self.values(&mut out, Role::Neutral, &[*neutral]);
                self.body(&mut out, map);
                self.body(&mut out, body);
                self.inputs(&mut out, inputs);
            }
        }
        out
    }
}
